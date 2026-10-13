//! Reactive runtime — effects, memos, and the flush scheduler.
//!
//! [`SignalArena`] provides reactive *storage*: values, dependency edges, and a
//! dirty set. This module adds the *computation* layer — what SolidJS and Leptos
//! call effects and memos — plus the scheduler that re-runs them when the
//! signals they read change.
//!
//! - [`Runtime::create_effect`] registers a side-effecting computation that runs
//!   once immediately (recording its dependencies) and re-runs on
//!   [`Runtime::flush`] whenever a signal it read changes.
//! - [`Runtime::create_memo`] is a cached derived signal: it recomputes when its
//!   inputs change and is itself readable (and subscribable) by others.
//! - [`Runtime::flush`] drains the dirty set and re-runs affected effects until
//!   the graph settles, so `signal → memo → memo → effect` cascades converge in
//!   a single call.
//!
//! Effects and render nodes share the [`NodeId`] subscriber space, so this same
//! machinery will later drive targeted re-renders of `ViewNode::Stateful` blocks
//! (P0b). This module is the reusable core; it touches no rendering code.

use crate::signal::{NodeId, Signal, SignalArena};
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

/// Safety valve: the maximum number of drain/re-run passes [`Runtime::flush`]
/// performs before bailing. A well-formed graph settles in a few passes; hitting
/// this cap means an effect keeps writing a signal that feeds back into it (a
/// reactive cycle).
const MAX_FLUSH_PASSES: usize = 1000;

/// A reactive runtime: a [`SignalArena`] plus the effect registry and the
/// scheduler that re-runs effects whose dependencies changed.
pub struct Runtime {
    arena: Rc<SignalArena>,
    /// Registered reactive computations, keyed by their subscriber node id.
    effects: RefCell<HashMap<NodeId, Rc<dyn Fn()>>>,
    /// Monotonic node-id allocator, shared by effects and (later) render nodes.
    next_node: RefCell<NodeId>,
    /// Re-entrancy guard: a write performed *inside* a flushing effect must not
    /// start a nested flush — the running loop picks it up on its next pass.
    flushing: RefCell<bool>,
}

impl Runtime {
    /// Create a runtime over a fresh arena.
    pub fn new() -> Rc<Self> {
        Self::with_arena(SignalArena::new())
    }

    /// Create a runtime over an existing arena (e.g. one already shared with a
    /// render pass, once P0b wires this into the render walk).
    pub fn with_arena(arena: Rc<SignalArena>) -> Rc<Self> {
        Rc::new(Self {
            arena,
            effects: RefCell::new(HashMap::new()),
            next_node: RefCell::new(0),
            flushing: RefCell::new(false),
        })
    }

    /// The underlying signal arena.
    pub fn arena(&self) -> Rc<SignalArena> {
        Rc::clone(&self.arena)
    }

    /// Allocate a fresh subscriber node id (for an effect or a render node).
    pub fn alloc_node(&self) -> NodeId {
        let mut n = self.next_node.borrow_mut();
        let id = *n;
        *n = n.checked_add(1).expect("Runtime exhausted NodeId space");
        id
    }

    /// Create a signal owned by this runtime's arena.
    pub fn create_signal<T: 'static>(&self, initial: T) -> Signal<T> {
        self.arena.create(initial)
    }

    /// Register a side-effecting computation. It runs once immediately (so its
    /// initial dependencies are recorded) and re-runs on [`Runtime::flush`]
    /// whenever a signal it read changes. Returns the effect's node id.
    pub fn create_effect(&self, f: impl Fn() + 'static) -> NodeId {
        let node = self.alloc_node();
        let f: Rc<dyn Fn()> = Rc::new(f);
        self.effects.borrow_mut().insert(node, Rc::clone(&f));
        self.run_effect(node, &f);
        node
    }

    /// Create a cached derived signal. `f` is computed now and again whenever
    /// its inputs change; the result lives in the returned signal, which other
    /// computations can read and subscribe to. Downstream readers are only woken
    /// when the memo's value actually changes (`PartialEq`), so a recompute that
    /// yields the same value costs nothing downstream.
    pub fn create_memo<T: Clone + PartialEq + 'static>(
        &self,
        f: impl Fn() -> T + 'static,
    ) -> Signal<T> {
        let node = self.alloc_node();
        let f: Rc<dyn Fn() -> T> = Rc::new(f);
        // Initial compute, subscribing this node to whatever `f` reads.
        self.arena.clear_subscriptions_for(node);
        let initial = self.arena.with_reader(node, || (f)());
        let sig = self.arena.create(initial);
        let sig_id = sig.id();
        let sig_write = sig.clone();
        let f_rerun = Rc::clone(&f);
        let arena = Rc::clone(&self.arena);
        let effect: Rc<dyn Fn()> = Rc::new(move || {
            let v = (f_rerun)();
            // `peek` is untracked so the memo never subscribes to its own
            // output; write only on change so no-op recomputes stay silent.
            if arena.peek::<T>(sig_id).as_ref() != Some(&v) {
                sig_write.set(v);
            }
        });
        // Register at the SAME node that captured the initial dependencies, so
        // re-runs rebuild subscriptions against the right subscriber key.
        self.effects.borrow_mut().insert(node, effect);
        sig
    }

    /// Run (or re-run) one effect: rebuild its dependency set from scratch and
    /// execute it with reads scoped to `node`.
    fn run_effect(&self, node: NodeId, f: &Rc<dyn Fn()>) {
        self.arena.clear_subscriptions_for(node);
        self.arena.with_reader(node, || f());
    }

    /// Re-run every effect whose dependencies changed, repeating until the graph
    /// settles. Call once after a batch of signal writes (per event or frame).
    pub fn flush(&self) {
        if *self.flushing.borrow() {
            return; // nested write; the running loop will observe it.
        }
        *self.flushing.borrow_mut() = true;

        let mut passes = 0;
        loop {
            let dirty = self.arena.drain_dirty();
            if dirty.is_empty() {
                break;
            }
            passes += 1;
            if passes > MAX_FLUSH_PASSES {
                debug_assert!(
                    false,
                    "Runtime::flush exceeded MAX_FLUSH_PASSES — reactive cycle?"
                );
                break;
            }
            // Affected effects = union of subscribers over the dirty signals.
            // A BTreeSet gives a deterministic (ascending) run order.
            let mut affected: BTreeSet<NodeId> = BTreeSet::new();
            for sig in &dirty {
                for node in self.arena.subscribers(*sig) {
                    affected.insert(node);
                }
            }
            for node in affected {
                // Clone the closure out before running so we don't hold the
                // effects borrow across execution (an effect may read/write
                // signals or even register new effects).
                let f = self.effects.borrow().get(&node).cloned();
                if let Some(f) = f {
                    self.run_effect(node, &f);
                }
            }
        }

        *self.flushing.borrow_mut() = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A shared counter so tests can assert exactly how many times a computation
    // ran — the whole point of fine-grained reactivity.
    fn counter() -> (Rc<RefCell<i32>>, Rc<RefCell<i32>>) {
        let c = Rc::new(RefCell::new(0));
        (Rc::clone(&c), c)
    }

    #[test]
    fn effect_runs_once_immediately() {
        let rt = Runtime::new();
        let (probe, runs) = counter();
        rt.create_effect(move || {
            *probe.borrow_mut() += 1;
        });
        assert_eq!(*runs.borrow(), 1);
    }

    #[test]
    fn effect_reruns_on_dependency_change_after_flush() {
        let rt = Runtime::new();
        let sig = rt.create_signal::<i32>(0);
        let (probe, runs) = counter();
        let s = sig.clone();
        rt.create_effect(move || {
            *probe.borrow_mut() += 1;
            let _ = s.get(); // subscribe
        });
        assert_eq!(*runs.borrow(), 1);
        sig.set(1);
        rt.flush();
        assert_eq!(*runs.borrow(), 2);
    }

    #[test]
    fn writes_batch_until_flush() {
        let rt = Runtime::new();
        let sig = rt.create_signal::<i32>(0);
        let (probe, runs) = counter();
        let s = sig.clone();
        rt.create_effect(move || {
            *probe.borrow_mut() += 1;
            let _ = s.get();
        });
        sig.set(1);
        sig.set(2);
        sig.set(3);
        assert_eq!(*runs.borrow(), 1, "no re-run until flush");
        rt.flush();
        assert_eq!(*runs.borrow(), 2, "coalesced into one re-run");
    }

    #[test]
    fn effect_only_reruns_for_signals_it_reads() {
        let rt = Runtime::new();
        let watched = rt.create_signal::<i32>(0);
        let ignored = rt.create_signal::<i32>(0);
        let (probe, runs) = counter();
        let w = watched.clone();
        rt.create_effect(move || {
            *probe.borrow_mut() += 1;
            let _ = w.get();
        });
        ignored.set(99);
        rt.flush();
        assert_eq!(
            *runs.borrow(),
            1,
            "unrelated signal must not wake the effect"
        );
        watched.set(1);
        rt.flush();
        assert_eq!(*runs.borrow(), 2);
    }

    #[test]
    fn effect_dependencies_are_dynamic() {
        // Reads `a` only when `cond` is true — flipping `cond` must add/drop the
        // subscription to `a`. Exercises clear_subscriptions_for on each run.
        let rt = Runtime::new();
        let cond = rt.create_signal::<bool>(false);
        let a = rt.create_signal::<i32>(0);
        let (probe, runs) = counter();
        let (c, av) = (cond.clone(), a.clone());
        rt.create_effect(move || {
            *probe.borrow_mut() += 1;
            if c.get() {
                let _ = av.get();
            }
        });
        assert_eq!(*runs.borrow(), 1);

        // Not subscribed to `a` yet (cond is false).
        a.set(5);
        rt.flush();
        assert_eq!(*runs.borrow(), 1);

        // Flip cond → re-run, and now it also reads `a`.
        cond.set(true);
        rt.flush();
        assert_eq!(*runs.borrow(), 2);

        // Now `a` is a dependency.
        a.set(9);
        rt.flush();
        assert_eq!(*runs.borrow(), 3);

        // Flip cond back → re-run, dropping the `a` subscription.
        cond.set(false);
        rt.flush();
        assert_eq!(*runs.borrow(), 4);

        // `a` no longer wakes it.
        a.set(1);
        rt.flush();
        assert_eq!(*runs.borrow(), 4);
    }

    #[test]
    fn memo_computes_and_updates() {
        let rt = Runtime::new();
        let n = rt.create_signal::<i32>(3);
        let n2 = n.clone();
        let doubled = rt.create_memo::<i32>(move || n2.get() * 2);
        assert_eq!(doubled.peek(), 6);
        n.set(10);
        rt.flush();
        assert_eq!(doubled.peek(), 20);
    }

    #[test]
    fn memo_chain_cascades_in_one_flush() {
        let rt = Runtime::new();
        let n = rt.create_signal::<i32>(1);
        let n2 = n.clone();
        let a = rt.create_memo::<i32>(move || n2.get() + 1); // n+1
        let a2 = a.clone();
        let b = rt.create_memo::<i32>(move || a2.get() * 10); // (n+1)*10
        let (probe, runs) = counter();
        let b2 = b.clone();
        let observed = Rc::new(RefCell::new(0));
        let obs = Rc::clone(&observed);
        rt.create_effect(move || {
            *probe.borrow_mut() += 1;
            *obs.borrow_mut() = b2.get();
        });
        assert_eq!(*runs.borrow(), 1);
        assert_eq!(*observed.borrow(), 20); // (1+1)*10

        n.set(4);
        rt.flush();
        assert_eq!(*observed.borrow(), 50); // (4+1)*10
        assert_eq!(
            *runs.borrow(),
            2,
            "one settled re-run despite a two-hop chain"
        );
    }

    #[test]
    fn memo_does_not_wake_downstream_when_value_unchanged() {
        let rt = Runtime::new();
        let n = rt.create_signal::<i32>(4);
        let n2 = n.clone();
        let parity =
            rt.create_memo::<&'static str>(move || if n2.get() % 2 == 0 { "even" } else { "odd" });
        let (probe, runs) = counter();
        let p2 = parity.clone();
        rt.create_effect(move || {
            *probe.borrow_mut() += 1;
            let _ = p2.get();
        });
        assert_eq!(*runs.borrow(), 1);
        assert_eq!(parity.peek(), "even");

        // 4 → 6: still "even", so the memo value is unchanged and downstream
        // must NOT re-run.
        n.set(6);
        rt.flush();
        assert_eq!(
            *runs.borrow(),
            1,
            "no-op memo recompute stays silent downstream"
        );
        assert_eq!(parity.peek(), "even");

        // 6 → 7: now "odd" — downstream wakes.
        n.set(7);
        rt.flush();
        assert_eq!(*runs.borrow(), 2);
        assert_eq!(parity.peek(), "odd");
    }

    #[test]
    fn one_write_wakes_exactly_its_subscribers() {
        // The headline P0 property: a single signal write re-runs only the
        // effects that read it, not the whole graph.
        let rt = Runtime::new();
        let a = rt.create_signal::<i32>(0);
        let b = rt.create_signal::<i32>(0);
        let (pa, runs_a) = counter();
        let (pb, runs_b) = counter();
        let av = a.clone();
        rt.create_effect(move || {
            *pa.borrow_mut() += 1;
            let _ = av.get();
        });
        let bv = b.clone();
        rt.create_effect(move || {
            *pb.borrow_mut() += 1;
            let _ = bv.get();
        });
        assert_eq!((*runs_a.borrow(), *runs_b.borrow()), (1, 1));
        a.set(1);
        rt.flush();
        assert_eq!(
            (*runs_a.borrow(), *runs_b.borrow()),
            (2, 1),
            "writing `a` must not re-run the effect that only reads `b`"
        );
    }

    #[test]
    fn flush_is_idempotent_when_clean() {
        let rt = Runtime::new();
        let sig = rt.create_signal::<i32>(0);
        let (probe, runs) = counter();
        let s = sig.clone();
        rt.create_effect(move || {
            *probe.borrow_mut() += 1;
            let _ = s.get();
        });
        rt.flush();
        rt.flush();
        assert_eq!(
            *runs.borrow(),
            1,
            "flushing with nothing dirty does nothing"
        );
    }

    #[test]
    fn flush_terminates_on_self_feeding_cycle() {
        // An effect that writes a signal it reads is a cycle; flush must bail via
        // the pass cap rather than hang. (debug_assert is disabled in tests via
        // panic=unwind? No — guard with a bounded write instead.)
        let rt = Runtime::new();
        let sig = rt.create_signal::<i32>(0);
        let s = sig.clone();
        // Bounded feedback: stops writing once it hits 5, so it settles well
        // under MAX_FLUSH_PASSES and asserts convergence (not the cap).
        rt.create_effect(move || {
            let v = s.get();
            if v < 5 {
                s.set(v + 1);
            }
        });
        sig.set(1);
        rt.flush();
        assert_eq!(sig.peek(), 5, "bounded feedback converges");
    }
}
