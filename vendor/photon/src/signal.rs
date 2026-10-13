//! Fine-grained reactive state for Photon.
//!
//! Photon's declarative views read state through [`Signal`] handles. When a
//! signal is read while the [`SignalArena`] has a *current reader* set (via
//! [`SignalArena::with_reader`]), the arena records a dependency edge from the
//! signal to that reader node. Subsequent writes to the signal mark it dirty
//! and the runtime can then re-run only the readers that depend on it.
//!
//! This is the same pattern used by SolidJS and Leptos. Phase 1 ships the
//! arena, the public [`Signal`] handle, and a dirty-set drain. The render
//! pipeline in [`crate::render::render_node`] consults
//! [`SignalArena::with_reader`] when entering a `ViewNode::Stateful` block so
//! that signals read inside the closure subscribe to that node.

use crate::animation::AnimationSpec;
use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::rc::Rc;

thread_local! {
    /// The arena that should receive `with_animation` envelopes for the
    /// current thread. Photon assumes one arena per thread (the SDK enforces
    /// this); the cell is set on entry to [`SignalArena::with_animation`]
    /// and cleared on exit.
    static CURRENT_ARENA: RefCell<Option<Rc<SignalArena>>> = const { RefCell::new(None) };
}

/// Stable identifier for a [`Signal`] within an arena.
pub type SignalId = u32;

/// Identifier for a render node — used as the subscriber key in the arena.
pub type NodeId = u32;

/// Reactive value store. Every [`Signal`] is owned by exactly one arena.
///
/// Use [`SignalArena::new`] to construct (returns an `Rc<Self>` because
/// [`Signal`] handles need to outlive a single borrow).
pub struct SignalArena {
    /// Heterogeneous storage indexed by [`SignalId`].
    values: RefCell<Vec<Box<dyn Any>>>,
    /// Reader subscriptions: signal id -> node ids that depend on it.
    subscribers: RefCell<HashMap<SignalId, HashSet<NodeId>>>,
    /// The reader currently being executed; signal reads inside
    /// [`SignalArena::with_reader`] auto-subscribe this node.
    current_reader: RefCell<Option<NodeId>>,
    /// Signals written since the last [`SignalArena::drain_dirty`] call.
    dirty: RefCell<HashSet<SignalId>>,
    /// Monotonic counter feeding [`Self::create`].
    next_id: RefCell<SignalId>,
    /// `(SignalId → AnimationSpec)` map populated by writes that happen
    /// inside [`SignalArena::with_animation`]. Drained alongside
    /// [`SignalArena::drain_dirty_with_animation`].
    pending_animations: RefCell<HashMap<SignalId, AnimationSpec>>,
    /// The animation envelope currently active for this arena, if any.
    /// Pushed by [`SignalArena::with_animation`] and popped on exit.
    current_animation: RefCell<Option<AnimationSpec>>,
}

impl SignalArena {
    /// Create an empty arena. Returns an `Rc<Self>` because every [`Signal`]
    /// keeps a strong reference back to its arena.
    pub fn new() -> Rc<Self> {
        Rc::new(Self {
            values: RefCell::new(Vec::new()),
            subscribers: RefCell::new(HashMap::new()),
            current_reader: RefCell::new(None),
            dirty: RefCell::new(HashSet::new()),
            next_id: RefCell::new(0),
            pending_animations: RefCell::new(HashMap::new()),
            current_animation: RefCell::new(None),
        })
    }

    /// Allocate a new signal seeded with `initial`. The returned [`Signal`]
    /// holds a clone of the arena's `Rc`.
    pub fn create<T: 'static>(self: &Rc<Self>, initial: T) -> Signal<T> {
        let id = {
            let mut next = self.next_id.borrow_mut();
            let id = *next;
            *next = next
                .checked_add(1)
                .expect("SignalArena exhausted SignalId space");
            id
        };
        let mut values = self.values.borrow_mut();
        // Pad the values vector so signal ids index directly.
        while values.len() as u32 <= id {
            values.push(Box::new(()) as Box<dyn Any>);
        }
        values[id as usize] = Box::new(initial);
        Signal {
            arena: Rc::clone(self),
            id,
            _t: PhantomData,
        }
    }

    /// Read the value at `id` as `T`. Records a subscription edge if a
    /// current reader is set. Returns `None` if `id` is unknown or `T`
    /// doesn't match the stored type.
    pub fn read<T: Clone + 'static>(&self, id: SignalId) -> Option<T> {
        // Track this signal as a dependency of the current reader.
        if let Some(node) = *self.current_reader.borrow() {
            self.subscribers
                .borrow_mut()
                .entry(id)
                .or_default()
                .insert(node);
        }
        let values = self.values.borrow();
        let slot = values.get(id as usize)?;
        slot.downcast_ref::<T>().cloned()
    }

    /// Read the value at `id` as `T` **without** recording a dependency edge,
    /// even if a current reader is set. Used by the reactive runtime to read a
    /// memo's own cached value while recomputing it (which must not subscribe
    /// the memo to itself) and anywhere an "untracked" peek is wanted.
    pub fn peek<T: Clone + 'static>(&self, id: SignalId) -> Option<T> {
        let values = self.values.borrow();
        values.get(id as usize)?.downcast_ref::<T>().cloned()
    }

    /// Remove `node` from every signal's subscriber set. The reactive runtime
    /// calls this immediately before re-running an effect so that the effect's
    /// dependency set is rebuilt from scratch on each run — a signal read only
    /// on the previous run (e.g. behind a since-flipped `if`) correctly stops
    /// triggering the effect.
    pub fn clear_subscriptions_for(&self, node: NodeId) {
        for set in self.subscribers.borrow_mut().values_mut() {
            set.remove(&node);
        }
    }

    /// Overwrite the value at `id`. Marks the signal dirty so subscribers can
    /// be re-run on the next frame. If a [`SignalArena::with_animation`]
    /// envelope is active, the write is tagged with that spec.
    pub fn write<T: 'static>(&self, id: SignalId, value: T) {
        let mut values = self.values.borrow_mut();
        if let Some(slot) = values.get_mut(id as usize) {
            *slot = Box::new(value);
            self.dirty.borrow_mut().insert(id);
            if let Some(spec) = *self.current_animation.borrow() {
                self.pending_animations.borrow_mut().insert(id, spec);
            }
        }
    }

    /// Mutate the value at `id` in place. No-op if `T` doesn't match the
    /// stored type. Honours any active [`SignalArena::with_animation`]
    /// envelope.
    pub fn update<T: 'static>(&self, id: SignalId, f: impl FnOnce(&mut T)) {
        let mut values = self.values.borrow_mut();
        if let Some(slot) = values.get_mut(id as usize) {
            if let Some(inner) = slot.downcast_mut::<T>() {
                f(inner);
                self.dirty.borrow_mut().insert(id);
                if let Some(spec) = *self.current_animation.borrow() {
                    self.pending_animations.borrow_mut().insert(id, spec);
                }
            }
        }
    }

    /// Snapshot of the subscriber set for `id`. Returns an empty vector if
    /// no readers depend on the signal.
    pub fn subscribers(&self, id: SignalId) -> Vec<NodeId> {
        self.subscribers
            .borrow()
            .get(&id)
            .map(|set| {
                let mut v: Vec<NodeId> = set.iter().copied().collect();
                v.sort_unstable();
                v
            })
            .unwrap_or_default()
    }

    /// Take the dirty set, leaving an empty one behind. The runtime calls
    /// this once per frame to discover which signals (and therefore which
    /// nodes) need to be recomputed.
    pub fn drain_dirty(&self) -> HashSet<SignalId> {
        // Drop any pending animation envelopes that no longer have a dirty
        // companion — the host calls this when it doesn't care about
        // animations, so the next frame should start clean.
        self.pending_animations.borrow_mut().clear();
        std::mem::take(&mut *self.dirty.borrow_mut())
    }

    /// Drain dirty signals paired with the animation spec (if any) that
    /// caused them. The host runtime invokes this once per frame to drive
    /// in-flight animations: each entry maps a freshly written signal to
    /// the [`AnimationSpec`] supplied by the surrounding
    /// [`SignalArena::with_animation`] block, or `None` for ordinary
    /// writes.
    pub fn drain_dirty_with_animation(&self) -> Vec<(SignalId, Option<AnimationSpec>)> {
        let dirty = std::mem::take(&mut *self.dirty.borrow_mut());
        let mut pending = std::mem::take(&mut *self.pending_animations.borrow_mut());
        let mut out: Vec<(SignalId, Option<AnimationSpec>)> = dirty
            .into_iter()
            .map(|id| (id, pending.remove(&id)))
            .collect();
        out.sort_by_key(|(id, _)| *id);
        out
    }

    /// Run `f` while marking `node` as the current reader. Any signal reads
    /// inside `f` auto-subscribe `node`. Restores the previous reader on
    /// exit (so reader scopes nest).
    pub fn with_reader<R>(&self, node: NodeId, f: impl FnOnce() -> R) -> R {
        let prev = self.current_reader.borrow_mut().replace(node);
        let r = f();
        *self.current_reader.borrow_mut() = prev;
        r
    }

    /// Tag every write performed inside `f` with the supplied animation
    /// envelope. Mirrors SwiftUI's
    /// [`withAnimation`](https://developer.apple.com/documentation/swiftui/withanimation):
    /// any signal mutated inside the closure surfaces in
    /// [`SignalArena::drain_dirty_with_animation`] paired with `spec`.
    ///
    /// The arena is also installed as the *current* arena for the duration
    /// of the call, so the free-standing [`crate::with_animation`] helper
    /// can dispatch to whichever arena is in scope without the caller
    /// passing it explicitly.
    pub fn with_animation<R>(self: &Rc<Self>, spec: AnimationSpec, f: impl FnOnce() -> R) -> R {
        let prev_anim = self.current_animation.borrow_mut().replace(spec);
        let prev_arena = CURRENT_ARENA.with(|cell| cell.borrow_mut().replace(Rc::clone(self)));
        let r = f();
        *self.current_animation.borrow_mut() = prev_anim;
        CURRENT_ARENA.with(|cell| *cell.borrow_mut() = prev_arena);
        r
    }
}

/// Look up the [`SignalArena`] currently installed by an enclosing
/// [`SignalArena::with_animation`] call on this thread, if any.
pub fn current_arena() -> Option<Rc<SignalArena>> {
    CURRENT_ARENA.with(|cell| cell.borrow().as_ref().map(Rc::clone))
}

/// Typed handle to a value owned by a [`SignalArena`].
///
/// `Signal` is the central reactive primitive: `get` reads (and subscribes
/// the current reader) and `set`/`update` write (and mark the signal dirty).
pub struct Signal<T> {
    arena: Rc<SignalArena>,
    id: SignalId,
    _t: PhantomData<T>,
}

impl<T: Clone + 'static> Signal<T> {
    /// Read the current value and record a dependency on the active reader.
    ///
    /// # Panics
    /// Panics if the underlying slot has been overwritten with a different
    /// type — this can only happen via direct `SignalArena::write::<U>(id, …)`
    /// with a mismatched `U`, which is a programmer error.
    pub fn get(&self) -> T {
        self.arena
            .read::<T>(self.id)
            .expect("Signal type mismatch — slot overwritten with a different type")
    }

    /// Replace the value. Marks the signal dirty.
    pub fn set(&self, value: T) {
        self.arena.write::<T>(self.id, value);
    }

    /// Mutate the value in place via `f`. Marks the signal dirty.
    pub fn update(&self, f: impl FnOnce(&mut T)) {
        self.arena.update::<T>(self.id, f);
    }

    /// Read the current value **without** subscribing the active reader.
    /// Use inside a reactive computation to read a signal you don't want to
    /// depend on.
    ///
    /// # Panics
    /// Same type-mismatch panic as [`Signal::get`].
    pub fn peek(&self) -> T {
        self.arena
            .peek::<T>(self.id)
            .expect("Signal type mismatch — slot overwritten with a different type")
    }

    /// The arena-local id for this signal.
    pub fn id(&self) -> SignalId {
        self.id
    }
}

impl<T> Clone for Signal<T> {
    fn clone(&self) -> Self {
        Self {
            arena: Rc::clone(&self.arena),
            id: self.id,
            _t: PhantomData,
        }
    }
}

/// Two-way binding alias. Conceptually identical to [`Signal`] — both ends
/// share a single arena slot — but consumers can distinguish the API at the
/// type level.
pub type Binding<T> = Signal<T>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_get_returns_initial_value() {
        let arena = SignalArena::new();
        let sig = arena.create::<i32>(42);
        assert_eq!(sig.get(), 42);
    }

    #[test]
    fn set_overwrites_value() {
        let arena = SignalArena::new();
        let sig = arena.create::<i32>(0);
        sig.set(7);
        assert_eq!(sig.get(), 7);
    }

    #[test]
    fn update_mutates_in_place() {
        let arena = SignalArena::new();
        let sig = arena.create::<i32>(10);
        sig.update(|v| *v += 5);
        assert_eq!(sig.get(), 15);
    }

    #[test]
    fn write_marks_signal_dirty() {
        let arena = SignalArena::new();
        let sig = arena.create::<&'static str>("a");
        sig.set("b");
        let dirty = arena.drain_dirty();
        assert!(dirty.contains(&sig.id()));
    }

    #[test]
    fn drain_dirty_clears_set() {
        let arena = SignalArena::new();
        let sig = arena.create::<i32>(0);
        sig.set(1);
        let _ = arena.drain_dirty();
        assert!(arena.drain_dirty().is_empty());
    }

    #[test]
    fn with_reader_records_subscription() {
        let arena = SignalArena::new();
        let sig = arena.create::<i32>(0);
        arena.with_reader(99, || {
            let _ = sig.get();
        });
        let subs = arena.subscribers(sig.id());
        assert_eq!(subs, vec![99]);
    }

    #[test]
    fn read_outside_with_reader_does_not_subscribe() {
        let arena = SignalArena::new();
        let sig = arena.create::<i32>(0);
        let _ = sig.get();
        assert!(arena.subscribers(sig.id()).is_empty());
    }

    #[test]
    fn with_reader_restores_previous_reader() {
        let arena = SignalArena::new();
        let outer_sig = arena.create::<i32>(0);
        let inner_sig = arena.create::<i32>(0);
        arena.with_reader(1, || {
            arena.with_reader(2, || {
                let _ = inner_sig.get();
            });
            let _ = outer_sig.get();
        });
        assert_eq!(arena.subscribers(inner_sig.id()), vec![2]);
        assert_eq!(arena.subscribers(outer_sig.id()), vec![1]);
    }

    #[test]
    fn multiple_subscribers_for_one_signal() {
        let arena = SignalArena::new();
        let sig = arena.create::<i32>(0);
        arena.with_reader(1, || {
            let _ = sig.get();
        });
        arena.with_reader(2, || {
            let _ = sig.get();
        });
        let subs = arena.subscribers(sig.id());
        assert_eq!(subs, vec![1, 2]);
    }

    #[test]
    fn signals_can_hold_distinct_types() {
        let arena = SignalArena::new();
        let int_sig = arena.create::<i32>(7);
        let str_sig = arena.create::<String>("hi".to_string());
        assert_eq!(int_sig.get(), 7);
        assert_eq!(str_sig.get(), "hi");
    }

    #[test]
    fn clone_signal_shares_storage() {
        let arena = SignalArena::new();
        let a = arena.create::<i32>(0);
        let b = a.clone();
        a.set(42);
        assert_eq!(b.get(), 42);
        assert_eq!(a.id(), b.id());
    }

    #[test]
    fn binding_alias_is_same_as_signal() {
        let arena = SignalArena::new();
        let s: Signal<i32> = arena.create(1);
        let b: Binding<i32> = s.clone();
        b.set(99);
        assert_eq!(s.get(), 99);
    }

    #[test]
    fn ids_are_unique_and_monotonic() {
        let arena = SignalArena::new();
        let a = arena.create::<i32>(0);
        let b = arena.create::<i32>(0);
        let c = arena.create::<i32>(0);
        assert!(a.id() < b.id());
        assert!(b.id() < c.id());
    }

    // -------------------------------------------------------------------------
    // Phase 4 — with_animation envelope
    // -------------------------------------------------------------------------

    #[test]
    fn with_animation_tags_writes_with_spec() {
        use crate::animation::{AnimationCurve, AnimationSpec};
        let arena = SignalArena::new();
        let sig = arena.create::<i32>(0);
        let spec = AnimationSpec::new(AnimationCurve::Linear, 100);
        arena.with_animation(spec, || {
            sig.set(7);
        });
        let drained = arena.drain_dirty_with_animation();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].0, sig.id());
        assert_eq!(drained[0].1, Some(spec));
    }

    #[test]
    fn write_outside_with_animation_has_no_spec() {
        let arena = SignalArena::new();
        let sig = arena.create::<i32>(0);
        sig.set(7);
        let drained = arena.drain_dirty_with_animation();
        assert_eq!(drained.len(), 1);
        assert!(drained[0].1.is_none());
    }

    #[test]
    fn with_animation_restores_previous_envelope() {
        use crate::animation::{AnimationCurve, AnimationSpec};
        let arena = SignalArena::new();
        let sig = arena.create::<i32>(0);
        let outer = AnimationSpec::new(AnimationCurve::Linear, 100);
        let inner = AnimationSpec::new(AnimationCurve::EaseIn, 200);
        arena.with_animation(outer, || {
            arena.with_animation(inner, || {
                sig.set(1);
            });
            sig.update(|v| *v += 10);
        });
        let drained = arena.drain_dirty_with_animation();
        // The most recent write wins — the update happens after the inner
        // block exits, so the outer spec is what survives in the table.
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].1, Some(outer));
    }

    #[test]
    fn current_arena_is_set_inside_with_animation() {
        use crate::animation::linear;
        let arena = SignalArena::new();
        arena.with_animation(linear(100), || {
            let cur = super::current_arena();
            assert!(cur.is_some());
            assert!(Rc::ptr_eq(&cur.unwrap(), &arena));
        });
        // Outside the block, nothing is installed.
        assert!(super::current_arena().is_none());
    }

    #[test]
    fn drain_dirty_with_animation_sorts_by_signal_id() {
        use crate::animation::ease_in;
        let arena = SignalArena::new();
        let a = arena.create::<i32>(0);
        let b = arena.create::<i32>(0);
        arena.with_animation(ease_in(100), || {
            b.set(2);
            a.set(1);
        });
        let drained = arena.drain_dirty_with_animation();
        assert_eq!(drained[0].0, a.id());
        assert_eq!(drained[1].0, b.id());
    }

    #[test]
    fn drain_dirty_clears_pending_animations() {
        use crate::animation::linear;
        let arena = SignalArena::new();
        let sig = arena.create::<i32>(0);
        arena.with_animation(linear(100), || {
            sig.set(1);
        });
        let _ = arena.drain_dirty();
        // Subsequent drain_dirty_with_animation should see no leftover spec.
        let next = arena.drain_dirty_with_animation();
        assert!(next.is_empty());
    }
}
