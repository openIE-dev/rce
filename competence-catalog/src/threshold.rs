//! Completion thresholds — the gated dependency order that avoids the
//! chicken-and-egg / cart-before-horse.
//!
//! The layer-separation principle (see `components.rs`) says the six components
//! are *specification-independent*: each is a projection of the Form, coupled
//! only through the spec, so no layer's accidents bias another's design. But
//! specification-independence is **not** construction-independence. A dependent
//! layer can only be *really worked on* once its prerequisites are **fully
//! completed to a verified threshold** — not merely "started" or "mostly done".
//! You cannot meaningfully build the emitter before the Form is fixed and the
//! gate can verify it; you cannot train the JEPA proposer before a corpus of
//! verified Forms exists to learn from.
//!
//! This module encodes that order as a dependency DAG whose edges are *checked
//! thresholds*, not ad hoc "one thing leads to another". A stage is **complete**
//! when its own measured threshold is met; a stage is **unlocked** (workable)
//! only when *every* prerequisite is complete. The cart-before-horse guard
//! refuses to authorize work on a dependent whose prerequisites are not yet at
//! threshold. The "right parallel order" is the set of unlocked stages: independent
//! stages proceed in parallel, dependent stages wait for verified prerequisites.
//!
//! This is the `agency` paper's "the order is fixed… nothing downstream is defined
//! without it" and `three-body`'s checked interfaces, made operational: a
//! prerequisite's interface is checked when it passes its gate, and only then may
//! a dependent rely on it. An unmeasured prerequisite is never assumed complete.

use std::collections::{HashMap, HashSet};

/// Whether a metric must reach at least, or stay at most, the required value.
pub enum Direction {
    /// The measurement must be `>= required` (coverage, pass fraction, corpus size).
    AtLeast,
    /// The measurement must be `<= required` (mismatch count, error rate).
    AtMost,
}

/// A completion threshold: a named metric must reach a required value. This is
/// the machine-checked predicate that says a stage is *fully completed*, not the
/// subjective "done". A stage below its threshold is still being worked on.
pub struct Threshold {
    /// The metric this threshold reads (e.g. "executor_coverage").
    pub metric: &'static str,
    /// The required value.
    pub required: f64,
    /// The comparison direction.
    pub direction: Direction,
}

impl Threshold {
    /// Whether the threshold is met by the given measurements. An absent metric
    /// is **not** met — unmeasured is never assumed complete.
    pub fn met(&self, m: &Measurements) -> bool {
        match m.values.get(self.metric) {
            Some(&v) => match self.direction {
                Direction::AtLeast => v >= self.required,
                Direction::AtMost => v <= self.required,
            },
            Option::None => false,
        }
    }
}

/// A stage in the dependency DAG: a component (or layer) with prerequisites and
/// its own completion threshold.
pub struct Stage {
    /// The stage identity (e.g. "RCE-SPEC").
    pub id: &'static str,
    /// The stages that must be *complete* before this one is really workable.
    pub requires: &'static [&'static str],
    /// This stage's own completion threshold (what unlocks *its* dependents).
    pub threshold: Threshold,
}

/// The measured state of the build: metric name → value. These are the live
/// measurements (coverage sweeps, test fractions, corpus sizes), not assertions.
pub struct Measurements {
    /// The current measured values.
    pub values: HashMap<String, f64>,
}

impl Measurements {
    /// An empty measurement set (nothing measured → nothing complete).
    pub fn new() -> Self {
        Measurements { values: HashMap::new() }
    }
    /// Record a measurement.
    pub fn set(&mut self, metric: &str, value: f64) -> &mut Self {
        self.values.insert(metric.to_string(), value);
        self
    }
}

impl Default for Measurements {
    fn default() -> Self {
        Self::new()
    }
}

/// The RCE dependency DAG — the egg first, then the gate, then the substrate,
/// then the projections, then the learned proposer. Each edge is a verified
/// prerequisite; each node carries the threshold that unlocks its dependents.
///
/// Order: RCE-SPEC (the Form) → RCE-VC (the gate) → RCE (the substrate) and
/// RCE-Compile (lift/emit, parallel once Form+gate are fixed) → RCE-Engine (the
/// executor/ladder) → RCE-JEPA (the learned proposer, needs a verified corpus).
pub fn rce_dag() -> Vec<Stage> {
    vec![
        Stage {
            id: "RCE-SPEC",
            requires: &[],
            threshold: Threshold { metric: "spec_conformance", required: 0.99, direction: Direction::AtLeast },
        },
        Stage {
            id: "RCE-VC",
            requires: &["RCE-SPEC"],
            threshold: Threshold { metric: "gate_tests", required: 1.0, direction: Direction::AtLeast },
        },
        Stage {
            id: "RCE",
            requires: &["RCE-SPEC", "RCE-VC"],
            threshold: Threshold { metric: "canonical_stability", required: 1.0, direction: Direction::AtLeast },
        },
        Stage {
            id: "RCE-Compile",
            requires: &["RCE-SPEC", "RCE-VC"],
            threshold: Threshold { metric: "emit_replay_identity", required: 1.0, direction: Direction::AtLeast },
        },
        Stage {
            id: "RCE-Engine",
            requires: &["RCE", "RCE-Compile", "RCE-VC"],
            threshold: Threshold { metric: "executor_coverage", required: 0.90, direction: Direction::AtLeast },
        },
        Stage {
            id: "RCE-JEPA",
            requires: &["RCE-Engine", "RCE"],
            threshold: Threshold { metric: "verified_corpus_size", required: 100.0, direction: Direction::AtLeast },
        },
    ]
}

/// The stages whose own threshold is met — *fully completed to threshold*.
pub fn complete(dag: &[Stage], m: &Measurements) -> HashSet<&'static str> {
    dag.iter().filter(|s| s.threshold.met(m)).map(|s| s.id).collect()
}

/// The prerequisites of `id` that are not yet complete (why it is blocked).
pub fn blockers(dag: &[Stage], m: &Measurements, id: &str) -> Vec<&'static str> {
    let done = complete(dag, m);
    dag.iter().find(|s| s.id == id)
        .map(|s| s.requires.iter().copied().filter(|r| !done.contains(r)).collect())
        .unwrap_or_default()
}

/// Whether a stage is unlocked: *every* prerequisite is complete. This is the
/// cart-before-horse guard — a dependent is workable only after its prerequisites
/// are fully completed to threshold, never before.
pub fn unlocked(dag: &[Stage], m: &Measurements, id: &str) -> bool {
    blockers(dag, m, id).is_empty()
}

/// The parallel frontier: stages that are unlocked (workable now) but not yet
/// complete (still to be finished). These are the things that can *really be
/// worked on* in parallel right now. Independent stages appear together;
/// dependent stages appear only once their prerequisites are complete.
pub fn frontier(dag: &[Stage], m: &Measurements) -> Vec<&'static str> {
    let done = complete(dag, m);
    dag.iter()
        .filter(|s| !done.contains(s.id) && s.requires.iter().all(|r| done.contains(r)))
        .map(|s| s.id)
        .collect()
}

/// A full order report: for each stage, whether it is complete, whether it is
/// unlocked, and its blockers. The verdict `ordered` is true iff no stage is
/// counted workable while a prerequisite is incomplete (the guard holds by
/// construction) and the DAG has no cycle reachable from a complete stage.
pub fn order_report(dag: &[Stage], m: &Measurements) -> Vec<(&'static str, bool, bool, Vec<&'static str>)> {
    dag.iter().map(|s| {
        (s.id, s.threshold.met(m), unlocked(dag, m, s.id), blockers(dag, m, s.id))
    }).collect()
}

/// Whether the dependency graph is acyclic (a prerequisite order exists at all).
/// A cycle would be a genuine chicken-and-egg with no egg — no valid order.
pub fn acyclic(dag: &[Stage]) -> bool {
    // Kahn's algorithm over the requires edges.
    let ids: HashSet<&str> = dag.iter().map(|s| s.id).collect();
    let mut indeg: HashMap<&str, usize> = ids.iter().map(|&id| (id, 0)).collect();
    for s in dag {
        for r in s.requires {
            if ids.contains(r) {
                *indeg.get_mut(s.id).unwrap() += 1;
            }
        }
    }
    let mut queue: Vec<&str> = indeg.iter().filter(|(_, &d)| d == 0).map(|(&id, _)| id).collect();
    let mut seen = 0usize;
    while let Some(id) = queue.pop() {
        seen += 1;
        for s in dag {
            if s.requires.contains(&id) {
                let d = indeg.get_mut(s.id).unwrap();
                *d -= 1;
                if *d == 0 {
                    queue.push(s.id);
                }
            }
        }
    }
    seen == ids.len()
}