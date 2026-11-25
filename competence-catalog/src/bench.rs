//! RCE-BENCH — the universal, physical, non-moving benchmark standard.
//!
//! Existing benchmarks measure accuracy on a *shifting* test set: static sets saturate and
//! contaminate (HumanEval/MBPP >95% by 2023), so the field either hardens the set (EvalPlus)
//! or harvests fresh problems (LiveCodeBench) or moves the goalpost (ARC-AGI-1→2→3). None is a
//! *true standard* because the metric is a saturating accuracy score, not a physical quantity.
//!
//! RCE-BENCH fixes the method, not the target: a task is a content-addressed **spec** (intent +
//! acceptance behaviour); correctness is the **gate** (replay reproduces the spec, measured by
//! spectests, not vacuous soundness); the ranking is **joules to correct completion** plus the
//! **information residual** (reused vs synthesized). Every language lowers to the same bounded
//! primitive floor, so the same spec filters every language identically. The suite is published
//! and replayable; new tasks extend it by content key without invalidating old results — the
//! standard cannot move its goalposts, because the standard is the gate + the joule, not a score.

use std::collections::HashSet;

use crate::canonical::{emit, normalize};
use crate::lift::lift_expr;
use crate::repair::eval_expr;

/// A benchmark task: a fixed, content-addressed spec (intent + acceptance behaviour).
pub struct BenchTask {
    pub id: &'static str,
    /// The intent source (the spec's candidate behaviour).
    pub intent: &'static str,
    /// The acceptance value the gate replays against (fixed; never moved).
    pub accept: i128,
}

/// A bench receipt: the gate verdict, the content key, the price, and whether it was reuse.
pub struct Receipt {
    pub id: String,
    /// The gate: candidate replays to the fixed acceptance (correctness as constraint).
    pub correct: bool,
    /// Name-free content key of the canonical form (the task's identity).
    pub key: String,
    /// The canonical form submitted.
    pub form: String,
    /// Joules to correct completion (gear-priced; est unless a meter backs it).
    pub est_uj: u64,
    /// Whether the form was already owned competence (G0 reuse, I≈0) vs synthesized.
    pub reused: bool,
}

/// The published, fixed suite. Extending it adds content keys; it never edits old ones.
pub fn suite() -> Vec<BenchTask> {
    vec![
        BenchTask { id: "add", intent: "2 + 3", accept: 5 },
        BenchTask { id: "muladd", intent: "2 * 3 + 4", accept: 10 },
        BenchTask { id: "fold", intent: "2 * 3", accept: 6 },
        BenchTask { id: "neg", intent: "-(4 + 5)", accept: -9 },
        BenchTask { id: "mod", intent: "7 % 3", accept: 1 },
    ]
}

/// The universal filter: lift the candidate (any surface in the fragment), replay it against
/// the fixed acceptance (the gate), content-address it, and price it by gear. The gate is the
/// same for every language and every submitter — no goalpost, no inspection, no vacuous pass.
pub fn filter(task: &BenchTask, candidate: &str, corpus: &HashSet<String>) -> Receipt {
    let lifted = lift_expr(candidate).ok();
    let (correct, form, key) = match lifted {
        Some(e) => {
            let correct = eval_expr(&e) == Some(task.accept);
            let form = emit(&normalize(e));
            let key = rcevc_core::sha256(form.as_bytes()).hex();
            (correct, form, key)
        }
        None => (false, String::new(), String::new()),
    };
    let reused = !key.is_empty() && corpus.contains(&key);
    // Gear-priced joules: reuse at the floor; synthesis mid; a failed gate is the model corner.
    let est_uj = if reused { 1 } else if correct { 1_000 } else { 5_000_000 };
    Receipt { id: task.id.to_string(), correct, key, form, est_uj, reused }
}

/// Run the whole suite through the universal filter with a per-task candidate chooser.
pub fn run_suite<F: Fn(&BenchTask) -> String>(choose: F, corpus: &HashSet<String>) -> Vec<Receipt> {
    suite().iter().map(|t| filter(t, &choose(t), corpus)).collect()
}

/// (pass rate, total joules) — the physical summary, not an accuracy leaderboard.
pub fn summary(rs: &[Receipt]) -> (f64, u64) {
    if rs.is_empty() {
        return (0.0, 0);
    }
    let pass = rs.iter().filter(|r| r.correct).count() as f64 / rs.len() as f64;
    let joules: u64 = rs.iter().map(|r| r.est_uj).sum();
    (pass, joules)
}