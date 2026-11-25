//! World Model Coding — a world model whose **physics is known exactly**.
//!
//! The RCE-JEPA thesis ("World Coding, with JEPA as premise"): in robotics and Code World Models
//! the forward dynamics must be *learned* (V-JEPA 2, MuZero, CWM). In RCE the transition function
//! is given and proved, so the forward model needs no learning at all — exact rollout *is* the
//! simulator. What remains is the **inverse**: which edit moves a candidate toward the intent.
//! That inverse can be searched (this module, deterministic) or learned (a JEPA/EBM proposer,
//! `jepa.rs`); either way it only *proposes*. Latents rank, they never render; the gate alone
//! admits. This is model-predictive planning over Forms with known physics: the model steers,
//! the engine renders.
//!
//! The planner here is the deterministic inverse: goal-conditioned beam search over typed edits
//! (`repair::mutations`), exact rollout (`eval_expr`) at every node, energy-ranked, gate-verified.

use crate::canonical::{emit, normalize};
use crate::lift::lift_expr;
use crate::repair::{eval_expr, mutations};

/// A verified plan: the reached Form, the edit path, and its estimated cost.
pub struct Plan {
    pub form: String,
    pub edits: Vec<String>,
    pub cost_uj: u64,
    pub verified: bool,
}

/// Goal-conditioned planning with exact physics: beam search over typed edits, exact rollout as
/// the simulator, cheapest-first. Returns the first gate-verified Form reaching `goal`, or None.
/// The forward dynamics are exact (no learned model); only the inverse is searched.
pub fn plan(start: &str, goal: i128, beam: usize, max_depth: usize) -> Option<Plan> {
    let e0 = lift_expr(start).ok()?;
    if eval_expr(&e0) == Some(goal) {
        return Some(Plan { form: emit(&normalize(e0)), edits: vec![], cost_uj: 1, verified: true });
    }
    let mut frontier: Vec<(rce_spec_check::ast::Expr, Vec<String>, u64)> = vec![(e0, vec![], 0)];
    for _ in 0..max_depth {
        let mut next = Vec::new();
        for (e, path, cost) in &frontier {
            for (m, kind) in mutations(e) {
                let c = cost + 1_000; // per-edit synthesis cost (est)
                let mut p = path.clone();
                p.push(kind);
                if eval_expr(&m) == Some(goal) {
                    return Some(Plan { form: emit(&normalize(m)), edits: p, cost_uj: c, verified: true });
                }
                next.push((m, p, c));
            }
        }
        // Bound the search: keep the `beam` candidates closest to the goal (exact-rollout heuristic).
        next.sort_by_key(|(m, _, c)| (eval_expr(m).map(|v| (v - goal).abs()).unwrap_or(i128::MAX), *c));
        next.truncate(beam);
        if next.is_empty() {
            return None;
        }
        frontier = next;
    }
    None
}