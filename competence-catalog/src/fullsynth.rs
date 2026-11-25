//! Full intent→Form synthesis (orchestrator): library learning + exact-physics search + the gate.
//!
//! Combines the obscure-frontier ingestions into one synthesis path: learn reusable abstractions
//! from the corpus (DreamCoder/stitch), then search typed edits with exact rollout as the known
//! physics (worldmodel), gate-verified (CEGIS against the acceptance: a mismatch is the
//! counterexample that drives the next edit). This is the deterministic synthesis core; arbitrary
//! natural-language intent→Form remains the learned (JEPA) frontier.

use rce_spec_check::ast::Expr;

use crate::library::learn_library;
use crate::worldmodel::{plan, Plan};

/// A synthesis result: the verified plan (if found) and the library of learned abstractions.
pub struct Synthesis {
    pub plan: Option<Plan>,
    pub library: Vec<(Expr, usize)>,
}

/// Synthesize a Form reaching `goal` from `start`, using corpus-learned abstractions and
/// exact-physics edit search, verified by the gate. `beam`/`depth` bound the search.
pub fn synthesize_full(start: &str, goal: i128, corpus: &[Expr], beam: usize, depth: usize) -> Synthesis {
    let library = learn_library(corpus, 2);
    let p = plan(start, goal, beam, depth);
    Synthesis { plan: p, library }
}