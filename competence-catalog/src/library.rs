//! Frontier 4 — library learning over the corpus (DreamCoder / stitch / babble discipline).
//!
//! The mature synthesis frontier the LLM crowd overlooks: learn reusable abstractions from a corpus
//! of solved Forms (wake-sleep library learning / corpus-guided top-down synthesis), then rewrite the
//! corpus with them to shrink description length. This is the inverse of crystallization: instead of
//! caching whole competences, it discovers the *common substructure* and promotes it to a library
//! function, so novel intent→Form synthesis composes learned abstractions rather than raw primitives.
//! Equality-saturation-lite: abstractions are identified by canonical (name-free) key.

use std::collections::HashMap;

use rce_spec_check::ast::Expr;

use crate::canonical::{emit, normalize};
use crate::structural::node_count;

/// Every subtree of a Form (candidate abstractions).
pub fn subtrees(e: &Expr) -> Vec<Expr> {
    use Expr::*;
    let mut out = vec![e.clone()];
    match e {
        Neg(a, _) => out.extend(subtrees(a)),
        Bin(_, l, r, _) => {
            out.extend(subtrees(l));
            out.extend(subtrees(r));
        }
        _ => {}
    }
    out
}

/// Name-free canonical key of a subtree (equality up to canonical form).
pub fn key(e: &Expr) -> String {
    emit(&normalize(e.clone()))
}

/// Learn the frequent, non-trivial subtrees across a corpus (DreamCoder/stitch-style library).
/// Returns (abstraction, occurrence count) for subtrees seen >= min_freq times, largest first.
pub fn learn_library(corpus: &[Expr], min_freq: usize) -> Vec<(Expr, usize)> {
    let mut count: HashMap<String, (Expr, usize)> = HashMap::new();
    for e in corpus {
        for s in subtrees(e) {
            if node_count(&s) < 2 {
                continue; // skip leaves; abstractions must be compound
            }
            let k = key(&s);
            let entry = count.entry(k).or_insert((s.clone(), 0));
            entry.1 += 1;
        }
    }
    let mut lib: Vec<(Expr, usize)> = count.into_values().filter(|(_, c)| *c >= min_freq).collect();
    lib.sort_by(|a, b| b.1.cmp(&a.1).then(node_count(&b.0).cmp(&node_count(&a.0))));
    lib
}

/// Total description length of the corpus.
pub fn description_length(corpus: &[Expr]) -> usize {
    corpus.iter().map(node_count).sum()
}

/// Description length after rewriting with the learned library: each occurrence of an abstraction
/// beyond the first collapses to a single reference node (saving (count-1)*(size-1)).
pub fn compressed_length(corpus: &[Expr], lib: &[(Expr, usize)]) -> usize {
    let orig = description_length(corpus);
    let saving: usize = lib.iter().map(|(a, c)| (c - 1) * (node_count(a) - 1)).sum();
    orig.saturating_sub(saving)
}