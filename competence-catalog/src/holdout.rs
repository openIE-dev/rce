//! Held-out reuse experiment (the thesis-deciding edge): does admitted competence generalize to
//! requests we did not build? Admit a base library, then issue held-out requests — paraphrases of
//! admitted behaviour (should hit via the name-free behavior key) and novel behaviour (should miss
//! and fall to the proposer). Hit-rate is measured, not assumed.

use std::collections::HashSet;

use crate::program::{generative_probes, lift_program};

/// Name-free behavior key of a program (sorted outputs over the generative battery).
pub fn behavior_key(src: &str) -> Option<String> {
    let stmt = lift_program(src).ok()?;
    let p = generative_probes(&stmt);
    let mut outs: Vec<i128> = p.iter().map(|(_, r)| *r).collect();
    outs.sort_unstable();
    Some(rcevc_core::sha256(format!("{outs:?}").as_bytes()).hex())
}

/// (hits, total) of held-out requests whose behavior key matches an admitted competence.
pub fn hit_rate(admitted: &[&str], requests: &[&str]) -> (usize, usize) {
    let keys: HashSet<String> = admitted.iter().filter_map(|s| behavior_key(s)).collect();
    let hits = requests.iter().filter(|s| behavior_key(s).map(|k| keys.contains(&k)).unwrap_or(false)).count();
    (hits, requests.len())
}
