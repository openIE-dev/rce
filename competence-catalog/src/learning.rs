//! The public learning loop — hindsight and promotion, no private engine.
//!
//! The engine learns from its own *verified* resolutions. Hindsight records every
//! resolution that passed the replay gate and recalls it for a repeated intent
//! (a learned cache ahead of re-search). Promotion ranks competence by reuse,
//! prunes the cold, and distills behavioral duplicates — the skill-lifecycle
//! operators (Rerank / Prune / Merge) over the same replay-gated corpus.
//!
//! Learning never bypasses the gate: a recalled handler is still replay-verified
//! before use, and pruning only removes competence, never admits unverified code.
//! This is the public counterpart of what was once private; it shares no code
//! with joulesperbit.

use std::collections::HashMap;

/// A verified resolution remembered by hindsight.
#[derive(Clone, Debug)]
pub struct Experience {
    /// The operation that covered the intent.
    pub handler: String,
    /// The gear that resolved it.
    pub gear: u8,
    /// The intent shape.
    pub shape: String,
    /// Monotonic sequence (recency).
    pub seq: u64,
}

/// Experience memory: the engine's hindsight over its verified resolutions.
pub struct Hindsight {
    by_intent: HashMap<String, Experience>,
    reuse: HashMap<String, u64>,
    seq: u64,
}

impl Hindsight {
    /// An empty experience memory.
    pub fn new() -> Self {
        Hindsight { by_intent: HashMap::new(), reuse: HashMap::new(), seq: 0 }
    }

    /// Remember a verified resolution and count the handler's reuse.
    pub fn record(&mut self, intent_key: &str, shape: &str, handler: &str, gear: u8) {
        self.seq += 1;
        self.by_intent.insert(intent_key.to_string(), Experience {
            handler: handler.to_string(),
            gear,
            shape: shape.to_string(),
            seq: self.seq,
        });
        *self.reuse.entry(handler.to_string()).or_insert(0) += 1;
    }

    /// Recall the handler that previously covered this exact intent (the learned
    /// cache). The caller still replay-verifies before use.
    pub fn recall(&self, intent_key: &str) -> Option<&Experience> {
        self.by_intent.get(intent_key)
    }

    /// How many verified resolutions a handler has covered (its reuse).
    pub fn reuse(&self, handler: &str) -> u64 {
        self.reuse.get(handler).copied().unwrap_or(0)
    }

    /// Number of distinct intents remembered.
    pub fn len(&self) -> usize {
        self.by_intent.len()
    }

    /// Whether nothing has been learned yet.
    pub fn is_empty(&self) -> bool {
        self.by_intent.is_empty()
    }

    /// Handlers seen, ranked by reuse (descending), then recency.
    pub fn ranked_handlers(&self) -> Vec<String> {
        let mut v: Vec<(String, u64)> = self.reuse.iter().map(|(h, n)| (h.clone(), *n)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.into_iter().map(|(h, _)| h).collect()
    }
}

impl Default for Hindsight {
    fn default() -> Self {
        Self::new()
    }
}

/// Promotion — the skill-lifecycle operators over the replay-gated corpus.
pub struct Promotion;

impl Promotion {
    /// **Rerank**: order candidate handlers by reuse, hottest first, so the
    /// cheapest-to-trust competence is tried before the cold.
    pub fn rerank(candidates: &[String], h: &Hindsight) -> Vec<String> {
        let mut v: Vec<String> = candidates.to_vec();
        v.sort_by(|a, b| h.reuse(b).cmp(&h.reuse(a)).then(a.cmp(b)));
        v
    }

    /// **Prune**: keep candidates with at least `min_reuse` verified resolutions;
    /// the cold tail is dropped from the index (competence removed, never added).
    pub fn prune(candidates: &[String], h: &Hindsight, min_reuse: u64) -> Vec<String> {
        candidates.iter().filter(|c| h.reuse(c) >= min_reuse).cloned().collect()
    }

    /// **Merge / distill**: collapse behavioral duplicates. Given (handler,
    /// fingerprint) pairs, keep the highest-reuse handler per fingerprint — one
    /// representative per behavior, the rest distilled away.
    pub fn distill(by_fingerprint: &[(String, String)], h: &Hindsight) -> Vec<String> {
        let mut best: HashMap<String, (String, u64)> = HashMap::new();
        for (handler, fp) in by_fingerprint {
            let reuse = h.reuse(handler);
            best.entry(fp.clone())
                .and_modify(|(bh, bn)| {
                    if reuse > *bn {
                        *bh = handler.clone();
                        *bn = reuse;
                    }
                })
                .or_insert((handler.clone(), reuse));
        }
        let mut keep: Vec<String> = best.into_values().map(|(h, _)| h).collect();
        keep.sort();
        keep
    }
}
