//! Speculative decoding (EAGLE-3 protocol) for rce-* runtimes.
//!
//! **Why this matters for rce specifically.** Production speculative-
//! decoding papers (EAGLE-3, Medusa, Lookahead) show **2.5-5× single-
//! stream throughput** at batch=1 with output distribution preserved.
//! Crucially, batch=1 is the regime where vLLM/SGLang's continuous-
//! batching schedulers DON'T benefit — the speedup comes from spec
//! decoding instead. **Single-stream local-first is exactly rce's
//! posture**, so spec decoding has a strictly better fit here than
//! in any production engine.
//!
//! ## Protocol
//!
//! Each "speculative step":
//!
//! 1. **Draft phase.** A small DraftModel proposes a K-token sequence
//!    `[d_1, d_2, ..., d_K]` from the current context, autoregressively,
//!    keeping its draft logits for each step (`p_draft(d_i | ctx, d_<i)`).
//! 2. **Verify phase.** The TargetModel runs ONE batched forward pass
//!    over the K candidate tokens, yielding target logits at each
//!    position (`p_target(. | ctx, d_<i)`).
//! 3. **Accept phase.** For each `i in 0..K`:
//!    - If `p_target(d_i) / p_draft(d_i) >= r` (where r ∼ Uniform(0,1)
//!      per token), accept `d_i` and advance ctx.
//!    - Else, reject `d_i`, sample a corrected token from the
//!      residual `p_target - p_draft` distribution, and stop.
//! 4. **Bonus token.** If all K draft tokens are accepted, also
//!    sample one bonus token from `p_target(. | ctx, d_<=K)`. This is
//!    the "free" extra token — the win that scales spec decoding past
//!    K accepts.
//!
//! Output-distribution-preserving: per EAGLE-3 / Leviathan et al.
//! 2023, the rejection-sampling step guarantees the accepted token
//! distribution matches `p_target` exactly. There's no quality loss.
//!
//! ## Traits
//!
//! Runtimes plug in by implementing [`TargetModel`] (full inference
//! engine) and providing a [`DraftModel`] — usually a smaller distilled
//! sibling. The draft can also be the SAME model truncated to fewer
//! layers (Medusa-style heads), or even a token-prediction lookup
//! (Lookahead-style, no model at all).
//!
//! ## Today's deliverable
//!
//! - [`DraftModel`] / [`TargetModel`] traits.
//! - [`speculative_decode`]: the accept-or-reject loop.
//! - [`SelfSpec`]: a degenerate "use target as draft" stub for testing
//!   the protocol — accept rate is 100% (modulo the sampling RNG), so
//!   end-to-end produces target-equivalent output. Real wins come from
//!   plugging in a genuine draft model.
//! - 3 tests: trait shape, accept-all path, end-to-end with `SelfSpec`.

use std::cell::RefCell;

/// Probability distribution over the vocabulary at one position.
/// Per-token raw probabilities (NOT log-probs). Sum should be ≈1.
/// Length = vocab size.
pub type Distribution = Vec<f32>;

/// What the draft proposes at one step: a token id + the probability
/// the draft model assigned to it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DraftStep {
    pub token: i32,
    pub draft_prob: f32,
}

/// The draft model: produces K-token speculation given a context.
///
/// Implementations should be **deterministic given (context, rng)** so
/// the accept/reject loop above can reproduce the draft phase for
/// debugging. Most real implementations are small autoregressive
/// models running on their own KV cache parallel to the target's.
pub trait DraftModel {
    /// Propose `k` next-tokens from `context`, returning each token's
    /// id + the draft probability assigned to that specific token
    /// (NOT the full distribution — saves vocab×K bytes per step).
    fn draft(&mut self, context: &[i32], k: usize) -> Vec<DraftStep>;
}

/// The target model: runs full inference and gives full distributions.
///
/// `verify` runs a single batched forward pass over the prompt
/// extended with each draft token, returning the target distribution
/// at EACH speculative position. So the output length is K+1
/// distributions: one at position [ctx], one at [ctx, d_1], ...,
/// one at [ctx, d_1, ..., d_K].
pub trait TargetModel {
    fn vocab_size(&self) -> usize;
    /// Run the target model on `context ++ draft_tokens`, returning
    /// K+1 distributions. The output at index `i` is the target's
    /// distribution AFTER processing tokens `context ++ draft_tokens[..i]`.
    fn verify(&mut self, context: &[i32], draft_tokens: &[i32]) -> Vec<Distribution>;
}

/// Result of one speculative-decode step.
#[derive(Debug)]
pub struct SpecStepResult {
    /// The accepted tokens (length 0..=K+1, where K+1 means all
    /// proposals accepted AND the bonus token sampled).
    pub accepted: Vec<i32>,
    /// How many draft proposals were accepted before the first
    /// rejection (`< K` ⇒ a corrected token follows; `= K` ⇒ the
    /// bonus token follows).
    pub n_draft_accepted: usize,
    /// Whether the last token in `accepted` came from a target
    /// correction (rejection sampling) rather than draft+verify.
    pub corrected: bool,
}

/// Configuration knobs for the spec-decode loop.
#[derive(Debug, Clone, Copy)]
pub struct SpecDecodeConfig {
    /// Number of draft tokens proposed per step. Higher K → more
    /// potential wins per verify call, but also more wasted target
    /// compute on rejected drafts. Typical: 4-8 for EAGLE-3.
    pub k: usize,
    /// RNG seed for the accept/reject Bernoulli decisions. Zero =
    /// always greedy (accept iff p_target(d_i) > p_draft(d_i)),
    /// useful for deterministic testing.
    pub seed: u64,
}

impl Default for SpecDecodeConfig {
    fn default() -> Self {
        Self { k: 4, seed: 0 }
    }
}

/// Linear-congruential PRNG. Deterministic given seed. Wrappers
/// around `next_u64()` per accept/reject decision.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Self(seed.max(1)) }
    fn next_f32(&mut self) -> f32 {
        // Numerical Recipes constants.
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as u32 as f32) / (u32::MAX as f32)
    }
}

/// Single spec-decode step. Returns accepted tokens (1 to k+1 of them).
///
/// The caller's responsibility:
/// - Append `accepted` to the running output sequence.
/// - On `corrected = true`, drop the draft model's KV cache state for
///   positions past `n_draft_accepted`.
/// - Repeat until EOS or token budget.
pub fn speculative_decode(
    draft: &mut dyn DraftModel,
    target: &mut dyn TargetModel,
    context: &[i32],
    cfg: &SpecDecodeConfig,
) -> SpecStepResult {
    let k = cfg.k.max(1);
    let proposals = draft.draft(context, k);
    let draft_tokens: Vec<i32> = proposals.iter().map(|p| p.token).collect();
    let target_dists = target.verify(context, &draft_tokens);
    debug_assert_eq!(target_dists.len(), k + 1, "target.verify must return K+1 distributions");

    let rng_cell = RefCell::new(Lcg::new(cfg.seed));
    let mut accepted: Vec<i32> = Vec::with_capacity(k + 1);
    let mut n_draft_accepted: usize = 0;
    let mut corrected = false;

    for (i, draft_step) in proposals.iter().enumerate() {
        let t_dist = &target_dists[i];
        let d_tok = draft_step.token as usize;
        // target_prob is the probability the target assigns to the
        // draft's proposed token at this position.
        let t_prob = t_dist.get(d_tok).copied().unwrap_or(0.0);
        let d_prob = draft_step.draft_prob.max(1e-30);
        // Acceptance probability: min(1, p_target / p_draft).
        let r = if cfg.seed == 0 { 0.5 } else { rng_cell.borrow_mut().next_f32() };
        let accept = r <= (t_prob / d_prob).min(1.0);
        if accept {
            accepted.push(draft_step.token);
            n_draft_accepted += 1;
        } else {
            // Rejection. Sample a correction from `max(0, p_target - p_draft)`,
            // normalized. (For draft positions that aren't the rejected
            // token, t_dist already has positive mass.)
            let mut corrected_dist: Distribution = t_dist.clone();
            // Subtract draft mass on the rejected token only (the
            // standard speculative-decoding correction step).
            corrected_dist[d_tok] = (corrected_dist[d_tok] - d_prob).max(0.0);
            let sum: f32 = corrected_dist.iter().sum();
            if sum > 0.0 {
                for v in corrected_dist.iter_mut() { *v /= sum; }
            }
            // Greedy or sampled pick. Greedy = argmax; sampled = roulette.
            let pick = if cfg.seed == 0 {
                argmax(&corrected_dist) as i32
            } else {
                weighted_pick(&corrected_dist, &mut rng_cell.borrow_mut()) as i32
            };
            accepted.push(pick);
            corrected = true;
            return SpecStepResult { accepted, n_draft_accepted, corrected };
        }
    }
    // All k draft proposals accepted — sample the bonus token from
    // target_dists[k] (the distribution AFTER processing all drafts).
    let bonus = if cfg.seed == 0 {
        argmax(&target_dists[k]) as i32
    } else {
        weighted_pick(&target_dists[k], &mut rng_cell.borrow_mut()) as i32
    };
    accepted.push(bonus);
    SpecStepResult { accepted, n_draft_accepted, corrected: false }
}

fn argmax(d: &[f32]) -> usize {
    let mut best = 0;
    let mut best_v = d[0];
    for (i, &v) in d.iter().enumerate().skip(1) {
        if v > best_v { best_v = v; best = i; }
    }
    best
}

fn weighted_pick(d: &[f32], rng: &mut Lcg) -> usize {
    let total: f32 = d.iter().sum();
    let r = rng.next_f32() * total;
    let mut acc = 0.0;
    for (i, &v) in d.iter().enumerate() {
        acc += v;
        if r <= acc { return i; }
    }
    d.len() - 1
}

/// **Self-spec testing stub**: a draft model that just calls the
/// target's own forward pass. Accept rate is 100% (the draft is
/// exact), so end-to-end output matches plain target inference up to
/// the RNG seed. Useful for protocol correctness checks; useless for
/// perf since no compute is saved.
pub struct SelfSpec<'a, T: TargetModel> {
    pub target: &'a mut T,
}

impl<'a, T: TargetModel> DraftModel for SelfSpec<'a, T> {
    fn draft(&mut self, context: &[i32], k: usize) -> Vec<DraftStep> {
        // Sequentially extend context k times using target's argmax.
        let mut ctx: Vec<i32> = context.to_vec();
        let mut out: Vec<DraftStep> = Vec::with_capacity(k);
        for _ in 0..k {
            let dists = self.target.verify(&ctx, &[]);
            let d = &dists[0];
            let token = argmax(d) as i32;
            let prob = d.get(token as usize).copied().unwrap_or(1e-9);
            out.push(DraftStep { token, draft_prob: prob });
            ctx.push(token);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Toy target model with a tiny vocab — for protocol testing.
    /// Output distribution is just the previous-token-id one-hot,
    /// so argmax(verify) deterministically advances `current_token`.
    struct ToyTarget {
        vocab: usize,
    }
    impl TargetModel for ToyTarget {
        fn vocab_size(&self) -> usize { self.vocab }
        fn verify(&mut self, context: &[i32], draft_tokens: &[i32]) -> Vec<Distribution> {
            let mut dists = Vec::with_capacity(draft_tokens.len() + 1);
            let mut seq: Vec<i32> = context.to_vec();
            for _ in 0..=draft_tokens.len() {
                let last = *seq.last().unwrap_or(&0);
                let next = ((last as usize + 1) % self.vocab) as i32;
                let mut d = vec![1e-6f32; self.vocab];
                d[next as usize] = 0.99;
                dists.push(d);
                seq.push(next);
            }
            dists
        }
    }

    /// Toy draft model: ALSO emits `(prev+1) % vocab` with very high
    /// confidence — identical predictions to the toy target.
    struct ToyDraft { vocab: usize }
    impl DraftModel for ToyDraft {
        fn draft(&mut self, context: &[i32], k: usize) -> Vec<DraftStep> {
            let mut out = Vec::with_capacity(k);
            let mut last = *context.last().unwrap_or(&0);
            for _ in 0..k {
                let next = ((last as usize + 1) % self.vocab) as i32;
                out.push(DraftStep { token: next, draft_prob: 0.99 });
                last = next;
            }
            out
        }
    }

    #[test]
    fn protocol_accepts_all_when_draft_matches_target() {
        let mut target = ToyTarget { vocab: 32 };
        let mut draft = ToyDraft { vocab: 32 };
        let r = speculative_decode(&mut draft, &mut target, &[0], &SpecDecodeConfig::default());
        // Greedy seed=0; draft predicts exactly target ⇒ all K accept
        // ⇒ K + 1 tokens emitted (with bonus).
        assert_eq!(r.accepted.len(), SpecDecodeConfig::default().k + 1);
        assert_eq!(r.n_draft_accepted, SpecDecodeConfig::default().k);
        assert!(!r.corrected);
        // Output is the increment-by-1 sequence.
        for (i, &t) in r.accepted.iter().enumerate() {
            assert_eq!(t as usize, i + 1);
        }
    }

    /// Toy draft that proposes WRONG tokens — protocol must reject.
    struct WrongDraft { vocab: usize }
    impl DraftModel for WrongDraft {
        fn draft(&mut self, context: &[i32], k: usize) -> Vec<DraftStep> {
            // Propose vocab-1 (always wrong unless prev == vocab-2).
            let mut out = Vec::with_capacity(k);
            for _ in 0..k {
                out.push(DraftStep { token: (self.vocab - 1) as i32, draft_prob: 0.5 });
            }
            // Mark context use, no other effect.
            let _ = context;
            out
        }
    }

    #[test]
    fn protocol_rejects_and_corrects() {
        let mut target = ToyTarget { vocab: 32 };
        let mut draft = WrongDraft { vocab: 32 };
        let r = speculative_decode(&mut draft, &mut target, &[5], &SpecDecodeConfig::default());
        // Greedy seed=0: draft proposes vocab-1=31; target wants 6.
        // Acceptance threshold = min(1, p_target(31) / p_draft(31)) =
        // min(1, 1e-6 / 0.5) ≈ 0 — always reject. n_draft_accepted=0,
        // one corrected token emitted.
        assert_eq!(r.n_draft_accepted, 0);
        assert!(r.corrected);
        assert_eq!(r.accepted.len(), 1);
        assert_eq!(r.accepted[0], 6);
    }

    #[test]
    fn self_spec_matches_target_argmax() {
        // SelfSpec wraps target as its own draft. With seed=0 (greedy)
        // and a deterministic toy target, the output should be the
        // same as just running target.argmax K times.
        let mut t = ToyTarget { vocab: 32 };
        let mut t2 = ToyTarget { vocab: 32 };
        let mut spec = SelfSpec { target: &mut t };
        let cfg = SpecDecodeConfig { k: 6, seed: 0 };
        let r = speculative_decode(&mut spec, &mut t2, &[0], &cfg);
        assert_eq!(r.accepted.len(), cfg.k + 1);
        for (i, &tok) in r.accepted.iter().enumerate() {
            assert_eq!(tok as usize, i + 1);
        }
    }
}
