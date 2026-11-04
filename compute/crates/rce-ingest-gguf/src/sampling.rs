//! Token sampling for LLM decoding.
//!
//! Replaces greedy argmax with the standard
//! **temperature + top-K + top-P** pipeline. Greedy decoding on
//! heavily-quantized weights (Q4_K_M) tends to lock onto degenerate
//! trajectories because quantization noise can flip the argmax by tiny
//! margins; sampling spreads probability mass and lets the model
//! recover.
//!
//! Pipeline (applied in order):
//! 1. **Temperature** — divide logits by `temperature` before softmax.
//!    `T = 0` collapses to greedy argmax (handled as a special case).
//! 2. **Top-K** — keep only the K highest-probability tokens, zero out
//!    the rest.
//! 3. **Top-P (nucleus)** — sort the remaining tokens by probability
//!    descending, take the smallest prefix whose cumulative
//!    probability ≥ top_p, zero out the rest.
//! 4. **Renormalize + multinomial draw** — use a seeded PCG-style RNG
//!    so tests are reproducible.

/// Sampling parameters. Sensible defaults for chat-style generation:
/// `temperature = 0.7, top_k = Some(40), top_p = Some(0.9)`.
#[derive(Debug, Clone, Copy)]
pub struct SampleConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    /// Multiplicative penalty on the logits of tokens that appear in
    /// the `recent` history passed to `sample`. `1.0` disables; common
    /// values are `1.05` – `1.2`. Positive logits get divided by the
    /// penalty (made smaller); negative logits get multiplied (made
    /// more negative). Matches llama.cpp's `--repeat-penalty`.
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl SampleConfig {
    pub const GREEDY: Self = Self {
        temperature: 0.0,
        top_k: None,
        top_p: None,
        repetition_penalty: 1.0,
        seed: 0,
    };

    pub fn chat(seed: u64) -> Self {
        Self {
            temperature: 0.7,
            top_k: Some(40),
            top_p: Some(0.9),
            repetition_penalty: 1.15,
            seed,
        }
    }
}

/// Sample a token id from `logits`. Convenience wrapper for the no-
/// history case (no repetition penalty even if configured).
pub fn sample(logits: &[f32], cfg: &SampleConfig) -> i32 {
    sample_with_history(logits, cfg, &[])
}

/// Sample a token id from `logits` with optional repetition penalty
/// applied against `recent` (most recent tokens, e.g. the last
/// 64 tokens of the generated sequence). Returns the argmax when
/// `temperature == 0` (repetition penalty still applied first so
/// even greedy decoding can break out of `\n\n\n…` loops).
pub fn sample_with_history(logits: &[f32], cfg: &SampleConfig, recent: &[i32]) -> i32 {
    if logits.is_empty() {
        return 0;
    }

    // 0. Repetition penalty: scale down (positive) or amplify-negative
    //    the logits of any token in `recent`. Applied before
    //    temperature so the relative effect is consistent across T.
    let mut working: Vec<f32> = logits.to_vec();
    if cfg.repetition_penalty != 1.0 && !recent.is_empty() {
        let p = cfg.repetition_penalty;
        for &tok in recent {
            if tok >= 0 {
                let idx = tok as usize;
                if idx < working.len() {
                    let v = working[idx];
                    working[idx] = if v > 0.0 { v / p } else { v * p };
                }
            }
        }
    }

    if cfg.temperature == 0.0 {
        return argmax(&working);
    }

    // 1. Temperature scaling.
    let inv_t = 1.0 / cfg.temperature;
    let scaled: Vec<f32> = working.iter().map(|&x| x * inv_t).collect();

    // 2. Numerically-stable softmax (subtract max before exp).
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = scaled.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return argmax(&working);
    }
    for p in &mut probs {
        *p /= sum;
    }
    drop(scaled);

    // Build an `(id, prob)` table sorted by prob descending for
    // top-K and top-P operations.
    let mut indexed: Vec<(i32, f32)> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| (i as i32, p))
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // 3. Top-K.
    if let Some(k) = cfg.top_k {
        if k < indexed.len() {
            indexed.truncate(k);
        }
    }

    // 4. Top-P (nucleus).
    if let Some(p_thresh) = cfg.top_p {
        let mut cum = 0.0;
        let mut cutoff = indexed.len();
        for (i, &(_, p)) in indexed.iter().enumerate() {
            cum += p;
            if cum >= p_thresh {
                cutoff = i + 1;
                break;
            }
        }
        indexed.truncate(cutoff);
    }

    // 5. Renormalize within the kept set.
    let kept_sum: f32 = indexed.iter().map(|&(_, p)| p).sum();
    if !(kept_sum > 0.0) || !kept_sum.is_finite() {
        return indexed.first().map(|&(id, _)| id).unwrap_or(0);
    }
    for entry in &mut indexed {
        entry.1 /= kept_sum;
    }

    // 6. Multinomial draw with a seeded RNG.
    let mut rng = Pcg32::new(cfg.seed);
    let u = rng.next_f32();
    let mut acc = 0.0;
    for &(id, p) in &indexed {
        acc += p;
        if u < acc {
            return id;
        }
    }
    // Floating-point slop: return the last kept token.
    indexed.last().map(|&(id, _)| id).unwrap_or(0)
}

fn argmax(logits: &[f32]) -> i32 {
    let mut best_i = 0i32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i as i32;
        }
    }
    best_i
}

/// Minimal PCG-XSH-RR 32-bit RNG. Deterministic given the seed, fast,
/// good-enough statistical quality for multinomial draws over a vocab.
struct Pcg32 {
    state: u64,
}

impl Pcg32 {
    fn new(seed: u64) -> Self {
        // Mix the seed so trivial seeds like `0` still produce a
        // reasonable initial state.
        let state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        Self { state }
    }

    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    fn next_f32(&mut self) -> f32 {
        // 24-bit mantissa; uniform in [0, 1).
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_returns_argmax() {
        let logits = [1.0, 5.0, 3.0, 2.0];
        assert_eq!(sample(&logits, &SampleConfig::GREEDY), 1);
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let logits = [1.0, 5.0, 3.0, 2.0];
        let cfg = SampleConfig { temperature: 0.0, top_k: Some(2), top_p: Some(0.5), repetition_penalty: 1.0, seed: 42 };
        assert_eq!(sample(&logits, &cfg), 1);
    }

    #[test]
    fn top_k_restricts_to_k_choices() {
        // Logits with one dominant token; top-K=1 must pick it
        // regardless of seed.
        let logits = [0.0, 10.0, 0.1, 0.2];
        let cfg = SampleConfig { temperature: 1.0, top_k: Some(1), top_p: None, repetition_penalty: 1.0, seed: 7 };
        for seed in 0..16 {
            let mut c = cfg;
            c.seed = seed;
            assert_eq!(sample(&logits, &c), 1, "seed {seed}");
        }
    }

    #[test]
    fn seeded_draws_are_reproducible() {
        // Same seed → same token; different seeds → distribution.
        let logits = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // uniform
        let cfg = SampleConfig { temperature: 1.0, top_k: None, top_p: None, repetition_penalty: 1.0, seed: 12345 };
        let a = sample(&logits, &cfg);
        let b = sample(&logits, &cfg);
        assert_eq!(a, b, "same seed must give same token");
        let c = sample(&logits, &SampleConfig { seed: 99999, ..cfg });
        // With 8 uniform options, two different seeds collide only ~12% of
        // the time. If they happen to collide here, the test is still
        // valid — what we really care about is determinism + range.
        assert!((0..8).contains(&a));
        assert!((0..8).contains(&c));
    }

    #[test]
    fn repetition_penalty_pushes_repeated_tokens_down() {
        // Without history, token 1 is the argmax (logit 5).
        let logits = [1.0, 5.0, 4.5, 0.5];
        let cfg = SampleConfig { temperature: 0.0, top_k: None, top_p: None, repetition_penalty: 1.2, seed: 0 };
        assert_eq!(sample_with_history(&logits, &cfg, &[]), 1);
        // With token 1 in history and penalty 1.2: 5.0 / 1.2 ≈ 4.17,
        // so token 2 (4.5) becomes the new argmax.
        assert_eq!(sample_with_history(&logits, &cfg, &[1]), 2);
        // History with negative-logit token: penalty makes it more
        // negative, doesn't promote it.
        let logits = [-1.0, 5.0, 4.5];
        assert_eq!(sample_with_history(&logits, &cfg, &[0]), 1);
    }

    #[test]
    fn top_p_keeps_cumulative_mass() {
        // Two tokens carry ~95% of mass; top_p = 0.9 should keep at
        // most those two and pick one of them.
        let logits = [5.0, 5.0, -10.0, -10.0, -10.0];
        let cfg = SampleConfig { temperature: 1.0, top_k: None, top_p: Some(0.9), repetition_penalty: 1.0, seed: 1 };
        for seed in 0..32 {
            let mut c = cfg;
            c.seed = seed;
            let id = sample(&logits, &c);
            assert!(id == 0 || id == 1, "seed {seed} produced {id}");
        }
    }
}
