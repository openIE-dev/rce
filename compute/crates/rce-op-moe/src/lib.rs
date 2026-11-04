//! MoE router + dispatch primitives.
//!
//! For each token, the router projects activation `x` (`[d_model]`)
//! through gate weight (`[n_experts, d_model]`) to produce logits over
//! experts. Softmax → pick top-K → renormalize the selected weights so
//! they sum to 1. The caller iterates the top-K experts, runs each
//! expert's FFN (gate/up/down or whatever the model's expert structure
//! is), and combines outputs by the renormalized weights.
//!
//! This keeps the MoE op generic — it doesn't know about SwiGLU /
//! GeGLU / which projection is "gate" — and stays decode-shape
//! optimal: only the K selected experts execute, not all N.
//!
//! Conventions (ZAYA1-8B, Qwen 3.6-35B-A3B, DeepSeek V4):
//! - `top_k = 2` for ZAYA1 and Qwen 3.6
//! - `top_k = 8` for DeepSeek V4
//! - `normalize = true` after top-K selection (the routed weights sum
//!   to 1 over the K chosen experts, not 1 over all N)

use rce_op_qmm::QuantWeight;

/// What the router decided for one token.
#[derive(Debug, Clone)]
pub struct RouterDecision {
    /// Indices of the top-K experts (length = `top_k`).
    pub expert_indices: Vec<usize>,
    /// Routing weights (length = `top_k`). When `normalize=true`,
    /// sum to 1. Otherwise these are the raw softmax probabilities of
    /// the selected experts (may sum to < 1).
    pub expert_weights: Vec<f32>,
}

/// Project activation through router gate, softmax over all experts,
/// pick top-K, optionally renormalize selected weights to sum 1.
pub fn moe_router(
    x: &[f32],
    gate_w: &QuantWeight,
    top_k: usize,
    normalize: bool,
) -> RouterDecision {
    debug_assert_eq!(x.len(), gate_w.n_cols, "router input dim mismatch");
    let n_experts = gate_w.n_rows;
    debug_assert!(top_k > 0 && top_k <= n_experts);

    // Step 1: project to logits [n_experts].
    let mut logits = vec![0.0f32; n_experts];
    rce_op_qmm::mm_1d(x, gate_w, &mut logits);

    // Step 2: softmax over all experts. Subtract max for numerical
    // stability; the exp() + sum is over n_experts (typically 8-128).
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 { for p in probs.iter_mut() { *p /= sum; } }

    // Step 3: top-K selection via partial sort. For small top_k (2-8)
    // and small n_experts (8-128) this is fine; for very large
    // n_experts a heap-based path would be better.
    let mut idx: Vec<usize> = (0..n_experts).collect();
    idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    idx.truncate(top_k);
    let mut weights: Vec<f32> = idx.iter().map(|&i| probs[i]).collect();

    if normalize {
        let s: f32 = weights.iter().sum();
        if s > 0.0 { for w in weights.iter_mut() { *w /= s; } }
    }

    RouterDecision {
        expert_indices: idx,
        expert_weights: weights,
    }
}

/// Greedy / argmax dispatch — top-1 special case. Equivalent to
/// `moe_router(x, gate_w, 1, true)` but skips the full sort.
/// Useful for Switch Transformer-style routing.
pub fn moe_router_top1(x: &[f32], gate_w: &QuantWeight) -> RouterDecision {
    debug_assert_eq!(x.len(), gate_w.n_cols);
    let n_experts = gate_w.n_rows;
    let mut logits = vec![0.0f32; n_experts];
    rce_op_qmm::mm_1d(x, gate_w, &mut logits);

    let (best, _) = logits.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    RouterDecision {
        expert_indices: vec![best],
        expert_weights: vec![1.0],
    }
}

/// Combine `top_k` expert outputs into the final token output using
/// the router's weights. `expert_outputs[i]` is the FFN output of the
/// `i`-th selected expert (caller-computed); length must equal
/// `decision.expert_indices.len()`.
pub fn moe_combine(
    decision: &RouterDecision,
    expert_outputs: &[Vec<f32>],
    d_model: usize,
) -> Vec<f32> {
    debug_assert_eq!(expert_outputs.len(), decision.expert_indices.len());
    debug_assert!(expert_outputs.iter().all(|o| o.len() == d_model));
    let mut out = vec![0.0f32; d_model];
    for (w, exp) in decision.expert_weights.iter().zip(expert_outputs.iter()) {
        for i in 0..d_model { out[i] += w * exp[i]; }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rce_ingest_gguf::GgufDtype;

    /// Build a synthetic Q8_0 gate matrix.
    fn synth_gate(n_experts: usize, d_model: usize) -> QuantWeight {
        assert!(d_model % 32 == 0);
        let blocks_per_row = d_model / 32;
        let bytes_per_row = blocks_per_row * 34;
        let mut bytes = vec![0u8; n_experts * bytes_per_row];
        for r in 0..n_experts {
            for blk in 0..blocks_per_row {
                let off = r * bytes_per_row + blk * 34;
                let scale = 0.0625f32 + (r as f32) * 0.01;
                let s = half::f16::from_f32(scale).to_le_bytes();
                bytes[off] = s[0];
                bytes[off + 1] = s[1];
                for i in 0..32 {
                    // Each row biased toward a different "favorite" position.
                    let v = if r == (blk * 32 + i) % n_experts { 64i8 } else { 0i8 };
                    bytes[off + 2 + i] = v as u8;
                }
            }
        }
        QuantWeight::from_owned_bytes(bytes, GgufDtype::Q8_0, n_experts, d_model)
    }

    #[test]
    fn router_returns_top_k_sorted() {
        let n_experts = 8;
        let d_model = 64;
        let gate = synth_gate(n_experts, d_model);
        let x: Vec<f32> = (0..d_model).map(|i| (i as f32 * 0.013).sin()).collect();

        let dec = moe_router(&x, &gate, 2, true);
        assert_eq!(dec.expert_indices.len(), 2);
        assert_eq!(dec.expert_weights.len(), 2);
        // Weights normalized.
        let s: f32 = dec.expert_weights.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "weights sum to {s}");
        // Weights sorted descending.
        assert!(dec.expert_weights[0] >= dec.expert_weights[1]);
        // Both weights in [0, 1].
        assert!(dec.expert_weights.iter().all(|w| (0.0..=1.0).contains(w)));
        // Indices distinct.
        assert_ne!(dec.expert_indices[0], dec.expert_indices[1]);
    }

    #[test]
    fn router_top1_matches_top_k_with_k_eq_1() {
        let n_experts = 16;
        let d_model = 64;
        let gate = synth_gate(n_experts, d_model);
        let x: Vec<f32> = (0..d_model).map(|i| (i as f32 * 0.017).cos()).collect();

        let dec_k1 = moe_router(&x, &gate, 1, true);
        let dec_top1 = moe_router_top1(&x, &gate);
        assert_eq!(dec_k1.expert_indices, dec_top1.expert_indices);
        assert!((dec_k1.expert_weights[0] - dec_top1.expert_weights[0]).abs() < 1e-6);
    }

    #[test]
    fn combine_weights_correctly() {
        let d_model = 32;
        let decision = RouterDecision {
            expert_indices: vec![2, 5, 1],
            expert_weights: vec![0.5, 0.3, 0.2],
        };
        let outputs: Vec<Vec<f32>> = vec![
            vec![1.0; d_model],
            vec![2.0; d_model],
            vec![3.0; d_model],
        ];
        let combined = moe_combine(&decision, &outputs, d_model);
        // 0.5*1 + 0.3*2 + 0.2*3 = 0.5 + 0.6 + 0.6 = 1.7
        for v in &combined {
            assert!((v - 1.7).abs() < 1e-5, "got {v}, expected 1.7");
        }
    }

    #[test]
    fn unnormalized_weights_can_sum_below_one() {
        let n_experts = 32;
        let d_model = 64;
        let gate = synth_gate(n_experts, d_model);
        let x: Vec<f32> = (0..d_model).map(|i| (i as f32 * 0.013).sin()).collect();

        let dec = moe_router(&x, &gate, 2, false);
        let s: f32 = dec.expert_weights.iter().sum();
        // With 32 experts and top-2, sum of two softmax probs is < 1.
        assert!(s > 0.0 && s < 1.0, "raw top-K sum should be < 1, got {s}");
    }
}
