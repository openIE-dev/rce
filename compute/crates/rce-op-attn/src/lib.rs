//! NEON-vectorized attention primitives.
//!
//! Three hot spots in every rce runtime's `attn_block`:
//!
//! 1. **Q·Kᵀ dot product** — inner loop is `for d in 0..d_head { dot += q[d] * k[d]; }`.
//!    Scalar at d_head=128 = 128 FMAs per (head, key) pair, repeated
//!    n_heads × seq_len times per token. The NEON path collapses
//!    this to `d_head/4` `vfmaq_f32`'s + one horizontal sum.
//!
//! 2. **RoPE pair rotation** — applies a 2×2 rotation to pairs
//!    `(q[i], q[i+half])`. The NEON path processes 4 pairs at a time
//!    with `vfmaq_f32` / `vfmsq_f32`. sin/cos are computed scalar
//!    (no NEON-native trig); the win is in the multiply-add.
//!
//! 3. **Softmax + V accumulation** — currently three passes: max-find,
//!    exp+sum, then weighted V-sum. The fused kernel below does one
//!    pass keeping `(max, sum, out)` updated online ("online softmax"
//!    from FlashAttention-1), which (a) cuts the score buffer to zero
//!    bytes per head and (b) halves the bandwidth pressure on the V
//!    cache.
//!
//! Scalar fallback ships for non-aarch64 (x86_64 dev machines, wasm).
//!
//! Verified bit-identical to the scalar reference on aarch64 within
//! 1e-5 relative error per element (FP-non-associative reductions
//! reorder slightly).

/// `out = Σ_d a[d] * b[d]`. Assumes `a.len() == b.len()`; panics in debug.
/// NEON path requires len % 4 == 0; falls back to scalar otherwise.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    if a.len() % 4 == 0 && a.len() >= 4 {
        // SAFETY: NEON intrinsics on aarch64; alignment and length checked.
        return unsafe { dot_neon(a, b) };
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 { unsafe {
    use std::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};
    let n = a.len();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 8 <= n {
        let av0 = vld1q_f32(a.as_ptr().add(i));
        let bv0 = vld1q_f32(b.as_ptr().add(i));
        acc0 = vfmaq_f32(acc0, av0, bv0);
        let av1 = vld1q_f32(a.as_ptr().add(i + 4));
        let bv1 = vld1q_f32(b.as_ptr().add(i + 4));
        acc1 = vfmaq_f32(acc1, av1, bv1);
        i += 8;
    }
    while i + 4 <= n {
        let av = vld1q_f32(a.as_ptr().add(i));
        let bv = vld1q_f32(b.as_ptr().add(i));
        acc0 = vfmaq_f32(acc0, av, bv);
        i += 4;
    }
    debug_assert_eq!(i, n);
    vaddvq_f32(acc0) + vaddvq_f32(acc1)
}}

/// `out[i] += scale * v[i]` for `i in 0..out.len()`. Used in the
/// V-accumulator step of attention.
#[inline]
pub fn axpy(out: &mut [f32], scale: f32, v: &[f32]) {
    debug_assert_eq!(out.len(), v.len());
    #[cfg(target_arch = "aarch64")]
    if out.len() % 4 == 0 && out.len() >= 4 {
        // SAFETY: NEON intrinsics on aarch64; len % 4 == 0 enforced.
        unsafe { axpy_neon(out, scale, v) };
        return;
    }
    for i in 0..out.len() { out[i] += scale * v[i]; }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn axpy_neon(out: &mut [f32], scale: f32, v: &[f32]) { unsafe {
    use std::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};
    let n = out.len();
    let s = vdupq_n_f32(scale);
    let mut i = 0;
    while i + 4 <= n {
        let ov = vld1q_f32(out.as_ptr().add(i));
        let vv = vld1q_f32(v.as_ptr().add(i));
        let r = vfmaq_f32(ov, s, vv);
        vst1q_f32(out.as_mut_ptr().add(i), r);
        i += 4;
    }
    debug_assert_eq!(i, n);
}}

/// Apply RoPE to `x` (a single head's worth of d_head floats), rotating
/// pairs `(x[i], x[i+half])` for `i in 0..rope_dim/2` by `theta_i =
/// pos · base^(-2i / rope_dim)`. Mirrors the canonical
/// `apply_rope_inplace` body but vectorizes the pair update.
///
/// NOTE: `rope_dim` must be even and ≤ x.len(). Dims beyond `rope_dim`
/// are untouched (NeoX/Llama convention partial-RoPE support).
#[inline]
pub fn rope_inplace(x: &mut [f32], rope_dim: usize, base: f32, pos: usize) {
    rope_inplace_scaled(x, rope_dim, base, pos, None);
}

/// RoPE variant with optional per-dim `freq_factors` (length ≥ rope_dim/2).
/// For each `i`, `inv_freq /= freq_factors[i]` when provided. This is the
/// shape Gemma 4 / Llama 3.1 / NTK-aware RoPE needs.
#[inline]
pub fn rope_inplace_scaled(
    x: &mut [f32], rope_dim: usize, base: f32, pos: usize,
    freq_factors: Option<&[f32]>,
) {
    debug_assert!(rope_dim % 2 == 0);
    debug_assert!(rope_dim <= x.len());
    let half = rope_dim / 2;
    let mut sin_buf = [0.0f32; 256];
    let mut cos_buf = [0.0f32; 256];
    debug_assert!(half <= 256, "rope_dim/2 > 256 — extend buffers");
    let log_base = base.ln();
    for i in 0..half {
        let mut inv_freq = (log_base * (2.0 * i as f32) / rope_dim as f32).exp().recip();
        if let Some(ff) = freq_factors {
            if i < ff.len() && ff[i] > 0.0 { inv_freq /= ff[i]; }
        }
        let theta = pos as f32 * inv_freq;
        let (s, c) = theta.sin_cos();
        sin_buf[i] = s;
        cos_buf[i] = c;
    }
    #[cfg(target_arch = "aarch64")]
    if half % 4 == 0 {
        unsafe { rope_neon(x, half, &sin_buf, &cos_buf) };
        return;
    }
    for i in 0..half {
        let a = x[i];
        let b = x[i + half];
        x[i]        = a * cos_buf[i] - b * sin_buf[i];
        x[i + half] = a * sin_buf[i] + b * cos_buf[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rope_neon(x: &mut [f32], half: usize, sin: &[f32], cos: &[f32]) { unsafe {
    use std::arch::aarch64::{vfmaq_f32, vfmsq_f32, vld1q_f32, vmulq_f32, vst1q_f32};
    let mut i = 0;
    while i + 4 <= half {
        let av = vld1q_f32(x.as_ptr().add(i));         // a_v
        let bv = vld1q_f32(x.as_ptr().add(i + half));  // b_v
        let cv = vld1q_f32(cos.as_ptr().add(i));
        let sv = vld1q_f32(sin.as_ptr().add(i));
        // out_a = a*c - b*s ; out_b = a*s + b*c
        let oa = vfmsq_f32(vmulq_f32(av, cv), bv, sv);
        let ob = vfmaq_f32(vmulq_f32(av, sv), bv, cv);
        vst1q_f32(x.as_mut_ptr().add(i), oa);
        vst1q_f32(x.as_mut_ptr().add(i + half), ob);
        i += 4;
    }
    debug_assert_eq!(i, half);
}}

/// **Fused attention head: scores + online softmax + V-weighted sum**,
/// in one pass over the seq dimension, never materializing the score
/// buffer. FlashAttention-1's online softmax: keep a running
/// (max, sum, out) such that the final out is identical to the
/// three-pass scalar reference up to FP reorder.
///
/// `q_h`     — query for this head, shape `[d_head]`.
/// `k_cache` — flat KV cache `[seq_len × kv_dim]` (kv_dim = n_kv × d_head).
/// `v_cache` — same shape.
/// `kv_head` — which of the n_kv heads this q-head pairs with (GQA).
/// `seq_len` — number of past+current keys.
/// `kv_dim`  — n_kv * d_head.
/// `d_head`  — per-head dim.
/// `scale`   — typically `1 / sqrt(d_head)`.
///
/// Returns the attended output for this head, `[d_head]`.
pub fn fused_attend_head(
    q_h: &[f32], k_cache: &[f32], v_cache: &[f32],
    kv_head: usize, seq_len: usize, kv_dim: usize, d_head: usize, scale: f32,
) -> Vec<f32> {
    fused_attend_head_window(q_h, k_cache, v_cache, kv_head, 0, seq_len, kv_dim, d_head, scale)
}

/// Windowed variant: only positions in `[lo, hi)` participate. Use for
/// sliding-window attention (Gemma 4 SWA layers) — set
/// `lo = pos + 1 - window` when `pos + 1 > window`, else `lo = 0`.
pub fn fused_attend_head_window(
    q_h: &[f32], k_cache: &[f32], v_cache: &[f32],
    kv_head: usize, lo: usize, hi: usize,
    kv_dim: usize, d_head: usize, scale: f32,
) -> Vec<f32> {
    debug_assert_eq!(q_h.len(), d_head);
    debug_assert!(lo <= hi);
    debug_assert!(hi * kv_dim <= k_cache.len());
    debug_assert!(hi * kv_dim <= v_cache.len());
    let head_off = kv_head * d_head;
    let mut out = vec![0.0f32; d_head];
    let mut m_running = f32::NEG_INFINITY;
    let mut l_running = 0.0f32;
    for j in lo..hi {
        let k_jh = &k_cache[j * kv_dim + head_off..j * kv_dim + head_off + d_head];
        let v_jh = &v_cache[j * kv_dim + head_off..j * kv_dim + head_off + d_head];
        let s = dot(q_h, k_jh) * scale;
        let m_new = m_running.max(s);
        // alpha = exp(m_running - m_new); β = exp(s - m_new)
        // out *= alpha; out += β * v_jh ; l = l*alpha + β
        if m_running.is_infinite() {
            // First iteration — running max was -inf; everything's β = exp(0) = 1 path.
            let beta = 1.0f32;
            axpy(&mut out, beta, v_jh);
            l_running = beta;
            m_running = s;
        } else {
            let alpha = (m_running - m_new).exp();
            let beta = (s - m_new).exp();
            // out *= alpha (in-place scale)
            for v in out.iter_mut() { *v *= alpha; }
            axpy(&mut out, beta, v_jh);
            l_running = l_running * alpha + beta;
            m_running = m_new;
        }
    }
    // Final normalization: out /= l_running.
    if l_running != 0.0 {
        let inv = 1.0 / l_running;
        for v in out.iter_mut() { *v *= inv; }
    }
    out
}

/// **Qwen-3.6 Gated Attention**: elementwise sigmoid gate on the
/// attention output (G1 form from Qwen's NeurIPS 2025 paper). Caller
/// provides pre-projected `gate_logits = W_gate @ x` of the same
/// shape as `attn_out`; this fn does in-place
/// `attn_out[i] *= sigmoid(gate_logits[i])`.
///
/// The sigmoid is computed numerically stably: `1 / (1 + exp(-x))`
/// branched to avoid overflow for very negative x.
#[inline]
pub fn apply_attention_gate(attn_out: &mut [f32], gate_logits: &[f32]) {
    debug_assert_eq!(attn_out.len(), gate_logits.len());
    for (o, g) in attn_out.iter_mut().zip(gate_logits.iter()) {
        let s = if *g >= 0.0 {
            let e = (-*g).exp();
            1.0 / (1.0 + e)
        } else {
            let e = g.exp();
            e / (1.0 + e)
        };
        *o *= s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot_ref(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn close(a: f32, b: f32, rel: f32) -> bool {
        let m = a.abs().max(b.abs()).max(1e-6);
        (a - b).abs() / m < rel
    }

    #[test]
    fn dot_matches_scalar_128() {
        let a: Vec<f32> = (0..128).map(|i| (i as f32 * 0.013).sin()).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32 * 0.017).cos()).collect();
        let r = dot(&a, &b);
        let s = dot_ref(&a, &b);
        assert!(close(r, s, 1e-5), "neon dot {r} vs scalar {s}");
    }

    #[test]
    fn axpy_matches_scalar() {
        let mut out = vec![0.5f32; 64];
        let v: Vec<f32> = (0..64).map(|i| (i as f32 * 0.025).sin()).collect();
        let mut ref_out = out.clone();
        axpy(&mut out, 0.3, &v);
        for i in 0..64 { ref_out[i] += 0.3 * v[i]; }
        for i in 0..64 {
            assert!(close(out[i], ref_out[i], 1e-5),
                    "axpy[{i}]: {} vs {}", out[i], ref_out[i]);
        }
    }

    #[test]
    fn rope_matches_scalar() {
        let d_head = 128;
        let rope_dim = 128;
        let base = 10000.0f32;
        let pos = 17usize;
        let mut x_neon: Vec<f32> = (0..d_head).map(|i| (i as f32 * 0.013).sin()).collect();
        let mut x_ref = x_neon.clone();

        rope_inplace(&mut x_neon, rope_dim, base, pos);

        // Reference: apply the scalar rotation in-place.
        let half = rope_dim / 2;
        let log_base = base.ln();
        for i in 0..half {
            let inv_freq = (log_base * (2.0 * i as f32) / rope_dim as f32).exp().recip();
            let theta = pos as f32 * inv_freq;
            let (sin, cos) = theta.sin_cos();
            let a = x_ref[i];
            let b = x_ref[i + half];
            x_ref[i]        = a * cos - b * sin;
            x_ref[i + half] = a * sin + b * cos;
        }
        for i in 0..d_head {
            assert!(close(x_neon[i], x_ref[i], 1e-5),
                    "rope[{i}]: {} vs {}", x_neon[i], x_ref[i]);
        }
    }

    #[test]
    fn apply_attention_gate_matches_reference() {
        let n = 64;
        let mut o: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin()).collect();
        let g: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).cos() * 3.0).collect();
        let mut ref_o = o.clone();
        for i in 0..n {
            ref_o[i] *= 1.0 / (1.0 + (-g[i]).exp());
        }
        apply_attention_gate(&mut o, &g);
        for i in 0..n {
            assert!((o[i] - ref_o[i]).abs() < 1e-6,
                "gate[{i}]: {} vs {}", o[i], ref_o[i]);
        }
    }

    #[test]
    fn apply_attention_gate_extreme_values_stable() {
        // Very negative + very positive gate logits should not produce NaN/Inf.
        let mut o = vec![1.0f32; 4];
        let g = vec![-1000.0f32, -50.0, 50.0, 1000.0];
        apply_attention_gate(&mut o, &g);
        assert!(o.iter().all(|v| v.is_finite()));
        // sigmoid(-inf) → 0; sigmoid(+inf) → 1.
        assert!(o[0].abs() < 1e-10, "sigmoid(-1000) should be ~0, got {}", o[0]);
        assert!((o[3] - 1.0).abs() < 1e-10, "sigmoid(1000) should be ~1, got {}", o[3]);
    }

    #[test]
    fn rope_scaled_with_freq_factors_matches_scalar() {
        // Gemma 4 full-attention layers feed `freq_factors` derived from
        // the model's `rope_freqs_full` tensor. Verify per-dim scaling.
        let d_head = 128;
        let rope_dim = 128;
        let base = 10000.0f32;
        let pos = 23usize;
        let half = rope_dim / 2;
        let ff: Vec<f32> = (0..half).map(|i| 1.0 + (i as f32) * 0.05).collect();

        let mut x_op: Vec<f32> = (0..d_head).map(|i| (i as f32 * 0.011).cos()).collect();
        let mut x_ref = x_op.clone();

        rope_inplace_scaled(&mut x_op, rope_dim, base, pos, Some(&ff));

        let log_base = base.ln();
        for i in 0..half {
            let mut inv_freq = (log_base * (2.0 * i as f32) / rope_dim as f32).exp().recip();
            if ff[i] > 0.0 { inv_freq /= ff[i]; }
            let theta = pos as f32 * inv_freq;
            let (sin, cos) = theta.sin_cos();
            let a = x_ref[i];
            let b = x_ref[i + half];
            x_ref[i]        = a * cos - b * sin;
            x_ref[i + half] = a * sin + b * cos;
        }
        for i in 0..d_head {
            assert!(close(x_op[i], x_ref[i], 1e-5),
                    "rope_scaled[{i}]: {} vs {}", x_op[i], x_ref[i]);
        }
    }

    #[test]
    fn fused_attend_window_matches_three_pass() {
        // Sliding window: only positions [lo, hi) attend.
        let d_head = 64;
        let kv_dim = d_head;
        let seq_len = 16;
        let lo = 5;
        let hi = 13;
        let q: Vec<f32> = (0..d_head).map(|i| (i as f32 * 0.019).sin()).collect();
        let k: Vec<f32> = (0..seq_len * kv_dim).map(|i| (i as f32 * 0.013).cos()).collect();
        let v: Vec<f32> = (0..seq_len * kv_dim).map(|i| (i as f32 * 0.017).sin()).collect();
        let scale = 1.0 / (d_head as f32).sqrt();

        let out_fused = fused_attend_head_window(
            &q, &k, &v, 0, lo, hi, kv_dim, d_head, scale,
        );

        let mut scores = vec![0.0f32; hi - lo];
        for (idx, j) in (lo..hi).enumerate() {
            let kj = &k[j * kv_dim..j * kv_dim + d_head];
            let mut s = 0.0f32;
            for d in 0..d_head { s += q[d] * kj[d]; }
            scores[idx] = s * scale;
        }
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() { *s = (*s - m).exp(); sum += *s; }
        let mut out_ref = vec![0.0f32; d_head];
        for (idx, j) in (lo..hi).enumerate() {
            let w = scores[idx] / sum;
            let vj = &v[j * kv_dim..j * kv_dim + d_head];
            for d in 0..d_head { out_ref[d] += w * vj[d]; }
        }
        for i in 0..d_head {
            assert!(close(out_fused[i], out_ref[i], 1e-4),
                    "fused_window[{i}]: {} vs {}", out_fused[i], out_ref[i]);
        }
    }

    #[test]
    fn fused_attend_matches_three_pass() {
        // d_head=64, n_kv=1, seq_len=12.
        let d_head = 64;
        let kv_dim = d_head;
        let seq_len = 12;
        let q: Vec<f32> = (0..d_head).map(|i| (i as f32 * 0.013).sin()).collect();
        let k: Vec<f32> = (0..seq_len * kv_dim).map(|i| (i as f32 * 0.017).cos()).collect();
        let v: Vec<f32> = (0..seq_len * kv_dim).map(|i| (i as f32 * 0.011).sin()).collect();
        let scale = 1.0 / (d_head as f32).sqrt();

        let out_fused = fused_attend_head(&q, &k, &v, 0, seq_len, kv_dim, d_head, scale);

        // Reference: three-pass scalar.
        let mut scores = vec![0.0f32; seq_len];
        for j in 0..seq_len {
            let kj = &k[j * kv_dim..j * kv_dim + d_head];
            let mut s = 0.0f32;
            for d in 0..d_head { s += q[d] * kj[d]; }
            scores[j] = s * scale;
        }
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() { *s = (*s - m).exp(); sum += *s; }
        let mut out_ref = vec![0.0f32; d_head];
        for j in 0..seq_len {
            let w = scores[j] / sum;
            let vj = &v[j * kv_dim..j * kv_dim + d_head];
            for d in 0..d_head { out_ref[d] += w * vj[d]; }
        }

        for i in 0..d_head {
            assert!(close(out_fused[i], out_ref[i], 1e-4),
                    "fused[{i}]: {} vs {}", out_fused[i], out_ref[i]);
        }
    }
}
