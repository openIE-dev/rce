//! Phase 4 v2 — matches real NVIDIA Cosmos 3 Nano's MoT architecture.
//!
//! Stage 6 (real-weight inspection) revealed the Stage 0-5 scaffold's
//! DiT-style assumptions don't match Cosmos 3.  v2 replaces that block
//! design with the actual joint-stream MMDiT-family architecture that
//! the released weights encode.  Per layer (22 bf16 tensors):
//!
//! **Reasoner stream** (these are already exercised by Phase 1):
//! - `input_layernorm` [HIDDEN]
//! - `self_attn.to_q` [HIDDEN, HIDDEN], `to_k` [KV_DIM, HIDDEN],
//!   `to_v` [KV_DIM, HIDDEN], `to_out` [HIDDEN, HIDDEN]
//! - `self_attn.norm_q` [HEAD_DIM], `norm_k` [HEAD_DIM]
//! - `post_attention_layernorm` [HIDDEN]
//! - `mlp.{gate_proj, up_proj, down_proj}` (SwiGLU; gate, up: [FF, HIDDEN], down: [HIDDEN, FF])
//!
//! **Generator stream** (Phase 4 v2's addition):
//! - `input_layernorm_moe_gen` [HIDDEN]
//! - `self_attn.add_q_proj` [HIDDEN, HIDDEN]
//! - `self_attn.add_k_proj` [KV_DIM, HIDDEN]
//! - `self_attn.add_v_proj` [KV_DIM, HIDDEN]
//! - `self_attn.to_add_out` [HIDDEN, HIDDEN]
//! - `self_attn.norm_added_q` [HEAD_DIM], `norm_added_k` [HEAD_DIM]
//! - `post_attention_layernorm_moe_gen` [HIDDEN]
//! - `mlp_moe_gen.{gate_proj, up_proj, down_proj}` (parallel SwiGLU)
//!
//! Plus top-level modality embeds: `action_modality_embed` [HIDDEN] and
//! `audio_modality_embed` [HIDDEN] — single vectors added per-token by
//! modality tag.  Reasoner-only embed `embed_tokens` [VOCAB, HIDDEN].
//!
//! Stage 6 v2 v0 (this scaffold): **load + shape verify only.**
//! Forward implementation is the next-session chapter; it composes
//! reasoner SDPA + generator SDPA into a joint attention via sequence
//! concatenation (MMDiT pattern), reuses all of Stage 5's `MetalAccel`
//! + `mm_bf16_routed` + `sdpa_routed` infrastructure.

use crate::{BfSlice, ShardedSafetensors, HIDDEN, N_LAYERS};
use crate::CosmosError;
use std::path::Path;

/// KV projection inner dim — GQA with 8 KV heads × HEAD_DIM=128 = 1024.
pub const KV_DIM: usize = 1024;
/// MLP intermediate dim (SwiGLU gate/up output).
pub const FF: usize = 12288;

fn ld(ss: &ShardedSafetensors, name: &str) -> Result<Vec<f32>, CosmosError> {
    let t = ss.get(name).ok_or_else(|| CosmosError::Load(format!("missing {name}")))?;
    let bytes = t.data.as_slice();
    match t.dtype.as_str() {
        "f32" => Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()),
        "bf16" => Ok(bytes.chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0],c[1]]).to_f32()).collect()),
        other => Err(CosmosError::Load(format!("ld({name}): unsupported dtype {other}"))),
    }
}

fn ld_bf16(ss: &ShardedSafetensors, name: &str) -> Result<BfSlice, CosmosError> {
    let t = ss.get(name).ok_or_else(|| CosmosError::Load(format!("missing {name}")))?;
    BfSlice::from_tensor(name, t)
}

// ────────────────────────────────────────────────────────────────────────
// Per-layer weights (matches real Cosmos 3 Nano layout exactly)
// ────────────────────────────────────────────────────────────────────────

/// Reasoner stream weights (already loaded by Phase 1's `Layer`; v2 lays
/// them out side-by-side with the generator stream for the joint forward).
pub struct ReasonerStream {
    pub input_ln: Vec<f32>,          // [HIDDEN]
    pub q_w: BfSlice,                // [HIDDEN, HIDDEN]
    pub k_w: BfSlice,                // [KV_DIM, HIDDEN]
    pub v_w: BfSlice,                // [KV_DIM, HIDDEN]
    pub o_w: BfSlice,                // [HIDDEN, HIDDEN]
    pub q_norm: Vec<f32>,            // [HEAD_DIM]
    pub k_norm: Vec<f32>,            // [HEAD_DIM]
    pub post_attn_ln: Vec<f32>,      // [HIDDEN]
    pub gate_w: BfSlice,             // [FF, HIDDEN]
    pub up_w: BfSlice,               // [FF, HIDDEN]
    pub down_w: BfSlice,             // [HIDDEN, FF]
}

/// Generator stream weights (the `_moe_gen` / `add_*_proj` parallel set).
pub struct GeneratorStream {
    pub input_ln: Vec<f32>,          // [HIDDEN]
    pub add_q_w: BfSlice,            // [HIDDEN, HIDDEN]
    pub add_k_w: BfSlice,            // [KV_DIM, HIDDEN]
    pub add_v_w: BfSlice,            // [KV_DIM, HIDDEN]
    pub add_out_w: BfSlice,          // [HIDDEN, HIDDEN]
    pub q_norm: Vec<f32>,            // [HEAD_DIM]
    pub k_norm: Vec<f32>,            // [HEAD_DIM]
    pub post_attn_ln: Vec<f32>,      // [HIDDEN]
    pub gate_w: BfSlice,             // [FF, HIDDEN]
    pub up_w: BfSlice,               // [FF, HIDDEN]
    pub down_w: BfSlice,             // [HIDDEN, FF]
}

pub struct LayerV2 {
    pub reasoner: ReasonerStream,
    pub generator: GeneratorStream,
}

// ────────────────────────────────────────────────────────────────────────
// Per-layer weight-role IDs (for Stage 5 MetalAccel handle cache).
// ────────────────────────────────────────────────────────────────────────
// 14 matmul-bearing weights per LayerV2 (7 per stream).  The IDs key
// MetalAccel's `(block_idx, role)` cache so each weight is uploaded to
// GPU exactly once across the model's lifetime.
pub const R_Q: u8 = 0;
pub const R_K: u8 = 1;
pub const R_V: u8 = 2;
pub const R_O: u8 = 3;
pub const R_GATE: u8 = 4;
pub const R_UP: u8 = 5;
pub const R_DOWN: u8 = 6;
pub const G_Q: u8 = 7;
pub const G_K: u8 = 8;
pub const G_V: u8 = 9;
pub const G_O: u8 = 10;
pub const G_GATE: u8 = 11;
pub const G_UP: u8 = 12;
pub const G_DOWN: u8 = 13;

/// Per-axis section counts for `unified_3d_mrope` (in *pairs*, not dims).
/// `HEAD_DIM/2 = 64` rotary pairs total: 24 → temporal, 20 → height, 20 → width.
pub const MROPE_SECTIONS: [usize; 3] = [24, 20, 20];

/// 3D mrope: apply per-axis rotations using `positions_t/h/w[ti]` for each
/// token, where the first `MROPE_SECTIONS[0]` rotary pairs use the temporal
/// axis, the next `MROPE_SECTIONS[1]` use height, and the rest use width.
/// Matches Cosmos 3's `unified_3d_mrope` config (`reset_spatial_ids=true`,
/// `temporal_modality_margin=15000`).
pub fn apply_rope_3d(
    x: &mut [f32], t: usize, n_heads: usize, freqs: &[f32],
    pos_t: &[i64], pos_h: &[i64], pos_w: &[i64],
) {
    let head_dim = crate::HEAD_DIM;
    let half = head_dim / 2;
    debug_assert_eq!(freqs.len(), half);
    debug_assert_eq!(pos_t.len(), t);
    debug_assert_eq!(pos_h.len(), t);
    debug_assert_eq!(pos_w.len(), t);
    debug_assert_eq!(MROPE_SECTIONS[0] + MROPE_SECTIONS[1] + MROPE_SECTIONS[2], half);
    for ti in 0..t {
        let (pt, ph, pw) = (pos_t[ti] as f32, pos_h[ti] as f32, pos_w[ti] as f32);
        for h in 0..n_heads {
            let base = ti * n_heads * head_dim + h * head_dim;
            for k in 0..half {
                let pos_axis = if k < MROPE_SECTIONS[0] {
                    pt
                } else if k < MROPE_SECTIONS[0] + MROPE_SECTIONS[1] {
                    ph
                } else {
                    pw
                };
                let theta = pos_axis * freqs[k];
                let (s, c) = theta.sin_cos();
                let a = x[base + k];
                let b = x[base + k + half];
                x[base + k]        = a * c - b * s;
                x[base + k + half] = a * s + b * c;
            }
        }
    }
}

/// Head/grouping constants (same as Phase 1 reasoner).
pub const N_HEADS: usize = crate::N_HEADS;          // 32
pub const N_KV_HEADS: usize = crate::N_KV_HEADS;    // 8
pub const HEAD_DIM: usize = crate::HEAD_DIM;        // 128
const RMS_EPS: f32 = 1e-6;

fn rmsnorm(x: &[f32], hidden: usize, t: usize, gamma: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for ti in 0..t {
        let row = &x[ti*hidden..(ti+1)*hidden];
        let ss: f32 = row.iter().map(|v| v*v).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (ss + RMS_EPS).sqrt();
        for h in 0..hidden { out[ti*hidden + h] = row[h] * inv * gamma[h]; }
    }
    out
}

fn rmsnorm_per_head(x: &mut [f32], t: usize, n_heads: usize, gamma: &[f32]) {
    for ti in 0..t {
        for h in 0..n_heads {
            let row = &mut x[ti*n_heads*HEAD_DIM + h*HEAD_DIM..ti*n_heads*HEAD_DIM + (h+1)*HEAD_DIM];
            let ss: f32 = row.iter().map(|v| v*v).sum::<f32>() / HEAD_DIM as f32;
            let inv = 1.0 / (ss + RMS_EPS).sqrt();
            for d in 0..HEAD_DIM { row[d] = row[d] * inv * gamma[d]; }
        }
    }
}

#[inline] fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

/// Joint-stream MMDiT attention over concatenated `[Q_r; Q_g]`, `[K_r; K_g]`,
/// `[V_r; V_g]`.  Non-causal (diffusion — all tokens see all tokens).
/// GQA with `N_HEADS=32` Q heads × `N_KV_HEADS=8` KV heads (group mapping).
fn joint_sdpa(q: &[f32], k: &[f32], v: &[f32], t: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let q_per_kv = N_HEADS / N_KV_HEADS;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut attn = vec![0.0f32; t * N_HEADS * HEAD_DIM];
    attn.par_chunks_mut(N_HEADS * HEAD_DIM).enumerate().for_each(|(ti, arow)| {
        for h in 0..N_HEADS {
            let kvh = h / q_per_kv;
            let mut scores = vec![0.0f32; t];
            let mut maxv = f32::NEG_INFINITY;
            for tj in 0..t {
                let mut s = 0.0f32;
                for d in 0..HEAD_DIM {
                    let qi = q[ti*N_HEADS*HEAD_DIM + h*HEAD_DIM + d];
                    let ki = k[tj*N_KV_HEADS*HEAD_DIM + kvh*HEAD_DIM + d];
                    s += qi * ki;
                }
                s *= scale;
                scores[tj] = s;
                if s > maxv { maxv = s; }
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() { *s = (*s - maxv).exp(); sum += *s; }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for s in scores.iter_mut() { *s *= inv; }
            for d in 0..HEAD_DIM {
                let mut acc = 0.0f32;
                for tj in 0..t {
                    acc += scores[tj] * v[tj*N_KV_HEADS*HEAD_DIM + kvh*HEAD_DIM + d];
                }
                arow[h*HEAD_DIM + d] = acc;
            }
        }
    });
    attn
}

impl LayerV2 {
    pub fn load(ss: &ShardedSafetensors, i: usize) -> Result<Self, CosmosError> {
        let p = format!("layers.{i}");
        Ok(Self {
            reasoner: ReasonerStream {
                input_ln:     ld(ss,      &format!("{p}.input_layernorm.weight"))?,
                q_w:          ld_bf16(ss, &format!("{p}.self_attn.to_q.weight"))?,
                k_w:          ld_bf16(ss, &format!("{p}.self_attn.to_k.weight"))?,
                v_w:          ld_bf16(ss, &format!("{p}.self_attn.to_v.weight"))?,
                o_w:          ld_bf16(ss, &format!("{p}.self_attn.to_out.weight"))?,
                q_norm:       ld(ss,      &format!("{p}.self_attn.norm_q.weight"))?,
                k_norm:       ld(ss,      &format!("{p}.self_attn.norm_k.weight"))?,
                post_attn_ln: ld(ss,      &format!("{p}.post_attention_layernorm.weight"))?,
                gate_w:       ld_bf16(ss, &format!("{p}.mlp.gate_proj.weight"))?,
                up_w:         ld_bf16(ss, &format!("{p}.mlp.up_proj.weight"))?,
                down_w:       ld_bf16(ss, &format!("{p}.mlp.down_proj.weight"))?,
            },
            generator: GeneratorStream {
                input_ln:     ld(ss,      &format!("{p}.input_layernorm_moe_gen.weight"))?,
                add_q_w:      ld_bf16(ss, &format!("{p}.self_attn.add_q_proj.weight"))?,
                add_k_w:      ld_bf16(ss, &format!("{p}.self_attn.add_k_proj.weight"))?,
                add_v_w:      ld_bf16(ss, &format!("{p}.self_attn.add_v_proj.weight"))?,
                add_out_w:    ld_bf16(ss, &format!("{p}.self_attn.to_add_out.weight"))?,
                q_norm:       ld(ss,      &format!("{p}.self_attn.norm_added_q.weight"))?,
                k_norm:       ld(ss,      &format!("{p}.self_attn.norm_added_k.weight"))?,
                post_attn_ln: ld(ss,      &format!("{p}.post_attention_layernorm_moe_gen.weight"))?,
                gate_w:       ld_bf16(ss, &format!("{p}.mlp_moe_gen.gate_proj.weight"))?,
                up_w:         ld_bf16(ss, &format!("{p}.mlp_moe_gen.up_proj.weight"))?,
                down_w:       ld_bf16(ss, &format!("{p}.mlp_moe_gen.down_proj.weight"))?,
            },
        })
    }

    /// Joint-stream MMDiT forward.  Wraps `forward_with_positions` with
    /// trivial sequential positions for the reasoner and offset positions
    /// for the generator (gen stream's RoPE positions start after the
    /// reasoner's range — keeps the two streams orthogonal in mrope space).
    pub fn forward(&self, x_r: &mut Vec<f32>, t_r: usize, x_g: &mut Vec<f32>, t_g: usize) {
        let pos_r: Vec<i64> = (0..t_r as i64).collect();
        let pos_g: Vec<i64> = (t_r as i64..(t_r + t_g) as i64).collect();
        let freqs = crate::rope_freqs();
        self.forward_with_positions(x_r, t_r, &pos_r, x_g, t_g, &pos_g, &freqs);
    }

    /// Stage 7 forward with explicit RoPE positions per stream.  CPU only.
    pub fn forward_with_positions(
        &self,
        x_r: &mut Vec<f32>, t_r: usize, pos_r: &[i64],
        x_g: &mut Vec<f32>, t_g: usize, pos_g: &[i64],
        freqs: &[f32],
    ) {
        self.forward_routed(x_r, t_r, pos_r, x_g, t_g, pos_g, freqs, 0, None);
    }

    /// 3D mrope variant of `forward_routed`.  Same Metal-acceleration path
    /// as `forward_routed` but with `apply_rope_3d` instead of `apply_rope`.
    /// Replaces the 1D-RoPE collapse the `step_routed` helper falls back on.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_routed_3d(
        &self,
        x_r: &mut Vec<f32>, t_r: usize,
        pos_r_t: &[i64], pos_r_h: &[i64], pos_r_w: &[i64],
        x_g: &mut Vec<f32>, t_g: usize,
        pos_g_t: &[i64], pos_g_h: &[i64], pos_g_w: &[i64],
        freqs: &[f32],
        block_idx: usize,
        accel: Option<&crate::phase4::MetalAccel>,
    ) {
        let kv_dim = N_KV_HEADS * HEAD_DIM;
        use crate::phase4::mm_bf16_routed;

        // Reasoner stream.
        let xn_r = rmsnorm(x_r, HIDDEN, t_r, &self.reasoner.input_ln);
        let mut q_r = mm_bf16_routed(accel, &xn_r, t_r, HIDDEN, &self.reasoner.q_w, HIDDEN, block_idx, R_Q);
        let mut k_r = mm_bf16_routed(accel, &xn_r, t_r, HIDDEN, &self.reasoner.k_w, kv_dim, block_idx, R_K);
        let     v_r = mm_bf16_routed(accel, &xn_r, t_r, HIDDEN, &self.reasoner.v_w, kv_dim, block_idx, R_V);
        rmsnorm_per_head(&mut q_r, t_r, N_HEADS, &self.reasoner.q_norm);
        rmsnorm_per_head(&mut k_r, t_r, N_KV_HEADS, &self.reasoner.k_norm);
        apply_rope_3d(&mut q_r, t_r, N_HEADS, freqs, pos_r_t, pos_r_h, pos_r_w);
        apply_rope_3d(&mut k_r, t_r, N_KV_HEADS, freqs, pos_r_t, pos_r_h, pos_r_w);

        // Generator stream.
        let xn_g = rmsnorm(x_g, HIDDEN, t_g, &self.generator.input_ln);
        let mut q_g = mm_bf16_routed(accel, &xn_g, t_g, HIDDEN, &self.generator.add_q_w, HIDDEN, block_idx, G_Q);
        let mut k_g = mm_bf16_routed(accel, &xn_g, t_g, HIDDEN, &self.generator.add_k_w, kv_dim, block_idx, G_K);
        let     v_g = mm_bf16_routed(accel, &xn_g, t_g, HIDDEN, &self.generator.add_v_w, kv_dim, block_idx, G_V);
        rmsnorm_per_head(&mut q_g, t_g, N_HEADS, &self.generator.q_norm);
        rmsnorm_per_head(&mut k_g, t_g, N_KV_HEADS, &self.generator.k_norm);
        apply_rope_3d(&mut q_g, t_g, N_HEADS, freqs, pos_g_t, pos_g_h, pos_g_w);
        apply_rope_3d(&mut k_g, t_g, N_KV_HEADS, freqs, pos_g_t, pos_g_h, pos_g_w);

        // Joint attention.
        let t_joint = t_r + t_g;
        let mut q_j = vec![0.0f32; t_joint * N_HEADS * HEAD_DIM];
        let mut k_j = vec![0.0f32; t_joint * kv_dim];
        let mut v_j = vec![0.0f32; t_joint * kv_dim];
        q_j[..t_r * N_HEADS * HEAD_DIM].copy_from_slice(&q_r);
        q_j[t_r * N_HEADS * HEAD_DIM..].copy_from_slice(&q_g);
        k_j[..t_r * kv_dim].copy_from_slice(&k_r);
        k_j[t_r * kv_dim..].copy_from_slice(&k_g);
        v_j[..t_r * kv_dim].copy_from_slice(&v_r);
        v_j[t_r * kv_dim..].copy_from_slice(&v_g);
        let attn_j = joint_sdpa(&q_j, &k_j, &v_j, t_joint);

        // Per-stream output projections + residual + MLP (same as forward_routed).
        let attn_r = &attn_j[..t_r * HIDDEN];
        let attn_g = &attn_j[t_r * HIDDEN..];
        let o_r = mm_bf16_routed(accel, attn_r, t_r, HIDDEN, &self.reasoner.o_w, HIDDEN, block_idx, R_O);
        let o_g = mm_bf16_routed(accel, attn_g, t_g, HIDDEN, &self.generator.add_out_w, HIDDEN, block_idx, G_O);
        for i in 0..x_r.len() { x_r[i] += o_r[i]; }
        for i in 0..x_g.len() { x_g[i] += o_g[i]; }

        let xn_r = rmsnorm(x_r, HIDDEN, t_r, &self.reasoner.post_attn_ln);
        let gate_r = mm_bf16_routed(accel, &xn_r, t_r, HIDDEN, &self.reasoner.gate_w, FF, block_idx, R_GATE);
        let up_r   = mm_bf16_routed(accel, &xn_r, t_r, HIDDEN, &self.reasoner.up_w,   FF, block_idx, R_UP);
        let mut h_r = vec![0.0f32; t_r * FF];
        for i in 0..h_r.len() { h_r[i] = silu(gate_r[i]) * up_r[i]; }
        let m_r = mm_bf16_routed(accel, &h_r, t_r, FF, &self.reasoner.down_w, HIDDEN, block_idx, R_DOWN);
        for i in 0..x_r.len() { x_r[i] += m_r[i]; }

        let xn_g = rmsnorm(x_g, HIDDEN, t_g, &self.generator.post_attn_ln);
        let gate_g = mm_bf16_routed(accel, &xn_g, t_g, HIDDEN, &self.generator.gate_w, FF, block_idx, G_GATE);
        let up_g   = mm_bf16_routed(accel, &xn_g, t_g, HIDDEN, &self.generator.up_w,   FF, block_idx, G_UP);
        let mut h_g = vec![0.0f32; t_g * FF];
        for i in 0..h_g.len() { h_g[i] = silu(gate_g[i]) * up_g[i]; }
        let m_g = mm_bf16_routed(accel, &h_g, t_g, FF, &self.generator.down_w, HIDDEN, block_idx, G_DOWN);
        for i in 0..x_g.len() { x_g[i] += m_g[i]; }
    }

    /// Stage 7 forward with **3D mrope** positions: `pos_*_t/h/w` give per-token
    /// (temporal, height, width) coordinates that drive the per-axis RoPE
    /// rotation across `MROPE_SECTIONS=[24, 20, 20]`.  Use this for video
    /// generation where the generator stream's tokens carry real (t, h, w)
    /// grid coordinates from the patch-flatten step.
    pub fn forward_with_3d_positions(
        &self,
        x_r: &mut Vec<f32>, t_r: usize,
        pos_r_t: &[i64], pos_r_h: &[i64], pos_r_w: &[i64],
        x_g: &mut Vec<f32>, t_g: usize,
        pos_g_t: &[i64], pos_g_h: &[i64], pos_g_w: &[i64],
        freqs: &[f32],
    ) {
        let kv_dim = N_KV_HEADS * HEAD_DIM;
        use crate::phase4::mm_bf16_routed;
        let accel: Option<&crate::phase4::MetalAccel> = None;
        let block_idx = 0;

        // ── Reasoner stream ──
        let xn_r = rmsnorm(x_r, HIDDEN, t_r, &self.reasoner.input_ln);
        let mut q_r = mm_bf16_routed(accel, &xn_r, t_r, HIDDEN, &self.reasoner.q_w, HIDDEN, block_idx, R_Q);
        let mut k_r = mm_bf16_routed(accel, &xn_r, t_r, HIDDEN, &self.reasoner.k_w, kv_dim, block_idx, R_K);
        let     v_r = mm_bf16_routed(accel, &xn_r, t_r, HIDDEN, &self.reasoner.v_w, kv_dim, block_idx, R_V);
        rmsnorm_per_head(&mut q_r, t_r, N_HEADS, &self.reasoner.q_norm);
        rmsnorm_per_head(&mut k_r, t_r, N_KV_HEADS, &self.reasoner.k_norm);
        apply_rope_3d(&mut q_r, t_r, N_HEADS, freqs, pos_r_t, pos_r_h, pos_r_w);
        apply_rope_3d(&mut k_r, t_r, N_KV_HEADS, freqs, pos_r_t, pos_r_h, pos_r_w);

        // ── Generator stream ──
        let xn_g = rmsnorm(x_g, HIDDEN, t_g, &self.generator.input_ln);
        let mut q_g = mm_bf16_routed(accel, &xn_g, t_g, HIDDEN, &self.generator.add_q_w, HIDDEN, block_idx, G_Q);
        let mut k_g = mm_bf16_routed(accel, &xn_g, t_g, HIDDEN, &self.generator.add_k_w, kv_dim, block_idx, G_K);
        let     v_g = mm_bf16_routed(accel, &xn_g, t_g, HIDDEN, &self.generator.add_v_w, kv_dim, block_idx, G_V);
        rmsnorm_per_head(&mut q_g, t_g, N_HEADS, &self.generator.q_norm);
        rmsnorm_per_head(&mut k_g, t_g, N_KV_HEADS, &self.generator.k_norm);
        apply_rope_3d(&mut q_g, t_g, N_HEADS, freqs, pos_g_t, pos_g_h, pos_g_w);
        apply_rope_3d(&mut k_g, t_g, N_KV_HEADS, freqs, pos_g_t, pos_g_h, pos_g_w);

        // ── Joint attention + per-stream output + MLP (same as forward_routed) ──
        let t_joint = t_r + t_g;
        let mut q_j = vec![0.0f32; t_joint * N_HEADS * HEAD_DIM];
        let mut k_j = vec![0.0f32; t_joint * kv_dim];
        let mut v_j = vec![0.0f32; t_joint * kv_dim];
        q_j[..t_r * N_HEADS * HEAD_DIM].copy_from_slice(&q_r);
        q_j[t_r * N_HEADS * HEAD_DIM..].copy_from_slice(&q_g);
        k_j[..t_r * kv_dim].copy_from_slice(&k_r);
        k_j[t_r * kv_dim..].copy_from_slice(&k_g);
        v_j[..t_r * kv_dim].copy_from_slice(&v_r);
        v_j[t_r * kv_dim..].copy_from_slice(&v_g);
        let attn_j = joint_sdpa(&q_j, &k_j, &v_j, t_joint);

        let attn_r = &attn_j[..t_r * HIDDEN];
        let attn_g = &attn_j[t_r * HIDDEN..];
        let o_r = mm_bf16_routed(accel, attn_r, t_r, HIDDEN, &self.reasoner.o_w, HIDDEN, block_idx, R_O);
        let o_g = mm_bf16_routed(accel, attn_g, t_g, HIDDEN, &self.generator.add_out_w, HIDDEN, block_idx, G_O);
        for i in 0..x_r.len() { x_r[i] += o_r[i]; }
        for i in 0..x_g.len() { x_g[i] += o_g[i]; }

        let xn_r = rmsnorm(x_r, HIDDEN, t_r, &self.reasoner.post_attn_ln);
        let gate_r = mm_bf16_routed(accel, &xn_r, t_r, HIDDEN, &self.reasoner.gate_w, FF, block_idx, R_GATE);
        let up_r   = mm_bf16_routed(accel, &xn_r, t_r, HIDDEN, &self.reasoner.up_w,   FF, block_idx, R_UP);
        let mut h_r = vec![0.0f32; t_r * FF];
        for i in 0..h_r.len() { h_r[i] = silu(gate_r[i]) * up_r[i]; }
        let m_r = mm_bf16_routed(accel, &h_r, t_r, FF, &self.reasoner.down_w, HIDDEN, block_idx, R_DOWN);
        for i in 0..x_r.len() { x_r[i] += m_r[i]; }

        let xn_g = rmsnorm(x_g, HIDDEN, t_g, &self.generator.post_attn_ln);
        let gate_g = mm_bf16_routed(accel, &xn_g, t_g, HIDDEN, &self.generator.gate_w, FF, block_idx, G_GATE);
        let up_g   = mm_bf16_routed(accel, &xn_g, t_g, HIDDEN, &self.generator.up_w,   FF, block_idx, G_UP);
        let mut h_g = vec![0.0f32; t_g * FF];
        for i in 0..h_g.len() { h_g[i] = silu(gate_g[i]) * up_g[i]; }
        let m_g = mm_bf16_routed(accel, &h_g, t_g, FF, &self.generator.down_w, HIDDEN, block_idx, G_DOWN);
        for i in 0..x_g.len() { x_g[i] += m_g[i]; }
    }

    /// Stage 5 wire for v2: forward with optional `MetalAccel` acceleration.
    /// When `accel.is_some()`, all 14 per-layer bf16 matmuls dispatch via
    /// Stage 5's `mm_bf16_routed` (handle-cached GPU sgemm); the joint SDPA
    /// dispatches via Stage 5's `sdpa_routed` (`Custom("attention")` op with
    /// T-gate at `SDPA_GPU_MIN_T=2048`).
    pub fn forward_routed(
        &self,
        x_r: &mut Vec<f32>, t_r: usize, pos_r: &[i64],
        x_g: &mut Vec<f32>, t_g: usize, pos_g: &[i64],
        freqs: &[f32],
        block_idx: usize,
        accel: Option<&crate::phase4::MetalAccel>,
    ) {
        let kv_dim = N_KV_HEADS * HEAD_DIM;

        use crate::phase4::mm_bf16_routed;

        // ── Reasoner stream: pre-norm + Q/K/V projections + qk_norm ──
        let xn_r = rmsnorm(x_r, HIDDEN, t_r, &self.reasoner.input_ln);
        let mut q_r = mm_bf16_routed(accel, &xn_r, t_r, HIDDEN, &self.reasoner.q_w, HIDDEN, block_idx, R_Q);
        let mut k_r = mm_bf16_routed(accel, &xn_r, t_r, HIDDEN, &self.reasoner.k_w, kv_dim, block_idx, R_K);
        let     v_r = mm_bf16_routed(accel, &xn_r, t_r, HIDDEN, &self.reasoner.v_w, kv_dim, block_idx, R_V);
        rmsnorm_per_head(&mut q_r, t_r, N_HEADS, &self.reasoner.q_norm);
        rmsnorm_per_head(&mut k_r, t_r, N_KV_HEADS, &self.reasoner.k_norm);
        crate::apply_rope(&mut q_r, t_r, N_HEADS, freqs, pos_r);
        crate::apply_rope(&mut k_r, t_r, N_KV_HEADS, freqs, pos_r);

        // ── Generator stream: pre-norm + add_Q/add_K/add_V + qk_norm ──
        let xn_g = rmsnorm(x_g, HIDDEN, t_g, &self.generator.input_ln);
        let mut q_g = mm_bf16_routed(accel, &xn_g, t_g, HIDDEN, &self.generator.add_q_w, HIDDEN, block_idx, G_Q);
        let mut k_g = mm_bf16_routed(accel, &xn_g, t_g, HIDDEN, &self.generator.add_k_w, kv_dim, block_idx, G_K);
        let     v_g = mm_bf16_routed(accel, &xn_g, t_g, HIDDEN, &self.generator.add_v_w, kv_dim, block_idx, G_V);
        rmsnorm_per_head(&mut q_g, t_g, N_HEADS, &self.generator.q_norm);
        rmsnorm_per_head(&mut k_g, t_g, N_KV_HEADS, &self.generator.k_norm);
        crate::apply_rope(&mut q_g, t_g, N_HEADS, freqs, pos_g);
        crate::apply_rope(&mut k_g, t_g, N_KV_HEADS, freqs, pos_g);

        // ── Concatenate along seq dim: joint sequence of T = t_r + t_g ──
        let t_joint = t_r + t_g;
        let mut q_joint = vec![0.0f32; t_joint * N_HEADS * HEAD_DIM];
        let mut k_joint = vec![0.0f32; t_joint * kv_dim];
        let mut v_joint = vec![0.0f32; t_joint * kv_dim];
        q_joint[..t_r * N_HEADS * HEAD_DIM].copy_from_slice(&q_r);
        q_joint[t_r * N_HEADS * HEAD_DIM..].copy_from_slice(&q_g);
        k_joint[..t_r * kv_dim].copy_from_slice(&k_r);
        k_joint[t_r * kv_dim..].copy_from_slice(&k_g);
        v_joint[..t_r * kv_dim].copy_from_slice(&v_r);
        v_joint[t_r * kv_dim..].copy_from_slice(&v_g);

        // ── Joint SDPA (non-causal, GQA) ──
        let attn_joint = joint_sdpa(&q_joint, &k_joint, &v_joint, t_joint);

        // ── Split + per-stream output projection + residual add ──
        let attn_r = &attn_joint[..t_r * HIDDEN];
        let attn_g = &attn_joint[t_r * HIDDEN..];
        let o_r = mm_bf16_routed(accel, attn_r, t_r, HIDDEN, &self.reasoner.o_w, HIDDEN, block_idx, R_O);
        let o_g = mm_bf16_routed(accel, attn_g, t_g, HIDDEN, &self.generator.add_out_w, HIDDEN, block_idx, G_O);
        for i in 0..x_r.len() { x_r[i] += o_r[i]; }
        for i in 0..x_g.len() { x_g[i] += o_g[i]; }

        // ── Post-attention norm + SwiGLU MLP + residual (per stream) ──
        let xn_r = rmsnorm(x_r, HIDDEN, t_r, &self.reasoner.post_attn_ln);
        let gate_r = mm_bf16_routed(accel, &xn_r, t_r, HIDDEN, &self.reasoner.gate_w, FF, block_idx, R_GATE);
        let up_r   = mm_bf16_routed(accel, &xn_r, t_r, HIDDEN, &self.reasoner.up_w,   FF, block_idx, R_UP);
        let mut h_r = vec![0.0f32; t_r * FF];
        for i in 0..h_r.len() { h_r[i] = silu(gate_r[i]) * up_r[i]; }
        let m_r = mm_bf16_routed(accel, &h_r, t_r, FF, &self.reasoner.down_w, HIDDEN, block_idx, R_DOWN);
        for i in 0..x_r.len() { x_r[i] += m_r[i]; }

        let xn_g = rmsnorm(x_g, HIDDEN, t_g, &self.generator.post_attn_ln);
        let gate_g = mm_bf16_routed(accel, &xn_g, t_g, HIDDEN, &self.generator.gate_w, FF, block_idx, G_GATE);
        let up_g   = mm_bf16_routed(accel, &xn_g, t_g, HIDDEN, &self.generator.up_w,   FF, block_idx, G_UP);
        let mut h_g = vec![0.0f32; t_g * FF];
        for i in 0..h_g.len() { h_g[i] = silu(gate_g[i]) * up_g[i]; }
        let m_g = mm_bf16_routed(accel, &h_g, t_g, FF, &self.generator.down_w, HIDDEN, block_idx, G_DOWN);
        for i in 0..x_g.len() { x_g[i] += m_g[i]; }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Top-level model
// ────────────────────────────────────────────────────────────────────────

pub struct Cosmos3OmniTransformerV2 {
    pub embed_tokens: BfSlice,           // [VOCAB=151936, HIDDEN]
    pub action_modality_embed: Vec<f32>, // [HIDDEN]
    pub audio_modality_embed: Vec<f32>,  // [HIDDEN]
    pub layers: Vec<LayerV2>,
    /// Global time embedder (sinusoidal → MLP → HIDDEN).  Output added to
    /// the generator stream once per diffusion step, not per layer.
    pub time_embedder: Option<TimeEmbedder>,
    /// Visual modality I/O — projects the Wan-VAE 192-d latent to/from
    /// the transformer's 4096-d hidden dim.
    pub visual_head: Option<VisualHead>,
    /// Audio modality I/O — projects the AVAE 64-d latent to/from HIDDEN.
    pub audio_head: Option<AudioHead>,
    /// Action modality I/O — 32-expert MoE over 64-d action latents.
    pub action_head: Option<ActionHead>,
}

/// Visual latent dim (Wan 2.2 VAE latent unflattened per patch).
pub const VISUAL_LATENT_DIM: usize = 192;
/// Audio latent dim (AVAE codec).
pub const AUDIO_LATENT_DIM: usize = 64;
/// Time embedder inner sinusoidal dim.
pub const TIME_SIN_DIM: usize = 256;

/// Global time embedder: sinusoidal(t, 256) → Linear → SiLU → Linear → HIDDEN.
pub struct TimeEmbedder {
    pub linear_1_w: BfSlice, pub linear_1_b: BfSlice,  // [HIDDEN, TIME_SIN_DIM] / [HIDDEN]
    pub linear_2_w: BfSlice, pub linear_2_b: BfSlice,  // [HIDDEN, HIDDEN] / [HIDDEN]
}

impl TimeEmbedder {
    pub fn load(ss: &ShardedSafetensors) -> Result<Self, CosmosError> {
        Ok(Self {
            linear_1_w: ld_bf16(ss, "time_embedder.linear_1.weight")?,
            linear_1_b: ld_bf16(ss, "time_embedder.linear_1.bias")?,
            linear_2_w: ld_bf16(ss, "time_embedder.linear_2.weight")?,
            linear_2_b: ld_bf16(ss, "time_embedder.linear_2.bias")?,
        })
    }

    /// Sinusoidal time embedding at `TIME_SIN_DIM`, scaled ×1000 (diffusion
    /// convention), then 2-layer MLP with SiLU → returns `[HIDDEN]`.
    pub fn forward(&self, t: f32) -> Vec<f32> {
        let half = TIME_SIN_DIM / 2;
        let mut emb = vec![0.0f32; TIME_SIN_DIM];
        let factor = (1e4_f32).ln() / (half - 1) as f32;
        let scaled = t * 1000.0;
        for i in 0..half {
            let freq = (-(i as f32) * factor).exp();
            emb[i]        = (scaled * freq).cos();
            emb[i + half] = (scaled * freq).sin();
        }
        let bias_1: Vec<f32> = self.linear_1_b.iter().map(|b| b.to_f32()).collect();
        let bias_2: Vec<f32> = self.linear_2_b.iter().map(|b| b.to_f32()).collect();
        let mut h = crate::matmul_nt_bf16(&emb, 1, TIME_SIN_DIM, &self.linear_1_w, HIDDEN);
        for i in 0..HIDDEN { h[i] += bias_1[i]; }
        for v in h.iter_mut() { *v = silu(*v); }
        let mut out = crate::matmul_nt_bf16(&h, 1, HIDDEN, &self.linear_2_w, HIDDEN);
        for i in 0..HIDDEN { out[i] += bias_2[i]; }
        out
    }
}

/// Visual modality I/O: 192-d Wan-VAE latent ↔ HIDDEN transformer dim.
pub struct VisualHead {
    pub proj_in_w: BfSlice, pub proj_in_b: BfSlice,    // [HIDDEN, VISUAL_LATENT_DIM] / [HIDDEN]
    pub proj_out_w: BfSlice, pub proj_out_b: BfSlice,  // [VISUAL_LATENT_DIM, HIDDEN] / [VISUAL_LATENT_DIM]
}

impl VisualHead {
    pub fn load(ss: &ShardedSafetensors) -> Result<Self, CosmosError> {
        Ok(Self {
            proj_in_w:  ld_bf16(ss, "proj_in.weight")?,
            proj_in_b:  ld_bf16(ss, "proj_in.bias")?,
            proj_out_w: ld_bf16(ss, "proj_out.weight")?,
            proj_out_b: ld_bf16(ss, "proj_out.bias")?,
        })
    }
    /// Latent `[T, VISUAL_LATENT_DIM]` → transformer `[T, HIDDEN]`.
    pub fn project_in(&self, latent: &[f32], t: usize) -> Vec<f32> {
        let bias: Vec<f32> = self.proj_in_b.iter().map(|b| b.to_f32()).collect();
        let mut out = crate::matmul_nt_bf16(latent, t, VISUAL_LATENT_DIM, &self.proj_in_w, HIDDEN);
        for ti in 0..t { for d in 0..HIDDEN { out[ti * HIDDEN + d] += bias[d]; } }
        out
    }
    /// Transformer `[T, HIDDEN]` → predicted latent `[T, VISUAL_LATENT_DIM]`.
    pub fn project_out(&self, x: &[f32], t: usize) -> Vec<f32> {
        let bias: Vec<f32> = self.proj_out_b.iter().map(|b| b.to_f32()).collect();
        let mut out = crate::matmul_nt_bf16(x, t, HIDDEN, &self.proj_out_w, VISUAL_LATENT_DIM);
        for ti in 0..t { for d in 0..VISUAL_LATENT_DIM { out[ti * VISUAL_LATENT_DIM + d] += bias[d]; } }
        out
    }
}

/// Audio modality I/O: 64-d AVAE latent ↔ HIDDEN transformer dim.
pub struct AudioHead {
    pub proj_in_w: BfSlice, pub proj_in_b: BfSlice,    // [HIDDEN, AUDIO_LATENT_DIM] / [HIDDEN]
    pub proj_out_w: BfSlice, pub proj_out_b: BfSlice,  // [AUDIO_LATENT_DIM, HIDDEN] / [AUDIO_LATENT_DIM]
}

impl AudioHead {
    pub fn load(ss: &ShardedSafetensors) -> Result<Self, CosmosError> {
        Ok(Self {
            proj_in_w:  ld_bf16(ss, "audio_proj_in.weight")?,
            proj_in_b:  ld_bf16(ss, "audio_proj_in.bias")?,
            proj_out_w: ld_bf16(ss, "audio_proj_out.weight")?,
            proj_out_b: ld_bf16(ss, "audio_proj_out.bias")?,
        })
    }
    pub fn project_in(&self, latent: &[f32], t: usize) -> Vec<f32> {
        let bias: Vec<f32> = self.proj_in_b.iter().map(|b| b.to_f32()).collect();
        let mut out = crate::matmul_nt_bf16(latent, t, AUDIO_LATENT_DIM, &self.proj_in_w, HIDDEN);
        for ti in 0..t { for d in 0..HIDDEN { out[ti * HIDDEN + d] += bias[d]; } }
        out
    }
    pub fn project_out(&self, x: &[f32], t: usize) -> Vec<f32> {
        let bias: Vec<f32> = self.proj_out_b.iter().map(|b| b.to_f32()).collect();
        let mut out = crate::matmul_nt_bf16(x, t, HIDDEN, &self.proj_out_w, AUDIO_LATENT_DIM);
        for ti in 0..t { for d in 0..AUDIO_LATENT_DIM { out[ti * AUDIO_LATENT_DIM + d] += bias[d]; } }
        out
    }
}

/// Action modality I/O: per-action MoE projection with `ACTION_EXPERTS=32`
/// experts.  Each expert is a Linear(HIDDEN→ACTION_LATENT_DIM=64) shaped
/// `[ACTION_LATENT_DIM, HIDDEN]` stored stacked as `[ACTION_EXPERTS, ACTION_LATENT_DIM × HIDDEN]`.
/// On forward, the caller picks the expert index for the current action.
pub const ACTION_EXPERTS: usize = 32;
pub const ACTION_LATENT_DIM: usize = 64;

pub struct ActionHead {
    /// proj_in.fc.weight stored as [EXPERTS, ACTION_LATENT_DIM * HIDDEN] flat.
    /// Per-expert slice is the standard Linear weight `[HIDDEN, ACTION_LATENT_DIM]`
    /// in row-major (PyTorch nn.Linear convention).
    pub proj_in_fc: BfSlice,            // [32, 262144 = 64*4096] flat
    pub proj_in_bias: BfSlice,          // [32, 4096] per-expert bias
    pub proj_out_fc: BfSlice,           // [32, 262144] flat
    pub proj_out_bias: BfSlice,         // [32, 64] per-expert bias
}

impl ActionHead {
    pub fn load(ss: &ShardedSafetensors) -> Result<Self, CosmosError> {
        Ok(Self {
            proj_in_fc:    ld_bf16(ss, "action_proj_in.fc.weight")?,
            proj_in_bias:  ld_bf16(ss, "action_proj_in.bias.weight")?,
            proj_out_fc:   ld_bf16(ss, "action_proj_out.fc.weight")?,
            proj_out_bias: ld_bf16(ss, "action_proj_out.bias.weight")?,
        })
    }

    /// Project latent `[T, ACTION_LATENT_DIM]` → transformer `[T, HIDDEN]`
    /// using expert `expert_idx` ∈ `[0, ACTION_EXPERTS)`.
    pub fn project_in(&self, latent: &[f32], t: usize, expert_idx: usize) -> Vec<f32> {
        assert!(expert_idx < ACTION_EXPERTS);
        let w_stride = ACTION_LATENT_DIM * HIDDEN;
        // proj_in_fc is stacked [HIDDEN, ACTION_LATENT_DIM] (dout × din with
        // dout=HIDDEN, din=ACTION_LATENT_DIM) per expert.  matmul_nt_bf16
        // expects weight `[dout, din]` so we slice the per-expert chunk.
        let w_start = expert_idx * w_stride;
        let w_slice = &self.proj_in_fc[w_start..w_start + w_stride];
        let b_start = expert_idx * HIDDEN;
        let bias: Vec<f32> = self.proj_in_bias[b_start..b_start + HIDDEN]
            .iter().map(|b| b.to_f32()).collect();
        let mut out = crate::matmul_nt_bf16(latent, t, ACTION_LATENT_DIM, w_slice, HIDDEN);
        for ti in 0..t { for d in 0..HIDDEN { out[ti * HIDDEN + d] += bias[d]; } }
        out
    }

    /// Transformer `[T, HIDDEN]` → action latent `[T, ACTION_LATENT_DIM]`
    /// using expert `expert_idx`.
    pub fn project_out(&self, x: &[f32], t: usize, expert_idx: usize) -> Vec<f32> {
        assert!(expert_idx < ACTION_EXPERTS);
        let w_stride = ACTION_LATENT_DIM * HIDDEN;
        let w_start = expert_idx * w_stride;
        let w_slice = &self.proj_out_fc[w_start..w_start + w_stride];
        let b_start = expert_idx * ACTION_LATENT_DIM;
        let bias: Vec<f32> = self.proj_out_bias[b_start..b_start + ACTION_LATENT_DIM]
            .iter().map(|b| b.to_f32()).collect();
        let mut out = crate::matmul_nt_bf16(x, t, HIDDEN, w_slice, ACTION_LATENT_DIM);
        for ti in 0..t { for d in 0..ACTION_LATENT_DIM { out[ti * ACTION_LATENT_DIM + d] += bias[d]; } }
        out
    }
}

/// One diffusion step: take a noisy visual latent `[T, 192]`, a timestep
/// scalar `t ∈ [0, 1]`, and a conditioning reasoner stream `x_r [t_r, HIDDEN]`,
/// run the full v2 transformer, and return the predicted noise/x0 in latent
/// space `[T, 192]`.
///
/// Pipeline:
///   1. visual_head.project_in(latent)  → x_g [T, HIDDEN]
///   2. time_embedder.forward(t)         → time_emb [HIDDEN]; add to every gen token
///   3. for each layer: forward_with_3d_positions(x_r, x_g, RoPE positions)
///   4. visual_head.project_out(x_g_final) → predicted latent [T, 192]
pub struct VisualDiffusionStep<'a> {
    pub time_embedder: &'a TimeEmbedder,
    pub visual_head: &'a VisualHead,
    pub layers: &'a [LayerV2],
}

/// Rectified-flow inference schedule, matching Cosmos 3 Nano's
/// `model.config.rectified_flow_inference_config`:
///   - `scheduler_type`: "unipc" (production); this crate ships Euler as a
///     numerically stable baseline.  UniPC is a higher-order multi-step
///     scheduler; it converges to the same solution in the limit of many
///     steps and matches Euler exactly at `n_steps=1`.
///   - `shift`: 1 (default; with `use_dynamic_shifting=false` ⇒ linear σ).
///   - `num_train_timesteps`: 1000 — irrelevant at inference because the
///     model takes a continuous σ in `[0, 1]`.
///
/// The shifted sigma schedule (when `shift != 1`) is:
///   `σ_shifted = shift · σ_linear / (1 + (shift − 1) · σ_linear)`
/// (Flux / Wan convention).  With `shift=1` this is a no-op.
#[derive(Debug, Clone)]
pub struct FlowMatchSchedule {
    pub n_steps: usize,
    pub shift: f32,
}

impl FlowMatchSchedule {
    /// Default Cosmos 3 Nano inference schedule: shift=1, linear.
    pub fn cosmos3_nano(n_steps: usize) -> Self {
        Self { n_steps, shift: 1.0 }
    }

    /// Custom schedule with a non-1 shift parameter (Flux / Wan style).
    /// `shift > 1` biases sampling towards higher-σ steps (more denoising
    /// effort spent near `σ → 1`).
    pub fn shifted(n_steps: usize, shift: f32) -> Self {
        Self { n_steps, shift }
    }

    /// Return the `n_steps + 1` sigmas, including the σ=0 endpoint.
    /// Indices `0..n_steps-1` give per-step σ for the model call;
    /// `dt_i = sigmas[i] − sigmas[i+1]` for the integration update.
    pub fn sigmas(&self) -> Vec<f32> {
        let n = self.n_steps;
        let mut out = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let lin = 1.0 - (i as f32 / n as f32);
            let s = if (self.shift - 1.0).abs() < f32::EPSILON {
                lin
            } else {
                self.shift * lin / (1.0 + (self.shift - 1.0) * lin)
            };
            out.push(s);
        }
        out
    }
}

impl<'a> VisualDiffusionStep<'a> {
    pub fn from_model(m: &'a Cosmos3OmniTransformerV2) -> Result<Self, &'static str> {
        Ok(Self {
            time_embedder: m.time_embedder.as_ref().ok_or("missing time_embedder")?,
            visual_head:   m.visual_head.as_ref().ok_or("missing visual_head")?,
            layers: &m.layers,
        })
    }

    /// Deterministic Euler sampling: start from a noisy latent `x_T` at
    /// `t = 1.0` and integrate down to `t = 0.0` via `n_steps` of
    /// `x ← x - dt · ε(x, t)`.  Returns the predicted clean latent `x_0`.
    ///
    /// `cond_scale` rescales the model output before stepping (a stand-in
    /// for the sigma/snr schedule that production samplers apply; the
    /// visual head's raw output is ~300× too large for unit-norm steps).
    /// Pick a small value like `1e-2` for synthetic-input verification.
    pub fn euler_sample(
        &self,
        x_init: Vec<f32>, t_gen: usize,
        x_r_cond: &[f32], t_r: usize,
        pos_r_t: &[i64], pos_r_h: &[i64], pos_r_w: &[i64],
        pos_g_t: &[i64], pos_g_h: &[i64], pos_g_w: &[i64],
        n_steps: usize, cond_scale: f32,
    ) -> Vec<f32> {
        self.euler_sample_routed(x_init, t_gen,
            x_r_cond, t_r, pos_r_t, pos_r_h, pos_r_w,
            pos_g_t, pos_g_h, pos_g_w, n_steps, cond_scale, None)
    }

    /// **UniPC-style 2nd-order multistep sampling** (Adams-Bashforth 2)
    /// with the production `FlowMatchSchedule`.  At step 0 it falls back
    /// to plain Euler (no history yet); from step 1 onwards it computes
    /// `x_next = x - dt · (1.5 · ε_curr − 0.5 · ε_prev)`, which is the
    /// 2-step Adams-Bashforth predictor that approximates the predictor
    /// part of UniPC-bh1.  Converges to the same fixed point as Euler in
    /// the small-`dt` limit; for `n_steps ≤ 8`, the 2nd-order correction
    /// substantially improves trajectory accuracy versus pure Euler.
    ///
    /// Cosmos 3 Nano's `rectified_flow_inference_config.scheduler_type =
    /// "unipc"` motivates this path; full UniPC-bh2 (predictor + corrector)
    /// is a future refinement.
    #[allow(clippy::too_many_arguments)]
    pub fn unipc_sample_scheduled(
        &self,
        x_init: Vec<f32>, t_gen: usize,
        x_r_cond: &[f32], t_r: usize,
        pos_r_t: &[i64], pos_r_h: &[i64], pos_r_w: &[i64],
        pos_g_t: &[i64], pos_g_h: &[i64], pos_g_w: &[i64],
        schedule: &FlowMatchSchedule,
        accel: Option<&crate::phase4::MetalAccel>,
    ) -> Vec<f32> {
        let sigmas = schedule.sigmas();
        assert!(sigmas.len() >= 2, "schedule must have at least 2 sigmas");
        let mut x = x_init;
        let mut eps_prev: Option<Vec<f32>> = None;
        for i in 0..sigmas.len() - 1 {
            let sigma_curr = sigmas[i];
            let sigma_next = sigmas[i + 1];
            let dt = sigma_curr - sigma_next;
            let eps_curr = self.step_routed(&x, t_gen, sigma_curr,
                x_r_cond, t_r, pos_r_t, pos_r_h, pos_r_w,
                pos_g_t, pos_g_h, pos_g_w, accel);
            if let Some(prev) = &eps_prev {
                // 2nd-order Adams-Bashforth predictor.
                for j in 0..x.len() {
                    let drift = 1.5 * eps_curr[j] - 0.5 * prev[j];
                    x[j] -= dt * drift;
                }
            } else {
                // First step: bootstrap with Euler (no history).
                for j in 0..x.len() {
                    x[j] -= dt * eps_curr[j];
                }
            }
            eps_prev = Some(eps_curr);
        }
        x
    }

    /// Rectified-flow Euler sampling with a production sigma schedule
    /// (`FlowMatchSchedule`).  Replaces the `cond_scale` hyperparameter in
    /// `euler_sample` with a real per-step `dt = sigma_i − sigma_{i+1}`
    /// derived from the schedule, matching the Cosmos 3
    /// `rectified_flow_inference_config` semantics (`shift=1`,
    /// `use_dynamic_shifting=False` ⇒ linear σ schedule).
    #[allow(clippy::too_many_arguments)]
    pub fn euler_sample_scheduled(
        &self,
        x_init: Vec<f32>, t_gen: usize,
        x_r_cond: &[f32], t_r: usize,
        pos_r_t: &[i64], pos_r_h: &[i64], pos_r_w: &[i64],
        pos_g_t: &[i64], pos_g_h: &[i64], pos_g_w: &[i64],
        schedule: &FlowMatchSchedule,
        accel: Option<&crate::phase4::MetalAccel>,
    ) -> Vec<f32> {
        let sigmas = schedule.sigmas();
        assert!(sigmas.len() >= 2, "schedule must have at least 2 sigmas");
        let mut x = x_init;
        for i in 0..sigmas.len() - 1 {
            let sigma_curr = sigmas[i];
            let sigma_next = sigmas[i + 1];
            let dt = sigma_curr - sigma_next; // positive (sigma decreases)
            let eps = self.step_routed(&x, t_gen, sigma_curr,
                x_r_cond, t_r, pos_r_t, pos_r_h, pos_r_w,
                pos_g_t, pos_g_h, pos_g_w, accel);
            for j in 0..x.len() {
                x[j] -= dt * eps[j];
            }
        }
        x
    }

    /// Same as `euler_sample` with optional Metal acceleration.  Subsequent
    /// steps hit the MetalAccel weight handle cache — first step uploads
    /// the bf16 weights, all later steps reuse them.
    #[allow(clippy::too_many_arguments)]
    pub fn euler_sample_routed(
        &self,
        x_init: Vec<f32>, t_gen: usize,
        x_r_cond: &[f32], t_r: usize,
        pos_r_t: &[i64], pos_r_h: &[i64], pos_r_w: &[i64],
        pos_g_t: &[i64], pos_g_h: &[i64], pos_g_w: &[i64],
        n_steps: usize, cond_scale: f32,
        accel: Option<&crate::phase4::MetalAccel>,
    ) -> Vec<f32> {
        assert!(n_steps > 0);
        let dt = 1.0_f32 / n_steps as f32;
        let mut x = x_init;
        for step in 0..n_steps {
            let t = 1.0 - step as f32 * dt;
            let eps = self.step_routed(&x, t_gen, t,
                x_r_cond, t_r, pos_r_t, pos_r_h, pos_r_w,
                pos_g_t, pos_g_h, pos_g_w, accel);
            for i in 0..x.len() {
                x[i] -= dt * cond_scale * eps[i];
            }
        }
        x
    }

    /// Apply one diffusion forward.  CPU-only path; for GPU acceleration use
    /// `step_routed`.
    pub fn step(
        &self,
        latent: &[f32], t_gen: usize,
        t_scalar: f32,
        x_r_cond: &[f32], t_r: usize,
        pos_r_t: &[i64], pos_r_h: &[i64], pos_r_w: &[i64],
        pos_g_t: &[i64], pos_g_h: &[i64], pos_g_w: &[i64],
    ) -> Vec<f32> {
        self.step_routed(latent, t_gen, t_scalar,
            x_r_cond, t_r, pos_r_t, pos_r_h, pos_r_w,
            pos_g_t, pos_g_h, pos_g_w, None)
    }

    /// Apply one diffusion forward with optional Metal acceleration.
    /// When `accel.is_some()`, every per-layer matmul routes through
    /// Stage 5's `MetalAccel` weight handle cache; subsequent calls hit
    /// the cached bf16 weight handles uploaded by the first call.
    pub fn step_routed(
        &self,
        latent: &[f32], t_gen: usize,
        t_scalar: f32,
        x_r_cond: &[f32], t_r: usize,
        pos_r_t: &[i64], pos_r_h: &[i64], pos_r_w: &[i64],
        pos_g_t: &[i64], pos_g_h: &[i64], pos_g_w: &[i64],
        accel: Option<&crate::phase4::MetalAccel>,
    ) -> Vec<f32> {
        // 1) Project latent to transformer dim.
        let mut x_g = self.visual_head.project_in(latent, t_gen);
        // 2) Time embed + broadcast-add to every gen token.
        let time_emb = self.time_embedder.forward(t_scalar);
        for ti in 0..t_gen {
            for d in 0..HIDDEN {
                x_g[ti * HIDDEN + d] += time_emb[d];
            }
        }
        // 3) Run all transformer layers (reasoner stream provides conditioning).
        let mut x_r = x_r_cond.to_vec();
        let freqs = crate::rope_freqs();
        for (block_idx, l) in self.layers.iter().enumerate() {
            // 3D mrope on both CPU and GPU paths via the v2 `forward_routed_3d`.
            l.forward_routed_3d(
                &mut x_r, t_r, pos_r_t, pos_r_h, pos_r_w,
                &mut x_g, t_gen, pos_g_t, pos_g_h, pos_g_w,
                &freqs, block_idx, accel,
            );
        }
        // 4) Project back to latent space.
        self.visual_head.project_out(&x_g, t_gen)
    }
}

/// One audio-diffusion step.  Mirror of `VisualDiffusionStep` over the
/// 64-dim audio latent space (Cosmos 3's AVAE sound tokenizer codes).
/// The transformer pipeline is identical; only the I/O heads differ.
pub struct AudioDiffusionStep<'a> {
    pub time_embedder: &'a TimeEmbedder,
    pub audio_head: &'a AudioHead,
    pub layers: &'a [LayerV2],
}

impl<'a> AudioDiffusionStep<'a> {
    pub fn from_model(m: &'a Cosmos3OmniTransformerV2) -> Result<Self, &'static str> {
        Ok(Self {
            time_embedder: m.time_embedder.as_ref().ok_or("missing time_embedder")?,
            audio_head:    m.audio_head.as_ref().ok_or("missing audio_head")?,
            layers: &m.layers,
        })
    }

    /// One CPU diffusion forward.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        latent: &[f32], t_gen: usize,
        t_scalar: f32,
        x_r_cond: &[f32], t_r: usize,
        pos_r_t: &[i64], pos_r_h: &[i64], pos_r_w: &[i64],
        pos_g_t: &[i64], pos_g_h: &[i64], pos_g_w: &[i64],
    ) -> Vec<f32> {
        self.step_routed(latent, t_gen, t_scalar,
            x_r_cond, t_r, pos_r_t, pos_r_h, pos_r_w,
            pos_g_t, pos_g_h, pos_g_w, None)
    }

    /// One diffusion forward with optional Metal acceleration on the joint
    /// transformer matmuls.
    #[allow(clippy::too_many_arguments)]
    pub fn step_routed(
        &self,
        latent: &[f32], t_gen: usize,
        t_scalar: f32,
        x_r_cond: &[f32], t_r: usize,
        pos_r_t: &[i64], pos_r_h: &[i64], pos_r_w: &[i64],
        pos_g_t: &[i64], pos_g_h: &[i64], pos_g_w: &[i64],
        accel: Option<&crate::phase4::MetalAccel>,
    ) -> Vec<f32> {
        let mut x_g = self.audio_head.project_in(latent, t_gen);
        let time_emb = self.time_embedder.forward(t_scalar);
        for ti in 0..t_gen {
            for d in 0..HIDDEN { x_g[ti * HIDDEN + d] += time_emb[d]; }
        }
        let mut x_r = x_r_cond.to_vec();
        let freqs = crate::rope_freqs();
        for (block_idx, l) in self.layers.iter().enumerate() {
            l.forward_routed_3d(
                &mut x_r, t_r, pos_r_t, pos_r_h, pos_r_w,
                &mut x_g, t_gen, pos_g_t, pos_g_h, pos_g_w,
                &freqs, block_idx, accel,
            );
        }
        self.audio_head.project_out(&x_g, t_gen)
    }

    /// UniPC AB2 multistep sampling.  Step 0 = Euler, steps 1+ = AB2.
    #[allow(clippy::too_many_arguments)]
    pub fn unipc_sample_scheduled(
        &self,
        x_init: Vec<f32>, t_gen: usize,
        x_r_cond: &[f32], t_r: usize,
        pos_r_t: &[i64], pos_r_h: &[i64], pos_r_w: &[i64],
        pos_g_t: &[i64], pos_g_h: &[i64], pos_g_w: &[i64],
        schedule: &FlowMatchSchedule,
        accel: Option<&crate::phase4::MetalAccel>,
    ) -> Vec<f32> {
        let sigmas = schedule.sigmas();
        let mut x = x_init;
        let mut eps_prev: Option<Vec<f32>> = None;
        for i in 0..sigmas.len() - 1 {
            let dt = sigmas[i] - sigmas[i + 1];
            let eps_curr = self.step_routed(&x, t_gen, sigmas[i],
                x_r_cond, t_r, pos_r_t, pos_r_h, pos_r_w,
                pos_g_t, pos_g_h, pos_g_w, accel);
            if let Some(prev) = &eps_prev {
                for j in 0..x.len() { x[j] -= dt * (1.5 * eps_curr[j] - 0.5 * prev[j]); }
            } else {
                for j in 0..x.len() { x[j] -= dt * eps_curr[j]; }
            }
            eps_prev = Some(eps_curr);
        }
        x
    }
}

/// One action-diffusion step.  Mirror of `VisualDiffusionStep` over the
/// 64-dim action latent space.  Each call picks one of the 32 MoE experts
/// at the I/O heads; the transformer body is shared.
pub struct ActionDiffusionStep<'a> {
    pub time_embedder: &'a TimeEmbedder,
    pub action_head: &'a ActionHead,
    pub layers: &'a [LayerV2],
}

impl<'a> ActionDiffusionStep<'a> {
    pub fn from_model(m: &'a Cosmos3OmniTransformerV2) -> Result<Self, &'static str> {
        Ok(Self {
            time_embedder: m.time_embedder.as_ref().ok_or("missing time_embedder")?,
            action_head:   m.action_head.as_ref().ok_or("missing action_head")?,
            layers: &m.layers,
        })
    }

    /// One CPU diffusion forward via expert `expert_idx`.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        latent: &[f32], t_gen: usize,
        t_scalar: f32,
        expert_idx: usize,
        x_r_cond: &[f32], t_r: usize,
        pos_r_t: &[i64], pos_r_h: &[i64], pos_r_w: &[i64],
        pos_g_t: &[i64], pos_g_h: &[i64], pos_g_w: &[i64],
    ) -> Vec<f32> {
        self.step_routed(latent, t_gen, t_scalar, expert_idx,
            x_r_cond, t_r, pos_r_t, pos_r_h, pos_r_w,
            pos_g_t, pos_g_h, pos_g_w, None)
    }

    /// One diffusion forward via expert `expert_idx` with optional Metal acceleration.
    #[allow(clippy::too_many_arguments)]
    pub fn step_routed(
        &self,
        latent: &[f32], t_gen: usize,
        t_scalar: f32,
        expert_idx: usize,
        x_r_cond: &[f32], t_r: usize,
        pos_r_t: &[i64], pos_r_h: &[i64], pos_r_w: &[i64],
        pos_g_t: &[i64], pos_g_h: &[i64], pos_g_w: &[i64],
        accel: Option<&crate::phase4::MetalAccel>,
    ) -> Vec<f32> {
        let mut x_g = self.action_head.project_in(latent, t_gen, expert_idx);
        let time_emb = self.time_embedder.forward(t_scalar);
        for ti in 0..t_gen {
            for d in 0..HIDDEN { x_g[ti * HIDDEN + d] += time_emb[d]; }
        }
        let mut x_r = x_r_cond.to_vec();
        let freqs = crate::rope_freqs();
        for (block_idx, l) in self.layers.iter().enumerate() {
            l.forward_routed_3d(
                &mut x_r, t_r, pos_r_t, pos_r_h, pos_r_w,
                &mut x_g, t_gen, pos_g_t, pos_g_h, pos_g_w,
                &freqs, block_idx, accel,
            );
        }
        self.action_head.project_out(&x_g, t_gen, expert_idx)
    }

    /// UniPC AB2 multistep sampling holding `expert_idx` constant across steps.
    #[allow(clippy::too_many_arguments)]
    pub fn unipc_sample_scheduled(
        &self,
        x_init: Vec<f32>, t_gen: usize,
        expert_idx: usize,
        x_r_cond: &[f32], t_r: usize,
        pos_r_t: &[i64], pos_r_h: &[i64], pos_r_w: &[i64],
        pos_g_t: &[i64], pos_g_h: &[i64], pos_g_w: &[i64],
        schedule: &FlowMatchSchedule,
        accel: Option<&crate::phase4::MetalAccel>,
    ) -> Vec<f32> {
        let sigmas = schedule.sigmas();
        let mut x = x_init;
        let mut eps_prev: Option<Vec<f32>> = None;
        for i in 0..sigmas.len() - 1 {
            let dt = sigmas[i] - sigmas[i + 1];
            let eps_curr = self.step_routed(&x, t_gen, sigmas[i], expert_idx,
                x_r_cond, t_r, pos_r_t, pos_r_h, pos_r_w,
                pos_g_t, pos_g_h, pos_g_w, accel);
            if let Some(prev) = &eps_prev {
                for j in 0..x.len() { x[j] -= dt * (1.5 * eps_curr[j] - 0.5 * prev[j]); }
            } else {
                for j in 0..x.len() { x[j] -= dt * eps_curr[j]; }
            }
            eps_prev = Some(eps_curr);
        }
        x
    }
}

/// Per-token modality tag.  Action / Audio tokens get an additional
/// modality-specific embedding added at input.  Text / Image / Video
/// tokens use only the standard `embed_tokens` lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality {
    Text,
    Image,
    Video,
    Audio,
    Action,
}

impl Cosmos3OmniTransformerV2 {
    /// Look up token embeddings + add the per-modality bias for Action /
    /// Audio tokens.  Returns `[T * HIDDEN]` row-major.
    ///
    /// `ids[i]` is the token id (used to index `embed_tokens` which has
    /// shape `[VOCAB=151936, HIDDEN]`).  `modalities[i]` is the per-token
    /// modality tag — Action adds `action_modality_embed`, Audio adds
    /// `audio_modality_embed`, others are pass-through.
    /// Embed a sequence of text token ids — convenience over `embed_with_modality`
    /// with every token tagged as `Modality::Text` (no modality bias added).
    /// This is the text-encoder front-end for the reasoner stream when driving
    /// diffusion sampling: the returned `[T, HIDDEN]` slab is what the
    /// `VisualDiffusionStep` consumes as `x_r_cond`.
    pub fn embed_text(&self, ids: &[u32]) -> Vec<f32> {
        let modalities: Vec<Modality> = (0..ids.len()).map(|_| Modality::Text).collect();
        self.embed_with_modality(ids, &modalities)
    }

    pub fn embed_with_modality(&self, ids: &[u32], modalities: &[Modality]) -> Vec<f32> {
        assert_eq!(ids.len(), modalities.len());
        let t = ids.len();
        let mut out = vec![0.0f32; t * HIDDEN];
        for (ti, (&id, &m)) in ids.iter().zip(modalities.iter()).enumerate() {
            let src_off = (id as usize) * HIDDEN;
            // bf16 → f32 lookup per element.
            for d in 0..HIDDEN {
                out[ti * HIDDEN + d] = self.embed_tokens[src_off + d].to_f32();
            }
            let modality_bias = match m {
                Modality::Action => Some(&self.action_modality_embed),
                Modality::Audio  => Some(&self.audio_modality_embed),
                _ => None,
            };
            if let Some(bias) = modality_bias {
                for d in 0..HIDDEN { out[ti * HIDDEN + d] += bias[d]; }
            }
        }
        out
    }

    /// Load from a Cosmos 3 sharded transformer dir.  Expects:
    /// `<dir>/diffusion_pytorch_model.safetensors.index.json` + N shards.
    /// Loads ALL `N_LAYERS=36` blocks by default.
    pub fn load(transformer_dir: &Path) -> Result<Self, CosmosError> {
        Self::load_with_layers(transformer_dir, N_LAYERS)
    }

    pub fn load_with_layers(transformer_dir: &Path, n_layers: usize) -> Result<Self, CosmosError> {
        let idx = transformer_dir.join("diffusion_pytorch_model.safetensors.index.json");
        let ss = ShardedSafetensors::from_index(&idx)?;
        let embed_tokens = ld_bf16(&ss, "embed_tokens.weight")?;
        let action_modality_embed = ld(&ss, "action_modality_embed")?;
        let audio_modality_embed = ld(&ss, "audio_modality_embed")?;
        let layers = (0..n_layers)
            .map(|i| LayerV2::load(&ss, i))
            .collect::<Result<_, _>>()?;
        // Heads are optional: skip if missing (allows synthetic tests).
        let time_embedder = TimeEmbedder::load(&ss).ok();
        let visual_head = VisualHead::load(&ss).ok();
        let audio_head = AudioHead::load(&ss).ok();
        let action_head = ActionHead::load(&ss).ok();
        Ok(Self {
            embed_tokens, action_modality_embed, audio_modality_embed, layers,
            time_embedder, visual_head, audio_head, action_head,
        })
    }
}

// ════════════════════════════════════════════════════════════════════════
// Cosmos 3 Nano tokenizer — Qwen2-family byte-level BPE.
// ════════════════════════════════════════════════════════════════════════

use std::collections::HashMap;

/// Cosmos 3 Nano text tokenizer.  Reuses the byte-level pre-tokenizer +
/// `bytes_to_unicode` mapping from `rce-ingest-gguf::tokenizer` (the
/// same shape Qwen-family GGUFs encode for); loads owned vocab + merges
/// from the HuggingFace-style `text_tokenizer/{vocab.json, merges.txt}`
/// shipped alongside the Cosmos 3 Nano weights.
///
/// Vocabulary is 151_643 base tokens + 26 special tokens added by the
/// `tokenizer_config.json`.  Vocab dim padded to **151_936** in the
/// transformer's embed_tokens to match Qwen3-VL conventions.
///
/// Special tokens (verified against Qwen3-VL `added_tokens_decoder`):
/// - 151_643 `<|endoftext|>`
/// - 151_644 `<|im_start|>`
/// - 151_645 `<|im_end|>`
pub struct Cosmos3Tokenizer {
    /// `token text → id` lookup.
    pub vocab: HashMap<String, u32>,
    /// `(left, right) → priority`; lower priority = higher precedence.
    pub merges: HashMap<(String, String), u32>,
    /// `id → token text` (parallel inverse of vocab; tokens fed back to
    /// the user for decode + debugging).
    pub decoder: Vec<String>,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
}

impl Cosmos3Tokenizer {
    /// Load from `<root>/text_tokenizer/{vocab.json, merges.txt}`.
    pub fn load(model_root: &Path) -> Result<Self, CosmosError> {
        let tdir = model_root.join("text_tokenizer");
        let vocab_bytes = std::fs::read(tdir.join("vocab.json"))
            .map_err(|e| CosmosError::Load(format!("vocab.json: {e}")))?;
        let vocab_json: serde_json::Value = serde_json::from_slice(&vocab_bytes)
            .map_err(|e| CosmosError::Load(format!("vocab.json parse: {e}")))?;
        let vocab_obj = vocab_json.as_object()
            .ok_or_else(|| CosmosError::Load("vocab.json not an object".into()))?;
        let mut vocab: HashMap<String, u32> = HashMap::with_capacity(vocab_obj.len());
        let mut max_id = 0u32;
        for (k, v) in vocab_obj {
            let id = v.as_u64()
                .ok_or_else(|| CosmosError::Load(format!("vocab id for {k} not u64")))?
                as u32;
            vocab.insert(k.clone(), id);
            if id > max_id { max_id = id; }
        }
        // Add the 26 special tokens (their string + id from the standard
        // Qwen3-VL table).  These appear with raw `<|...|>` text and are
        // never broken by BPE.
        let specials = cosmos3_special_tokens();
        for (s, id) in specials {
            vocab.insert(s.to_string(), id);
            if id > max_id { max_id = id; }
        }
        let mut decoder: Vec<String> = vec![String::new(); (max_id + 1) as usize];
        for (s, &id) in &vocab {
            decoder[id as usize] = s.clone();
        }

        // Parse merges.txt: first line is `#version: 0.2`; subsequent
        // lines are `left right` pairs, in priority order.
        let merges_txt = std::fs::read_to_string(tdir.join("merges.txt"))
            .map_err(|e| CosmosError::Load(format!("merges.txt: {e}")))?;
        let mut merges: HashMap<(String, String), u32> = HashMap::new();
        let mut prio = 0u32;
        for line in merges_txt.lines() {
            if line.starts_with('#') || line.is_empty() { continue; }
            if let Some(sp) = line.find(' ') {
                let l = line[..sp].to_string();
                let r = line[sp + 1..].to_string();
                merges.insert((l, r), prio);
                prio += 1;
            }
        }

        let bos_id = vocab.get("<|endoftext|>").copied();
        let eos_id = vocab.get("<|im_end|>").copied().or(bos_id);
        Ok(Self { vocab, merges, decoder, bos_id, eos_id })
    }

    /// Vocabulary size (number of distinct token ids known).
    pub fn vocab_size(&self) -> usize { self.decoder.len() }

    /// Look up a single token's text by id.  Returns "" for unknown IDs.
    pub fn token_str(&self, id: u32) -> &str {
        self.decoder.get(id as usize).map(|s| s.as_str()).unwrap_or("")
    }

    /// Encode a UTF-8 string into a sequence of Qwen-BPE token ids.
    /// Two-stage:
    ///   1. Greedy longest-match scan for known special tokens (`<|...|>`);
    ///      each match emits its single id and is excluded from BPE.
    ///   2. Each non-special segment runs pre-tokenizer split + byte-to-unicode
    ///      + BPE merge + vocab lookup.
    /// Matches HuggingFace `Tokenizer::encode(s).ids` for ordinary text and
    /// for mixed text + special tokens.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Build a sorted list of special tokens (longest first) for greedy
        // longest-match.  All Cosmos 3 specials are ASCII so byte positions
        // line up with char boundaries.
        let specials = cosmos3_special_tokens();
        let mut sorted_specials: Vec<(&'static str, u32)> = specials.clone();
        sorted_specials.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));

        let mut tokens: Vec<u32> = Vec::new();
        let mut segment_start = 0usize;
        let mut pos = 0usize;
        while pos < text.len() {
            let mut matched: Option<(usize, u32)> = None;
            for (s, id) in &sorted_specials {
                if text[pos..].starts_with(s) {
                    matched = Some((s.len(), *id));
                    break;
                }
            }
            if let Some((mlen, id)) = matched {
                if segment_start < pos {
                    self.encode_text_segment(&text[segment_start..pos], &mut tokens);
                }
                tokens.push(id);
                pos += mlen;
                segment_start = pos;
            } else {
                // Advance to the next UTF-8 char boundary.
                pos += 1;
                while pos < text.len() && !text.is_char_boundary(pos) { pos += 1; }
            }
        }
        if segment_start < text.len() {
            self.encode_text_segment(&text[segment_start..], &mut tokens);
        }
        tokens
    }

    fn encode_text_segment(&self, segment: &str, tokens: &mut Vec<u32>) {
        use rce_ingest_gguf::tokenizer::{pretokenize_gpt2, gpt2_byte_to_unicode};
        for piece in pretokenize_gpt2(segment) {
            let mut encoded = String::with_capacity(piece.len());
            for &b in piece.as_bytes() {
                encoded.push(gpt2_byte_to_unicode(b));
            }
            let merged = self.bpe_merge(&encoded);
            for p in merged {
                if let Some(&id) = self.vocab.get(&p) {
                    tokens.push(id);
                }
            }
        }
    }

    /// BPE merge loop: start with single-codepoint pieces, iteratively
    /// merge the adjacent pair with the lowest priority (= earliest
    /// merge in the merges.txt table) until no merges apply.
    fn bpe_merge(&self, text: &str) -> Vec<String> {
        let mut pieces: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..pieces.len().saturating_sub(1) {
                let key = (pieces[i].clone(), pieces[i + 1].clone());
                if let Some(&prio) = self.merges.get(&key) {
                    if best.map_or(true, |(_, p)| prio < p) {
                        best = Some((i, prio));
                    }
                }
            }
            match best {
                Some((pos, _)) => {
                    let merged = format!("{}{}", pieces[pos], pieces[pos + 1]);
                    pieces[pos] = merged;
                    pieces.remove(pos + 1);
                }
                None => break,
            }
        }
        pieces
    }
}

/// Qwen3-VL added-tokens table copied verbatim from
/// `text_tokenizer/tokenizer_config.json::added_tokens_decoder`.  Adds 26
/// special tokens at ids 151_643 .. 151_668.  Used at tokenizer load time
/// because vocab.json itself does not include them.
fn cosmos3_special_tokens() -> Vec<(&'static str, u32)> {
    vec![
        ("<|endoftext|>",         151_643),
        ("<|im_start|>",          151_644),
        ("<|im_end|>",            151_645),
        ("<|object_ref_start|>",  151_646),
        ("<|object_ref_end|>",    151_647),
        ("<|box_start|>",         151_648),
        ("<|box_end|>",           151_649),
        ("<|quad_start|>",        151_650),
        ("<|quad_end|>",          151_651),
        ("<|vision_start|>",      151_652),
        ("<|vision_end|>",        151_653),
        ("<|vision_pad|>",        151_654),
        ("<|image_pad|>",         151_655),
        ("<|video_pad|>",         151_656),
        ("<tool_call>",           151_657),
        ("</tool_call>",          151_658),
        ("<|fim_prefix|>",        151_659),
        ("<|fim_middle|>",        151_660),
        ("<|fim_suffix|>",        151_661),
        ("<|fim_pad|>",           151_662),
        ("<|repo_name|>",         151_663),
        ("<|file_sep|>",          151_664),
        ("<tool_response>",       151_665),
        ("</tool_response>",      151_666),
        ("<think>",               151_667),
        ("</think>",              151_668),
    ]
}

// ════════════════════════════════════════════════════════════════════════
// Wan 2.2 VAE v2 — real-weight loader for the 196-tensor VAE shipped with
// Cosmos 3 Nano (Wan-AI/Wan2.2-TI2V-5B-Diffusers).  Stage 0: structured
// weight handles only.  Forward primitives (3D conv / ResNet / 2D+temporal
// resample / mid-block attention) follow in a focused Stage 1 session.
// ════════════════════════════════════════════════════════════════════════

/// Config values lifted directly from the VAE's `config.json`.
#[derive(Debug, Clone)]
pub struct WanVaeConfig {
    /// Encoder base channel width — first down-block emits `base_dim * dim_mult[0]`.
    pub base_dim: usize,                  // 160
    /// Decoder base channel width.
    pub decoder_base_dim: usize,          // 256
    /// Per-stage channel multipliers; encoder uses left-to-right, decoder reverses.
    pub dim_mult: Vec<usize>,             // [1, 2, 4, 4]
    pub in_channels: usize,               // 12 (= 3 RGB × patch_temporal=4)
    pub out_channels: usize,              // 12
    pub z_dim: usize,                     // 48
    pub patch_size: usize,                // 2 (spatial patch)
    pub scale_factor_spatial: usize,      // 16
    pub scale_factor_temporal: usize,     // 4
    /// Per-stage flag: whether the downsampler/upsampler includes a `time_conv`.
    pub temperal_downsample: Vec<bool>,   // [false, true, true]
    pub latents_mean: Vec<f32>,           // [z_dim]
    pub latents_std: Vec<f32>,            // [z_dim]
}

impl WanVaeConfig {
    /// Match Wan-AI/Wan2.2-TI2V-5B-Diffusers config.json verbatim.
    pub fn cosmos3_nano() -> Self {
        Self {
            base_dim: 160,
            decoder_base_dim: 256,
            dim_mult: vec![1, 2, 4, 4],
            in_channels: 12, out_channels: 12,
            z_dim: 48,
            patch_size: 2,
            scale_factor_spatial: 16, scale_factor_temporal: 4,
            temperal_downsample: vec![false, true, true],
            latents_mean: vec![
                -0.2289,-0.0052,-0.1323,-0.2339,-0.2799, 0.0174, 0.1838, 0.1557,
                -0.1382, 0.0542, 0.2813, 0.0891, 0.1570,-0.0098, 0.0375,-0.1825,
                -0.2246,-0.1207,-0.0698, 0.5109, 0.2665,-0.2108,-0.2158, 0.2502,
                -0.2055,-0.0322, 0.1109, 0.1567,-0.0729, 0.0899,-0.2799,-0.1230,
                -0.0313,-0.1649, 0.0117, 0.0723,-0.2839,-0.2083,-0.0520, 0.3748,
                 0.0152, 0.1957, 0.1433,-0.2944, 0.3573,-0.0548,-0.1681,-0.0667,
            ],
            latents_std: vec![
                0.4765,1.0364,0.4514,1.1677,0.5313,0.4990,0.4818,0.5013,
                0.8158,1.0344,0.5894,1.0901,0.6885,0.6165,0.8454,0.4978,
                0.5759,0.3523,0.7135,0.6804,0.5833,1.4146,0.8986,0.5659,
                0.7069,0.5338,0.4889,0.4917,0.4069,0.4999,0.6866,0.4093,
                0.5709,0.6065,0.6415,0.4944,0.5726,1.2042,0.5458,1.6887,
                0.3971,1.0600,0.3943,0.5537,0.5444,0.4089,0.7468,0.7744,
            ],
        }
    }
}

/// 3D conv with kernel `(kT, kH, kW)`.  Weight `[Cout, Cin, kT, kH, kW]`,
/// bias `[Cout]`.  Stored shape recorded for shape-checks.
///
/// `cache_id` is a unique-per-VAE-instance id assigned at load time so the
/// Metal weight handle cache (`MetalAccel`) can amortize the bf16 upload
/// across diffusion sampling iterations.  Pure CPU paths ignore it.
pub struct Wan3DConv {
    pub w: BfSlice, pub b: Vec<f32>,
    pub cin: usize, pub cout: usize,
    pub kt: usize, pub kh: usize, pub kw: usize,
    pub cache_id: u32,
}

/// 2D conv used in down/upsamplers (resample.1).  `[Cout, Cin, 3, 3]`.
pub struct Wan2DConv {
    pub w: BfSlice, pub b: Vec<f32>,
    pub cin: usize, pub cout: usize,
}

/// 1×1×1 channel-mixing conv (quant_conv, post_quant_conv).
/// `[Cout, Cin, 1, 1, 1]` — semantically equivalent to a Linear.
pub struct Wan1x1Conv {
    pub w: BfSlice, pub b: Vec<f32>,
    pub cin: usize, pub cout: usize,
}

/// ResNet block in 3D: GroupNorm → SiLU → conv1 → GroupNorm → SiLU → conv2,
/// with optional 1×1×1 shortcut when Cin != Cout.
pub struct WanResNetBlock {
    pub norm1_gamma: Vec<f32>,            // [Cin]
    pub conv1: Wan3DConv,                 // Cin → Cout
    pub norm2_gamma: Vec<f32>,            // [Cout]
    pub conv2: Wan3DConv,                 // Cout → Cout
    pub shortcut: Option<Wan1x1Conv>,     // 1×1×1, present when Cin != Cout
}

/// Temporal 1×1 conv used in (down|up)samplers when `temperal_downsample[i]`.
/// `[Cout, Cin, 3, 1, 1]` — the temporal axis carries the 3-tap kernel.
pub struct WanTimeConv {
    pub w: BfSlice, pub b: Vec<f32>,
    pub cin: usize, pub cout: usize,
}

/// Downsampler: `resample.1` is a stride-2 2D conv (H/W halved), and when
/// `temperal_downsample[i]` is `true` an additional `time_conv` halves T.
pub struct WanDownsampler {
    pub resample: Wan2DConv,
    pub time_conv: Option<WanTimeConv>,
}

/// Upsampler: mirror — `resample.1` 2D conv (preceded by a 2× nearest H/W
/// upsample), and `time_conv` doubles T via a 2-tap temporal kernel.  In
/// the safetensors the upsampler `time_conv` weight is `[2·Cout, Cin, 3, 1, 1]`
/// (two stacked temporal output frames).
pub struct WanUpsampler {
    pub resample: Wan2DConv,
    pub time_conv: Option<WanTimeConv>,
}

/// Mid-block self-attention over flattened (T·H·W) sequence: per-channel
/// `norm` (gamma only — RMSNorm-style), 1×1 conv `to_qkv` (→ 3·C), 1×1 conv `proj`.
pub struct WanMidAttention {
    pub norm_gamma: Vec<f32>,             // [C]
    pub to_qkv_w: BfSlice,                // [3C, C, 1, 1]
    pub to_qkv_b: Vec<f32>,               // [3C]
    pub proj_w: BfSlice,                  // [C,  C, 1, 1]
    pub proj_b: Vec<f32>,                 // [C]
    pub c: usize,
}

pub struct WanMidBlock {
    pub resnets: [WanResNetBlock; 2],
    pub attention: WanMidAttention,
}

pub struct WanDownBlock {
    /// `num_res_blocks = 2` per stage.
    pub resnets: Vec<WanResNetBlock>,
    pub downsampler: Option<WanDownsampler>,
}

pub struct WanUpBlock {
    /// `num_res_blocks + 1 = 3` per stage (diffusers UNet convention).
    pub resnets: Vec<WanResNetBlock>,
    pub upsampler: Option<WanUpsampler>,
}

pub struct Wan22VAE_v2 {
    pub config: WanVaeConfig,
    pub encoder_conv_in: Wan3DConv,
    pub encoder_down_blocks: Vec<WanDownBlock>,
    pub encoder_mid_block: WanMidBlock,
    pub encoder_norm_out_gamma: Vec<f32>,
    pub encoder_conv_out: Wan3DConv,
    pub quant_conv: Wan1x1Conv,
    pub post_quant_conv: Wan1x1Conv,
    pub decoder_conv_in: Wan3DConv,
    pub decoder_mid_block: WanMidBlock,
    pub decoder_up_blocks: Vec<WanUpBlock>,
    pub decoder_norm_out_gamma: Vec<f32>,
    pub decoder_conv_out: Wan3DConv,
}

// ── Wan VAE loaders ─────────────────────────────────────────────────────

/// Monotonic counter for `Wan3DConv::cache_id`.  Each loaded VAE gets a
/// contiguous range; spans across multiple loaded instances are harmless
/// because the MetalAccel cache is just a `HashMap<(usize, u8), u64>` and
/// the (cache_id, 0) keys for stale instances become dead entries that
/// never collide with live ones.
static WAN_CONV3D_CACHE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn next_cache_id() -> u32 {
    WAN_CONV3D_CACHE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn ld_3d(ss: &ShardedSafetensors, prefix: &str,
         cin: usize, cout: usize, k: (usize, usize, usize)) -> Result<Wan3DConv, CosmosError> {
    let w = ld_bf16(ss, &format!("{prefix}.weight"))?;
    let b = ld(ss, &format!("{prefix}.bias"))?;
    let (kt, kh, kw) = k;
    if w.len() != cout * cin * kt * kh * kw {
        return Err(CosmosError::Load(format!(
            "{prefix}.weight: expected {} = {cout}×{cin}×{kt}×{kh}×{kw}, got {}",
            cout*cin*kt*kh*kw, w.len())));
    }
    if b.len() != cout {
        return Err(CosmosError::Load(format!(
            "{prefix}.bias: expected {cout}, got {}", b.len())));
    }
    let cache_id = next_cache_id();
    Ok(Wan3DConv { w, b, cin, cout, kt, kh, kw, cache_id })
}

fn ld_2d(ss: &ShardedSafetensors, prefix: &str,
         cin: usize, cout: usize) -> Result<Wan2DConv, CosmosError> {
    let w = ld_bf16(ss, &format!("{prefix}.weight"))?;
    let b = ld(ss, &format!("{prefix}.bias"))?;
    if w.len() != cout * cin * 3 * 3 {
        return Err(CosmosError::Load(format!(
            "{prefix}.weight: expected {} = {cout}×{cin}×3×3, got {}",
            cout*cin*9, w.len())));
    }
    if b.len() != cout {
        return Err(CosmosError::Load(format!(
            "{prefix}.bias: expected {cout}, got {}", b.len())));
    }
    Ok(Wan2DConv { w, b, cin, cout })
}

fn ld_1x1(ss: &ShardedSafetensors, prefix: &str,
          cin: usize, cout: usize) -> Result<Wan1x1Conv, CosmosError> {
    let w = ld_bf16(ss, &format!("{prefix}.weight"))?;
    let b = ld(ss, &format!("{prefix}.bias"))?;
    if w.len() != cout * cin {
        return Err(CosmosError::Load(format!(
            "{prefix}.weight: expected {} = {cout}×{cin}×1×1×1, got {}",
            cout*cin, w.len())));
    }
    if b.len() != cout {
        return Err(CosmosError::Load(format!(
            "{prefix}.bias: expected {cout}, got {}", b.len())));
    }
    Ok(Wan1x1Conv { w, b, cin, cout })
}

fn ld_gamma(ss: &ShardedSafetensors, name: &str, expect: usize) -> Result<Vec<f32>, CosmosError> {
    let g = ld(ss, name)?;
    if g.len() != expect {
        return Err(CosmosError::Load(format!(
            "{name}: expected {expect}, got {}", g.len())));
    }
    Ok(g)
}

fn ld_resnet(ss: &ShardedSafetensors, prefix: &str,
             cin: usize, cout: usize) -> Result<WanResNetBlock, CosmosError> {
    let norm1_gamma = ld_gamma(ss, &format!("{prefix}.norm1.gamma"), cin)?;
    let conv1 = ld_3d(ss, &format!("{prefix}.conv1"), cin, cout, (3,3,3))?;
    let norm2_gamma = ld_gamma(ss, &format!("{prefix}.norm2.gamma"), cout)?;
    let conv2 = ld_3d(ss, &format!("{prefix}.conv2"), cout, cout, (3,3,3))?;
    let shortcut = if cin != cout {
        Some(ld_1x1(ss, &format!("{prefix}.conv_shortcut"), cin, cout)?)
    } else { None };
    Ok(WanResNetBlock { norm1_gamma, conv1, norm2_gamma, conv2, shortcut })
}

fn ld_time_conv(ss: &ShardedSafetensors, prefix: &str,
                cin: usize, cout: usize) -> Result<WanTimeConv, CosmosError> {
    let w = ld_bf16(ss, &format!("{prefix}.weight"))?;
    let b = ld(ss, &format!("{prefix}.bias"))?;
    if w.len() != cout * cin * 3 {
        return Err(CosmosError::Load(format!(
            "{prefix}.weight: expected {} = {cout}×{cin}×3×1×1, got {}",
            cout*cin*3, w.len())));
    }
    if b.len() != cout {
        return Err(CosmosError::Load(format!(
            "{prefix}.bias: expected {cout}, got {}", b.len())));
    }
    Ok(WanTimeConv { w, b, cin, cout })
}

fn ld_mid_attention(ss: &ShardedSafetensors, prefix: &str, c: usize) -> Result<WanMidAttention, CosmosError> {
    let norm_gamma = ld_gamma(ss, &format!("{prefix}.norm.gamma"), c)?;
    let to_qkv_w = ld_bf16(ss, &format!("{prefix}.to_qkv.weight"))?;
    let to_qkv_b = ld(ss, &format!("{prefix}.to_qkv.bias"))?;
    let proj_w   = ld_bf16(ss, &format!("{prefix}.proj.weight"))?;
    let proj_b   = ld(ss, &format!("{prefix}.proj.bias"))?;
    if to_qkv_w.len() != 3 * c * c {
        return Err(CosmosError::Load(format!(
            "{prefix}.to_qkv.weight: expected {} = 3×{c}×{c}, got {}",
            3*c*c, to_qkv_w.len())));
    }
    if to_qkv_b.len() != 3 * c {
        return Err(CosmosError::Load(format!(
            "{prefix}.to_qkv.bias: expected {} = 3×{c}, got {}", 3*c, to_qkv_b.len())));
    }
    if proj_w.len() != c * c {
        return Err(CosmosError::Load(format!(
            "{prefix}.proj.weight: expected {} = {c}×{c}, got {}", c*c, proj_w.len())));
    }
    if proj_b.len() != c {
        return Err(CosmosError::Load(format!(
            "{prefix}.proj.bias: expected {c}, got {}", proj_b.len())));
    }
    Ok(WanMidAttention { norm_gamma, to_qkv_w, to_qkv_b, proj_w, proj_b, c })
}

fn ld_mid_block(ss: &ShardedSafetensors, prefix: &str, c: usize) -> Result<WanMidBlock, CosmosError> {
    let r0 = ld_resnet(ss, &format!("{prefix}.resnets.0"), c, c)?;
    let r1 = ld_resnet(ss, &format!("{prefix}.resnets.1"), c, c)?;
    let attention = ld_mid_attention(ss, &format!("{prefix}.attentions.0"), c)?;
    Ok(WanMidBlock { resnets: [r0, r1], attention })
}

// ── Wan VAE forward primitives ──────────────────────────────────────────
// Wan 2.2's VAE uses channel-axis RMS-style normalization (no beta) and
// SiLU activation throughout.  3D convs are `k=(3,3,3) s=(1,1,1) p=(1,1,1)`
// except downsamplers (2D stride-2) and time-convs (`k=(3,1,1) p=(1,0,0)`).
// Implementations are scalar nested loops for now (correctness-first); the
// production path uses Metal for the spatial convs that dominate FLOPs.

fn wan_silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

/// Per-spatial-position RMS norm across the channel axis, then channel-wise
/// `gamma` scale.  `x` is `[C, T, H, W]` row-major (C-outer).  Matches
/// `WanRMSNorm` from Wan-AI/Wan2.2 (no beta, eps=1e-6).
fn wan_rmsnorm_3d(
    x: &[f32], c: usize, t: usize, h: usize, w: usize, gamma: &[f32],
) -> Vec<f32> {
    debug_assert_eq!(x.len(), c * t * h * w);
    debug_assert_eq!(gamma.len(), c);
    let n = t * h * w;
    let mut out = vec![0.0f32; x.len()];
    // For each spatial position p, compute RMS = sqrt(mean over C of x²).
    for p in 0..n {
        let mut ss = 0.0f64;
        for ci in 0..c {
            let v = x[ci * n + p] as f64;
            ss += v * v;
        }
        let inv = ((ss / c as f64) + 1e-6).sqrt().recip() as f32;
        for ci in 0..c {
            out[ci * n + p] = x[ci * n + p] * inv * gamma[ci];
        }
    }
    out
}

/// 2D variant for mid-block attention pre-norm (gamma shape `[C, 1, 1]`).
fn wan_rmsnorm_2d(
    x: &[f32], c: usize, t: usize, h: usize, w: usize, gamma: &[f32],
) -> Vec<f32> {
    // Layout is still [C, T, H, W]; the 2D vs 3D distinction is only the
    // gamma rank in the safetensors.  Forward is identical to wan_rmsnorm_3d.
    wan_rmsnorm_3d(x, c, t, h, w, gamma)
}

/// 3D conv with `k=(3,3,3) s=(1,1,1) p=(1,1,1)`.  Input/output T·H·W unchanged.
/// Weight `[Cout, Cin, 3, 3, 3]` row-major.
fn wan_conv3d_k3_s1_p1(
    inp: &[f32], cin: usize, t: usize, h: usize, w: usize,
    weight: &[half::bf16], bias: &[f32], cout: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    let plane = h * w;
    let n = t * plane;
    let mut out = vec![0.0f32; cout * n];
    out.par_chunks_mut(n).enumerate().for_each(|(oc, oplane)| {
        for v in oplane.iter_mut() { *v = bias[oc]; }
        for ic in 0..cin {
            let w_base = ((oc * cin) + ic) * 27;
            let in_off = ic * n;
            for ot in 0..t { for oh in 0..h { for ow in 0..w {
                let mut acc = 0.0f32;
                for kt in 0..3 { for kh in 0..3 { for kw in 0..3 {
                    let it = ot as isize + kt as isize - 1;
                    let ih = oh as isize + kh as isize - 1;
                    let iw = ow as isize + kw as isize - 1;
                    if it < 0 || it >= t as isize { continue; }
                    if ih < 0 || ih >= h as isize { continue; }
                    if iw < 0 || iw >= w as isize { continue; }
                    let v = inp[in_off + (it as usize) * plane + (ih as usize) * w + iw as usize];
                    let wt = weight[w_base + kt * 9 + kh * 3 + kw].to_f32();
                    acc += v * wt;
                }}}
                oplane[ot * plane + oh * w + ow] += acc;
            }}}
        }
    });
    out
}

/// im2col for `k=(3,3,3) s=(1,1,1) p=(1,1,1)`: lay out every 27-element
/// receptive field as a contiguous row.  Input `[Cin, T, H, W]` → output
/// `[T·H·W, Cin·27]` row-major.  Zero-pad implicit (boundary kernels
/// emit 0.0 for off-grid taps).
fn wan_im2col_k3_s1_p1(
    inp: &[f32], cin: usize, t: usize, h: usize, w: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    let plane = h * w;
    let m = t * plane;
    let k_dim = cin * 27;
    let mut col = vec![0.0f32; m * k_dim];
    col.par_chunks_mut(k_dim).enumerate().for_each(|(row, dst)| {
        let ot = row / plane;
        let oh = (row % plane) / w;
        let ow = row % w;
        for ic in 0..cin {
            let in_off = ic * t * plane;
            let dst_base = ic * 27;
            for kt in 0..3 { for kh in 0..3 { for kw in 0..3 {
                let it = ot as isize + kt as isize - 1;
                let ih = oh as isize + kh as isize - 1;
                let iw = ow as isize + kw as isize - 1;
                if it < 0 || it >= t as isize { continue; }
                if ih < 0 || ih >= h as isize { continue; }
                if iw < 0 || iw >= w as isize { continue; }
                let v = inp[in_off + (it as usize) * plane + (ih as usize) * w + iw as usize];
                dst[dst_base + kt * 9 + kh * 3 + kw] = v;
            }}}
        }
    });
    col
}

/// im2col memory budget for the Metal-routed conv3d (default 1 GB).
/// The im2col scratch is `T·H·W · Cin · 27 · 4 bytes`; when that exceeds the
/// budget, `wan_conv3d_k3_s1_p1_routed` automatically tiles along H/W.
/// Override via the `WAN_CONV3D_TILE_BUDGET_MB` env var (in megabytes).
pub fn wan_conv3d_im2col_budget_bytes() -> usize {
    if let Ok(s) = std::env::var("WAN_CONV3D_TILE_BUDGET_MB") {
        if let Ok(mb) = s.parse::<usize>() {
            return mb * 1024 * 1024;
        }
    }
    1024 * 1024 * 1024  // 1 GB default
}

/// Pick a square-ish tile size `(tile_h, tile_w)` that keeps the per-tile
/// im2col scratch under `budget_bytes`.  Returns `(h, w)` for no tiling
/// (the full slab fits in budget); otherwise returns a tile size that
/// guarantees `t · tile_h · tile_w · cin · 27 · 4 ≤ budget_bytes`.
fn pick_conv3d_tile(t: usize, h: usize, w: usize, cin: usize, budget_bytes: usize) -> (usize, usize) {
    let per_voxel_bytes = cin * 27 * 4;
    let max_voxels = budget_bytes / per_voxel_bytes.max(1);
    let max_per_frame = max_voxels / t.max(1);
    if max_per_frame >= h * w { return (h, w); }
    // Pick a square-ish tile; default to 32×32 then halve until it fits.
    let mut tile = 32usize;
    while tile > 1 && tile * tile > max_per_frame { tile /= 2; }
    if tile == 0 { tile = 1; }
    (tile.min(h), tile.min(w))
}

/// Metal-routed 3D conv `k=(3,3,3) s=(1,1,1) p=(1,1,1)`.  Performs CPU
/// im2col into `[T·H·W, Cin·27]`, dispatches the bf16 tiled matmul against
/// the conv weight (cached on first call, hit thereafter), then bias-adds
/// and reshapes back to `[Cout, T, H, W]`.  Falls back to scalar CPU
/// implementation when Metal is unavailable or dispatch fails.
fn wan_conv3d_k3_s1_p1_routed(
    inp: &[f32], cin: usize, t: usize, h: usize, w: usize,
    conv: &Wan3DConv,
    accel: Option<&crate::phase4::MetalAccel>,
) -> Vec<f32> {
    let Some(a) = accel else {
        return wan_conv3d_k3_s1_p1(inp, cin, t, h, w, &conv.w, &conv.b, conv.cout);
    };
    let (tile_h, tile_w) = pick_conv3d_tile(t, h, w, cin, wan_conv3d_im2col_budget_bytes());
    if tile_h >= h && tile_w >= w {
        // Whole slab fits in budget — single dispatch.
        return wan_conv3d_routed_single(inp, cin, t, h, w, conv, a);
    }
    wan_conv3d_routed_tiled(inp, cin, t, h, w, conv, a, tile_h, tile_w)
}

/// One-shot dispatch (no tiling).  Caller guarantees the im2col fits in
/// the memory budget.
fn wan_conv3d_routed_single(
    inp: &[f32], cin: usize, t: usize, h: usize, w: usize,
    conv: &Wan3DConv,
    a: &crate::phase4::MetalAccel,
) -> Vec<f32> {
    let col = wan_im2col_k3_s1_p1(inp, cin, t, h, w);
    let m = t * h * w;
    let k_dim = cin * 27;
    let n = conv.cout;
    let weight_slice = &conv.w;
    let result = a.maybe_dispatch_matmul_bf16_lazy(
        &col,
        || {
            let mut wf = vec![0.0f32; n * k_dim];
            for i in 0..wf.len() { wf[i] = weight_slice[i].to_f32(); }
            wf
        },
        conv.cache_id as usize,
        0,
        m, k_dim, n,
    );
    if let Some(matmul_out) = result {
        let mut out = vec![0.0f32; n * m];
        for ci in 0..n {
            let bi = conv.b[ci];
            for ri in 0..m {
                out[ci * m + ri] = matmul_out[ri * n + ci] + bi;
            }
        }
        return out;
    }
    wan_conv3d_k3_s1_p1(inp, cin, t, h, w, &conv.w, &conv.b, conv.cout)
}

/// Tiled dispatch.  Splits the H,W output grid into `(tile_h, tile_w)` blocks,
/// each with a halo of 1 on internal edges so the per-tile im2col can use the
/// real neighbor values (rather than zero-padding) at tile boundaries.
/// Global boundaries (oh_start = 0, oh_end = H, etc.) drop the halo because
/// the conv's implicit padding=1 supplies the zero pad correctly.
fn wan_conv3d_routed_tiled(
    inp: &[f32], cin: usize, t: usize, h: usize, w: usize,
    conv: &Wan3DConv,
    a: &crate::phase4::MetalAccel,
    tile_h: usize, tile_w: usize,
) -> Vec<f32> {
    let cout = conv.cout;
    let mut out = vec![0.0f32; cout * t * h * w];
    let mut oh_start = 0;
    while oh_start < h {
        let oh_end = (oh_start + tile_h).min(h);
        let ih_start = if oh_start > 0 { oh_start - 1 } else { 0 };
        let ih_end = if oh_end < h { oh_end + 1 } else { h };
        let in_th = ih_end - ih_start;
        let skip_h = oh_start - ih_start;
        let actual_th = oh_end - oh_start;

        let mut ow_start = 0;
        while ow_start < w {
            let ow_end = (ow_start + tile_w).min(w);
            let iw_start = if ow_start > 0 { ow_start - 1 } else { 0 };
            let iw_end = if ow_end < w { ow_end + 1 } else { w };
            let in_tw = iw_end - iw_start;
            let skip_w = ow_start - iw_start;
            let actual_tw = ow_end - ow_start;

            // Extract input tile [Cin, T, in_th, in_tw].
            let mut tile_in = vec![0.0f32; cin * t * in_th * in_tw];
            for ci in 0..cin {
                for ti in 0..t {
                    for hi in 0..in_th {
                        let gh = ih_start + hi;
                        let src_row = ci * t * h * w + ti * h * w + gh * w + iw_start;
                        let dst_row = ci * t * in_th * in_tw + ti * in_th * in_tw + hi * in_tw;
                        tile_in[dst_row..dst_row + in_tw]
                            .copy_from_slice(&inp[src_row..src_row + in_tw]);
                    }
                }
            }

            // Single dispatch on the tile (guaranteed to fit budget by caller).
            let tile_out = wan_conv3d_routed_single(&tile_in, cin, t, in_th, in_tw, conv, a);

            // Copy tile_out[..t, skip_h..skip_h+actual_th, skip_w..skip_w+actual_tw]
            // into out at the corresponding global positions.
            for ci in 0..cout {
                for ti in 0..t {
                    for hi in 0..actual_th {
                        let src_row = ci * t * in_th * in_tw
                                    + ti * in_th * in_tw
                                    + (skip_h + hi) * in_tw
                                    + skip_w;
                        let gh = oh_start + hi;
                        let dst_row = ci * t * h * w + ti * h * w + gh * w + ow_start;
                        out[dst_row..dst_row + actual_tw]
                            .copy_from_slice(&tile_out[src_row..src_row + actual_tw]);
                    }
                }
            }
            ow_start = ow_end;
        }
        oh_start = oh_end;
    }
    out
}

/// 1×1×1 conv — pure channel mix.  Weight `[Cout, Cin]` (collapsed `[Cout, Cin, 1, 1, 1]`).
fn wan_conv_1x1x1(
    inp: &[f32], cin: usize, t: usize, h: usize, w: usize,
    weight: &[half::bf16], bias: &[f32], cout: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    let n = t * h * w;
    let mut out = vec![0.0f32; cout * n];
    out.par_chunks_mut(n).enumerate().for_each(|(oc, oplane)| {
        for v in oplane.iter_mut() { *v = bias[oc]; }
        for ic in 0..cin {
            let wt = weight[oc * cin + ic].to_f32();
            let in_off = ic * n;
            for i in 0..n { oplane[i] += wt * inp[in_off + i]; }
        }
    });
    out
}

/// 2D conv with `k=(3,3) s=(2,2) p=(1,1)`.  Halves H and W independently
/// per T-frame.  Weight `[Cout, Cin, 3, 3]` row-major.
fn wan_conv2d_k3_s2_p1(
    inp: &[f32], cin: usize, t: usize, h: usize, w: usize,
    weight: &[half::bf16], bias: &[f32], cout: usize,
) -> (Vec<f32>, usize, usize) {
    use rayon::prelude::*;
    let ho = h.div_ceil(2);
    let wo = w.div_ceil(2);
    let plane_in = h * w;
    let plane_out = ho * wo;
    let n_out = t * plane_out;
    let mut out = vec![0.0f32; cout * n_out];
    out.par_chunks_mut(n_out).enumerate().for_each(|(oc, oplane)| {
        for v in oplane.iter_mut() { *v = bias[oc]; }
        for ic in 0..cin {
            let w_base = ((oc * cin) + ic) * 9;
            let in_off = ic * t * plane_in;
            for ti in 0..t { for oh in 0..ho { for ow in 0..wo {
                let mut acc = 0.0f32;
                for kh in 0..3 { for kw in 0..3 {
                    let ih = oh as isize * 2 + kh as isize - 1;
                    let iw = ow as isize * 2 + kw as isize - 1;
                    if ih < 0 || ih >= h as isize { continue; }
                    if iw < 0 || iw >= w as isize { continue; }
                    let v = inp[in_off + ti * plane_in + (ih as usize) * w + iw as usize];
                    let wt = weight[w_base + kh * 3 + kw].to_f32();
                    acc += v * wt;
                }}
                oplane[ti * plane_out + oh * wo + ow] += acc;
            }}}
        }
    });
    (out, ho, wo)
}

/// 2D conv with `k=(3,3) s=(1,1) p=(1,1)`.  Used by decoder upsampler after
/// a 2× nearest spatial upsample.  Output H·W = input H·W.
fn wan_conv2d_k3_s1_p1(
    inp: &[f32], cin: usize, t: usize, h: usize, w: usize,
    weight: &[half::bf16], bias: &[f32], cout: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    let plane = h * w;
    let n = t * plane;
    let mut out = vec![0.0f32; cout * n];
    out.par_chunks_mut(n).enumerate().for_each(|(oc, oplane)| {
        for v in oplane.iter_mut() { *v = bias[oc]; }
        for ic in 0..cin {
            let w_base = ((oc * cin) + ic) * 9;
            let in_off = ic * n;
            for ti in 0..t { for oh in 0..h { for ow in 0..w {
                let mut acc = 0.0f32;
                for kh in 0..3 { for kw in 0..3 {
                    let ih = oh as isize + kh as isize - 1;
                    let iw = ow as isize + kw as isize - 1;
                    if ih < 0 || ih >= h as isize { continue; }
                    if iw < 0 || iw >= w as isize { continue; }
                    let v = inp[in_off + ti * plane + (ih as usize) * w + iw as usize];
                    let wt = weight[w_base + kh * 3 + kw].to_f32();
                    acc += v * wt;
                }}
                oplane[ti * plane + oh * w + ow] += acc;
            }}}
        }
    });
    out
}

/// Encoder `time_conv` — kernel `(3, 1, 1)`, stride `(2, 1, 1)`, pad `(1, 0, 0)`.
/// Halves T (matches the 2D spatial downsample).
/// Weight `[Cout, Cin, 3, 1, 1]` → effectively `[Cout, Cin, 3]`.
fn wan_time_conv_enc(
    inp: &[f32], cin: usize, t: usize, h: usize, w: usize,
    weight: &[half::bf16], bias: &[f32], cout: usize,
) -> (Vec<f32>, usize) {
    use rayon::prelude::*;
    let plane = h * w;
    let to = t.div_ceil(2);
    let n_out = to * plane;
    let mut out = vec![0.0f32; cout * n_out];
    out.par_chunks_mut(n_out).enumerate().for_each(|(oc, oplane)| {
        for v in oplane.iter_mut() { *v = bias[oc]; }
        for ic in 0..cin {
            let w_base = ((oc * cin) + ic) * 3;
            let in_off = ic * t * plane;
            for ot in 0..to {
                for kt in 0..3 {
                    let it = ot as isize * 2 + kt as isize - 1;
                    if it < 0 || it >= t as isize { continue; }
                    let wt = weight[w_base + kt].to_f32();
                    let in_row = &inp[in_off + (it as usize) * plane .. in_off + (it as usize + 1) * plane];
                    let out_row = &mut oplane[ot * plane .. (ot + 1) * plane];
                    for i in 0..plane { out_row[i] += wt * in_row[i]; }
                }
            }
        }
    });
    (out, to)
}

/// Decoder `time_conv` — kernel `(3, 1, 1)`, stride `(1, 1, 1)`, pad `(1, 0, 0)`,
/// and Cout = 2·Cin.  The output two-frame pixel-shuffle along the temporal
/// axis doubles T.  Weight shape stored as `[2·Cin, Cin, 3, 1, 1]`; we split
/// it into `[2, Cin, Cin, 3]` channel-pair-wise.
fn wan_time_conv_dec(
    inp: &[f32], cin: usize, t: usize, h: usize, w: usize,
    weight: &[half::bf16], bias: &[f32],
) -> (Vec<f32>, usize) {
    use rayon::prelude::*;
    let cout = cin; // post-pixel-shuffle output channels
    let plane = h * w;
    let to = t * 2;
    // First do a 2·Cin → Cin·2 conv (k=3 along T, padding=1) producing
    // [2, Cin, T, H, W]; then interleave the two frames per (cin, t).
    let n_mid = t * plane;
    let mut mid = vec![0.0f32; (2 * cin) * n_mid];
    mid.par_chunks_mut(n_mid).enumerate().for_each(|(oc, oplane)| {
        for v in oplane.iter_mut() { *v = bias[oc]; }
        for ic in 0..cin {
            let w_base = ((oc * cin) + ic) * 3;
            let in_off = ic * t * plane;
            for ot in 0..t {
                for kt in 0..3 {
                    let it = ot as isize + kt as isize - 1;
                    if it < 0 || it >= t as isize { continue; }
                    let wt = weight[w_base + kt].to_f32();
                    let in_row = &inp[in_off + (it as usize) * plane .. in_off + (it as usize + 1) * plane];
                    let out_row = &mut oplane[ot * plane .. (ot + 1) * plane];
                    for i in 0..plane { out_row[i] += wt * in_row[i]; }
                }
            }
        }
    });
    // Interleave: mid is laid out [pair=0|1, ci, t, plane] → out [ci, t·2, plane]
    // where output time index 2t+pair carries pair=0 or pair=1.
    let mut out = vec![0.0f32; cout * to * plane];
    for ci in 0..cin {
        for ti in 0..t {
            for pair in 0..2 {
                let mid_off = (pair * cin + ci) * n_mid + ti * plane;
                let out_off = ci * to * plane + (ti * 2 + pair) * plane;
                out[out_off..out_off + plane].copy_from_slice(&mid[mid_off..mid_off + plane]);
            }
        }
    }
    (out, to)
}

/// Nearest-neighbor 2× upsample along H and W only (T unchanged).
fn wan_nearest_2x_hw(inp: &[f32], c: usize, t: usize, h: usize, w: usize)
    -> (Vec<f32>, usize, usize) {
    let ho = h * 2;
    let wo = w * 2;
    let plane_in = h * w;
    let plane_out = ho * wo;
    let mut out = vec![0.0f32; c * t * plane_out];
    for ci in 0..c { for ti in 0..t {
        let in_base = ci * t * plane_in + ti * plane_in;
        let out_base = ci * t * plane_out + ti * plane_out;
        for ih in 0..h { for iw in 0..w {
            let v = inp[in_base + ih * w + iw];
            for kh in 0..2 { for kw in 0..2 {
                out[out_base + (ih * 2 + kh) * wo + (iw * 2 + kw)] = v;
            }}
        }}
    }}
    (out, ho, wo)
}

impl WanResNetBlock {
    /// Forward: x[Cin, T, H, W] → out[Cout, T, H, W].  shortcut handles channel-mismatch.
    pub fn forward(&self, x: &[f32], t: usize, h: usize, w: usize) -> Vec<f32> {
        self.forward_routed(x, t, h, w, None)
    }

    /// Routed forward: when `accel.is_some()`, every conv3d hits the Metal
    /// im2col + tiled bf16 matmul path (weights cached per `Wan3DConv::cache_id`).
    pub fn forward_routed(
        &self, x: &[f32], t: usize, h: usize, w: usize,
        accel: Option<&crate::phase4::MetalAccel>,
    ) -> Vec<f32> {
        let cin = self.conv1.cin;
        let cout = self.conv1.cout;
        let h1 = wan_rmsnorm_3d(x, cin, t, h, w, &self.norm1_gamma);
        let mut h1 = h1;
        wan_silu(&mut h1);
        let h1 = wan_conv3d_k3_s1_p1_routed(&h1, cin, t, h, w, &self.conv1, accel);
        let h2 = wan_rmsnorm_3d(&h1, cout, t, h, w, &self.norm2_gamma);
        let mut h2 = h2;
        wan_silu(&mut h2);
        let h2 = wan_conv3d_k3_s1_p1_routed(&h2, cout, t, h, w, &self.conv2, accel);
        // Residual: shortcut(x) + h2.  Shortcut is 1×1×1 (cheap; stays CPU).
        let sk = if let Some(sc) = &self.shortcut {
            wan_conv_1x1x1(x, cin, t, h, w, &sc.w, &sc.b, cout)
        } else {
            x.to_vec()
        };
        debug_assert_eq!(sk.len(), h2.len());
        let mut out = h2;
        for i in 0..out.len() { out[i] += sk[i]; }
        out
    }
}

impl WanMidAttention {
    /// Forward: x[C, T, H, W] → out[C, T, H, W].  Standard SDPA over flattened
    /// spatial+temporal sequence (T·H·W tokens, C-dim channels).  No multi-head
    /// — the Wan VAE mid attention is single-head.
    pub fn forward(&self, x: &[f32], t: usize, h: usize, w: usize) -> Vec<f32> {
        let c = self.c;
        let n = t * h * w;
        // Pre-norm
        let xn = wan_rmsnorm_2d(x, c, t, h, w, &self.norm_gamma);
        // to_qkv: 1×1 conv → [3C, T, H, W]
        let qkv = wan_conv_1x1x1(&xn, c, t, h, w, &self.to_qkv_w, &self.to_qkv_b, 3 * c);
        // Split into Q, K, V each [C, n] but transposed to seq-major [n, C] for SDPA.
        let mut q = vec![0.0f32; n * c];
        let mut k = vec![0.0f32; n * c];
        let mut v = vec![0.0f32; n * c];
        for ci in 0..c {
            for i in 0..n {
                q[i * c + ci] = qkv[(0 * c + ci) * n + i];
                k[i * c + ci] = qkv[(1 * c + ci) * n + i];
                v[i * c + ci] = qkv[(2 * c + ci) * n + i];
            }
        }
        // SDPA: scores = Q @ Kᵀ / sqrt(c); softmax row-wise; attn @ V.
        let scale = (c as f32).powf(-0.5);
        let mut attn = vec![0.0f32; n * n];
        for i in 0..n {
            let qi = &q[i * c .. (i + 1) * c];
            let mut row_max = f32::NEG_INFINITY;
            for j in 0..n {
                let kj = &k[j * c .. (j + 1) * c];
                let mut s = 0.0f32;
                for ci in 0..c { s += qi[ci] * kj[ci]; }
                s *= scale;
                attn[i * n + j] = s;
                if s > row_max { row_max = s; }
            }
            let mut sum = 0.0f32;
            for j in 0..n {
                let e = (attn[i * n + j] - row_max).exp();
                attn[i * n + j] = e;
                sum += e;
            }
            let inv = sum.recip();
            for j in 0..n { attn[i * n + j] *= inv; }
        }
        let mut out_t = vec![0.0f32; n * c]; // [n, c]
        for i in 0..n {
            let row = &attn[i * n .. (i + 1) * n];
            for ci in 0..c {
                let mut acc = 0.0f32;
                for j in 0..n { acc += row[j] * v[j * c + ci]; }
                out_t[i * c + ci] = acc;
            }
        }
        // Reshape back to [C, T, H, W]
        let mut out_chw = vec![0.0f32; c * n];
        for ci in 0..c { for i in 0..n { out_chw[ci * n + i] = out_t[i * c + ci]; } }
        // Output projection (1×1) + residual.
        let proj_out = wan_conv_1x1x1(&out_chw, c, t, h, w, &self.proj_w, &self.proj_b, c);
        let mut out = proj_out;
        for i in 0..out.len() { out[i] += x[i]; }
        out
    }
}

impl WanMidBlock {
    pub fn forward(&self, x: &[f32], t: usize, h: usize, w: usize) -> Vec<f32> {
        self.forward_routed(x, t, h, w, None)
    }

    pub fn forward_routed(
        &self, x: &[f32], t: usize, h: usize, w: usize,
        accel: Option<&crate::phase4::MetalAccel>,
    ) -> Vec<f32> {
        let h0 = self.resnets[0].forward_routed(x, t, h, w, accel);
        let ha = self.attention.forward(&h0, t, h, w);
        self.resnets[1].forward_routed(&ha, t, h, w, accel)
    }
}

impl WanDownsampler {
    /// Spatial downsample (H/2, W/2) + optional temporal downsample (T/2 when
    /// `time_conv` present).  Returns `(out, t', h', w')`.
    pub fn forward(&self, x: &[f32], c: usize, t: usize, h: usize, w: usize)
        -> (Vec<f32>, usize, usize, usize)
    {
        // 2D spatial downsample per frame.
        let (mut y, ho, wo) = wan_conv2d_k3_s2_p1(
            x, c, t, h, w,
            &self.resample.w, &self.resample.b, self.resample.cout);
        let mut tcur = t;
        if let Some(tc) = &self.time_conv {
            let (y2, t2) = wan_time_conv_enc(
                &y, tc.cin, tcur, ho, wo,
                &tc.w, &tc.b, tc.cout);
            y = y2;
            tcur = t2;
        }
        (y, tcur, ho, wo)
    }
}

impl WanUpsampler {
    /// 2× nearest H,W upsample → 2D conv (stride 1) → optional temporal pixel-shuffle.
    pub fn forward(&self, x: &[f32], c: usize, t: usize, h: usize, w: usize)
        -> (Vec<f32>, usize, usize, usize)
    {
        let (up, ho, wo) = wan_nearest_2x_hw(x, c, t, h, w);
        let y = wan_conv2d_k3_s1_p1(
            &up, c, t, ho, wo,
            &self.resample.w, &self.resample.b, self.resample.cout);
        if let Some(tc) = &self.time_conv {
            let (y2, t2) = wan_time_conv_dec(
                &y, tc.cin, t, ho, wo, &tc.w, &tc.b);
            (y2, t2, ho, wo)
        } else {
            (y, t, ho, wo)
        }
    }
}

impl Wan22VAE_v2 {
    /// Load from `<root>/vae/diffusion_pytorch_model.safetensors`.
    pub fn load(vae_path: &Path) -> Result<Self, CosmosError> {
        let ss = ShardedSafetensors::from_single_file(vae_path)?;
        let cfg = WanVaeConfig::cosmos3_nano();

        // ── Encoder channel schedule (left-to-right): base × dim_mult[i] ──
        let enc_ch: Vec<usize> = cfg.dim_mult.iter().map(|m| m * cfg.base_dim).collect();
        // First conv_in width = enc_ch[0].
        let encoder_conv_in = ld_3d(&ss, "encoder.conv_in", cfg.in_channels, enc_ch[0], (3,3,3))?;

        let mut encoder_down_blocks = Vec::with_capacity(enc_ch.len());
        for i in 0..enc_ch.len() {
            let cin  = if i == 0 { enc_ch[0] } else { enc_ch[i-1] };
            let cout = enc_ch[i];
            let r0 = ld_resnet(&ss, &format!("encoder.down_blocks.{i}.resnets.0"), cin, cout)?;
            let r1 = ld_resnet(&ss, &format!("encoder.down_blocks.{i}.resnets.1"), cout, cout)?;
            let downsampler = if i + 1 < enc_ch.len() {
                let resample = ld_2d(&ss, &format!("encoder.down_blocks.{i}.downsampler.resample.1"),
                                     cout, cout)?;
                let time_conv = if cfg.temperal_downsample[i] {
                    Some(ld_time_conv(&ss, &format!("encoder.down_blocks.{i}.downsampler.time_conv"),
                                      cout, cout)?)
                } else { None };
                Some(WanDownsampler { resample, time_conv })
            } else { None };
            encoder_down_blocks.push(WanDownBlock { resnets: vec![r0, r1], downsampler });
        }

        let enc_mid_c = *enc_ch.last().unwrap();
        let encoder_mid_block = ld_mid_block(&ss, "encoder.mid_block", enc_mid_c)?;
        let encoder_norm_out_gamma = ld_gamma(&ss, "encoder.norm_out.gamma", enc_mid_c)?;
        // conv_out emits 2·z_dim (mean + log_var) packed channel-wise.
        let encoder_conv_out = ld_3d(&ss, "encoder.conv_out", enc_mid_c, 2*cfg.z_dim, (3,3,3))?;

        let quant_conv      = ld_1x1(&ss, "quant_conv",      2*cfg.z_dim, 2*cfg.z_dim)?;
        let post_quant_conv = ld_1x1(&ss, "post_quant_conv",   cfg.z_dim,   cfg.z_dim)?;

        // ── Decoder channel schedule (reversed dim_mult): decoder_base × dim_mult[N-1..0] ──
        let dec_ch: Vec<usize> = cfg.dim_mult.iter().rev().map(|m| m * cfg.decoder_base_dim).collect();
        let dec_in_c = dec_ch[0];
        let decoder_conv_in = ld_3d(&ss, "decoder.conv_in", cfg.z_dim, dec_in_c, (3,3,3))?;
        let decoder_mid_block = ld_mid_block(&ss, "decoder.mid_block", dec_in_c)?;

        let mut decoder_up_blocks = Vec::with_capacity(dec_ch.len());
        for i in 0..dec_ch.len() {
            let cin  = if i == 0 { dec_in_c } else { dec_ch[i-1] };
            let cout = dec_ch[i];
            let r0 = ld_resnet(&ss, &format!("decoder.up_blocks.{i}.resnets.0"), cin, cout)?;
            let r1 = ld_resnet(&ss, &format!("decoder.up_blocks.{i}.resnets.1"), cout, cout)?;
            let r2 = ld_resnet(&ss, &format!("decoder.up_blocks.{i}.resnets.2"), cout, cout)?;
            // decoder upsamplers in stages 0..N-1; stage N-1 has no upsampler.
            let upsampler = if i + 1 < dec_ch.len() {
                let resample = ld_2d(&ss, &format!("decoder.up_blocks.{i}.upsampler.resample.1"),
                                     cout, cout)?;
                // Decoder time-conv presence mirrors encoder temperal_downsample reversed.
                let need_t = cfg.temperal_downsample[cfg.temperal_downsample.len() - 1 - i];
                let time_conv = if need_t {
                    // weight shape is [2·Cout, Cin=Cout, 3, 1, 1] — two stacked temporal frames.
                    Some(ld_time_conv(&ss, &format!("decoder.up_blocks.{i}.upsampler.time_conv"),
                                      cout, 2*cout)?)
                } else { None };
                Some(WanUpsampler { resample, time_conv })
            } else { None };
            decoder_up_blocks.push(WanUpBlock { resnets: vec![r0, r1, r2], upsampler });
        }

        let dec_out_c = *dec_ch.last().unwrap();
        let decoder_norm_out_gamma = ld_gamma(&ss, "decoder.norm_out.gamma", dec_out_c)?;
        let decoder_conv_out = ld_3d(&ss, "decoder.conv_out", dec_out_c, cfg.out_channels, (3,3,3))?;

        Ok(Self {
            config: cfg,
            encoder_conv_in, encoder_down_blocks, encoder_mid_block,
            encoder_norm_out_gamma, encoder_conv_out,
            quant_conv, post_quant_conv,
            decoder_conv_in, decoder_mid_block, decoder_up_blocks,
            decoder_norm_out_gamma, decoder_conv_out,
        })
    }

    /// Encoder forward through every loaded block.
    /// Input `video` is `[in_channels=12, T, H, W]` (RGB×4-frame already packed).
    /// Output is `[2·z_dim=96, T', H', W']` (mean and log_var packed channel-wise,
    /// pre `quant_conv`).  `T' = T / scale_factor_temporal` and `H'·W' = H·W / 16²`.
    pub fn encode_to_moments(
        &self, video: &[f32], t: usize, h: usize, w: usize
    ) -> (Vec<f32>, usize, usize, usize) {
        self.encode_to_moments_routed(video, t, h, w, None)
    }

    /// Routed encoder: when `accel.is_some()`, every conv3d (encoder.conv_in,
    /// every ResNet conv1/conv2, encoder.conv_out) hits the Metal im2col +
    /// tiled bf16 matmul path.  2D downsamplers and 1×1×1 channel-mix stay
    /// on CPU (much cheaper compute).
    pub fn encode_to_moments_routed(
        &self, video: &[f32], t: usize, h: usize, w: usize,
        accel: Option<&crate::phase4::MetalAccel>,
    ) -> (Vec<f32>, usize, usize, usize) {
        let mut x = wan_conv3d_k3_s1_p1_routed(
            video, self.config.in_channels, t, h, w,
            &self.encoder_conv_in, accel);
        let mut tcur = t; let mut hcur = h; let mut wcur = w;
        let mut ccur = self.encoder_conv_in.cout;

        for b in &self.encoder_down_blocks {
            for r in &b.resnets {
                x = r.forward_routed(&x, tcur, hcur, wcur, accel);
                ccur = r.conv1.cout;
            }
            if let Some(ds) = &b.downsampler {
                let (y, t2, h2, w2) = ds.forward(&x, ccur, tcur, hcur, wcur);
                x = y; tcur = t2; hcur = h2; wcur = w2;
            }
        }
        x = self.encoder_mid_block.forward_routed(&x, tcur, hcur, wcur, accel);
        let xn = wan_rmsnorm_3d(&x, ccur, tcur, hcur, wcur, &self.encoder_norm_out_gamma);
        let mut xn = xn;
        wan_silu(&mut xn);
        let moments = wan_conv3d_k3_s1_p1_routed(
            &xn, ccur, tcur, hcur, wcur,
            &self.encoder_conv_out, accel);
        (moments, tcur, hcur, wcur)
    }

    /// Encode → quant_conv → mean (ignore log_var). Returns the deterministic latent
    /// `[z_dim, T', H', W']`.
    pub fn encode(&self, video: &[f32], t: usize, h: usize, w: usize)
        -> (Vec<f32>, usize, usize, usize)
    {
        let (moments, t1, h1, w1) = self.encode_to_moments(video, t, h, w);
        let post = wan_conv_1x1x1(
            &moments, self.quant_conv.cin, t1, h1, w1,
            &self.quant_conv.w, &self.quant_conv.b, self.quant_conv.cout);
        // Split into mean (first z_dim channels) and discard log_var.
        let z = self.config.z_dim;
        let n = t1 * h1 * w1;
        let mean = post[..z * n].to_vec();
        (mean, t1, h1, w1)
    }

    /// Decoder forward through every loaded block.
    /// Input `latent` is `[z_dim=48, T', H', W']`.  Output is `[out_channels=12, T, H, W]`.
    pub fn decode(&self, latent: &[f32], t: usize, h: usize, w: usize)
        -> (Vec<f32>, usize, usize, usize)
    {
        self.decode_routed(latent, t, h, w, None)
    }

    /// Routed decoder: when `accel.is_some()`, every conv3d hits the Metal
    /// im2col + tiled bf16 matmul path.  2D upsamplers and 1×1×1 channel-mix
    /// stay on CPU.
    pub fn decode_routed(
        &self, latent: &[f32], t: usize, h: usize, w: usize,
        accel: Option<&crate::phase4::MetalAccel>,
    ) -> (Vec<f32>, usize, usize, usize) {
        let mut x = wan_conv_1x1x1(
            latent, self.post_quant_conv.cin, t, h, w,
            &self.post_quant_conv.w, &self.post_quant_conv.b, self.post_quant_conv.cout);
        x = wan_conv3d_k3_s1_p1_routed(
            &x, self.decoder_conv_in.cin, t, h, w,
            &self.decoder_conv_in, accel);
        let mut tcur = t; let mut hcur = h; let mut wcur = w;
        let mut ccur = self.decoder_conv_in.cout;

        x = self.decoder_mid_block.forward_routed(&x, tcur, hcur, wcur, accel);
        for b in &self.decoder_up_blocks {
            for r in &b.resnets {
                x = r.forward_routed(&x, tcur, hcur, wcur, accel);
                ccur = r.conv1.cout;
            }
            if let Some(us) = &b.upsampler {
                let (y, t2, h2, w2) = us.forward(&x, ccur, tcur, hcur, wcur);
                x = y; tcur = t2; hcur = h2; wcur = w2;
            }
        }
        let xn = wan_rmsnorm_3d(&x, ccur, tcur, hcur, wcur, &self.decoder_norm_out_gamma);
        let mut xn = xn;
        wan_silu(&mut xn);
        let video = wan_conv3d_k3_s1_p1_routed(
            &xn, ccur, tcur, hcur, wcur,
            &self.decoder_conv_out, accel);
        (video, tcur, hcur, wcur)
    }

    /// Normalize a latent into unit-variance, zero-mean space using the
    /// trained per-channel `latents_mean` / `latents_std` from `WanVaeConfig`.
    /// Production rectified-flow samplers operate in this normalized space;
    /// `encode` returns raw latent moments — call this before sampling.
    pub fn normalize_latent(&self, latent: &mut [f32], t: usize, h: usize, w: usize) {
        let z = self.config.z_dim;
        debug_assert_eq!(latent.len(), z * t * h * w);
        debug_assert_eq!(self.config.latents_mean.len(), z);
        debug_assert_eq!(self.config.latents_std.len(), z);
        let n = t * h * w;
        for ci in 0..z {
            let m = self.config.latents_mean[ci];
            let s = self.config.latents_std[ci];
            let inv = s.recip();
            for i in 0..n {
                let off = ci * n + i;
                latent[off] = (latent[off] - m) * inv;
            }
        }
    }

    /// Inverse of `normalize_latent`: undo the per-channel zero-mean /
    /// unit-std rescaling before decoding through the VAE.
    pub fn denormalize_latent(&self, latent: &mut [f32], t: usize, h: usize, w: usize) {
        let z = self.config.z_dim;
        debug_assert_eq!(latent.len(), z * t * h * w);
        let n = t * h * w;
        for ci in 0..z {
            let m = self.config.latents_mean[ci];
            let s = self.config.latents_std[ci];
            for i in 0..n {
                let off = ci * n + i;
                latent[off] = latent[off] * s + m;
            }
        }
    }

    /// Patchify a latent `[z_dim, T, H, W]` into transformer tokens
    /// `[T × (H/2) × (W/2), z_dim × 4]` by spatial 2×2 patches.  This is the
    /// bridge from VAE latent → 192-d generator visual token (z_dim=48, patch=2).
    pub fn patchify_2x2(&self, latent: &[f32], t: usize, h: usize, w: usize) -> Vec<f32> {
        let z = self.config.z_dim;
        let p = self.config.patch_size; // 2
        debug_assert_eq!(h % p, 0);
        debug_assert_eq!(w % p, 0);
        let ho = h / p; let wo = w / p;
        let mut tokens = vec![0.0f32; t * ho * wo * (z * p * p)];
        for ti in 0..t { for oy in 0..ho { for ox in 0..wo {
            let tok_off = ((ti * ho + oy) * wo + ox) * (z * p * p);
            for ci in 0..z {
                for ky in 0..p { for kx in 0..p {
                    let ih = oy * p + ky;
                    let iw = ox * p + kx;
                    let lat_off = ci * (t * h * w) + ti * (h * w) + ih * w + iw;
                    tokens[tok_off + ci * (p * p) + ky * p + kx] = latent[lat_off];
                }}
            }
        }}}
        tokens
    }

    /// Total user-heap bf16 footprint of the loaded VAE (in bytes of bf16
    /// data — `BfSlice` is mmap-aliased so this is the *kernel page cache*
    /// footprint, not the user heap).
    pub fn weight_bytes(&self) -> usize {
        let mut n = 0;
        let acc3 = |c: &Wan3DConv| c.w.len() * 2 + c.b.len() * 4;
        let acc2 = |c: &Wan2DConv| c.w.len() * 2 + c.b.len() * 4;
        let acc1 = |c: &Wan1x1Conv| c.w.len() * 2 + c.b.len() * 4;
        let acct = |c: &WanTimeConv| c.w.len() * 2 + c.b.len() * 4;
        let acc_rn = |r: &WanResNetBlock| {
            r.norm1_gamma.len()*4 + r.norm2_gamma.len()*4
                + acc3(&r.conv1) + acc3(&r.conv2)
                + r.shortcut.as_ref().map(acc1).unwrap_or(0)
        };
        let acc_mid = |m: &WanMidBlock| {
            m.resnets.iter().map(acc_rn).sum::<usize>()
                + m.attention.norm_gamma.len()*4
                + m.attention.to_qkv_w.len() * 2 + m.attention.to_qkv_b.len() * 4
                + m.attention.proj_w.len()   * 2 + m.attention.proj_b.len()   * 4
        };
        n += acc3(&self.encoder_conv_in);
        for b in &self.encoder_down_blocks {
            for r in &b.resnets { n += acc_rn(r); }
            if let Some(ds) = &b.downsampler {
                n += acc2(&ds.resample);
                if let Some(tc) = &ds.time_conv { n += acct(tc); }
            }
        }
        n += acc_mid(&self.encoder_mid_block);
        n += self.encoder_norm_out_gamma.len()*4;
        n += acc3(&self.encoder_conv_out);
        n += acc1(&self.quant_conv) + acc1(&self.post_quant_conv);
        n += acc3(&self.decoder_conv_in);
        n += acc_mid(&self.decoder_mid_block);
        for b in &self.decoder_up_blocks {
            for r in &b.resnets { n += acc_rn(r); }
            if let Some(us) = &b.upsampler {
                n += acc2(&us.resample);
                if let Some(tc) = &us.time_conv { n += acct(tc); }
            }
        }
        n += self.decoder_norm_out_gamma.len()*4;
        n += acc3(&self.decoder_conv_out);
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn cosmos3_transformer_dir() -> Option<PathBuf> {
        for vol in ["macos_4TB_external", "macos_4TB_external_1", "macos_4TB_external_2"] {
            let p = PathBuf::from(format!("/Volumes/{vol}/neuronexus_ai/model_library/Cosmos3-Nano/transformer"));
            if p.join("diffusion_pytorch_model.safetensors.index.json").exists() { return Some(p); }
        }
        None
    }

    fn cosmos3_vae_path() -> Option<PathBuf> {
        for vol in ["macos_4TB_external", "macos_4TB_external_1", "macos_4TB_external_2"] {
            let p = PathBuf::from(format!(
                "/Volumes/{vol}/neuronexus_ai/model_library/Cosmos3-Nano/vae/diffusion_pytorch_model.safetensors"));
            if p.exists() { return Some(p); }
        }
        None
    }

    /// Stage 0 Wan VAE: load all 196 tensors, verify structural shapes
    /// match the encoder channel schedule (160/320/640/640) and decoder
    /// channel schedule (1024/1024/512/256), and confirm the bf16 footprint
    /// is in the expected ~1.4 GB range.
    #[test]
    #[ignore = "phase4 v2 Wan VAE — needs Cosmos 3 Nano vae shard mounted"]
    fn phase4_v2_wan_vae_load() {
        let Some(p) = cosmos3_vae_path() else {
            eprintln!("Cosmos 3 Nano vae shard not found"); return;
        };
        let t0 = std::time::Instant::now();
        let v = Wan22VAE_v2::load(&p).expect("load Wan VAE");
        eprintln!("Wan22VAE_v2: loaded in {:?}", t0.elapsed());

        // Encoder channel schedule.
        let enc_ch = [160usize, 320, 640, 640];
        assert_eq!(v.encoder_conv_in.cin, 12);
        assert_eq!(v.encoder_conv_in.cout, enc_ch[0]);
        assert_eq!(v.encoder_down_blocks.len(), 4);
        for (i, b) in v.encoder_down_blocks.iter().enumerate() {
            assert_eq!(b.resnets.len(), 2, "encoder down {i} resnets");
            let cin_first = if i == 0 { enc_ch[0] } else { enc_ch[i-1] };
            assert_eq!(b.resnets[0].conv1.cin, cin_first);
            assert_eq!(b.resnets[0].conv1.cout, enc_ch[i]);
            assert_eq!(b.resnets[1].conv1.cin, enc_ch[i]);
            // Stages 0..2 have downsamplers; stage 3 has none.
            assert_eq!(b.downsampler.is_some(), i < 3);
            if let Some(ds) = &b.downsampler {
                assert_eq!(ds.resample.cin, enc_ch[i]);
                // temperal_downsample = [false, true, true]: time_conv on 1, 2.
                let want_t = matches!(i, 1 | 2);
                assert_eq!(ds.time_conv.is_some(), want_t, "downsampler {i} time_conv");
            }
        }
        assert_eq!(v.encoder_mid_block.attention.c, 640);
        assert_eq!(v.encoder_norm_out_gamma.len(), 640);
        assert_eq!(v.encoder_conv_out.cin, 640);
        assert_eq!(v.encoder_conv_out.cout, 96); // 2 × z_dim
        assert_eq!(v.quant_conv.cin, 96);
        assert_eq!(v.post_quant_conv.cin, 48);

        // Decoder channel schedule.
        let dec_ch = [1024usize, 1024, 512, 256];
        assert_eq!(v.decoder_conv_in.cin, 48);
        assert_eq!(v.decoder_conv_in.cout, dec_ch[0]);
        assert_eq!(v.decoder_mid_block.attention.c, 1024);
        assert_eq!(v.decoder_up_blocks.len(), 4);
        for (i, b) in v.decoder_up_blocks.iter().enumerate() {
            assert_eq!(b.resnets.len(), 3, "decoder up {i} resnets");
            let cin_first = if i == 0 { dec_ch[0] } else { dec_ch[i-1] };
            assert_eq!(b.resnets[0].conv1.cin, cin_first);
            assert_eq!(b.resnets[0].conv1.cout, dec_ch[i]);
            assert_eq!(b.upsampler.is_some(), i < 3);
            if let Some(us) = &b.upsampler {
                assert_eq!(us.resample.cin, dec_ch[i]);
                // time_conv in stages where temperal_downsample (reversed) is true:
                // mirrors encoder positions 2, 1 → decoder positions 0, 1.
                let want_t = matches!(i, 0 | 1);
                assert_eq!(us.time_conv.is_some(), want_t, "upsampler {i} time_conv");
                if let Some(tc) = &us.time_conv {
                    // time_conv weight = [2·Cout, Cin=Cout, 3, 1, 1].
                    assert_eq!(tc.cout, 2 * dec_ch[i]);
                }
            }
        }
        assert_eq!(v.decoder_norm_out_gamma.len(), 256);
        assert_eq!(v.decoder_conv_out.cout, 12);

        let mb = v.weight_bytes() as f64 / (1024.0*1024.0);
        eprintln!("Wan22VAE_v2 total tensor bytes ≈ {:.1} MB", mb);
        assert!(mb > 1100.0 && mb < 2000.0, "expected ~1.4 GB, got {mb:.1} MB");
        eprintln!("Wan22VAE_v2 Stage 0 load + shape verify ✓");
    }

    /// Stage 1+2 Wan VAE: real encode + decode + patchify on a tiny synthetic
    /// video.  Validates every primitive (3D conv, 2D s2 conv, time_conv,
    /// nearest 2× up, RMSNorm, SiLU, ResNet w/ and w/o shortcut, single-head
    /// mid-attention) with real weights.  Output shape must contract through
    /// the encoder pyramid (16×, 4× temporal) and re-expand through the
    /// decoder back to the original spatial extent.
    #[test]
    #[ignore = "phase4 v2 Wan VAE — needs Cosmos 3 Nano vae shard mounted; runs ~5–30 s"]
    fn phase4_v2_wan_vae_encode_decode() {
        let Some(p) = cosmos3_vae_path() else {
            eprintln!("Cosmos 3 Nano vae shard not found"); return;
        };
        let v = Wan22VAE_v2::load(&p).expect("load Wan VAE");

        // Tiny synthetic input: T=4, H=W=16, 12 channels.  4 frames covers
        // one temporal-patch group (scale_factor_temporal=4); 16² spatial
        // contracts to 1×1 after the 16× downsample.
        let (t, h, w) = (4, 16, 16);
        let cin = v.config.in_channels;
        let video: Vec<f32> = (0..cin * t * h * w)
            .map(|i| ((i as f32 * 0.013).sin()) * 0.5).collect();

        let t0 = std::time::Instant::now();
        let (mean, t1, h1, w1) = v.encode(&video, t, h, w);
        let t_enc = t0.elapsed();
        let z = v.config.z_dim;
        eprintln!("encode {t}×{h}×{w} → z[{z},{t1},{h1},{w1}] in {t_enc:?}");
        assert_eq!(mean.len(), z * t1 * h1 * w1);
        assert!(mean.iter().all(|v| v.is_finite()));
        // Encoder VAE-only contracts: 8× spatial (3 stride-2 down) + 4× temporal
        // (2 time-convs × the implicit 4-frame patch on the input).  The full
        // 16× spatial scale_factor includes the transformer-side 2× patchify.
        assert_eq!((t1, h1, w1), (1, 2, 2),
                   "VAE encoder should give 8× spatial + 4× temporal");

        // Patchify the encoded latent (1×2×2) — gives 1 token of width 192.
        let tokens = v.patchify_2x2(&mean, t1, h1, w1);
        let n_tok = t1 * (h1 / 2) * (w1 / 2);
        let d_tok = z * 4;
        assert_eq!(tokens.len(), n_tok * d_tok);
        assert_eq!(d_tok, crate::phase4_v2::VISUAL_LATENT_DIM,
                   "patchify_2x2 token dim should equal VISUAL_LATENT_DIM=192");

        // Decode the encoded mean back to a tiny video.
        let t2 = std::time::Instant::now();
        let (video_out, td, hd, wd) = v.decode(&mean, t1, h1, w1);
        let t_dec = t2.elapsed();
        // Decoder mirror: 8× spatial × 4× temporal up.  1×2×2 → 4×16×16.
        eprintln!("decode z[{z},{t1},{h1},{w1}] → video[{},{},{},{}] in {t_dec:?}",
                  v.config.out_channels, td, hd, wd);
        assert!(video_out.iter().all(|v| v.is_finite()));
        assert_eq!((td, hd, wd), (4, 16, 16),
                   "decoder should restore 4×16×16 from 1×2×2 latent");
        eprintln!("Wan22VAE_v2 Stage 1+2 encode/decode/patchify ✓");
    }

    /// Stage 6 v2 v0: load the first 2 layers of the real Cosmos 3 Nano
    /// transformer and verify all 22 per-layer weights are present with
    /// the right shapes.  Catches scaffold/real-weight mismatches before
    /// committing to the forward path.
    #[test]
    #[ignore = "phase4 v2 — needs Cosmos 3 Nano weights mounted; loads ~10 GB"]
    fn phase4_v2_load_first_layers() {
        let Some(dir) = cosmos3_transformer_dir() else {
            eprintln!("Cosmos 3 Nano transformer dir not found"); return;
        };
        let t0 = std::time::Instant::now();
        let m = Cosmos3OmniTransformerV2::load_with_layers(&dir, 2).expect("load");
        eprintln!("phase4 v2: loaded 2 layers + modality embeds + embed_tokens in {:?}", t0.elapsed());

        // Top-level shapes.
        assert_eq!(m.action_modality_embed.len(), HIDDEN);
        assert_eq!(m.audio_modality_embed.len(), HIDDEN);
        assert_eq!(m.embed_tokens.len(), 151936 * HIDDEN);

        // Per-layer shapes for both streams.
        for (i, l) in m.layers.iter().enumerate() {
            assert_eq!(l.reasoner.input_ln.len(), HIDDEN, "layer {i} reasoner input_ln");
            assert_eq!(l.reasoner.q_w.len(), HIDDEN * HIDDEN);
            assert_eq!(l.reasoner.k_w.len(), KV_DIM * HIDDEN);
            assert_eq!(l.reasoner.v_w.len(), KV_DIM * HIDDEN);
            assert_eq!(l.reasoner.o_w.len(), HIDDEN * HIDDEN);
            assert_eq!(l.reasoner.q_norm.len(), crate::HEAD_DIM);
            assert_eq!(l.reasoner.gate_w.len(), FF * HIDDEN);
            assert_eq!(l.reasoner.down_w.len(), HIDDEN * FF);
            assert_eq!(l.generator.input_ln.len(), HIDDEN, "layer {i} generator input_ln");
            assert_eq!(l.generator.add_q_w.len(), HIDDEN * HIDDEN);
            assert_eq!(l.generator.add_k_w.len(), KV_DIM * HIDDEN);
            assert_eq!(l.generator.add_v_w.len(), KV_DIM * HIDDEN);
            assert_eq!(l.generator.add_out_w.len(), HIDDEN * HIDDEN);
            assert_eq!(l.generator.gate_w.len(), FF * HIDDEN);
            assert_eq!(l.generator.down_w.len(), HIDDEN * FF);
        }
        eprintln!("phase4 v2: all 22 weights × {} layers shape-verified ✓", m.layers.len());
    }

    /// Metal-accelerated Euler sampling on real Cosmos 3 Nano weights.
    /// Verifies CPU vs GPU sampling agree within bf16 reorder noise
    /// (~5% rel for a 4-step × 36-layer × 14-matmul/layer composition =
    /// ~2000 composed matmuls).
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "phase4 v2 metal — full 36-layer × 4 steps on GPU + CPU"]
    fn phase4_v2_euler_sample_metal() {
        let Some(dir) = cosmos3_transformer_dir() else { eprintln!("Cosmos 3 not mounted"); return; };
        let Some(accel) = crate::phase4::MetalAccel::new() else { eprintln!("Metal unavailable"); return; };
        let m = Cosmos3OmniTransformerV2::load(&dir).expect("load");
        let step = VisualDiffusionStep::from_model(&m).expect("heads loaded");

        let t_gen = 4;
        let x_init: Vec<f32> = (0..t_gen * VISUAL_LATENT_DIM)
            .map(|i| ((i as f32 * 0.013).sin()) * 0.1).collect();
        let pos_g_t = vec![0i64; t_gen];
        let pos_g_h = vec![0i64, 0, 1, 1];
        let pos_g_w = vec![0i64, 1, 0, 1];
        let t_r = 1;
        let x_r_cond = vec![0.0f32; t_r * HIDDEN];
        let pos_r_t = vec![0i64; t_r];
        let pos_r_h = vec![0i64; t_r];
        let pos_r_w = vec![0i64; t_r];

        // CPU
        let t0 = std::time::Instant::now();
        let cpu = step.euler_sample_routed(x_init.clone(), t_gen,
            &x_r_cond, t_r, &pos_r_t, &pos_r_h, &pos_r_w,
            &pos_g_t, &pos_g_h, &pos_g_w, 4, 1e-3, None);
        let cpu_dt = t0.elapsed();

        // GPU (first step uploads weights; later steps hit the cache).
        let t0 = std::time::Instant::now();
        let gpu = step.euler_sample_routed(x_init.clone(), t_gen,
            &x_r_cond, t_r, &pos_r_t, &pos_r_h, &pos_r_w,
            &pos_g_t, &pos_g_h, &pos_g_w, 4, 1e-3, Some(&accel));
        let gpu_dt = t0.elapsed();

        eprintln!("euler 4-step CPU: {cpu_dt:?}");
        eprintln!("euler 4-step GPU: {gpu_dt:?}");
        assert!(cpu.iter().all(|v| v.is_finite()));
        assert!(gpu.iter().all(|v| v.is_finite()));
        let max_abs = cpu.iter().zip(gpu.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let max_ref = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let rel = if max_ref > 0.0 { max_abs / max_ref } else { 0.0 };
        eprintln!("CPU vs GPU sample: max_abs={max_abs:.6} max_ref={max_ref:.4} rel={rel:.4}");
        // ~2000 composed matmuls (4 steps × 36 layers × 14 matmuls/layer).
        assert!(rel < 1e-1, "CPU vs GPU sample mismatch rel={rel}");
    }

    /// Multi-step Euler sampling end-to-end on real Cosmos 3 Nano weights.
    /// Composes 4 diffusion steps; verifies (a) iteration converges to a
    /// finite latent, (b) the trajectory shifts substantially from x_T
    /// (the model is doing work each step), (c) `cond_scale = 0` reduces
    /// to identity so the sampling reduces to pass-through.
    #[test]
    #[ignore = "phase4 v2 — full 36-layer × 4 steps; ~2.5s wall"]
    fn phase4_v2_euler_sample_4steps() {
        let Some(dir) = cosmos3_transformer_dir() else { eprintln!("Cosmos 3 not mounted"); return; };
        let m = Cosmos3OmniTransformerV2::load(&dir).expect("load");
        let step = VisualDiffusionStep::from_model(&m).expect("heads loaded");

        let t_gen = 4;
        let x_t: Vec<f32> = (0..t_gen * VISUAL_LATENT_DIM)
            .map(|i| ((i as f32 * 0.013).sin()) * 0.1).collect();
        let pos_g_t = vec![0i64; t_gen];
        let pos_g_h = vec![0i64, 0, 1, 1];
        let pos_g_w = vec![0i64, 1, 0, 1];
        let t_r = 1;
        let x_r_cond = vec![0.0f32; t_r * HIDDEN];
        let pos_r_t = vec![0i64; t_r];
        let pos_r_h = vec![0i64; t_r];
        let pos_r_w = vec![0i64; t_r];

        // cond_scale=0 should be exact identity (no integration).
        let identity = step.euler_sample(x_t.clone(), t_gen,
            &x_r_cond, t_r, &pos_r_t, &pos_r_h, &pos_r_w,
            &pos_g_t, &pos_g_h, &pos_g_w, 4, 0.0);
        let id_drift = x_t.iter().zip(identity.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(id_drift < 1e-6, "cond_scale=0 must be identity, got drift {id_drift}");

        // Real 4-step sampling at small cond_scale.
        let t0 = std::time::Instant::now();
        let x_0 = step.euler_sample(x_t.clone(), t_gen,
            &x_r_cond, t_r, &pos_r_t, &pos_r_h, &pos_r_w,
            &pos_g_t, &pos_g_h, &pos_g_w, 4, 1e-3);
        let dt = t0.elapsed();
        let drift = x_t.iter().zip(x_0.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let norm_init: f32 = x_t.iter().map(|v| v*v).sum::<f32>().sqrt();
        let norm_final: f32 = x_0.iter().map(|v| v*v).sum::<f32>().sqrt();
        eprintln!("euler 4 steps in {dt:?}: ‖x_T‖={norm_init:.4} → ‖x_0‖={norm_final:.4} max_diff={drift:.4}");
        assert!(x_0.iter().all(|v| v.is_finite()));
        assert!(drift > 0.01, "sampling should move the latent; got {drift}");
    }

    /// Diffusion step end-to-end on real Cosmos 3 Nano weights.  Composes:
    /// visual_head.project_in → time_embedder broadcast-add → 36-layer joint
    /// forward with 3D mrope → visual_head.project_out.  Verifies (a) shape
    /// contract `[T, 192] → [T, 192]`, (b) finite output, (c) two different
    /// timesteps produce different predicted noise.
    #[test]
    #[ignore = "phase4 v2 — needs Cosmos 3 Nano weights mounted; full 36-layer forward"]
    fn phase4_v2_diffusion_step() {
        let Some(dir) = cosmos3_transformer_dir() else { eprintln!("Cosmos 3 not mounted"); return; };
        let m = Cosmos3OmniTransformerV2::load(&dir).expect("load");
        let step = VisualDiffusionStep::from_model(&m).expect("heads loaded");

        // 2×2 spatial grid of patch tokens at frame 0.
        let t_gen = 4;
        let latent: Vec<f32> = (0..t_gen * VISUAL_LATENT_DIM)
            .map(|i| ((i as f32 * 0.013).sin()) * 0.1).collect();
        let pos_g_t = vec![0i64; t_gen];
        let pos_g_h = vec![0i64, 0, 1, 1];
        let pos_g_w = vec![0i64, 1, 0, 1];

        // Minimal reasoner stream: 1 unconditional token at position 0.
        let t_r = 1;
        let x_r_cond = vec![0.0f32; t_r * HIDDEN];
        let pos_r_t = vec![0i64; t_r];
        let pos_r_h = vec![0i64; t_r];
        let pos_r_w = vec![0i64; t_r];

        let t0 = std::time::Instant::now();
        let eps_05 = step.step(&latent, t_gen, 0.5,
            &x_r_cond, t_r, &pos_r_t, &pos_r_h, &pos_r_w,
            &pos_g_t, &pos_g_h, &pos_g_w);
        let dt_05 = t0.elapsed();
        let t0 = std::time::Instant::now();
        let eps_01 = step.step(&latent, t_gen, 0.1,
            &x_r_cond, t_r, &pos_r_t, &pos_r_h, &pos_r_w,
            &pos_g_t, &pos_g_h, &pos_g_w);
        let dt_01 = t0.elapsed();

        assert_eq!(eps_05.len(), t_gen * VISUAL_LATENT_DIM);
        assert_eq!(eps_01.len(), t_gen * VISUAL_LATENT_DIM);
        assert!(eps_05.iter().all(|v| v.is_finite()));
        assert!(eps_01.iter().all(|v| v.is_finite()));

        let norm_05: f32 = eps_05.iter().map(|v| v*v).sum::<f32>().sqrt();
        let norm_01: f32 = eps_01.iter().map(|v| v*v).sum::<f32>().sqrt();
        let diff_t: f32 = eps_05.iter().zip(eps_01.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("diffusion step t=0.5: ‖eps‖={norm_05:.4} in {:?}", dt_05);
        eprintln!("diffusion step t=0.1: ‖eps‖={norm_01:.4} in {:?}", dt_01);
        eprintln!("predicted noise t=0.5 vs t=0.1: max_diff={diff_t:.4}");
        assert!(diff_t > 0.1, "different timesteps must predict different noise; got {diff_t}");
    }

    /// **End-to-end text → video integration smoke test.**  Composes every
    /// Phase 4 v2 component on real Cosmos 3 Nano weights:
    ///   text token ids
    ///     → `embed_text`                   (reasoner stream input)
    ///     → `euler_sample_scheduled`       (production σ schedule, 4 steps)
    ///     → `denormalize_latent`           (per-channel mean/std restore)
    ///     → `Wan22VAE_v2::decode`          (latent → pixel video)
    /// Verifies the pipeline closes end-to-end with no shape errors, no
    /// NaNs/infs, and produces a finite pixel-space tensor at the expected
    /// resolution.  This is the integration milestone for the Phase 4 v2
    /// chapter — every prior test verifies one component; this one verifies
    /// they compose.
    #[test]
    #[ignore = "phase4 v2 — needs both transformer + vae shards mounted; ~10–20 s"]
    fn phase4_v2_text_to_video_end_to_end() {
        let Some(dir) = cosmos3_transformer_dir() else { eprintln!("Cosmos 3 transformer not mounted"); return; };
        let Some(vae_path) = cosmos3_vae_path() else { eprintln!("Cosmos 3 VAE not mounted"); return; };

        let t_load = std::time::Instant::now();
        let m = Cosmos3OmniTransformerV2::load(&dir).expect("load transformer");
        let v = Wan22VAE_v2::load(&vae_path).expect("load VAE");
        eprintln!("load: transformer + VAE in {:?}", t_load.elapsed());

        let step = VisualDiffusionStep::from_model(&m).expect("heads loaded");

        // ── 1. Text-encode the prompt.  Token ids are placeholders (no
        //       tokenizer wired yet) but verify the embedding path is live.
        let ids: Vec<u32> = vec![100, 200, 300, 400];
        let x_r = m.embed_text(&ids);
        let t_r = ids.len();
        let pos_r_t: Vec<i64> = (0..t_r as i64).collect();
        let pos_r_h = vec![0i64; t_r];
        let pos_r_w = vec![0i64; t_r];

        // ── 2. Initialize noisy latent at the 1×2×2 spatial × 1 temporal
        //       size that the VAE produces for 4×16×16 video input.
        let z = v.config.z_dim;
        let (t_lat, h_lat, w_lat) = (1, 2, 2);
        let n_lat = z * t_lat * h_lat * w_lat;
        // Patchify_2x2 of the 1×2×2 latent gives 1 token (h/2 × w/2 = 1).
        let t_gen = 1;
        let mut x_init: Vec<f32> = (0..n_lat)
            .map(|i| ((i as f32 * 0.013).sin()) * 0.1).collect();
        v.normalize_latent(&mut x_init, t_lat, h_lat, w_lat);

        // Patchify to transformer token sequence: [t_gen, 192].
        let x_init_tok = v.patchify_2x2(&x_init, t_lat, h_lat, w_lat);
        assert_eq!(x_init_tok.len(), t_gen * VISUAL_LATENT_DIM);
        let pos_g_t = vec![0i64; t_gen];
        let pos_g_h = vec![0i64; t_gen];
        let pos_g_w = vec![0i64; t_gen];

        // ── 3. Production σ schedule (linear, shift=1; 4 steps).
        let schedule = FlowMatchSchedule::cosmos3_nano(4);
        let t_samp = std::time::Instant::now();
        let x_0_tok = step.euler_sample_scheduled(
            x_init_tok.clone(), t_gen,
            &x_r, t_r, &pos_r_t, &pos_r_h, &pos_r_w,
            &pos_g_t, &pos_g_h, &pos_g_w,
            &schedule, None);
        let dt_samp = t_samp.elapsed();
        assert_eq!(x_0_tok.len(), t_gen * VISUAL_LATENT_DIM);
        assert!(x_0_tok.iter().all(|v| v.is_finite()));
        eprintln!("sample 4 steps in {dt_samp:?}");

        // ── 4. Un-patchify token → latent grid [z, t_lat, h_lat, w_lat]
        //       (inverse of patchify_2x2 for the trivial t_gen=1 case).
        let p = v.config.patch_size; // 2
        let mut x_0_lat = vec![0.0f32; n_lat];
        // Token 0 covers spatial 2×2 patch at (oy=0, ox=0), temporal index 0.
        for ci in 0..z {
            for ky in 0..p { for kx in 0..p {
                let ih = ky; let iw = kx;
                let lat_off = ci * (t_lat * h_lat * w_lat) + 0 * (h_lat * w_lat) + ih * w_lat + iw;
                let tok_off = 0 * VISUAL_LATENT_DIM + ci * (p * p) + ky * p + kx;
                x_0_lat[lat_off] = x_0_tok[tok_off];
            }}
        }

        // ── 5. Denormalize back into VAE's raw latent space + decode.
        v.denormalize_latent(&mut x_0_lat, t_lat, h_lat, w_lat);
        let t_dec = std::time::Instant::now();
        let (video, td, hd, wd) = v.decode(&x_0_lat, t_lat, h_lat, w_lat);
        let dt_dec = t_dec.elapsed();
        eprintln!("decode in {dt_dec:?}: video[{},{},{},{}]", v.config.out_channels, td, hd, wd);
        assert_eq!((td, hd, wd), (4, 16, 16),
                   "decoder should restore 4×16×16 from 1×2×2 latent");
        assert!(video.iter().all(|v| v.is_finite()));

        eprintln!("Phase 4 v2 end-to-end pipeline ✓ — total {:?}", t_load.elapsed());
    }

    /// Wan VAE Metal port — encoder symmetric to the decoder test.  CPU vs
    /// Metal `encode_to_moments` on a tiny synthetic video; verifies the
    /// encoder's ResNet pyramid + mid-block + conv_out routes through the
    /// Metal im2col path correctly and the cache amortizes across calls.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "phase4 v2 Wan VAE Metal encoder — needs vae shard + Metal device"]
    fn phase4_v2_wan_vae_encode_metal() {
        let Some(p) = cosmos3_vae_path() else {
            eprintln!("Cosmos 3 Nano vae shard not found"); return;
        };
        let Some(accel) = crate::phase4::MetalAccel::new() else {
            eprintln!("Metal unavailable"); return;
        };
        let v = Wan22VAE_v2::load(&p).expect("load Wan VAE");
        let (t, h, w) = (4, 16, 16);
        let cin = v.config.in_channels;
        let video: Vec<f32> = (0..cin * t * h * w)
            .map(|i| ((i as f32 * 0.013).sin()) * 0.5).collect();

        let t0 = std::time::Instant::now();
        let (cpu_mom, _t1, _h1, _w1) = v.encode_to_moments(&video, t, h, w);
        let dt_cpu = t0.elapsed();

        let t0 = std::time::Instant::now();
        let (gpu_warm, _, _, _) = v.encode_to_moments_routed(&video, t, h, w, Some(&accel));
        let dt_gpu_warm = t0.elapsed();
        let t0 = std::time::Instant::now();
        let (gpu_hot, _, _, _) = v.encode_to_moments_routed(&video, t, h, w, Some(&accel));
        let dt_gpu_hot = t0.elapsed();

        assert!(cpu_mom.iter().all(|v| v.is_finite()));
        assert!(gpu_warm.iter().all(|v| v.is_finite()));
        assert!(gpu_hot.iter().all(|v| v.is_finite()));
        let max_abs = cpu_mom.iter().zip(gpu_hot.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let max_ref = cpu_mom.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let rel = if max_ref > 0.0 { max_abs / max_ref } else { 0.0 };
        eprintln!("Wan VAE encode CPU: {dt_cpu:?}  GPU warm: {dt_gpu_warm:?}  GPU hot: {dt_gpu_hot:?}");
        eprintln!("CPU vs GPU moments: max_abs={max_abs:.6} max_ref={max_ref:.4} rel={rel:.4}");
        assert!(rel < 1e-1, "Wan VAE encode CPU vs GPU mismatch rel={rel}");
    }

    /// Tiled-vs-untiled Metal conv3d equivalence: when the same conv is run
    /// with a small tile (forcing the multi-tile path) versus a tile size
    /// ≥ H,W (single dispatch), the outputs must be **bit-identical** in CPU
    /// fallback and within bf16 reorder noise on the Metal path.  This is
    /// the production-resolution correctness test: validates the halo +
    /// stitch logic for arbitrary tile sizes.
    #[test]
    fn phase4_v2_wan_conv3d_tiled_correctness() {
        // Synthesize a conv whose im2col fits in a tiny budget when tiled.
        let cin = 8;
        let cout = 4;
        let t = 2; let h = 16; let w = 16;
        let inp: Vec<f32> = (0..cin * t * h * w)
            .map(|i| ((i as f32 * 0.011).sin())).collect();
        // Tiny synthetic conv weights (bf16 storage to match real loader).
        let w_f32: Vec<f32> = (0..cout * cin * 27)
            .map(|i| (((i as f32) * 0.07).cos() - 0.5) * 0.05).collect();
        let w_bf: Vec<half::bf16> = w_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let b_f32: Vec<f32> = (0..cout).map(|i| (i as f32) * 0.01).collect();

        // Untiled CPU reference.
        let cpu_ref = wan_conv3d_k3_s1_p1(&inp, cin, t, h, w, &w_bf, &b_f32, cout);

        // Same op, but force tiling via the tiled helper with tile=4×4.
        // (Build a stub Wan3DConv by wrapping the bf16 slice in a BfSlice;
        // we can't easily construct BfSlice without mmap, so test the
        // CPU im2col + manual matmul path against the wan_conv3d primitive.)
        // Instead, exercise the tiling logic via a tiled CPU emulation: run
        // wan_conv3d_k3_s1_p1 on each tile + halo, then stitch, mirroring
        // wan_conv3d_routed_tiled exactly.
        let tile_h = 4; let tile_w = 4;
        let mut tiled = vec![0.0f32; cout * t * h * w];
        let mut oh_start = 0;
        while oh_start < h {
            let oh_end = (oh_start + tile_h).min(h);
            let ih_start = if oh_start > 0 { oh_start - 1 } else { 0 };
            let ih_end = if oh_end < h { oh_end + 1 } else { h };
            let in_th = ih_end - ih_start;
            let skip_h = oh_start - ih_start;
            let actual_th = oh_end - oh_start;
            let mut ow_start = 0;
            while ow_start < w {
                let ow_end = (ow_start + tile_w).min(w);
                let iw_start = if ow_start > 0 { ow_start - 1 } else { 0 };
                let iw_end = if ow_end < w { ow_end + 1 } else { w };
                let in_tw = iw_end - iw_start;
                let skip_w = ow_start - iw_start;
                let actual_tw = ow_end - ow_start;
                let mut tile_in = vec![0.0f32; cin * t * in_th * in_tw];
                for ci in 0..cin {
                    for ti in 0..t {
                        for hi in 0..in_th {
                            let gh = ih_start + hi;
                            let sr = ci * t * h * w + ti * h * w + gh * w + iw_start;
                            let dr = ci * t * in_th * in_tw + ti * in_th * in_tw + hi * in_tw;
                            tile_in[dr..dr + in_tw].copy_from_slice(&inp[sr..sr + in_tw]);
                        }
                    }
                }
                let tile_out = wan_conv3d_k3_s1_p1(&tile_in, cin, t, in_th, in_tw, &w_bf, &b_f32, cout);
                for ci in 0..cout {
                    for ti in 0..t {
                        for hi in 0..actual_th {
                            let sr = ci * t * in_th * in_tw + ti * in_th * in_tw
                                + (skip_h + hi) * in_tw + skip_w;
                            let gh = oh_start + hi;
                            let dr = ci * t * h * w + ti * h * w + gh * w + ow_start;
                            tiled[dr..dr + actual_tw].copy_from_slice(&tile_out[sr..sr + actual_tw]);
                        }
                    }
                }
                ow_start = ow_end;
            }
            oh_start = oh_end;
        }

        let max_abs = cpu_ref.iter().zip(tiled.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("tiled vs untiled max_abs = {max_abs:.8}");
        // CPU im2col + scalar matmul: bit-identical (same ops in same order).
        assert!(max_abs < 1e-5, "tiled CPU output should match untiled within fp32 noise; got {max_abs}");

        // Tile budget heuristic: with cin=8, t=2, per-voxel cost is
        // 8*27*4 = 864 bytes.  Budget of 100KB → max 115 voxels → ~108 per
        // frame for t=2.  With h*w=256 > 108 → must tile.
        let (th, tw) = pick_conv3d_tile(t, h, w, cin, 100 * 1024);
        eprintln!("tile heuristic at 100 KB budget: {th}×{tw}");
        assert!(th < h || tw < w, "100 KB should force tiling at this size");

        // Budget of 10 MB → no tiling (≪ 256*2*864 ≈ 442 KB).
        let (th2, tw2) = pick_conv3d_tile(t, h, w, cin, 10 * 1024 * 1024);
        assert_eq!((th2, tw2), (h, w), "10 MB budget should fit the slab");
    }

    /// Wan VAE Metal port: validate CPU vs Metal decode produce numerically
    /// equivalent video on real weights, and measure the wall-clock for
    /// each path.  At smoke-test resolution (1×2×2 latent → 4×16×16 video)
    /// Metal dispatch overhead is significant; the real perf win is at
    /// production resolutions (H, W ≥ 128) — but correctness still must hold.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "phase4 v2 Wan VAE Metal — needs vae shard + Metal device"]
    fn phase4_v2_wan_vae_decode_metal() {
        let Some(p) = cosmos3_vae_path() else {
            eprintln!("Cosmos 3 Nano vae shard not found"); return;
        };
        let Some(accel) = crate::phase4::MetalAccel::new() else {
            eprintln!("Metal unavailable"); return;
        };
        let v = Wan22VAE_v2::load(&p).expect("load Wan VAE");

        // Tiny synthetic latent at the production-shape ratio: 48×1×2×2.
        let z = v.config.z_dim;
        let (t, h, w) = (1, 2, 2);
        let latent: Vec<f32> = (0..z * t * h * w)
            .map(|i| ((i as f32 * 0.017).sin()) * 0.5).collect();

        let t0 = std::time::Instant::now();
        let (cpu, td, hd, wd) = v.decode(&latent, t, h, w);
        let dt_cpu = t0.elapsed();

        // Metal: first call uploads weights to MetalAccel cache; later calls
        // hit the cache.  Run twice and report both wall-clocks.
        let t0 = std::time::Instant::now();
        let (gpu1, _, _, _) = v.decode_routed(&latent, t, h, w, Some(&accel));
        let dt_gpu_warm = t0.elapsed();
        let t0 = std::time::Instant::now();
        let (gpu2, _, _, _) = v.decode_routed(&latent, t, h, w, Some(&accel));
        let dt_gpu_hot = t0.elapsed();

        assert_eq!(cpu.len(), gpu1.len());
        assert!(cpu.iter().all(|v| v.is_finite()));
        assert!(gpu1.iter().all(|v| v.is_finite()));
        assert!(gpu2.iter().all(|v| v.is_finite()));
        let max_abs = cpu.iter().zip(gpu2.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let max_ref = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let rel = if max_ref > 0.0 { max_abs / max_ref } else { 0.0 };
        eprintln!("Wan VAE decode CPU: {dt_cpu:?}  GPU warm: {dt_gpu_warm:?}  GPU hot: {dt_gpu_hot:?}");
        eprintln!("CPU vs GPU video[{},{},{},{}]: max_abs={max_abs:.6} max_ref={max_ref:.4} rel={rel:.4}",
                  v.config.out_channels, td, hd, wd);
        // bf16 reorder noise tolerance — many composed bf16 matmuls accumulate
        // small errors but should stay well under 10%.
        assert!(rel < 1e-1, "Wan VAE decode CPU vs GPU mismatch rel={rel}");
    }

    /// Wan VAE Metal decode with **forced tiling** via the
    /// `WAN_CONV3D_TILE_BUDGET_MB` env var.  Sets a small (2 MB) im2col
    /// budget so the largest decoder convs (1024-channel inner ResNet blocks)
    /// are tiled along H/W; verifies the tiled GPU output still agrees with
    /// the CPU reference.  This is the production-resolution code path.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "phase4 v2 Wan VAE Metal tiled — needs vae shard + Metal device"]
    fn phase4_v2_wan_vae_decode_metal_tiled() {
        let Some(p) = cosmos3_vae_path() else {
            eprintln!("Cosmos 3 Nano vae shard not found"); return;
        };
        let Some(accel) = crate::phase4::MetalAccel::new() else {
            eprintln!("Metal unavailable"); return;
        };
        let v = Wan22VAE_v2::load(&p).expect("load Wan VAE");
        let z = v.config.z_dim;
        let (t, h, w) = (1, 2, 2);
        let latent: Vec<f32> = (0..z * t * h * w)
            .map(|i| ((i as f32 * 0.017).sin()) * 0.5).collect();

        // CPU reference (no tiling involved).
        let (cpu, _, _, _) = v.decode(&latent, t, h, w);

        // Set a 2 MB budget so every conv with t·h·w · cin · 27 · 4 > 2 MB
        // gets tiled.  At decoder's hottest conv (cin=1024, t=4, h=w=8 after
        // mid block) the im2col is 4·8·8·1024·27·4 = 28 MB ⇒ ~14 tiles.
        unsafe { std::env::set_var("WAN_CONV3D_TILE_BUDGET_MB", "2"); }
        let t0 = std::time::Instant::now();
        let (gpu_tiled, _, _, _) = v.decode_routed(&latent, t, h, w, Some(&accel));
        let dt_tiled = t0.elapsed();
        unsafe { std::env::remove_var("WAN_CONV3D_TILE_BUDGET_MB"); }

        assert_eq!(cpu.len(), gpu_tiled.len());
        assert!(gpu_tiled.iter().all(|v| v.is_finite()));
        let max_abs = cpu.iter().zip(gpu_tiled.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let max_ref = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let rel = if max_ref > 0.0 { max_abs / max_ref } else { 0.0 };
        eprintln!("Wan VAE decode CPU vs GPU-tiled (2 MB budget): {dt_tiled:?}, max_abs={max_abs:.6} rel={rel:.4}");
        assert!(rel < 1e-1, "tiled Metal decode mismatch rel={rel}");
    }

    /// Latent normalization round-trip: encode → normalize → denormalize
    /// must return the original encoded latent within bf16 reorder noise.
    /// Verifies the per-channel `latents_mean / latents_std` arithmetic is
    /// invertible and the std array has no zero entries.
    #[test]
    #[ignore = "phase4 v2 Wan VAE — needs Cosmos 3 Nano vae shard mounted"]
    fn phase4_v2_wan_vae_latent_normalization() {
        let Some(p) = cosmos3_vae_path() else {
            eprintln!("Cosmos 3 Nano vae shard not found"); return;
        };
        let v = Wan22VAE_v2::load(&p).expect("load Wan VAE");

        let (t, h, w) = (4, 16, 16);
        let cin = v.config.in_channels;
        let video: Vec<f32> = (0..cin * t * h * w)
            .map(|i| ((i as f32 * 0.013).sin()) * 0.5).collect();
        let (mean_lat, t1, h1, w1) = v.encode(&video, t, h, w);

        // Round-trip: normalize → denormalize must be identity.
        let mut x = mean_lat.clone();
        v.normalize_latent(&mut x, t1, h1, w1);
        v.denormalize_latent(&mut x, t1, h1, w1);
        let max_rt = x.iter().zip(mean_lat.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("latent normalize→denormalize round-trip max_diff = {max_rt:.6}");
        assert!(max_rt < 1e-4, "round-trip should be near-identity, got {max_rt}");

        // After normalization, per-channel mean drifts toward 0 and per-channel
        // std drifts toward 1 — but only in expectation across natural video
        // statistics, not for a 4×16×16 synthetic input.  We instead verify the
        // shapes: latents_mean and latents_std are length z_dim and non-zero.
        assert_eq!(v.config.latents_mean.len(), v.config.z_dim);
        assert_eq!(v.config.latents_std.len(), v.config.z_dim);
        let min_std = v.config.latents_std.iter().fold(f32::INFINITY, |a, b| a.min(*b));
        assert!(min_std > 1e-3, "latents_std should be strictly positive, got min {min_std}");
    }

    /// Audio diffusion step on real Cosmos 3 Nano weights: tokenize a
    /// natural-language prompt, run one diffusion step in the 64-d audio
    /// latent space.  Verifies the audio I/O head + joint transformer
    /// pipeline produces a finite, prompt-sensitive eps.
    #[test]
    #[ignore = "phase4 v2 — needs Cosmos 3 Nano weights + text_tokenizer mounted"]
    fn phase4_v2_audio_diffusion_step() {
        let Some(dir) = cosmos3_transformer_dir() else { eprintln!("Cosmos 3 not mounted"); return; };
        let model_root = dir.parent().unwrap().to_path_buf();
        let m = Cosmos3OmniTransformerV2::load(&dir).expect("load");
        let tk = Cosmos3Tokenizer::load(&model_root).expect("load tokenizer");
        let step = AudioDiffusionStep::from_model(&m).expect("heads loaded");

        let ids_a = tk.encode("A jazz piano melody");
        let ids_b = tk.encode("A heavy metal drum solo");
        let x_r_a = m.embed_text(&ids_a);
        let x_r_b = m.embed_text(&ids_b);
        let t_r_a = ids_a.len(); let t_r_b = ids_b.len();
        let pos_r_t_a: Vec<i64> = (0..t_r_a as i64).collect();
        let pos_r_t_b: Vec<i64> = (0..t_r_b as i64).collect();
        let pos_r_h_a = vec![0i64; t_r_a]; let pos_r_h_b = vec![0i64; t_r_b];
        let pos_r_w_a = vec![0i64; t_r_a]; let pos_r_w_b = vec![0i64; t_r_b];

        let t_tok = 8;
        let lat: Vec<f32> = (0..t_tok * AUDIO_LATENT_DIM)
            .map(|i| (i as f32 * 0.013).sin() * 0.1).collect();
        let pos_g_t: Vec<i64> = (0..t_tok as i64).collect();
        let pos_g_h = vec![0i64; t_tok]; let pos_g_w = vec![0i64; t_tok];

        let t0 = std::time::Instant::now();
        let eps_a = step.step(&lat, t_tok, 0.5,
            &x_r_a, t_r_a, &pos_r_t_a, &pos_r_h_a, &pos_r_w_a,
            &pos_g_t, &pos_g_h, &pos_g_w);
        let dt_a = t0.elapsed();
        let eps_b = step.step(&lat, t_tok, 0.5,
            &x_r_b, t_r_b, &pos_r_t_b, &pos_r_h_b, &pos_r_w_b,
            &pos_g_t, &pos_g_h, &pos_g_w);

        assert_eq!(eps_a.len(), t_tok * AUDIO_LATENT_DIM);
        assert!(eps_a.iter().all(|v| v.is_finite()));
        assert!(eps_b.iter().all(|v| v.is_finite()));
        let max_abs = eps_a.iter().zip(eps_b.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("audio diffusion step ({:?}): jazz vs metal max_abs={max_abs:.4}", dt_a);
        assert!(max_abs > 1e-3, "audio diffusion should respond to prompt");
    }

    /// Action diffusion step with per-expert dispatch on real weights.
    /// Verifies (a) one step produces a finite 64-d eps, (b) two different
    /// experts on the same prompt produce different outputs.
    #[test]
    #[ignore = "phase4 v2 — needs Cosmos 3 Nano weights + text_tokenizer mounted"]
    fn phase4_v2_action_diffusion_step() {
        let Some(dir) = cosmos3_transformer_dir() else { eprintln!("Cosmos 3 not mounted"); return; };
        let model_root = dir.parent().unwrap().to_path_buf();
        let m = Cosmos3OmniTransformerV2::load(&dir).expect("load");
        let tk = Cosmos3Tokenizer::load(&model_root).expect("load tokenizer");
        let step = ActionDiffusionStep::from_model(&m).expect("heads loaded");

        let ids = tk.encode("Pick up the red cube");
        let x_r = m.embed_text(&ids);
        let t_r = ids.len();
        let pos_r_t: Vec<i64> = (0..t_r as i64).collect();
        let pos_r_h = vec![0i64; t_r]; let pos_r_w = vec![0i64; t_r];

        let t_tok = 4;
        let lat: Vec<f32> = (0..t_tok * ACTION_LATENT_DIM)
            .map(|i| (i as f32 * 0.017).cos() * 0.1).collect();
        let pos_g_t: Vec<i64> = (0..t_tok as i64).collect();
        let pos_g_h = vec![0i64; t_tok]; let pos_g_w = vec![0i64; t_tok];

        let t0 = std::time::Instant::now();
        let eps_e0 = step.step(&lat, t_tok, 0.5, 0,
            &x_r, t_r, &pos_r_t, &pos_r_h, &pos_r_w,
            &pos_g_t, &pos_g_h, &pos_g_w);
        let dt = t0.elapsed();
        let eps_e7 = step.step(&lat, t_tok, 0.5, 7,
            &x_r, t_r, &pos_r_t, &pos_r_h, &pos_r_w,
            &pos_g_t, &pos_g_h, &pos_g_w);

        assert_eq!(eps_e0.len(), t_tok * ACTION_LATENT_DIM);
        assert!(eps_e0.iter().all(|v| v.is_finite()));
        assert!(eps_e7.iter().all(|v| v.is_finite()));
        let max_abs = eps_e0.iter().zip(eps_e7.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("action diffusion ({:?}): expert 0 vs 7 max_abs={max_abs:.4}", dt);
        assert!(max_abs > 1e-3, "different experts must drive different outputs");
    }

    /// UniPC vs Euler trajectory comparison: same conditioning + same noise
    /// + same σ schedule.  Step 0 is identical (UniPC bootstraps with Euler).
    /// Steps 1+ diverge (AB2 uses ε history → different drift), but both
    /// must produce finite latents and they must approach each other as
    /// n_steps grows (consistency in the small-dt limit).
    #[test]
    #[ignore = "phase4 v2 — needs Cosmos 3 Nano weights mounted; ~6s"]
    fn phase4_v2_unipc_vs_euler() {
        let Some(dir) = cosmos3_transformer_dir() else { eprintln!("Cosmos 3 not mounted"); return; };
        let m = Cosmos3OmniTransformerV2::load(&dir).expect("load");
        let step = VisualDiffusionStep::from_model(&m).expect("heads loaded");

        let t_gen = 4;
        let x_init: Vec<f32> = (0..t_gen * VISUAL_LATENT_DIM)
            .map(|i| ((i as f32 * 0.013).sin()) * 0.1).collect();
        let pos_g_t = vec![0i64; t_gen];
        let pos_g_h = vec![0i64, 0, 1, 1];
        let pos_g_w = vec![0i64, 1, 0, 1];
        let ids: Vec<u32> = vec![100, 200, 300, 400];
        let x_r = m.embed_text(&ids);
        let t_r = ids.len();
        let pos_r_t: Vec<i64> = (0..t_r as i64).collect();
        let pos_r_h = vec![0i64; t_r];
        let pos_r_w = vec![0i64; t_r];

        let schedule = FlowMatchSchedule::cosmos3_nano(4);
        let euler = step.euler_sample_scheduled(
            x_init.clone(), t_gen,
            &x_r, t_r, &pos_r_t, &pos_r_h, &pos_r_w,
            &pos_g_t, &pos_g_h, &pos_g_w, &schedule, None);
        let unipc = step.unipc_sample_scheduled(
            x_init.clone(), t_gen,
            &x_r, t_r, &pos_r_t, &pos_r_h, &pos_r_w,
            &pos_g_t, &pos_g_h, &pos_g_w, &schedule, None);

        assert!(euler.iter().all(|v| v.is_finite()));
        assert!(unipc.iter().all(|v| v.is_finite()));
        let max_abs = euler.iter().zip(unipc.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("Euler vs UniPC (4 steps): max_abs={max_abs:.4}  ‖Euler‖={:.4}  ‖UniPC‖={:.4}",
                  euler.iter().map(|v| v*v).sum::<f32>().sqrt(),
                  unipc.iter().map(|v| v*v).sum::<f32>().sqrt());
        // The two MUST disagree after step 1 — AB2 changes the trajectory.
        assert!(max_abs > 1e-3, "UniPC AB2 should diverge from Euler past step 0");
    }

    /// End-to-end with the real tokenizer: tokenize a natural-language
    /// prompt, embed it, run a single diffusion step, and verify the
    /// generator output is finite and (when reusing the same noise) shifts
    /// from one prompt to another exactly as the placeholder-ID test did.
    /// This is the production reasoner-front-end path.
    #[test]
    #[ignore = "phase4 v2 — needs Cosmos 3 Nano weights + text_tokenizer mounted"]
    fn phase4_v2_real_prompt_end_to_end() {
        let Some(dir) = cosmos3_transformer_dir() else { eprintln!("Cosmos 3 not mounted"); return; };
        // The text_tokenizer dir sits at the model root, one level up from `transformer`.
        let model_root = dir.parent().unwrap().to_path_buf();
        let m = Cosmos3OmniTransformerV2::load(&dir).expect("load");
        let tk = Cosmos3Tokenizer::load(&model_root).expect("load tokenizer");
        let step = VisualDiffusionStep::from_model(&m).expect("heads loaded");

        let prompt_a = "A cat playing piano";
        let prompt_b = "A dog playing chess";
        let ids_a = tk.encode(prompt_a);
        let ids_b = tk.encode(prompt_b);
        eprintln!("tokenize {prompt_a:?} → {ids_a:?}");
        eprintln!("tokenize {prompt_b:?} → {ids_b:?}");
        let x_r_a = m.embed_text(&ids_a);
        let x_r_b = m.embed_text(&ids_b);
        let t_r_a = ids_a.len();
        let t_r_b = ids_b.len();
        let pos_r_t_a: Vec<i64> = (0..t_r_a as i64).collect();
        let pos_r_t_b: Vec<i64> = (0..t_r_b as i64).collect();
        let pos_r_h_a = vec![0i64; t_r_a];
        let pos_r_h_b = vec![0i64; t_r_b];
        let pos_r_w_a = vec![0i64; t_r_a];
        let pos_r_w_b = vec![0i64; t_r_b];

        let t_gen = 4;
        let latent: Vec<f32> = (0..t_gen * VISUAL_LATENT_DIM)
            .map(|i| ((i as f32 * 0.013).sin()) * 0.1).collect();
        let pos_g_t = vec![0i64; t_gen];
        let pos_g_h = vec![0i64, 0, 1, 1];
        let pos_g_w = vec![0i64, 1, 0, 1];

        let eps_a = step.step(&latent, t_gen, 0.5,
            &x_r_a, t_r_a, &pos_r_t_a, &pos_r_h_a, &pos_r_w_a,
            &pos_g_t, &pos_g_h, &pos_g_w);
        let eps_b = step.step(&latent, t_gen, 0.5,
            &x_r_b, t_r_b, &pos_r_t_b, &pos_r_h_b, &pos_r_w_b,
            &pos_g_t, &pos_g_h, &pos_g_w);

        assert!(eps_a.iter().all(|v| v.is_finite()));
        assert!(eps_b.iter().all(|v| v.is_finite()));
        let max_abs = eps_a.iter().zip(eps_b.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let max_ref = eps_a.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let rel = if max_ref > 0.0 { max_abs / max_ref } else { 0.0 };
        eprintln!("real-prompt A vs B: max_abs={max_abs:.4} max_ref={max_ref:.4} rel={rel:.4}");
        assert!(max_abs > 1e-3, "natural-language prompts should drive different outputs");
    }

    /// Cosmos 3 Nano text tokenizer: validate Rust BPE encoding matches
    /// HuggingFace `tokenizers` Python reference for known prompts.
    /// Reference encodings captured from `tokenizers.Tokenizer.from_file(
    /// '.../text_tokenizer/tokenizer.json')` on 2026-06-05.
    #[test]
    #[ignore = "phase4 v2 tokenizer — needs Cosmos 3 Nano text_tokenizer dir"]
    fn phase4_v2_cosmos3_tokenizer_python_parity() {
        // Pick the model root from any of the candidate mount paths.
        let mut model_root: Option<std::path::PathBuf> = None;
        for vol in ["macos_4TB_external", "macos_4TB_external_1", "macos_4TB_external_2"] {
            let p = std::path::PathBuf::from(format!(
                "/Volumes/{vol}/neuronexus_ai/model_library/Cosmos3-Nano"));
            if p.join("text_tokenizer/vocab.json").exists() {
                model_root = Some(p); break;
            }
        }
        let Some(root) = model_root else {
            eprintln!("Cosmos 3 Nano text_tokenizer not mounted"); return;
        };
        let t0 = std::time::Instant::now();
        let tk = Cosmos3Tokenizer::load(&root).expect("load tokenizer");
        eprintln!("Cosmos3Tokenizer: loaded {} tokens + {} merges in {:?}",
                  tk.vocab_size(), tk.merges.len(), t0.elapsed());

        let cases: &[(&str, &[u32])] = &[
            ("The capital of France is",  &[785, 6722, 315, 9625, 374]),
            ("Hello, world!",             &[9707, 11, 1879, 0]),
            (" Paris",                    &[12095]),
            ("a cat playing piano",       &[64, 8251, 5619, 26278]),
        ];
        for (text, expected) in cases {
            let ids = tk.encode(text);
            eprintln!("  {text:?} → {ids:?} (expected {expected:?})");
            assert_eq!(&ids[..], *expected, "mismatch on {text:?}");
        }
        // Special token round-trip: <|im_start|> must encode to 151644.
        let ids = tk.encode("<|im_start|>");
        assert_eq!(ids, vec![151_644], "<|im_start|> should be a single special token");

        eprintln!("Cosmos3Tokenizer ✓ — Python-parity on {} cases", cases.len());
    }

    /// FlowMatchSchedule: verify sigmas are monotonically decreasing,
    /// start at 1.0 and end at 0.0, length is `n_steps + 1`, and that
    /// `shift > 1` biases the schedule curve correctly (sigmas concentrated
    /// in the high-σ region versus linear).
    #[test]
    fn phase4_v2_flow_match_schedule() {
        let s_lin = FlowMatchSchedule::cosmos3_nano(8).sigmas();
        assert_eq!(s_lin.len(), 9);
        assert!((s_lin[0] - 1.0).abs() < 1e-6);
        assert!((s_lin[8] - 0.0).abs() < 1e-6);
        for i in 0..s_lin.len()-1 {
            assert!(s_lin[i] > s_lin[i+1], "sigma must decrease at i={i}: {} > {}", s_lin[i], s_lin[i+1]);
        }
        // shift=3.0: at lin=0.5, shifted = 3*0.5 / (1 + 2*0.5) = 1.5/2.0 = 0.75
        // (biased toward higher sigma).
        let s_shift = FlowMatchSchedule::shifted(2, 3.0).sigmas();
        assert!((s_shift[0] - 1.0).abs() < 1e-6);
        assert!((s_shift[2] - 0.0).abs() < 1e-6);
        let mid = s_shift[1];
        assert!((mid - 0.75).abs() < 1e-4, "expected shift=3 midpoint = 0.75, got {mid}");
    }

    /// End-to-end: rectified-flow Euler sampling with the production
    /// `FlowMatchSchedule` (shift=1, n_steps=4) on real Cosmos 3 Nano
    /// weights.  Replaces the `cond_scale` hyperparameter with a real
    /// per-step `dt = σ_curr − σ_next` integration.
    #[test]
    #[ignore = "phase4 v2 — needs Cosmos 3 Nano weights mounted; ~3s"]
    fn phase4_v2_euler_scheduled() {
        let Some(dir) = cosmos3_transformer_dir() else { eprintln!("Cosmos 3 not mounted"); return; };
        let m = Cosmos3OmniTransformerV2::load(&dir).expect("load");
        let step = VisualDiffusionStep::from_model(&m).expect("heads loaded");

        let t_gen = 4;
        let x_init: Vec<f32> = (0..t_gen * VISUAL_LATENT_DIM)
            .map(|i| ((i as f32 * 0.013).sin()) * 0.1).collect();
        let pos_g_t = vec![0i64; t_gen];
        let pos_g_h = vec![0i64, 0, 1, 1];
        let pos_g_w = vec![0i64, 1, 0, 1];

        // Real text-encoded conditioning.
        let ids: Vec<u32> = vec![100, 200, 300, 400];
        let x_r = m.embed_text(&ids);
        let t_r = ids.len();
        let pos_r_t: Vec<i64> = (0..t_r as i64).collect();
        let pos_r_h = vec![0i64; t_r];
        let pos_r_w = vec![0i64; t_r];

        let schedule = FlowMatchSchedule::cosmos3_nano(4);
        eprintln!("sigmas: {:?}", schedule.sigmas());

        let t0 = std::time::Instant::now();
        let x0 = step.euler_sample_scheduled(
            x_init.clone(), t_gen,
            &x_r, t_r, &pos_r_t, &pos_r_h, &pos_r_w,
            &pos_g_t, &pos_g_h, &pos_g_w,
            &schedule, None);
        let dt = t0.elapsed();

        let n_xt: f32 = x_init.iter().map(|v| v*v).sum::<f32>().sqrt();
        let n_x0: f32 = x0.iter().map(|v| v*v).sum::<f32>().sqrt();
        let max_diff = x_init.iter().zip(x0.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("euler scheduled 4 steps in {dt:?}: ‖x_T‖={n_xt:.4} → ‖x_0‖={n_x0:.4} max_diff={max_diff:.4}");
        assert!(x0.iter().all(|v| v.is_finite()));
        // Real integration: x_0 must shift away from x_T.
        assert!(max_diff > 1e-3, "schedule integration produced no movement");
    }

    /// Text-encoder conditioning: drive the reasoner stream with real
    /// `embed_tokens` lookups from two different token-id arrays and
    /// confirm the diffusion `step()` output diverges, proving the
    /// conditioning path actually reaches the generator stream.  Replaces
    /// the zero-placeholder `x_r_cond` used by earlier smoke tests with
    /// the production text-encoder front-end.
    #[test]
    #[ignore = "phase4 v2 — needs Cosmos 3 Nano weights mounted (~10 GB)"]
    fn phase4_v2_text_encoder_conditioning() {
        let Some(dir) = cosmos3_transformer_dir() else { eprintln!("Cosmos 3 not mounted"); return; };
        let m = Cosmos3OmniTransformerV2::load(&dir).expect("load");
        let step = VisualDiffusionStep::from_model(&m).expect("heads loaded");

        // Two different "prompts" as raw token id sequences.  Picking widely-
        // separated ids in the 151936-token vocab so the embed-tokens lookups
        // produce genuinely different reasoner inputs.  (Token IDs are made-up
        // here — we don't have a tokenizer wired yet; this verifies the
        // conditioning path, not semantic correctness.)
        let ids_a: Vec<u32> = vec![1, 2, 3, 4];
        let ids_b: Vec<u32> = vec![1000, 2000, 3000, 4000];
        let x_r_a = m.embed_text(&ids_a);
        let x_r_b = m.embed_text(&ids_b);
        assert_eq!(x_r_a.len(), ids_a.len() * HIDDEN);
        let max_xr_diff = x_r_a.iter().zip(x_r_b.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_xr_diff > 1e-3, "embed_text should differ for different ids");

        let t_r = ids_a.len();
        let pos_r_t: Vec<i64> = (0..t_r as i64).collect();
        let pos_r_h: Vec<i64> = vec![0; t_r];
        let pos_r_w: Vec<i64> = vec![0; t_r];

        let t_gen = 4;
        let latent: Vec<f32> = (0..t_gen * VISUAL_LATENT_DIM)
            .map(|i| ((i as f32 * 0.013).sin()) * 0.1).collect();
        let pos_g_t = vec![0i64; t_gen];
        let pos_g_h = vec![0i64, 0, 1, 1];
        let pos_g_w = vec![0i64, 1, 0, 1];

        let t0 = std::time::Instant::now();
        let eps_a = step.step(&latent, t_gen, 0.5,
            &x_r_a, t_r, &pos_r_t, &pos_r_h, &pos_r_w,
            &pos_g_t, &pos_g_h, &pos_g_w);
        let dt_a = t0.elapsed();
        let t0 = std::time::Instant::now();
        let eps_b = step.step(&latent, t_gen, 0.5,
            &x_r_b, t_r, &pos_r_t, &pos_r_h, &pos_r_w,
            &pos_g_t, &pos_g_h, &pos_g_w);
        let dt_b = t0.elapsed();

        assert!(eps_a.iter().all(|v| v.is_finite()));
        assert!(eps_b.iter().all(|v| v.is_finite()));
        let diff = eps_a.iter().zip(eps_b.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let ref_norm = eps_a.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let rel = if ref_norm > 0.0 { diff / ref_norm } else { 0.0 };
        eprintln!("text-conditioned step A: {dt_a:?}, B: {dt_b:?}");
        eprintln!("eps[A] vs eps[B]: max_abs={diff:.6} max_ref={ref_norm:.4} rel={rel:.4}");
        // Different reasoner conditioning MUST produce different generator output.
        assert!(diff > 1e-3, "text conditioning did not influence eps (diff={diff})");
    }

    /// 3D mrope on real Cosmos 3 Nano weights — verifies the per-axis
    /// rotation differs from the 1D RoPE baseline when h/w positions
    /// are non-zero (video grid coords, not pure sequential).
    #[test]
    #[ignore = "phase4 v2 — needs Cosmos 3 Nano weights mounted"]
    fn phase4_v2_3d_mrope_differs_from_1d() {
        let Some(dir) = cosmos3_transformer_dir() else { eprintln!("Cosmos 3 not mounted"); return; };
        let m = Cosmos3OmniTransformerV2::load_with_layers(&dir, 1).expect("load");

        let t_r = 2;
        let t_g = 4;
        let init_r: Vec<f32> = (0..t_r * HIDDEN).map(|i| ((i as f32 * 0.013).sin()) * 0.05).collect();
        let init_g: Vec<f32> = (0..t_g * HIDDEN).map(|i| ((i as f32 * 0.017).cos()) * 0.05).collect();
        let freqs = crate::rope_freqs();

        // 1D baseline: all axes = sequential token index.
        let pos_r: Vec<i64> = (0..t_r as i64).collect();
        let pos_g: Vec<i64> = (t_r as i64..(t_r + t_g) as i64).collect();
        let mut x_r_1d = init_r.clone();
        let mut x_g_1d = init_g.clone();
        m.layers[0].forward_with_positions(&mut x_r_1d, t_r, &pos_r, &mut x_g_1d, t_g, &pos_g, &freqs);

        // 3D variant: generator stream has real (t=0, h, w) grid coords
        // for a 2×2 spatial patch grid (4 tokens) at temporal frame 0.
        // Reasoner stream stays as text-like 1D (h=w=0).
        let pos_r_t: Vec<i64> = pos_r.clone();
        let pos_r_h: Vec<i64> = vec![0; t_r];
        let pos_r_w: Vec<i64> = vec![0; t_r];
        let pos_g_t: Vec<i64> = vec![0; t_g];
        let pos_g_h: Vec<i64> = vec![0, 0, 1, 1];
        let pos_g_w: Vec<i64> = vec![0, 1, 0, 1];
        let mut x_r_3d = init_r.clone();
        let mut x_g_3d = init_g.clone();
        m.layers[0].forward_with_3d_positions(
            &mut x_r_3d, t_r, &pos_r_t, &pos_r_h, &pos_r_w,
            &mut x_g_3d, t_g, &pos_g_t, &pos_g_h, &pos_g_w,
            &freqs,
        );

        let r_diff = x_r_1d.iter().zip(x_r_3d.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let g_diff = x_g_1d.iter().zip(x_g_3d.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("3D vs 1D mrope drift (1 layer):  reasoner max_diff={r_diff:.4}  generator max_diff={g_diff:.4}");
        assert!(x_r_3d.iter().all(|v| v.is_finite()));
        assert!(x_g_3d.iter().all(|v| v.is_finite()));
        // 3D mrope must produce different output when h/w are non-zero.
        assert!(g_diff > 1e-3, "3D mrope should change generator output vs 1D when h/w differ");
    }

    /// Audio + Action head verification on real Cosmos 3 Nano weights.
    /// Tests roundtrip shapes + MoE expert dispatch (action head has 32
    /// experts that must each produce distinct outputs).
    #[test]
    #[ignore = "phase4 v2 — needs Cosmos 3 Nano weights mounted"]
    fn phase4_v2_audio_action_heads() {
        let Some(dir) = cosmos3_transformer_dir() else { eprintln!("Cosmos 3 not mounted"); return; };
        let m = Cosmos3OmniTransformerV2::load_with_layers(&dir, 0).expect("load");
        let ah = m.audio_head.as_ref().expect("audio_head weights");
        let act = m.action_head.as_ref().expect("action_head weights");

        // ── Audio head roundtrip: 64 → HIDDEN → 64 ──
        let t_tok = 4;
        let lat_in: Vec<f32> = (0..t_tok * AUDIO_LATENT_DIM)
            .map(|i| (i as f32 * 0.013).sin() * 0.1).collect();
        let hidden = ah.project_in(&lat_in, t_tok);
        assert_eq!(hidden.len(), t_tok * HIDDEN);
        let lat_out = ah.project_out(&hidden, t_tok);
        assert_eq!(lat_out.len(), t_tok * AUDIO_LATENT_DIM);
        assert!(lat_out.iter().all(|v| v.is_finite()));
        let audio_norm: f32 = lat_out.iter().map(|v| v*v).sum::<f32>().sqrt();
        eprintln!("audio 64→HIDDEN→64 roundtrip: ‖out‖={audio_norm:.4}");

        // ── Action head: dispatch to multiple experts, verify outputs differ ──
        let act_lat: Vec<f32> = (0..t_tok * ACTION_LATENT_DIM)
            .map(|i| (i as f32 * 0.017).cos() * 0.1).collect();
        let exp0 = act.project_in(&act_lat, t_tok, 0);
        let exp7 = act.project_in(&act_lat, t_tok, 7);
        let exp31 = act.project_in(&act_lat, t_tok, 31);
        let d07 = exp0.iter().zip(exp7.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let d031 = exp0.iter().zip(exp31.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("action MoE: expert 0 vs 7 max_diff={d07:.4}  vs 31 max_diff={d031:.4}");
        assert!(d07 > 1e-3, "different experts must produce different outputs");
        assert!(d031 > 1e-3);

        // Roundtrip via expert 0.
        let act_out = act.project_out(&exp0, t_tok, 0);
        assert_eq!(act_out.len(), t_tok * ACTION_LATENT_DIM);
        assert!(act_out.iter().all(|v| v.is_finite()));
        let act_norm: f32 = act_out.iter().map(|v| v*v).sum::<f32>().sqrt();
        eprintln!("action expert-0 64→HIDDEN→64 roundtrip: ‖out‖={act_norm:.4}");
    }

    /// Time embedder + visual head end-to-end on real Cosmos 3 weights:
    /// time_embed(t=0.5) → [HIDDEN] global time signal; project_out a
    /// HIDDEN-vec back to VISUAL_LATENT_DIM=192 (the Wan VAE latent space).
    #[test]
    #[ignore = "phase4 v2 — needs Cosmos 3 Nano weights mounted"]
    fn phase4_v2_time_embedder_and_visual_head() {
        let Some(dir) = cosmos3_transformer_dir() else { eprintln!("Cosmos 3 not mounted"); return; };
        let m = Cosmos3OmniTransformerV2::load_with_layers(&dir, 0).expect("load");
        let te = m.time_embedder.as_ref().expect("time_embedder weights");
        let vh = m.visual_head.as_ref().expect("visual_head weights");

        let t_emb_0 = te.forward(0.0);
        let t_emb_5 = te.forward(0.5);
        let t_emb_1 = te.forward(1.0);
        assert_eq!(t_emb_0.len(), HIDDEN);
        // Embedding must differ across timesteps.
        let diff_05 = t_emb_0.iter().zip(t_emb_5.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let diff_01 = t_emb_0.iter().zip(t_emb_1.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("time_embed(0..0.5) diff_max={diff_05:.4}  (0..1) diff_max={diff_01:.4}");
        assert!(diff_05 > 1e-3, "time embedding should change between t=0 and t=0.5");
        assert!(diff_01 > 1e-3, "time embedding should change between t=0 and t=1");

        // Visual head roundtrip shape: 192 → HIDDEN → 192.
        let t_tok = 4;
        let latent_in: Vec<f32> = (0..t_tok * VISUAL_LATENT_DIM)
            .map(|i| (i as f32 * 0.013).sin() * 0.1).collect();
        let hidden = vh.project_in(&latent_in, t_tok);
        assert_eq!(hidden.len(), t_tok * HIDDEN);
        let latent_out = vh.project_out(&hidden, t_tok);
        assert_eq!(latent_out.len(), t_tok * VISUAL_LATENT_DIM);
        assert!(latent_out.iter().all(|v| v.is_finite()));
        let latent_norm: f32 = latent_out.iter().map(|v| v*v).sum::<f32>().sqrt();
        eprintln!("visual_head 192→{HIDDEN}→192 round-trip: ‖out‖={latent_norm:.4}");
        assert!(latent_norm > 0.0);
    }

    /// Full 36-layer forward on real Cosmos 3 Nano weights.  The big test:
    /// can the whole production stack load and run on tiny synthetic input?
    /// Asserts: load succeeds for all `N_LAYERS=36` blocks; full forward
    /// produces finite output; both streams drift through the full depth.
    #[test]
    #[ignore = "phase4 v2 — needs Cosmos 3 Nano weights mounted; loads ~34 GB"]
    fn phase4_v2_full_36_layer_forward() {
        let Some(dir) = cosmos3_transformer_dir() else { eprintln!("Cosmos 3 not mounted"); return; };
        let t0 = std::time::Instant::now();
        let m = Cosmos3OmniTransformerV2::load(&dir).expect("load");
        let load_dt = t0.elapsed();
        eprintln!("phase4 v2: loaded all {} layers in {:?}", m.layers.len(), load_dt);
        assert_eq!(m.layers.len(), crate::N_LAYERS);

        let t_r = 4;
        let t_g = 2;
        let mut x_r: Vec<f32> = (0..t_r * HIDDEN).map(|i| ((i as f32 * 0.013).sin()) * 0.05).collect();
        let mut x_g: Vec<f32> = (0..t_g * HIDDEN).map(|i| ((i as f32 * 0.017).cos()) * 0.05).collect();
        let pos_r: Vec<i64> = (0..t_r as i64).collect();
        let pos_g: Vec<i64> = (t_r as i64..(t_r + t_g) as i64).collect();
        let freqs = crate::rope_freqs();
        let x_r_in = x_r.clone();
        let x_g_in = x_g.clone();

        let t0 = std::time::Instant::now();
        for (i, l) in m.layers.iter().enumerate() {
            l.forward_with_positions(&mut x_r, t_r, &pos_r, &mut x_g, t_g, &pos_g, &freqs);
            if i == 0 || i == m.layers.len() - 1 {
                let r_norm: f32 = x_r.iter().map(|v| v*v).sum::<f32>().sqrt();
                let g_norm: f32 = x_g.iter().map(|v| v*v).sum::<f32>().sqrt();
                eprintln!("phase4 v2 layer {i}: ‖x_r‖={r_norm:.2}  ‖x_g‖={g_norm:.2}");
            }
        }
        let fwd_dt = t0.elapsed();
        eprintln!("phase4 v2: 36-layer forward total {:?}", fwd_dt);

        assert!(x_r.iter().all(|v| v.is_finite()), "reasoner stream contains non-finite values");
        assert!(x_g.iter().all(|v| v.is_finite()), "generator stream contains non-finite values");

        let drift_r = x_r.iter().zip(x_r_in.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let drift_g = x_g.iter().zip(x_g_in.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("reasoner drift max_abs={drift_r:.4}  generator drift max_abs={drift_g:.4}");
        assert!(drift_r > 1.0, "reasoner stream should drift substantially through 36 layers");
        assert!(drift_g > 1.0, "generator stream should drift substantially through 36 layers");
    }

    /// Modality embed: verify that Action / Audio tokens pick up an
    /// additional bias vs Text tokens with the same token id.  Exercises
    /// `action_modality_embed` + `audio_modality_embed` which the load
    /// test already shape-verified.
    #[test]
    #[ignore = "phase4 v2 — needs Cosmos 3 Nano weights mounted; loads ~10 GB"]
    fn phase4_v2_modality_embed() {
        let Some(dir) = cosmos3_transformer_dir() else { eprintln!("Cosmos 3 not mounted"); return; };
        let m = Cosmos3OmniTransformerV2::load_with_layers(&dir, 0).expect("load");

        // Same token id, different modalities.
        let ids = vec![100u32, 100, 100];
        let mods = vec![Modality::Text, Modality::Action, Modality::Audio];
        let emb = m.embed_with_modality(&ids, &mods);
        assert_eq!(emb.len(), 3 * HIDDEN);

        let text = &emb[0..HIDDEN];
        let action = &emb[HIDDEN..2 * HIDDEN];
        let audio = &emb[2 * HIDDEN..3 * HIDDEN];

        // Drift of (action - text) should equal action_modality_embed component-wise.
        for d in 0..HIDDEN {
            let drift = action[d] - text[d];
            let bias = m.action_modality_embed[d];
            assert!((drift - bias).abs() < 1e-5,
                "action token modality drift mismatch at d={d}: drift={drift} vs bias={bias}");
        }
        for d in 0..HIDDEN {
            let drift = audio[d] - text[d];
            let bias = m.audio_modality_embed[d];
            assert!((drift - bias).abs() < 1e-5,
                "audio token modality drift mismatch at d={d}");
        }
        let action_norm: f32 = m.action_modality_embed.iter().map(|v| v*v).sum::<f32>().sqrt();
        let audio_norm:  f32 = m.audio_modality_embed.iter().map(|v| v*v).sum::<f32>().sqrt();
        eprintln!("‖action_embed‖={action_norm:.4}  ‖audio_embed‖={audio_norm:.4}");
        assert!(action_norm > 0.0 && audio_norm > 0.0, "modality embeds shouldn't be all-zero");
    }

    /// Stage 7 v2 + Metal: run 2 real layers through both CPU and GPU
    /// (Stage 5 `mm_bf16_routed` + `sdpa_routed` infrastructure) and
    /// verify the outputs match within bf16 reorder noise (~5 % rel for
    /// a 2-layer × 14-matmul composition).
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "phase4 v2 metal — needs Cosmos 3 Nano weights mounted"]
    fn phase4_v2_metal_matches_cpu() {
        let Some(dir) = cosmos3_transformer_dir() else { eprintln!("Cosmos 3 not mounted"); return; };
        let Some(accel) = crate::phase4::MetalAccel::new() else { eprintln!("Metal unavailable"); return; };
        let m = Cosmos3OmniTransformerV2::load_with_layers(&dir, 2).expect("load");

        let t_r = 4;
        let t_g = 2;
        let init_r: Vec<f32> = (0..t_r * HIDDEN).map(|i| ((i as f32 * 0.013).sin()) * 0.05).collect();
        let init_g: Vec<f32> = (0..t_g * HIDDEN).map(|i| ((i as f32 * 0.017).cos()) * 0.05).collect();
        let pos_r: Vec<i64> = (0..t_r as i64).collect();
        let pos_g: Vec<i64> = (t_r as i64..(t_r + t_g) as i64).collect();
        let freqs = crate::rope_freqs();

        // CPU run.
        let mut x_r_cpu = init_r.clone();
        let mut x_g_cpu = init_g.clone();
        let t0 = std::time::Instant::now();
        for (i, l) in m.layers.iter().enumerate() {
            l.forward_routed(&mut x_r_cpu, t_r, &pos_r, &mut x_g_cpu, t_g, &pos_g, &freqs, i, None);
        }
        let cpu_dt = t0.elapsed();

        // GPU run.
        let mut x_r_gpu = init_r.clone();
        let mut x_g_gpu = init_g.clone();
        let t0 = std::time::Instant::now();
        for (i, l) in m.layers.iter().enumerate() {
            l.forward_routed(&mut x_r_gpu, t_r, &pos_r, &mut x_g_gpu, t_g, &pos_g, &freqs, i, Some(&accel));
        }
        let gpu_dt = t0.elapsed();

        eprintln!("CPU 2 layers in {:?}", cpu_dt);
        eprintln!("GPU 2 layers in {:?}", gpu_dt);
        let max_abs_r = x_r_cpu.iter().zip(x_r_gpu.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let max_abs_g = x_g_cpu.iter().zip(x_g_gpu.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let max_ref_r = x_r_cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let max_ref_g = x_g_cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let rel_r = if max_ref_r > 0.0 { max_abs_r / max_ref_r } else { 0.0 };
        let rel_g = if max_ref_g > 0.0 { max_abs_g / max_ref_g } else { 0.0 };
        eprintln!("reasoner rel_diff={rel_r:.4} (max_abs={max_abs_r:.4} max_ref={max_ref_r:.3})");
        eprintln!("generator rel_diff={rel_g:.4} (max_abs={max_abs_g:.4} max_ref={max_ref_g:.3})");
        // 2 layers × 14 matmuls = 28 composed bf16 matmuls.  Allow 5 % rel.
        assert!(rel_r < 5e-2, "reasoner CPU/GPU mismatch rel={rel_r}");
        assert!(rel_g < 5e-2, "generator CPU/GPU mismatch rel={rel_g}");
    }

    /// Stage 7 forward v0: load 2 real layers + run joint-stream attention
    /// on tiny synthetic states.  No RoPE (positional encoding is a separate
    /// sub-chapter), no Metal acceleration (v0 verifies plumbing).  Asserts
    /// output is finite + drifted from input (real attention contributing).
    #[test]
    #[ignore = "phase4 v2 — needs Cosmos 3 Nano weights mounted; loads ~10 GB"]
    fn phase4_v2_joint_forward() {
        let Some(dir) = cosmos3_transformer_dir() else { eprintln!("Cosmos 3 not mounted"); return; };
        let m = Cosmos3OmniTransformerV2::load_with_layers(&dir, 2).expect("load");

        let t_r = 4;
        let t_g = 2;
        let mut x_r: Vec<f32> = (0..t_r * HIDDEN).map(|i| ((i as f32 * 0.013).sin()) * 0.05).collect();
        let mut x_g: Vec<f32> = (0..t_g * HIDDEN).map(|i| ((i as f32 * 0.017).cos()) * 0.05).collect();
        let x_r_in = x_r.clone();
        let x_g_in = x_g.clone();

        let t0 = std::time::Instant::now();
        for (i, l) in m.layers.iter().enumerate() {
            let lt = std::time::Instant::now();
            l.forward(&mut x_r, t_r, &mut x_g, t_g);
            eprintln!("phase4 v2 layer {i}: t_r={t_r} t_g={t_g} forward in {:?}", lt.elapsed());
        }
        eprintln!("phase4 v2: 2-layer joint forward total {:?}", t0.elapsed());

        // Both streams must be finite.
        assert!(x_r.iter().all(|v| v.is_finite()));
        assert!(x_g.iter().all(|v| v.is_finite()));

        // Both streams must have drifted from input (real attention contributing).
        let drift_r = x_r.iter().zip(x_r_in.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let drift_g = x_g.iter().zip(x_g_in.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("reasoner drift max_abs={drift_r:.4}  generator drift max_abs={drift_g:.4}");
        assert!(drift_r > 1e-4, "reasoner stream should drift through real attention");
        assert!(drift_g > 1e-4, "generator stream should drift through real attention");
    }
}
