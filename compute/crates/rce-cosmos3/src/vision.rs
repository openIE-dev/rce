//! Cosmos 3 Nano / Qwen3-VL vision encoder — Phase 2.A.
//!
//! Maps to HF `Qwen3VLVisionModel`. Takes pre-flattened patches (seq_len × 1536)
//! and produces (seq_len × 1152) ViT-output features.  The merger + deepstack
//! mergers (1152 → 4096) are Phase 2.B.

use rce_weight_loader::{LoadedTensor, SafetensorsFile};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;

use crate::BfSlice;

pub const HIDDEN: usize = 1152;
pub const FF: usize = 4304;
pub const N_HEADS: usize = 16;
pub const HEAD_DIM: usize = HIDDEN / N_HEADS;     // 72
pub const N_LAYERS: usize = 27;
pub const PATCH_SIZE: usize = 16;
pub const TEMPORAL_PATCH: usize = 2;
pub const IN_CH: usize = 3;
pub const PATCH_DIM: usize = IN_CH * TEMPORAL_PATCH * PATCH_SIZE * PATCH_SIZE;   // 1536
pub const NUM_POS: usize = 2304;
pub const NUM_GRID_PER_SIDE: usize = 48;          // sqrt(2304)
pub const SPATIAL_MERGE: usize = 2;
pub const DEEPSTACK_INDEXES: [usize; 3] = [8, 16, 24];
pub const ROPE_THETA: f32 = 10_000.0;
pub const LN_EPS: f32 = 1e-6;

#[derive(Debug)]
pub enum VisionError { Load(String) }
impl std::fmt::Display for VisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::Load(m) => write!(f, "cosmos3-vision load: {m}") }
    }
}
impl std::error::Error for VisionError {}

fn tf32(t: &LoadedTensor) -> Vec<f32> {
    let bytes = t.data.as_slice();
    match t.dtype.as_str() {
        "f32"  => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect(),
        "f16"  => bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0],c[1]]).to_f32()).collect(),
        "bf16" => bytes.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0],c[1]]).to_f32()).collect(),
        other  => panic!("vision: unsupported dtype {other}"),
    }
}
fn ld(sf: &SafetensorsFile, name: &str) -> Result<Vec<f32>, VisionError> {
    let t = sf.get(name).ok_or_else(|| VisionError::Load(format!("missing {name}")))?;
    Ok(tf32(t))
}

/// Borrow-direct BF16 loader.  See [`crate::BfSlice`] — the zero-copy variant
/// holds an `Arc<Mmap>` clone, so the ViT's ~1 GB of weight matrices live in
/// the kernel page cache and the user heap stores only struct pointers.
fn ld_bf16(sf: &SafetensorsFile, name: &str) -> Result<BfSlice, VisionError> {
    let t = sf.get(name).ok_or_else(|| VisionError::Load(format!("missing {name}")))?;
    BfSlice::from_tensor(name, t).map_err(|e| VisionError::Load(format!("{e}")))
}

/// BF16-weight Linear matching `linear()` shape contract, with bias add.
fn linear_bf16(x: &[f32], t: usize, din: usize,
               w: &[half::bf16], dout: usize, b: Option<&[f32]>) -> Vec<f32> {
    let mut y = crate::matmul_nt_bf16(x, t, din, w, dout);
    if let Some(bb) = b {
        for ti in 0..t { for o in 0..dout { y[ti*dout + o] += bb[o]; } }
    }
    y
}

// ── ops ─────────────────────────────────────────────────────────────────────

fn gelu_tanh(x: f32) -> f32 {
    // PyTorch `gelu_pytorch_tanh`: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    const K: f32 = 0.7978845608028654;   // sqrt(2/π)
    0.5 * x * (1.0 + (K * (x + 0.044715 * x * x * x)).tanh())
}

fn layernorm(x: &[f32], hidden: usize, t: usize, gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for ti in 0..t {
        let row = &x[ti*hidden..(ti+1)*hidden];
        let mean: f32 = row.iter().sum::<f32>() / hidden as f32;
        let var: f32 = row.iter().map(|v| { let d = v - mean; d*d }).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for h in 0..hidden {
            out[ti*hidden + h] = (row[h] - mean) * inv * gamma[h] + beta[h];
        }
    }
    out
}

/// y[t, o] = sum_i x[t, i] * w[o, i] + b[o]
fn linear(x: &[f32], t: usize, din: usize, w: &[f32], dout: usize, b: Option<&[f32]>) -> Vec<f32> {
    let mut y = vec![0.0f32; t * dout];
    y.par_chunks_mut(dout).enumerate().for_each(|(ti, out_row)| {
        let xrow = &x[ti*din..(ti+1)*din];
        for o in 0..dout {
            let wr = &w[o*din..(o+1)*din];
            let mut acc = 0.0f32;
            for i in 0..din { acc += xrow[i] * wr[i]; }
            out_row[o] = if let Some(bb) = b { acc + bb[o] } else { acc };
        }
    });
    y
}

// ── position machinery (matches HF Qwen3VLVisionModel exactly) ──────────────

/// `fast_pos_embed_interpolate`: bilinear interpolation of the 48×48 learned
/// `pos_embed` table to a tile of size `t × h × w`, with the trailing per-block
/// permutation that groups by 2×2 spatial blocks.
fn fast_pos_embed_interpolate(
    pos_embed: &[f32],
    t: usize, h: usize, w: usize,
) -> Vec<f32> {
    let merge = SPATIAL_MERGE;
    let n_side = NUM_GRID_PER_SIDE;
    // h_idxs = linspace(0, n_side-1, h)
    let lin = |n_out: usize| -> Vec<f32> {
        if n_out == 1 { return vec![0.0]; }
        let step = (n_side - 1) as f32 / (n_out - 1) as f32;
        (0..n_out).map(|i| i as f32 * step).collect()
    };
    let h_idxs = lin(h);
    let w_idxs = lin(w);

    let h_floor: Vec<usize> = h_idxs.iter().map(|v| v.floor() as usize).collect();
    let w_floor: Vec<usize> = w_idxs.iter().map(|v| v.floor() as usize).collect();
    let h_ceil:  Vec<usize> = h_idxs.iter().map(|v| ((v.floor() as usize) + 1).min(n_side - 1)).collect();
    let w_ceil:  Vec<usize> = w_idxs.iter().map(|v| ((v.floor() as usize) + 1).min(n_side - 1)).collect();
    let dh: Vec<f32> = h_idxs.iter().zip(&h_floor).map(|(v, &f)| v - f as f32).collect();
    let dw: Vec<f32> = w_idxs.iter().zip(&w_floor).map(|(v, &f)| v - f as f32).collect();

    // Per-token (idx, weight) accumulator: 4 corners × seq tokens.
    let mut patch_pe = vec![0.0f32; h * w * HIDDEN];   // pre-permute
    for hi in 0..h {
        for wi in 0..w {
            let token = hi * w + wi;
            let bases = [
                h_floor[hi] * n_side + w_floor[wi],
                h_floor[hi] * n_side + w_ceil[wi],
                h_ceil[hi]  * n_side + w_floor[wi],
                h_ceil[hi]  * n_side + w_ceil[wi],
            ];
            let weights = [
                (1.0 - dh[hi]) * (1.0 - dw[wi]),
                (1.0 - dh[hi]) * dw[wi],
                dh[hi]         * (1.0 - dw[wi]),
                dh[hi]         * dw[wi],
            ];
            for c in 0..4 {
                let src = &pos_embed[bases[c] * HIDDEN..(bases[c] + 1) * HIDDEN];
                let weight = weights[c];
                let dst = &mut patch_pe[token * HIDDEN..(token + 1) * HIDDEN];
                for k in 0..HIDDEN { dst[k] += weight * src[k]; }
            }
        }
    }

    // Repeat for t temporal frames, then permute (t, h/m, m, w/m, m, hidden) →
    // (t, h/m, w/m, m, m, hidden), flatten to (t*h*w, hidden).
    let h_merged = h / merge;
    let w_merged = w / merge;
    let mut out = vec![0.0f32; t * h * w * HIDDEN];
    for ti in 0..t {
        for hb in 0..h_merged {
            for wb in 0..w_merged {
                for mh in 0..merge {
                    for mw in 0..merge {
                        let src_token = (hb * merge + mh) * w + (wb * merge + mw);
                        let dst_token = ((((ti * h_merged + hb) * w_merged + wb) * merge + mh) * merge) + mw;
                        let src = &patch_pe[src_token * HIDDEN..(src_token + 1) * HIDDEN];
                        let dst = &mut out[dst_token * HIDDEN..(dst_token + 1) * HIDDEN];
                        dst.copy_from_slice(src);
                    }
                }
            }
        }
    }
    out
}

/// `rot_pos_emb`: build per-token (h_pos, w_pos) coordinates and look up rotary
/// frequencies (shape `head_dim/2 = 36`).  Repeated for `num_frames`.
fn rot_pos_emb(t: usize, h: usize, w: usize) -> Vec<f32> {
    // Per HF: dim = head_dim // 2 = 36; inv_freq has len dim/2 = 18.
    let dim = HEAD_DIM / 2;       // 36
    let half_dim = dim / 2;       // 18
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / ROPE_THETA.powf((2 * i) as f32 / dim as f32))
        .collect();
    let max_hw = h.max(w);
    // freq_table[p, k] = p * inv_freq[k]
    let mut freq_table = vec![0.0f32; max_hw * half_dim];
    for p in 0..max_hw {
        for k in 0..half_dim {
            freq_table[p * half_dim + k] = p as f32 * inv_freq[k];
        }
    }
    // Build coords like HF: groups by (hblock, wblock, mh, mw).
    let merge = SPATIAL_MERGE;
    let h_blocks = h / merge;
    let w_blocks = w / merge;
    let frame_tokens = h * w;
    let mut coords: Vec<(usize, usize)> = Vec::with_capacity(frame_tokens);
    for hb in 0..h_blocks {
        for wb in 0..w_blocks {
            for mh in 0..merge {
                for mw in 0..merge {
                    coords.push((hb * merge + mh, wb * merge + mw));
                }
            }
        }
    }
    // Per token: lookup row freq_table[h] and col freq_table[w], stack → flatten.
    let per_token_dim = 2 * half_dim;   // 36 = full rotary dim per token
    let mut per_frame = vec![0.0f32; frame_tokens * per_token_dim];
    for (i, (rp, cp)) in coords.iter().enumerate() {
        let dst = &mut per_frame[i * per_token_dim..(i + 1) * per_token_dim];
        let row = &freq_table[rp * half_dim..(rp + 1) * half_dim];
        let col = &freq_table[cp * half_dim..(cp + 1) * half_dim];
        dst[..half_dim].copy_from_slice(row);
        dst[half_dim..].copy_from_slice(col);
    }
    // Repeat for t frames.
    let mut out = vec![0.0f32; t * frame_tokens * per_token_dim];
    for ti in 0..t {
        out[ti * frame_tokens * per_token_dim..(ti + 1) * frame_tokens * per_token_dim]
            .copy_from_slice(&per_frame);
    }
    out
}

/// Build the cos/sin tables for vision attention.  Input is `rotary_pos_emb`
/// of shape `(seq, dim=36)`.  Output is `(seq, head_dim=72)` for each of cos,sin.
fn vision_cos_sin(rotary_pos_emb: &[f32], seq: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    // emb = cat(rope, rope, -1)  → shape (seq, 2*dim) = (seq, 72)
    let full = 2 * dim;
    let mut cos = vec![0.0f32; seq * full];
    let mut sin = vec![0.0f32; seq * full];
    for s in 0..seq {
        for d in 0..dim {
            let theta = rotary_pos_emb[s * dim + d];
            let (sn, cs) = theta.sin_cos();
            cos[s * full + d]       = cs;
            cos[s * full + d + dim] = cs;
            sin[s * full + d]       = sn;
            sin[s * full + d + dim] = sn;
        }
    }
    (cos, sin)
}

/// Apply RoPE (vision-style: rotate_half on full head_dim) in-place.
/// `x` layout: `[seq, n_heads * head_dim]`.  cos/sin shape `[seq, head_dim]`.
fn apply_rope_vision(x: &mut [f32], seq: usize, n_heads: usize, cos: &[f32], sin: &[f32]) {
    let half = HEAD_DIM / 2;
    for s in 0..seq {
        for h in 0..n_heads {
            let base = s * n_heads * HEAD_DIM + h * HEAD_DIM;
            for d in 0..HEAD_DIM {
                let _ = (base, d);  // silence unused before the rotate_half pass below
            }
            // rotate_half: r(x) = cat((-x[half:], x[:half]))
            // y = x * cos + r(x) * sin
            let mut tmp = vec![0.0f32; HEAD_DIM];
            tmp.copy_from_slice(&x[base..base + HEAD_DIM]);
            for d in 0..HEAD_DIM {
                let rotated = if d < half { -tmp[d + half] } else { tmp[d - half] };
                x[base + d] = tmp[d] * cos[s * HEAD_DIM + d] + rotated * sin[s * HEAD_DIM + d];
            }
        }
    }
}

// ── ViT block ───────────────────────────────────────────────────────────────

struct Block {
    norm1_w: Vec<f32>, norm1_b: Vec<f32>,
    // Big projection matrices kept as BF16 to halve DRAM bandwidth (NEON path lives in lib.rs).
    qkv_w:  BfSlice, qkv_b: Vec<f32>,           // [3456, 1152] / [3456]
    proj_w: BfSlice, proj_b: Vec<f32>,          // [1152, 1152] / [1152]
    norm2_w: Vec<f32>, norm2_b: Vec<f32>,
    mlp1_w: BfSlice, mlp1_b: Vec<f32>,          // [4304, 1152] / [4304]
    mlp2_w: BfSlice, mlp2_b: Vec<f32>,          // [1152, 4304] / [1152]
}
impl Block {
    fn load(sf: &SafetensorsFile, i: usize) -> Result<Self, VisionError> {
        let p = format!("blocks.{i}");
        Ok(Self {
            norm1_w: ld(sf,      &format!("{p}.norm1.weight"))?,
            norm1_b: ld(sf,      &format!("{p}.norm1.bias"))?,
            qkv_w:   ld_bf16(sf, &format!("{p}.attn.qkv.weight"))?,
            qkv_b:   ld(sf,      &format!("{p}.attn.qkv.bias"))?,
            proj_w:  ld_bf16(sf, &format!("{p}.attn.proj.weight"))?,
            proj_b:  ld(sf,      &format!("{p}.attn.proj.bias"))?,
            norm2_w: ld(sf,      &format!("{p}.norm2.weight"))?,
            norm2_b: ld(sf,      &format!("{p}.norm2.bias"))?,
            mlp1_w:  ld_bf16(sf, &format!("{p}.mlp.linear_fc1.weight"))?,
            mlp1_b:  ld(sf,      &format!("{p}.mlp.linear_fc1.bias"))?,
            mlp2_w:  ld_bf16(sf, &format!("{p}.mlp.linear_fc2.weight"))?,
            mlp2_b:  ld(sf,      &format!("{p}.mlp.linear_fc2.bias"))?,
        })
    }
    fn forward(&self, x: &mut Vec<f32>, t: usize, cos: &[f32], sin: &[f32]) {
        // Pre-attn LN
        let xn = layernorm(x, HIDDEN, t, &self.norm1_w, &self.norm1_b, LN_EPS);
        // Fused QKV: [t, 3*HIDDEN]
        let qkv = linear_bf16(&xn, t, HIDDEN, &self.qkv_w, 3 * HIDDEN, Some(&self.qkv_b));
        // HF: qkv.reshape(seq, 3, n_heads, head_dim).permute(1, 0, 2, 3) → split → q/k/v each shape (seq, n_heads, head_dim).
        // The reshape over qkv's TRAILING dim 3*HIDDEN gives [seq, 3, n_heads, head_dim] (NOT [seq, n_heads, 3, head_dim])
        // — so for each seq position the layout is [Q_h0_d0..Q_h0_d71, Q_h1_d0..., ..., K_h0_d0..., V...].
        let stride = 3 * HIDDEN;
        let mut q = vec![0.0f32; t * HIDDEN];
        let mut k = vec![0.0f32; t * HIDDEN];
        let mut v = vec![0.0f32; t * HIDDEN];
        for s in 0..t {
            let src = &qkv[s * stride..(s + 1) * stride];
            q[s * HIDDEN..(s + 1) * HIDDEN].copy_from_slice(&src[0..HIDDEN]);
            k[s * HIDDEN..(s + 1) * HIDDEN].copy_from_slice(&src[HIDDEN..2 * HIDDEN]);
            v[s * HIDDEN..(s + 1) * HIDDEN].copy_from_slice(&src[2 * HIDDEN..3 * HIDDEN]);
        }
        // Apply RoPE on Q and K.
        apply_rope_vision(&mut q, t, N_HEADS, cos, sin);
        apply_rope_vision(&mut k, t, N_HEADS, cos, sin);

        // Multi-head attention (single segment in Phase 2.A — no cu_seqlens packing).
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut attn = vec![0.0f32; t * HIDDEN];
        let head_buf_size = t * HEAD_DIM;
        let mut per_head: Vec<f32> = vec![0.0f32; N_HEADS * head_buf_size];
        per_head.par_chunks_mut(head_buf_size).enumerate().for_each(|(h, buf)| {
            let h_off = h * HEAD_DIM;
            for ti in 0..t {
                let mut scores = vec![0.0f32; t];
                let mut maxv = f32::NEG_INFINITY;
                for tj in 0..t {
                    let mut s = 0.0f32;
                    for d in 0..HEAD_DIM {
                        s += q[ti * HIDDEN + h_off + d] * k[tj * HIDDEN + h_off + d];
                    }
                    s *= scale;
                    scores[tj] = s;
                    if s > maxv { maxv = s; }
                }
                let mut sum = 0.0f32;
                for tj in 0..t { scores[tj] = (scores[tj] - maxv).exp(); sum += scores[tj]; }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for tj in 0..t { scores[tj] *= inv; }
                for d in 0..HEAD_DIM {
                    let mut acc = 0.0f32;
                    for tj in 0..t {
                        acc += scores[tj] * v[tj * HIDDEN + h_off + d];
                    }
                    buf[ti * HEAD_DIM + d] = acc;
                }
            }
        });
        for h in 0..N_HEADS {
            for ti in 0..t {
                for d in 0..HEAD_DIM {
                    attn[ti * HIDDEN + h * HEAD_DIM + d] = per_head[h * head_buf_size + ti * HEAD_DIM + d];
                }
            }
        }
        // Output projection + residual.
        let o = linear_bf16(&attn, t, HIDDEN, &self.proj_w, HIDDEN, Some(&self.proj_b));
        for i in 0..x.len() { x[i] += o[i]; }

        // Pre-MLP LN
        let xn = layernorm(x, HIDDEN, t, &self.norm2_w, &self.norm2_b, LN_EPS);
        // MLP: fc1 → GELU(tanh) → fc2 + residual
        let mut h1 = linear_bf16(&xn, t, HIDDEN, &self.mlp1_w, FF, Some(&self.mlp1_b));
        for v in h1.iter_mut() { *v = gelu_tanh(*v); }
        let h2 = linear_bf16(&h1, t, FF, &self.mlp2_w, HIDDEN, Some(&self.mlp2_b));
        for i in 0..x.len() { x[i] += h2[i]; }
    }
}

// ── full model ──────────────────────────────────────────────────────────────

pub const OUT_HIDDEN: usize = 4096;
pub const MERGED: usize = HIDDEN * SPATIAL_MERGE * SPATIAL_MERGE;   // 4608

/// PatchMerger: 2×2 spatial reshape → LN(eps=1e-6) → fc1 (4608→4608) → GELU → fc2 (4608→4096).
/// Two variants per HF: `use_postshuffle_norm` controls whether LN runs on 1152 (pre-shuffle)
/// or 4608 (post-shuffle). The main `merger` uses pre-shuffle norm; the three
/// `deepstack_merger_list` mergers use post-shuffle norm.
pub struct PatchMerger {
    norm_w: Vec<f32>, norm_b: Vec<f32>,
    fc1_w: BfSlice, fc1_b: Vec<f32>,    // [4608, 4608]
    fc2_w: BfSlice, fc2_b: Vec<f32>,    // [4096, 4608]
    use_postshuffle_norm: bool,
}
impl PatchMerger {
    fn load(sf: &SafetensorsFile, prefix: &str, use_postshuffle_norm: bool) -> Result<Self, VisionError> {
        Ok(Self {
            norm_w: ld(sf,      &format!("{prefix}.norm.weight"))?,
            norm_b: ld(sf,      &format!("{prefix}.norm.bias"))?,
            fc1_w:  ld_bf16(sf, &format!("{prefix}.linear_fc1.weight"))?,
            fc1_b:  ld(sf,      &format!("{prefix}.linear_fc1.bias"))?,
            fc2_w:  ld_bf16(sf, &format!("{prefix}.linear_fc2.weight"))?,
            fc2_b:  ld(sf,      &format!("{prefix}.linear_fc2.bias"))?,
            use_postshuffle_norm,
        })
    }
    /// Input: `[seq_len × HIDDEN]` where `seq_len` is divisible by `SPATIAL_MERGE²`.
    /// Output: `[seq_len/4 × OUT_HIDDEN]`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len() % MERGED, 0);
        let n_out = x.len() / MERGED;
        // Step 1: norm.
        let normed: Vec<f32> = if self.use_postshuffle_norm {
            // post-shuffle: view(-1, 4608) THEN LayerNorm over the 4608 dim.
            layernorm(x, MERGED, n_out, &self.norm_w, &self.norm_b, LN_EPS)
        } else {
            // pre-shuffle: LayerNorm over 1152, then view(-1, 4608).
            // Resulting shape after view stays the same memory but tokens regroup.
            // Implementation: just apply LN per HIDDEN row, the view is implicit.
            let seq = x.len() / HIDDEN;
            layernorm(x, HIDDEN, seq, &self.norm_w, &self.norm_b, LN_EPS)
        };
        // Step 2: fc1 → GELU → fc2.
        let mut h = linear_bf16(&normed, n_out, MERGED, &self.fc1_w, MERGED, Some(&self.fc1_b));
        for v in h.iter_mut() { *v = 0.5 * *v * (1.0 + (*v / std::f32::consts::SQRT_2).erf_approx()); }
        linear_bf16(&h, n_out, MERGED, &self.fc2_w, OUT_HIDDEN, Some(&self.fc2_b))
    }
}

// HF `nn.GELU()` (no approx flag) uses the exact erf form, not gelu_pytorch_tanh.
// We only need a single erf for the PatchMerger.  Use Abramowitz–Stegun 7.1.26.
trait ErfApprox { fn erf_approx(self) -> f32; }
impl ErfApprox for f32 {
    fn erf_approx(self) -> f32 {
        let t = 1.0 / (1.0 + 0.3275911 * self.abs());
        let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-self * self).exp();
        if self < 0.0 { -y } else { y }
    }
}

pub struct VisionEncoder {
    patch_embed_w: BfSlice,   // [1152, 1536]  (flattened Conv3d weight)
    patch_embed_b: Vec<f32>,          // [1152]
    pos_embed: Vec<f32>,              // [2304, 1152]  (kept f32 — small + needed for bilinear interp)
    blocks: Vec<Block>,
    pub merger: PatchMerger,
    pub deepstack_mergers: Vec<PatchMerger>,
}

impl VisionEncoder {
    pub fn load(path: &Path) -> Result<Arc<Self>, VisionError> {
        let sf = SafetensorsFile::load(path).map_err(|e| VisionError::Load(e.to_string()))?;

        // Conv3d weight [1152, 3, 2, 16, 16] = [1152, 1536] when flattened along trailing dims.
        let patch_embed_w = ld_bf16(&sf, "patch_embed.proj.weight")?;
        if patch_embed_w.len() != HIDDEN * PATCH_DIM {
            return Err(VisionError::Load(format!(
                "patch_embed.proj.weight: {} elements, expected {}", patch_embed_w.len(), HIDDEN * PATCH_DIM)));
        }
        let patch_embed_b = ld(&sf, "patch_embed.proj.bias")?;
        let pos_embed     = ld(&sf, "pos_embed.weight")?;
        if pos_embed.len() != NUM_POS * HIDDEN {
            return Err(VisionError::Load(format!("pos_embed.weight: {} elements", pos_embed.len())));
        }

        let mut blocks = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            blocks.push(Block::load(&sf, i)?);
        }
        let merger = PatchMerger::load(&sf, "merger", false)?;
        let mut deepstack_mergers = Vec::with_capacity(DEEPSTACK_INDEXES.len());
        for j in 0..DEEPSTACK_INDEXES.len() {
            deepstack_mergers.push(PatchMerger::load(&sf, &format!("deepstack_merger_list.{j}"), true)?);
        }
        Ok(Arc::new(Self { patch_embed_w, patch_embed_b, pos_embed, blocks, merger, deepstack_mergers }))
    }

    /// Forward.  `patches` is a flat `[seq_len × PATCH_DIM]` slice already in the
    /// HF "permute(C, T, H, W).flatten()" layout.  Returns
    /// (last_hidden_state, deepstack_features_at_layers_[8,16,24]) — the latter
    /// is the raw post-block-N output, *not* the merger output (Phase 2.B).
    pub fn forward(&self, patches: &[f32], grid_thw: (usize, usize, usize))
        -> (Vec<f32>, Vec<Vec<f32>>)
    {
        let (t, h, w) = grid_thw;
        let seq = t * h * w;
        assert_eq!(patches.len(), seq * PATCH_DIM,
                   "patches.len()={} expected {seq}*{PATCH_DIM}", patches.len());

        // patch_embed = Linear(PATCH_DIM, HIDDEN)
        let mut hidden = linear_bf16(patches, seq, PATCH_DIM, &self.patch_embed_w, HIDDEN, Some(&self.patch_embed_b));

        // Bilinear-interp pos_embed and add.
        let pos = fast_pos_embed_interpolate(&self.pos_embed, t, h, w);
        for i in 0..hidden.len() { hidden[i] += pos[i]; }

        // Rotary positions: shape (seq, 36).  Build cos/sin (seq, 72).
        let rpe = rot_pos_emb(t, h, w);
        let rope_dim = HEAD_DIM / 2;     // 36
        let (cos, sin) = vision_cos_sin(&rpe, seq, rope_dim);

        // Run 27 blocks; collect deepstack outputs (post-merger to OUT_HIDDEN).
        let mut deepstack: Vec<Vec<f32>> = Vec::with_capacity(DEEPSTACK_INDEXES.len());
        for (li, blk) in self.blocks.iter().enumerate() {
            blk.forward(&mut hidden, seq, &cos, &sin);
            if let Some(j) = DEEPSTACK_INDEXES.iter().position(|&x| x == li) {
                deepstack.push(self.deepstack_mergers[j].forward(&hidden));
            }
        }
        (hidden, deepstack)
    }

    /// Apply the main merger (spatial-2×2 → `OUT_HIDDEN`).
    pub fn merge(&self, last_hidden_state: &[f32]) -> Vec<f32> {
        self.merger.forward(last_hidden_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Instant;

    fn model_path() -> Option<PathBuf> {
        let p = PathBuf::from(
            "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/Cosmos3-Nano/vision_encoder/model.safetensors");
        if p.exists() { Some(p) } else { None }
    }

    fn read_f32(name: &str) -> Option<Vec<f32>> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
        let bytes = std::fs::read(&p).ok()?;
        Some(bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]]))
            .collect())
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x,y)| (x-y).abs()).fold(0.0f32, f32::max)
    }
    fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len() as f32;
        a.iter().zip(b).map(|(x,y)| (x-y).abs()).sum::<f32>() / n
    }

    /// Bit-close match against the PyTorch reference dumped to tests/fixtures.
    #[test]
    #[ignore = "loads 1.1 GB vision encoder; run with --ignored"]
    fn cosmos3_vision_against_pytorch() {
        let Some(p) = model_path() else { eprintln!("skip"); return; };
        let Some(patches) = read_f32("vision_input_patches_4x1536.f32") else { eprintln!("skip: fixture missing"); return; };
        let Some(golden_last) = read_f32("vision_golden_last_4x1152.f32") else { eprintln!("skip: golden missing"); return; };

        // Optional intermediates for diagnostics.
        let golden_after_patch = read_f32("vision_after_patch_embed.f32");
        let golden_pos = read_f32("vision_pos_embeds.f32");
        let golden_after_pos = read_f32("vision_after_pos_embed.f32");
        let golden_after_blk0 = read_f32("vision_after_block_0.f32");

        eprintln!("loading vision encoder from {}…", p.display());
        let t0 = Instant::now();
        let m = VisionEncoder::load(&p).expect("load");
        eprintln!("loaded in {:?}", t0.elapsed());

        // Phase-1: patch_embed
        let pe = linear_bf16(&patches, 4, PATCH_DIM, &m.patch_embed_w, HIDDEN, Some(&m.patch_embed_b));
        if let Some(g) = golden_after_patch.as_ref() {
            eprintln!("patch_embed max_abs_diff = {:.2e}, mean = {:.2e}",
                      max_abs_diff(&pe, g), mean_abs_diff(&pe, g));
            assert!(max_abs_diff(&pe, g) < 5e-3, "patch_embed drift too large");
        }

        // Phase-1: bilinear pos_embed
        let pos = fast_pos_embed_interpolate(&m.pos_embed, 1, 2, 2);
        if let Some(g) = golden_pos.as_ref() {
            eprintln!("pos_embed   max_abs_diff = {:.2e}, mean = {:.2e}",
                      max_abs_diff(&pos, g), mean_abs_diff(&pos, g));
            assert!(max_abs_diff(&pos, g) < 5e-3, "pos_embed drift too large");
        }

        // Phase-1: sum
        let mut hidden = pe.clone();
        for i in 0..hidden.len() { hidden[i] += pos[i]; }
        if let Some(g) = golden_after_pos.as_ref() {
            eprintln!("after_pos   max_abs_diff = {:.2e}", max_abs_diff(&hidden, g));
            assert!(max_abs_diff(&hidden, g) < 5e-3);
        }

        // Block 0
        let rpe = rot_pos_emb(1, 2, 2);
        let (cos, sin) = vision_cos_sin(&rpe, 4, HEAD_DIM / 2);
        let mut hidden_blk = hidden.clone();
        m.blocks[0].forward(&mut hidden_blk, 4, &cos, &sin);
        if let Some(g) = golden_after_blk0.as_ref() {
            let mad = max_abs_diff(&hidden_blk, g);
            let mead = mean_abs_diff(&hidden_blk, g);
            eprintln!("block_0     max_abs_diff = {:.4e}, mean = {:.4e}", mad, mead);
            assert!(mad < 5e-3, "block 0 drift: {mad}");
        }

        // Full forward
        let t0 = Instant::now();
        let (last, ds) = m.forward(&patches, (1, 2, 2));
        eprintln!("forward: {:?}", t0.elapsed());
        let mad = max_abs_diff(&last, &golden_last);
        let mead = mean_abs_diff(&last, &golden_last);
        eprintln!("last_hidden_state   max_abs_diff = {:.4e}, mean = {:.4e}", mad, mead);
        assert!(mad < 0.5, "last_hidden_state drift too large: {mad}");
        assert!(mead < 0.05, "last_hidden_state mean drift too large: {mead}");

        // Phase 2.B: deepstack mergers at layers [8, 16, 24].
        assert_eq!(ds.len(), DEEPSTACK_INDEXES.len());
        for (j, _layer_idx) in DEEPSTACK_INDEXES.iter().enumerate() {
            let g = read_f32(&format!("vision_golden_deepstack_{j}_1x4096.f32"))
                .expect("missing deepstack golden");
            let mad = max_abs_diff(&ds[j], &g);
            let mead = mean_abs_diff(&ds[j], &g);
            eprintln!("deepstack[{j}]      max_abs_diff = {:.4e}, mean = {:.4e}", mad, mead);
            assert!(mad < 0.5, "deepstack[{j}] drift too large: {mad}");
            assert!(mead < 0.05);
        }

        // Phase 2.B: main merger.
        let merged = m.merge(&last);
        eprintln!("merger output: {} f32 (= {} tokens × {OUT_HIDDEN})",
                  merged.len(), merged.len() / OUT_HIDDEN);
        assert_eq!(merged.len(), OUT_HIDDEN, "merger should produce 1 token of {OUT_HIDDEN}");
    }
}
