//! MotionBERT 2D → 3D human pose lifting — rce's first attention-based
//! ONNX-sourced crate.
//!
//! # Architecture (Zhu et al., ICCV 2023)
//!
//! Input:  `[T=243 frames, J=17 joints, 2 coords]`  (2D pose sequence)
//! Output: `[T,   J,    3 coords]`                  (3D pose sequence)
//!
//! - `joints_embed`: per-joint Linear(2 → d=256).
//! - `pos_embed`: `[1, J=17, d]` learned spatial positional embedding.
//! - `temp_embed`: `[1, T=243, 1, d]` learned temporal positional embedding.
//! - **Dual-stream transformer** (depth=5 typically):
//!   - `blocks_st[i]`: SpaTemporal — `attn_s` over joints (J as seq, T as batch)
//!     then `attn_t` over time (T as seq, J as batch); MLPs after each.
//!   - `blocks_ts[i]`: TempoSpatial — same ops in the reversed order.
//!   - `ts_attn[i]`: Linear(2d → 2) + softmax — learnable per-token weighting
//!     `x_i = α[i,0] · st_out + α[i,1] · ts_out`.
//! - `pre_logits.fc`: Linear(d → 2d=512) + GELU.
//! - `head`: Linear(2d → 3).
//!
//! # Weight provenance
//!
//! Produced by `python3 joulesperbit/tools/convert_onnx.py
//!   model_library/motionbert/motionbert.onnx model.safetensors`.
//! The converter's MatMul-rename + layout transpose passes give PyTorch-style
//! `[Cout, Cin]` weight shapes that drop into `linear_f32` directly.
//! `pos_embed` and `temp_embed` are stored explicitly (not folded into Constants)
//! — this is the rare ViT-family ONNX that survives the converter cleanly.

use rce_op_conv2d::linear_f32;
use rce_weight_loader::{F32Slice, SafetensorsFile};
use rayon::prelude::*;
use std::path::Path;

// ── architecture constants ────────────────────────────────────────────
pub const D: usize = 256;
pub const N_HEADS: usize = 8;
pub const HD: usize = D / N_HEADS;          // 32
pub const D_FF: usize = 1024;
pub const N_JOINTS: usize = 17;
pub const N_FRAMES: usize = 243;
pub const DEPTH: usize = 5;
pub const EPS: f32 = 1e-6;

#[derive(Debug)]
pub enum MbError { Load(String) }
impl std::fmt::Display for MbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::Load(m) => write!(f, "motionbert load: {m}") }
    }
}
impl std::error::Error for MbError {}

fn load(sf: &SafetensorsFile, name: &str) -> Result<F32Slice, MbError> {
    let t = sf.get(name).ok_or_else(|| MbError::Load(format!("missing {name}")))?;
    F32Slice::from_tensor(name, t).map_err(MbError::Load)
}

// ── ops ───────────────────────────────────────────────────────────────

fn layer_norm(x: &mut [f32], d: usize, w: &[f32], b: &[f32]) {
    x.par_chunks_mut(d).for_each(|row| {
        let mean: f32 = row.iter().sum::<f32>() / d as f32;
        let var: f32 = row.iter().map(|v| { let z = v - mean; z * z }).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        for ((v, g), bb) in row.iter_mut().zip(w).zip(b) {
            *v = (*v - mean) * inv * *g + *bb;
        }
    });
}

#[inline] fn gelu(x: f32) -> f32 {
    const K: f32 = 0.7978845608028654;  // sqrt(2/π)
    0.5 * x * (1.0 + (K * (x + 0.044715 * x * x * x)).tanh())
}

#[inline] fn softmax_row(row: &mut [f32]) {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in row.iter_mut() { *v = (*v - m).exp(); sum += *v; }
    let inv = 1.0 / sum;
    for v in row.iter_mut() { *v *= inv; }
}

/// Multi-head bidirectional self-attention over `[N, D]` activation.
/// Combined-QKV variant: `qkv_w: [3D, D]` produces stacked Q,K,V.
fn mha_qkv(x: &[f32], n: usize,
           qkv_w: &[f32], qkv_b: &[f32],
           proj_w: &[f32], proj_b: &[f32]) -> Vec<f32> {
    let qkv = linear_f32(x, n, D, qkv_w, 3 * D, Some(qkv_b));
    // qkv layout: [N, 3*D] where slot 0 = Q, slot 1 = K, slot 2 = V (each [D]).
    let mut q = vec![0.0f32; n * D];
    let mut k = vec![0.0f32; n * D];
    let mut v = vec![0.0f32; n * D];
    for i in 0..n {
        let off = i * 3 * D;
        q[i*D..(i+1)*D].copy_from_slice(&qkv[off..off + D]);
        k[i*D..(i+1)*D].copy_from_slice(&qkv[off + D..off + 2 * D]);
        v[i*D..(i+1)*D].copy_from_slice(&qkv[off + 2 * D..off + 3 * D]);
    }
    let scale = 1.0 / (HD as f32).sqrt();
    let mut attn = vec![0.0f32; n * D];
    attn.par_chunks_mut(D).enumerate().for_each(|(i, arow)| {
        for h in 0..N_HEADS {
            let qh = &q[i*D + h*HD..i*D + h*HD + HD];
            let mut sc = vec![0.0f32; n];
            for j in 0..n {
                let kh = &k[j*D + h*HD..j*D + h*HD + HD];
                let mut s = 0.0f32;
                for d in 0..HD { s += qh[d] * kh[d]; }
                sc[j] = s * scale;
            }
            softmax_row(&mut sc);
            for d in 0..HD { let mut acc = 0.0f32;
                for j in 0..n { acc += sc[j] * v[j*D + h*HD + d]; }
                arow[h*HD + d] = acc;
            }
        }
    });
    linear_f32(&attn, n, D, proj_w, D, Some(proj_b))
}

/// MLP: Linear(D → D_FF) → GELU → Linear(D_FF → D).
fn mlp(x: &[f32], n: usize,
       fc1_w: &[f32], fc1_b: &[f32],
       fc2_w: &[f32], fc2_b: &[f32]) -> Vec<f32> {
    let mut h = linear_f32(x, n, D, fc1_w, D_FF, Some(fc1_b));
    for v in h.iter_mut() { *v = gelu(*v); }
    linear_f32(&h, n, D_FF, fc2_w, D, Some(fc2_b))
}

// ── weights ───────────────────────────────────────────────────────────

struct AttnBlock {
    n1_w: F32Slice, n1_b: F32Slice,
    qkv_w: F32Slice, qkv_b: F32Slice,
    proj_w: F32Slice, proj_b: F32Slice,
    n2_w: F32Slice, n2_b: F32Slice,
    mlp1_w: F32Slice, mlp1_b: F32Slice,
    mlp2_w: F32Slice, mlp2_b: F32Slice,
}

impl AttnBlock {
    fn load_half(sf: &SafetensorsFile, prefix: &str, suf: &str) -> Result<Self, MbError> {
        let p = format!("{prefix}.norm1_{suf}");
        let q = format!("{prefix}.attn_{suf}");
        let n = format!("{prefix}.norm2_{suf}");
        let m = format!("{prefix}.mlp_{suf}");
        Ok(Self {
            n1_w: load(sf, &format!("{p}.weight"))?,
            n1_b: load(sf, &format!("{p}.bias"))?,
            qkv_w: load(sf, &format!("{q}.qkv.weight"))?,
            qkv_b: load(sf, &format!("{q}.qkv.bias"))?,
            proj_w: load(sf, &format!("{q}.proj.weight"))?,
            proj_b: load(sf, &format!("{q}.proj.bias"))?,
            n2_w: load(sf, &format!("{n}.weight"))?,
            n2_b: load(sf, &format!("{n}.bias"))?,
            mlp1_w: load(sf, &format!("{m}.fc1.weight"))?,
            mlp1_b: load(sf, &format!("{m}.fc1.bias"))?,
            mlp2_w: load(sf, &format!("{m}.fc2.weight"))?,
            mlp2_b: load(sf, &format!("{m}.fc2.bias"))?,
        })
    }

    /// Apply pre-norm + MHA + residual + pre-norm + MLP + residual to a
    /// `[N, D]` activation (in place via output buffer add).
    fn forward(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut xn = x.to_vec();
        layer_norm(&mut xn, D, &self.n1_w, &self.n1_b);
        let a = mha_qkv(&xn, n, &self.qkv_w, &self.qkv_b, &self.proj_w, &self.proj_b);
        let mut y = x.to_vec();
        for i in 0..x.len() { y[i] += a[i]; }
        let mut yn = y.clone();
        layer_norm(&mut yn, D, &self.n2_w, &self.n2_b);
        let m = mlp(&yn, n, &self.mlp1_w, &self.mlp1_b, &self.mlp2_w, &self.mlp2_b);
        for i in 0..y.len() { y[i] += m[i]; }
        y
    }
}

/// One dual-stream block: spatial-then-temporal half + temporal-then-spatial half
/// + learnable cross-stream weighting (`ts_attn` Linear(2D → 2) + softmax).
struct DualBlock {
    st_s: AttnBlock, st_t: AttnBlock,    // SpaTemporal stream: attn_s then attn_t
    ts_t: AttnBlock, ts_s: AttnBlock,    // TempoSpatial stream: attn_t then attn_s
    cross_w: F32Slice, cross_b: F32Slice, // [2, 2D] [2]
}

impl DualBlock {
    fn load(sf: &SafetensorsFile, i: usize) -> Result<Self, MbError> {
        Ok(Self {
            st_s: AttnBlock::load_half(sf, &format!("blocks_st.{i}"), "s")?,
            st_t: AttnBlock::load_half(sf, &format!("blocks_st.{i}"), "t")?,
            ts_t: AttnBlock::load_half(sf, &format!("blocks_ts.{i}"), "t")?,
            ts_s: AttnBlock::load_half(sf, &format!("blocks_ts.{i}"), "s")?,
            cross_w: load(sf, &format!("ts_attn.{i}.weight"))?,
            cross_b: load(sf, &format!("ts_attn.{i}.bias"))?,
        })
    }

    /// `tokens: [T × J × D]` row-major-by-frame (`tokens[t * J * D + j * D + d]`).
    /// 1. ST stream: reshape to `[T batches × J as seq × D]`, apply `attn_s`;
    ///    transpose to `[J batches × T as seq × D]`, apply `attn_t`.
    /// 2. TS stream: same but ordered T-first then J.
    /// 3. Cross-stream: per-token softmax over `Linear(concat(st, ts), 2)`.
    fn forward(&self, tokens: &[f32]) -> Vec<f32> {
        // SpaTemporal: spatial first.
        let mut st = apply_spatial(&self.st_s, tokens);
        st = apply_temporal(&self.st_t, &st);

        // TempoSpatial: temporal first.
        let mut ts = apply_temporal(&self.ts_t, tokens);
        ts = apply_spatial(&self.ts_s, &ts);

        // Cross-stream weighting: per-token Linear(concat[st;ts], 2 outs) → softmax.
        // concat shape = [T*J, 2D]; cross_w shape = [2, 2D]; output [T*J, 2].
        let n_tok = N_FRAMES * N_JOINTS;
        let mut concat = vec![0.0f32; n_tok * 2 * D];
        for t in 0..n_tok {
            concat[t * 2 * D..t * 2 * D + D].copy_from_slice(&st[t * D..t * D + D]);
            concat[t * 2 * D + D..t * 2 * D + 2 * D].copy_from_slice(&ts[t * D..t * D + D]);
        }
        let mut weights = linear_f32(&concat, n_tok, 2 * D, &self.cross_w, 2, Some(&self.cross_b));
        weights.par_chunks_mut(2).for_each(|w| softmax_row(w));
        let mut out = vec![0.0f32; n_tok * D];
        out.par_chunks_mut(D).enumerate().for_each(|(t, orow)| {
            let alpha = &weights[t * 2..t * 2 + 2];
            for d in 0..D {
                orow[d] = alpha[0] * st[t * D + d] + alpha[1] * ts[t * D + d];
            }
        });
        out
    }
}

/// Apply per-frame spatial attention: each of T frames carries J=17 joints
/// as its sequence.  Tokens are `[T * J * D]`; we reshape into T independent
/// `[J × D]` sub-batches and run `attn` per frame.
fn apply_spatial(attn: &AttnBlock, tokens: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; tokens.len()];
    out.par_chunks_mut(N_JOINTS * D).enumerate().for_each(|(t, frame_out)| {
        let frame_in = &tokens[t * N_JOINTS * D..(t + 1) * N_JOINTS * D];
        let res = attn.forward(frame_in, N_JOINTS);
        frame_out.copy_from_slice(&res);
    });
    out
}

/// Apply per-joint temporal attention: each of J joints carries T=243 frames
/// as its sequence.  Tokens are `[T * J * D]`; we transpose to `[J * T * D]`,
/// run `attn` per joint, then transpose back.
fn apply_temporal(attn: &AttnBlock, tokens: &[f32]) -> Vec<f32> {
    // Transpose [T, J, D] → [J, T, D].
    let mut jtd = vec![0.0f32; tokens.len()];
    for t in 0..N_FRAMES { for j in 0..N_JOINTS {
        let src = &tokens[(t * N_JOINTS + j) * D..(t * N_JOINTS + j + 1) * D];
        jtd[(j * N_FRAMES + t) * D..(j * N_FRAMES + t + 1) * D].copy_from_slice(src);
    }}
    // Per-joint attn.
    let mut jtd_out = vec![0.0f32; tokens.len()];
    jtd_out.par_chunks_mut(N_FRAMES * D).enumerate().for_each(|(j, joint_out)| {
        let joint_in = &jtd[j * N_FRAMES * D..(j + 1) * N_FRAMES * D];
        let res = attn.forward(joint_in, N_FRAMES);
        joint_out.copy_from_slice(&res);
    });
    // Transpose back [J, T, D] → [T, J, D].
    let mut out = vec![0.0f32; tokens.len()];
    for j in 0..N_JOINTS { for t in 0..N_FRAMES {
        let src = &jtd_out[(j * N_FRAMES + t) * D..(j * N_FRAMES + t + 1) * D];
        out[(t * N_JOINTS + j) * D..(t * N_JOINTS + j + 1) * D].copy_from_slice(src);
    }}
    out
}

// ── full model ────────────────────────────────────────────────────────

pub struct MotionBERT {
    je_w: F32Slice, je_b: F32Slice,             // joints_embed: [D, 2] [D]
    pos_embed: F32Slice,                         // [J × D] (reshape of [1, J, D])
    temp_embed: F32Slice,                        // [T × D] (reshape of [1, T, 1, D])
    blocks: Vec<DualBlock>,
    pre_w: F32Slice, pre_b: F32Slice,            // pre_logits.fc: [2D, D] [2D]
    head_w: F32Slice, head_b: F32Slice,          // head: [3, 2D] [3]
}

impl MotionBERT {
    pub fn load(path: &Path) -> Result<Self, MbError> {
        let sf = SafetensorsFile::load(path).map_err(|e| MbError::Load(e.to_string()))?;
        let blocks = (0..DEPTH).map(|i| DualBlock::load(&sf, i)).collect::<Result<_, _>>()?;
        Ok(Self {
            je_w: load(&sf, "joints_embed.weight")?,
            je_b: load(&sf, "joints_embed.bias")?,
            pos_embed: load(&sf, "pos_embed")?,
            temp_embed: load(&sf, "temp_embed")?,
            blocks,
            pre_w: load(&sf, "pre_logits.fc.weight")?,
            pre_b: load(&sf, "pre_logits.fc.bias")?,
            head_w: load(&sf, "head.weight")?,
            head_b: load(&sf, "head.bias")?,
        })
    }

    /// Lift a 2D pose sequence `[T=243, J=17, 2]` → 3D `[T, J, 3]`.
    pub fn lift(&self, joints_2d: &[f32]) -> Vec<f32> {
        assert_eq!(joints_2d.len(), N_FRAMES * N_JOINTS * 2);

        // Joint embedding: Linear(2 → D) per token.
        // Reshape [T*J, 2] then linear → [T*J, D].
        let n_tok = N_FRAMES * N_JOINTS;
        let mut x = linear_f32(joints_2d, n_tok, 2, &self.je_w, D, Some(&self.je_b));

        // Add positional + temporal embeds.  pos_embed is per-joint (shared across T);
        // temp_embed is per-frame (shared across J).
        for t in 0..N_FRAMES { for j in 0..N_JOINTS { for d in 0..D {
            let off = (t * N_JOINTS + j) * D + d;
            x[off] += self.pos_embed[j * D + d] + self.temp_embed[t * D + d];
        }}}

        // Dual-stream transformer.
        for blk in &self.blocks {
            x = blk.forward(&x);
        }

        // pre_logits + GELU + head.
        let mut h = linear_f32(&x, n_tok, D, &self.pre_w, 2 * D, Some(&self.pre_b));
        for v in h.iter_mut() { *v = gelu(*v); }
        linear_f32(&h, n_tok, 2 * D, &self.head_w, 3, Some(&self.head_b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn weight_path() -> Option<PathBuf> {
        for p in [
            "/tmp/onnx_sweep/motionbert.safetensors",
            "/tmp/motionbert.safetensors",
        ] {
            let pb = PathBuf::from(p);
            if pb.exists() { return Some(pb); }
        }
        None
    }

    /// Build a synthetic safetensors file with all the tensors MotionBERT
    /// expects.  This validates the architecture wiring without needing the
    /// real ONNX-converted weights.  Small Gaussian random init for every
    /// tensor; the output won't be meaningful but it MUST be finite,
    /// shape-correct, and exercise every code path.
    fn write_synthetic_weights(path: &PathBuf) {
        use rce_weight_loader::write_safetensors;
        // Deterministic pseudo-random generator: linear congruential.
        let mut state: u64 = 0xC0FFEE_42;
        let mut rng = || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Map upper 24 bits to [-0.05, 0.05] — small init keeps activations bounded.
            let f = ((state >> 40) & 0xFFFFFF) as f32 / 16777216.0;
            (f - 0.5) * 0.1
        };
        let mut bytes = |shape: &[usize]| -> Vec<u8> {
            let n: usize = shape.iter().product();
            let mut out = Vec::with_capacity(n * 4);
            for _ in 0..n { out.extend_from_slice(&rng().to_le_bytes()); }
            out
        };

        // Collect (name, shape, bytes) so we can build references for write_safetensors.
        // We keep `bytes_storage` alive long enough to take slices.
        let mut entries: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        // joints_embed
        entries.push(("joints_embed.weight".into(), vec![D, 2], bytes(&[D, 2])));
        entries.push(("joints_embed.bias".into(),   vec![D],    bytes(&[D])));
        // positional embeds
        entries.push(("pos_embed".into(),  vec![1, N_JOINTS, D],   bytes(&[1, N_JOINTS, D])));
        entries.push(("temp_embed".into(), vec![1, N_FRAMES, 1, D], bytes(&[1, N_FRAMES, 1, D])));
        // 5 dual blocks: blocks_st.{i}.{half}, blocks_ts.{i}.{half}, ts_attn.{i}
        for i in 0..DEPTH {
            for stream in ["st", "ts"] {
                for half in ["s", "t"] {
                    let p = format!("blocks_{stream}.{i}");
                    entries.push((format!("{p}.norm1_{half}.weight"), vec![D], bytes(&[D])));
                    entries.push((format!("{p}.norm1_{half}.bias"),   vec![D], bytes(&[D])));
                    entries.push((format!("{p}.attn_{half}.qkv.weight"), vec![3*D, D], bytes(&[3*D, D])));
                    entries.push((format!("{p}.attn_{half}.qkv.bias"),   vec![3*D],    bytes(&[3*D])));
                    entries.push((format!("{p}.attn_{half}.proj.weight"), vec![D, D], bytes(&[D, D])));
                    entries.push((format!("{p}.attn_{half}.proj.bias"),   vec![D],    bytes(&[D])));
                    entries.push((format!("{p}.norm2_{half}.weight"), vec![D], bytes(&[D])));
                    entries.push((format!("{p}.norm2_{half}.bias"),   vec![D], bytes(&[D])));
                    entries.push((format!("{p}.mlp_{half}.fc1.weight"), vec![D_FF, D], bytes(&[D_FF, D])));
                    entries.push((format!("{p}.mlp_{half}.fc1.bias"),   vec![D_FF],    bytes(&[D_FF])));
                    entries.push((format!("{p}.mlp_{half}.fc2.weight"), vec![D, D_FF], bytes(&[D, D_FF])));
                    entries.push((format!("{p}.mlp_{half}.fc2.bias"),   vec![D],       bytes(&[D])));
                }
            }
            entries.push((format!("ts_attn.{i}.weight"), vec![2, 2*D], bytes(&[2, 2*D])));
            entries.push((format!("ts_attn.{i}.bias"),   vec![2],       bytes(&[2])));
        }
        // head + pre_logits
        entries.push(("pre_logits.fc.weight".into(), vec![2*D, D], bytes(&[2*D, D])));
        entries.push(("pre_logits.fc.bias".into(),   vec![2*D],    bytes(&[2*D])));
        entries.push(("head.weight".into(), vec![3, 2*D], bytes(&[3, 2*D])));
        entries.push(("head.bias".into(),   vec![3],      bytes(&[3])));
        // Initialize norm weights to 1.0 (instead of small random) for stability.
        for e in entries.iter_mut() {
            if e.0.ends_with(".norm1_s.weight") || e.0.ends_with(".norm1_t.weight")
                || e.0.ends_with(".norm2_s.weight") || e.0.ends_with(".norm2_t.weight")
            {
                for chunk in e.2.chunks_exact_mut(4) {
                    chunk.copy_from_slice(&1.0f32.to_le_bytes());
                }
            }
        }
        // Borrow refs (write_safetensors takes [(&str, &str, &[usize], &[u8])]).
        let refs: Vec<(&str, &str, &[usize], &[u8])> = entries.iter()
            .map(|(n, sh, b)| (n.as_str(), "f32", sh.as_slice(), b.as_slice()))
            .collect();
        let blob = write_safetensors(&refs, None);
        std::fs::write(path, blob).expect("write synthetic safetensors");
    }

    /// Synthetic-weights end-to-end: builds a fully-zeroed/random-init
    /// motionbert safetensors, loads it, runs `lift` on a synthetic 2D pose,
    /// asserts output shape + finiteness + non-trivial spread.  Validates
    /// the architecture wiring without needing the real ONNX-converted
    /// weights (external drive isn't always mounted).
    #[test]
    fn motionbert_synthetic_e2e() {
        let p = std::env::temp_dir().join("motionbert_synthetic.safetensors");
        write_synthetic_weights(&p);
        let m = MotionBERT::load(&p).expect("load synthetic");
        let mut joints_2d = vec![0.0f32; N_FRAMES * N_JOINTS * 2];
        for t in 0..N_FRAMES { for j in 0..N_JOINTS {
            let off = (t * N_JOINTS + j) * 2;
            joints_2d[off]     = (t as f32 * 0.07 + j as f32 * 0.3).sin() * 0.5;
            joints_2d[off + 1] = (t as f32 * 0.05 + j as f32 * 0.4).cos() * 0.5;
        }}
        let t0 = std::time::Instant::now();
        let out = m.lift(&joints_2d);
        eprintln!("motionbert synthetic lift: T={N_FRAMES} J={N_JOINTS} → 3D in {:?}", t0.elapsed());
        assert_eq!(out.len(), N_FRAMES * N_JOINTS * 3);
        let finite = out.iter().all(|v| v.is_finite());
        assert!(finite, "non-finite output");
        let mx = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mn = out.iter().cloned().fold(f32::INFINITY, f32::min);
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        eprintln!("3D output: range [{mn:.3}, {mx:.3}] mean {mean:.3}");
        // Random-init weights → output isn't physically meaningful but should
        // still be non-degenerate (some variance from the input structure).
        assert!(mx - mn > 1e-5, "completely flat output {}", mx - mn);
        let _ = std::fs::remove_file(&p);
    }

    /// Smoke test: when real (ONNX-converted) weights are present, load + run
    /// on a synthetic 2D pose sequence and assert shape + finite values.
    /// Skips silently when the converted safetensors aren't available
    /// (e.g. external drive unmounted).  Re-run with the drive mounted:
    ///   `python3 tools/convert_onnx.py model_library/motionbert/motionbert.onnx /tmp/motionbert.safetensors`
    #[test]
    fn motionbert_smoke() {
        let Some(p) = weight_path() else {
            eprintln!("motionbert: weights not found — run convert_onnx.py first");
            return;
        };
        let m = MotionBERT::load(&p).expect("load");
        // Synthetic 2D pose: smooth sinusoid per joint coord.
        let mut joints_2d = vec![0.0f32; N_FRAMES * N_JOINTS * 2];
        for t in 0..N_FRAMES { for j in 0..N_JOINTS {
            let off = (t * N_JOINTS + j) * 2;
            joints_2d[off]     = (t as f32 * 0.07 + j as f32 * 0.3).sin() * 0.5;
            joints_2d[off + 1] = (t as f32 * 0.05 + j as f32 * 0.4).cos() * 0.5;
        }}
        let t0 = std::time::Instant::now();
        let out = m.lift(&joints_2d);
        eprintln!("motionbert lift: T={N_FRAMES} J={N_JOINTS} → 3D in {:?}", t0.elapsed());
        assert_eq!(out.len(), N_FRAMES * N_JOINTS * 3);
        let finite = out.iter().all(|v| v.is_finite());
        assert!(finite, "non-finite output");
        let max_v = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_v = out.iter().cloned().fold(f32::INFINITY, f32::min);
        let spread = max_v - min_v;
        eprintln!("3D output range [{min_v:.3}, {max_v:.3}] spread {spread:.3}");
        assert!(spread > 0.01, "output too flat: {spread}");
    }
}
