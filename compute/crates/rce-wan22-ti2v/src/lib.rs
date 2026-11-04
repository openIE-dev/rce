//! rce wiring for Wan-AI/Wan2.2-TI2V-5B.
//!
//! Architecture (from `config.json`):
//! - WanModel DiT, 30 blocks × `dim=3072`, `ffn_dim=14336`, `num_heads=24`.
//! - 3D patch embedder `[3072, 48, 1, 2, 2]` — VAE z (in_dim=48) → transformer
//!   dim, 1×2×2 spatial patches (T unchanged, H/W halved).  Direct successor
//!   to `rce-cosmos3::phase4_v2::Wan22VAE_v2::patchify_2x2` (this model
//!   reads VAE latents the VAE encoder produces).
//! - Self-attention (with qk_norm) + cross-attention to T5-XXL text embeddings
//!   (4096-d → 3072-d via the `text_embedding` 2-layer MLP).
//! - AdaLN-Zero style modulation: per-block learned `modulation [1, 6, 3072]`
//!   added to time-conditioned (scale, shift, gate) × (attn, ffn) parameters.
//! - FFN: 2-layer GELU MLP (`ffn.0` → GELU → `ffn.2`).
//! - Output head: per-block-shared norm + `head [192, 3072]` → 48 × 2² = 192
//!   (un-patchify back to VAE latent space).
//!
//! All weights ship as F32 in safetensors (3 shards, ~18.6 GB).  Borrow-direct
//! via `F32Slice` so the user heap holds only struct pointers.

use std::collections::HashMap;
use std::path::Path;
use rce_weight_loader::{F32Slice, LoadedTensor, SafetensorsFile};

pub const DIM: usize = 3072;
pub const FFN_DIM: usize = 14336;
pub const FREQ_DIM: usize = 256;
pub const IN_DIM: usize = 48;
pub const OUT_DIM: usize = 48;
pub const N_HEADS: usize = 24;
pub const HEAD_DIM: usize = DIM / N_HEADS; // 128
pub const N_LAYERS: usize = 30;
pub const TEXT_LEN: usize = 512;
pub const T5_DIM: usize = 4096;
pub const PATCH_T: usize = 1;
pub const PATCH_H: usize = 2;
pub const PATCH_W: usize = 2;
pub const PATCH_OUT_DIM: usize = OUT_DIM * PATCH_H * PATCH_W; // 192

#[derive(Debug)]
pub enum Wan22Error {
    Load(String),
}

impl std::fmt::Display for Wan22Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::Load(s) => write!(f, "wan22-ti2v: {s}") }
    }
}

impl std::error::Error for Wan22Error {}

// ── multi-shard loader (mirror of rce-cosmos3::ShardedSafetensors) ─────
pub struct ShardedSafetensors {
    shards: Vec<SafetensorsFile>,
    index: HashMap<String, usize>,
}

impl ShardedSafetensors {
    pub fn from_index(json_path: &Path) -> Result<Self, Wan22Error> {
        let raw = std::fs::read_to_string(json_path)
            .map_err(|e| Wan22Error::Load(format!("read index: {e}")))?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| Wan22Error::Load(format!("parse index: {e}")))?;
        let wm = v.get("weight_map").cloned().unwrap_or(v);
        let wm = wm.as_object().ok_or_else(|| Wan22Error::Load("weight_map not object".into()))?;
        let parent = json_path.parent().unwrap_or(Path::new(""));
        let mut path_to_idx: HashMap<String, usize> = HashMap::new();
        let mut shards: Vec<SafetensorsFile> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::with_capacity(wm.len());
        for (name, file_v) in wm {
            let file = file_v.as_str()
                .ok_or_else(|| Wan22Error::Load(format!("entry {name} not string")))?;
            let idx = if let Some(&i) = path_to_idx.get(file) { i } else {
                let abs = parent.join(file);
                eprintln!("[wan22-ti2v] loading shard {}", abs.display());
                let sf = SafetensorsFile::load(&abs)
                    .map_err(|e| Wan22Error::Load(format!("load {file}: {e}")))?;
                let i = shards.len();
                shards.push(sf);
                path_to_idx.insert(file.to_string(), i);
                i
            };
            index.insert(name.clone(), idx);
        }
        Ok(Self { shards, index })
    }
    pub fn get(&self, name: &str) -> Option<&LoadedTensor> {
        let i = *self.index.get(name)?;
        self.shards[i].get(name)
    }
}

// ── tensor loaders ───────────────────────────────────────────────────────
fn lf(ss: &ShardedSafetensors, name: &str) -> Result<F32Slice, Wan22Error> {
    let t = ss.get(name)
        .ok_or_else(|| Wan22Error::Load(format!("missing {name}")))?;
    F32Slice::from_tensor(name, t).map_err(Wan22Error::Load)
}

fn lf_vec(ss: &ShardedSafetensors, name: &str) -> Result<Vec<f32>, Wan22Error> {
    let s = lf(ss, name)?;
    Ok(s.as_slice().to_vec())
}

// ── weight structs ───────────────────────────────────────────────────────

/// One attention block's projection + qk-norm weights.  Shared shape for
/// both self-attn and cross-attn (Wan 2.2 uses uniform `dim` for all
/// QKVO; cross-attn K/V also come from a `dim`-wide projection of the
/// already-projected `[T_text, 3072]` text embedding, not directly from
/// T5's 4096-d output — text MLP handles the dimension change).
pub struct AttnWeights {
    pub q_w: F32Slice, pub q_b: Vec<f32>,
    pub k_w: F32Slice, pub k_b: Vec<f32>,
    pub v_w: F32Slice, pub v_b: Vec<f32>,
    pub o_w: F32Slice, pub o_b: Vec<f32>,
    /// Per-head RMSNorm gamma applied to Q before SDPA (`norm_q.weight`).
    pub norm_q_gamma: Vec<f32>,
    /// Per-head RMSNorm gamma for K.
    pub norm_k_gamma: Vec<f32>,
}

impl AttnWeights {
    fn load(ss: &ShardedSafetensors, prefix: &str) -> Result<Self, Wan22Error> {
        let load_proj = |slot: &str| -> Result<(F32Slice, Vec<f32>), Wan22Error> {
            let w = lf(ss, &format!("{prefix}.{slot}.weight"))?;
            let b = lf_vec(ss, &format!("{prefix}.{slot}.bias"))?;
            if w.len() != DIM * DIM {
                return Err(Wan22Error::Load(format!(
                    "{prefix}.{slot}.weight: expected {}, got {}", DIM*DIM, w.len())));
            }
            if b.len() != DIM {
                return Err(Wan22Error::Load(format!(
                    "{prefix}.{slot}.bias: expected {DIM}, got {}", b.len())));
            }
            Ok((w, b))
        };
        let (q_w, q_b) = load_proj("q")?;
        let (k_w, k_b) = load_proj("k")?;
        let (v_w, v_b) = load_proj("v")?;
        let (o_w, o_b) = load_proj("o")?;
        let norm_q_gamma = lf_vec(ss, &format!("{prefix}.norm_q.weight"))?;
        let norm_k_gamma = lf_vec(ss, &format!("{prefix}.norm_k.weight"))?;
        if norm_q_gamma.len() != DIM {
            return Err(Wan22Error::Load(format!(
                "{prefix}.norm_q.weight: expected {DIM}, got {}", norm_q_gamma.len())));
        }
        Ok(Self { q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b, norm_q_gamma, norm_k_gamma })
    }
}

/// One DiT block: self-attn → cross-attn → FFN, all gated by `modulation`
/// + time embedding (AdaLN-Zero style).
pub struct Wan22Block {
    /// `[6, DIM]` learned per-block bias added to time-conditioned modulation.
    /// 6 = (scale, shift, gate) × (attn, ffn).
    pub modulation: Vec<f32>,
    pub self_attn: AttnWeights,
    pub cross_attn: AttnWeights,
    /// LayerNorm applied after cross-attn and before FFN.
    pub norm3_weight: Vec<f32>,
    pub norm3_bias: Vec<f32>,
    /// FFN: `ffn.0` (3072 → 14336) → GELU → `ffn.2` (14336 → 3072).
    pub ffn_0_w: F32Slice, pub ffn_0_b: Vec<f32>,
    pub ffn_2_w: F32Slice, pub ffn_2_b: Vec<f32>,
}

impl Wan22Block {
    fn load(ss: &ShardedSafetensors, i: usize) -> Result<Self, Wan22Error> {
        let p = format!("blocks.{i}");
        let modulation = lf_vec(ss, &format!("{p}.modulation"))?;
        if modulation.len() != 6 * DIM {
            return Err(Wan22Error::Load(format!(
                "blocks.{i}.modulation: expected {} = 6×{DIM}, got {}",
                6 * DIM, modulation.len())));
        }
        let self_attn = AttnWeights::load(ss, &format!("{p}.self_attn"))?;
        let cross_attn = AttnWeights::load(ss, &format!("{p}.cross_attn"))?;
        let norm3_weight = lf_vec(ss, &format!("{p}.norm3.weight"))?;
        let norm3_bias = lf_vec(ss, &format!("{p}.norm3.bias"))?;
        let ffn_0_w = lf(ss, &format!("{p}.ffn.0.weight"))?;
        let ffn_0_b = lf_vec(ss, &format!("{p}.ffn.0.bias"))?;
        let ffn_2_w = lf(ss, &format!("{p}.ffn.2.weight"))?;
        let ffn_2_b = lf_vec(ss, &format!("{p}.ffn.2.bias"))?;
        if ffn_0_w.len() != FFN_DIM * DIM {
            return Err(Wan22Error::Load(format!(
                "blocks.{i}.ffn.0.weight: expected {}, got {}",
                FFN_DIM * DIM, ffn_0_w.len())));
        }
        if ffn_2_w.len() != DIM * FFN_DIM {
            return Err(Wan22Error::Load(format!(
                "blocks.{i}.ffn.2.weight: expected {}, got {}",
                DIM * FFN_DIM, ffn_2_w.len())));
        }
        Ok(Self {
            modulation, self_attn, cross_attn,
            norm3_weight, norm3_bias,
            ffn_0_w, ffn_0_b, ffn_2_w, ffn_2_b,
        })
    }
}

/// Full Wan 2.2 TI2V-5B transformer.  Pairs with `rce-cosmos3::phase4_v2::Wan22VAE_v2`
/// for image/video latent ↔ pixel conversion.
pub struct Wan22Ti2v {
    /// `patch_embedding`: 3D conv `[DIM=3072, IN_DIM=48, 1, 2, 2]`.
    pub patch_embed_w: F32Slice, pub patch_embed_b: Vec<f32>,
    /// Text MLP: T5_DIM=4096 → DIM (with GELU between).
    pub text_embed_0_w: F32Slice, pub text_embed_0_b: Vec<f32>,
    pub text_embed_2_w: F32Slice, pub text_embed_2_b: Vec<f32>,
    /// Time MLP: FREQ_DIM=256 → DIM (with SiLU/GELU between).
    pub time_embed_0_w: F32Slice, pub time_embed_0_b: Vec<f32>,
    pub time_embed_2_w: F32Slice, pub time_embed_2_b: Vec<f32>,
    pub blocks: Vec<Wan22Block>,
    /// Output head: `[1, 2, DIM]` modulation + Linear `[PATCH_OUT_DIM=192, DIM]`.
    pub head_modulation: Vec<f32>,
    pub head_w: F32Slice, pub head_b: Vec<f32>,
}

impl Wan22Ti2v {
    /// Load from `<root>/diffusion_pytorch_model.safetensors.index.json`.
    pub fn load(model_root: &Path) -> Result<Self, Wan22Error> {
        Self::load_with_layers(model_root, N_LAYERS)
    }

    /// Load with a subset of `n_layers` blocks (for quick smoke tests
    /// without paying the full ~18.6 GB mmap cost).
    pub fn load_with_layers(model_root: &Path, n_layers: usize) -> Result<Self, Wan22Error> {
        let idx = model_root.join("diffusion_pytorch_model.safetensors.index.json");
        let ss = ShardedSafetensors::from_index(&idx)?;
        let patch_embed_w = lf(&ss, "patch_embedding.weight")?;
        let patch_embed_b = lf_vec(&ss, "patch_embedding.bias")?;
        if patch_embed_w.len() != DIM * IN_DIM * PATCH_T * PATCH_H * PATCH_W {
            return Err(Wan22Error::Load(format!(
                "patch_embedding.weight: expected {} = {DIM}×{IN_DIM}×{PATCH_T}×{PATCH_H}×{PATCH_W}, got {}",
                DIM * IN_DIM * PATCH_T * PATCH_H * PATCH_W, patch_embed_w.len())));
        }
        let text_embed_0_w = lf(&ss, "text_embedding.0.weight")?;
        let text_embed_0_b = lf_vec(&ss, "text_embedding.0.bias")?;
        let text_embed_2_w = lf(&ss, "text_embedding.2.weight")?;
        let text_embed_2_b = lf_vec(&ss, "text_embedding.2.bias")?;
        if text_embed_0_w.len() != DIM * T5_DIM {
            return Err(Wan22Error::Load(format!(
                "text_embedding.0.weight: expected {} = {DIM}×{T5_DIM}, got {}",
                DIM * T5_DIM, text_embed_0_w.len())));
        }
        let time_embed_0_w = lf(&ss, "time_embedding.0.weight")?;
        let time_embed_0_b = lf_vec(&ss, "time_embedding.0.bias")?;
        let time_embed_2_w = lf(&ss, "time_embedding.2.weight")?;
        let time_embed_2_b = lf_vec(&ss, "time_embedding.2.bias")?;
        if time_embed_0_w.len() != DIM * FREQ_DIM {
            return Err(Wan22Error::Load(format!(
                "time_embedding.0.weight: expected {}, got {}",
                DIM * FREQ_DIM, time_embed_0_w.len())));
        }
        let blocks: Vec<Wan22Block> = (0..n_layers)
            .map(|i| Wan22Block::load(&ss, i))
            .collect::<Result<_, _>>()?;
        let head_modulation = lf_vec(&ss, "head.modulation")?;
        if head_modulation.len() != 2 * DIM {
            return Err(Wan22Error::Load(format!(
                "head.modulation: expected {} = 2×{DIM}, got {}",
                2 * DIM, head_modulation.len())));
        }
        let head_w = lf(&ss, "head.head.weight")?;
        let head_b = lf_vec(&ss, "head.head.bias")?;
        if head_w.len() != PATCH_OUT_DIM * DIM {
            return Err(Wan22Error::Load(format!(
                "head.head.weight: expected {} = {PATCH_OUT_DIM}×{DIM}, got {}",
                PATCH_OUT_DIM * DIM, head_w.len())));
        }
        Ok(Self {
            patch_embed_w, patch_embed_b,
            text_embed_0_w, text_embed_0_b,
            text_embed_2_w, text_embed_2_b,
            time_embed_0_w, time_embed_0_b,
            time_embed_2_w, time_embed_2_b,
            blocks,
            head_modulation, head_w, head_b,
        })
    }

    /// Rough heap-resident size estimate (in user-heap bytes).  Borrow-direct
    /// weights live in mmap so this is small (just biases + norm gammas);
    /// the full ~18.6 GB stays in the kernel page cache.
    pub fn user_heap_bytes(&self) -> usize {
        let mut n = 0;
        n += self.patch_embed_b.len() * 4;
        n += (self.text_embed_0_b.len() + self.text_embed_2_b.len()) * 4;
        n += (self.time_embed_0_b.len() + self.time_embed_2_b.len()) * 4;
        for b in &self.blocks {
            n += b.modulation.len() * 4;
            n += (b.self_attn.q_b.len() + b.self_attn.k_b.len()
                  + b.self_attn.v_b.len() + b.self_attn.o_b.len()) * 4;
            n += (b.self_attn.norm_q_gamma.len() + b.self_attn.norm_k_gamma.len()) * 4;
            n += (b.cross_attn.q_b.len() + b.cross_attn.k_b.len()
                  + b.cross_attn.v_b.len() + b.cross_attn.o_b.len()) * 4;
            n += (b.cross_attn.norm_q_gamma.len() + b.cross_attn.norm_k_gamma.len()) * 4;
            n += (b.norm3_weight.len() + b.norm3_bias.len()) * 4;
            n += (b.ffn_0_b.len() + b.ffn_2_b.len()) * 4;
        }
        n += self.head_modulation.len() * 4;
        n += self.head_b.len() * 4;
        n
    }
}

// ════════════════════════════════════════════════════════════════════════
// Forward primitives (Stage 1)
// ════════════════════════════════════════════════════════════════════════

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

/// Tanh-approx GELU (the BERT/T5 variant that the Wan reference uses for
/// the FFN; matches diffusers `gelu_pytorch_tanh` activation).
fn gelu_exact(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// Apply SiLU in place.
fn apply_silu(x: &mut [f32]) { for v in x.iter_mut() { *v = silu(*v); } }

/// Apply GELU in place.
fn apply_gelu(x: &mut [f32]) { for v in x.iter_mut() { *v = gelu_exact(*v); } }

/// Sinusoidal time embedding: `t * 1000` → `[FREQ_DIM]` with sin/cos pairs
/// (cos first, then sin — matches the diffusers convention used by Wan).
fn sinusoidal_time_embed(t: f32) -> [f32; FREQ_DIM] {
    let mut out = [0.0f32; FREQ_DIM];
    let half = FREQ_DIM / 2;
    let log_max = (10000.0_f32).ln();
    for k in 0..half {
        let freq = (-log_max * (k as f32) / (half as f32)).exp();
        let arg = t * 1000.0 * freq;
        out[k] = arg.cos();
        out[k + half] = arg.sin();
    }
    out
}

/// `out[m, n] = sum_k(input[m, k] * weight[n, k]) + bias[n]` for f32 weight
/// in PyTorch nn.Linear layout `[N, K]`.  Rayon-parallel over the M rows.
fn matmul_nt_f32_bias(
    input: &[f32], m: usize, k: usize,
    weight: &[f32], bias: Option<&[f32]>, n: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    debug_assert_eq!(input.len(), m * k);
    debug_assert_eq!(weight.len(), n * k);
    let mut out = vec![0.0f32; m * n];
    out.par_chunks_mut(n).enumerate().for_each(|(mi, row)| {
        let in_row = &input[mi * k .. (mi + 1) * k];
        for ni in 0..n {
            let w_row = &weight[ni * k .. (ni + 1) * k];
            let mut s = 0.0f32;
            for ki in 0..k { s += in_row[ki] * w_row[ki]; }
            row[ni] = s + bias.map(|b| b[ni]).unwrap_or(0.0);
        }
    });
    out
}

/// Affine LayerNorm with learned weight + bias, eps from config.
/// `x` is `[T, D]` row-major.
fn layernorm_affine(x: &[f32], t: usize, d: usize, weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    debug_assert_eq!(x.len(), t * d);
    debug_assert_eq!(weight.len(), d);
    debug_assert_eq!(bias.len(), d);
    let mut out = vec![0.0f32; x.len()];
    for ti in 0..t {
        let row = &x[ti * d .. (ti + 1) * d];
        let mean: f32 = row.iter().sum::<f32>() / (d as f32);
        let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (d as f32);
        let inv = (var + eps).sqrt().recip();
        for di in 0..d {
            out[ti * d + di] = (row[di] - mean) * inv * weight[di] + bias[di];
        }
    }
    out
}

/// Affine-free LayerNorm: mean-variance normalize then apply external
/// scale and shift (AdaLN-Zero), `out = norm(x) * (1 + scale) + shift`.
fn adaln_layernorm(x: &[f32], t: usize, d: usize, scale: &[f32], shift: &[f32], eps: f32) -> Vec<f32> {
    debug_assert_eq!(scale.len(), d);
    debug_assert_eq!(shift.len(), d);
    let mut out = vec![0.0f32; x.len()];
    for ti in 0..t {
        let row = &x[ti * d .. (ti + 1) * d];
        let mean: f32 = row.iter().sum::<f32>() / (d as f32);
        let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (d as f32);
        let inv = (var + eps).sqrt().recip();
        for di in 0..d {
            let normed = (row[di] - mean) * inv;
            out[ti * d + di] = normed * (1.0 + scale[di]) + shift[di];
        }
    }
    out
}

/// Per-head RMSNorm applied to Q or K after splitting into heads.
/// `q` shape is `[T, N_HEADS, HEAD_DIM]`.  `gamma` is `[DIM]` (per-channel,
/// applied uniformly across heads since channels are striped into heads).
fn qk_rmsnorm(q: &mut [f32], t: usize, gamma: &[f32], eps: f32) {
    debug_assert_eq!(q.len(), t * DIM);
    debug_assert_eq!(gamma.len(), DIM);
    for ti in 0..t {
        for h in 0..N_HEADS {
            let off = ti * DIM + h * HEAD_DIM;
            let row = &q[off .. off + HEAD_DIM];
            let ss: f32 = row.iter().map(|v| v * v).sum::<f32>() / (HEAD_DIM as f32);
            let inv = (ss + eps).sqrt().recip();
            for di in 0..HEAD_DIM {
                q[off + di] = q[off + di] * inv * gamma[h * HEAD_DIM + di];
            }
        }
    }
}

/// Wan 2.2 3D RoPE axis split — Wan's reference layout: the temporal axis
/// gets the "trunk" of `head_dim − 4·(head_dim/6)` channels, and each of
/// the two spatial axes (height, width) gets `2·(head_dim/6)` channels.
/// For HEAD_DIM=128: `t=44, h=42, w=42` ⇒ `t+h+w = 128` ✓.  Number of
/// complex pairs per axis: `T_PAIRS=22, H_PAIRS=21, W_PAIRS=21`.
pub const T_PAIRS: usize = (HEAD_DIM - 4 * (HEAD_DIM / 6)) / 2;
pub const H_PAIRS: usize = (2 * (HEAD_DIM / 6)) / 2;
pub const W_PAIRS: usize = (2 * (HEAD_DIM / 6)) / 2;

/// Maximum positions for which RoPE frequency tables are precomputed.
/// 1024 covers production resolutions on T/H/W axes.
pub const ROPE_MAX_LEN: usize = 1024;

/// Precomputed RoPE cos/sin tables for one axis.  Flat layout
/// `[ROPE_MAX_LEN × n_pairs × 2]` with `(cos, sin)` interleaved.
pub struct RopeFreqsAxis {
    pub n_pairs: usize,
    pub table: Vec<f32>,
}

impl RopeFreqsAxis {
    pub fn new(n_pairs: usize) -> Self {
        let mut table = vec![0.0f32; ROPE_MAX_LEN * n_pairs * 2];
        let log_max = (10000.0_f32).ln();
        for p in 0..ROPE_MAX_LEN {
            for k in 0..n_pairs {
                // freq = 10000^(-k/n_pairs) (n_pairs = dim/2 effectively)
                let freq = (-log_max * (k as f32) / (n_pairs as f32)).exp();
                let arg = (p as f32) * freq;
                table[(p * n_pairs + k) * 2] = arg.cos();
                table[(p * n_pairs + k) * 2 + 1] = arg.sin();
            }
        }
        Self { n_pairs, table }
    }
}

/// All three axis tables together.  Cached once per model load via
/// `Wan22Ti2v::rope_freqs()`.
pub struct RopeFreqs3D {
    pub t: RopeFreqsAxis,
    pub h: RopeFreqsAxis,
    pub w: RopeFreqsAxis,
}

impl RopeFreqs3D {
    pub fn new() -> Self {
        Self {
            t: RopeFreqsAxis::new(T_PAIRS),
            h: RopeFreqsAxis::new(H_PAIRS),
            w: RopeFreqsAxis::new(W_PAIRS),
        }
    }
}

impl Default for RopeFreqs3D {
    fn default() -> Self { Self::new() }
}

/// Apply 3D axis-fractional RoPE to a `[T_seq, N_HEADS, HEAD_DIM]` Q or K
/// tensor in place.  Per-head channels are split into three contiguous
/// chunks `[T_PAIRS, H_PAIRS, W_PAIRS]` (counted in complex pairs); each
/// chunk gets rotated by its axis's `cos/sin` at the token's position.
pub fn rope_apply_3d(
    x: &mut [f32],
    t_pos: &[usize], h_pos: &[usize], w_pos: &[usize],
    freqs: &RopeFreqs3D,
) {
    let t_seq = t_pos.len();
    debug_assert_eq!(x.len(), t_seq * DIM);
    debug_assert_eq!(h_pos.len(), t_seq);
    debug_assert_eq!(w_pos.len(), t_seq);
    let rotate = |x: &mut [f32], off: usize, pair: usize, cos_v: f32, sin_v: f32| {
        let a = x[off + 2 * pair];
        let b = x[off + 2 * pair + 1];
        x[off + 2 * pair]     = a * cos_v - b * sin_v;
        x[off + 2 * pair + 1] = a * sin_v + b * cos_v;
    };
    for ti in 0..t_seq {
        let (pt, ph, pw) = (t_pos[ti], h_pos[ti], w_pos[ti]);
        assert!(pt < ROPE_MAX_LEN && ph < ROPE_MAX_LEN && pw < ROPE_MAX_LEN,
                "rope position out of range (max {ROPE_MAX_LEN}); resize or clamp");
        for hi in 0..N_HEADS {
            let off = ti * DIM + hi * HEAD_DIM;
            // t-axis rotates pairs [0..T_PAIRS).
            for k in 0..T_PAIRS {
                let cos_v = freqs.t.table[(pt * T_PAIRS + k) * 2];
                let sin_v = freqs.t.table[(pt * T_PAIRS + k) * 2 + 1];
                rotate(x, off, k, cos_v, sin_v);
            }
            // h-axis rotates pairs [T_PAIRS..T_PAIRS+H_PAIRS).
            let h_base = T_PAIRS;
            for k in 0..H_PAIRS {
                let cos_v = freqs.h.table[(ph * H_PAIRS + k) * 2];
                let sin_v = freqs.h.table[(ph * H_PAIRS + k) * 2 + 1];
                rotate(x, off, h_base + k, cos_v, sin_v);
            }
            // w-axis rotates pairs [T_PAIRS+H_PAIRS..T_PAIRS+H_PAIRS+W_PAIRS).
            let w_base = T_PAIRS + H_PAIRS;
            for k in 0..W_PAIRS {
                let cos_v = freqs.w.table[(pw * W_PAIRS + k) * 2];
                let sin_v = freqs.w.table[(pw * W_PAIRS + k) * 2 + 1];
                rotate(x, off, w_base + k, cos_v, sin_v);
            }
        }
    }
}

/// Standard multi-head SDPA without RoPE.  Q/K/V shape `[T, DIM]` viewed
/// as `[T, N_HEADS, HEAD_DIM]`.  K/V can come from a different sequence
/// length (cross-attention).
fn sdpa(
    q: &[f32], t_q: usize, k: &[f32], v: &[f32], t_kv: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    let scale = (HEAD_DIM as f32).powf(-0.5);
    let mut out = vec![0.0f32; t_q * DIM];
    out.par_chunks_mut(DIM).enumerate().for_each(|(ti, out_row)| {
        for h in 0..N_HEADS {
            let q_off = ti * DIM + h * HEAD_DIM;
            // attn = softmax_j(q[ti, h, :] · k[j, h, :] * scale)
            let mut scores = vec![0.0f32; t_kv];
            let mut row_max = f32::NEG_INFINITY;
            for j in 0..t_kv {
                let k_off = j * DIM + h * HEAD_DIM;
                let mut s = 0.0f32;
                for di in 0..HEAD_DIM { s += q[q_off + di] * k[k_off + di]; }
                let sv = s * scale;
                scores[j] = sv;
                if sv > row_max { row_max = sv; }
            }
            let mut sum = 0.0f32;
            for j in 0..t_kv {
                let e = (scores[j] - row_max).exp();
                scores[j] = e;
                sum += e;
            }
            let inv = sum.recip();
            for j in 0..t_kv { scores[j] *= inv; }
            for di in 0..HEAD_DIM {
                let mut acc = 0.0f32;
                for j in 0..t_kv {
                    let v_off = j * DIM + h * HEAD_DIM;
                    acc += scores[j] * v[v_off + di];
                }
                out_row[h * HEAD_DIM + di] = acc;
            }
        }
    });
    out
}

/// 3D patch embed: conv3d with kernel `(1, 2, 2)` stride `(1, 2, 2)`.
/// Input `[IN_DIM=48, T, H, W]` → output `[T, H/2, W/2, DIM]` (token sequence).
/// Output H/W are halved; T is unchanged.
fn patch_embed(
    inp: &[f32], t: usize, h: usize, w: usize,
    weight: &[f32], bias: &[f32],
) -> Vec<f32> {
    use rayon::prelude::*;
    debug_assert_eq!(inp.len(), IN_DIM * t * h * w);
    debug_assert_eq!(weight.len(), DIM * IN_DIM * PATCH_T * PATCH_H * PATCH_W);
    debug_assert_eq!(bias.len(), DIM);
    let ho = h / PATCH_H;
    let wo = w / PATCH_W;
    let n_tok = t * ho * wo;
    let mut out = vec![0.0f32; n_tok * DIM];
    out.par_chunks_mut(DIM).enumerate().for_each(|(tok, dst)| {
        let ti = tok / (ho * wo);
        let hi = (tok % (ho * wo)) / wo;
        let wi = tok % wo;
        // Output channel oc; sum over (ic, kh, kw) since kT=1.
        for oc in 0..DIM {
            let mut acc = bias[oc];
            for ic in 0..IN_DIM {
                for kh in 0..PATCH_H {
                    for kw in 0..PATCH_W {
                        let w_idx = ((oc * IN_DIM + ic) * PATCH_T * PATCH_H * PATCH_W)
                            + 0 * PATCH_H * PATCH_W
                            + kh * PATCH_W + kw;
                        let in_idx = ic * t * h * w
                            + ti * h * w
                            + (hi * PATCH_H + kh) * w
                            + (wi * PATCH_W + kw);
                        acc += weight[w_idx] * inp[in_idx];
                    }
                }
            }
            dst[oc] = acc;
        }
    });
    out
}

/// Un-patchify head output `[N_tok, PATCH_OUT_DIM=192]` back to latent grid
/// `[OUT_DIM=48, T, H, W]`.  Inverse of `patch_embed` shape transform.
fn unpatchify(
    tokens: &[f32], t: usize, ho: usize, wo: usize,
) -> Vec<f32> {
    debug_assert_eq!(tokens.len(), t * ho * wo * PATCH_OUT_DIM);
    let h = ho * PATCH_H;
    let w = wo * PATCH_W;
    let mut out = vec![0.0f32; OUT_DIM * t * h * w];
    for ti in 0..t {
        for hi in 0..ho {
            for wi in 0..wo {
                let tok_off = ((ti * ho + hi) * wo + wi) * PATCH_OUT_DIM;
                for oc in 0..OUT_DIM {
                    for kh in 0..PATCH_H {
                        for kw in 0..PATCH_W {
                            let v = tokens[tok_off + oc * (PATCH_H * PATCH_W) + kh * PATCH_W + kw];
                            let out_idx = oc * t * h * w
                                + ti * h * w
                                + (hi * PATCH_H + kh) * w
                                + (wi * PATCH_W + kw);
                            out[out_idx] = v;
                        }
                    }
                }
            }
        }
    }
    out
}

// ── Time / Text embed helpers ───────────────────────────────────────────

impl Wan22Ti2v {
    /// Compute the time-conditioned hidden vector `e_t` from a scalar
    /// timestep `t ∈ [0, 1]`.  Path: sinusoidal embed → Linear → SiLU → Linear.
    pub fn time_embed(&self, t_scalar: f32) -> Vec<f32> {
        let freq = sinusoidal_time_embed(t_scalar);
        let mut h = matmul_nt_f32_bias(
            &freq, 1, FREQ_DIM,
            self.time_embed_0_w.as_slice(),
            Some(&self.time_embed_0_b), DIM);
        apply_silu(&mut h);
        matmul_nt_f32_bias(
            &h, 1, DIM,
            self.time_embed_2_w.as_slice(),
            Some(&self.time_embed_2_b), DIM)
    }

    /// Project T5-XXL output `[T_text, T5_DIM=4096]` to transformer
    /// context `[T_text, DIM=3072]` via the text MLP.
    pub fn text_embed(&self, t5_out: &[f32], t_text: usize) -> Vec<f32> {
        let mut h = matmul_nt_f32_bias(
            t5_out, t_text, T5_DIM,
            self.text_embed_0_w.as_slice(),
            Some(&self.text_embed_0_b), DIM);
        apply_gelu(&mut h);
        matmul_nt_f32_bias(
            &h, t_text, DIM,
            self.text_embed_2_w.as_slice(),
            Some(&self.text_embed_2_b), DIM)
    }
}

impl Wan22Block {
    /// One DiT block forward without RoPE (Stage 1 baseline).  Kept for
    /// regression tests against the no-RoPE first-cut behavior.
    pub fn forward(
        &self,
        x: &[f32], t_seq: usize,
        e_t: &[f32],
        text_ctx: &[f32], t_text: usize,
        eps: f32,
    ) -> Vec<f32> {
        self.forward_inner(x, t_seq, e_t, text_ctx, t_text, eps, None)
    }

    /// One DiT block forward with 3D RoPE applied to self-attention Q/K
    /// after qk-norm.  Cross-attention does NOT receive RoPE (text tokens
    /// have no spatial position).  Caller supplies the per-token
    /// `(t_pos, h_pos, w_pos)` integer grid coordinates and the cached
    /// `RopeFreqs3D` tables.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_rope(
        &self,
        x: &[f32], t_seq: usize,
        e_t: &[f32],
        text_ctx: &[f32], t_text: usize,
        eps: f32,
        t_pos: &[usize], h_pos: &[usize], w_pos: &[usize],
        freqs: &RopeFreqs3D,
    ) -> Vec<f32> {
        self.forward_inner(
            x, t_seq, e_t, text_ctx, t_text, eps,
            Some((t_pos, h_pos, w_pos, freqs)))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        x: &[f32], t_seq: usize,
        e_t: &[f32],
        text_ctx: &[f32], t_text: usize,
        eps: f32,
        rope: Option<(&[usize], &[usize], &[usize], &RopeFreqs3D)>,
    ) -> Vec<f32> {
        debug_assert_eq!(x.len(), t_seq * DIM);
        debug_assert_eq!(e_t.len(), DIM);
        debug_assert_eq!(text_ctx.len(), t_text * DIM);

        // ── e_t + per-block modulation [6, DIM] → six [DIM] params ──
        let mut e_mod = vec![0.0f32; 6 * DIM];
        for k in 0..6 {
            for d in 0..DIM {
                e_mod[k * DIM + d] = self.modulation[k * DIM + d] + e_t[d];
            }
        }
        let scale_attn = &e_mod[0 * DIM .. 1 * DIM];
        let shift_attn = &e_mod[1 * DIM .. 2 * DIM];
        let gate_attn  = &e_mod[2 * DIM .. 3 * DIM];
        let scale_ffn  = &e_mod[3 * DIM .. 4 * DIM];
        let shift_ffn  = &e_mod[4 * DIM .. 5 * DIM];
        let gate_ffn   = &e_mod[5 * DIM .. 6 * DIM];

        // ── Self-attn with AdaLN modulation ──
        let mut x_norm = adaln_layernorm(x, t_seq, DIM, scale_attn, shift_attn, eps);
        let q = self.self_attn_qkvo_proj(&x_norm, t_seq, &self.self_attn, ProjKind::Q);
        let k = self.self_attn_qkvo_proj(&x_norm, t_seq, &self.self_attn, ProjKind::K);
        let v = self.self_attn_qkvo_proj(&x_norm, t_seq, &self.self_attn, ProjKind::V);
        let mut q = q; let mut k = k;
        qk_rmsnorm(&mut q, t_seq, &self.self_attn.norm_q_gamma, eps);
        qk_rmsnorm(&mut k, t_seq, &self.self_attn.norm_k_gamma, eps);
        if let Some((tp, hp, wp, fr)) = rope {
            rope_apply_3d(&mut q, tp, hp, wp, fr);
            rope_apply_3d(&mut k, tp, hp, wp, fr);
        }
        let attn = sdpa(&q, t_seq, &k, &v, t_seq);
        let attn_out = self.self_attn_qkvo_proj(&attn, t_seq, &self.self_attn, ProjKind::O);

        // x = x + gate_attn * attn_out (broadcast gate over the sequence)
        let mut x_new = x.to_vec();
        for ti in 0..t_seq {
            for d in 0..DIM {
                x_new[ti * DIM + d] += gate_attn[d] * attn_out[ti * DIM + d];
            }
        }
        x_norm = layernorm_affine(&x_new, t_seq, DIM, &self.norm3_weight, &self.norm3_bias, eps);

        // ── Cross-attn: Q from x_norm, K/V from text_ctx ──
        let q_c = self.self_attn_qkvo_proj(&x_norm, t_seq, &self.cross_attn, ProjKind::Q);
        let k_c = self.self_attn_qkvo_proj(text_ctx, t_text, &self.cross_attn, ProjKind::K);
        let v_c = self.self_attn_qkvo_proj(text_ctx, t_text, &self.cross_attn, ProjKind::V);
        let mut q_c = q_c; let mut k_c = k_c;
        qk_rmsnorm(&mut q_c, t_seq, &self.cross_attn.norm_q_gamma, eps);
        qk_rmsnorm(&mut k_c, t_text, &self.cross_attn.norm_k_gamma, eps);
        let cross = sdpa(&q_c, t_seq, &k_c, &v_c, t_text);
        let cross_out = self.self_attn_qkvo_proj(&cross, t_seq, &self.cross_attn, ProjKind::O);
        for ti in 0..t_seq {
            for d in 0..DIM { x_new[ti * DIM + d] += cross_out[ti * DIM + d]; }
        }

        // ── FFN with AdaLN modulation ──
        let x_norm2 = adaln_layernorm(&x_new, t_seq, DIM, scale_ffn, shift_ffn, eps);
        let mut h = matmul_nt_f32_bias(
            &x_norm2, t_seq, DIM,
            self.ffn_0_w.as_slice(),
            Some(&self.ffn_0_b), FFN_DIM);
        apply_gelu(&mut h);
        let ffn_out = matmul_nt_f32_bias(
            &h, t_seq, FFN_DIM,
            self.ffn_2_w.as_slice(),
            Some(&self.ffn_2_b), DIM);
        for ti in 0..t_seq {
            for d in 0..DIM {
                x_new[ti * DIM + d] += gate_ffn[d] * ffn_out[ti * DIM + d];
            }
        }
        x_new
    }

    fn self_attn_qkvo_proj(
        &self, input: &[f32], t: usize, attn: &AttnWeights, kind: ProjKind,
    ) -> Vec<f32> {
        let (w, b) = match kind {
            ProjKind::Q => (attn.q_w.as_slice(), Some(&attn.q_b[..])),
            ProjKind::K => (attn.k_w.as_slice(), Some(&attn.k_b[..])),
            ProjKind::V => (attn.v_w.as_slice(), Some(&attn.v_b[..])),
            ProjKind::O => (attn.o_w.as_slice(), Some(&attn.o_b[..])),
        };
        matmul_nt_f32_bias(input, t, DIM, w, b, DIM)
    }
}

#[derive(Clone, Copy)]
enum ProjKind { Q, K, V, O }

impl Wan22Ti2v {
    /// Full transformer forward without RoPE (regression / first-cut baseline).
    pub fn forward(
        &self,
        latent: &[f32], t: usize, h: usize, w: usize,
        t5_out: &[f32], t_text: usize,
        t_scalar: f32,
        eps: f32,
    ) -> Vec<f32> {
        self.forward_inner(latent, t, h, w, t5_out, t_text, t_scalar, eps, None)
    }

    /// Full transformer forward with 3D RoPE on self-attention.  Builds the
    /// canonical `(t_pos, h_pos, w_pos)` grid from the post-patch latent
    /// shape: `(t·ho·wo)` tokens with positions iterating outer-temporal,
    /// inner-height, innermost-width.
    pub fn forward_with_rope(
        &self,
        latent: &[f32], t: usize, h: usize, w: usize,
        t5_out: &[f32], t_text: usize,
        t_scalar: f32,
        eps: f32,
        freqs: &RopeFreqs3D,
    ) -> Vec<f32> {
        self.forward_inner(latent, t, h, w, t5_out, t_text, t_scalar, eps, Some(freqs))
    }

    fn forward_inner(
        &self,
        latent: &[f32], t: usize, h: usize, w: usize,
        t5_out: &[f32], t_text: usize,
        t_scalar: f32,
        eps: f32,
        rope: Option<&RopeFreqs3D>,
    ) -> Vec<f32> {
        let e_t = self.time_embed(t_scalar);
        let text_ctx = self.text_embed(t5_out, t_text);

        let mut x = patch_embed(
            latent, t, h, w,
            self.patch_embed_w.as_slice(),
            &self.patch_embed_b);
        let ho = h / PATCH_H;
        let wo = w / PATCH_W;
        let t_seq = t * ho * wo;

        // Build canonical position grids when RoPE is enabled.
        let positions = rope.map(|f| {
            let mut t_pos = Vec::with_capacity(t_seq);
            let mut h_pos = Vec::with_capacity(t_seq);
            let mut w_pos = Vec::with_capacity(t_seq);
            for ti in 0..t {
                for hi in 0..ho {
                    for wi in 0..wo {
                        t_pos.push(ti); h_pos.push(hi); w_pos.push(wi);
                    }
                }
            }
            (t_pos, h_pos, w_pos, f)
        });

        for b in &self.blocks {
            x = match &positions {
                Some((tp, hp, wp, fr)) => b.forward_with_rope(
                    &x, t_seq, &e_t, &text_ctx, t_text, eps, tp, hp, wp, fr),
                None => b.forward(&x, t_seq, &e_t, &text_ctx, t_text, eps),
            };
        }

        // ── Output head: AdaLN with [scale, shift] from e_t + head_modulation ──
        let mut head_mod = vec![0.0f32; 2 * DIM];
        for k in 0..2 {
            for d in 0..DIM {
                head_mod[k * DIM + d] = self.head_modulation[k * DIM + d] + e_t[d];
            }
        }
        let scale_head = &head_mod[0 * DIM .. 1 * DIM];
        let shift_head = &head_mod[1 * DIM .. 2 * DIM];
        let x_norm = adaln_layernorm(&x, t_seq, DIM, scale_head, shift_head, eps);

        let tokens = matmul_nt_f32_bias(
            &x_norm, t_seq, DIM,
            self.head_w.as_slice(),
            Some(&self.head_b), PATCH_OUT_DIM);

        unpatchify(&tokens, t, ho, wo)
    }
}

// ════════════════════════════════════════════════════════════════════════
// Multi-step diffusion sampling (FlowMatch / UniPC AB2)
// ════════════════════════════════════════════════════════════════════════

/// Rectified-flow inference schedule.  Wan 2.2 uses linear σ from 1.0 → 0.0
/// over `n_steps`; a `shift` parameter (≠1) biases the schedule toward the
/// high-σ region (Flux convention: `σ_shifted = shift·σ_lin / (1 + (shift−1)·σ_lin)`).
#[derive(Debug, Clone)]
pub struct FlowMatchSchedule {
    pub n_steps: usize,
    pub shift: f32,
}

impl FlowMatchSchedule {
    pub fn linear(n_steps: usize) -> Self { Self { n_steps, shift: 1.0 } }
    pub fn shifted(n_steps: usize, shift: f32) -> Self { Self { n_steps, shift } }

    /// Returns `n_steps + 1` sigmas (including the σ=0 endpoint).
    pub fn sigmas(&self) -> Vec<f32> {
        let n = self.n_steps;
        (0..=n).map(|i| {
            let lin = 1.0 - (i as f32) / (n as f32);
            if (self.shift - 1.0).abs() < f32::EPSILON { lin }
            else { self.shift * lin / (1.0 + (self.shift - 1.0) * lin) }
        }).collect()
    }
}

impl Wan22Ti2v {
    /// One-step Euler integration in the Wan22 latent space.
    /// `latent` is the noisy x at σ_curr; `dt = σ_curr − σ_next`;
    /// returns `x_next = x − dt · ε` where ε = `forward_with_rope(latent, …)`.
    #[allow(clippy::too_many_arguments)]
    fn euler_step(
        &self,
        latent: &[f32], t: usize, h: usize, w: usize,
        text_ctx: &[f32], t_text: usize,
        sigma: f32, dt: f32,
        eps_const: f32,
        freqs: &RopeFreqs3D,
    ) -> Vec<f32> {
        let velocity = self.forward_with_rope(
            latent, t, h, w, text_ctx, t_text, sigma, eps_const, freqs);
        let mut out = latent.to_vec();
        for i in 0..out.len() { out[i] -= dt * velocity[i]; }
        out
    }

    /// Multi-step Euler rectified-flow sampling.  `x_init` is noise at σ=1;
    /// integration follows the schedule down to σ=0.
    #[allow(clippy::too_many_arguments)]
    pub fn euler_sample_scheduled(
        &self,
        x_init: Vec<f32>, t: usize, h: usize, w: usize,
        text_ctx: &[f32], t_text: usize,
        schedule: &FlowMatchSchedule,
        eps_const: f32,
        freqs: &RopeFreqs3D,
    ) -> Vec<f32> {
        let sigmas = schedule.sigmas();
        let mut x = x_init;
        for i in 0..sigmas.len() - 1 {
            let dt = sigmas[i] - sigmas[i + 1];
            x = self.euler_step(&x, t, h, w, text_ctx, t_text,
                sigmas[i], dt, eps_const, freqs);
        }
        x
    }

    /// UniPC-style 2nd-order multistep sampling (Adams-Bashforth 2).
    /// Step 0 bootstraps with Euler; steps 1+ use `1.5·ε_curr − 0.5·ε_prev`
    /// as the drift, matching the predictor part of UniPC-bh1.
    #[allow(clippy::too_many_arguments)]
    pub fn unipc_sample_scheduled(
        &self,
        x_init: Vec<f32>, t: usize, h: usize, w: usize,
        text_ctx: &[f32], t_text: usize,
        schedule: &FlowMatchSchedule,
        eps_const: f32,
        freqs: &RopeFreqs3D,
    ) -> Vec<f32> {
        let sigmas = schedule.sigmas();
        let mut x = x_init;
        let mut eps_prev: Option<Vec<f32>> = None;
        for i in 0..sigmas.len() - 1 {
            let dt = sigmas[i] - sigmas[i + 1];
            let eps_curr = self.forward_with_rope(
                &x, t, h, w, text_ctx, t_text, sigmas[i], eps_const, freqs);
            if let Some(prev) = &eps_prev {
                for j in 0..x.len() {
                    x[j] -= dt * (1.5 * eps_curr[j] - 0.5 * prev[j]);
                }
            } else {
                for j in 0..x.len() { x[j] -= dt * eps_curr[j]; }
            }
            eps_prev = Some(eps_curr);
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn wan22_ti2v_dir() -> Option<PathBuf> {
        for vol in ["macos_4TB_external", "macos_4TB_external_1", "macos_4TB_external_2"] {
            let p = PathBuf::from(format!(
                "/Volumes/{vol}/neuronexus_ai/model_library/Wan-AI_Wan2.2-TI2V-5B"));
            if p.join("diffusion_pytorch_model.safetensors.index.json").exists() {
                return Some(p);
            }
        }
        None
    }

    /// Stage 0: load all 825 tensors (or a subset of N layers' worth),
    /// verify the expected shapes, and confirm borrow-direct keeps the
    /// user heap small (≪ 1 GB even for the full 30-layer load).
    #[test]
    #[ignore = "wan22-ti2v Stage 0 — needs Wan 2.2 TI2V-5B shards mounted"]
    fn wan22_ti2v_load_first_layers() {
        let Some(dir) = wan22_ti2v_dir() else {
            eprintln!("Wan 2.2 TI2V-5B dir not found"); return;
        };
        let t0 = std::time::Instant::now();
        let m = Wan22Ti2v::load_with_layers(&dir, 2).expect("load");
        eprintln!("wan22-ti2v: loaded 2 layers + embeds + head in {:?}", t0.elapsed());

        assert_eq!(m.patch_embed_w.len(), DIM * IN_DIM * PATCH_T * PATCH_H * PATCH_W);
        assert_eq!(m.patch_embed_b.len(), DIM);
        assert_eq!(m.head_w.len(), PATCH_OUT_DIM * DIM);
        assert_eq!(m.head_modulation.len(), 2 * DIM);

        for (i, b) in m.blocks.iter().enumerate() {
            assert_eq!(b.modulation.len(), 6 * DIM, "block {i} modulation");
            assert_eq!(b.self_attn.q_w.len(), DIM * DIM, "block {i} self_attn.q_w");
            assert_eq!(b.self_attn.norm_q_gamma.len(), DIM);
            assert_eq!(b.cross_attn.q_w.len(), DIM * DIM, "block {i} cross_attn.q_w");
            assert_eq!(b.ffn_0_w.len(), FFN_DIM * DIM, "block {i} ffn.0.weight");
            assert_eq!(b.ffn_2_w.len(), DIM * FFN_DIM, "block {i} ffn.2.weight");
        }
        let mb = m.user_heap_bytes() as f64 / (1024.0 * 1024.0);
        eprintln!("wan22-ti2v user heap (borrow-direct): {:.2} MB for 2 layers", mb);
        assert!(mb < 50.0, "user heap should be small with borrow-direct, got {mb} MB");
        eprintln!("wan22-ti2v Stage 0 ✓");
    }

    /// Stage 1 forward smoke: load a 2-block subset, run the full forward
    /// (time embed + text embed + patch embed + blocks + head + un-patchify)
    /// on a tiny synthetic latent + synthetic T5 context.  Verifies every
    /// primitive composes into a finite output with the right output shape
    /// (`[OUT_DIM=48, T, H, W]`).  Note: no RoPE in first-cut — numerics are
    /// not in reference agreement, but shape + finiteness must hold.
    #[test]
    #[ignore = "wan22-ti2v Stage 1 — needs Wan 2.2 TI2V-5B shards mounted"]
    fn wan22_ti2v_forward_smoke() {
        let Some(dir) = wan22_ti2v_dir() else {
            eprintln!("Wan 2.2 TI2V-5B dir not found"); return;
        };
        let m = Wan22Ti2v::load_with_layers(&dir, 2).expect("load");

        // Tiny synthetic latent at the smallest meaningful resolution:
        // T=1 (single frame), H=4, W=4 (post-VAE-encode shape for a 64×64 input).
        // After 1×2×2 patch: T·(H/2)·(W/2) = 1·2·2 = 4 tokens.
        let (t, h, w) = (1, 4, 4);
        let latent: Vec<f32> = (0..IN_DIM * t * h * w)
            .map(|i| ((i as f32) * 0.013).sin() * 0.1).collect();

        // Synthetic T5 context: T_text=8 tokens, T5_DIM=4096.
        let t_text = 8;
        let t5_out: Vec<f32> = (0..t_text * T5_DIM)
            .map(|i| ((i as f32) * 0.011).cos() * 0.05).collect();

        let t0 = std::time::Instant::now();
        let pred = m.forward(&latent, t, h, w, &t5_out, t_text, 0.5, 1e-6);
        let dt = t0.elapsed();

        let expected = OUT_DIM * t * h * w;
        assert_eq!(pred.len(), expected, "output shape");
        assert!(pred.iter().all(|v| v.is_finite()), "all outputs finite");
        let max_abs = pred.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let mean_abs = pred.iter().map(|v| v.abs()).sum::<f32>() / (pred.len() as f32);
        eprintln!("wan22-ti2v forward ({} blocks): {dt:?} max_abs={max_abs:.4} mean_abs={mean_abs:.4}",
                  m.blocks.len());
        assert!(max_abs > 0.0, "output should be nonzero");
        eprintln!("wan22-ti2v Stage 1 forward ✓");
    }

    /// Wan22Ti2v full forward vs hand-rolled PyTorch reference.  Verifies
    /// rce's port composes the architecture correctly (AdaLN-Zero
    /// modulation + qk-norm + 3D RoPE + cross-attn + GELU FFN + patch
    /// embed conv + head + un-patchify) at numerical equivalence to a
    /// reference Python implementation that mirrors the documented
    /// operations.  Reference captured 2026-06-05 to
    /// `/tmp/wan22_ref_2layer.f32.bin` (768 f32 elements = [48, 1, 4, 4]).
    #[test]
    #[ignore = "wan22-ti2v reference — needs shards + /tmp/wan22_ref_2layer.f32.bin"]
    fn wan22_ti2v_forward_vs_python_reference() {
        let Some(dir) = wan22_ti2v_dir() else {
            eprintln!("Wan 2.2 TI2V-5B dir not found"); return;
        };
        let ref_path = std::path::PathBuf::from("/tmp/wan22_ref_2layer.f32.bin");
        if !ref_path.exists() {
            eprintln!("Wan22 reference binary not found"); return;
        }
        let ref_bytes = std::fs::read(&ref_path).expect("read ref");
        let expected_bytes = OUT_DIM * 1 * 4 * 4 * 4;
        assert_eq!(ref_bytes.len(), expected_bytes,
                   "ref binary size mismatch ({} vs {})", ref_bytes.len(), expected_bytes);
        let ref_data: Vec<f32> = ref_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let ref_norm: f32 = ref_data.iter().map(|v| v*v).sum::<f32>().sqrt();
        let ref_max: f32 = ref_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("Python ref: ‖.‖={ref_norm:.4} max_abs={ref_max:.4}");

        let m = Wan22Ti2v::load_with_layers(&dir, 2).expect("load");
        let freqs = RopeFreqs3D::new();
        // Same deterministic input as Python script:
        // latent = sin(i * 0.013) * 0.1, t5_out = cos(i * 0.011) * 0.05, t=0.5
        let (t, h, w) = (1, 4, 4);
        let latent: Vec<f32> = (0..IN_DIM * t * h * w)
            .map(|i| ((i as f32) * 0.013).sin() * 0.1).collect();
        let t_text = 8;
        let t5_out: Vec<f32> = (0..t_text * T5_DIM)
            .map(|i| ((i as f32) * 0.011).cos() * 0.05).collect();

        let t0 = std::time::Instant::now();
        let rce_out = m.forward_with_rope(
            &latent, t, h, w, &t5_out, t_text, 0.5, 1e-6, &freqs);
        let dt = t0.elapsed();
        let rce_norm: f32 = rce_out.iter().map(|v| v*v).sum::<f32>().sqrt();
        let rce_max: f32 = rce_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("rce forward: {dt:?} ‖.‖={rce_norm:.4} max_abs={rce_max:.4}");

        assert_eq!(rce_out.len(), ref_data.len());
        let mut max_abs_diff = 0.0f32;
        let mut sum_sq_diff = 0.0f32;
        for (a, b) in rce_out.iter().zip(ref_data.iter()) {
            let d = (a - b).abs();
            if d > max_abs_diff { max_abs_diff = d; }
            sum_sq_diff += d * d;
        }
        let rms_diff = (sum_sq_diff / (rce_out.len() as f32)).sqrt();
        let rel_max = max_abs_diff / ref_max;
        eprintln!("vs Python ref: max_abs_diff={max_abs_diff:.4}  rms_diff={rms_diff:.4}  rel_max={:.2}%",
                  rel_max * 100.0);
        // f32 throughout, same accumulation order → expect very tight agreement.
        assert!(rel_max < 0.01,
                "rce Wan22Ti2v forward should match Python ref within 1%; got {}%",
                rel_max * 100.0);
    }

    /// Stage 1c: 3D RoPE — verify that enabling RoPE produces a finite
    /// output that materially diverges from the no-RoPE baseline (RoPE
    /// rotates per-token Q/K by their grid position; with non-zero
    /// position grids the attention weights must change).
    #[test]
    #[ignore = "wan22-ti2v Stage 1c — needs Wan 2.2 TI2V-5B shards mounted"]
    fn wan22_ti2v_forward_rope_smoke() {
        let Some(dir) = wan22_ti2v_dir() else {
            eprintln!("Wan 2.2 TI2V-5B dir not found"); return;
        };
        let m = Wan22Ti2v::load_with_layers(&dir, 2).expect("load");
        let freqs = RopeFreqs3D::new();

        // Verify the 3-axis split adds up to HEAD_DIM.
        assert_eq!(T_PAIRS + H_PAIRS + W_PAIRS, HEAD_DIM / 2,
                   "axis pairs must sum to head_dim/2; got {}+{}+{} != {}",
                   T_PAIRS, H_PAIRS, W_PAIRS, HEAD_DIM / 2);

        let (t, h, w) = (1, 4, 4);
        let latent: Vec<f32> = (0..IN_DIM * t * h * w)
            .map(|i| ((i as f32) * 0.013).sin() * 0.1).collect();
        let t_text = 8;
        let t5_out: Vec<f32> = (0..t_text * T5_DIM)
            .map(|i| ((i as f32) * 0.011).cos() * 0.05).collect();

        let t0 = std::time::Instant::now();
        let no_rope = m.forward(&latent, t, h, w, &t5_out, t_text, 0.5, 1e-6);
        let dt_no = t0.elapsed();
        let t0 = std::time::Instant::now();
        let with_rope = m.forward_with_rope(&latent, t, h, w, &t5_out, t_text, 0.5, 1e-6, &freqs);
        let dt_yes = t0.elapsed();

        assert_eq!(no_rope.len(), with_rope.len());
        assert!(with_rope.iter().all(|v| v.is_finite()));
        let max_abs = no_rope.iter().zip(with_rope.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let ref_max = no_rope.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("wan22-ti2v RoPE smoke: no-rope {dt_no:?}  with-rope {dt_yes:?}  max_diff={max_abs:.4} (ref max_abs={ref_max:.4})");
        // RoPE must change the attention output when positions vary.
        assert!(max_abs > 1e-3, "RoPE should diverge from no-RoPE; got {max_abs}");
    }
}
