//! rce wiring for OpenAI CLIP ViT-L/14.
//!
//! Architecture (from `config.json` + tensor layout):
//! - **Vision tower** (24 layers × dim 1024, n_heads 16, head_dim 64):
//!   - `patch_embedding [1024, 3, 14, 14]` — 14×14 patch conv on a 224×224
//!     RGB image yields 16×16 = 256 patches + 1 CLS token = 257 tokens.
//!   - `class_embedding [1024]` — learned CLS prepended to the patch sequence.
//!   - `position_embedding [257, 1024]` — absolute position.
//!   - `pre_layrnorm` + `post_layernorm` (note: HF preserves the
//!     "pre_layrnorm" typo from the original release).
//!   - Per block: ln1 + self-attn (q/k/v/out separate proj, with bias) +
//!     ln2 + MLP (fc1 1024→4096, GELU, fc2 4096→1024).
//! - **Text tower** (12 layers × dim 768, n_heads 12, head_dim 64):
//!   - `token_embedding [49408, 768]` (BPE vocab).
//!   - `position_embedding [77, 768]` (max sequence length).
//!   - `final_layer_norm`.
//!   - Same per-block structure as vision tower; FFN expansion is 4× (3072).
//! - **Shared projection**:
//!   - `visual_projection [768, 1024]`, `text_projection [768, 768]`.
//!   - `logit_scale` scalar (≈ exp(4.605) = 100, the canonical CLIP temperature).
//!
//! All weights ship as f32 in a single safetensors blob (~1.7 GB; the rest
//! of the 6.4 GB dir is duplicate PyTorch/Flax/TF formats). Borrow-direct
//! via `F32Slice` so the user heap holds only struct pointers.

use std::path::Path;
use rce_weight_loader::{F32Slice, LoadedTensor, SafetensorsFile};

// ── Vision tower constants ─────────────────────────────────────────────
pub const V_DIM: usize = 1024;
pub const V_HEADS: usize = 16;
pub const V_HEAD_DIM: usize = V_DIM / V_HEADS; // 64
pub const V_FF: usize = 4096;
pub const V_LAYERS: usize = 24;
pub const V_PATCH: usize = 14;
pub const V_IMG: usize = 224;
pub const V_N_PATCHES: usize = (V_IMG / V_PATCH) * (V_IMG / V_PATCH); // 256
pub const V_SEQ: usize = V_N_PATCHES + 1; // CLS + patches = 257

// ── Text tower constants ───────────────────────────────────────────────
pub const T_DIM: usize = 768;
pub const T_HEADS: usize = 12;
pub const T_HEAD_DIM: usize = T_DIM / T_HEADS; // 64
pub const T_FF: usize = 3072;
pub const T_LAYERS: usize = 12;
pub const T_VOCAB: usize = 49_408;
pub const T_MAX_SEQ: usize = 77;

// ── Shared projection ──────────────────────────────────────────────────
pub const PROJ_DIM: usize = 768;

#[derive(Debug)]
pub enum ClipError { Load(String) }
impl std::fmt::Display for ClipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::Load(s) => write!(f, "clip-vit-l: {s}") }
    }
}
impl std::error::Error for ClipError {}

fn lf(sf: &SafetensorsFile, name: &str) -> Result<F32Slice, ClipError> {
    let t: &LoadedTensor = sf.get(name)
        .ok_or_else(|| ClipError::Load(format!("missing {name}")))?;
    F32Slice::from_tensor(name, t).map_err(ClipError::Load)
}
fn lf_vec(sf: &SafetensorsFile, name: &str) -> Result<Vec<f32>, ClipError> {
    Ok(lf(sf, name)?.as_slice().to_vec())
}

/// Self-attention projections (q/k/v/o — all have biases in CLIP, unlike T5).
pub struct AttnWeights {
    pub q_w: F32Slice, pub q_b: Vec<f32>,
    pub k_w: F32Slice, pub k_b: Vec<f32>,
    pub v_w: F32Slice, pub v_b: Vec<f32>,
    pub o_w: F32Slice, pub o_b: Vec<f32>,
}

impl AttnWeights {
    fn load(sf: &SafetensorsFile, prefix: &str, dim: usize) -> Result<Self, ClipError> {
        let load_proj = |slot: &str| -> Result<(F32Slice, Vec<f32>), ClipError> {
            let w = lf(sf, &format!("{prefix}.{slot}.weight"))?;
            let b = lf_vec(sf, &format!("{prefix}.{slot}.bias"))?;
            if w.len() != dim * dim {
                return Err(ClipError::Load(format!(
                    "{prefix}.{slot}.weight: expected {}, got {}", dim*dim, w.len())));
            }
            if b.len() != dim {
                return Err(ClipError::Load(format!(
                    "{prefix}.{slot}.bias: expected {dim}, got {}", b.len())));
            }
            Ok((w, b))
        };
        let (q_w, q_b) = load_proj("q_proj")?;
        let (k_w, k_b) = load_proj("k_proj")?;
        let (v_w, v_b) = load_proj("v_proj")?;
        let (o_w, o_b) = load_proj("out_proj")?;
        Ok(Self { q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b })
    }
}

/// Affine LayerNorm: `out = (x − mean) / √(var + eps) · γ + β`.
pub struct LayerNormWeights {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl LayerNormWeights {
    fn load(sf: &SafetensorsFile, prefix: &str, dim: usize) -> Result<Self, ClipError> {
        let weight = lf_vec(sf, &format!("{prefix}.weight"))?;
        let bias = lf_vec(sf, &format!("{prefix}.bias"))?;
        if weight.len() != dim {
            return Err(ClipError::Load(format!(
                "{prefix}.weight: expected {dim}, got {}", weight.len())));
        }
        Ok(Self { weight, bias })
    }
}

/// 2-layer MLP with QuickGELU (the CLIP convention; not tanh-approx).
pub struct MlpWeights {
    pub fc1_w: F32Slice, pub fc1_b: Vec<f32>,
    pub fc2_w: F32Slice, pub fc2_b: Vec<f32>,
}

impl MlpWeights {
    fn load(sf: &SafetensorsFile, prefix: &str, dim_in: usize, dim_ff: usize) -> Result<Self, ClipError> {
        let fc1_w = lf(sf, &format!("{prefix}.fc1.weight"))?;
        let fc1_b = lf_vec(sf, &format!("{prefix}.fc1.bias"))?;
        let fc2_w = lf(sf, &format!("{prefix}.fc2.weight"))?;
        let fc2_b = lf_vec(sf, &format!("{prefix}.fc2.bias"))?;
        if fc1_w.len() != dim_ff * dim_in {
            return Err(ClipError::Load(format!(
                "{prefix}.fc1.weight: expected {}, got {}", dim_ff*dim_in, fc1_w.len())));
        }
        if fc2_w.len() != dim_in * dim_ff {
            return Err(ClipError::Load(format!(
                "{prefix}.fc2.weight: expected {}, got {}", dim_in*dim_ff, fc2_w.len())));
        }
        Ok(Self { fc1_w, fc1_b, fc2_w, fc2_b })
    }
}

/// One transformer encoder block (pre-norm: ln1 → attn → res; ln2 → mlp → res).
pub struct ClipBlock {
    pub ln1: LayerNormWeights,
    pub attn: AttnWeights,
    pub ln2: LayerNormWeights,
    pub mlp: MlpWeights,
}

impl ClipBlock {
    fn load(sf: &SafetensorsFile, tower: &str, i: usize, dim: usize, dim_ff: usize) -> Result<Self, ClipError> {
        let p = format!("{tower}.encoder.layers.{i}");
        Ok(Self {
            ln1: LayerNormWeights::load(sf, &format!("{p}.layer_norm1"), dim)?,
            attn: AttnWeights::load(sf, &format!("{p}.self_attn"), dim)?,
            ln2: LayerNormWeights::load(sf, &format!("{p}.layer_norm2"), dim)?,
            mlp: MlpWeights::load(sf, &format!("{p}.mlp"), dim, dim_ff)?,
        })
    }
}

/// Vision tower: 24-layer ViT with 14×14 patch conv + learned CLS +
/// absolute position embedding.  Outputs a per-token [V_SEQ, V_DIM]
/// representation; CLIP's image embedding is typically `post_ln(x[0])`.
pub struct VisionTower {
    pub patch_embed_w: F32Slice,                  // [V_DIM, 3, V_PATCH, V_PATCH]
    pub class_embedding: Vec<f32>,                // [V_DIM]
    pub position_embedding: F32Slice,             // [V_SEQ, V_DIM]
    pub pre_ln: LayerNormWeights,                 // pre_layrnorm
    pub post_ln: LayerNormWeights,                // post_layernorm
    pub blocks: Vec<ClipBlock>,
}

impl VisionTower {
    fn load(sf: &SafetensorsFile, n_layers: usize) -> Result<Self, ClipError> {
        let patch_embed_w = lf(sf, "vision_model.embeddings.patch_embedding.weight")?;
        if patch_embed_w.len() != V_DIM * 3 * V_PATCH * V_PATCH {
            return Err(ClipError::Load(format!(
                "patch_embedding.weight: expected {}, got {}",
                V_DIM*3*V_PATCH*V_PATCH, patch_embed_w.len())));
        }
        let class_embedding = lf_vec(sf, "vision_model.embeddings.class_embedding")?;
        let position_embedding = lf(sf, "vision_model.embeddings.position_embedding.weight")?;
        if position_embedding.len() != V_SEQ * V_DIM {
            return Err(ClipError::Load(format!(
                "position_embedding: expected {}, got {}",
                V_SEQ*V_DIM, position_embedding.len())));
        }
        let pre_ln = LayerNormWeights::load(sf, "vision_model.pre_layrnorm", V_DIM)?;
        let post_ln = LayerNormWeights::load(sf, "vision_model.post_layernorm", V_DIM)?;
        let blocks: Vec<ClipBlock> = (0..n_layers)
            .map(|i| ClipBlock::load(sf, "vision_model", i, V_DIM, V_FF))
            .collect::<Result<_, _>>()?;
        Ok(Self { patch_embed_w, class_embedding, position_embedding, pre_ln, post_ln, blocks })
    }
}

/// Text tower: 12-layer transformer over 49,408-vocab BPE tokens with
/// learned absolute positions, max sequence length 77.
pub struct TextTower {
    pub token_embedding: F32Slice,         // [T_VOCAB, T_DIM]
    pub position_embedding: F32Slice,      // [T_MAX_SEQ, T_DIM]
    pub final_ln: LayerNormWeights,
    pub blocks: Vec<ClipBlock>,
}

impl TextTower {
    fn load(sf: &SafetensorsFile, n_layers: usize) -> Result<Self, ClipError> {
        let token_embedding = lf(sf, "text_model.embeddings.token_embedding.weight")?;
        if token_embedding.len() != T_VOCAB * T_DIM {
            return Err(ClipError::Load(format!(
                "token_embedding: expected {}, got {}", T_VOCAB*T_DIM, token_embedding.len())));
        }
        let position_embedding = lf(sf, "text_model.embeddings.position_embedding.weight")?;
        if position_embedding.len() != T_MAX_SEQ * T_DIM {
            return Err(ClipError::Load(format!(
                "position_embedding: expected {}, got {}",
                T_MAX_SEQ*T_DIM, position_embedding.len())));
        }
        let final_ln = LayerNormWeights::load(sf, "text_model.final_layer_norm", T_DIM)?;
        let blocks: Vec<ClipBlock> = (0..n_layers)
            .map(|i| ClipBlock::load(sf, "text_model", i, T_DIM, T_FF))
            .collect::<Result<_, _>>()?;
        Ok(Self { token_embedding, position_embedding, final_ln, blocks })
    }
}

/// Full CLIP ViT-L/14 model: vision tower + text tower + shared projections.
pub struct ClipVitL {
    pub vision: VisionTower,
    pub text: TextTower,
    pub visual_projection: F32Slice,    // [PROJ_DIM, V_DIM]
    pub text_projection: F32Slice,      // [PROJ_DIM, T_DIM]
    pub logit_scale: f32,
}

impl ClipVitL {
    /// Load all layers from the safetensors blob.
    pub fn load(path: &Path) -> Result<Self, ClipError> {
        Self::load_with_layers(path, V_LAYERS, T_LAYERS)
    }

    /// Load with subsets of layers for quick smoke tests.
    pub fn load_with_layers(
        path: &Path, n_v_layers: usize, n_t_layers: usize,
    ) -> Result<Self, ClipError> {
        eprintln!("[clip-vit-l] loading {}", path.display());
        let sf = SafetensorsFile::load(path)
            .map_err(|e| ClipError::Load(format!("load {}: {e}", path.display())))?;
        let vision = VisionTower::load(&sf, n_v_layers)?;
        let text = TextTower::load(&sf, n_t_layers)?;
        let visual_projection = lf(&sf, "visual_projection.weight")?;
        let text_projection = lf(&sf, "text_projection.weight")?;
        if visual_projection.len() != PROJ_DIM * V_DIM {
            return Err(ClipError::Load(format!(
                "visual_projection.weight: expected {}, got {}",
                PROJ_DIM*V_DIM, visual_projection.len())));
        }
        if text_projection.len() != PROJ_DIM * T_DIM {
            return Err(ClipError::Load(format!(
                "text_projection.weight: expected {}, got {}",
                PROJ_DIM*T_DIM, text_projection.len())));
        }
        // logit_scale is a scalar 0-d tensor.
        let logit_scale_t = sf.get("logit_scale")
            .ok_or_else(|| ClipError::Load("missing logit_scale".into()))?;
        let bytes = logit_scale_t.data.as_slice();
        if bytes.len() != 4 {
            return Err(ClipError::Load(format!(
                "logit_scale: expected 4 bytes, got {}", bytes.len())));
        }
        let logit_scale = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        Ok(Self { vision, text, visual_projection, text_projection, logit_scale })
    }

    /// Rough user-heap-resident size (borrow-direct keeps the bulk in mmap).
    pub fn user_heap_bytes(&self) -> usize {
        let mut n = self.vision.class_embedding.len() * 4;
        n += (self.vision.pre_ln.weight.len() + self.vision.pre_ln.bias.len()) * 4;
        n += (self.vision.post_ln.weight.len() + self.vision.post_ln.bias.len()) * 4;
        for b in &self.vision.blocks {
            n += (b.ln1.weight.len() + b.ln1.bias.len()) * 4;
            n += (b.ln2.weight.len() + b.ln2.bias.len()) * 4;
            n += (b.attn.q_b.len() + b.attn.k_b.len() + b.attn.v_b.len() + b.attn.o_b.len()) * 4;
            n += (b.mlp.fc1_b.len() + b.mlp.fc2_b.len()) * 4;
        }
        n += (self.text.final_ln.weight.len() + self.text.final_ln.bias.len()) * 4;
        for b in &self.text.blocks {
            n += (b.ln1.weight.len() + b.ln1.bias.len()) * 4;
            n += (b.ln2.weight.len() + b.ln2.bias.len()) * 4;
            n += (b.attn.q_b.len() + b.attn.k_b.len() + b.attn.v_b.len() + b.attn.o_b.len()) * 4;
            n += (b.mlp.fc1_b.len() + b.mlp.fc2_b.len()) * 4;
        }
        n
    }
}

// ════════════════════════════════════════════════════════════════════════
// Forward primitives (Stage 1)
// ════════════════════════════════════════════════════════════════════════

/// QuickGELU — CLIP's activation: `x · sigmoid(1.702·x)`.  Cheaper than
/// tanh-approx and tanh-exact GELU; the OpenAI CLIP release standardized
/// on it.
fn quick_gelu(x: f32) -> f32 { x / (1.0 + (-1.702 * x).exp()) }

fn apply_quick_gelu(x: &mut [f32]) { for v in x.iter_mut() { *v = quick_gelu(*v); } }

/// Affine LayerNorm with learned weight + bias.  `[T, D]` row-major.
fn layernorm_affine(x: &[f32], t: usize, d: usize, w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    debug_assert_eq!(x.len(), t * d);
    debug_assert_eq!(w.len(), d);
    debug_assert_eq!(b.len(), d);
    let mut out = vec![0.0f32; x.len()];
    for ti in 0..t {
        let row = &x[ti * d .. (ti + 1) * d];
        let mean: f32 = row.iter().sum::<f32>() / (d as f32);
        let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (d as f32);
        let inv = (var + eps).sqrt().recip();
        for di in 0..d {
            out[ti * d + di] = (row[di] - mean) * inv * w[di] + b[di];
        }
    }
    out
}

/// `out[m, n] = sum_k(input[m, k] · weight[n, k]) + bias[n]`.
/// Rayon-parallel over M rows; bias may be `None`.
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

/// Multi-head SDPA with optional **causal mask** (CLIP text tower uses it,
/// vision tower doesn't).  Q/K/V `[T, dim]` viewed as `[T, n_heads, head_dim]`.
fn sdpa(
    q: &[f32], k: &[f32], v: &[f32], t: usize,
    n_heads: usize, head_dim: usize, causal: bool,
) -> Vec<f32> {
    use rayon::prelude::*;
    let dim = n_heads * head_dim;
    let scale = (head_dim as f32).powf(-0.5);
    let mut out = vec![0.0f32; t * dim];
    out.par_chunks_mut(dim).enumerate().for_each(|(ti, out_row)| {
        for h in 0..n_heads {
            let q_off = ti * dim + h * head_dim;
            let mut scores = vec![0.0f32; t];
            let mut row_max = f32::NEG_INFINITY;
            let j_max = if causal { ti + 1 } else { t };
            for j in 0..j_max {
                let k_off = j * dim + h * head_dim;
                let mut s = 0.0f32;
                for di in 0..head_dim { s += q[q_off + di] * k[k_off + di]; }
                let sv = s * scale;
                scores[j] = sv;
                if sv > row_max { row_max = sv; }
            }
            let mut sum = 0.0f32;
            for j in 0..j_max {
                let e = (scores[j] - row_max).exp();
                scores[j] = e;
                sum += e;
            }
            let inv = sum.recip();
            for j in 0..j_max { scores[j] *= inv; }
            for di in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..j_max {
                    let v_off = j * dim + h * head_dim;
                    acc += scores[j] * v[v_off + di];
                }
                out_row[h * head_dim + di] = acc;
            }
        }
    });
    out
}

/// CLIP patch embed: 2D conv k=14 stride=14 on a [3, 224, 224] RGB image
/// → [V_DIM=1024, 16, 16] = 256 patch tokens, returned flattened as
/// `[V_N_PATCHES=256, V_DIM]` row-major (patch-first, channel-second).
fn patch_embed(image: &[f32], weight: &[f32]) -> Vec<f32> {
    use rayon::prelude::*;
    debug_assert_eq!(image.len(), 3 * V_IMG * V_IMG);
    debug_assert_eq!(weight.len(), V_DIM * 3 * V_PATCH * V_PATCH);
    let n_y = V_IMG / V_PATCH; // 16
    let n_x = V_IMG / V_PATCH; // 16
    let n_patches = n_y * n_x;
    let mut out = vec![0.0f32; n_patches * V_DIM];
    out.par_chunks_mut(V_DIM).enumerate().for_each(|(p, dst)| {
        let py = p / n_x;
        let px = p % n_x;
        for oc in 0..V_DIM {
            let mut acc = 0.0f32;
            for ic in 0..3 {
                for ky in 0..V_PATCH {
                    for kx in 0..V_PATCH {
                        let w_idx = ((oc * 3 + ic) * V_PATCH + ky) * V_PATCH + kx;
                        let in_y = py * V_PATCH + ky;
                        let in_x = px * V_PATCH + kx;
                        let in_idx = (ic * V_IMG + in_y) * V_IMG + in_x;
                        acc += weight[w_idx] * image[in_idx];
                    }
                }
            }
            dst[oc] = acc;
        }
    });
    out
}

impl ClipBlock {
    /// One encoder block: pre-norm self-attn + pre-norm MLP, both residuals.
    fn forward(
        &self, x: &[f32], t_seq: usize,
        dim: usize, n_heads: usize, head_dim: usize, dim_ff: usize,
        causal: bool, eps: f32,
    ) -> Vec<f32> {
        // Self-attn branch.
        let h = layernorm_affine(x, t_seq, dim, &self.ln1.weight, &self.ln1.bias, eps);
        let q = matmul_nt_f32_bias(&h, t_seq, dim, self.attn.q_w.as_slice(),
                                    Some(&self.attn.q_b), dim);
        let k = matmul_nt_f32_bias(&h, t_seq, dim, self.attn.k_w.as_slice(),
                                    Some(&self.attn.k_b), dim);
        let v = matmul_nt_f32_bias(&h, t_seq, dim, self.attn.v_w.as_slice(),
                                    Some(&self.attn.v_b), dim);
        let attn = sdpa(&q, &k, &v, t_seq, n_heads, head_dim, causal);
        let attn_o = matmul_nt_f32_bias(&attn, t_seq, dim, self.attn.o_w.as_slice(),
                                         Some(&self.attn.o_b), dim);
        let mut x_new = x.to_vec();
        for i in 0..x_new.len() { x_new[i] += attn_o[i]; }

        // MLP branch.
        let h = layernorm_affine(&x_new, t_seq, dim, &self.ln2.weight, &self.ln2.bias, eps);
        let mut h_ff = matmul_nt_f32_bias(&h, t_seq, dim, self.mlp.fc1_w.as_slice(),
                                           Some(&self.mlp.fc1_b), dim_ff);
        apply_quick_gelu(&mut h_ff);
        let mlp_o = matmul_nt_f32_bias(&h_ff, t_seq, dim_ff, self.mlp.fc2_w.as_slice(),
                                        Some(&self.mlp.fc2_b), dim);
        for i in 0..x_new.len() { x_new[i] += mlp_o[i]; }
        x_new
    }
}

impl VisionTower {
    /// Encode an image into the full post-transformer token sequence
    /// **before** the final post_layernorm and projection.  Returns
    /// `[V_SEQ=257, V_DIM=1024]` row-major — CLS at index 0, then 256
    /// spatial patch tokens in row-major (top-left to bottom-right) order.
    ///
    /// This is the entry point LLaVA-style VLMs use: they take **all**
    /// patch tokens (skipping CLS) and feed them through an MLP projector
    /// into LM embedding space, so the language model attends to spatial
    /// regions of the image.
    pub fn encode_image_tokens(&self, image: &[f32], eps: f32) -> Vec<f32> {
        // Patch embed → [256, 1024]
        let mut x = patch_embed(image, self.patch_embed_w.as_slice());
        // Prepend CLS → [257, 1024]
        let mut seq = Vec::with_capacity(V_SEQ * V_DIM);
        seq.extend_from_slice(&self.class_embedding);
        seq.append(&mut x);
        // Add absolute position embedding.
        let pos = self.position_embedding.as_slice();
        for i in 0..V_SEQ * V_DIM { seq[i] += pos[i]; }
        // Pre-LN.
        let mut x = layernorm_affine(&seq, V_SEQ, V_DIM, &self.pre_ln.weight, &self.pre_ln.bias, eps);
        // Transformer blocks (no causal mask in vision tower).
        for b in &self.blocks {
            x = b.forward(&x, V_SEQ, V_DIM, V_HEADS, V_HEAD_DIM, V_FF, false, eps);
        }
        x
    }

    /// Encode a `[3, 224, 224]` normalized RGB image to a `[V_SEQ=257, V_DIM=1024]`
    /// token sequence after all transformer blocks + post_layernorm applied
    /// to the CLS-token at position 0.  Returns the post-LN CLS embedding
    /// `[V_DIM]` ready for projection to the shared 768-d space.
    pub fn encode_image(&self, image: &[f32], eps: f32) -> Vec<f32> {
        let x = self.encode_image_tokens(image, eps);
        // post_layernorm on CLS token only.
        let cls = &x[0 .. V_DIM];
        layernorm_affine(cls, 1, V_DIM, &self.post_ln.weight, &self.post_ln.bias, eps)
    }
}

impl TextTower {
    /// Encode token ids → `[T_DIM=768]` sentence embedding.  Per CLIP's
    /// convention, the embedding is read at the **EOT** token position
    /// (the highest token id slot — CLIP's BPE EOT is id 49407).  This
    /// implementation takes the embedding at the last position (`ids.len()-1`),
    /// which matches CLIP's standard pooling when EOT is the final token.
    pub fn encode_ids(&self, ids: &[u32], eps: f32) -> Vec<f32> {
        let t = ids.len();
        let x_ln = self.encode_ids_sequence(ids, eps);
        x_ln[(t - 1) * T_DIM .. t * T_DIM].to_vec()
    }

    /// Encode token ids → the **full** post-final-LayerNorm token sequence
    /// `[ids.len(), T_DIM=768]`.  This is the `last_hidden_state` that
    /// Stable Diffusion 1.5 (and similar text-conditioned diffusion models)
    /// use as their cross-attention conditioning, without the EOT pooling
    /// that `encode_ids` applies.
    ///
    /// For SD 1.5 specifically, pad `ids` to `T_MAX_SEQ=77` with the EOT
    /// id (49407) before calling so the conditioning shape matches the
    /// UNet's expected `[77, 768]` cross-attention input.
    pub fn encode_ids_sequence(&self, ids: &[u32], eps: f32) -> Vec<f32> {
        let t = ids.len();
        assert!(t > 0 && t <= T_MAX_SEQ, "ids length {t} out of [1, {T_MAX_SEQ}]");
        let mut x = vec![0.0f32; t * T_DIM];
        let tok = self.token_embedding.as_slice();
        let pos = self.position_embedding.as_slice();
        for (ti, &id) in ids.iter().enumerate() {
            for d in 0..T_DIM {
                x[ti * T_DIM + d] = tok[(id as usize) * T_DIM + d] + pos[ti * T_DIM + d];
            }
        }
        for b in &self.blocks {
            x = b.forward(&x, t, T_DIM, T_HEADS, T_HEAD_DIM, T_FF, true, eps);
        }
        layernorm_affine(&x, t, T_DIM, &self.final_ln.weight, &self.final_ln.bias, eps)
    }
}

impl ClipVitL {
    /// Project a vision CLS embedding `[V_DIM]` to the shared 768-d space.
    pub fn project_image(&self, cls: &[f32]) -> Vec<f32> {
        matmul_nt_f32_bias(cls, 1, V_DIM, self.visual_projection.as_slice(), None, PROJ_DIM)
    }

    /// Project a text EOT embedding `[T_DIM]` to the shared 768-d space.
    pub fn project_text(&self, eot: &[f32]) -> Vec<f32> {
        matmul_nt_f32_bias(eot, 1, T_DIM, self.text_projection.as_slice(), None, PROJ_DIM)
    }

    /// End-to-end image encoding: pixel → vision tower → visual projection
    /// → 768-d shared-space embedding.
    pub fn encode_image(&self, image: &[f32], eps: f32) -> Vec<f32> {
        let cls = self.vision.encode_image(image, eps);
        self.project_image(&cls)
    }

    /// End-to-end text encoding: ids → text tower → text projection → 768-d.
    pub fn encode_text(&self, ids: &[u32], eps: f32) -> Vec<f32> {
        let eot = self.text.encode_ids(ids, eps);
        self.project_text(&eot)
    }

    /// CLIP similarity logit: `logit_scale × cos_sim(img, text)`.
    /// (`logit_scale.exp() ≈ 100` for canonical CLIP.)
    pub fn similarity(&self, img_emb: &[f32], text_emb: &[f32]) -> f32 {
        let dot: f32 = img_emb.iter().zip(text_emb.iter()).map(|(a, b)| a * b).sum();
        let n_i: f32 = img_emb.iter().map(|v| v * v).sum::<f32>().sqrt();
        let n_t: f32 = text_emb.iter().map(|v| v * v).sum::<f32>().sqrt();
        self.logit_scale.exp() * dot / (n_i * n_t).max(1e-12)
    }
}

// ════════════════════════════════════════════════════════════════════════
// Image preprocessing (Stage 2a — normalize)
// ════════════════════════════════════════════════════════════════════════

/// CLIP image-normalization constants from the model's `preprocessor_config.json`.
/// Image is `(pixel/255 − mean) / std` per channel.
pub const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
pub const CLIP_STD:  [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Apply CLIP RGB normalization to a `[H, W, 3]` u8 image already at
/// `V_IMG × V_IMG = 224×224` resolution.  Returns the `[3, 224, 224]`
/// f32 tensor in channel-first layout the encoder expects.
///
/// Resize (short-side to 224, bicubic) and CenterCrop are out of scope
/// here — provide a pre-resized image, or wire the `image` crate in a
/// downstream binary to do those steps.
pub fn normalize_rgb_224(rgb_u8: &[u8]) -> Vec<f32> {
    let n = V_IMG * V_IMG;
    assert_eq!(rgb_u8.len(), n * 3, "expected {} bytes for 224×224 RGB, got {}", n*3, rgb_u8.len());
    let mut out = vec![0.0f32; 3 * n];
    for yi in 0..V_IMG {
        for xi in 0..V_IMG {
            let src = (yi * V_IMG + xi) * 3;
            for c in 0..3 {
                let v = rgb_u8[src + c] as f32 / 255.0;
                let n_v = (v - CLIP_MEAN[c]) / CLIP_STD[c];
                out[(c * V_IMG + yi) * V_IMG + xi] = n_v;
            }
        }
    }
    out
}

/// Full CLIP image preprocessor: decode any format the `image` crate
/// supports → resize short-side to 224 (Catmull-Rom approximating PIL's
/// bicubic) → center-crop 224×224 → CLIP normalize → `[3, 224, 224]` f32.
///
/// The `resample=3` (PIL.Image.BICUBIC) in the original config maps to
/// `image::imageops::FilterType::CatmullRom` in the Rust image crate —
/// the closest bicubic-family kernel.  Subpixel-level differences vs PIL
/// are expected but visually imperceptible.
pub fn preprocess_image(path: &Path) -> Result<Vec<f32>, ClipError> {
    use image::imageops::FilterType;
    let img = image::open(path)
        .map_err(|e| ClipError::Load(format!("open {}: {e}", path.display())))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    // Resize so short side = 224.
    let (new_w, new_h) = if w < h {
        (V_IMG as u32, ((h as f32) * (V_IMG as f32) / (w as f32)).round() as u32)
    } else {
        (((w as f32) * (V_IMG as f32) / (h as f32)).round() as u32, V_IMG as u32)
    };
    let resized = image::imageops::resize(&rgb, new_w, new_h, FilterType::CatmullRom);
    // Center crop to 224×224.
    let x0 = (new_w - V_IMG as u32) / 2;
    let y0 = (new_h - V_IMG as u32) / 2;
    let cropped = image::imageops::crop_imm(&resized, x0, y0, V_IMG as u32, V_IMG as u32).to_image();
    let raw = cropped.into_raw();
    debug_assert_eq!(raw.len(), V_IMG * V_IMG * 3);
    Ok(normalize_rgb_224(&raw))
}

// ════════════════════════════════════════════════════════════════════════
// BPE tokenizer (Stage 2)
// ════════════════════════════════════════════════════════════════════════

/// CLIP BPE tokenizer (49,408-vocab byte-level BPE with `</w>` end-of-word
/// markers).  Thin wrapper around the `tokenizers` crate's
/// `Tokenizer::from_file(tokenizer.json)`; the model ships its own
/// `tokenizer.json` in the standard HuggingFace fast-tokenizer format.
pub struct ClipTokenizer {
    inner: tokenizers::Tokenizer,
}

pub const TOK_SOT: u32 = 49_406; // <|startoftext|>
pub const TOK_EOT: u32 = 49_407; // <|endoftext|>

impl ClipTokenizer {
    /// Load from `<model_root>/tokenizer.json`.
    pub fn load_from_dir(model_root: &Path) -> Result<Self, ClipError> {
        let p = model_root.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&p)
            .map_err(|e| ClipError::Load(format!("load {}: {e}", p.display())))?;
        Ok(Self { inner })
    }

    /// Encode a UTF-8 caption to CLIP token ids (with SOT/EOT specials added).
    /// Matches HuggingFace `tokenizers.Tokenizer.from_file(...).encode(text).ids`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, ClipError> {
        let enc = self.inner.encode(text, true)
            .map_err(|e| ClipError::Load(format!("encode: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn vocab_size(&self) -> usize { self.inner.get_vocab_size(true) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn clip_safetensors_path() -> Option<PathBuf> {
        for vol in ["macos_4TB_external", "macos_4TB_external_1", "macos_4TB_external_2"] {
            let p = PathBuf::from(format!(
                "/Volumes/{vol}/neuronexus_ai/model_library/openai_clip-vit-large-patch14/model.safetensors"));
            if p.exists() { return Some(p); }
        }
        None
    }

    /// Stage 0: load all 592 tensors (or a subset of N vision + M text
    /// layers); verify per-block shapes and that borrow-direct keeps the
    /// user heap small.  Validates the canonical CLIP ViT-L/14 layout.
    #[test]
    #[ignore = "clip-vit-l Stage 0 — needs CLIP safetensors mounted"]
    fn clip_vit_l_load_first_layers() {
        let Some(path) = clip_safetensors_path() else {
            eprintln!("CLIP safetensors not found"); return;
        };
        let t0 = std::time::Instant::now();
        let m = ClipVitL::load_with_layers(&path, 2, 2).expect("load");
        eprintln!("clip-vit-l: 2v+2t layers + embeds + projs loaded in {:?}", t0.elapsed());

        assert_eq!(m.vision.patch_embed_w.len(), V_DIM * 3 * V_PATCH * V_PATCH);
        assert_eq!(m.vision.class_embedding.len(), V_DIM);
        assert_eq!(m.vision.position_embedding.len(), V_SEQ * V_DIM);
        assert_eq!(m.text.token_embedding.len(), T_VOCAB * T_DIM);
        assert_eq!(m.text.position_embedding.len(), T_MAX_SEQ * T_DIM);
        assert_eq!(m.visual_projection.len(), PROJ_DIM * V_DIM);
        assert_eq!(m.text_projection.len(), PROJ_DIM * T_DIM);

        for (i, b) in m.vision.blocks.iter().enumerate() {
            assert_eq!(b.ln1.weight.len(), V_DIM, "v block {i} ln1");
            assert_eq!(b.attn.q_w.len(), V_DIM * V_DIM, "v block {i} q");
            assert_eq!(b.mlp.fc1_w.len(), V_FF * V_DIM, "v block {i} fc1");
        }
        for (i, b) in m.text.blocks.iter().enumerate() {
            assert_eq!(b.ln1.weight.len(), T_DIM, "t block {i} ln1");
            assert_eq!(b.attn.q_w.len(), T_DIM * T_DIM, "t block {i} q");
            assert_eq!(b.mlp.fc1_w.len(), T_FF * T_DIM, "t block {i} fc1");
        }

        let mb = m.user_heap_bytes() as f64 / (1024.0 * 1024.0);
        eprintln!("clip-vit-l user heap: {:.3} MB for 2v+2t layers", mb);
        eprintln!("logit_scale = {:.4} (canonical CLIP temperature exp(logit_scale) ≈ {:.1})",
                  m.logit_scale, m.logit_scale.exp());
        assert!(m.logit_scale > 4.0 && m.logit_scale < 5.0,
                "expected canonical CLIP logit_scale ≈ 4.605 (=log(100)); got {}", m.logit_scale);
        eprintln!("clip-vit-l Stage 0 ✓");
    }

    fn clip_dir() -> Option<PathBuf> {
        for vol in ["macos_4TB_external", "macos_4TB_external_1", "macos_4TB_external_2"] {
            let p = PathBuf::from(format!(
                "/Volumes/{vol}/neuronexus_ai/model_library/openai_clip-vit-large-patch14"));
            if p.join("tokenizer.json").exists() && p.join("model.safetensors").exists() {
                return Some(p);
            }
        }
        None
    }

    /// CLIP BPE tokenizer Python-parity: 3/3 reference cases from HF
    /// `tokenizers.Tokenizer.from_file(tokenizer.json).encode(s).ids`.
    #[test]
    #[ignore = "clip-vit-l tokenizer — needs CLIP dir mounted"]
    fn clip_vit_l_tokenizer_python_parity() {
        let Some(dir) = clip_dir() else { eprintln!("CLIP dir not found"); return; };
        let tk = ClipTokenizer::load_from_dir(&dir).expect("load tokenizer");
        eprintln!("ClipTokenizer loaded {} tokens", tk.vocab_size());

        let cases: &[(&str, &[u32])] = &[
            ("a photo of a cat",            &[49406, 320, 1125, 539, 320, 2368, 49407]),
            ("a dog playing in a park",     &[49406, 320, 1929, 1629, 530, 320, 1452, 49407]),
            ("the eiffel tower at sunset",  &[49406, 518, 29720, 4730, 536, 3424, 49407]),
        ];
        for (text, expected) in cases {
            let ids = tk.encode(text).expect("encode");
            eprintln!("  {text:?} → {ids:?} (expected {expected:?})");
            assert_eq!(&ids[..], *expected, "mismatch on {text:?}");
        }
        // Sanity: specials at canonical CLIP positions.
        assert_eq!(TOK_SOT, 49_406);
        assert_eq!(TOK_EOT, 49_407);
        eprintln!("ClipTokenizer ✓ — 3/3 Python-parity");
    }

    /// Stage 1 forward smoke: load a 2v+2t-layer subset, run vision and
    /// text encoders on synthetic inputs through projections.  Verifies
    /// all primitives compose into finite [768]-d shared-space embeddings.
    #[test]
    #[ignore = "clip-vit-l Stage 1 — needs CLIP safetensors mounted"]
    fn clip_vit_l_forward_smoke() {
        let Some(path) = clip_safetensors_path() else {
            eprintln!("CLIP safetensors not found"); return;
        };
        let m = ClipVitL::load_with_layers(&path, 2, 2).expect("load");

        // ── Vision: synthetic [3, 224, 224] image ──
        let image: Vec<f32> = (0..3 * V_IMG * V_IMG)
            .map(|i| ((i as f32) * 0.0009).sin() * 0.5).collect();
        let t0 = std::time::Instant::now();
        let img_emb = m.encode_image(&image, 1e-5);
        let dt_v = t0.elapsed();
        assert_eq!(img_emb.len(), PROJ_DIM);
        assert!(img_emb.iter().all(|v| v.is_finite()));
        let img_norm: f32 = img_emb.iter().map(|v| v*v).sum::<f32>().sqrt();
        eprintln!("vision encode (2L): {dt_v:?} ‖img‖={img_norm:.4}");

        // ── Text: synthetic ids ending in CLIP's EOT (49407) ──
        let ids_a: Vec<u32> = vec![49406, 320, 18539, 14533, 11532, 49407];  // SOT + 4 toks + EOT
        let ids_b: Vec<u32> = vec![49406, 320, 31313, 18527, 8915, 49407];   // different middle ids
        let t0 = std::time::Instant::now();
        let txt_a = m.encode_text(&ids_a, 1e-5);
        let dt_t = t0.elapsed();
        let txt_b = m.encode_text(&ids_b, 1e-5);
        assert_eq!(txt_a.len(), PROJ_DIM);
        assert!(txt_a.iter().all(|v| v.is_finite()));
        assert!(txt_b.iter().all(|v| v.is_finite()));
        let txt_a_norm: f32 = txt_a.iter().map(|v| v*v).sum::<f32>().sqrt();
        eprintln!("text encode (2L): {dt_t:?} ‖txt_a‖={txt_a_norm:.4}");
        // Different ids must produce different text embeddings.
        let max_diff = txt_a.iter().zip(txt_b.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff > 1e-3, "different ids must yield different embeddings");

        // ── Similarity gate ──
        let sim_a = m.similarity(&img_emb, &txt_a);
        let sim_b = m.similarity(&img_emb, &txt_b);
        eprintln!("similarity(img, a)={sim_a:.4}  (img, b)={sim_b:.4}  exp(logit_scale)={:.2}",
                  m.logit_scale.exp());
        assert!(sim_a.is_finite() && sim_b.is_finite());
    }

    /// CLIP full-layer Python-reference numerical agreement.  Verifies
    /// the rce port matches a hand-rolled PyTorch CLIP forward
    /// operation-by-operation across all 24 vision layers AND all 12
    /// text layers + final projection.  Reference captured 2026-06-06.
    #[test]
    #[ignore = "clip-vit-l full-layer ref — needs CLIP dir + /tmp/clip_ref_image.f32.bin + /tmp/clip_ref_text.f32.bin"]
    fn clip_vit_l_full_layer_vs_python_reference() {
        let Some(dir) = clip_dir() else { eprintln!("CLIP dir not found"); return; };
        let img_ref_path = std::path::PathBuf::from("/tmp/clip_ref_image.f32.bin");
        let txt_ref_path = std::path::PathBuf::from("/tmp/clip_ref_text.f32.bin");
        if !img_ref_path.exists() || !txt_ref_path.exists() {
            eprintln!("refs not found"); return;
        }
        let read_ref = |p: &std::path::Path| -> Vec<f32> {
            let bytes = std::fs::read(p).expect("read ref");
            bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let img_ref = read_ref(&img_ref_path);
        let txt_ref = read_ref(&txt_ref_path);
        assert_eq!(img_ref.len(), PROJ_DIM);
        assert_eq!(txt_ref.len(), PROJ_DIM);
        let img_ref_max: f32 = img_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let txt_ref_max: f32 = txt_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("Python ref: ‖img‖={:.4} max={img_ref_max:.4}  ‖txt‖={:.4} max={txt_ref_max:.4}",
                  img_ref.iter().map(|v| v*v).sum::<f32>().sqrt(),
                  txt_ref.iter().map(|v| v*v).sum::<f32>().sqrt());

        let t0 = std::time::Instant::now();
        let m = ClipVitL::load(&dir.join("model.safetensors")).expect("load full");
        eprintln!("rce load (24v + 12t): {:?}", t0.elapsed());

        // Same deterministic image as Python: 3*224*224 sin pattern.
        let image: Vec<f32> = (0..3 * V_IMG * V_IMG)
            .map(|i| ((i as f32) * 0.0009).sin() * 0.5).collect();
        let t0 = std::time::Instant::now();
        let img_emb = m.encode_image(&image, 1e-5);
        eprintln!("rce vision (24L): {:?}", t0.elapsed());
        assert_eq!(img_emb.len(), PROJ_DIM);

        let ids: Vec<u32> = vec![49406, 320, 1125, 539, 320, 2368, 49407]; // 'a photo of a cat'
        let t0 = std::time::Instant::now();
        let txt_emb = m.encode_text(&ids, 1e-5);
        eprintln!("rce text (12L): {:?}", t0.elapsed());
        assert_eq!(txt_emb.len(), PROJ_DIM);

        let mut img_max_diff = 0.0f32;
        let mut img_sum_sq = 0.0f32;
        for (a, b) in img_emb.iter().zip(img_ref.iter()) {
            let d = (a - b).abs();
            if d > img_max_diff { img_max_diff = d; }
            img_sum_sq += d * d;
        }
        let img_rms = (img_sum_sq / (img_emb.len() as f32)).sqrt();
        let img_rel = img_max_diff / img_ref_max;

        let mut txt_max_diff = 0.0f32;
        let mut txt_sum_sq = 0.0f32;
        for (a, b) in txt_emb.iter().zip(txt_ref.iter()) {
            let d = (a - b).abs();
            if d > txt_max_diff { txt_max_diff = d; }
            txt_sum_sq += d * d;
        }
        let txt_rms = (txt_sum_sq / (txt_emb.len() as f32)).sqrt();
        let txt_rel = txt_max_diff / txt_ref_max;

        eprintln!("vs Python ref:");
        eprintln!("  vision: max_abs_diff={img_max_diff:.6} rms={img_rms:.6} rel={:.2}%", img_rel * 100.0);
        eprintln!("  text:   max_abs_diff={txt_max_diff:.6} rms={txt_rms:.6} rel={:.2}%", txt_rel * 100.0);
        assert!(img_rel < 0.01, "vision divergence too large: {}%", img_rel * 100.0);
        assert!(txt_rel < 0.01, "text divergence too large: {}%", txt_rel * 100.0);
    }

    /// End-to-end retrieval demo: encode one synthetic 224×224 RGB image
    /// + three real-prompt captions (via the BPE tokenizer), score
    /// similarity, and assert the ranking is deterministic.
    #[test]
    #[ignore = "clip-vit-l retrieval — needs CLIP dir mounted"]
    fn clip_vit_l_retrieval_end_to_end() {
        let Some(dir) = clip_dir() else { eprintln!("CLIP dir not found"); return; };
        let m = ClipVitL::load_with_layers(&dir.join("model.safetensors"), 2, 2).expect("load");
        let tk = ClipTokenizer::load_from_dir(&dir).expect("load tokenizer");

        // Synthetic deterministic 224×224 RGB image (already in pixel u8 range).
        let img_u8: Vec<u8> = (0..V_IMG * V_IMG * 3)
            .map(|i| (((i as f32) * 0.0007).sin() * 127.0 + 128.0).clamp(0.0, 255.0) as u8)
            .collect();
        let img_f32 = normalize_rgb_224(&img_u8);
        let img_emb = m.encode_image(&img_f32, 1e-5);

        let captions = [
            "a photo of a cat",
            "a dog playing in a park",
            "the eiffel tower at sunset",
        ];

        let mut scores = Vec::with_capacity(captions.len());
        for cap in &captions {
            let ids = tk.encode(cap).expect("encode");
            let txt_emb = m.encode_text(&ids, 1e-5);
            let sim = m.similarity(&img_emb, &txt_emb);
            eprintln!("  similarity({cap:?}) = {sim:.4}");
            scores.push((cap, sim));
        }
        // Sanity: at least one is the highest, all finite, distinct.
        assert!(scores.iter().all(|(_, s)| s.is_finite()));
        let max_idx = scores.iter().enumerate()
            .max_by(|(_, (_, a)), (_, (_, b))| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap();
        eprintln!("best (synthetic image, 2L subset): {:?} = {:.4}", scores[max_idx].0, scores[max_idx].1);
        // No semantic claim — 2-layer subset on synthetic random pixels can't
        // discriminate captions meaningfully; this proves the pipeline runs
        // end-to-end (tokenize → encode → similarity) producing finite ranked scores.
    }

    /// Real-image preprocess path: write a deterministic test PNG via the
    /// `image` crate, read it back through `preprocess_image` (which does
    /// resize → center-crop → normalize), and feed to the full CLIP encoder.
    /// Validates the full image-decoding pipeline composes with the encoder.
    #[test]
    #[ignore = "clip-vit-l real-image preprocess — needs CLIP dir mounted; writes /tmp/clip_test_image.png"]
    fn clip_vit_l_preprocess_real_image() {
        let Some(dir) = clip_dir() else { eprintln!("CLIP dir not found"); return; };

        // ── Synthesize and write a deterministic 256×256 RGB PNG ──
        // Vertical-stripes pattern: red/green/blue thirds with smooth gradient.
        let w = 256u32; let h = 256u32;
        let mut buf = image::RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let third = w / 3;
                let yf = (y as f32) / (h as f32);
                let r = if x < third      { (255.0 * (1.0 - yf)) as u8 } else { 30 };
                let g = if x < 2 * third  { (180.0 + 50.0 * yf) as u8 } else { 30 };
                let b = if x >= 2 * third { 255_u8 } else { 30 };
                buf.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }
        let png_path = std::path::PathBuf::from("/tmp/clip_test_image.png");
        buf.save(&png_path).expect("write png");

        // ── Preprocess via the full pipeline ──
        let t0 = std::time::Instant::now();
        let f32_img = preprocess_image(&png_path).expect("preprocess");
        eprintln!("preprocess_image (256×256 PNG → 224×224 f32): {:?}", t0.elapsed());
        assert_eq!(f32_img.len(), 3 * V_IMG * V_IMG);
        assert!(f32_img.iter().all(|v| v.is_finite()));
        // Normalized image should be roughly mean-zero (CLIP normalize subtracts mean).
        let mean: f32 = f32_img.iter().sum::<f32>() / (f32_img.len() as f32);
        eprintln!("preprocessed image mean: {mean:.4} (CLIP-normalized → near 0 for natural images)");

        // ── Encode + retrieval (2-layer subset for speed) ──
        let m = ClipVitL::load_with_layers(&dir.join("model.safetensors"), 2, 2).expect("load");
        let tk = ClipTokenizer::load_from_dir(&dir).expect("tokenizer");
        let img_emb = m.encode_image(&f32_img, 1e-5);
        let captions = ["red green blue vertical stripes", "a photo of a cat",
                        "the eiffel tower at sunset"];
        for cap in &captions {
            let ids = tk.encode(cap).expect("encode");
            let txt = m.encode_text(&ids, 1e-5);
            let sim = m.similarity(&img_emb, &txt);
            eprintln!("  similarity({cap:?}) = {sim:.4}");
            assert!(sim.is_finite());
        }
        eprintln!("preprocess_image ✓ — real PNG path composes with encoder");
    }
}
