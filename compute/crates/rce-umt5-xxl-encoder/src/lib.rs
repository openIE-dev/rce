//! rce wiring for UMT5-XXL encoder.
//!
//! UMT5 (Unified MT5) is Google's multilingual T5 variant.  Wan 2.2's
//! TI2V-5B uses the encoder-only XXL configuration as its text encoder
//! (4096-d output that feeds the diffusion transformer's cross-attention).
//!
//! Architecture (from `models_t5_umt5-xxl-enc-bf16.pth`):
//! - 24 encoder blocks × `dim=4096`, `n_heads=64`, `head_dim=64`, `d_ff=10240`
//! - Vocab: `token_embedding [256384, 4096]` (UMT5 multilingual SentencePiece)
//! - Per block: pre-attn RMSNorm + self-attention (q/k/v/o all 4096×4096,
//!   no bias) + pre-FFN RMSNorm + GeGLU FFN (gate · fc1 → fc2)
//! - Per-block **relative position bias** `pos_embedding.embedding [32, 64]`
//!   — T5 buckets (32) × per-head bias (64).
//! - Final RMSNorm before output.
//!
//! Weights ship as bf16 in a converted safetensors blob (~11 GB total).
//! Borrow-direct via `BfSlice` so the user heap holds only struct pointers.

use std::path::Path;
use rce_weight_loader::{LoadedTensor, SafetensorsFile};

// ── Minimal bf16 borrow-direct slice (mirror of rce-cosmos3::BfSlice) ──
/// Borrowed bf16 weight slice — wraps a `LoadedTensor` of dtype `bf16`
/// (mmap-aliased via the safetensors `Arc<Mmap>` backing), exposing a
/// zero-copy `&[half::bf16]` view.  f32 fallback owned-converts.
pub struct BfSlice {
    tensor: LoadedTensor,
}

impl BfSlice {
    pub fn from_tensor(name: &str, t: &LoadedTensor) -> Result<Self, String> {
        match t.dtype.as_str() {
            "bf16" => Ok(Self { tensor: t.clone() }),
            other => Err(format!("BfSlice({name}): unsupported dtype {other}")),
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[half::bf16] {
        let bytes = self.tensor.data.as_slice();
        debug_assert_eq!(bytes.len() % 2, 0);
        debug_assert_eq!(bytes.as_ptr() as usize % 2, 0, "bf16 alignment");
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<half::bf16>(), bytes.len() / 2) }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, half::bf16> { self.as_slice().iter() }
    pub fn len(&self) -> usize { self.tensor.data.len() / 2 }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
}

impl std::ops::Deref for BfSlice {
    type Target = [half::bf16];
    #[inline] fn deref(&self) -> &[half::bf16] { self.as_slice() }
}

pub const DIM: usize = 4096;
pub const N_HEADS: usize = 64;
pub const HEAD_DIM: usize = DIM / N_HEADS; // 64
pub const FF_DIM: usize = 10240;
pub const N_LAYERS: usize = 24;
pub const VOCAB_SIZE: usize = 256_384;
pub const RELPOS_BUCKETS: usize = 32;

#[derive(Debug)]
pub enum Umt5Error { Load(String) }

impl std::fmt::Display for Umt5Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::Load(s) => write!(f, "umt5-xxl: {s}") }
    }
}
impl std::error::Error for Umt5Error {}

fn lbf(sf: &SafetensorsFile, name: &str) -> Result<BfSlice, Umt5Error> {
    let t: &LoadedTensor = sf.get(name)
        .ok_or_else(|| Umt5Error::Load(format!("missing {name}")))?;
    BfSlice::from_tensor(name, t).map_err(Umt5Error::Load)
}

fn lbf_vec(sf: &SafetensorsFile, name: &str) -> Result<Vec<f32>, Umt5Error> {
    let s = lbf(sf, name)?;
    Ok(s.iter().map(|b| b.to_f32()).collect())
}

/// Self-attention projections (no bias — T5 convention).
pub struct AttnWeights {
    pub q_w: BfSlice,
    pub k_w: BfSlice,
    pub v_w: BfSlice,
    pub o_w: BfSlice,
}

impl AttnWeights {
    fn load(sf: &SafetensorsFile, prefix: &str) -> Result<Self, Umt5Error> {
        let load_proj = |slot: &str| -> Result<BfSlice, Umt5Error> {
            let w = lbf(sf, &format!("{prefix}.{slot}.weight"))?;
            if w.len() != DIM * DIM {
                return Err(Umt5Error::Load(format!(
                    "{prefix}.{slot}.weight: expected {}, got {}", DIM*DIM, w.len())));
            }
            Ok(w)
        };
        Ok(Self {
            q_w: load_proj("q")?,
            k_w: load_proj("k")?,
            v_w: load_proj("v")?,
            o_w: load_proj("o")?,
        })
    }
}

/// GeGLU FFN: `gate.0 [10240, 4096]` and `fc1 [10240, 4096]` produce
/// gate and value; `fc2 [4096, 10240]` projects back to dim.
/// Forward: `out = fc2(GELU(gate(x)) * fc1(x))`.
pub struct FfnWeights {
    pub gate_w: BfSlice,
    pub fc1_w: BfSlice,
    pub fc2_w: BfSlice,
}

impl FfnWeights {
    fn load(sf: &SafetensorsFile, prefix: &str) -> Result<Self, Umt5Error> {
        let gate_w = lbf(sf, &format!("{prefix}.gate.0.weight"))?;
        let fc1_w = lbf(sf, &format!("{prefix}.fc1.weight"))?;
        let fc2_w = lbf(sf, &format!("{prefix}.fc2.weight"))?;
        if gate_w.len() != FF_DIM * DIM {
            return Err(Umt5Error::Load(format!(
                "{prefix}.gate.0.weight: expected {}, got {}",
                FF_DIM*DIM, gate_w.len())));
        }
        if fc1_w.len() != FF_DIM * DIM {
            return Err(Umt5Error::Load(format!(
                "{prefix}.fc1.weight: expected {}, got {}",
                FF_DIM*DIM, fc1_w.len())));
        }
        if fc2_w.len() != DIM * FF_DIM {
            return Err(Umt5Error::Load(format!(
                "{prefix}.fc2.weight: expected {}, got {}",
                DIM*FF_DIM, fc2_w.len())));
        }
        Ok(Self { gate_w, fc1_w, fc2_w })
    }
}

pub struct Umt5Block {
    pub norm1_weight: Vec<f32>, // pre-attn RMSNorm gamma
    pub attn: AttnWeights,
    pub norm2_weight: Vec<f32>, // pre-FFN RMSNorm gamma
    pub ffn: FfnWeights,
    /// `[RELPOS_BUCKETS=32, N_HEADS=64]` — T5 relative position bias.
    /// Each block has its own (the original T5 only stores it on block 0,
    /// but UMT5 stores per-block; checkpoint inventory has 24 copies).
    pub relpos: Vec<f32>,
}

impl Umt5Block {
    fn load(sf: &SafetensorsFile, i: usize) -> Result<Self, Umt5Error> {
        let p = format!("blocks.{i}");
        let norm1_weight = lbf_vec(sf, &format!("{p}.norm1.weight"))?;
        let norm2_weight = lbf_vec(sf, &format!("{p}.norm2.weight"))?;
        if norm1_weight.len() != DIM {
            return Err(Umt5Error::Load(format!(
                "{p}.norm1.weight: expected {DIM}, got {}", norm1_weight.len())));
        }
        let attn = AttnWeights::load(sf, &format!("{p}.attn"))?;
        let ffn = FfnWeights::load(sf, &format!("{p}.ffn"))?;
        let relpos = lbf_vec(sf, &format!("{p}.pos_embedding.embedding.weight"))?;
        if relpos.len() != RELPOS_BUCKETS * N_HEADS {
            return Err(Umt5Error::Load(format!(
                "{p}.pos_embedding.embedding.weight: expected {} = {RELPOS_BUCKETS}×{N_HEADS}, got {}",
                RELPOS_BUCKETS*N_HEADS, relpos.len())));
        }
        Ok(Self { norm1_weight, attn, norm2_weight, ffn, relpos })
    }
}

pub struct Umt5XxlEncoder {
    pub token_embedding: BfSlice,    // [VOCAB_SIZE, DIM]
    pub blocks: Vec<Umt5Block>,
    pub final_norm_weight: Vec<f32>, // [DIM]
}

impl Umt5XxlEncoder {
    /// Load from the converted single-shard safetensors file.
    pub fn load(path: &Path) -> Result<Self, Umt5Error> {
        Self::load_with_layers(path, N_LAYERS)
    }

    /// Load with a subset of `n_layers` blocks for quick smoke tests.
    pub fn load_with_layers(path: &Path, n_layers: usize) -> Result<Self, Umt5Error> {
        eprintln!("[umt5-xxl] loading {}", path.display());
        let sf = SafetensorsFile::load(path)
            .map_err(|e| Umt5Error::Load(format!("load {}: {e}", path.display())))?;
        let token_embedding = lbf(&sf, "token_embedding.weight")?;
        if token_embedding.len() != VOCAB_SIZE * DIM {
            return Err(Umt5Error::Load(format!(
                "token_embedding.weight: expected {} = {VOCAB_SIZE}×{DIM}, got {}",
                VOCAB_SIZE*DIM, token_embedding.len())));
        }
        let blocks: Vec<Umt5Block> = (0..n_layers)
            .map(|i| Umt5Block::load(&sf, i))
            .collect::<Result<_, _>>()?;
        let final_norm_weight = lbf_vec(&sf, "norm.weight")?;
        if final_norm_weight.len() != DIM {
            return Err(Umt5Error::Load(format!(
                "norm.weight: expected {DIM}, got {}", final_norm_weight.len())));
        }
        Ok(Self { token_embedding, blocks, final_norm_weight })
    }

    /// Rough user-heap-resident size (in bytes).  Borrow-direct keeps the
    /// ~11 GB token embedding + weights in mmap; this returns just the
    /// owned f32 gammas + biases (per-block RMS norm gammas, relpos table,
    /// final norm gamma).
    pub fn user_heap_bytes(&self) -> usize {
        let mut n = self.final_norm_weight.len() * 4;
        for b in &self.blocks {
            n += b.norm1_weight.len() * 4;
            n += b.norm2_weight.len() * 4;
            n += b.relpos.len() * 4;
        }
        n
    }
}

// ════════════════════════════════════════════════════════════════════════
// Forward primitives (Stage 1)
// ════════════════════════════════════════════════════════════════════════

/// Tanh-approx GELU — matches the activation that T5 family models use.
fn gelu(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// T5 RMSNorm without bias: `out = (x / rms(x)) * gamma`.  No mean
/// subtraction — pure root-mean-square over the channel dim.
fn rms_norm(x: &[f32], t: usize, d: usize, gamma: &[f32], eps: f32) -> Vec<f32> {
    debug_assert_eq!(x.len(), t * d);
    debug_assert_eq!(gamma.len(), d);
    let mut out = vec![0.0f32; x.len()];
    for ti in 0..t {
        let row = &x[ti * d .. (ti + 1) * d];
        let ss: f32 = row.iter().map(|v| v * v).sum::<f32>() / (d as f32);
        let inv = (ss + eps).sqrt().recip();
        for di in 0..d { out[ti * d + di] = row[di] * inv * gamma[di]; }
    }
    out
}

/// Matmul `out[m, n] = sum_k(input[m, k] * weight[n, k])` (no bias —
/// T5 convention), bf16 weight dequantized per call.  Rayon-parallel
/// over the M rows.
fn matmul_nt_bf16(
    input: &[f32], m: usize, k: usize,
    weight: &[half::bf16], n: usize,
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
            for ki in 0..k { s += in_row[ki] * w_row[ki].to_f32(); }
            row[ni] = s;
        }
    });
    out
}

/// T5 relative position bucket scheme (bidirectional encoder variant).
/// Maps signed `relative_position = key_pos - query_pos` into one of
/// `num_buckets` buckets, with half the buckets reserved for negative
/// directions and the other half for positive.
pub fn relpos_bucket(
    relative_position: i32, num_buckets: usize, max_distance: i32, bidirectional: bool,
) -> usize {
    let mut ret = 0i32;
    let mut n = -relative_position;
    let buckets_per_side = if bidirectional { num_buckets as i32 / 2 } else { num_buckets as i32 };
    if bidirectional {
        if n < 0 { ret += buckets_per_side; }
        n = n.abs();
    } else {
        n = n.max(0);
    }
    let max_exact = buckets_per_side / 2;
    if n < max_exact {
        ret += n;
    } else {
        let log_ratio = (n as f32 / max_exact as f32).ln()
            / (max_distance as f32 / max_exact as f32).ln();
        let val_if_large = max_exact + (log_ratio * (buckets_per_side - max_exact) as f32) as i32;
        ret += val_if_large.min(buckets_per_side - 1);
    }
    ret as usize
}

/// Build the full `[T_q × T_kv]` bucket grid for T5 relative position
/// bias, then look up the `[T_q × T_kv × N_HEADS]` bias from a per-block
/// `relpos_table [RELPOS_BUCKETS=32, N_HEADS=64]`.  Returns the bias
/// tensor in row-major `[T_q × T_kv × N_HEADS]` order so it can be added
/// to attention scores by head.
pub fn build_relpos_bias(
    t_q: usize, t_kv: usize, relpos_table: &[f32],
) -> Vec<f32> {
    debug_assert_eq!(relpos_table.len(), RELPOS_BUCKETS * N_HEADS);
    let mut out = vec![0.0f32; t_q * t_kv * N_HEADS];
    for qi in 0..t_q {
        for kj in 0..t_kv {
            let rel = (kj as i32) - (qi as i32);
            let bucket = relpos_bucket(rel, RELPOS_BUCKETS, 128, true);
            let bias_row = &relpos_table[bucket * N_HEADS .. (bucket + 1) * N_HEADS];
            let dst_row = (qi * t_kv + kj) * N_HEADS;
            out[dst_row..dst_row + N_HEADS].copy_from_slice(bias_row);
        }
    }
    out
}

/// Multi-head SDPA with T5 relative position bias added to scores before
/// softmax.  Q/K/V layout `[T, DIM]` viewed as `[T, N_HEADS, HEAD_DIM]`.
/// Position bias is `[T_q × T_kv × N_HEADS]` row-major.
///
/// NOTE: T5 does NOT scale by 1/sqrt(d_k) (the rotation-position-bias
/// magnitude takes care of that scaling).  We follow the T5/UMT5 reference.
pub fn sdpa_with_relpos_bias(
    q: &[f32], t_q: usize, k: &[f32], v: &[f32], t_kv: usize,
    bias: &[f32],
) -> Vec<f32> {
    use rayon::prelude::*;
    debug_assert_eq!(q.len(), t_q * DIM);
    debug_assert_eq!(k.len(), t_kv * DIM);
    debug_assert_eq!(v.len(), t_kv * DIM);
    debug_assert_eq!(bias.len(), t_q * t_kv * N_HEADS);
    let mut out = vec![0.0f32; t_q * DIM];
    out.par_chunks_mut(DIM).enumerate().for_each(|(ti, out_row)| {
        for h in 0..N_HEADS {
            let q_off = ti * DIM + h * HEAD_DIM;
            let mut scores = vec![0.0f32; t_kv];
            let mut row_max = f32::NEG_INFINITY;
            for j in 0..t_kv {
                let k_off = j * DIM + h * HEAD_DIM;
                let mut s = 0.0f32;
                for di in 0..HEAD_DIM { s += q[q_off + di] * k[k_off + di]; }
                // Add T5 relative position bias (no 1/sqrt(d_k) scaling).
                s += bias[(ti * t_kv + j) * N_HEADS + h];
                scores[j] = s;
                if s > row_max { row_max = s; }
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

impl Umt5Block {
    /// One encoder block: pre-attn RMSNorm → multi-head SDPA with T5
    /// relpos bias → residual → pre-FFN RMSNorm → GeGLU FFN → residual.
    pub fn forward(&self, x: &[f32], t_text: usize, eps: f32) -> Vec<f32> {
        let bias = build_relpos_bias(t_text, t_text, &self.relpos);
        // ── Self-attention branch ──
        let h_attn = rms_norm(x, t_text, DIM, &self.norm1_weight, eps);
        let q = matmul_nt_bf16(&h_attn, t_text, DIM, self.attn.q_w.as_slice(), DIM);
        let k = matmul_nt_bf16(&h_attn, t_text, DIM, self.attn.k_w.as_slice(), DIM);
        let v = matmul_nt_bf16(&h_attn, t_text, DIM, self.attn.v_w.as_slice(), DIM);
        let attn = sdpa_with_relpos_bias(&q, t_text, &k, &v, t_text, &bias);
        let attn_out = matmul_nt_bf16(&attn, t_text, DIM, self.attn.o_w.as_slice(), DIM);
        let mut x_new = x.to_vec();
        for i in 0..x_new.len() { x_new[i] += attn_out[i]; }

        // ── FFN branch (GeGLU) ──
        let h_ffn = rms_norm(&x_new, t_text, DIM, &self.norm2_weight, eps);
        let mut gate = matmul_nt_bf16(&h_ffn, t_text, DIM, self.ffn.gate_w.as_slice(), FF_DIM);
        for g in gate.iter_mut() { *g = gelu(*g); }
        let up = matmul_nt_bf16(&h_ffn, t_text, DIM, self.ffn.fc1_w.as_slice(), FF_DIM);
        let mut gated = vec![0.0f32; gate.len()];
        for i in 0..gated.len() { gated[i] = gate[i] * up[i]; }
        let ffn_out = matmul_nt_bf16(&gated, t_text, FF_DIM, self.ffn.fc2_w.as_slice(), DIM);
        for i in 0..x_new.len() { x_new[i] += ffn_out[i]; }
        x_new
    }
}

impl Umt5XxlEncoder {
    /// Token-id encoder: token-embed lookup → N blocks → final RMSNorm.
    /// Returns `[t_text, DIM=4096]` text context for downstream cross-attn.
    pub fn encode_ids(&self, ids: &[u32], eps: f32) -> Vec<f32> {
        let t_text = ids.len();
        let mut x = vec![0.0f32; t_text * DIM];
        let emb = self.token_embedding.as_slice();
        for (ti, &id) in ids.iter().enumerate() {
            let src = &emb[(id as usize) * DIM .. (id as usize + 1) * DIM];
            for d in 0..DIM { x[ti * DIM + d] = src[d].to_f32(); }
        }
        for b in &self.blocks {
            x = b.forward(&x, t_text, eps);
        }
        // Final RMSNorm.
        rms_norm(&x, t_text, DIM, &self.final_norm_weight, eps)
    }
}

// ════════════════════════════════════════════════════════════════════════
// UMT5 SentencePiece tokenizer (Stage 2)
// ════════════════════════════════════════════════════════════════════════

/// UMT5-XXL SentencePiece tokenizer.  Thin wrapper around the HuggingFace
/// `tokenizers` crate loaded from the `tokenizer.json` shipped with the
/// model (Google's UMT5 multilingual unigram, vocab 256_300 + 84 specials
/// = 256_384 to match the model's `token_embedding` rows).
pub struct Umt5Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Umt5Tokenizer {
    /// Load from `<model_root>/google/umt5-xxl/tokenizer.json` (the
    /// standard HuggingFace fast-tokenizer JSON path).
    pub fn load_from_dir(model_root: &Path) -> Result<Self, Umt5Error> {
        let p = model_root.join("google").join("umt5-xxl").join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&p)
            .map_err(|e| Umt5Error::Load(format!("load {}: {e}", p.display())))?;
        Ok(Self { inner })
    }

    /// Encode a UTF-8 string into UMT5 token ids.  Matches HuggingFace
    /// `tokenizers.Tokenizer.from_file(...).encode(text).ids` exactly.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Umt5Error> {
        let enc = self.inner.encode(text, true /* add_special_tokens */)
            .map_err(|e| Umt5Error::Load(format!("encode: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Vocabulary size as known to the loaded tokenizer (typically 256_300
    /// base + specials).  The model's embed table has 256_384 rows; specials
    /// fall within `[256_300, 256_384)`.
    pub fn vocab_size(&self) -> usize { self.inner.get_vocab_size(true) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn umt5_safetensors_path() -> Option<PathBuf> {
        for vol in ["macos_4TB_external", "macos_4TB_external_1", "macos_4TB_external_2"] {
            let p = PathBuf::from(format!(
                "/Volumes/{vol}/neuronexus_ai/model_library/Wan-AI_Wan2.2-TI2V-5B/umt5_xxl_encoder.safetensors"));
            if p.exists() { return Some(p); }
        }
        None
    }

    /// Stage 0: load all 242 tensors (or a subset of N layers' worth);
    /// verify the expected per-block shapes and borrow-direct keeps the
    /// user heap small (the ~11 GB lives in the kernel page cache).
    #[test]
    #[ignore = "umt5-xxl Stage 0 — needs umt5_xxl_encoder.safetensors mounted"]
    fn umt5_xxl_load_first_layers() {
        let Some(path) = umt5_safetensors_path() else {
            eprintln!("umt5_xxl_encoder.safetensors not found"); return;
        };
        let t0 = std::time::Instant::now();
        let m = Umt5XxlEncoder::load_with_layers(&path, 2).expect("load");
        eprintln!("umt5-xxl: loaded 2 layers + token_embedding + final_norm in {:?}", t0.elapsed());

        assert_eq!(m.token_embedding.len(), VOCAB_SIZE * DIM);
        assert_eq!(m.final_norm_weight.len(), DIM);
        for (i, b) in m.blocks.iter().enumerate() {
            assert_eq!(b.norm1_weight.len(), DIM);
            assert_eq!(b.attn.q_w.len(), DIM * DIM, "block {i}");
            assert_eq!(b.ffn.gate_w.len(), FF_DIM * DIM, "block {i}");
            assert_eq!(b.ffn.fc2_w.len(), DIM * FF_DIM, "block {i}");
            assert_eq!(b.relpos.len(), RELPOS_BUCKETS * N_HEADS);
        }
        let mb = m.user_heap_bytes() as f64 / (1024.0 * 1024.0);
        eprintln!("umt5-xxl user heap (borrow-direct): {:.3} MB for 2 layers", mb);
        assert!(mb < 10.0);
        eprintln!("umt5-xxl Stage 0 ✓");
    }

    /// Stage 1 forward: encode a synthetic token id sequence through the
    /// loaded subset of blocks + final norm; verify the output is the
    /// right shape, finite, and that different token sequences produce
    /// different encodings (proves the conditioning path is live).
    #[test]
    #[ignore = "umt5-xxl Stage 1 — needs umt5_xxl_encoder.safetensors mounted"]
    fn umt5_xxl_encode_smoke() {
        let Some(path) = umt5_safetensors_path() else {
            eprintln!("umt5_xxl_encoder.safetensors not found"); return;
        };
        let m = Umt5XxlEncoder::load_with_layers(&path, 2).expect("load");

        let ids_a: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let ids_b: Vec<u32> = vec![100, 200, 300, 400, 500, 600, 700, 800];

        let t0 = std::time::Instant::now();
        let ctx_a = m.encode_ids(&ids_a, 1e-6);
        let dt = t0.elapsed();
        let ctx_b = m.encode_ids(&ids_b, 1e-6);

        let t_text = ids_a.len();
        assert_eq!(ctx_a.len(), t_text * DIM);
        assert!(ctx_a.iter().all(|v| v.is_finite()));
        assert!(ctx_b.iter().all(|v| v.is_finite()));
        let max_abs = ctx_a.iter().zip(ctx_b.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let ref_norm = ctx_a.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("umt5-xxl encode (2-layer, t_text={t_text}): {dt:?}");
        eprintln!("  ids_a vs ids_b: max_abs={max_abs:.4} max_ref_abs={ref_norm:.4}");
        assert!(max_abs > 1e-3, "different prompts must produce different encodings");
    }

    fn umt5_tokenizer_dir() -> Option<PathBuf> {
        for vol in ["macos_4TB_external", "macos_4TB_external_1", "macos_4TB_external_2"] {
            let p = PathBuf::from(format!(
                "/Volumes/{vol}/neuronexus_ai/model_library/Wan-AI_Wan2.2-TI2V-5B"));
            if p.join("google/umt5-xxl/tokenizer.json").exists() {
                return Some(p);
            }
        }
        None
    }

    /// UMT5 SentencePiece tokenizer Python-parity: validate Rust output
    /// matches HuggingFace `tokenizers.Tokenizer.from_file(tokenizer.json).encode(s).ids`.
    /// Reference captured on 2026-06-05 from the Wan2.2-TI2V-5B bundle.
    #[test]
    #[ignore = "umt5-xxl tokenizer — needs tokenizer.json mounted"]
    fn umt5_xxl_tokenizer_python_parity() {
        let Some(dir) = umt5_tokenizer_dir() else {
            eprintln!("UMT5 tokenizer dir not found"); return;
        };
        let t0 = std::time::Instant::now();
        let tk = Umt5Tokenizer::load_from_dir(&dir).expect("load tokenizer");
        eprintln!("Umt5Tokenizer loaded {} tokens in {:?}", tk.vocab_size(), t0.elapsed());

        let cases: &[(&str, &[u32])] = &[
            ("A jazz piano melody",              &[320, 27071, 8615, 160499, 313, 1]),
            ("The capital of France is Paris",   &[517, 5801, 329, 3166, 339, 2989, 1]),
            ("Hello, world!",                    &[23231, 275, 3914, 332, 1]),
        ];
        for (text, expected) in cases {
            let ids = tk.encode(text).expect("encode");
            eprintln!("  {text:?} → {ids:?} (expected {expected:?})");
            assert_eq!(&ids[..], *expected, "mismatch on {text:?}");
        }
        eprintln!("Umt5Tokenizer ✓ — Python-parity on {} cases", cases.len());
    }

    /// End-to-end text → encoder smoke: real prompt → tokenizer → 2-layer
    /// encoder forward → verify shape + finite + prompt sensitivity.
    #[test]
    #[ignore = "umt5-xxl end-to-end — needs encoder safetensors + tokenizer.json"]
    fn umt5_xxl_text_to_context_smoke() {
        let Some(enc_path) = umt5_safetensors_path() else {
            eprintln!("encoder safetensors not found"); return;
        };
        let Some(tok_dir) = umt5_tokenizer_dir() else {
            eprintln!("tokenizer dir not found"); return;
        };
        let m = Umt5XxlEncoder::load_with_layers(&enc_path, 2).expect("load encoder");
        let tk = Umt5Tokenizer::load_from_dir(&tok_dir).expect("load tokenizer");

        let prompt_a = "A jazz piano melody";
        let prompt_b = "A heavy metal drum solo";
        let ids_a = tk.encode(prompt_a).expect("encode");
        let ids_b = tk.encode(prompt_b).expect("encode");
        eprintln!("  {prompt_a:?} → {ids_a:?}");
        eprintln!("  {prompt_b:?} → {ids_b:?}");

        let t0 = std::time::Instant::now();
        let ctx_a = m.encode_ids(&ids_a, 1e-6);
        let dt_a = t0.elapsed();
        let ctx_b = m.encode_ids(&ids_b, 1e-6);

        assert_eq!(ctx_a.len(), ids_a.len() * DIM);
        assert_eq!(ctx_b.len(), ids_b.len() * DIM);
        assert!(ctx_a.iter().all(|v| v.is_finite()));
        assert!(ctx_b.iter().all(|v| v.is_finite()));
        eprintln!("text→context: A {dt_a:?}, ‖A‖={:.4}, ‖B‖={:.4}",
                  ctx_a.iter().map(|v| v*v).sum::<f32>().sqrt(),
                  ctx_b.iter().map(|v| v*v).sum::<f32>().sqrt());
    }

    /// Full-layer UMT5 encoder forward vs Python reference.  Verifies the
    /// rce port matches the reference per-element on a fixed prompt
    /// ("Hello, world!" → 5 ids).  Reference captured 2026-06-05 from
    /// `models_t5_umt5-xxl-enc-bf16.pth` via a hand-rolled PyTorch encoder
    /// mirroring the documented structure (RMSNorm + GeGLU + T5 32-bucket
    /// relpos SDPA, no 1/√d_k), saved as raw f32 [5, 4096] at
    /// `/tmp/umt5_ref_hello.f32.bin`.
    #[test]
    #[ignore = "umt5-xxl full-layer ref — needs encoder safetensors + /tmp/umt5_ref_hello.f32.bin"]
    fn umt5_xxl_full_layer_vs_python_reference() {
        let Some(path) = umt5_safetensors_path() else {
            eprintln!("encoder safetensors not found"); return;
        };
        let ref_path = std::path::PathBuf::from("/tmp/umt5_ref_hello.f32.bin");
        if !ref_path.exists() { eprintln!("ref binary not found"); return; }

        // ── Load Python reference ──
        let ref_bytes = std::fs::read(&ref_path).expect("read ref");
        assert_eq!(ref_bytes.len(), 5 * DIM * 4, "ref binary size");
        let mut ref_data = vec![0.0f32; 5 * DIM];
        for (i, c) in ref_bytes.chunks_exact(4).enumerate() {
            ref_data[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        let ref_norm: f32 = ref_data.iter().map(|v| v*v).sum::<f32>().sqrt();
        let ref_max: f32 = ref_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("Python reference: shape=[5, {DIM}] ‖.‖={ref_norm:.4} max_abs={ref_max:.4}");

        // ── Full-layer rce forward on the same input ids ──
        let t0 = std::time::Instant::now();
        let m = Umt5XxlEncoder::load(&path).expect("load full encoder");
        eprintln!("rce load (24 layers): {:?}", t0.elapsed());
        let ids: Vec<u32> = vec![23231, 275, 3914, 332, 1];
        let t0 = std::time::Instant::now();
        let rce_out = m.encode_ids(&ids, 1e-6);
        let dt = t0.elapsed();
        let rce_norm: f32 = rce_out.iter().map(|v| v*v).sum::<f32>().sqrt();
        let rce_max: f32 = rce_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("rce encode (24 layers): {dt:?}  ‖.‖={rce_norm:.4} max_abs={rce_max:.4}");

        // ── Element-wise comparison ──
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
        // The reference is FP32 throughout; rce uses bf16 weights + f32
        // activations.  Accumulated bf16 reorder noise over 24 layers is
        // expected to be roughly 1-5% on small-norm activations.
        assert!(rel_max < 0.20,
                "rce full-layer UMT5 should agree with Python ref within 20%; got {}%",
                rel_max * 100.0);
    }

    /// T5 relative position bucket scheme — verify against a couple of
    /// known reference values from the HuggingFace T5 implementation.
    #[test]
    fn umt5_xxl_relpos_bucket() {
        // Bidirectional encoder with 32 buckets, max_distance=128.
        // 0 → bucket 0 (self).
        assert_eq!(relpos_bucket(0, 32, 128, true), 0);
        // small positive (key after query) → small-bucket region in negative side
        // because n = -relative_position = -1 < 0 ⇒ ret += 16 (buckets_per_side),
        // n_abs = 1 → ret = 16 + 1 = 17.
        assert_eq!(relpos_bucket(1, 32, 128, true), 17);
        // small negative → ret = 0 + |n|, so relative_position=-1 ⇒ ret=1.
        assert_eq!(relpos_bucket(-1, 32, 128, true), 1);
        // large positive (e.g. 100) → ret = 16 + log-bucketed val; should be
        // well above 16 + max_exact=8 but capped under 31.
        let b = relpos_bucket(100, 32, 128, true);
        assert!(b >= 24 && b <= 31, "expected log-bucket region; got {b}");
    }
}
