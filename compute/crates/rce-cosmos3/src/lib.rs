//! NVIDIA Cosmos 3 Nano on rce — Phase 1: text reasoner (Qwen3-VL-8B family).
//!
//! Loads the reasoner-path tensors from the 7-shard `transformer/diffusion_pytorch_model-*.safetensors`
//! pack and runs a text-only forward to produce next-token logits.  Validation: golden top-1
//! match against a PyTorch reference (`tests/reference.py`).  Skipped: the `*_moe_gen` generator
//! pathway, vision encoder, sound tokenizer, Wan 2.2 VAE.

use rce_weight_loader::{LoadedTensor, SafetensorsFile, TensorData};
use std::sync::Arc;

/// Borrow-direct BF16 weight handle.
///
/// For bf16 tensors (the Cosmos 3 hot path), `BfSlice` wraps the original
/// `LoadedTensor`, which holds an `Arc<Mmap>` clone (cheap refcount) and
/// keeps the safetensors mmap alive for as long as this handle exists.
/// The deref returns a slice that points directly into that mmap — zero
/// copy, zero user-heap allocation for weights.
///
/// For f32 tensors (rare), we materialize a one-time bf16 conversion into
/// an `Arc<[u8]>`-backed `Owned` tensor.
///
/// Deref to `[bf16]` means `&self.q_w` (a `&BfSlice`) coerces to `&[bf16]`
/// at call sites of `matmul_nt_bf16` with no source change.
pub(crate) struct BfSlice {
    tensor: LoadedTensor,
}

impl BfSlice {
    pub(crate) fn from_tensor(name: &str, t: &LoadedTensor) -> Result<Self, CosmosError> {
        match t.dtype.as_str() {
            "bf16" => Ok(Self { tensor: t.clone() }),
            "f32" => {
                let bytes = t.data.as_slice();
                if bytes.len() % 4 != 0 {
                    return Err(CosmosError::Load(format!("{name}: f32 byte-len {} not /4", bytes.len())));
                }
                let mut bf_bytes = Vec::with_capacity(bytes.len() / 2);
                for c in bytes.chunks_exact(4) {
                    let bits = half::bf16::from_f32(f32::from_le_bytes([c[0],c[1],c[2],c[3]])).to_bits();
                    let r = bits.to_le_bytes();
                    bf_bytes.push(r[0]); bf_bytes.push(r[1]);
                }
                let arc: Arc<[u8]> = bf_bytes.into();
                let mut shape = t.shape.clone();
                if shape.is_empty() { shape.push(arc.len() / 2); }
                Ok(Self { tensor: LoadedTensor {
                    name: t.name.clone(),
                    dtype: "bf16".to_string(),
                    shape,
                    data: TensorData::Owned(arc),
                }})
            }
            other => Err(CosmosError::Load(format!("BfSlice({name}): unsupported dtype {other}"))),
        }
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[half::bf16] {
        let bytes = self.tensor.data.as_slice();
        debug_assert_eq!(bytes.len() % 2, 0);
        debug_assert_eq!(bytes.as_ptr() as usize % 2, 0, "bf16 alignment");
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<half::bf16>(), bytes.len() / 2) }
    }

    pub(crate) fn len(&self) -> usize { self.tensor.data.len() / 2 }
}

impl std::ops::Deref for BfSlice {
    type Target = [half::bf16];
    #[inline] fn deref(&self) -> &[half::bf16] { self.as_slice() }
}
use std::collections::HashMap;
use std::path::Path;

pub mod vision;
pub mod image_preproc;

/// Phase 4 diffusion generator scaffold (Cosmos3OmniTransformer + Wan 2.2 VAE
/// + AVAE audio + action token I/O).  Gated behind `--features phase4` —
/// adds ~700 LoC of scaffolding and a ~3 GB synthetic-weights test.
#[cfg(feature = "phase4")]
pub mod phase4;

/// Phase 4 v2 — matches the real NVIDIA Cosmos 3 Nano architecture
/// discovered in Stage 6 (joint MMDiT-style reasoner + generator streams
/// with separate Q/K/V, qk-norm, SwiGLU MLP, additive `add_*_proj` join).
/// Gated behind `--features phase4` alongside the original scaffold.
#[cfg(feature = "phase4")]
pub mod phase4_v2;

pub const HIDDEN: usize = 4096;
pub const N_LAYERS: usize = 36;
pub const N_HEADS: usize = 32;
pub const N_KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const FF: usize = 12288;
pub const VOCAB: usize = 151_936;
pub const RMS_EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 5_000_000.0;
pub const MROPE_SECTION: [usize; 3] = [24, 20, 20];   // sum = 64 = head_dim/2

#[derive(Debug)]
pub enum CosmosError { Load(String) }
impl std::fmt::Display for CosmosError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::Load(m) => write!(f, "cosmos3 load: {m}") }
    }
}
impl std::error::Error for CosmosError {}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0; let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() { if x > bv { bv = x; best = i; } }
    best
}

/// Copy one row of the BF16 embedding table into an f32 destination slice
/// of length `HIDDEN`, converting per element.
#[inline]
fn embed_lookup(embed: &[half::bf16], id: usize, dst: &mut [f32]) {
    let src = &embed[id * HIDDEN .. (id + 1) * HIDDEN];
    for i in 0..HIDDEN { dst[i] = src[i].to_f32(); }
}

// ── multi-shard loader ────────────────────────────────────────────────────────
pub struct ShardedSafetensors {
    shards: Vec<SafetensorsFile>,
    index: HashMap<String, usize>,
}
impl ShardedSafetensors {
    pub fn from_index(json_path: &Path) -> Result<Self, CosmosError> {
        let raw = std::fs::read_to_string(json_path)
            .map_err(|e| CosmosError::Load(format!("read index: {e}")))?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| CosmosError::Load(format!("parse index: {e}")))?;
        let wm = v.get("weight_map").cloned().unwrap_or(v);
        let wm = wm.as_object().ok_or_else(|| CosmosError::Load("weight_map not object".into()))?;
        let parent = json_path.parent().unwrap_or(Path::new(""));
        let mut path_to_idx: HashMap<String, usize> = HashMap::new();
        let mut shards: Vec<SafetensorsFile> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::with_capacity(wm.len());
        for (name, file_v) in wm {
            let file = file_v.as_str().ok_or_else(|| CosmosError::Load(format!("entry {name} not string")))?;
            let idx = if let Some(&i) = path_to_idx.get(file) { i } else {
                let abs = parent.join(file);
                eprintln!("[cosmos3] loading shard {}", abs.display());
                let sf = SafetensorsFile::load(&abs).map_err(|e| CosmosError::Load(format!("load {file}: {e}")))?;
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
    /// Wrap a single-file safetensors as a one-shard `ShardedSafetensors`,
    /// indexing every tensor name to shard 0.  Used by Wan 2.2 VAE / audio
    /// codec / vision_encoder which ship as single-file safetensors.
    pub fn from_single_file(path: &Path) -> Result<Self, CosmosError> {
        eprintln!("[cosmos3] loading single shard {}", path.display());
        let sf = SafetensorsFile::load(path)
            .map_err(|e| CosmosError::Load(format!("load {}: {e}", path.display())))?;
        let mut index: HashMap<String, usize> = HashMap::new();
        for t in sf.tensors() { index.insert(t.name.clone(), 0); }
        Ok(Self { shards: vec![sf], index })
    }
}

fn tf32(t: &LoadedTensor) -> Vec<f32> {
    let bytes = t.data.as_slice();
    match t.dtype.as_str() {
        "f32" => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect(),
        "f16" => bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0],c[1]]).to_f32()).collect(),
        "bf16" => bytes.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0],c[1]]).to_f32()).collect(),
        other => panic!("cosmos3: unsupported dtype {other}"),
    }
}
fn ld(ss: &ShardedSafetensors, name: &str) -> Result<Vec<f32>, CosmosError> {
    let t = ss.get(name).ok_or_else(|| CosmosError::Load(format!("missing {name}")))?;
    Ok(tf32(t))
}

/// Borrow-direct BF16 loader.  Returns a `BfSlice` that holds an `Arc<Mmap>`
/// clone (zero-copy) for the common bf16 case, or a one-time owned vector for
/// stray f32 fallbacks.  Cosmos 3 ships everything as bf16 so the hot path is
/// pure mmap aliasing — the model's 16 GB of weights live in the kernel page
/// cache and the user heap holds only struct pointers.
fn ld_bf16(ss: &ShardedSafetensors, name: &str) -> Result<BfSlice, CosmosError> {
    let t = ss.get(name).ok_or_else(|| CosmosError::Load(format!("missing {name}")))?;
    BfSlice::from_tensor(name, t)
}

// ── ops ──────────────────────────────────────────────────────────────────────
fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

fn rmsnorm(x: &[f32], hidden: usize, t: usize, gamma: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for ti in 0..t {
        let row = &x[ti*hidden..(ti+1)*hidden];
        let ss: f32 = row.iter().map(|v| v*v).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (ss + eps).sqrt();
        for h in 0..hidden { out[ti*hidden + h] = row[h] * inv * gamma[h]; }
    }
    out
}

/// BF16-weight variant of `matmul_nt`.  Activations stay f32; weights are read as
/// bf16 from DRAM (2 bytes/element instead of 4).  Halves the heap footprint of
/// the 8-B parameter model (16 GB instead of 32 GB).
///
/// On aarch64 (M-series), the inner dot product uses a hand-NEON loop that
/// converts 8 bf16 → 8 f32 per iteration via `shll #16` (bf16 *is* the upper
/// half of an f32), then does two FMLAs.  Without this, the auto-vectorizer
/// fails to issue NEON and falls back to scalar — costing more than the
/// bandwidth savings.
#[inline(always)]
pub(crate) fn bf16_f32_dot(xrow: &[f32], wr: &[half::bf16]) -> f32 {
    debug_assert_eq!(xrow.len(), wr.len());
    let n = xrow.len();
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut j = 0usize;
        while j + 8 <= n {
            // 8 bf16 weights = 16 bytes
            let wu = vld1q_u16(wr.as_ptr().add(j) as *const u16);
            // bf16 → f32: shift left by 16 to place bits in the upper half of f32.
            let lo_u32 = vshll_n_u16::<16>(vget_low_u16(wu));
            let hi_u32 = vshll_high_n_u16::<16>(wu);
            let w_lo = vreinterpretq_f32_u32(lo_u32);
            let w_hi = vreinterpretq_f32_u32(hi_u32);
            let x_lo = vld1q_f32(xrow.as_ptr().add(j));
            let x_hi = vld1q_f32(xrow.as_ptr().add(j + 4));
            acc0 = vfmaq_f32(acc0, x_lo, w_lo);
            acc1 = vfmaq_f32(acc1, x_hi, w_hi);
            j += 8;
        }
        let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        while j < n {
            sum += xrow[j] * wr[j].to_f32();
            j += 1;
        }
        sum
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum = 0.0f32;
        for j in 0..n { sum += xrow[j] * wr[j].to_f32(); }
        sum
    }
}

pub(crate) fn matmul_nt_bf16(x: &[f32], t: usize, din: usize, w: &[half::bf16], dout: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let mut out = vec![0.0f32; t * dout];
    let target_chunks = (rayon::current_num_threads() * 4).max(8);
    let total = t * dout;
    let chunk = ((total + target_chunks - 1) / target_chunks).max(64);
    out.par_chunks_mut(chunk).enumerate().for_each(|(ck_idx, out_chunk)| {
        let base = ck_idx * chunk;
        for (i, slot) in out_chunk.iter_mut().enumerate() {
            let global = base + i;
            let ti = global / dout;
            let o = global % dout;
            let xrow = &x[ti*din..(ti+1)*din];
            let wr   = &w[o*din..(o+1)*din];
            *slot = bf16_f32_dot(xrow, wr);
        }
    });
    out
}

/// y = x @ W.T      (x: [T,in], W: [out,in], y: [T,out])
/// Parallelized so it stays multi-threaded even at T=1 (decode-step matmuls).
fn matmul_nt(x: &[f32], t: usize, din: usize, w: &[f32], dout: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let mut out = vec![0.0f32; t * dout];
    // Chunk size: aim for ~ #cpus×4 chunks of work so rayon can balance.
    let target_chunks = (rayon::current_num_threads() * 4).max(8);
    let total = t * dout;
    let chunk = ((total + target_chunks - 1) / target_chunks).max(64);
    out.par_chunks_mut(chunk).enumerate().for_each(|(ck_idx, out_chunk)| {
        let base = ck_idx * chunk;
        for (i, slot) in out_chunk.iter_mut().enumerate() {
            let global = base + i;
            let ti = global / dout;
            let o = global % dout;
            let xrow = &x[ti*din..(ti+1)*din];
            let wr   = &w[o*din..(o+1)*din];
            let mut acc = 0.0f32;
            for j in 0..din { acc += xrow[j] * wr[j]; }
            *slot = acc;
        }
    });
    out
}

/// Apply mrope to a single head's [T, HEAD_DIM] activation in-place, interleaved variant.
///
/// In Qwen3-VL the head_dim = 128 splits into 64 rotary pairs.  mrope_section [24,20,20]
/// assigns 24 dims to axis 0 (t), 20 to axis 1 (h), 20 to axis 2 (w) — and "interleaved"
/// means the pairs are interleaved across axes by index `i % 3`.  For text-only inference
/// the three axes carry the same time index, so the rotation collapses to standard
/// 1-D RoPE with the same theta — but using interleaved pair indices means the freq
/// at pair `p` uses `freq[idx(p)]` where idx skips ahead inside its assigned section.
///
/// For TEXT-ONLY (single time axis), all axes share position p_t, so the rotation per
/// pair is `rotate_pair(idx_h_within_section(pair))`.  Concretely we precompute a single
/// 1-D freq vector of length head_dim/2.
pub(crate) fn rope_freqs() -> Vec<f32> {
    let half = HEAD_DIM / 2;
    let mut f = vec![0.0f32; half];
    for i in 0..half {
        f[i] = 1.0 / ROPE_THETA.powf((2*i) as f32 / HEAD_DIM as f32);
    }
    f
}

/// Apply RoPE to [T, n_heads, HEAD_DIM] activation laid out as [T, n_heads*HEAD_DIM] (head-fast).
/// PyTorch convention: rotate `(x_a, x_b)` where x_a = first half of head_dim, x_b = second half.
pub(crate) fn apply_rope(x: &mut [f32], t: usize, n_heads: usize, freqs: &[f32], positions: &[i64]) {
    let half = HEAD_DIM / 2;
    debug_assert_eq!(positions.len(), t);
    debug_assert_eq!(freqs.len(), half);
    for ti in 0..t {
        let pos = positions[ti] as f32;
        for h in 0..n_heads {
            let base = ti * n_heads * HEAD_DIM + h * HEAD_DIM;
            for k in 0..half {
                let theta = pos * freqs[k];
                let (s, c) = theta.sin_cos();
                let a = x[base + k];
                let b = x[base + k + half];
                x[base + k]        = a * c - b * s;
                x[base + k + half] = a * s + b * c;
            }
        }
    }
}

// per-head RMSNorm:  norm.weight has shape [HEAD_DIM]; applied independently to each (t, head) slice.
fn rmsnorm_per_head(x: &mut [f32], t: usize, n_heads: usize, gamma: &[f32], eps: f32) {
    debug_assert_eq!(gamma.len(), HEAD_DIM);
    for ti in 0..t {
        for h in 0..n_heads {
            let base = ti * n_heads * HEAD_DIM + h * HEAD_DIM;
            let slice = &mut x[base..base + HEAD_DIM];
            let ss: f32 = slice.iter().map(|v| v*v).sum::<f32>() / HEAD_DIM as f32;
            let inv = 1.0 / (ss + eps).sqrt();
            for d in 0..HEAD_DIM { slice[d] = slice[d] * inv * gamma[d]; }
        }
    }
}

// ── layer ────────────────────────────────────────────────────────────────────
struct Layer {
    input_ln: Vec<f32>,
    // Big projection matrices stay BF16 to halve DRAM bandwidth in the matmul hot path.
    q_w: BfSlice, k_w: BfSlice, v_w: BfSlice, o_w: BfSlice,
    q_norm: Vec<f32>, k_norm: Vec<f32>,
    post_attn_ln: Vec<f32>,
    gate_w: BfSlice, up_w: BfSlice, down_w: BfSlice,
}
impl Layer {
    fn load(ss: &ShardedSafetensors, i: usize) -> Result<Self, CosmosError> {
        let p = format!("layers.{i}");
        Ok(Self {
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
        })
    }
    /// Same as `forward`, but with a persistent KV cache for incremental decode.
    /// `cached_t` is the number of K/V rows already in the cache; `positions_new`
    /// gives the rotary positions for the **new** tokens only.  After the call,
    /// `cache.k`/`cache.v` have `cached_t + t_new` rows each.
    fn forward_step(
        &self,
        x_new: &mut Vec<f32>,
        t_new: usize,
        cached_t: usize,
        freqs: &[f32],
        positions_new: &[i64],
        cache: &mut LayerKvCache,
    ) {
        let x_norm = rmsnorm(x_new, HIDDEN, t_new, &self.input_ln, RMS_EPS);
        let mut q = matmul_nt_bf16(&x_norm, t_new, HIDDEN, &self.q_w, HIDDEN);
        let mut k_new = matmul_nt_bf16(&x_norm, t_new, HIDDEN, &self.k_w, N_KV_HEADS * HEAD_DIM);
        let     v_new = matmul_nt_bf16(&x_norm, t_new, HIDDEN, &self.v_w, N_KV_HEADS * HEAD_DIM);
        rmsnorm_per_head(&mut q,     t_new, N_HEADS,    &self.q_norm, RMS_EPS);
        rmsnorm_per_head(&mut k_new, t_new, N_KV_HEADS, &self.k_norm, RMS_EPS);
        apply_rope(&mut q,     t_new, N_HEADS,    freqs, positions_new);
        apply_rope(&mut k_new, t_new, N_KV_HEADS, freqs, positions_new);
        cache.k.extend_from_slice(&k_new);
        cache.v.extend_from_slice(&v_new);

        // Attention: parallelize over (head, query-row).  Each (h, ti_new) slot
        // is independent — perfect for rayon over `N_HEADS * t_new` chunks.
        use rayon::prelude::*;
        let q_per_kv = N_HEADS / N_KV_HEADS;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut per_head = vec![0.0f32; N_HEADS * t_new * HEAD_DIM];
        per_head.par_chunks_mut(t_new * HEAD_DIM).enumerate().for_each(|(h, head_buf)| {
            let kvh = h / q_per_kv;
            for ti_new in 0..t_new {
                let ti_global = cached_t + ti_new;
                let q_off = ti_new * N_HEADS * HEAD_DIM + h * HEAD_DIM;
                let mut scores = vec![0.0f32; ti_global + 1];
                let mut maxv = f32::NEG_INFINITY;
                for tj in 0..=ti_global {
                    let k_off = tj * N_KV_HEADS * HEAD_DIM + kvh * HEAD_DIM;
                    let mut s = 0.0f32;
                    for d in 0..HEAD_DIM { s += q[q_off + d] * cache.k[k_off + d]; }
                    s *= scale;
                    scores[tj] = s;
                    if s > maxv { maxv = s; }
                }
                let mut sum = 0.0f32;
                for tj in 0..=ti_global { scores[tj] = (scores[tj] - maxv).exp(); sum += scores[tj]; }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for tj in 0..=ti_global { scores[tj] *= inv; }
                for d in 0..HEAD_DIM {
                    let mut acc = 0.0f32;
                    for tj in 0..=ti_global {
                        acc += scores[tj] * cache.v[tj * N_KV_HEADS * HEAD_DIM + kvh * HEAD_DIM + d];
                    }
                    head_buf[ti_new * HEAD_DIM + d] = acc;
                }
            }
        });
        // Restripe per_head [H, T, D] → attn [T, H, D].
        let mut attn = vec![0.0f32; t_new * N_HEADS * HEAD_DIM];
        for h in 0..N_HEADS {
            for ti_new in 0..t_new {
                let src = h * t_new * HEAD_DIM + ti_new * HEAD_DIM;
                let dst = ti_new * N_HEADS * HEAD_DIM + h * HEAD_DIM;
                attn[dst..dst+HEAD_DIM].copy_from_slice(&per_head[src..src+HEAD_DIM]);
            }
        }
        let o = matmul_nt_bf16(&attn, t_new, HIDDEN, &self.o_w, HIDDEN);
        for i in 0..x_new.len() { x_new[i] += o[i]; }

        let x_norm = rmsnorm(x_new, HIDDEN, t_new, &self.post_attn_ln, RMS_EPS);
        let gate = matmul_nt_bf16(&x_norm, t_new, HIDDEN, &self.gate_w, FF);
        let up   = matmul_nt_bf16(&x_norm, t_new, HIDDEN, &self.up_w,   FF);
        let mut act = vec![0.0f32; gate.len()];
        for i in 0..gate.len() { act[i] = silu(gate[i]) * up[i]; }
        let down = matmul_nt_bf16(&act, t_new, FF, &self.down_w, HIDDEN);
        for i in 0..x_new.len() { x_new[i] += down[i]; }
    }

    fn forward(&self, x: &mut Vec<f32>, t: usize, freqs: &[f32], positions: &[i64]) {
        // Pre-attn RMSNorm.
        let x_norm = rmsnorm(x, HIDDEN, t, &self.input_ln, RMS_EPS);

        // Q/K/V projections.
        let mut q = matmul_nt_bf16(&x_norm, t, HIDDEN, &self.q_w, HIDDEN);                      // [T, 32*128]
        let mut k = matmul_nt_bf16(&x_norm, t, HIDDEN, &self.k_w, N_KV_HEADS * HEAD_DIM);        // [T, 8*128]
        let     v = matmul_nt_bf16(&x_norm, t, HIDDEN, &self.v_w, N_KV_HEADS * HEAD_DIM);        // [T, 8*128]

        // qk_norm + rope.
        rmsnorm_per_head(&mut q, t, N_HEADS, &self.q_norm, RMS_EPS);
        rmsnorm_per_head(&mut k, t, N_KV_HEADS, &self.k_norm, RMS_EPS);
        apply_rope(&mut q, t, N_HEADS, freqs, positions);
        apply_rope(&mut k, t, N_KV_HEADS, freqs, positions);

        // GQA SDPA.  Each Q head h uses K/V head h / (N_HEADS / N_KV_HEADS) — group mapping.
        let q_per_kv = N_HEADS / N_KV_HEADS;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut attn = vec![0.0f32; t * N_HEADS * HEAD_DIM];
        for ti in 0..t {
            for h in 0..N_HEADS {
                let kvh = h / q_per_kv;  // GROUP not TILE.
                // Build score row over all positions tj (causal).
                let mut scores = vec![f32::NEG_INFINITY; t];
                let mut maxv = f32::NEG_INFINITY;
                for tj in 0..=ti {
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
                for tj in 0..=ti { scores[tj] = (scores[tj] - maxv).exp(); sum += scores[tj]; }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for tj in 0..=ti { scores[tj] *= inv; }
                for d in 0..HEAD_DIM {
                    let mut acc = 0.0f32;
                    for tj in 0..=ti {
                        acc += scores[tj] * v[tj*N_KV_HEADS*HEAD_DIM + kvh*HEAD_DIM + d];
                    }
                    attn[ti*N_HEADS*HEAD_DIM + h*HEAD_DIM + d] = acc;
                }
            }
        }

        // Output projection + residual.
        let o = matmul_nt_bf16(&attn, t, HIDDEN, &self.o_w, HIDDEN);
        for i in 0..x.len() { x[i] += o[i]; }

        // Pre-MLP norm.
        let x_norm = rmsnorm(x, HIDDEN, t, &self.post_attn_ln, RMS_EPS);

        // SwiGLU MLP.
        let gate = matmul_nt_bf16(&x_norm, t, HIDDEN, &self.gate_w, FF);
        let up   = matmul_nt_bf16(&x_norm, t, HIDDEN, &self.up_w,   FF);
        let mut act = vec![0.0f32; gate.len()];
        for i in 0..gate.len() { act[i] = silu(gate[i]) * up[i]; }
        let down = matmul_nt_bf16(&act, t, FF, &self.down_w, HIDDEN);
        for i in 0..x.len() { x[i] += down[i]; }
    }
}

pub struct Cosmos3Reasoner {
    embed: BfSlice,    // [VOCAB, HIDDEN]
    layers: Vec<Layer>,
    final_norm: Vec<f32>,
    lm_head: BfSlice,  // [VOCAB, HIDDEN]
    freqs: Vec<f32>,
}

/// Per-layer KV cache for incremental decode.
pub struct LayerKvCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

/// Generation state — one KV cache per layer plus the running position counter.
pub struct GenState {
    pub layer_caches: Vec<LayerKvCache>,
    pub cached_t: usize,
}

impl Cosmos3Reasoner {
    pub fn load(root: &Path) -> Result<Arc<Self>, CosmosError> {
        let idx = root.join("model.safetensors.index.json");
        let ss = ShardedSafetensors::from_index(&idx)?;
        eprintln!("[cosmos3] loaded {} shards; resolving tensors…", ss.shards.len());

        let embed = ld_bf16(&ss, "embed_tokens.weight")?;
        let final_norm = ld(&ss, "norm.weight")?;
        let lm_head = ld_bf16(&ss, "lm_head.weight")?;

        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            eprintln!("[cosmos3] layer {i}");
            layers.push(Layer::load(&ss, i)?);
        }

        Ok(Arc::new(Self {
            embed, layers, final_norm, lm_head,
            freqs: rope_freqs(),
        }))
    }

    /// Forward over token ids. Returns logits[VOCAB] for the last position only.
    pub fn forward_last_logits(&self, ids: &[u32]) -> Vec<f32> {
        let t = ids.len();
        let positions: Vec<i64> = (0..t as i64).collect();

        // Embed.
        let mut x = vec![0.0f32; t * HIDDEN];
        for ti in 0..t {
            let id = ids[ti] as usize;
            embed_lookup(&self.embed, id, &mut x[ti*HIDDEN..(ti+1)*HIDDEN]);
        }

        for (li, layer) in self.layers.iter().enumerate() {
            layer.forward(&mut x, t, &self.freqs, &positions);
            if std::env::var("COSMOS3_TRACE").is_ok() && li < 3 {
                let row = &x[(t-1)*HIDDEN..t*HIDDEN];
                let m: f32 = row.iter().sum::<f32>() / HIDDEN as f32;
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                eprintln!("[cosmos3] layer {li:2} last-row mean={m:.4} max={max:.4}");
            }
        }

        let x_norm = rmsnorm(&x, HIDDEN, t, &self.final_norm, RMS_EPS);
        // Only need last position's logits.
        let last = &x_norm[(t-1)*HIDDEN..t*HIDDEN];
        let logits = matmul_nt_bf16(last, 1, HIDDEN, &self.lm_head, VOCAB);
        logits
    }

    /// Multimodal forward: takes pre-built `inputs_embeds` (text embeddings with image
    /// positions already replaced by merged vision tokens) and per-token visual mask +
    /// deepstack features.  Mirrors HF `Qwen3VLTextModel.forward(inputs_embeds=…,
    /// visual_pos_masks=…, deepstack_visual_embeds=…)`.
    ///
    /// `visual_pos_mask[t] = true` ⇔ token `t` is a vision-feature slot.
    /// `deepstack[i]` is a flat `[n_visual × HIDDEN]` tensor that gets added to the LLM
    /// hidden state at the masked positions, AFTER layer `i` (for `i ∈ 0..deepstack.len()`).
    ///
    /// Returns the full `[seq_len × HIDDEN]` final hidden state (post final norm).
    pub fn multimodal_forward(
        &self,
        inputs_embeds: &[f32],
        visual_pos_mask: &[bool],
        deepstack: &[Vec<f32>],
    ) -> Vec<f32> {
        let t = visual_pos_mask.len();
        assert_eq!(inputs_embeds.len(), t * HIDDEN);
        let n_visual = visual_pos_mask.iter().filter(|b| **b).count();
        for (i, ds) in deepstack.iter().enumerate() {
            assert_eq!(ds.len(), n_visual * HIDDEN,
                       "deepstack[{i}] len {} expected {n_visual}×{HIDDEN}", ds.len());
        }
        let positions: Vec<i64> = (0..t as i64).collect();
        let mut x = inputs_embeds.to_vec();

        for (li, layer) in self.layers.iter().enumerate() {
            layer.forward(&mut x, t, &self.freqs, &positions);
            if let Some(ds) = deepstack.get(li) {
                // Add ds[v, :] to x[t_pos, :] for each visual position in order.
                let mut v_idx = 0usize;
                for t_pos in 0..t {
                    if visual_pos_mask[t_pos] {
                        let off_x = t_pos * HIDDEN;
                        let off_d = v_idx * HIDDEN;
                        for h in 0..HIDDEN { x[off_x + h] += ds[off_d + h]; }
                        v_idx += 1;
                    }
                }
            }
        }
        rmsnorm(&x, HIDDEN, t, &self.final_norm, RMS_EPS)
    }

    /// Fresh generation state — `N_LAYERS` empty caches, position counter at 0.
    pub fn new_gen_state(&self) -> GenState {
        GenState {
            layer_caches: (0..N_LAYERS)
                .map(|_| LayerKvCache { k: Vec::new(), v: Vec::new() })
                .collect(),
            cached_t: 0,
        }
    }

    /// Incremental forward over `t_new` already-embedded tokens.
    /// `inputs_embeds_new` is `[t_new × HIDDEN]`.
    /// `deepstack_new` is applied at masked positions of the new tokens for the
    /// first `deepstack_new.len()` layers (used during multimodal prefill;
    /// `None` for plain text decode).
    /// `visual_pos_mask_new` matches the layout of the **new** tokens only.
    /// Returns the post-final-norm hidden state for the new tokens: `[t_new × HIDDEN]`.
    pub fn forward_step(
        &self,
        state: &mut GenState,
        inputs_embeds_new: Vec<f32>,
        visual_pos_mask_new: Option<&[bool]>,
        deepstack_new: Option<&[Vec<f32>]>,
    ) -> Vec<f32> {
        let t_new = inputs_embeds_new.len() / HIDDEN;
        let positions: Vec<i64> =
            (state.cached_t as i64 .. (state.cached_t + t_new) as i64).collect();
        let mut x = inputs_embeds_new;
        for (li, layer) in self.layers.iter().enumerate() {
            layer.forward_step(
                &mut x, t_new, state.cached_t,
                &self.freqs, &positions, &mut state.layer_caches[li],
            );
            if let (Some(mask), Some(ds_list)) = (visual_pos_mask_new, deepstack_new) {
                if let Some(ds) = ds_list.get(li) {
                    let mut v_idx = 0usize;
                    for t_pos in 0..t_new {
                        if mask[t_pos] {
                            let off_x = t_pos * HIDDEN;
                            let off_d = v_idx * HIDDEN;
                            for h in 0..HIDDEN { x[off_x + h] += ds[off_d + h]; }
                            v_idx += 1;
                        }
                    }
                }
            }
        }
        state.cached_t += t_new;
        rmsnorm(&x, HIDDEN, t_new, &self.final_norm, RMS_EPS)
    }

    /// Greedy text generation with a per-token callback.  The closure receives
    /// each newly-decoded token; return `false` to stop early.  EOS tokens stop
    /// without firing the callback.
    pub fn generate_greedy_cached_streaming<F: FnMut(u32) -> bool>(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        eos_ids: &[u32],
        mut on_token: F,
    ) -> Vec<u32> {
        let mut state = self.new_gen_state();
        let t = prompt_ids.len();
        let mut x = vec![0.0f32; t * HIDDEN];
        for ti in 0..t {
            let id = prompt_ids[ti] as usize;
            embed_lookup(&self.embed, id, &mut x[ti*HIDDEN..(ti+1)*HIDDEN]);
        }
        let last = self.forward_step(&mut state, x, None, None);
        let last_row = &last[(t-1)*HIDDEN..t*HIDDEN];
        let mut logits = matmul_nt_bf16(last_row, 1, HIDDEN, &self.lm_head, VOCAB);
        let mut out = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let next = argmax(&logits) as u32;
            if eos_ids.contains(&next) { break; }
            out.push(next);
            if !on_token(next) { break; }
            let id = next as usize;
            let mut x_new = vec![0.0f32; HIDDEN];
            embed_lookup(&self.embed, id, &mut x_new);
            let h = self.forward_step(&mut state, x_new, None, None);
            logits = matmul_nt_bf16(&h, 1, HIDDEN, &self.lm_head, VOCAB);
        }
        out
    }

    /// Streaming multimodal greedy generation.
    pub fn multimodal_generate_greedy_cached_streaming<F: FnMut(u32) -> bool>(
        &self,
        initial_inputs_embeds: Vec<f32>,
        visual_pos_mask: &[bool],
        deepstack: &[Vec<f32>],
        max_new_tokens: usize,
        eos_ids: &[u32],
        mut on_token: F,
    ) -> Vec<u32> {
        let mut state = self.new_gen_state();
        let t = visual_pos_mask.len();
        let last = self.forward_step(
            &mut state, initial_inputs_embeds, Some(visual_pos_mask), Some(deepstack));
        let last_row = &last[(t-1)*HIDDEN..t*HIDDEN];
        let mut logits = matmul_nt_bf16(last_row, 1, HIDDEN, &self.lm_head, VOCAB);
        let mut out = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let next = argmax(&logits) as u32;
            if eos_ids.contains(&next) { break; }
            out.push(next);
            if !on_token(next) { break; }
            let id = next as usize;
            let mut x_new = vec![0.0f32; HIDDEN];
            embed_lookup(&self.embed, id, &mut x_new);
            let h = self.forward_step(&mut state, x_new, None, None);
            logits = matmul_nt_bf16(&h, 1, HIDDEN, &self.lm_head, VOCAB);
        }
        out
    }

    /// Greedy text generation with KV cache.  Runs one big prefill on the prompt
    /// then one decode step per new token.  Stops at `eos_ids` or `max_new_tokens`.
    pub fn generate_greedy_cached(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        eos_ids: &[u32],
    ) -> Vec<u32> {
        let mut state = self.new_gen_state();
        // Prefill: embed all prompt tokens.
        let t = prompt_ids.len();
        let mut x = vec![0.0f32; t * HIDDEN];
        for ti in 0..t {
            let id = prompt_ids[ti] as usize;
            embed_lookup(&self.embed, id, &mut x[ti*HIDDEN..(ti+1)*HIDDEN]);
        }
        let last = self.forward_step(&mut state, x, None, None);
        let last_row = &last[(t-1)*HIDDEN..t*HIDDEN];
        let mut logits = matmul_nt_bf16(last_row, 1, HIDDEN, &self.lm_head, VOCAB);
        let mut out = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let next = argmax(&logits) as u32;
            if eos_ids.contains(&next) { break; }
            out.push(next);
            let id = next as usize;
            let mut x_new = vec![0.0f32; HIDDEN];
            embed_lookup(&self.embed, id, &mut x_new);
            let h = self.forward_step(&mut state, x_new, None, None);
            logits = matmul_nt_bf16(&h, 1, HIDDEN, &self.lm_head, VOCAB);
        }
        out
    }

    /// Forward a sequence of token IDs (already-tokenized text) into an existing
    /// `GenState`, extending the KV cache.  Returns logits for the last position.
    /// This is the building block for cache-reuse across chat turns.
    pub fn forward_ids_into(&self, state: &mut GenState, ids: &[u32]) -> Vec<f32> {
        assert!(!ids.is_empty());
        let t = ids.len();
        let mut x = vec![0.0f32; t * HIDDEN];
        for ti in 0..t {
            embed_lookup(&self.embed, ids[ti] as usize, &mut x[ti*HIDDEN..(ti+1)*HIDDEN]);
        }
        let last = self.forward_step(state, x, None, None);
        let last_row = &last[(t-1)*HIDDEN..t*HIDDEN];
        matmul_nt_bf16(last_row, 1, HIDDEN, &self.lm_head, VOCAB)
    }

    /// Continue greedy generation from an existing `GenState` whose last
    /// forward call returned `logits`.  Each emitted token also enters the KV
    /// cache.  Used for multi-turn chat where the cache survives between turns.
    pub fn continue_greedy_cached_streaming<F: FnMut(u32) -> bool>(
        &self,
        state: &mut GenState,
        initial_logits: Vec<f32>,
        max_new_tokens: usize,
        eos_ids: &[u32],
        mut on_token: F,
    ) -> Vec<u32> {
        let mut logits = initial_logits;
        let mut out = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let next = argmax(&logits) as u32;
            if eos_ids.contains(&next) {
                // Push the EOS into the cache too, so the next turn's prefix
                // sees the proper sentence terminator (e.g. `<|im_end|>`).
                let mut x_eos = vec![0.0f32; HIDDEN];
                embed_lookup(&self.embed, next as usize, &mut x_eos);
                let _ = self.forward_step(state, x_eos, None, None);
                break;
            }
            out.push(next);
            if !on_token(next) { break; }
            let id = next as usize;
            let mut x_new = vec![0.0f32; HIDDEN];
            embed_lookup(&self.embed, id, &mut x_new);
            let h = self.forward_step(state, x_new, None, None);
            logits = matmul_nt_bf16(&h, 1, HIDDEN, &self.lm_head, VOCAB);
        }
        out
    }

    /// Multimodal greedy generation with KV cache: prefill over the
    /// prepared `inputs_embeds` (with vision tokens placed and deepstack ready),
    /// then decode one text token at a time.
    pub fn multimodal_generate_greedy_cached(
        &self,
        initial_inputs_embeds: Vec<f32>,
        visual_pos_mask: &[bool],
        deepstack: &[Vec<f32>],
        max_new_tokens: usize,
        eos_ids: &[u32],
    ) -> Vec<u32> {
        let mut state = self.new_gen_state();
        let t = visual_pos_mask.len();
        let last = self.forward_step(
            &mut state, initial_inputs_embeds, Some(visual_pos_mask), Some(deepstack));
        let last_row = &last[(t-1)*HIDDEN..t*HIDDEN];
        let mut logits = matmul_nt_bf16(last_row, 1, HIDDEN, &self.lm_head, VOCAB);
        let mut out = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let next = argmax(&logits) as u32;
            if eos_ids.contains(&next) { break; }
            out.push(next);
            let id = next as usize;
            let mut x_new = vec![0.0f32; HIDDEN];
            embed_lookup(&self.embed, id, &mut x_new);
            let h = self.forward_step(&mut state, x_new, None, None);
            logits = matmul_nt_bf16(&h, 1, HIDDEN, &self.lm_head, VOCAB);
        }
        out
    }

    /// Greedy text generation.  Runs `forward_last_logits` once per generated token —
    /// no KV cache yet, so cost is O(n²) in the suffix length.  Stops at `eos_token_id`
    /// (151645 = `<|im_end|>`, or 151643 = `<|endoftext|>`) or `max_new_tokens`.
    pub fn generate_greedy(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        eos_ids: &[u32],
    ) -> Vec<u32> {
        let mut ids: Vec<u32> = prompt_ids.to_vec();
        let mut out: Vec<u32> = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let logits = self.forward_last_logits(&ids);
            let next = argmax(&logits) as u32;
            if eos_ids.contains(&next) { break; }
            ids.push(next);
            out.push(next);
        }
        out
    }

    /// Multimodal greedy generation.  Starts from an `inputs_embeds` tensor with
    /// vision tokens already placed at masked positions, runs `multimodal_forward`
    /// to get the next-token logits, embeds the predicted token, appends, and
    /// repeats.  Deepstack is applied each step at the original masked positions.
    /// No KV cache → O(n²) in the suffix length.
    pub fn multimodal_generate_greedy(
        &self,
        initial_inputs_embeds: &[f32],
        visual_pos_mask: &[bool],
        deepstack: &[Vec<f32>],
        max_new_tokens: usize,
        eos_ids: &[u32],
    ) -> Vec<u32> {
        let mut embeds: Vec<f32> = initial_inputs_embeds.to_vec();
        let mut mask: Vec<bool> = visual_pos_mask.to_vec();
        let mut out: Vec<u32> = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let last = self.multimodal_forward(&embeds, &mask, deepstack);
            let t = mask.len();
            let last_row = &last[(t-1)*HIDDEN..t*HIDDEN];
            let logits = matmul_nt_bf16(last_row, 1, HIDDEN, &self.lm_head, VOCAB);
            let next = argmax(&logits) as u32;
            if eos_ids.contains(&next) { break; }
            out.push(next);
            // Append the token embedding.
            let mut row = vec![0.0f32; HIDDEN];
            embed_lookup(&self.embed, next as usize, &mut row);
            embeds.extend_from_slice(&row);
            mask.push(false);
        }
        out
    }

    /// Convenience: build `inputs_embeds` by embedding `input_ids` and overwriting the
    /// `image_token_id` positions with the supplied merged vision tokens.
    /// Returns `(inputs_embeds[seq×HIDDEN], visual_pos_mask[seq])`.
    pub fn build_multimodal_inputs(
        &self,
        input_ids: &[u32],
        image_token_id: u32,
        merged_vision_tokens: &[f32],
    ) -> (Vec<f32>, Vec<bool>) {
        let t = input_ids.len();
        let mut x = vec![0.0f32; t * HIDDEN];
        let mut mask = vec![false; t];
        let mut v_idx = 0usize;
        for ti in 0..t {
            let id = input_ids[ti];
            if id == image_token_id {
                let off = v_idx * HIDDEN;
                x[ti*HIDDEN..(ti+1)*HIDDEN].copy_from_slice(&merged_vision_tokens[off..off+HIDDEN]);
                mask[ti] = true;
                v_idx += 1;
            } else {
                embed_lookup(&self.embed, id as usize, &mut x[ti*HIDDEN..(ti+1)*HIDDEN]);
            }
        }
        assert!(merged_vision_tokens.len() >= v_idx * HIDDEN,
                "merged vision tokens too short for {} image-token positions", v_idx);
        (x, mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn model_root() -> Option<std::path::PathBuf> {
        let p = std::path::PathBuf::from("/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/Cosmos3-Nano");
        if p.join("model.safetensors.index.json").exists() { Some(p) } else { None }
    }

    #[test]
    #[ignore = "loads 16 GB BF16; run with --ignored"]
    fn cosmos3_reasoner_paris() {
        let Some(root) = model_root() else { eprintln!("skip: model not present"); return; };
        eprintln!("loading reasoner from {}…", root.display());
        let t0 = Instant::now();
        let m = Cosmos3Reasoner::load(&root).expect("load");
        eprintln!("loaded in {:?}", t0.elapsed());

        // "The capital of France is" — Qwen2 tokenizer IDs (verified via Python).
        let ids: Vec<u32> = vec![785, 6722, 315, 9625, 374];
        let t0 = Instant::now();
        let logits = m.forward_last_logits(&ids);
        eprintln!("forward: {:?}", t0.elapsed());

        // Top-5 next-token argmax.
        let mut idx: Vec<usize> = (0..VOCAB).collect();
        idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        let top5: Vec<(usize, f32)> = idx.iter().take(5).map(|&i| (i, logits[i])).collect();
        eprintln!("top-5 next-token: {:?}", top5);

        // Golden: PyTorch reference loaded directly from the Cosmos 3 reasoner
        // weights (see workspace's reference dump).  Bit-identical f32 match on a
        // CPU forward.  Top-1 = " Paris" — capital of France.
        let golden_ids: [u32; 5] = [12095, 537, 7148, 1304, 7407];
        let golden_logits: [f32; 5] = [17.0800, 13.9279, 13.9200, 13.3606, 13.2980];
        for k in 0..5 {
            assert_eq!(top5[k].0 as u32, golden_ids[k],
                       "top-{k} id mismatch: mine {} vs golden {}", top5[k].0, golden_ids[k]);
            let diff = (top5[k].1 - golden_logits[k]).abs();
            assert!(diff < 0.01,
                    "top-{k} logit drift {diff}: mine {} vs golden {}", top5[k].1, golden_logits[k]);
        }
        eprintln!("golden match: top-5 ids + logits within 0.01 of PyTorch reference");
    }

    fn read_f32(name: &str) -> Option<Vec<f32>> {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
        let bytes = std::fs::read(p).ok()?;
        Some(bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())
    }

    /// Phase 3: multimodal forward with deepstack injection.  Verifies bit-close
    /// match against HF `Qwen3VLTextModel(inputs_embeds, visual_pos_masks,
    /// deepstack_visual_embeds)` on a 4-token prompt with 1 image-token slot
    /// replaced by a deterministic dummy vision token, and dummy deepstack
    /// features [0.01, 0.02, 0.03].
    #[test]
    #[ignore = "loads 16 GB BF16; run with --ignored"]
    fn cosmos3_multimodal_dummy_vision() {
        let Some(root) = model_root() else { eprintln!("skip"); return; };
        let Some(inputs_embeds) = read_f32("mm_inputs_embeds_4x4096.f32") else { eprintln!("skip: golden missing"); return; };
        let Some(golden_last) = read_f32("mm_last_hidden_4x4096.f32") else { eprintln!("skip"); return; };

        let m = Cosmos3Reasoner::load(&root).expect("load");
        // 4 tokens: <|vision_start|>, <|image|>, <|vision_end|>, "Describe".
        // image_token (151655) is at position 1.
        let visual_pos_mask = vec![false, true, false, false];
        // Dummy deepstack: each level fills its single visual position with 0.01 * (j+1).
        let deepstack: Vec<Vec<f32>> = (0..3)
            .map(|j| vec![0.01 * (j as f32 + 1.0); HIDDEN])
            .collect();

        let t0 = Instant::now();
        let last = m.multimodal_forward(&inputs_embeds, &visual_pos_mask, &deepstack);
        eprintln!("multimodal forward: {:?}", t0.elapsed());
        assert_eq!(last.len(), 4 * HIDDEN);

        let max_abs_diff = last.iter().zip(&golden_last).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        let mean_abs_diff = last.iter().zip(&golden_last).map(|(a,b)| (a-b).abs()).sum::<f32>() / last.len() as f32;
        for p in 0..4 {
            let row = &last[p*HIDDEN..p*HIDDEN+6];
            eprintln!("  pos {p}: first 6 = {:?}", row);
        }
        eprintln!("multimodal max_abs_diff = {max_abs_diff:.4e}, mean = {mean_abs_diff:.4e}");
        assert!(max_abs_diff < 0.5, "multimodal max drift too large: {max_abs_diff}");
        assert!(mean_abs_diff < 0.01, "multimodal mean drift too large: {mean_abs_diff}");
    }

    /// Greedy text generation — extends Phase 1 from one-shot logits to a real
    /// continuation. Matches HF `Qwen3VLTextModel` greedy decode token-for-token.
    #[test]
    #[ignore = "loads 16 GB BF16; ~10× Phase 1 forward; run with --ignored"]
    fn cosmos3_generate_paris_continuation() {
        let Some(root) = model_root() else { eprintln!("skip"); return; };
        let m = Cosmos3Reasoner::load(&root).expect("load");

        let prompt: Vec<u32> = vec![785, 6722, 315, 9625, 374];   // "The capital of France is"
        let t0 = Instant::now();
        let generated = m.generate_greedy(&prompt, 10, &[151643, 151645]);
        eprintln!("generate 10 tokens in {:?}", t0.elapsed());
        eprintln!("generated ids: {generated:?}");

        // HF golden: " Paris. The capital of Germany is Berlin. The"
        let golden: [u32; 10] = [12095, 13, 576, 6722, 315, 9856, 374, 19846, 13, 576];
        assert_eq!(generated.as_slice(), &golden);
        eprintln!("greedy continuation matches HF token-for-token");
    }

    /// Phase 3.B: end-to-end real-vision multimodal forward — the same 32×32 image
    /// as the vision golden runs through the real vision encoder + merger + deepstack
    /// mergers, then through the multimodal LLM with the real features wired in.
    /// Matches HF on the full chain.
    #[test]
    #[ignore = "loads 16 GB BF16 reasoner + 1.1 GB vision encoder; run with --ignored"]
    fn cosmos3_real_vision_e2e() {
        use crate::vision::VisionEncoder;
        let Some(root) = model_root() else { eprintln!("skip"); return; };
        let vision_path = root.join("vision_encoder/model.safetensors");
        if !vision_path.exists() { eprintln!("skip: vision encoder missing"); return; }
        let Some(golden_last) = read_f32("mm_real_last_hidden_4x4096.f32") else { eprintln!("skip: golden missing"); return; };

        // Generate the same deterministic 32×32 RGB gradient image patches the
        // Python golden used.
        let h: usize = 32; let w: usize = 32; let t: usize = 2;
        let patch = 16usize; let in_ch = 3usize; let temporal_patch = 2usize;
        let patch_dim = in_ch * temporal_patch * patch * patch;   // 1536
        let mut img = vec![0.0f32; t * in_ch * h * w];   // [T, C, H, W]
        for y in 0..h {
            for x in 0..w {
                for tt in 0..t {
                    img[((tt * in_ch + 0) * h + y) * w + x] = (x as f32 / (w-1) as f32) * 2.0 - 1.0;
                    img[((tt * in_ch + 1) * h + y) * w + x] = (y as f32 / (h-1) as f32) * 2.0 - 1.0;
                    img[((tt * in_ch + 2) * h + y) * w + x] = ((x+y) as f32 / (h+w-2) as f32) * 2.0 - 1.0;
                }
            }
        }
        // Patchify: 1×2×2 = 4 patches, each laid out as [C, T, H, W] flattened.
        let n_patches = 4;
        let mut patches = vec![0.0f32; n_patches * patch_dim];
        let mut token = 0;
        for tg in 0..(t / temporal_patch) {
            for hg in 0..(h / patch) {
                for wg in 0..(w / patch) {
                    let p_off = token * patch_dim;
                    for c in 0..in_ch {
                        for tp in 0..temporal_patch {
                            for ph in 0..patch {
                                for pw in 0..patch {
                                    let yi = hg * patch + ph;
                                    let xi = wg * patch + pw;
                                    let tt = tg * temporal_patch + tp;
                                    let v = img[((tt * in_ch + c) * h + yi) * w + xi];
                                    let idx = ((c * temporal_patch + tp) * patch + ph) * patch + pw;
                                    patches[p_off + idx] = v;
                                }
                            }
                        }
                    }
                    token += 1;
                }
            }
        }

        eprintln!("vision encoder load…");
        let t0 = Instant::now();
        let vis = VisionEncoder::load(&vision_path).expect("vision load");
        eprintln!("loaded vision in {:?}", t0.elapsed());

        let t0 = Instant::now();
        let (last_vit, deepstack) = vis.forward(&patches, (1, 2, 2));
        eprintln!("vision forward: {:?}", t0.elapsed());
        let merged = vis.merge(&last_vit);
        eprintln!("merger: {} merged tokens × {}", merged.len() / 4096, 4096);
        assert_eq!(merged.len(), 4096);                 // 1 merged token
        assert_eq!(deepstack.len(), 3);
        for d in &deepstack { assert_eq!(d.len(), 4096); }

        // Load LLM and run multimodal forward with the REAL vision features.
        eprintln!("reasoner load…");
        let t0 = Instant::now();
        let m = Cosmos3Reasoner::load(&root).expect("reasoner load");
        eprintln!("loaded reasoner in {:?}", t0.elapsed());

        let input_ids = [151652u32, 151655, 151653, 74785];
        let (inputs_embeds, visual_pos_mask) =
            m.build_multimodal_inputs(&input_ids, 151655, &merged);

        let t0 = Instant::now();
        let last = m.multimodal_forward(&inputs_embeds, &visual_pos_mask, &deepstack);
        eprintln!("multimodal forward: {:?}", t0.elapsed());

        for p in 0..4 {
            let row = &last[p*HIDDEN..p*HIDDEN+6];
            eprintln!("  pos {p}: first 6 = {row:?}");
        }
        let max_diff = last.iter().zip(&golden_last).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        let mean_diff = last.iter().zip(&golden_last).map(|(a,b)| (a-b).abs()).sum::<f32>() / last.len() as f32;
        eprintln!("e2e vs HF golden:  max_abs_diff = {max_diff:.4e}, mean = {mean_diff:.4e}");
        assert!(mean_diff < 0.005, "e2e mean drift {mean_diff} too large");
        assert!(max_diff  < 1.0,   "e2e max drift {max_diff} too large");
    }

    /// KV-cached greedy generation: same token-for-token output as the uncached
    /// `generate_greedy`, but ~50× faster after the prefill.
    #[test]
    #[ignore = "loads 16 GB BF16; run with --ignored"]
    fn cosmos3_generate_paris_continuation_cached() {
        let Some(root) = model_root() else { eprintln!("skip"); return; };
        let m = Cosmos3Reasoner::load(&root).expect("load");
        let prompt: Vec<u32> = vec![785, 6722, 315, 9625, 374];

        let t0 = Instant::now();
        let generated = m.generate_greedy_cached(&prompt, 10, &[151643, 151645]);
        let dt = t0.elapsed();
        eprintln!("cached generate 10 tokens in {dt:?} ({:.1} ms/token after prefill)",
                  dt.as_millis() as f32 / 11.0);
        eprintln!("generated ids: {generated:?}");
        let golden: [u32; 10] = [12095, 13, 576, 6722, 315, 9856, 374, 19846, 13, 576];
        assert_eq!(generated.as_slice(), &golden);
    }

    /// KV-cached multimodal greedy generation matches the uncached version.
    #[test]
    #[ignore = "loads 16 GB BF16 reasoner + 1.1 GB vision encoder; run with --ignored"]
    fn cosmos3_multimodal_generate_describe_cached() {
        use crate::vision::VisionEncoder;
        let Some(root) = model_root() else { eprintln!("skip"); return; };
        let vision_path = root.join("vision_encoder/model.safetensors");
        if !vision_path.exists() { eprintln!("skip"); return; }

        let h = 32usize; let w = 32usize; let t = 2usize;
        let patch = 16usize; let in_ch = 3usize; let temporal_patch = 2usize;
        let patch_dim = in_ch * temporal_patch * patch * patch;
        let mut img = vec![0.0f32; t * in_ch * h * w];
        for y in 0..h { for x in 0..w {
            for tt in 0..t {
                img[((tt * in_ch + 0) * h + y) * w + x] = (x as f32 / (w-1) as f32) * 2.0 - 1.0;
                img[((tt * in_ch + 1) * h + y) * w + x] = (y as f32 / (h-1) as f32) * 2.0 - 1.0;
                img[((tt * in_ch + 2) * h + y) * w + x] = ((x+y) as f32 / (h+w-2) as f32) * 2.0 - 1.0;
            }
        }}
        let mut patches = vec![0.0f32; 4 * patch_dim];
        let mut token = 0;
        for tg in 0..(t / temporal_patch) {
            for hg in 0..(h / patch) {
                for wg in 0..(w / patch) {
                    let p_off = token * patch_dim;
                    for c in 0..in_ch {
                        for tp in 0..temporal_patch {
                            for ph in 0..patch {
                                for pw in 0..patch {
                                    let yi = hg * patch + ph;
                                    let xi = wg * patch + pw;
                                    let tt = tg * temporal_patch + tp;
                                    let v = img[((tt * in_ch + c) * h + yi) * w + xi];
                                    let idx = ((c * temporal_patch + tp) * patch + ph) * patch + pw;
                                    patches[p_off + idx] = v;
                                }
                            }
                        }
                    }
                    token += 1;
                }
            }
        }

        let vis = VisionEncoder::load(&vision_path).expect("vision load");
        let (last_vit, deepstack) = vis.forward(&patches, (1, 2, 2));
        let merged = vis.merge(&last_vit);
        let m = Cosmos3Reasoner::load(&root).expect("reasoner load");
        let input_ids = [151652u32, 151655, 151653, 74785];
        let (embeds, mask) = m.build_multimodal_inputs(&input_ids, 151655, &merged);

        let t0 = Instant::now();
        let generated = m.multimodal_generate_greedy_cached(
            embeds, &mask, &deepstack, 5, &[151643, 151645]);
        let dt = t0.elapsed();
        eprintln!("cached multimodal gen 5 in {dt:?} ({:.1} ms/token after prefill)",
                  dt.as_millis() as f32 / 6.0);
        eprintln!("generated ids: {generated:?}");
        let golden: [u32; 5] = [279, 2168, 271, 785, 2168];
        assert_eq!(generated.as_slice(), &golden);
    }

    /// Phase 3.C: full multimodal greedy generation. Same image + prompt as
    /// `cosmos3_real_vision_e2e`, but actually generates 5 continuation tokens.
    /// Matches HF token-for-token: " the image\n\nThe image".
    #[test]
    #[ignore = "loads 16 GB BF16 reasoner + 1.1 GB vision encoder; ~5× multimodal_forward; run with --ignored"]
    fn cosmos3_multimodal_generate_describe() {
        use crate::vision::VisionEncoder;
        let Some(root) = model_root() else { eprintln!("skip"); return; };
        let vision_path = root.join("vision_encoder/model.safetensors");
        if !vision_path.exists() { eprintln!("skip: vision encoder missing"); return; }

        // Same synthetic gradient image as the Python golden.
        let h = 32usize; let w = 32usize; let t = 2usize;
        let patch = 16usize; let in_ch = 3usize; let temporal_patch = 2usize;
        let patch_dim = in_ch * temporal_patch * patch * patch;
        let mut img = vec![0.0f32; t * in_ch * h * w];
        for y in 0..h { for x in 0..w {
            for tt in 0..t {
                img[((tt * in_ch + 0) * h + y) * w + x] = (x as f32 / (w-1) as f32) * 2.0 - 1.0;
                img[((tt * in_ch + 1) * h + y) * w + x] = (y as f32 / (h-1) as f32) * 2.0 - 1.0;
                img[((tt * in_ch + 2) * h + y) * w + x] = ((x+y) as f32 / (h+w-2) as f32) * 2.0 - 1.0;
            }
        }}
        let mut patches = vec![0.0f32; 4 * patch_dim];
        let mut token = 0;
        for tg in 0..(t / temporal_patch) {
            for hg in 0..(h / patch) {
                for wg in 0..(w / patch) {
                    let p_off = token * patch_dim;
                    for c in 0..in_ch {
                        for tp in 0..temporal_patch {
                            for ph in 0..patch {
                                for pw in 0..patch {
                                    let yi = hg * patch + ph;
                                    let xi = wg * patch + pw;
                                    let tt = tg * temporal_patch + tp;
                                    let v = img[((tt * in_ch + c) * h + yi) * w + xi];
                                    let idx = ((c * temporal_patch + tp) * patch + ph) * patch + pw;
                                    patches[p_off + idx] = v;
                                }
                            }
                        }
                    }
                    token += 1;
                }
            }
        }

        let vis = VisionEncoder::load(&vision_path).expect("vision load");
        let (last_vit, deepstack) = vis.forward(&patches, (1, 2, 2));
        let merged = vis.merge(&last_vit);

        let m = Cosmos3Reasoner::load(&root).expect("reasoner load");
        let input_ids = [151652u32, 151655, 151653, 74785];
        let (inputs_embeds, visual_pos_mask) =
            m.build_multimodal_inputs(&input_ids, 151655, &merged);

        let t0 = Instant::now();
        let generated = m.multimodal_generate_greedy(
            &inputs_embeds, &visual_pos_mask, &deepstack, 5, &[151643, 151645]);
        eprintln!("multimodal generate 5 tokens in {:?}", t0.elapsed());
        eprintln!("generated ids: {generated:?}");

        // HF golden: " the image\n\nThe image"
        let golden: [u32; 5] = [279, 2168, 271, 785, 2168];
        assert_eq!(generated.as_slice(), &golden,
                   "multimodal generation diverges from HF token-for-token");
    }
}
