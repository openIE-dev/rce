//! Gemma 4v vision tower (`gemma4v` projector) on the rce substrate.
//!
//! Authoritative spec: `tools/mtmd/models/gemma4v.cpp` +
//! `tools/mtmd/clip.cpp::build_vit`. Per-layer block structure mirrors
//! the **rce-gemma4 text block** (RMS pre/post-norms, per-head Q/K-
//! RMSNorm, SwiGLU FFN, `kq_scale = 1.0`), with three vision-specific
//! changes:
//!   1. **Bidirectional** attention (no causal mask).
//!   2. **2D RoPE per layer**: head_dim split — first half rotates by
//!      `pos_x`, second half by `pos_y`, NeoX style, theta = 100.
//!   3. **2D additive position embedding** at the input (the
//!      `v.position_embd.weight` 3D tensor stacks an X-lookup table
//!      and a Y-lookup table along the last axis).
//!
//! Tensor layout (probed):
//!   - `v.patch_embd.weight` [16, 16, 3, 768] F32, NO bias
//!   - `v.position_embd.weight` [768, 10240, 2] F32 (X table | Y table)
//!   - 16 blocks of `{ ln1, ln2, attn_post_norm, ffn_post_norm }` F32
//!     + `attn_q_norm[64], attn_k_norm[64]` F32 (per-head)
//!     + `attn_{q,k,v,out}.weight [768,768]` Q8_0, NO biases
//!     + `ffn_{gate,up,down}.weight` Q8_0, NO biases
//!     + each weight has companion `.input_max/.input_min/.output_max/
//!       .output_min` scalars for `Gemma4ClippableLinear` (ignored in
//!       this f32 reference; matches clip.cpp fallback path).
//!   - `mm.input_projection.weight [768, 1536]` Q8_0 — single linear
//!     projector to text d_model.
//!   - No `v.post_ln`, no `std_bias`, no `std_scale` (this build).
//!
//! Output: ⌊image_size/patch_size/n_merge⌋² image tokens of 1536.
//! For 224×224 image with patch=16, n_merge=3 → 4×4 = 16 tokens.

use rce_ingest_gguf::{read_tensor_f32, GgufFile, GgufValue};
use rce_op_qmm::QuantWeight;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

const D_EMBD: usize = 768;
const D_HEAD: usize = 64;
const N_HEAD: usize = 12;
const D_FF: usize = 3072;
const N_BLOCK: usize = 16;
const PATCH: usize = 16;
const IMG: usize = 224;
const GRID: usize = IMG / PATCH; // 14
const N_PATCH: usize = GRID * GRID; // 196
const ROPE_THETA: f32 = 100.0;
const N_MERGE_DEFAULT: usize = 3;
const EPS: f32 = 1e-6;

#[derive(Clone, Debug)]
pub struct G4vConfig {
    pub d_embd: usize,
    pub d_head: usize,
    pub n_head: usize,
    pub d_ff: usize,
    pub n_block: usize,
    pub patch: usize,
    pub img: usize,
    pub n_merge: usize,
    pub proj_dim: usize,
    pub rope_theta: f32,
    pub eps: f32,
}
impl Default for G4vConfig {
    fn default() -> Self {
        Self {
            d_embd: D_EMBD, d_head: D_HEAD, n_head: N_HEAD, d_ff: D_FF,
            n_block: N_BLOCK, patch: PATCH, img: IMG, n_merge: N_MERGE_DEFAULT,
            proj_dim: 1536, rope_theta: ROPE_THETA, eps: EPS,
        }
    }
}

struct VBlock {
    ln1: Vec<f32>,             // pre-attn RMS norm weight
    q_norm: Vec<f32>,          // [d_head] per-head Q RMS norm
    k_norm: Vec<f32>,          // [d_head] per-head K RMS norm
    attn_post_norm: Vec<f32>,  // post-attn RMS norm weight
    ln2: Vec<f32>,             // pre-FFN RMS norm weight
    ffn_post_norm: Vec<f32>,   // post-FFN RMS norm weight
    wq: QuantWeight, wk: QuantWeight, wv: QuantWeight, wo: QuantWeight,
    ffn_gate: QuantWeight, ffn_up: QuantWeight, ffn_down: QuantWeight,
}

pub struct VisionTower {
    pub cfg: G4vConfig,
    patch_w: Vec<f32>,    // [16*16*3 * 768] flat
    pos_x: Vec<f32>,      // [d_embd * pos_size]  X table
    pos_y: Vec<f32>,      // [d_embd * pos_size]  Y table
    pos_size: usize,
    blocks: Vec<VBlock>,
    mm_proj: QuantWeight, // [proj_dim, d_embd]
    _mmap: Arc<Mmap>,
}

fn rms_norm_w(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w.iter()).map(|(v, ww)| v * inv * ww).collect()
}

/// Per-head RMS norm: `x` is [n_head * d_head]; norm independently
/// over each head slice using shared weight `w` of length `d_head`.
fn rms_norm_per_head(x: &[f32], w: &[f32], n_head: usize, d_head: usize, eps: f32) -> Vec<f32> {
    debug_assert_eq!(x.len(), n_head * d_head);
    debug_assert_eq!(w.len(), d_head);
    let mut out = vec![0.0f32; x.len()];
    for h in 0..n_head {
        let off = h * d_head;
        let ms: f32 = x[off..off + d_head].iter().map(|v| v * v).sum::<f32>() / d_head as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for d in 0..d_head { out[off + d] = x[off + d] * inv * w[d]; }
    }
    out
}

fn gelu_tanh(x: f32) -> f32 {
    let c = 0.797_884_56f32;
    0.5 * x * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())
}
fn silu(x: f32) -> f32 {
    if x >= 0.0 { x / (1.0 + (-x).exp()) }
    else { let e = x.exp(); x * e / (1.0 + e) }
}

fn qmm(a: &[f32], w: &QuantWeight) -> Vec<f32> {
    let mut o = vec![0.0f32; w.n_rows];
    rce_op_qmm::mm_1d(a, w, &mut o);
    o
}

/// Apply NeoX RoPE to one half of a per-head vector at position `pos`.
/// `half_len = d_head/2`; rotates pairs (i, i+half/2 within the half).
fn rope_neox_half(x: &mut [f32], half_off: usize, half_len: usize, base: f32, pos: usize) {
    debug_assert_eq!(half_len % 2, 0);
    let inner = half_len / 2;
    for i in 0..inner {
        let inv_freq = (base.ln() * (2.0 * i as f32) / half_len as f32).exp().recip();
        let theta = pos as f32 * inv_freq;
        let (sin, cos) = theta.sin_cos();
        let a = x[half_off + i];
        let b = x[half_off + i + inner];
        x[half_off + i] = a * cos - b * sin;
        x[half_off + i + inner] = a * sin + b * cos;
    }
}

/// 2D RoPE: split `d_head` in half. First half rotates by pos_x, second
/// half by pos_y. NeoX style. Applied per-head, per-token.
fn apply_2d_rope_inplace(
    x: &mut [f32], n_head: usize, d_head: usize, base: f32, pos_x: usize, pos_y: usize,
) {
    debug_assert_eq!(d_head % 2, 0);
    let half = d_head / 2;
    for h in 0..n_head {
        let head_off = h * d_head;
        rope_neox_half(x, head_off, half, base, pos_x);
        rope_neox_half(x, head_off + half, half, base, pos_y);
    }
}

impl VisionTower {
    pub fn load(mmproj_path: impl AsRef<Path>) -> Result<Self, String> {
        let f = File::open(mmproj_path.as_ref()).map_err(|e| format!("open: {e}"))?;
        let mmap = Arc::new(unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap: {e}"))?);
        let g = GgufFile::parse(&mmap[..]).map_err(|e| format!("parse: {e:?}"))?;
        let bytes: &[u8] = &mmap[..];

        let mut cfg = G4vConfig::default();
        let u32m = |k: &str| -> Option<u32> {
            g.metadata.iter().find(|kv| kv.key == k)
                .and_then(|kv| match &kv.value { GgufValue::U32(v) => Some(*v), _ => None })
        };
        if let Some(v) = u32m("clip.vision.image_size") { cfg.img = v as usize; }
        if let Some(v) = u32m("clip.vision.patch_size") { cfg.patch = v as usize; }
        if let Some(v) = u32m("clip.vision.block_count") { cfg.n_block = v as usize; }
        if let Some(v) = u32m("clip.vision.attention.head_count") { cfg.n_head = v as usize; }
        if let Some(v) = u32m("clip.vision.embedding_length") {
            cfg.d_embd = v as usize;
            cfg.d_head = cfg.d_embd / cfg.n_head;
        }
        if let Some(v) = u32m("clip.vision.feed_forward_length") { cfg.d_ff = v as usize; }
        if let Some(v) = u32m("clip.vision.projection_dim") { cfg.proj_dim = v as usize; }
        if let Some(v) = u32m("clip.vision.projector.scale_factor") { cfg.n_merge = v as usize; }

        let rd = |n: &str| read_tensor_f32(bytes, &g, n).map_err(|e| format!("{n}: {e:?}"));
        let qw = |n: &str, r: usize, c: usize| QuantWeight::from_gguf(
            Arc::clone(&mmap), &g, n, r, c).map_err(|e| format!("{n}: {e}"));

        let patch_w = rd("v.patch_embd.weight")?; // [16,16,3,768] F32

        // Position embedding: 3D tensor [d_embd, pos_size, 2]. The two
        // axes-of-the-last-dim are X-table (0) and Y-table (1).
        let pos_all = rd("v.position_embd.weight")?;
        let pos_tensor = g.tensors.iter()
            .find(|t| t.name == "v.position_embd.weight")
            .ok_or("v.position_embd.weight missing")?;
        let pos_size = pos_tensor.dims.get(1).copied().ok_or("pos dim[1]")? as usize;
        let stride = cfg.d_embd * pos_size;
        if pos_all.len() != 2 * stride {
            return Err(format!("pos_embd size {} != 2*{stride}", pos_all.len()));
        }
        let pos_x = pos_all[..stride].to_vec();
        let pos_y = pos_all[stride..].to_vec();

        let mut blocks = Vec::with_capacity(cfg.n_block);
        for il in 0..cfg.n_block {
            let bn = format!("v.blk.{il}");
            blocks.push(VBlock {
                ln1: rd(&format!("{bn}.ln1.weight"))?,
                ln2: rd(&format!("{bn}.ln2.weight"))?,
                q_norm: rd(&format!("{bn}.attn_q_norm.weight"))?,
                k_norm: rd(&format!("{bn}.attn_k_norm.weight"))?,
                attn_post_norm: rd(&format!("{bn}.attn_post_norm.weight"))?,
                ffn_post_norm: rd(&format!("{bn}.ffn_post_norm.weight"))?,
                wq: qw(&format!("{bn}.attn_q.weight"), cfg.d_embd, cfg.d_embd)?,
                wk: qw(&format!("{bn}.attn_k.weight"), cfg.d_embd, cfg.d_embd)?,
                wv: qw(&format!("{bn}.attn_v.weight"), cfg.d_embd, cfg.d_embd)?,
                wo: qw(&format!("{bn}.attn_out.weight"), cfg.d_embd, cfg.d_embd)?,
                ffn_gate: qw(&format!("{bn}.ffn_gate.weight"), cfg.d_ff, cfg.d_embd)?,
                ffn_up: qw(&format!("{bn}.ffn_up.weight"), cfg.d_ff, cfg.d_embd)?,
                ffn_down: qw(&format!("{bn}.ffn_down.weight"), cfg.d_embd, cfg.d_ff)?,
            });
        }
        let mm_proj = qw("mm.input_projection.weight", cfg.proj_dim, cfg.d_embd)?;
        eprintln!(
            "g4v: {n_block}L d{d}h{nh} ff{ff} img{img} patch{p} merge{m} → proj{pd} (pos_size={ps})",
            n_block = cfg.n_block, d = cfg.d_embd, nh = cfg.n_head, ff = cfg.d_ff,
            img = cfg.img, p = cfg.patch, m = cfg.n_merge, pd = cfg.proj_dim, ps = pos_size
        );
        Ok(Self { cfg, patch_w, pos_x, pos_y, pos_size, blocks, mm_proj, _mmap: mmap })
    }

    /// Decode image → resize to `IMG×IMG` bilinear → normalize
    /// `pixel/255 × 2 − 1` (image_mean=0, image_std=1 nominal — the
    /// `×2−1` is hard-coded in gemma4v.cpp:7).
    fn preprocess(&self, path: &Path) -> Result<Vec<f32>, String> {
        let img = image::open(path).map_err(|e| format!("image open: {e}"))?;
        let size = self.cfg.img as u32;
        let img = img.resize_exact(size, size, image::imageops::FilterType::Triangle);
        let rgb = img.to_rgb8();
        // ggml inp_raw layout [nx, ny, c]: idx = x + y*w + c*w*h
        let w = self.cfg.img;
        let mut out = vec![0.0f32; w * w * 3];
        for y in 0..w {
            for x in 0..w {
                let px = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    out[x + y * w + c * w * w] = (px[c] as f32 / 255.0) * 2.0 - 1.0;
                }
            }
        }
        Ok(out)
    }

    /// Patch embedding + 2D position add → [n_patches][d_embd].
    fn patchify(&self, raw: &[f32]) -> Vec<Vec<f32>> {
        let cfg = &self.cfg;
        let grid = cfg.img / cfg.patch;
        let n_patches = grid * grid;
        let mut patches = vec![vec![0.0f32; cfg.d_embd]; n_patches];
        let ksz = cfg.patch;
        for p in 0..n_patches {
            let ow = p % grid;
            let oh = p / grid;
            let x0 = ow * ksz;
            let y0 = oh * ksz;
            // Flatten patch in ggml conv2d weight order: kw + kh*K + ic*K*K.
            let mut pv = vec![0.0f32; ksz * ksz * 3];
            for kh in 0..ksz {
                for kw in 0..ksz {
                    for ic in 0..3 {
                        pv[kw + kh * ksz + ic * ksz * ksz] =
                            raw[(x0 + kw) + (y0 + kh) * cfg.img + ic * cfg.img * cfg.img];
                    }
                }
            }
            // conv_out[oc] = dot(pv, filter_oc); filter_oc = patch_w[oc*FILTER..]
            // FILTER = ksz*ksz*3 = d_embd.
            let f_len = ksz * ksz * 3;
            debug_assert_eq!(f_len, cfg.d_embd);
            let out_p = &mut patches[p];
            for oc in 0..cfg.d_embd {
                let f = &self.patch_w[oc * f_len..oc * f_len + f_len];
                let mut acc = 0.0f32;
                for k in 0..f_len { acc += pv[k] * f[k]; }
                out_p[oc] = acc;
            }
            // 2D additive position: pos_x[ow] + pos_y[oh] (each [d_embd]).
            let px = &self.pos_x[ow * cfg.d_embd..(ow + 1) * cfg.d_embd];
            let py = &self.pos_y[oh * cfg.d_embd..(oh + 1) * cfg.d_embd];
            for d in 0..cfg.d_embd { out_p[d] += px[d] + py[d]; }
        }
        patches
    }

    fn vit(&self, patches: &mut Vec<Vec<f32>>) {
        let cfg = &self.cfg;
        let grid = cfg.img / cfg.patch;
        let scale = 1.0f32; // kq_scale = 1.0 (Gemma 4v)
        let n_pos = patches.len();
        for blk in &self.blocks {
            // --- attention ---
            let normed: Vec<Vec<f32>> = patches.iter()
                .map(|x| rms_norm_w(x, &blk.ln1, cfg.eps)).collect();
            // Q/K/V (no biases), reshape to [n_head, d_head] implicit,
            // per-head Q/K-norm, then 2D RoPE.
            let mut qs = Vec::with_capacity(n_pos);
            let mut ks = Vec::with_capacity(n_pos);
            let mut vs = Vec::with_capacity(n_pos);
            for (p, n) in normed.iter().enumerate() {
                let ow = p % grid;
                let oh = p / grid;
                let mut q = qmm(n, &blk.wq);
                let mut k = qmm(n, &blk.wk);
                let v = qmm(n, &blk.wv);
                q = rms_norm_per_head(&q, &blk.q_norm, cfg.n_head, cfg.d_head, cfg.eps);
                k = rms_norm_per_head(&k, &blk.k_norm, cfg.n_head, cfg.d_head, cfg.eps);
                apply_2d_rope_inplace(&mut q, cfg.n_head, cfg.d_head, cfg.rope_theta, ow, oh);
                apply_2d_rope_inplace(&mut k, cfg.n_head, cfg.d_head, cfg.rope_theta, ow, oh);
                qs.push(q); ks.push(k); vs.push(v);
            }
            // Bidirectional attention (every i attends to every j).
            // Flatten K/V per layer to [n_pos × d_embd] for the fused kernel.
            let mut k_flat = Vec::with_capacity(n_pos * cfg.d_embd);
            let mut v_flat = Vec::with_capacity(n_pos * cfg.d_embd);
            for j in 0..n_pos {
                k_flat.extend_from_slice(&ks[j]);
                v_flat.extend_from_slice(&vs[j]);
            }
            let mut attn = vec![vec![0.0f32; cfg.d_embd]; n_pos];
            for i in 0..n_pos {
                for h in 0..cfg.n_head {
                    let o = h * cfg.d_head;
                    let qh = &qs[i][o..o + cfg.d_head];
                    let attended_h = rce_op_attn::fused_attend_head_window(
                        qh, &k_flat, &v_flat,
                        h, 0, n_pos, cfg.d_embd, cfg.d_head, scale,
                    );
                    attn[i][o..o + cfg.d_head].copy_from_slice(&attended_h);
                }
            }
            // wo + attn_post_norm (RMSnorm of post-wo) + residual to inpL.
            for i in 0..n_pos {
                let proj = qmm(&attn[i], &blk.wo);
                let normed_proj = rms_norm_w(&proj, &blk.attn_post_norm, cfg.eps);
                for d in 0..cfg.d_embd { patches[i][d] += normed_proj[d]; }
            }
            // --- FFN: SwiGLU (gate, up, down) with ffn_post_norm + residual ---
            for i in 0..n_pos {
                let n = rms_norm_w(&patches[i], &blk.ln2, cfg.eps);
                let g = qmm(&n, &blk.ffn_gate);
                let u = qmm(&n, &blk.ffn_up);
                let mut h = vec![0.0f32; cfg.d_ff];
                for j in 0..cfg.d_ff { h[j] = silu(g[j]) * u[j]; }
                let dn = qmm(&h, &blk.ffn_down);
                let normed_dn = rms_norm_w(&dn, &blk.ffn_post_norm, cfg.eps);
                for d in 0..cfg.d_embd { patches[i][d] += normed_dn[d]; }
            }
        }
    }

    /// 2D average-pool over the patch grid with kernel = stride = n_merge,
    /// no padding. Truncates if grid not divisible by n_merge (matches
    /// ggml_pool_2d behavior). Then × √d_embd.
    fn pool_merge(&self, patches: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let cfg = &self.cfg;
        let grid = cfg.img / cfg.patch;
        let m = cfg.n_merge;
        let out_grid = grid / m;
        let scale = (cfg.d_embd as f32).sqrt();
        let mut out = Vec::with_capacity(out_grid * out_grid);
        let inv = 1.0 / (m * m) as f32;
        for or in 0..out_grid {
            for oc in 0..out_grid {
                let mut tok = vec![0.0f32; cfg.d_embd];
                for ki in 0..m {
                    for kj in 0..m {
                        let py = or * m + ki;
                        let px = oc * m + kj;
                        let p = py * grid + px;
                        for d in 0..cfg.d_embd { tok[d] += patches[p][d]; }
                    }
                }
                for d in 0..cfg.d_embd { tok[d] = tok[d] * inv * scale; }
                out.push(tok);
            }
        }
        out
    }

    /// Full vision forward: image file → N image embeddings of `proj_dim`.
    pub fn encode_image(&self, path: impl AsRef<Path>) -> Result<Vec<Vec<f32>>, String> {
        let raw = self.preprocess(path.as_ref())?;
        let mut patches = self.patchify(&raw);
        self.vit(&mut patches);
        let pooled = self.pool_merge(&patches);
        // Projector (single linear, NO bias) + post-projection RMSNorm
        // (no learned weight — pure normalization).
        let mut out = Vec::with_capacity(pooled.len());
        for tok in &pooled {
            let proj = qmm(tok, &self.mm_proj);
            // Pure RMS normalize (no learned weight, multiply by 1.0).
            let n = proj.len();
            let ms: f32 = proj.iter().map(|v| v * v).sum::<f32>() / n as f32;
            let inv = 1.0 / (ms + self.cfg.eps).sqrt();
            let normed: Vec<f32> = proj.iter().map(|v| v * inv).collect();
            out.push(normed);
        }
        Ok(out)
    }
}

// Suppress dead-code warning for gelu_tanh (kept for future
// non-SwiGLU FFN variants).
#[allow(dead_code)]
fn _kept() { let _ = gelu_tanh(0.0); }

// ───── Multimodal runtime: vision tower + text decoder + splice ─────

use rce_gemma4::Gemma4Runtime;
use rce_ingest_gguf::{sample_with_history, SampleConfig};

/// Gemma 4v image-region boundary text strings (from
/// `tools/mtmd/mtmd.cpp:478-484` `PROJECTOR_TYPE_GEMMA4V`):
/// `<|image>` (begin) ... raw image embeddings ... `<image|>` (end).
/// These are multi-byte strings tokenized via the regular BPE — NOT
/// single special-token ids. The N image embeddings are spliced
/// between the begin/end token sequences without any placeholder id.
pub const G4V_IMG_BEGIN: &str = "<|image>";
pub const G4V_IMG_END: &str = "<image|>";

pub struct Gemma4VlRuntime {
    pub text: Gemma4Runtime,
    pub vision: VisionTower,
}

#[derive(Clone, Copy, Debug)]
pub struct G4VlGenConfig {
    pub max_new_tokens: usize,
    pub sample: SampleConfig,
    pub rep_window: usize,
}
impl Default for G4VlGenConfig {
    fn default() -> Self {
        Self { max_new_tokens: 48, sample: SampleConfig::GREEDY, rep_window: 64 }
    }
}

impl Gemma4VlRuntime {
    pub fn load(
        text_gguf: impl AsRef<Path>, mmproj_gguf: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let text = Gemma4Runtime::load(text_gguf)?;
        let vision = VisionTower::load(mmproj_gguf)?;
        if vision.cfg.proj_dim != text.cfg.d_model {
            return Err(format!(
                "projector out {} != gemma4 d_model {}",
                vision.cfg.proj_dim, text.cfg.d_model
            ));
        }
        Ok(Self { text, vision })
    }

    /// Caption/answer for `image` given `question`. Chat-frames the
    /// prompt as `<bos><start_of_turn>user\n<|image|>×N{question}
    /// <end_of_turn>\n<start_of_turn>model\n`, encodes the image into
    /// N image embeddings, splices them at the placeholder positions,
    /// then greedily decodes.
    pub fn describe(
        &self, image: impl AsRef<Path>, question: &str, gcfg: &G4VlGenConfig,
        mut on_token: impl FnMut(&str),
    ) -> Result<String, String> {
        let tk = self.text.tokenizer();
        let bos = tk.bos_token_id.unwrap_or(2);
        let eos = tk.eos_token_id.unwrap_or(1);

        // Gemma 4 tokenizer is NOT gpt2 — use `encode`.
        let is_gpt2 = self.text.gguf.metadata.iter()
            .find(|kv| kv.key == "tokenizer.ggml.model")
            .map(|kv| matches!(&kv.value, GgufValue::String(s) if s == "gpt2"))
            .unwrap_or(false);
        let encode = |s: &str| -> Vec<i32> {
            if is_gpt2 { tk.encode_gpt2(s, false) } else { tk.encode(s, false) }
        };

        // Encode image → N embeddings of d_model.
        let img_embds = self.vision.encode_image(image)?;
        let n_img = img_embds.len();

        // Build prompt stream: <bos><start_of_turn>user\n<|image>{N
        // image embeddings}<image|>{question}<end_of_turn>\n
        // <start_of_turn>model\n. The image embeddings are RAW
        // injections between the begin/end text marker tokens — no
        // placeholder id in the prompt id stream.
        let pre = {
            let mut v = vec![bos];
            v.extend(encode("<start_of_turn>user\n"));
            v.extend(encode(G4V_IMG_BEGIN));
            v
        };
        let mid = encode(G4V_IMG_END);
        let post = {
            let mut v = encode(question);
            v.extend(encode("<end_of_turn>\n<start_of_turn>model\n"));
            v
        };

        let mut state = self.text.new_state();
        let mut last_logits: Option<Vec<f32>> = None;
        let mut history: Vec<i32> = Vec::new();

        // Prefill: pre text → image embeddings → mid text → post text.
        for &id in &pre {
            let e = self.text.embed_token(id);
            last_logits = Some(self.text.forward_embed(&mut state, e, id));
            history.push(id);
        }
        for e in &img_embds {
            last_logits = Some(self.text.forward_image_embed(&mut state, e.clone()));
        }
        for &id in mid.iter().chain(post.iter()) {
            let e = self.text.embed_token(id);
            last_logits = Some(self.text.forward_embed(&mut state, e, id));
            history.push(id);
        }

        // Decode (greedy/sampled per gcfg).
        let mut out_ids: Vec<i32> = Vec::new();
        let seed_for = |s: u64| {
            let mut c = gcfg.sample;
            c.seed = gcfg.sample.seed ^ s.wrapping_mul(0x9E37_79B9);
            c
        };
        let mut step = 0u64;
        let mut next = {
            let lg = last_logits.ok_or("empty prompt")?;
            let recent = &history[history.len().saturating_sub(gcfg.rep_window)..];
            sample_with_history(&lg, &seed_for(step), recent)
        };
        step += 1;
        for _ in 0..gcfg.max_new_tokens {
            if next == eos { break; }
            out_ids.push(next);
            history.push(next);
            on_token(&tk.decode_sequence(&[next], false));
            let e = self.text.embed_token(next);
            let lg = self.text.forward_embed(&mut state, e, next);
            let recent = &history[history.len().saturating_sub(gcfg.rep_window)..];
            next = sample_with_history(&lg, &seed_for(step), recent);
            step += 1;
        }
        Ok(tk.decode_sequence(&out_ids, true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const TXT: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-text-q8_0.gguf";
    const MMP: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-mmproj-q8_0.gguf";
    // E4B: same SigLIP vision encoder as E2B (768/3072/16L/12H); only the
    // projector output dim differs (1536 → 2560, = E4B text d_model). proj_dim
    // is read from `clip.vision.projection_dim`, so the mmproj is drop-in.
    const TXT_E4B: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/gemma-4-e4b-gguf/gemma-4-E4B-it-Q8_0.gguf";
    const MMP_E4B: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/gemma-4-e4b-gguf/mmproj-gemma-4-E4B-it-Q8_0.gguf";

    /// **g4vl.e4b_load — E4B mmproj loads with proj_dim=2560 == text d_model.**
    /// Proves the larger projector weights parse on the unchanged vision tower.
    #[test]
    fn g4vl_e4b_load() {
        if !std::path::Path::new(TXT_E4B).exists() || !std::path::Path::new(MMP_E4B).exists() {
            eprintln!("g4vl.e4b: E4B not mounted — skip"); return;
        }
        let rt = Gemma4VlRuntime::load(TXT_E4B, MMP_E4B).expect("load E4B vl");
        assert_eq!(rt.vision.cfg.proj_dim, rt.text.cfg.d_model);
        assert_eq!(rt.vision.cfg.proj_dim, 2560, "E4B vision projects into d_model=2560");
        eprintln!("g4vl.e4b OK: vision proj_dim={} == text d_model", rt.vision.cfg.proj_dim);
    }

    #[test]
    fn g4vl_a_load() {
        if !std::path::Path::new(TXT).exists() || !std::path::Path::new(MMP).exists() {
            eprintln!("g4vl.a: skip"); return;
        }
        let rt = Gemma4VlRuntime::load(TXT, MMP).expect("load");
        assert_eq!(rt.vision.cfg.proj_dim, rt.text.cfg.d_model);
        eprintln!("g4vl.a OK: vision proj_dim={} == text d_model", rt.vision.cfg.proj_dim);
    }

    /// **g4vl.b — end-to-end: real image → description.** Coherence
    /// gate (common function words OR majority-alphabetic) — a broken
    /// vision encoder, splice, or PLE-padding bug would garble this.
    #[test]
    fn g4vl_b_describe() {
        if !std::path::Path::new(TXT).exists() || !std::path::Path::new(MMP).exists() {
            eprintln!("g4vl.b: skip"); return;
        }
        let rt = Gemma4VlRuntime::load(TXT, MMP).expect("load");
        let img = "/tmp/sherry-llamacpp/tools/mtmd/test-1.jpeg";
        if !std::path::Path::new(img).exists() {
            eprintln!("g4vl.b: test image missing — skipping"); return;
        }
        let cfg = G4VlGenConfig {
            max_new_tokens: 40, sample: SampleConfig::GREEDY, rep_window: 64,
        };
        let mut chunks: Vec<String> = Vec::new();
        let t0 = std::time::Instant::now();
        let _ = rt.describe(img, "Describe this image briefly.", &cfg,
            |s| chunks.push(s.to_string())).expect("describe");
        let dt = t0.elapsed();
        let full = chunks.concat();
        eprintln!(
            "g4vl.b: img={img:?} → {full:?}  [{} tok in {dt:?}]",
            chunks.len()
        );
        assert!(!full.is_empty(), "empty description");
        let coherent = [" the ", " a ", " is ", " of ", " in ", " and ", " with ", " image ", " this "]
            .iter().any(|w| full.to_lowercase().contains(w))
            || full.chars().filter(|c| c.is_alphabetic()).count() >= full.len() / 2;
        assert!(coherent, "incoherent: {full:?}");
    }
}
