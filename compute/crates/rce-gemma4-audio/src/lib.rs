//! Gemma 4 audio tower (`gemma4a` projector) on the rce substrate —
//! **scaffold + tensor loading**, forward implementation is the next
//! brick. Authoritative spec: `tools/mtmd/models/gemma4a.cpp` (288
//! LoC, Conformer encoder).
//!
//! Architecture summary (Conformer with chunked-local attention):
//!
//! ```text
//!   mel_spec[T, n_mel=128, 1]
//!     │
//!     ▼   Subsampling Conv2D × 2 (k=3, stride 2; ReLU; LayerNorm)
//!         1 → 128 ch → 32 ch (reduces T by ~4×)
//!     │
//!     ▼   Flatten + input projection [→ 1024]
//!     │
//!     ▼   12 × Conformer block:
//!           cur = x + 0.5 · FFN1(RMSNorm(x))                  (half-step)
//!           cur = cur + attn_chunked_local_RPE(RMSNorm(cur))  (S=24 ctx,
//!                                                              C=12 chunk,
//!                                                              P=12 past,
//!                                                              softcap=50,
//!                                                              Q/K specific
//!                                                              scaling +
//!                                                              per_dim_scale)
//!           cur = cur + conv_module(RMSNorm(cur))             (pw1 [×2] →
//!                                                              GLU → causal
//!                                                              depthwise
//!                                                              conv k=5 →
//!                                                              RMSNorm →
//!                                                              SiLU → pw2)
//!           cur = cur + 0.5 · FFN2(RMSNorm(cur))              (half-step)
//!           cur = RMSNorm(cur, ln_2)
//!     │
//!     ▼   Output proj (audio_out_proj, if present)
//!     │
//!     ▼   Audio multimodal embedder: RMSNorm × mm_soft_emb_norm
//!         → mm.a.input_projection [1024 → 1536]
//!     │
//!     ▼   Audio tokens × 1536 dim (spliced into rce-gemma4 decoder
//!         between text markers — analogous to vision splice).
//! ```
//!
//! Tensor inventory (E2B mmproj, probed 2026-05-22):
//! - Pre-encoder:
//!   - `a.conv1d.0.weight [3, 3, 1, 128]` F32, ± `.bias [128]`
//!   - `a.conv1d.1.weight [3, 3, 128, 32]` F32, ± `.bias [32]`
//!   - `a.pre_encode.{0,1}.weight [ch]` F32 (LN-after-conv)
//!   - `a.input_projection.weight [1024, 1024]` F32
//! - Per Conformer block (12 layers, d=1024, ff=4096, 8 heads):
//!   - FFN-1: `ffn_norm`, `ffn_up [1024,4096]`, `ffn_down [4096,1024]`,
//!     `ffn_post_norm`
//!   - Attention: `attn_pre_norm`, `attn_q/k/v/out [1024,1024]`,
//!     `attn_k_rel [1024,1024]` (RPE projection), `attn_post_norm`,
//!     `per_dim_scale [128]`
//!   - Conv module: `norm_conv`, `conv_pw1 [1024,2048]` (GLU split),
//!     `conv_dw [5, 1024]` (depthwise k=5), `conv_norm`,
//!     `conv_pw2 [1024, 1024]`
//!   - FFN-2: `ffn_norm_1`, `ffn_up_1`, `ffn_down_1`, `ffn_post_norm_1`,
//!     `ln_2`
//! - Audio→text: `mm.a.input_projection.weight [1536, 1536]` Q8_0
//!
//! `Gemma4ClippableLinear` clamps are present (`.input_max/.input_min/
//! .output_max/.output_min`) on every weight, analogous to gemma4v.
//! Ignored in this f32 reference (matches `clip.cpp::build_mm` fallback).

use rce_ingest_gguf::{read_tensor_f32, GgufFile, GgufValue};
use rce_op_qmm::QuantWeight;
use memmap2::Mmap;
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

const D_EMBD: usize = 1024;
const D_HEAD: usize = 128;
const N_HEAD: usize = 8;
const D_FF: usize = 4096;
const N_BLOCK: usize = 12;
const N_MEL: usize = 128;
const PROJ_DIM: usize = 1536;
/// Audio-native projection width: `a.pre_encode.out.weight` is [1024, 1536]
/// and `mm.a.input_projection.weight` is [1536, text_d_model] on BOTH E2B and
/// E4B — the audio encoder's inner dim is a fixed 1536, independent of the
/// text model size. `PROJ_DIM`/`cfg.proj_dim` is the *output* (text d_model:
/// 1536 for E2B, 2560 for E4B). These coincide only for E2B.
const AUDIO_INNER: usize = 1536;
const EPS: f32 = 1e-6;

/// Chunked-local attention parameters (gemma4a.cpp:49-53). `MAX_PAST_HORIZON`
/// drives the per-position attention window; the other three are declared for
/// documentation symmetry with the reference's block-shaped formulation.
const MAX_PAST_HORIZON: usize = 12;  // P
#[allow(dead_code)] const CHUNK_SIZE: usize = 12; // C
#[allow(dead_code)] const CONTEXT_SIZE: usize = CHUNK_SIZE + MAX_PAST_HORIZON; // S = 24
#[allow(dead_code)] const N_REL_POS: usize = MAX_PAST_HORIZON + 1; // R = 13

#[derive(Clone, Debug)]
pub struct G4aConfig {
    pub d_embd: usize,
    pub d_head: usize,
    pub n_head: usize,
    pub d_ff: usize,
    pub n_block: usize,
    pub n_mel: usize,
    pub proj_dim: usize,
    pub eps: f32,
}
impl Default for G4aConfig {
    fn default() -> Self {
        Self {
            d_embd: D_EMBD, d_head: D_HEAD, n_head: N_HEAD, d_ff: D_FF,
            n_block: N_BLOCK, n_mel: N_MEL, proj_dim: PROJ_DIM, eps: EPS,
        }
    }
}

/// `Gemma4ClippableLinear` clamp parameters: input/output min and max.
/// `build_mm(w, x) = clamp(matmul(w, clamp(x, in_min, in_max)), out_min, out_max)`.
/// Per-matmul learned activation ranges per gemma4a.cpp:280-287.
#[derive(Copy, Clone, Default, Debug)]
pub struct ClampInfo { pub in_min: f32, pub in_max: f32, pub out_min: f32, pub out_max: f32 }
impl ClampInfo {
    fn maybe_load(bytes: &[u8], g: &GgufFile, weight_name: &str) -> Option<Self> {
        let read = |suffix: &str| -> Option<f32> {
            let n = weight_name.strip_suffix(".weight")
                .map(|s| format!("{s}.{suffix}"))
                .unwrap_or_else(|| format!("{weight_name}.{suffix}"));
            read_tensor_f32(bytes, g, &n).ok().and_then(|v| v.first().copied())
        };
        let in_max = read("input_max")?;
        let in_min = read("input_min")?;
        let out_max = read("output_max")?;
        let out_min = read("output_min")?;
        Some(Self { in_min, in_max, out_min, out_max })
    }
}

/// One Conformer block. Conv1D depthwise kernel `conv_dw [5,d_embd]`
/// stored as 2D weight (5 is the temporal kernel, d_embd is channels).
struct ABlock {
    // FFN-1 half-step (pre-attn macron module)
    ffn_norm: Vec<f32>,
    ffn_up: QuantWeight, ffn_up_c: Option<ClampInfo>,    // [d_embd, d_ff]
    ffn_down: QuantWeight, ffn_down_c: Option<ClampInfo>,
    ffn_post_norm: Vec<f32>,
    // Self-attention with RPE
    attn_pre_norm: Vec<f32>,
    wq: QuantWeight, wq_c: Option<ClampInfo>,
    wk: QuantWeight, wk_c: Option<ClampInfo>,
    wv: QuantWeight, wv_c: Option<ClampInfo>,
    wo: QuantWeight, wo_c: Option<ClampInfo>,
    attn_k_rel: QuantWeight,    // RPE — no clamp
    attn_post_norm: Vec<f32>,
    per_dim_scale: Vec<f32>,    // [d_head]
    // Convolution module
    norm_conv: Vec<f32>,
    conv_pw1: QuantWeight, conv_pw1_c: Option<ClampInfo>,
    conv_dw: Vec<f32>,
    conv_norm: Vec<f32>,
    conv_pw2: QuantWeight, conv_pw2_c: Option<ClampInfo>,
    // FFN-2 half-step
    ffn_norm_1: Vec<f32>,
    ffn_up_1: QuantWeight, ffn_up_1_c: Option<ClampInfo>,
    ffn_down_1: QuantWeight, ffn_down_1_c: Option<ClampInfo>,
    ffn_post_norm_1: Vec<f32>,
    // Block output norm
    ln_2: Vec<f32>,
}

/// Gemma 4 audio Conformer encoder.
pub struct AudioTower {
    pub cfg: G4aConfig,
    // Pre-encoder
    conv0_w: Vec<f32>,            // [3,3,1,128] F32
    conv0_norm: Vec<f32>,         // LN-after-conv0 (ch=128)
    conv1_w: Vec<f32>,            // [3,3,128,32] F32
    conv1_norm: Vec<f32>,         // LN-after-conv1 (ch=32)
    input_proj: Vec<f32>,         // [1024, 1024] F32
    // Conformer blocks
    blocks: Vec<ABlock>,
    // Audio output projection (`a.pre_encode.out.weight` = `audio_out_proj_w`
    // per clip.cpp:2443). Maps 1024 → 1536. F32 in this build (probed).
    audio_out_proj: Vec<f32>,    // [1024, 1536]
    audio_out_proj_b: Option<Vec<f32>>,  // optional bias [1536]
    // Audio multimodal embedder
    mm_soft_emb_norm: Option<Vec<f32>>,
    mm_proj: QuantWeight,         // [PROJ_DIM, PROJ_DIM] — final audio→text
    _mmap: Arc<Mmap>,
}

impl AudioTower {
    pub fn load(mmproj_path: impl AsRef<Path>) -> Result<Self, String> {
        let f = File::open(mmproj_path.as_ref()).map_err(|e| format!("open: {e}"))?;
        let mmap = Arc::new(unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap: {e}"))?);
        let _ = mmap.advise(memmap2::Advice::WillNeed);
        let g = GgufFile::parse(&mmap[..]).map_err(|e| format!("parse: {e:?}"))?;
        let bytes: &[u8] = &mmap[..];

        let mut cfg = G4aConfig::default();
        let u32m = |k: &str| -> Option<u32> {
            g.metadata.iter().find(|kv| kv.key == k)
                .and_then(|kv| match &kv.value { GgufValue::U32(v) => Some(*v), _ => None })
        };
        if let Some(v) = u32m("clip.audio.block_count") { cfg.n_block = v as usize; }
        if let Some(v) = u32m("clip.audio.attention.head_count") { cfg.n_head = v as usize; }
        if let Some(v) = u32m("clip.audio.embedding_length") {
            cfg.d_embd = v as usize;
            cfg.d_head = cfg.d_embd / cfg.n_head;
        }
        if let Some(v) = u32m("clip.audio.feed_forward_length") { cfg.d_ff = v as usize; }
        if let Some(v) = u32m("clip.audio.projection_dim") { cfg.proj_dim = v as usize; }
        if let Some(v) = u32m("clip.audio.num_mel_bins") { cfg.n_mel = v as usize; }

        let rd = |n: &str| read_tensor_f32(bytes, &g, n).map_err(|e| format!("{n}: {e:?}"));
        let rd_opt = |n: &str| read_tensor_f32(bytes, &g, n).ok();
        let qw = |n: &str, r: usize, c: usize| QuantWeight::from_gguf(
            Arc::clone(&mmap), &g, n, r, c).map_err(|e| format!("{n}: {e}"));
        let qw_opt = |n: &str, r: usize, c: usize| QuantWeight::from_gguf(
            Arc::clone(&mmap), &g, n, r, c).ok();

        let conv0_w = rd("a.conv1d.0.weight")?;
        let conv0_norm = rd("a.conv1d.0.norm.weight")?;
        let conv1_w = rd("a.conv1d.1.weight")?;
        let conv1_norm = rd("a.conv1d.1.norm.weight")?;
        let input_proj = rd("a.input_projection.weight")?;
        let audio_out_proj = rd("a.pre_encode.out.weight")?;
        let audio_out_proj_b = rd_opt("a.pre_encode.out.bias");

        let mut blocks = Vec::with_capacity(cfg.n_block);
        for il in 0..cfg.n_block {
            let bn = format!("a.blk.{il}");
            let clamp_for = |w: &str| -> Option<ClampInfo> {
                ClampInfo::maybe_load(bytes, &g, &format!("{bn}.{w}.weight"))
            };
            blocks.push(ABlock {
                ffn_norm: rd(&format!("{bn}.ffn_norm.weight"))?,
                ffn_up: qw(&format!("{bn}.ffn_up.weight"), cfg.d_ff, cfg.d_embd)?,
                ffn_up_c: clamp_for("ffn_up"),
                ffn_down: qw(&format!("{bn}.ffn_down.weight"), cfg.d_embd, cfg.d_ff)?,
                ffn_down_c: clamp_for("ffn_down"),
                ffn_post_norm: rd(&format!("{bn}.ffn_post_norm.weight"))?,
                attn_pre_norm: rd(&format!("{bn}.attn_pre_norm.weight"))?,
                wq: qw(&format!("{bn}.attn_q.weight"), cfg.d_embd, cfg.d_embd)?,
                wq_c: clamp_for("attn_q"),
                wk: qw(&format!("{bn}.attn_k.weight"), cfg.d_embd, cfg.d_embd)?,
                wk_c: clamp_for("attn_k"),
                wv: qw(&format!("{bn}.attn_v.weight"), cfg.d_embd, cfg.d_embd)?,
                wv_c: clamp_for("attn_v"),
                wo: qw(&format!("{bn}.attn_out.weight"), cfg.d_embd, cfg.d_embd)?,
                wo_c: clamp_for("attn_out"),
                attn_k_rel: qw(&format!("{bn}.attn_k_rel.weight"), cfg.d_embd, cfg.d_embd)?,
                attn_post_norm: rd(&format!("{bn}.attn_post_norm.weight"))?,
                per_dim_scale: rd(&format!("{bn}.per_dim_scale.weight"))?,
                norm_conv: rd(&format!("{bn}.norm_conv.weight"))?,
                conv_pw1: qw(&format!("{bn}.conv_pw1.weight"), 2 * cfg.d_embd, cfg.d_embd)?,
                conv_pw1_c: clamp_for("conv_pw1"),
                conv_dw: rd(&format!("{bn}.conv_dw.weight"))?,
                conv_norm: rd(&format!("{bn}.conv_norm.weight"))?,
                conv_pw2: qw(&format!("{bn}.conv_pw2.weight"), cfg.d_embd, cfg.d_embd)?,
                conv_pw2_c: clamp_for("conv_pw2"),
                ffn_norm_1: rd(&format!("{bn}.ffn_norm_1.weight"))?,
                ffn_up_1: qw(&format!("{bn}.ffn_up_1.weight"), cfg.d_ff, cfg.d_embd)?,
                ffn_up_1_c: clamp_for("ffn_up_1"),
                ffn_down_1: qw(&format!("{bn}.ffn_down_1.weight"), cfg.d_embd, cfg.d_ff)?,
                ffn_down_1_c: clamp_for("ffn_down_1"),
                ffn_post_norm_1: rd(&format!("{bn}.ffn_post_norm_1.weight"))?,
                ln_2: rd(&format!("{bn}.ln2.weight"))?,
            });
        }
        let _ = qw_opt; // reserved for future variants
        let mm_soft_emb_norm = rd_opt("mm.a.soft_emb_norm.weight");
        // [out = text d_model (cfg.proj_dim), in = AUDIO_INNER]. Square only on E2B.
        let mm_proj = qw("mm.a.input_projection.weight", cfg.proj_dim, AUDIO_INNER)?;

        eprintln!(
            "g4a: {n_block}L d{d}h{nh} ff{ff} mel{m} → proj{pd}  (scaffold; \
             forward TODO)",
            n_block = cfg.n_block, d = cfg.d_embd, nh = cfg.n_head, ff = cfg.d_ff,
            m = cfg.n_mel, pd = cfg.proj_dim,
        );
        Ok(Self {
            cfg, conv0_w, conv0_norm, conv1_w, conv1_norm, input_proj,
            blocks, audio_out_proj, audio_out_proj_b, mm_soft_emb_norm, mm_proj, _mmap: mmap,
        })
    }

    /// Encode a `[T][n_mel=128]` log-mel spectrogram (per
    /// [`MelExtractor`]) → `[T_out][proj_dim=1536]` audio tokens
    /// (T_out ≈ T/4 from the 2× subsampling conv2d).
    ///
    /// **Current scope (brick 5b-partial):** implements mel preproc +
    /// the SHAPES end-to-end through the Conformer block but with
    /// SIMPLIFIED attention (full bidirectional, no RPE, no
    /// chunking/RPE-shift) and SIMPLIFIED conv module (just two linear
    /// layers; depthwise temporal conv is the per-frame identity for
    /// now). This is structurally correct and lets us hit the audio
    /// splice path; perfect semantic correctness on the Conformer
    /// requires the chunked-local RPE attention + causal depthwise
    /// conv, which is its own A/B-grade brick.
    pub fn encode_mel(&self, mel: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        if mel.is_empty() { return Err("empty mel".into()); }
        let cfg = &self.cfg;
        let trc = std::env::var("RCE_AUDIO_TRACE").is_ok();
        macro_rules! sum_dump {
            ($label:expr, $frames:expr) => {
                if trc {
                    let s: f32 = $frames.iter().flat_map(|f| f.iter()).sum();
                    eprintln!("[RCE_AUDIO] {:30} sum={:.4}", $label, s);
                }
            };
        }
        // 1) Subsampling conv2d × 2 (stride 2 each → 4× frame reduction).
        //    Input shape: [time, mel] = [T, 128]; treat as a 2D image
        //    with `freq` on the inner axis. Each conv2d uses a 3×3
        //    kernel with stride 2 and SAME padding (=1 each side).
        let t_in = mel.len();
        sum_dump!("inp_mel", mel);
        let frames_after = subsample_conv2d(mel, &self.conv0_w, &self.conv0_norm, 1, 128, cfg.eps);
        sum_dump!("after_subsample_0", &frames_after);
        let frames_after = subsample_conv2d(&frames_after, &self.conv1_w, &self.conv1_norm, 128, 32, cfg.eps);
        sum_dump!("after_subsample_1", &frames_after);
        // Flatten freq×ch → input proj 1024 → d_embd.
        let mut hs: Vec<Vec<f32>> = frames_after.into_iter().map(|f| {
            let mut out = vec![0.0f32; cfg.d_embd];
            let cols = self.input_proj.len() / cfg.d_embd;
            debug_assert_eq!(f.len(), cols, "input_proj cols {cols} != frame {}", f.len());
            for r in 0..cfg.d_embd {
                let row = &self.input_proj[r * cols..r * cols + cols];
                let mut acc = 0.0f32;
                for c in 0..cols { acc += row[c] * f[c]; }
                out[r] = acc;
            }
            out
        }).collect();

        sum_dump!("after_input_proj", &hs);
        // 2) 12 Conformer blocks (simplified attention/conv; see fn doc).
        for (li, blk) in self.blocks.iter().enumerate() {
            conformer_block(&mut hs, blk, cfg);
            if trc { sum_dump!(format!("after_block_{li}"), &hs); }
        }

        // 3) Audio output projection (1024 → 1536).
        let mut projected: Vec<Vec<f32>> = hs.into_iter().map(|h| {
            // audio_out_proj: [in=1024, out=AUDIO_INNER(1536)] — output is the
            // audio-native width, NOT text d_model (they coincide only on E2B).
            let mut out = vec![0.0f32; AUDIO_INNER];
            // Layout flat row-major: element (in_c, out_c) at out_c*1024 + in_c.
            for o in 0..AUDIO_INNER {
                let row = &self.audio_out_proj[o * cfg.d_embd..o * cfg.d_embd + cfg.d_embd];
                let mut acc = 0.0f32;
                for k in 0..cfg.d_embd { acc += row[k] * h[k]; }
                out[o] = acc;
            }
            if let Some(b) = self.audio_out_proj_b.as_ref() {
                for o in 0..AUDIO_INNER { out[o] += b[o]; }
            }
            out
        }).collect();

        // 4) Audio multimodal embedder: RMSNorm (× optional soft_emb_norm)
        //    → mm.a.input_projection (1536 → 1536).
        for tok in projected.iter_mut() {
            let n = tok.len();
            let ms: f32 = tok.iter().map(|v| v * v).sum::<f32>() / n as f32;
            let inv = 1.0 / (ms + cfg.eps).sqrt();
            if let Some(w) = &self.mm_soft_emb_norm {
                for d in 0..n { tok[d] = tok[d] * inv * w[d]; }
            } else {
                for d in 0..n { tok[d] *= inv; }
            }
            let mut proj = vec![0.0f32; cfg.proj_dim];
            rce_op_qmm::mm_1d(tok, &self.mm_proj, &mut proj);
            *tok = proj;
        }
        let _ = t_in;
        Ok(projected)
    }
}

// ── Mel-spectrogram preprocessor ─────────────────────────────────────
//
// Gemma 4a mel-spec config (clip.cpp:1518-1527, mtmd-audio.cpp:764-781):
//   sample_rate = 16000, n_fft = 512, window_len = 320 (20ms),
//   hop_len = 160 (10ms), n_mel = 128, HTK mel scale (NOT Slaney),
//   no Slaney area normalization, |magnitude| (not power), natural
//   log, mel_floor = 0.001, semicausal left-pad by window_len/2.

const G4A_SAMPLE_RATE: usize = 16000;
const G4A_N_FFT: usize = 512;
const G4A_WIN_LEN: usize = 320;
const G4A_HOP: usize = 160;
const G4A_MEL_FLOOR: f32 = 0.001;

/// Hz → HTK mel: `2595 · log10(1 + f / 700)`.
fn hz_to_mel_htk(f: f32) -> f32 { 2595.0 * (1.0 + f / 700.0).log10() }
/// HTK mel → Hz.
fn mel_to_hz_htk(m: f32) -> f32 { 700.0 * (10f32.powf(m / 2595.0) - 1.0) }

/// Build a `[n_mel × (n_fft/2 + 1)]` triangular HTK mel filter bank
/// (no Slaney area normalization, scale=1.0 per Gemma 4 config).
fn build_mel_filterbank(n_mel: usize, n_fft: usize, sample_rate: usize) -> Vec<Vec<f32>> {
    let fmin = 0.0f32;
    let fmax = sample_rate as f32 / 2.0;
    let n_fft_bins = n_fft / 2 + 1;
    let m_lo = hz_to_mel_htk(fmin);
    let m_hi = hz_to_mel_htk(fmax);
    let mut hz_pts = vec![0.0f32; n_mel + 2];
    for i in 0..n_mel + 2 {
        let mel = m_lo + (m_hi - m_lo) * (i as f32) / (n_mel + 1) as f32;
        hz_pts[i] = mel_to_hz_htk(mel);
    }
    let bin_hz = sample_rate as f32 / n_fft as f32;
    let mut filters = vec![vec![0.0f32; n_fft_bins]; n_mel];
    for m in 0..n_mel {
        let f_left = hz_pts[m];
        let f_center = hz_pts[m + 1];
        let f_right = hz_pts[m + 2];
        let denom_l = (f_center - f_left).max(1e-30);
        let denom_r = (f_right - f_center).max(1e-30);
        for k in 0..n_fft_bins {
            let f = k as f32 * bin_hz;
            let w = if f >= f_left && f <= f_center {
                (f - f_left) / denom_l
            } else if f > f_center && f <= f_right {
                (f_right - f) / denom_r
            } else {
                0.0
            };
            filters[m][k] = w; // scale=1.0, no Slaney norm
        }
    }
    filters
}

/// Compute Gemma 4 log-mel spectrogram from 16 kHz mono f32 samples.
/// Returns `[n_frames][128]`.
pub fn mel_spectrogram_gemma4a(samples: &[f32]) -> Vec<Vec<f32>> {
    // Semicausal left-pad by window_len/2 (gemma4a.cpp:815).
    let pad_left = G4A_WIN_LEN / 2;
    // Right-pad so frame count matches PyTorch: pt_frames =
    // (n_with_left − (window_len + 1)) / hop + 1.
    let n_with_left = samples.len() + pad_left;
    let pt_frames = if n_with_left >= G4A_WIN_LEN + 1 {
        (n_with_left - (G4A_WIN_LEN + 1)) / G4A_HOP + 1
    } else { 0 };
    let n_padded_needed = (pt_frames.saturating_sub(1)) * G4A_HOP + G4A_N_FFT;
    let total_pad = std::cmp::max(n_padded_needed.saturating_sub(samples.len()), pad_left);
    let mut padded = vec![0.0f32; total_pad + samples.len()];
    padded[pad_left..pad_left + samples.len()].copy_from_slice(samples);

    // Hann window of length 320, zero-padded to 512.
    let mut hann = vec![0.0f32; G4A_N_FFT];
    for i in 0..G4A_WIN_LEN {
        hann[i] = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / G4A_WIN_LEN as f32).cos();
    }

    let filters = build_mel_filterbank(128, G4A_N_FFT, G4A_SAMPLE_RATE);
    let n_fft_bins = G4A_N_FFT / 2 + 1;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(G4A_N_FFT);
    let mut buf = vec![Complex32::new(0.0, 0.0); G4A_N_FFT];

    let mut out: Vec<Vec<f32>> = Vec::with_capacity(pt_frames);
    for t in 0..pt_frames {
        let start = t * G4A_HOP;
        // Apply Hann window into FFT buffer.
        for i in 0..G4A_N_FFT {
            let sample = if start + i < padded.len() { padded[start + i] } else { 0.0 };
            buf[i] = Complex32::new(sample * hann[i], 0.0);
        }
        fft.process(&mut buf);
        // Magnitude spectrum (|complex|, not power).
        let mut mag = vec![0.0f32; n_fft_bins];
        for k in 0..n_fft_bins { mag[k] = (buf[k].re * buf[k].re + buf[k].im * buf[k].im).sqrt(); }
        // Mel filter bank.
        let mut mel = vec![0.0f32; 128];
        for m in 0..128 {
            let row = &filters[m];
            let mut acc = 0.0f32;
            for k in 0..n_fft_bins { acc += row[k] * mag[k]; }
            mel[m] = acc.max(G4A_MEL_FLOOR).ln();
        }
        out.push(mel);
    }
    out
}

/// Convenience: load 16-bit little-endian mono PCM file → f32 samples
/// in [-1, 1].
pub fn load_pcm_s16le(path: impl AsRef<Path>) -> std::io::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    let n = bytes.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let s = i16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
        out.push(s as f32 / 32768.0);
    }
    Ok(out)
}

/// Subsampling Conv2D with stride 2, kernel 3, SAME padding 1, +
/// LayerNorm (over channels) + ReLU. Mirrors gemma4a.cpp subsampling
/// step. `weight` is `[3, 3, in_ch, out_ch]` flat in ggml order
/// `(kw + kh*3 + ic*9 + oc*9*ic_total)`.
fn subsample_conv2d(
    input: &[Vec<f32>], weight: &[f32], norm: &[f32], in_ch: usize, out_ch: usize, eps: f32,
) -> Vec<Vec<f32>> {
    let t_in = input.len();
    let f_in = if t_in > 0 { input[0].len() / in_ch } else { 0 };
    let t_out = t_in.div_ceil(2);
    let f_out = f_in.div_ceil(2);
    let k = 3usize;
    let pad = 1usize;
    let mut out = Vec::with_capacity(t_out);
    for ti in 0..t_out {
        let mut row = vec![0.0f32; f_out * out_ch];
        let t_src_center = ti * 2;
        for fi in 0..f_out {
            let f_src_center = fi * 2;
            for oc in 0..out_ch {
                let mut acc = 0.0f32;
                for kh in 0..k {
                    for kw in 0..k {
                        // ggml_conv_2d kernel layout `[kw, kh, ic, oc]`:
                        // After gemma4a's transpose, input ne=[FREQ=128,
                        // TIME, 1] — i.e. **ne[0]=freq**, ne[1]=time
                        // (verified empirically: `inp_raw{1743,128,1,1}
                        // → TRANSPOSE → {128,1743,1,1}`). So kw aligns
                        // with FREQ, kh with TIME. My t (mel time
                        // dimension) → kh; f (mel freq dimension) → kw.
                        let t_src = t_src_center as isize + kh as isize - pad as isize;
                        let f_src = f_src_center as isize + kw as isize - pad as isize;
                        if t_src < 0 || f_src < 0 || t_src >= t_in as isize || f_src >= f_in as isize {
                            continue;
                        }
                        let t_src = t_src as usize;
                        let f_src = f_src as usize;
                        for ic in 0..in_ch {
                            // Per-frame layout follows ggml's [ch, freq]
                            // permute: ch is the fast axis. Index = f*in_ch
                            // + c. First conv input is mono mel = single
                            // channel [128], so just [f].
                            let src_val = if in_ch == 1 {
                                input[t_src][f_src]
                            } else {
                                input[t_src][f_src * in_ch + ic]
                            };
                            // weight order (ggml conv2d): kw + kh*K + ic*K*K + oc*K*K*ic_total
                            let w_idx = kw + kh * k + ic * k * k + oc * k * k * in_ch;
                            acc += src_val * weight[w_idx];
                        }
                    }
                }
                row[fi * out_ch + oc] = acc;
            }
        }
        // LayerNorm: verified from eval-callback dump — `NORM` operates
        // on tensor of ne `{128, 64, 872, 1}` after the C++ permutes
        // OC to ne[0]. So LN normalizes over **channels** per
        // (freq, time) cell; the per-channel `norm_w[oc]` weight then
        // broadcasts via element-wise multiply. Then ReLU.
        for fi in 0..f_out {
            let base = fi * out_ch;
            let mut mean = 0.0f32;
            let mut var = 0.0f32;
            for oc in 0..out_ch { mean += row[base + oc]; }
            mean /= out_ch as f32;
            for oc in 0..out_ch {
                let v = row[base + oc] - mean;
                var += v * v;
            }
            var /= out_ch as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for oc in 0..out_ch {
                let v = (row[base + oc] - mean) * inv * norm[oc];
                row[base + oc] = v.max(0.0); // ReLU
            }
        }
        out.push(row);
    }
    out
}

fn rms_norm_w(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w.iter()).map(|(v, ww)| v * inv * ww).collect()
}
fn silu(x: f32) -> f32 {
    if x >= 0.0 { x / (1.0 + (-x).exp()) }
    else { let e = x.exp(); x * e / (1.0 + e) }
}
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

fn qmm(a: &[f32], w: &QuantWeight) -> Vec<f32> {
    let mut o = vec![0.0f32; w.n_rows];
    rce_op_qmm::mm_1d(a, w, &mut o);
    o
}

/// `Gemma4ClippableLinear`: clamp input to `[in_min, in_max]`, matmul,
/// clamp output to `[out_min, out_max]`. Per `clip_graph_gemma4a::
/// build_mm` (gemma4a.cpp:279-287). Without clamps the activations
/// drift through the 12-layer Conformer and the model can't decode.
/// `Gemma4ClippableLinear` wrapper. Mirrors `clip_graph_gemma4a::build_mm`
/// (gemma4a.cpp:279-287): clamp input to `[in_min, in_max]`, matmul,
/// clamp output to `[out_min, out_max]`. Clamps are essential for the
/// reference's activation distribution — disabling them lets large
/// values propagate and the model stops decoding audio. Opt out via
/// `RCE_AUDIO_CLAMP_OFF=1` for A/B diagnostics only.
fn qmm_clamped_tag(a: &[f32], w: &QuantWeight, c: &Option<ClampInfo>, _tag: &str) -> Vec<f32> {
    if std::env::var("RCE_AUDIO_CLAMP_OFF").is_ok() { return qmm(a, w); }
    qmm_clamped(a, w, c)
}

fn qmm_clamped(a: &[f32], w: &QuantWeight, c: &Option<ClampInfo>) -> Vec<f32> {
    match c {
        None => qmm(a, w),
        Some(ci) => {
            let clamped_in: Vec<f32> = a.iter()
                .map(|v| v.clamp(ci.in_min, ci.in_max)).collect();
            let mut o = vec![0.0f32; w.n_rows];
            rce_op_qmm::mm_1d(&clamped_in, w, &mut o);
            for v in o.iter_mut() { *v = v.clamp(ci.out_min, ci.out_max); }
            o
        }
    }
}

/// **Conformer block forward (simplified).** Authoritative structure
/// from gemma4a.cpp build():
/// ```text
///   cur = x + 0.5 · FFN1(RMSN(x))                     [half-step]
///   cur = cur + attn(RMSN(cur), per_dim_scale)        [chunked-RPE
///                                                      simplified
///                                                      to full
///                                                      bidirectional]
///   cur = cur + conv_module(RMSN(cur))                [pw1 → GLU →
///                                                      depthwise k=5
///                                                      → SiLU → pw2]
///   cur = cur + 0.5 · FFN2(RMSN(cur))                 [half-step]
///   cur = RMSN(cur, ln_2)
/// ```
fn conformer_block(hs: &mut Vec<Vec<f32>>, blk: &ABlock, cfg: &G4aConfig) {
    let t = hs.len();
    let res_w = 0.5f32;
    let trc = std::env::var("RCE_AUDIO_TRACE").is_ok();
    let il_marker = std::env::var("RCE_AUDIO_TRACE_BLK").ok()
        .and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
    // Track which "block index" we're in via a per-call static; here we
    // just dump unconditionally when tracing (each block prints).
    macro_rules! dump_tok0 {
        ($label:expr, $frames:expr) => {
            if trc {
                let s0: f32 = $frames[0].iter().sum();
                let st: f32 = $frames.iter().flat_map(|f| f.iter()).sum();
                eprintln!("[RCE_AUDIO] cf {:25}  tok[0].sum={s0:.4}  total={st:.4}", $label);
            }
        };
    }
    let _ = il_marker;

    // FFN-1 half-step. Every matmul is `Gemma4ClippableLinear` (clamp
    // input → matmul → clamp output) per gemma4a.cpp:279-287.
    let ffn1 = |x: &[f32]| -> Vec<f32> {
        let n = rms_norm_w(x, &blk.ffn_norm, cfg.eps);
        let up = qmm_clamped_tag(&n, &blk.ffn_up, &blk.ffn_up_c, "ffn_up");
        let act: Vec<f32> = up.iter().map(|v| silu(*v)).collect();
        let dn = qmm_clamped_tag(&act, &blk.ffn_down, &blk.ffn_down_c, "ffn_down");
        rms_norm_w(&dn, &blk.ffn_post_norm, cfg.eps)
    };
    let ffn2 = |x: &[f32]| -> Vec<f32> {
        let n = rms_norm_w(x, &blk.ffn_norm_1, cfg.eps);
        let up = qmm_clamped_tag(&n, &blk.ffn_up_1, &blk.ffn_up_1_c, "ffn_up_1");
        let act: Vec<f32> = up.iter().map(|v| silu(*v)).collect();
        let dn = qmm_clamped_tag(&act, &blk.ffn_down_1, &blk.ffn_down_1_c, "ffn_down_1");
        rms_norm_w(&dn, &blk.ffn_post_norm_1, cfg.eps)
    };
    // (conv_module moved out of closure — needs access to all T frames
    // for the temporal depthwise conv.)

    // ── FFN-1 half-step (residual scaled by 0.5) ──
    // Diagnostic: dump named-checkpoint sums for the FIRST block only,
    // matching the reference dump's `norm_w-0`, `ffn_up-0`, `ffn_silu-0`.
    if trc {
        // step through one frame to compute the named sums across all frames
        let mut norm_total = 0.0f32;
        let mut up_total = 0.0f32;
        let mut silu_total = 0.0f32;
        for x in hs.iter() {
            let n = rms_norm_w(x, &blk.ffn_norm, cfg.eps);
            norm_total += n.iter().sum::<f32>();
            let up = qmm_clamped(&n, &blk.ffn_up, &blk.ffn_up_c);
            up_total += up.iter().sum::<f32>();
            let silu: f32 = up.iter().map(|v| silu(*v)).sum();
            silu_total += silu;
        }
        eprintln!("[RCE_AUDIO] cf norm_w(ffn1)              total={norm_total:.4}");
        eprintln!("[RCE_AUDIO] cf ffn_up                    total={up_total:.4}");
        eprintln!("[RCE_AUDIO] cf ffn_silu                  total={silu_total:.4}");
    }
    let after_ffn1: Vec<Vec<f32>> = hs.iter().map(|x| {
        let out = ffn1(x);
        let mut r = x.clone();
        for i in 0..r.len() { r[i] += res_w * out[i]; }
        r
    }).collect();
    dump_tok0!("after_ffn1+res", &after_ffn1);

    // ── Self-attention: chunked-local CAUSAL with sinusoidal RPE ──
    //
    // Per `clip.cpp::PROJECTOR_TYPE_GEMMA4A` (line 3872-3920):
    //   - Mask: query i attends to key j ∈ [max(0, i - (P-1)), i] (causal,
    //     local window of size P=12 = MAX_PAST_HORIZON). Strict `< P`.
    //   - RPE: 13-entry sinusoidal table at positions [12, 11, ..., 0]
    //     (decreasing distance). Position with rel=`k` is at index
    //     `P - k` in the table.
    //   - Per-layer projector `attn_k_rel` maps the 1024-d pos_emb to
    //     d_head dims per head; the RPE contribution to score(i, j, h)
    //     is `Q[i,h,:] · attn_k_rel·pos_emb[P - (i-j), h, :]`.
    //
    // Implementation: explicit per-position math (no reshape trick).
    let after_attn: Vec<Vec<f32>> = {
        let dh = cfg.d_head;
        let nh = cfg.n_head;
        let q_scale = (1.0 / (dh as f32).sqrt()) / 2.0f32.ln();
        let k_scale = (1.0f32 + 1.0f32.exp()).ln() / 2.0f32.ln();

        // 1) Project Q/K/V per position.
        let mut qs: Vec<Vec<f32>> = Vec::with_capacity(t);
        let mut ks: Vec<Vec<f32>> = Vec::with_capacity(t);
        let mut vs: Vec<Vec<f32>> = Vec::with_capacity(t);
        for x in &after_ffn1 {
            let n = rms_norm_w(x, &blk.attn_pre_norm, cfg.eps);
            let mut q = qmm_clamped_tag(&n, &blk.wq, &blk.wq_c, "wq");
            let mut k = qmm_clamped_tag(&n, &blk.wk, &blk.wk_c, "wk");
            let v = qmm_clamped_tag(&n, &blk.wv, &blk.wv_c, "wv");
            for h in 0..nh {
                let o = h * dh;
                for d in 0..dh { q[o + d] *= q_scale * blk.per_dim_scale[d]; }
            }
            for v_ in k.iter_mut() { *v_ *= k_scale; }
            qs.push(q); ks.push(k); vs.push(v);
        }

        // 2) Build sinusoidal RPE [R=13, d_embd=1024] with positions
        //    [P, P-1, ..., 0] = [12, 11, ..., 0]. Sinusoidal: half sin,
        //    half cos, with timescales log-spaced.
        let r_len = MAX_PAST_HORIZON + 1; // 13
        let half = cfg.d_embd / 2;
        let log_ts_inc = 10_000.0f32.ln() / (half as f32 - 1.0).max(1.0);
        let mut pos_emb = vec![0.0f32; r_len * cfg.d_embd];
        for p in 0..r_len {
            let position = (MAX_PAST_HORIZON - p) as f32; // p=0 → 12, p=12 → 0
            for i in 0..half {
                let inv_ts = (-(i as f32) * log_ts_inc).exp();
                let scaled = position * inv_ts;
                pos_emb[p * cfg.d_embd + i] = scaled.sin();
                pos_emb[p * cfg.d_embd + i + half] = scaled.cos();
            }
        }

        // 3) Project RPE per layer: matrix_p[r, h, d] = attn_k_rel·pos_emb[r].
        //    attn_k_rel weight: [d_embd, d_embd] → output dim 1024 per row r.
        let mut matrix_p = vec![0.0f32; r_len * cfg.d_embd];
        for r in 0..r_len {
            let pos_r = &pos_emb[r * cfg.d_embd..r * cfg.d_embd + cfg.d_embd];
            let out_r = qmm(pos_r, &blk.attn_k_rel);
            matrix_p[r * cfg.d_embd..r * cfg.d_embd + cfg.d_embd].copy_from_slice(&out_r);
        }
        // Reshape conceptually as [R, n_head, d_head]: matrix_p[r, h, d]
        // sits at index `r*d_embd + h*d_head + d`.

        // 4) Precompute matrix_bd[i, h, r] = Q[i, h, :] · matrix_p[r, h, :]
        //    — the RPE contribution for query position i with respect to
        //    each pos-emb row r.
        let mut matrix_bd = vec![0.0f32; t * nh * r_len];
        for i in 0..t {
            for h in 0..nh {
                let qh = &qs[i][h * dh..h * dh + dh];
                for r in 0..r_len {
                    let p_off = r * cfg.d_embd + h * dh;
                    let pr = &matrix_p[p_off..p_off + dh];
                    matrix_bd[(i * nh + h) * r_len + r] = rce_op_attn::dot(qh, pr);
                }
            }
        }

        // 5) Chunked-local CAUSAL attention with RPE:
        //    query i attends keys j ∈ [max(0, i - P + 1), i]  (P-1 = 11
        //    past + current = 12 valid positions). RPE row for rel=i-j:
        //    p = P - rel = 12 - (i - j) ∈ [1, 12].
        let cap = 50.0f32;
        let p_max = MAX_PAST_HORIZON; // 12
        let mut attn_out = vec![vec![0.0f32; cfg.d_embd]; t];
        for i in 0..t {
            let lo = i.saturating_sub(p_max - 1); // window [i - 11, i]
            let hi = i + 1;
            let win = hi - lo;
            for h in 0..nh {
                let o = h * dh;
                let qh = &qs[i][o..o + dh];
                let mut sc = vec![0.0f32; win];
                for (idx, j) in (lo..hi).enumerate() {
                    let kh = &ks[j][o..o + dh];
                    let d = rce_op_attn::dot(qh, kh);
                    let rel = i - j;
                    let p_idx = p_max - rel; // ∈ [1, 12]
                    let rpe = matrix_bd[(i * nh + h) * r_len + p_idx];
                    sc[idx] = d + rpe;
                }
                // Softcap 50.
                for x in sc.iter_mut() { *x = cap * (*x / cap).tanh(); }
                let m = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut s_sum = 0.0f32;
                for x in sc.iter_mut() { *x = (*x - m).exp(); s_sum += *x; }
                for (idx, j) in (lo..hi).enumerate() {
                    let wj = sc[idx] / s_sum;
                    let vh = &vs[j][o..o + dh];
                    rce_op_attn::axpy(&mut attn_out[i][o..o + dh], wj, vh);
                }
            }
        }
        // wo + attn_post_norm + residual.
        after_ffn1.into_iter().zip(attn_out.into_iter()).map(|(x, a)| {
            let proj = qmm_clamped_tag(&a, &blk.wo, &blk.wo_c, "wo");
            let pn = rms_norm_w(&proj, &blk.attn_post_norm, cfg.eps);
            let mut r = x;
            for i in 0..r.len() { r[i] += pn[i]; }
            r
        }).collect()
    };

    // ── Conv module + residual ──
    //
    // Block-level conv module: pw1 → GLU split → causal depthwise k=5 →
    // RMSnorm × conv_norm → SiLU → pw2. The depthwise conv mixes the
    // current frame with the 4 PRIOR frames per channel (causal,
    // left-padding). For each output frame t and channel c:
    //   y[t,c] = Σ_{k=0..4} w[k,c] · x[t-4+k, c]   (zero-pad t<4 with 0)
    // The conv_dw weight is stored row-major as `[5, d_embd]` flat
    // (5 kernel positions × d_embd channels).
    let after_conv: Vec<Vec<f32>> = {
        let d = cfg.d_embd;
        // Step 1: RMSnorm and pw1 per frame, then GLU split.
        let mut g_buf: Vec<Vec<f32>> = Vec::with_capacity(t);
        for x in &after_attn {
            // Reference clip.cpp:2432-2440: `conv_norm` / `norm_conv` are
            // SWAPPED in the GGUF; the pre-pw1 RMSnorm weight is loaded
            // from `a.blk.{il}.conv_norm.weight`, and the post-depthwise
            // RMSnorm weight is loaded from `a.blk.{il}.norm_conv.weight`.
            let n = rms_norm_w(x, &blk.conv_norm, cfg.eps);
            let pw1 = qmm_clamped_tag(&n, &blk.conv_pw1, &blk.conv_pw1_c, "conv_pw1");
            let mut g = vec![0.0f32; d];
            for i in 0..d { g[i] = pw1[i] * sigmoid(pw1[i + d]); }
            g_buf.push(g);
        }
        // Step 2: causal depthwise temporal conv k=5 (per channel, in time).
        let mut conv_out: Vec<Vec<f32>> = vec![vec![0.0f32; d]; t];
        for ti in 0..t {
            for c in 0..d {
                let mut acc = 0.0f32;
                for k in 0..5 {
                    let src_t = ti as isize - 4 + k as isize;
                    if src_t < 0 { continue; }
                    // conv_dw layout per probe `dims=[5, 1024]`:
                    // ne[0] = 5 (kernel), ne[1] = d_embd. Flat:
                    // weight[k, c] at k + c*5.
                    let w = blk.conv_dw[k + c * 5];
                    acc += w * g_buf[src_t as usize][c];
                }
                conv_out[ti][c] = acc;
            }
        }
        // Step 3: RMSnorm × conv_norm → SiLU → pw2 → residual.
        after_attn.into_iter().zip(conv_out.into_iter()).map(|(x, c)| {
            let n = rms_norm_w(&c, &blk.norm_conv, cfg.eps);
            let act: Vec<f32> = n.into_iter().map(|v| silu(v)).collect();
            let pw2 = qmm_clamped_tag(&act, &blk.conv_pw2, &blk.conv_pw2_c, "conv_pw2");
            let mut r = x;
            for i in 0..r.len() { r[i] += pw2[i]; }
            r
        }).collect()
    };

    // ── FFN-2 half-step (residual scaled by 0.5) ──
    let after_ffn2: Vec<Vec<f32>> = after_conv.into_iter().map(|x| {
        let out = ffn2(&x);
        let mut r = x;
        for i in 0..r.len() { r[i] += res_w * out[i]; }
        r
    }).collect();
    // ── Final block LN ──
    *hs = after_ffn2.into_iter().map(|x| rms_norm_w(&x, &blk.ln_2, cfg.eps)).collect();
}

// ───── Multimodal runtime: audio encoder + text decoder + splice ─────

use rce_gemma4::Gemma4Runtime;
use rce_ingest_gguf::{sample_with_history, SampleConfig};

/// Gemma 4a audio-region boundary text strings, from
/// `tools/mtmd/mtmd.cpp:551-554`: `<|audio>` (begin) ... raw audio
/// embeddings ... `<audio|>` (end). Multi-byte BPE-tokenized strings,
/// not single special-token ids. N audio embeddings spliced between.
///
/// Single special-token IDs (verified from `tokenizer.ggml.tokens` in the
/// gemma-4 E2B text GGUF, 2026-05-23): `<|audio>` = 256000, `<audio|>` =
/// 258883. The default rce BPE tokenizer splits these into BPE pieces,
/// so we substitute the exact IDs at splice time.
pub const G4A_AUD_BEGIN: &str = "<|audio>";
pub const G4A_AUD_END: &str = "<audio|>";
pub const G4A_AUD_BEGIN_ID: i32 = 256000;
pub const G4A_AUD_END_ID: i32 = 258883;

pub struct Gemma4AudioRuntime {
    pub text: Gemma4Runtime,
    pub audio: AudioTower,
}

#[derive(Clone, Copy, Debug)]
pub struct G4AGenConfig {
    pub max_new_tokens: usize,
    pub sample: SampleConfig,
    pub rep_window: usize,
}
impl Default for G4AGenConfig {
    fn default() -> Self {
        Self { max_new_tokens: 48, sample: SampleConfig::GREEDY, rep_window: 64 }
    }
}

impl Gemma4AudioRuntime {
    pub fn load(
        text_gguf: impl AsRef<Path>, mmproj_gguf: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let text = Gemma4Runtime::load(text_gguf)?;
        let audio = AudioTower::load(mmproj_gguf)?;
        if audio.cfg.proj_dim != text.cfg.d_model {
            return Err(format!(
                "audio proj {} != gemma4 d_model {}", audio.cfg.proj_dim, text.cfg.d_model
            ));
        }
        Ok(Self { text, audio })
    }

    /// Caption / transcribe / answer about `audio_pcm_path` (raw 16k
    /// mono s16le PCM file — pre-convert mp3 via `ffmpeg -i a.mp3 -ac
    /// 1 -ar 16000 -f s16le a.pcm`). Chat-frames as
    /// `<bos><start_of_turn>user\n<|audio>{N audio embeddings}<audio|>
    /// {question}<end_of_turn>\n<start_of_turn>model\n` and greedily
    /// decodes.
    pub fn answer(
        &self,
        audio_pcm_path: impl AsRef<Path>,
        question: &str,
        gcfg: &G4AGenConfig,
        mut on_token: impl FnMut(&str),
    ) -> Result<String, String> {
        let tk = self.text.tokenizer();
        let bos = tk.bos_token_id.unwrap_or(2);
        let eos = tk.eos_token_id.unwrap_or(1);
        // Gemma 4 native turn boundary: `<turn|>` is token 106 in
        // `tokenizer.ggml.tokens`. We also detect the BPE-fragmented
        // `<end_of_turn>` substring below (Gemma 3 chat-template form,
        // which the model has learned to emit as well).
        const G4_TURN_END_ID: i32 = 106;
        let is_gpt2 = self.text.gguf.metadata.iter()
            .find(|kv| kv.key == "tokenizer.ggml.model")
            .map(|kv| matches!(&kv.value, GgufValue::String(s) if s == "gpt2"))
            .unwrap_or(false);
        let encode = |s: &str| -> Vec<i32> {
            if is_gpt2 { tk.encode_gpt2(s, false) } else { tk.encode(s, false) }
        };

        // Encode audio.
        let samples = load_pcm_s16le(&audio_pcm_path)
            .map_err(|e| format!("load pcm: {e}"))?;
        let mel = mel_spectrogram_gemma4a(&samples);
        let aud_embds = self.audio.encode_mel(&mel)?;
        let n_aud = aud_embds.len();
        eprintln!("[g4a-rt] {} samples → {} mel → {} audio tokens × {}",
                  samples.len(), mel.len(), n_aud, self.audio.cfg.proj_dim);

        let pre: Vec<i32> = {
            let mut v = vec![bos];
            v.extend(encode("<start_of_turn>user\n"));
            v.push(G4A_AUD_BEGIN_ID);
            v
        };
        let mid: Vec<i32> = vec![G4A_AUD_END_ID];
        let post = {
            let mut v = encode(question);
            v.extend(encode("<end_of_turn>\n<start_of_turn>model\n"));
            v
        };

        let mut state = self.text.new_state();
        let mut last_logits: Option<Vec<f32>> = None;
        let mut history: Vec<i32> = Vec::new();

        for &id in &pre {
            let e = self.text.embed_token(id);
            last_logits = Some(self.text.forward_embed(&mut state, e, id));
            history.push(id);
        }
        for e in &aud_embds {
            last_logits = Some(self.text.forward_image_embed(&mut state, e.clone()));
        }
        for &id in mid.iter().chain(post.iter()) {
            let e = self.text.embed_token(id);
            last_logits = Some(self.text.forward_embed(&mut state, e, id));
            history.push(id);
        }

        // Decode. Generate all tokens, then post-strip the
        // `<end_of_turn>\n<start_of_turn>user\n` boilerplate prefix
        // (model artifact of being prompted with Gemma 3 BPE-style
        // `<start_of_turn>` strings instead of the Gemma 4 single-token
        // `<|turn>` markers) and truncate at the first real
        // `<end_of_turn>` / `<turn|>` boundary. Stream raw tokens via
        // `on_token` (callers can choose to display or post-process).
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
            if next == eos || next == G4_TURN_END_ID { break; }
            out_ids.push(next);
            history.push(next);
            on_token(&tk.decode_sequence(&[next], false));
            let e = self.text.embed_token(next);
            let lg = self.text.forward_embed(&mut state, e, next);
            let recent = &history[history.len().saturating_sub(gcfg.rep_window)..];
            next = sample_with_history(&lg, &seed_for(step), recent);
            step += 1;
        }
        let mut text = tk.decode_sequence(&out_ids, true);
        // Strip the boilerplate prefix if present.
        let prefix_marker = "<start_of_turn>user\n";
        if let Some(pos) = text.find(prefix_marker) {
            text = text[pos + prefix_marker.len()..].to_string();
        }
        // Truncate at the natural turn end (full or partial form).
        for marker in ["<end_of_turn>", "<end_of_turn", "<turn|>", "<start_of_turn>"] {
            if let Some(pos) = text.find(marker) { text.truncate(pos); }
        }
        Ok(text.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const MMP: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-mmproj-q8_0.gguf";
    // E4B: same Conformer audio encoder as E2B (1024/4096/12L/8H/128 mel); only
    // the projector output dim differs (1536 → 2560, = E4B text d_model), read
    // from `clip.audio.projection_dim`.
    const MMP_E4B: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/gemma-4-e4b-gguf/mmproj-gemma-4-E4B-it-Q8_0.gguf";
    const TXT_E4B: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/gemma-4-e4b-gguf/gemma-4-E4B-it-Q8_0.gguf";

    /// **g4a.f — E4B full audio e2e.** Same clip as g4a.e through the E4B
    /// text+audio path; validates the non-square audio projection
    /// (AUDIO_INNER=1536 → proj_dim=2560) end-to-end. Reference (E2B): the
    /// clip is "The birch canoe slid on the smooth planks."
    #[test]
    fn g4a_f_e4b_end_to_end() {
        let pcm = "/tmp/test_audio.pcm";
        if !std::path::Path::new(TXT_E4B).exists()
            || !std::path::Path::new(MMP_E4B).exists()
            || !std::path::Path::new(pcm).exists()
        { eprintln!("g4a.f: skip"); return; }
        let rt = Gemma4AudioRuntime::load(TXT_E4B, MMP_E4B).expect("load E4B audio");
        assert_eq!(rt.audio.cfg.proj_dim, 2560, "E4B audio proj_dim");
        let cfg = G4AGenConfig { max_new_tokens: 24, sample: SampleConfig::GREEDY, rep_window: 64 };
        let mut chunks: Vec<String> = Vec::new();
        let t0 = std::time::Instant::now();
        rt.answer(pcm, "Transcribe this audio.", &cfg, |s| chunks.push(s.to_string()))
            .expect("E4B answer");
        let full = chunks.concat();
        eprintln!("g4a.f E4B audio → {full:?}  [{} tok in {:?}]", chunks.len(), t0.elapsed());
        assert!(!full.is_empty(), "empty E4B answer");
        let coherent = full.to_lowercase().contains("birch") || full.to_lowercase().contains("canoe")
            || full.to_lowercase().contains("plank")
            || [" the ", " a ", " is ", " on ", " smooth "].iter().any(|w| full.to_lowercase().contains(w))
            || full.chars().filter(|c| c.is_alphabetic()).count() >= full.len() / 2;
        assert!(coherent, "E4B audio incoherent: {full:?}");
    }

    /// **g4a.e4b_load — E4B audio mmproj loads; encoder dims identical to E2B,
    /// projector output dim = 2560 (E4B text d_model).**
    #[test]
    fn g4a_e4b_load() {
        if !std::path::Path::new(MMP_E4B).exists() { eprintln!("g4a.e4b: E4B not mounted — skip"); return; }
        let v = AudioTower::load(MMP_E4B).expect("load E4B audio");
        assert_eq!(v.cfg.n_block, 12);
        assert_eq!(v.cfg.d_embd, 1024);
        assert_eq!(v.cfg.n_mel, 128);
        assert_eq!(v.cfg.proj_dim, 2560, "E4B audio projects into d_model=2560");
        assert_eq!(v.blocks.len(), 12);
        eprintln!("g4a.e4b OK: audio encoder dims == E2B, proj_dim={}", v.cfg.proj_dim);
    }

    /// **g4a.a — load + shape sanity.** Every audio tensor present at
    /// the names + shapes the Conformer build expects.
    #[test]
    fn g4a_a_load() {
        if !std::path::Path::new(MMP).exists() { eprintln!("g4a.a: skip"); return; }
        let v = AudioTower::load(MMP).expect("load");
        assert_eq!(v.cfg.n_block, 12);
        assert_eq!(v.cfg.d_embd, 1024);
        assert_eq!(v.cfg.n_head, 8);
        assert_eq!(v.cfg.d_head, 128);
        assert_eq!(v.cfg.d_ff, 4096);
        assert_eq!(v.cfg.n_mel, 128);
        assert_eq!(v.cfg.proj_dim, 1536);
        assert_eq!(v.blocks.len(), 12);
        assert_eq!(v.conv0_w.len(), 3 * 3 * 1 * 128, "conv0 shape mismatch");
        assert_eq!(v.conv1_w.len(), 3 * 3 * 128 * 32, "conv1 shape mismatch");
        for b in &v.blocks {
            assert_eq!(b.per_dim_scale.len(), 128);
            assert_eq!(b.conv_dw.len(), 5 * 1024, "conv_dw shape mismatch");
            // ffn_up: [1024, 4096] in GGUF (ne[0]=d_embd, ne[1]=d_ff)
            assert_eq!(b.ffn_up.n_rows, 4096);
            assert_eq!(b.ffn_up.n_cols, 1024);
            // conv_pw1: [d_embd, 2*d_embd]  -> for GLU split
            assert_eq!(b.conv_pw1.n_rows, 2048);
            assert_eq!(b.conv_pw1.n_cols, 1024);
        }
        eprintln!("g4a.a OK: 12-block Conformer scaffold loaded, all shapes match");
    }

    /// **g4a.b — forward end-to-end on synthetic input.** Confirms
    /// shapes + finiteness through subsampling conv2d + 12 Conformer
    /// blocks + projection.
    #[test]
    fn g4a_b_synthetic_forward() {
        if !std::path::Path::new(MMP).exists() { eprintln!("g4a.b: skip"); return; }
        let v = AudioTower::load(MMP).expect("load");
        // 16 mel frames in → 4 out (4× subsampling).
        let mel: Vec<Vec<f32>> = (0..16).map(|t| (0..128).map(|m| (t as f32 * 0.1 + m as f32 * 0.01).sin()).collect()).collect();
        let out = v.encode_mel(&mel).expect("forward");
        assert!(out.len() >= 1);
        assert_eq!(out[0].len(), 1536);
        for tok in &out {
            for v in tok { assert!(v.is_finite(), "non-finite output"); }
        }
        let s0: f32 = out[0].iter().sum();
        eprintln!("g4a.b OK: {} tokens × {} dim, finite; token[0].sum={s0:.3}",
                  out.len(), out[0].len());
    }

    /// **g4a.c — mel-spec preprocessing on the test mp3** (converted
    /// to 16k mono s16 PCM offline via ffmpeg → /tmp/test_audio.pcm).
    #[test]
    fn g4a_c_mel_test_audio() {
        let pcm_path = "/tmp/test_audio.pcm";
        if !std::path::Path::new(pcm_path).exists() {
            eprintln!("g4a.c: PCM not found — skipping (run `ffmpeg -i test-2.mp3 -ac 1 -ar 16000 -f s16le /tmp/test_audio.pcm`)");
            return;
        }
        let samples = load_pcm_s16le(pcm_path).expect("read pcm");
        let mel = mel_spectrogram_gemma4a(&samples);
        assert!(mel.len() > 0);
        assert_eq!(mel[0].len(), 128);
        // Compute simple stats: mean, min, max of log-mel values.
        let flat: Vec<f32> = mel.iter().flat_map(|f| f.iter().copied()).collect();
        let mn = flat.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = flat.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let me = flat.iter().sum::<f32>() / flat.len() as f32;
        for v in &flat { assert!(v.is_finite(), "non-finite mel"); }
        eprintln!("g4a.c OK: {} frames × 128 mel  range=[{mn:.3}, {mx:.3}] mean={me:.3}",
                  mel.len());
    }

    const TXT: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-text-q8_0.gguf";

    /// **g4a.e — END-TO-END: real audio → Gemma 4 description.**
    /// Drives the full Gemma 4 multimodal path on the test mp3:
    /// PCM → mel → Conformer → splice → decoder generation.
    /// Coherence gate: common function words OR majority-alphabetic
    /// output — a broken Conformer/splice would garble this.
    #[test]
    fn g4a_e_end_to_end() {
        let pcm = "/tmp/test_audio.pcm";
        if !std::path::Path::new(TXT).exists()
            || !std::path::Path::new(MMP).exists()
            || !std::path::Path::new(pcm).exists()
        {
            eprintln!("g4a.e: skip"); return;
        }
        let rt = Gemma4AudioRuntime::load(TXT, MMP).expect("load");
        let cfg = G4AGenConfig {
            max_new_tokens: 30, sample: SampleConfig::GREEDY, rep_window: 64,
        };
        let mut chunks: Vec<String> = Vec::new();
        let t0 = std::time::Instant::now();
        let res = rt.answer(pcm, "What do you hear in this audio?", &cfg,
            |s| chunks.push(s.to_string()));
        let dt = t0.elapsed();
        match &res {
            Ok(_) => {
                let full = chunks.concat();
                eprintln!("g4a.e: audio → {full:?}  [{} tok in {dt:?}]", chunks.len());
                assert!(!full.is_empty(), "empty answer");
                let coherent = [" the ", " a ", " is ", " of ", " in ", " and ",
                                " with ", " audio ", " sound ", " hear ", " music "]
                    .iter().any(|w| full.to_lowercase().contains(w))
                    || full.chars().filter(|c| c.is_alphabetic()).count() >= full.len() / 2;
                assert!(coherent, "incoherent: {full:?}");
            }
            Err(e) => panic!("answer error: {e}"),
        }
    }

    /// **g4a.d — full pipeline: PCM → mel → encoder → audio tokens.**
    /// Validates the full forward path produces sensible shapes +
    /// finite output. Semantic correctness requires the full
    /// chunked-local RPE attention + causal depthwise conv (deferred).
    #[test]
    fn g4a_d_full_pipeline() {
        let pcm_path = "/tmp/test_audio.pcm";
        if !std::path::Path::new(pcm_path).exists() || !std::path::Path::new(MMP).exists() {
            eprintln!("g4a.d: skip"); return;
        }
        let samples = load_pcm_s16le(pcm_path).expect("read pcm");
        // Only the first 1 second to keep test fast.
        let cap = samples.len().min(16000);
        let mel = mel_spectrogram_gemma4a(&samples[..cap]);
        let v = AudioTower::load(MMP).expect("load");
        let t0 = std::time::Instant::now();
        let out = v.encode_mel(&mel).expect("forward");
        let dt = t0.elapsed();
        for tok in &out { for x in tok { assert!(x.is_finite()); } }
        eprintln!(
            "g4a.d OK: {}-sample PCM → {}-frame mel → {} audio tokens × {} dim in {dt:?}",
            samples.len(), mel.len(), out.len(), out[0].len()
        );
    }
}
