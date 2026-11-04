//! # rce-lfm2-audio — LFM2.5-Audio-1.5B audio encoder
//!
//! The audio mmproj (`arch=clip`, projector `lfm2a`) is a FastConformer,
//! structurally the same as [`rce_canary`]'s encoder. Pipeline:
//!
//! 1. **log-mel** (128 bins, mtmd conformer preprocessor: preemph 0.97 →
//!    zero-center-pad → STFT win-400/hop-160 in a 512-pt FFT → POWER spectrum →
//!    **Slaney** mel (NOT HTK) fmin0/fmax8000 Slaney-area-norm → `log(max(·,
//!    2^-24))` → per-feature z-score (unbiased var+1e-5)).
//! 2. **dw_striding 8× subsampling**: `a.conv1d.{0,2,3,5,6}` 2D convs
//!    (conv→ReLU, depthwise k3 s2, pointwise→ReLU, depthwise k3 s2,
//!    pointwise→ReLU) → flatten 256·16 → `a.pre_encode.out` Linear→512.
//! 3. **17 macaron Conformer blocks** (`a.blk.N`): ½·FFN1(`ffn_*`, swish) →
//!    rel-pos Transformer-XL MHSA (`linear_pos` + per-layer `pos_bias_u/v` +
//!    rel_shift, 8 heads × 64) → conv module (pointwise→GLU→depthwise k9 →
//!    **affine** `conv_norm` (folded BatchNorm, NOT LayerNorm) → swish →
//!    pointwise; pre-norm is LayerNorm `norm_conv`) → ½·FFN2(`ffn_*_1`, swish)
//!    → block LayerNorm. (FFN1 uses the non-`_1` weights, FFN2 the `_1` — order
//!    matters, matches conformer.cpp `build()`.)
//! 4. **projector** `mm.a.mlp`: LN(512) → Linear 512→2048 → GELU(erf) →
//!    Linear 2048→2048, yielding `[T, 2048]` audio embeddings at the LM
//!    `d_model`, ready to splice into [`rce_lfm2`].
//!
//! Loaded from the F16 mmproj GGUF (all tensors dequant uniformly to f32).
//! VERIFIED verbatim vs `llama-mtmd-cli`: "The birch canoe slid on the smooth
//! planks." (final-embedding cosine 0.987 against the reference).

use rce_ingest_gguf::{read_tensor_f32, GgufFile};
use rce_lfm2::Lfm2Runtime;
use memmap2::Mmap;
use rayon::prelude::*;
use rustfft::{num_complex::Complex32, FftPlanner};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

// ── dims (from clip.audio.* metadata) ──────────────────────────────
const D: usize = 512;
const H: usize = 8;
const DK: usize = 64; // D / H
const N_ENC: usize = 17;
const FF: usize = 2048; // macaron FFN hidden
const CONV_K: usize = 9;
const PROJ_DIM: usize = 2048; // LM d_model
const SUB_CH: usize = 256; // subsampling conv channels
const LN_EPS: f32 = 1e-5;

// ── mel (mtmd `mtmd_audio_preprocessor_conformer` = NeMo FastConformer, per
// llama.cpp source: preemph 0.97, POWER spectrum, periodic Hann win=400 in a
// 512-pt FFT, center padding, HTK mel fmin0/fmaxNyquist (Slaney filter norm),
// natural log, per-feature z-score norm var+1e-5). Defaults since the mmproj
// stores no audio_n_fft/window/hop. ──
const N_FFT: usize = 512;
const WIN: usize = 400;
const HOP: usize = 160;
const N_MEL: usize = 128;
const N_BINS: usize = N_FFT / 2 + 1; // 257
const SR: usize = 16000;
const PREEMPH: f32 = 0.97;
const MEL_FLOOR: f32 = 5.960_464_477_539_063e-8; // 2^-24, mtmd filter_params default

#[inline]
fn silu(x: f32) -> f32 {
    if x >= 0.0 { x / (1.0 + (-x).exp()) } else { let e = x.exp(); x * e / (1.0 + e) }
}
#[inline]
fn relu(x: f32) -> f32 { x.max(0.0) }
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) }
}
#[inline]
fn gelu(x: f32) -> f32 {
    // exact erf GELU: 0.5 x (1 + erf(x/√2))
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}
/// Abramowitz–Stegun 7.1.26 erf approximation (max err ~1.5e-7).
fn erf(x: f32) -> f32 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592) * t * (-x * x).exp();
    s * y
}

/// `y[t, dout] = x[t, din] · W[dout, din]^T (+bias)`, parallel over rows.
/// GGUF stores the weight with `din` contiguous (ne0), so a flat
/// `read_tensor_f32` gives exactly `w[o*din + k]` — same convention.
fn linear(x: &[f32], t: usize, w: &[f32], dout: usize, din: usize, bias: Option<&[f32]>) -> Vec<f32> {
    let mut out = vec![0.0f32; t * dout];
    out.par_chunks_mut(dout).enumerate().for_each(|(i, orow)| {
        let xr = &x[i * din..(i + 1) * din];
        for o in 0..dout {
            let wr = &w[o * din..(o + 1) * din];
            let mut acc = bias.map_or(0.0, |b| b[o]);
            for k in 0..din { acc += xr[k] * wr[k]; }
            orow[o] = acc;
        }
    });
    out
}

/// RMSNorm affine — kept for reference. lfm2a's conformer uses LayerNorm
/// (its norms have bias; RMSnorm regressed ASR), unlike gemma4a (RMSnorm).
#[allow(dead_code)]
fn rms_norm(x: &mut [f32], d: usize, w: &[f32], b: &[f32]) {
    x.par_chunks_mut(d).for_each(|row| {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ms + LN_EPS).sqrt();
        for ((v, g), bb) in row.iter_mut().zip(w).zip(b) { *v = *v * inv * *g + *bb; }
    });
}

/// LayerNorm (weight+bias) over last dim `d`, per row.
fn layernorm(x: &mut [f32], d: usize, w: &[f32], b: &[f32]) {
    x.par_chunks_mut(d).for_each(|row| {
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| { let z = v - mean; z * z }).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        for ((v, g), bb) in row.iter_mut().zip(w).zip(b) { *v = (*v - mean) * inv * *g + *bb; }
    });
}

// ── mel frontend ───────────────────────────────────────────────────
// Slaney mel scale (librosa default; mtmd conformer uses use_htk=false):
// linear below 1 kHz, log above. NOT HTK (2595·log10).
fn hz_to_mel(f: f32) -> f32 {
    const MIN_LOG_HZ: f32 = 1000.0;
    const LIN_SLOPE: f32 = 3.0 / 200.0; // 0.015
    let min_log_mel = MIN_LOG_HZ * LIN_SLOPE; // 15.0
    let log_step = 6.4f32.ln() / 27.0;
    if f < MIN_LOG_HZ { f * LIN_SLOPE } else { min_log_mel + (f / MIN_LOG_HZ).ln() / log_step }
}
fn mel_to_hz(m: f32) -> f32 {
    const MIN_LOG_HZ: f32 = 1000.0;
    const LIN_SLOPE: f32 = 3.0 / 200.0;
    let min_log_mel = MIN_LOG_HZ * LIN_SLOPE;
    let log_step = 6.4f32.ln() / 27.0;
    if m < min_log_mel { m / LIN_SLOPE } else { MIN_LOG_HZ * ((m - min_log_mel) * log_step).exp() }
}

fn build_mel_filters() -> Vec<f32> {
    let (mmin, mmax) = (hz_to_mel(0.0), hz_to_mel(SR as f32 / 2.0));
    let fpts: Vec<f32> = (0..N_MEL + 2)
        .map(|i| mel_to_hz(mmin + (mmax - mmin) * i as f32 / (N_MEL + 1) as f32))
        .collect();
    let fft_freqs: Vec<f32> = (0..N_BINS).map(|i| (SR / 2) as f32 * i as f32 / (N_BINS - 1) as f32).collect();
    let mut filt = vec![0.0f32; N_MEL * N_BINS];
    for m in 0..N_MEL {
        let (l, c, r) = (fpts[m], fpts[m + 1], fpts[m + 2]);
        let enorm = 2.0 / (r - l); // Slaney area norm (NeMo/librosa default)
        for (k, &fk) in fft_freqs.iter().enumerate() {
            let w = ((fk - l) / (c - l)).min((r - fk) / (r - c)).max(0.0);
            filt[m * N_BINS + k] = w * enorm;
        }
    }
    filt
}

/// NeMo FastConformer log-mel (the mtmd conformer preprocessor) →
/// `([N_MEL, T] channel-major, T)`. Preemph 0.97 → center-pad → periodic Hann
/// win=400 centered in a 512-pt FFT → POWER spectrum → HTK Slaney mel →
/// natural log → per-feature (per-mel) z-score normalization (var+1e-5).
fn log_mel(audio: &[f32], filters: &[f32]) -> (Vec<f32>, usize) {
    // Exact mtmd order: ZERO center-pad (frame_size/2 each side), THEN preemph
    // IN-PLACE on the real region [pad+1, pad+n) with prev=padded[pad] (first
    // real sample left unchanged). frame_size == N_FFT.
    let pad = N_FFT / 2;
    let n = audio.len();
    let mut padded = vec![0.0f32; n + 2 * pad];
    padded[pad..pad + n].copy_from_slice(audio);
    let mut prev = padded[pad];
    for i in pad + 1..pad + n {
        let cur = padded[i];
        padded[i] = cur - PREEMPH * prev;
        prev = cur;
    }

    let window: Vec<f32> =
        (0..WIN).map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / WIN as f32).cos()).collect();

    let n_frames = 1 + (padded.len() - N_FFT) / HOP;
    let off = (N_FFT - WIN) / 2;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);

    let cols: Vec<Vec<f32>> = (0..n_frames).into_par_iter().map(|f| {
        let mut buf = vec![Complex32::new(0.0, 0.0); N_FFT];
        for i in 0..WIN { buf[off + i] = Complex32::new(padded[f * HOP + i] * window[i], 0.0); }
        fft.process(&mut buf);
        let mut col = vec![0.0f32; N_MEL];
        for m in 0..N_MEL {
            let frow = &filters[m * N_BINS..(m + 1) * N_BINS];
            let mut acc = 0.0f32;
            for k in 0..N_BINS { let p = buf[k].re * buf[k].re + buf[k].im * buf[k].im; acc += frow[k] * p; }
            col[m] = acc.max(MEL_FLOOR).ln(); // ref: log(max(sum, 2^-24)) — clamps low-energy bins

        }
        col
    }).collect();

    let t = n_frames;
    let mut mel = vec![0.0f32; N_MEL * t];
    for (f, col) in cols.iter().enumerate() { for m in 0..N_MEL { mel[m * t + f] = col[m]; } }
    // per-feature (per-mel) z-score normalization over time, unbiased var.
    for m in 0..N_MEL {
        let row = &mut mel[m * t..(m + 1) * t];
        let mean = row.iter().sum::<f32>() / t as f32;
        let var = row.iter().map(|v| { let z = v - mean; z * z }).sum::<f32>() / (t.max(2) - 1) as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for v in row.iter_mut() { *v = (*v - mean) * inv; }
    }
    (mel, t)
}

// ── 2D conv (subsampling) ──────────────────────────────────────────
#[allow(clippy::too_many_arguments)]
fn conv2d(
    inp: &[f32], cin: usize, h: usize, w: usize, weight: &[f32], cout: usize,
    kh: usize, kw: usize, stride: usize, pad: usize, groups: usize, bias: &[f32],
) -> (Vec<f32>, usize, usize) {
    let ho = (h + 2 * pad - kh) / stride + 1;
    let wo = (w + 2 * pad - kw) / stride + 1;
    let cin_g = cin / groups;
    let cout_g = cout / groups;
    let mut out = vec![0.0f32; cout * ho * wo];
    out.par_chunks_mut(ho * wo).enumerate().for_each(|(oc, oplane)| {
        let g = oc / cout_g;
        let wbase = oc * cin_g * kh * kw;
        for oh in 0..ho {
            for ow in 0..wo {
                let mut acc = bias[oc];
                for icg in 0..cin_g {
                    let ic = g * cin_g + icg;
                    for ph in 0..kh {
                        let ih = (oh * stride + ph) as isize - pad as isize;
                        if ih < 0 || ih as usize >= h { continue; }
                        for pw in 0..kw {
                            let iw = (ow * stride + pw) as isize - pad as isize;
                            if iw < 0 || iw as usize >= w { continue; }
                            let wv = weight[wbase + icg * kh * kw + ph * kw + pw];
                            acc += wv * inp[ic * h * w + ih as usize * w + iw as usize];
                        }
                    }
                }
                oplane[oh * wo + ow] = acc;
            }
        }
    });
    (out, ho, wo)
}

// ── Conformer block weights ────────────────────────────────────────
struct EncLayer {
    nff1_w: Vec<f32>, nff1_b: Vec<f32>, ff1_1w: Vec<f32>, ff1_1b: Vec<f32>, ff1_2w: Vec<f32>, ff1_2b: Vec<f32>,
    nsa_w: Vec<f32>, nsa_b: Vec<f32>,
    q_w: Vec<f32>, q_b: Vec<f32>, k_w: Vec<f32>, k_b: Vec<f32>, v_w: Vec<f32>, v_b: Vec<f32>,
    out_w: Vec<f32>, out_b: Vec<f32>, pos_w: Vec<f32>, pbu: Vec<f32>, pbv: Vec<f32>,
    nconv_w: Vec<f32>, nconv_b: Vec<f32>,
    pw1_w: Vec<f32>, pw1_b: Vec<f32>, dw_w: Vec<f32>, dw_b: Vec<f32>,
    ncn_w: Vec<f32>, ncn_b: Vec<f32>, pw2_w: Vec<f32>, pw2_b: Vec<f32>,
    nff2_w: Vec<f32>, nff2_b: Vec<f32>, ff2_1w: Vec<f32>, ff2_1b: Vec<f32>, ff2_2w: Vec<f32>, ff2_2b: Vec<f32>,
    nout_w: Vec<f32>, nout_b: Vec<f32>,
}

pub struct Lfm2AudioEncoder {
    // subsampling
    s0w: Vec<f32>, s0b: Vec<f32>, s2w: Vec<f32>, s2b: Vec<f32>, s3w: Vec<f32>, s3b: Vec<f32>,
    s5w: Vec<f32>, s5b: Vec<f32>, s6w: Vec<f32>, s6b: Vec<f32>, pre_w: Vec<f32>, pre_b: Vec<f32>,
    enc: Vec<EncLayer>,
    // projector mm.a.mlp: LN(512) → Linear 512→2048 → GELU → Linear 2048→2048
    mlp0_w: Vec<f32>, mlp0_b: Vec<f32>, mlp1_w: Vec<f32>, mlp1_b: Vec<f32>, mlp3_w: Vec<f32>, mlp3_b: Vec<f32>,
    mel_filters: Vec<f32>,
    _mmap: Arc<Mmap>,
}

impl Lfm2AudioEncoder {
    pub fn load(mmproj_path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(mmproj_path.as_ref()).map_err(|e| format!("open: {e}"))?;
        let mmap = Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?);
        let gguf = GgufFile::parse(&mmap[..]).map_err(|e| format!("parse: {e:?}"))?;
        let bytes: &[u8] = &mmap[..];
        let g = |n: &str| read_tensor_f32(bytes, &gguf, n).map_err(|e| format!("{n}: {e:?}"));

        let enc = (0..N_ENC).map(|i| {
            let p = format!("a.blk.{i}");
            Ok(EncLayer {
                // FFN1 (before attention) = GGUF ffn_norm/ffn_up/ffn_down (NO _1),
                // FFN2 (after conv) = ffn_norm_1/... — per conformer.cpp build() order.
                nff1_w: g(&format!("{p}.ffn_norm.weight"))?, nff1_b: g(&format!("{p}.ffn_norm.bias"))?,
                ff1_1w: g(&format!("{p}.ffn_up.weight"))?, ff1_1b: g(&format!("{p}.ffn_up.bias"))?,
                ff1_2w: g(&format!("{p}.ffn_down.weight"))?, ff1_2b: g(&format!("{p}.ffn_down.bias"))?,
                nsa_w: g(&format!("{p}.ln1.weight"))?, nsa_b: g(&format!("{p}.ln1.bias"))?,
                q_w: g(&format!("{p}.attn_q.weight"))?, q_b: g(&format!("{p}.attn_q.bias"))?,
                k_w: g(&format!("{p}.attn_k.weight"))?, k_b: g(&format!("{p}.attn_k.bias"))?,
                v_w: g(&format!("{p}.attn_v.weight"))?, v_b: g(&format!("{p}.attn_v.bias"))?,
                out_w: g(&format!("{p}.attn_out.weight"))?, out_b: g(&format!("{p}.attn_out.bias"))?,
                pos_w: g(&format!("{p}.linear_pos.weight"))?,
                pbu: g(&format!("{p}.pos_bias_u"))?, pbv: g(&format!("{p}.pos_bias_v"))?,
                nconv_w: g(&format!("{p}.conv_norm.weight"))?, nconv_b: g(&format!("{p}.conv_norm.bias"))?,
                pw1_w: g(&format!("{p}.conv_pw1.weight"))?, pw1_b: g(&format!("{p}.conv_pw1.bias"))?,
                dw_w: g(&format!("{p}.conv_dw.weight"))?, dw_b: g(&format!("{p}.conv_dw.bias"))?,
                ncn_w: g(&format!("{p}.norm_conv.weight"))?, ncn_b: g(&format!("{p}.norm_conv.bias"))?,
                pw2_w: g(&format!("{p}.conv_pw2.weight"))?, pw2_b: g(&format!("{p}.conv_pw2.bias"))?,
                nff2_w: g(&format!("{p}.ffn_norm_1.weight"))?, nff2_b: g(&format!("{p}.ffn_norm_1.bias"))?,
                ff2_1w: g(&format!("{p}.ffn_up_1.weight"))?, ff2_1b: g(&format!("{p}.ffn_up_1.bias"))?,
                ff2_2w: g(&format!("{p}.ffn_down_1.weight"))?, ff2_2b: g(&format!("{p}.ffn_down_1.bias"))?,
                nout_w: g(&format!("{p}.ln2.weight"))?, nout_b: g(&format!("{p}.ln2.bias"))?,
            })
        }).collect::<Result<Vec<_>, String>>()?;

        Ok(Self {
            s0w: g("a.conv1d.0.weight")?, s0b: g("a.conv1d.0.bias")?,
            s2w: g("a.conv1d.2.weight")?, s2b: g("a.conv1d.2.bias")?,
            s3w: g("a.conv1d.3.weight")?, s3b: g("a.conv1d.3.bias")?,
            s5w: g("a.conv1d.5.weight")?, s5b: g("a.conv1d.5.bias")?,
            s6w: g("a.conv1d.6.weight")?, s6b: g("a.conv1d.6.bias")?,
            pre_w: g("a.pre_encode.out.weight")?, pre_b: g("a.pre_encode.out.bias")?,
            enc,
            mlp0_w: g("mm.a.mlp.0.weight")?, mlp0_b: g("mm.a.mlp.0.bias")?,
            mlp1_w: g("mm.a.mlp.1.weight")?, mlp1_b: g("mm.a.mlp.1.bias")?,
            mlp3_w: g("mm.a.mlp.3.weight")?, mlp3_b: g("mm.a.mlp.3.bias")?,
            mel_filters: build_mel_filters(),
            _mmap: mmap,
        })
    }

    /// Encode raw 16k mono audio → `([T*PROJ_DIM] row-major, T)` audio
    /// embeddings at the LM `d_model`, ready to splice into rce-lfm2.
    pub fn encode(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        let (mel, tf) = log_mel(audio, &self.mel_filters); // [N_MEL, T]
        // image [1, H=T, W=128]
        let mut img = vec![0.0f32; tf * N_MEL];
        for m in 0..N_MEL { for t in 0..tf { img[t * N_MEL + m] = mel[m * tf + t]; } }

        let (c0, h0, w0) = conv2d(&img, 1, tf, N_MEL, &self.s0w, SUB_CH, 3, 3, 2, 1, 1, &self.s0b);
        let c0: Vec<f32> = c0.iter().map(|x| relu(*x)).collect();
        let (c2, h2, w2) = conv2d(&c0, SUB_CH, h0, w0, &self.s2w, SUB_CH, 3, 3, 2, 1, SUB_CH, &self.s2b);
        let (c3, h3, w3) = conv2d(&c2, SUB_CH, h2, w2, &self.s3w, SUB_CH, 1, 1, 1, 0, 1, &self.s3b);
        let c3: Vec<f32> = c3.iter().map(|x| relu(*x)).collect();
        let (c5, h5, w5) = conv2d(&c3, SUB_CH, h3, w3, &self.s5w, SUB_CH, 3, 3, 2, 1, SUB_CH, &self.s5b);
        let (c6, h6, w6) = conv2d(&c5, SUB_CH, h5, w5, &self.s6w, SUB_CH, 1, 1, 1, 0, 1, &self.s6b);
        let c6: Vec<f32> = c6.iter().map(|x| relu(*x)).collect();

        let (t, freq) = (h6, w6);
        let flat_dim = SUB_CH * freq;
        let mut flat = vec![0.0f32; t * flat_dim];
        for c in 0..SUB_CH {
            for tt in 0..t {
                for f in 0..freq { flat[tt * flat_dim + c * freq + f] = c6[c * h6 * w6 + tt * w6 + f]; }
            }
        }
        let mut hidden = linear(&flat, t, &self.pre_w, D, flat_dim, Some(&self.pre_b));

        let pe = rel_pos_encoding(t);
        for layer in &self.enc {
            let mut xn = hidden.clone();
            layernorm(&mut xn, D, &layer.nff1_w, &layer.nff1_b);
            let ff = ff_swish(&xn, t, &layer.ff1_1w, &layer.ff1_1b, &layer.ff1_2w, &layer.ff1_2b);
            for i in 0..hidden.len() { hidden[i] += 0.5 * ff[i]; }

            let mut xn = hidden.clone();
            layernorm(&mut xn, D, &layer.nsa_w, &layer.nsa_b);
            let attn = rel_pos_attn(&xn, t, &pe, layer);
            for i in 0..hidden.len() { hidden[i] += attn[i]; }

            // Conv-module pre-norm = LayerNorm with GGUF `norm_conv` (ncn), per
            // conformer.cpp `build_norm(residual, layer.norm_conv_w, NORM_NORMAL)`.
            // (Post-depthwise uses `conv_norm`/nconv as pure AFFINE — see conv_module.)
            let mut xn = hidden.clone();
            layernorm(&mut xn, D, &layer.ncn_w, &layer.ncn_b);
            let cv = conv_module(&xn, t, layer);
            for i in 0..hidden.len() { hidden[i] += cv[i]; }

            let mut xn = hidden.clone();
            layernorm(&mut xn, D, &layer.nff2_w, &layer.nff2_b);
            let ff = ff_swish(&xn, t, &layer.ff2_1w, &layer.ff2_1b, &layer.ff2_2w, &layer.ff2_2b);
            for i in 0..hidden.len() { hidden[i] += 0.5 * ff[i]; }

            layernorm(&mut hidden, D, &layer.nout_w, &layer.nout_b);
        }

        // projector: LN(512) → Linear 512→2048 → GELU → Linear 2048→2048
        layernorm(&mut hidden, D, &self.mlp0_w, &self.mlp0_b);
        let mut h1 = linear(&hidden, t, &self.mlp1_w, PROJ_DIM, D, Some(&self.mlp1_b));
        for v in h1.iter_mut() { *v = gelu(*v); }
        let proj = linear(&h1, t, &self.mlp3_w, PROJ_DIM, PROJ_DIM, Some(&self.mlp3_b));
        (proj, t)
    }
}

impl Lfm2AudioEncoder {
    /// Transcribe/answer about `audio` using the LFM2.5-Audio LM backbone
    /// (`lm`, an `arch=lfm2` Lfm2Runtime). Encodes audio → splices the
    /// `[T, 2048]` embeddings between `<|audio_start|>` and the question text
    /// in a ChatML user turn, then greedy-decodes. Markers (main LM
    /// tokenizer): bos `<|startoftext|>`=1, `<|im_start|>`=6, `<|im_end|>`=7,
    /// `<|audio_start|>`=128.
    pub fn transcribe(&self, lm: &Lfm2Runtime, audio: &[f32], question: &str, max_new: usize) -> String {
        self.transcribe_opts(lm, audio, question, max_new, false)
    }

    /// As [`Self::transcribe`], but `zero_audio` replaces the audio
    /// embeddings with zeros — a diagnostic to tell whether the LM is
    /// conditioning on the audio at all (output should change vs real audio
    /// if it is) vs. echoing the prompt regardless (frame/position fault).
    /// An empty `question` omits the instruction text (canonical ASR frame).
    pub fn transcribe_opts(&self, lm: &Lfm2Runtime, audio: &[f32], question: &str, max_new: usize, zero_audio: bool) -> String {
        self.transcribe_full(lm, audio, question, max_new, zero_audio, true)
    }

    /// `q_before`: place the instruction text before the audio (mtmd order)
    /// vs after. With the question after the audio the model tends to echo it;
    /// before, it conditions on the instruction then the audio.
    pub fn transcribe_full(&self, lm: &Lfm2Runtime, audio: &[f32], question: &str, max_new: usize, zero_audio: bool, q_before: bool) -> String {
        let (emb, t) = self.encode(audio);
        let tk = lm.tokenizer();
        let eos = tk.eos_token_id.unwrap_or(7);
        let mut state = lm.new_state();

        // Build the input-embedding stream: text ids → embed_token; audio
        // frames → encoder rows (already at d_model=2048).
        let mut feed: Vec<Vec<f32>> = Vec::new();
        for &id in &[1i32, 6] { feed.push(lm.embed_token(id)); }       // bos, im_start
        for id in tk.encode("user\n", false) { feed.push(lm.embed_token(id)); }
        if q_before && !question.is_empty() {
            for id in tk.encode(question, false) { feed.push(lm.embed_token(id)); }
            for id in tk.encode("\n", false) { feed.push(lm.embed_token(id)); }
        }
        // NOTE: LFM2A uses NO audio marker tokens (mtmd sets aud_beg/aud_end
        // empty for PROJECTOR_TYPE_LFM2A) — the audio embeddings are spliced
        // directly into the user content, unwrapped.
        for f in 0..t {
            if zero_audio { feed.push(vec![0.0f32; PROJ_DIM]); }
            else { feed.push(emb[f * PROJ_DIM..(f + 1) * PROJ_DIM].to_vec()); }
        }
        if !q_before && !question.is_empty() {
            for id in tk.encode(question, false) { feed.push(lm.embed_token(id)); }
        }
        feed.push(lm.embed_token(7));                                   // <|im_end|>
        for id in tk.encode("\n", false) { feed.push(lm.embed_token(id)); }
        feed.push(lm.embed_token(6));                                   // <|im_start|>
        for id in tk.encode("assistant\n", false) { feed.push(lm.embed_token(id)); }

        let mut logits = vec![0.0f32; 1];
        for x in feed { logits = lm.forward_embed(&mut state, x); }

        let mut out: Vec<i32> = Vec::new();
        for _ in 0..max_new {
            let next = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as i32).unwrap();
            if next == eos || next == 7 { break; }
            out.push(next);
            logits = lm.forward_embed(&mut state, lm.embed_token(next));
        }
        tk.decode_sequence(&out, true)
    }
}

fn ff_swish(x: &[f32], t: usize, w1: &[f32], b1: &[f32], w2: &[f32], b2: &[f32]) -> Vec<f32> {
    let mut h = linear(x, t, w1, FF, D, Some(b1));
    for z in h.iter_mut() { *z = silu(*z); }
    linear(&h, t, w2, D, FF, Some(b2))
}

/// Relative sinusoidal positional encoding `[2T-1, D]`.
fn rel_pos_encoding(t: usize) -> Vec<f32> {
    let l = 2 * t - 1;
    let mut pe = vec![0.0f32; l * D];
    let logb = (10000f32).ln();
    for m in 0..l {
        let rel = (t as f32 - 1.0) - m as f32;
        for i in 0..D / 2 {
            let div = (-logb * (2 * i) as f32 / D as f32).exp();
            let a = rel * div;
            pe[m * D + 2 * i] = a.sin();
            pe[m * D + 2 * i + 1] = a.cos();
        }
    }
    pe
}

/// Transformer-XL relative-shift `[T, 2T-1] → [T, T]`.
fn rel_shift(bd: &[f32], t: usize) -> Vec<f32> {
    let l = 2 * t - 1;
    let mut padded = vec![0.0f32; t * (l + 1)];
    for i in 0..t { for j in 0..l { padded[i * (l + 1) + (j + 1)] = bd[i * l + j]; } }
    let nf = &padded[t..];
    let mut out = vec![0.0f32; t * t];
    for i in 0..t { for m in 0..t { out[i * t + m] = nf[i * l + m]; } }
    out
}

/// Conformer rel-pos MHSA (Transformer-XL), parallel over heads.
fn rel_pos_attn(xn: &[f32], t: usize, pe: &[f32], layer: &EncLayer) -> Vec<f32> {
    let q = linear(xn, t, &layer.q_w, D, D, Some(&layer.q_b));
    let k = linear(xn, t, &layer.k_w, D, D, Some(&layer.k_b));
    let v = linear(xn, t, &layer.v_w, D, D, Some(&layer.v_b));
    let p = linear(pe, 2 * t - 1, &layer.pos_w, D, D, None);
    let l = 2 * t - 1;
    let sc = 1.0 / (DK as f32).sqrt();

    let heads: Vec<Vec<f32>> = (0..H).into_par_iter().map(|h| {
        // pos_bias_u/v are [DK, H] in GGUF (ne0=DK contiguous) → head h at stride? stored [64,8]:
        // ne0=64 (DK), ne1=8 (H) → element (d,h) at h*DK + d.
        let pbu = &layer.pbu[h * DK..(h + 1) * DK];
        let pbv = &layer.pbv[h * DK..(h + 1) * DK];
        let mut bd_raw = vec![0.0f32; t * l];
        for i in 0..t {
            let qi = &q[i * D + h * DK..i * D + h * DK + DK];
            for m in 0..l {
                let pm = &p[m * D + h * DK..m * D + h * DK + DK];
                let mut s = 0.0f32; for d in 0..DK { s += (qi[d] + pbv[d]) * pm[d]; }
                bd_raw[i * l + m] = s;
            }
        }
        let bd = rel_shift(&bd_raw, t);
        let mut oh = vec![0.0f32; t * DK];
        for i in 0..t {
            let qi = &q[i * D + h * DK..i * D + h * DK + DK];
            let mut scores = vec![0.0f32; t];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..t {
                let kj = &k[j * D + h * DK..j * D + h * DK + DK];
                let mut ac = 0.0f32; for d in 0..DK { ac += (qi[d] + pbu[d]) * kj[d]; }
                let s = (ac + bd[i * t + j]) * sc;
                scores[j] = s; if s > mx { mx = s; }
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
            let inv = 1.0 / sum;
            let o = &mut oh[i * DK..(i + 1) * DK];
            for j in 0..t {
                let w = scores[j] * inv;
                let vj = &v[j * D + h * DK..j * D + h * DK + DK];
                for d in 0..DK { o[d] += w * vj[d]; }
            }
        }
        oh
    }).collect();

    let mut attn = vec![0.0f32; t * D];
    for h in 0..H {
        for i in 0..t {
            attn[i * D + h * DK..i * D + h * DK + DK].copy_from_slice(&heads[h][i * DK..(i + 1) * DK]);
        }
    }
    linear(&attn, t, &layer.out_w, D, D, Some(&layer.out_b))
}

/// Conformer conv module (per conformer.cpp): pw1 → GLU(sigmoid) → CAUSAL
/// depthwise(k9, left-pad K-1) → AFFINE(`conv_norm`: x*w+b, folded BatchNorm,
/// NO normalization) → SiLU → pw2.
fn conv_module(xn: &[f32], t: usize, layer: &EncLayer) -> Vec<f32> {
    let pw1 = linear(xn, t, &layer.pw1_w, 2 * D, D, Some(&layer.pw1_b));
    let mut glu = vec![0.0f32; t * D];
    for tt in 0..t {
        for c in 0..D { glu[tt * D + c] = pw1[tt * 2 * D + c] * sigmoid(pw1[tt * 2 * D + D + c]); }
    }
    // CENTERED depthwise: y[t,c] = Σ_{k=0..K-1} w[k,c]·x[t-(K-1)/2+k]. conformer.cpp
    // does pad(4)→roll(4)→pad(4)→ssm_conv = left-4 + right-4 = symmetric pad-4.
    let pad = (CONV_K - 1) / 2;
    let mut dw = vec![0.0f32; D * t]; // [C, T]
    dw.par_chunks_mut(t).enumerate().for_each(|(c, drow)| {
        let wbase = c * CONV_K;
        for tt in 0..t {
            let mut acc = layer.dw_b[c];
            for kk in 0..CONV_K {
                let idx = tt as isize + kk as isize - pad as isize;
                if idx >= 0 && (idx as usize) < t { acc += layer.dw_w[wbase + kk] * glu[idx as usize * D + c]; }
            }
            drow[tt] = acc;
        }
    });
    // Post-depthwise = AFFINE with `conv_norm` (nconv): x*w+b, NO normalization
    // (folded BatchNorm — conformer.cpp `ggml_add(ggml_mul(x, conv_norm_w), conv_norm_b)`).
    // Then SiLU. Back to [t, D].
    let mut act = vec![0.0f32; t * D];
    for tt in 0..t {
        for c in 0..D {
            let y = dw[c * t + tt] * layer.nconv_w[c] + layer.nconv_b[c];
            act[tt * D + c] = silu(y);
        }
    }
    linear(&act, t, &layer.pw2_w, D, D, Some(&layer.pw2_b))
}

#[cfg(test)]
mod tests {
    use super::*;
    const MMP: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/lfm2.5-audio-1.5b-gguf/mmproj-LFM2.5-Audio-1.5B-F16.gguf";
    const LM: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/lfm2.5-audio-1.5b-gguf/LFM2.5-Audio-1.5B-Q8_0.gguf";
    const PCM: &str = "/tmp/test_audio.pcm";

    fn read_pcm(p: &str) -> Option<Vec<f32>> {
        let b = std::fs::read(p).ok()?;
        Some(b.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect())
    }

    #[test]
    fn lfm2a_load() {
        if !std::path::Path::new(MMP).exists() { eprintln!("lfm2a: not mounted — skip"); return; }
        let m = Lfm2AudioEncoder::load(MMP).expect("load");
        assert_eq!(m.enc.len(), N_ENC);
        assert_eq!(m.enc[0].q_w.len(), D * D);
        assert_eq!(m.enc[0].dw_w.len(), CONV_K * D);
        assert_eq!(m.mlp1_w.len(), PROJ_DIM * D);
        eprintln!("lfm2a load OK: {N_ENC} conformer blocks, D={D}");
    }

    #[test]
    fn lfm2a_encode_shape_finite() {
        if !std::path::Path::new(MMP).exists() { eprintln!("lfm2a: not mounted — skip"); return; }
        let Some(audio) = read_pcm(PCM) else { eprintln!("lfm2a: {PCM} missing — skip"); return; };
        let m = Lfm2AudioEncoder::load(MMP).expect("load");
        let t0 = std::time::Instant::now();
        let (emb, t) = m.encode(&audio);
        eprintln!(
            "lfm2a encode OK: {:.1}s audio → {t} frames × {PROJ_DIM}  [{:?}]",
            audio.len() as f32 / 16000.0, t0.elapsed(),
        );
        assert_eq!(emb.len(), t * PROJ_DIM);
        assert!(t > 0, "no frames");
        assert!(emb.iter().all(|v| v.is_finite()), "non-finite audio embeddings");
        let max_abs = emb.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        assert!(max_abs > 0.0 && max_abs < 1e3, "embeddings look wrong: max|v|={max_abs}");
    }

    /// **lfm2a ASR e2e** — encode + splice into the lfm2 LM backbone +
    /// transcribe. Verbatim-correct vs llama-mtmd-cli: "The birch canoe slid on
    /// the smooth planks." (embedding cosine 0.987). Asserts the key content
    /// words to lock in the encoder fixes (Slaney mel, conv affine, FFN order).
    #[test]
    fn lfm2a_transcribe_e2e() {
        if !std::path::Path::new(MMP).exists() || !std::path::Path::new(LM).exists() { eprintln!("lfm2a: skip"); return; }
        let Some(audio) = read_pcm(PCM) else { eprintln!("lfm2a: {PCM} missing — skip"); return; };
        let enc = Lfm2AudioEncoder::load(MMP).expect("load encoder");
        let lm = Lfm2Runtime::load(LM).expect("load LM");
        let t0 = std::time::Instant::now();
        let text = enc.transcribe(&lm, &audio, "Transcribe this audio.", 24);
        eprintln!("lfm2a transcribe → {text:?}  [{:?}]  (ref: \"The birch canoe slid on the smooth planks.\")", t0.elapsed());
        let low = text.to_lowercase();
        assert!(!text.is_empty(), "empty transcription");
        for w in ["birch", "canoe", "smooth", "planks"] {
            assert!(low.contains(w), "transcription missing '{w}': {text:?}");
        }
    }

    /// Disambiguation diagnostic (frame fault vs encoder fault): run the
    /// splice 3 ways. If (a) real+Q and (c) zeroed-audio give the SAME output
    /// → the LM isn't conditioning on the audio (frame/position fault). If
    /// they DIFFER → audio is read but the encoder output is wrong. (b) is the
    /// canonical audio-only ASR frame.
    #[test]
    #[ignore]
    fn lfm2a_diag_frame_vs_encoder() {
        if !std::path::Path::new(MMP).exists() || !std::path::Path::new(LM).exists() { eprintln!("skip"); return; }
        let Some(audio) = read_pcm(PCM) else { eprintln!("{PCM} missing"); return; };
        let enc = Lfm2AudioEncoder::load(MMP).expect("enc");
        let lm = Lfm2Runtime::load(LM).expect("lm");
        let a = enc.transcribe_opts(&lm, &audio, "Transcribe this audio.", 24, false);
        let b = enc.transcribe_opts(&lm, &audio, "", 24, false);
        let c = enc.transcribe_opts(&lm, &audio, "Transcribe this audio.", 24, true);
        eprintln!("(a) real+Q     → {a:?}");
        eprintln!("(b) audio-only → {b:?}");
        eprintln!("(c) zeroed-aud → {c:?}");
        eprintln!("VERDICT: a==c ? {}  (true ⇒ frame/position fault; false ⇒ encoder-output fault)", a == c);
    }

    /// Frame tuning — the encoder works (hears the audio), so find the frame
    /// that makes it TRANSCRIBE. Reference: "The birch canoe slid on the
    /// smooth planks." Tries instruction-before-audio with a few phrasings.
    #[test]
    #[ignore]
    fn lfm2a_diag_q_before() {
        if !std::path::Path::new(MMP).exists() || !std::path::Path::new(LM).exists() { eprintln!("skip"); return; }
        let Some(audio) = read_pcm(PCM) else { eprintln!("{PCM} missing"); return; };
        let enc = Lfm2AudioEncoder::load(MMP).expect("enc");
        let lm = Lfm2Runtime::load(LM).expect("lm");
        for q in ["Transcribe this audio.", "Transcribe the speech.", "Repeat what is said in the audio."] {
            let r = enc.transcribe_full(&lm, &audio, q, 30, false, true);
            eprintln!("q_before {q:?} → {r:?}");
        }
    }

    /// Numerical diff of rce encode() vs the llama.cpp reference embeddings
    /// dumped to /tmp/ref_emb.bin (header: i32 ne0,ne1; then ne0*ne1 f32,
    /// layout ref[token*ne0 + d]). Pinpoints whether the encoder is correct.
    #[test]
    #[ignore]
    fn lfm2a_emb_vs_ref() {
        if !std::path::Path::new(MMP).exists() { eprintln!("skip"); return; }
        let Some(audio) = read_pcm(PCM) else { eprintln!("{PCM} missing"); return; };
        let enc = Lfm2AudioEncoder::load(MMP).expect("enc");
        let (emb, t) = enc.encode(&audio);
        let stat = |v: &[f32]| {
            let n = v.len() as f64;
            let m = v.iter().map(|&x| x as f64).sum::<f64>() / n;
            let var = v.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / n;
            (m, var.sqrt(), v.iter().cloned().fold(f32::MAX, f32::min), v.iter().cloned().fold(f32::MIN, f32::max))
        };
        let (m, sd, mn, mx) = stat(&emb);
        eprintln!("RCE_EMB shape=[{PROJ_DIM},{t}] mean={m:.6} std={sd:.6} min={mn:.6} max={mx:.6}");
        eprint!("RCE_EMB tok0[0..16]:"); for i in 0..16 { eprint!(" {:.5}", emb[i]); }
        eprint!("\nRCE_EMB tok0[last16]:"); for i in PROJ_DIM-16..PROJ_DIM { eprint!(" {:.5}", emb[i]); } eprintln!();

        let Ok(bytes) = std::fs::read("/tmp/ref_emb.bin") else { eprintln!("no /tmp/ref_emb.bin — run llama-mtmd-cli first"); return; };
        let ne0 = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let ne1 = i32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let refv: Vec<f32> = bytes[8..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        eprintln!("REF shape=[{ne0},{ne1}]  rce=[{PROJ_DIM},{t}]");
        if ne0 != PROJ_DIM { eprintln!("DIM MISMATCH"); return; }
        let nt = t.min(ne1);
        // per-token cosine similarity + relative L2
        let mut cos_sum = 0.0f64; let mut rel_sum = 0.0f64;
        for tok in 0..nt {
            let a = &emb[tok*PROJ_DIM..(tok+1)*PROJ_DIM];
            let b = &refv[tok*ne0..(tok+1)*ne0];
            let (mut dot, mut na, mut nb, mut diff) = (0.0f64,0.0f64,0.0f64,0.0f64);
            for i in 0..ne0 { dot += a[i] as f64*b[i] as f64; na += (a[i] as f64).powi(2); nb += (b[i] as f64).powi(2); diff += (a[i] as f64-b[i] as f64).powi(2); }
            cos_sum += dot/(na.sqrt()*nb.sqrt()+1e-9); rel_sum += (diff.sqrt())/(nb.sqrt()+1e-9);
        }
        eprintln!("PER-TOKEN avg cosine={:.4}  avg relL2={:.4}  (over {nt} tokens)", cos_sum/nt as f64, rel_sum/nt as f64);
        eprint!("REF   tok0[0..16]:"); for i in 0..16 { eprint!(" {:.5}", refv[i]); } eprintln!();
    }
}
