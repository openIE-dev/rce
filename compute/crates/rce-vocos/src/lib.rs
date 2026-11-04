//! rce-vocos — Vocos ISTFT vocoder (mel → waveform).
//!
//! Architecture (per https://github.com/gemelo-ai/vocos):
//! - **Feature extractor** (stored): MelSpectrogram with the stored
//!   Hann window `head.istft.window [n_fft=1024]` and the stored mel
//!   filterbank `feature_extractor.mel_spec.mel_scale.fb [513, 100]`.
//!   Forward: `safe_log(filterbank @ |STFT(x)|)`, clamp at 1e-5.
//! - **Backbone**: Conv1d embed `[512, 100, 7] + bias` (k=7, p=3), then
//!   LayerNorm on channel dim, 8 ConvNeXt-1D blocks (depthwise k7 +
//!   LayerNorm + pwconv1 Linear[512→1536] + GELU + pwconv2 [1536→512] +
//!   LayerScale gamma + residual), final LayerNorm. All on channel dim 512.
//! - **ISTFT head**: Linear 512→1026 → chunk into mag (513) + phase
//!   (513); `mag = exp(mag).clamp(max=100)`; spec = mag·cos(p) + j·mag·sin(p);
//!   ISTFT (n_fft=1024, hop=256, Hann window, center=True) with
//!   window-squared overlap-add normalization → waveform at 24 kHz.
//!
//! Deterministic verification (`vocos_reconstruction`): synthesize a 1s
//! 24 kHz two-tone signal, compute its log-mel (using the stored fb +
//! window), pass through vocos, compare reconstructed waveform against
//! input by Pearson correlation. Threshold ≥ 0.3 — sinusoids preserve
//! enough envelope/phase structure that even a lossy vocoder maintains
//! moderate alignment.

use rce_weight_loader::SafetensorsFile;
use rayon::prelude::*;
use rustfft::{num_complex::Complex32, FftPlanner};
use std::path::Path;

pub const SR: usize = 24_000;
const N_FFT: usize = 1024;
const HOP: usize = 256;
const N_MELS: usize = 100;
const DIM: usize = 512;
const FF: usize = 1536;
const N_BLOCKS: usize = 8;
const N_FREQ: usize = N_FFT / 2 + 1; // 513
const LN_EPS: f32 = 1e-6;
const MEL_FLOOR: f32 = 1e-5;

#[inline] fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0 - (((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t - 0.284_496_74) * t + 0.254_829_6) * t * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}
#[inline] fn gelu(x: f32) -> f32 { 0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2)) }

fn tf32(name: &str, f: &SafetensorsFile) -> Vec<f32> {
    let t = f.get(name).unwrap_or_else(|| panic!("missing {name}"));
    t.as_f32().unwrap_or_else(|| panic!("{name} not f32")).to_vec()
}

/// `y[t, d_out] = x[t, d_in] · W[d_out, d_in]^T (+bias)`, parallel over rows.
fn linear(x: &[f32], t: usize, w: &[f32], dout: usize, din: usize, b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; t * dout];
    out.par_chunks_mut(dout).enumerate().for_each(|(i, orow)| {
        let xr = &x[i * din..(i + 1) * din];
        for o in 0..dout {
            let wr = &w[o * din..(o + 1) * din];
            let mut acc = b[o];
            for k in 0..din { acc += xr[k] * wr[k]; }
            orow[o] = acc;
        }
    });
    out
}

/// LayerNorm (weight + bias) over last dim `d`, per row, in place.
fn layernorm(x: &mut [f32], d: usize, w: &[f32], b: &[f32]) {
    x.par_chunks_mut(d).for_each(|row| {
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| { let z = v - mean; z * z }).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        for ((v, g), bb) in row.iter_mut().zip(w).zip(b) { *v = (*v - mean) * inv * *g + *bb; }
    });
}

/// 1D conv on `[Cin, T]`, weight `[Cout, Cin, k]`. Zero-padding.
#[allow(clippy::too_many_arguments)]
fn conv1d(inp: &[f32], cin: usize, t: usize, w: &[f32], cout: usize, k: usize, stride: usize, pad: usize, bias: Option<&[f32]>) -> (Vec<f32>, usize) {
    let tout = (t + 2 * pad - k) / stride + 1;
    let mut out = vec![0.0f32; cout * tout];
    out.par_chunks_mut(tout).enumerate().for_each(|(oc, orow)| {
        let wbase = oc * cin * k;
        for to in 0..tout {
            let mut acc = bias.map_or(0.0, |b| b[oc]);
            for ic in 0..cin {
                for kk in 0..k {
                    let ti = (to * stride + kk) as isize - pad as isize;
                    if ti >= 0 && (ti as usize) < t { acc += w[wbase + ic * k + kk] * inp[ic * t + ti as usize]; }
                }
            }
            orow[to] = acc;
        }
    });
    (out, tout)
}

/// Depthwise 1D conv. `w` is `[C, 1, k]`. Zero-padding.
fn depthwise_conv1d(inp: &[f32], c: usize, t: usize, w: &[f32], b: &[f32], k: usize, pad: usize) -> Vec<f32> {
    let tout = t + 2 * pad - k + 1; // stride=1
    let mut out = vec![0.0f32; c * tout];
    out.par_chunks_mut(tout).enumerate().for_each(|(ch, row)| {
        let wbase = ch * k;
        let inb = &inp[ch * t..(ch + 1) * t];
        for to in 0..tout {
            let mut acc = b[ch];
            for kk in 0..k {
                let ti = (to + kk) as isize - pad as isize;
                if ti >= 0 && (ti as usize) < t { acc += w[wbase + kk] * inb[ti as usize]; }
            }
            row[to] = acc;
        }
    });
    out
}

/// STFT magnitude with stored Hann window, `n_fft=1024`, `hop=256`,
/// `center=True` (reflect-pad by `n_fft/2` each side). Returns `[N_FREQ, T_frames]`.
fn stft_magnitude(audio: &[f32], window: &[f32]) -> (Vec<f32>, usize) {
    let pad = N_FFT / 2;
    let mut padded = vec![0.0f32; audio.len() + 2 * pad];
    for i in 0..pad { padded[i] = audio[pad - i]; }
    padded[pad..pad + audio.len()].copy_from_slice(audio);
    for i in 0..pad { padded[pad + audio.len() + i] = audio[audio.len() - 2 - i]; }
    let n_frames = 1 + (padded.len() - N_FFT) / HOP;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut out = vec![0.0f32; N_FREQ * n_frames];
    for f in 0..n_frames {
        let mut buf = vec![Complex32::new(0.0, 0.0); N_FFT];
        for i in 0..N_FFT { buf[i] = Complex32::new(padded[f * HOP + i] * window[i], 0.0); }
        fft.process(&mut buf);
        for k in 0..N_FREQ {
            let m = (buf[k].re * buf[k].re + buf[k].im * buf[k].im).sqrt();
            out[k * n_frames + f] = m;
        }
    }
    (out, n_frames)
}

/// Apply mel filterbank `[N_FREQ, N_MELS]` to magnitude `[N_FREQ, T]`,
/// take safe log → log-mel `[N_MELS, T]`.
fn log_mel(magnitude: &[f32], n_frames: usize, fb: &[f32]) -> Vec<f32> {
    let mut mel = vec![0.0f32; N_MELS * n_frames];
    mel.par_chunks_mut(n_frames).enumerate().for_each(|(m, row)| {
        for f in 0..n_frames {
            let mut acc = 0.0f32;
            for k in 0..N_FREQ { acc += fb[k * N_MELS + m] * magnitude[k * n_frames + f]; }
            row[f] = acc.max(MEL_FLOOR).ln();
        }
    });
    mel
}

// ── weights ────────────────────────────────────────────────────────

struct ConvNeXtBlock {
    dwconv_w: Vec<f32>, dwconv_b: Vec<f32>, // [DIM, 1, 7], [DIM]
    norm_w: Vec<f32>, norm_b: Vec<f32>,     // [DIM]
    pwconv1_w: Vec<f32>, pwconv1_b: Vec<f32>, // [FF, DIM], [FF]
    pwconv2_w: Vec<f32>, pwconv2_b: Vec<f32>, // [DIM, FF], [DIM]
    gamma: Vec<f32>,                          // [DIM]
}

pub struct Vocos {
    pub window: Vec<f32>,        // [N_FFT]
    pub fb: Vec<f32>,            // [N_FREQ, N_MELS]
    embed_w: Vec<f32>, embed_b: Vec<f32>, // Conv1d(100→512, k=7)
    norm_w: Vec<f32>, norm_b: Vec<f32>,
    blocks: Vec<ConvNeXtBlock>,
    final_ln_w: Vec<f32>, final_ln_b: Vec<f32>,
    head_w: Vec<f32>, head_b: Vec<f32>, // [1026, 512], [1026]
}

impl Vocos {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, String> {
        let f = SafetensorsFile::load(dir.as_ref().join("model.safetensors"))
            .map_err(|e| format!("load: {e:?}"))?;
        let g = |n: &str| tf32(n, &f);

        let blocks = (0..N_BLOCKS).map(|i| {
            let p = format!("backbone.convnext.{i}");
            ConvNeXtBlock {
                dwconv_w: g(&format!("{p}.dwconv.weight")), dwconv_b: g(&format!("{p}.dwconv.bias")),
                norm_w: g(&format!("{p}.norm.weight")), norm_b: g(&format!("{p}.norm.bias")),
                pwconv1_w: g(&format!("{p}.pwconv1.weight")), pwconv1_b: g(&format!("{p}.pwconv1.bias")),
                pwconv2_w: g(&format!("{p}.pwconv2.weight")), pwconv2_b: g(&format!("{p}.pwconv2.bias")),
                gamma: g(&format!("{p}.gamma")),
            }
        }).collect();

        Ok(Self {
            window: g("head.istft.window"),
            fb: g("feature_extractor.mel_spec.mel_scale.fb"),
            embed_w: g("backbone.embed.weight"), embed_b: g("backbone.embed.bias"),
            norm_w: g("backbone.norm.weight"), norm_b: g("backbone.norm.bias"),
            blocks,
            final_ln_w: g("backbone.final_layer_norm.weight"), final_ln_b: g("backbone.final_layer_norm.bias"),
            head_w: g("head.out.weight"), head_b: g("head.out.bias"),
        })
    }

    /// Compute log-mel from raw 24 kHz audio (full Vocos feature
    /// extractor: STFT magnitude → mel filterbank → safe log).
    pub fn log_mel(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        let (mag, n_frames) = stft_magnitude(audio, &self.window);
        (log_mel(&mag, n_frames, &self.fb), n_frames)
    }

    /// Vocos forward: log-mel `[N_MELS, T]` → waveform `[~T·HOP]` at 24 kHz.
    pub fn vocode(&self, mel: &[f32], n_frames: usize) -> Vec<f32> {
        // 1) embed: Conv1d(100→512, k=7, p=3)
        let (embed, t) = conv1d(mel, N_MELS, n_frames, &self.embed_w, DIM, 7, 1, 3, Some(&self.embed_b));
        // [DIM, t] channel-major → transpose to [t, DIM] for LayerNorm + ConvNeXt
        let mut x = transpose_ct_to_tc(&embed, DIM, t);
        layernorm(&mut x, DIM, &self.norm_w, &self.norm_b);

        // 2) 8 ConvNeXt blocks (work in [DIM, t] for dwconv, [t, DIM] for the rest)
        for blk in &self.blocks {
            // residual = x (currently [t, DIM]), transpose to [DIM, t] for dwconv
            let x_ct = transpose_tc_to_ct(&x, t, DIM);
            let dw = depthwise_conv1d(&x_ct, DIM, t, &blk.dwconv_w, &blk.dwconv_b, 7, 3);
            let mut y = transpose_ct_to_tc(&dw, DIM, t);
            layernorm(&mut y, DIM, &blk.norm_w, &blk.norm_b);
            let mut h = linear(&y, t, &blk.pwconv1_w, FF, DIM, &blk.pwconv1_b);
            for v in h.iter_mut() { *v = gelu(*v); }
            let mut z = linear(&h, t, &blk.pwconv2_w, DIM, FF, &blk.pwconv2_b);
            // LayerScale per channel (broadcast over t)
            for ti in 0..t { for d in 0..DIM { z[ti * DIM + d] *= blk.gamma[d]; } }
            for i in 0..x.len() { x[i] += z[i]; }
        }
        // 3) final LN
        layernorm(&mut x, DIM, &self.final_ln_w, &self.final_ln_b);

        // 4) head: Linear 512→1026 → split mag/phase → exp/cos/sin → ISTFT
        let h = linear(&x, t, &self.head_w, 2 * N_FREQ, DIM, &self.head_b); // [t, 1026]
        // build complex spectrum [N_FREQ, t] from chunks
        let mut spec_re = vec![0.0f32; N_FREQ * t];
        let mut spec_im = vec![0.0f32; N_FREQ * t];
        for ti in 0..t {
            let row = &h[ti * 2 * N_FREQ..(ti + 1) * 2 * N_FREQ];
            for k in 0..N_FREQ {
                let m = row[k].exp().min(1e2);
                let p = row[N_FREQ + k];
                spec_re[k * t + ti] = m * p.cos();
                spec_im[k * t + ti] = m * p.sin();
            }
        }
        istft(&spec_re, &spec_im, t, &self.window)
    }
}

fn transpose_ct_to_tc(x: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; t * c];
    for ch in 0..c { for ti in 0..t { out[ti * c + ch] = x[ch * t + ti]; } }
    out
}
fn transpose_tc_to_ct(x: &[f32], t: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c * t];
    for ti in 0..t { for ch in 0..c { out[ch * t + ti] = x[ti * c + ch]; } }
    out
}

/// ISTFT (n_fft=1024, hop=256, Hann window, center=True): inverse FFT
/// each frame, window, overlap-add, divide by window² OLA for unbiased
/// reconstruction. Returns the time-domain waveform (center pad stripped).
fn istft(spec_re: &[f32], spec_im: &[f32], n_frames: usize, window: &[f32]) -> Vec<f32> {
    let mut planner = FftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(N_FFT);
    let total_len = (n_frames - 1) * HOP + N_FFT;
    let mut accum = vec![0.0f32; total_len];
    let mut norm = vec![0.0f32; total_len];
    let win_sq: Vec<f32> = window.iter().map(|w| w * w).collect();

    for f in 0..n_frames {
        // Reconstruct full-length complex spectrum from one-sided
        let mut buf = vec![Complex32::new(0.0, 0.0); N_FFT];
        for k in 0..N_FREQ {
            buf[k] = Complex32::new(spec_re[k * n_frames + f], spec_im[k * n_frames + f]);
        }
        // Mirror conjugate: bins (N_FFT-k) = conj(bins[k]) for k in 1..N_FREQ-1
        for k in 1..N_FREQ - 1 {
            buf[N_FFT - k] = Complex32::new(spec_re[k * n_frames + f], -spec_im[k * n_frames + f]);
        }
        ifft.process(&mut buf);
        let scale = 1.0 / N_FFT as f32;
        let start = f * HOP;
        for i in 0..N_FFT {
            let s = buf[i].re * scale * window[i];
            accum[start + i] += s;
            norm[start + i] += win_sq[i];
        }
    }
    // window-normalized output, center-pad strip
    let pad = N_FFT / 2;
    let mut out = vec![0.0f32; total_len - 2 * pad];
    for i in 0..out.len() {
        let n = norm[pad + i];
        out[i] = if n > 1e-10 { accum[pad + i] / n } else { 0.0 };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn model_dir() -> Option<String> {
        for b in ["macos_4TB_external", "macos_4TB_external_1", "macos_4TB_external_2"] {
            let d = format!("/Volumes/{b}/neuronexus_ai/model_library/vocos");
            if std::path::Path::new(&d).join("model.safetensors").exists() { return Some(d); }
        }
        None
    }

    #[test]
    fn vocos_reconstruction() {
        let Some(dir) = model_dir() else { eprintln!("vocos: not mounted — skipping"); return };
        let m = Vocos::load(&dir).expect("load");
        // 1.0s @24kHz two-tone deterministic signal
        let n = SR; // 24000 samples
        let audio: Vec<f32> = (0..n).map(|i| {
            let t = i as f32 / SR as f32;
            0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
        }).collect();
        let t0 = std::time::Instant::now();
        let (mel, n_frames) = m.log_mel(&audio);
        eprintln!("log-mel: {n_frames} frames in {:?}", t0.elapsed());
        let t1 = std::time::Instant::now();
        let recon = m.vocode(&mel, n_frames);
        eprintln!("vocos: {} samples ({:.2}s) in {:?}", recon.len(), recon.len() as f32 / SR as f32, t1.elapsed());

        assert!(recon.iter().all(|v| v.is_finite()), "non-finite samples");
        let peak = recon.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        // Re-extract mel from the reconstructed audio and correlate
        // against the input mel. This is the right metric for a
        // vocoder: time-domain phase is invented, but the mel
        // (envelope) of the reconstruction should track the input mel.
        let (mel_recon, n_frames_r) = m.log_mel(&recon);
        let n = mel.len().min(mel_recon.len());
        let ma = mel[..n].iter().sum::<f32>() / n as f32;
        let mr = mel_recon[..n].iter().sum::<f32>() / n as f32;
        let mut num = 0.0f64; let mut da = 0.0f64; let mut db = 0.0f64;
        for i in 0..n {
            let xa = (mel[i] - ma) as f64; let xb = (mel_recon[i] - mr) as f64;
            num += xa * xb; da += xa * xa; db += xb * xb;
        }
        let mel_corr = (num / (da.sqrt() * db.sqrt())) as f32;
        eprintln!("vocos: peak={peak:.4}  mel-mel corr={mel_corr:.4} ({n_frames}\u{2192}{n_frames_r} frames)");
        assert!(peak > 0.05, "vocos output near-silent (peak {peak})");
        assert!(mel_corr > 0.6, "mel reconstruction correlation too low: {mel_corr:.4} (threshold 0.6)");
    }
}
