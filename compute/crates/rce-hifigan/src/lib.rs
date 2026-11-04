//! SpeechT5 HiFi-GAN vocoder: log-mel @ 16 kHz → waveform.
//!
//! Pipeline:  (mel − mean) / scale → conv_pre k=7 → 4 × upsample stage
//!            ( LeakyReLU → ConvT1d k=8 s=4 → Σ_{3 kernels} ResBlock / 3 )
//!            → LeakyReLU → conv_post k=7 → tanh
//! Channels: 80 → 512 → 256 → 128 → 64 → 32 → 1.   Total upsample 4⁴ = 256.
//!
//! Weights are already weight_norm-fused at save time, so we load `.weight` /
//! `.bias` directly — no `g·v/‖v‖₂` resolution like SNAC / Vocos.

use rce_weight_loader::SafetensorsFile;
use std::path::Path;
use std::sync::Arc;

pub const SR: u32 = 16000;
pub const N_MELS: usize = 80;
pub const N_FFT: usize = 1024;
pub const HOP: usize = 256;
pub const WIN: usize = 1024;
pub const UPSAMPLE_RATES: [usize; 4] = [4, 4, 4, 4];
pub const UPSAMPLE_KERNELS: [usize; 4] = [8, 8, 8, 8];
pub const RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];
pub const RESBLOCK_DILATIONS: [[usize; 3]; 3] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];
pub const LEAKY_SLOPE: f32 = 0.1;
pub const FMIN: f32 = 80.0;
pub const FMAX: f32 = 7600.0;

#[derive(Debug)]
pub enum HifiGanError { Load(String) }
impl std::fmt::Display for HifiGanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::Load(m) => write!(f, "hifigan load error: {m}") }
    }
}
impl std::error::Error for HifiGanError {}

fn tf32(t: &rce_weight_loader::LoadedTensor) -> Vec<f32> {
    let bytes = t.data.as_slice();
    match t.dtype.as_str() {
        "f32" => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect(),
        "f16" => bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0],c[1]]).to_f32()).collect(),
        "bf16" => bytes.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0],c[1]]).to_f32()).collect(),
        other => panic!("hifigan: unsupported dtype {other}"),
    }
}

fn load_tensor(sf: &SafetensorsFile, name: &str) -> Result<(Vec<f32>, Vec<usize>), HifiGanError> {
    let t = sf.get(name).ok_or_else(|| HifiGanError::Load(format!("missing {name}")))?;
    Ok((tf32(t), t.shape.clone()))
}

fn leaky_relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() { if *v < 0.0 { *v *= LEAKY_SLOPE; } }
}

// 1D conv with zero pad, dilation, stride. x: [Cin, T], w: [Cout, Cin, K], b: [Cout].
fn conv1d(x: &[f32], cin: usize, t: usize,
          w: &[f32], cout: usize, k: usize,
          stride: usize, dilation: usize, pad: usize,
          bias: &[f32]) -> (Vec<f32>, usize) {
    let kspan = dilation*(k-1) + 1;
    let t_out = if t + 2*pad < kspan { 0 } else { (t + 2*pad - kspan) / stride + 1 };
    let mut out = vec![0.0f32; cout * t_out];
    for to in 0..t_out {
        let t_start = to * stride;
        for oc in 0..cout {
            let mut acc = bias[oc];
            for ic in 0..cin {
                for kk in 0..k {
                    let tp = t_start + kk * dilation;
                    if tp < pad { continue; }
                    let xi = tp - pad;
                    if xi >= t { continue; }
                    acc += x[ic * t + xi] * w[(oc * cin + ic) * k + kk];
                }
            }
            out[oc * t_out + to] = acc;
        }
    }
    (out, t_out)
}

// 1D transposed conv. x: [Cin, T], w: [Cin, Cout, K], b: [Cout].
fn conv_transpose1d(x: &[f32], cin: usize, t: usize,
                    w: &[f32], cout: usize, k: usize,
                    stride: usize, pad: usize,
                    bias: &[f32]) -> (Vec<f32>, usize) {
    let t_out_full = t.saturating_sub(1) * stride + k;
    let mut full = vec![0.0f32; cout * t_out_full];
    for ti in 0..t {
        for ic in 0..cin {
            let xv = x[ic * t + ti];
            for oc in 0..cout {
                let row = oc * t_out_full;
                let w_base = (ic * cout + oc) * k;
                let t_base = ti * stride;
                for kk in 0..k {
                    full[row + t_base + kk] += xv * w[w_base + kk];
                }
            }
        }
    }
    let t_out = t_out_full.saturating_sub(2 * pad);
    let mut out = vec![0.0f32; cout * t_out];
    for oc in 0..cout {
        for to in 0..t_out {
            out[oc * t_out + to] = full[oc * t_out_full + to + pad] + bias[oc];
        }
    }
    (out, t_out)
}

// HiFi-GAN MRF residual block:  k = kernel_size, 3 dilations on convs1, plain convs2.
struct ResBlock {
    c: usize,
    k: usize,
    dilations: [usize; 3],
    convs1_w: [Vec<f32>; 3], convs1_b: [Vec<f32>; 3],
    convs2_w: [Vec<f32>; 3], convs2_b: [Vec<f32>; 3],
}
impl ResBlock {
    fn load(sf: &SafetensorsFile, idx: usize, c: usize, k: usize, dilations: [usize; 3]) -> Result<Self, HifiGanError> {
        let mut cw1: [Vec<f32>; 3] = Default::default();
        let mut cb1: [Vec<f32>; 3] = Default::default();
        let mut cw2: [Vec<f32>; 3] = Default::default();
        let mut cb2: [Vec<f32>; 3] = Default::default();
        for j in 0..3 {
            let (w, sh) = load_tensor(sf, &format!("resblocks.{idx}.convs1.{j}.weight"))?;
            if sh != vec![c, c, k] {
                return Err(HifiGanError::Load(format!("resblocks.{idx}.convs1.{j}.weight: shape {:?} != [{c},{c},{k}]", sh)));
            }
            let (b, _) = load_tensor(sf, &format!("resblocks.{idx}.convs1.{j}.bias"))?;
            cw1[j] = w; cb1[j] = b;
            let (w, _) = load_tensor(sf, &format!("resblocks.{idx}.convs2.{j}.weight"))?;
            let (b, _) = load_tensor(sf, &format!("resblocks.{idx}.convs2.{j}.bias"))?;
            cw2[j] = w; cb2[j] = b;
        }
        Ok(Self { c, k, dilations, convs1_w: cw1, convs1_b: cb1, convs2_w: cw2, convs2_b: cb2 })
    }
    fn forward(&self, x: &mut Vec<f32>, t: usize) {
        // Three (LeakyReLU → dilated conv1 → LeakyReLU → conv2 → +residual) sub-blocks.
        for j in 0..3 {
            let d = self.dilations[j];
            let pad = (self.k - 1) * d / 2;
            let mut residual = x.clone();
            leaky_relu_inplace(x);
            let (y, _) = conv1d(x, self.c, t, &self.convs1_w[j], self.c, self.k, 1, d, pad, &self.convs1_b[j]);
            *x = y;
            leaky_relu_inplace(x);
            let pad2 = (self.k - 1) / 2;
            let (y, _) = conv1d(x, self.c, t, &self.convs2_w[j], self.c, self.k, 1, 1, pad2, &self.convs2_b[j]);
            *x = y;
            for i in 0..x.len() { x[i] += residual[i]; }
            residual.clear();
        }
    }
}

pub struct HifiGan {
    mean: Vec<f32>,
    scale: Vec<f32>,
    conv_pre_w: Vec<f32>, conv_pre_b: Vec<f32>,    // [512, 80, 7]
    ups_w: [Vec<f32>; 4], ups_b: [Vec<f32>; 4],     // ConvT [Cin, Cout, 8]
    ups_ch_in: [usize; 4],  ups_ch_out: [usize; 4],
    resblocks: Vec<ResBlock>,                       // 12 (4 upsample × 3 kernels)
    conv_post_w: Vec<f32>, conv_post_b: Vec<f32>,   // [1, 32, 7]
}

impl HifiGan {
    pub fn load(path: &Path) -> Result<Arc<Self>, HifiGanError> {
        let sf = SafetensorsFile::load(path).map_err(|e| HifiGanError::Load(e.to_string()))?;
        let (mean, _)  = load_tensor(&sf, "mean")?;
        let (scale, _) = load_tensor(&sf, "scale")?;
        if mean.len() != N_MELS || scale.len() != N_MELS {
            return Err(HifiGanError::Load("mean/scale shape mismatch".into()));
        }
        let (conv_pre_w, sh) = load_tensor(&sf, "conv_pre.weight")?;
        if sh != vec![512, N_MELS, 7] {
            return Err(HifiGanError::Load(format!("conv_pre.weight: {:?} != [512,80,7]", sh)));
        }
        let (conv_pre_b, _) = load_tensor(&sf, "conv_pre.bias")?;

        let mut ups_w: [Vec<f32>; 4] = Default::default();
        let mut ups_b: [Vec<f32>; 4] = Default::default();
        let mut ups_ch_in: [usize; 4] = [0; 4];
        let mut ups_ch_out: [usize; 4] = [0; 4];
        for i in 0..4 {
            let c_in = 512 >> i;            // 512, 256, 128, 64
            let c_out = 512 >> (i + 1);     // 256, 128,  64, 32
            let (w, sh) = load_tensor(&sf, &format!("upsampler.{i}.weight"))?;
            if sh != vec![c_in, c_out, 8] {
                return Err(HifiGanError::Load(format!("upsampler.{i}.weight: {:?} != [{c_in},{c_out},8]", sh)));
            }
            let (b, _) = load_tensor(&sf, &format!("upsampler.{i}.bias"))?;
            ups_w[i] = w; ups_b[i] = b;
            ups_ch_in[i] = c_in;
            ups_ch_out[i] = c_out;
        }

        let mut resblocks = Vec::with_capacity(12);
        for i in 0..4 {
            let c = 512 >> (i + 1);  // 256, 128, 64, 32
            for j in 0..3 {
                resblocks.push(ResBlock::load(&sf, i*3 + j, c, RESBLOCK_KERNELS[j], RESBLOCK_DILATIONS[j])?);
            }
        }

        let (conv_post_w, sh) = load_tensor(&sf, "conv_post.weight")?;
        if sh != vec![1, 32, 7] {
            return Err(HifiGanError::Load(format!("conv_post.weight: {:?} != [1,32,7]", sh)));
        }
        let (conv_post_b, _) = load_tensor(&sf, "conv_post.bias")?;

        Ok(Arc::new(Self {
            mean, scale, conv_pre_w, conv_pre_b,
            ups_w, ups_b, ups_ch_in, ups_ch_out,
            resblocks,
            conv_post_w, conv_post_b,
        }))
    }

    /// Forward: log-mel `[N_MELS × T_mel]` → waveform of length `T_mel × 256`.
    pub fn vocode(&self, mel: &[f32], t_mel: usize) -> Vec<f32> {
        assert_eq!(mel.len(), N_MELS * t_mel, "mel.len()={} expected {}*{t_mel}", mel.len(), N_MELS);
        // Normalize: (mel - mean) / scale, per-mel-band broadcast over time.
        let mut x = vec![0.0f32; N_MELS * t_mel];
        for c in 0..N_MELS {
            let inv_s = if self.scale[c].abs() > 1e-12 { 1.0 / self.scale[c] } else { 0.0 };
            for ti in 0..t_mel {
                x[c * t_mel + ti] = (mel[c * t_mel + ti] - self.mean[c]) * inv_s;
            }
        }
        // conv_pre: 80→512, k=7, pad=3.
        let (y, t) = conv1d(&x, N_MELS, t_mel, &self.conv_pre_w, 512, 7, 1, 1, 3, &self.conv_pre_b);
        let mut x = y;
        let mut t = t;

        for i in 0..4 {
            leaky_relu_inplace(&mut x);
            let kup = UPSAMPLE_KERNELS[i];
            let s  = UPSAMPLE_RATES[i];
            let pad = (kup - s) / 2;
            let (y, t1) = conv_transpose1d(&x, self.ups_ch_in[i], t,
                                           &self.ups_w[i], self.ups_ch_out[i], kup,
                                           s, pad, &self.ups_b[i]);
            x = y; t = t1;
            // MRF: sum over 3 resblocks, divide by 3.
            let mut acc = vec![0.0f32; x.len()];
            for j in 0..3 {
                let mut branch = x.clone();
                self.resblocks[i*3 + j].forward(&mut branch, t);
                for k in 0..acc.len() { acc[k] += branch[k]; }
            }
            let inv3 = 1.0 / 3.0;
            for k in 0..acc.len() { acc[k] *= inv3; }
            x = acc;
        }
        leaky_relu_inplace(&mut x);
        // conv_post: 32→1, k=7, pad=3.
        let (y, _) = conv1d(&x, 32, t, &self.conv_post_w, 1, 7, 1, 1, 3, &self.conv_post_b);
        x = y;
        for v in x.iter_mut() { *v = v.tanh(); }
        x
    }
}

// --------------- Slaney log-mel front-end (matches what SpeechT5 was trained on) ---------------

fn hz_to_mel(f: f32) -> f32 { 2595.0 * (1.0 + f / 700.0).log10() }
fn mel_to_hz(m: f32) -> f32 { 700.0 * (10f32.powf(m / 2595.0) - 1.0) }

pub fn slaney_mel_filterbank(sr: f32, n_fft: usize, n_mels: usize, fmin: f32, fmax: f32) -> Vec<f32> {
    let n_bin = n_fft / 2 + 1;
    let m_min = hz_to_mel(fmin);
    let m_max = hz_to_mel(fmax);
    let mel_pts: Vec<f32> = (0..n_mels + 2).map(|i| m_min + (m_max - m_min) * i as f32 / (n_mels + 1) as f32).collect();
    let hz_pts: Vec<f32> = mel_pts.iter().map(|&m| mel_to_hz(m)).collect();
    let fft_pts: Vec<f32> = hz_pts.iter().map(|&h| h * n_fft as f32 / sr).collect();
    let mut fb = vec![0.0f32; n_mels * n_bin];
    for m in 0..n_mels {
        let l = fft_pts[m];
        let c = fft_pts[m + 1];
        let r = fft_pts[m + 2];
        // Slaney-norm: divide triangle by (hz_pts[m+2] - hz_pts[m]) / 2.
        let enorm = 2.0 / (hz_pts[m + 2] - hz_pts[m]);
        for b in 0..n_bin {
            let fb_idx = b as f32;
            let lo = (fb_idx - l) / (c - l).max(1e-9);
            let hi = (r - fb_idx) / (r - c).max(1e-9);
            let v = lo.min(hi).max(0.0);
            fb[m * n_bin + b] = v * enorm;
        }
    }
    fb
}

pub fn log_mel_spectrogram(audio: &[f32]) -> (Vec<f32>, usize) {
    use rustfft::{FftPlanner, num_complex::Complex};
    let n_fft = N_FFT;
    let hop = HOP;
    let n_bin = n_fft / 2 + 1;
    let win: Vec<f32> = (0..WIN).map(|i| 0.5 - 0.5 * ((2.0 * std::f32::consts::PI * i as f32) / (WIN as f32 - 1.0)).cos()).collect();
    // Center-pad audio with `n_fft/2` reflection — SpeechT5 uses center=True.
    let pad = n_fft / 2;
    let mut padded = vec![0.0f32; audio.len() + 2 * pad];
    for i in 0..pad { padded[i] = audio[(pad - i).min(audio.len() - 1)]; }
    for i in 0..audio.len() { padded[pad + i] = audio[i]; }
    for i in 0..pad { padded[pad + audio.len() + i] = audio[audio.len().saturating_sub(2 + i)]; }
    let n_frames = (padded.len() - n_fft) / hop + 1;

    let fb = slaney_mel_filterbank(SR as f32, n_fft, N_MELS, FMIN, FMAX);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    let mut mel = vec![0.0f32; N_MELS * n_frames];
    let mut buf = vec![Complex::<f32>::new(0.0, 0.0); n_fft];
    for f in 0..n_frames {
        for i in 0..n_fft { buf[i] = Complex::new(padded[f * hop + i] * win[i], 0.0); }
        fft.process(&mut buf);
        let mut power = vec![0.0f32; n_bin];
        for b in 0..n_bin { power[b] = buf[b].norm_sqr(); }
        for m in 0..N_MELS {
            let mut acc = 0.0f32;
            for b in 0..n_bin { acc += fb[m * n_bin + b] * power[b]; }
            mel[m * n_frames + f] = acc.max(1e-10).log10();
        }
    }
    (mel, n_frames)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn pearson(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len()) as f32;
        let ma: f32 = a.iter().sum::<f32>() / n;
        let mb: f32 = b.iter().sum::<f32>() / n;
        let mut num=0.0f32; let mut da=0.0f32; let mut db=0.0f32;
        for i in 0..a.len().min(b.len()) {
            let xa = a[i]-ma; let xb = b[i]-mb;
            num += xa*xb; da += xa*xa; db += xb*xb;
        }
        if da*db > 1e-12 { num / (da*db).sqrt() } else { 0.0 }
    }

    fn safetensors_path() -> Option<std::path::PathBuf> {
        let p = std::path::PathBuf::from("/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/hifigan/model.safetensors");
        if p.exists() { Some(p) } else { None }
    }

    #[test]
    fn hifigan_mel_to_wav() {
        let Some(p) = safetensors_path() else {
            eprintln!("skip: hifigan safetensors not present"); return;
        };
        eprintln!("loading {}…", p.display());
        let t0 = Instant::now();
        let hg = HifiGan::load(&p).expect("load");
        eprintln!("loaded in {:?}", t0.elapsed());

        // 1.024 s of 440 Hz + 880 Hz two-tone at 16 kHz.
        let sr = SR as f32;
        let n = 16384;   // 1.024 s; mel will be 65 frames (1024/256 - 1 + 1 + pad).
        let audio: Vec<f32> = (0..n).map(|i| {
            let t = i as f32 / sr;
            0.4 * (2.0*std::f32::consts::PI * 440.0 * t).sin()
              + 0.3 * (2.0*std::f32::consts::PI * 880.0 * t).sin()
        }).collect();

        let t0 = Instant::now();
        let (mel, t_mel) = log_mel_spectrogram(&audio);
        eprintln!("log-mel: {t_mel} frames in {:?}", t0.elapsed());
        assert_eq!(mel.len(), N_MELS * t_mel);

        let t0 = Instant::now();
        let wav = hg.vocode(&mel, t_mel);
        eprintln!("hifigan: {} samples in {:?}", wav.len(), t0.elapsed());
        assert_eq!(wav.len(), t_mel * HOP);

        let peak = wav.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        let rms: f32 = (wav.iter().map(|x| x*x).sum::<f32>() / wav.len() as f32).sqrt();
        assert!(peak.is_finite(), "non-finite output");
        assert!(peak > 0.005, "output essentially silent (peak {peak})");
        assert!(peak <= 1.0, "tanh failed?  peak {peak}");

        // Re-extract mel from reconstruction and correlate.
        let (rec_mel, _) = log_mel_spectrogram(&wav);
        let m_in_len = N_MELS * t_mel.min(rec_mel.len() / N_MELS);
        let corr = pearson(&mel[..m_in_len], &rec_mel[..m_in_len]);
        eprintln!("hifigan: peak={peak:.4} rms={rms:.4} mel-mel corr={corr:.4}");
        assert!(corr > 0.5, "mel-mel correlation too low: {corr}");
    }
}
