//! OpenAI Whisper large-v3-turbo ASR on the rce substrate.
//!
//! Reuses the encoder-decoder + cross-attention pattern proven by
//! `rce-moonshine`, adding the log-mel STFT frontend Moonshine
//! didn't need. Spec verified against HF `modeling_whisper.py` +
//! `feature_extraction_whisper.py` + `audio_utils.py`.
//!
//! - **Log-mel frontend**: center reflect-pad 200; periodic Hann
//!   `n_fft=400`, `hop=160`; power spectrogram (201 bins); 128 **Slaney**
//!   mel filters; drop last frame; `log10(max(1e-10,·))`; clamp to
//!   `global_max − 8`; `(x+4)/4`.
//! - **Encoder**: conv1(128→1280,k3,s1,p1)+GELU → conv2(1280→1280,k3,s2,p1)
//!   +GELU → +stored positions → 32 pre-norm MHA layers (20 heads × 64,
//!   no rotary, k_proj has no bias, GELU MLP) → final LN.
//! - **Decoder** (4 layers): embed_tokens[tok] + learned embed_positions[pos]
//!   → pre-norm causal self-attn → cross-attn (q=decoder, k/v=encoder) →
//!   GELU MLP → final LN → tied lm_head.
//! - F16 weights (loaded → f32). GPT-2 ByteLevel-BPE detokenizer.
//!   Forced English-transcribe prefix `[50258, 50259, 50360, 50364]`,
//!   eos `50257`.

use rce_op_attn::fused_attend_head;
use rce_op_conv2d::F16Slice;
use rce_weight_loader::{LoadedTensor, SafetensorsFile};
use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use std::collections::HashMap;
use std::path::Path;

const D_MODEL: usize = 1280;
const N_HEADS: usize = 20;
const D_HEAD: usize = 64; // 1280 / 20
const N_ENC: usize = 32;
const N_DEC: usize = 4;
const FFN: usize = 5120;
const VOCAB: usize = 51866;
const EPS: f32 = 1e-5;

// mel frontend
const N_FFT: usize = 400;
const HOP: usize = 160;
const N_MEL: usize = 128;
const N_BINS: usize = 201; // N_FFT/2 + 1
const SR: usize = 16000;

// decoding
const SOT: i32 = 50258;
const LANG_EN: i32 = 50259;
const TRANSCRIBE: i32 = 50360;
const NO_TS: i32 = 50364;
const EOS: i32 = 50257;
const SPECIAL_MIN: i32 = 50257; // ids ≥ this are special markers

// ── numeric helpers ────────────────────────────────────────────────

#[inline] fn erf(x: f32) -> f32 {
    // Abramowitz-Stegun 7.1.26, ~1.5e-7 max error.
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0 - (((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t - 0.284_496_74) * t + 0.254_829_6) * t * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}
#[inline] fn gelu(x: f32) -> f32 { 0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2)) }
#[inline] fn scale() -> f32 { 1.0 / (D_HEAD as f32).sqrt() }

/// `y[t, d_out] = x[t, d_in] · W[d_out, d_in]^T (+ bias)`. Delegates to
/// `rce_op_conv2d::linear_f32` (Apple Accelerate sgemm on macOS).
fn linear(x: &[f32], t: usize, w: &[f32], d_out: usize, d_in: usize, bias: Option<&[f32]>) -> Vec<f32> {
    rce_op_conv2d::linear_f32(x, t, d_in, w, d_out, bias)
}

/// f16-weight linear via dequant + Accelerate sgemm.  Same shape contract as
/// `linear` but reads weights from a borrow-direct `&[half::f16]` slice.
fn linear_f16(x: &[f32], t: usize, w: &[half::f16], d_out: usize, d_in: usize, bias: Option<&[f32]>) -> Vec<f32> {
    rce_op_conv2d::linear_f16(x, t, d_in, w, d_out, bias)
}

/// LayerNorm with weight + bias over the last dim, per row.
fn layernorm(x: &mut [f32], d: usize, w: &[f32], b: &[f32]) {
    x.par_chunks_mut(d).for_each(|row| {
        let mean: f32 = row.iter().sum::<f32>() / d as f32;
        let var: f32 = row.iter().map(|v| { let z = v - mean; z * z }).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        for ((v, g), bb) in row.iter_mut().zip(w.iter()).zip(b.iter()) { *v = (*v - mean) * inv * *g + *bb; }
    });
}

/// Conv1d with zero-padding `pad`, parallel over output channels.
/// `inp` is `[c_in, l_in]` row-major, `w` is `[c_out, c_in, k]`.
fn conv1d(inp: &[f32], c_in: usize, l_in: usize, w: &[f32], c_out: usize, k: usize, stride: usize, pad: usize, bias: &[f32]) -> (Vec<f32>, usize) {
    let l_out = (l_in + 2 * pad - k) / stride + 1;
    let mut out = vec![0.0f32; c_out * l_out];
    out.par_chunks_mut(l_out).enumerate().for_each(|(oc, orow)| {
        let wbase = oc * c_in * k;
        for (t, slot) in orow.iter_mut().enumerate() {
            let start = (t * stride) as isize - pad as isize;
            let mut acc = bias[oc];
            for ic in 0..c_in {
                let wic = &w[wbase + ic * k..wbase + ic * k + k];
                let inic = &inp[ic * l_in..(ic + 1) * l_in];
                for kk in 0..k {
                    let idx = start + kk as isize;
                    if idx >= 0 && (idx as usize) < l_in { acc += wic[kk] * inic[idx as usize]; }
                }
            }
            *slot = acc;
        }
    });
    (out, l_out)
}

// ── mel frontend ───────────────────────────────────────────────────

fn hz_to_mel_slaney(f: f32) -> f32 {
    if f >= 1000.0 { 15.0 + (f / 1000.0).ln() * (27.0 / 6.4f32.ln()) } else { 3.0 * f / 200.0 }
}
fn mel_to_hz_slaney(m: f32) -> f32 {
    if m >= 15.0 { 1000.0 * (6.4f32.ln() / 27.0 * (m - 15.0)).exp() } else { 200.0 * m / 3.0 }
}

/// Slaney mel filter bank, `[N_MEL][N_BINS]` row-major (mel-major).
fn build_mel_filters() -> Vec<f32> {
    let mel_min = hz_to_mel_slaney(0.0);
    let mel_max = hz_to_mel_slaney(8000.0);
    // N_MEL + 2 centre points in mel, mapped back to Hz.
    let filter_freqs: Vec<f32> = (0..N_MEL + 2)
        .map(|i| mel_to_hz_slaney(mel_min + (mel_max - mel_min) * i as f32 / (N_MEL + 1) as f32))
        .collect();
    // FFT bin frequencies in Hz: linspace(0, sr/2, N_BINS).
    let fft_freqs: Vec<f32> = (0..N_BINS).map(|i| (SR / 2) as f32 * i as f32 / (N_BINS - 1) as f32).collect();

    let mut filt = vec![0.0f32; N_MEL * N_BINS];
    for m in 0..N_MEL {
        let f_left = filter_freqs[m];
        let f_cent = filter_freqs[m + 1];
        let f_right = filter_freqs[m + 2];
        let d_lo = f_cent - f_left;
        let d_hi = f_right - f_cent;
        // Slaney area normalization.
        let enorm = 2.0 / (f_right - f_left);
        for (k, &fk) in fft_freqs.iter().enumerate() {
            let down = (fk - f_left) / d_lo;
            let up = (f_right - fk) / d_hi;
            let w = down.min(up).max(0.0);
            filt[m * N_BINS + k] = w * enorm;
        }
    }
    filt
}

/// Log-mel spectrogram, `[N_MEL, n_frames]` channel-major. Audio is a
/// 16 kHz mono waveform in [-1, 1]; processed at its natural length
/// (positions sliced from the encoder's stored table, not padded to 30s).
fn log_mel(audio: &[f32], filters: &[f32]) -> (Vec<f32>, usize) {
    // periodic Hann window
    let window: Vec<f32> = (0..N_FFT).map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / N_FFT as f32).cos()).collect();
    // center reflect-pad by N_FFT/2 = 200
    let pad = N_FFT / 2;
    let mut padded = vec![0.0f32; audio.len() + 2 * pad];
    for i in 0..pad { padded[i] = audio[pad - i]; } // reflect (exclude edge sample)
    padded[pad..pad + audio.len()].copy_from_slice(audio);
    for i in 0..pad { padded[pad + audio.len() + i] = audio[audio.len() - 2 - i]; }

    let n_frames_full = 1 + (padded.len() - N_FFT) / HOP;
    let n_frames = n_frames_full - 1; // drop last frame (Whisper: log_spec[:, :-1])

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);

    // mel_spec[N_MEL, n_frames] channel-major; compute per frame in parallel.
    let mut mel = vec![0.0f32; N_MEL * n_frames];
    let cols: Vec<Vec<f32>> = (0..n_frames).into_par_iter().map(|f| {
        let mut buf: Vec<Complex32> = (0..N_FFT)
            .map(|i| Complex32::new(padded[f * HOP + i] * window[i], 0.0))
            .collect();
        fft.process(&mut buf);
        let mut col = vec![0.0f32; N_MEL];
        for m in 0..N_MEL {
            let frow = &filters[m * N_BINS..(m + 1) * N_BINS];
            let mut acc = 0.0f32;
            for k in 0..N_BINS {
                let p = buf[k].re * buf[k].re + buf[k].im * buf[k].im; // power
                acc += frow[k] * p;
            }
            col[m] = acc.max(1e-10).log10();
        }
        col
    }).collect();
    for (f, col) in cols.iter().enumerate() {
        for m in 0..N_MEL { mel[m * n_frames + f] = col[m]; }
    }
    // clamp to global_max - 8, then (x + 4) / 4
    let gmax = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let floor = gmax - 8.0;
    for v in mel.iter_mut() { *v = (v.max(floor) + 4.0) / 4.0; }
    (mel, n_frames)
}

// ── weights ────────────────────────────────────────────────────────

struct EncLayer {
    ln1_w: Vec<f32>, ln1_b: Vec<f32>, ln2_w: Vec<f32>, ln2_b: Vec<f32>,
    q: F16Slice, q_b: Vec<f32>, k: F16Slice, v: F16Slice, v_b: Vec<f32>, o: F16Slice, o_b: Vec<f32>,
    fc1: F16Slice, fc1_b: Vec<f32>, fc2: F16Slice, fc2_b: Vec<f32>,
}
struct DecLayer {
    ln1_w: Vec<f32>, ln1_b: Vec<f32>,         // self_attn_layer_norm
    lnc_w: Vec<f32>, lnc_b: Vec<f32>,         // encoder_attn_layer_norm
    ln2_w: Vec<f32>, ln2_b: Vec<f32>,         // final_layer_norm
    q: F16Slice, q_b: Vec<f32>, k: F16Slice, v: F16Slice, v_b: Vec<f32>, o: F16Slice, o_b: Vec<f32>,
    cq: F16Slice, cq_b: Vec<f32>, ck: F16Slice, cv: F16Slice, cv_b: Vec<f32>, co: F16Slice, co_b: Vec<f32>,
    fc1: F16Slice, fc1_b: Vec<f32>, fc2: F16Slice, fc2_b: Vec<f32>,
}

pub struct Whisper {
    conv1_w: Vec<f32>, conv1_b: Vec<f32>, conv2_w: Vec<f32>, conv2_b: Vec<f32>,
    enc_pos: Vec<f32>,          // [1500, D_MODEL]
    enc: Vec<EncLayer>,
    enc_ln_w: Vec<f32>, enc_ln_b: Vec<f32>,
    embed: Vec<f32>,            // [VOCAB, D_MODEL] — tied lm_head
    dec_pos: Vec<f32>,          // [448, D_MODEL]
    dec: Vec<DecLayer>,
    dec_ln_w: Vec<f32>, dec_ln_b: Vec<f32>,
    mel_filters: Vec<f32>,
    id2piece: HashMap<i32, String>,
    byte_decoder: HashMap<char, u8>,
}

/// Any f16/f32 tensor → owned `Vec<f32>`.
fn tf32(t: &LoadedTensor) -> Vec<f32> {
    if let Some(s) = t.as_f32() { return s.to_vec(); }
    match t.dtype.as_str() {
        "f16" => t.data.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        "bf16" => t.data.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        d => panic!("unsupported dtype {d}"),
    }
}

impl Whisper {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = model_dir.as_ref();
        let f = SafetensorsFile::load(dir.join("model.safetensors")).map_err(|e| format!("load: {e:?}"))?;
        let g = |n: &str| -> Vec<f32> {
            tf32(f.get(n).unwrap_or_else(|| panic!("missing tensor {n}")))
        };
        // Borrow-direct f16 loader for the big projection matrices — the
        // raw f16 bytes alias into the safetensors mmap, no decode at load.
        let g16 = |n: &str| -> F16Slice {
            let t = f.get(n).unwrap_or_else(|| panic!("missing tensor {n}"));
            F16Slice::from_tensor(n, t).unwrap_or_else(|e| panic!("{e}"))
        };

        let enc = (0..N_ENC).map(|i| {
            let p = format!("model.encoder.layers.{i}");
            EncLayer {
                ln1_w: g(&format!("{p}.self_attn_layer_norm.weight")), ln1_b: g(&format!("{p}.self_attn_layer_norm.bias")),
                ln2_w: g(&format!("{p}.final_layer_norm.weight")), ln2_b: g(&format!("{p}.final_layer_norm.bias")),
                q: g16(&format!("{p}.self_attn.q_proj.weight")), q_b: g(&format!("{p}.self_attn.q_proj.bias")),
                k: g16(&format!("{p}.self_attn.k_proj.weight")),
                v: g16(&format!("{p}.self_attn.v_proj.weight")), v_b: g(&format!("{p}.self_attn.v_proj.bias")),
                o: g16(&format!("{p}.self_attn.out_proj.weight")), o_b: g(&format!("{p}.self_attn.out_proj.bias")),
                fc1: g16(&format!("{p}.fc1.weight")), fc1_b: g(&format!("{p}.fc1.bias")),
                fc2: g16(&format!("{p}.fc2.weight")), fc2_b: g(&format!("{p}.fc2.bias")),
            }
        }).collect();

        let dec = (0..N_DEC).map(|i| {
            let p = format!("model.decoder.layers.{i}");
            DecLayer {
                ln1_w: g(&format!("{p}.self_attn_layer_norm.weight")), ln1_b: g(&format!("{p}.self_attn_layer_norm.bias")),
                lnc_w: g(&format!("{p}.encoder_attn_layer_norm.weight")), lnc_b: g(&format!("{p}.encoder_attn_layer_norm.bias")),
                ln2_w: g(&format!("{p}.final_layer_norm.weight")), ln2_b: g(&format!("{p}.final_layer_norm.bias")),
                q: g16(&format!("{p}.self_attn.q_proj.weight")), q_b: g(&format!("{p}.self_attn.q_proj.bias")),
                k: g16(&format!("{p}.self_attn.k_proj.weight")),
                v: g16(&format!("{p}.self_attn.v_proj.weight")), v_b: g(&format!("{p}.self_attn.v_proj.bias")),
                o: g16(&format!("{p}.self_attn.out_proj.weight")), o_b: g(&format!("{p}.self_attn.out_proj.bias")),
                cq: g16(&format!("{p}.encoder_attn.q_proj.weight")), cq_b: g(&format!("{p}.encoder_attn.q_proj.bias")),
                ck: g16(&format!("{p}.encoder_attn.k_proj.weight")),
                cv: g16(&format!("{p}.encoder_attn.v_proj.weight")), cv_b: g(&format!("{p}.encoder_attn.v_proj.bias")),
                co: g16(&format!("{p}.encoder_attn.out_proj.weight")), co_b: g(&format!("{p}.encoder_attn.out_proj.bias")),
                fc1: g16(&format!("{p}.fc1.weight")), fc1_b: g(&format!("{p}.fc1.bias")),
                fc2: g16(&format!("{p}.fc2.weight")), fc2_b: g(&format!("{p}.fc2.bias")),
            }
        }).collect();

        let (id2piece, byte_decoder) = load_tokenizer(&dir.join("tokenizer.json"))?;

        Ok(Self {
            conv1_w: g("model.encoder.conv1.weight"), conv1_b: g("model.encoder.conv1.bias"),
            conv2_w: g("model.encoder.conv2.weight"), conv2_b: g("model.encoder.conv2.bias"),
            enc_pos: g("model.encoder.embed_positions.weight"),
            enc,
            enc_ln_w: g("model.encoder.layer_norm.weight"), enc_ln_b: g("model.encoder.layer_norm.bias"),
            embed: g("model.decoder.embed_tokens.weight"),
            dec_pos: g("model.decoder.embed_positions.weight"),
            dec,
            dec_ln_w: g("model.decoder.layer_norm.weight"), dec_ln_b: g("model.decoder.layer_norm.bias"),
            mel_filters: build_mel_filters(),
            id2piece,
            byte_decoder,
        })
    }

    /// conv stem + sinusoidal positions + 32 encoder layers → `[T, D_MODEL]`.
    pub fn encode(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        let (mel, n_frames) = log_mel(audio, &self.mel_filters);
        let (h1, l1) = conv1d(&mel, N_MEL, n_frames, &self.conv1_w, D_MODEL, 3, 1, 1, &self.conv1_b);
        let h1: Vec<f32> = h1.iter().map(|x| gelu(*x)).collect();
        let (h2, l2) = conv1d(&h1, D_MODEL, l1, &self.conv2_w, D_MODEL, 3, 2, 1, &self.conv2_b);
        let h2: Vec<f32> = h2.iter().map(|x| gelu(*x)).collect();
        let t = l2; // encoder positions
        // transpose [D_MODEL, t] → [t, D_MODEL] and add stored positions.
        let mut hidden = vec![0.0f32; t * D_MODEL];
        for c in 0..D_MODEL {
            for tt in 0..t { hidden[tt * D_MODEL + c] = h2[c * l2 + tt] + self.enc_pos[tt * D_MODEL + c]; }
        }

        for layer in &self.enc {
            let mut xn = hidden.clone();
            layernorm(&mut xn, D_MODEL, &layer.ln1_w, &layer.ln1_b);
            let attn = attend_full(&xn, t, &layer.q, Some(&layer.q_b), &layer.k, None, &layer.v, Some(&layer.v_b), &layer.o, &layer.o_b);
            for i in 0..hidden.len() { hidden[i] += attn[i]; }
            let mut xn = hidden.clone();
            layernorm(&mut xn, D_MODEL, &layer.ln2_w, &layer.ln2_b);
            let mut hf = linear_f16(&xn, t, &layer.fc1, FFN, D_MODEL, Some(&layer.fc1_b));
            for z in hf.iter_mut() { *z = gelu(*z); }
            let ff = linear_f16(&hf, t, &layer.fc2, D_MODEL, FFN, Some(&layer.fc2_b));
            for i in 0..hidden.len() { hidden[i] += ff[i]; }
        }
        layernorm(&mut hidden, D_MODEL, &self.enc_ln_w, &self.enc_ln_b);
        (hidden, t)
    }

    pub fn transcribe(&self, audio: &[f32], max_new: usize) -> String {
        let (enc_h, t_enc) = self.encode(audio);
        // Per-layer cross-attn K/V from encoder output (k_proj has no bias).
        let cross_kv: Vec<(Vec<f32>, Vec<f32>)> = self.dec.iter().map(|l| {
            let ck = linear_f16(&enc_h, t_enc, &l.ck, D_MODEL, D_MODEL, None);
            let cv = linear_f16(&enc_h, t_enc, &l.cv, D_MODEL, D_MODEL, Some(&l.cv_b));
            (ck, cv)
        }).collect();

        let mut self_k: Vec<Vec<f32>> = vec![Vec::new(); N_DEC];
        let mut self_v: Vec<Vec<f32>> = vec![Vec::new(); N_DEC];
        let prefix = [SOT, LANG_EN, TRANSCRIBE, NO_TS];
        let sc = scale();

        let mut out_ids: Vec<i32> = Vec::new();
        let mut pos = 0usize;
        let mut next: i32 = -1;
        let total = prefix.len() + max_new;
        for step in 0..total {
            let tok = if step < prefix.len() { prefix[step] } else { next };
            if step >= prefix.len() {
                if tok == EOS { break; }
                out_ids.push(tok);
            }
            // embed + learned position
            let mut x = vec![0.0f32; D_MODEL];
            let e = &self.embed[tok as usize * D_MODEL..(tok as usize + 1) * D_MODEL];
            let p = &self.dec_pos[pos * D_MODEL..(pos + 1) * D_MODEL];
            for i in 0..D_MODEL { x[i] = e[i] + p[i]; }

            for (li, layer) in self.dec.iter().enumerate() {
                // causal self-attn
                let mut xn = x.clone();
                layernorm(&mut xn, D_MODEL, &layer.ln1_w, &layer.ln1_b);
                let q = linear_f16(&xn, 1, &layer.q, D_MODEL, D_MODEL, Some(&layer.q_b));
                let kk = linear_f16(&xn, 1, &layer.k, D_MODEL, D_MODEL, None);
                let vv = linear_f16(&xn, 1, &layer.v, D_MODEL, D_MODEL, Some(&layer.v_b));
                self_k[li].extend_from_slice(&kk);
                self_v[li].extend_from_slice(&vv);
                let seq = pos + 1;
                let mut attn = vec![0.0f32; D_MODEL];
                for h in 0..N_HEADS {
                    let qh = &q[h * D_HEAD..(h + 1) * D_HEAD];
                    let oh = fused_attend_head(qh, &self_k[li], &self_v[li], h, seq, D_MODEL, D_HEAD, sc);
                    attn[h * D_HEAD..(h + 1) * D_HEAD].copy_from_slice(&oh);
                }
                let attn = linear_f16(&attn, 1, &layer.o, D_MODEL, D_MODEL, Some(&layer.o_b));
                for i in 0..D_MODEL { x[i] += attn[i]; }

                // cross-attn
                let mut xn = x.clone();
                layernorm(&mut xn, D_MODEL, &layer.lnc_w, &layer.lnc_b);
                let cq = linear_f16(&xn, 1, &layer.cq, D_MODEL, D_MODEL, Some(&layer.cq_b));
                let (ck, cv) = &cross_kv[li];
                let mut cattn = vec![0.0f32; D_MODEL];
                for h in 0..N_HEADS {
                    let qh = &cq[h * D_HEAD..(h + 1) * D_HEAD];
                    let oh = fused_attend_head(qh, ck, cv, h, t_enc, D_MODEL, D_HEAD, sc);
                    cattn[h * D_HEAD..(h + 1) * D_HEAD].copy_from_slice(&oh);
                }
                let cattn = linear_f16(&cattn, 1, &layer.co, D_MODEL, D_MODEL, Some(&layer.co_b));
                for i in 0..D_MODEL { x[i] += cattn[i]; }

                // GELU MLP
                let mut xn = x.clone();
                layernorm(&mut xn, D_MODEL, &layer.ln2_w, &layer.ln2_b);
                let mut hf = linear_f16(&xn, 1, &layer.fc1, FFN, D_MODEL, Some(&layer.fc1_b));
                for z in hf.iter_mut() { *z = gelu(*z); }
                let ff = linear_f16(&hf, 1, &layer.fc2, D_MODEL, FFN, Some(&layer.fc2_b));
                for i in 0..D_MODEL { x[i] += ff[i]; }
            }
            layernorm(&mut x, D_MODEL, &self.dec_ln_w, &self.dec_ln_b);
            let logits = linear(&x, 1, &self.embed, VOCAB, D_MODEL, None);
            let mut best = 0i32; let mut bv = f32::NEG_INFINITY;
            for (i, &l) in logits.iter().enumerate() { if l > bv { bv = l; best = i as i32; } }
            next = best;
            pos += 1;
        }
        self.decode(&out_ids)
    }

    fn decode(&self, ids: &[i32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if id >= SPECIAL_MIN { continue; }
            let Some(p) = self.id2piece.get(&id) else { continue };
            for ch in p.chars() {
                if let Some(&b) = self.byte_decoder.get(&ch) { bytes.push(b); }
            }
        }
        String::from_utf8_lossy(&bytes).trim().to_string()
    }
}

/// Full-sequence bidirectional self-attention (encoder), no rotary.
/// Takes f16 weight slices (borrow-direct from mmap) and dequant-on-call
/// via Apple Accelerate sgemm inside `linear_f16`.
#[allow(clippy::too_many_arguments)]
fn attend_full(
    xn: &[f32], t: usize,
    qw: &[half::f16], qb: Option<&[f32]>, kw: &[half::f16], kb: Option<&[f32]>,
    vw: &[half::f16], vb: Option<&[f32]>, ow: &[half::f16], ob: &[f32],
) -> Vec<f32> {
    let q = linear_f16(xn, t, qw, D_MODEL, D_MODEL, qb);
    let k = linear_f16(xn, t, kw, D_MODEL, D_MODEL, kb);
    let vv = linear_f16(xn, t, vw, D_MODEL, D_MODEL, vb);
    let sc = scale();
    let mut attn = vec![0.0f32; t * D_MODEL];
    attn.par_chunks_mut(D_MODEL).enumerate().for_each(|(i, arow)| {
        for h in 0..N_HEADS {
            let qh = &q[i * D_MODEL + h * D_HEAD..i * D_MODEL + h * D_HEAD + D_HEAD];
            let oh = fused_attend_head(qh, &k, &vv, h, t, D_MODEL, D_HEAD, sc);
            arow[h * D_HEAD..(h + 1) * D_HEAD].copy_from_slice(&oh);
        }
    });
    linear_f16(&attn, t, ow, D_MODEL, D_MODEL, Some(ob))
}

// ── tokenizer (GPT-2 ByteLevel BPE detokenization) ─────────────────

fn bytes_to_unicode() -> HashMap<char, u8> {
    let mut bs: Vec<u32> = Vec::new();
    bs.extend(b'!' as u32..=b'~' as u32);
    bs.extend(0xA1..=0xAC);
    bs.extend(0xAE..=0xFF);
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0..256u32 {
        if !bs.contains(&b) { bs.push(b); cs.push(256 + n); n += 1; }
    }
    let mut rev = HashMap::new();
    for (b, c) in bs.iter().zip(cs.iter()) {
        rev.insert(char::from_u32(*c).unwrap(), *b as u8);
    }
    rev
}

fn load_tokenizer(path: &Path) -> Result<(HashMap<i32, String>, HashMap<char, u8>), String> {
    let txt = std::fs::read_to_string(path).map_err(|e| format!("read tokenizer.json: {e}"))?;
    let json: serde_json::Value = serde_json::from_str(&txt).map_err(|e| format!("parse: {e}"))?;
    let mut map = HashMap::new();
    if let Some(obj) = json.get("model").and_then(|m| m.get("vocab")).and_then(|v| v.as_object()) {
        for (piece, id) in obj {
            if let Some(i) = id.as_i64() { map.insert(i as i32, piece.clone()); }
        }
    }
    if map.is_empty() { return Err("empty vocab".into()); }
    Ok((map, bytes_to_unicode()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_dir() -> Option<String> {
        for base in ["macos_4TB_external", "macos_4TB_external_1", "macos_4TB_external_2"] {
            let d = format!("/Volumes/{base}/neuronexus_ai/model_library/whisper-large-v3-turbo");
            if std::path::Path::new(&d).join("model.safetensors").exists() { return Some(d); }
        }
        None
    }
    const PCM: &str = "/tmp/moonshine_test.pcm"; // reuse the same 16k mono s16le clip

    fn read_pcm(path: &str) -> Option<Vec<f32>> {
        let b = std::fs::read(path).ok()?;
        Some(b.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect())
    }

    #[test]
    fn whisper_load_sanity() {
        let Some(dir) = model_dir() else { eprintln!("whisper: model not mounted — skipping"); return };
        let m = Whisper::load(&dir).expect("load");
        assert_eq!(m.enc.len(), N_ENC);
        assert_eq!(m.dec.len(), N_DEC);
        assert_eq!(m.embed.len(), VOCAB * D_MODEL);
        assert_eq!(m.mel_filters.len(), N_MEL * N_BINS);
        assert!(m.id2piece.len() > 50000);
    }

    #[test]
    fn whisper_transcribe() {
        let Some(dir) = model_dir() else { eprintln!("whisper: model not mounted — skipping"); return };
        let Some(audio) = read_pcm(PCM) else {
            eprintln!("whisper: {PCM} not found — run ffmpeg to create it"); return;
        };
        let m = Whisper::load(&dir).expect("load");
        let t0 = std::time::Instant::now();
        let text = m.transcribe(&audio, 200);
        eprintln!("whisper transcript ({:.1}s audio, {:?}): {text:?}",
            audio.len() as f32 / 16000.0, t0.elapsed());
        let alpha = text.chars().filter(|c| c.is_alphabetic()).count();
        assert!(alpha >= 5, "transcript looks empty/garbage: {text:?}");
    }
}
