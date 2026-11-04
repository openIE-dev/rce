//! NVIDIA Canary-1b-flash ASR (FastConformer attention-encoder-decoder)
//! on the rce substrate. rce's first Conformer / relative-position
//! attention model.
//!
//! Pipeline (verified vs NeMo source + the model's `.nemo` config):
//! - **NeMo mel**: preemph 0.97 → STFT `n_fft=512, win=400, hop=160`
//!   (periodic Hann) → power → **HTK-128** mel (Slaney area-norm) → ln →
//!   **per-feature** normalize (per-mel mean/std over time).
//! - **dw_striding 8× subsampling**: conv2d(1→256,k3,s2)+ReLU →
//!   [depthwise k3 s2 + pointwise k1]+ReLU ×2 → flatten 256·16 → Linear→1024.
//! - **32 Conformer blocks** (macaron): ½·FFN(swish) → rel-pos MHSA
//!   (Transformer-XL: linear_pos + per-layer pos_bias_u/v + rel_shift) →
//!   conv module (pointwise→GLU→depthwise k9→BatchNorm→swish→pointwise) →
//!   ½·FFN → norm_out. 8 heads × 128.
//! - **4-layer pre-LN Transformer decoder**: embed(token+sinusoidal, LN) →
//!   [LN→self-attn]+res, [LN→cross-attn]+res, [LN→ReLU-FFN]+res → final LN
//!   → tied head (token_embedding + bias). vocab 5248.
//! - Aggregate CanaryTokenizer (spl 1152 + en/de/es/fr 1024 each). canary2
//!   English-ASR prompt; greedy decode to `<|endoftext|>`.
//!
//! Encoder weights: `model.safetensors`. Decoder+head: offline-converted
//! from the `.nemo` PyTorch ckpt → `decoder_head.safetensors`. Detok vocab:
//! `canary_vocab.txt` (5248 lines). All three live in the model dir.

use rce_weight_loader::{LoadedTensor, SafetensorsFile};
use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use std::path::Path;

const D: usize = 1024;
const H: usize = 8;
const DK: usize = 128; // D / H
const N_ENC: usize = 32;
const N_DEC: usize = 4;
const FF_ENC: usize = 4096;
const FF_DEC: usize = 4096;
const VOCAB: usize = 5248;
const CONV_K: usize = 9;

// mel
const N_FFT: usize = 512;
const WIN: usize = 400;
const HOP: usize = 160;
const N_MEL: usize = 128;
const N_BINS: usize = N_FFT / 2 + 1; // 257
const SR: usize = 16000;
const PREEMPH: f32 = 0.97;
const LN_EPS: f32 = 1e-5;
const BN_EPS: f32 = 1e-5;

// canary2 English-ASR prompt (ids into the aggregate vocab) + eos
const PROMPT: [i32; 9] = [7, 4, 16, 62, 62, 5, 9, 11, 13];
//  startofcontext, startoftranscript, emo:undefined, en(src), en(tgt), pnc, noitn, notimestamp, nodiarize
const EOS: i32 = 3; // <|endoftext|>

#[inline] fn silu(x: f32) -> f32 { if x >= 0.0 { x / (1.0 + (-x).exp()) } else { let e = x.exp(); x * e / (1.0 + e) } }
#[inline] fn relu(x: f32) -> f32 { x.max(0.0) }
#[inline] fn sigmoid(x: f32) -> f32 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) } }

fn tf32(t: &LoadedTensor) -> Vec<f32> {
    if let Some(s) = t.as_f32() { return s.to_vec(); }
    match t.dtype.as_str() {
        "f16" => t.data.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        "bf16" => t.data.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        d => panic!("dtype {d}"),
    }
}

/// `y[t, dout] = x[t, din] · W[dout, din]^T (+bias)`, parallel over rows.
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

/// LayerNorm (weight+bias) over last dim `d`, per row.
fn layernorm(x: &mut [f32], d: usize, w: &[f32], b: &[f32]) {
    x.par_chunks_mut(d).for_each(|row| {
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| { let z = v - mean; z * z }).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        for ((v, g), bb) in row.iter_mut().zip(w).zip(b) { *v = (*v - mean) * inv * *g + *bb; }
    });
}

// ── mel frontend (NeMo AudioToMelSpectrogram) ──────────────────────

fn hz_to_mel_htk(f: f32) -> f32 { 2595.0 * (1.0 + f / 700.0).log10() }
fn mel_to_hz_htk(m: f32) -> f32 { 700.0 * (10f32.powf(m / 2595.0) - 1.0) }

fn build_mel_filters() -> Vec<f32> {
    let (mmin, mmax) = (hz_to_mel_htk(0.0), hz_to_mel_htk(8000.0));
    let fpts: Vec<f32> = (0..N_MEL + 2).map(|i| mel_to_hz_htk(mmin + (mmax - mmin) * i as f32 / (N_MEL + 1) as f32)).collect();
    let fft_freqs: Vec<f32> = (0..N_BINS).map(|i| (SR / 2) as f32 * i as f32 / (N_BINS - 1) as f32).collect();
    let mut filt = vec![0.0f32; N_MEL * N_BINS];
    for m in 0..N_MEL {
        let (l, c, r) = (fpts[m], fpts[m + 1], fpts[m + 2]);
        let enorm = 2.0 / (r - l); // slaney area norm
        for (k, &fk) in fft_freqs.iter().enumerate() {
            let w = ((fk - l) / (c - l)).min((r - fk) / (r - c)).max(0.0);
            filt[m * N_BINS + k] = w * enorm;
        }
    }
    filt
}

/// NeMo log-mel: preemph → STFT(win=400 in 512-fft, periodic Hann) →
/// power → HTK mel → ln → per-feature (per-mel) mean/std normalize.
/// Returns `([N_MEL, T], T)` channel-major.
fn log_mel(audio: &[f32], filters: &[f32]) -> (Vec<f32>, usize) {
    // preemphasis y[n] = x[n] - 0.97 x[n-1]
    let mut sig = vec![0.0f32; audio.len()];
    sig[0] = audio[0];
    for n in 1..audio.len() { sig[n] = audio[n] - PREEMPH * audio[n - 1]; }

    let window: Vec<f32> = (0..WIN).map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / WIN as f32).cos()).collect();
    // NeMo centers the win-400 frame inside the 512 FFT buffer, center pad by n_fft/2.
    let pad = N_FFT / 2;
    let mut padded = vec![0.0f32; sig.len() + 2 * pad];
    for i in 0..pad { padded[i] = sig[pad - i]; }
    padded[pad..pad + sig.len()].copy_from_slice(&sig);
    for i in 0..pad { padded[pad + sig.len() + i] = sig[sig.len() - 2 - i]; }

    let n_frames = 1 + (padded.len() - N_FFT) / HOP;
    let off = (N_FFT - WIN) / 2; // window centered in the 512 buffer
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);

    // per-frame mel (parallel)
    let cols: Vec<Vec<f32>> = (0..n_frames).into_par_iter().map(|f| {
        let mut buf = vec![Complex32::new(0.0, 0.0); N_FFT];
        for i in 0..WIN { buf[off + i] = Complex32::new(padded[f * HOP + i] * window[i], 0.0); }
        fft.process(&mut buf);
        let mut col = vec![0.0f32; N_MEL];
        for m in 0..N_MEL {
            let frow = &filters[m * N_BINS..(m + 1) * N_BINS];
            let mut acc = 0.0f32;
            for k in 0..N_BINS { let p = buf[k].re * buf[k].re + buf[k].im * buf[k].im; acc += frow[k] * p; }
            col[m] = (acc + 1e-9).ln(); // NeMo log(x + 2^-24≈1e-7); 1e-9 close enough post-norm
        }
        col
    }).collect();

    // [N_MEL, T] channel-major
    let t = n_frames;
    let mut mel = vec![0.0f32; N_MEL * t];
    for (f, col) in cols.iter().enumerate() { for m in 0..N_MEL { mel[m * t + f] = col[m]; } }
    // per-feature normalization: per mel channel, subtract mean / divide std over time.
    for m in 0..N_MEL {
        let row = &mut mel[m * t..(m + 1) * t];
        let mean = row.iter().sum::<f32>() / t as f32;
        let var = row.iter().map(|v| { let z = v - mean; z * z }).sum::<f32>() / t as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for v in row.iter_mut() { *v = (*v - mean) * inv; }
    }
    (mel, t)
}

// ── 2D conv (subsampling) ──────────────────────────────────────────

/// conv2d over `[Cin, H, W]` → `[Cout, Hout, Wout]`, with `groups`.
/// weight `[Cout, Cin/groups, kh, kw]`.
#[allow(clippy::too_many_arguments)]
fn conv2d(inp: &[f32], cin: usize, h: usize, w: usize, weight: &[f32], cout: usize, kh: usize, kw: usize, stride: usize, pad: usize, groups: usize, bias: &[f32]) -> (Vec<f32>, usize, usize) {
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

// ── weights ────────────────────────────────────────────────────────

struct EncLayer {
    nff1_w: Vec<f32>, nff1_b: Vec<f32>, ff1_1w: Vec<f32>, ff1_1b: Vec<f32>, ff1_2w: Vec<f32>, ff1_2b: Vec<f32>,
    nsa_w: Vec<f32>, nsa_b: Vec<f32>,
    q_w: Vec<f32>, q_b: Vec<f32>, k_w: Vec<f32>, k_b: Vec<f32>, v_w: Vec<f32>, v_b: Vec<f32>,
    out_w: Vec<f32>, out_b: Vec<f32>, pos_w: Vec<f32>, pbu: Vec<f32>, pbv: Vec<f32>,
    nconv_w: Vec<f32>, nconv_b: Vec<f32>,
    pw1_w: Vec<f32>, pw1_b: Vec<f32>, dw_w: Vec<f32>, dw_b: Vec<f32>,
    bn_w: Vec<f32>, bn_b: Vec<f32>, bn_m: Vec<f32>, bn_v: Vec<f32>, pw2_w: Vec<f32>, pw2_b: Vec<f32>,
    nff2_w: Vec<f32>, nff2_b: Vec<f32>, ff2_1w: Vec<f32>, ff2_1b: Vec<f32>, ff2_2w: Vec<f32>, ff2_2b: Vec<f32>,
    nout_w: Vec<f32>, nout_b: Vec<f32>,
}
struct DecLayer {
    ln1_w: Vec<f32>, ln1_b: Vec<f32>, ln2_w: Vec<f32>, ln2_b: Vec<f32>, ln3_w: Vec<f32>, ln3_b: Vec<f32>,
    sq_w: Vec<f32>, sq_b: Vec<f32>, sk_w: Vec<f32>, sk_b: Vec<f32>, sv_w: Vec<f32>, sv_b: Vec<f32>, so_w: Vec<f32>, so_b: Vec<f32>,
    cq_w: Vec<f32>, cq_b: Vec<f32>, ck_w: Vec<f32>, ck_b: Vec<f32>, cv_w: Vec<f32>, cv_b: Vec<f32>, co_w: Vec<f32>, co_b: Vec<f32>,
    fi_w: Vec<f32>, fi_b: Vec<f32>, fo_w: Vec<f32>, fo_b: Vec<f32>,
}

pub struct Canary {
    // subsampling
    sub0_w: Vec<f32>, sub0_b: Vec<f32>, sub2_w: Vec<f32>, sub2_b: Vec<f32>, sub3_w: Vec<f32>, sub3_b: Vec<f32>,
    sub5_w: Vec<f32>, sub5_b: Vec<f32>, sub6_w: Vec<f32>, sub6_b: Vec<f32>, sub_out_w: Vec<f32>, sub_out_b: Vec<f32>,
    enc: Vec<EncLayer>,
    // decoder
    tok_emb: Vec<f32>, pos_enc: Vec<f32>, emb_ln_w: Vec<f32>, emb_ln_b: Vec<f32>,
    dec: Vec<DecLayer>, final_ln_w: Vec<f32>, final_ln_b: Vec<f32>,
    head_b: Vec<f32>,
    mel_filters: Vec<f32>,
    vocab: Vec<String>,
}

impl Canary {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = dir.as_ref();
        let enc_f = SafetensorsFile::load(dir.join("model.safetensors")).map_err(|e| format!("enc load: {e:?}"))?;
        let dec_f = SafetensorsFile::load(dir.join("decoder_head.safetensors")).map_err(|e| format!("dec load: {e:?}"))?;
        let ge = |n: &str| tf32(enc_f.get(n).unwrap_or_else(|| panic!("enc missing {n}")));
        let gd = |n: &str| tf32(dec_f.get(n).unwrap_or_else(|| panic!("dec missing {n}")));

        let enc = (0..N_ENC).map(|i| {
            let p = format!("encoder.layers.{i}");
            EncLayer {
                nff1_w: ge(&format!("{p}.norm_feed_forward1.weight")), nff1_b: ge(&format!("{p}.norm_feed_forward1.bias")),
                ff1_1w: ge(&format!("{p}.feed_forward1.linear1.weight")), ff1_1b: ge(&format!("{p}.feed_forward1.linear1.bias")),
                ff1_2w: ge(&format!("{p}.feed_forward1.linear2.weight")), ff1_2b: ge(&format!("{p}.feed_forward1.linear2.bias")),
                nsa_w: ge(&format!("{p}.norm_self_att.weight")), nsa_b: ge(&format!("{p}.norm_self_att.bias")),
                q_w: ge(&format!("{p}.self_attn.linear_q.weight")), q_b: ge(&format!("{p}.self_attn.linear_q.bias")),
                k_w: ge(&format!("{p}.self_attn.linear_k.weight")), k_b: ge(&format!("{p}.self_attn.linear_k.bias")),
                v_w: ge(&format!("{p}.self_attn.linear_v.weight")), v_b: ge(&format!("{p}.self_attn.linear_v.bias")),
                out_w: ge(&format!("{p}.self_attn.linear_out.weight")), out_b: ge(&format!("{p}.self_attn.linear_out.bias")),
                pos_w: ge(&format!("{p}.self_attn.linear_pos.weight")),
                pbu: ge(&format!("{p}.self_attn.pos_bias_u")), pbv: ge(&format!("{p}.self_attn.pos_bias_v")),
                nconv_w: ge(&format!("{p}.norm_conv.weight")), nconv_b: ge(&format!("{p}.norm_conv.bias")),
                pw1_w: ge(&format!("{p}.conv.pointwise_conv1.weight")), pw1_b: ge(&format!("{p}.conv.pointwise_conv1.bias")),
                dw_w: ge(&format!("{p}.conv.depthwise_conv.weight")), dw_b: ge(&format!("{p}.conv.depthwise_conv.bias")),
                bn_w: ge(&format!("{p}.conv.batch_norm.weight")), bn_b: ge(&format!("{p}.conv.batch_norm.bias")),
                bn_m: ge(&format!("{p}.conv.batch_norm.running_mean")), bn_v: ge(&format!("{p}.conv.batch_norm.running_var")),
                pw2_w: ge(&format!("{p}.conv.pointwise_conv2.weight")), pw2_b: ge(&format!("{p}.conv.pointwise_conv2.bias")),
                nff2_w: ge(&format!("{p}.norm_feed_forward2.weight")), nff2_b: ge(&format!("{p}.norm_feed_forward2.bias")),
                ff2_1w: ge(&format!("{p}.feed_forward2.linear1.weight")), ff2_1b: ge(&format!("{p}.feed_forward2.linear1.bias")),
                ff2_2w: ge(&format!("{p}.feed_forward2.linear2.weight")), ff2_2b: ge(&format!("{p}.feed_forward2.linear2.bias")),
                nout_w: ge(&format!("{p}.norm_out.weight")), nout_b: ge(&format!("{p}.norm_out.bias")),
            }
        }).collect();

        let dec = (0..N_DEC).map(|i| {
            let p = format!("transf_decoder._decoder.layers.{i}");
            DecLayer {
                ln1_w: gd(&format!("{p}.layer_norm_1.weight")), ln1_b: gd(&format!("{p}.layer_norm_1.bias")),
                ln2_w: gd(&format!("{p}.layer_norm_2.weight")), ln2_b: gd(&format!("{p}.layer_norm_2.bias")),
                ln3_w: gd(&format!("{p}.layer_norm_3.weight")), ln3_b: gd(&format!("{p}.layer_norm_3.bias")),
                sq_w: gd(&format!("{p}.first_sub_layer.query_net.weight")), sq_b: gd(&format!("{p}.first_sub_layer.query_net.bias")),
                sk_w: gd(&format!("{p}.first_sub_layer.key_net.weight")), sk_b: gd(&format!("{p}.first_sub_layer.key_net.bias")),
                sv_w: gd(&format!("{p}.first_sub_layer.value_net.weight")), sv_b: gd(&format!("{p}.first_sub_layer.value_net.bias")),
                so_w: gd(&format!("{p}.first_sub_layer.out_projection.weight")), so_b: gd(&format!("{p}.first_sub_layer.out_projection.bias")),
                cq_w: gd(&format!("{p}.second_sub_layer.query_net.weight")), cq_b: gd(&format!("{p}.second_sub_layer.query_net.bias")),
                ck_w: gd(&format!("{p}.second_sub_layer.key_net.weight")), ck_b: gd(&format!("{p}.second_sub_layer.key_net.bias")),
                cv_w: gd(&format!("{p}.second_sub_layer.value_net.weight")), cv_b: gd(&format!("{p}.second_sub_layer.value_net.bias")),
                co_w: gd(&format!("{p}.second_sub_layer.out_projection.weight")), co_b: gd(&format!("{p}.second_sub_layer.out_projection.bias")),
                fi_w: gd(&format!("{p}.third_sub_layer.dense_in.weight")), fi_b: gd(&format!("{p}.third_sub_layer.dense_in.bias")),
                fo_w: gd(&format!("{p}.third_sub_layer.dense_out.weight")), fo_b: gd(&format!("{p}.third_sub_layer.dense_out.bias")),
            }
        }).collect();

        let vocab: Vec<String> = std::fs::read_to_string(dir.join("canary_vocab.txt"))
            .map_err(|e| format!("vocab: {e}"))?.lines().map(|l| l.to_string()).collect();
        if vocab.len() != VOCAB { return Err(format!("vocab {} != {VOCAB}", vocab.len())); }

        Ok(Self {
            sub0_w: ge("encoder.subsampling.conv.0.weight"), sub0_b: ge("encoder.subsampling.conv.0.bias"),
            sub2_w: ge("encoder.subsampling.conv.2.weight"), sub2_b: ge("encoder.subsampling.conv.2.bias"),
            sub3_w: ge("encoder.subsampling.conv.3.weight"), sub3_b: ge("encoder.subsampling.conv.3.bias"),
            sub5_w: ge("encoder.subsampling.conv.5.weight"), sub5_b: ge("encoder.subsampling.conv.5.bias"),
            sub6_w: ge("encoder.subsampling.conv.6.weight"), sub6_b: ge("encoder.subsampling.conv.6.bias"),
            sub_out_w: ge("encoder.subsampling.out.weight"), sub_out_b: ge("encoder.subsampling.out.bias"),
            enc,
            tok_emb: gd("transf_decoder._embedding.token_embedding.weight"),
            pos_enc: gd("transf_decoder._embedding.position_embedding.pos_enc"),
            emb_ln_w: gd("transf_decoder._embedding.layer_norm.weight"), emb_ln_b: gd("transf_decoder._embedding.layer_norm.bias"),
            dec,
            final_ln_w: gd("transf_decoder._decoder.final_layer_norm.weight"), final_ln_b: gd("transf_decoder._decoder.final_layer_norm.bias"),
            head_b: gd("log_softmax.mlp.layer0.bias"),
            mel_filters: build_mel_filters(),
            vocab,
        })
    }

    pub fn encode(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        let (mel, tf) = log_mel(audio, &self.mel_filters); // [N_MEL, T]
        // image [1, H=T, W=128]: transpose mel.
        let mut img = vec![0.0f32; tf * N_MEL];
        for m in 0..N_MEL { for t in 0..tf { img[t * N_MEL + m] = mel[m * tf + t]; } }

        let (c0, h0, w0) = conv2d(&img, 1, tf, N_MEL, &self.sub0_w, 256, 3, 3, 2, 1, 1, &self.sub0_b);
        let c0: Vec<f32> = c0.iter().map(|x| relu(*x)).collect();
        let (c2, h2, w2) = conv2d(&c0, 256, h0, w0, &self.sub2_w, 256, 3, 3, 2, 1, 256, &self.sub2_b);
        let (c3, h3, w3) = conv2d(&c2, 256, h2, w2, &self.sub3_w, 256, 1, 1, 1, 0, 1, &self.sub3_b);
        let c3: Vec<f32> = c3.iter().map(|x| relu(*x)).collect();
        let (c5, h5, w5) = conv2d(&c3, 256, h3, w3, &self.sub5_w, 256, 3, 3, 2, 1, 256, &self.sub5_b);
        let (c6, h6, w6) = conv2d(&c5, 256, h5, w5, &self.sub6_w, 256, 1, 1, 1, 0, 1, &self.sub6_b);
        let c6: Vec<f32> = c6.iter().map(|x| relu(*x)).collect();
        // flatten [256, T', 16] → [T', 256*16] (channel outer, freq inner) → Linear→1024
        let (t, freq) = (h6, w6);
        let flat_dim = 256 * freq;
        let mut flat = vec![0.0f32; t * flat_dim];
        for c in 0..256 { for tt in 0..t { for f in 0..freq { flat[tt * flat_dim + c * freq + f] = c6[c * h6 * w6 + tt * w6 + f]; } } }
        let mut hidden = linear(&flat, t, &self.sub_out_w, D, flat_dim, Some(&self.sub_out_b));

        // relative positional embeddings: length 2T-1, p = linear_pos(pe)
        let pe = rel_pos_encoding(t);
        let p = linear(&pe, 2 * t - 1, &self.enc.first().unwrap().pos_w, D, D, None);
        // (linear_pos differs per layer; recompute p per layer below)
        let _ = p;

        for layer in &self.enc {
            // macaron FFN1 (half)
            let mut xn = hidden.clone();
            layernorm(&mut xn, D, &layer.nff1_w, &layer.nff1_b);
            let ff = ff_swish(&xn, t, &layer.ff1_1w, &layer.ff1_1b, &layer.ff1_2w, &layer.ff1_2b);
            for i in 0..hidden.len() { hidden[i] += 0.5 * ff[i]; }
            // rel-pos self-attn
            let mut xn = hidden.clone();
            layernorm(&mut xn, D, &layer.nsa_w, &layer.nsa_b);
            let attn = rel_pos_attn(&xn, t, &pe, layer);
            for i in 0..hidden.len() { hidden[i] += attn[i]; }
            // conv module
            let mut xn = hidden.clone();
            layernorm(&mut xn, D, &layer.nconv_w, &layer.nconv_b);
            let cv = conv_module(&xn, t, layer);
            for i in 0..hidden.len() { hidden[i] += cv[i]; }
            // macaron FFN2 (half)
            let mut xn = hidden.clone();
            layernorm(&mut xn, D, &layer.nff2_w, &layer.nff2_b);
            let ff = ff_swish(&xn, t, &layer.ff2_1w, &layer.ff2_1b, &layer.ff2_2w, &layer.ff2_2b);
            for i in 0..hidden.len() { hidden[i] += 0.5 * ff[i]; }
            // norm_out
            layernorm(&mut hidden, D, &layer.nout_w, &layer.nout_b);
        }
        (hidden, t)
    }

    pub fn transcribe(&self, audio: &[f32], max_new: usize) -> String {
        let (enc_h, t_enc) = self.encode(audio);
        let sc = 1.0 / (DK as f32).sqrt();
        // cross-attn K/V per layer from encoder output
        let cross: Vec<(Vec<f32>, Vec<f32>)> = self.dec.iter().map(|l| {
            (linear(&enc_h, t_enc, &l.ck_w, D, D, Some(&l.ck_b)), linear(&enc_h, t_enc, &l.cv_w, D, D, Some(&l.cv_b)))
        }).collect();
        let mut sk: Vec<Vec<f32>> = vec![Vec::new(); N_DEC];
        let mut sv: Vec<Vec<f32>> = vec![Vec::new(); N_DEC];
        let mut out_ids: Vec<i32> = Vec::new();
        let mut next: i32 = -1;
        let total = PROMPT.len() + max_new;
        for step in 0..total {
            let tok = if step < PROMPT.len() { PROMPT[step] } else { next };
            if step >= PROMPT.len() { if tok == EOS { break; } out_ids.push(tok); }
            let pos = step;
            // embed: token + sinusoidal pos, then LN
            let mut x = vec![0.0f32; D];
            let te = &self.tok_emb[tok as usize * D..(tok as usize + 1) * D];
            let pe = &self.pos_enc[pos * D..(pos + 1) * D];
            for i in 0..D { x[i] = te[i] + pe[i]; }
            layernorm(&mut x, D, &self.emb_ln_w, &self.emb_ln_b);

            for (li, layer) in self.dec.iter().enumerate() {
                // self-attn (pre-LN, causal)
                let mut xn = x.clone(); layernorm(&mut xn, D, &layer.ln1_w, &layer.ln1_b);
                let q = linear(&xn, 1, &layer.sq_w, D, D, Some(&layer.sq_b));
                let k = linear(&xn, 1, &layer.sk_w, D, D, Some(&layer.sk_b));
                let v = linear(&xn, 1, &layer.sv_w, D, D, Some(&layer.sv_b));
                sk[li].extend_from_slice(&k); sv[li].extend_from_slice(&v);
                let seq = pos + 1;
                let att = mha(&q, &sk[li], &sv[li], seq, sc);
                let att = linear(&att, 1, &layer.so_w, D, D, Some(&layer.so_b));
                for i in 0..D { x[i] += att[i]; }
                // cross-attn
                let mut xn = x.clone(); layernorm(&mut xn, D, &layer.ln2_w, &layer.ln2_b);
                let cq = linear(&xn, 1, &layer.cq_w, D, D, Some(&layer.cq_b));
                let (ck, cv) = &cross[li];
                let catt = mha(&cq, ck, cv, t_enc, sc);
                let catt = linear(&catt, 1, &layer.co_w, D, D, Some(&layer.co_b));
                for i in 0..D { x[i] += catt[i]; }
                // FFN (relu)
                let mut xn = x.clone(); layernorm(&mut xn, D, &layer.ln3_w, &layer.ln3_b);
                let mut hf = linear(&xn, 1, &layer.fi_w, FF_DEC, D, Some(&layer.fi_b));
                for z in hf.iter_mut() { *z = relu(*z); }
                let ff = linear(&hf, 1, &layer.fo_w, D, FF_DEC, Some(&layer.fo_b));
                for i in 0..D { x[i] += ff[i]; }
            }
            layernorm(&mut x, D, &self.final_ln_w, &self.final_ln_b);
            let logits = linear(&x, 1, &self.tok_emb, VOCAB, D, Some(&self.head_b)); // tied head
            let mut best = 0i32; let mut bv = f32::NEG_INFINITY;
            for (i, &l) in logits.iter().enumerate() { if l > bv { bv = l; best = i as i32; } }
            next = best;
        }
        self.detok(&out_ids)
    }

    fn detok(&self, ids: &[i32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let p = &self.vocab[id as usize];
            if p.starts_with('<') && p.ends_with('>') { continue; } // special markers
            // SentencePiece byte-fallback <0xXX>
            if p.len() == 6 && p.as_bytes()[0] == b'<' && p.as_bytes()[2] == b'x' {
                if let Ok(b) = u8::from_str_radix(&p[3..5], 16) { bytes.push(b); continue; }
            }
            for ch in p.chars() {
                if ch == '\u{2581}' { bytes.push(b' '); }
                else { let mut buf = [0u8; 4]; bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes()); }
            }
        }
        String::from_utf8_lossy(&bytes).trim().to_string()
    }
}

/// Standard scaled-dot MHA: single query `[D]` over `[seq, D]` K/V.
fn mha(q: &[f32], k: &[f32], v: &[f32], seq: usize, sc: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; D];
    for h in 0..H {
        let qh = &q[h * DK..(h + 1) * DK];
        let mut scores = vec![0.0f32; seq];
        let mut mx = f32::NEG_INFINITY;
        for j in 0..seq {
            let kj = &k[j * D + h * DK..j * D + h * DK + DK];
            let mut s = 0.0f32; for d in 0..DK { s += qh[d] * kj[d]; }
            s *= sc; scores[j] = s; if s > mx { mx = s; }
        }
        let mut sum = 0.0f32;
        for s in scores.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
        let inv = 1.0 / sum;
        let oh = &mut out[h * DK..(h + 1) * DK];
        for j in 0..seq {
            let w = scores[j] * inv;
            let vj = &v[j * D + h * DK..j * D + h * DK + DK];
            for d in 0..DK { oh[d] += w * vj[d]; }
        }
    }
    out
}

fn ff_swish(x: &[f32], t: usize, w1: &[f32], b1: &[f32], w2: &[f32], b2: &[f32]) -> Vec<f32> {
    let mut h = linear(x, t, w1, FF_ENC, D, Some(b1));
    for z in h.iter_mut() { *z = silu(*z); }
    linear(&h, t, w2, D, FF_ENC, Some(b2))
}

/// Relative sinusoidal positional encoding, `[2T-1, D]`. Index m encodes
/// relative position `(T-1) - m`.
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

/// Transformer-XL relative-shift on a per-head `[T, 2T-1]` matrix → `[T, T]`.
fn rel_shift(bd: &[f32], t: usize) -> Vec<f32> {
    let l = 2 * t - 1;
    // pad left of last dim by 1 → [t, l+1]
    let mut padded = vec![0.0f32; t * (l + 1)];
    for i in 0..t { for j in 0..l { padded[i * (l + 1) + (j + 1)] = bd[i * l + j]; } }
    // drop first row (first t elements), then view [t, l] and take first t cols
    let nf = &padded[t..]; // length t*l
    let mut out = vec![0.0f32; t * t];
    for i in 0..t { for m in 0..t { out[i * t + m] = nf[i * l + m]; } }
    out
}

/// Conformer rel-pos MHSA (Transformer-XL). Parallel over query rows.
fn rel_pos_attn(xn: &[f32], t: usize, pe: &[f32], layer: &EncLayer) -> Vec<f32> {
    let q = linear(xn, t, &layer.q_w, D, D, Some(&layer.q_b));
    let k = linear(xn, t, &layer.k_w, D, D, Some(&layer.k_b));
    let v = linear(xn, t, &layer.v_w, D, D, Some(&layer.v_b));
    let p = linear(pe, 2 * t - 1, &layer.pos_w, D, D, None); // [2t-1, D]
    let l = 2 * t - 1;
    let sc = 1.0 / (DK as f32).sqrt();

    // per head, compute attended output, then concat.
    let heads: Vec<Vec<f32>> = (0..H).into_par_iter().map(|h| {
        let pbu = &layer.pbu[h * DK..(h + 1) * DK];
        let pbv = &layer.pbv[h * DK..(h + 1) * DK];
        // matrix_bd_raw[i, m] = (q[i,h]+pbv) · p[m,h]
        let mut bd_raw = vec![0.0f32; t * l];
        for i in 0..t {
            let qi = &q[i * D + h * DK..i * D + h * DK + DK];
            for m in 0..l {
                let pm = &p[m * D + h * DK..m * D + h * DK + DK];
                let mut s = 0.0f32; for d in 0..DK { s += (qi[d] + pbv[d]) * pm[d]; }
                bd_raw[i * l + m] = s;
            }
        }
        let bd = rel_shift(&bd_raw, t); // [t, t]
        let mut oh = vec![0.0f32; t * DK];
        for i in 0..t {
            let qi = &q[i * D + h * DK..i * D + h * DK + DK];
            // scores[j] = ((qi+pbu)·k[j] + bd[i,j]) * sc
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

    // concat heads → [t, D]
    let mut attn = vec![0.0f32; t * D];
    for h in 0..H { for i in 0..t { attn[i * D + h * DK..i * D + h * DK + DK].copy_from_slice(&heads[h][i * DK..(i + 1) * DK]); } }
    linear(&attn, t, &layer.out_w, D, D, Some(&layer.out_b))
}

/// Conformer conv module: pointwise→GLU→depthwise(k9)→BatchNorm→swish→pointwise.
fn conv_module(xn: &[f32], t: usize, layer: &EncLayer) -> Vec<f32> {
    // pointwise_conv1: [2D, D] (1x1 conv = linear), over each time step → [t, 2D]
    let pw1 = linear(xn, t, &layer.pw1_w, 2 * D, D, Some(&layer.pw1_b));
    // GLU over channel: a = pw1[:D], b = pw1[D:2D]; glu = a * sigmoid(b) → [t, D]
    let mut glu = vec![0.0f32; t * D];
    for tt in 0..t { for c in 0..D { glu[tt * D + c] = pw1[tt * 2 * D + c] * sigmoid(pw1[tt * 2 * D + D + c]); } }
    // depthwise conv1d over time, per channel, k=9 pad=4. Work in [C, T] layout.
    let pad = (CONV_K - 1) / 2;
    let mut dw = vec![0.0f32; D * t];
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
    // BatchNorm (eval) per channel + swish, in [C, T]
    for c in 0..D {
        let inv = 1.0 / (layer.bn_v[c] + BN_EPS).sqrt();
        let (g, b, m) = (layer.bn_w[c], layer.bn_b[c], layer.bn_m[c]);
        let row = &mut dw[c * t..(c + 1) * t];
        for v in row.iter_mut() { let y = (*v - m) * inv * g + b; *v = silu(y); }
    }
    // back to [t, D]
    let mut act = vec![0.0f32; t * D];
    for c in 0..D { for tt in 0..t { act[tt * D + c] = dw[c * t + tt]; } }
    // pointwise_conv2: [D, D] (1x1)
    linear(&act, t, &layer.pw2_w, D, D, Some(&layer.pw2_b))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn model_dir() -> Option<String> {
        for b in ["macos_4TB_external", "macos_4TB_external_1", "macos_4TB_external_2"] {
            let d = format!("/Volumes/{b}/neuronexus_ai/model_library/canary-1b-flash");
            if std::path::Path::new(&d).join("decoder_head.safetensors").exists() { return Some(d); }
        }
        None
    }
    const PCM: &str = "/tmp/moonshine_test.pcm";
    fn read_pcm(p: &str) -> Option<Vec<f32>> {
        let b = std::fs::read(p).ok()?;
        Some(b.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect())
    }

    #[test]
    fn canary_load_sanity() {
        let Some(d) = model_dir() else { eprintln!("canary: not mounted — skipping"); return };
        let m = Canary::load(&d).expect("load");
        assert_eq!(m.enc.len(), N_ENC);
        assert_eq!(m.dec.len(), N_DEC);
        assert_eq!(m.tok_emb.len(), VOCAB * D);
        assert_eq!(m.vocab.len(), VOCAB);
        assert_eq!(m.mel_filters.len(), N_MEL * N_BINS);
    }

    #[test]
    fn canary_transcribe() {
        let Some(d) = model_dir() else { eprintln!("canary: not mounted — skipping"); return };
        let Some(audio) = read_pcm(PCM) else { eprintln!("canary: {PCM} missing"); return };
        let m = Canary::load(&d).expect("load");
        let t0 = std::time::Instant::now();
        let text = m.transcribe(&audio, 200);
        eprintln!("canary transcript ({:.1}s audio, {:?}): {text:?}", audio.len() as f32 / 16000.0, t0.elapsed());
        let alpha = text.chars().filter(|c| c.is_alphabetic()).count();
        assert!(alpha >= 5, "garbage: {text:?}");
    }
}
