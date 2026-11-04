//! facebook/wav2vec2-base-960h on rce.
//!
//! Pipeline:
//!   raw 16 kHz mono → 7-layer Conv1d feature extractor → [512, T/320]
//!   feature_projection: LayerNorm + Linear(512→768)
//!   pos_conv_embed (Conv1d k=128 groups=16, weight_norm dim=2) + GELU + crop-1
//!   encoder.layer_norm
//!   12 × post-norm encoder layer:
//!     x ← LN(x + MHSA(x))
//!     x ← finalLN(x + FFN(x))   (FFN = 768→3072 GELU 3072→768)
//!   lm_head: Linear(768 → 32 vocab)  → CTC greedy decode
//!
//! Pre-fused: feature_extractor convs have no bias (`conv_bias=false`);
//! only conv layer 0 carries the GroupNorm (`feat_extract_norm="group"`).

use rce_weight_loader::SafetensorsFile;
use std::path::Path;
use std::sync::Arc;

pub const SR: u32 = 16000;
pub const HIDDEN: usize = 768;
pub const FF: usize = 3072;
pub const N_HEADS: usize = 12;
pub const HEAD_DIM: usize = HIDDEN / N_HEADS;
pub const N_LAYERS: usize = 12;
pub const VOCAB: usize = 32;
pub const CONV_DIM: usize = 512;
pub const CONV_KERNELS: [usize; 7] = [10, 3, 3, 3, 3, 2, 2];
pub const CONV_STRIDES: [usize; 7] = [5, 2, 2, 2, 2, 2, 2];
pub const POS_CONV_K: usize = 128;
pub const POS_CONV_GROUPS: usize = 16;
pub const LAYER_NORM_EPS: f32 = 1e-5;

pub const VOCAB_CHARS: [&str; 32] = [
    "<pad>", "<s>", "</s>", "<unk>", "|",
    "E","T","A","O","N","I","H","S","R","D","L","U","M","W","C",
    "F","G","Y","P","B","V","K","'","X","J","Q","Z",
];

#[derive(Debug)]
pub enum WavError { Load(String) }
impl std::fmt::Display for WavError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::Load(m) => write!(f, "wav2vec2 load: {m}") }
    }
}
impl std::error::Error for WavError {}

fn load(sf: &SafetensorsFile, name: &str) -> Result<rce_weight_loader::F32Slice, WavError> {
    let t = sf.get(name).ok_or_else(|| WavError::Load(format!("missing {name}")))?;
    rce_weight_loader::F32Slice::from_tensor(name, t).map_err(WavError::Load)
}
fn load_shape(sf: &SafetensorsFile, name: &str) -> Result<(rce_weight_loader::F32Slice, Vec<usize>), WavError> {
    let t = sf.get(name).ok_or_else(|| WavError::Load(format!("missing {name}")))?;
    let shape = t.shape.clone();
    Ok((rce_weight_loader::F32Slice::from_tensor(name, t).map_err(WavError::Load)?, shape))
}

fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}
fn gelu(x: f32) -> f32 { 0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2)) }
fn gelu_inplace(x: &mut [f32]) { for v in x.iter_mut() { *v = gelu(*v); } }

fn layer_norm(x: &[f32], hidden: usize, t: usize, gamma: &[f32], beta: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for ti in 0..t {
        let row = &x[ti*hidden..(ti+1)*hidden];
        let mean: f32 = row.iter().sum::<f32>() / hidden as f32;
        let var: f32 = row.iter().map(|v| { let d = v - mean; d*d }).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (var + LAYER_NORM_EPS).sqrt();
        for h in 0..hidden {
            out[ti*hidden + h] = (row[h] - mean) * inv * gamma[h] + beta[h];
        }
    }
    out
}

// GroupNorm with G = C (per-channel affine; normalize each channel across T).
fn channel_norm(x: &mut [f32], c: usize, t: usize, gamma: &[f32], beta: &[f32]) {
    for ci in 0..c {
        let row: Vec<f32> = (0..t).map(|ti| x[ci*t + ti]).collect();
        let mean: f32 = row.iter().sum::<f32>() / t as f32;
        let var: f32 = row.iter().map(|v| { let d = v - mean; d*d }).sum::<f32>() / t as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for ti in 0..t {
            x[ci*t + ti] = (x[ci*t + ti] - mean) * inv * gamma[ci] + beta[ci];
        }
    }
}

// Conv1d on channel-major [Cin, T]. w: [Cout, Cin/g, K]. Parallel over output channels.
fn conv1d_no_bias(x: &[f32], cin: usize, t: usize,
                  w: &[f32], cout: usize, k: usize,
                  groups: usize, stride: usize, dilation: usize, pad: usize) -> (Vec<f32>, usize) {
    use rayon::prelude::*;
    let kspan = dilation*(k-1) + 1;
    let t_out = if t + 2*pad < kspan { 0 } else { (t + 2*pad - kspan) / stride + 1 };
    let cin_g = cin / groups;
    let cout_g = cout / groups;
    let mut out = vec![0.0f32; cout * t_out];
    // Each output channel's row is independent → parallelize by channel.
    out.par_chunks_mut(t_out).enumerate().for_each(|(oc_g, row)| {
        let g_ = oc_g / cout_g;
        for to in 0..t_out {
            let t_start = to * stride;
            let mut acc = 0.0f32;
            for ic in 0..cin_g {
                let ic_g = g_ * cin_g + ic;
                let w_base = (oc_g * cin_g + ic) * k;
                for kk in 0..k {
                    let tp = t_start + kk * dilation;
                    if tp < pad { continue; }
                    let xi = tp - pad;
                    if xi >= t { continue; }
                    acc += x[ic_g * t + xi] * w[w_base + kk];
                }
            }
            row[to] = acc;
        }
    });
    (out, t_out)
}

fn conv1d(x: &[f32], cin: usize, t: usize,
          w: &[f32], cout: usize, k: usize,
          groups: usize, stride: usize, dilation: usize, pad: usize,
          bias: &[f32]) -> (Vec<f32>, usize) {
    let (mut out, t_out) = conv1d_no_bias(x, cin, t, w, cout, k, groups, stride, dilation, pad);
    for oc in 0..cout {
        for ti in 0..t_out { out[oc*t_out + ti] += bias[oc]; }
    }
    (out, t_out)
}

// Linear: x [T, in] @ w [out, in] + b → [T, out].
// Delegates to rce-op-conv2d (Apple Accelerate sgemm on macOS).
fn linear(x: &[f32], t: usize, din: usize, w: &[f32], dout: usize, b: Option<&[f32]>) -> Vec<f32> {
    rce_op_conv2d::linear_f32(x, t, din, w, dout, b)
}

// weight_norm dim=2 (kernel): keep K, normalize over (Cout × Cin/g).
fn wn_resolve_dim2(g: &[f32], v: &[f32], cout: usize, cin_g: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; v.len()];
    for kk in 0..k {
        let mut n2 = 0.0f32;
        for oc in 0..cout { for ic in 0..cin_g {
            let val = v[(oc * cin_g + ic) * k + kk];
            n2 += val * val;
        }}
        let s = if n2 > 1e-30 { g[kk] / n2.sqrt() } else { 0.0 };
        for oc in 0..cout { for ic in 0..cin_g {
            let idx = (oc * cin_g + ic) * k + kk;
            out[idx] = v[idx] * s;
        }}
    }
    out
}

use rce_weight_loader::F32Slice;

struct EncoderLayer {
    q_w: F32Slice, q_b: F32Slice,
    k_w: F32Slice, k_b: F32Slice,
    v_w: F32Slice, v_b: F32Slice,
    o_w: F32Slice, o_b: F32Slice,
    ln1_w: F32Slice, ln1_b: F32Slice,
    ff_in_w: F32Slice, ff_in_b: F32Slice,
    ff_out_w: F32Slice, ff_out_b: F32Slice,
    ln2_w: F32Slice, ln2_b: F32Slice,
}
impl EncoderLayer {
    fn load(sf: &SafetensorsFile, idx: usize) -> Result<Self, WavError> {
        let p = format!("wav2vec2.encoder.layers.{idx}");
        Ok(Self {
            q_w: load(sf, &format!("{p}.attention.q_proj.weight"))?,
            q_b: load(sf, &format!("{p}.attention.q_proj.bias"))?,
            k_w: load(sf, &format!("{p}.attention.k_proj.weight"))?,
            k_b: load(sf, &format!("{p}.attention.k_proj.bias"))?,
            v_w: load(sf, &format!("{p}.attention.v_proj.weight"))?,
            v_b: load(sf, &format!("{p}.attention.v_proj.bias"))?,
            o_w: load(sf, &format!("{p}.attention.out_proj.weight"))?,
            o_b: load(sf, &format!("{p}.attention.out_proj.bias"))?,
            ln1_w: load(sf, &format!("{p}.layer_norm.weight"))?,
            ln1_b: load(sf, &format!("{p}.layer_norm.bias"))?,
            ff_in_w:  load(sf, &format!("{p}.feed_forward.intermediate_dense.weight"))?,
            ff_in_b:  load(sf, &format!("{p}.feed_forward.intermediate_dense.bias"))?,
            ff_out_w: load(sf, &format!("{p}.feed_forward.output_dense.weight"))?,
            ff_out_b: load(sf, &format!("{p}.feed_forward.output_dense.bias"))?,
            ln2_w: load(sf, &format!("{p}.final_layer_norm.weight"))?,
            ln2_b: load(sf, &format!("{p}.final_layer_norm.bias"))?,
        })
    }
    fn forward(&self, x: &mut Vec<f32>, t: usize) {
        let q = linear(x, t, HIDDEN, &self.q_w, HIDDEN, Some(&self.q_b));
        let k = linear(x, t, HIDDEN, &self.k_w, HIDDEN, Some(&self.k_b));
        let v = linear(x, t, HIDDEN, &self.v_w, HIDDEN, Some(&self.v_b));
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        // Parallel over (head, query-time): each [h, ti] slot is independent.
        // Write into a per-head buffer then stripe back into [T, HIDDEN].
        use rayon::prelude::*;
        let mut per_head: Vec<f32> = vec![0.0f32; N_HEADS * t * HEAD_DIM];
        per_head.par_chunks_mut(t * HEAD_DIM).enumerate().for_each(|(h, head_buf)| {
            let q_off = h * HEAD_DIM;
            for ti in 0..t {
                let mut scores = vec![0.0f32; t];
                let mut maxv = f32::NEG_INFINITY;
                for tj in 0..t {
                    let mut s = 0.0f32;
                    for d in 0..HEAD_DIM {
                        s += q[ti*HIDDEN + q_off + d] * k[tj*HIDDEN + q_off + d];
                    }
                    s *= scale;
                    scores[tj] = s;
                    if s > maxv { maxv = s; }
                }
                let mut sum = 0.0f32;
                for tj in 0..t { scores[tj] = (scores[tj] - maxv).exp(); sum += scores[tj]; }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for tj in 0..t { scores[tj] *= inv; }
                for d in 0..HEAD_DIM {
                    let mut acc = 0.0f32;
                    for tj in 0..t { acc += scores[tj] * v[tj*HIDDEN + q_off + d]; }
                    head_buf[ti * HEAD_DIM + d] = acc;
                }
            }
        });
        // Restripe per_head [H, T, D] → attn_out [T, H, D].
        let mut attn_out = vec![0.0f32; t * HIDDEN];
        for h in 0..N_HEADS {
            for ti in 0..t {
                for d in 0..HEAD_DIM {
                    attn_out[ti*HIDDEN + h*HEAD_DIM + d] = per_head[h*t*HEAD_DIM + ti*HEAD_DIM + d];
                }
            }
        }
        let attn_proj = linear(&attn_out, t, HIDDEN, &self.o_w, HIDDEN, Some(&self.o_b));
        for i in 0..x.len() { x[i] += attn_proj[i]; }
        *x = layer_norm(x, HIDDEN, t, &self.ln1_w, &self.ln1_b);

        let mut ff = linear(x, t, HIDDEN, &self.ff_in_w, FF, Some(&self.ff_in_b));
        gelu_inplace(&mut ff);
        let ff_out = linear(&ff, t, FF, &self.ff_out_w, HIDDEN, Some(&self.ff_out_b));
        for i in 0..x.len() { x[i] += ff_out[i]; }
        *x = layer_norm(x, HIDDEN, t, &self.ln2_w, &self.ln2_b);
    }
}

pub struct Wav2Vec2 {
    fe_w: Vec<F32Slice>,                                       // 7 conv weights, mmap-aliased
    fe0_norm_w: F32Slice, fe0_norm_b: F32Slice,
    fp_ln_w: F32Slice, fp_ln_b: F32Slice,
    fp_w: F32Slice, fp_b: F32Slice,
    pos_w: Vec<f32>, pos_b: F32Slice,                          // pos_w is wn-resolved (computed; stays owned)
    enc_ln_w: F32Slice, enc_ln_b: F32Slice,
    layers: Vec<EncoderLayer>,
    lm_w: F32Slice, lm_b: F32Slice,
}

impl Wav2Vec2 {
    pub fn load(path: &Path) -> Result<Arc<Self>, WavError> {
        let sf = SafetensorsFile::load(path).map_err(|e| WavError::Load(e.to_string()))?;

        let mut fe_w: Vec<F32Slice> = Vec::with_capacity(7);
        for i in 0..7 {
            fe_w.push(load(&sf, &format!("wav2vec2.feature_extractor.conv_layers.{i}.conv.weight"))?);
        }
        let fe0_norm_w = load(&sf, "wav2vec2.feature_extractor.conv_layers.0.layer_norm.weight")?;
        let fe0_norm_b = load(&sf, "wav2vec2.feature_extractor.conv_layers.0.layer_norm.bias")?;

        let fp_ln_w = load(&sf, "wav2vec2.feature_projection.layer_norm.weight")?;
        let fp_ln_b = load(&sf, "wav2vec2.feature_projection.layer_norm.bias")?;
        let fp_w    = load(&sf, "wav2vec2.feature_projection.projection.weight")?;
        let fp_b    = load(&sf, "wav2vec2.feature_projection.projection.bias")?;

        let (g, gsh) = load_shape(&sf, "wav2vec2.encoder.pos_conv_embed.conv.weight_g")?;
        let (v, vsh) = load_shape(&sf, "wav2vec2.encoder.pos_conv_embed.conv.weight_v")?;
        let pos_b = load(&sf, "wav2vec2.encoder.pos_conv_embed.conv.bias")?;
        if gsh != vec![1, 1, POS_CONV_K] {
            return Err(WavError::Load(format!("pos_conv g shape {:?} != [1,1,128]", gsh)));
        }
        let cin_per_g = HIDDEN / POS_CONV_GROUPS;
        if vsh != vec![HIDDEN, cin_per_g, POS_CONV_K] {
            return Err(WavError::Load(format!("pos_conv v shape {:?} != [{HIDDEN},{cin_per_g},{POS_CONV_K}]", vsh)));
        }
        let pos_w = wn_resolve_dim2(&g, &v, HIDDEN, cin_per_g, POS_CONV_K);

        let enc_ln_w = load(&sf, "wav2vec2.encoder.layer_norm.weight")?;
        let enc_ln_b = load(&sf, "wav2vec2.encoder.layer_norm.bias")?;

        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS { layers.push(EncoderLayer::load(&sf, i)?); }

        let lm_w = load(&sf, "lm_head.weight")?;
        let lm_b = load(&sf, "lm_head.bias")?;

        Ok(Arc::new(Self {
            fe_w, fe0_norm_w, fe0_norm_b,
            fp_ln_w, fp_ln_b, fp_w, fp_b,
            pos_w, pos_b, enc_ln_w, enc_ln_b,
            layers, lm_w, lm_b,
        }))
    }

    fn feature_extract(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        let (mut x, mut t) = conv1d_no_bias(audio, 1, audio.len(),
                                            &self.fe_w[0], CONV_DIM, CONV_KERNELS[0],
                                            1, CONV_STRIDES[0], 1, 0);
        channel_norm(&mut x, CONV_DIM, t, &self.fe0_norm_w, &self.fe0_norm_b);
        for v in x.iter_mut() { *v = gelu(*v); }
        for i in 1..7 {
            let (y, t_out) = conv1d_no_bias(&x, CONV_DIM, t,
                                            &self.fe_w[i], CONV_DIM, CONV_KERNELS[i],
                                            1, CONV_STRIDES[i], 1, 0);
            x = y; t = t_out;
            for v in x.iter_mut() { *v = gelu(*v); }
        }
        (x, t)
    }

    /// Zero-mean unit-variance normalization (matches HF Wav2Vec2FeatureExtractor with do_normalize=True).
    pub fn normalize_audio(audio: &[f32]) -> Vec<f32> {
        let n = audio.len() as f32;
        let mean: f32 = audio.iter().sum::<f32>() / n;
        let var: f32 = audio.iter().map(|x| { let d = x - mean; d*d }).sum::<f32>() / n;
        let inv = 1.0 / (var + 1e-7).sqrt();
        audio.iter().map(|x| (x - mean) * inv).collect()
    }

    pub fn forward(&self, audio_raw: &[f32]) -> (Vec<f32>, usize) {
        let audio = Self::normalize_audio(audio_raw);
        let audio = &audio[..];
        let (feat_ct, t) = self.feature_extract(audio);
        // Transpose to [T, 512].
        let mut feat_tc = vec![0.0f32; t * CONV_DIM];
        for ti in 0..t { for c in 0..CONV_DIM {
            feat_tc[ti*CONV_DIM + c] = feat_ct[c*t + ti];
        }}
        let feat_ln = layer_norm(&feat_tc, CONV_DIM, t, &self.fp_ln_w, &self.fp_ln_b);
        let mut hidden_tc = linear(&feat_ln, t, CONV_DIM, &self.fp_w, HIDDEN, Some(&self.fp_b));

        // Pos conv embed (channel-major), then add residual.
        let mut hidden_ct = vec![0.0f32; t * HIDDEN];
        for ti in 0..t { for h in 0..HIDDEN {
            hidden_ct[h*t + ti] = hidden_tc[ti*HIDDEN + h];
        }}
        let pad = POS_CONV_K / 2;
        let (pos_ct_full, t_full) = conv1d(&hidden_ct, HIDDEN, t,
                                           &self.pos_w, HIDDEN, POS_CONV_K,
                                           POS_CONV_GROUPS, 1, 1, pad, &self.pos_b);
        // Even-kernel "same"-pad crop: drop last 1 frame.  Verify length.
        debug_assert!(t_full == t + 1);
        for ti in 0..t { for h in 0..HIDDEN {
            let val = gelu(pos_ct_full[h*t_full + ti]);
            hidden_tc[ti*HIDDEN + h] += val;
        }}

        let mut hidden = layer_norm(&hidden_tc, HIDDEN, t, &self.enc_ln_w, &self.enc_ln_b);
        for layer in &self.layers { layer.forward(&mut hidden, t); }

        let logits = linear(&hidden, t, HIDDEN, &self.lm_w, VOCAB, Some(&self.lm_b));
        (logits, t)
    }

    pub fn ctc_greedy_decode(&self, logits: &[f32], t: usize) -> String {
        let mut out = String::new();
        let mut prev = usize::MAX;
        for ti in 0..t {
            let row = &logits[ti*VOCAB..(ti+1)*VOCAB];
            let mut best = 0; let mut bv = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() { if v > bv { bv = v; best = i; } }
            if best != prev {
                if best != 0 {
                    let c = VOCAB_CHARS[best];
                    if c == "|" { out.push(' '); } else { out.push_str(c); }
                }
                prev = best;
            }
        }
        out
    }

    /// Convenience: read a PCM-16 mono WAV file, optionally resample to 16 kHz, and run
    /// `forward` + `ctc_greedy_decode`. Errors propagate from I/O / parsing.
    pub fn transcribe_wav(&self, path: &Path) -> Result<String, WavError> {
        let audio = read_wav_mono_16k(path)?;
        let (logits, t) = self.forward(&audio);
        Ok(self.ctc_greedy_decode(&logits, t))
    }
}

/// Read a PCM-16 mono RIFF WAV file, resample to 16 kHz (if needed) via a
/// short polyphase filter, and return [-1, 1] f32 samples.
///
/// Supports the common case: `RIFF/WAVE` container, format code 1 (PCM), mono,
/// 16-bit. Anything else returns an error.
pub fn read_wav_mono_16k(path: &Path) -> Result<Vec<f32>, WavError> {
    let bytes = std::fs::read(path).map_err(|e| WavError::Load(format!("read {}: {e}", path.display())))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(WavError::Load("not a RIFF/WAVE file".into()));
    }
    // Scan chunks for `fmt ` and `data`.
    let mut sr: u32 = 0;
    let mut ch: u16 = 0;
    let mut bits: u16 = 0;
    let mut fmt: u16 = 0;
    let mut data_off: usize = 0;
    let mut data_len: usize = 0;
    let mut pos = 12usize;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos+4];
        let sz = u32::from_le_bytes(bytes[pos+4..pos+8].try_into().unwrap()) as usize;
        let payload = pos + 8;
        match id {
            b"fmt " => {
                fmt = u16::from_le_bytes(bytes[payload..payload+2].try_into().unwrap());
                ch  = u16::from_le_bytes(bytes[payload+2..payload+4].try_into().unwrap());
                sr  = u32::from_le_bytes(bytes[payload+4..payload+8].try_into().unwrap());
                bits = u16::from_le_bytes(bytes[payload+14..payload+16].try_into().unwrap());
            }
            b"data" => { data_off = payload; data_len = sz; break; }
            _ => {}
        }
        pos = payload + sz + (sz & 1);   // chunks are 2-byte aligned
    }
    if fmt != 1 { return Err(WavError::Load(format!("unsupported fmt code {fmt}; need PCM (1)"))); }
    if ch != 1 { return Err(WavError::Load(format!("unsupported channel count {ch}; need mono"))); }
    if bits != 16 { return Err(WavError::Load(format!("unsupported bit depth {bits}; need 16"))); }
    if data_off == 0 { return Err(WavError::Load("no data chunk".into())); }

    let n_samples = data_len / 2;
    let mut pcm = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let v = i16::from_le_bytes(bytes[data_off+2*i..data_off+2*i+2].try_into().unwrap());
        pcm.push(v as f32 / 32768.0);
    }
    if sr == 16000 { return Ok(pcm); }
    Ok(resample_to_16k(&pcm, sr))
}

/// Linear-phase polyphase resample from arbitrary sample rate to 16 kHz.
/// Uses a 32-tap windowed-sinc kernel — plenty for speech-band ASR.
fn resample_to_16k(input: &[f32], sr_in: u32) -> Vec<f32> {
    // Simple ratio rational approximation via gcd; supports common SRs (8k, 22050, 44100, 48000, etc.).
    let g = gcd(sr_in, 16000);
    let up = 16000 / g;
    let dn = sr_in / g;
    polyphase_resample(input, up as usize, dn as usize)
}

fn gcd(a: u32, b: u32) -> u32 { if b == 0 { a } else { gcd(b, a % b) } }

fn polyphase_resample(x: &[f32], up: usize, dn: usize) -> Vec<f32> {
    // Build a 64-tap (per-phase) Hann-windowed sinc with cutoff at min(up, dn) / (2 * up).
    const TAPS: usize = 64;            // per-phase
    let cutoff = 1.0_f32 / (up.max(dn) as f32);  // normalized to upsampled rate
    let kernel_len = TAPS * up;
    let mut k = vec![0.0f32; kernel_len];
    let center = (kernel_len as f32 - 1.0) / 2.0;
    for i in 0..kernel_len {
        let t = i as f32 - center;
        let s = if t == 0.0 { 1.0 } else {
            let pt = std::f32::consts::PI * t * cutoff;
            pt.sin() / pt
        };
        let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (kernel_len as f32 - 1.0)).cos();
        k[i] = s * w;
    }
    // Normalize per-phase gain.
    let g: f32 = k.iter().sum::<f32>() / up as f32;
    for v in k.iter_mut() { *v /= g; }

    // Output length: ceil(len(x) * up / dn).
    let n_out = (x.len() * up + dn - 1) / dn;
    let mut y = vec![0.0f32; n_out];
    // For each output index n, the upsampled sample index is m = n * dn.
    // Kernel maps from upsampled index → input index = (m - tap) / up (integer).
    for n in 0..n_out {
        let m = n * dn;
        let mut acc = 0.0f32;
        for tap in 0..kernel_len {
            // upsampled[m - tap] is nonzero only when (m - tap) divisible by up,
            // i.e., the input sample at index (m - tap) / up.
            if tap > m { break; }
            let up_idx = m - tap;
            if up_idx % up != 0 { continue; }
            let xi = up_idx / up;
            if xi < x.len() { acc += k[tap] * x[xi]; }
        }
        y[n] = acc;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn model_path() -> Option<std::path::PathBuf> {
        let p = std::path::PathBuf::from("/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/wav2vec2-base-960h/model.safetensors");
        if p.exists() { Some(p) } else { None }
    }

    fn load_fixture() -> Option<Vec<f32>> {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/speech_16k.f32");
        let bytes = std::fs::read(&p).ok()?;
        Some(bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]]))
            .collect())
    }

    fn load_real_speech_fixture() -> Option<Vec<f32>> {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/female_speech_16k_8s.f32");
        let bytes = std::fs::read(&p).ok()?;
        Some(bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]]))
            .collect())
    }

    /// Bit-identical CTC match to HF reference on a short TTS fixture.
    /// The model produces deterministic gibberish on TTS audio (it's trained on
    /// LibriSpeech read speech), so we lock the exact output as the golden.
    #[test]
    fn wav2vec2_decode_tts_fixture() {
        let Some(p) = model_path() else { eprintln!("skip: model not present"); return; };
        let Some(audio) = load_fixture() else { eprintln!("skip: fixture missing"); return; };
        let t0 = Instant::now();
        let m = Wav2Vec2::load(&p).expect("load");
        eprintln!("loaded in {:?}", t0.elapsed());
        eprintln!("audio: {} samples ({:.2}s)", audio.len(), audio.len() as f32 / 16000.0);

        let t0 = Instant::now();
        let (logits, t) = m.forward(&audio);
        let dt = t0.elapsed();
        eprintln!("forward: {t} frames in {dt:?} ({:.1}× realtime)",
                  audio.len() as f32 / 16000.0 / dt.as_secs_f32());

        let text = m.ctc_greedy_decode(&logits, t);
        eprintln!("transcript: {text:?}");
        // Golden from `transformers.Wav2Vec2ForCTC` on the same fixture.
        assert_eq!(text, " VR MAN RE MATY YAN U ");
    }

    /// Real-speech (Harvard Sentences) golden — semantically meaningful transcript,
    /// also bit-identical to the HF reference.
    #[test]
    fn wav2vec2_decode_real_speech() {
        let Some(p) = model_path() else { eprintln!("skip: model not present"); return; };
        let Some(audio) = load_real_speech_fixture() else { eprintln!("skip: real-speech fixture missing"); return; };
        let m = Wav2Vec2::load(&p).expect("load");
        eprintln!("audio: {} samples ({:.2}s)", audio.len(), audio.len() as f32 / 16000.0);

        let t0 = Instant::now();
        let (logits, t) = m.forward(&audio);
        let dt = t0.elapsed();
        eprintln!("forward: {t} frames in {dt:?} ({:.1}× realtime)",
                  audio.len() as f32 / 16000.0 / dt.as_secs_f32());

        let text = m.ctc_greedy_decode(&logits, t);
        eprintln!("transcript: {text:?}");
        // HF golden — Harvard Sentences read speech, 8 s clip resampled from 8 kHz.
        // Note: the model is uppercase-only by training; the slight word-glue around
        // "PLANK GLE" is the model's own behavior, not a bug in our impl.
        let expected = "THE BIRCH CANOE SLID ON THE SMOOTH PLANK GLE THE SHEET TO THE DARK BLUE BACKGROUND IT";
        assert_eq!(text.trim(), expected);
    }

    /// Long-audio: process the full 33 s clip in one shot. Verifies that the O(T²)
    /// attention scales without blowing up at T ≈ 1675 frames.
    #[test]
    #[ignore = "33-second forward (~30s wall); run with --ignored"]
    fn wav2vec2_long_audio() {
        let Some(p) = model_path() else { eprintln!("skip: model not present"); return; };
        let wav = std::path::PathBuf::from(
            "/Users/dcharlot/data-share/vibe-coding/new-ai/prometheus/crates/voice/reference_speech/female_sample.wav");
        if !wav.exists() { eprintln!("skip: source WAV missing"); return; }
        let m = Wav2Vec2::load(&p).expect("load");
        let audio = read_wav_mono_16k(&wav).expect("read");
        eprintln!("long-audio: {} samples ({:.2}s)", audio.len(), audio.len() as f32 / 16000.0);
        let t0 = Instant::now();
        let (logits, t) = m.forward(&audio);
        let dt = t0.elapsed();
        eprintln!("forward: {t} frames in {dt:?} ({:.2}× realtime)",
                  audio.len() as f32 / 16000.0 / dt.as_secs_f32());
        let text = m.ctc_greedy_decode(&logits, t);
        eprintln!("transcript ({} chars): {text:?}", text.len());
        // Sanity: full Harvard #1 stems should all be present (rainbow passage).
        for stem in ["BIRCH", "CANOE", "PLANK", "DARK", "BLUE"] {
            assert!(text.contains(stem), "missing {stem:?}");
        }
        assert!(text.len() > 80, "long-audio transcript suspiciously short: {text:?}");
    }

    /// End-to-end public API: read an unprocessed WAV (any SR, mono, PCM-16) → transcript.
    #[test]
    fn wav2vec2_transcribe_wav_api() {
        let Some(p) = model_path() else { eprintln!("skip: model not present"); return; };
        let wav = std::path::PathBuf::from(
            "/Users/dcharlot/data-share/vibe-coding/new-ai/prometheus/crates/voice/reference_speech/female_sample.wav");
        if !wav.exists() { eprintln!("skip: source WAV missing"); return; }

        let m = Wav2Vec2::load(&p).expect("load");
        // Read the full 33s clip via the public API (which does its own resample).
        let audio = read_wav_mono_16k(&wav).expect("read wav");
        eprintln!("loaded {} samples ({:.2}s) via internal WAV reader",
                  audio.len(), audio.len() as f32 / 16000.0);
        assert!(audio.len() > 16000, "expected >1s of 16 kHz audio");

        // Run on first 8s slice and verify the transcript carries the same content
        // words as the scipy-resampled golden. Our internal polyphase resampler is
        // shorter than scipy's default (64 taps vs ~221), so single-character
        // disagreements are expected — the SEMANTIC content must match.
        let slice = &audio[..(8 * 16000).min(audio.len())];
        let (logits, t) = m.forward(slice);
        let text = m.ctc_greedy_decode(&logits, t);
        eprintln!("transcribe_wav: {text:?}");
        for word in ["BIRCH", "CANOE", "SLID", "SMOOTH", "PLANK", "DARK", "BLUE", "BACKGROUND"] {
            assert!(text.contains(word),
                    "WAV-API transcript missing key word {word:?}: {text:?}");
        }
    }
}
