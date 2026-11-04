//! WavTokenizer 24 kHz — single-codebook neural speech codec.
//!
//! `wavtokenizer_large_speech_320_v2`.  Pipeline:
//!
//! ```text
//! audio 24 kHz
//!   → encoder (EnCodec-style, strides [2,4,5,8], channels 32→64→128→256→512,
//!              2-layer LSTM @ 512, ResidualUnits w/ shortcut conv)
//!   → 1-stage VQ (codebook [4096, 512])
//!   → latent [512, T]                 (T = audio_len / 320, e.g. 75 Hz @ 24 kHz)
//!   → backbone.embed (Conv1d 512→768 k=7)
//!   → AdaLN outer (bandwidth-conditioned scale+shift; we pick slot 0)
//!   → 12 ConvNeXt-1D blocks  + 6 pos_net blocks (mixed: 5 conv-residual + 1 self-attn)
//!   → final_layer_norm
//!   → head.out (Linear 768→1282 → split 641 magnitude + 641 phase)
//!   → ISTFT (n_fft=1280, hop=320, Hann window, one-sided spec)
//! ```
//!
//! Weight convention: PyTorch `weight_norm` (legacy `weight_g [Cout,1,1]` +
//! `weight_v [Cout,Cin,K]`) on encoder convs.  Backbone uses plain `.weight` /
//! `.bias` (already fused, no weight_norm).

use rce_op_lstm::{lstm_forward, LstmConfig, LstmLayer};
use rce_weight_loader::{LoadedTensor, SafetensorsFile};
use std::path::Path;
use std::sync::Arc;

pub const SR: u32 = 24_000;
pub const N_FFT: usize = 1280;
pub const HOP: usize = 320;
pub const N_BIN: usize = N_FFT / 2 + 1;          // 641
pub const HEAD_OUT_DIM: usize = 2 * N_BIN;       // 1282 (mag + phase)
pub const BACKBONE_HIDDEN: usize = 768;
pub const FF: usize = 2304;
pub const N_CONVNEXT: usize = 12;
pub const N_POS_NET: usize = 6;
pub const HIDDEN_LATENT: usize = 512;
pub const LSTM_HIDDEN: usize = 512;
pub const CODEBOOK_SIZE: usize = 4096;
pub const STRIDES: [usize; 4] = [2, 4, 5, 8];
pub const KERNELS: [usize; 4] = [4, 8, 10, 16];
pub const CHANNELS: [usize; 5] = [32, 64, 128, 256, 512];
pub const COMPRESS: usize = 2;
pub const RES_KERNEL: usize = 3;
pub const INIT_KERNEL: usize = 7;
pub const FINAL_KERNEL: usize = 7;
pub const TOTAL_DOWNSAMPLE: usize = 2 * 4 * 5 * 8;   // 320 (75 Hz @ 24 kHz)
const LN_EPS: f32 = 1e-6;
const BW_SLOT: usize = 0;   // single-bandwidth config

#[derive(Debug)]
pub enum WtError { Load(String) }
impl std::fmt::Display for WtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::Load(m) => write!(f, "wavtokenizer load: {m}") }
    }
}
impl std::error::Error for WtError {}

fn tf32(t: &LoadedTensor) -> Vec<f32> {
    let bytes = t.data.as_slice();
    match t.dtype.as_str() {
        "f32"  => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect(),
        "f16"  => bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0],c[1]]).to_f32()).collect(),
        "bf16" => bytes.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0],c[1]]).to_f32()).collect(),
        other  => panic!("wavtokenizer: unsupported dtype {other}"),
    }
}
fn g(sf: &SafetensorsFile, name: &str) -> Result<Vec<f32>, WtError> {
    let t = sf.get(name).ok_or_else(|| WtError::Load(format!("missing {name}")))?;
    Ok(tf32(t))
}
fn g_opt(sf: &SafetensorsFile, name: &str) -> Option<Vec<f32>> {
    sf.get(name).map(tf32)
}

// ── ops ─────────────────────────────────────────────────────────────────────

fn elu(x: f32) -> f32 { if x >= 0.0 { x } else { x.exp() - 1.0 } }
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}
fn gelu(x: f32) -> f32 { 0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2)) }

fn wn_dim0(g: &[f32], v: &[f32], shape: &[usize]) -> Vec<f32> {
    let keep = shape[0];
    let inner: usize = shape[1..].iter().product();
    let mut out = vec![0.0f32; v.len()];
    for k in 0..keep {
        let base = k * inner;
        let n2: f32 = v[base..base+inner].iter().map(|x| x*x).sum();
        let s = if n2 > 1e-30 { g[k] / n2.sqrt() } else { 0.0 };
        for i in 0..inner { out[base+i] = v[base+i] * s; }
    }
    out
}

fn pad_reflect_1d(x: &[f32], c: usize, t: usize, k: usize, dilation: usize, stride: usize) -> (Vec<f32>, usize) {
    let total_pad = (k - 1) * dilation;
    let ideal_len = ((t as f32 / stride as f32).ceil() as usize) * stride;
    let extra_padding = ideal_len.saturating_sub(t);
    let right = total_pad / 2 + extra_padding;
    let left = total_pad - total_pad / 2;
    let new_t = t + left + right;
    let mut out = vec![0.0f32; c * new_t];
    for ci in 0..c {
        for i in 0..new_t {
            let j = i as i64 - left as i64;
            let idx = if j < 0 { (-j) as usize } else if (j as usize) >= t {
                let jj = 2 * t as i64 - 2 - j;
                if jj < 0 { 0 } else { jj as usize }
            } else { j as usize };
            out[ci * new_t + i] = x[ci * t + idx.min(t - 1)];
        }
    }
    (out, new_t)
}

fn conv1d(x: &[f32], cin: usize, t: usize,
          w: &[f32], cout: usize, k: usize,
          stride: usize, dilation: usize,
          bias: &[f32]) -> (Vec<f32>, usize) {
    let kspan = dilation*(k-1) + 1;
    let t_out = if t < kspan { 0 } else { (t - kspan) / stride + 1 };
    let mut out = vec![0.0f32; cout * t_out];
    for to in 0..t_out {
        let t_start = to * stride;
        for oc in 0..cout {
            let mut acc = bias[oc];
            for ic in 0..cin {
                for kk in 0..k {
                    let tp = t_start + kk * dilation;
                    if tp >= t { break; }
                    acc += x[ic * t + tp] * w[(oc * cin + ic) * k + kk];
                }
            }
            out[oc * t_out + to] = acc;
        }
    }
    (out, t_out)
}

fn conv1d_reflect(x: &[f32], cin: usize, t: usize,
                  w: &[f32], cout: usize, k: usize,
                  stride: usize, dilation: usize,
                  bias: &[f32]) -> (Vec<f32>, usize) {
    let (padded, pt) = pad_reflect_1d(x, cin, t, k, dilation, stride);
    conv1d(&padded, cin, pt, w, cout, k, stride, dilation, bias)
}

// LayerNorm on channel-major [C, T] — normalize across C per time step.
fn layer_norm_ct(x: &[f32], c: usize, t: usize, gamma: &[f32], beta: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for ti in 0..t {
        let mut mean = 0.0f32;
        for ci in 0..c { mean += x[ci*t + ti]; }
        mean /= c as f32;
        let mut var = 0.0f32;
        for ci in 0..c { let d = x[ci*t + ti] - mean; var += d*d; }
        var /= c as f32;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        for ci in 0..c {
            out[ci*t + ti] = (x[ci*t + ti] - mean) * inv * gamma[ci] + beta[ci];
        }
    }
    out
}

/// Linear with batched output across time: y[ti, o] = Σ_i x[ti, i] · w[o, i] + b[o].
/// Input/output in `[T, C]` layout.
fn linear(x: &[f32], t: usize, din: usize, w: &[f32], dout: usize, b: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; t * dout];
    for ti in 0..t {
        for o in 0..dout {
            let mut acc = b[o];
            let row = &x[ti*din..(ti+1)*din];
            let wr  = &w[o*din..(o+1)*din];
            for i in 0..din { acc += row[i] * wr[i]; }
            y[ti*dout + o] = acc;
        }
    }
    y
}

// ── EnCodec-style encoder pieces (with shortcut conv) ──────────────────────

struct ConvW { w: Vec<f32>, b: Vec<f32>, k: usize, stride: usize }
impl ConvW {
    fn load(sf: &SafetensorsFile, prefix: &str, k: usize, stride: usize, cout: usize, cin: usize)
        -> Result<Self, WtError>
    {
        let g_raw = g(sf, &format!("{prefix}.conv.conv.weight_g"))?;
        let v_raw = g(sf, &format!("{prefix}.conv.conv.weight_v"))?;
        let b = g(sf, &format!("{prefix}.conv.conv.bias"))?;
        let shape = vec![cout, cin, k];
        Ok(Self { w: wn_dim0(&g_raw, &v_raw, &shape), b, k, stride })
    }
}

struct ResidualUnit { c: usize, w_compress: Vec<f32>, b_compress: Vec<f32>,
                      w_expand: Vec<f32>, b_expand: Vec<f32>,
                      shortcut: Option<(Vec<f32>, Vec<f32>)> }
impl ResidualUnit {
    fn load(sf: &SafetensorsFile, prefix: &str, c: usize) -> Result<Self, WtError> {
        let cmid = c / COMPRESS;
        let g_raw = g(sf, &format!("{prefix}.block.1.conv.conv.weight_g"))?;
        let v_raw = g(sf, &format!("{prefix}.block.1.conv.conv.weight_v"))?;
        let b_compress = g(sf, &format!("{prefix}.block.1.conv.conv.bias"))?;
        let w_compress = wn_dim0(&g_raw, &v_raw, &[cmid, c, RES_KERNEL]);

        let g_raw = g(sf, &format!("{prefix}.block.3.conv.conv.weight_g"))?;
        let v_raw = g(sf, &format!("{prefix}.block.3.conv.conv.weight_v"))?;
        let b_expand = g(sf, &format!("{prefix}.block.3.conv.conv.bias"))?;
        let w_expand = wn_dim0(&g_raw, &v_raw, &[c, cmid, 1]);

        // Optional k=1 shortcut conv (used in WavTokenizer's encoder).
        let shortcut = if let Some(g_raw) = g_opt(sf, &format!("{prefix}.shortcut.conv.conv.weight_g")) {
            let v_raw = g(sf, &format!("{prefix}.shortcut.conv.conv.weight_v"))?;
            let b_s   = g(sf, &format!("{prefix}.shortcut.conv.conv.bias"))?;
            Some((wn_dim0(&g_raw, &v_raw, &[c, c, 1]), b_s))
        } else { None };
        Ok(Self { c, w_compress, b_compress, w_expand, b_expand, shortcut })
    }
    fn forward(&self, x: &[f32], t: usize) -> (Vec<f32>, usize) {
        let cin = self.c; let cmid = self.c / COMPRESS;
        let mut x0 = x.to_vec();
        for v in x0.iter_mut() { *v = elu(*v); }
        let (y1, t1) = conv1d_reflect(&x0, cin, t, &self.w_compress, cmid, RES_KERNEL, 1, 1, &self.b_compress);
        let mut y1 = y1; for v in y1.iter_mut() { *v = elu(*v); }
        let (y2, _) = conv1d_reflect(&y1, cmid, t1, &self.w_expand, cin, 1, 1, 1, &self.b_expand);
        // Shortcut path
        let resid = match &self.shortcut {
            Some((sw, sb)) => {
                let (r, _) = conv1d_reflect(x, cin, t, sw, cin, 1, 1, 1, sb);
                r
            }
            None => x.to_vec(),
        };
        let mut out = vec![0.0f32; cin * t];
        for i in 0..(cin * t) { out[i] = resid[i] + y2[i]; }
        (out, t)
    }
}

// ── encoder ─────────────────────────────────────────────────────────────────
struct Encoder {
    pre:    ConvW,
    res:    [ResidualUnit; 4],
    down:   [ConvW; 4],
    lstm:   Vec<LstmLayer>,
    post:   ConvW,
}
impl Encoder {
    fn load(sf: &SafetensorsFile) -> Result<Self, WtError> {
        let p = "feature_extractor.encodec.encoder.model";
        let pre  = ConvW::load(sf, &format!("{p}.0"),  INIT_KERNEL,  1, CHANNELS[0], 1)?;
        let res = [
            ResidualUnit::load(sf, &format!("{p}.1"),  CHANNELS[0])?,
            ResidualUnit::load(sf, &format!("{p}.4"),  CHANNELS[1])?,
            ResidualUnit::load(sf, &format!("{p}.7"),  CHANNELS[2])?,
            ResidualUnit::load(sf, &format!("{p}.10"), CHANNELS[3])?,
        ];
        let down = [
            ConvW::load(sf, &format!("{p}.3"),  KERNELS[0], STRIDES[0], CHANNELS[1], CHANNELS[0])?,
            ConvW::load(sf, &format!("{p}.6"),  KERNELS[1], STRIDES[1], CHANNELS[2], CHANNELS[1])?,
            ConvW::load(sf, &format!("{p}.9"),  KERNELS[2], STRIDES[2], CHANNELS[3], CHANNELS[2])?,
            ConvW::load(sf, &format!("{p}.12"), KERNELS[3], STRIDES[3], CHANNELS[4], CHANNELS[3])?,
        ];
        let mut lstm = Vec::with_capacity(2);
        for li in 0..2 {
            lstm.push(LstmLayer {
                weight_ih: g(sf, &format!("{p}.13.lstm.weight_ih_l{li}"))?,
                weight_hh: g(sf, &format!("{p}.13.lstm.weight_hh_l{li}"))?,
                bias_ih:   g(sf, &format!("{p}.13.lstm.bias_ih_l{li}"))?,
                bias_hh:   g(sf, &format!("{p}.13.lstm.bias_hh_l{li}"))?,
            });
        }
        let post = ConvW::load(sf, &format!("{p}.15"), FINAL_KERNEL, 1, CHANNELS[4], CHANNELS[4])?;
        Ok(Self { pre, res, down, lstm, post })
    }
    fn forward(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        let (mut x, mut t) = conv1d_reflect(audio, 1, audio.len(),
            &self.pre.w, CHANNELS[0], self.pre.k, self.pre.stride, 1, &self.pre.b);
        for s in 0..4 {
            let (y, t1) = self.res[s].forward(&x, t);
            x = y; t = t1;
            for v in x.iter_mut() { *v = elu(*v); }
            let (y, t1) = conv1d_reflect(&x, CHANNELS[s], t,
                &self.down[s].w, CHANNELS[s+1], self.down[s].k, self.down[s].stride, 1, &self.down[s].b);
            x = y; t = t1;
        }
        // LSTM as residual
        let c = CHANNELS[4];
        let mut x_tc = vec![0.0f32; t * c];
        for ti in 0..t { for ci in 0..c { x_tc[ti*c + ci] = x[ci*t + ti]; }}
        let cfg = LstmConfig { input_size: c, hidden_size: LSTM_HIDDEN, num_layers: 2 };
        let (y_tc, _, _) = lstm_forward(&x_tc, t, &cfg, &self.lstm, None, None);
        let mut x2 = vec![0.0f32; c * t];
        for ti in 0..t { for ci in 0..c { x2[ci*t + ti] = y_tc[ti*c + ci] + x[ci*t + ti]; }}
        x = x2;
        let (y, t1) = conv1d_reflect(&x, c, t,
            &self.post.w, CHANNELS[4], self.post.k, self.post.stride, 1, &self.post.b);
        (y, t1)
    }
}

// ── single-stage quantizer ──────────────────────────────────────────────────
struct Quantizer { codebook: Vec<f32> }   // [4096 * 512]
impl Quantizer {
    fn load(sf: &SafetensorsFile) -> Result<Self, WtError> {
        let codebook = g(sf, "feature_extractor.encodec.quantizer.vq.layers.0._codebook.embed")?;
        Ok(Self { codebook })
    }
    fn encode(&self, z: &[f32], t: usize) -> (Vec<u32>, Vec<f32>) {
        let d = HIDDEN_LATENT;
        let mut codes = vec![0u32; t];
        let mut z_q = vec![0.0f32; d * t];
        for ti in 0..t {
            let mut best = 0u32; let mut best_d = f32::INFINITY;
            for ci in 0..CODEBOOK_SIZE {
                let mut s = 0.0f32;
                for k in 0..d {
                    let diff = z[k*t + ti] - self.codebook[ci*d + k];
                    s += diff * diff;
                }
                if s < best_d { best_d = s; best = ci as u32; }
            }
            codes[ti] = best;
            for k in 0..d { z_q[k*t + ti] = self.codebook[best as usize * d + k]; }
        }
        (codes, z_q)
    }
}

// ── backbone ────────────────────────────────────────────────────────────────

/// AdaLN over `[C, T]` features.  `scale` and `shift` have shape `[4, 768]`;
/// pick bandwidth slot `bw_slot` and apply per-channel.
fn adaln_ct(x: &[f32], c: usize, t: usize, scale_4c: &[f32], shift_4c: &[f32], bw_slot: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    // Channel-wise normalize first (LayerNorm-style across channels per time step),
    // then scale + shift.  HF Vocos uses LayerNorm; WavTokenizer uses AdaLayerNorm.
    for ti in 0..t {
        let mut mean = 0.0f32;
        for ci in 0..c { mean += x[ci*t + ti]; }
        mean /= c as f32;
        let mut var = 0.0f32;
        for ci in 0..c { let d = x[ci*t + ti] - mean; var += d*d; }
        var /= c as f32;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        for ci in 0..c {
            let xn = (x[ci*t + ti] - mean) * inv;
            let sc = scale_4c[bw_slot * c + ci];
            let sh = shift_4c[bw_slot * c + ci];
            out[ci*t + ti] = xn * sc + sh;
        }
    }
    out
}

struct ConvNextBlock {
    dw_w: Vec<f32>, dw_b: Vec<f32>,         // depthwise k=7
    norm_scale: Vec<f32>, norm_shift: Vec<f32>,
    pw1_w: Vec<f32>, pw1_b: Vec<f32>,       // [2304, 768]
    pw2_w: Vec<f32>, pw2_b: Vec<f32>,       // [768, 2304]
    gamma: Vec<f32>,
}
impl ConvNextBlock {
    fn load(sf: &SafetensorsFile, prefix: &str) -> Result<Self, WtError> {
        Ok(Self {
            dw_w:       g(sf, &format!("{prefix}.dwconv.weight"))?,
            dw_b:       g(sf, &format!("{prefix}.dwconv.bias"))?,
            norm_scale: g(sf, &format!("{prefix}.norm.scale.weight"))?,
            norm_shift: g(sf, &format!("{prefix}.norm.shift.weight"))?,
            pw1_w:      g(sf, &format!("{prefix}.pwconv1.weight"))?,
            pw1_b:      g(sf, &format!("{prefix}.pwconv1.bias"))?,
            pw2_w:      g(sf, &format!("{prefix}.pwconv2.weight"))?,
            pw2_b:      g(sf, &format!("{prefix}.pwconv2.bias"))?,
            gamma:      g(sf, &format!("{prefix}.gamma"))?,
        })
    }
    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let c = BACKBONE_HIDDEN;
        // dwconv k=7 (depthwise: groups=c, k=7), reflect-pad SAME.
        let pad = 3;
        // depthwise: for each channel, conv1d k=7 on its own row
        let mut after_dw = vec![0.0f32; c * t];
        for ci in 0..c {
            let mut row = vec![0.0f32; t + 2*pad];
            for i in 0..t { row[i + pad] = x[ci*t + i]; }
            // reflect-pad in place
            for i in 0..pad {
                row[pad - 1 - i] = x[ci*t + (i + 1).min(t-1)];
                row[pad + t + i] = x[ci*t + t.saturating_sub(2 + i).min(t-1)];
            }
            for to in 0..t {
                let mut acc = self.dw_b[ci];
                for kk in 0..7 {
                    acc += row[to + kk] * self.dw_w[ci*7 + kk];
                }
                after_dw[ci*t + to] = acc;
            }
        }
        // AdaLN
        let after_norm = adaln_ct(&after_dw, c, t, &self.norm_scale, &self.norm_shift, BW_SLOT);
        // pwconv1: 768 → 2304 (over channel dim, per-time-step).
        // pwconv1.weight is [2304, 768] applied as Linear in C dim.
        // Convert [C, T] to [T, C] for the linear, run, GELU, run pwconv2, then back to [C, T].
        let mut tc = vec![0.0f32; t * c];
        for ci in 0..c { for ti in 0..t { tc[ti*c + ci] = after_norm[ci*t + ti]; }}
        let mut h = linear(&tc, t, c, &self.pw1_w, FF, &self.pw1_b);
        for v in h.iter_mut() { *v = gelu(*v); }
        let h2 = linear(&h, t, FF, &self.pw2_w, c, &self.pw2_b);
        // LayerScale (gamma per channel) + residual
        let mut out = vec![0.0f32; c * t];
        for ti in 0..t { for ci in 0..c {
            let scaled = h2[ti*c + ci] * self.gamma[ci];
            out[ci*t + ti] = x[ci*t + ti] + scaled;
        }}
        out
    }
}

enum PosNetBlock {
    Conv {
        norm1_w: Vec<f32>, norm1_b: Vec<f32>,
        conv1_w: Vec<f32>, conv1_b: Vec<f32>,
        norm2_w: Vec<f32>, norm2_b: Vec<f32>,
        conv2_w: Vec<f32>, conv2_b: Vec<f32>,
    },
    Attn {
        q_w: Vec<f32>, q_b: Vec<f32>,
        k_w: Vec<f32>, k_b: Vec<f32>,
        v_w: Vec<f32>, v_b: Vec<f32>,
        o_w: Vec<f32>, o_b: Vec<f32>,
        norm_w: Vec<f32>, norm_b: Vec<f32>,
    },
    FinalNorm { w: Vec<f32>, b: Vec<f32> },
}
impl PosNetBlock {
    fn load(sf: &SafetensorsFile, idx: usize) -> Result<Self, WtError> {
        let pre = format!("backbone.pos_net.{idx}");
        if sf.get(&format!("{pre}.q.weight")).is_some() {
            Ok(PosNetBlock::Attn {
                q_w: g(sf, &format!("{pre}.q.weight"))?,
                q_b: g(sf, &format!("{pre}.q.bias"))?,
                k_w: g(sf, &format!("{pre}.k.weight"))?,
                k_b: g(sf, &format!("{pre}.k.bias"))?,
                v_w: g(sf, &format!("{pre}.v.weight"))?,
                v_b: g(sf, &format!("{pre}.v.bias"))?,
                o_w: g(sf, &format!("{pre}.proj_out.weight"))?,
                o_b: g(sf, &format!("{pre}.proj_out.bias"))?,
                norm_w: g(sf, &format!("{pre}.norm.weight"))?,
                norm_b: g(sf, &format!("{pre}.norm.bias"))?,
            })
        } else if sf.get(&format!("{pre}.conv1.weight")).is_some() {
            Ok(PosNetBlock::Conv {
                norm1_w: g(sf, &format!("{pre}.norm1.weight"))?,
                norm1_b: g(sf, &format!("{pre}.norm1.bias"))?,
                conv1_w: g(sf, &format!("{pre}.conv1.weight"))?,
                conv1_b: g(sf, &format!("{pre}.conv1.bias"))?,
                norm2_w: g(sf, &format!("{pre}.norm2.weight"))?,
                norm2_b: g(sf, &format!("{pre}.norm2.bias"))?,
                conv2_w: g(sf, &format!("{pre}.conv2.weight"))?,
                conv2_b: g(sf, &format!("{pre}.conv2.bias"))?,
            })
        } else {
            // FinalNorm: only `.weight` and `.bias` (no nesting).
            Ok(PosNetBlock::FinalNorm {
                w: g(sf, &format!("{pre}.weight"))?,
                b: g(sf, &format!("{pre}.bias"))?,
            })
        }
    }
    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let c = BACKBONE_HIDDEN;
        match self {
            PosNetBlock::Conv { norm1_w, norm1_b, conv1_w, conv1_b,
                                norm2_w, norm2_b, conv2_w, conv2_b } => {
                // Residual: x + conv2(silu(norm2(conv1(silu(norm1(x))))))
                // Actually HF Vocos uses GroupNorm + SiLU + conv k=3.
                let mut h = layer_norm_ct(x, c, t, norm1_w, norm1_b);
                for v in h.iter_mut() { *v = silu(*v); }
                let (h, _) = conv1d_reflect(&h, c, t, conv1_w, c, 3, 1, 1, conv1_b);
                let mut h = layer_norm_ct(&h, c, t, norm2_w, norm2_b);
                for v in h.iter_mut() { *v = silu(*v); }
                let (h, _) = conv1d_reflect(&h, c, t, conv2_w, c, 3, 1, 1, conv2_b);
                let mut out = vec![0.0f32; c * t];
                for i in 0..(c * t) { out[i] = x[i] + h[i]; }
                out
            }
            PosNetBlock::Attn { q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b, norm_w, norm_b } => {
                // Single-head self-attention over time axis with norm + residual.
                let h = layer_norm_ct(x, c, t, norm_w, norm_b);
                // q, k, v: conv1d k=1 on [C, T] → [C, T].
                let (q, _) = conv1d_reflect(&h, c, t, q_w, c, 1, 1, 1, q_b);
                let (k, _) = conv1d_reflect(&h, c, t, k_w, c, 1, 1, 1, k_b);
                let (v, _) = conv1d_reflect(&h, c, t, v_w, c, 1, 1, 1, v_b);
                let scale = 1.0 / (c as f32).sqrt();
                let mut attn_out = vec![0.0f32; c * t];
                for ti in 0..t {
                    let mut scores = vec![0.0f32; t];
                    let mut maxv = f32::NEG_INFINITY;
                    for tj in 0..t {
                        let mut s = 0.0f32;
                        for ci in 0..c { s += q[ci*t + ti] * k[ci*t + tj]; }
                        s *= scale;
                        scores[tj] = s;
                        if s > maxv { maxv = s; }
                    }
                    let mut sum = 0.0f32;
                    for tj in 0..t { scores[tj] = (scores[tj] - maxv).exp(); sum += scores[tj]; }
                    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                    for tj in 0..t { scores[tj] *= inv; }
                    for ci in 0..c {
                        let mut acc = 0.0f32;
                        for tj in 0..t { acc += scores[tj] * v[ci*t + tj]; }
                        attn_out[ci*t + ti] = acc;
                    }
                }
                let (h, _) = conv1d_reflect(&attn_out, c, t, o_w, c, 1, 1, 1, o_b);
                let mut out = vec![0.0f32; c * t];
                for i in 0..(c * t) { out[i] = x[i] + h[i]; }
                out
            }
            PosNetBlock::FinalNorm { w, b } => {
                layer_norm_ct(x, c, t, w, b)
            }
        }
    }
}

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

struct Backbone {
    embed_w: Vec<f32>, embed_b: Vec<f32>,             // [768, 512, 7]
    norm_scale: Vec<f32>, norm_shift: Vec<f32>,       // [4, 768]
    convnext: Vec<ConvNextBlock>,
    pos_net:  Vec<PosNetBlock>,
    final_ln_w: Vec<f32>, final_ln_b: Vec<f32>,
}
impl Backbone {
    fn load(sf: &SafetensorsFile) -> Result<Self, WtError> {
        let embed_w = g(sf, "backbone.embed.weight")?;
        let embed_b = g(sf, "backbone.embed.bias")?;
        let norm_scale = g(sf, "backbone.norm.scale.weight")?;
        let norm_shift = g(sf, "backbone.norm.shift.weight")?;
        let mut convnext = Vec::with_capacity(N_CONVNEXT);
        for i in 0..N_CONVNEXT {
            convnext.push(ConvNextBlock::load(sf, &format!("backbone.convnext.{i}"))?);
        }
        let mut pos_net = Vec::with_capacity(N_POS_NET);
        for i in 0..N_POS_NET { pos_net.push(PosNetBlock::load(sf, i)?); }
        let final_ln_w = g(sf, "backbone.final_layer_norm.weight")?;
        let final_ln_b = g(sf, "backbone.final_layer_norm.bias")?;
        Ok(Self { embed_w, embed_b, norm_scale, norm_shift, convnext, pos_net, final_ln_w, final_ln_b })
    }
    fn forward(&self, z_q: &[f32], t_lat: usize) -> Vec<f32> {
        let (y, t) = conv1d_reflect(z_q, HIDDEN_LATENT, t_lat,
            &self.embed_w, BACKBONE_HIDDEN, 7, 1, 1, &self.embed_b);
        let mut x = adaln_ct(&y, BACKBONE_HIDDEN, t, &self.norm_scale, &self.norm_shift, BW_SLOT);
        for blk in &self.convnext { x = blk.forward(&x, t); }
        for blk in &self.pos_net  { x = blk.forward(&x, t); }
        // Final LN on per-time-step channel vector.
        layer_norm_ct(&x, BACKBONE_HIDDEN, t, &self.final_ln_w, &self.final_ln_b)
    }
}

// ── ISTFT head ──────────────────────────────────────────────────────────────

struct Head { window: Vec<f32>, out_w: Vec<f32>, out_b: Vec<f32> }
impl Head {
    fn load(sf: &SafetensorsFile) -> Result<Self, WtError> {
        Ok(Self {
            window: g(sf, "head.istft.window")?,
            out_w:  g(sf, "head.out.weight")?,
            out_b:  g(sf, "head.out.bias")?,
        })
    }
    fn forward(&self, hidden_ct: &[f32], t: usize) -> Vec<f32> {
        // Reshape [C=768, T] → [T, 768] and project to [T, 1282] via Linear.
        let mut tc = vec![0.0f32; t * BACKBONE_HIDDEN];
        for ti in 0..t { for ci in 0..BACKBONE_HIDDEN {
            tc[ti*BACKBONE_HIDDEN + ci] = hidden_ct[ci*t + ti];
        }}
        let proj = linear(&tc, t, BACKBONE_HIDDEN, &self.out_w, HEAD_OUT_DIM, &self.out_b);
        // Split per time step: first 641 = magnitude (we will exp-clamp), second 641 = phase.
        let mag_dim = N_BIN;
        let phs_dim = N_BIN;
        // Build complex spectrogram [T_frames, N_FFT] (one-sided then mirror).
        let n_frames = t;
        let mut spec_re = vec![0.0f32; n_frames * N_FFT];
        let mut spec_im = vec![0.0f32; n_frames * N_FFT];
        for f in 0..n_frames {
            for b in 0..N_BIN {
                let mag = proj[f*HEAD_OUT_DIM + b].exp().min(1e2);
                let phs = proj[f*HEAD_OUT_DIM + mag_dim + b];
                let re = mag * phs.cos();
                let im = mag * phs.sin();
                spec_re[f*N_FFT + b] = re;
                spec_im[f*N_FFT + b] = im;
                // Conjugate mirror for the upper half.
                if b > 0 && b < N_BIN - 1 {
                    spec_re[f*N_FFT + N_FFT - b] = re;
                    spec_im[f*N_FFT + N_FFT - b] = -im;
                }
            }
        }
        // ISTFT: for each frame, IFFT → multiply by window → overlap-add.
        use rustfft::{FftPlanner, num_complex::Complex};
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_inverse(N_FFT);
        let n_out = (n_frames - 1) * HOP + N_FFT;
        let mut accum = vec![0.0f32; n_out];
        let mut win_sq = vec![0.0f32; n_out];
        let mut buf = vec![Complex::<f32>::new(0.0, 0.0); N_FFT];
        for f in 0..n_frames {
            for i in 0..N_FFT {
                buf[i] = Complex::new(spec_re[f*N_FFT + i], spec_im[f*N_FFT + i]);
            }
            fft.process(&mut buf);
            let inv = 1.0 / N_FFT as f32;
            for i in 0..N_FFT {
                let sample = buf[i].re * inv * self.window[i];
                let pos = f * HOP + i;
                accum[pos] += sample;
                win_sq[pos] += self.window[i] * self.window[i];
            }
        }
        for i in 0..n_out {
            if win_sq[i] > 1e-12 { accum[i] /= win_sq[i]; }
        }
        // Strip the center-pad — HF Vocos / WavTokenizer use a half-window pad.
        let center_pad = N_FFT / 2;
        let kept = accum.len().saturating_sub(2 * center_pad);
        accum[center_pad..center_pad + kept].to_vec()
    }
}

// ── full model ───────────────────────────────────────────────────────────────
pub struct WavTokenizer {
    encoder: Encoder,
    quantizer: Quantizer,
    backbone: Backbone,
    head: Head,
}
impl WavTokenizer {
    pub fn load(path: &Path) -> Result<Arc<Self>, WtError> {
        let sf = SafetensorsFile::load(path).map_err(|e| WtError::Load(e.to_string()))?;
        Ok(Arc::new(Self {
            encoder:   Encoder::load(&sf)?,
            quantizer: Quantizer::load(&sf)?,
            backbone:  Backbone::load(&sf)?,
            head:      Head::load(&sf)?,
        }))
    }
    pub fn encode(&self, audio: &[f32]) -> (Vec<u32>, Vec<f32>, usize) {
        let (z, t) = self.encoder.forward(audio);
        let (codes, z_q) = self.quantizer.encode(&z, t);
        (codes, z_q, t)
    }
    pub fn decode(&self, z_q: &[f32], t_lat: usize) -> Vec<f32> {
        let hidden = self.backbone.forward(z_q, t_lat);
        self.head.forward(&hidden, t_lat)
    }
    pub fn roundtrip(&self, audio: &[f32]) -> (Vec<f32>, Vec<u32>) {
        let (codes, z_q, t) = self.encode(audio);
        let wav = self.decode(&z_q, t);
        (wav, codes)
    }
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

    fn mag_signature(audio: &[f32], n_fft: usize, hop: usize) -> Vec<f32> {
        use rustfft::{FftPlanner, num_complex::Complex};
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);
        let win: Vec<f32> = (0..n_fft).map(|i| 0.5 - 0.5 * ((2.0*std::f32::consts::PI*i as f32)/(n_fft as f32 - 1.0)).cos()).collect();
        let n_frames = if audio.len() < n_fft { 0 } else { (audio.len() - n_fft) / hop + 1 };
        let n_bin = n_fft/2 + 1;
        let mut sig = vec![0.0f32; n_frames * n_bin];
        let mut buf = vec![Complex::<f32>::new(0.0, 0.0); n_fft];
        for f in 0..n_frames {
            for i in 0..n_fft { buf[i] = Complex::new(audio[f*hop + i] * win[i], 0.0); }
            fft.process(&mut buf);
            for b in 0..n_bin { sig[f*n_bin + b] = (buf[b].norm_sqr() + 1e-8).ln(); }
        }
        sig
    }

    fn model_path() -> Option<std::path::PathBuf> {
        let p = std::path::PathBuf::from(
            "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/wavtokenizer/model.safetensors");
        if p.exists() { Some(p) } else { None }
    }

    #[test]
    #[ignore = "wavtokenizer 24khz; loads ~600 MB; multi-second forward"]
    fn wavtokenizer_roundtrip() {
        let Some(p) = model_path() else { eprintln!("skip"); return; };
        eprintln!("loading {}…", p.display());
        let t0 = Instant::now();
        let m = WavTokenizer::load(&p).expect("load");
        eprintln!("loaded in {:?}", t0.elapsed());

        // 0.96 s 24 kHz two-tone signal (divisible by 320 → 72 latent frames).
        let sr = SR as f32;
        let n = TOTAL_DOWNSAMPLE * 72;
        let audio: Vec<f32> = (0..n).map(|i| {
            let t = i as f32 / sr;
            0.4 * (2.0*std::f32::consts::PI * 440.0 * t).sin()
              + 0.3 * (2.0*std::f32::consts::PI * 880.0 * t).sin()
        }).collect();

        let t0 = Instant::now();
        let (codes, z_q, t_lat) = m.encode(&audio);
        eprintln!("encode: {t_lat} latent frames, {} codes, in {:?}", codes.len(), t0.elapsed());
        let t0 = Instant::now();
        let recon = m.decode(&z_q, t_lat);
        eprintln!("decode: {} samples in {:?}", recon.len(), t0.elapsed());

        let peak = recon.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        let rms_in:  f32 = (audio.iter().map(|x| x*x).sum::<f32>() / audio.len() as f32).sqrt();
        let rms_out: f32 = (recon.iter().map(|x| x*x).sum::<f32>() / recon.len() as f32).sqrt();
        let ref_sig = mag_signature(&audio, 1024, 256);
        let rec_sig = mag_signature(&recon, 1024, 256);
        let corr = pearson(&ref_sig, &rec_sig);
        eprintln!("wavtokenizer: peak={peak:.4} rms_in={rms_in:.4} rms_out={rms_out:.4} spec corr={corr:.4}");
        assert!(peak.is_finite() && peak > 0.0, "silent reconstruction");
        assert!(corr > 0.4, "spec corr too low: {corr}");
    }
}
