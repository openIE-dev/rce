//! `facebook/encodec_32khz` on rce.
//!
//! ```text
//! encoder
//!   L0   Conv1d 1→64,  k=7  s=1
//!   L1   ResidualUnit 64   (compress=2: 64→32→64, k=3 then k=1)
//!   L2   ELU
//!   L3   Conv1d 64→128, k=8 s=4   (downsample ×4)
//!   L4   ResidualUnit 128
//!   L5   ELU
//!   L6   Conv1d 128→256, k=8 s=4
//!   L7   ResidualUnit 256
//!   L8   ELU
//!   L9   Conv1d 256→512, k=10 s=5
//!   L10  ResidualUnit 512
//!   L11  ELU
//!   L12  Conv1d 512→1024, k=16 s=8
//!   L13  LSTM (2 layers, hidden=1024)  — residual: lstm_out + input
//!   L14  ELU
//!   L15  Conv1d 1024→128, k=7 s=1
//!
//! quantizer.layers.{0..3}.codebook.embed [2048, 128]     (4-stage residual VQ)
//!
//! decoder: symmetric to encoder using ConvTranspose1d for upsampling.
//! ```
//!
//! All convs are weight-norm'd with `weight_g [Cout,1,1]` + `weight_v [Cout,Cin,K]`
//! (and `[Cin, Cout, K]` for ConvT).  Padding is **reflect** (not zero) per HF
//! EnCodec; `pad_mode="reflect"` and `use_causal_conv=false` in `config.json`.

use rce_op_lstm::{lstm_forward, LstmConfig, LstmLayer};
use rce_weight_loader::{LoadedTensor, SafetensorsFile};
use std::path::Path;
use std::sync::Arc;

pub const SR: u32 = 32_000;
pub const HIDDEN_LATENT: usize = 128;   // codebook_dim
pub const LSTM_HIDDEN: usize = 1024;
pub const N_RVQ: usize = 4;
pub const CODEBOOK_SIZE: usize = 2048;
pub const STRIDES: [usize; 4] = [4, 4, 5, 8];   // encoder downsample (= decoder upsample reversed)
pub const KERNELS: [usize; 4] = [8, 8, 10, 16]; // = 2*stride
pub const CHANNELS: [usize; 5] = [64, 128, 256, 512, 1024];
pub const COMPRESS: usize = 2;
pub const RES_KERNEL: usize = 3;
pub const INIT_KERNEL: usize = 7;
pub const FINAL_KERNEL: usize = 7;
pub const TOTAL_DOWNSAMPLE: usize = 4 * 4 * 5 * 8;   // 640 → 32k / 640 = 50 Hz latent

#[derive(Debug)]
pub enum EncodecError { Load(String) }
impl std::fmt::Display for EncodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::Load(m) => write!(f, "encodec load: {m}") }
    }
}
impl std::error::Error for EncodecError {}

fn tf32(t: &LoadedTensor) -> Vec<f32> {
    let bytes = t.data.as_slice();
    match t.dtype.as_str() {
        "f32" => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect(),
        "f16" => {
            // NEON-vectorized f16 → f32 via `vcvt_f32_f16` (4 lanes/cycle on Apple Silicon).
            // Whole EnCodec file is 117 MB f16 → ~5-10× faster load than scalar.
            debug_assert_eq!(bytes.len() % 2, 0);
            debug_assert_eq!(bytes.as_ptr() as usize % 2, 0);
            let src: &[half::f16] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr().cast::<half::f16>(), bytes.len() / 2)
            };
            rce_op_conv2d::dequant_f16_to_f32(src)
        }
        "bf16" => bytes.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0],c[1]]).to_f32()).collect(),
        other => panic!("encodec: unsupported dtype {other}"),
    }
}
fn get(sf: &SafetensorsFile, name: &str) -> Result<Vec<f32>, EncodecError> {
    let t = sf.get(name).ok_or_else(|| EncodecError::Load(format!("missing {name}")))?;
    Ok(tf32(t))
}
fn get_shape(sf: &SafetensorsFile, name: &str) -> Result<(Vec<f32>, Vec<usize>), EncodecError> {
    let t = sf.get(name).ok_or_else(|| EncodecError::Load(format!("missing {name}")))?;
    Ok((tf32(t), t.shape.clone()))
}

// ── ops ─────────────────────────────────────────────────────────────────────

fn elu(x: f32) -> f32 { if x >= 0.0 { x } else { x.exp() - 1.0 } }

/// PyTorch weight_norm with `dim=0`: keep dim 0 (= Cout for Conv1d), normalize
/// over the remaining dims. Effective weight = `w_v * (g / ‖w_v[k, :, :]‖₂)`.
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

/// HF EnCodec uses asymmetric reflect padding so that
///   output_length = ceil(input_length / stride)
/// regardless of kernel size.  The right-side pad pre-includes a stride-related
/// adjustment to make this happen.  Source: `Wav2Vec2Pad` / `Encodec._pad1d` in
/// transformers; equivalent algorithm reproduced here.
fn pad_reflect_1d(x: &[f32], c: usize, t: usize, k: usize, dilation: usize, stride: usize) -> (Vec<f32>, usize) {
    // total_pad = (k - 1) * dilation
    let total_pad = (k - 1) * dilation;
    // HF EnCodec also pads the right so that `(t + extra) % stride == 0`
    // and then `(padded_len - k) / stride + 1` matches `ceil(t / stride)`.
    // Their `_get_extra_padding_for_conv1d` does this.
    let ideal_len = ((t as f32 / stride as f32).ceil() as usize) * stride;
    let extra_padding = ideal_len.saturating_sub(t);
    let right = total_pad / 2 + extra_padding;
    let left = total_pad - total_pad / 2;
    // Reflect pad: pad with `x[idx]` where idx mirrors at the boundaries.
    let new_t = t + left + right;
    let mut out = vec![0.0f32; c * new_t];
    for ci in 0..c {
        for i in 0..new_t {
            let j = i as i64 - left as i64;
            let idx = if j < 0 { (-j) as usize } else if (j as usize) >= t {
                // reflect: t-2-(j-t) = 2t - 2 - j
                let jj = 2 * t as i64 - 2 - j;
                if jj < 0 || (jj as usize) >= t { 0 } else { jj as usize }
            } else { j as usize };
            out[ci * new_t + i] = x[ci * t + idx.min(t - 1)];
        }
    }
    (out, new_t)
}

/// Conv1d on channel-major `[Cin, T]`. `w: [Cout, Cin, K]`, `b: [Cout]`.
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

fn conv1d_with_reflect(x: &[f32], cin: usize, t: usize,
                       w: &[f32], cout: usize, k: usize,
                       stride: usize, dilation: usize,
                       bias: &[f32]) -> (Vec<f32>, usize) {
    let (padded, pt) = pad_reflect_1d(x, cin, t, k, dilation, stride);
    conv1d(&padded, cin, pt, w, cout, k, stride, dilation, bias)
}

/// ConvTranspose1d. w: `[Cin, Cout, K]`, b: `[Cout]`.  No padding inside; HF
/// EnCodec instead trims the output ends to match the up-sampled length.
fn conv_t1d(x: &[f32], cin: usize, t: usize,
            w: &[f32], cout: usize, k: usize, stride: usize,
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
    for oc in 0..cout {
        for to in 0..t_out_full { full[oc * t_out_full + to] += bias[oc]; }
    }
    // HF EnCodec strips `(kernel - stride) // 2` from each end:
    let trim = (k - stride) / 2;
    let trim_right = k - stride - trim;
    let t_kept = t_out_full.saturating_sub(trim + trim_right);
    let mut out = vec![0.0f32; cout * t_kept];
    for oc in 0..cout {
        for to in 0..t_kept {
            out[oc * t_kept + to] = full[oc * t_out_full + to + trim];
        }
    }
    (out, t_kept)
}

// ── ResidualUnit ────────────────────────────────────────────────────────────
struct ResidualUnit {
    c: usize,
    w_compress: Vec<f32>, b_compress: Vec<f32>,
    w_expand:   Vec<f32>, b_expand:   Vec<f32>,
}
impl ResidualUnit {
    fn load(sf: &SafetensorsFile, prefix: &str, c: usize) -> Result<Self, EncodecError> {
        // block.1 = compress (c → c/compress, k=RES_KERNEL)
        let (g, gs) = get_shape(sf, &format!("{prefix}.block.1.conv.weight_g"))?;
        let (v, vs) = get_shape(sf, &format!("{prefix}.block.1.conv.weight_v"))?;
        let b_compress = get(sf, &format!("{prefix}.block.1.conv.bias"))?;
        debug_assert_eq!(vs, vec![c / COMPRESS, c, RES_KERNEL]);
        debug_assert_eq!(gs, vec![c / COMPRESS, 1, 1]);
        let w_compress = wn_dim0(&g, &v, &vs);
        // block.3 = expand (c/compress → c, k=1)
        let (g, gs) = get_shape(sf, &format!("{prefix}.block.3.conv.weight_g"))?;
        let (v, vs) = get_shape(sf, &format!("{prefix}.block.3.conv.weight_v"))?;
        let b_expand = get(sf, &format!("{prefix}.block.3.conv.bias"))?;
        debug_assert_eq!(vs, vec![c, c / COMPRESS, 1]);
        debug_assert_eq!(gs, vec![c, 1, 1]);
        let w_expand = wn_dim0(&g, &v, &vs);
        Ok(Self { c, w_compress, b_compress, w_expand, b_expand })
    }
    fn forward(&self, x: &[f32], t: usize) -> (Vec<f32>, usize) {
        let cin = self.c;
        let cmid = self.c / COMPRESS;
        // block.0 = ELU
        let mut x0 = x.to_vec();
        for v in x0.iter_mut() { *v = elu(*v); }
        // block.1 = Conv1d (compress)
        let (y1, t1) = conv1d_with_reflect(&x0, cin, t, &self.w_compress, cmid, RES_KERNEL, 1, 1, &self.b_compress);
        // block.2 = ELU
        let mut y1 = y1; for v in y1.iter_mut() { *v = elu(*v); }
        // block.3 = Conv1d (expand, k=1)
        let (y2, _) = conv1d_with_reflect(&y1, cmid, t1, &self.w_expand, cin, 1, 1, 1, &self.b_expand);
        // Residual: x + y2
        let mut out = vec![0.0f32; cin * t];
        for i in 0..(cin * t) { out[i] = x[i] + y2[i]; }
        (out, t)
    }
}

// ── Conv1d/T1d weight bundle ────────────────────────────────────────────────
struct ConvW { w: Vec<f32>, b: Vec<f32>, k: usize, stride: usize }
impl ConvW {
    fn load(sf: &SafetensorsFile, prefix: &str, k: usize, stride: usize, cout: usize, cin: usize)
        -> Result<Self, EncodecError>
    {
        let (g, gs) = get_shape(sf, &format!("{prefix}.conv.weight_g"))?;
        let (v, vs) = get_shape(sf, &format!("{prefix}.conv.weight_v"))?;
        let b = get(sf, &format!("{prefix}.conv.bias"))?;
        if vs != vec![cout, cin, k] {
            return Err(EncodecError::Load(format!("{prefix}.conv.weight_v: {:?} != [{cout},{cin},{k}]", vs)));
        }
        if gs != vec![cout, 1, 1] {
            return Err(EncodecError::Load(format!("{prefix}.conv.weight_g: {:?} != [{cout},1,1]", gs)));
        }
        Ok(Self { w: wn_dim0(&g, &v, &vs), b, k, stride })
    }
}
struct ConvTW { w: Vec<f32>, b: Vec<f32>, k: usize, stride: usize }
impl ConvTW {
    fn load(sf: &SafetensorsFile, prefix: &str, k: usize, stride: usize, cin: usize, cout: usize)
        -> Result<Self, EncodecError>
    {
        // ConvTranspose1d w shape is [Cin, Cout, K], weight_norm dim=0 keeps Cin.
        let (g, gs) = get_shape(sf, &format!("{prefix}.conv.weight_g"))?;
        let (v, vs) = get_shape(sf, &format!("{prefix}.conv.weight_v"))?;
        let b = get(sf, &format!("{prefix}.conv.bias"))?;
        if vs != vec![cin, cout, k] {
            return Err(EncodecError::Load(format!("{prefix}.conv.weight_v: {:?} != [{cin},{cout},{k}]", vs)));
        }
        if gs != vec![cin, 1, 1] {
            return Err(EncodecError::Load(format!("{prefix}.conv.weight_g: {:?} != [{cin},1,1]", gs)));
        }
        Ok(Self { w: wn_dim0(&g, &v, &vs), b, k, stride })
    }
}

// ── encoder ─────────────────────────────────────────────────────────────────
struct Encoder {
    pre:    ConvW,                  // L0
    res:    [ResidualUnit; 4],      // L1, L4, L7, L10
    down:   [ConvW; 4],             // L3, L6, L9, L12
    lstm:   Vec<LstmLayer>,         // L13 (2 layers)
    post:   ConvW,                  // L15
}
impl Encoder {
    fn load(sf: &SafetensorsFile) -> Result<Self, EncodecError> {
        let pre  = ConvW::load(sf, "encoder.layers.0", INIT_KERNEL, 1, CHANNELS[0], 1)?;
        let res = [
            ResidualUnit::load(sf, "encoder.layers.1",  CHANNELS[0])?,
            ResidualUnit::load(sf, "encoder.layers.4",  CHANNELS[1])?,
            ResidualUnit::load(sf, "encoder.layers.7",  CHANNELS[2])?,
            ResidualUnit::load(sf, "encoder.layers.10", CHANNELS[3])?,
        ];
        let down = [
            ConvW::load(sf, "encoder.layers.3",  KERNELS[0], STRIDES[0], CHANNELS[1], CHANNELS[0])?,
            ConvW::load(sf, "encoder.layers.6",  KERNELS[1], STRIDES[1], CHANNELS[2], CHANNELS[1])?,
            ConvW::load(sf, "encoder.layers.9",  KERNELS[2], STRIDES[2], CHANNELS[3], CHANNELS[2])?,
            ConvW::load(sf, "encoder.layers.12", KERNELS[3], STRIDES[3], CHANNELS[4], CHANNELS[3])?,
        ];
        let mut lstm = Vec::with_capacity(2);
        for li in 0..2 {
            lstm.push(LstmLayer {
                weight_ih: get(sf, &format!("encoder.layers.13.lstm.weight_ih_l{li}"))?,
                weight_hh: get(sf, &format!("encoder.layers.13.lstm.weight_hh_l{li}"))?,
                bias_ih:   get(sf, &format!("encoder.layers.13.lstm.bias_ih_l{li}"))?,
                bias_hh:   get(sf, &format!("encoder.layers.13.lstm.bias_hh_l{li}"))?,
            });
        }
        let post = ConvW::load(sf, "encoder.layers.15", FINAL_KERNEL, 1, HIDDEN_LATENT, CHANNELS[4])?;
        Ok(Self { pre, res, down, lstm, post })
    }
    fn forward(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        // L0
        let (mut x, mut t) = conv1d_with_reflect(audio, 1, audio.len(),
            &self.pre.w, CHANNELS[0], self.pre.k, self.pre.stride, 1, &self.pre.b);
        // Stages
        for s in 0..4 {
            // ResidualUnit
            let (y, t1) = self.res[s].forward(&x, t);
            x = y; t = t1;
            // ELU
            for v in x.iter_mut() { *v = elu(*v); }
            // Downsample
            let (y, t1) = conv1d_with_reflect(&x, CHANNELS[s], t,
                &self.down[s].w, CHANNELS[s+1], self.down[s].k, self.down[s].stride, 1, &self.down[s].b);
            x = y; t = t1;
        }
        // L13: LSTM — input is [C=1024, T] channel-major; convert to [T, C] for LSTM.
        let c = CHANNELS[4];
        let mut x_tc = vec![0.0f32; t * c];
        for ti in 0..t { for ci in 0..c {
            x_tc[ti*c + ci] = x[ci*t + ti];
        }}
        let cfg = LstmConfig { input_size: c, hidden_size: LSTM_HIDDEN, num_layers: 2 };
        let (y_tc, _, _) = lstm_forward(&x_tc, t, &cfg, &self.lstm, None, None);
        // HF EnCodec adds the LSTM output as a residual: out = lstm(x) + x
        let mut x_after_lstm = vec![0.0f32; c * t];
        for ti in 0..t { for ci in 0..c {
            x_after_lstm[ci*t + ti] = y_tc[ti*c + ci] + x[ci*t + ti];
        }}
        x = x_after_lstm;
        // L14 ELU
        for v in x.iter_mut() { *v = elu(*v); }
        // L15
        let (y, t1) = conv1d_with_reflect(&x, c, t,
            &self.post.w, HIDDEN_LATENT, self.post.k, self.post.stride, 1, &self.post.b);
        (y, t1)
    }
}

// ── quantizer ───────────────────────────────────────────────────────────────
struct Quantizer { codebooks: Vec<Vec<f32>> }  // 4 × [CODEBOOK_SIZE × HIDDEN_LATENT]
impl Quantizer {
    fn load(sf: &SafetensorsFile) -> Result<Self, EncodecError> {
        let mut codebooks = Vec::with_capacity(N_RVQ);
        for i in 0..N_RVQ {
            codebooks.push(get(sf, &format!("quantizer.layers.{i}.codebook.embed"))?);
        }
        Ok(Self { codebooks })
    }
    /// Encode latent `z [HIDDEN_LATENT, T]` → 4 × `[T]` code arrays + reconstructed
    /// latent (sum across the 4 stages).
    fn encode(&self, z: &[f32], t: usize) -> (Vec<Vec<u32>>, Vec<f32>) {
        let d = HIDDEN_LATENT;
        let mut residual = z.to_vec();
        let mut codes_all = Vec::with_capacity(N_RVQ);
        let mut z_q_sum = vec![0.0f32; d * t];
        for stage in 0..N_RVQ {
            let cb = &self.codebooks[stage];
            let mut codes = vec![0u32; t];
            for ti in 0..t {
                // Nearest-neighbor in 128-dim space.
                let mut best = 0u32; let mut best_d = f32::INFINITY;
                for ci in 0..CODEBOOK_SIZE {
                    let mut s = 0.0f32;
                    for k in 0..d {
                        let diff = residual[k*t + ti] - cb[ci*d + k];
                        s += diff * diff;
                    }
                    if s < best_d { best_d = s; best = ci as u32; }
                }
                codes[ti] = best;
                // Subtract codebook entry from residual; add to z_q_sum.
                for k in 0..d {
                    let v = cb[best as usize * d + k];
                    residual[k*t + ti] -= v;
                    z_q_sum[k*t + ti]  += v;
                }
            }
            codes_all.push(codes);
        }
        (codes_all, z_q_sum)
    }
    /// Decode codes → latent `[HIDDEN_LATENT, T]`.
    pub fn decode(&self, codes: &[Vec<u32>], t: usize) -> Vec<f32> {
        let d = HIDDEN_LATENT;
        let mut out = vec![0.0f32; d * t];
        for stage in 0..codes.len().min(N_RVQ) {
            let cb = &self.codebooks[stage];
            for ti in 0..t {
                let id = codes[stage][ti] as usize;
                for k in 0..d {
                    out[k*t + ti] += cb[id * d + k];
                }
            }
        }
        out
    }
}

// ── decoder ─────────────────────────────────────────────────────────────────
struct Decoder {
    pre:  ConvW,                  // L0  Conv 128→1024 k=7
    lstm: Vec<LstmLayer>,         // L1  LSTM 2-layer
    up:   [ConvTW; 4],            // L3, L6, L9, L12 ConvT
    res:  [ResidualUnit; 4],      // L4, L7, L10, L13
    post: ConvW,                  // L15 Conv 64→1 k=7
}
impl Decoder {
    fn load(sf: &SafetensorsFile) -> Result<Self, EncodecError> {
        let pre = ConvW::load(sf, "decoder.layers.0", INIT_KERNEL, 1, CHANNELS[4], HIDDEN_LATENT)?;
        let mut lstm = Vec::with_capacity(2);
        for li in 0..2 {
            lstm.push(LstmLayer {
                weight_ih: get(sf, &format!("decoder.layers.1.lstm.weight_ih_l{li}"))?,
                weight_hh: get(sf, &format!("decoder.layers.1.lstm.weight_hh_l{li}"))?,
                bias_ih:   get(sf, &format!("decoder.layers.1.lstm.bias_ih_l{li}"))?,
                bias_hh:   get(sf, &format!("decoder.layers.1.lstm.bias_hh_l{li}"))?,
            });
        }
        // Decoder upsample order reverses encoder: K=[16,10,8,8], S=[8,5,4,4],
        // channels 1024→512→256→128→64.
        let up = [
            ConvTW::load(sf, "decoder.layers.3",  KERNELS[3], STRIDES[3], CHANNELS[4], CHANNELS[3])?,
            ConvTW::load(sf, "decoder.layers.6",  KERNELS[2], STRIDES[2], CHANNELS[3], CHANNELS[2])?,
            ConvTW::load(sf, "decoder.layers.9",  KERNELS[1], STRIDES[1], CHANNELS[2], CHANNELS[1])?,
            ConvTW::load(sf, "decoder.layers.12", KERNELS[0], STRIDES[0], CHANNELS[1], CHANNELS[0])?,
        ];
        let res = [
            ResidualUnit::load(sf, "decoder.layers.4",  CHANNELS[3])?,
            ResidualUnit::load(sf, "decoder.layers.7",  CHANNELS[2])?,
            ResidualUnit::load(sf, "decoder.layers.10", CHANNELS[1])?,
            ResidualUnit::load(sf, "decoder.layers.13", CHANNELS[0])?,
        ];
        let post = ConvW::load(sf, "decoder.layers.15", FINAL_KERNEL, 1, 1, CHANNELS[0])?;
        Ok(Self { pre, lstm, up, res, post })
    }
    fn forward(&self, z_q: &[f32], t_lat: usize) -> Vec<f32> {
        // L0 Conv 128→1024
        let (x0, t0) = conv1d_with_reflect(z_q, HIDDEN_LATENT, t_lat,
            &self.pre.w, CHANNELS[4], self.pre.k, self.pre.stride, 1, &self.pre.b);
        let mut x = x0; let mut t = t0;
        // L1 LSTM (residual): out = lstm(x) + x
        let c = CHANNELS[4];
        let mut x_tc = vec![0.0f32; t * c];
        for ti in 0..t { for ci in 0..c { x_tc[ti*c + ci] = x[ci*t + ti]; }}
        let cfg = LstmConfig { input_size: c, hidden_size: LSTM_HIDDEN, num_layers: 2 };
        let (y_tc, _, _) = lstm_forward(&x_tc, t, &cfg, &self.lstm, None, None);
        let mut x_after = vec![0.0f32; c * t];
        for ti in 0..t { for ci in 0..c {
            x_after[ci*t + ti] = y_tc[ti*c + ci] + x[ci*t + ti];
        }}
        x = x_after;
        // 4 upsample stages
        for s in 0..4 {
            // ELU
            for v in x.iter_mut() { *v = elu(*v); }
            // ConvT
            let (y, t1) = conv_t1d(&x, CHANNELS[4 - s], t,
                &self.up[s].w, CHANNELS[3 - s], self.up[s].k, self.up[s].stride, &self.up[s].b);
            x = y; t = t1;
            // ResidualUnit
            let (y, t1) = self.res[s].forward(&x, t);
            x = y; t = t1;
        }
        // L14 ELU
        for v in x.iter_mut() { *v = elu(*v); }
        // L15 final Conv 64→1
        let (y, _) = conv1d_with_reflect(&x, CHANNELS[0], t,
            &self.post.w, 1, self.post.k, self.post.stride, 1, &self.post.b);
        y
    }
}

// ── full model ───────────────────────────────────────────────────────────────
pub struct Encodec {
    encoder: Encoder,
    quantizer: Quantizer,
    decoder: Decoder,
}
impl Encodec {
    pub fn load(path: &Path) -> Result<Arc<Self>, EncodecError> {
        let sf = SafetensorsFile::load(path).map_err(|e| EncodecError::Load(e.to_string()))?;
        Ok(Arc::new(Self {
            encoder:   Encoder::load(&sf)?,
            quantizer: Quantizer::load(&sf)?,
            decoder:   Decoder::load(&sf)?,
        }))
    }
    pub fn encode(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        self.encoder.forward(audio)
    }
    pub fn quantize(&self, z: &[f32], t: usize) -> (Vec<Vec<u32>>, Vec<f32>) {
        self.quantizer.encode(z, t)
    }
    pub fn decode(&self, z_q: &[f32], t: usize) -> Vec<f32> {
        self.decoder.forward(z_q, t)
    }
    pub fn decode_codes(&self, codes: &[Vec<u32>], t_lat: usize) -> Vec<f32> {
        let z_q = self.quantizer.decode(codes, t_lat);
        self.decoder.forward(&z_q, t_lat)
    }
    pub fn roundtrip(&self, audio: &[f32]) -> (Vec<f32>, Vec<Vec<u32>>) {
        let (z, t) = self.encode(audio);
        let (codes, z_q) = self.quantize(&z, t);
        (self.decode(&z_q, t), codes)
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
            "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/encodec-32khz/model.safetensors");
        if p.exists() { Some(p) } else { None }
    }

    #[test]
    #[ignore = "encodec 32khz; loads 236 MB; long forward"]
    fn encodec_roundtrip() {
        let Some(p) = model_path() else { eprintln!("skip: model missing"); return; };
        eprintln!("loading {}…", p.display());
        let t0 = Instant::now();
        let m = Encodec::load(&p).expect("load");
        eprintln!("loaded in {:?}", t0.elapsed());

        // Deterministic 2-tone signal: 440 Hz + 880 Hz at 32 kHz, 0.64 s.
        // 0.64 × 32000 = 20480, divisible by TOTAL_DOWNSAMPLE = 640 → 32 latent frames.
        let sr = SR as f32;
        let n = TOTAL_DOWNSAMPLE * 32;
        let audio: Vec<f32> = (0..n).map(|i| {
            let t = i as f32 / sr;
            0.4 * (2.0*std::f32::consts::PI * 440.0 * t).sin()
              + 0.3 * (2.0*std::f32::consts::PI * 880.0 * t).sin()
        }).collect();

        let t0 = Instant::now();
        let (z, t_lat) = m.encode(&audio);
        eprintln!("encode: {t_lat} latent frames in {:?}", t0.elapsed());
        let t0 = Instant::now();
        let (codes, z_q) = m.quantize(&z, t_lat);
        eprintln!("quantize: codes lens = {:?} in {:?}",
                  codes.iter().map(|c| c.len()).collect::<Vec<_>>(), t0.elapsed());
        let t0 = Instant::now();
        let recon = m.decode(&z_q, t_lat);
        eprintln!("decode: {} samples in {:?}", recon.len(), t0.elapsed());

        let peak = recon.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        let rms_in:  f32 = (audio.iter().map(|x| x*x).sum::<f32>() / audio.len() as f32).sqrt();
        let rms_out: f32 = (recon.iter().map(|x| x*x).sum::<f32>() / recon.len() as f32).sqrt();
        let ref_sig = mag_signature(&audio, 1024, 256);
        let rec_sig = mag_signature(&recon, 1024, 256);
        let corr = pearson(&ref_sig, &rec_sig);
        eprintln!("encodec: peak={peak:.4} rms_in={rms_in:.4} rms_out={rms_out:.4} spec corr={corr:.4}");
        assert!(peak.is_finite() && peak > 0.0, "silent reconstruction");
        assert!(corr > 0.5, "spec correlation too low: {corr}");
    }
}
