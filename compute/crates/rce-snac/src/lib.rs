//! SNAC: Multi-Scale Residual Vector Quantization codec @ 24 kHz, downsample 512×.
//!
//! Encoder:   1 → 48 → 96 → 192 → 384 → 768   (rates [2,4,8,8]; 4 EncoderBlocks)
//! Quantizer: 3 stages with vq_strides [4,2,1], 4096-entry codebooks, dim 8
//! Decoder:   768 → 1024 → 512 → 256 → 128 → 64 → 1   (rates [8,8,4,2])
//!
//! Tensors live under `parametrizations.weight.original0/1` (PyTorch weight_norm:
//! `w = g · v / ‖v‖₂` reducing over all dims except dim 0). NoiseBlocks inject
//! per-frame Gaussian noise on the decoder side — we use a deterministic
//! xorshift PRNG so encode→decode is reproducible.

use rce_weight_loader::SafetensorsFile;
use std::path::Path;
use std::sync::Arc;

pub const SR: u32 = 24000;
pub const LATENT_DIM: usize = 768;
pub const DEC_DIM: usize = 1024;
pub const CODEBOOK_SIZE: usize = 4096;
pub const CODEBOOK_DIM: usize = 8;
pub const VQ_STRIDES: [usize; 3] = [4, 2, 1];
pub const ENC_RATES: [usize; 4] = [2, 4, 8, 8];
pub const DEC_RATES: [usize; 4] = [8, 8, 4, 2];
pub const TOTAL_HOP: usize = 512;
pub const RES_DILATIONS: [usize; 3] = [1, 3, 9];

#[derive(Debug)]
pub enum SnacError { Load(String) }
impl std::fmt::Display for SnacError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::Load(m) => write!(f, "snac load error: {m}") }
    }
}
impl std::error::Error for SnacError {}

fn tf32(t: &rce_weight_loader::LoadedTensor) -> Vec<f32> {
    let bytes = t.data.as_slice();
    match t.dtype.as_str() {
        "f32" => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect(),
        "f16" => {
            debug_assert_eq!(bytes.len() % 2, 0);
            debug_assert_eq!(bytes.as_ptr() as usize % 2, 0);
            let src: &[half::f16] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr().cast::<half::f16>(), bytes.len() / 2)
            };
            rce_op_conv2d::dequant_f16_to_f32(src)
        }
        "bf16" => bytes.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0],c[1]]).to_f32()).collect(),
        other => panic!("snac: unsupported dtype {other}"),
    }
}

fn wn_resolve(g: &[f32], v: &[f32], v_shape: &[usize]) -> Vec<f32> {
    let keep = v_shape[0];
    let inner: usize = v_shape[1..].iter().product();
    debug_assert_eq!(v.len(), keep * inner);
    debug_assert_eq!(g.len(), keep);
    let mut out = vec![0.0f32; v.len()];
    for k in 0..keep {
        let base = k * inner;
        let n2: f32 = v[base..base+inner].iter().map(|x| x*x).sum();
        let s = if n2 > 1e-30 { g[k] / n2.sqrt() } else { 0.0 };
        for i in 0..inner { out[base+i] = v[base+i] * s; }
    }
    out
}

fn snake1d_inplace(x: &mut [f32], alpha: &[f32], t: usize) {
    let c = alpha.len();
    debug_assert_eq!(x.len(), c * t);
    for ci in 0..c {
        let a = alpha[ci];
        let inv = if a.abs() > 1e-12 { 1.0 / a } else { 0.0 };
        for ti in 0..t {
            let v = x[ci*t + ti];
            let s = (a * v).sin();
            x[ci*t + ti] = v + inv * s * s;
        }
    }
}

// 1D conv with zero pad, dilation, stride, groups.  x: [Cin, T], w: [Cout, Cin/g, K], bias: [Cout].
fn conv1d(x: &[f32], cin: usize, t: usize,
          w: &[f32], cout: usize, k: usize,
          groups: usize, stride: usize, dilation: usize, pad: usize,
          bias: Option<&[f32]>) -> (Vec<f32>, usize) {
    let kspan = dilation*(k-1) + 1;
    let t_out = if t + 2*pad < kspan { 0 } else { (t + 2*pad - kspan) / stride + 1 };
    let cin_g = cin / groups;
    let cout_g = cout / groups;
    let mut out = vec![0.0f32; cout * t_out];
    for to in 0..t_out {
        let t_start = to * stride;  // in padded coords
        for g_ in 0..groups {
            for oc in 0..cout_g {
                let oc_g = g_ * cout_g + oc;
                let mut acc = bias.map(|b| b[oc_g]).unwrap_or(0.0);
                for ic in 0..cin_g {
                    let ic_g = g_ * cin_g + ic;
                    for kk in 0..k {
                        let tp = t_start + kk * dilation;
                        if tp < pad { continue; }
                        let xi = tp - pad;
                        if xi >= t { continue; }
                        let w_idx = (oc_g * cin_g + ic) * k + kk;
                        acc += x[ic_g * t + xi] * w[w_idx];
                    }
                }
                out[oc_g * t_out + to] = acc;
            }
        }
    }
    (out, t_out)
}

// 1D transposed conv (zero pad-strip, no groups).  x: [Cin, T], w: [Cin, Cout, K], bias: [Cout].
fn conv_transpose1d(x: &[f32], cin: usize, t: usize,
                    w: &[f32], cout: usize, k: usize,
                    stride: usize, pad: usize,
                    bias: Option<&[f32]>) -> (Vec<f32>, usize) {
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
            let mut v = full[oc * t_out_full + to + pad];
            if let Some(b) = bias { v += b[oc]; }
            out[oc * t_out + to] = v;
        }
    }
    (out, t_out)
}

fn load_conv(sf: &SafetensorsFile, prefix: &str, v_shape_hint: &[usize], with_bias: bool)
    -> Result<(Vec<f32>, Vec<usize>, Vec<f32>), SnacError> {
    let g_name = format!("{prefix}.parametrizations.weight.original0");
    let v_name = format!("{prefix}.parametrizations.weight.original1");
    let g_t = sf.get(&g_name).ok_or_else(|| SnacError::Load(format!("missing {g_name}")))?;
    let v_t = sf.get(&v_name).ok_or_else(|| SnacError::Load(format!("missing {v_name}")))?;
    let g = tf32(g_t);
    let v = tf32(v_t);
    let v_shape: Vec<usize> = v_t.shape.iter().map(|d| *d as usize).collect();
    if v_shape != v_shape_hint {
        return Err(SnacError::Load(format!("{v_name}: shape {:?} != expected {:?}", v_shape, v_shape_hint)));
    }
    let w = wn_resolve(&g, &v, &v_shape);
    let bias = if with_bias {
        let b_name = format!("{prefix}.bias");
        let b_t = sf.get(&b_name).ok_or_else(|| SnacError::Load(format!("missing {b_name}")))?;
        tf32(b_t)
    } else { Vec::new() };
    Ok((w, v_shape, bias))
}

fn load_alpha(sf: &SafetensorsFile, prefix: &str) -> Result<Vec<f32>, SnacError> {
    let name = format!("{prefix}.alpha");
    let t = sf.get(&name).ok_or_else(|| SnacError::Load(format!("missing {name}")))?;
    Ok(tf32(t))
}

struct ResUnit {
    alpha1: Vec<f32>,
    dw_w: Vec<f32>, dw_b: Vec<f32>,
    alpha2: Vec<f32>,
    pw_w: Vec<f32>, pw_b: Vec<f32>,
    dilation: usize,
    c: usize,
}
impl ResUnit {
    fn load(sf: &SafetensorsFile, prefix: &str, c: usize, dilation: usize) -> Result<Self, SnacError> {
        let alpha1 = load_alpha(sf, &format!("{prefix}.block.0"))?;
        let (dw_w, _, dw_b) = load_conv(sf, &format!("{prefix}.block.1"), &[c, 1, 7], true)?;
        let alpha2 = load_alpha(sf, &format!("{prefix}.block.2"))?;
        let (pw_w, _, pw_b) = load_conv(sf, &format!("{prefix}.block.3"), &[c, c, 1], true)?;
        Ok(Self { alpha1, dw_w, dw_b, alpha2, pw_w, pw_b, dilation, c })
    }
    fn forward(&self, x: &mut Vec<f32>, t: &mut usize) {
        let resid = x.clone();
        snake1d_inplace(x, &self.alpha1, *t);
        let pad = 3 * self.dilation;
        let (y, t1) = conv1d(x, self.c, *t, &self.dw_w, self.c, 7,
                             self.c, 1, self.dilation, pad, Some(&self.dw_b));
        debug_assert_eq!(t1, *t);
        *x = y;
        snake1d_inplace(x, &self.alpha2, *t);
        let (y, t2) = conv1d(x, self.c, *t, &self.pw_w, self.c, 1, 1, 1, 1, 0, Some(&self.pw_b));
        debug_assert_eq!(t2, *t);
        *x = y;
        for i in 0..x.len() { x[i] += resid[i]; }
    }
}

struct EncoderBlock {
    ru: [ResUnit; 3],
    snake_alpha: Vec<f32>,
    down_w: Vec<f32>, down_b: Vec<f32>,
    c_in: usize, c_out: usize, stride: usize, k: usize, pad: usize,
}
impl EncoderBlock {
    fn load(sf: &SafetensorsFile, prefix: &str, c_in: usize, c_out: usize, stride: usize) -> Result<Self, SnacError> {
        let ru0 = ResUnit::load(sf, &format!("{prefix}.block.0"), c_in, RES_DILATIONS[0])?;
        let ru1 = ResUnit::load(sf, &format!("{prefix}.block.1"), c_in, RES_DILATIONS[1])?;
        let ru2 = ResUnit::load(sf, &format!("{prefix}.block.2"), c_in, RES_DILATIONS[2])?;
        let snake_alpha = load_alpha(sf, &format!("{prefix}.block.3"))?;
        let k = 2 * stride;
        let pad = stride.div_ceil(2);
        let (down_w, _, down_b) = load_conv(sf, &format!("{prefix}.block.4"), &[c_out, c_in, k], true)?;
        Ok(Self { ru: [ru0, ru1, ru2], snake_alpha, down_w, down_b, c_in, c_out, stride, k, pad })
    }
    fn forward(&self, x: &mut Vec<f32>, t: &mut usize) {
        for ru in &self.ru { ru.forward(x, t); }
        snake1d_inplace(x, &self.snake_alpha, *t);
        let (y, to) = conv1d(x, self.c_in, *t, &self.down_w, self.c_out, self.k,
                             1, self.stride, 1, self.pad, Some(&self.down_b));
        *x = y;
        *t = to;
    }
}

struct NoiseBlock { lin_w: Vec<f32>, c: usize }
impl NoiseBlock {
    fn load(sf: &SafetensorsFile, prefix: &str, c: usize) -> Result<Self, SnacError> {
        let g_name = format!("{prefix}.linear.parametrizations.weight.original0");
        let v_name = format!("{prefix}.linear.parametrizations.weight.original1");
        let g_t = sf.get(&g_name).ok_or_else(|| SnacError::Load(format!("missing {g_name}")))?;
        let v_t = sf.get(&v_name).ok_or_else(|| SnacError::Load(format!("missing {v_name}")))?;
        let g = tf32(g_t); let v = tf32(v_t);
        let v_shape: Vec<usize> = v_t.shape.iter().map(|d| *d as usize).collect();
        if v_shape != vec![c, c, 1] {
            return Err(SnacError::Load(format!("{v_name} shape {:?} != [{c},{c},1]", v_shape)));
        }
        Ok(Self { lin_w: wn_resolve(&g, &v, &v_shape), c })
    }
    fn forward(&self, x: &mut [f32], t: usize, rng: &mut Xorshift) {
        let (h, _) = conv1d(x, self.c, t, &self.lin_w, self.c, 1, 1, 1, 1, 0, None);
        let mut noise = vec![0.0f32; t];
        for ti in 0..t { noise[ti] = rng.gaussian(); }
        for ci in 0..self.c {
            for ti in 0..t {
                x[ci*t + ti] += h[ci*t + ti] * noise[ti];
            }
        }
    }
}

struct DecoderBlock {
    snake_alpha: Vec<f32>,
    up_w: Vec<f32>, up_b: Vec<f32>,
    noise: NoiseBlock,
    ru: [ResUnit; 3],
    c_in: usize, c_out: usize, stride: usize, k: usize, pad: usize,
}
impl DecoderBlock {
    fn load(sf: &SafetensorsFile, prefix: &str, c_in: usize, c_out: usize, stride: usize) -> Result<Self, SnacError> {
        let snake_alpha = load_alpha(sf, &format!("{prefix}.block.0"))?;
        let k = 2 * stride;
        let pad = stride.div_ceil(2);
        let (up_w, _, up_b) = load_conv(sf, &format!("{prefix}.block.1"), &[c_in, c_out, k], true)?;
        let noise = NoiseBlock::load(sf, &format!("{prefix}.block.2"), c_out)?;
        let ru0 = ResUnit::load(sf, &format!("{prefix}.block.3"), c_out, RES_DILATIONS[0])?;
        let ru1 = ResUnit::load(sf, &format!("{prefix}.block.4"), c_out, RES_DILATIONS[1])?;
        let ru2 = ResUnit::load(sf, &format!("{prefix}.block.5"), c_out, RES_DILATIONS[2])?;
        Ok(Self { snake_alpha, up_w, up_b, noise, ru: [ru0, ru1, ru2], c_in, c_out, stride, k, pad })
    }
    fn forward(&self, x: &mut Vec<f32>, t: &mut usize, rng: &mut Xorshift) {
        snake1d_inplace(x, &self.snake_alpha, *t);
        let (y, to) = conv_transpose1d(x, self.c_in, *t, &self.up_w, self.c_out, self.k,
                                       self.stride, self.pad, Some(&self.up_b));
        *x = y; *t = to;
        self.noise.forward(x, *t, rng);
        for ru in &self.ru { ru.forward(x, t); }
    }
}

struct Quantizer {
    stride: usize,
    in_w: Vec<f32>, in_b: Vec<f32>,
    out_w: Vec<f32>, out_b: Vec<f32>,
    codebook: Vec<f32>,
    codebook_norm: Vec<f32>,
}
impl Quantizer {
    fn load(sf: &SafetensorsFile, prefix: &str, stride: usize) -> Result<Self, SnacError> {
        let (in_w, _, in_b)   = load_conv(sf, &format!("{prefix}.in_proj"),
                                          &[CODEBOOK_DIM, LATENT_DIM, 1], true)?;
        let (out_w, _, out_b) = load_conv(sf, &format!("{prefix}.out_proj"),
                                          &[LATENT_DIM, CODEBOOK_DIM, 1], true)?;
        let cb_t = sf.get(&format!("{prefix}.codebook.weight"))
            .ok_or_else(|| SnacError::Load("missing codebook".into()))?;
        let codebook = tf32(cb_t);
        let mut codebook_norm = codebook.clone();
        for c in 0..CODEBOOK_SIZE {
            let base = c * CODEBOOK_DIM;
            let n2: f32 = codebook_norm[base..base+CODEBOOK_DIM].iter().map(|x| x*x).sum();
            let s = if n2 > 1e-30 { 1.0 / n2.sqrt() } else { 0.0 };
            for i in 0..CODEBOOK_DIM { codebook_norm[base+i] *= s; }
        }
        Ok(Self { stride, in_w, in_b, out_w, out_b, codebook, codebook_norm })
    }
    fn forward(&self, z: &[f32], t: usize) -> (Vec<f32>, Vec<u32>) {
        let (z_in, t_in) = if self.stride > 1 {
            let t_in = t / self.stride;
            let mut y = vec![0.0f32; LATENT_DIM * t_in];
            for c in 0..LATENT_DIM {
                for tj in 0..t_in {
                    let mut acc = 0.0;
                    for s in 0..self.stride { acc += z[c*t + tj*self.stride + s]; }
                    y[c*t_in + tj] = acc / self.stride as f32;
                }
            }
            (y, t_in)
        } else {
            (z.to_vec(), t)
        };
        let (z_e, _) = conv1d(&z_in, LATENT_DIM, t_in, &self.in_w, CODEBOOK_DIM, 1, 1, 1, 1, 0, Some(&self.in_b));
        let mut codes = vec![0u32; t_in];
        let mut z_q_e = vec![0.0f32; CODEBOOK_DIM * t_in];
        for tj in 0..t_in {
            let mut vec_n = [0f32; CODEBOOK_DIM];
            let mut n2 = 0.0f32;
            for d in 0..CODEBOOK_DIM { let v = z_e[d*t_in + tj]; vec_n[d] = v; n2 += v*v; }
            let s = if n2 > 1e-30 { 1.0 / n2.sqrt() } else { 0.0 };
            for d in 0..CODEBOOK_DIM { vec_n[d] *= s; }
            let mut best_idx = 0u32;
            let mut best_sim = f32::NEG_INFINITY;
            for ci in 0..CODEBOOK_SIZE {
                let mut dot = 0.0f32;
                for d in 0..CODEBOOK_DIM { dot += vec_n[d] * self.codebook_norm[ci*CODEBOOK_DIM + d]; }
                if dot > best_sim { best_sim = dot; best_idx = ci as u32; }
            }
            codes[tj] = best_idx;
            for d in 0..CODEBOOK_DIM {
                z_q_e[d*t_in + tj] = self.codebook[best_idx as usize * CODEBOOK_DIM + d];
            }
        }
        let (z_q_l, _) = conv1d(&z_q_e, CODEBOOK_DIM, t_in, &self.out_w, LATENT_DIM, 1, 1, 1, 1, 0, Some(&self.out_b));
        let z_q = if self.stride > 1 {
            let mut y = vec![0.0f32; LATENT_DIM * t];
            for c in 0..LATENT_DIM {
                for tj in 0..t_in {
                    for s in 0..self.stride { y[c*t + tj*self.stride + s] = z_q_l[c*t_in + tj]; }
                }
            }
            y
        } else { z_q_l };
        (z_q, codes)
    }
}

pub struct Xorshift { state: u64, spare: Option<f32> }
impl Xorshift {
    pub fn new(seed: u64) -> Self { Self { state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed }, spare: None } }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state; x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.state = x; (x >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u32() >> 8) as f32 + 1.0) / ((1u32 << 24) as f32 + 1.0)
    }
    pub fn gaussian(&mut self) -> f32 {
        if let Some(s) = self.spare.take() { return s; }
        let u1 = self.next_f32();
        let u2 = self.next_f32();
        let mag = (-2.0 * u1.ln()).sqrt();
        let z0 = mag * (2.0 * std::f32::consts::PI * u2).cos();
        let z1 = mag * (2.0 * std::f32::consts::PI * u2).sin();
        self.spare = Some(z1);
        z0
    }
}

pub struct Snac {
    enc_pre_w: Vec<f32>, enc_pre_b: Vec<f32>,
    enc_blocks: Vec<EncoderBlock>,
    enc_post_w: Vec<f32>, enc_post_b: Vec<f32>,
    qs: Vec<Quantizer>,
    dec_pre_w: Vec<f32>, dec_pre_b: Vec<f32>,
    dec_proj_w: Vec<f32>, dec_proj_b: Vec<f32>,
    dec_blocks: Vec<DecoderBlock>,
    dec_final_alpha: Vec<f32>,
    dec_out_w: Vec<f32>, dec_out_b: Vec<f32>,
}

impl Snac {
    pub fn load(safetensors: &Path) -> Result<Arc<Self>, SnacError> {
        let sf = SafetensorsFile::load(safetensors).map_err(|e| SnacError::Load(e.to_string()))?;

        let (enc_pre_w, _, enc_pre_b) = load_conv(&sf, "encoder.block.0", &[48, 1, 7], true)?;
        let enc_blocks = vec![
            EncoderBlock::load(&sf, "encoder.block.1", 48, 96, ENC_RATES[0])?,
            EncoderBlock::load(&sf, "encoder.block.2", 96, 192, ENC_RATES[1])?,
            EncoderBlock::load(&sf, "encoder.block.3", 192, 384, ENC_RATES[2])?,
            EncoderBlock::load(&sf, "encoder.block.4", 384, 768, ENC_RATES[3])?,
        ];
        let (enc_post_w, _, enc_post_b) = load_conv(&sf, "encoder.block.5", &[768, 1, 7], true)?;

        let mut qs = Vec::with_capacity(3);
        for (i, s) in VQ_STRIDES.iter().enumerate() {
            qs.push(Quantizer::load(&sf, &format!("quantizer.quantizers.{i}"), *s)?);
        }

        let (dec_pre_w, _, dec_pre_b)   = load_conv(&sf, "decoder.model.0", &[768, 1, 7], true)?;
        let (dec_proj_w, _, dec_proj_b) = load_conv(&sf, "decoder.model.1", &[1024, 768, 1], true)?;
        let dec_blocks = vec![
            DecoderBlock::load(&sf, "decoder.model.2", 1024, 512, DEC_RATES[0])?,
            DecoderBlock::load(&sf, "decoder.model.3", 512,  256, DEC_RATES[1])?,
            DecoderBlock::load(&sf, "decoder.model.4", 256,  128, DEC_RATES[2])?,
            DecoderBlock::load(&sf, "decoder.model.5", 128,   64, DEC_RATES[3])?,
        ];
        let dec_final_alpha = load_alpha(&sf, "decoder.model.6")?;
        let (dec_out_w, _, dec_out_b) = load_conv(&sf, "decoder.model.7", &[1, 64, 7], true)?;

        Ok(Arc::new(Self {
            enc_pre_w, enc_pre_b, enc_blocks, enc_post_w, enc_post_b,
            qs,
            dec_pre_w, dec_pre_b, dec_proj_w, dec_proj_b, dec_blocks, dec_final_alpha,
            dec_out_w, dec_out_b,
        }))
    }

    pub fn encode(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        let mut x: Vec<f32> = audio.to_vec();
        let mut t = audio.len();
        let (y, t1) = conv1d(&x, 1, t, &self.enc_pre_w, 48, 7, 1, 1, 1, 3, Some(&self.enc_pre_b));
        x = y; t = t1;
        for blk in &self.enc_blocks { blk.forward(&mut x, &mut t); }
        let (y, t1) = conv1d(&x, 768, t, &self.enc_post_w, 768, 7, 768, 1, 1, 3, Some(&self.enc_post_b));
        x = y; t = t1;
        (x, t)
    }

    pub fn quantize(&self, z: &[f32], t: usize) -> (Vec<f32>, Vec<Vec<u32>>) {
        let mut residual = z.to_vec();
        let mut z_q_sum = vec![0.0f32; LATENT_DIM * t];
        let mut codes = Vec::with_capacity(3);
        for q in &self.qs {
            let (z_q_i, c_i) = q.forward(&residual, t);
            for i in 0..z_q_sum.len() { z_q_sum[i] += z_q_i[i]; }
            for i in 0..residual.len() { residual[i] -= z_q_i[i]; }
            codes.push(c_i);
        }
        (z_q_sum, codes)
    }

    pub fn decode(&self, z_q: &[f32], t: usize, seed: u64) -> Vec<f32> {
        let mut rng = Xorshift::new(seed);
        let mut x = z_q.to_vec();
        let mut t = t;
        let (y, t1) = conv1d(&x, 768, t, &self.dec_pre_w, 768, 7, 768, 1, 1, 3, Some(&self.dec_pre_b));
        x = y; t = t1;
        let (y, t1) = conv1d(&x, 768, t, &self.dec_proj_w, 1024, 1, 1, 1, 1, 0, Some(&self.dec_proj_b));
        x = y; t = t1;
        for blk in &self.dec_blocks { blk.forward(&mut x, &mut t, &mut rng); }
        snake1d_inplace(&mut x, &self.dec_final_alpha, t);
        let (y, _) = conv1d(&x, 64, t, &self.dec_out_w, 1, 7, 1, 1, 1, 3, Some(&self.dec_out_b));
        x = y;
        for v in x.iter_mut() { *v = v.tanh(); }
        x
    }

    pub fn roundtrip(&self, audio: &[f32], seed: u64) -> (Vec<f32>, Vec<Vec<u32>>) {
        let (z, t) = self.encode(audio);
        let (z_q, codes) = self.quantize(&z, t);
        (self.decode(&z_q, t, seed), codes)
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
            for b in 0..n_bin {
                sig[f*n_bin + b] = (buf[b].norm_sqr() + 1e-8).ln();
            }
        }
        sig
    }

    fn safetensors_path() -> Option<std::path::PathBuf> {
        let p = std::path::PathBuf::from("/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/snac-24khz/model.safetensors");
        if p.exists() { Some(p) } else { None }
    }

    #[test]
    fn snac_roundtrip() {
        let Some(p) = safetensors_path() else {
            eprintln!("skip: snac safetensors not present"); return;
        };
        eprintln!("loading {}…", p.display());
        let t0 = Instant::now();
        let snac = Snac::load(&p).expect("load");
        eprintln!("loaded in {:?}", t0.elapsed());

        let sr = SR as f32;
        let n = 12 * 2048;   // 24576 samples (1.024 s); divisible by 512 * max_vq_stride
        let audio: Vec<f32> = (0..n).map(|i| {
            let t = i as f32 / sr;
            0.4 * (2.0*std::f32::consts::PI * 440.0 * t).sin()
              + 0.3 * (2.0*std::f32::consts::PI * 880.0 * t).sin()
        }).collect();

        let t0 = Instant::now();
        let (z, tl) = snac.encode(&audio);
        eprintln!("encode: {} latent frames in {:?}", tl, t0.elapsed());
        assert_eq!(z.len(), LATENT_DIM * tl);

        let t0 = Instant::now();
        let (z_q, codes) = snac.quantize(&z, tl);
        eprintln!("quantize: codes lens = [{}, {}, {}] in {:?}",
                  codes[0].len(), codes[1].len(), codes[2].len(), t0.elapsed());
        assert_eq!(codes[0].len(), tl / 4);
        assert_eq!(codes[1].len(), tl / 2);
        assert_eq!(codes[2].len(), tl);
        for cs in &codes { for c in cs { assert!((*c as usize) < CODEBOOK_SIZE); } }

        let t0 = Instant::now();
        let recon = snac.decode(&z_q, tl, 0xC0FFEE);
        eprintln!("decode: {} samples in {:?}", recon.len(), t0.elapsed());
        assert_eq!(recon.len(), n);

        let ref_sig = mag_signature(&audio, 1024, 256);
        let rec_sig = mag_signature(&recon, 1024, 256);
        let corr = pearson(&ref_sig, &rec_sig);
        let peak = recon.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        let rms_in:  f32 = (audio.iter().map(|x| x*x).sum::<f32>() / n as f32).sqrt();
        let rms_out: f32 = (recon.iter().map(|x| x*x).sum::<f32>() / n as f32).sqrt();
        eprintln!("snac: peak={:.4}  rms_in={:.4} rms_out={:.4}  spec corr={:.4}",
                  peak, rms_in, rms_out, corr);
        assert!(peak.is_finite() && peak > 0.0, "silent reconstruction");
        assert!(corr > 0.6, "spectral correlation too low: {corr}");
    }
}
