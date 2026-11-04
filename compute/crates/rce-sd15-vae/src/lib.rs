//! rce wiring for the Stable Diffusion 1.5 AutoencoderKL VAE.
//!
//! Architecture (from `config.json`):
//! - `in_channels = 3` (RGB pixels), `out_channels = 3`, `latent_channels = 4`.
//! - Encoder: `conv_in → 4 down_blocks → mid_block → conv_norm_out → conv_out`.
//!   Down blocks have `layers_per_block = 2` ResnetBlock2D each; blocks 0-2
//!   have a stride-2 downsampler (conv k=3 s=2 padding=1); block 3 has none.
//!   Output channels: 128 → 256 → 512 → 512.  Each downsample halves H,W
//!   so the final latent is at 1/8 spatial resolution.
//! - Mid block (encoder + decoder): 2 ResnetBlocks + 1 self-attention block
//!   (GroupNorm + q/k/v/proj_attn Linears).
//! - Decoder mirrors encoder: `conv_in → mid_block → 4 up_blocks → conv_norm_out
//!   → conv_out`.  Up blocks have `layers_per_block + 1 = 3` ResnetBlocks each;
//!   blocks 0-2 have an upsampler (nearest 2× + conv k=3 s=1 p=1); block 3 has
//!   none.  Output channels: 512 → 512 → 256 → 128.
//! - **KL bottleneck**: encoder output is `[8, h/8, w/8]` (mean + logvar);
//!   `quant_conv 1×1` projects 8→8 (kept for compatibility), then sampling
//!   gives `[4, h/8, w/8]`; `post_quant_conv 1×1` projects 4→4 in the decoder.
//! - **Activations**: SiLU.  **Norm**: GroupNorm with `num_groups = 32`.
//! - All weights ship as f32 (~335 MB safetensors after converting from
//!   the original `.bin` PyTorch pickle).  Borrow-direct via `F32Slice`.

use std::path::Path;
use rce_weight_loader::{F32Slice, LoadedTensor, SafetensorsFile};

#[derive(Debug)]
pub enum SdVaeError { Load(String) }
impl std::fmt::Display for SdVaeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::Load(s) => write!(f, "sd15-vae: {s}") }
    }
}
impl std::error::Error for SdVaeError {}

pub const IN_CHANNELS: usize = 3;
pub const OUT_CHANNELS: usize = 3;
pub const LATENT_CHANNELS: usize = 4;
/// Channel widths in encoder order (decoder reverses).
pub const BLOCK_CHANNELS: [usize; 4] = [128, 256, 512, 512];
pub const N_DOWN_BLOCKS: usize = 4;
pub const N_UP_BLOCKS: usize = 4;
pub const LAYERS_PER_DOWN_BLOCK: usize = 2;
pub const LAYERS_PER_UP_BLOCK: usize = 3; // up = layers_per_block + 1
pub const N_NORM_GROUPS: usize = 32;
pub const SAMPLE_SIZE: usize = 512;

fn lf(sf: &SafetensorsFile, name: &str) -> Result<F32Slice, SdVaeError> {
    let t: &LoadedTensor = sf.get(name)
        .ok_or_else(|| SdVaeError::Load(format!("missing {name}")))?;
    F32Slice::from_tensor(name, t).map_err(SdVaeError::Load)
}
fn lf_vec(sf: &SafetensorsFile, name: &str) -> Result<Vec<f32>, SdVaeError> {
    Ok(lf(sf, name)?.as_slice().to_vec())
}
fn lf_opt(sf: &SafetensorsFile, name: &str) -> Option<F32Slice> {
    sf.get(name).and_then(|t| F32Slice::from_tensor(name, t).ok())
}
fn lf_opt_vec(sf: &SafetensorsFile, name: &str) -> Option<Vec<f32>> {
    sf.get(name).map(|t| t.as_f32().unwrap_or(&[]).to_vec()).filter(|v| !v.is_empty())
}

/// One ResnetBlock2D: GroupNorm → SiLU → Conv3×3 → GroupNorm → SiLU → Conv3×3 + (skip).
/// Skip path is identity when `in_ch == out_ch`, otherwise a 1×1 `conv_shortcut`.
pub struct ResnetBlock {
    pub in_ch: usize, pub out_ch: usize,
    pub norm1_weight: Vec<f32>, pub norm1_bias: Vec<f32>,
    pub conv1_w: F32Slice, pub conv1_b: Vec<f32>,
    pub norm2_weight: Vec<f32>, pub norm2_bias: Vec<f32>,
    pub conv2_w: F32Slice, pub conv2_b: Vec<f32>,
    pub conv_shortcut_w: Option<F32Slice>,
    pub conv_shortcut_b: Option<Vec<f32>>,
}

impl ResnetBlock {
    fn load(sf: &SafetensorsFile, prefix: &str, in_ch: usize, out_ch: usize) -> Result<Self, SdVaeError> {
        let norm1_weight = lf_vec(sf, &format!("{prefix}.norm1.weight"))?;
        let norm1_bias = lf_vec(sf, &format!("{prefix}.norm1.bias"))?;
        let conv1_w = lf(sf, &format!("{prefix}.conv1.weight"))?;
        let conv1_b = lf_vec(sf, &format!("{prefix}.conv1.bias"))?;
        let norm2_weight = lf_vec(sf, &format!("{prefix}.norm2.weight"))?;
        let norm2_bias = lf_vec(sf, &format!("{prefix}.norm2.bias"))?;
        let conv2_w = lf(sf, &format!("{prefix}.conv2.weight"))?;
        let conv2_b = lf_vec(sf, &format!("{prefix}.conv2.bias"))?;
        let conv_shortcut_w = lf_opt(sf, &format!("{prefix}.conv_shortcut.weight"));
        let conv_shortcut_b = lf_opt_vec(sf, &format!("{prefix}.conv_shortcut.bias"));
        if conv1_w.len() != out_ch * in_ch * 3 * 3 {
            return Err(SdVaeError::Load(format!(
                "{prefix}.conv1.weight: expected {} ({}×{}×3×3), got {}",
                out_ch * in_ch * 3 * 3, out_ch, in_ch, conv1_w.len())));
        }
        if conv2_w.len() != out_ch * out_ch * 3 * 3 {
            return Err(SdVaeError::Load(format!(
                "{prefix}.conv2.weight: expected {}, got {}", out_ch*out_ch*9, conv2_w.len())));
        }
        if (in_ch != out_ch) && conv_shortcut_w.is_none() {
            return Err(SdVaeError::Load(format!(
                "{prefix}: in_ch={in_ch} != out_ch={out_ch} but no conv_shortcut.weight")));
        }
        Ok(Self {
            in_ch, out_ch,
            norm1_weight, norm1_bias, conv1_w, conv1_b,
            norm2_weight, norm2_bias, conv2_w, conv2_b,
            conv_shortcut_w, conv_shortcut_b,
        })
    }
}

/// Self-attention block inside the mid_block: GroupNorm → Linear q/k/v →
/// SDPA → proj_attn → residual.  No multi-head specified in SD 1.5 (single
/// "head" of dim = full channel count).
pub struct AttnBlock {
    pub group_norm_weight: Vec<f32>, pub group_norm_bias: Vec<f32>,
    pub q_w: F32Slice, pub q_b: Vec<f32>,
    pub k_w: F32Slice, pub k_b: Vec<f32>,
    pub v_w: F32Slice, pub v_b: Vec<f32>,
    pub proj_attn_w: F32Slice, pub proj_attn_b: Vec<f32>,
}

impl AttnBlock {
    fn load(sf: &SafetensorsFile, prefix: &str, ch: usize) -> Result<Self, SdVaeError> {
        let load_proj = |slot: &str| -> Result<(F32Slice, Vec<f32>), SdVaeError> {
            let w = lf(sf, &format!("{prefix}.{slot}.weight"))?;
            let b = lf_vec(sf, &format!("{prefix}.{slot}.bias"))?;
            if w.len() != ch * ch {
                return Err(SdVaeError::Load(format!(
                    "{prefix}.{slot}.weight: expected {}, got {}", ch*ch, w.len())));
            }
            Ok((w, b))
        };
        let (q_w, q_b) = load_proj("query")?;
        let (k_w, k_b) = load_proj("key")?;
        let (v_w, v_b) = load_proj("value")?;
        let (proj_attn_w, proj_attn_b) = load_proj("proj_attn")?;
        Ok(Self {
            group_norm_weight: lf_vec(sf, &format!("{prefix}.group_norm.weight"))?,
            group_norm_bias:   lf_vec(sf, &format!("{prefix}.group_norm.bias"))?,
            q_w, q_b, k_w, k_b, v_w, v_b,
            proj_attn_w, proj_attn_b,
        })
    }
}

/// One downsampler (encoder): conv k=3 s=2 p=1.
pub struct Downsampler {
    pub conv_w: F32Slice, pub conv_b: Vec<f32>,
    pub in_ch: usize, pub out_ch: usize,
}
impl Downsampler {
    fn load(sf: &SafetensorsFile, prefix: &str, ch: usize) -> Result<Self, SdVaeError> {
        let conv_w = lf(sf, &format!("{prefix}.conv.weight"))?;
        if conv_w.len() != ch * ch * 9 {
            return Err(SdVaeError::Load(format!(
                "{prefix}.conv.weight: expected {} (ch×ch×3×3), got {}", ch*ch*9, conv_w.len())));
        }
        Ok(Self {
            conv_w,
            conv_b: lf_vec(sf, &format!("{prefix}.conv.bias"))?,
            in_ch: ch, out_ch: ch,
        })
    }
}

/// One upsampler (decoder): nearest 2× then conv k=3 s=1 p=1.
pub struct Upsampler {
    pub conv_w: F32Slice, pub conv_b: Vec<f32>,
    pub ch: usize,
}
impl Upsampler {
    fn load(sf: &SafetensorsFile, prefix: &str, ch: usize) -> Result<Self, SdVaeError> {
        let conv_w = lf(sf, &format!("{prefix}.conv.weight"))?;
        if conv_w.len() != ch * ch * 9 {
            return Err(SdVaeError::Load(format!(
                "{prefix}.conv.weight: expected {}, got {}", ch*ch*9, conv_w.len())));
        }
        Ok(Self {
            conv_w,
            conv_b: lf_vec(sf, &format!("{prefix}.conv.bias"))?,
            ch,
        })
    }
}

pub struct DownBlock {
    pub resnets: Vec<ResnetBlock>,
    pub downsampler: Option<Downsampler>,
    pub in_ch: usize, pub out_ch: usize,
}

pub struct UpBlock {
    pub resnets: Vec<ResnetBlock>,
    pub upsampler: Option<Upsampler>,
    pub in_ch: usize, pub out_ch: usize,
}

pub struct MidBlock {
    pub resnet_0: ResnetBlock,
    pub attention: AttnBlock,
    pub resnet_1: ResnetBlock,
}

impl MidBlock {
    fn load(sf: &SafetensorsFile, prefix: &str, ch: usize) -> Result<Self, SdVaeError> {
        Ok(Self {
            resnet_0:   ResnetBlock::load(sf, &format!("{prefix}.resnets.0"), ch, ch)?,
            attention:  AttnBlock::load(sf, &format!("{prefix}.attentions.0"), ch)?,
            resnet_1:   ResnetBlock::load(sf, &format!("{prefix}.resnets.1"), ch, ch)?,
        })
    }
}

pub struct Encoder {
    pub conv_in_w: F32Slice, pub conv_in_b: Vec<f32>,
    pub down_blocks: Vec<DownBlock>,
    pub mid_block: MidBlock,
    pub conv_norm_out_weight: Vec<f32>, pub conv_norm_out_bias: Vec<f32>,
    pub conv_out_w: F32Slice, pub conv_out_b: Vec<f32>,
}

impl Encoder {
    fn load(sf: &SafetensorsFile) -> Result<Self, SdVaeError> {
        let conv_in_w = lf(sf, "encoder.conv_in.weight")?;
        let conv_in_b = lf_vec(sf, "encoder.conv_in.bias")?;
        if conv_in_w.len() != BLOCK_CHANNELS[0] * IN_CHANNELS * 9 {
            return Err(SdVaeError::Load(format!(
                "encoder.conv_in.weight: expected {}, got {}",
                BLOCK_CHANNELS[0]*IN_CHANNELS*9, conv_in_w.len())));
        }
        let mut down_blocks = Vec::with_capacity(N_DOWN_BLOCKS);
        for bi in 0..N_DOWN_BLOCKS {
            // SD VAE quirk: down_block i operates at the output channel width
            // [BLOCK_CHANNELS[i]] except block 0 where in=out=128 since conv_in
            // already widened.  For other blocks, in_ch = BLOCK_CHANNELS[i-1],
            // out_ch = BLOCK_CHANNELS[i], with conv_shortcut on the first resnet.
            let in_ch  = if bi == 0 { BLOCK_CHANNELS[0] } else { BLOCK_CHANNELS[bi - 1] };
            let out_ch = BLOCK_CHANNELS[bi];
            let mut resnets = Vec::with_capacity(LAYERS_PER_DOWN_BLOCK);
            for li in 0..LAYERS_PER_DOWN_BLOCK {
                let res_in = if li == 0 { in_ch } else { out_ch };
                resnets.push(ResnetBlock::load(
                    sf, &format!("encoder.down_blocks.{bi}.resnets.{li}"),
                    res_in, out_ch)?);
            }
            let downsampler = if bi < N_DOWN_BLOCKS - 1 {
                Some(Downsampler::load(
                    sf, &format!("encoder.down_blocks.{bi}.downsamplers.0"), out_ch)?)
            } else { None };
            down_blocks.push(DownBlock { resnets, downsampler, in_ch, out_ch });
        }
        let mid_block = MidBlock::load(sf, "encoder.mid_block", BLOCK_CHANNELS[3])?;
        let conv_norm_out_weight = lf_vec(sf, "encoder.conv_norm_out.weight")?;
        let conv_norm_out_bias   = lf_vec(sf, "encoder.conv_norm_out.bias")?;
        let conv_out_w = lf(sf, "encoder.conv_out.weight")?;
        let conv_out_b = lf_vec(sf, "encoder.conv_out.bias")?;
        if conv_out_w.len() != 2 * LATENT_CHANNELS * BLOCK_CHANNELS[3] * 9 {
            return Err(SdVaeError::Load(format!(
                "encoder.conv_out.weight: expected {} (8×512×3×3), got {}",
                2 * LATENT_CHANNELS * BLOCK_CHANNELS[3] * 9, conv_out_w.len())));
        }
        Ok(Self {
            conv_in_w, conv_in_b,
            down_blocks, mid_block,
            conv_norm_out_weight, conv_norm_out_bias,
            conv_out_w, conv_out_b,
        })
    }
}

pub struct Decoder {
    pub conv_in_w: F32Slice, pub conv_in_b: Vec<f32>,
    pub mid_block: MidBlock,
    pub up_blocks: Vec<UpBlock>,
    pub conv_norm_out_weight: Vec<f32>, pub conv_norm_out_bias: Vec<f32>,
    pub conv_out_w: F32Slice, pub conv_out_b: Vec<f32>,
}

impl Decoder {
    fn load(sf: &SafetensorsFile) -> Result<Self, SdVaeError> {
        let conv_in_w = lf(sf, "decoder.conv_in.weight")?;
        let conv_in_b = lf_vec(sf, "decoder.conv_in.bias")?;
        let mid_block = MidBlock::load(sf, "decoder.mid_block", BLOCK_CHANNELS[3])?;
        let mut up_blocks = Vec::with_capacity(N_UP_BLOCKS);
        // Decoder channel widths reverse: 512, 512, 256, 128 (mirroring encoder's BLOCK_CHANNELS reversed).
        let dec_ch: [usize; 4] = [BLOCK_CHANNELS[3], BLOCK_CHANNELS[3], BLOCK_CHANNELS[1], BLOCK_CHANNELS[0]];
        for bi in 0..N_UP_BLOCKS {
            let in_ch  = if bi == 0 { dec_ch[0] } else { dec_ch[bi - 1] };
            let out_ch = dec_ch[bi];
            let mut resnets = Vec::with_capacity(LAYERS_PER_UP_BLOCK);
            for li in 0..LAYERS_PER_UP_BLOCK {
                let res_in = if li == 0 { in_ch } else { out_ch };
                resnets.push(ResnetBlock::load(
                    sf, &format!("decoder.up_blocks.{bi}.resnets.{li}"),
                    res_in, out_ch)?);
            }
            let upsampler = if bi < N_UP_BLOCKS - 1 {
                Some(Upsampler::load(
                    sf, &format!("decoder.up_blocks.{bi}.upsamplers.0"), out_ch)?)
            } else { None };
            up_blocks.push(UpBlock { resnets, upsampler, in_ch, out_ch });
        }
        let conv_norm_out_weight = lf_vec(sf, "decoder.conv_norm_out.weight")?;
        let conv_norm_out_bias   = lf_vec(sf, "decoder.conv_norm_out.bias")?;
        let conv_out_w = lf(sf, "decoder.conv_out.weight")?;
        let conv_out_b = lf_vec(sf, "decoder.conv_out.bias")?;
        Ok(Self {
            conv_in_w, conv_in_b,
            mid_block, up_blocks,
            conv_norm_out_weight, conv_norm_out_bias,
            conv_out_w, conv_out_b,
        })
    }
}

pub struct Sd15Vae {
    pub encoder: Encoder,
    pub decoder: Decoder,
    pub quant_conv_w: F32Slice, pub quant_conv_b: Vec<f32>,
    pub post_quant_conv_w: F32Slice, pub post_quant_conv_b: Vec<f32>,
}

impl Sd15Vae {
    pub fn load(path: &Path) -> Result<Self, SdVaeError> {
        eprintln!("[sd15-vae] loading {}", path.display());
        let sf = SafetensorsFile::load(path)
            .map_err(|e| SdVaeError::Load(format!("load {}: {e}", path.display())))?;
        let encoder = Encoder::load(&sf)?;
        let decoder = Decoder::load(&sf)?;
        let quant_conv_w = lf(&sf, "quant_conv.weight")?;
        let quant_conv_b = lf_vec(&sf, "quant_conv.bias")?;
        let post_quant_conv_w = lf(&sf, "post_quant_conv.weight")?;
        let post_quant_conv_b = lf_vec(&sf, "post_quant_conv.bias")?;
        Ok(Self {
            encoder, decoder,
            quant_conv_w, quant_conv_b,
            post_quant_conv_w, post_quant_conv_b,
        })
    }

    pub fn user_heap_bytes(&self) -> usize {
        let resnet_bytes = |r: &ResnetBlock| {
            (r.norm1_weight.len() + r.norm1_bias.len() + r.conv1_b.len() +
             r.norm2_weight.len() + r.norm2_bias.len() + r.conv2_b.len() +
             r.conv_shortcut_b.as_ref().map(|v| v.len()).unwrap_or(0)) * 4
        };
        let attn_bytes = |a: &AttnBlock| {
            (a.group_norm_weight.len() + a.group_norm_bias.len() +
             a.q_b.len() + a.k_b.len() + a.v_b.len() + a.proj_attn_b.len()) * 4
        };
        let mid_bytes = |m: &MidBlock| resnet_bytes(&m.resnet_0) + attn_bytes(&m.attention) + resnet_bytes(&m.resnet_1);
        let mut n = self.encoder.conv_in_b.len() * 4;
        n += mid_bytes(&self.encoder.mid_block);
        for b in &self.encoder.down_blocks {
            for r in &b.resnets { n += resnet_bytes(r); }
            if let Some(d) = &b.downsampler { n += d.conv_b.len() * 4; }
        }
        n += (self.encoder.conv_norm_out_weight.len() + self.encoder.conv_norm_out_bias.len()) * 4;
        n += self.encoder.conv_out_b.len() * 4;
        n += self.decoder.conv_in_b.len() * 4 + mid_bytes(&self.decoder.mid_block);
        for b in &self.decoder.up_blocks {
            for r in &b.resnets { n += resnet_bytes(r); }
            if let Some(u) = &b.upsampler { n += u.conv_b.len() * 4; }
        }
        n += (self.decoder.conv_norm_out_weight.len() + self.decoder.conv_norm_out_bias.len()) * 4;
        n += self.decoder.conv_out_b.len() * 4;
        n += (self.quant_conv_b.len() + self.post_quant_conv_b.len()) * 4;
        n
    }
}

// ════════════════════════════════════════════════════════════════════════
// Forward primitives (Stage 1)
// ════════════════════════════════════════════════════════════════════════

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

/// GroupNorm with `N_NORM_GROUPS=32` groups: split `c_out` channels into
/// 32 groups, normalize within each group across `(group_size, H, W)`,
/// then apply per-channel learned scale `gamma` and bias `beta`.
pub fn group_norm32(x: &mut [f32], c: usize, h: usize, w: usize, gamma: &[f32], beta: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), c * h * w);
    debug_assert_eq!(gamma.len(), c);
    debug_assert_eq!(beta.len(), c);
    let g = N_NORM_GROUPS;
    debug_assert!(c % g == 0, "c={c} must be divisible by num_groups={g}");
    let gs = c / g;
    let plane = h * w;
    let group_n = (gs * plane) as f32;
    for gi in 0..g {
        let start = gi * gs;
        let mut mean = 0.0f32;
        for ci in start..start + gs {
            let row = &x[ci * plane .. (ci + 1) * plane];
            for v in row { mean += *v; }
        }
        mean /= group_n;
        let mut var = 0.0f32;
        for ci in start..start + gs {
            let row = &x[ci * plane .. (ci + 1) * plane];
            for v in row { let d = *v - mean; var += d * d; }
        }
        let inv = (var / group_n + eps).sqrt().recip();
        for ci in start..start + gs {
            let g_c = gamma[ci];
            let b_c = beta[ci];
            let row = &mut x[ci * plane .. (ci + 1) * plane];
            for v in row { *v = (*v - mean) * inv * g_c + b_c; }
        }
    }
}

/// 2D conv: kernel size `k`, stride 1, padding `(k-1)/2` (same-pad).
/// Works for k=1 (1×1) and k=3 (3×3 same).  Input `[c_in, h, w]`,
/// weight `[c_out, c_in, k, k]`, bias `[c_out]`.  Returns `[c_out, h, w]`.
pub fn conv2d_same(
    input: &[f32], c_in: usize, h: usize, w: usize,
    weight: &[f32], bias: &[f32], c_out: usize, k: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    debug_assert_eq!(input.len(), c_in * h * w);
    debug_assert_eq!(weight.len(), c_out * c_in * k * k);
    debug_assert_eq!(bias.len(), c_out);
    let pad = k / 2; // 0 for k=1, 1 for k=3
    let mut out = vec![0.0f32; c_out * h * w];
    let plane = h * w;
    out.par_chunks_mut(plane).enumerate().for_each(|(oc, dst)| {
        let w_off_oc = oc * c_in * k * k;
        for y in 0..h {
            for x in 0..w {
                let mut acc = bias[oc];
                for ic in 0..c_in {
                    for ky in 0..k {
                        let iy = (y + ky).wrapping_sub(pad);
                        if iy >= h { continue; }
                        for kx in 0..k {
                            let ix = (x + kx).wrapping_sub(pad);
                            if ix >= w { continue; }
                            let w_idx = w_off_oc + (ic * k + ky) * k + kx;
                            let in_idx = ic * plane + iy * w + ix;
                            acc += weight[w_idx] * input[in_idx];
                        }
                    }
                }
                dst[y * w + x] = acc;
            }
        }
    });
    out
}

/// Nearest-neighbor 2× upsample then `conv2d_same` k=3 — the diffusers
/// `Upsample2D` block.  Input `[c, h, w]` → output `[c, 2h, 2w]` after
/// nearest expansion, then conv 3×3 preserves channels.
pub fn upsample_nearest_then_conv(
    input: &[f32], c: usize, h: usize, w: usize,
    conv_w: &[f32], conv_b: &[f32],
) -> Vec<f32> {
    // 1) Nearest expand: replicate each pixel into a 2×2 block.
    let h2 = h * 2;
    let w2 = w * 2;
    let mut upscaled = vec![0.0f32; c * h2 * w2];
    for ci in 0..c {
        for y in 0..h2 {
            for x in 0..w2 {
                let src = ci * h * w + (y / 2) * w + (x / 2);
                upscaled[ci * h2 * w2 + y * w2 + x] = input[src];
            }
        }
    }
    // 2) Conv 3×3 same-pad.
    conv2d_same(&upscaled, c, h2, w2, conv_w, conv_b, c, 3)
}

/// ResnetBlock2D forward (diffusers convention):
/// `h = silu(norm1(x))` → `conv1(h)` → `silu(norm2(.))` → `conv2(.)`,
/// added to skip-path `x` (identity if `in_ch == out_ch`, else `conv_shortcut(x)`).
pub fn resnet_block_forward(
    block: &ResnetBlock, x: &[f32], h: usize, w: usize, eps: f32,
) -> Vec<f32> {
    let plane = h * w;
    debug_assert_eq!(x.len(), block.in_ch * plane);
    // First norm + activation + conv.
    let mut h_buf = x.to_vec();
    group_norm32(&mut h_buf, block.in_ch, h, w, &block.norm1_weight, &block.norm1_bias, eps);
    for v in h_buf.iter_mut() { *v = silu(*v); }
    let mut h_buf = conv2d_same(
        &h_buf, block.in_ch, h, w,
        block.conv1_w.as_slice(), &block.conv1_b, block.out_ch, 3);
    // Second norm + activation + conv.
    group_norm32(&mut h_buf, block.out_ch, h, w, &block.norm2_weight, &block.norm2_bias, eps);
    for v in h_buf.iter_mut() { *v = silu(*v); }
    let h_buf = conv2d_same(
        &h_buf, block.out_ch, h, w,
        block.conv2_w.as_slice(), &block.conv2_b, block.out_ch, 3);
    // Skip path.
    let skip: Vec<f32> = if block.in_ch == block.out_ch {
        x.to_vec()
    } else {
        // 1×1 conv (no padding).
        conv2d_same(
            x, block.in_ch, h, w,
            block.conv_shortcut_w.as_ref().unwrap().as_slice(),
            block.conv_shortcut_b.as_ref().unwrap(),
            block.out_ch, 1)
    };
    let mut out = h_buf;
    for i in 0..out.len() { out[i] += skip[i]; }
    out
}

/// Mid-block self-attention: GroupNorm → flatten HW → q/k/v Linears →
/// SDPA (1/√c scaling, single "head" of dim=c) → proj_attn Linear →
/// reshape back to [c, h, w] → add residual.
pub fn attn_block_forward(
    block: &AttnBlock, x: &[f32], c: usize, h: usize, w: usize, eps: f32,
) -> Vec<f32> {
    use rayon::prelude::*;
    let plane = h * w;
    let mut h_buf = x.to_vec();
    group_norm32(&mut h_buf, c, h, w, &block.group_norm_weight, &block.group_norm_bias, eps);
    // Flatten to [HW, C] for the linear projections.  Layout in rce is
    // [c, h, w] row-major → transpose to [hw, c].
    let mut hw_c = vec![0.0f32; plane * c];
    for ci in 0..c {
        for p in 0..plane {
            hw_c[p * c + ci] = h_buf[ci * plane + p];
        }
    }
    // q/k/v projections [hw, c] → [hw, c]
    let proj = |w: &[f32], b: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; plane * c];
        out.par_chunks_mut(c).enumerate().for_each(|(pi, row)| {
            let in_row = &hw_c[pi * c .. (pi + 1) * c];
            for j in 0..c {
                let w_row = &w[j * c .. (j + 1) * c];
                let mut s = 0.0f32;
                for k in 0..c { s += in_row[k] * w_row[k]; }
                row[j] = s + b[j];
            }
        });
        out
    };
    let q = proj(block.q_w.as_slice(), &block.q_b);
    let k = proj(block.k_w.as_slice(), &block.k_b);
    let v = proj(block.v_w.as_slice(), &block.v_b);
    // SDPA single-head, scale = 1/√c.
    let scale = (c as f32).powf(-0.5);
    let mut attn_out = vec![0.0f32; plane * c];
    attn_out.par_chunks_mut(c).enumerate().for_each(|(qi, dst)| {
        let mut scores = vec![0.0f32; plane];
        let mut row_max = f32::NEG_INFINITY;
        let qi_row = &q[qi * c .. (qi + 1) * c];
        for kj in 0..plane {
            let kj_row = &k[kj * c .. (kj + 1) * c];
            let mut s = 0.0f32;
            for ci in 0..c { s += qi_row[ci] * kj_row[ci]; }
            let sv = s * scale;
            scores[kj] = sv;
            if sv > row_max { row_max = sv; }
        }
        let mut sum = 0.0f32;
        for j in 0..plane {
            let e = (scores[j] - row_max).exp();
            scores[j] = e;
            sum += e;
        }
        let inv = sum.recip();
        for j in 0..plane { scores[j] *= inv; }
        for ci in 0..c {
            let mut acc = 0.0f32;
            for kj in 0..plane {
                acc += scores[kj] * v[kj * c + ci];
            }
            dst[ci] = acc;
        }
    });
    // proj_attn linear [hw, c] → [hw, c]
    let mut out_hw_c = vec![0.0f32; plane * c];
    let pw = block.proj_attn_w.as_slice();
    let pb = &block.proj_attn_b;
    out_hw_c.par_chunks_mut(c).enumerate().for_each(|(pi, row)| {
        let in_row = &attn_out[pi * c .. (pi + 1) * c];
        for j in 0..c {
            let w_row = &pw[j * c .. (j + 1) * c];
            let mut s = 0.0f32;
            for k in 0..c { s += in_row[k] * w_row[k]; }
            row[j] = s + pb[j];
        }
    });
    // Transpose back to [c, h, w] and add residual.
    let mut out = vec![0.0f32; c * plane];
    for ci in 0..c {
        for p in 0..plane {
            out[ci * plane + p] = out_hw_c[p * c + ci] + x[ci * plane + p];
        }
    }
    out
}

/// Diffusers `Downsample2D`: stride-2 conv k=3 with **asymmetric padding**
/// — pad bottom and right by 1 with zeros (top-left = 0), then conv k=3
/// stride=2 with no further padding.  For even `h` and `w`, output is
/// exactly `(h/2, w/2)` channels preserved.  Per `diffusers.models.upsampling.Downsample2D`.
pub fn downsample_stride2_asym_pad(
    input: &[f32], c: usize, h: usize, w: usize,
    weight: &[f32], bias: &[f32],
) -> Vec<f32> {
    use rayon::prelude::*;
    debug_assert_eq!(input.len(), c * h * w);
    debug_assert_eq!(weight.len(), c * c * 9);
    debug_assert_eq!(bias.len(), c);
    debug_assert!(h % 2 == 0 && w % 2 == 0, "expected even h,w; got {h}×{w}");
    let h_out = h / 2;
    let w_out = w / 2;
    let mut out = vec![0.0f32; c * h_out * w_out];
    let plane_in = h * w;
    let plane_out = h_out * w_out;
    out.par_chunks_mut(plane_out).enumerate().for_each(|(oc, dst)| {
        let w_off_oc = oc * c * 9;
        for y_out in 0..h_out {
            for x_out in 0..w_out {
                let mut acc = bias[oc];
                let y_in_base = 2 * y_out as i32;
                let x_in_base = 2 * x_out as i32;
                for ic in 0..c {
                    for ky in 0..3 {
                        let iy = y_in_base + ky as i32; // implicit top-left pad = 0
                        if iy < 0 || iy as usize >= h { continue; }
                        for kx in 0..3 {
                            let ix = x_in_base + kx as i32;
                            if ix < 0 || ix as usize >= w { continue; }
                            let w_idx = w_off_oc + (ic * 3 + ky) * 3 + kx;
                            let in_idx = ic * plane_in + (iy as usize) * w + (ix as usize);
                            acc += weight[w_idx] * input[in_idx];
                        }
                    }
                }
                dst[y_out * w_out + x_out] = acc;
            }
        }
    });
    out
}

fn mid_block_forward(m: &MidBlock, x: Vec<f32>, c: usize, h: usize, w: usize, eps: f32) -> Vec<f32> {
    let x = resnet_block_forward(&m.resnet_0, &x, h, w, eps);
    let x = attn_block_forward(&m.attention, &x, c, h, w, eps);
    resnet_block_forward(&m.resnet_1, &x, h, w, eps)
}

impl Sd15Vae {
    /// Encode an RGB image `[3, h, w]` (h, w divisible by 8) into the
    /// 8-channel pre-sample tensor `[8, h/8, w/8]` carrying `[mean | logvar]`
    /// along the channel axis (each 4-channel slice).  Subsequent samplers
    /// take `latent = mean + exp(logvar/2) · randn()`; for deterministic
    /// reproduction, taking `mean` directly is the no-noise variant.
    pub fn encode(&self, image: &[f32], h: usize, w: usize, eps: f32) -> Vec<f32> {
        debug_assert_eq!(image.len(), IN_CHANNELS * h * w);
        assert!(h % 8 == 0 && w % 8 == 0, "encoder requires h,w divisible by 8");
        // 1) conv_in 3×3 (3 → 128)
        let mut x = conv2d_same(
            image, IN_CHANNELS, h, w,
            self.encoder.conv_in_w.as_slice(), &self.encoder.conv_in_b,
            BLOCK_CHANNELS[0], 3);
        let mut ch = BLOCK_CHANNELS[0];
        let mut hh = h;
        let mut ww = w;
        // 2) 4 down_blocks: 2 resnets per block + optional asymmetric-pad stride-2 downsample
        for blk in &self.encoder.down_blocks {
            for res in &blk.resnets {
                x = resnet_block_forward(res, &x, hh, ww, eps);
            }
            ch = blk.out_ch;
            if let Some(down) = &blk.downsampler {
                x = downsample_stride2_asym_pad(
                    &x, ch, hh, ww,
                    down.conv_w.as_slice(), &down.conv_b);
                hh /= 2; ww /= 2;
            }
        }
        // 3) mid_block at 512 ch
        x = mid_block_forward(&self.encoder.mid_block, x, BLOCK_CHANNELS[3], hh, ww, eps);
        // 4) conv_norm_out + SiLU + conv_out 3×3 → 8 channels
        group_norm32(&mut x, BLOCK_CHANNELS[3], hh, ww,
                     &self.encoder.conv_norm_out_weight,
                     &self.encoder.conv_norm_out_bias, eps);
        for v in x.iter_mut() { *v = silu(*v); }
        let x = conv2d_same(
            &x, BLOCK_CHANNELS[3], hh, ww,
            self.encoder.conv_out_w.as_slice(), &self.encoder.conv_out_b,
            2 * LATENT_CHANNELS, 3);
        // 5) quant_conv 1×1 (8 → 8)
        conv2d_same(
            &x, 2 * LATENT_CHANNELS, hh, ww,
            self.quant_conv_w.as_slice(), &self.quant_conv_b,
            2 * LATENT_CHANNELS, 1)
    }

    /// Decode a 4-channel latent `[LATENT_CHANNELS=4, h_lat, w_lat]`
    /// into an RGB image `[3, h_lat·8, w_lat·8]`.
    pub fn decode(&self, latent: &[f32], h_lat: usize, w_lat: usize, eps: f32) -> Vec<f32> {
        debug_assert_eq!(latent.len(), LATENT_CHANNELS * h_lat * w_lat);
        // 1) post_quant_conv 1×1, 4→4
        let x = conv2d_same(
            latent, LATENT_CHANNELS, h_lat, w_lat,
            self.post_quant_conv_w.as_slice(), &self.post_quant_conv_b,
            LATENT_CHANNELS, 1);
        // 2) conv_in 3×3, 4→512
        let mut x = conv2d_same(
            &x, LATENT_CHANNELS, h_lat, w_lat,
            self.decoder.conv_in_w.as_slice(), &self.decoder.conv_in_b,
            BLOCK_CHANNELS[3], 3);
        let mut h = h_lat;
        let mut w = w_lat;
        // 3) mid_block at 512 ch, no spatial change
        x = mid_block_forward(&self.decoder.mid_block, x, BLOCK_CHANNELS[3], h, w, eps);
        // 4) up_blocks: each does 3 resnets then optional upsample
        for blk in &self.decoder.up_blocks {
            for res in &blk.resnets {
                x = resnet_block_forward(res, &x, h, w, eps);
            }
            if let Some(up) = &blk.upsampler {
                x = upsample_nearest_then_conv(&x, blk.out_ch, h, w,
                    up.conv_w.as_slice(), &up.conv_b);
                h *= 2; w *= 2;
            }
        }
        // 5) conv_norm_out + SiLU + conv_out
        let final_ch = BLOCK_CHANNELS[0];
        group_norm32(&mut x, final_ch, h, w,
                     &self.decoder.conv_norm_out_weight,
                     &self.decoder.conv_norm_out_bias, eps);
        for v in x.iter_mut() { *v = silu(*v); }
        conv2d_same(
            &x, final_ch, h, w,
            self.decoder.conv_out_w.as_slice(), &self.decoder.conv_out_b,
            OUT_CHANNELS, 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn sd_vae_path() -> Option<PathBuf> {
        for vol in ["macos_4TB_external", "macos_4TB_external_1", "macos_4TB_external_2"] {
            let p = PathBuf::from(format!(
                "/Volumes/{vol}/neuronexus_ai/model_library/stable-diffusion-v1-5/vae/diffusion_pytorch_model.safetensors"));
            if p.exists() { return Some(p); }
        }
        None
    }

    /// Stage 0: load all 248 tensors, verify the SD 1.5 VAE architecture
    /// (4 down-blocks of 2 resnets + 3 downsamplers; mirror decoder of 4 up-blocks
    /// of 3 resnets + 3 upsamplers; mid blocks with self-attention; 8↔4 KL bottleneck).
    /// Borrow-direct keeps user heap small.
    #[test]
    #[ignore = "sd15-vae Stage 0 — needs SD 1.5 VAE safetensors mounted"]
    fn sd15_vae_load_full() {
        let Some(path) = sd_vae_path() else { eprintln!("SD VAE not mounted"); return; };
        let t0 = std::time::Instant::now();
        let m = Sd15Vae::load(&path).expect("load");
        eprintln!("sd15-vae: loaded full encoder + decoder in {:?}", t0.elapsed());

        assert_eq!(m.encoder.down_blocks.len(), N_DOWN_BLOCKS);
        assert_eq!(m.decoder.up_blocks.len(), N_UP_BLOCKS);
        // Encoder: blocks 0,1,2 have downsamplers; block 3 does not.
        for (bi, blk) in m.encoder.down_blocks.iter().enumerate() {
            assert_eq!(blk.resnets.len(), LAYERS_PER_DOWN_BLOCK, "down block {bi} resnets");
            assert_eq!(blk.downsampler.is_some(), bi < N_DOWN_BLOCKS - 1,
                       "down block {bi} downsampler presence");
        }
        // Decoder: blocks 0,1,2 have upsamplers; block 3 does not.
        for (bi, blk) in m.decoder.up_blocks.iter().enumerate() {
            assert_eq!(blk.resnets.len(), LAYERS_PER_UP_BLOCK, "up block {bi} resnets");
            assert_eq!(blk.upsampler.is_some(), bi < N_UP_BLOCKS - 1,
                       "up block {bi} upsampler presence");
        }
        // Self-attention in mid-blocks (single-head 512×512).
        assert_eq!(m.encoder.mid_block.attention.q_w.len(), 512 * 512);
        assert_eq!(m.decoder.mid_block.attention.q_w.len(), 512 * 512);
        // KL bottleneck: encoder out is 8 channels, decoder in is 4.
        assert_eq!(m.encoder.conv_out_b.len(), 2 * LATENT_CHANNELS);
        assert_eq!(m.decoder.conv_in_b.len(), BLOCK_CHANNELS[3]);
        assert_eq!(m.quant_conv_w.len(), 8 * 8);    // 1×1 conv
        assert_eq!(m.post_quant_conv_w.len(), 4 * 4);

        let mb = m.user_heap_bytes() as f64 / (1024.0 * 1024.0);
        eprintln!("sd15-vae user heap (borrow-direct): {:.3} MB (full 248 tensors)", mb);
        assert!(mb < 5.0, "borrow-direct should keep heap tiny; got {mb} MB");
        eprintln!("sd15-vae Stage 0 ✓");
    }

    /// Stage 1 decode smoke: 4×8×8 synthetic latent → 3×64×64 RGB image.
    /// Verifies post_quant_conv + decoder conv_in + mid-block self-attn
    /// + 4 up-blocks (12 resnets + 3 nearest+conv upsamples) + conv_norm
    /// + conv_out all compose into a finite output of the expected shape.
    /// Uses a small latent so 4 doublings stay tractable (8 → 64 px).
    #[test]
    #[ignore = "sd15-vae Stage 1 — needs SD VAE safetensors mounted"]
    fn sd15_vae_decode_smoke() {
        let Some(path) = sd_vae_path() else { eprintln!("SD VAE not mounted"); return; };
        let m = Sd15Vae::load(&path).expect("load");
        let h_lat = 8;
        let w_lat = 8;
        let latent: Vec<f32> = (0..LATENT_CHANNELS * h_lat * w_lat)
            .map(|i| ((i as f32) * 0.07).sin() * 0.5).collect();
        let t0 = std::time::Instant::now();
        let img = m.decode(&latent, h_lat, w_lat, 1e-6);
        let dt = t0.elapsed();
        let expected = OUT_CHANNELS * h_lat * 8 * w_lat * 8;
        assert_eq!(img.len(), expected,
                   "expected {} = {OUT_CHANNELS}·{}·{} got {}", expected, h_lat*8, w_lat*8, img.len());
        assert!(img.iter().all(|v| v.is_finite()));
        let max_abs: f32 = img.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let mean_abs: f32 = img.iter().map(|v| v.abs()).sum::<f32>() / (img.len() as f32);
        eprintln!("sd15-vae decode 4×{h_lat}×{w_lat} → 3×{}×{}: {dt:?} max_abs={max_abs:.4} mean_abs={mean_abs:.4}",
                  h_lat*8, w_lat*8);
        assert!(max_abs > 0.0, "decode should produce nonzero output");
        eprintln!("sd15-vae Stage 1 decode ✓");
    }

    /// Stage 2 encode smoke + Python-reference numerical agreement.
    /// Captures `vae.quant_conv(vae.encoder(image))` from official diffusers
    /// `AutoencoderKL` on a deterministic 3×64×64 image → 8×8×8 moments.
    #[test]
    #[ignore = "sd15-vae encoder + ref — needs SD VAE safetensors + /tmp/sd_vae_ref_encode.f32.bin"]
    fn sd15_vae_encoder_vs_python_reference() {
        let Some(path) = sd_vae_path() else { eprintln!("SD VAE not mounted"); return; };
        let ref_path = std::path::PathBuf::from("/tmp/sd_vae_ref_encode.f32.bin");
        if !ref_path.exists() { eprintln!("ref not found"); return; }
        let ref_bytes = std::fs::read(&ref_path).expect("read ref");
        let ref_data: Vec<f32> = ref_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(ref_data.len(), 8 * 8 * 8);
        let ref_max: f32 = ref_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("Python ref: ‖.‖={:.4} max_abs={ref_max:.4}",
                  ref_data.iter().map(|v| v*v).sum::<f32>().sqrt());

        let m = Sd15Vae::load(&path).expect("load");
        let h = 64; let w = 64;
        let image: Vec<f32> = (0..3 * h * w)
            .map(|i| ((i as f32) * 0.0009).sin() * 0.5).collect();
        let t0 = std::time::Instant::now();
        let moments = m.encode(&image, h, w, 1e-6);
        eprintln!("rce encode 3×{h}×{w} → 8×{}×{}: {:?}", h/8, w/8, t0.elapsed());
        assert_eq!(moments.len(), ref_data.len(), "shape mismatch");

        let mut max_abs_diff = 0.0f32;
        let mut sum_sq = 0.0f32;
        for (a, b) in moments.iter().zip(ref_data.iter()) {
            let d = (a - b).abs();
            if d > max_abs_diff { max_abs_diff = d; }
            sum_sq += d * d;
        }
        let rms = (sum_sq / (moments.len() as f32)).sqrt();
        let rel = max_abs_diff / ref_max;
        eprintln!("vs Python ref: max_abs_diff={max_abs_diff:.6} rms={rms:.6} rel={:.2}%", rel * 100.0);
        assert!(rel < 0.01, "encoder should match diffusers ref to f32 noise; got rel={}%", rel * 100.0);
    }

    /// Full-decoder Python-reference numerical agreement.  Reference
    /// captured 2026-06-06 from diffusers `AutoencoderKL.from_pretrained`
    /// on the same deterministic 4×8×8 latent → 3×64×64 image.
    #[test]
    #[ignore = "sd15-vae decoder ref — needs SD VAE safetensors + /tmp/sd_vae_ref_decode.f32.bin"]
    fn sd15_vae_decoder_vs_python_reference() {
        let Some(path) = sd_vae_path() else { eprintln!("SD VAE not mounted"); return; };
        let ref_path = std::path::PathBuf::from("/tmp/sd_vae_ref_decode.f32.bin");
        if !ref_path.exists() { eprintln!("ref not found"); return; }
        let ref_bytes = std::fs::read(&ref_path).expect("read ref");
        let ref_data: Vec<f32> = ref_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(ref_data.len(), 3 * 64 * 64);
        let ref_norm: f32 = ref_data.iter().map(|v| v*v).sum::<f32>().sqrt();
        let ref_max: f32 = ref_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("Python ref: ‖.‖={ref_norm:.4} max_abs={ref_max:.4}");

        let m = Sd15Vae::load(&path).expect("load");
        let h_lat = 8; let w_lat = 8;
        let latent: Vec<f32> = (0..LATENT_CHANNELS * h_lat * w_lat)
            .map(|i| ((i as f32) * 0.07).sin() * 0.5).collect();
        let img = m.decode(&latent, h_lat, w_lat, 1e-6);
        assert_eq!(img.len(), ref_data.len());

        let mut max_abs_diff = 0.0f32;
        let mut sum_sq = 0.0f32;
        for (a, b) in img.iter().zip(ref_data.iter()) {
            let d = (a - b).abs();
            if d > max_abs_diff { max_abs_diff = d; }
            sum_sq += d * d;
        }
        let rms = (sum_sq / (img.len() as f32)).sqrt();
        let rel = max_abs_diff / ref_max;
        eprintln!("vs Python ref: max_abs_diff={max_abs_diff:.6} rms={rms:.6} rel={:.2}%", rel * 100.0);
        assert!(rel < 0.01, "SD VAE decoder should match Python ref to f32 noise; got rel={}%", rel * 100.0);
    }
}
