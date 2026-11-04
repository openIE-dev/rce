//! rce wiring for the Stable Diffusion 1.5 UNet2DConditionModel.
//!
//! Architecture (from `unet/config.json` and the 686-tensor checkpoint):
//! - **Latent diffusion**: `in_channels=4` (matches SD VAE latent),
//!   `out_channels=4` (predicted velocity / epsilon), `sample_size=64`.
//! - **Cross-attention**: `cross_attention_dim=768` (matches `rce-clip-vit-l::TextTower`
//!   output post text-projection — actually pre-projection, the last_hidden_state
//!   which is 768-d in CLIP-L).
//! - **Block channel widths**: `[320, 640, 1280, 1280]` at the 4 resolution levels.
//! - **Head dim**: `attention_head_dim = 8` (so n_heads = ch / 8 at each level).
//! - **Time embedding**: sinusoidal at `block_out_channels[0] = 320` →
//!   `linear_1 (320→1280)` + SiLU + `linear_2 (1280→1280)`.
//! - **Encoder side** (downsampling):
//!   - 3 `CrossAttnDownBlock2D` (each: 2 time-resnets + 2 transformer-2d + downsampler)
//!   - 1 `DownBlock2D` (2 time-resnets only, no attentions, no downsampler at the bottom)
//! - **Mid block** (`UNetMidBlock2DCrossAttn`): 1 time-resnet → 1 transformer-2d → 1 time-resnet
//! - **Decoder side** (upsampling, mirror):
//!   - 1 `UpBlock2D` (3 time-resnets + upsampler)
//!   - 3 `CrossAttnUpBlock2D` (each: 3 time-resnets + 3 transformer-2d + optional upsampler)
//! - **ResnetBlock2DTimeCond**: GroupNorm → SiLU → Conv3×3 → `time_emb_proj (1280→out_ch)`
//!   added per-channel → GroupNorm → SiLU → Conv3×3 + skip (identity or 1×1 conv_shortcut).
//! - **Transformer2DBlock** (a.k.a. `SpatialTransformer`):
//!   GroupNorm → `proj_in` (1×1 conv) → flatten HW → `N=1` `BasicTransformerBlock`:
//!     - `norm1 + attn1` (self-attention: q/k/v Linear, multi-head SDPA, `to_out.0`)
//!     - `norm2 + attn2` (cross-attention: Q from x, K/V from `encoder_hidden_states [77, 768]`)
//!     - `norm3 + ff` (GEGLU feedforward: `net.0.proj 1280→10240` (gate+value), `net.2 5120→1280`)
//!   → `proj_out` (1×1 conv) → unflatten + residual.
//!
//! Total: 686 tensors, ~860M parameters, ~3.4 GB at f32.  Borrow-direct via `F32Slice`.

use std::path::Path;
use rce_weight_loader::{F32Slice, LoadedTensor, SafetensorsFile};

#[derive(Debug)]
pub enum UnetError { Load(String) }
impl std::fmt::Display for UnetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::Load(s) => write!(f, "sd15-unet: {s}") }
    }
}
impl std::error::Error for UnetError {}

pub const IN_CHANNELS: usize = 4;
pub const OUT_CHANNELS: usize = 4;
pub const CROSS_ATTN_DIM: usize = 768;
/// SD 1.5's `attention_head_dim: 8` in the config is **legacy convention**:
/// it actually specifies `num_attention_heads = 8`, NOT `head_dim = 8`.
/// The real head_dim is `inner_dim / 8` (so 40 / 80 / 160 / 160 across the
/// 4 block widths).  This caught us out during numerical-reference debugging.
pub const N_ATTN_HEADS: usize = 8;
pub const TIME_EMBED_DIM: usize = 1280;
pub const FREQ_DIM: usize = 320; // = block_out_channels[0]
pub const BLOCK_CHANNELS: [usize; 4] = [320, 640, 1280, 1280];
pub const N_DOWN_BLOCKS: usize = 4;
pub const N_UP_BLOCKS: usize = 4;
pub const LAYERS_PER_DOWN_BLOCK: usize = 2;
pub const LAYERS_PER_UP_BLOCK: usize = 3;
pub const N_NORM_GROUPS: usize = 32;
pub const SAMPLE_SIZE: usize = 64;

fn lf(sf: &SafetensorsFile, name: &str) -> Result<F32Slice, UnetError> {
    let t: &LoadedTensor = sf.get(name)
        .ok_or_else(|| UnetError::Load(format!("missing {name}")))?;
    F32Slice::from_tensor(name, t).map_err(UnetError::Load)
}
fn lf_vec(sf: &SafetensorsFile, name: &str) -> Result<Vec<f32>, UnetError> {
    Ok(lf(sf, name)?.as_slice().to_vec())
}
fn lf_opt(sf: &SafetensorsFile, name: &str) -> Option<F32Slice> {
    sf.get(name).and_then(|t| F32Slice::from_tensor(name, t).ok())
}
fn lf_opt_vec(sf: &SafetensorsFile, name: &str) -> Option<Vec<f32>> {
    sf.get(name).map(|t| t.as_f32().unwrap_or(&[]).to_vec()).filter(|v| !v.is_empty())
}

/// One time-conditioned ResnetBlock2D (SD U-Net flavor): same as the VAE's
/// ResnetBlock plus a Linear-from-time-embedding inserted between conv1
/// and the second norm.
pub struct TimeCondResnet {
    pub in_ch: usize, pub out_ch: usize,
    pub norm1_weight: Vec<f32>, pub norm1_bias: Vec<f32>,
    pub conv1_w: F32Slice, pub conv1_b: Vec<f32>,
    pub time_emb_proj_w: F32Slice, pub time_emb_proj_b: Vec<f32>, // [out_ch, TIME_EMBED_DIM]
    pub norm2_weight: Vec<f32>, pub norm2_bias: Vec<f32>,
    pub conv2_w: F32Slice, pub conv2_b: Vec<f32>,
    pub conv_shortcut_w: Option<F32Slice>,
    pub conv_shortcut_b: Option<Vec<f32>>,
}

impl TimeCondResnet {
    fn load(sf: &SafetensorsFile, prefix: &str, in_ch: usize, out_ch: usize) -> Result<Self, UnetError> {
        let norm1_weight = lf_vec(sf, &format!("{prefix}.norm1.weight"))?;
        let norm1_bias = lf_vec(sf, &format!("{prefix}.norm1.bias"))?;
        let conv1_w = lf(sf, &format!("{prefix}.conv1.weight"))?;
        let conv1_b = lf_vec(sf, &format!("{prefix}.conv1.bias"))?;
        let time_emb_proj_w = lf(sf, &format!("{prefix}.time_emb_proj.weight"))?;
        let time_emb_proj_b = lf_vec(sf, &format!("{prefix}.time_emb_proj.bias"))?;
        let norm2_weight = lf_vec(sf, &format!("{prefix}.norm2.weight"))?;
        let norm2_bias = lf_vec(sf, &format!("{prefix}.norm2.bias"))?;
        let conv2_w = lf(sf, &format!("{prefix}.conv2.weight"))?;
        let conv2_b = lf_vec(sf, &format!("{prefix}.conv2.bias"))?;
        let conv_shortcut_w = lf_opt(sf, &format!("{prefix}.conv_shortcut.weight"));
        let conv_shortcut_b = lf_opt_vec(sf, &format!("{prefix}.conv_shortcut.bias"));
        if conv1_w.len() != out_ch * in_ch * 9 {
            return Err(UnetError::Load(format!(
                "{prefix}.conv1.weight: expected {} ({}×{}×3×3), got {}",
                out_ch*in_ch*9, out_ch, in_ch, conv1_w.len())));
        }
        if conv2_w.len() != out_ch * out_ch * 9 {
            return Err(UnetError::Load(format!(
                "{prefix}.conv2.weight: expected {}, got {}", out_ch*out_ch*9, conv2_w.len())));
        }
        if time_emb_proj_w.len() != out_ch * TIME_EMBED_DIM {
            return Err(UnetError::Load(format!(
                "{prefix}.time_emb_proj.weight: expected {}, got {}",
                out_ch * TIME_EMBED_DIM, time_emb_proj_w.len())));
        }
        if (in_ch != out_ch) && conv_shortcut_w.is_none() {
            return Err(UnetError::Load(format!(
                "{prefix}: in_ch={in_ch} != out_ch={out_ch} but no conv_shortcut.weight")));
        }
        Ok(Self {
            in_ch, out_ch,
            norm1_weight, norm1_bias, conv1_w, conv1_b,
            time_emb_proj_w, time_emb_proj_b,
            norm2_weight, norm2_bias, conv2_w, conv2_b,
            conv_shortcut_w, conv_shortcut_b,
        })
    }
}

/// One `BasicTransformerBlock` inside a `Transformer2DModel`:
/// self-attn → cross-attn → GEGLU feedforward.
pub struct BasicTransformerBlock {
    pub norm1_weight: Vec<f32>, pub norm1_bias: Vec<f32>,
    /// Self-attention `attn1`.
    pub attn1_q: F32Slice, // [dim, dim]
    pub attn1_k: F32Slice,
    pub attn1_v: F32Slice,
    pub attn1_out_w: F32Slice, pub attn1_out_b: Vec<f32>,
    pub norm2_weight: Vec<f32>, pub norm2_bias: Vec<f32>,
    /// Cross-attention `attn2` (q from x, k/v from `encoder_hidden_states`).
    pub attn2_q: F32Slice,        // [dim, dim]
    pub attn2_k: F32Slice,        // [dim, CROSS_ATTN_DIM=768]
    pub attn2_v: F32Slice,        // [dim, CROSS_ATTN_DIM=768]
    pub attn2_out_w: F32Slice, pub attn2_out_b: Vec<f32>,
    pub norm3_weight: Vec<f32>, pub norm3_bias: Vec<f32>,
    /// GEGLU feedforward: `ff.net.0.proj [4·dim, dim]` (gate + value concat),
    /// `ff.net.2 [dim, 2·dim]` (projection back).
    pub ff_net0_proj_w: F32Slice, pub ff_net0_proj_b: Vec<f32>,
    pub ff_net2_w: F32Slice, pub ff_net2_b: Vec<f32>,
}

impl BasicTransformerBlock {
    fn load(sf: &SafetensorsFile, prefix: &str, dim: usize) -> Result<Self, UnetError> {
        let proj = |slot: &str, dout: usize, din: usize| -> Result<F32Slice, UnetError> {
            let w = lf(sf, &format!("{prefix}.{slot}.weight"))?;
            if w.len() != dout * din {
                return Err(UnetError::Load(format!(
                    "{prefix}.{slot}.weight: expected {} (= {dout}×{din}), got {}",
                    dout * din, w.len())));
            }
            Ok(w)
        };
        let proj_b = |slot: &str| lf_vec(sf, &format!("{prefix}.{slot}.bias"));
        Ok(Self {
            norm1_weight: lf_vec(sf, &format!("{prefix}.norm1.weight"))?,
            norm1_bias:   lf_vec(sf, &format!("{prefix}.norm1.bias"))?,
            attn1_q:      proj("attn1.to_q", dim, dim)?,
            attn1_k:      proj("attn1.to_k", dim, dim)?,
            attn1_v:      proj("attn1.to_v", dim, dim)?,
            attn1_out_w:  proj("attn1.to_out.0", dim, dim)?,
            attn1_out_b:  proj_b("attn1.to_out.0")?,
            norm2_weight: lf_vec(sf, &format!("{prefix}.norm2.weight"))?,
            norm2_bias:   lf_vec(sf, &format!("{prefix}.norm2.bias"))?,
            attn2_q:      proj("attn2.to_q", dim, dim)?,
            attn2_k:      proj("attn2.to_k", dim, CROSS_ATTN_DIM)?,
            attn2_v:      proj("attn2.to_v", dim, CROSS_ATTN_DIM)?,
            attn2_out_w:  proj("attn2.to_out.0", dim, dim)?,
            attn2_out_b:  proj_b("attn2.to_out.0")?,
            norm3_weight: lf_vec(sf, &format!("{prefix}.norm3.weight"))?,
            norm3_bias:   lf_vec(sf, &format!("{prefix}.norm3.bias"))?,
            ff_net0_proj_w: proj("ff.net.0.proj", 4 * dim * 2, dim)?, // GEGLU 2× expansion
            ff_net0_proj_b: proj_b("ff.net.0.proj")?,
            ff_net2_w:      proj("ff.net.2", dim, 4 * dim)?,
            ff_net2_b:      proj_b("ff.net.2")?,
        })
    }
}

/// One `Transformer2DModel`: GroupNorm + `proj_in` 1×1 conv → flatten HW
/// → N `BasicTransformerBlock` (N=1 in SD 1.5) → `proj_out` 1×1 conv → residual.
pub struct Transformer2D {
    pub dim: usize,
    pub norm_weight: Vec<f32>, pub norm_bias: Vec<f32>,
    pub proj_in_w: F32Slice, pub proj_in_b: Vec<f32>,
    pub proj_out_w: F32Slice, pub proj_out_b: Vec<f32>,
    pub blocks: Vec<BasicTransformerBlock>, // SD 1.5 always has exactly 1
}

impl Transformer2D {
    fn load(sf: &SafetensorsFile, prefix: &str, dim: usize) -> Result<Self, UnetError> {
        let norm_weight = lf_vec(sf, &format!("{prefix}.norm.weight"))?;
        let norm_bias = lf_vec(sf, &format!("{prefix}.norm.bias"))?;
        let proj_in_w = lf(sf, &format!("{prefix}.proj_in.weight"))?;
        let proj_in_b = lf_vec(sf, &format!("{prefix}.proj_in.bias"))?;
        let proj_out_w = lf(sf, &format!("{prefix}.proj_out.weight"))?;
        let proj_out_b = lf_vec(sf, &format!("{prefix}.proj_out.bias"))?;
        if proj_in_w.len() != dim * dim {
            return Err(UnetError::Load(format!(
                "{prefix}.proj_in.weight: expected {} (= {dim}×{dim}×1×1), got {}",
                dim*dim, proj_in_w.len())));
        }
        let blocks = vec![BasicTransformerBlock::load(sf, &format!("{prefix}.transformer_blocks.0"), dim)?];
        Ok(Self { dim, norm_weight, norm_bias, proj_in_w, proj_in_b, proj_out_w, proj_out_b, blocks })
    }
}

/// Downsampler / upsampler shared shape (stride-2 conv for down, nearest+conv for up).
pub struct Downsampler { pub conv_w: F32Slice, pub conv_b: Vec<f32>, pub ch: usize }
pub struct Upsampler   { pub conv_w: F32Slice, pub conv_b: Vec<f32>, pub ch: usize }

/// Time-MLP from sinusoidal embedding to LM-style 1280-d signal.
pub struct TimeEmbedding {
    pub linear_1_w: F32Slice, pub linear_1_b: Vec<f32>,
    pub linear_2_w: F32Slice, pub linear_2_b: Vec<f32>,
}

pub struct DownBlock {
    pub kind: DownBlockKind,
    pub in_ch: usize, pub out_ch: usize,
    pub resnets: Vec<TimeCondResnet>,
    pub attentions: Vec<Transformer2D>,
    pub downsampler: Option<Downsampler>,
}
pub struct UpBlock {
    pub kind: UpBlockKind,
    pub in_ch: usize, pub out_ch: usize, pub prev_out_ch: usize,
    pub resnets: Vec<TimeCondResnet>,
    pub attentions: Vec<Transformer2D>,
    pub upsampler: Option<Upsampler>,
}

#[derive(Debug, Copy, Clone, PartialEq)] pub enum DownBlockKind { CrossAttnDown2D, Down2D }
#[derive(Debug, Copy, Clone, PartialEq)] pub enum UpBlockKind { CrossAttnUp2D, Up2D }

pub struct MidBlockCrossAttn {
    pub resnet_0: TimeCondResnet,
    pub attention: Transformer2D,
    pub resnet_1: TimeCondResnet,
}

/// Full SD 1.5 UNet2DConditionModel.
pub struct Sd15Unet {
    pub conv_in_w: F32Slice, pub conv_in_b: Vec<f32>,
    pub time_embedding: TimeEmbedding,
    pub down_blocks: Vec<DownBlock>,
    pub mid_block: MidBlockCrossAttn,
    pub up_blocks: Vec<UpBlock>,
    pub conv_norm_out_weight: Vec<f32>, pub conv_norm_out_bias: Vec<f32>,
    pub conv_out_w: F32Slice, pub conv_out_b: Vec<f32>,
}

impl Sd15Unet {
    pub fn load(path: &Path) -> Result<Self, UnetError> {
        eprintln!("[sd15-unet] loading {}", path.display());
        let sf = SafetensorsFile::load(path)
            .map_err(|e| UnetError::Load(format!("load {}: {e}", path.display())))?;
        let conv_in_w = lf(&sf, "conv_in.weight")?;
        let conv_in_b = lf_vec(&sf, "conv_in.bias")?;
        let time_embedding = TimeEmbedding {
            linear_1_w: lf(&sf, "time_embedding.linear_1.weight")?,
            linear_1_b: lf_vec(&sf, "time_embedding.linear_1.bias")?,
            linear_2_w: lf(&sf, "time_embedding.linear_2.weight")?,
            linear_2_b: lf_vec(&sf, "time_embedding.linear_2.bias")?,
        };

        // Down blocks (3 CrossAttnDown + 1 Down).
        let down_kinds = [
            DownBlockKind::CrossAttnDown2D,
            DownBlockKind::CrossAttnDown2D,
            DownBlockKind::CrossAttnDown2D,
            DownBlockKind::Down2D,
        ];
        let mut down_blocks = Vec::with_capacity(N_DOWN_BLOCKS);
        for bi in 0..N_DOWN_BLOCKS {
            let in_ch  = if bi == 0 { BLOCK_CHANNELS[0] } else { BLOCK_CHANNELS[bi - 1] };
            let out_ch = BLOCK_CHANNELS[bi];
            let mut resnets = Vec::with_capacity(LAYERS_PER_DOWN_BLOCK);
            for li in 0..LAYERS_PER_DOWN_BLOCK {
                let r_in = if li == 0 { in_ch } else { out_ch };
                resnets.push(TimeCondResnet::load(
                    &sf, &format!("down_blocks.{bi}.resnets.{li}"), r_in, out_ch)?);
            }
            let attentions = if down_kinds[bi] == DownBlockKind::CrossAttnDown2D {
                let mut a = Vec::with_capacity(LAYERS_PER_DOWN_BLOCK);
                for li in 0..LAYERS_PER_DOWN_BLOCK {
                    a.push(Transformer2D::load(
                        &sf, &format!("down_blocks.{bi}.attentions.{li}"), out_ch)?);
                }
                a
            } else { Vec::new() };
            // Down blocks 0/1/2 have downsamplers; block 3 (Down2D) doesn't.
            let downsampler = if bi < N_DOWN_BLOCKS - 1 {
                let w = lf(&sf, &format!("down_blocks.{bi}.downsamplers.0.conv.weight"))?;
                let b = lf_vec(&sf, &format!("down_blocks.{bi}.downsamplers.0.conv.bias"))?;
                if w.len() != out_ch * out_ch * 9 {
                    return Err(UnetError::Load(format!(
                        "down_blocks.{bi}.downsamplers.0.conv.weight: expected {}, got {}",
                        out_ch*out_ch*9, w.len())));
                }
                Some(Downsampler { conv_w: w, conv_b: b, ch: out_ch })
            } else { None };
            down_blocks.push(DownBlock {
                kind: down_kinds[bi], in_ch, out_ch, resnets, attentions, downsampler,
            });
        }

        // Mid block.
        let mid_ch = BLOCK_CHANNELS[3];
        let mid_block = MidBlockCrossAttn {
            resnet_0:  TimeCondResnet::load(&sf, "mid_block.resnets.0", mid_ch, mid_ch)?,
            attention: Transformer2D::load(&sf, "mid_block.attentions.0", mid_ch)?,
            resnet_1:  TimeCondResnet::load(&sf, "mid_block.resnets.1", mid_ch, mid_ch)?,
        };

        // Up blocks (1 Up + 3 CrossAttnUp).  Channel widths reverse: 1280, 1280, 640, 320.
        let up_kinds = [
            UpBlockKind::Up2D,
            UpBlockKind::CrossAttnUp2D,
            UpBlockKind::CrossAttnUp2D,
            UpBlockKind::CrossAttnUp2D,
        ];
        let up_ch = [BLOCK_CHANNELS[3], BLOCK_CHANNELS[3], BLOCK_CHANNELS[1], BLOCK_CHANNELS[0]];
        let prev_ch = [BLOCK_CHANNELS[3], BLOCK_CHANNELS[3], BLOCK_CHANNELS[3], BLOCK_CHANNELS[1]];
        let mut up_blocks = Vec::with_capacity(N_UP_BLOCKS);
        for bi in 0..N_UP_BLOCKS {
            let in_ch = if bi == 0 { up_ch[0] } else { up_ch[bi - 1] };
            let out_ch = up_ch[bi];
            let prev_out_ch = prev_ch[bi];
            // Resnets handle skip concatenation from the down-block path.
            // First resnet of each up_block has in_ch = (in_ch + prev_out_ch) (concat skip).
            // Middle resnet has in_ch = (out_ch + prev_out_ch).
            // Last resnet has in_ch = (out_ch + in_ch).  Per UNet skip pattern.
            // For Stage 0 we only verify the saved tensor shapes; the precise
            // skip-channel pattern is enforced at forward time.
            let mut resnets = Vec::with_capacity(LAYERS_PER_UP_BLOCK);
            for li in 0..LAYERS_PER_UP_BLOCK {
                // We let the per-tensor shape-check inside TimeCondResnet::load
                // discover the actual in_ch from the saved conv1.weight shape.
                let conv1_name = format!("up_blocks.{bi}.resnets.{li}.conv1.weight");
                let t = sf.get(&conv1_name).ok_or_else(|| UnetError::Load(format!(
                    "missing {conv1_name}")))?;
                // shape is [out_ch, in_ch, 3, 3]
                let in_ch_actual = t.shape[1];
                resnets.push(TimeCondResnet::load(
                    &sf, &format!("up_blocks.{bi}.resnets.{li}"), in_ch_actual, out_ch)?);
            }
            let attentions = if up_kinds[bi] == UpBlockKind::CrossAttnUp2D {
                let mut a = Vec::with_capacity(LAYERS_PER_UP_BLOCK);
                for li in 0..LAYERS_PER_UP_BLOCK {
                    a.push(Transformer2D::load(
                        &sf, &format!("up_blocks.{bi}.attentions.{li}"), out_ch)?);
                }
                a
            } else { Vec::new() };
            // Up blocks 0/1/2 have upsamplers; block 3 (final, full resolution) doesn't.
            let upsampler = if bi < N_UP_BLOCKS - 1 {
                let w = lf(&sf, &format!("up_blocks.{bi}.upsamplers.0.conv.weight"))?;
                let b = lf_vec(&sf, &format!("up_blocks.{bi}.upsamplers.0.conv.bias"))?;
                if w.len() != out_ch * out_ch * 9 {
                    return Err(UnetError::Load(format!(
                        "up_blocks.{bi}.upsamplers.0.conv.weight: expected {}, got {}",
                        out_ch*out_ch*9, w.len())));
                }
                Some(Upsampler { conv_w: w, conv_b: b, ch: out_ch })
            } else { None };
            up_blocks.push(UpBlock {
                kind: up_kinds[bi], in_ch, out_ch, prev_out_ch, resnets, attentions, upsampler,
            });
        }

        let conv_norm_out_weight = lf_vec(&sf, "conv_norm_out.weight")?;
        let conv_norm_out_bias = lf_vec(&sf, "conv_norm_out.bias")?;
        let conv_out_w = lf(&sf, "conv_out.weight")?;
        let conv_out_b = lf_vec(&sf, "conv_out.bias")?;
        if conv_out_w.len() != OUT_CHANNELS * BLOCK_CHANNELS[0] * 9 {
            return Err(UnetError::Load(format!(
                "conv_out.weight: expected {}, got {}",
                OUT_CHANNELS * BLOCK_CHANNELS[0] * 9, conv_out_w.len())));
        }
        Ok(Self {
            conv_in_w, conv_in_b,
            time_embedding,
            down_blocks, mid_block, up_blocks,
            conv_norm_out_weight, conv_norm_out_bias,
            conv_out_w, conv_out_b,
        })
    }

    /// Borrow-direct user-heap footprint (mmap-aliased weights stay in kernel cache).
    pub fn user_heap_bytes(&self) -> usize {
        let resnet_bytes = |r: &TimeCondResnet| {
            (r.norm1_weight.len() + r.norm1_bias.len() + r.conv1_b.len()
             + r.time_emb_proj_b.len() + r.norm2_weight.len() + r.norm2_bias.len()
             + r.conv2_b.len() + r.conv_shortcut_b.as_ref().map(|v| v.len()).unwrap_or(0)) * 4
        };
        let tx_bytes = |t: &Transformer2D| {
            (t.norm_weight.len() + t.norm_bias.len() + t.proj_in_b.len() + t.proj_out_b.len()) * 4
            + t.blocks.iter().map(|b| {
                (b.norm1_weight.len() + b.norm1_bias.len() + b.attn1_out_b.len()
                 + b.norm2_weight.len() + b.norm2_bias.len() + b.attn2_out_b.len()
                 + b.norm3_weight.len() + b.norm3_bias.len()
                 + b.ff_net0_proj_b.len() + b.ff_net2_b.len()) * 4
            }).sum::<usize>()
        };
        let mut n = self.conv_in_b.len() * 4;
        n += (self.time_embedding.linear_1_b.len() + self.time_embedding.linear_2_b.len()) * 4;
        for b in &self.down_blocks {
            for r in &b.resnets { n += resnet_bytes(r); }
            for a in &b.attentions { n += tx_bytes(a); }
            if let Some(d) = &b.downsampler { n += d.conv_b.len() * 4; }
        }
        n += resnet_bytes(&self.mid_block.resnet_0)
           + tx_bytes(&self.mid_block.attention)
           + resnet_bytes(&self.mid_block.resnet_1);
        for b in &self.up_blocks {
            for r in &b.resnets { n += resnet_bytes(r); }
            for a in &b.attentions { n += tx_bytes(a); }
            if let Some(u) = &b.upsampler { n += u.conv_b.len() * 4; }
        }
        n += (self.conv_norm_out_weight.len() + self.conv_norm_out_bias.len()) * 4;
        n += self.conv_out_b.len() * 4;
        n
    }
}

// ════════════════════════════════════════════════════════════════════════
// Forward primitives (Stage 1)
// ════════════════════════════════════════════════════════════════════════

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
fn gelu_tanh(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

fn group_norm32(x: &mut [f32], c: usize, h: usize, w: usize, gamma: &[f32], beta: &[f32], eps: f32) {
    let g = N_NORM_GROUPS;
    let gs = c / g;
    let plane = h * w;
    let group_n = (gs * plane) as f32;
    for gi in 0..g {
        let start = gi * gs;
        let mut mean = 0.0f32;
        for ci in start..start + gs {
            for v in &x[ci * plane .. (ci + 1) * plane] { mean += *v; }
        }
        mean /= group_n;
        let mut var = 0.0f32;
        for ci in start..start + gs {
            for v in &x[ci * plane .. (ci + 1) * plane] { let d = *v - mean; var += d * d; }
        }
        let inv = (var / group_n + eps).sqrt().recip();
        for ci in start..start + gs {
            let g_c = gamma[ci];
            let b_c = beta[ci];
            for v in &mut x[ci * plane .. (ci + 1) * plane] {
                *v = (*v - mean) * inv * g_c + b_c;
            }
        }
    }
}

fn conv2d_same(input: &[f32], c_in: usize, h: usize, w: usize,
               weight: &[f32], bias: &[f32], c_out: usize, k: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let pad = k / 2;
    let plane = h * w;
    let mut out = vec![0.0f32; c_out * plane];
    out.par_chunks_mut(plane).enumerate().for_each(|(oc, dst)| {
        let w_off_oc = oc * c_in * k * k;
        for y in 0..h {
            for x_ in 0..w {
                let mut acc = bias[oc];
                for ic in 0..c_in {
                    for ky in 0..k {
                        let iy = (y + ky).wrapping_sub(pad);
                        if iy >= h { continue; }
                        for kx in 0..k {
                            let ix = (x_ + kx).wrapping_sub(pad);
                            if ix >= w { continue; }
                            let w_idx = w_off_oc + (ic * k + ky) * k + kx;
                            acc += weight[w_idx] * input[ic * plane + iy * w + ix];
                        }
                    }
                }
                dst[y * w + x_] = acc;
            }
        }
    });
    out
}

/// SD UNet downsampler: stride-2 conv k=3 with **symmetric padding=1**
/// (`downsample_padding: 1` per config).  Output spatial dim = (H + 2 - 3) / 2 + 1 = (H - 1) / 2 + 1.
/// Differs from `rce-sd15-vae`'s asymmetric-pad downsampler (SD VAE uses padding=0 + (0,1,0,1) pad).
fn downsample_stride2_pad1(input: &[f32], c: usize, h: usize, w: usize,
                            weight: &[f32], bias: &[f32]) -> Vec<f32> {
    use rayon::prelude::*;
    let pad: i32 = 1;
    let h_out = (h + 2 * pad as usize - 3) / 2 + 1;
    let w_out = (w + 2 * pad as usize - 3) / 2 + 1;
    let plane_in = h * w;
    let plane_out = h_out * w_out;
    let mut out = vec![0.0f32; c * plane_out];
    out.par_chunks_mut(plane_out).enumerate().for_each(|(oc, dst)| {
        let w_off_oc = oc * c * 9;
        for y_out in 0..h_out {
            for x_out in 0..w_out {
                let mut acc = bias[oc];
                let y_in_base = 2 * y_out as i32 - pad;
                let x_in_base = 2 * x_out as i32 - pad;
                for ic in 0..c {
                    for ky in 0..3 {
                        let iy = y_in_base + ky as i32;
                        if iy < 0 || iy as usize >= h { continue; }
                        for kx in 0..3 {
                            let ix = x_in_base + kx as i32;
                            if ix < 0 || ix as usize >= w { continue; }
                            let w_idx = w_off_oc + (ic * 3 + ky) * 3 + kx;
                            acc += weight[w_idx] * input[ic * plane_in + (iy as usize) * w + (ix as usize)];
                        }
                    }
                }
                dst[y_out * w_out + x_out] = acc;
            }
        }
    });
    out
}

fn upsample_nearest_then_conv(input: &[f32], c: usize, h: usize, w: usize,
                               conv_w: &[f32], conv_b: &[f32]) -> Vec<f32> {
    let h2 = h * 2;
    let w2 = w * 2;
    let mut upscaled = vec![0.0f32; c * h2 * w2];
    for ci in 0..c {
        for y in 0..h2 {
            for x in 0..w2 {
                upscaled[ci * h2 * w2 + y * w2 + x] = input[ci * h * w + (y / 2) * w + (x / 2)];
            }
        }
    }
    conv2d_same(&upscaled, c, h2, w2, conv_w, conv_b, c, 3)
}

/// PyTorch-Linear matmul `out[m, n] = sum_k(input[m, k] · weight[n, k]) + bias[n]`.
fn linear_nt(input: &[f32], m: usize, k: usize, weight: &[f32], bias: Option<&[f32]>, n: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let mut out = vec![0.0f32; m * n];
    out.par_chunks_mut(n).enumerate().for_each(|(mi, row)| {
        let in_row = &input[mi * k .. (mi + 1) * k];
        for j in 0..n {
            let w_row = &weight[j * k .. (j + 1) * k];
            let mut s = 0.0f32;
            for kk in 0..k { s += in_row[kk] * w_row[kk]; }
            row[j] = s + bias.map(|b| b[j]).unwrap_or(0.0);
        }
    });
    out
}

/// Sinusoidal time embedding matching diffusers `get_timestep_embedding`:
/// `flip_sin_to_cos=True, freq_shift=0` → returns `[FREQ_DIM=320]`.
fn sinusoidal_time_embed(timestep: f32) -> [f32; FREQ_DIM] {
    let mut out = [0.0f32; FREQ_DIM];
    let half = FREQ_DIM / 2;
    let max_period = 10000.0_f32;
    let log_max = max_period.ln();
    for i in 0..half {
        let freq = (-log_max * (i as f32) / (half as f32)).exp();
        let arg = timestep * freq;
        // diffusers flip_sin_to_cos: cos first, then sin
        out[i] = arg.cos();
        out[i + half] = arg.sin();
    }
    out
}

fn time_embedding_forward(te: &TimeEmbedding, freq: &[f32]) -> Vec<f32> {
    let mut h = linear_nt(freq, 1, FREQ_DIM,
        te.linear_1_w.as_slice(), Some(&te.linear_1_b), TIME_EMBED_DIM);
    for v in h.iter_mut() { *v = silu(*v); }
    linear_nt(&h, 1, TIME_EMBED_DIM,
        te.linear_2_w.as_slice(), Some(&te.linear_2_b), TIME_EMBED_DIM)
}

/// Multi-head SDPA: q `[T_q, n_heads, head_dim]`, k/v `[T_kv, n_heads, head_dim]`,
/// returns `[T_q, n_heads, head_dim]`.  No bias, no mask.  Scaling 1/√head_dim.
fn mh_sdpa(q: &[f32], k: &[f32], v: &[f32],
           t_q: usize, t_kv: usize, n_heads: usize, head_dim: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let dim = n_heads * head_dim;
    let scale = (head_dim as f32).powf(-0.5);
    let mut out = vec![0.0f32; t_q * dim];
    out.par_chunks_mut(dim).enumerate().for_each(|(qi, dst)| {
        for h in 0..n_heads {
            let q_off = qi * dim + h * head_dim;
            let mut scores = vec![0.0f32; t_kv];
            let mut row_max = f32::NEG_INFINITY;
            for j in 0..t_kv {
                let k_off = j * dim + h * head_dim;
                let mut s = 0.0f32;
                for d in 0..head_dim { s += q[q_off + d] * k[k_off + d]; }
                let sv = s * scale;
                scores[j] = sv;
                if sv > row_max { row_max = sv; }
            }
            let mut sum = 0.0f32;
            for j in 0..t_kv { let e = (scores[j] - row_max).exp(); scores[j] = e; sum += e; }
            let inv = sum.recip();
            for j in 0..t_kv { scores[j] *= inv; }
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..t_kv {
                    acc += scores[j] * v[j * dim + h * head_dim + d];
                }
                dst[h * head_dim + d] = acc;
            }
        }
    });
    out
}

fn layernorm_affine(x: &[f32], t: usize, d: usize, w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for ti in 0..t {
        let row = &x[ti * d .. (ti + 1) * d];
        let mean: f32 = row.iter().sum::<f32>() / (d as f32);
        let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (d as f32);
        let inv = (var + eps).sqrt().recip();
        for di in 0..d {
            out[ti * d + di] = (row[di] - mean) * inv * w[di] + b[di];
        }
    }
    out
}

/// TimeCondResnet forward: same as VAE ResnetBlock + a time-projection
/// `silu(t_emb) → Linear → out_ch` added per-channel between conv1 and norm2.
fn time_cond_resnet_forward(block: &TimeCondResnet, x: &[f32], t_emb: &[f32], h: usize, w: usize, eps: f32) -> Vec<f32> {
    let plane = h * w;
    debug_assert_eq!(x.len(), block.in_ch * plane);
    debug_assert_eq!(t_emb.len(), TIME_EMBED_DIM);
    // First norm + activation + conv.
    let mut h_buf = x.to_vec();
    group_norm32(&mut h_buf, block.in_ch, h, w, &block.norm1_weight, &block.norm1_bias, eps);
    for v in h_buf.iter_mut() { *v = silu(*v); }
    let mut h_buf = conv2d_same(&h_buf, block.in_ch, h, w,
        block.conv1_w.as_slice(), &block.conv1_b, block.out_ch, 3);
    // Time emb projection.
    let mut t_act = t_emb.to_vec();
    for v in t_act.iter_mut() { *v = silu(*v); }
    let t_proj = linear_nt(&t_act, 1, TIME_EMBED_DIM,
        block.time_emb_proj_w.as_slice(), Some(&block.time_emb_proj_b), block.out_ch);
    // Add t_proj per-channel broadcast over HxW.
    for ci in 0..block.out_ch {
        let val = t_proj[ci];
        for p in 0..plane { h_buf[ci * plane + p] += val; }
    }
    // Second norm + activation + conv.
    group_norm32(&mut h_buf, block.out_ch, h, w, &block.norm2_weight, &block.norm2_bias, eps);
    for v in h_buf.iter_mut() { *v = silu(*v); }
    let h_buf = conv2d_same(&h_buf, block.out_ch, h, w,
        block.conv2_w.as_slice(), &block.conv2_b, block.out_ch, 3);
    // Skip.
    let skip: Vec<f32> = if block.in_ch == block.out_ch {
        x.to_vec()
    } else {
        conv2d_same(x, block.in_ch, h, w,
            block.conv_shortcut_w.as_ref().unwrap().as_slice(),
            block.conv_shortcut_b.as_ref().unwrap(), block.out_ch, 1)
    };
    let mut out = h_buf;
    for i in 0..out.len() { out[i] += skip[i]; }
    out
}

/// Transformer2D forward: input `[c, h, w]` + encoder_hidden_states `[t_text, CROSS_ATTN_DIM=768]`.
/// 1. GroupNorm → proj_in (1×1 conv) → reshape to [HW, C]
/// 2. For each BasicTransformerBlock:
///    a. norm1 + self-attn (q/k/v from x, multi-head, head_dim=8) + residual
///    b. norm2 + cross-attn (q from x, k/v from text) + residual
///    c. norm3 + GEGLU FFN + residual
/// 3. Reshape back, proj_out (1×1 conv), add to original residual
fn transformer2d_forward(block: &Transformer2D, x: &[f32], c: usize, h: usize, w: usize,
                          encoder_hidden_states: &[f32], t_text: usize, eps: f32) -> Vec<f32> {
    let plane = h * w;
    // 1) GroupNorm + proj_in 1x1
    let mut h_buf = x.to_vec();
    group_norm32(&mut h_buf, c, h, w, &block.norm_weight, &block.norm_bias, eps);
    let h_buf = conv2d_same(&h_buf, c, h, w,
        block.proj_in_w.as_slice(), &block.proj_in_b, c, 1);
    // 2) Flatten to [HW, C]
    let mut hw_c = vec![0.0f32; plane * c];
    for ci in 0..c {
        for p in 0..plane { hw_c[p * c + ci] = h_buf[ci * plane + p]; }
    }
    // 3) BasicTransformerBlock loop
    let n_heads = N_ATTN_HEADS;
    let head_dim = c / n_heads;
    for tb in &block.blocks {
        // ── Self-attn branch ──
        let h_norm = layernorm_affine(&hw_c, plane, c, &tb.norm1_weight, &tb.norm1_bias, 1e-5);
        let q = linear_nt(&h_norm, plane, c, tb.attn1_q.as_slice(), None, c);
        let k = linear_nt(&h_norm, plane, c, tb.attn1_k.as_slice(), None, c);
        let v = linear_nt(&h_norm, plane, c, tb.attn1_v.as_slice(), None, c);
        let attn = mh_sdpa(&q, &k, &v, plane, plane, n_heads, head_dim);
        let attn_out = linear_nt(&attn, plane, c, tb.attn1_out_w.as_slice(), Some(&tb.attn1_out_b), c);
        for i in 0..hw_c.len() { hw_c[i] += attn_out[i]; }
        // ── Cross-attn branch ──
        let h_norm = layernorm_affine(&hw_c, plane, c, &tb.norm2_weight, &tb.norm2_bias, 1e-5);
        let q = linear_nt(&h_norm, plane, c, tb.attn2_q.as_slice(), None, c);
        let k = linear_nt(encoder_hidden_states, t_text, CROSS_ATTN_DIM,
                          tb.attn2_k.as_slice(), None, c);
        let v = linear_nt(encoder_hidden_states, t_text, CROSS_ATTN_DIM,
                          tb.attn2_v.as_slice(), None, c);
        let attn = mh_sdpa(&q, &k, &v, plane, t_text, n_heads, head_dim);
        let attn_out = linear_nt(&attn, plane, c, tb.attn2_out_w.as_slice(), Some(&tb.attn2_out_b), c);
        for i in 0..hw_c.len() { hw_c[i] += attn_out[i]; }
        // ── GEGLU FFN ──
        let h_norm = layernorm_affine(&hw_c, plane, c, &tb.norm3_weight, &tb.norm3_bias, 1e-5);
        // net.0.proj : [4·C·2, C] → gate + value
        let gv = linear_nt(&h_norm, plane, c,
            tb.ff_net0_proj_w.as_slice(), Some(&tb.ff_net0_proj_b), 8 * c);
        // Diffusers GEGLU: `value, gate = chunk(proj(x), 2, dim=-1); return value * gelu(gate)`.
        // First half along channel dim is the value, second half is the gate.
        let ff = 4 * c;
        let mut gated = vec![0.0f32; plane * ff];
        for pi in 0..plane {
            for j in 0..ff {
                let value = gv[pi * 8 * c + j];
                let gate = gv[pi * 8 * c + ff + j];
                gated[pi * ff + j] = value * gelu_tanh(gate);
            }
        }
        let ff_out = linear_nt(&gated, plane, ff, tb.ff_net2_w.as_slice(), Some(&tb.ff_net2_b), c);
        for i in 0..hw_c.len() { hw_c[i] += ff_out[i]; }
    }
    // 4) Reshape back to [C, H, W], proj_out 1x1, add residual.
    let mut h_buf = vec![0.0f32; c * plane];
    for ci in 0..c {
        for p in 0..plane { h_buf[ci * plane + p] = hw_c[p * c + ci]; }
    }
    let proj_out = conv2d_same(&h_buf, c, h, w,
        block.proj_out_w.as_slice(), &block.proj_out_b, c, 1);
    let mut out = vec![0.0f32; c * plane];
    for i in 0..out.len() { out[i] = x[i] + proj_out[i]; }
    out
}

/// Concatenate `[c1, h, w]` and `[c2, h, w]` along channel axis → `[c1+c2, h, w]`.
fn concat_channels(a: &[f32], c1: usize, b: &[f32], c2: usize, h: usize, w: usize) -> Vec<f32> {
    let plane = h * w;
    debug_assert_eq!(a.len(), c1 * plane);
    debug_assert_eq!(b.len(), c2 * plane);
    let mut out = vec![0.0f32; (c1 + c2) * plane];
    out[0 .. c1 * plane].copy_from_slice(a);
    out[c1 * plane ..].copy_from_slice(b);
    out
}

impl Sd15Unet {
    /// One U-Net forward step: `latent [4, h, w]` + scalar `t` (DDIM-style 0..1000)
    /// + `encoder_hidden_states [t_text, 768]` → predicted velocity `[4, h, w]`.
    pub fn forward(&self, latent: &[f32], h: usize, w: usize,
                   t_scalar: f32,
                   encoder_hidden_states: &[f32], t_text: usize,
                   eps: f32) -> Vec<f32> {
        debug_assert_eq!(latent.len(), IN_CHANNELS * h * w);

        // Time embed.
        let freq = sinusoidal_time_embed(t_scalar);
        let t_emb = time_embedding_forward(&self.time_embedding, &freq);

        // conv_in (4 → 320)
        let mut x = conv2d_same(latent, IN_CHANNELS, h, w,
            self.conv_in_w.as_slice(), &self.conv_in_b, BLOCK_CHANNELS[0], 3);
        let mut hh = h;
        let mut ww = w;
        let mut skips: Vec<(Vec<f32>, usize, usize, usize)> = Vec::new();
        skips.push((x.clone(), BLOCK_CHANNELS[0], hh, ww));

        // Down blocks.
        for blk in &self.down_blocks {
            for (li, res) in blk.resnets.iter().enumerate() {
                x = time_cond_resnet_forward(res, &x, &t_emb, hh, ww, eps);
                if !blk.attentions.is_empty() {
                    let attn = &blk.attentions[li];
                    x = transformer2d_forward(attn, &x, blk.out_ch, hh, ww,
                        encoder_hidden_states, t_text, eps);
                }
                skips.push((x.clone(), blk.out_ch, hh, ww));
            }
            if let Some(down) = &blk.downsampler {
                x = downsample_stride2_pad1(&x, blk.out_ch, hh, ww,
                    down.conv_w.as_slice(), &down.conv_b);
                hh /= 2; ww /= 2;
                skips.push((x.clone(), blk.out_ch, hh, ww));
            }
        }

        // Mid block.
        x = time_cond_resnet_forward(&self.mid_block.resnet_0, &x, &t_emb, hh, ww, eps);
        x = transformer2d_forward(&self.mid_block.attention, &x, BLOCK_CHANNELS[3], hh, ww,
            encoder_hidden_states, t_text, eps);
        x = time_cond_resnet_forward(&self.mid_block.resnet_1, &x, &t_emb, hh, ww, eps);

        // Up blocks: pop a skip for each resnet, concat along channels.
        for blk in &self.up_blocks {
            for (li, res) in blk.resnets.iter().enumerate() {
                let (skip, skip_c, skip_h, skip_w) = skips.pop().expect("skip underflow");
                debug_assert_eq!((skip_h, skip_w), (hh, ww),
                    "skip spatial mismatch: got {skip_h}×{skip_w}, expected {hh}×{ww}");
                let mut cur_c = if li == 0 { blk.in_ch } else { blk.out_ch };
                // The first resnet of each up_block takes (in_ch + skip_c); subsequent
                // ones take (out_ch + skip_c).  conv1.weight in_ch == res.in_ch matches
                // that exactly (auto-detected at load time).
                let _ = cur_c;
                let concatenated = concat_channels(&x, if li == 0 { blk.in_ch } else { blk.out_ch },
                                                    &skip, skip_c, hh, ww);
                x = time_cond_resnet_forward(res, &concatenated, &t_emb, hh, ww, eps);
                if !blk.attentions.is_empty() {
                    let attn = &blk.attentions[li];
                    x = transformer2d_forward(attn, &x, blk.out_ch, hh, ww,
                        encoder_hidden_states, t_text, eps);
                }
            }
            if let Some(up) = &blk.upsampler {
                x = upsample_nearest_then_conv(&x, blk.out_ch, hh, ww,
                    up.conv_w.as_slice(), &up.conv_b);
                hh *= 2; ww *= 2;
            }
        }

        // Output head: GroupNorm + SiLU + conv_out 3×3 → 4 channels.
        group_norm32(&mut x, BLOCK_CHANNELS[0], hh, ww,
            &self.conv_norm_out_weight, &self.conv_norm_out_bias, eps);
        for v in x.iter_mut() { *v = silu(*v); }
        conv2d_same(&x, BLOCK_CHANNELS[0], hh, ww,
            self.conv_out_w.as_slice(), &self.conv_out_b, OUT_CHANNELS, 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn unet_path() -> Option<PathBuf> {
        for vol in ["macos_4TB_external", "macos_4TB_external_1", "macos_4TB_external_2"] {
            let p = PathBuf::from(format!(
                "/Volumes/{vol}/neuronexus_ai/model_library/stable-diffusion-v1-5/unet/diffusion_pytorch_model.safetensors"));
            if p.exists() { return Some(p); }
        }
        None
    }

    /// Stage 0: load all 686 tensors and verify the UNet2DConditionModel
    /// architecture closes (counts + shapes).  Borrow-direct keeps the
    /// 3.4 GB checkpoint in mmap so user heap stays tiny.
    #[test]
    #[ignore = "sd15-unet Stage 0 — needs SD 1.5 UNet safetensors mounted"]
    fn sd15_unet_load_full() {
        let Some(path) = unet_path() else { eprintln!("SD UNet not mounted"); return; };
        let t0 = std::time::Instant::now();
        let m = Sd15Unet::load(&path).expect("load");
        eprintln!("sd15-unet loaded in {:?}", t0.elapsed());

        assert_eq!(m.down_blocks.len(), N_DOWN_BLOCKS);
        assert_eq!(m.up_blocks.len(), N_UP_BLOCKS);
        // Encoder side: 3 CrossAttnDown + 1 Down.
        for (bi, b) in m.down_blocks.iter().enumerate() {
            let expected_attn = bi < 3;
            assert_eq!(!b.attentions.is_empty(), expected_attn,
                       "down block {bi} attentions presence");
            assert_eq!(b.resnets.len(), LAYERS_PER_DOWN_BLOCK);
            assert_eq!(b.downsampler.is_some(), bi < N_DOWN_BLOCKS - 1,
                       "down block {bi} downsampler presence");
        }
        // Decoder side: 1 Up + 3 CrossAttnUp.
        for (bi, b) in m.up_blocks.iter().enumerate() {
            let expected_attn = bi >= 1;
            assert_eq!(!b.attentions.is_empty(), expected_attn,
                       "up block {bi} attentions presence");
            assert_eq!(b.resnets.len(), LAYERS_PER_UP_BLOCK);
            assert_eq!(b.upsampler.is_some(), bi < N_UP_BLOCKS - 1,
                       "up block {bi} upsampler presence");
        }
        // Cross-attention K/V dims = CROSS_ATTN_DIM (768 = CLIP-L text dim).
        let mid_tb = &m.mid_block.attention.blocks[0];
        assert_eq!(mid_tb.attn2_k.len(), 1280 * CROSS_ATTN_DIM);
        assert_eq!(mid_tb.attn2_v.len(), 1280 * CROSS_ATTN_DIM);

        let mb = m.user_heap_bytes() as f64 / (1024.0 * 1024.0);
        eprintln!("sd15-unet user heap (borrow-direct): {:.3} MB (full 686 tensors)", mb);
        assert!(mb < 50.0, "borrow-direct should keep heap small; got {mb} MB");
        eprintln!("sd15-unet Stage 0 ✓");
    }

    /// Stage 1 forward smoke: synthetic 4×8×8 latent + synthetic [77, 768]
    /// text context + scalar t=0.5 → predicted 4×8×8 velocity.  Walks the
    /// full down → mid → up path with skip-concat.  Verifies all primitives
    /// (sinusoidal time embed + time-conditioned resnets + multi-head SDPA
    /// + cross-attn to text + GEGLU FFN + Transformer2D wrapper + skip-concat)
    /// compose into a finite output.
    #[test]
    #[ignore = "sd15-unet Stage 1 — needs SD UNet safetensors mounted; ~30 s"]
    fn sd15_unet_forward_smoke() {
        let Some(path) = unet_path() else { eprintln!("UNet not mounted"); return; };
        let m = Sd15Unet::load(&path).expect("load");

        let h = 8;
        let w = 8;
        let latent: Vec<f32> = (0..IN_CHANNELS * h * w)
            .map(|i| ((i as f32) * 0.07).sin() * 0.5).collect();
        let t_text = 77;
        let ctx: Vec<f32> = (0..t_text * CROSS_ATTN_DIM)
            .map(|i| ((i as f32) * 0.011).cos() * 0.05).collect();

        let t0 = std::time::Instant::now();
        let pred = m.forward(&latent, h, w, 500.0, &ctx, t_text, 1e-5);
        let dt = t0.elapsed();

        assert_eq!(pred.len(), OUT_CHANNELS * h * w);
        assert!(pred.iter().all(|v| v.is_finite()));
        let max_abs: f32 = pred.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let mean_abs: f32 = pred.iter().map(|v| v.abs()).sum::<f32>() / (pred.len() as f32);
        eprintln!("sd15-unet forward 4×{h}×{w}: {dt:?} max_abs={max_abs:.4} mean_abs={mean_abs:.4}");
        assert!(max_abs > 0.0);
        eprintln!("sd15-unet Stage 1 forward ✓");
    }

    /// Full-UNet Python-reference numerical agreement vs the official
    /// diffusers `UNet2DConditionModel.forward`.  Reference captured on
    /// 2026-06-06: deterministic 4×8×8 latent + [77, 768] text + t=500.
    #[test]
    #[ignore = "sd15-unet ref — needs SD UNet safetensors + /tmp/sd_unet_ref.f32.bin"]
    fn sd15_unet_forward_vs_python_reference() {
        let Some(path) = unet_path() else { eprintln!("UNet not mounted"); return; };
        let ref_path = std::path::PathBuf::from("/tmp/sd_unet_ref.f32.bin");
        if !ref_path.exists() { eprintln!("ref not found"); return; }
        let ref_bytes = std::fs::read(&ref_path).expect("read ref");
        let ref_data: Vec<f32> = ref_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(ref_data.len(), OUT_CHANNELS * 8 * 8);
        let ref_max: f32 = ref_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("Python ref: ‖.‖={:.4} max_abs={ref_max:.4}",
                  ref_data.iter().map(|v| v*v).sum::<f32>().sqrt());

        let m = Sd15Unet::load(&path).expect("load");
        let h = 8;
        let w = 8;
        let latent: Vec<f32> = (0..IN_CHANNELS * h * w)
            .map(|i| ((i as f32) * 0.07).sin() * 0.5).collect();
        let t_text = 77;
        let ctx: Vec<f32> = (0..t_text * CROSS_ATTN_DIM)
            .map(|i| ((i as f32) * 0.011).cos() * 0.05).collect();

        let t0 = std::time::Instant::now();
        let pred = m.forward(&latent, h, w, 500.0, &ctx, t_text, 1e-5);
        eprintln!("rce forward: {:?}", t0.elapsed());

        let mut max_abs_diff = 0.0f32;
        let mut sum_sq = 0.0f32;
        for (a, b) in pred.iter().zip(ref_data.iter()) {
            let d = (a - b).abs();
            if d > max_abs_diff { max_abs_diff = d; }
            sum_sq += d * d;
        }
        let rms = (sum_sq / (pred.len() as f32)).sqrt();
        let rel = max_abs_diff / ref_max;
        let pred_max: f32 = pred.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("vs Python ref: rce max_abs={pred_max:.4}, max_abs_diff={max_abs_diff:.6} rms={rms:.6} rel={:.2}%",
                  rel * 100.0);
        // rce UNet matches diffusers UNet2DConditionModel.forward at f32 noise floor.
        // Bugs fixed during numerical debugging (per-block trace harness):
        //   1. **Attention head semantics**: SD 1.5 config `attention_head_dim=8` is
        //      LEGACY convention meaning `num_attention_heads=8` (NOT head_dim=8).
        //      Real head_dim is `inner_dim / 8` (= 40/80/160/160 per block width).
        //   2. **UNet downsampler padding**: symmetric pad=1 (config `downsample_padding: 1`),
        //      NOT the asymmetric (0,1,0,1) pad used by SD VAE's downsamplers.
        //   3. **GEGLU chunk order**: diffusers does `value, gate = chunk(.); value * gelu(gate)`
        //      — first half is value, second is gate.
        assert!(rel < 0.01, "UNet must match diffusers ref at f32 noise; got rel={}%", rel * 100.0);
    }

    /// Per-block trace comparison vs diffusers checkpoints to isolate
    /// the layer at which rce first diverges from the reference.
    /// Reads /tmp/sd_unet_trace_{conv_in, down0_resnet0, ...}.f32.bin
    /// captured by a Python hook script.
    #[test]
    #[ignore = "sd15-unet per-block trace — needs SD UNet safetensors + /tmp/sd_unet_trace_*.f32.bin"]
    fn sd15_unet_per_block_trace() {
        let Some(path) = unet_path() else { eprintln!("UNet not mounted"); return; };
        let read_ref = |name: &str, expected_len: usize| -> Option<Vec<f32>> {
            // Try both trace_ (block-level) and tb_ (transformer-block-internal) prefixes.
            let p1 = format!("/tmp/sd_unet_trace_{name}.f32.bin");
            let p2 = format!("/tmp/sd_unet_{name}.f32.bin");
            let p = if std::path::Path::new(&p1).exists() { p1 } else { p2 };
            let p = std::path::PathBuf::from(p);
            if !p.exists() { eprintln!("  ?? ref not found for {name}"); return None; }
            let bytes = std::fs::read(&p).ok()?;
            if bytes.len() != expected_len * 4 {
                eprintln!("  WARN {name}: ref bytes {} != {}", bytes.len(), expected_len * 4);
                return None;
            }
            Some(bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        };
        let cmp = |name: &str, rce: &[f32], r: &[f32]| {
            let mut max = 0.0f32;
            let mut sumsq = 0.0f32;
            for (a, b) in rce.iter().zip(r.iter()) {
                let d = (a - b).abs();
                if d > max { max = d; }
                sumsq += d * d;
            }
            let rms = (sumsq / (rce.len() as f32)).sqrt();
            let ref_max = r.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let pred_max = rce.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let rel = max / ref_max.max(1e-9);
            eprintln!("  {name}: rce max={pred_max:.4} ref max={ref_max:.4}  max_diff={max:.6} rms={rms:.6} rel={:.2}%",
                      rel * 100.0);
        };

        let m = Sd15Unet::load(&path).expect("load");
        let h = 8;
        let w = 8;
        let latent: Vec<f32> = (0..IN_CHANNELS * h * w)
            .map(|i| ((i as f32) * 0.07).sin() * 0.5).collect();
        let t_text = 77;
        let ctx: Vec<f32> = (0..t_text * CROSS_ATTN_DIM)
            .map(|i| ((i as f32) * 0.011).cos() * 0.05).collect();
        let freq = sinusoidal_time_embed(500.0);
        let t_emb = time_embedding_forward(&m.time_embedding, &freq);

        // 1) conv_in
        let post_conv_in = conv2d_same(&latent, IN_CHANNELS, h, w,
            m.conv_in_w.as_slice(), &m.conv_in_b, BLOCK_CHANNELS[0], 3);
        if let Some(r) = read_ref("conv_in", BLOCK_CHANNELS[0] * h * w) {
            cmp("conv_in", &post_conv_in, &r);
        }

        // 2) down0.resnet0
        let down0 = &m.down_blocks[0];
        let post_d0_r0 = time_cond_resnet_forward(&down0.resnets[0], &post_conv_in, &t_emb, h, w, 1e-5);
        if let Some(r) = read_ref("down0_resnet0", down0.out_ch * h * w) {
            cmp("down0_resnet0", &post_d0_r0, &r);
        }

        // 3) down0.attn0
        let post_d0_a0 = transformer2d_forward(&down0.attentions[0], &post_d0_r0,
            down0.out_ch, h, w, &ctx, t_text, 1e-5);
        if let Some(r) = read_ref("down0_attn0", down0.out_ch * h * w) {
            cmp("down0_attn0", &post_d0_a0, &r);
        }

        // Continue down with granular checkpoints.
        let mut x = post_d0_a0;
        let mut hh = h;
        let mut ww = w;
        // Down block 0: resnet1 + attn1, then downsampler.
        let blk0 = &m.down_blocks[0];
        x = time_cond_resnet_forward(&blk0.resnets[1], &x, &t_emb, hh, ww, 1e-5);
        if let Some(r) = read_ref("d0r1", blk0.out_ch * hh * ww) {
            cmp("d0r1", &x, &r);
        }
        x = transformer2d_forward(&blk0.attentions[1], &x, blk0.out_ch, hh, ww, &ctx, t_text, 1e-5);
        if let Some(r) = read_ref("d0a1", blk0.out_ch * hh * ww) {
            cmp("d0a1", &x, &r);
        }
        if let Some(down) = &blk0.downsampler {
            x = downsample_stride2_pad1(&x, blk0.out_ch, hh, ww,
                down.conv_w.as_slice(), &down.conv_b);
            hh /= 2; ww /= 2;
        }
        if let Some(r) = read_ref("d0_down", blk0.out_ch * hh * ww) {
            cmp("d0_down", &x, &r);
        }
        // Down block 1.
        let blk1 = &m.down_blocks[1];
        for (li, res) in blk1.resnets.iter().enumerate() {
            x = time_cond_resnet_forward(res, &x, &t_emb, hh, ww, 1e-5);
            if !blk1.attentions.is_empty() {
                x = transformer2d_forward(&blk1.attentions[li], &x, blk1.out_ch, hh, ww,
                    &ctx, t_text, 1e-5);
            }
        }
        if let Some(down) = &blk1.downsampler {
            x = downsample_stride2_pad1(&x, blk1.out_ch, hh, ww,
                down.conv_w.as_slice(), &down.conv_b);
            hh /= 2; ww /= 2;
        }
        if let Some(r) = read_ref("d1_block", blk1.out_ch * hh * ww) {
            cmp("d1_block", &x, &r);
        }
        // Down block 2.
        let blk2 = &m.down_blocks[2];
        for (li, res) in blk2.resnets.iter().enumerate() {
            x = time_cond_resnet_forward(res, &x, &t_emb, hh, ww, 1e-5);
            if !blk2.attentions.is_empty() {
                x = transformer2d_forward(&blk2.attentions[li], &x, blk2.out_ch, hh, ww,
                    &ctx, t_text, 1e-5);
            }
        }
        if let Some(down) = &blk2.downsampler {
            x = downsample_stride2_pad1(&x, blk2.out_ch, hh, ww,
                down.conv_w.as_slice(), &down.conv_b);
            hh /= 2; ww /= 2;
        }
        if let Some(r) = read_ref("d2_block", blk2.out_ch * hh * ww) {
            cmp("d2_block", &x, &r);
        }
        // Down block 3.
        let blk3 = &m.down_blocks[3];
        for res in &blk3.resnets {
            x = time_cond_resnet_forward(res, &x, &t_emb, hh, ww, 1e-5);
        }
        if let Some(r) = read_ref("down3_block", blk3.out_ch * hh * ww) {
            cmp("down3_block", &x, &r);
        }
        x = time_cond_resnet_forward(&m.mid_block.resnet_0, &x, &t_emb, hh, ww, 1e-5);
        x = transformer2d_forward(&m.mid_block.attention, &x, BLOCK_CHANNELS[3], hh, ww,
            &ctx, t_text, 1e-5);
        x = time_cond_resnet_forward(&m.mid_block.resnet_1, &x, &t_emb, hh, ww, 1e-5);
        if let Some(r) = read_ref("mid_block", 1280 * hh * ww) {
            cmp("mid_block", &x, &r);
        }
        eprintln!("walk complete");
    }
}
