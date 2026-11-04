//! V-JEPA 2 video JEPA runtime on rce.
//!
//! Closes pattern-lang Phase 3 (JEPA world models). Loads Meta's
//! `facebook/vjepa2-vitl-fpc64-256` checkpoint and runs the encoder +
//! predictor end-to-end on a video tensor.
//!
//! Architecture (`config.json` + `configs/train/vitl16/cooldown-256px-64f.yaml`):
//!
//! ```text
//! input:  video clip, B × 3 ch × T_frames × H × W   (e.g. 1 × 3 × 64 × 256 × 256)
//!          ↓ patch_embed: 3D conv stride (2, 16, 16) — tubelet=2, patch=16
//!         → B × N × 1024   (N = T/2 · H/16 · W/16, e.g. 8192)
//!          ↓ encoder: 24 × { LN → MHA(16h, qkv_bias, RoPE-3D) → +res → LN → GELU FFN(×4) → +res }
//!          ↓ output_norm
//!         → B × N × 1024   (context embeddings)
//!          ↓ predictor:   proj 1024 → 384, mask-token broadcast at target positions,
//!                          12 × { ViT block (12 heads, 384 dim, RoPE-3D) } → output_norm → proj 384 → 1024
//!         → predicted target patch embeddings
//! ```
//!
//! Numerical contract at this brick: **loadable + forward-runnable on
//! real input + finite output**. Byte-equivalence vs PyTorch reference
//! is the next brick (Mantle JEPA already has a pytorch_equivalence
//! test; this crate is the runtime-side analogue).

use std::path::Path;

use rce_weight_loader::{F32Slice, LoadError, LoadedTensor, SafetensorsFile};

/// Frozen architecture constants for the ViT-L 256px / 64f variant.
#[derive(Copy, Clone, Debug)]
pub struct VJepa2Config {
    pub img_size: usize,
    pub num_frames: usize,
    pub patch_size: usize,
    pub tubelet_size: usize,
    pub in_chans: usize,

    pub enc_dim: usize,
    pub enc_layers: usize,
    pub enc_heads: usize,
    pub enc_mlp_ratio: usize,

    pub pred_dim: usize,
    pub pred_layers: usize,
    pub pred_heads: usize,
    pub pred_mlp_ratio: usize,
    pub pred_num_mask_tokens: usize,

    pub layer_norm_eps: f32,
    pub qkv_bias: bool,
}

impl VJepa2Config {
    pub const VITL_256PX_64F: Self = Self {
        img_size: 256, num_frames: 64,
        patch_size: 16, tubelet_size: 2, in_chans: 3,
        enc_dim: 1024, enc_layers: 24, enc_heads: 16, enc_mlp_ratio: 4,
        pred_dim: 384, pred_layers: 12, pred_heads: 12, pred_mlp_ratio: 4,
        pred_num_mask_tokens: 10,
        layer_norm_eps: 1e-6, qkv_bias: true,
    };
}

#[derive(Debug, thiserror::Error)]
pub enum LoadVJepaError {
    #[error("load: {0}")] Load(#[from] LoadError),
    #[error("missing tensor: {0}")] MissingTensor(String),
    #[error("tensor {name} has dtype {got}, expected f32")]
    DtypeMismatch { name: String, got: String },
    #[error("tensor {name} has shape {got:?}, expected {want:?}")]
    ShapeMismatch { name: String, got: Vec<usize>, want: Vec<usize> },
}

// ─── Tensor view helpers ─────────────────────────────────────────────

/// Borrow-direct f32 tensor with a fixed shape.  `data` is an [`F32Slice`]
/// that aliases directly into the safetensors mmap (zero-copy) and keeps the
/// mmap alive via an `Arc<Mmap>` refcount.  Field access stays ergonomic via
/// `Deref<Target=[f32]>`: `&tensor.data` coerces to `&[f32]` at call sites.
#[derive(Clone, Debug)]
pub struct F32Tensor {
    pub shape: Vec<usize>,
    pub data: F32Slice,
}

impl F32Tensor {
    fn from_loaded(t: &LoadedTensor, name: &str, want_shape: Option<&[usize]>) -> Result<Self, LoadVJepaError> {
        if t.dtype != "f32" {
            return Err(LoadVJepaError::DtypeMismatch { name: name.into(), got: t.dtype.clone() });
        }
        if let Some(w) = want_shape {
            if t.shape != w {
                return Err(LoadVJepaError::ShapeMismatch { name: name.into(), got: t.shape.clone(), want: w.to_vec() });
            }
        }
        let data = F32Slice::from_tensor(name, t)
            .map_err(|e| LoadVJepaError::DtypeMismatch { name: name.into(), got: e })?;
        Ok(Self { shape: t.shape.clone(), data })
    }
}

// ─── Block weights ───────────────────────────────────────────────────

#[derive(Debug)]
pub struct AttnWeights {
    pub q_w: F32Slice, pub q_b: F32Slice, // [d,d] [d]
    pub k_w: F32Slice, pub k_b: F32Slice,
    pub v_w: F32Slice, pub v_b: F32Slice,
    pub o_w: F32Slice, pub o_b: F32Slice,
}

#[derive(Debug)]
pub struct MlpWeights {
    pub fc1_w: F32Slice, pub fc1_b: F32Slice, // [4d,d] [4d]
    pub fc2_w: F32Slice, pub fc2_b: F32Slice, // [d,4d] [d]
}

#[derive(Debug)]
pub struct BlockWeights {
    pub norm1_w: F32Slice, pub norm1_b: F32Slice,
    pub norm2_w: F32Slice, pub norm2_b: F32Slice,
    pub attn: AttnWeights,
    pub mlp: MlpWeights,
}

#[derive(Debug)]
pub struct VJepa2 {
    pub cfg: VJepa2Config,

    // Encoder
    pub patch_w: F32Tensor,         // [1024, 3, 2, 16, 16]
    pub patch_b: F32Slice,
    pub enc_blocks: Vec<BlockWeights>,
    pub enc_norm_w: F32Slice,
    pub enc_norm_b: F32Slice,

    // Predictor
    pub pred_embed_w: F32Slice,     // [384, 1024]
    pub pred_embed_b: F32Slice,
    pub pred_mask_tokens: F32Tensor, // [10, 1, 1, 384]
    pub pred_blocks: Vec<BlockWeights>,
    pub pred_norm_w: F32Slice,
    pub pred_norm_b: F32Slice,
    pub pred_proj_w: F32Slice,      // [1024, 384]
    pub pred_proj_b: F32Slice,
}

impl VJepa2 {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, LoadVJepaError> {
        let file = SafetensorsFile::load(path)?;
        let cfg = VJepa2Config::VITL_256PX_64F;
        let get = |name: &str| -> Result<&LoadedTensor, LoadVJepaError> {
            file.get(name).ok_or_else(|| LoadVJepaError::MissingTensor(name.into()))
        };
        let read_v = |name: &str| -> Result<F32Slice, LoadVJepaError> {
            let t = get(name)?;
            F32Slice::from_tensor(name, t)
                .map_err(|e| LoadVJepaError::DtypeMismatch { name: name.into(), got: e })
        };

        let patch_w = F32Tensor::from_loaded(
            get("encoder.embeddings.patch_embeddings.proj.weight")?,
            "patch.weight",
            Some(&[cfg.enc_dim, cfg.in_chans, cfg.tubelet_size, cfg.patch_size, cfg.patch_size]),
        )?;
        let patch_b = read_v("encoder.embeddings.patch_embeddings.proj.bias")?;

        let mut enc_blocks = Vec::with_capacity(cfg.enc_layers);
        for i in 0..cfg.enc_layers {
            enc_blocks.push(load_block(&file, &format!("encoder.layer.{i}"), cfg.enc_dim, cfg.enc_dim * cfg.enc_mlp_ratio)?);
        }
        let enc_norm_w = read_v("encoder.layernorm.weight")?;
        let enc_norm_b = read_v("encoder.layernorm.bias")?;

        let pred_embed_w = read_v("predictor.embeddings.predictor_embeddings.weight")?;
        let pred_embed_b = read_v("predictor.embeddings.predictor_embeddings.bias")?;
        let pred_mask_tokens = F32Tensor::from_loaded(
            get("predictor.embeddings.mask_tokens")?,
            "predictor.embeddings.mask_tokens",
            Some(&[cfg.pred_num_mask_tokens, 1, 1, cfg.pred_dim]),
        )?;

        let mut pred_blocks = Vec::with_capacity(cfg.pred_layers);
        for i in 0..cfg.pred_layers {
            pred_blocks.push(load_block(&file, &format!("predictor.layer.{i}"), cfg.pred_dim, cfg.pred_dim * cfg.pred_mlp_ratio)?);
        }
        let pred_norm_w = read_v("predictor.layernorm.weight")?;
        let pred_norm_b = read_v("predictor.layernorm.bias")?;
        let pred_proj_w = read_v("predictor.proj.weight")?;
        let pred_proj_b = read_v("predictor.proj.bias")?;

        Ok(Self {
            cfg, patch_w, patch_b, enc_blocks, enc_norm_w, enc_norm_b,
            pred_embed_w, pred_embed_b, pred_mask_tokens, pred_blocks,
            pred_norm_w, pred_norm_b, pred_proj_w, pred_proj_b,
        })
    }

    /// Forward the encoder on a video tensor.
    ///
    /// `video` shape: `[T, 3, H, W]` flat (channels-first per frame),
    /// T frames each `H × W` pixels in [0, 1] roughly (we don't
    /// re-normalize here — caller should already have applied the
    /// ImageNet-style mean/std the checkpoint was trained on).
    ///
    /// Returns `(tokens, n_t, n_h, n_w)` where `tokens.shape = [N, enc_dim]`
    /// and `N = n_t * n_h * n_w`.
    pub fn encode(&self, video: &[f32], t_frames: usize, h: usize, w: usize) -> (Vec<Vec<f32>>, usize, usize, usize) {
        let cfg = &self.cfg;
        assert_eq!(video.len(), 3 * t_frames * h * w, "video shape mismatch");
        assert_eq!(t_frames % cfg.tubelet_size, 0);
        assert_eq!(h % cfg.patch_size, 0);
        assert_eq!(w % cfg.patch_size, 0);
        let n_t = t_frames / cfg.tubelet_size;
        let n_h = h / cfg.patch_size;
        let n_w = w / cfg.patch_size;

        // 1) Patch embed.
        let mut tokens = patch_embed_3d(video, &self.patch_w.data, &self.patch_b,
            cfg.enc_dim, cfg.in_chans, cfg.tubelet_size, cfg.patch_size,
            t_frames, h, w);

        // 2) Encoder blocks.
        let d_head = cfg.enc_dim / cfg.enc_heads;
        let rope_dims = rope_axis_dims(d_head);
        for blk in &self.enc_blocks {
            block_forward(&mut tokens, blk, cfg.enc_heads, d_head, rope_dims, n_t, n_h, n_w, cfg.layer_norm_eps);
        }

        // 3) Final norm.
        for tok in tokens.iter_mut() {
            *tok = layer_norm(tok, &self.enc_norm_w, &self.enc_norm_b, cfg.layer_norm_eps);
        }

        (tokens, n_t, n_h, n_w)
    }

    /// Forward predictor on `context` (visible token embeddings of
    /// shape `[N_visible, enc_dim]`) and predict embeddings for
    /// `target_positions` (flat ids into the full grid).
    pub fn predict(
        &self,
        context: &[Vec<f32>],
        visible_positions: &[usize],
        target_positions: &[usize],
        n_t: usize, n_h: usize, n_w: usize,
        mask_token_idx: usize,
    ) -> Vec<Vec<f32>> {
        let cfg = &self.cfg;
        let d_head = cfg.pred_dim / cfg.pred_heads;
        let rope_dims = rope_axis_dims(d_head);

        // 1) Project context 1024 → 384, plus mask tokens at target positions.
        let mut tokens: Vec<Vec<f32>> = Vec::with_capacity(visible_positions.len() + target_positions.len());
        let mut positions: Vec<usize> = Vec::with_capacity(tokens.capacity());
        for (i, &pos) in visible_positions.iter().enumerate() {
            tokens.push(linear(&context[i], &self.pred_embed_w, &self.pred_embed_b, cfg.pred_dim, cfg.enc_dim));
            positions.push(pos);
        }
        let mask_start = mask_token_idx * cfg.pred_dim;
        let mask_vec: Vec<f32> = self.pred_mask_tokens.data[mask_start..mask_start + cfg.pred_dim].to_vec();
        for &pos in target_positions {
            tokens.push(mask_vec.clone());
            positions.push(pos);
        }

        // 2) Predictor blocks (with custom positions for RoPE — packed sequence).
        for blk in &self.pred_blocks {
            block_forward_with_positions(&mut tokens, &positions, blk,
                cfg.pred_heads, d_head, rope_dims, n_t, n_h, n_w, cfg.layer_norm_eps);
        }

        // 3) Final norm + project back to enc_dim.
        let n_visible = visible_positions.len();
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(target_positions.len());
        for tok in tokens.iter_mut().skip(n_visible) {
            *tok = layer_norm(tok, &self.pred_norm_w, &self.pred_norm_b, cfg.layer_norm_eps);
            out.push(linear(tok, &self.pred_proj_w, &self.pred_proj_b, cfg.enc_dim, cfg.pred_dim));
        }
        out
    }
}

fn load_block(file: &SafetensorsFile, prefix: &str, d: usize, d_ff: usize) -> Result<BlockWeights, LoadVJepaError> {
    let get = |name: &str| -> Result<&LoadedTensor, LoadVJepaError> {
        file.get(name).ok_or_else(|| LoadVJepaError::MissingTensor(name.into()))
    };
    let read_v = |name: &str| -> Result<F32Slice, LoadVJepaError> {
        let t = get(name)?;
        F32Slice::from_tensor(name, t)
            .map_err(|e| LoadVJepaError::DtypeMismatch { name: name.into(), got: e })
    };
    let read_shape = |name: &str, want: &[usize]| -> Result<F32Slice, LoadVJepaError> {
        let t = get(name)?;
        if t.shape != want {
            return Err(LoadVJepaError::ShapeMismatch { name: name.into(), got: t.shape.clone(), want: want.to_vec() });
        }
        F32Slice::from_tensor(name, t)
            .map_err(|e| LoadVJepaError::DtypeMismatch { name: name.into(), got: e })
    };
    Ok(BlockWeights {
        norm1_w: read_shape(&format!("{prefix}.norm1.weight"), &[d])?,
        norm1_b: read_shape(&format!("{prefix}.norm1.bias"), &[d])?,
        norm2_w: read_shape(&format!("{prefix}.norm2.weight"), &[d])?,
        norm2_b: read_shape(&format!("{prefix}.norm2.bias"), &[d])?,
        attn: AttnWeights {
            q_w: read_shape(&format!("{prefix}.attention.query.weight"), &[d, d])?,
            q_b: read_shape(&format!("{prefix}.attention.query.bias"), &[d])?,
            k_w: read_shape(&format!("{prefix}.attention.key.weight"), &[d, d])?,
            k_b: read_shape(&format!("{prefix}.attention.key.bias"), &[d])?,
            v_w: read_shape(&format!("{prefix}.attention.value.weight"), &[d, d])?,
            v_b: read_shape(&format!("{prefix}.attention.value.bias"), &[d])?,
            o_w: read_shape(&format!("{prefix}.attention.proj.weight"), &[d, d])?,
            o_b: read_shape(&format!("{prefix}.attention.proj.bias"), &[d])?,
        },
        mlp: MlpWeights {
            fc1_w: read_shape(&format!("{prefix}.mlp.fc1.weight"), &[d_ff, d])?,
            fc1_b: read_shape(&format!("{prefix}.mlp.fc1.bias"), &[d_ff])?,
            fc2_w: read_shape(&format!("{prefix}.mlp.fc2.weight"), &[d, d_ff])?,
            fc2_b: read_v(&format!("{prefix}.mlp.fc2.bias"))?,
        },
    })
}

// ─── Numerics ────────────────────────────────────────────────────────

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter().zip(w.iter()).zip(b.iter())
        .map(|((v, ww), bb)| (v - mean) * inv * ww + bb)
        .collect()
}

/// Batched per-token linear: gather `Vec<Vec<f32>>` tokens into a flat buffer,
/// call `rce_op_conv2d::linear_f32` (Apple Accelerate sgemm on macOS), scatter
/// back into per-token `Vec<f32>`s.  Costs one extra flat alloc + n*d copies on
/// each side, but the GEMM win on large matrices (qkv/proj/fc1/fc2 at d=1024,
/// d_ff=4096, n≈1568) far outweighs the memcpy overhead.
fn linear_tokens(tokens: &[Vec<f32>], w: &[f32], b: &[f32], dout: usize, din: usize) -> Vec<Vec<f32>> {
    let n = tokens.len();
    if n == 0 { return Vec::new(); }
    let mut flat = Vec::with_capacity(n * din);
    for tk in tokens { flat.extend_from_slice(tk); }
    let y = rce_op_conv2d::linear_f32(&flat, n, din, w, dout, Some(b));
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(n);
    for i in 0..n {
        out.push(y[i*dout..(i+1)*dout].to_vec());
    }
    out
}

/// `out = w · x + b` where `w` is row-major `[out, in]` and x is `[in]`.
fn linear(x: &[f32], w: &[f32], b: &[f32], out: usize, in_: usize) -> Vec<f32> {
    let mut y = b.to_vec();
    debug_assert_eq!(w.len(), out * in_);
    debug_assert_eq!(x.len(), in_);
    for r in 0..out {
        let row = &w[r * in_..r * in_ + in_];
        let mut acc = y[r];
        for i in 0..in_ { acc += row[i] * x[i]; }
        y[r] = acc;
    }
    y
}

fn gelu(x: f32) -> f32 {
    // erf-based exact GELU (matches PyTorch default `nn.GELU()`).
    0.5 * x * (1.0 + libm_erf(x / std::f32::consts::SQRT_2))
}

/// Abramowitz-Stegun 7.1.26 erf approximation (max abs err ~1.5e-7).
fn libm_erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

// ─── 3D RoPE ─────────────────────────────────────────────────────────

#[derive(Copy, Clone, Debug)]
pub struct RopeAxisDims { pub d: usize, pub h: usize, pub w: usize }

fn rope_axis_dims(head_dim: usize) -> RopeAxisDims {
    // From modules.py:139:
    //   d_dim = 2 * ((head_dim // 3) // 2)
    let a = 2 * ((head_dim / 3) / 2);
    RopeAxisDims { d: a, h: a, w: a }
}

/// Rotate first `n` dims of a vector by RoPE at scalar position `pos`.
/// Matches `rotate_queries_or_keys` in modules.py — **including** the
/// "subtle bug" (repeat instead of repeat_interleave) that the V-JEPA 2
/// pretrained checkpoint was trained with.
fn rope_rotate(x: &mut [f32], pos: f32, n: usize) {
    debug_assert_eq!(n % 2, 0);
    let half = n / 2;
    for i in 0..half {
        // omega = (1 / 10000^(i / half))
        let omega = (-(i as f32) * (10000.0f32).ln() / half as f32).exp();
        let theta = pos * omega;
        let c = theta.cos();
        let s = theta.sin();
        // The Python "bug": repeat (not repeat_interleave) means pair (i, i+half)
        // shares the same (sin, cos) value (treating the vector as [a..., b...]
        // where each half holds one "frequency channel" instead of interleaving).
        let y1 = x[i];
        let y2 = x[i + half];
        // y = (-y2, y1) per the rotate; then output = x*cos + y*sin
        x[i]        = y1 * c + (-y2) * s;
        x[i + half] = y2 * c + y1 * s;
    }
}

// ─── Attention block ─────────────────────────────────────────────────

fn block_forward(
    tokens: &mut Vec<Vec<f32>>, blk: &BlockWeights,
    n_heads: usize, d_head: usize, rope_dims: RopeAxisDims,
    n_t: usize, n_h: usize, n_w: usize, eps: f32,
) {
    let n = tokens.len();
    let positions: Vec<usize> = (0..n).collect();
    block_forward_with_positions(tokens, &positions, blk, n_heads, d_head, rope_dims, n_t, n_h, n_w, eps);
}

fn block_forward_with_positions(
    tokens: &mut Vec<Vec<f32>>, positions: &[usize], blk: &BlockWeights,
    n_heads: usize, d_head: usize, rope_dims: RopeAxisDims,
    _n_t: usize, n_h: usize, n_w: usize, eps: f32,
) {
    let n = tokens.len();
    let d = n_heads * d_head;
    debug_assert_eq!(positions.len(), n);

    // ── Sub-block 1: norm1 → MHA(qkv_bias, RoPE-3D) → +residual ──
    let normed: Vec<Vec<f32>> = tokens.iter()
        .map(|x| layer_norm(x, &blk.norm1_w, &blk.norm1_b, eps))
        .collect();
    let qs = linear_tokens(&normed, &blk.attn.q_w, &blk.attn.q_b, d, d);
    let ks = linear_tokens(&normed, &blk.attn.k_w, &blk.attn.k_b, d, d);
    let vs = linear_tokens(&normed, &blk.attn.v_w, &blk.attn.v_b, d, d);

    // Apply 3D RoPE to Q and K, per-head along (d, h, w) axes.
    let tokens_per_frame = n_h * n_w;
    let tokens_per_row = n_w;
    let scale = 1.0f32 / (d_head as f32).sqrt();
    let mut qs = qs; let mut ks = ks;
    for i in 0..n {
        let pos = positions[i];
        let frame_pos = (pos / tokens_per_frame) as f32;
        let row_pos = ((pos % tokens_per_frame) / tokens_per_row) as f32;
        let col_pos = (pos % tokens_per_row) as f32;
        for h in 0..n_heads {
            let off = h * d_head;
            let q = &mut qs[i][off..off + d_head];
            let mut s = 0usize;
            rope_rotate(&mut q[s..s + rope_dims.d], frame_pos, rope_dims.d); s += rope_dims.d;
            rope_rotate(&mut q[s..s + rope_dims.h], row_pos, rope_dims.h);   s += rope_dims.h;
            rope_rotate(&mut q[s..s + rope_dims.w], col_pos, rope_dims.w);   let _ = s;
            let k = &mut ks[i][off..off + d_head];
            let mut s = 0usize;
            rope_rotate(&mut k[s..s + rope_dims.d], frame_pos, rope_dims.d); s += rope_dims.d;
            rope_rotate(&mut k[s..s + rope_dims.h], row_pos, rope_dims.h);   s += rope_dims.h;
            rope_rotate(&mut k[s..s + rope_dims.w], col_pos, rope_dims.w);   let _ = s;
        }
    }

    // Full attention: out[i, h] = softmax(q[i,h] · k[:,h] / sqrt(d_head)) · v[:,h].
    let mut attn_out: Vec<Vec<f32>> = vec![vec![0.0f32; d]; n];
    for i in 0..n {
        for h in 0..n_heads {
            let off = h * d_head;
            let qh = &qs[i][off..off + d_head];
            let mut sc = vec![0.0f32; n];
            for j in 0..n {
                let kh = &ks[j][off..off + d_head];
                let mut s = 0.0f32;
                for k in 0..d_head { s += qh[k] * kh[k]; }
                sc[j] = s * scale;
            }
            let m = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in sc.iter_mut() { *s = (*s - m).exp(); sum += *s; }
            for s in sc.iter_mut() { *s /= sum; }
            for j in 0..n {
                let vh = &vs[j][off..off + d_head];
                let w = sc[j];
                for k in 0..d_head { attn_out[i][off + k] += w * vh[k]; }
            }
        }
    }
    // o_w projection + residual — batched via linear_tokens.
    let proj_all = linear_tokens(&attn_out, &blk.attn.o_w, &blk.attn.o_b, d, d);
    for i in 0..n {
        for k in 0..d { tokens[i][k] += proj_all[i][k]; }
    }

    // ── Sub-block 2: norm2 → fc1 → GELU → fc2 → +residual ──
    let d_ff = blk.mlp.fc1_w.len() / d;
    let normed2: Vec<Vec<f32>> = tokens.iter()
        .map(|x| layer_norm(x, &blk.norm2_w, &blk.norm2_b, eps))
        .collect();
    let mut ups = linear_tokens(&normed2, &blk.mlp.fc1_w, &blk.mlp.fc1_b, d_ff, d);
    for tk in ups.iter_mut() { for v in tk.iter_mut() { *v = gelu(*v); } }
    let dns = linear_tokens(&ups, &blk.mlp.fc2_w, &blk.mlp.fc2_b, d, d_ff);
    for i in 0..n {
        for k in 0..d { tokens[i][k] += dns[i][k]; }
    }
}

// ─── Patch embed (3D conv) ───────────────────────────────────────────

/// 3D conv: input video `[3, T, H, W]` flat (channels-first per frame
/// arranged as ch, frame, row, col), weight `[out_ch, in_ch, kT, kH, kW]`
/// flat, applied with stride `(tubelet, patch, patch)` and zero
/// padding. Output token at (t, h, w) flattened as `t*n_h*n_w + h*n_w + w`.
fn patch_embed_3d(
    video: &[f32], w: &[f32], b: &[f32],
    out_ch: usize, in_ch: usize, kt: usize, ks: usize,
    t_frames: usize, h_pix: usize, w_pix: usize,
) -> Vec<Vec<f32>> {
    let n_t = t_frames / kt;
    let n_h = h_pix / ks;
    let n_w = w_pix / ks;

    // Video flat indexing: video[c, t, h, w] = video[((c*t_frames+t)*h_pix+h)*w_pix+w]
    let stride_c = t_frames * h_pix * w_pix;
    let stride_t = h_pix * w_pix;
    let stride_h = w_pix;
    // Weight flat indexing: w[o, c, kt_idx, kh, kw]
    //   = w[(((o*in_ch+c)*kt+kt_idx)*ks+kh)*ks+kw]
    let w_stride_c = kt * ks * ks;
    let w_stride_t = ks * ks;
    let w_stride_h = ks;

    let mut out: Vec<Vec<f32>> = Vec::with_capacity(n_t * n_h * n_w);
    for ti in 0..n_t {
        for hi in 0..n_h {
            for wi in 0..n_w {
                let mut tok = vec![0.0f32; out_ch];
                let t0 = ti * kt;
                let h0 = hi * ks;
                let w0 = wi * ks;
                for o in 0..out_ch {
                    let mut acc = b[o];
                    let w_o = o * in_ch * w_stride_c;
                    for c in 0..in_ch {
                        let v_c = c * stride_c;
                        let w_c = w_o + c * w_stride_c;
                        for kti in 0..kt {
                            let v_t = v_c + (t0 + kti) * stride_t;
                            let w_t = w_c + kti * w_stride_t;
                            for kh in 0..ks {
                                let v_h = v_t + (h0 + kh) * stride_h;
                                let w_h = w_t + kh * w_stride_h;
                                for kw in 0..ks {
                                    acc += video[v_h + w0 + kw] * w[w_h + kw];
                                }
                            }
                        }
                    }
                    tok[o] = acc;
                }
                out.push(tok);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL: &str =
        "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/vjepa2-vitl-fpc64-256.safetensors";

    #[test]
    fn vjepa_a_load_real_checkpoint() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("vjepa.a: model not mounted — skipping"); return;
        }
        let m = VJepa2::load(MODEL).expect("load");
        let cfg = &m.cfg;
        assert_eq!(m.enc_blocks.len(), cfg.enc_layers);
        assert_eq!(m.pred_blocks.len(), cfg.pred_layers);
        eprintln!("vjepa.a OK: V-JEPA 2 ViT-L loaded — enc {}L d{} + pred {}L d{}",
                  cfg.enc_layers, cfg.enc_dim, cfg.pred_layers, cfg.pred_dim);
    }

    #[test]
    fn vjepa_c_tiny_predict() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("vjepa.c: model not mounted — skipping"); return;
        }
        let m = VJepa2::load(MODEL).expect("load");
        // 2 frames × 32×32 → 1 tubelet × 2×2 spatial = 4 tokens.
        let t = 2; let h = 32; let w = 32;
        let video: Vec<f32> = (0..3*t*h*w).map(|i| ((i as f32 * 0.013).sin()) * 0.5).collect();
        let (ctx, n_t, n_h, n_w) = m.encode(&video, t, h, w);
        // Mask out positions 2 and 3 (predict them from visible 0, 1).
        let visible: Vec<usize> = vec![0, 1];
        let target:  Vec<usize> = vec![2, 3];
        let ctx_v: Vec<Vec<f32>> = visible.iter().map(|&i| ctx[i].clone()).collect();
        let t0 = std::time::Instant::now();
        let pred = m.predict(&ctx_v, &visible, &target, n_t, n_h, n_w, 0);
        let dt = t0.elapsed();
        assert_eq!(pred.len(), target.len());
        assert_eq!(pred[0].len(), m.cfg.enc_dim);
        for p in &pred {
            for v in p { assert!(v.is_finite(), "non-finite pred"); }
        }
        let amax = pred.iter().flat_map(|t| t.iter()).fold(0.0f32, |m, v| m.max(v.abs()));
        eprintln!("vjepa.c OK: predicted {}×{} embeddings, amax={amax:.3} ({dt:?})",
                  pred.len(), m.cfg.enc_dim);
    }

    #[test]
    fn vjepa_b_tiny_encode() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("vjepa.b: model not mounted — skipping"); return;
        }
        let m = VJepa2::load(MODEL).expect("load");
        // Smallest meaningful input: 2 frames (1 tubelet) × 16x16 (1x1 patches)
        // = 1 token. Bump to 32x32 → 4 tokens for slightly less trivial attention.
        let t = 2; let h = 32; let w = 32;
        let video: Vec<f32> = (0..3*t*h*w).map(|i| ((i as f32 * 0.013).sin()) * 0.5).collect();
        let t0 = std::time::Instant::now();
        let (toks, n_t, n_h, n_w) = m.encode(&video, t, h, w);
        let dt = t0.elapsed();
        assert_eq!(toks.len(), n_t * n_h * n_w);
        assert_eq!(toks.len(), 1 * 2 * 2);
        assert_eq!(toks[0].len(), m.cfg.enc_dim);
        for tok in &toks {
            for v in tok { assert!(v.is_finite(), "non-finite enc embedding"); }
        }
        let s: f32 = toks.iter().flat_map(|t| t.iter()).sum();
        let amax = toks.iter().flat_map(|t| t.iter()).fold(0.0f32, |m, v| m.max(v.abs()));
        eprintln!("vjepa.b OK: {}×{} embeddings, amax={amax:.3} sum={s:.3} ({dt:?})", toks.len(), m.cfg.enc_dim);
    }
}
