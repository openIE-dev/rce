//! Depth-Anything V2 small — rce's first per-pixel vision model.
//!
//! Architecture (verified against
//! `https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py`):
//!
//! - **DINOv2 ViT-S/14** backbone: Conv2d(3→384, k14, s14) patch embed +
//!   cls token + learned positional embeddings; 12 pre-norm Transformer
//!   blocks with **LayerScale** (`ls1.gamma`, `ls2.gamma` per-channel
//!   multiply on each sub-block output before the residual add); GELU
//!   (exact erf), LayerNorm eps=1e-6.
//! - **DPT head**: tap intermediate features from blocks [2, 5, 8, 11];
//!   per-stage reassemble (1×1 conv 384→`[48,96,192,384]` + resize
//!   layers — ConvT(s4) / ConvT(s2) / Identity / Conv(s2)); per-stage
//!   3×3 conv into common 64 channels; 4 RefineNet `FeatureFusionBlock`s
//!   (deepest-first, each = two `ResidualConvUnit`s + bilinear upsample
//!   to the next finer scale + 1×1 out_conv); final `output_conv1`
//!   (64→32, 3×3) + bilinear upsample to (patch_h·14, patch_w·14) +
//!   `output_conv2` (32→32 3×3 → ReLU → 32→1 1×1 → ReLU).
//!
//! Input: RGB image, dims must be multiples of 14, normalized with
//! ImageNet mean/std. Output: monocular depth map (larger = closer per
//! DA-V2's relative-depth convention). All weights F32.
//!
//! Deterministic verification (`tests::depth_anything_two_region`):
//! procedurally render a 224×224 image with a bright central foreground
//! square on a dark background; assert the central region's mean depth
//! is distinct from the surrounding region (clear foreground/background
//! separation) and the map is finite + structured (high variance).

use rce_weight_loader::{F32Slice, LoadedTensor, SafetensorsFile};
use rayon::prelude::*;
use std::path::Path;

const D: usize = 384;
const N_HEADS: usize = 6;
const HD: usize = 64; // D / N_HEADS
const FF: usize = 1536;
const N_LAYERS: usize = 12;
const PATCH: usize = 14;
const NATIVE_GRID: usize = 37; // 518/14 → pos_embed [1, 1+37², 384]
const EPS: f32 = 1e-6;
const INTER_IDX: [usize; 4] = [2, 5, 8, 11];
const STAGE_CH: [usize; 4] = [48, 96, 192, 384];
const FUSION: usize = 64;
const HEAD_F2: usize = 32;
const IMNET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMNET_STD: [f32; 3] = [0.229, 0.224, 0.225];

// ── numeric helpers ────────────────────────────────────────────────

#[inline] fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0 - (((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t - 0.284_496_74) * t + 0.254_829_6) * t * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}
#[inline] fn gelu(x: f32) -> f32 { 0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2)) }

fn slice_of(t: &LoadedTensor) -> F32Slice {
    F32Slice::from_tensor(&t.name, t).unwrap_or_else(|e| panic!("{e}"))
}

/// `y[t, d_out] = x[t, d_in] · W[d_out, d_in]^T (+bias)`. Delegates to
/// `rce_op_conv2d::linear_f32` which uses Apple Accelerate's AMX sgemm on
/// macOS (~1.5 TFLOPS) and `matrixmultiply` elsewhere.
fn linear(x: &[f32], t: usize, w: &[f32], dout: usize, din: usize, bias: Option<&[f32]>) -> Vec<f32> {
    rce_op_conv2d::linear_f32(x, t, din, w, dout, bias)
}

/// LayerNorm with weight + bias over last dim, per row, in place.
fn layernorm(x: &mut [f32], d: usize, w: &[f32], b: &[f32]) {
    x.par_chunks_mut(d).for_each(|row| {
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| { let z = v - mean; z * z }).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        for ((v, g), bb) in row.iter_mut().zip(w).zip(b) { *v = (*v - mean) * inv * *g + *bb; }
    });
}

/// Multi-head bidirectional SDPA, scale = 1/√d_head.
fn attention(q: &[f32], k: &[f32], v: &[f32], t: usize) -> Vec<f32> {
    let sc = 1.0 / (HD as f32).sqrt();
    let mut out = vec![0.0f32; t * D];
    out.par_chunks_mut(D).enumerate().for_each(|(i, orow)| {
        for h in 0..N_HEADS {
            let qh = &q[i * D + h * HD..i * D + h * HD + HD];
            let mut sc_arr = vec![0.0f32; t];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..t {
                let kj = &k[j * D + h * HD..j * D + h * HD + HD];
                let mut s = 0.0f32; for d in 0..HD { s += qh[d] * kj[d]; }
                s *= sc; sc_arr[j] = s; if s > mx { mx = s; }
            }
            let mut sum = 0.0f32;
            for s in sc_arr.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
            let inv = 1.0 / sum;
            let oh = &mut orow[h * HD..(h + 1) * HD];
            for j in 0..t {
                let w = sc_arr[j] * inv;
                let vj = &v[j * D + h * HD..j * D + h * HD + HD];
                for d in 0..HD { oh[d] += w * vj[d]; }
            }
        }
    });
    out
}

// ── 2D conv + transposed conv (channel-major [C, H, W]) ────────────

/// 2D conv. `w` row-major `[Cout, Cin, kH, kW]`.
#[allow(clippy::too_many_arguments)]
fn conv2d(inp: &[f32], cin: usize, h: usize, w: usize, weight: &[f32], cout: usize, kh: usize, kw: usize, sh: usize, sw: usize, ph: usize, pw: usize, bias: Option<&[f32]>) -> (Vec<f32>, usize, usize) {
    let ho = (h + 2 * ph - kh) / sh + 1;
    let wo = (w + 2 * pw - kw) / sw + 1;
    let mut out = vec![0.0f32; cout * ho * wo];
    out.par_chunks_mut(ho * wo).enumerate().for_each(|(oc, oplane)| {
        let wbase = oc * cin * kh * kw;
        for oh in 0..ho {
            for ow in 0..wo {
                let mut acc = bias.map_or(0.0, |b| b[oc]);
                for ic in 0..cin {
                    for ky in 0..kh {
                        let ih = (oh * sh + ky) as isize - ph as isize;
                        if ih < 0 || ih as usize >= h { continue; }
                        for kx in 0..kw {
                            let iw = (ow * sw + kx) as isize - pw as isize;
                            if iw < 0 || iw as usize >= w { continue; }
                            let wv = weight[wbase + ic * kh * kw + ky * kw + kx];
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

/// 2D ConvTranspose. PyTorch weight layout `[Cin, Cout, kH, kW]`.
#[allow(clippy::too_many_arguments)]
use rce_op_conv2d::conv_transpose2d;

/// Bilinear interpolation with `align_corners=True`, `[C, H, W]` → `[C, H_out, W_out]`.
fn bilinear_resize(inp: &[f32], c: usize, h: usize, w: usize, ho: usize, wo: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c * ho * wo];
    let sy = if ho > 1 { (h - 1) as f32 / (ho - 1) as f32 } else { 0.0 };
    let sx = if wo > 1 { (w - 1) as f32 / (wo - 1) as f32 } else { 0.0 };
    out.par_chunks_mut(ho * wo).enumerate().for_each(|(ch, plane)| {
        let inp_c = &inp[ch * h * w..(ch + 1) * h * w];
        for y in 0..ho {
            let fy = y as f32 * sy;
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(h - 1);
            let wy = fy - y0 as f32;
            for x in 0..wo {
                let fx = x as f32 * sx;
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(w - 1);
                let wx = fx - x0 as f32;
                let v00 = inp_c[y0 * w + x0];
                let v01 = inp_c[y0 * w + x1];
                let v10 = inp_c[y1 * w + x0];
                let v11 = inp_c[y1 * w + x1];
                let v0 = v00 * (1.0 - wx) + v01 * wx;
                let v1 = v10 * (1.0 - wx) + v11 * wx;
                plane[y * wo + x] = v0 * (1.0 - wy) + v1 * wy;
            }
        }
    });
    out
}

// ── weights ────────────────────────────────────────────────────────

struct VitBlock {
    norm1_w: F32Slice, norm1_b: F32Slice,
    qkv_w: F32Slice, qkv_b: F32Slice, proj_w: F32Slice, proj_b: F32Slice,
    ls1: F32Slice,
    norm2_w: F32Slice, norm2_b: F32Slice,
    fc1_w: F32Slice, fc1_b: F32Slice, fc2_w: F32Slice, fc2_b: F32Slice,
    ls2: F32Slice,
}

struct ResConvUnit { c1_w: F32Slice, c1_b: F32Slice, c2_w: F32Slice, c2_b: F32Slice }
struct RefineNet { ru1: ResConvUnit, ru2: ResConvUnit, out_w: F32Slice, out_b: F32Slice }

pub struct DepthAnything {
    patch_w: F32Slice, patch_b: F32Slice, // Conv2d(3, 384, k14, s14)
    cls_token: F32Slice, // [384]
    pos_embed: F32Slice, // [1370, 384]
    blocks: Vec<VitBlock>,
    // DA-V2 takes raw intermediate hidden states (no final encoder LN);
    // pretrained.norm.* is loaded for completeness but unused.
    #[allow(dead_code)] norm_w: F32Slice,
    #[allow(dead_code)] norm_b: F32Slice,
    // DPT head
    projects_w: Vec<F32Slice>, projects_b: Vec<F32Slice>, // 4 × 1×1 conv
    resize_w: Vec<Option<F32Slice>>, resize_b: Vec<Option<F32Slice>>,    // 4 × resize (idx 2 is Identity → None)
    layer_rn_w: Vec<F32Slice>,                            // 4 × 3×3 conv into 64ch (no bias)
    fusion: Vec<RefineNet>,                               // 4
    out_conv1_w: F32Slice, out_conv1_b: F32Slice,         // 64→32, 3×3
    out_conv2_a_w: F32Slice, out_conv2_a_b: F32Slice,     // 32→32, 3×3
    out_conv2_b_w: F32Slice, out_conv2_b_b: F32Slice,     // 32→1, 1×1
}

impl DepthAnything {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = dir.as_ref();
        let f = SafetensorsFile::load(dir.join("model.safetensors")).map_err(|e| format!("load: {e:?}"))?;
        let g = |n: &str| slice_of(f.get(n).unwrap_or_else(|| panic!("missing {n}")));

        let blocks = (0..N_LAYERS).map(|i| {
            let p = format!("pretrained.blocks.{i}");
            VitBlock {
                norm1_w: g(&format!("{p}.norm1.weight")), norm1_b: g(&format!("{p}.norm1.bias")),
                qkv_w: g(&format!("{p}.attn.qkv.weight")), qkv_b: g(&format!("{p}.attn.qkv.bias")),
                proj_w: g(&format!("{p}.attn.proj.weight")), proj_b: g(&format!("{p}.attn.proj.bias")),
                ls1: g(&format!("{p}.ls1.gamma")),
                norm2_w: g(&format!("{p}.norm2.weight")), norm2_b: g(&format!("{p}.norm2.bias")),
                fc1_w: g(&format!("{p}.mlp.fc1.weight")), fc1_b: g(&format!("{p}.mlp.fc1.bias")),
                fc2_w: g(&format!("{p}.mlp.fc2.weight")), fc2_b: g(&format!("{p}.mlp.fc2.bias")),
                ls2: g(&format!("{p}.ls2.gamma")),
            }
        }).collect();

        let projects_w: Vec<_> = (0..4).map(|i| g(&format!("depth_head.projects.{i}.weight"))).collect();
        let projects_b: Vec<_> = (0..4).map(|i| g(&format!("depth_head.projects.{i}.bias"))).collect();
        let resize_w: Vec<Option<F32Slice>> = (0..4).map(|i| {
            if i == 2 { None } else { Some(g(&format!("depth_head.resize_layers.{i}.weight"))) }
        }).collect();
        let resize_b: Vec<Option<F32Slice>> = (0..4).map(|i| {
            if i == 2 { None } else { Some(g(&format!("depth_head.resize_layers.{i}.bias"))) }
        }).collect();
        let layer_rn_w: Vec<_> = (1..=4).map(|i| g(&format!("depth_head.scratch.layer{i}_rn.weight"))).collect();

        let fusion: Vec<_> = (1..=4).map(|i| {
            let p = format!("depth_head.scratch.refinenet{i}");
            RefineNet {
                ru1: ResConvUnit {
                    c1_w: g(&format!("{p}.resConfUnit1.conv1.weight")), c1_b: g(&format!("{p}.resConfUnit1.conv1.bias")),
                    c2_w: g(&format!("{p}.resConfUnit1.conv2.weight")), c2_b: g(&format!("{p}.resConfUnit1.conv2.bias")),
                },
                ru2: ResConvUnit {
                    c1_w: g(&format!("{p}.resConfUnit2.conv1.weight")), c1_b: g(&format!("{p}.resConfUnit2.conv1.bias")),
                    c2_w: g(&format!("{p}.resConfUnit2.conv2.weight")), c2_b: g(&format!("{p}.resConfUnit2.conv2.bias")),
                },
                out_w: g(&format!("{p}.out_conv.weight")), out_b: g(&format!("{p}.out_conv.bias")),
            }
        }).collect();

        Ok(Self {
            patch_w: g("pretrained.patch_embed.proj.weight"), patch_b: g("pretrained.patch_embed.proj.bias"),
            cls_token: g("pretrained.cls_token"), pos_embed: g("pretrained.pos_embed"),
            blocks,
            norm_w: g("pretrained.norm.weight"), norm_b: g("pretrained.norm.bias"),
            projects_w, projects_b, resize_w, resize_b, layer_rn_w, fusion,
            out_conv1_w: g("depth_head.scratch.output_conv1.weight"), out_conv1_b: g("depth_head.scratch.output_conv1.bias"),
            out_conv2_a_w: g("depth_head.scratch.output_conv2.0.weight"), out_conv2_a_b: g("depth_head.scratch.output_conv2.0.bias"),
            out_conv2_b_w: g("depth_head.scratch.output_conv2.2.weight"), out_conv2_b_b: g("depth_head.scratch.output_conv2.2.bias"),
        })
    }

    /// Run DA-V2 on a normalized image `[3, H, W]` (H, W must be multiples
    /// of 14). Returns the depth map `[H, W]`.
    pub fn infer(&self, img: &[f32], h: usize, w: usize) -> Vec<f32> {
        assert!(h % PATCH == 0 && w % PATCH == 0, "H,W must be multiples of {PATCH}");
        let ph = h / PATCH;
        let pw = w / PATCH;

        // 1) patch embed: Conv2d(3→384, k14, s14)
        let (patches, _, _) = conv2d(img, 3, h, w, &self.patch_w, D, PATCH, PATCH, PATCH, PATCH, 0, 0, Some(&self.patch_b));
        // patches: [D, ph, pw] channel-major → reshape to [N=ph*pw, D] token-major
        let n_patches = ph * pw;
        let mut tokens = vec![0.0f32; (n_patches + 1) * D];
        tokens[..D].copy_from_slice(&self.cls_token[..D]);
        for tk in 0..n_patches {
            let py = tk / pw;
            let px = tk % pw;
            for d in 0..D {
                tokens[(tk + 1) * D + d] = patches[d * ph * pw + py * pw + px];
            }
        }
        self.add_pos_embed(&mut tokens, ph, pw);

        // 2) 12 ViT blocks, capturing intermediates at indices [2,5,8,11]
        let mut x = tokens;
        let total_tokens = n_patches + 1;
        let mut intermediates: Vec<Vec<f32>> = Vec::with_capacity(4);
        for (li, blk) in self.blocks.iter().enumerate() {
            let mut xn = x.clone();
            layernorm(&mut xn, D, &blk.norm1_w, &blk.norm1_b);
            let qkv = linear(&xn, total_tokens, &blk.qkv_w, 3 * D, D, Some(&blk.qkv_b));
            let mut q = vec![0.0f32; total_tokens * D];
            let mut k = vec![0.0f32; total_tokens * D];
            let mut v = vec![0.0f32; total_tokens * D];
            for tk in 0..total_tokens {
                q[tk * D..(tk + 1) * D].copy_from_slice(&qkv[tk * 3 * D..tk * 3 * D + D]);
                k[tk * D..(tk + 1) * D].copy_from_slice(&qkv[tk * 3 * D + D..tk * 3 * D + 2 * D]);
                v[tk * D..(tk + 1) * D].copy_from_slice(&qkv[tk * 3 * D + 2 * D..tk * 3 * D + 3 * D]);
            }
            let a = attention(&q, &k, &v, total_tokens);
            let mut a = linear(&a, total_tokens, &blk.proj_w, D, D, Some(&blk.proj_b));
            for tk in 0..total_tokens { for d in 0..D { a[tk * D + d] *= blk.ls1[d]; } }
            for i in 0..x.len() { x[i] += a[i]; }

            let mut xn = x.clone();
            layernorm(&mut xn, D, &blk.norm2_w, &blk.norm2_b);
            let mut h_ff = linear(&xn, total_tokens, &blk.fc1_w, FF, D, Some(&blk.fc1_b));
            for v in h_ff.iter_mut() { *v = gelu(*v); }
            let mut m = linear(&h_ff, total_tokens, &blk.fc2_w, D, FF, Some(&blk.fc2_b));
            for tk in 0..total_tokens { for d in 0..D { m[tk * D + d] *= blk.ls2[d]; } }
            for i in 0..x.len() { x[i] += m[i]; }

            if INTER_IDX.contains(&li) { intermediates.push(x.clone()); }
        }

        // 3) DPT head: reassemble each intermediate's patch tokens (drop cls)
        let mut stage_feats: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(4);
        for (si, hidden) in intermediates.iter().enumerate() {
            let mut img_feat = vec![0.0f32; D * ph * pw];
            for tk in 0..n_patches {
                let py = tk / pw;
                let px = tk % pw;
                for d in 0..D { img_feat[d * ph * pw + py * pw + px] = hidden[(tk + 1) * D + d]; }
            }
            let (proj, _, _) = conv2d(&img_feat, D, ph, pw, &self.projects_w[si], STAGE_CH[si], 1, 1, 1, 1, 0, 0, Some(&self.projects_b[si]));
            let (resized, hr, wr) = match si {
                0 => conv_transpose2d(&proj, STAGE_CH[0], ph, pw, self.resize_w[0].as_ref().unwrap(), STAGE_CH[0], 4, 4, 4, 4, 0, 0, Some(self.resize_b[0].as_ref().unwrap())),
                1 => conv_transpose2d(&proj, STAGE_CH[1], ph, pw, self.resize_w[1].as_ref().unwrap(), STAGE_CH[1], 2, 2, 2, 2, 0, 0, Some(self.resize_b[1].as_ref().unwrap())),
                2 => (proj, ph, pw),
                3 => conv2d(&proj, STAGE_CH[3], ph, pw, self.resize_w[3].as_ref().unwrap(), STAGE_CH[3], 3, 3, 2, 2, 1, 1, Some(self.resize_b[3].as_ref().unwrap())),
                _ => unreachable!(),
            };
            let (rn, hr2, wr2) = conv2d(&resized, STAGE_CH[si], hr, wr, &self.layer_rn_w[si], FUSION, 3, 3, 1, 1, 1, 1, None);
            stage_feats.push((rn, hr2, wr2));
        }

        // 4) RefineNet fusion (deepest-first)
        let layer_sizes: Vec<(usize, usize)> = stage_feats.iter().map(|s| (s.1, s.2)).collect();
        let mut path = self.fusion_apply(&[&stage_feats[3].0], FUSION, stage_feats[3].1, stage_feats[3].2, &self.fusion[3], Some(layer_sizes[2]));
        let mut psize = layer_sizes[2];
        for stage in (0..3).rev() {
            let (rn_data, _rh, _rw) = &stage_feats[stage];
            let target_size = if stage == 0 { (psize.0 * 2, psize.1 * 2) } else { layer_sizes[stage - 1] };
            path = self.fusion_apply(&[&path, rn_data], FUSION, psize.0, psize.1, &self.fusion[stage], Some(target_size));
            psize = target_size;
        }

        // 5) output head
        let (o1, h1, w1) = conv2d(&path, FUSION, psize.0, psize.1, &self.out_conv1_w, HEAD_F2, 3, 3, 1, 1, 1, 1, Some(&self.out_conv1_b));
        let target_h = ph * PATCH;
        let target_w = pw * PATCH;
        let resized = bilinear_resize(&o1, HEAD_F2, h1, w1, target_h, target_w);
        let (o2a, h2, w2) = conv2d(&resized, HEAD_F2, target_h, target_w, &self.out_conv2_a_w, HEAD_F2, 3, 3, 1, 1, 1, 1, Some(&self.out_conv2_a_b));
        let mut o2a = o2a;
        for v in o2a.iter_mut() { *v = v.max(0.0); }
        let (mut depth, _, _) = conv2d(&o2a, HEAD_F2, h2, w2, &self.out_conv2_b_w, 1, 1, 1, 1, 1, 0, 0, Some(&self.out_conv2_b_b));
        for v in depth.iter_mut() { *v = v.max(0.0); }
        depth
    }

    /// Add (optionally interpolated) positional embeddings to tokens
    /// `[N+1, D]`. pos_embed is `[1+37², 384]`. For ph×pw ≠ 37×37, we
    /// bilinearly interpolate the patch positions.
    fn add_pos_embed(&self, tokens: &mut [f32], ph: usize, pw: usize) {
        for d in 0..D { tokens[d] += self.pos_embed[d]; }
        let n_patches = ph * pw;
        if ph == NATIVE_GRID && pw == NATIVE_GRID {
            for tk in 0..n_patches {
                for d in 0..D {
                    tokens[(tk + 1) * D + d] += self.pos_embed[(tk + 1) * D + d];
                }
            }
            return;
        }
        let mut grid = vec![0.0f32; D * NATIVE_GRID * NATIVE_GRID];
        for y in 0..NATIVE_GRID { for x in 0..NATIVE_GRID {
            let src = 1 + y * NATIVE_GRID + x;
            for d in 0..D { grid[d * NATIVE_GRID * NATIVE_GRID + y * NATIVE_GRID + x] = self.pos_embed[src * D + d]; }
        }}
        let interp = bilinear_resize(&grid, D, NATIVE_GRID, NATIVE_GRID, ph, pw);
        for y in 0..ph { for x in 0..pw {
            let tk = y * pw + x;
            for d in 0..D {
                tokens[(tk + 1) * D + d] += interp[d * ph * pw + y * pw + x];
            }
        }}
    }

    /// FeatureFusionBlock: output = out_conv(bilinear_resize(RU2(input + RU1(skip))))
    fn fusion_apply(&self, inputs: &[&Vec<f32>], c: usize, h: usize, w: usize, blk: &RefineNet, size: Option<(usize, usize)>) -> Vec<f32> {
        let mut x = inputs[0].clone();
        if inputs.len() == 2 {
            let res = res_conv_unit(inputs[1], c, h, w, &blk.ru1);
            for i in 0..x.len() { x[i] += res[i]; }
        }
        let x = res_conv_unit(&x, c, h, w, &blk.ru2);
        let (ho, wo) = size.unwrap_or((h * 2, w * 2));
        let resized = bilinear_resize(&x, c, h, w, ho, wo);
        let (oc, _, _) = conv2d(&resized, c, ho, wo, &blk.out_w, c, 1, 1, 1, 1, 0, 0, Some(&blk.out_b));
        oc
    }
}

/// ResidualConvUnit: x + Conv2d(ReLU(Conv2d(ReLU(x))))
fn res_conv_unit(x: &[f32], c: usize, h: usize, w: usize, u: &ResConvUnit) -> Vec<f32> {
    let mut t = x.to_vec();
    for v in t.iter_mut() { *v = v.max(0.0); }
    let (t2, _, _) = conv2d(&t, c, h, w, &u.c1_w, c, 3, 3, 1, 1, 1, 1, Some(&u.c1_b));
    let mut t2 = t2;
    for v in t2.iter_mut() { *v = v.max(0.0); }
    let (mut out, _, _) = conv2d(&t2, c, h, w, &u.c2_w, c, 3, 3, 1, 1, 1, 1, Some(&u.c2_b));
    for i in 0..out.len() { out[i] += x[i]; }
    out
}

/// ImageNet-normalize a `[H, W, 3]` u8 RGB image to a `[3, H, W]` f32
/// tensor (channel-major).
pub fn preprocess_rgb_u8(rgb: &[u8], h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let r = rgb[(y * w + x) * 3] as f32 / 255.0;
            let g = rgb[(y * w + x) * 3 + 1] as f32 / 255.0;
            let b = rgb[(y * w + x) * 3 + 2] as f32 / 255.0;
            out[y * w + x] = (r - IMNET_MEAN[0]) / IMNET_STD[0];
            out[h * w + y * w + x] = (g - IMNET_MEAN[1]) / IMNET_STD[1];
            out[2 * h * w + y * w + x] = (b - IMNET_MEAN[2]) / IMNET_STD[2];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_dir() -> Option<String> {
        for b in ["macos_4TB_external", "macos_4TB_external_1", "macos_4TB_external_2"] {
            let d = format!("/Volumes/{b}/neuronexus_ai/model_library/depth-anything-v2-small");
            if std::path::Path::new(&d).join("model.safetensors").exists() { return Some(d); }
        }
        None
    }

    /// Render a 224×224 image with a bright central foreground square on
    /// a dark background. Run DA-V2; assert (a) finite, (b) high variance
    /// (structured, not constant), (c) center mean depth differs from
    /// corner mean (model perceives them as distinct).
    #[test]
    fn depth_anything_two_region() {
        let Some(dir) = model_dir() else { eprintln!("depth-anything: not mounted — skipping"); return };
        const SZ: usize = 224;
        let mut rgb = vec![20u8; SZ * SZ * 3];
        let (cx, cy, half) = (SZ / 2, SZ / 2, 40);
        for y in (cy - half)..(cy + half) {
            for x in (cx - half)..(cx + half) {
                rgb[(y * SZ + x) * 3] = 220;
                rgb[(y * SZ + x) * 3 + 1] = 220;
                rgb[(y * SZ + x) * 3 + 2] = 220;
            }
        }
        let img = preprocess_rgb_u8(&rgb, SZ, SZ);

        let m = DepthAnything::load(&dir).expect("load");
        let t0 = std::time::Instant::now();
        let depth = m.infer(&img, SZ, SZ);
        eprintln!("depth-anything: {SZ}×{SZ} in {:?}, out len {}", t0.elapsed(), depth.len());

        assert_eq!(depth.len(), SZ * SZ);
        assert!(depth.iter().all(|v| v.is_finite()), "non-finite values in depth map");
        let mean = depth.iter().sum::<f32>() / depth.len() as f32;
        let var = depth.iter().map(|v| { let z = v - mean; z * z }).sum::<f32>() / depth.len() as f32;
        eprintln!("depth stats: mean={mean:.4} var={var:.4} min={:.4} max={:.4}",
            depth.iter().cloned().fold(f32::INFINITY, f32::min),
            depth.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        assert!(var > 1e-4, "depth map looks constant — model didn't engage (var {var})");

        let mut center_sum = 0.0f32; let mut center_n = 0;
        for y in (cy - 20)..(cy + 20) { for x in (cx - 20)..(cx + 20) {
            center_sum += depth[y * SZ + x]; center_n += 1;
        }}
        let center_mean = center_sum / center_n as f32;
        let mut corner_sum = 0.0f32; let mut corner_n = 0;
        for y in 0..30 { for x in 0..30 { corner_sum += depth[y * SZ + x]; corner_n += 1; } }
        let corner_mean = corner_sum / corner_n as f32;
        eprintln!("center={center_mean:.4}  corner={corner_mean:.4}  |Δ|={:.4}",
            (center_mean - corner_mean).abs());
        assert!((center_mean - corner_mean).abs() > 0.01,
            "center and corner depth indistinguishable — degenerate output");
    }
}
