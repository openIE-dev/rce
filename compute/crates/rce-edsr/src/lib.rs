//! EDSR baseline x2 super-resolution — rce's first model built from a
//! converted ONNX (see `joulesperbit/tools/convert_onnx.py`).
//!
//! Architecture (verified against
//! `https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/src/model/edsr.py`):
//!
//! - `sub_mean`: Conv2d(3, 3, k=1, bias=True) — RGB mean subtract.
//! - `head.0`: Conv2d(3, 64, k=3, p=1) — initial feature extraction.
//! - `body.0..15`: 16 residual blocks, each
//!     `r = ReLU(conv1(x)) → conv2 → x + r * res_scale`
//!   (baseline uses `res_scale=1.0`).
//! - `body.16`: Conv2d(64, 64, k=3, p=1) — final body conv before global skip.
//! - `tail.0.0`: Conv2d(64, 256, k=3, p=1) — pre-pixel-shuffle expansion.
//! - **PixelShuffle(scale=2)**: [256, H, W] → [64, 2H, 2W].
//! - `tail.1`: Conv2d(64, 3, k=3, p=1) — final RGB head.
//! - `add_mean`: Conv2d(3, 3, k=1, bias=True) — RGB mean re-add.  The ONNX
//!   export folds the identity weight away, leaving only `add_mean.bias` —
//!   apply as channel-wise add.
//!
//! Input: RGB image in `[0, 255]` range, channel-first `[3, H, W]`.
//! Output: super-resolved `[3, 2H, 2W]` in the same range.
//!
//! Weight provenance: produced by
//!     python3 tools/convert_onnx.py model_library/edsr/edsr_baseline_x2.onnx
//! and consumed via the borrow-direct [`F32Slice`] loader.

use rce_op_conv2d::conv2d;
use rce_weight_loader::{F32Slice, SafetensorsFile};
use std::path::Path;

pub const N_CHANNELS: usize = 64;     // feature width
pub const N_BLOCKS: usize = 16;
pub const SCALE: usize = 2;
pub const RES_SCALE: f32 = 1.0;       // baseline EDSR

#[derive(Debug)]
pub enum EdsrError { Load(String) }
impl std::fmt::Display for EdsrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::Load(m) => write!(f, "edsr load: {m}") }
    }
}
impl std::error::Error for EdsrError {}

fn load(sf: &SafetensorsFile, name: &str) -> Result<F32Slice, EdsrError> {
    let t = sf.get(name).ok_or_else(|| EdsrError::Load(format!("missing {name}")))?;
    F32Slice::from_tensor(name, t).map_err(EdsrError::Load)
}

// conv2d lives in `rce-op-conv2d` — shared NEON 3×3 kernel + general fallback.
//
// (Earlier inline NEON kernel preserved below as documentation of the pattern
// but commented out; the real call goes through `rce_op_conv2d::conv2d`.)

/*
/// NEON 3×3 same-padding interior conv: vectorize 4 output xs per FMA.
///
/// For each (oc, yo, xo+0..3), accumulate `sum_{ic, ky, kx} inp[ic, yo+ky-1, xo+kx-1] * w[oc, ic, ky, kx]`.
/// The weight `w[oc, ic, ky, kx]` is a scalar broadcast across the 4 output
/// lanes; the input `inp[ic, yi, xi..xi+4]` is a contiguous `vld1q_f32`.
///
/// The kernel is invoked only on the *interior* region where all 9 (ky, kx)
/// taps land inside `[0, h)×[0, w)`, so no bounds checks are needed.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn conv3x3_interior_neon(
    inp: &[f32], cin: usize, _h: usize, w: usize,
    weight: &[f32], oc: usize,
    out: &mut [f32], oc_plane_off: usize,
    yo_lo: usize, yo_hi: usize, xo_lo: usize, xo_hi: usize,
    bias_oc: f32,
) {
    use std::arch::aarch64::*;
    let bias_v = vdupq_n_f32(bias_oc);
    for yo in yo_lo..yo_hi {
        let mut xo = xo_lo;
        // 4-wide vectorized core
        while xo + 4 <= xo_hi {
            let mut acc = bias_v;
            for ic in 0..cin {
                let w_base = ((oc * cin) + ic) * 9;
                let in_off = ic * _h * w;
                for ky in 0..3 {
                    let yi = yo + ky - 1;
                    let row_base = in_off + yi * w + (xo - 1);
                    // 3 contiguous input loads, each shifted by kx:
                    let v0 = vld1q_f32(inp.as_ptr().add(row_base));
                    let v1 = vld1q_f32(inp.as_ptr().add(row_base + 1));
                    let v2 = vld1q_f32(inp.as_ptr().add(row_base + 2));
                    let w_row = w_base + ky * 3;
                    acc = vfmaq_n_f32(acc, v0, weight[w_row]);
                    acc = vfmaq_n_f32(acc, v1, weight[w_row + 1]);
                    acc = vfmaq_n_f32(acc, v2, weight[w_row + 2]);
                }
            }
            vst1q_f32(out.as_mut_ptr().add(oc_plane_off + yo * w + xo), acc);
            xo += 4;
        }
        // Scalar tail for remaining xo in [xo, xo_hi)
        while xo < xo_hi {
            let mut acc = bias_oc;
            for ic in 0..cin {
                let w_base = ((oc * cin) + ic) * 9;
                let in_off = ic * _h * w;
                for ky in 0..3 {
                    let yi = yo + ky - 1;
                    let row = in_off + yi * w + (xo - 1);
                    let w_row = w_base + ky * 3;
                    acc += inp[row]     * weight[w_row]
                         + inp[row + 1] * weight[w_row + 1]
                         + inp[row + 2] * weight[w_row + 2];
                }
            }
            out[oc_plane_off + yo * w + xo] = acc;
            xo += 1;
        }
    }
}

/// 2D convolution with stride 1.  `inp` is channel-first `[C, H, W]`.
/// Output is also channel-first `[Cout, H, W]` (stride-1 + same padding).
///
/// **Three-way path selection:**
/// 1. K=3, pad=1, aarch64: NEON 4-wide interior kernel + scalar border (the
///    EDSR hot path; ~6× faster than scalar interior).
/// 2. K=3, pad=1, other arches: scalar interior with no bounds checks.
/// 3. Other shapes: general bounds-checked path (small models, sub_mean 1×1).
fn conv2d(inp: &[f32], cin: usize, h: usize, w: usize,
          weight: &[f32], cout: usize, k: usize, pad: usize,
          bias: Option<&[f32]>) -> Vec<f32> {
    let mut out = vec![0.0f32; cout * h * w];
    let lo = pad;
    let hi_y = if h + pad >= k { h + pad - (k - 1) } else { 0 };
    let hi_x = if w + pad >= k { w + pad - (k - 1) } else { 0 };
    let is_3x3_pad1 = k == 3 && pad == 1;

    out.par_chunks_mut(h * w).enumerate().for_each(|(oc, oc_plane)| {
        let bb = bias.map_or(0.0, |b| b[oc]);

        // ── Interior path ──
        if is_3x3_pad1 && hi_y > lo && hi_x > lo {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                conv3x3_interior_neon(
                    inp, cin, h, w, weight, oc,
                    oc_plane, 0,                 // oc_plane is the per-channel slice
                    lo, hi_y, lo, hi_x, bb,
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                // Scalar interior: init to bias then per-ic accumulate.
                for yo in lo..hi_y { for xo in lo..hi_x {
                    let mut acc = bb;
                    for ic in 0..cin {
                        let in_off = ic * h * w;
                        let w_base = ((oc * cin) + ic) * 9;
                        for ky in 0..3 {
                            let yi = yo + ky - 1;
                            let row = in_off + yi * w + (xo - 1);
                            acc += inp[row]     * weight[w_base + ky * 3]
                                 + inp[row + 1] * weight[w_base + ky * 3 + 1]
                                 + inp[row + 2] * weight[w_base + ky * 3 + 2];
                        }
                    }
                    oc_plane[yo * w + xo] = acc;
                }}
            }
        } else if hi_y > lo && hi_x > lo {
            // General-K scalar interior (no bounds checks)
            for v in oc_plane.iter_mut() { *v = bb; }
            for ic in 0..cin {
                let w_off = ((oc * cin) + ic) * k * k;
                let in_off = ic * h * w;
                for yo in lo..hi_y { for xo in lo..hi_x {
                    let mut acc = 0.0f32;
                    for ky in 0..k {
                        let yi = yo + ky - pad;
                        let row = in_off + yi * w;
                        let wr = w_off + ky * k;
                        for kx in 0..k {
                            acc += inp[row + (xo + kx - pad)] * weight[wr + kx];
                        }
                    }
                    oc_plane[yo * w + xo] += acc;
                }}
            }
        }

        // ── Border path (handles k=1 conv too — sub_mean / add_mean: the
        // interior path is empty for k=1 because hi_*=lo, so this loop runs
        // over the whole plane). ──
        for yo in 0..h { for xo in 0..w {
            // Skip interior already filled by 3×3 path.
            if is_3x3_pad1 && yo >= lo && yo < hi_y && xo >= lo && xo < hi_x { continue; }
            // For non-3×3 interior, the initial fill loop above already set bias + interior
            // accumulation; skip those positions to avoid double-counting.
            if !is_3x3_pad1 && hi_y > lo && hi_x > lo && yo >= lo && yo < hi_y && xo >= lo && xo < hi_x { continue; }
            let mut acc = bb;
            for ic in 0..cin {
                let in_off = ic * h * w;
                let w_off = ((oc * cin) + ic) * k * k;
                for ky in 0..k { for kx in 0..k {
                    let yi = (yo + ky) as isize - pad as isize;
                    let xi = (xo + kx) as isize - pad as isize;
                    if yi >= 0 && yi < h as isize && xi >= 0 && xi < w as isize {
                        acc += inp[in_off + (yi as usize) * w + (xi as usize)]
                             * weight[w_off + ky * k + kx];
                    }
                }}
            }
            oc_plane[yo * w + xo] = acc;
        }}
    });
    out
}
*/

/// PixelShuffle with upscale_factor=r: `[C*r², H, W]` → `[C, H*r, W*r]`.
/// PyTorch's `nn.PixelShuffle` channel-shuffle convention:
///     out[c, y*r + dy, x*r + dx] = in[c*r² + dy*r + dx, y, x]
fn pixel_shuffle(inp: &[f32], cin: usize, h: usize, w: usize, r: usize) -> Vec<f32> {
    assert_eq!(cin % (r * r), 0, "PixelShuffle: cin {cin} not divisible by r²={}", r * r);
    let cout = cin / (r * r);
    let ho = h * r;
    let wo = w * r;
    let mut out = vec![0.0f32; cout * ho * wo];
    for c in 0..cout {
        for dy in 0..r { for dx in 0..r {
            let src_c = c * r * r + dy * r + dx;
            for y in 0..h { for x in 0..w {
                let yo = y * r + dy;
                let xo = x * r + dx;
                out[c * ho * wo + yo * wo + xo] = inp[src_c * h * w + y * w + x];
            }}
        }}
    }
    out
}

#[inline] fn relu_inplace(v: &mut [f32]) { for x in v.iter_mut() { if *x < 0.0 { *x = 0.0; } } }

struct ResBlock {
    c1_w: F32Slice, c1_b: F32Slice,
    c2_w: F32Slice, c2_b: F32Slice,
}
impl ResBlock {
    fn load(sf: &SafetensorsFile, i: usize) -> Result<Self, EdsrError> {
        let p = format!("body.{i}.body");
        Ok(Self {
            c1_w: load(sf, &format!("{p}.0.weight"))?,
            c1_b: load(sf, &format!("{p}.0.bias"))?,
            c2_w: load(sf, &format!("{p}.2.weight"))?,
            c2_b: load(sf, &format!("{p}.2.bias"))?,
        })
    }
    fn forward(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let mut r = conv2d(x, N_CHANNELS, h, w, &self.c1_w, N_CHANNELS, 3, 1, Some(&self.c1_b));
        relu_inplace(&mut r);
        let r = conv2d(&r, N_CHANNELS, h, w, &self.c2_w, N_CHANNELS, 3, 1, Some(&self.c2_b));
        let mut out = vec![0.0f32; x.len()];
        for i in 0..x.len() { out[i] = x[i] + r[i] * RES_SCALE; }
        out
    }
}

pub struct Edsr {
    sub_w: F32Slice, sub_b: F32Slice,            // [3,3,1,1] [3]
    add_b: F32Slice,                             // [3]   — add_mean.weight folded into bias
    head_w: F32Slice, head_b: F32Slice,          // [64,3,3,3] [64]
    blocks: Vec<ResBlock>,
    body_last_w: F32Slice, body_last_b: F32Slice,    // [64,64,3,3] [64]
    tail0_w: F32Slice, tail0_b: F32Slice,        // [256,64,3,3] [256]
    tail1_w: F32Slice, tail1_b: F32Slice,        // [3,64,3,3] [3]
}

impl Edsr {
    pub fn load(path: &Path) -> Result<Self, EdsrError> {
        let sf = SafetensorsFile::load(path).map_err(|e| EdsrError::Load(e.to_string()))?;
        let blocks = (0..N_BLOCKS).map(|i| ResBlock::load(&sf, i)).collect::<Result<_, _>>()?;
        Ok(Self {
            sub_w: load(&sf, "sub_mean.weight")?,
            sub_b: load(&sf, "sub_mean.bias")?,
            add_b: load(&sf, "add_mean.bias")?,
            head_w: load(&sf, "head.0.weight")?,
            head_b: load(&sf, "head.0.bias")?,
            blocks,
            body_last_w: load(&sf, "body.16.weight")?,
            body_last_b: load(&sf, "body.16.bias")?,
            tail0_w: load(&sf, "tail.0.0.weight")?,
            tail0_b: load(&sf, "tail.0.0.bias")?,
            tail1_w: load(&sf, "tail.1.weight")?,
            tail1_b: load(&sf, "tail.1.bias")?,
        })
    }

    /// Run EDSR x2 on a channel-first RGB image `[3, H, W]` in `[0, 255]`.
    /// Returns `[3, 2H, 2W]` in the same range.
    pub fn infer(&self, img: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        assert_eq!(img.len(), 3 * h * w, "expected [3, {h}, {w}] = {} elems, got {}", 3 * h * w, img.len());

        // 1) sub_mean: 1×1 conv with identity-init weight (sign=-1) + mean bias.
        let x = conv2d(img, 3, h, w, &self.sub_w, 3, 1, 0, Some(&self.sub_b));

        // 2) head.0: 3×3 conv 3 → 64.
        let head = conv2d(&x, 3, h, w, &self.head_w, N_CHANNELS, 3, 1, Some(&self.head_b));

        // 3) 16 residual blocks.
        let mut feat = head.clone();
        for blk in &self.blocks { feat = blk.forward(&feat, h, w); }

        // 4) body.16 + global skip (back to `head`).
        let body_out = conv2d(&feat, N_CHANNELS, h, w, &self.body_last_w, N_CHANNELS, 3, 1, Some(&self.body_last_b));
        let mut sum = vec![0.0f32; head.len()];
        for i in 0..head.len() { sum[i] = head[i] + body_out[i]; }

        // 5) tail.0.0: 3×3 conv 64 → 256 (=64 * scale²).
        let t0 = conv2d(&sum, N_CHANNELS, h, w, &self.tail0_w, N_CHANNELS * SCALE * SCALE, 3, 1, Some(&self.tail0_b));

        // 6) PixelShuffle: [256, H, W] → [64, 2H, 2W].
        let ps = pixel_shuffle(&t0, N_CHANNELS * SCALE * SCALE, h, w, SCALE);
        let ho = h * SCALE;
        let wo = w * SCALE;

        // 7) tail.1: 3×3 conv 64 → 3.
        let mut rgb = conv2d(&ps, N_CHANNELS, ho, wo, &self.tail1_w, 3, 3, 1, Some(&self.tail1_b));

        // 8) add_mean: identity 1×1 conv folded into channel bias.
        for c in 0..3 {
            for i in 0..(ho * wo) { rgb[c * ho * wo + i] += self.add_b[c]; }
        }
        (rgb, ho, wo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn weight_path() -> Option<PathBuf> {
        let p = PathBuf::from("/tmp/onnx_sweep/edsr_baseline_x2.safetensors");
        if p.exists() { Some(p) } else { None }
    }

    /// Constant 128.0 input — output should be constant ~128 in interior.
    /// This tests numerical equivalence to ORT without edge-effect noise.
    #[test]
    fn edsr_constant_input() {
        let Some(p) = weight_path() else { eprintln!("skip"); return; };
        let m = Edsr::load(&p).expect("load");
        let h = 16; let w = 16;
        let img = vec![128.0f32; 3 * h * w];
        let (out, ho, wo) = m.infer(&img, h, w);
        // Interior: [4..28, 4..28]
        let mut interior = Vec::new();
        for c in 0..3 { for y in 4..(ho-4) { for x in 4..(wo-4) {
            interior.push(out[c * ho * wo + y * wo + x]);
        }}}
        let mean: f32 = interior.iter().sum::<f32>() / interior.len() as f32;
        let mn = interior.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = interior.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("constant-128 interior: [{mn:.4}, {mx:.4}] mean={mean:.4}");
        // ORT reference on the same input: interior mean 128.04, range [127.62, 128.70].
        assert!((mean - 128.04).abs() < 0.5,
                "mean {mean:.4} too far from ORT 128.04");
        assert!(mn > 127.0 && mx < 129.0, "interior out of ORT range");
    }

    /// Pure plumbing test: load weights, run a 16×16 RGB toy image, assert
    /// output shape is 32×32 and values are finite + non-trivially structured.
    #[test]
    fn edsr_smoke() {
        let Some(p) = weight_path() else { eprintln!("skip: run tools/convert_onnx.py first"); return; };
        let m = Edsr::load(&p).expect("load");
        // Construct a synthetic RGB image: vertical bars 0..255.
        let h = 16; let w = 16;
        let mut img = vec![0.0f32; 3 * h * w];
        for c in 0..3 { for y in 0..h { for x in 0..w {
            img[c * h * w + y * w + x] = (x as f32 / w as f32) * 255.0;
        }}}
        let (out, ho, wo) = m.infer(&img, h, w);
        assert_eq!(ho, 32);
        assert_eq!(wo, 32);
        assert_eq!(out.len(), 3 * 32 * 32);
        let finite = out.iter().all(|v| v.is_finite());
        assert!(finite, "non-finite output");
        let max_v = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_v = out.iter().cloned().fold(f32::INFINITY, f32::min);
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        eprintln!("EDSR smoke: out=[{ho}×{wo}×3] range [{min_v:.2}, {max_v:.2}] mean={mean:.2}");
        // Structured output: max-min spread should be substantial (not constant).
        assert!(max_v - min_v > 50.0, "output too flat: spread {}", max_v - min_v);

        // Dump to /tmp for ORT comparison.
        let mut bytes = Vec::with_capacity(out.len() * 4);
        for v in &out { bytes.extend_from_slice(&v.to_le_bytes()); }
        std::fs::write("/tmp/edsr_rce_out.f32", &bytes).expect("write");
    }
}
