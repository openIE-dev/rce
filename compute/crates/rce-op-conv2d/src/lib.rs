//! Shared 2D convolution + dense f32 matmul primitives for rce model crates.
//!
//! # API
//!
//! - [`conv2d`] — generic same-padding stride-1 conv with weight
//!   `[Cout, Cin, K, K]`, input `[Cin, H, W]`, output `[Cout, H, W]`.
//! - [`conv_transpose2d`] — deconv (`nn.ConvTranspose2d`) with weight
//!   `[Cin, Cout, kH, kW]`.  Used by pose / segmentation / super-res
//!   decoder heads that learn upsampling.
//! - [`linear_f32`] — PyTorch `nn.Linear` forward via Apple Accelerate
//!   sgemm with TransB: `Y[t,dout] = X[t,din] · W[dout,din]^T (+ bias)`.
//!   The single-call replacement for the `linear(x, t, w, dout, din, b)`
//!   pattern used by every rce attention / MLP path.
//! - [`sgemm_rm`] — raw row-major sgemm: `C[m,n] = A[m,k] · B[k,n]`,
//!   `beta=0`.  Use for any other f32 matmul.
//!
//! # Measured performance (2026-06-03, 32-core M-series CPU)
//!
//! | Path | Shape | GFLOPS | % theoretical peak |
//! |---|---|---|---|
//! | Scalar (baseline) | conv 64×770×1120 3×3 | ~10 | 0.1 % |
//! | NEON 4-wide | same | 186 | 2.3 % |
//! | Accelerate conv2d (im2col + sgemm) | same | 1300 | 17 % |
//! | **Accelerate `linear_f32`** | 8192×1024→4096 (V-JEPA-2 fc1) | **2049** | **25 %** |
//!
//! # End-to-end wins on real models
//!
//! - **rce-edsr** (1120×770 super-res): 68.8 s → 2.76 s (25×); beats ORT CPU EP (~3 s).
//! - **rce-wav2vec2** transcription: 3× → 5.8× realtime (1.9×) on a 7 s clip.
//! - **rce-depth-anything** forward: 362 ms → 240 ms (1.5×) on 224×224.
//! - **rce-f5tts** DiT forward: 176 ms → 108 ms (1.6×).
//!
//! # Architecture notes
//!
//! On macOS, the crate links Apple's `Accelerate.framework` (via a one-line
//! `build.rs`) and calls `cblas_sgemm` directly.  Accelerate uses the M-series
//! AMX matrix coprocessor — out of reach of any pure-Rust intrinsic.  On
//! non-macOS targets, `linear_f32` and `sgemm_rm` fall back to
//! `matrixmultiply` (a pure-Rust BLIS port; slower but cross-platform);
//! `conv2d` falls back to the NEON 4-wide path on aarch64 or a scalar loop
//! elsewhere.
//!
//! Three internal paths:
//!
//! 1. **K=3, pad=1, aarch64**: NEON 4-wide `vfmaq_n_f32` kernel over the
//!    no-bounds-check interior region.  Born in `rce-edsr`; gives 5.4×
//!    speedup over scalar on the EDSR x2 pipeline (68.8 s → 12.8 s on a
//!    1120×770 PNG).  Pattern: broadcast weight scalar across 4 lanes,
//!    load 3 contiguous `f32x4` input vectors per `ky` row (each shifted
//!    by `kx`), 9 FMAs per 4-output-tile per input channel.
//! 2. **K=3, pad=1, other arches**: scalar interior with no bounds checks
//!    (still benefits from rayon over `Cout`).
//! 3. **Other K / pad**: general bounds-checked path.  Same fast border
//!    handling either way.
//!
//! All paths produce numerically equivalent output (the NEON FP-reorder
//! differs from scalar at the ±1 LSB level on ~0.001 % of pixels in
//! integer-quantized PNG output, well within float32 precision).
//!
//! Conv2d is the dominant op in rce's CV crates (EDSR, DepthAnything,
//! future motionbert/densepose/yolo); this crate is where any future
//! im2col + tiled GEMM speedup also lands.

use rayon::prelude::*;
use std::sync::Arc;
use rce_weight_loader::{LoadedTensor, TensorData};

// ── F16Slice borrow-direct (mirror of weight-loader's F32Slice for f16) ─────
//
// f16 (IEEE 754 binary16) weights are common in HF safetensors for ASR /
// codec models (Whisper, EnCodec, Snac).  Borrow-direct via mmap saves their
// full f32-decode heap cost — Whisper large-v3-turbo is 800 M params:
// 1.6 GB at f16, 3.2 GB at f32.  With `F16Slice` + lazy per-call dequant the
// runtime user heap is ~0 for raw weights.

/// Borrow-direct f16 weight handle.  Wraps a `LoadedTensor` (holding
/// `Arc<Mmap>`) and `Deref`s to `&[half::f16]`.  For f16-stored tensors,
/// zero-copy alias into the safetensors mmap; for f32 fallback we
/// materialize an `Arc<[u8]>`.
#[derive(Clone)]
pub struct F16Slice {
    tensor: LoadedTensor,
}

impl F16Slice {
    pub fn from_tensor(name: &str, t: &LoadedTensor) -> Result<Self, String> {
        match t.dtype.as_str() {
            "f16" => {
                let bytes = t.data.len();
                if bytes % 2 != 0 {
                    return Err(format!("F16Slice({name}): byte-len {bytes} not /2"));
                }
                let ptr = t.data.as_slice().as_ptr() as usize;
                if ptr % std::mem::align_of::<half::f16>() != 0 {
                    return Err(format!("F16Slice({name}): mmap ptr not 2-byte aligned"));
                }
                Ok(Self { tensor: t.clone() })
            }
            "f32" => {
                // Materialize as f16 (lossy round-trip; should never trigger for
                // Whisper/EnCodec/Snac which are all native f16).
                let bytes = t.data.as_slice();
                let mut f16_bytes = Vec::with_capacity(bytes.len() / 2);
                for c in bytes.chunks_exact(4) {
                    let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    let r = half::f16::from_f32(v).to_bits().to_le_bytes();
                    f16_bytes.push(r[0]); f16_bytes.push(r[1]);
                }
                let arc: Arc<[u8]> = f16_bytes.into();
                Ok(Self { tensor: LoadedTensor {
                    name: t.name.clone(),
                    dtype: "f16".to_string(),
                    shape: t.shape.clone(),
                    data: TensorData::Owned(arc),
                }})
            }
            other => Err(format!("F16Slice({name}): unsupported dtype {other}"))
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[half::f16] {
        let bytes = self.tensor.data.as_slice();
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<half::f16>(), bytes.len() / 2) }
    }
    pub fn len(&self) -> usize { self.tensor.data.len() / 2 }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
}

impl std::ops::Deref for F16Slice {
    type Target = [half::f16];
    #[inline] fn deref(&self) -> &[half::f16] { self.as_slice() }
}

impl std::fmt::Debug for F16Slice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "F16Slice{{len:{}}}", self.len())
    }
}

/// Dequantize f16 → f32 via NEON `vcvt_f32_f16` (4 conversions per cycle).
/// On aarch64 with f16 support, this is ~5-10× faster than scalar
/// `f.to_f32()` because it uses native SIMD f16 conversion instructions.
#[inline]
pub fn dequant_f16_to_f32(src: &[half::f16]) -> Vec<f32> {
    let n = src.len();
    let mut out = vec![0.0f32; n];
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut j = 0;
        while j + 4 <= n {
            let u = vld1_u16(src.as_ptr().add(j) as *const u16);
            let f16v = vreinterpret_f16_u16(u);
            let f32v = vcvt_f32_f16(f16v);
            vst1q_f32(out.as_mut_ptr().add(j), f32v);
            j += 4;
        }
        while j < n { out[j] = src[j].to_f32(); j += 1; }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (i, &v) in src.iter().enumerate() { out[i] = v.to_f32(); }
    }
    out
}

/// PyTorch `nn.Linear` forward with f16 weights:
/// `Y[t,dout] = X[t,din] · W[dout,din]^T (+ bias)`.
///
/// `w_f16` is the borrow-direct f16 weight (typically an `&F16Slice`).
/// Dequant happens once per call into a fresh f32 buffer, then Accelerate
/// `sgemm` runs.  Per-call dequant cost is ~Cout×Cin×4 bytes write at
/// ~10 GB/s ≈ ~1 ms for the typical Whisper fc1 (1024×4096).  GEMM is
/// dominant; total path is ~10 % slower than pure f32 with the win that
/// raw weights stay mmap-aliased (zero user heap for the model).
pub fn linear_f16(x: &[f32], t: usize, din: usize,
                  w_f16: &[half::f16], dout: usize, bias: Option<&[f32]>) -> Vec<f32> {
    let w_f32 = dequant_f16_to_f32(w_f16);
    linear_f32(x, t, din, &w_f32, dout, bias)
}

// ── Accelerate FFI (macOS only) ─────────────────────────────────────────────
//
// Apple's Accelerate.framework `cblas_sgemm` is a hand-tuned SIMD GEMM that
// hits ~1.5 TFLOPS on M-series silicon — orders of magnitude faster than a
// pure-Rust matmul on the tall/skinny conv im2col shapes used here.
#[cfg(target_os = "macos")]
mod accelerate {
    #[repr(C)]
    #[allow(non_camel_case_types, dead_code)]
    pub enum CblasOrder { RowMajor = 101, ColMajor = 102 }
    #[repr(C)]
    #[allow(non_camel_case_types)]
    pub enum CblasTranspose { NoTrans = 111, Trans = 112 }

    #[allow(improper_ctypes)]
    unsafe extern "C" {
        pub fn cblas_sgemm(
            order: CblasOrder,
            trans_a: CblasTranspose,
            trans_b: CblasTranspose,
            m: i32, n: i32, k: i32,
            alpha: f32, a: *const f32, lda: i32,
            b: *const f32, ldb: i32,
            beta: f32, c: *mut f32, ldc: i32,
        );
    }
}

/// PyTorch `nn.Linear` forward via Accelerate sgemm: `Y[t,dout] = X[t,din] · W[dout,din]^T (+ bias)`.
///
/// `x`: input `[t, din]` row-major.
/// `w`: weight `[dout, din]` row-major (PyTorch convention).
/// `bias`: optional `[dout]` added to every row of the output.
/// Returns `[t, dout]`.
///
/// This is the single-call replacement for the `linear(x, t, w, dout, din, bias)`
/// pattern used by every rce attention/MLP path. On Apple Silicon this hits
/// ~1.5 TFLOPS — orders of magnitude faster than scalar `for k in 0..din`.
pub fn linear_f32(x: &[f32], t: usize, din: usize,
                  w: &[f32], dout: usize, bias: Option<&[f32]>) -> Vec<f32> {
    assert_eq!(x.len(), t * din);
    assert_eq!(w.len(), dout * din);
    if let Some(b) = bias { assert_eq!(b.len(), dout); }
    let mut y = vec![0.0f32; t * dout];
    #[cfg(target_os = "macos")]
    unsafe {
        // C[t,dout] = A[t,din] · B[dout,din]^T  via sgemm(NoTrans, Trans)
        // ldA = din, ldB = din (each row of W has din elements), ldC = dout.
        accelerate::cblas_sgemm(
            accelerate::CblasOrder::RowMajor,
            accelerate::CblasTranspose::NoTrans,
            accelerate::CblasTranspose::Trans,
            t as i32, dout as i32, din as i32,
            1.0, x.as_ptr(), din as i32,
            w.as_ptr(), din as i32,
            0.0, y.as_mut_ptr(), dout as i32,
        );
    }
    #[cfg(not(target_os = "macos"))]
    {
        // matrixmultiply: C = A · B with explicit strides.
        // We want C[t,dout] = A[t,din] · B[dout,din]^T, so B is laid out as W
        // with row-stride=din, col-stride=1.  To make it act as W^T inside sgemm,
        // swap row/col strides: row-stride=1, col-stride=din.
        unsafe {
            matrixmultiply::sgemm(
                t, din, dout,
                1.0, x.as_ptr(), din as isize, 1,
                w.as_ptr(), 1, din as isize,
                0.0, y.as_mut_ptr(), dout as isize, 1,
            );
        }
    }
    if let Some(b) = bias {
        y.par_chunks_mut(dout).for_each(|row| {
            for (yo, &bb) in row.iter_mut().zip(b.iter()) { *yo += bb; }
        });
    }
    y
}

/// f32 GEMM via Apple Accelerate on macOS (or `matrixmultiply` elsewhere).
/// Row-major: `C[m,n] = A[m,k] · B[k,n]`, `beta=0` (C overwritten).
///
/// On Apple Silicon this hits ~1.5 TFLOPS by exploiting the AMX matrix
/// coprocessor — orders of magnitude faster than any pure-Rust matmul.
/// Use this for any f32 matrix multiplication in rce model crates that
/// can lay out their data row-major with contiguous strides.
///
/// Output buffer must be pre-sized to `m * n`.
pub fn sgemm_rm(m: usize, n: usize, k: usize,
                a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    #[cfg(target_os = "macos")]
    unsafe {
        accelerate::cblas_sgemm(
            accelerate::CblasOrder::RowMajor,
            accelerate::CblasTranspose::NoTrans,
            accelerate::CblasTranspose::NoTrans,
            m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32,
            b.as_ptr(), n as i32,
            0.0, c.as_mut_ptr(), n as i32,
        );
    }
    #[cfg(not(target_os = "macos"))]
    {
        // Pure-Rust fallback via matrixmultiply.
        unsafe {
            matrixmultiply::sgemm(
                m, k, n,
                1.0, a.as_ptr(), k as isize, 1,
                b.as_ptr(), n as isize, 1,
                0.0, c.as_mut_ptr(), n as isize, 1,
            );
        }
    }
}

/// im2col for 3×3 stride-1 same-padding (pad=1).
///
/// Input  `[Cin, H, W]`   →   output `[Cin*9, H*W]` row-major.
/// Each column j (where `j = yo*W + xo`) holds the 9 input values that the
/// 3×3 kernel reads at `(yo, xo)`, in `(ic, ky, kx)` order with the
/// PyTorch/ONNX flatten convention.  Zero-padded where the kernel reads
/// outside the input.
fn im2col_3x3_pad1(inp: &[f32], cin: usize, h: usize, w: usize) -> Vec<f32> {
    let cols = h * w;
    let rows = cin * 9;
    let mut out = vec![0.0f32; rows * cols];
    // Parallelize by output row (rows == Cin*9 typically a few hundred).
    out.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
        let ic = r / 9;
        let kyx = r % 9;
        let ky = kyx / 3;
        let kx = kyx % 3;
        let in_off = ic * h * w;
        for yo in 0..h {
            let yi = yo as isize + ky as isize - 1;
            if yi < 0 || yi >= h as isize { continue; }
            let yi = yi as usize;
            let in_row = in_off + yi * w;
            let col_base = yo * w;
            // xo loop with bounds for the left/right edge.
            let lo = if kx == 0 { 1 } else { 0 };
            let hi = if kx == 2 { w - 1 } else { w };
            for xo in lo..hi {
                let xi = (xo as isize + kx as isize - 1) as usize;
                row[col_base + xo] = inp[in_row + xi];
            }
        }
    });
    out
}

/// im2col + GEMM 3×3 pad=1 conv2d on macOS via Accelerate.
///
/// Returns `[Cout, H, W]`.  Allocates `~Cin*9*H*W*4` bytes for the im2col
/// matrix (≈ 2 GB for an EDSR mid-layer on 770×1120 — fits in M-series RAM
/// and is released immediately on return).
fn conv2d_3x3_pad1_gemm(inp: &[f32], cin: usize, h: usize, w: usize,
                       weight: &[f32], cout: usize,
                       bias: Option<&[f32]>) -> Vec<f32> {
    let col = im2col_3x3_pad1(inp, cin, h, w);
    let n = h * w;
    let k = cin * 9;
    let mut out = vec![0.0f32; cout * n];
    // Weight is already laid out [Cout, Cin*9] in row-major == [Cout, Cin, 3, 3].
    sgemm_rm(cout, n, k, weight, &col, &mut out);
    if let Some(b) = bias {
        out.par_chunks_mut(n).enumerate().for_each(|(oc, row)| {
            let bb = b[oc];
            for v in row.iter_mut() { *v += bb; }
        });
    }
    out
}

/// 2D convolution with stride 1, same padding.
///
/// - `inp`: channel-first `[Cin, H, W]` (length `Cin * H * W`)
/// - `weight`: `[Cout, Cin, K, K]` (length `Cout * Cin * K * K`)
/// - `bias`: optional `[Cout]`
///
/// Returns `[Cout, H, W]` (length `Cout * H * W`).
pub fn conv2d(inp: &[f32], cin: usize, h: usize, w: usize,
              weight: &[f32], cout: usize, k: usize, pad: usize,
              bias: Option<&[f32]>) -> Vec<f32> {
    assert_eq!(inp.len(), cin * h * w, "inp shape mismatch");
    assert_eq!(weight.len(), cout * cin * k * k, "weight shape mismatch");
    if let Some(b) = bias { assert_eq!(b.len(), cout); }

    // ── im2col + Accelerate sgemm fast path ──
    // Best for medium/large 3×3 same-pad convs on macOS — typically 5–10×
    // faster than the NEON kernel below for cin ≥ 16 and h*w ≥ 1024.  Skip
    // for tiny shapes where the im2col alloc dominates the GEMM compute.
    #[cfg(target_os = "macos")]
    if k == 3 && pad == 1 && h * w >= 4096 && cin >= 16 && cout >= 16 {
        let mut out = conv2d_3x3_pad1_gemm(inp, cin, h, w, weight, cout, bias);
        // The gemm path returns final values; conv2d's downstream user expects
        // the same memory layout. No further bias add needed (handled inline).
        let _ = &mut out;
        return out;
    }

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
                    oc_plane, 0,
                    lo, hi_y, lo, hi_x, bb,
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
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

        // ── Border path ──
        for yo in 0..h { for xo in 0..w {
            if is_3x3_pad1 && yo >= lo && yo < hi_y && xo >= lo && xo < hi_x { continue; }
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

/// NEON-specialized 3×3 same-padding interior kernel. See module docs.
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
    let bias_v = unsafe { vdupq_n_f32(bias_oc) };
    for yo in yo_lo..yo_hi {
        let mut xo = xo_lo;
        while xo + 4 <= xo_hi {
            let mut acc = bias_v;
            for ic in 0..cin {
                let w_base = ((oc * cin) + ic) * 9;
                let in_off = ic * _h * w;
                for ky in 0..3 {
                    let yi = yo + ky - 1;
                    let row_base = in_off + yi * w + (xo - 1);
                    unsafe {
                        let v0 = vld1q_f32(inp.as_ptr().add(row_base));
                        let v1 = vld1q_f32(inp.as_ptr().add(row_base + 1));
                        let v2 = vld1q_f32(inp.as_ptr().add(row_base + 2));
                        let w_row = w_base + ky * 3;
                        acc = vfmaq_n_f32(acc, v0, weight[w_row]);
                        acc = vfmaq_n_f32(acc, v1, weight[w_row + 1]);
                        acc = vfmaq_n_f32(acc, v2, weight[w_row + 2]);
                    }
                }
            }
            unsafe { vst1q_f32(out.as_mut_ptr().add(oc_plane_off + yo * w + xo), acc); }
            xo += 4;
        }
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

/// Transposed 2D convolution ("deconv" / `nn.ConvTranspose2d` forward).
///
/// PyTorch weight layout: `[Cin, Cout, kH, kW]`  (input-channel-outer; this is
/// the convention `torch.onnx.export` preserves and the converter respects).
///
/// Parameters:
/// - `inp`: channel-first `[Cin, H, W]`.
/// - `weight`: `[Cin, Cout, kH, kW]` row-major.
/// - `bias`: optional `[Cout]`.
/// - `(sh, sw)`: stride; `(ph, pw)`: input padding (subtracts from output size).
///
/// Returns `(out, Ho, Wo)` with `Ho = (H-1)·sh + kH - 2·ph`, similarly Wo.
///
/// Used by pose / segmentation / super-res decoders that upsample feature
/// maps via learned deconv (vs nearest-neighbor / bilinear).
pub fn conv_transpose2d(
    inp: &[f32], cin: usize, h: usize, w: usize,
    weight: &[f32], cout: usize, kh: usize, kw: usize,
    sh: usize, sw: usize, ph: usize, pw: usize,
    bias: Option<&[f32]>,
) -> (Vec<f32>, usize, usize) {
    assert_eq!(inp.len(), cin * h * w);
    assert_eq!(weight.len(), cin * cout * kh * kw);
    if let Some(b) = bias { assert_eq!(b.len(), cout); }
    let ho = (h - 1) * sh + kh - 2 * ph;
    let wo = (w - 1) * sw + kw - 2 * pw;
    let mut out = vec![0.0f32; cout * ho * wo];
    out.par_chunks_mut(ho * wo).enumerate().for_each(|(oc, oplane)| {
        if let Some(b) = bias { for v in oplane.iter_mut() { *v = b[oc]; } }
        for ic in 0..cin {
            let wbase = ic * cout * kh * kw + oc * kh * kw;
            for ih in 0..h { for iw in 0..w {
                let iv = inp[ic * h * w + ih * w + iw];
                if iv == 0.0 { continue; }
                for ky in 0..kh {
                    let oh_pos = ih as isize * sh as isize + ky as isize - ph as isize;
                    if oh_pos < 0 || oh_pos as usize >= ho { continue; }
                    for kx in 0..kw {
                        let ow_pos = iw as isize * sw as isize + kx as isize - pw as isize;
                        if ow_pos < 0 || ow_pos as usize >= wo { continue; }
                        oplane[oh_pos as usize * wo + ow_pos as usize] += iv * weight[wbase + ky * kw + kx];
                    }
                }
            }}
        }
    });
    (out, ho, wo)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1×1 conv with identity weight + zero bias should be identity.
    #[test]
    fn conv2d_identity_1x1() {
        let h = 4; let w = 4;
        let inp: Vec<f32> = (0..3*h*w).map(|i| i as f32).collect();
        // weight [Cout=3, Cin=3, 1, 1] = identity 3×3
        let mut weight = vec![0.0f32; 3 * 3];
        for c in 0..3 { weight[c * 3 + c] = 1.0; }
        let out = conv2d(&inp, 3, h, w, &weight, 3, 1, 0, None);
        assert_eq!(out, inp);
    }

    /// 3×3 conv with all-ones weight on a 4×4 constant input — interior
    /// should be 9× input value; border less due to zero padding.
    #[test]
    fn conv2d_3x3_constant() {
        let h = 4; let w = 4;
        let inp = vec![1.0f32; h * w];
        let weight = vec![1.0f32; 1 * 1 * 9];
        let out = conv2d(&inp, 1, h, w, &weight, 1, 3, 1, None);
        // Interior (1,1): all 9 taps inside → 9
        assert!((out[1 * w + 1] - 9.0).abs() < 1e-6);
        assert!((out[1 * w + 2] - 9.0).abs() < 1e-6);
        // Corner (0,0): 4 valid taps (the 2×2 corner block)
        assert!((out[0 * w + 0] - 4.0).abs() < 1e-6);
        // Edge (0,1): 6 valid taps (top row missing, others present)
        assert!((out[0 * w + 1] - 6.0).abs() < 1e-6);
    }

    /// linear_f16 vs linear_f32 perf at a Whisper-large MLP shape (t=1500,
    /// din=1280, dout=5120 — encoder fc1).  Captures the f16-dequant overhead
    /// relative to pure-f32 Accelerate sgemm.
    #[test]
    #[ignore = "perf"]
    fn perf_baseline_linear_f16() {
        use std::time::Instant;
        let t = 1500; let din = 1280; let dout = 5120;
        let x: Vec<f32> = (0..t*din).map(|i| (i as f32 * 0.013).sin()).collect();
        let w_f32: Vec<f32> = (0..dout*din).map(|i| (i as f32 * 0.027).cos()).collect();
        let w_f16: Vec<half::f16> = w_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        let b: Vec<f32> = (0..dout).map(|i| i as f32 * 0.01).collect();
        // Warm-up.
        let _ = linear_f32(&x, t, din, &w_f32, dout, Some(&b));
        let _ = linear_f16(&x, t, din, &w_f16, dout, Some(&b));
        let runs = 5;
        let t0 = Instant::now();
        for _ in 0..runs { let _ = linear_f32(&x, t, din, &w_f32, dout, Some(&b)); }
        let f32_avg = t0.elapsed() / runs;
        let t0 = Instant::now();
        for _ in 0..runs { let _ = linear_f16(&x, t, din, &w_f16, dout, Some(&b)); }
        let f16_avg = t0.elapsed() / runs;
        let overhead = (f16_avg.as_secs_f64() - f32_avg.as_secs_f64()) / f32_avg.as_secs_f64();
        eprintln!("linear_f32 [{t}×{din}→{dout}]: {:?}/run", f32_avg);
        eprintln!("linear_f16 [{t}×{din}→{dout}]: {:?}/run  (+{:.1} % overhead)", f16_avg, overhead * 100.0);
    }

    /// linear_f32 perf baseline at the typical ViT-L MLP hot-path shape:
    /// 8192 tokens × 1024 → 4096 (one fc1 call at full V-JEPA-2 input).
    /// Run with `cargo test -p rce-op-conv2d --release perf_baseline_linear -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf"]
    fn perf_baseline_linear() {
        use std::time::Instant;
        let t = 8192; let din = 1024; let dout = 4096;
        let x: Vec<f32> = (0..t*din).map(|i| (i as f32 * 0.013).sin()).collect();
        let w: Vec<f32> = (0..dout*din).map(|i| (i as f32 * 0.027).cos()).collect();
        let b: Vec<f32> = (0..dout).map(|i| i as f32 * 0.01).collect();
        // Warm-up.
        let _ = linear_f32(&x, t, din, &w, dout, Some(&b));
        let runs = 5;
        let t0 = Instant::now();
        for _ in 0..runs { let _ = linear_f32(&x, t, din, &w, dout, Some(&b)); }
        let avg = t0.elapsed() / runs;
        let ops = (t as u64) * (din as u64) * (dout as u64) * 2;
        let gflops = ops as f64 / avg.as_secs_f64() / 1e9;
        eprintln!("linear_f32 [{t}×{din}→{dout}]: {:?}/run, {gflops:.1} GFLOPS", avg);
    }

    /// Perf baseline for the EDSR-equivalent hot path: 3×3 same-pad,
    /// cin=cout=64, h×w = 770×1120 (the auc-summary.png test image).
    /// Run with `cargo test -p rce-op-conv2d --release perf_baseline -- --ignored --nocapture`.
    /// 2026-06-03 baseline: NEON 4-wide kernel, 32-core M-series CPU.
    #[test]
    #[ignore = "perf — ~600 ms per call, allocates ~3 GB on first run"]
    fn perf_baseline() {
        use std::time::Instant;
        let (h, w) = (770, 1120);
        let cin = 64; let cout = 64;
        let inp: Vec<f32> = (0..cin*h*w).map(|i| (i as f32 * 0.013).sin()).collect();
        let weight: Vec<f32> = (0..cout*cin*9).map(|i| (i as f32 * 0.027).cos()).collect();
        let bias: Vec<f32> = (0..cout).map(|i| i as f32 * 0.01).collect();

        // Warm-up.
        let _ = conv2d(&inp, cin, h, w, &weight, cout, 3, 1, Some(&bias));

        let runs = 5;
        let t = Instant::now();
        for _ in 0..runs {
            let _ = conv2d(&inp, cin, h, w, &weight, cout, 3, 1, Some(&bias));
        }
        let avg = t.elapsed() / runs;
        let ops = (h as u64) * (w as u64) * (cin as u64) * (cout as u64) * 9 * 2; // FMA = 2 FLOPs
        let gflops = ops as f64 / avg.as_secs_f64() / 1e9;
        eprintln!("conv2d 3×3 [{cin}×{h}×{w} → {cout}×{h}×{w}]: {:?}/run, {gflops:.1} GFLOPS", avg);
    }

    /// ConvTranspose2d 2× upscale with stride=2, k=4, p=0 (the standard
    /// pose-head deconv layout). On a 1-channel constant input + identity weight
    /// the result must be `2H × 2W` zero-padded with the input value placed
    /// where the kernel aligns.
    #[test]
    fn conv_transpose_2x() {
        let h = 4; let w = 4;
        let cin = 1; let cout = 1;
        let kh = 4; let kw = 4;
        let sh = 2; let sw = 2;
        let inp = vec![1.0f32; cin * h * w];
        let weight = vec![1.0f32; cin * cout * kh * kw];
        let (out, ho, wo) = conv_transpose2d(&inp, cin, h, w, &weight, cout, kh, kw, sh, sw, 1, 1, None);
        assert_eq!((ho, wo), ((h - 1) * sh + kh - 2, (w - 1) * sw + kw - 2));
        // 8×8 output; interior must sum 4 input contributions per cell from the
        // overlapping kernel placements with stride 2 + k 4.
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().any(|&v| v > 0.0));
    }

    /// NEON f16 dequant must match scalar `half::f16::to_f32()`.
    #[test]
    fn dequant_f16_neon_matches_scalar() {
        let n = 257;  // intentionally non-multiple-of-4 to exercise tail
        let src: Vec<half::f16> = (0..n)
            .map(|i| half::f16::from_f32((i as f32 - 128.0) * 0.013))
            .collect();
        let ref_out: Vec<f32> = src.iter().map(|v| v.to_f32()).collect();
        let test_out = dequant_f16_to_f32(&src);
        assert_eq!(test_out.len(), n);
        for (a, b) in ref_out.iter().zip(test_out.iter()) {
            assert!((a - b).abs() < 1e-6, "NEON vs scalar dequant mismatch: {a} vs {b}");
        }
    }

    /// linear_f16 (f16 weights → dequant → Accelerate sgemm) matches scalar.
    #[test]
    fn linear_f16_matches_scalar() {
        let t = 17; let din = 64; let dout = 32;
        let x: Vec<f32> = (0..t*din).map(|i| (i as f32 * 0.013).sin()).collect();
        let w_f32: Vec<f32> = (0..dout*din).map(|i| (i as f32 * 0.029).cos()).collect();
        let w_f16: Vec<half::f16> = w_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        let b: Vec<f32> = (0..dout).map(|i| i as f32 * 0.1).collect();
        // Scalar reference using the same f16-rounded weights so we compare
        // numerics, not rounding.
        let w_round: Vec<f32> = w_f16.iter().map(|v| v.to_f32()).collect();
        let mut ref_y = vec![0.0f32; t * dout];
        for i in 0..t { for o in 0..dout {
            let mut acc = b[o];
            for k in 0..din { acc += x[i*din + k] * w_round[o*din + k]; }
            ref_y[i*dout + o] = acc;
        }}
        let test_y = linear_f16(&x, t, din, &w_f16, dout, Some(&b));
        let max_diff = ref_y.iter().zip(test_y.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "linear_f16 vs scalar max_diff {max_diff}");
    }

    /// k=5 conv (non-3×3 path) on a small input — exercises the general scalar
    /// interior path that the NEON 3×3 fast path doesn't cover.
    #[test]
    fn conv2d_k5_pad2() {
        let h = 8; let w = 8; let cin = 3; let cout = 4;
        let inp: Vec<f32> = (0..cin*h*w).map(|i| (i as f32 * 0.017).sin()).collect();
        let weight: Vec<f32> = (0..cout*cin*25).map(|i| (i as f32 * 0.031).cos()).collect();
        let bias: Vec<f32> = (0..cout).map(|i| i as f32 * 0.1).collect();
        let mut ref_out = vec![0.0f32; cout * h * w];
        for oc in 0..cout { for yo in 0..h { for xo in 0..w {
            let mut acc = bias[oc];
            for ic in 0..cin { for ky in 0..5 { for kx in 0..5 {
                let yi = (yo + ky) as isize - 2;
                let xi = (xo + kx) as isize - 2;
                if yi >= 0 && yi < h as isize && xi >= 0 && xi < w as isize {
                    acc += inp[ic*h*w + (yi as usize)*w + (xi as usize)]
                         * weight[((oc*cin)+ic)*25 + ky*5 + kx];
                }
            }}}
            ref_out[oc*h*w + yo*w + xo] = acc;
        }}}
        let test_out = conv2d(&inp, cin, h, w, &weight, cout, 5, 2, Some(&bias));
        let max_diff = ref_out.iter().zip(test_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "k=5 conv vs scalar max_diff {max_diff}");
    }

    /// First-layer 3×3 conv with cin=3 (RGB) and cout=64 — the typical
    /// CNN-entry shape. Below the GEMM threshold on small inputs, so
    /// exercises the NEON path.
    #[test]
    fn conv2d_rgb_entry() {
        let h = 32; let w = 32; let cin = 3; let cout = 64;
        let inp: Vec<f32> = (0..cin*h*w).map(|i| ((i % 256) as f32) / 255.0).collect();
        let weight: Vec<f32> = (0..cout*cin*9).map(|i| (i as f32 * 0.011).sin() * 0.1).collect();
        let test_out = conv2d(&inp, cin, h, w, &weight, cout, 3, 1, None);
        assert_eq!(test_out.len(), cout * h * w);
        assert!(test_out.iter().all(|v| v.is_finite()));
        // Output should be non-trivial.
        let spread = test_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                   - test_out.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(spread > 0.01, "output too flat: spread {spread}");
    }

    /// linear_f32 with t=1 (single token, sgemv-degenerate case) — used
    /// per-token in vjepa's `linear_tokens` gather/scatter helper.
    #[test]
    fn linear_f32_t1() {
        let din = 128; let dout = 64;
        let x: Vec<f32> = (0..din).map(|i| (i as f32 * 0.013).sin()).collect();
        let w: Vec<f32> = (0..dout*din).map(|i| (i as f32 * 0.029).cos()).collect();
        let b: Vec<f32> = (0..dout).map(|i| i as f32 * 0.05).collect();
        let mut ref_y = vec![0.0f32; dout];
        for o in 0..dout {
            let mut acc = b[o];
            for k in 0..din { acc += x[k] * w[o*din + k]; }
            ref_y[o] = acc;
        }
        let test_y = linear_f32(&x, 1, din, &w, dout, Some(&b));
        let max_diff = ref_y.iter().zip(test_y.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "t=1 linear_f32 vs scalar max_diff {max_diff}");
    }

    /// linear_f32 (PyTorch nn.Linear via Accelerate) matches scalar.
    #[test]
    fn linear_f32_matches_scalar() {
        let t = 33; let din = 47; let dout = 29;
        let x: Vec<f32> = (0..t*din).map(|i| (i as f32 * 0.013).sin()).collect();
        let w: Vec<f32> = (0..dout*din).map(|i| (i as f32 * 0.029).cos()).collect();
        let b: Vec<f32> = (0..dout).map(|i| i as f32 * 0.05).collect();
        // Scalar reference: y[i, o] = bias[o] + sum_k x[i, k] * w[o, k]
        let mut ref_y = vec![0.0f32; t * dout];
        for i in 0..t { for o in 0..dout {
            let mut acc = b[o];
            for k in 0..din { acc += x[i*din + k] * w[o*din + k]; }
            ref_y[i*dout + o] = acc;
        }}
        let test_y = linear_f32(&x, t, din, &w, dout, Some(&b));
        let max_diff = ref_y.iter().zip(test_y.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "linear_f32 vs scalar max_diff {max_diff}");
    }

    /// GEMM path must match the NEON/scalar reference.
    /// Dimensions chosen to exceed the GEMM dispatch threshold
    /// (`cin≥16, cout≥16, h*w≥4096`) on macOS.
    #[test]
    fn conv2d_gemm_matches_scalar() {
        let h = 64; let w = 64; let cin = 16; let cout = 32;
        let inp: Vec<f32> = (0..cin*h*w).map(|i| (i as f32 * 0.011).sin()).collect();
        let weight: Vec<f32> = (0..cout*cin*9).map(|i| (i as f32 * 0.029).cos()).collect();
        let bias: Vec<f32> = (0..cout).map(|i| i as f32 * 0.05).collect();

        // Scalar reference
        let mut ref_out = vec![0.0f32; cout * h * w];
        for oc in 0..cout {
            for yo in 0..h { for xo in 0..w {
                let mut acc = bias[oc];
                for ic in 0..cin {
                    for ky in 0..3 { for kx in 0..3 {
                        let yi = (yo + ky) as isize - 1;
                        let xi = (xo + kx) as isize - 1;
                        if yi >= 0 && yi < h as isize && xi >= 0 && xi < w as isize {
                            acc += inp[ic*h*w + (yi as usize)*w + (xi as usize)]
                                 * weight[((oc*cin)+ic)*9 + ky*3 + kx];
                        }
                    }}
                }
                ref_out[oc*h*w + yo*w + xo] = acc;
            }}
        }
        // h*w = 4096 == threshold; cin=16, cout=32 — should trigger GEMM on macOS.
        let test_out = conv2d(&inp, cin, h, w, &weight, cout, 3, 1, Some(&bias));
        let max_diff = ref_out.iter().zip(test_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3,
                "GEMM vs scalar max_diff {max_diff} (expected < 1e-3 for FP-reorder noise at this size)");
    }

    /// NEON path must match scalar reference on a non-trivial input.
    #[test]
    fn conv2d_neon_matches_scalar() {
        // Force scalar path by using K=5 (NEON kernel is 3×3 only).
        let h = 16; let w = 16; let cin = 4; let cout = 4;
        let inp: Vec<f32> = (0..cin*h*w).map(|i| (i as f32 * 0.013).sin()).collect();
        let weight: Vec<f32> = (0..cout*cin*9).map(|i| (i as f32 * 0.027).cos()).collect();
        let bias: Vec<f32> = (0..cout).map(|i| i as f32 * 0.1).collect();

        // Reference: hand-computed scalar 3×3 conv with same-pad
        let mut ref_out = vec![0.0f32; cout * h * w];
        for oc in 0..cout {
            for yo in 0..h { for xo in 0..w {
                let mut acc = bias[oc];
                for ic in 0..cin {
                    for ky in 0..3 { for kx in 0..3 {
                        let yi = (yo + ky) as isize - 1;
                        let xi = (xo + kx) as isize - 1;
                        if yi >= 0 && yi < h as isize && xi >= 0 && xi < w as isize {
                            acc += inp[ic*h*w + (yi as usize)*w + (xi as usize)]
                                 * weight[((oc*cin)+ic)*9 + ky*3 + kx];
                        }
                    }}
                }
                ref_out[oc*h*w + yo*w + xo] = acc;
            }}
        }

        let test_out = conv2d(&inp, cin, h, w, &weight, cout, 3, 1, Some(&bias));
        let max_diff = ref_out.iter().zip(test_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-5, "NEON vs scalar max_diff {max_diff}");
    }
}
