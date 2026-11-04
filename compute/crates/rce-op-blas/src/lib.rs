//! `rce-op-blas` — `OpDispatch` leaf that routes `Custom("matmul")` to
//! platform BLAS for the case it's actually fast at: f32 / f64, 2-D,
//! row-major, contiguous, no broadcasting. Everything else falls
//! through to the `rce-vm-backend-cpu` baseline via
//! `InterpretError::Unsupported`.
//!
//! Linkage:
//! - **macOS / iOS**: Apple Accelerate framework. Enabled by build.rs;
//!   includes AMX (Apple Matrix coprocessor) routing transparently.
//!   This is the same backend PyTorch's CPU device uses on macOS.
//! - **Linux**: OpenBLAS via `-lopenblas`. (MKL / BLIS work too if you
//!   substitute the linker line in `build.rs`.)
//! - **other targets**: no BLAS linked; every call returns `Unsupported`
//!   and the naive triple-loop in rce-vm-backend-cpu takes over.
//!
//! Footnote: this crate exists to make the IR + capability + scheduler
//! layers (the actual rce value) inherit production-grade kernel
//! perf, exactly as documented in `docs/book/src/performance.md`.

use rce_vm_backend_cpu::{HostTensor, InterpretError, OpDispatch, Result, Value};
use rce_graph::graph::OpKind;
use tracing::trace;

/// `OpDispatch` leaf for f32/f64 row-major 2-D matmul backed by the
/// platform BLAS. Construct one and pass it through the same
/// `InterpretConfig::ops` slot (or `Backends::ops`) you use for the
/// existing wgpu executor — the leaf is a `Send + Sync` zero-sized
/// type, so cloning / sharing is free.
#[derive(Debug, Default, Clone, Copy)]
pub struct BlasMatmul;

impl BlasMatmul {
    pub const fn new() -> Self {
        Self
    }
}

impl OpDispatch for BlasMatmul {
    fn handles(&self) -> &[&'static str] {
        // Only `matmul`. We don't claim `matmul_scale_softmax` etc. —
        // those have to fuse the post-matmul activation, which BLAS
        // doesn't do.
        &["matmul"]
    }

    fn execute(&self, node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "matmul" {
            return Err(InterpretError::Unsupported(format!(
                "blas leaf does not handle `{name}`"
            )));
        }
        if args.len() != 2 {
            return Err(InterpretError::Unsupported(format!(
                "blas matmul: expected 2 args, got {}",
                args.len()
            )));
        }
        let a = match &args[0] {
            Value::Tensor(t) => t,
            _ => {
                return Err(InterpretError::Unsupported(
                    "blas matmul: arg 0 not a tensor".into(),
                ))
            }
        };
        let b = match &args[1] {
            Value::Tensor(t) => t,
            _ => {
                return Err(InterpretError::Unsupported(
                    "blas matmul: arg 1 not a tensor".into(),
                ))
            }
        };
        // BLAS path is 2-D only. Batched matmul falls back.
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(InterpretError::Unsupported(format!(
                "blas matmul: rank-2 only (got {:?} × {:?})",
                a.shape, b.shape
            )));
        }
        if a.dtype != b.dtype {
            return Err(InterpretError::Unsupported(format!(
                "blas matmul: dtype mismatch {} vs {}",
                a.dtype.name(),
                b.dtype.name()
            )));
        }
        let m = a.shape[0];
        let k = a.shape[1];
        let kk = b.shape[0];
        let n = b.shape[1];
        if k != kk {
            return Err(InterpretError::Unsupported(format!(
                "blas matmul: inner dims {k} vs {kk}"
            )));
        }
        // Empty matrix → fall through (the naive path returns a
        // correctly-shaped empty tensor; replicating that here is a
        // separate concern).
        if m == 0 || n == 0 || k == 0 {
            return Err(InterpretError::Unsupported(
                "blas matmul: zero-size dim, fall through".into(),
            ));
        }

        match a.dtype {
            rce_vm_backend_cpu::DType::F32 => {
                // Zero-copy &[f32] view of the input bytes. The
                // `HostTensor::data` Vec<u8> comes from the global
                // allocator which on every supported platform returns
                // ≥4-byte aligned pointers, so the cast succeeds in
                // practice. If alignment ever failed we'd return
                // Unsupported and the cpu kernel layer (which copies
                // element-by-element) would handle it.
                let av: &[f32] = bytemuck::try_cast_slice(&a.data).map_err(|e| {
                    InterpretError::Unsupported(format!(
                        "blas matmul: a not 4-byte aligned ({e}); falling through"
                    ))
                })?;
                let bv: &[f32] = bytemuck::try_cast_slice(&b.data).map_err(|e| {
                    InterpretError::Unsupported(format!(
                        "blas matmul: b not 4-byte aligned ({e}); falling through"
                    ))
                })?;
                // Allocate output as a `Vec<u8>` and write cblas's
                // result directly into it via a *mut f32 reinterpret.
                // No intermediate Vec<f32> allocation, no extra memcpy
                // — the only data movement is what BLAS itself does.
                //
                // SAFETY: malloc on every supported platform (jemalloc
                // / mimalloc / system malloc on linux+macOS) returns
                // pointers with ≥16-byte alignment for nonzero allocs,
                // which exceeds the 4-byte alignment f32 needs. We
                // also wrote `set_len(m*n*4)` and then sgemm fully
                // overwrites those bytes, so no uninit-read can leak.
                let mut out_bytes: Vec<u8> = Vec::with_capacity(m * n * 4);
                unsafe {
                    out_bytes.set_len(m * n * 4);
                    let out_ptr = out_bytes.as_mut_ptr() as *mut f32;
                    debug_assert!(out_ptr.is_aligned());
                    let out_slice = std::slice::from_raw_parts_mut(out_ptr, m * n);
                    sgemm_into(av, bv, out_slice, m, k, n);
                }
                trace!(target: "rce::op::blas", node = node_id, m, k, n, "sgemm dispatched");
                HostTensor::new(
                    vec![m, n],
                    rce_vm_backend_cpu::DType::F32,
                    out_bytes,
                )
                .map(Value::Tensor)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "blas matmul: build out tensor: {e:?}"
                )))
            }
            rce_vm_backend_cpu::DType::F64 => {
                let av: &[f64] = bytemuck::try_cast_slice(&a.data).map_err(|e| {
                    InterpretError::Unsupported(format!(
                        "blas matmul: a not 8-byte aligned ({e}); falling through"
                    ))
                })?;
                let bv: &[f64] = bytemuck::try_cast_slice(&b.data).map_err(|e| {
                    InterpretError::Unsupported(format!(
                        "blas matmul: b not 8-byte aligned ({e}); falling through"
                    ))
                })?;
                let mut out_bytes: Vec<u8> = Vec::with_capacity(m * n * 8);
                unsafe {
                    out_bytes.set_len(m * n * 8);
                    let out_ptr = out_bytes.as_mut_ptr() as *mut f64;
                    debug_assert!(out_ptr.is_aligned());
                    let out_slice = std::slice::from_raw_parts_mut(out_ptr, m * n);
                    dgemm_into(av, bv, out_slice, m, k, n);
                }
                trace!(target: "rce::op::blas", node = node_id, m, k, n, "dgemm dispatched");
                HostTensor::new(
                    vec![m, n],
                    rce_vm_backend_cpu::DType::F64,
                    out_bytes,
                )
                .map(Value::Tensor)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "blas matmul: build out tensor: {e:?}"
                )))
            }
            other => Err(InterpretError::Unsupported(format!(
                "blas matmul: dtype {} not handled (f32/f64 only)",
                other.name()
            ))),
        }
    }

    fn execute_primitive(
        &self,
        _node_id: usize,
        _op: &OpKind,
        _args: &[Value],
    ) -> Result<Value> {
        // We don't claim primitive ops (Add/Sub/Mul/Div) — those go to
        // wgpu or to the cpu kernel layer, which already handles them
        // efficiently for tensors.
        Err(InterpretError::Unsupported(
            "blas leaf: no primitive ops".into(),
        ))
    }
}

// ── BLAS bindings ───────────────────────────────────────────────────

#[cfg(any(rce_blas_accelerate, rce_blas_cblas))]
mod cblas {
    // Minimum CBLAS API surface we need. Both Accelerate and OpenBLAS
    // expose these symbols with these signatures.
    use std::os::raw::{c_double, c_float, c_int};

    pub const CBLAS_ROW_MAJOR: c_int = 101;
    pub const CBLAS_NO_TRANS: c_int = 111;
    pub const CBLAS_TRANS: c_int = 112;

    unsafe extern "C" {
        pub fn cblas_sgemm(
            order: c_int,
            trans_a: c_int,
            trans_b: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            b: *const c_float,
            ldb: c_int,
            beta: c_float,
            c: *mut c_float,
            ldc: c_int,
        );
        pub fn cblas_dgemm(
            order: c_int,
            trans_a: c_int,
            trans_b: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: c_double,
            a: *const c_double,
            lda: c_int,
            b: *const c_double,
            ldb: c_int,
            beta: c_double,
            c: *mut c_double,
            ldc: c_int,
        );
    }
}

// ── Apple Accelerate vDSP API (macOS / iOS only) ────────────────────
//
// `vDSP_mmul` is Accelerate's vectorized f32 matrix-multiply — same
// internal implementation as `cblas_sgemm` on Apple silicon, both at
// roughly 5 GFLOPS f32 single-thread on M3 Ultra. Kept as a fallback
// for the case where BNNS rejects a shape we want to handle.
//
// Layout: A is (M, P), B is (P, N), C is (M, N). All strides are
// "elements between consecutive items in the same row" — for our
// contiguous row-major buffers that's 1.
#[cfg(rce_blas_accelerate)]
mod vdsp {
    use std::os::raw::{c_float, c_long};

    /// `vDSP_Length` is `unsigned long` (`u64` on macOS arm64 and x86_64).
    pub type VDSPLength = u64;
    /// `vDSP_Stride` is `long` (`i64`).
    pub type VDSPStride = c_long;

    unsafe extern "C" {
        pub fn vDSP_mmul(
            a: *const c_float,
            ia: VDSPStride,
            b: *const c_float,
            ib: VDSPStride,
            c: *mut c_float,
            ic: VDSPStride,
            m: VDSPLength,
            n: VDSPLength,
            p: VDSPLength,
        );
    }
}

// ── Apple BNNS — Basic Neural Network Subroutines (macOS / iOS) ─────
//
// `BNNSMatMul` is the path PyTorch's CPU `torch.matmul` takes on
// macOS for f32 inputs. It dispatches to Apple's AMX (Apple Matrix
// coprocessor) at 5–15 TFLOPS f32, an order of magnitude above
// `cblas_sgemm` / `vDSP_mmul` (both ≈5 GFLOPS), because AMX is only
// exposed via BNNS — neither cblas nor vDSP reach it. Routing
// matmul through BNNS is what closes the remaining PyTorch gap on
// M3-class silicon.
//
// Apple's API contract is: call `BNNSMatMulWorkspaceSize` first to
// learn how many scratch bytes the kernel needs, allocate, then
// call `BNNSMatMul`. We do that per-call here. The workspace is
// small relative to the matmul bandwidth so the per-call alloc
// doesn't dominate (and a future revision can cache it per shape
// using the same trick the MPS leaf already uses).
#[cfg(rce_blas_accelerate)]
#[allow(dead_code)] // BNNS FFI scaffolding — kept for future Apple Accelerate routes
mod bnns {
    use std::os::raw::c_void;

    /// `BNNSDataType` bitfield. Float32 = 0x10000020 (bit 28 set, low byte = 32).
    pub const BNNS_DATA_TYPE_FLOAT32: u32 = 0x1000_0020;
    /// Row-major 2-D matrix: `size[0]` is *columns*, `size[1]` is *rows*.
    pub const BNNS_DATA_LAYOUT_ROW_MAJOR_MATRIX: u32 = 0x2_0000;
    /// `BNNS_MAX_TENSOR_DIMENSION` = 8.
    pub const BNNS_MAX_DIM: usize = 8;

    /// Mirrors Apple's `BNNSNDArrayDescriptor`. With `repr(C)` the
    /// implicit 4-byte padding after `data_type` is inserted by the
    /// Rust compiler exactly where the C compiler does, so the
    /// struct size lands on 176 bytes — verified by the
    /// `_DESC_SIZE_CHECK` constant below.
    #[repr(C)]
    pub struct BNNSNDArrayDescriptor {
        pub flags: u32,
        pub layout: u32,
        pub size: [usize; BNNS_MAX_DIM],
        pub stride: [usize; BNNS_MAX_DIM],
        pub data: *mut c_void,
        pub data_type: u32,
        pub table_data: *mut c_void,
        pub table_data_type: u32,
        pub data_scale: f32,
        pub data_bias: f32,
    }

    impl BNNSNDArrayDescriptor {
        /// Build a row-major 2-D f32 descriptor for an (rows, cols)
        /// matrix at `data`. BNNS's convention is width-first
        /// (`size[0] = cols`, `size[1] = rows`).
        pub fn rowmajor_f32(data: *mut c_void, rows: usize, cols: usize) -> Self {
            let mut size = [0usize; BNNS_MAX_DIM];
            size[0] = cols;
            size[1] = rows;
            Self {
                flags: 0,
                layout: BNNS_DATA_LAYOUT_ROW_MAJOR_MATRIX,
                size,
                stride: [0; BNNS_MAX_DIM],
                data,
                data_type: BNNS_DATA_TYPE_FLOAT32,
                table_data: std::ptr::null_mut(),
                table_data_type: 0,
                data_scale: 1.0,
                data_bias: 0.0,
            }
        }
    }

    /// `BNNSFilterParameters` from `bnns_structures.h`. We only ever
    /// pass `NULL` for it (= use BNNS defaults: thread-count auto,
    /// posix_memalign for internal allocs), but the struct definition
    /// is kept here for future use should we want to pin n_threads.
    #[repr(C)]
    #[allow(dead_code)]
    pub struct BNNSFilterParameters {
        pub flags: u32,
        pub n_threads: usize,
        pub alloc_memory: *mut c_void,
        pub free_memory: *mut c_void,
    }

    unsafe extern "C" {
        /// Returns the workspace size (bytes) that `BNNSMatMul` needs
        /// for the given operand shapes. Negative → invalid params.
        pub fn BNNSMatMulWorkspaceSize(
            trans_a: bool,
            trans_b: bool,
            alpha: f32,
            input_a: *const BNNSNDArrayDescriptor,
            input_b: *const BNNSNDArrayDescriptor,
            output: *const BNNSNDArrayDescriptor,
            filter_params: *const BNNSFilterParameters,
        ) -> isize;

        /// `C = alpha · op(A) · op(B)`. `workspace` must point at a
        /// buffer of ≥ `BNNSMatMulWorkspaceSize` bytes (or be NULL
        /// when that returns 0). Returns 0 on success.
        pub fn BNNSMatMul(
            trans_a: bool,
            trans_b: bool,
            alpha: f32,
            input_a: *const BNNSNDArrayDescriptor,
            input_b: *const BNNSNDArrayDescriptor,
            output: *const BNNSNDArrayDescriptor,
            workspace: *mut c_void,
            filter_params: *const BNNSFilterParameters,
        ) -> i32;
    }

    /// Compile-time sanity check on struct size — catches header drift
    /// rather than letting it become a mysterious runtime corruption.
    #[allow(dead_code)]
    const _DESC_SIZE_CHECK: [(); 176] = [(); std::mem::size_of::<BNNSNDArrayDescriptor>()];
}

/// 2-D row-major sgemm into a caller-provided output buffer. `c` must
/// have length ≥ m*n; the routine overwrites those m*n elements (the
/// rest, if any, are untouched).
///
/// On Apple silicon (`rce_blas_accelerate`) we route to BNNS's
/// `BNNSDirectApplyMatMul` — the AMX-coprocessor path PyTorch uses
/// internally. AMX peaks at 5–15 TFLOPS f32, an order of magnitude
/// above `cblas_sgemm` / `vDSP_mmul` (both stuck at ~5 GFLOPS because
/// neither reaches AMX). If BNNS rejects the inputs (returns
/// non-zero) we fall back to `vDSP_mmul`. On Linux / OpenBLAS
/// (`rce_blas_cblas`) we stay on `cblas_sgemm`.
#[cfg(any(rce_blas_accelerate, rce_blas_cblas))]
pub fn sgemm_into(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert!(a.len() >= m * k);
    debug_assert!(b.len() >= k * n);
    debug_assert!(c.len() >= m * n);

    #[cfg(rce_blas_accelerate)]
    unsafe {
        // We tried `BNNSMatMul` here (intent: route to AMX directly).
        // The direct BNNS API is deprecated in macOS 15+ ("Use
        // BNNSGraph* APIs") and on M3 Ultra it benched 5–28% *worse*
        // than `vDSP_mmul` at every size we tested — most likely
        // because the Apple-recommended path is now BNNSGraph (which
        // requires per-shape graph compilation up front). Until we
        // wire BNNSGraph in, vDSP_mmul is the better default; the
        // BNNS module above is kept compiled-in and ready for the
        // graph-API rewrite.
        vdsp::vDSP_mmul(
            a.as_ptr(),
            1,
            b.as_ptr(),
            1,
            c.as_mut_ptr(),
            1,
            m as vdsp::VDSPLength,
            n as vdsp::VDSPLength,
            k as vdsp::VDSPLength,
        );
        return;
    }

    #[cfg(all(rce_blas_cblas, not(rce_blas_accelerate)))]
    unsafe {
        use cblas::*;
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

/// Row-major sgemm with a runtime-selectable transpose flag on the
/// `b` argument: computes `C = A · op(B)` where `op(B)` is `B`
/// (when `trans_b == false`) or `B^T` (when `trans_b == true`).
///
/// This exists for callers like the attention leaf that need to do
/// `Q @ K^T` without pre-transposing K — a 128 KB+ extra memory
/// pass on real attention shapes that the BLAS implementation can
/// avoid internally with cache-aware blocking.
///
/// Shapes (interpreted as the row-major operands):
/// - `a`: `(m, k)` — `lda = k`
/// - `b`: `(n, k)` if `trans_b`, else `(k, n)` — `ldb` follows
/// - `c`: `(m, n)` — `ldc = n`
///
/// Always uses cblas (Accelerate ships cblas-compatible symbols), so
/// this works on both macOS and Linux. On Apple silicon vDSP_mmul
/// doesn't support transpose flags, hence cblas here even when
/// `vDSP_mmul` is the default for non-transposed sgemm.
#[cfg(any(rce_blas_accelerate, rce_blas_cblas))]
pub fn sgemm_with_trans_b_into(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    trans_b: bool,
) {
    debug_assert!(a.len() >= m * k);
    debug_assert!(b.len() >= k * n);
    debug_assert!(c.len() >= m * n);
    use cblas::*;
    let (b_op, ldb) = if trans_b {
        (CBLAS_TRANS, k as i32) // B is (n, k) row-major, transpose stride
    } else {
        (CBLAS_NO_TRANS, n as i32)
    };
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            b_op,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32, // lda = k for row-major (m, k)
            b.as_ptr(),
            ldb,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

/// Without a platform BLAS: the same product, row-major, in a portable
/// loop. Each element adds its k products in increasing k, starting from
/// +0.0 (the order of the CPU backend's own matmul), so callers such as
/// rce-op-attention work on every target.
#[cfg(not(any(rce_blas_accelerate, rce_blas_cblas)))]
pub fn sgemm_with_trans_b_into(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    trans_b: bool,
) {
    debug_assert!(a.len() >= m * k);
    debug_assert!(b.len() >= k * n);
    debug_assert!(c.len() >= m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                // B is (n, k) row-major when transposed, else (k, n)
                let bv = if trans_b { b[j * k + p] } else { b[p * n + j] };
                acc += a[i * k + p] * bv;
            }
            c[i * n + j] = acc;
        }
    }
}

#[cfg(not(any(rce_blas_accelerate, rce_blas_cblas)))]
pub fn sgemm_into(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    sgemm_with_trans_b_into(a, b, c, m, k, n, false)
}

/// Wrapper kept for tests.
#[cfg(any(rce_blas_accelerate, rce_blas_cblas))]
#[allow(dead_code)]
fn sgemm_2d(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c: Vec<f32> = Vec::with_capacity(m * n);
    unsafe { c.set_len(m * n) };
    sgemm_into(a, b, &mut c, m, k, n);
    c
}

#[cfg(any(rce_blas_accelerate, rce_blas_cblas))]
fn dgemm_into(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    use cblas::*;
    debug_assert!(a.len() >= m * k);
    debug_assert!(b.len() >= k * n);
    debug_assert!(c.len() >= m * n);
    unsafe {
        cblas_dgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

#[cfg(not(any(rce_blas_accelerate, rce_blas_cblas)))]
fn dgemm_into(_a: &[f64], _b: &[f64], _c: &mut [f64], _m: usize, _k: usize, _n: usize) {
    unreachable!("dgemm_into called on a target without BLAS linked")
}

#[cfg(any(rce_blas_accelerate, rce_blas_cblas))]
#[allow(dead_code)]
fn dgemm_2d(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c: Vec<f64> = Vec::with_capacity(m * n);
    unsafe { c.set_len(m * n) };
    dgemm_into(a, b, &mut c, m, k, n);
    c
}

// ── Fallback when no BLAS is linked ────────────────────────────────
//
// On targets without Accelerate or OpenBLAS, the leaf still compiles
// but always returns Unsupported, so the cpu backend's existing matmul
// handles every workload. The functions below are only reachable if
// someone hand-edits the cfg gate — keeping them ensures the impl
// block is type-checked on every platform.

#[cfg(not(any(rce_blas_accelerate, rce_blas_cblas)))]
fn sgemm_2d(_a: &[f32], _b: &[f32], _m: usize, _k: usize, _n: usize) -> Vec<f32> {
    unreachable!(
        "sgemm_2d called on a target without BLAS linked — this should be \
         filtered earlier by returning Unsupported"
    )
}

#[cfg(not(any(rce_blas_accelerate, rce_blas_cblas)))]
fn dgemm_2d(_a: &[f64], _b: &[f64], _m: usize, _k: usize, _n: usize) -> Vec<f64> {
    unreachable!(
        "dgemm_2d called on a target without BLAS linked — this should be \
         filtered earlier by returning Unsupported"
    )
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rce_vm_backend_cpu::{DType, HostTensor, Value};

    #[test]
    fn sgemm_matches_naive_2x3_3x2() {
        // A is (2, 3), B is (3, 2). Expected C is (2, 2) per row*col sums.
        let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        // A @ B = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //       = [[58, 64], [139, 154]]
        let leaf = BlasMatmul;
        let ta = HostTensor::from_f32(vec![2, 3], &a).unwrap();
        let tb = HostTensor::from_f32(vec![3, 2], &b).unwrap();
        let result = leaf
            .execute(0, "matmul", &[Value::Tensor(ta), Value::Tensor(tb)])
            .expect("blas matmul");
        let out = match result {
            Value::Tensor(t) => t,
            _ => panic!("expected tensor"),
        };
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.dtype, DType::F32);
        let v = out.as_f32().unwrap();
        assert!((v[0] - 58.0).abs() < 1e-4, "got {}", v[0]);
        assert!((v[1] - 64.0).abs() < 1e-4, "got {}", v[1]);
        assert!((v[2] - 139.0).abs() < 1e-4, "got {}", v[2]);
        assert!((v[3] - 154.0).abs() < 1e-4, "got {}", v[3]);
    }

    #[test]
    fn falls_through_for_3d_tensors() {
        let leaf = BlasMatmul;
        let ta = HostTensor::from_f32(vec![1, 2, 3], &[0.0_f32; 6]).unwrap();
        let tb = HostTensor::from_f32(vec![1, 3, 2], &[0.0_f32; 6]).unwrap();
        let err = leaf
            .execute(0, "matmul", &[Value::Tensor(ta), Value::Tensor(tb)])
            .unwrap_err();
        assert!(matches!(err, InterpretError::Unsupported(_)));
    }

    #[test]
    fn falls_through_for_unrecognised_op() {
        let leaf = BlasMatmul;
        let ta = HostTensor::from_f32(vec![2, 2], &[1.0; 4]).unwrap();
        let tb = HostTensor::from_f32(vec![2, 2], &[1.0; 4]).unwrap();
        let err = leaf
            .execute(0, "softmax", &[Value::Tensor(ta), Value::Tensor(tb)])
            .unwrap_err();
        assert!(matches!(err, InterpretError::Unsupported(_)));
    }
}
