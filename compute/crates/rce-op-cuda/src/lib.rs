//! `rce-op-cuda` — `OpDispatch` leaf that routes f32 row-major
//! 2-D matmul to NVIDIA cuBLAS via the CUDA Driver API. cuBLAS is
//! the kernel PyTorch's CUDA path uses (and every other production
//! ML framework on Nvidia silicon), so this is the apples-to-apples
//! GPU comparison: same kernel layer, lighter dispatch.
//!
//! ## Architecture
//!
//! Mirrors `rce-op-mps` exactly:
//!
//! 1. Lazy CUDA context initialisation — `cuInit` + `cuCtxCreate` once
//!    per process.
//! 2. Per-shape resource cache — for every `(m, k, n)` we've seen, hold
//!    a `(d_a, d_b, d_c)` device-buffer triple. Steady-state dispatch
//!    is one `cuMemcpyHtoD` per input + one `cublasSgemm` + one
//!    `cuMemcpyDtoH` for the output.
//! 3. cuBLAS column-major trick — `cublasSgemm` is column-major, but
//!    we want row-major. Compute `C = A·B` (row-major) by calling
//!    `sgemm(B, A) → C` in column-major layout: the result has the
//!    same bytes as a row-major C.
//!
//! ## Cross-platform
//!
//! Non-Linux builds, or Linux builds without CUDA installed, compile
//! to a stub that always returns `Unsupported`. The build script
//! decides at compile-time via `cargo:rustc-cfg=rce_cuda`.

use rce_vm_backend_cpu::{
    DeviceId, DeviceMem, DeviceTensor, InterpretError, OpDispatch, Result, Value,
};
use rce_graph::graph::OpKind;
use std::sync::Arc;

pub struct CublasMatmul {
    #[cfg(rce_cuda)]
    inner: imp::CudaContext,
}

impl CublasMatmul {
    /// Construct a leaf bound to CUDA device 0. Returns `None` if
    /// CUDA is unavailable (no toolkit installed, no GPU, non-Linux
    /// target).
    pub fn new() -> Option<Self> {
        #[cfg(rce_cuda)]
        {
            imp::CudaContext::new(0).map(|inner| Self { inner })
        }
        #[cfg(not(rce_cuda))]
        {
            None
        }
    }

    /// Upload a HostTensor to GPU memory and return a `DeviceTensor`
    /// the caller can pass into subsequent `execute(...)` calls
    /// without paying the host→device PCIe transfer per matmul.
    ///
    /// Returns the tensor's lifetime tied to the caller's hold on the
    /// returned `Value::DeviceTensor`; dropping it frees the device
    /// memory. This is the bench's pre-loading primitive (mirrors
    /// PyTorch's `tensor.to('cuda')`).
    #[cfg(rce_cuda)]
    pub fn host_to_device(
        &self,
        host: &rce_vm_backend_cpu::HostTensor,
    ) -> std::result::Result<DeviceTensor, String> {
        if host.dtype != rce_vm_backend_cpu::DType::F32 {
            return Err(format!(
                "host_to_device: f32 only (got {})",
                host.dtype.name()
            ));
        }
        let bytes = host.data.len();
        let mem = self.inner.upload(&host.data, bytes)?;
        Ok(DeviceTensor::new(
            DeviceId::Cuda(0),
            host.dtype,
            host.shape.clone(),
            mem,
        ))
    }

    #[cfg(not(rce_cuda))]
    pub fn host_to_device(
        &self,
        _host: &rce_vm_backend_cpu::HostTensor,
    ) -> std::result::Result<DeviceTensor, String> {
        Err("cublas: not built with CUDA".into())
    }

    // ── Internal dispatch helpers ──────────────────────────────────

    fn execute_device_resident(
        &self,
        node_id: usize,
        a: &DeviceTensor,
        b: &DeviceTensor,
    ) -> Result<Value> {
        if a.device != DeviceId::Cuda(0) || b.device != DeviceId::Cuda(0) {
            return Err(InterpretError::Unsupported(format!(
                "cublas matmul: device-resident inputs on {} / {}, expected Cuda(0)",
                a.device.name(),
                b.device.name()
            )));
        }
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(InterpretError::Unsupported(format!(
                "cublas matmul: rank-2 only (got {:?} × {:?})",
                a.shape, b.shape
            )));
        }
        if a.dtype != rce_vm_backend_cpu::DType::F32
            || b.dtype != rce_vm_backend_cpu::DType::F32
        {
            return Err(InterpretError::Unsupported(format!(
                "cublas matmul: f32 only (got {} × {})",
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
                "cublas matmul: inner dim {k} vs {kk}"
            )));
        }

        #[cfg(rce_cuda)]
        {
            let mem = self
                .inner
                .matmul_device_resident(a.ptr(), b.ptr(), m, k, n)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "cublas matmul: {e}"
                )))?;
            tracing::trace!(target: "rce::op::cuda", node = node_id, m, k, n, "cublasSgemm device-resident");
            Ok(Value::DeviceTensor(DeviceTensor::new(
                DeviceId::Cuda(0),
                rce_vm_backend_cpu::DType::F32,
                vec![m, n],
                mem,
            )))
        }

        #[cfg(not(rce_cuda))]
        {
            let _ = (node_id, m, k, n);
            Err(InterpretError::Unsupported(
                "cublas matmul: not built with CUDA support".into(),
            ))
        }
    }

    fn execute_host_input(
        &self,
        node_id: usize,
        a: &rce_vm_backend_cpu::HostTensor,
        b: &rce_vm_backend_cpu::HostTensor,
    ) -> Result<Value> {
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(InterpretError::Unsupported(format!(
                "cublas matmul: rank-2 only (got {:?} × {:?})",
                a.shape, b.shape
            )));
        }
        if a.dtype != rce_vm_backend_cpu::DType::F32
            || b.dtype != rce_vm_backend_cpu::DType::F32
        {
            return Err(InterpretError::Unsupported(format!(
                "cublas matmul: f32 only (got {} × {})",
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
                "cublas matmul: inner dim {k} vs {kk}"
            )));
        }
        if m == 0 || n == 0 || k == 0 {
            return Err(InterpretError::Unsupported(
                "cublas matmul: zero-size dim, fall through".into(),
            ));
        }
        const CUDA_MIN_FLOPS: usize = 5_000_000_000; // ~matmul-1300³
        let work = 2usize.saturating_mul(m).saturating_mul(k).saturating_mul(n);
        if work < CUDA_MIN_FLOPS {
            return Err(InterpretError::Unsupported(
                "cublas matmul: host inputs below CUDA_MIN_FLOPS, defer to BLAS"
                    .into(),
            ));
        }

        #[cfg(rce_cuda)]
        {
            let av: &[f32] = bytemuck::try_cast_slice(&a.data).map_err(|e| {
                InterpretError::Unsupported(format!(
                    "cublas matmul: a not 4-byte aligned ({e})"
                ))
            })?;
            let bv: &[f32] = bytemuck::try_cast_slice(&b.data).map_err(|e| {
                InterpretError::Unsupported(format!(
                    "cublas matmul: b not 4-byte aligned ({e})"
                ))
            })?;
            let mut out_bytes: Vec<u8> = Vec::with_capacity(m * n * 4);
            unsafe {
                out_bytes.set_len(m * n * 4);
                let out_ptr = out_bytes.as_mut_ptr() as *mut f32;
                let out_slice = std::slice::from_raw_parts_mut(out_ptr, m * n);
                self.inner
                    .matmul(av, bv, out_slice, m, k, n)
                    .map_err(|e| InterpretError::Unsupported(format!(
                        "cublas matmul: {e}"
                    )))?;
            }
            tracing::trace!(target: "rce::op::cuda", node = node_id, m, k, n, "cublasSgemm host-input");
            rce_vm_backend_cpu::HostTensor::new(
                vec![m, n],
                rce_vm_backend_cpu::DType::F32,
                out_bytes,
            )
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!(
                "cublas matmul: build out tensor: {e:?}"
            )))
        }

        #[cfg(not(rce_cuda))]
        {
            let _ = (node_id, m, k, n);
            Err(InterpretError::Unsupported(
                "cublas matmul: not built with CUDA support".into(),
            ))
        }
    }
}

impl OpDispatch for CublasMatmul {
    fn handles(&self) -> &[&'static str] {
        &["matmul"]
    }

    fn execute(&self, node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "matmul" {
            return Err(InterpretError::Unsupported(format!(
                "cublas leaf does not handle `{name}`"
            )));
        }
        if args.len() != 2 {
            return Err(InterpretError::Unsupported(format!(
                "cublas matmul: expected 2 args, got {}",
                args.len()
            )));
        }

        // Two paths:
        // - DEVICE-RESIDENT: both args are DeviceTensors on `Cuda(0)` —
        //   skip host↔device memcpy entirely, run sgemm directly on the
        //   already-resident pointers, return a fresh DeviceTensor for C.
        // - HOST-INPUT: both args are HostTensors — current memcpy path
        //   into the cached d_a/d_b/d_c, return a HostTensor.
        // Mixed (one host, one device) → Unsupported; force the user to
        // pre-load or read back so the call is unambiguous.
        match (&args[0], &args[1]) {
            (Value::DeviceTensor(da), Value::DeviceTensor(db)) => {
                self.execute_device_resident(node_id, da, db)
            }
            (Value::Tensor(a), Value::Tensor(b)) => {
                self.execute_host_input(node_id, a, b)
            }
            (Value::DeviceTensor(_), _) | (_, Value::DeviceTensor(_)) => {
                Err(InterpretError::Unsupported(
                    "cublas matmul: mixed host + device inputs; pre-load or \
                     read back so both args live on the same residency"
                        .into(),
                ))
            }
            _ => Err(InterpretError::Unsupported(
                "cublas matmul: both args must be tensors".into(),
            )),
        }
    }

    fn execute_primitive(
        &self,
        _node_id: usize,
        _op: &OpKind,
        _args: &[Value],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "cublas leaf: no primitive ops".into(),
        ))
    }
}

// ── Linux + CUDA implementation ─────────────────────────────────────

#[cfg(rce_cuda)]
mod imp {
    //! Direct CUDA Driver API + cuBLAS bindings.
    //!
    //! No `cudarc` / `cust` / `rust-cuda` dependency — the surface we
    //! need is small enough (~10 functions) that hand-written FFI is
    //! lighter than dragging in the ecosystem. Same trade-off the MPS
    //! leaf made for objc.

    use rce_vm_backend_cpu::DeviceMem;
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::os::raw::c_int;
    use std::sync::{Arc, Mutex};

    // ── CUDA Driver API types & constants ──────────────────────────

    pub type CUresult = c_int;
    pub type CUdevice = c_int;
    pub type CUcontext = *mut c_void;
    pub type CUdeviceptr = u64;

    const CUDA_SUCCESS: CUresult = 0;

    // cuBLAS opaque handle.
    pub type CublasHandle = *mut c_void;
    pub type CublasStatus = c_int;
    const CUBLAS_STATUS_SUCCESS: CublasStatus = 0;
    pub const CUBLAS_OP_N: c_int = 0;
    /// `cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH` — enables TF32 on
    /// Ampere/Hopper tensor cores for f32 GEMM, ~8× speedup over the
    /// default IEEE-strict path. Matches PyTorch's default mode on H100.
    pub const CUBLAS_TF32_TENSOR_OP_MATH: c_int = 1;

    unsafe extern "C" {
        // Driver API.
        fn cuInit(flags: u32) -> CUresult;
        fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
        fn cuCtxCreate_v2(ctx: *mut CUcontext, flags: u32, dev: CUdevice) -> CUresult;
        fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
        fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
        fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
        fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
        fn cuMemcpyHtoD_v2(
            dst: CUdeviceptr,
            src: *const c_void,
            bytesize: usize,
        ) -> CUresult;
        fn cuMemcpyDtoH_v2(
            dst: *mut c_void,
            src: CUdeviceptr,
            bytesize: usize,
        ) -> CUresult;
        fn cuCtxSynchronize() -> CUresult;

        // cuBLAS API.
        fn cublasCreate_v2(handle: *mut CublasHandle) -> CublasStatus;
        fn cublasDestroy_v2(handle: CublasHandle) -> CublasStatus;
        fn cublasSetMathMode(handle: CublasHandle, mode: c_int) -> CublasStatus;
        fn cublasSgemm_v2(
            handle: CublasHandle,
            transa: c_int,
            transb: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: *const f32,
            a: CUdeviceptr,
            lda: c_int,
            b: CUdeviceptr,
            ldb: c_int,
            beta: *const f32,
            c: CUdeviceptr,
            ldc: c_int,
        ) -> CublasStatus;
    }

    /// Persistent CUDA + cuBLAS state. One per process; handle is
    /// thread-safe.
    pub struct CudaContext {
        // Keep ctx alive for the program lifetime (drops are no-ops at
        // process shutdown — Linux reclaims everything).
        _ctx: CUcontext,
        cublas: CublasHandle,
        cache: Mutex<HashMap<(usize, usize, usize), CachedShape>>,
    }

    /// Pre-allocated device buffers for one (m, k, n) shape.
    struct CachedShape {
        d_a: CUdeviceptr,
        d_b: CUdeviceptr,
        d_c: CUdeviceptr,
    }

    /// Owned wrapper around a single `cuMemAlloc`'d buffer. Used by
    /// the device-resident path: every `matmul_device_resident` call
    /// allocates a fresh d_c whose Drop frees the allocation when the
    /// caller's `DeviceTensor` goes out of scope.
    pub struct CudaOwnedBuffer {
        pub ptr: CUdeviceptr,
        pub size: usize,
    }

    impl DeviceMem for CudaOwnedBuffer {
        fn ptr(&self) -> u64 {
            self.ptr
        }
        fn size_bytes(&self) -> usize {
            self.size
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl rce_vm_backend_cpu::DownloadToHost for CudaOwnedBuffer {
        fn download_bytes(&self) -> Option<Vec<u8>> {
            if self.size == 0 {
                return Some(Vec::new());
            }
            let mut host = vec![0u8; self.size];
            // SAFETY: cuMemcpyDtoH copies device → host bytes; we own
            // the destination Vec and the device pointer for the duration.
            let rc = unsafe {
                cuMemcpyDtoH_v2(
                    host.as_mut_ptr() as *mut c_void,
                    self.ptr,
                    self.size,
                )
            };
            if rc == CUDA_SUCCESS { Some(host) } else { None }
        }
    }

    impl Drop for CudaOwnedBuffer {
        fn drop(&mut self) {
            // Best-effort free: silently ignore failure (e.g., if the
            // CUDA context has already been torn down by another
            // shutdown path, nothing we can do).
            unsafe {
                let _ = cuMemFree_v2(self.ptr);
            }
        }
    }

    // SAFETY: CUDA device pointers are not bound to any single
    // host-side thread. cuMemFree_v2 is safe to call from any thread
    // that has the CUDA context current; the bench thread always
    // does. CudaOwnedBuffer is Arc-able from any thread.
    unsafe impl Send for CudaOwnedBuffer {}
    unsafe impl Sync for CudaOwnedBuffer {}

    // SAFETY: cuBLAS handles are documented thread-safe; CUcontext is
    // current-thread only but we set it at every entry via
    // cuCtxSetCurrent. CachedShape's CUdeviceptr values are immutable
    // once allocated.
    unsafe impl Send for CudaContext {}
    unsafe impl Sync for CudaContext {}
    unsafe impl Send for CachedShape {}
    unsafe impl Sync for CachedShape {}

    impl CudaContext {
        pub fn new(device_ordinal: c_int) -> Option<Self> {
            unsafe {
                if cuInit(0) != CUDA_SUCCESS {
                    return None;
                }
                let mut dev: CUdevice = 0;
                if cuDeviceGet(&mut dev, device_ordinal) != CUDA_SUCCESS {
                    return None;
                }
                let mut ctx: CUcontext = std::ptr::null_mut();
                if cuCtxCreate_v2(&mut ctx, 0, dev) != CUDA_SUCCESS {
                    return None;
                }
                let mut handle: CublasHandle = std::ptr::null_mut();
                if cublasCreate_v2(&mut handle) != CUBLAS_STATUS_SUCCESS {
                    cuCtxDestroy_v2(ctx);
                    return None;
                }
                // Match PyTorch's default Hopper f32 fast-path: TF32
                // tensor-core math for f32 GEMM. Without this, cuBLAS
                // runs strict-IEEE FP32 which on H100 is ~8× slower
                // than the tensor-core TF32 path.
                let _ = cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
                Some(Self {
                    _ctx: ctx,
                    cublas: handle,
                    cache: Mutex::new(HashMap::new()),
                })
            }
        }

        fn ensure_shape(&self, m: usize, k: usize, n: usize) -> std::result::Result<(), String> {
            let mut cache = self.cache.lock().expect("cuda cache mutex");
            if cache.contains_key(&(m, k, n)) {
                return Ok(());
            }
            unsafe {
                let mut d_a: CUdeviceptr = 0;
                let mut d_b: CUdeviceptr = 0;
                let mut d_c: CUdeviceptr = 0;
                if cuMemAlloc_v2(&mut d_a, m * k * 4) != CUDA_SUCCESS {
                    return Err("cuMemAlloc d_a failed".into());
                }
                if cuMemAlloc_v2(&mut d_b, k * n * 4) != CUDA_SUCCESS {
                    cuMemFree_v2(d_a);
                    return Err("cuMemAlloc d_b failed".into());
                }
                if cuMemAlloc_v2(&mut d_c, m * n * 4) != CUDA_SUCCESS {
                    cuMemFree_v2(d_a);
                    cuMemFree_v2(d_b);
                    return Err("cuMemAlloc d_c failed".into());
                }
                cache.insert((m, k, n), CachedShape { d_a, d_b, d_c });
            }
            Ok(())
        }

        /// Upload `bytes` host bytes to a fresh device allocation,
        /// returning a `DeviceMem` handle whose Drop frees the buffer.
        pub fn upload(
            &self,
            host: &[u8],
            bytes: usize,
        ) -> std::result::Result<Arc<dyn DeviceMem>, String> {
            unsafe {
                if cuCtxSetCurrent(self._ctx) != CUDA_SUCCESS {
                    return Err("cuCtxSetCurrent (upload) failed".into());
                }
                let mut dptr: CUdeviceptr = 0;
                if cuMemAlloc_v2(&mut dptr, bytes) != CUDA_SUCCESS {
                    return Err(format!("cuMemAlloc({bytes}) failed"));
                }
                if cuMemcpyHtoD_v2(dptr, host.as_ptr() as *const c_void, bytes)
                    != CUDA_SUCCESS
                {
                    cuMemFree_v2(dptr);
                    return Err("cuMemcpyHtoD (upload) failed".into());
                }
                Ok(Arc::new(CudaOwnedBuffer { ptr: dptr, size: bytes }))
            }
        }

        /// Run `C = A · B` where A and B are already-resident device
        /// pointers. Allocates a fresh d_c, returns it wrapped in a
        /// `DeviceMem` whose Drop frees it. No host↔device memcpy.
        pub fn matmul_device_resident(
            &self,
            a_ptr: u64,
            b_ptr: u64,
            m: usize,
            k: usize,
            n: usize,
        ) -> std::result::Result<Arc<dyn DeviceMem>, String> {
            let c_bytes = m * n * 4;
            unsafe {
                if cuCtxSetCurrent(self._ctx) != CUDA_SUCCESS {
                    return Err("cuCtxSetCurrent (mm-device) failed".into());
                }
                let mut d_c: CUdeviceptr = 0;
                if cuMemAlloc_v2(&mut d_c, c_bytes) != CUDA_SUCCESS {
                    return Err(format!("cuMemAlloc({c_bytes}) for d_c failed"));
                }
                let alpha: f32 = 1.0;
                let beta: f32 = 0.0;
                let status = cublasSgemm_v2(
                    self.cublas,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    n as c_int,
                    m as c_int,
                    k as c_int,
                    &alpha,
                    b_ptr,
                    n as c_int,
                    a_ptr,
                    k as c_int,
                    &beta,
                    d_c,
                    n as c_int,
                );
                if status != CUBLAS_STATUS_SUCCESS {
                    cuMemFree_v2(d_c);
                    return Err(format!("cublasSgemm (device-resident) status={status}"));
                }
                if cuCtxSynchronize() != CUDA_SUCCESS {
                    cuMemFree_v2(d_c);
                    return Err("cuCtxSynchronize after device-resident sgemm failed".into());
                }
                Ok(Arc::new(CudaOwnedBuffer { ptr: d_c, size: c_bytes }))
            }
        }

        /// Run C = A @ B on the GPU. Reuses cached device buffers for
        /// repeat-shape calls. Blocks until the result is back on host.
        pub fn matmul(
            &self,
            a: &[f32],
            b: &[f32],
            c: &mut [f32],
            m: usize,
            k: usize,
            n: usize,
        ) -> std::result::Result<(), String> {
            self.ensure_shape(m, k, n)?;
            let cache = self.cache.lock().expect("cuda cache mutex");
            let shape = cache
                .get(&(m, k, n))
                .ok_or_else(|| "cuda: shape vanished from cache".to_string())?;

            unsafe {
                // Set context current on this thread (idempotent if it's
                // already current).
                if cuCtxSetCurrent(self._ctx) != CUDA_SUCCESS {
                    return Err("cuCtxSetCurrent failed".into());
                }

                // Host → device.
                if cuMemcpyHtoD_v2(shape.d_a, a.as_ptr() as *const c_void, m * k * 4)
                    != CUDA_SUCCESS
                {
                    return Err("cuMemcpyHtoD A failed".into());
                }
                if cuMemcpyHtoD_v2(shape.d_b, b.as_ptr() as *const c_void, k * n * 4)
                    != CUDA_SUCCESS
                {
                    return Err("cuMemcpyHtoD B failed".into());
                }

                // cuBLAS column-major sgemm trick:
                //   row-major C = A · B   (A: m×k, B: k×n, C: m×n)
                // ≡ col-major C^T = B^T · A^T
                //   computed as: sgemm(handle, NoTrans, NoTrans, n, m, k,
                //                      alpha, B, ldb=n, A, lda=k,
                //                      beta, C, ldc=n)
                // The bytes of the col-major result are exactly the
                // bytes of the row-major result we want.
                let alpha: f32 = 1.0;
                let beta: f32 = 0.0;
                let status = cublasSgemm_v2(
                    self.cublas,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    n as c_int,
                    m as c_int,
                    k as c_int,
                    &alpha,
                    shape.d_b,
                    n as c_int,
                    shape.d_a,
                    k as c_int,
                    &beta,
                    shape.d_c,
                    n as c_int,
                );
                if status != CUBLAS_STATUS_SUCCESS {
                    return Err(format!("cublasSgemm failed (status={status})"));
                }

                // Wait for the kernel + copy back to host.
                if cuCtxSynchronize() != CUDA_SUCCESS {
                    return Err("cuCtxSynchronize after sgemm failed".into());
                }
                if cuMemcpyDtoH_v2(c.as_mut_ptr() as *mut c_void, shape.d_c, m * n * 4)
                    != CUDA_SUCCESS
                {
                    return Err("cuMemcpyDtoH C failed".into());
                }
            }
            Ok(())
        }
    }

    impl Drop for CudaContext {
        fn drop(&mut self) {
            unsafe {
                if !self.cublas.is_null() {
                    cublasDestroy_v2(self.cublas);
                }
                let cache = self.cache.lock().unwrap();
                for shape in cache.values() {
                    cuMemFree_v2(shape.d_a);
                    cuMemFree_v2(shape.d_b);
                    cuMemFree_v2(shape.d_c);
                }
                if !self._ctx.is_null() {
                    cuCtxDestroy_v2(self._ctx);
                }
            }
        }
    }
}
