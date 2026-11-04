//! `rce-op-mps` — `OpDispatch` leaf that routes large f32 row-major
//! 2-D matmul to Apple's `MPSMatrixMultiplication` (MetalPerformance-
//! Shaders). On M3-class silicon this lights up the GPU's simdgroup
//! matrix-multiply tensor units, hitting ~5–15 TFLOPS f32 — roughly an
//! order of magnitude above PyTorch's default CPU Accelerate path for
//! workloads ≥ matmul-256.
//!
//! ## Why a separate leaf
//!
//! The existing `rce-vm-backend-wgpu` matmul kernel goes through a
//! generic WGSL tiled GEMM. wgpu can't currently emit Apple's
//! `simdgroup_multiply_accumulate` MSL intrinsic — that path goes
//! through MPS, which is a hand-tuned Apple kernel that wgpu doesn't
//! call into. So we add a *separate* OpDispatch leaf that talks to
//! MetalPerformanceShaders directly via objc message sends. The leaf
//! is registered ahead of the wgpu executor for matmul; wgpu still
//! handles softmax, transpose, activations, etc.
//!
//! ## Threshold
//!
//! Per-call, the MPS path pays roughly 80–120 µs of dispatch overhead
//! (Metal command-buffer + commit + wait). Below `MPS_MIN_FLOPS` the
//! BLAS leaf wins, so we return `Unsupported` and let the chain fall
//! through. Above the threshold the GPU dominates.
//!
//! ## Cross-platform
//!
//! Non-macOS / non-iOS builds compile to a stub that always returns
//! `Unsupported`, so the workspace builds cleanly on Linux / Windows
//! and the bench's CPU/BLAS path remains the active matmul backend.

use rce_vm_backend_cpu::{InterpretError, OpDispatch, Result, Value};
use rce_graph::graph::OpKind;

/// `OpDispatch` leaf that runs f32 row-major 2-D matmul on the GPU
/// via `MPSMatrixMultiplication`. Shares one Metal device + command
/// queue across all calls (created at construction time).
pub struct MpsMatmul {
    #[cfg(rce_mps)]
    inner: imp::MpsContext,
    /// Below this FLOP count, the leaf returns `Unsupported` so the
    /// chain can fall through to BLAS (which beats MPS on small
    /// matmuls due to the GPU dispatch overhead). Tunable per-instance
    /// because workloads with **many small matmuls** (e.g. LLM decode
    /// with batch=1) want a lower threshold than the default
    /// matmul-2048-ish crossover.
    min_flops: usize,
}

impl MpsMatmul {
    /// Default matmul-2048-ish threshold (10 GFLOPs) — BLAS wins below
    /// this on M3 Ultra.
    pub const DEFAULT_MIN_FLOPS: usize = 10_000_000_000;

    /// Create a new MPS leaf bound to the system default Metal device.
    /// Returns `None` if Metal is unavailable (headless CI, sandbox,
    /// non-macOS target).
    pub fn new() -> Option<Self> {
        Self::with_min_flops(Self::DEFAULT_MIN_FLOPS)
    }

    /// Like `new` but with a tunable BLAS-fallback threshold (in FLOPs
    /// = `2·m·k·n`). Set `0` to always run on MPS regardless of size.
    /// Useful for LLM decode where batch=1 keeps individual matmuls
    /// below the default crossover, but the *aggregate* per-forward
    /// matmul work still benefits from the GPU.
    pub fn with_min_flops(min_flops: usize) -> Option<Self> {
        #[cfg(rce_mps)]
        {
            imp::MpsContext::new().map(|inner| Self { inner, min_flops })
        }
        #[cfg(not(rce_mps))]
        {
            let _ = min_flops;
            None
        }
    }
}

impl OpDispatch for MpsMatmul {
    fn handles(&self) -> &[&'static str] {
        &["matmul"]
    }

    fn execute(&self, node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "matmul" {
            return Err(InterpretError::Unsupported(format!(
                "mps leaf does not handle `{name}`"
            )));
        }
        if args.len() != 2 {
            return Err(InterpretError::Unsupported(format!(
                "mps matmul: expected 2 args, got {}",
                args.len()
            )));
        }
        // Accept Value::Tensor (host) OR Value::DeviceTensor (Metal) for
        // each input. The second case borrows the underlying MTLBuffer's
        // shared-storage contents() pointer via rce-op-metal's
        // metal_buffer_of accessor — zero intermediate Vec<f32>, no
        // demote_to_host. Tensor + DeviceTensor inputs mix freely.
        let (a_shape, a_dtype) = match &args[0] {
            Value::Tensor(t) => (t.shape.clone(), t.dtype),
            Value::DeviceTensor(d) => (d.shape.clone(), d.dtype),
            _ => return Err(InterpretError::Unsupported(
                "mps matmul: arg 0 must be a tensor (host or device)".into())),
        };
        let (b_shape, b_dtype) = match &args[1] {
            Value::Tensor(t) => (t.shape.clone(), t.dtype),
            Value::DeviceTensor(d) => (d.shape.clone(), d.dtype),
            _ => return Err(InterpretError::Unsupported(
                "mps matmul: arg 1 must be a tensor (host or device)".into())),
        };
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(InterpretError::Unsupported(format!(
                "mps matmul: rank-2 only (got {a_shape:?} × {b_shape:?})"
            )));
        }
        if a_dtype != rce_vm_backend_cpu::DType::F32
            || b_dtype != rce_vm_backend_cpu::DType::F32
        {
            return Err(InterpretError::Unsupported(format!(
                "mps matmul: f32 only (got {} × {})",
                a_dtype.name(), b_dtype.name()
            )));
        }
        let m = a_shape[0];
        let k = a_shape[1];
        let kk = b_shape[0];
        let n = b_shape[1];
        if k != kk {
            return Err(InterpretError::Unsupported(format!(
                "mps matmul: inner dim {k} vs {kk}"
            )));
        }
        if m == 0 || n == 0 || k == 0 {
            return Err(InterpretError::Unsupported(
                "mps matmul: zero-size dim, fall through".into(),
            ));
        }

        // Below threshold, BLAS wins. (Calibration as before.)
        let work = 2usize.saturating_mul(m).saturating_mul(k).saturating_mul(n);
        if work < self.min_flops {
            return Err(InterpretError::Unsupported(
                "mps matmul: below configured min_flops, defer to BLAS leaf".into(),
            ));
        }

        #[cfg(rce_mps)]
        {
            // Resolve (ptr, byte_len) for each input regardless of residency.
            // For DeviceTensor we borrow the MTLBuffer's contents() — a
            // CPU-visible pointer to the same unified-memory address the
            // GPU was reading from on the upstream dispatch. Zero
            // demote_to_host.
            let (a_ptr, a_bytes) = match &args[0] {
                Value::Tensor(t) => (t.data.as_ptr(), t.data.len()),
                Value::DeviceTensor(d) => {
                    let buf = rce_op_metal::metal_buffer_of(d).ok_or_else(|| {
                        InterpretError::Unsupported(
                            "mps matmul: DeviceTensor arg 0 not on Metal or not from MetalLeaf"
                                .into(),
                        )
                    })?;
                    (buf.contents() as *const u8, d.mem.size_bytes())
                }
                _ => unreachable!(),
            };
            let (b_ptr, b_bytes) = match &args[1] {
                Value::Tensor(t) => (t.data.as_ptr(), t.data.len()),
                Value::DeviceTensor(d) => {
                    let buf = rce_op_metal::metal_buffer_of(d).ok_or_else(|| {
                        InterpretError::Unsupported(
                            "mps matmul: DeviceTensor arg 1 not on Metal or not from MetalLeaf"
                                .into(),
                        )
                    })?;
                    (buf.contents() as *const u8, d.mem.size_bytes())
                }
                _ => unreachable!(),
            };
            // Preservation: any DeviceTensor input ⇒ DeviceTensor output
            // (matches MetalLeaf's host-in/host-out, device-in/device-out
            // policy). Chains of consecutive same-device ops stay resident
            // end-to-end including the matmul output.
            let any_device = matches!(args[0], Value::DeviceTensor(_))
                || matches!(args[1], Value::DeviceTensor(_));
            tracing::trace!(target: "rce::op::mps", node = node_id, m, k, n, "MPSMatrixMultiplication dispatched");
            if any_device {
                let buf = unsafe {
                    self.inner
                        .matmul_bytes_to_buffer(a_ptr, a_bytes, b_ptr, b_bytes, m, k, n)
                        .map_err(|e| InterpretError::Unsupported(format!("mps matmul: {e}")))?
                };
                Ok(rce_op_metal::wrap_metal_buffer(
                    buf, vec![m, n], rce_vm_backend_cpu::DType::F32,
                ))
            } else {
                let out = unsafe {
                    self.inner
                        .matmul_bytes(a_ptr, a_bytes, b_ptr, b_bytes, m, k, n)
                        .map_err(|e| InterpretError::Unsupported(format!("mps matmul: {e}")))?
                };
                rce_vm_backend_cpu::HostTensor::from_f32(vec![m, n], &out)
                    .map(Value::Tensor)
                    .map_err(|e| InterpretError::Unsupported(format!(
                        "mps matmul: build out tensor: {e:?}"
                    )))
            }
        }

        #[cfg(not(rce_mps))]
        {
            let _ = (node_id, m, k, n);
            Err(InterpretError::Unsupported(
                "mps matmul: not built with MPS support".into(),
            ))
        }
    }

    fn execute_primitive(
        &self,
        _node_id: usize,
        _op: &OpKind,
        _args: &[Value],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "mps leaf: no primitive ops".into(),
        ))
    }
}

// ── macOS / iOS implementation ──────────────────────────────────────

#[cfg(rce_mps)]
mod imp {
    //! Direct MPSMatrixMultiplication binding via objc message sends.
    //!
    //! Per-shape resource cache: every distinct (m, k, n) shape gets
    //! its own MTLBuffer trio + descriptor trio + matrix-wrapper trio
    //! + MPSMatrixMultiplication kernel, all created on first use and
    //! reused on every subsequent call. After the first warm-up pass
    //! per shape, dispatch cost is ~one memcpy per input + one
    //! commit/wait round-trip — no allocation, no obj-c churn — so
    //! the GPU's tensor units actually get to dominate the timing.
    //!
    //! We don't rely on `metal-rs`'s MPS module because the workspace-
    //! pinned 0.33 release covers ray-tracing accel structures only —
    //! `MPSMatrixMultiplication` is absent. Talking to the Objective-C
    //! runtime ourselves is ~150 lines and avoids dragging in the
    //! `objc2-metal-performance-shaders` ecosystem.

    use foreign_types::{ForeignType, ForeignTypeRef};
    use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};
    use objc::rc::StrongPtr;
    use objc::runtime::{Class, Object, BOOL, NO};
    use objc::{class, msg_send, sel, sel_impl};
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::sync::Mutex;

    /// `MPSDataType` is a 32-bit bitfield: bit 28 = float, low byte = bit width.
    /// `MPSDataTypeFloat32 = 0x10000020`.
    const MPS_DATA_TYPE_FLOAT32: u32 = 0x1000_0020;

    /// Owned wrapper around a Metal device + queue + per-shape cache.
    /// Send + Sync so it can sit in `Arc<dyn OpDispatch + Send + Sync>`.
    pub struct MpsContext {
        device: Device,
        queue: CommandQueue,
        cache: Mutex<HashMap<(usize, usize, usize), CachedShape>>,
    }

    /// Pre-allocated Metal + MPS objects for a particular (m, k, n)
    /// shape. The MTLBuffers are shared-storage so `Buffer::contents`
    /// gives us a CPU-visible pointer for memcpy. We hold StrongPtrs
    /// (+1 retained) on every NS object so they outlive each call.
    struct CachedShape {
        buf_a: Buffer,
        buf_b: Buffer,
        buf_c: Buffer,
        mat_a: StrongPtr,
        mat_b: StrongPtr,
        mat_c: StrongPtr,
        kernel: StrongPtr,
    }

    // SAFETY: Metal `Device`, `CommandQueue`, and `Buffer` are
    // thread-safe per Apple's docs as long as command buffers aren't
    // shared mid-encode. The cached MTLBuffers are only ever read /
    // written from the locked-cache path. The retained NS objects
    // (StrongPtr) are immutable after creation.
    unsafe impl Send for CachedShape {}
    unsafe impl Sync for CachedShape {}
    unsafe impl Send for MpsContext {}
    unsafe impl Sync for MpsContext {}

    impl MpsContext {
        pub fn new() -> Option<Self> {
            let device = Device::system_default()?;
            // Probe: bail early if MPS isn't linked in.
            let cls_name = std::ffi::CString::new("MPSMatrixMultiplication").ok()?;
            let cls_ptr = unsafe { Class::get(cls_name.to_str().ok()?) };
            cls_ptr?;
            let queue = device.new_command_queue();
            Some(Self {
                device,
                queue,
                cache: Mutex::new(HashMap::new()),
            })
        }

        /// Lazily build (or fetch) the cached resources for shape (m, k, n).
        /// Holds the cache mutex briefly while installing a new entry; the
        /// rest of the call uses the cached buffers/kernel by reference.
        fn ensure_shape(&self, m: usize, k: usize, n: usize) {
            let mut cache = self.cache.lock().expect("mps cache mutex");
            if cache.contains_key(&(m, k, n)) {
                return;
            }
            let shape = unsafe { build_shape(&self.device, m, k, n) };
            cache.insert((m, k, n), shape);
        }

        /// Run C = A @ B on the GPU from typed f32 host slices. Returns C
        /// as a host `Vec<f32>`. Thin wrapper around `matmul_bytes`.
        pub fn matmul(
            &self,
            a: &[f32],
            b: &[f32],
            m: usize,
            k: usize,
            n: usize,
        ) -> std::result::Result<Vec<f32>, String> {
            unsafe {
                self.matmul_bytes(
                    a.as_ptr() as *const u8, a.len() * std::mem::size_of::<f32>(),
                    b.as_ptr() as *const u8, b.len() * std::mem::size_of::<f32>(),
                    m, k, n,
                )
            }
        }

        /// Run C = A @ B from raw byte pointers. The caller guarantees the
        /// pointers are valid for `m*k*4` and `k*n*4` bytes respectively
        /// and point to f32 row-major data. Returns C as a host `Vec<f32>`.
        ///
        /// **Zero host roundtrip on input**: callers can pass
        /// `MTLBuffer::contents()` from a Metal `DeviceTensor` directly
        /// (shared-storage buffers are CPU-visible at the same address the
        /// GPU reads from), so a chain like `MetalLeaf → MpsMatmul` skips
        /// the device→host→device staging the per-arg `as_f32()` path
        /// would force.
        ///
        /// # Safety
        /// `a_ptr` valid for `a_bytes`, `b_ptr` valid for `b_bytes`, both
        /// readable for the duration of the call. `a_bytes >= m*k*4` and
        /// `b_bytes >= k*n*4`. f32 element layout (little-endian, native
        /// alignment satisfied since memcpy is byte-wise).
        pub unsafe fn matmul_bytes(
            &self,
            a_ptr: *const u8, a_bytes: usize,
            b_ptr: *const u8, b_bytes: usize,
            m: usize, k: usize, n: usize,
        ) -> std::result::Result<Vec<f32>, String> {
            let need_a = m * k * std::mem::size_of::<f32>();
            let need_b = k * n * std::mem::size_of::<f32>();
            if a_bytes < need_a {
                return Err(format!("matmul_bytes: a_bytes={a_bytes} < {need_a} (need m·k·4)"));
            }
            if b_bytes < need_b {
                return Err(format!("matmul_bytes: b_bytes={b_bytes} < {need_b} (need k·n·4)"));
            }
            self.ensure_shape(m, k, n);
            let cache = self.cache.lock().expect("mps cache mutex");
            let shape = cache
                .get(&(m, k, n))
                .ok_or_else(|| "mps: shape vanished from cache".to_string())?;

            // ── 1. Copy src → cached shared-storage MTLBuffer ───────
            //
            // Single CPU memcpy per input regardless of where the source
            // lives (host Vec, MTLBuffer contents, anywhere). For an MPS
            // input that came in as a Metal DeviceTensor, this *is* the
            // only memcpy — no intermediate Vec<f32> materialization, no
            // demote_to_host. Apple unified memory makes contents() a
            // CPU-visible pointer for shared-storage MTLBuffers.
            let dst_a = shape.buf_a.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(a_ptr, dst_a, need_a);
            let dst_b = shape.buf_b.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(b_ptr, dst_b, need_b);

            // ── 2. Encode + commit + wait ───────────────────────────
            let cmd_buffer = self.queue.new_command_buffer();
            let cmd_buffer_ptr: *mut Object =
                ForeignTypeRef::as_ptr(cmd_buffer) as *mut Object;
            let _: () = msg_send![
                *shape.kernel,
                encodeToCommandBuffer: cmd_buffer_ptr
                leftMatrix: *shape.mat_a
                rightMatrix: *shape.mat_b
                resultMatrix: *shape.mat_c
            ];
            cmd_buffer.commit();
            cmd_buffer.wait_until_completed();

            // ── 3. Read C back from the shared buffer ───────────────
            let mut out = vec![0.0_f32; m * n];
            let src = shape.buf_c.contents() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), m * n);
            Ok(out)
        }

        /// Same as `matmul_bytes` but returns a fresh shared-storage
        /// `Buffer` carrying the result instead of materialising a host
        /// `Vec<f32>` — the entry point for the device-resident output
        /// path. Callers wrap the buffer via
        /// `rce_op_metal::wrap_metal_buffer` and ship it on the wire
        /// as `Value::DeviceTensor`, so a downstream same-device
        /// consumer reads from it with zero host roundtrip.
        ///
        /// One CPU memcpy from the cached shape buffer into the fresh
        /// output buffer (the cache stays warm for the next call); the
        /// downstream consumer's upload is what this trade saves.
        ///
        /// # Safety
        /// Same contract as `matmul_bytes`.
        pub unsafe fn matmul_bytes_to_buffer(
            &self,
            a_ptr: *const u8, a_bytes: usize,
            b_ptr: *const u8, b_bytes: usize,
            m: usize, k: usize, n: usize,
        ) -> std::result::Result<Buffer, String> {
            let need_a = m * k * std::mem::size_of::<f32>();
            let need_b = k * n * std::mem::size_of::<f32>();
            let need_c = m * n * std::mem::size_of::<f32>();
            if a_bytes < need_a {
                return Err(format!("matmul_bytes_to_buffer: a_bytes={a_bytes} < {need_a}"));
            }
            if b_bytes < need_b {
                return Err(format!("matmul_bytes_to_buffer: b_bytes={b_bytes} < {need_b}"));
            }
            self.ensure_shape(m, k, n);
            let cache = self.cache.lock().expect("mps cache mutex");
            let shape = cache.get(&(m, k, n))
                .ok_or_else(|| "mps: shape vanished from cache".to_string())?;

            let dst_a = shape.buf_a.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(a_ptr, dst_a, need_a);
            let dst_b = shape.buf_b.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(b_ptr, dst_b, need_b);

            let cmd_buffer = self.queue.new_command_buffer();
            let cmd_buffer_ptr: *mut Object =
                ForeignTypeRef::as_ptr(cmd_buffer) as *mut Object;
            let _: () = msg_send![
                *shape.kernel,
                encodeToCommandBuffer: cmd_buffer_ptr
                leftMatrix: *shape.mat_a
                rightMatrix: *shape.mat_b
                resultMatrix: *shape.mat_c
            ];
            cmd_buffer.commit();
            cmd_buffer.wait_until_completed();

            // Allocate a fresh device-resident output buffer + memcpy
            // from the cached buf_c contents. Both buffers live in
            // unified memory; the memcpy is just CPU-side DRAM-to-DRAM.
            // The fresh buffer is what we hand the caller — the cache's
            // buf_c stays for the next dispatch.
            let fresh = self.device.new_buffer(need_c as u64, MTLResourceOptions::StorageModeShared);
            let src = shape.buf_c.contents() as *const u8;
            let dst = fresh.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, need_c);
            Ok(fresh)
        }
    }

    /// Build all per-shape MTLBuffers + MPS objects for one (m, k, n)
    /// shape. Called once per shape from `ensure_shape`.
    unsafe fn build_shape(device: &Device, m: usize, k: usize, n: usize) -> CachedShape {
        let bytes_a = (m * k * std::mem::size_of::<f32>()) as u64;
        let bytes_b = (k * n * std::mem::size_of::<f32>()) as u64;
        let bytes_c = (m * n * std::mem::size_of::<f32>()) as u64;

        let buf_a = device.new_buffer(bytes_a, MTLResourceOptions::StorageModeShared);
        let buf_b = device.new_buffer(bytes_b, MTLResourceOptions::StorageModeShared);
        let buf_c = device.new_buffer(bytes_c, MTLResourceOptions::StorageModeShared);

        let desc_class = class!(MPSMatrixDescriptor);
        let desc_a: *mut Object = msg_send![
            desc_class,
            matrixDescriptorWithRows: m as u64
            columns: k as u64
            rowBytes: (k * std::mem::size_of::<f32>()) as u64
            dataType: MPS_DATA_TYPE_FLOAT32
        ];
        let desc_b: *mut Object = msg_send![
            desc_class,
            matrixDescriptorWithRows: k as u64
            columns: n as u64
            rowBytes: (n * std::mem::size_of::<f32>()) as u64
            dataType: MPS_DATA_TYPE_FLOAT32
        ];
        let desc_c: *mut Object = msg_send![
            desc_class,
            matrixDescriptorWithRows: m as u64
            columns: n as u64
            rowBytes: (n * std::mem::size_of::<f32>()) as u64
            dataType: MPS_DATA_TYPE_FLOAT32
        ];

        let mat_class = class!(MPSMatrix);
        let mat_a = init_with_buffer_descriptor(mat_class, buffer_object(&buf_a), desc_a);
        let mat_b = init_with_buffer_descriptor(mat_class, buffer_object(&buf_b), desc_b);
        let mat_c = init_with_buffer_descriptor(mat_class, buffer_object(&buf_c), desc_c);

        let mm_class = class!(MPSMatrixMultiplication);
        let mm_alloc: *mut Object = msg_send![mm_class, alloc];
        let device_ptr: *mut Object = ForeignTypeRef::as_ptr(&**device) as *mut Object;
        let mm_obj: *mut Object = msg_send![
            mm_alloc,
            initWithDevice: device_ptr
            transposeLeft: NO as BOOL
            transposeRight: NO as BOOL
            resultRows: m as u64
            resultColumns: n as u64
            interiorColumns: k as u64
            alpha: 1.0_f64
            beta: 0.0_f64
        ];
        let kernel = StrongPtr::new(mm_obj);

        CachedShape {
            buf_a,
            buf_b,
            buf_c,
            mat_a,
            mat_b,
            mat_c,
            kernel,
        }
    }

    /// Wrap a `Buffer`'s underlying NS object as an `*mut Object`.
    unsafe fn buffer_object(b: &Buffer) -> *mut Object {
        ForeignTypeRef::as_ptr(&**b) as *mut Object
    }

    /// Send `[[<class> alloc] initWithBuffer:offset:descriptor:]` and
    /// return a +1-retained StrongPtr.
    unsafe fn init_with_buffer_descriptor(
        cls: &Class,
        buffer: *mut Object,
        descriptor: *mut Object,
    ) -> StrongPtr {
        let alloc: *mut Object = msg_send![cls, alloc];
        let inited: *mut Object = msg_send![
            alloc,
            initWithBuffer: buffer
            offset: 0u64
            descriptor: descriptor
        ];
        StrongPtr::new(inited)
    }
}

// ── Tests (cfg-gated, since they need a Metal device) ────────────────

#[cfg(all(test, rce_mps))]
mod tests {
    use super::*;
    use rce_vm_backend_cpu::{HostTensor, Value};

    #[test]
    fn mps_matches_naive_above_threshold() {
        // Pick a size whose 2*m*k*n exceeds MPS_MIN_FLOPS (10 GFLOPs)
        // so the leaf actually dispatches to the GPU. matmul-1700^3
        // ≈ 9.83 GFLOPs is just under, so use 1750.
        let leaf = match MpsMatmul::new() {
            Some(l) => l,
            None => {
                eprintln!("[mps test] no Metal device available; skipping");
                return;
            }
        };
        let m = 1750;
        let k = 1750;
        let n = 1750;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32) * 0.02 + 0.1).collect();

        let ta = HostTensor::from_f32(vec![m, k], &a).unwrap();
        let tb = HostTensor::from_f32(vec![k, n], &b).unwrap();
        let result = leaf
            .execute(0, "matmul", &[Value::Tensor(ta), Value::Tensor(tb)])
            .expect("mps matmul");
        let out = match result {
            Value::Tensor(t) => t,
            _ => panic!("expected tensor"),
        };
        let got = out.as_f32().unwrap();

        // Naive reference for matmul-1750 would be ~5 GFLOP — too
        // slow for a test. Spot-check the corner element (0, 0)
        // which is just `k` multiplies, plus the (m-1, n-1) corner.
        let want_00: f32 = (0..k).map(|kk| a[kk] * b[kk * n]).sum();
        let want_corner: f32 = (0..k)
            .map(|kk| a[(m - 1) * k + kk] * b[kk * n + (n - 1)])
            .sum();
        let got_00 = got[0];
        let got_corner = got[(m - 1) * n + (n - 1)];
        let rel_00 = (want_00 - got_00).abs() / want_00.abs().max(1.0);
        let rel_corner =
            (want_corner - got_corner).abs() / want_corner.abs().max(1.0);
        assert!(rel_00 < 1e-3, "(0,0): want={want_00} got={got_00} rel={rel_00}");
        assert!(
            rel_corner < 1e-3,
            "corner: want={want_corner} got={got_corner} rel={rel_corner}"
        );

        // Sanity: every output is finite.
        for v in &got {
            assert!(v.is_finite(), "non-finite output value {v}");
        }
    }

    #[test]
    fn falls_through_below_threshold() {
        let leaf = match MpsMatmul::new() {
            Some(l) => l,
            None => {
                eprintln!("[mps test] no Metal device available; skipping");
                return;
            }
        };
        // 4x4 matmul is below threshold (~30M flops); should fall through.
        let a = HostTensor::from_f32(vec![4, 4], &[1.0; 16]).unwrap();
        let b = HostTensor::from_f32(vec![4, 4], &[1.0; 16]).unwrap();
        let err = leaf
            .execute(0, "matmul", &[Value::Tensor(a), Value::Tensor(b)])
            .unwrap_err();
        assert!(matches!(err, InterpretError::Unsupported(_)));
    }
}
