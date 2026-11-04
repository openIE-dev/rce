//! Device-resident tensor support.
//!
//! `DeviceTensor` is the dual of `HostTensor`: a shape + dtype handle
//! that points at memory living on an accelerator (GPU, NPU, …)
//! rather than CPU host memory. Op-leaves that route to a specific
//! silicon (e.g., `rce-op-cuda` → cuBLAS, `rce-op-mps` →
//! MetalPerformanceShaders) can return DeviceTensor outputs that the
//! next op consumes directly — eliminating the host↔device round-trip
//! that PyTorch's eager mode hides via persistent device tensors.
//!
//! Lifetime: each `DeviceTensor` carries an `Arc<dyn DeviceMem>`
//! handle whose `Drop` releases the underlying allocation. Cloning a
//! DeviceTensor bumps the refcount; the device memory is released
//! when the last clone drops. This mirrors `HostTensor` whose
//! `Vec<u8>` is dropped automatically.
//!
//! Device identity: the [`DeviceId`] enum tags which silicon the
//! pointer belongs to. A leaf MUST refuse a DeviceTensor whose
//! `device` doesn't match its own (returning `Unsupported` so the
//! dispatch chain falls through to a leaf that can transcribe).

use crate::tensor::DType;
use std::any::Any;
use std::sync::Arc;

/// Identifies which device a `DeviceTensor` lives on. Equality
/// determines whether two tensors can be operated on without going
/// through host memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceId {
    /// NVIDIA CUDA device, indexed by ordinal.
    Cuda(u32),
    /// Apple Metal device. Only one default device; ordinal omitted
    /// because Apple silicon today has a single integrated GPU.
    Metal,
}

impl DeviceId {
    pub fn name(&self) -> String {
        match self {
            DeviceId::Cuda(o) => format!("cuda:{o}"),
            DeviceId::Metal => "metal".to_string(),
        }
    }
}

/// Trait every leaf-specific device-allocation implementor erases to.
/// Held in `DeviceTensor::mem` as `Arc<dyn DeviceMem>` so the
/// underlying allocation is freed by the leaf that allocated it
/// (cuMemFree, releaseBuffer, etc.) when refcount → 0.
pub trait DeviceMem: Send + Sync + Any {
    /// Raw device pointer. Interpretation is leaf-specific:
    /// - Cuda: `CUdeviceptr` cast as `u64`
    /// - Metal: `MTLBuffer` object pointer
    fn ptr(&self) -> u64;

    /// Total bytes allocated. The slice `[ptr, ptr+size_bytes)` is
    /// addressable on the device. Useful for bounds checks.
    fn size_bytes(&self) -> usize;

    /// Downcast hook so a leaf can recover its own concrete
    /// `DeviceMem` impl (e.g., the metal::Buffer wrapped inside
    /// `MetalDeviceMem`) from a `&dyn DeviceMem`. Default impl returns
    /// `self`; concrete types just need `&Self: Any` which the
    /// `DeviceMem: Any` supertrait already guarantees.
    fn as_any(&self) -> &dyn Any;
}

/// A tensor whose data lives on an accelerator. Same shape/dtype
/// surface as `HostTensor`; pointer + memory handle replace the
/// host `Vec<u8>`. `Clone` is `Arc::clone`-cheap and keeps the
/// memory alive.
#[derive(Clone)]
pub struct DeviceTensor {
    pub device: DeviceId,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Lifetime-managed device allocation. Drop frees on the device.
    pub mem: Arc<dyn DeviceMem>,
}

impl DeviceTensor {
    pub fn new(
        device: DeviceId,
        dtype: DType,
        shape: Vec<usize>,
        mem: Arc<dyn DeviceMem>,
    ) -> Self {
        Self { device, dtype, shape, mem }
    }

    pub fn ptr(&self) -> u64 {
        self.mem.ptr()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.byte_size()
    }
}

impl std::fmt::Debug for DeviceTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DeviceTensor{{ device={}, dtype={}, shape={:?}, ptr=0x{:x}, bytes={} }}",
            self.device.name(),
            self.dtype.name(),
            self.shape,
            self.mem.ptr(),
            self.size_bytes()
        )
    }
}

impl PartialEq for DeviceTensor {
    /// Two DeviceTensors are equal iff they refer to the same
    /// underlying device allocation. There is no facility to compare
    /// element-by-element without a host round-trip; that's a
    /// deliberate restriction — host-side equality on device tensors
    /// is almost always a bug, since it implies a hidden memcpy.
    fn eq(&self, other: &Self) -> bool {
        self.device == other.device
            && self.dtype == other.dtype
            && self.shape == other.shape
            && Arc::ptr_eq(&self.mem, &other.mem)
    }
}

// ── Cross-device transfer ─────────────────────────────────────────

/// Trait every leaf-specific `DeviceMem` impl can implement to
/// expose download-to-host capability. Leaf crates (`rce-op-metal`,
/// `rce-op-cuda`) implement this for their concrete
/// `MetalDeviceMem` / `CudaOwnedBuffer` types so the orchestration
/// layer can read device bytes back without hard-coding
/// device-specific knowledge.
///
/// **Cross-device transfer recipes**:
///
/// Metal → CUDA (on a hybrid host where both leaves are loaded):
/// ```ignore
/// let host = rce_op_metal::download_via_host(&metal_dt)?;
/// let cuda_dt = cublas.host_to_device(&host)?;
/// ```
///
/// CUDA → Metal:
/// ```ignore
/// let host = rce_op_cuda::download_via_host(&cuda_dt)?;
/// let metal_v = metal.promote_to_device(&host)?;
/// ```
///
/// Both directions route through the host as the universal
/// interconnect. Direct device-to-device paths (CUDA peer-to-peer,
/// multi-GPU Metal IOSurface) plug in as overrides on specific
/// device pairs, dispatched via `as_any` downcast on the `DeviceMem`
/// impl. Today the host roundtrip is the only path; the API surface
/// for direct paths exists (`as_any`) but no leaf pair has shipped a
/// fast-path implementation yet.
///
/// This trait is intentionally minimal: a leaf can implement it
/// without committing to a particular cross-device protocol.
pub trait DownloadToHost {
    /// Read all bytes addressable by this `DeviceMem` back into a
    /// host-side byte buffer. Returns `None` on failure (e.g., if the
    /// underlying resource has been torn down).
    fn download_bytes(&self) -> Option<Vec<u8>>;
}
