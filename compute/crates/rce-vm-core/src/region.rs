/// Memory region kinds. The scheduler tracks region affinity per-tensor
/// and inserts `Move`/copy edges as required by `WireKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegionKind {
    /// CPU heap.
    Host,
    /// GPU/NPU device memory, identified by `device_id`.
    Device,
    /// UMA shared (Apple Silicon, Adreno).
    Shared,
    /// CUDA/ROCm Unified Virtual Memory.
    Unified,
    /// Page-locked host memory for DMA.
    Pinned,
    /// File-backed mmap (e.g. weight sidecar pages).
    Mapped,
}

#[derive(Debug, Clone)]
pub struct Region {
    pub kind: RegionKind,
    /// Backend or device identifier; format is backend-specific.
    pub locator: String,
}

impl Region {
    pub fn host() -> Self {
        Self {
            kind: RegionKind::Host,
            locator: "host".into(),
        }
    }

    pub fn device(locator: impl Into<String>) -> Self {
        Self {
            kind: RegionKind::Device,
            locator: locator.into(),
        }
    }

    pub fn shared() -> Self {
        Self {
            kind: RegionKind::Shared,
            locator: "shared".into(),
        }
    }
}
