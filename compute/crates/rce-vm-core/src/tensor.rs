use crate::region::Region;
use smallvec::SmallVec;

/// rce dtype names — must match `rce_graph::graph::dtype_byte_size`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    I8,
    U8,
    I16,
    U16,
    F16,
    Bf16,
    F32,
    I32,
    U32,
    F64,
    I64,
    U64,
    Complex64,
}

impl DType {
    pub fn byte_size(&self) -> usize {
        match self {
            DType::Bool | DType::I8 | DType::U8 => 1,
            DType::F16 | DType::Bf16 | DType::I16 | DType::U16 => 2,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F64 | DType::I64 | DType::U64 | DType::Complex64 => 8,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "bool" => DType::Bool,
            "i8" => DType::I8,
            "u8" => DType::U8,
            "i16" => DType::I16,
            "u16" => DType::U16,
            "f16" => DType::F16,
            "bf16" => DType::Bf16,
            "f32" => DType::F32,
            "i32" => DType::I32,
            "u32" => DType::U32,
            "f64" => DType::F64,
            "i64" => DType::I64,
            "u64" => DType::U64,
            "complex64" => DType::Complex64,
            _ => return None,
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::I8 => "i8",
            DType::U8 => "u8",
            DType::I16 => "i16",
            DType::U16 => "u16",
            DType::F16 => "f16",
            DType::Bf16 => "bf16",
            DType::F32 => "f32",
            DType::I32 => "i32",
            DType::U32 => "u32",
            DType::F64 => "f64",
            DType::I64 => "i64",
            DType::U64 => "u64",
            DType::Complex64 => "complex64",
        }
    }
}

/// Tensor descriptor: shape + dtype + region. Storage is via `TensorHandle`.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: SmallVec<[u64; 6]>,
    pub dtype: DType,
    pub region: Region,
}

impl Tensor {
    pub fn numel(&self) -> u64 {
        self.shape.iter().product()
    }

    pub fn byte_size(&self) -> u64 {
        self.numel() * self.dtype.byte_size() as u64
    }
}

/// Opaque handle a backend hands to the scheduler. The interpretation is
/// backend-specific (e.g. a `wgpu::Buffer` ID, a CUDA device pointer, a
/// Wasm host-call result index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorHandle(pub u64);
