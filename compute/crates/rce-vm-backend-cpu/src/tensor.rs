//! Host-resident tensor representation for the v0.1 CPU interpreter.
//!
//! `HostTensor` holds row-major (C-order) bytes plus shape and dtype.
//! Element-wise kernels for `Add | Sub | Mul | Div | Rem | Neg` and
//! comparisons live in `kernel.rs`. Broadcasting in v0.1 is limited to
//! scalar-on-tensor; full NumPy-style broadcasting is on the next iteration.

use crate::error::{InterpretError, Result};
use crate::value::ScalarValue;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    Bool,
    I32,
    I64,
    F32,
    F64,
    /// IEEE 754 half precision (5-bit exponent, 10-bit mantissa).
    /// Phase O of the codegen rollout. Most existing kernels promote
    /// f16 → f32 transparently via `as_f32_promoted`; the GPU
    /// dispatchers fast-path on raw f16 byte data.
    F16,
    /// Brain float 16 (8-bit exponent, 7-bit mantissa). Same exponent
    /// range as f32 — preferred over f16 for matmul accumulation
    /// stability. Apple silicon supports `bfloat` natively from M2.
    BF16,
}

impl DType {
    pub fn byte_size(&self) -> usize {
        match self {
            DType::Bool => 1,
            DType::I32 | DType::F32 => 4,
            DType::I64 | DType::F64 => 8,
            DType::F16 | DType::BF16 => 2,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "bool" => DType::Bool,
            "i32" => DType::I32,
            "i64" => DType::I64,
            "f32" => DType::F32,
            "f64" => DType::F64,
            "f16" => DType::F16,
            "bf16" => DType::BF16,
            _ => return None,
        })
    }

    /// True iff this is one of the narrow float dtypes (f16 / bf16).
    /// Used by kernels that have native handling vs those that
    /// promote-to-f32.
    pub fn is_narrow_float(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HostTensor {
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// Row-major C-order bytes. Length equals `numel() * dtype.byte_size()`.
    pub data: Vec<u8>,
}

impl HostTensor {
    pub fn new(shape: Vec<usize>, dtype: DType, data: Vec<u8>) -> Result<Self> {
        let expected = shape.iter().product::<usize>() * dtype.byte_size();
        if data.len() != expected {
            return Err(InterpretError::TypeMismatch {
                node_id: usize::MAX,
                msg: format!(
                    "tensor byte length {} != expected {} (shape={:?}, dtype={})",
                    data.len(),
                    expected,
                    shape,
                    dtype.name()
                ),
            });
        }
        Ok(Self { shape, dtype, data })
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn from_f32(shape: Vec<usize>, vals: &[f32]) -> Result<Self> {
        // Single memcpy via the byte view of `vals`. f32 →
        // little-endian bytes is a no-op on every platform we run on
        // (x86_64, aarch64), so this matches the semantics of the
        // previous per-element `to_le_bytes()` loop while avoiding
        // hundreds of µs on KV-cache-sized inputs (seq*head_dim
        // ≥ 100K). Hot-path-load-bearing for `rce-op-attention` and
        // any other leaf that converts host f32 buffers.
        let bytes: &[u8] = bytemuck::cast_slice(vals);
        Self::new(shape, DType::F32, bytes.to_vec())
    }

    pub fn from_f64(shape: Vec<usize>, vals: &[f64]) -> Result<Self> {
        let bytes: &[u8] = bytemuck::cast_slice(vals);
        Self::new(shape, DType::F64, bytes.to_vec())
    }

    pub fn from_i64(shape: Vec<usize>, vals: &[i64]) -> Result<Self> {
        let bytes: &[u8] = bytemuck::cast_slice(vals);
        Self::new(shape, DType::I64, bytes.to_vec())
    }

    pub fn from_i32(shape: Vec<usize>, vals: &[i32]) -> Result<Self> {
        let bytes: &[u8] = bytemuck::cast_slice(vals);
        Self::new(shape, DType::I32, bytes.to_vec())
    }

    /// Build an `f16` tensor from a slice of f32 values, narrowing on
    /// the way in. Phase O of the codegen rollout — lets callers
    /// produce f16-byte tensors that the GPU narrow-dispatch path
    /// uploads without re-conversion.
    pub fn from_f32_as_f16(shape: Vec<usize>, vals: &[f32]) -> Result<Self> {
        let mut bytes = Vec::with_capacity(vals.len() * 2);
        for &x in vals {
            bytes.extend_from_slice(&half::f16::from_f32(x).to_le_bytes());
        }
        Self::new(shape, DType::F16, bytes)
    }

    /// Same, but bf16.
    pub fn from_f32_as_bf16(shape: Vec<usize>, vals: &[f32]) -> Result<Self> {
        let mut bytes = Vec::with_capacity(vals.len() * 2);
        for &x in vals {
            bytes.extend_from_slice(&half::bf16::from_f32(x).to_le_bytes());
        }
        Self::new(shape, DType::BF16, bytes)
    }

    /// Construct an f16 tensor from raw 2-byte little-endian f16 byte
    /// data, bypassing any conversion. Used by the GPU dispatchers
    /// when copying f16-produced output back into a HostTensor.
    pub fn from_f16_bytes(shape: Vec<usize>, bytes: Vec<u8>) -> Result<Self> {
        Self::new(shape, DType::F16, bytes)
    }

    /// Same for bf16.
    pub fn from_bf16_bytes(shape: Vec<usize>, bytes: Vec<u8>) -> Result<Self> {
        Self::new(shape, DType::BF16, bytes)
    }

    /// Raw f16 byte view. Returns `None` if the tensor isn't f16. The
    /// `&[u8]` is a borrow of the tensor's storage — caller may
    /// memcpy from it directly (the bandwidth fast-path the codegen
    /// dispatchers use).
    pub fn as_f16_bytes(&self) -> Option<&[u8]> {
        if self.dtype == DType::F16 { Some(&self.data) } else { None }
    }

    /// Same for bf16.
    pub fn as_bf16_bytes(&self) -> Option<&[u8]> {
        if self.dtype == DType::BF16 { Some(&self.data) } else { None }
    }

    /// Element accessors. Caller is responsible for matching dtype.
    pub fn as_f32(&self) -> Result<Vec<f32>> {
        if self.dtype != DType::F32 {
            return Err(self.dtype_err("f32"));
        }
        Ok(self
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn as_f64(&self) -> Result<Vec<f64>> {
        if self.dtype != DType::F64 {
            return Err(self.dtype_err("f64"));
        }
        Ok(self
            .data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    }

    pub fn as_i32(&self) -> Result<Vec<i32>> {
        if self.dtype != DType::I32 {
            return Err(self.dtype_err("i32"));
        }
        Ok(self
            .data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn as_i64(&self) -> Result<Vec<i64>> {
        if self.dtype != DType::I64 {
            return Err(self.dtype_err("i64"));
        }
        Ok(self
            .data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    }

    pub fn as_bool(&self) -> Result<Vec<bool>> {
        if self.dtype != DType::Bool {
            return Err(self.dtype_err("bool"));
        }
        Ok(self.data.iter().map(|&b| b != 0).collect())
    }

    fn dtype_err(&self, want: &str) -> InterpretError {
        InterpretError::TypeMismatch {
            node_id: usize::MAX,
            msg: format!(
                "tensor dtype {} cannot be read as {}",
                self.dtype.name(),
                want
            ),
        }
    }

    /// Promote a scalar to a 0-D tensor of the dtype best matching the
    /// scalar's runtime type.
    pub fn from_scalar(s: &ScalarValue) -> Result<Self> {
        match s {
            ScalarValue::F32(v) => Self::from_f32(vec![], &[*v]),
            ScalarValue::F64(v) => Self::from_f64(vec![], &[*v]),
            ScalarValue::I32(v) => Self::from_i32(vec![], &[*v]),
            ScalarValue::I64(v) => Self::from_i64(vec![], &[*v]),
            ScalarValue::Bool(v) => Self::new(vec![], DType::Bool, vec![*v as u8]),
            ScalarValue::String(_) => Err(InterpretError::TypeMismatch {
                node_id: usize::MAX,
                msg: "cannot promote String scalar to tensor".into(),
            }),
        }
    }

    /// Convert a single-element tensor back to a scalar.
    pub fn try_to_scalar(&self) -> Option<ScalarValue> {
        if self.numel() != 1 {
            return None;
        }
        Some(match self.dtype {
            DType::F32 => ScalarValue::F32(self.as_f32().ok()?[0]),
            DType::F64 => ScalarValue::F64(self.as_f64().ok()?[0]),
            // F16 / BF16 widen to F32 at the scalar boundary — there's
            // no ScalarValue::F16 today, and the scalar API is f32-rooted.
            DType::F16 => {
                let c = self.data.get(0..2)?;
                ScalarValue::F32(half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            }
            DType::BF16 => {
                let c = self.data.get(0..2)?;
                ScalarValue::F32(half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            }
            DType::I32 => ScalarValue::I32(self.as_i32().ok()?[0]),
            DType::I64 => ScalarValue::I64(self.as_i64().ok()?[0]),
            DType::Bool => ScalarValue::Bool(self.as_bool().ok()?[0]),
        })
    }
}

impl std::fmt::Display for HostTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(shape={:?}, dtype={}, ", self.shape, self.dtype.name())?;
        match self.dtype {
            DType::F32 => write!(f, "data={:?})", self.as_f32().unwrap_or_default()),
            DType::F64 => write!(f, "data={:?})", self.as_f64().unwrap_or_default()),
            // F16 / BF16: widen to f32 for display.
            DType::F16 | DType::BF16 => write!(f, "data={:?})", self.as_f32_promoted().unwrap_or_default()),
            DType::I32 => write!(f, "data={:?})", self.as_i32().unwrap_or_default()),
            DType::I64 => write!(f, "data={:?})", self.as_i64().unwrap_or_default()),
            DType::Bool => write!(f, "data={:?})", self.as_bool().unwrap_or_default()),
        }
    }
}
