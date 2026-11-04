use crate::tensor::{DType, Tensor, TensorHandle};
use smallvec::SmallVec;
use thiserror::Error;

/// Runtime value flowing on a wire between graph nodes.
///
/// Tensors are by-handle (backend-resident). Scalars are inline.
/// `Capability` is an unforgeable handle resolved by `rce-vm-cap`.
#[derive(Debug, Clone)]
pub enum Value {
    Tensor(Tensor, TensorHandle),
    I64(i64),
    I32(i32),
    F64(f64),
    F32(f32),
    Bool(bool),
    String(String),
    /// Capability handle: 16-byte unforgeable ID. Validation is done by the
    /// `rce-vm-cap::CapTable` at use-site, not on the wire.
    Capability([u8; 16]),
    /// Stream of values (used with `WireKind::Stream`/`Channel`); represented
    /// at the value level as an opaque ID the scheduler resolves.
    Stream(u64),
    /// A void / unit value.
    Unit,
}

impl Value {
    pub fn dtype(&self) -> Option<DType> {
        Some(match self {
            Value::Tensor(t, _) => t.dtype,
            Value::I64(_) => DType::I64,
            Value::I32(_) => DType::I32,
            Value::F64(_) => DType::F64,
            Value::F32(_) => DType::F32,
            Value::Bool(_) => DType::Bool,
            _ => return None,
        })
    }
}

#[derive(Debug, Error)]
pub enum ValueError {
    #[error("type mismatch: expected {expected}, got {got}")]
    TypeMismatch { expected: String, got: String },

    #[error("capability not bound")]
    UnboundCapability,
}

/// Inline scalar packing for small constants.
pub type Scalars = SmallVec<[Value; 4]>;
