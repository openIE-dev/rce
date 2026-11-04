use crate::device::DeviceTensor;
use crate::error::{InterpretError, Result};
use crate::tensor::HostTensor;
use serde::{Deserialize, Serialize};

/// Runtime value the interpreter operates on. `Scalar` is a host-resident
/// primitive; `Tensor` is a host-resident shaped buffer; `DeviceTensor`
/// is a shaped buffer living on an accelerator (GPU/NPU/…). Wires carry
/// any of the three uniformly so a graph can flow values between
/// host-only ops and device-only ops without the interpreter having to
/// know which is which.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Scalar(ScalarValue),
    Tensor(HostTensor),
    DeviceTensor(DeviceTensor),
}

impl Value {
    pub fn type_name(&self) -> String {
        match self {
            Value::Scalar(s) => s.type_name().to_string(),
            Value::Tensor(t) => format!("tensor<{}, shape={:?}>", t.dtype.name(), t.shape),
            Value::DeviceTensor(d) => format!(
                "device-tensor<{} on {}, shape={:?}>",
                d.dtype.name(),
                d.device.name(),
                d.shape
            ),
        }
    }

    pub fn as_scalar(&self, node_id: usize) -> Result<&ScalarValue> {
        match self {
            Value::Scalar(s) => Ok(s),
            other => Err(InterpretError::TypeMismatch {
                node_id,
                msg: format!("expected scalar, got {}", other.type_name()),
            }),
        }
    }

    /// True iff the value is a tensor of any residency (host or device).
    pub fn is_tensor(&self) -> bool {
        matches!(self, Value::Tensor(_) | Value::DeviceTensor(_))
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Scalar(s) => write!(f, "{s}"),
            Value::Tensor(t) => write!(f, "{t}"),
            Value::DeviceTensor(d) => write!(f, "{d:?}"),
        }
    }
}

impl From<ScalarValue> for Value {
    fn from(s: ScalarValue) -> Self {
        Value::Scalar(s)
    }
}

impl From<HostTensor> for Value {
    fn from(t: HostTensor) -> Self {
        Value::Tensor(t)
    }
}

impl From<DeviceTensor> for Value {
    fn from(d: DeviceTensor) -> Self {
        Value::DeviceTensor(d)
    }
}

/// Scalar runtime value. Strings are scalars in v0.1; tensors live in
/// `Value::Tensor`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScalarValue {
    Bool(bool),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    String(String),
}

impl ScalarValue {
    pub fn type_name(&self) -> &'static str {
        match self {
            ScalarValue::Bool(_) => "bool",
            ScalarValue::I32(_) => "i32",
            ScalarValue::I64(_) => "i64",
            ScalarValue::F32(_) => "f32",
            ScalarValue::F64(_) => "f64",
            ScalarValue::String(_) => "string",
        }
    }

    pub fn as_bool(&self, node_id: usize) -> Result<bool> {
        match self {
            ScalarValue::Bool(b) => Ok(*b),
            other => Err(InterpretError::TypeMismatch {
                node_id,
                msg: format!("expected bool, got {}", other.type_name()),
            }),
        }
    }
}

impl std::fmt::Display for ScalarValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarValue::Bool(b) => write!(f, "{b}"),
            ScalarValue::I32(v) => write!(f, "{v}"),
            ScalarValue::I64(v) => write!(f, "{v}"),
            ScalarValue::F32(v) => write!(f, "{v}"),
            ScalarValue::F64(v) => write!(f, "{v}"),
            ScalarValue::String(s) => write!(f, "{s:?}"),
        }
    }
}
