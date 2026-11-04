//! ML primitive layer for the v0.1 CPU interpreter.
//!
//! Custom op dispatch table for `OpKind::Custom(<name>)` — dispatches
//! known ML ops to specialised kernels. Unknown names return
//! `InterpretError::Unsupported`.
//!
//! Naming convention (stable wire identifiers; see also docs/spec/fg-abi.md):
//!
//! | Name               | Inputs                          | Output  | Notes |
//! |--------------------|---------------------------------|---------|-------|
//! | `matmul`           | `(A, B)`                        | tensor  | 2D × 2D, or batched ≥3D |
//! | `transpose`        | `(x, axes_json)`                | tensor  | axes_json: Literal::String("[0,2,1]") |
//! | `reshape`          | `(x, shape_json)`               | tensor  | shape_json: Literal::String("[1,3,4]") |
//! | `tensor_concat`    | `(a, b, axis_i32)`              | tensor  | axis: Literal::I32 |
//! | `reduce_sum`       | `(x, axes_json, keepdim)`       | tensor  | keepdim: Literal::Bool |
//! | `reduce_mean`      | `(x, axes_json, keepdim)`       | tensor  | |
//! | `reduce_max`       | `(x, axes_json, keepdim)`       | tensor  | |
//! | `reduce_min`       | `(x, axes_json, keepdim)`       | tensor  | |
//! | `relu`             | `(x,)`                          | tensor  | f32/f64 |
//! | `exp`, `tanh`      | `(x,)`                          | tensor  | f32/f64; RCE's correctly rounded functions (`nn`) |
//! | `sigmoid`, `silu`  | `(x,)`                          | tensor  | f32/f64 (`nn`) |
//! | `gelu`, `gelu_erf` | `(x,)`                          | tensor  | f32/f64; the tanh form and the erf form (`nn`) |
//! | `softmax`          | `(x, axis_i32)`                 | tensor  | f32/f64 (`nn`) |
//! | `rms_norm`         | `(x, w, ngroups_i32, eps)`      | tensor  | f32/f64 (`nn`) |
//! | `layer_norm`       | `(x, w, b, ngroups_i32, eps)`   | tensor  | f32/f64 (`nn`) |
//! | `attention`        | `(q, k, v, scale, window_i32[, causal_bool])` | tensor | f32/f64; one head or heads first (`nn`) |
//! | `embedding_lookup` | `(table, ids_i32)`              | tensor  | (`nn`) |
//! | `matmul_q8_0`, `matmul_q4_0` | `(x, q, d)`           | tensor  | f32 x, i8 q in i32 lanes, f32 block scales (`nn`) |
//! | `round_q8_0`, `round_q4_0` | `(x,)`                  | tensor  | f32; lowers precision (`nn`) |
//! | `scalar_mul`       | `(x, scalar)`                   | tensor  | tensor scaled by scalar literal |
//! | `round_bf16`       | `(x,)`                          | tensor  | f32/f64: each element rounded to the nearest bfloat16, ties to even, kept in its type |

pub mod activation;
pub mod broadcast;
pub mod matmul;
pub mod nn;
pub mod reduce;
pub mod shape;

use crate::error::{InterpretError, Result};
use crate::tensor::HostTensor;
use crate::value::{ScalarValue, Value};

// Stable Custom-op names.
pub const OP_MATMUL: &str = "matmul";
pub const OP_TRANSPOSE: &str = "transpose";
pub const OP_RESHAPE: &str = "reshape";
pub const OP_TENSOR_CONCAT: &str = "tensor_concat";
pub const OP_REDUCE_SUM: &str = "reduce_sum";
pub const OP_REDUCE_MEAN: &str = "reduce_mean";
pub const OP_REDUCE_MAX: &str = "reduce_max";
pub const OP_REDUCE_MIN: &str = "reduce_min";
pub const OP_SOFTMAX: &str = "softmax";
pub const OP_RELU: &str = "relu";
pub const OP_EXP: &str = "exp";
/// The tanh form of GELU.
pub const OP_GELU: &str = "gelu";
/// The erf form of GELU.
pub const OP_GELU_ERF: &str = "gelu_erf";
pub const OP_SIGMOID: &str = "sigmoid";
pub const OP_SILU: &str = "silu";
pub const OP_TANH: &str = "tanh";
pub const OP_RMS_NORM: &str = "rms_norm";
pub const OP_LAYER_NORM: &str = "layer_norm";
pub const OP_ATTENTION: &str = "attention";
pub const OP_EMBEDDING_LOOKUP: &str = "embedding_lookup";
pub const OP_MATMUL_Q8_0: &str = "matmul_q8_0";
pub const OP_MATMUL_Q4_0: &str = "matmul_q4_0";
pub const OP_ROUND_Q8_0: &str = "round_q8_0";
pub const OP_ROUND_Q4_0: &str = "round_q4_0";
/// Fused: `softmax((a @ b) * scale, axis=-1)`. CPU path is composition;
/// wgpu has a single-kernel fused implementation.
pub const OP_MATMUL_SCALE_SOFTMAX: &str = "matmul_scale_softmax";
/// PyTorch `F.linear(input, weight, bias?)` semantics:
/// `input @ weight.T + bias`. Weight is `(out_features, in_features)`.
pub const OP_LINEAR: &str = "linear";
/// Each element rounded to the nearest bfloat16 value (ties to even) and
/// widened back to its type, which is exact. It lowers precision, so the
/// checker requires its level `lossy` to be declared (spec 6.13).
pub const OP_ROUND_BF16: &str = "round_bf16";

/// Dispatch a `Custom(name)` op. Returns `InterpretError::Unsupported`
/// if `name` is not in the table.
pub fn dispatch_custom(node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
    match name {
        OP_MATMUL => matmul::matmul(node_id, args),
        OP_TRANSPOSE => shape::transpose(node_id, args),
        OP_RESHAPE => shape::reshape(node_id, args),
        OP_TENSOR_CONCAT => shape::concat(node_id, args),
        OP_REDUCE_SUM => reduce::reduce(node_id, args, reduce::Op::Sum),
        OP_REDUCE_MEAN => reduce::reduce(node_id, args, reduce::Op::Mean),
        OP_REDUCE_MAX => reduce::reduce(node_id, args, reduce::Op::Max),
        OP_REDUCE_MIN => reduce::reduce(node_id, args, reduce::Op::Min),
        OP_SOFTMAX => nn::softmax(node_id, args),
        OP_RELU => activation::relu(node_id, args),
        OP_EXP | OP_TANH | OP_SIGMOID | OP_SILU | OP_GELU | OP_GELU_ERF => nn::unary(node_id, name, args),
        OP_RMS_NORM => nn::rms_norm(node_id, args),
        OP_LAYER_NORM => nn::layer_norm(node_id, args),
        OP_ATTENTION => nn::attention(node_id, args),
        OP_EMBEDDING_LOOKUP => nn::embedding_lookup(node_id, args),
        OP_MATMUL_Q8_0 => nn::matmul_q(node_id, args, false),
        OP_MATMUL_Q4_0 => nn::matmul_q(node_id, args, true),
        OP_ROUND_Q8_0 => nn::round_q(node_id, args, false),
        OP_ROUND_Q4_0 => nn::round_q(node_id, args, true),
        OP_MATMUL_SCALE_SOFTMAX => matmul_scale_softmax(node_id, args),
        OP_LINEAR => linear(node_id, args),
        OP_ROUND_BF16 => round_bf16(node_id, args),
        other => Err(InterpretError::Unsupported(format!(
            "OpKind::Custom({other})"
        ))),
    }
}

// ── Shared helpers ──────────────────────────────────────────────────

/// CPU implementation of `matmul_scale_softmax`. Composition: matmul → mul
/// (scalar broadcast) → softmax(-1). The fused kernel in wgpu makes this
/// one GPU dispatch; here we run the three primitives sequentially.
pub fn matmul_scale_softmax(node_id: usize, args: &[Value]) -> Result<Value> {
    if args.len() != 3 {
        return Err(InterpretError::ArityMismatch {
            node_id,
            kind: "matmul_scale_softmax".into(),
            expected: 3,
            got: args.len(),
        });
    }
    // matmul(a, b)
    let mm = matmul::matmul(node_id, &args[..2])?;
    // (matmul) * scale
    let scaled = {
        use crate::value::Value;
        let scale = match &args[2] {
            Value::Scalar(s) => s.clone(),
            other => {
                return Err(InterpretError::TypeMismatch {
                    node_id,
                    msg: format!(
                        "matmul_scale_softmax scale: expected scalar, got {}",
                        other.type_name()
                    ),
                });
            }
        };
        // Re-use the kernel binary by going through scalar promotion + bcast.
        let a = arg_tensor(node_id, &mm)?;
        let b = HostTensor::from_scalar(&scale)?;
        let (a2, b2, _) = crate::ml::broadcast::align_pair(node_id, &a, &b)?;
        Value::Tensor(crate::kernel::binary(
            node_id,
            &rce_graph::graph::OpKind::Mul,
            &a2,
            &b2,
        )?)
    };
    // softmax(scaled, axis=-1)
    let axis = Value::Scalar(crate::value::ScalarValue::I32(-1));
    nn::softmax(node_id, &[scaled, axis])
}

/// `round_bf16(x)`: each element rounded once to the nearest bfloat16 value
/// (ties to even) and widened back to its type, which is exact.
pub fn round_bf16(node_id: usize, args: &[Value]) -> Result<Value> {
    if args.len() != 1 {
        return Err(InterpretError::ArityMismatch {
            node_id,
            kind: OP_ROUND_BF16.into(),
            expected: 1,
            got: args.len(),
        });
    }
    let x = arg_tensor(node_id, &args[0])?;
    match x.dtype {
        crate::tensor::DType::F32 => {
            let v: Vec<f32> = x
                .as_f32()?
                .into_iter()
                .map(|e| round_to_bf16(e as f64) as f32)
                .collect();
            HostTensor::from_f32(x.shape.clone(), &v).map(Value::Tensor)
        }
        // one rounding, from the f64 value straight to bfloat16
        crate::tensor::DType::F64 => {
            let v: Vec<f64> = x.as_f64()?.into_iter().map(round_to_bf16).collect();
            HostTensor::from_f64(x.shape.clone(), &v).map(Value::Tensor)
        }
        other => Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!("round_bf16: dtype {} (f32 or f64 only)", other.name()),
        }),
    }
}

/// `x` rounded once to the nearest bfloat16 value, ties to even, as an f64
/// (every bfloat16 value is one). bfloat16 has 8 significant bits and the
/// exponent range of binary32: the spacing of its values at `x` is
/// q = 2^(e - 7) for 2^e <= |x| < 2^(e + 1) and e >= -126, and 2^-133 below
/// 2^-126. `x / q` and `n * q` are exact (scalings by a power of two that
/// stay inside binary64's range), so the one rounding is that of `x / q` to
/// an integer n. A result of 2^128 or more has no bfloat16 value; it is
/// returned as an infinity, which the reference reports as a non-finite
/// float. (`half::bf16::from_f64` drops the low 32 bits of the mantissa
/// before it rounds, so it rounds a value just above a tie down.)
pub fn round_to_bf16(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let bits = x.abs().to_bits();
    let e = ((bits >> 52) & 0x7ff) as i32 - 1023; // f64 subnormals give -1023
    let q_exp = if e >= -126 { e - 7 } else { -133 };
    let pow2 = |k: i32| f64::from_bits(((k + 1023) as u64) << 52);
    let q = pow2(q_exp);
    let n = (x / q).round_ties_even();
    let r = n * q;
    if r.abs() >= pow2(128) {
        return if r > 0.0 { f64::INFINITY } else { f64::NEG_INFINITY };
    }
    r
}

/// PyTorch `F.linear(input, weight, bias?)` — `input @ weight.T + bias`.
/// Weight shape: `(out_features, in_features)`. v0.1 supports 2D weights;
/// rank-N weights would need a generalised transpose, deferred to v0.2.
pub fn linear(node_id: usize, args: &[Value]) -> Result<Value> {
    if args.len() != 2 && args.len() != 3 {
        return Err(InterpretError::ArityMismatch {
            node_id,
            kind: "linear".into(),
            expected: 3,
            got: args.len(),
        });
    }
    let input = arg_tensor(node_id, &args[0])?;
    let weight = arg_tensor(node_id, &args[1])?;
    if weight.shape.len() != 2 {
        return Err(InterpretError::Unsupported(format!(
            "linear: weight must be rank-2 (got {:?})",
            weight.shape
        )));
    }
    if input.dtype != weight.dtype {
        return Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!(
                "linear dtype mismatch: input={}, weight={}",
                input.dtype.name(),
                weight.dtype.name()
            ),
        });
    }
    // Transpose weight: (out, in) → (in, out) by reshape + 2D transpose.
    let weight_t = transpose_2d(&weight)?;
    let mm = matmul::matmul(
        node_id,
        &[Value::Tensor(input), Value::Tensor(weight_t)],
    )?;
    match args.len() {
        2 => Ok(mm),
        3 => {
            let bias = arg_tensor(node_id, &args[2])?;
            let mm_t = arg_tensor(node_id, &mm)?;
            let (a, b, _) = crate::ml::broadcast::align_pair(node_id, &mm_t, &bias)?;
            Ok(Value::Tensor(crate::kernel::binary(
                node_id,
                &rce_graph::graph::OpKind::Add,
                &a,
                &b,
            )?))
        }
        _ => unreachable!(),
    }
}

fn transpose_2d(t: &HostTensor) -> Result<HostTensor> {
    if t.shape.len() != 2 {
        return Err(InterpretError::Unsupported(format!(
            "transpose_2d: expected rank-2, got {:?}",
            t.shape
        )));
    }
    let rows = t.shape[0];
    let cols = t.shape[1];
    let elem = t.dtype.byte_size();
    let mut out = vec![0u8; rows * cols * elem];
    for i in 0..rows {
        for j in 0..cols {
            let src = (i * cols + j) * elem;
            let dst = (j * rows + i) * elem;
            out[dst..dst + elem].copy_from_slice(&t.data[src..src + elem]);
        }
    }
    HostTensor::new(vec![cols, rows], t.dtype, out)
}

pub(crate) fn arg_tensor(_node_id: usize, v: &Value) -> Result<HostTensor> {
    match v {
        Value::Tensor(t) => Ok(t.clone()),
        Value::Scalar(s) => HostTensor::from_scalar(s),
        Value::DeviceTensor(d) => Err(InterpretError::Unsupported(format!(
            "ml primitive on cpu requires HostTensor, got DeviceTensor on {}; \
             a leaf must claim this op or the value must be materialised first.",
            d.device.name()
        ))),
    }
}

pub(crate) fn arg_string(node_id: usize, v: &Value) -> Result<String> {
    match v {
        Value::Scalar(ScalarValue::String(s)) => Ok(s.clone()),
        other => Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!("expected string-typed metadata arg, got {}", other.type_name()),
        }),
    }
}

pub(crate) fn arg_i32(node_id: usize, v: &Value) -> Result<i32> {
    match v {
        Value::Scalar(ScalarValue::I32(x)) => Ok(*x),
        Value::Scalar(ScalarValue::I64(x)) => Ok(*x as i32),
        other => Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!("expected i32-typed arg, got {}", other.type_name()),
        }),
    }
}

pub(crate) fn arg_bool(node_id: usize, v: &Value) -> Result<bool> {
    match v {
        Value::Scalar(ScalarValue::Bool(b)) => Ok(*b),
        other => Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!("expected bool-typed arg, got {}", other.type_name()),
        }),
    }
}

pub(crate) fn parse_axes(node_id: usize, s: &str, rank: usize) -> Result<Vec<usize>> {
    let v: serde_json::Value = serde_json::from_str(s).map_err(|e| InterpretError::TypeMismatch {
        node_id,
        msg: format!("axes_json `{s}` not valid JSON: {e}"),
    })?;
    let arr = v.as_array().ok_or_else(|| InterpretError::TypeMismatch {
        node_id,
        msg: format!("axes_json must be a JSON array, got `{s}`"),
    })?;
    let mut out = Vec::with_capacity(arr.len());
    for x in arr {
        let i = x.as_i64().ok_or_else(|| InterpretError::TypeMismatch {
            node_id,
            msg: format!("axes_json element `{x}` is not an integer"),
        })?;
        let normalized = if i < 0 { (rank as i64 + i) as usize } else { i as usize };
        if normalized >= rank {
            return Err(InterpretError::TypeMismatch {
                node_id,
                msg: format!("axis {i} out of range for rank {rank}"),
            });
        }
        out.push(normalized);
    }
    Ok(out)
}

pub(crate) fn parse_shape(node_id: usize, s: &str) -> Result<Vec<usize>> {
    let v: serde_json::Value = serde_json::from_str(s).map_err(|e| InterpretError::TypeMismatch {
        node_id,
        msg: format!("shape_json `{s}` not valid JSON: {e}"),
    })?;
    let arr = v.as_array().ok_or_else(|| InterpretError::TypeMismatch {
        node_id,
        msg: format!("shape_json must be a JSON array, got `{s}`"),
    })?;
    let mut out = Vec::with_capacity(arr.len());
    for x in arr {
        let i = x.as_i64().ok_or_else(|| InterpretError::TypeMismatch {
            node_id,
            msg: format!("shape_json element `{x}` is not an integer"),
        })?;
        if i < 0 {
            return Err(InterpretError::TypeMismatch {
                node_id,
                msg: "negative dims not yet supported (no -1 inference in v0.1)".into(),
            });
        }
        out.push(i as usize);
    }
    Ok(out)
}

#[cfg(test)]
mod round_bf16_tests {
    use super::round_to_bf16;

    #[test]
    fn exact_values_stay() {
        for x in [1.0, -2.5, 0.0078125, 3.0e38_f32 as f64, -0.0] {
            let b = round_to_bf16(x);
            assert_eq!(round_to_bf16(b).to_bits(), b.to_bits());
        }
        assert_eq!(round_to_bf16(-0.0).to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn ties_go_to_even_and_a_low_bit_breaks_the_tie() {
        // 1 + 2^-8 lies halfway between 1 and 1 + 2^-7: to even, 1
        assert_eq!(round_to_bf16(1.0 + 2f64.powi(-8)), 1.0);
        // 1 + 3 * 2^-8 lies halfway between 1 + 2^-7 and 1 + 2^-6: to even, 1 + 2^-6
        assert_eq!(round_to_bf16(1.0 + 3.0 * 2f64.powi(-8)), 1.0 + 2f64.powi(-6));
        // a bit 2^-40 above the tie, below the 32 bits a truncating
        // conversion keeps, rounds up
        assert_eq!(round_to_bf16(1.0 + 2f64.powi(-8) + 2f64.powi(-40)), 1.0 + 2f64.powi(-7));
    }

    #[test]
    fn subnormals_and_overflow() {
        // the least bfloat16 subnormal is 2^-133; half of it ties to 0
        assert_eq!(round_to_bf16(2f64.powi(-133)), 2f64.powi(-133));
        assert_eq!(round_to_bf16(2f64.powi(-134)), 0.0);
        assert_eq!(round_to_bf16(2f64.powi(-134) * 1.5), 2f64.powi(-133));
        // the largest value is (2 - 2^-7) 2^127; from 2^128 - 2^119 up it overflows
        let max = (2.0 - 2f64.powi(-7)) * 2f64.powi(127);
        assert_eq!(round_to_bf16(max), max);
        assert!(round_to_bf16(2f64.powi(128) - 2f64.powi(119)).is_infinite());
        assert_eq!(round_to_bf16(2f64.powi(128) - 2f64.powi(119) - 2f64.powi(100)), max);
    }
}
