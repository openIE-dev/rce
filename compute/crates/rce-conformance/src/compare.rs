//! Output comparison across backends.
//!
//! Two `Value`s agree if:
//! - their structural shape matches (Scalar vs Tensor, dtype, shape)
//! - integer / boolean components are bit-equal
//! - floating-point components are within tolerance (relative or
//!   absolute, configurable per case); NaN / ±Inf must match exactly
//!
//! Tolerance is intentionally permissive at default — bit-equality on
//! float reductions across architectures (CPU SIMD vs GPU shader vs
//! Wasm scalar) is unrealistic. Strict bit-equality is opt-in via
//! `ToleranceMode::Bitwise` for cases where determinism is contractual.

use rce_vm_backend_cpu::tensor::DType;
use rce_vm_backend_cpu::{ScalarValue, Value};

#[derive(Debug, Clone, Copy)]
pub enum ToleranceMode {
    /// Bit-equality required. The default for integer/bool tensors and
    /// for cases marked `Determinism::Strict`.
    Bitwise,
    /// Relative tolerance for floats: `|got - expected| / max(|got|, |expected|, eps)`
    /// must be within `rel`. Absolute floor `abs` for values near zero.
    Relative { rel: f64, abs: f64 },
}

impl Default for ToleranceMode {
    fn default() -> Self {
        // Defaults calibrated for f32: relative 1e-4 with absolute floor
        // 1e-5. f64 cases override to 1e-12 / 1e-15.
        ToleranceMode::Relative { rel: 1e-4, abs: 1e-5 }
    }
}

/// Compare two outputs. Returns `Ok(())` on agreement, `Err(diff)` on
/// divergence; the error string names the first mismatching element.
pub fn compare(a: &Value, b: &Value, tol: ToleranceMode) -> Result<(), String> {
    match (a, b) {
        (Value::Scalar(s1), Value::Scalar(s2)) => compare_scalar(s1, s2, tol),
        (Value::Tensor(t1), Value::Tensor(t2)) => {
            if t1.shape != t2.shape {
                return Err(format!("shape mismatch: {:?} vs {:?}", t1.shape, t2.shape));
            }
            if t1.dtype != t2.dtype {
                return Err(format!(
                    "dtype mismatch: {} vs {}",
                    t1.dtype.name(),
                    t2.dtype.name()
                ));
            }
            compare_tensor(t1, t2, tol)
        }
        (a, b) => Err(format!(
            "structural mismatch: {} vs {}",
            a.type_name(),
            b.type_name()
        )),
    }
}

fn compare_scalar(a: &ScalarValue, b: &ScalarValue, tol: ToleranceMode) -> Result<(), String> {
    match (a, b) {
        (ScalarValue::Bool(x), ScalarValue::Bool(y)) if x == y => Ok(()),
        (ScalarValue::I32(x), ScalarValue::I32(y)) if x == y => Ok(()),
        (ScalarValue::I64(x), ScalarValue::I64(y)) if x == y => Ok(()),
        (ScalarValue::String(x), ScalarValue::String(y)) if x == y => Ok(()),
        (ScalarValue::F32(x), ScalarValue::F32(y)) => f64_within(*x as f64, *y as f64, tol),
        (ScalarValue::F64(x), ScalarValue::F64(y)) => f64_within(*x, *y, tol),
        // Cross-type comparisons happen when one backend produced i32
        // and another i64 from the same op; we promote and compare.
        (ScalarValue::I32(x), ScalarValue::I64(y)) | (ScalarValue::I64(y), ScalarValue::I32(x)) => {
            if (*x as i64) == *y {
                Ok(())
            } else {
                Err(format!("scalar diverge: {x} vs {y}"))
            }
        }
        (a, b) => Err(format!(
            "scalar diverge: {} ({:?}) vs {} ({:?})",
            a.type_name(),
            a,
            b.type_name(),
            b
        )),
    }
}

fn compare_tensor(
    t1: &rce_vm_backend_cpu::HostTensor,
    t2: &rce_vm_backend_cpu::HostTensor,
    tol: ToleranceMode,
) -> Result<(), String> {
    match t1.dtype {
        DType::Bool => {
            let a = t1.as_bool().map_err(|e| format!("read a: {e}"))?;
            let b = t2.as_bool().map_err(|e| format!("read b: {e}"))?;
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                if x != y {
                    return Err(format!("tensor diverge at index {i}: {x} vs {y}"));
                }
            }
            Ok(())
        }
        DType::I32 => {
            let a = t1.as_i32().map_err(|e| format!("read a: {e}"))?;
            let b = t2.as_i32().map_err(|e| format!("read b: {e}"))?;
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                if x != y {
                    return Err(format!("tensor diverge at index {i}: {x} vs {y}"));
                }
            }
            Ok(())
        }
        DType::I64 => {
            let a = t1.as_i64().map_err(|e| format!("read a: {e}"))?;
            let b = t2.as_i64().map_err(|e| format!("read b: {e}"))?;
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                if x != y {
                    return Err(format!("tensor diverge at index {i}: {x} vs {y}"));
                }
            }
            Ok(())
        }
        DType::F32 => {
            let a = t1.as_f32().map_err(|e| format!("read a: {e}"))?;
            let b = t2.as_f32().map_err(|e| format!("read b: {e}"))?;
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                f64_within(*x as f64, *y as f64, tol).map_err(|e| {
                    format!("tensor diverge at index {i}: {e} (a={x}, b={y})")
                })?;
            }
            Ok(())
        }
        DType::F64 => {
            let a = t1.as_f64().map_err(|e| format!("read a: {e}"))?;
            let b = t2.as_f64().map_err(|e| format!("read b: {e}"))?;
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                f64_within(*x, *y, tol).map_err(|e| {
                    format!("tensor diverge at index {i}: {e} (a={x}, b={y})")
                })?;
            }
            Ok(())
        }
        DType::F16 | DType::BF16 => {
            Err(format!(
                "{:?} comparison not yet implemented",
                t1.dtype
            ))
        }
    }
}

fn f64_within(a: f64, b: f64, tol: ToleranceMode) -> Result<(), String> {
    // NaN must match exactly; ±Inf must match.
    if a.is_nan() && b.is_nan() {
        return Ok(());
    }
    if a.is_nan() != b.is_nan() {
        return Err(format!("nan mismatch: {a} vs {b}"));
    }
    if a.is_infinite() || b.is_infinite() {
        if a == b {
            return Ok(());
        }
        return Err(format!("inf mismatch: {a} vs {b}"));
    }
    match tol {
        ToleranceMode::Bitwise => {
            if a.to_bits() == b.to_bits() {
                Ok(())
            } else {
                Err(format!("bitwise diff: {a:?} vs {b:?}"))
            }
        }
        ToleranceMode::Relative { rel, abs } => {
            let diff = (a - b).abs();
            let scale = a.abs().max(b.abs()).max(abs);
            if diff <= rel * scale {
                Ok(())
            } else {
                Err(format!("|Δ|={diff:.3e} > rel·scale={:.3e}", rel * scale))
            }
        }
    }
}
