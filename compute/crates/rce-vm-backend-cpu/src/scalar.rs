//! Scalar op dispatch for the v0.1 CPU interpreter.
//!
//! Implements the full non-Custom `OpKind` set over `ScalarValue`. Each
//! op picks a numeric domain by promoting both operands to a common type,
//! the rule being: floats win over ints, wider wins over narrower, and
//! mixed-kind comparisons are an error.

use crate::error::{InterpretError, Result};
use crate::value::ScalarValue;
use rce_graph::graph::OpKind;

/// Promote two operands to a common numeric type. Returns `None` for
/// boolean/string operands, which require their own dispatch path.
fn promote(a: &ScalarValue, b: &ScalarValue) -> Option<(NumPair, NumPair)> {
    use ScalarValue::*;
    let (na, nb) = match (a, b) {
        (F64(_), _) | (_, F64(_)) => (NumPair::F64(to_f64(a)?), NumPair::F64(to_f64(b)?)),
        (F32(_), _) | (_, F32(_)) => (NumPair::F32(to_f32(a)?), NumPair::F32(to_f32(b)?)),
        (I64(_), _) | (_, I64(_)) => (NumPair::I64(to_i64(a)?), NumPair::I64(to_i64(b)?)),
        (I32(_), I32(_)) => (NumPair::I32(to_i32(a)?), NumPair::I32(to_i32(b)?)),
        _ => return None,
    };
    Some((na, nb))
}

#[derive(Debug, Clone, Copy)]
enum NumPair {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

fn to_f64(v: &ScalarValue) -> Option<f64> {
    Some(match v {
        ScalarValue::F64(x) => *x,
        ScalarValue::F32(x) => *x as f64,
        ScalarValue::I64(x) => *x as f64,
        ScalarValue::I32(x) => *x as f64,
        _ => return None,
    })
}

fn to_f32(v: &ScalarValue) -> Option<f32> {
    Some(match v {
        ScalarValue::F32(x) => *x,
        ScalarValue::I32(x) => *x as f32,
        ScalarValue::I64(x) => *x as f32,
        ScalarValue::F64(x) => *x as f32,
        _ => return None,
    })
}

fn to_i64(v: &ScalarValue) -> Option<i64> {
    Some(match v {
        ScalarValue::I64(x) => *x,
        ScalarValue::I32(x) => *x as i64,
        _ => return None,
    })
}

fn to_i32(v: &ScalarValue) -> Option<i32> {
    Some(match v {
        ScalarValue::I32(x) => *x,
        _ => return None,
    })
}

pub fn apply_op(node_id: usize, op: &OpKind, args: &[ScalarValue]) -> Result<ScalarValue> {
    use OpKind::*;
    match op {
        // Unary numeric.
        Neg => {
            let [v] = arity::<1>(node_id, op, args)?;
            match v {
                ScalarValue::I32(x) => Ok(ScalarValue::I32(-x)),
                ScalarValue::I64(x) => Ok(ScalarValue::I64(-x)),
                ScalarValue::F32(x) => Ok(ScalarValue::F32(-x)),
                ScalarValue::F64(x) => Ok(ScalarValue::F64(-x)),
                other => Err(InterpretError::TypeMismatch {
                    node_id,
                    msg: format!("Neg: expected numeric, got {}", other.type_name()),
                }),
            }
        }
        // Unary bool.
        Not => {
            let [v] = arity::<1>(node_id, op, args)?;
            Ok(ScalarValue::Bool(!v.as_bool(node_id)?))
        }
        // Binary arithmetic.
        Add | Sub | Mul | Div | Rem => {
            let [a, b] = arity::<2>(node_id, op, args)?;
            let pair = promote(&a, &b).ok_or_else(|| InterpretError::TypeMismatch {
                node_id,
                msg: format!(
                    "{op:?}: arguments not numerically promotable ({}, {})",
                    a.type_name(),
                    b.type_name()
                ),
            })?;
            apply_arith(node_id, op, pair)
        }
        // Comparison.
        Eq | Ne | Lt | Le | Gt | Ge => {
            let [a, b] = arity::<2>(node_id, op, args)?;
            apply_cmp(node_id, op, &a, &b)
        }
        // Boolean.
        And | Or => {
            let [a, b] = arity::<2>(node_id, op, args)?;
            let lhs = a.as_bool(node_id)?;
            let rhs = b.as_bool(node_id)?;
            Ok(ScalarValue::Bool(match op {
                And => lhs && rhs,
                Or => lhs || rhs,
                _ => unreachable!(),
            }))
        }
        // String concat.
        Concat => {
            let [a, b] = arity::<2>(node_id, op, args)?;
            match (a, b) {
                (ScalarValue::String(x), ScalarValue::String(y)) => {
                    Ok(ScalarValue::String(format!("{x}{y}")))
                }
                (a, b) => Err(InterpretError::TypeMismatch {
                    node_id,
                    msg: format!(
                        "Concat: expected (string, string), got ({}, {})",
                        a.type_name(),
                        b.type_name()
                    ),
                }),
            }
        }
        Custom(name) => Err(InterpretError::Unsupported(format!(
            "OpKind::Custom({name})"
        ))),
    }
}

fn arity<const N: usize>(
    node_id: usize,
    op: &OpKind,
    args: &[ScalarValue],
) -> Result<[ScalarValue; N]> {
    if args.len() != N {
        return Err(InterpretError::ArityMismatch {
            node_id,
            kind: format!("{op:?}"),
            expected: N,
            got: args.len(),
        });
    }
    let arr: [ScalarValue; N] = args
        .iter()
        .cloned()
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| InterpretError::Internal("arity ToArray".into()))?;
    Ok(arr)
}

fn apply_arith(node_id: usize, op: &OpKind, pair: (NumPair, NumPair)) -> Result<ScalarValue> {
    use OpKind::*;
    macro_rules! arith {
        ($a:expr, $b:expr, $kind:ident, $zero:expr) => {
            match op {
                Add => Ok(ScalarValue::$kind($a + $b)),
                Sub => Ok(ScalarValue::$kind($a - $b)),
                Mul => Ok(ScalarValue::$kind($a * $b)),
                Div => {
                    if $b == $zero {
                        Err(InterpretError::DivByZero(node_id))
                    } else {
                        Ok(ScalarValue::$kind($a / $b))
                    }
                }
                Rem => {
                    if $b == $zero {
                        Err(InterpretError::DivByZero(node_id))
                    } else {
                        Ok(ScalarValue::$kind($a % $b))
                    }
                }
                _ => unreachable!(),
            }
        };
    }
    match pair {
        (NumPair::I32(a), NumPair::I32(b)) => arith!(a, b, I32, 0_i32),
        (NumPair::I64(a), NumPair::I64(b)) => arith!(a, b, I64, 0_i64),
        (NumPair::F32(a), NumPair::F32(b)) => arith!(a, b, F32, 0.0_f32),
        (NumPair::F64(a), NumPair::F64(b)) => arith!(a, b, F64, 0.0_f64),
        _ => Err(InterpretError::Internal("promote returned mixed".into())),
    }
}

fn apply_cmp(
    node_id: usize,
    op: &OpKind,
    a: &ScalarValue,
    b: &ScalarValue,
) -> Result<ScalarValue> {
    use OpKind::*;
    // Try numeric promotion first; fall back to bool/string equality.
    if let Some(pair) = promote(a, b) {
        let result = match pair {
            (NumPair::I32(a), NumPair::I32(b)) => cmp_op(op, a, b),
            (NumPair::I64(a), NumPair::I64(b)) => cmp_op(op, a, b),
            (NumPair::F32(a), NumPair::F32(b)) => cmp_op(op, a, b),
            (NumPair::F64(a), NumPair::F64(b)) => cmp_op(op, a, b),
            _ => unreachable!(),
        };
        return Ok(ScalarValue::Bool(result));
    }
    match (a, b, op) {
        (ScalarValue::Bool(x), ScalarValue::Bool(y), Eq) => Ok(ScalarValue::Bool(x == y)),
        (ScalarValue::Bool(x), ScalarValue::Bool(y), Ne) => Ok(ScalarValue::Bool(x != y)),
        (ScalarValue::String(x), ScalarValue::String(y), Eq) => Ok(ScalarValue::Bool(x == y)),
        (ScalarValue::String(x), ScalarValue::String(y), Ne) => Ok(ScalarValue::Bool(x != y)),
        (ScalarValue::String(x), ScalarValue::String(y), Lt) => Ok(ScalarValue::Bool(x < y)),
        (ScalarValue::String(x), ScalarValue::String(y), Le) => Ok(ScalarValue::Bool(x <= y)),
        (ScalarValue::String(x), ScalarValue::String(y), Gt) => Ok(ScalarValue::Bool(x > y)),
        (ScalarValue::String(x), ScalarValue::String(y), Ge) => Ok(ScalarValue::Bool(x >= y)),
        _ => Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!(
                "{op:?}: incompatible operand types ({}, {})",
                a.type_name(),
                b.type_name()
            ),
        }),
    }
}

fn cmp_op<T: PartialOrd>(op: &OpKind, a: T, b: T) -> bool {
    use OpKind::*;
    match op {
        Eq => a == b,
        Ne => a != b,
        Lt => a < b,
        Le => a <= b,
        Gt => a > b,
        Ge => a >= b,
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_f32() {
        let r = apply_op(
            0,
            &OpKind::Add,
            &[ScalarValue::F32(1.5), ScalarValue::F32(2.25)],
        )
        .unwrap();
        assert_eq!(r, ScalarValue::F32(3.75));
    }

    #[test]
    fn add_promotes_i32_to_i64() {
        let r = apply_op(
            0,
            &OpKind::Add,
            &[ScalarValue::I32(1), ScalarValue::I64(2)],
        )
        .unwrap();
        assert_eq!(r, ScalarValue::I64(3));
    }

    #[test]
    fn div_by_zero_errors() {
        let err = apply_op(
            7,
            &OpKind::Div,
            &[ScalarValue::I32(1), ScalarValue::I32(0)],
        )
        .unwrap_err();
        assert!(matches!(err, InterpretError::DivByZero(7)));
    }

    #[test]
    fn lt_f32() {
        let r = apply_op(
            0,
            &OpKind::Lt,
            &[ScalarValue::F32(1.0), ScalarValue::F32(2.0)],
        )
        .unwrap();
        assert_eq!(r, ScalarValue::Bool(true));
    }

    #[test]
    fn and_or() {
        let a = apply_op(
            0,
            &OpKind::And,
            &[ScalarValue::Bool(true), ScalarValue::Bool(false)],
        )
        .unwrap();
        assert_eq!(a, ScalarValue::Bool(false));
        let o = apply_op(
            0,
            &OpKind::Or,
            &[ScalarValue::Bool(true), ScalarValue::Bool(false)],
        )
        .unwrap();
        assert_eq!(o, ScalarValue::Bool(true));
    }

    #[test]
    fn concat_strings() {
        let r = apply_op(
            0,
            &OpKind::Concat,
            &[
                ScalarValue::String("hello ".into()),
                ScalarValue::String("rce".into()),
            ],
        )
        .unwrap();
        assert_eq!(r, ScalarValue::String("hello rce".into()));
    }
}
