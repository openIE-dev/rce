//! Element-wise tensor kernels for the v0.1 CPU interpreter.
//!
//! Two operands of equal shape produce a tensor of the same shape. A
//! scalar (0-D tensor or a `ScalarValue` upstream) broadcasts to every
//! element of the other operand. Mixed-shape (non-scalar) broadcasting
//! is not yet supported.
//!
//! Type-promotion follows the same rule as scalar arithmetic: the wider
//! float wins; floats win over ints; otherwise the wider int wins.

use crate::error::{InterpretError, Result};
use crate::tensor::{DType, HostTensor};
use rce_graph::graph::OpKind;

fn promote_pair(a: DType, b: DType) -> DType {
    use DType::*;
    match (a, b) {
        (F64, _) | (_, F64) => F64,
        (F32, _) | (_, F32) => F32,
        // f16 / bf16 promote to f32 for binary arithmetic — kernels
        // operate in f32 and the result narrows back at the boundary
        // (or stays f32 — promotion is a widening, by design).
        (F16, _) | (_, F16) => F32,
        (BF16, _) | (_, BF16) => F32,
        (I64, _) | (_, I64) => I64,
        (I32, _) | (_, I32) => I32,
        (Bool, Bool) => Bool,
    }
}

fn broadcast_shape(node_id: usize, a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    crate::ml::broadcast::broadcast_shapes(node_id, a, b)
}

/// Element-wise binary op over two tensors (with full NumPy broadcasting).
pub fn binary(
    node_id: usize,
    op: &OpKind,
    a: &HostTensor,
    b: &HostTensor,
) -> Result<HostTensor> {
    let out_shape = broadcast_shape(node_id, &a.shape, &b.shape)?;
    // Materialise broadcasts so kernels see equal-shape operands.
    let a = crate::ml::broadcast::broadcast_to(node_id, a, &out_shape)?;
    let b = crate::ml::broadcast::broadcast_to(node_id, b, &out_shape)?;
    let a = &a;
    let b = &b;
    let out_dtype = match op {
        OpKind::Eq | OpKind::Ne | OpKind::Lt | OpKind::Le | OpKind::Gt | OpKind::Ge => DType::Bool,
        OpKind::And | OpKind::Or => DType::Bool,
        _ => promote_pair(a.dtype, b.dtype),
    };

    let n: usize = out_shape.iter().product::<usize>().max(1);

    macro_rules! arith {
        ($av:ident, $bv:ident, $ctor:ident, $zero:expr) => {{
            let av = expand($av, n)?;
            let bv = expand($bv, n)?;
            let out: Vec<_> = match op {
                OpKind::Add => av.iter().zip(&bv).map(|(x, y)| *x + *y).collect(),
                OpKind::Sub => av.iter().zip(&bv).map(|(x, y)| *x - *y).collect(),
                OpKind::Mul => av.iter().zip(&bv).map(|(x, y)| *x * *y).collect(),
                OpKind::Div => av
                    .iter()
                    .zip(&bv)
                    .map(|(x, y)| {
                        if *y == $zero {
                            Err(InterpretError::DivByZero(node_id))
                        } else {
                            Ok(*x / *y)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?,
                OpKind::Rem => av
                    .iter()
                    .zip(&bv)
                    .map(|(x, y)| {
                        if *y == $zero {
                            Err(InterpretError::DivByZero(node_id))
                        } else {
                            Ok(*x % *y)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?,
                _ => unreachable!(),
            };
            HostTensor::$ctor(out_shape.clone(), &out)
        }};
    }

    macro_rules! cmp {
        ($av:ident, $bv:ident) => {{
            let av = expand($av, n)?;
            let bv = expand($bv, n)?;
            let bools: Vec<bool> = match op {
                OpKind::Eq => av.iter().zip(&bv).map(|(x, y)| *x == *y).collect(),
                OpKind::Ne => av.iter().zip(&bv).map(|(x, y)| *x != *y).collect(),
                OpKind::Lt => av.iter().zip(&bv).map(|(x, y)| *x < *y).collect(),
                OpKind::Le => av.iter().zip(&bv).map(|(x, y)| *x <= *y).collect(),
                OpKind::Gt => av.iter().zip(&bv).map(|(x, y)| *x > *y).collect(),
                OpKind::Ge => av.iter().zip(&bv).map(|(x, y)| *x >= *y).collect(),
                _ => unreachable!(),
            };
            let bytes: Vec<u8> = bools.iter().map(|b| *b as u8).collect();
            HostTensor::new(out_shape.clone(), DType::Bool, bytes)
        }};
    }

    match op {
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div | OpKind::Rem => match out_dtype {
            DType::F64 => {
                let av = a.as_f64_promoted()?;
                let bv = b.as_f64_promoted()?;
                arith!(av, bv, from_f64, 0.0_f64)
            }
            DType::F32 => {
                let av = a.as_f32_promoted()?;
                let bv = b.as_f32_promoted()?;
                arith!(av, bv, from_f32, 0.0_f32)
            }
            DType::I64 => {
                let av = a.as_i64_promoted()?;
                let bv = b.as_i64_promoted()?;
                arith!(av, bv, from_i64, 0_i64)
            }
            DType::I32 => {
                let av = a.as_i32_promoted()?;
                let bv = b.as_i32_promoted()?;
                arith!(av, bv, from_i32, 0_i32)
            }
            DType::Bool => Err(InterpretError::TypeMismatch {
                node_id,
                msg: format!("{op:?}: cannot apply to bool tensors"),
            }),
            // `promote_pair` widens F16 / BF16 to F32, so out_dtype is
            // never one of these in practice. The arm is for
            // exhaustiveness only.
            DType::F16 | DType::BF16 => unreachable!(
                "promote_pair widens F16 / BF16 to F32; out_dtype = {} unreachable",
                out_dtype.name(),
            ),
        },
        OpKind::Eq | OpKind::Ne | OpKind::Lt | OpKind::Le | OpKind::Gt | OpKind::Ge => {
            let common = promote_pair(a.dtype, b.dtype);
            match common {
                DType::F64 => {
                    let av = a.as_f64_promoted()?;
                    let bv = b.as_f64_promoted()?;
                    cmp!(av, bv)
                }
                DType::F32 => {
                    let av = a.as_f32_promoted()?;
                    let bv = b.as_f32_promoted()?;
                    cmp!(av, bv)
                }
                DType::I64 => {
                    let av = a.as_i64_promoted()?;
                    let bv = b.as_i64_promoted()?;
                    cmp!(av, bv)
                }
                DType::I32 => {
                    let av = a.as_i32_promoted()?;
                    let bv = b.as_i32_promoted()?;
                    cmp!(av, bv)
                }
                DType::Bool => {
                    let av = a.as_bool()?;
                    let bv = b.as_bool()?;
                    cmp!(av, bv)
                }
                DType::F16 | DType::BF16 => unreachable!(
                    "promote_pair widens F16 / BF16 to F32; common = {} unreachable",
                    common.name(),
                ),
            }
        }
        OpKind::And | OpKind::Or => {
            let av = a.as_bool()?;
            let bv = b.as_bool()?;
            let av = expand(av, n)?;
            let bv = expand(bv, n)?;
            let bools: Vec<bool> = match op {
                OpKind::And => av.iter().zip(&bv).map(|(x, y)| *x && *y).collect(),
                OpKind::Or => av.iter().zip(&bv).map(|(x, y)| *x || *y).collect(),
                _ => unreachable!(),
            };
            let bytes: Vec<u8> = bools.iter().map(|b| *b as u8).collect();
            HostTensor::new(out_shape, DType::Bool, bytes)
        }
        _ => Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!("{op:?}: unexpected op in tensor binary kernel"),
        }),
    }
}

pub fn unary(node_id: usize, op: &OpKind, a: &HostTensor) -> Result<HostTensor> {
    match op {
        OpKind::Neg => match a.dtype {
            DType::F64 => HostTensor::from_f64(a.shape.clone(), &neg(a.as_f64()?)),
            DType::F32 => HostTensor::from_f32(a.shape.clone(), &neg(a.as_f32()?)),
            // F16 / BF16: promote to F32, negate, narrow back. Matches
            // the same widen-then-narrow pattern binary arithmetic uses.
            DType::F16 => {
                let vs = a.as_f32_promoted()?;
                HostTensor::from_f32_as_f16(a.shape.clone(), &neg(vs))
            }
            DType::BF16 => {
                let vs = a.as_f32_promoted()?;
                HostTensor::from_f32_as_bf16(a.shape.clone(), &neg(vs))
            }
            DType::I64 => HostTensor::from_i64(a.shape.clone(), &neg(a.as_i64()?)),
            DType::I32 => HostTensor::from_i32(a.shape.clone(), &neg(a.as_i32()?)),
            DType::Bool => Err(InterpretError::TypeMismatch {
                node_id,
                msg: "Neg: cannot apply to bool tensor".into(),
            }),
        },
        OpKind::Not => {
            let v = a.as_bool()?;
            let bytes: Vec<u8> = v.iter().map(|b| (!b) as u8).collect();
            HostTensor::new(a.shape.clone(), DType::Bool, bytes)
        }
        _ => Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!("{op:?}: not a unary op"),
        }),
    }
}

fn neg<T: std::ops::Neg<Output = T> + Copy>(v: Vec<T>) -> Vec<T> {
    v.into_iter().map(|x| -x).collect()
}

/// If `v` has length `n`, return as-is. If length 1 (scalar), broadcast.
fn expand<T: Clone>(v: Vec<T>, n: usize) -> Result<Vec<T>> {
    if v.len() == n {
        Ok(v)
    } else if v.len() == 1 && n > 0 {
        Ok(vec![v[0].clone(); n])
    } else if v.is_empty() && n == 0 {
        Ok(v)
    } else {
        Err(InterpretError::Internal(format!(
            "broadcast: vec len {} cannot expand to {n}",
            v.len()
        )))
    }
}

// ── HostTensor numeric promotion helpers (read with promotion to wider) ──

impl HostTensor {
    pub fn as_f64_promoted(&self) -> Result<Vec<f64>> {
        Ok(match self.dtype {
            DType::F64 => self.as_f64()?,
            DType::F32 => self.as_f32()?.into_iter().map(|x| x as f64).collect(),
            DType::F16 => self.data.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32() as f64).collect(),
            DType::BF16 => self.data.chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32() as f64).collect(),
            DType::I64 => self.as_i64()?.into_iter().map(|x| x as f64).collect(),
            DType::I32 => self.as_i32()?.into_iter().map(|x| x as f64).collect(),
            DType::Bool => self
                .as_bool()?
                .into_iter()
                .map(|b| b as i64 as f64)
                .collect(),
        })
    }

    pub fn as_f32_promoted(&self) -> Result<Vec<f32>> {
        Ok(match self.dtype {
            DType::F32 => self.as_f32()?,
            DType::F64 => self.as_f64()?.into_iter().map(|x| x as f32).collect(),
            DType::F16 => self.data.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            DType::BF16 => self.data.chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            DType::I64 => self.as_i64()?.into_iter().map(|x| x as f32).collect(),
            DType::I32 => self.as_i32()?.into_iter().map(|x| x as f32).collect(),
            DType::Bool => self
                .as_bool()?
                .into_iter()
                .map(|b| b as i32 as f32)
                .collect(),
        })
    }

    pub fn as_i64_promoted(&self) -> Result<Vec<i64>> {
        Ok(match self.dtype {
            DType::I64 => self.as_i64()?,
            DType::I32 => self.as_i32()?.into_iter().map(|x| x as i64).collect(),
            DType::Bool => self.as_bool()?.into_iter().map(|b| b as i64).collect(),
            DType::F32 | DType::F64 | DType::F16 | DType::BF16 => {
                return Err(InterpretError::TypeMismatch {
                    node_id: usize::MAX,
                    msg: format!("cannot promote {} to i64", self.dtype.name()),
                });
            }
        })
    }

    pub fn as_i32_promoted(&self) -> Result<Vec<i32>> {
        Ok(match self.dtype {
            DType::I32 => self.as_i32()?,
            DType::Bool => self.as_bool()?.into_iter().map(|b| b as i32).collect(),
            DType::I64 | DType::F32 | DType::F64 | DType::F16 | DType::BF16 => {
                return Err(InterpretError::TypeMismatch {
                    node_id: usize::MAX,
                    msg: format!("cannot promote {} to i32", self.dtype.name()),
                });
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elementwise_add_f32() {
        let a = HostTensor::from_f32(vec![3], &[1.0, 2.0, 3.0]).unwrap();
        let b = HostTensor::from_f32(vec![3], &[4.0, 5.0, 6.0]).unwrap();
        let c = binary(0, &OpKind::Add, &a, &b).unwrap();
        assert_eq!(c.as_f32().unwrap(), vec![5.0, 7.0, 9.0]);
        assert_eq!(c.shape, vec![3]);
    }

    #[test]
    fn scalar_broadcast() {
        let a = HostTensor::from_f32(vec![3], &[1.0, 2.0, 3.0]).unwrap();
        let s = HostTensor::from_f32(vec![], &[10.0]).unwrap();
        let c = binary(0, &OpKind::Mul, &a, &s).unwrap();
        assert_eq!(c.as_f32().unwrap(), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn promote_i32_plus_f32_gives_f32() {
        let a = HostTensor::from_i32(vec![3], &[1, 2, 3]).unwrap();
        let b = HostTensor::from_f32(vec![3], &[0.5, 0.5, 0.5]).unwrap();
        let c = binary(0, &OpKind::Add, &a, &b).unwrap();
        assert_eq!(c.dtype, DType::F32);
        assert_eq!(c.as_f32().unwrap(), vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn elementwise_compare_emits_bool_tensor() {
        let a = HostTensor::from_f32(vec![3], &[1.0, 2.0, 3.0]).unwrap();
        let b = HostTensor::from_f32(vec![3], &[2.0, 2.0, 2.0]).unwrap();
        let c = binary(0, &OpKind::Lt, &a, &b).unwrap();
        assert_eq!(c.dtype, DType::Bool);
        assert_eq!(c.as_bool().unwrap(), vec![true, false, false]);
    }

    #[test]
    fn neg_f64() {
        let a = HostTensor::from_f64(vec![2, 2], &[1.0, -2.0, 3.0, -4.0]).unwrap();
        let c = unary(0, &OpKind::Neg, &a).unwrap();
        assert_eq!(c.as_f64().unwrap(), vec![-1.0, 2.0, -3.0, 4.0]);
        assert_eq!(c.shape, vec![2, 2]);
    }

    #[test]
    fn divide_by_zero_in_tensor_kernel() {
        let a = HostTensor::from_i32(vec![3], &[1, 2, 3]).unwrap();
        let b = HostTensor::from_i32(vec![3], &[1, 0, 1]).unwrap();
        let err = binary(7, &OpKind::Div, &a, &b).unwrap_err();
        assert!(matches!(err, InterpretError::DivByZero(7)));
    }

    #[test]
    fn shape_mismatch_errors() {
        let a = HostTensor::from_f32(vec![3], &[1.0, 2.0, 3.0]).unwrap();
        let b = HostTensor::from_f32(vec![2], &[1.0, 2.0]).unwrap();
        let err = binary(0, &OpKind::Add, &a, &b).unwrap_err();
        assert!(matches!(err, InterpretError::TypeMismatch { .. }));
    }
}
