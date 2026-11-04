//! Dense matmul. v0.1 supports `f32` and `f64`.
//!
//! - 2D: `[M, K] × [K, N] → [M, N]`.
//! - Batched: leading dims broadcast NumPy-style; trailing 2 dims are the matmul.

use super::{arg_tensor, broadcast::broadcast_shapes};
use crate::error::{InterpretError, Result};
use crate::tensor::{DType, HostTensor};
use crate::value::Value;

pub fn matmul(node_id: usize, args: &[Value]) -> Result<Value> {
    if args.len() != 2 {
        return Err(InterpretError::ArityMismatch {
            node_id,
            kind: "matmul".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let a = arg_tensor(node_id, &args[0])?;
    let b = arg_tensor(node_id, &args[1])?;
    if a.shape.len() < 2 || b.shape.len() < 2 {
        return Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!("matmul requires rank ≥ 2 (got {:?} and {:?})", a.shape, b.shape),
        });
    }
    if a.dtype != b.dtype {
        return Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!(
                "matmul dtype mismatch: a={}, b={}; promote explicitly",
                a.dtype.name(),
                b.dtype.name()
            ),
        });
    }

    let arank = a.shape.len();
    let brank = b.shape.len();
    let m = a.shape[arank - 2];
    let k_a = a.shape[arank - 1];
    let k_b = b.shape[brank - 2];
    let n = b.shape[brank - 1];
    if k_a != k_b {
        return Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!(
                "matmul inner dims mismatch: {:?} × {:?} ({k_a} vs {k_b})",
                a.shape, b.shape
            ),
        });
    }

    // Broadcast leading batch dims.
    let a_batch = &a.shape[..arank - 2];
    let b_batch = &b.shape[..brank - 2];
    let batch = broadcast_shapes(node_id, a_batch, b_batch)?;
    let mut out_shape = batch.clone();
    out_shape.push(m);
    out_shape.push(n);

    match a.dtype {
        DType::F32 => {
            let av = a.as_f32()?;
            let bv = b.as_f32()?;
            let out = batched_matmul_f32(&av, &a.shape, &bv, &b.shape, &batch, m, k_a, n);
            HostTensor::from_f32(out_shape, &out).map(Value::Tensor)
        }
        DType::F64 => {
            let av = a.as_f64()?;
            let bv = b.as_f64()?;
            let out = batched_matmul_f64(&av, &a.shape, &bv, &b.shape, &batch, m, k_a, n);
            HostTensor::from_f64(out_shape, &out).map(Value::Tensor)
        }
        _ => Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!("matmul: dtype {} not supported in v0.1 (use f32/f64)", a.dtype.name()),
        }),
    }
}

macro_rules! batched_matmul {
    ($name:ident, $T:ty, $zero:expr) => {
        fn $name(
            a: &[$T],
            a_shape: &[usize],
            b: &[$T],
            b_shape: &[usize],
            batch: &[usize],
            m: usize,
            k: usize,
            n: usize,
        ) -> Vec<$T> {
            let batch_size: usize = batch.iter().product::<usize>().max(1);
            let a_batch = &a_shape[..a_shape.len() - 2];
            let b_batch = &b_shape[..b_shape.len() - 2];
            let a_mat: usize = m * k;
            let b_mat: usize = k * n;
            let out_mat: usize = m * n;
            let mut out = vec![$zero; batch_size * out_mat];

            // Pre-compute strides for batch index → matrix offset for both
            // operands, accounting for size-1 broadcasts.
            for batch_idx in 0..batch_size {
                // unflatten batch_idx into idx[]
                let mut idx = vec![0usize; batch.len()];
                let mut rem = batch_idx;
                for i in (0..batch.len()).rev() {
                    idx[i] = rem % batch[i];
                    rem /= batch[i];
                }

                let a_off = locate_batch(a_batch, &idx) * a_mat;
                let b_off = locate_batch(b_batch, &idx) * b_mat;
                let out_off = batch_idx * out_mat;

                let am = &a[a_off..a_off + a_mat];
                let bm = &b[b_off..b_off + b_mat];
                let om = &mut out[out_off..out_off + out_mat];

                // Naive triple loop. Fine for v0.1 (correctness > speed);
                // SIMD/blocking lands with `rce-vm-backend-cpu` v0.2.
                for i in 0..m {
                    for j in 0..n {
                        let mut acc: $T = $zero;
                        for kk in 0..k {
                            acc += am[i * k + kk] * bm[kk * n + j];
                        }
                        om[i * n + j] = acc;
                    }
                }
            }
            out
        }
    };
}

batched_matmul!(batched_matmul_f32, f32, 0.0);
batched_matmul!(batched_matmul_f64, f64, 0.0);

/// Map a broadcast batch index back into the operand's local index, treating
/// leading dims (rank padding) and any size-1 dims as broadcasted.
fn locate_batch(operand_batch: &[usize], full_idx: &[usize]) -> usize {
    let pad = full_idx.len() - operand_batch.len();
    let mut linear = 0usize;
    let mut stride = 1usize;
    for i in (0..operand_batch.len()).rev() {
        let d = operand_batch[i];
        let target_axis = pad + i;
        let coord = if d == 1 { 0 } else { full_idx[target_axis] };
        linear += coord * stride;
        stride *= d;
    }
    linear
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_2d_f32() {
        // [[1,2,3],[4,5,6]] × [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
        let a = HostTensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = HostTensor::from_f32(vec![3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let out = matmul(0, &[Value::Tensor(a), Value::Tensor(b)]).unwrap();
        match out {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.as_f32().unwrap(), vec![58.0, 64.0, 139.0, 154.0]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn matmul_batched_f32() {
        // (2,2,3) × (2,3,2) → (2,2,2)
        let a = HostTensor::from_f32(
            vec![2, 2, 3],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, // batch 1
            ],
        )
        .unwrap();
        let b = HostTensor::from_f32(
            vec![2, 3, 2],
            &[
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch 0
                1.0, 0.0, 0.0, 1.0, 0.0, 0.0, // batch 1
            ],
        )
        .unwrap();
        let out = matmul(0, &[Value::Tensor(a), Value::Tensor(b)]).unwrap();
        let Value::Tensor(t) = out else { panic!() };
        assert_eq!(t.shape, vec![2, 2, 2]);
        let v = t.as_f32().unwrap();
        assert_eq!(&v[..4], &[58.0, 64.0, 139.0, 154.0]);
        assert_eq!(&v[4..], &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn matmul_inner_dim_mismatch_errors() {
        let a = HostTensor::from_f32(vec![2, 3], &[0.0; 6]).unwrap();
        let b = HostTensor::from_f32(vec![4, 2], &[0.0; 8]).unwrap();
        let err = matmul(7, &[Value::Tensor(a), Value::Tensor(b)]).unwrap_err();
        assert!(matches!(err, InterpretError::TypeMismatch { node_id: 7, .. }));
    }
}
