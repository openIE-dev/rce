//! Full NumPy-style broadcasting.
//!
//! Two shapes are compatible if, when right-aligned, every dim is either
//! equal or one of them is `1`. The output shape uses the larger of the
//! two at each position. `broadcast_to` materialises a contiguous tensor
//! at the broadcast shape (no lazy strides yet — that lands with the SIMD
//! kernel pass in v0.2).

use crate::error::{InterpretError, Result};
use crate::tensor::HostTensor;

pub fn broadcast_shapes(node_id: usize, a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = vec![0usize; n];
    for i in 0..n {
        let av = if i < n - a.len() { 1 } else { a[i - (n - a.len())] };
        let bv = if i < n - b.len() { 1 } else { b[i - (n - b.len())] };
        if av == bv {
            out[i] = av;
        } else if av == 1 {
            out[i] = bv;
        } else if bv == 1 {
            out[i] = av;
        } else {
            return Err(InterpretError::TypeMismatch {
                node_id,
                msg: format!(
                    "shapes {a:?} and {b:?} not broadcast-compatible at dim {i} ({av} vs {bv})"
                ),
            });
        }
    }
    Ok(out)
}

/// Materialise `t` to the broadcast `target` shape.
pub fn broadcast_to(node_id: usize, t: &HostTensor, target: &[usize]) -> Result<HostTensor> {
    if t.shape == target {
        return Ok(t.clone());
    }
    let elem = t.dtype.byte_size();
    let total: usize = target.iter().product();
    let mut out = vec![0u8; total * elem];

    // For each linear index in `target`, compute the corresponding source
    // index in `t` by mapping each dim: source_i = target_i if t.dim==target.dim
    // else 0 (dim is 1 in source). Right-aligned mapping for rank padding.
    let src_rank = t.shape.len();
    let dst_rank = target.len();
    if dst_rank < src_rank {
        return Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!("cannot broadcast {:?} to {:?} (rank shrinks)", t.shape, target),
        });
    }
    // Validate dims.
    for i in 0..src_rank {
        let s = t.shape[src_rank - 1 - i];
        let d = target[dst_rank - 1 - i];
        if s != d && s != 1 {
            return Err(InterpretError::TypeMismatch {
                node_id,
                msg: format!(
                    "dim {} mismatch broadcasting {:?} → {:?}: src={s} dst={d}",
                    dst_rank - 1 - i,
                    t.shape,
                    target
                ),
            });
        }
    }
    // Pre-compute source strides (row-major C-order).
    let mut src_strides = vec![0usize; src_rank];
    if src_rank > 0 {
        src_strides[src_rank - 1] = 1;
        for i in (0..src_rank - 1).rev() {
            src_strides[i] = src_strides[i + 1] * t.shape[i + 1];
        }
    }
    // Iterate target; map to source.
    let mut idx = vec![0usize; dst_rank];
    for linear in 0..total {
        // unflatten linear → idx
        let mut rem = linear;
        for i in (0..dst_rank).rev() {
            idx[i] = rem % target[i];
            rem /= target[i];
        }
        // map to src
        let mut src_off = 0usize;
        for i in 0..src_rank {
            let dst_axis = dst_rank - 1 - i;
            let src_axis = src_rank - 1 - i;
            let coord = if t.shape[src_axis] == 1 {
                0
            } else {
                idx[dst_axis]
            };
            src_off += coord * src_strides[src_axis];
        }
        let dst_off = linear * elem;
        let s_off = src_off * elem;
        out[dst_off..dst_off + elem].copy_from_slice(&t.data[s_off..s_off + elem]);
    }
    HostTensor::new(target.to_vec(), t.dtype, out)
}

/// Convenience: promote dtype + materialise broadcast for a binary kernel.
pub fn align_pair(
    node_id: usize,
    a: &HostTensor,
    b: &HostTensor,
) -> Result<(HostTensor, HostTensor, Vec<usize>)> {
    let shape = broadcast_shapes(node_id, &a.shape, &b.shape)?;
    let aa = broadcast_to(node_id, a, &shape)?;
    let bb = broadcast_to(node_id, b, &shape)?;
    Ok((aa, bb, shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_rules() {
        assert_eq!(broadcast_shapes(0, &[3, 4], &[3, 4]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shapes(0, &[3, 1], &[1, 4]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shapes(0, &[5], &[3, 5]).unwrap(), vec![3, 5]);
        assert_eq!(broadcast_shapes(0, &[], &[3, 4]).unwrap(), vec![3, 4]);
        assert!(broadcast_shapes(0, &[3, 5], &[3, 4]).is_err());
    }

    #[test]
    fn broadcasts_row_vector_over_matrix() {
        // shape (3,) → (2,3): each row of the (2,3) result is the row vector.
        let v = HostTensor::from_f32(vec![3], &[1.0, 2.0, 3.0]).unwrap();
        let out = broadcast_to(0, &v, &[2, 3]).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(out.as_f32().unwrap(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn broadcasts_column_vector_over_matrix() {
        // shape (3,1) → (3,4): each column of the result equals the column vector.
        let v = HostTensor::from_f32(vec![3, 1], &[10.0, 20.0, 30.0]).unwrap();
        let out = broadcast_to(0, &v, &[3, 4]).unwrap();
        assert_eq!(out.shape, vec![3, 4]);
        assert_eq!(
            out.as_f32().unwrap(),
            vec![10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 30.0]
        );
    }
}
