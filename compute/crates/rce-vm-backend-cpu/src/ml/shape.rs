//! Shape ops: transpose, reshape, concat.

use super::{arg_i32, arg_string, arg_tensor, parse_axes, parse_shape};
use crate::error::{InterpretError, Result};
use crate::tensor::HostTensor;
use crate::value::Value;

pub fn transpose(node_id: usize, args: &[Value]) -> Result<Value> {
    if args.len() != 2 {
        return Err(InterpretError::ArityMismatch {
            node_id,
            kind: "transpose".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let x = arg_tensor(node_id, &args[0])?;
    let axes_str = arg_string(node_id, &args[1])?;
    let axes = parse_axes(node_id, &axes_str, x.shape.len())?;
    if axes.len() != x.shape.len() {
        return Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!(
                "transpose axes {axes:?} length {} != tensor rank {}",
                axes.len(),
                x.shape.len()
            ),
        });
    }
    // Validate it's a permutation.
    let mut seen = vec![false; x.shape.len()];
    for &a in &axes {
        if seen[a] {
            return Err(InterpretError::TypeMismatch {
                node_id,
                msg: format!("transpose axes must be a permutation; got {axes:?}"),
            });
        }
        seen[a] = true;
    }

    let new_shape: Vec<usize> = axes.iter().map(|&a| x.shape[a]).collect();
    let elem = x.dtype.byte_size();
    let total: usize = new_shape.iter().product::<usize>().max(1);
    let mut out = vec![0u8; total * elem];

    // src strides in C-order.
    let rank = x.shape.len();
    let mut src_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * x.shape[i + 1];
    }
    let mut dst_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        dst_strides[i] = dst_strides[i + 1] * new_shape[i + 1];
    }

    let mut idx = vec![0usize; rank];
    for linear in 0..total {
        // unflatten linear → idx (in dst space)
        let mut rem = linear;
        for i in 0..rank {
            idx[i] = rem / dst_strides[i];
            rem %= dst_strides[i];
        }
        // map to src index using axes
        let mut src_off = 0usize;
        for i in 0..rank {
            src_off += idx[i] * src_strides[axes[i]];
        }
        out[linear * elem..(linear + 1) * elem]
            .copy_from_slice(&x.data[src_off * elem..(src_off + 1) * elem]);
    }
    HostTensor::new(new_shape, x.dtype, out).map(Value::Tensor)
}

pub fn reshape(node_id: usize, args: &[Value]) -> Result<Value> {
    if args.len() != 2 {
        return Err(InterpretError::ArityMismatch {
            node_id,
            kind: "reshape".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let x = arg_tensor(node_id, &args[0])?;
    let shape_str = arg_string(node_id, &args[1])?;
    let new_shape = parse_shape(node_id, &shape_str)?;
    let old_numel: usize = x.shape.iter().product::<usize>().max(1);
    let new_numel: usize = new_shape.iter().product::<usize>().max(1);
    if old_numel != new_numel {
        return Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!(
                "reshape numel mismatch: {:?} ({old_numel}) → {:?} ({new_numel})",
                x.shape, new_shape
            ),
        });
    }
    HostTensor::new(new_shape, x.dtype, x.data.clone()).map(Value::Tensor)
}

pub fn concat(node_id: usize, args: &[Value]) -> Result<Value> {
    if args.len() != 3 {
        return Err(InterpretError::ArityMismatch {
            node_id,
            kind: "tensor_concat".into(),
            expected: 3,
            got: args.len(),
        });
    }
    let a = arg_tensor(node_id, &args[0])?;
    let b = arg_tensor(node_id, &args[1])?;
    let axis_signed = arg_i32(node_id, &args[2])?;
    if a.shape.len() != b.shape.len() {
        return Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!(
                "concat: rank mismatch {:?} vs {:?}",
                a.shape, b.shape
            ),
        });
    }
    if a.dtype != b.dtype {
        return Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!(
                "concat: dtype mismatch {} vs {}",
                a.dtype.name(),
                b.dtype.name()
            ),
        });
    }
    let rank = a.shape.len();
    let axis = if axis_signed < 0 {
        (rank as i32 + axis_signed) as usize
    } else {
        axis_signed as usize
    };
    if axis >= rank {
        return Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!("concat: axis {axis_signed} out of range for rank {rank}"),
        });
    }
    for i in 0..rank {
        if i != axis && a.shape[i] != b.shape[i] {
            return Err(InterpretError::TypeMismatch {
                node_id,
                msg: format!(
                    "concat: shapes {:?} and {:?} disagree at non-axis dim {i}",
                    a.shape, b.shape
                ),
            });
        }
    }

    // Build output shape.
    let mut new_shape = a.shape.clone();
    new_shape[axis] = a.shape[axis] + b.shape[axis];

    let elem = a.dtype.byte_size();
    let outer: usize = a.shape[..axis].iter().product::<usize>().max(1);
    let a_inner: usize = a.shape[axis..].iter().product();
    let b_inner: usize = b.shape[axis..].iter().product();
    let out_inner = a_inner + b_inner;
    let mut out = vec![0u8; outer * out_inner * elem];
    for o in 0..outer {
        out[o * out_inner * elem..o * out_inner * elem + a_inner * elem]
            .copy_from_slice(&a.data[o * a_inner * elem..(o + 1) * a_inner * elem]);
        out[o * out_inner * elem + a_inner * elem..(o + 1) * out_inner * elem]
            .copy_from_slice(&b.data[o * b_inner * elem..(o + 1) * b_inner * elem]);
    }
    HostTensor::new(new_shape, a.dtype, out).map(Value::Tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::ScalarValue;

    fn s(t: &str) -> Value {
        Value::Scalar(ScalarValue::String(t.into()))
    }
    fn i(n: i32) -> Value {
        Value::Scalar(ScalarValue::I32(n))
    }

    #[test]
    fn transpose_2d() {
        let x = HostTensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let v = transpose(0, &[Value::Tensor(x), s("[1,0]")]).unwrap();
        let Value::Tensor(t) = v else { panic!() };
        assert_eq!(t.shape, vec![3, 2]);
        assert_eq!(t.as_f32().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn reshape_round_trips_data() {
        let x = HostTensor::from_f32(vec![6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let v = reshape(0, &[Value::Tensor(x), s("[2,3]")]).unwrap();
        let Value::Tensor(t) = v else { panic!() };
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.as_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn concat_along_axis_1() {
        let a = HostTensor::from_f32(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = HostTensor::from_f32(vec![2, 1], &[10.0, 20.0]).unwrap();
        let v = concat(0, &[Value::Tensor(a), Value::Tensor(b), i(1)]).unwrap();
        let Value::Tensor(t) = v else { panic!() };
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.as_f32().unwrap(), vec![1.0, 2.0, 10.0, 3.0, 4.0, 20.0]);
    }

    #[test]
    fn concat_dim_mismatch_errors() {
        let a = HostTensor::from_f32(vec![2, 2], &[0.0; 4]).unwrap();
        let b = HostTensor::from_f32(vec![3, 2], &[0.0; 6]).unwrap();
        let err = concat(7, &[Value::Tensor(a), Value::Tensor(b), i(1)]).unwrap_err();
        assert!(matches!(err, InterpretError::TypeMismatch { node_id: 7, .. }));
    }
}
