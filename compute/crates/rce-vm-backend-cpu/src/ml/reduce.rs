//! Reductions (sum/mean/max/min). Softmax is in `nn`.

use super::{arg_bool, arg_string, arg_tensor, parse_axes};
use crate::error::{InterpretError, Result};
use crate::tensor::{DType, HostTensor};
use crate::value::Value;

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Sum,
    Mean,
    Max,
    Min,
}

pub fn reduce(node_id: usize, args: &[Value], op: Op) -> Result<Value> {
    if args.len() != 3 {
        return Err(InterpretError::ArityMismatch {
            node_id,
            kind: format!("reduce_{op:?}"),
            expected: 3,
            got: args.len(),
        });
    }
    let x = arg_tensor(node_id, &args[0])?;
    let axes_str = arg_string(node_id, &args[1])?;
    let keepdim = arg_bool(node_id, &args[2])?;
    let axes = parse_axes(node_id, &axes_str, x.shape.len())?;

    macro_rules! reduce_one {
        ($T:ty, $as:ident, $from:ident) => {{
            let data = x.$as()?;
            let (out_data, out_shape) = reduce_inner::<$T>(&data, &x.shape, &axes, keepdim, op);
            HostTensor::$from(out_shape, &out_data).map(Value::Tensor)
        }};
    }
    match x.dtype {
        DType::F32 => reduce_one!(f32, as_f32, from_f32),
        DType::F64 => reduce_one!(f64, as_f64, from_f64),
        DType::I32 => reduce_one!(i32, as_i32, from_i32),
        DType::I64 => reduce_one!(i64, as_i64, from_i64),
        // F16 / BF16: promote to f32, reduce, narrow back. The
        // accumulator inside `reduce_inner` already runs in f32
        // precision regardless of the input dtype — only the storage
        // changes here.
        DType::F16 => {
            let data = x.as_f32_promoted()?;
            let (out_data, out_shape) = reduce_inner::<f32>(&data, &x.shape, &axes, keepdim, op);
            HostTensor::from_f32_as_f16(out_shape, &out_data).map(Value::Tensor)
        }
        DType::BF16 => {
            let data = x.as_f32_promoted()?;
            let (out_data, out_shape) = reduce_inner::<f32>(&data, &x.shape, &axes, keepdim, op);
            HostTensor::from_f32_as_bf16(out_shape, &out_data).map(Value::Tensor)
        }
        DType::Bool => Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!("reduce: bool tensors not supported in v0.1"),
        }),
    }
}

trait Numeric: Copy + std::ops::Add<Output = Self> + PartialOrd {
    const ZERO: Self;
    fn from_count(c: usize) -> Self;
    fn div(self, n: Self) -> Self;
    fn min_value() -> Self;
    fn max_value() -> Self;
}

impl Numeric for f32 {
    const ZERO: Self = 0.0;
    fn from_count(c: usize) -> Self { c as f32 }
    fn div(self, n: Self) -> Self { self / n }
    fn min_value() -> Self { f32::NEG_INFINITY }
    fn max_value() -> Self { f32::INFINITY }
}
impl Numeric for f64 {
    const ZERO: Self = 0.0;
    fn from_count(c: usize) -> Self { c as f64 }
    fn div(self, n: Self) -> Self { self / n }
    fn min_value() -> Self { f64::NEG_INFINITY }
    fn max_value() -> Self { f64::INFINITY }
}
impl Numeric for i32 {
    const ZERO: Self = 0;
    fn from_count(c: usize) -> Self { c as i32 }
    fn div(self, n: Self) -> Self { self / n }
    fn min_value() -> Self { i32::MIN }
    fn max_value() -> Self { i32::MAX }
}
impl Numeric for i64 {
    const ZERO: Self = 0;
    fn from_count(c: usize) -> Self { c as i64 }
    fn div(self, n: Self) -> Self { self / n }
    fn min_value() -> Self { i64::MIN }
    fn max_value() -> Self { i64::MAX }
}

fn reduce_inner<T: Numeric>(
    data: &[T],
    shape: &[usize],
    axes: &[usize],
    keepdim: bool,
    op: Op,
) -> (Vec<T>, Vec<usize>) {
    let rank = shape.len();
    let kept: Vec<usize> = (0..rank).filter(|i| !axes.contains(i)).collect();
    let kept_shape: Vec<usize> = kept.iter().map(|&i| shape[i]).collect();
    let kept_size: usize = kept_shape.iter().product::<usize>().max(1);
    let red_size: usize = axes.iter().map(|&i| shape[i]).product::<usize>().max(1);

    let init: T = match op {
        Op::Sum | Op::Mean => T::ZERO,
        Op::Max => T::min_value(),
        Op::Min => T::max_value(),
    };
    let mut out = vec![init; kept_size];
    // Pre-compute row-major C-order strides.
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let total: usize = shape.iter().product::<usize>().max(1);
    for linear in 0..total {
        // unflatten linear → coord
        let mut rem = linear;
        let mut coord = vec![0usize; rank];
        for i in 0..rank {
            coord[i] = rem / strides[i];
            rem %= strides[i];
        }
        // compute kept index
        let mut kept_linear = 0usize;
        let mut kept_stride = 1usize;
        for &k in kept.iter().rev() {
            kept_linear += coord[k] * kept_stride;
            kept_stride *= shape[k];
        }
        let v = data[linear];
        out[kept_linear] = match op {
            Op::Sum | Op::Mean => out[kept_linear] + v,
            Op::Max => if v > out[kept_linear] { v } else { out[kept_linear] },
            Op::Min => if v < out[kept_linear] { v } else { out[kept_linear] },
        };
    }
    if let Op::Mean = op {
        let n = T::from_count(red_size);
        for x in &mut out {
            *x = (*x).div(n);
        }
    }
    let out_shape = if keepdim {
        let mut s = vec![1usize; rank];
        for i in 0..rank {
            if !axes.contains(&i) {
                s[i] = shape[i];
            }
        }
        s
    } else {
        kept_shape
    };
    (out, out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::ScalarValue;

    fn b(v: bool) -> Value {
        Value::Scalar(ScalarValue::Bool(v))
    }
    fn s(t: &str) -> Value {
        Value::Scalar(ScalarValue::String(t.into()))
    }
    fn i(n: i32) -> Value {
        Value::Scalar(ScalarValue::I32(n))
    }

    #[test]
    fn sum_full() {
        let x = HostTensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let v = reduce(0, &[Value::Tensor(x), s("[0,1]"), b(false)], Op::Sum).unwrap();
        let Value::Tensor(t) = v else { panic!() };
        assert_eq!(t.shape, Vec::<usize>::new());
        assert_eq!(t.as_f32().unwrap(), vec![21.0]);
    }

    #[test]
    fn sum_axis_1_keepdim() {
        let x = HostTensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let v = reduce(0, &[Value::Tensor(x), s("[1]"), b(true)], Op::Sum).unwrap();
        let Value::Tensor(t) = v else { panic!() };
        assert_eq!(t.shape, vec![2, 1]);
        assert_eq!(t.as_f32().unwrap(), vec![6.0, 15.0]);
    }

    #[test]
    fn mean_axis_0() {
        let x = HostTensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let v = reduce(0, &[Value::Tensor(x), s("[0]"), b(false)], Op::Mean).unwrap();
        let Value::Tensor(t) = v else { panic!() };
        assert_eq!(t.shape, vec![3]);
        assert_eq!(t.as_f32().unwrap(), vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn max_axis_1() {
        let x = HostTensor::from_f32(vec![2, 3], &[1.0, 5.0, 3.0, 7.0, 2.0, 6.0]).unwrap();
        let v = reduce(0, &[Value::Tensor(x), s("[1]"), b(false)], Op::Max).unwrap();
        let Value::Tensor(t) = v else { panic!() };
        assert_eq!(t.shape, vec![2]);
        assert_eq!(t.as_f32().unwrap(), vec![5.0, 7.0]);
    }

    #[test]
    fn softmax_along_last_axis() {
        let x = HostTensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();
        let v = crate::ml::nn::softmax(0, &[Value::Tensor(x), i(-1)]).unwrap();
        let Value::Tensor(t) = v else { panic!() };
        let v = t.as_f32().unwrap();
        // each row should sum to 1
        let s0: f32 = v[..3].iter().sum();
        let s1: f32 = v[3..].iter().sum();
        assert!((s0 - 1.0).abs() < 1e-5);
        assert!((s1 - 1.0).abs() < 1e-5);
        // and rows are equal (same input)
        assert_eq!(v[..3], v[3..]);
    }
}
