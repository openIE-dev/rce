//! `relu`, f32/f64. The other activations are in `nn`, with RCE's own
//! transcendental functions.

use super::arg_tensor;
use crate::error::{InterpretError, Result};
use crate::tensor::{DType, HostTensor};
use crate::value::Value;

fn arity1(node_id: usize, name: &str, args: &[Value]) -> Result<HostTensor> {
    if args.len() != 1 {
        return Err(InterpretError::ArityMismatch {
            node_id,
            kind: name.into(),
            expected: 1,
            got: args.len(),
        });
    }
    arg_tensor(node_id, &args[0])
}

macro_rules! unary_act {
    ($name:ident, $public:ident, $expr_f32:expr, $expr_f64:expr) => {
        pub fn $public(node_id: usize, args: &[Value]) -> Result<Value> {
            let x = arity1(node_id, stringify!($public), args)?;
            match x.dtype {
                DType::F32 => {
                    let v: Vec<f32> = x.as_f32()?.into_iter().map($expr_f32).collect();
                    HostTensor::from_f32(x.shape.clone(), &v).map(Value::Tensor)
                }
                DType::F64 => {
                    let v: Vec<f64> = x.as_f64()?.into_iter().map($expr_f64).collect();
                    HostTensor::from_f64(x.shape.clone(), &v).map(Value::Tensor)
                }
                _ => Err(InterpretError::TypeMismatch {
                    node_id,
                    msg: format!(
                        "{}: dtype {} not supported (f32/f64 only)",
                        stringify!($public),
                        x.dtype.name()
                    ),
                }),
            }
        }
    };
}

// relu(x) is x when x > 0 and +0.0 otherwise. `f32::max` leaves the sign of
// a zero result to the platform (x86 maxss and aarch64 fmaxnm differ on
// max(-0.0, 0.0)), and RCE keeps -0.0 distinct from 0.0 (spec 6.3 F6), so
// the reference names the result for every input.
unary_act!(
    relu_impl,
    relu,
    |x: f32| if x > 0.0 { x } else { 0.0 },
    |x: f64| if x > 0.0 { x } else { 0.0 }
);
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_of_negative_zero_is_positive_zero() {
        let x = HostTensor::from_f32(vec![2], &[-0.0, 0.0]).unwrap();
        let Value::Tensor(t) = relu(0, &[Value::Tensor(x)]).unwrap() else {
            panic!()
        };
        let bits: Vec<u32> = t.as_f32().unwrap().iter().map(|v| v.to_bits()).collect();
        assert_eq!(bits, vec![0, 0]);
    }

    #[test]
    fn relu_f32() {
        let x = HostTensor::from_f32(vec![5], &[-1.0, 0.0, 1.0, -2.0, 3.0]).unwrap();
        let Value::Tensor(t) = relu(0, &[Value::Tensor(x)]).unwrap() else {
            panic!()
        };
        assert_eq!(t.as_f32().unwrap(), vec![0.0, 0.0, 1.0, 0.0, 3.0]);
    }




    #[test]
    fn gelu_monotone_for_positive() {
        let x = HostTensor::from_f32(vec![3], &[0.5, 1.0, 2.0]).unwrap();
        let Value::Tensor(t) = crate::ml::nn::unary(0, "gelu", &[Value::Tensor(x)]).unwrap() else {
            panic!()
        };
        let v = t.as_f32().unwrap();
        assert!(v[0] < v[1] && v[1] < v[2]);
    }
}
