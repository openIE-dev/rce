//! RCE's CPU backend: the reference evaluator of graph terms.
//!
//! Walks a `rce_graph::graph::FunctionGraph` in topological order and
//! evaluates each node: scalar arithmetic, comparison and logic over
//! `f32 | f64 | i32 | i64 | bool | String`, and the tensor kernels of `ml`.
//! [`interpret_reference`] is the evaluation the checker's reference
//! interpreter uses for a step's graph term (spec 6.13): a fixed node
//! order, CPU kernels only, and the finite-value rule of 6.3 N3.
//! `rce graph run FILE INPUTS` evaluates a `.fg` file with it. The model
//! kernels (`ml::nn`) take `exp`, `tanh` and `erf` from [`crmath`], the
//! functions of RCE's `math.rce`, correctly rounded.

pub mod crmath;
pub mod device;
pub mod error;
pub mod interpret;
pub mod kernel;
pub mod leaf;
pub mod ml;
pub mod op_dispatch;
pub mod scalar;
pub mod tensor;
pub mod value;

pub use device::{DeviceId, DeviceMem, DeviceTensor, DownloadToHost};
pub use error::{InterpretError, Result};
pub use interpret::{
    interpret, interpret_program, interpret_reference, interpret_with_config, reference_node, reference_order,
    interpret_with_leaves, Inputs, InterpretConfig,
};
pub use leaf::LeafDispatch;
pub use op_dispatch::OpDispatch;
pub use tensor::{DType, HostTensor};
pub use value::{ScalarValue, Value};
