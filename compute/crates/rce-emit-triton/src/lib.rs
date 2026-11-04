//! rce-emit-triton — emit Triton Python DSL from RCE `FunctionGraph`.
//!
//! Produces valid Triton kernel code (`@triton.jit`-decorated Python
//! functions) that can be compiled and launched on NVIDIA/AMD GPUs via
//! the Triton compiler. Each graph param becomes a pointer + `tl.load`,
//! each Apply becomes a `tl.*` expression, and the output becomes a
//! `tl.store` with automatic masking.
//!
//! Element-wise ops are mapped directly. Block-tile patterns (matmul) and
//! reductions (softmax) return `EmitError::UnmappedOp` — they need
//! dedicated tiling codegen in a later phase.

mod emit;
mod op_map;
mod types;

pub use emit::{emit_kernel, write_triton, EmitError};
pub use op_map::{mapped_triton_count, triton_op_for, TritonOp};
pub use types::triton_type;
