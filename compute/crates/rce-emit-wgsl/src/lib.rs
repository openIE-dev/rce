//! rce-emit-wgsl — emit WGSL compute shaders from RCE `FunctionGraph`.
//!
//! Produces valid WGSL text for WebGPU compute shaders, consumable by:
//! - `wgpu` (Rust WebGPU implementation)
//! - Chrome/Firefox/Safari WebGPU runtime
//! - Dawn (Google's native WebGPU implementation)
//! - Any WGSL-compatible shader compiler
//!
//! Each `FunctionGraph` becomes an element-wise compute kernel where:
//! - Params become `@group(0) @binding(N)` storage buffers
//! - Apply nodes become WGSL expressions
//! - Output goes to a final storage buffer binding

mod emit;
mod op_map;
mod types;

pub use emit::{emit_kernel, emit_module, write_wgsl, EmitError};
pub use op_map::{wgsl_op_for, mapped_wgsl_count, WgslOp};
pub use types::wgsl_type;
