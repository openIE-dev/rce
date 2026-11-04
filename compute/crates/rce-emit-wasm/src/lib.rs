//! `rce-emit-wasm` — emit a self-contained WebAssembly module from a
//! rce `FunctionGraph`. The output is portable WAT text plus
//! optionally compiled bytes; either form runs through any Wasm
//! runtime (wasmtime, browsers, edge workers).
//!
//! ## Why this exists
//!
//! The rce-emit-wgsl path produces compute shaders for WebGPU.
//! rce-vm-backend-wasm consumes Wasm modules at runtime as leaves.
//! This crate is the missing inverse — produce Wasm directly from
//! `.fg` so AI-emitted graphs ship to the browser without going
//! through a Linalg→LLVM→Wasm intermediate. Preserves the
//! "graph-is-canonical, text-is-artifact" framing all the way down to
//! the deploy target.
//!
//! ## v0 scope
//!
//! - **Tensors**: f32, rank-2, contiguous row-major.
//! - **Ops**: `OpKind::{Add, Mul, Custom("matmul"), Custom("relu"),
//!   Custom("softmax")}`. These are the primitives the bench's MLP
//!   uses; covers the simplest non-trivial AI workload end to end.
//! - **Inputs**: rank-2 f32 tensors via `Param` nodes; shapes are
//!   provided via [`EmitOptions::param_shapes`] (no shape inference
//!   from `Param.data_type` strings yet — that's a v0.1 follow-up).
//! - **Weights**: rank-2 f32 tensors via `TensorLiteral` nodes —
//!   shape comes from the literal directly. Data is embedded into a
//!   Wasm `(data ...)` segment.
//! - **Output**: the graph's `output_node` value.
//!
//! Anything outside this scope returns [`EmitError::UnsupportedNode`]
//! or [`EmitError::UnsupportedOp`] so callers know to widen scope.

mod emit;
mod kernels;
mod shape;

pub use emit::{emit_wat, emit_wasm, EmitError, EmitOptions};
pub use shape::infer_node_shapes;
