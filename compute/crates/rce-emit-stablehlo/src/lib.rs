//! rce-emit-stablehlo — emit StableHLO MLIR text from RCE `FunctionGraph`.
//!
//! Produces valid MLIR text in the StableHLO dialect, consumable by:
//! - `stablehlo-opt` (StableHLO's own optimizer)
//! - `iree-compile` (IREE compiler — CPU/GPU/NPU codegen)
//! - `xla-opt` (XLA's MLIR frontend)
//! - Mojo/MAX Engine (Modular's runtime)
//! - Any MLIR-based compiler pipeline
//!
//! 65 ops mapped (16 core + 49 custom). Unmapped ops (HDC, FHE,
//! neuromorphic, attractors) return `EmitError::UnmappedOp`.

mod emit;
mod op_map;
mod types;

pub use emit::{emit_function, emit_module, write_stablehlo, EmitError};
pub use op_map::{hlo_op_for, mapped_hlo_count, HloOp};
pub use types::{mlir_element_type, mlir_scalar_tensor};
