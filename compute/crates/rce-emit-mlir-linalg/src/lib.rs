//! rce-emit-mlir-linalg -- emit MLIR linalg/arith/math/tensor dialect text
//! from RCE `FunctionGraph`.
//!
//! Produces valid MLIR text in the `arith`, `math`, `linalg`, and `tensor`
//! dialects, consumable by:
//! - IREE compiler (`iree-compile`)
//! - MLIR's own optimization passes (`mlir-opt`)
//! - Polygeist (C/C++ to MLIR)
//! - Any downstream MLIR-based compiler pipeline
//!
//! Key difference from StableHLO: arithmetic splits into integer (`arith.addi`)
//! and floating-point (`arith.addf`) variants. Comparisons use `arith.cmpf`/
//! `arith.cmpi` with predicate attributes. Math ops use the `math.*` dialect.

mod emit;
mod op_map;
mod types;

pub use emit::{emit_function, emit_module, write_mlir_linalg, EmitError};
pub use op_map::{linalg_op_for, mapped_linalg_count, LinalgOp};
pub use types::{mlir_element_type, mlir_scalar_tensor, is_float_type, is_int_type, is_bool_type};
