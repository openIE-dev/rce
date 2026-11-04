//! rce-emit-onnx — emit ONNX ModelProto from a RCE `FunctionGraph`.
//!
//! This is the tracer-bullet direction of a larger ONNX ingest+emit bridge.
//! It takes a `rce_graph::FunctionGraph` and produces a binary-serializable
//! ONNX `ModelProto` that can be loaded by any ONNX runtime (ORT, tract,
//! TensorRT, CoreML, OpenVINO, QNN, Hailo).
//!
//! # Scope (phase 0 tracer)
//!
//! One working op end-to-end (Add), plus the infrastructure for the full
//! ~40-op mapping table. Unmapped ops return a structured `EmitError` with
//! the TensorCell coordinate — so the gap is legible, not silent.
//!
//! # Usage
//!
//! ```ignore
//! use rce_emit_onnx::emit_model;
//! let model = emit_model(&graph)?;
//! let bytes = model.encode_to_vec();   // prost::Message
//! std::fs::write("graph.onnx", bytes)?;
//! ```

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

mod diag;
mod emit;
mod ingest;
mod op_map;

pub use diag::{EmitError, UnmappedOp};
pub use emit::{emit_function, emit_model, read_onnx, write_onnx, WriteError};
pub use ingest::{ingest_graph, ingest_model, ingest_onnx, IngestError};
pub use op_map::{mapped_op_count, onnx_op_for, OnnxOp};
