//! rce-ingest-fx: Ingest PyTorch FX graph JSON into RCE's FunctionGraph.
//!
//! PyTorch's `torch.fx` framework traces Python models into a graph IR.
//! When serialized to JSON (via `torch.export` or custom serialization),
//! it produces a node list with op names, args, and kwargs. This crate
//! parses that JSON and converts to RCE `FunctionGraph`.

pub mod op_map;
pub mod parse;
pub mod to_rce;

pub use parse::{parse_fx_json, FxArg, FxGraph, FxNode, FxOp};
pub use to_rce::to_function_graph;

/// Errors that can occur during FX graph ingestion.
#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("missing field: {0}")]
    MissingField(String),

    #[error("unknown FX op type: {0}")]
    UnknownOp(String),

    #[error("invalid argument: {0}")]
    InvalidArg(String),

    #[error("node reference not found: {0}")]
    NodeNotFound(String),

    #[error("no output node in FX graph")]
    NoOutput,
}

/// Convenience: parse JSON string and convert to `FunctionGraph` in one call.
pub fn ingest(json: &str) -> Result<rce_graph::graph::FunctionGraph, IngestError> {
    let fx = parse_fx_json(&serde_json::from_str::<serde_json::Value>(json)?)?;
    to_function_graph(&fx)
}

#[cfg(test)]
mod tests;
