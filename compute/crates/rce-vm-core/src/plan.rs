use crate::backend::BackendId;
use thiserror::Error;

/// A compiled, backend-specific execution plan for a sub-graph slice.
///
/// `Plan` is opaque to the scheduler — only the backend that produced it
/// can dispatch it. `placements` is a publicly-readable map from each
/// graph node ID in the slice to the chosen backend, used by the trace
/// bus and the cost re-estimator.
#[derive(Debug, Clone)]
pub struct Plan {
    pub backend: BackendId,
    /// Backend-specific bytes (e.g. compiled SPIR-V, PTX, Wasm bytecode).
    pub artifact: Vec<u8>,
    pub placements: Vec<Placement>,
    pub determinism_strict: bool,
}

#[derive(Debug, Clone)]
pub struct Placement {
    pub node_id: u32,
    pub backend: BackendId,
}

#[derive(Debug, Error)]
pub enum PlanError {
    #[error("graph slice too large for backend `{0:?}`")]
    SliceTooLarge(BackendId),

    #[error("compile error: {0}")]
    Compile(String),

    #[error("required capability missing: {0}")]
    MissingCapability(String),
}
