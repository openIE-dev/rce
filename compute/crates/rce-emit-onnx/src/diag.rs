//! Diagnostics for unmapped ops and emission errors.
//!
//! The goal is legibility: every graph op that cannot be emitted to ONNX
//! should report *why* — typically because the op belongs to a long-tail
//! domain (intelligence attractors, HDC, FHE, neuromorphic) that ONNX has
//! no notion of.

use rce_graph::graph::OpKind;

/// Top-level emission error.
#[derive(Debug, thiserror::Error)]
pub enum EmitError {
    #[error("unmapped op: {0}")]
    Unmapped(#[from] UnmappedOp),
    #[error("unsupported node kind: {0}")]
    UnsupportedNode(String),
    #[error("unsupported literal type for ONNX initializer: {0}")]
    UnsupportedLiteral(String),
    #[error("unsupported data type: {0}")]
    UnsupportedType(String),
    #[error("graph has no output node")]
    NoOutput,
    #[error("wire references node index {0} out of range")]
    BadWire(usize),
    #[error("node {node} missing input on port {port}")]
    MissingInput { node: usize, port: i32 },
}

/// A graph op had no ONNX mapping. Carries enough context for the
/// diagnostic to point at a concrete gap in the op-map table.
#[derive(Debug, thiserror::Error)]
#[error("graph op {op_name:?} has no ONNX mapping (domain: {domain}, reason: {reason})")]
pub struct UnmappedOp {
    pub op_name: String,
    pub domain: &'static str,
    pub reason: &'static str,
}

impl UnmappedOp {
    pub fn from_opkind(op: &OpKind) -> Self {
        let op_name = format!("{op:?}");
        let (domain, reason) = classify(op);
        Self { op_name, domain, reason }
    }
}

fn classify(op: &OpKind) -> (&'static str, &'static str) {
    match op {
        OpKind::Custom(name) => (
            "custom",
            if name.starts_with("hdc.") { "HDC ops are not representable in ONNX — stays a graph op" }
            else if name.starts_with("fhe.") { "FHE ops are not representable in ONNX — stays a graph op" }
            else if name.starts_with("neuro.") { "neuromorphic ops are not representable in ONNX — stays a graph op" }
            else if name.starts_with("attractor.") { "intelligence attractors are RCE-native — stays a graph op" }
            else { "no mapping registered for this custom op" },
        ),
        _ => ("core", "op is a core graph op but not yet wired into op_map.rs"),
    }
}
