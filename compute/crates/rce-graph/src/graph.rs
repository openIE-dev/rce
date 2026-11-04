//! RCE's graph term: the FunctionGraph (spec 6.13).
//!
//! Nodes, wires between ports, one output node and a return type. A graph
//! term of RCE's core is a FunctionGraph with its typing (`term.rs`),
//! named by the content hash of its `.fg` file. Its shapes are close to
//! inv-ai-compiler's `FunctionGraph` IR; no conversion between the two is
//! kept here.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ── Wire ────────────────────────────────────────────────────────────

/// Resource-semantic wire kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WireKind {
    Move,
    Borrow,
    MutBorrow,
    Copy,
    Channel,
    Shared,
    Weak,
    Stream,
    Error,
    Lazy,
    Feedback,
}

/// A directed edge in the function graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wire {
    pub from: usize,
    pub to: usize,
    pub port_in_idx: i32,
    pub wire_kind: WireKind,
}

// ── Literal values ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LiteralValue {
    I64(i64),
    I32(i32),
    F64(f64),
    F32(f32),
    Bool(bool),
    String(String),
}

// ── Tensor literals ─────────────────────────────────────────────────

/// Shaped, typed tensor literal. Used for model weights, biases, lookup
/// tables — any rank-≥1 constant tensor loaded from a checkpoint.
///
/// `data` holds raw bytes in row-major (C-order) layout. Length must equal
/// `shape.iter().product() * dtype_byte_size(&dtype)`. `Arc` backing allows
/// a multi-MB weight blob to be shared cheaply across compiler passes.
///
/// `data` is **not serialized**: large weight blobs persist via safetensors
/// sidecar files alongside the graph JSON. After deserialization, `data`
/// is an empty `Arc<[u8]>` until reattached by the weight loader.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorLiteralData {
    #[serde(skip, default = "empty_arc_bytes")]
    pub data: Arc<[u8]>,
    pub shape: Vec<usize>,
    /// Scalar dtype name ("f32", "f16", "bf16", "i32", "i64", "u8", "bool").
    pub dtype: String,
    /// Stable name used to match the literal against its safetensors entry
    /// after deserialization (e.g., "model.layers.0.mlp.gate_proj.weight").
    pub name: String,
}

fn empty_arc_bytes() -> Arc<[u8]> {
    Arc::from(Vec::<u8>::new().into_boxed_slice())
}

/// Size in bytes of a single element of a graph dtype name.
///
/// Returns `None` for unrecognised dtype names so callers can surface a
/// typed error instead of producing silently-wrong offsets.
pub fn dtype_byte_size(dtype: &str) -> Option<usize> {
    Some(match dtype {
        "bool" | "i8" | "u8" => 1,
        "f16" | "bf16" | "i16" | "u16" => 2,
        "f32" | "i32" | "u32" => 4,
        "f64" | "i64" | "u64" | "complex64" => 8,
        _ => return None,
    })
}

impl TensorLiteralData {
    /// Construct and validate a tensor literal. Returns `Err` if
    /// `data.len() != shape.iter().product() * dtype_byte_size(&dtype)`
    /// or if the dtype name is not recognised.
    pub fn new(
        name: impl Into<String>,
        data: Arc<[u8]>,
        shape: Vec<usize>,
        dtype: impl Into<String>,
    ) -> Result<Self, TensorLiteralError> {
        let dtype = dtype.into();
        let byte_size = dtype_byte_size(&dtype)
            .ok_or_else(|| TensorLiteralError::UnknownDtype(dtype.clone()))?;
        let expected: usize = shape.iter().product::<usize>() * byte_size;
        if data.len() != expected {
            return Err(TensorLiteralError::ByteLengthMismatch {
                expected,
                got: data.len(),
            });
        }
        Ok(Self {
            data,
            shape,
            dtype,
            name: name.into(),
        })
    }

    /// Number of elements (product of shape dims; 1 for scalar).
    pub fn numel(&self) -> usize {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorLiteralError {
    UnknownDtype(String),
    ByteLengthMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for TensorLiteralError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorLiteralError::UnknownDtype(d) => write!(f, "unknown dtype: {d}"),
            TensorLiteralError::ByteLengthMismatch { expected, got } => {
                write!(f, "tensor byte length {got} does not match expected {expected}")
            }
        }
    }
}

impl std::error::Error for TensorLiteralError {}

// ── Operation kind ──────────────────────────────────────────────────

/// The operation kinds of a graph's `Apply` nodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpKind {
    // Arithmetic
    Add, Sub, Mul, Div, Rem, Neg,
    // Comparison
    Eq, Ne, Lt, Le, Gt, Ge,
    // Logic
    And, Or, Not,
    // String
    Concat,
    // Extension
    Custom(String),
}

// ── Graph nodes ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub data_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopVarDef {
    pub name: String,
    pub data_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopSubGraph {
    pub nodes: Vec<GraphNode>,
    pub wires: Vec<Wire>,
    pub output_nodes: Vec<usize>,
}

/// A node in the function graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphNode {
    Param(Param),
    Literal { value: LiteralValue, data_type: String },
    /// A shaped, typed tensor constant — weights, biases, or any rank-≥1
    /// literal loaded from a checkpoint file.
    TensorLiteral(TensorLiteralData),
    Apply { op: OpKind, output_type: String },
    Select { output_type: String },
    WhileLoop {
        loop_vars: Vec<LoopVarDef>,
        condition: LoopSubGraph,
        body: LoopSubGraph,
        output_var: usize,
    },
    Call {
        callee: String,
        arg_types: Vec<String>,
        return_type: String,
    },
    Match {
        case_values: Vec<i64>,
        output_type: String,
    },
    Mutate { target_type: String },
    Sequence { output_idx: usize },
    CodeBlock {
        language: String,
        code_text: String,
        output_type: String,
    },
    StructCreate {
        type_name: String,
        field_names: Vec<String>,
    },
    FieldAccess {
        type_name: String,
        field_name: String,
        output_type: String,
    },
    ArrayCreate {
        element_type: String,
        length: u32,
    },
    IndexAccess { element_type: String },
    ErrorPropagate {
        error_kind: String,
        output_type: String,
    },
    Spawn { model: String, callee: Option<String> },
    JoinAwait { party_count: u32 },
    YieldSuspend,

    // ── Enum / tagged union support ────────────────────────────
    /// Declare a named type (enum, struct, alias) in the graph.
    TypeDefine {
        type_name: String,
        kind: TypeDefineKind,
        /// (field_name, field_type) for structs; (variant_name, payload_types) for enums.
        variants: Vec<(String, Vec<String>)>,
    },
    /// Construct a tagged union value: tag + optional payload.
    EnumCreate {
        type_name: String,
        variant_tag: i64,
        variant_name: String,
    },
    /// Extract the integer discriminant from a tagged union value.
    EnumTag { type_name: String },
    /// Extract the payload from a tagged union value given the expected variant tag.
    EnumPayload {
        type_name: String,
        variant_tag: i64,
        output_type: String,
    },
    /// Lookup a method in a vtable (dynamic dispatch).
    VTableLookup {
        trait_name: String,
        method_name: String,
        return_type: String,
    },

    /// Reference to a registered capability + method, resolved
    /// through a Capability Registry handle at execution time.
    /// Mirrors the graph's `OpKind::CapabilityRef` exactly so a
    /// program that calls a slot-bound function lowers straight
    /// through to the substrate without re-encoding.
    ///
    /// Created by the `capability_rewrite` pass: it walks an
    /// existing ProgramGraph and replaces `Call` / `Apply` nodes
    /// whose name matches a registered slot binding.
    CapabilityRef {
        /// Capability id (e.g., "joulesperbit.pay.ledger").
        capability_id: String,
        /// Semver string from the slot binding.
        version: String,
        /// Graph-spec slot name (carried for trace + debug).
        slot: String,
        /// Method on the capability surface (e.g., "transfer").
        method: String,
        /// Output type (preserved from the original Call/Apply).
        output_type: String,
        /// Effects declared by the slot's capability (e.g.
        /// `["regulatory:financial", "pii:read"]`). Carried per-node
        /// so frontend gates can fire on `regulatory:*` / `crypto:*`
        /// / `pii:*` without rejoining against the slot table.
        #[serde(default)]
        effects: Vec<String>,
    },
}

/// Kind of type definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeDefineKind {
    Struct,
    Enum,
    Union,
    Alias,
}

// ── Function and program graphs ─────────────────────────────────────

/// A single function as a directed acyclic graph of nodes and wires.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionGraph {
    pub name: String,
    pub nodes: Vec<GraphNode>,
    pub wires: Vec<Wire>,
    pub output_node: usize,
    pub return_type: String,
}

/// A program: multiple function graphs with a named entry point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramGraph {
    pub functions: Vec<FunctionGraph>,
    pub entry: String,
    /// Per-function effect-allow declarations extracted from
    /// frontend source. Keyed by function name → list of effect
    /// strings the function is permitted to use.
    ///
    /// Populated by `frontend_attrs::extract_allows_from_source`,
    /// which scans the original source string for `# @allows:` /
    /// `// @allows:` comments immediately above a `fn` definition.
    /// `cap check` reads this in addition to its `--allows fn=effect`
    /// flag — explicit flag wins on conflict.
    ///
    /// `#[serde(default)]` keeps existing program-graph JSON
    /// (without the field) backward-compatible.
    #[serde(default)]
    pub allows: std::collections::BTreeMap<String, Vec<String>>,
}

// ── Structural validation ───────────────────────────────────────────

/// Why a graph failed structural validation.
///
/// These are the structural faults that would otherwise make the
/// tree-walking interpreter panic (out-of-bounds node indexing, or an
/// `input_at_port(..).unwrap()` on a port a node requires but that no wire
/// connects). A graph loaded from untrusted JSON should be run through
/// [`FunctionGraph::validate`] / [`ProgramGraph::validate`] (or executed via
/// `interpret_checked`) so these surface as typed errors rather than panics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// `output_node` (or a sub-graph output) indexes past the node list.
    OutputNodeOutOfBounds { output_node: usize, node_count: usize },
    /// A wire endpoint references a node index that does not exist.
    WireEndpointOutOfBounds { from: usize, to: usize, node_count: usize },
    /// A node requires an input wire at `port` but none is connected.
    MissingRequiredInput { node: usize, port: i32 },
    /// A nested loop sub-graph (`"condition"` or `"body"`) failed validation.
    SubGraph { node: usize, which: &'static str, source: Box<GraphError> },
    /// A `ProgramGraph` names an entry function that does not exist.
    EntryNotFound { entry: String },
    /// A function within a `ProgramGraph` failed validation.
    FunctionInvalid { name: String, source: Box<GraphError> },
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::OutputNodeOutOfBounds { output_node, node_count } => write!(
                f, "output_node {output_node} out of bounds (node_count={node_count})"
            ),
            GraphError::WireEndpointOutOfBounds { from, to, node_count } => write!(
                f, "wire {from}->{to} endpoint out of bounds (node_count={node_count})"
            ),
            GraphError::MissingRequiredInput { node, port } => write!(
                f, "node {node} requires an input at port {port} but none is connected"
            ),
            GraphError::SubGraph { node, which, source } => write!(
                f, "node {node} {which} sub-graph: {source}"
            ),
            GraphError::EntryNotFound { entry } => write!(f, "entry function {entry:?} not found"),
            GraphError::FunctionInvalid { name, source } => write!(f, "function {name:?}: {source}"),
        }
    }
}

impl std::error::Error for GraphError {}

/// Input ports a node *requires* to be connected — i.e. the ports the
/// interpreter reads with `input_at_port(..).unwrap()`. Nodes that consume
/// their inputs via `inputs_sorted` or guarded `if let Some(..)` are tolerant
/// of missing ports and impose no requirement here.
fn required_input_ports(node: &GraphNode) -> Vec<i32> {
    match node {
        // cond, then, else
        GraphNode::Select { .. } => vec![0, 1, 2],
        // discriminant (port 0) + one arm per case value (ports 1..=len);
        // any case may match at runtime, so every arm port must exist.
        GraphNode::Match { case_values, .. } => (0..=case_values.len() as i32).collect(),
        // array, index
        GraphNode::IndexAccess { .. } => vec![0, 1],
        // object
        GraphNode::FieldAccess { .. } => vec![0],
        // tagged value
        GraphNode::EnumTag { .. } | GraphNode::EnumPayload { .. } => vec![0],
        // the recoverable path unwraps its ok-value at port 1
        GraphNode::ErrorPropagate { error_kind, .. } if error_kind == "recoverable" => vec![1],
        _ => vec![],
    }
}

/// Validate a node list + wires + outputs against the interpreter's
/// structural preconditions. Shared by `FunctionGraph` and loop sub-graphs.
fn validate_node_list(
    nodes: &[GraphNode],
    wires: &[Wire],
    outputs: &[usize],
) -> Result<(), GraphError> {
    let n = nodes.len();
    for w in wires {
        if w.from >= n || w.to >= n {
            return Err(GraphError::WireEndpointOutOfBounds {
                from: w.from,
                to: w.to,
                node_count: n,
            });
        }
    }
    for &o in outputs {
        if o >= n {
            return Err(GraphError::OutputNodeOutOfBounds { output_node: o, node_count: n });
        }
    }
    for (i, node) in nodes.iter().enumerate() {
        for port in required_input_ports(node) {
            let present = wires.iter().any(|w| w.to == i && w.port_in_idx == port);
            if !present {
                return Err(GraphError::MissingRequiredInput { node: i, port });
            }
        }
        if let GraphNode::WhileLoop { condition, body, .. } = node {
            validate_node_list(&condition.nodes, &condition.wires, &condition.output_nodes)
                .map_err(|e| GraphError::SubGraph {
                    node: i,
                    which: "condition",
                    source: Box::new(e),
                })?;
            validate_node_list(&body.nodes, &body.wires, &body.output_nodes).map_err(|e| {
                GraphError::SubGraph { node: i, which: "body", source: Box::new(e) }
            })?;
        }
    }
    Ok(())
}

impl FunctionGraph {
    /// Check that the graph is structurally well-formed enough that the
    /// interpreter cannot panic on it: every wire endpoint and `output_node`
    /// is in bounds, every node's required input ports are connected, and the
    /// same holds recursively for loop sub-graphs. Run this at any boundary
    /// where a graph arrives from untrusted JSON.
    pub fn validate(&self) -> Result<(), GraphError> {
        validate_node_list(&self.nodes, &self.wires, std::slice::from_ref(&self.output_node))
    }
}

impl ProgramGraph {
    /// Validate every function and confirm the named `entry` exists.
    pub fn validate(&self) -> Result<(), GraphError> {
        if !self.functions.iter().any(|f| f.name == self.entry) {
            return Err(GraphError::EntryNotFound { entry: self.entry.clone() });
        }
        for f in &self.functions {
            f.validate().map_err(|e| GraphError::FunctionInvalid {
                name: f.name.clone(),
                source: Box::new(e),
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod validate_tests {
    use super::*;

    fn lit_i64(v: i64) -> GraphNode {
        GraphNode::Literal { value: LiteralValue::I64(v), data_type: "i64".into() }
    }
    fn wire(from: usize, to: usize, port: i32) -> Wire {
        Wire { from, to, port_in_idx: port, wire_kind: WireKind::Move }
    }

    #[test]
    fn valid_add_graph_passes() {
        let fg = FunctionGraph {
            name: "f".into(),
            nodes: vec![
                lit_i64(1),
                lit_i64(2),
                GraphNode::Apply { op: OpKind::Add, output_type: "i64".into() },
            ],
            wires: vec![wire(0, 2, 0), wire(1, 2, 1)],
            output_node: 2,
            return_type: "i64".into(),
        };
        assert_eq!(fg.validate(), Ok(()));
    }

    #[test]
    fn out_of_bounds_output_node_rejected() {
        let fg = FunctionGraph {
            name: "f".into(),
            nodes: vec![lit_i64(1)],
            wires: vec![],
            output_node: 5,
            return_type: "i64".into(),
        };
        assert_eq!(
            fg.validate(),
            Err(GraphError::OutputNodeOutOfBounds { output_node: 5, node_count: 1 })
        );
    }

    #[test]
    fn dangling_wire_endpoint_rejected() {
        let fg = FunctionGraph {
            name: "f".into(),
            nodes: vec![lit_i64(1)],
            wires: vec![wire(0, 9, 0)],
            output_node: 0,
            return_type: "i64".into(),
        };
        assert_eq!(
            fg.validate(),
            Err(GraphError::WireEndpointOutOfBounds { from: 0, to: 9, node_count: 1 })
        );
    }

    #[test]
    fn select_missing_else_port_rejected() {
        // Select reads ports 0,1,2 with .unwrap(); omitting port 2 would panic
        // the interpreter. validate() must catch it first.
        let fg = FunctionGraph {
            name: "f".into(),
            nodes: vec![
                GraphNode::Literal { value: LiteralValue::Bool(true), data_type: "bool".into() },
                lit_i64(1),
                GraphNode::Select { output_type: "i64".into() },
            ],
            wires: vec![wire(0, 2, 0), wire(1, 2, 1)], // missing port 2
            output_node: 2,
            return_type: "i64".into(),
        };
        assert_eq!(
            fg.validate(),
            Err(GraphError::MissingRequiredInput { node: 2, port: 2 })
        );
    }

    #[test]
    fn recoverable_error_propagate_requires_ok_port() {
        // The recoverable path unwraps port 1; a missing port 1 would panic.
        let fg = FunctionGraph {
            name: "f".into(),
            nodes: vec![
                lit_i64(7),
                GraphNode::ErrorPropagate {
                    error_kind: "recoverable".into(),
                    output_type: "i64".into(),
                },
            ],
            wires: vec![wire(0, 1, 0)], // port 0 present, port 1 missing
            output_node: 1,
            return_type: "i64".into(),
        };
        assert_eq!(
            fg.validate(),
            Err(GraphError::MissingRequiredInput { node: 1, port: 1 })
        );
    }
}
