//! Parse PyTorch FX graph JSON into an intermediate representation.

use serde_json::Value;

use crate::IngestError;

// ── Intermediate types ─────────────────────────────────────────────

/// The top-level FX graph.
#[derive(Debug, Clone)]
pub struct FxGraph {
    pub nodes: Vec<FxNode>,
}

/// A single node in the FX graph.
#[derive(Debug, Clone)]
pub struct FxNode {
    pub op: FxOp,
    pub name: String,
    pub target: String,
    pub args: Vec<FxArg>,
    /// Optional shape (rank-N integer dims). When present on a `get_attr`
    /// node, the lowering emits a `TensorLiteral` (with empty data,
    /// reattached at load time from a sidecar) instead of a `Param`.
    /// Other op types ignore this field.
    pub shape: Option<Vec<usize>>,
    /// Optional dtype string ("f32", "f16", "bf16", "i32", …). Same role
    /// as `shape`: only consumed for `get_attr`.
    pub dtype: Option<String>,
}

/// FX node operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FxOp {
    Placeholder,
    CallFunction,
    CallMethod,
    CallModule,
    GetAttr,
    Output,
}

/// An argument to an FX node.
#[derive(Debug, Clone)]
pub enum FxArg {
    /// Reference to another node by name.
    Name(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    List(Vec<FxArg>),
    None,
}

// ── Parsing ────────────────────────────────────────────────────────

/// Parse an FX graph from a `serde_json::Value`.
///
/// Expected shape: `{ "graph": { "nodes": [ ... ] } }`
pub fn parse_fx_json(value: &Value) -> Result<FxGraph, IngestError> {
    let graph_obj = value
        .get("graph")
        .ok_or_else(|| IngestError::MissingField("graph".into()))?;

    let nodes_arr = graph_obj
        .get("nodes")
        .and_then(|v| v.as_array())
        .ok_or_else(|| IngestError::MissingField("graph.nodes".into()))?;

    let mut nodes = Vec::with_capacity(nodes_arr.len());
    for node_val in nodes_arr {
        nodes.push(parse_node(node_val)?);
    }

    Ok(FxGraph { nodes })
}

fn parse_node(value: &Value) -> Result<FxNode, IngestError> {
    let op_str = value
        .get("op")
        .and_then(|v| v.as_str())
        .ok_or_else(|| IngestError::MissingField("node.op".into()))?;

    let op = parse_op(op_str)?;

    let name = value
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let target = value
        .get("target")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let args = match value.get("args") {
        Some(Value::Array(arr)) => arr.iter().map(parse_arg).collect::<Result<Vec<_>, _>>()?,
        _ => Vec::new(),
    };

    let shape = value.get("shape").and_then(|v| v.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|x| x.as_u64().map(|u| u as usize))
            .collect()
    });
    let dtype = value.get("dtype").and_then(|v| v.as_str()).map(String::from);

    Ok(FxNode {
        op,
        name,
        target,
        args,
        shape,
        dtype,
    })
}

fn parse_op(s: &str) -> Result<FxOp, IngestError> {
    match s {
        "placeholder" => Ok(FxOp::Placeholder),
        "call_function" => Ok(FxOp::CallFunction),
        "call_method" => Ok(FxOp::CallMethod),
        "call_module" => Ok(FxOp::CallModule),
        "get_attr" => Ok(FxOp::GetAttr),
        "output" => Ok(FxOp::Output),
        other => Err(IngestError::UnknownOp(other.into())),
    }
}

fn parse_arg(value: &Value) -> Result<FxArg, IngestError> {
    match value {
        Value::String(s) => Ok(FxArg::Name(s.clone())),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(FxArg::Int(i))
            } else if let Some(f) = n.as_f64() {
                Ok(FxArg::Float(f))
            } else {
                Err(IngestError::InvalidArg(format!("unparseable number: {n}")))
            }
        }
        Value::Bool(b) => Ok(FxArg::Bool(*b)),
        Value::Null => Ok(FxArg::None),
        Value::Array(arr) => {
            let items = arr.iter().map(parse_arg).collect::<Result<Vec<_>, _>>()?;
            Ok(FxArg::List(items))
        }
        other => Err(IngestError::InvalidArg(format!(
            "unsupported arg type: {other}"
        ))),
    }
}
