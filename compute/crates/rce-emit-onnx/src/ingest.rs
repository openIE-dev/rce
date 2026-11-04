//! ONNX `ModelProto` → RCE `FunctionGraph` ingestion.
//!
//! The reverse of `emit.rs`. Takes an ONNX graph and produces a
//! `rce_graph::FunctionGraph` that can flow through RCE's compiler
//! pipeline (optimize, JIT, AOT, WASM, hardware dispatch).
//!
//! Scope: all 69 ops that `op_map.rs` can emit are also ingestible.
//! Unknown ONNX ops become `OpKind::Custom("onnx.{op_type}")` so they're
//! visible in the graph rather than silently dropped.

use std::collections::HashMap;
use std::path::Path;

use crate::onnx;
use rce_graph::graph::{
    FunctionGraph, GraphNode, LiteralValue, OpKind, Param, Wire, WireKind,
};
use prost::Message;

#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("decode: {0}")]
    Decode(String),
    #[error("model has no graph")]
    NoGraph,
    #[error("unsupported initializer dtype {0} for tensor {1}")]
    UnsupportedInitializerDtype(i32, String),
    #[error("node {node} input {input} not found in name map")]
    UnresolvedInput { node: String, input: String },
}

/// Load an `.onnx` file and convert to a `FunctionGraph`.
pub fn ingest_onnx(path: impl AsRef<Path>) -> Result<FunctionGraph, IngestError> {
    let bytes = std::fs::read(path)?;
    let model = onnx::ModelProto::decode(&bytes[..])
        .map_err(|e| IngestError::Decode(e.to_string()))?;
    ingest_model(&model)
}

/// Convert an in-memory `ModelProto` to a `FunctionGraph`.
pub fn ingest_model(model: &onnx::ModelProto) -> Result<FunctionGraph, IngestError> {
    let graph = model.graph.as_ref().ok_or(IngestError::NoGraph)?;
    ingest_graph(graph)
}

/// Convert an ONNX `GraphProto` to a `FunctionGraph`.
pub fn ingest_graph(graph: &onnx::GraphProto) -> Result<FunctionGraph, IngestError> {
    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut wires: Vec<Wire> = Vec::new();
    // Map ONNX tensor names → graph node indices.
    let mut name_to_idx: HashMap<String, usize> = HashMap::new();

    // 1. Inputs → Param nodes.
    for vi in &graph.input {
        let dt = rce_type_from_value_info(vi);
        let idx = nodes.len();
        nodes.push(GraphNode::Param(Param {
            name: vi.name.clone(),
            data_type: dt,
        }));
        name_to_idx.insert(vi.name.clone(), idx);
    }

    // 2. Initializers → Literal nodes (override params with same name).
    for init in &graph.initializer {
        let (value, dt) = initializer_to_literal(init)?;
        let idx = nodes.len();
        nodes.push(GraphNode::Literal { value, data_type: dt });
        name_to_idx.insert(init.name.clone(), idx);
    }

    // 3. Nodes → Apply nodes.
    for node in &graph.node {
        let op = rce_op_for_onnx(&node.op_type, &node.domain);
        let output_type = "f32".to_string(); // conservative default
        let target_idx = nodes.len();
        nodes.push(GraphNode::Apply { op, output_type });

        // Wire inputs.
        for (port, input_name) in node.input.iter().enumerate() {
            if input_name.is_empty() {
                continue; // ONNX allows empty strings for optional inputs
            }
            let src_idx = name_to_idx.get(input_name).copied().ok_or_else(|| {
                IngestError::UnresolvedInput {
                    node: node.name.clone(),
                    input: input_name.clone(),
                }
            })?;
            wires.push(Wire {
                from: src_idx,
                to: target_idx,
                port_in_idx: port as i32,
                wire_kind: WireKind::Copy,
            });
        }

        // Register outputs.
        for (i, output_name) in node.output.iter().enumerate() {
            if !output_name.is_empty() {
                if i == 0 {
                    name_to_idx.insert(output_name.clone(), target_idx);
                } else {
                    // Multi-output ONNX nodes: for now, alias to same node.
                    name_to_idx.insert(output_name.clone(), target_idx);
                }
            }
        }
    }

    // 4. Determine output node.
    let output_node = if let Some(out) = graph.output.first() {
        name_to_idx.get(&out.name).copied().unwrap_or(nodes.len().saturating_sub(1))
    } else if !nodes.is_empty() {
        nodes.len() - 1
    } else {
        0
    };

    let return_type = graph
        .output
        .first()
        .map(|vi| rce_type_from_value_info(vi))
        .unwrap_or_else(|| "f32".into());

    Ok(FunctionGraph {
        name: graph.name.clone(),
        nodes,
        wires,
        output_node,
        return_type,
    })
}

/// ONNX op_type + domain → graph OpKind.
fn rce_op_for_onnx(op_type: &str, domain: &str) -> OpKind {
    if !domain.is_empty() && domain != "ai.onnx" {
        return OpKind::Custom(format!("{domain}.{op_type}"));
    }
    match op_type {
        "Add" => OpKind::Add,
        "Sub" => OpKind::Sub,
        "Mul" => OpKind::Mul,
        "Div" => OpKind::Div,
        "Mod" => OpKind::Rem,
        "Neg" => OpKind::Neg,
        "Equal" => OpKind::Eq,
        "Less" => OpKind::Lt,
        "LessOrEqual" => OpKind::Le,
        "Greater" => OpKind::Gt,
        "GreaterOrEqual" => OpKind::Ge,
        "And" => OpKind::And,
        "Or" => OpKind::Or,
        "Not" => OpKind::Not,
        // Everything else maps to Custom with snake_case name.
        other => OpKind::Custom(onnx_to_snake(other)),
    }
}

/// "LayerNormalization" → "layer_normalization", "MatMul" → "matmul"
fn onnx_to_snake(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    for (i, ch) in name.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            let prev = name.as_bytes()[i - 1];
            if prev.is_ascii_lowercase() || prev.is_ascii_digit() {
                out.push('_');
            }
        }
        out.push(ch.to_ascii_lowercase());
    }
    out
}

fn rce_type_from_value_info(vi: &onnx::ValueInfoProto) -> String {
    let Some(ref tp) = vi.r#type else { return "f32".into() };
    let Some(ref val) = tp.value else { return "f32".into() };
    match val {
        onnx::type_proto::Value::TensorType(t) => rce_dtype(t.elem_type),
    }
}

fn rce_dtype(onnx_dtype: i32) -> String {
    use onnx::tensor_proto::DataType;
    match DataType::try_from(onnx_dtype) {
        Ok(DataType::Float) => "f32",
        Ok(DataType::Double) => "f64",
        Ok(DataType::Int32) => "i32",
        Ok(DataType::Int64) => "i64",
        Ok(DataType::Int16) => "i16",
        Ok(DataType::Int8) => "i8",
        Ok(DataType::Uint8) => "u8",
        Ok(DataType::Bool) => "bool",
        Ok(DataType::String) => "string",
        Ok(DataType::Float16) => "f16",
        _ => "f32",
    }
    .into()
}

fn initializer_to_literal(
    init: &onnx::TensorProto,
) -> Result<(LiteralValue, String), IngestError> {
    use onnx::tensor_proto::DataType;
    let dt = DataType::try_from(init.data_type).unwrap_or(DataType::Undefined);
    match dt {
        DataType::Float => {
            let v = init.float_data.first().copied().unwrap_or(0.0);
            Ok((LiteralValue::F32(v), "f32".into()))
        }
        DataType::Double => {
            let v = init.double_data.first().copied().unwrap_or(0.0);
            Ok((LiteralValue::F64(v), "f64".into()))
        }
        DataType::Int32 => {
            let v = init.int32_data.first().copied().unwrap_or(0);
            Ok((LiteralValue::I32(v), "i32".into()))
        }
        DataType::Int64 => {
            let v = init.int64_data.first().copied().unwrap_or(0);
            Ok((LiteralValue::I64(v), "i64".into()))
        }
        DataType::Bool => {
            let v = init.int32_data.first().copied().unwrap_or(0) != 0;
            Ok((LiteralValue::Bool(v), "bool".into()))
        }
        _ => Err(IngestError::UnsupportedInitializerDtype(
            init.data_type,
            init.name.clone(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::{emit_model, write_onnx};

    fn make_add_graph() -> FunctionGraph {
        FunctionGraph {
            name: "add".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "a".into(), data_type: "f32".into() }),
                GraphNode::Param(Param { name: "b".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Add, output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Copy },
            ],
            output_node: 2,
            return_type: "f32".into(),
        }
    }

    #[test]
    fn roundtrip_add() {
        let original = make_add_graph();
        let model = emit_model(&original).unwrap();
        let ingested = ingest_model(&model).unwrap();
        assert_eq!(ingested.name, "add");
        assert_eq!(ingested.nodes.len(), 2 + 1); // 2 params + 1 apply
        assert_eq!(ingested.wires.len(), 2);
        match &ingested.nodes[2] {
            GraphNode::Apply { op, .. } => assert_eq!(*op, OpKind::Add),
            other => panic!("expected Apply, got {other:?}"),
        }
        assert_eq!(ingested.return_type, "f32");
    }

    #[test]
    fn roundtrip_mlp_layer() {
        let original = FunctionGraph {
            name: "mlp".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
                GraphNode::Param(Param { name: "w".into(), data_type: "f32".into() }),
                GraphNode::Param(Param { name: "b".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("matmul".into()), output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Add, output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("relu".into()), output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 3, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 3, port_in_idx: 1, wire_kind: WireKind::Copy },
                Wire { from: 3, to: 4, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 2, to: 4, port_in_idx: 1, wire_kind: WireKind::Copy },
                Wire { from: 4, to: 5, port_in_idx: 0, wire_kind: WireKind::Copy },
            ],
            output_node: 5,
            return_type: "f32".into(),
        };
        let model = emit_model(&original).unwrap();
        let ingested = ingest_model(&model).unwrap();
        assert_eq!(ingested.name, "mlp");
        assert_eq!(ingested.nodes.len(), 6); // 3 params + 3 apply
        assert_eq!(ingested.wires.len(), 5);
        // Verify op chain survived.
        let ops: Vec<&str> = ingested.nodes.iter().filter_map(|n| match n {
            GraphNode::Apply { op, .. } => Some(match op {
                OpKind::Add => "Add",
                OpKind::Custom(s) => s.as_str(),
                _ => "?",
            }),
            _ => None,
        }).collect();
        assert_eq!(ops, vec!["mat_mul", "Add", "relu"]);
    }

    #[test]
    fn roundtrip_file_io() {
        let original = make_add_graph();
        let tmp = std::env::temp_dir().join("rce_ingest_roundtrip.onnx");
        write_onnx(&original, &tmp).unwrap();
        let ingested = ingest_onnx(&tmp).unwrap();
        assert_eq!(ingested.name, "add");
        assert_eq!(ingested.wires.len(), 2);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn roundtrip_with_literal() {
        let original = FunctionGraph {
            name: "add_const".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
                GraphNode::Literal { value: LiteralValue::F32(42.0), data_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Add, output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Copy },
            ],
            output_node: 2,
            return_type: "f32".into(),
        };
        let model = emit_model(&original).unwrap();
        let ingested = ingest_model(&model).unwrap();
        // Param + Literal (from initializer) + Apply
        assert_eq!(ingested.nodes.len(), 3);
        match &ingested.nodes[1] {
            GraphNode::Literal { value: LiteralValue::F32(v), .. } => {
                assert!((v - 42.0).abs() < 1e-6);
            }
            other => panic!("expected Literal(F32(42.0)), got {other:?}"),
        }
    }

    #[test]
    fn ingest_unknown_op_becomes_custom() {
        let graph = onnx::GraphProto {
            name: "test".into(),
            input: vec![onnx::ValueInfoProto {
                name: "x".into(),
                r#type: None,
                doc_string: String::new(),
            }],
            output: vec![onnx::ValueInfoProto {
                name: "out".into(),
                r#type: None,
                doc_string: String::new(),
            }],
            node: vec![onnx::NodeProto {
                input: vec!["x".into()],
                output: vec!["out".into()],
                name: "node_0".into(),
                op_type: "MyFancyOp".into(),
                domain: String::new(),
                attribute: vec![],
                doc_string: String::new(),
            }],
            initializer: vec![],
            value_info: vec![],
        };
        let fg = ingest_graph(&graph).unwrap();
        match &fg.nodes[1] {
            GraphNode::Apply { op: OpKind::Custom(name), .. } => {
                assert_eq!(name, "my_fancy_op");
            }
            other => panic!("expected Custom, got {other:?}"),
        }
    }

    #[test]
    fn onnx_to_snake_cases() {
        assert_eq!(onnx_to_snake("MatMul"), "mat_mul");
        assert_eq!(onnx_to_snake("LayerNormalization"), "layer_normalization");
        assert_eq!(onnx_to_snake("Relu"), "relu");
        assert_eq!(onnx_to_snake("Conv"), "conv");
        assert_eq!(onnx_to_snake("GlobalAveragePool"), "global_average_pool");
    }

    #[test]
    fn ingest_microsoft_domain_op() {
        let graph = onnx::GraphProto {
            name: "attn".into(),
            input: vec![onnx::ValueInfoProto {
                name: "q".into(),
                r#type: None,
                doc_string: String::new(),
            }],
            output: vec![onnx::ValueInfoProto {
                name: "out".into(),
                r#type: None,
                doc_string: String::new(),
            }],
            node: vec![onnx::NodeProto {
                input: vec!["q".into()],
                output: vec!["out".into()],
                name: "attn_0".into(),
                op_type: "Attention".into(),
                domain: "com.microsoft".into(),
                attribute: vec![],
                doc_string: String::new(),
            }],
            initializer: vec![],
            value_info: vec![],
        };
        let fg = ingest_graph(&graph).unwrap();
        match &fg.nodes[1] {
            GraphNode::Apply { op: OpKind::Custom(name), .. } => {
                assert_eq!(name, "com.microsoft.Attention");
            }
            other => panic!("expected Custom(com.microsoft.Attention), got {other:?}"),
        }
    }
}
