//! Emit ONNX `ModelProto` from `rce_graph::FunctionGraph`.
//!
//! The walk is straightforward for the phase-0 tracer scope:
//!   - `GraphNode::Param` → `GraphProto.input` (ValueInfoProto)
//!   - `GraphNode::Literal` → `GraphProto.initializer` (TensorProto, scalar)
//!   - `GraphNode::Apply { op }` → `GraphProto.node` (NodeProto)
//!   - `graph.output_node` → `GraphProto.output` (ValueInfoProto)
//!
//! Wires carry port information; we reconstruct each `NodeProto`'s input
//! list by collecting incoming wires sorted by `port_in_idx`.
//!
//! Control-flow nodes (WhileLoop, Select, Match, Call, Sequence, Mutate,
//! Spawn, etc.) are *not* in scope for the tracer — they return
//! `EmitError::UnsupportedNode`. ONNX has `Loop`, `If`, and `Scan`
//! sub-graphs that cover this, but that's a later phase.

use std::io;
use std::path::Path;

use crate::diag::EmitError;
use crate::onnx;
use crate::op_map::{onnx_dtype_for, onnx_op_for};
use rce_graph::graph::{FunctionGraph, GraphNode, LiteralValue};
use prost::Message;

/// ONNX IR version 10 (released with ONNX 1.16+, corresponds to opset 20+).
const ONNX_IR_VERSION: i64 = 10;
/// Opset version for the default domain "". Opset 20 is the baseline we target.
const ONNX_DEFAULT_OPSET: i64 = 20;

/// Emit a complete `ModelProto` wrapping one function graph.
pub fn emit_model(graph: &FunctionGraph) -> Result<onnx::ModelProto, EmitError> {
    let inner = emit_function(graph)?;
    Ok(onnx::ModelProto {
        ir_version: ONNX_IR_VERSION,
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: String::new(),
            version: ONNX_DEFAULT_OPSET,
        }],
        producer_name: "rce-emit-onnx".into(),
        producer_version: env!("CARGO_PKG_VERSION").into(),
        domain: "ai.openie.rce".into(),
        model_version: 1,
        doc_string: format!("RCE graph `{}` emitted via rce-emit-onnx", graph.name),
        graph: Some(inner),
    })
}

/// Emit just the `GraphProto` for a single function (no model wrapper).
pub fn emit_function(graph: &FunctionGraph) -> Result<onnx::GraphProto, EmitError> {
    if graph.nodes.is_empty() {
        return Err(EmitError::NoOutput);
    }

    let mut inputs: Vec<onnx::ValueInfoProto> = Vec::new();
    let mut initializers: Vec<onnx::TensorProto> = Vec::new();
    let mut nodes: Vec<onnx::NodeProto> = Vec::new();

    // Name each graph node index uniquely for ONNX tensor-naming.
    let node_name = |idx: usize| -> String { format!("v{idx}") };

    // Pass 1: emit params + literals.
    for (idx, node) in graph.nodes.iter().enumerate() {
        match node {
            GraphNode::Param(p) => {
                inputs.push(value_info(&p.name, &p.data_type)?);
                // Aliased: the wire sees `node_name(idx)` but ONNX consumers
                // reference by param name, so we re-emit a trivial Identity
                // if names differ. For the tracer, require wires to use the
                // param's actual name — which is what the builder does.
            }
            GraphNode::Literal { value, data_type } => {
                initializers.push(literal_to_initializer(&node_name(idx), value, data_type)?);
            }
            _ => {}
        }
    }

    // Pass 2: emit apply nodes.
    for (idx, node) in graph.nodes.iter().enumerate() {
        match node {
            GraphNode::Param(_) | GraphNode::Literal { .. } => {}
            GraphNode::Apply { op, output_type: _ } => {
                let onnx_op = onnx_op_for(op)?;
                let inputs_for_node = collect_inputs(graph, idx, &node_name)?;
                nodes.push(onnx::NodeProto {
                    input: inputs_for_node,
                    output: vec![node_name(idx)],
                    name: format!("node_{idx}"),
                    op_type: onnx_op.op_type.into(),
                    domain: onnx_op.domain.into(),
                    attribute: vec![],
                    doc_string: String::new(),
                });
            }
            other => {
                return Err(EmitError::UnsupportedNode(format!("{:?}", std::mem::discriminant(other))));
            }
        }
    }

    // Output.
    let out_idx = graph.output_node;
    if out_idx >= graph.nodes.len() {
        return Err(EmitError::BadWire(out_idx));
    }
    let out_name = match &graph.nodes[out_idx] {
        GraphNode::Param(p) => p.name.clone(),
        _ => node_name(out_idx),
    };
    let outputs = vec![value_info(&out_name, &graph.return_type)?];

    Ok(onnx::GraphProto {
        node: nodes,
        name: graph.name.clone(),
        initializer: initializers,
        input: inputs,
        output: outputs,
        value_info: vec![],
    })
}

fn collect_inputs(
    graph: &FunctionGraph,
    target: usize,
    name_of: &impl Fn(usize) -> String,
) -> Result<Vec<String>, EmitError> {
    let mut incoming: Vec<(i32, String)> = graph
        .wires
        .iter()
        .filter(|w| w.to == target)
        .map(|w| {
            let src_name = match graph.nodes.get(w.from) {
                Some(GraphNode::Param(p)) => p.name.clone(),
                Some(_) => name_of(w.from),
                None => return Err(EmitError::BadWire(w.from)),
            };
            Ok((w.port_in_idx, src_name))
        })
        .collect::<Result<_, _>>()?;
    incoming.sort_by_key(|(port, _)| *port);
    // Sanity: ports should be a dense 0..N sequence.
    for (expected, (actual, _)) in incoming.iter().enumerate() {
        if *actual as usize != expected {
            return Err(EmitError::MissingInput { node: target, port: expected as i32 });
        }
    }
    Ok(incoming.into_iter().map(|(_, name)| name).collect())
}

fn value_info(name: &str, rce_type: &str) -> Result<onnx::ValueInfoProto, EmitError> {
    let elem = onnx_dtype_for(rce_type)
        .ok_or_else(|| EmitError::UnsupportedType(rce_type.into()))?;
    Ok(onnx::ValueInfoProto {
        name: name.into(),
        r#type: Some(onnx::TypeProto {
            value: Some(onnx::type_proto::Value::TensorType(onnx::type_proto::Tensor {
                elem_type: elem,
                // Scalar: zero-dim tensor shape.
                shape: Some(onnx::TensorShapeProto { dim: vec![] }),
            })),
        }),
        doc_string: String::new(),
    })
}

fn literal_to_initializer(
    name: &str,
    value: &LiteralValue,
    rce_type: &str,
) -> Result<onnx::TensorProto, EmitError> {
    let dtype = onnx_dtype_for(rce_type)
        .ok_or_else(|| EmitError::UnsupportedType(rce_type.into()))?;
    let mut proto = onnx::TensorProto {
        dims: vec![],
        data_type: dtype,
        raw_data: vec![],
        float_data: vec![],
        double_data: vec![],
        int64_data: vec![],
        int32_data: vec![],
        name: name.into(),
    };
    match value {
        LiteralValue::F32(v) => proto.float_data = vec![*v],
        LiteralValue::F64(v) => proto.double_data = vec![*v],
        LiteralValue::I32(v) => proto.int32_data = vec![*v],
        LiteralValue::I64(v) => proto.int64_data = vec![*v],
        LiteralValue::Bool(v) => proto.int32_data = vec![if *v { 1 } else { 0 }],
        LiteralValue::String(_) => {
            return Err(EmitError::UnsupportedLiteral("string".into()));
        }
    }
    Ok(proto)
}

/// Write a `FunctionGraph` to an `.onnx` file (binary protobuf).
pub fn write_onnx(graph: &FunctionGraph, path: impl AsRef<Path>) -> Result<(), WriteError> {
    let model = emit_model(graph).map_err(WriteError::Emit)?;
    let bytes = model.encode_to_vec();
    std::fs::write(path, bytes).map_err(WriteError::Io)
}

/// Read an `.onnx` file back to a `ModelProto` (for inspection / round-trip).
pub fn read_onnx(path: impl AsRef<Path>) -> Result<onnx::ModelProto, WriteError> {
    let bytes = std::fs::read(path).map_err(WriteError::Io)?;
    onnx::ModelProto::decode(&bytes[..]).map_err(|e| WriteError::Decode(e.to_string()))
}

#[derive(Debug, thiserror::Error)]
pub enum WriteError {
    #[error("emit: {0}")]
    Emit(#[source] EmitError),
    #[error("io: {0}")]
    Io(#[source] io::Error),
    #[error("decode: {0}")]
    Decode(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use rce_graph::graph::{
        FunctionGraph, GraphNode, LiteralValue, OpKind, Param, Wire, WireKind,
    };
    use prost::Message;

    /// Build: fn add(a: f32, b: f32) -> f32 = a + b
    fn add_graph() -> FunctionGraph {
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
    fn emit_add_graph_shape() {
        let g = add_graph();
        let proto = emit_function(&g).unwrap();
        assert_eq!(proto.name, "add");
        assert_eq!(proto.input.len(), 2);
        assert_eq!(proto.output.len(), 1);
        assert_eq!(proto.node.len(), 1);
        assert_eq!(proto.node[0].op_type, "Add");
        assert_eq!(proto.node[0].input, vec!["a", "b"]);
        assert_eq!(proto.node[0].output, vec!["v2"]);
    }

    #[test]
    fn emit_add_model_has_opset() {
        let g = add_graph();
        let m = emit_model(&g).unwrap();
        assert_eq!(m.opset_import.len(), 1);
        assert_eq!(m.opset_import[0].version, 20);
        assert_eq!(m.ir_version, 10);
    }

    #[test]
    fn emit_add_model_roundtrips_protobuf() {
        let g = add_graph();
        let m = emit_model(&g).unwrap();
        let bytes = m.encode_to_vec();
        assert!(!bytes.is_empty());
        let decoded = onnx::ModelProto::decode(&bytes[..]).unwrap();
        assert_eq!(decoded.producer_name, "rce-emit-onnx");
        assert_eq!(decoded.graph.as_ref().unwrap().node[0].op_type, "Add");
    }

    #[test]
    fn emit_rejects_unmapped_custom_op() {
        let mut g = add_graph();
        g.nodes[2] = GraphNode::Apply {
            op: OpKind::Custom("attractor.competitive_comparison".into()),
            output_type: "f32".into(),
        };
        let err = emit_function(&g).unwrap_err();
        assert!(matches!(err, EmitError::Unmapped(_)), "got {err:?}");
    }

    #[test]
    fn emit_rejects_while_loop() {
        use rce_graph::graph::LoopSubGraph;
        let mut g = add_graph();
        g.nodes[2] = GraphNode::WhileLoop {
            loop_vars: vec![],
            condition: LoopSubGraph { nodes: vec![], wires: vec![], output_nodes: vec![] },
            body: LoopSubGraph { nodes: vec![], wires: vec![], output_nodes: vec![] },
            output_var: 0,
        };
        let err = emit_function(&g).unwrap_err();
        assert!(matches!(err, EmitError::UnsupportedNode(_)), "got {err:?}");
    }

    #[test]
    fn emit_literal_initializer() {
        // fn const_plus_a(a: f32) -> f32 = a + 1.0
        let g = FunctionGraph {
            name: "const_plus_a".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "a".into(), data_type: "f32".into() }),
                GraphNode::Literal { value: LiteralValue::F32(1.0), data_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Add, output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Copy },
            ],
            output_node: 2,
            return_type: "f32".into(),
        };
        let proto = emit_function(&g).unwrap();
        assert_eq!(proto.initializer.len(), 1);
        assert_eq!(proto.initializer[0].name, "v1");
        assert_eq!(proto.initializer[0].float_data, vec![1.0]);
    }

    /// Multi-op chain: relu(matmul(x, w) + b)
    /// Nodes: x(0), w(1), b(2), matmul(3), add(4), relu(5)
    fn mlp_layer_graph() -> FunctionGraph {
        FunctionGraph {
            name: "mlp_layer".into(),
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
        }
    }

    #[test]
    fn emit_mlp_layer_chain() {
        let g = mlp_layer_graph();
        let proto = emit_function(&g).unwrap();
        assert_eq!(proto.input.len(), 3);
        assert_eq!(proto.node.len(), 3);
        assert_eq!(proto.node[0].op_type, "MatMul");
        assert_eq!(proto.node[0].input, vec!["x", "w"]);
        assert_eq!(proto.node[1].op_type, "Add");
        assert_eq!(proto.node[1].input, vec!["v3", "b"]);
        assert_eq!(proto.node[2].op_type, "Relu");
        assert_eq!(proto.node[2].input, vec!["v4"]);
        assert_eq!(proto.output[0].name, "v5");
    }

    #[test]
    fn emit_mlp_layer_roundtrips_protobuf() {
        let g = mlp_layer_graph();
        let m = emit_model(&g).unwrap();
        let bytes = m.encode_to_vec();
        let decoded = onnx::ModelProto::decode(&bytes[..]).unwrap();
        let graph = decoded.graph.unwrap();
        assert_eq!(graph.node.len(), 3);
        assert_eq!(graph.node[0].op_type, "MatMul");
        assert_eq!(graph.node[1].op_type, "Add");
        assert_eq!(graph.node[2].op_type, "Relu");
    }

    /// Subtraction chain: a - b - c (tests wiring intermediate node outputs)
    #[test]
    fn emit_sub_chain() {
        let g = FunctionGraph {
            name: "sub_chain".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "a".into(), data_type: "f64".into() }),
                GraphNode::Param(Param { name: "b".into(), data_type: "f64".into() }),
                GraphNode::Param(Param { name: "c".into(), data_type: "f64".into() }),
                GraphNode::Apply { op: OpKind::Sub, output_type: "f64".into() },
                GraphNode::Apply { op: OpKind::Sub, output_type: "f64".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 3, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 3, port_in_idx: 1, wire_kind: WireKind::Copy },
                Wire { from: 3, to: 4, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 2, to: 4, port_in_idx: 1, wire_kind: WireKind::Copy },
            ],
            output_node: 4,
            return_type: "f64".into(),
        };
        let proto = emit_function(&g).unwrap();
        assert_eq!(proto.node.len(), 2);
        assert_eq!(proto.node[0].op_type, "Sub");
        assert_eq!(proto.node[0].input, vec!["a", "b"]);
        assert_eq!(proto.node[1].op_type, "Sub");
        assert_eq!(proto.node[1].input, vec!["v3", "c"]);
    }

    /// softmax(layer_norm(x))
    #[test]
    fn emit_norm_plus_softmax() {
        let g = FunctionGraph {
            name: "norm_softmax".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("layer_norm".into()), output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("softmax".into()), output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 1, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 2, port_in_idx: 0, wire_kind: WireKind::Copy },
            ],
            output_node: 2,
            return_type: "f32".into(),
        };
        let proto = emit_function(&g).unwrap();
        assert_eq!(proto.node[0].op_type, "LayerNormalization");
        assert_eq!(proto.node[1].op_type, "Softmax");
    }

    #[test]
    fn write_and_read_onnx_file() {
        let g = mlp_layer_graph();
        let tmp = std::env::temp_dir().join("rce_emit_test.onnx");
        write_onnx(&g, &tmp).unwrap();
        let decoded = read_onnx(&tmp).unwrap();
        assert_eq!(decoded.producer_name, "rce-emit-onnx");
        let graph = decoded.graph.unwrap();
        assert_eq!(graph.node.len(), 3);
        assert_eq!(graph.node[0].op_type, "MatMul");
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn emit_with_i64_literal() {
        let g = FunctionGraph {
            name: "add_ten".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "i64".into() }),
                GraphNode::Literal { value: LiteralValue::I64(10), data_type: "i64".into() },
                GraphNode::Apply { op: OpKind::Add, output_type: "i64".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Copy },
            ],
            output_node: 2,
            return_type: "i64".into(),
        };
        let proto = emit_function(&g).unwrap();
        assert_eq!(proto.initializer[0].int64_data, vec![10]);
        assert_eq!(proto.output[0].name, "v2");
    }

    #[test]
    fn emit_bool_comparison_chain() {
        // (a > b) AND (c < d)
        let g = FunctionGraph {
            name: "bool_chain".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "a".into(), data_type: "f32".into() }),
                GraphNode::Param(Param { name: "b".into(), data_type: "f32".into() }),
                GraphNode::Param(Param { name: "c".into(), data_type: "f32".into() }),
                GraphNode::Param(Param { name: "d".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Gt, output_type: "bool".into() },
                GraphNode::Apply { op: OpKind::Lt, output_type: "bool".into() },
                GraphNode::Apply { op: OpKind::And, output_type: "bool".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 4, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 4, port_in_idx: 1, wire_kind: WireKind::Copy },
                Wire { from: 2, to: 5, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 3, to: 5, port_in_idx: 1, wire_kind: WireKind::Copy },
                Wire { from: 4, to: 6, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 5, to: 6, port_in_idx: 1, wire_kind: WireKind::Copy },
            ],
            output_node: 6,
            return_type: "bool".into(),
        };
        let proto = emit_function(&g).unwrap();
        assert_eq!(proto.node.len(), 3);
        assert_eq!(proto.node[0].op_type, "Greater");
        assert_eq!(proto.node[1].op_type, "Less");
        assert_eq!(proto.node[2].op_type, "And");
    }
}
