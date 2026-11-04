//! Tests for rce-ingest-fx.

use serde_json::json;

use rce_graph::graph::{GraphNode, LiteralValue, OpKind, TensorLiteralData, WireKind};

use crate::parse::{parse_fx_json, FxArg, FxOp};
use crate::to_rce::to_function_graph;
use crate::IngestError;

// ── Parsing tests ──────────────────────────────────────────────────

#[test]
fn parse_simple_add_graph() {
    let val = json!({
        "graph": {
            "nodes": [
                {"op": "placeholder", "name": "x", "target": "x", "args": [], "kwargs": {}},
                {"op": "placeholder", "name": "y", "target": "y", "args": [], "kwargs": {}},
                {"op": "call_function", "name": "add", "target": "torch.add", "args": ["x", "y"], "kwargs": {}},
                {"op": "output", "name": "output", "target": "output", "args": ["add"], "kwargs": {}}
            ]
        }
    });

    let fx = parse_fx_json(&val).unwrap();
    assert_eq!(fx.nodes.len(), 4);
    assert_eq!(fx.nodes[0].op, FxOp::Placeholder);
    assert_eq!(fx.nodes[0].name, "x");
    assert_eq!(fx.nodes[2].op, FxOp::CallFunction);
    assert_eq!(fx.nodes[2].target, "torch.add");
    assert_eq!(fx.nodes[3].op, FxOp::Output);
}

#[test]
fn parse_with_call_method() {
    let val = json!({
        "graph": {
            "nodes": [
                {"op": "placeholder", "name": "x", "target": "x", "args": [], "kwargs": {}},
                {"op": "call_method", "name": "relu_result", "target": "relu", "args": ["x"], "kwargs": {}},
                {"op": "output", "name": "output", "target": "output", "args": ["relu_result"], "kwargs": {}}
            ]
        }
    });

    let fx = parse_fx_json(&val).unwrap();
    assert_eq!(fx.nodes.len(), 3);
    assert_eq!(fx.nodes[1].op, FxOp::CallMethod);
    assert_eq!(fx.nodes[1].target, "relu");
}

#[test]
fn parse_with_literal_args() {
    let val = json!({
        "graph": {
            "nodes": [
                {"op": "placeholder", "name": "x", "target": "x", "args": [], "kwargs": {}},
                {"op": "call_function", "name": "add_scalar", "target": "torch.add", "args": ["x", 42], "kwargs": {}},
                {"op": "output", "name": "output", "target": "output", "args": ["add_scalar"], "kwargs": {}}
            ]
        }
    });

    let fx = parse_fx_json(&val).unwrap();
    assert_eq!(fx.nodes[1].args.len(), 2);
    match &fx.nodes[1].args[1] {
        FxArg::Int(v) => assert_eq!(*v, 42),
        other => panic!("expected Int, got {other:?}"),
    }
}

#[test]
fn parse_with_float_and_bool_args() {
    let val = json!({
        "graph": {
            "nodes": [
                {"op": "placeholder", "name": "x", "target": "x", "args": [], "kwargs": {}},
                {"op": "call_function", "name": "op", "target": "torch.clamp", "args": ["x", 0.0, 6.0], "kwargs": {}},
                {"op": "call_function", "name": "drop", "target": "torch.nn.functional.dropout", "args": ["op", false], "kwargs": {}},
                {"op": "output", "name": "output", "target": "output", "args": ["drop"], "kwargs": {}}
            ]
        }
    });

    let fx = parse_fx_json(&val).unwrap();
    match &fx.nodes[1].args[1] {
        FxArg::Float(v) => assert!((v - 0.0).abs() < f64::EPSILON),
        other => panic!("expected Float, got {other:?}"),
    }
    match &fx.nodes[2].args[1] {
        FxArg::Bool(v) => assert!(!v),
        other => panic!("expected Bool, got {other:?}"),
    }
}

// ── Conversion tests ───────────────────────────────────────────────

#[test]
fn convert_placeholder_to_param() {
    let val = json!({
        "graph": {
            "nodes": [
                {"op": "placeholder", "name": "x", "target": "x", "args": [], "kwargs": {}},
                {"op": "call_function", "name": "id", "target": "torch.add", "args": ["x", "x"], "kwargs": {}},
                {"op": "output", "name": "output", "target": "output", "args": ["id"], "kwargs": {}}
            ]
        }
    });

    let fx = parse_fx_json(&val).unwrap();
    let fg = to_function_graph(&fx).unwrap();

    match &fg.nodes[0] {
        GraphNode::Param(p) => {
            assert_eq!(p.name, "x");
            assert_eq!(p.data_type, "f32");
        }
        other => panic!("expected Param, got {other:?}"),
    }
}

#[test]
fn convert_add_to_add_opkind() {
    let val = json!({
        "graph": {
            "nodes": [
                {"op": "placeholder", "name": "a", "target": "a", "args": [], "kwargs": {}},
                {"op": "placeholder", "name": "b", "target": "b", "args": [], "kwargs": {}},
                {"op": "call_function", "name": "sum", "target": "torch.add", "args": ["a", "b"], "kwargs": {}},
                {"op": "output", "name": "output", "target": "output", "args": ["sum"], "kwargs": {}}
            ]
        }
    });

    let fx = parse_fx_json(&val).unwrap();
    let fg = to_function_graph(&fx).unwrap();

    match &fg.nodes[2] {
        GraphNode::Apply { op, output_type } => {
            assert_eq!(*op, OpKind::Add);
            assert_eq!(output_type, "f32");
        }
        other => panic!("expected Apply, got {other:?}"),
    }
}

#[test]
fn convert_relu_to_custom() {
    let val = json!({
        "graph": {
            "nodes": [
                {"op": "placeholder", "name": "x", "target": "x", "args": [], "kwargs": {}},
                {"op": "call_function", "name": "act", "target": "torch.nn.functional.relu", "args": ["x"], "kwargs": {}},
                {"op": "output", "name": "output", "target": "output", "args": ["act"], "kwargs": {}}
            ]
        }
    });

    let fx = parse_fx_json(&val).unwrap();
    let fg = to_function_graph(&fx).unwrap();

    match &fg.nodes[1] {
        GraphNode::Apply { op, .. } => {
            assert_eq!(*op, OpKind::Custom("relu".into()));
        }
        other => panic!("expected Apply(relu), got {other:?}"),
    }
}

#[test]
fn convert_matmul_to_custom() {
    let val = json!({
        "graph": {
            "nodes": [
                {"op": "placeholder", "name": "a", "target": "a", "args": [], "kwargs": {}},
                {"op": "placeholder", "name": "b", "target": "b", "args": [], "kwargs": {}},
                {"op": "call_function", "name": "mm", "target": "torch.matmul", "args": ["a", "b"], "kwargs": {}},
                {"op": "output", "name": "output", "target": "output", "args": ["mm"], "kwargs": {}}
            ]
        }
    });

    let fx = parse_fx_json(&val).unwrap();
    let fg = to_function_graph(&fx).unwrap();

    match &fg.nodes[2] {
        GraphNode::Apply { op, .. } => {
            assert_eq!(*op, OpKind::Custom("matmul".into()));
        }
        other => panic!("expected Apply(matmul), got {other:?}"),
    }
}

#[test]
fn full_mlp_linear_relu() {
    let val = json!({
        "graph": {
            "nodes": [
                {"op": "placeholder", "name": "x", "target": "x", "args": [], "kwargs": {}},
                {"op": "placeholder", "name": "weight", "target": "weight", "args": [], "kwargs": {}},
                {"op": "call_function", "name": "linear", "target": "torch.nn.functional.linear", "args": ["x", "weight"], "kwargs": {}},
                {"op": "call_function", "name": "relu", "target": "torch.nn.functional.relu", "args": ["linear"], "kwargs": {}},
                {"op": "output", "name": "output", "target": "output", "args": ["relu"], "kwargs": {}}
            ]
        }
    });

    let fx = parse_fx_json(&val).unwrap();
    let fg = to_function_graph(&fx).unwrap();

    // 2 params + linear Apply + relu Apply = 4 nodes
    assert_eq!(fg.nodes.len(), 4);
    assert_eq!(fg.name, "fx_graph");
    assert_eq!(fg.return_type, "f32");

    // linear has Custom("linear"), relu has Custom("relu")
    match &fg.nodes[2] {
        GraphNode::Apply { op, .. } => assert_eq!(*op, OpKind::Custom("linear".into())),
        other => panic!("expected linear Apply, got {other:?}"),
    }
    match &fg.nodes[3] {
        GraphNode::Apply { op, .. } => assert_eq!(*op, OpKind::Custom("relu".into())),
        other => panic!("expected relu Apply, got {other:?}"),
    }
}

#[test]
fn wire_connections_are_correct() {
    let val = json!({
        "graph": {
            "nodes": [
                {"op": "placeholder", "name": "a", "target": "a", "args": [], "kwargs": {}},
                {"op": "placeholder", "name": "b", "target": "b", "args": [], "kwargs": {}},
                {"op": "call_function", "name": "sum", "target": "torch.add", "args": ["a", "b"], "kwargs": {}},
                {"op": "output", "name": "output", "target": "output", "args": ["sum"], "kwargs": {}}
            ]
        }
    });

    let fx = parse_fx_json(&val).unwrap();
    let fg = to_function_graph(&fx).unwrap();

    // Should have 2 wires: a→sum (port 0), b→sum (port 1)
    assert_eq!(fg.wires.len(), 2);

    let w0 = &fg.wires[0];
    assert_eq!(w0.from, 0); // node "a"
    assert_eq!(w0.to, 2); // node "sum"
    assert_eq!(w0.port_in_idx, 0);
    assert_eq!(w0.wire_kind, WireKind::Copy);

    let w1 = &fg.wires[1];
    assert_eq!(w1.from, 1); // node "b"
    assert_eq!(w1.to, 2);
    assert_eq!(w1.port_in_idx, 1);
    assert_eq!(w1.wire_kind, WireKind::Copy);
}

#[test]
fn output_node_is_correct() {
    let val = json!({
        "graph": {
            "nodes": [
                {"op": "placeholder", "name": "x", "target": "x", "args": [], "kwargs": {}},
                {"op": "call_function", "name": "neg_x", "target": "torch.neg", "args": ["x"], "kwargs": {}},
                {"op": "output", "name": "output", "target": "output", "args": ["neg_x"], "kwargs": {}}
            ]
        }
    });

    let fx = parse_fx_json(&val).unwrap();
    let fg = to_function_graph(&fx).unwrap();

    assert_eq!(fg.output_node, 1); // "neg_x" is at index 1
}

#[test]
fn reject_invalid_json() {
    let result = crate::ingest("not json at all {{{");
    assert!(result.is_err());
    match result.unwrap_err() {
        IngestError::Json(_) => {}
        other => panic!("expected Json error, got {other:?}"),
    }
}

#[test]
fn reject_unknown_op_type() {
    let val = json!({
        "graph": {
            "nodes": [
                {"op": "teleport", "name": "x", "target": "x", "args": [], "kwargs": {}}
            ]
        }
    });

    let result = parse_fx_json(&val);
    assert!(result.is_err());
    match result.unwrap_err() {
        IngestError::UnknownOp(op) => assert_eq!(op, "teleport"),
        other => panic!("expected UnknownOp, got {other:?}"),
    }
}

#[test]
fn roundtrip_node_and_wire_counts() {
    // A small residual block: x → linear → relu → add(x, relu) → output
    let val = json!({
        "graph": {
            "nodes": [
                {"op": "placeholder", "name": "x", "target": "x", "args": [], "kwargs": {}},
                {"op": "placeholder", "name": "w", "target": "w", "args": [], "kwargs": {}},
                {"op": "call_function", "name": "fc", "target": "torch.nn.functional.linear", "args": ["x", "w"], "kwargs": {}},
                {"op": "call_function", "name": "act", "target": "torch.nn.functional.relu", "args": ["fc"], "kwargs": {}},
                {"op": "call_function", "name": "res", "target": "torch.add", "args": ["x", "act"], "kwargs": {}},
                {"op": "output", "name": "output", "target": "output", "args": ["res"], "kwargs": {}}
            ]
        }
    });

    let fx = parse_fx_json(&val).unwrap();
    let fg = to_function_graph(&fx).unwrap();

    // Nodes: x(0), w(1), fc(2), act(3), res(4) = 5
    assert_eq!(fg.nodes.len(), 5);

    // Wires: x→fc(0), w→fc(1), fc→act(0), x→res(0), act→res(1) = 5
    assert_eq!(fg.wires.len(), 5);

    // Output is res
    assert_eq!(fg.output_node, 4);
}

#[test]
fn literal_args_create_literal_nodes() {
    let val = json!({
        "graph": {
            "nodes": [
                {"op": "placeholder", "name": "x", "target": "x", "args": [], "kwargs": {}},
                {"op": "call_function", "name": "scaled", "target": "torch.mul", "args": ["x", 2], "kwargs": {}},
                {"op": "output", "name": "output", "target": "output", "args": ["scaled"], "kwargs": {}}
            ]
        }
    });

    let fx = parse_fx_json(&val).unwrap();
    let fg = to_function_graph(&fx).unwrap();

    // Nodes: x(0), mul_apply(1), literal_2(2) = 3
    assert_eq!(fg.nodes.len(), 3);

    // The literal node should be at index 2 (created during wire emission for the mul)
    match &fg.nodes[2] {
        GraphNode::Literal { value, data_type } => {
            match value {
                LiteralValue::I64(v) => assert_eq!(*v, 2),
                other => panic!("expected I64, got {other:?}"),
            }
            assert_eq!(data_type, "i64");
        }
        other => panic!("expected Literal, got {other:?}"),
    }

    // Wire from literal → mul
    let lit_wire = fg.wires.iter().find(|w| w.from == 2).unwrap();
    assert_eq!(lit_wire.to, 1);
    assert_eq!(lit_wire.port_in_idx, 1);
}

#[test]
fn get_attr_becomes_param() {
    let val = json!({
        "graph": {
            "nodes": [
                {"op": "placeholder", "name": "x", "target": "x", "args": [], "kwargs": {}},
                {"op": "get_attr", "name": "weight_0", "target": "layer1.weight", "args": [], "kwargs": {}},
                {"op": "call_function", "name": "mm", "target": "torch.matmul", "args": ["x", "weight_0"], "kwargs": {}},
                {"op": "output", "name": "output", "target": "output", "args": ["mm"], "kwargs": {}}
            ]
        }
    });

    let fx = parse_fx_json(&val).unwrap();
    let fg = to_function_graph(&fx).unwrap();

    match &fg.nodes[1] {
        GraphNode::Param(p) => {
            assert_eq!(p.name, "layer1.weight");
            assert_eq!(p.data_type, "f32");
        }
        other => panic!("expected Param for get_attr, got {other:?}"),
    }
}

#[test]
fn call_module_target_preserved() {
    let val = json!({
        "graph": {
            "nodes": [
                {"op": "placeholder", "name": "x", "target": "x", "args": [], "kwargs": {}},
                {"op": "call_module", "name": "conv", "target": "features.0", "args": ["x"], "kwargs": {}},
                {"op": "output", "name": "output", "target": "output", "args": ["conv"], "kwargs": {}}
            ]
        }
    });

    let fx = parse_fx_json(&val).unwrap();
    let fg = to_function_graph(&fx).unwrap();

    match &fg.nodes[1] {
        GraphNode::Apply { op, .. } => {
            assert_eq!(*op, OpKind::Custom("features.0".into()));
        }
        other => panic!("expected Apply with module target, got {other:?}"),
    }
}

#[test]
fn no_output_node_is_error() {
    let val = json!({
        "graph": {
            "nodes": [
                {"op": "placeholder", "name": "x", "target": "x", "args": [], "kwargs": {}}
            ]
        }
    });

    let fx = parse_fx_json(&val).unwrap();
    let result = to_function_graph(&fx);
    assert!(result.is_err());
    match result.unwrap_err() {
        IngestError::NoOutput => {}
        other => panic!("expected NoOutput, got {other:?}"),
    }
}

#[test]
fn ingest_convenience_function() {
    let json_str = r#"{
        "graph": {
            "nodes": [
                {"op": "placeholder", "name": "x", "target": "x", "args": [], "kwargs": {}},
                {"op": "call_function", "name": "exp_x", "target": "torch.exp", "args": ["x"], "kwargs": {}},
                {"op": "output", "name": "output", "target": "output", "args": ["exp_x"], "kwargs": {}}
            ]
        }
    }"#;

    let fg = crate::ingest(json_str).unwrap();
    assert_eq!(fg.nodes.len(), 2);
    assert_eq!(fg.output_node, 1);
    match &fg.nodes[1] {
        GraphNode::Apply { op, .. } => assert_eq!(*op, OpKind::Custom("exp".into())),
        other => panic!("expected exp Apply, got {other:?}"),
    }
}

#[test]
fn operator_aliases_map_correctly() {
    let val = json!({
        "graph": {
            "nodes": [
                {"op": "placeholder", "name": "a", "target": "a", "args": [], "kwargs": {}},
                {"op": "placeholder", "name": "b", "target": "b", "args": [], "kwargs": {}},
                {"op": "call_function", "name": "s", "target": "operator.sub", "args": ["a", "b"], "kwargs": {}},
                {"op": "call_function", "name": "m", "target": "operator.mul", "args": ["a", "b"], "kwargs": {}},
                {"op": "call_function", "name": "d", "target": "operator.truediv", "args": ["s", "m"], "kwargs": {}},
                {"op": "output", "name": "output", "target": "output", "args": ["d"], "kwargs": {}}
            ]
        }
    });

    let fx = parse_fx_json(&val).unwrap();
    let fg = to_function_graph(&fx).unwrap();

    match &fg.nodes[2] {
        GraphNode::Apply { op, .. } => assert_eq!(*op, OpKind::Sub),
        other => panic!("expected Sub, got {other:?}"),
    }
    match &fg.nodes[3] {
        GraphNode::Apply { op, .. } => assert_eq!(*op, OpKind::Mul),
        other => panic!("expected Mul, got {other:?}"),
    }
    match &fg.nodes[4] {
        GraphNode::Apply { op, .. } => assert_eq!(*op, OpKind::Div),
        other => panic!("expected Div, got {other:?}"),
    }
}

// ── Shape/dtype enrichment for get_attr (PyTorch path) ─────────────

#[test]
fn get_attr_with_shape_dtype_emits_tensor_literal() {
    let val = json!({
        "graph": {
            "nodes": [
                {"op": "placeholder", "name": "x", "target": "x", "args": []},
                {"op": "get_attr", "name": "weight", "target": "fc1.weight",
                 "args": [], "shape": [8, 4], "dtype": "f32"},
                {"op": "call_function", "name": "linear", "target": "torch.ops.aten.linear",
                 "args": ["x", "weight"]},
                {"op": "output", "name": "out", "target": "output", "args": ["linear"]}
            ]
        }
    });
    let fx = parse_fx_json(&val).unwrap();
    let fg = to_function_graph(&fx).unwrap();

    // Param x, then TensorLiteral fc1.weight (NOT a Param), then Apply.
    match &fg.nodes[1] {
        GraphNode::TensorLiteral(td) => {
            assert_eq!(td.name, "fc1.weight");
            assert_eq!(td.shape, vec![8, 4]);
            assert_eq!(td.dtype, "f32");
            assert!(td.data.is_empty(), "data should be empty (sidecar reattach)");
        }
        other => panic!("expected TensorLiteral, got {other:?}"),
    }
}

#[test]
fn get_attr_without_shape_falls_back_to_param() {
    // Backwards-compat: FX JSON without shape/dtype on get_attr produces
    // a Param node, matching the original behavior.
    let val = json!({
        "graph": {
            "nodes": [
                {"op": "placeholder", "name": "x", "target": "x", "args": []},
                {"op": "get_attr", "name": "weight", "target": "fc1.weight", "args": []},
                {"op": "call_function", "name": "add", "target": "torch.add",
                 "args": ["x", "weight"]},
                {"op": "output", "name": "out", "target": "output", "args": ["add"]}
            ]
        }
    });
    let fx = parse_fx_json(&val).unwrap();
    let fg = to_function_graph(&fx).unwrap();

    match &fg.nodes[1] {
        GraphNode::Param(p) => assert_eq!(p.name, "fc1.weight"),
        other => panic!("expected Param fallback, got {other:?}"),
    }
}

