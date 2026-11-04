//! Emit StableHLO MLIR text from `rce_graph::FunctionGraph`.
//!
//! Produces valid MLIR that can be consumed by `stablehlo-opt`, `iree-compile`,
//! or any tool that accepts the StableHLO dialect.
//!
//! Output format:
//! ```mlir
//! func.func @add(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
//!   %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
//!   return %0 : tensor<f32>
//! }
//! ```

use std::collections::HashMap;
use std::fmt::Write;
use std::path::Path;

use crate::op_map::{hlo_op_for, HloOp};
use crate::types::mlir_scalar_tensor;
use rce_graph::graph::{FunctionGraph, GraphNode, LiteralValue};

#[derive(Debug, thiserror::Error)]
pub enum EmitError {
    #[error("unsupported type: {0}")]
    UnsupportedType(String),
    #[error("unsupported node kind: {0}")]
    UnsupportedNode(String),
    #[error("unmapped op: {0:?}")]
    UnmappedOp(String),
    #[error("graph has no nodes")]
    EmptyGraph,
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("node {node} missing input on port {port}")]
    MissingInput { node: usize, port: i32 },
}

/// Emit a complete MLIR module containing one function.
pub fn emit_module(graph: &FunctionGraph) -> Result<String, EmitError> {
    let func = emit_function(graph)?;
    Ok(format!("module {{\n{func}}}\n"))
}

/// Emit a single `func.func` for the graph.
pub fn emit_function(graph: &FunctionGraph) -> Result<String, EmitError> {
    if graph.nodes.is_empty() {
        return Err(EmitError::EmptyGraph);
    }

    let mut out = String::with_capacity(1024);
    let mut ssa: HashMap<usize, String> = HashMap::new();

    // Collect params.
    let mut param_strs: Vec<String> = Vec::new();
    let mut arg_idx = 0usize;
    for (idx, node) in graph.nodes.iter().enumerate() {
        if let GraphNode::Param(p) = node {
            let ty = mlir_scalar_tensor(&p.data_type)
                .ok_or_else(|| EmitError::UnsupportedType(p.data_type.clone()))?;
            let name = format!("%arg{arg_idx}");
            param_strs.push(format!("{name}: {ty}"));
            ssa.insert(idx, name);
            arg_idx += 1;
        }
    }

    let ret_ty = mlir_scalar_tensor(&graph.return_type)
        .ok_or_else(|| EmitError::UnsupportedType(graph.return_type.clone()))?;

    writeln!(
        out,
        "  func.func @{}({}) -> {} {{",
        graph.name,
        param_strs.join(", "),
        ret_ty,
    )
    .unwrap();

    // SSA counter for intermediate values.
    let mut val_idx = 0usize;
    let val_name = |i: &mut usize| -> String {
        let n = format!("%{i}");
        *i += 1;
        n
    };

    // Build incoming-wire index per node.
    let mut inputs_for: HashMap<usize, Vec<(i32, usize)>> = HashMap::new();
    for w in &graph.wires {
        inputs_for.entry(w.to).or_default().push((w.port_in_idx, w.from));
    }

    // Emit nodes in order.
    for (idx, node) in graph.nodes.iter().enumerate() {
        match node {
            GraphNode::Param(_) => {} // already handled
            GraphNode::Literal { value, data_type } => {
                let ty = mlir_scalar_tensor(data_type)
                    .ok_or_else(|| EmitError::UnsupportedType(data_type.clone()))?;
                let name = val_name(&mut val_idx);
                let lit = format_literal(value);
                writeln!(out, "    {name} = stablehlo.constant dense<{lit}> : {ty}").unwrap();
                ssa.insert(idx, name);
            }
            GraphNode::Apply { op, output_type } => {
                let ty = mlir_scalar_tensor(output_type)
                    .ok_or_else(|| EmitError::UnsupportedType(output_type.clone()))?;
                let hlo = hlo_op_for(op);
                let ins = resolve_inputs(&inputs_for, &ssa, idx)?;
                let name = val_name(&mut val_idx);

                match hlo {
                    HloOp::Binary(op_name) => {
                        let a = ins.get(0).ok_or(EmitError::MissingInput { node: idx, port: 0 })?;
                        let b = ins.get(1).ok_or(EmitError::MissingInput { node: idx, port: 1 })?;
                        writeln!(out, "    {name} = stablehlo.{op_name} {a}, {b} : {ty}").unwrap();
                    }
                    HloOp::Unary(op_name) => {
                        let a = ins.get(0).ok_or(EmitError::MissingInput { node: idx, port: 0 })?;
                        writeln!(out, "    {name} = stablehlo.{op_name} {a} : {ty}").unwrap();
                    }
                    HloOp::Compare(direction) => {
                        let a = ins.get(0).ok_or(EmitError::MissingInput { node: idx, port: 0 })?;
                        let b = ins.get(1).ok_or(EmitError::MissingInput { node: idx, port: 1 })?;
                        // Input type comes from the wired-in value, output is tensor<i1>.
                        let input_ty = infer_input_type(graph, &inputs_for, idx)
                            .unwrap_or_else(|| "tensor<f32>".into());
                        writeln!(
                            out,
                            "    {name} = stablehlo.compare {a}, {b}, {direction} : ({input_ty}, {input_ty}) -> tensor<i1>",
                        ).unwrap();
                    }
                    HloOp::CustomCall(call_name) => {
                        let args = ins.join(", ");
                        let in_types: Vec<String> = ins.iter().map(|_| ty.clone()).collect();
                        writeln!(
                            out,
                            "    {name} = stablehlo.custom_call @{call_name}({args}) : ({}) -> {ty}",
                            in_types.join(", "),
                        ).unwrap();
                    }
                    HloOp::Unmapped => {
                        return Err(EmitError::UnmappedOp(format!("{op:?}")));
                    }
                }
                ssa.insert(idx, name);
            }
            other => {
                return Err(EmitError::UnsupportedNode(
                    format!("{:?}", std::mem::discriminant(other)),
                ));
            }
        }
    }

    // Return.
    let ret_name = ssa
        .get(&graph.output_node)
        .cloned()
        .unwrap_or_else(|| "%0".into());
    writeln!(out, "    return {ret_name} : {ret_ty}").unwrap();
    writeln!(out, "  }}").unwrap();

    Ok(out)
}

/// Write StableHLO MLIR to a file.
pub fn write_stablehlo(graph: &FunctionGraph, path: impl AsRef<Path>) -> Result<(), EmitError> {
    let text = emit_module(graph)?;
    std::fs::write(path, text)?;
    Ok(())
}

fn resolve_inputs(
    inputs_for: &HashMap<usize, Vec<(i32, usize)>>,
    ssa: &HashMap<usize, String>,
    target: usize,
) -> Result<Vec<String>, EmitError> {
    let mut pairs: Vec<(i32, usize)> = inputs_for
        .get(&target)
        .cloned()
        .unwrap_or_default();
    pairs.sort_by_key(|(port, _)| *port);
    pairs
        .iter()
        .map(|(_, src)| {
            ssa.get(src)
                .cloned()
                .ok_or(EmitError::MissingInput { node: target, port: 0 })
        })
        .collect()
}

fn infer_input_type(
    graph: &FunctionGraph,
    inputs_for: &HashMap<usize, Vec<(i32, usize)>>,
    target: usize,
) -> Option<String> {
    let pairs = inputs_for.get(&target)?;
    let (_, src) = pairs.first()?;
    let dt = match graph.nodes.get(*src)? {
        GraphNode::Param(p) => &p.data_type,
        GraphNode::Literal { data_type, .. } => data_type,
        GraphNode::Apply { output_type, .. } => output_type,
        _ => return None,
    };
    mlir_scalar_tensor(dt)
}

fn format_literal(value: &LiteralValue) -> String {
    match value {
        LiteralValue::I32(v) => format!("{v}"),
        LiteralValue::I64(v) => format!("{v}"),
        LiteralValue::F32(v) => format_float(*v as f64),
        LiteralValue::F64(v) => format_float(*v),
        LiteralValue::Bool(true) => "1".into(),
        LiteralValue::Bool(false) => "0".into(),
        LiteralValue::String(s) => format!("\"{s}\""),
    }
}

fn format_float(v: f64) -> String {
    if v == v.floor() && v.abs() < 1e15 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rce_graph::graph::{
        FunctionGraph, GraphNode, LiteralValue, OpKind, Param, Wire, WireKind,
    };

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
    fn emit_add_function() {
        let text = emit_function(&add_graph()).unwrap();
        assert!(text.contains("func.func @add"));
        assert!(text.contains("stablehlo.add %arg0, %arg1 : tensor<f32>"));
        assert!(text.contains("return %0 : tensor<f32>"));
    }

    #[test]
    fn emit_add_module() {
        let text = emit_module(&add_graph()).unwrap();
        assert!(text.starts_with("module {"));
        assert!(text.contains("func.func @add"));
        assert!(text.trim_end().ends_with('}'));
    }

    #[test]
    fn emit_mlp_layer() {
        let g = FunctionGraph {
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
        let text = emit_function(&g).unwrap();
        assert!(text.contains("stablehlo.dot %arg0, %arg1 : tensor<f32>"));
        assert!(text.contains("stablehlo.add %0, %arg2 : tensor<f32>"));
        assert!(text.contains("stablehlo.custom_call @relu(%1)"));
        assert!(text.contains("return %2 : tensor<f32>"));
    }

    #[test]
    fn emit_comparison() {
        let g = FunctionGraph {
            name: "gt".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "a".into(), data_type: "f32".into() }),
                GraphNode::Param(Param { name: "b".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Gt, output_type: "bool".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Copy },
            ],
            output_node: 2,
            return_type: "bool".into(),
        };
        let text = emit_function(&g).unwrap();
        assert!(text.contains("stablehlo.compare %arg0, %arg1, GT"));
        assert!(text.contains("-> tensor<i1>"));
    }

    #[test]
    fn emit_with_constant() {
        let g = FunctionGraph {
            name: "add_one".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
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
        let text = emit_function(&g).unwrap();
        assert!(text.contains("stablehlo.constant dense<1.0> : tensor<f32>"));
        assert!(text.contains("stablehlo.add %arg0, %0 : tensor<f32>"));
    }

    #[test]
    fn emit_unary_chain() {
        // exp(neg(x))
        let g = FunctionGraph {
            name: "exp_neg".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "f64".into() }),
                GraphNode::Apply { op: OpKind::Neg, output_type: "f64".into() },
                GraphNode::Apply { op: OpKind::Custom("exp".into()), output_type: "f64".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 1, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 2, port_in_idx: 0, wire_kind: WireKind::Copy },
            ],
            output_node: 2,
            return_type: "f64".into(),
        };
        let text = emit_function(&g).unwrap();
        assert!(text.contains("stablehlo.negate %arg0 : tensor<f64>"));
        assert!(text.contains("stablehlo.exponential %0 : tensor<f64>"));
    }

    #[test]
    fn emit_norm_softmax() {
        let g = FunctionGraph {
            name: "norm_sm".into(),
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
        let text = emit_function(&g).unwrap();
        assert!(text.contains("@layer_norm"));
        assert!(text.contains("@softmax"));
    }

    #[test]
    fn emit_rejects_unmapped() {
        let g = FunctionGraph {
            name: "bad".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("hdc.bundle".into()), output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 1, port_in_idx: 0, wire_kind: WireKind::Copy },
            ],
            output_node: 1,
            return_type: "f32".into(),
        };
        assert!(emit_function(&g).is_err());
    }

    #[test]
    fn write_stablehlo_file() {
        let g = add_graph();
        let tmp = std::env::temp_dir().join("rce_stablehlo_test.mlir");
        write_stablehlo(&g, &tmp).unwrap();
        let content = std::fs::read_to_string(&tmp).unwrap();
        assert!(content.contains("stablehlo.add"));
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn emit_i64_types() {
        let g = FunctionGraph {
            name: "add_i64".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "a".into(), data_type: "i64".into() }),
                GraphNode::Param(Param { name: "b".into(), data_type: "i64".into() }),
                GraphNode::Apply { op: OpKind::Add, output_type: "i64".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Copy },
            ],
            output_node: 2,
            return_type: "i64".into(),
        };
        let text = emit_function(&g).unwrap();
        assert!(text.contains("tensor<i64>"));
    }

    #[test]
    fn emit_sub_chain() {
        let g = FunctionGraph {
            name: "sub_chain".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "a".into(), data_type: "f32".into() }),
                GraphNode::Param(Param { name: "b".into(), data_type: "f32".into() }),
                GraphNode::Param(Param { name: "c".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Sub, output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Sub, output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 3, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 3, port_in_idx: 1, wire_kind: WireKind::Copy },
                Wire { from: 3, to: 4, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 2, to: 4, port_in_idx: 1, wire_kind: WireKind::Copy },
            ],
            output_node: 4,
            return_type: "f32".into(),
        };
        let text = emit_function(&g).unwrap();
        assert!(text.contains("stablehlo.subtract %arg0, %arg1"));
        assert!(text.contains("stablehlo.subtract %0, %arg2"));
    }
}
