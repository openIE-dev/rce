//! Emit MLIR linalg/arith/math/tensor dialect text from `rce_graph::FunctionGraph`.
//!
//! Produces valid MLIR that can be consumed by `mlir-opt`, `iree-compile`,
//! Polygeist, or any tool that accepts the linalg/arith/math/tensor dialects.
//!
//! Output format:
//! ```mlir
//! func.func @add(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
//!   %0 = arith.addf %arg0, %arg1 : tensor<f32>
//!   return %0 : tensor<f32>
//! }
//! ```

use std::collections::HashMap;
use std::fmt::Write;
use std::path::Path;

use crate::op_map::{linalg_op_for, LinalgOp};
use crate::types::{is_float_type, mlir_scalar_tensor};
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
                writeln!(out, "    {name} = arith.constant dense<{lit}> : {ty}").unwrap();
                ssa.insert(idx, name);
            }
            GraphNode::Apply { op, output_type } => {
                let ty = mlir_scalar_tensor(output_type)
                    .ok_or_else(|| EmitError::UnsupportedType(output_type.clone()))?;
                let lop = linalg_op_for(op);
                let ins = resolve_inputs(&inputs_for, &ssa, idx)?;
                let name = val_name(&mut val_idx);

                // Determine if the output (or input for comparisons) is float.
                let is_float = is_float_type(output_type);
                // For comparisons the output is bool, so check input type.
                let input_is_float = infer_input_is_float(graph, &inputs_for, idx);

                match lop {
                    LinalgOp::ArithBinary { int_op, float_op } => {
                        let a = ins.get(0).ok_or(EmitError::MissingInput { node: idx, port: 0 })?;
                        let b = ins.get(1).ok_or(EmitError::MissingInput { node: idx, port: 1 })?;
                        let op_name = if is_float { float_op } else { int_op };
                        writeln!(out, "    {name} = {op_name} {a}, {b} : {ty}").unwrap();
                    }
                    LinalgOp::ArithUnary { int_op, float_op } => {
                        let a = ins.get(0).ok_or(EmitError::MissingInput { node: idx, port: 0 })?;
                        if is_float {
                            writeln!(out, "    {name} = {float_op} {a} : {ty}").unwrap();
                        } else if let Some(iop) = int_op {
                            writeln!(out, "    {name} = {iop} {a} : {ty}").unwrap();
                        } else {
                            // Integer negation: 0 - x
                            let zero = val_name(&mut val_idx);
                            writeln!(out, "    {zero} = arith.constant dense<0> : {ty}").unwrap();
                            writeln!(out, "    {name} = arith.subi {zero}, {a} : {ty}").unwrap();
                        }
                    }
                    LinalgOp::CmpFloat(pred) => {
                        let a = ins.get(0).ok_or(EmitError::MissingInput { node: idx, port: 0 })?;
                        let b = ins.get(1).ok_or(EmitError::MissingInput { node: idx, port: 1 })?;
                        let input_ty = infer_input_type(graph, &inputs_for, idx)
                            .unwrap_or_else(|| "tensor<f32>".into());
                        writeln!(
                            out,
                            "    {name} = arith.cmpf \"{pred}\", {a}, {b} : {input_ty}",
                        )
                        .unwrap();
                    }
                    LinalgOp::CmpInt(pred) => {
                        let a = ins.get(0).ok_or(EmitError::MissingInput { node: idx, port: 0 })?;
                        let b = ins.get(1).ok_or(EmitError::MissingInput { node: idx, port: 1 })?;
                        let input_ty = infer_input_type(graph, &inputs_for, idx)
                            .unwrap_or_else(|| "tensor<i64>".into());
                        writeln!(
                            out,
                            "    {name} = arith.cmpi \"{pred}\", {a}, {b} : {input_ty}",
                        )
                        .unwrap();
                    }
                    LinalgOp::Cmp { int_pred, float_pred } => {
                        let a = ins.get(0).ok_or(EmitError::MissingInput { node: idx, port: 0 })?;
                        let b = ins.get(1).ok_or(EmitError::MissingInput { node: idx, port: 1 })?;
                        let input_ty = infer_input_type(graph, &inputs_for, idx)
                            .unwrap_or_else(|| "tensor<f32>".into());
                        if input_is_float {
                            writeln!(
                                out,
                                "    {name} = arith.cmpf \"{float_pred}\", {a}, {b} : {input_ty}",
                            )
                            .unwrap();
                        } else {
                            writeln!(
                                out,
                                "    {name} = arith.cmpi \"{int_pred}\", {a}, {b} : {input_ty}",
                            )
                            .unwrap();
                        }
                    }
                    LinalgOp::LogicBinary(op_name) => {
                        let a = ins.get(0).ok_or(EmitError::MissingInput { node: idx, port: 0 })?;
                        let b = ins.get(1).ok_or(EmitError::MissingInput { node: idx, port: 1 })?;
                        writeln!(out, "    {name} = {op_name} {a}, {b} : {ty}").unwrap();
                    }
                    LinalgOp::LogicNot => {
                        let a = ins.get(0).ok_or(EmitError::MissingInput { node: idx, port: 0 })?;
                        // NOT via XOR with all-ones constant.
                        let ones = val_name(&mut val_idx);
                        writeln!(out, "    {ones} = arith.constant dense<true> : {ty}").unwrap();
                        writeln!(out, "    {name} = arith.xori {a}, {ones} : {ty}").unwrap();
                    }
                    LinalgOp::MathUnary(op_name) => {
                        let a = ins.get(0).ok_or(EmitError::MissingInput { node: idx, port: 0 })?;
                        writeln!(out, "    {name} = {op_name} {a} : {ty}").unwrap();
                    }
                    LinalgOp::MathBinary(op_name) => {
                        let a = ins.get(0).ok_or(EmitError::MissingInput { node: idx, port: 0 })?;
                        let b = ins.get(1).ok_or(EmitError::MissingInput { node: idx, port: 1 })?;
                        writeln!(out, "    {name} = {op_name} {a}, {b} : {ty}").unwrap();
                    }
                    LinalgOp::LinalgNamed(op_name) => {
                        let args = ins.join(", ");
                        writeln!(
                            out,
                            "    {name} = {op_name} ins({args} : {ty}, {ty}) outs(%empty : {ty}) -> {ty}",
                        )
                        .unwrap();
                    }
                    LinalgOp::Relu => {
                        let a = ins.get(0).ok_or(EmitError::MissingInput { node: idx, port: 0 })?;
                        let zero = val_name(&mut val_idx);
                        if is_float {
                            writeln!(out, "    {zero} = arith.constant dense<0.0> : {ty}").unwrap();
                            writeln!(out, "    {name} = arith.maximumf {a}, {zero} : {ty}").unwrap();
                        } else {
                            writeln!(out, "    {zero} = arith.constant dense<0> : {ty}").unwrap();
                            writeln!(out, "    {name} = arith.maxsi {a}, {zero} : {ty}").unwrap();
                        }
                    }
                    LinalgOp::Unmapped => {
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

/// Write MLIR linalg dialect text to a file.
pub fn write_mlir_linalg(graph: &FunctionGraph, path: impl AsRef<Path>) -> Result<(), EmitError> {
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

/// Check whether the first input to a node is a float type.
fn infer_input_is_float(
    graph: &FunctionGraph,
    inputs_for: &HashMap<usize, Vec<(i32, usize)>>,
    target: usize,
) -> bool {
    let Some(pairs) = inputs_for.get(&target) else {
        return false;
    };
    let Some((_, src)) = pairs.first() else {
        return false;
    };
    let Some(node) = graph.nodes.get(*src) else {
        return false;
    };
    let dt = match node {
        GraphNode::Param(p) => &p.data_type,
        GraphNode::Literal { data_type, .. } => data_type,
        GraphNode::Apply { output_type, .. } => output_type,
        _ => return false,
    };
    is_float_type(dt)
}

fn format_literal(value: &LiteralValue) -> String {
    match value {
        LiteralValue::I32(v) => format!("{v}"),
        LiteralValue::I64(v) => format!("{v}"),
        LiteralValue::F32(v) => format_float(*v as f64),
        LiteralValue::F64(v) => format_float(*v),
        LiteralValue::Bool(true) => "true".into(),
        LiteralValue::Bool(false) => "false".into(),
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

    // ---- helpers ----

    fn float_add_graph() -> FunctionGraph {
        FunctionGraph {
            name: "add_f".into(),
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

    fn int_add_graph() -> FunctionGraph {
        FunctionGraph {
            name: "add_i".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "a".into(), data_type: "i32".into() }),
                GraphNode::Param(Param { name: "b".into(), data_type: "i32".into() }),
                GraphNode::Apply { op: OpKind::Add, output_type: "i32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Copy },
            ],
            output_node: 2,
            return_type: "i32".into(),
        }
    }

    // ---- tests ----

    #[test]
    fn emit_float_add() {
        let text = emit_function(&float_add_graph()).unwrap();
        assert!(text.contains("func.func @add_f"), "missing func header");
        assert!(text.contains("arith.addf %arg0, %arg1 : tensor<f32>"), "missing arith.addf");
        assert!(text.contains("return %0 : tensor<f32>"), "missing return");
    }

    #[test]
    fn emit_int_add() {
        let text = emit_function(&int_add_graph()).unwrap();
        assert!(text.contains("func.func @add_i"), "missing func header");
        assert!(text.contains("arith.addi %arg0, %arg1 : tensor<i32>"), "missing arith.addi");
        assert!(text.contains("return %0 : tensor<i32>"), "missing return");
    }

    #[test]
    fn emit_float_comparison() {
        let g = FunctionGraph {
            name: "gt_f".into(),
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
        assert!(text.contains("arith.cmpf \"ogt\""), "missing arith.cmpf with ogt predicate");
        assert!(text.contains(": tensor<f32>"), "comparison should reference input type");
    }

    #[test]
    fn emit_int_comparison() {
        let g = FunctionGraph {
            name: "lt_i".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "a".into(), data_type: "i64".into() }),
                GraphNode::Param(Param { name: "b".into(), data_type: "i64".into() }),
                GraphNode::Apply { op: OpKind::Lt, output_type: "bool".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Copy },
            ],
            output_node: 2,
            return_type: "bool".into(),
        };
        let text = emit_function(&g).unwrap();
        assert!(text.contains("arith.cmpi \"slt\""), "missing arith.cmpi with slt predicate");
        assert!(text.contains(": tensor<i64>"), "comparison should reference input type");
    }

    #[test]
    fn emit_math_exp_unary() {
        let g = FunctionGraph {
            name: "my_exp".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("exp".into()), output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 1, port_in_idx: 0, wire_kind: WireKind::Copy },
            ],
            output_node: 1,
            return_type: "f32".into(),
        };
        let text = emit_function(&g).unwrap();
        assert!(text.contains("math.exp %arg0 : tensor<f32>"), "missing math.exp");
    }

    #[test]
    fn emit_neg_then_exp_chain() {
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
        assert!(text.contains("arith.negf %arg0 : tensor<f64>"), "missing arith.negf");
        assert!(text.contains("math.exp %0 : tensor<f64>"), "missing math.exp chained from neg");
        assert!(text.contains("return %1 : tensor<f64>"), "missing return");
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
        assert!(text.contains("arith.constant dense<1.0> : tensor<f32>"), "missing arith.constant");
        assert!(text.contains("arith.addf %arg0, %0 : tensor<f32>"), "missing arith.addf with constant");
    }

    #[test]
    fn emit_matmul_as_linalg() {
        let g = FunctionGraph {
            name: "mm".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "a".into(), data_type: "f32".into() }),
                GraphNode::Param(Param { name: "b".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("matmul".into()), output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Copy },
            ],
            output_node: 2,
            return_type: "f32".into(),
        };
        let text = emit_function(&g).unwrap();
        assert!(text.contains("linalg.matmul"), "missing linalg.matmul");
        assert!(text.contains("ins(%arg0, %arg1"), "missing matmul inputs");
    }

    #[test]
    fn reject_unmapped_custom_ops() {
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
        let err = emit_function(&g).unwrap_err();
        assert!(matches!(err, EmitError::UnmappedOp(_)));
    }

    #[test]
    fn write_to_file() {
        let g = float_add_graph();
        let tmp = std::env::temp_dir().join("rce_mlir_linalg_test.mlir");
        write_mlir_linalg(&g, &tmp).unwrap();
        let content = std::fs::read_to_string(&tmp).unwrap();
        assert!(content.contains("arith.addf"), "file should contain arith.addf");
        assert!(content.starts_with("module {"), "file should start with module");
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn module_wrapper() {
        let text = emit_module(&float_add_graph()).unwrap();
        assert!(text.starts_with("module {"), "module must open");
        assert!(text.contains("func.func @add_f"), "must contain func");
        assert!(text.trim_end().ends_with('}'), "module must close");
    }

    #[test]
    fn emit_relu_as_maximumf() {
        let g = FunctionGraph {
            name: "my_relu".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("relu".into()), output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 1, port_in_idx: 0, wire_kind: WireKind::Copy },
            ],
            output_node: 1,
            return_type: "f32".into(),
        };
        let text = emit_function(&g).unwrap();
        assert!(text.contains("arith.constant dense<0.0>"), "relu needs zero constant");
        assert!(text.contains("arith.maximumf"), "relu should use arith.maximumf");
    }

    #[test]
    fn emit_logic_not_as_xori() {
        let g = FunctionGraph {
            name: "my_not".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "bool".into() }),
                GraphNode::Apply { op: OpKind::Not, output_type: "bool".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 1, port_in_idx: 0, wire_kind: WireKind::Copy },
            ],
            output_node: 1,
            return_type: "bool".into(),
        };
        let text = emit_function(&g).unwrap();
        assert!(text.contains("arith.constant dense<true>"), "NOT needs all-ones constant");
        assert!(text.contains("arith.xori"), "NOT should use arith.xori");
    }

    #[test]
    fn emit_int_neg_as_subi() {
        let g = FunctionGraph {
            name: "neg_i".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "i32".into() }),
                GraphNode::Apply { op: OpKind::Neg, output_type: "i32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 1, port_in_idx: 0, wire_kind: WireKind::Copy },
            ],
            output_node: 1,
            return_type: "i32".into(),
        };
        let text = emit_function(&g).unwrap();
        assert!(text.contains("arith.constant dense<0>"), "int neg needs zero constant");
        assert!(text.contains("arith.subi"), "int neg should use arith.subi");
    }

    #[test]
    fn emit_empty_graph_errors() {
        let g = FunctionGraph {
            name: "empty".into(),
            nodes: vec![],
            wires: vec![],
            output_node: 0,
            return_type: "f32".into(),
        };
        assert!(matches!(emit_function(&g), Err(EmitError::EmptyGraph)));
    }

    #[test]
    fn emit_math_pow_binary() {
        let g = FunctionGraph {
            name: "my_pow".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "base".into(), data_type: "f32".into() }),
                GraphNode::Param(Param { name: "exp".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("pow".into()), output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Copy },
            ],
            output_node: 2,
            return_type: "f32".into(),
        };
        let text = emit_function(&g).unwrap();
        assert!(text.contains("math.powf %arg0, %arg1 : tensor<f32>"), "missing math.powf");
    }

    #[test]
    fn emit_i64_constant() {
        let g = FunctionGraph {
            name: "const_i64".into(),
            nodes: vec![
                GraphNode::Literal { value: LiteralValue::I64(42), data_type: "i64".into() },
            ],
            wires: vec![],
            output_node: 0,
            return_type: "i64".into(),
        };
        let text = emit_function(&g).unwrap();
        assert!(text.contains("arith.constant dense<42> : tensor<i64>"), "missing i64 constant");
    }

    #[test]
    fn emit_bool_constant() {
        let g = FunctionGraph {
            name: "const_bool".into(),
            nodes: vec![
                GraphNode::Literal { value: LiteralValue::Bool(true), data_type: "bool".into() },
            ],
            wires: vec![],
            output_node: 0,
            return_type: "bool".into(),
        };
        let text = emit_function(&g).unwrap();
        assert!(text.contains("arith.constant dense<true> : tensor<i1>"), "missing bool constant");
    }

    #[test]
    fn emit_conv_as_linalg() {
        let g = FunctionGraph {
            name: "my_conv".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "input".into(), data_type: "f32".into() }),
                GraphNode::Param(Param { name: "kernel".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("conv".into()), output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Copy },
            ],
            output_node: 2,
            return_type: "f32".into(),
        };
        let text = emit_function(&g).unwrap();
        assert!(text.contains("linalg.conv_2d_nhwc_hwcf"), "missing linalg.conv_2d_nhwc_hwcf");
    }
}
