//! Emit WGSL compute shaders from `rce_graph::FunctionGraph`.
//!
//! Produces valid WGSL compute kernels where each graph parameter becomes a
//! storage buffer binding and element-wise operations map to WGSL expressions.
//!
//! Output format:
//! ```wgsl
//! @group(0) @binding(0) var<storage, read> input_a: array<f32>;
//! @group(0) @binding(1) var<storage, read> input_b: array<f32>;
//! @group(0) @binding(2) var<storage, read_write> output: array<f32>;
//!
//! @compute @workgroup_size(64)
//! fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
//!     let a = input_a[gid.x];
//!     let b = input_b[gid.x];
//!     let result = a + b;
//!     output[gid.x] = result;
//! }
//! ```

use std::collections::HashMap;
use std::fmt::Write;
use std::path::Path;

use crate::op_map::{wgsl_op_for, WgslOp};
use crate::types::wgsl_type;
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

/// Emit a complete WGSL module: buffer declarations + compute kernel.
pub fn emit_module(graph: &FunctionGraph) -> Result<String, EmitError> {
    if graph.nodes.is_empty() {
        return Err(EmitError::EmptyGraph);
    }

    let mut out = String::with_capacity(2048);

    // Collect params for buffer declarations.
    let mut params: Vec<(usize, String, String)> = Vec::new(); // (node_idx, name, wgsl_type)
    for (idx, node) in graph.nodes.iter().enumerate() {
        if let GraphNode::Param(p) = node {
            let ty = wgsl_type(&p.data_type)
                .ok_or_else(|| EmitError::UnsupportedType(p.data_type.clone()))?;
            params.push((idx, p.name.clone(), ty.to_string()));
        }
    }

    let ret_ty = wgsl_type(&graph.return_type)
        .ok_or_else(|| EmitError::UnsupportedType(graph.return_type.clone()))?;

    // Emit storage buffer bindings for each parameter.
    for (binding_idx, (_node_idx, name, ty)) in params.iter().enumerate() {
        writeln!(
            out,
            "@group(0) @binding({binding_idx}) var<storage, read> input_{name}: array<{ty}>;",
        )
        .unwrap();
    }

    // Output buffer binding.
    let output_binding = params.len();
    writeln!(
        out,
        "@group(0) @binding({output_binding}) var<storage, read_write> output: array<{ret_ty}>;",
    )
    .unwrap();

    writeln!(out).unwrap();

    // Emit the kernel body.
    let kernel = emit_kernel(graph)?;
    out.push_str(&kernel);

    Ok(out)
}

/// Emit a single `@compute` kernel function for the graph.
pub fn emit_kernel(graph: &FunctionGraph) -> Result<String, EmitError> {
    if graph.nodes.is_empty() {
        return Err(EmitError::EmptyGraph);
    }

    let mut out = String::with_capacity(1024);
    let mut locals: HashMap<usize, String> = HashMap::new();

    // Collect params and their WGSL names.
    let mut param_names: Vec<(usize, String)> = Vec::new();
    for (idx, node) in graph.nodes.iter().enumerate() {
        if let GraphNode::Param(p) = node {
            let local_name = p.name.clone();
            param_names.push((idx, local_name));
        }
    }

    // Function header.
    writeln!(
        out,
        "@compute @workgroup_size(64)\nfn {}(@builtin(global_invocation_id) gid: vec3<u32>) {{",
        graph.name,
    )
    .unwrap();

    // Load each param from its buffer.
    for (idx, name) in &param_names {
        writeln!(out, "    let {name} = input_{name}[gid.x];").unwrap();
        locals.insert(*idx, name.clone());
    }

    // Build incoming-wire index per node.
    let mut inputs_for: HashMap<usize, Vec<(i32, usize)>> = HashMap::new();
    for w in &graph.wires {
        inputs_for.entry(w.to).or_default().push((w.port_in_idx, w.from));
    }

    // SSA counter for intermediate values.
    let mut val_idx = 0usize;
    let val_name = |i: &mut usize| -> String {
        let n = format!("v{i}");
        *i += 1;
        n
    };

    // Emit nodes in order (skip params, already handled).
    for (idx, node) in graph.nodes.iter().enumerate() {
        match node {
            GraphNode::Param(_) => {} // already handled as buffer loads
            GraphNode::Literal { value, data_type } => {
                let _ty = wgsl_type(data_type)
                    .ok_or_else(|| EmitError::UnsupportedType(data_type.clone()))?;
                let name = val_name(&mut val_idx);
                let lit = format_literal(value);
                writeln!(out, "    let {name} = {lit};").unwrap();
                locals.insert(idx, name);
            }
            GraphNode::Apply { op, output_type } => {
                let _ty = wgsl_type(output_type)
                    .ok_or_else(|| EmitError::UnsupportedType(output_type.clone()))?;
                let wgsl = wgsl_op_for(op);
                let ins = resolve_inputs(&inputs_for, &locals, idx)?;
                let name = val_name(&mut val_idx);

                match wgsl {
                    WgslOp::BinaryOp(sym) => {
                        let a = ins.get(0).ok_or(EmitError::MissingInput { node: idx, port: 0 })?;
                        let b = ins.get(1).ok_or(EmitError::MissingInput { node: idx, port: 1 })?;
                        writeln!(out, "    let {name} = {a} {sym} {b};").unwrap();
                    }
                    WgslOp::UnaryOp(sym) => {
                        let a = ins.get(0).ok_or(EmitError::MissingInput { node: idx, port: 0 })?;
                        writeln!(out, "    let {name} = {sym}{a};").unwrap();
                    }
                    WgslOp::BuiltinUnary(func) => {
                        let a = ins.get(0).ok_or(EmitError::MissingInput { node: idx, port: 0 })?;
                        writeln!(out, "    let {name} = {func}({a});").unwrap();
                    }
                    WgslOp::BuiltinBinary(func) => {
                        let a = ins.get(0).ok_or(EmitError::MissingInput { node: idx, port: 0 })?;
                        let b = ins.get(1).ok_or(EmitError::MissingInput { node: idx, port: 1 })?;
                        writeln!(out, "    let {name} = {func}({a}, {b});").unwrap();
                    }
                    WgslOp::BuiltinTernary(func) => {
                        let a = ins.get(0).ok_or(EmitError::MissingInput { node: idx, port: 0 })?;
                        let b = ins.get(1).ok_or(EmitError::MissingInput { node: idx, port: 1 })?;
                        let c = ins.get(2).ok_or(EmitError::MissingInput { node: idx, port: 2 })?;
                        writeln!(out, "    let {name} = {func}({a}, {b}, {c});").unwrap();
                    }
                    WgslOp::Relu => {
                        let a = ins.get(0).ok_or(EmitError::MissingInput { node: idx, port: 0 })?;
                        writeln!(out, "    let {name} = max({a}, 0.0);").unwrap();
                    }
                    WgslOp::Sigmoid => {
                        let a = ins.get(0).ok_or(EmitError::MissingInput { node: idx, port: 0 })?;
                        writeln!(out, "    let {name} = 1.0 / (1.0 + exp(-{a}));").unwrap();
                    }
                    WgslOp::Unmapped => {
                        return Err(EmitError::UnmappedOp(format!("{op:?}")));
                    }
                }
                locals.insert(idx, name);
            }
            other => {
                return Err(EmitError::UnsupportedNode(
                    format!("{:?}", std::mem::discriminant(other)),
                ));
            }
        }
    }

    // Write output.
    let result_name = locals
        .get(&graph.output_node)
        .cloned()
        .unwrap_or_else(|| "v0".into());
    writeln!(out, "    output[gid.x] = {result_name};").unwrap();
    writeln!(out, "}}").unwrap();

    Ok(out)
}

/// Write WGSL to a file.
pub fn write_wgsl(graph: &FunctionGraph, path: impl AsRef<Path>) -> Result<(), EmitError> {
    let text = emit_module(graph)?;
    std::fs::write(path, text)?;
    Ok(())
}

fn resolve_inputs(
    inputs_for: &HashMap<usize, Vec<(i32, usize)>>,
    locals: &HashMap<usize, String>,
    target: usize,
) -> Result<Vec<String>, EmitError> {
    let mut pairs: Vec<(i32, usize)> = inputs_for
        .get(&target)
        .cloned()
        .unwrap_or_default();
    pairs.sort_by_key(|(port, _)| *port);
    pairs
        .iter()
        .enumerate()
        .map(|(i, (_, src))| {
            locals
                .get(src)
                .cloned()
                .ok_or(EmitError::MissingInput { node: target, port: i as i32 })
        })
        .collect()
}

fn format_literal(value: &LiteralValue) -> String {
    match value {
        LiteralValue::I32(v) => format!("{v}i"),
        LiteralValue::I64(v) => format!("{v}i"),
        LiteralValue::F32(v) => format_wgsl_float(*v as f64),
        LiteralValue::F64(v) => format_wgsl_float(*v),
        LiteralValue::Bool(true) => "true".into(),
        LiteralValue::Bool(false) => "false".into(),
        LiteralValue::String(s) => {
            // Strings have no WGSL representation; emit as a comment placeholder.
            format!("/* \"{s}\" */")
        }
    }
}

/// Format a float for WGSL (always has decimal point, never scientific for small values).
fn format_wgsl_float(v: f64) -> String {
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
    fn emit_simple_add_kernel() {
        let text = emit_kernel(&add_graph()).unwrap();
        assert!(text.contains("@compute @workgroup_size(64)"));
        assert!(text.contains("fn add("));
        assert!(text.contains("let a = input_a[gid.x];"));
        assert!(text.contains("let b = input_b[gid.x];"));
        assert!(text.contains("let v0 = a + b;"));
        assert!(text.contains("output[gid.x] = v0;"));
    }

    #[test]
    fn emit_add_module_with_bindings() {
        let text = emit_module(&add_graph()).unwrap();
        assert!(text.contains("@group(0) @binding(0) var<storage, read> input_a: array<f32>;"));
        assert!(text.contains("@group(0) @binding(1) var<storage, read> input_b: array<f32>;"));
        assert!(text.contains("@group(0) @binding(2) var<storage, read_write> output: array<f32>;"));
        assert!(text.contains("@compute @workgroup_size(64)"));
    }

    #[test]
    fn emit_mul_kernel() {
        let g = FunctionGraph {
            name: "mul".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
                GraphNode::Param(Param { name: "y".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Mul, output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Copy },
            ],
            output_node: 2,
            return_type: "f32".into(),
        };
        let text = emit_kernel(&g).unwrap();
        assert!(text.contains("let v0 = x * y;"));
        assert!(text.contains("output[gid.x] = v0;"));
    }

    #[test]
    fn emit_comparison_kernel() {
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
        let text = emit_kernel(&g).unwrap();
        assert!(text.contains("let v0 = a > b;"));
    }

    #[test]
    fn emit_with_constant_literal() {
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
        let text = emit_kernel(&g).unwrap();
        assert!(text.contains("let v0 = 1.0;"));
        assert!(text.contains("let v1 = x + v0;"));
    }

    #[test]
    fn emit_unary_chain_neg_exp() {
        // exp(neg(x))
        let g = FunctionGraph {
            name: "exp_neg".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Neg, output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("exp".into()), output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 1, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 2, port_in_idx: 0, wire_kind: WireKind::Copy },
            ],
            output_node: 2,
            return_type: "f32".into(),
        };
        let text = emit_kernel(&g).unwrap();
        assert!(text.contains("let v0 = -x;"));
        assert!(text.contains("let v1 = exp(v0);"));
        assert!(text.contains("output[gid.x] = v1;"));
    }

    #[test]
    fn emit_relu_via_custom_op() {
        let g = FunctionGraph {
            name: "relu_kernel".into(),
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
        let text = emit_kernel(&g).unwrap();
        assert!(text.contains("let v0 = max(x, 0.0);"));
    }

    #[test]
    fn emit_sigmoid_via_custom_op() {
        let g = FunctionGraph {
            name: "sigmoid_kernel".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("sigmoid".into()), output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 1, port_in_idx: 0, wire_kind: WireKind::Copy },
            ],
            output_node: 1,
            return_type: "f32".into(),
        };
        let text = emit_kernel(&g).unwrap();
        assert!(text.contains("let v0 = 1.0 / (1.0 + exp(-x));"));
    }

    #[test]
    fn reject_unmapped_ops() {
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
        let err = emit_kernel(&g).unwrap_err();
        assert!(err.to_string().contains("unmapped op"));
    }

    #[test]
    fn reject_matmul_unmapped() {
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
        assert!(emit_kernel(&g).is_err());
    }

    #[test]
    fn write_wgsl_to_file() {
        let g = add_graph();
        let tmp = std::env::temp_dir().join("rce_wgsl_test.wgsl");
        write_wgsl(&g, &tmp).unwrap();
        let content = std::fs::read_to_string(&tmp).unwrap();
        assert!(content.contains("@compute @workgroup_size(64)"));
        assert!(content.contains("@group(0) @binding(0)"));
        assert!(content.contains("input_a"));
        std::fs::remove_file(&tmp).ok();
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
        assert!(emit_kernel(&g).is_err());
        assert!(emit_module(&g).is_err());
    }

    #[test]
    fn emit_f64_demotes_to_f32() {
        let g = FunctionGraph {
            name: "add_double".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "a".into(), data_type: "f64".into() }),
                GraphNode::Param(Param { name: "b".into(), data_type: "f64".into() }),
                GraphNode::Apply { op: OpKind::Add, output_type: "f64".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Copy },
            ],
            output_node: 2,
            return_type: "f64".into(),
        };
        let text = emit_module(&g).unwrap();
        // f64 demotes to f32 in WGSL
        assert!(text.contains("array<f32>"));
        assert!(!text.contains("f64"));
    }

    #[test]
    fn emit_integer_literal() {
        let g = FunctionGraph {
            name: "add_int".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "i32".into() }),
                GraphNode::Literal { value: LiteralValue::I32(42), data_type: "i32".into() },
                GraphNode::Apply { op: OpKind::Add, output_type: "i32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Copy },
            ],
            output_node: 2,
            return_type: "i32".into(),
        };
        let text = emit_kernel(&g).unwrap();
        assert!(text.contains("let v0 = 42i;"));
        assert!(text.contains("let v1 = x + v0;"));
    }

    #[test]
    fn emit_bool_literal() {
        let g = FunctionGraph {
            name: "and_const".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "bool".into() }),
                GraphNode::Literal { value: LiteralValue::Bool(true), data_type: "bool".into() },
                GraphNode::Apply { op: OpKind::And, output_type: "bool".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Copy },
            ],
            output_node: 2,
            return_type: "bool".into(),
        };
        let text = emit_kernel(&g).unwrap();
        assert!(text.contains("let v0 = true;"));
        assert!(text.contains("let v1 = x && v0;"));
    }

    #[test]
    fn emit_chained_sub() {
        // (a - b) - c
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
        let text = emit_kernel(&g).unwrap();
        assert!(text.contains("let v0 = a - b;"));
        assert!(text.contains("let v1 = v0 - c;"));
    }

    #[test]
    fn emit_unsupported_node_errors() {
        let g = FunctionGraph {
            name: "bad".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
                GraphNode::Select { output_type: "f32".into() },
            ],
            wires: vec![],
            output_node: 1,
            return_type: "f32".into(),
        };
        let err = emit_kernel(&g).unwrap_err();
        assert!(err.to_string().contains("unsupported node"));
    }
}
