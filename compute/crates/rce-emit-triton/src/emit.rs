//! Emit Triton Python DSL from `rce_graph::FunctionGraph`.
//!
//! Produces a self-contained Triton kernel: each `Param` becomes a pointer
//! argument with `tl.load`, each `Apply` becomes a Python expression, and
//! the output node becomes a `tl.store`. The boilerplate `pid`, `offsets`,
//! and `mask` pattern is generated automatically.
//!
//! Output format:
//! ```python
//! import triton
//! import triton.language as tl
//!
//! @triton.jit
//! def add_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
//!     pid = tl.program_id(axis=0)
//!     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
//!     mask = offsets < n_elements
//!     a = tl.load(a_ptr + offsets, mask=mask)
//!     b = tl.load(b_ptr + offsets, mask=mask)
//!     result = a + b
//!     tl.store(output_ptr + offsets, result, mask=mask)
//! ```

use std::collections::HashMap;
use std::fmt::Write;
use std::path::Path;

use crate::op_map::{triton_op_for, TritonOp};
use crate::types::triton_type;
use rce_graph::graph::{FunctionGraph, GraphNode, LiteralValue, OpKind};

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

/// Emit a complete Triton Python file containing one kernel.
pub fn emit_kernel(graph: &FunctionGraph) -> Result<String, EmitError> {
    if graph.nodes.is_empty() {
        return Err(EmitError::EmptyGraph);
    }

    let mut out = String::with_capacity(2048);

    // -- Header ----------------------------------------------------------
    writeln!(out, "import triton").unwrap();
    writeln!(out, "import triton.language as tl").unwrap();
    writeln!(out).unwrap();
    writeln!(out).unwrap();

    // -- Collect params --------------------------------------------------
    let mut params: Vec<(usize, String, String)> = Vec::new(); // (node_idx, name, data_type)
    for (idx, node) in graph.nodes.iter().enumerate() {
        if let GraphNode::Param(p) = node {
            // Validate the type is known.
            triton_type(&p.data_type)
                .ok_or_else(|| EmitError::UnsupportedType(p.data_type.clone()))?;
            params.push((idx, p.name.clone(), p.data_type.clone()));
        }
    }

    // -- Function signature ----------------------------------------------
    let mut sig_parts: Vec<String> = params
        .iter()
        .map(|(_, name, _)| format!("{name}_ptr"))
        .collect();
    sig_parts.push("output_ptr".into());
    sig_parts.push("n_elements".into());
    sig_parts.push("BLOCK_SIZE: tl.constexpr".into());

    writeln!(out, "@triton.jit").unwrap();
    writeln!(
        out,
        "def {}_kernel({}):",
        graph.name,
        sig_parts.join(", "),
    )
    .unwrap();

    let indent = "    ";

    // -- Boilerplate: pid, offsets, mask ----------------------------------
    writeln!(out, "{indent}pid = tl.program_id(axis=0)").unwrap();
    writeln!(
        out,
        "{indent}offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
    )
    .unwrap();
    writeln!(out, "{indent}mask = offsets < n_elements").unwrap();

    // -- Load each param -------------------------------------------------
    let mut var_names: HashMap<usize, String> = HashMap::new();
    for (idx, name, _dt) in &params {
        writeln!(
            out,
            "{indent}{name} = tl.load({name}_ptr + offsets, mask=mask)",
        )
        .unwrap();
        var_names.insert(*idx, name.clone());
    }

    // -- Build wire index ------------------------------------------------
    let mut inputs_for: HashMap<usize, Vec<(i32, usize)>> = HashMap::new();
    for w in &graph.wires {
        inputs_for
            .entry(w.to)
            .or_default()
            .push((w.port_in_idx, w.from));
    }

    // -- Emit nodes in topological order ---------------------------------
    let mut tmp_idx = 0usize;
    let next_tmp = |i: &mut usize| -> String {
        let n = format!("v{i}");
        *i += 1;
        n
    };

    for (idx, node) in graph.nodes.iter().enumerate() {
        match node {
            GraphNode::Param(_) => {} // already handled
            GraphNode::Literal { value, data_type } => {
                // Validate type.
                triton_type(data_type)
                    .ok_or_else(|| EmitError::UnsupportedType(data_type.clone()))?;
                let name = next_tmp(&mut tmp_idx);
                let lit = format_literal(value);
                writeln!(out, "{indent}{name} = {lit}").unwrap();
                var_names.insert(idx, name);
            }
            GraphNode::Apply { op, output_type } => {
                let triton = triton_op_for(op);
                let ins = resolve_inputs(&inputs_for, &var_names, idx)?;
                let name = next_tmp(&mut tmp_idx);

                match triton {
                    TritonOp::BinaryInfix(sym) => {
                        let a = get_input(&ins, 0, idx)?;
                        let b = get_input(&ins, 1, idx)?;
                        writeln!(out, "{indent}{name} = {a} {sym} {b}").unwrap();
                    }
                    TritonOp::UnaryPrefix(sym) => {
                        let a = get_input(&ins, 0, idx)?;
                        writeln!(out, "{indent}{name} = {sym}{a}").unwrap();
                    }
                    TritonOp::UnaryCall(func) => {
                        let a = get_input(&ins, 0, idx)?;
                        writeln!(out, "{indent}{name} = {func}({a})").unwrap();
                    }
                    TritonOp::BinaryCall(func) => {
                        // Special case: relu uses tl.maximum(x, 0.0)
                        if matches!(op, OpKind::Custom(s) if s == "relu" || s == "Relu") {
                            let a = get_input(&ins, 0, idx)?;
                            writeln!(out, "{indent}{name} = {func}({a}, 0.0)").unwrap();
                        } else {
                            let a = get_input(&ins, 0, idx)?;
                            let b = get_input(&ins, 1, idx)?;
                            writeln!(out, "{indent}{name} = {func}({a}, {b})").unwrap();
                        }
                    }
                    TritonOp::TernaryCall(func) => {
                        let a = get_input(&ins, 0, idx)?;
                        let b = get_input(&ins, 1, idx)?;
                        let c = get_input(&ins, 2, idx)?;
                        writeln!(out, "{indent}{name} = {func}({a}, {b}, {c})").unwrap();
                    }
                    TritonOp::Cast => {
                        let a = get_input(&ins, 0, idx)?;
                        let ty = triton_type(output_type)
                            .ok_or_else(|| EmitError::UnsupportedType(output_type.clone()))?;
                        writeln!(out, "{indent}{name} = {a}.to({ty})").unwrap();
                    }
                    TritonOp::Unmapped => {
                        return Err(EmitError::UnmappedOp(format!("{op:?}")));
                    }
                }
                var_names.insert(idx, name);
            }
            other => {
                return Err(EmitError::UnsupportedNode(format!(
                    "{:?}",
                    std::mem::discriminant(other),
                )));
            }
        }
    }

    // -- Store output ----------------------------------------------------
    let result_var = var_names
        .get(&graph.output_node)
        .cloned()
        .unwrap_or_else(|| "v0".into());
    writeln!(
        out,
        "{indent}tl.store(output_ptr + offsets, {result_var}, mask=mask)",
    )
    .unwrap();

    Ok(out)
}

/// Write a Triton kernel to a `.py` file.
pub fn write_triton(graph: &FunctionGraph, path: impl AsRef<Path>) -> Result<(), EmitError> {
    let text = emit_kernel(graph)?;
    std::fs::write(path, text)?;
    Ok(())
}

// ── Helpers ────────────────────────────────────────────────────────────

fn resolve_inputs(
    inputs_for: &HashMap<usize, Vec<(i32, usize)>>,
    var_names: &HashMap<usize, String>,
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
            var_names
                .get(src)
                .cloned()
                .ok_or(EmitError::MissingInput {
                    node: target,
                    port: 0,
                })
        })
        .collect()
}

fn get_input(ins: &[String], port: usize, node: usize) -> Result<&str, EmitError> {
    ins.get(port)
        .map(|s| s.as_str())
        .ok_or(EmitError::MissingInput {
            node,
            port: port as i32,
        })
}

fn format_literal(value: &LiteralValue) -> String {
    match value {
        LiteralValue::I32(v) => format!("{v}"),
        LiteralValue::I64(v) => format!("{v}"),
        LiteralValue::F32(v) => format_float(*v as f64),
        LiteralValue::F64(v) => format_float(*v),
        LiteralValue::Bool(true) => "True".into(),
        LiteralValue::Bool(false) => "False".into(),
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
    use rce_graph::graph::{FunctionGraph, GraphNode, LiteralValue, OpKind, Param, Wire, WireKind};

    fn add_graph() -> FunctionGraph {
        FunctionGraph {
            name: "add".into(),
            nodes: vec![
                GraphNode::Param(Param {
                    name: "a".into(),
                    data_type: "f32".into(),
                }),
                GraphNode::Param(Param {
                    name: "b".into(),
                    data_type: "f32".into(),
                }),
                GraphNode::Apply {
                    op: OpKind::Add,
                    output_type: "f32".into(),
                },
            ],
            wires: vec![
                Wire {
                    from: 0,
                    to: 2,
                    port_in_idx: 0,
                    wire_kind: WireKind::Copy,
                },
                Wire {
                    from: 1,
                    to: 2,
                    port_in_idx: 1,
                    wire_kind: WireKind::Copy,
                },
            ],
            output_node: 2,
            return_type: "f32".into(),
        }
    }

    #[test]
    fn emit_add_kernel_structure() {
        let text = emit_kernel(&add_graph()).unwrap();
        assert!(text.contains("def add_kernel("));
        assert!(text.contains("a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr"));
        assert!(text.contains("a = tl.load(a_ptr + offsets, mask=mask)"));
        assert!(text.contains("b = tl.load(b_ptr + offsets, mask=mask)"));
        assert!(text.contains("v0 = a + b"));
        assert!(text.contains("tl.store(output_ptr + offsets, v0, mask=mask)"));
    }

    #[test]
    fn emit_with_constant() {
        let g = FunctionGraph {
            name: "add_one".into(),
            nodes: vec![
                GraphNode::Param(Param {
                    name: "x".into(),
                    data_type: "f32".into(),
                }),
                GraphNode::Literal {
                    value: LiteralValue::F32(1.0),
                    data_type: "f32".into(),
                },
                GraphNode::Apply {
                    op: OpKind::Add,
                    output_type: "f32".into(),
                },
            ],
            wires: vec![
                Wire {
                    from: 0,
                    to: 2,
                    port_in_idx: 0,
                    wire_kind: WireKind::Copy,
                },
                Wire {
                    from: 1,
                    to: 2,
                    port_in_idx: 1,
                    wire_kind: WireKind::Copy,
                },
            ],
            output_node: 2,
            return_type: "f32".into(),
        };
        let text = emit_kernel(&g).unwrap();
        assert!(text.contains("v0 = 1.0"));
        assert!(text.contains("v1 = x + v0"));
    }

    #[test]
    fn emit_unary_chain() {
        // exp(neg(x))
        let g = FunctionGraph {
            name: "exp_neg".into(),
            nodes: vec![
                GraphNode::Param(Param {
                    name: "x".into(),
                    data_type: "f64".into(),
                }),
                GraphNode::Apply {
                    op: OpKind::Neg,
                    output_type: "f64".into(),
                },
                GraphNode::Apply {
                    op: OpKind::Custom("exp".into()),
                    output_type: "f64".into(),
                },
            ],
            wires: vec![
                Wire {
                    from: 0,
                    to: 1,
                    port_in_idx: 0,
                    wire_kind: WireKind::Copy,
                },
                Wire {
                    from: 1,
                    to: 2,
                    port_in_idx: 0,
                    wire_kind: WireKind::Copy,
                },
            ],
            output_node: 2,
            return_type: "f64".into(),
        };
        let text = emit_kernel(&g).unwrap();
        assert!(text.contains("v0 = -x"));
        assert!(text.contains("v1 = tl.exp(v0)"));
    }

    #[test]
    fn emit_comparison() {
        let g = FunctionGraph {
            name: "gt".into(),
            nodes: vec![
                GraphNode::Param(Param {
                    name: "a".into(),
                    data_type: "f32".into(),
                }),
                GraphNode::Param(Param {
                    name: "b".into(),
                    data_type: "f32".into(),
                }),
                GraphNode::Apply {
                    op: OpKind::Gt,
                    output_type: "bool".into(),
                },
            ],
            wires: vec![
                Wire {
                    from: 0,
                    to: 2,
                    port_in_idx: 0,
                    wire_kind: WireKind::Copy,
                },
                Wire {
                    from: 1,
                    to: 2,
                    port_in_idx: 1,
                    wire_kind: WireKind::Copy,
                },
            ],
            output_node: 2,
            return_type: "bool".into(),
        };
        let text = emit_kernel(&g).unwrap();
        assert!(text.contains("v0 = a > b"));
    }

    #[test]
    fn emit_relu_via_custom() {
        let g = FunctionGraph {
            name: "relu".into(),
            nodes: vec![
                GraphNode::Param(Param {
                    name: "x".into(),
                    data_type: "f32".into(),
                }),
                GraphNode::Apply {
                    op: OpKind::Custom("relu".into()),
                    output_type: "f32".into(),
                },
            ],
            wires: vec![Wire {
                from: 0,
                to: 1,
                port_in_idx: 0,
                wire_kind: WireKind::Copy,
            }],
            output_node: 1,
            return_type: "f32".into(),
        };
        let text = emit_kernel(&g).unwrap();
        assert!(text.contains("v0 = tl.maximum(x, 0.0)"));
    }

    #[test]
    fn emit_sigmoid_via_custom() {
        let g = FunctionGraph {
            name: "sigm".into(),
            nodes: vec![
                GraphNode::Param(Param {
                    name: "x".into(),
                    data_type: "f32".into(),
                }),
                GraphNode::Apply {
                    op: OpKind::Custom("sigmoid".into()),
                    output_type: "f32".into(),
                },
            ],
            wires: vec![Wire {
                from: 0,
                to: 1,
                port_in_idx: 0,
                wire_kind: WireKind::Copy,
            }],
            output_node: 1,
            return_type: "f32".into(),
        };
        let text = emit_kernel(&g).unwrap();
        assert!(text.contains("v0 = tl.sigmoid(x)"));
    }

    #[test]
    fn reject_unmapped_matmul() {
        let g = FunctionGraph {
            name: "bad".into(),
            nodes: vec![
                GraphNode::Param(Param {
                    name: "a".into(),
                    data_type: "f32".into(),
                }),
                GraphNode::Param(Param {
                    name: "b".into(),
                    data_type: "f32".into(),
                }),
                GraphNode::Apply {
                    op: OpKind::Custom("matmul".into()),
                    output_type: "f32".into(),
                },
            ],
            wires: vec![
                Wire {
                    from: 0,
                    to: 2,
                    port_in_idx: 0,
                    wire_kind: WireKind::Copy,
                },
                Wire {
                    from: 1,
                    to: 2,
                    port_in_idx: 1,
                    wire_kind: WireKind::Copy,
                },
            ],
            output_node: 2,
            return_type: "f32".into(),
        };
        let err = emit_kernel(&g).unwrap_err();
        assert!(err.to_string().contains("unmapped op"));
    }

    #[test]
    fn reject_unmapped_softmax() {
        let g = FunctionGraph {
            name: "bad".into(),
            nodes: vec![
                GraphNode::Param(Param {
                    name: "x".into(),
                    data_type: "f32".into(),
                }),
                GraphNode::Apply {
                    op: OpKind::Custom("softmax".into()),
                    output_type: "f32".into(),
                },
            ],
            wires: vec![Wire {
                from: 0,
                to: 1,
                port_in_idx: 0,
                wire_kind: WireKind::Copy,
            }],
            output_node: 1,
            return_type: "f32".into(),
        };
        let err = emit_kernel(&g).unwrap_err();
        assert!(err.to_string().contains("unmapped op"));
    }

    #[test]
    fn write_triton_file() {
        let g = add_graph();
        let tmp = std::env::temp_dir().join("rce_triton_test.py");
        write_triton(&g, &tmp).unwrap();
        let content = std::fs::read_to_string(&tmp).unwrap();
        assert!(content.contains("def add_kernel("));
        assert!(content.contains("tl.store"));
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn verify_import_header() {
        let text = emit_kernel(&add_graph()).unwrap();
        assert!(text.starts_with("import triton\nimport triton.language as tl\n"));
    }

    #[test]
    fn verify_jit_decorator() {
        let text = emit_kernel(&add_graph()).unwrap();
        assert!(text.contains("@triton.jit"));
    }

    #[test]
    fn verify_mask_pattern() {
        let text = emit_kernel(&add_graph()).unwrap();
        assert!(text.contains("pid = tl.program_id(axis=0)"));
        assert!(text.contains("offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)"));
        assert!(text.contains("mask = offsets < n_elements"));
    }

    #[test]
    fn emit_cast() {
        let g = FunctionGraph {
            name: "to_f16".into(),
            nodes: vec![
                GraphNode::Param(Param {
                    name: "x".into(),
                    data_type: "f32".into(),
                }),
                GraphNode::Apply {
                    op: OpKind::Custom("cast".into()),
                    output_type: "f16".into(),
                },
            ],
            wires: vec![Wire {
                from: 0,
                to: 1,
                port_in_idx: 0,
                wire_kind: WireKind::Copy,
            }],
            output_node: 1,
            return_type: "f16".into(),
        };
        let text = emit_kernel(&g).unwrap();
        assert!(text.contains("v0 = x.to(tl.float16)"));
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
    }

    #[test]
    fn emit_bool_literal() {
        let g = FunctionGraph {
            name: "with_bool".into(),
            nodes: vec![
                GraphNode::Param(Param {
                    name: "x".into(),
                    data_type: "f32".into(),
                }),
                GraphNode::Literal {
                    value: LiteralValue::Bool(true),
                    data_type: "bool".into(),
                },
                GraphNode::Param(Param {
                    name: "y".into(),
                    data_type: "f32".into(),
                }),
                GraphNode::Apply {
                    op: OpKind::Custom("where".into()),
                    output_type: "f32".into(),
                },
            ],
            wires: vec![
                Wire {
                    from: 1,
                    to: 3,
                    port_in_idx: 0,
                    wire_kind: WireKind::Copy,
                },
                Wire {
                    from: 0,
                    to: 3,
                    port_in_idx: 1,
                    wire_kind: WireKind::Copy,
                },
                Wire {
                    from: 2,
                    to: 3,
                    port_in_idx: 2,
                    wire_kind: WireKind::Copy,
                },
            ],
            output_node: 3,
            return_type: "f32".into(),
        };
        let text = emit_kernel(&g).unwrap();
        assert!(text.contains("v0 = True"));
        assert!(text.contains("tl.where(v0, x, y)"));
    }
}
