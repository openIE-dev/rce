//! Convert `FxGraph` intermediate representation into RCE `FunctionGraph`.

use std::collections::HashMap;

use rce_graph::graph::{
    FunctionGraph, GraphNode, LiteralValue, OpKind, Param, TensorLiteralData, Wire, WireKind,
};
use std::sync::Arc;

use crate::op_map::{map_function_target, map_method_name};
use crate::parse::{FxArg, FxGraph, FxOp};
use crate::IngestError;

/// Default data type when FX graph does not provide explicit type info.
const DEFAULT_TYPE: &str = "f32";

/// Convert a parsed `FxGraph` into a RCE `FunctionGraph`.
pub fn to_function_graph(fx: &FxGraph) -> Result<FunctionGraph, IngestError> {
    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut wires: Vec<Wire> = Vec::new();

    // Map FX node name → index in `nodes` vec.
    let mut name_to_idx: HashMap<&str, usize> = HashMap::new();
    let mut output_node: Option<usize> = None;

    for fx_node in &fx.nodes {
        match fx_node.op {
            FxOp::Placeholder => {
                let idx = nodes.len();
                nodes.push(GraphNode::Param(Param {
                    name: fx_node.name.clone(),
                    data_type: DEFAULT_TYPE.into(),
                }));
                name_to_idx.insert(&fx_node.name, idx);
            }
            FxOp::GetAttr => {
                let idx = nodes.len();
                // If the FX JSON enriches this get_attr with shape + dtype,
                // emit a TensorLiteral whose data is reattached at load time
                // from the safetensors sidecar (this is the path PyTorch
                // exporters use to ship trained weights through rce).
                // Otherwise fall back to a Param node — the consumer must
                // bind the value as a runtime input.
                if let (Some(shape), Some(dtype)) = (&fx_node.shape, &fx_node.dtype) {
                    nodes.push(GraphNode::TensorLiteral(TensorLiteralData {
                        data: Arc::from(Vec::<u8>::new().into_boxed_slice()),
                        shape: shape.clone(),
                        dtype: dtype.clone(),
                        name: fx_node.target.clone(),
                    }));
                } else {
                    nodes.push(GraphNode::Param(Param {
                        name: fx_node.target.clone(),
                        data_type: DEFAULT_TYPE.into(),
                    }));
                }
                name_to_idx.insert(&fx_node.name, idx);
            }
            FxOp::CallFunction => {
                let op = map_function_target(&fx_node.target);
                let apply_idx = nodes.len();
                nodes.push(GraphNode::Apply {
                    op,
                    output_type: DEFAULT_TYPE.into(),
                });
                name_to_idx.insert(&fx_node.name, apply_idx);

                emit_arg_wires(
                    &fx_node.args,
                    apply_idx,
                    &name_to_idx,
                    &mut nodes,
                    &mut wires,
                )?;
            }
            FxOp::CallMethod => {
                let op = map_method_name(&fx_node.target);
                let apply_idx = nodes.len();
                nodes.push(GraphNode::Apply {
                    op,
                    output_type: DEFAULT_TYPE.into(),
                });
                name_to_idx.insert(&fx_node.name, apply_idx);

                emit_arg_wires(
                    &fx_node.args,
                    apply_idx,
                    &name_to_idx,
                    &mut nodes,
                    &mut wires,
                )?;
            }
            FxOp::CallModule => {
                // call_module target is the module path (e.g. "layer1.conv").
                // We map it as a Custom op with the module path.
                let op = OpKind::Custom(fx_node.target.clone());
                let apply_idx = nodes.len();
                nodes.push(GraphNode::Apply {
                    op,
                    output_type: DEFAULT_TYPE.into(),
                });
                name_to_idx.insert(&fx_node.name, apply_idx);

                emit_arg_wires(
                    &fx_node.args,
                    apply_idx,
                    &name_to_idx,
                    &mut nodes,
                    &mut wires,
                )?;
            }
            FxOp::Output => {
                // The output node's args reference the actual output node(s).
                // We find the referenced node and mark it as output.
                if let Some(first_arg) = fx_node.args.first() {
                    match first_arg {
                        FxArg::Name(name) => {
                            if let Some(&idx) = name_to_idx.get(name.as_str()) {
                                output_node = Some(idx);
                            } else {
                                return Err(IngestError::NodeNotFound(name.clone()));
                            }
                        }
                        FxArg::List(items) => {
                            // Multi-output: take the first for now.
                            if let Some(FxArg::Name(name)) = items.first() {
                                if let Some(&idx) = name_to_idx.get(name.as_str()) {
                                    output_node = Some(idx);
                                } else {
                                    return Err(IngestError::NodeNotFound(name.clone()));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let output_node = output_node.ok_or(IngestError::NoOutput)?;

    Ok(FunctionGraph {
        name: "fx_graph".into(),
        nodes,
        wires,
        output_node,
        return_type: DEFAULT_TYPE.into(),
    })
}

/// Emit wires (and literal nodes) for the arguments of a call node.
fn emit_arg_wires(
    args: &[FxArg],
    target_node: usize,
    name_to_idx: &HashMap<&str, usize>,
    nodes: &mut Vec<GraphNode>,
    wires: &mut Vec<Wire>,
) -> Result<(), IngestError> {
    for (port, arg) in args.iter().enumerate() {
        match arg {
            FxArg::Name(name) => {
                if let Some(&from_idx) = name_to_idx.get(name.as_str()) {
                    wires.push(Wire {
                        from: from_idx,
                        to: target_node,
                        port_in_idx: port as i32,
                        wire_kind: WireKind::Copy,
                    });
                } else {
                    return Err(IngestError::NodeNotFound(name.clone()));
                }
            }
            FxArg::Int(v) => {
                let lit_idx = nodes.len();
                nodes.push(GraphNode::Literal {
                    value: LiteralValue::I64(*v),
                    data_type: "i64".into(),
                });
                wires.push(Wire {
                    from: lit_idx,
                    to: target_node,
                    port_in_idx: port as i32,
                    wire_kind: WireKind::Copy,
                });
            }
            FxArg::Float(v) => {
                let lit_idx = nodes.len();
                nodes.push(GraphNode::Literal {
                    value: LiteralValue::F64(*v),
                    data_type: "f64".into(),
                });
                wires.push(Wire {
                    from: lit_idx,
                    to: target_node,
                    port_in_idx: port as i32,
                    wire_kind: WireKind::Copy,
                });
            }
            FxArg::Bool(v) => {
                let lit_idx = nodes.len();
                nodes.push(GraphNode::Literal {
                    value: LiteralValue::Bool(*v),
                    data_type: "bool".into(),
                });
                wires.push(Wire {
                    from: lit_idx,
                    to: target_node,
                    port_in_idx: port as i32,
                    wire_kind: WireKind::Copy,
                });
            }
            FxArg::List(items) => {
                // Recurse into list arguments (e.g. shape tuples).
                // Each sub-arg gets a sequential port index starting from
                // the current port position.
                for (sub_port, item) in items.iter().enumerate() {
                    let effective_port = port + sub_port;
                    emit_single_arg(
                        item,
                        target_node,
                        effective_port as i32,
                        name_to_idx,
                        nodes,
                        wires,
                    )?;
                }
            }
            FxArg::None => {
                // None args are skipped — no wire emitted.
            }
        }
    }
    Ok(())
}

/// Emit a single wire for one argument value.
fn emit_single_arg(
    arg: &FxArg,
    target_node: usize,
    port: i32,
    name_to_idx: &HashMap<&str, usize>,
    nodes: &mut Vec<GraphNode>,
    wires: &mut Vec<Wire>,
) -> Result<(), IngestError> {
    match arg {
        FxArg::Name(name) => {
            if let Some(&from_idx) = name_to_idx.get(name.as_str()) {
                wires.push(Wire {
                    from: from_idx,
                    to: target_node,
                    port_in_idx: port,
                    wire_kind: WireKind::Copy,
                });
            } else {
                return Err(IngestError::NodeNotFound(name.clone()));
            }
        }
        FxArg::Int(v) => {
            let lit_idx = nodes.len();
            nodes.push(GraphNode::Literal {
                value: LiteralValue::I64(*v),
                data_type: "i64".into(),
            });
            wires.push(Wire {
                from: lit_idx,
                to: target_node,
                port_in_idx: port,
                wire_kind: WireKind::Copy,
            });
        }
        FxArg::Float(v) => {
            let lit_idx = nodes.len();
            nodes.push(GraphNode::Literal {
                value: LiteralValue::F64(*v),
                data_type: "f64".into(),
            });
            wires.push(Wire {
                from: lit_idx,
                to: target_node,
                port_in_idx: port,
                wire_kind: WireKind::Copy,
            });
        }
        FxArg::Bool(v) => {
            let lit_idx = nodes.len();
            nodes.push(GraphNode::Literal {
                value: LiteralValue::Bool(*v),
                data_type: "bool".into(),
            });
            wires.push(Wire {
                from: lit_idx,
                to: target_node,
                port_in_idx: port,
                wire_kind: WireKind::Copy,
            });
        }
        FxArg::List(_) | FxArg::None => {
            // Nested lists: skip. None: no wire.
        }
    }
    Ok(())
}
