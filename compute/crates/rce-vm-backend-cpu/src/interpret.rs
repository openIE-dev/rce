//! Topological evaluator for `rce_graph::graph::FunctionGraph`.
//!
//! v0.1 supports: `Param`, `Literal`, `TensorLiteral`, and `Apply` over the
//! non-Custom `OpKind` set. Both `ScalarValue` and `HostTensor` flow on
//! wires (unified via `Value`). Unsupported nodes return
//! `InterpretError::Unsupported`.

use crate::error::{InterpretError, Result};
use crate::kernel;
use crate::leaf::LeafDispatch;
use crate::op_dispatch::OpDispatch;
use crate::scalar::apply_op;
use crate::tensor::{DType, HostTensor};
use crate::value::{ScalarValue, Value};
use rce_graph::graph::{
    FunctionGraph, GraphNode, LiteralValue, OpKind, ProgramGraph, TensorLiteralData, Wire,
};
use std::collections::HashMap;

/// Param-name → bound value mapping for a function call.
pub type Inputs = HashMap<String, Value>;

/// Bundle of optional dispatchers consulted during interpretation.
#[derive(Default, Clone, Copy)]
pub struct InterpretConfig<'a> {
    /// Leaf dispatchers — handle `GraphNode::CodeBlock { language, … }`.
    pub leaves: &'a [&'a dyn LeafDispatch],
    /// Op dispatchers — preempt the built-in `ml::dispatch_custom` for
    /// `Apply { op: Custom(name) }` when their `handles()` claims `name`.
    pub ops: &'a [&'a dyn OpDispatch],
}

/// Walk `graph` in topological order, returning the value of `output_node`.
/// Equivalent to [`interpret_with_config`] with no dispatchers attached.
pub fn interpret(graph: &FunctionGraph, inputs: &Inputs) -> Result<Value> {
    interpret_with_config(graph, inputs, InterpretConfig::default())
}

/// Backwards-compatible: only leaf dispatchers, no op dispatchers.
pub fn interpret_with_leaves(
    graph: &FunctionGraph,
    inputs: &Inputs,
    leaves: &[&dyn LeafDispatch],
) -> Result<Value> {
    interpret_with_config(graph, inputs, InterpretConfig { leaves, ops: &[] })
}

/// Full-fat entry point. Consult `cfg.ops` for `Custom(name)` ops, then
/// `cfg.leaves` for code blocks, then the built-in CPU paths.
///
/// Emits a `tracing` span for the function call as a whole and one for
/// each node evaluated; attach a `tracing-subscriber` (JSON layer or
/// OpenTelemetry layer) to capture them. Defaults to no-op when no
/// subscriber is installed.
#[tracing::instrument(level = "debug", skip(graph, inputs, cfg), fields(graph = %graph.name, n_nodes = graph.nodes.len()))]
pub fn interpret_with_config(
    graph: &FunctionGraph,
    inputs: &Inputs,
    cfg: InterpretConfig<'_>,
) -> Result<Value> {
    if graph.output_node >= graph.nodes.len() {
        return Err(InterpretError::OutputNodeOutOfRange(
            graph.output_node,
            graph.nodes.len(),
        ));
    }

    let order = topo_sort(graph)?;
    let mut values: HashMap<usize, Value> = HashMap::with_capacity(graph.nodes.len());

    for node_idx in order {
        let node = &graph.nodes[node_idx];
        let kind_label = node_kind_label(node);
        let _span = tracing::debug_span!(
            "node",
            id = node_idx,
            kind = kind_label,
        )
        .entered();
        let value = eval_node(node_idx, node, graph, &values, inputs, cfg)?;
        values.insert(node_idx, value);
    }

    values.remove(&graph.output_node).ok_or_else(|| {
        InterpretError::Internal(format!(
            "output node {} not produced by topo walk",
            graph.output_node
        ))
    })
}

/// The reference evaluation of a graph term (spec 6.13, GT-6).
///
/// The nodes run in the fixed order of `topo_sort` (Kahn's algorithm, the
/// ready nodes taken in ascending index, then in wire order), each by the
/// CPU kernels of this crate alone: no op or leaf dispatcher is consulted.
/// Every kernel it reaches is single-threaded and performs its operations
/// in a fixed order (a matrix product and a sum add in increasing index,
/// starting from +0.0), one IEEE 754 rounding per operation and no fused
/// multiply-add, so the bits of the result are a function of the inputs
/// alone, on every machine.
///
/// RCE's floats are finite (spec 6.3 N3). After each node, every float
/// element of its value is tested; the first node, in that order, whose
/// value holds NaN or an infinity stops the evaluation with
/// [`InterpretError::NonFinite`], which the checker reports as the trap
/// `non-finite float`. A node whose declared element type is an integer
/// type is computed exactly and checked against that type (see
/// `integer_node`); a trap of it, or of a kernel (an index out of range),
/// is [`InterpretError::Trap`] with RCE's trap text. The transcendental
/// functions are RCE's own ([`crate::crmath`]).
pub fn interpret_reference(graph: &FunctionGraph, inputs: &Inputs) -> Result<Value> {
    if graph.output_node >= graph.nodes.len() {
        return Err(InterpretError::OutputNodeOutOfRange(
            graph.output_node,
            graph.nodes.len(),
        ));
    }
    let order = topo_sort(graph)?;
    let mut values: HashMap<usize, Value> = HashMap::with_capacity(graph.nodes.len());
    for node_idx in order {
        let node = &graph.nodes[node_idx];
        let value = match node {
            GraphNode::Apply { op, output_type }
                if crate::ml::nn::int_range(crate::ml::nn::declared_elem(output_type)).is_some() =>
            {
                let args = gather_inputs(node_idx, graph, &values)?;
                integer_node(node_idx, op, &args, crate::ml::nn::declared_elem(output_type))?
            }
            _ => eval_node(node_idx, node, graph, &values, inputs, InterpretConfig::default())?,
        };
        if !all_finite(&value)? {
            return Err(InterpretError::NonFinite(node_idx));
        }
        values.insert(node_idx, value);
    }
    values.remove(&graph.output_node).ok_or_else(|| {
        InterpretError::Internal(format!(
            "output node {} not produced by topo walk",
            graph.output_node
        ))
    })
}

/// An operation node whose declared element type is an integer type (`i8`,
/// `i32`; an i8 tensor is held in i32 lanes): lane-wise `+ - * /` and a sum
/// are exact and checked against the declared type, as RCE's integer
/// arithmetic is (spec 6.3 N2); any other operation (reshape, transpose,
/// concatenation) moves elements, and its result is checked to lie in the
/// declared type.
fn integer_node(node_id: usize, op: &OpKind, args: &[Value], elem: &str) -> Result<Value> {
    use crate::ml::nn;
    let tensor = |v: &Value| match v {
        Value::Tensor(t) if t.dtype == DType::I32 => Ok(t.clone()),
        other => Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!("an {elem} operation on {}", other.type_name()),
        }),
    };
    match (op, args) {
        (OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div, [a, b]) => {
            nn::int_binary(node_id, op, &tensor(a)?, &tensor(b)?, elem).map(Value::Tensor)
        }
        (OpKind::Custom(n), [x, ..]) if n == crate::ml::OP_REDUCE_SUM => {
            nn::int_sum(node_id, &tensor(x)?, elem).map(Value::Tensor)
        }
        (OpKind::Custom(n), _) => {
            let v = crate::ml::dispatch_custom(node_id, n, args)?;
            nn::int_check(node_id, &tensor(&v)?, elem)?;
            Ok(v)
        }
        (other, _) => Err(InterpretError::Unsupported(format!("{other:?} on {elem} tensors"))),
    }
}

/// Whether every float element of `v` is finite (integers and booleans are).
fn all_finite(v: &Value) -> Result<bool> {
    Ok(match v {
        Value::Scalar(ScalarValue::F32(x)) => x.is_finite(),
        Value::Scalar(ScalarValue::F64(x)) => x.is_finite(),
        Value::Scalar(_) => true,
        Value::Tensor(t) => match t.dtype {
            DType::F32 => t.as_f32()?.iter().all(|x| x.is_finite()),
            DType::F64 => t.as_f64()?.iter().all(|x| x.is_finite()),
            DType::F16 | DType::BF16 => t.as_f32_promoted()?.iter().all(|x| x.is_finite()),
            DType::Bool | DType::I32 | DType::I64 => true,
        },
        Value::DeviceTensor(_) => {
            return Err(InterpretError::Unsupported(
                "the reference evaluation holds host values only".into(),
            ))
        }
    })
}

/// Max nested `Call` depth — guards against runaway/cyclic recursion
/// in a `ProgramGraph` (the per-function graph is already DAG-checked
/// by `topo_sort`; this bounds *inter*-function recursion).
const MAX_CALL_DEPTH: usize = 256;

/// Execute a multi-function [`ProgramGraph`]: interpret its `entry`
/// function, resolving `GraphNode::Call { callee, .. }` by recursively
/// interpreting the named callee with the call site's port-ordered
/// inputs bound to the callee's `Param` nodes **positionally, in
/// declaration order** (the builder controls both sides — this is the
/// fg ProgramGraph call ABI).
///
/// Purely additive: every non-`Call` node is delegated to the exact
/// same [`eval_node`] the single-graph [`interpret_with_config`] uses,
/// so a `Call`-free entry is bit-identical to interpreting that
/// function directly, and `Call` only moves `Value`s around (no
/// arithmetic) — FP-neutral by construction.
#[tracing::instrument(level = "debug", skip(program, inputs, cfg), fields(entry = %program.entry, n_fns = program.functions.len()))]
pub fn interpret_program(
    program: &ProgramGraph,
    inputs: &Inputs,
    cfg: InterpretConfig<'_>,
) -> Result<Value> {
    let entry = program
        .functions
        .iter()
        .find(|f| f.name == program.entry)
        .ok_or_else(|| {
            InterpretError::Unsupported(format!(
                "ProgramGraph entry function `{}` not found among {} functions",
                program.entry,
                program.functions.len()
            ))
        })?;
    interpret_fn_in_program(entry, inputs, program, cfg, 0)
}

fn interpret_fn_in_program(
    graph: &FunctionGraph,
    inputs: &Inputs,
    program: &ProgramGraph,
    cfg: InterpretConfig<'_>,
    depth: usize,
) -> Result<Value> {
    if depth > MAX_CALL_DEPTH {
        return Err(InterpretError::Unsupported(format!(
            "Call recursion exceeded {MAX_CALL_DEPTH} (cyclic/runaway program graph?)"
        )));
    }
    if graph.output_node >= graph.nodes.len() {
        return Err(InterpretError::OutputNodeOutOfRange(
            graph.output_node,
            graph.nodes.len(),
        ));
    }
    let order = topo_sort(graph)?;
    let mut values: HashMap<usize, Value> = HashMap::with_capacity(graph.nodes.len());
    for node_idx in order {
        let node = &graph.nodes[node_idx];
        let value = match node {
            GraphNode::Call { callee, .. } => {
                let args = gather_inputs(node_idx, graph, &values)?;
                let cf = program
                    .functions
                    .iter()
                    .find(|f| &f.name == callee)
                    .ok_or_else(|| {
                        InterpretError::Unsupported(format!(
                            "Call to `{callee}`: no such function in program"
                        ))
                    })?;
                let param_names: Vec<&str> = cf
                    .nodes
                    .iter()
                    .filter_map(|n| match n {
                        GraphNode::Param(p) => Some(p.name.as_str()),
                        _ => None,
                    })
                    .collect();
                if param_names.len() != args.len() {
                    return Err(InterpretError::Unsupported(format!(
                        "Call to `{callee}`: {} params but {} args supplied",
                        param_names.len(),
                        args.len()
                    )));
                }
                let mut sub: Inputs = HashMap::with_capacity(param_names.len());
                for (nm, a) in param_names.iter().zip(args.into_iter()) {
                    sub.insert(nm.to_string(), a);
                }
                interpret_fn_in_program(cf, &sub, program, cfg, depth + 1)?
            }
            _ => eval_node(node_idx, node, graph, &values, inputs, cfg)?,
        };
        values.insert(node_idx, value);
    }
    values.remove(&graph.output_node).ok_or_else(|| {
        InterpretError::Internal(format!(
            "output node {} not produced by topo walk",
            graph.output_node
        ))
    })
}

/// Kahn's algorithm topological sort.
fn topo_sort(graph: &FunctionGraph) -> Result<Vec<usize>> {
    let n = graph.nodes.len();
    let mut indeg = vec![0usize; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for w in &graph.wires {
        if w.from >= n || w.to >= n {
            return Err(InterpretError::InvalidWire(format!(
                "wire {}->{} out of bounds (n={n})",
                w.from, w.to
            )));
        }
        adj[w.from].push(w.to);
        indeg[w.to] += 1;
    }
    let mut queue: std::collections::VecDeque<usize> = (0..n).filter(|i| indeg[*i] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &v in &adj[u] {
            indeg[v] -= 1;
            if indeg[v] == 0 {
                queue.push_back(v);
            }
        }
    }
    if order.len() != n {
        return Err(InterpretError::Cyclic);
    }
    Ok(order)
}

fn gather_inputs(
    node_idx: usize,
    graph: &FunctionGraph,
    values: &HashMap<usize, Value>,
) -> Result<Vec<Value>> {
    let mut wires: Vec<&Wire> = graph
        .wires
        .iter()
        .filter(|w| w.to == node_idx)
        .collect();
    wires.sort_by_key(|w| w.port_in_idx);
    let mut args = Vec::with_capacity(wires.len());
    for w in wires {
        let v = values.get(&w.from).cloned().ok_or_else(|| {
            InterpretError::Internal(format!(
                "wire from {} to {} (port {}) had no source value",
                w.from, w.to, w.port_in_idx
            ))
        })?;
        args.push(v);
    }
    Ok(args)
}

fn node_kind_label(node: &GraphNode) -> &'static str {
    match node {
        GraphNode::Param(_) => "param",
        GraphNode::Literal { .. } => "literal",
        GraphNode::TensorLiteral(_) => "tensor_literal",
        GraphNode::Apply { .. } => "apply",
        GraphNode::CodeBlock { .. } => "code_block",
        GraphNode::Select { .. } => "select",
        GraphNode::WhileLoop { .. } => "while_loop",
        GraphNode::Call { .. } => "call",
        GraphNode::Match { .. } => "match",
        GraphNode::Mutate { .. } => "mutate",
        GraphNode::Sequence { .. } => "sequence",
        GraphNode::StructCreate { .. } => "struct_create",
        GraphNode::FieldAccess { .. } => "field_access",
        GraphNode::ArrayCreate { .. } => "array_create",
        GraphNode::IndexAccess { .. } => "index_access",
        GraphNode::ErrorPropagate { .. } => "error_propagate",
        GraphNode::Spawn { .. } => "spawn",
        GraphNode::JoinAwait { .. } => "join_await",
        GraphNode::YieldSuspend => "yield_suspend",
        GraphNode::TypeDefine { .. } => "type_define",
        GraphNode::EnumCreate { .. } => "enum_create",
        GraphNode::EnumTag { .. } => "enum_tag",
        GraphNode::EnumPayload { .. } => "enum_payload",
        GraphNode::VTableLookup { .. } => "vtable_lookup",
        GraphNode::CapabilityRef { .. } => "capability_ref",
    }
}

fn eval_node(
    idx: usize,
    node: &GraphNode,
    graph: &FunctionGraph,
    values: &HashMap<usize, Value>,
    inputs: &Inputs,
    cfg: InterpretConfig<'_>,
) -> Result<Value> {
    match node {
        GraphNode::Param(p) => inputs
            .get(&p.name)
            .cloned()
            .ok_or_else(|| InterpretError::UnboundParam(p.name.clone())),

        GraphNode::Literal { value, .. } => Ok(Value::Scalar(literal_to_scalar(value))),

        GraphNode::TensorLiteral(td) => Ok(Value::Tensor(tensor_literal_to_host(idx, td)?)),

        GraphNode::Apply { op, .. } => {
            let args = gather_inputs(idx, graph, values)?;
            apply_value_op(idx, op, &args, cfg)
        }

        GraphNode::CodeBlock {
            language,
            code_text,
            output_type,
        } => {
            let args_v = gather_inputs(idx, graph, values)?;
            // v0.2 leaf protocol: scalar OR tensor inputs. The leaf decides
            // how to handle tensor args (Wasm v0.2 routes through linear
            // memory per docs/spec/fg-wasm-leaf-abi.md; older v0.1 leaves
            // return Unsupported on tensors and the interpreter falls
            // through to the next dispatcher).
            for d in cfg.leaves {
                if d.languages().iter().any(|l| *l == language) {
                    match d.invoke(idx, language, code_text, output_type, &args_v) {
                        Ok(v) => return Ok(v),
                        Err(InterpretError::Unsupported(_)) => continue,
                        Err(e) => return Err(e),
                    }
                }
            }
            Err(InterpretError::Unsupported(format!(
                "CodeBlock language `{language}` has no registered leaf dispatcher \
                 capable of handling these arguments"
            )))
        }

        // v0.1 unsupported.
        other => Err(InterpretError::Unsupported(format!("{other:?}"))),
    }
}

/// Dispatch op over `Value`. Custom ops route through `cfg.ops` (preempt),
/// then fall back to `ml::dispatch_custom`. Scalar args promote to 0-D
/// tensors when any input is a tensor; pure scalar pairs fall through to
/// the original scalar dispatch.
fn apply_value_op(
    node_id: usize,
    op: &OpKind,
    args: &[Value],
    cfg: InterpretConfig<'_>,
) -> Result<Value> {
    if let OpKind::Custom(name) = op {
        for d in cfg.ops {
            if d.claims(name.as_str()) {
                match d.execute(node_id, name, args) {
                    Ok(v) => return Ok(v),
                    Err(InterpretError::Unsupported(_)) => {
                        // Dispatcher claimed the name but cannot handle these
                        // specific inputs — fall through to next candidate.
                        continue;
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        return crate::ml::dispatch_custom(node_id, name, args);
    }
    let any_tensor = args.iter().any(|v| matches!(v, Value::Tensor(_)));
    // Primitive op + at least one tensor: try op dispatchers (e.g. wgpu
    // elementwise), then fall through to the CPU kernel below.
    if any_tensor {
        for d in cfg.ops {
            match d.execute_primitive(node_id, op, args) {
                Ok(v) => return Ok(v),
                Err(InterpretError::Unsupported(_)) => continue,
                Err(e) => return Err(e),
            }
        }
    }
    if !any_tensor {
        // All scalars — original scalar dispatch.
        let scalars: Vec<ScalarValue> = args
            .iter()
            .map(|v| match v {
                Value::Scalar(s) => s.clone(),
                _ => unreachable!(),
            })
            .collect();
        return Ok(Value::Scalar(apply_op(node_id, op, &scalars)?));
    }

    // At least one tensor. Promote scalars to 0-D tensors, then dispatch.
    let tensors: Vec<HostTensor> = args
        .iter()
        .map(|v| match v {
            Value::Tensor(t) => Ok(t.clone()),
            Value::Scalar(s) => HostTensor::from_scalar(s),
            Value::DeviceTensor(d) => Err(InterpretError::Unsupported(format!(
                "primitive op on cpu kernel layer cannot consume DeviceTensor \
                 (device={}, shape={:?}). A leaf earlier in the OpDispatch \
                 chain must claim this op for the device, OR the producer \
                 must materialise to host first.",
                d.device.name(),
                d.shape
            ))),
        })
        .collect::<Result<Vec<_>>>()?;

    let result = match (op, tensors.as_slice()) {
        (OpKind::Neg | OpKind::Not, [a]) => kernel::unary(node_id, op, a)?,
        (
            OpKind::Add
            | OpKind::Sub
            | OpKind::Mul
            | OpKind::Div
            | OpKind::Rem
            | OpKind::Eq
            | OpKind::Ne
            | OpKind::Lt
            | OpKind::Le
            | OpKind::Gt
            | OpKind::Ge
            | OpKind::And
            | OpKind::Or,
            [a, b],
        ) => kernel::binary(node_id, op, a, b)?,
        (OpKind::Custom(_), _) => {
            // Custom ops are not unpacked here — they are handled before
            // tensor-promotion below.
            unreachable!("Custom dispatch handled in apply_value_op above")
        }
        (OpKind::Concat, _) => {
            return Err(InterpretError::TypeMismatch {
                node_id,
                msg: "Concat is not defined on tensors in v0.1".into(),
            });
        }
        (op, args) => {
            return Err(InterpretError::ArityMismatch {
                node_id,
                kind: format!("{op:?}"),
                expected: 2,
                got: args.len(),
            });
        }
    };
    Ok(Value::Tensor(result))
}

fn literal_to_scalar(lit: &LiteralValue) -> ScalarValue {
    match lit {
        LiteralValue::I64(x) => ScalarValue::I64(*x),
        LiteralValue::I32(x) => ScalarValue::I32(*x),
        LiteralValue::F64(x) => ScalarValue::F64(*x),
        LiteralValue::F32(x) => ScalarValue::F32(*x),
        LiteralValue::Bool(x) => ScalarValue::Bool(*x),
        LiteralValue::String(s) => ScalarValue::String(s.clone()),
    }
}

fn tensor_literal_to_host(node_id: usize, td: &TensorLiteralData) -> Result<HostTensor> {
    if td.data.is_empty() && !td.shape.is_empty() {
        return Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!(
                "TensorLiteral '{}' has empty data — weights must be reattached \
                 (sidecar / EMBED_WEIGHTS) before interpretation",
                td.name
            ),
        });
    }
    let dtype = DType::from_name(&td.dtype).ok_or_else(|| InterpretError::TypeMismatch {
        node_id,
        msg: format!("TensorLiteral dtype `{}` not supported by v0.1", td.dtype),
    })?;
    HostTensor::new(td.shape.clone(), dtype, td.data.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rce_graph::graph::{OpKind, Param, WireKind};

    fn add_graph() -> FunctionGraph {
        FunctionGraph {
            name: "add".into(),
            nodes: vec![
                GraphNode::Param(Param {
                    name: "x".into(),
                    data_type: "f32".into(),
                }),
                GraphNode::Param(Param {
                    name: "y".into(),
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
                    wire_kind: WireKind::Move,
                },
                Wire {
                    from: 1,
                    to: 2,
                    port_in_idx: 1,
                    wire_kind: WireKind::Move,
                },
            ],
            output_node: 2,
            return_type: "f32".into(),
        }
    }

    #[test]
    fn evaluates_add() {
        let g = add_graph();
        let mut inputs = Inputs::new();
        inputs.insert("x".into(), ScalarValue::F32(3.0).into());
        inputs.insert("y".into(), ScalarValue::F32(4.0).into());
        let out = interpret(&g, &inputs).unwrap();
        assert_eq!(out, Value::Scalar(ScalarValue::F32(7.0)));
    }

    #[test]
    fn errors_on_unbound_param() {
        let g = add_graph();
        let mut inputs = Inputs::new();
        inputs.insert("x".into(), ScalarValue::F32(3.0).into());
        let err = interpret(&g, &inputs).unwrap_err();
        assert!(matches!(err, InterpretError::UnboundParam(s) if s == "y"));
    }

    #[test]
    fn evaluates_tensor_add() {
        let g = add_graph();
        let mut inputs = Inputs::new();
        inputs.insert(
            "x".into(),
            HostTensor::from_f32(vec![3], &[1.0, 2.0, 3.0]).unwrap().into(),
        );
        inputs.insert(
            "y".into(),
            HostTensor::from_f32(vec![3], &[4.0, 5.0, 6.0]).unwrap().into(),
        );
        let out = interpret(&g, &inputs).unwrap();
        match out {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3]);
                assert_eq!(t.as_f32().unwrap(), vec![5.0, 7.0, 9.0]);
            }
            _ => panic!("expected tensor output"),
        }
    }

    #[test]
    fn evaluates_scalar_tensor_broadcast() {
        let g = add_graph();
        let mut inputs = Inputs::new();
        inputs.insert(
            "x".into(),
            HostTensor::from_f32(vec![3], &[10.0, 20.0, 30.0]).unwrap().into(),
        );
        inputs.insert("y".into(), ScalarValue::F32(5.0).into());
        let out = interpret(&g, &inputs).unwrap();
        match out {
            Value::Tensor(t) => assert_eq!(t.as_f32().unwrap(), vec![15.0, 25.0, 35.0]),
            _ => panic!("expected tensor output"),
        }
    }

    #[test]
    fn detects_cycle() {
        let g = FunctionGraph {
            name: "cyc".into(),
            nodes: vec![
                GraphNode::Apply {
                    op: OpKind::Add,
                    output_type: "i32".into(),
                },
                GraphNode::Apply {
                    op: OpKind::Add,
                    output_type: "i32".into(),
                },
            ],
            wires: vec![
                Wire {
                    from: 0,
                    to: 1,
                    port_in_idx: 0,
                    wire_kind: WireKind::Move,
                },
                Wire {
                    from: 1,
                    to: 0,
                    port_in_idx: 0,
                    wire_kind: WireKind::Move,
                },
            ],
            output_node: 1,
            return_type: "i32".into(),
        };
        assert!(matches!(interpret(&g, &Inputs::new()), Err(InterpretError::Cyclic)));
    }

    #[test]
    fn literals_and_arithmetic() {
        // (5 * 2) - 3 = 7
        let g = FunctionGraph {
            name: "mix".into(),
            nodes: vec![
                GraphNode::Literal {
                    value: LiteralValue::I32(5),
                    data_type: "i32".into(),
                },
                GraphNode::Literal {
                    value: LiteralValue::I32(2),
                    data_type: "i32".into(),
                },
                GraphNode::Apply {
                    op: OpKind::Mul,
                    output_type: "i32".into(),
                },
                GraphNode::Literal {
                    value: LiteralValue::I32(3),
                    data_type: "i32".into(),
                },
                GraphNode::Apply {
                    op: OpKind::Sub,
                    output_type: "i32".into(),
                },
            ],
            wires: vec![
                Wire {
                    from: 0,
                    to: 2,
                    port_in_idx: 0,
                    wire_kind: WireKind::Move,
                },
                Wire {
                    from: 1,
                    to: 2,
                    port_in_idx: 1,
                    wire_kind: WireKind::Move,
                },
                Wire {
                    from: 2,
                    to: 4,
                    port_in_idx: 0,
                    wire_kind: WireKind::Move,
                },
                Wire {
                    from: 3,
                    to: 4,
                    port_in_idx: 1,
                    wire_kind: WireKind::Move,
                },
            ],
            output_node: 4,
            return_type: "i32".into(),
        };
        let out = interpret(&g, &Inputs::new()).unwrap();
        assert_eq!(out, Value::Scalar(ScalarValue::I32(7)));
    }

    // ── Brick 4a: ProgramGraph / Call executor ─────────────────────

    /// An entry fn that just `Call`s `callee` with its own params,
    /// passed through positionally. Output = the call's value.
    fn passthrough_entry(callee: &str, nparams: usize) -> FunctionGraph {
        let mut nodes: Vec<GraphNode> = (0..nparams)
            .map(|i| {
                GraphNode::Param(Param {
                    name: format!("p{i}"),
                    data_type: "f32".into(),
                })
            })
            .collect();
        let call_idx = nodes.len();
        nodes.push(GraphNode::Call {
            callee: callee.into(),
            arg_types: vec!["f32".into(); nparams],
            return_type: "f32".into(),
        });
        let wires = (0..nparams)
            .map(|i| Wire {
                from: i,
                to: call_idx,
                port_in_idx: i as i32,
                wire_kind: WireKind::Move,
            })
            .collect();
        FunctionGraph {
            name: "main".into(),
            nodes,
            wires,
            output_node: call_idx,
            return_type: "f32".into(),
        }
    }

    fn prog(funcs: Vec<FunctionGraph>, entry: &str) -> ProgramGraph {
        ProgramGraph {
            functions: funcs,
            entry: entry.into(),
            allows: Default::default(),
        }
    }

    /// **The lossless anchor.** A `Call`-only entry wrapping a callee
    /// must be **byte-identical** to interpreting that callee directly
    /// — `Call` only rebinds/moves `Value`s, zero arithmetic, so the
    /// ProgramGraph executor is FP-neutral over function execution.
    /// Tensor f32 so "bit-identical" is meaningful.
    #[test]
    fn program_call_passthrough_bit_identical_to_direct() {
        let callee = add_graph();
        let entry = passthrough_entry("add", 2);
        let p = prog(vec![entry, callee.clone()], "main");

        let mut direct_in = Inputs::new();
        direct_in.insert(
            "x".into(),
            HostTensor::from_f32(vec![4], &[1.5, -2.25, 3.0, 0.125]).unwrap().into(),
        );
        direct_in.insert(
            "y".into(),
            HostTensor::from_f32(vec![4], &[0.5, 2.0, -1.0, 7.875]).unwrap().into(),
        );
        let direct = interpret(&callee, &direct_in).unwrap();

        // Entry params are p0, p1 (positional → callee x, y).
        let mut prog_in = Inputs::new();
        prog_in.insert("p0".into(), direct_in["x"].clone());
        prog_in.insert("p1".into(), direct_in["y"].clone());
        let viaprog = interpret_program(&p, &prog_in, InterpretConfig::default()).unwrap();

        match (direct, viaprog) {
            (Value::Tensor(a), Value::Tensor(b)) => {
                let (af, bf) = (a.as_f32().unwrap(), b.as_f32().unwrap());
                assert_eq!(
                    af.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    bf.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "ProgramGraph Call path not byte-identical to direct interpret"
                );
            }
            other => panic!("expected tensors, got {other:?}"),
        }
    }

    #[test]
    fn program_nested_calls_resolve() {
        // add(x,y); twice(a)=add(a,a); main(x)=twice(x) ⇒ 2x.
        let add = add_graph();
        let twice = FunctionGraph {
            name: "twice".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "a".into(), data_type: "f32".into() }),
                GraphNode::Call {
                    callee: "add".into(),
                    arg_types: vec!["f32".into(), "f32".into()],
                    return_type: "f32".into(),
                },
            ],
            wires: vec![
                Wire { from: 0, to: 1, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 0, to: 1, port_in_idx: 1, wire_kind: WireKind::Move },
            ],
            output_node: 1,
            return_type: "f32".into(),
        };
        let main = passthrough_entry("twice", 1);
        let p = prog(vec![main, twice, add], "main");
        let mut inp = Inputs::new();
        inp.insert("p0".into(), ScalarValue::F32(5.0).into());
        let out = interpret_program(&p, &inp, InterpretConfig::default()).unwrap();
        assert_eq!(out, Value::Scalar(ScalarValue::F32(10.0)));
    }

    #[test]
    fn program_call_errors_are_clean() {
        let add = add_graph();

        // Unknown entry.
        let p = prog(vec![add.clone()], "nope");
        let e = interpret_program(&p, &Inputs::new(), InterpretConfig::default()).unwrap_err();
        assert!(matches!(e, InterpretError::Unsupported(s) if s.contains("entry")));

        // Unknown callee.
        let p = prog(vec![passthrough_entry("ghost", 2), add.clone()], "main");
        let mut inp = Inputs::new();
        inp.insert("p0".into(), ScalarValue::F32(1.0).into());
        inp.insert("p1".into(), ScalarValue::F32(2.0).into());
        let e = interpret_program(&p, &inp, InterpretConfig::default()).unwrap_err();
        assert!(matches!(e, InterpretError::Unsupported(s) if s.contains("no such function")));

        // Arity mismatch (entry passes 1 arg, `add` needs 2).
        let p = prog(vec![passthrough_entry("add", 1), add], "main");
        let mut inp = Inputs::new();
        inp.insert("p0".into(), ScalarValue::F32(1.0).into());
        let e = interpret_program(&p, &inp, InterpretConfig::default()).unwrap_err();
        assert!(matches!(e, InterpretError::Unsupported(s) if s.contains("params but")));
    }
}
