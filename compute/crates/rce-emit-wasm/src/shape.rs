//! Shape inference for the v0 emitter. Walks the graph in topological
//! order and resolves every node's output shape from its input shapes
//! plus the op semantics. f32 rank-2 only.

use crate::emit::EmitError;
use rce_graph::graph::{FunctionGraph, GraphNode, OpKind};
use std::collections::HashMap;

/// Maps `node_id` → output shape (a `Vec<usize>`). Returned by
/// [`infer_node_shapes`] and consumed by the WAT emitter to size
/// scratch regions and pass `(rows, cols)` to kernel calls.
pub type NodeShapes = HashMap<usize, Vec<usize>>;

/// Infer every node's output shape. The `param_shapes` slice
/// supplies shapes for `Param` nodes in graph order; everything else
/// is derived from the graph topology + op semantics.
pub fn infer_node_shapes(
    graph: &FunctionGraph,
    param_shapes: &[Vec<usize>],
) -> Result<NodeShapes, EmitError> {
    let mut shapes: NodeShapes = HashMap::new();
    let mut param_idx = 0;

    // Build incoming-wire map for each node.
    let mut incoming: HashMap<usize, Vec<(i32, usize)>> = HashMap::new();
    for w in &graph.wires {
        incoming
            .entry(w.to)
            .or_default()
            .push((w.port_in_idx, w.from));
    }
    for v in incoming.values_mut() {
        v.sort_by_key(|(p, _)| *p);
    }

    for (id, node) in graph.nodes.iter().enumerate() {
        let shape = match node {
            GraphNode::Param(_) => {
                let s = param_shapes
                    .get(param_idx)
                    .ok_or_else(|| EmitError::ShapeMismatch(format!(
                        "graph has more Param nodes than param_shapes ({} provided)",
                        param_shapes.len()
                    )))?
                    .clone();
                param_idx += 1;
                s
            }
            GraphNode::TensorLiteral(t) => t.shape.clone(),
            GraphNode::Literal { .. } => {
                // Scalar literals are 0-rank; skip — they aren't tensor outputs.
                continue;
            }
            GraphNode::Apply { op, .. } => {
                let inputs = incoming
                    .get(&id)
                    .ok_or_else(|| EmitError::ShapeMismatch(format!(
                        "node {id}: Apply has no incoming wires"
                    )))?;
                infer_apply_shape(id, op, inputs, &shapes)?
            }
            other => {
                return Err(EmitError::UnsupportedNode(format!(
                    "node {id}: {} not supported in v0 emitter",
                    node_kind_name(other)
                )));
            }
        };
        shapes.insert(id, shape);
    }
    Ok(shapes)
}

/// Compute the output shape of an Apply node from its op + input
/// shapes. Captures the v0 op contract: matmul, elementwise binops,
/// relu, softmax.
fn infer_apply_shape(
    id: usize,
    op: &OpKind,
    inputs: &[(i32, usize)],
    shapes: &NodeShapes,
) -> Result<Vec<usize>, EmitError> {
    let lookup = |from: usize| -> Result<&Vec<usize>, EmitError> {
        shapes.get(&from).ok_or_else(|| EmitError::ShapeMismatch(format!(
            "node {id}: input from {from} has no shape (skipped or out of order)"
        )))
    };

    // Helper: extract input shapes by port order.
    let input_shape = |port: i32| -> Result<&Vec<usize>, EmitError> {
        let (_, from) = inputs
            .iter()
            .find(|(p, _)| *p == port)
            .ok_or_else(|| EmitError::ShapeMismatch(format!(
                "node {id}: missing input on port {port}"
            )))?;
        lookup(*from)
    };

    match op {
        OpKind::Custom(name) if name == "matmul" => {
            let a = input_shape(0)?;
            let b = input_shape(1)?;
            if a.len() != 2 || b.len() != 2 {
                return Err(EmitError::ShapeMismatch(format!(
                    "node {id}: matmul requires rank-2 inputs (got {a:?} × {b:?})"
                )));
            }
            if a[1] != b[0] {
                return Err(EmitError::ShapeMismatch(format!(
                    "node {id}: matmul inner dim {} ≠ {}",
                    a[1], b[0]
                )));
            }
            Ok(vec![a[0], b[1]])
        }
        OpKind::Custom(name) if name == "relu" || name == "softmax" => {
            // Element-wise / per-row reduction; preserves shape.
            // softmax has an axis literal at port 1 in the graph, but
            // we only support last-axis softmax in v0 — port 0 carries
            // the tensor.
            let s = input_shape(0)?;
            Ok(s.clone())
        }
        OpKind::Add | OpKind::Mul => {
            // Bias-add semantics: if rhs is rank-1, broadcast across
            // last dim of lhs (matching the bench's MLP layout).
            // Otherwise expect identical shapes.
            let a = input_shape(0)?;
            let b = input_shape(1)?;
            if a == b {
                Ok(a.clone())
            } else if b.len() == 1 && a.len() == 2 && a[1] == b[0] {
                Ok(a.clone())
            } else {
                Err(EmitError::ShapeMismatch(format!(
                    "node {id}: {op:?} requires same shape or rank-1 bias (got {a:?} vs {b:?})"
                )))
            }
        }
        other => Err(EmitError::UnsupportedOp(format!(
            "node {id}: op {other:?} not supported in v0 emitter"
        ))),
    }
}

/// Diagnostic helper for unsupported variants — keeps the error
/// message useful without pulling in a Display impl.
fn node_kind_name(n: &GraphNode) -> &'static str {
    match n {
        GraphNode::Param(_) => "Param",
        GraphNode::Literal { .. } => "Literal",
        GraphNode::TensorLiteral(_) => "TensorLiteral",
        GraphNode::Apply { .. } => "Apply",
        GraphNode::Select { .. } => "Select",
        GraphNode::WhileLoop { .. } => "WhileLoop",
        GraphNode::Call { .. } => "Call",
        GraphNode::Match { .. } => "Match",
        GraphNode::Mutate { .. } => "Mutate",
        GraphNode::Sequence { .. } => "Sequence",
        GraphNode::CodeBlock { .. } => "CodeBlock",
        _ => "<other>",
    }
}
