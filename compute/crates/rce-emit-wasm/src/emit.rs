//! WAT emitter — walks a `FunctionGraph`, lays out scratch memory
//! per node, and produces a self-contained Wasm module that runs the
//! whole graph end-to-end via one exported `run(...)` function.

use crate::kernels;
use crate::shape::{infer_node_shapes, NodeShapes};
use rce_graph::graph::{FunctionGraph, GraphNode, OpKind};
use std::collections::HashMap;
use std::fmt::Write as _;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmitError {
    #[error("unsupported node: {0}")]
    UnsupportedNode(String),
    #[error("unsupported op: {0}")]
    UnsupportedOp(String),
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("wat parse: {0}")]
    Wat(String),
}

/// Per-emission options. Shapes for `Param` nodes are required (no
/// shape inference from `data_type` strings yet). Output buffer
/// region is appended automatically.
pub struct EmitOptions {
    /// Shape (rows, cols) for each `Param` node, in graph-order.
    pub param_shapes: Vec<Vec<usize>>,
}

impl EmitOptions {
    pub fn new(param_shapes: Vec<Vec<usize>>) -> Self {
        Self { param_shapes }
    }
}

/// Emit a self-contained WAT module for the given graph.
///
/// Exports:
/// - `memory` — linear memory containing scratch regions
/// - `run(p0_off, p1_off, ..., out_off)` — runs the graph;
///   each `pN_off` is the byte offset where the caller wrote the
///   N-th Param's data; `out_off` is where the result will be written.
pub fn emit_wat(graph: &FunctionGraph, opts: &EmitOptions) -> Result<String, EmitError> {
    let shapes = infer_node_shapes(graph, &opts.param_shapes)?;

    // Identify Param nodes (in graph order) and their output offsets
    // — those are *runtime parameters* to `run`, written by the caller.
    let mut param_node_ids: Vec<usize> = Vec::new();
    for (id, node) in graph.nodes.iter().enumerate() {
        if matches!(node, GraphNode::Param(_)) {
            param_node_ids.push(id);
        }
    }

    // Lay out scratch in linear memory.
    //
    // Memory layout:
    //   [scratch for TensorLiteral nodes — emitted as data segments]
    //   [scratch for Apply node outputs — produced at runtime]
    //
    // Param tensors live wherever the caller put them (pointer comes in
    // as a `run` arg); their offset is *not* in this layout.
    //
    // We assign each non-param tensor-producing node a stable byte
    // offset starting at SCRATCH_BASE, allocated sequentially.
    const SCRATCH_BASE: usize = 0;
    let mut node_offsets: HashMap<usize, usize> = HashMap::new();
    let mut next_off: usize = SCRATCH_BASE;
    for (id, node) in graph.nodes.iter().enumerate() {
        match node {
            GraphNode::Param(_) => {
                // Param offsets come from `run` args, not the layout.
            }
            GraphNode::TensorLiteral(t) => {
                let bytes = t.shape.iter().product::<usize>() * 4; // f32
                node_offsets.insert(id, next_off);
                next_off += align_up(bytes, 16);
            }
            GraphNode::Apply { .. } => {
                let s = shapes.get(&id).ok_or_else(|| {
                    EmitError::ShapeMismatch(format!("node {id}: Apply missing shape"))
                })?;
                let bytes = s.iter().product::<usize>() * 4;
                node_offsets.insert(id, next_off);
                next_off += align_up(bytes, 16);
            }
            GraphNode::Literal { .. } => { /* not allocated in memory */ }
            other => {
                return Err(EmitError::UnsupportedNode(format!(
                    "{:?}",
                    std::mem::discriminant(other)
                )));
            }
        }
    }
    let total_scratch = next_off;
    let pages = (total_scratch.max(65536) + 65535) / 65536; // ≥1 page

    // Walk the graph in node order to produce the body. Apply order
    // matches node order because rce's graphs are emitted in topo
    // order at construction time (the bench builders, ingest paths,
    // and ::new() helpers all preserve this).
    let mut body = String::new();
    for (id, node) in graph.nodes.iter().enumerate() {
        if let GraphNode::Apply { op, .. } = node {
            emit_apply_call(&mut body, graph, id, op, &shapes, &node_offsets, &param_node_ids)?;
        }
    }

    // The graph's `output_node` is the value `run` should leave at the
    // caller-supplied `out_off`. We memcpy from the node's scratch
    // region into the caller's output region as the final step.
    let out_node = graph.output_node;
    let out_shape = shapes.get(&out_node).ok_or_else(|| {
        EmitError::ShapeMismatch(format!("output_node {out_node} has no shape"))
    })?;
    let out_elems: usize = out_shape.iter().product();
    let out_src_off = if let Some(off) = node_offsets.get(&out_node) {
        OutputSrc::Scratch(*off)
    } else if let Some(pos) = param_node_ids.iter().position(|&pid| pid == out_node) {
        OutputSrc::Param(pos)
    } else {
        return Err(EmitError::ShapeMismatch(format!(
            "output_node {out_node} is neither scratch nor a Param"
        )));
    };

    // Param args + final output arg.
    let mut run_params = String::new();
    for i in 0..param_node_ids.len() {
        write!(run_params, " (param $p{i} i32)").unwrap();
    }
    write!(run_params, " (param $out_off i32)").unwrap();

    let copy_to_out = match out_src_off {
        OutputSrc::Scratch(off) => format!(
            "(call $copy_f32 (i32.const {off}) (local.get $out_off) (i32.const {out_elems}))"
        ),
        OutputSrc::Param(idx) => format!(
            "(call $copy_f32 (local.get $p{idx}) (local.get $out_off) (i32.const {out_elems}))"
        ),
    };

    // Emit data segments for TensorLiteral weights so the caller
    // doesn't have to write them at runtime.
    let mut data_segments = String::new();
    for (id, node) in graph.nodes.iter().enumerate() {
        if let GraphNode::TensorLiteral(t) = node {
            let off = *node_offsets.get(&id).unwrap();
            let escaped = escape_wat_bytes(&t.data);
            writeln!(
                data_segments,
                r#"  (data (i32.const {off}) "{}")"#,
                escaped
            )
            .unwrap();
        }
    }

    // Stitch the module together.
    let mut wat = String::new();
    writeln!(wat, "(module").unwrap();
    writeln!(wat, "  (memory (export \"memory\") {pages})").unwrap();
    wat.push_str(kernels::FAST_EXP_WAT);
    wat.push_str(kernels::MATMUL_WAT);
    wat.push_str(kernels::ELEMWISE_ADD_WAT);
    wat.push_str(kernels::ELEMWISE_MUL_WAT);
    wat.push_str(kernels::ADD_BIAS_WAT);
    wat.push_str(kernels::RELU_WAT);
    wat.push_str(kernels::SOFTMAX_ROW_WAT);
    // Internal helper: byte-wise memcpy of n f32s.
    wat.push_str(
        r#"
  (func $copy_f32 (param $src i32) (param $dst i32) (param $n i32)
    (local $i i32) (local $off i32)
    (local.set $i (i32.const 0))
    (block $done
      (loop $next
        (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
        (local.set $off (i32.shl (local.get $i) (i32.const 2)))
        (f32.store offset=0
          (i32.add (local.get $dst) (local.get $off))
          (f32.load offset=0 (i32.add (local.get $src) (local.get $off))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $next))))
"#,
    );
    wat.push_str(&data_segments);
    writeln!(wat).unwrap();
    writeln!(wat, "  (func (export \"run\"){run_params}").unwrap();
    wat.push_str(&body);
    writeln!(wat, "    {copy_to_out})").unwrap();
    writeln!(wat, ")").unwrap();
    Ok(wat)
}

/// Compile WAT to wasm bytes. Equivalent to `wat::parse_str` but
/// shapes the error into `EmitError`.
pub fn emit_wasm(graph: &FunctionGraph, opts: &EmitOptions) -> Result<Vec<u8>, EmitError> {
    let wat = emit_wat(graph, opts)?;
    wat::parse_str(&wat).map_err(|e| EmitError::Wat(format!("{e}")))
}

enum OutputSrc {
    Scratch(usize),
    Param(usize),
}

/// Round up to the nearest multiple of `align`.
fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

/// Escape a byte slice for embedding in a WAT `(data ...)` segment.
/// WAT strings use C-style `\xx` for non-ASCII bytes.
fn escape_wat_bytes(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 4);
    for &b in bytes {
        if b == b'\\' || b == b'"' {
            out.push('\\');
            out.push(b as char);
        } else if (0x20..0x7e).contains(&b) {
            out.push(b as char);
        } else {
            write!(out, "\\{:02x}", b).unwrap();
        }
    }
    out
}

/// Emit the WAT body for one Apply node — looks up input offsets,
/// chooses the kernel, writes the call.
fn emit_apply_call(
    body: &mut String,
    graph: &FunctionGraph,
    id: usize,
    op: &OpKind,
    shapes: &NodeShapes,
    node_offsets: &HashMap<usize, usize>,
    param_node_ids: &[usize],
) -> Result<(), EmitError> {
    // Sort incoming wires by port for stable arg order.
    let mut inputs: Vec<(i32, usize)> = graph
        .wires
        .iter()
        .filter(|w| w.to == id)
        .map(|w| (w.port_in_idx, w.from))
        .collect();
    inputs.sort_by_key(|(p, _)| *p);

    let off_expr = |from: usize| -> String {
        if let Some(idx) = param_node_ids.iter().position(|&p| p == from) {
            format!("(local.get $p{idx})")
        } else if let Some(off) = node_offsets.get(&from) {
            format!("(i32.const {off})")
        } else {
            // Literal scalars are inline; not addressable. Caller
            // checks shape inference passed before reaching here.
            format!("(unreachable)")
        }
    };

    let out_off = node_offsets.get(&id).ok_or_else(|| {
        EmitError::ShapeMismatch(format!("node {id}: Apply has no scratch offset"))
    })?;

    let out_shape = shapes.get(&id).unwrap();

    match op {
        OpKind::Custom(name) if name == "matmul" => {
            let (_, from_a) = inputs[0];
            let (_, from_b) = inputs[1];
            let a_shape = shapes.get(&from_a).unwrap();
            let b_shape = shapes.get(&from_b).unwrap();
            let m = a_shape[0];
            let k = a_shape[1];
            let n = b_shape[1];
            writeln!(
                body,
                "    (call $matmul {} {} (i32.const {out_off}) (i32.const {m}) (i32.const {k}) (i32.const {n}))",
                off_expr(from_a),
                off_expr(from_b)
            )
            .unwrap();
        }
        OpKind::Add => {
            let (_, from_a) = inputs[0];
            let (_, from_b) = inputs[1];
            let a_shape = shapes.get(&from_a).unwrap();
            let b_shape = shapes.get(&from_b).unwrap();
            if a_shape == b_shape {
                let n: usize = a_shape.iter().product();
                writeln!(
                    body,
                    "    (call $elemwise_add {} {} (i32.const {out_off}) (i32.const {n}))",
                    off_expr(from_a),
                    off_expr(from_b)
                )
                .unwrap();
            } else if b_shape.len() == 1 && a_shape.len() == 2 && a_shape[1] == b_shape[0] {
                writeln!(
                    body,
                    "    (call $add_bias {} {} (i32.const {out_off}) (i32.const {}) (i32.const {}))",
                    off_expr(from_a),
                    off_expr(from_b),
                    a_shape[0],
                    a_shape[1]
                )
                .unwrap();
            } else {
                return Err(EmitError::ShapeMismatch(format!(
                    "node {id}: Add shape mismatch {a_shape:?} vs {b_shape:?}"
                )));
            }
        }
        OpKind::Mul => {
            let (_, from_a) = inputs[0];
            let (_, from_b) = inputs[1];
            let a_shape = shapes.get(&from_a).unwrap();
            let b_shape = shapes.get(&from_b).unwrap();
            if a_shape != b_shape {
                return Err(EmitError::ShapeMismatch(format!(
                    "node {id}: Mul requires same shape (got {a_shape:?} vs {b_shape:?})"
                )));
            }
            let n: usize = a_shape.iter().product();
            writeln!(
                body,
                "    (call $elemwise_mul {} {} (i32.const {out_off}) (i32.const {n}))",
                off_expr(from_a),
                off_expr(from_b)
            )
            .unwrap();
        }
        OpKind::Custom(name) if name == "relu" => {
            let (_, from_a) = inputs[0];
            let n: usize = out_shape.iter().product();
            writeln!(
                body,
                "    (call $relu {} (i32.const {out_off}) (i32.const {n}))",
                off_expr(from_a)
            )
            .unwrap();
        }
        OpKind::Custom(name) if name == "softmax" => {
            let (_, from_a) = inputs[0];
            let in_shape = shapes.get(&from_a).unwrap();
            if in_shape.len() != 2 {
                return Err(EmitError::ShapeMismatch(format!(
                    "node {id}: softmax requires rank-2 (got {in_shape:?})"
                )));
            }
            writeln!(
                body,
                "    (call $softmax_row {} (i32.const {out_off}) (i32.const {}) (i32.const {}))",
                off_expr(from_a),
                in_shape[0],
                in_shape[1]
            )
            .unwrap();
        }
        other => {
            return Err(EmitError::UnsupportedOp(format!(
                "node {id}: op {other:?} not in v0 emitter scope"
            )));
        }
    }
    Ok(())
}
