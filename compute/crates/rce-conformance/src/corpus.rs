//! The conformance corpus: a programmatically-generated set of test
//! cases that exercise every backend against the same `FunctionGraph`s.

use rce_vm_backend_cpu::{HostTensor, Inputs, ScalarValue, Value};
use rce_graph::graph::{
    FunctionGraph, GraphNode, LiteralValue, OpKind, Param, TensorLiteralData, Wire, WireKind,
};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtypeFamily {
    Int,
    Float,
    Bool,
    /// Mixed-dtype case (e.g. comparison op produces bool from float).
    Mixed,
}

#[derive(Debug, Clone)]
pub struct ConformanceCase {
    pub name: String,
    pub graph: FunctionGraph,
    pub inputs: Inputs,
    pub family: DtypeFamily,
}

/// Build the full v0.1 conformance corpus.
pub fn full_corpus() -> Vec<ConformanceCase> {
    let mut out = Vec::new();
    out.extend(scalar_arith_cases());
    out.extend(tensor_elementwise_cases());
    out.extend(broadcast_cases());
    out.extend(matmul_cases());
    out.extend(reduce_cases());
    out.extend(softmax_cases());
    out.extend(activation_cases());
    out.extend(transpose_reshape_cases());
    out.extend(end_to_end_cases());
    out
}

// ── Scalar arithmetic ───────────────────────────────────────────────

fn scalar_arith_cases() -> Vec<ConformanceCase> {
    let mut out = Vec::new();
    for (op_name, op) in [
        ("add", OpKind::Add),
        ("sub", OpKind::Sub),
        ("mul", OpKind::Mul),
    ] {
        out.push(scalar_binary_case(
            &format!("scalar/{op_name}/i32"),
            op.clone(),
            ScalarValue::I32(7),
            ScalarValue::I32(3),
            DtypeFamily::Int,
        ));
        out.push(scalar_binary_case(
            &format!("scalar/{op_name}/f32"),
            op.clone(),
            ScalarValue::F32(2.5),
            ScalarValue::F32(0.25),
            DtypeFamily::Float,
        ));
        out.push(scalar_binary_case(
            &format!("scalar/{op_name}/f64"),
            op,
            ScalarValue::F64(2.5),
            ScalarValue::F64(0.25),
            DtypeFamily::Float,
        ));
    }
    // Comparison: produces bool.
    for (op_name, op) in [
        ("lt", OpKind::Lt),
        ("le", OpKind::Le),
        ("gt", OpKind::Gt),
        ("ge", OpKind::Ge),
        ("eq", OpKind::Eq),
    ] {
        out.push(scalar_binary_case(
            &format!("scalar/cmp_{op_name}/f32"),
            op,
            ScalarValue::F32(1.0),
            ScalarValue::F32(2.0),
            DtypeFamily::Mixed,
        ));
    }
    out
}

fn scalar_binary_case(
    name: &str,
    op: OpKind,
    a: ScalarValue,
    b: ScalarValue,
    family: DtypeFamily,
) -> ConformanceCase {
    let dtype = a.type_name();
    let graph = FunctionGraph {
        name: name.into(),
        nodes: vec![
            GraphNode::Param(Param { name: "x".into(), data_type: dtype.into() }),
            GraphNode::Param(Param { name: "y".into(), data_type: dtype.into() }),
            GraphNode::Apply { op, output_type: dtype.into() },
        ],
        wires: vec![
            Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Move },
            Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Move },
        ],
        output_node: 2,
        return_type: dtype.into(),
    };
    let mut inputs = Inputs::new();
    inputs.insert("x".into(), Value::Scalar(a));
    inputs.insert("y".into(), Value::Scalar(b));
    ConformanceCase { name: name.into(), graph, inputs, family }
}

// ── Tensor elementwise (same shape) ─────────────────────────────────

fn tensor_elementwise_cases() -> Vec<ConformanceCase> {
    let mut out = Vec::new();
    for (op_name, op) in [
        ("add", OpKind::Add),
        ("sub", OpKind::Sub),
        ("mul", OpKind::Mul),
        ("div", OpKind::Div),
    ] {
        for shape in [vec![5usize], vec![3, 4], vec![2, 3, 5]] {
            out.push(tensor_binary_case_f32(
                &format!("tensor_ew/{op_name}/{shape:?}"),
                op.clone(),
                shape.clone(),
                shape,
            ));
        }
    }
    out
}

fn tensor_binary_case_f32(
    name: &str,
    op: OpKind,
    shape_a: Vec<usize>,
    shape_b: Vec<usize>,
) -> ConformanceCase {
    let numel_a: usize = shape_a.iter().product::<usize>().max(1);
    let numel_b: usize = shape_b.iter().product::<usize>().max(1);
    let a_data: Vec<f32> = (0..numel_a).map(|i| (i as f32) * 0.5 + 1.0).collect();
    // Avoid zeros in `b` to prevent Div by zero.
    let b_data: Vec<f32> = (0..numel_b).map(|i| (i as f32) * 0.25 + 0.5).collect();
    let graph = FunctionGraph {
        name: name.into(),
        nodes: vec![
            GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
            GraphNode::Param(Param { name: "y".into(), data_type: "f32".into() }),
            GraphNode::Apply { op, output_type: "f32".into() },
        ],
        wires: vec![
            Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Move },
            Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Move },
        ],
        output_node: 2,
        return_type: "f32".into(),
    };
    let mut inputs = Inputs::new();
    inputs.insert(
        "x".into(),
        Value::Tensor(HostTensor::from_f32(shape_a, &a_data).unwrap()),
    );
    inputs.insert(
        "y".into(),
        Value::Tensor(HostTensor::from_f32(shape_b, &b_data).unwrap()),
    );
    ConformanceCase { name: name.into(), graph, inputs, family: DtypeFamily::Float }
}

// ── Broadcasting ────────────────────────────────────────────────────

fn broadcast_cases() -> Vec<ConformanceCase> {
    let mut out = Vec::new();
    // (3,) + (2, 3) → (2, 3); covers row-vector broadcast
    out.push(tensor_binary_case_f32(
        "bcast/row_vec_add",
        OpKind::Add,
        vec![3],
        vec![2, 3],
    ));
    // (3, 1) + (3, 4) → (3, 4); covers column-vector broadcast
    out.push(tensor_binary_case_f32(
        "bcast/col_vec_add",
        OpKind::Add,
        vec![3, 1],
        vec![3, 4],
    ));
    // (1,) + (4,); scalar-like broadcast
    out.push(tensor_binary_case_f32(
        "bcast/scalar_mul",
        OpKind::Mul,
        vec![1],
        vec![4],
    ));
    out
}

// ── Matmul ──────────────────────────────────────────────────────────

fn matmul_cases() -> Vec<ConformanceCase> {
    let mut out = Vec::new();
    for (m, k, n) in [(2usize, 3, 2), (4, 4, 4), (8, 16, 8), (16, 32, 16)] {
        out.push(matmul_case(m, k, n));
    }
    out
}

fn matmul_case(m: usize, k: usize, n: usize) -> ConformanceCase {
    let name = format!("matmul/f32/{m}x{k}x{n}");
    let a_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.1 + 0.05).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.05 - 0.1).collect();
    let graph = FunctionGraph {
        name: name.clone(),
        nodes: vec![
            GraphNode::Param(Param { name: "a".into(), data_type: "f32".into() }),
            GraphNode::Param(Param { name: "b".into(), data_type: "f32".into() }),
            GraphNode::Apply {
                op: OpKind::Custom("matmul".into()),
                output_type: "f32".into(),
            },
        ],
        wires: vec![
            Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Move },
            Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Move },
        ],
        output_node: 2,
        return_type: "f32".into(),
    };
    let mut inputs = Inputs::new();
    inputs.insert(
        "a".into(),
        Value::Tensor(HostTensor::from_f32(vec![m, k], &a_data).unwrap()),
    );
    inputs.insert(
        "b".into(),
        Value::Tensor(HostTensor::from_f32(vec![k, n], &b_data).unwrap()),
    );
    ConformanceCase { name, graph, inputs, family: DtypeFamily::Float }
}

// ── Reduce ──────────────────────────────────────────────────────────

fn reduce_cases() -> Vec<ConformanceCase> {
    let mut out = Vec::new();
    for op in ["reduce_sum", "reduce_mean", "reduce_max"] {
        for (shape, axes, keepdim) in [
            (vec![3usize, 4], "[1]", false),
            (vec![2, 3, 4], "[0]", false),
            (vec![2, 3, 4], "[0, 2]", true),
        ] {
            out.push(reduce_case(op, shape, axes, keepdim));
        }
    }
    out
}

fn reduce_case(op: &str, shape: Vec<usize>, axes: &str, keepdim: bool) -> ConformanceCase {
    let name = format!("reduce/{op}/{shape:?}/axes={axes}/keepdim={keepdim}");
    let numel: usize = shape.iter().product::<usize>().max(1);
    let data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 + 1.0).collect();
    let graph = FunctionGraph {
        name: name.clone(),
        nodes: vec![
            GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
            GraphNode::Literal {
                value: LiteralValue::String(axes.into()),
                data_type: "string".into(),
            },
            GraphNode::Literal {
                value: LiteralValue::Bool(keepdim),
                data_type: "bool".into(),
            },
            GraphNode::Apply {
                op: OpKind::Custom(op.into()),
                output_type: "f32".into(),
            },
        ],
        wires: vec![
            Wire { from: 0, to: 3, port_in_idx: 0, wire_kind: WireKind::Move },
            Wire { from: 1, to: 3, port_in_idx: 1, wire_kind: WireKind::Move },
            Wire { from: 2, to: 3, port_in_idx: 2, wire_kind: WireKind::Move },
        ],
        output_node: 3,
        return_type: "f32".into(),
    };
    let mut inputs = Inputs::new();
    inputs.insert(
        "x".into(),
        Value::Tensor(HostTensor::from_f32(shape, &data).unwrap()),
    );
    ConformanceCase { name, graph, inputs, family: DtypeFamily::Float }
}

// ── Softmax ─────────────────────────────────────────────────────────

fn softmax_cases() -> Vec<ConformanceCase> {
    let mut out = Vec::new();
    for (shape, axis) in [
        (vec![5usize], -1),
        (vec![3, 4], -1),
        (vec![2, 3, 4], -1),
    ] {
        out.push(softmax_case(shape, axis));
    }
    out
}

fn softmax_case(shape: Vec<usize>, axis: i32) -> ConformanceCase {
    let name = format!("softmax/{shape:?}/axis={axis}");
    let numel: usize = shape.iter().product::<usize>().max(1);
    let data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let graph = FunctionGraph {
        name: name.clone(),
        nodes: vec![
            GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
            GraphNode::Literal {
                value: LiteralValue::I32(axis),
                data_type: "i32".into(),
            },
            GraphNode::Apply {
                op: OpKind::Custom("softmax".into()),
                output_type: "f32".into(),
            },
        ],
        wires: vec![
            Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Move },
            Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Move },
        ],
        output_node: 2,
        return_type: "f32".into(),
    };
    let mut inputs = Inputs::new();
    inputs.insert(
        "x".into(),
        Value::Tensor(HostTensor::from_f32(shape, &data).unwrap()),
    );
    ConformanceCase { name, graph, inputs, family: DtypeFamily::Float }
}

// ── Activations ─────────────────────────────────────────────────────

fn activation_cases() -> Vec<ConformanceCase> {
    let mut out = Vec::new();
    for op in ["relu", "sigmoid", "tanh", "gelu"] {
        for shape in [vec![5usize], vec![2, 4]] {
            out.push(activation_case(op, shape));
        }
    }
    out
}

fn activation_case(op: &str, shape: Vec<usize>) -> ConformanceCase {
    let name = format!("activation/{op}/{shape:?}");
    let numel: usize = shape.iter().product::<usize>().max(1);
    let data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let graph = FunctionGraph {
        name: name.clone(),
        nodes: vec![
            GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
            GraphNode::Apply {
                op: OpKind::Custom(op.into()),
                output_type: "f32".into(),
            },
        ],
        wires: vec![
            Wire { from: 0, to: 1, port_in_idx: 0, wire_kind: WireKind::Move },
        ],
        output_node: 1,
        return_type: "f32".into(),
    };
    let mut inputs = Inputs::new();
    inputs.insert(
        "x".into(),
        Value::Tensor(HostTensor::from_f32(shape, &data).unwrap()),
    );
    ConformanceCase { name, graph, inputs, family: DtypeFamily::Float }
}

// ── Transpose / reshape ─────────────────────────────────────────────

fn transpose_reshape_cases() -> Vec<ConformanceCase> {
    let mut out = Vec::new();
    out.push(transpose_case(vec![3, 4], "[1,0]"));
    out.push(reshape_case(vec![6], "[2,3]"));
    out.push(reshape_case(vec![2, 3, 4], "[6,4]"));
    out
}

fn transpose_case(shape: Vec<usize>, axes_json: &str) -> ConformanceCase {
    let name = format!("transpose/{shape:?}/{axes_json}");
    let numel: usize = shape.iter().product::<usize>().max(1);
    let data: Vec<f32> = (0..numel).map(|i| i as f32).collect();
    let graph = FunctionGraph {
        name: name.clone(),
        nodes: vec![
            GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
            GraphNode::Literal {
                value: LiteralValue::String(axes_json.into()),
                data_type: "string".into(),
            },
            GraphNode::Apply {
                op: OpKind::Custom("transpose".into()),
                output_type: "f32".into(),
            },
        ],
        wires: vec![
            Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Move },
            Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Move },
        ],
        output_node: 2,
        return_type: "f32".into(),
    };
    let mut inputs = Inputs::new();
    inputs.insert(
        "x".into(),
        Value::Tensor(HostTensor::from_f32(shape, &data).unwrap()),
    );
    ConformanceCase { name, graph, inputs, family: DtypeFamily::Float }
}

fn reshape_case(shape: Vec<usize>, target_json: &str) -> ConformanceCase {
    let name = format!("reshape/{shape:?}/{target_json}");
    let numel: usize = shape.iter().product::<usize>().max(1);
    let data: Vec<f32> = (0..numel).map(|i| i as f32).collect();
    let graph = FunctionGraph {
        name: name.clone(),
        nodes: vec![
            GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
            GraphNode::Literal {
                value: LiteralValue::String(target_json.into()),
                data_type: "string".into(),
            },
            GraphNode::Apply {
                op: OpKind::Custom("reshape".into()),
                output_type: "f32".into(),
            },
        ],
        wires: vec![
            Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Move },
            Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Move },
        ],
        output_node: 2,
        return_type: "f32".into(),
    };
    let mut inputs = Inputs::new();
    inputs.insert(
        "x".into(),
        Value::Tensor(HostTensor::from_f32(shape, &data).unwrap()),
    );
    ConformanceCase { name, graph, inputs, family: DtypeFamily::Float }
}

// ── End-to-end ──────────────────────────────────────────────────────

fn end_to_end_cases() -> Vec<ConformanceCase> {
    vec![attention_case(), mlp_case()]
}

fn attention_case() -> ConformanceCase {
    let seq_len = 4;
    let dim = 8;
    let scale = 1.0_f32 / (dim as f32).sqrt();
    let m = WireKind::Move;
    let nodes = vec![
        GraphNode::Param(Param { name: "q".into(), data_type: "f32".into() }),
        GraphNode::Param(Param { name: "k".into(), data_type: "f32".into() }),
        GraphNode::Param(Param { name: "v".into(), data_type: "f32".into() }),
        GraphNode::Literal { value: LiteralValue::String("[1,0]".into()), data_type: "string".into() },
        GraphNode::Apply { op: OpKind::Custom("transpose".into()), output_type: "f32".into() },
        GraphNode::Apply { op: OpKind::Custom("matmul".into()), output_type: "f32".into() },
        GraphNode::Literal { value: LiteralValue::F32(scale), data_type: "f32".into() },
        GraphNode::Apply { op: OpKind::Mul, output_type: "f32".into() },
        GraphNode::Literal { value: LiteralValue::I32(-1), data_type: "i32".into() },
        GraphNode::Apply { op: OpKind::Custom("softmax".into()), output_type: "f32".into() },
        GraphNode::Apply { op: OpKind::Custom("matmul".into()), output_type: "f32".into() },
    ];
    let wires = vec![
        Wire { from: 1, to: 4, port_in_idx: 0, wire_kind: m },
        Wire { from: 3, to: 4, port_in_idx: 1, wire_kind: m },
        Wire { from: 0, to: 5, port_in_idx: 0, wire_kind: m },
        Wire { from: 4, to: 5, port_in_idx: 1, wire_kind: m },
        Wire { from: 5, to: 7, port_in_idx: 0, wire_kind: m },
        Wire { from: 6, to: 7, port_in_idx: 1, wire_kind: m },
        Wire { from: 7, to: 9, port_in_idx: 0, wire_kind: m },
        Wire { from: 8, to: 9, port_in_idx: 1, wire_kind: m },
        Wire { from: 9, to: 10, port_in_idx: 0, wire_kind: m },
        Wire { from: 2, to: 10, port_in_idx: 1, wire_kind: m },
    ];
    let graph = FunctionGraph {
        name: "end_to_end/attention".into(),
        nodes,
        wires,
        output_node: 10,
        return_type: "f32".into(),
    };
    let mut inputs = Inputs::new();
    let mk = |seed: f32| -> HostTensor {
        let n = seq_len * dim;
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 + seed).collect();
        HostTensor::from_f32(vec![seq_len, dim], &data).unwrap()
    };
    inputs.insert("q".into(), Value::Tensor(mk(0.0)));
    inputs.insert("k".into(), Value::Tensor(mk(1.0)));
    inputs.insert("v".into(), Value::Tensor(mk(2.0)));
    ConformanceCase {
        name: "end_to_end/attention".into(),
        graph,
        inputs,
        family: DtypeFamily::Float,
    }
}

fn mlp_case() -> ConformanceCase {
    let dim = 4;
    let h = 8;
    let mk_weight = |name: &str, shape: Vec<usize>, seed: f32| -> GraphNode {
        let numel: usize = shape.iter().product();
        let mut bytes = Vec::with_capacity(numel * 4);
        for i in 0..numel {
            bytes.extend_from_slice(&((i as f32 * 0.05 + seed) as f32).to_le_bytes());
        }
        GraphNode::TensorLiteral(TensorLiteralData {
            data: Arc::from(bytes.into_boxed_slice()),
            shape,
            dtype: "f32".into(),
            name: name.into(),
        })
    };
    let m = WireKind::Move;
    let nodes = vec![
        GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
        mk_weight("w1", vec![dim, h], 0.1),
        mk_weight("b1", vec![h], 0.0),
        GraphNode::Apply { op: OpKind::Custom("matmul".into()), output_type: "f32".into() },
        GraphNode::Apply { op: OpKind::Add, output_type: "f32".into() },
        GraphNode::Apply { op: OpKind::Custom("relu".into()), output_type: "f32".into() },
        mk_weight("w2", vec![h, dim], 0.05),
        mk_weight("b2", vec![dim], 0.0),
        GraphNode::Apply { op: OpKind::Custom("matmul".into()), output_type: "f32".into() },
        GraphNode::Apply { op: OpKind::Add, output_type: "f32".into() },
        GraphNode::Literal { value: LiteralValue::I32(-1), data_type: "i32".into() },
        GraphNode::Apply { op: OpKind::Custom("softmax".into()), output_type: "f32".into() },
    ];
    let wires = vec![
        Wire { from: 0, to: 3, port_in_idx: 0, wire_kind: m },
        Wire { from: 1, to: 3, port_in_idx: 1, wire_kind: m },
        Wire { from: 3, to: 4, port_in_idx: 0, wire_kind: m },
        Wire { from: 2, to: 4, port_in_idx: 1, wire_kind: m },
        Wire { from: 4, to: 5, port_in_idx: 0, wire_kind: m },
        Wire { from: 5, to: 8, port_in_idx: 0, wire_kind: m },
        Wire { from: 6, to: 8, port_in_idx: 1, wire_kind: m },
        Wire { from: 8, to: 9, port_in_idx: 0, wire_kind: m },
        Wire { from: 7, to: 9, port_in_idx: 1, wire_kind: m },
        Wire { from: 9, to: 11, port_in_idx: 0, wire_kind: m },
        Wire { from: 10, to: 11, port_in_idx: 1, wire_kind: m },
    ];
    let graph = FunctionGraph {
        name: "end_to_end/mlp".into(),
        nodes,
        wires,
        output_node: 11,
        return_type: "f32".into(),
    };
    let mut inputs = Inputs::new();
    let data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.5 + 1.0).collect();
    inputs.insert(
        "x".into(),
        Value::Tensor(HostTensor::from_f32(vec![1, dim], &data).unwrap()),
    );
    ConformanceCase {
        name: "end_to_end/mlp".into(),
        graph,
        inputs,
        family: DtypeFamily::Float,
    }
}
