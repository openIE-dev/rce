//! IR-level sparse-MoE rewriter pass — shipping the
//! `MoeBlockTemplate` abstraction whose compile methods emit two
//! `FunctionGraph`s from one specification:
//!
//! - `compile_dense()`: every expert wired in. All N experts run; the
//!   per-expert weights (zero for non-routed) zero out non-routed
//!   contributions in the final reduce.
//! - `compile_sparse(&active_experts)`: ONLY the routed experts wired
//!   in. The non-routed experts' Apply nodes (matmul, mul, add) are
//!   absent from the graph entirely. Same numerical output as the
//!   dense form when `active_experts` carries the same routing
//!   decision; strictly fewer Apply nodes.
//!
//! **What this proves**: the substrate's IR carries the structure of
//! a sparse-MoE block such that a graph-rewriter pass can produce a
//! smaller graph from the dense form, given a routing decision. The
//! pass runs at *graph-build time* given a runtime mask (computed by
//! partial-evaluating the upstream router); it is the sparse
//! evaluator that the [`rce_op_nn_prims::sparse_moe_eval`] runtime
//! helper formalises at the Rust-fn level.
//!
//! Both graphs run through `interpret_with_config` with the same
//! input weights; the test asserts numerical equivalence and shows
//! node-count reduction.

use rce_vm_backend_cpu::{
    interpret_with_config, HostTensor, Inputs, InterpretConfig, OpDispatch, ScalarValue, Value,
};
use rce_graph::graph::{
    FunctionGraph, GraphNode, LiteralValue, OpKind, Param, Wire, WireKind,
};

#[derive(Clone, Copy, Debug)]
struct MoeBlockTemplate {
    num_experts: usize,
    d_model: usize,
}

impl MoeBlockTemplate {
    /// Build a dense `FunctionGraph`: every expert evaluated, each
    /// scaled by its (runtime-supplied) weight Param. The non-routed
    /// experts' weights are 0 so their contributions zero out at the
    /// reduce — but the Apply nodes still execute.
    ///
    /// Inputs (Param names):
    /// - `input` (f32, shape `[1, d_model]`)
    /// - `expert_W_<i>` (f32, shape `[d_model, d_model]`) for i in 0..N
    /// - `weight_<i>` (f32, scalar; shape `[]`) for i in 0..N
    fn compile_dense(&self) -> FunctionGraph {
        let n = self.num_experts;
        let m = WireKind::Move;
        let mut nodes: Vec<GraphNode> = Vec::new();

        // Param: input @ index 0
        nodes.push(p_param("input"));
        // Params: expert_W_<i> @ indices 1..=N
        for i in 0..n {
            nodes.push(p_param(&format!("expert_W_{i}")));
        }
        // Params: weight_<i> @ indices N+1..=2N
        for i in 0..n {
            nodes.push(p_param(&format!("weight_{i}")));
        }
        // Apply matmul @ indices 2N+1..=3N: matmul(input, expert_W_<i>)
        let matmul_base = 1 + 2 * n;
        for _ in 0..n {
            nodes.push(apply_custom("matmul"));
        }
        // Apply mul (broadcasting) @ indices 3N+1..=4N: mul(expert_out_<i>, weight_<i>)
        let mul_base = matmul_base + n;
        for _ in 0..n {
            nodes.push(GraphNode::Apply {
                op: OpKind::Mul,
                output_type: "f32".into(),
            });
        }
        // Add chain @ indices 4N+1..=5N-1: cumulative sum of N contributions.
        // (N-1 add nodes for a chain of N inputs.)
        let add_base = mul_base + n;
        for _ in 0..(n - 1) {
            nodes.push(GraphNode::Apply {
                op: OpKind::Add,
                output_type: "f32".into(),
            });
        }

        // Wires.
        let mut wires: Vec<Wire> = Vec::new();
        // Each matmul: matmul(input=0, expert_W_<i>=1+i)
        for i in 0..n {
            let mm_node = matmul_base + i;
            wires.push(Wire { from: 0, to: mm_node, port_in_idx: 0, wire_kind: m });
            wires.push(Wire { from: 1 + i, to: mm_node, port_in_idx: 1, wire_kind: m });
        }
        // Each mul: mul(expert_out_<i>=matmul_base+i, weight_<i>=1+n+i)
        for i in 0..n {
            let mu_node = mul_base + i;
            wires.push(Wire { from: matmul_base + i, to: mu_node, port_in_idx: 0, wire_kind: m });
            wires.push(Wire { from: 1 + n + i, to: mu_node, port_in_idx: 1, wire_kind: m });
        }
        // Add chain: each Add takes (running_sum, contrib_i).
        // First Add: contrib_0 + contrib_1 → running.
        // Next Add: running + contrib_2.
        // ... last Add: running + contrib_{N-1}.
        for i in 0..(n - 1) {
            let add_node = add_base + i;
            let lhs = if i == 0 { mul_base } else { add_base + i - 1 };
            let rhs = mul_base + i + 1;
            wires.push(Wire { from: lhs, to: add_node, port_in_idx: 0, wire_kind: m });
            wires.push(Wire { from: rhs, to: add_node, port_in_idx: 1, wire_kind: m });
        }

        let output_node = add_base + (n - 1) - 1;
        FunctionGraph {
            name: "moe_block_dense".into(),
            nodes,
            wires,
            output_node,
            return_type: "f32".into(),
        }
    }

    /// Build a sparse `FunctionGraph`: only the active experts are
    /// wired in. `active` is a list of `(expert_id, weight)` pairs;
    /// other experts simply do not appear in the graph at all — no
    /// matmul, no mul, no add for them.
    ///
    /// The weight for each active expert is baked into the graph as
    /// an f32 Literal (rather than a Param), since the routing
    /// decision is fixed at graph-build time.
    ///
    /// Inputs (Param names):
    /// - `input` (f32, shape `[1, d_model]`)
    /// - `expert_W_<i>` (f32, shape `[d_model, d_model]`) for `i` in
    ///   the active set only.
    fn compile_sparse(&self, active: &[(u32, f32)]) -> FunctionGraph {
        assert!(!active.is_empty(), "sparse compile requires ≥1 active expert");
        let k = active.len();
        let m = WireKind::Move;
        let mut nodes: Vec<GraphNode> = Vec::new();

        // 0: Param input.
        nodes.push(p_param("input"));
        // 1..=K: Param expert_W_<i> for the K active experts.
        for (eid, _w) in active {
            nodes.push(p_param(&format!("expert_W_{eid}")));
        }
        // K+1..=2K: Literal weight scalar.
        let lit_base = 1 + k;
        for (_eid, w) in active {
            nodes.push(GraphNode::Literal {
                value: LiteralValue::F32(*w),
                data_type: "f32".into(),
            });
        }
        // 2K+1..=3K: matmul(input, expert_W_<eid>) for each active expert.
        let matmul_base = 1 + 2 * k;
        for _ in 0..k {
            nodes.push(apply_custom("matmul"));
        }
        // 3K+1..=4K: mul(expert_out, weight_literal).
        let mul_base = matmul_base + k;
        for _ in 0..k {
            nodes.push(GraphNode::Apply {
                op: OpKind::Mul,
                output_type: "f32".into(),
            });
        }
        // 4K+1..=5K-1: add chain (K-1 nodes if K>1, else 0 nodes).
        let add_base = mul_base + k;
        if k > 1 {
            for _ in 0..(k - 1) {
                nodes.push(GraphNode::Apply {
                    op: OpKind::Add,
                    output_type: "f32".into(),
                });
            }
        }

        let mut wires: Vec<Wire> = Vec::new();
        for i in 0..k {
            let mm_node = matmul_base + i;
            wires.push(Wire { from: 0, to: mm_node, port_in_idx: 0, wire_kind: m });
            wires.push(Wire { from: 1 + i, to: mm_node, port_in_idx: 1, wire_kind: m });
        }
        for i in 0..k {
            let mu_node = mul_base + i;
            wires.push(Wire { from: matmul_base + i, to: mu_node, port_in_idx: 0, wire_kind: m });
            wires.push(Wire { from: lit_base + i, to: mu_node, port_in_idx: 1, wire_kind: m });
        }
        if k > 1 {
            for i in 0..(k - 1) {
                let add_node = add_base + i;
                let lhs = if i == 0 { mul_base } else { add_base + i - 1 };
                let rhs = mul_base + i + 1;
                wires.push(Wire { from: lhs, to: add_node, port_in_idx: 0, wire_kind: m });
                wires.push(Wire { from: rhs, to: add_node, port_in_idx: 1, wire_kind: m });
            }
        }

        let output_node = if k == 1 { mul_base } else { add_base + (k - 1) - 1 };
        FunctionGraph {
            name: "moe_block_sparse".into(),
            nodes,
            wires,
            output_node,
            return_type: "f32".into(),
        }
    }
}

fn p_param(name: &str) -> GraphNode {
    GraphNode::Param(Param { name: name.into(), data_type: "f32".into() })
}

fn apply_custom(name: &str) -> GraphNode {
    GraphNode::Apply {
        op: OpKind::Custom(name.into()),
        output_type: "f32".into(),
    }
}

fn count_apply_nodes(graph: &FunctionGraph) -> usize {
    graph.nodes.iter().filter(|n| matches!(n, GraphNode::Apply { .. })).count()
}

// ── Tests ──────────────────────────────────────────────────────────

#[test]
fn rewriter_dense_and_sparse_produce_equivalent_output() {
    let template = MoeBlockTemplate { num_experts: 8, d_model: 16 };

    // Routing decision: experts 1 and 5 are top-2 with softmax weights.
    let active: Vec<(u32, f32)> = vec![(1, 0.3), (5, 0.7)];

    // Deterministic input + per-expert weights.
    let input: Vec<f32> = (0..template.d_model).map(|i| (i as f32 * 0.013).sin()).collect();
    let expert_w: Vec<Vec<f32>> = (0..template.num_experts)
        .map(|e| (0..template.d_model * template.d_model)
            .map(|i| ((i as f32 * 0.001 + e as f32 * 0.05) - 0.3).cos() * 0.4)
            .collect())
        .collect();

    // Build dense weights vector: 0 everywhere except the active ones.
    let dense_weights: Vec<f32> = {
        let mut v = vec![0.0f32; template.num_experts];
        for (eid, w) in &active {
            v[*eid as usize] = *w;
        }
        v
    };

    // ── Run dense graph ────────────────────────────────────
    let dense_graph = template.compile_dense();
    let mut dense_inputs = Inputs::new();
    dense_inputs.insert(
        "input".into(),
        Value::Tensor(HostTensor::from_f32(vec![1, template.d_model], &input).unwrap()),
    );
    for (e, w) in expert_w.iter().enumerate() {
        dense_inputs.insert(
            format!("expert_W_{e}"),
            Value::Tensor(HostTensor::from_f32(vec![template.d_model, template.d_model], w).unwrap()),
        );
    }
    for (e, w) in dense_weights.iter().enumerate() {
        // Weight as 0-D scalar tensor.
        dense_inputs.insert(
            format!("weight_{e}"),
            Value::Tensor(HostTensor::from_f32(vec![], &[*w]).unwrap()),
        );
    }
    // No external ops needed beyond the cpu kernel layer (matmul +
    // primitive Add/Mul). Empty op chain → fall through to cpu.
    let ops: Vec<&dyn OpDispatch> = vec![];
    let cfg = InterpretConfig { leaves: &[], ops: &ops };
    let dense_result = interpret_with_config(&dense_graph, &dense_inputs, cfg).expect("dense");
    let dense_out = match dense_result {
        Value::Tensor(t) => t.as_f32().expect("f32"),
        other => panic!("dense returned {other:?}"),
    };

    // ── Run sparse graph ───────────────────────────────────
    let sparse_graph = template.compile_sparse(&active);
    let mut sparse_inputs = Inputs::new();
    sparse_inputs.insert(
        "input".into(),
        Value::Tensor(HostTensor::from_f32(vec![1, template.d_model], &input).unwrap()),
    );
    for (eid, _w) in &active {
        sparse_inputs.insert(
            format!("expert_W_{eid}"),
            Value::Tensor(HostTensor::from_f32(
                vec![template.d_model, template.d_model],
                &expert_w[*eid as usize],
            ).unwrap()),
        );
    }
    let ops: Vec<&dyn OpDispatch> = vec![];
    let cfg = InterpretConfig { leaves: &[], ops: &ops };
    let sparse_result = interpret_with_config(&sparse_graph, &sparse_inputs, cfg).expect("sparse");
    let sparse_out = match sparse_result {
        Value::Tensor(t) => t.as_f32().expect("f32"),
        other => panic!("sparse returned {other:?}"),
    };

    // ── Numerical equivalence ──────────────────────────────
    assert_eq!(dense_out.len(), sparse_out.len(), "output length mismatch");
    let max_err = dense_out.iter().zip(sparse_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-5, "rewriter output diverges: max_abs_err = {max_err}");

    // ── Node-count reduction ───────────────────────────────
    let dense_apply = count_apply_nodes(&dense_graph);
    let sparse_apply = count_apply_nodes(&sparse_graph);
    let _ = ScalarValue::I32(0);
    eprintln!(
        "moe rewriter: dense={dense_apply} apply nodes, sparse={sparse_apply} apply nodes \
         (output match: max_abs_err={max_err:.3e})"
    );

    // For N=8, K=2: dense = 8 matmul + 8 mul + 7 add = 23 apply.
    // Sparse = 2 matmul + 2 mul + 1 add = 5 apply.
    assert_eq!(dense_apply, 8 + 8 + 7);
    assert_eq!(sparse_apply, 2 + 2 + 1);
    assert!(sparse_apply * 4 < dense_apply, "expected ≥4× reduction");
}

#[test]
fn rewriter_with_single_active_expert_collapses_to_one_branch() {
    // K=1 case: only 1 expert active. The sparse graph degenerates
    // to (matmul → mul) — no add chain. Useful to catch off-by-one.
    let template = MoeBlockTemplate { num_experts: 4, d_model: 8 };
    let active: Vec<(u32, f32)> = vec![(2, 1.0)];

    let input: Vec<f32> = (0..template.d_model).map(|i| (i as f32) * 0.1).collect();
    let expert_w: Vec<Vec<f32>> = (0..template.num_experts)
        .map(|e| (0..template.d_model * template.d_model).map(|i| (i + e * 100) as f32 * 0.01).collect())
        .collect();

    let sparse_graph = template.compile_sparse(&active);
    let sparse_apply = count_apply_nodes(&sparse_graph);
    assert_eq!(sparse_apply, 2, "K=1 sparse graph has only matmul + mul");

    let mut inputs = Inputs::new();
    inputs.insert(
        "input".into(),
        Value::Tensor(HostTensor::from_f32(vec![1, template.d_model], &input).unwrap()),
    );
    inputs.insert(
        format!("expert_W_2"),
        Value::Tensor(HostTensor::from_f32(
            vec![template.d_model, template.d_model],
            &expert_w[2],
        ).unwrap()),
    );
    let ops: Vec<&dyn OpDispatch> = vec![];
    let cfg = InterpretConfig { leaves: &[], ops: &ops };
    let result = interpret_with_config(&sparse_graph, &inputs, cfg).expect("interpret");
    let got = match result { Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!() };

    // Reference: input [1, d_model] @ expert_W_2 [d_model, d_model] * 1.0
    let mut want = vec![0.0f32; template.d_model];
    for j in 0..template.d_model {
        let mut acc = 0.0f32;
        for k in 0..template.d_model {
            acc += input[k] * expert_w[2][k * template.d_model + j];
        }
        want[j] = acc * 1.0;
    }
    let max_err = want.iter().zip(got.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-5, "K=1 sparse divergence: {max_err}");
}

#[test]
fn rewriter_node_count_scales_linearly_with_k() {
    // Verify: dense apply count is independent of routing; sparse
    // apply count scales with K. This is the substrate-level claim:
    // the rewriter strictly reduces graph size as routing sparsity
    // increases.
    let template = MoeBlockTemplate { num_experts: 16, d_model: 8 };
    let dense_graph = template.compile_dense();
    let dense_apply = count_apply_nodes(&dense_graph);
    // 16 matmul + 16 mul + 15 add = 47.
    assert_eq!(dense_apply, 16 + 16 + 15);

    for k in [1, 2, 4, 8] {
        let active: Vec<(u32, f32)> = (0..k as u32).map(|i| (i, 1.0 / k as f32)).collect();
        let sparse_graph = template.compile_sparse(&active);
        let sparse_apply = count_apply_nodes(&sparse_graph);
        // K matmul + K mul + (K-1) add (or 0 add if K=1).
        let expected = if k == 1 { 2 } else { k + k + (k - 1) };
        assert_eq!(sparse_apply, expected, "K={k}");
        eprintln!(
            "K={k}: sparse_apply={sparse_apply} (dense={dense_apply}, reduction={:.1}x)",
            dense_apply as f32 / sparse_apply as f32,
        );
    }
}
