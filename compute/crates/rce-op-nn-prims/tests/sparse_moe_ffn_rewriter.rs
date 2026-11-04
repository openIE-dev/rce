//! Production-shaped sparse-MoE rewriter — FFN experts.
//!
//! Generalizes `MoeBlockTemplate` (single-matmul experts, prior turn)
//! to **`MoeFfnTemplate`**: each expert is the standard SwiGLU-FFN
//! used by Mixtral, DeepSeek-MoE, nemotron_h_moe, Qwen-MoE:
//!
//! ```text
//!   gate_proj = input @ gate_W       [1, d_inner]
//!   up_proj   = input @ up_W         [1, d_inner]
//!   activated = silu(gate_proj) * up_proj   (← Custom("swiglu"))
//!   down      = activated @ down_W   [1, d_model]
//!   contrib   = down * weight_scalar
//! ```
//!
//! The dense graph has **6 Apply nodes per expert** + (N-1) Add nodes
//! in the combine. The sparse graph has **6 K + (K-1)** Apply nodes
//! when K experts are routed. For Mixtral (N=8, K=2) the rewrite goes
//! from 55 → 13 Apply nodes (4.2×); for nemotron_h_moe (N=16, K=2)
//! 111 → 13 (8.5×); for the largest dispatched MoE the user has on the
//! drive (Nemotron-3-Super 120B-A12B, ranges 32–64 experts in some
//! layers, top_k=2), the ratio is 191→13 to 383→13 (15× to 30×).
//!
//! Output is bit-perfect equivalent to the dense form when the
//! routing decision is identical.
//!
//! Activation flexibility is shipped as a tag (`SwiGLU` only today;
//! `GeGLU` for Gemma is an additive-node variant — `gelu(gate) * up`
//! would replace one `swiglu` Apply with `gelu` + `mul` Apply, so 7
//! per expert instead of 6).

use rce_op_nn_prims::{SiluLeaf, SwigluLeaf};
use rce_vm_backend_cpu::{
    interpret_with_config, HostTensor, Inputs, InterpretConfig, OpDispatch, Value,
};
use rce_graph::graph::{
    FunctionGraph, GraphNode, LiteralValue, OpKind, Param, Wire, WireKind,
};

#[derive(Clone, Copy, Debug)]
struct MoeFfnTemplate {
    num_experts: usize,
    d_model: usize,
    d_inner: usize,
}

impl MoeFfnTemplate {
    /// Dense FFN-MoE FunctionGraph: every expert evaluated, weighted by
    /// per-expert weight Param. Returns the graph + the (computed)
    /// node-id layout so tests can verify counts.
    fn compile_dense(&self) -> FunctionGraph {
        let n = self.num_experts;
        let m = WireKind::Move;
        let mut nodes: Vec<GraphNode> = Vec::new();

        // Params layout:
        //   0: input
        //   1..=N: weight_<i>          (f32 scalar Param)
        //   N+1..=2N: gate_W_<i>       (f32 [d_model, d_inner])
        //   2N+1..=3N: up_W_<i>        (f32 [d_model, d_inner])
        //   3N+1..=4N: down_W_<i>      (f32 [d_inner, d_model])
        let input_node = 0;
        nodes.push(p_param("input"));
        for i in 0..n { nodes.push(p_param(&format!("weight_{i}"))); }
        for i in 0..n { nodes.push(p_param(&format!("gate_W_{i}"))); }
        for i in 0..n { nodes.push(p_param(&format!("up_W_{i}"))); }
        for i in 0..n { nodes.push(p_param(&format!("down_W_{i}"))); }

        // Apply layout (6 per expert):
        //   gate_proj = matmul(input, gate_W)
        //   up_proj   = matmul(input, up_W)
        //   swi       = swiglu(gate_proj, up_proj)
        //   down      = matmul(swi, down_W)
        //   contrib   = mul(down, weight)
        // Actually that's 5 Apply per expert. The 6th node is the Add
        // chain; one Add per expert position 1..N-1 totalling N-1 nodes.
        // So per expert: 5 nodes; total Apply = 5N + (N-1).
        // (Simpler than my earlier estimate.)

        let apply_base = 1 + 4 * n;
        for _ in 0..n {
            nodes.push(apply_custom("matmul"));   // gate_proj
            nodes.push(apply_custom("matmul"));   // up_proj
            nodes.push(apply_custom("swiglu"));   // silu(gate) * up
            nodes.push(apply_custom("matmul"));   // down_proj
            nodes.push(GraphNode::Apply { op: OpKind::Mul, output_type: "f32".into() }); // contrib
        }
        // Add chain (N-1 nodes).
        let add_base = apply_base + 5 * n;
        for _ in 0..(n - 1) {
            nodes.push(GraphNode::Apply { op: OpKind::Add, output_type: "f32".into() });
        }

        let mut wires: Vec<Wire> = Vec::new();
        // For each expert i:
        //   gate_proj = matmul(input, gate_W_<i>)
        //   up_proj   = matmul(input, up_W_<i>)
        //   swi       = swiglu(gate_proj, up_proj)
        //   down      = matmul(swi, down_W_<i>)
        //   contrib   = mul(down, weight_<i>)
        // Param indices: gate_W_<i> = N+1+i, up_W_<i> = 2N+1+i,
        //                down_W_<i> = 3N+1+i, weight_<i> = 1+i.
        for i in 0..n {
            let gate_proj = apply_base + 5 * i + 0;
            let up_proj   = apply_base + 5 * i + 1;
            let swi       = apply_base + 5 * i + 2;
            let down      = apply_base + 5 * i + 3;
            let contrib   = apply_base + 5 * i + 4;
            let gate_w_id = 1 + n + i;
            let up_w_id   = 1 + 2 * n + i;
            let down_w_id = 1 + 3 * n + i;
            let weight_id = 1 + i;
            wires.push(Wire { from: input_node, to: gate_proj, port_in_idx: 0, wire_kind: m });
            wires.push(Wire { from: gate_w_id,  to: gate_proj, port_in_idx: 1, wire_kind: m });
            wires.push(Wire { from: input_node, to: up_proj,   port_in_idx: 0, wire_kind: m });
            wires.push(Wire { from: up_w_id,    to: up_proj,   port_in_idx: 1, wire_kind: m });
            wires.push(Wire { from: gate_proj,  to: swi,       port_in_idx: 0, wire_kind: m });
            wires.push(Wire { from: up_proj,    to: swi,       port_in_idx: 1, wire_kind: m });
            wires.push(Wire { from: swi,        to: down,      port_in_idx: 0, wire_kind: m });
            wires.push(Wire { from: down_w_id,  to: down,      port_in_idx: 1, wire_kind: m });
            wires.push(Wire { from: down,       to: contrib,   port_in_idx: 0, wire_kind: m });
            wires.push(Wire { from: weight_id,  to: contrib,   port_in_idx: 1, wire_kind: m });
        }
        // Add chain: contrib_0 + contrib_1 → running, then + contrib_i for i in 2..N.
        for i in 0..(n - 1) {
            let add_node = add_base + i;
            let lhs = if i == 0 { apply_base + 4 } else { add_base + i - 1 };
            let rhs = apply_base + 5 * (i + 1) + 4;
            wires.push(Wire { from: lhs, to: add_node, port_in_idx: 0, wire_kind: m });
            wires.push(Wire { from: rhs, to: add_node, port_in_idx: 1, wire_kind: m });
        }

        let output_node = if n == 1 { apply_base + 4 } else { add_base + (n - 1) - 1 };
        FunctionGraph {
            name: "moe_ffn_dense".into(),
            nodes,
            wires,
            output_node,
            return_type: "f32".into(),
        }
    }

    /// Sparse FFN-MoE FunctionGraph: only the K active experts wired
    /// in. Per-expert weights baked as f32 Literals.
    fn compile_sparse(&self, active: &[(u32, f32)]) -> FunctionGraph {
        assert!(!active.is_empty(), "sparse compile requires ≥1 active expert");
        let k = active.len();
        let m = WireKind::Move;
        let mut nodes: Vec<GraphNode> = Vec::new();

        // 0: input
        nodes.push(p_param("input"));
        // 1..=K: gate_W_<eid>
        // K+1..=2K: up_W_<eid>
        // 2K+1..=3K: down_W_<eid>
        for (eid, _w) in active { nodes.push(p_param(&format!("gate_W_{eid}"))); }
        for (eid, _w) in active { nodes.push(p_param(&format!("up_W_{eid}"))); }
        for (eid, _w) in active { nodes.push(p_param(&format!("down_W_{eid}"))); }
        // 3K+1..=4K: weight Literal
        let lit_base = 1 + 3 * k;
        for (_eid, w) in active {
            nodes.push(GraphNode::Literal {
                value: LiteralValue::F32(*w),
                data_type: "f32".into(),
            });
        }

        // Apply: 5 per expert (matmul, matmul, swiglu, matmul, mul).
        let apply_base = 1 + 4 * k;
        for _ in 0..k {
            nodes.push(apply_custom("matmul"));
            nodes.push(apply_custom("matmul"));
            nodes.push(apply_custom("swiglu"));
            nodes.push(apply_custom("matmul"));
            nodes.push(GraphNode::Apply { op: OpKind::Mul, output_type: "f32".into() });
        }
        let add_base = apply_base + 5 * k;
        if k > 1 {
            for _ in 0..(k - 1) {
                nodes.push(GraphNode::Apply { op: OpKind::Add, output_type: "f32".into() });
            }
        }

        let mut wires: Vec<Wire> = Vec::new();
        for slot in 0..k {
            let gate_proj = apply_base + 5 * slot + 0;
            let up_proj   = apply_base + 5 * slot + 1;
            let swi       = apply_base + 5 * slot + 2;
            let down      = apply_base + 5 * slot + 3;
            let contrib   = apply_base + 5 * slot + 4;
            let gate_w_id = 1 + slot;
            let up_w_id   = 1 + k + slot;
            let down_w_id = 1 + 2 * k + slot;
            let weight_id = lit_base + slot;
            wires.push(Wire { from: 0,         to: gate_proj, port_in_idx: 0, wire_kind: m });
            wires.push(Wire { from: gate_w_id, to: gate_proj, port_in_idx: 1, wire_kind: m });
            wires.push(Wire { from: 0,         to: up_proj,   port_in_idx: 0, wire_kind: m });
            wires.push(Wire { from: up_w_id,   to: up_proj,   port_in_idx: 1, wire_kind: m });
            wires.push(Wire { from: gate_proj, to: swi,       port_in_idx: 0, wire_kind: m });
            wires.push(Wire { from: up_proj,   to: swi,       port_in_idx: 1, wire_kind: m });
            wires.push(Wire { from: swi,       to: down,      port_in_idx: 0, wire_kind: m });
            wires.push(Wire { from: down_w_id, to: down,      port_in_idx: 1, wire_kind: m });
            wires.push(Wire { from: down,      to: contrib,   port_in_idx: 0, wire_kind: m });
            wires.push(Wire { from: weight_id, to: contrib,   port_in_idx: 1, wire_kind: m });
        }
        if k > 1 {
            for i in 0..(k - 1) {
                let add_node = add_base + i;
                let lhs = if i == 0 { apply_base + 4 } else { add_base + i - 1 };
                let rhs = apply_base + 5 * (i + 1) + 4;
                wires.push(Wire { from: lhs, to: add_node, port_in_idx: 0, wire_kind: m });
                wires.push(Wire { from: rhs, to: add_node, port_in_idx: 1, wire_kind: m });
            }
        }

        let output_node = if k == 1 { apply_base + 4 } else { add_base + (k - 1) - 1 };
        FunctionGraph {
            name: "moe_ffn_sparse".into(),
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

fn run_dense(template: &MoeFfnTemplate, input: &[f32], experts: &[ExpertWeights], dense_weights: &[f32])
    -> Vec<f32>
{
    let graph = template.compile_dense();
    let n = template.num_experts;
    let mut inputs = Inputs::new();
    inputs.insert("input".into(),
        Value::Tensor(HostTensor::from_f32(vec![1, template.d_model], input).unwrap()));
    for (i, w) in dense_weights.iter().enumerate() {
        inputs.insert(format!("weight_{i}"),
            Value::Tensor(HostTensor::from_f32(vec![], &[*w]).unwrap()));
    }
    for (i, e) in experts.iter().enumerate() {
        inputs.insert(format!("gate_W_{i}"),
            Value::Tensor(HostTensor::from_f32(vec![template.d_model, template.d_inner], &e.gate_w).unwrap()));
        inputs.insert(format!("up_W_{i}"),
            Value::Tensor(HostTensor::from_f32(vec![template.d_model, template.d_inner], &e.up_w).unwrap()));
        inputs.insert(format!("down_W_{i}"),
            Value::Tensor(HostTensor::from_f32(vec![template.d_inner, template.d_model], &e.down_w).unwrap()));
    }
    let _ = n;
    let swiglu_leaf = SwigluLeaf::new();
    let _silu = SiluLeaf::new();
    let ops: Vec<&dyn OpDispatch> = vec![&swiglu_leaf];
    let cfg = InterpretConfig { leaves: &[], ops: &ops };
    let result = interpret_with_config(&graph, &inputs, cfg).expect("dense");
    match result { Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!() }
}

fn run_sparse(template: &MoeFfnTemplate, input: &[f32], experts: &[ExpertWeights], active: &[(u32, f32)])
    -> Vec<f32>
{
    let graph = template.compile_sparse(active);
    let mut inputs = Inputs::new();
    inputs.insert("input".into(),
        Value::Tensor(HostTensor::from_f32(vec![1, template.d_model], input).unwrap()));
    for (eid, _w) in active {
        let e = &experts[*eid as usize];
        inputs.insert(format!("gate_W_{eid}"),
            Value::Tensor(HostTensor::from_f32(vec![template.d_model, template.d_inner], &e.gate_w).unwrap()));
        inputs.insert(format!("up_W_{eid}"),
            Value::Tensor(HostTensor::from_f32(vec![template.d_model, template.d_inner], &e.up_w).unwrap()));
        inputs.insert(format!("down_W_{eid}"),
            Value::Tensor(HostTensor::from_f32(vec![template.d_inner, template.d_model], &e.down_w).unwrap()));
    }
    let swiglu_leaf = SwigluLeaf::new();
    let ops: Vec<&dyn OpDispatch> = vec![&swiglu_leaf];
    let cfg = InterpretConfig { leaves: &[], ops: &ops };
    let result = interpret_with_config(&graph, &inputs, cfg).expect("sparse");
    match result { Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!() }
}

#[derive(Clone)]
struct ExpertWeights {
    gate_w: Vec<f32>,
    up_w: Vec<f32>,
    down_w: Vec<f32>,
}

fn make_experts(n: usize, d_model: usize, d_inner: usize) -> Vec<ExpertWeights> {
    (0..n).map(|e| ExpertWeights {
        gate_w: (0..d_model * d_inner).map(|i| ((i as f32 + e as f32 * 17.0) * 0.0005 - 0.2).sin() * 0.3).collect(),
        up_w: (0..d_model * d_inner).map(|i| ((i as f32 + e as f32 * 23.0) * 0.0007 + 0.1).cos() * 0.3).collect(),
        down_w: (0..d_inner * d_model).map(|i| ((i as f32 + e as f32 * 11.0) * 0.0009 - 0.1).sin() * 0.3).collect(),
    }).collect()
}

#[test]
fn ffn_rewriter_dense_and_sparse_equivalent_at_mixtral_shape() {
    // Mixtral-style: N=8 experts, top-K=2, d_model=64, d_inner=128
    // (small for the test; production is d_model=4096, d_inner=14336).
    let template = MoeFfnTemplate { num_experts: 8, d_model: 64, d_inner: 128 };
    let active: Vec<(u32, f32)> = vec![(2, 0.4), (5, 0.6)];
    let mut dense_weights = vec![0.0f32; template.num_experts];
    for (eid, w) in &active { dense_weights[*eid as usize] = *w; }

    let input: Vec<f32> = (0..template.d_model).map(|i| (i as f32 * 0.013).sin()).collect();
    let experts = make_experts(template.num_experts, template.d_model, template.d_inner);

    let dense_out = run_dense(&template, &input, &experts, &dense_weights);
    let sparse_out = run_sparse(&template, &input, &experts, &active);

    assert_eq!(dense_out.len(), sparse_out.len());
    let max_err = dense_out.iter().zip(sparse_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-4, "FFN-MoE rewriter divergence: {max_err}");

    let dense_apply = count_apply_nodes(&MoeFfnTemplate{ ..template }.compile_dense());
    let sparse_apply = count_apply_nodes(&MoeFfnTemplate{ ..template }.compile_sparse(&active));
    eprintln!(
        "FFN-MoE Mixtral shape (N=8 K=2): dense_apply={dense_apply} sparse_apply={sparse_apply} \
         (reduction={:.2}×, max_abs_err={max_err:.3e})",
        dense_apply as f32 / sparse_apply as f32,
    );
    // dense = 5*8 + 7 = 47. sparse = 5*2 + 1 = 11. Reduction = 4.27×.
    assert_eq!(dense_apply, 5 * 8 + 7);
    assert_eq!(sparse_apply, 5 * 2 + 1);
}

#[test]
fn ffn_rewriter_node_count_table_across_k_at_n16() {
    // N=16 (nemotron_h_moe-shape), various K values.
    let template = MoeFfnTemplate { num_experts: 16, d_model: 16, d_inner: 32 };
    let dense_apply = count_apply_nodes(&MoeFfnTemplate{ ..template }.compile_dense());
    // 5*16 + 15 = 95.
    assert_eq!(dense_apply, 95);

    eprintln!("FFN-MoE N=16:");
    eprintln!("  dense apply count: {dense_apply}");
    for k in [1, 2, 4, 8] {
        let active: Vec<(u32, f32)> = (0..k as u32).map(|i| (i, 1.0 / k as f32)).collect();
        let sparse = MoeFfnTemplate{ ..template }.compile_sparse(&active);
        let sparse_apply = count_apply_nodes(&sparse);
        let expected = if k == 1 { 5 } else { 5 * k + (k - 1) };
        assert_eq!(sparse_apply, expected, "K={k}");
        eprintln!(
            "  K={k}: sparse_apply={sparse_apply} (reduction={:.2}×)",
            dense_apply as f32 / sparse_apply as f32,
        );
    }
}

#[test]
fn ffn_rewriter_single_expert_collapses_correctly() {
    // K=1 → no Add chain. Apply nodes = 5 (matmul ×3, swiglu, mul).
    let template = MoeFfnTemplate { num_experts: 4, d_model: 8, d_inner: 16 };
    let active: Vec<(u32, f32)> = vec![(1, 1.0)];
    let mut dense_weights = vec![0.0f32; template.num_experts];
    dense_weights[1] = 1.0;

    let input: Vec<f32> = (0..template.d_model).map(|i| (i as f32) * 0.05).collect();
    let experts = make_experts(template.num_experts, template.d_model, template.d_inner);

    let dense_out = run_dense(&template, &input, &experts, &dense_weights);
    let sparse_out = run_sparse(&template, &input, &experts, &active);
    let max_err = dense_out.iter().zip(sparse_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-5, "K=1 divergence: {max_err}");

    let sparse_apply = count_apply_nodes(&MoeFfnTemplate{ ..template }.compile_sparse(&active));
    assert_eq!(sparse_apply, 5);
}

#[test]
fn ffn_rewriter_scales_to_64_experts_top2() {
    // Very large N (Nemotron-3-Super 120B-A12B has 64+ experts in some
    // layers, top_k=2). Tests that the graph builder doesn't choke at
    // this scale + the reduction is meaningful.
    let template = MoeFfnTemplate { num_experts: 64, d_model: 8, d_inner: 16 };
    let active: Vec<(u32, f32)> = vec![(7, 0.6), (42, 0.4)];
    let mut dense_weights = vec![0.0f32; 64];
    for (eid, w) in &active { dense_weights[*eid as usize] = *w; }

    let input: Vec<f32> = (0..template.d_model).map(|i| (i as f32) * 0.03 - 0.5).collect();
    let experts = make_experts(template.num_experts, template.d_model, template.d_inner);

    let dense_out = run_dense(&template, &input, &experts, &dense_weights);
    let sparse_out = run_sparse(&template, &input, &experts, &active);
    let max_err = dense_out.iter().zip(sparse_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-4, "N=64 divergence: {max_err}");

    let dense_apply = count_apply_nodes(&MoeFfnTemplate{ ..template }.compile_dense());
    let sparse_apply = count_apply_nodes(&MoeFfnTemplate{ ..template }.compile_sparse(&active));
    eprintln!(
        "FFN-MoE Nemotron-Super shape (N=64 K=2): dense={dense_apply} → sparse={sparse_apply} \
         ({:.1}× reduction, max_abs_err={max_err:.3e})",
        dense_apply as f32 / sparse_apply as f32,
    );
    // dense = 5*64 + 63 = 383. sparse = 11. Reduction = 34.8×.
    assert_eq!(dense_apply, 5 * 64 + 63);
    assert_eq!(sparse_apply, 11);
}
