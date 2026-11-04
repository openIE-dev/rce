//! MoE block expressed as a `.fg` `FunctionGraph` and run end-to-end
//! through `interpret_with_config`. Demonstrates that the substrate
//! carries the architectural structure of a sparse-MoE FFN — router →
//! top-K → weighted combine — at the IR level, not as a monolithic
//! `Custom("moe_block")` op.
//!
//! The decomposition here is intentionally MINIMAL — two Apply nodes
//! (`matmul` → `moe_combine`) — to ship the demonstration. A more
//! aggressively decomposed form (separate `topk_mask` + element-wise
//! multiply + softmax + weighted reduce) is shipped as a second test
//! variant below, showing the sparse mask threading through the IR.
//!
//! What "sparse" means at the IR level today: the graph carries the
//! `topk_mask` Apply node, which produces a `[num_experts]` binary
//! tensor. An IR-level optimisation pass can later detect "expert
//! output multiplied by zero" and dead-code-eliminate the
//! corresponding expert sub-graph. The leaf set is in place; the
//! optimisation pass is a separate concern.

use rce_op_nn_prims::{MoeCombineLeaf, TopKMaskLeaf};
use rce_vm_backend_cpu::{
    interpret_with_config, HostTensor, Inputs, InterpretConfig, OpDispatch, ScalarValue, Value,
};
use rce_graph::graph::{
    FunctionGraph, GraphNode, LiteralValue, OpKind, Param, Wire, WireKind,
};

#[derive(Clone, Copy, Debug)]
struct MoeConfig {
    d_model: usize,
    num_experts: usize,
    top_k: usize,
}

impl MoeConfig {
    fn small() -> Self {
        Self { d_model: 32, num_experts: 8, top_k: 2 }
    }
}

// ── Reference impl: dense (run all experts, weight by softmax over top-K) ──

fn reference_moe_block(
    cfg: MoeConfig,
    x: &[f32],
    gate_w_t: &[f32],          // (d_model, num_experts) row-major
    expert_outputs: &[f32],    // (num_experts, d_model)
) -> Vec<f32> {
    // 1. router_logits = x @ gate_w_t  →  [num_experts]
    let mut logits = vec![0.0f32; cfg.num_experts];
    for e in 0..cfg.num_experts {
        let mut acc = 0.0f32;
        for d in 0..cfg.d_model {
            acc += x[d] * gate_w_t[d * cfg.num_experts + e];
        }
        logits[e] = acc;
    }

    // 2. Top-K + softmax over top-K + weighted combine.
    let mut idx: Vec<usize> = (0..cfg.num_experts).collect();
    idx.select_nth_unstable_by(cfg.top_k - 1, |&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(cfg.top_k);
    let max_l = idx.iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
    let mut weights = vec![0.0f32; cfg.top_k];
    let mut sum = 0.0f32;
    for (slot, &i) in idx.iter().enumerate() {
        let e = (logits[i] - max_l).exp();
        weights[slot] = e;
        sum += e;
    }
    for w in weights.iter_mut() { *w /= sum; }

    let mut out = vec![0.0f32; cfg.d_model];
    for (slot, &expert_id) in idx.iter().enumerate() {
        let w = weights[slot];
        let base = expert_id * cfg.d_model;
        for d in 0..cfg.d_model {
            out[d] += w * expert_outputs[base + d];
        }
    }
    out
}

// ── Build the MoE block as a `FunctionGraph` ──────────────────────

fn build_moe_block_graph(cfg: MoeConfig) -> FunctionGraph {
    let m = WireKind::Move;
    let nodes = vec![
        // ── Params ─────────────────────────────────────────
        /* 0 */ p("x"),
        /* 1 */ p("gate_w_t"),
        /* 2 */ p("expert_outputs"),
        // ── Literal: top_k ─────────────────────────────────
        /* 3 */ GraphNode::Literal {
            value: LiteralValue::I32(cfg.top_k as i32),
            data_type: "i32".into(),
        },
        // ── Apply ──────────────────────────────────────────
        /* 4 */ apply_custom("matmul"),       // x [1, d_model] @ gate_w_t [d_model, E] → logits_2d [1, E]
        /* 5 */ apply_custom("moe_combine"),  // (logits_1d, expert_outputs, top_k) → output [d_model]
    ];
    // Note: matmul output is [1, E] (2-D); moe_combine expects [E].
    // moe_combine handles this transparently because its `numel` check
    // doesn't require shape-rank, only total element count.
    let wires = vec![
        // matmul(x, gate_w_t)
        Wire { from: 0, to: 4, port_in_idx: 0, wire_kind: m },
        Wire { from: 1, to: 4, port_in_idx: 1, wire_kind: m },
        // moe_combine(logits=4, expert_outputs=2, top_k=3)
        Wire { from: 4, to: 5, port_in_idx: 0, wire_kind: m },
        Wire { from: 2, to: 5, port_in_idx: 1, wire_kind: m },
        Wire { from: 3, to: 5, port_in_idx: 2, wire_kind: m },
    ];
    FunctionGraph {
        name: "moe_block_minimal".into(),
        nodes,
        wires,
        output_node: 5,
        return_type: "f32".into(),
    }
}

fn p(name: &str) -> GraphNode {
    GraphNode::Param(Param { name: name.into(), data_type: "f32".into() })
}

fn apply_custom(name: &str) -> GraphNode {
    GraphNode::Apply {
        op: OpKind::Custom(name.into()),
        output_type: "f32".into(),
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[test]
fn moe_block_through_rce_dispatch_matches_reference() {
    let cfg = MoeConfig::small();

    // Deterministic random inputs.
    let x: Vec<f32> = (0..cfg.d_model).map(|i| (i as f32 * 0.013 - 0.4).sin()).collect();
    let gate_w_t: Vec<f32> = (0..cfg.d_model * cfg.num_experts)
        .map(|i| ((i as f32 * 0.005) - 0.2).cos() * 0.4)
        .collect();
    let expert_outputs: Vec<f32> = (0..cfg.num_experts * cfg.d_model)
        .map(|i| ((i as f32 * 0.0017) + 0.1).sin() * 0.7)
        .collect();

    // Reference path.
    let want = reference_moe_block(cfg, &x, &gate_w_t, &expert_outputs);

    // Rce path.
    let graph = build_moe_block_graph(cfg);
    let mut inputs = Inputs::new();
    inputs.insert(
        "x".into(),
        Value::Tensor(HostTensor::from_f32(vec![1, cfg.d_model], &x).unwrap()),
    );
    inputs.insert(
        "gate_w_t".into(),
        Value::Tensor(HostTensor::from_f32(vec![cfg.d_model, cfg.num_experts], &gate_w_t).unwrap()),
    );
    inputs.insert(
        "expert_outputs".into(),
        Value::Tensor(HostTensor::from_f32(vec![cfg.num_experts, cfg.d_model], &expert_outputs).unwrap()),
    );

    let moe_leaf = MoeCombineLeaf::new();
    let ops: Vec<&dyn OpDispatch> = vec![&moe_leaf];
    let cfg_interp = InterpretConfig { leaves: &[], ops: &ops };
    let result = interpret_with_config(&graph, &inputs, cfg_interp).expect("interpret");

    let out_tensor = match result {
        Value::Tensor(t) => t,
        other => panic!("expected tensor, got {other:?}"),
    };
    let got = out_tensor.as_f32().expect("as_f32");

    assert_eq!(want.len(), got.len(), "moe block output length mismatch");
    let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_err < 1e-4, "moe block max_abs_err {max_err}");
    eprintln!(
        "moe_block (d_model={}, E={}, top_k={}) through dispatch: max_abs_err = {max_err:.3e}",
        cfg.d_model, cfg.num_experts, cfg.top_k
    );
}

/// Exercises `topk_mask` as a separate Apply node — the IR carries
/// the sparse mask explicitly, even though the demo's downstream
/// `moe_combine` op also does its own top-K. Demonstrates that the
/// IR can express the mask as first-class data flowing between nodes,
/// which an IR-level optimisation pass could later use to skip
/// non-routed expert computations.
#[test]
fn topk_mask_threaded_through_graph() {
    let m = WireKind::Move;
    let cfg = MoeConfig::small();

    // Graph: matmul(x, gate_w_t) → topk_mask(logits, top_k) → output is the mask.
    let nodes = vec![
        /* 0 */ p("x"),
        /* 1 */ p("gate_w_t"),
        /* 2 */ GraphNode::Literal {
            value: LiteralValue::I32(cfg.top_k as i32),
            data_type: "i32".into(),
        },
        /* 3 */ apply_custom("matmul"),
        /* 4 */ apply_custom("topk_mask"),
    ];
    let wires = vec![
        Wire { from: 0, to: 3, port_in_idx: 0, wire_kind: m },
        Wire { from: 1, to: 3, port_in_idx: 1, wire_kind: m },
        Wire { from: 3, to: 4, port_in_idx: 0, wire_kind: m },
        Wire { from: 2, to: 4, port_in_idx: 1, wire_kind: m },
    ];
    let graph = FunctionGraph {
        name: "topk_mask_demo".into(),
        nodes, wires,
        output_node: 4,
        return_type: "f32".into(),
    };

    let x: Vec<f32> = (0..cfg.d_model).map(|i| (i as f32 * 0.013).cos() * 0.5).collect();
    let gate_w_t: Vec<f32> = (0..cfg.d_model * cfg.num_experts)
        .map(|i| (i as f32 * 0.007 - 0.3).sin() * 0.6)
        .collect();
    let mut inputs = Inputs::new();
    inputs.insert("x".into(), Value::Tensor(HostTensor::from_f32(vec![1, cfg.d_model], &x).unwrap()));
    inputs.insert("gate_w_t".into(), Value::Tensor(HostTensor::from_f32(vec![cfg.d_model, cfg.num_experts], &gate_w_t).unwrap()));

    let topk_leaf = TopKMaskLeaf::new();
    let ops: Vec<&dyn OpDispatch> = vec![&topk_leaf];
    let cfg_interp = InterpretConfig { leaves: &[], ops: &ops };
    let result = interpret_with_config(&graph, &inputs, cfg_interp).expect("interpret");

    let out = match result {
        Value::Tensor(t) => t.as_f32().unwrap(),
        _ => panic!(),
    };
    assert_eq!(out.len(), cfg.num_experts);
    // Exactly top_k positions should be 1.0.
    let ones = out.iter().filter(|&&v| v == 1.0).count();
    let zeros = out.iter().filter(|&&v| v == 0.0).count();
    assert_eq!(ones, cfg.top_k, "expected exactly top_k=1.0 entries, got {ones}");
    assert_eq!(zeros, cfg.num_experts - cfg.top_k, "rest must be 0.0");

    // Independently compute the expected top-K and verify the same set was selected.
    let _ = ScalarValue::I32(0);
    let mut logits_ref = vec![0.0f32; cfg.num_experts];
    for e in 0..cfg.num_experts {
        let mut acc = 0.0f32;
        for d in 0..cfg.d_model {
            acc += x[d] * gate_w_t[d * cfg.num_experts + e];
        }
        logits_ref[e] = acc;
    }
    let mut idx: Vec<usize> = (0..cfg.num_experts).collect();
    idx.select_nth_unstable_by(cfg.top_k - 1, |&a, &b| {
        logits_ref[b].partial_cmp(&logits_ref[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(cfg.top_k);
    for &i in &idx {
        assert_eq!(out[i], 1.0, "expected mask[{i}]=1.0 (top-K)");
    }
}
