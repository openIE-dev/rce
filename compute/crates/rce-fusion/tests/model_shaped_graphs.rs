//! Integration tests that run `fuse_with_codegen` on graphs shaped like
//! real model blocks, verifying what the codegen actually does on
//! realistic topologies rather than the minimal `silu → rms_norm`
//! fixtures used by the unit tests.
//!
//! These tests are **diagnostic** as well as regression-guard: they
//! document, in code, how today's transformer/SSM block structures
//! interact with the codegen's Free-chain detector. When the answer is
//! "zero fusions" — as it is for Jamba-style FFN blocks today — the
//! test surfaces the limitation in writing rather than letting it stay
//! a quiet absence.

use rce_fusion::{fuse_with_codegen, FusionRegistry};
use rce_graph::graph::{FunctionGraph, GraphNode, OpKind, Param, Wire, WireKind};

// ── Shared graph-building utilities ─────────────────────────────────

fn param(name: &str, dtype: &str) -> GraphNode {
    GraphNode::Param(Param { name: name.into(), data_type: dtype.into() })
}
fn apply(op: &str) -> GraphNode {
    GraphNode::Apply { op: OpKind::Custom(op.into()), output_type: "f32".into() }
}
fn w(from: usize, to: usize, port: i32) -> Wire {
    Wire { from, to, port_in_idx: port, wire_kind: WireKind::Move }
}

fn node_op_names(g: &FunctionGraph) -> Vec<String> {
    g.nodes.iter().filter_map(|n| match n {
        GraphNode::Apply { op: OpKind::Custom(name), .. } => Some(name.clone()),
        _ => None,
    }).collect()
}

// ── Realistic shapes ────────────────────────────────────────────────

/// Mimics a Jamba-style FFN block's topology *without* depending on
/// the Jamba crate (it's still in active editing). The Free-fraction
/// surface here is exactly: `rms_norm` once, then three `matmul`s and
/// a `swiglu` — none of which form a length-≥2 Free chain *with
/// fragments* (swiglu has no fragment today; matmul is Free per CALM
/// but is bandwidth-Bound for the fusion-codegen target).
///
/// Documents the honest finding: today's FFN block yields zero
/// codegen fusions. The bottleneck is fragment-table coverage, not
/// the rewriter.
fn jamba_style_ffn_block() -> FunctionGraph {
    FunctionGraph {
        name: "ffn_block".into(),
        nodes: vec![
            param("x", "f32"),
            param("norm_w", "f32"),
            param("ngroups", "i32"),
            param("eps", "f32"),
            param("gate_w", "f32"),
            param("up_w", "f32"),
            param("down_w", "f32"),
            apply("rms_norm"),  // 7
            apply("matmul"),    // 8 — gate proj
            apply("matmul"),    // 9 — up proj
            apply("swiglu"),    // 10
            apply("matmul"),    // 11 — down proj
        ],
        wires: vec![
            // rms_norm(x, norm_w, ngroups, eps)
            w(0, 7, 0), w(1, 7, 1), w(2, 7, 2), w(3, 7, 3),
            // gate = matmul(rms_out, gate_w); up = matmul(rms_out, up_w)
            w(7, 8, 0), w(4, 8, 1),
            w(7, 9, 0), w(5, 9, 1),
            // swiglu(gate, up)
            w(8, 10, 0), w(9, 10, 1),
            // down = matmul(swiglu_out, down_w)
            w(10, 11, 0), w(6, 11, 1),
        ],
        output_node: 11,
        return_type: "f32".into(),
    }
}

/// A hypothetical "norm-stacked" sub-block — the kind of pattern that
/// shows up in some recent architectures (and in graph forms that have
/// been pre-coalesced to remove intervening reshapes / split_ats). Two
/// Free chains of length 2 + 3, each with exactly one reduction:
///
///   rms_norm → silu     (length 2, one reduction)
///   silu → rms_norm → gelu  (length 3, one reduction, fully Phase C-covered)
///
/// Plus a non-Free `matmul` separating them. The pass should produce
/// exactly two fused nodes and leave the matmul untouched.
fn norm_stacked_sub_block() -> FunctionGraph {
    FunctionGraph {
        name: "norm_stacked".into(),
        nodes: vec![
            param("x", "f32"),         // 0
            param("w1", "f32"),        // 1
            param("ng1", "i32"),       // 2
            param("e1", "f32"),        // 3
            param("proj_w", "f32"),    // 4
            param("w2", "f32"),        // 5
            param("ng2", "i32"),       // 6
            param("e2", "f32"),        // 7
            apply("rms_norm"),         // 8 — chain A head
            apply("silu"),             // 9 — chain A tail
            apply("matmul"),           // 10 — separator, not in fragments
            apply("silu"),             // 11 — chain B head
            apply("rms_norm"),         // 12
            apply("gelu"),             // 13 — chain B tail
        ],
        wires: vec![
            // Chain A: rms_norm(x, w1, ng1, e1) → silu
            w(0, 8, 0), w(1, 8, 1), w(2, 8, 2), w(3, 8, 3),
            w(8, 9, 0),
            // matmul(silu_out, proj_w)
            w(9, 10, 0), w(4, 10, 1),
            // Chain B: silu → rms_norm(., w2, ng2, e2) → gelu
            w(10, 11, 0),
            w(11, 12, 0), w(5, 12, 1), w(6, 12, 2), w(7, 12, 3),
            w(12, 13, 0),
        ],
        output_node: 13,
        return_type: "f32".into(),
    }
}

/// Two adjacent reductions: `silu → rms_norm → gelu → rms_norm → silu`.
/// Phase C rejects chains with > 1 reduction at any single parent, but
/// the multi-pass loop should converge to a valid fused state by
/// fusing one sub-chain at a time. Documents the algorithmic behaviour.
fn double_reduction_chain() -> FunctionGraph {
    FunctionGraph {
        name: "double_red".into(),
        nodes: vec![
            param("x", "f32"),         // 0
            param("w1", "f32"),        // 1
            param("ng1", "i32"),       // 2
            param("e1", "f32"),        // 3
            param("w2", "f32"),        // 4
            param("ng2", "i32"),       // 5
            param("e2", "f32"),        // 6
            apply("silu"),             // 7
            apply("rms_norm"),         // 8
            apply("gelu"),             // 9
            apply("rms_norm"),         // 10
            apply("silu"),             // 11
        ],
        wires: vec![
            // silu(x)
            w(0, 7, 0),
            // rms_norm(silu_out, w1, ng1, e1)
            w(7, 8, 0), w(1, 8, 1), w(2, 8, 2), w(3, 8, 3),
            // gelu(.)
            w(8, 9, 0),
            // rms_norm(., w2, ng2, e2)
            w(9, 10, 0), w(4, 10, 1), w(5, 10, 2), w(6, 10, 3),
            // silu(.)
            w(10, 11, 0),
        ],
        output_node: 11,
        return_type: "f32".into(),
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[test]
fn jamba_style_ffn_block_fuses_matmul_swiglu_via_phase_g() {
    // Renamed from `…_yields_zero_codegen_fusions_today`: Phase G's
    // matmul-epilogue walker now picks up the `matmul(gate) → swiglu`
    // shape this block carries. The contract the original test set
    // ("when coverage expands, the assertion needs an explicit update
    // reflecting the new behaviour") — this is that update.
    //
    // What still doesn't fuse here:
    //   - `rms_norm` alone (length-1 chain).
    //   - `matmul(up) → swiglu`: matmul(up)'s output feeds swiglu's
    //     port 1 (the `up` slot), not port 0; the single-arg-0-consumer
    //     constraint correctly refuses to fuse across non-port-0
    //     connections.
    //   - `matmul(down)` alone — the FFN's last matmul has no
    //     downstream Free fragment in this fixture.
    //
    // Net: exactly one fusion fires: matmul(gate) + swiglu.
    let g = jamba_style_ffn_block();
    let (out, apps) = fuse_with_codegen(g, &FusionRegistry::default());
    assert_eq!(apps.len(), 1,
        "Phase G fuses matmul(gate)→swiglu; got {:?}", apps);
    assert_eq!(apps[0].fused_op, "fused_matmul_swiglu_f32");
    // The fused node absorbs matmul(gate) and swiglu, dropping one node.
    let remaining_ops = node_op_names(&out);
    assert!(remaining_ops.iter().any(|n| n == "fused_matmul_swiglu_f32"));
    assert!(remaining_ops.iter().any(|n| n == "rms_norm"));
    // The second matmul (up) survives — its output feeds swiglu's
    // port 1 (now absorbed into the fused node's args).
    let matmul_count = remaining_ops.iter().filter(|n| *n == "matmul").count();
    assert_eq!(matmul_count, 2, "matmul(up) and matmul(down) both survive; got {remaining_ops:?}");
}

#[test]
fn norm_stacked_block_fuses_both_chains_and_matmul_with_silu() {
    // Renamed from `…_fuses_both_chains_independently`. Phase G changes
    // this graph's fusion behaviour: the `matmul → silu` connection
    // that was previously a chain separator is now itself an epilogue
    // fusion target. The matmul steals silu from the longer chain,
    // leaving the rest as a `rms_norm → gelu` tail.
    //
    // Original chain B was `silu → rms_norm → gelu` (one big fusion).
    // After Phase G:
    //   - chain A (`rms_norm → silu`) still fuses as before.
    //   - `matmul → silu` fuses to `fused_matmul_silu_f32`.
    //   - what's left of chain B (`rms_norm → gelu`) is its own fusion.
    let g = norm_stacked_sub_block();
    let (_, apps) = fuse_with_codegen(g, &FusionRegistry::default());
    assert_eq!(apps.len(), 3,
        "Phase G produces three fusions (chain A, matmul-epilogue, chain B tail); got {:?}", apps);
    let names: Vec<&str> = apps.iter().map(|a| a.fused_op.as_str()).collect();
    assert!(names.contains(&"fused_rms_norm_silu_f32"), "chain A: {names:?}");
    assert!(names.contains(&"fused_matmul_silu_f32"), "matmul epilogue: {names:?}");
    assert!(names.contains(&"fused_rms_norm_gelu_f32"), "chain B tail: {names:?}");
}

/// SSM-block-style fragment: `silu → reshape → rms_norm`. Mirrors the
/// shape the Jamba SSM block emits at one of its norm sites (silu's
/// output goes through a reshape that adjusts (batch, d_inner) to
/// (batch, ngroups, group_size) before rms_norm groups its reduction).
/// Before Phase E.b's coalesce pre-pass this stayed unfused — the
/// reshape broke the chain. After Phase E.b, the pre-pass drops the
/// reshape and the codegen fuses silu → rms_norm.
fn ssm_style_silu_reshape_rms_norm() -> FunctionGraph {
    FunctionGraph {
        name: "ssm_norm_site".into(),
        nodes: vec![
            param("x", "f32"),         // 0
            param("weight", "f32"),    // 1
            param("ngroups", "i32"),   // 2
            param("eps", "f32"),       // 3
            apply("silu"),             // 4
            apply("reshape"),          // 5
            apply("rms_norm"),         // 6
        ],
        wires: vec![
            w(0, 4, 0),
            w(4, 5, 0),
            w(5, 6, 0), w(1, 6, 1), w(2, 6, 2), w(3, 6, 3),
        ],
        output_node: 6,
        return_type: "f32".into(),
    }
}

#[test]
fn ssm_style_silu_reshape_rms_norm_fuses_via_coalesce_pre_pass() {
    // End-to-end demonstration of Phase E.b: a chain that was
    // un-fusable before Phase E (because reshape sat between the two
    // Free fragments) now lands a single fusion thanks to
    // `coalesce_transparent` running ahead of the codegen walker.
    let g = ssm_style_silu_reshape_rms_norm();
    let (out, apps) = fuse_with_codegen(g, &FusionRegistry::default());
    assert_eq!(apps.len(), 1,
        "coalesce + fuse should yield one fusion; got {:?}", apps);
    assert_eq!(apps[0].fused_op, "fused_silu_rms_norm_f32");
    // Original 7 nodes minus reshape (coalesced) minus 1 (silu+rms_norm
    // collapse to 1 node) = 5.
    assert_eq!(out.nodes.len(), 5,
        "expected 5 nodes after coalesce + fuse, got {}", out.nodes.len());
    // No reshape survives.
    let ops = node_op_names(&out);
    assert!(!ops.contains(&"reshape".to_string()),
        "reshape should have been coalesced; remaining ops: {ops:?}");
}

#[test]
fn double_reduction_chain_converges_via_multi_pass_loop() {
    // The greedy walker rejects a length-5 chain with 2 reductions
    // outright at the chain head. But it then tries each subsequent
    // parent, and the multi-pass loop converges. End state: two fused
    // nodes covering different segments of the original chain. The
    // exact partition depends on traversal order; the contract is
    // (a) ≥ 1 fusion fires, (b) every reduction in the original
    // ends up inside exactly one fused node, (c) idempotence under a
    // second pass.
    let g = double_reduction_chain();
    let original_nodes = g.nodes.len();
    let (out, apps) = fuse_with_codegen(g, &FusionRegistry::default());

    assert!(!apps.is_empty(),
        "greedy + multi-pass should fuse at least one sub-chain; got 0");

    // Every fused op name must parse via the codegen module.
    for a in &apps {
        let chain = rce_fusion::codegen::parse_function_name(&a.fused_op)
            .unwrap_or_else(|| panic!("apps yielded unparseable name {}", a.fused_op));
        // Per Phase C scope, each fused chain must contain exactly one
        // reduction-bearing op.
        let n_red = chain.iter().filter(|n| {
            rce_fusion::codegen::fragments::lookup(n).map(|f| f.has_reduction()).unwrap_or(false)
        }).count();
        assert_eq!(n_red, 1,
            "fused chain {} should carry exactly one reduction, got {}",
            a.fused_op, n_red);
    }

    // Total reductions absorbed across all fusions == 2 (the original count).
    let total_red_absorbed: usize = apps.iter().filter_map(|a| {
        rce_fusion::codegen::parse_function_name(&a.fused_op)
    }).map(|chain| {
        chain.iter().filter(|n| {
            rce_fusion::codegen::fragments::lookup(n).map(|f| f.has_reduction()).unwrap_or(false)
        }).count()
    }).sum();
    assert_eq!(total_red_absorbed, 2,
        "both original reductions should land inside some fused node");

    // Node-count strictly drops.
    assert!(out.nodes.len() < original_nodes,
        "fusion must drop node count; {} → {}", original_nodes, out.nodes.len());

    // Idempotence: a second pass is a no-op.
    let (g2, apps2) = fuse_with_codegen(out, &FusionRegistry::default());
    assert!(apps2.is_empty(),
        "second pass should find no new fusions; got {:?}", apps2);
    // (Node count stable across the no-op.)
    let _ = g2;
}
