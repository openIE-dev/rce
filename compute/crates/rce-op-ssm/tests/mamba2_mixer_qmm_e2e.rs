//! **The production mamba2 mixer, as a persistent `.fg` `FunctionGraph`,
//! executed through the rce VM — bit-identical to the production
//! op-functions called in sequence.**
//!
//! This is step 1 of the speed-via-substrate plan: prove the real
//! runtime's mixer (qmm in/out_proj, conv1d, whole-band silu,
//! f64 selective scan, gate-then-group-RMS) can be expressed as one
//! IR graph the VM runs — the precondition for graph-level fusion /
//! prefill batching / layout-per-target, the things that beat
//! op-by-op engines.
//!
//! Reference = the exact production functions in production order
//! (`rce_op_qmm::mm_1d`, `conv1d_causal_step`,
//! `selective_scan_step_f64`, group-RMS, silu). The graph routes the
//! same dataflow through `interpret_with_config` with the leaves the
//! production mixer would use, and must match every output element
//! **byte-for-byte** (same kernels under the leaves; only the dispatch
//! path differs).
//!
//! Graph also threads state: a `FunctionGraph` has one `output_node`,
//! so the per-token mixer returns a packed
//! `[out ‖ new_conv_state ‖ new_ssm_state]` via `ConcatLeaf`; the
//! driver (here, the test) splits it back.

use rce_op_conv1d::{conv1d_causal_step, Conv1dCausalLeaf};
use rce_op_nn_prims::{ConcatLeaf, RmsNormLeaf, SiluLeaf, SplitAtLeaf};
use rce_op_qmm::{QmmLeaf, QuantWeight};
use rce_op_ssm::{selective_scan_step_f64, SsmScanF64Leaf};
use rce_vm_backend_cpu::{
    interpret_with_config, HostTensor, Inputs, InterpretConfig, OpDispatch, Value,
};
use rce_ingest_gguf::GgufDtype;
use rce_graph::graph::{FunctionGraph, GraphNode, LiteralValue, OpKind, Param, Wire, WireKind};

// ── Dims (small, but structurally real; all matmul-K %32 for Q4_0) ──
const D_MODEL: usize = 32;
const NH: usize = 8;
const HD: usize = 8;
const D_INNER: usize = NH * HD; // 64
const DS: usize = 16;
const NG: usize = 2;
const D_CONV: usize = 4;
const CONV_DIM: usize = D_INNER + 2 * NG * DS; // 64 + 64 = 128
const IN_PROJ_OUT: usize = D_INNER + CONV_DIM + NH; // 64+128+8 = 200
const EPS: f32 = 1e-5;

fn p(name: &str) -> GraphNode {
    GraphNode::Param(Param { name: name.into(), data_type: "f32".into() })
}
fn lit_i32(v: i32) -> GraphNode {
    GraphNode::Literal { value: LiteralValue::I32(v), data_type: "i32".into() }
}
fn lit_f32(v: f32) -> GraphNode {
    GraphNode::Literal { value: LiteralValue::F32(v), data_type: "f32".into() }
}
fn ac(name: &str) -> GraphNode {
    GraphNode::Apply { op: OpKind::Custom(name.into()), output_type: "f32".into() }
}
fn mul() -> GraphNode {
    GraphNode::Apply { op: OpKind::Mul, output_type: "f32".into() }
}

/// Build the mixer graph once. Returns `(graph, qmm_in_node,
/// qmm_out_node)` so the driver can register per-block weights.
fn build_qmm_mixer_graph() -> (FunctionGraph, usize, usize) {
    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut push = |g: GraphNode, v: &mut Vec<GraphNode>| -> usize {
        v.push(g);
        v.len() - 1
    };

    // Params (graph inputs). ssm_in/ssm_out are NOT params — QmmLeaf
    // registry by node_id keeps them zero-copy.
    let n_x_in = push(p("x_in"), &mut nodes);
    let n_conv_w = push(p("conv_w"), &mut nodes);
    let n_conv_b = push(p("conv_b"), &mut nodes);
    let n_dt_bias = push(p("dt_bias"), &mut nodes);
    let n_a_log = push(p("a_log"), &mut nodes);
    let n_d_param = push(p("d_param"), &mut nodes);
    let n_ssm_norm_w = push(p("ssm_norm_w"), &mut nodes);
    let n_conv_state = push(p("conv_state"), &mut nodes);
    let n_ssm_state = push(p("ssm_state"), &mut nodes);

    // Literals.
    let l_0 = push(lit_i32(0), &mut nodes);
    let l_d_inner = push(lit_i32(D_INNER as i32), &mut nodes);
    let l_conv_dim = push(lit_i32(CONV_DIM as i32), &mut nodes);
    let l_dt_off = push(lit_i32((D_INNER + CONV_DIM) as i32), &mut nodes);
    let l_nh = push(lit_i32(NH as i32), &mut nodes);
    let l_ngds = push(lit_i32((NG * DS) as i32), &mut nodes);
    let l_d_inner_plus_ngds = push(lit_i32((D_INNER + NG * DS) as i32), &mut nodes);
    let l_conv_state_len = push(lit_i32((CONV_DIM * D_CONV) as i32), &mut nodes);
    let l_ssm_state_len = push(lit_i32((D_INNER * DS) as i32), &mut nodes);
    let l_d_conv = push(lit_i32(D_CONV as i32), &mut nodes);
    let l_hd = push(lit_i32(HD as i32), &mut nodes);
    let l_ds = push(lit_i32(DS as i32), &mut nodes);
    let l_ng = push(lit_i32(NG as i32), &mut nodes);
    let l_eps = push(lit_f32(EPS), &mut nodes);

    // Apply nodes — production order.
    let n_proj = push(ac("qmm"), &mut nodes); // qmm(x_in) ; weight=ssm_in
    let n_z = push(ac("split_at"), &mut nodes); // proj[0..d_inner]
    let n_xbc = push(ac("split_at"), &mut nodes); // proj[d_inner..+conv_dim]
    let n_dt_raw = push(ac("split_at"), &mut nodes); // proj[dt_off..+nh]
    let n_pconv = push(ac("conv1d_causal_depthwise"), &mut nodes);
    let n_xbc_conv = push(ac("split_at"), &mut nodes); // pconv[0..conv_dim]
    let n_new_conv = push(ac("split_at"), &mut nodes); // pconv[conv_dim..+conv_dim*d_conv]
    let n_xbc_act = push(ac("silu"), &mut nodes); // silu(whole conv band)
    let n_x_act = push(ac("split_at"), &mut nodes); // xbc_act[0..d_inner]
    let n_b_act = push(ac("split_at"), &mut nodes); // xbc_act[d_inner..+ng*ds]
    let n_c_act = push(ac("split_at"), &mut nodes); // xbc_act[d_inner+ng*ds..+ng*ds]
    let n_pssm = push(ac("ssm_scan_f64"), &mut nodes);
    let n_y = push(ac("split_at"), &mut nodes); // pssm[0..d_inner]
    let n_new_ssm = push(ac("split_at"), &mut nodes); // pssm[d_inner..+d_inner*ds]
    let n_silu_z = push(ac("silu"), &mut nodes); // silu(z)
    let n_gated = push(mul(), &mut nodes); // silu(z) * y
    let n_normed = push(ac("rms_norm"), &mut nodes); // group rms, ng groups
    let n_out = push(ac("qmm"), &mut nodes); // qmm(normed) ; weight=ssm_out
    let n_result = push(ac("concat"), &mut nodes); // [out ‖ new_conv ‖ new_ssm]

    // Wires: (from, to, port_in_idx).
    let w: Vec<(usize, usize, usize)> = vec![
        (n_x_in, n_proj, 0),
        // splits of proj
        (n_proj, n_z, 0), (l_0, n_z, 1), (l_d_inner, n_z, 2),
        (n_proj, n_xbc, 0), (l_d_inner, n_xbc, 1), (l_conv_dim, n_xbc, 2),
        (n_proj, n_dt_raw, 0), (l_dt_off, n_dt_raw, 1), (l_nh, n_dt_raw, 2),
        // conv1d(xbc, conv_w, conv_b, conv_state, d_conv)
        (n_xbc, n_pconv, 0), (n_conv_w, n_pconv, 1), (n_conv_b, n_pconv, 2),
        (n_conv_state, n_pconv, 3), (l_d_conv, n_pconv, 4),
        // unpack packed_conv
        (n_pconv, n_xbc_conv, 0), (l_0, n_xbc_conv, 1), (l_conv_dim, n_xbc_conv, 2),
        (n_pconv, n_new_conv, 0), (l_conv_dim, n_new_conv, 1), (l_conv_state_len, n_new_conv, 2),
        // silu whole band, then split x/b/c
        (n_xbc_conv, n_xbc_act, 0),
        (n_xbc_act, n_x_act, 0), (l_0, n_x_act, 1), (l_d_inner, n_x_act, 2),
        (n_xbc_act, n_b_act, 0), (l_d_inner, n_b_act, 1), (l_ngds, n_b_act, 2),
        (n_xbc_act, n_c_act, 0), (l_d_inner_plus_ngds, n_c_act, 1), (l_ngds, n_c_act, 2),
        // ssm_scan_f64(x,B,C,dt_raw,dt_bias,A_log,D,state,headdim,d_state)
        (n_x_act, n_pssm, 0), (n_b_act, n_pssm, 1), (n_c_act, n_pssm, 2),
        (n_dt_raw, n_pssm, 3), (n_dt_bias, n_pssm, 4), (n_a_log, n_pssm, 5),
        (n_d_param, n_pssm, 6), (n_ssm_state, n_pssm, 7),
        (l_hd, n_pssm, 8), (l_ds, n_pssm, 9),
        // unpack packed_ssm
        (n_pssm, n_y, 0), (l_0, n_y, 1), (l_d_inner, n_y, 2),
        (n_pssm, n_new_ssm, 0), (l_d_inner, n_new_ssm, 1), (l_ssm_state_len, n_new_ssm, 2),
        // gate: silu(z) * y
        (n_z, n_silu_z, 0),
        (n_silu_z, n_gated, 0), (n_y, n_gated, 1),
        // group rms_norm(gated, ssm_norm_w, ng, eps)
        (n_gated, n_normed, 0), (n_ssm_norm_w, n_normed, 1),
        (l_ng, n_normed, 2), (l_eps, n_normed, 3),
        // out_proj
        (n_normed, n_out, 0),
        // pack [out ‖ new_conv_state ‖ new_ssm_state]
        (n_out, n_result, 0), (n_new_conv, n_result, 1), (n_new_ssm, n_result, 2),
    ];
    let wires = w
        .into_iter()
        .map(|(from, to, port_in_idx)| Wire {
            from,
            to,
            port_in_idx: port_in_idx as i32,
            wire_kind: WireKind::Move,
        })
        .collect();

    (
        FunctionGraph {
            name: "mamba2_mixer_qmm".into(),
            nodes,
            wires,
            output_node: n_result,
            return_type: "f32".into(),
        },
        n_proj,
        n_out,
    )
}

// ── Synthetic Q4_0 weight bytes (reuse rce-op-qmm's test pattern) ──
fn q4_0_bytes(n_rows: usize, k: usize, salt: usize) -> Vec<u8> {
    assert!(k % 32 == 0);
    let bpr = (k / 32) * 18;
    let mut b = vec![0u8; n_rows * bpr];
    for (i, x) in b.iter_mut().enumerate() {
        *x = ((i * 13 + salt) & 0xFF) as u8;
    }
    for r in 0..n_rows {
        for blk in 0..(k / 32) {
            let base = r * bpr + blk * 18;
            let d = half::f16::from_f32(0.0625).to_le_bytes();
            b[base..base + 2].copy_from_slice(&d);
        }
    }
    b
}

#[test]
fn production_mamba2_mixer_graph_is_bit_exact() {
    // Synthetic block inputs/weights.
    let x_in: Vec<f32> = (0..D_MODEL).map(|i| ((i as f32) * 0.07 - 1.0).sin()).collect();
    let conv_w: Vec<f32> = (0..CONV_DIM * D_CONV).map(|i| (i as f32 * 0.001 - 0.2).cos()).collect();
    let conv_b: Vec<f32> = (0..CONV_DIM).map(|i| i as f32 * 0.0003).collect();
    let dt_bias: Vec<f32> = (0..NH).map(|i| i as f32 * 0.02 + 0.1).collect();
    // ssm_a stores A directly (already -exp(A_log)); negative.
    let a_log: Vec<f32> = (0..NH).map(|i| -((i + 1) as f32) * 0.3).collect();
    let d_param: Vec<f32> = (0..NH).map(|i| i as f32 * 0.005).collect();
    let ssm_norm_w: Vec<f32> = (0..D_INNER).map(|i| 1.0 + i as f32 * 0.001).collect();
    let conv_state0 = vec![0.0f32; CONV_DIM * D_CONV];
    let ssm_state0 = vec![0.0f32; D_INNER * DS];

    let ssm_in = QuantWeight::from_owned_bytes(
        q4_0_bytes(IN_PROJ_OUT, D_MODEL, 1), GgufDtype::Q4_0, IN_PROJ_OUT, D_MODEL,
    );
    let ssm_out = QuantWeight::from_owned_bytes(
        q4_0_bytes(D_MODEL, D_INNER, 7), GgufDtype::Q4_0, D_MODEL, D_INNER,
    );

    // ── Reference: production functions in production order ──
    let mut proj = vec![0.0f32; IN_PROJ_OUT];
    rce_op_qmm::mm_1d(&x_in, &ssm_in, &mut proj);
    let z = &proj[0..D_INNER];
    let xbc = &proj[D_INNER..D_INNER + CONV_DIM];
    let dt_raw = &proj[D_INNER + CONV_DIM..D_INNER + CONV_DIM + NH];
    let (xbc_conv, new_conv_ref) =
        conv1d_causal_step(xbc, &conv_w, &conv_b, &conv_state0, CONV_DIM, D_CONV);
    let silu = |v: f32| if v >= 0.0 { v / (1.0 + (-v).exp()) } else { let e = v.exp(); v * e / (1.0 + e) };
    let xbc_act: Vec<f32> = xbc_conv.iter().map(|v| silu(*v)).collect();
    let x_act = &xbc_act[0..D_INNER];
    let b_act = &xbc_act[D_INNER..D_INNER + NG * DS];
    let c_act = &xbc_act[D_INNER + NG * DS..CONV_DIM];
    let (y, new_ssm_ref) = selective_scan_step_f64(
        x_act, b_act, c_act, dt_raw, &dt_bias, &a_log, &d_param, &ssm_state0,
        NH, HD, DS, NG,
    );
    let mut gated = vec![0.0f32; D_INNER];
    for i in 0..D_INNER {
        gated[i] = silu(z[i]) * y[i];
    }
    // group rms (ng groups), bit-identical to RmsNormLeaf.
    let gsz = D_INNER / NG;
    let mut normed = vec![0.0f32; D_INNER];
    for g in 0..NG {
        let off = g * gsz;
        let ms: f32 = gated[off..off + gsz].iter().map(|v| v * v).sum::<f32>() / gsz as f32;
        let inv = 1.0 / (ms + EPS).sqrt();
        for i in 0..gsz {
            normed[off + i] = gated[off + i] * inv * ssm_norm_w[off + i];
        }
    }
    let mut out_ref = vec![0.0f32; D_MODEL];
    rce_op_qmm::mm_1d(&normed, &ssm_out, &mut out_ref);

    // ── Graph through the VM ──
    let (graph, qmm_in_node, qmm_out_node) = build_qmm_mixer_graph();
    let mut qmm_leaf = QmmLeaf::new();
    qmm_leaf.register(qmm_in_node, ssm_in.clone());
    qmm_leaf.register(qmm_out_node, ssm_out.clone());
    let conv_leaf = Conv1dCausalLeaf::new();
    let silu_leaf = SiluLeaf::new();
    let split_leaf = SplitAtLeaf::new();
    let rms_leaf = RmsNormLeaf::new();
    let scan_leaf = SsmScanF64Leaf::new();
    let concat_leaf = ConcatLeaf::new();
    let ops: Vec<&dyn OpDispatch> = vec![
        &qmm_leaf, &conv_leaf, &silu_leaf, &split_leaf, &rms_leaf, &scan_leaf, &concat_leaf,
    ];

    let t = |v: &[f32]| Value::Tensor(HostTensor::from_f32(vec![v.len()], v).unwrap());
    let mut inputs: Inputs = Inputs::new();
    inputs.insert("x_in".into(), t(&x_in));
    inputs.insert("conv_w".into(), t(&conv_w));
    inputs.insert("conv_b".into(), t(&conv_b));
    inputs.insert("dt_bias".into(), t(&dt_bias));
    inputs.insert("a_log".into(), t(&a_log));
    inputs.insert("d_param".into(), t(&d_param));
    inputs.insert("ssm_norm_w".into(), t(&ssm_norm_w));
    inputs.insert("conv_state".into(), t(&conv_state0));
    inputs.insert("ssm_state".into(), t(&ssm_state0));

    let packed = match interpret_with_config(
        &graph,
        &inputs,
        InterpretConfig { leaves: &[], ops: &ops },
    )
    .expect("interpret mixer graph")
    {
        Value::Tensor(t) => t.as_f32().expect("f32 out"),
        other => panic!("non-tensor result {other:?}"),
    };

    // Unpack [out ‖ new_conv_state ‖ new_ssm_state].
    assert_eq!(packed.len(), D_MODEL + CONV_DIM * D_CONV + D_INNER * DS);
    let out_g = &packed[0..D_MODEL];
    let new_conv_g = &packed[D_MODEL..D_MODEL + CONV_DIM * D_CONV];
    let new_ssm_g = &packed[D_MODEL + CONV_DIM * D_CONV..];

    for (i, (a, b)) in out_g.iter().zip(out_ref.iter()).enumerate() {
        assert_eq!(a, b, "out[{i}]: graph {a} ≠ production {b} (must be bit-identical)");
    }
    for (i, (a, b)) in new_conv_g.iter().zip(new_conv_ref.iter()).enumerate() {
        assert_eq!(a, b, "new_conv_state[{i}]: graph {a} ≠ production {b}");
    }
    for (i, (a, b)) in new_ssm_g.iter().zip(new_ssm_ref.iter()).enumerate() {
        assert_eq!(a, b, "new_ssm_state[{i}]: graph {a} ≠ production {b}");
    }
}
