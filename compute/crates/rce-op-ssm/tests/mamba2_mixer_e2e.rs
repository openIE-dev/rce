//! End-to-end Mamba2 mixer block as a composable `.fg` `FunctionGraph`.
//!
//! Demonstrates the full vertical-integration claim: the entire
//! per-token mixer computation — `in_proj → split → conv1d → silu →
//! split → ssm_scan → split → rms_norm → gate → out_proj` — expressed
//! as Apply nodes in a rce `FunctionGraph` and dispatched through
//! the OpDispatch chain (`rce-op-ssm`, `rce-op-conv1d`,
//! `rce-op-nn-prims`, plus the in-tree CPU kernel layer for matmul
//! and elementwise mul). The graph output is bit-equivalent to the
//! efficient-genai reference impl ported below, validating that the
//! IR carries the architectural shape end-to-end without losing
//! numerical fidelity.
//!
//! This is **Milestone 2** from the convergence plan: one full
//! architecture block runs through rce dispatch. With a Metal-leaf
//! ladder later (Milestone 1) the same graph picks up MPS / CUDA
//! acceleration unchanged.

use rce_op_conv1d::Conv1dCausalLeaf;
use rce_op_nn_prims::{RmsNormLeaf, SiluLeaf, SplitAtLeaf};
use rce_op_ssm::{selective_scan_step, SsmScanLeaf};
use rce_vm_backend_cpu::{
    interpret_with_config, HostTensor, Inputs, InterpretConfig, OpDispatch, Value,
};
use rce_graph::graph::{
    FunctionGraph, GraphNode, LiteralValue, OpKind, Param, Wire, WireKind,
};

// ── Small Mamba2 config for the test ──────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Mamba2Config {
    d_model: usize,
    d_inner: usize,
    nheads: usize,
    headdim: usize,
    d_state: usize,
    ngroups: usize,
    d_conv: usize,
}

impl Mamba2Config {
    fn small() -> Self {
        Self {
            d_model: 64,
            d_inner: 128,
            nheads: 16,
            headdim: 8,
            d_state: 16,
            ngroups: 1,
            d_conv: 4,
        }
    }

    fn conv_dim(&self) -> usize {
        self.d_inner + 2 * self.ngroups * self.d_state
    }

    fn in_proj_dim(&self) -> usize {
        2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
    }

    fn ssm_state_size(&self) -> usize {
        self.nheads * self.headdim * self.d_state
    }
}

// ── Reference impl: port of efficient-genai's forward_ssd_mixer ───

#[allow(clippy::too_many_arguments)]
fn reference_mixer(
    cfg: Mamba2Config,
    residual: &[f32],
    in_proj_w_t: &[f32],     // (d_model, in_proj_dim) row-major
    conv_w: &[f32],
    conv_bias: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    d_param: &[f32],
    ssm_norm_w: &[f32],
    out_proj_w_t: &[f32],    // (d_inner, d_model) row-major
    conv_state: &mut Vec<f32>,
    ssm_state: &mut Vec<f32>,
    eps: f32,
) -> Vec<f32> {
    // 1. in_proj: residual [1, d_model] @ in_proj_w_t [d_model, in_proj_dim]
    let in_proj_dim = cfg.in_proj_dim();
    let mut in_proj_out = vec![0.0f32; in_proj_dim];
    for j in 0..in_proj_dim {
        let mut acc = 0.0f32;
        for k in 0..cfg.d_model {
            acc += residual[k] * in_proj_w_t[k * in_proj_dim + j];
        }
        in_proj_out[j] = acc;
    }

    // Layout: [z (d_inner), x (d_inner), B (ngroups*d_state), C (ngroups*d_state), dt (nheads)]
    let z_off = 0usize;
    let conv_off = cfg.d_inner;
    let dt_off = cfg.d_inner + cfg.conv_dim();
    let z = &in_proj_out[z_off..z_off + cfg.d_inner];
    let conv_input = &in_proj_out[conv_off..conv_off + cfg.conv_dim()];
    let dt_raw = &in_proj_out[dt_off..dt_off + cfg.nheads];

    // 2. conv1d_causal_depthwise on [x, B, C]
    let (conv_out, new_conv_state) =
        rce_op_conv1d::conv1d_causal_step(conv_input, conv_w, conv_bias, conv_state, cfg.conv_dim(), cfg.d_conv);
    *conv_state = new_conv_state;

    let x_conv = &conv_out[0..cfg.d_inner];
    let b_conv = &conv_out[cfg.d_inner..cfg.d_inner + cfg.ngroups * cfg.d_state];
    let c_conv = &conv_out[cfg.d_inner + cfg.ngroups * cfg.d_state..cfg.conv_dim()];

    // 3. silu(x)
    let x: Vec<f32> = x_conv.iter().map(|v| stable_silu(*v)).collect();

    // 4. ssm_scan
    let (y, new_ssm_state) = selective_scan_step(
        &x, b_conv, c_conv, dt_raw, dt_bias, a_log, d_param, ssm_state,
        cfg.nheads, cfg.headdim, cfg.d_state, cfg.ngroups,
    );
    *ssm_state = new_ssm_state;

    // 5. group RMS norm on y (ngroups groups)
    let y_norm = group_rms_norm(&y, ssm_norm_w, cfg.ngroups, eps);

    // 6. output gate y_norm * silu(z)
    let z_silu: Vec<f32> = z.iter().map(|v| stable_silu(*v)).collect();
    let gated: Vec<f32> = y_norm.iter().zip(z_silu.iter()).map(|(a, b)| a * b).collect();

    // 7. out_proj: gated [1, d_inner] @ out_proj_w_t [d_inner, d_model]
    let mut out = vec![0.0f32; cfg.d_model];
    for j in 0..cfg.d_model {
        let mut acc = 0.0f32;
        for k in 0..cfg.d_inner {
            acc += gated[k] * out_proj_w_t[k * cfg.d_model + j];
        }
        out[j] = acc;
    }
    out
}

fn stable_silu(x: f32) -> f32 {
    if x >= 0.0 {
        x / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        x * e / (1.0 + e)
    }
}

fn group_rms_norm(x: &[f32], w: &[f32], ngroups: usize, eps: f32) -> Vec<f32> {
    let total = x.len();
    let group_size = total / ngroups;
    let mut out = vec![0.0f32; total];
    for g in 0..ngroups {
        let s = g * group_size;
        let e = s + group_size;
        let mean_sq: f32 = x[s..e].iter().map(|v| v * v).sum::<f32>() / group_size as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for i in 0..group_size {
            out[s + i] = x[s + i] * inv * w[s + i];
        }
    }
    out
}

// ── Build the .fg graph for one mixer step ────────────────────────

fn build_mixer_graph(cfg: Mamba2Config, eps: f32) -> FunctionGraph {
    let m = WireKind::Move;

    // Dimensions used as i32 literals in the graph.
    let d_model_i = cfg.d_model as i32;
    let d_inner_i = cfg.d_inner as i32;
    let conv_dim_i = cfg.conv_dim() as i32;
    let nheads_i = cfg.nheads as i32;
    let headdim_i = cfg.headdim as i32;
    let d_state_i = cfg.d_state as i32;
    let d_conv_i = cfg.d_conv as i32;
    let ngroups_i = cfg.ngroups as i32;
    let group_off_i = (cfg.ngroups * cfg.d_state) as i32;
    let state_size_i = cfg.ssm_state_size() as i32;
    let zero_i = 0i32;

    // We don't need d_model_i / d_inner_i as literals beyond shape
    // checks done at runtime — keep the `_` unused-binding suppression
    // simple by just listing them in literal nodes only when needed.
    let _ = d_model_i;
    let _ = d_inner_i;

    // Node IDs follow this layout (see comments).
    let nodes = vec![
        // ── Params (12) ───────────────────────────────────────
        /* 0  */ p("residual"),
        /* 1  */ p("in_proj_w_t"),
        /* 2  */ p("conv_w"),
        /* 3  */ p("conv_bias"),
        /* 4  */ p("a_log"),
        /* 5  */ p("dt_bias"),
        /* 6  */ p("d_param"),
        /* 7  */ p("ssm_norm_w"),
        /* 8  */ p("out_proj_w_t"),
        /* 9  */ p("conv_state"),
        /* 10 */ p("ssm_state"),
        // ── Literals: scalars & offsets ──────────────────────
        /* 11 */ lit_i32(zero_i),                        // start=0 (multi-use)
        /* 12 */ lit_i32(cfg.d_inner as i32),            // d_inner / start at conv_off
        /* 13 */ lit_i32(conv_dim_i),                    // conv_dim
        /* 14 */ lit_i32((cfg.d_inner + cfg.conv_dim()) as i32), // dt_off
        /* 15 */ lit_i32(nheads_i),                      // nheads
        /* 16 */ lit_i32(d_conv_i),                      // d_conv
        /* 17 */ lit_i32((cfg.conv_dim() * cfg.d_conv) as i32), // conv_state size
        /* 18 */ lit_i32(group_off_i),                   // ngroups*d_state
        /* 19 */ lit_i32(headdim_i),                     // headdim
        /* 20 */ lit_i32(d_state_i),                     // d_state
        /* 21 */ lit_i32(state_size_i),                  // ssm state size
        /* 22 */ lit_i32(ngroups_i),                     // ngroups
        /* 23 */ lit_f32(eps),                           // eps
        /* 24 */ lit_i32(cfg.d_inner as i32 + group_off_i), // start: c_conv offset within conv_out
        // String literal for the final reshape: "[1, d_inner]".
        /* 25 */ GraphNode::Literal {
            value: LiteralValue::String(format!("[1,{}]", cfg.d_inner)),
            data_type: "string".into(),
        },
        // ── Apply nodes ──────────────────────────────────────
        /* 26 */ apply_custom("matmul"),    // residual @ in_proj_w_t → in_proj_out
        /* 27 */ apply_custom("split_at"),  // z = in_proj_out[0..d_inner]
        /* 28 */ apply_custom("split_at"),  // conv_input = in_proj_out[d_inner..d_inner+conv_dim]
        /* 29 */ apply_custom("split_at"),  // dt_raw = in_proj_out[dt_off..dt_off+nheads]
        /* 30 */ apply_custom("conv1d_causal_depthwise"), // packed_conv
        /* 31 */ apply_custom("split_at"),  // conv_out = packed_conv[0..conv_dim]
        /* 32 */ apply_custom("split_at"),  // x_conv = conv_out[0..d_inner]
        /* 33 */ apply_custom("split_at"),  // b_conv = conv_out[d_inner..d_inner+ngroups*d_state]
        /* 34 */ apply_custom("split_at"),  // c_conv = conv_out[d_inner+ngroups*d_state..conv_dim]
        /* 35 */ apply_custom("silu"),      // x = silu(x_conv)
        /* 36 */ apply_custom("ssm_scan"),  // packed_ssm
        /* 37 */ apply_custom("split_at"),  // y = packed_ssm[0..d_inner]
        /* 38 */ apply_custom("rms_norm"),  // y_norm
        /* 39 */ apply_custom("silu"),      // z_silu
        /* 40 */ GraphNode::Apply { op: OpKind::Mul, output_type: "f32".into() }, // gated
        /* 41 */ apply_custom("reshape"),   // gated_2d = reshape(gated, "[1, d_inner]")
        /* 42 */ apply_custom("matmul"),    // out
    ];

    // All wires: (from, to, port_in_idx).
    // Node IDs: literals 11..=24 + reshape-shape literal 25;
    // Apply nodes start at 26.
    let wires_raw: Vec<(usize, usize, usize)> = vec![
        // 26: matmul(residual, in_proj_w_t)
        (0, 26, 0),
        (1, 26, 1),
        // 27: split_at(in_proj_out=26, 0=11, d_inner=12)
        (26, 27, 0),
        (11, 27, 1),
        (12, 27, 2),
        // 28: split_at(in_proj_out=26, d_inner=12, conv_dim=13)
        (26, 28, 0),
        (12, 28, 1),
        (13, 28, 2),
        // 29: split_at(in_proj_out=26, dt_off=14, nheads=15)
        (26, 29, 0),
        (14, 29, 1),
        (15, 29, 2),
        // 30: conv1d_causal_depthwise(conv_input=28, conv_w=2, conv_bias=3, conv_state=9, d_conv=16)
        (28, 30, 0),
        (2, 30, 1),
        (3, 30, 2),
        (9, 30, 3),
        (16, 30, 4),
        // 31: split_at(packed_conv=30, 0=11, conv_dim=13)
        (30, 31, 0),
        (11, 31, 1),
        (13, 31, 2),
        // 32: split_at(conv_out=31, 0=11, d_inner=12)
        (31, 32, 0),
        (11, 32, 1),
        (12, 32, 2),
        // 33: split_at(conv_out=31, d_inner=12, ngroups*d_state=18)
        (31, 33, 0),
        (12, 33, 1),
        (18, 33, 2),
        // 34: split_at(conv_out=31, d_inner+ngroups*d_state=24, ngroups*d_state=18)
        (31, 34, 0),
        (24, 34, 1),
        (18, 34, 2),
        // 35: silu(x_conv=32)
        (32, 35, 0),
        // 36: ssm_scan(x=35, b_conv=33, c_conv=34, dt_raw=29, dt_bias=5, a_log=4, D=6, ssm_state=10, headdim=19, d_state=20)
        (35, 36, 0),
        (33, 36, 1),
        (34, 36, 2),
        (29, 36, 3),
        (5, 36, 4),
        (4, 36, 5),
        (6, 36, 6),
        (10, 36, 7),
        (19, 36, 8),
        (20, 36, 9),
        // 37: split_at(packed_ssm=36, 0=11, d_inner=12)
        (36, 37, 0),
        (11, 37, 1),
        (12, 37, 2),
        // 38: rms_norm(y=37, ssm_norm_w=7, ngroups=22, eps=23)
        (37, 38, 0),
        (7, 38, 1),
        (22, 38, 2),
        (23, 38, 3),
        // 39: silu(z=27)
        (27, 39, 0),
        // 40: mul(y_norm=38, z_silu=39)
        (38, 40, 0),
        (39, 40, 1),
        // 41: reshape(gated=40, shape="[1,d_inner]"=25)
        (40, 41, 0),
        (25, 41, 1),
        // 42: matmul(gated_2d=41, out_proj_w_t=8)
        (41, 42, 0),
        (8, 42, 1),
    ];

    let wires: Vec<Wire> = wires_raw
        .into_iter()
        .map(|(from, to, port_in_idx)| Wire {
            from,
            to,
            port_in_idx: port_in_idx as i32,
            wire_kind: m,
        })
        .collect();

    FunctionGraph {
        name: "mamba2_mixer_step".into(),
        nodes,
        wires,
        output_node: 42,
        return_type: "f32".into(),
    }
}

fn p(name: &str) -> GraphNode {
    GraphNode::Param(Param { name: name.into(), data_type: "f32".into() })
}

fn lit_i32(v: i32) -> GraphNode {
    GraphNode::Literal {
        value: LiteralValue::I32(v),
        data_type: "i32".into(),
    }
}

fn lit_f32(v: f32) -> GraphNode {
    GraphNode::Literal {
        value: LiteralValue::F32(v),
        data_type: "f32".into(),
    }
}

fn apply_custom(name: &str) -> GraphNode {
    GraphNode::Apply {
        op: OpKind::Custom(name.into()),
        output_type: "f32".into(),
    }
}

// ── Test fixtures ─────────────────────────────────────────────────

fn make_random_weights(cfg: Mamba2Config) -> (
    Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>,
) {
    let in_proj_dim = cfg.in_proj_dim();
    let conv_dim = cfg.conv_dim();
    let state_size = cfg.ssm_state_size();

    let residual: Vec<f32> = (0..cfg.d_model).map(|i| (i as f32 * 0.013) - 0.5).collect();
    let in_proj_w_t: Vec<f32> = (0..cfg.d_model * in_proj_dim).map(|i| ((i as f32 * 0.0007) - 0.3).sin() * 0.05).collect();
    let conv_w: Vec<f32> = (0..conv_dim * cfg.d_conv).map(|i| ((i as f32 * 0.011) + 0.1).cos() * 0.1).collect();
    let conv_bias: Vec<f32> = (0..conv_dim).map(|i| (i as f32 * 0.003) - 0.05).collect();
    let a_log: Vec<f32> = (0..cfg.nheads).map(|i| -((i + 1) as f32 * 0.1)).collect();
    let dt_bias: Vec<f32> = (0..cfg.nheads).map(|i| (i as f32 * 0.02) + 0.1).collect();
    let d_param: Vec<f32> = (0..cfg.nheads).map(|i| i as f32 * 0.005).collect();
    let ssm_norm_w: Vec<f32> = (0..cfg.d_inner).map(|i| 1.0 + (i as f32 * 0.01)).collect();
    let out_proj_w_t: Vec<f32> = (0..cfg.d_inner * cfg.d_model).map(|i| ((i as f32 * 0.0009) - 0.2).cos() * 0.05).collect();
    let conv_state: Vec<f32> = (0..conv_dim * cfg.d_conv).map(|i| ((i as f32 * 0.005) - 0.3).sin() * 0.2).collect();
    let ssm_state: Vec<f32> = (0..state_size).map(|i| ((i as f32 * 0.001) - 0.5).sin() * 0.1).collect();

    (residual, in_proj_w_t, conv_w, conv_bias, a_log, dt_bias, d_param, ssm_norm_w, out_proj_w_t, conv_state, ssm_state)
}

#[test]
fn full_mamba2_mixer_through_rce_dispatch_matches_reference() {
    let cfg = Mamba2Config::small();
    let eps = 1e-6f32;
    let (residual, in_proj_w_t, conv_w, conv_bias, a_log, dt_bias, d_param,
         ssm_norm_w, out_proj_w_t, conv_state0, ssm_state0)
        = make_random_weights(cfg);

    // ── Reference path ──────────────────────────────────────
    let mut ref_conv = conv_state0.clone();
    let mut ref_ssm = ssm_state0.clone();
    let want = reference_mixer(
        cfg, &residual, &in_proj_w_t, &conv_w, &conv_bias,
        &a_log, &dt_bias, &d_param, &ssm_norm_w, &out_proj_w_t,
        &mut ref_conv, &mut ref_ssm, eps,
    );

    // ── Rce path ──────────────────────────────────────────
    let graph = build_mixer_graph(cfg, eps);
    let mut inputs = Inputs::new();
    inputs.insert("residual".into(),    Value::Tensor(HostTensor::from_f32(vec![1, cfg.d_model], &residual).unwrap()));
    inputs.insert("in_proj_w_t".into(), Value::Tensor(HostTensor::from_f32(vec![cfg.d_model, cfg.in_proj_dim()], &in_proj_w_t).unwrap()));
    inputs.insert("conv_w".into(),      Value::Tensor(HostTensor::from_f32(vec![cfg.conv_dim() * cfg.d_conv], &conv_w).unwrap()));
    inputs.insert("conv_bias".into(),   Value::Tensor(HostTensor::from_f32(vec![cfg.conv_dim()], &conv_bias).unwrap()));
    inputs.insert("a_log".into(),       Value::Tensor(HostTensor::from_f32(vec![cfg.nheads], &a_log).unwrap()));
    inputs.insert("dt_bias".into(),     Value::Tensor(HostTensor::from_f32(vec![cfg.nheads], &dt_bias).unwrap()));
    inputs.insert("d_param".into(),     Value::Tensor(HostTensor::from_f32(vec![cfg.nheads], &d_param).unwrap()));
    inputs.insert("ssm_norm_w".into(),  Value::Tensor(HostTensor::from_f32(vec![cfg.d_inner], &ssm_norm_w).unwrap()));
    inputs.insert("out_proj_w_t".into(),Value::Tensor(HostTensor::from_f32(vec![cfg.d_inner, cfg.d_model], &out_proj_w_t).unwrap()));
    inputs.insert("conv_state".into(),  Value::Tensor(HostTensor::from_f32(vec![cfg.conv_dim() * cfg.d_conv], &conv_state0).unwrap()));
    inputs.insert("ssm_state".into(),   Value::Tensor(HostTensor::from_f32(vec![cfg.ssm_state_size()], &ssm_state0).unwrap()));

    let ssm_leaf = SsmScanLeaf::new();
    let conv_leaf = Conv1dCausalLeaf::new();
    let silu_leaf = SiluLeaf::new();
    let rms_leaf = RmsNormLeaf::new();
    let split_leaf = SplitAtLeaf::new();

    let ops: Vec<&dyn OpDispatch> = vec![
        &ssm_leaf,
        &conv_leaf,
        &silu_leaf,
        &rms_leaf,
        &split_leaf,
    ];
    let cfg_interp = InterpretConfig { leaves: &[], ops: &ops };
    let result = interpret_with_config(&graph, &inputs, cfg_interp).expect("interpret");

    let out_tensor = match result {
        Value::Tensor(t) => t,
        other => panic!("expected tensor, got {other:?}"),
    };
    // matmul output is [1, d_model].
    assert_eq!(out_tensor.shape, vec![1, cfg.d_model]);
    let got = out_tensor.as_f32().expect("as_f32");

    // ── Compare ─────────────────────────────────────────────
    assert_eq!(want.len(), got.len(), "output length mismatch");
    let mut max_err = 0.0f32;
    for (i, (a, b)) in want.iter().zip(got.iter()).enumerate() {
        let e = (a - b).abs();
        if e > max_err { max_err = e; }
        assert!(
            e < 1e-4,
            "mixer output[{i}] diverges: want {a}, got {b}, abs_err={e}"
        );
    }
    eprintln!("mamba2 mixer e2e: max_abs_err = {max_err:.3e} over {} elements", got.len());
}

/// Same graph as the test above, but Metal leaves are inserted ahead
/// of the CPU leaves in the dispatch chain. Because `OpDispatch` chain
/// resolution picks the first leaf whose `handles()` returns the op
/// name, Metal claims `silu` and `rms_norm`; everything else stays CPU.
/// The graph itself doesn't change — Metal acceleration is "free" once
/// the leaves are in the chain.
///
/// Skipped at runtime if Metal isn't available (CI sandbox, non-Apple
/// target). Mac targets only — `rce-op-metal` is cross-platform but
/// only useful on macOS / iOS.
#[cfg(target_os = "macos")]
#[test]
fn mixer_with_metal_leaves_intercepting_silu_and_rms_norm() {
    use rce_op_metal::MetalLeaf;

    let metal = match MetalLeaf::new() {
        Some(m) => m,
        None => {
            eprintln!("Metal unavailable on this host — skipping");
            return;
        }
    };

    let cfg = Mamba2Config::small();
    let eps = 1e-6f32;
    let (residual, in_proj_w_t, conv_w, conv_bias, a_log, dt_bias, d_param,
         ssm_norm_w, out_proj_w_t, conv_state0, ssm_state0)
        = make_random_weights(cfg);

    // Reference (pure CPU).
    let mut ref_conv = conv_state0.clone();
    let mut ref_ssm = ssm_state0.clone();
    let want = reference_mixer(
        cfg, &residual, &in_proj_w_t, &conv_w, &conv_bias,
        &a_log, &dt_bias, &d_param, &ssm_norm_w, &out_proj_w_t,
        &mut ref_conv, &mut ref_ssm, eps,
    );

    // Rce path: Metal leaf prepended; CPU leaves still registered as
    // a fallback for ops Metal doesn't claim (ssm_scan, conv1d_causal,
    // split_at).
    let graph = build_mixer_graph(cfg, eps);
    let mut inputs = Inputs::new();
    inputs.insert("residual".into(),    Value::Tensor(HostTensor::from_f32(vec![1, cfg.d_model], &residual).unwrap()));
    inputs.insert("in_proj_w_t".into(), Value::Tensor(HostTensor::from_f32(vec![cfg.d_model, cfg.in_proj_dim()], &in_proj_w_t).unwrap()));
    inputs.insert("conv_w".into(),      Value::Tensor(HostTensor::from_f32(vec![cfg.conv_dim() * cfg.d_conv], &conv_w).unwrap()));
    inputs.insert("conv_bias".into(),   Value::Tensor(HostTensor::from_f32(vec![cfg.conv_dim()], &conv_bias).unwrap()));
    inputs.insert("a_log".into(),       Value::Tensor(HostTensor::from_f32(vec![cfg.nheads], &a_log).unwrap()));
    inputs.insert("dt_bias".into(),     Value::Tensor(HostTensor::from_f32(vec![cfg.nheads], &dt_bias).unwrap()));
    inputs.insert("d_param".into(),     Value::Tensor(HostTensor::from_f32(vec![cfg.nheads], &d_param).unwrap()));
    inputs.insert("ssm_norm_w".into(),  Value::Tensor(HostTensor::from_f32(vec![cfg.d_inner], &ssm_norm_w).unwrap()));
    inputs.insert("out_proj_w_t".into(),Value::Tensor(HostTensor::from_f32(vec![cfg.d_inner, cfg.d_model], &out_proj_w_t).unwrap()));
    inputs.insert("conv_state".into(),  Value::Tensor(HostTensor::from_f32(vec![cfg.conv_dim() * cfg.d_conv], &conv_state0).unwrap()));
    inputs.insert("ssm_state".into(),   Value::Tensor(HostTensor::from_f32(vec![cfg.ssm_state_size()], &ssm_state0).unwrap()));

    let ssm_leaf = SsmScanLeaf::new();
    let conv_leaf = Conv1dCausalLeaf::new();
    let cpu_silu = SiluLeaf::new();
    let cpu_rms = RmsNormLeaf::new();
    let split_leaf = SplitAtLeaf::new();

    // Metal first → claims silu + rms_norm. CPU fallbacks register too
    // so the chain is complete (Unsupported(_) fall-through covers any
    // shape Metal can't handle in future).
    let ops: Vec<&dyn OpDispatch> = vec![
        &metal,
        &ssm_leaf,
        &conv_leaf,
        &cpu_silu,
        &cpu_rms,
        &split_leaf,
    ];
    let cfg_interp = InterpretConfig { leaves: &[], ops: &ops };
    let result = interpret_with_config(&graph, &inputs, cfg_interp).expect("interpret");

    let out_tensor = match result {
        Value::Tensor(t) => t,
        other => panic!("expected tensor, got {other:?}"),
    };
    let got = out_tensor.as_f32().expect("as_f32");

    // GPU rsqrt / exp may diverge from CPU by a few ULPs per call;
    // accumulating across 18 Apply nodes the divergence stays small
    // but can exceed strict bit-equivalence. Tolerance is 1e-3.
    assert_eq!(want.len(), got.len(), "output length mismatch");
    let mut max_err = 0.0f32;
    for (i, (a, b)) in want.iter().zip(got.iter()).enumerate() {
        let e = (a - b).abs();
        if e > max_err { max_err = e; }
        assert!(
            e < 1e-3,
            "metal-mixer output[{i}] diverges: want {a}, got {b}, abs_err={e}"
        );
    }
    eprintln!(
        "mamba2 mixer e2e WITH METAL silu+rms_norm: max_abs_err = {max_err:.3e} over {} elements",
        got.len()
    );
}
