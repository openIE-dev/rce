//! **Full Jamba single-token forward through rce dispatch with the
//! real SSM mixer wired in.** Replaces the identity-mixer stub from
//! `jamba_28_block_forward.rs` with the validated Jamba SSM block on
//! all 26 SSM blocks. The 2 attention blocks (7 and 21) are computed
//! inline (single-token MQA degenerates to V-projection-and-tile, no
//! attention scoring needed for the first token); future turns will
//! route them through `rce-op-attention`.
//!
//! ## Pipeline (28-block forward, single token)
//!
//! ```text
//!   token_id 1 → embedding lookup → residual [1, d_model=2560]
//!     │
//!     ├── for each block in 0..28:
//!     │     residual += mixer(rms_norm(residual, attn_norm.weight)):
//!     │       SSM block (26 of 28):  rce dispatch through 23-Apply graph
//!     │       attention block (2):   inline MQA (single-token degenerate)
//!     │     residual += ffn(rms_norm(residual, ffn_norm.weight)):
//!     │       rce dispatch through 5-Apply FFN graph (matmul × 3 + swiglu)
//!     │
//!     ▼
//!   output_norm + lm_head + argmax → token id
//! ```
//!
//! ## Validation
//!
//! Inline reference forward computes the same forward in pure Rust.
//! We assert rce's per-block residual matches at every block within
//! a tight tolerance (3e-3 relative — accumulated drift across 28
//! blocks × ~30 Apply nodes per block from f32 IR machinery's
//! intermediate Vec<u8> roundtrips). Final token id must match
//! exactly.

use rce_ingest_gguf::{read_tensor_f32, GgufFile};
use rce_op_conv1d::Conv1dCausalLeaf;
use rce_op_nn_prims::{ArgmaxLeaf, EmbeddingLookupLeaf, RmsNormLeaf, SiluLeaf, SplitAtLeaf, SwigluLeaf};
use rce_op_ssm::SsmScanMamba1Leaf;
use rce_vm_backend_cpu::{
    interpret_with_config, HostTensor, Inputs, InterpretConfig, OpDispatch, ScalarValue, Value,
};
use rce_graph::graph::{
    FunctionGraph, GraphNode, LiteralValue, OpKind, Param, Wire, WireKind,
};
use memmap2::Mmap;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

const JAMBA_GGUF: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/jamba-reasoning-3b-gguf/jamba-reasoning-3b-Q4_K_M.gguf";

#[test]
fn jamba_full_forward_with_ssm_mixer_emits_token() {
    let path = PathBuf::from(JAMBA_GGUF);
    if !path.exists() {
        eprintln!("jamba_full_forward_with_ssm: GGUF not found — skipping.");
        return;
    }

    let file = File::open(&path).expect("open");
    let mmap = unsafe { Mmap::map(&file) }.expect("mmap");
    let bytes: &[u8] = &mmap;
    let gguf = GgufFile::parse(bytes).expect("parse");

    // Architectural constants for Jamba-Reasoning-3B.
    let d_model = 2560usize;
    let d_inner_ssm = 5120usize;
    let d_state = 16usize;
    let dt_rank = 160usize;
    let d_conv = 4usize;
    let d_inner_ffn = 8192usize;
    let n_heads = 20usize;
    let d_head = 128usize;
    let _n_kv_heads = 1usize; // MQA
    let eps = 1e-6f32;

    // Block-type detection.
    let ssm_blocks: Vec<usize> = (0..28).filter(|i| {
        gguf.tensors.iter().any(|t| t.name == format!("blk.{i}.ssm_in.weight"))
    }).collect();
    let attn_blocks: Vec<usize> = (0..28).filter(|i| {
        gguf.tensors.iter().any(|t| t.name == format!("blk.{i}.attn_q.weight"))
    }).collect();
    eprintln!("SSM blocks ({}): {:?}", ssm_blocks.len(), ssm_blocks);
    eprintln!("attention blocks ({}): {:?}", attn_blocks.len(), attn_blocks);
    assert_eq!(ssm_blocks.len() + attn_blocks.len(), 28);

    // ── Embedding ─────────────────────────────────────────────
    let token_id = 1i32;
    let t_load = Instant::now();
    let token_embd = read_tensor_f32(bytes, &gguf, "token_embd.weight").expect("token_embd");
    let vocab = token_embd.len() / d_model;
    eprintln!("vocab={vocab}, embedding dequant {:?}", t_load.elapsed());

    let row_start = token_id as usize * d_model;
    let mut ref_residual: Vec<f32> = token_embd[row_start..row_start + d_model].to_vec();
    let mut rce_residual = ref_residual.clone();
    drop(token_embd); // free 1.34 GB

    // ── Build reusable FFN block graph ────────────────────────
    // Same as jamba_28_block_forward's per-block FFN graph.
    let ffn_graph = build_ffn_block_graph();

    let conv_leaf = Conv1dCausalLeaf::new();
    let ssm_leaf = SsmScanMamba1Leaf::new();
    let silu_leaf = SiluLeaf::new();
    let rms_leaf = RmsNormLeaf::new();
    let split_leaf = SplitAtLeaf::new();
    let swiglu_leaf = SwigluLeaf::new();
    let _emb_leaf = EmbeddingLookupLeaf::new();
    let _argmax_leaf = ArgmaxLeaf::new();
    let ops: Vec<&dyn OpDispatch> = vec![
        &conv_leaf, &ssm_leaf, &silu_leaf, &rms_leaf, &split_leaf, &swiglu_leaf,
    ];

    let t_blocks = Instant::now();
    let mut total_ssm_dispatch_ms = 0u128;
    let mut total_ffn_dispatch_ms = 0u128;

    for block_idx in 0..28 {
        let bn = format!("blk.{block_idx}");
        let attn_norm_w = read_tensor_f32(bytes, &gguf, &format!("{bn}.attn_norm.weight"))
            .expect("attn_norm");

        // ── Mixer step ─────────────────────────────────────────
        // attn_normed = rms_norm(residual, attn_norm_w)
        let ref_attn_normed = rms_norm_inline(&ref_residual, &attn_norm_w, eps);
        let rce_attn_normed = rms_norm_inline(&rce_residual, &attn_norm_w, eps);

        let (ref_mixer_delta, rce_mixer_delta);
        if ssm_blocks.contains(&block_idx) {
            // SSM mixer — rce dispatch.
            let ssm_in    = read_tensor_f32(bytes, &gguf, &format!("{bn}.ssm_in.weight")).expect("ssm_in");
            let conv_w    = read_tensor_f32(bytes, &gguf, &format!("{bn}.ssm_conv1d.weight")).expect("conv_w");
            let conv_b    = read_tensor_f32(bytes, &gguf, &format!("{bn}.ssm_conv1d.bias")).expect("conv_b");
            let ssm_x     = read_tensor_f32(bytes, &gguf, &format!("{bn}.ssm_x.weight")).expect("ssm_x");
            let dt_norm_w = read_tensor_f32(bytes, &gguf, &format!("{bn}.ssm_dt_norm.weight")).expect("dt_norm");
            let b_norm_w  = read_tensor_f32(bytes, &gguf, &format!("{bn}.ssm_b_norm.weight")).expect("b_norm");
            let c_norm_w  = read_tensor_f32(bytes, &gguf, &format!("{bn}.ssm_c_norm.weight")).expect("c_norm");
            let ssm_dt    = read_tensor_f32(bytes, &gguf, &format!("{bn}.ssm_dt.weight")).expect("ssm_dt");
            let dt_b      = read_tensor_f32(bytes, &gguf, &format!("{bn}.ssm_dt.bias")).expect("dt_b");
            let a_log     = read_tensor_f32(bytes, &gguf, &format!("{bn}.ssm_a")).expect("ssm_a");
            let d_param   = read_tensor_f32(bytes, &gguf, &format!("{bn}.ssm_d")).expect("ssm_d");
            let ssm_out   = read_tensor_f32(bytes, &gguf, &format!("{bn}.ssm_out.weight")).expect("ssm_out");

            let ssm_in_t  = transpose_2d(&ssm_in, 2 * d_inner_ssm, d_model);
            let ssm_x_t   = transpose_2d(&ssm_x, dt_rank + 2 * d_state, d_inner_ssm);
            let ssm_dt_t  = transpose_2d(&ssm_dt, d_inner_ssm, dt_rank);
            let ssm_out_t = transpose_2d(&ssm_out, d_model, d_inner_ssm);

            let conv_state0 = vec![0.0f32; d_inner_ssm * d_conv];
            let ssm_state0  = vec![0.0f32; d_inner_ssm * d_state];

            // Reference SSM mixer.
            ref_mixer_delta = ssm_block_forward_inline(
                &ref_attn_normed, &ssm_in_t, &conv_w, &conv_b,
                &ssm_x_t, &dt_norm_w, &b_norm_w, &c_norm_w,
                &ssm_dt_t, &dt_b, &a_log, &d_param, &ssm_out_t,
                &conv_state0, &ssm_state0,
                d_model, d_inner_ssm, d_state, dt_rank, d_conv, eps,
            );

            // Rce SSM mixer dispatch.
            let t_ssm = Instant::now();
            let graph = build_ssm_block_graph(d_model, d_inner_ssm, d_state, dt_rank, d_conv, eps);
            let mut inputs = Inputs::new();
            inputs.insert("attn_normed".into(), Value::Tensor(HostTensor::from_f32(vec![1, d_model], &rce_attn_normed).unwrap()));
            inputs.insert("ssm_in".into(),     Value::Tensor(HostTensor::from_f32(vec![d_model, 2 * d_inner_ssm], &ssm_in_t).unwrap()));
            inputs.insert("conv_w".into(),     Value::Tensor(HostTensor::from_f32(vec![d_inner_ssm * d_conv], &conv_w).unwrap()));
            inputs.insert("conv_b".into(),     Value::Tensor(HostTensor::from_f32(vec![d_inner_ssm], &conv_b).unwrap()));
            inputs.insert("conv_state".into(), Value::Tensor(HostTensor::from_f32(vec![d_inner_ssm * d_conv], &conv_state0).unwrap()));
            inputs.insert("ssm_x".into(),      Value::Tensor(HostTensor::from_f32(vec![d_inner_ssm, dt_rank + 2 * d_state], &ssm_x_t).unwrap()));
            inputs.insert("dt_norm_w".into(),  Value::Tensor(HostTensor::from_f32(vec![dt_rank], &dt_norm_w).unwrap()));
            inputs.insert("b_norm_w".into(),   Value::Tensor(HostTensor::from_f32(vec![d_state], &b_norm_w).unwrap()));
            inputs.insert("c_norm_w".into(),   Value::Tensor(HostTensor::from_f32(vec![d_state], &c_norm_w).unwrap()));
            inputs.insert("ssm_dt".into(),     Value::Tensor(HostTensor::from_f32(vec![dt_rank, d_inner_ssm], &ssm_dt_t).unwrap()));
            inputs.insert("dt_b".into(),       Value::Tensor(HostTensor::from_f32(vec![d_inner_ssm], &dt_b).unwrap()));
            inputs.insert("a_log".into(),      Value::Tensor(HostTensor::from_f32(vec![d_inner_ssm * d_state], &a_log).unwrap()));
            inputs.insert("d_param".into(),    Value::Tensor(HostTensor::from_f32(vec![d_inner_ssm], &d_param).unwrap()));
            inputs.insert("ssm_out".into(),    Value::Tensor(HostTensor::from_f32(vec![d_inner_ssm, d_model], &ssm_out_t).unwrap()));
            inputs.insert("ssm_state".into(),  Value::Tensor(HostTensor::from_f32(vec![d_inner_ssm * d_state], &ssm_state0).unwrap()));
            let result = interpret_with_config(&graph, &inputs, InterpretConfig { leaves: &[], ops: &ops })
                .unwrap_or_else(|e| panic!("ssm block {block_idx}: {e:?}"));
            rce_mixer_delta = match result {
                Value::Tensor(t) => t.as_f32().unwrap(),
                _ => panic!(),
            };
            total_ssm_dispatch_ms += t_ssm.elapsed().as_millis();
        } else {
            // Attention block — inline MQA forward (same for both paths).
            // For first-token, softmax over single position is 1.0, so
            // attention output for each head = V (which is shared across
            // heads via MQA). Concat-and-project gives us residual update.
            let attn_q = read_tensor_f32(bytes, &gguf, &format!("{bn}.attn_q.weight")).expect("attn_q");
            let attn_k = read_tensor_f32(bytes, &gguf, &format!("{bn}.attn_k.weight")).expect("attn_k");
            let attn_v = read_tensor_f32(bytes, &gguf, &format!("{bn}.attn_v.weight")).expect("attn_v");
            let attn_o = read_tensor_f32(bytes, &gguf, &format!("{bn}.attn_output.weight")).expect("attn_o");

            // GGUF [d_in, d_out] flat-row-major, so attn_q [d_model, n_heads*d_head]
            // -- reading: d_in=2560 (d_model), d_out=2560 (n_heads*d_head)
            // already in [d_model, n_heads*d_head] flat order? Let me check:
            // gguf shape was [2560, 2560] for attn_q. Per the GGUF
            // convention (dim 0 = innermost = fastest-varying = cols):
            //   shape=[d_model, n_heads*d_head]  →  rows = n_heads*d_head, cols = d_model
            // So math W = [n_heads*d_head, d_model] (rows=out, cols=in).
            // To use as rce matmul second operand [d_model, n_heads*d_head]:
            // transpose.
            let attn_q_t = transpose_2d(&attn_q, n_heads * d_head, d_model);
            // attn_k.weight shape was [d_model=2560, d_head=128] in GGUF dim list,
            // which means math W = [d_head=128, d_model=2560] (rows=out=128, cols=in=2560).
            // For matmul [1, d_model] @ [d_model, d_head] we need transpose.
            let attn_k_t = transpose_2d(&attn_k, d_head, d_model);
            let attn_v_t = transpose_2d(&attn_v, d_head, d_model);
            let attn_o_t = transpose_2d(&attn_o, d_model, d_model);

            // Inline single-token MQA — same for ref and rce paths.
            let mixer_delta_inline = single_token_mqa_inline(
                &ref_attn_normed, &attn_q_t, &attn_k_t, &attn_v_t, &attn_o_t,
                d_model, n_heads, d_head,
            );
            // Both paths get the same delta (because the inputs rce_attn_normed
            // and ref_attn_normed are bit-identical at this point).
            ref_mixer_delta = mixer_delta_inline.clone();
            rce_mixer_delta = single_token_mqa_inline(
                &rce_attn_normed, &attn_q_t, &attn_k_t, &attn_v_t, &attn_o_t,
                d_model, n_heads, d_head,
            );
        }

        // residual += mixer_delta
        for i in 0..d_model {
            ref_residual[i] += ref_mixer_delta[i];
            rce_residual[i] += rce_mixer_delta[i];
        }

        // ── FFN step ───────────────────────────────────────────
        let ffn_norm_w = read_tensor_f32(bytes, &gguf, &format!("{bn}.ffn_norm.weight")).expect("ffn_norm");
        let gate_w = read_tensor_f32(bytes, &gguf, &format!("{bn}.ffn_gate.weight")).expect("gate");
        let up_w   = read_tensor_f32(bytes, &gguf, &format!("{bn}.ffn_up.weight")).expect("up");
        let down_w = read_tensor_f32(bytes, &gguf, &format!("{bn}.ffn_down.weight")).expect("down");

        let gate_t = transpose_2d(&gate_w, d_inner_ffn, d_model);
        let up_t   = transpose_2d(&up_w, d_inner_ffn, d_model);
        let down_t = transpose_2d(&down_w, d_model, d_inner_ffn);

        // Reference FFN forward.
        let ffn_normed_ref = rms_norm_inline(&ref_residual, &ffn_norm_w, eps);
        let gate_proj = matmul_1xN_NxM(&ffn_normed_ref, &gate_t, d_model, d_inner_ffn);
        let up_proj = matmul_1xN_NxM(&ffn_normed_ref, &up_t, d_model, d_inner_ffn);
        let activated: Vec<f32> = gate_proj.iter().zip(up_proj.iter())
            .map(|(g, u)| stable_silu(*g) * *u).collect();
        let ffn_delta_ref = matmul_1xN_NxM(&activated, &down_t, d_inner_ffn, d_model);

        // Rce FFN dispatch.
        let t_ffn = Instant::now();
        let mut inputs = Inputs::new();
        inputs.insert("residual".into(),  Value::Tensor(HostTensor::from_f32(vec![1, d_model], &rce_residual).unwrap()));
        inputs.insert("ffn_norm_w".into(),Value::Tensor(HostTensor::from_f32(vec![d_model], &ffn_norm_w).unwrap()));
        inputs.insert("gate_W".into(),    Value::Tensor(HostTensor::from_f32(vec![d_model, d_inner_ffn], &gate_t).unwrap()));
        inputs.insert("up_W".into(),      Value::Tensor(HostTensor::from_f32(vec![d_model, d_inner_ffn], &up_t).unwrap()));
        inputs.insert("down_W".into(),    Value::Tensor(HostTensor::from_f32(vec![d_inner_ffn, d_model], &down_t).unwrap()));
        let result = interpret_with_config(&ffn_graph, &inputs, InterpretConfig { leaves: &[], ops: &ops })
            .unwrap_or_else(|e| panic!("ffn block {block_idx}: {e:?}"));
        let ffn_delta_rce = match result {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        total_ffn_dispatch_ms += t_ffn.elapsed().as_millis();

        for i in 0..d_model {
            ref_residual[i] += ffn_delta_ref[i];
            rce_residual[i] += ffn_delta_rce[i];
        }

        // Per-block validation.
        let max_err = ref_residual.iter().zip(rce_residual.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_abs = ref_residual.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let rel = max_err / max_abs.max(1e-9);
        if rel > 3e-3 {
            panic!("block {block_idx} ({}): max_abs_err={max_err:.3e} rel={rel:.3e} too large",
                if ssm_blocks.contains(&block_idx) { "SSM" } else { "ATTN" });
        }
        if block_idx % 4 == 0 || block_idx == 27 {
            eprintln!(
                "block {block_idx:>2} {}: ||·||∞={max_abs:.3e}, rel={rel:.3e}",
                if ssm_blocks.contains(&block_idx) { "SSM" } else { "ATTN" }
            );
        }
    }
    eprintln!(
        "all 28 blocks: {:?} (SSM dispatch: {} ms, FFN dispatch: {} ms)",
        t_blocks.elapsed(), total_ssm_dispatch_ms, total_ffn_dispatch_ms
    );

    // ── Final norm + lm_head + argmax ────────────────────────
    let output_norm_w = read_tensor_f32(bytes, &gguf, "output_norm.weight").expect("output_norm");
    let lm_head_w     = read_tensor_f32(bytes, &gguf, "output.weight").expect("output");
    let lm_head_t = transpose_2d(&lm_head_w, vocab, d_model);

    let final_normed_ref = rms_norm_inline(&ref_residual, &output_norm_w, eps);
    let logits_ref = matmul_1xN_NxM(&final_normed_ref, &lm_head_t, d_model, vocab);
    let mut ref_token = 0i32;
    let mut ref_max = f32::NEG_INFINITY;
    for (i, &v) in logits_ref.iter().enumerate() {
        if v > ref_max { ref_max = v; ref_token = i as i32; }
    }

    let final_normed_rce = rms_norm_inline(&rce_residual, &output_norm_w, eps);
    let logits_rce = matmul_1xN_NxM(&final_normed_rce, &lm_head_t, d_model, vocab);
    let mut rce_token = 0i32;
    let mut rce_max = f32::NEG_INFINITY;
    for (i, &v) in logits_rce.iter().enumerate() {
        if v > rce_max { rce_max = v; rce_token = i as i32; }
    }

    eprintln!("══════════════════════════════════════════════════");
    eprintln!("Jamba full forward (SSM via rce, FFN via rce, attention inline)");
    eprintln!("  → ref_token   = {ref_token}  (logit = {ref_max:.3e})");
    eprintln!("  → rce_token = {rce_token}  (logit = {rce_max:.3e})");
    eprintln!("══════════════════════════════════════════════════");
    assert_eq!(rce_token, ref_token, "rce-emitted token diverges from reference");
}

// ── Inline math primitives ─────────────────────────────────────────

fn rms_norm_inline(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(w.iter()).map(|(v, ww)| v * inv * ww).collect()
}

fn matmul_1xN_NxM(a: &[f32], b: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for j in 0..n {
        let mut acc = 0.0f32;
        for i in 0..k {
            acc += a[i] * b[i * n + j];
        }
        out[j] = acc;
    }
    out
}

fn stable_silu(x: f32) -> f32 {
    if x >= 0.0 { x / (1.0 + (-x).exp()) } else { let e = x.exp(); x * e / (1.0 + e) }
}

fn transpose_2d(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    out
}

/// Single-token MQA attention forward. For a length-1 sequence,
/// softmax(scores) = 1.0 (single element), so attention output for
/// each query head h is just V (same V for all 20 heads since MQA
/// with 1 KV head). The output projection mixes the 20 V copies.
#[allow(clippy::too_many_arguments)]
fn single_token_mqa_inline(
    residual: &[f32],
    q_t: &[f32],     // [d_model, n_heads * d_head]
    k_t: &[f32],     // [d_model, d_head]  (1 KV head)
    v_t: &[f32],     // [d_model, d_head]
    o_t: &[f32],     // [d_model, d_model]
    d_model: usize,
    n_heads: usize,
    d_head: usize,
) -> Vec<f32> {
    // q = residual @ q_t  →  [n_heads * d_head]
    let q = matmul_1xN_NxM(residual, q_t, d_model, n_heads * d_head);
    let _ = q; // q values matter for softmax-scaled output, but with
               // single-token softmax always = 1, we don't need q.
    // v = residual @ v_t  →  [d_head]
    let v = matmul_1xN_NxM(residual, v_t, d_model, d_head);
    let _k = matmul_1xN_NxM(residual, k_t, d_model, d_head);

    // For each query head h (0..n_heads): attn_out[h] = softmax(...) · v = 1 · v = v.
    // Concat across heads: [n_heads * d_head] = [n_heads copies of v].
    let mut multi_v = Vec::with_capacity(n_heads * d_head);
    for _h in 0..n_heads {
        multi_v.extend_from_slice(&v);
    }

    // Output projection: multi_v @ O = [d_model]
    matmul_1xN_NxM(&multi_v, o_t, n_heads * d_head, d_model)
}

#[allow(clippy::too_many_arguments)]
fn ssm_block_forward_inline(
    residual: &[f32],   // attn_normed input (already rms-normed)
    ssm_in_t: &[f32],
    conv_w: &[f32],
    conv_b: &[f32],
    ssm_x_t: &[f32],
    dt_norm_w: &[f32],
    b_norm_w: &[f32],
    c_norm_w: &[f32],
    ssm_dt_t: &[f32],
    dt_bias: &[f32],
    a_log: &[f32],
    d_param: &[f32],
    ssm_out_t: &[f32],
    conv_state: &[f32],
    ssm_state: &[f32],
    d_model: usize,
    d_inner: usize,
    d_state: usize,
    dt_rank: usize,
    d_conv: usize,
    eps: f32,
) -> Vec<f32> {
    // 1. xz = residual @ ssm_in_t  →  [2 * d_inner]
    let xz = matmul_1xN_NxM(residual, ssm_in_t, d_model, 2 * d_inner);
    let x_raw = &xz[..d_inner];
    let z = &xz[d_inner..];

    // 2. conv1d
    let (x_conv, _) = rce_op_conv1d::conv1d_causal_step(
        x_raw, conv_w, conv_b, conv_state, d_inner, d_conv,
    );

    // 3. silu
    let x_act: Vec<f32> = x_conv.iter().map(|v| stable_silu(*v)).collect();

    // 4. dt_bc = x_act @ ssm_x_t  →  [dt_rank + 2*d_state]
    let bc_dim = dt_rank + 2 * d_state;
    let dt_bc = matmul_1xN_NxM(&x_act, ssm_x_t, d_inner, bc_dim);
    let dt_raw_lr = &dt_bc[..dt_rank];
    let b_raw = &dt_bc[dt_rank..dt_rank + d_state];
    let c_raw = &dt_bc[dt_rank + d_state..];

    // 5. RMSNorms
    let dt_normed = rms_norm_inline(dt_raw_lr, dt_norm_w, eps);
    let b_normed = rms_norm_inline(b_raw, b_norm_w, eps);
    let c_normed = rms_norm_inline(c_raw, c_norm_w, eps);

    // 6. dt_proj = dt_normed @ ssm_dt_t  →  [d_inner]
    let dt_proj = matmul_1xN_NxM(&dt_normed, ssm_dt_t, dt_rank, d_inner);

    // 7. SSM scan
    let (y, _) = rce_op_ssm::selective_scan_mamba1_step(
        &x_act, &b_normed, &c_normed, &dt_proj, dt_bias, a_log, d_param, ssm_state,
        d_inner, d_state,
    );

    // 8. Output gate
    let z_silu: Vec<f32> = z.iter().map(|v| stable_silu(*v)).collect();
    let gated: Vec<f32> = y.iter().zip(z_silu.iter()).map(|(a, b)| a * b).collect();

    // 9. out = gated @ ssm_out_t  →  [d_model]
    matmul_1xN_NxM(&gated, ssm_out_t, d_inner, d_model)
}

// ── FunctionGraph builders ─────────────────────────────────────────

fn build_ssm_block_graph(
    d_model: usize, d_inner: usize, d_state: usize, dt_rank: usize, d_conv: usize, eps: f32,
) -> FunctionGraph {
    // Same shape as jamba_ssm_block_e2e's graph but takes pre-normed
    // input ("attn_normed") and outputs the mixer DELTA (no residual
    // add; caller adds outside). 22 Apply nodes.
    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut wires: Vec<Wire> = Vec::new();

    let id_residual    = nodes.len(); nodes.push(p_param("attn_normed"));
    let id_ssm_in      = nodes.len(); nodes.push(p_param("ssm_in"));
    let id_conv_w      = nodes.len(); nodes.push(p_param("conv_w"));
    let id_conv_b      = nodes.len(); nodes.push(p_param("conv_b"));
    let id_conv_state  = nodes.len(); nodes.push(p_param("conv_state"));
    let id_ssm_x       = nodes.len(); nodes.push(p_param("ssm_x"));
    let id_dt_norm_w   = nodes.len(); nodes.push(p_param("dt_norm_w"));
    let id_b_norm_w    = nodes.len(); nodes.push(p_param("b_norm_w"));
    let id_c_norm_w    = nodes.len(); nodes.push(p_param("c_norm_w"));
    let id_ssm_dt      = nodes.len(); nodes.push(p_param("ssm_dt"));
    let id_dt_b        = nodes.len(); nodes.push(p_param("dt_b"));
    let id_a_log       = nodes.len(); nodes.push(p_param("a_log"));
    let id_d_param     = nodes.len(); nodes.push(p_param("d_param"));
    let id_ssm_out     = nodes.len(); nodes.push(p_param("ssm_out"));
    let id_ssm_state   = nodes.len(); nodes.push(p_param("ssm_state"));

    let id_zero        = nodes.len(); nodes.push(lit_i32(0));
    let id_d_inner_lit = nodes.len(); nodes.push(lit_i32(d_inner as i32));
    let id_d_conv_lit  = nodes.len(); nodes.push(lit_i32(d_conv as i32));
    let id_dt_rank_lit = nodes.len(); nodes.push(lit_i32(dt_rank as i32));
    let id_d_state_lit = nodes.len(); nodes.push(lit_i32(d_state as i32));
    let id_dt_bc_skip  = nodes.len(); nodes.push(lit_i32((dt_rank + d_state) as i32));
    let id_one_lit     = nodes.len(); nodes.push(lit_i32(1));
    let id_eps_lit     = nodes.len(); nodes.push(lit_f32(eps));
    let id_shape_1_di  = nodes.len(); nodes.push(lit_str(&format!("[1,{d_inner}]")));
    let id_shape_1_dt  = nodes.len(); nodes.push(lit_str(&format!("[1,{dt_rank}]")));

    let _ = d_model;

    let id_xz = nodes.len(); nodes.push(apply_custom("matmul"));
    push_wire(&mut wires, id_residual, id_xz, 0);
    push_wire(&mut wires, id_ssm_in,   id_xz, 1);

    let id_x_raw = nodes.len(); nodes.push(apply_custom("split_at"));
    push_wire(&mut wires, id_xz, id_x_raw, 0);
    push_wire(&mut wires, id_zero, id_x_raw, 1);
    push_wire(&mut wires, id_d_inner_lit, id_x_raw, 2);

    let id_z = nodes.len(); nodes.push(apply_custom("split_at"));
    push_wire(&mut wires, id_xz, id_z, 0);
    push_wire(&mut wires, id_d_inner_lit, id_z, 1);
    push_wire(&mut wires, id_d_inner_lit, id_z, 2);

    let id_conv = nodes.len(); nodes.push(apply_custom("conv1d_causal_depthwise"));
    push_wire(&mut wires, id_x_raw,     id_conv, 0);
    push_wire(&mut wires, id_conv_w,    id_conv, 1);
    push_wire(&mut wires, id_conv_b,    id_conv, 2);
    push_wire(&mut wires, id_conv_state,id_conv, 3);
    push_wire(&mut wires, id_d_conv_lit,id_conv, 4);

    let id_x_conv = nodes.len(); nodes.push(apply_custom("split_at"));
    push_wire(&mut wires, id_conv, id_x_conv, 0);
    push_wire(&mut wires, id_zero, id_x_conv, 1);
    push_wire(&mut wires, id_d_inner_lit, id_x_conv, 2);

    let id_x_act = nodes.len(); nodes.push(apply_custom("silu"));
    push_wire(&mut wires, id_x_conv, id_x_act, 0);

    let id_x_act_2d = nodes.len(); nodes.push(apply_custom("reshape"));
    push_wire(&mut wires, id_x_act,      id_x_act_2d, 0);
    push_wire(&mut wires, id_shape_1_di, id_x_act_2d, 1);

    let id_dt_bc = nodes.len(); nodes.push(apply_custom("matmul"));
    push_wire(&mut wires, id_x_act_2d, id_dt_bc, 0);
    push_wire(&mut wires, id_ssm_x,    id_dt_bc, 1);

    let id_dt_raw_lr = nodes.len(); nodes.push(apply_custom("split_at"));
    push_wire(&mut wires, id_dt_bc, id_dt_raw_lr, 0);
    push_wire(&mut wires, id_zero, id_dt_raw_lr, 1);
    push_wire(&mut wires, id_dt_rank_lit, id_dt_raw_lr, 2);

    let id_b_raw = nodes.len(); nodes.push(apply_custom("split_at"));
    push_wire(&mut wires, id_dt_bc, id_b_raw, 0);
    push_wire(&mut wires, id_dt_rank_lit, id_b_raw, 1);
    push_wire(&mut wires, id_d_state_lit, id_b_raw, 2);

    let id_c_raw = nodes.len(); nodes.push(apply_custom("split_at"));
    push_wire(&mut wires, id_dt_bc, id_c_raw, 0);
    push_wire(&mut wires, id_dt_bc_skip, id_c_raw, 1);
    push_wire(&mut wires, id_d_state_lit, id_c_raw, 2);

    let id_dt_normed = nodes.len(); nodes.push(apply_custom("rms_norm"));
    push_wire(&mut wires, id_dt_raw_lr, id_dt_normed, 0);
    push_wire(&mut wires, id_dt_norm_w, id_dt_normed, 1);
    push_wire(&mut wires, id_one_lit,   id_dt_normed, 2);
    push_wire(&mut wires, id_eps_lit,   id_dt_normed, 3);

    let id_b_normed = nodes.len(); nodes.push(apply_custom("rms_norm"));
    push_wire(&mut wires, id_b_raw,    id_b_normed, 0);
    push_wire(&mut wires, id_b_norm_w, id_b_normed, 1);
    push_wire(&mut wires, id_one_lit,  id_b_normed, 2);
    push_wire(&mut wires, id_eps_lit,  id_b_normed, 3);

    let id_c_normed = nodes.len(); nodes.push(apply_custom("rms_norm"));
    push_wire(&mut wires, id_c_raw,    id_c_normed, 0);
    push_wire(&mut wires, id_c_norm_w, id_c_normed, 1);
    push_wire(&mut wires, id_one_lit,  id_c_normed, 2);
    push_wire(&mut wires, id_eps_lit,  id_c_normed, 3);

    let id_dt_normed_2d = nodes.len(); nodes.push(apply_custom("reshape"));
    push_wire(&mut wires, id_dt_normed,  id_dt_normed_2d, 0);
    push_wire(&mut wires, id_shape_1_dt, id_dt_normed_2d, 1);

    let id_dt_proj = nodes.len(); nodes.push(apply_custom("matmul"));
    push_wire(&mut wires, id_dt_normed_2d, id_dt_proj, 0);
    push_wire(&mut wires, id_ssm_dt,       id_dt_proj, 1);

    let id_ssm_packed = nodes.len(); nodes.push(apply_custom("ssm_scan_mamba1"));
    push_wire(&mut wires, id_x_act,    id_ssm_packed, 0);
    push_wire(&mut wires, id_b_normed, id_ssm_packed, 1);
    push_wire(&mut wires, id_c_normed, id_ssm_packed, 2);
    push_wire(&mut wires, id_dt_proj,  id_ssm_packed, 3);
    push_wire(&mut wires, id_dt_b,     id_ssm_packed, 4);
    push_wire(&mut wires, id_a_log,    id_ssm_packed, 5);
    push_wire(&mut wires, id_d_param,  id_ssm_packed, 6);
    push_wire(&mut wires, id_ssm_state,id_ssm_packed, 7);
    push_wire(&mut wires, id_d_inner_lit, id_ssm_packed, 8);
    push_wire(&mut wires, id_d_state_lit, id_ssm_packed, 9);

    let id_y = nodes.len(); nodes.push(apply_custom("split_at"));
    push_wire(&mut wires, id_ssm_packed, id_y, 0);
    push_wire(&mut wires, id_zero, id_y, 1);
    push_wire(&mut wires, id_d_inner_lit, id_y, 2);

    let id_z_silu = nodes.len(); nodes.push(apply_custom("silu"));
    push_wire(&mut wires, id_z, id_z_silu, 0);

    let id_gated = nodes.len();
    nodes.push(GraphNode::Apply { op: OpKind::Mul, output_type: "f32".into() });
    push_wire(&mut wires, id_y,      id_gated, 0);
    push_wire(&mut wires, id_z_silu, id_gated, 1);

    let id_gated_2d = nodes.len(); nodes.push(apply_custom("reshape"));
    push_wire(&mut wires, id_gated, id_gated_2d, 0);
    push_wire(&mut wires, id_shape_1_di, id_gated_2d, 1);

    let id_delta = nodes.len(); nodes.push(apply_custom("matmul"));
    push_wire(&mut wires, id_gated_2d, id_delta, 0);
    push_wire(&mut wires, id_ssm_out,  id_delta, 1);

    FunctionGraph {
        name: "jamba_ssm_block_delta".into(),
        nodes,
        wires,
        output_node: id_delta,
        return_type: "f32".into(),
    }
}

fn build_ffn_block_graph() -> FunctionGraph {
    let m = WireKind::Move;
    let nodes = vec![
        /*  0 */ p_param("residual"),
        /*  1 */ p_param("ffn_norm_w"),
        /*  2 */ p_param("gate_W"),
        /*  3 */ p_param("up_W"),
        /*  4 */ p_param("down_W"),
        /*  5 */ lit_i32(1),
        /*  6 */ lit_f32(1e-6),
        /*  7 */ apply_custom("rms_norm"),
        /*  8 */ apply_custom("matmul"),
        /*  9 */ apply_custom("matmul"),
        /* 10 */ apply_custom("swiglu"),
        /* 11 */ apply_custom("matmul"),
    ];
    let wires = vec![
        Wire { from: 0, to: 7, port_in_idx: 0, wire_kind: m },
        Wire { from: 1, to: 7, port_in_idx: 1, wire_kind: m },
        Wire { from: 5, to: 7, port_in_idx: 2, wire_kind: m },
        Wire { from: 6, to: 7, port_in_idx: 3, wire_kind: m },
        Wire { from: 7, to: 8, port_in_idx: 0, wire_kind: m },
        Wire { from: 2, to: 8, port_in_idx: 1, wire_kind: m },
        Wire { from: 7, to: 9, port_in_idx: 0, wire_kind: m },
        Wire { from: 3, to: 9, port_in_idx: 1, wire_kind: m },
        Wire { from: 8, to: 10, port_in_idx: 0, wire_kind: m },
        Wire { from: 9, to: 10, port_in_idx: 1, wire_kind: m },
        Wire { from: 10, to: 11, port_in_idx: 0, wire_kind: m },
        Wire { from: 4, to: 11, port_in_idx: 1, wire_kind: m },
    ];
    FunctionGraph {
        name: "ffn_block_delta".into(),
        nodes,
        wires,
        output_node: 11,
        return_type: "f32".into(),
    }
}

fn p_param(name: &str) -> GraphNode {
    GraphNode::Param(Param { name: name.into(), data_type: "f32".into() })
}
fn apply_custom(name: &str) -> GraphNode {
    GraphNode::Apply { op: OpKind::Custom(name.into()), output_type: "f32".into() }
}
fn lit_i32(v: i32) -> GraphNode {
    GraphNode::Literal { value: LiteralValue::I32(v), data_type: "i32".into() }
}
fn lit_f32(v: f32) -> GraphNode {
    GraphNode::Literal { value: LiteralValue::F32(v), data_type: "f32".into() }
}
fn lit_str(s: &str) -> GraphNode {
    GraphNode::Literal { value: LiteralValue::String(s.into()), data_type: "string".into() }
}
fn push_wire(wires: &mut Vec<Wire>, from: usize, to: usize, port: i32) {
    wires.push(Wire { from, to, port_in_idx: port, wire_kind: WireKind::Move });
}
fn _unused(_: &ScalarValue) {}
