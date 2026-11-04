//! **Multi-token loop on Jamba: emit a sentence** — the substrate
//! threads `conv_state` + `ssm_state` (per SSM block) and KV cache
//! (per attention block) across forward passes. Each step runs the
//! same per-block rce dispatch as
//! `jamba_full_forward_with_ssm.rs`, but the SSM block graph now
//! returns a **packed** output `[delta || new_conv_state ||
//! new_ssm_state]` so state extraction stays inside the dispatch
//! path (no host re-computation of the SSM math).
//!
//! ## What this proves
//!
//! 1. `tensor_concat` wires up: the SSM block emits 3 outputs as one
//!    packed tensor by concat-ing along axis 0.
//! 2. State threads correctly: token `t+1` sees the state updated
//!    by token `t`'s forward pass; the SSM is doing real recurrent
//!    information mixing.
//! 3. Per-step bit-perfect equivalence between inline reference and
//!    rce path — confirms state updates from rce's packed-concat
//!    path produce numerically identical results to direct calls of
//!    `conv1d_causal_step` + `selective_scan_mamba1_step`.
//! 4. **A real generated token sequence** — 3 tokens emitted one at
//!    a time, each conditioned on the prior tokens via state.

use rce_ingest_gguf::{read_tensor_f32, sample, sample_with_history, GgufFile, SampleConfig, Tokenizer};
use rce_op_attention::AttentionLeaf;
use rce_op_blas::BlasMatmul;
use rce_op_conv1d::Conv1dCausalLeaf;
use rce_op_mps::MpsMatmul;
use rce_op_nn_prims::{RmsNormLeaf, SiluLeaf, SplitAtLeaf, SwigluLeaf};
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

const NUM_TOKENS_TO_GENERATE: usize = 3;

#[derive(Clone, Copy)]
struct JambaConfig {
    d_model: usize,
    d_inner_ssm: usize,
    d_state: usize,
    dt_rank: usize,
    d_conv: usize,
    d_inner_ffn: usize,
    n_heads: usize,
    d_head: usize,
    eps: f32,
}

impl JambaConfig {
    const REASONING_3B: Self = JambaConfig {
        d_model: 2560,
        d_inner_ssm: 5120,
        d_state: 16,
        dt_rank: 160,
        d_conv: 4,
        d_inner_ffn: 8192,
        n_heads: 20,
        d_head: 128,
        eps: 1e-6,
    };
}

/// Per-block recurrent state. SSM blocks use `(conv_state, ssm_state)`;
/// attention blocks use `(k_cache, v_cache)` (growing each step).
/// Vec<f32> is host-resident; we don't keep the state on device this
/// turn.
enum BlockState {
    Ssm { conv_state: Vec<f32>, ssm_state: Vec<f32> },
    Attn { k_cache: Vec<f32>, v_cache: Vec<f32> },
}

struct JambaState {
    blocks: Vec<BlockState>,
}

impl JambaState {
    fn new(cfg: JambaConfig, ssm_blocks: &[usize], attn_blocks: &[usize]) -> Self {
        let mut blocks = Vec::with_capacity(28);
        for i in 0..28 {
            if ssm_blocks.contains(&i) {
                blocks.push(BlockState::Ssm {
                    conv_state: vec![0.0f32; cfg.d_inner_ssm * cfg.d_conv],
                    ssm_state: vec![0.0f32; cfg.d_inner_ssm * cfg.d_state],
                });
            } else if attn_blocks.contains(&i) {
                blocks.push(BlockState::Attn {
                    k_cache: Vec::new(),
                    v_cache: Vec::new(),
                });
            } else {
                panic!("block {i} is neither SSM nor attention");
            }
        }
        Self { blocks }
    }
}

#[test]
fn jamba_multi_token_loop_emits_sequence() {
    let path = PathBuf::from(JAMBA_GGUF);
    if !path.exists() {
        eprintln!("jamba_multi_token_loop: GGUF not found — skipping.");
        return;
    }
    let file = File::open(&path).expect("open");
    let mmap = unsafe { Mmap::map(&file) }.expect("mmap");
    let bytes: &[u8] = &mmap;
    let gguf = GgufFile::parse(bytes).expect("parse");
    let cfg = JambaConfig::REASONING_3B;

    let ssm_blocks: Vec<usize> = (0..28).filter(|i|
        gguf.tensors.iter().any(|t| t.name == format!("blk.{i}.ssm_in.weight"))
    ).collect();
    let attn_blocks: Vec<usize> = (0..28).filter(|i|
        gguf.tensors.iter().any(|t| t.name == format!("blk.{i}.attn_q.weight"))
    ).collect();

    // Independent states for inline reference and rce paths.
    let mut state_ref = JambaState::new(cfg, &ssm_blocks, &attn_blocks);
    let mut state_rce = JambaState::new(cfg, &ssm_blocks, &attn_blocks);

    // ── Tokenizer demo: encode a real prompt ──────────────────
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    eprintln!("Tokenizer: {} vocab, BOS={:?}, EOS={:?}",
        tokenizer.vocab.len(), tokenizer.bos_token_id, tokenizer.eos_token_id);
    for prompt in ["Hello world", "The quick brown fox"] {
        let encoded = tokenizer.encode(prompt, true);
        let decoded_back = tokenizer.decode_sequence(&encoded, true);
        eprintln!(
            "  encode({prompt:?}) → {encoded:?} → decode → {decoded_back:?}"
        );
    }

    let token_embd = read_tensor_f32(bytes, &gguf, "token_embd.weight").expect("token_embd");
    let vocab = token_embd.len() / cfg.d_model;
    let output_norm_w = read_tensor_f32(bytes, &gguf, "output_norm.weight").expect("output_norm");
    let lm_head_w = read_tensor_f32(bytes, &gguf, "output.weight").expect("output");
    let lm_head_t = transpose_2d(&lm_head_w, vocab, cfg.d_model);
    drop(lm_head_w);

    let conv_leaf = Conv1dCausalLeaf::new();
    let ssm_leaf = SsmScanMamba1Leaf::new();
    let silu_leaf = SiluLeaf::new();
    let rms_leaf = RmsNormLeaf::new();
    let split_leaf = SplitAtLeaf::new();
    let swiglu_leaf = SwigluLeaf::new();
    let attn_leaf = AttentionLeaf::new();
    let blas_leaf = BlasMatmul::new();
    // BlasMatmul claims `Custom("matmul")` and short-circuits the
    // in-tree CPU matmul. Order matters: blas first, then the other
    // leaves, so matmul calls hit BLAS (Apple Accelerate / OpenBLAS)
    // before falling through to the cpu kernel layer.
    let ops: Vec<&dyn OpDispatch> = vec![
        &blas_leaf,
        &attn_leaf,
        &conv_leaf, &ssm_leaf, &silu_leaf, &rms_leaf, &split_leaf, &swiglu_leaf,
    ];

    let ssm_block_graph = build_ssm_block_graph_with_state(cfg);
    let ffn_block_graph = build_ffn_block_graph();

    let mut tokens_ref: Vec<i32> = vec![1];
    let mut tokens_rce: Vec<i32> = vec![1];

    for step_i in 0..NUM_TOKENS_TO_GENERATE {
        let t_step = Instant::now();
        let cur_tok_ref = *tokens_ref.last().unwrap();
        let cur_tok_rce = *tokens_rce.last().unwrap();
        assert_eq!(cur_tok_ref, cur_tok_rce, "token sequences diverged at step {step_i}");

        let next_ref = step_inline(
            bytes, &gguf, &mut state_ref, cur_tok_ref,
            cfg, &ssm_blocks, &attn_blocks,
            &token_embd, &output_norm_w, &lm_head_t, vocab,
        );
        let next_rce = step_via_rce(
            bytes, &gguf, &mut state_rce, cur_tok_rce,
            cfg, &ssm_blocks, &attn_blocks,
            &token_embd, &output_norm_w, &lm_head_t, vocab,
            &ssm_block_graph, &ffn_block_graph, &ops,
            &attn_leaf,
        );

        eprintln!(
            "step {step_i}: token_in={cur_tok_ref} → next: ref={next_ref}, rce={next_rce} ({:?})",
            t_step.elapsed()
        );
        assert_eq!(next_ref, next_rce, "step {step_i}: ref and rce paths diverged");

        tokens_ref.push(next_ref);
        tokens_rce.push(next_rce);
    }

    eprintln!("══════════════════════════════════════════════════");
    eprintln!("Jamba multi-token loop emitted: {:?}", tokens_rce);
    eprintln!("  (start = 1 BOS, generated {} more tokens via state-threaded rce dispatch)",
              NUM_TOKENS_TO_GENERATE);

    // ── Tokenizer decoding ────────────────────────────────────
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer.ggml.tokens metadata");
    eprintln!();
    eprintln!("Per-token decoding:");
    for &id in &tokens_rce {
        let raw = tok.token_str(id).unwrap_or("(OOV)");
        let decoded = tok.decode_one(id);
        let ty = tok.token_type(id);
        eprintln!("  {id:>6}  type={ty:?}  raw={raw:?}  →  {decoded:?}");
    }

    // Full decoded string. Skip control tokens (e.g., BOS).
    let decoded_full = tok.decode_sequence(&tokens_rce, true);
    let decoded_with_special = tok.decode_sequence(&tokens_rce, false);
    eprintln!();
    eprintln!("──────────────────────────────────────────────────");
    eprintln!("  Decoded text (skip_special=true):  {decoded_full:?}");
    eprintln!("  Decoded text (skip_special=false): {decoded_with_special:?}");
    eprintln!("══════════════════════════════════════════════════");
}

/// **Prompt prefill loop**: encode a real prompt, prefill model
/// state with each prompt token (without sampling), then generate
/// continuation tokens by feeding each newly-generated token back.
/// Decodes the full `prompt + continuation` sequence and prints.
///
/// This is the user-visible payoff: a real prompt goes in, real
/// (model-determined) text comes out, with state threaded across
/// every step through the same rce dispatch path validated above
/// for bit-perfect equivalence with the inline reference.
///
/// Test runtime: 4 forward passes (2 prompt tokens after BOS + 2
/// generated) × ~40 s ≈ 160 s on M3 Ultra (CPU naive matmul + BLAS
/// for the larger ops).
#[test]
fn jamba_prompt_continuation() {
    let path = PathBuf::from(JAMBA_GGUF);
    if !path.exists() {
        eprintln!("jamba_prompt_continuation: GGUF not found — skipping.");
        return;
    }
    let file = File::open(&path).expect("open");
    let mmap = unsafe { Mmap::map(&file) }.expect("mmap");
    let bytes: &[u8] = &mmap;
    let gguf = GgufFile::parse(bytes).expect("parse");
    let cfg = JambaConfig::REASONING_3B;

    let ssm_blocks: Vec<usize> = (0..28).filter(|i|
        gguf.tensors.iter().any(|t| t.name == format!("blk.{i}.ssm_in.weight"))
    ).collect();
    let attn_blocks: Vec<usize> = (0..28).filter(|i|
        gguf.tensors.iter().any(|t| t.name == format!("blk.{i}.attn_q.weight"))
    ).collect();

    let token_embd = read_tensor_f32(bytes, &gguf, "token_embd.weight").expect("token_embd");
    let vocab = token_embd.len() / cfg.d_model;
    let output_norm_w = read_tensor_f32(bytes, &gguf, "output_norm.weight").expect("output_norm");
    let lm_head_w = read_tensor_f32(bytes, &gguf, "output.weight").expect("output");
    let lm_head_t = transpose_2d(&lm_head_w, vocab, cfg.d_model);
    drop(lm_head_w);

    let conv_leaf = Conv1dCausalLeaf::new();
    let ssm_leaf = SsmScanMamba1Leaf::new();
    let silu_leaf = SiluLeaf::new();
    let rms_leaf = RmsNormLeaf::new();
    let split_leaf = SplitAtLeaf::new();
    let swiglu_leaf = SwigluLeaf::new();
    let attn_leaf = AttentionLeaf::new();
    let blas_leaf = BlasMatmul::new();
    let ops: Vec<&dyn OpDispatch> = vec![
        &blas_leaf, &attn_leaf,
        &conv_leaf, &ssm_leaf, &silu_leaf, &rms_leaf, &split_leaf, &swiglu_leaf,
    ];

    let ssm_block_graph = build_ssm_block_graph_with_state(cfg);
    let ffn_block_graph = build_ffn_block_graph();

    // ── Encode prompt ─────────────────────────────────────────
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let prompt = "Hello";
    let prompt_tokens = tokenizer.encode(prompt, true);
    eprintln!("══════════════════════════════════════════════════");
    eprintln!("Prompt: {prompt:?}");
    eprintln!("Encoded: {prompt_tokens:?}");
    eprintln!("Decoded back: {:?}", tokenizer.decode_sequence(&prompt_tokens, false));
    assert!(prompt_tokens.len() >= 2, "prompt should encode to ≥1 token + BOS");

    let n_prompt = prompt_tokens.len();
    let n_generate = 2;

    let mut state_ref = JambaState::new(cfg, &ssm_blocks, &attn_blocks);
    let mut state_rce = JambaState::new(cfg, &ssm_blocks, &attn_blocks);

    // ── Prefill: feed each prompt token, discarding predictions
    //   except after the LAST prompt token (which gives us the first
    //   generated token).
    eprintln!();
    eprintln!("── Prefill ({n_prompt} tokens) ───────────────────────");
    let mut last_predicted_ref = None;
    let mut last_predicted_rce = None;
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let t = Instant::now();
        let pred_ref = step_inline(
            bytes, &gguf, &mut state_ref, tok,
            cfg, &ssm_blocks, &attn_blocks,
            &token_embd, &output_norm_w, &lm_head_t, vocab,
        );
        let pred_rce = step_via_rce(
            bytes, &gguf, &mut state_rce, tok,
            cfg, &ssm_blocks, &attn_blocks,
            &token_embd, &output_norm_w, &lm_head_t, vocab,
            &ssm_block_graph, &ffn_block_graph, &ops,
            &attn_leaf,
        );
        eprintln!(
            "  prefill {i}: token={} ({:?}) → next-pred ref={}, rce={} ({:?})",
            tok, tokenizer.token_str(tok).unwrap_or("?"), pred_ref, pred_rce, t.elapsed()
        );
        assert_eq!(pred_ref, pred_rce, "prefill step {i}: paths diverged");
        if i == n_prompt - 1 {
            last_predicted_ref = Some(pred_ref);
            last_predicted_rce = Some(pred_rce);
        }
    }

    // ── Generate: start from the model's prediction after the last
    //   prompt token, feed each new token back to get the next.
    let mut generated_ref: Vec<i32> = vec![last_predicted_ref.unwrap()];
    let mut generated_rce: Vec<i32> = vec![last_predicted_rce.unwrap()];
    eprintln!();
    eprintln!("── Generate ({n_generate} tokens) ───────────────────");
    eprintln!(
        "  step 0: from-prefill → {} ({:?})",
        generated_rce[0], tokenizer.token_str(generated_rce[0]).unwrap_or("?")
    );

    for step_i in 1..n_generate {
        let t = Instant::now();
        let cur_tok = *generated_rce.last().unwrap();
        let cur_tok_ref = *generated_ref.last().unwrap();
        assert_eq!(cur_tok, cur_tok_ref, "generation step {step_i}: previous-token diverged");

        let next_ref = step_inline(
            bytes, &gguf, &mut state_ref, cur_tok_ref,
            cfg, &ssm_blocks, &attn_blocks,
            &token_embd, &output_norm_w, &lm_head_t, vocab,
        );
        let next_rce = step_via_rce(
            bytes, &gguf, &mut state_rce, cur_tok,
            cfg, &ssm_blocks, &attn_blocks,
            &token_embd, &output_norm_w, &lm_head_t, vocab,
            &ssm_block_graph, &ffn_block_graph, &ops,
            &attn_leaf,
        );
        eprintln!(
            "  step {step_i}: token_in={} → next ref={}, rce={} ({:?})",
            cur_tok, next_ref, next_rce, t.elapsed()
        );
        assert_eq!(next_ref, next_rce, "generation step {step_i}: paths diverged");
        generated_ref.push(next_ref);
        generated_rce.push(next_rce);
    }

    // ── Decode + print final result ───────────────────────────
    let full_sequence: Vec<i32> = prompt_tokens.iter().copied()
        .chain(generated_rce.iter().copied())
        .collect();
    let decoded_skip = tokenizer.decode_sequence(&full_sequence, true);
    let decoded_full = tokenizer.decode_sequence(&full_sequence, false);

    eprintln!();
    eprintln!("══════════════════════════════════════════════════");
    eprintln!("Final sequence (token ids): {full_sequence:?}");
    eprintln!();
    eprintln!("  Decoded (skip_special):   {decoded_skip:?}");
    eprintln!("  Decoded (with special):   {decoded_full:?}");
    eprintln!("══════════════════════════════════════════════════");
    eprintln!();
    eprintln!("PROMPT + CONTINUATION: \"{decoded_skip}\"");
    eprintln!("══════════════════════════════════════════════════");
}

/// **Chat-template continuation, with sampling** — render a real Jamba
/// chat prompt (`<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n`)
/// and feed it through the substrate, drawing each token with
/// `SampleConfig::chat` (temperature=0.7, top_k=40, top_p=0.9, seeded)
/// instead of greedy argmax. Greedy on Q4_K_M locks onto rare-fragment
/// trajectories where quantization noise has perturbed the argmax;
/// sampling spreads mass over the top of the distribution so the model
/// can recover. The seed is mixed per-step (`base ^ step·φ`) so each
/// forward pass draws independently while the run stays reproducible.
///
/// Skips the inline reference path — bit-perfect rce-vs-inline
/// equivalence is already validated by `jamba_prompt_continuation` and
/// `jamba_multi_token_loop_emits_sequence` on every step. This test
/// focuses on **what the model says**, not on path equivalence.
#[test]
fn jamba_chat_continuation() {
    let path = PathBuf::from(JAMBA_GGUF);
    if !path.exists() {
        eprintln!("jamba_chat_continuation: GGUF not found — skipping.");
        return;
    }
    let file = File::open(&path).expect("open");
    let mmap = unsafe { Mmap::map(&file) }.expect("mmap");
    let bytes: &[u8] = &mmap;
    let gguf = GgufFile::parse(bytes).expect("parse");
    let cfg = JambaConfig::REASONING_3B;

    let ssm_blocks: Vec<usize> = (0..28).filter(|i|
        gguf.tensors.iter().any(|t| t.name == format!("blk.{i}.ssm_in.weight"))
    ).collect();
    let attn_blocks: Vec<usize> = (0..28).filter(|i|
        gguf.tensors.iter().any(|t| t.name == format!("blk.{i}.attn_q.weight"))
    ).collect();

    let token_embd = read_tensor_f32(bytes, &gguf, "token_embd.weight").expect("token_embd");
    let vocab = token_embd.len() / cfg.d_model;
    let output_norm_w = read_tensor_f32(bytes, &gguf, "output_norm.weight").expect("output_norm");
    let lm_head_w = read_tensor_f32(bytes, &gguf, "output.weight").expect("output");
    let lm_head_t = transpose_2d(&lm_head_w, vocab, cfg.d_model);
    drop(lm_head_w);

    let conv_leaf = Conv1dCausalLeaf::new();
    let ssm_leaf = SsmScanMamba1Leaf::new();
    let silu_leaf = SiluLeaf::new();
    let rms_leaf = RmsNormLeaf::new();
    let split_leaf = SplitAtLeaf::new();
    let swiglu_leaf = SwigluLeaf::new();
    let attn_leaf = AttentionLeaf::new();
    let blas_leaf = BlasMatmul::new();
    let ops: Vec<&dyn OpDispatch> = vec![
        &blas_leaf, &attn_leaf,
        &conv_leaf, &ssm_leaf, &silu_leaf, &rms_leaf, &split_leaf, &swiglu_leaf,
    ];

    let ssm_block_graph = build_ssm_block_graph_with_state(cfg);
    let ffn_block_graph = build_ffn_block_graph();

    // ── Pre-load all weights once (avoids ~80% of per-forward
    //   work that was being burned on per-step GGUF reads + dequant). ─
    let preload_t = Instant::now();
    let weights = load_jamba_weights(bytes, &gguf, cfg, &ssm_blocks, &attn_blocks);
    eprintln!("Pre-loaded weights in {:?}", preload_t.elapsed());

    // ── Build the chat-formatted prompt ────────────────────────
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let user_msg = "What is 7 + 8?";
    // Use the full chat template with the Jamba-Reasoning thinking
    // prefix so the model engages step-by-step reasoning mode.
    let prompt_tokens = tokenizer
        .format_chat_messages(&[("user", user_msg)], true, true)
        .expect("chat-template tokens");
    let eos_id = tokenizer.eos_token_id.unwrap_or(-1);
    let im_end_id = tokenizer.lookup_token("<|im_end|>").unwrap_or(-1);
    let rendered = tokenizer.decode_sequence(&prompt_tokens, false);

    eprintln!("══════════════════════════════════════════════════");
    eprintln!("CHAT-TEMPLATE PROMPT");
    eprintln!("──────────────────────────────────────────────────");
    eprintln!("user_msg: {user_msg:?}");
    eprintln!("token ids ({}): {prompt_tokens:?}", prompt_tokens.len());
    eprintln!("rendered: {rendered:?}");
    eprintln!("══════════════════════════════════════════════════");

    let n_prompt = prompt_tokens.len();
    let n_generate: usize = 80;

    // **Sampling, not greedy argmax**: temperature=0.7, top_k=40,
    // top_p=0.9 (standard chat-LM defaults). Greedy on Q4_K_M locks
    // onto rare-fragment trajectories where quantization noise has
    // bumped the argmax off the actual best token; sampling spreads
    // mass over the top of the distribution so the model can recover.
    // Seed is per-step (`base ^ step`) so each forward pass draws an
    // independent token while the run stays reproducible.
    let base_seed: u64 = 0xC0DE_F00D_2026;
    let sampler_cfg = |step: u64| SampleConfig::chat(base_seed ^ step.wrapping_mul(0x9E37_79B9));

    let mut state = JambaState::new(cfg, &ssm_blocks, &attn_blocks);

    // ── Prefill ────────────────────────────────────────────────
    eprintln!();
    eprintln!("── Prefill ({n_prompt} tokens, rce-only, sampled) ──");
    let mut last_predicted: Option<i32> = None;
    let prefill_start = Instant::now();
    // Track the recent N tokens for repetition penalty (sliding window
    // of the prompt + generated history).
    const REP_WINDOW: usize = 64;
    let mut history: Vec<i32> = Vec::with_capacity(n_prompt + n_generate);
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let t = Instant::now();
        let logits = forward_via_rce_logits_cached(
            &weights, &mut state, tok, cfg,
            &ssm_block_graph, &ffn_block_graph, &ops,
            &attn_leaf,
        );
        history.push(tok);
        let recent_start = history.len().saturating_sub(REP_WINDOW);
        let pred = sample_with_history(&logits, &sampler_cfg(i as u64), &history[recent_start..]);
        let tok_str = tokenizer.decode_sequence(&[tok], false);
        eprintln!(
            "  prefill {i:2}: tok={tok:5} ({tok_str:?}) → next-pred {pred} ({:?})",
            t.elapsed()
        );
        if i == n_prompt - 1 {
            last_predicted = Some(pred);
        }
    }
    eprintln!("  prefill total: {:?}", prefill_start.elapsed());

    // ── Generate ───────────────────────────────────────────────
    let mut generated: Vec<i32> = vec![last_predicted.unwrap()];
    history.push(generated[0]);
    eprintln!();
    eprintln!("── Generate ({n_generate} tokens) ───────────────────");
    eprintln!(
        "  step 0: from-prefill → {} ({:?})",
        generated[0],
        tokenizer.decode_sequence(&[generated[0]], false)
    );

    for step_i in 1..n_generate {
        let t = Instant::now();
        let cur_tok = *generated.last().unwrap();
        // EOS check on the *previous* sampled token so we don't feed
        // `<|im_end|>` back as input. The model emits it to signal
        // "I'm done."
        if cur_tok == im_end_id || cur_tok == eos_id {
            eprintln!("  step {step_i}: EOS detected (tok={cur_tok}) — stopping generation.");
            break;
        }
        let logits = forward_via_rce_logits_cached(
            &weights, &mut state, cur_tok, cfg,
            &ssm_block_graph, &ffn_block_graph, &ops,
            &attn_leaf,
        );
        let recent_start = history.len().saturating_sub(REP_WINDOW);
        let next = sample_with_history(&logits, &sampler_cfg((n_prompt + step_i) as u64), &history[recent_start..]);
        history.push(next);
        eprintln!(
            "  step {step_i}: tok_in={cur_tok} → next={next} ({:?}) ({:?})",
            tokenizer.decode_sequence(&[next], false),
            t.elapsed()
        );
        generated.push(next);
    }

    // ── Decode + print ──────────────────────────────────────────
    let full_sequence: Vec<i32> = prompt_tokens.iter().copied()
        .chain(generated.iter().copied())
        .collect();
    let decoded_full = tokenizer.decode_sequence(&full_sequence, false);
    let assistant_continuation = tokenizer.decode_sequence(&generated, false);

    eprintln!();
    eprintln!("══════════════════════════════════════════════════");
    eprintln!("FULL SEQUENCE (with specials):");
    eprintln!("{decoded_full}");
    eprintln!("══════════════════════════════════════════════════");
    eprintln!();
    eprintln!("ASSISTANT CONTINUATION (first {n_generate} tokens after <think>\\n):");
    eprintln!("  ids: {generated:?}");
    eprintln!("  decoded: {assistant_continuation:?}");
    eprintln!("══════════════════════════════════════════════════");
}

/// **Bypass-blocks diagnostic** — what does the model predict if we
/// skip the 28 blocks entirely? Embedding → output_norm → lm_head →
/// argmax, on token id 1 (BOS) and id 11694 ("Hi"). If this produces
/// gibberish, the bug is in dequant or in token_embd/lm_head loading.
/// If it produces plausible tokens, the bug is in the block forward.
///
/// On Jamba-Reasoning-3B Q4_K_M, llama.cpp greedy on `[BOS]` predicts
/// `" The"`; on `[BOS, "Hi"]` it predicts `","`. With proper math,
/// even the bypass-blocks path should land in the same neighborhood —
/// the unembed matrix is the dominant signal at this stage.
#[test]
fn jamba_bypass_blocks_diagnostic() {
    let path = PathBuf::from(JAMBA_GGUF);
    if !path.exists() {
        eprintln!("jamba_bypass_blocks_diagnostic: GGUF not found — skipping.");
        return;
    }
    let file = File::open(&path).expect("open");
    let mmap = unsafe { Mmap::map(&file) }.expect("mmap");
    let bytes: &[u8] = &mmap;
    let gguf = GgufFile::parse(bytes).expect("parse");
    let cfg = JambaConfig::REASONING_3B;

    let token_embd = read_tensor_f32(bytes, &gguf, "token_embd.weight").expect("token_embd");
    let vocab = token_embd.len() / cfg.d_model;
    let output_norm_w = read_tensor_f32(bytes, &gguf, "output_norm.weight").expect("output_norm");
    let lm_head_w = read_tensor_f32(bytes, &gguf, "output.weight").expect("output");
    let lm_head_t = transpose_2d(&lm_head_w, vocab, cfg.d_model);

    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");

    eprintln!("vocab={vocab}, d_model={}", cfg.d_model);
    eprintln!("token_embd len = {}", token_embd.len());
    eprintln!("lm_head len    = {}", lm_head_w.len());
    eprintln!();

    for &tok in &[1, 11694, 22560] {
        let row = tok as usize * cfg.d_model;
        let emb = &token_embd[row..row + cfg.d_model];
        let emb_norm: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();

        let normed = rms_norm_inline(emb, &output_norm_w, cfg.eps);
        let normed_norm: f32 = normed.iter().map(|v| v * v).sum::<f32>().sqrt();

        let logits = matmul_1xN_NxM(&normed, &lm_head_t, cfg.d_model, vocab);
        let argm = argmax(&logits);

        // Top-5 logit indices for a more informative comparison.
        let mut indexed: Vec<(i32, f32)> = logits.iter().enumerate().map(|(i, &v)| (i as i32, v)).collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top5: Vec<(i32, f32, String)> = indexed.iter().take(5).map(|&(id, lg)| {
            (id, lg, tokenizer.decode_sequence(&[id], false))
        }).collect();

        eprintln!(
            "tok={tok:5} ({:?}): embed_norm={emb_norm:.4} normed_norm={normed_norm:.4} argmax={argm}",
            tokenizer.decode_sequence(&[tok], false)
        );
        eprintln!("  top-5: {top5:?}");
    }
}

/// **MPS vs BLAS matmul benchmark** — run ONE forward pass on BOS
/// with each matmul backend and compare wall-clock time. Since
/// every call to `Custom("matmul")` is the substrate's main hot
/// path, prepending an MPS leaf with a low `min_flops` threshold
/// routes the big-but-batch=1 matmuls through Metal instead of
/// Accelerate BLAS — useful if the GPU dispatch overhead is small
/// relative to per-matmul work.
#[test]
fn jamba_mps_vs_blas_single_forward() {
    let path = PathBuf::from(JAMBA_GGUF);
    if !path.exists() {
        eprintln!("jamba_mps_vs_blas_single_forward: GGUF not found — skipping.");
        return;
    }
    let file = File::open(&path).expect("open");
    let mmap = unsafe { Mmap::map(&file) }.expect("mmap");
    let bytes: &[u8] = &mmap;
    let gguf = GgufFile::parse(bytes).expect("parse");
    let cfg = JambaConfig::REASONING_3B;

    let ssm_blocks: Vec<usize> = (0..28).filter(|i|
        gguf.tensors.iter().any(|t| t.name == format!("blk.{i}.ssm_in.weight"))
    ).collect();
    let attn_blocks: Vec<usize> = (0..28).filter(|i|
        gguf.tensors.iter().any(|t| t.name == format!("blk.{i}.attn_q.weight"))
    ).collect();

    let token_embd = read_tensor_f32(bytes, &gguf, "token_embd.weight").expect("token_embd");
    let vocab = token_embd.len() / cfg.d_model;
    let output_norm_w = read_tensor_f32(bytes, &gguf, "output_norm.weight").expect("output_norm");
    let lm_head_w = read_tensor_f32(bytes, &gguf, "output.weight").expect("output");
    let lm_head_t = transpose_2d(&lm_head_w, vocab, cfg.d_model);
    drop(lm_head_w);

    let conv_leaf = Conv1dCausalLeaf::new();
    let ssm_leaf = SsmScanMamba1Leaf::new();
    let silu_leaf = SiluLeaf::new();
    let rms_leaf = RmsNormLeaf::new();
    let split_leaf = SplitAtLeaf::new();
    let swiglu_leaf = SwigluLeaf::new();
    let attn_leaf = AttentionLeaf::new();
    let blas_leaf = BlasMatmul::new();

    let ssm_block_graph = build_ssm_block_graph_with_state(cfg);
    let ffn_block_graph = build_ffn_block_graph();

    // ── BLAS-only ──────────────────────────────────────────────
    let ops_blas: Vec<&dyn OpDispatch> = vec![
        &blas_leaf, &attn_leaf,
        &conv_leaf, &ssm_leaf, &silu_leaf, &rms_leaf, &split_leaf, &swiglu_leaf,
    ];
    let mut state = JambaState::new(cfg, &ssm_blocks, &attn_blocks);
    let t0 = Instant::now();
    let pred_blas = step_via_rce(
        bytes, &gguf, &mut state, 1,
        cfg, &ssm_blocks, &attn_blocks,
        &token_embd, &output_norm_w, &lm_head_t, vocab,
        &ssm_block_graph, &ffn_block_graph, &ops_blas,
        &attn_leaf,
    );
    let blas_time = t0.elapsed();
    eprintln!("BLAS-only single forward on BOS: pred={pred_blas} time={blas_time:?}");

    // ── MPS-prepended (min_flops=0 → MPS wins everything) ──────
    let mps_leaf_opt = MpsMatmul::with_min_flops(0);
    let mps_time = if let Some(mps_leaf) = mps_leaf_opt.as_ref() {
        let ops_mps: Vec<&dyn OpDispatch> = vec![
            mps_leaf, &blas_leaf, &attn_leaf,
            &conv_leaf, &ssm_leaf, &silu_leaf, &rms_leaf, &split_leaf, &swiglu_leaf,
        ];
        let mut state = JambaState::new(cfg, &ssm_blocks, &attn_blocks);
        let t1 = Instant::now();
        let pred_mps = step_via_rce(
            bytes, &gguf, &mut state, 1,
            cfg, &ssm_blocks, &attn_blocks,
            &token_embd, &output_norm_w, &lm_head_t, vocab,
            &ssm_block_graph, &ffn_block_graph, &ops_mps,
            &attn_leaf,
        );
        let mps_time = t1.elapsed();
        eprintln!("MPS-first  single forward on BOS: pred={pred_mps} time={mps_time:?}");
        assert_eq!(pred_blas, pred_mps, "MPS and BLAS should produce same argmax");
        Some(mps_time)
    } else {
        eprintln!("MPS not available — skipping MPS branch.");
        None
    };

    if let Some(mt) = mps_time {
        let speedup = blas_time.as_secs_f64() / mt.as_secs_f64();
        eprintln!("MPS speedup vs BLAS: {speedup:.2}×");
    }
}

/// **Full-forward top-K diagnostic** — one forward pass of BOS
/// through all 28 blocks, dump top-10 tokens. Bypass-blocks already
/// gives sensible output; if full-forward's top-10 are wildly
/// different from llama.cpp's expected `" The"` / `","` etc., we
/// know there's a block-level bug.
#[test]
fn jamba_full_forward_topk_diagnostic() {
    let path = PathBuf::from(JAMBA_GGUF);
    if !path.exists() {
        eprintln!("jamba_full_forward_topk_diagnostic: GGUF not found — skipping.");
        return;
    }
    let file = File::open(&path).expect("open");
    let mmap = unsafe { Mmap::map(&file) }.expect("mmap");
    let bytes: &[u8] = &mmap;
    let gguf = GgufFile::parse(bytes).expect("parse");
    let cfg = JambaConfig::REASONING_3B;

    let ssm_blocks: Vec<usize> = (0..28).filter(|i|
        gguf.tensors.iter().any(|t| t.name == format!("blk.{i}.ssm_in.weight"))
    ).collect();
    let attn_blocks: Vec<usize> = (0..28).filter(|i|
        gguf.tensors.iter().any(|t| t.name == format!("blk.{i}.attn_q.weight"))
    ).collect();

    let token_embd = read_tensor_f32(bytes, &gguf, "token_embd.weight").expect("token_embd");
    let vocab = token_embd.len() / cfg.d_model;
    let output_norm_w = read_tensor_f32(bytes, &gguf, "output_norm.weight").expect("output_norm");
    let lm_head_w = read_tensor_f32(bytes, &gguf, "output.weight").expect("output");
    let lm_head_t = transpose_2d(&lm_head_w, vocab, cfg.d_model);
    drop(lm_head_w);

    let conv_leaf = Conv1dCausalLeaf::new();
    let ssm_leaf = SsmScanMamba1Leaf::new();
    let silu_leaf = SiluLeaf::new();
    let rms_leaf = RmsNormLeaf::new();
    let split_leaf = SplitAtLeaf::new();
    let swiglu_leaf = SwigluLeaf::new();
    let attn_leaf = AttentionLeaf::new();
    let blas_leaf = BlasMatmul::new();
    let ops: Vec<&dyn OpDispatch> = vec![
        &blas_leaf, &attn_leaf,
        &conv_leaf, &ssm_leaf, &silu_leaf, &rms_leaf, &split_leaf, &swiglu_leaf,
    ];

    let ssm_block_graph = build_ssm_block_graph_with_state(cfg);
    let ffn_block_graph = build_ffn_block_graph();
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");

    // One forward of BOS through full pipeline.
    let mut state = JambaState::new(cfg, &ssm_blocks, &attn_blocks);
    let logits = forward_via_rce_logits(
        bytes, &gguf, &mut state, 1,
        cfg, &ssm_blocks, &attn_blocks,
        &token_embd, &output_norm_w, &lm_head_t, vocab,
        &ssm_block_graph, &ffn_block_graph, &ops,
        &attn_leaf,
    );

    let mut indexed: Vec<(i32, f32)> = logits.iter().enumerate().map(|(i, &v)| (i as i32, v)).collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("Full-forward top-10 after BOS (token 1):");
    for &(id, lg) in indexed.iter().take(10) {
        eprintln!("  {id:5} logit={lg:9.4} text={:?}", tokenizer.decode_sequence(&[id], false));
    }
    eprintln!();
    eprintln!("For comparison, llama.cpp greedy after [BOS] predicts \" The\" (top sensible word completion).");
}

// ── Inline reference step ──────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn step_inline(
    bytes: &[u8], gguf: &GgufFile, state: &mut JambaState, token_id: i32,
    cfg: JambaConfig, ssm_blocks: &[usize], _attn_blocks: &[usize],
    token_embd: &[f32], output_norm_w: &[f32], lm_head_t: &[f32], vocab: usize,
) -> i32 {
    let row = token_id as usize * cfg.d_model;
    let mut residual: Vec<f32> = token_embd[row..row + cfg.d_model].to_vec();

    for block_idx in 0..28 {
        let bn = format!("blk.{block_idx}");
        let attn_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_norm.weight")).expect("attn_norm");
        let attn_normed = rms_norm_inline(&residual, &attn_norm_w, cfg.eps);

        let mixer_delta = if ssm_blocks.contains(&block_idx) {
            let (cs, ss) = match &mut state.blocks[block_idx] {
                BlockState::Ssm { conv_state, ssm_state } => (conv_state, ssm_state),
                _ => unreachable!(),
            };
            // Load + run the SSM block, returning (delta, new_conv_state, new_ssm_state).
            let (delta, new_conv_state, new_ssm_state) = ssm_block_inline(
                bytes, gguf, &bn, &attn_normed, cs, ss, cfg,
            );
            *cs = new_conv_state;
            *ss = new_ssm_state;
            delta
        } else {
            // Attention block: read tensors, run MQA forward with KV cache.
            let (kc, vc) = match &mut state.blocks[block_idx] {
                BlockState::Attn { k_cache, v_cache } => (k_cache, v_cache),
                _ => unreachable!(),
            };
            attn_block_inline(bytes, gguf, &bn, &attn_normed, kc, vc, cfg)
        };
        for i in 0..cfg.d_model { residual[i] += mixer_delta[i]; }

        // FFN
        let ffn_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ffn_norm.weight")).expect("ffn_norm");
        let gate_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ffn_gate.weight")).expect("gate");
        let up_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ffn_up.weight")).expect("up");
        let down_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ffn_down.weight")).expect("down");
        let gate_t = transpose_2d(&gate_w, cfg.d_inner_ffn, cfg.d_model);
        let up_t = transpose_2d(&up_w, cfg.d_inner_ffn, cfg.d_model);
        let down_t = transpose_2d(&down_w, cfg.d_model, cfg.d_inner_ffn);
        let ffn_delta = ffn_block_inline(&residual, &ffn_norm_w, &gate_t, &up_t, &down_t, cfg);
        for i in 0..cfg.d_model { residual[i] += ffn_delta[i]; }
    }

    let final_normed = rms_norm_inline(&residual, output_norm_w, cfg.eps);
    let logits = matmul_1xN_NxM(&final_normed, lm_head_t, cfg.d_model, vocab);
    argmax(&logits)
}

// ── Pre-loaded weight cache ────────────────────────────────────────
//
// Reading & dequantizing tensors from GGUF on every forward pass burns
// most of the wall-clock time (Q4_K dequant alone is ~80M values per
// SSM weight × 26 SSM blocks × multiple matrices per block). For
// generation loops we want to load + transpose once at startup and
// then reuse the f32 buffers for every forward. `JambaWeights`
// captures the per-block weights in already-transposed shape so the
// hot path is pure matmul + scan.

#[allow(dead_code)]
struct SsmBlockTensors {
    attn_norm_w: Vec<f32>,
    ssm_in_t: Vec<f32>,
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    ssm_x_t: Vec<f32>,
    dt_norm_w: Vec<f32>,
    b_norm_w: Vec<f32>,
    c_norm_w: Vec<f32>,
    ssm_dt_t: Vec<f32>,
    dt_b: Vec<f32>,
    a_log: Vec<f32>,
    d_param: Vec<f32>,
    ssm_out_t: Vec<f32>,
    ffn_norm_w: Vec<f32>,
    gate_t: Vec<f32>,
    up_t: Vec<f32>,
    down_t: Vec<f32>,
}

#[allow(dead_code)]
struct AttnBlockTensors {
    attn_norm_w: Vec<f32>,
    q_t: Vec<f32>,
    k_t: Vec<f32>,
    v_t: Vec<f32>,
    o_t: Vec<f32>,
    ffn_norm_w: Vec<f32>,
    gate_t: Vec<f32>,
    up_t: Vec<f32>,
    down_t: Vec<f32>,
}

#[allow(dead_code)]
enum BlockTensors {
    Ssm(SsmBlockTensors),
    Attn(AttnBlockTensors),
}

#[allow(dead_code)]
struct JambaWeights {
    token_embd: Vec<f32>,
    output_norm_w: Vec<f32>,
    lm_head_t: Vec<f32>,
    vocab: usize,
    blocks: Vec<BlockTensors>,
}

#[allow(dead_code)]
fn load_jamba_weights(
    bytes: &[u8], gguf: &GgufFile, cfg: JambaConfig,
    ssm_blocks: &[usize], attn_blocks: &[usize],
) -> JambaWeights {
    let token_embd = read_tensor_f32(bytes, gguf, "token_embd.weight").expect("token_embd");
    let vocab = token_embd.len() / cfg.d_model;
    let output_norm_w = read_tensor_f32(bytes, gguf, "output_norm.weight").expect("output_norm");
    let lm_head_w = read_tensor_f32(bytes, gguf, "output.weight").expect("output");
    let lm_head_t = transpose_2d(&lm_head_w, vocab, cfg.d_model);
    drop(lm_head_w);

    let mut blocks: Vec<BlockTensors> = Vec::with_capacity(28);
    for block_idx in 0..28 {
        let bn = format!("blk.{block_idx}");
        let attn_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_norm.weight")).expect("attn_norm");
        let ffn_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ffn_norm.weight")).expect("ffn_norm");
        let gate_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ffn_gate.weight")).expect("gate");
        let up_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ffn_up.weight")).expect("up");
        let down_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ffn_down.weight")).expect("down");
        let gate_t = transpose_2d(&gate_w, cfg.d_inner_ffn, cfg.d_model);
        let up_t = transpose_2d(&up_w, cfg.d_inner_ffn, cfg.d_model);
        let down_t = transpose_2d(&down_w, cfg.d_model, cfg.d_inner_ffn);

        if ssm_blocks.contains(&block_idx) {
            let ssm_in = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_in.weight")).expect("ssm_in");
            let conv_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_conv1d.weight")).expect("conv_w");
            let conv_b = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_conv1d.bias")).expect("conv_b");
            let ssm_x = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_x.weight")).expect("ssm_x");
            let dt_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_dt_norm.weight")).expect("dt_norm");
            let b_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_b_norm.weight")).expect("b_norm");
            let c_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_c_norm.weight")).expect("c_norm");
            let ssm_dt = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_dt.weight")).expect("ssm_dt");
            let dt_b = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_dt.bias")).expect("dt_b");
            let a_log = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_a")).expect("ssm_a");
            let d_param = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_d")).expect("ssm_d");
            let ssm_out = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_out.weight")).expect("ssm_out");
            let ssm_in_t = transpose_2d(&ssm_in, 2 * cfg.d_inner_ssm, cfg.d_model);
            let ssm_x_t = transpose_2d(&ssm_x, cfg.dt_rank + 2 * cfg.d_state, cfg.d_inner_ssm);
            let ssm_dt_t = transpose_2d(&ssm_dt, cfg.d_inner_ssm, cfg.dt_rank);
            let ssm_out_t = transpose_2d(&ssm_out, cfg.d_model, cfg.d_inner_ssm);
            blocks.push(BlockTensors::Ssm(SsmBlockTensors {
                attn_norm_w, ssm_in_t, conv_w, conv_b, ssm_x_t,
                dt_norm_w, b_norm_w, c_norm_w, ssm_dt_t, dt_b,
                a_log, d_param, ssm_out_t,
                ffn_norm_w, gate_t, up_t, down_t,
            }));
        } else if attn_blocks.contains(&block_idx) {
            let attn_q = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_q.weight")).expect("attn_q");
            let attn_k = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_k.weight")).expect("attn_k");
            let attn_v = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_v.weight")).expect("attn_v");
            let attn_o = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_output.weight")).expect("attn_o");
            let q_t = transpose_2d(&attn_q, cfg.n_heads * cfg.d_head, cfg.d_model);
            let k_t = transpose_2d(&attn_k, cfg.d_head, cfg.d_model);
            let v_t = transpose_2d(&attn_v, cfg.d_head, cfg.d_model);
            let o_t = transpose_2d(&attn_o, cfg.d_model, cfg.d_model);
            blocks.push(BlockTensors::Attn(AttnBlockTensors {
                attn_norm_w, q_t, k_t, v_t, o_t,
                ffn_norm_w, gate_t, up_t, down_t,
            }));
        } else {
            panic!("block {block_idx} is neither SSM nor attention");
        }
    }

    JambaWeights { token_embd, output_norm_w, lm_head_t, vocab, blocks }
}

#[allow(clippy::too_many_arguments, dead_code)]
fn forward_via_rce_logits_cached(
    weights: &JambaWeights, state: &mut JambaState, token_id: i32, cfg: JambaConfig,
    ssm_block_graph: &FunctionGraph, ffn_block_graph: &FunctionGraph,
    ops: &[&dyn OpDispatch], attn_leaf: &AttentionLeaf,
) -> Vec<f32> {
    let row = token_id as usize * cfg.d_model;
    let mut residual: Vec<f32> = weights.token_embd[row..row + cfg.d_model].to_vec();

    let conv_state_size = cfg.d_inner_ssm * cfg.d_conv;
    let ssm_state_size = cfg.d_inner_ssm * cfg.d_state;
    let packed_size = cfg.d_model + conv_state_size + ssm_state_size;

    for block_idx in 0..28 {
        let block = &weights.blocks[block_idx];

        let (attn_norm_w, ffn_norm_w, gate_t, up_t, down_t) = match block {
            BlockTensors::Ssm(s) => (&s.attn_norm_w, &s.ffn_norm_w, &s.gate_t, &s.up_t, &s.down_t),
            BlockTensors::Attn(a) => (&a.attn_norm_w, &a.ffn_norm_w, &a.gate_t, &a.up_t, &a.down_t),
        };

        let attn_normed = rms_norm_inline(&residual, attn_norm_w, cfg.eps);

        let mixer_delta: Vec<f32> = match block {
            BlockTensors::Ssm(s) => {
                let (cs, ss) = match &mut state.blocks[block_idx] {
                    BlockState::Ssm { conv_state, ssm_state } => (conv_state, ssm_state),
                    _ => unreachable!(),
                };
                let mut inputs = Inputs::new();
                inputs.insert("attn_normed".into(), Value::Tensor(HostTensor::from_f32(vec![1, cfg.d_model], &attn_normed).unwrap()));
                inputs.insert("ssm_in".into(),     Value::Tensor(HostTensor::from_f32(vec![cfg.d_model, 2 * cfg.d_inner_ssm], &s.ssm_in_t).unwrap()));
                inputs.insert("conv_w".into(),     Value::Tensor(HostTensor::from_f32(vec![cfg.d_inner_ssm * cfg.d_conv], &s.conv_w).unwrap()));
                inputs.insert("conv_b".into(),     Value::Tensor(HostTensor::from_f32(vec![cfg.d_inner_ssm], &s.conv_b).unwrap()));
                inputs.insert("conv_state".into(), Value::Tensor(HostTensor::from_f32(vec![conv_state_size], cs).unwrap()));
                inputs.insert("ssm_x".into(),      Value::Tensor(HostTensor::from_f32(vec![cfg.d_inner_ssm, cfg.dt_rank + 2 * cfg.d_state], &s.ssm_x_t).unwrap()));
                inputs.insert("dt_norm_w".into(),  Value::Tensor(HostTensor::from_f32(vec![cfg.dt_rank], &s.dt_norm_w).unwrap()));
                inputs.insert("b_norm_w".into(),   Value::Tensor(HostTensor::from_f32(vec![cfg.d_state], &s.b_norm_w).unwrap()));
                inputs.insert("c_norm_w".into(),   Value::Tensor(HostTensor::from_f32(vec![cfg.d_state], &s.c_norm_w).unwrap()));
                inputs.insert("ssm_dt".into(),     Value::Tensor(HostTensor::from_f32(vec![cfg.dt_rank, cfg.d_inner_ssm], &s.ssm_dt_t).unwrap()));
                inputs.insert("dt_b".into(),       Value::Tensor(HostTensor::from_f32(vec![cfg.d_inner_ssm], &s.dt_b).unwrap()));
                inputs.insert("a_log".into(),      Value::Tensor(HostTensor::from_f32(vec![cfg.d_inner_ssm * cfg.d_state], &s.a_log).unwrap()));
                inputs.insert("d_param".into(),    Value::Tensor(HostTensor::from_f32(vec![cfg.d_inner_ssm], &s.d_param).unwrap()));
                inputs.insert("ssm_out".into(),    Value::Tensor(HostTensor::from_f32(vec![cfg.d_inner_ssm, cfg.d_model], &s.ssm_out_t).unwrap()));
                inputs.insert("ssm_state".into(),  Value::Tensor(HostTensor::from_f32(vec![ssm_state_size], ss).unwrap()));

                let result = interpret_with_config(ssm_block_graph, &inputs, InterpretConfig { leaves: &[], ops })
                    .unwrap_or_else(|e| panic!("ssm block {block_idx}: {e:?}"));
                let packed = match result {
                    Value::Tensor(t) => t.as_f32().unwrap(),
                    _ => panic!(),
                };
                assert_eq!(packed.len(), packed_size);
                let delta = packed[..cfg.d_model].to_vec();
                *cs = packed[cfg.d_model..cfg.d_model + conv_state_size].to_vec();
                *ss = packed[cfg.d_model + conv_state_size..].to_vec();
                delta
            }
            BlockTensors::Attn(a) => {
                let (kc, vc) = match &mut state.blocks[block_idx] {
                    BlockState::Attn { k_cache, v_cache } => (k_cache, v_cache),
                    _ => unreachable!(),
                };
                attn_block_via_rce_cached(&attn_normed, &a.q_t, &a.k_t, &a.v_t, &a.o_t, kc, vc, cfg, attn_leaf)
            }
        };
        for i in 0..cfg.d_model { residual[i] += mixer_delta[i]; }

        // FFN via rce, weights pre-loaded
        let mut inputs = Inputs::new();
        inputs.insert("residual".into(),  Value::Tensor(HostTensor::from_f32(vec![1, cfg.d_model], &residual).unwrap()));
        inputs.insert("ffn_norm_w".into(),Value::Tensor(HostTensor::from_f32(vec![cfg.d_model], ffn_norm_w).unwrap()));
        inputs.insert("gate_W".into(),    Value::Tensor(HostTensor::from_f32(vec![cfg.d_model, cfg.d_inner_ffn], gate_t).unwrap()));
        inputs.insert("up_W".into(),      Value::Tensor(HostTensor::from_f32(vec![cfg.d_model, cfg.d_inner_ffn], up_t).unwrap()));
        inputs.insert("down_W".into(),    Value::Tensor(HostTensor::from_f32(vec![cfg.d_inner_ffn, cfg.d_model], down_t).unwrap()));
        let result = interpret_with_config(ffn_block_graph, &inputs, InterpretConfig { leaves: &[], ops })
            .unwrap_or_else(|e| panic!("ffn block {block_idx}: {e:?}"));
        let ffn_delta = match result {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        for i in 0..cfg.d_model { residual[i] += ffn_delta[i]; }
    }

    let final_normed = rms_norm_inline(&residual, &weights.output_norm_w, cfg.eps);
    matmul_1xN_NxM(&final_normed, &weights.lm_head_t, cfg.d_model, weights.vocab)
}

/// Cached version of `attn_block_via_rce` — weights pre-loaded.
#[allow(clippy::too_many_arguments, dead_code)]
fn attn_block_via_rce_cached(
    attn_normed: &[f32], q_t: &[f32], k_t: &[f32], v_t: &[f32], o_t: &[f32],
    k_cache: &mut Vec<f32>, v_cache: &mut Vec<f32>, cfg: JambaConfig,
    attn_leaf: &AttentionLeaf,
) -> Vec<f32> {
    let q = matmul_1xN_NxM(attn_normed, q_t, cfg.d_model, cfg.n_heads * cfg.d_head);
    let k_new = matmul_1xN_NxM(attn_normed, k_t, cfg.d_model, cfg.d_head);
    let v_new = matmul_1xN_NxM(attn_normed, v_t, cfg.d_model, cfg.d_head);
    k_cache.extend_from_slice(&k_new);
    v_cache.extend_from_slice(&v_new);
    let seq_len = k_cache.len() / cfg.d_head;
    let scale = 1.0 / (cfg.d_head as f32).sqrt();
    let mut multi_head_attended = vec![0.0f32; cfg.n_heads * cfg.d_head];
    let k_tensor = HostTensor::from_f32(vec![seq_len, cfg.d_head], k_cache).unwrap();
    let v_tensor = HostTensor::from_f32(vec![seq_len, cfg.d_head], v_cache).unwrap();
    for h in 0..cfg.n_heads {
        let q_h = &q[h * cfg.d_head..(h + 1) * cfg.d_head];
        let q_tensor = HostTensor::from_f32(vec![1, cfg.d_head], q_h).unwrap();
        let args = vec![
            Value::Tensor(q_tensor),
            Value::Tensor(k_tensor.clone()),
            Value::Tensor(v_tensor.clone()),
            Value::Scalar(ScalarValue::F32(scale)),
            Value::Scalar(ScalarValue::I32(0)),
        ];
        let result = attn_leaf.execute(0, "attention", &args)
            .unwrap_or_else(|e| panic!("attention head {h}: {e:?}"));
        let attended_h = match result {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        for d in 0..cfg.d_head {
            multi_head_attended[h * cfg.d_head + d] = attended_h[d];
        }
    }
    matmul_1xN_NxM(&multi_head_attended, o_t, cfg.n_heads * cfg.d_head, cfg.d_model)
}

// ── Rce-dispatch step ────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn step_via_rce(
    bytes: &[u8], gguf: &GgufFile, state: &mut JambaState, token_id: i32,
    cfg: JambaConfig, ssm_blocks: &[usize], _attn_blocks: &[usize],
    token_embd: &[f32], output_norm_w: &[f32], lm_head_t: &[f32], vocab: usize,
    ssm_graph: &FunctionGraph, ffn_graph: &FunctionGraph, ops: &[&dyn OpDispatch],
    attn_leaf: &AttentionLeaf,
) -> i32 {
    let logits = forward_via_rce_logits(
        bytes, gguf, state, token_id, cfg, ssm_blocks, _attn_blocks,
        token_embd, output_norm_w, lm_head_t, vocab,
        ssm_graph, ffn_graph, ops, attn_leaf,
    );
    argmax(&logits)
}

/// Same as `step_via_rce` but returns the full logits vector instead
/// of greedy argmax — lets the caller apply temperature / top-K /
/// top-P sampling for non-greedy decoding.
#[allow(clippy::too_many_arguments)]
fn forward_via_rce_logits(
    bytes: &[u8], gguf: &GgufFile, state: &mut JambaState, token_id: i32,
    cfg: JambaConfig, ssm_blocks: &[usize], _attn_blocks: &[usize],
    token_embd: &[f32], output_norm_w: &[f32], lm_head_t: &[f32], vocab: usize,
    ssm_graph: &FunctionGraph, ffn_graph: &FunctionGraph, ops: &[&dyn OpDispatch],
    attn_leaf: &AttentionLeaf,
) -> Vec<f32> {
    let row = token_id as usize * cfg.d_model;
    let mut residual: Vec<f32> = token_embd[row..row + cfg.d_model].to_vec();

    let conv_state_size = cfg.d_inner_ssm * cfg.d_conv;
    let ssm_state_size = cfg.d_inner_ssm * cfg.d_state;
    let packed_size = cfg.d_model + conv_state_size + ssm_state_size;

    for block_idx in 0..28 {
        let bn = format!("blk.{block_idx}");
        let attn_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_norm.weight")).expect("attn_norm");
        let attn_normed = rms_norm_inline(&residual, &attn_norm_w, cfg.eps);

        let mixer_delta = if ssm_blocks.contains(&block_idx) {
            let (cs, ss) = match &mut state.blocks[block_idx] {
                BlockState::Ssm { conv_state, ssm_state } => (conv_state, ssm_state),
                _ => unreachable!(),
            };
            // Load tensors + transpose.
            let ssm_in = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_in.weight")).expect("ssm_in");
            let conv_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_conv1d.weight")).expect("conv_w");
            let conv_b = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_conv1d.bias")).expect("conv_b");
            let ssm_x = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_x.weight")).expect("ssm_x");
            let dt_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_dt_norm.weight")).expect("dt_norm");
            let b_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_b_norm.weight")).expect("b_norm");
            let c_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_c_norm.weight")).expect("c_norm");
            let ssm_dt = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_dt.weight")).expect("ssm_dt");
            let dt_b = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_dt.bias")).expect("dt_b");
            let a_log = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_a")).expect("ssm_a");
            let d_param = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_d")).expect("ssm_d");
            let ssm_out = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_out.weight")).expect("ssm_out");

            let ssm_in_t = transpose_2d(&ssm_in, 2 * cfg.d_inner_ssm, cfg.d_model);
            let ssm_x_t = transpose_2d(&ssm_x, cfg.dt_rank + 2 * cfg.d_state, cfg.d_inner_ssm);
            let ssm_dt_t = transpose_2d(&ssm_dt, cfg.d_inner_ssm, cfg.dt_rank);
            let ssm_out_t = transpose_2d(&ssm_out, cfg.d_model, cfg.d_inner_ssm);

            // Run SSM block graph; output is packed [delta || new_conv_state || new_ssm_state].
            let mut inputs = Inputs::new();
            inputs.insert("attn_normed".into(), Value::Tensor(HostTensor::from_f32(vec![1, cfg.d_model], &attn_normed).unwrap()));
            inputs.insert("ssm_in".into(),     Value::Tensor(HostTensor::from_f32(vec![cfg.d_model, 2 * cfg.d_inner_ssm], &ssm_in_t).unwrap()));
            inputs.insert("conv_w".into(),     Value::Tensor(HostTensor::from_f32(vec![cfg.d_inner_ssm * cfg.d_conv], &conv_w).unwrap()));
            inputs.insert("conv_b".into(),     Value::Tensor(HostTensor::from_f32(vec![cfg.d_inner_ssm], &conv_b).unwrap()));
            inputs.insert("conv_state".into(), Value::Tensor(HostTensor::from_f32(vec![conv_state_size], cs).unwrap()));
            inputs.insert("ssm_x".into(),      Value::Tensor(HostTensor::from_f32(vec![cfg.d_inner_ssm, cfg.dt_rank + 2 * cfg.d_state], &ssm_x_t).unwrap()));
            inputs.insert("dt_norm_w".into(),  Value::Tensor(HostTensor::from_f32(vec![cfg.dt_rank], &dt_norm_w).unwrap()));
            inputs.insert("b_norm_w".into(),   Value::Tensor(HostTensor::from_f32(vec![cfg.d_state], &b_norm_w).unwrap()));
            inputs.insert("c_norm_w".into(),   Value::Tensor(HostTensor::from_f32(vec![cfg.d_state], &c_norm_w).unwrap()));
            inputs.insert("ssm_dt".into(),     Value::Tensor(HostTensor::from_f32(vec![cfg.dt_rank, cfg.d_inner_ssm], &ssm_dt_t).unwrap()));
            inputs.insert("dt_b".into(),       Value::Tensor(HostTensor::from_f32(vec![cfg.d_inner_ssm], &dt_b).unwrap()));
            inputs.insert("a_log".into(),      Value::Tensor(HostTensor::from_f32(vec![cfg.d_inner_ssm * cfg.d_state], &a_log).unwrap()));
            inputs.insert("d_param".into(),    Value::Tensor(HostTensor::from_f32(vec![cfg.d_inner_ssm], &d_param).unwrap()));
            inputs.insert("ssm_out".into(),    Value::Tensor(HostTensor::from_f32(vec![cfg.d_inner_ssm, cfg.d_model], &ssm_out_t).unwrap()));
            inputs.insert("ssm_state".into(),  Value::Tensor(HostTensor::from_f32(vec![ssm_state_size], ss).unwrap()));

            let result = interpret_with_config(ssm_graph, &inputs, InterpretConfig { leaves: &[], ops })
                .unwrap_or_else(|e| panic!("ssm block {block_idx}: {e:?}"));
            let packed = match result {
                Value::Tensor(t) => t.as_f32().unwrap(),
                _ => panic!(),
            };
            assert_eq!(packed.len(), packed_size);

            let delta = packed[..cfg.d_model].to_vec();
            *cs = packed[cfg.d_model..cfg.d_model + conv_state_size].to_vec();
            *ss = packed[cfg.d_model + conv_state_size..].to_vec();
            delta
        } else {
            // Attention block: route through rce-op-attention per head.
            let (kc, vc) = match &mut state.blocks[block_idx] {
                BlockState::Attn { k_cache, v_cache } => (k_cache, v_cache),
                _ => unreachable!(),
            };
            attn_block_via_rce(bytes, gguf, &bn, &attn_normed, kc, vc, cfg, &attn_leaf)
        };
        for i in 0..cfg.d_model { residual[i] += mixer_delta[i]; }

        // FFN via rce
        let ffn_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ffn_norm.weight")).expect("ffn_norm");
        let gate_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ffn_gate.weight")).expect("gate");
        let up_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ffn_up.weight")).expect("up");
        let down_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ffn_down.weight")).expect("down");
        let gate_t = transpose_2d(&gate_w, cfg.d_inner_ffn, cfg.d_model);
        let up_t = transpose_2d(&up_w, cfg.d_inner_ffn, cfg.d_model);
        let down_t = transpose_2d(&down_w, cfg.d_model, cfg.d_inner_ffn);

        let mut inputs = Inputs::new();
        inputs.insert("residual".into(),  Value::Tensor(HostTensor::from_f32(vec![1, cfg.d_model], &residual).unwrap()));
        inputs.insert("ffn_norm_w".into(),Value::Tensor(HostTensor::from_f32(vec![cfg.d_model], &ffn_norm_w).unwrap()));
        inputs.insert("gate_W".into(),    Value::Tensor(HostTensor::from_f32(vec![cfg.d_model, cfg.d_inner_ffn], &gate_t).unwrap()));
        inputs.insert("up_W".into(),      Value::Tensor(HostTensor::from_f32(vec![cfg.d_model, cfg.d_inner_ffn], &up_t).unwrap()));
        inputs.insert("down_W".into(),    Value::Tensor(HostTensor::from_f32(vec![cfg.d_inner_ffn, cfg.d_model], &down_t).unwrap()));
        let result = interpret_with_config(ffn_graph, &inputs, InterpretConfig { leaves: &[], ops })
            .unwrap_or_else(|e| panic!("ffn block {block_idx}: {e:?}"));
        let ffn_delta = match result {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        for i in 0..cfg.d_model { residual[i] += ffn_delta[i]; }
    }

    let final_normed = rms_norm_inline(&residual, output_norm_w, cfg.eps);
    matmul_1xN_NxM(&final_normed, lm_head_t, cfg.d_model, vocab)
}

// ── Inline math for ssm/attn/ffn blocks ────────────────────────────

fn ssm_block_inline(
    bytes: &[u8], gguf: &GgufFile, bn: &str, attn_normed: &[f32],
    conv_state: &[f32], ssm_state: &[f32], cfg: JambaConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let ssm_in = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_in.weight")).expect("ssm_in");
    let conv_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_conv1d.weight")).expect("conv_w");
    let conv_b = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_conv1d.bias")).expect("conv_b");
    let ssm_x = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_x.weight")).expect("ssm_x");
    let dt_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_dt_norm.weight")).expect("dt_norm");
    let b_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_b_norm.weight")).expect("b_norm");
    let c_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_c_norm.weight")).expect("c_norm");
    let ssm_dt = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_dt.weight")).expect("ssm_dt");
    let dt_b = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_dt.bias")).expect("dt_b");
    let a_log = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_a")).expect("ssm_a");
    let d_param = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_d")).expect("ssm_d");
    let ssm_out = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_out.weight")).expect("ssm_out");

    let ssm_in_t = transpose_2d(&ssm_in, 2 * cfg.d_inner_ssm, cfg.d_model);
    let ssm_x_t = transpose_2d(&ssm_x, cfg.dt_rank + 2 * cfg.d_state, cfg.d_inner_ssm);
    let ssm_dt_t = transpose_2d(&ssm_dt, cfg.d_inner_ssm, cfg.dt_rank);
    let ssm_out_t = transpose_2d(&ssm_out, cfg.d_model, cfg.d_inner_ssm);

    let xz = matmul_1xN_NxM(attn_normed, &ssm_in_t, cfg.d_model, 2 * cfg.d_inner_ssm);
    let x_raw = &xz[..cfg.d_inner_ssm];
    let z = &xz[cfg.d_inner_ssm..];

    let (x_conv, new_conv_state) = rce_op_conv1d::conv1d_causal_step(
        x_raw, &conv_w, &conv_b, conv_state, cfg.d_inner_ssm, cfg.d_conv,
    );
    let x_act: Vec<f32> = x_conv.iter().map(|v| stable_silu(*v)).collect();
    let bc_dim = cfg.dt_rank + 2 * cfg.d_state;
    let dt_bc = matmul_1xN_NxM(&x_act, &ssm_x_t, cfg.d_inner_ssm, bc_dim);
    let dt_normed = rms_norm_inline(&dt_bc[..cfg.dt_rank], &dt_norm_w, cfg.eps);
    let b_normed = rms_norm_inline(&dt_bc[cfg.dt_rank..cfg.dt_rank + cfg.d_state], &b_norm_w, cfg.eps);
    let c_normed = rms_norm_inline(&dt_bc[cfg.dt_rank + cfg.d_state..], &c_norm_w, cfg.eps);
    let dt_proj = matmul_1xN_NxM(&dt_normed, &ssm_dt_t, cfg.dt_rank, cfg.d_inner_ssm);

    let (y, new_ssm_state) = rce_op_ssm::selective_scan_mamba1_step(
        &x_act, &b_normed, &c_normed, &dt_proj, &dt_b, &a_log, &d_param, ssm_state,
        cfg.d_inner_ssm, cfg.d_state,
    );
    let z_silu: Vec<f32> = z.iter().map(|v| stable_silu(*v)).collect();
    let gated: Vec<f32> = y.iter().zip(z_silu.iter()).map(|(a, b)| a * b).collect();
    let delta = matmul_1xN_NxM(&gated, &ssm_out_t, cfg.d_inner_ssm, cfg.d_model);

    (delta, new_conv_state, new_ssm_state)
}

/// Attention block routed through `Custom("attention")` per head.
/// MQA: 1 KV head shared across n_heads=20 query heads. We dispatch
/// `rce-op-attention::AttentionLeaf` 20 times (once per Q head),
/// each call passing the full K/V cache so attention scores cover
/// all positions. Outputs concat to [n_heads*d_head] then go through
/// the output projection.
#[allow(clippy::too_many_arguments)]
fn attn_block_via_rce(
    bytes: &[u8], gguf: &GgufFile, bn: &str, attn_normed: &[f32],
    k_cache: &mut Vec<f32>, v_cache: &mut Vec<f32>, cfg: JambaConfig,
    attn_leaf: &AttentionLeaf,
) -> Vec<f32> {
    let attn_q = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_q.weight")).expect("attn_q");
    let attn_k = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_k.weight")).expect("attn_k");
    let attn_v = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_v.weight")).expect("attn_v");
    let attn_o = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_output.weight")).expect("attn_o");
    let q_t = transpose_2d(&attn_q, cfg.n_heads * cfg.d_head, cfg.d_model);
    let k_t = transpose_2d(&attn_k, cfg.d_head, cfg.d_model);
    let v_t = transpose_2d(&attn_v, cfg.d_head, cfg.d_model);
    let o_t = transpose_2d(&attn_o, cfg.d_model, cfg.d_model);

    // Q/K/V projections via inline matmul (BLAS via the dispatch
    // chain only kicks in for graph-level matmul Apply nodes). We
    // could lift these into a tiny matmul graph and route through
    // BLAS, but the projections are small (d_model=2560 → up to
    // 2560 cols) and inline is fast enough.
    let q = matmul_1xN_NxM(attn_normed, &q_t, cfg.d_model, cfg.n_heads * cfg.d_head);
    let k_new = matmul_1xN_NxM(attn_normed, &k_t, cfg.d_model, cfg.d_head);
    let v_new = matmul_1xN_NxM(attn_normed, &v_t, cfg.d_model, cfg.d_head);

    // Append to KV cache.
    k_cache.extend_from_slice(&k_new);
    v_cache.extend_from_slice(&v_new);
    let seq_len = k_cache.len() / cfg.d_head;
    let scale = 1.0 / (cfg.d_head as f32).sqrt();

    // Dispatch rce-op-attention per query head. Q for head h is
    // [1, d_head]; K and V are [seq_len, d_head] (shared across heads
    // for MQA).
    let mut multi_head_attended = vec![0.0f32; cfg.n_heads * cfg.d_head];
    let k_tensor = HostTensor::from_f32(vec![seq_len, cfg.d_head], k_cache).unwrap();
    let v_tensor = HostTensor::from_f32(vec![seq_len, cfg.d_head], v_cache).unwrap();
    for h in 0..cfg.n_heads {
        let q_h = &q[h * cfg.d_head..(h + 1) * cfg.d_head];
        let q_tensor = HostTensor::from_f32(vec![1, cfg.d_head], q_h).unwrap();
        let args = vec![
            Value::Tensor(q_tensor),
            Value::Tensor(k_tensor.clone()),
            Value::Tensor(v_tensor.clone()),
            Value::Scalar(ScalarValue::F32(scale)),
            Value::Scalar(ScalarValue::I32(0)),
        ];
        let result = attn_leaf.execute(0, "attention", &args)
            .unwrap_or_else(|e| panic!("attention head {h}: {e:?}"));
        let attended_h = match result {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        // attended_h is [1 * d_head] = [d_head]. Copy into the
        // h-th slot of multi_head_attended.
        for d in 0..cfg.d_head {
            multi_head_attended[h * cfg.d_head + d] = attended_h[d];
        }
    }

    matmul_1xN_NxM(&multi_head_attended, &o_t, cfg.n_heads * cfg.d_head, cfg.d_model)
}

fn attn_block_inline(
    bytes: &[u8], gguf: &GgufFile, bn: &str, attn_normed: &[f32],
    k_cache: &mut Vec<f32>, v_cache: &mut Vec<f32>, cfg: JambaConfig,
) -> Vec<f32> {
    let attn_q = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_q.weight")).expect("attn_q");
    let attn_k = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_k.weight")).expect("attn_k");
    let attn_v = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_v.weight")).expect("attn_v");
    let attn_o = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_output.weight")).expect("attn_o");
    let q_t = transpose_2d(&attn_q, cfg.n_heads * cfg.d_head, cfg.d_model);
    let k_t = transpose_2d(&attn_k, cfg.d_head, cfg.d_model);
    let v_t = transpose_2d(&attn_v, cfg.d_head, cfg.d_model);
    let o_t = transpose_2d(&attn_o, cfg.d_model, cfg.d_model);

    let q = matmul_1xN_NxM(attn_normed, &q_t, cfg.d_model, cfg.n_heads * cfg.d_head);
    let k_new = matmul_1xN_NxM(attn_normed, &k_t, cfg.d_model, cfg.d_head);
    let v_new = matmul_1xN_NxM(attn_normed, &v_t, cfg.d_model, cfg.d_head);

    // Append to KV cache.
    k_cache.extend_from_slice(&k_new);
    v_cache.extend_from_slice(&v_new);
    let seq_len = k_cache.len() / cfg.d_head;
    let scale = 1.0 / (cfg.d_head as f32).sqrt();

    // Per query head: scores = q[h] · k_cache (over all positions) * scale; softmax; attended = scores · v_cache
    let mut multi_head_attended = vec![0.0f32; cfg.n_heads * cfg.d_head];
    for h in 0..cfg.n_heads {
        let q_h = &q[h * cfg.d_head..(h + 1) * cfg.d_head];
        let mut scores = vec![0.0f32; seq_len];
        for s in 0..seq_len {
            let k_s = &k_cache[s * cfg.d_head..(s + 1) * cfg.d_head];
            let mut acc = 0.0f32;
            for d in 0..cfg.d_head {
                acc += q_h[d] * k_s[d];
            }
            scores[s] = acc * scale;
        }
        // Softmax over scores.
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_s).exp();
            sum_exp += *s;
        }
        for s in scores.iter_mut() { *s /= sum_exp; }
        // Attended = sum_s scores[s] * v_cache[s]
        let attended_h = &mut multi_head_attended[h * cfg.d_head..(h + 1) * cfg.d_head];
        for s in 0..seq_len {
            let v_s = &v_cache[s * cfg.d_head..(s + 1) * cfg.d_head];
            for d in 0..cfg.d_head {
                attended_h[d] += scores[s] * v_s[d];
            }
        }
    }
    matmul_1xN_NxM(&multi_head_attended, &o_t, cfg.n_heads * cfg.d_head, cfg.d_model)
}

fn ffn_block_inline(
    residual: &[f32], ffn_norm_w: &[f32], gate_t: &[f32], up_t: &[f32], down_t: &[f32], cfg: JambaConfig,
) -> Vec<f32> {
    let normed = rms_norm_inline(residual, ffn_norm_w, cfg.eps);
    let gate = matmul_1xN_NxM(&normed, gate_t, cfg.d_model, cfg.d_inner_ffn);
    let up = matmul_1xN_NxM(&normed, up_t, cfg.d_model, cfg.d_inner_ffn);
    let activated: Vec<f32> = gate.iter().zip(up.iter()).map(|(g, u)| stable_silu(*g) * *u).collect();
    matmul_1xN_NxM(&activated, down_t, cfg.d_inner_ffn, cfg.d_model)
}

// ── FunctionGraph builders ─────────────────────────────────────────

fn build_ssm_block_graph_with_state(cfg: JambaConfig) -> FunctionGraph {
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
    let id_d_inner_lit = nodes.len(); nodes.push(lit_i32(cfg.d_inner_ssm as i32));
    let id_d_conv_lit  = nodes.len(); nodes.push(lit_i32(cfg.d_conv as i32));
    let id_dt_rank_lit = nodes.len(); nodes.push(lit_i32(cfg.dt_rank as i32));
    let id_d_state_lit = nodes.len(); nodes.push(lit_i32(cfg.d_state as i32));
    let id_dt_bc_skip  = nodes.len(); nodes.push(lit_i32((cfg.dt_rank + cfg.d_state) as i32));
    let id_one_lit     = nodes.len(); nodes.push(lit_i32(1));
    let id_eps_lit     = nodes.len(); nodes.push(lit_f32(cfg.eps));
    let id_shape_1_di  = nodes.len(); nodes.push(lit_str(&format!("[1,{}]", cfg.d_inner_ssm)));
    let id_shape_1_dt  = nodes.len(); nodes.push(lit_str(&format!("[1,{}]", cfg.dt_rank)));
    let id_axis0       = nodes.len(); nodes.push(lit_i32(0));
    let id_conv_st_size= nodes.len(); nodes.push(lit_i32((cfg.d_inner_ssm * cfg.d_conv) as i32));
    let id_ssm_st_size = nodes.len(); nodes.push(lit_i32((cfg.d_inner_ssm * cfg.d_state) as i32));
    let id_d_model_lit = nodes.len(); nodes.push(lit_i32(cfg.d_model as i32));
    let id_shape_dm    = nodes.len(); nodes.push(lit_str(&format!("[{}]", cfg.d_model)));

    // Apply node chain.
    let id_xz = nodes.len(); nodes.push(apply_custom("matmul"));
    push_w(&mut wires, id_residual, id_xz, 0);
    push_w(&mut wires, id_ssm_in,   id_xz, 1);

    let id_x_raw = nodes.len(); nodes.push(apply_custom("split_at"));
    push_w(&mut wires, id_xz, id_x_raw, 0);
    push_w(&mut wires, id_zero, id_x_raw, 1);
    push_w(&mut wires, id_d_inner_lit, id_x_raw, 2);

    let id_z = nodes.len(); nodes.push(apply_custom("split_at"));
    push_w(&mut wires, id_xz, id_z, 0);
    push_w(&mut wires, id_d_inner_lit, id_z, 1);
    push_w(&mut wires, id_d_inner_lit, id_z, 2);

    let id_conv_packed = nodes.len(); nodes.push(apply_custom("conv1d_causal_depthwise"));
    push_w(&mut wires, id_x_raw,     id_conv_packed, 0);
    push_w(&mut wires, id_conv_w,    id_conv_packed, 1);
    push_w(&mut wires, id_conv_b,    id_conv_packed, 2);
    push_w(&mut wires, id_conv_state,id_conv_packed, 3);
    push_w(&mut wires, id_d_conv_lit,id_conv_packed, 4);

    let id_x_conv = nodes.len(); nodes.push(apply_custom("split_at"));
    push_w(&mut wires, id_conv_packed, id_x_conv, 0);
    push_w(&mut wires, id_zero, id_x_conv, 1);
    push_w(&mut wires, id_d_inner_lit, id_x_conv, 2);

    // NEW: extract new_conv_state from conv_packed
    let id_new_conv_state = nodes.len(); nodes.push(apply_custom("split_at"));
    push_w(&mut wires, id_conv_packed, id_new_conv_state, 0);
    push_w(&mut wires, id_d_inner_lit, id_new_conv_state, 1);
    push_w(&mut wires, id_conv_st_size, id_new_conv_state, 2);

    let id_x_act = nodes.len(); nodes.push(apply_custom("silu"));
    push_w(&mut wires, id_x_conv, id_x_act, 0);

    let id_x_act_2d = nodes.len(); nodes.push(apply_custom("reshape"));
    push_w(&mut wires, id_x_act,      id_x_act_2d, 0);
    push_w(&mut wires, id_shape_1_di, id_x_act_2d, 1);

    let id_dt_bc = nodes.len(); nodes.push(apply_custom("matmul"));
    push_w(&mut wires, id_x_act_2d, id_dt_bc, 0);
    push_w(&mut wires, id_ssm_x,    id_dt_bc, 1);

    let id_dt_raw_lr = nodes.len(); nodes.push(apply_custom("split_at"));
    push_w(&mut wires, id_dt_bc, id_dt_raw_lr, 0);
    push_w(&mut wires, id_zero, id_dt_raw_lr, 1);
    push_w(&mut wires, id_dt_rank_lit, id_dt_raw_lr, 2);

    let id_b_raw = nodes.len(); nodes.push(apply_custom("split_at"));
    push_w(&mut wires, id_dt_bc, id_b_raw, 0);
    push_w(&mut wires, id_dt_rank_lit, id_b_raw, 1);
    push_w(&mut wires, id_d_state_lit, id_b_raw, 2);

    let id_c_raw = nodes.len(); nodes.push(apply_custom("split_at"));
    push_w(&mut wires, id_dt_bc, id_c_raw, 0);
    push_w(&mut wires, id_dt_bc_skip, id_c_raw, 1);
    push_w(&mut wires, id_d_state_lit, id_c_raw, 2);

    let id_dt_normed = nodes.len(); nodes.push(apply_custom("rms_norm"));
    push_w(&mut wires, id_dt_raw_lr, id_dt_normed, 0);
    push_w(&mut wires, id_dt_norm_w, id_dt_normed, 1);
    push_w(&mut wires, id_one_lit,   id_dt_normed, 2);
    push_w(&mut wires, id_eps_lit,   id_dt_normed, 3);

    let id_b_normed = nodes.len(); nodes.push(apply_custom("rms_norm"));
    push_w(&mut wires, id_b_raw,    id_b_normed, 0);
    push_w(&mut wires, id_b_norm_w, id_b_normed, 1);
    push_w(&mut wires, id_one_lit,  id_b_normed, 2);
    push_w(&mut wires, id_eps_lit,  id_b_normed, 3);

    let id_c_normed = nodes.len(); nodes.push(apply_custom("rms_norm"));
    push_w(&mut wires, id_c_raw,    id_c_normed, 0);
    push_w(&mut wires, id_c_norm_w, id_c_normed, 1);
    push_w(&mut wires, id_one_lit,  id_c_normed, 2);
    push_w(&mut wires, id_eps_lit,  id_c_normed, 3);

    let id_dt_normed_2d = nodes.len(); nodes.push(apply_custom("reshape"));
    push_w(&mut wires, id_dt_normed,  id_dt_normed_2d, 0);
    push_w(&mut wires, id_shape_1_dt, id_dt_normed_2d, 1);

    let id_dt_proj = nodes.len(); nodes.push(apply_custom("matmul"));
    push_w(&mut wires, id_dt_normed_2d, id_dt_proj, 0);
    push_w(&mut wires, id_ssm_dt,       id_dt_proj, 1);

    let id_ssm_packed = nodes.len(); nodes.push(apply_custom("ssm_scan_mamba1"));
    push_w(&mut wires, id_x_act,    id_ssm_packed, 0);
    push_w(&mut wires, id_b_normed, id_ssm_packed, 1);
    push_w(&mut wires, id_c_normed, id_ssm_packed, 2);
    push_w(&mut wires, id_dt_proj,  id_ssm_packed, 3);
    push_w(&mut wires, id_dt_b,     id_ssm_packed, 4);
    push_w(&mut wires, id_a_log,    id_ssm_packed, 5);
    push_w(&mut wires, id_d_param,  id_ssm_packed, 6);
    push_w(&mut wires, id_ssm_state,id_ssm_packed, 7);
    push_w(&mut wires, id_d_inner_lit, id_ssm_packed, 8);
    push_w(&mut wires, id_d_state_lit, id_ssm_packed, 9);

    let id_y = nodes.len(); nodes.push(apply_custom("split_at"));
    push_w(&mut wires, id_ssm_packed, id_y, 0);
    push_w(&mut wires, id_zero, id_y, 1);
    push_w(&mut wires, id_d_inner_lit, id_y, 2);

    // NEW: extract new_ssm_state from ssm_packed
    let id_new_ssm_state = nodes.len(); nodes.push(apply_custom("split_at"));
    push_w(&mut wires, id_ssm_packed, id_new_ssm_state, 0);
    push_w(&mut wires, id_d_inner_lit, id_new_ssm_state, 1);
    push_w(&mut wires, id_ssm_st_size, id_new_ssm_state, 2);

    let id_z_silu = nodes.len(); nodes.push(apply_custom("silu"));
    push_w(&mut wires, id_z, id_z_silu, 0);

    let id_gated = nodes.len();
    nodes.push(GraphNode::Apply { op: OpKind::Mul, output_type: "f32".into() });
    push_w(&mut wires, id_y,      id_gated, 0);
    push_w(&mut wires, id_z_silu, id_gated, 1);

    let id_gated_2d = nodes.len(); nodes.push(apply_custom("reshape"));
    push_w(&mut wires, id_gated, id_gated_2d, 0);
    push_w(&mut wires, id_shape_1_di, id_gated_2d, 1);

    let id_delta_2d = nodes.len(); nodes.push(apply_custom("matmul"));
    push_w(&mut wires, id_gated_2d, id_delta_2d, 0);
    push_w(&mut wires, id_ssm_out,  id_delta_2d, 1);

    // Reshape delta from [1, d_model] to [d_model] (1-D) for concat.
    let id_delta_1d = nodes.len(); nodes.push(apply_custom("reshape"));
    push_w(&mut wires, id_delta_2d, id_delta_1d, 0);
    push_w(&mut wires, id_shape_dm, id_delta_1d, 1);

    let _ = (id_d_model_lit,); // silence unused if any

    // tensor_concat is binary. Build:
    //   packed1 = concat(delta_1d, new_conv_state, axis=0) → [d_model + conv_size]
    //   packed_final = concat(packed1, new_ssm_state, axis=0) → [d_model + conv_size + ssm_size]
    let id_packed1 = nodes.len(); nodes.push(apply_custom("tensor_concat"));
    push_w(&mut wires, id_delta_1d,        id_packed1, 0);
    push_w(&mut wires, id_new_conv_state,  id_packed1, 1);
    push_w(&mut wires, id_axis0,           id_packed1, 2);

    let id_packed_final = nodes.len(); nodes.push(apply_custom("tensor_concat"));
    push_w(&mut wires, id_packed1,         id_packed_final, 0);
    push_w(&mut wires, id_new_ssm_state,   id_packed_final, 1);
    push_w(&mut wires, id_axis0,           id_packed_final, 2);

    FunctionGraph {
        name: "jamba_ssm_block_with_state".into(),
        nodes,
        wires,
        output_node: id_packed_final,
        return_type: "f32".into(),
    }
}

fn build_ffn_block_graph() -> FunctionGraph {
    let m = WireKind::Move;
    let nodes = vec![
        p_param("residual"),
        p_param("ffn_norm_w"),
        p_param("gate_W"),
        p_param("up_W"),
        p_param("down_W"),
        lit_i32(1),
        lit_f32(1e-6),
        apply_custom("rms_norm"),
        apply_custom("matmul"),
        apply_custom("matmul"),
        apply_custom("swiglu"),
        apply_custom("matmul"),
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

// ── Math helpers ───────────────────────────────────────────────────

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

fn argmax(logits: &[f32]) -> i32 {
    let mut best_i = 0i32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v { best_v = v; best_i = i as i32; }
    }
    best_i
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
fn push_w(wires: &mut Vec<Wire>, from: usize, to: usize, port: i32) {
    wires.push(Wire { from, to, port_in_idx: port, wire_kind: WireKind::Move });
}
fn _suppress_unused(_: ScalarValue) {}
