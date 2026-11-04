//! Jamba-Reasoning inference runtime on the rce substrate.
//!
//! Loads a Jamba-Reasoning-3B GGUF, pre-loads + transposes all block
//! weights into f32 buffers, and runs the 28-block hybrid forward
//! (26 Mamba-1 SSM mixers + 2 MQA attention + 28 dense-gated FFN)
//! through rce `OpDispatch` leaves: `rce-op-blas` for matmul,
//! `rce-op-conv1d` for the causal depthwise conv, `rce-op-ssm`'s
//! `SsmScanMamba1Leaf` for the selective scan, `rce-op-attention`
//! for per-head MQA, and `rce-op-nn-prims` for silu / rms_norm /
//! split_at / swiglu — all dispatched via the in-tree CPU VM.
//!
//! State (`conv_state` + `ssm_state` per SSM block, growing `k_cache`
//! + `v_cache` per attention block) is threaded across forward passes
//! for autoregressive generation. Sampling (temperature + top-K +
//! top-P + repetition penalty + seeded RNG) comes from
//! `rce-ingest-gguf::sampling`. The chat template (`<|im_start|>` /
//! `<|im_end|>` / `<think>` framing + the Jamba thinking prefix) and
//! byte-fallback tokenization come from `rce-ingest-gguf::tokenizer`.
//!
//! ## Public API
//!
//! ```ignore
//! use rce_jamba::{JambaRuntime, GenConfig};
//!
//! let rt = JambaRuntime::load("jamba-reasoning-3b-Q4_K_M.gguf")?;
//! let reply = rt.chat(&[("user", "What is the capital of France?")],
//!                     &GenConfig::default(), |tok_text| print!("{tok_text}"))?;
//! ```
//!
//! Lower-level entry points:
//!
//! - [`JambaRuntime::forward_logits`] — single-token forward, returns
//!   the raw vocab logits. Mutates `JambaState` in place. The decode
//!   loop's primitive.
//! - [`JambaRuntime::prefill_logits`] — batched prefill: one forward
//!   pass over the whole prompt with `[T, d]` activations, returning
//!   the logits for position `T-1`. Per-layer matmuls collapse to
//!   single `[T,K]×[K,N]` BLAS calls (vs `T` separate `[1,K]×[K,N]`
//!   calls in the sequential path). SSM scan / conv1d / attention are
//!   inherently O(T) and still loop. Equivalent state to `T`
//!   sequential `forward_logits` calls.
//! - [`JambaRuntime::chat_with_budget`] — `chat` plus an
//!   `over_budget: FnMut() → bool` closure called once per generated
//!   token. Returning `true` stops generation. Designed for joule-meter
//!   integration (qmm-adapter or any other energy meter).
//! - [`JambaRuntime::new_state`], [`JambaRuntime::tokenizer`],
//!   [`JambaRuntime::config`] — accessors for the per-conversation
//!   state, tokenizer, and parsed model config.

use rce_ingest_gguf::{read_tensor_f32, sample_with_history, GgufFile};
pub use rce_ingest_gguf::{SampleConfig, Tokenizer};
use rce_op_attention::AttentionLeaf;
use rce_op_blas::BlasMatmul;
use rce_op_conv1d::Conv1dCausalLeaf;
use rce_op_ssm::SsmScanMamba1Leaf;
use rce_op_nn_prims::{RmsNormLeaf, SiluLeaf, SplitAtLeaf, SwigluLeaf};
use rce_vm_backend_cpu::{
    interpret_with_config, HostTensor, Inputs, InterpretConfig, OpDispatch, Value,
};
use rce_graph::graph::{FunctionGraph, GraphNode, LiteralValue, OpKind, Param, Wire, WireKind};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

// ── Model config ───────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct JambaConfig {
    pub d_model: usize,
    pub d_inner_ssm: usize,
    pub d_state: usize,
    pub dt_rank: usize,
    pub d_conv: usize,
    pub d_inner_ffn: usize,
    pub n_heads: usize,
    pub d_head: usize,
    pub n_layers: usize,
    pub eps: f32,
}

impl JambaConfig {
    /// Jamba-Reasoning-3B (AI21). 28 blocks: 26 SSM + 2 attention (7, 21).
    pub const REASONING_3B: Self = JambaConfig {
        d_model: 2560,
        d_inner_ssm: 5120,
        d_state: 16,
        dt_rank: 160,
        d_conv: 4,
        d_inner_ffn: 8192,
        n_heads: 20,
        d_head: 128,
        n_layers: 28,
        eps: 1e-6,
    };
}

// ── Pre-loaded weights ─────────────────────────────────────────────

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

enum BlockTensors {
    Ssm(Box<SsmBlockTensors>),
    Attn(Box<AttnBlockTensors>),
}

struct JambaWeights {
    token_embd: Vec<f32>,
    output_norm_w: Vec<f32>,
    lm_head_t: Vec<f32>,
    vocab: usize,
    blocks: Vec<BlockTensors>,
}

// ── Recurrent state ────────────────────────────────────────────────

enum BlockState {
    Ssm { conv_state: Vec<f32>, ssm_state: Vec<f32> },
    Attn { k_cache: Vec<f32>, v_cache: Vec<f32> },
}

/// Per-conversation recurrent state. Allocate one with
/// `JambaRuntime::new_state`, feed it through `forward_logits`, and
/// reuse it across forward passes for autoregressive decoding. Drop
/// and re-create it to reset the conversation.
pub struct JambaState {
    blocks: Vec<BlockState>,
}

impl JambaState {
    fn new(cfg: JambaConfig, ssm_blocks: &[usize], attn_blocks: &[usize]) -> Self {
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            if ssm_blocks.contains(&i) {
                blocks.push(BlockState::Ssm {
                    conv_state: vec![0.0f32; cfg.d_inner_ssm * cfg.d_conv],
                    ssm_state: vec![0.0f32; cfg.d_inner_ssm * cfg.d_state],
                });
            } else if attn_blocks.contains(&i) {
                blocks.push(BlockState::Attn { k_cache: Vec::new(), v_cache: Vec::new() });
            } else {
                panic!("block {i} is neither SSM nor attention");
            }
        }
        Self { blocks }
    }
}

// ── Generation config ──────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct GenConfig {
    pub max_new_tokens: usize,
    pub sample: SampleConfig,
    /// Sliding-window size for the repetition penalty (recent tokens).
    pub rep_window: usize,
    /// Whether to prepend the Jamba thinking prefix to the last user
    /// message in the chat template.
    pub thinking_prefix: bool,
    /// Use batched prefill: process the whole prompt in one forward
    /// pass with `[T, d]` activations (the per-layer matmuls become
    /// single `[T,K]×[K,N]` BLAS calls instead of `T` separate
    /// `[1,K]×[K,N]` calls — much better cache/SIMD utilisation). The
    /// SSM scan / conv1d / attention still loop over the sequence
    /// (inherently O(T)). State after batched prefill is equivalent
    /// to `T` sequential single-token forwards (the kernels are the
    /// same; only BLAS accumulation order differs, which doesn't flip
    /// the argmax). Default true.
    pub batched_prefill: bool,
}

impl Default for GenConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            sample: SampleConfig::chat(0xC0DE_F00D_2026),
            rep_window: 64,
            thinking_prefix: true,
            batched_prefill: true,
        }
    }
}

// ── Runtime ────────────────────────────────────────────────────────

/// A loaded Jamba runtime: weights pre-loaded into f32 buffers, plus
/// the parsed GGUF (kept for on-demand tokenizer reconstruction), the
/// per-block graphs, and the OpDispatch leaves. Cheap to clone-free
/// share across threads via `&self`.
pub struct JambaRuntime {
    cfg: JambaConfig,
    gguf: GgufFile,
    weights: JambaWeights,
    ssm_blocks: Vec<usize>,
    attn_blocks: Vec<usize>,
    ssm_graph: FunctionGraph,
    ffn_graph: FunctionGraph,
}

impl JambaRuntime {
    /// Load a Jamba-Reasoning GGUF from disk. mmaps the file, parses
    /// the header, detects which blocks are SSM vs attention, and
    /// pre-loads + transposes every block weight into f32 buffers
    /// (~12 GB resident for the 3B Q4_K_M). The mmap is dropped after
    /// loading; only the f32 buffers + parsed metadata are retained.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        Self::load_with_config(path, JambaConfig::REASONING_3B)
    }

    pub fn load_with_config(path: impl AsRef<Path>, cfg: JambaConfig) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| format!("open: {e}"))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?;
        let _ = mmap.advise(memmap2::Advice::WillNeed);
        let bytes: &[u8] = &mmap;
        let gguf = GgufFile::parse(bytes).map_err(|e| format!("parse: {e:?}"))?;

        let ssm_blocks: Vec<usize> = (0..cfg.n_layers)
            .filter(|i| gguf.tensors.iter().any(|t| t.name == format!("blk.{i}.ssm_in.weight")))
            .collect();
        let attn_blocks: Vec<usize> = (0..cfg.n_layers)
            .filter(|i| gguf.tensors.iter().any(|t| t.name == format!("blk.{i}.attn_q.weight")))
            .collect();
        if ssm_blocks.len() + attn_blocks.len() != cfg.n_layers {
            return Err(format!(
                "block classification mismatch: {} ssm + {} attn != {} layers",
                ssm_blocks.len(), attn_blocks.len(), cfg.n_layers
            ));
        }

        let weights = load_jamba_weights(bytes, &gguf, cfg, &ssm_blocks, &attn_blocks);
        let ssm_graph = build_ssm_block_graph_with_state(cfg);
        let ffn_graph = build_ffn_block_graph(cfg);
        drop(mmap);

        Ok(Self { cfg, gguf, weights, ssm_blocks, attn_blocks, ssm_graph, ffn_graph })
    }

    /// A borrowed view of the GGUF-embedded tokenizer. Cheap to call —
    /// it just collects `&str` references over the parsed vocab.
    pub fn tokenizer(&self) -> Tokenizer<'_> {
        Tokenizer::from_gguf(&self.gguf).expect("GGUF carries a tokenizer")
    }

    pub fn config(&self) -> JambaConfig {
        self.cfg
    }

    /// Fresh per-conversation recurrent state.
    pub fn new_state(&self) -> JambaState {
        JambaState::new(self.cfg, &self.ssm_blocks, &self.attn_blocks)
    }

    fn ops_chain<'a>(
        blas: &'a BlasMatmul, attn: &'a AttentionLeaf, conv: &'a Conv1dCausalLeaf,
        ssm: &'a SsmScanMamba1Leaf, silu: &'a SiluLeaf, rms: &'a RmsNormLeaf,
        split: &'a SplitAtLeaf, swiglu: &'a SwigluLeaf,
    ) -> Vec<&'a dyn OpDispatch> {
        vec![blas, attn, conv, ssm, silu, rms, split, swiglu]
    }

    /// Run one forward pass for `token_id`, threading `state`, and
    /// return the raw vocab logits (length = `vocab`). The state's
    /// SSM `conv_state`/`ssm_state` and attention `k_cache`/`v_cache`
    /// are updated in place.
    pub fn forward_logits(&self, state: &mut JambaState, token_id: i32) -> Vec<f32> {
        let blas = BlasMatmul::new();
        let attn = AttentionLeaf::new();
        let conv = Conv1dCausalLeaf::new();
        let ssm = SsmScanMamba1Leaf::new();
        let silu = SiluLeaf::new();
        let rms = RmsNormLeaf::new();
        let split = SplitAtLeaf::new();
        let swiglu = SwigluLeaf::new();
        let ops = Self::ops_chain(&blas, &attn, &conv, &ssm, &silu, &rms, &split, &swiglu);
        forward_via_rce_logits_cached(
            &self.weights, state, token_id, self.cfg, &self.ssm_graph, &self.ffn_graph, &ops, &attn,
        )
    }

    /// **Batched prefill**: run the whole `prompt` through one forward
    /// pass with `[T, d]` activations, threading `state`, and return
    /// the logits for the *last* prompt position (i.e. the prediction
    /// for token T+1). The state's SSM `conv_state`/`ssm_state` and
    /// attention `k_cache`/`v_cache` end up reflecting all T tokens —
    /// equivalent to feeding the prompt one token at a time, but with
    /// the per-layer matmuls batched into single BLAS calls. Returns
    /// an empty vec for an empty prompt.
    pub fn prefill_logits(&self, state: &mut JambaState, prompt: &[i32]) -> Vec<f32> {
        if prompt.is_empty() {
            return Vec::new();
        }
        let blas = BlasMatmul::new();
        forward_prefill_batched(&self.weights, state, prompt, self.cfg, &blas)
    }

    /// Tokenize a chat conversation, prefill the prompt, then generate
    /// up to `cfg.max_new_tokens` tokens (stopping early on
    /// `<|im_end|>` / EOS). Calls `on_token(text)` for each generated
    /// token's decoded text as it's produced (streaming). Returns the
    /// full decoded reply (special tokens stripped).
    pub fn chat(
        &self,
        messages: &[(&str, &str)],
        cfg: &GenConfig,
        on_token: impl FnMut(&str),
    ) -> Result<String, String> {
        // Default: no per-token energy budget. Callers who want one go
        // through `chat_with_budget` and supply a closure that asks the
        // qmm-adapter (or any other meter) whether to keep going.
        self.chat_with_budget(messages, cfg, on_token, || false)
    }

    /// Same as [`chat`], but polls `over_budget()` between each generated
    /// token's forward pass. When it returns `true`, generation stops and
    /// the partial reply (decoded so far) is returned — same shape as the
    /// EOS-stop path. Threading model: `over_budget` is called serially on
    /// the generation thread, so it can read non-`Sync` state freely.
    ///
    /// Typical wiring: pass `|| adapter.over_budget(budget_uj)` where
    /// `adapter` is the `PlacementQmmAdapter` whose `stats` are updated by
    /// every qmm dispatch in the forward pass. This makes Jamba inference
    /// behave the same way the verity-cascade L3 short-circuit does:
    /// check headroom *before* paying the next forward's cost. Setting
    /// `cfg.max_new_tokens` still caps the loop independently — both
    /// stops are honored.
    pub fn chat_with_budget(
        &self,
        messages: &[(&str, &str)],
        cfg: &GenConfig,
        mut on_token: impl FnMut(&str),
        mut over_budget: impl FnMut() -> bool,
    ) -> Result<String, String> {
        let tk = self.tokenizer();
        let prompt = tk
            .format_chat_messages(messages, true, cfg.thinking_prefix)
            .ok_or("model lacks chat-template special tokens")?;
        let eos_id = tk.eos_token_id.unwrap_or(-1);
        let im_end_id = tk.lookup_token("<|im_end|>").unwrap_or(-1);

        let mut state = self.new_state();
        let mut history: Vec<i32> = Vec::with_capacity(prompt.len() + cfg.max_new_tokens);
        let mut step: u64 = 0;
        let seed_for = |s: u64| {
            let mut c = cfg.sample;
            c.seed = cfg.sample.seed ^ s.wrapping_mul(0x9E37_79B9);
            c
        };

        // ── Prefill ───────────────────────────────────────────────
        let mut next: i32;
        if cfg.batched_prefill {
            // One forward over the whole prompt; the per-layer matmuls
            // batch into single BLAS calls.
            let logits = self.prefill_logits(&mut state, &prompt);
            for &tok in &prompt {
                history.push(tok);
            }
            // Advance the sampler seed by the prompt length to keep
            // generation-step seeds consistent with the per-token path.
            step += prompt.len() as u64;
            let recent = &history[history.len().saturating_sub(cfg.rep_window)..];
            next = sample_with_history(&logits, &seed_for(step - 1), recent);
        } else {
            // Feed every prompt token, keep only the prediction after
            // the last one.
            next = -1;
            for (i, &tok) in prompt.iter().enumerate() {
                let logits = self.forward_logits(&mut state, tok);
                history.push(tok);
                let recent = &history[history.len().saturating_sub(cfg.rep_window)..];
                let pred = sample_with_history(&logits, &seed_for(step), recent);
                step += 1;
                if i == prompt.len() - 1 {
                    next = pred;
                }
            }
        }

        // ── Generate. ─────────────────────────────────────────────
        let mut out_ids: Vec<i32> = Vec::new();
        for _ in 0..cfg.max_new_tokens {
            if next == im_end_id || next == eos_id {
                break;
            }
            // Check the budget *before* paying the next forward's cost
            // (matches verity-cascade's L3 short-circuit semantics). The
            // already-sampled `next` is still emitted — the cost was
            // already paid by the previous forward — but no new forward
            // runs.
            if over_budget() {
                break;
            }
            out_ids.push(next);
            history.push(next);
            let text = tk.decode_sequence(&[next], false);
            on_token(&text);

            let logits = self.forward_logits(&mut state, next);
            let recent = &history[history.len().saturating_sub(cfg.rep_window)..];
            next = sample_with_history(&logits, &seed_for(step), recent);
            step += 1;
        }

        Ok(tk.decode_sequence(&out_ids, true))
    }
}

// ── Weight loading ─────────────────────────────────────────────────

fn load_jamba_weights(
    bytes: &[u8], gguf: &GgufFile, cfg: JambaConfig, ssm_blocks: &[usize], attn_blocks: &[usize],
) -> JambaWeights {
    let token_embd = read_tensor_f32(bytes, gguf, "token_embd.weight").expect("token_embd");
    let vocab = token_embd.len() / cfg.d_model;
    let output_norm_w = read_tensor_f32(bytes, gguf, "output_norm.weight").expect("output_norm");
    let lm_head_w = read_tensor_f32(bytes, gguf, "output.weight").expect("output");
    let lm_head_t = transpose_2d(&lm_head_w, vocab, cfg.d_model);
    drop(lm_head_w);

    let mut blocks: Vec<BlockTensors> = Vec::with_capacity(cfg.n_layers);
    for block_idx in 0..cfg.n_layers {
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
            blocks.push(BlockTensors::Ssm(Box::new(SsmBlockTensors {
                attn_norm_w, ssm_in_t, conv_w, conv_b, ssm_x_t, dt_norm_w, b_norm_w, c_norm_w,
                ssm_dt_t, dt_b, a_log, d_param, ssm_out_t, ffn_norm_w, gate_t, up_t, down_t,
            })));
        } else if attn_blocks.contains(&block_idx) {
            let attn_q = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_q.weight")).expect("attn_q");
            let attn_k = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_k.weight")).expect("attn_k");
            let attn_v = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_v.weight")).expect("attn_v");
            let attn_o = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_output.weight")).expect("attn_o");
            let q_t = transpose_2d(&attn_q, cfg.n_heads * cfg.d_head, cfg.d_model);
            let k_t = transpose_2d(&attn_k, cfg.d_head, cfg.d_model);
            let v_t = transpose_2d(&attn_v, cfg.d_head, cfg.d_model);
            let o_t = transpose_2d(&attn_o, cfg.d_model, cfg.d_model);
            blocks.push(BlockTensors::Attn(Box::new(AttnBlockTensors {
                attn_norm_w, q_t, k_t, v_t, o_t, ffn_norm_w, gate_t, up_t, down_t,
            })));
        } else {
            panic!("block {block_idx} is neither SSM nor attention");
        }
    }

    JambaWeights { token_embd, output_norm_w, lm_head_t, vocab, blocks }
}

// ── Forward ────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
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

    for block_idx in 0..cfg.n_layers {
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
                    _ => panic!("ssm block returned non-tensor"),
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
                attn_block(&attn_normed, &a.q_t, &a.k_t, &a.v_t, &a.o_t, kc, vc, cfg, attn_leaf)
            }
        };
        for i in 0..cfg.d_model { residual[i] += mixer_delta[i]; }

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
            _ => panic!("ffn block returned non-tensor"),
        };
        for i in 0..cfg.d_model { residual[i] += ffn_delta[i]; }
    }

    let final_normed = rms_norm_inline(&residual, &weights.output_norm_w, cfg.eps);
    matmul_1xN_NxM(&final_normed, &weights.lm_head_t, cfg.d_model, weights.vocab)
}

#[allow(clippy::too_many_arguments)]
fn attn_block(
    attn_normed: &[f32], q_t: &[f32], k_t: &[f32], v_t: &[f32], o_t: &[f32],
    k_cache: &mut Vec<f32>, v_cache: &mut Vec<f32>, cfg: JambaConfig, _attn_leaf: &AttentionLeaf,
) -> Vec<f32> {
    let q = matmul_1xN_NxM(attn_normed, q_t, cfg.d_model, cfg.n_heads * cfg.d_head);
    let k_new = matmul_1xN_NxM(attn_normed, k_t, cfg.d_model, cfg.d_head);
    let v_new = matmul_1xN_NxM(attn_normed, v_t, cfg.d_model, cfg.d_head);
    k_cache.extend_from_slice(&k_new);
    v_cache.extend_from_slice(&v_new);
    let seq_len = k_cache.len() / cfg.d_head;
    let scale = 1.0 / (cfg.d_head as f32).sqrt();
    let mut multi_head_attended = vec![0.0f32; cfg.n_heads * cfg.d_head];
    // MQA: single shared K/V head (kv_dim = d_head); all query heads
    // attend against kv_head=0. Routed through rce-op-attn's fused
    // online-softmax kernel (NEON dot + single-pass V accumulator).
    for h in 0..cfg.n_heads {
        let q_h = &q[h * cfg.d_head..(h + 1) * cfg.d_head];
        let attended_h = rce_op_attn::fused_attend_head(
            q_h, k_cache, v_cache,
            0, seq_len, cfg.d_head, cfg.d_head, scale,
        );
        multi_head_attended[h * cfg.d_head..(h + 1) * cfg.d_head]
            .copy_from_slice(&attended_h);
    }
    matmul_1xN_NxM(&multi_head_attended, o_t, cfg.n_heads * cfg.d_head, cfg.d_model)
}

// ── Batched prefill ────────────────────────────────────────────────

/// `[T, d]`-activation forward over a prompt. The per-layer matmuls
/// (ssm_in / ssm_x / dt_proj / ssm_out, q/k/v/o, gate/up/down, and
/// the final lm_head) go through `BlasMatmul` as single `[T,K]×[K,N]`
/// calls. The depthwise conv1d (`conv1d_causal_step`) and the
/// selective scan (`selective_scan_mamba1_step`) loop over the T
/// positions threading state — same kernels as the per-token path,
/// so the state ends up identical. Attention is computed causally
/// over the sequence and the KV cache is set to all T positions.
/// rms_norm and silu/swiglu match the nn-prims leaves exactly.
fn forward_prefill_batched(
    weights: &JambaWeights, state: &mut JambaState, prompt: &[i32], cfg: JambaConfig,
    blas: &BlasMatmul,
) -> Vec<f32> {
    let t = prompt.len();
    let dm = cfg.d_model;
    let di = cfg.d_inner_ssm;
    let ds = cfg.d_state;

    // matmul([m,k] @ [k,n]) -> [m,n] via the BLAS leaf.
    let mm = |a: &[f32], m: usize, k: usize, b: &[f32], n: usize| -> Vec<f32> {
        let ta = HostTensor::from_f32(vec![m, k], a).unwrap();
        let tb = HostTensor::from_f32(vec![k, n], b).unwrap();
        match blas.execute(0, "matmul", &[Value::Tensor(ta), Value::Tensor(tb)]) {
            Ok(Value::Tensor(out)) => out.as_f32().unwrap(),
            other => panic!("blas matmul failed: {other:?}"),
        }
    };

    // Embeddings: gather T rows.
    let mut residual = vec![0.0f32; t * dm];
    for (i, &tok) in prompt.iter().enumerate() {
        let src = tok as usize * dm;
        residual[i * dm..(i + 1) * dm].copy_from_slice(&weights.token_embd[src..src + dm]);
    }

    for block_idx in 0..cfg.n_layers {
        let block = &weights.blocks[block_idx];
        let (attn_norm_w, ffn_norm_w, gate_t, up_t, down_t) = match block {
            BlockTensors::Ssm(s) => (&s.attn_norm_w, &s.ffn_norm_w, &s.gate_t, &s.up_t, &s.down_t),
            BlockTensors::Attn(a) => (&a.attn_norm_w, &a.ffn_norm_w, &a.gate_t, &a.up_t, &a.down_t),
        };

        // rms_norm per row.
        let mut attn_normed = vec![0.0f32; t * dm];
        for i in 0..t {
            let r = rms_norm_inline(&residual[i * dm..(i + 1) * dm], attn_norm_w, cfg.eps);
            attn_normed[i * dm..(i + 1) * dm].copy_from_slice(&r);
        }

        let mixer_delta: Vec<f32> = match block {
            BlockTensors::Ssm(s) => {
                let (cs, ss) = match &mut state.blocks[block_idx] {
                    BlockState::Ssm { conv_state, ssm_state } => (conv_state, ssm_state),
                    _ => unreachable!(),
                };
                // ssm_in: [t, dm] @ [dm, 2*di] -> [t, 2*di]
                let xz = mm(&attn_normed, t, dm, &s.ssm_in_t, 2 * di);
                let mut x_inner = vec![0.0f32; t * di];
                let mut z = vec![0.0f32; t * di];
                for i in 0..t {
                    let base = i * 2 * di;
                    x_inner[i * di..(i + 1) * di].copy_from_slice(&xz[base..base + di]);
                    z[i * di..(i + 1) * di].copy_from_slice(&xz[base + di..base + 2 * di]);
                }
                // conv1d over the sequence (threads conv_state).
                let mut x_conv = vec![0.0f32; t * di];
                for i in 0..t {
                    let (y, new_cs) = rce_op_conv1d::conv1d_causal_step(
                        &x_inner[i * di..(i + 1) * di], &s.conv_w, &s.conv_b, cs, di, cfg.d_conv,
                    );
                    x_conv[i * di..(i + 1) * di].copy_from_slice(&y);
                    *cs = new_cs;
                }
                let x_act: Vec<f32> = x_conv.iter().map(|v| stable_silu(*v)).collect();
                // ssm_x: [t, di] @ [di, dt_rank+2*ds] -> [t, dt_rank+2*ds]
                let bc = cfg.dt_rank + 2 * ds;
                let dt_bc = mm(&x_act, t, di, &s.ssm_x_t, bc);
                let mut dt_normed = vec![0.0f32; t * cfg.dt_rank];
                let mut b_normed = vec![0.0f32; t * ds];
                let mut c_normed = vec![0.0f32; t * ds];
                for i in 0..t {
                    let base = i * bc;
                    let dn = rms_norm_inline(&dt_bc[base..base + cfg.dt_rank], &s.dt_norm_w, cfg.eps);
                    let bn = rms_norm_inline(&dt_bc[base + cfg.dt_rank..base + cfg.dt_rank + ds], &s.b_norm_w, cfg.eps);
                    let cn = rms_norm_inline(&dt_bc[base + cfg.dt_rank + ds..base + bc], &s.c_norm_w, cfg.eps);
                    dt_normed[i * cfg.dt_rank..(i + 1) * cfg.dt_rank].copy_from_slice(&dn);
                    b_normed[i * ds..(i + 1) * ds].copy_from_slice(&bn);
                    c_normed[i * ds..(i + 1) * ds].copy_from_slice(&cn);
                }
                // dt_proj: [t, dt_rank] @ [dt_rank, di] -> [t, di]
                let dt_proj = mm(&dt_normed, t, cfg.dt_rank, &s.ssm_dt_t, di);
                // selective scan over the sequence (threads ssm_state).
                let mut y = vec![0.0f32; t * di];
                for i in 0..t {
                    let (yi, new_ss) = rce_op_ssm::selective_scan_mamba1_step(
                        &x_act[i * di..(i + 1) * di],
                        &b_normed[i * ds..(i + 1) * ds],
                        &c_normed[i * ds..(i + 1) * ds],
                        &dt_proj[i * di..(i + 1) * di],
                        &s.dt_b, &s.a_log, &s.d_param, ss, di, ds,
                    );
                    y[i * di..(i + 1) * di].copy_from_slice(&yi);
                    *ss = new_ss;
                }
                let gated: Vec<f32> = y.iter().zip(z.iter()).map(|(a, b)| a * stable_silu(*b)).collect();
                // ssm_out: [t, di] @ [di, dm] -> [t, dm]
                mm(&gated, t, di, &s.ssm_out_t, dm)
            }
            BlockTensors::Attn(a) => {
                let (kc, vc) = match &mut state.blocks[block_idx] {
                    BlockState::Attn { k_cache, v_cache } => (k_cache, v_cache),
                    _ => unreachable!(),
                };
                let dh = cfg.d_head;
                let nh = cfg.n_heads;
                let q = mm(&attn_normed, t, dm, &a.q_t, nh * dh);
                let k = mm(&attn_normed, t, dm, &a.k_t, dh);
                let v = mm(&attn_normed, t, dm, &a.v_t, dh);
                let scale = 1.0 / (dh as f32).sqrt();
                let mut attended = vec![0.0f32; t * nh * dh];
                // MQA causal prefill: for each (head, time) — fused online
                // softmax over [0, ti+1) against the single shared K/V head.
                for h in 0..nh {
                    for ti in 0..t {
                        let q_off = ti * nh * dh + h * dh;
                        let q_i = &q[q_off..q_off + dh];
                        let attended_h = rce_op_attn::fused_attend_head_window(
                            q_i, &k, &v, 0, 0, ti + 1, dh, dh, scale,
                        );
                        let out_off = ti * nh * dh + h * dh;
                        attended[out_off..out_off + dh].copy_from_slice(&attended_h);
                    }
                }
                // KV cache now holds all T positions.
                *kc = k;
                *vc = v;
                // o: [t, nh*dh] @ [nh*dh, dm] -> [t, dm]
                mm(&attended, t, nh * dh, &a.o_t, dm)
            }
        };
        for i in 0..t * dm {
            residual[i] += mixer_delta[i];
        }

        // FFN.
        let mut ffn_normed = vec![0.0f32; t * dm];
        for i in 0..t {
            let r = rms_norm_inline(&residual[i * dm..(i + 1) * dm], ffn_norm_w, cfg.eps);
            ffn_normed[i * dm..(i + 1) * dm].copy_from_slice(&r);
        }
        let dff = cfg.d_inner_ffn;
        let g = mm(&ffn_normed, t, dm, gate_t, dff);
        let u = mm(&ffn_normed, t, dm, up_t, dff);
        let hidden: Vec<f32> = g.iter().zip(u.iter()).map(|(gg, uu)| stable_silu(*gg) * uu).collect();
        let ffn_delta = mm(&hidden, t, dff, down_t, dm);
        for i in 0..t * dm {
            residual[i] += ffn_delta[i];
        }
    }

    // Logits for the last prompt position only.
    let last = &residual[(t - 1) * dm..t * dm];
    let final_normed = rms_norm_inline(last, &weights.output_norm_w, cfg.eps);
    matmul_1xN_NxM(&final_normed, &weights.lm_head_t, dm, weights.vocab)
}

fn stable_silu(x: f32) -> f32 {
    if x >= 0.0 {
        x / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        x * e / (1.0 + e)
    }
}

// ── Block graphs ───────────────────────────────────────────────────

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
    let id_shape_dm    = nodes.len(); nodes.push(lit_str(&format!("[{}]", cfg.d_model)));

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

    let id_delta_1d = nodes.len(); nodes.push(apply_custom("reshape"));
    push_w(&mut wires, id_delta_2d, id_delta_1d, 0);
    push_w(&mut wires, id_shape_dm, id_delta_1d, 1);

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

fn build_ffn_block_graph(cfg: JambaConfig) -> FunctionGraph {
    let m = WireKind::Move;
    let nodes = vec![
        p_param("residual"),
        p_param("ffn_norm_w"),
        p_param("gate_W"),
        p_param("up_W"),
        p_param("down_W"),
        lit_i32(1),
        lit_f32(cfg.eps),
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

#[allow(non_snake_case)]
/// Row-vector × matrix: `out[1, n] = a[1, k] @ b[k, n]`. Routes
/// directly to `cblas_sgemm` via `rce-op-blas::sgemm_into` —
/// previously this was a naive triple-loop (no SIMD / no AMX),
/// dominating Jamba's decode latency. AMX-accelerated sgemm typically
/// gives 30–100× speedup on real model sizes.
fn matmul_1xN_NxM(a: &[f32], b: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    rce_op_blas::sgemm_into(a, b, &mut out, 1, k, n);
    out
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
