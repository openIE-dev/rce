//! Pure Mamba-1 inference runtime on the rce substrate.
//!
//! Targets the `mamba` GGUF architecture — the original Mamba paper
//! (Gu & Dao 2024 NeurIPS). Validated against `Falcon3-Mamba-7B-Instruct`
//! (TII, 2024) at Q4_K_M: 64 SSM-only blocks, d_model=4096, d_inner=8192
//! (= 2× expansion), d_state=16, dt_rank=256, conv_kernel=4, GPT-2 BPE
//! with Falcon pre-tokenizer.
//!
//! Mamba-1 differs from Mamba-2 in three ways:
//! - **Per-channel `Δt`** (not per-head): Δt has shape `[d_inner]`. There's
//!   a low-rank `ssm_x → [dt_rank, B, C]` split, then `ssm_dt: dt_rank → d_inner`.
//! - **A is per-(channel, state)**: shape `[d_state, d_inner]`. Mamba-2's
//!   A is one scalar per head.
//! - **B and C are not group-shared**: each is a single `[d_state]` vector
//!   per token. Mamba-2 has `n_groups` B/C bands.
//!
//! Some `mamba` GGUFs flag `mamba.ssm.dt_b_c_rms = true`, which means
//! the model was trained with **RMSNorm without learnable scale** applied
//! to `dt`, `B`, `C` after the `ssm_x` projection (per llama.cpp's
//! convention). We honor that flag.
//!
//! Reuses [`rce_op_ssm::selective_scan_mamba1_step`] (now bit-correct
//! vs llama.cpp post the `ssm_a` convention fix — see
//! `project_gguf_ssm_a_convention.md`), [`rce_op_conv1d`], and
//! [`rce_op_blas::sgemm_into`] (the direct-sgemm path that avoids the
//! HostTensor memcpy round-trip).

use rce_ingest_gguf::{
    read_tensor_f32, sample_with_history, GgufFile, GgufValue, Tokenizer,
};
pub use rce_ingest_gguf::SampleConfig;
use rce_op_conv1d::conv1d_causal_step;
use rce_op_qmm::QuantWeight;
use rce_op_ssm::selective_scan_mamba1_step;
use memmap2::Mmap;
use std::sync::Arc;
use std::fs::File;
use std::path::Path;

// ── Config ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Mamba1Config {
    pub d_model: usize,
    pub vocab: usize,
    pub n_layers: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub dt_rank: usize,
    pub d_conv: usize,
    /// When true, apply RMSNorm with **no learnable scale** to dt/B/C
    /// after the `ssm_x` projection. Falcon3-Mamba sets this; original
    /// Mamba checkpoints typically don't.
    pub dt_b_c_rms: bool,
    pub eps: f32,
}

impl Mamba1Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf.metadata.iter()
            .find(|kv| kv.key == "general.architecture")
            .and_then(|kv| match &kv.value {
                GgufValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .ok_or("missing general.architecture")?;
        if arch != "mamba" {
            return Err(format!("expected mamba, got {arch}"));
        }

        let u32_meta = |k: &str| -> Option<u32> {
            gguf.metadata.iter().find(|kv| kv.key == format!("mamba.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::U32(v) => Some(*v), _ => None })
        };
        let f32_meta = |k: &str| -> Option<f32> {
            gguf.metadata.iter().find(|kv| kv.key == format!("mamba.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::F32(v) => Some(*v), _ => None })
        };
        let bool_meta = |k: &str| -> Option<bool> {
            gguf.metadata.iter().find(|kv| kv.key == format!("mamba.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::Bool(v) => Some(*v), _ => None })
        };

        let d_model = u32_meta("embedding_length").ok_or("embedding_length")? as usize;
        let vocab = u32_meta("vocab_size")
            .or_else(|| gguf.tensors.iter()
                .find(|t| t.name == "token_embd.weight")
                .and_then(|t| t.dims.get(1).copied())
                .map(|d| d as u32))
            .ok_or("vocab_size")? as usize;
        let n_layers = u32_meta("block_count").ok_or("block_count")? as usize;
        let d_inner = u32_meta("ssm.inner_size").ok_or("ssm.inner_size")? as usize;
        let d_state = u32_meta("ssm.state_size").ok_or("ssm.state_size")? as usize;
        let dt_rank = u32_meta("ssm.time_step_rank").ok_or("ssm.time_step_rank")? as usize;
        let d_conv = u32_meta("ssm.conv_kernel").ok_or("ssm.conv_kernel")? as usize;
        let dt_b_c_rms = bool_meta("ssm.dt_b_c_rms").unwrap_or(false);
        let eps = f32_meta("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);

        Ok(Mamba1Config {
            d_model, vocab, n_layers,
            d_inner, d_state, dt_rank, d_conv, dt_b_c_rms, eps,
        })
    }
}

// ── Weights ────────────────────────────────────────────────────────

struct SsmBlock {
    attn_norm_w: Vec<f32>,
    ssm_in: QuantWeight,   // [2*d_inner, d_model]   native GGUF dtype
    conv_w: Vec<f32>,      // [d_inner, d_conv] depthwise; GGUF natural layout
    conv_b: Vec<f32>,      // [d_inner]
    ssm_x: QuantWeight,    // [dt_rank + 2*d_state, d_inner]
    ssm_dt: QuantWeight,   // [d_inner, dt_rank]
    dt_bias: Vec<f32>,     // [d_inner]
    a_log: Vec<f32>,       // [d_inner, d_state]   (GGUF stores already-transformed A)
    d_param: Vec<f32>,     // [d_inner]
    ssm_out: QuantWeight,  // [d_model, d_inner]
}

struct Mamba1Weights {
    token_embd: QuantWeight,
    output_norm_w: Vec<f32>,
    lm_head: QuantWeight,  // [vocab, d_model]
    blocks: Vec<SsmBlock>,
}

// ── State ──────────────────────────────────────────────────────────

struct BlockState {
    conv_state: Vec<f32>,
    ssm_state: Vec<f32>,
}

pub struct Mamba1State {
    blocks: Vec<BlockState>,
    pos: usize,
}

impl Mamba1State {
    fn new(cfg: &Mamba1Config) -> Self {
        let blocks = (0..cfg.n_layers).map(|_| BlockState {
            conv_state: vec![0.0f32; cfg.d_inner * cfg.d_conv],
            ssm_state: vec![0.0f32; cfg.d_inner * cfg.d_state],
        }).collect();
        Self { blocks, pos: 0 }
    }
}

// ── Loading ────────────────────────────────────────────────────────

fn load_weights(
    mmap: &Arc<Mmap>, gguf: &GgufFile, cfg: &Mamba1Config,
) -> Result<Mamba1Weights, String> {
    let bytes: &[u8] = &mmap[..];
    let token_embd = QuantWeight::from_gguf(Arc::clone(mmap), gguf, "token_embd.weight", cfg.vocab, cfg.d_model)
        .map_err(|e| format!("token_embd: {e}"))?;
    let output_norm_w = read_tensor_f32(bytes, gguf, "output_norm.weight")
        .map_err(|e| format!("output_norm: {e:?}"))?;
    let lm_head = QuantWeight::from_gguf(Arc::clone(mmap), gguf, "output.weight", cfg.vocab, cfg.d_model)
        .or_else(|_| QuantWeight::from_gguf(Arc::clone(mmap), gguf, "token_embd.weight", cfg.vocab, cfg.d_model))
        .map_err(|e| format!("output: {e}"))?;

    let mut blocks = Vec::with_capacity(cfg.n_layers);
    for b in 0..cfg.n_layers {
        let bn = format!("blk.{b}");
        let attn_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_norm.weight"))
            .map_err(|e| format!("blk.{b}.attn_norm: {e:?}"))?;
        // `ssm_conv1d.weight` GGUF flat = `[d_inner, d_conv]` row-major
        // (channel-first, kernel inner), exactly what `conv1d_causal_step`
        // indexes. No transpose needed; small enough to keep as f32.
        let conv_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_conv1d.weight")).map_err(|e| format!("{e:?}"))?;
        let conv_b = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_conv1d.bias")).map_err(|e| format!("{e:?}"))?;
        let dt_bias = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_dt.bias")).map_err(|e| format!("{e:?}"))?;
        // `ssm_a` GGUF flat = `[d_inner, d_state]` row-major. GGUF
        // already stores `A = -exp(A_log)`; use directly.
        let a_log = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_a")).map_err(|e| format!("{e:?}"))?;
        let d_param = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_d")).map_err(|e| format!("{e:?}"))?;

        // Matmul-heavy weights: native dtype, no transpose.
        let ssm_in = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.ssm_in.weight"), 2 * cfg.d_inner, cfg.d_model)
            .map_err(|e| format!("blk.{b}.ssm_in: {e}"))?;
        let ssm_x = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.ssm_x.weight"), cfg.dt_rank + 2 * cfg.d_state, cfg.d_inner)
            .map_err(|e| format!("blk.{b}.ssm_x: {e}"))?;
        let ssm_dt = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.ssm_dt.weight"), cfg.d_inner, cfg.dt_rank)
            .map_err(|e| format!("blk.{b}.ssm_dt: {e}"))?;
        let ssm_out = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.ssm_out.weight"), cfg.d_model, cfg.d_inner)
            .map_err(|e| format!("blk.{b}.ssm_out: {e}"))?;
        blocks.push(SsmBlock {
            attn_norm_w, ssm_in, conv_w, conv_b, ssm_x, ssm_dt,
            dt_bias, a_log, d_param, ssm_out,
        });
    }
    Ok(Mamba1Weights { token_embd, output_norm_w, lm_head, blocks })
}

// ── Runtime ────────────────────────────────────────────────────────

pub struct Mamba1Runtime {
    cfg: Mamba1Config,
    gguf: GgufFile,
    weights: Mamba1Weights,
}

impl Mamba1Runtime {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| format!("open: {e}"))?;
        let mmap = Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?);
        // Hint sequential reads during the load-time weight scan; ignore
        // errors (kernel may refuse). See rce_ingest_gguf::mmap_load_sequential.
        let _ = mmap.advise(memmap2::Advice::WillNeed);
        let gguf = GgufFile::parse(&mmap[..]).map_err(|e| format!("parse: {e:?}"))?;
        let cfg = Mamba1Config::from_gguf(&gguf)?;
        eprintln!(
            "mamba: d_model={} vocab={} n_layers={} d_inner={} d_state={} dt_rank={} d_conv={} dt_b_c_rms={}",
            cfg.d_model, cfg.vocab, cfg.n_layers, cfg.d_inner, cfg.d_state, cfg.dt_rank, cfg.d_conv, cfg.dt_b_c_rms,
        );
        // mmap kept alive transitively via QuantWeight Arc clones.
        let weights = load_weights(&mmap, &gguf, &cfg)?;
        Ok(Self { cfg, gguf, weights })
    }

    pub fn config(&self) -> &Mamba1Config { &self.cfg }

    pub fn tokenizer(&self) -> Tokenizer<'_> {
        Tokenizer::from_gguf(&self.gguf).expect("tokenizer")
    }

    pub fn new_state(&self) -> Mamba1State {
        Mamba1State::new(&self.cfg)
    }

    pub fn forward_logits(&self, state: &mut Mamba1State, token_id: i32) -> Vec<f32> {
        let logits = forward_token(&self.weights, &self.cfg, state, token_id);
        state.pos += 1;
        logits
    }
}

fn forward_token(
    w: &Mamba1Weights, cfg: &Mamba1Config,
    state: &mut Mamba1State, token_id: i32,
) -> Vec<f32> {
    let dm = cfg.d_model;
    let mut x: Vec<f32> = w.token_embd.dequant_row(token_id as usize);

    for (b, block) in w.blocks.iter().enumerate() {
        let normed = rms_norm(&x, &block.attn_norm_w, cfg.eps);
        let bs = &mut state.blocks[b];
        let delta = ssm_mixer(&normed, block, &mut bs.conv_state, &mut bs.ssm_state, cfg);
        for i in 0..dm { x[i] += delta[i]; }
    }

    let final_normed = rms_norm(&x, &w.output_norm_w, cfg.eps);
    qmm_1d(&final_normed, &w.lm_head)
}

// ── Mamba-1 mixer ──────────────────────────────────────────────────

fn ssm_mixer(
    x_in: &[f32], s: &SsmBlock, conv_state: &mut Vec<f32>, ssm_state: &mut Vec<f32>,
    cfg: &Mamba1Config,
) -> Vec<f32> {
    let dm = cfg.d_model;
    let di = cfg.d_inner;
    let ds = cfg.d_state;
    let dr = cfg.dt_rank;

    // ssm_in: [d_model] → [2*d_inner]  → split into [x_inner, z]
    let xz = qmm_1d(x_in, &s.ssm_in);
    let x_inner = &xz[0..di];
    let z = &xz[di..2 * di];

    // depthwise causal conv1d, threading conv_state
    let (x_conv, new_conv_state) = conv1d_causal_step(
        x_inner, &s.conv_w, &s.conv_b, conv_state, di, cfg.d_conv,
    );
    *conv_state = new_conv_state;
    let x_act: Vec<f32> = x_conv.iter().map(|v| silu(*v)).collect();

    // ssm_x: [d_inner] → [dt_rank + 2*d_state]  → split into [dt_low, B, C]
    let bc = dr + 2 * ds;
    let dt_bc = qmm_1d(&x_act, &s.ssm_x);
    let (dt_low_raw, rest) = dt_bc.split_at(dr);
    let (b_raw, c_raw) = rest.split_at(ds);

    // `mamba.ssm.dt_b_c_rms = true` ⇒ apply RMSNorm with no learnable
    // scale (variance normalisation only). Matches llama.cpp's
    // `llm_build_norm(..., mw=nullptr, ..., LLM_NORM_RMS)`.
    let dt_low = if cfg.dt_b_c_rms { rms_norm_no_scale(dt_low_raw, cfg.eps) } else { dt_low_raw.to_vec() };
    let b_act = if cfg.dt_b_c_rms { rms_norm_no_scale(b_raw, cfg.eps) } else { b_raw.to_vec() };
    let c_act = if cfg.dt_b_c_rms { rms_norm_no_scale(c_raw, cfg.eps) } else { c_raw.to_vec() };

    // dt_proj: [dt_rank] → [d_inner]  (per-channel Δt pre-softplus)
    let dt_raw = qmm_1d(&dt_low, &s.ssm_dt);

    // Mamba-1 selective scan (per-channel Δt, per-(channel, state) A,
    // single B/C vector). Same kernel Jamba uses.
    let (y, new_ssm) = selective_scan_mamba1_step(
        &x_act, &b_act, &c_act, &dt_raw, &s.dt_bias, &s.a_log, &s.d_param, ssm_state,
        di, ds,
    );
    *ssm_state = new_ssm;

    // Gate by silu(z), then output projection.
    let mut gated = vec![0.0f32; di];
    for i in 0..di { gated[i] = silu(z[i]) * y[i]; }
    qmm_1d(&gated, &s.ssm_out)
}

// ── Math helpers ───────────────────────────────────────────────────

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(w.iter()).map(|(v, ww)| v * inv * ww).collect()
}

fn rms_norm_no_scale(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().map(|v| v * inv).collect()
}

fn silu(x: f32) -> f32 {
    if x >= 0.0 { x / (1.0 + (-x).exp()) } else { let e = x.exp(); x * e / (1.0 + e) }
}

fn qmm_1d(a: &[f32], w: &QuantWeight) -> Vec<f32> {
    let mut out = vec![0.0f32; w.n_rows];
    rce_op_qmm::mm_1d(a, w, &mut out);
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

// ── Generation ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct Mamba1GenConfig {
    pub max_new_tokens: usize,
    pub sample: SampleConfig,
    pub rep_window: usize,
}

impl Default for Mamba1GenConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            sample: SampleConfig::chat(0xC0DE_F00D_2026),
            rep_window: 64,
        }
    }
}

impl Mamba1Runtime {
    /// Multi-turn chat using the model's ChatML template
    /// (`<|im_start|>role\ncontent<|im_end|>\n`). Stops on `<|im_end|>`
    /// or `eos_token_id`. Mirrors `NemotronRuntime::chat`'s shape but
    /// uses `format_chat_chatml` (no Nemotron-specific empty-system /
    /// `<think>` quirks).
    pub fn chat(
        &self,
        messages: &[(&str, &str)],
        cfg: &Mamba1GenConfig,
        mut on_token: impl FnMut(&str),
    ) -> Result<String, String> {
        let tk = self.tokenizer();
        let prompt = tk
            .format_chat_chatml(messages, true)
            .ok_or("model lacks <|im_start|>/<|im_end|> tokens (no ChatML template)")?;
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

        let mut next: i32 = -1;
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

        let mut out_ids: Vec<i32> = Vec::new();
        for _ in 0..cfg.max_new_tokens {
            if next == im_end_id || next == eos_id {
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

    /// Base-LM text continuation. Use this when the model has no chat
    /// template (e.g. raw Mamba-1 paper checkpoints). For instruct
    /// models like Falcon3-Mamba-7B-Instruct, prefer `chat()`.
    pub fn generate(
        &self,
        prompt: &str,
        cfg: &Mamba1GenConfig,
        mut on_token: impl FnMut(&str),
    ) -> Result<String, String> {
        let tk = self.tokenizer();
        let is_gpt2 = self.gguf.metadata.iter()
            .find(|kv| kv.key == "tokenizer.ggml.model")
            .map(|kv| matches!(&kv.value, GgufValue::String(s) if s == "gpt2"))
            .unwrap_or(false);
        let prompt_ids = if is_gpt2 {
            tk.encode_gpt2(prompt, false)
        } else {
            tk.encode(prompt, false)
        };
        if prompt_ids.is_empty() {
            return Err("tokenizer produced empty prompt".into());
        }
        let eos_id = tk.eos_token_id.unwrap_or(-1);

        let mut state = self.new_state();
        let mut history: Vec<i32> = Vec::with_capacity(prompt_ids.len() + cfg.max_new_tokens);
        let mut step: u64 = 0;
        let seed_for = |s: u64| {
            let mut c = cfg.sample;
            c.seed = cfg.sample.seed ^ s.wrapping_mul(0x9E37_79B9);
            c
        };

        let mut next: i32 = -1;
        for (i, &tok) in prompt_ids.iter().enumerate() {
            let logits = self.forward_logits(&mut state, tok);
            history.push(tok);
            let recent = &history[history.len().saturating_sub(cfg.rep_window)..];
            let pred = sample_with_history(&logits, &seed_for(step), recent);
            step += 1;
            if i == prompt_ids.len() - 1 {
                next = pred;
            }
        }

        let mut out_ids: Vec<i32> = Vec::new();
        for _ in 0..cfg.max_new_tokens {
            if next == eos_id { break; }
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
