//! Qwen 3.6-35B-A3B (MoE) inference runtime on the rce substrate.
//!
//! `arch=qwen35moe`. Same hybrid Gated DeltaNet + Gated Attention
//! mixer as the dense Qwen 3.6-27B, but every FFN block is an MoE:
//!
//! - **256 routed experts**, top-**8** softmax routing (normalized).
//!   Each expert is a SwiGLU `d_model → n_ff_exp → d_model` block.
//!   `n_ff_exp = 512`.
//! - **Always-on shared expert** SwiGLU `d_model → n_ff_shexp → d_model`,
//!   `n_ff_shexp = 512`. Its output is scaled by a **per-token scalar
//!   sigmoid gate** computed from `ffn_gate_inp_shexp · x` (a `[d_model]`
//!   linear-to-scalar projection).
//!
//! Final FFN out = `moe_out + sigmoid(W_shexp_gate · x) * shexp_out`.
//!
//! Sizes (40 layers, 30 DeltaNet + 10 Gated Attention):
//! d_model=2048, n_q=16, n_kv=2, d_head=256, n_v_heads=32, n_k_heads=16,
//! d_state=128, d_conv=4. Tied lm_head (auto-falls back to token_embd).
//!
//! Reference: llama.cpp Draft PR `src/models/qwen35moe.cpp`.

use rce_ingest_gguf::{
    read_tensor_f32, sample_with_history, GgufFile, GgufValue, SampleConfig, Tokenizer,
};
// Re-export so the rce-qwen35-moe wrapper (sharing arch=qwen35moe) can
// configure generation without re-depending on internals.
pub use rce_ingest_gguf::SampleConfig as ExportedSampleConfig;
use rce_op_moe::moe_router;
use rce_op_qmm::QuantWeight;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

// ── Config ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Qwen36MoeConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub vocab: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub d_head: usize,
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub d_inner: usize,
    pub full_attention_interval: usize,
    pub rope_freq_base: f32,
    pub rope_dim: usize,
    pub rope_dimension_sections: [u32; 4],
    pub eps: f32,
    pub attention_scale: f32,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
}

impl Qwen36MoeConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf.metadata.iter()
            .find(|kv| kv.key == "general.architecture")
            .and_then(|kv| match &kv.value { GgufValue::String(s) => Some(s.as_str()), _ => None })
            .ok_or("missing general.architecture")?;
        if arch != "qwen35moe" {
            return Err(format!("expected qwen35moe, got {arch}"));
        }
        let u32_meta = |k: &str| -> Option<u32> {
            gguf.metadata.iter().find(|kv| kv.key == format!("qwen35moe.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::U32(v) => Some(*v), _ => None })
        };
        let f32_meta = |k: &str| -> Option<f32> {
            gguf.metadata.iter().find(|kv| kv.key == format!("qwen35moe.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::F32(v) => Some(*v), _ => None })
        };

        let d_model = u32_meta("embedding_length").ok_or("embedding_length")? as usize;
        // Trailing `nextn.*` blocks are Multi-Token-Prediction (MTP) speculative
        // heads, not trunk layers — exclude them from the autoregressive pass.
        // Qwen 3.6-35B-A3B ships none (subtracts 0); Qwen 3.5 MoE checkpoints
        // append one. Mirrors the dense fix in rce-qwen36.
        let block_count = u32_meta("block_count").ok_or("block_count")? as usize;
        let n_mtp = (0..block_count)
            .rev()
            .take_while(|il| {
                let p = format!("blk.{il}.nextn.");
                gguf.tensors.iter().any(|t| t.name.starts_with(&p))
            })
            .count();
        let n_layers = block_count - n_mtp;
        let n_q_heads = u32_meta("attention.head_count").ok_or("attention.head_count")? as usize;
        let n_kv_heads = u32_meta("attention.head_count_kv").unwrap_or(n_q_heads as u32) as usize;
        let d_head = u32_meta("attention.key_length").ok_or("attention.key_length")? as usize;
        let n_k_heads = u32_meta("ssm.group_count").unwrap_or(16) as usize;
        let n_v_heads = u32_meta("ssm.time_step_rank").unwrap_or(32) as usize;
        let d_state = u32_meta("ssm.state_size").unwrap_or(128) as usize;
        let d_conv = u32_meta("ssm.conv_kernel").unwrap_or(4) as usize;
        let d_inner = u32_meta("ssm.inner_size").unwrap_or((n_v_heads * d_state) as u32) as usize;
        let full_attention_interval = u32_meta("full_attention_interval").unwrap_or(4) as usize;
        let rope_freq_base = f32_meta("rope.freq_base").unwrap_or(10_000_000.0);
        let rope_dim = u32_meta("rope.dimension_count").unwrap_or(64) as usize;
        let eps = f32_meta("attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let attention_scale = f32_meta("attention.scale")
            .unwrap_or_else(|| 1.0 / (d_head as f32).sqrt());
        let n_expert = u32_meta("expert_count").ok_or("expert_count")? as usize;
        let n_expert_used = u32_meta("expert_used_count").ok_or("expert_used_count")? as usize;
        let n_ff_exp = u32_meta("expert_feed_forward_length").ok_or("expert_feed_forward_length")? as usize;
        let n_ff_shexp = u32_meta("expert_shared_feed_forward_length").ok_or("expert_shared_feed_forward_length")? as usize;

        let rope_dimension_sections = gguf.metadata.iter()
            .find(|kv| kv.key == "qwen35moe.rope.dimension_sections")
            .and_then(|kv| match &kv.value {
                GgufValue::Array(arr) => Some(
                    arr.iter().take(4).filter_map(|v| match v {
                        GgufValue::U32(u) => Some(*u),
                        GgufValue::I32(i) => Some(*i as u32),
                        GgufValue::U64(u) => Some(*u as u32),
                        GgufValue::I64(i) => Some(*i as u32),
                        _ => None,
                    }).collect::<Vec<u32>>()
                ),
                _ => None,
            })
            .map(|v| { let mut a = [0u32; 4]; for (i, x) in v.iter().take(4).enumerate() { a[i] = *x; } a })
            .unwrap_or([0; 4]);

        let vocab = gguf.tensors.iter()
            .find(|t| t.name == "token_embd.weight")
            .and_then(|t| t.dims.get(1).copied())
            .map(|d| d as usize)
            .ok_or("token_embd.weight missing")?;

        Ok(Qwen36MoeConfig {
            d_model, n_layers, vocab,
            n_q_heads, n_kv_heads, d_head,
            n_k_heads, n_v_heads, d_state, d_conv, d_inner,
            full_attention_interval, rope_freq_base, rope_dim,
            rope_dimension_sections, eps, attention_scale,
            n_expert, n_expert_used, n_ff_exp, n_ff_shexp,
        })
    }

    #[inline] pub fn is_attention_layer(&self, il: usize) -> bool {
        self.full_attention_interval > 0 && (il + 1) % self.full_attention_interval == 0
    }
    #[inline] pub fn key_dim(&self) -> usize { self.n_k_heads * self.d_state }
    #[inline] pub fn value_dim(&self) -> usize { self.n_v_heads * self.d_state }
    #[inline] pub fn conv_dim(&self) -> usize { self.key_dim() * 2 + self.value_dim() }
    #[inline] pub fn q_per_kv(&self) -> usize { self.n_q_heads / self.n_kv_heads }
    #[inline] pub fn v_per_k(&self) -> usize { self.n_v_heads / self.n_k_heads }
}

// ── Weights ───────────────────────────────────────────────────────────

pub struct MoeFfn {
    /// Router gate, `[n_expert, d_model]` F32.
    pub gate_inp: QuantWeight,
    /// Per-expert SwiGLU gate: `n_expert` × `[n_ff_exp, d_model]`.
    pub gate_exps: Vec<QuantWeight>,
    pub up_exps: Vec<QuantWeight>,
    /// Per-expert down: `n_expert` × `[d_model, n_ff_exp]`.
    pub down_exps: Vec<QuantWeight>,
    /// Shared expert gate input: `[d_model]`.
    pub shexp_gate_inp: Vec<f32>,
    pub shexp_gate: QuantWeight,   // [n_ff_shexp, d_model]
    pub shexp_up: QuantWeight,     // [n_ff_shexp, d_model]
    pub shexp_down: QuantWeight,   // [d_model, n_ff_shexp]
}

pub struct DeltaNetBlock {
    pub attn_norm_w: Vec<f32>,
    pub attn_post_norm_w: Vec<f32>,
    pub wqkv: QuantWeight,
    pub wqkv_gate: QuantWeight,
    pub ssm_conv1d_w: Vec<f32>,
    pub ssm_dt_bias: Vec<f32>,
    pub ssm_a: Vec<f32>,
    pub ssm_alpha: QuantWeight,
    pub ssm_beta: QuantWeight,
    pub ssm_norm_w: Vec<f32>,
    pub ssm_out: QuantWeight,
    pub ffn: MoeFfn,
}

pub struct AttentionBlock {
    pub attn_norm_w: Vec<f32>,
    pub attn_post_norm_w: Vec<f32>,
    pub wq: QuantWeight,
    pub wk: QuantWeight,
    pub wv: QuantWeight,
    pub wo: QuantWeight,
    pub q_norm_w: Vec<f32>,
    pub k_norm_w: Vec<f32>,
    pub ffn: MoeFfn,
}

pub enum Block {
    DeltaNet(Box<DeltaNetBlock>),
    Attention(Box<AttentionBlock>),
}

pub struct Qwen36MoeWeights {
    pub token_embd: QuantWeight,
    pub output_norm_w: Vec<f32>,
    pub lm_head: QuantWeight,
    pub blocks: Vec<Block>,
}

/// Split a 3D `[n_expert, n_rows_per, n_cols_per]` stacked tensor into
/// per-expert QuantWeight views. Assumes per-expert byte stride is
/// `total_len / n_expert` — true for K-quants and Q8_0 where each
/// expert's element count is a multiple of `block_size`.
fn split_experts(
    full: &QuantWeight, n_expert: usize, n_rows_per: usize, n_cols_per: usize,
) -> Vec<QuantWeight> {
    let total = full.bytes().len();
    assert!(total % n_expert == 0, "expert stack bytes {} not divisible by n_expert {}", total, n_expert);
    let bpe = total / n_expert;
    (0..n_expert).map(|e| full.sub_view(e * bpe, bpe, n_rows_per, n_cols_per)).collect()
}

fn load_moe_ffn(
    mmap: &Arc<Mmap>, gguf: &GgufFile, cfg: &Qwen36MoeConfig, prefix: &str,
) -> Result<MoeFfn, String> {
    let bytes: &[u8] = &mmap[..];
    let dm = cfg.d_model;
    let ne = cfg.n_expert;
    let nf_e = cfg.n_ff_exp;
    let nf_s = cfg.n_ff_shexp;

    let gate_inp = QuantWeight::from_gguf(
        Arc::clone(mmap), gguf, &format!("{prefix}.ffn_gate_inp.weight"), ne, dm,
    ).map_err(|e| format!("{prefix}.ffn_gate_inp: {e}"))?;
    // Stacked expert tensors: numel = n_expert × n_rows_per × n_cols_per.
    let gate_stack = QuantWeight::from_gguf(
        Arc::clone(mmap), gguf, &format!("{prefix}.ffn_gate_exps.weight"), ne * nf_e, dm,
    ).map_err(|e| format!("{prefix}.ffn_gate_exps: {e}"))?;
    let up_stack = QuantWeight::from_gguf(
        Arc::clone(mmap), gguf, &format!("{prefix}.ffn_up_exps.weight"), ne * nf_e, dm,
    ).map_err(|e| format!("{prefix}.ffn_up_exps: {e}"))?;
    let down_stack = QuantWeight::from_gguf(
        Arc::clone(mmap), gguf, &format!("{prefix}.ffn_down_exps.weight"), ne * dm, nf_e,
    ).map_err(|e| format!("{prefix}.ffn_down_exps: {e}"))?;
    let gate_exps = split_experts(&gate_stack, ne, nf_e, dm);
    let up_exps = split_experts(&up_stack, ne, nf_e, dm);
    let down_exps = split_experts(&down_stack, ne, dm, nf_e);

    let shexp_gate_inp = read_tensor_f32(bytes, gguf, &format!("{prefix}.ffn_gate_inp_shexp.weight"))
        .map_err(|e| format!("{prefix}.ffn_gate_inp_shexp: {e:?}"))?;
    let shexp_gate = QuantWeight::from_gguf(
        Arc::clone(mmap), gguf, &format!("{prefix}.ffn_gate_shexp.weight"), nf_s, dm,
    ).map_err(|e| format!("{prefix}.ffn_gate_shexp: {e}"))?;
    let shexp_up = QuantWeight::from_gguf(
        Arc::clone(mmap), gguf, &format!("{prefix}.ffn_up_shexp.weight"), nf_s, dm,
    ).map_err(|e| format!("{prefix}.ffn_up_shexp: {e}"))?;
    let shexp_down = QuantWeight::from_gguf(
        Arc::clone(mmap), gguf, &format!("{prefix}.ffn_down_shexp.weight"), dm, nf_s,
    ).map_err(|e| format!("{prefix}.ffn_down_shexp: {e}"))?;

    Ok(MoeFfn {
        gate_inp, gate_exps, up_exps, down_exps,
        shexp_gate_inp, shexp_gate, shexp_up, shexp_down,
    })
}

fn load_weights(
    mmap: &Arc<Mmap>, gguf: &GgufFile, cfg: &Qwen36MoeConfig,
) -> Result<Qwen36MoeWeights, String> {
    let bytes: &[u8] = &mmap[..];
    let token_embd = QuantWeight::from_gguf(
        Arc::clone(mmap), gguf, "token_embd.weight", cfg.vocab, cfg.d_model,
    ).map_err(|e| format!("token_embd: {e}"))?;
    let output_norm_w = read_tensor_f32(bytes, gguf, "output_norm.weight")
        .map_err(|e| format!("output_norm: {e:?}"))?;
    let has_output = gguf.tensors.iter().any(|t| t.name == "output.weight");
    let lm_head = if has_output {
        QuantWeight::from_gguf(
            Arc::clone(mmap), gguf, "output.weight", cfg.vocab, cfg.d_model,
        ).map_err(|e| format!("output: {e}"))?
    } else {
        // Tied: lm_head ← token_embd.
        QuantWeight::from_gguf(
            Arc::clone(mmap), gguf, "token_embd.weight", cfg.vocab, cfg.d_model,
        ).map_err(|e| format!("token_embd (tied lm_head): {e}"))?
    };

    let mut blocks = Vec::with_capacity(cfg.n_layers);
    for il in 0..cfg.n_layers {
        let prefix = format!("blk.{il}");
        let attn_norm_w = read_tensor_f32(bytes, gguf, &format!("{prefix}.attn_norm.weight"))
            .map_err(|e| format!("{prefix}.attn_norm: {e:?}"))?;
        let attn_post_norm_w = read_tensor_f32(bytes, gguf, &format!("{prefix}.post_attention_norm.weight"))
            .map_err(|e| format!("{prefix}.post_attention_norm: {e:?}"))?;
        let ffn = load_moe_ffn(mmap, gguf, cfg, &prefix)?;

        if cfg.is_attention_layer(il) {
            let q_out = 2 * cfg.n_q_heads * cfg.d_head;
            let wq = QuantWeight::from_gguf(
                Arc::clone(mmap), gguf, &format!("{prefix}.attn_q.weight"), q_out, cfg.d_model,
            ).map_err(|e| format!("{prefix}.attn_q: {e}"))?;
            let kv_out = cfg.n_kv_heads * cfg.d_head;
            let wk = QuantWeight::from_gguf(
                Arc::clone(mmap), gguf, &format!("{prefix}.attn_k.weight"), kv_out, cfg.d_model,
            ).map_err(|e| format!("{prefix}.attn_k: {e}"))?;
            let wv = QuantWeight::from_gguf(
                Arc::clone(mmap), gguf, &format!("{prefix}.attn_v.weight"), kv_out, cfg.d_model,
            ).map_err(|e| format!("{prefix}.attn_v: {e}"))?;
            let wo = QuantWeight::from_gguf(
                Arc::clone(mmap), gguf, &format!("{prefix}.attn_output.weight"), cfg.d_model, cfg.n_q_heads * cfg.d_head,
            ).map_err(|e| format!("{prefix}.attn_output: {e}"))?;
            let q_norm_w = read_tensor_f32(bytes, gguf, &format!("{prefix}.attn_q_norm.weight"))
                .map_err(|e| format!("{prefix}.attn_q_norm: {e:?}"))?;
            let k_norm_w = read_tensor_f32(bytes, gguf, &format!("{prefix}.attn_k_norm.weight"))
                .map_err(|e| format!("{prefix}.attn_k_norm: {e:?}"))?;
            blocks.push(Block::Attention(Box::new(AttentionBlock {
                attn_norm_w, attn_post_norm_w, wq, wk, wv, wo, q_norm_w, k_norm_w, ffn,
            })));
        } else {
            let conv_dim = cfg.conv_dim();
            let value_dim = cfg.value_dim();
            let wqkv = QuantWeight::from_gguf(
                Arc::clone(mmap), gguf, &format!("{prefix}.attn_qkv.weight"), conv_dim, cfg.d_model,
            ).map_err(|e| format!("{prefix}.attn_qkv: {e}"))?;
            let wqkv_gate = QuantWeight::from_gguf(
                Arc::clone(mmap), gguf, &format!("{prefix}.attn_gate.weight"), value_dim, cfg.d_model,
            ).map_err(|e| format!("{prefix}.attn_gate: {e}"))?;
            let ssm_conv1d_w = read_tensor_f32(bytes, gguf, &format!("{prefix}.ssm_conv1d.weight"))
                .map_err(|e| format!("{prefix}.ssm_conv1d: {e:?}"))?;
            let ssm_dt_bias = read_tensor_f32(bytes, gguf, &format!("{prefix}.ssm_dt.bias"))
                .map_err(|e| format!("{prefix}.ssm_dt.bias: {e:?}"))?;
            let ssm_a = read_tensor_f32(bytes, gguf, &format!("{prefix}.ssm_a"))
                .map_err(|e| format!("{prefix}.ssm_a: {e:?}"))?;
            let ssm_alpha = QuantWeight::from_gguf(
                Arc::clone(mmap), gguf, &format!("{prefix}.ssm_alpha.weight"), cfg.n_v_heads, cfg.d_model,
            ).map_err(|e| format!("{prefix}.ssm_alpha: {e}"))?;
            let ssm_beta = QuantWeight::from_gguf(
                Arc::clone(mmap), gguf, &format!("{prefix}.ssm_beta.weight"), cfg.n_v_heads, cfg.d_model,
            ).map_err(|e| format!("{prefix}.ssm_beta: {e}"))?;
            let ssm_norm_w = read_tensor_f32(bytes, gguf, &format!("{prefix}.ssm_norm.weight"))
                .map_err(|e| format!("{prefix}.ssm_norm: {e:?}"))?;
            let ssm_out = QuantWeight::from_gguf(
                Arc::clone(mmap), gguf, &format!("{prefix}.ssm_out.weight"), cfg.d_model, value_dim,
            ).map_err(|e| format!("{prefix}.ssm_out: {e}"))?;
            blocks.push(Block::DeltaNet(Box::new(DeltaNetBlock {
                attn_norm_w, attn_post_norm_w, wqkv, wqkv_gate,
                ssm_conv1d_w, ssm_dt_bias, ssm_a, ssm_alpha, ssm_beta, ssm_norm_w, ssm_out, ffn,
            })));
        }
    }

    Ok(Qwen36MoeWeights { token_embd, output_norm_w, lm_head, blocks })
}

// ── State ─────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct DeltaNetState {
    pub conv_state: Vec<f32>,
    pub ssm_state: Vec<f64>,
}

#[derive(Clone)]
pub struct AttentionState {
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
}

#[derive(Clone)]
pub enum BlockState {
    DeltaNet(DeltaNetState),
    Attention(AttentionState),
}

#[derive(Clone)]
pub struct Qwen36MoeState {
    pub blocks: Vec<BlockState>,
    pub pos: usize,
}

impl Qwen36MoeState {
    pub fn new(cfg: &Qwen36MoeConfig, layer_kinds: &[LayerKind]) -> Self {
        let conv_dim = cfg.conv_dim();
        let blocks = layer_kinds.iter().map(|k| match k {
            LayerKind::GatedDeltaNet => BlockState::DeltaNet(DeltaNetState {
                conv_state: vec![0.0f32; cfg.d_conv * conv_dim],
                ssm_state: vec![0.0f64; cfg.n_v_heads * cfg.d_state * cfg.d_state],
            }),
            LayerKind::GatedAttention => BlockState::Attention(AttentionState {
                k_cache: Vec::new(),
                v_cache: Vec::new(),
            }),
        }).collect();
        Self { blocks, pos: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind { GatedAttention, GatedDeltaNet }

// ── Tracing (env-gated) ───────────────────────────────────────────────

#[inline]
fn trace_enabled() -> bool {
    std::env::var("RCE_QWEN36_MOE_TRACE").is_ok()
}

fn trace_tensor(name: &str, x: &[f32]) {
    if !trace_enabled() { return; }
    let sum: f64 = x.iter().map(|v| *v as f64).sum();
    let n = x.len().min(3);
    let mut first = [0f32; 3];
    first[..n].copy_from_slice(&x[..n]);
    eprintln!(
        "rce_trace: {name:<36} sum={sum:+.6e}  first=[{:.4}, {:.4}, {:.4}]  len={}",
        first[0], first[1], first[2], x.len(),
    );
}

// ── Forward helpers ───────────────────────────────────────────────────

fn qmm_1d(a: &[f32], w: &QuantWeight) -> Vec<f32> {
    let mut out = vec![0.0f32; w.n_rows];
    rce_op_qmm::mm_1d(a, w, &mut out);
    out
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w.iter()).map(|(v, ww)| v * inv * ww).collect()
}

fn rms_norm_per_head(x: &mut [f32], w: &[f32], n_heads: usize, head_dim: usize, eps: f32) {
    debug_assert_eq!(x.len(), n_heads * head_dim);
    for h in 0..n_heads {
        let off = h * head_dim;
        let slice = &mut x[off..off + head_dim];
        let ms: f32 = slice.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for (v, ww) in slice.iter_mut().zip(w.iter()) { *v = *v * inv * *ww; }
    }
}

fn l2_norm_per_head(x: &mut [f32], n_heads: usize, head_dim: usize, eps: f32) {
    debug_assert_eq!(x.len(), n_heads * head_dim);
    for h in 0..n_heads {
        let off = h * head_dim;
        let slice = &mut x[off..off + head_dim];
        let ms: f32 = slice.iter().map(|v| v * v).sum::<f32>();
        let inv = 1.0 / (ms + eps).sqrt();
        for v in slice.iter_mut() { *v *= inv; }
    }
}

#[inline] fn silu(x: f32) -> f32 {
    if x >= 0.0 { x / (1.0 + (-x).exp()) } else { let e = x.exp(); x * e / (1.0 + e) }
}
#[inline] fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}
#[inline] fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) }
}

fn conv1d_step(
    x_new: &[f32], weight: &[f32], state: &mut Vec<f32>, conv_dim: usize, d_conv: usize,
) -> Vec<f32> {
    debug_assert_eq!(x_new.len(), conv_dim);
    debug_assert_eq!(weight.len(), d_conv * conv_dim);
    debug_assert_eq!(state.len(), d_conv * conv_dim);
    let zero_bias = vec![0.0f32; conv_dim];
    let (y, new_state) = rce_op_conv1d::conv1d_causal_step(
        x_new, weight, &zero_bias, state, conv_dim, d_conv,
    );
    *state = new_state;
    y
}

fn multi_axis_rope_inplace(
    x: &mut [f32], head_dim: usize, sections: &[u32; 4], base: f32, pos: usize,
) {
    let mut section_start = 0usize;
    for &sec_pairs in sections.iter() {
        let sec_pairs = sec_pairs as usize;
        if sec_pairs == 0 { continue; }
        let rope_dim_here = sec_pairs * 2;
        let half = sec_pairs;
        let log_base = base.ln();
        let total_pairs: usize = sections.iter().map(|s| *s as usize).sum();
        let total_rope_dim = 2 * total_pairs;
        for i in 0..half {
            let global_pair_idx = (section_start / 2) + i;
            let inv_freq = (log_base * (2.0 * global_pair_idx as f32) / total_rope_dim as f32).exp().recip();
            let theta = pos as f32 * inv_freq;
            let (s, c) = theta.sin_cos();
            let a = x[section_start + i];
            let b = x[section_start + i + half];
            x[section_start + i] = a * c - b * s;
            x[section_start + i + half] = a * s + b * c;
        }
        let _ = rope_dim_here;
        section_start += sec_pairs * 2;
        if section_start >= head_dim { break; }
    }
}

// ── DeltaNet forward (mixer; identical to dense Qwen 3.6 path) ────────

fn delta_net_forward(
    x_in: &[f32], blk: &DeltaNetBlock, state: &mut DeltaNetState, cfg: &Qwen36MoeConfig, il: usize,
) -> Vec<f32> {
    let trace = il == 0;
    let key_dim = cfg.key_dim();
    let value_dim = cfg.value_dim();
    let conv_dim = cfg.conv_dim();
    let n_v = cfg.n_v_heads;
    let n_k = cfg.n_k_heads;
    let dh = cfg.d_state;

    let qkv_mixed = qmm_1d(x_in, &blk.wqkv);
    if trace { trace_tensor("linear_attn_qkv_mixed-0", &qkv_mixed); }
    let z = qmm_1d(x_in, &blk.wqkv_gate);
    if trace { trace_tensor("z-0", &z); }

    let beta_raw = qmm_1d(x_in, &blk.ssm_beta);
    let mut beta = vec![0.0f32; n_v];
    for i in 0..n_v { beta[i] = sigmoid(beta_raw[i]); }
    if trace { trace_tensor("beta-0", &beta); }
    let alpha_raw = qmm_1d(x_in, &blk.ssm_alpha);
    let mut decay_log = vec![0.0f32; n_v];
    for i in 0..n_v {
        let a_biased = alpha_raw[i] + blk.ssm_dt_bias[i];
        let a_sp = softplus(a_biased);
        decay_log[i] = a_sp * blk.ssm_a[i];
    }

    let conv_out = conv1d_step(&qkv_mixed, &blk.ssm_conv1d_w, &mut state.conv_state, conv_dim, cfg.d_conv);
    let conv_silu: Vec<f32> = conv_out.iter().map(|v| silu(*v)).collect();
    if trace { trace_tensor("conv_output_silu-0", &conv_silu); }

    let q_slice = &conv_silu[0..key_dim];
    let k_slice = &conv_silu[key_dim..2 * key_dim];
    let v_slice = &conv_silu[2 * key_dim..conv_dim];
    let mut q_buf = q_slice.to_vec();
    let mut k_buf = k_slice.to_vec();
    l2_norm_per_head(&mut q_buf, n_k, dh, cfg.eps);
    l2_norm_per_head(&mut k_buf, n_k, dh, cfg.eps);

    let q_scale = 1.0f64 / (dh as f64).sqrt();
    let mut y = vec![0.0f32; value_dim];
    // The V heads are independent: each owns a disjoint `dh×dh` state
    // block and writes a disjoint `dh`-slice of y. Run them across cores.
    use rayon::prelude::*;
    state.ssm_state
        .par_chunks_mut(dh * dh)
        .zip(y.par_chunks_mut(dh))
        .enumerate()
        .for_each(|(v_h, (s, y_h))| {
            // GQA: tile mapping `k_h = v_h % n_k` (Qwen 3.6 bug-fix convention).
            let k_h = v_h % n_k;
            let q_v = &q_buf[k_h * dh..(k_h + 1) * dh];
            let k_v = &k_buf[k_h * dh..(k_h + 1) * dh];
            let v_v = &v_slice[v_h * dh..(v_h + 1) * dh];
            let beta_v = beta[v_h] as f64;
            let decay_v = (decay_log[v_h] as f64).exp();

            // Fused decay + retrieve: r[v] = Σ_k (decay·S[k,v])·key[k],
            // writing the decayed state back in the same pass.
            let mut r = vec![0.0f64; dh];
            for k_idx in 0..dh {
                let k_kv = k_v[k_idx] as f64;
                let row_off = k_idx * dh;
                for v_idx in 0..dh {
                    let decayed = s[row_off + v_idx] * decay_v;
                    s[row_off + v_idx] = decayed;
                    r[v_idx] += decayed * k_kv;
                }
            }
            let mut delta_v = vec![0.0f64; dh];
            for v_idx in 0..dh { delta_v[v_idx] = beta_v * (v_v[v_idx] as f64 - r[v_idx]); }
            for k_idx in 0..dh {
                let k_kv = k_v[k_idx] as f64;
                let row_off = k_idx * dh;
                for v_idx in 0..dh { s[row_off + v_idx] += k_kv * delta_v[v_idx]; }
            }
            for v_idx in 0..dh {
                let mut acc = 0.0f64;
                for k_idx in 0..dh {
                    acc += s[k_idx * dh + v_idx] * (q_v[k_idx] as f64 * q_scale);
                }
                y_h[v_idx] = acc as f32;
            }
        });

    rms_norm_per_head(&mut y, &blk.ssm_norm_w, n_v, dh, cfg.eps);
    for (i, yi) in y.iter_mut().enumerate() { *yi *= silu(z[i]); }
    if trace { trace_tensor("final_output-0", &y); }
    qmm_1d(&y, &blk.ssm_out)
}

// ── Gated Attention forward (same as dense Qwen 3.6) ──────────────────

fn gated_attention_forward(
    x_in: &[f32], blk: &AttentionBlock, state: &mut AttentionState,
    cfg: &Qwen36MoeConfig, pos: usize,
) -> Vec<f32> {
    let nq = cfg.n_q_heads;
    let nkv = cfg.n_kv_heads;
    let dh = cfg.d_head;
    let q_per_kv = cfg.q_per_kv();

    let q_full = qmm_1d(x_in, &blk.wq);
    let mut q = vec![0.0f32; nq * dh];
    let mut gate = vec![0.0f32; nq * dh];
    for h in 0..nq {
        let src = h * 2 * dh;
        q[h * dh..(h + 1) * dh].copy_from_slice(&q_full[src..src + dh]);
        gate[h * dh..(h + 1) * dh].copy_from_slice(&q_full[src + dh..src + 2 * dh]);
    }
    rms_norm_per_head(&mut q, &blk.q_norm_w, nq, dh, cfg.eps);

    let mut k_new = qmm_1d(x_in, &blk.wk);
    let v_new = qmm_1d(x_in, &blk.wv);
    rms_norm_per_head(&mut k_new, &blk.k_norm_w, nkv, dh, cfg.eps);

    for h in 0..nq {
        let off = h * dh;
        multi_axis_rope_inplace(
            &mut q[off..off + dh], dh, &cfg.rope_dimension_sections, cfg.rope_freq_base, pos,
        );
    }
    for h in 0..nkv {
        let off = h * dh;
        multi_axis_rope_inplace(
            &mut k_new[off..off + dh], dh, &cfg.rope_dimension_sections, cfg.rope_freq_base, pos,
        );
    }

    state.k_cache.extend_from_slice(&k_new);
    state.v_cache.extend_from_slice(&v_new);
    let seq_len = state.k_cache.len() / (nkv * dh);

    let scale = cfg.attention_scale;
    let mut attn = vec![0.0f32; nq * dh];
    for h in 0..nq {
        let kv_h = h / q_per_kv;
        let q_h = &q[h * dh..(h + 1) * dh];
        let out_h = rce_op_attn::fused_attend_head(
            q_h, &state.k_cache, &state.v_cache, kv_h, seq_len, nkv * dh, dh, scale,
        );
        attn[h * dh..(h + 1) * dh].copy_from_slice(&out_h);
    }
    rce_op_attn::apply_attention_gate(&mut attn, &gate);
    qmm_1d(&attn, &blk.wo)
}

// ── MoE FFN forward (routed top-K + sigmoid-gated shared expert) ──────

/// One expert's SwiGLU using **serial** matmuls. Called inside a
/// rayon par-over-experts region, so the matmuls must NOT re-enter
/// rayon (per-expert matrices are too small to parallelize — the
/// fork/join overhead cancels the win; see rce-op-qmm::mm_1d_serial).
fn swiglu_expert_serial(x_in: &[f32], gate_w: &QuantWeight, up_w: &QuantWeight, down_w: &QuantWeight, n_ff: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; n_ff];
    rce_op_qmm::mm_1d_serial(x_in, gate_w, &mut g);
    let mut u = vec![0.0f32; n_ff];
    rce_op_qmm::mm_1d_serial(x_in, up_w, &mut u);
    for i in 0..n_ff { g[i] = silu(g[i]) * u[i]; }
    let mut out = vec![0.0f32; down_w.n_rows];
    rce_op_qmm::mm_1d_serial(&g, down_w, &mut out);
    out
}

fn moe_ffn_forward(x_in: &[f32], ffn: &MoeFfn, cfg: &Qwen36MoeConfig) -> Vec<f32> {
    use rayon::prelude::*;
    // Routed experts: softmax → top-K (with normalization).
    let decision = moe_router(x_in, &ffn.gate_inp, cfg.n_expert_used, true);
    let k = decision.expert_indices.len();

    // ONE coarse rayon region over the K routed experts + the shared
    // expert (job index k). Each task runs its SwiGLU serially. This is
    // ~9 big tasks/layer (one fork/join) vs the old ~27 tiny per-matmul
    // par_iters that didn't actually parallelize.
    let outs: Vec<Vec<f32>> = (0..=k).into_par_iter().map(|job| {
        if job < k {
            let e = decision.expert_indices[job];
            swiglu_expert_serial(x_in, &ffn.gate_exps[e], &ffn.up_exps[e], &ffn.down_exps[e], cfg.n_ff_exp)
        } else {
            swiglu_expert_serial(x_in, &ffn.shexp_gate, &ffn.shexp_up, &ffn.shexp_down, cfg.n_ff_shexp)
        }
    }).collect();

    let mut out = vec![0.0f32; cfg.d_model];
    for (j, &w) in decision.expert_weights.iter().enumerate() {
        for i in 0..cfg.d_model { out[i] += w * outs[j][i]; }
    }
    // Shared expert: sigmoid scalar gate `shexp_gate_inp · x`.
    let mut g: f32 = 0.0;
    for i in 0..cfg.d_model { g += ffn.shexp_gate_inp[i] * x_in[i]; }
    let alpha = sigmoid(g);
    let sh = &outs[k];
    for i in 0..cfg.d_model { out[i] += alpha * sh[i]; }
    out
}

// ── forward_token ─────────────────────────────────────────────────────

pub fn forward_token(
    w: &Qwen36MoeWeights, cfg: &Qwen36MoeConfig, state: &mut Qwen36MoeState, token_id: i32,
) -> Vec<f32> {
    let dm = cfg.d_model;
    let mut x: Vec<f32> = w.token_embd.dequant_row(token_id as usize);
    trace_tensor("model.input_embed", &x);
    let pos = state.pos;
    let perf = std::env::var("RCE_QWEN36_MOE_PERF").is_ok();
    let (mut t_dn, mut t_at, mut t_moe) = (0.0f64, 0.0f64, 0.0f64);
    for (il, block) in w.blocks.iter().enumerate() {
        let normed = rms_norm(&x, match block {
            Block::DeltaNet(b) => &b.attn_norm_w,
            Block::Attention(b) => &b.attn_norm_w,
        }, cfg.eps);

        let tb = std::time::Instant::now();
        let attn_out = match (block, &mut state.blocks[il]) {
            (Block::DeltaNet(b), BlockState::DeltaNet(s)) => delta_net_forward(&normed, b, s, cfg, il),
            (Block::Attention(b), BlockState::Attention(s)) => gated_attention_forward(&normed, b, s, cfg, pos),
            _ => panic!("block/state kind mismatch at layer {il}"),
        };
        if perf {
            let dt = tb.elapsed().as_secs_f64();
            match block { Block::DeltaNet(_) => t_dn += dt, Block::Attention(_) => t_at += dt }
        }
        for i in 0..dm { x[i] += attn_out[i]; }

        let ffn_in = rms_norm(&x, match block {
            Block::DeltaNet(b) => &b.attn_post_norm_w,
            Block::Attention(b) => &b.attn_post_norm_w,
        }, cfg.eps);
        let ffn = match block {
            Block::DeltaNet(b) => &b.ffn,
            Block::Attention(b) => &b.ffn,
        };
        let tf = std::time::Instant::now();
        let ffn_out = moe_ffn_forward(&ffn_in, ffn, cfg);
        if perf { t_moe += tf.elapsed().as_secs_f64(); }
        for i in 0..dm { x[i] += ffn_out[i]; }
    }
    if perf {
        eprintln!("[qwen36-moe perf pos={pos}] deltanet={:.1}ms attn={:.1}ms moe={:.1}ms",
            t_dn * 1e3, t_at * 1e3, t_moe * 1e3);
    }
    let final_normed = rms_norm(&x, &w.output_norm_w, cfg.eps);
    let logits = qmm_1d(&final_normed, &w.lm_head);
    state.pos += 1;
    logits
}

// ── Runtime ───────────────────────────────────────────────────────────

pub struct Qwen36MoeRuntime {
    pub cfg: Qwen36MoeConfig,
    pub gguf: GgufFile,
    pub weights: Qwen36MoeWeights,
    pub layer_kinds: Vec<LayerKind>,
    _mmap: Arc<Mmap>,
}

impl Qwen36MoeRuntime {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| format!("open: {e}"))?;
        let mmap = Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?);
        // WillNeed (not Sequential): MoE decode re-reads different experts
        // each token; Sequential's drop-behind eviction forces cold
        // re-faults. WillNeed keeps the model page-resident (RAM ≫ model).
        let _ = mmap.advise(memmap2::Advice::WillNeed);
        let gguf = GgufFile::parse(&mmap[..]).map_err(|e| format!("parse: {e:?}"))?;
        let cfg = Qwen36MoeConfig::from_gguf(&gguf)?;

        let mut layer_kinds = Vec::with_capacity(cfg.n_layers);
        for il in 0..cfg.n_layers {
            let has_ssm = gguf.tensors.iter().any(|t| t.name == format!("blk.{il}.ssm_a"));
            let has_split_qkv = gguf.tensors.iter().any(|t| t.name == format!("blk.{il}.attn_q.weight"));
            let kind = match (has_ssm, has_split_qkv) {
                (true, false) => LayerKind::GatedDeltaNet,
                (false, true) => LayerKind::GatedAttention,
                _ => return Err(format!("layer {il}: ambiguous tensors (ssm={has_ssm}, split_qkv={has_split_qkv})")),
            };
            layer_kinds.push(kind);
        }

        let weights = load_weights(&mmap, &gguf, &cfg)?;
        let n_attn = layer_kinds.iter().filter(|k| **k == LayerKind::GatedAttention).count();
        let n_delta = layer_kinds.iter().filter(|k| **k == LayerKind::GatedDeltaNet).count();
        eprintln!(
            "qwen36-moe: d_model={} vocab={} n_layers={} ({n_attn} GA + {n_delta} GDN)  n_expert={} top_k={} n_ff_exp={} n_ff_shexp={}",
            cfg.d_model, cfg.vocab, cfg.n_layers, cfg.n_expert, cfg.n_expert_used, cfg.n_ff_exp, cfg.n_ff_shexp,
        );
        Ok(Self { cfg, gguf, weights, layer_kinds, _mmap: mmap })
    }

    pub fn new_state(&self) -> Qwen36MoeState { Qwen36MoeState::new(&self.cfg, &self.layer_kinds) }

    pub fn forward_logits(&self, state: &mut Qwen36MoeState, token_id: i32) -> Vec<f32> {
        forward_token(&self.weights, &self.cfg, state, token_id)
    }

    pub fn tokenizer(&self) -> Tokenizer<'_> {
        Tokenizer::from_gguf(&self.gguf).expect("tokenizer")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Qwen36MoeGenConfig {
    pub max_new_tokens: usize,
    pub sample: SampleConfig,
    pub rep_window: usize,
}
impl Default for Qwen36MoeGenConfig {
    fn default() -> Self {
        Self { max_new_tokens: 24, sample: SampleConfig::chat(0x9E36_36AB_3500_0001), rep_window: 64 }
    }
}

impl Qwen36MoeRuntime {
    pub fn generate(
        &self, prompt: &str, gcfg: &Qwen36MoeGenConfig, mut on_token: impl FnMut(&str),
    ) -> Result<String, String> {
        let tk = self.tokenizer();
        let is_gpt2 = self.gguf.metadata.iter()
            .find(|kv| kv.key == "tokenizer.ggml.model")
            .map(|kv| matches!(&kv.value, GgufValue::String(s) if s == "gpt2"))
            .unwrap_or(false);
        let add_bos = self.gguf.metadata.iter()
            .find(|kv| kv.key == "tokenizer.ggml.add_bos_token")
            .map(|kv| matches!(&kv.value, GgufValue::Bool(b) if *b))
            .unwrap_or(false);
        let prompt_ids = if is_gpt2 {
            tk.encode_gpt2(prompt, add_bos)
        } else {
            tk.encode(prompt, add_bos)
        };
        if prompt_ids.is_empty() { return Err("empty prompt".into()); }
        let eos_id = tk.eos_token_id.unwrap_or(-1);

        let mut state = self.new_state();
        let mut history: Vec<i32> = Vec::with_capacity(prompt_ids.len() + gcfg.max_new_tokens);
        let mut step: u64 = 0;
        let seed_for = |s: u64| {
            let mut c = gcfg.sample;
            c.seed = gcfg.sample.seed ^ s.wrapping_mul(0x9E37_79B9);
            c
        };
        let mut next: i32 = -1;
        for (i, &tok) in prompt_ids.iter().enumerate() {
            let logits = self.forward_logits(&mut state, tok);
            history.push(tok);
            let recent = &history[history.len().saturating_sub(gcfg.rep_window)..];
            let pred = sample_with_history(&logits, &seed_for(step), recent);
            step += 1;
            if i == prompt_ids.len() - 1 { next = pred; }
        }
        let mut out_ids = Vec::new();
        for _ in 0..gcfg.max_new_tokens {
            if next == eos_id { break; }
            out_ids.push(next);
            history.push(next);
            let text = tk.decode_sequence(&[next], false);
            on_token(&text);
            let logits = self.forward_logits(&mut state, next);
            let recent = &history[history.len().saturating_sub(gcfg.rep_window)..];
            next = sample_with_history(&logits, &seed_for(step), recent);
            step += 1;
        }
        Ok(tk.decode_sequence(&out_ids, true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const MODEL: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/qwen3.6-35b-a3b-gguf/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf";

    #[test]
    fn qwen36_moe_load_sanity() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("qwen36-moe: GGUF not mounted — skipping");
            return;
        }
        let rt = Qwen36MoeRuntime::load(MODEL).expect("load Qwen 3.6-35B-A3B");
        let c = &rt.cfg;
        assert_eq!(c.d_model, 2048);
        assert_eq!(c.n_layers, 40);
        assert_eq!(c.n_q_heads, 16);
        assert_eq!(c.n_kv_heads, 2);
        assert_eq!(c.d_head, 256);
        assert_eq!(c.full_attention_interval, 4);
        assert_eq!(c.n_expert, 256);
        assert_eq!(c.n_expert_used, 8);
        assert_eq!(c.n_ff_exp, 512);
        assert_eq!(c.n_ff_shexp, 512);
        assert_eq!(c.n_k_heads, 16);
        assert_eq!(c.n_v_heads, 32);
        assert_eq!(c.d_state, 128);
        let n_attn = rt.layer_kinds.iter().filter(|k| **k == LayerKind::GatedAttention).count();
        let n_delta = rt.layer_kinds.iter().filter(|k| **k == LayerKind::GatedDeltaNet).count();
        assert_eq!(n_attn, 10);
        assert_eq!(n_delta, 30);
    }

    #[test]
    fn qwen36_moe_one_forward_finite() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("qwen36-moe: GGUF not mounted — skipping");
            return;
        }
        let rt = Qwen36MoeRuntime::load(MODEL).expect("load");
        let mut state = rt.new_state();
        let logits = rt.forward_logits(&mut state, 1);
        assert_eq!(logits.len(), rt.cfg.vocab);
        let n_finite = logits.iter().filter(|v| v.is_finite()).count();
        assert_eq!(n_finite, logits.len(), "expected all-finite logits");
        let max_abs = logits.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        eprintln!("qwen36-moe one forward → max|logit|={max_abs:.3}");
        assert!(max_abs > 0.0);
        assert!(max_abs < 1e6, "logits exploded ({max_abs})");
    }

    #[test]
    fn qwen36_moe_generate_proves_coherence() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("qwen36-moe: GGUF not mounted — skipping");
            return;
        }
        let rt = Qwen36MoeRuntime::load(MODEL).expect("load");
        let cfg = Qwen36MoeGenConfig {
            max_new_tokens: 24,
            sample: SampleConfig::GREEDY,
            rep_window: 64,
        };
        let prompt = "The capital of France is";
        let mut chunks: Vec<String> = Vec::new();
        let t0 = std::time::Instant::now();
        let _reply = rt.generate(prompt, &cfg, |tok| {
            eprint!("{tok}");
            chunks.push(tok.to_string());
        }).expect("generate");
        let dt = t0.elapsed();
        eprintln!();
        let full = chunks.concat();
        eprintln!(
            "qwen36-moe generate OK: prompt={prompt:?} → continuation={full:?}  [{} tok in {dt:?} = {:.2} tok/s]",
            chunks.len(),
            chunks.len() as f64 / dt.as_secs_f64(),
        );
        assert!(!full.is_empty(), "empty continuation");
        let alpha = full.chars().filter(|c| c.is_alphabetic()).count();
        let total = full.chars().count();
        assert!(
            alpha * 2 >= total,
            "output looks like garbage (alpha {alpha} / total {total}): {full:?}",
        );
    }
}
