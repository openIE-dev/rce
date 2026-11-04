//! Qwen 3.6-27B inference runtime on the rce substrate.
//!
//! Hybrid arch (per llama.cpp `src/models/qwen35.cpp`):
//!
//! - 64 layers. `full_attention_interval = 4`: layers where
//!   `(il + 1) % 4 == 0` are **Gated Attention** (16 layers);
//!   the other 48 are **Gated DeltaNet** (linear-attention recurrent
//!   state with delta-rule update).
//! - Per-layer flow (same for both kinds):
//!   ```text
//!   attn_out = mixer(rms_norm(x, attn_norm))   // DeltaNet or Gated Attn
//!   x = x + attn_out
//!   ffn_in = rms_norm(x, attn_post_norm)       // post-attn norm doubles as pre-FFN norm
//!   ffn_out = SwiGLU(ffn_in)
//!   x = x + ffn_out
//!   ```
//! - **Gated Attention** (16 layers): standard SDPA with Q-gate split (Q
//!   projection is 2× wide; first half is Q proper, second half is
//!   sigmoid-gate). Per-head Q-norm + K-norm. M-RoPE with
//!   `dimension_sections=[11, 11, 10, 0]` (3 active sections × 11/11/10
//!   dim pairs = 32 freq pairs = 64 RoPE dims). GQA: 24 Q / 4 KV
//!   heads, d_head=256. Gate = `sigmoid(W_gate @ x)` from the second
//!   half of Q applied per-head after SDPA.
//! - **Gated DeltaNet** (48 layers): linear-attention recurrent state
//!   with delta-rule update. Per layer:
//!   1. Fused QKV projection `wqkv` (d_model → 10240 = 2*key_dim + value_dim)
//!   2. Z-gate projection `wqkv_gate` (d_model → 6144)
//!   3. Per-V-head gates: `beta = sigmoid(ssm_beta @ x)` ∈ [48];
//!      `alpha_sp = softplus(ssm_alpha @ x + ssm_dt.bias) * ssm_a` ∈ [48]
//!   4. Depthwise conv1d (kernel=4) over the 10240-dim qkv_mixed;
//!      SiLU activation
//!   5. Split convolved into q[16×128], k[16×128], v[48×128];
//!      L2-norm q and k; GQA-repeat q,k to 48 heads (each K head
//!      serves 3 V heads)
//!   6. Recurrent state per V head v:
//!      `S_v ← exp(alpha_sp_v) * S_v + β_v · (v_v - S_v @ k_v) k_v^T`
//!      `y_v = S_v @ q_v`
//!   7. Per-head RMSnorm of y; multiply by SiLU(z) per channel
//!   8. Output projection `ssm_out` (6144 → d_model)
//!
//! This is the initial scalar implementation that mirrors llama.cpp's
//! `build_layer_attn_linear` step-for-step. Optimization (NEON, AMX
//! batching, fused ops) is the follow-on; correctness first.

use rce_ingest_gguf::{
    read_tensor_f32, sample_with_history, GgufFile, GgufValue, SampleConfig, Tokenizer,
};
// Re-export so wrapper crates (e.g. rce-qwen35, sharing the `qwen35` arch and
// forward pass) can configure generation without re-depending on internals.
pub use rce_ingest_gguf::SampleConfig as ExportedSampleConfig;
use rce_op_qmm::QuantWeight;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

// ── Config ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Qwen36Config {
    pub d_model: usize,
    pub d_ff: usize,
    pub n_layers: usize,
    pub vocab: usize,
    /// Gated-Attention layer dims (every 4th layer).
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub d_head: usize,
    /// DeltaNet layer dims.
    pub n_k_heads: usize,     // ssm_n_group (= 16 for 27B)
    pub n_v_heads: usize,     // ssm_dt_rank (= 48 for 27B)
    pub d_state: usize,       // ssm_d_state = head_k_dim = head_v_dim (= 128 for 27B)
    pub d_conv: usize,        // ssm_d_conv (= 4 for 27B)
    pub d_inner: usize,       // ssm_d_inner = n_v_heads × d_state (= 6144 for 27B)
    pub full_attention_interval: usize,
    pub rope_freq_base: f32,
    pub rope_dim: usize,
    pub rope_dimension_sections: [u32; 4],
    pub eps: f32,
    pub attention_scale: f32,
}

impl Qwen36Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf.metadata.iter()
            .find(|kv| kv.key == "general.architecture")
            .and_then(|kv| match &kv.value { GgufValue::String(s) => Some(s.as_str()), _ => None })
            .ok_or("missing general.architecture")?;
        if arch != "qwen35" {
            return Err(format!("expected qwen35, got {arch}"));
        }
        let u32_meta = |k: &str| -> Option<u32> {
            gguf.metadata.iter().find(|kv| kv.key == format!("qwen35.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::U32(v) => Some(*v), _ => None })
        };
        let f32_meta = |k: &str| -> Option<f32> {
            gguf.metadata.iter().find(|kv| kv.key == format!("qwen35.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::F32(v) => Some(*v), _ => None })
        };

        let d_model = u32_meta("embedding_length").ok_or("embedding_length")? as usize;
        let d_ff = u32_meta("feed_forward_length").ok_or("feed_forward_length")? as usize;
        let block_count = u32_meta("block_count").ok_or("block_count")? as usize;
        // Trailing blocks carrying `nextn.*` tensors are Multi-Token-Prediction
        // (MTP / "nextn") speculative-decoding heads, not trunk layers. The dense
        // Qwen 3.5 checkpoints append one (e.g. 4B: 32 trunk layers + 1 MTP head =
        // block_count 33). They must NOT run in the standard autoregressive pass —
        // final norm + lm_head produce logits after the last trunk layer. The 27B
        // ships no nextn block, so this subtracts 0 there.
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
        let n_v_heads = u32_meta("ssm.time_step_rank").unwrap_or(48) as usize;
        let d_state = u32_meta("ssm.state_size").unwrap_or(128) as usize;
        let d_conv = u32_meta("ssm.conv_kernel").unwrap_or(4) as usize;
        let d_inner = u32_meta("ssm.inner_size").unwrap_or((n_v_heads * d_state) as u32) as usize;
        let full_attention_interval = u32_meta("full_attention_interval").unwrap_or(4) as usize;
        let rope_freq_base = f32_meta("rope.freq_base").unwrap_or(10_000_000.0);
        let rope_dim = u32_meta("rope.dimension_count").unwrap_or(64) as usize;
        let eps = f32_meta("attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let attention_scale = f32_meta("attention.scale")
            .unwrap_or_else(|| 1.0 / (d_head as f32).sqrt());

        let rope_dimension_sections = gguf.metadata.iter()
            .find(|kv| kv.key == "qwen35.rope.dimension_sections")
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

        Ok(Qwen36Config {
            d_model, d_ff, n_layers, vocab,
            n_q_heads, n_kv_heads, d_head,
            n_k_heads, n_v_heads, d_state, d_conv, d_inner,
            full_attention_interval, rope_freq_base, rope_dim,
            rope_dimension_sections, eps, attention_scale,
        })
    }

    #[inline] pub fn is_attention_layer(&self, il: usize) -> bool {
        self.full_attention_interval > 0 && (il + 1) % self.full_attention_interval == 0
    }
    #[inline] pub fn key_dim(&self) -> usize { self.n_k_heads * self.d_state }
    #[inline] pub fn value_dim(&self) -> usize { self.n_v_heads * self.d_state }
    /// `key_dim*2 + value_dim` — width of the fused `wqkv` output and the conv1d.
    #[inline] pub fn conv_dim(&self) -> usize { self.key_dim() * 2 + self.value_dim() }
    /// Q-head GQA repeat factor for attention layers.
    #[inline] pub fn q_per_kv(&self) -> usize { self.n_q_heads / self.n_kv_heads }
    /// V-head GQA repeat factor for DeltaNet layers (each K head serves N V heads).
    #[inline] pub fn v_per_k(&self) -> usize { self.n_v_heads / self.n_k_heads }
}

// ── Weights ───────────────────────────────────────────────────────────

pub struct DeltaNetBlock {
    pub attn_norm_w: Vec<f32>,
    pub attn_post_norm_w: Vec<f32>,
    pub wqkv: QuantWeight,            // [conv_dim, d_model]
    pub wqkv_gate: QuantWeight,       // [value_dim, d_model]
    pub ssm_conv1d_w: Vec<f32>,       // [d_conv, conv_dim]
    pub ssm_dt_bias: Vec<f32>,        // [n_v_heads]
    pub ssm_a: Vec<f32>,              // [n_v_heads] — negative (already exp(-A_log))
    pub ssm_alpha: QuantWeight,       // [n_v_heads, d_model]
    pub ssm_beta: QuantWeight,        // [n_v_heads, d_model]
    pub ssm_norm_w: Vec<f32>,         // [d_state]
    pub ssm_out: QuantWeight,         // [d_model, value_dim]
    pub ffn_gate: QuantWeight,        // [d_ff, d_model]
    pub ffn_up: QuantWeight,          // [d_ff, d_model]
    pub ffn_down: QuantWeight,        // [d_model, d_ff]
}

pub struct AttentionBlock {
    pub attn_norm_w: Vec<f32>,
    pub attn_post_norm_w: Vec<f32>,
    /// Q + Q-gate combined. Shape: `[2 * n_q_heads * d_head, d_model]`.
    /// First half (per head) is Q proper, second half is the sigmoid gate.
    pub wq: QuantWeight,
    pub wk: QuantWeight,              // [n_kv_heads * d_head, d_model]
    pub wv: QuantWeight,              // [n_kv_heads * d_head, d_model]
    pub wo: QuantWeight,              // [d_model, n_q_heads * d_head]
    pub q_norm_w: Vec<f32>,           // [d_head] — per-head RMS norm weight
    pub k_norm_w: Vec<f32>,           // [d_head]
    pub ffn_gate: QuantWeight,
    pub ffn_up: QuantWeight,
    pub ffn_down: QuantWeight,
}

pub enum Block {
    DeltaNet(Box<DeltaNetBlock>),
    Attention(Box<AttentionBlock>),
}

pub struct Qwen36Weights {
    pub token_embd: QuantWeight,      // [vocab, d_model]
    pub output_norm_w: Vec<f32>,
    pub lm_head: QuantWeight,         // [vocab, d_model]
    pub blocks: Vec<Block>,
}

fn load_weights(
    mmap: &Arc<Mmap>, gguf: &GgufFile, cfg: &Qwen36Config,
) -> Result<Qwen36Weights, String> {
    let bytes: &[u8] = &mmap[..];
    let token_embd = QuantWeight::from_gguf(
        Arc::clone(mmap), gguf, "token_embd.weight", cfg.vocab, cfg.d_model,
    ).map_err(|e| format!("token_embd: {e}"))?;
    let output_norm_w = read_tensor_f32(bytes, gguf, "output_norm.weight")
        .map_err(|e| format!("output_norm: {e:?}"))?;
    // Tied embeddings: smaller dense members of the family (e.g. Qwen 3.5-4B)
    // omit `output.weight` and reuse `token_embd.weight` as the lm_head. Larger
    // checkpoints (Qwen 3.6-27B) ship an untied `output.weight`.
    let has_output = gguf.tensors.iter().any(|t| t.name == "output.weight");
    let lm_head = QuantWeight::from_gguf(
        Arc::clone(mmap),
        gguf,
        if has_output { "output.weight" } else { "token_embd.weight" },
        cfg.vocab,
        cfg.d_model,
    )
    .map_err(|e| format!("output: {e}"))?;

    let mut blocks = Vec::with_capacity(cfg.n_layers);
    for il in 0..cfg.n_layers {
        let prefix = format!("blk.{il}");
        let attn_norm_w = read_tensor_f32(bytes, gguf, &format!("{prefix}.attn_norm.weight"))
            .map_err(|e| format!("{prefix}.attn_norm: {e:?}"))?;
        let attn_post_norm_w = read_tensor_f32(bytes, gguf, &format!("{prefix}.post_attention_norm.weight"))
            .map_err(|e| format!("{prefix}.post_attention_norm: {e:?}"))?;
        let ffn_gate = QuantWeight::from_gguf(
            Arc::clone(mmap), gguf, &format!("{prefix}.ffn_gate.weight"), cfg.d_ff, cfg.d_model,
        ).map_err(|e| format!("{prefix}.ffn_gate: {e}"))?;
        let ffn_up = QuantWeight::from_gguf(
            Arc::clone(mmap), gguf, &format!("{prefix}.ffn_up.weight"), cfg.d_ff, cfg.d_model,
        ).map_err(|e| format!("{prefix}.ffn_up: {e}"))?;
        let ffn_down = QuantWeight::from_gguf(
            Arc::clone(mmap), gguf, &format!("{prefix}.ffn_down.weight"), cfg.d_model, cfg.d_ff,
        ).map_err(|e| format!("{prefix}.ffn_down: {e}"))?;

        if cfg.is_attention_layer(il) {
            // Q tensor is 2× wide: [d_model, 2 * n_q_heads * d_head]
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
                attn_norm_w, attn_post_norm_w, wq, wk, wv, wo, q_norm_w, k_norm_w,
                ffn_gate, ffn_up, ffn_down,
            })));
        } else {
            // DeltaNet layer.
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
                ssm_conv1d_w, ssm_dt_bias, ssm_a, ssm_alpha, ssm_beta, ssm_norm_w, ssm_out,
                ffn_gate, ffn_up, ffn_down,
            })));
        }
    }

    Ok(Qwen36Weights { token_embd, output_norm_w, lm_head, blocks })
}

// ── State ─────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct DeltaNetState {
    /// Last `d_conv` columns of the conv1d input, per channel. Shape:
    /// `[d_conv, conv_dim]` row-major (newest at position d_conv-1).
    pub conv_state: Vec<f32>,
    /// Per V-head recurrent state. Shape: `[n_v_heads, d_state, d_state]`
    /// row-major (head, row=value-dim-index, col=key-dim-index).
    /// Using f64 internally for accumulation stability.
    pub ssm_state: Vec<f64>,
}

#[derive(Clone)]
pub struct AttentionState {
    pub k_cache: Vec<f32>,  // [seq_len, n_kv_heads, d_head] row-major
    pub v_cache: Vec<f32>,
}

#[derive(Clone)]
pub enum BlockState {
    DeltaNet(DeltaNetState),
    Attention(AttentionState),
}

#[derive(Clone)]
pub struct Qwen36State {
    pub blocks: Vec<BlockState>,
    pub pos: usize,
}

impl Qwen36State {
    pub fn new(cfg: &Qwen36Config, layer_kinds: &[LayerKind]) -> Self {
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
    std::env::var("RCE_QWEN36_TRACE").is_ok()
}

fn trace_tensor(name: &str, x: &[f32]) {
    if !trace_enabled() { return; }
    let sum: f64 = x.iter().map(|v| *v as f64).sum();
    let n = x.len().min(3);
    let mut first = [0f32; 3];
    first[..n].copy_from_slice(&x[..n]);
    eprintln!(
        "rce_trace: {name:<32} sum={sum:+.6e}  first=[{:.4}, {:.4}, {:.4}]  len={}",
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

/// Per-head RMS-norm: `x` shaped `[n_heads, head_dim]` row-major, weight
/// shape `[head_dim]` shared across heads.
fn rms_norm_per_head(x: &mut [f32], w: &[f32], n_heads: usize, head_dim: usize, eps: f32) {
    debug_assert_eq!(x.len(), n_heads * head_dim);
    for h in 0..n_heads {
        let off = h * head_dim;
        let slice = &mut x[off..off + head_dim];
        let ms: f32 = slice.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for (v, ww) in slice.iter_mut().zip(w.iter()) {
            *v = *v * inv * *ww;
        }
    }
}

/// Per-head L2-norm (no weight, no eps in the divisor — pure L2): each
/// head's vector is divided by its L2 magnitude (+ small eps for stability).
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

/// Wrapper over `rce_op_conv1d::conv1d_causal_step` matching the
/// canonical rce layout: state `[conv_dim, d_conv]` (channel-outer,
/// kernel-inner), weight same layout, NO bias (Qwen 3.6 conv1d has no
/// bias in the GGUF — only `.weight`).
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

/// Multi-axis RoPE: rotate `x` (one head's worth, `d_head` floats) in
/// place, with rotation pairs split across `sections`. The first
/// `2*sections[0]` dims rotate with rate from section-0; the next
/// `2*sections[1]` from section-1; etc. All sections currently use the
/// same `pos` (text-only inference; M-RoPE's multi-position support is
/// for vision-language with image patch positions).
fn multi_axis_rope_inplace(
    x: &mut [f32], head_dim: usize, sections: &[u32; 4], base: f32, pos: usize,
) {
    let mut section_start = 0usize;
    for &sec_pairs in sections.iter() {
        let sec_pairs = sec_pairs as usize;
        if sec_pairs == 0 { continue; }
        let rope_dim_here = sec_pairs * 2;
        let half = sec_pairs;
        // The rotation pairs in this section are at positions
        // (section_start + i, section_start + i + half) for i in 0..half.
        let log_base = base.ln();
        // Total span of all active sections (for inv_freq exponent normalization).
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

// ── DeltaNet forward step (single token) ─────────────────────────────

fn delta_net_forward(
    x_in: &[f32], blk: &DeltaNetBlock, state: &mut DeltaNetState, cfg: &Qwen36Config, il: usize,
) -> Vec<f32> {
    let trace = il == 0;
    let key_dim = cfg.key_dim();
    let value_dim = cfg.value_dim();
    let conv_dim = cfg.conv_dim();
    let n_v = cfg.n_v_heads;
    let n_k = cfg.n_k_heads;
    let v_per_k = cfg.v_per_k();
    let dh = cfg.d_state;

    // 1) Fused QKV projection [conv_dim].
    let qkv_mixed = qmm_1d(x_in, &blk.wqkv);
    debug_assert_eq!(qkv_mixed.len(), conv_dim);
    if trace { trace_tensor("linear_attn_qkv_mixed-0", &qkv_mixed); }

    // 2) Z-gate projection [value_dim].
    let z = qmm_1d(x_in, &blk.wqkv_gate);
    debug_assert_eq!(z.len(), value_dim);
    if trace { trace_tensor("z-0", &z); }

    // 3) Beta gate (sigmoid'd) and alpha gate (softplus + ssm_a multiply).
    let beta_raw = qmm_1d(x_in, &blk.ssm_beta);  // [n_v]
    if trace { trace_tensor("beta_raw-0", &beta_raw); }
    let mut beta = vec![0.0f32; n_v];
    for i in 0..n_v { beta[i] = sigmoid(beta_raw[i]); }
    if trace { trace_tensor("beta-0", &beta); }
    let alpha_raw = qmm_1d(x_in, &blk.ssm_alpha);  // [n_v]
    if trace { trace_tensor("alpha-0", &alpha_raw); }
    let mut decay_log = vec![0.0f32; n_v];
    let mut a_sp_dbg = vec![0.0f32; n_v];
    for i in 0..n_v {
        let a_biased = alpha_raw[i] + blk.ssm_dt_bias[i];
        let a_sp = softplus(a_biased);
        a_sp_dbg[i] = a_sp;
        // ssm_a is already negative (= -exp(A_log_orig) at conversion time);
        // multiply gives the per-step log-decay rate.
        decay_log[i] = a_sp * blk.ssm_a[i];
    }
    if trace { trace_tensor("a_softplus-0", &a_sp_dbg); }
    if trace { trace_tensor("gate-0", &decay_log); }

    // 4) Depthwise conv1d (kernel=d_conv) over qkv_mixed; SiLU.
    let conv_out = conv1d_step(
        &qkv_mixed, &blk.ssm_conv1d_w, &mut state.conv_state, conv_dim, cfg.d_conv,
    );
    if trace { trace_tensor("conv_output_raw-0", &conv_out); }
    let conv_silu: Vec<f32> = conv_out.iter().map(|v| silu(*v)).collect();
    if trace { trace_tensor("conv_output_silu-0", &conv_silu); }

    // 5) Split into q, k, v slices.
    let q_slice = &conv_silu[0..key_dim];                  // [16 heads × 128]
    let k_slice = &conv_silu[key_dim..2 * key_dim];        // [16 heads × 128]
    let v_slice = &conv_silu[2 * key_dim..conv_dim];       // [48 heads × 128]
    if trace { trace_tensor("q_conv-0", q_slice); }
    if trace { trace_tensor("k_conv-0", k_slice); }
    if trace { trace_tensor("v_conv-0", v_slice); }
    let mut q_buf = q_slice.to_vec();
    let mut k_buf = k_slice.to_vec();
    l2_norm_per_head(&mut q_buf, n_k, dh, cfg.eps);
    l2_norm_per_head(&mut k_buf, n_k, dh, cfg.eps);
    if trace { trace_tensor("q_conv_l2norm-0", &q_buf); }
    if trace { trace_tensor("k_conv_l2norm-0", &k_buf); }

    // 6) Recurrent state update per V head. Convention (from llama.cpp
    //    `build_delta_net_autoregressive` in delta-net-base.cpp):
    //    - State S is indexed [k_idx, v_idx] where k_idx is the KEY axis
    //      (innermost / row stride) and v_idx is the VALUE axis (column).
    //    - Q is pre-scaled by 1/sqrt(d_state).
    //    - State decay applied first (multiplies all S entries by decay).
    //    - Retrieve: r[v] = Σ_k S[k, v] * key[k]   (sum over KEY axis)
    //    - delta[v] = beta * (v[v] - r[v])
    //    - State update: S[k, v] += key[k] * delta[v]   (outer product)
    //    - Output: y[v] = Σ_k S[k, v] * q_scaled[k]
    //
    //    Linear-storage layout: `s[k * dh + v]` = S[k, v].
    //
    //    The V heads are independent: each owns a disjoint `dh×dh` state
    //    block and writes a disjoint `dh`-slice of y. Run them across
    //    cores with rayon (this recurrence is the DeltaNet hot path).
    let q_scale = 1.0f64 / (dh as f64).sqrt();
    let mut y = vec![0.0f32; value_dim];
    let _ = v_per_k; // kept for doc; actual mapping is tile (ggml_repeat).
    use rayon::prelude::*;
    state.ssm_state
        .par_chunks_mut(dh * dh)
        .zip(y.par_chunks_mut(dh))
        .enumerate()
        .for_each(|(v_h, (s, y_h))| {
            // GQA mapping for DeltaNet: ggml_repeat_4d tiles, so V head
            // v_h pairs with K head (v_h % n_k_heads) — NOT the standard
            // grouped mapping (v_h / v_per_k).
            let k_h = v_h % n_k;
            let q_v = &q_buf[k_h * dh..(k_h + 1) * dh];
            let k_v = &k_buf[k_h * dh..(k_h + 1) * dh];
            let v_v = &v_slice[v_h * dh..(v_h + 1) * dh];
            let beta_v = beta[v_h] as f64;
            let decay_v = (decay_log[v_h] as f64).exp();

            // Fused decay + retrieve: S ← decay·S and
            // r[v] = Σ_k S[k,v]·key[k] in one pass.
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

            // delta[v] = β * (v[v] - r[v])
            let mut delta_v = vec![0.0f64; dh];
            for v_idx in 0..dh {
                delta_v[v_idx] = beta_v * (v_v[v_idx] as f64 - r[v_idx]);
            }

            // State update: S[k, v] += key[k] * delta[v]
            for k_idx in 0..dh {
                let k_kv = k_v[k_idx] as f64;
                let row_off = k_idx * dh;
                for v_idx in 0..dh {
                    s[row_off + v_idx] += k_kv * delta_v[v_idx];
                }
            }

            // Output: y[v] = Σ_k S[k, v] * q_scaled[k]
            for v_idx in 0..dh {
                let mut acc = 0.0f64;
                for k_idx in 0..dh {
                    acc += s[k_idx * dh + v_idx] * (q_v[k_idx] as f64 * q_scale);
                }
                y_h[v_idx] = acc as f32;
            }
        });

    if trace { trace_tensor("attn_output_raw-0", &y); }
    // 7) Per-V-head RMSnorm of y; multiply by SiLU(z) per channel.
    rms_norm_per_head(&mut y, &blk.ssm_norm_w, n_v, dh, cfg.eps);
    if trace { trace_tensor("attn_output_normed-0", &y); }
    for (i, yi) in y.iter_mut().enumerate() { *yi *= silu(z[i]); }
    if trace { trace_tensor("final_output-0", &y); }

    // 8) Output projection.
    let out = qmm_1d(&y, &blk.ssm_out);
    if trace { trace_tensor("linear_attn_out-0", &out); }
    out
}

// ── Gated Attention forward (single token, append-to-cache) ──────────

fn gated_attention_forward(
    x_in: &[f32], blk: &AttentionBlock, state: &mut AttentionState,
    cfg: &Qwen36Config, pos: usize,
) -> Vec<f32> {
    let nq = cfg.n_q_heads;
    let nkv = cfg.n_kv_heads;
    let dh = cfg.d_head;
    let q_per_kv = cfg.q_per_kv();

    // Q + gate combined: [2 * nq * dh]
    let q_full = qmm_1d(x_in, &blk.wq);
    debug_assert_eq!(q_full.len(), 2 * nq * dh);

    // Split per-head: for head h, Q is q_full[h*2*dh .. h*2*dh + dh],
    // gate is q_full[h*2*dh + dh .. (h+1)*2*dh].
    let mut q = vec![0.0f32; nq * dh];
    let mut gate = vec![0.0f32; nq * dh];
    for h in 0..nq {
        let src = h * 2 * dh;
        q[h * dh..(h + 1) * dh].copy_from_slice(&q_full[src..src + dh]);
        gate[h * dh..(h + 1) * dh].copy_from_slice(&q_full[src + dh..src + 2 * dh]);
    }

    // Per-head Q normalization.
    rms_norm_per_head(&mut q, &blk.q_norm_w, nq, dh, cfg.eps);

    // K and V.
    let mut k_new = qmm_1d(x_in, &blk.wk);
    let v_new = qmm_1d(x_in, &blk.wv);
    rms_norm_per_head(&mut k_new, &blk.k_norm_w, nkv, dh, cfg.eps);

    // M-RoPE on Q and K (each head independently).
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

    // Append to KV cache.
    state.k_cache.extend_from_slice(&k_new);
    state.v_cache.extend_from_slice(&v_new);
    let seq_len = state.k_cache.len() / (nkv * dh);

    // SDPA per Q head with GQA mapping.
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

    // Apply sigmoid gate to attn output (Qwen 3.6 Gated Attention).
    rce_op_attn::apply_attention_gate(&mut attn, &gate);

    // Output projection.
    qmm_1d(&attn, &blk.wo)
}

// ── FFN forward (SwiGLU) ──────────────────────────────────────────────

fn ffn_forward_swiglu(
    x_in: &[f32], gate_w: &QuantWeight, up_w: &QuantWeight, down_w: &QuantWeight, d_ff: usize,
) -> Vec<f32> {
    let g = qmm_1d(x_in, gate_w);
    let u = qmm_1d(x_in, up_w);
    let mut h = vec![0.0f32; d_ff];
    for i in 0..d_ff { h[i] = silu(g[i]) * u[i]; }
    qmm_1d(&h, down_w)
}

// ── forward_token ─────────────────────────────────────────────────────

pub fn forward_token(
    w: &Qwen36Weights, cfg: &Qwen36Config, state: &mut Qwen36State, token_id: i32,
) -> Vec<f32> {
    let embed: Vec<f32> = w.token_embd.dequant_row(token_id as usize);
    forward_embedding(w, cfg, state, embed)
}

/// LM forward from a **pre-computed** `[d_model]` input embedding,
/// skipping the token-id → embedding lookup.  Same path as `forward_token`
/// from there on.  Entry point used by VLMs like LLaVA where vision
/// tokens are projected into LM embedding space and fed here.
pub fn forward_embedding(
    w: &Qwen36Weights, cfg: &Qwen36Config, state: &mut Qwen36State, embedding: Vec<f32>,
) -> Vec<f32> {
    let dm = cfg.d_model;
    assert_eq!(embedding.len(), dm,
               "embedding length {} must equal d_model={}", embedding.len(), dm);
    let mut x = embedding;
    trace_tensor("model.input_embed", &x);
    let pos = state.pos;
    for (il, block) in w.blocks.iter().enumerate() {
        if il == 0 { trace_tensor("x_pre_norm-0", &x); }
        // Pre-attn norm.
        let normed = rms_norm(&x, match block {
            Block::DeltaNet(b) => &b.attn_norm_w,
            Block::Attention(b) => &b.attn_norm_w,
        }, cfg.eps);
        if il == 0 { trace_tensor("attn_norm-0", &normed); }

        // Mixer (DeltaNet or Gated Attention).
        let attn_out = match (block, &mut state.blocks[il]) {
            (Block::DeltaNet(b), BlockState::DeltaNet(s)) => {
                delta_net_forward(&normed, b, s, cfg, il)
            }
            (Block::Attention(b), BlockState::Attention(s)) => {
                gated_attention_forward(&normed, b, s, cfg, pos)
            }
            _ => panic!("block/state kind mismatch at layer {il}"),
        };

        // Residual.
        for i in 0..dm { x[i] += attn_out[i]; }

        // Post-attn norm (doubles as pre-FFN).
        let ffn_in = rms_norm(&x, match block {
            Block::DeltaNet(b) => &b.attn_post_norm_w,
            Block::Attention(b) => &b.attn_post_norm_w,
        }, cfg.eps);

        // FFN.
        let (g_w, u_w, d_w) = match block {
            Block::DeltaNet(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down),
            Block::Attention(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down),
        };
        let ffn_out = ffn_forward_swiglu(&ffn_in, g_w, u_w, d_w, cfg.d_ff);

        // Residual.
        for i in 0..dm { x[i] += ffn_out[i]; }
    }
    // Final norm + lm_head.
    let final_normed = rms_norm(&x, &w.output_norm_w, cfg.eps);
    let logits = qmm_1d(&final_normed, &w.lm_head);
    state.pos += 1;
    logits
}

// ── Runtime ───────────────────────────────────────────────────────────

pub struct Qwen36Runtime {
    pub cfg: Qwen36Config,
    pub gguf: GgufFile,
    pub weights: Qwen36Weights,
    pub layer_kinds: Vec<LayerKind>,
    _mmap: Arc<Mmap>,
}

impl Qwen36Runtime {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| format!("open: {e}"))?;
        let mmap = Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?);
        // WillNeed (not Sequential): keeps the model page-resident so
        // decode hits RAM speed instead of re-faulting weights from disk.
        let _ = mmap.advise(memmap2::Advice::WillNeed);
        let gguf = GgufFile::parse(&mmap[..]).map_err(|e| format!("parse: {e:?}"))?;
        let cfg = Qwen36Config::from_gguf(&gguf)?;

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
            "qwen36: d_model={} d_ff={} vocab={} n_layers={} ({n_attn} GA + {n_delta} GDN) d_head={} n_q/kv={}/{} d_state={} n_v_heads={} eps={}",
            cfg.d_model, cfg.d_ff, cfg.vocab, cfg.n_layers, cfg.d_head, cfg.n_q_heads, cfg.n_kv_heads, cfg.d_state, cfg.n_v_heads, cfg.eps,
        );
        Ok(Self { cfg, gguf, weights, layer_kinds, _mmap: mmap })
    }

    pub fn new_state(&self) -> Qwen36State { Qwen36State::new(&self.cfg, &self.layer_kinds) }

    pub fn forward_logits(&self, state: &mut Qwen36State, token_id: i32) -> Vec<f32> {
        forward_token(&self.weights, &self.cfg, state, token_id)
    }

    /// LM forward from a pre-computed `[d_model]` embedding (e.g., a
    /// LLaVA-style projected vision token), bypassing the embedding lookup.
    pub fn forward_logits_embed(&self, state: &mut Qwen36State, embedding: Vec<f32>) -> Vec<f32> {
        forward_embedding(&self.weights, &self.cfg, state, embedding)
    }

    pub fn tokenizer(&self) -> Tokenizer<'_> {
        Tokenizer::from_gguf(&self.gguf).expect("tokenizer")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Qwen36GenConfig {
    pub max_new_tokens: usize,
    pub sample: SampleConfig,
    pub rep_window: usize,
}
impl Default for Qwen36GenConfig {
    fn default() -> Self {
        Self { max_new_tokens: 32, sample: SampleConfig::chat(0x9E36_36AA), rep_window: 64 }
    }
}

impl Qwen36Runtime {
    /// Run a single-prompt greedy/sampled generation. Same shape as
    /// `rce-granite::GraniteRuntime::generate` — GPT-2 BPE tokenization,
    /// `sample_with_history` for repetition-aware sampling, decode per token
    /// for streaming.
    pub fn generate(
        &self, prompt: &str, gcfg: &Qwen36GenConfig, mut on_token: impl FnMut(&str),
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
    const MODEL: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/qwen3.6-27b-gguf/Qwen3.6-27B-Q5_K_M.gguf";

    #[test]
    fn qwen36_load_and_layer_pattern_sanity() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("qwen36: GGUF not mounted — skipping");
            return;
        }
        let rt = Qwen36Runtime::load(MODEL).expect("load Qwen 3.6-27B");
        let c = &rt.cfg;
        assert_eq!(c.d_model, 5120);
        assert_eq!(c.d_ff, 17408);
        assert_eq!(c.n_layers, 64);
        assert_eq!(c.n_q_heads, 24);
        assert_eq!(c.n_kv_heads, 4);
        assert_eq!(c.d_head, 256);
        assert_eq!(c.full_attention_interval, 4);
        assert_eq!(c.vocab, 248320);
        assert_eq!(c.rope_dimension_sections, [11, 11, 10, 0]);
        assert_eq!(c.n_k_heads, 16);
        assert_eq!(c.n_v_heads, 48);
        assert_eq!(c.d_state, 128);
        let n_attn = rt.layer_kinds.iter().filter(|k| **k == LayerKind::GatedAttention).count();
        let n_delta = rt.layer_kinds.iter().filter(|k| **k == LayerKind::GatedDeltaNet).count();
        assert_eq!(n_attn, 16);
        assert_eq!(n_delta, 48);
    }

    #[test]
    fn qwen36_generate_proves_coherence() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("qwen36: GGUF not mounted — skipping");
            return;
        }
        let rt = Qwen36Runtime::load(MODEL).expect("load");
        let cfg = Qwen36GenConfig {
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
            "qwen36 generate OK: prompt={prompt:?} → continuation={full:?}  [{} tok in {dt:?} = {:.2} tok/s]",
            chunks.len(),
            chunks.len() as f64 / dt.as_secs_f64(),
        );
        assert!(!full.is_empty(), "empty continuation");
        // Coherence check: should be readable English. Look for common
        // words OR a high fraction of alphabetic chars (rejects all-
        // garbage byte streams).
        let alpha = full.chars().filter(|c| c.is_alphabetic()).count();
        let total = full.chars().count();
        assert!(
            alpha * 2 >= total,
            "output looks like garbage (alpha {alpha} / total {total}): {full:?}",
        );
    }

    /// Trace single-token forward (token 760 = "The") to compare against
    /// llama-eval-callback output. Run with:
    /// `RCE_QWEN36_TRACE=1 cargo test --release -p rce-qwen36 qwen36_trace_layer0 -- --nocapture --ignored`
    #[test]
    #[ignore]
    fn qwen36_trace_layer0() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("qwen36: GGUF not mounted — skipping");
            return;
        }
        let rt = Qwen36Runtime::load(MODEL).expect("load");
        let mut state = rt.new_state();
        // Token 760 = "The" per llama-eval-callback's tokenization.
        let _logits = rt.forward_logits(&mut state, 760);
    }

    #[test]
    fn qwen36_one_forward_finite() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("qwen36: GGUF not mounted — skipping");
            return;
        }
        let rt = Qwen36Runtime::load(MODEL).expect("load");
        let mut state = rt.new_state();
        // Qwen tokenizer BOS isn't strictly required; just feed a small token id.
        let logits = rt.forward_logits(&mut state, 1);
        assert_eq!(logits.len(), rt.cfg.vocab);
        let n_finite = logits.iter().filter(|v| v.is_finite()).count();
        assert_eq!(n_finite, logits.len(), "expected all-finite logits");
        let max_abs = logits.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        eprintln!("qwen36 one forward → max|logit|={max_abs:.3}");
        assert!(max_abs > 0.0, "all-zero logits suspicious");
        assert!(max_abs < 1e6, "logits exploded ({max_abs})");
    }
}
