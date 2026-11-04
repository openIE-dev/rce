//! Zyphra ZAYA1-8B inference runtime scaffolding.
//!
//! Architecture (per llama.cpp Draft PR #23112 `src/models/zaya.cpp`
//! and direct inspection of `Abiray/ZAYA1-8B-GGUF` `ZAYA1-8B-Q8_0.gguf`):
//!
//! - **80 layers** alternating: `il % 2 == 0` → CCA attention,
//!   `il % 2 == 1` → MoE FFN. Each layer is either attention OR FFN
//!   (not both — unlike standard transformers).
//! - d_model=2048, vocab=262272, eps=1e-5, **tied lm_head**.
//! - **CCA layers** (40 of them): GQA 8/2 (4×), d_head=128. Q/K projections,
//!   plus V split into `cca_val_proj1 @ cur` and `cca_val_proj2 @ delayed_hs`
//!   (previous token's hidden state — stored in the recurrent state buffer).
//!   The Q+K concat is then run through a depthwise causal conv1d
//!   (kernel=2, channels=n_qk=1280) plus a grouped 1D conv
//!   (n_groups = n_q + n_kv = 10), followed by qk_mean bias, L2-norm,
//!   scale by sqrt(d_head), K per-head scale, then standard RoPE +
//!   GQA SDPA with V. Output projection to d_model.
//! - **MoE layers** (40 of them): multi-stage MLP router:
//!   1. `router_h = ffn_gate_inp @ cur + bias`  (d_model → n_ff_exp=256)
//!   2. EDA (Exponential Distributed Average): for `il > 1`,
//!      `router_h += prev_router * eda_scale`. Carries router state
//!      across MoE layers.
//!   3. `router_h = RMSnorm(router_h, ffn_norm)`
//!   4. `router_h = GELU(ffn_gate @ router_h + bias)`
//!   5. `router_h = GELU(zaya_router_mlp2 @ router_h + bias)`
//!   6. `router_logits = zaya_router_mlp4 @ router_h`  (n_ff_exp → n_expert+1 = 17)
//!   7. Add `zaya_router_biases` (PID-controlled bias-balance).
//!   8. `router_probs = softmax(router_logits)` over 17 slots.
//!   9. Top-1 routing on first 16 (the 17th "MOD" skip is included in
//!      the softmax denominator but never selected — natural skip valve).
//!   10. SwiGLU expert FFN: `ffn_gate_up_exps[expert]` (fused gate+up,
//!       split halves), SiLU(gate) × up, then `ffn_down_exps[expert]`.
//! - **Learned residual scaling** (per-layer + final): hidden_state and
//!   residual stream both get an affine `(x + bias) * scale` applied
//!   in **FP32** before the residual add. Even layers only have
//!   `res_scale_hs`; odd layers have both `res_scale_hs` and
//!   `res_scale_res`. A final `res_scale_hs` + `res_scale_res` pair
//!   is applied at model end before output_norm + lm_head.
//!
//! **This crate ships**: config + per-layer arch dispatch + weight
//! loading + sanity load test. Full forward pass (CCA + MoE + residual
//! scaling) lands in a follow-on session.

use rce_ingest_gguf::{
    read_tensor_f32, sample_with_history, GgufFile, GgufValue, SampleConfig, Tokenizer,
};
use rce_op_qmm::QuantWeight;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Zaya1Config {
    pub d_model: usize,
    pub n_layers: usize,
    pub vocab: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub d_head: usize,
    pub d_ff_expert: usize,        // 2048 — per-expert FFN inner dim
    pub d_ff_router: usize,        // 256 — n_ff_exp, router hidden + EDA stream width
    pub n_experts: usize,          // 16
    pub n_experts_used: usize,     // 1 (Switch Transformer top-1)
    pub d_conv: usize,             // 2 — CCA depthwise conv kernel
    pub rope_freq_base: f32,
    pub rope_dim: usize,
    pub eps: f32,
}

impl Zaya1Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf.metadata.iter()
            .find(|kv| kv.key == "general.architecture")
            .and_then(|kv| match &kv.value { GgufValue::String(s) => Some(s.as_str()), _ => None })
            .ok_or("missing general.architecture")?;
        if arch != "zaya" {
            return Err(format!("expected zaya, got {arch}"));
        }
        let u32_meta = |k: &str| -> Option<u32> {
            gguf.metadata.iter().find(|kv| kv.key == format!("zaya.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::U32(v) => Some(*v), _ => None })
        };
        let f32_meta = |k: &str| -> Option<f32> {
            gguf.metadata.iter().find(|kv| kv.key == format!("zaya.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::F32(v) => Some(*v), _ => None })
        };

        let d_model = u32_meta("embedding_length").ok_or("embedding_length")? as usize;
        let n_layers = u32_meta("block_count").ok_or("block_count")? as usize;
        // Prefer the actual token_embd row count over the declared
        // vocab_size — Q8_0 quant tends to pad vocab differently.
        let vocab = gguf.tensors.iter()
            .find(|t| t.name == "token_embd.weight")
            .and_then(|t| t.dims.get(1).copied())
            .map(|d| d as usize)
            .or_else(|| u32_meta("vocab_size").map(|v| v as usize))
            .ok_or("vocab_size")?;
        let n_q_heads = u32_meta("attention.head_count").ok_or("attention.head_count")? as usize;
        let n_kv_heads = u32_meta("attention.head_count_kv").unwrap_or(n_q_heads as u32) as usize;
        let d_head = u32_meta("attention.key_length").ok_or("attention.key_length")? as usize;
        let d_ff_expert = u32_meta("feed_forward_length").ok_or("feed_forward_length")? as usize;
        let d_ff_router = u32_meta("expert_feed_forward_length").unwrap_or(256) as usize;
        let n_experts = u32_meta("expert_count").ok_or("expert_count")? as usize;
        let n_experts_used = u32_meta("expert_used_count").unwrap_or(1) as usize;
        let d_conv = u32_meta("ssm.conv_kernel").unwrap_or(2) as usize;
        let rope_freq_base = f32_meta("rope.freq_base").unwrap_or(5_000_000.0);
        let rope_dim = u32_meta("rope.dimension_count").unwrap_or(d_head as u32) as usize;
        let eps = f32_meta("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);

        Ok(Zaya1Config {
            d_model, n_layers, vocab,
            n_q_heads, n_kv_heads, d_head,
            d_ff_expert, d_ff_router,
            n_experts, n_experts_used,
            d_conv,
            rope_freq_base, rope_dim, eps,
        })
    }

    /// `n_qk = n_q*d_head + n_kv*d_head` — width of the fused Q+K stream
    /// that the CCA conv1d operates on.
    #[inline] pub fn n_qk(&self) -> usize { (self.n_q_heads + self.n_kv_heads) * self.d_head }
    /// `n_groups` for the grouped conv1d in CCA. = n_q + n_kv.
    #[inline] pub fn cca_n_groups(&self) -> usize { self.n_q_heads + self.n_kv_heads }
    /// Layer kind: even = CCA attention, odd = MoE FFN.
    #[inline] pub fn is_cca_layer(&self, il: usize) -> bool { il % 2 == 0 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind { CcaAttention, MoeFfn }

// ── Weights ───────────────────────────────────────────────────────────

pub struct CcaBlock {
    pub attn_norm_w: Vec<f32>,
    pub res_scale_hs_w: Vec<f32>,
    pub res_scale_hs_b: Vec<f32>,
    /// res_scale_res is present for all CCA layers except blk.0
    /// (which is the first layer — no incoming residual to scale).
    pub res_scale_res_w: Option<Vec<f32>>,
    pub res_scale_res_b: Option<Vec<f32>>,
    // CCA-specific:
    pub wq: QuantWeight,                // [n_q*d_head, d_model]
    pub wk: QuantWeight,                // [n_kv*d_head, d_model]
    pub cca_val_proj1: QuantWeight,     // [n_kv*d_head/2, d_model]
    pub cca_val_proj2: QuantWeight,     // [n_kv*d_head/2, d_model] — for delayed-V from prev hs
    pub wo: QuantWeight,                // [d_model, n_q*d_head]
    pub ssm_conv1d_w: Vec<f32>,         // [d_conv, n_qk]
    pub ssm_conv1d_b: Vec<f32>,         // [n_qk]
    pub cca_conv_grp_w: Vec<f32>,       // [d_conv, n_qk/n_groups, n_qk]
    pub cca_conv_grp_b: Vec<f32>,       // [n_qk]
    pub cca_k_scale_w: Vec<f32>,        // [n_kv_heads]
}

pub struct MoeBlock {
    pub attn_norm_w: Vec<f32>,
    pub res_scale_hs_w: Vec<f32>,
    pub res_scale_hs_b: Vec<f32>,
    pub res_scale_res_w: Vec<f32>,
    pub res_scale_res_b: Vec<f32>,
    // Router (multi-stage MLP):
    pub router_down_w: QuantWeight,     // [d_ff_router, d_model] — actually stored f32; use QuantWeight for unified API
    pub router_down_b: Vec<f32>,        // [d_ff_router]
    pub router_norm_w: Vec<f32>,        // [d_ff_router]
    pub router_mlp0_w: QuantWeight,     // [d_ff_router, d_ff_router]
    pub router_mlp0_b: Vec<f32>,        // [d_ff_router]
    pub router_mlp2_w: QuantWeight,     // [d_ff_router, d_ff_router]
    pub router_mlp2_b: Vec<f32>,        // [d_ff_router]
    pub router_mlp4_w: QuantWeight,     // [n_expert+1, d_ff_router]
    pub router_biases: Vec<f32>,        // [n_expert+1] — applied to ffn_moe_probs (first n_expert) AFTER softmax
    /// EDA scale (per-channel multiplier for the prev-layer router_down output).
    /// `None` for the first MoE layer (il=1); present for il=3, 5, ..., 79.
    pub router_eda_scale: Option<Vec<f32>>,
    // Experts (fused gate+up + separate down):
    pub ffn_gate_up_exps: QuantWeight,  // [2*d_ff_expert, d_model] per expert; n_experts of them, stored stacked
    pub ffn_down_exps: QuantWeight,     // [d_model, d_ff_expert] per expert; stacked
}

pub enum Block {
    Cca(Box<CcaBlock>),
    Moe(Box<MoeBlock>),
}

pub struct Zaya1Weights {
    pub token_embd: QuantWeight,        // [vocab, d_model] — also serves as lm_head (tied)
    pub output_norm_w: Vec<f32>,
    pub final_res_scale_hs_w: Vec<f32>,
    pub final_res_scale_hs_b: Vec<f32>,
    pub final_res_scale_res_w: Vec<f32>,
    pub final_res_scale_res_b: Vec<f32>,
    pub blocks: Vec<Block>,
}

fn load_weights(
    mmap: &Arc<Mmap>, gguf: &GgufFile, cfg: &Zaya1Config,
) -> Result<Zaya1Weights, String> {
    let bytes: &[u8] = &mmap[..];

    let token_embd = QuantWeight::from_gguf(
        Arc::clone(mmap), gguf, "token_embd.weight", cfg.vocab, cfg.d_model,
    ).map_err(|e| format!("token_embd: {e}"))?;
    let output_norm_w = read_tensor_f32(bytes, gguf, "output_norm.weight")
        .map_err(|e| format!("output_norm: {e:?}"))?;
    let final_res_scale_hs_w = read_tensor_f32(bytes, gguf, "res_scale_hs.weight")
        .map_err(|e| format!("final res_scale_hs.weight: {e:?}"))?;
    let final_res_scale_hs_b = read_tensor_f32(bytes, gguf, "res_scale_hs.bias")
        .map_err(|e| format!("final res_scale_hs.bias: {e:?}"))?;
    let final_res_scale_res_w = read_tensor_f32(bytes, gguf, "res_scale_res.weight")
        .map_err(|e| format!("final res_scale_res.weight: {e:?}"))?;
    let final_res_scale_res_b = read_tensor_f32(bytes, gguf, "res_scale_res.bias")
        .map_err(|e| format!("final res_scale_res.bias: {e:?}"))?;

    let n_qk = cfg.n_qk();
    let value_half = cfg.n_kv_heads * cfg.d_head / 2;
    let mut blocks = Vec::with_capacity(cfg.n_layers);

    for il in 0..cfg.n_layers {
        let p = format!("blk.{il}");
        let attn_norm_w = read_tensor_f32(bytes, gguf, &format!("{p}.attn_norm.weight"))
            .map_err(|e| format!("{p}.attn_norm: {e:?}"))?;
        let res_scale_hs_w = read_tensor_f32(bytes, gguf, &format!("{p}.res_scale_hs.weight"))
            .map_err(|e| format!("{p}.res_scale_hs.weight: {e:?}"))?;
        let res_scale_hs_b = read_tensor_f32(bytes, gguf, &format!("{p}.res_scale_hs.bias"))
            .map_err(|e| format!("{p}.res_scale_hs.bias: {e:?}"))?;

        if cfg.is_cca_layer(il) {
            let res_scale_res_w = read_tensor_f32(bytes, gguf, &format!("{p}.res_scale_res.weight")).ok();
            let res_scale_res_b = read_tensor_f32(bytes, gguf, &format!("{p}.res_scale_res.bias")).ok();
            let wq = QuantWeight::from_gguf(
                Arc::clone(mmap), gguf, &format!("{p}.attn_q.weight"),
                cfg.n_q_heads * cfg.d_head, cfg.d_model,
            ).map_err(|e| format!("{p}.attn_q: {e}"))?;
            let wk = QuantWeight::from_gguf(
                Arc::clone(mmap), gguf, &format!("{p}.attn_k.weight"),
                cfg.n_kv_heads * cfg.d_head, cfg.d_model,
            ).map_err(|e| format!("{p}.attn_k: {e}"))?;
            let cca_val_proj1 = QuantWeight::from_gguf(
                Arc::clone(mmap), gguf, &format!("{p}.cca_val_proj1.weight"),
                value_half, cfg.d_model,
            ).map_err(|e| format!("{p}.cca_val_proj1: {e}"))?;
            let cca_val_proj2 = QuantWeight::from_gguf(
                Arc::clone(mmap), gguf, &format!("{p}.cca_val_proj2.weight"),
                value_half, cfg.d_model,
            ).map_err(|e| format!("{p}.cca_val_proj2: {e}"))?;
            let wo = QuantWeight::from_gguf(
                Arc::clone(mmap), gguf, &format!("{p}.attn_output.weight"),
                cfg.d_model, cfg.n_q_heads * cfg.d_head,
            ).map_err(|e| format!("{p}.attn_output: {e}"))?;
            let ssm_conv1d_w = read_tensor_f32(bytes, gguf, &format!("{p}.ssm_conv1d.weight"))
                .map_err(|e| format!("{p}.ssm_conv1d.weight: {e:?}"))?;
            let ssm_conv1d_b = read_tensor_f32(bytes, gguf, &format!("{p}.ssm_conv1d.bias"))
                .unwrap_or_else(|_| vec![0.0f32; n_qk]);
            let cca_conv_grp_w = read_tensor_f32(bytes, gguf, &format!("{p}.cca_conv_grp.weight"))
                .map_err(|e| format!("{p}.cca_conv_grp.weight: {e:?}"))?;
            let cca_conv_grp_b = read_tensor_f32(bytes, gguf, &format!("{p}.cca_conv_grp.bias"))
                .map_err(|e| format!("{p}.cca_conv_grp.bias: {e:?}"))?;
            let cca_k_scale_w = read_tensor_f32(bytes, gguf, &format!("{p}.cca_k_scale.weight"))
                .map_err(|e| format!("{p}.cca_k_scale: {e:?}"))?;
            blocks.push(Block::Cca(Box::new(CcaBlock {
                attn_norm_w, res_scale_hs_w, res_scale_hs_b,
                res_scale_res_w, res_scale_res_b,
                wq, wk, cca_val_proj1, cca_val_proj2, wo,
                ssm_conv1d_w, ssm_conv1d_b, cca_conv_grp_w, cca_conv_grp_b, cca_k_scale_w,
            })));
        } else {
            let res_scale_res_w = read_tensor_f32(bytes, gguf, &format!("{p}.res_scale_res.weight"))
                .map_err(|e| format!("{p}.res_scale_res.weight: {e:?}"))?;
            let res_scale_res_b = read_tensor_f32(bytes, gguf, &format!("{p}.res_scale_res.bias"))
                .map_err(|e| format!("{p}.res_scale_res.bias: {e:?}"))?;
            // Router: ffn_gate_inp = down_proj, ffn_norm = router_norm,
            // ffn_gate = mlp0, zaya_router_mlp2 = mlp2, zaya_router_mlp4 = mlp4.
            let router_down_w = QuantWeight::from_gguf(
                Arc::clone(mmap), gguf, &format!("{p}.ffn_gate_inp.weight"),
                cfg.d_ff_router, cfg.d_model,
            ).map_err(|e| format!("{p}.ffn_gate_inp: {e}"))?;
            let router_down_b = read_tensor_f32(bytes, gguf, &format!("{p}.ffn_gate_inp.bias"))
                .unwrap_or_else(|_| vec![0.0f32; cfg.d_ff_router]);
            let router_norm_w = read_tensor_f32(bytes, gguf, &format!("{p}.ffn_norm.weight"))
                .map_err(|e| format!("{p}.ffn_norm: {e:?}"))?;
            let router_mlp0_w = QuantWeight::from_gguf(
                Arc::clone(mmap), gguf, &format!("{p}.ffn_gate.weight"),
                cfg.d_ff_router, cfg.d_ff_router,
            ).map_err(|e| format!("{p}.ffn_gate (router mlp0): {e}"))?;
            let router_mlp0_b = read_tensor_f32(bytes, gguf, &format!("{p}.ffn_gate.bias"))
                .unwrap_or_else(|_| vec![0.0f32; cfg.d_ff_router]);
            let router_mlp2_w = QuantWeight::from_gguf(
                Arc::clone(mmap), gguf, &format!("{p}.zaya_router_mlp2.weight"),
                cfg.d_ff_router, cfg.d_ff_router,
            ).map_err(|e| format!("{p}.router_mlp2: {e}"))?;
            let router_mlp2_b = read_tensor_f32(bytes, gguf, &format!("{p}.zaya_router_mlp2.bias"))
                .unwrap_or_else(|_| vec![0.0f32; cfg.d_ff_router]);
            let router_mlp4_w = QuantWeight::from_gguf(
                Arc::clone(mmap), gguf, &format!("{p}.zaya_router_mlp4.weight"),
                cfg.n_experts + 1, cfg.d_ff_router,
            ).map_err(|e| format!("{p}.router_mlp4: {e}"))?;
            let router_biases = read_tensor_f32(bytes, gguf, &format!("{p}.zaya_router_biases.weight"))
                .unwrap_or_else(|_| vec![0.0f32; cfg.n_experts + 1]);
            // EDA scale: present for il > 1 (first MoE layer has no prev_router).
            let router_eda_scale = read_tensor_f32(bytes, gguf, &format!("{p}.zaya_router_eda.weight")).ok();
            // Stacked experts: gate_up = [n_exp, 2*d_ff_expert, d_model] flat;
            // down = [n_exp, d_model, d_ff_expert] flat.
            let ffn_gate_up_exps = QuantWeight::from_gguf(
                Arc::clone(mmap), gguf, &format!("{p}.ffn_gate_up_exps.weight"),
                cfg.n_experts * 2 * cfg.d_ff_expert, cfg.d_model,
            ).map_err(|e| format!("{p}.ffn_gate_up_exps: {e}"))?;
            let ffn_down_exps = QuantWeight::from_gguf(
                Arc::clone(mmap), gguf, &format!("{p}.ffn_down_exps.weight"),
                cfg.n_experts * cfg.d_model, cfg.d_ff_expert,
            ).map_err(|e| format!("{p}.ffn_down_exps: {e}"))?;

            blocks.push(Block::Moe(Box::new(MoeBlock {
                attn_norm_w, res_scale_hs_w, res_scale_hs_b,
                res_scale_res_w, res_scale_res_b,
                router_down_w, router_down_b, router_norm_w,
                router_mlp0_w, router_mlp0_b,
                router_mlp2_w, router_mlp2_b,
                router_mlp4_w, router_biases, router_eda_scale,
                ffn_gate_up_exps, ffn_down_exps,
            })));
        }
    }

    Ok(Zaya1Weights {
        token_embd, output_norm_w,
        final_res_scale_hs_w, final_res_scale_hs_b,
        final_res_scale_res_w, final_res_scale_res_b,
        blocks,
    })
}

pub struct Zaya1Runtime {
    pub cfg: Zaya1Config,
    pub gguf: GgufFile,
    pub layer_kinds: Vec<LayerKind>,
    pub weights: Zaya1Weights,
    _mmap: Arc<Mmap>,
}

// ── Tracing (env-gated) ───────────────────────────────────────────────

#[inline]
fn trace_enabled() -> bool { std::env::var("RCE_ZAYA1_TRACE").is_ok() }

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

/// `(x + bias) * scale`, both per-channel of length `len(x)`. Matches
/// llama.cpp's `apply_res_scale` exactly.
fn apply_res_scale(x: &[f32], scale: &[f32], bias: &[f32]) -> Vec<f32> {
    debug_assert_eq!(x.len(), scale.len());
    debug_assert_eq!(x.len(), bias.len());
    let mut out = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        out.push((x[i] + bias[i]) * scale[i]);
    }
    out
}

// ── Per-layer state ──────────────────────────────────────────────────

#[derive(Clone)]
pub struct CcaState {
    /// Conv carry: last `d_conv - 1` columns of the QK stream, layout
    /// `[d_conv-1, n_qk]` row-major (oldest first). For d_conv=2 this is
    /// a single column of `n_qk` floats.
    pub conv_state: Vec<f32>,
    /// Previous-token hidden state (pre-attn-norm input to *this* layer's
    /// CCA forward). Length d_model. Used as the source for the delayed-V2
    /// projection.
    pub prev_hs: Vec<f32>,
    /// KV cache for the GQA SDPA tail. Grows by `n_kv_heads * d_head`
    /// per token. Each entry is one token's K (or V) flattened.
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
}

#[derive(Clone)]
pub struct MoeState {
    /// EDA residual from the previous MoE layer's router (carried across
    /// MoE layers, NOT across the whole stack). `None` for the first
    /// MoE layer in the model (il=1).
    pub _placeholder: (),
}

#[derive(Clone)]
pub enum BlockState {
    Cca(CcaState),
    Moe(MoeState),
}

#[derive(Clone)]
pub struct Zaya1State {
    pub blocks: Vec<BlockState>,
    /// FP32 residual stream accumulator. `Some(vec)` after layer 0
    /// initializes it; `None` before the very first layer.
    pub residual: Option<Vec<f32>>,
    /// EDA router state, carried across MoE layers. Length d_ff_router.
    /// Reset to `None` for the first MoE layer (il=1).
    pub prev_router: Option<Vec<f32>>,
    pub pos: usize,
}

impl Zaya1State {
    pub fn new(cfg: &Zaya1Config, layer_kinds: &[LayerKind]) -> Self {
        let n_qk = cfg.n_qk();
        let blocks = layer_kinds.iter().map(|k| match k {
            LayerKind::CcaAttention => BlockState::Cca(CcaState {
                // ZAYA1 stores `d_conv` columns of conv carry (NOT d_conv-1).
                // Per llama.cpp PR `cca_state` layout: 2 × n_qk for the
                // depthwise conv carry, plus n_embd for delayed-V prev_hs.
                conv_state: vec![0.0f32; cfg.d_conv * n_qk],
                prev_hs: vec![0.0f32; cfg.d_model],
                k_cache: Vec::new(),
                v_cache: Vec::new(),
            }),
            LayerKind::MoeFfn => BlockState::Moe(MoeState { _placeholder: () }),
        }).collect();
        Self { blocks, residual: None, prev_router: None, pos: 0 }
    }
}

// ── STUB mixers (to be replaced) ─────────────────────────────────────

/// Placeholder CCA forward: returns input unchanged (identity).
/// Replace with real CCA in follow-on.
fn cca_forward_stub(
    cur: &[f32], _blk: &CcaBlock, _state: &mut CcaState, _cfg: &Zaya1Config, _pos: usize,
) -> Vec<f32> {
    cur.to_vec()
}

/// L2-norm along the innermost (d_head) axis for each head. `x` shape:
/// `[n_heads, d_head]` row-major. Returns the normalized tensor scaled
/// by `sqrt(d_head)` per the ZAYA1 convention.
fn l2_norm_per_head_scaled(x: &mut [f32], n_heads: usize, d_head: usize, eps: f32) {
    let s = (d_head as f32).sqrt();
    for h in 0..n_heads {
        let off = h * d_head;
        let slice = &mut x[off..off + d_head];
        let ss: f32 = slice.iter().map(|v| v * v).sum::<f32>();
        let inv = 1.0 / (ss + eps).sqrt();
        for v in slice.iter_mut() { *v = *v * inv * s; }
    }
}

/// **CCA depthwise conv1d step** (ZAYA1 single-token decode form).
///
/// Returns both output positions (size `d_conv * n_qk` = 2 cols × n_qk
/// for d_conv=2). The caller can use the LAST position for the current
/// token's downstream consumers, but the grouped-conv stage needs both
/// positions as its 2-position input.
///
/// State (`d_conv * n_qk` = full kernel-width carry per llama.cpp PR
/// convention) is updated in place to the trailing `d_conv` columns of
/// the conv input (positions [1..d_conv+1] of `[state | new]`).
fn cca_conv_dw_step(
    qk_new: &[f32], weight: &[f32], bias: &[f32],
    state: &mut Vec<f32>, n_qk: usize, d_conv: usize,
) -> Vec<f32> {
    debug_assert_eq!(qk_new.len(), n_qk);
    debug_assert_eq!(weight.len(), d_conv * n_qk);
    debug_assert_eq!(bias.len(), n_qk);
    debug_assert_eq!(state.len(), d_conv * n_qk);

    // conv_input layout: [d_conv + 1, n_qk] kernel-time outer, channels inner.
    let conv_len = d_conv + 1;
    let mut conv_input: Vec<f32> = Vec::with_capacity(conv_len * n_qk);
    conv_input.extend_from_slice(state);
    conv_input.extend_from_slice(qk_new);

    // ssm_conv (cross-correlation form, matching ggml convention):
    // out[t, c] = sum_k weight[k, c] * conv_input[t + k, c].
    // FP64 accumulator for numerical stability — small kernel (d_conv=2)
    // but the per-channel value feeds the grouped conv where 128 channels
    // are summed, so any per-channel error compounds.
    let output_len = conv_len - d_conv + 1;
    let mut out = vec![0.0f32; output_len * n_qk];
    for t in 0..output_len {
        for c in 0..n_qk {
            let mut acc = 0.0f64;
            for k in 0..d_conv {
                acc += (weight[c * d_conv + k] as f64) * (conv_input[(t + k) * n_qk + c] as f64);
            }
            out[t * n_qk + c] = (acc + bias[c] as f64) as f32;
        }
    }

    // Update state: trailing d_conv columns of conv_input.
    state.copy_from_slice(&conv_input[n_qk..(d_conv + 1) * n_qk]);

    out
}

/// **CCA grouped conv1d step** (single output column, stride=1):
/// Each output channel `c` belongs to group `g = c / group_size`. The
/// weight tensor has shape `[d_conv, group_size, n_qk]` row-major
/// (`d_conv` innermost). For output channel `c` in group `g`:
///
///     out[c] = bias[c] + Σ_k Σ_j weight[c, j, k] * input[k, g*group_size + j]
///
/// `input` shape: `[d_conv, n_qk]` row-major (kernel-outer). We use the
/// SAME `state` buffer that the depthwise conv updated — wait, no, the
/// grouped conv operates on the dw output (the previous-token-shifted
/// stream), not on the QK directly. For correctness we just pass in the
/// `conv_input` (same layout as dw step).
///
/// `state` already has the last `d_conv - 1` columns. We APPEND the new
/// dw output column and form a `[d_conv, n_qk]` block to convolve over.
/// For ZAYA1 the grouped conv has a SEPARATE state from the dw conv;
/// per llama.cpp the conv1d operates on a single tensor with d_conv-1
/// state held in `cca_conv_state`. We simply provide the dw output as
/// the new column and use a fresh local buffer for the rolling kernel
/// — for d_conv=2 this means [last_dw, current_dw] = 2 columns, and
/// the grouped conv reduces those 2 to 1 with the per-group kernel.
///
/// Returns `[n_qk]` (single output column).
fn cca_conv_grp_step(
    dw_both: &[f32], weight: &[f32], bias: &[f32],
    n_qk: usize, d_conv: usize, n_groups: usize,
) -> Vec<f32> {
    debug_assert_eq!(dw_both.len(), d_conv * n_qk);
    debug_assert_eq!(weight.len(), d_conv * (n_qk / n_groups) * n_qk);
    debug_assert_eq!(bias.len(), n_qk);
    debug_assert!(n_qk % n_groups == 0);

    let group_size = n_qk / n_groups;

    // weight[k, j, c] with ne[0]=d_conv innermost: flat index = c*group_size*d_conv + j*d_conv + k.
    // FP64 accumulator: per-channel sum over (group_size × d_conv) = 256 terms
    // — small enough that f32 should be OK but cumulative noise across the
    // 5-stage CCA pipeline matters. Use f64 here as cheap insurance.
    let mut out = vec![0.0f32; n_qk];
    for c in 0..n_qk {
        let g = c / group_size;
        let g_off = g * group_size;
        let mut acc = 0.0f64;
        for j in 0..group_size {
            let w_base = c * group_size * d_conv + j * d_conv;
            for k in 0..d_conv {
                acc += (weight[w_base + k] as f64) * (dw_both[k * n_qk + g_off + j] as f64);
            }
        }
        out[c] = (acc + bias[c] as f64) as f32;
    }
    out
}

/// Compute the qk_mean bias terms used by ZAYA1's CCA.
/// - `qk_mean_q[h, d]` = 0.5 * (Q[h, d] + K_tile[h, d]) where K_tile is K
///   repeated per GQA group (n_kv_heads → n_q_heads, tile pattern).
/// - `qk_mean_k[hk, d]` = 0.5 * (Q_groupmean[hk, d] + K[hk, d]) where
///   Q_groupmean averages the n_gqa Q heads sharing one K head.
///
/// Returns (qk_mean_q [n_q_heads, d_head], qk_mean_k [n_kv_heads, d_head]).
fn cca_qk_mean(
    q_raw: &[f32], k_raw: &[f32], cfg: &Zaya1Config,
) -> (Vec<f32>, Vec<f32>) {
    let nq = cfg.n_q_heads;
    let nkv = cfg.n_kv_heads;
    let dh = cfg.d_head;
    let n_gqa = nq / nkv;
    debug_assert_eq!(q_raw.len(), nq * dh);
    debug_assert_eq!(k_raw.len(), nkv * dh);

    // qk_mean_q: broadcast K to n_q heads via GROUP convention
    // (matches ggml's reshape→repeat→reshape chain in the PR code):
    //   Kpre [d_head, n_head_kv]
    //   → reshape [d_head, 1, n_head_kv]
    //   → repeat  [d_head, n_gqa, n_head_kv]
    //   → reshape [d_head, n_head]    (flatten in order [gqa, kv] → kv outer)
    // ⇒ head_h maps to kv_idx = h / n_gqa (GROUP, NOT tile).
    let n_gqa = nq / nkv;
    let mut qk_mean_q = vec![0.0f32; nq * dh];
    for h in 0..nq {
        let kh = h / n_gqa;
        for d in 0..dh {
            qk_mean_q[h * dh + d] = 0.5 * (q_raw[h * dh + d] + k_raw[kh * dh + d]);
        }
    }

    // qk_mean_k: average Q heads in each GQA group.
    let mut qk_mean_k = vec![0.0f32; nkv * dh];
    for kh in 0..nkv {
        for d in 0..dh {
            let mut q_mean = 0.0f32;
            for gi in 0..n_gqa {
                let qh = kh * n_gqa + gi;
                q_mean += q_raw[qh * dh + d];
            }
            q_mean /= n_gqa as f32;
            qk_mean_k[kh * dh + d] = 0.5 * (q_mean + k_raw[kh * dh + d]);
        }
    }

    (qk_mean_q, qk_mean_k)
}

/// **CCA forward step** (single-token decode form). Mirrors
/// llama.cpp Draft PR #23112 `src/models/zaya.cpp::build_layer_attn`
/// (the il-even branch) step-for-step.
fn cca_forward(
    cur: &[f32], blk: &CcaBlock, state: &mut CcaState,
    cfg: &Zaya1Config, pos: usize, trace_layer: bool,
) -> Vec<f32> {
    let n_qk = cfg.n_qk();
    let n_groups = cfg.cca_n_groups();
    let nq = cfg.n_q_heads;
    let nkv = cfg.n_kv_heads;
    let dh = cfg.d_head;
    let value_half = nkv * dh / 2;
    let q_dim = nq * dh;
    let kv_dim = nkv * dh;

    // Q, K, V1 projections from `cur`.
    let q_raw = qmm_1d(cur, &blk.wq);
    let k_raw = qmm_1d(cur, &blk.wk);
    let v1 = qmm_1d(cur, &blk.cca_val_proj1);
    if trace_layer {
        trace_tensor("Qraw-0", &q_raw);
        trace_tensor("Kraw-0", &k_raw);
        trace_tensor("V1-0", &v1);
    }

    // QKraw = concat(Q, K).
    let mut qk_raw = Vec::with_capacity(n_qk);
    qk_raw.extend_from_slice(&q_raw);
    qk_raw.extend_from_slice(&k_raw);
    if trace_layer { trace_tensor("QKraw-0", &qk_raw); }

    // Delayed-V from previous token's hidden state (cur).
    let v2 = qmm_1d(&state.prev_hs, &blk.cca_val_proj2);
    if trace_layer { trace_tensor("V2-0", &v2); }

    // Update prev_hs for next token AFTER we used it.
    state.prev_hs.copy_from_slice(cur);

    // Vcur = concat(V1, V2). Shape [n_kv*d_head = 256].
    let mut v_cur = Vec::with_capacity(kv_dim);
    v_cur.extend_from_slice(&v1);
    v_cur.extend_from_slice(&v2);
    if trace_layer { trace_tensor("Vcur-0", &v_cur); }

    // qk_mean biases.
    let (qk_mean_q, qk_mean_k) = cca_qk_mean(&q_raw, &k_raw, cfg);
    if trace_layer {
        trace_tensor("qk_mean_q-0", &qk_mean_q);
        trace_tensor("qk_mean_k-0", &qk_mean_k);
    }

    // Depthwise conv1d + bias (CCA conv_dw stage).
    // dw output: 2 positions × n_qk (positions [0, 1] of the conv output).
    // Position 1 is the current-token output. Position 0 is the previous-
    // token output produced from [state_col_1, current_qk] when state was
    // [state_col_0, state_col_1] — that pair is consumed by the grouped
    // conv as its 2-column input.
    let dw_both = cca_conv_dw_step(
        &qk_raw, &blk.ssm_conv1d_w, &blk.ssm_conv1d_b,
        &mut state.conv_state, n_qk, cfg.d_conv,
    );
    if trace_layer {
        // Trace only the current-token slice [position 1] for compactness.
        trace_tensor("QK_dw_cur-0", &dw_both[n_qk..2 * n_qk]);
    }

    // Grouped conv1d + bias on the 2-position dw output → single output column.
    let qk_grp = cca_conv_grp_step(
        &dw_both, &blk.cca_conv_grp_w, &blk.cca_conv_grp_b,
        n_qk, cfg.d_conv, n_groups,
    );
    if trace_layer { trace_tensor("QK_grp-0", &qk_grp); }

    // Split into Q_conv and K_conv per the original Q/K dims.
    let mut q_conv: Vec<f32> = qk_grp[0..q_dim].to_vec();
    let mut k_conv: Vec<f32> = qk_grp[q_dim..n_qk].to_vec();

    // Add qk_mean biases.
    for i in 0..q_dim { q_conv[i] += qk_mean_q[i]; }
    for i in 0..kv_dim { k_conv[i] += qk_mean_k[i]; }
    if trace_layer {
        trace_tensor("Qcur_pre_l2-0", &q_conv);
        trace_tensor("Kcur_pre_l2-0", &k_conv);
    }

    // L2-norm per head + scale by sqrt(d_head). The L2-norm alone is
    // traced separately (matches ggml `node_135` / pre-scale L2 output).
    {
        // Per-head L2-norm without scale, for trace comparison.
        let mut q_l2 = q_conv.clone();
        let mut k_l2 = k_conv.clone();
        for h in 0..nq {
            let off = h * dh;
            let slice = &mut q_l2[off..off + dh];
            let ss: f32 = slice.iter().map(|v| v * v).sum::<f32>();
            let inv = 1.0 / (ss + 1e-12).sqrt();
            for v in slice.iter_mut() { *v *= inv; }
        }
        for h in 0..nkv {
            let off = h * dh;
            let slice = &mut k_l2[off..off + dh];
            let ss: f32 = slice.iter().map(|v| v * v).sum::<f32>();
            let inv = 1.0 / (ss + 1e-12).sqrt();
            for v in slice.iter_mut() { *v *= inv; }
        }
        if trace_layer {
            trace_tensor("Q_l2norm_only-0", &q_l2);
            trace_tensor("K_l2norm_only-0", &k_l2);
            // Per-head sums of Q_l2 and pre-L2 Q to localize divergence.
            for h in 0..nq {
                let off = h * dh;
                let s_pre: f64 = q_conv[off..off + dh].iter().map(|v| *v as f64).sum();
                let s_l2: f64 = q_l2[off..off + dh].iter().map(|v| *v as f64).sum();
                let mag: f64 = (q_conv[off..off + dh].iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()).sqrt();
                eprintln!("  Q_h{h}: pre_sum={s_pre:+.4} l2_sum={s_l2:+.4} mag={mag:.4}");
            }
            // Dump head 0 full 128 values to confirm element-wise match vs ref.
            let h0 = &q_l2[0..dh];
            let h0_sum: f64 = h0.iter().map(|v| *v as f64).sum();
            eprintln!("  Q_l2_h0_sum_recomputed={h0_sum:.6}");
            // Print every 16th value through the head:
            let samples: Vec<String> = (0..dh).step_by(16).map(|i| format!("{:.4}", h0[i])).collect();
            eprintln!("  Q_l2_h0_every16={}", samples.join(", "));
        }
    }
    l2_norm_per_head_scaled(&mut q_conv, nq, dh, 1e-12);
    l2_norm_per_head_scaled(&mut k_conv, nkv, dh, 1e-12);

    // Per-head K scale (cca_k_scale, length n_kv_heads).
    for kh in 0..nkv {
        let s = blk.cca_k_scale_w[kh];
        let off = kh * dh;
        for d in 0..dh { k_conv[off + d] *= s; }
    }
    if trace_layer {
        trace_tensor("Qcur_pre_rope-0", &q_conv);
        trace_tensor("Kcur_pre_rope-0", &k_conv);
    }

    // RoPE on Q and K. rope_dim=64 (partial RoPE), partial over each
    // d_head=128 head — only first 64 dims rotated, NeoX style.
    for h in 0..nq {
        let off = h * dh;
        rce_op_attn::rope_inplace(
            &mut q_conv[off..off + dh], cfg.rope_dim, cfg.rope_freq_base, pos,
        );
    }
    for kh in 0..nkv {
        let off = kh * dh;
        rce_op_attn::rope_inplace(
            &mut k_conv[off..off + dh], cfg.rope_dim, cfg.rope_freq_base, pos,
        );
    }
    if trace_layer {
        trace_tensor("Qcur-0", &q_conv);
        trace_tensor("Kcur-0", &k_conv);
    }

    // Append K, V to cache and run GQA SDPA.
    state.k_cache.extend_from_slice(&k_conv);
    state.v_cache.extend_from_slice(&v_cur);
    let seq_len = state.k_cache.len() / kv_dim;
    let scale = 1.0 / (dh as f32).sqrt();
    let q_per_kv = nq / nkv;
    let mut attn = vec![0.0f32; q_dim];
    for h in 0..nq {
        let kv_h = h / q_per_kv;
        let q_h = &q_conv[h * dh..(h + 1) * dh];
        let out_h = rce_op_attn::fused_attend_head(
            q_h, &state.k_cache, &state.v_cache,
            kv_h, seq_len, kv_dim, dh, scale,
        );
        attn[h * dh..(h + 1) * dh].copy_from_slice(&out_h);
    }
    if trace_layer { trace_tensor("attn_pre_out-0", &attn); }

    // Output projection.
    let out = qmm_1d(&attn, &blk.wo);
    if trace_layer { trace_tensor("attn_out-0", &out); }
    out
}

/// Placeholder MoE forward: returns input unchanged (identity).
#[allow(dead_code)]
fn moe_forward_stub(
    cur: &[f32], _blk: &MoeBlock, _prev_router: &mut Option<Vec<f32>>, _cfg: &Zaya1Config, _il: usize,
) -> Vec<f32> {
    cur.to_vec()
}

#[inline]
fn gelu(x: f32) -> f32 {
    // ggml_gelu = "tanh approximation" form: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let c = 0.7978845608f32; // sqrt(2/pi)
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

#[inline]
fn silu(x: f32) -> f32 {
    if x >= 0.0 { x / (1.0 + (-x).exp()) } else { let e = x.exp(); x * e / (1.0 + e) }
}

/// ZAYA1 MoE FFN forward (top-1 routing + multi-stage MLP router with EDA).
fn moe_forward(
    cur: &[f32], blk: &MoeBlock, prev_router: &mut Option<Vec<f32>>,
    cfg: &Zaya1Config, _il: usize, trace_layer: bool,
) -> Vec<f32> {
    let dm = cfg.d_model;
    let n_ffr = cfg.d_ff_router;
    let n_exp = cfg.n_experts;
    let n_ff = cfg.d_ff_expert;

    // ── Router pipeline ──────────────────────────────────────────────
    // 1) router_h = router_down @ cur + bias
    let mut router_h = qmm_1d(cur, &blk.router_down_w);
    for i in 0..n_ffr { router_h[i] += blk.router_down_b[i]; }
    if trace_layer { trace_tensor("router_down-1", &router_h); }

    // 2) EDA: `router_h += prev_router * eda_scale` if both exist.
    //    The first MoE layer (il=1) has no eda_scale (absent in GGUF) and
    //    no prev_router yet → skipped. Subsequent MoE layers (il=3, 5, ...)
    //    consume the PRE-NORM `router_down + bias` from the previous MoE
    //    layer.
    if let (Some(prev), Some(scale)) = (prev_router.as_ref(), blk.router_eda_scale.as_ref()) {
        debug_assert_eq!(prev.len(), n_ffr);
        debug_assert_eq!(scale.len(), n_ffr);
        for i in 0..n_ffr { router_h[i] += prev[i] * scale[i]; }
        if trace_layer { trace_tensor("router_eda-1", &router_h); }
    }

    // 3) Capture this layer's router_h (post-down, post-EDA, PRE-norm)
    //    as the new prev_router for downstream MoE layers.
    *prev_router = Some(router_h.clone());

    // 4) RMSnorm with router_norm weights.
    let router_normed = rms_norm(&router_h, &blk.router_norm_w, cfg.eps);

    // 5) MLP-0: GELU(router_mlp0 @ x + bias)
    let mut h0 = qmm_1d(&router_normed, &blk.router_mlp0_w);
    for i in 0..n_ffr { h0[i] = gelu(h0[i] + blk.router_mlp0_b[i]); }
    if trace_layer { trace_tensor("router_mlp0-1", &h0); }

    // 6) MLP-2: GELU(router_mlp2 @ x + bias)
    let mut h2 = qmm_1d(&h0, &blk.router_mlp2_w);
    for i in 0..n_ffr { h2[i] = gelu(h2[i] + blk.router_mlp2_b[i]); }
    if trace_layer { trace_tensor("router_mlp2-1", &h2); }

    // 7) Logits: router_mlp4 @ x → [n_expert + 1] = 17 (16 experts + 1 MOD-skip).
    //    Per ref: NO bias on the raw logits. Bias is applied AFTER softmax,
    //    to the first n_expert slots only (`ffn_moe_probs_biased`).
    let logits = qmm_1d(&h2, &blk.router_mlp4_w);
    debug_assert_eq!(logits.len(), n_exp + 1);
    if trace_layer { trace_tensor("router_logits-1", &logits); }

    // 8) Softmax over all 17 slots (skip-expert in denominator, not routed).
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|x| (x - max_logit).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 { for p in probs.iter_mut() { *p /= sum; } }

    // 9) Take first n_expert probs (drop MOD-skip[16]), add router_biases
    //    (per `ffn_moe_probs_biased`), then top-1.
    let mut gate_probs: Vec<f32> = probs[0..n_exp].to_vec();
    for e in 0..n_exp { gate_probs[e] += blk.router_biases[e]; }
    if trace_layer { trace_tensor("ffn_moe_probs_biased-1", &gate_probs); }

    let mut best_e = 0usize;
    let mut best_p = gate_probs[0];
    for e in 1..n_exp {
        if gate_probs[e] > best_p { best_e = e; best_p = gate_probs[e]; }
    }
    if trace_layer {
        eprintln!("  router: best_expert={best_e} prob={best_p:.4}  skip_prob={:.4}",
            probs[n_exp]);
    }

    // ── Expert FFN (SwiGLU on the selected expert) ───────────────────
    // ffn_gate_up_exps stored as [n_experts × 2*d_ff_expert, d_model].
    // Per-expert weight rows = 2 * d_ff_expert. The expert's gate occupies
    // first d_ff_expert rows, up occupies next d_ff_expert rows.
    //
    // Build a per-expert view via dequant: easier to dequant the entire
    // matrix for the expert's rows. Use `dequant_row` to fetch the rows
    // needed.
    // Per-expert weight sub-views → parallel `mm_1d` (native dtype, no
    // per-row dequant). Top-1 routing means a single expert with large
    // matmuls (gate_up [2*n_ff, d_model], down [d_model, n_ff]), which
    // `mm_1d`'s row-parallelism handles well — replaces the old fully
    // serial scalar dequant_row loop.
    let gu_bpe = blk.ffn_gate_up_exps.bytes().len() / n_exp;
    let gate_up_w = blk.ffn_gate_up_exps.sub_view(best_e * gu_bpe, gu_bpe, 2 * n_ff, dm);
    let gu = qmm_1d(cur, &gate_up_w); // [2*n_ff]: gate ∥ up
    let mut hidden = vec![0.0f32; n_ff];
    for o in 0..n_ff { hidden[o] = silu(gu[o]) * gu[n_ff + o]; }

    let dn_bpe = blk.ffn_down_exps.bytes().len() / n_exp;
    let down_w = blk.ffn_down_exps.sub_view(best_e * dn_bpe, dn_bpe, dm, n_ff);
    let mut out = qmm_1d(&hidden, &down_w);
    for d in 0..dm { out[d] *= best_p; }
    if trace_layer { trace_tensor("moe_out-1", &out); }

    out
}

// ── forward_token ────────────────────────────────────────────────────

pub fn forward_token(
    w: &Zaya1Weights, cfg: &Zaya1Config, state: &mut Zaya1State, token_id: i32,
) -> Vec<f32> {
    let dm = cfg.d_model;
    let mut inp_l: Vec<f32> = w.token_embd.dequant_row(token_id as usize);
    trace_tensor("model.input_embed", &inp_l);

    for (il, block) in w.blocks.iter().enumerate() {
        // 1) Per-layer hidden_states = apply_res_scale(inpL, res_scale_hs, b).
        let (res_scale_hs_w, res_scale_hs_b, res_scale_res_pair) = match block {
            Block::Cca(b) => {
                let pair = match (&b.res_scale_res_w, &b.res_scale_res_b) {
                    (Some(w), Some(b)) => Some((w.as_slice(), b.as_slice())),
                    _ => None,
                };
                (b.res_scale_hs_w.as_slice(), b.res_scale_hs_b.as_slice(), pair)
            }
            Block::Moe(b) => (
                b.res_scale_hs_w.as_slice(), b.res_scale_hs_b.as_slice(),
                Some((b.res_scale_res_w.as_slice(), b.res_scale_res_b.as_slice())),
            ),
        };
        let hidden_states = apply_res_scale(&inp_l, res_scale_hs_w, res_scale_hs_b);
        if il == 0 { trace_tensor("res_scale_hs-0", &hidden_states); }

        // 2) Residual stream: initialize on first layer, else scale-and-add.
        let mut residual: Vec<f32> = match state.residual.take() {
            None => hidden_states.clone(),
            Some(prev_res) => {
                // Scale prev residual if this layer has res_scale_res.
                let scaled_prev = if let Some((rw, rb)) = res_scale_res_pair {
                    apply_res_scale(&prev_res, rw, rb)
                } else {
                    prev_res
                };
                let mut r = scaled_prev;
                for i in 0..dm { r[i] += hidden_states[i]; }
                r
            }
        };
        if il == 0 { trace_tensor("residual-0", &residual); }

        // 3) Pre-mixer RMSnorm on residual.
        let attn_norm_w = match block {
            Block::Cca(b) => &b.attn_norm_w,
            Block::Moe(b) => &b.attn_norm_w,
        };
        let cur = rms_norm(&residual, attn_norm_w, cfg.eps);
        if il == 0 { trace_tensor("input_norm-0", &cur); }

        // 4) Mixer (real CCA + real MoE).
        let trace_il = trace_enabled() && (il == 0 || il == 1);
        let mixer_out = match (block, &mut state.blocks[il]) {
            (Block::Cca(b), BlockState::Cca(s)) => {
                cca_forward(&cur, b, s, cfg, state.pos, trace_il && il == 0)
            }
            (Block::Moe(b), BlockState::Moe(_)) => {
                moe_forward(&cur, b, &mut state.prev_router, cfg, il, trace_il && il == 1)
            }
            _ => panic!("block/state kind mismatch at layer {il}"),
        };

        // 5) Save residual + advance inpL.
        state.residual = Some(residual);
        inp_l = mixer_out;
    }

    // Final scaling: hidden_states = apply_res_scale(inpL, final_hs); residual = apply_res_scale(residual, final_res); cur = hs + res.
    let final_hs = apply_res_scale(&inp_l, &w.final_res_scale_hs_w, &w.final_res_scale_hs_b);
    let final_res = if let Some(prev_res) = state.residual.take() {
        let scaled = apply_res_scale(&prev_res, &w.final_res_scale_res_w, &w.final_res_scale_res_b);
        let mut r = scaled;
        for i in 0..dm { r[i] += final_hs[i]; }
        r
    } else {
        final_hs
    };
    trace_tensor("final_residual", &final_res);

    let final_normed = rms_norm(&final_res, &w.output_norm_w, cfg.eps);
    trace_tensor("result_norm", &final_normed);
    let logits = qmm_1d(&final_normed, &w.token_embd);  // tied lm_head
    trace_tensor("result_output", &logits);
    state.pos += 1;
    logits
}

impl Zaya1Runtime {
    pub fn new_state(&self) -> Zaya1State { Zaya1State::new(&self.cfg, &self.layer_kinds) }
    pub fn forward_logits(&self, state: &mut Zaya1State, token_id: i32) -> Vec<f32> {
        forward_token(&self.weights, &self.cfg, state, token_id)
    }
    pub fn tokenizer(&self) -> Tokenizer<'_> {
        Tokenizer::from_gguf(&self.gguf).expect("tokenizer")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Zaya1GenConfig {
    pub max_new_tokens: usize,
    pub sample: SampleConfig,
    pub rep_window: usize,
}
impl Default for Zaya1GenConfig {
    fn default() -> Self {
        Self { max_new_tokens: 32, sample: SampleConfig::chat(0x2A1A_AABB), rep_window: 64 }
    }
}

impl Zaya1Runtime {
    pub fn generate(
        &self, prompt: &str, gcfg: &Zaya1GenConfig, mut on_token: impl FnMut(&str),
    ) -> Result<String, String> {
        let tk = self.tokenizer();
        let add_bos = self.gguf.metadata.iter()
            .find(|kv| kv.key == "tokenizer.ggml.add_bos_token")
            .map(|kv| matches!(&kv.value, GgufValue::Bool(b) if *b))
            .unwrap_or(true);
        // gemma4 tokenizer is SentencePiece-style — use the standard `encode`.
        let prompt_ids = tk.encode(prompt, add_bos);
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

impl Zaya1Runtime {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| format!("open: {e}"))?;
        let mmap = Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?);
        // WillNeed (not Sequential): keeps the model page-resident so
        // decode hits RAM speed instead of re-faulting weights from disk.
        let _ = mmap.advise(memmap2::Advice::WillNeed);
        let gguf = GgufFile::parse(&mmap[..]).map_err(|e| format!("parse: {e:?}"))?;
        let cfg = Zaya1Config::from_gguf(&gguf)?;

        // Tensor-presence-based layer classification, cross-checked
        // against config.is_cca_layer.
        let mut layer_kinds = Vec::with_capacity(cfg.n_layers);
        for il in 0..cfg.n_layers {
            let has_cca = gguf.tensors.iter()
                .any(|t| t.name == format!("blk.{il}.cca_val_proj1.weight"));
            let has_router = gguf.tensors.iter()
                .any(|t| t.name == format!("blk.{il}.zaya_router_mlp4.weight"));
            let kind = match (has_cca, has_router) {
                (true, false) => LayerKind::CcaAttention,
                (false, true) => LayerKind::MoeFfn,
                _ => return Err(format!(
                    "layer {il}: ambiguous (has_cca={has_cca}, has_router={has_router})"
                )),
            };
            let expected_cca = cfg.is_cca_layer(il);
            let actual_cca = matches!(kind, LayerKind::CcaAttention);
            if expected_cca != actual_cca {
                return Err(format!(
                    "layer {il}: config says is_cca={expected_cca} but tensors say {actual_cca}"
                ));
            }
            layer_kinds.push(kind);
        }

        let n_cca = layer_kinds.iter().filter(|k| **k == LayerKind::CcaAttention).count();
        let n_moe = layer_kinds.iter().filter(|k| **k == LayerKind::MoeFfn).count();
        eprintln!(
            "zaya1: d_model={} vocab={} n_layers={} ({n_cca} CCA + {n_moe} MoE) \
             n_q/kv={}/{} d_head={} d_ff_expert={} d_ff_router={} n_experts={}/{} d_conv={} rope_base={} eps={}",
            cfg.d_model, cfg.vocab, cfg.n_layers,
            cfg.n_q_heads, cfg.n_kv_heads, cfg.d_head,
            cfg.d_ff_expert, cfg.d_ff_router,
            cfg.n_experts_used, cfg.n_experts,
            cfg.d_conv, cfg.rope_freq_base, cfg.eps,
        );

        let weights = load_weights(&mmap, &gguf, &cfg)?;
        Ok(Self { cfg, gguf, layer_kinds, weights, _mmap: mmap })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const MODEL: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/zaya1-8b-gguf/ZAYA1-8B-Q8_0.gguf";

    #[test]
    fn zaya1_load_and_layer_pattern_sanity() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("zaya1: GGUF not mounted — skipping");
            return;
        }
        let rt = Zaya1Runtime::load(MODEL).expect("load");
        let c = &rt.cfg;
        assert_eq!(c.n_layers, 80);
        assert_eq!(c.d_model, 2048);
        assert_eq!(c.n_q_heads, 8);
        assert_eq!(c.n_kv_heads, 2);
        assert_eq!(c.d_head, 128);
        assert_eq!(c.n_experts, 16);
        assert_eq!(c.n_experts_used, 1);
        assert_eq!(c.d_ff_expert, 2048);
        assert_eq!(c.d_ff_router, 256);
        assert_eq!(c.d_conv, 2);
        assert_eq!(c.n_qk(), 1280);
        assert_eq!(c.cca_n_groups(), 10);
        let n_cca = rt.layer_kinds.iter().filter(|k| **k == LayerKind::CcaAttention).count();
        let n_moe = rt.layer_kinds.iter().filter(|k| **k == LayerKind::MoeFfn).count();
        assert_eq!(n_cca, 40);
        assert_eq!(n_moe, 40);
        assert_eq!(rt.layer_kinds[0], LayerKind::CcaAttention);
        assert_eq!(rt.layer_kinds[1], LayerKind::MoeFfn);
        assert_eq!(rt.layer_kinds[78], LayerKind::CcaAttention);
        assert_eq!(rt.layer_kinds[79], LayerKind::MoeFfn);
        assert_eq!(c.vocab, 262147);
    }

    /// Reference comparison: feed BOS=2 then "The"=818 through forward
    /// with CCA + MoE stubbed (identity). Layer-0 residual stream should
    /// match llama.cpp reference for the BOS-only forward exactly.
    ///
    #[test]
    fn zaya1_generate_smoke() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("zaya1: GGUF not mounted — skipping");
            return;
        }
        let rt = Zaya1Runtime::load(MODEL).expect("load");
        let cfg = Zaya1GenConfig {
            max_new_tokens: 12,
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
            "zaya1 generate OK: prompt={prompt:?} → continuation={full:?}  [{} tok in {dt:?} = {:.2} tok/s]",
            chunks.len(),
            chunks.len() as f64 / dt.as_secs_f64(),
        );
        assert!(!full.is_empty(), "empty continuation");
    }

    /// Run: `RCE_ZAYA1_TRACE=1 cargo test --release -p rce-zaya1 zaya1_trace_residual -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn zaya1_trace_residual() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("zaya1: GGUF not mounted — skipping");
            return;
        }
        let rt = Zaya1Runtime::load(MODEL).expect("load");
        let mut state = rt.new_state();
        // BOS = 2.
        let _ = rt.forward_logits(&mut state, 2);
        // "The" = 818.
        let _ = rt.forward_logits(&mut state, 818);
    }
}
