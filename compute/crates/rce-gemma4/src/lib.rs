//! Gemma 4 (E2B) text decoder on the rce substrate.
//!
//! Authoritative: sjl623/llama.cpp `src/models/gemma4.cpp`
//! (`llama_model_gemma4::graph`). Per layer:
//!
//! ```text
//!   inpL_scaled = inpL                                  // input embed × √n_embd (token path)
//!   cur  = rms_norm(inpL, attn_norm)
//!   Q    = wq(cur)            reshape [head_dim,n_head,1]
//!   Q    = rms_norm_per_head(Q, attn_q_norm)            // dim = head_dim
//!   Q    = rope_neox(Q, pos, rope_base, rope_dim)       // per-layer base+dim
//!   K    = wk(cur)            reshape [head_dim,n_kv,1]
//!   K    = rms_norm_per_head(K, attn_k_norm)
//!   K    = rope_neox(K, pos, rope_base, rope_dim)
//!   V    = wv(cur)            reshape [head_dim,n_kv,1]
//!   V    = rms_norm(V, eps)                              // V-norm: NO learned weight
//!   cur  = attn(Q,K,V, scale=1.0)                        // SWA: window=512 ; full: causal
//!   cur  = rms_norm(cur, attn_post_norm)
//!   attn_out = inpL + cur                                // residual 1 (post-attn)
//!   cur  = rms_norm(attn_out, ffn_norm)
//!   cur  = ffn_down(gelu(ffn_gate(cur)) * ffn_up(cur))   // GELU-PAR, no SwiGLU
//!   cur  = rms_norm(cur, ffn_post_norm)
//!   cur  = attn_out + cur                                // residual 2 (post-ffn)
//!   // PLE: pe_in = cur ; cur = inp_gate(cur) ; cur = gelu(cur) ;
//!   //      cur *= per_layer_slice[il] ; cur = proj(cur) ;
//!   //      cur = rms_norm(cur, per_layer_post_norm) ; cur = pe_in + cur
//!   if out_scale: cur *= out_scale[il]                    // scalar
//! ```
//! Final: `cur = rms_norm(inpL, output_norm)` →
//!        `logits = output(cur)` →
//!        `logits = softcap · tanh(logits / softcap)` (softcap=30).
//!
//! **Per-layer hparams**: SWA layers have head_dim=256, rope_dim=256,
//! rope_base=10000, window=512. Full layers (every 5th): head_dim=512,
//! rope_dim=512, rope_base=1e6, no window.

use rce_ingest_gguf::{
    read_tensor_f32, sample_with_history, GgufFile, GgufValue, SampleConfig, Tokenizer,
};
use rce_op_qmm::QuantWeight;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct LayerH {
    pub is_swa: bool,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_base: f32,
    pub d_ff: usize,
    pub q_dim: usize,  // n_head * head_dim
    pub kv_dim: usize, // n_kv * head_dim
}

#[derive(Clone, Debug)]
pub struct Gemma4Config {
    pub d_model: usize,
    pub vocab: usize,
    pub n_layers: usize,
    pub n_embd_per_layer: usize,
    pub context_length: usize,
    pub eps: f32,
    pub final_logit_softcap: f32,
    pub sliding_window: usize,
    pub layers: Vec<LayerH>,
    /// Gemma 4 shared-KV: first `n_layer_kv_from_start` layers have
    /// their own K/V cache; later layers reuse an earlier layer's
    /// cache (per the `reuse` lambda in `llama_model.cpp:2009-2015`).
    /// For E2B: `block_count - attention.shared_kv_layers = 35 - 20 = 15`.
    pub n_layer_kv_from_start: usize,
}

impl Gemma4Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf.architecture().ok_or("missing arch")?;
        if arch != "gemma4" {
            return Err(format!("expected gemma4, got {arch}"));
        }
        let u32m = |k: &str| -> Option<u32> {
            gguf.metadata.iter().find(|kv| kv.key == format!("gemma4.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::U32(v) => Some(*v), _ => None })
        };
        let f32m = |k: &str| -> Option<f32> {
            gguf.metadata.iter().find(|kv| kv.key == format!("gemma4.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::F32(v) => Some(*v), _ => None })
        };
        let d_model = u32m("embedding_length").ok_or("embedding_length")? as usize;
        let vocab = gguf.tensors.iter().find(|t| t.name == "token_embd.weight")
            .and_then(|t| t.dims.get(1).copied()).map(|d| d as usize)
            .ok_or("vocab from token_embd")?;
        let n_layers = u32m("block_count").ok_or("block_count")? as usize;
        let n_head = u32m("attention.head_count").ok_or("head_count")? as usize;
        let n_kv = u32m("attention.head_count_kv").unwrap_or(n_head as u32) as usize;
        let key_len = u32m("attention.key_length").ok_or("key_length")? as usize;
        let key_len_swa = u32m("attention.key_length_swa").unwrap_or(key_len as u32) as usize;
        let rope_dim = u32m("rope.dimension_count").unwrap_or(key_len as u32) as usize;
        let rope_dim_swa = u32m("rope.dimension_count_swa").unwrap_or(rope_dim as u32) as usize;
        let rope_base = f32m("rope.freq_base").unwrap_or(1e6);
        let rope_base_swa = f32m("rope.freq_base_swa").unwrap_or(10_000.0);
        let eps = f32m("attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let n_embd_per_layer = u32m("embedding_length_per_layer_input").unwrap_or(0) as usize;
        let final_logit_softcap = f32m("final_logit_softcapping").unwrap_or(0.0);
        let context_length = u32m("context_length").unwrap_or(8192) as usize;
        let sliding_window = u32m("attention.sliding_window").unwrap_or(0) as usize;
        let shared_kv = u32m("attention.shared_kv_layers").unwrap_or(0) as usize;
        let n_layer_kv_from_start = n_layers.saturating_sub(shared_kv);

        // sliding_window_pattern: Array<Bool> of length n_layers, true = SWA.
        let swa_pat: Vec<bool> = gguf.metadata.iter()
            .find(|kv| kv.key == "gemma4.attention.sliding_window_pattern")
            .and_then(|kv| match &kv.value {
                GgufValue::Array(a) => Some(a.iter().filter_map(|v| match v {
                    GgufValue::Bool(b) => Some(*b), _ => None,
                }).collect()),
                _ => None,
            })
            .ok_or("missing sliding_window_pattern")?;
        if swa_pat.len() != n_layers {
            return Err(format!("swa_pat len {} != n_layers {n_layers}", swa_pat.len()));
        }

        // feed_forward_length: per-layer Array<I32/U32> (E2B) OR a uniform
        // scalar (E4B) broadcast across all layers.
        let ff_arr: Vec<usize> = gguf.metadata.iter()
            .find(|kv| kv.key == "gemma4.feed_forward_length")
            .and_then(|kv| match &kv.value {
                GgufValue::Array(a) => Some(a.iter().filter_map(|v| match v {
                    GgufValue::I32(x) => Some(*x as usize),
                    GgufValue::U32(x) => Some(*x as usize),
                    _ => None,
                }).collect::<Vec<_>>()),
                GgufValue::I32(x) => Some(vec![*x as usize; n_layers]),
                GgufValue::U32(x) => Some(vec![*x as usize; n_layers]),
                _ => None,
            })
            .ok_or("missing feed_forward_length")?;
        if ff_arr.len() != n_layers {
            return Err(format!("ff array len {} != n_layers {}", ff_arr.len(), n_layers));
        }

        let layers: Vec<LayerH> = (0..n_layers).map(|i| {
            let is_swa = swa_pat[i];
            let head_dim = if is_swa { key_len_swa } else { key_len };
            let rope_dim_l = if is_swa { rope_dim_swa } else { rope_dim };
            let rope_base_l = if is_swa { rope_base_swa } else { rope_base };
            LayerH {
                is_swa, n_head, n_kv, head_dim,
                rope_dim: rope_dim_l, rope_base: rope_base_l,
                d_ff: ff_arr[i],
                q_dim: n_head * head_dim,
                kv_dim: n_kv * head_dim,
            }
        }).collect();

        Ok(Gemma4Config {
            d_model, vocab, n_layers, n_embd_per_layer, context_length, eps,
            final_logit_softcap, sliding_window, layers, n_layer_kv_from_start,
        })
    }
}

pub struct Gemma4Block {
    pub attn_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub attn_post_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub ffn_post_norm: Vec<f32>,
    pub per_layer_post_norm: Vec<f32>, // PLE post-norm = "post_norm.weight"
    pub wq: QuantWeight, pub wk: QuantWeight, pub wv: QuantWeight, pub wo: QuantWeight,
    pub ffn_gate: QuantWeight, pub ffn_up: QuantWeight, pub ffn_down: QuantWeight,
    pub inp_gate: QuantWeight,    // PLE: n_embd → n_embd_per_layer
    pub proj: QuantWeight,        // PLE: n_embd_per_layer → n_embd
    pub out_scale: Option<f32>,   // optional scalar (read F32, [1])
}

pub struct Gemma4Weights {
    pub token_embd: QuantWeight,              // [d_model, vocab]
    pub per_layer_tok_embd: QuantWeight,      // [n_embd_per_layer * n_layers, vocab]
    pub per_layer_model_proj: QuantWeight,    // [d_model, n_embd_per_layer * n_layers]
    pub per_layer_proj_norm: Vec<f32>,        // [n_embd_per_layer]
    pub output_norm: Vec<f32>,                // [d_model]
    pub output: QuantWeight,                  // tied to token_embd (TENSOR_DUPLICATED in source)
    /// Proportional-RoPE frequency factors for FULL-attention layers
    /// (per-dim, `head_dim/2` entries). Source: `rope_freqs.weight`,
    /// TENSOR_DUPLICATED across full-attn layers. SWA layers do NOT use
    /// it. `inv_freq[i] /= rope_freqs[i]` modifies the per-dim RoPE
    /// frequency for proper context-length extension.
    pub rope_freqs_full: Option<Vec<f32>>,
    pub blocks: Vec<Gemma4Block>,
}

pub fn load_weights(
    mmap: &Arc<Mmap>, gguf: &GgufFile, cfg: &Gemma4Config,
) -> Result<Gemma4Weights, String> {
    let bytes: &[u8] = &mmap[..];
    let f32n = |n: &str| read_tensor_f32(bytes, gguf, n).map_err(|e| format!("{n}: {e:?}"));
    let qw = |n: &str, r: usize, c: usize| QuantWeight::from_gguf(
        Arc::clone(mmap), gguf, n, r, c).map_err(|e| format!("{n}: {e}"));

    let token_embd = qw("token_embd.weight", cfg.vocab, cfg.d_model)?;
    let per_layer_tok_embd = qw(
        "per_layer_token_embd.weight",
        cfg.vocab, cfg.n_embd_per_layer * cfg.n_layers,
    )?;
    let per_layer_model_proj = qw(
        "per_layer_model_proj.weight",
        cfg.n_embd_per_layer * cfg.n_layers, cfg.d_model,
    )?;
    let per_layer_proj_norm = f32n("per_layer_proj_norm.weight")?;
    let output_norm = f32n("output_norm.weight")?;
    // Output is tied — same tensor as token_embd, separately materialised
    // (TENSOR_DUPLICATED in the source). We just re-use token_embd handle.
    let output = qw("token_embd.weight", cfg.vocab, cfg.d_model)?;

    let mut blocks = Vec::with_capacity(cfg.n_layers);
    for il in 0..cfg.n_layers {
        let l = &cfg.layers[il];
        let bn = format!("blk.{il}");
        let attn_norm = f32n(&format!("{bn}.attn_norm.weight"))?;
        let q_norm = f32n(&format!("{bn}.attn_q_norm.weight"))?;
        let k_norm = f32n(&format!("{bn}.attn_k_norm.weight"))?;
        let attn_post_norm = f32n(&format!("{bn}.post_attention_norm.weight"))?;
        let ffn_norm = f32n(&format!("{bn}.ffn_norm.weight"))?;
        let ffn_post_norm = f32n(&format!("{bn}.post_ffw_norm.weight"))?;
        let per_layer_post_norm = f32n(&format!("{bn}.post_norm.weight"))?;
        let wq = qw(&format!("{bn}.attn_q.weight"), l.q_dim, cfg.d_model)?;
        let wk = qw(&format!("{bn}.attn_k.weight"), l.kv_dim, cfg.d_model)?;
        let wv = qw(&format!("{bn}.attn_v.weight"), l.kv_dim, cfg.d_model)?;
        let wo = qw(&format!("{bn}.attn_output.weight"), cfg.d_model, l.q_dim)?;
        let ffn_gate = qw(&format!("{bn}.ffn_gate.weight"), l.d_ff, cfg.d_model)?;
        let ffn_up = qw(&format!("{bn}.ffn_up.weight"), l.d_ff, cfg.d_model)?;
        let ffn_down = qw(&format!("{bn}.ffn_down.weight"), cfg.d_model, l.d_ff)?;
        let inp_gate = qw(&format!("{bn}.inp_gate.weight"), cfg.n_embd_per_layer, cfg.d_model)?;
        let proj = qw(&format!("{bn}.proj.weight"), cfg.d_model, cfg.n_embd_per_layer)?;
        let out_scale = read_tensor_f32(bytes, gguf, &format!("{bn}.layer_output_scale.weight"))
            .ok().and_then(|v| v.first().copied());
        blocks.push(Gemma4Block {
            attn_norm, q_norm, k_norm, attn_post_norm, ffn_norm, ffn_post_norm,
            per_layer_post_norm, wq, wk, wv, wo, ffn_gate, ffn_up, ffn_down,
            inp_gate, proj, out_scale,
        });
    }
    let rope_freqs_full = read_tensor_f32(bytes, gguf, "rope_freqs.weight").ok();
    Ok(Gemma4Weights {
        token_embd, per_layer_tok_embd, per_layer_model_proj, per_layer_proj_norm,
        output_norm, output, rope_freqs_full, blocks,
    })
}

pub struct BlockState {
    pub k_cache: Vec<f32>,  // flat [seq_len * kv_dim]
    pub v_cache: Vec<f32>,
}
pub struct Gemma4State { pub blocks: Vec<BlockState>, pub pos: usize }
impl Gemma4State {
    pub fn new(cfg: &Gemma4Config) -> Self {
        let blocks = cfg.layers.iter().map(|l| {
            let cap = cfg.context_length.min(8192) * l.kv_dim;
            BlockState { k_cache: Vec::with_capacity(cap), v_cache: Vec::with_capacity(cap) }
        }).collect();
        Self { blocks, pos: 0 }
    }
}

fn rms_norm_weighted(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(w.iter()).map(|(v, ww)| v * inv * ww).collect()
}

#[allow(dead_code)] // kept for trace builds; the in-use rms_norm path always carries a weight tensor
fn rms_norm_eps_only(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().map(|v| v * inv).collect()
}

/// Per-head RMS norm (Gemma's Q/K-norm convention): slice into
/// `n_heads × d_head`, normalise each head independently with shared
/// `w` (length `d_head`).
fn rms_norm_per_head(v: &[f32], w: &[f32], n_heads: usize, d_head: usize, eps: f32) -> Vec<f32> {
    debug_assert_eq!(v.len(), n_heads * d_head);
    let mut out = vec![0.0f32; v.len()];
    for h in 0..n_heads {
        let off = h * d_head;
        let mean_sq = v[off..off + d_head].iter().map(|x| x * x).sum::<f32>() / d_head as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for d in 0..d_head { out[off + d] = v[off + d] * inv * w[d]; }
    }
    out
}

/// Per-head RMS norm with NO learned weight (Gemma V-norm).
fn rms_norm_per_head_no_w(v: &[f32], n_heads: usize, d_head: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; v.len()];
    for h in 0..n_heads {
        let off = h * d_head;
        let mean_sq = v[off..off + d_head].iter().map(|x| x * x).sum::<f32>() / d_head as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for d in 0..d_head { out[off + d] = v[off + d] * inv; }
    }
    out
}

/// ggml-default GELU (tanh approx).
fn gelu(x: f32) -> f32 {
    let c = 0.797_884_56_f32;
    0.5 * x * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())
}

fn qmm(a: &[f32], w: &QuantWeight) -> Vec<f32> {
    let mut o = vec![0.0f32; w.n_rows];
    rce_op_qmm::mm_1d(a, w, &mut o);
    o
}

/// NeoX RoPE (i / i+half pair convention), per-head, in-place over the
/// first `rope_dim` of each head. `freq_factors` (if Some) scales each
/// per-dim `inv_freq /= freq_factors[i]` — proportional RoPE for the
/// Gemma 4 full-attention layers.
fn apply_rope_inplace(
    x: &mut [f32], n_heads: usize, d_head: usize, rope_dim: usize, base: f32, pos: usize,
    freq_factors: Option<&[f32]>,
) {
    for h in 0..n_heads {
        let off = h * d_head;
        rce_op_attn::rope_inplace_scaled(
            &mut x[off..off + d_head], rope_dim, base, pos, freq_factors,
        );
    }
}

/// One causal-attention step (Q is the current token only; K/V from cache).
/// `window` > 0 enables sliding-window: only positions in
/// `[pos − window + 1 ..= pos]` attend.
fn attend_one(
    q: &[f32], k_cache: &[f32], v_cache: &[f32],
    n_head: usize, n_kv: usize, d_head: usize,
    pos: usize, window: Option<usize>,
) -> Vec<f32> {
    let kv_dim = n_kv * d_head;
    let q_dim = n_head * d_head;
    let seq_len = (k_cache.len() / kv_dim).max(1);
    let scale = 1.0; // Gemma4 self.scaling = 1.0 (no /√d)
    let q_per_kv = n_head / n_kv;
    let lo = match window {
        Some(w) if w > 0 && pos + 1 > w => pos + 1 - w,
        _ => 0,
    };
    let hi = (pos + 1).min(seq_len);
    let mut out = vec![0.0f32; q_dim];
    for h in 0..n_head {
        let kv_h = h / q_per_kv;
        let qh = &q[h * d_head..(h + 1) * d_head];
        let head_out = rce_op_attn::fused_attend_head_window(
            qh, k_cache, v_cache, kv_h, lo, hi, kv_dim, d_head, scale,
        );
        out[h * d_head..(h + 1) * d_head].copy_from_slice(&head_out);
    }
    out
}

/// One full token forward (one position) → logits.
/// `RCE_TRACE=N` → dump activation sums for tensors matching layer ≤ N.
/// A/B comparison with llama-eval-callback (same named checkpoints).
fn trace() -> Option<usize> {
    std::env::var("RCE_TRACE").ok().and_then(|s| s.parse().ok())
}
fn tsum(name: &str, il: Option<usize>, x: &[f32], lim: Option<usize>) {
    if let Some(max_il) = lim {
        if il.map_or(false, |i| i > max_il) { return; }
    }
    let s: f32 = x.iter().sum();
    match il {
        Some(i) => eprintln!("[RCE] {name}-{i:<3} sum={s}"),
        None    => eprintln!("[RCE] {name:<26} sum={s}"),
    }
}

/// Token-id-derived input embedding `dequant(token_embd[id]) × √d_model`.
/// For text positions: feed this into `forward_embed` along with
/// `ple_token = id`. For image positions: use `forward_image_embed`
/// which sources the embedding from the vision tower and the PLE
/// from the padding token (id=0), per gemma4.cpp:410-422.
pub fn embed_token(w: &Gemma4Weights, cfg: &Gemma4Config, token_id: i32) -> Vec<f32> {
    let raw = w.token_embd.dequant_row(token_id as usize);
    let s = (cfg.d_model as f32).sqrt();
    raw.iter().map(|v| v * s).collect()
}

pub fn forward_token(
    w: &Gemma4Weights, cfg: &Gemma4Config, state: &mut Gemma4State, token_id: i32,
) -> Vec<f32> {
    let x = embed_token(w, cfg, token_id);
    forward_embed(w, cfg, state, x, token_id)
}

/// Per-position forward over a **precomputed embedding** `x` (already
/// scaled by √d_model if text, raw if image-projected). `ple_token`
/// selects the per-layer-embedding source: the GGUF row read by
/// `per_layer_tok_embd[ple_token]`. For text positions, pass the
/// text token id. For image-injected positions, pass `0` (padding)
/// per gemma4.cpp:410-422 (LLM_KV multimodal embedding path).
#[allow(non_snake_case)] // `inpL` mirrors the source-side identifier from gemma4.cpp
pub fn forward_embed(
    w: &Gemma4Weights, cfg: &Gemma4Config, state: &mut Gemma4State,
    x: Vec<f32>, ple_token: i32,
) -> Vec<f32> {
    let dm = cfg.d_model;
    let dpl = cfg.n_embd_per_layer;
    let nl = cfg.n_layers;
    let trc = trace();

    let mut inpL = x;
    debug_assert_eq!(inpL.len(), dm, "forward_embed: x.len {} != d_model {dm}", inpL.len());
    tsum("inp_scaled", None, &inpL, trc);

    // 2) Per-Layer Embeddings (PLE) — built from `ple_token` row.
    //
    // From source: per_layer_tok_embd row `token_id` is a flat
    // [n_embd_per_layer * n_layers] vector; reshape into rows of
    // length n_embd_per_layer, one per layer; scale by √n_embd_per_layer.
    // Then per_layer_model_proj × inpL_raw (pre-norm), scale 1/√d_model,
    // reshape, RMSnorm with per_layer_proj_norm, ADD to per-layer-tok,
    // scale by 1/√2. Result: per-layer input vector of length `dpl`.
    let per_layer: Vec<Vec<f32>>;
    {
        let row = w.per_layer_tok_embd.dequant_row(ple_token as usize); // [dpl*nl]
        let tok_scale = (dpl as f32).sqrt();
        // pre-scale and partition.
        let mut tok_pl: Vec<Vec<f32>> = (0..nl).map(|il| {
            let off = il * dpl;
            row[off..off + dpl].iter().map(|v| v * tok_scale).collect()
        }).collect();
        // per_layer_model_proj × raw inpL (the token_embd row BEFORE √d
        // scaling — the source uses `inpL` which has been scaled, but
        // then immediately divides by √d_model again).
        let proj = qmm(&inpL, &w.per_layer_model_proj);
        let proj_scale = 1.0 / (dm as f32).sqrt();
        let inv_root2 = 1.0 / (2.0f32).sqrt();
        for il in 0..nl {
            let off = il * dpl;
            let slice = &proj[off..off + dpl];
            let scaled: Vec<f32> = slice.iter().map(|v| v * proj_scale).collect();
            let normed = rms_norm_weighted(&scaled, &w.per_layer_proj_norm, cfg.eps);
            for k in 0..dpl {
                tok_pl[il][k] = (normed[k] + tok_pl[il][k]) * inv_root2;
            }
        }
        per_layer = tok_pl;
    }
    if let Some(_) = trc {
        let s: f32 = per_layer[0].iter().sum();
        eprintln!("[RCE] inp_per_layer (l0)   sum={s}");
    }

    let pos = state.pos;

    // 3) Layer loop.
    for il in 0..nl {
        let blk = &w.blocks[il];
        let lh = &cfg.layers[il];

        // Pre-attn RMS-norm.
        let cur = rms_norm_weighted(&inpL, &blk.attn_norm, cfg.eps);
        tsum("attn_norm", Some(il), &cur, trc);

        // Q
        let mut q = qmm(&cur, &blk.wq);
        tsum("Qcur", Some(il), &q, trc);
        q = rms_norm_per_head(&q, &blk.q_norm, lh.n_head, lh.head_dim, cfg.eps);
        tsum("Qcur_normed", Some(il), &q, trc);
        let ff: Option<&[f32]> = if !lh.is_swa {
            w.rope_freqs_full.as_deref()
        } else { None };
        apply_rope_inplace(&mut q, lh.n_head, lh.head_dim, lh.rope_dim, lh.rope_base, pos, ff);
        tsum("Qcur_pos", Some(il), &q, trc);

        // Shared-KV (Gemma 4): layers `il >= n_layer_kv_from_start` reuse
        // the K/V cache of an earlier layer per llama_model.cpp:2009-2015:
        //   reuse(il) = n_layer_kv_from_start - (is_swa(il) ? 2 : 1)
        // ⇒ SWA layers reuse the LAST own-KV SWA layer; full layers reuse
        // the LAST own-KV full layer. K/V projection is SKIPPED for these.
        let kv_src = if il < cfg.n_layer_kv_from_start {
            il
        } else {
            cfg.n_layer_kv_from_start - if lh.is_swa { 2 } else { 1 }
        };

        if kv_src == il {
            // Own KV: compute K/V, append to own cache.
            let mut k = qmm(&cur, &blk.wk);
            tsum("Kcur", Some(il), &k, trc);
            k = rms_norm_per_head(&k, &blk.k_norm, lh.n_kv, lh.head_dim, cfg.eps);
            tsum("Kcur_normed", Some(il), &k, trc);
            apply_rope_inplace(&mut k, lh.n_kv, lh.head_dim, lh.rope_dim, lh.rope_base, pos, ff);
            tsum("Kcur_pos", Some(il), &k, trc);

            let v_raw = qmm(&cur, &blk.wv);
            tsum("Vcur", Some(il), &v_raw, trc);
            let v = rms_norm_per_head_no_w(&v_raw, lh.n_kv, lh.head_dim, cfg.eps);
            tsum("Vcur_normed", Some(il), &v, trc);

            let bs = &mut state.blocks[il];
            bs.k_cache.extend_from_slice(&k);
            bs.v_cache.extend_from_slice(&v);
        }

        // Attention (causal; SWA window for SWA layers). For shared-KV
        // layers (`kv_src != il`), read K/V from the source layer's cache.
        let window = if lh.is_swa { Some(cfg.sliding_window) } else { None };
        let bs_src = &state.blocks[kv_src];
        let attn = attend_one(
            &q, &bs_src.k_cache, &bs_src.v_cache,
            lh.n_head, lh.n_kv, lh.head_dim, pos, window,
        );

        // attn_post_norm  +  residual (post-attn).
        tsum("attn_raw", Some(il), &attn, trc);
        let kqv_out = qmm(&attn, &blk.wo);
        tsum("kqv_out", Some(il), &kqv_out, trc);
        if il == 0 && trc.is_some() {
            eprintln!(
                "[RCE] wo dims: n_rows={} n_cols={} dtype={:?}",
                blk.wo.n_rows, blk.wo.n_cols, blk.wo.dtype
            );
        }
        let post = rms_norm_weighted(&kqv_out, &blk.attn_post_norm, cfg.eps);
        tsum("attn_post_norm", Some(il), &post, trc);
        let mut attn_out = vec![0.0f32; dm];
        for i in 0..dm { attn_out[i] = inpL[i] + post[i]; }
        tsum("attn_out", Some(il), &attn_out, trc);

        // FFN: rms_norm → GELU-PAR(gate, up) → down → post-norm → +residual.
        let fn1 = rms_norm_weighted(&attn_out, &blk.ffn_norm, cfg.eps);
        tsum("ffn_norm", Some(il), &fn1, trc);
        let g = qmm(&fn1, &blk.ffn_gate);
        tsum("ffn_gate", Some(il), &g, trc);
        let u = qmm(&fn1, &blk.ffn_up);
        tsum("ffn_up", Some(il), &u, trc);
        let mut h = vec![0.0f32; lh.d_ff];
        for j in 0..lh.d_ff { h[j] = gelu(g[j]) * u[j]; }
        tsum("ffn_geglu", Some(il), &h, trc);
        let fout = qmm(&h, &blk.ffn_down);
        tsum("ffn_out", Some(il), &fout, trc);
        let fout = rms_norm_weighted(&fout, &blk.ffn_post_norm, cfg.eps);
        tsum("ffn_post_norm", Some(il), &fout, trc);
        let mut layer_out = vec![0.0f32; dm];
        for i in 0..dm { layer_out[i] = attn_out[i] + fout[i]; }
        tsum("pe_in", Some(il), &layer_out, trc);

        // PLE step: pe_in = layer_out
        // cur = inp_gate(layer_out) ∈ ℝ^dpl ; cur = GELU(cur) ;
        // cur *= per_layer[il] ; cur = proj(cur) ∈ ℝ^d_model ;
        // cur = rms_norm(cur, per_layer_post_norm) ; cur = pe_in + cur
        {
            let gated = qmm(&layer_out, &blk.inp_gate); // [dpl]
            tsum("pe_gate", Some(il), &gated, trc);
            let gated: Vec<f32> = gated.iter().enumerate()
                .map(|(j, v)| gelu(*v) * per_layer[il][j]).collect();
            tsum("pe_gelu_mul", Some(il), &gated, trc);
            let projed = qmm(&gated, &blk.proj); // [d_model]
            tsum("pe_proj", Some(il), &projed, trc);
            let normed = rms_norm_weighted(&projed, &blk.per_layer_post_norm, cfg.eps);
            tsum("per_layer_embd_out", Some(il), &normed, trc);
            for i in 0..dm { layer_out[i] += normed[i]; }
            tsum("l_out", Some(il), &layer_out, trc);
        }

        // out_scale (optional per-layer scalar).
        if let Some(s) = blk.out_scale {
            for i in 0..dm { layer_out[i] *= s; }
            if il == 0 && trc.is_some() {
                eprintln!("[RCE] out_scale-0    = {s}");
            }
        } else if il == 0 && trc.is_some() {
            eprintln!("[RCE] out_scale-0    = MISSING (None)");
        }
        tsum("l_out_final", Some(il), &layer_out, trc);

        inpL = layer_out;
    }

    // Final RMS-norm + tied lm_head.
    let final_normed = rms_norm_weighted(&inpL, &w.output_norm, cfg.eps);
    tsum("result_norm", None, &final_normed, trc);
    let mut logits = qmm(&final_normed, &w.output);
    tsum("result_pre_softcap", None, &logits, trc);

    // Final logit softcapping: softcap·tanh(logits / softcap).
    if cfg.final_logit_softcap > 0.0 {
        let sc = cfg.final_logit_softcap;
        for v in logits.iter_mut() { *v = sc * (*v / sc).tanh(); }
    }
    tsum("result_output", None, &logits, trc);
    state.pos += 1;
    logits
}

pub struct Gemma4Runtime {
    pub cfg: Gemma4Config,
    pub gguf: GgufFile,
    pub weights: Gemma4Weights,
    _mmap: Arc<Mmap>,
}

impl Gemma4Runtime {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| format!("open: {e}"))?;
        let mmap = Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?);
        // Hint sequential reads during the load-time weight scan; ignore
        // errors (kernel may refuse). See rce_ingest_gguf::mmap_load_sequential.
        let _ = mmap.advise(memmap2::Advice::WillNeed);
        let gguf = GgufFile::parse(&mmap[..]).map_err(|e| format!("parse: {e:?}"))?;
        let cfg = Gemma4Config::from_gguf(&gguf)?;
        eprintln!(
            "gemma4: d_model={} vocab={} n_layers={} n_swa={} d_ff={{first={}, last={}}} ctx={} \
             PLE={} softcap={}",
            cfg.d_model, cfg.vocab, cfg.n_layers,
            cfg.layers.iter().filter(|l| l.is_swa).count(),
            cfg.layers.first().map(|l| l.d_ff).unwrap_or(0),
            cfg.layers.last().map(|l| l.d_ff).unwrap_or(0),
            cfg.context_length, cfg.n_embd_per_layer, cfg.final_logit_softcap,
        );
        let weights = load_weights(&mmap, &gguf, &cfg)?;
        Ok(Self { cfg, gguf, weights, _mmap: mmap })
    }
    pub fn new_state(&self) -> Gemma4State { Gemma4State::new(&self.cfg) }
    /// Token-id-derived embedding (scaled × √d_model). For text-position
    /// splicing into a mixed text+image embedding stream.
    pub fn embed_token(&self, token_id: i32) -> Vec<f32> {
        embed_token(&self.weights, &self.cfg, token_id)
    }
    /// Forward at one position with a precomputed embedding `x` and an
    /// explicit `ple_token` for the PLE branch. For text positions pass
    /// `ple_token = token_id`; for image-injected positions use
    /// `forward_image_embed` (PLE from padding token id=0).
    pub fn forward_embed(&self, state: &mut Gemma4State, x: Vec<f32>, ple_token: i32) -> Vec<f32> {
        forward_embed(&self.weights, &self.cfg, state, x, ple_token)
    }
    /// Image-token-injection forward: use the vision-tower projected `x`
    /// directly (NO `× √d_model`) and source PLE from the padding token
    /// (id=0), per gemma4.cpp:410-422 (`inp_per_layer_multimodal`).
    pub fn forward_image_embed(&self, state: &mut Gemma4State, x: Vec<f32>) -> Vec<f32> {
        forward_embed(&self.weights, &self.cfg, state, x, 0)
    }
    pub fn forward_logits(&self, state: &mut Gemma4State, token_id: i32) -> Vec<f32> {
        forward_token(&self.weights, &self.cfg, state, token_id)
    }
    pub fn tokenizer(&self) -> Tokenizer<'_> { Tokenizer::from_gguf(&self.gguf).expect("tokenizer") }
}

#[derive(Clone, Copy, Debug)]
pub struct Gemma4GenConfig {
    pub max_new_tokens: usize,
    pub sample: SampleConfig,
    pub rep_window: usize,
}
impl Default for Gemma4GenConfig {
    fn default() -> Self {
        Self { max_new_tokens: 32, sample: SampleConfig::GREEDY, rep_window: 64 }
    }
}

impl Gemma4Runtime {
    pub fn generate(
        &self, prompt: &str, gcfg: &Gemma4GenConfig, mut on_token: impl FnMut(&str),
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
        let prompt_ids = if is_gpt2 { tk.encode_gpt2(prompt, add_bos) }
                          else { tk.encode(prompt, add_bos) };
        if prompt_ids.is_empty() { return Err("empty prompt".into()); }
        let eos_id = tk.eos_token_id.unwrap_or(-1);
        // Gemma 4 instruction-tuned models end a reply with a turn-end token
        // (`<turn|>` = id 106 here), distinct from `<eos>`. Stop on it too, or
        // an -it model runs past its answer and degenerates into turn-marker
        // repetition. Look it up by string so the id isn't hard-coded.
        let turn_end_id = ["<turn|>", "<end_of_turn>", "<|im_end|>"]
            .iter()
            .find_map(|m| tk.lookup_token(m))
            .unwrap_or(-1);
        let is_stop = |t: i32| t == eos_id || (turn_end_id >= 0 && t == turn_end_id);
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
            if is_stop(next) { break; }
            out_ids.push(next);
            history.push(next);
            on_token(&tk.decode_sequence(&[next], false));
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
    const MODEL: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-text-q8_0.gguf";
    // Larger sibling — same gemma4 arch, dims read from GGUF (42 blocks, d_model 2560).
    const MODEL_E4B: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/gemma-4-e4b-gguf/gemma-4-E4B-it-Q8_0.gguf";

    /// **G4.e — E4B loads + chat-framed generation is coherent.** Proves the
    /// same runtime serves the larger E4B checkpoint (dims from GGUF, no code
    /// changes). Same chat framing as G4.d.
    #[test]
    fn g4_e_e4b_chat() {
        if !Path::new(MODEL_E4B).exists() { eprintln!("G4.e: E4B not mounted — skip"); return; }
        let rt = Gemma4Runtime::load(MODEL_E4B).expect("load E4B");
        let cfg = Gemma4GenConfig { max_new_tokens: 24, sample: SampleConfig::GREEDY, rep_window: 64 };
        // Gemma turn framing (the model's turn-end token is `<turn|>`, id 106).
        let framed = "<start_of_turn>user\nWhat is the capital of France? Answer briefly.<end_of_turn>\n<start_of_turn>model\n";
        let mut o = String::new();
        let t0 = std::time::Instant::now();
        let n = std::cell::Cell::new(0u32);
        let _ = rt.generate(framed, &cfg, |s| { o.push_str(s); n.set(n.get() + 1); }).expect("generate");
        eprintln!("G4.e E4B CHAT → {:?}  [{} tok, {:.2} tok/s]", o, n.get(), n.get() as f64 / t0.elapsed().as_secs_f64());
        let coherent = ["Paris", " the ", " is ", " of ", " France "].iter().any(|w| o.contains(w))
            || o.chars().filter(|c| c.is_alphabetic()).count() >= o.len() / 2;
        assert!(coherent, "E4B chat lacks coherence: {o:?}");
    }

    /// **G4.a — load + shape sanity.**
    #[test]
    fn g4_a_load() {
        if !Path::new(MODEL).exists() { eprintln!("G4.a: skip"); return; }
        let rt = Gemma4Runtime::load(MODEL).expect("load");
        let c = &rt.cfg;
        assert_eq!(c.d_model, 1536);
        assert_eq!(c.n_layers, 35);
        assert_eq!(c.n_embd_per_layer, 256);
        assert_eq!(c.final_logit_softcap, 30.0);
        assert_eq!(c.sliding_window, 512);
        let swa = c.layers.iter().filter(|l| l.is_swa).count();
        let full = c.n_layers - swa;
        assert_eq!(full, 7, "expected 7 full-attn layers in 35-layer E2B (every 5th)");
        assert_eq!(swa, 28);
        eprintln!("G4.a OK: gemma4-E2B loaded — 35L ({swa} SWA / {full} full), PLE=256");
    }

    /// **G4.b — one forward produces finite logits.**
    #[test]
    fn g4_b_one_forward() {
        if !Path::new(MODEL).exists() { eprintln!("G4.b: skip"); return; }
        let rt = Gemma4Runtime::load(MODEL).expect("load");
        let mut state = rt.new_state();
        let logits = rt.forward_logits(&mut state, 1);
        assert_eq!(logits.len(), rt.cfg.vocab);
        for v in &logits { assert!(v.is_finite(), "non-finite logit"); }
        let max_abs = logits.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        // Softcap=30 means |logits| ≤ 30 strictly.
        assert!(max_abs <= 30.5, "softcapping not applied? max|logit|={max_abs}");
        let (argmax, _) = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        assert!(argmax > 0 && argmax < rt.cfg.vocab - 1);
        eprintln!("G4.b OK: one forward → argmax={argmax}, max|logit|={max_abs:.3}");
    }

    /// **G4.d — chat-framed generation (instruction-tuned model).**
    /// Gemma 4 is heavily instruction-tuned, so bare-prompt greedy can
    /// veer off; the model expects `<|im_start|>user\n…<|im_end|>\n
    /// <|im_start|>assistant\n` framing per its chat_template.
    #[test]
    fn g4_d_chat() {
        if !Path::new(MODEL).exists() { eprintln!("G4.d: skip"); return; }
        let rt = Gemma4Runtime::load(MODEL).expect("load");
        let cfg = Gemma4GenConfig { max_new_tokens: 24, sample: SampleConfig::GREEDY, rep_window: 64 };
        let framed = "<|im_start|>user\nWhat is the capital of France? Answer briefly.<|im_end|>\n<|im_start|>assistant\n";
        let mut o = String::new();
        let _ = rt.generate(framed, &cfg, |s| o.push_str(s)).expect("generate");
        eprintln!("G4.d CHAT → {:?}", o);
        let coherent = [" the ", " a ", " is ", " of ", " France ", "Paris"]
            .iter().any(|w| o.contains(w))
            || o.chars().filter(|c| c.is_alphabetic()).count() >= o.len() / 2;
        assert!(coherent, "chat-framed lacks coherence: {o:?}");
    }

    /// **G4.c — coherent generation.** Greedy "The capital of France is".
    #[test]
    fn g4_c_generate() {
        if !Path::new(MODEL).exists() { eprintln!("G4.c: skip"); return; }
        let rt = Gemma4Runtime::load(MODEL).expect("load");
        let cfg = Gemma4GenConfig { max_new_tokens: 16, sample: SampleConfig::GREEDY, rep_window: 64 };
        let mut chunks: Vec<String> = Vec::new();
        let t0 = std::time::Instant::now();
        let _ = rt.generate("The capital of France is", &cfg, |s| chunks.push(s.to_string()))
            .expect("generate");
        let dt = t0.elapsed();
        let full = chunks.concat();
        assert!(!full.is_empty(), "empty continuation");
        let coherent = [" the ", " a ", " of ", " is ", " in ", " and ", " to ", "Paris"]
            .iter().any(|w| full.contains(w))
            || full.chars().filter(|c| c.is_alphabetic()).count() >= full.len() / 2;
        assert!(coherent, "no common English: {full:?}");
        eprintln!(
            "G4.c OK: prompt=\"The capital of France is\" → {full:?}  [{} tok in {dt:?} = {:.2} tok/s]",
            chunks.len(), chunks.len() as f64 / dt.as_secs_f64()
        );
    }
}
