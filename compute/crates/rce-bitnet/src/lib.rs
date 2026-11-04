//! Microsoft BitNet b1.58 inference runtime on the rce substrate.
//!
//! Targets the `bitnet-b1.58` GGUF architecture (the official i2_s
//! 2-bit-ternary format proven readable + decodable by
//! `rce_ingest_gguf::dequant_i2_s`, with body matmuls routed through
//! `rce_op_qmm::mm_1d → mm_1d_i2_s_neon` — the multiply-free kernel
//! the SoA/4-bit layout was designed to enable, T2.b).
//!
//! ## Block forward (authoritative, per vendored llama.cpp
//! `llm_build_bitnet`)
//!
//! ```text
//!   inpSA = x
//!   cur   = rms_norm(x, attn_norm)
//!   {
//!       Q,K,V = wq(cur), wk(cur), wv(cur)      // 3× I2_S BitLinear
//!       Q,K   = rope(Q), rope(K)               // partial NeoX rope
//!       cur   = attention(Q, K, V)             // GQA, softmax(QK^T/√d)V
//!       cur   = rms_norm(cur, attn_sub_norm)   // *** BitNet-specific ***
//!       cur   = wo(cur)                        // I2_S BitLinear
//!   }
//!   ffn_inp = cur + inpSA
//!   cur     = rms_norm(ffn_inp, ffn_norm)
//!   cur     = silu(gate(cur)) * up(cur)        // 2× I2_S BitLinear + SiLU
//!   cur     = rms_norm(cur, ffn_sub_norm)      // *** BitNet-specific ***
//!   cur     = down(cur)                        // I2_S BitLinear
//!   out     = cur + ffn_inp
//! ```
//!
//! `lm_head` is **tied to `token_embd`** (no separate `output.weight`
//! tensor in the GGUF; logits = `token_embd · final_norm(x)^T`, with
//! `token_embd` an F16 `QuantWeight`).
//!
//! T3.a here: config + weights load + shape-sanity test on the real
//! `bitnet-b1.58-2b-4t-i2s.gguf`. Per-block forward, generate, and the
//! f32-reference accuracy gate are T3.b/T3.c/T3.d.

use rce_ingest_gguf::{
    read_tensor_f32, sample_with_history, GgufFile, GgufValue, SampleConfig, Tokenizer,
};
use rce_op_qmm::QuantWeight;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

// ── Config ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct BitNetConfig {
    pub d_model: usize,
    pub vocab: usize,
    pub n_layers: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub d_head: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub d_ff: usize,
    pub context_length: usize,
    pub eps: f32,
}

impl BitNetConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf
            .metadata
            .iter()
            .find(|kv| kv.key == "general.architecture")
            .and_then(|kv| match &kv.value {
                GgufValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .ok_or("missing general.architecture")?;
        if arch != "bitnet-b1.58" {
            return Err(format!("expected bitnet-b1.58, got {arch}"));
        }

        let u32_meta = |k: &str| -> Option<u32> {
            gguf.metadata
                .iter()
                .find(|kv| kv.key == format!("bitnet-b1.58.{k}"))
                .and_then(|kv| match &kv.value {
                    GgufValue::U32(v) => Some(*v),
                    _ => None,
                })
        };
        let f32_meta = |k: &str| -> Option<f32> {
            gguf.metadata
                .iter()
                .find(|kv| kv.key == format!("bitnet-b1.58.{k}"))
                .and_then(|kv| match &kv.value {
                    GgufValue::F32(v) => Some(*v),
                    _ => None,
                })
        };

        let d_model = u32_meta("embedding_length").ok_or("embedding_length")? as usize;
        let vocab = u32_meta("vocab_size").ok_or("vocab_size")? as usize;
        let n_layers = u32_meta("block_count").ok_or("block_count")? as usize;
        let n_q_heads = u32_meta("attention.head_count").ok_or("attention.head_count")? as usize;
        let n_kv_heads = u32_meta("attention.head_count_kv").unwrap_or(n_q_heads as u32) as usize;
        let d_head = if n_q_heads > 0 { d_model / n_q_heads } else { 0 };
        let rope_dim = u32_meta("rope.dimension_count").unwrap_or(d_head as u32) as usize;
        let rope_freq_base = f32_meta("rope.freq_base").unwrap_or(500_000.0);
        let d_ff = u32_meta("feed_forward_length").ok_or("feed_forward_length")? as usize;
        let context_length = u32_meta("context_length").unwrap_or(4096) as usize;
        let eps = f32_meta("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);

        Ok(BitNetConfig {
            d_model,
            vocab,
            n_layers,
            n_q_heads,
            n_kv_heads,
            d_head,
            rope_dim,
            rope_freq_base,
            d_ff,
            context_length,
            eps,
        })
    }

    #[inline]
    pub fn n_embd_q(&self) -> usize {
        self.n_q_heads * self.d_head
    }
    #[inline]
    pub fn n_embd_kv(&self) -> usize {
        self.n_kv_heads * self.d_head
    }
}

// ── Weights ────────────────────────────────────────────────────────

pub struct BitNetBlock {
    pub attn_norm: Vec<f32>,
    pub attn_sub_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub ffn_sub_norm: Vec<f32>,
    /// `[n_q_heads·d_head, d_model]` I2_S
    pub wq: QuantWeight,
    /// `[n_kv_heads·d_head, d_model]` I2_S
    pub wk: QuantWeight,
    /// `[n_kv_heads·d_head, d_model]` I2_S
    pub wv: QuantWeight,
    /// `[d_model, n_q_heads·d_head]` I2_S
    pub wo: QuantWeight,
    /// `[d_ff, d_model]` I2_S
    pub ffn_gate: QuantWeight,
    /// `[d_ff, d_model]` I2_S
    pub ffn_up: QuantWeight,
    /// `[d_model, d_ff]` I2_S
    pub ffn_down: QuantWeight,
}

pub struct BitNetWeights {
    /// `[vocab, d_model]` F16 — and tied as the lm_head per the
    /// vendored `llm_build_bitnet`.
    pub token_embd: QuantWeight,
    pub output_norm: Vec<f32>,
    pub blocks: Vec<BitNetBlock>,
}

pub fn load_weights(
    mmap: &Arc<Mmap>,
    gguf: &GgufFile,
    cfg: &BitNetConfig,
) -> Result<BitNetWeights, String> {
    let bytes: &[u8] = &mmap[..];
    let token_embd = QuantWeight::from_gguf(
        Arc::clone(mmap),
        gguf,
        "token_embd.weight",
        cfg.vocab,
        cfg.d_model,
    )
    .map_err(|e| format!("token_embd: {e}"))?;
    let output_norm = read_tensor_f32(bytes, gguf, "output_norm.weight")
        .map_err(|e| format!("output_norm: {e:?}"))?;

    let n_embd_q = cfg.n_embd_q();
    let n_embd_kv = cfg.n_embd_kv();
    let mut blocks = Vec::with_capacity(cfg.n_layers);
    for b in 0..cfg.n_layers {
        let bn = format!("blk.{b}");
        let f32n = |s: &str| {
            read_tensor_f32(bytes, gguf, &format!("{bn}.{s}.weight"))
                .map_err(|e| format!("blk.{b}.{s}: {e:?}"))
        };
        let i2sn = |s: &str, n_rows: usize, n_cols: usize| {
            QuantWeight::from_gguf(
                Arc::clone(mmap),
                gguf,
                &format!("{bn}.{s}.weight"),
                n_rows,
                n_cols,
            )
            .map_err(|e| format!("blk.{b}.{s}: {e}"))
        };
        let attn_norm = f32n("attn_norm")?;
        let attn_sub_norm = f32n("attn_sub_norm")?;
        let ffn_norm = f32n("ffn_norm")?;
        let ffn_sub_norm = f32n("ffn_sub_norm")?;
        let wq = i2sn("attn_q", n_embd_q, cfg.d_model)?;
        let wk = i2sn("attn_k", n_embd_kv, cfg.d_model)?;
        let wv = i2sn("attn_v", n_embd_kv, cfg.d_model)?;
        let wo = i2sn("attn_output", cfg.d_model, n_embd_q)?;
        let ffn_gate = i2sn("ffn_gate", cfg.d_ff, cfg.d_model)?;
        let ffn_up = i2sn("ffn_up", cfg.d_ff, cfg.d_model)?;
        let ffn_down = i2sn("ffn_down", cfg.d_model, cfg.d_ff)?;
        blocks.push(BitNetBlock {
            attn_norm,
            attn_sub_norm,
            ffn_norm,
            ffn_sub_norm,
            wq,
            wk,
            wv,
            wo,
            ffn_gate,
            ffn_up,
            ffn_down,
        });
    }

    Ok(BitNetWeights {
        token_embd,
        output_norm,
        blocks,
    })
}

// ── State (per-block KV cache) ─────────────────────────────────────

pub struct BlockState {
    /// `[seq * n_kv_heads * d_head]` flat row-major.
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
}

pub struct BitNetState {
    pub blocks: Vec<BlockState>,
    pub pos: usize,
}

impl BitNetState {
    pub fn new(cfg: &BitNetConfig) -> Self {
        let kv_dim = cfg.n_embd_kv();
        // Reserve a sensible upper bound to avoid reallocation churn.
        let cap = cfg.context_length * kv_dim;
        let blocks = (0..cfg.n_layers)
            .map(|_| BlockState {
                k_cache: Vec::with_capacity(cap),
                v_cache: Vec::with_capacity(cap),
            })
            .collect();
        Self { blocks, pos: 0 }
    }
}

// ── Math helpers (same conventions as rce-bamba; deliberately
//     mirroring the proven RoPE + RMS + SiLU code) ──────────────────

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(w.iter()).map(|(v, ww)| v * inv * ww).collect()
}

fn silu(x: f32) -> f32 {
    if x >= 0.0 {
        x / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        x * e / (1.0 + e)
    }
}

/// Matmul with a BitNet-aware wrapper for **I2_S body weights**: the
/// BitLinear inference rule (per the BitNet b1.58 paper / `bitnet.cpp`)
/// **quantizes activations to int8-absmax per token** *before* the
/// ternary matmul, then post-multiplies by `γ`:
///
/// ```text
///   γ      = max|a| / 127
///   x_q    = round(a / γ)               // int8 values, kept as f32
///   out    = γ · s_w · Σ x_q[i] · w_ternary[i]
/// ```
///
/// `s_w` (the per-tensor absmean stored in the i2_s trailer) is applied
/// inside [`mm_1d`] → `mm_1d_i2_s`. The model was *trained* with this
/// rounded `x_q`; feeding raw f32 activations compounds a magnitude
/// error per BitLinear × 7/layer × 30 layers → garbage (the T3.c
/// finding). F16/F32 weights (e.g. tied lm_head, token_embd) take the
/// plain path — no act-quant.
fn qmm_1d(a: &[f32], w: &QuantWeight) -> Vec<f32> {
    if w.dtype == rce_ingest_gguf::GgufDtype::I2S {
        // I2_S act-quant path: BitNet's accuracy requires the rounded
        // `x_q`; the placement backends do not replicate act-quant, so the I2_S path
        // stays on rce-op-qmm.
        let absmax = a.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if absmax == 0.0 {
            return vec![0.0f32; w.n_rows];
        }
        let gamma = absmax / 127.0;
        let x_q: Vec<f32> = a.iter().map(|&v| (v / gamma).round()).collect();
        let mut out = vec![0.0f32; w.n_rows];
        rce_op_qmm::mm_1d(&x_q, w, &mut out);
        for v in &mut out {
            *v *= gamma;
        }
        return out;
    }
    // F16/F32 path (token_embd, lm_head): can route through placement.
    // Precedence: placement → direct. I2_S stays on the
    // act-quant path above (placement doesn't replicate BitNet's
    // act-quant; routing them would silently drift accuracy).
    #[cfg(feature = "placement")]
    if let Some(out) = placement_thread_local::try_qmm(a, w) {
        return out;
    }
    let mut out = vec![0.0f32; w.n_rows];
    rce_op_qmm::mm_1d(a, w, &mut out);
    out
}

/// Placement dispatch hook: cost-driven qmm routing.
/// F16/F32 paths only; I2_S weights stay on rce-op-qmm's act-quant path.
/// Activation: `RCE_BITNET_PLACEMENT=1` (CPU-only) or `=hybrid` (CPU + native Metal).
#[cfg(feature = "placement")]
pub mod placement_thread_local {
    use super::QuantWeight;
    pub use rce_placement_adapter::{PlacementAdapterStats, PlacementQmmAdapter};
    use std::cell::RefCell;

    thread_local! {
        static ADAPTER: RefCell<Option<PlacementQmmAdapter>> = const { RefCell::new(None) };
    }

    pub fn install(a: PlacementQmmAdapter) {
        ADAPTER.with(|s| *s.borrow_mut() = Some(a));
    }
    pub fn uninstall() -> Option<PlacementQmmAdapter> {
        ADAPTER.with(|s| s.borrow_mut().take())
    }
    pub fn install_from_env() -> bool {
        match std::env::var("RCE_BITNET_PLACEMENT").as_deref() {
            Ok("hybrid") => { install(PlacementQmmAdapter::with_metal()); true }
            Ok(v) if !v.is_empty() && v != "0" => { install(PlacementQmmAdapter::new()); true }
            _ => false,
        }
    }
    pub fn stats() -> Option<PlacementAdapterStats> {
        ADAPTER.with(|s| s.borrow().as_ref().map(|a| a.stats.clone()))
    }
    pub(crate) fn try_qmm(a: &[f32], w: &QuantWeight) -> Option<Vec<f32>> {
        ADAPTER.with(|s| s.borrow_mut().as_mut().map(|adp| adp.qmm_1d(a, w)))
    }
}

/// Partial NeoX RoPE — rotate the first `rope_dim` dims of each head
/// using the i / i+half pair convention. Authoritative for BitNet per
/// vendored `llama-model.cpp` (BITNET in the NeoX rope_type switch).
fn apply_rope_inplace(
    x: &mut [f32], n_heads: usize, d_head: usize, rope_dim: usize, base: f32, pos: usize,
) {
    // NEON-vectorized RoPE via rce-op-attn; numerics match scalar to 1e-5.
    for h in 0..n_heads {
        let head_off = h * d_head;
        rce_op_attn::rope_inplace(&mut x[head_off..head_off + d_head], rope_dim, base, pos);
    }
}

// ── BitNet attention block ─────────────────────────────────────────
//
// Per `llm_build_bitnet`: Q,K,V projections → reshape per head →
// RoPE(Q), RoPE(K) → GQA scaled dot-product → **rms_norm(attn_out,
// attn_sub_norm)** ← BitNet-specific → output projection.

fn attn_block(
    x_in: &[f32], blk: &BitNetBlock, cfg: &BitNetConfig,
    k_cache: &mut Vec<f32>, v_cache: &mut Vec<f32>, pos: usize,
) -> Vec<f32> {
    let dh = cfg.d_head;
    let nq = cfg.n_q_heads;
    let nkv = cfg.n_kv_heads;
    let q_dim = nq * dh;
    let kv_dim = nkv * dh;

    let mut q = qmm_1d(x_in, &blk.wq);
    let mut k_new = qmm_1d(x_in, &blk.wk);
    let v_new = qmm_1d(x_in, &blk.wv);
    apply_rope_inplace(&mut q, nq, dh, cfg.rope_dim, cfg.rope_freq_base, pos);
    apply_rope_inplace(&mut k_new, nkv, dh, cfg.rope_dim, cfg.rope_freq_base, pos);

    k_cache.extend_from_slice(&k_new);
    v_cache.extend_from_slice(&v_new);
    let seq_len = k_cache.len() / kv_dim;
    let scale = 1.0 / (dh as f32).sqrt();
    let q_per_kv = nq / nkv;

    // NEON-vectorized fused attention via rce-op-attn; one pass per
    // (q-head, seq) keeping (max, sum, out) online (FlashAttention-1
    // online softmax). Matches scalar reference to 1e-4.
    let mut attended = vec![0.0f32; q_dim];
    for h in 0..nq {
        let kv_head = h / q_per_kv;
        let q_h = &q[h * dh..(h + 1) * dh];
        let out_h = rce_op_attn::fused_attend_head(
            q_h, k_cache, v_cache, kv_head, seq_len, kv_dim, dh, scale,
        );
        let out_off = h * dh;
        attended[out_off..out_off + dh].copy_from_slice(&out_h);
    }

    // *** BitNet-specific *** — sub-RMS-norm between V-mixing and `wo`.
    let attended_sub = rms_norm(&attended, &blk.attn_sub_norm, cfg.eps);
    qmm_1d(&attended_sub, &blk.wo)
}

// ── BitNet gated FFN ───────────────────────────────────────────────
//
// Per `llm_build_bitnet`: x = silu(gate(in)) * up(in); then
// **rms_norm(x, ffn_sub_norm)** ← BitNet-specific; then down(x).

fn ffn_block(x_in: &[f32], blk: &BitNetBlock, cfg: &BitNetConfig) -> Vec<f32> {
    let g = qmm_1d(x_in, &blk.ffn_gate);
    let u = qmm_1d(x_in, &blk.ffn_up);
    let mut h = vec![0.0f32; cfg.d_ff];
    for i in 0..cfg.d_ff {
        h[i] = silu(g[i]) * u[i];
    }
    // *** BitNet-specific *** — sub-RMS-norm between gate·up and `down`.
    let h_sub = rms_norm(&h, &blk.ffn_sub_norm, cfg.eps);
    qmm_1d(&h_sub, &blk.ffn_down)
}

/// One full forward pass for `token_id` at sequence position
/// `state.pos`. Threads the per-block KV cache, increments `state.pos`,
/// returns logits over the vocab. lm_head is **tied** to `token_embd`
/// (per vendored `llm_build_bitnet`'s FIXME: the embedding doubles as
/// the output projection); we compute logits via `mm_1d(x, &token_embd)`
/// where `token_embd` is `[vocab, d_model]` row-major.
pub fn forward_token(
    weights: &BitNetWeights, cfg: &BitNetConfig, state: &mut BitNetState, token_id: i32,
) -> Vec<f32> {
    let dm = cfg.d_model;
    let mut x: Vec<f32> = weights.token_embd.dequant_row(token_id as usize);
    let pos = state.pos;
    for (b, blk) in weights.blocks.iter().enumerate() {
        let normed = rms_norm(&x, &blk.attn_norm, cfg.eps);
        let bs = &mut state.blocks[b];
        let attn_out = attn_block(
            &normed, blk, cfg, &mut bs.k_cache, &mut bs.v_cache, pos,
        );
        for i in 0..dm {
            x[i] += attn_out[i];
        }
        let normed2 = rms_norm(&x, &blk.ffn_norm, cfg.eps);
        let ffn_out = ffn_block(&normed2, blk, cfg);
        for i in 0..dm {
            x[i] += ffn_out[i];
        }
    }
    let final_normed = rms_norm(&x, &weights.output_norm, cfg.eps);
    let logits = qmm_1d(&final_normed, &weights.token_embd); // tied lm_head
    state.pos += 1;
    logits
}

// ── Runtime entry (T3.b/c/d will build on this) ────────────────────

pub struct BitNetRuntime {
    pub cfg: BitNetConfig,
    pub gguf: GgufFile,
    pub weights: BitNetWeights,
    _mmap: Arc<Mmap>,
}

impl BitNetRuntime {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| format!("open: {e}"))?;
        let mmap = Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?);
        // Hint sequential reads during the load-time weight scan; ignore
        // errors (kernel may refuse). See rce_ingest_gguf::mmap_load_sequential.
        let _ = mmap.advise(memmap2::Advice::WillNeed);
        let gguf = GgufFile::parse(&mmap[..]).map_err(|e| format!("parse: {e:?}"))?;
        let cfg = BitNetConfig::from_gguf(&gguf)?;
        eprintln!(
            "bitnet: d_model={} vocab={} n_layers={} n_q_heads={} n_kv_heads={} \
             d_head={} rope_dim={} d_ff={} ctx={}",
            cfg.d_model,
            cfg.vocab,
            cfg.n_layers,
            cfg.n_q_heads,
            cfg.n_kv_heads,
            cfg.d_head,
            cfg.rope_dim,
            cfg.d_ff,
            cfg.context_length,
        );
        let weights = load_weights(&mmap, &gguf, &cfg)?;
        Ok(Self {
            cfg,
            gguf,
            weights,
            _mmap: mmap,
        })
    }

    pub fn new_state(&self) -> BitNetState {
        BitNetState::new(&self.cfg)
    }

    pub fn forward_logits(&self, state: &mut BitNetState, token_id: i32) -> Vec<f32> {
        forward_token(&self.weights, &self.cfg, state, token_id)
    }

    pub fn tokenizer(&self) -> Tokenizer<'_> {
        Tokenizer::from_gguf(&self.gguf).expect("tokenizer")
    }
}

/// Generation config — short, fits the base-model continuation case.
#[derive(Clone, Copy, Debug)]
pub struct BitNetGenConfig {
    pub max_new_tokens: usize,
    pub sample: SampleConfig,
    pub rep_window: usize,
}

impl Default for BitNetGenConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 32,
            sample: SampleConfig::chat(0xB17_BE7_2B),
            rep_window: 64,
        }
    }
}

impl BitNetRuntime {
    /// Pure base-model text continuation (no chat template — BitNet
    /// b1.58 2B-4T is a base LM). Mirrors `rce-mamba2::generate`:
    /// branch on `tokenizer.ggml.model` for the encoder (Llama-3 ships
    /// as "gpt2" = byte-level BPE), prefill the prompt through
    /// `forward_logits`, then sample-and-feed the continuation.
    pub fn generate(
        &self, prompt: &str, gcfg: &BitNetGenConfig, mut on_token: impl FnMut(&str),
    ) -> Result<String, String> {
        let tk = self.tokenizer();
        let is_gpt2 = self
            .gguf
            .metadata
            .iter()
            .find(|kv| kv.key == "tokenizer.ggml.model")
            .map(|kv| matches!(&kv.value, GgufValue::String(s) if s == "gpt2"))
            .unwrap_or(false);
        // BitNet's tokenizer.ggml.add_bos_token = true; honor it.
        let add_bos = self
            .gguf
            .metadata
            .iter()
            .find(|kv| kv.key == "tokenizer.ggml.add_bos_token")
            .map(|kv| matches!(&kv.value, GgufValue::Bool(b) if *b))
            .unwrap_or(false);
        let prompt_ids = if is_gpt2 {
            tk.encode_gpt2(prompt, add_bos)
        } else {
            tk.encode(prompt, add_bos)
        };
        if prompt_ids.is_empty() {
            return Err("tokenizer produced empty prompt".into());
        }
        let eos_id = tk.eos_token_id.unwrap_or(-1);

        let mut state = self.new_state();
        let mut history: Vec<i32> =
            Vec::with_capacity(prompt_ids.len() + gcfg.max_new_tokens);
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
            if i == prompt_ids.len() - 1 {
                next = pred;
            }
        }

        let mut out_ids: Vec<i32> = Vec::new();
        for _ in 0..gcfg.max_new_tokens {
            if next == eos_id {
                break;
            }
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
    use rce_ingest_gguf::GgufDtype;

    const MODEL: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/bitnet-b1.58-2b-4t-i2s.gguf";

    /// **T3.a — shape-sanity gate on the real model.** Loads the
    /// BitNet b1.58 2B GGUF, parses config, materialises every block's
    /// weights via the existing zero-copy `QuantWeight::from_gguf` (no
    /// f32 expansion), and asserts each tensor's dtype + dims match
    /// the BitNet architecture spec. A mismatch here would indicate a
    /// wrong key-namespace assumption or a shape-derivation error
    /// BEFORE any forward / accuracy work depends on it.
    #[test]
    fn t3_a_load_real_bitnet_shape_sanity() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("t3.a: BitNet model not mounted — skipping");
            return;
        }
        let rt = BitNetRuntime::load(MODEL).expect("load");
        let c = &rt.cfg;
        assert_eq!(c.d_model, 2560);
        assert_eq!(c.vocab, 128256);
        assert_eq!(c.n_layers, 30);
        assert_eq!(c.n_q_heads, 20);
        assert_eq!(c.n_kv_heads, 5);
        assert_eq!(c.d_head, 128);
        assert_eq!(c.rope_dim, 128);
        assert_eq!(c.d_ff, 6912);
        assert_eq!(c.context_length, 4096);
        assert!((c.rope_freq_base - 500_000.0).abs() < 1.0);
        assert!((c.eps - 1e-5).abs() < 1e-9);

        // token_embd: F16, [vocab, d_model].
        assert_eq!(rt.weights.token_embd.dtype, GgufDtype::F16);
        assert_eq!(rt.weights.token_embd.n_rows, c.vocab);
        assert_eq!(rt.weights.token_embd.n_cols, c.d_model);
        assert_eq!(rt.weights.output_norm.len(), c.d_model);
        assert_eq!(rt.weights.blocks.len(), c.n_layers);

        // Every block: norms F32 of correct length; matmuls I2_S of
        // correct shape (the I2_S dtype confirms T1 is wired through
        // QuantWeight::from_gguf for the body weights).
        for (b, blk) in rt.weights.blocks.iter().enumerate() {
            assert_eq!(blk.attn_norm.len(), c.d_model, "blk.{b}.attn_norm");
            assert_eq!(blk.attn_sub_norm.len(), c.d_model, "blk.{b}.attn_sub_norm");
            assert_eq!(blk.ffn_norm.len(), c.d_model, "blk.{b}.ffn_norm");
            assert_eq!(blk.ffn_sub_norm.len(), c.d_ff, "blk.{b}.ffn_sub_norm");

            let check_i2s = |w: &QuantWeight, n_rows: usize, n_cols: usize, name: &str| {
                assert_eq!(w.dtype, GgufDtype::I2S, "blk.{b}.{name} dtype");
                assert_eq!(w.n_rows, n_rows, "blk.{b}.{name} rows");
                assert_eq!(w.n_cols, n_cols, "blk.{b}.{name} cols");
            };
            check_i2s(&blk.wq, c.n_embd_q(), c.d_model, "wq");
            check_i2s(&blk.wk, c.n_embd_kv(), c.d_model, "wk");
            check_i2s(&blk.wv, c.n_embd_kv(), c.d_model, "wv");
            check_i2s(&blk.wo, c.d_model, c.n_embd_q(), "wo");
            check_i2s(&blk.ffn_gate, c.d_ff, c.d_model, "ffn_gate");
            check_i2s(&blk.ffn_up, c.d_ff, c.d_model, "ffn_up");
            check_i2s(&blk.ffn_down, c.d_model, c.d_ff, "ffn_down");
        }
        eprintln!(
            "T3.a OK: bitnet-b1.58-2b loaded — {} layers, {} body I2_S tensors + {} sub-norms + F16 token_embd, all shapes match",
            c.n_layers,
            c.n_layers * 7,
            c.n_layers * 2
        );
    }

    /// **T3.b — one full forward on the real BitNet 2B, sanity gates.**
    /// Not yet an accuracy gate (that's T3.d vs f32 reference). Just:
    /// (i) the forward returns without panic, (ii) logits len == vocab,
    /// (iii) every logit is finite, (iv) argmax is non-trivial (not
    /// 0, not vocab-1 — a 1/vocab-class random sanity that catches
    /// degenerate all-zeros / all-NaN outputs), (v) per-block KV cache
    /// grew by exactly `n_embd_kv` after the call, (vi) `state.pos`
    /// incremented. This catches structural wiring bugs (wrong sub-norm
    /// placement / wrong RoPE / wrong tied-lm_head shape) before T3.d
    /// quantifies accuracy.
    #[test]
    fn t3_b_one_real_forward_sanity() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("t3.b: BitNet model not mounted — skipping");
            return;
        }
        let rt = BitNetRuntime::load(MODEL).expect("load");
        let mut state = rt.new_state();
        // BOS for the BitNet llama-3 tokenizer is typically 128000;
        // any valid id works for a structural sanity gate.
        let token: i32 = 128000;
        let logits = rt.forward_logits(&mut state, token);

        assert_eq!(logits.len(), rt.cfg.vocab, "logits len = vocab");
        for (i, v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] not finite: {v}");
        }
        let (argmax, _) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        // Degenerate-output guards (not a quality claim — that's T3.d).
        assert!(argmax > 0 && argmax < rt.cfg.vocab - 1,
                "argmax {argmax} at trivial endpoint");
        let n_embd_kv = rt.cfg.n_embd_kv();
        for (b, bs) in state.blocks.iter().enumerate() {
            assert_eq!(bs.k_cache.len(), n_embd_kv, "blk.{b} k_cache grew by n_embd_kv");
            assert_eq!(bs.v_cache.len(), n_embd_kv, "blk.{b} v_cache grew by n_embd_kv");
        }
        assert_eq!(state.pos, 1, "state.pos incremented");
        let max_abs = logits.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        eprintln!(
            "T3.b OK: one forward (token={token}) → argmax={argmax}  max|logit|={max_abs:.3}  \
             (KV cache: {} rows × {n_embd_kv} = {} f32 per block)",
            state.pos,
            state.blocks[0].k_cache.len()
        );
    }

    /// **Diagnostic for T3.c bug** — print, per prompt token, the top-5
    /// predictions with decoded strings + the encoded token IDs. Tells
    /// us at a glance whether the model is conditioning on the prompt
    /// (top-5 should evolve sensibly as the prompt grows) or producing
    /// input-independent output. `#[ignore]`, on-demand.
    #[test]
    #[ignore = "diagnostic; on-demand"]
    fn t3_diag_top5_per_token() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("skip");
            return;
        }
        let rt = BitNetRuntime::load(MODEL).expect("load");
        let tk = rt.tokenizer();
        let prompt = "The capital of France is";
        let ids = tk.encode_gpt2(prompt, true);
        eprintln!("\nprompt: {prompt:?}");
        eprintln!("token IDs (add_bos=true): {ids:?}");
        for (i, &id) in ids.iter().enumerate() {
            let piece = tk.decode_sequence(&[id], false);
            eprintln!("  ids[{i}] = {id} → {piece:?}");
        }
        let mut state = rt.new_state();
        for (i, &id) in ids.iter().enumerate() {
            let logits = rt.forward_logits(&mut state, id);
            // Top-5 predicted next tokens.
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            idx.truncate(5);
            let top: Vec<(i32, f32, String)> = idx
                .iter()
                .map(|&j| (j as i32, logits[j], tk.decode_sequence(&[j as i32], false)))
                .collect();
            eprintln!("after token[{i}] = {id:?} (pos={i}):");
            for (tid, lg, s) in &top {
                eprintln!("    top: id={tid:6}  logit={lg:>8.3}  {s:?}");
            }
        }
    }

    /// **T3.c — generate on the real BitNet 2B, FIXED (2026-05-18).**
    /// The 2026-05-17 version reported degenerate text ("adooadoo
    /// clusteradoo cluster...") and `#[ignore]`'d itself as a recorded
    /// finding. Root cause turned out to be a **4-way value rotation
    /// in `map2bit`** in the I2_S decoder — every real `-1` was being
    /// decoded as `0`, every real `0` as `+1`, every real `+1` as `-1`.
    /// A systematic value-swap, not a permutation, not a missing
    /// activation-quant — pure code mapping. Verified against
    /// microsoft/BitNet's `dequantize_row_i2_s`:
    /// `static const float map2bit[4] = {-1, 0, +1, 0}`. Fix in
    /// `rce-ingest-gguf::dequant_i2_s` + `rce-op-qmm::mm_1d_i2_s`
    /// (scalar) + `mm_1d_i2_s_neon`. Generates coherent English now;
    /// gate is strengthened to actually require it (the prior
    /// "has-letters" gate had been too weak to catch the gibberish —
    /// itself a documented lesson).
    #[test]
    fn t3_c_generate_real_bitnet() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("t3.c: BitNet model not mounted — skipping");
            return;
        }
        let rt = BitNetRuntime::load(MODEL).expect("load");
        let cfg = BitNetGenConfig {
            max_new_tokens: 12,
            sample: SampleConfig::GREEDY,
            rep_window: 64,
        };
        let prompt = "The capital of France is";
        let mut chunks: Vec<String> = Vec::new();
        let t0 = std::time::Instant::now();
        let reply = rt
            .generate(prompt, &cfg, |tok| chunks.push(tok.to_string()))
            .expect("generate");
        let dt = t0.elapsed();
        let full = chunks.concat();
        assert!(!chunks.is_empty(), "no tokens emitted");
        assert!(!full.is_empty(), "empty continuation");
        assert!(
            full.chars().any(|c| c.is_alphabetic()),
            "continuation has no letters: {full:?}"
        );
        // **Strengthened gate** — the 2026-05-17 gibberish ("adooadoo
        // cluster...") had letters too. A coherent English continuation
        // of "The capital of France is" must contain at least one of
        // these ultra-common English function words.
        let coherent = [" the ", " a ", " of ", " is ", " in ", " and ", " to "]
            .iter()
            .any(|w| full.contains(w));
        assert!(
            coherent,
            "continuation has no common English function words \
             (likely the 2026-05-17 codec-mapping regression returning): {full:?}"
        );
        eprintln!(
            "T3.c OK: prompt={prompt:?} → continuation={full:?}  \
             [{} tok in {dt:?} = {:.2} tok/s]  reply={reply:?}",
            chunks.len(),
            chunks.len() as f64 / dt.as_secs_f64()
        );
    }
}
