//! IBM Granite 4.1-8B inference runtime on the rce substrate.
//! Dense GQA Llama-family decoder (32 Q / 8 KV heads, d_head=128,
//! 40 layers, NeoX RoPE base 10M, SwiGLU FFN, untied lm_head, vocab
//! 100352, GPT-2 BPE tokenizer with dbrx pre).
//!
//! **muP-style scaling factors** (the part that differs from plain
//! Llama-2):
//!
//! - `embedding_scale` (12.0 for 4.1-8B): token embedding is multiplied
//!   by this before feeding the decoder stack.
//! - `residual_scale` (0.22): every sub-block output (attn_out, ffn_out)
//!   is multiplied by this before being added to the residual stream.
//! - `attention.scale` (1/d_head = 0.0078125 for 4.1-8B, NOT 1/√d_head):
//!   replaces the standard `1 / sqrt(d_head)` in the softmax argument.
//! - `logit_scale` (16.0): final logits are DIVIDED by this before
//!   sampling — softens the distribution.
//!
//! Per layer:
//! ```text
//!   inpSA = x
//!   cur   = rms_norm(x, attn_norm)
//!   Q,K,V = wq(cur), wk(cur), wv(cur)
//!   Q,K   = rope_neox(Q,K, pos)         // NeoX, base=10M
//!   cur   = softmax(QK^T · attention.scale) · V    // GQA 32/8
//!   cur   = wo(cur)
//!   x     = inpSA + residual_scale · cur
//!   ffn_in = rms_norm(x, ffn_norm)
//!   ffn   = silu(gate(ffn_in)) * up(ffn_in) → down(...)
//!   x     = x + residual_scale · ffn
//! ```
//! Final: `cur = rms_norm(x, output_norm)` → `logits = (output · cur) / logit_scale`.

use rce_ingest_gguf::{
    read_tensor_f32, sample_with_history, GgufFile, GgufValue, SampleConfig, Tokenizer,
};
use rce_op_qmm::QuantWeight;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct GraniteConfig {
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
    /// muP-style scaling factors. All read from GGUF metadata if present;
    /// defaults are no-ops (1.0 / 1/√d_head) so a hand-edited GGUF
    /// without them still loads.
    pub embedding_scale: f32,
    pub residual_scale: f32,
    pub logit_scale: f32,
    /// Replaces `1 / sqrt(d_head)` in the attention softmax. For
    /// 4.1-8B this is `1 / d_head = 1/128 = 0.0078125`.
    pub attention_scale: f32,
}

impl GraniteConfig {
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
        if arch != "granite" {
            return Err(format!("expected granite, got {arch}"));
        }
        let u32_meta = |k: &str| -> Option<u32> {
            gguf.metadata
                .iter()
                .find(|kv| kv.key == format!("granite.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::U32(v) => Some(*v), _ => None })
        };
        let f32_meta = |k: &str| -> Option<f32> {
            gguf.metadata
                .iter()
                .find(|kv| kv.key == format!("granite.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::F32(v) => Some(*v), _ => None })
        };
        let d_model = u32_meta("embedding_length").ok_or("embedding_length")? as usize;
        let vocab = u32_meta("vocab_size")
            .or_else(|| gguf.tensors.iter()
                .find(|t| t.name == "token_embd.weight")
                .and_then(|t| t.dims.get(1).copied())
                .map(|d| d as u32))
            .ok_or("vocab_size")? as usize;
        let n_layers = u32_meta("block_count").ok_or("block_count")? as usize;
        let n_q_heads = u32_meta("attention.head_count").ok_or("attention.head_count")? as usize;
        let n_kv_heads = u32_meta("attention.head_count_kv").unwrap_or(n_q_heads as u32) as usize;
        let d_head = d_model / n_q_heads;
        let rope_dim = u32_meta("rope.dimension_count").unwrap_or(d_head as u32) as usize;
        let rope_freq_base = f32_meta("rope.freq_base").unwrap_or(10_000_000.0);
        let d_ff = u32_meta("feed_forward_length").ok_or("feed_forward_length")? as usize;
        let context_length = u32_meta("context_length").unwrap_or(131072) as usize;
        let eps = f32_meta("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        // Granite-specific scaling factors. Default to no-ops (1.0 /
        // 1/√d_head) if absent — a hand-built GGUF can still load.
        let embedding_scale = f32_meta("embedding_scale").unwrap_or(1.0);
        let residual_scale = f32_meta("residual_scale").unwrap_or(1.0);
        let logit_scale = f32_meta("logit_scale").unwrap_or(1.0);
        let attention_scale = f32_meta("attention.scale")
            .unwrap_or_else(|| 1.0 / (d_head as f32).sqrt());
        Ok(GraniteConfig {
            d_model, vocab, n_layers, n_q_heads, n_kv_heads, d_head,
            rope_dim, rope_freq_base, d_ff, context_length, eps,
            embedding_scale, residual_scale, logit_scale, attention_scale,
        })
    }
    #[inline] pub fn n_embd_q(&self) -> usize { self.n_q_heads * self.d_head }
    #[inline] pub fn n_embd_kv(&self) -> usize { self.n_kv_heads * self.d_head }
}

pub struct GraniteBlock {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub wq: QuantWeight, pub wk: QuantWeight, pub wv: QuantWeight, pub wo: QuantWeight,
    pub ffn_gate: QuantWeight, pub ffn_up: QuantWeight, pub ffn_down: QuantWeight,
}

pub struct GraniteWeights {
    pub token_embd: QuantWeight,
    pub output_norm: Vec<f32>,
    pub lm_head: QuantWeight, // separate `output.weight` (NOT tied)
    pub blocks: Vec<GraniteBlock>,
}

pub fn load_weights(
    mmap: &Arc<Mmap>, gguf: &GgufFile, cfg: &GraniteConfig,
) -> Result<GraniteWeights, String> {
    let bytes: &[u8] = &mmap[..];
    let token_embd = QuantWeight::from_gguf(
        Arc::clone(mmap), gguf, "token_embd.weight", cfg.vocab, cfg.d_model,
    ).map_err(|e| format!("token_embd: {e}"))?;
    let output_norm = read_tensor_f32(bytes, gguf, "output_norm.weight")
        .map_err(|e| format!("output_norm: {e:?}"))?;
    let lm_head = QuantWeight::from_gguf(
        Arc::clone(mmap), gguf, "output.weight", cfg.vocab, cfg.d_model,
    ).map_err(|e| format!("output: {e}"))?;
    let mut blocks = Vec::with_capacity(cfg.n_layers);
    for il in 0..cfg.n_layers {
        let bn = format!("blk.{il}");
        let f32n = |s: &str| read_tensor_f32(bytes, gguf, &format!("{bn}.{s}.weight"))
            .map_err(|e| format!("blk.{il}.{s}: {e:?}"));
        let qw = |s: &str, n_rows: usize, n_cols: usize| QuantWeight::from_gguf(
            Arc::clone(mmap), gguf, &format!("{bn}.{s}.weight"), n_rows, n_cols,
        ).map_err(|e| format!("blk.{il}.{s}: {e}"));
        blocks.push(GraniteBlock {
            attn_norm: f32n("attn_norm")?,
            ffn_norm: f32n("ffn_norm")?,
            wq: qw("attn_q", cfg.n_embd_q(), cfg.d_model)?,
            wk: qw("attn_k", cfg.n_embd_kv(), cfg.d_model)?,
            wv: qw("attn_v", cfg.n_embd_kv(), cfg.d_model)?,
            wo: qw("attn_output", cfg.d_model, cfg.n_embd_q())?,
            ffn_gate: qw("ffn_gate", cfg.d_ff, cfg.d_model)?,
            ffn_up: qw("ffn_up", cfg.d_ff, cfg.d_model)?,
            ffn_down: qw("ffn_down", cfg.d_model, cfg.d_ff)?,
        });
    }
    Ok(GraniteWeights { token_embd, output_norm, lm_head, blocks })
}

#[derive(Clone)]
pub struct BlockState { pub k_cache: Vec<f32>, pub v_cache: Vec<f32> }
#[derive(Clone)]
pub struct GraniteState { pub blocks: Vec<BlockState>, pub pos: usize }
impl GraniteState {
    pub fn new(cfg: &GraniteConfig) -> Self {
        let cap = cfg.context_length * cfg.n_embd_kv();
        let blocks = (0..cfg.n_layers).map(|_| BlockState {
            k_cache: Vec::with_capacity(cap),
            v_cache: Vec::with_capacity(cap),
        }).collect();
        Self { blocks, pos: 0 }
    }
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(w.iter()).map(|(v, ww)| v * inv * ww).collect()
}
fn silu(x: f32) -> f32 {
    if x >= 0.0 { x / (1.0 + (-x).exp()) } else { let e = x.exp(); x * e / (1.0 + e) }
}
fn qmm_1d(a: &[f32], w: &QuantWeight) -> Vec<f32> {
    // Adapter precedence: placement → direct. Zero overhead when off.
    #[cfg(feature = "placement")]
    if let Some(out) = placement_thread_local::try_qmm(a, w) {
        return out;
    }
    let mut out = vec![0.0f32; w.n_rows];
    rce_op_qmm::mm_1d(a, w, &mut out);
    out
}

/// Placement dispatch hook: cost-driven qmm routing.
/// See rce-tinyllama's same-named module for the canonical doc.
/// Activation: `RCE_GRANITE_PLACEMENT=1` (CPU-only) or `=hybrid` (CPU + native Metal).
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
        match std::env::var("RCE_GRANITE_PLACEMENT").as_deref() {
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
fn apply_rope_inplace(
    x: &mut [f32], n_heads: usize, d_head: usize, rope_dim: usize, base: f32, pos: usize,
) {
    // Per-head NEON-vectorized RoPE via rce-op-attn::rope_inplace.
    // Numerics match scalar to 1e-5; see rce-op-attn::tests::rope_matches_scalar.
    for h in 0..n_heads {
        let head_off = h * d_head;
        rce_op_attn::rope_inplace(&mut x[head_off..head_off + d_head], rope_dim, base, pos);
    }
}

fn attn_block(
    x_in: &[f32], blk: &GraniteBlock, cfg: &GraniteConfig,
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
    // Granite uses an explicit `attention.scale` from GGUF (= 1/d_head
    // for 4.1-8B, vs the standard 1/√d_head). Falls back to 1/√d_head
    // if metadata absent.
    let scale = cfg.attention_scale;
    let q_per_kv = nq / nkv;
    // NEON-vectorized fused attention via rce-op-attn::fused_attend_head —
    // one pass per (q-head, seq) keeping (max, sum, out) online; never
    // materializes the seq-length score buffer. Matches the three-pass
    // scalar reference to 1e-4 (FP reduction reorder); see
    // rce-op-attn::tests::fused_attend_matches_three_pass.
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
    qmm_1d(&attended, &blk.wo)
}

pub fn forward_token(
    w: &GraniteWeights, cfg: &GraniteConfig, state: &mut GraniteState, token_id: i32,
) -> Vec<f32> {
    let dm = cfg.d_model;
    let mut x: Vec<f32> = w.token_embd.dequant_row(token_id as usize);
    // Granite muP: scale embedding before the decoder stack.
    let es = cfg.embedding_scale;
    if es != 1.0 { for v in x.iter_mut() { *v *= es; } }
    let pos = state.pos;
    let rs = cfg.residual_scale;
    for (il, blk) in w.blocks.iter().enumerate() {
        let normed = rms_norm(&x, &blk.attn_norm, cfg.eps);
        let bs = &mut state.blocks[il];
        let attn_out = attn_block(&normed, blk, cfg, &mut bs.k_cache, &mut bs.v_cache, pos);
        // Residual add with muP scale on the sub-block output.
        for i in 0..dm { x[i] += rs * attn_out[i]; }
        let ffn_in = rms_norm(&x, &blk.ffn_norm, cfg.eps);
        let g = qmm_1d(&ffn_in, &blk.ffn_gate);
        let u = qmm_1d(&ffn_in, &blk.ffn_up);
        let mut h = vec![0.0f32; cfg.d_ff];
        for i in 0..cfg.d_ff { h[i] = silu(g[i]) * u[i]; }
        let ffn_out = qmm_1d(&h, &blk.ffn_down);
        for i in 0..dm { x[i] += rs * ffn_out[i]; }
    }
    let final_normed = rms_norm(&x, &w.output_norm, cfg.eps);
    let mut logits = qmm_1d(&final_normed, &w.lm_head);
    // Granite muP: divide logits by logit_scale before sampling.
    let ls = cfg.logit_scale;
    if ls != 1.0 { let inv = 1.0 / ls; for v in logits.iter_mut() { *v *= inv; } }
    state.pos += 1;
    logits
}

pub struct GraniteRuntime {
    pub cfg: GraniteConfig,
    pub gguf: GgufFile,
    pub weights: GraniteWeights,
    _mmap: Arc<Mmap>,
}

impl GraniteRuntime {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| format!("open: {e}"))?;
        let mmap = Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?);
        // WillNeed (not Sequential): keeps the model page-resident so
        // decode hits RAM speed instead of re-faulting weights from disk.
        let _ = mmap.advise(memmap2::Advice::WillNeed);
        let gguf = GgufFile::parse(&mmap[..]).map_err(|e| format!("parse: {e:?}"))?;
        let cfg = GraniteConfig::from_gguf(&gguf)?;
        eprintln!(
            "granite: d_model={} vocab={} n_layers={} n_heads={}/{} d_head={} d_ff={} ctx={} rope={}",
            cfg.d_model, cfg.vocab, cfg.n_layers, cfg.n_q_heads, cfg.n_kv_heads,
            cfg.d_head, cfg.d_ff, cfg.context_length, cfg.rope_freq_base,
        );
        let weights = load_weights(&mmap, &gguf, &cfg)?;
        Ok(Self { cfg, gguf, weights, _mmap: mmap })
    }
    pub fn new_state(&self) -> GraniteState { GraniteState::new(&self.cfg) }
    pub fn forward_logits(&self, state: &mut GraniteState, token_id: i32) -> Vec<f32> {
        forward_token(&self.weights, &self.cfg, state, token_id)
    }
    pub fn tokenizer(&self) -> Tokenizer<'_> { Tokenizer::from_gguf(&self.gguf).expect("tokenizer") }
}

#[derive(Clone, Copy, Debug)]
pub struct GraniteGenConfig {
    pub max_new_tokens: usize,
    pub sample: SampleConfig,
    pub rep_window: usize,
}
impl Default for GraniteGenConfig {
    fn default() -> Self {
        Self { max_new_tokens: 32, sample: SampleConfig::chat(0x7111_7AAA), rep_window: 64 }
    }
}

impl GraniteRuntime {
    pub fn generate(
        &self, prompt: &str, gcfg: &GraniteGenConfig, mut on_token: impl FnMut(&str),
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
        let prompt_ids = if is_gpt2 { tk.encode_gpt2(prompt, add_bos) } else { tk.encode(prompt, add_bos) };
        if prompt_ids.is_empty() { return Err("empty prompt".into()); }
        let eos_id = tk.eos_token_id.unwrap_or(-1);
        let mut state = self.new_state();
        let mut history: Vec<i32> = Vec::with_capacity(prompt_ids.len() + gcfg.max_new_tokens);
        let mut step: u64 = 0;
        let seed_for = |s: u64| { let mut c = gcfg.sample; c.seed = gcfg.sample.seed ^ s.wrapping_mul(0x9E37_79B9); c };
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

    /// Multi-turn `generate` with prefix-cache hit/skip. Same return
    /// semantics as [`Self::generate`], but the caller's
    /// `rce_prefix_cache::PrefixCache<GraniteState>` is consulted
    /// for a longest-matching prefix snapshot before prefill. On hit,
    /// the K/V cache is restored from the snapshot and only the
    /// remaining suffix is forward-passed. On miss, behaves identically
    /// to `generate`. After the response is generated, the state is
    /// snapshotted into the cache at the post-prefill boundary so the
    /// NEXT turn that shares this prefix can skip it.
    ///
    /// Per [SGLang](https://www.lmsys.org/blog/2024-01-17-sglang/),
    /// 75-90% hit rate on multi-turn chat / 6.4× speedup on RAG.
    pub fn generate_cached(
        &self,
        prompt: &str,
        gcfg: &GraniteGenConfig,
        cache: &mut rce_prefix_cache::PrefixCache<GraniteState>,
        mut on_token: impl FnMut(&str),
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
        let prompt_ids = if is_gpt2 { tk.encode_gpt2(prompt, add_bos) } else { tk.encode(prompt, add_bos) };
        if prompt_ids.is_empty() { return Err("empty prompt".into()); }
        let eos_id = tk.eos_token_id.unwrap_or(-1);

        // Snapshot boundary: cache state at length N-1 (NOT N), so a
        // future turn that re-enters with the same prompt finds a
        // length-(N-1) match and still does ONE fresh forward to
        // produce the sampling logits. Without this guarantee, a full
        // cache hit (match == N) would have no token to forward and
        // `next` would stay unset.
        let cache_at = prompt_ids.len().saturating_sub(1);

        // Prefix-cache lookup: longest matching prefix snapshot.
        let (start_idx, mut state) = match cache.longest_prefix(&prompt_ids) {
            Some((n, snap)) if n <= cache_at => (n, snap),
            _ => (0, self.new_state()),
        };

        let mut history: Vec<i32> = prompt_ids[..start_idx].to_vec();
        history.reserve(prompt_ids.len() - start_idx + gcfg.max_new_tokens);
        let mut step: u64 = start_idx as u64;
        let seed_for = |s: u64| { let mut c = gcfg.sample; c.seed = gcfg.sample.seed ^ s.wrapping_mul(0x9E37_79B9); c };
        let mut next: i32 = -1;
        for (i, &tok) in prompt_ids[start_idx..].iter().enumerate() {
            let pos_in_prompt = start_idx + i;
            // Snapshot just BEFORE the final prompt token so future
            // turns get a length-(N-1) hit.
            if pos_in_prompt == cache_at && cache_at > 0 {
                cache.insert(&prompt_ids[..cache_at], state.clone());
            }
            let logits = self.forward_logits(&mut state, tok);
            history.push(tok);
            let recent = &history[history.len().saturating_sub(gcfg.rep_window)..];
            let pred = sample_with_history(&logits, &seed_for(step), recent);
            step += 1;
            if pos_in_prompt == prompt_ids.len() - 1 { next = pred; }
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
    const MODEL: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/granite-4.1-8b-gguf/granite-4.1-8b-Q8_0.gguf";

    #[test]
    fn g41_a_load_real_model_shape_sanity() {
        if !std::path::Path::new(MODEL).exists() { eprintln!("G41.a: model not mounted — skipping"); return; }
        let rt = GraniteRuntime::load(MODEL).expect("load");
        let c = &rt.cfg;
        assert_eq!(c.d_model, 4096);
        assert_eq!(c.n_layers, 40);
        assert_eq!(c.n_q_heads, 32);
        assert_eq!(c.n_kv_heads, 8);
        assert_eq!(c.d_head, 128);
        assert_eq!(c.d_ff, 12800);
        assert_eq!(c.vocab, 100352);
        assert!((c.embedding_scale - 12.0).abs() < 1e-3, "embedding_scale={}", c.embedding_scale);
        assert!((c.residual_scale - 0.22).abs() < 1e-2, "residual_scale={}", c.residual_scale);
        assert!((c.logit_scale - 16.0).abs() < 1e-3, "logit_scale={}", c.logit_scale);
        assert!((c.attention_scale - (1.0 / 128.0)).abs() < 1e-6, "attention_scale={}", c.attention_scale);
        assert_eq!(rt.weights.lm_head.n_rows, c.vocab);
        assert_eq!(rt.weights.lm_head.n_cols, c.d_model);
        assert_eq!(rt.weights.blocks.len(), c.n_layers);
        eprintln!(
            "G41.a OK: Granite 4.1-8B loaded — {} layers, embedding_scale={}, residual_scale={}, logit_scale={}, attention_scale={}",
            c.n_layers, c.embedding_scale, c.residual_scale, c.logit_scale, c.attention_scale,
        );
    }

    #[test]
    fn g41_b_one_real_forward_sanity() {
        if !std::path::Path::new(MODEL).exists() { eprintln!("G41.b: model not mounted — skipping"); return; }
        let rt = GraniteRuntime::load(MODEL).expect("load");
        let mut state = rt.new_state();
        // Granite BOS = 100257.
        let logits = rt.forward_logits(&mut state, 100257);
        assert_eq!(logits.len(), rt.cfg.vocab);
        for v in &logits { assert!(v.is_finite(), "non-finite logit"); }
        let (argmax, _) = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        assert!(argmax < rt.cfg.vocab);
        eprintln!("G41.b OK: one forward → argmax={argmax}, max|logit|={:.3}",
            logits.iter().fold(0.0f32, |a, v| a.max(v.abs())));
    }

    #[test]
    fn tl_c_generate_real_granite() {
        if !std::path::Path::new(MODEL).exists() { eprintln!("TL.c: model not mounted — skipping"); return; }
        let rt = GraniteRuntime::load(MODEL).expect("load");
        let cfg = GraniteGenConfig { max_new_tokens: 16, sample: SampleConfig::GREEDY, rep_window: 64 };
        let prompt = "The capital of France is";
        let mut chunks: Vec<String> = Vec::new();
        let t0 = std::time::Instant::now();
        let _reply = rt.generate(prompt, &cfg, |tok| chunks.push(tok.to_string())).expect("generate");
        let dt = t0.elapsed();
        let full = chunks.concat();
        assert!(!full.is_empty(), "empty");
        let coherent = [" the ", " a ", " of ", " is ", " in ", " and ", " to ", "Paris"]
            .iter().any(|w| full.contains(w))
            || full.chars().filter(|c| c.is_alphabetic()).count() >= full.len() / 2;
        assert!(coherent, "no common English: {full:?}");
        eprintln!(
            "TL.c OK: prompt={prompt:?} → continuation={full:?}  [{} tok in {dt:?} = {:.2} tok/s]",
            chunks.len(), chunks.len() as f64 / dt.as_secs_f64()
        );
    }

    #[test]
    fn tl_d_prefix_cache_skip_prefill() {
        if !std::path::Path::new(MODEL).exists() { eprintln!("TL.d: model not mounted — skipping"); return; }
        let rt = GraniteRuntime::load(MODEL).expect("load");
        let cfg = GraniteGenConfig { max_new_tokens: 8, sample: SampleConfig::GREEDY, rep_window: 64 };
        let mut cache: rce_prefix_cache::PrefixCache<GraniteState> =
            rce_prefix_cache::PrefixCache::new(64);

        // Turn 1: cold cache.
        let prompt = "The capital of France is";
        let t0 = std::time::Instant::now();
        let r1 = rt.generate_cached(prompt, &cfg, &mut cache, |_| {}).expect("turn 1");
        let dt1 = t0.elapsed();

        // Turn 2: same prompt → should be a full cache hit, prefill skipped.
        let t0 = std::time::Instant::now();
        let r2 = rt.generate_cached(prompt, &cfg, &mut cache, |_| {}).expect("turn 2");
        let dt2 = t0.elapsed();

        // Same prompt, greedy sampling → same continuation.
        assert_eq!(r1, r2, "cached run produced different output");
        // Telemetry: at least one hit recorded.
        assert!(cache.hits >= 1, "expected ≥1 hit, got {} hits / {} misses", cache.hits, cache.misses);
        eprintln!(
            "TL.d OK: turn1 (cold) {dt1:?}, turn2 (hit) {dt2:?}, hits={}, misses={}, output={r1:?}",
            cache.hits, cache.misses,
        );
    }
}
