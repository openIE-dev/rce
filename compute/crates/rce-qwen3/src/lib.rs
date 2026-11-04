//! Qwen3 inference runtime on the rce substrate. Plain Llama-2-style
//! transformer + **per-head QK-norm** (`attn_q_norm`/`attn_k_norm`,
//! dim = `d_head`) + NeoX RoPE + tied lm_head. Mirrors the vendored
//! `llm_build_qwen3` (`use_kq_norm=true`, no QKV biases, separate
//! `output.weight` falls back to tied `token_embd.weight`).
//!
//! Per layer:
//! ```text
//!   inpSA = x
//!   cur   = rms_norm(x, attn_norm)
//!   Q,K,V = wq(cur), wk(cur), wv(cur)
//!   Q     = rms_norm_per_head(Q, attn_q_norm)   ← *** per-head QK-norm ***
//!   K     = rms_norm_per_head(K, attn_k_norm)
//!   Q,K   = rope_neox(Q, K, pos)
//!   cur   = softmax(QK^T/√d) · V               ← GQA (16q / 8kv on 1.7B)
//!   cur   = wo(cur)
//!   x     = inpSA + cur
//!   ffn   = silu(gate(rms_norm(x, ffn_norm))) * up(...) → down(...)
//!   x     = x + ffn
//! ```
//! Final: `cur = rms_norm(x, output_norm)` → `logits = token_embd · cur`
//! (tied lm_head; no separate `output.weight` in the prism-ml Bonsai
//! GGUFs).

use rce_ingest_gguf::{
    read_tensor_f32, sample_with_history, GgufFile, GgufValue, SampleConfig, Tokenizer,
};
use rce_op_qmm::QuantWeight;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Qwen3Config {
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

impl Qwen3Config {
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
        if arch != "qwen3" {
            return Err(format!("expected qwen3, got {arch}"));
        }
        let u32_meta = |k: &str| -> Option<u32> {
            gguf.metadata
                .iter()
                .find(|kv| kv.key == format!("qwen3.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::U32(v) => Some(*v), _ => None })
        };
        let f32_meta = |k: &str| -> Option<f32> {
            gguf.metadata
                .iter()
                .find(|kv| kv.key == format!("qwen3.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::F32(v) => Some(*v), _ => None })
        };
        let d_model = u32_meta("embedding_length").ok_or("embedding_length")? as usize;
        let vocab = u32_meta("vocab_size")
            .or_else(|| gguf.tensors.iter()
                .find(|t| t.name == "token_embd.weight")
                .and_then(|t| t.dims.get(1).copied())
                .map(|d| d as u32))
            .ok_or("vocab_size")? as usize;
        // block_count missing in some prism-ml gguf metadata — derive
        // from highest `blk.{n}.*` tensor index when absent.
        let n_layers = u32_meta("block_count")
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                let mut max_il: i32 = -1;
                for t in &gguf.tensors {
                    if let Some(rest) = t.name.strip_prefix("blk.") {
                        if let Some(dot) = rest.find('.') {
                            if let Ok(il) = rest[..dot].parse::<i32>() {
                                if il > max_il { max_il = il; }
                            }
                        }
                    }
                }
                (max_il + 1).max(1) as usize
            });
        let n_q_heads = u32_meta("attention.head_count").ok_or("attention.head_count")? as usize;
        let n_kv_heads = u32_meta("attention.head_count_kv").unwrap_or(n_q_heads as u32) as usize;
        // Qwen3 stores per-head dim via the q_norm tensor (dim = d_head).
        let d_head = gguf.tensors.iter()
            .find(|t| t.name == "blk.0.attn_q_norm.weight")
            .and_then(|t| t.dims.first().copied())
            .map(|d| d as usize)
            .unwrap_or(d_model / n_q_heads);
        let rope_dim = u32_meta("rope.dimension_count").unwrap_or(d_head as u32) as usize;
        let rope_freq_base = f32_meta("rope.freq_base").unwrap_or(1_000_000.0);
        let d_ff = u32_meta("feed_forward_length").ok_or("feed_forward_length")? as usize;
        let context_length = u32_meta("context_length").unwrap_or(8192) as usize;
        let eps = f32_meta("attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        Ok(Qwen3Config {
            d_model, vocab, n_layers, n_q_heads, n_kv_heads, d_head,
            rope_dim, rope_freq_base, d_ff, context_length, eps,
        })
    }
    #[inline] pub fn n_embd_q(&self) -> usize { self.n_q_heads * self.d_head }
    #[inline] pub fn n_embd_kv(&self) -> usize { self.n_kv_heads * self.d_head }
}

pub struct Qwen3Block {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub q_norm: Vec<f32>,           // per-head, dim = d_head
    pub k_norm: Vec<f32>,           // per-head, dim = d_head
    pub wq: QuantWeight, pub wk: QuantWeight, pub wv: QuantWeight, pub wo: QuantWeight,
    pub ffn_gate: QuantWeight, pub ffn_up: QuantWeight, pub ffn_down: QuantWeight,
}

pub struct Qwen3Weights {
    pub token_embd: QuantWeight,    // doubles as lm_head (tied)
    pub output_norm: Vec<f32>,
    pub blocks: Vec<Qwen3Block>,
}

pub fn load_weights(
    mmap: &Arc<Mmap>, gguf: &GgufFile, cfg: &Qwen3Config,
) -> Result<Qwen3Weights, String> {
    let bytes: &[u8] = &mmap[..];
    let token_embd = QuantWeight::from_gguf(
        Arc::clone(mmap), gguf, "token_embd.weight", cfg.vocab, cfg.d_model,
    ).map_err(|e| format!("token_embd: {e}"))?;
    let output_norm = read_tensor_f32(bytes, gguf, "output_norm.weight")
        .map_err(|e| format!("output_norm: {e:?}"))?;
    let mut blocks = Vec::with_capacity(cfg.n_layers);
    for il in 0..cfg.n_layers {
        let bn = format!("blk.{il}");
        let f32n = |s: &str| read_tensor_f32(bytes, gguf, &format!("{bn}.{s}.weight"))
            .map_err(|e| format!("blk.{il}.{s}: {e:?}"));
        let qw = |s: &str, n_rows: usize, n_cols: usize| QuantWeight::from_gguf(
            Arc::clone(mmap), gguf, &format!("{bn}.{s}.weight"), n_rows, n_cols,
        ).map_err(|e| format!("blk.{il}.{s}: {e}"));
        blocks.push(Qwen3Block {
            attn_norm: f32n("attn_norm")?,
            ffn_norm: f32n("ffn_norm")?,
            q_norm: f32n("attn_q_norm")?,
            k_norm: f32n("attn_k_norm")?,
            wq: qw("attn_q", cfg.n_embd_q(), cfg.d_model)?,
            wk: qw("attn_k", cfg.n_embd_kv(), cfg.d_model)?,
            wv: qw("attn_v", cfg.n_embd_kv(), cfg.d_model)?,
            wo: qw("attn_output", cfg.d_model, cfg.n_embd_q())?,
            ffn_gate: qw("ffn_gate", cfg.d_ff, cfg.d_model)?,
            ffn_up: qw("ffn_up", cfg.d_ff, cfg.d_model)?,
            ffn_down: qw("ffn_down", cfg.d_model, cfg.d_ff)?,
        });
    }
    Ok(Qwen3Weights { token_embd, output_norm, blocks })
}

pub struct BlockState { pub k_cache: Vec<f32>, pub v_cache: Vec<f32> }
pub struct Qwen3State { pub blocks: Vec<BlockState>, pub pos: usize }
impl Qwen3State {
    pub fn new(cfg: &Qwen3Config) -> Self {
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

/// Per-head RMS norm: split `v` of length `n_heads * d_head` into
/// `n_heads` slices of `d_head`, normalize each independently with the
/// shared `w` (length `d_head`). Qwen3 QK-norm convention (matches LFM2).
fn rms_norm_per_head(v: &[f32], w: &[f32], n_heads: usize, d_head: usize, eps: f32) -> Vec<f32> {
    debug_assert_eq!(v.len(), n_heads * d_head);
    debug_assert_eq!(w.len(), d_head);
    let mut out = vec![0.0f32; v.len()];
    for h in 0..n_heads {
        let off = h * d_head;
        let mean_sq: f32 = v[off..off + d_head].iter().map(|v| v * v).sum::<f32>()
            / d_head as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for d in 0..d_head {
            out[off + d] = v[off + d] * inv * w[d];
        }
    }
    out
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
/// Activation: `RCE_QWEN3_PLACEMENT=1` (CPU-only) or `=hybrid` (CPU + native Metal).
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
        match std::env::var("RCE_QWEN3_PLACEMENT").as_deref() {
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

/// NeoX RoPE — rotate first `rope_dim` dims of each head using the
/// `i / i+half` pair convention. Same as rce-tinyllama / rce-lfm2.
fn apply_rope_inplace(
    x: &mut [f32], n_heads: usize, d_head: usize, rope_dim: usize, base: f32, pos: usize,
) {
    // NEON-vectorized RoPE via rce-op-attn.
    for h in 0..n_heads {
        let head_off = h * d_head;
        rce_op_attn::rope_inplace(&mut x[head_off..head_off + d_head], rope_dim, base, pos);
    }
}

fn attn_block(
    x_in: &[f32], blk: &Qwen3Block, cfg: &Qwen3Config,
    k_cache: &mut Vec<f32>, v_cache: &mut Vec<f32>, pos: usize,
) -> Vec<f32> {
    let dh = cfg.d_head;
    let nq = cfg.n_q_heads;
    let nkv = cfg.n_kv_heads;
    let q_dim = nq * dh;
    let kv_dim = nkv * dh;

    let q_raw = qmm_1d(x_in, &blk.wq);
    let k_raw = qmm_1d(x_in, &blk.wk);
    let v_new = qmm_1d(x_in, &blk.wv);

    // *** Qwen3-specific *** — per-head QK-norm BEFORE RoPE.
    let mut q = rms_norm_per_head(&q_raw, &blk.q_norm, nq, dh, cfg.eps);
    let mut k_new = rms_norm_per_head(&k_raw, &blk.k_norm, nkv, dh, cfg.eps);

    apply_rope_inplace(&mut q, nq, dh, cfg.rope_dim, cfg.rope_freq_base, pos);
    apply_rope_inplace(&mut k_new, nkv, dh, cfg.rope_dim, cfg.rope_freq_base, pos);

    k_cache.extend_from_slice(&k_new);
    v_cache.extend_from_slice(&v_new);
    let seq_len = k_cache.len() / kv_dim;
    let scale = 1.0 / (dh as f32).sqrt();
    let q_per_kv = nq / nkv;
    // NEON-vectorized fused attention via rce-op-attn.
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
    w: &Qwen3Weights, cfg: &Qwen3Config, state: &mut Qwen3State, token_id: i32,
) -> Vec<f32> {
    let dm = cfg.d_model;
    let mut x: Vec<f32> = w.token_embd.dequant_row(token_id as usize);
    let pos = state.pos;
    for (il, blk) in w.blocks.iter().enumerate() {
        let normed = rms_norm(&x, &blk.attn_norm, cfg.eps);
        let bs = &mut state.blocks[il];
        let attn_out = attn_block(&normed, blk, cfg, &mut bs.k_cache, &mut bs.v_cache, pos);
        for i in 0..dm { x[i] += attn_out[i]; }
        let ffn_in = rms_norm(&x, &blk.ffn_norm, cfg.eps);
        let g = qmm_1d(&ffn_in, &blk.ffn_gate);
        let u = qmm_1d(&ffn_in, &blk.ffn_up);
        let mut h = vec![0.0f32; cfg.d_ff];
        for i in 0..cfg.d_ff { h[i] = silu(g[i]) * u[i]; }
        let ffn_out = qmm_1d(&h, &blk.ffn_down);
        for i in 0..dm { x[i] += ffn_out[i]; }
    }
    let final_normed = rms_norm(&x, &w.output_norm, cfg.eps);
    let logits = qmm_1d(&final_normed, &w.token_embd); // tied lm_head
    state.pos += 1;
    logits
}

pub struct Qwen3Runtime {
    pub cfg: Qwen3Config,
    pub gguf: GgufFile,
    pub weights: Qwen3Weights,
    _mmap: Arc<Mmap>,
}

impl Qwen3Runtime {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| format!("open: {e}"))?;
        let mmap = Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?);
        // Hint sequential reads during the load-time weight scan; ignore
        // errors (kernel may refuse). See rce_ingest_gguf::mmap_load_sequential.
        let _ = mmap.advise(memmap2::Advice::WillNeed);
        let gguf = GgufFile::parse(&mmap[..]).map_err(|e| format!("parse: {e:?}"))?;
        let cfg = Qwen3Config::from_gguf(&gguf)?;
        eprintln!(
            "qwen3: d_model={} vocab={} n_layers={} n_heads={}/{} d_head={} d_ff={} ctx={} rope={}",
            cfg.d_model, cfg.vocab, cfg.n_layers, cfg.n_q_heads, cfg.n_kv_heads,
            cfg.d_head, cfg.d_ff, cfg.context_length, cfg.rope_freq_base,
        );
        let weights = load_weights(&mmap, &gguf, &cfg)?;
        Ok(Self { cfg, gguf, weights, _mmap: mmap })
    }
    pub fn new_state(&self) -> Qwen3State { Qwen3State::new(&self.cfg) }
    pub fn forward_logits(&self, state: &mut Qwen3State, token_id: i32) -> Vec<f32> {
        forward_token(&self.weights, &self.cfg, state, token_id)
    }
    pub fn tokenizer(&self) -> Tokenizer<'_> { Tokenizer::from_gguf(&self.gguf).expect("tokenizer") }
}

#[derive(Clone, Copy, Debug)]
pub struct Qwen3GenConfig {
    pub max_new_tokens: usize,
    pub sample: SampleConfig,
    pub rep_window: usize,
}
impl Default for Qwen3GenConfig {
    fn default() -> Self {
        Self { max_new_tokens: 32, sample: SampleConfig::chat(0x7111_7AAA), rep_window: 64 }
    }
}

impl Qwen3Runtime {
    pub fn generate(
        &self, prompt: &str, gcfg: &Qwen3GenConfig, mut on_token: impl FnMut(&str),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    const MODEL_Q1: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/bonsai-1.7b-q1_0.gguf";
    const MODEL_Q2: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/ternary-bonsai-1.7b-q2_0.gguf";

    /// **Q3.a — load real Bonsai-1.7B (Q1_0) + shape sanity.**
    /// Asserts the model's reported shape matches what the parser
    /// derived, and that all weight tensors land in the expected
    /// QuantWeight slots without dtype-mapping errors.
    #[test]
    fn q3a_load_bonsai_q1_0() {
        if !std::path::Path::new(MODEL_Q1).exists() { eprintln!("Q3.a: model not mounted — skipping"); return; }
        let rt = Qwen3Runtime::load(MODEL_Q1).expect("load");
        let c = &rt.cfg;
        assert_eq!(c.d_model, 2048);
        assert_eq!(c.n_q_heads, 16);
        assert_eq!(c.n_kv_heads, 8);
        assert_eq!(c.d_head, 128);
        assert_eq!(c.d_ff, 6144);
        assert!(c.n_layers >= 24, "expected ≥24 layers, got {}", c.n_layers);
        assert_eq!(rt.weights.token_embd.n_rows, c.vocab);
        assert_eq!(rt.weights.token_embd.n_cols, c.d_model);
        assert_eq!(rt.weights.blocks.len(), c.n_layers);
        eprintln!("Q3.a OK: bonsai-1.7b-q1_0 loaded — {} layers, shapes match", c.n_layers);
    }

    /// **Q3.b — one forward on real Bonsai (Q1_0) produces finite logits.**
    /// Smoke test of the full forward path on the 1-bit codec; argmax
    /// must be a valid vocab index, magnitudes must be bounded.
    #[test]
    fn q3b_one_forward_bonsai_q1_0() {
        if !std::path::Path::new(MODEL_Q1).exists() { eprintln!("Q3.b: model not mounted — skipping"); return; }
        let rt = Qwen3Runtime::load(MODEL_Q1).expect("load");
        let mut state = rt.new_state();
        let logits = rt.forward_logits(&mut state, 1);
        assert_eq!(logits.len(), rt.cfg.vocab);
        for v in &logits { assert!(v.is_finite(), "non-finite logit"); }
        let (argmax, _) = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        assert!(argmax > 0 && argmax < rt.cfg.vocab - 1);
        eprintln!("Q3.b OK: one forward → argmax={argmax}, max|logit|={:.3}",
            logits.iter().fold(0.0f32, |a, v| a.max(v.abs())));
    }

    /// **Q3.c — coherent text generation on Bonsai Q1_0.**
    /// Greedy generation must produce common English function words OR
    /// a majority-alphabetic continuation (matches the discipline from
    /// LFM2/TinyLlama/BitNet's coherence gate — strong enough to catch
    /// a code-mapping rotation that lets garbled output sneak by).
    #[test]
    fn q3c_generate_bonsai_q1_0() {
        if !std::path::Path::new(MODEL_Q1).exists() { eprintln!("Q3.c: model not mounted — skipping"); return; }
        let rt = Qwen3Runtime::load(MODEL_Q1).expect("load");
        let cfg = Qwen3GenConfig { max_new_tokens: 16, sample: SampleConfig::GREEDY, rep_window: 64 };
        let prompt = "The capital of France is";
        let mut chunks: Vec<String> = Vec::new();
        let t0 = std::time::Instant::now();
        let _ = rt.generate(prompt, &cfg, |tok| chunks.push(tok.to_string())).expect("generate");
        let dt = t0.elapsed();
        let full = chunks.concat();
        assert!(!full.is_empty(), "empty");
        let coherent = [" the ", " a ", " of ", " is ", " in ", " and ", " to ", "Paris"]
            .iter().any(|w| full.contains(w))
            || full.chars().filter(|c| c.is_alphabetic()).count() >= full.len() / 2;
        assert!(coherent, "no common English: {full:?}");
        eprintln!(
            "Q3.c OK (Q1_0): prompt={prompt:?} → {full:?}  [{} tok in {dt:?} = {:.2} tok/s]",
            chunks.len(), chunks.len() as f64 / dt.as_secs_f64()
        );
    }

    /// **Q3.e — Bonsai-4B Q1_0 loads + one forward.**
    /// Larger sibling of Q3.a/b on the same arch+codec — verifies the
    /// runtime scales to the 4B shape (n_layers, d_model, d_ff scaling).
    #[test]
    fn q3e_bonsai_4b_q1_0_load_and_forward() {
        const M: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/bonsai-4b-q1_0.gguf";
        if !std::path::Path::new(M).exists() { eprintln!("Q3.e: model not mounted — skipping"); return; }
        let rt = Qwen3Runtime::load(M).expect("load");
        let mut state = rt.new_state();
        let logits = rt.forward_logits(&mut state, 1);
        assert_eq!(logits.len(), rt.cfg.vocab);
        for v in &logits { assert!(v.is_finite(), "non-finite logit"); }
        eprintln!("Q3.e OK: bonsai-4b-q1_0 loaded + forward — {} layers, d={}", rt.cfg.n_layers, rt.cfg.d_model);
    }

    /// **Q3.f — Bonsai-8B Q1_0 loads + one forward.** Largest of the
    /// pulled set; mainly verifies the load path handles ~1.2 GB of
    /// Q1_0 tensors without OOM/UB.
    #[test]
    fn q3f_bonsai_8b_q1_0_load_and_forward() {
        const M: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/bonsai-8b-q1_0.gguf";
        if !std::path::Path::new(M).exists() { eprintln!("Q3.f: model not mounted — skipping"); return; }
        let rt = Qwen3Runtime::load(M).expect("load");
        let mut state = rt.new_state();
        let logits = rt.forward_logits(&mut state, 1);
        assert_eq!(logits.len(), rt.cfg.vocab);
        for v in &logits { assert!(v.is_finite(), "non-finite logit"); }
        eprintln!("Q3.f OK: bonsai-8b-q1_0 loaded + forward — {} layers, d={}", rt.cfg.n_layers, rt.cfg.d_model);
    }

    /// **Q3.g — Ternary-Bonsai-4B Q2_0 loads + one forward.**
    #[test]
    fn q3g_ternary_bonsai_4b_q2_0_load_and_forward() {
        const M: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/ternary-bonsai-4b-q2_0.gguf";
        if !std::path::Path::new(M).exists() { eprintln!("Q3.g: model not mounted — skipping"); return; }
        let rt = Qwen3Runtime::load(M).expect("load");
        let mut state = rt.new_state();
        let logits = rt.forward_logits(&mut state, 1);
        assert_eq!(logits.len(), rt.cfg.vocab);
        for v in &logits { assert!(v.is_finite(), "non-finite logit"); }
        eprintln!("Q3.g OK: ternary-bonsai-4b-q2_0 loaded + forward — {} layers, d={}", rt.cfg.n_layers, rt.cfg.d_model);
    }

    /// **Q3.h — Ternary-Bonsai-8B Q2_0 loads + one forward.**
    #[test]
    fn q3h_ternary_bonsai_8b_q2_0_load_and_forward() {
        const M: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/ternary-bonsai-8b-q2_0.gguf";
        if !std::path::Path::new(M).exists() { eprintln!("Q3.h: model not mounted — skipping"); return; }
        let rt = Qwen3Runtime::load(M).expect("load");
        let mut state = rt.new_state();
        let logits = rt.forward_logits(&mut state, 1);
        assert_eq!(logits.len(), rt.cfg.vocab);
        for v in &logits { assert!(v.is_finite(), "non-finite logit"); }
        eprintln!("Q3.h OK: ternary-bonsai-8b-q2_0 loaded + forward — {} layers, d={}", rt.cfg.n_layers, rt.cfg.d_model);
    }

    /// **Q3.d — coherent text generation on Ternary-Bonsai Q2_0.**
    /// Mirror of Q3.c on the 2-bit quaternary codec — exercises the
    /// q=3 (+2d) branch that's specific to Q2_0 (a code-rotation bug
    /// would surface here but not on Q1_0).
    #[test]
    fn q3d_generate_ternary_bonsai_q2_0() {
        if !std::path::Path::new(MODEL_Q2).exists() { eprintln!("Q3.d: model not mounted — skipping"); return; }
        let rt = Qwen3Runtime::load(MODEL_Q2).expect("load");
        let cfg = Qwen3GenConfig { max_new_tokens: 16, sample: SampleConfig::GREEDY, rep_window: 64 };
        let prompt = "The capital of France is";
        let mut chunks: Vec<String> = Vec::new();
        let t0 = std::time::Instant::now();
        let _ = rt.generate(prompt, &cfg, |tok| chunks.push(tok.to_string())).expect("generate");
        let dt = t0.elapsed();
        let full = chunks.concat();
        assert!(!full.is_empty(), "empty");
        let coherent = [" the ", " a ", " of ", " is ", " in ", " and ", " to ", "Paris"]
            .iter().any(|w| full.contains(w))
            || full.chars().filter(|c| c.is_alphabetic()).count() >= full.len() / 2;
        assert!(coherent, "no common English: {full:?}");
        eprintln!(
            "Q3.d OK (Q2_0): prompt={prompt:?} → {full:?}  [{} tok in {dt:?} = {:.2} tok/s]",
            chunks.len(), chunks.len() as f64 / dt.as_secs_f64()
        );
    }
}
