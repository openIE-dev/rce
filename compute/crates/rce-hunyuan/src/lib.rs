//! Hunyuan-dense inference runtime on the rce substrate. Plain
//! Llama-2-style transformer + **per-head QK-norm applied AFTER RoPE**
//! (the one structural divergence from qwen3 — rce-qwen3 norms Q/K
//! BEFORE RoPE; hunyuan-dense norms AFTER) + NeoX RoPE + tied lm_head.
//! Mirrors the vendored `llm_build_hunyuan` (sjl623/llama.cpp `STQ_0`,
//! `src/models/hunyuan-vl.cpp` — dense reuses the VL graph):
//!
//! ```text
//!   inpSA = x
//!   cur   = rms_norm(x, attn_norm)
//!   Q,K,V = wq(cur), wk(cur), wv(cur)              (no QKV bias)
//!   Q,K   = rope_neox(Q, K, pos)                   ← RoPE FIRST
//!   Q     = rms_norm_per_head(Q, attn_q_norm)      ← QK-norm AFTER RoPE
//!   K     = rms_norm_per_head(K, attn_k_norm)
//!   cur   = softmax(QK^T/√d) · V                   GQA (16q / 4kv on 1.8B)
//!   cur   = wo(cur)
//!   x     = inpSA + cur
//!   ffn   = silu(gate(rms_norm(x, ffn_norm))) * up(...) → down(...)
//!   x     = x + ffn
//! ```
//! Final: `cur = rms_norm(x, output_norm)` → `logits = token_embd · cur`
//! (tied lm_head; token_embd is Q6K, body matmuls are STQ1_0 "Sherry").

use rce_ingest_gguf::{
    read_tensor_f32, sample_with_history, GgufFile, GgufValue, SampleConfig, Tokenizer,
};
use rce_op_qmm::QuantWeight;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct HunyuanConfig {
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

impl HunyuanConfig {
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
        if arch != "hunyuan-dense" {
            return Err(format!("expected hunyuan-dense, got {arch}"));
        }
        let u32_meta = |k: &str| -> Option<u32> {
            gguf.metadata
                .iter()
                .find(|kv| kv.key == format!("hunyuan-dense.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::U32(v) => Some(*v), _ => None })
        };
        let f32_meta = |k: &str| -> Option<f32> {
            gguf.metadata
                .iter()
                .find(|kv| kv.key == format!("hunyuan-dense.{k}"))
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
        // hunyuan-dense stores per-head dim explicitly via key_length;
        // fall back to the q_norm tensor dim, then d_model/n_heads.
        let d_head = u32_meta("attention.key_length")
            .map(|v| v as usize)
            .or_else(|| gguf.tensors.iter()
                .find(|t| t.name == "blk.0.attn_q_norm.weight")
                .and_then(|t| t.dims.first().copied())
                .map(|d| d as usize))
            .unwrap_or(d_model / n_q_heads);
        let rope_dim = u32_meta("rope.dimension_count").unwrap_or(d_head as u32) as usize;
        let rope_freq_base = f32_meta("rope.freq_base").unwrap_or(10_000.0);
        let d_ff = u32_meta("feed_forward_length").ok_or("feed_forward_length")? as usize;
        let context_length = u32_meta("context_length").unwrap_or(8192) as usize;
        let eps = f32_meta("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        Ok(HunyuanConfig {
            d_model, vocab, n_layers, n_q_heads, n_kv_heads, d_head,
            rope_dim, rope_freq_base, d_ff, context_length, eps,
        })
    }
    #[inline] pub fn n_embd_q(&self) -> usize { self.n_q_heads * self.d_head }
    #[inline] pub fn n_embd_kv(&self) -> usize { self.n_kv_heads * self.d_head }
}

pub struct HunyuanBlock {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub q_norm: Vec<f32>,           // per-head, dim = d_head
    pub k_norm: Vec<f32>,           // per-head, dim = d_head
    pub wq: QuantWeight, pub wk: QuantWeight, pub wv: QuantWeight, pub wo: QuantWeight,
    pub ffn_gate: QuantWeight, pub ffn_up: QuantWeight, pub ffn_down: QuantWeight,
}

pub struct HunyuanWeights {
    pub token_embd: QuantWeight,    // doubles as lm_head (tied); Q6K
    pub output_norm: Vec<f32>,
    pub blocks: Vec<HunyuanBlock>,
}

pub fn load_weights(
    mmap: &Arc<Mmap>, gguf: &GgufFile, cfg: &HunyuanConfig,
) -> Result<HunyuanWeights, String> {
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
        blocks.push(HunyuanBlock {
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
    Ok(HunyuanWeights { token_embd, output_norm, blocks })
}

pub struct BlockState { pub k_cache: Vec<f32>, pub v_cache: Vec<f32> }
pub struct HunyuanState { pub blocks: Vec<BlockState>, pub pos: usize }
impl HunyuanState {
    pub fn new(cfg: &HunyuanConfig) -> Self {
        let cap = cfg.context_length.min(8192) * cfg.n_embd_kv();
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

/// Per-head RMS norm: split into `n_heads` slices of `d_head`,
/// normalize each independently with shared `w` (len `d_head`).
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
    let mut out = vec![0.0f32; w.n_rows];
    rce_op_qmm::mm_1d(a, w, &mut out);
    out
}

/// NeoX RoPE — rotate first `rope_dim` dims of each head using the
/// `i / i+half` pair convention.
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
    x_in: &[f32], blk: &HunyuanBlock, cfg: &HunyuanConfig,
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

    // *** hunyuan-dense order *** — RoPE FIRST, then per-head QK-norm
    // (the divergence from qwen3, per vendored llm_build_hunyuan in
    // sjl623/llama.cpp `src/models/hunyuan-vl.cpp`).
    apply_rope_inplace(&mut q, nq, dh, cfg.rope_dim, cfg.rope_freq_base, pos);
    apply_rope_inplace(&mut k_new, nkv, dh, cfg.rope_dim, cfg.rope_freq_base, pos);
    let q = rms_norm_per_head(&q, &blk.q_norm, nq, dh, cfg.eps);
    let k_new = rms_norm_per_head(&k_new, &blk.k_norm, nkv, dh, cfg.eps);

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
    w: &HunyuanWeights, cfg: &HunyuanConfig, state: &mut HunyuanState, token_id: i32,
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

pub struct HunyuanRuntime {
    pub cfg: HunyuanConfig,
    pub gguf: GgufFile,
    pub weights: HunyuanWeights,
    _mmap: Arc<Mmap>,
}

impl HunyuanRuntime {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| format!("open: {e}"))?;
        let mmap = Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?);
        // Hint sequential reads during the load-time weight scan; ignore
        // errors (kernel may refuse). See rce_ingest_gguf::mmap_load_sequential.
        let _ = mmap.advise(memmap2::Advice::WillNeed);
        let gguf = GgufFile::parse(&mmap[..]).map_err(|e| format!("parse: {e:?}"))?;
        let cfg = HunyuanConfig::from_gguf(&gguf)?;
        eprintln!(
            "hunyuan: d_model={} vocab={} n_layers={} n_heads={}/{} d_head={} d_ff={} ctx={} rope={}",
            cfg.d_model, cfg.vocab, cfg.n_layers, cfg.n_q_heads, cfg.n_kv_heads,
            cfg.d_head, cfg.d_ff, cfg.context_length, cfg.rope_freq_base,
        );
        let weights = load_weights(&mmap, &gguf, &cfg)?;
        Ok(Self { cfg, gguf, weights, _mmap: mmap })
    }
    pub fn new_state(&self) -> HunyuanState { HunyuanState::new(&self.cfg) }
    pub fn forward_logits(&self, state: &mut HunyuanState, token_id: i32) -> Vec<f32> {
        forward_token(&self.weights, &self.cfg, state, token_id)
    }
    pub fn tokenizer(&self) -> Tokenizer<'_> { Tokenizer::from_gguf(&self.gguf).expect("tokenizer") }
}

#[derive(Clone, Copy, Debug)]
pub struct HunyuanGenConfig {
    pub max_new_tokens: usize,
    pub sample: SampleConfig,
    pub rep_window: usize,
}
impl Default for HunyuanGenConfig {
    fn default() -> Self {
        Self { max_new_tokens: 32, sample: SampleConfig::chat(0x7111_7AAA), rep_window: 64 }
    }
}

impl HunyuanRuntime {
    /// Build the HY-MT chat-framed token sequence for a single user
    /// turn: `<bos><hy_User>{text}<hy_Assistant>` (per the GGUF
    /// `tokenizer.chat_template`). HY-MT is an instruction/translation
    /// model — a raw fragment yields immediate-EOS; this framing is
    /// what makes generation meaningful.
    pub fn frame_user_turn(&self, text: &str) -> Result<Vec<i32>, String> {
        let tk = self.tokenizer();
        let bos = tk.lookup_token("<｜hy_begin▁of▁sentence｜>")
            .or(tk.bos_token_id)
            .ok_or("no HY bos token")?;
        let user = tk.lookup_token("<｜hy_User｜>").ok_or("no <hy_User> token")?;
        let asst = tk.lookup_token("<｜hy_Assistant｜>").ok_or("no <hy_Assistant> token")?;
        let body = tk.encode_gpt2(text, false);
        if body.is_empty() { return Err("empty prompt body".into()); }
        let mut ids = Vec::with_capacity(body.len() + 3);
        ids.push(bos);
        ids.push(user);
        ids.extend_from_slice(&body);
        ids.push(asst);
        Ok(ids)
    }

    /// Generate from an explicit token sequence (e.g. from
    /// [`Self::frame_user_turn`]). Greedy/sampled per `gcfg`.
    pub fn generate_from_ids(
        &self, prompt_ids: &[i32], gcfg: &HunyuanGenConfig, mut on_token: impl FnMut(&str),
    ) -> Result<String, String> {
        let tk = self.tokenizer();
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

    pub fn generate(
        &self, prompt: &str, gcfg: &HunyuanGenConfig, mut on_token: impl FnMut(&str),
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
    const MODEL: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/tencent-hy-mt1.5-1.8b-1.25bit.gguf";

    /// **H.a — load real Tencent HY-MT1.5 (STQ1_0) + shape sanity.**
    /// Confirms the arch-aware tag-42→Stq1_0 remap fired and every
    /// weight slot maps cleanly (token_embd Q6K, body STQ1_0).
    #[test]
    fn ha_load_tencent_stq1_0() {
        if !std::path::Path::new(MODEL).exists() { eprintln!("H.a: model not mounted — skipping"); return; }
        let rt = HunyuanRuntime::load(MODEL).expect("load");
        let c = &rt.cfg;
        assert_eq!(c.d_model, 2048);
        assert_eq!(c.n_q_heads, 16);
        assert_eq!(c.n_kv_heads, 4);
        assert_eq!(c.d_head, 128);
        assert_eq!(c.d_ff, 6144);
        assert_eq!(c.n_layers, 32);
        assert_eq!(rt.weights.token_embd.n_rows, c.vocab);
        assert_eq!(rt.weights.blocks.len(), c.n_layers);
        eprintln!("H.a OK: tencent-hy-mt1.5 loaded — {} layers, shapes match", c.n_layers);
    }

    /// **H.b — one forward → finite logits.** Smoke test of the full
    /// hunyuan-dense forward (QK-norm-after-RoPE order) over the
    /// STQ1_0 body + Q6K tied lm_head.
    #[test]
    fn hb_one_forward_tencent() {
        if !std::path::Path::new(MODEL).exists() { eprintln!("H.b: model not mounted — skipping"); return; }
        let rt = HunyuanRuntime::load(MODEL).expect("load");
        let mut state = rt.new_state();
        let logits = rt.forward_logits(&mut state, 1);
        assert_eq!(logits.len(), rt.cfg.vocab);
        for v in &logits { assert!(v.is_finite(), "non-finite logit"); }
        let (argmax, _) = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        assert!(argmax > 0 && argmax < rt.cfg.vocab - 1);
        eprintln!("H.b OK: one forward → argmax={argmax}, max|logit|={:.3}",
            logits.iter().fold(0.0f32, |a, v| a.max(v.abs())));
    }

    /// **H.c — real translation on Tencent HY-MT1.5 (STQ1_0).** HY-MT
    /// is an instruction/MT model: it needs its chat framing
    /// (`<bos><hy_User>…<hy_Assistant>`), and a raw fragment yields
    /// immediate-EOS (verified during bring-up — NOT a forward bug).
    /// This gate frames a real English→French translation request and
    /// asserts the output contains the expected French. A codec/codebook
    /// or arch (QK-norm-after-RoPE) error would garble this and fail.
    #[test]
    fn hc_translate_tencent() {
        if !std::path::Path::new(MODEL).exists() { eprintln!("H.c: model not mounted — skipping"); return; }
        let rt = HunyuanRuntime::load(MODEL).expect("load");
        let cfg = HunyuanGenConfig { max_new_tokens: 24, sample: SampleConfig::GREEDY, rep_window: 64 };
        let ids = rt.frame_user_turn(
            "Translate the following into French:\nThe capital of France is Paris."
        ).expect("frame");
        let mut chunks: Vec<String> = Vec::new();
        let t0 = std::time::Instant::now();
        let _ = rt.generate_from_ids(&ids, &cfg, |tok| chunks.push(tok.to_string())).expect("generate");
        let dt = t0.elapsed();
        let full = chunks.concat();
        assert!(!full.is_empty(), "empty generation");
        let low = full.to_lowercase();
        // Correct French is "La capitale de la France est Paris."
        let translated = low.contains("capitale") && low.contains("france") && low.contains("paris");
        assert!(translated, "expected French translation, got: {full:?}");
        eprintln!(
            "H.c OK (STQ1_0 / hunyuan-dense): EN→FR → {full:?}  [{} tok in {dt:?} = {:.2} tok/s]",
            chunks.len(), chunks.len() as f64 / dt.as_secs_f64()
        );
    }
}
