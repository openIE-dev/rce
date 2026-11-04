//! EuroBERT retrieval-embedding runtime on the rce substrate.
//!
//! rce's **first encoder model** — every prior runtime (LFM2, BitNet,
//! TinyLlama, Qwen3, Hunyuan) is a causal decoder. EuroBERT is a
//! "modern BERT": a Llama-style pre-norm block (RoPE + RMSNorm + SwiGLU,
//! no biases) but **bidirectional** — `build_attn_inp_no_cache` in the
//! vendored `llm_build_eurobert` (sjl623/llama.cpp `src/models/
//! eurobert.cpp`): no causal mask, no KV cache, every token attends to
//! every token. The output is not logits but the final-norm hidden
//! states; a single sentence embedding is produced by **LAST-token
//! pooling** (`eurobert.pooling_type == 3`) then L2-normalisation.
//!
//! Per layer (pre-norm, residual on the pre-norm input):
//! ```text
//!   cur     = rms_norm(inpL, attn_norm)
//!   Q,K,V   = wq(cur), wk(cur), wv(cur)            (no bias)
//!   Q,K     = rope_neox(Q, K, pos)                 freq_base 1e6
//!   cur     = softmax(QKᵀ/√d)·V                    BIDIRECTIONAL (no mask)
//!   cur     = wo(cur)
//!   inpL    = cur + inpL                            (post-attn residual)
//!   cur     = rms_norm(inpL, ffn_norm)
//!   cur     = down(silu(gate(cur)) * up(cur))       SwiGLU
//!   inpL    = cur + inpL                            (post-ffn residual)
//! ```
//! Final: `h = rms_norm(inpL, output_norm)` → pool LAST → L2-normalise.
//!
//! **Matryoshka (MRL):** Jina v5 embeddings are trained so any prefix
//! of the 768-d vector is itself a usable embedding. [`Self::embed_mrl`]
//! truncates to the first `dims` and re-normalises — no extra tensors.

use rce_ingest_gguf::{read_tensor_f32, GgufFile, GgufValue, Tokenizer};
use rce_op_qmm::QuantWeight;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct EuroBertConfig {
    pub d_model: usize,
    pub vocab: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_head: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub d_ff: usize,
    pub context_length: usize,
    pub eps: f32,
    /// llama.cpp pooling enum: 1=MEAN, 2=CLS, 3=LAST. Jina v5 = 3.
    pub pooling_type: u32,
}

impl EuroBertConfig {
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
        if arch != "eurobert" {
            return Err(format!("expected eurobert, got {arch}"));
        }
        let u32m = |k: &str| -> Option<u32> {
            gguf.metadata.iter()
                .find(|kv| kv.key == format!("eurobert.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::U32(v) => Some(*v), _ => None })
        };
        let f32m = |k: &str| -> Option<f32> {
            gguf.metadata.iter()
                .find(|kv| kv.key == format!("eurobert.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::F32(v) => Some(*v), _ => None })
        };
        let d_model = u32m("embedding_length").ok_or("embedding_length")? as usize;
        let vocab = u32m("vocab_size")
            .or_else(|| gguf.tensors.iter()
                .find(|t| t.name == "token_embd.weight")
                .and_then(|t| t.dims.get(1).copied())
                .map(|d| d as u32))
            .ok_or("vocab_size")? as usize;
        let n_layers = u32m("block_count").ok_or("block_count")? as usize;
        let n_heads = u32m("attention.head_count").ok_or("attention.head_count")? as usize;
        let d_head = u32m("attention.value_length")
            .map(|v| v as usize)
            .unwrap_or(d_model / n_heads);
        let rope_dim = u32m("rope.dimension_count").unwrap_or(d_head as u32) as usize;
        let rope_freq_base = f32m("rope.freq_base").unwrap_or(1_000_000.0);
        let d_ff = u32m("feed_forward_length").ok_or("feed_forward_length")? as usize;
        let context_length = u32m("context_length").unwrap_or(8192) as usize;
        let eps = f32m("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let pooling_type = u32m("pooling_type").unwrap_or(3);
        Ok(EuroBertConfig {
            d_model, vocab, n_layers, n_heads, d_head, rope_dim,
            rope_freq_base, d_ff, context_length, eps, pooling_type,
        })
    }
}

pub struct EuroBertBlock {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub wq: QuantWeight, pub wk: QuantWeight, pub wv: QuantWeight, pub wo: QuantWeight,
    pub ffn_gate: QuantWeight, pub ffn_up: QuantWeight, pub ffn_down: QuantWeight,
}

pub struct EuroBertWeights {
    pub token_embd: QuantWeight,
    pub output_norm: Vec<f32>,
    pub blocks: Vec<EuroBertBlock>,
}

pub fn load_weights(
    mmap: &Arc<Mmap>, gguf: &GgufFile, cfg: &EuroBertConfig,
) -> Result<EuroBertWeights, String> {
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
        blocks.push(EuroBertBlock {
            attn_norm: f32n("attn_norm")?,
            ffn_norm: f32n("ffn_norm")?,
            wq: qw("attn_q", cfg.d_model, cfg.d_model)?,
            wk: qw("attn_k", cfg.d_model, cfg.d_model)?,
            wv: qw("attn_v", cfg.d_model, cfg.d_model)?,
            wo: qw("attn_output", cfg.d_model, cfg.d_model)?,
            ffn_gate: qw("ffn_gate", cfg.d_ff, cfg.d_model)?,
            ffn_up: qw("ffn_up", cfg.d_ff, cfg.d_model)?,
            ffn_down: qw("ffn_down", cfg.d_model, cfg.d_ff)?,
        });
    }
    Ok(EuroBertWeights { token_embd, output_norm, blocks })
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
    let mut out = vec![0.0f32; w.n_rows];
    rce_op_qmm::mm_1d(a, w, &mut out);
    out
}

/// NeoX RoPE at absolute position `pos`, in place over `rope_dim`.
/// NEON-vectorized via rce-op-attn; numerics match scalar to 1e-5.
fn apply_rope_inplace(
    x: &mut [f32], n_heads: usize, d_head: usize, rope_dim: usize, base: f32, pos: usize,
) {
    for h in 0..n_heads {
        let off = h * d_head;
        rce_op_attn::rope_inplace(&mut x[off..off + d_head], rope_dim, base, pos);
    }
}

pub struct EuroBertRuntime {
    pub cfg: EuroBertConfig,
    pub gguf: GgufFile,
    pub weights: EuroBertWeights,
    _mmap: Arc<Mmap>,
}

impl EuroBertRuntime {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| format!("open: {e}"))?;
        let mmap = Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?);
        // Hint sequential reads during the load-time weight scan; ignore
        // errors (kernel may refuse). See rce_ingest_gguf::mmap_load_sequential.
        let _ = mmap.advise(memmap2::Advice::WillNeed);
        let gguf = GgufFile::parse(&mmap[..]).map_err(|e| format!("parse: {e:?}"))?;
        let cfg = EuroBertConfig::from_gguf(&gguf)?;
        eprintln!(
            "eurobert: d_model={} vocab={} n_layers={} n_heads={} d_head={} d_ff={} ctx={} rope={} pool={}",
            cfg.d_model, cfg.vocab, cfg.n_layers, cfg.n_heads, cfg.d_head,
            cfg.d_ff, cfg.context_length, cfg.rope_freq_base, cfg.pooling_type,
        );
        let weights = load_weights(&mmap, &gguf, &cfg)?;
        Ok(Self { cfg, gguf, weights, _mmap: mmap })
    }

    pub fn tokenizer(&self) -> Tokenizer<'_> {
        Tokenizer::from_gguf(&self.gguf).expect("tokenizer")
    }

    fn token_ids(&self, text: &str) -> Vec<i32> {
        let tk = self.tokenizer();
        let is_gpt2 = self.gguf.metadata.iter()
            .find(|kv| kv.key == "tokenizer.ggml.model")
            .map(|kv| matches!(&kv.value, GgufValue::String(s) if s == "gpt2"))
            .unwrap_or(true);
        let mut ids = if is_gpt2 { tk.encode_gpt2(text, false) } else { tk.encode(text, false) };
        // EuroBERT/jina-v5: add_eos_token = true → append EOS as the
        // sentinel whose hidden state LAST-pooling reads.
        let add_eos = self.gguf.metadata.iter()
            .find(|kv| kv.key == "tokenizer.ggml.add_eos_token")
            .map(|kv| matches!(&kv.value, GgufValue::Bool(b) if *b))
            .unwrap_or(true);
        if add_eos {
            if let Some(eos) = tk.eos_token_id {
                ids.push(eos);
            }
        }
        ids
    }

    /// Full-length L2-normalised sentence embedding (`d_model` dims).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let ids = self.token_ids(text);
        if ids.is_empty() { return Err("empty input".into()); }
        let v = self.encode_pooled(&ids);
        Ok(l2_normalize(v))
    }

    /// Matryoshka embedding: the first `dims` of the (un-normalised)
    /// pooled vector, then L2-normalised. Jina v5 is MRL-trained, so a
    /// prefix is a valid (smaller) embedding. `dims == 0` or `>= d_model`
    /// returns the full embedding.
    pub fn embed_mrl(&self, text: &str, dims: usize) -> Result<Vec<f32>, String> {
        let ids = self.token_ids(text);
        if ids.is_empty() { return Err("empty input".into()); }
        let mut v = self.encode_pooled(&ids);
        if dims > 0 && dims < v.len() {
            v.truncate(dims);
        }
        Ok(l2_normalize(v))
    }

    /// Bidirectional encode of the whole token sequence, then pool to a
    /// single (un-normalised) vector per `cfg.pooling_type`.
    fn encode_pooled(&self, ids: &[i32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let w = &self.weights;
        let dm = cfg.d_model;
        let dh = cfg.d_head;
        let nh = cfg.n_heads;
        let seq = ids.len();

        // Token embeddings → residual stream [seq][d_model].
        let mut hs: Vec<Vec<f32>> = ids.iter()
            .map(|&t| w.token_embd.dequant_row(t as usize))
            .collect();

        for blk in &w.blocks {
            // ── Attention sub-layer (pre-norm, bidirectional) ──
            let normed: Vec<Vec<f32>> = hs.iter()
                .map(|x| rms_norm(x, &blk.attn_norm, cfg.eps))
                .collect();
            // Project Q/K/V for every position; RoPE by absolute pos.
            let mut qs = Vec::with_capacity(seq);
            let mut ks = Vec::with_capacity(seq);
            let mut vs = Vec::with_capacity(seq);
            for (pos, n) in normed.iter().enumerate() {
                let mut q = qmm_1d(n, &blk.wq);
                let mut k = qmm_1d(n, &blk.wk);
                let v = qmm_1d(n, &blk.wv);
                apply_rope_inplace(&mut q, nh, dh, cfg.rope_dim, cfg.rope_freq_base, pos);
                apply_rope_inplace(&mut k, nh, dh, cfg.rope_dim, cfg.rope_freq_base, pos);
                qs.push(q); ks.push(k); vs.push(v);
            }
            let scale = 1.0 / (dh as f32).sqrt();
            // Bidirectional: every query i attends over all keys j.
            let mut attn_out = vec![vec![0.0f32; dm]; seq];
            for i in 0..seq {
                for h in 0..nh {
                    let qh = &qs[i][h * dh..(h + 1) * dh];
                    let mut scores = vec![0.0f32; seq];
                    for j in 0..seq {
                        let kh = &ks[j][h * dh..(h + 1) * dh];
                        let mut d = 0.0f32;
                        for t in 0..dh { d += qh[t] * kh[t]; }
                        scores[j] = d * scale;
                    }
                    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for s in scores.iter_mut() { *s = (*s - m).exp(); sum += *s; }
                    let o = h * dh;
                    for j in 0..seq {
                        let wj = scores[j] / sum;
                        let vh = &vs[j][o..o + dh];
                        for t in 0..dh { attn_out[i][o + t] += wj * vh[t]; }
                    }
                }
            }
            for i in 0..seq {
                let proj = qmm_1d(&attn_out[i], &blk.wo);
                for d in 0..dm { hs[i][d] += proj[d]; }
            }
            // ── FFN sub-layer (pre-norm, SwiGLU) ──
            for i in 0..seq {
                let n = rms_norm(&hs[i], &blk.ffn_norm, cfg.eps);
                let g = qmm_1d(&n, &blk.ffn_gate);
                let u = qmm_1d(&n, &blk.ffn_up);
                let mut hmid = vec![0.0f32; cfg.d_ff];
                for k in 0..cfg.d_ff { hmid[k] = silu(g[k]) * u[k]; }
                let down = qmm_1d(&hmid, &blk.ffn_down);
                for d in 0..dm { hs[i][d] += down[d]; }
            }
        }

        // Final norm, then pool.
        let pooled = match cfg.pooling_type {
            1 => {
                // MEAN over tokens (of final-normed states).
                let mut acc = vec![0.0f32; dm];
                for x in &hs {
                    let n = rms_norm(x, &w.output_norm, cfg.eps);
                    for d in 0..dm { acc[d] += n[d]; }
                }
                for d in 0..dm { acc[d] /= seq as f32; }
                acc
            }
            2 => rms_norm(&hs[0], &w.output_norm, cfg.eps),          // CLS = first
            _ => rms_norm(&hs[seq - 1], &w.output_norm, cfg.eps),    // 3 = LAST (default)
        };
        pooled
    }
}

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() { *x /= norm; }
    }
    v
}

/// Cosine similarity of two equal-length vectors (both L2-normalised
/// embeddings → this is just the dot product, but compute it robustly).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 { return 0.0; }
    dot / (na.sqrt() * nb.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    const MODEL: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/jina-v5-omni-nano-retrieval-q4_k_m.gguf";

    /// **EB.a — load + shape sanity.**
    #[test]
    fn eb_a_load() {
        if !std::path::Path::new(MODEL).exists() { eprintln!("EB.a: skip"); return; }
        let rt = EuroBertRuntime::load(MODEL).expect("load");
        let c = &rt.cfg;
        assert_eq!(c.d_model, 768);
        assert_eq!(c.n_layers, 12);
        assert_eq!(c.n_heads, 12);
        assert_eq!(c.d_head, 64);
        assert_eq!(c.d_ff, 3072);
        assert_eq!(c.pooling_type, 3);
        assert_eq!(rt.weights.blocks.len(), 12);
        eprintln!("EB.a OK: jina-v5-omni-nano loaded — 12-layer bidirectional encoder");
    }

    /// **EB.b — one embedding is finite + unit-norm.**
    #[test]
    fn eb_b_embed_finite_unit() {
        if !std::path::Path::new(MODEL).exists() { eprintln!("EB.b: skip"); return; }
        let rt = EuroBertRuntime::load(MODEL).expect("load");
        let e = rt.embed("The capital of France is Paris.").expect("embed");
        assert_eq!(e.len(), 768);
        for v in &e { assert!(v.is_finite(), "non-finite component"); }
        let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "not unit norm: {norm}");
        eprintln!("EB.b OK: 768-d embedding, ‖e‖={norm:.6}");
    }

    /// **EB.c — retrieval semantics (the meaningful correctness gate).**
    /// A working embedding model must rank a paraphrase strictly above
    /// an unrelated sentence by cosine similarity. This is the encoder
    /// analogue of the decoders' "coherent generation" gate — a codec,
    /// RoPE, or bidirectional-attention bug would collapse the ranking.
    #[test]
    fn eb_c_retrieval_semantics() {
        if !std::path::Path::new(MODEL).exists() { eprintln!("EB.c: skip"); return; }
        let rt = EuroBertRuntime::load(MODEL).expect("load");
        let t0 = std::time::Instant::now();
        let q = rt.embed("The capital of France is Paris.").expect("q");
        let para = rt.embed("Paris is the capital city of France.").expect("para");
        let unrel = rt.embed("I enjoy eating pizza on the weekend.").expect("unrel");
        let dt = t0.elapsed();
        let s_para = cosine(&q, &para);
        let s_unrel = cosine(&q, &unrel);
        eprintln!(
            "EB.c: cos(q,paraphrase)={s_para:.4}  cos(q,unrelated)={s_unrel:.4}  \
             margin={:.4}  [3 embeds in {dt:?}]",
            s_para - s_unrel
        );
        assert!(
            s_para > s_unrel + 0.05,
            "retrieval ranking collapsed: paraphrase {s_para:.4} !> unrelated {s_unrel:.4}"
        );
    }

    /// **EB.d — Matryoshka (MRL) property.** Truncating to a prefix
    /// (256, 128, 64) must preserve the paraphrase>unrelated ranking —
    /// the defining promise of an MRL-trained embedding.
    #[test]
    fn eb_d_matryoshka() {
        if !std::path::Path::new(MODEL).exists() { eprintln!("EB.d: skip"); return; }
        let rt = EuroBertRuntime::load(MODEL).expect("load");
        let texts = (
            "The capital of France is Paris.",
            "Paris is the capital city of France.",
            "I enjoy eating pizza on the weekend.",
        );
        for &dims in &[768usize, 256, 128, 64] {
            let q = rt.embed_mrl(texts.0, dims).expect("q");
            let para = rt.embed_mrl(texts.1, dims).expect("para");
            let unrel = rt.embed_mrl(texts.2, dims).expect("unrel");
            assert_eq!(q.len(), dims.min(768));
            let sp = cosine(&q, &para);
            let su = cosine(&q, &unrel);
            eprintln!("EB.d dims={dims:>3}: cos(para)={sp:.4} cos(unrel)={su:.4} margin={:.4}", sp - su);
            assert!(
                sp > su + 0.03,
                "MRL@{dims}: ranking collapsed (para {sp:.4} !> unrel {su:.4})"
            );
        }
        eprintln!("EB.d OK: Matryoshka prefixes 768/256/128/64 all preserve ranking");
    }
}
