//! # rce-qwen35 — Qwen 3.5 dense inference (0.8B / 2B / 4B)
//!
//! Qwen 3.5 dense checkpoints carry GGUF `general.architecture = qwen35`: the
//! *same* hybrid design as Qwen 3.6 — Gated DeltaNet linear-attention layers
//! interleaved with sigmoid-gated full-attention layers (every 4th per
//! `full_attention_interval`), multi-axis RoPE (`dimension_sections=[11,11,10,0]`,
//! `freq_base=1e7`). Because the arch tag and tensor layout are identical, the
//! forward pass in [`rce_qwen36`] serves the dense family verbatim; the
//! per-layer Gated-DeltaNet-vs-Gated-Attention split is auto-detected from
//! `ssm_a` tensor presence at load time, so no dims are hard-coded to the 3.6
//! shape. This crate is the dense-family entry point: it re-exports that runtime
//! under `Qwen35*` names and validates the dense 4B checkpoint end-to-end.
//!
//! Verified dense 4B dims (Q5_K_M GGUF): 33 layers, d_model=2560, d_ff=9216,
//! 16 Q / 4 KV attention heads, vocab from tokenizer.

pub use rce_qwen36::{
    Qwen36Config as Qwen35Config, Qwen36GenConfig as Qwen35GenConfig,
    Qwen36Runtime as Qwen35Runtime, Qwen36State as Qwen35State,
    ExportedSampleConfig as SampleConfig,
};

#[cfg(test)]
mod tests {
    use super::*;

    const ML: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library";
    const MODEL: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/qwen3.5-4b-gguf/Qwen_Qwen3.5-4B-Q5_K_M.gguf";

    /// Load a dense checkpoint, greedily continue "The capital of France is",
    /// and assert the continuation is readable English (rejects garbage byte
    /// streams). No per-size dim asserts — the same runtime serves 0.8B/2B/4B.
    fn assert_coherent(path: &str, tag: &str) {
        if !std::path::Path::new(path).exists() {
            eprintln!("{tag}: GGUF not mounted — skipping");
            return;
        }
        let rt = Qwen35Runtime::load(path).expect("load");
        let cfg = Qwen35GenConfig { max_new_tokens: 24, sample: SampleConfig::GREEDY, rep_window: 64 };
        let prompt = "The capital of France is";
        let mut chunks: Vec<String> = Vec::new();
        let t0 = std::time::Instant::now();
        rt.generate(prompt, &cfg, |t| { eprint!("{t}"); chunks.push(t.to_string()); }).expect("generate");
        let dt = t0.elapsed();
        let full = chunks.concat();
        eprintln!(
            "\n{tag} generate OK: {prompt:?} → {full:?}  [{} tok, {:.2} tok/s]",
            chunks.len(), chunks.len() as f64 / dt.as_secs_f64(),
        );
        assert!(!full.is_empty(), "{tag}: empty continuation");
        let alpha = full.chars().filter(|c| c.is_alphabetic()).count();
        let total = full.chars().count();
        assert!(alpha * 2 >= total, "{tag}: garbage (alpha {alpha}/{total}): {full:?}");
    }

    #[test]
    fn qwen35_0_8b_generate_proves_coherence() {
        assert_coherent(&format!("{ML}/qwen3.5-0.8b-gguf/Qwen_Qwen3.5-0.8B-Q5_K_M.gguf"), "qwen35-0.8b");
    }

    #[test]
    fn qwen35_2b_generate_proves_coherence() {
        assert_coherent(&format!("{ML}/qwen3.5-2b-gguf/Qwen_Qwen3.5-2B-Q5_K_M.gguf"), "qwen35-2b");
    }

    #[test]
    fn qwen35_9b_generate_proves_coherence() {
        assert_coherent(&format!("{ML}/qwen3.5-9b-gguf/Qwen_Qwen3.5-9B-Q5_K_M.gguf"), "qwen35-9b");
    }

    #[test]
    fn qwen35_4b_dims_sanity() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("qwen35: GGUF not mounted — skipping");
            return;
        }
        let rt = Qwen35Runtime::load(MODEL).expect("load Qwen 3.5-4B");
        let c = &rt.cfg;
        assert_eq!(c.d_model, 2560);
        assert_eq!(c.d_ff, 9216);
        // 33 GGUF blocks = 32 trunk layers + 1 MTP/nextn head (excluded).
        assert_eq!(c.n_layers, 32);
        assert_eq!(c.n_q_heads, 16);
        assert_eq!(c.n_kv_heads, 4);
        assert_eq!(c.rope_dimension_sections, [11, 11, 10, 0]);
    }

    #[test]
    fn qwen35_4b_generate_proves_coherence() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("qwen35: GGUF not mounted — skipping");
            return;
        }
        let rt = Qwen35Runtime::load(MODEL).expect("load");
        let cfg = Qwen35GenConfig {
            max_new_tokens: 24,
            sample: SampleConfig::GREEDY,
            rep_window: 64,
        };
        let prompt = "The capital of France is";
        let mut chunks: Vec<String> = Vec::new();
        let t0 = std::time::Instant::now();
        let _reply = rt
            .generate(prompt, &cfg, |tok| {
                eprint!("{tok}");
                chunks.push(tok.to_string());
            })
            .expect("generate");
        let dt = t0.elapsed();
        eprintln!();
        let full = chunks.concat();
        eprintln!(
            "qwen35-4b generate OK: prompt={prompt:?} → continuation={full:?}  [{} tok in {dt:?} = {:.2} tok/s]",
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
