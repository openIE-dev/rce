//! # rce-qwen35-moe — Qwen 3.5 MoE inference (35B-A3B / 122B-A10B / 397B-A17B)
//!
//! Qwen 3.5 MoE checkpoints carry GGUF `general.architecture = qwen35moe`: the
//! *same* design as Qwen 3.6-35B-A3B — hybrid Gated DeltaNet + Gated Attention
//! mixer, with every FFN replaced by a sparse-MoE block (routed experts under a
//! top-k softmax router + an always-on sigmoid-gated shared expert). Expert
//! counts, the per-layer DeltaNet/Attention split, tied embeddings, and trailing
//! MTP/`nextn` speculative heads are all read from GGUF metadata at load time, so
//! the forward pass in [`rce_qwen36_moe`] serves the whole MoE family verbatim.
//! This crate is the Qwen 3.5 MoE entry point: it re-exports that runtime under
//! `Qwen35Moe*` names and validates the checkpoints end-to-end.

pub use rce_qwen36_moe::{
    Qwen36MoeConfig as Qwen35MoeConfig, Qwen36MoeGenConfig as Qwen35MoeGenConfig,
    Qwen36MoeRuntime as Qwen35MoeRuntime, Qwen36MoeState as Qwen35MoeState,
    ExportedSampleConfig as SampleConfig,
};

#[cfg(test)]
mod tests {
    use super::*;

    const ML: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library";

    fn assert_coherent(path: &str, tag: &str) {
        if !std::path::Path::new(path).exists() {
            eprintln!("{tag}: GGUF not mounted — skipping");
            return;
        }
        let rt = Qwen35MoeRuntime::load(path).expect("load");
        let cfg = Qwen35MoeGenConfig { max_new_tokens: 24, sample: SampleConfig::GREEDY, rep_window: 64 };
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
    fn qwen35_35b_a3b_generate_proves_coherence() {
        assert_coherent(
            &format!("{ML}/qwen3.5-35b-a3b-gguf/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf"),
            "qwen35-35b-a3b",
        );
    }
}
