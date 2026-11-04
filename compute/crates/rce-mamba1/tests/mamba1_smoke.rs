//! Load the local Falcon3-Mamba-7B-Instruct GGUF, run one BOS forward,
//! sanity-check shape + finite-ness. Skips silently if the model file
//! isn't on this machine.

use rce_mamba1::Mamba1Runtime;

const PATH: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/falcon3-mamba-7b-gguf/Falcon3-Mamba-7B-Instruct.Q4_K_M.gguf";

#[test]
fn mamba1_load_and_forward_bos() {
    let path = std::path::Path::new(PATH);
    if !path.exists() {
        eprintln!("mamba1_load_and_forward_bos: GGUF not found — skipping.");
        return;
    }
    let rt = Mamba1Runtime::load(path).expect("load");
    let cfg = rt.config().clone();
    assert_eq!(cfg.d_model, 4096, "falcon3-mamba-7b d_model");
    assert_eq!(cfg.n_layers, 64, "falcon3-mamba-7b n_layers");
    assert_eq!(cfg.d_inner, 8192, "falcon3-mamba-7b d_inner");
    assert_eq!(cfg.d_state, 16, "falcon3-mamba-7b d_state");
    assert_eq!(cfg.dt_rank, 256, "falcon3-mamba-7b dt_rank");
    assert!(cfg.dt_b_c_rms, "falcon3-mamba sets dt_b_c_rms=true");

    let mut state = rt.new_state();
    // BOS for Falcon GPT-2-style tokenizer is id=8 per GGUF metadata.
    let logits = rt.forward_logits(&mut state, 8);
    assert_eq!(logits.len(), cfg.vocab, "logits vocab mismatch");
    assert!(logits.iter().all(|v| v.is_finite()), "non-finite logit");
}
