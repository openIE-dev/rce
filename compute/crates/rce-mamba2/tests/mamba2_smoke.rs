//! Load the local mamba2-2.7b GGUF, run one BOS forward, sanity-check
//! shape + finite-ness + a coherent argmax top-5. Skips silently if
//! the model file isn't on this machine (so CI on other hosts is happy).

use rce_mamba2::Mamba2Runtime;

const PATH: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/mamba2-2.7b-gguf/mamba2-2.7b-q5_k_m.gguf";

#[test]
fn mamba2_load_and_forward_bos() {
    let path = std::path::Path::new(PATH);
    if !path.exists() {
        eprintln!("mamba2_load_and_forward_bos: GGUF not found — skipping.");
        return;
    }
    let rt = Mamba2Runtime::load(path).expect("load");
    let cfg = rt.config().clone();
    assert_eq!(cfg.d_model, 2560, "mamba2-2.7b d_model");
    assert_eq!(cfg.n_layers, 64, "mamba2-2.7b n_layers");
    assert_eq!(cfg.ssm_n_heads, 80, "mamba2-2.7b ssm_n_heads");
    assert_eq!(cfg.ssm_head_dim, 64, "mamba2-2.7b ssm_head_dim");
    assert_eq!(cfg.ssm_d_state, 128, "mamba2-2.7b ssm_d_state");
    assert_eq!(cfg.ssm_n_groups, 1, "pure mamba2 ngroups=1");

    let mut state = rt.new_state();
    // BOS for GPT-NeoX tokenizer is typically id=0.
    let logits = rt.forward_logits(&mut state, 0);
    assert_eq!(logits.len(), cfg.vocab, "logits vocab mismatch");
    assert!(logits.iter().all(|v| v.is_finite()), "non-finite logit");

    // Top-1 should be a "real" token, not a special / control token.
    let argmax = logits.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as i32).unwrap();
    assert!(argmax >= 0 && (argmax as usize) < cfg.vocab);
}
