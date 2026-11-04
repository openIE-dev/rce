//! Smoke test: load Nemotron-Nano-4B and run a single forward pass on
//! BOS to validate that block-type detection, weight loading, and the
//! Mamba2 / RoPE-GQA / dense-FFN block forwards all wire up. Skips if
//! the GGUF isn't on the external drive.

use rce_nemotron::NemotronRuntime;

const PATH: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/nemotron-nano-raw/models--unsloth--NVIDIA-Nemotron-3-Nano-4B-GGUF/snapshots/8e81be55c5aa3d63bb82b6ceec62d50805d9e1bb/NVIDIA-Nemotron-3-Nano-4B-Q4_0.gguf";

#[test]
fn nemotron_load_and_forward_bos() {
    let path = std::path::Path::new(PATH);
    if !path.exists() {
        eprintln!("nemotron_load_and_forward_bos: GGUF not found — skipping.");
        return;
    }
    let t0 = std::time::Instant::now();
    let rt = NemotronRuntime::load(path).expect("load");
    eprintln!("loaded in {:?}", t0.elapsed());
    let kinds = rt.block_kinds();
    let n_ssm = kinds.iter().filter(|k| matches!(k, rce_nemotron::BlockKind::Ssm)).count();
    let n_attn = kinds.iter().filter(|k| matches!(k, rce_nemotron::BlockKind::Attn)).count();
    let n_ffn = kinds.iter().filter(|k| matches!(k, rce_nemotron::BlockKind::Ffn)).count();
    eprintln!("block kinds: {n_ssm} ssm + {n_attn} attn + {n_ffn} ffn = {} (config n_layers={})",
        n_ssm + n_attn + n_ffn, rt.config().n_layers);
    assert_eq!(n_ssm + n_attn + n_ffn, rt.config().n_layers);

    let mut state = rt.new_state();
    let t1 = std::time::Instant::now();
    let logits = rt.forward_logits(&mut state, 1);
    eprintln!("forward(BOS) in {:?}, logits.len={}", t1.elapsed(), logits.len());
    assert_eq!(logits.len(), rt.config().vocab);
    let any_finite = logits.iter().any(|v| v.is_finite());
    let any_nan = logits.iter().any(|v| v.is_nan());
    let any_inf = logits.iter().any(|v| v.is_infinite());
    assert!(any_finite, "all logits are non-finite");
    assert!(!any_nan, "logits contain NaN");
    assert!(!any_inf, "logits contain Inf");

    // Print top-10 to show the model is alive.
    let tk = rt.tokenizer();
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("Top-10 after BOS:");
    for &(id, lg) in indexed.iter().take(10) {
        eprintln!("  {id:6} logit={lg:9.4} text={:?}", tk.decode_sequence(&[id as i32], false));
    }
}
