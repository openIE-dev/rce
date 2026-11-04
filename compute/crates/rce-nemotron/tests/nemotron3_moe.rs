//! Nemotron 3 Nano 30B-A3B (`nemotron_h_moe`) end-to-end: load, verify
//! the 6 attn + 23 ssm + 23 moe block split, run a finite forward, and
//! greedily complete a prompt to prove coherence. Skips if the GGUF
//! isn't mounted.

use rce_ingest_gguf::{sample_with_history, SampleConfig};
use rce_nemotron::{BlockKind, NemotronRuntime};

const PATH: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/nemotron-3-nano-Q4/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf";

#[test]
fn nemotron3_moe_load_and_block_split() {
    let path = std::path::Path::new(PATH);
    if !path.exists() {
        eprintln!("nemotron3_moe: GGUF not mounted — skipping.");
        return;
    }
    let rt = NemotronRuntime::load(path).expect("load Nemotron 3 Nano 30B-A3B");
    let c = rt.config();
    assert_eq!(c.d_model, 2688);
    assert_eq!(c.n_layers, 52);
    assert_eq!(c.n_expert, 128);
    assert_eq!(c.n_expert_used, 6);
    assert_eq!(c.n_ff_exp, 1856);
    assert_eq!(c.n_ff_shexp, 3712);
    assert!(c.expert_weights_norm);
    assert_eq!(c.expert_weights_scale, 2.5);

    let kinds = rt.block_kinds();
    let n_ssm = kinds.iter().filter(|k| matches!(k, BlockKind::Ssm)).count();
    let n_attn = kinds.iter().filter(|k| matches!(k, BlockKind::Attn)).count();
    let n_moe = kinds.iter().filter(|k| matches!(k, BlockKind::MoeFfn)).count();
    let n_ffn = kinds.iter().filter(|k| matches!(k, BlockKind::Ffn)).count();
    eprintln!("block kinds: {n_ssm} ssm + {n_attn} attn + {n_moe} moe + {n_ffn} dense-ffn");
    assert_eq!(n_attn, 6);
    assert_eq!(n_ssm, 23);
    assert_eq!(n_moe, 23);
    assert_eq!(n_ffn, 0);
    assert_eq!(n_ssm + n_attn + n_moe + n_ffn, c.n_layers);
}

#[test]
fn nemotron3_moe_generate_proves_coherence() {
    let path = std::path::Path::new(PATH);
    if !path.exists() {
        eprintln!("nemotron3_moe: GGUF not mounted — skipping.");
        return;
    }
    let rt = NemotronRuntime::load(path).expect("load");
    let tk = rt.tokenizer();
    let prompt = "The capital of France is";
    let prompt_ids = tk.encode_gpt2(prompt, true);
    assert!(!prompt_ids.is_empty(), "empty prompt");

    let sample = SampleConfig::GREEDY;
    let rep_window = 64;
    let mut state = rt.new_state();
    let mut history: Vec<i32> = Vec::new();
    let mut next: i32 = -1;
    let t0 = std::time::Instant::now();
    for (i, &tok) in prompt_ids.iter().enumerate() {
        let logits = rt.forward_logits(&mut state, tok);
        history.push(tok);
        let recent = &history[history.len().saturating_sub(rep_window)..];
        let pred = sample_with_history(&logits, &sample, recent);
        if i == prompt_ids.len() - 1 { next = pred; }
    }
    let eos = tk.eos_token_id.unwrap_or(-1);
    let mut out_ids: Vec<i32> = Vec::new();
    for _ in 0..24 {
        if next == eos { break; }
        out_ids.push(next);
        history.push(next);
        let logits = rt.forward_logits(&mut state, next);
        let recent = &history[history.len().saturating_sub(rep_window)..];
        next = sample_with_history(&logits, &sample, recent);
    }
    let full = tk.decode_sequence(&out_ids, true);
    let dt = t0.elapsed();
    eprintln!(
        "nemotron3-moe generate: prompt={prompt:?} → {full:?}  [{} tok in {dt:?} = {:.2} tok/s]",
        out_ids.len(), out_ids.len() as f64 / dt.as_secs_f64(),
    );
    assert!(!full.is_empty(), "empty continuation");
    let alpha = full.chars().filter(|c| c.is_alphabetic()).count();
    let total = full.chars().count();
    assert!(alpha * 2 >= total, "output looks like garbage: {full:?}");
}
