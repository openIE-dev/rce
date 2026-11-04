//! Run Nemotron-Nano-4B with the same 2-token prompt llama-eval-callback uses
//! ([BOS, "Hi"] = [1, 37133]), trace per-block residual values for the first
//! two forward calls, and compare against the reference values dumped from
//! llama-eval-callback. The first call (BOS) should match block-by-block;
//! the second call (Hi) is the apples-to-apples comparison for llama.cpp's
//! batch=2 last-token output.

use rce_nemotron::NemotronRuntime;

const PATH: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/nemotron-nano-raw/models--unsloth--NVIDIA-Nemotron-3-Nano-4B-GGUF/snapshots/8e81be55c5aa3d63bb82b6ceec62d50805d9e1bb/NVIDIA-Nemotron-3-Nano-4B-Q4_0.gguf";

#[test]
fn nemotron_bos_then_hi_per_block_trace() {
    let path = std::path::Path::new(PATH);
    if !path.exists() {
        eprintln!("nemotron_bos_then_hi_per_block_trace: GGUF not found — skipping.");
        return;
    }
    // Force-enable the per-block trace via the env var the runtime checks.
    // SAFETY: we are the only thread running this test.
    unsafe { std::env::set_var("RCE_NEMOTRON_TRACE", "1"); }

    let rt = NemotronRuntime::load(path).expect("load");
    eprintln!("---- block kinds ----");
    for (i, k) in rt.block_kinds().iter().enumerate() {
        eprintln!("  blk{i:2}: {k:?}");
    }
    let mut state = rt.new_state();
    eprintln!("\n==== pos=0: process BOS (token 1) ====");
    let _logits0 = rt.forward_logits(&mut state, 1);
    eprintln!("\n==== pos=1: process Hi (token 37133) ====");
    let logits1 = rt.forward_logits(&mut state, 37133);

    // llama-eval-callback's result_output for this exact 2-token prefill was:
    //   [-5.6422, -0.2728, -1.6399, ..., -6.6194, -4.9938, -4.7909]
    eprintln!("\n[final] logits1[0..3]={:?}", &logits1[..3]);
    eprintln!("[final] logits1[131069..131072]={:?}",
        &logits1[131069..131072]);
}
