//! Profile mamba2 forward passes — report per-op timing breakdown.
//!
//! Usage: `mamba2-profile [-n N] [-m PATH]`. Runs N forward passes
//! (default 16) with throwaway state and prints the cumulative time
//! spent in each op. Use this to pick optimization targets before
//! swapping in GPU/SIMD kernels.

use rce_mamba2::{Mamba2Runtime, Profile};
use std::cell::RefCell;
use std::time::Instant;

const DEFAULT_GGUF: &str =
    "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/mamba2-2.7b-gguf/mamba2-2.7b-q5_k_m.gguf";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model_path: Option<String> = None;
    let mut n_passes: usize = 16;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-n" => {
                i += 1;
                n_passes = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(16);
            }
            "-m" | "--model" => {
                i += 1;
                model_path = args.get(i).cloned();
            }
            s if s.ends_with(".gguf") && model_path.is_none() => model_path = Some(s.to_string()),
            _ => {}
        }
        i += 1;
    }
    let model_path = model_path.unwrap_or_else(|| DEFAULT_GGUF.to_string());
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("model not found: {model_path}");
        std::process::exit(1);
    }

    eprintln!("Loading {model_path} …");
    let t0 = Instant::now();
    let rt = Mamba2Runtime::load(&model_path).unwrap();
    eprintln!("Loaded in {:?}", t0.elapsed());

    let mut state = rt.new_state();
    // Warmup: 2 untimed forwards
    for _ in 0..2 {
        rt.forward_logits(&mut state, 1);
    }

    // Profile run
    let prof = RefCell::new(Profile::default());
    let t1 = Instant::now();
    for tok in 0..n_passes {
        // Use varying token ids to avoid degenerate state.
        rt.forward_logits_profiled(&mut state, ((tok * 37) % 50000) as i32, &prof);
    }
    let total = t1.elapsed();
    eprintln!(
        "\nRan {n_passes} forward passes in {:?} = {:.2} tok/s wall time.",
        total, n_passes as f64 / total.as_secs_f64()
    );

    prof.borrow().report(&format!("{n_passes}-pass mamba2-2.7b"));
}
