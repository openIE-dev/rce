//! `rce-jamba-chat` — single-turn CLI chat against a Jamba-Reasoning
//! GGUF, with token-by-token streaming to stdout.
//!
//! Usage:
//!   rce-jamba-chat <model.gguf> [-n MAX_TOKENS] [--no-thinking] "your prompt"
//!   rce-jamba-chat "your prompt"     # uses $RCE_JAMBA_GGUF or the default path
//!
//! Everything runs through the rce substrate: matmul → BLAS leaf,
//! conv1d → Conv1dCausalLeaf, SSM scan → SsmScanMamba1Leaf, attention
//! → AttentionLeaf, silu/rms_norm/split/swiglu → nn-prims leaves.

use rce_jamba::{GenConfig, JambaRuntime, SampleConfig};
use std::io::Write;
use std::time::Instant;

const DEFAULT_GGUF: &str =
    "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/jamba-reasoning-3b-gguf/jamba-reasoning-3b-Q4_K_M.gguf";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model_path: Option<String> = None;
    let mut max_tokens: usize = 256;
    let mut thinking = true;
    let mut prompt_parts: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-n" | "--max-tokens" => {
                i += 1;
                max_tokens = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(256);
            }
            "--no-thinking" => thinking = false,
            "-m" | "--model" => {
                i += 1;
                model_path = args.get(i).cloned();
            }
            s if s.ends_with(".gguf") && model_path.is_none() => model_path = Some(s.to_string()),
            s => prompt_parts.push(s.to_string()),
        }
        i += 1;
    }

    let prompt = if prompt_parts.is_empty() {
        eprintln!("usage: rce-jamba-chat [<model.gguf>] [-n N] [--no-thinking] \"prompt\"");
        std::process::exit(2);
    } else {
        prompt_parts.join(" ")
    };

    let model_path = model_path
        .or_else(|| std::env::var("RCE_JAMBA_GGUF").ok())
        .unwrap_or_else(|| DEFAULT_GGUF.to_string());

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("model not found: {model_path}");
        eprintln!("(pass a path, set $RCE_JAMBA_GGUF, or mount the default location)");
        std::process::exit(1);
    }

    eprintln!("Loading {model_path} …");
    let t0 = Instant::now();
    let rt = match JambaRuntime::load(&model_path) {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("load failed: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("Loaded in {:?}. Generating (max {max_tokens} tokens)…\n", t0.elapsed());

    let cfg = GenConfig {
        max_new_tokens: max_tokens,
        sample: SampleConfig::chat(0xC0DE_F00D_2026),
        rep_window: 64,
        thinking_prefix: thinking,
        batched_prefill: true,
    };

    print!("> {prompt}\n\n");
    let _ = std::io::stdout().flush();
    let t1 = Instant::now();
    let mut n_tokens = 0usize;
    let reply = rt.chat(&[("user", &prompt)], &cfg, |tok| {
        print!("{tok}");
        let _ = std::io::stdout().flush();
        n_tokens += 1;
    });
    println!();
    match reply {
        Ok(full) => {
            let dt = t1.elapsed();
            eprintln!(
                "\n[{n_tokens} tokens in {dt:?} = {:.2} tok/s]",
                n_tokens as f64 / dt.as_secs_f64()
            );
            let _ = full;
        }
        Err(e) => {
            eprintln!("generation failed: {e}");
            std::process::exit(1);
        }
    }
}
