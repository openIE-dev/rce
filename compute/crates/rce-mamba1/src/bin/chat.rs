//! `mamba1-chat` — base-LM text continuation CLI for the Mamba-1 arch.
//! Defaults to Falcon3-Mamba-7B-Instruct.

use rce_mamba1::{Mamba1GenConfig, Mamba1Runtime, SampleConfig};
use std::io::Write;
use std::time::Instant;

const DEFAULT_GGUF: &str =
    "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/falcon3-mamba-7b-gguf/Falcon3-Mamba-7B-Instruct.Q4_K_M.gguf";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model_path: Option<String> = None;
    let mut max_tokens: usize = 64;
    let mut greedy = false;
    let mut prompt_parts: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-n" | "--max-tokens" => {
                i += 1;
                max_tokens = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(64);
            }
            "--greedy" => greedy = true,
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
        eprintln!("usage: mamba1-chat [<model.gguf>] [-n N] [--greedy] \"prompt\"");
        std::process::exit(2);
    } else {
        prompt_parts.join(" ")
    };

    let model_path = model_path
        .or_else(|| std::env::var("RCE_MAMBA1_GGUF").ok())
        .unwrap_or_else(|| DEFAULT_GGUF.to_string());

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("model not found: {model_path}");
        std::process::exit(1);
    }

    eprintln!("Loading {model_path} …");
    let t0 = Instant::now();
    let rt = Mamba1Runtime::load(&model_path).unwrap_or_else(|e| {
        eprintln!("load failed: {e}");
        std::process::exit(1);
    });
    eprintln!("Loaded in {:?}. Generating (max {max_tokens})…\n", t0.elapsed());

    let cfg = Mamba1GenConfig {
        max_new_tokens: max_tokens,
        sample: if greedy { SampleConfig::GREEDY } else { SampleConfig::chat(0xC0DE_F00D_2026) },
        rep_window: 64,
    };

    print!("{prompt}");
    let _ = std::io::stdout().flush();
    let t1 = Instant::now();
    let mut n_tokens = 0usize;
    let reply = rt.generate(&prompt, &cfg, |tok| {
        print!("{tok}");
        let _ = std::io::stdout().flush();
        n_tokens += 1;
    });
    println!();
    match reply {
        Ok(_full) => {
            let dt = t1.elapsed();
            eprintln!(
                "\n[{n_tokens} tokens in {dt:?} = {:.2} tok/s]",
                n_tokens as f64 / dt.as_secs_f64()
            );
        }
        Err(e) => {
            eprintln!("generation failed: {e}");
            std::process::exit(1);
        }
    }
}
