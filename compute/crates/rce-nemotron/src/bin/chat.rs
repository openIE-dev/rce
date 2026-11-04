//! `rce-nemotron-chat` — single-turn CLI against a Nemotron-H GGUF,
//! token-by-token streaming.

use rce_nemotron::{NemotronGenConfig, NemotronRuntime, SampleConfig};
use std::io::Write;
use std::time::Instant;

const DEFAULT_GGUF: &str =
    "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/nemotron-nano-raw/models--unsloth--NVIDIA-Nemotron-3-Nano-4B-GGUF/snapshots/8e81be55c5aa3d63bb82b6ceec62d50805d9e1bb/NVIDIA-Nemotron-3-Nano-4B-Q4_0.gguf";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model_path: Option<String> = None;
    let mut max_tokens: usize = 64;
    let mut think = true;
    let mut greedy = false;
    let mut prompt_parts: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-n" | "--max-tokens" => {
                i += 1;
                max_tokens = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(64);
            }
            "--no-thinking" => think = false,
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
        eprintln!("usage: rce-nemotron-chat [<model.gguf>] [-n N] [--no-thinking] \"prompt\"");
        std::process::exit(2);
    } else {
        prompt_parts.join(" ")
    };

    let model_path = model_path
        .or_else(|| std::env::var("RCE_NEMOTRON_GGUF").ok())
        .unwrap_or_else(|| DEFAULT_GGUF.to_string());

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("model not found: {model_path}");
        std::process::exit(1);
    }

    eprintln!("Loading {model_path} …");
    let t0 = Instant::now();
    let rt = NemotronRuntime::load(&model_path).unwrap_or_else(|e| {
        eprintln!("load failed: {e}");
        std::process::exit(1);
    });
    eprintln!("Loaded in {:?}. Generating (max {max_tokens})…\n", t0.elapsed());

    let cfg = NemotronGenConfig {
        max_new_tokens: max_tokens,
        sample: if greedy { SampleConfig::GREEDY } else { SampleConfig::chat(0xC0DE_F00D_2026) },
        rep_window: 64,
        enable_thinking: think,
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
