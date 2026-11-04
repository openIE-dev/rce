//! `bamba-chat` — Llama-3 ChatML chat CLI for Bamba-9B (IBM, May 2026).

use rce_bamba::{BambaGenConfig, BambaRuntime, SampleConfig};
use std::io::Write;
use std::time::Instant;

const DEFAULT_GGUF: &str =
    "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/bamba-9b-gguf/bamba-9b.gguf";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model_path: Option<String> = None;
    let mut max_tokens: usize = 64;
    let mut greedy = false;
    let mut prompt_parts: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-n" | "--max-tokens" => { i += 1; max_tokens = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(64); }
            "--greedy" => greedy = true,
            "-m" | "--model" => { i += 1; model_path = args.get(i).cloned(); }
            s if s.ends_with(".gguf") && model_path.is_none() => model_path = Some(s.to_string()),
            s => prompt_parts.push(s.to_string()),
        }
        i += 1;
    }

    let prompt = if prompt_parts.is_empty() {
        eprintln!("usage: bamba-chat [<model.gguf>] [-n N] [--greedy] \"prompt\"");
        std::process::exit(2);
    } else {
        prompt_parts.join(" ")
    };

    let model_path = model_path
        .or_else(|| std::env::var("RCE_BAMBA_GGUF").ok())
        .unwrap_or_else(|| DEFAULT_GGUF.to_string());

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("model not found: {model_path}");
        std::process::exit(1);
    }

    eprintln!("Loading {model_path} …");
    let t0 = Instant::now();
    let rt = BambaRuntime::load(&model_path).unwrap_or_else(|e| {
        eprintln!("load failed: {e}");
        std::process::exit(1);
    });
    eprintln!("Loaded in {:?}. Generating (max {max_tokens})…\n", t0.elapsed());

    let cfg = BambaGenConfig {
        max_new_tokens: max_tokens,
        sample: if greedy { SampleConfig::GREEDY } else { SampleConfig::chat(0xC0DE_F00D_2026) },
        rep_window: 64,
    };

    // Bamba-9B-v1 is a BASE model (pretrained, no instruct tune) —
    // use raw text continuation. The chat template would confuse it.
    // For Bamba-Instruct (when released), set RCE_BAMBA_CHAT=1.
    let use_chat_template = std::env::var("RCE_BAMBA_CHAT").is_ok();
    print!("{prompt}");
    let _ = std::io::stdout().flush();
    let t1 = Instant::now();
    let mut n_tokens = 0usize;
    let on_token = |tok: &str, n: &mut usize| {
        print!("{tok}");
        let _ = std::io::stdout().flush();
        *n += 1;
    };
    let reply = if use_chat_template {
        rt.chat(&[("user", &prompt)], &cfg, |t| on_token(t, &mut n_tokens))
    } else {
        rt.generate(&prompt, &cfg, |t| on_token(t, &mut n_tokens))
    };
    println!();
    match reply {
        Ok(_full) => {
            let dt = t1.elapsed();
            eprintln!(
                "\n[{n_tokens} tokens in {dt:?} = {:.2} tok/s]",
                n_tokens as f64 / dt.as_secs_f64()
            );
        }
        Err(e) => { eprintln!("generation failed: {e}"); std::process::exit(1); }
    }
}
