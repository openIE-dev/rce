//! `moonshine-cli` — Useful Sensors Moonshine ASR (raw-audio convs +
//! encoder-decoder Transformer with cross-attention) on rce.

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use rce_moonshine::Moonshine;
use rce_wav2vec2::read_wav_mono_16k;
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_MODEL_DIR: &str =
    "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/moonshine";

#[derive(Parser)]
#[command(
    name = "moonshine-cli",
    about = "Moonshine ASR — encoder-decoder cross-attention transformer on rce",
    version = env!("CARGO_PKG_VERSION")
)]
struct Cli {
    /// Directory containing `model.safetensors` and `tokenizer.json`.
    #[arg(long, default_value = DEFAULT_MODEL_DIR)]
    model: PathBuf,
    /// Maximum new tokens for the decoder.
    #[arg(short = 'n', long, default_value_t = 200)]
    max_new: usize,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Transcribe { wav: PathBuf },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if !cli.model.join("model.safetensors").exists() {
        return Err(anyhow!("model not found at {}", cli.model.display()));
    }

    eprintln!("[moonshine-cli] loading model…");
    let t0 = Instant::now();
    let m = Moonshine::load(&cli.model).map_err(|e| anyhow!(e))?;
    eprintln!("[moonshine-cli] loaded in {:?}", t0.elapsed());

    match cli.cmd {
        Cmd::Transcribe { wav } => {
            if !wav.exists() {
                return Err(anyhow!("audio not found at {}", wav.display()));
            }
            eprintln!("[moonshine-cli] reading {}…", wav.display());
            let audio = read_wav_mono_16k(&wav).map_err(|e| anyhow!("{e}"))?;
            eprintln!(
                "[moonshine-cli] {} samples ({:.2} s @ 16 kHz)",
                audio.len(),
                audio.len() as f32 / 16_000.0
            );

            let t0 = Instant::now();
            let text = m.transcribe(&audio, cli.max_new);
            let dt = t0.elapsed();
            eprintln!(
                "[moonshine-cli] forward + decode in {dt:?} ({:.2}× realtime)",
                audio.len() as f32 / 16_000.0 / dt.as_secs_f32()
            );
            println!("{text}");
            Ok(())
        }
    }
}
