//! `wav2vec2-cli` — facebook/wav2vec2-base-960h speech-to-text CLI on rce.
//!
//! Usage:
//!   wav2vec2-cli transcribe path/to/audio.wav

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use rce_wav2vec2::{read_wav_mono_16k, Wav2Vec2};
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_MODEL: &str =
    "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/wav2vec2-base-960h/model.safetensors";

#[derive(Parser)]
#[command(
    name = "wav2vec2-cli",
    about = "wav2vec2-base-960h speech-to-text — on rce",
    version = env!("CARGO_PKG_VERSION")
)]
struct Cli {
    /// Path to `model.safetensors` (default: the standard pull).
    #[arg(long, default_value = DEFAULT_MODEL)]
    model: PathBuf,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Read PCM-16 mono WAV (any sample rate; auto-resampled to 16 kHz) and
    /// produce a greedy-CTC transcript.
    Transcribe { wav: PathBuf },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if !cli.model.exists() {
        return Err(anyhow!("model not found at {}", cli.model.display()));
    }

    eprintln!("[wav2vec2-cli] loading model…");
    let t0 = Instant::now();
    let m = Wav2Vec2::load(&cli.model).map_err(|e| anyhow!("{e}"))?;
    eprintln!("[wav2vec2-cli] loaded in {:?}", t0.elapsed());

    match cli.cmd {
        Cmd::Transcribe { wav } => {
            if !wav.exists() {
                return Err(anyhow!("audio not found at {}", wav.display()));
            }
            eprintln!("[wav2vec2-cli] reading + resampling {}…", wav.display());
            let audio = read_wav_mono_16k(&wav).map_err(|e| anyhow!("{e}"))?;
            eprintln!(
                "[wav2vec2-cli] loaded {} samples ({:.2} s @ 16 kHz)",
                audio.len(),
                audio.len() as f32 / 16_000.0
            );

            let t0 = Instant::now();
            let text = m.transcribe_wav(&wav).map_err(|e| anyhow!("{e}"))?;
            let dt = t0.elapsed();
            eprintln!(
                "[wav2vec2-cli] forward + decode in {dt:?} ({:.2}× realtime)",
                audio.len() as f32 / 16_000.0 / dt.as_secs_f32()
            );

            println!("{text}");
        }
    }
    Ok(())
}
