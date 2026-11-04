//! `codec-cli` — neural-audio-codec roundtrip CLI on rce.
//!
//! Usage:
//!   codec-cli roundtrip --codec snac        in.wav out.wav
//!   codec-cli roundtrip --codec encodec     in.wav out.wav
//!   codec-cli roundtrip --codec wavtokenizer in.wav out.wav
//!
//! Each codec runs at its native sample rate; the CLI auto-resamples the input
//! wav (any SR, mono, PCM-16) to the codec's SR via `rce_wav2vec2`'s built-in
//! polyphase resampler, then writes the reconstruction as PCM-16 at the same SR.

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::{Path, PathBuf};
use std::time::Instant;

const MODEL_LIB: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library";

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Codec { Snac, Encodec, Wavtokenizer }

impl Codec {
    fn sr(self) -> u32 {
        match self {
            Self::Snac => rce_snac::SR,
            Self::Encodec => rce_encodec::SR,
            Self::Wavtokenizer => rce_wavtokenizer::SR,
        }
    }
    fn default_model_path(self) -> PathBuf {
        let sub = match self {
            Self::Snac => "snac-24khz/model.safetensors",
            Self::Encodec => "encodec-32khz/model.safetensors",
            Self::Wavtokenizer => "wavtokenizer/model.safetensors",
        };
        PathBuf::from(MODEL_LIB).join(sub)
    }
}

#[derive(Parser)]
#[command(
    name = "codec-cli",
    about = "Unified roundtrip CLI for rce neural audio codecs",
    version = env!("CARGO_PKG_VERSION")
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Encode → decode → write back a WAV.  Auto-resamples input to the codec's SR.
    Roundtrip {
        #[arg(long, value_enum)]
        codec: Codec,
        /// Optional override for the safetensors checkpoint.
        #[arg(long)]
        model: Option<PathBuf>,
        input: PathBuf,
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Roundtrip { codec, model, input, output } => {
            let model_path = model.unwrap_or_else(|| codec.default_model_path());
            if !model_path.exists() {
                return Err(anyhow!("model not found at {}", model_path.display()));
            }
            eprintln!("[codec-cli] codec = {codec:?} (SR = {} Hz)", codec.sr());
            eprintln!("[codec-cli] reading + resampling {} → {} Hz…", input.display(), codec.sr());
            let audio_16k = rce_wav2vec2::read_wav_mono_16k(&input)
                .map_err(|e| anyhow!("read wav: {e}"))?;
            // `read_wav_mono_16k` lands at 16 kHz; resample again to the codec rate.
            let audio = resample_to(&audio_16k, 16_000, codec.sr());
            eprintln!("[codec-cli] {} samples @ {} Hz ({:.2} s)",
                      audio.len(), codec.sr(), audio.len() as f32 / codec.sr() as f32);

            let t0 = Instant::now();
            let recon = match codec {
                Codec::Snac => {
                    let m = rce_snac::Snac::load(&model_path).map_err(|e| anyhow!("{e}"))?;
                    let (r, codes) = m.roundtrip(&audio, 0xC0FFEE);
                    eprintln!("[codec-cli] codes per stage: {:?}",
                              codes.iter().map(|c| c.len()).collect::<Vec<_>>());
                    r
                }
                Codec::Encodec => {
                    let m = rce_encodec::Encodec::load(&model_path).map_err(|e| anyhow!("{e}"))?;
                    let (r, codes) = m.roundtrip(&audio);
                    eprintln!("[codec-cli] codes per stage: {:?}",
                              codes.iter().map(|c| c.len()).collect::<Vec<_>>());
                    r
                }
                Codec::Wavtokenizer => {
                    let m = rce_wavtokenizer::WavTokenizer::load(&model_path)
                        .map_err(|e| anyhow!("{e}"))?;
                    let (r, codes) = m.roundtrip(&audio);
                    eprintln!("[codec-cli] codes: {}", codes.len());
                    r
                }
            };
            let dt = t0.elapsed();
            eprintln!("[codec-cli] roundtrip in {dt:?} ({:.2}× realtime)",
                      audio.len() as f32 / codec.sr() as f32 / dt.as_secs_f32());

            write_wav_pcm16(&output, &recon, codec.sr())
                .map_err(|e| anyhow!("write wav: {e}"))?;
            eprintln!("[codec-cli] wrote {} ({} samples)", output.display(), recon.len());
            Ok(())
        }
    }
}

/// Simple polyphase rate change (32-tap, Hann-windowed sinc). Same algorithm
/// `rce_wav2vec2` uses; kept inline so we don't expose it from there.
fn resample_to(input: &[f32], sr_in: u32, sr_out: u32) -> Vec<f32> {
    if sr_in == sr_out { return input.to_vec(); }
    let g = gcd(sr_in, sr_out);
    let up = (sr_out / g) as usize;
    let dn = (sr_in  / g) as usize;
    polyphase_resample(input, up, dn)
}
fn gcd(a: u32, b: u32) -> u32 { if b == 0 { a } else { gcd(b, a % b) } }
fn polyphase_resample(x: &[f32], up: usize, dn: usize) -> Vec<f32> {
    const TAPS: usize = 32;
    let cutoff = 1.0_f32 / (up.max(dn) as f32);
    let kernel_len = TAPS * up;
    let mut k = vec![0.0f32; kernel_len];
    let center = (kernel_len as f32 - 1.0) / 2.0;
    for i in 0..kernel_len {
        let t = i as f32 - center;
        let s = if t == 0.0 { 1.0 } else {
            let pt = std::f32::consts::PI * t * cutoff;
            pt.sin() / pt
        };
        let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (kernel_len as f32 - 1.0)).cos();
        k[i] = s * w;
    }
    let g_sum: f32 = k.iter().sum::<f32>() / up as f32;
    for v in k.iter_mut() { *v /= g_sum; }
    let n_out = (x.len() * up + dn - 1) / dn;
    let mut y = vec![0.0f32; n_out];
    for n in 0..n_out {
        let m = n * dn;
        let mut acc = 0.0f32;
        for tap in 0..kernel_len {
            if tap > m { break; }
            let up_idx = m - tap;
            if up_idx % up != 0 { continue; }
            let xi = up_idx / up;
            if xi < x.len() { acc += k[tap] * x[xi]; }
        }
        y[n] = acc;
    }
    y
}

/// Minimal PCM-16 mono RIFF/WAVE writer.
fn write_wav_pcm16(path: &Path, samples: &[f32], sr: u32) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    let bits_per_sample = 16u16;
    let channels = 1u16;
    let byte_rate = sr * channels as u32 * bits_per_sample as u32 / 8;
    let block_align = channels * bits_per_sample / 8;
    let data_size = (samples.len() as u32) * (bits_per_sample as u32 / 8);
    let chunk_size = 36 + data_size;
    f.write_all(b"RIFF")?;
    f.write_all(&chunk_size.to_le_bytes())?;
    f.write_all(b"WAVE")?;
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?;          // fmt chunk size
    f.write_all(&1u16.to_le_bytes())?;           // format = PCM
    f.write_all(&channels.to_le_bytes())?;
    f.write_all(&sr.to_le_bytes())?;
    f.write_all(&byte_rate.to_le_bytes())?;
    f.write_all(&block_align.to_le_bytes())?;
    f.write_all(&bits_per_sample.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_size.to_le_bytes())?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}
