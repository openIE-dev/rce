//! Bracket where the audio path breaks down: drive rce-gemma4-audio
//! with synthesized speech-like audio at progressively longer lengths.
//! Synthetic = chirp + amplitude-modulated noise (a stand-in for real
//! speech features so the encoder has structure to work with). The
//! semantic content won't be meaningful, but if the OUTPUT STRUCTURE
//! changes from "coherent text" to "<|audio> template noise" at some
//! length threshold, that locates the bug.
use rce_gemma4_audio::{Gemma4AudioRuntime, G4AGenConfig};
use rce_ingest_gguf::SampleConfig;

const TXT: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-text-q8_0.gguf";
const MMP: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-mmproj-q8_0.gguf";

fn synth_pcm(seconds: f32) -> Vec<u8> {
    let sr = 16000usize;
    let n = (sr as f32 * seconds) as usize;
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / sr as f32;
        // Chirp 200→1500 Hz with a slow amplitude modulation.
        let f = 200.0 + 1300.0 * (t / seconds).min(1.0);
        let phase = 2.0 * std::f32::consts::PI * f * t;
        let amp = 0.5 + 0.4 * (2.0 * std::f32::consts::PI * 3.0 * t).sin();
        let s = (amp * 0.3 * phase.sin() * 32767.0) as i16;
        samples.push(s);
    }
    let mut bytes = Vec::with_capacity(n * 2);
    for s in samples { bytes.extend_from_slice(&s.to_le_bytes()); }
    bytes
}

fn write_pcm(path: &str, seconds: f32) {
    std::fs::write(path, synth_pcm(seconds)).unwrap();
}

fn main() {
    let rt = Gemma4AudioRuntime::load(TXT, MMP).expect("load");
    let cfg = G4AGenConfig {
        max_new_tokens: 20, sample: SampleConfig::GREEDY, rep_window: 64,
    };
    let lengths_sec = [1.0f32, 2.0, 3.0, 5.0, 8.0, 12.0, 17.0];
    for &sec in &lengths_sec {
        let path = format!("/tmp/synth_{}s.pcm", (sec * 10.0) as i32);
        write_pcm(&path, sec);
        let n_tokens_est = ((sec * 16000.0 / 160.0) as usize).saturating_sub(1) / 4;
        let mut chunks: Vec<String> = Vec::new();
        let t0 = std::time::Instant::now();
        let r = rt.answer(&path, "What do you hear?", &cfg, |s| chunks.push(s.to_string()));
        let dt = t0.elapsed();
        let out = chunks.concat();
        let coherent = !out.contains("<|audio>\n<|audio>")
            && !out.contains("<end_of_turn>\n<start_of_turn>");
        let r_str = r.as_ref().map(|_| "ok").unwrap_or("err");
        println!(
            "[{sec:>4.1}s ≈ {n_tokens_est:>3} aud_tok] {r_str:3} coh={coherent:5} \
             {dt:?} → {out:?}"
        );
        let _ = std::fs::remove_file(&path);
    }
}
