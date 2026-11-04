use rce_gemma4_audio::{AudioTower, mel_spectrogram_gemma4a, load_pcm_s16le};
fn stats(label: &str, x: &[f32]) {
    let n = x.len();
    let s: f32 = x.iter().sum();
    let amax = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let rms = ms.sqrt();
    eprintln!("{label:30}  len={n}  sum={s:>10.3}  rms={rms:.4}  abs_max={amax:.4}");
}
fn main() {
    let pcm = std::env::args().nth(1).unwrap_or_else(|| "/tmp/test_audio_short.pcm".into());
    let mmp = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-mmproj-q8_0.gguf";
    let samples = load_pcm_s16le(&pcm).expect("pcm");
    eprintln!("Loaded {} samples", samples.len());
    let mel = mel_spectrogram_gemma4a(&samples);
    let flat: Vec<f32> = mel.iter().flat_map(|f| f.iter().copied()).collect();
    stats(&format!("mel ({} frames × 128)", mel.len()), &flat);
    let v = AudioTower::load(mmp).expect("load");
    let toks = v.encode_mel(&mel).expect("encode");
    let flat: Vec<f32> = toks.iter().flat_map(|t| t.iter().copied()).collect();
    stats(&format!("audio_tokens ({} × {})", toks.len(), toks[0].len()), &flat);
    for (i, t) in toks.iter().enumerate().take(3) {
        stats(&format!("  token[{i}]"), t);
    }
}
