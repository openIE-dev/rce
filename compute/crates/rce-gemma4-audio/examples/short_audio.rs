use rce_gemma4_audio::{Gemma4AudioRuntime, G4AGenConfig};
use rce_ingest_gguf::SampleConfig;
fn main() {
    let txt = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-text-q8_0.gguf";
    let mmp = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-mmproj-q8_0.gguf";
    let pcm = std::env::args().nth(1).unwrap_or_else(|| "/tmp/test_audio_short.pcm".into());
    let rt = Gemma4AudioRuntime::load(txt, mmp).expect("load");
    let max_new = std::env::var("RCE_MAX_NEW").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    let cfg = G4AGenConfig {
        max_new_tokens: max_new, sample: SampleConfig::GREEDY, rep_window: 64,
    };
    let mut tok_count: usize = 0;
    let t0 = std::time::Instant::now();
    let answer = rt.answer(pcm, "What do you hear in this audio?", &cfg,
        |_| tok_count += 1).expect("answer");
    let dt = t0.elapsed();
    eprintln!("output → {:?}  [{tok_count} tok in {dt:?}]", answer);
}
