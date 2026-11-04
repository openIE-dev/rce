use rce_gemma4_audio::{Gemma4AudioRuntime, G4AGenConfig};
use rce_ingest_gguf::SampleConfig;
fn main() {
    let txt = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-text-q8_0.gguf";
    let mmp = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-mmproj-q8_0.gguf";
    let rt = Gemma4AudioRuntime::load(txt, mmp).expect("load");
    let cfg = G4AGenConfig {
        max_new_tokens: 60, sample: SampleConfig::GREEDY, rep_window: 64,
    };
    for sec in ["1.5", "2", "3", "5", "8", "12"] {
        let path = format!("/tmp/real_{sec}s.pcm");
        let t0 = std::time::Instant::now();
        let out = rt.answer(&path, "What do you hear?", &cfg, |_| {}).unwrap_or_default();
        let dt = t0.elapsed();
        let coherent = !out.contains("<|audio>\n<|audio>") &&
                       !out.contains("<end_of_turn>\n<start_of_turn>");
        println!("[{sec:>4}s] coh={coherent:5}  {dt:?}  → {out:?}");
    }
}
