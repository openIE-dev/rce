//! A/B harness driver: load gemma-4-E2B, run one forward on BOS only,
//! print activation sums. Compare against llama-eval-callback's
//! single-token reference (`/tmp/evcb1.sums`). Run with
//! `RCE_TRACE=0` (layer 0 only) or `RCE_TRACE=N` (layers ≤ N).
use rce_gemma4::Gemma4Runtime;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/\
         gemma-4-E2B-text-q8_0.gguf".into()
    });
    let rt = Gemma4Runtime::load(&path).expect("load");
    let mut st = rt.new_state();
    let bos = 2i32; // Gemma 4 BOS = 2
    let lg = rt.forward_logits(&mut st, bos);
    let (am, mv) = lg.iter().copied().enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(i, v), (j, w)| if w > v { (j, w) } else { (i, v) });
    eprintln!("[RCE] logits.len={} argmax={am} max={mv}", lg.len());
}
