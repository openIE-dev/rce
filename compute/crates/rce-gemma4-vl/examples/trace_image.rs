//! A/B driver: load Gemma 4v mmproj, encode the test image, dump
//! shape + sums of intermediate stages for comparison against
//! llama-mtmd-cli + eval-callback.
use rce_gemma4_vl::VisionTower;
fn main() {
    let mmproj = std::env::args().nth(1).unwrap_or_else(|| {
        "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/\
         gemma-4-E2B-mmproj-q8_0.gguf".into()
    });
    let img = std::env::args().nth(2).unwrap_or_else(|| {
        "/tmp/sherry-llamacpp/tools/mtmd/test-1.jpeg".into()
    });
    let v = VisionTower::load(&mmproj).expect("load");
    let t0 = std::time::Instant::now();
    let tokens = v.encode_image(&img).expect("encode");
    let dt = t0.elapsed();
    eprintln!("[G4V] encoded {} tokens × {} dim in {dt:?}", tokens.len(), tokens[0].len());
    let total: f32 = tokens.iter().flat_map(|t| t.iter()).sum();
    let abs_max = tokens.iter().flat_map(|t| t.iter()).fold(0.0f32, |a, x| a.max(x.abs()));
    eprintln!("[G4V] total_sum={total:.4}  abs_max={abs_max:.4}");
    for (i, t) in tokens.iter().enumerate() {
        let s: f32 = t.iter().sum();
        eprintln!("[G4V] token[{i}].sum = {s:.4}");
    }
}
