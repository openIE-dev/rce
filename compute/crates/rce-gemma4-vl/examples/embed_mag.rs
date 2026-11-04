//! Compare magnitudes: text-token embedding × √d_model vs vision-tower
//! projected image embedding. If they're orders apart, the splice
//! produces out-of-distribution input to the decoder.
use rce_gemma4::Gemma4Runtime;
use rce_gemma4_vl::VisionTower;
fn rms(x: &[f32]) -> f32 {
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    ms.sqrt()
}
fn stats(label: &str, x: &[f32]) {
    let n = x.len();
    let s: f32 = x.iter().sum();
    let amax = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let r = rms(x);
    eprintln!("{label:32}  len={n}  sum={s:>10.3}  rms={r:.4}  abs_max={amax:.4}");
}
fn main() {
    let txt = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-text-q8_0.gguf";
    let mmp = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-mmproj-q8_0.gguf";
    let rt = Gemma4Runtime::load(txt).expect("txt load");
    // Text embeddings of a few representative tokens.
    for &id in &[0i32 /*pad*/, 1 /*eos*/, 2 /*bos*/, 26352 /*"Hello"*/, 258880 /*<|image|>*/, 36129 /*random*/] {
        let e = rt.embed_token(id);
        stats(&format!("embed_token({id})"), &e);
    }
    // Vision tower output on the test image.
    let v = VisionTower::load(mmp).expect("v load");
    let img = "/tmp/sherry-llamacpp/tools/mtmd/test-1.jpeg";
    if !std::path::Path::new(img).exists() {
        eprintln!("test image missing — skipping vision dump");
        return;
    }
    let tokens = v.encode_image(img).expect("encode");
    for (i, t) in tokens.iter().enumerate() {
        stats(&format!("vision_token[{i}]"), t);
    }
}
