//! edsr-cli — 2× super-resolution via rce-edsr on a real image.
//!
//! ```sh
//! edsr-cli upscale in.png out.png [--weights path/to/edsr_baseline_x2.safetensors]
//! ```
//!
//! Weights are produced by `joulesperbit/tools/convert_onnx.py
//! model_library/edsr/edsr_baseline_x2.onnx`.

use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_WEIGHTS: &str = "/tmp/onnx_sweep/edsr_baseline_x2.safetensors";

fn usage() -> ! {
    eprintln!("usage: edsr-cli upscale <in> <out> [--weights <path>]");
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 || args[1] != "upscale" { usage(); }
    let inp = PathBuf::from(&args[2]);
    let out = PathBuf::from(&args[3]);
    let mut weights = PathBuf::from(DEFAULT_WEIGHTS);
    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => { weights = PathBuf::from(&args[i + 1]); i += 2; }
            other => { eprintln!("unknown arg: {other}"); usage(); }
        }
    }
    if !weights.exists() {
        eprintln!("weights not found: {}", weights.display());
        eprintln!("  run: python3 joulesperbit/tools/convert_onnx.py edsr_baseline_x2.onnx {DEFAULT_WEIGHTS}");
        std::process::exit(1);
    }

    let t = Instant::now();
    let model = rce_edsr::Edsr::load(&weights).expect("load weights");
    eprintln!("loaded weights in {:?}", t.elapsed());

    let t = Instant::now();
    let img = image::open(&inp).expect("open image").to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    eprintln!("loaded {w}×{h} RGB image in {:?}", t.elapsed());

    // Layout: PyTorch channel-first [3, H, W] in [0, 255] f32.
    let mut chw = vec![0.0f32; 3 * h * w];
    for y in 0..h { for x in 0..w {
        let p = img.get_pixel(x as u32, y as u32);
        for c in 0..3 { chw[c * h * w + y * w + x] = p[c] as f32; }
    }}

    let t = Instant::now();
    let (sr, ho, wo) = model.infer(&chw, h, w);
    eprintln!("super-resolved {h}×{w} → {ho}×{wo} in {:?}", t.elapsed());

    // Clamp + convert to u8 PNG.
    let mut out_img = image::RgbImage::new(wo as u32, ho as u32);
    for y in 0..ho { for x in 0..wo {
        let p = [
            sr[0 * ho * wo + y * wo + x].clamp(0.0, 255.0) as u8,
            sr[1 * ho * wo + y * wo + x].clamp(0.0, 255.0) as u8,
            sr[2 * ho * wo + y * wo + x].clamp(0.0, 255.0) as u8,
        ];
        out_img.put_pixel(x as u32, y as u32, image::Rgb(p));
    }}
    out_img.save(&out).expect("save");
    eprintln!("wrote {} ({}×{})", out.display(), wo, ho);
}
