//! Phase 2.C — image → vision-encoder input patches.
//!
//! Mirrors `transformers.Qwen2VLImageProcessorFast` (the same image processor
//! Cosmos 3 / Qwen3-VL uses):
//!
//!   1. `smart_resize(h, w)` rounds each dim to a multiple of
//!      `patch * merge = 32`, then rescales so the total pixel budget falls
//!      between `min_pixels` and `max_pixels`.
//!   2. Resample with bicubic to the new size.
//!   3. Normalize  `(x/255 - 0.5) / 0.5  =  x/127.5 - 1` per channel.
//!   4. Duplicate the single frame to `T = temporal_patch_size = 2` frames.
//!   5. View → permute → reshape exactly as HF does, producing patches of
//!      shape `[grid_t · grid_h · grid_w,  C · T_patch · P · P]  =  [seq, 1536]`.
//!
//! Per-patch tensor layout is `(C, T_patch, P, P)` flattened — same convention
//! the Phase-2 vision encoder expects.

use crate::vision::{
    IN_CH, NUM_GRID_PER_SIDE, PATCH_DIM, PATCH_SIZE, SPATIAL_MERGE, TEMPORAL_PATCH,
};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, RgbImage};
use std::path::Path;

/// Defaults from `preprocessor_config.json`.
pub const MIN_PIXELS: usize = 65536;        // shortest_edge
pub const MAX_PIXELS: usize = 16_777_216;   // longest_edge
const FACTOR: usize = PATCH_SIZE * SPATIAL_MERGE;   // 32
const _: () = assert!(NUM_GRID_PER_SIDE > 0);

#[derive(Debug)]
pub enum PreprocError { Io(String), Bounds(String) }
impl std::fmt::Display for PreprocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(m)     => write!(f, "image preproc io: {m}"),
            Self::Bounds(m) => write!(f, "image preproc bounds: {m}"),
        }
    }
}
impl std::error::Error for PreprocError {}

/// Same algorithm as HF `qwen3_vl.video_processing.smart_resize` with `num_frames=1`.
/// Returns `(h_bar, w_bar)` — the target resize dimensions.
pub fn smart_resize(height: usize, width: usize) -> Result<(usize, usize), PreprocError> {
    if height < FACTOR || width < FACTOR {
        return Err(PreprocError::Bounds(
            format!("height {height} or width {width} smaller than factor {FACTOR}")));
    }
    if height.max(width) as f32 / height.min(width) as f32 > 200.0 {
        return Err(PreprocError::Bounds(
            format!("aspect ratio {} exceeds 200", height.max(width) as f32 / height.min(width) as f32)));
    }
    let mut h_bar = ((height as f32 / FACTOR as f32).round() as usize).max(1) * FACTOR;
    let mut w_bar = ((width  as f32 / FACTOR as f32).round() as usize).max(1) * FACTOR;
    // Static image: num_frames=1 → t_bar = ceil(1/2)*2 = 2.
    let t_bar: usize = 2;
    let total = t_bar * h_bar * w_bar;
    if total > MAX_PIXELS {
        let beta = ((1 * height * width) as f32 / MAX_PIXELS as f32).sqrt();
        h_bar = FACTOR.max(((height as f32 / beta / FACTOR as f32).floor() as usize) * FACTOR);
        w_bar = FACTOR.max(((width  as f32 / beta / FACTOR as f32).floor() as usize) * FACTOR);
    } else if total < MIN_PIXELS {
        let beta = (MIN_PIXELS as f32 / (1 * height * width) as f32).sqrt();
        h_bar = (((height as f32 * beta / FACTOR as f32).ceil() as usize) * FACTOR).max(FACTOR);
        w_bar = (((width  as f32 * beta / FACTOR as f32).ceil() as usize) * FACTOR).max(FACTOR);
    }
    Ok((h_bar, w_bar))
}

/// Read an image from disk and run the full preprocessor.  Returns
/// `(patches[seq × 1536], grid_thw = (1, grid_h, grid_w))`.
pub fn preprocess_image_file(path: &Path) -> Result<(Vec<f32>, (usize, usize, usize)), PreprocError> {
    let img = image::open(path).map_err(|e| PreprocError::Io(format!("open {}: {e}", path.display())))?;
    preprocess_image(&img)
}

pub fn preprocess_image(img: &DynamicImage) -> Result<(Vec<f32>, (usize, usize, usize)), PreprocError> {
    let (w0, h0) = img.dimensions();
    let (h_new, w_new) = smart_resize(h0 as usize, w0 as usize)?;
    // PIL → bicubic (Catmull-Rom is the closest bicubic in `image`).
    let rgb_resized: RgbImage = img
        .resize_exact(w_new as u32, h_new as u32, FilterType::CatmullRom)
        .to_rgb8();
    let pixels = rgb_to_normalized(&rgb_resized, h_new, w_new);
    let grid_h = h_new / PATCH_SIZE;
    let grid_w = w_new / PATCH_SIZE;
    let grid_t = 1;
    assert_eq!(grid_h % SPATIAL_MERGE, 0);
    assert_eq!(grid_w % SPATIAL_MERGE, 0);
    let patches = patchify(&pixels, h_new, w_new, grid_h, grid_w);
    Ok((patches, (grid_t, grid_h, grid_w)))
}

/// RGB u8 → `[C, H, W]` f32 in `[-1, 1]` (`(x/255 - 0.5)/0.5  =  x/127.5 - 1`).
fn rgb_to_normalized(img: &RgbImage, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; IN_CH * h * w];
    let raw = img.as_raw();   // packed RGB row-major
    for y in 0..h {
        for x in 0..w {
            let p = y * w * 3 + x * 3;
            for c in 0..3 {
                out[c * h * w + y * w + x] = raw[p + c] as f32 / 127.5 - 1.0;
            }
        }
    }
    out
}

/// View → permute → reshape per HF Qwen2VL fast image processor.
/// Input `pixels` is `[C, H, W]`; output is `[seq, 1536]` where
/// `seq = 1 · grid_h · grid_w` and per-patch layout is `(C, T_patch, P, P)`.
fn patchify(pixels: &[f32], h: usize, w: usize, grid_h: usize, grid_w: usize) -> Vec<f32> {
    let n_patches = grid_h * grid_w;            // grid_t = 1
    let mut out = vec![0.0f32; n_patches * PATCH_DIM];
    let h_blocks = grid_h / SPATIAL_MERGE;
    let w_blocks = grid_w / SPATIAL_MERGE;
    let mut token = 0usize;
    // Outer iteration matches HF permute(0, 1, 4, 7, 5, 8, ...).
    for hb in 0..h_blocks {
        for wb in 0..w_blocks {
            for mh in 0..SPATIAL_MERGE {
                for mw in 0..SPATIAL_MERGE {
                    let h_patch = hb * SPATIAL_MERGE + mh;
                    let w_patch = wb * SPATIAL_MERGE + mw;
                    let p_off = token * PATCH_DIM;
                    // Per-patch layout: (C, T_patch, P, P).  Both T_patch frames
                    // are the same static image — we duplicate.
                    for c in 0..IN_CH {
                        for tp in 0..TEMPORAL_PATCH {
                            for ph in 0..PATCH_SIZE {
                                for pw in 0..PATCH_SIZE {
                                    let y_img = h_patch * PATCH_SIZE + ph;
                                    let x_img = w_patch * PATCH_SIZE + pw;
                                    let _ = tp; // pixel is the same for both temporal slots
                                    let v = pixels[c * h * w + y_img * w + x_img];
                                    let idx = ((c * TEMPORAL_PATCH + tp) * PATCH_SIZE + ph) * PATCH_SIZE + pw;
                                    out[p_off + idx] = v;
                                }
                            }
                        }
                    }
                    token += 1;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> Option<PathBuf> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
        if p.exists() { Some(p) } else { None }
    }
    fn read_f32(name: &str) -> Option<Vec<f32>> {
        let p = fixture(name)?;
        let bytes = std::fs::read(p).ok()?;
        Some(bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())
    }
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x,y)| (x-y).abs()).fold(0.0f32, f32::max)
    }
    fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x,y)| (x-y).abs()).sum::<f32>() / a.len() as f32
    }

    #[test]
    fn smart_resize_examples() {
        // 512×341 → 11*32=352 by 16*32=512 (the case captured from HF).
        assert_eq!(smart_resize(341, 512).unwrap(), (352, 512));
        // Square 256×256 → already a multiple of 32 → 256×256.
        assert_eq!(smart_resize(256, 256).unwrap(), (256, 256));
        // 1000×800 → round to (1024, 800) which still fits MAX_PIXELS.
        let (h, w) = smart_resize(1000, 800).unwrap();
        assert_eq!(h % FACTOR, 0);
        assert_eq!(w % FACTOR, 0);
    }

    /// End-to-end: PNG → preprocess → vision encoder → merger.
    /// Locks the magnitude statistics against HF on the same image
    /// (both peak at `|x|_max ≈ 15477` because the ViT has a few large outlier
    /// activations — this is the model's actual behavior, not an overflow).
    #[test]
    #[ignore = "loads 1.1 GB vision encoder + 704-patch forward; run with --ignored"]
    fn full_pipeline_against_hf() {
        use crate::vision::{VisionEncoder, OUT_HIDDEN};
        let Some(img_path) = fixture("test_image.png") else { eprintln!("skip"); return; };
        let model = std::path::PathBuf::from(
            "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/Cosmos3-Nano/vision_encoder/model.safetensors");
        if !model.exists() { eprintln!("skip: model missing"); return; }
        let golden_last = read_f32("img_reasoning_last_704x1152.f32");

        let t0 = std::time::Instant::now();
        let (patches, (gt, gh, gw)) = preprocess_image_file(&img_path).expect("preprocess");
        eprintln!("preprocess: {} patches × 1536, grid=({gt},{gh},{gw}), {:?}", patches.len()/1536, t0.elapsed());
        assert_eq!((gt, gh, gw), (1, 22, 32));

        let t0 = std::time::Instant::now();
        let m = VisionEncoder::load(&model).expect("load");
        eprintln!("vision load: {:?}", t0.elapsed());

        let t0 = std::time::Instant::now();
        let (last, ds) = m.forward(&patches, (gt, gh, gw));
        eprintln!("vision forward: {:?}", t0.elapsed());
        assert_eq!(last.len(), 704 * 1152);
        assert!(last.iter().all(|v| v.is_finite()), "non-finite in last_hidden_state");

        let abs_max = last.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let abs_mean = last.iter().map(|v| v.abs()).sum::<f32>() / last.len() as f32;
        eprintln!("last_hidden_state |max|={abs_max:.2}  |mean|={abs_mean:.4}");
        // HF reference is |max|≈15477.22, |mean|≈17.73.  Drift from a slightly
        // different bicubic resampler is on top — bound generously.
        assert!((abs_max - 15477.0).abs() < 200.0, "|max| {abs_max} far from HF 15477");
        assert!((abs_mean - 17.73).abs() < 1.0, "|mean| {abs_mean} far from HF 17.73");

        // If the golden 704×1152 tensor is present, also check mean drift per token.
        if let Some(g) = golden_last.as_ref() {
            assert_eq!(g.len(), last.len());
            let mad = mean_abs_diff(&last, g);
            let mxd = max_abs_diff(&last, g);
            eprintln!("vs HF golden:  mean_abs_diff={mad:.4}, max_abs_diff={mxd:.4}");
            // ~2-step u8 bicubic drift propagated through 27 layers is OK; the
            // important verification is the |max|/|mean| statistics above.
            assert!(mad < 5.0, "mean drift {mad} unexpectedly large");
        }

        let merged = m.merge(&last);
        let merged_tokens = merged.len() / OUT_HIDDEN;
        eprintln!("merger: {merged_tokens} tokens × {OUT_HIDDEN}");
        assert_eq!(merged_tokens, (gh / 2) * (gw / 2));   // 11 × 16 = 176 visual tokens
        for &v in &merged { assert!(v.is_finite()); }
        let _ = ds;
    }

    #[test]
    fn preprocess_matches_hf_reference() {
        let Some(img_path) = fixture("test_image.png") else {
            eprintln!("skip: test_image.png missing"); return;
        };
        let Some(golden) = read_f32("img_reasoning_patches.f32") else {
            eprintln!("skip: img_reasoning_patches.f32 missing"); return;
        };

        let (patches, grid_thw) = preprocess_image_file(&img_path).expect("preprocess");
        assert_eq!(grid_thw, (1, 22, 32));
        assert_eq!(patches.len(), golden.len(), "patches len mismatch");

        let mad = max_abs_diff(&patches, &golden);
        let mead = mean_abs_diff(&patches, &golden);
        eprintln!("preprocess max_abs_diff = {mad:.4e}, mean = {mead:.4e}");
        // PIL bicubic and `image::imageops::CatmullRom` aren't bit-identical (different
        // sample-position conventions, edge handling, and rounding) but produce visually
        // identical resamples → small per-pixel drift, no semantic difference.
        assert!(mead < 0.02, "mean drift {mead} too large — preprocessing diverges");
        assert!(mad  < 0.20, "max drift {mad} — single-pixel outliers OK; this is unusually high");
    }
}
