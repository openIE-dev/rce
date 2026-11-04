//! End-to-end Wan 2.2 TI2V smoke test: real-prompt text → video pixels.
//!
//! Composes the three Wan 2.2 components shipped this session:
//! - `rce-umt5-xxl-encoder`: text → [T_text, 4096] context
//! - `rce-wan22-ti2v`: latent + context + timestep → predicted velocity (ε)
//! - `rce-cosmos3::phase4_v2::Wan22VAE_v2`: latent ↔ pixel video
//!
//! Pipeline:
//! 1. `Umt5Tokenizer::encode(prompt)` → token ids
//! 2. `Umt5XxlEncoder::encode_ids(ids)` → [T_text, 4096] text context
//! 3. Initialize a noisy 48-channel latent `x_T [48, 1, 4, 4]`
//! 4. `Wan22Ti2v::forward_with_rope(x_T, text_ctx, t=0.5)` → predicted velocity
//! 5. Single Euler step: `x_0 = x_T - dt · ε` (dt = 0.5)
//! 6. `Wan22VAE_v2::decode(x_0, 1, 4, 4)` → video pixels `[12, 4, 32, 32]`
//!
//! All three models loaded with 2-layer subsets (load fast + keep RAM
//! tractable).  Numerical output is **not** in agreement with the reference
//! Wan pipeline (subset layers + no calibration), but the shape closure
//! and finiteness prove the integration is wired correctly.

use std::path::{Path, PathBuf};

use rce_wan22_ti2v::*;
use rce_umt5_xxl_encoder::{Umt5Tokenizer, Umt5XxlEncoder};
use rce_cosmos3::phase4_v2::Wan22VAE_v2;

fn wan22_dir() -> Option<PathBuf> {
    for vol in ["macos_4TB_external", "macos_4TB_external_1", "macos_4TB_external_2"] {
        let p = PathBuf::from(format!(
            "/Volumes/{vol}/neuronexus_ai/model_library/Wan-AI_Wan2.2-TI2V-5B"));
        if p.join("diffusion_pytorch_model.safetensors.index.json").exists()
            && p.join("umt5_xxl_encoder.safetensors").exists()
            && p.join("google/umt5-xxl/tokenizer.json").exists() {
            return Some(p);
        }
    }
    None
}

fn wan_vae_path() -> Option<PathBuf> {
    // The Wan 2.2 VAE used by the Wan22Ti2v model is the same Wan 2.2 VAE
    // shipped inside the Cosmos 3 Nano bundle (Wan-AI/Wan2.2-TI2V-5B-Diffusers
    // VAE).  Reuse that file directly.
    for vol in ["macos_4TB_external", "macos_4TB_external_1", "macos_4TB_external_2"] {
        let p = PathBuf::from(format!(
            "/Volumes/{vol}/neuronexus_ai/model_library/Cosmos3-Nano/vae/diffusion_pytorch_model.safetensors"));
        if p.exists() { return Some(p); }
    }
    None
}

#[test]
#[ignore = "Wan 2.2 T2V end-to-end — needs transformer + UMT5 + Wan VAE shards mounted"]
fn wan22_t2v_end_to_end_smoke() {
    let Some(dir) = wan22_dir() else {
        eprintln!("Wan 2.2 TI2V-5B + UMT5 dir not found"); return;
    };
    let Some(vae_path) = wan_vae_path() else {
        eprintln!("Wan 2.2 VAE shard not found"); return;
    };

    // ── 1. Load all three models (subset layers for speed) ──
    let t_load = std::time::Instant::now();
    let tokenizer = Umt5Tokenizer::load_from_dir(&dir).expect("UMT5 tokenizer");
    let umt5 = Umt5XxlEncoder::load_with_layers(
        &dir.join("umt5_xxl_encoder.safetensors"), 2)
        .expect("UMT5 encoder");
    let wan = Wan22Ti2v::load_with_layers(&dir, 2).expect("Wan 2.2 TI2V");
    let vae = Wan22VAE_v2::load(&vae_path).expect("Wan 2.2 VAE");
    eprintln!("load (UMT5 + Wan22 + VAE, 2 layers each): {:?}", t_load.elapsed());

    // ── 2. Tokenize + encode the prompt ──
    let prompt = "A jazz piano melody";
    let ids = tokenizer.encode(prompt).expect("tokenize");
    let t_text = ids.len();
    eprintln!("tokenize {prompt:?} → {ids:?} ({} tokens)", t_text);
    let t0 = std::time::Instant::now();
    let text_ctx = umt5.encode_ids(&ids, 1e-6);
    eprintln!("UMT5 encode: {:?} ({} → [{}, 4096])", t0.elapsed(), t_text, t_text);
    assert_eq!(text_ctx.len(), t_text * 4096);

    // ── 3. Initial noisy latent at the VAE-compatible shape ──
    let (t, h, w) = (1usize, 4usize, 4usize);
    let x_t: Vec<f32> = (0..IN_DIM * t * h * w)
        .map(|i| ((i as f32) * 0.013).sin() * 0.1).collect();

    // ── 4. Wan 2.2 transformer forward (one diffusion step) ──
    let freqs = RopeFreqs3D::new();
    let t0 = std::time::Instant::now();
    let velocity = wan.forward_with_rope(
        &x_t, t, h, w,
        &text_ctx, t_text,
        0.5,        // timestep
        1e-6,       // eps
        &freqs);
    let dt_wan = t0.elapsed();
    assert_eq!(velocity.len(), OUT_DIM * t * h * w);
    assert!(velocity.iter().all(|v| v.is_finite()));
    eprintln!("Wan22Ti2v forward: {dt_wan:?} → velocity [{},{},{},{}]", OUT_DIM, t, h, w);

    // ── 5. Single Euler step toward x_0 ──
    let dt_step = 0.5;
    let mut x_0 = x_t.clone();
    for i in 0..x_0.len() { x_0[i] -= dt_step * velocity[i]; }

    // ── 6. Decode through the VAE ──
    let t0 = std::time::Instant::now();
    let (video, td, hd, wd) = vae.decode(&x_0, t, h, w);
    let dt_dec = t0.elapsed();
    eprintln!("Wan VAE decode: {dt_dec:?} → video [{},{},{},{}]",
              vae.config.out_channels, td, hd, wd);

    // ── 7. Verify shape closure + finiteness ──
    assert!(video.iter().all(|v| v.is_finite()));
    // Latent 1×4×4 → VAE: time-pixel-shuffle (×4) and 3 spatial upsamplers (×8 each) →
    // pixel video: T=4, H=32, W=32 with 12 channels (3 RGB × 4 temporal patches).
    assert_eq!((td, hd, wd), (4, 32, 32),
               "VAE should restore latent grid to pixel video shape");
    assert_eq!(vae.config.out_channels, 12);

    eprintln!("Wan 2.2 T2V end-to-end ✓ — total {:?}", t_load.elapsed());
}

/// Multi-step UniPC AB2 sampling end-to-end.  Drives Wan22Ti2v through
/// 4 steps of the production linear σ schedule, then decodes via the VAE.
/// Verifies the multi-step trajectory produces finite output and that
/// the σ-driven integration actually changes the latent.
#[test]
#[ignore = "Wan 2.2 T2V multi-step — needs all shards mounted; ~30s wall"]
fn wan22_t2v_unipc_4step_smoke() {
    let Some(dir) = wan22_dir() else {
        eprintln!("Wan 2.2 TI2V-5B + UMT5 dir not found"); return;
    };
    let Some(vae_path) = wan_vae_path() else {
        eprintln!("Wan 2.2 VAE shard not found"); return;
    };

    let t_load = std::time::Instant::now();
    let tokenizer = Umt5Tokenizer::load_from_dir(&dir).expect("UMT5 tokenizer");
    let umt5 = Umt5XxlEncoder::load_with_layers(
        &dir.join("umt5_xxl_encoder.safetensors"), 2)
        .expect("UMT5 encoder");
    let wan = Wan22Ti2v::load_with_layers(&dir, 2).expect("Wan 2.2 TI2V");
    let vae = Wan22VAE_v2::load(&vae_path).expect("Wan 2.2 VAE");
    eprintln!("load: {:?}", t_load.elapsed());

    let prompt = "A jazz piano melody";
    let ids = tokenizer.encode(prompt).expect("tokenize");
    let t_text = ids.len();
    let text_ctx = umt5.encode_ids(&ids, 1e-6);

    let (t, h, w) = (1usize, 4usize, 4usize);
    let x_init: Vec<f32> = (0..IN_DIM * t * h * w)
        .map(|i| ((i as f32) * 0.013).sin() * 0.1).collect();
    let freqs = RopeFreqs3D::new();

    let schedule = FlowMatchSchedule::linear(4);
    let sigmas = schedule.sigmas();
    eprintln!("schedule sigmas: {sigmas:?}");

    // UniPC AB2 sampling.
    let t0 = std::time::Instant::now();
    let x_0 = wan.unipc_sample_scheduled(
        x_init.clone(), t, h, w,
        &text_ctx, t_text,
        &schedule, 1e-6, &freqs);
    let dt_samp = t0.elapsed();
    assert!(x_0.iter().all(|v| v.is_finite()));
    let max_diff = x_init.iter().zip(x_0.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let init_norm: f32 = x_init.iter().map(|v| v*v).sum::<f32>().sqrt();
    let final_norm: f32 = x_0.iter().map(|v| v*v).sum::<f32>().sqrt();
    eprintln!("UniPC 4-step: {dt_samp:?}  ‖x_T‖={init_norm:.4} → ‖x_0‖={final_norm:.4}  max_diff={max_diff:.4}");
    assert!(max_diff > 1e-3, "multi-step UniPC must drive the latent away from init");

    // Decode the sampled latent.
    let t0 = std::time::Instant::now();
    let (video, td, hd, wd) = vae.decode(&x_0, t, h, w);
    eprintln!("VAE decode: {:?} → video [{},{},{},{}]",
              t0.elapsed(), vae.config.out_channels, td, hd, wd);
    assert!(video.iter().all(|v| v.is_finite()));
    assert_eq!((td, hd, wd), (4, 32, 32));
    eprintln!("Wan 2.2 T2V UniPC 4-step ✓ — total {:?}", t_load.elapsed());
}

/// Frame export: drive the full pipeline through 4 UniPC steps, take the
/// first 3 channels of the first temporal slice as RGB, and write a PPM
/// image to `/tmp/wan22_frame0.ppm` for visual inspection.  Provides a
/// user-visible artifact instead of just shape assertions.
#[test]
#[ignore = "Wan 2.2 T2V frame export — needs all shards mounted; writes /tmp/wan22_frame0.ppm"]
fn wan22_t2v_export_first_frame() {
    let Some(dir) = wan22_dir() else { eprintln!("Wan 2.2 dir not found"); return; };
    let Some(vae_path) = wan_vae_path() else { eprintln!("VAE not found"); return; };
    let tokenizer = Umt5Tokenizer::load_from_dir(&dir).expect("UMT5 tokenizer");
    let umt5 = Umt5XxlEncoder::load_with_layers(
        &dir.join("umt5_xxl_encoder.safetensors"), 2)
        .expect("UMT5 encoder");
    let wan = Wan22Ti2v::load_with_layers(&dir, 2).expect("Wan 2.2 TI2V");
    let vae = Wan22VAE_v2::load(&vae_path).expect("Wan 2.2 VAE");

    let prompt = "A jazz piano melody";
    let ids = tokenizer.encode(prompt).expect("tokenize");
    let t_text = ids.len();
    let text_ctx = umt5.encode_ids(&ids, 1e-6);
    let (t, h, w) = (1usize, 4usize, 4usize);
    let x_init: Vec<f32> = (0..IN_DIM * t * h * w)
        .map(|i| ((i as f32) * 0.013).sin() * 0.1).collect();
    let freqs = RopeFreqs3D::new();
    let schedule = FlowMatchSchedule::linear(4);
    let x_0 = wan.unipc_sample_scheduled(
        x_init, t, h, w, &text_ctx, t_text, &schedule, 1e-6, &freqs);

    let (video, td, hd, wd) = vae.decode(&x_0, t, h, w);
    eprintln!("Wan VAE decoded video: out_channels={}, [T,H,W]=[{},{},{}]",
              vae.config.out_channels, td, hd, wd);

    // Extract the first 3 channels at temporal slice 0 as RGB.
    // Layout: [out_channels=12, T=4, H=32, W=32] row-major.
    let oc = vae.config.out_channels;
    let plane = td * hd * wd;
    let mut rgb = vec![0u8; hd * wd * 3];
    // Find dynamic range for normalization.
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for c in 0..3 {
        for yi in 0..hd {
            for xi in 0..wd {
                let v = video[c * plane + 0 * (hd * wd) + yi * wd + xi];
                if v < mn { mn = v; }
                if v > mx { mx = v; }
            }
        }
    }
    let range = (mx - mn).max(1e-6);
    eprintln!("frame 0 RGB dynamic range: [{mn:.4}, {mx:.4}] (span {range:.4})");
    for c in 0..3 {
        for yi in 0..hd {
            for xi in 0..wd {
                let v = video[c * plane + 0 * (hd * wd) + yi * wd + xi];
                let n = ((v - mn) / range).clamp(0.0, 1.0);
                rgb[(yi * wd + xi) * 3 + c] = (n * 255.0).round() as u8;
            }
        }
    }
    let path = "/tmp/wan22_frame0.ppm";
    let mut f = std::fs::File::create(path).expect("create ppm");
    use std::io::Write;
    write!(f, "P6\n{wd} {hd}\n255\n").expect("ppm header");
    f.write_all(&rgb).expect("ppm bytes");
    eprintln!("wrote {path} (PPM, {} bytes, {wd}×{hd} RGB)", rgb.len() + 32);
}
