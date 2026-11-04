//! End-to-end Stable Diffusion 1.5 text-to-image pipeline on rce substrate.
//!
//! Composes the three numerically-validated SD 1.5 components:
//! 1. `rce-clip-vit-l::ClipVitL::text` — text encoder (last_hidden_state [77, 768])
//! 2. `rce-sd15-unet::Sd15Unet::forward` — diffusion U-Net (validated rel=0.10% vs diffusers)
//! 3. `rce-sd15-vae::Sd15Vae::decode` — VAE decoder (validated rel=0.00% vs diffusers)
//!
//! Pipeline:
//! ```text
//! prompt + neg_prompt → CLIP tokenizer
//!     → encode to [77, 768] each (padded with EOT to 77)
//!     → initialize noise latent [4, h_lat, w_lat]
//!     → for each timestep (DDIM, classifier-free guidance):
//!         eps_uncond = unet(latent, t, neg_ctx)
//!         eps_cond   = unet(latent, t, prompt_ctx)
//!         eps        = eps_uncond + cfg_scale · (eps_cond − eps_uncond)
//!         latent     = ddim_step(latent, eps, t, t_prev)
//!     → SD VAE decode([4, h_lat, w_lat]) → image [3, h_lat·8, w_lat·8]
//!     → write to PNG
//! ```
//!
//! Smoke test runs at 4×8×8 latent (= 64×64 image) with 2 sampling steps so wall
//! time stays under a minute; production resolution is 4×64×64 (= 512×512 image).

use std::path::{Path, PathBuf};
use rce_sd15_unet::*;
use rce_clip_vit_l::{ClipVitL, ClipTokenizer, T_MAX_SEQ, T_DIM, TOK_EOT};
use rce_sd15_vae::Sd15Vae;

fn sd_root() -> Option<PathBuf> {
    for vol in ["macos_4TB_external", "macos_4TB_external_1", "macos_4TB_external_2"] {
        let p = PathBuf::from(format!(
            "/Volumes/{vol}/neuronexus_ai/model_library/stable-diffusion-v1-5"));
        if p.join("unet/diffusion_pytorch_model.safetensors").exists()
            && p.join("vae/diffusion_pytorch_model.safetensors").exists() {
            return Some(p);
        }
    }
    None
}

fn clip_path() -> Option<PathBuf> {
    for vol in ["macos_4TB_external", "macos_4TB_external_1", "macos_4TB_external_2"] {
        let p = PathBuf::from(format!(
            "/Volumes/{vol}/neuronexus_ai/model_library/openai_clip-vit-large-patch14"));
        if p.join("model.safetensors").exists() && p.join("tokenizer.json").exists() {
            return Some(p);
        }
    }
    None
}

/// Pad a token id sequence to `T_MAX_SEQ=77` with the CLIP EOT id (49407),
/// the standard SD 1.5 convention for the cross-attention conditioning.
fn pad_to_max(ids: Vec<u32>) -> Vec<u32> {
    let mut out = ids;
    while out.len() < T_MAX_SEQ {
        out.push(TOK_EOT);
    }
    out.truncate(T_MAX_SEQ);
    out
}

/// One DDIM step: linear-σ rectified-flow style.  Wan22 / Cosmos3 pattern.
fn ddim_step(latent: &mut [f32], eps: &[f32], dt: f32) {
    for i in 0..latent.len() { latent[i] -= dt * eps[i]; }
}

#[test]
#[ignore = "SD 1.5 T2I end-to-end — needs all 3 SD components mounted; ~2 min wall"]
fn sd15_t2i_end_to_end() {
    let Some(sd) = sd_root() else { eprintln!("SD 1.5 not mounted"); return; };
    let Some(clip) = clip_path() else { eprintln!("CLIP not mounted"); return; };

    let t_load = std::time::Instant::now();
    let clip_model = ClipVitL::load_with_layers(&clip.join("model.safetensors"), 0, 12)
        .expect("load CLIP (text only)");
    let tk = ClipTokenizer::load_from_dir(&clip).expect("CLIP tokenizer");
    let unet = Sd15Unet::load(&sd.join("unet/diffusion_pytorch_model.safetensors"))
        .expect("load UNet");
    let vae = Sd15Vae::load(&sd.join("vae/diffusion_pytorch_model.safetensors"))
        .expect("load VAE");
    eprintln!("load CLIP+UNet+VAE: {:?}", t_load.elapsed());

    // ── Tokenize + encode prompt and negative ──
    let prompt = "a photo of an astronaut riding a horse";
    let neg = "";
    let prompt_ids = pad_to_max(tk.encode(prompt).expect("encode"));
    let neg_ids = pad_to_max(tk.encode(neg).expect("encode neg"));
    let t0 = std::time::Instant::now();
    let prompt_ctx = clip_model.text.encode_ids_sequence(&prompt_ids, 1e-5);
    let neg_ctx = clip_model.text.encode_ids_sequence(&neg_ids, 1e-5);
    eprintln!("CLIP text encode (×2): {:?}", t0.elapsed());
    assert_eq!(prompt_ctx.len(), T_MAX_SEQ * T_DIM);
    assert_eq!(neg_ctx.len(), T_MAX_SEQ * T_DIM);

    // ── Initialize noise latent ──
    let (h_lat, w_lat) = (8usize, 8usize);
    let mut latent: Vec<f32> = (0..4 * h_lat * w_lat)
        .map(|i| ((i as f32) * 0.13).sin()).collect();
    let init_norm: f32 = latent.iter().map(|v| v*v).sum::<f32>().sqrt();

    // ── 2 DDIM-style sampling steps with classifier-free guidance ──
    let n_steps = 2;
    let cfg_scale = 7.5_f32;
    let mut t_curr = 1000.0_f32;
    for step in 0..n_steps {
        let t_next = ((n_steps - step - 1) as f32 / n_steps as f32) * 1000.0;
        let dt = (t_curr - t_next) / 1000.0;
        let t0 = std::time::Instant::now();
        let eps_uncond = unet.forward(&latent, h_lat, w_lat, t_curr, &neg_ctx, T_MAX_SEQ, 1e-5);
        let eps_cond = unet.forward(&latent, h_lat, w_lat, t_curr, &prompt_ctx, T_MAX_SEQ, 1e-5);
        let dt_unet = t0.elapsed();
        let mut eps = vec![0.0f32; eps_uncond.len()];
        for i in 0..eps.len() {
            eps[i] = eps_uncond[i] + cfg_scale * (eps_cond[i] - eps_uncond[i]);
        }
        ddim_step(&mut latent, &eps, dt);
        eprintln!("step {step}: t={t_curr:.1}→{t_next:.1} dt={dt:.3} unet ×2 in {dt_unet:?} ‖latent‖={:.3}",
                  latent.iter().map(|v| v*v).sum::<f32>().sqrt());
        t_curr = t_next;
    }
    let final_norm: f32 = latent.iter().map(|v| v*v).sum::<f32>().sqrt();
    eprintln!("sampling: ‖x_T‖={init_norm:.3} → ‖x_0‖={final_norm:.3}");

    // ── VAE decode ──
    let t0 = std::time::Instant::now();
    let img = vae.decode(&latent, h_lat, w_lat, 1e-6);
    eprintln!("VAE decode 4×{h_lat}×{w_lat} → 3×{}×{}: {:?}",
              h_lat * 8, w_lat * 8, t0.elapsed());
    assert_eq!(img.len(), 3 * h_lat * 8 * w_lat * 8);
    assert!(img.iter().all(|v| v.is_finite()));

    // ── Save first frame as PNG ──
    let h_img = h_lat * 8;
    let w_img = w_lat * 8;
    let plane = h_img * w_img;
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for v in &img { if *v < mn { mn = *v; } if *v > mx { mx = *v; } }
    let mut rgb = vec![0u8; plane * 3];
    for y in 0..h_img {
        for x in 0..w_img {
            for c in 0..3 {
                let v = img[c * plane + y * w_img + x];
                let n = ((v - mn) / (mx - mn).max(1e-6)).clamp(0.0, 1.0);
                rgb[(y * w_img + x) * 3 + c] = (n * 255.0).round() as u8;
            }
        }
    }
    use std::io::Write;
    let png_path = "/tmp/sd15_t2i_output.png";
    let mut buf = image::RgbImage::new(w_img as u32, h_img as u32);
    for y in 0..h_img {
        for x in 0..w_img {
            let off = (y * w_img + x) * 3;
            buf.put_pixel(x as u32, y as u32, image::Rgb([rgb[off], rgb[off+1], rgb[off+2]]));
        }
    }
    buf.save(png_path).expect("write png");
    // Also raw PPM as a fallback.
    let ppm_path = "/tmp/sd15_t2i_output.ppm";
    let mut f = std::fs::File::create(ppm_path).expect("create ppm");
    write!(f, "P6\n{w_img} {h_img}\n255\n").expect("ppm hdr");
    f.write_all(&rgb).expect("ppm body");
    eprintln!("dynamic range: [{mn:.4}, {mx:.4}]");
    eprintln!("wrote {png_path} ({w_img}×{h_img} RGB)");
    eprintln!("SD 1.5 T2I end-to-end ✓ — total wall {:?}", t_load.elapsed());
}
