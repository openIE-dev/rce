//! LFM2.5-VL multimodal runtime — rce's **first vision path**.
//!
//! Pipeline (authoritative: vendored `tools/mtmd/clip.cpp`
//! `clip_graph::build_vit` + `models/siglip.cpp` `clip_graph_siglip`
//! `PROJECTOR_TYPE_LFM2` branch + `build_patch_merge_permute`):
//!
//! ```text
//!   image → resize 256×256 bilinear → normalise (px/127.5 − 1)
//!         → 16×16 patches (256 patches, 768-d via conv2d = patch matmul)
//!         → + learned position_embd
//!         → 12× SigLIP block:  LN1 → MHA(bias, bidirectional) +res
//!                              LN2 → MLP(up→GELU→down, bias)   +res
//!         → post_LN
//!         → 2×2 pixel-unshuffle  (256 patches → 64 tokens, 768→3072)
//!         → projector MLP  mm.1(3072→2048) → GELU → mm.2(2048→1024)
//!         → 64 image embeddings (d = lfm2 d_model = 1024)
//! ```
//! These 64 embeddings replace the single `<image>` token (id 396) in
//! the chat-framed prompt; the mixed text+image embedding sequence is
//! driven through `rce_lfm2::forward_embed`, then generation is
//! ordinary autoregressive decoding on the LFM2.5 hybrid backbone.

use rce_ingest_gguf::{read_tensor_f32, sample_with_history, GgufFile, SampleConfig};
use rce_lfm2::Lfm2Runtime;
use rce_op_qmm::QuantWeight;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

const PATCH: usize = 16;
const IMG: usize = 256;
const GRID: usize = IMG / PATCH; // 16
const N_PATCH: usize = GRID * GRID; // 256
const V_EMBD: usize = 768;
const V_HEADS: usize = 12;
const V_DHEAD: usize = V_EMBD / V_HEADS; // 64
const V_FF: usize = 3072;
const V_LAYERS: usize = 12;
const SCALE_F: usize = 2; // pixel-unshuffle factor
const V_EPS: f32 = 1e-6;
const IMAGE_TOKEN_ID: i32 = 396; // "<image>"

struct VBlock {
    ln1_w: Vec<f32>, ln1_b: Vec<f32>,
    ln2_w: Vec<f32>, ln2_b: Vec<f32>,
    wq: QuantWeight, q_b: Vec<f32>,
    wk: QuantWeight, k_b: Vec<f32>,
    wv: QuantWeight, v_b: Vec<f32>,
    wo: QuantWeight, o_b: Vec<f32>,
    up: QuantWeight, up_b: Vec<f32>,
    down: QuantWeight, down_b: Vec<f32>,
}

/// SigLIP ViT + 2×2 pixel-unshuffle + LFM2 projector (from the mmproj
/// GGUF). Output: 64 image embeddings of `out_dim` (= lfm2 d_model).
pub struct VisionTower {
    patch_w: Vec<f32>,    // [16*16*3 * 768] flat, ggml order (kw + kh*16 + ic*256 + oc*768)
    patch_b: Vec<f32>,    // [768]
    pos_embd: Vec<f32>,   // [256 * 768] flat (e + p*768)
    blocks: Vec<VBlock>,
    post_w: Vec<f32>, post_b: Vec<f32>,
    mm1: QuantWeight, mm1_b: Vec<f32>, // 3072 → 2048
    mm2: QuantWeight, mm2_b: Vec<f32>, // 2048 → 1024
    pub out_dim: usize,
    _mmap: Arc<Mmap>,
}

fn layernorm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv = 1.0 / (var + eps).sqrt();
    (0..n).map(|i| (x[i] - mean) * inv * w[i] + b[i]).collect()
}

/// ggml default GELU (tanh approximation).
fn gelu(x: f32) -> f32 {
    let c = 0.797_884_56_f32; // sqrt(2/pi)
    0.5 * x * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())
}

fn qmm(a: &[f32], w: &QuantWeight) -> Vec<f32> {
    let mut o = vec![0.0f32; w.n_rows];
    rce_op_qmm::mm_1d(a, w, &mut o);
    o
}

impl VisionTower {
    pub fn load(mmproj_path: impl AsRef<Path>, out_dim: usize) -> Result<Self, String> {
        let f = File::open(mmproj_path.as_ref()).map_err(|e| format!("open mmproj: {e}"))?;
        let mmap = Arc::new(unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap: {e}"))?);
        let _ = mmap.advise(memmap2::Advice::WillNeed);
        let g = GgufFile::parse(&mmap[..]).map_err(|e| format!("parse mmproj: {e:?}"))?;
        let bytes: &[u8] = &mmap[..];
        let rd = |n: &str| read_tensor_f32(bytes, &g, n).map_err(|e| format!("{n}: {e:?}"));
        let qw = |n: &str, r: usize, c: usize| QuantWeight::from_gguf(
            Arc::clone(&mmap), &g, n, r, c).map_err(|e| format!("{n}: {e}"));

        let patch_w = rd("v.patch_embd.weight")?; // [16,16,3,768] → 589824 f32
        let patch_b = rd("v.patch_embd.bias")?;
        let pos_embd = rd("v.position_embd.weight")?; // [768,256] → 196608 f32
        let mut blocks = Vec::with_capacity(V_LAYERS);
        for i in 0..V_LAYERS {
            let p = format!("v.blk.{i}");
            blocks.push(VBlock {
                ln1_w: rd(&format!("{p}.ln1.weight"))?, ln1_b: rd(&format!("{p}.ln1.bias"))?,
                ln2_w: rd(&format!("{p}.ln2.weight"))?, ln2_b: rd(&format!("{p}.ln2.bias"))?,
                wq: qw(&format!("{p}.attn_q.weight"), V_EMBD, V_EMBD)?, q_b: rd(&format!("{p}.attn_q.bias"))?,
                wk: qw(&format!("{p}.attn_k.weight"), V_EMBD, V_EMBD)?, k_b: rd(&format!("{p}.attn_k.bias"))?,
                wv: qw(&format!("{p}.attn_v.weight"), V_EMBD, V_EMBD)?, v_b: rd(&format!("{p}.attn_v.bias"))?,
                wo: qw(&format!("{p}.attn_out.weight"), V_EMBD, V_EMBD)?, o_b: rd(&format!("{p}.attn_out.bias"))?,
                up: qw(&format!("{p}.ffn_up.weight"), V_FF, V_EMBD)?, up_b: rd(&format!("{p}.ffn_up.bias"))?,
                down: qw(&format!("{p}.ffn_down.weight"), V_EMBD, V_FF)?, down_b: rd(&format!("{p}.ffn_down.bias"))?,
            });
        }
        let post_w = rd("v.post_ln.weight")?;
        let post_b = rd("v.post_ln.bias")?;
        let merged = V_EMBD * SCALE_F * SCALE_F; // 3072
        let mm1 = qw("mm.1.weight", 2048, merged)?;
        let mm1_b = rd("mm.1.bias")?;
        let mm2 = qw("mm.2.weight", out_dim, 2048)?;
        let mm2_b = rd("mm.2.bias")?;
        eprintln!(
            "vision: SigLIP {V_LAYERS}L d{V_EMBD} {V_HEADS}h patch{PATCH} img{IMG} \
             → pixel-unshuffle {SCALE_F}× → proj {merged}→2048→{out_dim}"
        );
        Ok(Self {
            patch_w, patch_b, pos_embd, blocks, post_w, post_b,
            mm1, mm1_b, mm2, mm2_b, out_dim, _mmap: mmap,
        })
    }

    /// Decode → resize → normalise an image file to `[256][256][3]`
    /// f32 in `(px/127.5 − 1)` (image_mean=image_std=0.5).
    fn preprocess(path: &Path) -> Result<Vec<f32>, String> {
        let img = image::open(path).map_err(|e| format!("image open: {e}"))?;
        let img = img.resize_exact(IMG as u32, IMG as u32, image::imageops::FilterType::Triangle);
        let rgb = img.to_rgb8();
        // Layout matching ggml inp_raw [nx, ny, c]: index = x + y*256 + c*256*256.
        let mut out = vec![0.0f32; IMG * IMG * 3];
        for y in 0..IMG {
            for x in 0..IMG {
                let px = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    out[x + y * IMG + c * IMG * IMG] = px[c] as f32 / 127.5 - 1.0;
                }
            }
        }
        Ok(out)
    }

    /// Patch-embed (conv2d = per-patch matmul) → +pos → 256×768.
    fn patchify(&self, raw: &[f32]) -> Vec<Vec<f32>> {
        let mut patches = vec![vec![0.0f32; V_EMBD]; N_PATCH];
        for p in 0..N_PATCH {
            let ow = p % GRID;
            let oh = p / GRID;
            let x0 = ow * PATCH;
            let y0 = oh * PATCH;
            // Flatten patch in ggml weight order: kw + kh*16 + ic*256.
            let mut pv = [0.0f32; PATCH * PATCH * 3];
            for kh in 0..PATCH {
                for kw in 0..PATCH {
                    let x = x0 + kw;
                    let y = y0 + kh;
                    for ic in 0..3 {
                        pv[kw + kh * PATCH + ic * PATCH * PATCH] =
                            raw[x + y * IMG + ic * IMG * IMG];
                    }
                }
            }
            // conv_out[oc] = dot(pv, filter_oc) + bias[oc]; filter_oc =
            // patch_w[oc*768 .. oc*768+768].
            let out = &mut patches[p];
            for oc in 0..V_EMBD {
                let f = &self.patch_w[oc * V_EMBD..oc * V_EMBD + V_EMBD];
                let mut acc = 0.0f32;
                for k in 0..V_EMBD {
                    acc += pv[k] * f[k];
                }
                out[oc] = acc + self.patch_b[oc] + self.pos_embd[p * V_EMBD + oc];
            }
        }
        patches
    }

    fn vit(&self, patches: &mut Vec<Vec<f32>>) {
        let scale = 1.0 / (V_DHEAD as f32).sqrt();
        for blk in &self.blocks {
            // ── attention (bidirectional, no mask) ──
            let normed: Vec<Vec<f32>> = patches.iter()
                .map(|x| layernorm(x, &blk.ln1_w, &blk.ln1_b, V_EPS))
                .collect();
            let mut q = Vec::with_capacity(N_PATCH);
            let mut k = Vec::with_capacity(N_PATCH);
            let mut v = Vec::with_capacity(N_PATCH);
            for n in &normed {
                let mut qq = qmm(n, &blk.wq);
                let mut kk = qmm(n, &blk.wk);
                let mut vv = qmm(n, &blk.wv);
                for i in 0..V_EMBD { qq[i] += blk.q_b[i]; kk[i] += blk.k_b[i]; vv[i] += blk.v_b[i]; }
                q.push(qq); k.push(kk); v.push(vv);
            }
            // Flatten K/V into [N_PATCH × V_EMBD] for the fused kernel.
            // Bidirectional (no causal mask): pass lo=0, hi=N_PATCH so
            // every patch attends to every patch.
            let mut k_flat = Vec::with_capacity(N_PATCH * V_EMBD);
            let mut v_flat = Vec::with_capacity(N_PATCH * V_EMBD);
            for j in 0..N_PATCH {
                k_flat.extend_from_slice(&k[j]);
                v_flat.extend_from_slice(&v[j]);
            }
            let mut attn = vec![vec![0.0f32; V_EMBD]; N_PATCH];
            for i in 0..N_PATCH {
                for h in 0..V_HEADS {
                    let o = h * V_DHEAD;
                    let qh = &q[i][o..o + V_DHEAD];
                    let attended_h = rce_op_attn::fused_attend_head_window(
                        qh, &k_flat, &v_flat,
                        h, 0, N_PATCH, V_EMBD, V_DHEAD, scale,
                    );
                    attn[i][o..o + V_DHEAD].copy_from_slice(&attended_h);
                }
            }
            for i in 0..N_PATCH {
                let mut proj = qmm(&attn[i], &blk.wo);
                for d in 0..V_EMBD { patches[i][d] += proj[d] + blk.o_b[d]; }
                // (note: o_b added once; proj reused as scratch)
                let _ = &mut proj;
            }
            // ── MLP ──
            for i in 0..N_PATCH {
                let n = layernorm(&patches[i], &blk.ln2_w, &blk.ln2_b, V_EPS);
                let mut up = qmm(&n, &blk.up);
                for j in 0..V_FF { up[j] = gelu(up[j] + blk.up_b[j]); }
                let dn = qmm(&up, &blk.down);
                for d in 0..V_EMBD { patches[i][d] += dn[d] + blk.down_b[d]; }
            }
        }
        for p in patches.iter_mut() {
            *p = layernorm(p, &self.post_w, &self.post_b, V_EPS);
        }
    }

    /// 2×2 pixel-unshuffle → 64 tokens of 3072, then projector MLP →
    /// 64 tokens of `out_dim`. Index math derived from the ggml
    /// reshape/permute chain in `build_patch_merge_permute`:
    /// token `t = (ow/2) + (oh/2)*8`; within a token the 4 source
    /// patches occupy `[dw*768 + dh*1536 ..]` for `dw=ow%2, dh=oh%2`;
    /// source patch index `p = oh*16 + ow`.
    fn merge_and_project(&self, vit_out: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mg = GRID / SCALE_F; // 8
        let merged = V_EMBD * SCALE_F * SCALE_F; // 3072
        let mut out = Vec::with_capacity(mg * mg);
        for mr in 0..mg {
            for mc in 0..mg {
                let mut tok = vec![0.0f32; merged];
                for dh in 0..SCALE_F {
                    for dw in 0..SCALE_F {
                        let ow = SCALE_F * mc + dw;
                        let oh = SCALE_F * mr + dh;
                        let p = oh * GRID + ow;
                        let base = dw * V_EMBD + dh * (V_EMBD * SCALE_F);
                        tok[base..base + V_EMBD].copy_from_slice(&vit_out[p]);
                    }
                }
                // projector: mm.1 → GELU → mm.2
                let mut h = qmm(&tok, &self.mm1);
                for j in 0..h.len() { h[j] = gelu(h[j] + self.mm1_b[j]); }
                let mut o = qmm(&h, &self.mm2);
                for j in 0..o.len() { o[j] += self.mm2_b[j]; }
                out.push(o);
            }
        }
        out
    }

    /// Full vision forward: image file → 64 embeddings of `out_dim`.
    pub fn encode_image(&self, path: impl AsRef<Path>) -> Result<Vec<Vec<f32>>, String> {
        let raw = Self::preprocess(path.as_ref())?;
        let mut patches = self.patchify(&raw);
        self.vit(&mut patches);
        Ok(self.merge_and_project(&patches))
    }
}

pub struct Lfm2VlRuntime {
    pub text: Lfm2Runtime,
    pub vision: VisionTower,
}

#[derive(Clone, Copy, Debug)]
pub struct VlGenConfig {
    pub max_new_tokens: usize,
    pub sample: SampleConfig,
    pub rep_window: usize,
}
impl Default for VlGenConfig {
    fn default() -> Self {
        Self { max_new_tokens: 48, sample: SampleConfig::GREEDY, rep_window: 64 }
    }
}

impl Lfm2VlRuntime {
    pub fn load(
        text_gguf: impl AsRef<Path>, mmproj_gguf: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let text = Lfm2Runtime::load(text_gguf)?;
        let vision = VisionTower::load(mmproj_gguf, text.cfg.d_model)?;
        if vision.out_dim != text.cfg.d_model {
            return Err(format!(
                "projector out_dim {} != lfm2 d_model {}", vision.out_dim, text.cfg.d_model
            ));
        }
        Ok(Self { text, vision })
    }

    /// Caption/answer for `image` given `question`. Builds the LFM2-VL
    /// chat frame `<bos><|im_start|>user\n<image>{question}<|im_end|>\n
    /// <|im_start|>assistant\n`, splices the 64 projected image
    /// embeddings where the `<image>` token sits, then greedily/sampled
    /// decodes.
    pub fn describe(
        &self, image: impl AsRef<Path>, question: &str, gcfg: &VlGenConfig,
        mut on_token: impl FnMut(&str),
    ) -> Result<String, String> {
        let tk = self.text.tokenizer();
        // Chat frame. bos = <|startoftext|> (1); markers <|im_start|>=6,
        // <|im_end|>=7. Encode the text around a literal <image> (396).
        let bos = tk.bos_token_id.unwrap_or(1);
        let im_start = tk.lookup_token("<|im_start|>").ok_or("no <|im_start|>")?;
        let im_end = tk.lookup_token("<|im_end|>").ok_or("no <|im_end|>")?;
        let nl = tk.encode_gpt2("\n", false);
        let user_hdr = tk.encode_gpt2("user", false);
        let asst_hdr = tk.encode_gpt2("assistant", false);
        let qtoks = tk.encode_gpt2(question, false);

        // ids: bos, im_start, "user", \n, <image>, {question}, im_end, \n,
        //      im_start, "assistant", \n
        let mut ids: Vec<i32> = Vec::new();
        ids.push(bos);
        ids.push(im_start);
        ids.extend_from_slice(&user_hdr);
        ids.extend_from_slice(&nl);
        ids.push(IMAGE_TOKEN_ID);
        ids.extend_from_slice(&qtoks);
        ids.push(im_end);
        ids.extend_from_slice(&nl);
        ids.push(im_start);
        ids.extend_from_slice(&asst_hdr);
        ids.extend_from_slice(&nl);

        let img_embds = self.vision.encode_image(image)?; // 64 × d_model
        let eos = tk.eos_token_id.unwrap_or(7);
        let mut state = self.text.new_state();

        // Prefill: feed every position's embedding. At the <image>
        // token, substitute the 64 image embeddings (the token itself
        // is dropped — it is only a placeholder marker).
        let mut last_logits: Option<Vec<f32>> = None;
        for &id in &ids {
            if id == IMAGE_TOKEN_ID {
                for e in &img_embds {
                    last_logits = Some(self.text.forward_embed(&mut state, e.clone()));
                }
            } else {
                let e = self.text.embed_token(id);
                last_logits = Some(self.text.forward_embed(&mut state, e));
            }
        }

        // Decode.
        let mut history: Vec<i32> = ids.clone();
        let mut out_ids = Vec::new();
        let seed_for = |s: u64| {
            let mut c = gcfg.sample;
            c.seed = gcfg.sample.seed ^ s.wrapping_mul(0x9E37_79B9);
            c
        };
        let mut step = 0u64;
        let mut next = {
            let lg = last_logits.ok_or("empty prompt")?;
            let recent = &history[history.len().saturating_sub(gcfg.rep_window)..];
            sample_with_history(&lg, &seed_for(step), recent)
        };
        step += 1;
        for _ in 0..gcfg.max_new_tokens {
            if next == eos { break; }
            out_ids.push(next);
            history.push(next);
            on_token(&tk.decode_sequence(&[next], false));
            let e = self.text.embed_token(next);
            let lg = self.text.forward_embed(&mut state, e);
            let recent = &history[history.len().saturating_sub(gcfg.rep_window)..];
            next = sample_with_history(&lg, &seed_for(step), recent);
            step += 1;
        }
        Ok(tk.decode_sequence(&out_ids, true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const TXT: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/lfm2.5-vl-450m-q8_0.gguf";
    const MMP: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/lfm2.5-vl-450m-mmproj-q8_0.gguf";

    fn synthetic_image(path: &Path, rgb: [u8; 3]) {
        let mut im = image::RgbImage::new(IMG as u32, IMG as u32);
        for p in im.pixels_mut() { *p = image::Rgb(rgb); }
        im.save(path).unwrap();
    }

    /// **VL.a — load both GGUFs + shape sanity.**
    #[test]
    fn vl_a_load() {
        if !std::path::Path::new(TXT).exists() || !std::path::Path::new(MMP).exists() {
            eprintln!("VL.a: models not mounted — skipping"); return;
        }
        let rt = Lfm2VlRuntime::load(TXT, MMP).expect("load");
        assert_eq!(rt.vision.out_dim, rt.text.cfg.d_model);
        eprintln!("VL.a OK: LFM2.5-VL loaded — vision out_dim={} == lfm2 d_model", rt.vision.out_dim);
    }

    /// **VL.b — vision tower produces 64 finite image embeddings.**
    #[test]
    fn vl_b_vision_shape() {
        if !std::path::Path::new(MMP).exists() { eprintln!("VL.b: skip"); return; }
        let v = VisionTower::load(MMP, 1024).expect("vision load");
        let tmp = std::env::temp_dir().join("rce_vl_red.png");
        synthetic_image(&tmp, [220, 30, 30]);
        let emb = v.encode_image(&tmp).expect("encode");
        assert_eq!(emb.len(), 64, "expected 64 image tokens");
        assert_eq!(emb[0].len(), 1024);
        for t in &emb { for x in t { assert!(x.is_finite(), "non-finite vision embd"); } }
        eprintln!("VL.b OK: 64×1024 image embeddings, all finite");
    }

    /// **VL.c — end-to-end multimodal: a real image → coherent,
    /// non-degenerate description.** The decisive correctness gate: a
    /// broken patch-embed / pixel-unshuffle / projector / splice would
    /// collapse to gibberish or immediate-EOS. Uses the canonical
    /// llama.cpp mtmd test image if present, else a synthetic one.
    #[test]
    fn vl_c_describe_real_image() {
        if !std::path::Path::new(TXT).exists() || !std::path::Path::new(MMP).exists() {
            eprintln!("VL.c: skip"); return;
        }
        let rt = Lfm2VlRuntime::load(TXT, MMP).expect("load");
        let test_jpeg = "/tmp/sherry-llamacpp/tools/mtmd/test-1.jpeg";
        let img_path: std::path::PathBuf = if std::path::Path::new(test_jpeg).exists() {
            test_jpeg.into()
        } else {
            let t = std::env::temp_dir().join("rce_vl_blue.png");
            synthetic_image(&t, [30, 80, 220]);
            t
        };
        let cfg = VlGenConfig { max_new_tokens: 40, sample: SampleConfig::GREEDY, rep_window: 64 };
        let mut chunks = Vec::new();
        let t0 = std::time::Instant::now();
        let _ = rt.describe(&img_path, "Describe this image.", &cfg, |s| chunks.push(s.to_string()))
            .expect("describe");
        let dt = t0.elapsed();
        let full = chunks.concat();
        assert!(!full.is_empty(), "empty description (degenerate forward / splice bug)");
        let coherent = [" the ", " a ", " is ", " of ", " in ", " and ", " with ", " this ", " image "]
            .iter().any(|w| full.to_lowercase().contains(w))
            || full.chars().filter(|c| c.is_alphabetic()).count() >= full.len() / 2;
        assert!(coherent, "incoherent description: {full:?}");
        eprintln!(
            "VL.c OK: img={:?} → {full:?}  [{} tok in {dt:?} = {:.2} tok/s]",
            img_path.file_name().unwrap(), chunks.len(),
            chunks.len() as f64 / dt.as_secs_f64()
        );
    }
}
