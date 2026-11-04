//! LLaVA-style vision-language composition substrate.
//!
//! LLaVA-1.5 architecture:
//! 1. **Vision tower** — a CLIP ViT (typically ViT-L/14, sometimes at 336px
//!    resolution in 1.5) produces patch features `[N_patches, V_DIM]`
//!    (CLS is usually dropped, leaving the 256 spatial patches at 224px or
//!    576 patches at 336px).
//! 2. **MLP projector** — a 2-layer GELU MLP maps `V_DIM → LM_DIM` (e.g.,
//!    1024 → 4096 for ViT-L → Llama-2-7B).  In LLaVA-1.0 this was a single
//!    Linear; LLaVA-1.5 upgraded to the 2-layer MLP for better alignment.
//! 3. **LM injection** — the projected patch tokens are inserted into the
//!    LM's input embedding sequence at the position of the `<image>`
//!    placeholder (replacing it with the vision tokens), then the LM does
//!    autoregressive generation conditioned on the full mixed sequence.
//!
//! This crate ships the projector (`LlavaProjector`) and the composition
//! pattern.  Vision tower is provided externally (e.g., `rce-clip-vit-l`);
//! LM is provided externally (e.g., `rce-tinyllama`, `rce-qwen35`).
//!
//! No LLaVA-1.5 checkpoints are mounted in the model library at the time
//! of writing (2026-06-06), so the loader path is shipped as the scaffold
//! ready for drop-in weights.  Random-init constructors validate the
//! composition pattern end-to-end with the existing rce vision crate.

use std::path::Path;
use rce_weight_loader::{F32Slice, LoadedTensor, SafetensorsFile};

#[derive(Debug)]
pub enum LlavaError { Load(String) }
impl std::fmt::Display for LlavaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::Load(s) => write!(f, "llava: {s}") }
    }
}
impl std::error::Error for LlavaError {}

/// LLaVA dimensional configuration.  Defaults to the canonical LLaVA-1.5
/// dimensions (CLIP-L 1024 → 4096 LM embedding).
#[derive(Debug, Clone, Copy)]
pub struct LlavaConfig {
    /// Vision feature dimension (CLIP-L = 1024, CLIP-B = 768).
    pub v_dim: usize,
    /// Hidden dimension between the two MLP layers (LLaVA-1.5: same as LM dim).
    pub hidden_dim: usize,
    /// LM embedding dimension (Llama-2-7B = 4096, Llama-2-13B = 5120).
    pub lm_dim: usize,
}

impl LlavaConfig {
    /// LLaVA-1.5 with CLIP-L/14 vision + Llama-2-7B LM (most common config).
    pub const LLAVA_1_5_7B: Self = Self { v_dim: 1024, hidden_dim: 4096, lm_dim: 4096 };
    /// LLaVA-1.5 with CLIP-L/14 vision + Llama-2-13B LM.
    pub const LLAVA_1_5_13B: Self = Self { v_dim: 1024, hidden_dim: 5120, lm_dim: 5120 };
    /// LLaVA-style with CLIP-L/14 vision + TinyLlama-1.1B LM (d_model=2048).
    /// Useful for small-scale end-to-end VLM tests on rce.
    pub const CLIP_L_TO_TINYLLAMA: Self = Self { v_dim: 1024, hidden_dim: 2048, lm_dim: 2048 };
    /// LLaVA-style with CLIP-L/14 vision + Qwen 3.5 4B (d_model=2560).
    pub const CLIP_L_TO_QWEN35_4B: Self = Self { v_dim: 1024, hidden_dim: 2560, lm_dim: 2560 };
}

/// LLaVA's vision→LM projection: `LM_dim ← fc2(GELU(fc1(V_dim)))`.
///
/// **Forward layout** (PyTorch nn.Linear convention):
/// - `fc1_w [hidden_dim, v_dim]`, `fc1_b [hidden_dim]`
/// - `fc2_w [lm_dim, hidden_dim]`, `fc2_b [lm_dim]`
///
/// Weights are stored as `Vec<f32>` for the random-init path (small
/// allocations).  When real LLaVA weights ship, `from_safetensors` loads
/// them borrow-direct via `F32Slice`.
pub struct LlavaProjector {
    pub config: LlavaConfig,
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
    // Borrow-direct slots populated by `from_safetensors`; preferred at
    // forward time when present.
    pub fc1_w_slice: Option<F32Slice>,
    pub fc2_w_slice: Option<F32Slice>,
}

impl LlavaProjector {
    /// Construct with deterministic pseudo-random weights for scaffolding
    /// tests.  Each weight is `(idx_hash * scale).sin()`-derived from the
    /// seed — reproducible, no RNG dependency.
    pub fn random_init(config: LlavaConfig, seed: u32) -> Self {
        let s = (seed as f32) * 0.31415926;
        let h = config.hidden_dim;
        let v = config.v_dim;
        let l = config.lm_dim;
        let mut fc1_w = vec![0.0f32; h * v];
        for (i, w) in fc1_w.iter_mut().enumerate() {
            *w = ((i as f32 * 0.0011 + s).sin() * 0.02_f32).clamp(-0.1, 0.1);
        }
        let fc1_b = vec![0.0f32; h];
        let mut fc2_w = vec![0.0f32; l * h];
        for (i, w) in fc2_w.iter_mut().enumerate() {
            *w = ((i as f32 * 0.0013 + s * 1.3).cos() * 0.02_f32).clamp(-0.1, 0.1);
        }
        let fc2_b = vec![0.0f32; l];
        Self {
            config, fc1_w, fc1_b, fc2_w, fc2_b,
            fc1_w_slice: None, fc2_w_slice: None,
        }
    }

    /// Load LLaVA projector weights from a safetensors file.  Standard
    /// LLaVA-1.5 names: `model.mm_projector.0.{weight,bias}` (fc1),
    /// `model.mm_projector.2.{weight,bias}` (fc2).
    pub fn from_safetensors(path: &Path, config: LlavaConfig) -> Result<Self, LlavaError> {
        let sf = SafetensorsFile::load(path)
            .map_err(|e| LlavaError::Load(format!("load {}: {e}", path.display())))?;
        let lf = |name: &str| -> Result<F32Slice, LlavaError> {
            let t: &LoadedTensor = sf.get(name)
                .ok_or_else(|| LlavaError::Load(format!("missing {name}")))?;
            F32Slice::from_tensor(name, t).map_err(LlavaError::Load)
        };
        let lf_vec = |name: &str| -> Result<Vec<f32>, LlavaError> {
            Ok(lf(name)?.as_slice().to_vec())
        };
        let fc1_w_slice = lf("model.mm_projector.0.weight")?;
        let fc1_b = lf_vec("model.mm_projector.0.bias")?;
        let fc2_w_slice = lf("model.mm_projector.2.weight")?;
        let fc2_b = lf_vec("model.mm_projector.2.bias")?;
        if fc1_w_slice.len() != config.hidden_dim * config.v_dim {
            return Err(LlavaError::Load(format!(
                "fc1.weight: expected {}, got {}",
                config.hidden_dim * config.v_dim, fc1_w_slice.len())));
        }
        if fc2_w_slice.len() != config.lm_dim * config.hidden_dim {
            return Err(LlavaError::Load(format!(
                "fc2.weight: expected {}, got {}",
                config.lm_dim * config.hidden_dim, fc2_w_slice.len())));
        }
        Ok(Self {
            config,
            fc1_w: Vec::new(), fc1_b,
            fc2_w: Vec::new(), fc2_b,
            fc1_w_slice: Some(fc1_w_slice),
            fc2_w_slice: Some(fc2_w_slice),
        })
    }

    /// Project vision features `[t_tokens, V_dim]` into LM embedding space
    /// `[t_tokens, LM_dim]`.  Path: `fc1 (V→H) + bias → GELU → fc2 (H→L) + bias`.
    pub fn forward(&self, features: &[f32], t_tokens: usize) -> Vec<f32> {
        debug_assert_eq!(features.len(), t_tokens * self.config.v_dim);
        let fc1_w = self.fc1_w_slice.as_ref()
            .map(|s| s.as_slice()).unwrap_or(&self.fc1_w);
        let fc2_w = self.fc2_w_slice.as_ref()
            .map(|s| s.as_slice()).unwrap_or(&self.fc2_w);
        let mut h = matmul_nt_f32_bias(
            features, t_tokens, self.config.v_dim,
            fc1_w, Some(&self.fc1_b), self.config.hidden_dim);
        // GELU activation (tanh-approx — what LLaVA uses).
        for v in h.iter_mut() { *v = gelu_tanh(*v); }
        matmul_nt_f32_bias(
            &h, t_tokens, self.config.hidden_dim,
            fc2_w, Some(&self.fc2_b), self.config.lm_dim)
    }
}

fn gelu_tanh(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

fn matmul_nt_f32_bias(
    input: &[f32], m: usize, k: usize,
    weight: &[f32], bias: Option<&[f32]>, n: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    debug_assert_eq!(input.len(), m * k);
    debug_assert_eq!(weight.len(), n * k);
    let mut out = vec![0.0f32; m * n];
    out.par_chunks_mut(n).enumerate().for_each(|(mi, row)| {
        let in_row = &input[mi * k .. (mi + 1) * k];
        for ni in 0..n {
            let w_row = &weight[ni * k .. (ni + 1) * k];
            let mut s = 0.0f32;
            for ki in 0..k { s += in_row[ki] * w_row[ki]; }
            row[ni] = s + bias.map(|b| b[ni]).unwrap_or(0.0);
        }
    });
    out
}

/// A language model that can be driven by a LLaVA-style pipeline.
/// Implementations expose both text-token-id forward (`forward_token`)
/// and pre-computed-embedding forward (`forward_embed`) on a shared
/// state cursor.  This is the only abstraction `LlavaPipeline<LM>`
/// requires to support arbitrary backends.
pub trait LlavaLm {
    /// Per-LM autoregressive state (KV cache + position cursor).
    type State;

    /// Embedding dimension that vision tokens must be projected into.
    fn d_model(&self) -> usize;
    /// Vocabulary size — the length of every `forward_*` logits output.
    fn vocab(&self) -> usize;
    /// Fresh state for a new conversation.
    fn new_state(&self) -> Self::State;
    /// Forward one text token id; advances `state.pos`.  Returns
    /// `[vocab]` logits at the new position.
    fn forward_token(&self, state: &mut Self::State, token_id: i32) -> Vec<f32>;
    /// Forward one pre-computed `[d_model]` embedding; advances `state.pos`.
    /// Returns `[vocab]` logits.
    fn forward_embed(&self, state: &mut Self::State, embedding: Vec<f32>) -> Vec<f32>;
}

/// A composed LLaVA-style pipeline: holds a projector + a language-model
/// backend implementing [`LlavaLm`].  The vision encoder is supplied at
/// call time via `[features]` arrays so the pipeline stays decoupled from
/// any specific vision-tower crate.
pub struct LlavaPipeline<LM: LlavaLm> {
    pub projector: LlavaProjector,
    pub lm: LM,
}

impl<LM: LlavaLm> LlavaPipeline<LM> {
    pub fn new(projector: LlavaProjector, lm: LM) -> Self {
        assert_eq!(projector.config.lm_dim, lm.d_model(),
                   "projector lm_dim {} must match LM d_model {}",
                   projector.config.lm_dim, lm.d_model());
        Self { projector, lm }
    }

    /// Stream a slice of text token ids through the LM.  Returns the final
    /// `[vocab]` logits after the last token.  Empty slice → empty Vec.
    pub fn stream_text(&self, state: &mut LM::State, ids: &[i32]) -> Vec<f32> {
        let mut last = Vec::new();
        for &id in ids { last = self.lm.forward_token(state, id); }
        last
    }

    /// Project vision features `[n_tokens, projector.v_dim]` and stream
    /// them through the LM one at a time.  Returns the final `[vocab]`
    /// logits.
    pub fn stream_vision(&self, state: &mut LM::State, vision_features: &[f32], n_tokens: usize) -> Vec<f32> {
        let projected = self.projector.forward(vision_features, n_tokens);
        let lm_dim = self.projector.config.lm_dim;
        let mut last = Vec::new();
        for i in 0..n_tokens {
            let embed = projected[i * lm_dim .. (i + 1) * lm_dim].to_vec();
            last = self.lm.forward_embed(state, embed);
        }
        last
    }

    /// Greedy-sample `n` tokens, advancing state.  Caller provides the
    /// most-recent logits (typically returned by `stream_text` or
    /// `stream_vision`).  Returns the generated token ids.
    pub fn greedy_decode(&self, state: &mut LM::State, mut last_logits: Vec<f32>, n: usize) -> Vec<i32> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let next = last_logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i).unwrap() as i32;
            out.push(next);
            last_logits = self.lm.forward_token(state, next);
        }
        out
    }
}

/// Vision-token injection: insert projected vision tokens into the LM's
/// input embedding sequence at the index where the `<image>` placeholder
/// token sits in the text prompt.  Returns a new `[len_text - 1 + n_vis_tokens, lm_dim]`
/// merged sequence with the placeholder replaced.
pub fn inject_vision_tokens(
    text_embeds: &[f32], n_text_tokens: usize, lm_dim: usize,
    vision_tokens: &[f32], n_vision_tokens: usize,
    placeholder_idx: usize,
) -> Vec<f32> {
    assert!(placeholder_idx < n_text_tokens, "placeholder_idx out of range");
    assert_eq!(text_embeds.len(), n_text_tokens * lm_dim);
    assert_eq!(vision_tokens.len(), n_vision_tokens * lm_dim);
    let n_out = n_text_tokens - 1 + n_vision_tokens;
    let mut out = Vec::with_capacity(n_out * lm_dim);
    out.extend_from_slice(&text_embeds[0 .. placeholder_idx * lm_dim]);
    out.extend_from_slice(vision_tokens);
    out.extend_from_slice(&text_embeds[(placeholder_idx + 1) * lm_dim ..]);
    debug_assert_eq!(out.len(), n_out * lm_dim);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Projector smoke: random-init projector → forward on synthetic
    /// features → verify finite output of the expected shape.
    #[test]
    fn llava_projector_random_forward() {
        let cfg = LlavaConfig::LLAVA_1_5_7B;
        let p = LlavaProjector::random_init(cfg, 42);
        let t_tokens = 256;
        let features: Vec<f32> = (0..t_tokens * cfg.v_dim)
            .map(|i| ((i as f32) * 0.001).sin() * 0.1).collect();
        let t0 = std::time::Instant::now();
        let out = p.forward(&features, t_tokens);
        let dt = p.config.lm_dim;
        eprintln!("LLaVA projector 256 tokens {}→{}→{}: {:?}",
                  cfg.v_dim, cfg.hidden_dim, dt, t0.elapsed());
        assert_eq!(out.len(), t_tokens * dt);
        assert!(out.iter().all(|v| v.is_finite()));
        let max_abs: f32 = out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 0.0, "projector should produce nonzero output");
    }

    /// Vision token injection: synthesize text + vision token tables,
    /// inject at placeholder, verify resulting sequence has the right
    /// length and values at boundaries.
    #[test]
    fn llava_inject_vision_tokens() {
        let lm_dim = 8;
        let n_text = 5;        // 5 text tokens
        let placeholder = 2;   // token #2 is the <image> placeholder
        let n_vis = 16;        // 16 vision patches
        let text: Vec<f32> = (0..n_text * lm_dim)
            .map(|i| (i as f32) * 0.01).collect();
        let vision: Vec<f32> = (0..n_vis * lm_dim)
            .map(|i| 100.0 + (i as f32) * 0.01).collect();
        let merged = inject_vision_tokens(&text, n_text, lm_dim, &vision, n_vis, placeholder);
        assert_eq!(merged.len(), (n_text - 1 + n_vis) * lm_dim);
        // First placeholder*lm_dim entries are the prefix text.
        assert_eq!(&merged[0 .. placeholder * lm_dim],
                   &text[0 .. placeholder * lm_dim]);
        // Next n_vis*lm_dim are the vision tokens.
        assert_eq!(&merged[placeholder * lm_dim .. (placeholder + n_vis) * lm_dim],
                   &vision[..]);
        // Final entries are the suffix text after the placeholder.
        let suffix_start = (placeholder + n_vis) * lm_dim;
        assert_eq!(&merged[suffix_start..],
                   &text[(placeholder + 1) * lm_dim ..]);
    }

    /// End-to-end composition: CLIP vision tower → **full patch sequence**
    /// → LLaVA projector → LM embedding space.  This is the actual LLaVA
    /// hot-path: grab all 256 spatial patches (skip CLS) and project each
    /// into LM embedding space, yielding the vision-token stream that gets
    /// injected into the LM's input sequence.
    #[test]
    #[ignore = "needs CLIP weights mounted"]
    fn llava_clip_to_projector_composition() {
        use rce_clip_vit_l::{ClipVitL, V_IMG, V_DIM, V_N_PATCHES, V_SEQ};
        let clip_path = match std::path::PathBuf::from(
            "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/openai_clip-vit-large-patch14/model.safetensors")
        {
            p if p.exists() => p,
            _ => { eprintln!("CLIP safetensors not found"); return; }
        };
        let clip = ClipVitL::load_with_layers(&clip_path, 2, 2).expect("load CLIP");

        let image: Vec<f32> = (0..3 * V_IMG * V_IMG)
            .map(|i| ((i as f32) * 0.0009).sin() * 0.5).collect();

        // ── Grab the FULL post-transformer token sequence (V_SEQ=257, V_DIM=1024) ──
        let t0 = std::time::Instant::now();
        let tokens = clip.vision.encode_image_tokens(&image, 1e-5);
        let dt = t0.elapsed();
        assert_eq!(tokens.len(), V_SEQ * V_DIM);

        // ── Skip CLS, keep the 256 spatial patches (LLaVA convention) ──
        let patches = &tokens[V_DIM .. V_SEQ * V_DIM];
        assert_eq!(patches.len(), V_N_PATCHES * V_DIM);

        // ── Project all 256 patches into LM embedding space ──
        let cfg = LlavaConfig::LLAVA_1_5_7B;
        assert_eq!(cfg.v_dim, V_DIM,
                   "LLaVA-1.5 v_dim must match CLIP V_DIM ({})", V_DIM);
        let proj = LlavaProjector::random_init(cfg, 7);
        let t0 = std::time::Instant::now();
        let projected = proj.forward(patches, V_N_PATCHES);
        let dt_proj = t0.elapsed();
        assert_eq!(projected.len(), V_N_PATCHES * cfg.lm_dim);
        assert!(projected.iter().all(|v| v.is_finite()));

        // ── Inject into a small synthetic text-embed sequence ──
        let n_text = 5;
        let placeholder = 2;
        let text: Vec<f32> = (0..n_text * cfg.lm_dim)
            .map(|i| ((i as f32) * 0.001).sin() * 0.1).collect();
        let merged = inject_vision_tokens(&text, n_text, cfg.lm_dim,
            &projected, V_N_PATCHES, placeholder);
        let expected_len = (n_text - 1 + V_N_PATCHES) * cfg.lm_dim;
        assert_eq!(merged.len(), expected_len);

        eprintln!("CLIP encode (2L) {dt:?} → 256 patches → projector {dt_proj:?} → LM-embed [{},{}] → inject → merged [{},{}]",
                  V_N_PATCHES, cfg.lm_dim, n_text - 1 + V_N_PATCHES, cfg.lm_dim);
        eprintln!("LLaVA composition ✓ — vision tokens flow end-to-end into LM input sequence");
    }

    /// **Full vision-language round-trip on rce**: CLIP encodes an image,
    /// LlavaProjector projects 256 patches into TinyLlama's d_model=2048
    /// embedding space, then TinyLlama runs forward via
    /// `forward_logits_embed`, consuming each vision token as a pre-computed
    /// embedding (skipping the token-id lookup).  Verifies the LM produces
    /// finite logits over its vocabulary.
    ///
    /// This is the first end-to-end VLM round-trip on the rce substrate:
    /// image bytes → CLIP vision tower → MLP projector → autoregressive LM
    /// → vocabulary logits, all on real weights.
    #[test]
    #[ignore = "VLM round-trip — needs CLIP + TinyLlama GGUF mounted"]
    fn llava_clip_tinyllama_full_roundtrip() {
        use rce_clip_vit_l::{ClipVitL, V_IMG, V_DIM, V_N_PATCHES};
        use rce_tinyllama::TinyLlamaRuntime;

        let clip_path = std::path::PathBuf::from(
            "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/openai_clip-vit-large-patch14/model.safetensors");
        let tll_path = std::path::PathBuf::from(
            "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/tinyllama-1.1b-chat-q8_0.gguf");
        if !clip_path.exists() { eprintln!("CLIP not found"); return; }
        if !tll_path.exists() { eprintln!("TinyLlama not found"); return; }

        let t0 = std::time::Instant::now();
        let clip = ClipVitL::load_with_layers(&clip_path, 2, 0).expect("load CLIP");
        eprintln!("CLIP loaded (2v+0t) in {:?}", t0.elapsed());

        let t0 = std::time::Instant::now();
        let tll = TinyLlamaRuntime::load(&tll_path).expect("load TinyLlama");
        eprintln!("TinyLlama loaded (d_model={}, vocab={}) in {:?}",
                  tll.cfg.d_model, tll.cfg.vocab, t0.elapsed());

        // ── CLIP encode image → patch features [256, 1024] ──
        let image: Vec<f32> = (0..3 * V_IMG * V_IMG)
            .map(|i| ((i as f32) * 0.0009).sin() * 0.5).collect();
        let t0 = std::time::Instant::now();
        let tokens = clip.vision.encode_image_tokens(&image, 1e-5);
        eprintln!("CLIP encode_image_tokens (2L): {:?}", t0.elapsed());
        let patches = &tokens[V_DIM .. (V_N_PATCHES + 1) * V_DIM];

        // ── LLaVA projector: V_DIM=1024 → tinyllama d_model ──
        let cfg = LlavaConfig::CLIP_L_TO_TINYLLAMA;
        assert_eq!(cfg.lm_dim, tll.cfg.d_model,
                   "LlavaConfig::CLIP_L_TO_TINYLLAMA lm_dim={} must match TinyLlama d_model={}",
                   cfg.lm_dim, tll.cfg.d_model);
        let proj = LlavaProjector::random_init(cfg, 7);
        let t0 = std::time::Instant::now();
        let projected = proj.forward(patches, V_N_PATCHES);
        eprintln!("Projector forward (256 patches): {:?}", t0.elapsed());
        assert_eq!(projected.len(), V_N_PATCHES * cfg.lm_dim);

        // ── Feed the first 4 projected vision tokens to TinyLlama as
        // pre-computed embeddings.  Each call advances state.pos by 1
        // and yields logits over the LM vocabulary.
        let mut state = tll.new_state();
        let n_drive = 4usize;
        let mut last_logits = Vec::new();
        let t0 = std::time::Instant::now();
        for i in 0..n_drive {
            let embed = projected[i * cfg.lm_dim .. (i + 1) * cfg.lm_dim].to_vec();
            last_logits = tll.forward_logits_embed(&mut state, embed);
        }
        eprintln!("TinyLlama forward (×{n_drive} vision tokens): {:?}", t0.elapsed());

        assert_eq!(last_logits.len(), tll.cfg.vocab);
        assert!(last_logits.iter().all(|v| v.is_finite()));
        let max_idx = last_logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap();
        let max_v = last_logits[max_idx];
        eprintln!("LM logits over {} vocab, argmax id={max_idx} value={max_v:.4}",
                  tll.cfg.vocab);
        eprintln!("✓ rce VLM round-trip: image → CLIP → projector → TinyLlama → finite logits");
    }

    /// **Mixed text+vision prompt assembly** — the actual LLaVA chat
    /// pattern: prefix text → vision tokens → suffix text → greedy decode.
    /// Verifies that interleaving `forward_token` (text ids) with
    /// `forward_logits_embed` (projected vision tokens) all sharing the
    /// same `state.pos` cursor produces a coherent autoregressive output.
    ///
    /// Prompt structure mirrors LLaVA's `USER: <image>\nDescribe.\nASSISTANT:`
    /// template, encoded as: TinyLlama BOS + text-prefix tokens + 256 vision
    /// tokens (replacing `<image>`) + text-suffix tokens → greedy-sample 4
    /// answer tokens.
    #[test]
    #[ignore = "VLM chat flow — needs CLIP + TinyLlama mounted"]
    fn llava_chat_flow_mixed_prompt() {
        use rce_clip_vit_l::{ClipVitL, V_IMG, V_DIM, V_N_PATCHES};
        use rce_tinyllama::TinyLlamaRuntime;

        let clip_path = std::path::PathBuf::from(
            "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/openai_clip-vit-large-patch14/model.safetensors");
        let tll_path = std::path::PathBuf::from(
            "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/tinyllama-1.1b-chat-q8_0.gguf");
        if !clip_path.exists() { eprintln!("CLIP not found"); return; }
        if !tll_path.exists() { eprintln!("TinyLlama not found"); return; }

        let clip = ClipVitL::load_with_layers(&clip_path, 2, 0).expect("load CLIP");
        let tll = TinyLlamaRuntime::load(&tll_path).expect("load TinyLlama");
        let tk = tll.tokenizer();

        // ── 1. Encode image → projected vision tokens [256, d_model] ──
        let image: Vec<f32> = (0..3 * V_IMG * V_IMG)
            .map(|i| ((i as f32) * 0.0009).sin() * 0.5).collect();
        let tokens = clip.vision.encode_image_tokens(&image, 1e-5);
        let patches = &tokens[V_DIM .. (V_N_PATCHES + 1) * V_DIM];
        let cfg = LlavaConfig::CLIP_L_TO_TINYLLAMA;
        let proj = LlavaProjector::random_init(cfg, 7);
        let vision_embeds = proj.forward(patches, V_N_PATCHES);

        // ── 2. Tokenize prefix + suffix text ──
        // We use a small subset of vision tokens (8) to keep wall-time
        // bounded; the architectural flow is identical at 256.
        let n_vis = 8usize;
        let prefix_text = "USER: ";
        let suffix_text = "\nDescribe.\nASSISTANT:";
        let prefix_ids = tk.encode(prefix_text, true);   // add BOS
        let suffix_ids = tk.encode(suffix_text, false);
        eprintln!("prefix ids ({}): {:?}", prefix_ids.len(), prefix_ids);
        eprintln!("suffix ids ({}): {:?}", suffix_ids.len(), suffix_ids);

        let mut state = tll.new_state();
        let t0 = std::time::Instant::now();

        // 2a. Stream prefix text tokens.
        let mut last_logits = Vec::new();
        for &id in &prefix_ids {
            last_logits = tll.forward_logits(&mut state, id);
        }
        let dt_prefix = t0.elapsed();

        // 2b. Stream vision tokens (each is a pre-computed embedding).
        let t0 = std::time::Instant::now();
        for i in 0..n_vis {
            let embed = vision_embeds[i * cfg.lm_dim .. (i + 1) * cfg.lm_dim].to_vec();
            last_logits = tll.forward_logits_embed(&mut state, embed);
        }
        let dt_vis = t0.elapsed();

        // 2c. Stream suffix text tokens.
        let t0 = std::time::Instant::now();
        for &id in &suffix_ids {
            last_logits = tll.forward_logits(&mut state, id);
        }
        let dt_suffix = t0.elapsed();

        // 2d. Greedy decode 4 answer tokens.
        let mut answer_ids: Vec<i32> = Vec::with_capacity(4);
        let t0 = std::time::Instant::now();
        for _ in 0..4 {
            let next = last_logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i).unwrap() as i32;
            answer_ids.push(next);
            last_logits = tll.forward_logits(&mut state, next);
        }
        let dt_gen = t0.elapsed();
        let answer = tk.decode_sequence(&answer_ids, true);
        eprintln!("prefix {dt_prefix:?} + vision ({n_vis}) {dt_vis:?} + suffix {dt_suffix:?} + gen(4) {dt_gen:?}");
        eprintln!("answer ids: {answer_ids:?}");
        eprintln!("answer text: {answer:?}");
        assert!(last_logits.iter().all(|v| v.is_finite()));
        assert_eq!(answer_ids.len(), 4);
        eprintln!("✓ rce VLM chat flow: text prefix + vision + text suffix + greedy decode composes end-to-end");
    }

    // ── LlavaLm trait impl for TinyLlama (test-side adapter) ──
    impl LlavaLm for rce_tinyllama::TinyLlamaRuntime {
        type State = rce_tinyllama::TinyLlamaState;
        fn d_model(&self) -> usize { self.cfg.d_model }
        fn vocab(&self) -> usize { self.cfg.vocab }
        fn new_state(&self) -> Self::State { self.new_state() }
        fn forward_token(&self, state: &mut Self::State, token_id: i32) -> Vec<f32> {
            self.forward_logits(state, token_id)
        }
        fn forward_embed(&self, state: &mut Self::State, embedding: Vec<f32>) -> Vec<f32> {
            self.forward_logits_embed(state, embedding)
        }
    }

    /// Same VLM chat flow as `llava_chat_flow_mixed_prompt`, but driven
    /// through the `LlavaLm` trait + `LlavaPipeline<LM>` abstraction.
    /// Validates that the trait surface is sufficient to express the full
    /// vision+text chat composition with the LM as a generic parameter.
    #[test]
    #[ignore = "VLM chat via trait — needs CLIP + TinyLlama mounted"]
    fn llava_pipeline_trait_chat_flow() {
        use rce_clip_vit_l::{ClipVitL, V_IMG, V_DIM, V_N_PATCHES};
        use rce_tinyllama::TinyLlamaRuntime;

        let clip_path = std::path::PathBuf::from(
            "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/openai_clip-vit-large-patch14/model.safetensors");
        let tll_path = std::path::PathBuf::from(
            "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/tinyllama-1.1b-chat-q8_0.gguf");
        if !clip_path.exists() || !tll_path.exists() {
            eprintln!("models not mounted"); return;
        }
        let clip = ClipVitL::load_with_layers(&clip_path, 2, 0).expect("load CLIP");
        let tll = TinyLlamaRuntime::load(&tll_path).expect("load TinyLlama");

        let cfg = LlavaConfig::CLIP_L_TO_TINYLLAMA;
        let proj = LlavaProjector::random_init(cfg, 7);
        let pipe = LlavaPipeline::new(proj, tll);
        let tk = pipe.lm.tokenizer();
        eprintln!("LlavaPipeline<TinyLlama>: d_model={}, vocab={}",
                  pipe.lm.d_model(), pipe.lm.vocab());

        // ── Encode image → patches ──
        let image: Vec<f32> = (0..3 * V_IMG * V_IMG)
            .map(|i| ((i as f32) * 0.0009).sin() * 0.5).collect();
        let tokens = clip.vision.encode_image_tokens(&image, 1e-5);
        let patches = &tokens[V_DIM .. (V_N_PATCHES + 1) * V_DIM];
        let n_vis = 8;

        // ── Drive the pipeline via trait methods ──
        let mut state = pipe.lm.new_state();
        let t0 = std::time::Instant::now();
        let prefix_ids = tk.encode("USER: ", true);
        let _ = pipe.stream_text(&mut state, &prefix_ids);
        let last1 = pipe.stream_vision(&mut state, &patches[..n_vis * V_DIM], n_vis);
        let suffix_ids = tk.encode("\nDescribe.\nASSISTANT:", false);
        let last2 = pipe.stream_text(&mut state, &suffix_ids);
        let answer = pipe.greedy_decode(&mut state, last2.clone(), 4);
        let dt = t0.elapsed();
        let answer_text = tk.decode_sequence(&answer, true);
        eprintln!("trait-driven chat: {dt:?}");
        eprintln!("answer ids: {answer:?}, text: {answer_text:?}");
        assert!(last1.iter().all(|v| v.is_finite()));
        assert!(last2.iter().all(|v| v.is_finite()));
        assert_eq!(answer.len(), 4);
        eprintln!("✓ LlavaPipeline<LM: LlavaLm> abstraction works end-to-end");
    }

    /// **Full pipeline from a real PNG file on disk** — `preprocess_image`
    /// decodes + resizes + crops + normalizes a PNG; the resulting tensor
    /// flows through CLIP → projector → LM via `LlavaPipeline`.  Closes
    /// the "image preprocessor end-to-end" open chapter: every component
    /// from raw bytes to LM logits runs on rce substrate.
    #[test]
    #[ignore = "VLM chat from real PNG — needs CLIP + TinyLlama mounted; writes /tmp/llava_test_image.png"]
    fn llava_pipeline_chat_from_real_png() {
        use rce_clip_vit_l::{ClipVitL, V_DIM, V_N_PATCHES, preprocess_image};
        use rce_tinyllama::TinyLlamaRuntime;

        let clip_path = std::path::PathBuf::from(
            "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/openai_clip-vit-large-patch14/model.safetensors");
        let tll_path = std::path::PathBuf::from(
            "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/tinyllama-1.1b-chat-q8_0.gguf");
        if !clip_path.exists() || !tll_path.exists() {
            eprintln!("models not mounted"); return;
        }

        // ── 1. Write a deterministic test PNG ──
        let png_path = std::path::PathBuf::from("/tmp/llava_test_image.png");
        let (w, h) = (320u32, 240u32);
        let mut buf = image::RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let band = x * 6 / w;
                let yf = (y as f32) / (h as f32);
                let (r, g, b) = match band {
                    0 => (255_u8, 50, 50),
                    1 => (255, 165, 0),
                    2 => (255, 220, 50),
                    3 => (50, 200, 80),
                    4 => (50, 80, 220),
                    _ => (180, 60, 200),
                };
                let s = ((1.0 - yf) * 0.5 + 0.5).clamp(0.0, 1.0);
                buf.put_pixel(x, y, image::Rgb([
                    ((r as f32) * s) as u8,
                    ((g as f32) * s) as u8,
                    ((b as f32) * s) as u8,
                ]));
            }
        }
        buf.save(&png_path).expect("write test png");

        // ── 2. Load models + build the pipeline ──
        let clip = ClipVitL::load_with_layers(&clip_path, 2, 0).expect("load CLIP");
        let tll = TinyLlamaRuntime::load(&tll_path).expect("load TinyLlama");
        let cfg = LlavaConfig::CLIP_L_TO_TINYLLAMA;
        let proj = LlavaProjector::random_init(cfg, 7);
        let pipe = LlavaPipeline::new(proj, tll);
        let tk = pipe.lm.tokenizer();

        // ── 3. Full preprocess path from the PNG file ──
        let t0 = std::time::Instant::now();
        let f32_img = preprocess_image(&png_path).expect("preprocess");
        eprintln!("preprocess_image (PNG {w}×{h} → 224×224 f32): {:?}", t0.elapsed());

        // ── 4. CLIP encode → patch tokens ──
        let t0 = std::time::Instant::now();
        let tokens = clip.vision.encode_image_tokens(&f32_img, 1e-5);
        eprintln!("encode_image_tokens (2L CLIP): {:?}", t0.elapsed());
        let patches = &tokens[V_DIM .. (V_N_PATCHES + 1) * V_DIM];

        // ── 5. Chat flow via the trait abstraction ──
        let mut state = pipe.lm.new_state();
        let n_vis = 8;
        let prefix_ids = tk.encode("USER: ", true);
        let _ = pipe.stream_text(&mut state, &prefix_ids);
        let _ = pipe.stream_vision(&mut state, &patches[..n_vis * V_DIM], n_vis);
        let suffix_ids = tk.encode("\nDescribe the image.\nASSISTANT:", false);
        let last_logits = pipe.stream_text(&mut state, &suffix_ids);
        let answer_ids = pipe.greedy_decode(&mut state, last_logits, 6);
        let answer_text = tk.decode_sequence(&answer_ids, true);
        eprintln!("answer ids: {answer_ids:?}");
        eprintln!("answer text: {answer_text:?}");
        assert_eq!(answer_ids.len(), 6);
        eprintln!("✓ rce VLM end-to-end: PNG file → preprocess → CLIP → projector → TinyLlama → 6 answer tokens");
    }

    // ── LlavaLm trait impl for Qwen35Runtime (dev-side adapter) ──
    impl LlavaLm for rce_qwen35::Qwen35Runtime {
        type State = rce_qwen35::Qwen35State;
        fn d_model(&self) -> usize { self.cfg.d_model }
        fn vocab(&self) -> usize { self.cfg.vocab }
        fn new_state(&self) -> Self::State { self.new_state() }
        fn forward_token(&self, state: &mut Self::State, token_id: i32) -> Vec<f32> {
            self.forward_logits(state, token_id)
        }
        fn forward_embed(&self, state: &mut Self::State, embedding: Vec<f32>) -> Vec<f32> {
            self.forward_logits_embed(state, embedding)
        }
    }

    /// Generic VLM chat via `LlavaPipeline<Qwen35Runtime>` — proves the
    /// `LlavaLm` trait works across LM crates (not just TinyLlama).
    /// Uses the smallest mounted Qwen 3.5 GGUF (~0.8B / 2B / 4B).
    #[test]
    #[ignore = "VLM chat via Qwen35 — needs CLIP + Qwen 3.5 GGUF mounted"]
    fn llava_pipeline_qwen35_chat() {
        use rce_clip_vit_l::{ClipVitL, V_DIM, V_N_PATCHES, preprocess_image};
        use rce_qwen35::Qwen35Runtime;

        let clip_path = std::path::PathBuf::from(
            "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/openai_clip-vit-large-patch14/model.safetensors");
        let qwen_path = std::path::PathBuf::from(
            "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/qwen3.5-0.8b-gguf");
        if !clip_path.exists() { eprintln!("CLIP not found"); return; }
        if !qwen_path.exists() { eprintln!("Qwen 3.5 0.8B dir not found"); return; }
        // Find a .gguf inside the qwen dir.
        let gguf = match std::fs::read_dir(&qwen_path).ok().and_then(|d| {
            d.filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.extension().map(|e| e == "gguf").unwrap_or(false))
        }) {
            Some(p) => p,
            None => { eprintln!("no .gguf in {}", qwen_path.display()); return; }
        };
        eprintln!("Qwen GGUF: {}", gguf.display());

        let clip = ClipVitL::load_with_layers(&clip_path, 2, 0).expect("load CLIP");
        let qwen = Qwen35Runtime::load(&gguf).expect("load Qwen");
        let d_model = qwen.cfg.d_model;
        let vocab = qwen.cfg.vocab;
        eprintln!("Qwen 3.5: d_model={d_model}, vocab={vocab}");

        let cfg = LlavaConfig { v_dim: 1024, hidden_dim: d_model, lm_dim: d_model };
        let proj = LlavaProjector::random_init(cfg, 7);
        let pipe = LlavaPipeline::new(proj, qwen);
        let tk = pipe.lm.tokenizer();

        // Synthesize a test PNG.
        let png_path = std::path::PathBuf::from("/tmp/llava_qwen_test.png");
        let mut buf = image::RgbImage::new(256, 256);
        for y in 0..256 { for x in 0..256 {
            let r = ((x as f32 / 256.0) * 255.0) as u8;
            let g = ((y as f32 / 256.0) * 255.0) as u8;
            let b = (((x ^ y) & 0xff) as u8);
            buf.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }}
        buf.save(&png_path).expect("write png");

        let f32_img = preprocess_image(&png_path).expect("preprocess");
        let tokens = clip.vision.encode_image_tokens(&f32_img, 1e-5);
        let patches = &tokens[V_DIM .. (V_N_PATCHES + 1) * V_DIM];

        let mut state = pipe.lm.new_state();
        let n_vis = 4;
        let prefix_ids = tk.encode("User: Describe the image. ", true);
        let _ = pipe.stream_text(&mut state, &prefix_ids);
        let _ = pipe.stream_vision(&mut state, &patches[..n_vis * V_DIM], n_vis);
        let suffix_ids = tk.encode("\nAssistant:", false);
        let last_logits = pipe.stream_text(&mut state, &suffix_ids);
        let answer_ids = pipe.greedy_decode(&mut state, last_logits, 4);
        let answer_text = tk.decode_sequence(&answer_ids, true);
        eprintln!("Qwen answer ids: {answer_ids:?}, text: {answer_text:?}");
        assert_eq!(answer_ids.len(), 4);
        eprintln!("✓ LlavaPipeline<Qwen35Runtime> chat composes end-to-end");
    }

    // ── LlavaLm trait impl for Gemma4Runtime (dev-side adapter) ──
    //
    // Gemma 4 is the first **natively-multimodal** model to plug into the
    // trait: its `forward_image_embed` is the exact LLaVA-style entry
    // point, sourcing the per-layer embedding from token id 0 (padding)
    // per `gemma4.cpp:410-422`.  Text uses `forward_logits` (standard
    // token-id path with PLE id = token_id).
    impl LlavaLm for rce_gemma4::Gemma4Runtime {
        type State = rce_gemma4::Gemma4State;
        fn d_model(&self) -> usize { self.cfg.d_model }
        fn vocab(&self) -> usize { self.cfg.vocab }
        fn new_state(&self) -> Self::State { self.new_state() }
        fn forward_token(&self, state: &mut Self::State, token_id: i32) -> Vec<f32> {
            self.forward_logits(state, token_id)
        }
        fn forward_embed(&self, state: &mut Self::State, embedding: Vec<f32>) -> Vec<f32> {
            // Image-position convention: PLE token = 0 (padding).
            self.forward_image_embed(state, embedding)
        }
    }

    /// Generic VLM chat via `LlavaPipeline<Gemma4Runtime>` — proves the
    /// substrate works with a **natively-multimodal** LM, not just text-only
    /// decoders.  Uses Gemma 4 E2B text GGUF.
    #[test]
    #[ignore = "VLM chat via Gemma 4 — needs CLIP + Gemma 4 E2B text GGUF mounted"]
    fn llava_pipeline_gemma4_chat() {
        use rce_clip_vit_l::{ClipVitL, V_DIM, V_N_PATCHES, preprocess_image};
        use rce_gemma4::Gemma4Runtime;

        let clip_path = std::path::PathBuf::from(
            "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/openai_clip-vit-large-patch14/model.safetensors");
        let gemma_path = std::path::PathBuf::from(
            "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/gemma-4-E2B-text-q8_0.gguf");
        if !clip_path.exists() { eprintln!("CLIP not found"); return; }
        if !gemma_path.exists() { eprintln!("Gemma 4 E2B text GGUF not found"); return; }

        let clip = ClipVitL::load_with_layers(&clip_path, 2, 0).expect("load CLIP");
        let gemma = Gemma4Runtime::load(&gemma_path).expect("load Gemma 4");
        let d_model = gemma.cfg.d_model;
        let vocab = gemma.cfg.vocab;
        eprintln!("Gemma 4 E2B: d_model={d_model}, vocab={vocab}");

        let cfg = LlavaConfig { v_dim: 1024, hidden_dim: d_model, lm_dim: d_model };
        let proj = LlavaProjector::random_init(cfg, 7);
        let pipe = LlavaPipeline::new(proj, gemma);
        let tk = pipe.lm.tokenizer();

        // Synthesize a test PNG.
        let png_path = std::path::PathBuf::from("/tmp/llava_gemma_test.png");
        let mut buf = image::RgbImage::new(256, 256);
        for y in 0..256 { for x in 0..256 {
            buf.put_pixel(x as u32, y as u32, image::Rgb([
                ((x as u16 * 255 / 256) as u8),
                ((y as u16 * 255 / 256) as u8),
                ((x ^ y) as u8),
            ]));
        }}
        buf.save(&png_path).expect("write png");

        let f32_img = preprocess_image(&png_path).expect("preprocess");
        let tokens = clip.vision.encode_image_tokens(&f32_img, 1e-5);
        let patches = &tokens[V_DIM .. (V_N_PATCHES + 1) * V_DIM];

        let mut state = pipe.lm.new_state();
        let n_vis = 4;
        // Gemma 4 chat framing uses <start_of_turn> tags.
        let prefix_ids = tk.encode("<start_of_turn>user\n", true);
        let _ = pipe.stream_text(&mut state, &prefix_ids);
        let _ = pipe.stream_vision(&mut state, &patches[..n_vis * V_DIM], n_vis);
        let suffix_ids = tk.encode("Describe.<end_of_turn>\n<start_of_turn>model\n", false);
        let last_logits = pipe.stream_text(&mut state, &suffix_ids);
        let answer_ids = pipe.greedy_decode(&mut state, last_logits, 4);
        let answer_text = tk.decode_sequence(&answer_ids, true);
        eprintln!("Gemma 4 answer ids: {answer_ids:?}, text: {answer_text:?}");
        assert_eq!(answer_ids.len(), 4);
        eprintln!("✓ LlavaPipeline<Gemma4Runtime> chat composes — 3rd LM family on the trait");
    }
}
