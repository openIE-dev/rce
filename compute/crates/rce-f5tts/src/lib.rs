//! F5-TTS v1 Base — character-level text-to-speech via flow matching.
//!
//! ```text
//! ref_mel [100, T_ref] + ref_text + gen_text
//!   → text tokenizer (char-level, 2546 vocab, PAD=-1 → 0)
//!   → text_embed (Embedding 2546→512)
//!   → 4 ConvNeXt-1D text_blocks (dwconv k=7, LN, pwconv1 512→1024, GELU, GRN, pwconv2 1024→512, residual)
//!   → text features [512, T_total]                  (zero-pad text for gen frames)
//!
//! flow matching iteration (Euler ODE, N steps from t=0 to t=1):
//!   y_t = (1-t) * noise + t * target_mel               (linear interpolation)
//!   inputs = concat([y_t (100), text_feats (512), cond_mel (100)], dim=channel) → [712, T_total]
//!   z = input_embed.proj(inputs)                       (Linear 712→1024)
//!   z = z + conv_pos_embed(z)                          (depthwise-conv positional, k=31, residual)
//!   for each DiT block:
//!     mod = silu(time_emb) → attn_norm.linear → split (6 × 1024 = s1, b1, g1, s2, b2, g2)
//!     z_n = LN(z) * (1+s1) + b1
//!     z   = z + g1 * Attn(z_n)             (16 heads × 64 head_dim, RoPE rotary[32 freqs])
//!     z_n = LN(z) * (1+s2) + b2
//!     z   = z + g2 * FFN(z_n)              (Linear 1024→2048, GELU, Linear 2048→1024)
//!   mod = silu(time_emb) → norm_out.linear → split (2 × 1024 = s, b)
//!   z_n = LN(z) * (1+s) + b
//!   v_pred = proj_out(z_n)                              (Linear 1024→100)
//!   y_t = y_t + dt * v_pred                             (Euler step)
//! mel_final = y_t
//!
//! mel_final → vocos → wav
//! ```

use rce_weight_loader::SafetensorsFile;
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;

pub const HIDDEN: usize = 1024;
pub const N_HEADS: usize = 16;
pub const HEAD_DIM: usize = 64;
pub const FF: usize = 2048;
pub const N_DIT: usize = 22;
pub const TEXT_HIDDEN: usize = 512;
pub const TEXT_FF: usize = 1024;
pub const N_TEXT_BLOCKS: usize = 4;
pub const TEXT_DW_K: usize = 7;
pub const VOCAB: usize = 2546;
pub const MEL_DIM: usize = 100;
pub const ROPE_DIM: usize = 64;         // applied to first 64 of HEAD_DIM (i.e. all)
pub const TIME_SIN_DIM: usize = 256;
pub const POS_CONV_K: usize = 31;
pub const POS_CONV_GROUPS: usize = 16;  // 1024 / 64 (depthwise-ish)
pub const POS_CONV_CH_PER_GROUP: usize = 64;
pub const INPUT_PROJ_IN: usize = 712;    // 100 (y_t) + 512 (text) + 100 (cond mel)
pub const LN_EPS: f32 = 1e-6;

#[derive(Debug)]
pub enum F5Error { Load(String), Tokenize(String) }
impl std::fmt::Display for F5Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(m)     => write!(f, "f5-tts load: {m}"),
            Self::Tokenize(m) => write!(f, "f5-tts tokenize: {m}"),
        }
    }
}
impl std::error::Error for F5Error {}

fn ld(sf: &SafetensorsFile, name: &str) -> Result<rce_weight_loader::F32Slice, F5Error> {
    let t = sf.get(name).ok_or_else(|| F5Error::Load(format!("missing {name}")))?;
    rce_weight_loader::F32Slice::from_tensor(name, t).map_err(F5Error::Load)
}
use rce_weight_loader::F32Slice;

// ── ops ─────────────────────────────────────────────────────────────────────
fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}
fn gelu(x: f32) -> f32 { 0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2)) }

/// LayerNorm on time-major `[T, C]` activation.
fn layer_norm(x: &[f32], t: usize, c: usize, gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for ti in 0..t {
        let row = &x[ti*c..(ti+1)*c];
        let mean: f32 = row.iter().sum::<f32>() / c as f32;
        let var: f32 = row.iter().map(|v| { let d = v - mean; d*d }).sum::<f32>() / c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for ci in 0..c { out[ti*c + ci] = (row[ci] - mean) * inv * gamma[ci] + beta[ci]; }
    }
    out
}

/// LayerNorm with `affine = False` (used inside AdaLN: bare normalize, no learned γ/β).
fn layer_norm_nb(x: &[f32], t: usize, c: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for ti in 0..t {
        let row = &x[ti*c..(ti+1)*c];
        let mean: f32 = row.iter().sum::<f32>() / c as f32;
        let var: f32 = row.iter().map(|v| { let d = v - mean; d*d }).sum::<f32>() / c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for ci in 0..c { out[ti*c + ci] = (row[ci] - mean) * inv; }
    }
    out
}

/// y = x @ W.T + b. Delegates to rce-op-conv2d (Apple Accelerate sgemm on macOS).
fn linear(x: &[f32], t: usize, din: usize, w: &[f32], dout: usize, b: Option<&[f32]>) -> Vec<f32> {
    rce_op_conv2d::linear_f32(x, t, din, w, dout, b)
}

/// Group convolution 1D on channel-major `[Cin, T]`.  w: `[Cout, Cin/groups, K]`.
fn conv1d(x: &[f32], cin: usize, t: usize,
          w: &[f32], cout: usize, k: usize,
          groups: usize, padding: usize,
          bias: &[f32]) -> Vec<f32> {
    let cin_g = cin / groups;
    let cout_g = cout / groups;
    let t_out = t;   // SAME padding
    let mut out = vec![0.0f32; cout * t_out];
    for to in 0..t_out {
        for g_ in 0..groups {
            for oc in 0..cout_g {
                let oc_g = g_ * cout_g + oc;
                let mut acc = bias[oc_g];
                for ic in 0..cin_g {
                    let ic_g = g_ * cin_g + ic;
                    for kk in 0..k {
                        let tp = to + kk;
                        if tp < padding { continue; }
                        let xi = tp - padding;
                        if xi >= t { continue; }
                        let w_idx = (oc_g * cin_g + ic) * k + kk;
                        acc += x[ic_g * t + xi] * w[w_idx];
                    }
                }
                out[oc_g * t_out + to] = acc;
            }
        }
    }
    out
}

// ── tokenizer ───────────────────────────────────────────────────────────────
pub struct Tokenizer { char_to_id: std::collections::HashMap<String, u32> }
impl Tokenizer {
    pub fn from_vocab_file(path: &Path) -> Result<Self, F5Error> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| F5Error::Tokenize(format!("read vocab: {e}")))?;
        // vocab.txt is one char per line; line index = token id (PAD slot is added by model implicitly).
        // F5-TTS reserves id 0 as PAD; first real char gets id 1.
        let mut char_to_id = std::collections::HashMap::new();
        for (idx, line) in content.lines().enumerate() {
            char_to_id.insert(line.to_string(), (idx + 1) as u32);   // +1 to skip PAD
        }
        Ok(Self { char_to_id })
    }
    /// Encode text to token ids (PAD = 0 for unknown chars).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for ch in text.chars() {
            let s = ch.to_string();
            // Underscore is the special "space" token in F5-TTS by convention.
            let key = if ch == ' ' { "_".to_string() } else { s };
            let id = *self.char_to_id.get(&key).unwrap_or(&0);
            out.push(id);
        }
        out
    }
}

// ── text encoder (4 ConvNeXt-1D blocks) ─────────────────────────────────────
struct TextBlock {
    dw_w: F32Slice, dw_b: F32Slice,
    norm_w: F32Slice, norm_b: F32Slice,
    pw1_w: F32Slice, pw1_b: F32Slice,
    pw2_w: F32Slice, pw2_b: F32Slice,
    grn_gamma: F32Slice, grn_beta: F32Slice,
}
impl TextBlock {
    fn load(sf: &SafetensorsFile, i: usize) -> Result<Self, F5Error> {
        let p = format!("ema_model.transformer.text_embed.text_blocks.{i}");
        Ok(Self {
            dw_w: ld(sf, &format!("{p}.dwconv.weight"))?,
            dw_b: ld(sf, &format!("{p}.dwconv.bias"))?,
            norm_w: ld(sf, &format!("{p}.norm.weight"))?,
            norm_b: ld(sf, &format!("{p}.norm.bias"))?,
            pw1_w: ld(sf, &format!("{p}.pwconv1.weight"))?,
            pw1_b: ld(sf, &format!("{p}.pwconv1.bias"))?,
            pw2_w: ld(sf, &format!("{p}.pwconv2.weight"))?,
            pw2_b: ld(sf, &format!("{p}.pwconv2.bias"))?,
            grn_gamma: ld(sf, &format!("{p}.grn.gamma"))?,
            grn_beta:  ld(sf, &format!("{p}.grn.beta"))?,
        })
    }
    /// Input/output: `[T, TEXT_HIDDEN]` time-major.  Internally uses channel-major for dwconv.
    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let c = TEXT_HIDDEN;
        // Transpose to [C, T] for depthwise conv.
        let mut ct = vec![0.0f32; c * t];
        for ti in 0..t { for ci in 0..c { ct[ci*t + ti] = x[ti*c + ci]; }}
        // Depthwise k=7 SAME pad.
        let pad = 3;
        let mut after_dw = vec![0.0f32; c * t];
        for ci in 0..c {
            for to in 0..t {
                let mut acc = self.dw_b[ci];
                for kk in 0..7 {
                    let tp = to + kk;
                    if tp < pad { continue; }
                    let xi = tp - pad;
                    if xi >= t { continue; }
                    acc += ct[ci*t + xi] * self.dw_w[ci*7 + kk];
                }
                after_dw[ci*t + to] = acc;
            }
        }
        // Back to [T, C] for LN + pwconv.
        let mut tc = vec![0.0f32; t * c];
        for ti in 0..t { for ci in 0..c { tc[ti*c + ci] = after_dw[ci*t + ti]; }}
        let normed = layer_norm(&tc, t, c, &self.norm_w, &self.norm_b, 1e-6);
        // pwconv1: 512 → 1024 (per-time-step Linear)
        let mut h = linear(&normed, t, c, &self.pw1_w, TEXT_FF, Some(&self.pw1_b));
        for v in h.iter_mut() { *v = gelu(*v); }
        // GRN (global response normalization) on [T, TEXT_FF]
        // gx = ||h(:, c)||_2 over T;  nx = gx / (mean(gx) + eps);  out = gamma * (h * nx) + beta + h
        let mut gx = vec![0.0f32; TEXT_FF];
        for ci in 0..TEXT_FF {
            let mut s = 0.0f32;
            for ti in 0..t { let v = h[ti*TEXT_FF + ci]; s += v*v; }
            gx[ci] = s.sqrt();
        }
        let mean_gx: f32 = gx.iter().sum::<f32>() / TEXT_FF as f32;
        let mut nx = vec![0.0f32; TEXT_FF];
        for ci in 0..TEXT_FF { nx[ci] = gx[ci] / (mean_gx + 1e-6); }
        // ConvNeXt-v2 GRN: out = γ · (x · nx) + β + x   (γ, β shape [1,1,C])
        let mut grn_out = vec![0.0f32; t * TEXT_FF];
        for ti in 0..t {
            for ci in 0..TEXT_FF {
                let hv = h[ti*TEXT_FF + ci];
                grn_out[ti*TEXT_FF + ci] = self.grn_gamma[ci] * (hv * nx[ci]) + self.grn_beta[ci] + hv;
            }
        }
        // pwconv2: 1024 → 512
        let h2 = linear(&grn_out, t, TEXT_FF, &self.pw2_w, c, Some(&self.pw2_b));
        // Residual
        let mut out = vec![0.0f32; t * c];
        for i in 0..(t * c) { out[i] = x[i] + h2[i]; }
        out
    }
}

// ── DiT block ───────────────────────────────────────────────────────────────
struct DiTBlock {
    ada_w: F32Slice, ada_b: F32Slice,     // attn_norm.linear [6*HIDDEN, HIDDEN]
    q_w: F32Slice, q_b: F32Slice,
    k_w: F32Slice, k_b: F32Slice,
    v_w: F32Slice, v_b: F32Slice,
    o_w: F32Slice, o_b: F32Slice,
    ff1_w: F32Slice, ff1_b: F32Slice,
    ff2_w: F32Slice, ff2_b: F32Slice,
}
impl DiTBlock {
    fn load(sf: &SafetensorsFile, i: usize) -> Result<Self, F5Error> {
        let p = format!("ema_model.transformer.transformer_blocks.{i}");
        Ok(Self {
            ada_w: ld(sf, &format!("{p}.attn_norm.linear.weight"))?,
            ada_b: ld(sf, &format!("{p}.attn_norm.linear.bias"))?,
            q_w:   ld(sf, &format!("{p}.attn.to_q.weight"))?,
            q_b:   ld(sf, &format!("{p}.attn.to_q.bias"))?,
            k_w:   ld(sf, &format!("{p}.attn.to_k.weight"))?,
            k_b:   ld(sf, &format!("{p}.attn.to_k.bias"))?,
            v_w:   ld(sf, &format!("{p}.attn.to_v.weight"))?,
            v_b:   ld(sf, &format!("{p}.attn.to_v.bias"))?,
            o_w:   ld(sf, &format!("{p}.attn.to_out.0.weight"))?,
            o_b:   ld(sf, &format!("{p}.attn.to_out.0.bias"))?,
            ff1_w: ld(sf, &format!("{p}.ff.ff.0.0.weight"))?,
            ff1_b: ld(sf, &format!("{p}.ff.ff.0.0.bias"))?,
            ff2_w: ld(sf, &format!("{p}.ff.ff.2.weight"))?,
            ff2_b: ld(sf, &format!("{p}.ff.ff.2.bias"))?,
        })
    }
    /// Forward.  `time_emb` is `[HIDDEN]`; broadcast across all T positions.
    /// `x` and output are `[T, HIDDEN]`.
    /// `freqs_cos`, `freqs_sin` precomputed per position: `[T, HEAD_DIM/2]`.
    fn forward(&self, x: &[f32], t: usize, time_emb: &[f32],
               freqs_cos: &[f32], freqs_sin: &[f32]) -> Vec<f32> {
        // AdaLN-Zero modulation: silu(time) → Linear → split 6 chunks.
        let time_sil: Vec<f32> = time_emb.iter().map(|v| silu(*v)).collect();
        let mod_ = linear(&time_sil, 1, HIDDEN, &self.ada_w, 6 * HIDDEN, Some(&self.ada_b));
        let s1 = &mod_[0..HIDDEN];
        let b1 = &mod_[HIDDEN..2*HIDDEN];
        let g1 = &mod_[2*HIDDEN..3*HIDDEN];
        let s2 = &mod_[3*HIDDEN..4*HIDDEN];
        let b2 = &mod_[4*HIDDEN..5*HIDDEN];
        let g2 = &mod_[5*HIDDEN..6*HIDDEN];

        // Path 1: attn = z + g1 * Attn(LN(z)*(1+s1)+b1)
        let xn = layer_norm_nb(x, t, HIDDEN, LN_EPS);
        let mut x_mod = vec![0.0f32; xn.len()];
        for ti in 0..t {
            for ci in 0..HIDDEN {
                x_mod[ti*HIDDEN + ci] = xn[ti*HIDDEN + ci] * (1.0 + s1[ci]) + b1[ci];
            }
        }
        let mut q = linear(&x_mod, t, HIDDEN, &self.q_w, HIDDEN, Some(&self.q_b));
        let mut k = linear(&x_mod, t, HIDDEN, &self.k_w, HIDDEN, Some(&self.k_b));
        let v     = linear(&x_mod, t, HIDDEN, &self.v_w, HIDDEN, Some(&self.v_b));
        apply_rope(&mut q, t, N_HEADS, freqs_cos, freqs_sin);
        apply_rope(&mut k, t, N_HEADS, freqs_cos, freqs_sin);

        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut per_head = vec![0.0f32; N_HEADS * t * HEAD_DIM];
        per_head.par_chunks_mut(t * HEAD_DIM).enumerate().for_each(|(h, head_buf)| {
            let h_off = h * HEAD_DIM;
            for ti in 0..t {
                let mut scores = vec![0.0f32; t];
                let mut maxv = f32::NEG_INFINITY;
                for tj in 0..t {
                    let mut s = 0.0f32;
                    for d in 0..HEAD_DIM {
                        s += q[ti*HIDDEN + h_off + d] * k[tj*HIDDEN + h_off + d];
                    }
                    s *= scale;
                    scores[tj] = s;
                    if s > maxv { maxv = s; }
                }
                let mut sum = 0.0f32;
                for tj in 0..t { scores[tj] = (scores[tj] - maxv).exp(); sum += scores[tj]; }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for tj in 0..t { scores[tj] *= inv; }
                for d in 0..HEAD_DIM {
                    let mut acc = 0.0f32;
                    for tj in 0..t { acc += scores[tj] * v[tj*HIDDEN + h_off + d]; }
                    head_buf[ti*HEAD_DIM + d] = acc;
                }
            }
        });
        // Restripe + out proj
        let mut attn = vec![0.0f32; t * HIDDEN];
        for h in 0..N_HEADS { for ti in 0..t { for d in 0..HEAD_DIM {
            attn[ti*HIDDEN + h*HEAD_DIM + d] = per_head[h*t*HEAD_DIM + ti*HEAD_DIM + d];
        }}}
        let attn_proj = linear(&attn, t, HIDDEN, &self.o_w, HIDDEN, Some(&self.o_b));

        let mut x2 = vec![0.0f32; t * HIDDEN];
        for ti in 0..t { for ci in 0..HIDDEN {
            x2[ti*HIDDEN + ci] = x[ti*HIDDEN + ci] + g1[ci] * attn_proj[ti*HIDDEN + ci];
        }}

        // Path 2: FFN with AdaLN-Zero.
        let xn = layer_norm_nb(&x2, t, HIDDEN, LN_EPS);
        let mut x_mod = vec![0.0f32; xn.len()];
        for ti in 0..t { for ci in 0..HIDDEN {
            x_mod[ti*HIDDEN + ci] = xn[ti*HIDDEN + ci] * (1.0 + s2[ci]) + b2[ci];
        }}
        let mut ff = linear(&x_mod, t, HIDDEN, &self.ff1_w, FF, Some(&self.ff1_b));
        for v in ff.iter_mut() { *v = gelu(*v); }
        let ff_out = linear(&ff, t, FF, &self.ff2_w, HIDDEN, Some(&self.ff2_b));
        let mut out = vec![0.0f32; t * HIDDEN];
        for ti in 0..t { for ci in 0..HIDDEN {
            out[ti*HIDDEN + ci] = x2[ti*HIDDEN + ci] + g2[ci] * ff_out[ti*HIDDEN + ci];
        }}
        out
    }
}

fn apply_rope(x: &mut [f32], t: usize, n_heads: usize, freqs_cos: &[f32], freqs_sin: &[f32]) {
    let half = HEAD_DIM / 2;
    debug_assert_eq!(freqs_cos.len(), t * half);
    for ti in 0..t {
        for h in 0..n_heads {
            let base = ti * n_heads * HEAD_DIM + h * HEAD_DIM;
            for k in 0..half {
                let c = freqs_cos[ti * half + k];
                let s = freqs_sin[ti * half + k];
                let a = x[base + k];
                let b = x[base + k + half];
                x[base + k]        = a * c - b * s;
                x[base + k + half] = a * s + b * c;
            }
        }
    }
}

// ── model ──────────────────────────────────────────────────────────────────
pub struct F5TTS {
    // text encoder
    text_embed: F32Slice,           // [VOCAB, TEXT_HIDDEN]; index 0 IS PAD in the raw checkpoint
    text_blocks: Vec<TextBlock>,
    // time embed
    time_mlp0_w: F32Slice, time_mlp0_b: F32Slice,
    time_mlp2_w: F32Slice, time_mlp2_b: F32Slice,
    // input embed
    proj_w: F32Slice, proj_b: F32Slice,
    pos_conv_w0: F32Slice, pos_conv_b0: F32Slice,
    pos_conv_w2: F32Slice, pos_conv_b2: F32Slice,
    // DiT
    dit: Vec<DiTBlock>,
    norm_out_w: F32Slice, norm_out_b: F32Slice,   // [2*HIDDEN, HIDDEN]
    proj_out_w: F32Slice, proj_out_b: F32Slice,   // [MEL_DIM, HIDDEN]
    // RoPE
    inv_freq: F32Slice,             // [HEAD_DIM/2]
}
impl F5TTS {
    pub fn load(path: &Path) -> Result<Arc<Self>, F5Error> {
        let sf = SafetensorsFile::load(path).map_err(|e| F5Error::Load(e.to_string()))?;
        let text_embed_raw = ld(&sf, "ema_model.transformer.text_embed.text_embed.weight")?;
        // text_embed [VOCAB=2546, 512]. F5 uses id 0 for PAD; the model itself has 2546 slots
        // where index 0 IS the PAD embedding.  So we use the raw table directly.
        let mut text_blocks = Vec::with_capacity(N_TEXT_BLOCKS);
        for i in 0..N_TEXT_BLOCKS { text_blocks.push(TextBlock::load(&sf, i)?); }
        let mut dit = Vec::with_capacity(N_DIT);
        for i in 0..N_DIT { dit.push(DiTBlock::load(&sf, i)?); }
        Ok(Arc::new(Self {
            text_embed: text_embed_raw,
            text_blocks,
            time_mlp0_w: ld(&sf, "ema_model.transformer.time_embed.time_mlp.0.weight")?,
            time_mlp0_b: ld(&sf, "ema_model.transformer.time_embed.time_mlp.0.bias")?,
            time_mlp2_w: ld(&sf, "ema_model.transformer.time_embed.time_mlp.2.weight")?,
            time_mlp2_b: ld(&sf, "ema_model.transformer.time_embed.time_mlp.2.bias")?,
            proj_w: ld(&sf, "ema_model.transformer.input_embed.proj.weight")?,
            proj_b: ld(&sf, "ema_model.transformer.input_embed.proj.bias")?,
            pos_conv_w0: ld(&sf, "ema_model.transformer.input_embed.conv_pos_embed.conv1d.0.weight")?,
            pos_conv_b0: ld(&sf, "ema_model.transformer.input_embed.conv_pos_embed.conv1d.0.bias")?,
            pos_conv_w2: ld(&sf, "ema_model.transformer.input_embed.conv_pos_embed.conv1d.2.weight")?,
            pos_conv_b2: ld(&sf, "ema_model.transformer.input_embed.conv_pos_embed.conv1d.2.bias")?,
            dit,
            norm_out_w: ld(&sf, "ema_model.transformer.norm_out.linear.weight")?,
            norm_out_b: ld(&sf, "ema_model.transformer.norm_out.linear.bias")?,
            proj_out_w: ld(&sf, "ema_model.transformer.proj_out.weight")?,
            proj_out_b: ld(&sf, "ema_model.transformer.proj_out.bias")?,
            inv_freq:  ld(&sf, "ema_model.transformer.rotary_embed.inv_freq")?,
        }))
    }

    /// Encode text token ids → text features `[T, TEXT_HIDDEN]`.
    fn encode_text(&self, ids: &[u32]) -> Vec<f32> {
        let t = ids.len();
        let mut x = vec![0.0f32; t * TEXT_HIDDEN];
        for (ti, &id) in ids.iter().enumerate() {
            let src = &self.text_embed[(id as usize)*TEXT_HIDDEN..(id as usize+1)*TEXT_HIDDEN];
            x[ti*TEXT_HIDDEN..(ti+1)*TEXT_HIDDEN].copy_from_slice(src);
        }
        for blk in &self.text_blocks { x = blk.forward(&x, t); }
        x
    }

    fn time_embed(&self, t_scalar: f32) -> Vec<f32> {
        // Sinusoidal embedding of dimension TIME_SIN_DIM, scaled by `t * 1000` for diffusion
        // convention.  Half cosines, half sines.
        let dim = TIME_SIN_DIM;
        let mut emb = vec![0.0f32; dim];
        let half = dim / 2;
        let factor = (1e4_f32).ln() / (half - 1) as f32;
        let scaled = t_scalar * 1000.0;
        for i in 0..half {
            let freq = (-factor * i as f32).exp();
            emb[i]        = (scaled * freq).cos();
            emb[i + half] = (scaled * freq).sin();
        }
        // time_mlp: Linear → SiLU → Linear
        let mut h = linear(&emb, 1, TIME_SIN_DIM, &self.time_mlp0_w, HIDDEN, Some(&self.time_mlp0_b));
        for v in h.iter_mut() { *v = silu(*v); }
        linear(&h, 1, HIDDEN, &self.time_mlp2_w, HIDDEN, Some(&self.time_mlp2_b))
    }

    /// Run one DiT forward.  `text_feats` and `y_t` and `cond_mel` are all `[T, *]`.
    fn dit_forward(
        &self,
        y_t: &[f32], t: usize,
        text_feats: &[f32], cond_mel: &[f32],
        time_scalar: f32,
    ) -> Vec<f32> {
        // Build joint input [T, 712] = concat(y_t [100], text_feats [512], cond_mel [100]).
        let mut joint = vec![0.0f32; t * INPUT_PROJ_IN];
        for ti in 0..t {
            joint[ti*INPUT_PROJ_IN..ti*INPUT_PROJ_IN + MEL_DIM]
                .copy_from_slice(&y_t[ti*MEL_DIM..(ti+1)*MEL_DIM]);
            joint[ti*INPUT_PROJ_IN + MEL_DIM..ti*INPUT_PROJ_IN + MEL_DIM + TEXT_HIDDEN]
                .copy_from_slice(&text_feats[ti*TEXT_HIDDEN..(ti+1)*TEXT_HIDDEN]);
            joint[ti*INPUT_PROJ_IN + MEL_DIM + TEXT_HIDDEN..ti*INPUT_PROJ_IN + INPUT_PROJ_IN]
                .copy_from_slice(&cond_mel[ti*MEL_DIM..(ti+1)*MEL_DIM]);
        }
        // Linear 712 → 1024
        let mut z = linear(&joint, t, INPUT_PROJ_IN, &self.proj_w, HIDDEN, Some(&self.proj_b));
        // conv_pos_embed: depthwise conv k=31 (channel-major) → mish → conv k=31 → add residual
        let mut ct = vec![0.0f32; HIDDEN * t];
        for ti in 0..t { for ci in 0..HIDDEN { ct[ci*t + ti] = z[ti*HIDDEN + ci]; }}
        let h0 = conv1d(&ct, HIDDEN, t, &self.pos_conv_w0, HIDDEN, POS_CONV_K,
                        POS_CONV_GROUPS, POS_CONV_K / 2, &self.pos_conv_b0);
        let mut h0 = h0;
        for v in h0.iter_mut() { let s = silu(*v); *v = s; }
        let h1 = conv1d(&h0, HIDDEN, t, &self.pos_conv_w2, HIDDEN, POS_CONV_K,
                        POS_CONV_GROUPS, POS_CONV_K / 2, &self.pos_conv_b2);
        // Add residual (in channel-major)
        for ci in 0..HIDDEN { for ti in 0..t {
            ct[ci*t + ti] += h1[ci*t + ti];
        }}
        // Back to time-major
        for ti in 0..t { for ci in 0..HIDDEN { z[ti*HIDDEN + ci] = ct[ci*t + ti]; }}

        // Time embedding
        let time_emb = self.time_embed(time_scalar);

        // Precompute RoPE freqs for length T.
        let half = HEAD_DIM / 2;
        let mut fcos = vec![0.0f32; t * half];
        let mut fsin = vec![0.0f32; t * half];
        for ti in 0..t {
            for k in 0..half {
                let theta = ti as f32 * self.inv_freq[k];
                fcos[ti*half + k] = theta.cos();
                fsin[ti*half + k] = theta.sin();
            }
        }

        // 22 DiT blocks
        for blk in &self.dit {
            z = blk.forward(&z, t, &time_emb, &fcos, &fsin);
        }

        // Final AdaLN: silu(time) → Linear → split (s, b)
        let time_sil: Vec<f32> = time_emb.iter().map(|v| silu(*v)).collect();
        let mod_ = linear(&time_sil, 1, HIDDEN, &self.norm_out_w, 2 * HIDDEN, Some(&self.norm_out_b));
        let s = &mod_[0..HIDDEN];
        let b = &mod_[HIDDEN..2*HIDDEN];
        let zn = layer_norm_nb(&z, t, HIDDEN, LN_EPS);
        let mut z_mod = vec![0.0f32; zn.len()];
        for ti in 0..t { for ci in 0..HIDDEN {
            z_mod[ti*HIDDEN + ci] = zn[ti*HIDDEN + ci] * (1.0 + s[ci]) + b[ci];
        }}
        // proj_out: 1024 → 100 (predicted velocity)
        linear(&z_mod, t, HIDDEN, &self.proj_out_w, MEL_DIM, Some(&self.proj_out_b))
    }

    /// Synthesize mel from text via Euler ODE flow matching.  `n_steps` is typically 16–32.
    /// `cond_mel` is the reference mel (zeros for unconditional / no voice clone).
    /// `text_ids` is the *combined* (ref + gen) token sequence padded to `t_total` length.
    pub fn sample_mel(
        &self,
        text_ids: &[u32],
        cond_mel: &[f32],          // [t_total, MEL_DIM] — reference mel + zero-pad for gen
        n_steps: usize,
        rng_seed: u64,
    ) -> Vec<f32> {
        let t = text_ids.len();
        assert_eq!(cond_mel.len(), t * MEL_DIM);
        let text_feats = self.encode_text(text_ids);
        let mut rng = Xorshift::new(rng_seed);
        let mut y = vec![0.0f32; t * MEL_DIM];
        for v in y.iter_mut() { *v = rng.gaussian(); }
        let dt = 1.0 / n_steps as f32;
        for step in 0..n_steps {
            let ti = step as f32 * dt;
            let v_pred = self.dit_forward(&y, t, &text_feats, cond_mel, ti);
            for i in 0..y.len() { y[i] += dt * v_pred[i]; }
        }
        y
    }

    /// CFG-sampled mel.  At each Euler step, runs the model twice — once with
    /// the real text, once with a null-text (all PAD) — and combines the two
    /// velocity predictions per the standard formulation
    ///   v = v_uncond + cfg_strength · (v_cond − v_uncond).
    /// F5-TTS uses cfg_strength ≈ 2.0 by default.  Doubles the per-step cost
    /// vs `sample_mel` but is what produces coherent speech.
    pub fn sample_mel_cfg(
        &self,
        text_ids: &[u32],
        cond_mel: &[f32],
        n_steps: usize,
        cfg_strength: f32,
        rng_seed: u64,
    ) -> Vec<f32> {
        let t = text_ids.len();
        assert_eq!(cond_mel.len(), t * MEL_DIM);
        let text_feats = self.encode_text(text_ids);
        // Null text features: same length but ALL PAD tokens (id 0).
        let null_ids: Vec<u32> = vec![0; t];
        let null_text_feats = self.encode_text(&null_ids);

        let mut rng = Xorshift::new(rng_seed);
        let mut y = vec![0.0f32; t * MEL_DIM];
        for v in y.iter_mut() { *v = rng.gaussian(); }
        let dt = 1.0 / n_steps as f32;
        for step in 0..n_steps {
            let ti = step as f32 * dt;
            let v_cond   = self.dit_forward(&y, t, &text_feats,      cond_mel, ti);
            let v_uncond = self.dit_forward(&y, t, &null_text_feats, cond_mel, ti);
            for i in 0..y.len() {
                let v = v_uncond[i] + cfg_strength * (v_cond[i] - v_uncond[i]);
                y[i] += dt * v;
            }
        }
        y
    }
}

// ── deterministic Gaussian PRNG (Box-Muller on xorshift) ────────────────────
pub struct Xorshift { state: u64, spare: Option<f32> }
impl Xorshift {
    pub fn new(seed: u64) -> Self { Self { state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed }, spare: None } }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state; x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.state = x; (x >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u32() >> 8) as f32 + 1.0) / ((1u32 << 24) as f32 + 1.0)
    }
    pub fn gaussian(&mut self) -> f32 {
        if let Some(s) = self.spare.take() { return s; }
        let u1 = self.next_f32();
        let u2 = self.next_f32();
        let mag = (-2.0 * u1.ln()).sqrt();
        let z0 = mag * (2.0 * std::f32::consts::PI * u2).cos();
        let z1 = mag * (2.0 * std::f32::consts::PI * u2).sin();
        self.spare = Some(z1);
        z0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Instant;

    fn model_path() -> Option<PathBuf> {
        let p = PathBuf::from("/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/f5-tts/F5TTS_v1_Base/model_1250000.safetensors");
        if p.exists() { Some(p) } else { None }
    }

    /// Voice-cloning end-to-end: load reference mel from `/tmp/ref_mel.f32`
    /// (produced offline by the F5-style mel extractor in the session's test
    /// scripts), build the concatenated `[ref_text + gen_text]` token sequence
    /// and `[ref_mel + zero_pad]` condition, CFG-sample for 32 steps, decode
    /// the gen portion via vocos, and write a WAV.  This is the path that
    /// produces intelligible speech.
    #[test]
    #[ignore = "loads F5-TTS + vocos; 32 step CFG forward of full ref+gen sequence; writes /tmp/f5_voice_clone.wav"]
    fn f5tts_voice_clone() {
        let Some(p) = model_path() else { eprintln!("skip"); return; };
        let m = F5TTS::load(&p).expect("load");
        let vocab_path = PathBuf::from("/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/f5-tts/F5TTS_v1_Base/vocab.txt");
        let tok = Tokenizer::from_vocab_file(&vocab_path).expect("vocab");

        // Reference mel: 376 frames × 100, produced by /tmp Python helper.
        let ref_path = PathBuf::from("/tmp/ref_mel.f32");
        if !ref_path.exists() {
            eprintln!("skip: /tmp/ref_mel.f32 missing — run the Python mel extractor first");
            return;
        }
        let bytes = std::fs::read(&ref_path).expect("ref mel");
        let ref_mel: Vec<f32> = bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
        let ref_frames = ref_mel.len() / MEL_DIM;
        eprintln!("ref mel: {ref_frames} frames");

        let ref_text = "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background.";
        let gen_text = "Hello world.";
        let ref_ids = tok.encode(ref_text);
        let gen_ids = tok.encode(gen_text);
        // Estimate gen mel length from char ratio: gen_frames = ref_frames * gen_len / ref_len.
        let gen_frames_est = (ref_frames as f32 * gen_ids.len() as f32 / ref_ids.len() as f32).round() as usize;
        let gen_frames = gen_frames_est.max(20);
        let t_total = ref_frames + gen_frames;
        eprintln!("ref text {} chars → ref {} frames; gen text {} chars → gen {} frames (total {})",
                  ref_ids.len(), ref_frames, gen_ids.len(), gen_frames, t_total);

        // Concatenated token sequence: ref_ids + gen_ids, then PAD to t_total.
        let mut text_ids: Vec<u32> = Vec::with_capacity(t_total);
        text_ids.extend_from_slice(&ref_ids);
        text_ids.extend_from_slice(&gen_ids);
        while text_ids.len() < t_total { text_ids.push(0); }

        // cond_mel: [ref_mel | zero_pad for gen]
        let mut cond = vec![0.0f32; t_total * MEL_DIM];
        cond[..ref_frames * MEL_DIM].copy_from_slice(&ref_mel);

        let t0 = Instant::now();
        let full_mel = m.sample_mel_cfg(&text_ids, &cond, 32, 2.0, 0xC0FFEE);
        let dt = t0.elapsed();
        eprintln!("sampled mel [{t_total} × {MEL_DIM}] CFG=2 32 steps in {dt:?}");

        // Take the gen portion only.
        let gen_mel = &full_mel[ref_frames * MEL_DIM..];

        // Reshape to [100, gen_frames] for vocos.
        let mut mel_ct = vec![0.0f32; gen_mel.len()];
        for ti in 0..gen_frames { for ci in 0..MEL_DIM {
            mel_ct[ci*gen_frames + ti] = gen_mel[ti*MEL_DIM + ci];
        }}

        let vocos_dir = PathBuf::from("/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/vocos");
        let v = rce_vocos::Vocos::load(&vocos_dir).expect("vocos");
        let t0 = Instant::now();
        let wav = v.vocode(&mel_ct, gen_frames);
        eprintln!("vocoded {} samples in {:?}", wav.len(), t0.elapsed());

        let peak = wav.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        let rms: f32 = (wav.iter().map(|x| x*x).sum::<f32>() / wav.len() as f32).sqrt();
        eprintln!("wav peak={peak:.4} rms={rms:.4}");
        let sr = rce_vocos::SR as u32;
        write_wav("/tmp/f5_voice_clone.wav", &wav, sr).expect("write");
        eprintln!("wrote /tmp/f5_voice_clone.wav ({} samples @ {} Hz)", wav.len(), sr);
        assert!(peak.is_finite() && peak > 0.001, "essentially silent");
    }

    /// Full end-to-end synthesis: sample mel via 32-step Euler CFG, decode via vocos,
    /// write a WAV.  Without a reference voice (zero cond_mel) the output won't
    /// be coherent speech — but it should produce finite, non-silent audio with
    /// a reasonable peak, which validates the full pipeline.
    #[test]
    #[ignore = "loads F5-TTS (1.35 GB) + vocos; 32-step CFG flow matching forward; writes /tmp/f5_demo.wav"]
    fn f5tts_end_to_end() {
        let Some(p) = model_path() else { eprintln!("skip"); return; };
        let m = F5TTS::load(&p).expect("load");
        let vocab_path = PathBuf::from("/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/f5-tts/F5TTS_v1_Base/vocab.txt");
        let tok = Tokenizer::from_vocab_file(&vocab_path).expect("vocab");

        // Pad text out to ~150 frames (~3 seconds @ 50 Hz).
        let prompt = "Hello world. The birch canoe slid on the smooth planks.";
        let mut ids = tok.encode(prompt);
        while ids.len() < 150 { ids.push(0); }
        let t = ids.len();
        let cond = vec![0.0f32; t * MEL_DIM];

        let t0 = Instant::now();
        let mel = m.sample_mel_cfg(&ids, &cond, 32, 2.0, 0xC0FFEE);
        let dt = t0.elapsed();
        eprintln!("sampled mel [{} frames × {}] CFG=2.0 32 steps in {dt:?}", mel.len()/MEL_DIM, MEL_DIM);
        assert!(mel.iter().all(|v| v.is_finite()), "non-finite mel");

        // Reshape mel from [T, 100] to [100, T] for vocos input.
        let mut mel_ct = vec![0.0f32; mel.len()];
        for ti in 0..t { for ci in 0..MEL_DIM {
            mel_ct[ci*t + ti] = mel[ti*MEL_DIM + ci];
        }}

        let vocos_dir = PathBuf::from("/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/vocos");
        if !vocos_dir.exists() { eprintln!("skip: vocos missing"); return; }
        let v = rce_vocos::Vocos::load(&vocos_dir).expect("vocos");
        let t0 = Instant::now();
        let wav = v.vocode(&mel_ct, t);
        eprintln!("vocoded {} samples in {:?}", wav.len(), t0.elapsed());

        let peak = wav.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        let rms: f32 = (wav.iter().map(|x| x*x).sum::<f32>() / wav.len() as f32).sqrt();
        eprintln!("wav peak={peak:.4} rms={rms:.4}");
        // Write a WAV so we can listen.
        let sr = rce_vocos::SR as u32;
        write_wav("/tmp/f5_demo.wav", &wav, sr).expect("write");
        eprintln!("wrote /tmp/f5_demo.wav ({} samples @ {} Hz)", wav.len(), sr);
        assert!(peak.is_finite());
        assert!(peak > 0.001, "essentially silent output");
    }

    fn write_wav(path: &str, samples: &[f32], sr: u32) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        let bits = 16u16; let ch = 1u16;
        let byte_rate = sr * ch as u32 * bits as u32 / 8;
        let block_align = ch * bits / 8;
        let data_size = (samples.len() as u32) * (bits as u32 / 8);
        let chunk_size = 36 + data_size;
        f.write_all(b"RIFF")?;
        f.write_all(&chunk_size.to_le_bytes())?;
        f.write_all(b"WAVE")?;
        f.write_all(b"fmt ")?;
        f.write_all(&16u32.to_le_bytes())?;
        f.write_all(&1u16.to_le_bytes())?;
        f.write_all(&ch.to_le_bytes())?;
        f.write_all(&sr.to_le_bytes())?;
        f.write_all(&byte_rate.to_le_bytes())?;
        f.write_all(&block_align.to_le_bytes())?;
        f.write_all(&bits.to_le_bytes())?;
        f.write_all(b"data")?;
        f.write_all(&data_size.to_le_bytes())?;
        for &s in samples {
            let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
            f.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    /// Phase-1 smoke test: weights load, tokenizer encodes, a single DiT forward
    /// produces finite-range velocity predictions.  Skip ODE iteration for speed.
    #[test]
    #[ignore = "loads 1.35 GB f32; single forward of 22 DiT blocks"]
    fn f5tts_smoke() {
        let Some(p) = model_path() else { eprintln!("skip"); return; };
        eprintln!("loading {}…", p.display());
        let t0 = Instant::now();
        let m = F5TTS::load(&p).expect("load");
        eprintln!("loaded in {:?}", t0.elapsed());

        let vocab_path = PathBuf::from("/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/f5-tts/F5TTS_v1_Base/vocab.txt");
        let tok = Tokenizer::from_vocab_file(&vocab_path).expect("vocab");
        let ids = tok.encode("Hello world.");
        eprintln!("tokenized {ids:?}");
        assert!(!ids.is_empty());

        // Run one DiT forward with random y, zero cond mel.
        let text_feats = m.encode_text(&ids);
        let t = ids.len();
        let mut y = vec![0.0f32; t * MEL_DIM];
        let mut rng = Xorshift::new(0xCAFE);
        for v in y.iter_mut() { *v = rng.gaussian(); }
        let cond = vec![0.0f32; t * MEL_DIM];

        let t0 = Instant::now();
        let v_pred = m.dit_forward(&y, t, &text_feats, &cond, 0.5);
        let dt = t0.elapsed();
        eprintln!("single DiT forward at t=0.5 ({t} frames) in {dt:?}");

        assert_eq!(v_pred.len(), t * MEL_DIM);
        let abs_max = v_pred.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let abs_mean = v_pred.iter().map(|v| v.abs()).sum::<f32>() / v_pred.len() as f32;
        eprintln!("v_pred |max|={abs_max:.4} |mean|={abs_mean:.4}");
        assert!(v_pred.iter().all(|v| v.is_finite()));
        assert!(abs_max > 0.0);
    }
}
