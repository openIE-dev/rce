//! Moonshine ASR on the rce substrate.
//!
//! rce's first **encoder-decoder + cross-attention** model and first
//! speech-to-text runtime. Architecture (per HF `modeling_moonshine.py`):
//!
//! - **Raw-audio conv stem** (no log-mel): `conv1(1→416, k127, s64, bias=0)`
//!   → tanh → GroupNorm(1 group) → `conv2(416→832, k7, s3)` → GELU →
//!   `conv3(832→416, k3, s2)` → GELU → permute to `[T, 416]`.
//! - **8 encoder layers** (pre-norm, bidirectional): LN → rotary self-attn
//!   → +res; LN → GELU-MLP → +res. Final encoder LayerNorm.
//! - **8 decoder layers** (pre-norm): LN → causal rotary self-attn → +res;
//!   LN → cross-attn (q=decoder, k/v=encoder, no rope) → +res; LN →
//!   SwiGLU-MLP → +res. Final norm + tied lm_head (`proj_out` = embed).
//! - **Interleaved partial RoPE**: rotary_dim = `int(52 * 0.62) = 32` of
//!   head_dim 52; GPT-J pairing `(x[2i], x[2i+1])`. Applied to self-attn
//!   only (encoder + decoder), never cross-attn.
//! - Heads: 8 × 52, scale = `52^-0.5`. All LayerNorm are bias-free but
//!   **mean-subtracting** (not RMSNorm). All weights F32.

use rce_op_attn::fused_attend_head;
use rce_weight_loader::{F32Slice, SafetensorsFile};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

const D_MODEL: usize = 416;
const N_HEADS: usize = 8;
const D_HEAD: usize = 52; // 416 / 8
const ROTARY_DIM: usize = 32; // int(52 * 0.62), GPT-J interleaved pairs
const ROPE_BASE: f32 = 10_000.0;
const N_ENC: usize = 8;
const N_DEC: usize = 8;
const ENC_FF: usize = 1664;
const DEC_FF: usize = 1664; // fc1 emits 2× (gate ∥ value), fc2 takes 1664
const VOCAB: usize = 32768;
const EPS: f32 = 1e-5;
const BOS: i32 = 1; // decoder_start_token_id
const EOS: i32 = 2;

// ── small numeric helpers ──────────────────────────────────────────

#[inline] fn gelu(x: f32) -> f32 {
    // tanh approximation (HF `gelu` default for these convs/MLPs).
    let c = 0.797_884_56_f32; // sqrt(2/pi)
    0.5 * x * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())
}
#[inline] fn silu(x: f32) -> f32 {
    if x >= 0.0 { x / (1.0 + (-x).exp()) } else { let e = x.exp(); x * e / (1.0 + e) }
}

/// `y[T, d_out] = x[T, d_in] · W[d_out, d_in]^T (+ bias)`. Delegates to
/// `rce_op_conv2d::linear_f32` (Apple Accelerate sgemm on macOS).
fn linear(x: &[f32], t: usize, w: &[f32], d_out: usize, d_in: usize, bias: Option<&[f32]>) -> Vec<f32> {
    rce_op_conv2d::linear_f32(x, t, d_in, w, d_out, bias)
}

/// Bias-free mean-subtracting LayerNorm over the last dim, per row
/// (row count is `x.len() / d`).
fn layernorm_rows(x: &mut [f32], d: usize, w: &[f32]) {
    x.par_chunks_mut(d).for_each(|row| {
        let mean: f32 = row.iter().sum::<f32>() / d as f32;
        let var: f32 = row.iter().map(|v| { let z = v - mean; z * z }).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        for (v, g) in row.iter_mut().zip(w.iter()) { *v = (*v - mean) * inv * *g; }
    });
}

/// Strided Conv1d, no padding. `inp` is `[c_in, l_in]` row-major,
/// `w` is `[c_out, c_in, k]`. Returns `([c_out, l_out], l_out)`.
fn conv1d(
    inp: &[f32], c_in: usize, l_in: usize,
    w: &[f32], c_out: usize, k: usize, stride: usize, bias: Option<&[f32]>,
) -> (Vec<f32>, usize) {
    let l_out = (l_in - k) / stride + 1;
    let mut out = vec![0.0f32; c_out * l_out];
    out.par_chunks_mut(l_out).enumerate().for_each(|(oc, orow)| {
        let wbase = oc * c_in * k;
        let b0 = bias.map_or(0.0, |b| b[oc]);
        for (t, slot) in orow.iter_mut().enumerate() {
            let start = t * stride;
            let mut acc = b0;
            for ic in 0..c_in {
                let wic = &w[wbase + ic * k..wbase + ic * k + k];
                let inic = &inp[ic * l_in + start..ic * l_in + start + k];
                for kk in 0..k { acc += wic[kk] * inic[kk]; }
            }
            *slot = acc;
        }
    });
    (out, l_out)
}

/// GroupNorm with a single group over `[c, t]` (channel-major): one
/// global mean/var across all c·t elements, then per-channel affine.
fn groupnorm1(x: &mut [f32], c: usize, t: usize, w: &[f32], b: &[f32]) {
    let n = (c * t) as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|v| { let z = v - mean; z * z }).sum::<f32>() / n;
    let inv = 1.0 / (var + EPS).sqrt();
    for ch in 0..c {
        let (g, bb) = (w[ch], b[ch]);
        for tt in 0..t { let idx = ch * t + tt; x[idx] = (x[idx] - mean) * inv * g + bb; }
    }
}

fn transpose(x: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; c * t];
    for ch in 0..c { for tt in 0..t { o[tt * c + ch] = x[ch * t + tt]; } }
    o
}

/// Interleaved (GPT-J) partial RoPE on one head's `[D_HEAD]` slice:
/// rotate adjacent pairs `(x[2i], x[2i+1])` for i in 0..ROTARY_DIM/2,
/// pass-through dims ≥ ROTARY_DIM.
fn rope_partial(head: &mut [f32], pos: usize) {
    for i in 0..(ROTARY_DIM / 2) {
        let inv_freq = ROPE_BASE.powf(-2.0 * i as f32 / ROTARY_DIM as f32);
        let (s, c) = (pos as f32 * inv_freq).sin_cos();
        let a = head[2 * i];
        let b = head[2 * i + 1];
        head[2 * i] = a * c - b * s;
        head[2 * i + 1] = b * c + a * s;
    }
}

#[inline] fn scale() -> f32 { 1.0 / (D_HEAD as f32).sqrt() }

// ── weights ────────────────────────────────────────────────────────

struct EncLayer {
    ln_in: F32Slice, ln_post: F32Slice,
    q: F32Slice, k: F32Slice, v: F32Slice, o: F32Slice,
    fc1: F32Slice, fc1_b: F32Slice, fc2: F32Slice, fc2_b: F32Slice,
}
struct DecLayer {
    ln_in: F32Slice, ln_post: F32Slice, ln_final: F32Slice,
    q: F32Slice, k: F32Slice, v: F32Slice, o: F32Slice,
    cq: F32Slice, ck: F32Slice, cv: F32Slice, co: F32Slice,
    fc1: F32Slice, fc1_b: F32Slice, fc2: F32Slice, fc2_b: F32Slice,
}

pub struct Moonshine {
    // encoder conv stem
    conv1_w: F32Slice, conv2_w: F32Slice, conv2_b: F32Slice, conv3_w: F32Slice, conv3_b: F32Slice,
    gn_w: F32Slice, gn_b: F32Slice, enc_ln: F32Slice,
    enc: Vec<EncLayer>,
    // decoder
    embed: F32Slice, // [VOCAB, D_MODEL] — also tied lm_head
    dec: Vec<DecLayer>,
    dec_norm: F32Slice,
    // detokenizer
    id2piece: HashMap<i32, String>,
}

fn v(f: &SafetensorsFile, name: &str) -> F32Slice {
    let t = f.get(name).unwrap_or_else(|| panic!("missing tensor {name}"));
    F32Slice::from_tensor(name, t).unwrap_or_else(|e| panic!("{e}"))
}

impl Moonshine {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = model_dir.as_ref();
        let f = SafetensorsFile::load(dir.join("model.safetensors"))
            .map_err(|e| format!("load safetensors: {e:?}"))?;

        let enc = (0..N_ENC).map(|i| {
            let p = format!("model.encoder.layers.{i}");
            EncLayer {
                ln_in: v(&f, &format!("{p}.input_layernorm.weight")),
                ln_post: v(&f, &format!("{p}.post_attention_layernorm.weight")),
                q: v(&f, &format!("{p}.self_attn.q_proj.weight")),
                k: v(&f, &format!("{p}.self_attn.k_proj.weight")),
                v: v(&f, &format!("{p}.self_attn.v_proj.weight")),
                o: v(&f, &format!("{p}.self_attn.o_proj.weight")),
                fc1: v(&f, &format!("{p}.mlp.fc1.weight")),
                fc1_b: v(&f, &format!("{p}.mlp.fc1.bias")),
                fc2: v(&f, &format!("{p}.mlp.fc2.weight")),
                fc2_b: v(&f, &format!("{p}.mlp.fc2.bias")),
            }
        }).collect();

        let dec = (0..N_DEC).map(|i| {
            let p = format!("model.decoder.layers.{i}");
            DecLayer {
                ln_in: v(&f, &format!("{p}.input_layernorm.weight")),
                ln_post: v(&f, &format!("{p}.post_attention_layernorm.weight")),
                ln_final: v(&f, &format!("{p}.final_layernorm.weight")),
                q: v(&f, &format!("{p}.self_attn.q_proj.weight")),
                k: v(&f, &format!("{p}.self_attn.k_proj.weight")),
                v: v(&f, &format!("{p}.self_attn.v_proj.weight")),
                o: v(&f, &format!("{p}.self_attn.o_proj.weight")),
                cq: v(&f, &format!("{p}.encoder_attn.q_proj.weight")),
                ck: v(&f, &format!("{p}.encoder_attn.k_proj.weight")),
                cv: v(&f, &format!("{p}.encoder_attn.v_proj.weight")),
                co: v(&f, &format!("{p}.encoder_attn.o_proj.weight")),
                fc1: v(&f, &format!("{p}.mlp.fc1.weight")),
                fc1_b: v(&f, &format!("{p}.mlp.fc1.bias")),
                fc2: v(&f, &format!("{p}.mlp.fc2.weight")),
                fc2_b: v(&f, &format!("{p}.mlp.fc2.bias")),
            }
        }).collect();

        let id2piece = load_vocab(&dir.join("tokenizer.json"))?;

        Ok(Self {
            conv1_w: v(&f, "model.encoder.conv1.weight"),
            conv2_w: v(&f, "model.encoder.conv2.weight"),
            conv2_b: v(&f, "model.encoder.conv2.bias"),
            conv3_w: v(&f, "model.encoder.conv3.weight"),
            conv3_b: v(&f, "model.encoder.conv3.bias"),
            gn_w: v(&f, "model.encoder.groupnorm.weight"),
            gn_b: v(&f, "model.encoder.groupnorm.bias"),
            enc_ln: v(&f, "model.encoder.layer_norm.weight"),
            enc,
            embed: v(&f, "model.decoder.embed_tokens.weight"),
            dec,
            dec_norm: v(&f, "model.decoder.norm.weight"),
            id2piece,
        })
    }

    /// Run the conv stem + 8 encoder layers on a 16 kHz mono waveform.
    /// Returns `(enc_hidden [T, D_MODEL], T)`.
    pub fn encode(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        // conv stem (channel-major [C, L]).
        let (h1, l1) = conv1d(audio, 1, audio.len(), &self.conv1_w, D_MODEL, 127, 64, None);
        let mut h1: Vec<f32> = h1.iter().map(|x| x.tanh()).collect();
        groupnorm1(&mut h1, D_MODEL, l1, &self.gn_w, &self.gn_b);
        let (h2, l2) = conv1d(&h1, D_MODEL, l1, &self.conv2_w, 2 * D_MODEL, 7, 3, Some(&self.conv2_b));
        let h2: Vec<f32> = h2.iter().map(|x| gelu(*x)).collect();
        let (h3, l3) = conv1d(&h2, 2 * D_MODEL, l2, &self.conv3_w, D_MODEL, 3, 2, Some(&self.conv3_b));
        let h3: Vec<f32> = h3.iter().map(|x| gelu(*x)).collect();
        let t = l3;
        let mut hidden = transpose(&h3, D_MODEL, t); // [T, D_MODEL]

        for layer in &self.enc {
            // self-attn (bidirectional, rotary)
            let mut xn = hidden.clone();
            layernorm_rows(&mut xn, D_MODEL, &layer.ln_in);
            let attn = self.attend_full(&xn, t, &layer.q, &layer.k, &layer.v, &layer.o, true, false);
            for i in 0..hidden.len() { hidden[i] += attn[i]; }
            // GELU MLP
            let mut xn = hidden.clone();
            layernorm_rows(&mut xn, D_MODEL, &layer.ln_post);
            let mut hf = linear(&xn, t, &layer.fc1, ENC_FF, D_MODEL, Some(&layer.fc1_b));
            for z in hf.iter_mut() { *z = gelu(*z); }
            let ff = linear(&hf, t, &layer.fc2, D_MODEL, ENC_FF, Some(&layer.fc2_b));
            for i in 0..hidden.len() { hidden[i] += ff[i]; }
        }
        layernorm_rows(&mut hidden, D_MODEL, &self.enc_ln);
        (hidden, t)
    }

    /// Full-sequence self-attention (used by the encoder). `rope`
    /// toggles rotary on q/k; `causal` restricts query i to keys ≤ i.
    fn attend_full(
        &self, xn: &[f32], t: usize,
        qw: &[f32], kw: &[f32], vw: &[f32], ow: &[f32], rope: bool, causal: bool,
    ) -> Vec<f32> {
        let mut q = linear(xn, t, qw, D_MODEL, D_MODEL, None);
        let mut k = linear(xn, t, kw, D_MODEL, D_MODEL, None);
        let vv = linear(xn, t, vw, D_MODEL, D_MODEL, None);
        if rope {
            for pos in 0..t {
                for h in 0..N_HEADS {
                    let off = pos * D_MODEL + h * D_HEAD;
                    rope_partial(&mut q[off..off + D_HEAD], pos);
                    rope_partial(&mut k[off..off + D_HEAD], pos);
                }
            }
        }
        let mut attn = vec![0.0f32; t * D_MODEL];
        let sc = scale();
        attn.par_chunks_mut(D_MODEL).enumerate().for_each(|(i, arow)| {
            let hi = if causal { i + 1 } else { t };
            for h in 0..N_HEADS {
                let qh = &q[i * D_MODEL + h * D_HEAD..i * D_MODEL + h * D_HEAD + D_HEAD];
                let oh = fused_attend_head(qh, &k, &vv, h, hi, D_MODEL, D_HEAD, sc);
                arow[h * D_HEAD..(h + 1) * D_HEAD].copy_from_slice(&oh);
            }
        });
        linear(&attn, t, ow, D_MODEL, D_MODEL, None)
    }

    /// Greedy transcription. Returns the decoded text.
    pub fn transcribe(&self, audio: &[f32], max_new: usize) -> String {
        let (enc_h, t_enc) = self.encode(audio);

        // Precompute per-layer cross-attn K/V from the (fixed) encoder output.
        let cross_kv: Vec<(Vec<f32>, Vec<f32>)> = self.dec.iter().map(|l| {
            let ck = linear(&enc_h, t_enc, &l.ck, D_MODEL, D_MODEL, None);
            let cv = linear(&enc_h, t_enc, &l.cv, D_MODEL, D_MODEL, None);
            (ck, cv)
        }).collect();

        // Per-layer self-attn KV caches (grow per generated token).
        let mut self_k: Vec<Vec<f32>> = vec![Vec::new(); N_DEC];
        let mut self_v: Vec<Vec<f32>> = vec![Vec::new(); N_DEC];

        let mut out_ids: Vec<i32> = Vec::new();
        let mut tok = BOS;
        let sc = scale();
        for pos in 0..max_new {
            // embed current token
            let mut x = self.embed[tok as usize * D_MODEL..(tok as usize + 1) * D_MODEL].to_vec();

            for (li, layer) in self.dec.iter().enumerate() {
                // causal rotary self-attn (single query at `pos`)
                let mut xn = x.clone();
                layernorm_rows(&mut xn, D_MODEL, &layer.ln_in);
                let mut q = linear(&xn, 1, &layer.q, D_MODEL, D_MODEL, None);
                let mut kk = linear(&xn, 1, &layer.k, D_MODEL, D_MODEL, None);
                let vvv = linear(&xn, 1, &layer.v, D_MODEL, D_MODEL, None);
                for h in 0..N_HEADS {
                    rope_partial(&mut q[h * D_HEAD..h * D_HEAD + D_HEAD], pos);
                    rope_partial(&mut kk[h * D_HEAD..h * D_HEAD + D_HEAD], pos);
                }
                self_k[li].extend_from_slice(&kk);
                self_v[li].extend_from_slice(&vvv);
                let seq = pos + 1;
                let mut attn = vec![0.0f32; D_MODEL];
                for h in 0..N_HEADS {
                    let qh = &q[h * D_HEAD..h * D_HEAD + D_HEAD];
                    let oh = fused_attend_head(qh, &self_k[li], &self_v[li], h, seq, D_MODEL, D_HEAD, sc);
                    attn[h * D_HEAD..(h + 1) * D_HEAD].copy_from_slice(&oh);
                }
                let attn = linear(&attn, 1, &layer.o, D_MODEL, D_MODEL, None);
                for i in 0..D_MODEL { x[i] += attn[i]; }

                // cross-attn (no rope; q from decoder, k/v from encoder)
                let mut xn = x.clone();
                layernorm_rows(&mut xn, D_MODEL, &layer.ln_post);
                let cq = linear(&xn, 1, &layer.cq, D_MODEL, D_MODEL, None);
                let (ck, cv) = &cross_kv[li];
                let mut cattn = vec![0.0f32; D_MODEL];
                for h in 0..N_HEADS {
                    let qh = &cq[h * D_HEAD..h * D_HEAD + D_HEAD];
                    let oh = fused_attend_head(qh, ck, cv, h, t_enc, D_MODEL, D_HEAD, sc);
                    cattn[h * D_HEAD..(h + 1) * D_HEAD].copy_from_slice(&oh);
                }
                let cattn = linear(&cattn, 1, &layer.co, D_MODEL, D_MODEL, None);
                for i in 0..D_MODEL { x[i] += cattn[i]; }

                // SwiGLU MLP: fc1 → (hidden ∥ gate) → silu(gate)·hidden → fc2
                let mut xn = x.clone();
                layernorm_rows(&mut xn, D_MODEL, &layer.ln_final);
                let gv = linear(&xn, 1, &layer.fc1, 2 * DEC_FF, D_MODEL, Some(&layer.fc1_b));
                let mut hmid = vec![0.0f32; DEC_FF];
                for i in 0..DEC_FF { hmid[i] = silu(gv[DEC_FF + i]) * gv[i]; }
                let ff = linear(&hmid, 1, &layer.fc2, D_MODEL, DEC_FF, Some(&layer.fc2_b));
                for i in 0..D_MODEL { x[i] += ff[i]; }
            }

            layernorm_rows(&mut x, D_MODEL, &self.dec_norm);
            let logits = linear(&x, 1, &self.embed, VOCAB, D_MODEL, None);
            let mut best = 0i32;
            let mut bestv = f32::NEG_INFINITY;
            for (i, &l) in logits.iter().enumerate() {
                if l > bestv { bestv = l; best = i as i32; }
            }
            if best == EOS { break; }
            out_ids.push(best);
            tok = best;
            let _ = pos;
        }
        decode(&out_ids, &self.id2piece)
    }
}

// ── tokenizer (SentencePiece-BPE detokenization) ───────────────────

fn load_vocab(path: &Path) -> Result<HashMap<i32, String>, String> {
    let txt = std::fs::read_to_string(path).map_err(|e| format!("read tokenizer.json: {e}"))?;
    let json: serde_json::Value = serde_json::from_str(&txt).map_err(|e| format!("parse tokenizer.json: {e}"))?;
    let mut map = HashMap::new();
    if let Some(obj) = json.get("model").and_then(|m| m.get("vocab")).and_then(|v| v.as_object()) {
        for (piece, id) in obj {
            if let Some(i) = id.as_i64() { map.insert(i as i32, piece.clone()); }
        }
    }
    // added_tokens (specials like <s>, </s>) — keep them so we can skip in decode.
    if let Some(arr) = json.get("added_tokens").and_then(|a| a.as_array()) {
        for a in arr {
            if let (Some(id), Some(content)) = (a.get("id").and_then(|x| x.as_i64()), a.get("content").and_then(|x| x.as_str())) {
                map.entry(id as i32).or_insert_with(|| content.to_string());
            }
        }
    }
    if map.is_empty() { return Err("tokenizer.json: empty vocab".into()); }
    Ok(map)
}

/// SentencePiece-BPE detokenize: `▁`→space, `<0xXX>`→raw byte, fuse,
/// strip one leading space.
fn decode(ids: &[i32], id2piece: &HashMap<i32, String>) -> String {
    let mut bytes: Vec<u8> = Vec::new();
    for &id in ids {
        let Some(p) = id2piece.get(&id) else { continue };
        // <0xXX> byte-fallback token
        if p.len() == 6 && p.as_bytes()[0] == b'<' && p.as_bytes()[1] == b'0' && p.as_bytes()[2] == b'x' && p.as_bytes()[5] == b'>' {
            if let Ok(b) = u8::from_str_radix(&p[3..5], 16) { bytes.push(b); continue; }
        }
        // skip reserved/special markers like <s>, </s>, <unk>, <<ST_n>>
        if p.starts_with('<') && p.ends_with('>') && (p.contains("ST_") || p == "<s>" || p == "</s>" || p == "<unk>" || p == "<pad>") {
            continue;
        }
        for ch in p.chars() {
            if ch == '\u{2581}' { bytes.push(b' '); }
            else { let mut buf = [0u8; 4]; bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes()); }
        }
    }
    let s = String::from_utf8_lossy(&bytes).into_owned();
    s.strip_prefix(' ').map(|x| x.to_string()).unwrap_or(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    const DIR: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/moonshine";
    const PCM: &str = "/tmp/moonshine_test.pcm"; // s16le 16k mono (ffmpeg pre-convert)

    fn read_pcm(path: &str) -> Option<Vec<f32>> {
        let bytes = std::fs::read(path).ok()?;
        Some(bytes.chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect())
    }

    #[test]
    fn moonshine_load_sanity() {
        if !std::path::Path::new(DIR).exists() { eprintln!("moonshine: model not mounted — skipping"); return; }
        let m = Moonshine::load(DIR).expect("load");
        assert_eq!(m.enc.len(), N_ENC);
        assert_eq!(m.dec.len(), N_DEC);
        assert_eq!(m.embed.len(), VOCAB * D_MODEL);
        assert!(m.id2piece.len() > 30000);
    }

    #[test]
    fn moonshine_transcribe() {
        if !std::path::Path::new(DIR).exists() { eprintln!("moonshine: model not mounted — skipping"); return; }
        let Some(audio) = read_pcm(PCM) else {
            eprintln!("moonshine: {PCM} not found — run `ffmpeg -i <clip> -ac 1 -ar 16000 -f s16le {PCM}`");
            return;
        };
        let m = Moonshine::load(DIR).expect("load");
        let t0 = std::time::Instant::now();
        let text = m.transcribe(&audio, 160);
        eprintln!("moonshine transcript ({:.1}s audio, {:?}): {text:?}",
            audio.len() as f32 / 16000.0, t0.elapsed());
        assert!(!text.trim().is_empty(), "empty transcript");
        let alpha = text.chars().filter(|c| c.is_alphabetic()).count();
        assert!(alpha >= 5, "transcript looks empty/garbage: {text:?}");
    }
}
