//! Dia-1.6B text-to-speech on the rce substrate — rce's first TTS.
//!
//! Byte text → 12-layer encoder (RoPE, SwiGLU, scaling=1) → 18-layer
//! decoder (GQA self-attn + cross-attn, 9 DAC codebook channels, delay
//! pattern) → DAC neural codec decoder (RVQ → weight-normed transposed-conv
//! stack with Snake) → 44 kHz waveform. Self-validates via round-trip ASR.
//! Spec: HF `modeling_dia.py`/`processing_dia.py` + DAC `DacModel`.

use rce_weight_loader::{LoadedTensor, SafetensorsFile};
use rayon::prelude::*;
use std::path::Path;

const ENC_D: usize = 1024;
const ENC_L: usize = 12;
const ENC_FF: usize = 4096;
const DEC_D: usize = 2048;
const DEC_L: usize = 18;
const DEC_FF: usize = 8192;
const HD: usize = 128;
const ENC_H: usize = 16;
const DEC_H: usize = 16;
const DEC_KV: usize = 4; // GQA
const N_CH: usize = 9;
const AUDIO_VOCAB: usize = 1028;
const EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 10000.0;
const BOS: i32 = 1026;
const EOS: i32 = 1024;
const PAD: i32 = 1025;
const DELAY: [usize; N_CH] = [0, 8, 9, 10, 11, 12, 13, 14, 15];

#[inline] fn silu(x: f32) -> f32 { if x >= 0.0 { x / (1.0 + (-x).exp()) } else { let e = x.exp(); x * e / (1.0 + e) } }

fn tf32(t: &LoadedTensor) -> Vec<f32> {
    t.as_f32().unwrap_or_else(|| panic!("tensor {} not f32 (dtype {})", t.name, t.dtype)).to_vec()
}

fn linear(x: &[f32], t: usize, w: &[f32], dout: usize, din: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; t * dout];
    out.par_chunks_mut(dout).enumerate().for_each(|(i, orow)| {
        let xr = &x[i * din..(i + 1) * din];
        for o in 0..dout {
            let wr = &w[o * din..(o + 1) * din];
            let mut acc = 0.0f32;
            for k in 0..din { acc += xr[k] * wr[k]; }
            orow[o] = acc;
        }
    });
    out
}

fn rmsnorm(x: &[f32], t: usize, d: usize, w: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; t * d];
    out.par_chunks_mut(d).zip(x.par_chunks(d)).for_each(|(orow, xr)| {
        let ms = xr.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ms + EPS).sqrt();
        for k in 0..d { orow[k] = xr[k] * inv * w[k]; }
    });
    out
}

/// NeoX half-split RoPE on one head's `[HD]` slice at position `pos`.
fn rope(head: &mut [f32], pos: usize) {
    let half = HD / 2;
    for i in 0..half {
        let inv = ROPE_THETA.powf(-2.0 * i as f32 / HD as f32);
        let (s, c) = (pos as f32 * inv).sin_cos();
        let a = head[i];
        let b = head[i + half];
        head[i] = a * c - b * s;
        head[i + half] = b * c + a * s;
    }
}

/// SwiGLU: `down( silu(gate) * up )` from a fused gate_up projection.
fn swiglu(x: &[f32], t: usize, gate_up: &[f32], down: &[f32], d: usize, ff: usize) -> Vec<f32> {
    let gu = linear(x, t, gate_up, 2 * ff, d);
    let mut h = vec![0.0f32; t * ff];
    h.par_chunks_mut(ff).enumerate().for_each(|(i, hr)| {
        let row = &gu[i * 2 * ff..(i + 1) * 2 * ff];
        for k in 0..ff { hr[k] = silu(row[k]) * row[ff + k]; }
    });
    linear(&h, t, down, d, ff)
}

/// Multi-head attention, scaling=1. q `[tq, nh*HD]`, k/v `[tk, nkv*HD]`.
/// `causal`: query i attends keys ≤ i (only meaningful when tq==tk).
fn attention(q: &[f32], k: &[f32], v: &[f32], tq: usize, tk: usize, nh: usize, nkv: usize, causal: bool) -> Vec<f32> {
    let groups = nh / nkv;
    let mut out = vec![0.0f32; tq * nh * HD];
    out.par_chunks_mut(nh * HD).enumerate().for_each(|(i, orow)| {
        for h in 0..nh {
            let kvh = h / groups;
            let qh = &q[i * nh * HD + h * HD..i * nh * HD + h * HD + HD];
            let lim = if causal { i + 1 } else { tk };
            let mut scores = vec![0.0f32; lim];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..lim {
                let kj = &k[j * nkv * HD + kvh * HD..j * nkv * HD + kvh * HD + HD];
                let mut s = 0.0f32; for d in 0..HD { s += qh[d] * kj[d]; }
                scores[j] = s; if s > mx { mx = s; }
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
            let inv = 1.0 / sum;
            let oh = &mut orow[h * HD..(h + 1) * HD];
            for j in 0..lim {
                let wj = scores[j] * inv;
                let vj = &v[j * nkv * HD + kvh * HD..j * nkv * HD + kvh * HD + HD];
                for d in 0..HD { oh[d] += wj * vj[d]; }
            }
        }
    });
    out
}

// ── weights ────────────────────────────────────────────────────────

struct EncLayer { pre: Vec<f32>, post: Vec<f32>, q: Vec<f32>, k: Vec<f32>, v: Vec<f32>, o: Vec<f32>, gate_up: Vec<f32>, down: Vec<f32> }
struct DecLayer {
    pre_sa: Vec<f32>, pre_ca: Vec<f32>, pre_mlp: Vec<f32>,
    sq: Vec<f32>, sk: Vec<f32>, sv: Vec<f32>, so: Vec<f32>,
    cq: Vec<f32>, ck: Vec<f32>, cv: Vec<f32>, co: Vec<f32>,
    gate_up: Vec<f32>, down: Vec<f32>,
}

/// A resolved (weight-norm applied) conv. `w` row-major `[out, in, k]`
/// (regular) or `[in, out, k]` (transposed — caller tracks).
struct Conv { w: Vec<f32>, b: Vec<f32>, d0: usize, d1: usize, k: usize }

pub struct Dia {
    enc_embed: Vec<f32>, // [256, ENC_D]
    enc: Vec<EncLayer>, enc_norm: Vec<f32>,
    dec_embed: Vec<f32>, // [9*1028, DEC_D]
    dec: Vec<DecLayer>, dec_norm: Vec<f32>,
    logits_w: Vec<f32>, // [9252, DEC_D]
    // DAC
    codebooks: Vec<Vec<f32>>, // 9 × [1024,8]
    dac_out_proj_w: Vec<f32>, dac_out_proj_b: Vec<f32>, // [1024,72]
    dac: DacDecoder,
}

struct Snake { alpha: Vec<f32> }
struct ResUnit { s1: Snake, c1: Conv, s2: Snake, c2: Conv, dilation: usize }
struct DecBlock { snake: Snake, tconv: Conv, tconv_stride: usize, tconv_pad: usize, res: Vec<ResUnit> }
struct DacDecoder { c0: Conv, blocks: Vec<DecBlock>, snake: Snake, c6: Conv }

fn merge_get<'a>(files: &'a [SafetensorsFile], name: &str) -> &'a LoadedTensor {
    for f in files { if let Some(t) = f.get(name) { return t; } }
    panic!("missing tensor {name}");
}

/// Resolve weight_norm: `w[d0] = g[d0] * v[d0] / ||v[d0]||` (norm over the
/// remaining dims). g shape `[d0,1,1]`, v shape `[d0, d1, k]`.
fn resolve_wn(v: &[f32], g: &[f32], d0: usize, d1: usize, k: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; d0 * d1 * k];
    for o in 0..d0 {
        let base = o * d1 * k;
        let mut nrm = 0.0f32;
        for j in 0..d1 * k { nrm += v[base + j] * v[base + j]; }
        let scale = g[o] / (nrm.sqrt() + 1e-12);
        for j in 0..d1 * k { w[base + j] = v[base + j] * scale; }
    }
    w
}

fn load_conv(files: &[SafetensorsFile], p: &str, d0: usize, d1: usize, k: usize) -> Conv {
    let v = tf32(merge_get(files, &format!("{p}.weight_v")));
    let g = tf32(merge_get(files, &format!("{p}.weight_g")));
    let b = tf32(merge_get(files, &format!("{p}.bias")));
    Conv { w: resolve_wn(&v, &g, d0, d1, k), b, d0, d1, k }
}
fn load_snake(files: &[SafetensorsFile], p: &str) -> Snake { Snake { alpha: tf32(merge_get(files, &format!("{p}.alpha"))) } }

impl Dia {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = dir.as_ref();
        let mut files = Vec::new();
        for i in 1..=2 {
            let p = dir.join(format!("model-{i:05}-of-00002.safetensors"));
            files.push(SafetensorsFile::load(&p).map_err(|e| format!("load {p:?}: {e:?}"))?);
        }
        let g = |n: &str| tf32(merge_get(&files, n));

        let enc = (0..ENC_L).map(|i| {
            let p = format!("model.encoder.layers.{i}");
            EncLayer {
                pre: g(&format!("{p}.pre_sa_norm.weight")), post: g(&format!("{p}.post_sa_norm.weight")),
                q: g(&format!("{p}.self_attention.q_proj.weight")), k: g(&format!("{p}.self_attention.k_proj.weight")),
                v: g(&format!("{p}.self_attention.v_proj.weight")), o: g(&format!("{p}.self_attention.o_proj.weight")),
                gate_up: g(&format!("{p}.mlp.gate_up_proj.weight")), down: g(&format!("{p}.mlp.down_proj.weight")),
            }
        }).collect();
        let dec = (0..DEC_L).map(|i| {
            let p = format!("model.decoder.layers.{i}");
            DecLayer {
                pre_sa: g(&format!("{p}.pre_sa_norm.weight")), pre_ca: g(&format!("{p}.pre_ca_norm.weight")), pre_mlp: g(&format!("{p}.pre_mlp_norm.weight")),
                sq: g(&format!("{p}.self_attention.q_proj.weight")), sk: g(&format!("{p}.self_attention.k_proj.weight")),
                sv: g(&format!("{p}.self_attention.v_proj.weight")), so: g(&format!("{p}.self_attention.o_proj.weight")),
                cq: g(&format!("{p}.cross_attention.q_proj.weight")), ck: g(&format!("{p}.cross_attention.k_proj.weight")),
                cv: g(&format!("{p}.cross_attention.v_proj.weight")), co: g(&format!("{p}.cross_attention.o_proj.weight")),
                gate_up: g(&format!("{p}.mlp.gate_up_proj.weight")), down: g(&format!("{p}.mlp.down_proj.weight")),
            }
        }).collect();

        // DAC
        let dfiles = vec![SafetensorsFile::load(dir.join("dac/model.safetensors")).map_err(|e| format!("dac: {e:?}"))?];
        let codebooks: Vec<Vec<f32>> = (0..N_CH).map(|i| tf32(merge_get(&dfiles, &format!("quantizer.quantizers.{i}.codebook.weight")))).collect();
        let dac_out_proj_w = tf32(merge_get(&dfiles, "quantizer.out_proj.weight"));
        let dac_out_proj_b = tf32(merge_get(&dfiles, "quantizer.out_proj.bias"));
        // decoder.model: 0 conv(1024->1536,k7); 1..4 blocks (rates 8,8,4,2); 5 snake; 6 conv(96->1,k7)
        let dims = [1536usize, 768, 384, 192, 96];
        let rates = [8usize, 8, 4, 2];
        let c0 = load_conv(&dfiles, "decoder.model.0", 1536, 1024, 7);
        let mut blocks = Vec::new();
        for bi in 0..4 {
            let p = format!("decoder.model.{}", bi + 1);
            let (cin, cout, rate) = (dims[bi], dims[bi + 1], rates[bi]);
            let snake = load_snake(&dfiles, &format!("{p}.block.0"));
            let tconv = load_conv(&dfiles, &format!("{p}.block.1"), cin, cout, 2 * rate);
            let pad = rate.div_ceil(2);
            let res = (0..3).map(|ri| {
                let rp = format!("{p}.block.{}", ri + 2);
                let dil = [1usize, 3, 9][ri];
                ResUnit {
                    s1: load_snake(&dfiles, &format!("{rp}.block.0")),
                    c1: load_conv(&dfiles, &format!("{rp}.block.1"), cout, cout, 7),
                    s2: load_snake(&dfiles, &format!("{rp}.block.2")),
                    c2: load_conv(&dfiles, &format!("{rp}.block.3"), cout, cout, 1),
                    dilation: dil,
                }
            }).collect();
            blocks.push(DecBlock { snake, tconv, tconv_stride: rate, tconv_pad: pad, res });
        }
        let snake = load_snake(&dfiles, "decoder.model.5");
        let c6 = load_conv(&dfiles, "decoder.model.6", 1, 96, 7);

        Ok(Self {
            enc_embed: g("model.encoder.embedding.weight"), enc, enc_norm: g("model.encoder.norm.weight"),
            dec_embed: g("model.decoder.embeddings.embed.weight"), dec, dec_norm: g("model.decoder.norm.weight"),
            logits_w: g("logits_dense.weight"),
            codebooks, dac_out_proj_w, dac_out_proj_b,
            dac: DacDecoder { c0, blocks, snake, c6 },
        })
    }

    fn encode_text(&self, text: &str) -> (Vec<f32>, usize) {
        let bytes: Vec<u8> = format!("[S1] {text}").into_bytes();
        let t = bytes.len();
        let mut x = vec![0.0f32; t * ENC_D];
        for (i, &b) in bytes.iter().enumerate() {
            x[i * ENC_D..(i + 1) * ENC_D].copy_from_slice(&self.enc_embed[b as usize * ENC_D..(b as usize + 1) * ENC_D]);
        }
        for layer in &self.enc {
            let xn = rmsnorm(&x, t, ENC_D, &layer.pre);
            let mut q = linear(&xn, t, &layer.q, ENC_H * HD, ENC_D);
            let mut k = linear(&xn, t, &layer.k, ENC_H * HD, ENC_D);
            let v = linear(&xn, t, &layer.v, ENC_H * HD, ENC_D);
            for p in 0..t { for h in 0..ENC_H { rope(&mut q[p * ENC_H * HD + h * HD..p * ENC_H * HD + h * HD + HD], p); rope(&mut k[p * ENC_H * HD + h * HD..p * ENC_H * HD + h * HD + HD], p); } }
            let a = attention(&q, &k, &v, t, t, ENC_H, ENC_H, false);
            let ao = linear(&a, t, &layer.o, ENC_D, ENC_H * HD);
            for i in 0..x.len() { x[i] += ao[i]; }
            let xn = rmsnorm(&x, t, ENC_D, &layer.post);
            let mo = swiglu(&xn, t, &layer.gate_up, &layer.down, ENC_D, ENC_FF);
            for i in 0..x.len() { x[i] += mo[i]; }
        }
        (rmsnorm(&x, t, ENC_D, &self.enc_norm), t)
    }

    /// Generate DAC codes `[N_CH, L]` for the text via delayed AR decoding.
    fn generate_codes(&self, enc_h: &[f32], t_enc: usize, max_frames: usize) -> Vec<Vec<i32>> {
        let max_delay = *DELAY.iter().max().unwrap();
        // cross-attn K/V per layer (computed once)
        let cross: Vec<(Vec<f32>, Vec<f32>)> = self.dec.iter().map(|l| {
            (linear(enc_h, t_enc, &l.ck, DEC_H * HD, ENC_D), linear(enc_h, t_enc, &l.cv, DEC_H * HD, ENC_D))
        }).collect();
        let mut sk: Vec<Vec<f32>> = vec![Vec::new(); DEC_L];
        let mut sv: Vec<Vec<f32>> = vec![Vec::new(); DEC_L];

        // D[s][c] delayed grid (the tokens fed/produced). Start row = bos.
        let mut grid: Vec<[i32; N_CH]> = vec![[BOS; N_CH]];
        let mut audio_len: Option<usize> = None;
        let total_cap = max_frames + max_delay;

        for s in 0..total_cap {
            // embed grid[s] (sum of 9 channel embeddings) and run decoder at position s.
            let cur = &grid[s];
            let mut x = vec![0.0f32; DEC_D];
            for c in 0..N_CH {
                let row = (c * AUDIO_VOCAB + cur[c] as usize) * DEC_D;
                for d in 0..DEC_D { x[d] += self.dec_embed[row + d]; }
            }
            for (li, layer) in self.dec.iter().enumerate() {
                let xn = rmsnorm(&x, 1, DEC_D, &layer.pre_sa);
                let mut q = linear(&xn, 1, &layer.sq, DEC_H * HD, DEC_D);
                let mut k = linear(&xn, 1, &layer.sk, DEC_KV * HD, DEC_D);
                let v = linear(&xn, 1, &layer.sv, DEC_KV * HD, DEC_D);
                for h in 0..DEC_H { rope(&mut q[h * HD..h * HD + HD], s); }
                for h in 0..DEC_KV { rope(&mut k[h * HD..h * HD + HD], s); }
                sk[li].extend_from_slice(&k); sv[li].extend_from_slice(&v);
                let seq = s + 1;
                let a = attention(&q, &sk[li], &sv[li], 1, seq, DEC_H, DEC_KV, false);
                let ao = linear(&a, 1, &layer.so, DEC_D, DEC_H * HD);
                for d in 0..DEC_D { x[d] += ao[d]; }
                // cross-attn
                let xn = rmsnorm(&x, 1, DEC_D, &layer.pre_ca);
                let cq = linear(&xn, 1, &layer.cq, DEC_H * HD, DEC_D);
                let (ck, cv) = &cross[li];
                let ca = attention(&cq, ck, cv, 1, t_enc, DEC_H, DEC_H, false);
                let cao = linear(&ca, 1, &layer.co, DEC_D, DEC_H * HD);
                for d in 0..DEC_D { x[d] += cao[d]; }
                // mlp
                let xn = rmsnorm(&x, 1, DEC_D, &layer.pre_mlp);
                let mo = swiglu(&xn, 1, &layer.gate_up, &layer.down, DEC_D, DEC_FF);
                for d in 0..DEC_D { x[d] += mo[d]; }
            }
            let xf = rmsnorm(&x, 1, DEC_D, &self.dec_norm);
            let logits = linear(&xf, 1, &self.logits_w, N_CH * AUDIO_VOCAB, DEC_D);
            // next row
            let mut next = [PAD; N_CH];
            for c in 0..N_CH {
                if s < DELAY[c] { next[c] = BOS; continue; }
                if let Some(l) = audio_len { if s - DELAY[c] >= l { next[c] = PAD; continue; } }
                // argmax over this channel's 1028 logits
                let base = c * AUDIO_VOCAB;
                let mut best = 0i32; let mut bv = f32::NEG_INFINITY;
                for vix in 0..AUDIO_VOCAB { let lv = logits[base + vix]; if lv > bv { bv = lv; best = vix as i32; } }
                next[c] = best;
                if c == 0 && best == EOS && audio_len.is_none() { audio_len = Some(s); }
            }
            grid.push(next);
            if let Some(l) = audio_len { if s >= l + max_delay { break; } }
        }

        let l = audio_len.unwrap_or(grid.len().saturating_sub(max_delay + 1)).max(1);
        // revert delay: codes[c][f] = grid[f + delay[c] + 1][c]  (+1: grid[0] is bos row)
        let mut codes = vec![vec![0i32; l]; N_CH];
        for c in 0..N_CH {
            for f in 0..l {
                let p = f + DELAY[c] + 1;
                let tok = if p < grid.len() { grid[p][c] } else { PAD };
                codes[c][f] = if (0..1024).contains(&tok) { tok } else { 0 };
            }
        }
        codes
    }

    /// DAC decode: codes `[N_CH, L]` → 44 kHz mono waveform.
    fn dac_decode(&self, codes: &[Vec<i32>]) -> Vec<f32> {
        let l = codes[0].len();
        // RVQ: lookup each codebook (8-dim), concat 9 → 72, out_proj → 1024.
        let mut concat = vec![0.0f32; 72 * l]; // [72, L] channel-major
        for c in 0..N_CH {
            let cb = &self.codebooks[c]; // [1024,8]
            for f in 0..l {
                let code = codes[c][f].clamp(0, 1023) as usize;
                for d in 0..8 { concat[(c * 8 + d) * l + f] = cb[code * 8 + d]; }
            }
        }
        // out_proj conv1d(72->1024, k1): z[o,f] = b[o] + sum_i w[o,i]*concat[i,f]
        let mut z = vec![0.0f32; 1024 * l];
        z.par_chunks_mut(l).enumerate().for_each(|(o, zrow)| {
            let wr = &self.dac_out_proj_w[o * 72..(o + 1) * 72];
            for f in 0..l {
                let mut acc = self.dac_out_proj_b[o];
                for i in 0..72 { acc += wr[i] * concat[i * l + f]; }
                zrow[f] = acc;
            }
        });
        // decoder.model.0 conv
        let (mut h, mut t) = conv1d(&z, 1024, l, &self.dac.c0, 1, 3, 1);
        for blk in &self.dac.blocks {
            snake_inplace(&mut h, blk.snake.alpha.len(), t, &blk.snake);
            let (ht, tt) = conv_transpose1d(&h, blk.tconv.d0, t, &blk.tconv, blk.tconv_stride, blk.tconv_pad);
            h = ht; t = tt;
            let c = blk.tconv.d1; // out channels
            for ru in &blk.res {
                let mut y = h.clone();
                snake_inplace(&mut y, c, t, &ru.s1);
                let (y1, _) = conv1d(&y, c, t, &ru.c1, ru.dilation * 3, 1, ru.dilation);
                let mut y1 = y1;
                snake_inplace(&mut y1, c, t, &ru.s2);
                let (y2, _) = conv1d(&y1, c, t, &ru.c2, 0, 1, 1);
                for i in 0..h.len() { h[i] += y2[i]; }
            }
        }
        snake_inplace(&mut h, 96, t, &self.dac.snake);
        let (wav, _) = conv1d(&h, 96, t, &self.dac.c6, 3, 1, 1);
        wav
    }

    /// text → 44 kHz mono waveform.
    pub fn synthesize(&self, text: &str, max_frames: usize) -> Vec<f32> {
        let (enc_h, t_enc) = self.encode_text(text);
        let codes = self.generate_codes(&enc_h, t_enc, max_frames);
        self.dac_decode(&codes)
    }
}

/// Snake activation in place: `x += sin(α x)² / (α + 1e-9)`, per channel.
/// `x` is `[C, T]` channel-major.
fn snake_inplace(x: &mut [f32], c: usize, t: usize, s: &Snake) {
    x.par_chunks_mut(t).enumerate().for_each(|(ch, row)| {
        let a = s.alpha[ch];
        let inv = 1.0 / (a + 1e-9);
        for v in row.iter_mut() { let sn = (a * *v).sin(); *v += sn * sn * inv; }
    });
    let _ = c;
}

/// Regular conv1d over `[Cin, T]` → `[Cout, Tout]`. weight `[Cout, Cin, k]`.
fn conv1d(inp: &[f32], cin: usize, t: usize, conv: &Conv, pad: usize, stride: usize, dilation: usize) -> (Vec<f32>, usize) {
    let (cout, k) = (conv.d0, conv.k);
    debug_assert_eq!(conv.d1, cin);
    let tout = (t + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
    let mut out = vec![0.0f32; cout * tout];
    out.par_chunks_mut(tout).enumerate().for_each(|(o, orow)| {
        let wbase = o * cin * k;
        for to in 0..tout {
            let mut acc = conv.b[o];
            for i in 0..cin {
                let wb = wbase + i * k;
                let ib = i * t;
                for kk in 0..k {
                    let ti = (to * stride + kk * dilation) as isize - pad as isize;
                    if ti >= 0 && (ti as usize) < t { acc += conv.w[wb + kk] * inp[ib + ti as usize]; }
                }
            }
            orow[to] = acc;
        }
    });
    (out, tout)
}

/// Transposed conv1d over `[Cin, T]` → `[Cout, Tout]`. weight `[Cin, Cout, k]`.
fn conv_transpose1d(inp: &[f32], cin: usize, t: usize, conv: &Conv, stride: usize, pad: usize) -> (Vec<f32>, usize) {
    let (cout, k) = (conv.d1, conv.k);
    debug_assert_eq!(conv.d0, cin);
    let tout = (t - 1) * stride + k - 2 * pad;
    let mut out = vec![0.0f32; cout * tout];
    // accumulate per output channel (parallel over cout)
    out.par_chunks_mut(tout).enumerate().for_each(|(o, orow)| {
        for ti in 0..t {
            for i in 0..cin {
                let wv = conv.w[i * cout * k + o * k..i * cout * k + o * k + k].to_vec();
                let iv = inp[i * t + ti];
                for kk in 0..k {
                    let p = ti as isize * stride as isize + kk as isize - pad as isize;
                    if p >= 0 && (p as usize) < tout { orow[p as usize] += iv * wv[kk]; }
                }
            }
        }
        for v in orow.iter_mut() { *v += conv.b[o]; }
    });
    (out, tout)
}

/// Write a mono 16-bit PCM WAV at the given sample rate.
pub fn write_wav(path: &str, samples: &[f32], sr: u32) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let n = samples.len() as u32;
    let data_bytes = n * 2;
    f.write_all(b"RIFF")?; f.write_all(&(36 + data_bytes).to_le_bytes())?; f.write_all(b"WAVE")?;
    f.write_all(b"fmt ")?; f.write_all(&16u32.to_le_bytes())?; f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?; f.write_all(&sr.to_le_bytes())?; f.write_all(&(sr * 2).to_le_bytes())?;
    f.write_all(&2u16.to_le_bytes())?; f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?; f.write_all(&data_bytes.to_le_bytes())?;
    for &s in samples { let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16; f.write_all(&v.to_le_bytes())?; }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn model_dir() -> Option<String> {
        for b in ["macos_4TB_external", "macos_4TB_external_1", "macos_4TB_external_2"] {
            let d = format!("/Volumes/{b}/neuronexus_ai/model_library/dia-1.6b");
            if std::path::Path::new(&d).join("dac/model.safetensors").exists() { return Some(d); }
        }
        None
    }
    #[test]
    fn dia_load_sanity() {
        let Some(d) = model_dir() else { eprintln!("dia: not mounted — skipping"); return };
        let m = Dia::load(&d).expect("load");
        assert_eq!(m.enc.len(), ENC_L);
        assert_eq!(m.dec.len(), DEC_L);
        assert_eq!(m.codebooks.len(), N_CH);
        assert_eq!(m.dac.blocks.len(), 4);
    }
    #[test]
    #[ignore]
    fn dia_synthesize() {
        let Some(d) = model_dir() else { eprintln!("dia: not mounted — skipping"); return };
        let m = Dia::load(&d).expect("load");
        let t0 = std::time::Instant::now();
        let wav = m.synthesize("The capital of France is Paris.", 400);
        eprintln!("dia synth: {} samples ({:.1}s @44.1k) in {:?}", wav.len(), wav.len() as f32 / 44100.0, t0.elapsed());
        write_wav("/tmp/dia_out.wav", &wav, 44100).expect("wav");
        eprintln!("wrote /tmp/dia_out.wav");
        let peak = wav.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(wav.len() > 4000 && peak > 0.001, "empty/silent output (peak {peak})");
    }
}
