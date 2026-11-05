//! The toy gpt2 model of the lift tests, as GGUF bytes: two blocks of
//! width 32, weights from a fixed seed, its products in Q8_0 and Q4_0
//! (shared by `lift_gguf.rs` and `emit_wasm.rs`).
#![allow(dead_code)]
use rce_ingest_gguf::parser::{build_gguf, encode_f32_value, encode_string_value, encode_u32_value};
use rce_vm_backend_cpu::ml::nn;

pub const D: usize = 32;
pub const HEADS: usize = 2;
pub const HD: usize = D / HEADS;
pub const FF: usize = 64;
pub const VOCAB: usize = 32;
pub const CTX: usize = 8;
pub const LAYERS: usize = 2;
pub const EPS: f32 = 1e-5;

/// Uniform values in [-a, a) from a fixed linear congruential sequence.
pub struct Lcg(pub u64);
impl Lcg {
    pub fn next(&mut self, a: f32) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * a
    }
    pub fn vec(&mut self, n: usize, a: f32) -> Vec<f32> {
        (0..n).map(|_| self.next(a)).collect()
    }
}

/// The binary16 bits of an f32 that is a binary16 value.
pub fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let a = x.abs();
    if a == 0.0 {
        return s;
    }
    if a < 2f32.powi(-14) {
        return s | (a / 2f32.powi(-24)) as u16;
    }
    let e = ((a.to_bits() >> 23) as i32) - 127 + 15;
    s | ((e as u16) << 10) | ((a.to_bits() >> 13) & 0x3ff) as u16
}

/// GGUF bytes of an n x k matrix (rows of k) in Q8_0 or Q4_0, quantized by
/// the reference quantizer of the block formats.
pub fn quantize(w: &[f32], q4: bool) -> Vec<u8> {
    let mut out = vec![];
    for blk in w.chunks(32) {
        let (q, d) = nn::quantize_block(blk, q4);
        out.extend_from_slice(&f16_bits(d).to_le_bytes());
        if q4 {
            out.extend((0..16).map(|t| ((q[t] + 8) as u8) | (((q[t + 16] + 8) as u8) << 4)));
        } else {
            out.extend(q.iter().map(|v| *v as i8 as u8));
        }
    }
    out
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The toy model's GGUF file.
pub fn toy_gguf() -> Vec<u8> {
    let mut r = Lcg(0x5eed_0001);
    let mut tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = vec![];
    let mut f32t = |name: &str, dims: Vec<u64>, v: Vec<f32>| tensors.push((name.into(), dims, 0, f32_bytes(&v)));
    f32t("token_embd.weight", vec![D as u64, VOCAB as u64], r.vec(D * VOCAB, 1.0));
    // the position table in F16: its values are binary16 values
    let pos: Vec<f32> = r.vec(D * CTX, 0.5).iter().map(|x| nn::round_to_f16(*x)).collect();
    let pos16: Vec<u8> = pos.iter().flat_map(|x| f16_bits(*x).to_le_bytes()).collect();
    tensors.push(("position_embd.weight".into(), vec![D as u64, CTX as u64], 1, pos16));
    for i in 0..LAYERS {
        let q4 = i == 1;
        let qt = if q4 { 2 } else { 8 };
        let mut quant = |name: String, k: usize, n: usize, q4: bool, r: &mut Lcg| {
            let w = r.vec(n * k, 0.3);
            tensors.push((name, vec![k as u64, n as u64], if q4 { 2 } else { 8 }, quantize(&w, q4)));
        };
        quant(format!("blk.{i}.attn_qkv.weight"), D, 3 * D, q4, &mut r);
        quant(format!("blk.{i}.attn_output.weight"), D, D, false, &mut r);
        quant(format!("blk.{i}.ffn_up.weight"), D, FF, q4, &mut r);
        quant(format!("blk.{i}.ffn_down.weight"), FF, D, q4, &mut r);
        let _ = qt;
        for (name, n, a, off) in [("attn_norm.weight", D, 0.2, 1.0), ("attn_norm.bias", D, 0.1, 0.0), ("attn_qkv.bias", 3 * D, 0.1, 0.0), ("attn_output.bias", D, 0.1, 0.0), ("ffn_norm.weight", D, 0.2, 1.0), ("ffn_norm.bias", D, 0.1, 0.0), ("ffn_up.bias", FF, 0.1, 0.0), ("ffn_down.bias", D, 0.1, 0.0)] {
            let v: Vec<f32> = r.vec(n, a).iter().map(|x| x + off).collect();
            tensors.push((format!("blk.{i}.{name}"), vec![n as u64], 0, f32_bytes(&v)));
        }
    }
    let v: Vec<f32> = r.vec(D, 0.2).iter().map(|x| x + 1.0).collect();
    tensors.push(("output_norm.weight".into(), vec![D as u64], 0, f32_bytes(&v)));
    tensors.push(("output_norm.bias".into(), vec![D as u64], 0, f32_bytes(&r.vec(D, 0.1))));
    let w = r.vec(VOCAB * D, 0.3);
    tensors.push(("output.weight".into(), vec![D as u64, VOCAB as u64], 8, quantize(&w, false)));
    let meta: Vec<(&str, u32, Vec<u8>)> = vec![
        ("general.architecture", 8, encode_string_value("gpt2")),
        ("gpt2.context_length", 4, encode_u32_value(CTX as u32)),
        ("gpt2.embedding_length", 4, encode_u32_value(D as u32)),
        ("gpt2.feed_forward_length", 4, encode_u32_value(FF as u32)),
        ("gpt2.block_count", 4, encode_u32_value(LAYERS as u32)),
        ("gpt2.attention.head_count", 4, encode_u32_value(HEADS as u32)),
        ("gpt2.attention.layer_norm_epsilon", 6, encode_f32_value(EPS)),
    ];
    let m: Vec<(&str, u32, &[u8])> = meta.iter().map(|(k, t, v)| (*k, *t, v.as_slice())).collect();
    let t: Vec<(&str, &[u64], u32, &[u8])> = tensors.iter().map(|(n, d, ty, b)| (n.as_str(), d.as_slice(), *ty, b.as_slice())).collect();
    build_gguf(&m, &t)
}

