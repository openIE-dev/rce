//! The toy llama model of the lift tests, as GGUF bytes: two blocks of
//! width 256 with 4 query heads and 2 key and value heads, weights from a
//! fixed seed, its products in Q4_K, Q5_K, Q6_K, Q8_0 and F16.
#![allow(dead_code)]
use rce_ingest_gguf::parser::{build_gguf, encode_f32_value, encode_string_value, encode_u32_value};
use rce_vm_backend_cpu::ml::kquant::{self, KFmt};
use rce_vm_backend_cpu::ml::nn;

#[path = "toy_gpt2.rs"]
mod gpt2;
pub use gpt2::{Lcg, f16_bits, f32_bytes, quantize};

pub const D: usize = 256;
pub const HEADS: usize = 4;
pub const KV_HEADS: usize = 2;
pub const HD: usize = D / HEADS;
pub const FF: usize = 512;
pub const VOCAB: usize = 32;
pub const CTX: usize = 8;
pub const LAYERS: usize = 2;
pub const EPS: f32 = 1e-5;
pub const BASE: f32 = 10000.0;

/// GGUF bytes of a matrix (rows of a multiple of 256) in a K-quant
/// format, quantized by RCE's reference quantizer.
pub fn quantize_k(w: &[f32], f: KFmt) -> Vec<u8> {
    w.chunks(256).flat_map(|b| kquant::pack(&kquant::quantize_k(b, f), f)).collect()
}

/// The GGUF type code and bytes of a weight of the given format.
fn encode(w: &[f32], fmt: &str) -> (u32, Vec<u8>) {
    match fmt {
        "q4_k" => (12, quantize_k(w, KFmt::Q4K)),
        "q5_k" => (13, quantize_k(w, KFmt::Q5K)),
        "q6_k" => (14, quantize_k(w, KFmt::Q6K)),
        "q8_0" => (8, quantize(w, false)),
        _ => (1, w.iter().flat_map(|x| f16_bits(nn::round_to_f16(*x)).to_le_bytes()).collect()),
    }
}

/// The format of each matrix: block 0 in Q4_K and Q5_K, block 1 in Q6_K,
/// Q8_0 and F16, the output projection in Q6_K.
pub fn format(layer: usize, name: &str) -> &'static str {
    match (layer, name) {
        (0, "attn_q" | "attn_k" | "ffn_gate" | "ffn_up") => "q4_k",
        (0, _) => "q5_k",
        (_, "attn_q" | "ffn_down") => "q6_k",
        (_, "attn_k" | "attn_v" | "ffn_gate") => "q8_0",
        _ => "f16",
    }
}

/// The toy model's GGUF file.
pub fn toy_gguf() -> Vec<u8> {
    let mut r = Lcg(0x5eed_11a3);
    let mut tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = vec![];
    tensors.push(("token_embd.weight".into(), vec![D as u64, VOCAB as u64], 0, f32_bytes(&r.vec(D * VOCAB, 1.0))));
    for i in 0..LAYERS {
        for (name, k, n) in [("attn_q", D, D), ("attn_k", D, KV_HEADS * HD), ("attn_v", D, KV_HEADS * HD), ("attn_output", D, D), ("ffn_gate", D, FF), ("ffn_up", D, FF), ("ffn_down", FF, D)] {
            let w = r.vec(n * k, 0.3);
            let (ty, b) = encode(&w, format(i, name));
            tensors.push((format!("blk.{i}.{name}.weight"), vec![k as u64, n as u64], ty, b));
        }
        for name in ["attn_norm.weight", "ffn_norm.weight"] {
            let v: Vec<f32> = r.vec(D, 0.2).iter().map(|x| x + 1.0).collect();
            tensors.push((format!("blk.{i}.{name}"), vec![D as u64], 0, f32_bytes(&v)));
        }
    }
    let v: Vec<f32> = r.vec(D, 0.2).iter().map(|x| x + 1.0).collect();
    tensors.push(("output_norm.weight".into(), vec![D as u64], 0, f32_bytes(&v)));
    let (ty, b) = encode(&r.vec(VOCAB * D, 0.3), "q6_k");
    tensors.push(("output.weight".into(), vec![D as u64, VOCAB as u64], ty, b));
    let meta: Vec<(&str, u32, Vec<u8>)> = vec![
        ("general.architecture", 8, encode_string_value("llama")),
        ("llama.context_length", 4, encode_u32_value(CTX as u32)),
        ("llama.embedding_length", 4, encode_u32_value(D as u32)),
        ("llama.feed_forward_length", 4, encode_u32_value(FF as u32)),
        ("llama.block_count", 4, encode_u32_value(LAYERS as u32)),
        ("llama.attention.head_count", 4, encode_u32_value(HEADS as u32)),
        ("llama.attention.head_count_kv", 4, encode_u32_value(KV_HEADS as u32)),
        ("llama.attention.layer_norm_rms_epsilon", 6, encode_f32_value(EPS)),
        ("llama.rope.freq_base", 6, encode_f32_value(BASE)),
    ];
    let m: Vec<(&str, u32, &[u8])> = meta.iter().map(|(k, t, v)| (*k, *t, v.as_slice())).collect();
    let t: Vec<(&str, &[u64], u32, &[u8])> = tensors.iter().map(|(n, d, ty, b)| (n.as_str(), d.as_slice(), *ty, b.as_slice())).collect();
    build_gguf(&m, &t)
}
