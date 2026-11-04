//! Authoritative empirical identification of an unknown GGUF quant
//! type: the file's own byte layout cannot lie. For each tensor whose
//! dtype the parser flags `Unknown(tag)`, derive bits-per-weight from
//! the on-disk span (next tensor's offset − this offset; alignment pad
//! is ≤31 B, negligible vs MB tensors) ÷ element count. The measured
//! bpw + the self-declared `general.*` metadata pins the format
//! (TQ1_0≈1.6875, I2_S/TQ2_0≈2.0/2.0625, Q2_K≈2.625, IQ2≈2.0–2.3).

use rce_ingest_gguf::{GgufDtype, GgufFile, GgufTensorInfo, GgufValue};
use std::fs::File;
use std::io::Read;

fn main() {
    let path = std::env::args().nth(1).expect("usage: ternary_probe <model.gguf>");
    let mut buf = Vec::new();
    File::open(&path).unwrap().read_to_end(&mut buf).unwrap();
    let g = GgufFile::parse(&buf).expect("parse");

    println!("══ {path} ══");
    println!("  data_offset={}  alignment={}", g.data_offset, g.alignment);
    for kv in &g.metadata {
        // Skip the giant tokenizer.* arrays (tokens/merges/scores); keep
        // the small scalar tokenizer hints (model, pre, *_token_id, etc.).
        let is_huge_tokenizer = kv.key.starts_with("tokenizer.")
            && (kv.key.ends_with(".tokens")
                || kv.key.ends_with(".merges")
                || kv.key.ends_with(".scores")
                || kv.key.ends_with(".token_type")
                || kv.key.ends_with(".chat_template"));
        if is_huge_tokenizer {
            continue;
        }
        if kv.key.starts_with("general.")
            || kv.key.starts_with("bitnet")
            || kv.key.starts_with("llama.")
            || kv.key.starts_with("qwen")
            || kv.key.starts_with("hunyuan")
            || kv.key.contains("rope")
            || kv.key.contains("head")
            || kv.key.contains("layer")
            || kv.key.contains("embed")
            || kv.key.contains("vocab")
            || kv.key.contains("context")
            || kv.key.contains("feed_forward")
            || kv.key.starts_with("tokenizer.")
        {
            let v = match &kv.value {
                GgufValue::U32(x) => format!("U32({x})"),
                GgufValue::I32(x) => format!("I32({x})"),
                GgufValue::U64(x) => format!("U64({x})"),
                GgufValue::String(s) => format!("\"{s}\""),
                other => format!("{other:?}"),
            };
            println!("  meta {} = {v}", kv.key);
        }
    }

    // Sort tensors by data offset → consecutive spans = on-disk bytes.
    let mut ts: Vec<_> = g.tensors.iter().collect();
    ts.sort_by_key(|t| t.offset);
    let file_end = buf.len() as u64 - g.data_offset; // data-relative end

    // Per distinct dtype tag, report a representative big tensor's bpw.
    use std::collections::BTreeMap;
    let mut seen: BTreeMap<String, ()> = BTreeMap::new();
    for i in 0..ts.len() {
        let t = ts[i];
        let tag = format!("{:?}", t.dtype);
        let numel: u64 = t.dims.iter().product();
        if numel == 0 {
            continue;
        }
        let end = if i + 1 < ts.len() { ts[i + 1].offset } else { file_end };
        let bytes = end.saturating_sub(t.offset);
        let bpw = (bytes as f64) * 8.0 / (numel as f64);
        // First occurrence of each dtype, and the largest tensor of it.
        let key = tag.clone();
        // For I2_S tensors of varying shape, dump the 32-byte trailer
        // as 8 f32s (it's NOT a single value replicated — the gate
        // proved that; see what's actually there).
        if t.dtype == GgufDtype::I2S {
            let numel = numel as usize;
            let off = g.data_offset as usize + t.offset as usize;
            let codes_len = numel.div_ceil(4);
            if buf.len() >= off + codes_len + 32 {
                let trailer = &buf[off + codes_len..off + codes_len + 32];
                let mut vs = [0f32; 8];
                for k in 0..8 {
                    vs[k] = f32::from_le_bytes(
                        trailer[4 * k..4 * k + 4].try_into().unwrap(),
                    );
                }
                println!(
                    "  TRAILER {:40} dims={:?}  f32x8 = {vs:?}",
                    t.name, t.dims
                );
            }
        }
        // For per-layer mixer-kind survey: always show any blk.*.attn_q
        // or blk.*.shortconv.conv (one tensor per layer of each kind).
        // Also include blk.2.* tensors fully for shape inspection.
        let is_lfm2_mixer = t.name.contains(".attn_q.")
            || t.name.contains(".shortconv.conv.")
            || t.name.starts_with("blk.2.");
        let first_of_kind = seen.insert(key, ()).is_none();
        if first_of_kind
            || bytes > 4_000_000
            || t.name.starts_with("blk.0.")
            || is_lfm2_mixer
        {
            println!(
                "  tensor {:42} dtype={:14} dims={:?} numel={numel} bytes={bytes} bpw={bpw:.4}",
                t.name, tag, t.dims
            );
        }

        // Forensic codec analysis: for the first tensor of any Unknown
        // dtype tag, dump structural hypotheses. Match measured bytes
        // against candidate (sub-block size, weight-bits, scale-bytes,
        // overhead-bytes) tuples; read scale candidates as f16/f32 and
        // print so a human can sanity-check (typical weight scales are
        // small positive floats, 1e-4 .. 1.0).
        if first_of_kind && matches!(t.dtype, GgufDtype::Unknown(_)) {
            forensic(&buf, g.data_offset, t, bytes, numel as usize);
        }
    }
}

/// For each candidate block layout (block_size_in_weights, total_bytes_per_block),
/// see whether `bytes` divides cleanly into that block size. If so, decode the
/// candidate scale fields at the assumed positions.
fn forensic(buf: &[u8], data_offset: u64, t: &GgufTensorInfo, bytes: u64, numel: usize) {
    let base = data_offset as usize + t.offset as usize;
    println!("    forensic for {}: numel={numel}, bytes={bytes}", t.name);
    // Hex dump of first 64 bytes — sometimes the leading bytes reveal a
    // sentinel/magic, sometimes they reveal a leading scale block.
    let dump_len = 64.min(buf.len() - base);
    print!("    head[{dump_len}]:");
    for &b in &buf[base..base + dump_len] {
        print!(" {b:02x}");
    }
    println!();
    // Candidate block layouts: (block_weights, weight_bytes, extra_bytes_per_block).
    // weight_bytes = block_weights * bits_per_weight / 8 (round up).
    let candidates: &[(usize, &str, usize, usize)] = &[
        // (block_weights, label, weight_bytes, extra_bytes_per_block)
        (256, "256w@1b+f32sc",   32, 4),   // 1.125 bpw
        (256, "256w@2b+f32sc",   64, 4),   // 2.125 bpw
        (256, "256w@1b+10B-meta", 32, 10), // 1.3125 bpw (Tencent hypothesis A)
        (128, "128w@1b+f16sc",   16, 2),   // 1.125 bpw — Bonsai Q1_0 confirmed
        (128, "128w@2b+f16sc",   32, 2),   // 2.125 bpw — Q2_0 hypothesis
        (128, "128w@1b+5B-meta", 16, 5),   // 1.3125 bpw alt
        (64,  "64w@1b+1B-meta",   8, 1),   // 1.125 bpw alt
        (32,  "32w@1b+f16sc",     4, 2),   // 1.5 bpw — sanity check
    ];
    for &(blk_w, label, w_bytes, x_bytes) in candidates {
        let blk_total = w_bytes + x_bytes;
        if numel % blk_w != 0 { continue; }
        let expected = (numel / blk_w) * blk_total;
        if expected as u64 != bytes { continue; }
        println!("    ✓ FITS layout {label}: block_weights={blk_w}, weight_bytes={w_bytes}, extra={x_bytes}");
        // Decode scale candidates at the assumed positions for the first 3 blocks.
        // Two common layouts:
        //   (a) [weight_bytes | extra_bytes]   — scale TRAILS the weights
        //   (b) [extra_bytes  | weight_bytes]  — scale LEADS the weights
        // Print both interpretations; the sensible-looking one wins.
        for blk_idx in 0..3.min(numel / blk_w) {
            let blk_base = base + blk_idx * blk_total;
            // (a) trailing scale
            let trail = &buf[blk_base + w_bytes..blk_base + blk_total];
            // (b) leading scale
            let lead = &buf[blk_base..blk_base + x_bytes];
            let to_floats = |sl: &[u8]| -> Vec<f32> {
                let mut out = Vec::new();
                let mut i = 0;
                while i + 4 <= sl.len() {
                    out.push(f32::from_le_bytes(sl[i..i + 4].try_into().unwrap()));
                    i += 4;
                }
                out
            };
            let to_f16s = |sl: &[u8]| -> Vec<f32> {
                let mut out = Vec::new();
                let mut i = 0;
                while i + 2 <= sl.len() {
                    let bits = u16::from_le_bytes(sl[i..i + 2].try_into().unwrap());
                    out.push(f16_to_f32(bits));
                    i += 2;
                }
                out
            };
            println!(
                "      blk{blk_idx}: trail-f32={:?} trail-f16={:?} lead-f32={:?} lead-f16={:?}",
                to_floats(trail), to_f16s(trail), to_floats(lead), to_f16s(lead)
            );
        }
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    // IEEE 754 binary16 → binary32.
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal: renormalize.
            let mut e: i32 = -14;
            let mut m = mant;
            while m & 0x400 == 0 { m <<= 1; e -= 1; }
            let m = m & 0x3ff;
            (sign << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}
