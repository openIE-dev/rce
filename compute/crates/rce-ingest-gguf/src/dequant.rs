//! Dequantization to f32 for the GGUF dtypes used by the models on
//! the user's external drive (jamba-reasoning-3b, nemotron-h-moe,
//! mamba2-2.7b).
//!
//! Math is portable Rust, no FFI. Sourced from
//! `inv-ssi-inference/src/{quantize,quant_kquant}.rs` (which is the
//! working in-house impl that already runs every model on the drive)
//! — same arithmetic, same byte layouts. Vendoring rather than
//! depending on inv-ssi-inference keeps `rce-ingest-gguf`'s
//! dependency footprint minimal: the model-load path doesn't need to
//! drag in a full LLM inference engine.

use crate::parser::{GgufDtype, GgufFile, GgufTensorInfo};
use half::{bf16, f16};

/// Number of bytes occupied by a tensor on disk (works for every
/// supported dtype + handles block-quantised types via ceil-divide).
pub fn tensor_byte_size(t: &GgufTensorInfo) -> usize {
    let n = t.numel() as usize;
    match t.dtype {
        GgufDtype::F32 => n * 4,
        GgufDtype::F64 => n * 8,
        GgufDtype::F16 | GgufDtype::BF16 => n * 2,
        GgufDtype::I8 => n,
        GgufDtype::I16 => n * 2,
        GgufDtype::I32 => n * 4,
        GgufDtype::I64 => n * 8,
        GgufDtype::Q4_0 => n.div_ceil(32) * 18,
        GgufDtype::Q4_1 => n.div_ceil(32) * 20,
        GgufDtype::Q5_0 => n.div_ceil(32) * 22,
        GgufDtype::Q5_1 => n.div_ceil(32) * 24,
        GgufDtype::Q8_0 => n.div_ceil(32) * 34,
        GgufDtype::Q8_1 => n.div_ceil(32) * 36,
        GgufDtype::Q2K => n.div_ceil(256) * 84,
        GgufDtype::Q3K => n.div_ceil(256) * 110,
        GgufDtype::Q4K => n.div_ceil(256) * 144,
        GgufDtype::Q5K => n.div_ceil(256) * 176,
        GgufDtype::Q6K => n.div_ceil(256) * 210,
        // BitNet I2_S: ceil(N/4) bytes of 2-bit codes + a 4-byte f32
        // per-tensor scale. (The +32 byte/tensor span observed via
        // offset-deltas is the data + up-to-28 B of file alignment
        // padding to the next tensor — empirically verified by the T1
        // probe; the "replicated 8×" claim from secondary sources is
        // wrong: bytes [scale_off+4 ..] are the NEXT tensor leaking in.)
        GgufDtype::I2S => n.div_ceil(4) + 4,
        // PrismML Q1_0: 128-weight block = 2 B f16 scale + 16 B 1-bit
        // codes = 18 B per block (block_q1_0, ggml-common.h).
        GgufDtype::Q1_0 => n.div_ceil(128) * 18,
        // PrismML Q2_0: 128-weight block = 2 B f16 scale + 32 B 2-bit
        // codes = 34 B per block (block_q2_0, ggml-common.h).
        GgufDtype::Q2_0 => n.div_ceil(128) * 34,
        // Tencent Sherry STQ1_0: 256-weight super-block = 32 B slot
        // codes + 8 B sign bits + 2 B f16 scale = 42 B (block_stq1_0,
        // ggml-common.h:281). 1.3125 bpw.
        GgufDtype::Stq1_0 => n.div_ceil(256) * 42,
        // Less common dtypes: rough estimates; only loaders that
        // actually encounter them need to handle them.
        _ => 0,
    }
}

/// Errors that can arise reading a tensor's bytes off the file or
/// dequantizing them to f32.
#[derive(Debug, thiserror::Error)]
pub enum ReadError {
    #[error("tensor `{name}` not found in GGUF file")]
    TensorNotFound { name: String },
    #[error("tensor `{name}` byte range [{start}..{end}] exceeds file size {file_len}")]
    OutOfBounds {
        name: String,
        start: usize,
        end: usize,
        file_len: usize,
    },
    #[error("dtype {dtype} not yet supported by dequant")]
    UnsupportedDtype { dtype: String },
}

/// Read tensor `name` from the file's mmap'd bytes and dequantize to
/// f32. Output length equals `tensor.numel()`. The supported dtype
/// set covers every dtype seen on the user's external drive
/// (F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K).
///
/// `file_bytes` is the full GGUF file as bytes (typically an mmap).
/// `data_offset` should be `gguf.data_offset`.
pub fn read_tensor_f32(
    file_bytes: &[u8],
    gguf: &GgufFile,
    name: &str,
) -> Result<Vec<f32>, ReadError> {
    let tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| ReadError::TensorNotFound { name: name.into() })?;
    let n = tensor.numel() as usize;
    let bytes_n = tensor_byte_size(tensor);
    let start = (gguf.data_offset + tensor.offset) as usize;
    let end = start + bytes_n;
    if end > file_bytes.len() {
        return Err(ReadError::OutOfBounds {
            name: name.into(),
            start,
            end,
            file_len: file_bytes.len(),
        });
    }
    let bytes = &file_bytes[start..end];
    dequantize_to_f32(bytes, tensor.dtype, n).ok_or_else(|| ReadError::UnsupportedDtype {
        dtype: format!("{:?}", tensor.dtype),
    })
}

/// Pure-bytes dequant. Returns `None` when the dtype isn't covered.
pub fn dequantize_to_f32(bytes: &[u8], dtype: GgufDtype, n: usize) -> Option<Vec<f32>> {
    match dtype {
        GgufDtype::F32 => Some(dequant_f32(bytes, n)),
        GgufDtype::F16 => Some(dequant_f16(bytes, n)),
        GgufDtype::BF16 => Some(dequant_bf16(bytes, n)),
        GgufDtype::Q4_0 => Some(dequant_q4_0(bytes, n)),
        GgufDtype::Q4_1 => Some(dequant_q4_1(bytes, n)),
        GgufDtype::Q5_0 => Some(dequant_q5_0(bytes, n)),
        GgufDtype::Q8_0 => Some(dequant_q8_0(bytes, n)),
        GgufDtype::Q3K => Some(dequant_q3_k(bytes, n)),
        GgufDtype::Q4K => Some(dequant_q4_k(bytes, n)),
        GgufDtype::Q5K => Some(dequant_q5_k(bytes, n)),
        GgufDtype::Q6K => Some(dequant_q6_k(bytes, n)),
        GgufDtype::I2S => Some(dequant_i2_s(bytes, n)),
        GgufDtype::Q1_0 => Some(dequant_q1_0(bytes, n)),
        GgufDtype::Q2_0 => Some(dequant_q2_0(bytes, n)),
        GgufDtype::Stq1_0 => Some(dequant_stq1_0(bytes, n)),
        _ => None,
    }
}

/// Tencent "Sherry" STQ1_0 32-entry codebook: `(sign<<4)|slot →
/// qpack` (a packed 4×2-bit ternary pattern, one zero + three ±1 per
/// group of 4). Verbatim from `stq1_0_codebook` in sjl623/llama.cpp
/// `STQ_0` (`ggml-common.h:530`). Per-lane 2-bit: `-1→0b00, 0→0b01,
/// +1→0b10`; `qpack = lane3<<6 | lane2<<4 | lane1<<2 | lane0`.
pub const STQ1_0_CODEBOOK: [u8; 32] = [
    // sign = 0 (first non-zero lane is +1)
    0xA9, 0x89, 0x29, 0x09, 0xA6, 0x86, 0x26, 0x06,
    0x9A, 0x92, 0x1A, 0x12, 0x6A, 0x62, 0x4A, 0x42,
    // sign = 1 (every non-zero lane negated)
    0x01, 0x21, 0x81, 0xA1, 0x04, 0x24, 0x84, 0xA4,
    0x10, 0x18, 0x90, 0x98, 0x40, 0x48, 0x60, 0x68,
];

/// **Tencent Sherry STQ1_0 dequant.** Authoritative — mirrors
/// `dequantize_row_stq1_0` in sjl623/llama.cpp `STQ_0` branch
/// (`ggml/src/ggml-quants.c:2468`, the codec behind
/// `tencent/Hy-MT1.5-1.8B-1.25bit`, PR ggml-org/llama.cpp#22836).
///
/// Layout: 256-weight super-block `{ uint8_t qs[32]; uint8_t sign[8];
/// ggml_half d; }` (42 B). For each of the 64 groups of 4 weights:
/// `code = (qs[g/2] >> (4·(g&1))) & 0xF` (two 4-bit slots/byte),
/// `sign = (sign[g/8] >> (g%8)) & 1` (8 groups/byte, LSB-first),
/// `qpack = STQ1_0_CODEBOOK[(sign<<4)|code]`, then per lane `p∈0..4`:
/// `q = (qpack >> 2p) & 3`, `y = (q − 1)·d` → `{−d, 0, +d}`. Each
/// group has exactly one zero (3:4 fine-grained sparsity).
pub fn dequant_stq1_0(bytes: &[u8], n: usize) -> Vec<f32> {
    const QK: usize = 256;
    const QS_LEN: usize = QK / 8;   // 32
    const SIGN_LEN: usize = QK / 32; // 8
    const BLOCK_BYTES: usize = QS_LEN + SIGN_LEN + 2; // 42
    debug_assert!(n % QK == 0, "STQ1_0 expects n divisible by 256, got {n}");
    let nb = n / QK;
    debug_assert!(
        bytes.len() >= nb * BLOCK_BYTES,
        "STQ1_0 tensor too small: have {} bytes, need {}",
        bytes.len(),
        nb * BLOCK_BYTES
    );
    let mut out = vec![0.0f32; n];
    for i in 0..nb {
        let base = i * BLOCK_BYTES;
        let qs = &bytes[base..base + QS_LEN];
        let sign = &bytes[base + QS_LEN..base + QS_LEN + SIGN_LEN];
        let d = f16::from_le_bytes([
            bytes[base + QS_LEN + SIGN_LEN],
            bytes[base + QS_LEN + SIGN_LEN + 1],
        ]).to_f32();
        for g in 0..(QK / 4) {
            let code = (qs[g / 2] >> (4 * (g & 1))) & 0x0F;
            let s = (sign[g / 8] >> (g % 8)) & 0x01;
            let qpack = STQ1_0_CODEBOOK[((s as usize) << 4) | code as usize];
            let o = i * QK + g * 4;
            for p in 0..4 {
                let q = (qpack >> (2 * p)) & 0x3;
                out[o + p] = (q as i32 - 1) as f32 * d;
            }
        }
    }
    out
}

/// **BitNet I2_S dequant.** Authoritative — read from Microsoft's
/// `quantize_i2_s` (`/3rdparty/llama.cpp/ggml/src/ggml-quants.c:3897`
/// `dequantize_row_i2_s`) after T3.c's degenerate output traced back
/// to a **wrong code mapping** in earlier versions of this code (and
/// in community docs).
///
/// Layout: `bytes[0..ceil(n/4)]` are 2-bit codes in 128-element
/// **interleaved** blocks — byte `gp` of block `b` holds positions
/// `128b+{gp, 32+gp, 64+gp, 96+gp}` at bits `[7:6, 5:4, 3:2, 1:0]`.
/// `bytes[ceil(n/4)..ceil(n/4)+4]` is a **single f32 scale** =
/// `max(|src|)` (absMAX, not absmean — Microsoft's encoder writes
/// `i2_scale = max(|src|)`).
///
/// **Code → ternary mapping (was wrong before; FIXED here per
/// `static const float map2bit[4] = {-1, 0, +1, 0};` in
/// `dequantize_row_i2_s`):**
/// `0b00 → −1`, `0b01 → 0`, `0b10 → +1`, `0b11 → 0`.
///
/// Dequant: `w = scale · map2bit[code]`. The earlier `{0,+1,−1,0}`
/// rotation systematically swapped every weight value (every real
/// −1 → 0, real 0 → +1, real +1 → −1) and was the root cause of T3.c's
/// gibberish output. T1's bit-exact round-trip passed regardless
/// because my encoder+decoder were self-consistent; the distribution
/// check looked right because permutations preserve histograms.
pub fn dequant_i2_s(bytes: &[u8], n: usize) -> Vec<f32> {
    let codes_len = n.div_ceil(4);
    debug_assert!(
        bytes.len() >= codes_len + 4,
        "I2_S tensor too small: have {} bytes, need {}",
        bytes.len(),
        codes_len + 4
    );
    let scale = f32::from_le_bytes([
        bytes[codes_len], bytes[codes_len + 1],
        bytes[codes_len + 2], bytes[codes_len + 3],
    ]);
    // Authoritative `map2bit` from Microsoft's dequantize_row_i2_s.
    const MAP2BIT: [f32; 4] = [-1.0, 0.0, 1.0, 0.0];
    let mut out = vec![0.0f32; n];
    let nblocks = n.div_ceil(128);
    for b in 0..nblocks {
        let blk_base = b * 128;
        let byte_base = b * 32;
        for gp in 0..32 {
            let byte = bytes[byte_base + gp];
            let c0 = ((byte >> 6) & 0x3) as usize;
            let c1 = ((byte >> 4) & 0x3) as usize;
            let c2 = ((byte >> 2) & 0x3) as usize;
            let c3 = (byte & 0x3) as usize;
            if blk_base + gp < n {
                out[blk_base + gp] = scale * MAP2BIT[c0];
            }
            if blk_base + 32 + gp < n {
                out[blk_base + 32 + gp] = scale * MAP2BIT[c1];
            }
            if blk_base + 64 + gp < n {
                out[blk_base + 64 + gp] = scale * MAP2BIT[c2];
            }
            if blk_base + 96 + gp < n {
                out[blk_base + 96 + gp] = scale * MAP2BIT[c3];
            }
        }
    }
    out
}

/// **PrismML Q1_0 dequant.** Authoritative — mirrors
/// `dequantize_row_q1_0` in `PrismML-Eng/llama.cpp` (prism branch,
/// `ggml/src/ggml-quants.c:415`):
///
/// ```c
/// const float d = GGML_FP16_TO_FP32(x[i].d);
/// const uint8_t bit = (x[i].qs[j/8] >> (j%8)) & 1;
/// y[i*qk + j] = bit ? d : -d;
/// ```
///
/// Layout: 128-weight blocks of `{ ggml_half d; uint8_t qs[16]; }` (18
/// B/block). `d = sum(|w|) / 128` from the encoder. Bit value 0 → −d,
/// 1 → +d (binary; no zero state).
pub fn dequant_q1_0(bytes: &[u8], n: usize) -> Vec<f32> {
    const QK: usize = 128;
    const BLOCK_BYTES: usize = 2 + QK / 8; // 18
    debug_assert!(n % QK == 0, "Q1_0 expects n divisible by 128, got {n}");
    let nb = n / QK;
    debug_assert!(
        bytes.len() >= nb * BLOCK_BYTES,
        "Q1_0 tensor too small: have {} bytes, need {}",
        bytes.len(),
        nb * BLOCK_BYTES
    );
    let mut out = vec![0.0f32; n];
    for i in 0..nb {
        let b = i * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[b], bytes[b + 1]]).to_f32();
        let neg_d = -d;
        let qs = &bytes[b + 2..b + BLOCK_BYTES];
        for j in 0..QK {
            let bit = (qs[j / 8] >> (j % 8)) & 1;
            out[i * QK + j] = if bit != 0 { d } else { neg_d };
        }
    }
    out
}

/// **PrismML Q2_0 dequant.** Authoritative — mirrors
/// `dequantize_row_q2_0` in `PrismML-Eng/llama.cpp` (prism branch,
/// `ggml/src/ggml-quants.c:435`):
///
/// ```c
/// const float d = GGML_FP16_TO_FP32(x[i].d);
/// const uint8_t q = (x[i].qs[j/4] >> ((j%4)*2)) & 0x03;
/// y[i*qk + j] = ((int)q - 1) * d;  // 00=-d, 01=0, 10=+d, 11=+2d
/// ```
///
/// Layout: 128-weight blocks of `{ ggml_half d; uint8_t qs[32]; }` (34
/// B/block). `d = max(|w|)` from the encoder. Quaternary value
/// mapping — `q ∈ {0,1,2,3}` maps to `(q-1)·d ∈ {−d, 0, +d, +2d}`.
/// (The model is "ternary" in its QAT distribution; the codec leaves a
/// 4th slot for outliers.)
pub fn dequant_q2_0(bytes: &[u8], n: usize) -> Vec<f32> {
    const QK: usize = 128;
    const BLOCK_BYTES: usize = 2 + QK / 4; // 34
    debug_assert!(n % QK == 0, "Q2_0 expects n divisible by 128, got {n}");
    let nb = n / QK;
    debug_assert!(
        bytes.len() >= nb * BLOCK_BYTES,
        "Q2_0 tensor too small: have {} bytes, need {}",
        bytes.len(),
        nb * BLOCK_BYTES
    );
    let mut out = vec![0.0f32; n];
    for i in 0..nb {
        let b = i * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[b], bytes[b + 1]]).to_f32();
        let qs = &bytes[b + 2..b + BLOCK_BYTES];
        for j in 0..QK {
            let q = ((qs[j / 4] >> ((j % 4) * 2)) & 0x03) as i32;
            out[i * QK + j] = ((q - 1) as f32) * d;
        }
    }
    out
}

// ── Per-element dtypes ─────────────────────────────────────────────

fn dequant_f32(data: &[u8], n: usize) -> Vec<f32> {
    let slice: &[f32] = bytemuck::cast_slice(&data[..n * 4]);
    slice.to_vec()
}

fn dequant_f16(data: &[u8], n: usize) -> Vec<f32> {
    let f16s: &[f16] = bytemuck::cast_slice(&data[..n * 2]);
    f16s.iter().map(|v| v.to_f32()).collect()
}

fn dequant_bf16(data: &[u8], n: usize) -> Vec<f32> {
    let bf16s: &[bf16] = bytemuck::cast_slice(&data[..n * 2]);
    bf16s.iter().map(|v| v.to_f32()).collect()
}

// ── Q4_0 / Q4_1 / Q5_0 / Q8_0 (32-element blocks) ────────────────

fn dequant_q4_0(data: &[u8], n: usize) -> Vec<f32> {
    // 32-element blocks; layout matches llama.cpp `dequantize_row_q4_0`:
    // first 16 outputs are low nibbles of bytes [0..16], next 16 are
    // high nibbles of the same bytes (not interleaved per-byte).
    const BLOCK: usize = 32;
    const BB: usize = 18;
    let n_blocks = n.div_ceil(BLOCK);
    let mut out = vec![0.0f32; n_blocks * BLOCK];
    for b in 0..n_blocks {
        let s = b * BB;
        if s + BB > data.len() { break; }
        let scale = f16::from_le_bytes([data[s], data[s + 1]]).to_f32();
        let out_base = b * BLOCK;
        for j in 0..16 {
            let byte = data[s + 2 + j];
            let v0 = (byte & 0x0F) as i8 - 8;
            let v1 = ((byte >> 4) & 0x0F) as i8 - 8;
            out[out_base + j] = v0 as f32 * scale;
            out[out_base + j + 16] = v1 as f32 * scale;
        }
    }
    out.truncate(n);
    out
}

fn dequant_q4_1(data: &[u8], n: usize) -> Vec<f32> {
    // Same split-nibble layout as Q4_0 (low halves first, then high
    // halves), with per-block (scale, min) offset.
    const BLOCK: usize = 32;
    const BB: usize = 20;
    let n_blocks = n.div_ceil(BLOCK);
    let mut out = vec![0.0f32; n_blocks * BLOCK];
    for b in 0..n_blocks {
        let s = b * BB;
        if s + BB > data.len() { break; }
        let scale = f16::from_le_bytes([data[s], data[s + 1]]).to_f32();
        let min = f16::from_le_bytes([data[s + 2], data[s + 3]]).to_f32();
        let out_base = b * BLOCK;
        for j in 0..16 {
            let byte = data[s + 4 + j];
            let v0 = (byte & 0x0F) as f32;
            let v1 = ((byte >> 4) & 0x0F) as f32;
            out[out_base + j] = v0 * scale + min;
            out[out_base + j + 16] = v1 * scale + min;
        }
    }
    out.truncate(n);
    out
}

fn dequant_q5_0(data: &[u8], n: usize) -> Vec<f32> {
    // Same split-nibble layout as Q4_0, with the 5th bit pulled from
    // a 32-bit `qh` block (low halves use bit `j`, high halves use
    // bit `j+16`).
    const BLOCK: usize = 32;
    const BB: usize = 22;
    let n_blocks = n.div_ceil(BLOCK);
    let mut out = vec![0.0f32; n_blocks * BLOCK];
    for b in 0..n_blocks {
        let s = b * BB;
        if s + BB > data.len() { break; }
        let scale = f16::from_le_bytes([data[s], data[s + 1]]).to_f32();
        let qh = u32::from_le_bytes([data[s + 2], data[s + 3], data[s + 4], data[s + 5]]);
        let out_base = b * BLOCK;
        for j in 0..16 {
            let byte = data[s + 6 + j];
            let h0 = ((qh >> j) & 1) as u8;
            let h1 = ((qh >> (j + 16)) & 1) as u8;
            let v0 = ((byte & 0x0F) | (h0 << 4)) as i8 - 16;
            let v1 = ((byte >> 4) | (h1 << 4)) as i8 - 16;
            out[out_base + j] = v0 as f32 * scale;
            out[out_base + j + 16] = v1 as f32 * scale;
        }
    }
    out.truncate(n);
    out
}

fn dequant_q8_0(data: &[u8], n: usize) -> Vec<f32> {
    const BLOCK: usize = 32;
    const BB: usize = 34;
    let n_blocks = n.div_ceil(BLOCK);
    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let s = b * BB;
        if s + BB > data.len() { break; }
        let scale = f16::from_le_bytes([data[s], data[s + 1]]).to_f32();
        for i in 0..32 {
            let v = data[s + 2 + i] as i8;
            out.push(v as f32 * scale);
        }
    }
    out.truncate(n);
    out
}

// ── Q3_K / Q4_K / Q5_K / Q6_K (256-element superblocks) ──────────
//
// Layout reference: ggml/llama.cpp common.h, k_quants.h.

/// llama.cpp `Q3_K`: 256-element superblock, 110 bytes.
///   `[hmask 32 B][qs 64 B][scales 12 B][d f16]`.
/// 16 6-bit signed scales (after −32) packed into the 12-byte scales
/// section via the standard `kmask1=0x03030303 / kmask2=0x0f0f0f0f`
/// re-permutation. Per output `y[j] = d · sc[is] · q`, where `q` is a
/// 3-bit value formed by the 2-bit low part from `qs` and a 1-bit
/// high mask from `hmask`. `q ∈ [−4, 3]` after the `((hm[l] & m) ? 0 : 4)`
/// shift. Mirrors `dequantize_row_q3_K`.
fn dequant_q3_k(data: &[u8], n: usize) -> Vec<f32> {
    const BLOCK: usize = 256;
    const BB: usize = 110;
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let n_blocks = n.div_ceil(BLOCK);
    let mut out = vec![0.0f32; n_blocks * BLOCK];

    let mut aux = [0u32; 4];
    for b in 0..n_blocks {
        let s = b * BB;
        if s + BB > data.len() { break; }
        let hmask = &data[s..s + 32];
        let qs = &data[s + 32..s + 96];
        let scales_bytes = &data[s + 96..s + 108];
        let d_all = f16::from_le_bytes([data[s + 108], data[s + 109]]).to_f32();

        // memcpy 12 B into aux (aux is 16 B; high 4 B unused initially).
        for k in 0..3 {
            aux[k] = u32::from_le_bytes([
                scales_bytes[k * 4], scales_bytes[k * 4 + 1],
                scales_bytes[k * 4 + 2], scales_bytes[k * 4 + 3],
            ]);
        }
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | (((tmp) & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        // scales[] now holds 16 i8 values in [0, 63]; subtract 32 to
        // recover the signed 6-bit value.
        let mut scales = [0i8; 16];
        for k in 0..4 {
            let b4 = aux[k].to_le_bytes();
            for j in 0..4 {
                scales[k * 4 + j] = b4[j] as i8;
            }
        }

        let mut m: u8 = 1;
        let mut is: usize = 0;
        let mut q_off = 0usize;
        for _n_outer in 0..2 {
            // Each outer iteration: 4 j-loops × (16+16) = 128 outputs.
            // shift goes 0,2,4,6 per j; m advances per j (left-shift).
            let mut shift = 0u32;
            for _j in 0..4 {
                let dl1 = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                let out_off1 = b * BLOCK + (_n_outer * 128) + 32 * (is - 1 - _n_outer * 4);
                let _ = out_off1; // silence (we use a direct counter)
                // (Use a running output cursor for clarity)
                for l in 0..16 {
                    let ql = ((qs[q_off + l] >> shift) & 0x03) as i32;
                    let high = if (hmask[l] & m) != 0 { 0 } else { 4 };
                    let q = ql - high;
                    let idx = b * BLOCK + _n_outer * 128 + (_j * 32) + l;
                    out[idx] = dl1 * q as f32;
                }
                let dl2 = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let ql = ((qs[q_off + l + 16] >> shift) & 0x03) as i32;
                    let high = if (hmask[l + 16] & m) != 0 { 0 } else { 4 };
                    let q = ql - high;
                    let idx = b * BLOCK + _n_outer * 128 + (_j * 32) + 16 + l;
                    out[idx] = dl2 * q as f32;
                }
                shift += 2;
                m = m.wrapping_shl(1);
            }
            q_off += 32;
        }
    }
    out.truncate(n);
    out
}

fn dequant_q4_k(data: &[u8], n: usize) -> Vec<f32> {
    // Q4_K block: 256 values per block.
    //   [d (f16)][dmin (f16)][scales (12 bytes)][qs (128 bytes)] = 144 bytes.
    //
    // Layout (matches llama.cpp `dequantize_row_q4_K`): the 256 values
    // are split into **4 groups of 64**, not 8 sub-blocks of 32. Each
    // group consumes 32 bytes of `qs` and 2 scales. Within a group:
    // - First 32 outputs = sc1·low_nibble(qs[0..32]) − min1
    // - Next 32 outputs  = sc2·high_nibble(qs[0..32]) − min2
    //
    // (Earlier impl assigned 16 bytes per 32-element sub-block, which
    // shuffled values into wrong positions — silently corrupting every
    // K-quant tensor and producing gibberish at decode time.)
    const BLOCK: usize = 256;
    const BB: usize = 144;
    let n_blocks = n.div_ceil(BLOCK);
    let mut out = vec![0.0f32; n_blocks * BLOCK];

    for b in 0..n_blocks {
        let s = b * BB;
        if s + BB > data.len() { break; }
        let d = f16::from_le_bytes([data[s], data[s + 1]]).to_f32();
        let dmin = f16::from_le_bytes([data[s + 2], data[s + 3]]).to_f32();
        let scales = &data[s + 4..s + 16];
        let qs = &data[s + 16..s + 144];

        for j in 0..4 {
            let q_base = j * 32;
            let out_base = b * BLOCK + j * 64;
            let (sc1, m1) = q4_k_get_scale_min(scales, j * 2);
            let (sc2, m2) = q4_k_get_scale_min(scales, j * 2 + 1);
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;
            for l in 0..32 {
                let byte = qs[q_base + l];
                out[out_base + l] = d1 * (byte & 0x0F) as f32 - min1;
                out[out_base + 32 + l] = d2 * ((byte >> 4) & 0x0F) as f32 - min2;
            }
        }
    }
    out.truncate(n);
    out
}

#[inline]
fn q4_k_get_scale_min(scales: &[u8], j: usize) -> (u8, u8) {
    // Standard llama.cpp packing for Q4_K subscales (12 bytes).
    if j < 4 {
        let d = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (d, m)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

fn dequant_q5_k(data: &[u8], n: usize) -> Vec<f32> {
    // Q5_K block: 256 values, 176 bytes.
    //   [d (f16)][dmin (f16)][scales (12 bytes)][qh (32 bytes)][qs (128 bytes)] = 176.
    //
    // Same 4-group-of-64 structure as Q4_K (matches llama.cpp
    // `dequantize_row_q5_K`). For each group `j` of 64 outputs:
    // - 32 bytes of `qs` advance per group (low + high nibbles)
    // - 32 bytes of `qh` are reused across all 4 groups; bits `u1` and
    //   `u2` (shifting left by 2 each group) supply the 5th bit per value
    // - 2 scales per group via `q4_k_get_scale_min(j*2 + {0,1})`
    const BLOCK: usize = 256;
    const BB: usize = 176;
    let n_blocks = n.div_ceil(BLOCK);
    let mut out = vec![0.0f32; n_blocks * BLOCK];

    for b in 0..n_blocks {
        let s = b * BB;
        if s + BB > data.len() { break; }
        let d = f16::from_le_bytes([data[s], data[s + 1]]).to_f32();
        let dmin = f16::from_le_bytes([data[s + 2], data[s + 3]]).to_f32();
        let scales = &data[s + 4..s + 16];
        let qh = &data[s + 16..s + 48];
        let qs = &data[s + 48..s + 176];

        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for j in 0..4 {
            let q_base = j * 32;
            let out_base = b * BLOCK + j * 64;
            let (sc1, m1) = q4_k_get_scale_min(scales, j * 2);
            let (sc2, m2) = q4_k_get_scale_min(scales, j * 2 + 1);
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;
            for l in 0..32 {
                let byte = qs[q_base + l];
                let h1 = if (qh[l] & u1) != 0 { 16u32 } else { 0 };
                let h2 = if (qh[l] & u2) != 0 { 16u32 } else { 0 };
                let v_lo = (byte & 0x0F) as u32 + h1;
                let v_hi = ((byte >> 4) & 0x0F) as u32 + h2;
                out[out_base + l] = d1 * v_lo as f32 - min1;
                out[out_base + 32 + l] = d2 * v_hi as f32 - min2;
            }
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    out.truncate(n);
    out
}

fn dequant_q6_k(data: &[u8], n: usize) -> Vec<f32> {
    // Q6_K block: 256 values, 210 bytes.
    //   [ql (128 bytes)][qh (64 bytes)][scales (16 bytes int8)][d (f16)] = 210.
    //
    // Layout (matches llama.cpp `dequantize_row_q6_K`): the 256 values
    // are processed in **2 outer iterations of 128**, each consuming
    // 64 bytes of `ql`, 32 bytes of `qh`, and 8 scales. Within an
    // iteration, the inner loop over `l in 0..32` produces 4 output
    // positions per iteration step (l + 0/32/64/96), each pulling a
    // 6-bit value `(lo4 | (hi2 << 4)) - 32` with a per-quartet scale.
    const BLOCK: usize = 256;
    const BB: usize = 210;
    let n_blocks = n.div_ceil(BLOCK);
    let mut out = vec![0.0f32; n_blocks * BLOCK];

    for b in 0..n_blocks {
        let s = b * BB;
        if s + BB > data.len() { break; }
        let ql = &data[s..s + 128];
        let qh = &data[s + 128..s + 192];
        let scales_i8: &[i8] = bytemuck::cast_slice(&data[s + 192..s + 208]);
        let d = f16::from_le_bytes([data[s + 208], data[s + 209]]).to_f32();

        for outer in 0..2 {
            let ql_off = outer * 64;
            let qh_off = outer * 32;
            let sc_off = outer * 8;
            let out_off = b * BLOCK + outer * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[ql_off + l] & 0x0F) as i32 | (((qh[qh_off + l] >> 0) & 0x03) as i32) << 4) - 32;
                let q2 = ((ql[ql_off + l + 32] & 0x0F) as i32 | (((qh[qh_off + l] >> 2) & 0x03) as i32) << 4) - 32;
                let q3 = ((ql[ql_off + l] >> 4) as i32 | (((qh[qh_off + l] >> 4) & 0x03) as i32) << 4) - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4) as i32 | (((qh[qh_off + l] >> 6) & 0x03) as i32) << 4) - 32;
                out[out_off + l] = d * scales_i8[sc_off + is] as f32 * q1 as f32;
                out[out_off + l + 32] = d * scales_i8[sc_off + is + 2] as f32 * q2 as f32;
                out[out_off + l + 64] = d * scales_i8[sc_off + is + 4] as f32 * q3 as f32;
                out[out_off + l + 96] = d * scales_i8[sc_off + is + 6] as f32 * q4 as f32;
            }
        }
    }
    out.truncate(n);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_byte_size_f32() {
        let t = GgufTensorInfo {
            name: "x".into(),
            dims: vec![10, 10],
            dtype: GgufDtype::F32,
            offset: 0,
        };
        assert_eq!(tensor_byte_size(&t), 10 * 10 * 4);
    }

    #[test]
    fn tensor_byte_size_q4_k() {
        // 256 elements → 1 block → 144 bytes.
        let t = GgufTensorInfo {
            name: "x".into(),
            dims: vec![256],
            dtype: GgufDtype::Q4K,
            offset: 0,
        };
        assert_eq!(tensor_byte_size(&t), 144);

        // 257 elements → 2 blocks (ceil) → 288 bytes.
        let t2 = GgufTensorInfo {
            name: "x".into(),
            dims: vec![257],
            dtype: GgufDtype::Q4K,
            offset: 0,
        };
        assert_eq!(tensor_byte_size(&t2), 288);
    }

    #[test]
    fn dequant_f32_roundtrip() {
        let xs: Vec<f32> = vec![1.0, 2.5, -0.3, 17.0];
        let bytes: &[u8] = bytemuck::cast_slice(&xs);
        let out = dequant_f32(bytes, xs.len());
        assert_eq!(out, xs);
    }

    #[test]
    fn dequant_q8_0_roundtrip_const_scale() {
        // Build a 32-element Q8_0 block manually with scale=1.0.
        let mut bytes = vec![0u8; 34];
        let scale_f16 = f16::from_f32(1.0);
        bytes[0..2].copy_from_slice(&scale_f16.to_le_bytes());
        for i in 0..32 {
            bytes[2 + i] = (i as i8 * 2) as u8;
        }
        let out = dequant_q8_0(&bytes, 32);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, (i as i8 * 2) as f32);
        }
    }

    /// **T1, gate 1 — crafted I2_S round-trip (no model needed).**
    /// Build known ternary values + a known f32 scale, encode per the
    /// authoritative spec, decode via `dequant_i2_s`, assert every
    /// element matches `scale·ternary` bit-for-bit. A spec-misread
    /// fails here.
    #[test]
    fn t1_i2_s_roundtrip_bit_exact() {
        // 256 elements = 2 full 128-elem interleaved blocks.
        let n = 256usize;
        let scale: f32 = 0.0625; // exact in f32, easy to spot in real file
        // Deterministic ternary pattern that exercises 0, +1, -1 across
        // all four group offsets within each block.
        let want: Vec<i8> = (0..n)
            .map(|i| match (i * 31 + 7) % 3 {
                0 => 0i8,
                1 => 1,
                _ => -1,
            })
            .collect();
        // Encode: per 128-elem block b, byte gp holds positions
        // {gp, 32+gp, 64+gp, 96+gp} at bits [7:6, 5:4, 3:2, 1:0].
        let codes_len = n.div_ceil(4);
        let mut bytes = vec![0u8; codes_len + 32];
        // Authoritative encoding: -1→0b00, 0→0b01, +1→0b10 (matches
        // Microsoft's `quantize_i2_s` + `map2bit[4] = {-1,0,+1,0}`).
        let to_code = |v: i8| -> u8 {
            match v {
                -1 => 0b00,
                0 => 0b01,
                1 => 0b10,
                _ => unreachable!(),
            }
        };
        for b in 0..(n / 128) {
            let blk = b * 128;
            for gp in 0..32 {
                let c0 = to_code(want[blk + gp]);
                let c1 = to_code(want[blk + 32 + gp]);
                let c2 = to_code(want[blk + 64 + gp]);
                let c3 = to_code(want[blk + 96 + gp]);
                bytes[b * 32 + gp] = (c0 << 6) | (c1 << 4) | (c2 << 2) | c3;
            }
        }
        // Trailer: single f32 scale (4 B). Allocate +28 garbage bytes
        // to mimic the inter-tensor alignment pad — dequant must IGNORE
        // them and read only the first 4 bytes after the codes.
        bytes[codes_len..codes_len + 4].copy_from_slice(&scale.to_le_bytes());
        for k in (codes_len + 4)..bytes.len() {
            bytes[k] = 0xAB; // alignment-padding garbage; codec must ignore
        }

        let got = dequant_i2_s(&bytes, n);
        assert_eq!(got.len(), n);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let expected = scale * (*w as f32);
            assert_eq!(
                g.to_bits(),
                expected.to_bits(),
                "I2_S[{i}]: got {g}, want {expected} (ternary {w}, scale {scale})"
            );
        }
        // tensor_byte_size matches the spec exactly.
        use crate::parser::GgufTensorInfo;
        let info = GgufTensorInfo {
            name: "test".into(),
            dims: vec![n as u64],
            dtype: GgufDtype::I2S,
            offset: 0,
        };
        assert_eq!(tensor_byte_size(&info), codes_len + 4);
    }

    /// **T1, gate 2 — real-file sanity on bitnet-b1.58-2b-4t-i2s.**
    /// Empirically-verified spec gates (the earlier "replicated 8×"
    /// trailer claim was disproven by this very test and retracted —
    /// see [`dequant_i2_s`]): (i) the scale is a sane positive
    /// absmean-like f32, (ii) every decoded element is **exactly** in
    /// `{-scale, 0, +scale}` (no `0b11` codes leak through), (iii)
    /// `tensor_byte_size` is `ceil(N/4) + 4`, and (iv) the result
    /// covers *several* real tensors (not just one — earlier this
    /// gate caught a spec bug that only one tensor would have hidden).
    #[test]
    fn t1_i2_s_real_bitnet_sanity() {
        use crate::parser::GgufFile;
        use std::fs::File;
        use std::io::Read;
        const MODEL: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/bitnet-b1.58-2b-4t-i2s.gguf";
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("t1 i2_s real: model not mounted — skipping");
            return;
        }
        let mut buf = Vec::new();
        File::open(MODEL).unwrap().read_to_end(&mut buf).unwrap();
        let g = GgufFile::parse(&buf).expect("parse");

        // Exercise the first 7 I2_S tensors (blk.0.* — different shapes
        // and consecutive-tensor padding patterns). The earlier
        // "replicated 8×" myth survived on small samples; running
        // multiple distinct tensors keeps the gate honest.
        let ts: Vec<_> = g.tensors.iter().filter(|t| t.dtype == GgufDtype::I2S).take(7).collect();
        assert!(!ts.is_empty(), "expected I2_S tensors");
        for t in &ts {
            let n = t.numel() as usize;
            let off = g.data_offset as usize + t.offset as usize;
            let exp = tensor_byte_size(t);
            let codes_len = n.div_ceil(4);
            assert_eq!(exp, codes_len + 4, "{}: size = ceil(N/4)+4", t.name);
            let tbytes = &buf[off..off + exp];

            let scale = f32::from_le_bytes(tbytes[codes_len..codes_len + 4].try_into().unwrap());
            assert!(
                scale.is_finite() && scale > 0.0 && scale < 10.0,
                "{}: scale {scale} not in (0, 10) absmean range",
                t.name
            );
            let out = dequant_i2_s(tbytes, n);
            let (mut z, mut p, mut neg) = (0usize, 0usize, 0usize);
            for &v in &out {
                if v == 0.0 {
                    z += 1;
                } else if v == scale {
                    p += 1;
                } else if v == -scale {
                    neg += 1;
                } else {
                    panic!(
                        "{}: element {v} ∉ {{-{scale}, 0, +{scale}}}",
                        t.name
                    );
                }
            }
            // Sanity: ternary distribution shouldn't be degenerate (a
            // dead tensor or a codec bug producing all zeros would
            // silently pass otherwise).
            assert!(p + neg > n / 10, "{}: too few nonzeros ({}+{} of {n})", t.name, p, neg);
            eprintln!(
                "T1 real I2_S OK: {:40} n={n}  scale={scale:.5}  0:{z}  +1:{p}  -1:{neg}",
                t.name
            );
        }
    }

    /// **Q1_0 crafted round-trip.** Encode known ±1 patterns + known
    /// f16 scale per the authoritative PrismML spec, decode, assert
    /// bit-exact match. Mirrors the I2_S round-trip discipline:
    /// permutation-invariant histogram checks would pass on a 4-way
    /// rotation bug, so we assert *positions* match too.
    #[test]
    fn t41_q1_0_roundtrip_bit_exact() {
        const QK: usize = 128;
        let n = 2 * QK; // 2 full blocks
        let scale_f32: f32 = 0.0625;
        let scale_f16 = f16::from_f32(scale_f32);
        // Deterministic ±1 pattern (no zero state in Q1_0).
        let want: Vec<i8> = (0..n).map(|i| if (i * 5 + 1) % 2 == 0 { 1i8 } else { -1 }).collect();
        // Encode per `quantize_row_q1_0_ref`: bit `1` iff `x[j] >= 0`.
        let nb = n / QK;
        let block_bytes = 2 + QK / 8;
        let mut bytes = vec![0u8; nb * block_bytes];
        for i in 0..nb {
            let b = i * block_bytes;
            bytes[b..b + 2].copy_from_slice(&scale_f16.to_le_bytes());
            for j in 0..QK {
                if want[i * QK + j] >= 0 {
                    bytes[b + 2 + j / 8] |= 1 << (j % 8);
                }
            }
        }
        let got = dequant_q1_0(&bytes, n);
        assert_eq!(got.len(), n);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            // Q1_0 stores *sign only*: +1 → +scale_f16 (as f32), -1 → -scale_f16.
            let expected = scale_f16.to_f32() * (*w as f32);
            assert_eq!(
                g.to_bits(),
                expected.to_bits(),
                "Q1_0[{i}]: got {g}, want {expected} (sign {w})"
            );
        }
        // tensor_byte_size matches the spec exactly.
        use crate::parser::GgufTensorInfo;
        let info = GgufTensorInfo {
            name: "test".into(),
            dims: vec![n as u64],
            dtype: GgufDtype::Q1_0,
            offset: 0,
        };
        assert_eq!(tensor_byte_size(&info), nb * block_bytes);
    }

    /// **Q2_0 crafted round-trip.** Cover ALL 4 codes {00,01,10,11} so a
    /// 4-way rotation bug (the BitNet I2_S precedent) is caught here, not
    /// in a real-model gen test 8 hours from now.
    #[test]
    fn t42_q2_0_roundtrip_bit_exact() {
        const QK: usize = 128;
        let n = 2 * QK;
        let scale_f32: f32 = 0.125;
        let scale_f16 = f16::from_f32(scale_f32);
        // Pattern uses all 4 codes: -1, 0, +1, +2 (the quaternary set).
        let want: Vec<i8> = (0..n).map(|i| (i % 4) as i8 - 1).collect();
        let nb = n / QK;
        let block_bytes = 2 + QK / 4;
        let mut bytes = vec![0u8; nb * block_bytes];
        for i in 0..nb {
            let b = i * block_bytes;
            bytes[b..b + 2].copy_from_slice(&scale_f16.to_le_bytes());
            for j in 0..QK {
                // Reverse of `q = round(w/d) + 1` clamped to [0,3].
                let q = (want[i * QK + j] + 1) as u8;
                bytes[b + 2 + j / 4] |= q << ((j % 4) * 2);
            }
        }
        let got = dequant_q2_0(&bytes, n);
        assert_eq!(got.len(), n);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let expected = scale_f16.to_f32() * (*w as f32);
            assert_eq!(
                g.to_bits(),
                expected.to_bits(),
                "Q2_0[{i}]: got {g}, want {expected} (q-1={w})"
            );
        }
        use crate::parser::GgufTensorInfo;
        let info = GgufTensorInfo {
            name: "test".into(),
            dims: vec![n as u64],
            dtype: GgufDtype::Q2_0,
            offset: 0,
        };
        assert_eq!(tensor_byte_size(&info), nb * block_bytes);
    }

    /// **Q1_0 real-file sanity gate** on `bonsai-1.7b-q1_0.gguf`.
    /// Asserts: scale is a sane positive f16 (~0.01-0.5 range per
    /// forensic decode), every dequantized element is exactly ±scale
    /// (no zero state in Q1_0), and a sample of real tensors is exercised.
    #[test]
    fn t41_q1_0_real_bonsai_sanity() {
        use crate::parser::GgufFile;
        use std::fs::File;
        use std::io::Read;
        const MODEL: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/bonsai-1.7b-q1_0.gguf";
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("t41 Q1_0 real: model not mounted — skipping");
            return;
        }
        let mut buf = Vec::new();
        File::open(MODEL).unwrap().read_to_end(&mut buf).unwrap();
        let g = GgufFile::parse(&buf).expect("parse");
        let ts: Vec<_> = g.tensors.iter().filter(|t| t.dtype == GgufDtype::Q1_0).take(7).collect();
        assert!(!ts.is_empty(), "expected Q1_0 tensors");
        for t in &ts {
            let n = t.numel() as usize;
            let off = g.data_offset as usize + t.offset as usize;
            let exp = tensor_byte_size(t);
            let nb = n / 128;
            assert_eq!(exp, nb * 18, "{}: size = nblocks * 18", t.name);
            let tbytes = &buf[off..off + exp];
            let out = dequant_q1_0(tbytes, n);
            // First block's scale (decoded from bytes[0..2]) bounds the
            // expected value magnitudes.
            let scale0 = f16::from_le_bytes([tbytes[0], tbytes[1]]).to_f32();
            assert!(
                scale0.is_finite() && scale0 > 0.0 && scale0 < 10.0,
                "{}: scale0={scale0} unreasonable",
                t.name
            );
            // First 128 elements (first block) must be in {-scale0, +scale0}.
            let mut p = 0usize;
            let mut neg = 0usize;
            for &v in &out[..128] {
                if v == scale0 { p += 1; }
                else if v == -scale0 { neg += 1; }
                else { panic!("{}: element {v} ∉ {{-{scale0}, +{scale0}}}", t.name); }
            }
            assert!(p > 8 && neg > 8, "{}: degenerate first block ({p}+/{neg}-)", t.name);
            eprintln!(
                "T41 real Q1_0 OK: {:40} n={n}  scale0={scale0:.5}  +:{p}  -:{neg}",
                t.name
            );
        }
    }

    /// **Q2_0 real-file sanity gate** on `ternary-bonsai-1.7b-q2_0.gguf`.
    /// Asserts: scale sane, every dequantized element in
    /// {-scale, 0, +scale, +2·scale}, distribution not degenerate.
    #[test]
    fn t42_q2_0_real_ternary_bonsai_sanity() {
        use crate::parser::GgufFile;
        use std::fs::File;
        use std::io::Read;
        const MODEL: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/ternary-bonsai-1.7b-q2_0.gguf";
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("t42 Q2_0 real: model not mounted — skipping");
            return;
        }
        let mut buf = Vec::new();
        File::open(MODEL).unwrap().read_to_end(&mut buf).unwrap();
        let g = GgufFile::parse(&buf).expect("parse");
        let ts: Vec<_> = g.tensors.iter().filter(|t| t.dtype == GgufDtype::Q2_0).take(7).collect();
        assert!(!ts.is_empty(), "expected Q2_0 tensors");
        for t in &ts {
            let n = t.numel() as usize;
            let off = g.data_offset as usize + t.offset as usize;
            let exp = tensor_byte_size(t);
            let nb = n / 128;
            assert_eq!(exp, nb * 34, "{}: size = nblocks * 34", t.name);
            let tbytes = &buf[off..off + exp];
            let out = dequant_q2_0(tbytes, n);
            let scale0 = f16::from_le_bytes([tbytes[0], tbytes[1]]).to_f32();
            assert!(
                scale0.is_finite() && scale0 > 0.0 && scale0 < 10.0,
                "{}: scale0={scale0} unreasonable",
                t.name
            );
            // First 128 elements must lie in the 4-valued quaternary set.
            let (mut nz, mut z, mut po, mut pt) = (0usize, 0usize, 0usize, 0usize);
            for &v in &out[..128] {
                if v == -scale0 { nz += 1; }
                else if v == 0.0 { z += 1; }
                else if v == scale0 { po += 1; }
                else if v == 2.0 * scale0 { pt += 1; }
                else {
                    panic!("{}: element {v} ∉ {{-{scale0},0,+{scale0},+{}}}",
                           t.name, 2.0 * scale0);
                }
            }
            // Ternary-Bonsai is QAT'd toward {-1,0,+1}; +2 should be rare
            // but the 4-way set must be covered legally.
            assert!(nz + po > 4, "{}: too few nonzeros ({nz}+/{po}+)", t.name);
            eprintln!(
                "T42 real Q2_0 OK: {:40} n={n}  scale0={scale0:.5}  -:{nz} 0:{z} +:{po} +2:{pt}",
                t.name
            );
        }
    }

    /// **STQ1_0 crafted round-trip.** Build a 256-weight super-block by
    /// emitting valid codebook entries (every (sign,slot) ∈ 0..32 → a
    /// real qpack), pack qs/sign per the authoritative byte layout,
    /// decode, and assert each lane matches `(q-1)·d` AND the 3:4
    /// sparsity invariant (exactly one zero per group of 4). A codebook
    /// transcription error or byte-layout misread fails here.
    #[test]
    fn t42_stq1_0_roundtrip_bit_exact() {
        const QK: usize = 256;
        const QS_LEN: usize = QK / 8;   // 32
        const SIGN_LEN: usize = QK / 32; // 8
        const BB: usize = QS_LEN + SIGN_LEN + 2; // 42
        let n = 2 * QK; // 2 super-blocks
        let d_f32 = 0.0625f32;
        let d_f16 = f16::from_f32(d_f32);
        let nb = n / QK;
        let mut bytes = vec![0u8; nb * BB];
        // For group g (0..64 per block) choose (sign, slot) deterministically
        // across the full 0..32 codebook range so every entry is exercised.
        let pick = |g: usize| -> (u8, u8) {
            let idx = (g * 7 + 1) % 32;
            ((idx >> 4) as u8 & 1, (idx & 0x0F) as u8)
        };
        for i in 0..nb {
            let base = i * BB;
            for g in 0..(QK / 4) {
                let (s, slot) = pick(g + i);
                // qs: two 4-bit slots per byte (low nibble = even g).
                bytes[base + g / 2] |= slot << (4 * (g & 1));
                // sign: 8 groups per byte, LSB-first.
                if s != 0 {
                    bytes[base + QS_LEN + g / 8] |= 1 << (g % 8);
                }
            }
            bytes[base + QS_LEN + SIGN_LEN..base + BB]
                .copy_from_slice(&d_f16.to_le_bytes());
        }
        let got = dequant_stq1_0(&bytes, n);
        assert_eq!(got.len(), n);
        for i in 0..nb {
            for g in 0..(QK / 4) {
                let (s, slot) = pick(g + i);
                let qpack = STQ1_0_CODEBOOK[((s as usize) << 4) | slot as usize];
                let mut zeros = 0;
                for p in 0..4 {
                    let q = (qpack >> (2 * p)) & 0x3;
                    let want = (q as i32 - 1) as f32 * d_f16.to_f32();
                    let idx = i * QK + g * 4 + p;
                    assert_eq!(
                        got[idx].to_bits(),
                        want.to_bits(),
                        "STQ1_0[{idx}] g={g} p={p}: got {} want {want}",
                        got[idx]
                    );
                    if q == 1 { zeros += 1; }
                }
                // 3:4 fine-grained sparsity: exactly one zero per group.
                assert_eq!(zeros, 1, "group {g}: {zeros} zeros (expected exactly 1)");
            }
        }
        use crate::parser::GgufTensorInfo;
        let info = GgufTensorInfo {
            name: "test".into(),
            dims: vec![n as u64],
            dtype: GgufDtype::Stq1_0,
            offset: 0,
        };
        assert_eq!(tensor_byte_size(&info), nb * BB);
    }

    /// **STQ1_0 real-file sanity gate** on `tencent-hy-mt1.5-1.8b-1.25bit`.
    /// After the arch-aware tag-42 → Stq1_0 remap, every body matmul
    /// tensor must (i) size as `ceil(N/256)·42`, (ii) decode to a
    /// 3-valued set `{−s, 0, +s}` with a sane positive f16 scale, and
    /// (iii) exhibit the 3:4 sparsity invariant (≈25% zeros — the
    /// codebook structurally forces exactly 1 zero per 4 lanes).
    #[test]
    fn t42_stq1_0_real_tencent_sanity() {
        use crate::parser::GgufFile;
        use std::fs::File;
        use std::io::Read;
        const MODEL: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/tencent-hy-mt1.5-1.8b-1.25bit.gguf";
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("t42 STQ1_0 real: model not mounted — skipping");
            return;
        }
        let mut buf = Vec::new();
        File::open(MODEL).unwrap().read_to_end(&mut buf).unwrap();
        let g = GgufFile::parse(&buf).expect("parse");
        // The arch-aware remap must have fired (hunyuan-dense).
        assert_eq!(g.architecture(), Some("hunyuan-dense"));
        let ts: Vec<_> = g.tensors.iter().filter(|t| t.dtype == GgufDtype::Stq1_0).take(6).collect();
        assert!(!ts.is_empty(), "expected Stq1_0 tensors after arch remap");
        for t in &ts {
            let n = t.numel() as usize;
            let off = g.data_offset as usize + t.offset as usize;
            let exp = tensor_byte_size(t);
            assert_eq!(exp, n.div_ceil(256) * 42, "{}: size = ceil(N/256)*42", t.name);
            let out = dequant_stq1_0(&buf[off..off + exp], n);
            // First super-block scale bounds the magnitudes.
            let s0 = f16::from_le_bytes([buf[off + 40], buf[off + 41]]).to_f32();
            assert!(s0.is_finite() && s0 > 0.0 && s0 < 10.0, "{}: scale0={s0}", t.name);
            let (mut nz, mut z, mut po) = (0usize, 0usize, 0usize);
            for &v in &out[..256] {
                if v == -s0 { nz += 1; }
                else if v == 0.0 { z += 1; }
                else if v == s0 { po += 1; }
                else { panic!("{}: element {v} ∉ {{-{s0},0,+{s0}}}", t.name); }
            }
            // 3:4 sparsity ⇒ exactly 64 zeros per 256 weights (25%).
            assert_eq!(z, 64, "{}: {z} zeros in 256 (3:4 sparsity ⇒ exactly 64)", t.name);
            assert!(nz + po == 192, "{}: {} nonzeros (expected 192)", t.name, nz + po);
            eprintln!(
                "T42 real STQ1_0 OK: {:40} n={n} scale0={s0:.5} -:{nz} 0:{z} +:{po}",
                t.name
            );
        }
    }
}
