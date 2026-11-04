//! Quantization type metadata.
//!
//! Maps each `GgufDtype` to its graph type string, bits per weight,
//! block size, and a human-readable dequantization description.

use crate::parser::GgufDtype;

/// Quantization metadata for a GGUF data type.
#[derive(Debug, Clone)]
pub struct QuantInfo {
    /// graph type string (e.g., `"q4_0"`, `"f16"`, `"bf16"`).
    pub rce_type: &'static str,
    /// Effective bits per scalar weight element.
    pub bits_per_weight: f32,
    /// Block size in elements (for block quantization). 1 for unquantized types.
    pub block_size: u32,
    /// Brief description of the dequantization method.
    pub description: &'static str,
}

/// Look up quantization metadata for a given dtype.
pub fn quant_info(dtype: GgufDtype) -> QuantInfo {
    match dtype {
        GgufDtype::F32 => QuantInfo {
            rce_type: "f32",
            bits_per_weight: 32.0,
            block_size: 1,
            description: "IEEE 754 single-precision float",
        },
        GgufDtype::F16 => QuantInfo {
            rce_type: "f16",
            bits_per_weight: 16.0,
            block_size: 1,
            description: "IEEE 754 half-precision float",
        },
        GgufDtype::BF16 => QuantInfo {
            rce_type: "bf16",
            bits_per_weight: 16.0,
            block_size: 1,
            description: "Brain float 16 — 8-bit exponent, 7-bit mantissa",
        },
        GgufDtype::F64 => QuantInfo {
            rce_type: "f64",
            bits_per_weight: 64.0,
            block_size: 1,
            description: "IEEE 754 double-precision float",
        },
        GgufDtype::Q4_0 => QuantInfo {
            rce_type: "q4_0",
            bits_per_weight: 4.5,
            block_size: 32,
            description: "4-bit quantization, 32-element blocks, 1 f16 scale per block",
        },
        GgufDtype::Q4_1 => QuantInfo {
            rce_type: "q4_1",
            bits_per_weight: 5.0,
            block_size: 32,
            description: "4-bit quantization, 32-element blocks, f16 scale + f16 min per block",
        },
        GgufDtype::Q5_0 => QuantInfo {
            rce_type: "q5_0",
            bits_per_weight: 5.5,
            block_size: 32,
            description: "5-bit quantization, 32-element blocks, 1 f16 scale per block",
        },
        GgufDtype::Q5_1 => QuantInfo {
            rce_type: "q5_1",
            bits_per_weight: 6.0,
            block_size: 32,
            description: "5-bit quantization, 32-element blocks, f16 scale + f16 min per block",
        },
        GgufDtype::Q8_0 => QuantInfo {
            rce_type: "q8_0",
            bits_per_weight: 8.5,
            block_size: 32,
            description: "8-bit quantization, 32-element blocks, 1 f32 scale per block",
        },
        GgufDtype::Q8_1 => QuantInfo {
            rce_type: "q8_1",
            bits_per_weight: 9.0,
            block_size: 32,
            description: "8-bit quantization, 32-element blocks, f32 scale + f32 sum per block",
        },
        GgufDtype::Q2K => QuantInfo {
            rce_type: "q2_k",
            bits_per_weight: 2.5625,
            block_size: 256,
            description: "K-quant 2-bit, 256-element super-blocks with nested sub-blocks",
        },
        GgufDtype::Q3K => QuantInfo {
            rce_type: "q3_k",
            bits_per_weight: 3.4375,
            block_size: 256,
            description: "K-quant 3-bit, 256-element super-blocks with nested sub-blocks",
        },
        GgufDtype::Q4K => QuantInfo {
            rce_type: "q4_k",
            bits_per_weight: 4.5,
            block_size: 256,
            description: "K-quant 4-bit, 256-element super-blocks, f16 scale + f16 min",
        },
        GgufDtype::Q5K => QuantInfo {
            rce_type: "q5_k",
            bits_per_weight: 5.5,
            block_size: 256,
            description: "K-quant 5-bit, 256-element super-blocks, f16 scale + f16 min",
        },
        GgufDtype::Q6K => QuantInfo {
            rce_type: "q6_k",
            bits_per_weight: 6.5625,
            block_size: 256,
            description: "K-quant 6-bit, 256-element super-blocks, f16 scale per sub-block",
        },
        GgufDtype::IQ2XXS => QuantInfo {
            rce_type: "iq2_xxs",
            bits_per_weight: 2.0625,
            block_size: 256,
            description: "Importance-matrix 2-bit, ~2.06 bpw, codebook lookup",
        },
        GgufDtype::IQ2XS => QuantInfo {
            rce_type: "iq2_xs",
            bits_per_weight: 2.3125,
            block_size: 256,
            description: "Importance-matrix 2-bit, ~2.31 bpw, codebook lookup",
        },
        GgufDtype::IQ2S => QuantInfo {
            rce_type: "iq2_s",
            bits_per_weight: 2.5,
            block_size: 256,
            description: "Importance-matrix 2-bit, ~2.5 bpw, codebook lookup",
        },
        GgufDtype::IQ3XXS => QuantInfo {
            rce_type: "iq3_xxs",
            bits_per_weight: 3.0625,
            block_size: 256,
            description: "Importance-matrix 3-bit, ~3.06 bpw, codebook lookup",
        },
        GgufDtype::IQ3S => QuantInfo {
            rce_type: "iq3_s",
            bits_per_weight: 3.4375,
            block_size: 256,
            description: "Importance-matrix 3-bit, ~3.44 bpw, codebook lookup",
        },
        GgufDtype::IQ1S => QuantInfo {
            rce_type: "iq1_s",
            bits_per_weight: 1.5625,
            block_size: 256,
            description: "Importance-matrix 1-bit, ~1.56 bpw, ternary codebook",
        },
        GgufDtype::IQ4NL => QuantInfo {
            rce_type: "iq4_nl",
            bits_per_weight: 4.5,
            block_size: 32,
            description: "Importance-matrix 4-bit, non-linear quant map",
        },
        GgufDtype::IQ4XS => QuantInfo {
            rce_type: "iq4_xs",
            bits_per_weight: 4.25,
            block_size: 256,
            description: "Importance-matrix 4-bit, ~4.25 bpw, codebook + scale",
        },
        GgufDtype::I2S => QuantInfo {
            rce_type: "i2_s",
            bits_per_weight: 2.0,
            block_size: 128,
            description: "BitNet b1.58 ternary: 2-bit codes (00=0, 01=+1, 10=-1) in 128-elem interleaved blocks (byte gp = positions {gp, 32+gp, 64+gp, 96+gp}); single per-tensor f32 absmean scale (4 B trailer; up to 28 B inter-tensor alignment padding follows but is not part of the tensor); w = scale · ternary",
        },
        GgufDtype::Q1_0 => QuantInfo {
            rce_type: "q1_0",
            bits_per_weight: 1.125,
            block_size: 128,
            description: "PrismML Bonsai 1-bit binary: 128-weight block of {ggml_half d; uint8_t qs[16];}; d = sum(|w|)/128 (mean-abs); bit 0=-d, 1=+d; layout `qs[j/8] >> (j%8) & 1`; 18 B/block = 1.125 bpw",
        },
        GgufDtype::Q2_0 => QuantInfo {
            rce_type: "q2_0",
            bits_per_weight: 2.125,
            block_size: 128,
            description: "PrismML Ternary-Bonsai 2-bit quaternary: 128-weight block of {ggml_half d; uint8_t qs[32];}; d = max(|w|); 2-bit codes 00=-d, 01=0, 10=+d, 11=+2d via `y = (q-1)*d`; layout `qs[j/4] >> ((j%4)*2) & 3`; 34 B/block = 2.125 bpw. WARNING: GGUF tag 42 is also used by Tencent Hunyuan's 1.25-bit codec — dispatch on general.architecture, not tag alone",
        },
        GgufDtype::Stq1_0 => QuantInfo {
            rce_type: "stq1_0",
            bits_per_weight: 1.3125,
            block_size: 256,
            description: "Tencent Sherry STQ1_0 ternary: 256-wt super-block {uint8 qs[32]; uint8 sign[8]; f16 d}; 3:4 fine-grained sparsity (exactly one 0 + three ±d per group of 4); (sign<<4|slot) indexes 32-entry codebook → 4×2-bit pattern; (q-1)*d ∈ {-d,0,+d}; 42 B/256 wt = 1.3125 bpw. GGUF tag 42 SHARED with PrismML Q2_0 — resolved by general.architecture=hunyuan-dense",
        },
        GgufDtype::I8 => QuantInfo {
            rce_type: "i8",
            bits_per_weight: 8.0,
            block_size: 1,
            description: "8-bit signed integer",
        },
        GgufDtype::I16 => QuantInfo {
            rce_type: "i16",
            bits_per_weight: 16.0,
            block_size: 1,
            description: "16-bit signed integer",
        },
        GgufDtype::I32 => QuantInfo {
            rce_type: "i32",
            bits_per_weight: 32.0,
            block_size: 1,
            description: "32-bit signed integer",
        },
        GgufDtype::I64 => QuantInfo {
            rce_type: "i64",
            bits_per_weight: 64.0,
            block_size: 1,
            description: "64-bit signed integer",
        },
        GgufDtype::Unknown(_tag) => QuantInfo {
            rce_type: "unknown",
            bits_per_weight: 0.0,
            block_size: 1,
            description: "unknown quantization type",
        },
    }
}

/// Estimated total bytes for a tensor given its element count and dtype.
pub fn estimated_bytes(numel: u64, dtype: GgufDtype) -> u64 {
    let info = quant_info(dtype);
    let total_bits = (numel as f64) * (info.bits_per_weight as f64);
    (total_bits / 8.0).ceil() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_info() {
        let info = quant_info(GgufDtype::F32);
        assert_eq!(info.rce_type, "f32");
        assert_eq!(info.bits_per_weight, 32.0);
        assert_eq!(info.block_size, 1);
    }

    #[test]
    fn f16_info() {
        let info = quant_info(GgufDtype::F16);
        assert_eq!(info.rce_type, "f16");
        assert_eq!(info.bits_per_weight, 16.0);
    }

    #[test]
    fn q4_0_info() {
        let info = quant_info(GgufDtype::Q4_0);
        assert_eq!(info.rce_type, "q4_0");
        assert_eq!(info.bits_per_weight, 4.5);
        assert_eq!(info.block_size, 32);
    }

    #[test]
    fn q4k_info() {
        let info = quant_info(GgufDtype::Q4K);
        assert_eq!(info.rce_type, "q4_k");
        assert_eq!(info.bits_per_weight, 4.5);
        assert_eq!(info.block_size, 256);
    }

    #[test]
    fn q6k_info() {
        let info = quant_info(GgufDtype::Q6K);
        assert_eq!(info.rce_type, "q6_k");
        assert_eq!(info.bits_per_weight, 6.5625);
    }

    #[test]
    fn bf16_info() {
        let info = quant_info(GgufDtype::BF16);
        assert_eq!(info.rce_type, "bf16");
        assert_eq!(info.bits_per_weight, 16.0);
    }

    #[test]
    fn iq_types() {
        assert_eq!(quant_info(GgufDtype::IQ1S).rce_type, "iq1_s");
        assert_eq!(quant_info(GgufDtype::IQ2XXS).rce_type, "iq2_xxs");
        assert_eq!(quant_info(GgufDtype::IQ2XS).rce_type, "iq2_xs");
        assert_eq!(quant_info(GgufDtype::IQ2S).rce_type, "iq2_s");
        assert_eq!(quant_info(GgufDtype::IQ3XXS).rce_type, "iq3_xxs");
        assert_eq!(quant_info(GgufDtype::IQ3S).rce_type, "iq3_s");
        assert_eq!(quant_info(GgufDtype::IQ4NL).rce_type, "iq4_nl");
        assert_eq!(quant_info(GgufDtype::IQ4XS).rce_type, "iq4_xs");
    }

    #[test]
    fn unknown_type() {
        let info = quant_info(GgufDtype::Unknown(999));
        assert_eq!(info.rce_type, "unknown");
        assert_eq!(info.bits_per_weight, 0.0);
    }

    #[test]
    fn estimated_bytes_q4_0() {
        // 32 elements * 4.5 bpw = 144 bits = 18 bytes
        assert_eq!(estimated_bytes(32, GgufDtype::Q4_0), 18);
    }

    #[test]
    fn estimated_bytes_f32() {
        // 1024 elements * 32 bpw = 32768 bits = 4096 bytes
        assert_eq!(estimated_bytes(1024, GgufDtype::F32), 4096);
    }

    #[test]
    fn all_known_dtypes_have_nonzero_bpw() {
        let known = [
            GgufDtype::F32, GgufDtype::F16, GgufDtype::BF16, GgufDtype::F64,
            GgufDtype::Q4_0, GgufDtype::Q4_1, GgufDtype::Q5_0, GgufDtype::Q5_1,
            GgufDtype::Q8_0, GgufDtype::Q8_1,
            GgufDtype::Q2K, GgufDtype::Q3K, GgufDtype::Q4K, GgufDtype::Q5K, GgufDtype::Q6K,
            GgufDtype::IQ1S, GgufDtype::IQ2XXS, GgufDtype::IQ2XS, GgufDtype::IQ2S,
            GgufDtype::IQ3XXS, GgufDtype::IQ3S, GgufDtype::IQ4NL, GgufDtype::IQ4XS,
            GgufDtype::I8, GgufDtype::I16, GgufDtype::I32, GgufDtype::I64,
        ];
        for dtype in known {
            let info = quant_info(dtype);
            assert!(info.bits_per_weight > 0.0, "dtype {dtype:?} has zero bpw");
        }
    }
}
