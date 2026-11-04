//! GGUF v3 binary parser.
//!
//! Reads GGUF headers, metadata key-value pairs, and tensor info entries
//! from a byte slice.  Never reads actual weight data blobs.

use std::fmt;

/// Magic bytes: ASCII `GGUF`.
const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

/// Minimum supported version.
const MIN_VERSION: u32 = 2;
/// Maximum supported version.
const MAX_VERSION: u32 = 3;

// ── Error ──────────────────────────────────────────────────────────────

/// Errors that can occur while parsing a GGUF file.
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("invalid magic: expected GGUF (0x47475546), got {0:#010x}")]
    InvalidMagic(u32),

    #[error("unsupported version {0} (supported: {MIN_VERSION}..={MAX_VERSION})")]
    UnsupportedVersion(u32),

    #[error("unexpected end of input at offset {0}")]
    UnexpectedEof(usize),

    #[error("invalid utf-8 in string at offset {offset}: {source}")]
    InvalidUtf8 {
        offset: usize,
        source: std::string::FromUtf8Error,
    },

    #[error("unknown metadata value type tag {0}")]
    UnknownValueType(u32),

    #[error("tensor dimension count {0} exceeds maximum (8)")]
    TooManyDimensions(u32),
}

// ── Cursor ─────────────────────────────────────────────────────────────

/// A zero-copy cursor over a `&[u8]` slice.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], ParseError> {
        if self.remaining() < n {
            return Err(ParseError::UnexpectedEof(self.pos));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, ParseError> {
        let b = self.read_bytes(1)?;
        Ok(b[0])
    }

    fn read_i8(&mut self) -> Result<i8, ParseError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, ParseError> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_i16(&mut self) -> Result<i16, ParseError> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32, ParseError> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self) -> Result<i32, ParseError> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64, ParseError> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64, ParseError> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32, ParseError> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f64(&mut self) -> Result<f64, ParseError> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Read a GGUF string: u64 length prefix + raw bytes.
    fn read_string(&mut self) -> Result<String, ParseError> {
        let len = self.read_u64()? as usize;
        let offset = self.pos;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| ParseError::InvalidUtf8 { offset, source: e })
    }

    /// Read a bool: single byte, 0 = false, nonzero = true.
    fn read_bool(&mut self) -> Result<bool, ParseError> {
        Ok(self.read_u8()? != 0)
    }
}

// ── GGUF value types ───────────────────────────────────────────────────

/// GGUF metadata value — can hold any scalar or array type.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<GgufValue>),
}

impl fmt::Display for GgufValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufValue::U8(v) => write!(f, "{v}"),
            GgufValue::I8(v) => write!(f, "{v}"),
            GgufValue::U16(v) => write!(f, "{v}"),
            GgufValue::I16(v) => write!(f, "{v}"),
            GgufValue::U32(v) => write!(f, "{v}"),
            GgufValue::I32(v) => write!(f, "{v}"),
            GgufValue::F32(v) => write!(f, "{v}"),
            GgufValue::Bool(v) => write!(f, "{v}"),
            GgufValue::String(v) => write!(f, "{v}"),
            GgufValue::U64(v) => write!(f, "{v}"),
            GgufValue::I64(v) => write!(f, "{v}"),
            GgufValue::F64(v) => write!(f, "{v}"),
            GgufValue::Array(vs) => {
                write!(f, "[")?;
                for (i, v) in vs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
        }
    }
}

/// GGUF metadata value type tags (from gguf spec).
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

fn read_value(cur: &mut Cursor<'_>, type_tag: u32) -> Result<GgufValue, ParseError> {
    match type_tag {
        GGUF_TYPE_UINT8 => Ok(GgufValue::U8(cur.read_u8()?)),
        GGUF_TYPE_INT8 => Ok(GgufValue::I8(cur.read_i8()?)),
        GGUF_TYPE_UINT16 => Ok(GgufValue::U16(cur.read_u16()?)),
        GGUF_TYPE_INT16 => Ok(GgufValue::I16(cur.read_i16()?)),
        GGUF_TYPE_UINT32 => Ok(GgufValue::U32(cur.read_u32()?)),
        GGUF_TYPE_INT32 => Ok(GgufValue::I32(cur.read_i32()?)),
        GGUF_TYPE_FLOAT32 => Ok(GgufValue::F32(cur.read_f32()?)),
        GGUF_TYPE_BOOL => Ok(GgufValue::Bool(cur.read_bool()?)),
        GGUF_TYPE_STRING => Ok(GgufValue::String(cur.read_string()?)),
        GGUF_TYPE_ARRAY => {
            let elem_type = cur.read_u32()?;
            let count = cur.read_u64()? as usize;
            let mut elems = Vec::with_capacity(count.min(65536));
            for _ in 0..count {
                elems.push(read_value(cur, elem_type)?);
            }
            Ok(GgufValue::Array(elems))
        }
        GGUF_TYPE_UINT64 => Ok(GgufValue::U64(cur.read_u64()?)),
        GGUF_TYPE_INT64 => Ok(GgufValue::I64(cur.read_i64()?)),
        GGUF_TYPE_FLOAT64 => Ok(GgufValue::F64(cur.read_f64()?)),
        _ => Err(ParseError::UnknownValueType(type_tag)),
    }
}

// ── Tensor dtype ───────────────────────────────────────────────────────

/// GGUF/GGML quantization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgufDtype {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ1S,
    IQ4NL,
    IQ3S,
    IQ2S,
    IQ4XS,
    /// **I2_S** — Microsoft BitNet b1.58's ternary format (GGUF tag 36).
    /// Per-tensor layout (empirically verified against
    /// `bitnet-b1.58-2b-4t-i2s.gguf` — corrects a wrong secondary spec):
    /// `ceil(N/4)` bytes of 2-bit codes in 128-element **interleaved**
    /// blocks (byte `gp` of block holds positions `gp, 32+gp, 64+gp,
    /// 96+gp` at bits `[7:6, 5:4, 3:2, 1:0]`), codes `00→0, 01→+1,
    /// 10→−1`; followed by a **single 4-byte f32 scale** (BitNet
    /// absmean). 2.0 bpw asymptotically + 4 B/tensor scale. The
    /// up-to-28 B inter-tensor padding (GGUF `general.alignment=32`)
    /// is NOT part of this tensor's data.
    I2S,
    /// **Q1_0** — PrismML/Bonsai's 1-bit format (GGUF tag 41). Per-block
    /// layout (authoritative, from PrismML-Eng/llama.cpp `block_q1_0` +
    /// `dequantize_row_q1_0`): 128-weight block = `ggml_half d` (2-byte
    /// f16 leading scale, `d = sum(|x|) / 128`) + 16 bytes `qs` (1 bit
    /// per weight, weight `j` at byte `j/8` bit `j%8`); bit value
    /// **0 → −d, 1 → +d** (binary, no zero). 18 bytes / 128 weights =
    /// 1.125 bpw. Used by `prism-ml/Bonsai-{1.7B,4B,8B}-gguf`.
    Q1_0,
    /// **Q2_0** — PrismML/Ternary-Bonsai's 2-bit format (GGUF tag 42).
    /// Per-block layout (authoritative, from PrismML-Eng/llama.cpp
    /// `block_q2_0` + `dequantize_row_q2_0`): 128-weight block =
    /// `ggml_half d` (2-byte f16 leading scale, `d = max(|x|)`) + 32
    /// bytes `qs` (2 bits per weight, weight `j` at byte `j/4` bits
    /// `(j%4)*2`); quaternary value mapping **00 → −d, 01 → 0, 10 → +d,
    /// 11 → +2d** (i.e. `y = ((int)q - 1) * d`). 34 bytes / 128 weights
    /// = 2.125 bpw. Used by `prism-ml/Ternary-Bonsai-{1.7B,4B,8B}-gguf`.
    /// **Note**: tag 42 is also squatted on by Tencent Hunyuan's 1.25-bit
    /// format with a structurally different codec — dispatch on
    /// `general.architecture` (or producer signature), not the tag alone.
    Q2_0,
    /// **STQ1_0** — Tencent "Sherry" 1.25-bit ternary (also GGUF tag 42,
    /// colliding with [`GgufDtype::Q2_0`]; resolved by
    /// `general.architecture == "hunyuan-dense"` in [`GgufFile::parse`]).
    /// Authoritative layout (`block_stq1_0` in sjl623/llama.cpp `STQ_0`,
    /// `ggml-common.h:281` + `dequantize_row_stq1_0`): 256-weight
    /// super-block = `uint8_t qs[32]` (4-bit slot code per group of 4) +
    /// `uint8_t sign[8]` (1-bit table select per group, LSB-first) +
    /// `ggml_half d` (2-byte f16 scale) = **42 B / 256 weights = 1.3125
    /// bpw**. Each group of 4 weights has **exactly one zero and three
    /// ±d** (3:4 fine-grained sparsity); `(sign<<4)|slot` indexes a
    /// 32-entry codebook → packed 4×2-bit pattern; per-lane `(q-1)·d`
    /// with `q ∈ {0,1,2}` → `{−d, 0, +d}`. Used by
    /// `tencent/Hy-MT1.5-1.8B-1.25bit`.
    Stq1_0,
    I8,
    I16,
    I32,
    I64,
    F64,
    BF16,
    Unknown(u32),
}

impl GgufDtype {
    /// Parse from the GGUF integer type tag.
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => GgufDtype::F32,
            1 => GgufDtype::F16,
            2 => GgufDtype::Q4_0,
            3 => GgufDtype::Q4_1,
            6 => GgufDtype::Q5_0,
            7 => GgufDtype::Q5_1,
            8 => GgufDtype::Q8_0,
            9 => GgufDtype::Q8_1,
            10 => GgufDtype::Q2K,
            11 => GgufDtype::Q3K,
            12 => GgufDtype::Q4K,
            13 => GgufDtype::Q5K,
            14 => GgufDtype::Q6K,
            16 => GgufDtype::IQ2XXS,
            17 => GgufDtype::IQ2XS,
            18 => GgufDtype::IQ3XXS,
            19 => GgufDtype::IQ1S,
            20 => GgufDtype::IQ4NL,
            21 => GgufDtype::IQ3S,
            22 => GgufDtype::IQ2S,
            23 => GgufDtype::IQ4XS,
            24 => GgufDtype::I8,
            25 => GgufDtype::I16,
            26 => GgufDtype::I32,
            27 => GgufDtype::I64,
            28 => GgufDtype::F64,
            30 => GgufDtype::BF16,
            36 => GgufDtype::I2S, // BitNet b1.58
            41 => GgufDtype::Q1_0, // PrismML Bonsai
            42 => GgufDtype::Q2_0, // PrismML Ternary-Bonsai (NB: tag 42 also collides with Tencent Hunyuan's 1.25-bit codec — see GgufDtype::Q2_0 docs)
            _ => GgufDtype::Unknown(v),
        }
    }

    /// Return the raw GGUF integer tag.
    pub fn to_u32(self) -> u32 {
        match self {
            GgufDtype::F32 => 0,
            GgufDtype::F16 => 1,
            GgufDtype::Q4_0 => 2,
            GgufDtype::Q4_1 => 3,
            GgufDtype::Q5_0 => 6,
            GgufDtype::Q5_1 => 7,
            GgufDtype::Q8_0 => 8,
            GgufDtype::Q8_1 => 9,
            GgufDtype::Q2K => 10,
            GgufDtype::Q3K => 11,
            GgufDtype::Q4K => 12,
            GgufDtype::Q5K => 13,
            GgufDtype::Q6K => 14,
            GgufDtype::IQ2XXS => 16,
            GgufDtype::IQ2XS => 17,
            GgufDtype::IQ3XXS => 18,
            GgufDtype::IQ1S => 19,
            GgufDtype::IQ4NL => 20,
            GgufDtype::IQ3S => 21,
            GgufDtype::IQ2S => 22,
            GgufDtype::IQ4XS => 23,
            GgufDtype::I8 => 24,
            GgufDtype::I16 => 25,
            GgufDtype::I32 => 26,
            GgufDtype::I64 => 27,
            GgufDtype::F64 => 28,
            GgufDtype::BF16 => 30,
            GgufDtype::I2S => 36,
            GgufDtype::Q1_0 => 41,
            GgufDtype::Q2_0 => 42,
            GgufDtype::Stq1_0 => 42, // shares tag 42 with Q2_0 (collision; see docs)
            GgufDtype::Unknown(v) => v,
        }
    }
}

impl fmt::Display for GgufDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", crate::quant::quant_info(*self).rce_type)
    }
}

// ── Tensor info ────────────────────────────────────────────────────────

/// A tensor descriptor from the GGUF file (no data, just layout).
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name, e.g. `blk.0.attn_q.weight`.
    pub name: String,
    /// Shape dimensions (row-major).
    pub dims: Vec<u64>,
    /// Quantization / data type.
    pub dtype: GgufDtype,
    /// Byte offset into the data blob section.
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Total number of elements.
    pub fn numel(&self) -> u64 {
        self.dims.iter().copied().product()
    }
}

// ── Metadata entry ─────────────────────────────────────────────────────

/// A single metadata key-value pair.
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub key: String,
    pub value: GgufValue,
}

// ── Top-level file ─────────────────────────────────────────────────────

/// Parsed GGUF file (metadata + tensor layout, no weight data).
#[derive(Debug, Clone)]
pub struct GgufFile {
    /// GGUF format version (2 or 3).
    pub version: u32,
    /// Metadata key-value pairs.
    pub metadata: Vec<GgufMetadata>,
    /// Tensor descriptors.
    pub tensors: Vec<GgufTensorInfo>,
    /// Absolute file offset where the tensor data section begins.
    /// Each `GgufTensorInfo::offset` is RELATIVE to this position; the
    /// absolute byte range for tensor `t` is
    /// `data_offset + t.offset .. data_offset + t.offset + byte_size(t)`.
    /// The data section starts after the header + metadata + tensor
    /// info, padded to `general.alignment` (default 32 bytes).
    pub data_offset: u64,
    /// Tensor-data alignment in bytes (32 by default; overridden by
    /// `general.alignment` metadata if present).
    pub alignment: u64,
}

impl GgufFile {
    /// Parse a GGUF file from a byte slice.
    ///
    /// Only reads the header, metadata, and tensor info sections.
    /// The tensor data blob is not read.
    pub fn parse(data: &[u8]) -> Result<Self, ParseError> {
        let mut cur = Cursor::new(data);

        // ── Magic ──────────────────────────────────────────────────
        let magic_bytes = cur.read_bytes(4)?;
        let magic = u32::from_le_bytes([magic_bytes[0], magic_bytes[1], magic_bytes[2], magic_bytes[3]]);
        if magic_bytes != GGUF_MAGIC {
            return Err(ParseError::InvalidMagic(magic));
        }

        // ── Version ────────────────────────────────────────────────
        let version = cur.read_u32()?;
        if version < MIN_VERSION || version > MAX_VERSION {
            return Err(ParseError::UnsupportedVersion(version));
        }

        // ── Counts ─────────────────────────────────────────────────
        let tensor_count = cur.read_u64()? as usize;
        let metadata_kv_count = cur.read_u64()? as usize;

        // ── Metadata ───────────────────────────────────────────────
        let mut metadata = Vec::with_capacity(metadata_kv_count.min(4096));
        for _ in 0..metadata_kv_count {
            let key = cur.read_string()?;
            let type_tag = cur.read_u32()?;
            let value = read_value(&mut cur, type_tag)?;
            metadata.push(GgufMetadata { key, value });
        }

        // ── Tensor info ────────────────────────────────────────────
        let mut tensors = Vec::with_capacity(tensor_count.min(65536));
        for _ in 0..tensor_count {
            let name = cur.read_string()?;
            let ndim = cur.read_u32()?;
            if ndim > 8 {
                return Err(ParseError::TooManyDimensions(ndim));
            }
            let mut dims = Vec::with_capacity(ndim as usize);
            for _ in 0..ndim {
                dims.push(cur.read_u64()?);
            }
            let dtype_raw = cur.read_u32()?;
            let dtype = GgufDtype::from_u32(dtype_raw);
            let offset = cur.read_u64()?;
            tensors.push(GgufTensorInfo {
                name,
                dims,
                dtype,
                offset,
            });
        }

        // ── Tensor-data section start ──────────────────────────
        //
        // The data section starts after the header + metadata + tensor
        // info, padded up to `general.alignment` (default 32). Each
        // tensor's `offset` is then relative to this position.
        let alignment = metadata
            .iter()
            .find(|kv| kv.key == "general.alignment")
            .and_then(|kv| match &kv.value {
                GgufValue::U32(v) => Some(*v as u64),
                GgufValue::U64(v) => Some(*v),
                _ => None,
            })
            .unwrap_or(32);
        let pos = cur.pos as u64;
        // Pad up to the next multiple of `alignment`.
        let data_offset = if pos % alignment == 0 {
            pos
        } else {
            pos + (alignment - (pos % alignment))
        };

        // ── Tag-42 collision resolution ────────────────────────────
        //
        // GGUF dtype tag 42 is squatted on by TWO unrelated producers
        // with structurally different codecs: PrismML's Q2_0 (qwen3,
        // 2.125 bpw, 128-wt blocks) and Tencent "Sherry"'s STQ1_0
        // (hunyuan-dense, 1.3125 bpw, 256-wt super-blocks). `from_u32`
        // can't see metadata, so it defaults tag 42 → Q2_0. The
        // producer's architecture is the authoritative discriminator
        // (the codec is producer-specific and the producer sets the
        // arch): for `hunyuan-dense`, rewrite every Q2_0 → Stq1_0.
        let is_hunyuan_dense = metadata.iter().any(|kv| {
            kv.key == "general.architecture"
                && matches!(&kv.value, GgufValue::String(s) if s == "hunyuan-dense")
        });
        if is_hunyuan_dense {
            for t in &mut tensors {
                if t.dtype == GgufDtype::Q2_0 {
                    t.dtype = GgufDtype::Stq1_0;
                }
            }
        }

        Ok(GgufFile {
            version,
            metadata,
            tensors,
            data_offset,
            alignment,
        })
    }

    // ── Convenience accessors ──────────────────────────────────────

    /// Look up a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata
            .iter()
            .find(|m| m.key == key)
            .map(|m| &m.value)
    }

    /// Get string metadata value by key.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.get_metadata(key) {
            Some(GgufValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Get u32 metadata value by key.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.get_metadata(key) {
            Some(GgufValue::U32(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get u64 metadata value by key.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.get_metadata(key) {
            Some(GgufValue::U64(v)) => Some(*v),
            _ => None,
        }
    }

    /// Model architecture string (e.g., "llama", "gpt2", "falcon").
    pub fn architecture(&self) -> Option<&str> {
        self.get_string("general.architecture")
    }

    /// Model name.
    pub fn model_name(&self) -> Option<&str> {
        self.get_string("general.name")
    }

    /// Number of transformer blocks.
    pub fn block_count(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.get_u32(&format!("{arch}.block_count"))
    }

    /// Embedding dimension.
    pub fn embedding_length(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.get_u32(&format!("{arch}.embedding_length"))
    }

    /// Attention head count.
    pub fn head_count(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.get_u32(&format!("{arch}.attention.head_count"))
    }

    /// Key-value head count (for GQA).
    pub fn head_count_kv(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.get_u32(&format!("{arch}.attention.head_count_kv"))
    }
}

// ── Test helpers ───────────────────────────────────────────────────────

/// Build synthetic GGUF v3 bytes for testing.
///
/// Public so downstream tests can also use it.
pub fn build_test_gguf(
    metadata: &[(&str, u32, &[u8])],
    tensors: &[(&str, &[u64], u32, u64)],
) -> Vec<u8> {
    let mut buf = Vec::new();
    // Magic
    buf.extend_from_slice(&GGUF_MAGIC);
    // Version = 3
    buf.extend_from_slice(&3u32.to_le_bytes());
    // Tensor count
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    // Metadata KV count
    buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    // Metadata entries
    for (key, type_tag, value_bytes) in metadata {
        // key string: u64 len + bytes
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        // type tag
        buf.extend_from_slice(&type_tag.to_le_bytes());
        // value bytes (pre-encoded)
        buf.extend_from_slice(value_bytes);
    }

    // Tensor info entries
    for (name, dims, dtype, offset) in tensors {
        // name string
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        // ndim
        buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        // dims
        for d in *dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        // dtype
        buf.extend_from_slice(&dtype.to_le_bytes());
        // offset
        buf.extend_from_slice(&offset.to_le_bytes());
    }

    buf
}

/// The bytes of a whole GGUF v3 file: the header, metadata and tensor
/// infos of [`build_test_gguf`], then the tensor data section at the
/// default alignment of 32 bytes, each tensor's bytes at an offset aligned
/// to 32. `tensors` holds name, dims (ne0 first), dtype tag and the
/// tensor's bytes in that dtype's layout.
pub fn build_gguf(metadata: &[(&str, u32, &[u8])], tensors: &[(&str, &[u64], u32, &[u8])]) -> Vec<u8> {
    const ALIGN: usize = 32;
    let up = |n: usize| n.div_ceil(ALIGN) * ALIGN;
    let mut offs = Vec::with_capacity(tensors.len());
    let mut off = 0usize;
    for (_, _, _, data) in tensors {
        offs.push(off as u64);
        off = up(off + data.len());
    }
    let infos: Vec<(&str, &[u64], u32, u64)> =
        tensors.iter().zip(&offs).map(|((n, d, t, _), o)| (*n, *d, *t, *o)).collect();
    let mut buf = build_test_gguf(metadata, &infos);
    let start = up(buf.len());
    buf.resize(start, 0);
    for ((_, _, _, data), o) in tensors.iter().zip(&offs) {
        buf.resize(start + *o as usize, 0);
        buf.extend_from_slice(data);
    }
    buf
}

/// Encode a GGUF f32 value (type tag 6) as raw bytes for `build_test_gguf`.
pub fn encode_f32_value(v: f32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

/// Encode a GGUF string value (type tag 8) as raw bytes for `build_test_gguf`.
pub fn encode_string_value(s: &str) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
    buf
}

/// Encode a GGUF u32 value (type tag 4) as raw bytes for `build_test_gguf`.
pub fn encode_u32_value(v: u32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

/// Encode a GGUF array value header for `build_test_gguf`.
pub fn encode_array_u32_value(values: &[u32]) -> Vec<u8> {
    let mut buf = Vec::new();
    // element type tag (u32 = 4)
    buf.extend_from_slice(&4u32.to_le_bytes());
    // count
    buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_gguf() -> Vec<u8> {
        build_test_gguf(&[], &[])
    }

    #[test]
    fn parse_minimal_header() {
        let data = minimal_gguf();
        let file = GgufFile::parse(&data).unwrap();
        assert_eq!(file.version, 3);
        assert!(file.metadata.is_empty());
        assert!(file.tensors.is_empty());
    }

    #[test]
    fn reject_invalid_magic() {
        let mut data = minimal_gguf();
        data[0] = 0x00;
        let err = GgufFile::parse(&data).unwrap_err();
        assert!(matches!(err, ParseError::InvalidMagic(_)));
    }

    #[test]
    fn reject_unsupported_version() {
        let mut data = minimal_gguf();
        // Set version to 99
        data[4..8].copy_from_slice(&99u32.to_le_bytes());
        let err = GgufFile::parse(&data).unwrap_err();
        assert!(matches!(err, ParseError::UnsupportedVersion(99)));
    }

    #[test]
    fn parse_string_metadata() {
        let val_bytes = encode_string_value("llama");
        let data = build_test_gguf(
            &[("general.architecture", GGUF_TYPE_STRING, &val_bytes)],
            &[],
        );
        let file = GgufFile::parse(&data).unwrap();
        assert_eq!(file.metadata.len(), 1);
        assert_eq!(file.metadata[0].key, "general.architecture");
        assert_eq!(file.metadata[0].value, GgufValue::String("llama".into()));
        assert_eq!(file.architecture(), Some("llama"));
    }

    #[test]
    fn parse_u32_metadata() {
        let val_bytes = encode_u32_value(32);
        let data = build_test_gguf(
            &[("llama.block_count", GGUF_TYPE_UINT32, &val_bytes)],
            &[],
        );
        let file = GgufFile::parse(&data).unwrap();
        assert_eq!(file.metadata[0].value, GgufValue::U32(32));
    }

    #[test]
    fn parse_array_metadata() {
        let val_bytes = encode_array_u32_value(&[100, 200, 300]);
        let data = build_test_gguf(
            &[("test.array", GGUF_TYPE_ARRAY, &val_bytes)],
            &[],
        );
        let file = GgufFile::parse(&data).unwrap();
        assert_eq!(
            file.metadata[0].value,
            GgufValue::Array(vec![GgufValue::U32(100), GgufValue::U32(200), GgufValue::U32(300)])
        );
    }

    #[test]
    fn parse_tensor_info() {
        let data = build_test_gguf(
            &[],
            &[("blk.0.attn_q.weight", &[4096, 4096], 12, 0)],
        );
        let file = GgufFile::parse(&data).unwrap();
        assert_eq!(file.tensors.len(), 1);
        let t = &file.tensors[0];
        assert_eq!(t.name, "blk.0.attn_q.weight");
        assert_eq!(t.dims, vec![4096, 4096]);
        assert_eq!(t.dtype, GgufDtype::Q4K);
        assert_eq!(t.offset, 0);
        assert_eq!(t.numel(), 4096 * 4096);
    }

    #[test]
    fn parse_multiple_tensors() {
        let data = build_test_gguf(
            &[],
            &[
                ("token_embd.weight", &[32000, 4096], 1, 0),
                ("blk.0.attn_q.weight", &[4096, 4096], 12, 65536),
                ("blk.0.ffn_gate.weight", &[11008, 4096], 14, 131072),
            ],
        );
        let file = GgufFile::parse(&data).unwrap();
        assert_eq!(file.tensors.len(), 3);
        assert_eq!(file.tensors[0].dtype, GgufDtype::F16);
        assert_eq!(file.tensors[1].dtype, GgufDtype::Q4K);
        assert_eq!(file.tensors[2].dtype, GgufDtype::Q6K);
    }

    #[test]
    fn unexpected_eof() {
        let data = &[0x47, 0x47, 0x55, 0x46]; // magic only, no version
        let err = GgufFile::parse(data).unwrap_err();
        assert!(matches!(err, ParseError::UnexpectedEof(_)));
    }

    #[test]
    fn dtype_roundtrip() {
        for tag in [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 36, 41, 42] {
            let dtype = GgufDtype::from_u32(tag);
            assert_eq!(dtype.to_u32(), tag);
        }
    }

    #[test]
    fn unknown_dtype() {
        let dtype = GgufDtype::from_u32(999);
        assert_eq!(dtype, GgufDtype::Unknown(999));
        assert_eq!(dtype.to_u32(), 999);
    }

    #[test]
    fn parse_full_model_header() {
        // Simulate a realistic llama-7B header
        let arch_val = encode_string_value("llama");
        let name_val = encode_string_value("LLaMA-7B-Q4_K_M");
        let blocks_val = encode_u32_value(32);
        let embed_val = encode_u32_value(4096);
        let heads_val = encode_u32_value(32);

        let mut tensor_descs: Vec<(&str, &[u64], u32, u64)> = Vec::new();
        tensor_descs.push(("token_embd.weight", &[32000, 4096][..], 1, 0));
        tensor_descs.push(("output_norm.weight", &[4096][..], 0, 0));
        tensor_descs.push(("output.weight", &[32000, 4096][..], 12, 0));
        for blk in 0..2 {
            let names: Vec<String> = [
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
                "attn_norm.weight",
                "ffn_norm.weight",
            ]
            .iter()
            .map(|s| format!("blk.{blk}.{s}"))
            .collect();
            // Leak names into static to use as &str in tensor_descs
            let names: Vec<&'static str> = names.into_iter().map(|s| &*s.leak()).collect();
            tensor_descs.push((names[0], &[4096, 4096][..], 12, 0));
            tensor_descs.push((names[1], &[4096, 4096][..], 12, 0));
            tensor_descs.push((names[2], &[4096, 4096][..], 12, 0));
            tensor_descs.push((names[3], &[4096, 4096][..], 12, 0));
            tensor_descs.push((names[4], &[11008, 4096][..], 12, 0));
            tensor_descs.push((names[5], &[11008, 4096][..], 12, 0));
            tensor_descs.push((names[6], &[4096, 11008][..], 12, 0));
            tensor_descs.push((names[7], &[4096][..], 0, 0));
            tensor_descs.push((names[8], &[4096][..], 0, 0));
        }

        let data = build_test_gguf(
            &[
                ("general.architecture", GGUF_TYPE_STRING, &arch_val),
                ("general.name", GGUF_TYPE_STRING, &name_val),
                ("llama.block_count", GGUF_TYPE_UINT32, &blocks_val),
                ("llama.embedding_length", GGUF_TYPE_UINT32, &embed_val),
                ("llama.attention.head_count", GGUF_TYPE_UINT32, &heads_val),
            ],
            &tensor_descs,
        );

        let file = GgufFile::parse(&data).unwrap();
        assert_eq!(file.architecture(), Some("llama"));
        assert_eq!(file.model_name(), Some("LLaMA-7B-Q4_K_M"));
        assert_eq!(file.block_count(), Some(32));
        assert_eq!(file.embedding_length(), Some(4096));
        assert_eq!(file.head_count(), Some(32));
        assert_eq!(file.tensors.len(), 3 + 2 * 9); // token_embd + output_norm + output + 2 blocks * 9 tensors
    }

    #[test]
    fn too_many_dimensions() {
        // Build a tensor with ndim = 9 (max is 8)
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        buf.extend_from_slice(&0u64.to_le_bytes()); // 0 metadata
        // tensor name
        let name = "bad";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        // ndim = 9
        buf.extend_from_slice(&9u32.to_le_bytes());
        // would need 9 u64 dims but we expect the error before reading them

        let err = GgufFile::parse(&buf).unwrap_err();
        assert!(matches!(err, ParseError::TooManyDimensions(9)));
    }

    #[test]
    fn version_2_accepted() {
        let mut data = minimal_gguf();
        data[4..8].copy_from_slice(&2u32.to_le_bytes());
        let file = GgufFile::parse(&data).unwrap();
        assert_eq!(file.version, 2);
    }

    #[test]
    fn metadata_convenience_accessors() {
        let arch_val = encode_string_value("mistral");
        let blocks_val = encode_u32_value(32);
        let embed_val = encode_u32_value(4096);
        let heads_val = encode_u32_value(32);
        let heads_kv_val = encode_u32_value(8);

        let data = build_test_gguf(
            &[
                ("general.architecture", GGUF_TYPE_STRING, &arch_val),
                ("mistral.block_count", GGUF_TYPE_UINT32, &blocks_val),
                ("mistral.embedding_length", GGUF_TYPE_UINT32, &embed_val),
                ("mistral.attention.head_count", GGUF_TYPE_UINT32, &heads_val),
                ("mistral.attention.head_count_kv", GGUF_TYPE_UINT32, &heads_kv_val),
            ],
            &[],
        );
        let file = GgufFile::parse(&data).unwrap();
        assert_eq!(file.architecture(), Some("mistral"));
        assert_eq!(file.block_count(), Some(32));
        assert_eq!(file.embedding_length(), Some(4096));
        assert_eq!(file.head_count(), Some(32));
        assert_eq!(file.head_count_kv(), Some(8));
    }
}
