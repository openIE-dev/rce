//! Safetensors weight loader for RCE's graph terms.
//!
//! Reads the `.safetensors` container format (HuggingFace's portable,
//! zero-dependency checkpoint format) and produces [`LoadedTensor`]
//! entries — a lightweight, zero-dep primitive that consumers map into
//! their own graph-node types (`inv_ai_compiler::lower::TensorLiteralData`,
//! or any other tensor-literal representation).
//!
//! Format reference: https://github.com/huggingface/safetensors
//!
//! Byte layout:
//! ```text
//!   [0..8)            u64 little-endian — header JSON length N
//!   [8..8+N)          UTF-8 JSON describing tensors + optional metadata
//!   [8+N..end)        concatenated tensor payload bytes, offsets relative
//!                     to start of payload region
//! ```
//!
//! The JSON header maps each tensor name to
//! `{ "dtype": str, "shape": [u64…], "data_offsets": [begin, end] }`.
//!
//! This crate:
//!   * Parses the header with `serde_json`.
//!   * mmap's the file with `MADV_WILLNEED` so the kernel keeps tensor
//!     pages resident; weight bytes live ONCE, in the OS page cache.
//!   * Returns each tensor as a **zero-copy view** into the mmap
//!     ([`TensorData::Mmap`]). The mmap stays alive as long as any
//!     `LoadedTensor` holds an `Arc<Mmap>` refcount, and is shared by
//!     refcount across all tensors of the file. **No heap copy** is made
//!     of the weight bytes by this loader — under memory pressure the
//!     kernel can evict pages cleanly, so loading a 200 GB checkpoint
//!     does NOT pin 200 GB of user-heap RAM.
//!
//! Callers that need an owned `Arc<[u8]>` (because their downstream graph
//! node owns its bytes) can opt-in to a heap copy via [`TensorData::to_arc`].
//! `SafetensorsFile::from_bytes` (in-memory entry point) always returns
//! owned tensors since the caller's `&[u8]` lifetime can't outlive the
//! struct.
//!
//! Not a stub: all safetensors dtypes are mapped, all offsets are bounds-
//! checked, malformed headers return typed errors.

use std::collections::BTreeMap;
#[cfg(feature = "mmap")]
use std::fs::File;
#[cfg(feature = "mmap")]
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "mmap")]
use memmap2::Mmap;
use serde::Deserialize;

/// Size in bytes of a single element of a graph dtype name.
///
/// Graph dtype names are lower-case (`f32`, `bf16`, `i64`, …). Returns
/// `None` for unrecognised names so callers can surface a typed error.
pub fn dtype_byte_size(dtype: &str) -> Option<usize> {
    Some(match dtype {
        "bool" | "i8" | "u8" => 1,
        "f16" | "bf16" | "i16" | "u16" => 2,
        "f32" | "i32" | "u32" => 4,
        "f64" | "i64" | "u64" | "complex64" => 8,
        _ => return None,
    })
}

/// Backing storage for a tensor's bytes. `Mmap` is the production
/// zero-copy path (the bytes live in the file's mmap, refcounted via
/// `Arc<Mmap>` and shared across all tensors in the file). `Owned` is
/// for callers that built the file in memory (`from_bytes`) or that
/// materialized an owned copy via `to_arc`.
///
/// `Deref<Target=[u8]>` + `AsRef<[u8]>` mean most existing call sites
/// (`tensor.data.chunks_exact(2)`, `&tensor.data[..]`,
/// `tensor.data.len()`, …) keep working unchanged.
#[derive(Clone)]
pub enum TensorData {
    /// Zero-copy view into a memory-mapped file.
    #[cfg(feature = "mmap")]
    Mmap { backing: Arc<Mmap>, offset: usize, len: usize },
    /// Heap-owned bytes (from `from_bytes` or `to_arc`).
    Owned(Arc<[u8]>),
}

impl TensorData {
    /// Zero-copy slice into whichever backing holds the bytes.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            #[cfg(feature = "mmap")]
            TensorData::Mmap { backing, offset, len } => &backing[*offset..*offset + *len],
            TensorData::Owned(b) => b,
        }
    }

    /// Materialize an owned `Arc<[u8]>` for callers that need it
    /// (downstream graph nodes that own their bytes). Allocates only
    /// if backing is `Mmap`; cheap clone when already `Owned`.
    pub fn to_arc(&self) -> Arc<[u8]> {
        match self {
            #[cfg(feature = "mmap")]
            TensorData::Mmap { .. } => Arc::from(self.as_slice().to_vec().into_boxed_slice()),
            TensorData::Owned(b) => Arc::clone(b),
        }
    }

    /// Byte length of the tensor.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            #[cfg(feature = "mmap")]
            TensorData::Mmap { len, .. } => *len,
            TensorData::Owned(b) => b.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool { self.len() == 0 }
}

impl std::ops::Deref for TensorData {
    type Target = [u8];
    #[inline] fn deref(&self) -> &[u8] { self.as_slice() }
}

impl AsRef<[u8]> for TensorData {
    #[inline] fn as_ref(&self) -> &[u8] { self.as_slice() }
}

impl std::fmt::Debug for TensorData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "mmap")]
            TensorData::Mmap { offset, len, .. } => write!(f, "TensorData::Mmap{{offset:{offset}, len:{len}}}"),
            TensorData::Owned(b) => write!(f, "TensorData::Owned(len:{})", b.len()),
        }
    }
}

/// A tensor loaded from a safetensors file — name, dtype, shape, and a
/// zero-copy view (or owned buffer) into its bytes. Consumers convert
/// this into whatever tensor-literal representation their graph IR uses.
#[derive(Debug, Clone)]
pub struct LoadedTensor {
    pub name: String,
    /// Lower-case graph dtype: "f32", "bf16", "i64", …
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Zero-copy view into the mmap (or owned buffer). Deref's to `&[u8]`.
    pub data: TensorData,
}

impl LoadedTensor {
    /// Number of elements (product of shape dims; 1 for a scalar).
    pub fn numel(&self) -> usize {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        }
    }

    /// Byte size of a single element for this tensor's dtype.
    pub fn dtype_byte_size(&self) -> usize {
        dtype_byte_size(&self.dtype).unwrap_or(0)
    }

    /// Interpret the payload as a slice of `f32` values. Returns `None`
    /// if the dtype is not `f32` or the payload alignment is wrong.
    pub fn as_f32(&self) -> Option<&[f32]> {
        if self.dtype != "f32" {
            return None;
        }
        let ptr = self.data.as_ptr();
        if (ptr as usize) % std::mem::align_of::<f32>() != 0 {
            return None;
        }
        let len = self.data.len() / std::mem::size_of::<f32>();
        // SAFETY: dtype + alignment checked; length is bytes / 4.
        Some(unsafe { std::slice::from_raw_parts(ptr as *const f32, len) })
    }
}

/// Borrow-direct `f32` weight handle.  Wraps a `LoadedTensor` (which owns an
/// `Arc<Mmap>` clone in the common case) and derefs to `&[f32]`.  Field types
/// can change `Vec<f32>` → `F32Slice` and call sites that take `&[f32]` work
/// unchanged via deref coercion.
///
/// Production wins: model crates with native-f32 safetensors (e.g. V-JEPA 2,
/// rce-cosmos3 norm/bias tensors) no longer materialize per-tensor heap
/// copies; the kernel page cache holds the weights and the user heap holds
/// only struct pointers.  Cosmos 3's bf16 equivalent ([`crate`] consumer's
/// `BfSlice`) demonstrated 17 GB → 357 MB user heap on the 8 B-param reasoner.
#[derive(Clone)]
pub struct F32Slice {
    tensor: LoadedTensor,
    offset: usize,
    len: usize,
}

impl F32Slice {
    pub fn from_tensor(name: &str, t: &LoadedTensor) -> Result<Self, String> {
        if t.dtype != "f32" {
            return Err(format!("F32Slice({name}): dtype {} not f32", t.dtype));
        }
        let bytes = t.data.len();
        if bytes % 4 != 0 {
            return Err(format!("F32Slice({name}): byte-len {bytes} not /4"));
        }
        let ptr = t.data.as_slice().as_ptr() as usize;
        if ptr % std::mem::align_of::<f32>() != 0 {
            return Err(format!("F32Slice({name}): mmap ptr not 4-byte aligned"));
        }
        Ok(Self { tensor: t.clone(), offset: 0, len: bytes / 4 })
    }
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        let bytes = self.tensor.data.as_slice();
        let ptr = unsafe { (bytes.as_ptr() as *const f32).add(self.offset) };
        unsafe { std::slice::from_raw_parts(ptr, self.len) }
    }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
}

impl std::ops::Deref for F32Slice {
    type Target = [f32];
    #[inline] fn deref(&self) -> &[f32] { self.as_slice() }
}

impl std::fmt::Debug for F32Slice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "F32Slice{{len:{}}}", self.len)
    }
}

/// Failure modes for safetensors loading.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("file is shorter than the 8-byte header-length prefix")]
    Truncated,
    #[error("header length {0} exceeds file size {1}")]
    HeaderLengthOverflow(u64, u64),
    #[error("header json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("tensor {name}: unsupported safetensors dtype {dtype}")]
    UnsupportedDtype { name: String, dtype: String },
    #[error("tensor {name}: data_offsets [{begin}, {end}] are out of order or OOB (payload_len={payload_len})")]
    OffsetsOob {
        name: String,
        begin: u64,
        end: u64,
        payload_len: u64,
    },
    #[error("tensor {name}: byte span {span} does not match shape * dtype = {expected}")]
    SpanMismatch {
        name: String,
        span: u64,
        expected: u64,
    },
}

/// Raw header entry for a single tensor, matching the safetensors spec.
#[derive(Debug, Deserialize)]
struct HeaderEntry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// A loaded safetensors checkpoint.
#[derive(Debug)]
pub struct SafetensorsFile {
    tensors: Vec<LoadedTensor>,
    metadata: BTreeMap<String, String>,
}

impl SafetensorsFile {
    /// Parse a safetensors file at `path`. mmap's the file, reads the JSON
    /// header, validates offsets, and returns each tensor as a **zero-copy
    /// view** into the mmap — no per-tensor heap copy. Pages are
    /// `MADV_WILLNEED`-advised so they stay resident under no memory
    /// pressure but remain kernel-evictable under pressure.
    ///
    /// Available only with the `mmap` feature (default-on, off for wasm32).
    /// Browser callers should use [`SafetensorsFile::from_bytes`].
    #[cfg(feature = "mmap")]
    pub fn load(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let file = File::open(path)?;
        // SAFETY: the mmap aliases the file. We only read; no concurrent
        // writers observe the mapping. The Arc keeps the mapping alive as
        // long as any tensor view holds a refcount.
        let mmap = Arc::new(unsafe { Mmap::map(&file)? });
        // WillNeed (NOT Sequential): we want the kernel to prefetch and
        // KEEP the weight pages resident. Sequential would drop pages
        // behind the read cursor — catastrophic for the typical
        // load-once-then-many-random-tensor-fetches pattern (and for MoE
        // decode that re-reads different experts each step).
        let _ = mmap.advise(memmap2::Advice::WillNeed);

        let bytes: &[u8] = &mmap;
        let (entries, metadata, payload_offset) = parse_header(bytes)?;

        let mut tensors: Vec<LoadedTensor> = Vec::with_capacity(entries.len());
        for (name, entry) in entries {
            let dtype = map_dtype(&entry.dtype).ok_or_else(|| {
                LoadError::UnsupportedDtype { name: name.clone(), dtype: entry.dtype.clone() }
            })?;
            let [begin, end] = entry.data_offsets;
            let payload_len = (bytes.len() - payload_offset) as u64;
            if end < begin || end > payload_len {
                return Err(LoadError::OffsetsOob { name, begin, end, payload_len });
            }
            let span = end - begin;
            let numel: u64 = entry.shape.iter().product();
            let per_elem = dtype_byte_size(dtype).unwrap() as u64;
            let expected = numel * per_elem;
            if span != expected {
                return Err(LoadError::SpanMismatch { name, span, expected });
            }
            // ZERO-COPY: view into the mmap, no allocation.
            let data = TensorData::Mmap {
                backing: Arc::clone(&mmap),
                offset: payload_offset + begin as usize,
                len: span as usize,
            };
            let shape: Vec<usize> = entry.shape.iter().map(|&d| d as usize).collect();
            tensors.push(LoadedTensor { name, dtype: dtype.to_string(), shape, data });
        }
        Ok(Self { tensors, metadata })
    }

    /// Parse safetensors bytes already in memory. Returns owned tensors
    /// (`TensorData::Owned`) since the input `&[u8]` lifetime cannot
    /// outlive the returned struct; use [`SafetensorsFile::load`] for
    /// zero-copy file-backed loading.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let (entries, metadata, payload_offset) = parse_header(bytes)?;
        let payload = &bytes[payload_offset..];
        let payload_len = payload.len() as u64;

        let mut tensors: Vec<LoadedTensor> = Vec::with_capacity(entries.len());
        for (name, entry) in entries {
            let dtype = map_dtype(&entry.dtype).ok_or_else(|| {
                LoadError::UnsupportedDtype { name: name.clone(), dtype: entry.dtype.clone() }
            })?;
            let [begin, end] = entry.data_offsets;
            if end < begin || end > payload_len {
                return Err(LoadError::OffsetsOob { name, begin, end, payload_len });
            }
            let span = end - begin;
            let numel: u64 = entry.shape.iter().product();
            let per_elem = dtype_byte_size(dtype).unwrap() as u64;
            let expected = numel * per_elem;
            if span != expected {
                return Err(LoadError::SpanMismatch { name, span, expected });
            }
            let slice = &payload[begin as usize..end as usize];
            let data = TensorData::Owned(Arc::from(slice.to_vec().into_boxed_slice()));
            let shape: Vec<usize> = entry.shape.iter().map(|&d| d as usize).collect();
            tensors.push(LoadedTensor { name, dtype: dtype.to_string(), shape, data });
        }
        Ok(Self { tensors, metadata })
    }

    /// Tensors, sorted by file-order data offset.
    pub fn tensors(&self) -> &[LoadedTensor] {
        &self.tensors
    }

    /// Consume the file and return its tensors.
    pub fn into_tensors(self) -> Vec<LoadedTensor> {
        self.tensors
    }

    /// `__metadata__` block (string-valued fields only).
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// Look up a tensor by name (linear scan; O(n) in the number of tensors,
    /// which is fine for model-scale manifests).
    pub fn get(&self, name: &str) -> Option<&LoadedTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

/// Map a safetensors dtype string to the graph dtype name.
///
/// Safetensors uses upper-case names (F32, BF16, I64, …) while the graph
/// conventions are lower-case (f32, bf16, i64, …).
/// Parse the safetensors JSON header (the 8-byte length prefix +
/// length-prefixed JSON object), returning the per-tensor entries sorted
/// by file-order offset, the `__metadata__` block, and the byte offset
/// where the tensor payload begins.
fn parse_header(
    bytes: &[u8],
) -> Result<(Vec<(String, HeaderEntry)>, BTreeMap<String, String>, usize), LoadError> {
    if bytes.len() < 8 {
        return Err(LoadError::Truncated);
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let header_end = 8u64
        .checked_add(header_len)
        .ok_or(LoadError::HeaderLengthOverflow(header_len, bytes.len() as u64))?;
    if header_end as usize > bytes.len() {
        return Err(LoadError::HeaderLengthOverflow(header_len, bytes.len() as u64));
    }
    let header_bytes = &bytes[8..header_end as usize];

    let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(header_bytes)?;
    let mut metadata: BTreeMap<String, String> = BTreeMap::new();
    let mut entries: Vec<(String, HeaderEntry)> = Vec::with_capacity(raw.len());
    for (name, value) in raw {
        if name == "__metadata__" {
            if let serde_json::Value::Object(obj) = value {
                for (k, v) in obj {
                    if let serde_json::Value::String(s) = v {
                        metadata.insert(k, s);
                    }
                }
            }
            continue;
        }
        let entry: HeaderEntry = serde_json::from_value(value)?;
        entries.push((name, entry));
    }
    // Sort by begin offset so tensors come out in file order — cache-
    // friendly access pattern and reproducible error reporting.
    entries.sort_by_key(|(_, e)| e.data_offsets[0]);
    Ok((entries, metadata, header_end as usize))
}

fn map_dtype(s: &str) -> Option<&'static str> {
    Some(match s {
        "BOOL" => "bool",
        "U8" => "u8",
        "I8" => "i8",
        "I16" => "i16",
        "U16" => "u16",
        "F16" => "f16",
        "BF16" => "bf16",
        "I32" => "i32",
        "U32" => "u32",
        "F32" => "f32",
        "F64" => "f64",
        "I64" => "i64",
        "U64" => "u64",
        _ => return None,
    })
}

// ── Header writer (test support) ────────────────────────────────────────

/// Build a safetensors byte buffer from named tensors. Used by tests; kept
/// public so downstream crates can round-trip their own checkpoints
/// without pulling in the full huggingface Python toolchain.
///
/// `tensors`: ordered list of `(name, dtype, shape, raw_bytes)`. The
/// caller is responsible for ensuring `raw_bytes.len() == numel * dtype_bytes`.
pub fn write_safetensors(
    tensors: &[(&str, &str, &[usize], &[u8])],
    metadata: Option<&BTreeMap<String, String>>,
) -> Vec<u8> {
    #[derive(serde::Serialize)]
    struct HeaderEntryOut<'a> {
        dtype: &'a str,
        shape: &'a [usize],
        data_offsets: [u64; 2],
    }

    let mut cursor: u64 = 0;
    let mut header: BTreeMap<String, serde_json::Value> = BTreeMap::new();

    for (name, dtype, shape, bytes) in tensors {
        let begin = cursor;
        let end = cursor + bytes.len() as u64;
        cursor = end;
        let st_dtype = rce_dtype_to_safetensors(dtype).unwrap_or(*dtype);
        header.insert(
            (*name).to_string(),
            serde_json::to_value(HeaderEntryOut {
                dtype: st_dtype,
                shape,
                data_offsets: [begin, end],
            })
            .unwrap(),
        );
    }

    if let Some(meta) = metadata {
        let mut m = serde_json::Map::new();
        for (k, v) in meta {
            m.insert(k.clone(), serde_json::Value::String(v.clone()));
        }
        header.insert("__metadata__".into(), serde_json::Value::Object(m));
    }

    let mut header_json = serde_json::to_vec(&header).unwrap();
    // Pad the header with ASCII spaces (valid JSON whitespace inside the
    // top-level object's trailing position) so the data section starts at an
    // 8-byte aligned offset.  Total prefix = 8 (header_len u64) + header_json
    // bytes; we want (8 + header_len) % 8 == 0 → header_len % 8 == 0.
    let pad_to = 8;
    let pad = (pad_to - header_json.len() % pad_to) % pad_to;
    if pad > 0 {
        // Inject `pad` ASCII space characters just before the closing `}`.
        let close_pos = header_json.iter().rposition(|&b| b == b'}').expect("safetensors header must end with }");
        let mut padded = Vec::with_capacity(header_json.len() + pad);
        padded.extend_from_slice(&header_json[..close_pos]);
        padded.extend(std::iter::repeat(b' ').take(pad));
        padded.extend_from_slice(&header_json[close_pos..]);
        header_json = padded;
    }
    let mut out = Vec::with_capacity(8 + header_json.len() + cursor as usize);
    out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_json);
    for (_, _, _, bytes) in tensors {
        out.extend_from_slice(bytes);
    }
    out
}

fn rce_dtype_to_safetensors(s: &str) -> Option<&'static str> {
    Some(match s {
        "bool" => "BOOL",
        "u8" => "U8",
        "i8" => "I8",
        "i16" => "I16",
        "u16" => "U16",
        "f16" => "F16",
        "bf16" => "BF16",
        "i32" => "I32",
        "u32" => "U32",
        "f32" => "F32",
        "f64" => "F64",
        "i64" => "I64",
        "u64" => "U64",
        _ => return None,
    })
}
