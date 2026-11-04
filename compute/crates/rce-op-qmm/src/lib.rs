//! Quantized matmul kernels for the rce substrate.
//!
//! Reads GGUF Q-family / f16 / bf16 weights directly from raw bytes and
//! produces f32 outputs — **skipping the dequant-to-f32 materialisation**
//! that was consuming most of forward time as memory bandwidth.
//!
//! Inputs:
//! - `a`: f32 activation vector of length `k` (the input feature dim).
//! - `w`: [`QuantWeight`] holding `n × k` weight values in GGUF row-major
//!   layout (each row = one output channel = `k/BLOCK` quantized blocks).
//!
//! Output:
//! - `out`: f32 vector of length `n` (the output feature dim).
//!
//! **Layout note.** GGUF stores torch's `[N_out, K_in]` as flat row-major,
//! same order. Reading row `j` gives K elements (the j-th output column we
//! want for `out[j] = Σ a[i] · W[j, i]`). No transpose needed — the natural
//! GGUF layout *is* the matmul-friendly one. This simplifies runtimes that
//! currently transpose at load: we keep raw bytes verbatim.
//!
//! Expected memory-bandwidth reduction per format:
//! - F32: 1× (baseline)
//! - F16 / BF16: 2×
//! - Q8_0: 3.8× (34 B / 32 vals)
//! - Q4_0: 7.1× (18 B / 32 vals)
//! - Q4_K: 7.1× (144 B / 256 vals)
//! - Q5_K: 5.8× (176 B / 256 vals)
//! - Q6_K: 4.9× (210 B / 256 vals)

use rce_ingest_gguf::{
    dequantize_to_f32, quant_info, GgufDtype, GgufFile, GgufTensorInfo, tensor_byte_size,
};
use half::f16;
use memmap2::Mmap;
use rayon::prelude::*;
use std::sync::Arc;

// ── Public API ─────────────────────────────────────────────────────

/// A weight tensor kept in its **native GGUF dtype**, as a **zero-copy
/// view into the mmap'd file** — no heap copy, no f32 expansion. Holds
/// an `Arc<Mmap>` (cheap refcount; the mapping stays resident as long as
/// any weight references it) plus the tensor's `(offset, len)` byte
/// range and shape.
///
/// On Apple Silicon UMA this means the model lives **once** in unified
/// memory via the OS page cache, addressable by CPU/AMX/GPU with no
/// duplication and no host↔device transfer. Cold pages evict and
/// re-fault under OS control — so a 60 GB model needs ~0 extra heap,
/// not 60 GB. (The previous `Vec<u8>` design copied every tensor out of
/// the mmap then dropped it: page-cache + an equal-size heap dup.)
///
/// Layout: row-major `[n_rows, n_cols]` where `n_rows = N_out` and
/// `n_cols = K_in` (matches GGUF's reversed-dim convention).
/// How the weight bytes are backed. `Mmap` is the production zero-copy
/// path (page-cached file, UMA-resident, never duplicated). `Owned` is
/// for tests / future substrate-chosen relayouts that legitimately need
/// to own bytes — both share the same `bytes()` accessor so kernels and
/// the VM op don't care which.
#[derive(Clone)]
enum Backing {
    Mmap(Arc<Mmap>),
    Owned(Arc<[u8]>),
}

impl Backing {
    #[inline]
    fn as_slice(&self) -> &[u8] {
        match self {
            Backing::Mmap(m) => &m[..],
            Backing::Owned(b) => &b[..],
        }
    }
}

#[derive(Clone)]
pub struct QuantWeight {
    backing: Backing,
    offset: usize,
    len: usize,
    pub dtype: GgufDtype,
    pub n_rows: usize,
    pub n_cols: usize,
}

impl QuantWeight {
    /// Zero-copy slice of this tensor's raw bytes.
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        &self.backing.as_slice()[self.offset..self.offset + self.len]
    }

    /// Create a sub-view of an existing tensor's byte range. Used for
    /// 3D MoE expert tensors stored as `[n_expert, n_rows, n_cols]` —
    /// caller passes per-expert byte offsets+lengths so each expert
    /// can be addressed as a normal `[n_rows, n_cols]` QuantWeight
    /// without copying. Backing (mmap or owned) is shared via the
    /// internal Backing clone.
    pub fn sub_view(
        &self,
        byte_offset_rel: usize,
        byte_len: usize,
        n_rows: usize,
        n_cols: usize,
    ) -> Self {
        debug_assert!(
            byte_offset_rel + byte_len <= self.len,
            "sub_view oob: {}+{} > {}", byte_offset_rel, byte_len, self.len,
        );
        Self {
            backing: self.backing.clone(),
            offset: self.offset + byte_offset_rel,
            len: byte_len,
            dtype: self.dtype,
            n_rows,
            n_cols,
        }
    }

    /// Construct from owned bytes (tests, or a substrate-chosen relayout
    /// that must own its buffer). Production GGUF loading uses
    /// `from_gguf` which is zero-copy mmap-backed.
    pub fn from_owned_bytes(
        bytes: Vec<u8>,
        dtype: GgufDtype,
        n_rows: usize,
        n_cols: usize,
    ) -> Self {
        let len = bytes.len();
        Self {
            backing: Backing::Owned(bytes.into()),
            offset: 0,
            len,
            dtype,
            n_rows,
            n_cols,
        }
    }

    /// Dequantize **one row** (`n_cols` f32 values) on demand. For an
    /// embedding table `[vocab, d_model]` this materialises exactly the
    /// one token's `d_model` vector instead of the whole `vocab×d_model`
    /// table in f32 — e.g. Bamba drops ~2.1 GB of resident f32 that was
    /// only ever gathered one row at a time.
    ///
    /// Requires the row to start on a block boundary, i.e.
    /// `n_cols % block_size == 0`. True for every model we run
    /// (d_model is 2560/4096 = ×256 for K-quants, 3136 = ×32 for Q4_0,
    /// and f16/f32 have block_size 1).
    pub fn dequant_row(&self, row: usize) -> Vec<f32> {
        debug_assert!(row < self.n_rows, "row {row} ≥ n_rows {}", self.n_rows);
        let bs = quant_info(self.dtype).block_size as usize;
        assert!(
            self.n_cols % bs == 0,
            "dequant_row: n_cols {} not a multiple of block_size {bs} for {:?}",
            self.n_cols, self.dtype,
        );
        let total_blocks = (self.n_rows * self.n_cols) / bs;
        debug_assert_eq!(self.len % total_blocks, 0, "tensor len not block-aligned");
        let block_bytes = self.len / total_blocks;
        let blocks_per_row = self.n_cols / bs;
        let row_len = blocks_per_row * block_bytes;
        let start = self.offset + row * row_len;
        let row_bytes = &self.backing.as_slice()[start..start + row_len];
        dequantize_to_f32(row_bytes, self.dtype, self.n_cols)
            .unwrap_or_else(|| panic!("dequant_row: unsupported dtype {:?}", self.dtype))
    }

    /// View a tensor inside an mmap'd GGUF. Takes a shared `Arc<Mmap>`
    /// (clone is a refcount bump — the caller keeps the mapping alive
    /// transitively through every weight). Caller declares the logical
    /// `(n_rows, n_cols)`; must equal `tensor.numel()`.
    pub fn from_gguf(
        mmap: Arc<Mmap>,
        gguf: &GgufFile,
        name: &str,
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Self, String> {
        let tensor: &GgufTensorInfo = gguf
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("tensor not found: {name}"))?;
        if tensor.numel() as usize != n_rows * n_cols {
            return Err(format!(
                "{name}: declared shape {n_rows}×{n_cols} = {} ≠ tensor numel = {}",
                n_rows * n_cols,
                tensor.numel()
            ));
        }
        let bytes_n = tensor_byte_size(tensor);
        let start = (gguf.data_offset + tensor.offset) as usize;
        let end = start + bytes_n;
        if end > mmap.len() {
            return Err(format!(
                "{name}: out of bounds [{start}, {end}) > file len {}",
                mmap.len()
            ));
        }
        Ok(Self {
            backing: Backing::Mmap(mmap),
            offset: start,
            len: bytes_n,
            dtype: tensor.dtype,
            n_rows,
            n_cols,
        })
    }
}

/// `out[1, n] = a[1, k] · w[n, k]^T`, i.e. one row of activations times
/// the full weight matrix. Dispatches on `w.dtype` to the right kernel.
/// All paths produce f32 output; `out` must have at least `n_rows`
/// elements.
pub fn mm_1d(a: &[f32], w: &QuantWeight, out: &mut [f32]) {
    let k = w.n_cols;
    let n = w.n_rows;
    debug_assert_eq!(a.len(), k, "activation len {} ≠ k {}", a.len(), k);
    debug_assert!(out.len() >= n, "output buffer too small");
    let wb = w.bytes();
    match w.dtype {
        GgufDtype::F32 => mm_1d_f32(a, wb, k, n, out),
        GgufDtype::F16 => mm_1d_f16(a, wb, k, n, out),
        GgufDtype::BF16 => mm_1d_bf16(a, wb, k, n, out),
        GgufDtype::Q4_0 => mm_1d_q4_0(a, wb, k, n, out),
        GgufDtype::Q4_1 => mm_1d_q4_1(a, wb, k, n, out),
        GgufDtype::Q5_0 => {
            #[cfg(target_arch = "aarch64")]
            mm_1d_q5_0_neon(a, wb, k, n, out);
            #[cfg(not(target_arch = "aarch64"))]
            mm_1d_q5_0(a, wb, k, n, out);
        }
        GgufDtype::Q8_0 => {
            #[cfg(target_arch = "aarch64")]
            mm_1d_q8_0_neon(a, wb, k, n, out);
            #[cfg(not(target_arch = "aarch64"))]
            mm_1d_q8_0(a, wb, k, n, out);
        }
        GgufDtype::Q3K => mm_1d_q3_k(a, wb, k, n, out),
        GgufDtype::Q4K => mm_1d_q4_k(a, wb, k, n, out),
        GgufDtype::Q5K => mm_1d_q5_k(a, wb, k, n, out),
        GgufDtype::I2S => {
            #[cfg(target_arch = "aarch64")]
            mm_1d_i2_s_neon(a, wb, k, n, out);
            #[cfg(not(target_arch = "aarch64"))]
            mm_1d_i2_s(a, wb, k, n, out);
        }
        GgufDtype::Q1_0 => {
            #[cfg(target_arch = "aarch64")]
            mm_1d_q1_0_neon(a, wb, k, n, out);
            #[cfg(not(target_arch = "aarch64"))]
            mm_1d_q1_0(a, wb, k, n, out);
        }
        GgufDtype::Q2_0 => {
            #[cfg(target_arch = "aarch64")]
            mm_1d_q2_0_neon(a, wb, k, n, out);
            #[cfg(not(target_arch = "aarch64"))]
            mm_1d_q2_0(a, wb, k, n, out);
        }
        GgufDtype::Stq1_0 => {
            #[cfg(target_arch = "aarch64")]
            mm_1d_stq1_0_neon(a, wb, k, n, out);
            #[cfg(not(target_arch = "aarch64"))]
            mm_1d_stq1_0(a, wb, k, n, out);
        }
        GgufDtype::Q6K => mm_1d_q6_k(a, wb, k, n, out),
        other => panic!("mm_1d: dtype {:?} not yet supported by rce-op-qmm", other),
    }
}

// ── Serial single-row matmul (no rayon) ────────────────────────────
//
// `mm_1d` parallelises over output rows. For MoE decode that's a loss:
// each expert matmul is tiny (n=512), so rayon fork/join overhead
// *cancels* the parallelism (measured: 32-core ≈ single-thread). The
// MoE layer instead parallelises across *experts* (coarse, ~9 big
// tasks) and calls this serial path inside each task — one fork/join
// per layer instead of ~27. Same arithmetic as `mm_1d`, just a plain
// row loop. Falls back to `mm_1d` for dtypes without a serial row-dot.
pub fn mm_1d_serial(a: &[f32], w: &QuantWeight, out: &mut [f32]) {
    let k = w.n_cols;
    let n = w.n_rows;
    debug_assert_eq!(a.len(), k);
    debug_assert!(out.len() >= n);
    let wb = w.bytes();
    macro_rules! rows {
        ($bb:expr, $dot:path) => {{
            let row_bytes = (k / blk_for(w.dtype)) * $bb;
            for j in 0..n {
                out[j] = $dot(a, &wb[j * row_bytes..(j + 1) * row_bytes], k);
            }
        }};
    }
    match w.dtype {
        GgufDtype::Q8_0 => rows!(34, dot_row_q8_0),
        GgufDtype::Q5_0 => rows!(22, dot_row_q5_0),
        GgufDtype::Q4K => rows!(144, dot_row_q4_k),
        GgufDtype::Q5K => rows!(176, dot_row_q5_k),
        GgufDtype::Q6K => rows!(210, dot_row_q6_k),
        // No serial row-dot for this dtype → parallel path is correct anyway.
        _ => mm_1d(a, w, out),
    }
}

#[inline]
fn blk_for(dt: GgufDtype) -> usize {
    match dt {
        GgufDtype::Q8_0 | GgufDtype::Q5_0 => 32,
        GgufDtype::Q4K | GgufDtype::Q5K | GgufDtype::Q6K => 256,
        _ => 1,
    }
}

#[inline]
fn dot_row_q8_0(a: &[f32], row: &[u8], k: usize) -> f32 {
    const BB: usize = 34;
    let mut sumf = 0.0f32;
    for ib in 0..(k / 32) {
        let s = ib * BB;
        let scale = f16::from_le_bytes([row[s], row[s + 1]]).to_f32();
        let abase = ib * 32;
        let mut sub = 0.0f32;
        for q in 0..32 { sub += (row[s + 2 + q] as i8) as f32 * a[abase + q]; }
        sumf += scale * sub;
    }
    sumf
}

#[inline]
fn dot_row_q5_0(a: &[f32], row: &[u8], k: usize) -> f32 {
    const BB: usize = 22;
    let mut sumf = 0.0f32;
    for ib in 0..(k / 32) {
        let s = ib * BB;
        let scale = f16::from_le_bytes([row[s], row[s + 1]]).to_f32();
        let qh = u32::from_le_bytes([row[s + 2], row[s + 3], row[s + 4], row[s + 5]]);
        let abase = ib * 32;
        let mut sub = 0.0f32;
        for q in 0..16 {
            let byte = row[s + 6 + q];
            let h0 = ((qh >> q) & 1) as u8;
            let h1 = ((qh >> (q + 16)) & 1) as u8;
            let v0 = ((byte & 0x0F) | (h0 << 4)) as i8 - 16;
            let v1 = ((byte >> 4) | (h1 << 4)) as i8 - 16;
            sub += v0 as f32 * a[abase + q];
            sub += v1 as f32 * a[abase + q + 16];
        }
        sumf += scale * sub;
    }
    sumf
}

#[inline]
fn dot_row_q4_k(a: &[f32], row: &[u8], k: usize) -> f32 {
    let mut sumf = 0.0f32;
    for ib in 0..(k / 256) {
        let s = ib * 144;
        let d = f16::from_le_bytes([row[s], row[s + 1]]).to_f32();
        let dmin = f16::from_le_bytes([row[s + 2], row[s + 3]]).to_f32();
        let scales = &row[s + 4..s + 16];
        let qs = &row[s + 16..s + 144];
        let a_super = ib * 256;
        for g in 0..4 {
            let q_base = g * 32;
            let a_base = a_super + g * 64;
            let (sc1, m1) = q4_k_get_scale_min(scales, g * 2);
            let (sc2, m2) = q4_k_get_scale_min(scales, g * 2 + 1);
            let (d1, min1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, min2) = (d * sc2 as f32, dmin * m2 as f32);
            let (mut va_lo, mut a_lo, mut va_hi, mut a_hi) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for l in 0..32 {
                let byte = qs[q_base + l];
                let al = a[a_base + l];
                let ah = a[a_base + 32 + l];
                va_lo += (byte & 0x0F) as f32 * al;
                a_lo += al;
                va_hi += ((byte >> 4) & 0x0F) as f32 * ah;
                a_hi += ah;
            }
            sumf += d1 * va_lo - min1 * a_lo + d2 * va_hi - min2 * a_hi;
        }
    }
    sumf
}

#[inline]
fn dot_row_q5_k(a: &[f32], row: &[u8], k: usize) -> f32 {
    let mut sumf = 0.0f32;
    for ib in 0..(k / 256) {
        let s = ib * 176;
        let d = f16::from_le_bytes([row[s], row[s + 1]]).to_f32();
        let dmin = f16::from_le_bytes([row[s + 2], row[s + 3]]).to_f32();
        let scales = &row[s + 4..s + 16];
        let qh = &row[s + 16..s + 48];
        let qs = &row[s + 48..s + 176];
        let a_super = ib * 256;
        for g in 0..4 {
            let q_base = g * 32;
            let a_base = a_super + g * 64;
            let (sc1, m1) = q4_k_get_scale_min(scales, g * 2);
            let (sc2, m2) = q4_k_get_scale_min(scales, g * 2 + 1);
            let (d1, min1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, min2) = (d * sc2 as f32, dmin * m2 as f32);
            let (shift_lo, shift_hi) = (2 * g, 2 * g + 1);
            let (mut va_lo, mut a_lo, mut va_hi, mut a_hi) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for l in 0..32 {
                let byte = qs[q_base + l];
                let qh_b = qh[l];
                let h1 = ((qh_b >> shift_lo) & 1) << 4;
                let h2 = ((qh_b >> shift_hi) & 1) << 4;
                let al = a[a_base + l];
                let ah = a[a_base + 32 + l];
                va_lo += ((byte & 0x0F) | h1) as f32 * al;
                a_lo += al;
                va_hi += (((byte >> 4) & 0x0F) | h2) as f32 * ah;
                a_hi += ah;
            }
            sumf += d1 * va_lo - min1 * a_lo + d2 * va_hi - min2 * a_hi;
        }
    }
    sumf
}

#[inline]
fn dot_row_q6_k(a: &[f32], row: &[u8], k: usize) -> f32 {
    let mut sumf = 0.0f32;
    for ib in 0..(k / 256) {
        let s = ib * 210;
        let ql = &row[s..s + 128];
        let qh = &row[s + 128..s + 192];
        let scales_i8: &[i8] = bytemuck::cast_slice(&row[s + 192..s + 208]);
        let d = f16::from_le_bytes([row[s + 208], row[s + 209]]).to_f32();
        let a_super = ib * 256;
        for outer in 0..2 {
            let ql_off = outer * 64;
            let qh_off = outer * 32;
            let sc_off = outer * 8;
            let a_off = a_super + outer * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[ql_off + l] & 0x0F) as i32 | (((qh[qh_off + l]) & 0x03) as i32) << 4) - 32;
                let q2 = ((ql[ql_off + l + 32] & 0x0F) as i32 | (((qh[qh_off + l] >> 2) & 0x03) as i32) << 4) - 32;
                let q3 = ((ql[ql_off + l] >> 4) as i32 | (((qh[qh_off + l] >> 4) & 0x03) as i32) << 4) - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4) as i32 | (((qh[qh_off + l] >> 6) & 0x03) as i32) << 4) - 32;
                sumf += d * scales_i8[sc_off + is] as f32 * q1 as f32 * a[a_off + l];
                sumf += d * scales_i8[sc_off + is + 2] as f32 * q2 as f32 * a[a_off + l + 32];
                sumf += d * scales_i8[sc_off + is + 4] as f32 * q3 as f32 * a[a_off + l + 64];
                sumf += d * scales_i8[sc_off + is + 6] as f32 * q4 as f32 * a[a_off + l + 96];
            }
        }
    }
    sumf
}

// ── F32 / F16 / BF16 (whole-element formats) ───────────────────────

fn mm_1d_f32(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    // f32 path: bytes already lay out as [n, k] row-major f32. We can
    // call sgemm with the natural layout — but sgemm wants `[k, n]` as
    // B, so we'd need a transpose. Instead, do a scalar dot per row.
    // This is the rare case (most weights are quantized); not the perf
    // critical path. For larger f32 cases callers can pre-transpose
    // and use `rce_op_blas::sgemm_into` directly.
    let w: &[f32] = bytemuck::cast_slice(w_bytes);
    for j in 0..n {
        let row = &w[j * k..(j + 1) * k];
        let mut acc = 0.0f32;
        for i in 0..k { acc += a[i] * row[i]; }
        out[j] = acc;
    }
}

fn mm_1d_f16(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    let w: &[f16] = bytemuck::cast_slice(w_bytes);
    out[..n].par_iter_mut().enumerate().for_each(|(j, slot)| {
        let row = &w[j * k..(j + 1) * k];
        let mut acc = 0.0f32;
        for i in 0..k { acc += a[i] * row[i].to_f32(); }
        *slot = acc;
    });
}

fn mm_1d_bf16(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    let w: &[half::bf16] = bytemuck::cast_slice(w_bytes);
    out[..n].par_iter_mut().enumerate().for_each(|(j, slot)| {
        let row = &w[j * k..(j + 1) * k];
        let mut acc = 0.0f32;
        for i in 0..k { acc += a[i] * row[i].to_f32(); }
        *slot = acc;
    });
}

// ── Q4_0 / Q8_0 (32-element blocks) ────────────────────────────────

fn mm_1d_q4_0(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    const BLOCK: usize = 32;
    const BB: usize = 18;
    debug_assert!(k % BLOCK == 0, "q4_0 mm_1d needs k divisible by {BLOCK}, got {k}");
    let blocks_per_row = k / BLOCK;
    let row_bytes = blocks_per_row * BB;
    out[..n].par_iter_mut().enumerate().for_each(|(j, slot)| {
        let row_start = j * row_bytes;
        let mut sumf = 0.0f32;
        for ib in 0..blocks_per_row {
            let s = row_start + ib * BB;
            let scale = f16::from_le_bytes([w_bytes[s], w_bytes[s + 1]]).to_f32();
            let a_base = ib * BLOCK;
            // Split-nibble: first 16 = LOW nibbles, next 16 = HIGH nibbles.
            let mut sub = 0.0f32;
            for q in 0..16 {
                let byte = w_bytes[s + 2 + q];
                let v0 = (byte & 0x0F) as i8 - 8;
                let v1 = ((byte >> 4) & 0x0F) as i8 - 8;
                sub += v0 as f32 * a[a_base + q];
                sub += v1 as f32 * a[a_base + q + 16];
            }
            sumf += scale * sub;
        }
        *slot = sumf;
    });
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))] // NEON path is production on aarch64; fallback + test reference
fn mm_1d_q5_0(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    // Q5_0 block (22 bytes): [d: f16][qh: u32][ql: 16 bytes]. Same
    // split-nibble layout as Q4_0 with the 5th bit pulled from `qh`
    // (low halves use bit j, high halves use bit j+16). v = q5 - 16.
    const BLOCK: usize = 32;
    const BB: usize = 22;
    debug_assert!(k % BLOCK == 0, "q5_0 mm_1d needs k divisible by {BLOCK}, got {k}");
    let blocks_per_row = k / BLOCK;
    let row_bytes = blocks_per_row * BB;
    out[..n].par_iter_mut().enumerate().for_each(|(j, slot)| {
        let row_start = j * row_bytes;
        let mut sumf = 0.0f32;
        for ib in 0..blocks_per_row {
            let s = row_start + ib * BB;
            let scale = f16::from_le_bytes([w_bytes[s], w_bytes[s + 1]]).to_f32();
            let qh = u32::from_le_bytes([w_bytes[s + 2], w_bytes[s + 3], w_bytes[s + 4], w_bytes[s + 5]]);
            let a_base = ib * BLOCK;
            let mut sub = 0.0f32;
            for q in 0..16 {
                let byte = w_bytes[s + 6 + q];
                let h0 = ((qh >> q) & 1) as u8;
                let h1 = ((qh >> (q + 16)) & 1) as u8;
                let v0 = ((byte & 0x0F) | (h0 << 4)) as i8 - 16;
                let v1 = ((byte >> 4) | (h1 << 4)) as i8 - 16;
                sub += v0 as f32 * a[a_base + q];
                sub += v1 as f32 * a[a_base + q + 16];
            }
            sumf += scale * sub;
        }
        *slot = sumf;
    });
}

/// NEON Q5_0. Same split-nibble + `qh` 5th-bit layout as the scalar
/// `mm_1d_q5_0`, vectorized: the per-block 5th bits are expanded from
/// the `qh` u32 into a {0,16} byte-add via a bit-select compare, then
/// `v = (nibble|bit) - 16` is widened to f32 and FMA'd 4-wide.
/// Validated against scalar by `q5_0_neon_matches_scalar`.
#[cfg(target_arch = "aarch64")]
fn mm_1d_q5_0_neon(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    use std::arch::aarch64::*;
    const BLOCK: usize = 32;
    const BB: usize = 22;
    debug_assert!(k % BLOCK == 0);
    let blocks_per_row = k / BLOCK;
    let row_bytes = blocks_per_row * BB;
    out[..n].par_iter_mut().enumerate().for_each(|(j, slot)| {
        let row_start = j * row_bytes;
        unsafe {
            let bitsel = vld1q_u8([1u8,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128].as_ptr());
            let nib_mask = vdupq_n_u8(0x0F);
            let off16_u = vdupq_n_u8(16);
            let off16_s = vdupq_n_s8(16);
            let mut acc = vdupq_n_f32(0.0);
            for ib in 0..blocks_per_row {
                let s = row_start + ib * BB;
                let scale = f16::from_le_bytes([w_bytes[s], w_bytes[s + 1]]).to_f32();
                let scale_v = vdupq_n_f32(scale);
                let qh = u32::from_le_bytes([w_bytes[s + 2], w_bytes[s + 3], w_bytes[s + 4], w_bytes[s + 5]]);
                let ql = vld1q_u8(w_bytes.as_ptr().add(s + 6)); // 16 bytes
                // Expand qh: lanes 0..7 use byte b, lanes 8..15 use byte b+1.
                let qh_lo_b = vcombine_u8(vdup_n_u8((qh & 0xFF) as u8), vdup_n_u8(((qh >> 8) & 0xFF) as u8));
                let qh_hi_b = vcombine_u8(vdup_n_u8(((qh >> 16) & 0xFF) as u8), vdup_n_u8(((qh >> 24) & 0xFF) as u8));
                let add_lo = vandq_u8(vceqq_u8(vandq_u8(qh_lo_b, bitsel), bitsel), off16_u);
                let add_hi = vandq_u8(vceqq_u8(vandq_u8(qh_hi_b, bitsel), bitsel), off16_u);
                // 5-bit values 0..31 → signed (−16) → elements 0..15 (lo) and 16..31 (hi).
                let v_lo = vsubq_s8(vreinterpretq_s8_u8(vaddq_u8(vandq_u8(ql, nib_mask), add_lo)), off16_s);
                let v_hi = vsubq_s8(vreinterpretq_s8_u8(vaddq_u8(vshrq_n_u8(ql, 4), add_hi)), off16_s);
                let aptr = a.as_ptr().add(ib * BLOCK);
                let mut blk = vdupq_n_f32(0.0);
                for (half, vv) in [(0usize, v_lo), (16usize, v_hi)] {
                    let v16_lo = vmovl_s8(vget_low_s8(vv));
                    let v16_hi = vmovl_s8(vget_high_s8(vv));
                    let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_lo)));
                    let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_lo)));
                    let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_hi)));
                    let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_hi)));
                    let ab = aptr.add(half);
                    blk = vfmaq_f32(blk, f0, vld1q_f32(ab));
                    blk = vfmaq_f32(blk, f1, vld1q_f32(ab.add(4)));
                    blk = vfmaq_f32(blk, f2, vld1q_f32(ab.add(8)));
                    blk = vfmaq_f32(blk, f3, vld1q_f32(ab.add(12)));
                }
                acc = vfmaq_f32(acc, blk, scale_v);
            }
            *slot = vaddvq_f32(acc);
        }
    });
}

fn mm_1d_q4_1(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    const BLOCK: usize = 32;
    const BB: usize = 20;
    debug_assert!(k % BLOCK == 0, "q4_1 mm_1d needs k divisible by {BLOCK}, got {k}");
    let blocks_per_row = k / BLOCK;
    let row_bytes = blocks_per_row * BB;
    out[..n].par_iter_mut().enumerate().for_each(|(j, slot)| {
        let row_start = j * row_bytes;
        let mut sumf = 0.0f32;
        for ib in 0..blocks_per_row {
            let s = row_start + ib * BB;
            let scale = f16::from_le_bytes([w_bytes[s], w_bytes[s + 1]]).to_f32();
            let min = f16::from_le_bytes([w_bytes[s + 2], w_bytes[s + 3]]).to_f32();
            let a_base = ib * BLOCK;
            // w[i] = scale * v[i] + min ⇒ Σ a·w = scale·Σ a·v + min·Σ a.
            let mut va = 0.0f32;
            let mut asum = 0.0f32;
            for q in 0..16 {
                let byte = w_bytes[s + 4 + q];
                let v0 = (byte & 0x0F) as f32;
                let v1 = ((byte >> 4) & 0x0F) as f32;
                let a_lo = a[a_base + q];
                let a_hi = a[a_base + q + 16];
                va += v0 * a_lo + v1 * a_hi;
                asum += a_lo + a_hi;
            }
            sumf += scale * va + min * asum;
        }
        *slot = sumf;
    });
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))] // NEON path is production on aarch64; this is the fallback + test reference
fn mm_1d_q8_0(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    const BLOCK: usize = 32;
    const BB: usize = 34;
    debug_assert!(k % BLOCK == 0);
    let blocks_per_row = k / BLOCK;
    let row_bytes = blocks_per_row * BB;
    out[..n].par_iter_mut().enumerate().for_each(|(j, slot)| {
        let row_start = j * row_bytes;
        let mut sumf = 0.0f32;
        for ib in 0..blocks_per_row {
            let s = row_start + ib * BB;
            let scale = f16::from_le_bytes([w_bytes[s], w_bytes[s + 1]]).to_f32();
            let a_base = ib * BLOCK;
            let mut sub = 0.0f32;
            for q in 0..32 {
                let v = w_bytes[s + 2 + q] as i8;
                sub += v as f32 * a[a_base + q];
            }
            sumf += scale * sub;
        }
        *slot = sumf;
    });
}

/// NEON Q8_0: per output row, dot f32 activations with int8×scale
/// weights. Widens int8→int16→int32→f32 16-at-a-time and FMAs into a
/// 4-lane accumulator; the f16 block scale is applied per 32-block.
/// ~4-8× the scalar loop on Apple Silicon. Bit-identical up to f32
/// reordering vs `mm_1d_q8_0` (validated by `q8_0_neon_matches_scalar`).
#[cfg(target_arch = "aarch64")]
fn mm_1d_q8_0_neon(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    use std::arch::aarch64::*;
    const BLOCK: usize = 32;
    const BB: usize = 34;
    debug_assert!(k % BLOCK == 0);
    let blocks_per_row = k / BLOCK;
    let row_bytes = blocks_per_row * BB;
    out[..n].par_iter_mut().enumerate().for_each(|(j, slot)| {
        let row_start = j * row_bytes;
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            for ib in 0..blocks_per_row {
                let s = row_start + ib * BB;
                let scale = f16::from_le_bytes([w_bytes[s], w_bytes[s + 1]]).to_f32();
                let scale_v = vdupq_n_f32(scale);
                let wptr = w_bytes.as_ptr().add(s + 2) as *const i8;
                let aptr = a.as_ptr().add(ib * BLOCK);
                let mut blk = vdupq_n_f32(0.0);
                for h in 0..2 {
                    let w8 = vld1q_s8(wptr.add(h * 16)); // 16 int8
                    let w16_lo = vmovl_s8(vget_low_s8(w8));
                    let w16_hi = vmovl_s8(vget_high_s8(w8));
                    let w32_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16_lo)));
                    let w32_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16_lo)));
                    let w32_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16_hi)));
                    let w32_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16_hi)));
                    let abase = aptr.add(h * 16);
                    blk = vfmaq_f32(blk, w32_0, vld1q_f32(abase));
                    blk = vfmaq_f32(blk, w32_1, vld1q_f32(abase.add(4)));
                    blk = vfmaq_f32(blk, w32_2, vld1q_f32(abase.add(8)));
                    blk = vfmaq_f32(blk, w32_3, vld1q_f32(abase.add(12)));
                }
                acc = vfmaq_f32(acc, blk, scale_v);
            }
            *slot = vaddvq_f32(acc);
        }
    });
}

// ── Q4_K / Q5_K / Q6_K (256-element K-quants) ──────────────────────

#[inline]
fn q4_k_get_scale_min(scales: &[u8], j: usize) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

fn mm_1d_q4_k(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    const BLOCK: usize = 256;
    const BB: usize = 144;
    debug_assert!(k % BLOCK == 0, "q4_k mm_1d needs k divisible by {BLOCK}, got {k}");
    let blocks_per_row = k / BLOCK;
    let row_bytes = blocks_per_row * BB;
    // Clean scalar inner loop, parallelised per output row by rayon.
    // (A hand-written NEON variant was tried and *regressed* — LLVM
    // already auto-vectorises this clean form at opt-level=3 + fat-LTO,
    // and the naive intrinsics paid a stack round-trip for nibble
    // unpacking that exceeded the FMA win. The real next lever for the
    // remaining matmul cost is multi-token batching, not micro-SIMD.)
    out[..n].par_iter_mut().enumerate().for_each(|(j, slot)| {
        let row_start = j * row_bytes;
        let mut sumf = 0.0f32;
        for ib in 0..blocks_per_row {
            let s = row_start + ib * BB;
            let d = f16::from_le_bytes([w_bytes[s], w_bytes[s + 1]]).to_f32();
            let dmin = f16::from_le_bytes([w_bytes[s + 2], w_bytes[s + 3]]).to_f32();
            let scales = &w_bytes[s + 4..s + 16];
            let qs = &w_bytes[s + 16..s + 144];
            let a_super_base = ib * BLOCK;
            for g in 0..4 {
                let q_base = g * 32;
                let a_base = a_super_base + g * 64;
                let (sc1, m1) = q4_k_get_scale_min(scales, g * 2);
                let (sc2, m2) = q4_k_get_scale_min(scales, g * 2 + 1);
                let d1 = d * sc1 as f32;
                let min1 = dmin * m1 as f32;
                let d2 = d * sc2 as f32;
                let min2 = dmin * m2 as f32;
                let mut va_lo = 0.0f32;
                let mut a_lo = 0.0f32;
                let mut va_hi = 0.0f32;
                let mut a_hi = 0.0f32;
                for l in 0..32 {
                    let byte = qs[q_base + l];
                    let v_lo = (byte & 0x0F) as f32;
                    let v_hi = ((byte >> 4) & 0x0F) as f32;
                    let al = a[a_base + l];
                    let ah = a[a_base + 32 + l];
                    va_lo += v_lo * al;
                    a_lo += al;
                    va_hi += v_hi * ah;
                    a_hi += ah;
                }
                sumf += d1 * va_lo - min1 * a_lo + d2 * va_hi - min2 * a_hi;
            }
        }
        *slot = sumf;
    });
}

fn mm_1d_q5_k(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    const BLOCK: usize = 256;
    const BB: usize = 176;
    debug_assert!(k % BLOCK == 0, "q5_k mm_1d needs k divisible by {BLOCK}, got {k}");
    let blocks_per_row = k / BLOCK;
    let row_bytes = blocks_per_row * BB;
    out[..n].par_iter_mut().enumerate().for_each(|(j, slot)| {
        let row_start = j * row_bytes;
        let mut sumf = 0.0f32;
        for ib in 0..blocks_per_row {
            let s = row_start + ib * BB;
            let d = f16::from_le_bytes([w_bytes[s], w_bytes[s + 1]]).to_f32();
            let dmin = f16::from_le_bytes([w_bytes[s + 2], w_bytes[s + 3]]).to_f32();
            let scales = &w_bytes[s + 4..s + 16];
            let qh = &w_bytes[s + 16..s + 48];
            let qs = &w_bytes[s + 48..s + 176];
            let a_super_base = ib * BLOCK;
            // Same 4-group-of-64 structure as Q4_K, with the 5th bit
            // pulled from `qh` (32 bytes reused across all 4 groups;
            // groups 0..3 read bits 0..1, 2..3, 4..5, 6..7 of each qh byte).
            for g in 0..4 {
                let q_base = g * 32;
                let a_base = a_super_base + g * 64;
                let (sc1, m1) = q4_k_get_scale_min(scales, g * 2);
                let (sc2, m2) = q4_k_get_scale_min(scales, g * 2 + 1);
                let d1 = d * sc1 as f32;
                let min1 = dmin * m1 as f32;
                let d2 = d * sc2 as f32;
                let min2 = dmin * m2 as f32;
                let shift_lo = 2 * g;
                let shift_hi = 2 * g + 1;
                let mut va_lo = 0.0f32;
                let mut a_lo = 0.0f32;
                let mut va_hi = 0.0f32;
                let mut a_hi = 0.0f32;
                // Branchless 5th-bit extract — LLVM vectorises cleanly.
                for l in 0..32 {
                    let byte = qs[q_base + l];
                    let qh_b = qh[l];
                    let h1 = ((qh_b >> shift_lo) & 1) << 4;
                    let h2 = ((qh_b >> shift_hi) & 1) << 4;
                    let v_lo = ((byte & 0x0F) | h1) as f32;
                    let v_hi = (((byte >> 4) & 0x0F) | h2) as f32;
                    let al = a[a_base + l];
                    let ah = a[a_base + 32 + l];
                    va_lo += v_lo * al;
                    a_lo += al;
                    va_hi += v_hi * ah;
                    a_hi += ah;
                }
                sumf += d1 * va_lo - min1 * a_lo + d2 * va_hi - min2 * a_hi;
            }
        }
        *slot = sumf;
    });
}

fn mm_1d_q3_k(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    // Q3_K block: 256 elements / 110 bytes.
    //   [hmask 32 B][qs 64 B][packed scales 12 B][d (f16) 2 B]
    //
    // Layout (matches llama.cpp `dequantize_row_q3_K` / `rce-ingest-gguf`'s
    // `dequant_q3_k`):
    // - `qs` holds the low 2 bits of each weight, 4 weights packed per byte
    //   (shift 0,2,4,6 selects which weight).
    // - `hmask` holds the 3rd (high) bit of each weight, 8 weights per byte
    //   addressed by a moving 1-of-8 mask `m`. A *set* high bit means **no**
    //   −4 offset; an *unset* high bit means subtract 4. (Yes, inverted —
    //   that's the GGUF convention.) Final weight ∈ {−4..3}.
    // - 12 bytes of packed 6-bit scales need a 4-step KMASK1/KMASK2
    //   dance to recover 16 i8 sub-scales in [0,63]; subtract 32 to
    //   recover the signed 6-bit value.
    // - Each block traversal: 2 outer × 4 j-loops × (16 + 16) = 256 outs.
    //   `m` and `is` carry across the outer iterations (m runs 1,2,4,...,128
    //   through all 8 j-loops, is runs 0..16).
    const BLOCK: usize = 256;
    const BB: usize = 110;
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    debug_assert!(k % BLOCK == 0, "q3_k mm_1d needs k divisible by {BLOCK}, got {k}");
    let blocks_per_row = k / BLOCK;
    let row_bytes = blocks_per_row * BB;
    out[..n].par_iter_mut().enumerate().for_each(|(j, slot)| {
        let row_start = j * row_bytes;
        let mut sumf = 0.0f32;
        for ib in 0..blocks_per_row {
            let s = row_start + ib * BB;
            let hmask = &w_bytes[s..s + 32];
            let qs = &w_bytes[s + 32..s + 96];
            let scales_bytes = &w_bytes[s + 96..s + 108];
            let d_all = f16::from_le_bytes([w_bytes[s + 108], w_bytes[s + 109]]).to_f32();
            // Unpack 12 packed bytes → 16 i8 sub-scales (KMASK1/KMASK2 dance).
            let mut aux = [0u32; 4];
            for kk in 0..3 {
                aux[kk] = u32::from_le_bytes([
                    scales_bytes[kk * 4], scales_bytes[kk * 4 + 1],
                    scales_bytes[kk * 4 + 2], scales_bytes[kk * 4 + 3],
                ]);
            }
            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            aux[0] = (aux[0] & KMASK2) | (((tmp) & KMASK1) << 4);
            aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
            let mut scales = [0i8; 16];
            for kk in 0..4 {
                let b4 = aux[kk].to_le_bytes();
                for jj in 0..4 {
                    scales[kk * 4 + jj] = b4[jj] as i8;
                }
            }

            let a_super_base = ib * BLOCK;
            let mut m: u8 = 1;
            let mut is: usize = 0;
            let mut q_off: usize = 0;
            for outer in 0..2 {
                let mut shift: u32 = 0;
                for jj in 0..4 {
                    let dl1 = d_all * (scales[is] as i32 - 32) as f32;
                    let dl2 = d_all * (scales[is + 1] as i32 - 32) as f32;
                    let a_base = a_super_base + outer * 128 + jj * 32;
                    let mut sub_dot1 = 0.0f32;
                    let mut sub_dot2 = 0.0f32;
                    for l in 0..16 {
                        let ql1 = ((qs[q_off + l] >> shift) & 0x03) as i32;
                        let high1 = if hmask[l] & m != 0 { 0 } else { 4 };
                        let q1 = (ql1 - high1) as f32;
                        sub_dot1 += q1 * a[a_base + l];

                        let ql2 = ((qs[q_off + l + 16] >> shift) & 0x03) as i32;
                        let high2 = if hmask[l + 16] & m != 0 { 0 } else { 4 };
                        let q2 = (ql2 - high2) as f32;
                        sub_dot2 += q2 * a[a_base + 16 + l];
                    }
                    sumf += dl1 * sub_dot1 + dl2 * sub_dot2;
                    is += 2;
                    shift += 2;
                    m = m.wrapping_shl(1);
                }
                q_off += 32;
            }
        }
        *slot = sumf;
    });
}

/// **Batched matmul** `out[T,N] = a[T,K] · w[N,K]^T`, the prefill lever.
///
/// vs calling [`mm_1d`] T times: the weight-block loop is **outermost**
/// and the per-token loop **innermost**, so each quantized weight block
/// is read from RAM (and its nibbles unpacked) **once** and reused
/// across all T activation rows — amortising the memory-bandwidth-bound
/// weight read T×. The per-(row,col) accumulation is the *identical*
/// float arithmetic [`mm_1d`] does, so the result is **bit-identical**
/// to per-row `mm_1d` (proven by the brick-1 test).
///
/// `a` is `[T, n_cols]` row-major; `out` is `[T, n_rows]` row-major.
/// Brick 1 scope: Q5_K only (mamba2-2.7b's body format) — enough to
/// prove or kill the prefill-batching premise before extending.
pub fn mm_1d_batched(a: &[f32], t_rows: usize, w: &QuantWeight, out: &mut [f32]) {
    let k = w.n_cols;
    let n = w.n_rows;
    debug_assert_eq!(a.len(), t_rows * k);
    debug_assert!(out.len() >= t_rows * n);
    match w.dtype {
        GgufDtype::Q5K => mm_batched_q5_k(a, t_rows, w.bytes(), k, n, out),
        other => panic!(
            "mm_1d_batched: brick-1 scope is Q5_K only; {other:?} not yet implemented"
        ),
    }
}

/// Dtype-dispatching batched matmul: `out[T,N] = a[T,K] · w[N,K]^T`.
///
/// **Byte-identical to calling [`mm_1d`] `t_rows` times**, for *every*
/// dtype. Q5_K takes the weight-block-amortised [`mm_1d_batched`] fast
/// path (the prefill lever — proven bit-identical to per-row by brick 1).
/// Every other dtype falls back to a per-row `mm_1d` loop: correct, no
/// speedup, no panic. This makes the prefill driver dtype-agnostic so
/// the fast path is extended one dtype at a time without breaking
/// callers — exactly the brick-by-brick scope.
///
/// `a` is `[t_rows, w.n_cols]` row-major; `out` is `[t_rows, w.n_rows]`.
pub fn mm_batched(a: &[f32], t_rows: usize, w: &QuantWeight, out: &mut [f32]) {
    let n = w.n_rows;
    let k = w.n_cols;

    // AMX fast path: Q8_0 with M ≥ 16, N % 16 == 0, K % 32 == 0 on
    // Apple Silicon. Tiles M into 16-row chunks, runs each through
    // `rce_op_amx::amx_qmm_q8_0_batched` (FMA32 GEMM tile), falls
    // through to the per-row mm_1d path for any trailing M < 16 rows.
    // Measured: 3.21× speedup over mm_batched fallback at TinyLlama
    // prefill shape (M=16, N=K=2048). See rce-op-amx benches.
    #[cfg(feature = "amx")]
    if w.dtype == GgufDtype::Q8_0
        && t_rows >= 16
        && n % 16 == 0
        && k % 32 == 0
        && cfg!(all(target_arch = "aarch64", target_os = "macos"))
    {
        if let Some(_g) = rce_op_amx::AmxGuard::enter() {
            let mut a_t = vec![0.0f32; k * 16];
            let mut tile_out = vec![0.0f32; 16 * n];
            let mut m = 0;
            while m + 16 <= t_rows {
                // Transpose 16 rows of A into [K, 16].
                for mi in 0..16 {
                    let src_row = &a[(m + mi) * k..(m + mi + 1) * k];
                    for kk in 0..k {
                        a_t[kk * 16 + mi] = src_row[kk];
                    }
                }
                // SAFETY: AmxGuard alive; lengths checked above.
                unsafe {
                    rce_op_amx::amx_qmm_q8_0_batched(&a_t, w.bytes(), n, k, &mut tile_out);
                }
                // Scatter [16, N] tile back into [M, N] output.
                for mi in 0..16 {
                    out[(m + mi) * n..(m + mi + 1) * n]
                        .copy_from_slice(&tile_out[mi * n..(mi + 1) * n]);
                }
                m += 16;
            }
            // Trailing M < 16 rows: scalar mm_1d.
            for t in m..t_rows {
                mm_1d(&a[t * k..(t + 1) * k], w, &mut out[t * n..(t + 1) * n]);
            }
            return;
        }
    }

    match w.dtype {
        GgufDtype::Q5K => mm_1d_batched(a, t_rows, w, out),
        _ => {
            for t in 0..t_rows {
                mm_1d(&a[t * k..(t + 1) * k], w, &mut out[t * n..(t + 1) * n]);
            }
        }
    }
}

fn mm_batched_q5_k(a: &[f32], t_rows: usize, w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    const BLOCK: usize = 256;
    const BB: usize = 176;
    debug_assert!(k % BLOCK == 0);
    let blocks_per_row = k / BLOCK;
    let row_bytes = blocks_per_row * BB;
    // Parallelise over output columns j. Each task computes the T-vector
    // for its column (weight block read+unpacked once, reused across all
    // T rows), returns it; we then scatter into row-major [T,N]. The
    // scatter is O(N·T) (cheap vs the O(N·T·K) matmul) and keeps this
    // safe (no cross-thread aliasing) without losing the weight reuse.
    let cols: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|j| {
        let row_start = j * row_bytes;
        let mut acc = vec![0.0f32; t_rows];
        for ib in 0..blocks_per_row {
            let s = row_start + ib * BB;
            let d = f16::from_le_bytes([w_bytes[s], w_bytes[s + 1]]).to_f32();
            let dmin = f16::from_le_bytes([w_bytes[s + 2], w_bytes[s + 3]]).to_f32();
            let scales = &w_bytes[s + 4..s + 16];
            let qh = &w_bytes[s + 16..s + 48];
            let qs = &w_bytes[s + 48..s + 176];
            let a_super_base = ib * BLOCK;
            for g in 0..4 {
                let q_base = g * 32;
                let a_base = a_super_base + g * 64;
                let (sc1, m1) = q4_k_get_scale_min(scales, g * 2);
                let (sc2, m2) = q4_k_get_scale_min(scales, g * 2 + 1);
                let d1 = d * sc1 as f32;
                let min1 = dmin * m1 as f32;
                let d2 = d * sc2 as f32;
                let min2 = dmin * m2 as f32;
                let shift_lo = 2 * g;
                let shift_hi = 2 * g + 1;
                // Unpack the 32 lo / 32 hi weight nibbles ONCE per
                // (j, block, group) — reused across all T tokens.
                let mut vlo = [0.0f32; 32];
                let mut vhi = [0.0f32; 32];
                for l in 0..32 {
                    let byte = qs[q_base + l];
                    let qh_b = qh[l];
                    let h1 = ((qh_b >> shift_lo) & 1) << 4;
                    let h2 = ((qh_b >> shift_hi) & 1) << 4;
                    vlo[l] = ((byte & 0x0F) | h1) as f32;
                    vhi[l] = (((byte >> 4) & 0x0F) | h2) as f32;
                }
                for t in 0..t_rows {
                    let at = &a[t * k..(t + 1) * k];
                    // Identical factored accumulation to mm_1d_q5_k →
                    // bit-identical per row.
                    let mut va_lo = 0.0f32;
                    let mut a_lo = 0.0f32;
                    let mut va_hi = 0.0f32;
                    let mut a_hi = 0.0f32;
                    for l in 0..32 {
                        let al = at[a_base + l];
                        let ah = at[a_base + 32 + l];
                        va_lo += vlo[l] * al;
                        a_lo += al;
                        va_hi += vhi[l] * ah;
                        a_hi += ah;
                    }
                    acc[t] += d1 * va_lo - min1 * a_lo + d2 * va_hi - min2 * a_hi;
                }
            }
        }
        acc
        })
        .collect();
    for (j, col) in cols.iter().enumerate() {
        for t in 0..t_rows {
            out[t * n + j] = col[t];
        }
    }
}

// ── rce-native quant (`fq4`) — brick 1 design spike ──────────────
//
// First-principles quant designed for the *batched matmul's access
// pattern on Apple Silicon*, NOT for storage density or x86 SIMD (which
// is all GGUF's K-quant geometry is). Two deliberate departures from
// ggml:
//
//  1. **Structure-of-Arrays.** GGUF interleaves scales + quants inside
//     every superblock, so dequanting one weight needs intra-block
//     bit-juggling to *find* its scale. `fq4` splits a tensor into three
//     contiguous planes — all 4-bit codes, all f16 block scales `d`,
//     all f16 block mins `m`. The kernel streams the code plane with
//     pure 16-byte loads; scales are a tiny side plane read once/block.
//
//  2. **4-bit codes, byte-clean.** GGUF Q5_K splits the 5th bit into a
//     separate plane, so an unpack must recombine bits from two places
//     (the documented reason hand-NEON lost to autovec — building the
//     dequant `[f32;4]` cost a scalar stack round-trip). 4-bit is the
//     only width whose unpack is a branchless `vld1q_u8 → vand/vshr →
//     vmovl → vcvt`, no scalar extraction. We fix the NEON dead-end *in
//     the format*, then write the clean scalar form LLVM auto-vectorises.
//
// Affine scheme `w = d*q + m`, q∈[0,15], one (d,m) per 32-element block:
// 4 b/code + (16+16) b/32 = **5.0 bpw** (vs Q5_K 5.5 — cheaper *and*,
// if the layout thesis holds, faster). On-disk single-blob packing +
// 128-byte plane alignment is the brick-3 container concern; brick 1
// proves only the access-pattern win, so the planes are plain `Vec`s
// (already contiguous-per-plane — that *is* the SoA thesis).

/// A weight matrix `[n_rows, k]` in the `fq4` structure-of-arrays quant.
/// `k` must be a multiple of 64 (the block size).
///
/// **v2 (post-brick-1-measurement):** block 64 (was 32) and scales as
/// **f32** planes (were f16). Brick-1 v1 measured *slower* than Q5_K
/// purely because B=32 → 2× the per-block scale decodes + outer MACs
/// vs Q5_K's 64-wide groups, swamping the 9% bandwidth win. Root cause
/// was the block-size/scale sub-choice, not the SoA/branchless thesis.
/// B=64 matches Q5_K's per-group outer-MAC cadence; f32 scales drop the
/// per-block f16 decode entirely. Still 5.0 bpw (`(64·4 + 32 + 32)/64`),
/// vs Q5_K 5.5 — cheaper *and*, if the thesis holds, faster.
#[derive(Clone)]
pub struct Fq4Tensor {
    /// 4-bit codes, `n_rows * k/2` bytes. Byte `c` of a row's block
    /// holds weight `2c` in the low nibble, `2c+1` in the high nibble.
    pub codes: Vec<u8>,
    /// Per-block scale, `n_rows * k/64` f32 (own plane — no decode).
    pub d: Vec<f32>,
    /// Per-block min (affine zero), `n_rows * k/64` f32.
    pub m: Vec<f32>,
    pub n_rows: usize,
    pub k: usize,
}

pub const FQ4_BLOCK: usize = 64;

/// Quantize a row-major `[n_rows, k]` f32 weight to `fq4`. Scales are
/// stored exactly (f32), so reconstruction error is provably `≤ d/2`
/// per element (the round-trip test asserts it).
pub fn fq4_quantize(weights: &[f32], n_rows: usize, k: usize) -> Fq4Tensor {
    assert_eq!(weights.len(), n_rows * k);
    assert_eq!(k % FQ4_BLOCK, 0, "fq4: k must be a multiple of {FQ4_BLOCK}");
    let bpr = k / FQ4_BLOCK;
    let mut codes = vec![0u8; n_rows * k / 2];
    let mut d = vec![0.0f32; n_rows * bpr];
    let mut m = vec![0.0f32; n_rows * bpr];

    for r in 0..n_rows {
        for b in 0..bpr {
            let w = &weights[r * k + b * FQ4_BLOCK..r * k + (b + 1) * FQ4_BLOCK];
            let mn = w.iter().copied().fold(f32::INFINITY, f32::min);
            let mx = w.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let d_f = (mx - mn) / 15.0;
            let si = r * bpr + b;
            d[si] = d_f;
            m[si] = mn;
            let cbase = r * k / 2 + b * (FQ4_BLOCK / 2);
            for c in 0..FQ4_BLOCK / 2 {
                let q0 = if d_f > 0.0 {
                    (((w[2 * c] - mn) / d_f).round() as i32).clamp(0, 15) as u8
                } else { 0 };
                let q1 = if d_f > 0.0 {
                    (((w[2 * c + 1] - mn) / d_f).round() as i32).clamp(0, 15) as u8
                } else { 0 };
                codes[cbase + c] = q0 | (q1 << 4);
            }
        }
    }
    Fq4Tensor { codes, d, m, n_rows, k }
}

/// Dequantize one row to `k` f32 values. Scalar reference (the
/// round-trip ground truth; not the perf path).
pub fn fq4_dequant_row(t: &Fq4Tensor, row: usize) -> Vec<f32> {
    let bpr = t.k / FQ4_BLOCK;
    let mut out = vec![0.0f32; t.k];
    for b in 0..bpr {
        let d = t.d[row * bpr + b];
        let m = t.m[row * bpr + b];
        let cbase = row * t.k / 2 + b * (FQ4_BLOCK / 2);
        for c in 0..FQ4_BLOCK / 2 {
            let byte = t.codes[cbase + c];
            out[b * FQ4_BLOCK + 2 * c] = d * (byte & 0x0F) as f32 + m;
            out[b * FQ4_BLOCK + 2 * c + 1] = d * (byte >> 4) as f32 + m;
        }
    }
    out
}

/// Batched matmul `out[T,N] = a[T,K] · W[N,K]^T` over an `fq4` weight.
/// Mirrors [`mm_batched_q5_k`]'s parallelisation **exactly** (par over
/// output col `j`, weight block unpacked once / reused across T, serial
/// scatter) so a bench against it isolates the *layout*, not the
/// strategy. Factored affine accumulation:
/// `dot = Σ aₗ(d·qₗ+m) = d·Σ(aₗqₗ) + m·Σaₗ` — same shape as Q5_K's, but
/// `q` comes from a branchless nibble split of a contiguous code plane
/// instead of a cross-plane bit recombination.
pub fn fq4_mm_batched(a: &[f32], t_rows: usize, w: &Fq4Tensor, out: &mut [f32]) {
    let k = w.k;
    let n = w.n_rows;
    let bpr = k / FQ4_BLOCK;
    debug_assert_eq!(a.len(), t_rows * k);
    debug_assert!(out.len() >= t_rows * n);

    // `dot = Σ aₗ(d·qₗ + m) = d·Σ(aₗqₗ) + m·(Σaₗ)`. The `Σaₗ` term is
    // **activation-only** — identical for all N output columns — so
    // hoist it: compute the per-(t, block) activation sum **once**
    // instead of N× inside the hot loop. The affine-SoA form makes this
    // factorisation clean; Q5_K's interleaved hi/lo plane split
    // obstructs the equivalent hoist (it recomputes a_lo/a_hi per col).
    let asum: Vec<f32> = (0..t_rows)
        .flat_map(|t| {
            (0..bpr).map(move |ib| {
                let base = t * k + ib * FQ4_BLOCK;
                a[base..base + FQ4_BLOCK].iter().sum::<f32>()
            })
        })
        .collect();

    let cols: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|j| {
            let code_base = j * k / 2;
            let sc_base = j * bpr;
            let mut acc = vec![0.0f32; t_rows];
            for ib in 0..bpr {
                let d = w.d[sc_base + ib];
                let m = w.m[sc_base + ib];
                let cb = code_base + ib * (FQ4_BLOCK / 2);
                // Branchless nibble unpack ONCE per (j, block) — reused
                // across all T rows. (Clean scalar form; the SoA layout
                // is what lets LLVM auto-vectorise this 16-wide.)
                let mut q = [0.0f32; FQ4_BLOCK];
                for c in 0..FQ4_BLOCK / 2 {
                    let byte = w.codes[cb + c];
                    q[2 * c] = (byte & 0x0F) as f32;
                    q[2 * c + 1] = (byte >> 4) as f32;
                }
                let a_base = ib * FQ4_BLOCK;
                for t in 0..t_rows {
                    let at = &a[t * k + a_base..t * k + a_base + FQ4_BLOCK];
                    let mut va = 0.0f32;
                    for l in 0..FQ4_BLOCK {
                        va += q[l] * at[l];
                    }
                    acc[t] += d * va + m * asum[t * bpr + ib];
                }
            }
            acc
        })
        .collect();
    for (j, col) in cols.iter().enumerate() {
        for t in 0..t_rows {
            out[t * n + j] = col[t];
        }
    }
}

/// Auditable serial reference for [`fq4_mm_batched`]. Performs the
/// **identical floating-point op sequence** per `(t, j)` — same `asum`
/// expression, same branchless `q` unpack, same `va` left-fold, same
/// `d·va + m·asum` per-block accumulation in the same block order —
/// just serial, with no rayon and a direct write instead of the
/// column-collect/scatter. Both omissions are provably FP-neutral, so
/// `fq4_mm_batched` MUST be **byte-identical** to this (brick 2). This
/// is the lossless-stage gate: the parallel, hoisted, branchless kernel
/// equals a ~20-line scalar anyone can read.
pub fn fq4_mm_ref(a: &[f32], t_rows: usize, w: &Fq4Tensor, out: &mut [f32]) {
    let k = w.k;
    let n = w.n_rows;
    let bpr = k / FQ4_BLOCK;
    assert_eq!(a.len(), t_rows * k);
    assert!(out.len() >= t_rows * n);
    let asum: Vec<f32> = (0..t_rows)
        .flat_map(|t| {
            (0..bpr).map(move |ib| {
                let base = t * k + ib * FQ4_BLOCK;
                a[base..base + FQ4_BLOCK].iter().sum::<f32>()
            })
        })
        .collect();
    for j in 0..n {
        let code_base = j * k / 2;
        let sc_base = j * bpr;
        let mut acc = vec![0.0f32; t_rows];
        for ib in 0..bpr {
            let d = w.d[sc_base + ib];
            let m = w.m[sc_base + ib];
            let cb = code_base + ib * (FQ4_BLOCK / 2);
            let mut q = [0.0f32; FQ4_BLOCK];
            for c in 0..FQ4_BLOCK / 2 {
                let byte = w.codes[cb + c];
                q[2 * c] = (byte & 0x0F) as f32;
                q[2 * c + 1] = (byte >> 4) as f32;
            }
            let a_base = ib * FQ4_BLOCK;
            for t in 0..t_rows {
                let at = &a[t * k + a_base..t * k + a_base + FQ4_BLOCK];
                let mut va = 0.0f32;
                for l in 0..FQ4_BLOCK {
                    va += q[l] * at[l];
                }
                acc[t] += d * va + m * asum[t * bpr + ib];
            }
        }
        for t in 0..t_rows {
            out[t * n + j] = acc[t];
        }
    }
}

/// **Hand-NEON fq4 batched matmul — the speed lever the SoA/4-bit
/// layout was *designed* to enable** (and Q5_K's interleaved split-bit
/// layout structurally forbids — the documented NEON dead-end).
///
/// The nibble unpack is fully branchless and scalar-free:
/// `vld1q_u8 → vand/vshr → vmovl → vcvt → vzip` into scalar `q[64]`
/// order, **once per (col, block)**, reused across all T rows. The dot
/// is a 4-accumulator `vfmaq_f32` reduction.
///
/// **Not byte-identical to [`fq4_mm_ref`]:** `vfmaq_f32` fuses the
/// mul-add (one rounding vs the scalar two) and the 4-way partial sums
/// reorder the reduction. So the gate is *tight rel-err* vs the ref,
/// the same honest contract as brick 2's dequant→dot comparison /
/// Q5_K's `q5_k_matmul_matches_dequant_then_dot` — a perf kernel, not
/// a lossless stage. Same rayon-over-cols + scatter as
/// [`fq4_mm_batched`] so a bench isolates the *kernel*, not strategy.
#[cfg(target_arch = "aarch64")]
fn fq4_mm_batched_neon(a: &[f32], t_rows: usize, w: &Fq4Tensor, out: &mut [f32]) {
    use std::arch::aarch64::*;
    let k = w.k;
    let n = w.n_rows;
    let bpr = k / FQ4_BLOCK;
    debug_assert_eq!(a.len(), t_rows * k);
    debug_assert!(out.len() >= t_rows * n);
    debug_assert_eq!(FQ4_BLOCK, 64);

    let asum: Vec<f32> = (0..t_rows)
        .flat_map(|t| {
            (0..bpr).map(move |ib| {
                let base = t * k + ib * FQ4_BLOCK;
                a[base..base + FQ4_BLOCK].iter().sum::<f32>()
            })
        })
        .collect();

    let cols: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|j| {
            let code_base = j * k / 2;
            let sc_base = j * bpr;
            let mut acc = vec![0.0f32; t_rows];
            let mut q = [0.0f32; FQ4_BLOCK];
            for ib in 0..bpr {
                let d = w.d[sc_base + ib];
                let m = w.m[sc_base + ib];
                let cb = code_base + ib * (FQ4_BLOCK / 2); // 32 code bytes
                unsafe {
                    let lo_mask = vdupq_n_u8(0x0F);
                    // Two 16-byte halves → 64 nibbles, vzip'd into the
                    // exact scalar order q[2c]=lo, q[2c+1]=hi.
                    for h in 0..2 {
                        let bytes = vld1q_u8(w.codes.as_ptr().add(cb + h * 16));
                        let lo = vandq_u8(bytes, lo_mask);
                        let hi = vshrq_n_u8::<4>(bytes);
                        let widen = |v: uint8x16_t| -> [float32x4_t; 4] {
                            let l16 = vmovl_u8(vget_low_u8(v));
                            let h16 = vmovl_u8(vget_high_u8(v));
                            [
                                vcvtq_f32_u32(vmovl_u16(vget_low_u16(l16))),
                                vcvtq_f32_u32(vmovl_u16(vget_high_u16(l16))),
                                vcvtq_f32_u32(vmovl_u16(vget_low_u16(h16))),
                                vcvtq_f32_u32(vmovl_u16(vget_high_u16(h16))),
                            ]
                        };
                        let lof = widen(lo);
                        let hif = widen(hi);
                        for g in 0..4 {
                            let base_c = h * 16 + g * 4; // byte index c
                            let z = vzipq_f32(lof[g], hif[g]); // interleave
                            vst1q_f32(q.as_mut_ptr().add(2 * base_c), z.0);
                            vst1q_f32(q.as_mut_ptr().add(2 * base_c + 4), z.1);
                        }
                    }
                    let a_base = ib * FQ4_BLOCK;
                    for t in 0..t_rows {
                        let ap = a.as_ptr().add(t * k + a_base);
                        let mut a0 = vdupq_n_f32(0.0);
                        let mut a1 = vdupq_n_f32(0.0);
                        let mut a2 = vdupq_n_f32(0.0);
                        let mut a3 = vdupq_n_f32(0.0);
                        // 64 elems / 4 lanes = 16 vfmaq, 4 rotating accs.
                        let mut c4 = 0;
                        while c4 < 16 {
                            let qv = vld1q_f32(q.as_ptr().add(4 * c4));
                            let av = vld1q_f32(ap.add(4 * c4));
                            a0 = vfmaq_f32(a0, qv, av);
                            c4 += 1;
                            let qv = vld1q_f32(q.as_ptr().add(4 * c4));
                            let av = vld1q_f32(ap.add(4 * c4));
                            a1 = vfmaq_f32(a1, qv, av);
                            c4 += 1;
                            let qv = vld1q_f32(q.as_ptr().add(4 * c4));
                            let av = vld1q_f32(ap.add(4 * c4));
                            a2 = vfmaq_f32(a2, qv, av);
                            c4 += 1;
                            let qv = vld1q_f32(q.as_ptr().add(4 * c4));
                            let av = vld1q_f32(ap.add(4 * c4));
                            a3 = vfmaq_f32(a3, qv, av);
                            c4 += 1;
                        }
                        let va = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
                        acc[t] += d * va + m * asum[t * bpr + ib];
                    }
                }
            }
            acc
        })
        .collect();
    for (j, col) in cols.iter().enumerate() {
        for t in 0..t_rows {
            out[t * n + j] = col[t];
        }
    }
}

/// Dispatcher: the hand-NEON fq4 kernel on aarch64 (Apple Silicon —
/// NEON is baseline), the portable scalar [`fq4_mm_batched`] elsewhere.
pub fn fq4_mm_neon(a: &[f32], t_rows: usize, w: &Fq4Tensor, out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        fq4_mm_batched_neon(a, t_rows, w, out);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fq4_mm_batched(a, t_rows, w, out);
    }
}

/// **BitNet I2_S matmul — the multiply-free kernel.** This is the
/// representational payoff: weights `∈ {−1, 0, +1}` mean the dot
/// product collapses to a signed sum of activations + one final
/// scale-multiply per row. *No multiplies in the hot loop.*
///
/// `out[j] = scale · (Σ_{i: W[j,i]=+1} a[i]  −  Σ_{i: W[j,i]=−1} a[i])`
///
/// Layout (per `rce_ingest_gguf::dequant_i2_s` doc, empirically
/// verified): row `j` codes occupy bytes `[j·k/4 .. (j+1)·k/4]` in
/// 128-element interleaved blocks (byte `gp` of block `b` holds
/// positions `128b+{gp, 32+gp, 64+gp, 96+gp}` at bits `[7:6, 5:4, 3:2,
/// 1:0]`; codes `00→0, 01→+1, 10→−1`). The per-tensor f32 absmean
/// scale is a single 4-byte trailer at `w_bytes[n·k/4 .. +4]`.
///
/// **Bit-exact** to a same-factorisation scalar reference; agrees with
/// `dequant_i2_s → naive dot` to tight rel-err (brick-2/Q5_K
/// precedent: factored grouping reorders the FP reduction).
#[allow(dead_code)] // scalar fallback for non-aarch64; NEON path is the production route
fn mm_1d_i2_s(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    debug_assert!(k % 128 == 0, "i2_s mm_1d needs k divisible by 128, got {k}");
    let row_bytes = k / 4;
    let codes_total = n * row_bytes;
    debug_assert!(
        w_bytes.len() >= codes_total + 4,
        "i2_s w_bytes {} < codes_total {} + 4-byte scale",
        w_bytes.len(),
        codes_total
    );
    let scale = f32::from_le_bytes(
        w_bytes[codes_total..codes_total + 4].try_into().unwrap(),
    );
    let blocks_per_row = k / 128;

    out[..n]
        .par_iter_mut()
        .enumerate()
        .for_each(|(j, slot)| {
            let row_off = j * row_bytes;
            let mut pos = 0.0f32;
            let mut neg = 0.0f32;
            for b in 0..blocks_per_row {
                let block_off = row_off + b * 32;
                let a_base = b * 128;
                for gp in 0..32 {
                    let byte = w_bytes[block_off + gp];
                    // 4 ternary codes per byte at positions
                    // {gp, 32+gp, 64+gp, 96+gp} of the block.
                    let c0 = (byte >> 6) & 0x3;
                    let c1 = (byte >> 4) & 0x3;
                    let c2 = (byte >> 2) & 0x3;
                    let c3 = byte & 0x3;
                    // Authoritative: code 0b10 = +1 (Microsoft's encoder
                    // `q8 = src>0?2:0` → bits `2 << shift` per quantize_i2_s),
                    // code 0b00 = -1, codes 0b01 and 0b11 = 0.
                    match c0 {
                        0b10 => pos += a[a_base + gp],
                        0b00 => neg += a[a_base + gp],
                        _ => {}
                    }
                    match c1 {
                        0b10 => pos += a[a_base + 32 + gp],
                        0b00 => neg += a[a_base + 32 + gp],
                        _ => {}
                    }
                    match c2 {
                        0b10 => pos += a[a_base + 64 + gp],
                        0b00 => neg += a[a_base + 64 + gp],
                        _ => {}
                    }
                    match c3 {
                        0b10 => pos += a[a_base + 96 + gp],
                        0b00 => neg += a[a_base + 96 + gp],
                        _ => {}
                    }
                }
            }
            *slot = scale * (pos - neg);
        });
}

/// **Hand-NEON BitNet I2_S matmul — the multiply-free claim, actually
/// expressed.** The scalar [`mm_1d_i2_s`] is structurally multiply-free
/// (only `pos += a`/`neg += a` conditionals) but its `match`-on-every-
/// 2-bit-code path branches per weight, swamping the saved multiplies
/// on random ternary data (brick-1 pattern). This kernel vector-unpacks
/// each 32-byte block of 128 codes into four `uint32x4` mask lanes per
/// group (compare-equal to 1 / 2 → `0xFFFFFFFF` or `0`), then keeps the
/// activation lanes via **bitwise-AND**:
///
///   `pos_acc += vreinterpret_f32(vandq_u32(a_u32, is_pos_mask))`
///   `neg_acc += vreinterpret_f32(vandq_u32(a_u32, is_neg_mask))`
///
/// The hot loop is `vceqq_u32` + `vandq_u32` + `vaddq_f32` — **zero
/// f32 multiplies per weight.** One `scale·(pos−neg)` per output row.
/// Bit-exact to a same-factored scalar reference within tight rel-err
/// (FP reduction order reorders by 4-lane partials + horizontal sum;
/// brick-K/2 precedent).
#[cfg(target_arch = "aarch64")]
fn mm_1d_i2_s_neon(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    use std::arch::aarch64::*;
    debug_assert!(k % 128 == 0, "i2_s mm_1d_neon needs k divisible by 128, got {k}");
    let row_bytes = k / 4;
    let codes_total = n * row_bytes;
    debug_assert!(w_bytes.len() >= codes_total + 4);
    let scale = f32::from_le_bytes(
        w_bytes[codes_total..codes_total + 4].try_into().unwrap(),
    );
    let blocks_per_row = k / 128;

    out[..n]
        .par_iter_mut()
        .enumerate()
        .for_each(|(j, slot)| {
            let row_off = j * row_bytes;
            unsafe {
                let mut pos_acc = vdupq_n_f32(0.0);
                let mut neg_acc = vdupq_n_f32(0.0);
                let mask3 = vdupq_n_u8(0x3);
                // Authoritative mapping from Microsoft's `dequantize_row_i2_s`:
                // map2bit = {-1, 0, +1, 0} → 0b00=-1, 0b01=0, 0b10=+1, 0b11=0.
                // is_pos = (code == 2);  is_neg = (code == 0).
                let pos_u32 = vdupq_n_u32(2);
                let neg_u32 = vdupq_n_u32(0);
                for b in 0..blocks_per_row {
                    let bp = w_bytes.as_ptr().add(row_off + b * 32);
                    let codes_lo = vld1q_u8(bp);
                    let codes_hi = vld1q_u8(bp.add(16));
                    // 4 ternary groups within each block, at activation
                    // offsets g*32..g*32+32. Group g uses bits at shift
                    // `6-2g`: g=0→bits[7:6], g=1→[5:4], g=2→[3:2], g=3→[1:0].
                    macro_rules! handle_group {
                        ($shift:expr, $g:expr) => {{
                            // $shift in 2/4/6 — bits[1:0] handled separately below.
                            let cg_lo = vandq_u8(vshrq_n_u8::<$shift>(codes_lo), mask3);
                            let cg_hi = vandq_u8(vshrq_n_u8::<$shift>(codes_hi), mask3);
                            let widen = |v: uint8x16_t| -> [uint32x4_t; 4] {
                                let v16_lo = vmovl_u8(vget_low_u8(v));
                                let v16_hi = vmovl_u8(vget_high_u8(v));
                                [
                                    vmovl_u16(vget_low_u16(v16_lo)),
                                    vmovl_u16(vget_high_u16(v16_lo)),
                                    vmovl_u16(vget_low_u16(v16_hi)),
                                    vmovl_u16(vget_high_u16(v16_hi)),
                                ]
                            };
                            let cw_lo = widen(cg_lo);
                            let cw_hi = widen(cg_hi);
                            let a_ptr = a.as_ptr().add(b * 128 + $g * 32);
                            let all = [
                                cw_lo[0], cw_lo[1], cw_lo[2], cw_lo[3],
                                cw_hi[0], cw_hi[1], cw_hi[2], cw_hi[3],
                            ];
                            for c in 0..8 {
                                let av = vld1q_f32(a_ptr.add(c * 4));
                                let av_u = vreinterpretq_u32_f32(av);
                                let is_pos = vceqq_u32(all[c], pos_u32);
                                let is_neg = vceqq_u32(all[c], neg_u32);
                                let p = vreinterpretq_f32_u32(vandq_u32(av_u, is_pos));
                                let n_ = vreinterpretq_f32_u32(vandq_u32(av_u, is_neg));
                                pos_acc = vaddq_f32(pos_acc, p);
                                neg_acc = vaddq_f32(neg_acc, n_);
                            }
                        }};
                    }
                    handle_group!(6, 0); // bits[7:6] → activations [0..32]
                    handle_group!(4, 1); // bits[5:4] → activations [32..64]
                    handle_group!(2, 2); // bits[3:2] → activations [64..96]
                    // Group 3 (bits[1:0]) — no shift needed, just mask.
                    {
                        let cg_lo = vandq_u8(codes_lo, mask3);
                        let cg_hi = vandq_u8(codes_hi, mask3);
                        let widen = |v: uint8x16_t| -> [uint32x4_t; 4] {
                            let v16_lo = vmovl_u8(vget_low_u8(v));
                            let v16_hi = vmovl_u8(vget_high_u8(v));
                            [
                                vmovl_u16(vget_low_u16(v16_lo)),
                                vmovl_u16(vget_high_u16(v16_lo)),
                                vmovl_u16(vget_low_u16(v16_hi)),
                                vmovl_u16(vget_high_u16(v16_hi)),
                            ]
                        };
                        let cw_lo = widen(cg_lo);
                        let cw_hi = widen(cg_hi);
                        let a_ptr = a.as_ptr().add(b * 128 + 3 * 32);
                        let all = [
                            cw_lo[0], cw_lo[1], cw_lo[2], cw_lo[3],
                            cw_hi[0], cw_hi[1], cw_hi[2], cw_hi[3],
                        ];
                        for c in 0..8 {
                            let av = vld1q_f32(a_ptr.add(c * 4));
                            let av_u = vreinterpretq_u32_f32(av);
                            let is_pos = vceqq_u32(all[c], pos_u32);
                            let is_neg = vceqq_u32(all[c], neg_u32);
                            let p = vreinterpretq_f32_u32(vandq_u32(av_u, is_pos));
                            let n_ = vreinterpretq_f32_u32(vandq_u32(av_u, is_neg));
                            pos_acc = vaddq_f32(pos_acc, p);
                            neg_acc = vaddq_f32(neg_acc, n_);
                        }
                    }
                }
                let pos = vaddvq_f32(pos_acc);
                let neg = vaddvq_f32(neg_acc);
                *slot = scale * (pos - neg);
            }
        });
}

fn mm_1d_q6_k(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    const BLOCK: usize = 256;
    const BB: usize = 210;
    debug_assert!(k % BLOCK == 0, "q6_k mm_1d needs k divisible by {BLOCK}, got {k}");
    let blocks_per_row = k / BLOCK;
    let row_bytes = blocks_per_row * BB;
    out[..n].par_iter_mut().enumerate().for_each(|(j, slot)| {
        let row_start = j * row_bytes;
        let mut sumf = 0.0f32;
        for ib in 0..blocks_per_row {
            let s = row_start + ib * BB;
            let ql = &w_bytes[s..s + 128];
            let qh = &w_bytes[s + 128..s + 192];
            let scales_i8: &[i8] = bytemuck::cast_slice(&w_bytes[s + 192..s + 208]);
            let d = f16::from_le_bytes([w_bytes[s + 208], w_bytes[s + 209]]).to_f32();
            let a_super_base = ib * BLOCK;
            // 2 outer iterations of 128 outputs each; inner loop produces
            // 4 quartet positions per `l`. Matches `dequant_q6_k`.
            for outer in 0..2 {
                let ql_off = outer * 64;
                let qh_off = outer * 32;
                let sc_off = outer * 8;
                let a_off = a_super_base + outer * 128;
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[ql_off + l] & 0x0F) as i32 | (((qh[qh_off + l]) & 0x03) as i32) << 4) - 32;
                    let q2 = ((ql[ql_off + l + 32] & 0x0F) as i32 | (((qh[qh_off + l] >> 2) & 0x03) as i32) << 4) - 32;
                    let q3 = ((ql[ql_off + l] >> 4) as i32 | (((qh[qh_off + l] >> 4) & 0x03) as i32) << 4) - 32;
                    let q4 = ((ql[ql_off + l + 32] >> 4) as i32 | (((qh[qh_off + l] >> 6) & 0x03) as i32) << 4) - 32;
                    sumf += d * scales_i8[sc_off + is] as f32 * q1 as f32 * a[a_off + l];
                    sumf += d * scales_i8[sc_off + is + 2] as f32 * q2 as f32 * a[a_off + l + 32];
                    sumf += d * scales_i8[sc_off + is + 4] as f32 * q3 as f32 * a[a_off + l + 64];
                    sumf += d * scales_i8[sc_off + is + 6] as f32 * q4 as f32 * a[a_off + l + 96];
                }
            }
        }
        *slot = sumf;
    });
}

/// **PrismML Q1_0 scalar matmul.** Per-row dot product across 128-wt
/// blocks, each block = `[ggml_half d | uint8_t qs[16]]` (18 B/block).
/// Bit value 0 → −d, 1 → +d (binary). Structurally multiply-free per
/// weight (only sign-conditional adds); one `d·(pos−neg)` per block.
///
/// Authoritative ref: `dequantize_row_q1_0` in PrismML-Eng/llama.cpp
/// (`ggml/src/ggml-quants.c:415`):
/// `y[j] = ((qs[j/8] >> (j%8)) & 1) ? d : -d`.
#[allow(dead_code)] // scalar fallback for non-aarch64
fn mm_1d_q1_0(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    const QK: usize = 128;
    const BB: usize = 18; // 2 + 16
    debug_assert!(k % QK == 0, "q1_0 mm_1d needs k divisible by {QK}, got {k}");
    let blocks_per_row = k / QK;
    let row_bytes = blocks_per_row * BB;
    debug_assert!(
        w_bytes.len() >= n * row_bytes,
        "q1_0 w_bytes {} < n·row_bytes {}",
        w_bytes.len(),
        n * row_bytes
    );

    out[..n].par_iter_mut().enumerate().for_each(|(j, slot)| {
        let row_off = j * row_bytes;
        let mut sumf = 0.0f32;
        for b in 0..blocks_per_row {
            let s = row_off + b * BB;
            let d = f16::from_le_bytes([w_bytes[s], w_bytes[s + 1]]).to_f32();
            let a_base = b * QK;
            // Per-block branchless-ish signed sum: bit=1 → +a, bit=0 → −a.
            // (Decompose into pos/neg sums; multiply by d once per block.)
            let mut pos = 0.0f32;
            let mut neg = 0.0f32;
            for q in 0..(QK / 8) {
                let byte = w_bytes[s + 2 + q];
                let a_q = a_base + q * 8;
                // 8 weights per byte, LSB-first.
                for bit in 0..8 {
                    let on = (byte >> bit) & 1;
                    let av = a[a_q + bit];
                    if on != 0 { pos += av; } else { neg += av; }
                }
            }
            sumf += d * (pos - neg);
        }
        *slot = sumf;
    });
}

/// **PrismML Q2_0 scalar matmul.** Per-row dot product across 128-wt
/// blocks, each block = `[ggml_half d | uint8_t qs[32]]` (34 B/block).
/// Quaternary value mapping: 00→−d, 01→0, 10→+d, 11→+2d (i.e.
/// `y = (q-1)·d`). Structurally multiply-free per weight; one
/// `d·signed_sum` per block.
///
/// Authoritative ref: `dequantize_row_q2_0` in PrismML-Eng/llama.cpp
/// (`ggml/src/ggml-quants.c:435`):
/// `y[j] = ((int)((qs[j/4] >> ((j%4)*2)) & 3) - 1) * d`.
#[allow(dead_code)] // scalar fallback for non-aarch64
fn mm_1d_q2_0(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    const QK: usize = 128;
    const BB: usize = 34; // 2 + 32
    debug_assert!(k % QK == 0, "q2_0 mm_1d needs k divisible by {QK}, got {k}");
    let blocks_per_row = k / QK;
    let row_bytes = blocks_per_row * BB;
    debug_assert!(
        w_bytes.len() >= n * row_bytes,
        "q2_0 w_bytes {} < n·row_bytes {}",
        w_bytes.len(),
        n * row_bytes
    );

    out[..n].par_iter_mut().enumerate().for_each(|(j, slot)| {
        let row_off = j * row_bytes;
        let mut sumf = 0.0f32;
        for b in 0..blocks_per_row {
            let s = row_off + b * BB;
            let d = f16::from_le_bytes([w_bytes[s], w_bytes[s + 1]]).to_f32();
            let a_base = b * QK;
            // `(q-1) ∈ {-1, 0, +1, +2}` → split into +/- accumulators and a
            // doubled-add for q=3, then multiply by d once per block.
            let mut pos = 0.0f32;  // q=2 (mapped +d)
            let mut neg = 0.0f32;  // q=0 (mapped -d)
            let mut plus2 = 0.0f32; // q=3 (mapped +2d)
            for q in 0..(QK / 4) {
                let byte = w_bytes[s + 2 + q];
                let a_q = a_base + q * 4;
                // 4 weights per byte, LSB-first 2-bit lanes.
                for lane in 0..4 {
                    let code = (byte >> (lane * 2)) & 0x3;
                    let av = a[a_q + lane];
                    match code {
                        0 => neg += av,
                        2 => pos += av,
                        3 => plus2 += av,
                        _ => {}
                    }
                }
            }
            sumf += d * (pos - neg + 2.0 * plus2);
        }
        *slot = sumf;
    });
}

/// **Tencent Sherry STQ1_0 scalar matmul.** Per-row dot across
/// 256-weight super-blocks `{ uint8 qs[32]; uint8 sign[8]; f16 d }`
/// (42 B/block). Each group of 4 lanes is one zero + three ±d
/// (3:4 sparsity); the dot is `d·(Σ a[+1] − Σ a[−1])` — multiply-free
/// per weight. Authoritative codebook + layout from
/// `dequantize_row_stq1_0` (sjl623/llama.cpp `STQ_0`); the 32-entry
/// table is re-exported from rce-ingest-gguf as the single source
/// of truth (no second transcription).
#[allow(dead_code)] // scalar fallback for non-aarch64
fn mm_1d_stq1_0(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    use rce_ingest_gguf::STQ1_0_CODEBOOK;
    const QK: usize = 256;
    const QS_LEN: usize = QK / 8;    // 32
    const SIGN_LEN: usize = QK / 32; // 8
    const BB: usize = QS_LEN + SIGN_LEN + 2; // 42
    debug_assert!(k % QK == 0, "stq1_0 mm_1d needs k divisible by {QK}, got {k}");
    let blocks_per_row = k / QK;
    let row_bytes = blocks_per_row * BB;
    debug_assert!(
        w_bytes.len() >= n * row_bytes,
        "stq1_0 w_bytes {} < n·row_bytes {}",
        w_bytes.len(),
        n * row_bytes
    );

    out[..n].par_iter_mut().enumerate().for_each(|(j, slot)| {
        let row_off = j * row_bytes;
        let mut sumf = 0.0f32;
        for b in 0..blocks_per_row {
            let base = row_off + b * BB;
            let qs = &w_bytes[base..base + QS_LEN];
            let sgn = &w_bytes[base + QS_LEN..base + QS_LEN + SIGN_LEN];
            let d = f16::from_le_bytes([
                w_bytes[base + QS_LEN + SIGN_LEN],
                w_bytes[base + QS_LEN + SIGN_LEN + 1],
            ]).to_f32();
            let a_base = b * QK;
            let mut pos = 0.0f32;
            let mut neg = 0.0f32;
            for grp in 0..(QK / 4) {
                let code = (qs[grp / 2] >> (4 * (grp & 1))) & 0x0F;
                let s = (sgn[grp / 8] >> (grp % 8)) & 0x01;
                let qpack = STQ1_0_CODEBOOK[((s as usize) << 4) | code as usize];
                let ao = a_base + grp * 4;
                // 4 lanes, 2-bit each: 0b00→−1, 0b01→0, 0b10→+1.
                for p in 0..4 {
                    let q = (qpack >> (2 * p)) & 0x3;
                    let av = a[ao + p];
                    if q == 0b10 { pos += av; }
                    else if q == 0b00 { neg += av; }
                }
            }
            sumf += d * (pos - neg);
        }
        *slot = sumf;
    });
}

/// **Hand-NEON PrismML Q1_0 matmul.** Per-row rayon; per 128-wt block
/// `[f16 d | 16 B bits]`. Processes 8 contiguous activations per code
/// byte: broadcast the byte, `vtstq_u32` against bit-select constants
/// `{1,2,4,8}` / `{16,32,64,128}` → per-weight is-set masks, then the
/// I2_S mask-accumulate trick — `vandq` keeps set-weight activations
/// into `pos`, `vbicq` (a & ¬mask) keeps clear-weight into `neg`; one
/// `d·(Σpos − Σneg)` per block (per-block f16 scale, unlike I2_S's
/// per-tensor scale). Zero per-weight branches, zero multiplies in the
/// hot loop. Within combined-tolerance of the scalar kernel (4-lane
/// partials + horizontal sum reorder the FP reduction).
#[cfg(target_arch = "aarch64")]
fn mm_1d_q1_0_neon(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    use std::arch::aarch64::*;
    const QK: usize = 128;
    const BB: usize = 18;
    debug_assert!(k % QK == 0, "q1_0 mm_1d_neon needs k%128==0, got {k}");
    let blocks_per_row = k / QK;
    let row_bytes = blocks_per_row * BB;
    debug_assert!(w_bytes.len() >= n * row_bytes);

    out[..n].par_iter_mut().enumerate().for_each(|(j, slot)| {
        let row_off = j * row_bytes;
        unsafe {
            let bsel_lo = {
                let v: [u32; 4] = [1, 2, 4, 8];
                vld1q_u32(v.as_ptr())
            };
            let bsel_hi = {
                let v: [u32; 4] = [16, 32, 64, 128];
                vld1q_u32(v.as_ptr())
            };
            let mut sumf = 0.0f32;
            for b in 0..blocks_per_row {
                let s = row_off + b * BB;
                let d = f16::from_le_bytes([w_bytes[s], w_bytes[s + 1]]).to_f32();
                let codes = w_bytes.as_ptr().add(s + 2); // 16 bytes
                let a_blk = a.as_ptr().add(b * QK);
                let mut pos = vdupq_n_f32(0.0);
                let mut neg = vdupq_n_f32(0.0);
                for bi in 0..16usize {
                    let cb = *codes.add(bi) as u32;
                    let cbv = vdupq_n_u32(cb);
                    let set_lo = vtstq_u32(cbv, bsel_lo); // weights bi*8 + {0,1,2,3}
                    let set_hi = vtstq_u32(cbv, bsel_hi); // weights bi*8 + {4,5,6,7}
                    let ap = a_blk.add(bi * 8);
                    let alo = vreinterpretq_u32_f32(vld1q_f32(ap));
                    let ahi = vreinterpretq_u32_f32(vld1q_f32(ap.add(4)));
                    pos = vaddq_f32(pos, vreinterpretq_f32_u32(vandq_u32(alo, set_lo)));
                    pos = vaddq_f32(pos, vreinterpretq_f32_u32(vandq_u32(ahi, set_hi)));
                    // vbicq_u32(a, m) = a & ¬m  → keep clear-bit lanes.
                    neg = vaddq_f32(neg, vreinterpretq_f32_u32(vbicq_u32(alo, set_lo)));
                    neg = vaddq_f32(neg, vreinterpretq_f32_u32(vbicq_u32(ahi, set_hi)));
                }
                sumf += d * (vaddvq_f32(pos) - vaddvq_f32(neg));
            }
            *slot = sumf;
        }
    });
}

/// **Hand-NEON PrismML Q2_0 matmul.** Per 128-wt block `[f16 d | 32 B
/// codes]`, 4 weights/byte (2-bit, LSB-first), value `(q−1)·d` with
/// `q∈{0,1,2,3}`→`{−d,0,+d,+2d}`. Processes 4 contiguous activations
/// per byte: broadcast byte, extract the 4 2-bit fields into a
/// `uint32x4` via per-lane shift+mask, build `q==0/2/3` equality masks
/// (`vceqq_u32`), accumulate `neg`(q=0)/`pos`(q=2)/`plus2`(q=3) by the
/// `vandq` keep-trick; `d·(Σpos − Σneg + 2·Σplus2)` per block. No
/// per-weight branches; combined-tolerance vs scalar.
#[cfg(target_arch = "aarch64")]
fn mm_1d_q2_0_neon(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    use std::arch::aarch64::*;
    const QK: usize = 128;
    const BB: usize = 34;
    debug_assert!(k % QK == 0, "q2_0 mm_1d_neon needs k%128==0, got {k}");
    let blocks_per_row = k / QK;
    let row_bytes = blocks_per_row * BB;
    debug_assert!(w_bytes.len() >= n * row_bytes);

    out[..n].par_iter_mut().enumerate().for_each(|(j, slot)| {
        let row_off = j * row_bytes;
        unsafe {
            // Per-lane right-shift amounts for the 4 2-bit fields.
            let shifts: int32x4_t = {
                let v: [i32; 4] = [0, -2, -4, -6]; // vshlq with negative = right shift
                vld1q_s32(v.as_ptr())
            };
            let three = vdupq_n_u32(3);
            let q0 = vdupq_n_u32(0);
            let q2 = vdupq_n_u32(2);
            let q3 = vdupq_n_u32(3);
            let mut sumf = 0.0f32;
            for b in 0..blocks_per_row {
                let s = row_off + b * BB;
                let d = f16::from_le_bytes([w_bytes[s], w_bytes[s + 1]]).to_f32();
                let codes = w_bytes.as_ptr().add(s + 2); // 32 bytes
                let a_blk = a.as_ptr().add(b * QK);
                let mut pos = vdupq_n_f32(0.0);
                let mut neg = vdupq_n_f32(0.0);
                let mut p2 = vdupq_n_f32(0.0);
                for by in 0..32usize {
                    let cb = *codes.add(by) as u32;
                    // 4 codes: lane i = (cb >> (2i)) & 3.
                    let codev = vandq_u32(
                        vshlq_u32(vdupq_n_u32(cb), shifts),
                        three,
                    );
                    let ap = a_blk.add(by * 4);
                    let au = vreinterpretq_u32_f32(vld1q_f32(ap));
                    let is0 = vceqq_u32(codev, q0);
                    let is2 = vceqq_u32(codev, q2);
                    let is3 = vceqq_u32(codev, q3);
                    neg = vaddq_f32(neg, vreinterpretq_f32_u32(vandq_u32(au, is0)));
                    pos = vaddq_f32(pos, vreinterpretq_f32_u32(vandq_u32(au, is2)));
                    p2 = vaddq_f32(p2, vreinterpretq_f32_u32(vandq_u32(au, is3)));
                }
                sumf += d * (vaddvq_f32(pos) - vaddvq_f32(neg) + 2.0 * vaddvq_f32(p2));
            }
            *slot = sumf;
        }
    });
}

/// **Hand-NEON Tencent STQ1_0 matmul.** Per 256-wt super-block
/// `[32 B qs | 8 B sign | f16 d]`. The codebook step is inherently a
/// table lookup (`STQ1_0_CODEBOOK[(sign<<4)|slot]`) so each group of 4
/// is decoded scalar to its `qpack`, but the 4 lanes' contribution is
/// done branch-free: a tiny `±1/0` LUT (`MAP=[-1,0,1,0]` indexed by
/// 2-bit `q`) yields a `float32x4` sign vector, FMA'd with the 4
/// activations. `d·Σ` per block. Within combined-tolerance of scalar.
#[cfg(target_arch = "aarch64")]
fn mm_1d_stq1_0_neon(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
    use rce_ingest_gguf::STQ1_0_CODEBOOK;
    use std::arch::aarch64::*;
    const QK: usize = 256;
    const QS: usize = QK / 8; // 32
    const SG: usize = QK / 32; // 8
    const BB: usize = QS + SG + 2; // 42
    debug_assert!(k % QK == 0, "stq1_0 mm_1d_neon needs k%256==0, got {k}");
    let blocks_per_row = k / QK;
    let row_bytes = blocks_per_row * BB;
    debug_assert!(w_bytes.len() >= n * row_bytes);
    // 2-bit code → ternary: 0b00=-1, 0b01=0, 0b10=+1 (q-1).
    const MAP: [f32; 4] = [-1.0, 0.0, 1.0, 0.0];

    out[..n].par_iter_mut().enumerate().for_each(|(j, slot)| {
        let row_off = j * row_bytes;
        unsafe {
            let mut sumf = 0.0f32;
            for b in 0..blocks_per_row {
                let base = row_off + b * BB;
                let qs = w_bytes.as_ptr().add(base);
                let sg = w_bytes.as_ptr().add(base + QS);
                let d = f16::from_le_bytes([
                    w_bytes[base + QS + SG], w_bytes[base + QS + SG + 1],
                ]).to_f32();
                let a_blk = a.as_ptr().add(b * QK);
                let mut acc = vdupq_n_f32(0.0);
                for g in 0..(QK / 4) {
                    let code = (*qs.add(g / 2) >> (4 * (g & 1))) & 0x0F;
                    let sign = (*sg.add(g / 8) >> (g % 8)) & 0x01;
                    let qp = STQ1_0_CODEBOOK[((sign as usize) << 4) | code as usize];
                    // 4 lanes of ±1/0 from the packed pattern.
                    let sv: [f32; 4] = [
                        MAP[(qp & 3) as usize],
                        MAP[((qp >> 2) & 3) as usize],
                        MAP[((qp >> 4) & 3) as usize],
                        MAP[((qp >> 6) & 3) as usize],
                    ];
                    let signv = vld1q_f32(sv.as_ptr());
                    let av = vld1q_f32(a_blk.add(g * 4));
                    acc = vfmaq_f32(acc, signv, av);
                }
                sumf += d * vaddvq_f32(acc);
            }
            *slot = sumf;
        }
    });
}

// ── rce VM op: quantized matmul as a dispatchable graph node ─────
//
// This is the bridge that lets a real model's forward run **through
// the rce `.fg` IR + VM** (`interpret_with_config`) instead of
// hand-calling `mm_1d`. It's the genuinely substrate-level piece the
// runtimes were missing.
//
// **Named architectural gap (not papered over):** the VM's `Value`
// enum is f32-`HostTensor`/scalar-only — it has no first-class
// *quantized* tensor. So the zero-copy `QuantWeight`s can't ride the
// graph as ordinary `Value::Tensor` params; instead this leaf owns a
// `node_id → QuantWeight` registry and the graph's `Custom("qmm")`
// node carries only the activation. That keeps the weights zero-copy
// and the IR unchanged, at the cost of weights being leaf-state rather
// than graph-params. Making quant tensors a first-class IR value (so
// weights flow as content-addressed graph inputs) is the next real
// substrate design decision — this code makes that gap concrete and
// testable rather than hypothetical.

use rce_vm_backend_cpu::{HostTensor, InterpretError, OpDispatch, Result as VmResult, Value};
use std::collections::HashMap;

/// `OpDispatch` leaf claiming `Custom("qmm")`: one row-vector ×
/// native-dtype weight matmul per graph node, weight selected by
/// `node_id`. Register each qmm node's weight before interpreting.
#[derive(Default)]
pub struct QmmLeaf {
    weights: HashMap<usize, QuantWeight>,
}

impl QmmLeaf {
    pub fn new() -> Self {
        Self { weights: HashMap::new() }
    }

    /// Bind `node_id`'s weight. The runtime calls this for every
    /// `Custom("qmm")` node before `interpret_with_config`.
    pub fn register(&mut self, node_id: usize, w: QuantWeight) {
        self.weights.insert(node_id, w);
    }
}

impl OpDispatch for QmmLeaf {
    fn handles(&self) -> &[&'static str] {
        &["qmm"]
    }

    fn execute(&self, node_id: usize, name: &str, args: &[Value]) -> VmResult<Value> {
        if name != "qmm" {
            return Err(InterpretError::Unsupported(format!(
                "QmmLeaf does not handle `{name}`"
            )));
        }
        let act = match args.first() {
            Some(Value::Tensor(t)) => t,
            _ => {
                return Err(InterpretError::Unsupported(
                    "qmm: arg 0 must be the activation tensor".into(),
                ))
            }
        };
        let w = self.weights.get(&node_id).ok_or_else(|| {
            InterpretError::Unsupported(format!(
                "qmm: no QuantWeight registered for node {node_id}"
            ))
        })?;
        let a = act
            .as_f32()
            .map_err(|e| InterpretError::Unsupported(format!("qmm: activation not f32: {e:?}")))?;
        if a.len() != w.n_cols {
            return Err(InterpretError::Unsupported(format!(
                "qmm: activation len {} ≠ k {}",
                a.len(),
                w.n_cols
            )));
        }
        let mut out = vec![0.0f32; w.n_rows];
        mm_1d(&a, w, &mut out);
        HostTensor::from_f32(vec![1, w.n_rows], &out)
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!("qmm: build output: {e:?}")))
    }
}

/// **Converter brick 6a.** The `fq4` analog of [`QmmLeaf`]: an
/// `OpDispatch` claiming `Custom("qmm")` backed by an [`Fq4Tensor`]
/// registry keyed by `node_id`. This is what lets the **proven mixer
/// `FunctionGraph`** run with `.fgm` fq4 weights instead of GGUF
/// `QuantWeight` — the precondition for the zero-relayout loader (6c).
/// Single-row (`t_rows = 1`) call into [`fq4_mm_batched`], which is
/// itself byte-identical to the auditable [`fq4_mm_ref`] (brick 2).
#[derive(Default)]
pub struct Fq4QmmLeaf {
    weights: HashMap<usize, Fq4Tensor>,
}

impl Fq4QmmLeaf {
    pub fn new() -> Self {
        Self { weights: HashMap::new() }
    }

    /// Bind `node_id`'s fq4 weight. The loader calls this for every
    /// `Custom("qmm")` node before interpreting (mirrors `QmmLeaf`).
    pub fn register(&mut self, node_id: usize, w: Fq4Tensor) {
        self.weights.insert(node_id, w);
    }
}

impl OpDispatch for Fq4QmmLeaf {
    fn handles(&self) -> &[&'static str] {
        &["qmm"]
    }

    fn execute(&self, node_id: usize, name: &str, args: &[Value]) -> VmResult<Value> {
        if name != "qmm" {
            return Err(InterpretError::Unsupported(format!(
                "Fq4QmmLeaf does not handle `{name}`"
            )));
        }
        let act = match args.first() {
            Some(Value::Tensor(t)) => t,
            _ => {
                return Err(InterpretError::Unsupported(
                    "qmm: arg 0 must be the activation tensor".into(),
                ))
            }
        };
        let w = self.weights.get(&node_id).ok_or_else(|| {
            InterpretError::Unsupported(format!(
                "qmm: no Fq4Tensor registered for node {node_id}"
            ))
        })?;
        let a = act
            .as_f32()
            .map_err(|e| InterpretError::Unsupported(format!("qmm: activation not f32: {e:?}")))?;
        if a.len() != w.k {
            return Err(InterpretError::Unsupported(format!(
                "qmm: activation len {} ≠ k {}",
                a.len(),
                w.k
            )));
        }
        let mut out = vec![0.0f32; w.n_rows];
        fq4_mm_batched(&a, 1, w, &mut out);
        HostTensor::from_f32(vec![1, w.n_rows], &out)
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!("qmm: build output: {e:?}")))
    }
}

/// Run `out = a · w^T` **through the rce VM** — builds a 2-node `.fg`
/// `FunctionGraph` (`x → Custom("qmm") → out`) and executes it via
/// `interpret_with_config` with a `QmmLeaf`. Result is bit-identical to
/// [`mm_1d`] (same kernel; the leaf just calls it) — the difference is
/// that the computation now flows through the substrate's IR + dispatch
/// rather than a direct function call.
///
/// **This is the correctness/architecture path, not the perf path** —
/// it allocates a graph + leaf per call. A runtime opts in (e.g.
/// `RCE_MAMBA2_IR=1`) to validate that its real forward runs through
/// the VM on a real model with identical output. The perf path stays
/// `mm_1d` until first-class quant IR values + a persistent compiled
/// graph land.
pub fn mm_1d_via_vm(a: &[f32], w: &QuantWeight) -> Vec<f32> {
    use rce_vm_backend_cpu::{interpret_with_config, InterpretConfig, Inputs};
    use rce_graph::graph::{FunctionGraph, GraphNode, OpKind, Param, Wire, WireKind};

    let k = w.n_cols;
    let n = w.n_rows;
    let g = FunctionGraph {
        name: "qmm".into(),
        nodes: vec![
            GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
            GraphNode::Apply { op: OpKind::Custom("qmm".into()), output_type: "f32".into() },
        ],
        wires: vec![Wire { from: 0, to: 1, port_in_idx: 0, wire_kind: WireKind::Move }],
        output_node: 1,
        return_type: "f32".into(),
    };
    let mut leaf = QmmLeaf::new();
    leaf.register(1, w.clone());
    let mut inputs: Inputs = Inputs::new();
    inputs.insert(
        "x".into(),
        Value::Tensor(
            HostTensor::from_f32(vec![1, k], a).expect("qmm via vm: build activation"),
        ),
    );
    let ops: [&dyn OpDispatch; 1] = [&leaf];
    match interpret_with_config(&g, &inputs, InterpretConfig { leaves: &[], ops: &ops })
        .expect("qmm via vm: interpret")
    {
        Value::Tensor(t) => {
            let v = t.as_f32().expect("qmm via vm: f32 out");
            debug_assert_eq!(v.len(), n);
            v
        }
        other => panic!("qmm via vm: non-tensor result {other:?}"),
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rce_ingest_gguf::dequantize_to_f32;

    /// **The point of the whole "address the IR debt" push.** Builds a
    /// minimal `.fg` `FunctionGraph` (`x → Custom("qmm") → out`) and
    /// executes it through the *actual rce VM* (`interpret_with_config`)
    /// with `QmmLeaf` providing the zero-copy quant weight. Asserts the
    /// VM-routed result is **bit-identical** to calling `mm_1d` directly.
    ///
    /// This proves the substrate's IR + dispatch genuinely runs a
    /// quantized matmul — the op that dominates every model's forward —
    /// rather than the runtimes always hand-calling the kernel. It is
    /// the first brick of "run the real model through the IR".
    #[test]
    fn qmm_executes_through_rce_vm_bit_exact() {
        use rce_vm_backend_cpu::{interpret_with_config, InterpretConfig, Inputs};
        use rce_graph::graph::{FunctionGraph, GraphNode, OpKind, Param, Wire, WireKind};

        // Synthetic Q4_0 weight: N=5 rows × K=64 cols (2 blocks/row).
        let (n, k) = (5usize, 64usize);
        let bytes_per_row = (k / 32) * 18;
        let mut wbytes = vec![0u8; n * bytes_per_row];
        for (i, b) in wbytes.iter_mut().enumerate() {
            *b = ((i * 11 + 1) & 0xFF) as u8;
        }
        for r in 0..n {
            for blk in 0..(k / 32) {
                let base = r * bytes_per_row + blk * 18;
                let d = f16::from_f32(0.125).to_le_bytes();
                wbytes[base..base + 2].copy_from_slice(&d);
            }
        }
        let w = QuantWeight::from_owned_bytes(wbytes, GgufDtype::Q4_0, n, k);
        let a: Vec<f32> = (0..k).map(|i| ((i as f32 * 0.05) - 1.0).sin()).collect();

        // Reference: call the kernel directly.
        let mut want = vec![0.0f32; n];
        mm_1d(&a, &w, &mut want);

        // Build the 2-node graph: Param("x") → Custom("qmm").
        let g = FunctionGraph {
            name: "qmm_smoke".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("qmm".into()), output_type: "f32".into() },
            ],
            wires: vec![Wire { from: 0, to: 1, port_in_idx: 0, wire_kind: WireKind::Move }],
            output_node: 1,
            return_type: "f32".into(),
        };

        let mut leaf = QmmLeaf::new();
        leaf.register(1, w); // qmm node is index 1

        let mut inputs: Inputs = Inputs::new();
        inputs.insert(
            "x".into(),
            Value::Tensor(HostTensor::from_f32(vec![1, k], &a).unwrap()),
        );

        let ops: [&dyn OpDispatch; 1] = [&leaf];
        let result = interpret_with_config(
            &g,
            &inputs,
            InterpretConfig { leaves: &[], ops: &ops },
        )
        .expect("VM interpret");

        let got = match result {
            Value::Tensor(t) => t.as_f32().expect("f32 out"),
            other => panic!("qmm graph returned non-tensor: {other:?}"),
        };
        assert_eq!(got.len(), n);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                g, w,
                "row {i}: VM-routed qmm {g} ≠ direct mm_1d {w} (must be bit-identical)"
            );
        }
    }

    /// Build a small synthetic GGUF-format Q5_K weight matrix and check
    /// `mm_1d_q5_k` matches `dequantize_to_f32(...) → naive scalar dot`
    /// bit-for-bit (or within float-noise rounding).
    fn make_q5_k_bytes(n_super: usize) -> Vec<u8> {
        // Each superblock is 176 bytes. Fill with a deterministic pattern.
        let mut bytes = vec![0u8; n_super * 176];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ((i * 31 + 7) & 0xFF) as u8;
        }
        // Force the f16 d/dmin to small reasonable values so the dot
        // product stays bounded.
        for s in 0..n_super {
            let base = s * 176;
            // d = 0.0625, dmin = 0.03125 (representable in f16)
            let d_bytes = f16::from_f32(0.0625).to_le_bytes();
            let dmin_bytes = f16::from_f32(0.03125).to_le_bytes();
            bytes[base..base + 2].copy_from_slice(&d_bytes);
            bytes[base + 2..base + 4].copy_from_slice(&dmin_bytes);
        }
        bytes
    }

    #[test]
    fn q5_k_matmul_matches_dequant_then_dot() {
        let k = 512;  // 2 superblocks
        let n = 7;
        let bytes_per_row = (k / 256) * 176;
        let mut all_rows = Vec::with_capacity(n * bytes_per_row);
        for _ in 0..n {
            all_rows.extend(make_q5_k_bytes(k / 256));
        }

        let a: Vec<f32> = (0..k).map(|i| ((i as f32 * 0.013) - 0.7).sin()).collect();
        let mut got = vec![0.0f32; n];
        mm_1d_q5_k(&a, &all_rows, k, n, &mut got);

        // Reference: dequant each row to f32, then scalar dot.
        let mut want = vec![0.0f32; n];
        for j in 0..n {
            let row_bytes = &all_rows[j * bytes_per_row..(j + 1) * bytes_per_row];
            let row_f32 = dequantize_to_f32(row_bytes, GgufDtype::Q5K, k).unwrap();
            let mut acc = 0.0f32;
            for i in 0..k { acc += a[i] * row_f32[i]; }
            want[j] = acc;
        }

        for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let rel_err = (g - w).abs() / w.abs().max(1e-6);
            assert!(rel_err < 1e-5, "row {j}: got {g}, want {w}, rel_err {rel_err}");
        }
    }

    #[test]
    fn q3_k_matmul_matches_dequant_then_dot() {
        // Q3_K superblock = 110 B / 256 elems. Mirror the Q4_K/Q5_K test:
        // synthesize deterministic bytes, force d to a representable f16,
        // then assert mm_1d_q3_k matches `dequantize_to_f32 → scalar dot`.
        let k = 512;
        let n = 6;
        let bytes_per_row = (k / 256) * 110;
        let mut all_rows = vec![0u8; n * bytes_per_row];
        for (i, b) in all_rows.iter_mut().enumerate() {
            *b = ((i * 23 + 11) & 0xFF) as u8;
        }
        for r in 0..n {
            for s in 0..(k / 256) {
                let base = r * bytes_per_row + s * 110;
                // d (f16) at offset 108..110
                let d = f16::from_f32(0.0625).to_le_bytes();
                all_rows[base + 108] = d[0];
                all_rows[base + 109] = d[1];
            }
        }

        let a: Vec<f32> = (0..k).map(|i| ((i as f32 * 0.017) - 0.3).sin()).collect();
        let mut got = vec![0.0f32; n];
        mm_1d_q3_k(&a, &all_rows, k, n, &mut got);

        let mut want = vec![0.0f32; n];
        for j in 0..n {
            let row_bytes = &all_rows[j * bytes_per_row..(j + 1) * bytes_per_row];
            let row_f32 = dequantize_to_f32(row_bytes, GgufDtype::Q3K, k).unwrap();
            let mut acc = 0.0f32;
            for i in 0..k { acc += a[i] * row_f32[i]; }
            want[j] = acc;
        }

        for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let rel_err = (g - w).abs() / w.abs().max(1e-6);
            assert!(rel_err < 1e-5, "row {j}: got {g}, want {w}, rel_err {rel_err}");
        }
    }

    #[test]
    fn q4_k_matmul_matches_dequant_then_dot() {
        let k = 512;
        let n = 5;
        let bytes_per_row = (k / 256) * 144;
        let mut all_rows = vec![0u8; n * bytes_per_row];
        for (i, b) in all_rows.iter_mut().enumerate() {
            *b = ((i * 17 + 3) & 0xFF) as u8;
        }
        for r in 0..n {
            for s in 0..(k / 256) {
                let base = r * bytes_per_row + s * 144;
                let d = f16::from_f32(0.0625).to_le_bytes();
                let dmin = f16::from_f32(0.03125).to_le_bytes();
                all_rows[base..base + 2].copy_from_slice(&d);
                all_rows[base + 2..base + 4].copy_from_slice(&dmin);
            }
        }

        let a: Vec<f32> = (0..k).map(|i| ((i as f32 * 0.021) + 0.5).cos()).collect();
        let mut got = vec![0.0f32; n];
        mm_1d_q4_k(&a, &all_rows, k, n, &mut got);

        let mut want = vec![0.0f32; n];
        for j in 0..n {
            let row_bytes = &all_rows[j * bytes_per_row..(j + 1) * bytes_per_row];
            let row_f32 = dequantize_to_f32(row_bytes, GgufDtype::Q4K, k).unwrap();
            let mut acc = 0.0f32;
            for i in 0..k { acc += a[i] * row_f32[i]; }
            want[j] = acc;
        }

        for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let rel_err = (g - w).abs() / w.abs().max(1e-6);
            assert!(rel_err < 1e-5, "row {j}: got {g}, want {w}, rel_err {rel_err}");
        }
    }

    #[test]
    fn q4_0_matmul_matches_dequant_then_dot() {
        let k = 128;  // 4 blocks of 32
        let n = 6;
        let bytes_per_row = (k / 32) * 18;
        let mut all_rows = vec![0u8; n * bytes_per_row];
        for (i, b) in all_rows.iter_mut().enumerate() {
            *b = ((i * 11 + 1) & 0xFF) as u8;
        }
        for r in 0..n {
            for blk in 0..(k / 32) {
                let base = r * bytes_per_row + blk * 18;
                let d = f16::from_f32(0.125).to_le_bytes();
                all_rows[base..base + 2].copy_from_slice(&d);
            }
        }

        let a: Vec<f32> = (0..k).map(|i| ((i as f32 * 0.05) - 1.0).sin()).collect();
        let mut got = vec![0.0f32; n];
        mm_1d_q4_0(&a, &all_rows, k, n, &mut got);

        let mut want = vec![0.0f32; n];
        for j in 0..n {
            let row_bytes = &all_rows[j * bytes_per_row..(j + 1) * bytes_per_row];
            let row_f32 = dequantize_to_f32(row_bytes, GgufDtype::Q4_0, k).unwrap();
            let mut acc = 0.0f32;
            for i in 0..k { acc += a[i] * row_f32[i]; }
            want[j] = acc;
        }

        for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let rel_err = (g - w).abs() / w.abs().max(1e-6);
            assert!(rel_err < 1e-5, "row {j}: got {g}, want {w}, rel_err {rel_err}");
        }
    }

    /// **Brick 1 of prefill batching: prove or kill the premise cheaply,
    /// in isolation, before any runtime refactor.**
    ///
    /// (1) Correctness: `mm_1d_batched` is **bit-identical** to calling
    ///     `mm_1d` per row (same factored accumulation; weight block
    ///     just dequanted once and reused).
    /// (2) Premise: on mamba2-2.7b's in_proj shape (K=2560, N=2560 here
    ///     to keep the test quick; real N=14464), 1× `mm_1d_batched`
    ///     over T rows must be meaningfully faster than T× `mm_1d` —
    ///     because the memory-bandwidth-bound weight read is amortised
    ///     T×. Printed with `--nocapture`. If it's NOT faster, the whole
    ///     step-2 premise is wrong and we learn it here, cheaply.
    #[test]
    fn brick1_batched_q5k_bit_exact_and_faster() {
        use std::time::Instant;
        let k = 2560; // mamba2-2.7b d_model (10 superblocks)
        let n = 2560; // smaller than real 14464 to keep the test fast
        let t_rows = 32; // representative prompt-batch
        let spr = k / 256;
        let bpr = spr * 176;
        let mut w = Vec::with_capacity(n * bpr);
        for _ in 0..n {
            w.extend(make_q5_k_bytes(spr));
        }
        let qw = QuantWeight::from_owned_bytes(w, GgufDtype::Q5K, n, k);
        let a: Vec<f32> = (0..t_rows * k)
            .map(|i| ((i as f32 * 0.0007) - 0.5).sin())
            .collect();

        // Reference: T× per-row mm_1d.
        let t0 = Instant::now();
        let mut want = vec![0.0f32; t_rows * n];
        for t in 0..t_rows {
            let mut row = vec![0.0f32; n];
            mm_1d(&a[t * k..(t + 1) * k], &qw, &mut row);
            want[t * n..(t + 1) * n].copy_from_slice(&row);
        }
        let dt_unbatched = t0.elapsed();

        // Batched.
        let t1 = Instant::now();
        let mut got = vec![0.0f32; t_rows * n];
        mm_1d_batched(&a, t_rows, &qw, &mut got);
        let dt_batched = t1.elapsed();

        // (1) Bit-identical.
        for i in 0..t_rows * n {
            assert_eq!(
                got[i], want[i],
                "idx {i} (t={}, j={}): batched {} ≠ per-row mm_1d {} — must be bit-identical",
                i / n, i % n, got[i], want[i]
            );
        }

        // (2) Premise: report the speedup.
        let speedup = dt_unbatched.as_secs_f64() / dt_batched.as_secs_f64();
        eprintln!(
            "brick1 Q5_K [T={t_rows} K={k} N={n}]: T×mm_1d={dt_unbatched:?}  \
             1×mm_1d_batched={dt_batched:?}  speedup={speedup:.2}×"
        );
        assert!(
            speedup > 1.5,
            "prefill-batching premise WEAK: only {speedup:.2}× (expected ≫1.5× from \
             T× weight-read amortisation). If this fails, rethink step 2 before any refactor."
        );
    }

    /// Brick 4 prep: the dtype-dispatching [`mm_batched`] must be
    /// **byte-identical** to looping `mm_1d` per row — for the Q5_K
    /// *fast path* AND the non-Q5_K *fallback path* — so the prefill
    /// driver can call it dtype-agnostically and stay bit-exact vs the
    /// per-token reference regardless of the model's weight format.
    #[test]
    fn mm_batched_byte_identical_to_per_row_for_q5k_and_q4k() {
        let k = 512; // 2 superblocks
        let n = 6;
        let t_rows = 4;
        let a: Vec<f32> = (0..t_rows * k)
            .map(|i| ((i as f32 * 0.011) - 0.5).sin())
            .collect();

        for &(dt, bb) in &[(GgufDtype::Q5K, 176usize), (GgufDtype::Q4K, 144usize)] {
            let bytes_per_row = (k / 256) * bb;
            let mut wb = vec![0u8; n * bytes_per_row];
            for (i, b) in wb.iter_mut().enumerate() {
                *b = ((i * 31 + 7) & 0xFF) as u8;
            }
            for r in 0..n {
                for s in 0..(k / 256) {
                    let base = r * bytes_per_row + s * bb;
                    wb[base..base + 2]
                        .copy_from_slice(&f16::from_f32(0.0625).to_le_bytes());
                    wb[base + 2..base + 4]
                        .copy_from_slice(&f16::from_f32(0.03125).to_le_bytes());
                }
            }
            let w = QuantWeight::from_owned_bytes(wb, dt, n, k);

            let mut want = vec![0.0f32; t_rows * n];
            for t in 0..t_rows {
                mm_1d(&a[t * k..(t + 1) * k], &w, &mut want[t * n..(t + 1) * n]);
            }
            let mut got = vec![0.0f32; t_rows * n];
            mm_batched(&a, t_rows, &w, &mut got);

            for (i, (g, wv)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    g.to_bits(), wv.to_bits(),
                    "{dt:?} elem {i} (t={}, j={}): mm_batched {g} ≠ per-row mm_1d {wv} \
                     — must be bit-identical",
                    i / n, i % n
                );
            }
        }
    }

    /// **Converter brick 1, gate 1 — the lossy stage, measured vs f32.**
    /// Always-on, deterministic, load-independent. `fq4` round-trip
    /// error is bounded *provably*: every element reconstructs within
    /// `d/2` (exact f32 step) + tiny f32-round slack. A broken quantiser
    /// fails here; a correct affine 4-bit one cannot exceed it. Confirms
    /// the agreed split correctness model (quant measured vs f32, not
    /// vs GGUF).
    #[test]
    fn brick1_fq4_roundtrip_vs_f32_bounded() {
        let k = 2560usize;
        let n = 256usize; // enough rows to exercise varied (d, m); fast
        let wf: Vec<f32> = (0..n * k)
            .map(|i| {
                let x = i as f32;
                (x * 0.000113).sin() * 1.7 + (x * 0.0007).cos() * 0.4 - 0.2
            })
            .collect();
        let q = fq4_quantize(&wf, n, k);
        let bpr = k / FQ4_BLOCK;
        let global_absmax = wf.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let mut max_abs_err = 0.0f32;
        let mut sum_abs_err = 0.0f64;
        for r in 0..n {
            let deq = fq4_dequant_row(&q, r);
            for b in 0..bpr {
                let dq = q.d[r * bpr + b];
                let bound = dq * 0.5 + global_absmax * 1e-6 + 1e-6;
                for l in 0..FQ4_BLOCK {
                    let idx = b * FQ4_BLOCK + l;
                    let e = (wf[r * k + idx] - deq[idx]).abs();
                    assert!(
                        e <= bound,
                        "fq4 round-trip: row {r} block {b} elem {l}: err {e} > bound {bound} \
                         (d={dq}) — quantiser is broken"
                    );
                    max_abs_err = max_abs_err.max(e);
                    sum_abs_err += e as f64;
                }
            }
        }
        eprintln!(
            "brick1 fq4 round-trip vs f32: max_abs={max_abs_err:.5} \
             mean_abs={:.6} (|w|max={global_absmax:.3}, 5.0 bpw)",
            sum_abs_err / (n * k) as f64
        );
    }

    /// **Converter brick 1, gate 2 — the layout thesis (perf).**
    /// `#[ignore]` ON PURPOSE: this is a wall-clock comparison with a
    /// ~1.16× margin; cargo runs test fns concurrently, so under the
    /// parallel suite the ±25% machine noise (documented in the perf
    /// memory) swamps the signal and it flaps. Run it deliberately, in
    /// isolation, single-threaded:
    ///
    /// ```text
    /// cargo test -p rce-op-qmm --release brick1_fq4_beats_q5k_speed \
    ///   -- --ignored --exact --test-threads=1 --nocapture
    /// ```
    ///
    /// **Honest finding:** on this box the scalar fq4 kernel is at
    /// *parity* with Q5_K within the measurement noise — runs have
    /// landed 1.00×–1.16× (bandwidth-normalised 0.91–1.05×); the
    /// ±25% sustained-load variance is wider than min-of-9 denoises.
    /// So this is **not** a confirm-the-win gate (that needs a human
    /// read on a quiet machine). It is a **regression guard**: it
    /// fails only if fq4 goes *materially slower* than Q5_K (a real
    /// kernel regression), and otherwise just reports the number.
    /// The format's real speed payoff is the hand-NEON/Metal kernel
    /// the SoA/branchless layout *enables* (a later brick); brick 1's
    /// solid, deterministic wins are accuracy (gate 1) + 9% smaller.
    #[test]
    #[ignore = "wall-clock perf; run isolated single-threaded (see doc)"]
    fn brick1_fq4_beats_q5k_speed() {
        use std::time::Instant;
        let k = 2560usize;
        let n = 2560usize;
        let t_rows = 32usize;
        let wf: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32 * 0.000113).sin() * 1.7 + (i as f32 * 0.0007).cos() * 0.4 - 0.2))
            .collect();
        let q = fq4_quantize(&wf, n, k);
        let a: Vec<f32> = (0..t_rows * k)
            .map(|i| ((i as f32 * 0.0007) - 0.5).sin())
            .collect();
        let spr = k / 256;
        let mut q5 = Vec::with_capacity(n * spr * 176);
        for _ in 0..n {
            q5.extend(make_q5_k_bytes(spr));
        }
        let q5w = QuantWeight::from_owned_bytes(q5, GgufDtype::Q5K, n, k);

        const ITERS: usize = 9;
        let mut o5 = vec![0.0f32; t_rows * n];
        let mut o4 = vec![0.0f32; t_rows * n];
        mm_batched(&a, t_rows, &q5w, &mut o5);
        fq4_mm_batched(&a, t_rows, &q, &mut o4);
        let mut min_q5k = f64::INFINITY;
        let mut min_fq4 = f64::INFINITY;
        for _ in 0..ITERS {
            let t0 = Instant::now();
            mm_batched(&a, t_rows, &q5w, &mut o5);
            min_q5k = min_q5k.min(t0.elapsed().as_secs_f64());
            let t1 = Instant::now();
            fq4_mm_batched(&a, t_rows, &q, &mut o4);
            min_fq4 = min_fq4.min(t1.elapsed().as_secs_f64());
        }
        let speedup = min_q5k / min_fq4;
        eprintln!(
            "brick1 layout thesis [K={k} N={n} T={t_rows}, min of {ITERS}]: \
             Q5_K={:.3}ms (5.5bpw)  fq4={:.3}ms (5.0bpw)  speedup={speedup:.2}×  \
             (bandwidth-normalised {:.2}×) — PARITY within noise; human-read on a \
             quiet machine to judge a real win",
            min_q5k * 1e3, min_fq4 * 1e3, speedup * (5.0 / 5.5)
        );
        // Regression guard only (NOT a confirm-the-win bar — see doc).
        // v1's bad geometry sat at 0.88×; a materially-slower fq4 means
        // a real kernel regression and must fail. Parity ± noise passes.
        assert!(
            speedup > 0.92,
            "fq4 KERNEL REGRESSION: {speedup:.2}× vs Q5_K (materially slower — \
             not noise). Something broke fq4_mm_batched; investigate before \
             proceeding. (This is a regression floor, not the win bar.)"
        );
    }

    /// **Converter brick 6a — `Fq4QmmLeaf` lossless gate.** The leaf the
    /// loader will register fq4 weights into must produce output
    /// **byte-identical** to the auditable [`fq4_mm_ref`] (single row),
    /// and fail cleanly (no panic) on an unregistered node / wrong k.
    #[test]
    fn brick6a_fq4_qmm_leaf_bit_exact_and_clean_errors() {
        use rce_vm_backend_cpu::{HostTensor, OpDispatch, Value};

        let k = 192usize; // 3 fq4 blocks
        let n = 7usize;
        let wf: Vec<f32> = (0..n * k)
            .map(|i| (i as f32 * 0.019).sin() * 0.8 - 0.1)
            .collect();
        let q = fq4_quantize(&wf, n, k);
        let a: Vec<f32> = (0..k).map(|i| ((i as f32 * 0.013) - 0.3).cos()).collect();

        let mut leaf = Fq4QmmLeaf::new();
        leaf.register(0, q.clone());
        let act = Value::Tensor(HostTensor::from_f32(vec![k], &a).unwrap());
        let got = match leaf.execute(0, "qmm", &[act]).expect("execute") {
            Value::Tensor(t) => t.as_f32().expect("f32"),
            o => panic!("non-tensor {o:?}"),
        };
        let mut want = vec![0.0f32; n];
        fq4_mm_ref(&a, 1, &q, &mut want);
        assert_eq!(got.len(), n);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                g.to_bits(), w.to_bits(),
                "Fq4QmmLeaf out[{i}] {g} ≠ fq4_mm_ref {w} (must be byte-identical)"
            );
        }

        // Clean errors, no panic.
        let act2 = Value::Tensor(HostTensor::from_f32(vec![k], &a).unwrap());
        assert!(leaf.execute(99, "qmm", &[act2]).is_err(), "unregistered node");
        let bad = Value::Tensor(HostTensor::from_f32(vec![k + 64], &vec![0.0; k + 64]).unwrap());
        assert!(leaf.execute(0, "qmm", &[bad]).is_err(), "wrong-k must error");
    }

    /// **Brick K (the speed lever) — hand-NEON fq4 correctness.**
    /// `fq4_mm_neon` vs the auditable `fq4_mm_ref`: tight rel-err, NOT
    /// byte-identical (vfmaq fuses rounding + reorders the reduction —
    /// a perf kernel, the brick-2/Q5_K precedent). Always-on,
    /// deterministic.
    #[test]
    fn brickk_fq4_mm_neon_matches_ref_within_tol() {
        let k = 256usize; // 4 fq4 blocks
        let n = 9usize;
        let t_rows = 5usize;
        let wf: Vec<f32> = (0..n * k)
            .map(|i| (i as f32 * 0.017).sin() * 0.9 + (i as f32 * 0.003).cos() * 0.3 - 0.1)
            .collect();
        let q = fq4_quantize(&wf, n, k);
        let a: Vec<f32> = (0..t_rows * k)
            .map(|i| ((i as f32 * 0.011) - 0.4).sin())
            .collect();

        let mut got = vec![0.0f32; t_rows * n];
        fq4_mm_neon(&a, t_rows, &q, &mut got);
        let mut want = vec![0.0f32; t_rows * n];
        fq4_mm_ref(&a, t_rows, &q, &mut want);

        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let rel = (g - w).abs() / w.abs().max(1e-4);
            assert!(
                rel < 1e-4,
                "fq4_mm_neon[{i}] {g} vs ref {w} rel {rel} (FMA tol, not bit-exact)"
            );
        }
    }

    /// **Brick K — does the hand-NEON fq4 kernel finally beat Q5_K?**
    /// The original "faster" question. `#[ignore]` + min-of-N like
    /// brick-1's perf test (single shots are ±25% noise on this box;
    /// run isolated/single-threaded). Regression-guard assert only;
    /// the real number is for a human read on a quiet machine:
    /// `cargo test -p rce-op-qmm --release brickk_fq4_neon_vs_q5k_speed -- --ignored --test-threads=1 --nocapture`
    #[test]
    #[ignore = "wall-clock perf; run isolated single-threaded (see doc)"]
    fn brickk_fq4_neon_vs_q5k_speed() {
        use std::time::Instant;
        let k = 2560usize;
        let n = 2560usize;
        let t_rows = 32usize;
        let wf: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32 * 0.000113).sin() * 1.7 + (i as f32 * 0.0007).cos() * 0.4 - 0.2))
            .collect();
        let q = fq4_quantize(&wf, n, k);
        let a: Vec<f32> = (0..t_rows * k).map(|i| ((i as f32 * 0.0007) - 0.5).sin()).collect();
        let spr = k / 256;
        let mut q5 = Vec::with_capacity(n * spr * 176);
        for _ in 0..n {
            q5.extend(make_q5_k_bytes(spr));
        }
        let q5w = QuantWeight::from_owned_bytes(q5, GgufDtype::Q5K, n, k);

        const ITERS: usize = 9;
        let mut o5 = vec![0.0f32; t_rows * n];
        let mut on = vec![0.0f32; t_rows * n];
        mm_batched(&a, t_rows, &q5w, &mut o5);
        fq4_mm_neon(&a, t_rows, &q, &mut on);
        let mut min_q5k = f64::INFINITY;
        let mut min_neon = f64::INFINITY;
        for _ in 0..ITERS {
            let t0 = Instant::now();
            mm_batched(&a, t_rows, &q5w, &mut o5);
            min_q5k = min_q5k.min(t0.elapsed().as_secs_f64());
            let t1 = Instant::now();
            fq4_mm_neon(&a, t_rows, &q, &mut on);
            min_neon = min_neon.min(t1.elapsed().as_secs_f64());
        }
        let sp = min_q5k / min_neon;
        eprintln!(
            "brickK NEON [K={k} N={n} T={t_rows}, min {ITERS}]: \
             Q5_K mm_batched={:.3}ms (5.5bpw)  fq4_NEON={:.3}ms (5.0bpw)  \
             speedup={sp:.2}×  (bw-norm {:.2}×) — human-read on a quiet box",
            min_q5k * 1e3, min_neon * 1e3, sp * (5.0 / 5.5)
        );
        assert!(
            sp > 0.80,
            "fq4 NEON REGRESSION: {sp:.2}× vs Q5_K (materially slower — \
             a real kernel regression, not noise). Investigate."
        );
    }

    /// **Converter brick 2 — lossless-stage gate on the fq4 kernel.**
    ///
    /// (1) `fq4_mm_batched` (rayon over columns, hoisted `asum`,
    ///     branchless unpack) MUST be **byte-identical** to
    ///     `fq4_mm_ref` (the same FP op sequence, serial, direct write).
    ///     Proves the parallelisation + output-scatter are provably
    ///     FP-neutral — the perf kernel == an auditable scalar.
    ///
    /// (2) Both MUST agree with the **naive dequant-row→sequential-dot**
    ///     within tight rel-err — but NOT bit-exactly: the factored form
    ///     `d·Σ(aq)+m·Σa` regroups the reduction, and FP add isn't
    ///     associative. Documents the expected, bounded reorder (same
    ///     precedent as `q5_k_matmul_matches_dequant_then_dot`).
    #[test]
    fn brick2_fq4_mm_batched_bit_exact_vs_scalar_ref() {
        let k = 128usize; // 2 fq4 blocks of 64
        let n = 5usize;
        let t_rows = 3usize;

        // Varied weights so blocks get distinct (d, m).
        let wf: Vec<f32> = (0..n * k)
            .map(|i| {
                let x = i as f32;
                (x * 0.017).sin() * 0.9 + (x * 0.003).cos() * 0.3 - 0.15
            })
            .collect();
        let q = fq4_quantize(&wf, n, k);
        let a: Vec<f32> = (0..t_rows * k)
            .map(|i| ((i as f32 * 0.011) - 0.4).sin())
            .collect();

        let mut got = vec![0.0f32; t_rows * n];
        fq4_mm_batched(&a, t_rows, &q, &mut got);
        let mut refv = vec![0.0f32; t_rows * n];
        fq4_mm_ref(&a, t_rows, &q, &mut refv);

        // (1) Byte-identical to the auditable scalar reference.
        for (i, (g, r)) in got.iter().zip(refv.iter()).enumerate() {
            assert_eq!(
                g.to_bits(), r.to_bits(),
                "fq4 elem {i} (t={}, j={}): kernel {g} ≠ scalar ref {r} \
                 — must be bit-identical (parallelisation is not FP-neutral?)",
                i / n, i % n
            );
        }

        // (2) Numerically agrees with naive dequant→dot (NOT bit-exact;
        // documents the factored-reduction reorder).
        for j in 0..n {
            let wh = fq4_dequant_row(&q, j);
            for t in 0..t_rows {
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += a[t * k + i] * wh[i];
                }
                let g = got[t * n + j];
                let rel = (g - acc).abs() / acc.abs().max(1e-6);
                assert!(
                    rel < 1e-4,
                    "fq4 (t={t}, j={j}): kernel {g} vs dequant→dot {acc} \
                     rel_err {rel} (expected close, not bit-exact)"
                );
            }
        }
    }

    // ── T2: BitNet I2_S multiply-free matmul ────────────────────────

    /// Build an I2_S weight blob `[n_rows, k]` from a known ternary
    /// matrix `wt` (values in {−1,0,+1}) plus a single f32 `scale`.
    fn pack_i2_s(wt: &[i8], n_rows: usize, k: usize, scale: f32) -> Vec<u8> {
        assert_eq!(wt.len(), n_rows * k);
        assert!(k % 128 == 0);
        let row_bytes = k / 4;
        let mut bytes = vec![0u8; n_rows * row_bytes + 4];
        // Authoritative mapping: code 0b00=-1, 0b01=0, 0b10=+1.
        let to_code = |v: i8| match v {
            -1 => 0b00u8,
            0 => 0b01,
            1 => 0b10,
            _ => unreachable!(),
        };
        for j in 0..n_rows {
            for b in 0..(k / 128) {
                let blk = j * k + b * 128;
                for gp in 0..32 {
                    let c0 = to_code(wt[blk + gp]);
                    let c1 = to_code(wt[blk + 32 + gp]);
                    let c2 = to_code(wt[blk + 64 + gp]);
                    let c3 = to_code(wt[blk + 96 + gp]);
                    bytes[j * row_bytes + b * 32 + gp] =
                        (c0 << 6) | (c1 << 4) | (c2 << 2) | c3;
                }
            }
        }
        bytes[n_rows * row_bytes..n_rows * row_bytes + 4]
            .copy_from_slice(&scale.to_le_bytes());
        bytes
    }

    /// Auditable serial reference for `mm_1d_i2_s` — same factored
    /// `(pos − neg) · scale` grouping, just serial / direct-write.
    /// FP-neutral vs the rayon kernel → MUST be byte-identical.
    fn mm_1d_i2_s_ref(a: &[f32], w_bytes: &[u8], k: usize, n: usize, out: &mut [f32]) {
        let row_bytes = k / 4;
        let codes_total = n * row_bytes;
        let scale = f32::from_le_bytes(
            w_bytes[codes_total..codes_total + 4].try_into().unwrap(),
        );
        for j in 0..n {
            let mut pos = 0.0f32;
            let mut neg = 0.0f32;
            for b in 0..(k / 128) {
                let block_off = j * row_bytes + b * 32;
                let a_base = b * 128;
                for gp in 0..32 {
                    let byte = w_bytes[block_off + gp];
                    // Authoritative mapping: code 0b10=+1, 0b00=-1.
                    let handle = |code: u8, idx: usize, p: &mut f32, n_: &mut f32| match code & 3 {
                        0b10 => *p += a[idx],
                        0b00 => *n_ += a[idx],
                        _ => {}
                    };
                    handle(byte >> 6, a_base + gp, &mut pos, &mut neg);
                    handle(byte >> 4, a_base + 32 + gp, &mut pos, &mut neg);
                    handle(byte >> 2, a_base + 64 + gp, &mut pos, &mut neg);
                    handle(byte, a_base + 96 + gp, &mut pos, &mut neg);
                }
            }
            out[j] = scale * (pos - neg);
        }
    }

    /// **T2, gate 1 — `mm_1d_i2_s` lossless.** rayon kernel
    /// byte-identical to a same-factorisation serial reference (Call /
    /// per-row parallelism is FP-neutral here). Plus tight rel-err vs
    /// dequant→naive-dot (different reduction order; same precedent as
    /// brick 2's fq4 vs dequant→dot).
    #[test]
    fn t2_i2_s_mm_bit_exact_vs_ref_and_close_to_dequant_dot() {
        use rce_ingest_gguf::{dequantize_to_f32, GgufDtype};
        // 2 rows × 256 cols (= 2 blocks/row) — small but exercises
        // every code group + multiple rows for the par_iter path.
        let (k, n) = (256usize, 2usize);
        let scale = 0.125f32;
        let wt: Vec<i8> = (0..n * k)
            .map(|i| match (i * 17 + 3) % 3 {
                0 => 0,
                1 => 1,
                _ => -1,
            })
            .collect();
        let wb = pack_i2_s(&wt, n, k, scale);
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.011 - 0.3).sin()).collect();

        let mut got = vec![0.0f32; n];
        mm_1d_i2_s(&a, &wb, k, n, &mut got);
        let mut want_ref = vec![0.0f32; n];
        mm_1d_i2_s_ref(&a, &wb, k, n, &mut want_ref);
        // (1) byte-identical to the auditable ref.
        for (i, (g, r)) in got.iter().zip(want_ref.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                r.to_bits(),
                "mm_1d_i2_s row {i}: kernel {g} ≠ scalar ref {r}"
            );
        }
        // (2) close to dequant→naive-dot per row (factored reorder).
        let _ = GgufDtype::I2S; // anchor import path
        for j in 0..n {
            let row_bytes = k / 4;
            // Build a per-row I2_S buffer (row's codes + the f32 scale)
            // so `dequant_i2_s(.., k)` decodes exactly k elements.
            let mut row_buf = vec![0u8; row_bytes + 4];
            row_buf[..row_bytes].copy_from_slice(&wb[j * row_bytes..(j + 1) * row_bytes]);
            row_buf[row_bytes..row_bytes + 4]
                .copy_from_slice(&scale.to_le_bytes());
            let row_f32 = dequantize_to_f32(&row_buf, GgufDtype::I2S, k)
                .expect("I2_S dequant");
            let mut naive = 0.0f32;
            for i in 0..k {
                naive += a[i] * row_f32[i];
            }
            let rel = (got[j] - naive).abs() / naive.abs().max(1e-6);
            // Tight rel-err, not bit-exact (factored grouping reorders
            // the FP reduction — same precedent as brick-2 fq4-vs-dequant→dot
            // which also uses 1e-4 for k of this scale).
            assert!(
                rel < 1e-4,
                "row {j}: kernel {} vs dequant→dot {naive} rel {rel}",
                got[j]
            );
        }
    }

    /// **T2, gate 2 — the multiply-free speed claim.** `#[ignore]`,
    /// run isolated/single-threaded (brick-K methodology, ±25% noise
    /// floor). At a BitNet d_model=2560 shape, mm_1d_i2_s should
    /// substantially beat a 5.5-bpw Q5_K matmul: ternary reads ~2.75×
    /// less weight memory AND eliminates the in-loop multiplies.
    /// Regression guard only (real win is for a human read on a quiet
    /// box; same honest tooling as brick-1 perf).
    #[test]
    #[ignore = "wall-clock perf; run isolated single-threaded (see doc)"]
    fn t2_i2_s_speed_vs_q5k() {
        use std::time::Instant;
        let k = 2560usize;
        let n = 2560usize;
        let scale = 1.0f32;
        // Synthetic ternary weights (random-ish via cheap PRG).
        let wt: Vec<i8> = (0..n * k)
            .map(|i| match (i.wrapping_mul(2654435761) >> 30) & 0x3 {
                0 => 0,
                1 => 1,
                2 => -1,
                _ => 0,
            })
            .collect();
        let wb_i2s = pack_i2_s(&wt, n, k, scale);
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.0007 - 0.5).sin()).collect();

        // Q5_K baseline at same shape.
        let spr = k / 256;
        let mut q5 = Vec::with_capacity(n * spr * 176);
        for _ in 0..n {
            q5.extend(make_q5_k_bytes(spr));
        }
        let q5w = QuantWeight::from_owned_bytes(q5, GgufDtype::Q5K, n, k);

        const ITERS: usize = 9;
        let mut o_i2s = vec![0.0f32; n];
        let mut o_q5k = vec![0.0f32; n];
        mm_1d_i2_s(&a, &wb_i2s, k, n, &mut o_i2s); // warm
        mm_1d(&a, &q5w, &mut o_q5k);                // warm
        let mut min_i2s = f64::INFINITY;
        let mut min_q5k = f64::INFINITY;
        for _ in 0..ITERS {
            let t0 = Instant::now();
            mm_1d_i2_s(&a, &wb_i2s, k, n, &mut o_i2s);
            min_i2s = min_i2s.min(t0.elapsed().as_secs_f64());
            let t1 = Instant::now();
            mm_1d(&a, &q5w, &mut o_q5k);
            min_q5k = min_q5k.min(t1.elapsed().as_secs_f64());
        }
        let sp = min_q5k / min_i2s;
        eprintln!(
            "T2 BitNet I2_S [K={k} N={n}, min {ITERS}]: \
             Q5_K mm_1d={:.3}ms (5.5bpw)  I2_S mm_1d={:.3}ms (2.0bpw)  \
             speedup={sp:.2}×  (bw-norm {:.2}×) — human-read on a quiet box",
            min_q5k * 1e3, min_i2s * 1e3, sp * (2.0 / 5.5)
        );
        assert!(
            sp > 0.80,
            "I2_S REGRESSION vs Q5_K: {sp:.2}× (materially slower — \
             unexpected for adds-only vs 5.5-bpw multiply matmul)"
        );
    }

    /// **T2.b, gate 1 — NEON I2_S correctness.** The hand-vectorized
    /// `mm_1d_i2_s_neon` must agree with the audited scalar
    /// `mm_1d_i2_s` to tight rel-err — NOT byte-identical, because
    /// the vector accumulators sum 4 lanes in parallel and then
    /// horizontally reduce, reordering the FP additions (brick-K / T2
    /// precedent on factored reductions). Multiple rows + a real-ish
    /// k (256, 2 blocks/row) exercises the bit-mask path on every code
    /// class (0, +1, −1).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn t2_b_i2_s_neon_matches_scalar_within_tol() {
        let (k, n) = (256usize, 5usize);
        let scale = 0.0625f32;
        let wt: Vec<i8> = (0..n * k)
            .map(|i| match (i * 23 + 1) % 3 {
                0 => 0,
                1 => 1,
                _ => -1,
            })
            .collect();
        let wb = pack_i2_s(&wt, n, k, scale);
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013 - 0.4).sin()).collect();

        let mut got = vec![0.0f32; n];
        mm_1d_i2_s_neon(&a, &wb, k, n, &mut got);
        let mut want = vec![0.0f32; n];
        mm_1d_i2_s(&a, &wb, k, n, &mut want);
        for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            // **Combined tolerance** — pure rel-err is the wrong metric
            // when pos & neg contributions nearly cancel (|w| → 0): the
            // absolute error stays at f32-rounding scale (~k·ulp) but the
            // relative blows up against a tiny denominator. Use the
            // larger of an absolute floor and a relative bound; both
            // mathematically equivalent kernels must satisfy this.
            let abs_err = (g - w).abs();
            let bound = 1e-4 * w.abs().max(1.0); // ≥ 1e-4 abs OR ≥ 1e-4 rel
            assert!(
                abs_err <= bound,
                "row {j}: NEON {g} vs scalar {w} abs_err {abs_err} > bound {bound}"
            );
        }
    }

    /// **T2.b, gate 2 — does the NEON kernel actually express the
    /// multiply-free claim?** Same shape as T2's bench (K=N=2560).
    /// Expectation: substantially beats both Q5_K and the scalar I2_S.
    /// `#[ignore]` regression-guard only — the real number is a
    /// human-read on a quiet box (brick-1 / brick-K methodology).
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore = "wall-clock perf; run isolated single-threaded (see doc)"]
    fn t2_b_i2_s_neon_speed() {
        use std::time::Instant;
        let k = 2560usize;
        let n = 2560usize;
        let scale = 1.0f32;
        let wt: Vec<i8> = (0..n * k)
            .map(|i| match (i.wrapping_mul(2654435761) >> 30) & 0x3 {
                0 => 0,
                1 => 1,
                2 => -1,
                _ => 0,
            })
            .collect();
        let wb_i2s = pack_i2_s(&wt, n, k, scale);
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.0007 - 0.5).sin()).collect();
        let spr = k / 256;
        let mut q5 = Vec::with_capacity(n * spr * 176);
        for _ in 0..n {
            q5.extend(make_q5_k_bytes(spr));
        }
        let q5w = QuantWeight::from_owned_bytes(q5, GgufDtype::Q5K, n, k);

        const ITERS: usize = 9;
        let mut o_neon = vec![0.0f32; n];
        let mut o_scalar = vec![0.0f32; n];
        let mut o_q5k = vec![0.0f32; n];
        mm_1d_i2_s_neon(&a, &wb_i2s, k, n, &mut o_neon);
        mm_1d_i2_s(&a, &wb_i2s, k, n, &mut o_scalar);
        mm_1d(&a, &q5w, &mut o_q5k);
        let (mut min_neon, mut min_scalar, mut min_q5k) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
        for _ in 0..ITERS {
            let t0 = Instant::now();
            mm_1d_i2_s_neon(&a, &wb_i2s, k, n, &mut o_neon);
            min_neon = min_neon.min(t0.elapsed().as_secs_f64());
            let t1 = Instant::now();
            mm_1d_i2_s(&a, &wb_i2s, k, n, &mut o_scalar);
            min_scalar = min_scalar.min(t1.elapsed().as_secs_f64());
            let t2 = Instant::now();
            mm_1d(&a, &q5w, &mut o_q5k);
            min_q5k = min_q5k.min(t2.elapsed().as_secs_f64());
        }
        let sp_vs_q5k = min_q5k / min_neon;
        let sp_vs_scalar = min_scalar / min_neon;
        eprintln!(
            "T2.b BitNet I2_S NEON [K={k} N={n}, min {ITERS}]: \
             Q5_K mm_1d={:.3}ms (5.5bpw)  scalar I2_S={:.3}ms (2.0bpw)  \
             NEON I2_S={:.3}ms (2.0bpw)  → NEON vs Q5_K={sp_vs_q5k:.2}×  \
             (bw-norm {:.2}×)  NEON vs scalar={sp_vs_scalar:.2}×  \
             — human-read on a quiet box",
            min_q5k * 1e3, min_scalar * 1e3, min_neon * 1e3,
            sp_vs_q5k * (2.0 / 5.5)
        );
        assert!(
            sp_vs_q5k > 0.80,
            "NEON I2_S REGRESSION vs Q5_K: {sp_vs_q5k:.2}× — investigate"
        );
    }

    /// Pack PrismML Q1_0: per-block [`ggml_half d` (2 B) | bits (16 B)].
    /// `sign[i] ∈ {-1, +1}` → bit 0 / 1; encoder mirrors
    /// `quantize_row_q1_0_ref`.
    fn pack_q1_0(sign: &[i8], n_rows: usize, k: usize, d_per_row: f32) -> Vec<u8> {
        const QK: usize = 128;
        const BB: usize = 18;
        assert_eq!(sign.len(), n_rows * k);
        assert!(k % QK == 0);
        let blocks_per_row = k / QK;
        let row_bytes = blocks_per_row * BB;
        let mut bytes = vec![0u8; n_rows * row_bytes];
        let d_f16 = f16::from_f32(d_per_row);
        for j in 0..n_rows {
            for b in 0..blocks_per_row {
                let s = j * row_bytes + b * BB;
                bytes[s..s + 2].copy_from_slice(&d_f16.to_le_bytes());
                let blk = j * k + b * QK;
                for w in 0..QK {
                    if sign[blk + w] >= 0 {
                        bytes[s + 2 + w / 8] |= 1 << (w % 8);
                    }
                }
            }
        }
        bytes
    }

    /// Pack PrismML Q2_0: per-block [`ggml_half d` (2 B) | codes (32 B)].
    /// `q[i] ∈ {-1, 0, +1, +2}` → 2-bit code `q+1 ∈ {0,1,2,3}`.
    fn pack_q2_0(qv: &[i8], n_rows: usize, k: usize, d_per_row: f32) -> Vec<u8> {
        const QK: usize = 128;
        const BB: usize = 34;
        assert_eq!(qv.len(), n_rows * k);
        assert!(k % QK == 0);
        let blocks_per_row = k / QK;
        let row_bytes = blocks_per_row * BB;
        let mut bytes = vec![0u8; n_rows * row_bytes];
        let d_f16 = f16::from_f32(d_per_row);
        for j in 0..n_rows {
            for b in 0..blocks_per_row {
                let s = j * row_bytes + b * BB;
                bytes[s..s + 2].copy_from_slice(&d_f16.to_le_bytes());
                let blk = j * k + b * QK;
                for w in 0..QK {
                    let code = (qv[blk + w] + 1) as u8;
                    bytes[s + 2 + w / 4] |= code << ((w % 4) * 2);
                }
            }
        }
        bytes
    }

    /// **Q1_0 matmul correctness gate.** Pack known ±1 weights with a
    /// known per-block scale, run `mm_1d_q1_0`, compare to
    /// `dequant_q1_0 → naive dot` (combined-tolerance bound, brick-K
    /// precedent — pure rel-err breaks when contributions cancel).
    /// Mixed positive/negative pattern covers both bit values.
    #[test]
    fn t41_q1_0_mm_matches_dequant_dot() {
        use rce_ingest_gguf::{dequantize_to_f32, GgufDtype};
        let (k, n) = (256usize, 3usize);
        let d = 0.0625f32;
        let sign: Vec<i8> = (0..n * k)
            .map(|i| if (i * 7 + 3) % 2 == 0 { 1i8 } else { -1 })
            .collect();
        let wb = pack_q1_0(&sign, n, k, d);
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.011 - 0.3).sin()).collect();

        let mut got = vec![0.0f32; n];
        mm_1d_q1_0(&a, &wb, k, n, &mut got);

        let blocks_per_row = k / 128;
        for j in 0..n {
            let row_bytes = blocks_per_row * 18;
            let row_buf = &wb[j * row_bytes..(j + 1) * row_bytes];
            let row_f32 = dequantize_to_f32(row_buf, GgufDtype::Q1_0, k)
                .expect("Q1_0 dequant");
            let mut naive = 0.0f32;
            for i in 0..k {
                naive += a[i] * row_f32[i];
            }
            let abs_err = (got[j] - naive).abs();
            let bound = 1e-4 * naive.abs().max(1.0);
            assert!(
                abs_err <= bound,
                "Q1_0 row {j}: kernel {} vs dequant→dot {naive} abs_err {abs_err} > {bound}",
                got[j]
            );
        }
    }

    /// **Q2_0 matmul correctness gate.** Pack all 4 codes
    /// {00=-1, 01=0, 10=+1, 11=+2} so a code-mapping rotation would
    /// trip — same discipline as the BitNet I2_S 4-way bug. Compare
    /// `mm_1d_q2_0` to `dequant_q2_0 → naive dot` with combined-tolerance.
    #[test]
    fn t42_q2_0_mm_matches_dequant_dot() {
        use rce_ingest_gguf::{dequantize_to_f32, GgufDtype};
        let (k, n) = (256usize, 3usize);
        let d = 0.125f32;
        // All 4 codes covered; cycle is offset by row so each row exercises
        // the codes against different activation positions.
        let qv: Vec<i8> = (0..n * k).map(|i| (i % 4) as i8 - 1).collect();
        let wb = pack_q2_0(&qv, n, k, d);
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013 - 0.4).sin()).collect();

        let mut got = vec![0.0f32; n];
        mm_1d_q2_0(&a, &wb, k, n, &mut got);

        let blocks_per_row = k / 128;
        for j in 0..n {
            let row_bytes = blocks_per_row * 34;
            let row_buf = &wb[j * row_bytes..(j + 1) * row_bytes];
            let row_f32 = dequantize_to_f32(row_buf, GgufDtype::Q2_0, k)
                .expect("Q2_0 dequant");
            let mut naive = 0.0f32;
            for i in 0..k {
                naive += a[i] * row_f32[i];
            }
            let abs_err = (got[j] - naive).abs();
            let bound = 1e-4 * naive.abs().max(1.0);
            assert!(
                abs_err <= bound,
                "Q2_0 row {j}: kernel {} vs dequant→dot {naive} abs_err {abs_err} > {bound}",
                got[j]
            );
        }
    }

    /// **Q1_0 + Q2_0 dispatch via the high-level `mm_1d`.** Confirms
    /// the routing wiring is correct (mirror of the I2_S dispatch test).
    #[test]
    fn t41_42_mm_1d_dispatch() {
        let (k, n) = (128usize, 2usize);
        let d1 = 0.0625f32;
        let sign: Vec<i8> = (0..n * k).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let wb1 = pack_q1_0(&sign, n, k, d1);
        let a: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();
        let qw1 = QuantWeight::from_owned_bytes(wb1, GgufDtype::Q1_0, n, k);
        let mut o1 = vec![0.0f32; n];
        mm_1d(&a, &qw1, &mut o1);
        // Sanity: outputs are finite, non-zero in general.
        for v in &o1 {
            assert!(v.is_finite(), "Q1_0 dispatch produced non-finite {v}");
        }

        let d2 = 0.125f32;
        let qv: Vec<i8> = (0..n * k).map(|i| (i % 4) as i8 - 1).collect();
        let wb2 = pack_q2_0(&qv, n, k, d2);
        let qw2 = QuantWeight::from_owned_bytes(wb2, GgufDtype::Q2_0, n, k);
        let mut o2 = vec![0.0f32; n];
        mm_1d(&a, &qw2, &mut o2);
        for v in &o2 {
            assert!(v.is_finite(), "Q2_0 dispatch produced non-finite {v}");
        }
    }

    /// Pack a 256-weight STQ1_0 super-block grid by choosing valid
    /// codebook (sign, slot) per group → guaranteed 3:4-sparse ternary.
    fn pack_stq1_0(n_rows: usize, k: usize, d: f32) -> Vec<u8> {
        const QK: usize = 256;
        const QS_LEN: usize = QK / 8;
        const SIGN_LEN: usize = QK / 32;
        const BB: usize = QS_LEN + SIGN_LEN + 2;
        assert!(k % QK == 0);
        let blocks_per_row = k / QK;
        let row_bytes = blocks_per_row * BB;
        let mut bytes = vec![0u8; n_rows * row_bytes];
        let d_f16 = f16::from_f32(d);
        for j in 0..n_rows {
            for b in 0..blocks_per_row {
                let base = j * row_bytes + b * BB;
                for grp in 0..(QK / 4) {
                    let idx = (grp * 5 + j + b) % 32;
                    let s = (idx >> 4) as u8 & 1;
                    let slot = (idx & 0x0F) as u8;
                    bytes[base + grp / 2] |= slot << (4 * (grp & 1));
                    if s != 0 { bytes[base + QS_LEN + grp / 8] |= 1 << (grp % 8); }
                }
                bytes[base + QS_LEN + SIGN_LEN..base + BB]
                    .copy_from_slice(&d_f16.to_le_bytes());
            }
        }
        bytes
    }

    /// **STQ1_0 matmul correctness gate.** Compare `mm_1d_stq1_0` to
    /// `dequant_stq1_0 → naive dot` with combined-tolerance. Exercises
    /// the codebook + 3:4-sparse signed-sum path; a codebook or
    /// byte-layout error fails here (not 8h later in a gen test).
    #[test]
    fn t42_stq1_0_mm_matches_dequant_dot() {
        use rce_ingest_gguf::{dequantize_to_f32, GgufDtype};
        let (k, n) = (512usize, 3usize); // 2 super-blocks/row, 3 rows
        let d = 0.0625f32;
        let wb = pack_stq1_0(n, k, d);
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.017 - 0.5).sin()).collect();

        let mut got = vec![0.0f32; n];
        mm_1d_stq1_0(&a, &wb, k, n, &mut got);

        let blocks_per_row = k / 256;
        for j in 0..n {
            let row_bytes = blocks_per_row * 42;
            let row_buf = &wb[j * row_bytes..(j + 1) * row_bytes];
            let row_f32 = dequantize_to_f32(row_buf, GgufDtype::Stq1_0, k)
                .expect("STQ1_0 dequant");
            let mut naive = 0.0f32;
            for i in 0..k { naive += a[i] * row_f32[i]; }
            let abs_err = (got[j] - naive).abs();
            let bound = 1e-4 * naive.abs().max(1.0);
            assert!(
                abs_err <= bound,
                "STQ1_0 row {j}: kernel {} vs dequant→dot {naive} abs_err {abs_err} > {bound}",
                got[j]
            );
        }
        // Dispatch wiring via high-level mm_1d.
        let qw = QuantWeight::from_owned_bytes(pack_stq1_0(n, k, d), GgufDtype::Stq1_0, n, k);
        let mut o = vec![0.0f32; n];
        mm_1d(&a, &qw, &mut o);
        for (j, v) in o.iter().enumerate() {
            assert!((v - got[j]).abs() < 1e-4, "dispatch mismatch row {j}");
        }
    }

    /// Combined-tolerance bound (brick-K precedent): abs error must be
    /// ≤ max(1e-4·|ref|, 1e-4). Pure rel-err is the wrong metric when
    /// pos/neg cancel toward zero; pure abs-err is wrong when |ref|≫1.
    #[cfg(target_arch = "aarch64")]
    fn within_tol(got: &[f32], want: &[f32], label: &str) {
        for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let abs_err = (g - w).abs();
            let bound = 1e-4 * w.abs().max(1.0);
            assert!(
                abs_err <= bound,
                "{label} row {j}: NEON {g} vs scalar {w} abs_err {abs_err} > {bound}"
            );
        }
    }

    /// **Q1_0 NEON ≡ scalar.** Multi-row, mixed ±1, 2 blocks/row —
    /// exercises every bit position and the per-block f16 scale.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn t41_q1_0_neon_matches_scalar() {
        let (k, n) = (256usize, 5usize);
        let d = 0.0625f32;
        let sign: Vec<i8> = (0..n * k)
            .map(|i| if (i * 11 + 5) % 3 == 0 { 1i8 } else { -1 })
            .collect();
        let wb = pack_q1_0(&sign, n, k, d);
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.017 - 0.4).sin()).collect();
        let mut g = vec![0.0f32; n];
        let mut s = vec![0.0f32; n];
        mm_1d_q1_0_neon(&a, &wb, k, n, &mut g);
        mm_1d_q1_0(&a, &wb, k, n, &mut s);
        within_tol(&g, &s, "Q1_0");
    }

    /// **Q2_0 NEON ≡ scalar.** Pattern covers all 4 codes {−1,0,+1,+2}
    /// so the +2 (`plus2`) accumulator path is exercised.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn t42_q2_0_neon_matches_scalar() {
        let (k, n) = (256usize, 5usize);
        let d = 0.125f32;
        let qv: Vec<i8> = (0..n * k).map(|i| (i % 4) as i8 - 1).collect();
        let wb = pack_q2_0(&qv, n, k, d);
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013 - 0.5).sin()).collect();
        let mut g = vec![0.0f32; n];
        let mut s = vec![0.0f32; n];
        mm_1d_q2_0_neon(&a, &wb, k, n, &mut g);
        mm_1d_q2_0(&a, &wb, k, n, &mut s);
        within_tol(&g, &s, "Q2_0");
    }

    /// **Q8_0 NEON ≡ scalar.** Synthetic Q8_0 weight (varied scales +
    /// full int8 range), checks the NEON dot matches the scalar
    /// reference within f32-reorder tolerance.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q8_0_neon_matches_scalar() {
        let (k, n) = (256usize, 5usize);
        const BB: usize = 34;
        let blocks_per_row = k / 32;
        let mut wb = vec![0u8; n * blocks_per_row * BB];
        for r in 0..n {
            for blk in 0..blocks_per_row {
                let off = (r * blocks_per_row + blk) * BB;
                let scale = 0.05f32 + (r as f32) * 0.013 + (blk as f32) * 0.002;
                let s = half::f16::from_f32(scale).to_le_bytes();
                wb[off] = s[0];
                wb[off + 1] = s[1];
                for i in 0..32 {
                    let v = (((r * 7 + blk * 13 + i * 5) as i32) % 256 - 128) as i8;
                    wb[off + 2 + i] = v as u8;
                }
            }
        }
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.017 - 0.5).sin()).collect();
        let mut g = vec![0.0f32; n];
        let mut s = vec![0.0f32; n];
        mm_1d_q8_0_neon(&a, &wb, k, n, &mut g);
        mm_1d_q8_0(&a, &wb, k, n, &mut s);
        within_tol(&g, &s, "Q8_0");
    }

    /// **Q5_0 NEON ≡ scalar.** Exercises the `qh` 5th-bit expansion for
    /// both nibble halves across multiple blocks.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q5_0_neon_matches_scalar() {
        let (k, n) = (256usize, 5usize);
        const BB: usize = 22;
        let blocks_per_row = k / 32;
        let mut wb = vec![0u8; n * blocks_per_row * BB];
        for r in 0..n {
            for blk in 0..blocks_per_row {
                let off = (r * blocks_per_row + blk) * BB;
                let scale = 0.05f32 + (r as f32) * 0.011 + (blk as f32) * 0.003;
                let s = half::f16::from_f32(scale).to_le_bytes();
                wb[off] = s[0];
                wb[off + 1] = s[1];
                let qh: u32 = 0x1357_9BDF ^ ((r as u32).wrapping_mul(2_654_435_761) >> (blk as u32 & 7));
                wb[off + 2..off + 6].copy_from_slice(&qh.to_le_bytes());
                for q in 0..16 {
                    let lo = ((r + blk + q) % 16) as u8;
                    let hi = ((r * 3 + q * 2) % 16) as u8;
                    wb[off + 6 + q] = lo | (hi << 4);
                }
            }
        }
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.021 - 0.4).cos()).collect();
        let mut g = vec![0.0f32; n];
        let mut s = vec![0.0f32; n];
        mm_1d_q5_0_neon(&a, &wb, k, n, &mut g);
        mm_1d_q5_0(&a, &wb, k, n, &mut s);
        within_tol(&g, &s, "Q5_0");
    }

    /// In-cache compute microbench: NEON vs scalar for Q5_0 and Q8_0 on
    /// a realistic expert shape (n=1856 rows, k=2688). Isolates compute
    /// (weights stay resident across iterations) to show the NEON win
    /// that single-token MoE *decode* can't realize because decode is
    /// weight-read-bandwidth-bound. Run:
    /// `cargo test --release -p rce-op-qmm neon_q5q8_compute_bench -- --ignored --nocapture`
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore]
    fn neon_q5q8_compute_bench() {
        let (k, n) = (2688usize, 1856usize);
        let bpr = k / 32;
        // Q5_0 weight.
        let mut q5 = vec![0u8; n * bpr * 22];
        for r in 0..n { for blk in 0..bpr {
            let off = (r * bpr + blk) * 22;
            let sb = half::f16::from_f32(0.06 + (r % 7) as f32 * 0.01).to_le_bytes();
            q5[off] = sb[0]; q5[off + 1] = sb[1];
            let qh = (r as u32).wrapping_mul(2_654_435_761) ^ (blk as u32);
            q5[off + 2..off + 6].copy_from_slice(&qh.to_le_bytes());
            for q in 0..16 { q5[off + 6 + q] = ((r + blk + q) % 256) as u8; }
        }}
        // Q8_0 weight.
        let mut q8 = vec![0u8; n * bpr * 34];
        for r in 0..n { for blk in 0..bpr {
            let off = (r * bpr + blk) * 34;
            let sb = half::f16::from_f32(0.05 + (r % 5) as f32 * 0.01).to_le_bytes();
            q8[off] = sb[0]; q8[off + 1] = sb[1];
            for i in 0..32 { q8[off + 2 + i] = (((r + blk + i) % 256) as i32 - 128) as u8; }
        }}
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).sin()).collect();
        let mut out = vec![0.0f32; n];
        let iters = 200;
        type Mm = fn(&[f32], &[u8], usize, usize, &mut [f32]);
        let cases: [(&str, Mm, Mm); 2] = [
            ("Q5_0", mm_1d_q5_0_neon, mm_1d_q5_0),
            ("Q8_0", mm_1d_q8_0_neon, mm_1d_q8_0),
        ];
        for (name, neon, scalar) in cases {
            let wb = if name == "Q5_0" { &q5 } else { &q8 };
            let t = std::time::Instant::now();
            for _ in 0..iters { scalar(&a, wb, k, n, &mut out); }
            let ts = t.elapsed().as_secs_f64() / iters as f64;
            let t = std::time::Instant::now();
            for _ in 0..iters { neon(&a, wb, k, n, &mut out); }
            let tn = t.elapsed().as_secs_f64() / iters as f64;
            eprintln!("{name}: scalar={:.3}ms  neon={:.3}ms  speedup={:.2}×", ts * 1e3, tn * 1e3, ts / tn);
        }
    }

    /// `mm_1d_serial` ≡ `mm_1d` for every supported dtype (the MoE
    /// coarse-parallel path must match the row-parallel path exactly).
    #[test]
    fn mm_1d_serial_matches_parallel() {
        let (k, n) = (512usize, 7usize);
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.017 - 0.3).sin()).collect();
        let dtypes = [
            (GgufDtype::Q8_0, 34usize, 32usize),
            (GgufDtype::Q5_0, 22, 32),
            (GgufDtype::Q4K, 144, 256),
            (GgufDtype::Q5K, 176, 256),
            (GgufDtype::Q6K, 210, 256),
        ];
        let sc = half::f16::from_f32(0.05).to_le_bytes(); // finite scale
        for (dt, bb, blk) in dtypes {
            let per_row = (k / blk) * bb;
            let mut wb = vec![0u8; n * per_row];
            for (i, b) in wb.iter_mut().enumerate() { *b = (i.wrapping_mul(37).wrapping_add(11)) as u8; }
            // Patch valid f16 scale field(s) per block so no NaN/Inf f16
            // sneaks out of the random payload.
            let nblk = (k / blk) * n;
            for ibk in 0..nblk {
                let s = ibk * bb;
                match dt {
                    GgufDtype::Q6K => { wb[s + 208] = sc[0]; wb[s + 209] = sc[1]; }
                    GgufDtype::Q4K | GgufDtype::Q5K => {
                        wb[s] = sc[0]; wb[s + 1] = sc[1]; wb[s + 2] = sc[0]; wb[s + 3] = sc[1];
                    }
                    _ => { wb[s] = sc[0]; wb[s + 1] = sc[1]; } // Q8_0, Q5_0
                }
            }
            let w = QuantWeight::from_owned_bytes(wb, dt, n, k);
            let mut par = vec![0.0f32; n];
            let mut ser = vec![0.0f32; n];
            mm_1d(&a, &w, &mut par);
            mm_1d_serial(&a, &w, &mut ser);
            for j in 0..n {
                assert!((par[j] - ser[j]).abs() < 1e-3, "{dt:?} row {j}: par={} ser={}", par[j], ser[j]);
            }
        }
    }

    /// Characterize scalar K-quant matmul at MoE expert shape (n=512,
    /// k=2048) in-cache: GFLOP/s + effective GB/s tell us whether these
    /// are compute-bound (NEON has headroom) or already memory/auto-vec
    /// saturated. Run:
    /// `cargo test --release -p rce-op-qmm kquant_compute_profile -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn kquant_compute_profile() {
        let (k, n) = (2048usize, 512usize);
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).sin()).collect();
        let mut out = vec![0.0f32; n];
        let macs = (k * n) as f64;
        // (name, bytes/superblock, kernel)
        type Mm = fn(&[f32], &[u8], usize, usize, &mut [f32]);
        let cases: [(&str, usize, Mm); 3] = [
            ("Q4_K", 144, mm_1d_q4_k),
            ("Q5_K", 176, mm_1d_q5_k),
            ("Q6_K", 210, mm_1d_q6_k),
        ];
        for (name, bb, kernel) in cases {
            let sb_per_row = k / 256;
            let mut wb = vec![0u8; n * sb_per_row * bb];
            for (i, b) in wb.iter_mut().enumerate() { *b = (i * 37 + 11) as u8; }
            let iters = 500;
            // warm
            kernel(&a, &wb, k, n, &mut out);
            let t = std::time::Instant::now();
            for _ in 0..iters { kernel(&a, &wb, k, n, &mut out); }
            let dt = t.elapsed().as_secs_f64() / iters as f64;
            let gflops = 2.0 * macs / dt / 1e9;
            let gbs = wb.len() as f64 / dt / 1e9;
            eprintln!("{name} (rayon mm_1d): {:.4}ms/call  {:.1} GFLOP/s  {:.1} GB/s",
                dt * 1e3, gflops, gbs);
        }
        // Serial-vs-parallel: dotting a few rows on ONE thread, no rayon.
        // If serial ≈ rayon mm_1d, the per-call fork/join overhead — not
        // compute — is the MoE decode bottleneck.
        {
            let sb_per_row = k / 256;
            let mut wb = vec![0u8; n * sb_per_row * 176]; // Q5_K
            for (i, b) in wb.iter_mut().enumerate() { *b = (i * 37 + 11) as u8; }
            let mut out_s = vec![0.0f32; n];
            let iters = 500;
            let serial = |out: &mut [f32]| {
                let row_bytes = sb_per_row * 176;
                for j in 0..n {
                    let mut tmp = [0.0f32; 1];
                    // reuse the kernel on a 1-row view to get pure serial work
                    mm_1d_q5_k(&a, &wb[j * row_bytes..(j + 1) * row_bytes], k, 1, &mut tmp);
                    out[j] = tmp[0];
                }
            };
            serial(&mut out_s);
            let t = std::time::Instant::now();
            for _ in 0..iters { serial(&mut out_s); }
            let dt = t.elapsed().as_secs_f64() / iters as f64;
            eprintln!("Q5_K (serial per-row, n calls of n=1): {:.4}ms/call  {:.1} GFLOP/s",
                dt * 1e3, 2.0 * macs / dt / 1e9);
        }
    }

    /// **STQ1_0 NEON ≡ scalar.** Full codebook coverage (every
    /// (sign,slot)), 2 super-blocks/row.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn t42_stq1_0_neon_matches_scalar() {
        let (k, n) = (512usize, 4usize);
        let d = 0.0625f32;
        let wb = pack_stq1_0(n, k, d);
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.019 - 0.3).sin()).collect();
        let mut g = vec![0.0f32; n];
        let mut s = vec![0.0f32; n];
        mm_1d_stq1_0_neon(&a, &wb, k, n, &mut g);
        mm_1d_stq1_0(&a, &wb, k, n, &mut s);
        within_tol(&g, &s, "STQ1_0");
    }

    /// `mm_1d_q5_0` ≡ dequantize-then-dot. Packs a synthetic Q5_0
    /// weight (exercising the 5th-bit `qh` field for both nibble halves)
    /// and checks the matmul against the canonical dequant reference.
    #[test]
    fn mm_1d_q5_0_matches_dequant() {
        use rce_ingest_gguf::dequant::dequantize_to_f32;
        let (k, n) = (64usize, 3usize);
        const BB: usize = 22;
        let blocks_per_row = k / 32;
        let mut bytes = vec![0u8; n * blocks_per_row * BB];
        for r in 0..n {
            for blk in 0..blocks_per_row {
                let off = (r * blocks_per_row + blk) * BB;
                let scale = 0.0625f32 + (r as f32) * 0.01 + (blk as f32) * 0.003;
                let s = half::f16::from_f32(scale).to_le_bytes();
                bytes[off] = s[0];
                bytes[off + 1] = s[1];
                // qh: set a spread of 5th bits across both halves.
                let qh: u32 = 0x1357_9BDF ^ ((r as u32) << (blk as u32 & 7));
                bytes[off + 2..off + 6].copy_from_slice(&qh.to_le_bytes());
                for q in 0..16 {
                    let lo = ((r + blk + q) % 16) as u8;
                    let hi = ((r * 2 + q) % 16) as u8;
                    bytes[off + 6 + q] = lo | (hi << 4);
                }
            }
        }
        let w = QuantWeight::from_owned_bytes(bytes.clone(), GgufDtype::Q5_0, n, k);
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.023 - 0.4).cos()).collect();

        let mut got = vec![0.0f32; n];
        mm_1d(&a, &w, &mut got);

        // Reference: dequant each row, manual dot.
        let row_bytes = blocks_per_row * BB;
        for r in 0..n {
            let deq = dequantize_to_f32(&bytes[r * row_bytes..(r + 1) * row_bytes], GgufDtype::Q5_0, k).unwrap();
            let dot: f32 = a.iter().zip(deq.iter()).map(|(x, w)| x * w).sum();
            assert!((got[r] - dot).abs() < 1e-4, "row {r}: mm={} ref={dot}", got[r]);
        }
    }

    /// **AMX `mm_batched` fast path ≡ default `mm_batched`.** When the
    /// `amx` feature is on AND the shape qualifies (Q8_0, M ≥ 16,
    /// N % 16 == 0, K % 32 == 0), `mm_batched` routes through
    /// `rce-op-amx`. This must produce results within f32-reorder
    /// tolerance of the per-row mm_1d fallback. Synthetic 32×64 Q8_0
    /// (2 M-chunks of 16, plus 16-divisible N at exactly 16 — testing
    /// the M-tiling).
    #[cfg(feature = "amx")]
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    #[test]
    fn amx_mm_batched_matches_default() {
        let m = 32;
        let n = 16;
        let k = 64;
        let blocks_per_row = k / 32;
        let bytes_per_row = blocks_per_row * 34;
        let mut bytes = vec![0u8; n * bytes_per_row];
        for r in 0..n {
            for blk in 0..blocks_per_row {
                let off = r * bytes_per_row + blk * 34;
                let scale = 0.125f32 + (r as f32) * 0.002 + (blk as f32) * 0.0005;
                let s = half::f16::from_f32(scale).to_le_bytes();
                bytes[off] = s[0];
                bytes[off + 1] = s[1];
                for i in 0..32 {
                    let v = (((r + blk * 32 + i) as i32) % 17 - 8) as i8;
                    bytes[off + 2 + i] = v as u8;
                }
            }
        }
        let w = QuantWeight::from_owned_bytes(bytes, GgufDtype::Q8_0, n, k);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.011).sin()).collect();

        // Reference: per-row mm_1d.
        let mut out_ref = vec![0.0f32; m * n];
        for mi in 0..m {
            let mut row = vec![0.0f32; n];
            mm_1d(&a[mi * k..(mi + 1) * k], &w, &mut row);
            out_ref[mi * n..(mi + 1) * n].copy_from_slice(&row);
        }

        // AMX-routed mm_batched.
        let mut out_amx = vec![0.0f32; m * n];
        mm_batched(&a, m, &w, &mut out_amx);

        let mut max_abs = 0.0f32;
        for (m_v, d_v) in out_amx.iter().zip(out_ref.iter()) {
            let abs = (m_v - d_v).abs();
            if abs > max_abs { max_abs = abs; }
        }
        eprintln!("amx_mm_batched: max_abs_err={max_abs:.3e}");
        assert!(max_abs < 1e-3, "AMX-routed mm_batched drifted: max_abs={max_abs:.3e}");
    }

    /// End-to-end perf check: `mm_batched` with the `amx` feature on
    /// vs the per-row mm_1d fallback. Same shape as `rce-op-amx`'s
    /// own prefill bench (M=16, N=K=2048).
    ///
    /// `cargo test -p rce-op-qmm --features amx mm_batched_amx_bench --release -- --ignored --nocapture`
    #[cfg(feature = "amx")]
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    #[test]
    #[ignore]
    fn mm_batched_amx_bench() {
        let m = 16;
        let n = 2048;
        let k = 2048;
        let blocks_per_row = k / 32;
        let bytes_per_row = blocks_per_row * 34;
        let mut bytes = vec![0u8; n * bytes_per_row];
        for r in 0..n {
            for blk in 0..blocks_per_row {
                let off = r * bytes_per_row + blk * 34;
                let scale = 0.125f32;
                let s = half::f16::from_f32(scale).to_le_bytes();
                bytes[off] = s[0];
                bytes[off + 1] = s[1];
                for i in 0..32 {
                    let v = (((r + blk * 32 + i) as i32) % 17 - 8) as i8;
                    bytes[off + 2 + i] = v as u8;
                }
            }
        }
        let w = QuantWeight::from_owned_bytes(bytes, GgufDtype::Q8_0, n, k);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();

        // Warmup AMX path.
        let mut out = vec![0.0f32; m * n];
        mm_batched(&a, m, &w, &mut out);

        let iters = 10;
        let total_flops = (2 * m * n * k * iters) as f64;

        // Per-row mm_1d (baseline; matches the M < 16 path).
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            for mi in 0..m {
                mm_1d(&a[mi * k..(mi + 1) * k], &w, &mut out[mi * n..(mi + 1) * n]);
            }
        }
        let dt_baseline = t0.elapsed();
        let gflops_baseline = total_flops / dt_baseline.as_secs_f64() / 1e9;

        // AMX-routed mm_batched.
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            mm_batched(&a, m, &w, &mut out);
        }
        let dt_amx = t1.elapsed();
        let gflops_amx = total_flops / dt_amx.as_secs_f64() / 1e9;

        eprintln!(
            "mm_batched amx-feature bench (M={m}, N={n}, K={k}, {iters} iters):\n  \
             per-row mm_1d:          {dt_baseline:?}, {gflops_baseline:.1} GFLOPS\n  \
             mm_batched (amx feat):  {dt_amx:?}, {gflops_amx:.1} GFLOPS\n  \
             speedup: {:.2}x",
            gflops_amx / gflops_baseline,
        );
    }
}
