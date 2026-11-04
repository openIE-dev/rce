//! `rce-op-qmm-metal` — native Metal qmm kernels.
//!
//! Formats supported:
//! - **Q8_0** (32-elem blocks, f16 scale + 32 i8 weights, 34 B/block)
//! - **Q4_K** (256-elem super-blocks, f16 d/dmin + 12 B packed 6-bit
//!   scales/mins + 128 B 4-bit weights, 144 B/block)
//! - **Q5_K** (Q4_K layout + 32 B of high-bits, 176 B/block)
//! - **Q6K**  (256-elem super-blocks, 128 B ql + 64 B qh + 16 B i8
//!   sub-scales + f16 d, 210 B/block)
//!
//! Each format gets its own MSL pipeline. One thread per output row;
//! walks the row's blocks, dequants on the fly, accumulates. Matches
//! the corresponding `rce_op_qmm::mm_1d_{q8_0,q4_k,q5_k,q6_k}` CPU
//! reference up to f32 reduction order.
//!
//! ## Why a separate Metal qmm path
//!
//! A GPU qmm that goes through a general graph runtime pays graph build,
//! scheduling and substrate discovery on every call. This crate is the
//! **direct** path: one
//! pre-compiled MSL pipeline per device, dispatch is just buffer-fill +
//! commit + wait. Designed to slot in as a third `PricedBackend` impl in
//! `rce-placement-adapter` so the cost selector can route per-call
//! between CPU (`rce_op_qmm::mm_1d`) and native Metal (this crate).
//!
//! ## Cross-platform
//!
//! Non-Apple targets compile to a stub: `MetalQmmContext::new()` returns
//! `None`. Workspace stays cross-target clean.

use rce_ingest_gguf::GgufDtype;
use rce_op_qmm::QuantWeight;

/// Direct Metal qmm dispatcher. Cheap to share via `Arc`; one device +
/// queue + per-shape buffer cache for the lifetime of the context.
pub struct MetalQmmContext {
    #[cfg(rce_qmm_metal)]
    inner: imp::Inner,
    #[cfg(not(rce_qmm_metal))]
    _phantom: std::marker::PhantomData<()>,
}

impl MetalQmmContext {
    /// Bind to the system default Metal device + compile the Q8_0 GEMV
    /// pipeline. Returns `None` on non-Apple targets, sandboxed CI, or
    /// any Metal failure (compile error, no device).
    pub fn new() -> Option<Self> {
        #[cfg(rce_qmm_metal)]
        { imp::Inner::new().map(|inner| Self { inner }) }
        #[cfg(not(rce_qmm_metal))]
        { None }
    }

    /// `out = w @ a` where `w: [n_rows, n_cols] (GGUF-packed bytes)`,
    /// `a: [n_cols] f32`. Returns the `n_rows`-length output vector.
    /// Falls back to `None` for any unsupported dtype — caller should
    /// downgrade to the CPU path via `rce_op_qmm::mm_1d`.
    pub fn qmm_1d(&self, a: &[f32], w: &QuantWeight) -> Option<Vec<f32>> {
        if a.len() != w.n_cols { return None; }
        match w.dtype {
            GgufDtype::Q8_0 if w.n_cols % 32  == 0 => {
                #[cfg(rce_qmm_metal)]
                { Some(self.inner.gemv(KERNEL_Q8_0, a, w.bytes(), w.n_rows, w.n_cols)) }
                #[cfg(not(rce_qmm_metal))]
                { let _ = (a, w); None }
            }
            GgufDtype::Q3K  if w.n_cols % 256 == 0 => {
                #[cfg(rce_qmm_metal)]
                { Some(self.inner.gemv(KERNEL_Q3_K, a, w.bytes(), w.n_rows, w.n_cols)) }
                #[cfg(not(rce_qmm_metal))]
                { let _ = (a, w); None }
            }
            GgufDtype::Q4K  if w.n_cols % 256 == 0 => {
                #[cfg(rce_qmm_metal)]
                { Some(self.inner.gemv(KERNEL_Q4_K, a, w.bytes(), w.n_rows, w.n_cols)) }
                #[cfg(not(rce_qmm_metal))]
                { let _ = (a, w); None }
            }
            GgufDtype::Q5K  if w.n_cols % 256 == 0 => {
                #[cfg(rce_qmm_metal)]
                { Some(self.inner.gemv(KERNEL_Q5_K, a, w.bytes(), w.n_rows, w.n_cols)) }
                #[cfg(not(rce_qmm_metal))]
                { let _ = (a, w); None }
            }
            GgufDtype::Q6K  if w.n_cols % 256 == 0 => {
                #[cfg(rce_qmm_metal)]
                { Some(self.inner.gemv(KERNEL_Q6_K, a, w.bytes(), w.n_rows, w.n_cols)) }
                #[cfg(not(rce_qmm_metal))]
                { let _ = (a, w); None }
            }
            // Legacy GGUF formats — 32-elem blocks.
            GgufDtype::Q4_0 if w.n_cols % 32 == 0 => {
                #[cfg(rce_qmm_metal)]
                { Some(self.inner.gemv(KERNEL_Q4_0, a, w.bytes(), w.n_rows, w.n_cols)) }
                #[cfg(not(rce_qmm_metal))]
                { let _ = (a, w); None }
            }
            GgufDtype::Q5_0 if w.n_cols % 32 == 0 => {
                #[cfg(rce_qmm_metal)]
                { Some(self.inner.gemv(KERNEL_Q5_0, a, w.bytes(), w.n_rows, w.n_cols)) }
                #[cfg(not(rce_qmm_metal))]
                { let _ = (a, w); None }
            }
            GgufDtype::Q4_1 if w.n_cols % 32 == 0 => {
                #[cfg(rce_qmm_metal)]
                { Some(self.inner.gemv(KERNEL_Q4_1, a, w.bytes(), w.n_rows, w.n_cols)) }
                #[cfg(not(rce_qmm_metal))]
                { let _ = (a, w); None }
            }
            // PrismML formats — 128-elem blocks.
            GgufDtype::Q1_0 if w.n_cols % 128 == 0 => {
                #[cfg(rce_qmm_metal)]
                { Some(self.inner.gemv(KERNEL_Q1_0, a, w.bytes(), w.n_rows, w.n_cols)) }
                #[cfg(not(rce_qmm_metal))]
                { let _ = (a, w); None }
            }
            GgufDtype::Q2_0 if w.n_cols % 128 == 0 => {
                #[cfg(rce_qmm_metal)]
                { Some(self.inner.gemv(KERNEL_Q2_0, a, w.bytes(), w.n_rows, w.n_cols)) }
                #[cfg(not(rce_qmm_metal))]
                { let _ = (a, w); None }
            }
            _ => None,
        }
    }

    /// True iff this context can dispatch `w` natively. Convenience for
    /// `PricedBackend.cost()` — return `f64::INFINITY` for unsupported
    /// dtypes so `pick_priced` selects another backend.
    pub fn supports(&self, w: &QuantWeight) -> bool {
        match w.dtype {
            GgufDtype::Q8_0 | GgufDtype::Q4_0 | GgufDtype::Q5_0 | GgufDtype::Q4_1 => {
                w.n_cols % 32  == 0
            }
            GgufDtype::Q1_0 | GgufDtype::Q2_0 => w.n_cols % 128 == 0,
            GgufDtype::Q3K | GgufDtype::Q4K | GgufDtype::Q5K | GgufDtype::Q6K => w.n_cols % 256 == 0,
            _ => false,
        }
    }
}

// Kernel-name constants exported for the dispatch entry point.
const KERNEL_Q8_0: &str = "q8_0_gemv_f32";
const KERNEL_Q3_K: &str = "q3_k_gemv_f32";
const KERNEL_Q4_K: &str = "q4_k_gemv_f32";
const KERNEL_Q5_K: &str = "q5_k_gemv_f32";
const KERNEL_Q6_K: &str = "q6_k_gemv_f32";
const KERNEL_Q4_0: &str = "q4_0_gemv_f32";
const KERNEL_Q5_0: &str = "q5_0_gemv_f32";
const KERNEL_Q4_1: &str = "q4_1_gemv_f32";
const KERNEL_Q1_0: &str = "q1_0_gemv_f32";
const KERNEL_Q2_0: &str = "q2_0_gemv_f32";

// ── macOS / iOS implementation ──────────────────────────────────────

#[cfg(rce_qmm_metal)]
mod imp {
    use metal::{
        Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
        Library,
    };
    use objc::rc::autoreleasepool;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// All four MSL kernels, registered together. The Q4_K / Q5_K
    /// kernels share `q4_k_get_scale_min` (defined once at the top); the
    /// per-format kernels follow. Each is bit-for-bit faithful to the
    /// corresponding `rce_op_qmm::mm_1d_*` reference up to f32
    /// reduction order.
    const QMM_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ── Q8_0 ────────────────────────────────────────────────────────
kernel void q8_0_gemv_f32(
    device const uchar* w_bytes [[buffer(0)]],
    device const float* a       [[buffer(1)]],
    device       float* out     [[buffer(2)]],
    constant uint& n_cols       [[buffer(3)]],
    constant uint& n_rows       [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= n_rows) return;
    const uint BLOCK = 32u;
    const uint BB    = 34u;
    uint n_blocks    = n_cols / BLOCK;
    uint bytes_per_row = n_blocks * BB;
    uint row_start   = gid * bytes_per_row;
    float sumf = 0.0;
    for (uint ib = 0u; ib < n_blocks; ib++) {
        uint s = row_start + ib * BB;
        ushort scale_bits = ushort(w_bytes[s]) | (ushort(w_bytes[s + 1u]) << 8);
        float  scale      = float(as_type<half>(scale_bits));
        float  sub        = 0.0;
        uint   a_base     = ib * BLOCK;
        for (uint q = 0u; q < BLOCK; q++) {
            char w_i8 = as_type<char>(w_bytes[s + 2u + q]);
            sub += float(w_i8) * a[a_base + q];
        }
        sumf += scale * sub;
    }
    out[gid] = sumf;
}

// Shared helper for Q4_K + Q5_K: unpack a (6-bit scale, 6-bit min) pair
// from the 12-byte packed `scales` area. Identical layout to the CPU
// reference `q4_k_get_scale_min` in rce-op-qmm.
inline void q4_k_get_scale_min(
    device const uchar* scales,
    uint j,
    thread uchar& sc_out,
    thread uchar& m_out
) {
    if (j < 4u) {
        sc_out = scales[j] & 63u;
        m_out  = scales[j + 4u] & 63u;
    } else {
        sc_out = (scales[j + 4u] & 0x0Fu) | (((scales[j - 4u] >> 6) & 0x03u) << 4);
        m_out  = (scales[j + 4u] >> 4)    | (((scales[j] >> 6) & 0x03u) << 4);
    }
}

// ── Q3_K ────────────────────────────────────────────────────────
// Block: 110 B / 256 elems. Layout:
//   [hmask 32 B][qs 64 B][packed 6-bit scales 12 B][d (f16) 2 B]
// Weight reconstruction:
//   q = ((qs[..] >> shift) & 0x03) - (hmask & m ? 0 : 4)        # ∈ {-4..3}
//   out_i = d * (scale[is] - 32) * q
// Scales need a 4-step KMASK1/KMASK2 dance to recover 16 i8 values in [0,63];
// subtract 32 to get the signed 6-bit value. The traversal carries `m` and
// `is` across the two outer iterations: m walks 1,2,4,...,128 over the 8
// j-loops; is walks 0..16 over the 8 sub-blocks. Mirrors the CPU reference
// `mm_1d_q3_k` in rce-op-qmm exactly.
kernel void q3_k_gemv_f32(
    device const uchar* w_bytes [[buffer(0)]],
    device const float* a       [[buffer(1)]],
    device       float* out     [[buffer(2)]],
    constant uint& n_cols       [[buffer(3)]],
    constant uint& n_rows       [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= n_rows) return;
    const uint BLOCK = 256u;
    const uint BB    = 110u;
    const uint KMASK1 = 0x03030303u;
    const uint KMASK2 = 0x0f0f0f0fu;
    uint n_blocks = n_cols / BLOCK;
    uint bytes_per_row = n_blocks * BB;
    uint row_start = gid * bytes_per_row;
    float sumf = 0.0;
    for (uint ib = 0u; ib < n_blocks; ib++) {
        uint s = row_start + ib * BB;
        device const uchar* hmask        = w_bytes + s;
        device const uchar* qs           = w_bytes + s + 32u;
        device const uchar* scales_bytes = w_bytes + s + 96u;
        ushort d_bits = ushort(w_bytes[s + 108u]) | (ushort(w_bytes[s + 109u]) << 8);
        float  d_all  = float(as_type<half>(d_bits));

        // Unpack 12 packed bytes → 16 i8 sub-scales (KMASK1/KMASK2 dance).
        uint aux0 = uint(scales_bytes[0])  | (uint(scales_bytes[1])  << 8)
                  | (uint(scales_bytes[2])  << 16) | (uint(scales_bytes[3])  << 24);
        uint aux1 = uint(scales_bytes[4])  | (uint(scales_bytes[5])  << 8)
                  | (uint(scales_bytes[6])  << 16) | (uint(scales_bytes[7])  << 24);
        uint aux2 = uint(scales_bytes[8])  | (uint(scales_bytes[9])  << 8)
                  | (uint(scales_bytes[10]) << 16) | (uint(scales_bytes[11]) << 24);
        uint tmp = aux2;
        uint na2 = ((aux0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        uint na3 = ((aux1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        uint na0 = (aux0 & KMASK2)        | (((tmp)      & KMASK1) << 4);
        uint na1 = (aux1 & KMASK2)        | (((tmp >> 2) & KMASK1) << 4);
        char scales[16];
        scales[0]  = char(na0 & 0xFFu);
        scales[1]  = char((na0 >> 8)  & 0xFFu);
        scales[2]  = char((na0 >> 16) & 0xFFu);
        scales[3]  = char((na0 >> 24) & 0xFFu);
        scales[4]  = char(na1 & 0xFFu);
        scales[5]  = char((na1 >> 8)  & 0xFFu);
        scales[6]  = char((na1 >> 16) & 0xFFu);
        scales[7]  = char((na1 >> 24) & 0xFFu);
        scales[8]  = char(na2 & 0xFFu);
        scales[9]  = char((na2 >> 8)  & 0xFFu);
        scales[10] = char((na2 >> 16) & 0xFFu);
        scales[11] = char((na2 >> 24) & 0xFFu);
        scales[12] = char(na3 & 0xFFu);
        scales[13] = char((na3 >> 8)  & 0xFFu);
        scales[14] = char((na3 >> 16) & 0xFFu);
        scales[15] = char((na3 >> 24) & 0xFFu);

        uint a_super_base = ib * BLOCK;
        uchar m  = 1u;
        uint  is = 0u;
        uint  q_off = 0u;
        for (uint outer = 0u; outer < 2u; outer++) {
            uint shift = 0u;
            for (uint jj = 0u; jj < 4u; jj++) {
                float dl1 = d_all * float(int(scales[is])      - 32);
                float dl2 = d_all * float(int(scales[is + 1u]) - 32);
                uint a_base = a_super_base + outer * 128u + jj * 32u;
                float sub_dot1 = 0.0;
                float sub_dot2 = 0.0;
                for (uint l = 0u; l < 16u; l++) {
                    int ql1   = int((qs[q_off + l] >> shift) & 0x03u);
                    int high1 = ((hmask[l] & m) != 0u) ? 0 : 4;
                    float q1  = float(ql1 - high1);
                    sub_dot1 += q1 * a[a_base + l];

                    int ql2   = int((qs[q_off + l + 16u] >> shift) & 0x03u);
                    int high2 = ((hmask[l + 16u] & m) != 0u) ? 0 : 4;
                    float q2  = float(ql2 - high2);
                    sub_dot2 += q2 * a[a_base + 16u + l];
                }
                sumf += dl1 * sub_dot1 + dl2 * sub_dot2;
                is    += 2u;
                shift += 2u;
                m = uchar(m << 1);
            }
            q_off += 32u;
        }
    }
    out[gid] = sumf;
}

// ── Q4_K ────────────────────────────────────────────────────────
kernel void q4_k_gemv_f32(
    device const uchar* w_bytes [[buffer(0)]],
    device const float* a       [[buffer(1)]],
    device       float* out     [[buffer(2)]],
    constant uint& n_cols       [[buffer(3)]],
    constant uint& n_rows       [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= n_rows) return;
    const uint BLOCK = 256u;
    const uint BB    = 144u;
    uint n_blocks = n_cols / BLOCK;
    uint bytes_per_row = n_blocks * BB;
    uint row_start = gid * bytes_per_row;
    float sumf = 0.0;
    for (uint ib = 0u; ib < n_blocks; ib++) {
        uint s = row_start + ib * BB;
        ushort d_bits    = ushort(w_bytes[s])      | (ushort(w_bytes[s + 1u]) << 8);
        float  d         = float(as_type<half>(d_bits));
        ushort dmin_bits = ushort(w_bytes[s + 2u]) | (ushort(w_bytes[s + 3u]) << 8);
        float  dmin      = float(as_type<half>(dmin_bits));
        device const uchar* scales = w_bytes + s + 4u;
        device const uchar* qs     = w_bytes + s + 16u;
        uint a_super_base = ib * BLOCK;
        for (uint g = 0u; g < 4u; g++) {
            uint q_base = g * 32u;
            uint a_base = a_super_base + g * 64u;
            uchar sc1, m1, sc2, m2;
            q4_k_get_scale_min(scales, g * 2u,      sc1, m1);
            q4_k_get_scale_min(scales, g * 2u + 1u, sc2, m2);
            float d1 = d * float(sc1);
            float m1f = dmin * float(m1);
            float d2 = d * float(sc2);
            float m2f = dmin * float(m2);
            float va_lo = 0.0, a_lo = 0.0, va_hi = 0.0, a_hi = 0.0;
            for (uint l = 0u; l < 32u; l++) {
                uchar byte_v = qs[q_base + l];
                float v_lo = float(byte_v & 0x0Fu);
                float v_hi = float((byte_v >> 4) & 0x0Fu);
                float al = a[a_base + l];
                float ah = a[a_base + 32u + l];
                va_lo += v_lo * al;  a_lo += al;
                va_hi += v_hi * ah;  a_hi += ah;
            }
            sumf += d1 * va_lo - m1f * a_lo + d2 * va_hi - m2f * a_hi;
        }
    }
    out[gid] = sumf;
}

// ── Q5_K ────────────────────────────────────────────────────────
kernel void q5_k_gemv_f32(
    device const uchar* w_bytes [[buffer(0)]],
    device const float* a       [[buffer(1)]],
    device       float* out     [[buffer(2)]],
    constant uint& n_cols       [[buffer(3)]],
    constant uint& n_rows       [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= n_rows) return;
    const uint BLOCK = 256u;
    const uint BB    = 176u;
    uint n_blocks = n_cols / BLOCK;
    uint bytes_per_row = n_blocks * BB;
    uint row_start = gid * bytes_per_row;
    float sumf = 0.0;
    for (uint ib = 0u; ib < n_blocks; ib++) {
        uint s = row_start + ib * BB;
        ushort d_bits    = ushort(w_bytes[s])      | (ushort(w_bytes[s + 1u]) << 8);
        float  d         = float(as_type<half>(d_bits));
        ushort dmin_bits = ushort(w_bytes[s + 2u]) | (ushort(w_bytes[s + 3u]) << 8);
        float  dmin      = float(as_type<half>(dmin_bits));
        device const uchar* scales = w_bytes + s + 4u;
        device const uchar* qh     = w_bytes + s + 16u;
        device const uchar* qs     = w_bytes + s + 48u;
        uint a_super_base = ib * BLOCK;
        for (uint g = 0u; g < 4u; g++) {
            uint q_base = g * 32u;
            uint a_base = a_super_base + g * 64u;
            uchar sc1, m1, sc2, m2;
            q4_k_get_scale_min(scales, g * 2u,      sc1, m1);
            q4_k_get_scale_min(scales, g * 2u + 1u, sc2, m2);
            float d1 = d * float(sc1);
            float m1f = dmin * float(m1);
            float d2 = d * float(sc2);
            float m2f = dmin * float(m2);
            uint shift_lo = 2u * g;
            uint shift_hi = 2u * g + 1u;
            float va_lo = 0.0, a_lo = 0.0, va_hi = 0.0, a_hi = 0.0;
            for (uint l = 0u; l < 32u; l++) {
                uchar byte_v = qs[q_base + l];
                uchar qh_b   = qh[l];
                uchar h1 = ((qh_b >> shift_lo) & 1u) << 4;
                uchar h2 = ((qh_b >> shift_hi) & 1u) << 4;
                float v_lo = float((byte_v & 0x0Fu) | h1);
                float v_hi = float(((byte_v >> 4) & 0x0Fu) | h2);
                float al = a[a_base + l];
                float ah = a[a_base + 32u + l];
                va_lo += v_lo * al;  a_lo += al;
                va_hi += v_hi * ah;  a_hi += ah;
            }
            sumf += d1 * va_lo - m1f * a_lo + d2 * va_hi - m2f * a_hi;
        }
    }
    out[gid] = sumf;
}

// ── Q6K ─────────────────────────────────────────────────────────
kernel void q6_k_gemv_f32(
    device const uchar* w_bytes [[buffer(0)]],
    device const float* a       [[buffer(1)]],
    device       float* out     [[buffer(2)]],
    constant uint& n_cols       [[buffer(3)]],
    constant uint& n_rows       [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= n_rows) return;
    const uint BLOCK = 256u;
    const uint BB    = 210u;
    uint n_blocks = n_cols / BLOCK;
    uint bytes_per_row = n_blocks * BB;
    uint row_start = gid * bytes_per_row;
    float sumf = 0.0;
    for (uint ib = 0u; ib < n_blocks; ib++) {
        uint s = row_start + ib * BB;
        device const uchar* ql       = w_bytes + s;
        device const uchar* qh       = w_bytes + s + 128u;
        device const uchar* scales_u = w_bytes + s + 192u;
        ushort d_bits = ushort(w_bytes[s + 208u]) | (ushort(w_bytes[s + 209u]) << 8);
        float  d = float(as_type<half>(d_bits));
        uint a_super_base = ib * BLOCK;
        for (uint outer = 0u; outer < 2u; outer++) {
            uint ql_off = outer * 64u;
            uint qh_off = outer * 32u;
            uint sc_off = outer * 8u;
            uint a_off  = a_super_base + outer * 128u;
            for (uint l = 0u; l < 32u; l++) {
                uint is = l / 16u;
                int q1 = int((ql[ql_off + l]       & 0x0Fu)
                          | (((qh[qh_off + l])       & 0x03u) << 4)) - 32;
                int q2 = int((ql[ql_off + l + 32u] & 0x0Fu)
                          | (((qh[qh_off + l] >> 2) & 0x03u) << 4)) - 32;
                int q3 = int((ql[ql_off + l] >> 4)
                          | (((qh[qh_off + l] >> 4) & 0x03u) << 4)) - 32;
                int q4 = int((ql[ql_off + l + 32u] >> 4)
                          | (((qh[qh_off + l] >> 6) & 0x03u) << 4)) - 32;
                float s1 = float(as_type<char>(scales_u[sc_off + is]));
                float s2 = float(as_type<char>(scales_u[sc_off + is + 2u]));
                float s3 = float(as_type<char>(scales_u[sc_off + is + 4u]));
                float s4 = float(as_type<char>(scales_u[sc_off + is + 6u]));
                sumf += d * s1 * float(q1) * a[a_off + l];
                sumf += d * s2 * float(q2) * a[a_off + l + 32u];
                sumf += d * s3 * float(q3) * a[a_off + l + 64u];
                sumf += d * s4 * float(q4) * a[a_off + l + 96u];
            }
        }
    }
    out[gid] = sumf;
}

// ── Q4_0 ────────────────────────────────────────────────────────
// 32-elem blocks, 18 B (f16 d + 16 split-nibble bytes). v = nibble − 8.
kernel void q4_0_gemv_f32(
    device const uchar* w_bytes [[buffer(0)]],
    device const float* a       [[buffer(1)]],
    device       float* out     [[buffer(2)]],
    constant uint& n_cols       [[buffer(3)]],
    constant uint& n_rows       [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= n_rows) return;
    const uint BLOCK = 32u;
    const uint BB    = 18u;
    uint n_blocks = n_cols / BLOCK;
    uint bytes_per_row = n_blocks * BB;
    uint row_start = gid * bytes_per_row;
    float sumf = 0.0;
    for (uint ib = 0u; ib < n_blocks; ib++) {
        uint s = row_start + ib * BB;
        ushort scale_bits = ushort(w_bytes[s]) | (ushort(w_bytes[s + 1u]) << 8);
        float  scale      = float(as_type<half>(scale_bits));
        float  sub        = 0.0;
        uint   a_base     = ib * BLOCK;
        for (uint q = 0u; q < 16u; q++) {
            uchar byte_v = w_bytes[s + 2u + q];
            float v0 = float(int(byte_v & 0x0Fu) - 8);
            float v1 = float(int((byte_v >> 4) & 0x0Fu) - 8);
            sub += v0 * a[a_base + q];
            sub += v1 * a[a_base + q + 16u];
        }
        sumf += scale * sub;
    }
    out[gid] = sumf;
}

// ── Q5_0 ────────────────────────────────────────────────────────
// 32-elem blocks, 22 B (f16 d + u32 qh + 16 split-nibble bytes).
// v = ((nibble | (qh_bit << 4)) − 16).
kernel void q5_0_gemv_f32(
    device const uchar* w_bytes [[buffer(0)]],
    device const float* a       [[buffer(1)]],
    device       float* out     [[buffer(2)]],
    constant uint& n_cols       [[buffer(3)]],
    constant uint& n_rows       [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= n_rows) return;
    const uint BLOCK = 32u;
    const uint BB    = 22u;
    uint n_blocks = n_cols / BLOCK;
    uint bytes_per_row = n_blocks * BB;
    uint row_start = gid * bytes_per_row;
    float sumf = 0.0;
    for (uint ib = 0u; ib < n_blocks; ib++) {
        uint s = row_start + ib * BB;
        ushort scale_bits = ushort(w_bytes[s]) | (ushort(w_bytes[s + 1u]) << 8);
        float  scale      = float(as_type<half>(scale_bits));
        // u32 qh, little-endian.
        uint qh = uint(w_bytes[s + 2u])
                | (uint(w_bytes[s + 3u]) << 8)
                | (uint(w_bytes[s + 4u]) << 16)
                | (uint(w_bytes[s + 5u]) << 24);
        float  sub        = 0.0;
        uint   a_base     = ib * BLOCK;
        for (uint q = 0u; q < 16u; q++) {
            uchar byte_v = w_bytes[s + 6u + q];
            uint h0 = (qh >> q) & 1u;
            uint h1 = (qh >> (q + 16u)) & 1u;
            float v0 = float(int((byte_v & 0x0Fu) | (h0 << 4)) - 16);
            float v1 = float(int(((byte_v >> 4) & 0x0Fu) | (h1 << 4)) - 16);
            sub += v0 * a[a_base + q];
            sub += v1 * a[a_base + q + 16u];
        }
        sumf += scale * sub;
    }
    out[gid] = sumf;
}

// ── Q4_1 ────────────────────────────────────────────────────────
// 32-elem blocks, 20 B (f16 d + f16 min + 16 split-nibble bytes).
// w[i] = d·v + min ⇒ Σ a·w = d·Σ(a·v) + min·Σa.
kernel void q4_1_gemv_f32(
    device const uchar* w_bytes [[buffer(0)]],
    device const float* a       [[buffer(1)]],
    device       float* out     [[buffer(2)]],
    constant uint& n_cols       [[buffer(3)]],
    constant uint& n_rows       [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= n_rows) return;
    const uint BLOCK = 32u;
    const uint BB    = 20u;
    uint n_blocks = n_cols / BLOCK;
    uint bytes_per_row = n_blocks * BB;
    uint row_start = gid * bytes_per_row;
    float sumf = 0.0;
    for (uint ib = 0u; ib < n_blocks; ib++) {
        uint s = row_start + ib * BB;
        ushort d_bits   = ushort(w_bytes[s])      | (ushort(w_bytes[s + 1u]) << 8);
        ushort min_bits = ushort(w_bytes[s + 2u]) | (ushort(w_bytes[s + 3u]) << 8);
        float  d        = float(as_type<half>(d_bits));
        float  m        = float(as_type<half>(min_bits));
        float  va       = 0.0;
        float  asum     = 0.0;
        uint   a_base   = ib * BLOCK;
        for (uint q = 0u; q < 16u; q++) {
            uchar byte_v = w_bytes[s + 4u + q];
            float v0 = float(byte_v & 0x0Fu);
            float v1 = float((byte_v >> 4) & 0x0Fu);
            float a_lo = a[a_base + q];
            float a_hi = a[a_base + q + 16u];
            va   += v0 * a_lo + v1 * a_hi;
            asum += a_lo + a_hi;
        }
        sumf += d * va + m * asum;
    }
    out[gid] = sumf;
}

// ── Q1_0 (PrismML) ──────────────────────────────────────────────
// 128-elem blocks, 18 B (f16 d + 16 bytes, 8 weights/byte LSB-first).
// bit=1 → +a, bit=0 → −a; sumf += d·(Σa[+] − Σa[−]).
kernel void q1_0_gemv_f32(
    device const uchar* w_bytes [[buffer(0)]],
    device const float* a       [[buffer(1)]],
    device       float* out     [[buffer(2)]],
    constant uint& n_cols       [[buffer(3)]],
    constant uint& n_rows       [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= n_rows) return;
    const uint QK = 128u;
    const uint BB = 18u;
    uint n_blocks = n_cols / QK;
    uint bytes_per_row = n_blocks * BB;
    uint row_start = gid * bytes_per_row;
    float sumf = 0.0;
    for (uint ib = 0u; ib < n_blocks; ib++) {
        uint s = row_start + ib * BB;
        ushort d_bits = ushort(w_bytes[s]) | (ushort(w_bytes[s + 1u]) << 8);
        float  d      = float(as_type<half>(d_bits));
        uint   a_base = ib * QK;
        float  pos    = 0.0;
        float  neg    = 0.0;
        for (uint q = 0u; q < QK / 8u; q++) {
            uchar byte_v = w_bytes[s + 2u + q];
            uint  a_q    = a_base + q * 8u;
            for (uint bit = 0u; bit < 8u; bit++) {
                float av = a[a_q + bit];
                if (((byte_v >> bit) & 1u) != 0u) { pos += av; }
                else                              { neg += av; }
            }
        }
        sumf += d * (pos - neg);
    }
    out[gid] = sumf;
}

// ── Q2_0 (PrismML) ──────────────────────────────────────────────
// 128-elem blocks, 34 B (f16 d + 32 bytes, 4 weights/byte 2-bit each).
// codes: 0→−1, 2→+1, 3→+2, 1→0 (skip). sumf += d·(pos − neg + 2·plus2).
kernel void q2_0_gemv_f32(
    device const uchar* w_bytes [[buffer(0)]],
    device const float* a       [[buffer(1)]],
    device       float* out     [[buffer(2)]],
    constant uint& n_cols       [[buffer(3)]],
    constant uint& n_rows       [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= n_rows) return;
    const uint QK = 128u;
    const uint BB = 34u;
    uint n_blocks = n_cols / QK;
    uint bytes_per_row = n_blocks * BB;
    uint row_start = gid * bytes_per_row;
    float sumf = 0.0;
    for (uint ib = 0u; ib < n_blocks; ib++) {
        uint s = row_start + ib * BB;
        ushort d_bits = ushort(w_bytes[s]) | (ushort(w_bytes[s + 1u]) << 8);
        float  d      = float(as_type<half>(d_bits));
        uint   a_base = ib * QK;
        float  pos    = 0.0;
        float  neg    = 0.0;
        float  plus2  = 0.0;
        for (uint q = 0u; q < QK / 4u; q++) {
            uchar byte_v = w_bytes[s + 2u + q];
            uint  a_q    = a_base + q * 4u;
            for (uint lane = 0u; lane < 4u; lane++) {
                uint code = (byte_v >> (lane * 2u)) & 0x3u;
                float av = a[a_q + lane];
                if      (code == 0u) { neg   += av; }
                else if (code == 2u) { pos   += av; }
                else if (code == 3u) { plus2 += av; }
            }
        }
        sumf += d * (pos - neg + 2.0 * plus2);
    }
    out[gid] = sumf;
}
"#;

    pub(super) struct Inner {
        device: Device,
        queue: CommandQueue,
        pipelines: HashMap<&'static str, ComputePipelineState>,
        // Per-(n_rows, n_cols) buffer cache: A and OUT sized to the
        // shape, reused per call of that shape. Weight buffers built
        // per-call (could be cached per-weight-pointer in a follow-on
        // as a graph runtime would, but the per-call alloc is small
        // relative to the dispatch + wait cost).
        shape_cache: Mutex<HashMap<(usize, usize), ShapeBufs>>,
    }

    struct ShapeBufs {
        a_buf: Buffer,
        out_buf: Buffer,
    }

    // SAFETY: Metal Device/Queue/Pipeline are documented thread-safe per
    // Apple's docs as long as command buffers aren't shared mid-encode.
    // Cached buffers are accessed under the shape_cache mutex.
    unsafe impl Send for Inner {}
    unsafe impl Sync for Inner {}
    unsafe impl Send for ShapeBufs {}
    unsafe impl Sync for ShapeBufs {}

    impl Inner {
        pub(super) fn new() -> Option<Self> {
            autoreleasepool(|| {
                let device = Device::system_default()?;
                let queue = device.new_command_queue();
                let lib: Library = device
                    .new_library_with_source(QMM_MSL, &metal::CompileOptions::new())
                    .ok()?;
                let mut pipelines = HashMap::new();
                for kernel in &[
                    super::KERNEL_Q8_0,
                    super::KERNEL_Q3_K,
                    super::KERNEL_Q4_K,
                    super::KERNEL_Q5_K,
                    super::KERNEL_Q6_K,
                    super::KERNEL_Q4_0,
                    super::KERNEL_Q5_0,
                    super::KERNEL_Q4_1,
                    super::KERNEL_Q1_0,
                    super::KERNEL_Q2_0,
                ] {
                    let func = lib.get_function(kernel, None).ok()?;
                    let p = device.new_compute_pipeline_state_with_function(&func).ok()?;
                    pipelines.insert(*kernel, p);
                }
                Some(Self {
                    device,
                    queue,
                    pipelines,
                    shape_cache: Mutex::new(HashMap::new()),
                })
            })
        }

        fn ensure_shape_bufs(&self, n_rows: usize, n_cols: usize) {
            let mut cache = self.shape_cache.lock().expect("qmm-metal cache mutex");
            if cache.contains_key(&(n_rows, n_cols)) {
                return;
            }
            let a_bytes = (n_cols * std::mem::size_of::<f32>()) as u64;
            let out_bytes = (n_rows * std::mem::size_of::<f32>()) as u64;
            let a_buf = self.device.new_buffer(a_bytes, MTLResourceOptions::StorageModeShared);
            let out_buf = self.device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
            cache.insert((n_rows, n_cols), ShapeBufs { a_buf, out_buf });
        }

        /// Dispatch the named GEMV kernel over `(a, w_bytes)`. All four
        /// per-format kernels share the same buffer/encoding/wait
        /// scaffolding; only the pipeline name changes per call.
        pub(super) fn gemv(
            &self,
            kernel: &str,
            a: &[f32],
            w_bytes: &[u8],
            n_rows: usize,
            n_cols: usize,
        ) -> Vec<f32> {
            self.ensure_shape_bufs(n_rows, n_cols);
            let cache = self.shape_cache.lock().expect("qmm-metal cache mutex");
            let bufs = cache.get(&(n_rows, n_cols)).expect("shape just inserted");
            let pipeline = self.pipelines.get(kernel).expect("kernel pipeline registered");

            let w_buf = self.device.new_buffer_with_data(
                w_bytes.as_ptr() as *const _,
                w_bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            unsafe {
                let dst_a = bufs.a_buf.contents() as *mut f32;
                std::ptr::copy_nonoverlapping(a.as_ptr(), dst_a, n_cols);
            }

            autoreleasepool(|| {
                let cmd_buffer = self.queue.new_command_buffer();
                let enc = cmd_buffer.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pipeline);
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&bufs.a_buf), 0);
                enc.set_buffer(2, Some(&bufs.out_buf), 0);

                let n_cols_u32 = n_cols as u32;
                let n_rows_u32 = n_rows as u32;
                enc.set_bytes(3, std::mem::size_of::<u32>() as u64,
                    &n_cols_u32 as *const u32 as *const _);
                enc.set_bytes(4, std::mem::size_of::<u32>() as u64,
                    &n_rows_u32 as *const u32 as *const _);

                let max_tg = pipeline.max_total_threads_per_threadgroup() as usize;
                let tg_w = max_tg.min(n_rows).max(1) as u64;
                let grid = MTLSize::new(n_rows as u64, 1, 1);
                let tg = MTLSize::new(tg_w, 1, 1);
                enc.dispatch_threads(grid, tg);
                enc.end_encoding();
                cmd_buffer.commit();
                cmd_buffer.wait_until_completed();
            });

            let mut out = vec![0.0f32; n_rows];
            unsafe {
                let src = bufs.out_buf.contents() as *const f32;
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n_rows);
            }
            out
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rce_op_qmm::mm_1d;

    /// Synthetic Q8_0 weight on the heap — mirror of the fixtures the
    /// other adapter crates use so cross-path comparison is meaningful.
    fn synthetic_q8(n_rows: usize, n_cols: usize) -> QuantWeight {
        assert_eq!(n_cols % 32, 0, "Q8_0 needs n_cols % 32 == 0");
        let blocks_per_row = n_cols / 32;
        let bytes_per_row = blocks_per_row * 34;
        let mut bytes = vec![0u8; n_rows * bytes_per_row];
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let off = r * bytes_per_row + b * 34;
                bytes[off] = 0x00;
                bytes[off + 1] = 0x3C; // f16 1.0 LE
                for i in 0..32 {
                    let v = ((r + b * 32 + i) as i32 % 7 - 3) as i8;
                    bytes[off + 2 + i] = v as u8;
                }
            }
        }
        QuantWeight::from_owned_bytes(bytes, GgufDtype::Q8_0, n_rows, n_cols)
    }

    /// Q8_0 with a real per-block scale so the scale·sub fold actually
    /// matters (the 1.0-scale fixture above tests only the i8 dot).
    fn synthetic_q8_varied_scales(n_rows: usize, n_cols: usize) -> QuantWeight {
        let blocks_per_row = n_cols / 32;
        let bytes_per_row = blocks_per_row * 34;
        let mut bytes = vec![0u8; n_rows * bytes_per_row];
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let off = r * bytes_per_row + b * 34;
                // f16 scale = 1/(r+b+1) — distinct per (row, block).
                let s_f16 = half::f16::from_f32(1.0 / ((r + b + 1) as f32));
                bytes[off..off + 2].copy_from_slice(&s_f16.to_le_bytes());
                for i in 0..32 {
                    let v = ((r * 13 + b * 7 + i) as i32 % 17 - 8) as i8;
                    bytes[off + 2 + i] = v as u8;
                }
            }
        }
        QuantWeight::from_owned_bytes(bytes, GgufDtype::Q8_0, n_rows, n_cols)
    }

    fn assert_close(label: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{label}: length mismatch");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let abs = (g - w).abs();
            let denom = w.abs().max(1e-6);
            let rel = abs / denom;
            assert!(abs < 1e-4 || rel < 1e-3,
                "{label} out[{i}]: metal={g}, ref={w}, abs={abs:.3e}, rel={rel:.3e}");
        }
    }

    #[test]
    fn metal_q8_0_gemv_matches_cpu_reference_unit_scale() {
        let ctx = match MetalQmmContext::new() {
            Some(c) => c,
            None => { eprintln!("metal not available — skipping"); return; }
        };
        let w = synthetic_q8(16, 32);
        let a: Vec<f32> = (0..32).map(|i| (i as f32 * 0.03).sin()).collect();

        let got = ctx.qmm_1d(&a, &w).expect("Q8_0 supported");
        let mut want = vec![0.0f32; w.n_rows];
        mm_1d(&a, &w, &mut want);
        assert_close("unit-scale", &got, &want);
    }

    #[test]
    fn metal_q8_0_gemv_matches_cpu_reference_varied_scales() {
        let ctx = match MetalQmmContext::new() {
            Some(c) => c,
            None => { eprintln!("metal not available — skipping"); return; }
        };
        let w = synthetic_q8_varied_scales(64, 128);
        let a: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.017).cos()).collect();

        let got = ctx.qmm_1d(&a, &w).expect("Q8_0 supported");
        let mut want = vec![0.0f32; w.n_rows];
        mm_1d(&a, &w, &mut want);
        assert_close("varied-scales", &got, &want);
    }

    #[test]
    fn metal_q8_0_gemv_matches_cpu_at_llm_typical_shape() {
        let ctx = match MetalQmmContext::new() {
            Some(c) => c,
            None => { eprintln!("metal not available — skipping"); return; }
        };
        // 1024 rows × 2048 cols — sits in the range TinyLlama / Qwen3
        // weight matmuls live in. Stresses block walking + scaling.
        let w = synthetic_q8_varied_scales(1024, 2048);
        let a: Vec<f32> = (0..2048).map(|i| ((i as f32) * 0.005).sin()).collect();

        let got = ctx.qmm_1d(&a, &w).expect("Q8_0 supported");
        let mut want = vec![0.0f32; w.n_rows];
        mm_1d(&a, &w, &mut want);
        assert_close("llm-typical", &got, &want);
    }

    #[test]
    fn supports_every_implemented_format_but_not_others() {
        let ctx = match MetalQmmContext::new() {
            Some(c) => c,
            None => return,
        };
        // K-quants.
        assert!(ctx.supports(&synthetic_q8(16, 32)));
        assert!(ctx.supports(&synthetic_q4_k(16, 256)));
        assert!(ctx.supports(&synthetic_q5_k(16, 256)));
        assert!(ctx.supports(&synthetic_q6_k(16, 256)));
        // Legacy GGUF 32-blocks.
        assert!(ctx.supports(&synthetic_q4_0(16, 32)));
        assert!(ctx.supports(&synthetic_q5_0(16, 32)));
        assert!(ctx.supports(&synthetic_q4_1(16, 32)));
        // PrismML 128-blocks.
        assert!(ctx.supports(&synthetic_q1_0(16, 128)));
        assert!(ctx.supports(&synthetic_q2_0(16, 128)));
        // Unsupported format (I2_S — BitNet ternary, kept on CPU for
        // accuracy; act-quant doesn't replicate cleanly via Metal).
        let i2s = QuantWeight::from_owned_bytes(
            vec![0u8; 16 * 32], GgufDtype::I2S, 16, 256);
        assert!(!ctx.supports(&i2s));
        assert!(ctx.qmm_1d(&[0.0f32; 256], &i2s).is_none(),
            "qmm_1d should return None for I2_S (caller falls back)");
    }

    // ── Q4_K / Q5_K / Q6K fixtures ────────────────────────────────────

    fn synthetic_q4_k(n_rows: usize, n_cols: usize) -> QuantWeight {
        assert_eq!(n_cols % 256, 0, "Q4_K block is 256");
        let blocks_per_row = n_cols / 256;
        let bytes_per_row = blocks_per_row * 144;
        let mut bytes = vec![0u8; n_rows * bytes_per_row];
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let off = r * bytes_per_row + b * 144;
                // f16 d = 1/(r+b+1), f16 dmin = -0.5/(r+b+2) — distinct
                // per (row, block) so the (d, dmin) fold actually
                // contributes; pulled into the f32 range Q4_K real
                // weights occupy.
                let d = half::f16::from_f32(1.0 / ((r + b + 1) as f32));
                bytes[off..off + 2].copy_from_slice(&d.to_le_bytes());
                let dmin = half::f16::from_f32(-0.5 / ((r + b + 2) as f32));
                bytes[off + 2..off + 4].copy_from_slice(&dmin.to_le_bytes());
                // 12 bytes of packed 6-bit scales+mins.
                for i in 0..12 {
                    bytes[off + 4 + i] = ((r * 7 + b * 5 + i) as u8) & 0x3F;
                }
                // 128 bytes of 4-bit weights (two nibbles per byte).
                for i in 0..128 {
                    bytes[off + 16 + i] = (r * 13 + b * 11 + i) as u8;
                }
            }
        }
        QuantWeight::from_owned_bytes(bytes, GgufDtype::Q4K, n_rows, n_cols)
    }

    fn synthetic_q3_k(n_rows: usize, n_cols: usize) -> QuantWeight {
        // Q3_K block: 110 B / 256 elems.
        //   [hmask 32 B][qs 64 B][packed 6-bit scales 12 B][d (f16) 2 B]
        assert_eq!(n_cols % 256, 0);
        let blocks_per_row = n_cols / 256;
        let bytes_per_row = blocks_per_row * 110;
        let mut bytes = vec![0u8; n_rows * bytes_per_row];
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let off = r * bytes_per_row + b * 110;
                // 32 bytes of hmask (high bits, addressed by the moving mask).
                for i in 0..32 {
                    bytes[off + i] = (r * 5 + b * 11 + i * 7) as u8;
                }
                // 64 bytes of qs (low 2 bits of each weight, 4 per byte).
                for i in 0..64 {
                    bytes[off + 32 + i] = (r * 13 + b * 11 + i) as u8;
                }
                // 12 bytes packed 6-bit scales.
                for i in 0..12 {
                    bytes[off + 96 + i] = ((r * 7 + b * 5 + i) as u8) & 0x3F;
                }
                // f16 d at offset 108..110.
                let d = half::f16::from_f32(1.0 / ((r + b + 1) as f32));
                bytes[off + 108..off + 110].copy_from_slice(&d.to_le_bytes());
            }
        }
        QuantWeight::from_owned_bytes(bytes, GgufDtype::Q3K, n_rows, n_cols)
    }

    fn synthetic_q5_k(n_rows: usize, n_cols: usize) -> QuantWeight {
        assert_eq!(n_cols % 256, 0);
        let blocks_per_row = n_cols / 256;
        let bytes_per_row = blocks_per_row * 176;
        let mut bytes = vec![0u8; n_rows * bytes_per_row];
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let off = r * bytes_per_row + b * 176;
                let d = half::f16::from_f32(1.0 / ((r + b + 1) as f32));
                bytes[off..off + 2].copy_from_slice(&d.to_le_bytes());
                let dmin = half::f16::from_f32(-0.5 / ((r + b + 2) as f32));
                bytes[off + 2..off + 4].copy_from_slice(&dmin.to_le_bytes());
                for i in 0..12 {
                    bytes[off + 4 + i] = ((r * 7 + b * 5 + i) as u8) & 0x3F;
                }
                // 32 bytes qh — high bits for the 5th-bit lift.
                for i in 0..32 {
                    bytes[off + 16 + i] = (r * 3 + b * 11 + i * 5) as u8;
                }
                // 128 bytes qs.
                for i in 0..128 {
                    bytes[off + 48 + i] = (r * 13 + b * 11 + i) as u8;
                }
            }
        }
        QuantWeight::from_owned_bytes(bytes, GgufDtype::Q5K, n_rows, n_cols)
    }

    fn synthetic_q6_k(n_rows: usize, n_cols: usize) -> QuantWeight {
        assert_eq!(n_cols % 256, 0);
        let blocks_per_row = n_cols / 256;
        let bytes_per_row = blocks_per_row * 210;
        let mut bytes = vec![0u8; n_rows * bytes_per_row];
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let off = r * bytes_per_row + b * 210;
                // 128 bytes ql (low 4 bits per weight).
                for i in 0..128 {
                    bytes[off + i] = (r * 13 + b * 11 + i) as u8;
                }
                // 64 bytes qh (high 2 bits per weight).
                for i in 0..64 {
                    bytes[off + 128 + i] = (r * 5 + b * 3 + i * 7) as u8;
                }
                // 16 i8 sub-scales (in the signed range Q6K uses).
                for i in 0..16 {
                    let v = ((r + b * 3 + i) as i32 % 11 - 5) as i8;
                    bytes[off + 192 + i] = v as u8;
                }
                // f16 d = 1/(r+b+1).
                let d = half::f16::from_f32(1.0 / ((r + b + 1) as f32));
                bytes[off + 208..off + 210].copy_from_slice(&d.to_le_bytes());
            }
        }
        QuantWeight::from_owned_bytes(bytes, GgufDtype::Q6K, n_rows, n_cols)
    }

    #[test]
    fn metal_q3_k_gemv_matches_cpu_reference() {
        let ctx = match MetalQmmContext::new() {
            Some(c) => c,
            None => { eprintln!("metal not available — skipping"); return; }
        };
        let w = synthetic_q3_k(16, 256);
        let a: Vec<f32> = (0..256).map(|i| (i as f32 * 0.017).sin()).collect();
        let got = ctx.qmm_1d(&a, &w).expect("Q3_K supported");
        let mut want = vec![0.0f32; w.n_rows];
        mm_1d(&a, &w, &mut want);
        assert_close("q3_k", &got, &want);
    }

    #[test]
    fn metal_q3_k_gemv_matches_cpu_reference_at_llm_typical_shape() {
        let ctx = match MetalQmmContext::new() {
            Some(c) => c,
            None => return,
        };
        let w = synthetic_q3_k(512, 1024);
        let a: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.009).cos()).collect();
        let got = ctx.qmm_1d(&a, &w).expect("Q3_K supported");
        let mut want = vec![0.0f32; w.n_rows];
        mm_1d(&a, &w, &mut want);
        assert_close("q3_k-llm", &got, &want);
    }

    #[test]
    fn metal_q4_k_gemv_matches_cpu_reference() {
        let ctx = match MetalQmmContext::new() {
            Some(c) => c,
            None => { eprintln!("metal not available — skipping"); return; }
        };
        let w = synthetic_q4_k(16, 256);
        let a: Vec<f32> = (0..256).map(|i| (i as f32 * 0.013).sin()).collect();
        let got = ctx.qmm_1d(&a, &w).expect("Q4_K supported");
        let mut want = vec![0.0f32; w.n_rows];
        mm_1d(&a, &w, &mut want);
        assert_close("q4_k", &got, &want);
    }

    #[test]
    fn metal_q4_k_gemv_matches_cpu_reference_at_llm_typical_shape() {
        let ctx = match MetalQmmContext::new() {
            Some(c) => c,
            None => return,
        };
        let w = synthetic_q4_k(512, 1024);
        let a: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.007).cos()).collect();
        let got = ctx.qmm_1d(&a, &w).expect("Q4_K supported");
        let mut want = vec![0.0f32; w.n_rows];
        mm_1d(&a, &w, &mut want);
        assert_close("q4_k-llm", &got, &want);
    }

    #[test]
    fn metal_q5_k_gemv_matches_cpu_reference() {
        let ctx = match MetalQmmContext::new() {
            Some(c) => c,
            None => return,
        };
        let w = synthetic_q5_k(16, 256);
        let a: Vec<f32> = (0..256).map(|i| (i as f32 * 0.019).sin()).collect();
        let got = ctx.qmm_1d(&a, &w).expect("Q5_K supported");
        let mut want = vec![0.0f32; w.n_rows];
        mm_1d(&a, &w, &mut want);
        assert_close("q5_k", &got, &want);
    }

    #[test]
    fn metal_q5_k_gemv_matches_cpu_reference_at_llm_typical_shape() {
        let ctx = match MetalQmmContext::new() {
            Some(c) => c,
            None => return,
        };
        let w = synthetic_q5_k(512, 1024);
        let a: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.011).cos()).collect();
        let got = ctx.qmm_1d(&a, &w).expect("Q5_K supported");
        let mut want = vec![0.0f32; w.n_rows];
        mm_1d(&a, &w, &mut want);
        assert_close("q5_k-llm", &got, &want);
    }

    #[test]
    fn metal_q6_k_gemv_matches_cpu_reference() {
        let ctx = match MetalQmmContext::new() {
            Some(c) => c,
            None => return,
        };
        let w = synthetic_q6_k(16, 256);
        let a: Vec<f32> = (0..256).map(|i| (i as f32 * 0.021).sin()).collect();
        let got = ctx.qmm_1d(&a, &w).expect("Q6K supported");
        let mut want = vec![0.0f32; w.n_rows];
        mm_1d(&a, &w, &mut want);
        assert_close("q6_k", &got, &want);
    }

    // ── Q4_0 / Q5_0 / Q4_1 / Q1_0 / Q2_0 fixtures + parity ─────────────

    fn synthetic_q4_0(n_rows: usize, n_cols: usize) -> QuantWeight {
        assert_eq!(n_cols % 32, 0, "Q4_0 block is 32");
        let blocks_per_row = n_cols / 32;
        let bytes_per_row = blocks_per_row * 18;
        let mut bytes = vec![0u8; n_rows * bytes_per_row];
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let off = r * bytes_per_row + b * 18;
                let d = half::f16::from_f32(1.0 / ((r + b + 1) as f32));
                bytes[off..off + 2].copy_from_slice(&d.to_le_bytes());
                for i in 0..16 {
                    bytes[off + 2 + i] = (r * 13 + b * 7 + i * 3) as u8;
                }
            }
        }
        QuantWeight::from_owned_bytes(bytes, GgufDtype::Q4_0, n_rows, n_cols)
    }

    fn synthetic_q5_0(n_rows: usize, n_cols: usize) -> QuantWeight {
        assert_eq!(n_cols % 32, 0);
        let blocks_per_row = n_cols / 32;
        let bytes_per_row = blocks_per_row * 22;
        let mut bytes = vec![0u8; n_rows * bytes_per_row];
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let off = r * bytes_per_row + b * 22;
                let d = half::f16::from_f32(0.5 / ((r + b + 1) as f32));
                bytes[off..off + 2].copy_from_slice(&d.to_le_bytes());
                // qh: pack 32 high bits — alternating pattern.
                let qh: u32 = 0xAAAAAAAA ^ ((r as u32).wrapping_mul(0x9E3779B9));
                bytes[off + 2..off + 6].copy_from_slice(&qh.to_le_bytes());
                for i in 0..16 {
                    bytes[off + 6 + i] = (r * 11 + b * 5 + i * 7) as u8;
                }
            }
        }
        QuantWeight::from_owned_bytes(bytes, GgufDtype::Q5_0, n_rows, n_cols)
    }

    fn synthetic_q4_1(n_rows: usize, n_cols: usize) -> QuantWeight {
        assert_eq!(n_cols % 32, 0);
        let blocks_per_row = n_cols / 32;
        let bytes_per_row = blocks_per_row * 20;
        let mut bytes = vec![0u8; n_rows * bytes_per_row];
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let off = r * bytes_per_row + b * 20;
                let d = half::f16::from_f32(1.0 / ((r + b + 1) as f32));
                let m = half::f16::from_f32(-0.5 / ((r + b + 2) as f32));
                bytes[off..off + 2].copy_from_slice(&d.to_le_bytes());
                bytes[off + 2..off + 4].copy_from_slice(&m.to_le_bytes());
                for i in 0..16 {
                    bytes[off + 4 + i] = (r * 13 + b * 7 + i * 3) as u8;
                }
            }
        }
        QuantWeight::from_owned_bytes(bytes, GgufDtype::Q4_1, n_rows, n_cols)
    }

    fn synthetic_q1_0(n_rows: usize, n_cols: usize) -> QuantWeight {
        assert_eq!(n_cols % 128, 0);
        let blocks_per_row = n_cols / 128;
        let bytes_per_row = blocks_per_row * 18;
        let mut bytes = vec![0u8; n_rows * bytes_per_row];
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let off = r * bytes_per_row + b * 18;
                let d = half::f16::from_f32(1.0 / ((r + b + 1) as f32));
                bytes[off..off + 2].copy_from_slice(&d.to_le_bytes());
                for i in 0..16 {
                    bytes[off + 2 + i] = (r * 17 + b * 11 + i * 13) as u8;
                }
            }
        }
        QuantWeight::from_owned_bytes(bytes, GgufDtype::Q1_0, n_rows, n_cols)
    }

    fn synthetic_q2_0(n_rows: usize, n_cols: usize) -> QuantWeight {
        assert_eq!(n_cols % 128, 0);
        let blocks_per_row = n_cols / 128;
        let bytes_per_row = blocks_per_row * 34;
        let mut bytes = vec![0u8; n_rows * bytes_per_row];
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let off = r * bytes_per_row + b * 34;
                let d = half::f16::from_f32(1.0 / ((r + b + 1) as f32));
                bytes[off..off + 2].copy_from_slice(&d.to_le_bytes());
                for i in 0..32 {
                    bytes[off + 2 + i] = (r * 19 + b * 13 + i * 7) as u8;
                }
            }
        }
        QuantWeight::from_owned_bytes(bytes, GgufDtype::Q2_0, n_rows, n_cols)
    }

    #[test]
    fn metal_q4_0_gemv_matches_cpu_reference() {
        let ctx = match MetalQmmContext::new() { Some(c) => c, None => return };
        let w = synthetic_q4_0(32, 128);
        let a: Vec<f32> = (0..128).map(|i| (i as f32 * 0.013).sin()).collect();
        let got = ctx.qmm_1d(&a, &w).expect("Q4_0 supported");
        let mut want = vec![0.0f32; w.n_rows];
        mm_1d(&a, &w, &mut want);
        assert_close("q4_0", &got, &want);
    }

    #[test]
    fn metal_q5_0_gemv_matches_cpu_reference() {
        let ctx = match MetalQmmContext::new() { Some(c) => c, None => return };
        let w = synthetic_q5_0(32, 128);
        let a: Vec<f32> = (0..128).map(|i| (i as f32 * 0.019).sin()).collect();
        let got = ctx.qmm_1d(&a, &w).expect("Q5_0 supported");
        let mut want = vec![0.0f32; w.n_rows];
        mm_1d(&a, &w, &mut want);
        assert_close("q5_0", &got, &want);
    }

    #[test]
    fn metal_q4_1_gemv_matches_cpu_reference() {
        let ctx = match MetalQmmContext::new() { Some(c) => c, None => return };
        let w = synthetic_q4_1(32, 128);
        let a: Vec<f32> = (0..128).map(|i| (i as f32 * 0.017).cos()).collect();
        let got = ctx.qmm_1d(&a, &w).expect("Q4_1 supported");
        let mut want = vec![0.0f32; w.n_rows];
        mm_1d(&a, &w, &mut want);
        assert_close("q4_1", &got, &want);
    }

    #[test]
    fn metal_q1_0_gemv_matches_cpu_reference() {
        let ctx = match MetalQmmContext::new() { Some(c) => c, None => return };
        // Q1_0 blocks are 128 elements.
        let w = synthetic_q1_0(32, 256);
        let a: Vec<f32> = (0..256).map(|i| (i as f32 * 0.011).sin()).collect();
        let got = ctx.qmm_1d(&a, &w).expect("Q1_0 supported");
        let mut want = vec![0.0f32; w.n_rows];
        mm_1d(&a, &w, &mut want);
        assert_close("q1_0", &got, &want);
    }

    #[test]
    fn metal_q2_0_gemv_matches_cpu_reference() {
        let ctx = match MetalQmmContext::new() { Some(c) => c, None => return };
        let w = synthetic_q2_0(32, 256);
        let a: Vec<f32> = (0..256).map(|i| (i as f32 * 0.023).cos()).collect();
        let got = ctx.qmm_1d(&a, &w).expect("Q2_0 supported");
        let mut want = vec![0.0f32; w.n_rows];
        mm_1d(&a, &w, &mut want);
        assert_close("q2_0", &got, &want);
    }

    // ── LLM-typical shapes (broader stress) ─────────────────────────

    #[test]
    fn metal_q4_0_gemv_matches_cpu_reference_at_llm_typical_shape() {
        let ctx = match MetalQmmContext::new() { Some(c) => c, None => return };
        let w = synthetic_q4_0(512, 1024);
        let a: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.007).sin()).collect();
        let got = ctx.qmm_1d(&a, &w).expect("Q4_0 supported");
        let mut want = vec![0.0f32; w.n_rows];
        mm_1d(&a, &w, &mut want);
        assert_close("q4_0-llm", &got, &want);
    }

    #[test]
    fn metal_q6_k_gemv_matches_cpu_reference_at_llm_typical_shape() {
        let ctx = match MetalQmmContext::new() {
            Some(c) => c,
            None => return,
        };
        let w = synthetic_q6_k(512, 1024);
        let a: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.009).cos()).collect();
        let got = ctx.qmm_1d(&a, &w).expect("Q6K supported");
        let mut want = vec![0.0f32; w.n_rows];
        mm_1d(&a, &w, &mut want);
        assert_close("q6_k-llm", &got, &want);
    }
}
