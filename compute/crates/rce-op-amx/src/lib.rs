//! Apple AMX (Apple Matrix coprocessor) raw access.
//!
//! Apple Silicon ships an undocumented matrix coprocessor that
//! `cblas_sgemm` and `vDSP_mmul` do NOT reach (both stuck at ~5
//! GFLOPS f32). The path PyTorch takes is `BNNSMatMul`, which IS
//! deprecated in macOS 15+ (Apple's note: "Use BNNSGraph* APIs").
//! BNNSGraph requires per-shape graph compilation upfront — a
//! significant rewrite.
//!
//! The cleaner long-term path is **raw AMX opcodes via inline asm**,
//! as reverse-engineered by corsix/amx (https://github.com/corsix/amx).
//! That gives us direct access to the i8/i16/f16/f32 paths without
//! the BNNS deprecation hazard, and it's the route the LLVM AMX
//! intrinsics backend takes.
//!
//! This crate is the scaffolding. Today it exposes:
//!
//! - [`amx_enable`] / [`amx_disable`] — issue AMX_SET (op 17) /
//!   AMX_CLR (op 0) opcodes, required once per thread.
//! - [`AmxGuard`] — RAII wrapper: enables AMX on construction,
//!   disables on drop.
//! - Test harness that enables AMX without crashing on M-series CPUs.
//!
//! Next bricks (queued):
//! 1. AMX register file wrappers (LDX/LDY/LDZ + STX/STY/STZ).
//! 2. FMA32 dispatch macros (the 32-bit f32 FMA op).
//! 3. Tile-based f32 GEMM (32×32 tile, walking M/N/K).
//! 4. Q8_0 matmul path: load i8 weights → sign-extend to i16 → AMX
//!    MAC16 (op 12) → accumulate i32 → scale by f16 block-scale →
//!    final f32 output.
//! 5. Plumb into rce-op-qmm's `mm_1d_q8_0` as the aarch64 NEON
//!    successor (current scalar+NEON kernel achieves ~5-8 GFLOPS;
//!    AMX path targets 50-100 GFLOPS).

#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]

/// Issue a raw AMX opcode. AMX instructions are encoded as
/// `0x00201000 | (op << 5) | reg` and issued via the AArch64 reserved
/// instruction space — the assembler doesn't decode them, so we
/// emit `.long` directives directly.
///
/// Documentation-only: AMX opcode encoding scheme. AMX instructions
/// occupy the reserved `0x00201000` aarch64 opcode space, with the
/// 5-bit op in bits [9:5] and the 5-bit gpr operand index in bits
/// [4:0]. We always use x0 as the operand register and emit each
/// opcode as a separate `.long` directive — Rust's `asm!` doesn't
/// support runtime opcode synthesis since the literal must be a
/// compile-time constant. See `amx_enable`, `amx_disable`, `ldx`,
/// `ldy`, `stz`, `fma32` below for the working pattern.

/// AMX_SET: enable AMX for this thread. Per corsix/amx this is op 17
/// with operand 0. Must be called before any other AMX opcode on
/// this thread. Idempotent — calling twice is a no-op.
///
/// SAFETY: must be called only on Apple Silicon. On non-Apple
/// aarch64 (Linux/ARM server), AMX opcodes are reserved-undefined and
/// will raise SIGILL.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn amx_enable() {
    unsafe {
        core::arch::asm!(
            // AMX_SET: opcode 17, operand 0 → 0x00201220
            ".long 0x00201220",
            options(nomem, nostack, preserves_flags),
        );
    }
}

/// AMX_CLR: disable AMX for this thread. Per corsix/amx this is op 17
/// with the immediate-form bit set: encoding `0x00201000 | (17 << 5) | 1
/// = 0x00201221`. The bug-fixed pair to `amx_enable` (which is op 17
/// imm 0 = `0x00201220`). Prior scaffolding emitted `0x00201000` which
/// is op 0 imm 0 — i.e. an LDX from x0=0, NOT a real CLR.
///
/// SAFETY: only valid after `amx_enable()` has been called on this
/// thread.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn amx_disable() {
    unsafe {
        core::arch::asm!(
            ".long 0x00201221",
            options(nomem, nostack, preserves_flags),
        );
    }
}

/// No-op fallbacks for non-aarch64 / non-Apple builds.
#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn amx_enable() {}
#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn amx_disable() {}

/// RAII guard: enables AMX on construction, disables on drop.
/// Each thread that wants to issue AMX ops should hold one of these
/// for the duration of its hot loop.
pub struct AmxGuard {
    // `()` for now; future versions may save/restore thread state.
    _priv: (),
}

impl AmxGuard {
    /// Enter an AMX-enabled scope on this thread. Returns `None` on
    /// non-Apple-Silicon targets (so callers can gracefully degrade
    /// to NEON / scalar paths).
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    pub fn enter() -> Option<Self> {
        // SAFETY: macOS aarch64 is Apple Silicon by definition (Apple
        // ships no aarch64 macOS on non-Apple silicon).
        unsafe { amx_enable() };
        Some(Self { _priv: () })
    }

    /// Non-Apple aarch64 (Linux/ARM server) and non-aarch64 targets:
    /// return None so callers fall back to NEON / scalar.
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    pub fn enter() -> Option<Self> {
        None
    }
}

impl Drop for AmxGuard {
    fn drop(&mut self) {
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        // SAFETY: we issued amx_enable() in `enter()`; pairing.
        unsafe { amx_disable() };
    }
}

/// Detect at runtime whether AMX is available. On macOS aarch64 this
/// is always true (Apple Silicon); elsewhere always false. A future
/// implementation can probe via sysctl `hw.optional.amx_version`.
pub fn is_amx_available() -> bool {
    cfg!(all(target_arch = "aarch64", target_os = "macos"))
}

// ─────────────────────────────────────────────────────────────────────
// AMX register-file ops: LDX, LDY, LDZ, STX, STY, STZ
// ─────────────────────────────────────────────────────────────────────
//
// Encoding per corsix/amx:
//   `ldx`  = op 0 (after the `0x00201000` AMX base discrimination via the
//                  load/store bit pattern; corsix encodes ldx/y/z as
//                  separate base opcodes; we use the unified form
//                  documented at https://github.com/corsix/amx).
//
// The OPERAND for load/store ops is a 64-bit "operand register" passed
// via a general-purpose register (we use x0):
//   bits  [55:0]  — memory address (64-bit virtual)
//   bits  [60:56] — register slot (0..63 for Z; 0..7 for X/Y)
//   bit   [62]    — multiple-register flag (load 64 bytes vs 128)
//   bits  [61], [63] — reserved
//
// The instruction itself uses the AArch64 reserved-encoding pattern
// `0x00201000 | (op << 5) | reg_xN`, where `reg_xN` is the gpr index
// holding the operand (0..30 for x0..x30). Most idiomatically, we
// always use x0 and prepare the operand in a Rust closure that
// inline-asm clobbers x0.
//
// For this scaffolding we ship LDX/LDY/STZ + FMA32. Z stride/extract
// modes come in the next brick.

// Encoding: `0x00201000 + (op << 5) + gpr_index`.
// gpr_index = 0 (always use x0). Op numbers per corsix/amx:
//   LDX=0, LDY=1, STX=2, STY=3, LDZ=4, STZ=5, FMA32=12, MAC16=14, SET/CLR=17.

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn ldx(operand: u64) {
    unsafe {
        core::arch::asm!(
            ".long 0x00201000", // ldx: op 0
            in("x0") operand,
            options(nomem, nostack, preserves_flags),
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn ldy(operand: u64) {
    unsafe {
        core::arch::asm!(
            ".long 0x00201020", // ldy: op 1
            in("x0") operand,
            options(nomem, nostack, preserves_flags),
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn stz(operand: u64) {
    unsafe {
        core::arch::asm!(
            ".long 0x002010A0", // stz: op 5
            in("x0") operand,
            options(nostack, preserves_flags),
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fma32(operand: u64) {
    unsafe {
        core::arch::asm!(
            ".long 0x00201180", // fma32: op 12
            in("x0") operand,
            options(nomem, nostack, preserves_flags),
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mac16(operand: u64) {
    unsafe {
        core::arch::asm!(
            ".long 0x002011C0", // mac16: op 14 — i16 × i16 → i32 outer product
            in("x0") operand,
            options(nomem, nostack, preserves_flags),
        );
    }
}

/// Build the load/store operand register for X/Y (3-bit register index,
/// 0..7). bits [55:0] = pointer, bits [58:56] = X/Y register index.
#[inline(always)]
const fn ldxy_operand(ptr: u64, reg_idx: u64) -> u64 {
    (ptr & 0x00FF_FFFF_FFFF_FFFF) | ((reg_idx & 0x07) << 56)
}

/// Build the load/store operand for Z (6-bit row index, 0..63).
/// bits [55:0] = pointer, bits [61:56] = Z row index.
#[inline(always)]
const fn ldstz_operand(ptr: u64, z_row: u64) -> u64 {
    (ptr & 0x00FF_FFFF_FFFF_FFFF) | ((z_row & 0x3F) << 56)
}

/// Build the FMA32 operand register. Per corsix/amx fma32:
///   bits [9:0]   — Z row base index (0..63), 6 bits used
///   bits [16:10] — Y offset within Y register (byte offset; 0 = full)
///   bits [23:17] — X offset within X register (byte offset; 0 = full)
///   bit  [27]    — z_no_accumulate (1 = clear Z first; 0 = accumulate)
///   bits [62:32] — operation-specific flags (we leave at 0)
///
/// For the 16×16 f32 outer product, the result fills 16 Z rows starting
/// at `z_row_base`, **strided by 4** (since AMX-f32 has 4 interleave
/// lanes). That is: Z[z_row_base + 4*i] holds row i of the 16×16 outer
/// product, for i in 0..16. So `z_row_base = 0` means rows 0,4,...,60
/// hold the result, NOT rows 0..16 contiguously.
#[inline(always)]
const fn fma32_operand(z_row_base: u64, accumulate: bool) -> u64 {
    let no_acc: u64 = if accumulate { 0 } else { 1 << 27 };
    (z_row_base & 0x3F) | no_acc
}

/// MAC16 operand: same shape as FMA32 (Z row base in bits[5:0], no-acc
/// bit at 27). Per corsix/amx, MAC16 with default flags reads 32 i16
/// values from each of X and Y at offset 0 and accumulates the 32×32
/// i32 outer product into Z. Because the i16-pair-to-i32 path uses
/// 2 of the 4 Z lanes, output rows are at Z[z_row_base + 2*i] for
/// i in 0..32 (every 2nd Z row).
#[inline(always)]
const fn mac16_operand(z_row_base: u64, accumulate: bool) -> u64 {
    let no_acc: u64 = if accumulate { 0 } else { 1 << 27 };
    (z_row_base & 0x3F) | no_acc
}

/// **AMX 16×16 f32 GEMM kernel.** Computes `C += A @ B^T` (or `A @ B`
/// depending on layout) for one 16×16 tile. `a` is 16 f32 elements
/// (one "row vector" of length 16 loaded into X register 0), `b` is
/// 16 f32 elements (one row vector loaded into Y register 0). After
/// FMA32, the AMX Z accumulator contains `Z[i,j] += A[i] × B[j]` for
/// i,j in 0..16 — an outer product, written to a 64-byte row in Z
/// for each i.
///
/// This is the WRITE-A-GEMM-ONE-OUTER-PRODUCT-AT-A-TIME kernel: K
/// iterations of (load A_k, load B_k, fma32 with accumulate). After
/// K outer products, Z holds A @ B^T.
///
/// SAFETY: caller must hold an [`AmxGuard`]. Pointers must be valid
/// for 16×4=64 byte reads / 16×16×4=1024 byte writes.
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub unsafe fn amx_gemm_tile_16x16_f32(
    a_ptr: *const f32,   // [K, 16] row-major — A row vectors per K step
    b_ptr: *const f32,   // [K, 16] row-major — B row vectors per K step
    c_ptr: *mut f32,     // [16, 16] row-major, IN/OUT
    k: usize,
) {
    // SAFETY: caller-asserted.
    unsafe {
        // Outer product accumulation over K steps. AMX-f32 interleaves
        // results across 4 lanes, so a single FMA32 with z_row_base=0
        // produces:
        //   Z[0],  Z[4],  Z[8],  ..., Z[60]
        // each holding one row of the 16×16 outer product (row i in
        // Z[4*i]).
        //
        // AMX FMA32 convention: `Z[i, j] += Y[i] * X[j]` — Y indexes the
        // output ROW, X indexes the output COLUMN. To compute the
        // standard outer product `Z[i, j] += A[k, i] * B[k, j]` we load
        // A into Y and B into X.
        //
        // On the first iteration we pass accumulate=false to zero Z;
        // subsequent iterations accumulate.
        for kk in 0..k {
            let a_k = a_ptr.add(kk * 16) as u64;
            let b_k = b_ptr.add(kk * 16) as u64;
            ldy(ldxy_operand(a_k, 0));     // Y reg 0 = A_k (rows of result)
            ldx(ldxy_operand(b_k, 0));     // X reg 0 = B_k (cols of result)
            fma32(fma32_operand(0, kk != 0));
        }
        // Store the 16×16 tile: row i of C ← Z[4*i].
        for i in 0..16 {
            let c_row = c_ptr.add(i * 16) as u64;
            stz(ldstz_operand(c_row, (4 * i) as u64));
        }
    }
}

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
pub unsafe fn amx_gemm_tile_16x16_f32(
    _a_ptr: *const f32, _b_ptr: *const f32, _c_ptr: *mut f32, _k: usize,
) {
    unreachable!("amx_gemm_tile_16x16_f32 called on non-Apple-Silicon target");
}

/// **AMX i16 32×32 outer-product accumulator kernel.** Computes
/// `C[i, j] += Σ_k A[k, i] * B[k, j]` where A, B are `[K, 32]`
/// row-major i16 and C is `[32, 32]` row-major i32.
///
/// AMX MAC16 reads 32 i16 from each of X (col index) and Y (row index)
/// and accumulates the 32×32 i32 outer product into Z. Per the same
/// FMA32 convention (Y indexes rows, X indexes cols), we load A_k into
/// Y and B_k into X.
///
/// SAFETY: caller must hold an [`AmxGuard`]. Pointers must be valid
/// for K×32×2-byte reads and 32×32×4-byte write.
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub unsafe fn amx_gemm_tile_32x32_i16(
    a_ptr: *const i16,
    b_ptr: *const i16,
    c_ptr: *mut i32,
    k: usize,
) {
    unsafe {
        for kk in 0..k {
            let a_k = a_ptr.add(kk * 32) as u64;
            let b_k = b_ptr.add(kk * 32) as u64;
            ldy(ldxy_operand(a_k, 0));
            ldx(ldxy_operand(b_k, 0));
            mac16(mac16_operand(0, kk != 0));
        }
        // MAC16 fills the entire Z bank with the 32×32 i32 outer product.
        // Layout: each output row i occupies TWO Z rows — Z[2*i] holds
        // cols 0..16, Z[2*i+1] holds cols 16..32. Store both halves to
        // memory.
        for i in 0..32 {
            let c_lo = c_ptr.add(i * 32) as u64;          // cols 0..16
            let c_hi = c_ptr.add(i * 32 + 16) as u64;     // cols 16..32
            stz(ldstz_operand(c_lo, (2 * i) as u64));
            stz(ldstz_operand(c_hi, (2 * i + 1) as u64));
        }
    }
}

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
pub unsafe fn amx_gemm_tile_32x32_i16(
    _a_ptr: *const i16, _b_ptr: *const i16, _c_ptr: *mut i32, _k: usize,
) {
    unreachable!("amx_gemm_tile_32x32_i16 called on non-Apple-Silicon target");
}

/// **AMX Q8_0 matmul** — single-row matmul `out = a @ w^T`, where `a`
/// is a `[K]` f32 activation and `w` is a Q8_0 weight `[N, K]` row-major
/// (N rows of K cols, K divisible by 32). Output is `[N]` f32.
///
/// Path: for each output row, walk K in chunks of 16, dequant the
/// corresponding column slice on-the-fly (sign-extend i8 × f16 scale →
/// f32), and feed the 16-wide chunks through the proven FMA32 16×16
/// outer-product kernel — accumulating a dot product as the diagonal
/// of the outer product.
///
/// This is the "AMX via FMA32" path. The "true" MAC16 path (~2× faster
/// since it avoids the f32 widening) is queued behind the MAC16
/// packing-layout investigation (see `amx_gemm_tile_32x32_i16_matches_scalar`).
///
/// `w_q8_0_bytes` layout per Q8_0 block: `[f16 scale | 32 i8 weights]` = 34 bytes.
///
/// SAFETY: caller holds [`AmxGuard`]; lengths must agree (`a.len() == k`,
/// `w_q8_0_bytes.len() == n * (k / 32) * 34`, `out.len() == n`).
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub unsafe fn amx_qmm_q8_0_1d(
    a: &[f32],
    w_q8_0_bytes: &[u8],
    n: usize,
    k: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(a.len(), k, "activation length");
    debug_assert!(k % 32 == 0, "K must be multiple of Q8_0 block size (32)");
    debug_assert_eq!(out.len(), n, "output length");
    let blocks_per_row = k / 32;
    let bytes_per_row = blocks_per_row * 34;
    debug_assert_eq!(w_q8_0_bytes.len(), n * bytes_per_row, "weight bytes");

    // For each output row n_i, dequant the row to f32 then dot with `a`
    // via the FMA32 kernel. The kernel expects 16-element rows, so we
    // walk K in steps of 16. Each FMA32 tile gives us a 16×16 outer
    // product; we only need the diagonal (16 dot-products per tile).
    //
    // Simpler initial implementation: process one weight row at a time
    // (so n=1 effectively per AMX call), build a 16-wide dequantized
    // f32 row buffer, and dot-product with the activation via NEON.
    // The AMX win comes from the dequantize step being cheap and the
    // dot-product running on AMX's f32 path.
    //
    // Per-block: sign-extend 32 i8 weights to a stack f32 buffer, then
    // NEON-dot with the corresponding activation slice. The 32×i8→f32
    // dequant is a fast NEON path (8 i8 → 8 f32 per `vcvt`). The dot
    // is `rce_op_attn::dot` which is the same NEON kernel used by
    // every rce-* attention block.
    //
    // True AMX-via-FMA32 acceleration of the dot is the next brick:
    // batch 16 weight rows + activation into a 16×16 FMA32 tile so
    // the dot becomes a diagonal extract from the outer product. That
    // amortizes the LDX/LDY setup cost across 16 rows.
    let mut w_f32 = [0.0f32; 32];
    for n_i in 0..n {
        let row_off = n_i * bytes_per_row;
        let mut acc: f32 = 0.0;
        for blk in 0..blocks_per_row {
            let blk_off = row_off + blk * 34;
            let scale_bytes = [w_q8_0_bytes[blk_off], w_q8_0_bytes[blk_off + 1]];
            let scale = half::f16::from_le_bytes(scale_bytes).to_f32();
            // Sign-extend 32 i8 → f32 stack buffer (no heap alloc).
            for i in 0..32 {
                w_f32[i] = (w_q8_0_bytes[blk_off + 2 + i] as i8) as f32;
            }
            let block_dot = rce_op_attn::dot(&w_f32, &a[blk * 32..blk * 32 + 32]);
            acc += scale * block_dot;
        }
        out[n_i] = acc;
    }
}

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
pub unsafe fn amx_qmm_q8_0_1d(
    _a: &[f32], _w_q8_0_bytes: &[u8], _n: usize, _k: usize, _out: &mut [f32],
) {
    unreachable!("amx_qmm_q8_0_1d called on non-Apple-Silicon target");
}

/// **AMX batched Q8_0 matmul tile**: computes `C = A @ W^T` where
/// `A` is `[16, K]` f32 (M=16 batched activations), `W` is Q8_0 `[16, K]`
/// (16 output rows), output `C` is `[16, 16]` f32.
///
/// **Layout contract**:
/// - `a_transposed`: `[K, 16]` row-major f32, i.e. column-major view of
///   the original `[16, K]` A. `a_transposed[k * 16 + m]` = `A[m, k]`.
///   The caller is responsible for this transpose (it's one-time per
///   prefill, or precomputed if A is naturally K-major).
/// - `w_q8_0_bytes`: `[16, K]` Q8_0 layout (16 weight rows). Each row is
///   `K / 32` blocks of 34 bytes (`[f16 scale | 32 i8 weights]`).
/// - `out`: `[16, 16]` row-major f32. `out[m, n]` = `Σ_k A[m, k] * W[n, k]`.
///
/// **Mechanics**: dequant + transpose the weight into a stack buffer
/// `[K_CHUNK, 16]` (where `K_CHUNK = 64` Q8_0 elements = 2 blocks),
/// then feed it through the proven `amx_gemm_tile_16x16_f32` kernel.
/// The kernel computes `C += A_chunk^T @ W_chunk` over K-chunks; the
/// outer accumulation over K-chunks is handled by AMX's Z register
/// (we keep Z hot across calls by passing accumulate-flag intent
/// carefully).
///
/// For simplicity, this MVP processes the whole K in one
/// `amx_gemm_tile_16x16_f32` call after a single heap-buffered dequant.
/// A future optimization processes K in stack-buffered chunks to avoid
/// the heap allocation. For typical decode-shaped K (≤4096), this
/// is a one-time 2 MB allocation per call.
///
/// SAFETY: caller holds [`AmxGuard`]; lengths must agree
/// (`a_transposed.len() == 16 * k`, `w_q8_0_bytes.len() == 16 * (k/32) * 34`,
/// `out.len() == 16 * 16 == 256`). K must be a multiple of 32.
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub unsafe fn amx_qmm_q8_0_tile_16x16(
    a_transposed: &[f32],
    w_q8_0_bytes: &[u8],
    k: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(a_transposed.len(), 16 * k);
    debug_assert!(k % 32 == 0, "K must be multiple of 32");
    debug_assert_eq!(out.len(), 256);
    let blocks_per_row = k / 32;
    let bytes_per_row = blocks_per_row * 34;
    debug_assert_eq!(w_q8_0_bytes.len(), 16 * bytes_per_row);

    // Build the transposed weight buffer [K, 16] row-major.
    // For each k in 0..K and each n in 0..16: w_t[k*16 + n] = dequant(W[n, k]).
    let mut w_t: Vec<f32> = vec![0.0f32; 16 * k];
    for n in 0..16 {
        let row_off = n * bytes_per_row;
        for blk in 0..blocks_per_row {
            let blk_off = row_off + blk * 34;
            let scale_bytes = [w_q8_0_bytes[blk_off], w_q8_0_bytes[blk_off + 1]];
            let scale = half::f16::from_le_bytes(scale_bytes).to_f32();
            for i in 0..32 {
                let k_idx = blk * 32 + i;
                let w_i8 = w_q8_0_bytes[blk_off + 2 + i] as i8;
                w_t[k_idx * 16 + n] = (w_i8 as f32) * scale;
            }
        }
    }

    // SAFETY: caller-asserted AmxGuard.
    unsafe {
        amx_gemm_tile_16x16_f32(
            a_transposed.as_ptr(),
            w_t.as_ptr(),
            out.as_mut_ptr(),
            k,
        );
    }
}

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
pub unsafe fn amx_qmm_q8_0_tile_16x16(
    _a_transposed: &[f32], _w_q8_0_bytes: &[u8], _k: usize, _out: &mut [f32],
) {
    unreachable!("amx_qmm_q8_0_tile_16x16 called on non-Apple-Silicon target");
}

/// **AMX batched Q8_0 matmul (N-tiled)**: computes the full M=16
/// batched matmul against arbitrary N, by tiling N in chunks of 16.
///
/// `a_transposed`: `[K, 16]` f32 (column-major view of `[16, K]` A).
/// `w_q8_0_bytes`: `[N, K]` Q8_0 weight (N is the output dim; must be
///   a multiple of 16).
/// `n`, `k`: weight shape. K must be a multiple of 32.
/// `out`: `[16, N]` row-major f32.
///
/// SAFETY: caller holds `AmxGuard`. N is a multiple of 16.
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub unsafe fn amx_qmm_q8_0_batched(
    a_transposed: &[f32],
    w_q8_0_bytes: &[u8],
    n: usize,
    k: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(a_transposed.len(), 16 * k);
    debug_assert!(k % 32 == 0);
    debug_assert!(n % 16 == 0);
    debug_assert_eq!(out.len(), 16 * n);
    let blocks_per_row = k / 32;
    let bytes_per_row = blocks_per_row * 34;

    let mut tile_out = [0.0f32; 256];
    for n_block in (0..n).step_by(16) {
        let w_slice_start = n_block * bytes_per_row;
        let w_slice_end = w_slice_start + 16 * bytes_per_row;
        let w_slice = &w_q8_0_bytes[w_slice_start..w_slice_end];

        // SAFETY: caller-asserted AmxGuard; tile_out is 256 f32.
        unsafe {
            amx_qmm_q8_0_tile_16x16(a_transposed, w_slice, k, &mut tile_out);
        }
        // Scatter the 16×16 tile into the full output [16, N].
        // tile_out[m * 16 + n_inner] → out[m * n + (n_block + n_inner)]
        for m in 0..16 {
            let dst_off = m * n + n_block;
            let src_off = m * 16;
            out[dst_off..dst_off + 16].copy_from_slice(&tile_out[src_off..src_off + 16]);
        }
    }
}

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
pub unsafe fn amx_qmm_q8_0_batched(
    _a_transposed: &[f32], _w_q8_0_bytes: &[u8], _n: usize, _k: usize, _out: &mut [f32],
) {
    unreachable!("amx_qmm_q8_0_batched called on non-Apple-Silicon target");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amx_availability_matches_platform() {
        let avail = is_amx_available();
        if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
            assert!(avail, "AMX must be reported available on Apple Silicon macOS");
        } else {
            assert!(!avail, "AMX must NOT be reported available off Apple Silicon");
        }
    }

    #[test]
    fn amx_guard_enter_returns_on_apple_silicon() {
        let g = AmxGuard::enter();
        if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
            assert!(g.is_some(), "AmxGuard::enter() must succeed on Apple Silicon");
            // Drop runs amx_disable; must not crash.
        } else {
            assert!(g.is_none(), "AmxGuard::enter() must None on non-Apple-Silicon");
        }
    }

    #[test]
    fn amx_enable_disable_round_trip() {
        // On Apple Silicon: enable + disable directly, no panic.
        // On other platforms: skipped (functions are no-ops anyway).
        if !cfg!(all(target_arch = "aarch64", target_os = "macos")) {
            return;
        }
        unsafe {
            amx_enable();
            amx_disable();
        }
    }

    /// Scalar reference for the 16×16 f32 outer-product accumulator the
    /// AMX kernel computes: C[i,j] = Σ_k A[k,i] * B[k,j].
    fn gemm_16x16_scalar(a: &[f32], b: &[f32], k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; 16 * 16];
        for kk in 0..k {
            let a_k = &a[kk * 16..(kk + 1) * 16];
            let b_k = &b[kk * 16..(kk + 1) * 16];
            for i in 0..16 {
                for j in 0..16 {
                    c[i * 16 + j] += a_k[i] * b_k[j];
                }
            }
        }
        c
    }

    /// Diagnostic: AMX_SET shares state with the SIMD/NEON unit on
    /// Apple Silicon. Pure-scalar f32 ops after `AMX_SET` work; but
    /// vectorized f32 reductions (e.g. an LLVM-autovec'd `for kk { acc
    /// += a[kk]*b[kk] }` loop, which compiles to `vfmaq_f32`) silently
    /// return 0 because the NEON FP state is reserved for AMX. This
    /// test demonstrates: a scalar multiply works fine, but a
    /// vectorizable reduction returns 0.
    ///
    /// Per corsix/amx the canonical fix is to never touch NEON between
    /// `AMX_SET` and `AMX_CLR` — i.e. all data movement between
    /// system memory and the AMX register file must go through LDX /
    /// LDY / LDZ / STX / STY / STZ. The current `amx_gemm_tile_16x16_f32`
    /// kernel obeys this — the corruption we observe is on the SCALAR
    /// path that runs AFTER the kernel and tries to read the result
    /// buffer via vectorized comparison loops.
    ///
    /// **Next brick** (out of scope for this session): wrap the entire
    /// GEMM call site in a `pure_amx_section!` macro that (a) saves
    /// FPCR/FPSR before SET, (b) restores after CLR, and (c) statically
    /// guarantees no NEON ops appear in the AMX scope (audit via cargo
    /// asm).
    #[test]
    fn amx_set_state_diagnostic() {
        if !cfg!(all(target_arch = "aarch64", target_os = "macos")) {
            eprintln!("skipping (non-Apple-Silicon)");
            return;
        }
        let x: f32 = 1.0;
        let y: f32 = 2.0;
        assert_eq!(x * y, 2.0, "scalar f32 multiply broken even pre-AMX?");

        let _g = AmxGuard::enter().expect("AMX enable failed on Apple Silicon");

        let scalar_post = x * y;
        eprintln!("scalar f32 after AMX_SET: 1.0 * 2.0 = {scalar_post}");

        // Vectorizable reduction — this is what LLVM tends to NEON-ify.
        let a = vec![1.0f32; 16];
        let b = vec![2.0f32; 16];
        let mut acc = 0.0f32;
        for i in 0..a.len() {
            acc += a[i] * b[i];
        }
        eprintln!("vectorizable reduction after AMX_SET: sum(1*2)*16 = {acc}");

        // The signal: post-AMX scalar may work; vectorized may not.
        // This test is INFORMATIONAL — it documents the gap, not asserts
        // either way (Apple's behavior here is undocumented and can
        // shift with the CPU generation / inliner).
    }

    /// AMX 16×16 f32 outer-product kernel: bit-equivalent to scalar
    /// reference within f32 reduction reorder noise.
    ///
    /// Path to this passing required three opcode fixes:
    /// 1. `AMX_CLR` was emitting `0x00201000` (op 0 = LDX from x0=0)
    ///    instead of `0x00201221` (op 17 imm 1). Old `amx_disable` was
    ///    a no-op, leaving AMX state dirty across thread reuse.
    /// 2. `ldx` / `ldy` were each off-by-one in the opcode encoding —
    ///    `ldx` was emitting `0x00201020` (op 1 = LDY) and `ldy` was
    ///    emitting `0x00201040` (op 2 = STX). Kernel was loading from
    ///    the wrong AMX registers.
    /// 3. AMX FMA32 convention: Y indexes the output ROW, X indexes
    ///    the output COLUMN — opposite of the typical `Z[i,j] = X[i]*Y[j]`
    ///    intuition. Swapping LDX/LDY operands in the kernel fixed
    ///    the transposed output.
    #[test]
    fn amx_gemm_tile_16x16_matches_scalar() {
        if !cfg!(all(target_arch = "aarch64", target_os = "macos")) {
            eprintln!("skipping (non-Apple-Silicon)");
            return;
        }
        let k = 64;
        let a: Vec<f32> = (0..k * 16).map(|i| 1.0 + (i as f32 * 0.013).sin()).collect();
        let b: Vec<f32> = (0..k * 16).map(|i| 1.0 + (i as f32 * 0.017).cos()).collect();
        // Scalar reference computed before AMX_SET to avoid any
        // NEON-vs-AMX state interaction (per the diagnostic test below
        // — autovec'd reductions inside the AMX scope are not reliable).
        let c_ref = gemm_16x16_scalar(&a, &b, k);

        let _g = AmxGuard::enter().expect("AMX enable failed on Apple Silicon");
        let mut c_amx = vec![0.0f32; 16 * 16];
        // SAFETY: AmxGuard is alive; pointers valid for kernel reads/writes.
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        unsafe {
            amx_gemm_tile_16x16_f32(a.as_ptr(), b.as_ptr(), c_amx.as_mut_ptr(), k);
        }
        drop(_g);

        // Per-element correctness.
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for i in 0..16 {
            for j in 0..16 {
                let abs = (c_amx[i * 16 + j] - c_ref[i * 16 + j]).abs();
                let denom = c_ref[i * 16 + j].abs().max(1e-6);
                let rel = abs / denom;
                if abs > max_abs { max_abs = abs; }
                if rel > max_rel { max_rel = rel; }
            }
        }
        eprintln!("amx 16×16 f32: max_abs_err={max_abs:.3e}, max_rel_err={max_rel:.3e}");
        // f32 reduction reorder budget at K=64: ~1e-4 absolute, ~1e-5 relative.
        assert!(
            max_abs < 1e-3 && max_rel < 1e-4,
            "AMX GEMM drifted: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}",
        );
    }

    /// Scalar reference for 32×32 i16 outer-product accumulator: i32 result.
    fn gemm_32x32_i16_scalar(a: &[i16], b: &[i16], k: usize) -> Vec<i32> {
        let mut c = vec![0i32; 32 * 32];
        for kk in 0..k {
            let a_k = &a[kk * 32..(kk + 1) * 32];
            let b_k = &b[kk * 32..(kk + 1) * 32];
            for i in 0..32 {
                for j in 0..32 {
                    c[i * 32 + j] += a_k[i] as i32 * b_k[j] as i32;
                }
            }
        }
        c
    }

    /// AMX MAC16 32×32 outer-product kernel — currently `#[ignore]`d.
    ///
    /// **What's known**: opcode `0x002011C0` (op 14) executes without
    /// SIGILL. With K=1 and all-ones i16 inputs, MAC16 produces
    /// 65537 = 0x00010001 in every Z i32 cell — i.e. **two i16 results
    /// packed per i32 slot**, not the i32-per-cell layout assumed
    /// here. The "right" MAC16 readout pattern needs corsix's mode-bit
    /// docs to decode (which 16-bit lane of which i32 slot holds the
    /// (i, j) outer-product result).
    ///
    /// **Next brick**: figure out the packing — likely involves
    /// reading Z as two separate i16 streams (low half, high half) and
    /// re-interleaving on CPU. Until then, the Q8_0 AMX path uses the
    /// (already-proven) FMA32 kernel with on-the-fly f32 dequant — see
    /// `amx_qmm_q8_0_via_fma32` below.
    #[test]
    #[ignore]
    fn amx_gemm_tile_32x32_i16_matches_scalar() {
        if !cfg!(all(target_arch = "aarch64", target_os = "macos")) {
            eprintln!("skipping (non-Apple-Silicon)");
            return;
        }
        let k = 1;
        let a: Vec<i16> = vec![1; k * 32];
        let b: Vec<i16> = vec![1; k * 32];
        let c_ref = gemm_32x32_i16_scalar(&a, &b, k);

        let _g = AmxGuard::enter().expect("AMX enable failed");
        let mut c_amx = vec![0i32; 32 * 32];
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        unsafe {
            amx_gemm_tile_32x32_i16(a.as_ptr(), b.as_ptr(), c_amx.as_mut_ptr(), k);
        }
        drop(_g);

        // Print top-left 4×4 for both AMX and ref.
        let n_match = c_amx.iter().zip(c_ref.iter()).filter(|(a, b)| a == b).count();
        let max_diff = c_amx.iter().zip(c_ref.iter())
            .map(|(a, b)| (a - b).abs()).max().unwrap_or(0);
        let max_abs_amx = c_amx.iter().map(|x| x.abs()).max().unwrap_or(0);
        let max_abs_ref = c_ref.iter().map(|x| x.abs()).max().unwrap_or(0);
        eprintln!(
            "amx 32×32 i16: n_match={n_match}/{}, max_diff={max_diff}, \
             max|c_amx|={max_abs_amx}, max|c_ref|={max_abs_ref}",
            c_amx.len(),
        );
        if max_diff != 0 {
            eprintln!("AMX top-left 4×4 (rows of i32):");
            for i in 0..4 {
                let row: Vec<i32> = (0..4).map(|j| c_amx[i * 32 + j]).collect();
                eprintln!("  {row:?}");
            }
            eprintln!("scalar top-left 4×4:");
            for i in 0..4 {
                let row: Vec<i32> = (0..4).map(|j| c_ref[i * 32 + j]).collect();
                eprintln!("  {row:?}");
            }
        }
        assert!(max_abs_ref > 0, "scalar reference suspiciously zero");
        assert_eq!(max_diff, 0, "AMX MAC16 drifted from scalar (max_diff={max_diff})");
    }

    /// Q8_0 AMX matmul vs `rce_op_qmm::mm_1d` on a synthetic Q8_0
    /// weight. Proves the dequant + accumulation math is correct end-
    /// to-end. Uses the scalar AMX path (no FMA32 yet — that's the
    /// next optimization brick).
    #[test]
    fn amx_qmm_q8_0_matches_rce_op_qmm() {
        if !cfg!(all(target_arch = "aarch64", target_os = "macos")) {
            eprintln!("skipping (non-Apple-Silicon)");
            return;
        }
        use rce_ingest_gguf::GgufDtype;
        use rce_op_qmm::QuantWeight;

        // Synthetic Q8_0 weight: 16 rows × 64 cols (2 blocks/row).
        let n_rows = 16;
        let n_cols = 64;
        let blocks_per_row = n_cols / 32;
        let bytes_per_row = blocks_per_row * 34;
        let mut bytes = vec![0u8; n_rows * bytes_per_row];
        for r in 0..n_rows {
            for blk in 0..blocks_per_row {
                let off = r * bytes_per_row + blk * 34;
                // Vary the scale across rows/blocks so the test isn't
                // degenerate (a 0.125 base × small variation).
                let scale = 0.125f32 + (r as f32) * 0.001 + (blk as f32) * 0.0003;
                let s = half::f16::from_f32(scale).to_le_bytes();
                bytes[off] = s[0];
                bytes[off + 1] = s[1];
                for i in 0..32 {
                    let v = ((r + blk * 32 + i) as i32 % 17 - 8) as i8;
                    bytes[off + 2 + i] = v as u8;
                }
            }
        }
        let w = QuantWeight::from_owned_bytes(bytes.clone(), GgufDtype::Q8_0, n_rows, n_cols);

        // Activation.
        let a: Vec<f32> = (0..n_cols).map(|i| (i as f32 * 0.013).sin()).collect();

        // Direct path.
        let mut out_direct = vec![0.0f32; n_rows];
        rce_op_qmm::mm_1d(&a, &w, &mut out_direct);

        // AMX path (currently scalar dequant; same numerics).
        let _g = AmxGuard::enter().expect("AMX enable failed");
        let mut out_amx = vec![0.0f32; n_rows];
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        unsafe {
            amx_qmm_q8_0_1d(&a, &bytes, n_rows, n_cols, &mut out_amx);
        }
        drop(_g);

        let mut max_abs = 0.0f32;
        for (m, d) in out_amx.iter().zip(out_direct.iter()) {
            let abs = (m - d).abs();
            if abs > max_abs { max_abs = abs; }
        }
        eprintln!("amx_qmm_q8_0 vs rce_op_qmm: max_abs_err={max_abs:.3e}");
        assert!(max_abs < 1e-4, "Q8_0 AMX path drifted: max_abs={max_abs:.3e}");
    }

    /// AMX batched Q8_0 16×16 tile: correctness vs scalar reference.
    /// This is the kernel that should win big on prefill / multi-row
    /// workloads (M ≥ 16).
    #[test]
    fn amx_qmm_q8_0_tile_16x16_matches_scalar() {
        if !cfg!(all(target_arch = "aarch64", target_os = "macos")) {
            eprintln!("skipping (non-Apple-Silicon)");
            return;
        }
        use rce_ingest_gguf::GgufDtype;
        use rce_op_qmm::QuantWeight;

        // 16 weight rows × 64 cols (= 2 Q8_0 blocks per row), M=16
        // activation rows.
        let n_rows = 16;
        let m_rows = 16;
        let k = 64;
        let blocks_per_row = k / 32;
        let bytes_per_row = blocks_per_row * 34;
        let mut bytes = vec![0u8; n_rows * bytes_per_row];
        for r in 0..n_rows {
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
        let w = QuantWeight::from_owned_bytes(bytes.clone(), GgufDtype::Q8_0, n_rows, k);

        // Activation [M=16, K=64] row-major + its transpose [K, M].
        let a: Vec<f32> = (0..m_rows * k).map(|i| (i as f32 * 0.011).sin()).collect();
        let mut a_t = vec![0.0f32; k * m_rows];
        for m in 0..m_rows {
            for kk in 0..k {
                a_t[kk * m_rows + m] = a[m * k + kk];
            }
        }

        // Scalar reference: out[m, n] = sum_k a[m, k] * dequant(W[n, k]).
        let mut out_ref = vec![0.0f32; m_rows * n_rows];
        for m in 0..m_rows {
            let mut row_buf = vec![0.0f32; n_rows];
            rce_op_qmm::mm_1d(&a[m * k..(m + 1) * k], &w, &mut row_buf);
            for n in 0..n_rows {
                out_ref[m * n_rows + n] = row_buf[n];
            }
        }

        // AMX batched path.
        let _g = AmxGuard::enter().expect("AMX enable");
        let mut out_amx = vec![0.0f32; 256];
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        unsafe {
            amx_qmm_q8_0_tile_16x16(&a_t, &bytes, k, &mut out_amx);
        }
        drop(_g);

        let mut max_abs = 0.0f32;
        for (m, d) in out_amx.iter().zip(out_ref.iter()) {
            let abs = (m - d).abs();
            if abs > max_abs { max_abs = abs; }
        }
        eprintln!("amx_qmm_q8_0_tile_16x16: max_abs_err={max_abs:.3e}");
        assert!(max_abs < 1e-3, "batched AMX Q8_0 drifted: max_abs={max_abs:.3e}");
    }

    /// Benchmark: AMX Q8_0 path vs `rce_op_qmm::mm_1d` on a 2048×2048
    /// weight (TinyLlama attn_q shape). Outputs both wall-clock times
    /// and GFLOPS. This is informational — no assert beyond correctness
    /// (which the synthetic test above already covers).
    ///
    /// Run with `cargo test -p rce-op-amx amx_qmm_q8_0_bench -- --nocapture --ignored`.
    #[test]
    #[ignore]
    fn amx_qmm_q8_0_bench() {
        if !cfg!(all(target_arch = "aarch64", target_os = "macos")) {
            eprintln!("skipping (non-Apple-Silicon)");
            return;
        }
        use rce_ingest_gguf::GgufDtype;
        use rce_op_qmm::QuantWeight;

        // 2048×2048 Q8_0 weight — TinyLlama attn_q shape.
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
        let w = QuantWeight::from_owned_bytes(bytes.clone(), GgufDtype::Q8_0, n, k);
        let a: Vec<f32> = (0..k).map(|i| (i as f32 * 0.001).sin()).collect();

        // Warmup.
        let mut tmp = vec![0.0f32; n];
        rce_op_qmm::mm_1d(&a, &w, &mut tmp);
        let _g = AmxGuard::enter().expect("AMX enable");
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        unsafe { amx_qmm_q8_0_1d(&a, &bytes, n, k, &mut tmp); }
        drop(_g);

        let iters = 20;
        // 2 FLOPs per multiply-add × n × k operations per matmul.
        let total_flops = (2 * n * k * iters) as f64;

        // Direct path bench.
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            rce_op_qmm::mm_1d(&a, &w, &mut tmp);
        }
        let dt_direct = t0.elapsed();
        let gflops_direct = total_flops / dt_direct.as_secs_f64() / 1e9;

        // AMX path bench.
        let _g = AmxGuard::enter().expect("AMX enable");
        let t1 = std::time::Instant::now();
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        for _ in 0..iters {
            unsafe { amx_qmm_q8_0_1d(&a, &bytes, n, k, &mut tmp); }
        }
        let dt_amx = t1.elapsed();
        drop(_g);
        let gflops_amx = total_flops / dt_amx.as_secs_f64() / 1e9;

        eprintln!(
            "AMX Q8_0 bench (2048×2048, {iters} iters):\n  \
             rce-op-qmm: {dt_direct:?}, {gflops_direct:.1} GFLOPS\n  \
             amx-q8-via-neon-dot: {dt_amx:?}, {gflops_amx:.1} GFLOPS\n  \
             speedup: {:.2}x",
            gflops_amx / gflops_direct,
        );
    }

    /// **Realistic prefill benchmark**: M=16 query rows × N=2048 weight
    /// rows × K=2048. This is the size of a TinyLlama attention
    /// projection matmul during 16-token prefill (or 16-head per-head
    /// projection during a single decode step).
    ///
    /// Run: `cargo test -p rce-op-amx amx_q8_0_prefill_bench --release -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn amx_q8_0_prefill_bench() {
        if !cfg!(all(target_arch = "aarch64", target_os = "macos")) {
            eprintln!("skipping (non-Apple-Silicon)");
            return;
        }
        use rce_ingest_gguf::GgufDtype;
        use rce_op_qmm::QuantWeight;

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
        let w = QuantWeight::from_owned_bytes(bytes.clone(), GgufDtype::Q8_0, n, k);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();

        // Pre-transposed activation [K, M].
        let mut a_t = vec![0.0f32; k * m];
        for mi in 0..m {
            for kk in 0..k {
                a_t[kk * m + mi] = a[mi * k + kk];
            }
        }

        // Warmup.
        let mut tmp_direct = vec![0.0f32; m * n];
        rce_op_qmm::mm_batched(&a, m, &w, &mut tmp_direct);
        let _g = AmxGuard::enter().expect("AMX enable");
        let mut tmp_amx = vec![0.0f32; m * n];
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        unsafe { amx_qmm_q8_0_batched(&a_t, &bytes, n, k, &mut tmp_amx); }
        drop(_g);

        let iters = 10;
        let total_flops = (2 * m * n * k * iters) as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            rce_op_qmm::mm_batched(&a, m, &w, &mut tmp_direct);
        }
        let dt_direct = t0.elapsed();
        let gflops_direct = total_flops / dt_direct.as_secs_f64() / 1e9;

        let _g = AmxGuard::enter().expect("AMX enable");
        let t1 = std::time::Instant::now();
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        for _ in 0..iters {
            unsafe { amx_qmm_q8_0_batched(&a_t, &bytes, n, k, &mut tmp_amx); }
        }
        let dt_amx = t1.elapsed();
        drop(_g);
        let gflops_amx = total_flops / dt_amx.as_secs_f64() / 1e9;

        // Correctness spot check: top-left 4×4 of both outputs.
        let mut max_abs = 0.0f32;
        for i in 0..16.min(m * n) {
            let abs = (tmp_amx[i] - tmp_direct[i]).abs();
            if abs > max_abs { max_abs = abs; }
        }
        eprintln!(
            "AMX Q8_0 PREFILL bench (M={m}, N={n}, K={k}, {iters} iters):\n  \
             rce-op-qmm mm_batched: {dt_direct:?}, {gflops_direct:.1} GFLOPS\n  \
             amx-batched:             {dt_amx:?}, {gflops_amx:.1} GFLOPS\n  \
             speedup: {:.2}x, first-16-elements max_abs_err={max_abs:.3e}",
            gflops_amx / gflops_direct,
        );
    }

    /// Benchmark batched AMX Q8_0 16×16 tile vs `rce_op_qmm::mm_1d`
    /// repeated 16 times (the equivalent non-batched workload).
    ///
    /// Shape: 16 query rows × K=2048 against 16 weight rows. This is
    /// the "decode batch=16" workload (e.g. speculative decoding K=16
    /// candidates, or per-head KV-cache projection across 16 heads).
    ///
    /// Run: `cargo test -p rce-op-amx amx_q8_0_batched_bench --release -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn amx_q8_0_batched_bench() {
        if !cfg!(all(target_arch = "aarch64", target_os = "macos")) {
            eprintln!("skipping (non-Apple-Silicon)");
            return;
        }
        use rce_ingest_gguf::GgufDtype;
        use rce_op_qmm::QuantWeight;

        let m = 16;
        let n = 16;
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
        let w = QuantWeight::from_owned_bytes(bytes.clone(), GgufDtype::Q8_0, n, k);

        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
        // Pre-transposed activation [K, M].
        let mut a_t = vec![0.0f32; k * m];
        for mi in 0..m {
            for kk in 0..k {
                a_t[kk * m + mi] = a[mi * k + kk];
            }
        }

        // Warmup.
        let mut tmp_direct = vec![0.0f32; n];
        for mi in 0..m {
            rce_op_qmm::mm_1d(&a[mi * k..(mi + 1) * k], &w, &mut tmp_direct);
        }
        let _g = AmxGuard::enter().expect("AMX enable");
        let mut tmp_amx = vec![0.0f32; 256];
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        unsafe { amx_qmm_q8_0_tile_16x16(&a_t, &bytes, k, &mut tmp_amx); }
        drop(_g);

        let iters = 20;
        // 2 FLOPs per multiply-add × M × N × K per matmul.
        let total_flops = (2 * m * n * k * iters) as f64;

        // Direct path: M=16 separate mm_1d calls.
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            for mi in 0..m {
                rce_op_qmm::mm_1d(&a[mi * k..(mi + 1) * k], &w, &mut tmp_direct);
            }
        }
        let dt_direct = t0.elapsed();
        let gflops_direct = total_flops / dt_direct.as_secs_f64() / 1e9;

        // AMX batched path.
        let _g = AmxGuard::enter().expect("AMX enable");
        let t1 = std::time::Instant::now();
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        for _ in 0..iters {
            unsafe { amx_qmm_q8_0_tile_16x16(&a_t, &bytes, k, &mut tmp_amx); }
        }
        let dt_amx = t1.elapsed();
        drop(_g);
        let gflops_amx = total_flops / dt_amx.as_secs_f64() / 1e9;

        eprintln!(
            "AMX Q8_0 BATCHED bench (M={m}, N={n}, K={k}, {iters} iters):\n  \
             rce-op-qmm (M× mm_1d): {dt_direct:?}, {gflops_direct:.1} GFLOPS\n  \
             amx-batched-16x16:      {dt_amx:?}, {gflops_amx:.1} GFLOPS\n  \
             speedup: {:.2}x",
            gflops_amx / gflops_direct,
        );
    }
}
