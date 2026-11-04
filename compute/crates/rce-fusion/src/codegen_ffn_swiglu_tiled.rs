//! `rce-fusion::codegen_ffn_swiglu_tiled` — Phase R. Tiled variant
//! of Phase Q's FFN-swiglu kernel. Uses threadgroup memory to share
//! tiles of `input`, `gate_W`, and `up_W` across the `TILE_M × TILE_N`
//! threads of a threadgroup, with the input tile reused in BOTH the
//! gate and up dot products per K-block.
//!
//! ## Why tile when Phase Q already exists
//!
//! Phase Q's naive kernel reads `input[i, k]` once per `(i, j)` pair
//! into a register, amortising it across the gate and up accumulators
//! for that thread only. Threads in the same row but different columns
//! still re-read the same `input[i, k]` value because registers don't
//! cross threads. Phase R cooperatively loads `A_tile` into
//! threadgroup memory once per K-block; every column-thread reuses it.
//! The reuse factor goes from 2 (gate+up per thread) to `2 * TILE_N`
//! (gate+up per thread × every column-thread).
//!
//! Same reasoning as Phase I's tiled matmul vs Phase F's naive matmul,
//! applied to the parallel-double-matmul shape Phase Q targets.
//!
//! ## Tile dimensions
//!
//! Baked at compile time. `TILE_M = TILE_N = TILE_K = 16` matching
//! Phase I — 256 threads per threadgroup, well within Apple's 1024
//! budget. Equal dimensions keep the cooperative load pattern simple
//! (each thread loads exactly one A, one gate_B, one up_B element per
//! K-block).
//!
//! ## Bounds handling
//!
//! Same conservative posture as Phase I: shapes don't have to be
//! tile-multiples; loads guard the global reads with `< M, < K, < N`
//! and the output write is guarded too. Boundary threadgroups load
//! `0.0f` into bounds-violating tile slots and contribute nothing to
//! the dot products.
//!
//! ## Args contract
//!
//! Same as Phase Q: input `[M, K]`, gate_W `[K, N]`, up_W `[K, N]`,
//! output `[M, N]`, then `M K N` constants. The dispatch shape
//! differs (full-threadgroup `dispatch_threadgroups` like Phase I, not
//! `dispatch_threads` like Phase Q).

use crate::codegen::{BufferDecl, CodegenError, Dtype};

pub const TILE_M: u32 = 16;
pub const TILE_N: u32 = 16;
pub const TILE_K: u32 = 16;

/// What `emit_ffn_swiglu_tiled_msl` returns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmittedFfnSwigluTiledKernel {
    pub source: String,
    pub function_name: String,
    /// `[input, gate_W, up_W, output]`.
    pub buffers: Vec<BufferDecl>,
}

/// Dtype-suffixed canonical name. f32 → `fused_ffn_swiglu_tiled_f32`,
/// f16 → `fused_ffn_swiglu_tiled_f16`, bf16 → `fused_ffn_swiglu_tiled_bf16`.
pub fn canonical_function_name_dtype(dtype: Dtype) -> String {
    format!("fused_ffn_swiglu_tiled_{}", dtype.name())
}

/// Backward-compatible f32 wrapper.
pub fn canonical_function_name() -> String {
    canonical_function_name_dtype(Dtype::F32)
}

/// Inverse of `canonical_function_name_dtype`. Returns the matched
/// dtype, or `None` if `name` isn't a Phase R canonical name.
pub fn parse_function_name_with_dtype(name: &str) -> Option<Dtype> {
    match name {
        "fused_ffn_swiglu_tiled_f32" => Some(Dtype::F32),
        "fused_ffn_swiglu_tiled_f16" => Some(Dtype::F16),
        "fused_ffn_swiglu_tiled_bf16" => Some(Dtype::BF16),
        _ => None,
    }
}

/// Backward-compatible f32-only parser. Returns `Some(())` iff `name`
/// is exactly the f32 canonical name.
pub fn parse_function_name(name: &str) -> Option<()> {
    if name == "fused_ffn_swiglu_tiled_f32" { Some(()) } else { None }
}

/// Emit MSL for the tiled FFN-swiglu kernel at the given element
/// dtype. Threadgroup tiles + accumulators stay f32 regardless of
/// element type; narrowing happens only on the output store. f16 /
/// bf16 storage saves bandwidth on the input + weight tile loads.
pub fn emit_ffn_swiglu_tiled_msl_dtype(
    dtype: Dtype,
) -> Result<EmittedFfnSwigluTiledKernel, CodegenError> {
    let function_name = canonical_function_name_dtype(dtype);
    let buffers = vec![
        BufferDecl { name: "input",  dtype: dtype.msl_name() },
        BufferDecl { name: "gate_W", dtype: dtype.msl_name() },
        BufferDecl { name: "up_W",   dtype: dtype.msl_name() },
        BufferDecl { name: "output", dtype: dtype.msl_name() },
    ];
    let source = render_kernel(&function_name, dtype);
    Ok(EmittedFfnSwigluTiledKernel { source, function_name, buffers })
}

/// Backward-compatible f32 wrapper.
pub fn emit_ffn_swiglu_tiled_msl() -> Result<EmittedFfnSwigluTiledKernel, CodegenError> {
    emit_ffn_swiglu_tiled_msl_dtype(Dtype::F32)
}

fn render_kernel(function_name: &str, dtype: Dtype) -> String {
    let t = dtype.msl_name();
    let mut s = String::new();
    s.push_str("#include <metal_stdlib>\n");
    s.push_str("using namespace metal;\n\n");
    s.push_str(&format!("constant uint TILE_M = {TILE_M};\n"));
    s.push_str(&format!("constant uint TILE_N = {TILE_N};\n"));
    s.push_str(&format!("constant uint TILE_K = {TILE_K};\n\n"));
    s.push_str("kernel void ");
    s.push_str(function_name);
    s.push_str("(\n");
    s.push_str(&format!("    device const {t}* input  [[buffer(0)]],\n"));
    s.push_str(&format!("    device const {t}* gate_W [[buffer(1)]],\n"));
    s.push_str(&format!("    device const {t}* up_W   [[buffer(2)]],\n"));
    s.push_str(&format!("    device       {t}* output [[buffer(3)]],\n"));
    s.push_str("    constant uint& M [[buffer(4)]],\n");
    s.push_str("    constant uint& K [[buffer(5)]],\n");
    s.push_str("    constant uint& N [[buffer(6)]],\n");
    s.push_str("    uint2 tid  [[thread_position_in_threadgroup]],\n");
    s.push_str("    uint2 tgid [[threadgroup_position_in_grid]]\n");
    s.push_str(") {\n");
    // Threadgroup tiles stay f32 — shared register-rank state the
    // accumulator depends on; narrowing here would shrink bandwidth
    // saves but cost precision in the inner K-loop. Implicit
    // `t → float` widening on load keeps the inner loop identical
    // across dtypes.
    s.push_str("    threadgroup float A_tile[TILE_M][TILE_K];\n");
    s.push_str("    threadgroup float G_tile[TILE_K][TILE_N];\n");
    s.push_str("    threadgroup float U_tile[TILE_K][TILE_N];\n");
    s.push_str("    uint i = tgid.y * TILE_M + tid.y;\n");
    s.push_str("    uint j = tgid.x * TILE_N + tid.x;\n");
    s.push_str("    float gate_acc = 0.0f;\n");
    s.push_str("    float up_acc   = 0.0f;\n");
    s.push_str("    for (uint kk = 0; kk < K; kk += TILE_K) {\n");
    // Cooperative loads, bounds-guarded. Each thread loads one A,
    // one G, one U element per K-block.
    s.push_str("        uint a_col = kk + tid.x;\n");
    s.push_str("        A_tile[tid.y][tid.x] = (i < M && a_col < K)\n");
    s.push_str("            ? input[i * K + a_col]\n");
    s.push_str("            : 0.0f;\n");
    s.push_str("        uint b_row = kk + tid.y;\n");
    s.push_str("        G_tile[tid.y][tid.x] = (b_row < K && j < N)\n");
    s.push_str("            ? gate_W[b_row * N + j]\n");
    s.push_str("            : 0.0f;\n");
    s.push_str("        U_tile[tid.y][tid.x] = (b_row < K && j < N)\n");
    s.push_str("            ? up_W[b_row * N + j]\n");
    s.push_str("            : 0.0f;\n");
    s.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    // Dual dot product per output element, sharing the A_tile row.
    s.push_str("        for (uint k = 0; k < TILE_K; k++) {\n");
    s.push_str("            float a = A_tile[tid.y][k];\n");
    s.push_str("            gate_acc += a * G_tile[k][tid.x];\n");
    s.push_str("            up_acc   += a * U_tile[k][tid.x];\n");
    s.push_str("        }\n");
    s.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    s.push_str("    }\n");
    s.push_str("    if (i >= M || j >= N) return;\n");
    s.push_str("    float g_silu = gate_acc / (1.0f + exp(-gate_acc));\n");
    match dtype {
        Dtype::F32 => s.push_str("    output[i * N + j] = g_silu * up_acc;\n"),
        Dtype::F16 => s.push_str("    output[i * N + j] = half(g_silu * up_acc);\n"),
        Dtype::BF16 => s.push_str("    output[i * N + j] = bfloat(g_silu * up_acc);\n"),
    }
    s.push_str("}\n");
    s
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_dims_consistent_with_threadgroup_budget() {
        assert!(TILE_M * TILE_N <= 1024);
        assert_eq!(TILE_M, TILE_K);
        assert_eq!(TILE_N, TILE_K);
    }

    #[test]
    fn function_name_is_stable() {
        assert_eq!(canonical_function_name(), "fused_ffn_swiglu_tiled_f32");
    }

    #[test]
    fn parse_function_name_disambiguates_from_phase_q_naive() {
        assert!(parse_function_name("fused_ffn_swiglu_tiled_f32").is_some());
        // Phase Q naive name is REJECTED — different module owns it.
        assert!(parse_function_name("fused_ffn_swiglu_f32").is_none());
        // And Phase Q's parser must reject Phase R's name.
        assert!(crate::codegen_ffn_swiglu::parse_function_name("fused_ffn_swiglu_tiled_f32").is_none());
    }

    #[test]
    fn emit_returns_three_input_buffers_plus_output() {
        let k = emit_ffn_swiglu_tiled_msl().unwrap();
        assert_eq!(k.function_name, "fused_ffn_swiglu_tiled_f32");
        let names: Vec<&str> = k.buffers.iter().map(|b| b.name).collect();
        assert_eq!(names, vec!["input", "gate_W", "up_W", "output"]);
    }

    #[test]
    fn emit_shares_input_tile_across_both_matmuls() {
        let k = emit_ffn_swiglu_tiled_msl().unwrap();
        // Three threadgroup tiles declared.
        assert!(k.source.contains("threadgroup float A_tile[TILE_M][TILE_K];"));
        assert!(k.source.contains("threadgroup float G_tile[TILE_K][TILE_N];"));
        assert!(k.source.contains("threadgroup float U_tile[TILE_K][TILE_N];"));
        // The A_tile row is referenced ONCE per inner-K-iteration into
        // a register `a`, then used in BOTH dot products. That's the
        // module's whole reason for existing — single load, double use.
        assert_eq!(k.source.matches("float a = A_tile[tid.y][k];").count(), 1);
        assert!(k.source.contains("gate_acc += a * G_tile[k][tid.x];"));
        assert!(k.source.contains("up_acc   += a * U_tile[k][tid.x];"));
        // Two barriers (one after the cooperative load, one after the
        // dot loop) just like Phase I.
        assert_eq!(k.source.matches("threadgroup_barrier(mem_flags::mem_threadgroup);").count(), 2);
    }

    #[test]
    fn emit_oob_guard_before_silu_and_store() {
        let k = emit_ffn_swiglu_tiled_msl().unwrap();
        let oob_pos   = k.source.find("if (i >= M || j >= N) return;").unwrap();
        let silu_pos  = k.source.find("g_silu = gate_acc").unwrap();
        let store_pos = k.source.find("output[i * N + j] = g_silu * up_acc;").unwrap();
        assert!(oob_pos < silu_pos && silu_pos < store_pos);
    }

    #[test]
    fn emit_buffer_slots_are_contiguous() {
        let k = emit_ffn_swiglu_tiled_msl().unwrap();
        let mut seen: Vec<u32> = Vec::new();
        for chunk in k.source.split("[[buffer(") {
            if let Some(rest) = chunk.split(")]]").next() {
                if let Ok(n) = rest.trim().parse::<u32>() { seen.push(n); }
            }
        }
        seen.sort();
        let expected: Vec<u32> = (0..seen.len() as u32).collect();
        assert_eq!(seen, expected);
    }

    #[test]
    fn canonical_name_per_dtype() {
        assert_eq!(canonical_function_name_dtype(Dtype::F32), "fused_ffn_swiglu_tiled_f32");
        assert_eq!(canonical_function_name_dtype(Dtype::F16), "fused_ffn_swiglu_tiled_f16");
        assert_eq!(canonical_function_name_dtype(Dtype::BF16), "fused_ffn_swiglu_tiled_bf16");
    }

    #[test]
    fn parse_with_dtype_round_trips_every_variant() {
        for d in [Dtype::F32, Dtype::F16, Dtype::BF16] {
            assert_eq!(parse_function_name_with_dtype(&canonical_function_name_dtype(d)), Some(d));
        }
        assert!(parse_function_name_with_dtype("fused_ffn_swiglu_f32").is_none()); // Phase Q name
        assert!(parse_function_name_with_dtype("fused_ffn_swiglu_tiled").is_none());
        // Naked f32 parser stays exact-match.
        assert!(parse_function_name("fused_ffn_swiglu_tiled_f16").is_none());
        assert!(parse_function_name("fused_ffn_swiglu_tiled_bf16").is_none());
    }

    #[test]
    fn emit_f16_uses_half_buffers_and_narrowing_store() {
        let k = emit_ffn_swiglu_tiled_msl_dtype(Dtype::F16).unwrap();
        assert_eq!(k.function_name, "fused_ffn_swiglu_tiled_f16");
        for b in &k.buffers { assert_eq!(b.dtype, "half"); }
        assert!(k.source.contains("device const half* input"));
        assert!(k.source.contains("device       half* output"));
        // Threadgroup tiles stay f32 even when storage is half.
        assert!(k.source.contains("threadgroup float A_tile[TILE_M][TILE_K];"));
        assert!(k.source.contains("threadgroup float G_tile[TILE_K][TILE_N];"));
        assert!(k.source.contains("threadgroup float U_tile[TILE_K][TILE_N];"));
        assert!(k.source.contains("output[i * N + j] = half(g_silu * up_acc);"));
    }

    #[test]
    fn emit_bf16_uses_bfloat_buffers_and_narrowing_store() {
        let k = emit_ffn_swiglu_tiled_msl_dtype(Dtype::BF16).unwrap();
        assert_eq!(k.function_name, "fused_ffn_swiglu_tiled_bf16");
        for b in &k.buffers { assert_eq!(b.dtype, "bfloat"); }
        assert!(k.source.contains("device const bfloat* input"));
        assert!(k.source.contains("device       bfloat* output"));
        assert!(k.source.contains("threadgroup float A_tile[TILE_M][TILE_K];"));
        assert!(k.source.contains("output[i * N + j] = bfloat(g_silu * up_acc);"));
    }

    #[test]
    fn tile_load_pattern_is_dtype_invariant() {
        // The single-A-read-per-inner-K-iteration invariant must hold
        // for every dtype — same justification as Phase Q.
        for d in [Dtype::F32, Dtype::F16, Dtype::BF16] {
            let k = emit_ffn_swiglu_tiled_msl_dtype(d).unwrap();
            assert_eq!(k.source.matches("float a = A_tile[tid.y][k];").count(), 1, "dtype {d:?}");
            assert!(k.source.contains("gate_acc += a * G_tile[k][tid.x];"));
            assert!(k.source.contains("up_acc   += a * U_tile[k][tid.x];"));
            assert_eq!(k.source.matches("threadgroup_barrier(mem_flags::mem_threadgroup);").count(), 2);
        }
    }
}
