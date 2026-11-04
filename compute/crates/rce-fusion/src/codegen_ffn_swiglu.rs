//! `rce-fusion::codegen_ffn_swiglu` — Phase Q. Emits MSL for the
//! transformer-FFN gate+up+swiglu pattern:
//!
//!   gate   = input @ gate_W           # [M, K] @ [K, N] → [M, N]
//!   up     = input @ up_W             # [M, K] @ [K, N] → [M, N]
//!   output = silu(gate) * up
//!
//! Both matmuls share the same `input`. In two separate matmul
//! dispatches (the path Phase F produces when paired with a CPU up
//! matmul) each `input[i, k]` is read TWICE — once into the gate
//! kernel, once into the up kernel. This module emits one kernel that
//! reads each `input[i, k]` once and accumulates BOTH dot products in
//! the same K-loop, then composes silu(gate_acc) * up_acc into a single
//! output store.
//!
//! ## Why this can't be expressed via Phase F or Phase H
//!
//! - **Phase F** (`codegen_matmul_epilogue`) handles `matmul →
//!   <epilogue>`. swiglu in its fragment-table form expects `up` as a
//!   precomputed `[M, N]` uniform — meaning the up matmul has to run
//!   on the host first. Phase F saves the gate matmul + swiglu, not
//!   the up matmul.
//! - **Phase H** (`codegen_matmul_matmul`) handles sequential
//!   `A @ B @ C` — a triple product, not two parallel matmuls
//!   sharing the left operand. Different chain shape.
//!
//! Phase Q is the missing third lane: parallel matmuls with a shared
//! input + an elementwise fuse-in.
//!
//! ## What this module covers
//!
//! - **f32, f16, bf16** (Phase U dtype rollout) — same suffix
//!   convention as Phase K/L/M/N. Accumulators stay f32 internally
//!   regardless of element dtype; narrowing happens on the output
//!   store. The naked entry points (`emit_ffn_swiglu_msl`,
//!   `canonical_function_name`, `parse_function_name`) stay f32-only
//!   for backwards compatibility; the `_dtype` / `_with_dtype`
//!   variants thread dtype through.
//! - Naive matmul (one thread per output element, K-loop in
//!   registers). No tiling — see [`codegen_ffn_swiglu_tiled`] for the
//!   tiled variant.
//! - The epilogue is **fixed** as `silu(gate) * up`. Generalising to
//!   `<elementwise>(gate) <op> up` would be a fragment-table
//!   extension; the FFN-swiglu shape is the one Jamba (and most
//!   transformer FFNs) actually emit.
//!
//! ## Args contract
//!
//! Buffer layout (returned in `EmittedFfnSwigluKernel.buffers`):
//!   slot 0:   input   (`device const float*`, shape `[M, K]`)
//!   slot 1:   gate_W  (`device const float*`, shape `[K, N]`)
//!   slot 2:   up_W    (`device const float*`, shape `[K, N]`)
//!   slot 3:   output  (`device       float*`, shape `[M, N]`)
//!
//! Constants:
//!   `constant uint& M`, `constant uint& K`, `constant uint& N`.

use crate::codegen::{BufferDecl, CodegenError, Dtype};

/// What `emit_ffn_swiglu_msl` returns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmittedFfnSwigluKernel {
    pub source: String,
    pub function_name: String,
    /// `[input, gate_W, up_W, output]`.
    pub buffers: Vec<BufferDecl>,
}

/// Dtype-suffixed canonical name. f32 → `fused_ffn_swiglu_f32`,
/// f16 → `fused_ffn_swiglu_f16`, bf16 → `fused_ffn_swiglu_bf16`.
pub fn canonical_function_name_dtype(dtype: Dtype) -> String {
    format!("fused_ffn_swiglu_{}", dtype.name())
}

/// Backward-compatible f32 wrapper. Defaults to `Dtype::F32`.
pub fn canonical_function_name() -> String {
    canonical_function_name_dtype(Dtype::F32)
}

/// Inverse of `canonical_function_name_dtype`. Returns the matched
/// dtype, or `None` if `name` isn't a Phase Q canonical name.
pub fn parse_function_name_with_dtype(name: &str) -> Option<Dtype> {
    match name {
        "fused_ffn_swiglu_f32" => Some(Dtype::F32),
        "fused_ffn_swiglu_f16" => Some(Dtype::F16),
        "fused_ffn_swiglu_bf16" => Some(Dtype::BF16),
        _ => None,
    }
}

/// Backward-compatible f32 parser. Returns `Some(())` iff `name` is
/// exactly the f32 canonical name. Use [`parse_function_name_with_dtype`]
/// to recognise the f16 / bf16 variants too.
pub fn parse_function_name(name: &str) -> Option<()> {
    if name == "fused_ffn_swiglu_f32" { Some(()) } else { None }
}

/// Emit MSL for `output = silu(input @ gate_W) * (input @ up_W)` at
/// the given element dtype. Accumulators stay f32 regardless of
/// element type; the only narrowing is on the output store.
pub fn emit_ffn_swiglu_msl_dtype(dtype: Dtype) -> Result<EmittedFfnSwigluKernel, CodegenError> {
    let function_name = canonical_function_name_dtype(dtype);
    let buffers = vec![
        BufferDecl { name: "input",   dtype: dtype.msl_name() },
        BufferDecl { name: "gate_W",  dtype: dtype.msl_name() },
        BufferDecl { name: "up_W",    dtype: dtype.msl_name() },
        BufferDecl { name: "output",  dtype: dtype.msl_name() },
    ];
    let source = render_kernel(&function_name, dtype);
    Ok(EmittedFfnSwigluKernel { source, function_name, buffers })
}

/// Backward-compatible f32 wrapper. Calls `emit_ffn_swiglu_msl_dtype(Dtype::F32)`.
pub fn emit_ffn_swiglu_msl() -> Result<EmittedFfnSwigluKernel, CodegenError> {
    emit_ffn_swiglu_msl_dtype(Dtype::F32)
}

fn render_kernel(function_name: &str, dtype: Dtype) -> String {
    let t = dtype.msl_name();
    let mut s = String::new();
    s.push_str("#include <metal_stdlib>\n");
    s.push_str("using namespace metal;\n\n");
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
    s.push_str("    uint2 gid [[thread_position_in_grid]]\n");
    s.push_str(") {\n");
    s.push_str("    if (gid.x >= N || gid.y >= M) return;\n");
    s.push_str("    uint i = gid.y;\n");
    s.push_str("    uint j = gid.x;\n");
    // Dual accumulator. ONE pass over K; each input[i, k] load is reused
    // for both gate and up — the bandwidth amplifier that motivates
    // this module. The `float a = input[...]` widens half/bfloat on
    // load, so `a * gate_W[...]` and `a * up_W[...]` run in f32.
    s.push_str("    float gate_acc = 0.0f;\n");
    s.push_str("    float up_acc   = 0.0f;\n");
    s.push_str("    for (uint k = 0; k < K; k++) {\n");
    s.push_str("        float a = input[i * K + k];\n");
    s.push_str("        gate_acc += a * gate_W[k * N + j];\n");
    s.push_str("        up_acc   += a * up_W[k * N + j];\n");
    s.push_str("    }\n");
    // silu(gate) * up, written inline. Store narrows on f16/bf16.
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
    fn function_name_is_stable() {
        assert_eq!(canonical_function_name(), "fused_ffn_swiglu_f32");
    }

    #[test]
    fn parse_function_name_is_exact_match_only() {
        assert!(parse_function_name("fused_ffn_swiglu_f32").is_some());
        assert!(parse_function_name("fused_ffn_swiglu").is_none());
        assert!(parse_function_name("fused_ffn_swiglu_f16").is_none());
        assert!(parse_function_name("fused_matmul_swiglu_f32").is_none());
        assert!(parse_function_name("not_fused").is_none());
    }

    #[test]
    fn emit_returns_three_input_buffers_plus_output() {
        let k = emit_ffn_swiglu_msl().unwrap();
        assert_eq!(k.function_name, "fused_ffn_swiglu_f32");
        let names: Vec<&str> = k.buffers.iter().map(|b| b.name).collect();
        assert_eq!(names, vec!["input", "gate_W", "up_W", "output"]);
        for b in &k.buffers { assert_eq!(b.dtype, "float"); }
    }

    #[test]
    fn emit_reads_input_once_per_k_iteration() {
        let k = emit_ffn_swiglu_msl().unwrap();
        // Exactly one read of `input[i * K + k]` per K-loop body —
        // that's the bandwidth amplifier the module's whole reason
        // for existing. Two reads (one per matmul, separate loops)
        // would defeat the whole point.
        assert_eq!(k.source.matches("input[i * K + k]").count(), 1);
        // Both weight reads + both accumulators present.
        assert!(k.source.contains("gate_W[k * N + j]"));
        assert!(k.source.contains("up_W[k * N + j]"));
        assert!(k.source.contains("gate_acc"));
        assert!(k.source.contains("up_acc"));
        // silu(gate) * up_acc, written inline.
        assert!(k.source.contains("gate_acc / (1.0f + exp(-gate_acc))"));
        assert!(k.source.contains("g_silu * up_acc"));
    }

    #[test]
    fn emit_buffer_slots_are_contiguous() {
        let k = emit_ffn_swiglu_msl().unwrap();
        let mut seen: Vec<u32> = Vec::new();
        for chunk in k.source.split("[[buffer(") {
            if let Some(rest) = chunk.split(")]]").next() {
                if let Ok(n) = rest.trim().parse::<u32>() { seen.push(n); }
            }
        }
        seen.sort();
        let expected: Vec<u32> = (0..seen.len() as u32).collect();
        assert_eq!(seen, expected, "buffer slots must be 0..n contiguous");
    }

    #[test]
    fn canonical_name_per_dtype() {
        assert_eq!(canonical_function_name_dtype(Dtype::F32), "fused_ffn_swiglu_f32");
        assert_eq!(canonical_function_name_dtype(Dtype::F16), "fused_ffn_swiglu_f16");
        assert_eq!(canonical_function_name_dtype(Dtype::BF16), "fused_ffn_swiglu_bf16");
    }

    #[test]
    fn parse_with_dtype_round_trips_every_variant() {
        for d in [Dtype::F32, Dtype::F16, Dtype::BF16] {
            let name = canonical_function_name_dtype(d);
            assert_eq!(parse_function_name_with_dtype(&name), Some(d));
        }
        assert!(parse_function_name_with_dtype("fused_ffn_swiglu").is_none());
        assert!(parse_function_name_with_dtype("not_fused").is_none());
        // The naked f32-only parser MUST NOT accept f16/bf16 (it's the
        // backwards-compat single-shape parser).
        assert!(parse_function_name("fused_ffn_swiglu_f16").is_none());
        assert!(parse_function_name("fused_ffn_swiglu_bf16").is_none());
    }

    #[test]
    fn emit_f16_uses_half_buffers_and_narrowing_store() {
        let k = emit_ffn_swiglu_msl_dtype(Dtype::F16).unwrap();
        assert_eq!(k.function_name, "fused_ffn_swiglu_f16");
        for b in &k.buffers { assert_eq!(b.dtype, "half"); }
        assert!(k.source.contains("device const half* input"));
        assert!(k.source.contains("device       half* output"));
        // Accumulators stay f32 internally.
        assert!(k.source.contains("float gate_acc = 0.0f;"));
        assert!(k.source.contains("float up_acc   = 0.0f;"));
        // Store narrows.
        assert!(k.source.contains("output[i * N + j] = half(g_silu * up_acc);"));
    }

    #[test]
    fn emit_bf16_uses_bfloat_buffers_and_narrowing_store() {
        let k = emit_ffn_swiglu_msl_dtype(Dtype::BF16).unwrap();
        assert_eq!(k.function_name, "fused_ffn_swiglu_bf16");
        for b in &k.buffers { assert_eq!(b.dtype, "bfloat"); }
        assert!(k.source.contains("device const bfloat* input"));
        assert!(k.source.contains("device       bfloat* output"));
        assert!(k.source.contains("float gate_acc = 0.0f;"));
        assert!(k.source.contains("output[i * N + j] = bfloat(g_silu * up_acc);"));
    }

    #[test]
    fn input_read_once_per_k_loop_regardless_of_dtype() {
        // The bandwidth amplifier — verified by the f32 test above —
        // MUST hold for f16/bf16 too. One `float a = input[...]`
        // load per K iteration; reused for both matmuls.
        for d in [Dtype::F16, Dtype::BF16] {
            let k = emit_ffn_swiglu_msl_dtype(d).unwrap();
            assert_eq!(k.source.matches("input[i * K + k]").count(), 1, "dtype {d:?}");
            assert_eq!(k.source.matches("float a = input").count(), 1, "dtype {d:?}");
        }
    }

    /// Golden MSL pin. Any drift in the dual-accumulator template
    /// must come with a deliberate update to this string.
    #[test]
    fn golden_msl_ffn_swiglu() {
        let k = emit_ffn_swiglu_msl().unwrap();
        let expected = "\
#include <metal_stdlib>
using namespace metal;

kernel void fused_ffn_swiglu_f32(
    device const float* input  [[buffer(0)]],
    device const float* gate_W [[buffer(1)]],
    device const float* up_W   [[buffer(2)]],
    device       float* output [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= N || gid.y >= M) return;
    uint i = gid.y;
    uint j = gid.x;
    float gate_acc = 0.0f;
    float up_acc   = 0.0f;
    for (uint k = 0; k < K; k++) {
        float a = input[i * K + k];
        gate_acc += a * gate_W[k * N + j];
        up_acc   += a * up_W[k * N + j];
    }
    float g_silu = gate_acc / (1.0f + exp(-gate_acc));
    output[i * N + j] = g_silu * up_acc;
}
";
        assert_eq!(k.source, expected, "golden MSL drift — update template + golden deliberately");
    }
}
