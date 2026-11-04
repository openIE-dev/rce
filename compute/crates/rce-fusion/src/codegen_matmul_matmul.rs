//! `rce-fusion::codegen_matmul_matmul` — Phase H of the codegen
//! rollout. Emits MSL for `output = epilogue(A @ B @ C)` kernels: two
//! matmuls and an optional elementwise epilogue, all in one dispatch
//! with the intermediate `[M, K2]` matrix kept in registers and never
//! written to memory.
//!
//! ## What it is
//!
//! A naive double-matmul template. Each thread owns one output element
//! `(i, j) ∈ [M, N]` and computes:
//!
//!   `D[i, j] = sum_{l=0..K2} (sum_{k=0..K1} A[i, k] * B[k, l]) * C[l, j]`
//!
//! followed by the epilogue fragments inline before the single output
//! store. No threadgroup memory, no tiling — that's Phase I.
//!
//! ## When it's a win
//!
//! Compute count vs sequential `(A @ B) @ C`:
//!   - Sequential: `M*K2*K1 + M*N*K2 = M*K2*(K1 + N)` ops, plus a
//!     `M*K2` write + read for the intermediate.
//!   - Fused: `M*N*K1*K2 + M*N*K2 ≈ M*N*K1*K2` ops, zero intermediate
//!     traffic.
//!
//! Fused does more arithmetic per element (factor `N` more for the
//! inner `A @ B` recomputation) but saves the intermediate bandwidth.
//! The crossover depends on shape: low `N` and bandwidth-bound
//! hardware (e.g. unified-memory Apple silicon at large `M*K2`) favour
//! fused; very large `N` favours sequential. Profile the specific
//! shape before choosing — Phase H exposes the mechanism, not a
//! routing decision.
//!
//! ## Args contract
//!
//! Buffer table (returned in `EmittedMatmul2Kernel.buffers`):
//!   slot 0:    A       (`device const float*`, shape `[M, K1]`)
//!   slot 1:    B       (`device const float*`, shape `[K1, K2]`)
//!   slot 2:    C       (`device const float*`, shape `[K2, N]`)
//!   slot 3:    output  (`device       float*`, shape `[M, N]`)
//!   slot 4..:  each epilogue fragment's uniforms, shape `[M, N]`,
//!              indexed at `i * N + j`.
//!
//! Constants: `constant uint& M`, `K1`, `K2`, `N` at the next four
//! slots after uniforms. Then each fragment's scalars in chain order.
//!
//! ## Scope limits
//!
//! - Exactly two matmuls (`A @ B @ C`). Three+ would need another K
//!   dimension and a separate template; out of scope for Phase H.
//! - f32 only. Same first-cut posture as Phases A–G.
//! - Epilogue is elementwise-only (no reductions). Same reason as
//!   Phase F: per-output-element threading + output-axis reduction
//!   don't compose.

use crate::codegen::{fragments, BufferDecl, CodegenError, Dtype, MslFragment, ScalarDecl};

/// What `emit_matmul_matmul_msl` returns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmittedMatmul2Kernel {
    pub source: String,
    pub function_name: String,
    /// `[A, B, C, output, <uniforms in chain order>]`.
    pub buffers: Vec<BufferDecl>,
    pub scalars: Vec<ScalarDecl>,
}

/// Stable function name for `A @ B @ C → epilogue`. Same epilogue
/// (and the implicit two-matmul head) ⇒ same name ⇒ same pipeline
/// cache hit.
pub fn canonical_function_name_dtype(epilogue: &[&str], dtype: Dtype) -> String {
    let mut s = String::from("fused_matmul2");
    for op in epilogue {
        s.push('_');
        s.push_str(op);
    }
    s.push('_');
    s.push_str(dtype.name());
    s
}

/// Backward-compatible f32 wrapper.
pub fn canonical_function_name(epilogue: &[&str]) -> String {
    canonical_function_name_dtype(epilogue, Dtype::F32)
}

/// Inverse of `canonical_function_name_dtype`: parse epilogue + dtype.
pub fn parse_function_name_with_dtype(name: &str) -> Option<(Vec<&'static str>, Dtype)> {
    let (body, dtype) = if let Some(b) = name.strip_prefix("fused_matmul2").and_then(|s| s.strip_suffix("_bf16")) {
        (b, Dtype::BF16)
    } else if let Some(b) = name.strip_prefix("fused_matmul2").and_then(|s| s.strip_suffix("_f32")) {
        (b, Dtype::F32)
    } else if let Some(b) = name.strip_prefix("fused_matmul2").and_then(|s| s.strip_suffix("_f16")) {
        (b, Dtype::F16)
    } else {
        return None;
    };
    if body.is_empty() {
        return Some((Vec::new(), dtype));
    }
    let body = body.strip_prefix('_')?;
    const KNOWN: &[&str] = &["silu", "gelu", "swiglu"];
    let mut ops: Vec<&'static str> = Vec::new();
    let mut s = body;
    loop {
        let mut best: Option<&'static str> = None;
        for &op in KNOWN {
            if let Some(rest) = s.strip_prefix(op) {
                if rest.is_empty() || rest.starts_with('_') {
                    if best.map(|b| op.len() > b.len()).unwrap_or(true) {
                        best = Some(op);
                    }
                }
            }
        }
        let op = best?;
        ops.push(op);
        s = &s[op.len()..];
        if s.is_empty() { break; }
        s = s.strip_prefix('_')?;
    }
    Some((ops, dtype))
}

/// Backward-compatible parse — discards dtype.
pub fn parse_function_name(name: &str) -> Option<Vec<&'static str>> {
    parse_function_name_with_dtype(name).map(|(c, _)| c)
}

/// Emit MSL for `output = epilogue(A @ B @ C)` at the given dtype.
pub fn emit_matmul_matmul_msl_dtype(
    epilogue: &[&str],
    dtype: Dtype,
) -> Result<EmittedMatmul2Kernel, CodegenError> {
    let mut frags: Vec<&MslFragment> = Vec::with_capacity(epilogue.len());
    for op in epilogue {
        match fragments::lookup(op) {
            Some(f) => frags.push(f),
            None => return Err(CodegenError::UnknownOp((*op).to_string())),
        }
    }
    let n_red = frags.iter().filter(|f| f.has_reduction()).count();
    if n_red > 0 {
        return Err(CodegenError::TooManyReductions { count: n_red });
    }

    let function_name = canonical_function_name_dtype(epilogue, dtype);

    let mut buffers: Vec<BufferDecl> = vec![
        BufferDecl { name: "A",      dtype: dtype.msl_name() },
        BufferDecl { name: "B",      dtype: dtype.msl_name() },
        BufferDecl { name: "C",      dtype: dtype.msl_name() },
        BufferDecl { name: "output", dtype: dtype.msl_name() },
    ];
    for f in &frags {
        for u in f.uniforms {
            if !buffers.iter().any(|b| b.name == u.name) {
                buffers.push(BufferDecl { name: u.name, dtype: dtype.msl_name() });
            }
        }
    }
    let mut scalars: Vec<ScalarDecl> = Vec::new();
    for f in &frags {
        for s in f.scalars {
            if !scalars.iter().any(|d| d.name == s.name) {
                scalars.push(s.clone());
            }
        }
    }

    let source = render_kernel(&function_name, dtype, &frags, &buffers, &scalars);
    Ok(EmittedMatmul2Kernel { source, function_name, buffers, scalars })
}

/// Backward-compatible f32 wrapper.
pub fn emit_matmul_matmul_msl(
    epilogue: &[&str],
) -> Result<EmittedMatmul2Kernel, CodegenError> {
    emit_matmul_matmul_msl_dtype(epilogue, Dtype::F32)
}

fn render_kernel(
    function_name: &str,
    dtype: Dtype,
    frags: &[&MslFragment],
    buffers: &[BufferDecl],
    scalars: &[ScalarDecl],
) -> String {
    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\n");
    out.push_str("using namespace metal;\n\n");

    out.push_str("kernel void ");
    out.push_str(function_name);
    out.push_str("(\n");

    let mut slot: usize = 0;
    out.push_str(&format!("    device const {}* A      [[buffer({})]],\n", buffers[0].dtype, slot)); slot += 1;
    out.push_str(&format!("    device const {}* B      [[buffer({})]],\n", buffers[1].dtype, slot)); slot += 1;
    out.push_str(&format!("    device const {}* C      [[buffer({})]],\n", buffers[2].dtype, slot)); slot += 1;
    out.push_str(&format!("    device       {}* output [[buffer({})]],\n", buffers[3].dtype, slot)); slot += 1;
    for b in &buffers[4..] {
        out.push_str(&format!(
            "    device const {}* {} [[buffer({})]],\n",
            b.dtype, b.name, slot,
        ));
        slot += 1;
    }
    out.push_str(&format!("    constant uint& M  [[buffer({})]],\n", slot)); slot += 1;
    out.push_str(&format!("    constant uint& K1 [[buffer({})]],\n", slot)); slot += 1;
    out.push_str(&format!("    constant uint& K2 [[buffer({})]],\n", slot)); slot += 1;
    out.push_str(&format!("    constant uint& N  [[buffer({})]],\n", slot)); slot += 1;
    for s in scalars {
        out.push_str(&format!(
            "    constant {}& {} [[buffer({})]],\n",
            s.dtype, s.name, slot,
        ));
        slot += 1;
    }
    out.push_str("    uint2 gid [[thread_position_in_grid]]\n");
    out.push_str(") {\n");
    out.push_str("    if (gid.x >= N || gid.y >= M) return;\n");
    out.push_str("    uint i = gid.y;\n");
    out.push_str("    uint j = gid.x;\n");
    // Double dot product. Outer over l ∈ [0, K2): accumulate
    // (A@B)[i, l] * C[l, j]. Inner over k ∈ [0, K1): build the
    // intermediate's `(i, l)` element in a register.
    out.push_str("    float v = 0.0f;\n");
    out.push_str("    for (uint l = 0; l < K2; l++) {\n");
    out.push_str("        float ab_il = 0.0f;\n");
    out.push_str("        for (uint k = 0; k < K1; k++) {\n");
    out.push_str("            ab_il += A[i * K1 + k] * B[k * K2 + l];\n");
    out.push_str("        }\n");
    out.push_str("        v += ab_il * C[l * N + j];\n");
    out.push_str("    }\n");

    // Epilogue: same 2D-indexing rewrite as Phase F.
    for f in frags {
        if !f.elementwise.is_empty() {
            let rewritten = f.elementwise.replace("base + i", "i * N + j");
            out.push_str(&rewritten);
            out.push('\n');
        }
    }

    match dtype {
        Dtype::F32 => out.push_str("    output[i * N + j] = v;\n"),
        Dtype::F16 => out.push_str("    output[i * N + j] = half(v);\n"),
        Dtype::BF16 => out.push_str("    output[i * N + j] = bfloat(v);\n"),
    }
    out.push_str("}\n");
    out
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_epilogue_emits_plain_double_matmul() {
        let k = emit_matmul_matmul_msl(&[]).unwrap();
        assert_eq!(k.function_name, "fused_matmul2_f32");
        assert_eq!(k.buffers.iter().map(|b| b.name).collect::<Vec<_>>(),
            vec!["A", "B", "C", "output"]);
        assert!(k.scalars.is_empty());
        // Double dot: outer K2 loop containing inner K1 loop.
        assert!(k.source.contains("for (uint l = 0; l < K2; l++)"));
        assert!(k.source.contains("for (uint k = 0; k < K1; k++)"));
        assert!(k.source.contains("ab_il += A[i * K1 + k] * B[k * K2 + l];"));
        assert!(k.source.contains("v += ab_il * C[l * N + j];"));
        assert!(k.source.contains("output[i * N + j] = v;"));
    }

    #[test]
    fn matmul2_silu_inlines_silu_after_double_dot() {
        let k = emit_matmul_matmul_msl(&["silu"]).unwrap();
        assert_eq!(k.function_name, "fused_matmul2_silu_f32");
        let close_k2_loop = k.source.rfind("    }\n").unwrap();
        let silu_pos = k.source.find("v = v / (1.0f + exp(-v));").unwrap();
        let store_pos = k.source.find("output[i * N + j] = v;").unwrap();
        assert!(silu_pos > close_k2_loop, "silu must run after the K2 loop closes");
        assert!(silu_pos < store_pos, "silu must precede the output store");
    }

    #[test]
    fn matmul2_swiglu_rewrites_up_to_2d_index() {
        let k = emit_matmul_matmul_msl(&["swiglu"]).unwrap();
        assert_eq!(k.function_name, "fused_matmul2_swiglu_f32");
        assert_eq!(k.buffers.iter().map(|b| b.name).collect::<Vec<_>>(),
            vec!["A", "B", "C", "output", "up"]);
        assert!(k.source.contains("v = (v / (1.0f + exp(-v))) * up[i * N + j];"));
        assert!(!k.source.contains("up[base + i]"));
    }

    #[test]
    fn rms_norm_epilogue_rejected() {
        let err = emit_matmul_matmul_msl(&["rms_norm"]).unwrap_err();
        assert_eq!(err, CodegenError::TooManyReductions { count: 1 });
    }

    #[test]
    fn unknown_op_rejected() {
        let err = emit_matmul_matmul_msl(&["silu", "not_a_real_op"]).unwrap_err();
        match err {
            CodegenError::UnknownOp(n) => assert_eq!(n, "not_a_real_op"),
            other => panic!("expected UnknownOp, got {other:?}"),
        }
    }

    #[test]
    fn parse_function_name_round_trips() {
        for epilogue in [
            vec![],
            vec!["silu"],
            vec!["gelu"],
            vec!["silu", "gelu"],
            vec!["swiglu"],
        ] {
            let name = canonical_function_name(&epilogue);
            let parsed = parse_function_name(&name)
                .unwrap_or_else(|| panic!("parse failed for `{name}`"));
            assert_eq!(parsed, epilogue, "round-trip failed for `{name}`");
        }
    }

    #[test]
    fn parse_function_name_disambiguates_from_single_matmul_family() {
        // `fused_matmul_silu_f32` belongs to Phase F (single-matmul +
        // epilogue), NOT to this module. This parser must reject it.
        assert!(parse_function_name("fused_matmul_silu_f32").is_none());
        // The matching prefix is `fused_matmul2`.
        assert!(parse_function_name("fused_matmul2_silu_f32").is_some());
        // And the Phase F module's parser must reject this module's names.
        assert!(crate::codegen_matmul_epilogue::parse_function_name("fused_matmul2_silu_f32").is_none());
    }

    #[test]
    fn buffer_slots_are_unique_and_sequential() {
        let k = emit_matmul_matmul_msl(&["swiglu"]).unwrap();
        let mut seen: Vec<u32> = Vec::new();
        for chunk in k.source.split("[[buffer(") {
            if let Some(rest) = chunk.split(")]]").next() {
                if let Ok(n) = rest.trim().parse::<u32>() {
                    seen.push(n);
                }
            }
        }
        seen.sort();
        let expected: Vec<u32> = (0..seen.len() as u32).collect();
        assert_eq!(seen, expected, "buffer slots must be 0..n contiguous");
    }

    /// Golden MSL pin for `fused_matmul2_silu_f32`. Drift in either the
    /// double-dot template or the SILU fragment must come with a
    /// deliberate update.
    #[test]
    fn golden_msl_matmul2_silu() {
        let k = emit_matmul_matmul_msl(&["silu"]).unwrap();
        let expected = "\
#include <metal_stdlib>
using namespace metal;

kernel void fused_matmul2_silu_f32(
    device const float* A      [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device const float* C      [[buffer(2)]],
    device       float* output [[buffer(3)]],
    constant uint& M  [[buffer(4)]],
    constant uint& K1 [[buffer(5)]],
    constant uint& K2 [[buffer(6)]],
    constant uint& N  [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= N || gid.y >= M) return;
    uint i = gid.y;
    uint j = gid.x;
    float v = 0.0f;
    for (uint l = 0; l < K2; l++) {
        float ab_il = 0.0f;
        for (uint k = 0; k < K1; k++) {
            ab_il += A[i * K1 + k] * B[k * K2 + l];
        }
        v += ab_il * C[l * N + j];
    }
    v = v / (1.0f + exp(-v));
    output[i * N + j] = v;
}
";
        assert_eq!(k.source, expected, "golden MSL drift — update fragments and golden deliberately");
    }
}
