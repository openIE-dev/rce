//! `rce-fusion::codegen_matmul_epilogue` — Phase F of the codegen
//! rollout. Emits MSL for `matmul → <chain of elementwise fragments>`
//! kernels: a single GPU dispatch computes `output = epilogue(input @
//! weight)`, with the dot-product running in registers and the
//! epilogue applied per output element before the single write.
//!
//! ## Why this is a different module from `codegen`
//!
//! The Phase A-E `codegen` module assumes a per-element or per-group
//! kernel shape: one thread per group, a single input slot, a register
//! `v` that walks the chain. Matmul breaks that contract:
//!
//! - Each thread owns one output element `(i, j)`, not one input element.
//! - The "input" inside the kernel is a `K`-long dot product, not a
//!   single buffer read.
//! - The grid is 2D `(M, N)`, not 1D `ngroups`.
//! - Uniforms (e.g. swiglu's `up`) are `[M, N]`-shaped and indexed at
//!   `(i, j)`, not `[total]`-shaped at `base + i`.
//!
//! Trying to retrofit the Phase A emitter to handle both shapes makes
//! both worse. A parallel module that produces a clean, matmul-specific
//! template is the right move.
//!
//! ## What this module covers
//!
//! - `matmul` is always the chain head — implicit, not in the
//!   `epilogue` argument.
//! - The epilogue is a chain of elementwise fragments from the Phase A
//!   table (`silu`, `gelu`, `swiglu`). Each fragment's per-element
//!   body is interpolated into the dot-product's tail, with uniforms
//!   reindexed for the `(i, j)` 2D layout.
//! - Reduction-bearing fragments (`rms_norm`) are **rejected** —
//!   their per-group accumulator can't compose with matmul's per-
//!   output-element threading without an extra synchronization step
//!   this module doesn't emit.
//! - f32 only, naive matmul (no tiling, no shared-memory K-blocking).
//!   Production-grade matmul still goes through BLAS / MPS; this
//!   kernel exists to make `matmul + elementwise` a single dispatch
//!   for small-to-medium shapes where the elementwise epilogue
//!   dominates kernel-launch overhead.
//!
//! ## Args contract
//!
//! Buffer table (returned in `EmittedMatmulKernel.buffers`):
//!   slot 0:    input   (`device const float*`, shape `[M, K]`)
//!   slot 1:    weight  (`device const float*`, shape `[K, N]`)
//!   slot 2:    output  (`device       float*`, shape `[M, N]`)
//!   slot 3..:  each epilogue fragment's uniforms, in chain order,
//!              deduped by name. Uniform tensors are shape `[M, N]`
//!              and indexed at `(i, j) = i * N + j` inside the kernel.
//!
//! Constants:
//!   `constant uint& M`, `constant uint& K`, `constant uint& N` at the
//!   next three slots after uniforms. Then each fragment's scalars in
//!   chain order, deduped by name.

use crate::codegen::{fragments, BufferDecl, CodegenError, Dtype, MslFragment, ScalarDecl};

/// Same buffer/scalar layout descriptors as the per-element codegen
/// module, re-exported for callers that consume both layouts.
pub use crate::codegen::{BufferDecl as MatmulBufferDecl, ScalarDecl as MatmulScalarDecl};

/// What `emit_matmul_epilogue_msl` returns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmittedMatmulKernel {
    pub source: String,
    pub function_name: String,
    /// `[input, weight, output, <uniforms in chain order>]`.
    pub buffers: Vec<BufferDecl>,
    /// `[<scalars in chain order>]` — the `M, K, N` constants are
    /// implicit (always present) and not listed here.
    pub scalars: Vec<ScalarDecl>,
}

/// Stable function name for a matmul + epilogue chain at a given
/// dtype. Same epilogue + dtype ⇒ same name ⇒ same pipeline cache hit.
pub fn canonical_function_name_dtype(epilogue: &[&str], dtype: Dtype) -> String {
    let mut s = String::from("fused_matmul");
    for op in epilogue {
        s.push('_');
        s.push_str(op);
    }
    s.push('_');
    s.push_str(dtype.name());
    s
}

/// Backward-compatible wrapper defaulting to f32.
pub fn canonical_function_name(epilogue: &[&str]) -> String {
    canonical_function_name_dtype(epilogue, Dtype::F32)
}

/// Inverse of `canonical_function_name_dtype`: parse the epilogue
/// chain + dtype out of a matmul-fused function name. Returns `None`
/// if the name doesn't have the `fused_matmul_<...>_<dtype>` shape,
/// or if the body can't be tokenised into known fragments.
///
/// Empty epilogue (`fused_matmul_<dtype>`) returns
/// `Some((vec![], dtype))`.
pub fn parse_function_name_with_dtype(name: &str) -> Option<(Vec<&'static str>, Dtype)> {
    // Strip the dtype suffix; longest first so `_bf16` wins over
    // a lexical prefix of `_f16`.
    let (body, dtype) = if let Some(b) = name.strip_prefix("fused_matmul").and_then(|s| s.strip_suffix("_bf16")) {
        (b, Dtype::BF16)
    } else if let Some(b) = name.strip_prefix("fused_matmul").and_then(|s| s.strip_suffix("_f32")) {
        (b, Dtype::F32)
    } else if let Some(b) = name.strip_prefix("fused_matmul").and_then(|s| s.strip_suffix("_f16")) {
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

/// Backward-compatible wrapper: returns the chain only, regardless of
/// dtype.
pub fn parse_function_name(name: &str) -> Option<Vec<&'static str>> {
    parse_function_name_with_dtype(name).map(|(c, _)| c)
}

/// Emit MSL for `output = epilogue(input @ weight)` at the given
/// dtype. f32 / f16 / bf16 produce distinct kernels: buffer types
/// differ, function name suffix differs. Per-element math stays in
/// f32 registers regardless; only the load / store narrows.
pub fn emit_matmul_epilogue_msl_dtype(
    epilogue: &[&str],
    dtype: Dtype,
) -> Result<EmittedMatmulKernel, CodegenError> {
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

    // Tensor buffers (input / weight / output / fragment uniforms)
    // take the element dtype. Scalars stay f32.
    let mut buffers: Vec<BufferDecl> = vec![
        BufferDecl { name: "input",  dtype: dtype.msl_name() },
        BufferDecl { name: "weight", dtype: dtype.msl_name() },
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
    Ok(EmittedMatmulKernel { source, function_name, buffers, scalars })
}

/// Backward-compatible wrapper defaulting to f32. The graph rewriter
/// + existing tests stay on this entry point unless they explicitly
/// thread dtype through.
pub fn emit_matmul_epilogue_msl(
    epilogue: &[&str],
) -> Result<EmittedMatmulKernel, CodegenError> {
    emit_matmul_epilogue_msl_dtype(epilogue, Dtype::F32)
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
    out.push_str(&format!(
        "    device const {}* input  [[buffer({})]],\n",
        buffers[0].dtype, slot,
    ));
    slot += 1;
    out.push_str(&format!(
        "    device const {}* weight [[buffer({})]],\n",
        buffers[1].dtype, slot,
    ));
    slot += 1;
    out.push_str(&format!(
        "    device       {}* output [[buffer({})]],\n",
        buffers[2].dtype, slot,
    ));
    slot += 1;
    for b in &buffers[3..] {
        out.push_str(&format!(
            "    device const {}* {} [[buffer({})]],\n",
            b.dtype, b.name, slot,
        ));
        slot += 1;
    }
    out.push_str(&format!("    constant uint& M [[buffer({})]],\n", slot)); slot += 1;
    out.push_str(&format!("    constant uint& K [[buffer({})]],\n", slot)); slot += 1;
    out.push_str(&format!("    constant uint& N [[buffer({})]],\n", slot)); slot += 1;
    for s in scalars {
        out.push_str(&format!(
            "    constant {}& {} [[buffer({})]],\n",
            s.dtype, s.name, slot,
        ));
        slot += 1;
    }
    out.push_str("    uint2 gid [[thread_position_in_grid]]\n");
    out.push_str(") {\n");
    // Bounds + indexing. gid.x → N, gid.y → M (matches MTLSize width = N,
    // height = M conventions used at dispatch time).
    out.push_str("    if (gid.x >= N || gid.y >= M) return;\n");
    out.push_str("    uint i = gid.y;\n");
    out.push_str("    uint j = gid.x;\n");
    // Dot product into register v.
    out.push_str("    float v = 0.0f;\n");
    out.push_str("    for (uint k = 0; k < K; k++) {\n");
    out.push_str("        v += input[i * K + k] * weight[k * N + j];\n");
    out.push_str("    }\n");

    // Epilogue: each fragment's body, with `base + i` rewritten to the
    // 2D `(i, j)` indexing matmul uses. We translate the per-element
    // fragment body verbatim — the only substitution is `base + i` →
    // `i * N + j`. Fragment bodies that reference `base + i` outside
    // uniform indexing will break; the SILU/GELU bodies don't, and
    // SWIGLU's only reference is `up[base + i]` → `up[i * N + j]`.
    for f in frags {
        if !f.elementwise.is_empty() {
            let rewritten = f.elementwise.replace("base + i", "i * N + j");
            out.push_str(&rewritten);
            out.push('\n');
        }
    }

    // Dtype-aware store: f32 plain, f16 / bf16 with explicit
    // narrowing cast (keeps the f32 golden test for f32 unchanged).
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
    fn empty_epilogue_emits_plain_matmul() {
        // Plain `fused_matmul_f32` — no elementwise after the dot. Useful
        // as the baseline against which epilogue chains are compared.
        let k = emit_matmul_epilogue_msl(&[]).unwrap();
        assert_eq!(k.function_name, "fused_matmul_f32");
        assert_eq!(k.buffers.iter().map(|b| b.name).collect::<Vec<_>>(),
            vec!["input", "weight", "output"]);
        assert!(k.scalars.is_empty());
        assert!(k.source.contains("v += input[i * K + k] * weight[k * N + j];"));
        assert!(k.source.contains("output[i * N + j] = v;"));
        // No epilogue fragments → no transformations between dot and write.
        assert!(!k.source.contains("v = v / (1.0f + exp(-v));"));
    }

    #[test]
    fn matmul_silu_inlines_silu_after_dot() {
        let k = emit_matmul_epilogue_msl(&["silu"]).unwrap();
        assert_eq!(k.function_name, "fused_matmul_silu_f32");
        // silu appears once, after the dot product, before the write.
        let dot_pos = k.source.find("v += input[i * K + k] * weight[k * N + j];").unwrap();
        let silu_pos = k.source.find("v = v / (1.0f + exp(-v));").unwrap();
        let store_pos = k.source.find("output[i * N + j] = v;").unwrap();
        assert!(dot_pos < silu_pos && silu_pos < store_pos);
    }

    #[test]
    fn matmul_silu_gelu_runs_both_in_order_post_dot() {
        let k = emit_matmul_epilogue_msl(&["silu", "gelu"]).unwrap();
        assert_eq!(k.function_name, "fused_matmul_silu_gelu_f32");
        let silu_pos = k.source.find("v = v / (1.0f + exp(-v));").unwrap();
        let gelu_pos = k.source.find("0.7978845608028654").unwrap();
        assert!(silu_pos < gelu_pos);
    }

    #[test]
    fn matmul_swiglu_inlines_up_at_2d_index() {
        // swiglu's body uses `up[base + i]` in the per-element module;
        // in this module that gets rewritten to `up[i * N + j]` so the
        // 2D matmul output indexes the uniform consistently.
        let k = emit_matmul_epilogue_msl(&["swiglu"]).unwrap();
        assert_eq!(k.function_name, "fused_matmul_swiglu_f32");
        assert_eq!(k.buffers.iter().map(|b| b.name).collect::<Vec<_>>(),
            vec!["input", "weight", "output", "up"]);
        assert!(k.source.contains("v = (v / (1.0f + exp(-v))) * up[i * N + j];"));
        // The original `base + i` form must NOT survive.
        assert!(!k.source.contains("up[base + i]"));
    }

    #[test]
    fn rms_norm_in_epilogue_rejected() {
        let err = emit_matmul_epilogue_msl(&["rms_norm"]).unwrap_err();
        assert_eq!(err, CodegenError::TooManyReductions { count: 1 });
    }

    #[test]
    fn unknown_op_in_epilogue_rejected() {
        let err = emit_matmul_epilogue_msl(&["silu", "not_a_real_op"]).unwrap_err();
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
            vec!["gelu", "silu"],
            vec!["swiglu"],
            vec!["silu", "swiglu"],
        ] {
            let name = canonical_function_name(&epilogue);
            let parsed = parse_function_name(&name)
                .unwrap_or_else(|| panic!("parse failed for `{name}`"));
            assert_eq!(parsed, epilogue, "round-trip failed for `{name}`");
        }
    }

    #[test]
    fn parse_function_name_rejects_rms_norm() {
        // rms_norm is intentionally NOT in this module's KNOWN list —
        // the epilogue layout can't carry an output-axis reduction.
        // A hypothetical `fused_matmul_rms_norm_f32` therefore won't
        // parse here (it'd parse via the per-element module, but that
        // dispatcher wouldn't handle a matmul-shaped call).
        assert!(parse_function_name("fused_matmul_rms_norm_f32").is_none());
    }

    #[test]
    fn parse_function_name_rejects_garbage() {
        assert!(parse_function_name("not_fused").is_none());
        assert!(parse_function_name("fused_matmul_zzz_f32").is_none());
        // Missing suffix.
        assert!(parse_function_name("fused_matmul_silu").is_none());
    }

    #[test]
    fn buffer_slots_are_unique_and_sequential() {
        let k = emit_matmul_epilogue_msl(&["swiglu"]).unwrap();
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

    /// Golden MSL pin for the simplest non-trivial case (matmul + silu).
    /// Drift here must be a deliberate update.
    #[test]
    // ── Phase M: matmul-epilogue × {f16, bf16} ──────────────────────────

    #[test]
    fn matmul_epilogue_f16_emits_half_buffers_and_cast() {
        let k = emit_matmul_epilogue_msl_dtype(&["silu"], Dtype::F16).unwrap();
        assert_eq!(k.function_name, "fused_matmul_silu_f16");
        assert!(k.source.contains("device const half* input"));
        assert!(k.source.contains("device const half* weight"));
        assert!(k.source.contains("device       half* output"));
        // Register and dot accumulator stay f32.
        assert!(k.source.contains("float v = 0.0f;"));
        // Output store has the narrowing cast.
        assert!(k.source.contains("output[i * N + j] = half(v);"));
    }

    #[test]
    fn matmul_epilogue_bf16_emits_bfloat_buffers_and_cast() {
        let k = emit_matmul_epilogue_msl_dtype(&["silu", "gelu"], Dtype::BF16).unwrap();
        assert_eq!(k.function_name, "fused_matmul_silu_gelu_bf16");
        assert!(k.source.contains("device const bfloat* input"));
        assert!(k.source.contains("device const bfloat* weight"));
        assert!(k.source.contains("device       bfloat* output"));
        assert!(k.source.contains("output[i * N + j] = bfloat(v);"));
    }

    #[test]
    fn matmul_epilogue_swiglu_f16_propagates_dtype_to_up_uniform() {
        let k = emit_matmul_epilogue_msl_dtype(&["swiglu"], Dtype::F16).unwrap();
        // `up` uniform follows the element dtype (tensor data).
        assert!(k.source.contains("device const half* up"));
    }

    #[test]
    fn matmul_epilogue_parse_function_name_with_dtype_round_trips() {
        for (epilogue, dtype) in [
            (vec![], Dtype::F32),
            (vec![], Dtype::F16),
            (vec![], Dtype::BF16),
            (vec!["silu"], Dtype::F32),
            (vec!["silu"], Dtype::F16),
            (vec!["silu"], Dtype::BF16),
            (vec!["silu", "gelu"], Dtype::BF16),
            (vec!["swiglu"], Dtype::F16),
        ] {
            let name = canonical_function_name_dtype(&epilogue, dtype);
            let (parsed_epilogue, parsed_dtype) = parse_function_name_with_dtype(&name)
                .unwrap_or_else(|| panic!("parse failed for `{name}`"));
            assert_eq!(parsed_epilogue, epilogue, "epilogue round-trip for `{name}`");
            assert_eq!(parsed_dtype, dtype, "dtype round-trip for `{name}`");
        }
    }

    #[test]
    fn matmul_epilogue_three_dtypes_produce_three_function_names() {
        let f32 = canonical_function_name_dtype(&["silu"], Dtype::F32);
        let f16 = canonical_function_name_dtype(&["silu"], Dtype::F16);
        let bf16 = canonical_function_name_dtype(&["silu"], Dtype::BF16);
        assert_eq!(f32, "fused_matmul_silu_f32");
        assert_eq!(f16, "fused_matmul_silu_f16");
        assert_eq!(bf16, "fused_matmul_silu_bf16");
    }

    #[test]
    fn matmul_epilogue_parser_bf16_before_f16_priority() {
        // `_bf16` ends with `_f16` — confirm dtype-suffix priority.
        let (e, d) = parse_function_name_with_dtype("fused_matmul_silu_bf16").unwrap();
        assert_eq!(e, vec!["silu"]);
        assert_eq!(d, Dtype::BF16);
        let (e, d) = parse_function_name_with_dtype("fused_matmul_silu_f16").unwrap();
        assert_eq!(e, vec!["silu"]);
        assert_eq!(d, Dtype::F16);
    }

    #[test]
    fn matmul_epilogue_backward_compat_parse_accepts_all_dtypes() {
        assert_eq!(parse_function_name("fused_matmul_silu_f32"), Some(vec!["silu"]));
        assert_eq!(parse_function_name("fused_matmul_silu_f16"), Some(vec!["silu"]));
        assert_eq!(parse_function_name("fused_matmul_silu_bf16"), Some(vec!["silu"]));
    }

    #[test]
    fn golden_msl_matmul_silu() {
        let k = emit_matmul_epilogue_msl(&["silu"]).unwrap();
        let expected = "\
#include <metal_stdlib>
using namespace metal;

kernel void fused_matmul_silu_f32(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= N || gid.y >= M) return;
    uint i = gid.y;
    uint j = gid.x;
    float v = 0.0f;
    for (uint k = 0; k < K; k++) {
        v += input[i * K + k] * weight[k * N + j];
    }
    v = v / (1.0f + exp(-v));
    output[i * N + j] = v;
}
";
        assert_eq!(k.source, expected, "golden MSL drift — update both fragments and golden deliberately");
    }
}
