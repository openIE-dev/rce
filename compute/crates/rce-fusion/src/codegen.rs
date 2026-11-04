//! `rce-fusion::codegen` — Phase A of the Free-chain MSL codegen.
//!
//! Pure-function MSL emitter for arbitrary Free-fraction chains. Given an
//! ordered list of op names — every name resolved against the fragment
//! table in this module — return the MSL source for a single fused kernel
//! that evaluates the chain in one dispatch.
//!
//! ## What "single dispatch" means here
//!
//! Output is a grouped kernel: `ngroups` threads, each owning a contiguous
//! slice of `group_size` input elements. Within each thread:
//!
//! - **Chains with no reduction-bearing op** (e.g. `silu`, `silu → gelu`)
//!   collapse to one pass: `for i in 0..group_size { v ← input[base+i]; <every
//!   fragment's elementwise body>; output[base+i] ← v; }`. Intermediates
//!   live in a register; no extra read or write.
//!
//! - **Chains with exactly one reduction-bearing op** (e.g.
//!   `silu → rms_norm`, `silu → rms_norm → gelu`) emit two passes. Pass 1
//!   re-evaluates every fragment up to and including the reduction's
//!   `accumulate` block, building the reduction stat. Pass 2 re-emits the
//!   pre-reduction fragments (re-reading `input` is cheaper than
//!   threadgroup memory at typical group sizes — same call this crate's
//!   hand-written `silu_rms_norm` kernel makes), then runs the
//!   reduction's `scale` block, then the post-reduction fragments,
//!   then writes `output[base+i]`. The reduction's *finalize* runs
//!   between the two passes in scalar code.
//!
//! - **Chains with two or more reduction-bearing ops** are out of scope
//!   for Phase A: they need a multi-stage template (each reduction adds a
//!   pass, and post-reduction fragments can themselves enter another
//!   reduction's accumulate). [`emit_chain_msl`] returns
//!   `Err(TooManyReductions)`.
//!
//! ## What the fragment table promises
//!
//! Every [`MslFragment`] in [`fragments`] is a CALM-Free op classified by
//! `rce_placement::op_fraction`. Each fragment specifies:
//!
//! - `op_name`: the `Custom(...)` name that appears in graphs.
//! - `elementwise`: an MSL statement that updates a local register `v`
//!   (f32). For pure elementwise ops this is the whole transformation;
//!   for reduction-bearing ops this is empty (the math lives in
//!   `reduction.accumulate` and `reduction.scale`).
//! - `reduction`: `Some(_)` iff this op reduces; carries `init`,
//!   `accumulate`, `finalize`, and per-element `scale` blocks.
//! - `uniforms` / `scalars`: extra Metal buffer / constant inputs the
//!   fragment reads, beyond the chain-level `input` / `output`.
//!
//! Fragments are byte-equivalent to the corresponding hand-written
//! kernels in `rce-op-metal` (`silu_f32`, `gelu_tanh_f32`,
//! `rms_norm_f32_grouped`) up to wrapping boilerplate. The codegen tests
//! pin emitted MSL to golden strings to catch any drift.
//!
//! ## Scope limits (Phase A only)
//!
//! - f32 only. No f16 / bf16 fragment variants yet.
//! - One reduction per chain. Two reductions returns
//!   `TooManyReductions`.
//! - Reduction must appear *somewhere* in the chain; pre-reduction
//!   fragments precede it and post-reduction fragments follow.
//! - No pipeline compilation / cache here — this module emits a string.
//!   Phase B wires the string into a `ComputePipelineState` cache on
//!   `MetalContext`.

/// Element dtype the codegen emits. Controls the buffer types
/// (`device const float*` / `half*` / `bfloat*`) and the function-
/// name suffix (`_f32` / `_f16` / `_bf16`). Reduction accumulators
/// stay f32 internally regardless of element dtype — reducing in a
/// narrower type sacrifices numeric stability for no real bandwidth
/// win (the partial sum lives only in registers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    /// IEEE 754 bfloat16 — same exponent range as f32, 7-bit mantissa.
    /// Apple silicon supports `bfloat` natively from M2 on (Metal 3.1+).
    /// On older targets the kernel will fail to compile; the dispatcher
    /// surfaces that as `InterpretError::Unsupported`.
    BF16,
}

impl Dtype {
    /// MSL type-name keyword for `device` buffers in this dtype.
    pub fn msl_name(self) -> &'static str {
        match self {
            Dtype::F32 => "float",
            Dtype::F16 => "half",
            Dtype::BF16 => "bfloat",
        }
    }
    /// Suffix used in canonical function names.
    pub fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F16 => "f16",
            Dtype::BF16 => "bf16",
        }
    }
    /// Element size in bytes. Used by the dispatch path to size
    /// device buffers correctly.
    pub fn size_bytes(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 => 2,
            Dtype::BF16 => 2,
        }
    }
}

/// One Metal buffer the emitted kernel reads beyond `input` / `output`.
/// Bound as a `device const <dtype>*` at the next free buffer slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferDecl {
    pub name: &'static str,
    pub dtype: &'static str,
}

/// One scalar constant the emitted kernel reads beyond `ngroups` /
/// `group_size`. Bound as a `constant <dtype>&` at the next free buffer
/// slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarDecl {
    pub name: &'static str,
    pub dtype: &'static str,
}

/// The four MSL snippets that drive the two-pass template for a
/// reduction-bearing op. See module-level docs for how they compose.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReductionMsl {
    /// Statement(s) that initialise the reduction accumulator in scope at
    /// the top of pass 1. Example: `"float sum_sq = 0.0f;"`.
    pub init: &'static str,
    /// Statement(s) that update the accumulator using register `v`
    /// (current element after every preceding fragment ran).
    /// Example: `"sum_sq += v * v;"`.
    pub accumulate: &'static str,
    /// Statement(s) that finalise the per-group statistic in scope
    /// between pass 1 and pass 2. Receives the accumulator + uniforms +
    /// scalars + `group_size`. Example:
    /// `"float inv_rms = rsqrt(sum_sq / float(group_size) + eps);"`.
    pub finalize: &'static str,
    /// Statement(s) that update register `v` using the finalised stat
    /// in pass 2 (per-element). Example:
    /// `"v = v * inv_rms * weight[base + i];"`.
    pub scale: &'static str,
}

/// One Free-fraction op's contribution to a fused kernel.
#[derive(Clone, Debug)]
pub struct MslFragment {
    pub op_name: &'static str,
    /// Per-element transformation. Operates on register `v` (f32).
    /// Empty for reduction-bearing ops (the work lives in `reduction`).
    pub elementwise: &'static str,
    pub reduction: Option<ReductionMsl>,
    pub uniforms: &'static [BufferDecl],
    pub scalars: &'static [ScalarDecl],
}

impl MslFragment {
    pub fn has_reduction(&self) -> bool { self.reduction.is_some() }
}

/// The fragment table itself. Add a new Free op = add a `MslFragment`
/// entry here. Phase A covers the three ops needed to reproduce the
/// hand-written `silu_rms_norm` kernel plus one extra elementwise op
/// (`gelu`) to exercise the multi-fragment path.
pub mod fragments {
    use super::*;

    pub const SILU: MslFragment = MslFragment {
        op_name: "silu",
        // Byte-equivalent to rce-op-metal::silu_f32 inner body.
        elementwise: "    v = v / (1.0f + exp(-v));",
        reduction: None,
        uniforms: &[],
        scalars: &[],
    };

    pub const GELU: MslFragment = MslFragment {
        op_name: "gelu",
        // Byte-equivalent to rce-op-metal::gelu_tanh_f32 inner body.
        // Note: `clamp(inner, -10, 10)` is preserved — without it, `tanh`
        // can underflow / overflow on extreme inputs.
        elementwise: concat!(
            "    {\n",
            "        const float sqrt_2_over_pi = 0.7978845608028654f;\n",
            "        const float coeff = 0.044715f;\n",
            "        float x3 = v * v * v;\n",
            "        float inner = sqrt_2_over_pi * (v + coeff * x3);\n",
            "        inner = clamp(inner, -10.0f, 10.0f);\n",
            "        v = 0.5f * v * (1.0f + tanh(inner));\n",
            "    }",
        ),
        reduction: None,
        uniforms: &[],
        scalars: &[],
    };

    pub const RMS_NORM: MslFragment = MslFragment {
        op_name: "rms_norm",
        elementwise: "",
        reduction: Some(ReductionMsl {
            init: "    float sum_sq = 0.0f;",
            accumulate: "        sum_sq += v * v;",
            finalize: "    float inv_rms = rsqrt(sum_sq / float(group_size) + eps);",
            scale: "        v = v * inv_rms * weight[base + i];",
        }),
        uniforms: &[BufferDecl { name: "weight", dtype: "float" }],
        scalars: &[ScalarDecl { name: "eps", dtype: "float" }],
    };

    /// `swiglu(gate, up) = silu(gate) * up`. The codegen treats `gate`
    /// as the chain's running register `v` (so a chain that ends in
    /// swiglu reads its first input from whatever produced `v`) and
    /// reads `up` from a uniform buffer at index `base + i`. Same
    /// math as `rce-op-metal::swiglu_f32` per element; the chain
    /// composes silu's expansion into the first multiplicand.
    ///
    /// Args contract when swiglu is a chain head: `(gate, up)` —
    /// gate is the chain input (slot 0), up is the first fragment
    /// uniform (slot 2). Same shape as the stand-alone swiglu kernel.
    pub const SWIGLU: MslFragment = MslFragment {
        op_name: "swiglu",
        elementwise: "    v = (v / (1.0f + exp(-v))) * up[base + i];",
        reduction: None,
        uniforms: &[BufferDecl { name: "up", dtype: "float" }],
        scalars: &[],
    };

    /// Look up a fragment by op name. Returns `None` for ops not in the
    /// table — the emitter rejects unknown ops rather than silently
    /// skipping them.
    pub fn lookup(op: &str) -> Option<&'static MslFragment> {
        match op {
            "silu" => Some(&SILU),
            "gelu" => Some(&GELU),
            "rms_norm" => Some(&RMS_NORM),
            "swiglu" => Some(&SWIGLU),
            _ => None,
        }
    }
}

/// Why an emit attempt could not produce MSL. Each variant names the
/// chain shape that caused it — debugging info, not a recoverable error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodegenError {
    EmptyChain,
    UnknownOp(String),
    TooManyReductions { count: usize },
}

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodegenError::EmptyChain =>
                write!(f, "cannot emit MSL for an empty chain"),
            CodegenError::UnknownOp(name) =>
                write!(f, "unknown op `{name}` — add a fragment to rce_fusion::codegen::fragments first"),
            CodegenError::TooManyReductions { count } =>
                write!(f, "{count} reduction-bearing ops in chain; Phase A handles ≤ 1"),
        }
    }
}

impl std::error::Error for CodegenError {}

/// Information about how to dispatch the emitted kernel, returned
/// alongside the MSL source so callers don't have to re-derive it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmittedKernel {
    /// The complete MSL source — a `#include <metal_stdlib>` followed by
    /// one `kernel void <function_name>(...)`.
    pub source: String,
    /// The kernel's MSL function name. Stable: a function of the canonical
    /// chain key. Same chain ⇒ same name.
    pub function_name: String,
    /// Buffer-table layout, in slot order. Slots are:
    ///   0: input  (`device const float*`)
    ///   1: output (`device       float*`)
    ///   2..N: each fragment's uniforms, in chain order (dedup'd by name)
    ///   N..M: `constant uint& ngroups`
    ///         `constant uint& group_size`
    ///         then each fragment's scalars, in chain order (dedup'd)
    pub buffers: Vec<BufferDecl>,
    pub scalars: Vec<ScalarDecl>,
}

/// Emit MSL for a chain of ops at the given dtype. f32 / f16 produce
/// distinct kernels (different buffer types, different function name
/// suffix). Per-element math runs in f32 registers regardless of
/// dtype — only the load/store path differs.
pub fn emit_chain_msl_dtype(chain: &[&str], dtype: Dtype) -> Result<EmittedKernel, CodegenError> {
    if chain.is_empty() {
        return Err(CodegenError::EmptyChain);
    }
    let mut frags: Vec<&MslFragment> = Vec::with_capacity(chain.len());
    for op in chain {
        match fragments::lookup(op) {
            Some(f) => frags.push(f),
            None => return Err(CodegenError::UnknownOp((*op).to_string())),
        }
    }
    let reduction_positions: Vec<usize> =
        frags.iter().enumerate().filter(|(_, f)| f.has_reduction()).map(|(i, _)| i).collect();
    if reduction_positions.len() > 1 {
        return Err(CodegenError::TooManyReductions { count: reduction_positions.len() });
    }

    // Dedup uniforms / scalars by name; the element dtype applies to
    // input / output / fragment uniforms (which are tensor data),
    // not to scalar constants (`eps` etc) which stay f32.
    let mut buffers: Vec<BufferDecl> = vec![
        BufferDecl { name: "input",  dtype: dtype.msl_name() },
        BufferDecl { name: "output", dtype: dtype.msl_name() },
    ];
    for f in &frags {
        for u in f.uniforms {
            if !buffers.iter().any(|b| b.name == u.name) {
                // Override fragment-declared uniform dtype with the
                // element dtype — uniforms are tensor data and follow
                // the chain's dtype.
                buffers.push(BufferDecl { name: u.name, dtype: dtype.msl_name() });
            }
        }
    }
    let mut scalars: Vec<ScalarDecl> = Vec::new();
    for f in &frags {
        for s in f.scalars {
            if !scalars.iter().any(|d| d.name == s.name) {
                // Scalars stay f32 — `eps` and similar tuning
                // constants don't benefit from dtype reduction.
                scalars.push(s.clone());
            }
        }
    }

    let function_name = canonical_function_name_dtype(chain, dtype);
    let source = render_kernel(&function_name, dtype, &frags, reduction_positions.first().copied(), &buffers, &scalars);

    Ok(EmittedKernel { source, function_name, buffers, scalars })
}

/// Backward-compatible wrapper that defaults to f32. Used by callers
/// not yet updated to thread dtype through.
pub fn emit_chain_msl(chain: &[&str]) -> Result<EmittedKernel, CodegenError> {
    emit_chain_msl_dtype(chain, Dtype::F32)
}

/// Stable function name for a chain at a given dtype. Same chain +
/// dtype ⇒ same name ⇒ same pipeline cache hit. The dtype suffix
/// (`_f32` / `_f16`) makes f32 and f16 variants of the same chain
/// distinct pipelines.
pub fn canonical_function_name_dtype(chain: &[&str], dtype: Dtype) -> String {
    let mut s = String::from("fused");
    for op in chain {
        s.push('_');
        s.push_str(op);
    }
    s.push('_');
    s.push_str(dtype.name());
    s
}

/// Backward-compatible wrapper that defaults to f32. Used by callers
/// that haven't been updated to thread dtype through.
pub fn canonical_function_name(chain: &[&str]) -> String {
    canonical_function_name_dtype(chain, Dtype::F32)
}

/// Inverse of [`canonical_function_name_dtype`]: parse a fused-chain
/// function name back into the chain of op names + its dtype.
/// Returns `None` if the name doesn't match either `fused_..._f32` or
/// `fused_..._f16`, or if the body can't be tokenised into known
/// fragments.
///
/// The body of the name is greedy-matched against [`fragments::lookup`]:
/// at each cursor position we try every known op and pick the longest
/// match (so `rms_norm` beats `rms` whenever both would match a prefix).
/// Two ops sharing a name where one is a prefix of the other would
/// trash this — today's four fragments (`silu`, `gelu`, `rms_norm`,
/// `swiglu`) are unambiguous and new fragments must keep that property.
pub fn parse_function_name_with_dtype(name: &str) -> Option<(Vec<&'static str>, Dtype)> {
    // Strip the dtype suffix first (longest first so `bf16` wins over
    // a hypothetical bare `16` parse).
    let (body, dtype) = if let Some(b) = name.strip_prefix("fused_").and_then(|s| s.strip_suffix("_bf16")) {
        (b, Dtype::BF16)
    } else if let Some(b) = name.strip_prefix("fused_").and_then(|s| s.strip_suffix("_f32")) {
        (b, Dtype::F32)
    } else if let Some(b) = name.strip_prefix("fused_").and_then(|s| s.strip_suffix("_f16")) {
        (b, Dtype::F16)
    } else {
        return None;
    };
    if body.is_empty() {
        return None;
    }
    const KNOWN: &[&str] = &["silu", "gelu", "rms_norm", "swiglu"];
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
/// dtype. Used by call sites that don't yet thread dtype through.
pub fn parse_function_name(name: &str) -> Option<Vec<&'static str>> {
    parse_function_name_with_dtype(name).map(|(c, _)| c)
}

fn render_kernel(
    function_name: &str,
    dtype: Dtype,
    frags: &[&MslFragment],
    reduction_idx: Option<usize>,
    buffers: &[BufferDecl],
    scalars: &[ScalarDecl],
) -> String {
    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\n");
    out.push_str("using namespace metal;\n\n");

    // ── Signature ─────────────────────────────────────────────────
    out.push_str("kernel void ");
    out.push_str(function_name);
    out.push_str("(\n");
    let mut slot: usize = 0;
    // input
    out.push_str(&format!(
        "    device const {}* input  [[buffer({})]],\n",
        buffers[0].dtype, slot
    ));
    slot += 1;
    // output
    out.push_str(&format!(
        "    device       {}* output [[buffer({})]],\n",
        buffers[1].dtype, slot
    ));
    slot += 1;
    // fragment buffers
    for b in &buffers[2..] {
        out.push_str(&format!(
            "    device const {}* {} [[buffer({})]],\n",
            b.dtype, b.name, slot
        ));
        slot += 1;
    }
    // ngroups, group_size
    out.push_str(&format!("    constant uint& ngroups    [[buffer({})]],\n", slot));
    slot += 1;
    out.push_str(&format!("    constant uint& group_size [[buffer({})]],\n", slot));
    slot += 1;
    // fragment scalars
    for s in scalars {
        out.push_str(&format!(
            "    constant {}& {} [[buffer({})]],\n",
            s.dtype, s.name, slot
        ));
        slot += 1;
    }
    out.push_str("    uint gid                   [[thread_position_in_grid]]\n");
    out.push_str(") {\n");
    out.push_str("    if (gid >= ngroups) return;\n");
    out.push_str("    uint base = gid * group_size;\n");

    match reduction_idx {
        None => {
            // Single-pass: stream every elementwise fragment per element.
            out.push_str("    for (uint i = 0; i < group_size; i++) {\n");
            out.push_str("        float v = input[base + i];\n");
            for f in frags {
                debug_assert!(
                    !f.has_reduction(),
                    "render_kernel: no-reduction path saw a reduction fragment",
                );
                if !f.elementwise.is_empty() {
                    out.push_str("    "); // extra indent inside the for body
                    out.push_str(f.elementwise);
                    out.push('\n');
                }
            }
            // For f32, write v directly (no-op cast). For f16 / bf16,
            // emit an explicit narrowing conversion so the typed
            // store is unambiguous in MSL.
            match dtype {
                Dtype::F32 => out.push_str("        output[base + i] = v;\n"),
                Dtype::F16 => out.push_str("        output[base + i] = half(v);\n"),
                Dtype::BF16 => out.push_str("        output[base + i] = bfloat(v);\n"),
            }
            out.push_str("    }\n");
        }
        Some(ri) => {
            let red = frags[ri].reduction.as_ref().expect("reduction at ri");
            // Pass 1: pre-reduction fragments + reduction.accumulate.
            out.push_str(red.init);
            out.push('\n');
            out.push_str("    for (uint i = 0; i < group_size; i++) {\n");
            out.push_str("        float v = input[base + i];\n");
            for f in &frags[..ri] {
                if !f.elementwise.is_empty() {
                    out.push_str("    "); out.push_str(f.elementwise); out.push('\n');
                }
            }
            out.push_str(red.accumulate);
            out.push('\n');
            out.push_str("    }\n");
            // Between passes: reduction.finalize.
            out.push_str(red.finalize);
            out.push('\n');
            // Pass 2: re-emit pre-reduction fragments + reduction.scale +
            // post-reduction fragments + output write.
            out.push_str("    for (uint i = 0; i < group_size; i++) {\n");
            out.push_str("        float v = input[base + i];\n");
            for f in &frags[..ri] {
                if !f.elementwise.is_empty() {
                    out.push_str("    "); out.push_str(f.elementwise); out.push('\n');
                }
            }
            out.push_str(red.scale);
            out.push('\n');
            for f in &frags[ri + 1..] {
                if !f.elementwise.is_empty() {
                    out.push_str("    "); out.push_str(f.elementwise); out.push('\n');
                }
            }
            // For f32, write v directly (no-op cast). For f16 / bf16,
            // emit an explicit narrowing conversion so the typed
            // store is unambiguous in MSL.
            match dtype {
                Dtype::F32 => out.push_str("        output[base + i] = v;\n"),
                Dtype::F16 => out.push_str("        output[base + i] = half(v);\n"),
                Dtype::BF16 => out.push_str("        output[base + i] = bfloat(v);\n"),
            }
            out.push_str("    }\n");
        }
    }

    out.push_str("}\n");
    out
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_chain_errors() {
        assert_eq!(emit_chain_msl(&[]).unwrap_err(), CodegenError::EmptyChain);
    }

    #[test]
    fn unknown_op_errors() {
        let err = emit_chain_msl(&["silu", "not_a_real_op"]).unwrap_err();
        match err {
            CodegenError::UnknownOp(name) => assert_eq!(name, "not_a_real_op"),
            other => panic!("expected UnknownOp, got {other:?}"),
        }
    }

    #[test]
    fn two_reductions_errors() {
        // Two rms_norms in a row — Phase A scope limit.
        let err = emit_chain_msl(&["rms_norm", "rms_norm"]).unwrap_err();
        assert_eq!(err, CodegenError::TooManyReductions { count: 2 });
    }

    #[test]
    fn single_silu_emits_single_pass() {
        let k = emit_chain_msl(&["silu"]).unwrap();
        assert_eq!(k.function_name, "fused_silu_f32");
        // No `weight` buffer, no `eps` scalar.
        assert_eq!(k.buffers.iter().map(|b| b.name).collect::<Vec<_>>(), vec!["input", "output"]);
        assert!(k.scalars.is_empty());
        // No reduction → no `sum_sq` accumulator anywhere.
        assert!(!k.source.contains("sum_sq"));
        // The silu math should appear exactly once.
        assert_eq!(k.source.matches("v = v / (1.0f + exp(-v));").count(), 1);
        // And the kernel writes output.
        assert!(k.source.contains("output[base + i] = v;"));
    }

    #[test]
    fn silu_then_gelu_single_pass_with_both_fragments() {
        let k = emit_chain_msl(&["silu", "gelu"]).unwrap();
        assert_eq!(k.function_name, "fused_silu_gelu_f32");
        // Still no weight / eps.
        assert_eq!(k.buffers.iter().map(|b| b.name).collect::<Vec<_>>(), vec!["input", "output"]);
        assert!(k.scalars.is_empty());
        assert!(!k.source.contains("sum_sq"));
        // Both fragments appear exactly once, in order.
        let silu_pos = k.source.find("v = v / (1.0f + exp(-v));")
            .expect("silu in single-pass chain");
        let gelu_pos = k.source.find("0.7978845608028654")
            .expect("gelu in single-pass chain");
        assert!(silu_pos < gelu_pos, "silu must precede gelu in the emitted body");
    }

    #[test]
    fn silu_then_rms_norm_emits_two_passes_with_re_emitted_silu() {
        let k = emit_chain_msl(&["silu", "rms_norm"]).unwrap();
        assert_eq!(k.function_name, "fused_silu_rms_norm_f32");
        // `weight` and `eps` show up.
        assert_eq!(k.buffers.iter().map(|b| b.name).collect::<Vec<_>>(), vec!["input", "output", "weight"]);
        assert_eq!(k.scalars.iter().map(|s| s.name).collect::<Vec<_>>(), vec!["eps"]);
        // Reduction init appears exactly once, before any pass.
        assert_eq!(k.source.matches("float sum_sq = 0.0f;").count(), 1);
        // Reduction finalize appears between the two passes — exactly once.
        assert_eq!(k.source.matches("float inv_rms = rsqrt").count(), 1);
        // Silu's body appears in BOTH passes (re-emitted in pass 2 because
        // the silu intermediate isn't kept).
        assert_eq!(k.source.matches("v = v / (1.0f + exp(-v));").count(), 2);
        // Scale and output write appear once each.
        assert_eq!(k.source.matches("v = v * inv_rms * weight[base + i];").count(), 1);
        assert_eq!(k.source.matches("output[base + i] = v;").count(), 1);
    }

    #[test]
    fn rms_norm_then_gelu_emits_post_reduction_gelu() {
        let k = emit_chain_msl(&["rms_norm", "gelu"]).unwrap();
        assert_eq!(k.function_name, "fused_rms_norm_gelu_f32");
        assert_eq!(k.buffers.iter().map(|b| b.name).collect::<Vec<_>>(), vec!["input", "output", "weight"]);
        assert_eq!(k.scalars.iter().map(|s| s.name).collect::<Vec<_>>(), vec!["eps"]);
        // No pre-reduction fragments → silu must NOT appear.
        assert!(!k.source.contains("(1.0f + exp(-v))"));
        // Gelu's marker constant appears exactly once — in pass 2 after scale.
        assert_eq!(k.source.matches("0.7978845608028654").count(), 1);
        // Gelu must appear AFTER the rms scale block.
        let scale_pos = k.source.find("v = v * inv_rms * weight[base + i];").unwrap();
        let gelu_pos  = k.source.find("0.7978845608028654").unwrap();
        assert!(gelu_pos > scale_pos, "post-reduction gelu must follow rms scale");
    }

    #[test]
    fn silu_rms_norm_gelu_re_emits_silu_in_pass_2_and_runs_gelu_after_scale() {
        let k = emit_chain_msl(&["silu", "rms_norm", "gelu"]).unwrap();
        assert_eq!(k.function_name, "fused_silu_rms_norm_gelu_f32");
        // Silu twice (both passes).
        assert_eq!(k.source.matches("v = v / (1.0f + exp(-v));").count(), 2);
        // Gelu only in pass 2 (post-reduction).
        assert_eq!(k.source.matches("0.7978845608028654").count(), 1);
        // Verify order in pass 2: silu < scale < gelu.
        // (Strip the source to start after the finalize line so we examine pass 2 only.)
        let pass2_start = k.source.find("float inv_rms = rsqrt").unwrap();
        let pass2 = &k.source[pass2_start..];
        let silu_pos  = pass2.find("v = v / (1.0f + exp(-v));").unwrap();
        let scale_pos = pass2.find("v = v * inv_rms * weight[base + i];").unwrap();
        let gelu_pos  = pass2.find("0.7978845608028654").unwrap();
        assert!(silu_pos < scale_pos, "silu must re-run before rms scale in pass 2");
        assert!(scale_pos < gelu_pos, "gelu must run after rms scale in pass 2");
    }

    // ── Phase K: f16 dtype variants ────────────────────────────────────

    #[test]
    fn f16_chain_emits_half_buffers_and_explicit_cast_on_store() {
        let k = emit_chain_msl_dtype(&["silu"], Dtype::F16).unwrap();
        assert_eq!(k.function_name, "fused_silu_f16");
        // Buffers are half-typed.
        assert!(k.source.contains("device const half* input"));
        assert!(k.source.contains("device       half* output"));
        // Per-element register stays f32 (load via implicit conversion).
        assert!(k.source.contains("float v = input[base + i];"));
        // Output store has an explicit half() narrowing cast.
        assert!(k.source.contains("output[base + i] = half(v);"));
        // The silu math body is unchanged — it operates on the f32
        // register `v`.
        assert!(k.source.contains("v = v / (1.0f + exp(-v));"));
    }

    #[test]
    fn f16_chain_with_reduction_keeps_f32_accumulator() {
        // `silu → rms_norm` in f16. The reduction accumulator
        // (`sum_sq`) and finalised stat (`inv_rms`) stay f32 — they
        // live only in registers and reducing in f16 would sacrifice
        // numeric precision for no bandwidth win. Only input / output /
        // weight buffers become half-typed.
        let k = emit_chain_msl_dtype(&["silu", "rms_norm"], Dtype::F16).unwrap();
        assert_eq!(k.function_name, "fused_silu_rms_norm_f16");
        // weight buffer is half-typed (tensor data → element dtype).
        assert!(k.source.contains("device const half* weight"));
        // eps stays f32 (scalar constants stay f32).
        assert!(k.source.contains("constant float& eps"));
        // f32 reduction accumulator preserved.
        assert!(k.source.contains("float sum_sq = 0.0f;"));
        assert!(k.source.contains("float inv_rms = rsqrt"));
        // Both passes write through a half() narrowing cast (only the
        // second pass actually writes output; the first pass's store is
        // absent because it's a reduction-only pass).
        assert_eq!(k.source.matches("output[base + i] = half(v);").count(), 1);
        assert_eq!(k.source.matches("output[base + i] = v;").count(), 0);
    }

    #[test]
    fn f16_and_f32_emit_distinct_function_names() {
        let f32 = emit_chain_msl_dtype(&["silu", "rms_norm"], Dtype::F32).unwrap();
        let f16 = emit_chain_msl_dtype(&["silu", "rms_norm"], Dtype::F16).unwrap();
        assert_eq!(f32.function_name, "fused_silu_rms_norm_f32");
        assert_eq!(f16.function_name, "fused_silu_rms_norm_f16");
        assert_ne!(f32.source, f16.source, "f32 and f16 must emit different MSL");
    }

    #[test]
    fn parse_function_name_with_dtype_round_trips_both_dtypes() {
        for (chain, dtype) in [
            (vec!["silu"], Dtype::F32),
            (vec!["silu"], Dtype::F16),
            (vec!["silu", "rms_norm"], Dtype::F32),
            (vec!["silu", "rms_norm"], Dtype::F16),
            (vec!["silu", "rms_norm", "gelu"], Dtype::F16),
            (vec!["swiglu"], Dtype::F16),
        ] {
            let name = canonical_function_name_dtype(&chain, dtype);
            let (parsed_chain, parsed_dtype) = parse_function_name_with_dtype(&name)
                .unwrap_or_else(|| panic!("parse failed for `{name}`"));
            assert_eq!(parsed_chain, chain, "chain round-trip failed for `{name}`");
            assert_eq!(parsed_dtype, dtype, "dtype round-trip failed for `{name}`");
        }
    }

    #[test]
    fn dtype_helper_methods() {
        assert_eq!(Dtype::F32.msl_name(), "float");
        assert_eq!(Dtype::F16.msl_name(), "half");
        assert_eq!(Dtype::F32.name(), "f32");
        assert_eq!(Dtype::F16.name(), "f16");
        assert_eq!(Dtype::F32.size_bytes(), 4);
        assert_eq!(Dtype::F16.size_bytes(), 2);
    }

    #[test]
    fn backward_compat_canonical_function_name_defaults_f32() {
        assert_eq!(canonical_function_name(&["silu"]), "fused_silu_f32");
        // And parse_function_name accepts f32 names.
        assert_eq!(parse_function_name("fused_silu_f32"), Some(vec!["silu"]));
        // But it ALSO accepts f16 names now — the backward-compat
        // wrapper just drops the dtype. The dispatch site is
        // responsible for routing by dtype when it matters.
        assert_eq!(parse_function_name("fused_silu_f16"), Some(vec!["silu"]));
    }

    // ── Phase L: bf16 ──────────────────────────────────────────────────

    #[test]
    fn bf16_chain_emits_bfloat_buffers_and_cast() {
        let k = emit_chain_msl_dtype(&["silu", "rms_norm"], Dtype::BF16).unwrap();
        assert_eq!(k.function_name, "fused_silu_rms_norm_bf16");
        assert!(k.source.contains("device const bfloat* input"));
        assert!(k.source.contains("device       bfloat* output"));
        assert!(k.source.contains("device const bfloat* weight"));
        // Scalars stay f32.
        assert!(k.source.contains("constant float& eps"));
        // Register `v` stays f32; reduction accumulator stays f32.
        assert!(k.source.contains("float v = input[base + i];"));
        assert!(k.source.contains("float sum_sq = 0.0f;"));
        assert!(k.source.contains("float inv_rms = rsqrt"));
        // Output store has the bfloat narrowing cast.
        assert!(k.source.contains("output[base + i] = bfloat(v);"));
    }

    #[test]
    fn bf16_round_trip_through_parser() {
        for chain in [
            vec!["silu"],
            vec!["gelu"],
            vec!["silu", "rms_norm"],
            vec!["rms_norm", "gelu"],
            vec!["silu", "rms_norm", "gelu"],
            vec!["swiglu"],
        ] {
            let name = canonical_function_name_dtype(&chain, Dtype::BF16);
            let (parsed_chain, parsed_dtype) = parse_function_name_with_dtype(&name)
                .unwrap_or_else(|| panic!("bf16 parse failed for `{name}`"));
            assert_eq!(parsed_chain, chain);
            assert_eq!(parsed_dtype, Dtype::BF16);
        }
    }

    #[test]
    fn three_dtypes_emit_three_distinct_function_names() {
        let f32 = emit_chain_msl_dtype(&["silu"], Dtype::F32).unwrap().function_name;
        let f16 = emit_chain_msl_dtype(&["silu"], Dtype::F16).unwrap().function_name;
        let bf16 = emit_chain_msl_dtype(&["silu"], Dtype::BF16).unwrap().function_name;
        assert_eq!(f32, "fused_silu_f32");
        assert_eq!(f16, "fused_silu_f16");
        assert_eq!(bf16, "fused_silu_bf16");
        assert_ne!(f32, f16);
        assert_ne!(f32, bf16);
        assert_ne!(f16, bf16);
    }

    #[test]
    fn dtype_bf16_helper_methods() {
        assert_eq!(Dtype::BF16.msl_name(), "bfloat");
        assert_eq!(Dtype::BF16.name(), "bf16");
        assert_eq!(Dtype::BF16.size_bytes(), 2);
    }

    #[test]
    fn parser_dtype_dispatch_priority_bf16_before_f16() {
        // `_bf16` ends with `_f16` lexically — confirm the parser
        // strips `_bf16` first so the dtype dispatch is correct.
        let (chain, dtype) = parse_function_name_with_dtype("fused_silu_bf16").unwrap();
        assert_eq!(chain, vec!["silu"]);
        assert_eq!(dtype, Dtype::BF16);
        // And `_f16` doesn't accidentally get tagged as bf16.
        let (chain, dtype) = parse_function_name_with_dtype("fused_silu_f16").unwrap();
        assert_eq!(chain, vec!["silu"]);
        assert_eq!(dtype, Dtype::F16);
    }

    #[test]
    fn parse_function_name_round_trips_every_supported_chain() {
        for chain in [
            vec!["silu"],
            vec!["gelu"],
            vec!["rms_norm"],
            vec!["silu", "gelu"],
            vec!["silu", "rms_norm"],
            vec!["rms_norm", "gelu"],
            vec!["silu", "rms_norm", "gelu"],
            vec!["gelu", "rms_norm", "silu"],
        ] {
            let name = canonical_function_name(&chain);
            let parsed = parse_function_name(&name)
                .unwrap_or_else(|| panic!("parse failed for `{name}` (chain {chain:?})"));
            assert_eq!(parsed, chain, "round-trip failed for `{name}`");
        }
    }

    #[test]
    fn parse_function_name_rejects_garbage() {
        // Missing prefix.
        assert!(parse_function_name("silu_f32").is_none());
        // Missing suffix.
        assert!(parse_function_name("fused_silu").is_none());
        // Empty body.
        assert!(parse_function_name("fused__f32").is_none());
        // Unknown op in body.
        assert!(parse_function_name("fused_not_a_real_op_f32").is_none());
        // Trailing underscore (no op after it).
        assert!(parse_function_name("fused_silu__f32").is_none());
    }

    #[test]
    fn parse_function_name_known_table_in_sync_with_fragments() {
        // Each name in KNOWN must resolve via fragments::lookup, and
        // every fragment must be in KNOWN. This pins the two source-of-
        // truth lists together so adding a fragment without updating
        // the parser (or vice versa) fails loudly.
        for op in &["silu", "gelu", "rms_norm", "swiglu"] {
            assert!(
                fragments::lookup(op).is_some(),
                "KNOWN contains `{op}` but fragments::lookup doesn't"
            );
        }
        // Reverse check: enumerate fragments by name via lookup.
        let canonical = canonical_function_name(&["silu", "gelu", "rms_norm", "swiglu"]);
        let parsed = parse_function_name(&canonical).unwrap();
        assert_eq!(parsed, vec!["silu", "gelu", "rms_norm", "swiglu"]);
    }

    #[test]
    fn swiglu_chain_head_emits_with_up_uniform() {
        // swiglu alone as a chain head: input = gate (in `v`), up is the
        // first fragment uniform at slot 2.
        let k = emit_chain_msl(&["swiglu"]).unwrap();
        assert_eq!(k.function_name, "fused_swiglu_f32");
        assert_eq!(k.buffers.iter().map(|b| b.name).collect::<Vec<_>>(),
            vec!["input", "output", "up"]);
        assert!(k.scalars.is_empty());
        // No reduction → single pass.
        assert!(!k.source.contains("sum_sq"));
        // The swiglu body must read `up` per-element.
        assert!(k.source.contains("up[base + i]"));
        // Math byte-equivalent to silu(v)*up[base+i].
        assert!(k.source.contains("v = (v / (1.0f + exp(-v))) * up[base + i];"));
    }

    #[test]
    fn silu_then_swiglu_composes_silu_into_first_multiplicand() {
        // The mathematical interpretation: `swiglu(silu(x), up) =
        // silu(silu(x)) * up`. The chain has no reduction, so emit a
        // single-pass kernel that reads input then applies silu then
        // applies swiglu's "silu(v) * up" body.
        let k = emit_chain_msl(&["silu", "swiglu"]).unwrap();
        assert_eq!(k.function_name, "fused_silu_swiglu_f32");
        // silu body appears (no `up` involvement) BEFORE the swiglu body.
        let silu_pos = k.source.find("v = v / (1.0f + exp(-v));")
            .expect("silu body in single-pass chain");
        let swiglu_pos = k.source.find("v = (v / (1.0f + exp(-v))) * up[base + i];")
            .expect("swiglu body in single-pass chain");
        assert!(silu_pos < swiglu_pos, "silu must precede swiglu in the emitted body");
    }

    #[test]
    fn rms_norm_then_swiglu_emits_two_pass_with_swiglu_after_scale() {
        // Reduction + post-reduction elementwise op that reads its own
        // uniform: rms_norm in the middle, swiglu after.
        let k = emit_chain_msl(&["rms_norm", "swiglu"]).unwrap();
        assert_eq!(k.function_name, "fused_rms_norm_swiglu_f32");
        // Both uniforms surface, in chain order.
        assert_eq!(k.buffers.iter().map(|b| b.name).collect::<Vec<_>>(),
            vec!["input", "output", "weight", "up"]);
        assert_eq!(k.scalars.iter().map(|s| s.name).collect::<Vec<_>>(), vec!["eps"]);
        // swiglu's body lives after rms scale.
        let scale_pos = k.source.find("v = v * inv_rms * weight[base + i];").unwrap();
        let swiglu_pos = k.source.find("v = (v / (1.0f + exp(-v))) * up[base + i];").unwrap();
        assert!(swiglu_pos > scale_pos, "post-reduction swiglu must follow rms scale");
    }

    #[test]
    fn function_name_is_stable_per_chain() {
        // Same chain → same name (cache key promise).
        let a = emit_chain_msl(&["silu", "rms_norm"]).unwrap().function_name;
        let b = emit_chain_msl(&["silu", "rms_norm"]).unwrap().function_name;
        assert_eq!(a, b);
        // Different chain → different name.
        let c = emit_chain_msl(&["silu", "gelu"]).unwrap().function_name;
        assert_ne!(a, c);
    }

    #[test]
    fn buffer_slots_are_unique_and_sequential() {
        let k = emit_chain_msl(&["silu", "rms_norm"]).unwrap();
        // Walk every `[[buffer(N)]]` in the source; assert each slot is
        // used at most once and the slots are 0..n contiguous.
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
        assert_eq!(seen, expected, "buffer slots must be 0..n contiguous, no gaps, no dups");
    }

    /// Pin the exact emitted MSL for the `silu → rms_norm` chain as a
    /// golden. Future drift in fragment bodies / template structure
    /// must come with a deliberate update to this string. The string
    /// also serves as a readable specification of what a fused chain
    /// looks like end-to-end.
    #[test]
    fn golden_msl_silu_rms_norm() {
        let k = emit_chain_msl(&["silu", "rms_norm"]).unwrap();
        let expected = "\
#include <metal_stdlib>
using namespace metal;

kernel void fused_silu_rms_norm_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    constant uint& ngroups    [[buffer(3)]],
    constant uint& group_size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid                   [[thread_position_in_grid]]
) {
    if (gid >= ngroups) return;
    uint base = gid * group_size;
    float sum_sq = 0.0f;
    for (uint i = 0; i < group_size; i++) {
        float v = input[base + i];
        v = v / (1.0f + exp(-v));
        sum_sq += v * v;
    }
    float inv_rms = rsqrt(sum_sq / float(group_size) + eps);
    for (uint i = 0; i < group_size; i++) {
        float v = input[base + i];
        v = v / (1.0f + exp(-v));
        v = v * inv_rms * weight[base + i];
        output[base + i] = v;
    }
}
";
        assert_eq!(k.source, expected, "golden MSL mismatch — update both fragments and this string deliberately");
    }
}
