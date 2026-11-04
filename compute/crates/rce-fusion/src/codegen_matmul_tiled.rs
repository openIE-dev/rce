//! `rce-fusion::codegen_matmul_tiled` — Phase I of the codegen
//! rollout. Same chain shape as Phase F (matmul + optional elementwise
//! epilogue), but emits a **tiled** kernel that uses threadgroup
//! memory to share `[A_tile, B_tile]` across the `TILE_M × TILE_N`
//! threads of a threadgroup. Each K-block load happens once per
//! threadgroup instead of once per thread — the bandwidth amplifier
//! tiled matmul provides for sufficiently-large shapes.
//!
//! ## What it is vs Phase F
//!
//! Phase F's `codegen_matmul_epilogue` emits a naive matmul: one
//! thread per output element, the K-loop dotted in registers without
//! sharing. That's fine for small M/N/K and short-lived dispatches,
//! but for FFN-sized matmuls (`d_model × d_inner ≈ 2560 × 8192`) the
//! redundant K-stream reads dominate bandwidth.
//!
//! Phase I emits the standard textbook tile-on-threadgroup-memory
//! kernel:
//!   1. Threadgroup owns one `[TILE_M × TILE_N]` output tile.
//!   2. K is walked in `TILE_K`-sized blocks.
//!   3. For each block, all `TILE_M × TILE_N` threads in the
//!      threadgroup cooperatively load one element each of
//!      `A_tile [TILE_M × TILE_K]` and `B_tile [TILE_K × TILE_N]`
//!      into threadgroup memory.
//!   4. Barrier.
//!   5. Each thread computes its partial dot using the shared tiles.
//!   6. Barrier before the next block load.
//!
//! Each `A` byte is read into threadgroup memory once and reused
//! `TILE_N` times across the tile's columns. Same for `B`. That's a
//! `TILE_N × TILE_M` reduction in DRAM reads of the inputs.
//!
//! ## Tile dimensions
//!
//! Baked at compile time as part of the kernel source — MSL doesn't
//! allow dynamic threadgroup array sizes. Phase I uses
//! `TILE_M = TILE_N = TILE_K = 16` (256 threads per threadgroup,
//! within the max-1024 budget on every Apple GPU). Different tile
//! sizes mean different kernels; future phases can emit multiple
//! variants and route by shape.
//!
//! ## Bounds handling
//!
//! Shapes don't have to be tile-multiples. The kernel guards every
//! global memory read with the `< M, < K, < N` constraints (loads
//! beyond-bounds tile slots get `0.0f`, which contributes nothing to
//! the dot). The output write is also guarded. Edges are slow but
//! correct.
//!
//! ## Args contract
//!
//! Same as Phase F: input `[M, K]`, weight `[K, N]`, output `[M, N]`,
//! then fragment uniforms `[M, N]`, then `M K N` constants, then
//! fragment scalars. The dispatch shape differs from Phase F (2D
//! grid sized to tile multiples, threadgroup = `[TILE_N, TILE_M]`).
//!
//! ## Scope
//!
//! - Elementwise epilogues only (silu, gelu, swiglu). Reductions
//!   would need a separate template — out of scope.
//! - f32 only.
//! - Fixed 16x16x16 tile. Tuning per shape is a follow-up.

use crate::codegen::{fragments, BufferDecl, CodegenError, Dtype, MslFragment, ScalarDecl};

/// Tile width = number of output columns each threadgroup owns.
pub const TILE_N: u32 = 16;
/// Tile height = number of output rows each threadgroup owns.
pub const TILE_M: u32 = 16;
/// K-block size = chunk of the inner dim each shared-memory load
/// covers.
pub const TILE_K: u32 = 16;

/// What `emit_matmul_tiled_msl` returns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmittedMatmulTiledKernel {
    pub source: String,
    pub function_name: String,
    pub buffers: Vec<BufferDecl>,
    pub scalars: Vec<ScalarDecl>,
}

/// Stable function name for tiled matmul + epilogue. Same epilogue
/// (and the implicit tiled head) ⇒ same name ⇒ same pipeline cache
/// hit.
pub fn canonical_function_name_dtype(epilogue: &[&str], dtype: Dtype) -> String {
    let mut s = String::from("fused_matmul_tiled");
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

/// Inverse of `canonical_function_name_dtype`. Empty epilogue
/// (`fused_matmul_tiled_<dtype>`) is a plain tiled matmul.
pub fn parse_function_name_with_dtype(name: &str) -> Option<(Vec<&'static str>, Dtype)> {
    let (body, dtype) = if let Some(b) = name.strip_prefix("fused_matmul_tiled").and_then(|s| s.strip_suffix("_bf16")) {
        (b, Dtype::BF16)
    } else if let Some(b) = name.strip_prefix("fused_matmul_tiled").and_then(|s| s.strip_suffix("_f32")) {
        (b, Dtype::F32)
    } else if let Some(b) = name.strip_prefix("fused_matmul_tiled").and_then(|s| s.strip_suffix("_f16")) {
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

/// Emit MSL for `output = epilogue(input @ weight)` using the tiled
/// kernel template at the given dtype.
pub fn emit_matmul_tiled_msl_dtype(
    epilogue: &[&str],
    dtype: Dtype,
) -> Result<EmittedMatmulTiledKernel, CodegenError> {
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
    Ok(EmittedMatmulTiledKernel { source, function_name, buffers, scalars })
}

/// Backward-compatible f32 wrapper.
pub fn emit_matmul_tiled_msl(
    epilogue: &[&str],
) -> Result<EmittedMatmulTiledKernel, CodegenError> {
    emit_matmul_tiled_msl_dtype(epilogue, Dtype::F32)
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
    // Bake the tile dims into the kernel — MSL needs them constant.
    out.push_str(&format!("constant uint TILE_M = {TILE_M};\n"));
    out.push_str(&format!("constant uint TILE_N = {TILE_N};\n"));
    out.push_str(&format!("constant uint TILE_K = {TILE_K};\n\n"));

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
    out.push_str("    uint2 tid  [[thread_position_in_threadgroup]],\n");
    out.push_str("    uint2 tgid [[threadgroup_position_in_grid]]\n");
    out.push_str(") {\n");

    // Threadgroup-shared tiles.
    out.push_str("    threadgroup float A_tile[TILE_M][TILE_K];\n");
    out.push_str("    threadgroup float B_tile[TILE_K][TILE_N];\n");
    // Per-thread global indices into the [M, N] output.
    out.push_str("    uint i = tgid.y * TILE_M + tid.y;  // global row\n");
    out.push_str("    uint j = tgid.x * TILE_N + tid.x;  // global col\n");
    out.push_str("    float v = 0.0f;\n");

    // K-block sweep. Each iteration loads one [TILE_M, TILE_K] +
    // [TILE_K, TILE_N] pair into threadgroup memory, then each thread
    // does TILE_K multiply-adds against its row/column of the tiles.
    out.push_str("    for (uint kk = 0; kk < K; kk += TILE_K) {\n");
    // Load A tile element at (tid.y, tid.x) ← input[i, kk + tid.x].
    // The tile is float; for narrow dtypes the load casts up to float
    // on read so the dot product stays in f32 precision.
    out.push_str("        uint a_global_col = kk + tid.x;\n");
    out.push_str("        A_tile[tid.y][tid.x] = (i < M && a_global_col < K)\n");
    out.push_str("            ? float(input[i * K + a_global_col])\n");
    out.push_str("            : 0.0f;\n");
    // Load B tile element at (tid.y, tid.x) ← weight[kk + tid.y, j].
    out.push_str("        uint b_global_row = kk + tid.y;\n");
    out.push_str("        B_tile[tid.y][tid.x] = (b_global_row < K && j < N)\n");
    out.push_str("            ? float(weight[b_global_row * N + j])\n");
    out.push_str("            : 0.0f;\n");
    out.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    // Dot product against the shared tiles.
    out.push_str("        for (uint k = 0; k < TILE_K; k++) {\n");
    out.push_str("            v += A_tile[tid.y][k] * B_tile[k][tid.x];\n");
    out.push_str("        }\n");
    out.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    out.push_str("    }\n");

    // Out-of-bounds threads exit before applying the epilogue or
    // writing the store. (Their `v` is meaningless because the tile
    // loads it got were guarded; the early-return is a no-op for
    // correctness but avoids spurious epilogue work + writes.)
    out.push_str("    if (i >= M || j >= N) return;\n");

    // Epilogue: per-element transformations. The fragment bodies use
    // the per-element module's `base + i` indexing; rewrite to
    // `i * N + j` for the 2D output layout.
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
    fn tile_dims_are_consistent_with_threadgroup_budget() {
        // 16 * 16 = 256, well within every Apple GPU's
        // max-threads-per-threadgroup (1024 on M-series).
        assert!(TILE_M * TILE_N <= 1024,
            "tile {TILE_M}x{TILE_N} = {} threads > 1024", TILE_M * TILE_N);
        // TILE_K must equal TILE_M and TILE_N for the cooperative load
        // pattern used in `render_kernel` (each of the TILE_M*TILE_N
        // threads loads exactly one A and one B element per K-block).
        assert_eq!(TILE_M, TILE_K);
        assert_eq!(TILE_N, TILE_K);
    }

    #[test]
    fn empty_epilogue_emits_plain_tiled_matmul() {
        let k = emit_matmul_tiled_msl(&[]).unwrap();
        assert_eq!(k.function_name, "fused_matmul_tiled_f32");
        assert_eq!(k.buffers.iter().map(|b| b.name).collect::<Vec<_>>(),
            vec!["input", "weight", "output"]);
        assert!(k.scalars.is_empty());
        // Threadgroup tiles declared.
        assert!(k.source.contains("threadgroup float A_tile[TILE_M][TILE_K];"));
        assert!(k.source.contains("threadgroup float B_tile[TILE_K][TILE_N];"));
        // K-block loop with cooperative load + barrier + inner dot.
        assert!(k.source.contains("for (uint kk = 0; kk < K; kk += TILE_K)"));
        assert!(k.source.contains("threadgroup_barrier(mem_flags::mem_threadgroup);"));
        assert!(k.source.contains("v += A_tile[tid.y][k] * B_tile[k][tid.x];"));
        // OOB guard before write.
        assert!(k.source.contains("if (i >= M || j >= N) return;"));
        assert!(k.source.contains("output[i * N + j] = v;"));
        // No epilogue ⇒ no silu / gelu fragments.
        assert!(!k.source.contains("v = v / (1.0f + exp(-v));"));
    }

    #[test]
    fn tiled_matmul_silu_inlines_silu_post_dot() {
        let k = emit_matmul_tiled_msl(&["silu"]).unwrap();
        assert_eq!(k.function_name, "fused_matmul_tiled_silu_f32");
        let oob_pos   = k.source.find("if (i >= M || j >= N) return;").unwrap();
        let silu_pos  = k.source.find("v = v / (1.0f + exp(-v));").unwrap();
        let store_pos = k.source.find("output[i * N + j] = v;").unwrap();
        assert!(oob_pos < silu_pos, "silu must run only after the OOB guard");
        assert!(silu_pos < store_pos, "silu must precede the output store");
    }

    #[test]
    fn tiled_matmul_swiglu_rewrites_up_index() {
        let k = emit_matmul_tiled_msl(&["swiglu"]).unwrap();
        assert!(k.source.contains("v = (v / (1.0f + exp(-v))) * up[i * N + j];"));
        assert!(!k.source.contains("up[base + i]"));
    }

    #[test]
    fn rms_norm_epilogue_rejected() {
        let err = emit_matmul_tiled_msl(&["rms_norm"]).unwrap_err();
        assert_eq!(err, CodegenError::TooManyReductions { count: 1 });
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
    fn parse_function_name_disambiguates_from_other_matmul_families() {
        // Tiled names: only this module's parser accepts.
        assert!(parse_function_name("fused_matmul_tiled_silu_f32").is_some());
        // Phase F (naive single matmul) parser rejects tiled names.
        assert!(crate::codegen_matmul_epilogue::parse_function_name("fused_matmul_tiled_silu_f32").is_none());
        // Phase H (double matmul) parser rejects tiled names.
        assert!(crate::codegen_matmul_matmul::parse_function_name("fused_matmul_tiled_silu_f32").is_none());
        // And this module's parser rejects the other families.
        assert!(parse_function_name("fused_matmul_silu_f32").is_none());
        assert!(parse_function_name("fused_matmul2_silu_f32").is_none());
    }

    #[test]
    fn buffer_slots_are_unique_and_sequential() {
        let k = emit_matmul_tiled_msl(&["swiglu"]).unwrap();
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
        assert_eq!(seen, expected);
    }
}
