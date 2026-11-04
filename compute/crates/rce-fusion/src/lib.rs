//! `rce-fusion` — Free-fraction graph rewriter + MSL codegen.
//!
//! Walks a `rce_graph::graph::FunctionGraph`, finds adjacent
//! `Apply { op: Custom(name) }` chains whose ops are in the **Free
//! fraction** per [`rce_placement::fusion_eligible`] (CALM I-confluence:
//! per-element functions whose composition is itself per-element), and
//! either rewrites them via a pattern registry or emits a fresh fused
//! MSL kernel for the chain. The CALM classifier proves the rewrite
//! preserves semantics; this crate mechanises the proof and pairs it
//! with a kernel cache so first-call MSL compile cost amortises over
//! every later dispatch.
//!
//! ## Two rewriting modes
//!
//! 1. **Pattern registry** ([`FusionRegistry`]). Hand-written fused
//!    kernel names. Default registry ships one entry: `silu → rms_norm
//!    ⇒ silu_rms_norm` (the kernel `rce-op-metal` shipped in
//!    1974dac87). Callers can register additional patterns. Conservative
//!    match: a chain fuses only when the parent's **single** outgoing
//!    wire feeds the child's **first input port** (the activation slot
//!    — not a weight or scalar) — keeps the rewrite local.
//! 2. **Arbitrary-chain codegen** ([`fuse_with_codegen`]). Walks the
//!    same Free fraction and emits MSL on demand for any chain shape
//!    the codegen modules below cover. The downstream backend
//!    (`rce-op-metal::MetalLeaf`) recognises the canonical names and
//!    compiles + caches the kernel on first dispatch.
//!
//! ## Codegen module map (Phase letters track the rollout commits)
//!
//! | Module | Phase | Chain shape | Tiling |
//! |---|---|---|---|
//! | [`codegen`] | A | Per-element Free chain: `f(g(h(x)))` | Naive (one thread per element) |
//! | [`codegen_matmul_epilogue`] | F | `epilogue(A @ B)` | Naive matmul + register epilogue |
//! | [`codegen_matmul_tiled`] | I | `epilogue(A @ B)` | Threadgroup-tiled (`A_tile` + `B_tile`) |
//! | [`codegen_matmul_matmul`] | H | `epilogue(A @ B @ C)` | Naive two-stage matmul |
//! | [`codegen_ffn_swiglu`] | Q | `silu(input @ gate_W) * (input @ up_W)` | Naive double-matmul |
//! | [`codegen_ffn_swiglu_tiled`] | R | Same shape as Phase Q | Tiled — input tile shared across BOTH matmuls (reuse factor `2 × TILE_N`) |
//!
//! Phase O + U are the dtype rollouts: every module above ships
//! `_dtype` variants of `canonical_function_name` /
//! `parse_function_name` / `emit_*_msl` that accept f32, f16, or bf16.
//! The naked (no-`_dtype`-suffix) functions are f32-only thin wrappers
//! preserved for compatibility. Phase O covered per-element +
//! matmul-epilogue + matmul-tiled + double-matmul (K/L/M/N); Phase U
//! adds the same surface to the FFN-swiglu naive (Q) + tiled (R)
//! modules. Output accumulators stay f32 internally regardless of
//! element dtype; narrowing happens only on the output store.
//!
//! Note: the dispatcher side (`rce-op-metal::dispatch_ffn_swiglu*`)
//! is f32-only as of Phase U. The f16 / bf16 codegen variants are
//! callable via `codegen_ffn_swiglu::emit_ffn_swiglu_msl_dtype` /
//! `codegen_ffn_swiglu_tiled::emit_ffn_swiglu_tiled_msl_dtype` but
//! the inline dispatchers still read f32 buffers; wiring narrow-dtype
//! dispatch is a separate follow-on (mirrors
//! `dispatch_matmul_epilogue_narrow`).
//!
//! Each module is a pure function:
//! `emit_*_msl(args) → EmittedKernel { source, function_name, buffers }`.
//! No I/O, no Metal handles touched. Pipeline build + cache lives in
//! `rce-op-metal::MetalContext::execute_named`.
//!
//! ## Reductions and the order-matters guard
//!
//! A chain `[silu, rms_norm]` puts silu in pass 1 of rms_norm's
//! two-pass mean-square reduction. A chain `[rms_norm, silu]` has to
//! finish rms_norm before silu starts. The codegen detects reduction-
//! bearing ops and emits the correct loop structure; mis-ordering would
//! silently produce wrong answers, so this is checked at emit time, not
//! just by the parity test.
//!
//! ## Equivalence vs bit-identity
//!
//! Fragment-level fusion is **not** bit-identical to running the ops
//! separately — float reduction order shifts when intermediates live in
//! registers instead of memory. Parity tests pin to `abs < 1e-4` and
//! every shipped canonical kernel has a matching parity test in the
//! consuming backend.
//!
//! ## What this crate is NOT
//!
//! - **A general graph optimiser.** It walks adjacent Free chains and
//!   rewrites them; it does not reorder, vectorise, batch, schedule, or
//!   otherwise touch the graph.
//! - **Cross-backend.** Emits MSL only. WGSL / CUDA re-emission would
//!   roughly double the fragment table; ship Metal first, port later.
//! - **A pattern discoverer.** The walker picks adjacent Free chains;
//!   the codegen is what changes about how a matched chain is
//!   *realised*. Cost-modelled pattern discovery against the placement
//!   layer's `CostTable` is separate work.

use std::collections::HashMap;

use rce_placement::fusion_eligible;
use rce_graph::graph::{FunctionGraph, GraphNode, OpKind, Wire};

pub mod codegen;
pub mod codegen_matmul_epilogue;
pub mod codegen_matmul_matmul;
pub mod codegen_matmul_tiled;
pub mod codegen_ffn_swiglu;
pub mod codegen_ffn_swiglu_tiled;

/// A registered fusion pattern: an ordered chain of op names that
/// collapses to a single fused op. Patterns are matched on **op-name
/// equality only** — types, shapes, and dtypes are not consulted (the
/// consumer kernel checks those at dispatch time).
#[derive(Clone, Debug)]
pub struct FusionPattern {
    /// Ordered chain of op names: `["silu", "rms_norm"]`.
    pub chain: Vec<&'static str>,
    /// Name of the fused op that replaces the chain.
    pub fused: &'static str,
}

impl FusionPattern {
    pub fn new(chain: Vec<&'static str>, fused: &'static str) -> Self {
        debug_assert!(chain.len() >= 2, "fusion pattern needs ≥2 ops");
        Self { chain, fused }
    }
}

/// Registry of fusion patterns. Defaults to the built-in set;
/// callers can extend via [`FusionRegistry::register`].
#[derive(Clone, Debug)]
pub struct FusionRegistry {
    patterns: Vec<FusionPattern>,
}

impl Default for FusionRegistry {
    /// The default registry ships every fusion pattern this crate
    /// has a tested kernel for. Currently: `silu → rms_norm`.
    fn default() -> Self {
        Self {
            patterns: vec![FusionPattern::new(vec!["silu", "rms_norm"], "silu_rms_norm")],
        }
    }
}

impl FusionRegistry {
    pub fn empty() -> Self { Self { patterns: vec![] } }

    pub fn register(&mut self, pattern: FusionPattern) {
        self.patterns.push(pattern);
    }

    pub fn patterns(&self) -> &[FusionPattern] { &self.patterns }

    /// Build a lookup keyed on the first op name in each pattern.
    /// `silu` → all patterns starting with `"silu"`. Single hashmap
    /// scan per parent Apply.
    fn by_head(&self) -> HashMap<&'static str, Vec<&FusionPattern>> {
        let mut out: HashMap<&'static str, Vec<&FusionPattern>> = HashMap::new();
        for p in &self.patterns {
            out.entry(p.chain[0]).or_default().push(p);
        }
        out
    }
}

/// Summary of a rewrite the pass applied. Useful for tests +
/// observability (a future stage could feed this back into the
/// transcript so post-fusion dispatch records what was rewritten).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FusionApplication {
    /// Indices of the original Apply nodes (in the pre-fusion graph)
    /// that were collapsed into `fused_node`.
    pub original_nodes: Vec<usize>,
    /// Index of the new fused Apply node in the post-fusion graph.
    pub fused_node: usize,
    /// The op name the fused node carries.
    pub fused_op: String,
}

/// Walk `graph`, apply every fusion the registry covers, and return
/// the list of applications + the rewritten graph. The input `graph`
/// is consumed; the returned `FunctionGraph` has the same `name`,
/// the same `output_node` (renumbered to track the rewrite), and a
/// strictly smaller node count when any fusion fires.
pub fn fuse(graph: FunctionGraph, registry: &FusionRegistry) -> (FunctionGraph, Vec<FusionApplication>) {
    let head_index = registry.by_head();
    let mut applications = Vec::new();
    let mut current = graph;

    loop {
        // Try one fusion per outer-loop iteration. When no candidate
        // fires, we're done. This handles overlapping fusions
        // (e.g., a→b→c with two patterns a→b and a→b→c both
        // registered — single-step fusion picks whichever matches
        // first and the loop reconsiders).
        let candidate = find_candidate(&current, &head_index);
        match candidate {
            None => break,
            Some((parent_idx, chain_indices, pattern)) => {
                let chain_indices_clone = chain_indices.clone();
                let fused_op = pattern.fused.to_string();
                let (rewritten, fused_node) =
                    apply_fusion(current, parent_idx, &chain_indices, pattern.fused);
                applications.push(FusionApplication {
                    original_nodes: chain_indices_clone,
                    fused_node,
                    fused_op,
                });
                current = rewritten;
            }
        }
    }

    (current, applications)
}

/// True iff `op` is a "transparent" shape-only graph node — one whose
/// output is bit-identical to its input under a flat-buffer view,
/// differing only in declared shape metadata. Removing such a node
/// from a chain of arg-0-connected single-consumer hops is
/// semantically a no-op for kernels (like every fragment in the
/// codegen table) that operate per element on flat buffers.
///
/// Currently lists only `reshape`. Future additions need the same
/// property: a chain `frag → op → frag` must satisfy
/// `frag(op(frag(x))) ≡ frag(frag(x))` under flat-buffer semantics
/// for it to be safe to add.
///
/// **Not classified Free by `rce-placement::op_fraction`.** That
/// classifier defaults to Bound for unknown ops as a safety
/// posture. This predicate is the codegen-level opt-in: ops the
/// graph rewriter *knows* are flat-buffer no-ops can be coalesced
/// out even though the conservative classifier wouldn't include them.
pub fn is_transparent_op(op: &str) -> bool {
    matches!(op, "reshape")
}

/// Pre-fusion graph pass: collapse [`is_transparent_op`] nodes that
/// sit between two other nodes on a single-arg-0-consumer connection,
/// rewiring the predecessor directly to the transparent node's
/// consumer. Intended to run before [`fuse_with_codegen`] so chains
/// like `silu → reshape → rms_norm` (today's SSM block has several)
/// look like `silu → rms_norm` to the fusion walker.
///
/// Safety constraints (the same conservative shape contract the
/// fusion walker uses):
///   1. The transparent node has exactly ONE incoming wire AND that
///      wire's `port_in_idx == 0`.
///   2. The transparent node has exactly ONE outgoing wire.
///   3. The transparent node is NOT the graph's `output_node`
///      (collapsing it would change the function's return shape).
///
/// If any constraint fails the node is left alone. The pass iterates
/// to fixed point — chains of consecutive transparent nodes collapse
/// in one call.
///
/// Returns the rewritten graph and the count of nodes coalesced. The
/// count is the same number of nodes the consumer can expect to be
/// missing from the result vs the input.
pub fn coalesce_transparent(graph: FunctionGraph) -> (FunctionGraph, usize) {
    let mut current = graph;
    let mut total_removed = 0usize;

    loop {
        let victim = pick_transparent_victim(&current);
        let Some((idx, pred_from, succ_to, succ_port)) = victim else {
            break;
        };
        // Rewire: drop the two wires touching `idx`, add a direct
        // wire from pred_from to succ_to at succ_port. Then renumber
        // by removing `idx` from the node list.
        current.wires.retain(|w| w.from != idx && w.to != idx);
        current.wires.push(Wire {
            from: pred_from,
            to: succ_to,
            port_in_idx: succ_port,
            wire_kind: rce_graph::graph::WireKind::Move,
        });
        current = remove_node(current, idx);
        total_removed += 1;
    }

    (current, total_removed)
}

/// Find one transparent-op node that can be safely coalesced.
/// Returns `(node_idx, predecessor_from, successor_to, successor_port)`
/// — everything the rewiring needs.
fn pick_transparent_victim(graph: &FunctionGraph) -> Option<(usize, usize, usize, i32)> {
    for idx in 0..graph.nodes.len() {
        let op_name = match &graph.nodes[idx] {
            GraphNode::Apply { op: OpKind::Custom(name), .. } => name.as_str(),
            _ => continue,
        };
        if !is_transparent_op(op_name) { continue; }
        if idx == graph.output_node { continue; }

        // Incoming wires.
        let incoming: Vec<&Wire> = graph.wires.iter().filter(|w| w.to == idx).collect();
        if incoming.len() != 1 { continue; }
        let inc = incoming[0];
        if inc.port_in_idx != 0 { continue; }

        // Outgoing wires.
        let outgoing: Vec<&Wire> = graph.wires.iter().filter(|w| w.from == idx).collect();
        if outgoing.len() != 1 { continue; }
        let out = outgoing[0];

        return Some((idx, inc.from, out.to, out.port_in_idx));
    }
    None
}

/// Remove node at `idx` from the graph: drop it from `nodes`, drop
/// every wire that touches it, and renumber every remaining index
/// that was greater than `idx`. `output_node` is renumbered (and the
/// caller has already ensured the victim isn't the output).
fn remove_node(mut graph: FunctionGraph, idx: usize) -> FunctionGraph {
    graph.nodes.remove(idx);
    let shift = |i: usize| -> usize { if i > idx { i - 1 } else { i } };
    for w in &mut graph.wires {
        w.from = shift(w.from);
        w.to = shift(w.to);
    }
    graph.output_node = shift(graph.output_node);
    graph
}

/// Codegen-driven variant of [`fuse`]. Walks the graph greedily,
/// collapsing any Free chain whose ops are all in
/// [`codegen::fragments`] and whose shape [`codegen::emit_chain_msl`]
/// accepts, into a single `Custom(<canonical_function_name>)` node.
/// After the codegen pass converges, [`fuse`] runs to pick up any
/// remaining registered patterns the codegen didn't take (e.g. a
/// pattern whose chain contains an op that has no fragment yet).
///
/// **Phase C scope limit.** This pass requires each candidate chain
/// to contain **exactly one** reduction-bearing op. The reason is the
/// args-shape contract: `rce_op_metal::dispatch_fused_chain` needs
/// `ngroups` + per-fragment scalars from the graph, and today those
/// arrive on the reduction op's existing arg slots (e.g. `rms_norm`'s
/// arg-2 `ngroups`, arg-3 `eps`). Pure-elementwise chains (zero
/// reductions) need a synthesised `ngroups=1` literal node before
/// fusion is safe — out of scope here. Multi-reduction chains need
/// a multi-stage emitter — also out of scope (codegen Phase A
/// returns `TooManyReductions`).
///
/// Callers who want the original pattern-only behaviour stay on
/// [`fuse`]; this entry point opts in to codegen.
pub fn fuse_with_codegen(
    graph: FunctionGraph,
    registry: &FusionRegistry,
) -> (FunctionGraph, Vec<FusionApplication>) {
    let mut applications = Vec::new();
    // Pre-pass: drop transparent shape-only nodes (e.g. `reshape`) so
    // chains like `silu → reshape → rms_norm` look contiguous to the
    // codegen walker. Safe by construction — `coalesce_transparent`
    // only removes nodes that pass the same single-arg-0-consumer
    // constraint the fusion walker uses, and skips the graph's output
    // node.
    let (mut current, _coalesced) = coalesce_transparent(graph);

    // Phase G: matmul-headed epilogue fusion. Detect chains shaped
    // `matmul → <elementwise fragment>+` and rewrite each to a single
    // `Custom("fused_matmul_<epilogue>_f32")` node. Runs before the
    // per-element walker so the matmul-rooted chain is taken first
    // (otherwise the per-element walker, which never starts at matmul,
    // would leave the matmul standalone and fuse only the elementwise
    // tail — the bandwidth-amplifier win lives in folding both).
    loop {
        let candidate = find_matmul_epilogue_candidate(&current);
        match candidate {
            None => break,
            Some((matmul_idx, chain_indices, fused_name)) => {
                let chain_indices_clone = chain_indices.clone();
                let (rewritten, fused_node) =
                    apply_fusion(current, matmul_idx, &chain_indices, &fused_name);
                applications.push(FusionApplication {
                    original_nodes: chain_indices_clone,
                    fused_node,
                    fused_op: fused_name,
                });
                current = rewritten;
            }
        }
    }

    // First pass: codegen-driven greedy walker.
    loop {
        let candidate = find_codegen_candidate(&current);
        match candidate {
            None => break,
            Some((parent_idx, chain_indices, fused_name)) => {
                let chain_indices_clone = chain_indices.clone();
                let (rewritten, fused_node) =
                    apply_fusion(current, parent_idx, &chain_indices, &fused_name);
                applications.push(FusionApplication {
                    original_nodes: chain_indices_clone,
                    fused_node,
                    fused_op: fused_name,
                });
                current = rewritten;
            }
        }
    }

    // Second pass: registered patterns. After codegen, this is usually a
    // no-op for chains where every op has a fragment, but it still picks
    // up patterns the codegen rejected (multi-reduction, fragment-less
    // ops, etc.).
    let (final_graph, reg_applications) = fuse(current, registry);
    applications.extend(reg_applications);
    (final_graph, applications)
}

/// Find one `matmul → <elementwise fragments>+` chain that Phase F's
/// codegen accepts. Returns `(matmul_idx, chain_indices,
/// canonical_function_name)` for the leftmost-by-node-index match,
/// or `None`.
///
/// Constraints (all conservative — leave the matmul standalone if any
/// fails):
///   1. Parent node is `Apply { Custom("matmul"), .. }`.
///   2. Parent has exactly one outgoing wire, going to its consumer's
///      arg-0 port (same single-arg-0-consumer contract the per-element
///      walker uses).
///   3. The consumer is a Phase A fragment, AND the chain extends
///      forward as far as fragments allow (greedy on length).
///   4. The chain has **at least one** epilogue fragment — Phase G
///      doesn't fuse plain matmul into `fused_matmul_f32` (that
///      would compete with BLAS/MPS for no good reason).
///   5. The codegen accepts the epilogue (no reduction-bearing
///      fragments after matmul).
///   6. Each chain link survives `codegen::parse_function_name`-style
///      skip-already-fused: parents that are themselves codegen output
///      names are passed over.
fn find_matmul_epilogue_candidate(graph: &FunctionGraph) -> Option<(usize, Vec<usize>, String)> {
    for parent_idx in 0..graph.nodes.len() {
        let parent_op = match &graph.nodes[parent_idx] {
            GraphNode::Apply { op: OpKind::Custom(name), .. } => name.as_str(),
            _ => continue,
        };
        if parent_op != "matmul" {
            continue;
        }
        // Skip nodes that already look like codegen output (defence in
        // depth; matmul wouldn't ever be a fused name, but be explicit).
        if codegen_matmul_epilogue::parse_function_name(parent_op).is_some() {
            continue;
        }

        // Greedy forward walk through fragment-table ops.
        let mut chain = vec![parent_idx];
        let mut epilogue_names: Vec<&'static str> = Vec::new();
        let mut current = parent_idx;
        while let Some(next) = find_single_arg0_consumer(graph, current) {
            let next_op = match &graph.nodes[next] {
                GraphNode::Apply { op: OpKind::Custom(n), .. } => n.as_str(),
                _ => break,
            };
            let frag = match codegen::fragments::lookup(next_op) {
                Some(f) => f,
                None => break,
            };
            // matmul-epilogue scope: no reductions after the matmul.
            if frag.has_reduction() {
                break;
            }
            chain.push(next);
            epilogue_names.push(frag.op_name);
            current = next;
        }
        if epilogue_names.is_empty() {
            // Plain matmul with no eligible epilogue — leave it.
            continue;
        }
        // Confirm the matmul-epilogue emitter accepts this chain.
        let emitted = match codegen_matmul_epilogue::emit_matmul_epilogue_msl(&epilogue_names) {
            Ok(e) => e,
            Err(_) => continue,
        };
        // CALM safety: every op classified Free (matmul is Free per the
        // placement classifier; every fragment in the table is Free).
        let mut all_names: Vec<&str> = vec!["matmul"];
        all_names.extend(epilogue_names.iter().copied());
        if !fusion_eligible(&all_names) {
            continue;
        }
        return Some((parent_idx, chain, emitted.function_name));
    }
    None
}

/// Find one codegen-eligible Free chain. Returns `(parent_idx,
/// chain_indices, canonical_function_name)` for the leftmost-by-node-
/// index match, or `None` if none.
///
/// The walker is **greedy on chain length**: at each parent it extends
/// the chain as far forward as the per-node single-arg-0-consumer
/// constraint allows, then checks whether [`codegen::emit_chain_msl`]
/// accepts the whole chain. A chain that fails the emitter (too many
/// reductions, etc.) is skipped entirely at that parent — we don't
/// truncate to a shorter still-valid prefix, because the conservative
/// shape constraint already ensures no other consumer would observe
/// the truncated fused output.
fn find_codegen_candidate(graph: &FunctionGraph) -> Option<(usize, Vec<usize>, String)> {
    for parent_idx in 0..graph.nodes.len() {
        let parent_op = match &graph.nodes[parent_idx] {
            GraphNode::Apply { op: OpKind::Custom(name), .. } => name.as_str(),
            _ => continue,
        };
        // Skip nodes that already look like a fused-codegen output —
        // re-fusing a fused node is a future phase (epilogue fusion).
        if codegen::parse_function_name(parent_op).is_some() {
            continue;
        }
        if codegen::fragments::lookup(parent_op).is_none() {
            continue;
        }

        // Greedy forward walk.
        let mut chain = vec![parent_idx];
        let mut chain_names: Vec<&'static str> = vec![
            codegen::fragments::lookup(parent_op).unwrap().op_name,
        ];
        let mut current = parent_idx;
        while let Some(next) = find_single_arg0_consumer(graph, current) {
            let next_op = match &graph.nodes[next] {
                GraphNode::Apply { op: OpKind::Custom(n), .. } => n.as_str(),
                _ => break,
            };
            let frag = match codegen::fragments::lookup(next_op) {
                Some(f) => f,
                None => break,
            };
            chain.push(next);
            chain_names.push(frag.op_name);
            current = next;
        }
        if chain.len() < 2 {
            continue;
        }

        // Phase C: require exactly one reduction in the chain.
        let n_reductions = chain_names.iter()
            .filter_map(|n| codegen::fragments::lookup(n))
            .filter(|f| f.has_reduction())
            .count();
        if n_reductions != 1 {
            continue;
        }

        // Confirm the emitter accepts this exact chain. (It should, by
        // construction — every name maps to a fragment and reduction
        // count is 1 — but make the contract explicit.)
        let emitted = match codegen::emit_chain_msl(&chain_names) {
            Ok(e) => e,
            Err(_) => continue,
        };
        // CALM sanity: every op classified Free.
        if !fusion_eligible(&chain_names) {
            continue;
        }
        return Some((parent_idx, chain, emitted.function_name));
    }
    None
}

/// Find one fusable chain. Returns `(parent_idx, chain_indices,
/// pattern)` for the leftmost-by-node-index match, or `None` if none.
///
/// Matching is conservative:
///   1. Parent is `Apply { Custom(name) }` and `name` heads at least
///      one registered pattern.
///   2. Parent has exactly ONE outgoing wire.
///   3. That wire goes to the child's input port 0 (the activation
///      slot — not a weight or scalar).
///   4. The child's op matches the pattern's second name.
///   5. Both ops are in the Free fraction (sanity check against
///      `rce_placement::fusion_eligible` — the pattern shouldn't
///      register if this fails, but we double-check).
///   6. For longer patterns (length > 2), recurse: child's output
///      must satisfy 2-4 for the next pattern step.
fn find_candidate<'a>(
    graph: &FunctionGraph,
    head_index: &'a HashMap<&'static str, Vec<&'a FusionPattern>>,
) -> Option<(usize, Vec<usize>, &'a FusionPattern)> {
    for parent_idx in 0..graph.nodes.len() {
        let parent_op = match &graph.nodes[parent_idx] {
            GraphNode::Apply { op: OpKind::Custom(name), .. } => name.as_str(),
            _ => continue,
        };
        let patterns = match head_index.get(parent_op) {
            Some(ps) => ps,
            None => continue,
        };
        for pattern in patterns {
            if let Some(chain) = try_match_chain(graph, parent_idx, pattern) {
                return Some((parent_idx, chain, pattern));
            }
        }
    }
    None
}

/// Walk forward from `start_idx`, attempting to match `pattern.chain`
/// step-by-step. Returns the indices of the matched chain nodes (in
/// order) on success.
fn try_match_chain(
    graph: &FunctionGraph,
    start_idx: usize,
    pattern: &FusionPattern,
) -> Option<Vec<usize>> {
    let mut chain = vec![start_idx];
    let mut current = start_idx;
    for step_op in &pattern.chain[1..] {
        let next = find_single_arg0_consumer(graph, current)?;
        let next_op = match &graph.nodes[next] {
            GraphNode::Apply { op: OpKind::Custom(name), .. } => name.as_str(),
            _ => return None,
        };
        if next_op != *step_op { return None; }
        chain.push(next);
        current = next;
    }
    // CALM safety check: every op in the matched chain must be Free.
    let names: Vec<&str> = pattern.chain.iter().copied().collect();
    if !fusion_eligible(&names) { return None; }
    Some(chain)
}

/// Find the single consumer of `parent_idx`'s output, IFF:
///   1. parent has exactly one outgoing wire,
///   2. that wire's `port_in_idx == 0` (the activation slot).
fn find_single_arg0_consumer(graph: &FunctionGraph, parent_idx: usize) -> Option<usize> {
    let mut out_wires: Vec<&Wire> = graph.wires.iter()
        .filter(|w| w.from == parent_idx).collect();
    if out_wires.len() != 1 { return None; }
    let w = out_wires.pop().unwrap();
    if w.port_in_idx != 0 { return None; }
    Some(w.to)
}

/// Rewrite `graph` so that the nodes in `chain` are collapsed into one
/// fused Apply node with op name `fused_op`. The fused node's inputs
/// are:
///   1. `port_in_idx == 0`: the activation from `parent`'s arg 0.
///   2. `port_in_idx >= 1`: ALL non-arg-0 inputs from EACH node in the
///      chain, concatenated in chain order. (For silu → rms_norm,
///      silu's arg 0 = x; rms_norm's args 1..3 = weight, ngroups, eps;
///      fused args = [x, weight, ngroups, eps]. Matches the fused
///      kernel's 4-arg contract.)
/// The fused node's outgoing wires take the LAST chain node's outgoing
/// wires (i.e. consumers of the fused chain's output get rewired to the
/// fused node).
fn apply_fusion(
    mut graph: FunctionGraph,
    parent_idx: usize,
    chain: &[usize],
    fused_op: &str,
) -> (FunctionGraph, usize) {
    let last_idx = *chain.last().unwrap();
    let last_output_type = match &graph.nodes[last_idx] {
        GraphNode::Apply { output_type, .. } => output_type.clone(),
        _ => unreachable!("chain entries are Apply by construction"),
    };

    // Collect the per-chain "non-arg-0" inputs in chain order. For
    // the first node in the chain we keep arg 0 too (the activation
    // input). For subsequent nodes we skip arg 0 (it's the upstream
    // chain output, which becomes internal).
    //
    // We'll build a list of (wire_index, port_for_fused_node) so we
    // can rewrite each input wire's `to` field to point at the
    // fused node with the appropriate port.
    let mut new_input_wires: Vec<Wire> = Vec::new();
    let mut next_port: i32 = 0;
    let mut chain_set: std::collections::HashSet<usize> =
        chain.iter().copied().collect();

    for (chain_step, &node_idx) in chain.iter().enumerate() {
        // Find this node's incoming wires.
        let mut incoming: Vec<&Wire> = graph.wires.iter()
            .filter(|w| w.to == node_idx).collect();
        incoming.sort_by_key(|w| w.port_in_idx);
        for w in incoming {
            // For non-first chain steps, the arg-0 wire comes from
            // the previous chain node; that becomes internal to the
            // fused node and disappears.
            if chain_step > 0 && w.port_in_idx == 0 {
                continue;
            }
            new_input_wires.push(Wire {
                from: w.from,
                to: 0, // patched below to the fused-node index
                port_in_idx: next_port,
                wire_kind: w.wire_kind.clone(),
            });
            next_port += 1;
        }
    }

    // Capture the last chain node's outgoing wires; they get rewired
    // to source from the new fused node.
    let mut new_output_wires: Vec<Wire> = graph.wires.iter()
        .filter(|w| w.from == last_idx)
        .cloned()
        .collect();

    // Build the post-fusion node list:
    //   - Keep every non-chain node at its original index slot.
    //   - Remove chain nodes.
    //   - Append the new fused Apply node at the tail.
    //
    // To do this without breaking other wires, we keep the node list
    // in place EXCEPT we re-purpose `parent_idx` (the chain head) into
    // the new fused node and DROP the rest of the chain (chain[1..]).
    // Then we compact + renumber.
    let fused_node = GraphNode::Apply {
        op: OpKind::Custom(fused_op.to_string()),
        output_type: last_output_type,
    };
    graph.nodes[parent_idx] = fused_node;
    chain_set.remove(&parent_idx);

    // Remove wires whose `from` or `to` is in chain[1..] OR is the
    // parent's old in/out (we're rebuilding inputs from
    // `new_input_wires` and outputs from `new_output_wires`).
    graph.wires.retain(|w| {
        !chain_set.contains(&w.from)
            && !chain_set.contains(&w.to)
            && !(w.to == parent_idx)
            && !(w.from == parent_idx)
    });

    // Patch + push new input wires (all target the parent_idx slot,
    // which now holds the fused node).
    for mut w in new_input_wires {
        w.to = parent_idx;
        graph.wires.push(w);
    }
    // Patch + push new output wires (source from parent_idx).
    for mut w in new_output_wires {
        w.from = parent_idx;
        graph.wires.push(w);
    }

    // Compact: remove the now-orphan chain[1..] nodes. We do this by
    // building a renumbering map and rewriting wires + output_node.
    let mut kept: Vec<usize> = (0..graph.nodes.len())
        .filter(|i| !chain_set.contains(i))
        .collect();
    kept.sort();
    let mut renumber: HashMap<usize, usize> = HashMap::new();
    for (new_idx, &old_idx) in kept.iter().enumerate() {
        renumber.insert(old_idx, new_idx);
    }
    let new_nodes: Vec<GraphNode> = kept.iter()
        .map(|&i| graph.nodes[i].clone())
        .collect();
    let new_wires: Vec<Wire> = graph.wires.iter().map(|w| Wire {
        from: *renumber.get(&w.from).expect("from index renumbered"),
        to:   *renumber.get(&w.to).expect("to index renumbered"),
        port_in_idx: w.port_in_idx,
        wire_kind: w.wire_kind.clone(),
    }).collect();
    // The original `output_node` may have pointed at one of the
    // chain's now-dropped tail nodes (typically chain[last] was the
    // last visible op in the graph). Redirect it to the fused node's
    // slot — which IS parent_idx — before renumbering.
    let output_pre_renumber = if chain_set.contains(&graph.output_node) {
        parent_idx
    } else {
        graph.output_node
    };
    let new_output_node = *renumber.get(&output_pre_renumber)
        .expect("output node renumbered after fusion redirect");
    let new_fused_node = *renumber.get(&parent_idx).expect("fused node renumbered");

    (FunctionGraph {
        name: graph.name,
        nodes: new_nodes,
        wires: new_wires,
        output_node: new_output_node,
        return_type: graph.return_type,
    }, new_fused_node)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rce_graph::graph::{Param, WireKind};

    /// Build a minimal `silu → rms_norm` chain matching the fused
    /// kernel's arg shape: param x → silu → silu_out → rms_norm
    /// with (silu_out, weight, ngroups, eps) → output.
    fn silu_then_rms_norm_graph() -> FunctionGraph {
        // nodes:
        //   0: Param "x"
        //   1: Param "weight"
        //   2: Param "ngroups"
        //   3: Param "eps"
        //   4: Apply Custom("silu"), inputs: [x]
        //   5: Apply Custom("rms_norm"), inputs: [silu_out, weight, ngroups, eps]
        // output_node = 5
        FunctionGraph {
            name: "ffn_norm".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
                GraphNode::Param(Param { name: "weight".into(), data_type: "f32".into() }),
                GraphNode::Param(Param { name: "ngroups".into(), data_type: "i32".into() }),
                GraphNode::Param(Param { name: "eps".into(), data_type: "f32".into() }),
                GraphNode::Apply {
                    op: OpKind::Custom("silu".into()),
                    output_type: "f32".into(),
                },
                GraphNode::Apply {
                    op: OpKind::Custom("rms_norm".into()),
                    output_type: "f32".into(),
                },
            ],
            wires: vec![
                // silu(x)
                Wire { from: 0, to: 4, port_in_idx: 0, wire_kind: WireKind::Move },
                // rms_norm(silu_out, weight, ngroups, eps)
                Wire { from: 4, to: 5, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 5, port_in_idx: 1, wire_kind: WireKind::Move },
                Wire { from: 2, to: 5, port_in_idx: 2, wire_kind: WireKind::Move },
                Wire { from: 3, to: 5, port_in_idx: 3, wire_kind: WireKind::Move },
            ],
            output_node: 5,
            return_type: "f32".into(),
        }
    }

    /// `silu → rms_norm → gelu`: silu produces silu_out, rms_norm
    /// consumes (silu_out, weight, ngroups, eps), gelu consumes the
    /// rms_norm output. Used to exercise codegen-driven fusion of a
    /// three-node Free chain that the registry has no pattern for.
    fn silu_rms_norm_gelu_graph() -> FunctionGraph {
        // nodes:
        //   0: Param "x"
        //   1: Param "weight"
        //   2: Param "ngroups"
        //   3: Param "eps"
        //   4: Apply Custom("silu"), inputs: [x]
        //   5: Apply Custom("rms_norm"), inputs: [silu_out, weight, ngroups, eps]
        //   6: Apply Custom("gelu"), inputs: [rms_out]
        // output_node = 6
        FunctionGraph {
            name: "ffn_norm_then_act".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(),       data_type: "f32".into() }),
                GraphNode::Param(Param { name: "weight".into(),  data_type: "f32".into() }),
                GraphNode::Param(Param { name: "ngroups".into(), data_type: "i32".into() }),
                GraphNode::Param(Param { name: "eps".into(),     data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("silu".into()),     output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("rms_norm".into()), output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("gelu".into()),     output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 4, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 4, to: 5, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 5, port_in_idx: 1, wire_kind: WireKind::Move },
                Wire { from: 2, to: 5, port_in_idx: 2, wire_kind: WireKind::Move },
                Wire { from: 3, to: 5, port_in_idx: 3, wire_kind: WireKind::Move },
                Wire { from: 5, to: 6, port_in_idx: 0, wire_kind: WireKind::Move },
            ],
            output_node: 6,
            return_type: "f32".into(),
        }
    }

    #[test]
    fn fuses_silu_then_rms_norm_into_silu_rms_norm() {
        let g = silu_then_rms_norm_graph();
        let (out, apps) = fuse(g, &FusionRegistry::default());
        assert_eq!(apps.len(), 1, "exactly one fusion applied");
        assert_eq!(apps[0].fused_op, "silu_rms_norm");
        // Node count drops by 1: silu(4) + rms_norm(5) → fused(was-4),
        // node 5 removed.
        assert_eq!(out.nodes.len(), 5);
        // The output node should now point at the fused Apply.
        match &out.nodes[out.output_node] {
            GraphNode::Apply { op: OpKind::Custom(name), .. } => {
                assert_eq!(name, "silu_rms_norm");
            }
            other => panic!("output node should be the fused Apply; got {other:?}"),
        }
        // Fused node has 4 inputs (x, weight, ngroups, eps) on ports 0..3.
        let in_wires: Vec<&Wire> = out.wires.iter()
            .filter(|w| w.to == out.output_node).collect();
        assert_eq!(in_wires.len(), 4, "fused node has 4 inputs");
        let mut ports: Vec<i32> = in_wires.iter().map(|w| w.port_in_idx).collect();
        ports.sort();
        assert_eq!(ports, vec![0, 1, 2, 3]);
    }

    #[test]
    fn does_not_fuse_when_silu_output_has_multiple_consumers() {
        // Same as above but silu's output ALSO feeds an unrelated
        // node — the rewrite would change observable semantics
        // (the second consumer would now see fused output rather
        // than silu output). Conservative pass refuses.
        let mut g = silu_then_rms_norm_graph();
        // Add a second consumer of silu_out (node 4): pretend there's
        // a residual connection going to some additional node.
        g.nodes.push(GraphNode::Apply {
            op: OpKind::Custom("residual_consumer".into()),
            output_type: "f32".into(),
        });
        let new_node_idx = g.nodes.len() - 1;
        g.wires.push(Wire { from: 4, to: new_node_idx, port_in_idx: 0, wire_kind: WireKind::Move });

        let (out, apps) = fuse(g, &FusionRegistry::default());
        assert!(apps.is_empty(),
            "fusion must skip when silu_out has multiple consumers; got {apps:?}");
        // Node count unchanged.
        assert_eq!(out.nodes.len(), 7);
    }

    #[test]
    fn does_not_fuse_when_silu_output_goes_to_non_arg0_port() {
        // silu's output feeds rms_norm's port 1 (weight slot) instead
        // of port 0 (activation slot). Semantically meaningless but
        // the pass shouldn't blindly rewrite this.
        let mut g = silu_then_rms_norm_graph();
        // Reroute silu(4)→rms_norm(5) from port 0 to port 1; reroute
        // weight(1)→rms_norm(5) from port 1 to port 0 to keep the
        // 4-port shape valid.
        for w in &mut g.wires {
            if w.from == 4 && w.to == 5 { w.port_in_idx = 1; }
            else if w.from == 1 && w.to == 5 { w.port_in_idx = 0; }
        }
        let (out, apps) = fuse(g, &FusionRegistry::default());
        assert!(apps.is_empty(),
            "fusion must skip when output goes to non-arg-0 port; got {apps:?}");
        assert_eq!(out.nodes.len(), 6);
    }

    #[test]
    fn fuses_idempotently_a_second_pass_is_a_noop() {
        let g = silu_then_rms_norm_graph();
        let (g1, apps1) = fuse(g, &FusionRegistry::default());
        assert_eq!(apps1.len(), 1);
        let (g2, apps2) = fuse(g1, &FusionRegistry::default());
        assert!(apps2.is_empty(), "second fuse pass finds no candidates");
        assert_eq!(g2.nodes.len(), 5);
    }

    #[test]
    fn empty_registry_makes_pass_a_noop() {
        let g = silu_then_rms_norm_graph();
        let original_len = g.nodes.len();
        let (out, apps) = fuse(g, &FusionRegistry::empty());
        assert!(apps.is_empty());
        assert_eq!(out.nodes.len(), original_len);
    }

    // ── Phase C: codegen-driven fusion ─────────────────────────────────

    #[test]
    fn codegen_fuses_silu_then_rms_norm_to_canonical_name() {
        // With codegen turned on, the same `silu → rms_norm` chain
        // collapses to `fused_silu_rms_norm_f32` (the codegen name)
        // instead of `silu_rms_norm` (the registered pattern). The
        // codegen path wins because it runs first.
        let g = silu_then_rms_norm_graph();
        let (out, apps) = fuse_with_codegen(g, &FusionRegistry::default());
        assert_eq!(apps.len(), 1, "exactly one fusion applied");
        assert_eq!(apps[0].fused_op, "fused_silu_rms_norm_f32");
        match &out.nodes[out.output_node] {
            GraphNode::Apply { op: OpKind::Custom(name), .. } =>
                assert_eq!(name, "fused_silu_rms_norm_f32"),
            other => panic!("output should be the codegen-fused Apply, got {other:?}"),
        }
        // Same 4-arg shape as the hand-fused variant: x, weight, ngroups, eps.
        let in_wires: Vec<&Wire> = out.wires.iter()
            .filter(|w| w.to == out.output_node).collect();
        assert_eq!(in_wires.len(), 4);
        let mut ports: Vec<i32> = in_wires.iter().map(|w| w.port_in_idx).collect();
        ports.sort();
        assert_eq!(ports, vec![0, 1, 2, 3]);
    }

    #[test]
    fn codegen_fuses_three_node_chain_with_no_registered_pattern() {
        // `silu → rms_norm → gelu` — no registered pattern, but every op
        // has a codegen fragment + the chain has exactly one reduction.
        // Codegen-driven fusion collapses it to one node; the registry
        // alone would leave it as three.
        let g = silu_rms_norm_gelu_graph();
        let (out, apps) = fuse_with_codegen(g, &FusionRegistry::default());
        assert_eq!(apps.len(), 1, "exactly one fusion applied");
        assert_eq!(apps[0].fused_op, "fused_silu_rms_norm_gelu_f32");
        match &out.nodes[out.output_node] {
            GraphNode::Apply { op: OpKind::Custom(name), .. } =>
                assert_eq!(name, "fused_silu_rms_norm_gelu_f32"),
            other => panic!("output should be the codegen-fused Apply, got {other:?}"),
        }
        // The chain's input args are the union of every node's non-arg-0
        // inputs: silu has none, rms_norm has weight/ngroups/eps, gelu has
        // none → same 4-arg shape as the silu→rms_norm chain.
        let in_wires: Vec<&Wire> = out.wires.iter()
            .filter(|w| w.to == out.output_node).collect();
        assert_eq!(in_wires.len(), 4);
        let mut ports: Vec<i32> = in_wires.iter().map(|w| w.port_in_idx).collect();
        ports.sort();
        assert_eq!(ports, vec![0, 1, 2, 3]);
        // Three original nodes collapsed.
        assert_eq!(apps[0].original_nodes.len(), 3);
    }

    #[test]
    fn codegen_does_not_fuse_zero_reduction_chain() {
        // `silu → silu` — both Free, both have fragments, but no reduction
        // in the chain. Phase C scope: chains need exactly 1 reduction
        // so `ngroups` can be sourced from the reduction op's args.
        let g = FunctionGraph {
            name: "double_silu".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("silu".into()), output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("silu".into()), output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 1, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 2, port_in_idx: 0, wire_kind: WireKind::Move },
            ],
            output_node: 2,
            return_type: "f32".into(),
        };
        let (out, apps) = fuse_with_codegen(g, &FusionRegistry::empty());
        assert!(apps.is_empty(),
            "zero-reduction chains are out of scope for Phase C; got {apps:?}");
        assert_eq!(out.nodes.len(), 3);
    }

    #[test]
    fn codegen_does_not_fuse_chain_containing_op_without_fragment() {
        // `silu → swiglu` — silu has a fragment, swiglu doesn't (yet).
        // The greedy walker stops at silu, leaving a length-1 chain →
        // no fusion.
        let g = FunctionGraph {
            name: "silu_swiglu".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(),  data_type: "f32".into() }),
                GraphNode::Param(Param { name: "up".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("silu".into()),   output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("swiglu".into()), output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 2, to: 3, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 3, port_in_idx: 1, wire_kind: WireKind::Move },
            ],
            output_node: 3,
            return_type: "f32".into(),
        };
        let (out, apps) = fuse_with_codegen(g, &FusionRegistry::empty());
        assert!(apps.is_empty(), "chain with fragment-less op should not fuse; got {apps:?}");
        assert_eq!(out.nodes.len(), 4);
    }

    #[test]
    fn codegen_pass_is_idempotent() {
        let g = silu_rms_norm_gelu_graph();
        let (g1, apps1) = fuse_with_codegen(g, &FusionRegistry::default());
        assert_eq!(apps1.len(), 1);
        let (g2, apps2) = fuse_with_codegen(g1, &FusionRegistry::default());
        assert!(apps2.is_empty(),
            "second codegen pass should be a no-op once the graph is fully fused; got {apps2:?}");
        assert_eq!(g2.nodes.len(), 5,
            "node count stable across the no-op second pass");
    }

    #[test]
    fn codegen_skips_already_fused_nodes() {
        // Hand-build a graph whose output IS a codegen-fused name
        // (mimicking the state after one fuse_with_codegen pass). The
        // walker must NOT try to extend it into another fusion.
        let g = FunctionGraph {
            name: "already_fused".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(),       data_type: "f32".into() }),
                GraphNode::Param(Param { name: "weight".into(),  data_type: "f32".into() }),
                GraphNode::Param(Param { name: "ngroups".into(), data_type: "i32".into() }),
                GraphNode::Param(Param { name: "eps".into(),     data_type: "f32".into() }),
                GraphNode::Apply {
                    op: OpKind::Custom("fused_silu_rms_norm_f32".into()),
                    output_type: "f32".into(),
                },
            ],
            wires: vec![
                Wire { from: 0, to: 4, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 4, port_in_idx: 1, wire_kind: WireKind::Move },
                Wire { from: 2, to: 4, port_in_idx: 2, wire_kind: WireKind::Move },
                Wire { from: 3, to: 4, port_in_idx: 3, wire_kind: WireKind::Move },
            ],
            output_node: 4,
            return_type: "f32".into(),
        };
        let (_, apps) = fuse_with_codegen(g, &FusionRegistry::empty());
        assert!(apps.is_empty(),
            "walker must not refuse already-fused nodes; got {apps:?}");
    }

    // ── Phase E.b: coalesce_transparent ────────────────────────────────

    #[test]
    fn coalesce_transparent_removes_single_consumer_reshape() {
        // silu → reshape → rms_norm with reshape as the only consumer
        // of silu's output. coalesce_transparent should drop the
        // reshape, rewire silu directly into rms_norm, producing
        // exactly the silu→rms_norm shape the codegen walker accepts.
        let g = FunctionGraph {
            name: "silu_reshape_rms".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(),       data_type: "f32".into() }),
                GraphNode::Param(Param { name: "weight".into(),  data_type: "f32".into() }),
                GraphNode::Param(Param { name: "ngroups".into(), data_type: "i32".into() }),
                GraphNode::Param(Param { name: "eps".into(),     data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("silu".into()),     output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("reshape".into()),  output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("rms_norm".into()), output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 4, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 4, to: 5, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 5, to: 6, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 6, port_in_idx: 1, wire_kind: WireKind::Move },
                Wire { from: 2, to: 6, port_in_idx: 2, wire_kind: WireKind::Move },
                Wire { from: 3, to: 6, port_in_idx: 3, wire_kind: WireKind::Move },
            ],
            output_node: 6,
            return_type: "f32".into(),
        };

        let (out, removed) = coalesce_transparent(g);
        assert_eq!(removed, 1, "exactly one reshape removed");
        assert_eq!(out.nodes.len(), 6, "node count drops by 1");
        // No remaining `reshape` node.
        for n in &out.nodes {
            if let GraphNode::Apply { op: OpKind::Custom(name), .. } = n {
                assert_ne!(name, "reshape");
            }
        }
        // Direct silu→rms_norm wire exists. After removing node 5
        // (reshape), node 6 (rms_norm) becomes index 5.
        let new_rms_idx = out.nodes.iter().position(|n| matches!(
            n, GraphNode::Apply { op: OpKind::Custom(name), .. } if name == "rms_norm"
        )).unwrap();
        let new_silu_idx = out.nodes.iter().position(|n| matches!(
            n, GraphNode::Apply { op: OpKind::Custom(name), .. } if name == "silu"
        )).unwrap();
        let direct = out.wires.iter().find(|w|
            w.from == new_silu_idx && w.to == new_rms_idx && w.port_in_idx == 0
        );
        assert!(direct.is_some(),
            "silu must wire directly into rms_norm's arg 0 after coalesce; got wires {:?}", out.wires);
    }

    #[test]
    fn coalesce_transparent_chains_collapse_to_fixed_point() {
        // silu → reshape → reshape → reshape → rms_norm — three
        // consecutive reshapes all single-consumer. All three should
        // collapse in a single coalesce call.
        let g = FunctionGraph {
            name: "many_reshapes".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(),      data_type: "f32".into() }),
                GraphNode::Param(Param { name: "weight".into(), data_type: "f32".into() }),
                GraphNode::Param(Param { name: "ngroups".into(),data_type: "i32".into() }),
                GraphNode::Param(Param { name: "eps".into(),    data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("silu".into()),     output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("reshape".into()),  output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("reshape".into()),  output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("reshape".into()),  output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("rms_norm".into()), output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 4, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 4, to: 5, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 5, to: 6, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 6, to: 7, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 7, to: 8, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 8, port_in_idx: 1, wire_kind: WireKind::Move },
                Wire { from: 2, to: 8, port_in_idx: 2, wire_kind: WireKind::Move },
                Wire { from: 3, to: 8, port_in_idx: 3, wire_kind: WireKind::Move },
            ],
            output_node: 8,
            return_type: "f32".into(),
        };
        let (out, removed) = coalesce_transparent(g);
        assert_eq!(removed, 3, "all three reshapes collapsed");
        assert_eq!(out.nodes.len(), 6);
        for n in &out.nodes {
            if let GraphNode::Apply { op: OpKind::Custom(name), .. } = n {
                assert_ne!(name, "reshape");
            }
        }
    }

    #[test]
    fn coalesce_transparent_skips_multi_consumer_reshape() {
        // reshape has TWO consumers — removing it would change
        // semantics for the second consumer. Conservative pass refuses.
        let g = FunctionGraph {
            name: "multi_consumer".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("silu".into()),    output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("reshape".into()), output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("gelu".into()),    output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("gelu".into()),    output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 1, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 2, port_in_idx: 0, wire_kind: WireKind::Move },
                // Both gelus consume reshape's output.
                Wire { from: 2, to: 3, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 2, to: 4, port_in_idx: 0, wire_kind: WireKind::Move },
            ],
            output_node: 3,
            return_type: "f32".into(),
        };
        let (out, removed) = coalesce_transparent(g);
        assert_eq!(removed, 0, "reshape with multiple consumers must survive");
        assert_eq!(out.nodes.len(), 5);
    }

    #[test]
    fn coalesce_transparent_skips_reshape_at_output_node() {
        // reshape IS the graph's output_node. Removing it would change
        // the function's declared return shape. Conservative pass
        // refuses.
        let g = FunctionGraph {
            name: "output_reshape".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("silu".into()),    output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("reshape".into()), output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 1, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 2, port_in_idx: 0, wire_kind: WireKind::Move },
            ],
            output_node: 2,
            return_type: "f32".into(),
        };
        let (out, removed) = coalesce_transparent(g);
        assert_eq!(removed, 0, "reshape at output must survive to preserve return shape");
        assert_eq!(out.nodes.len(), 3);
    }

    #[test]
    fn fuse_with_codegen_runs_coalesce_as_pre_pass() {
        // End-to-end: silu → reshape → rms_norm. The coalesce pre-pass
        // drops the reshape, then the codegen walker fuses the remaining
        // silu→rms_norm chain. Net: one fusion that wouldn't have fired
        // without the pre-pass.
        let g = FunctionGraph {
            name: "ssm_style".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(),       data_type: "f32".into() }),
                GraphNode::Param(Param { name: "weight".into(),  data_type: "f32".into() }),
                GraphNode::Param(Param { name: "ngroups".into(), data_type: "i32".into() }),
                GraphNode::Param(Param { name: "eps".into(),     data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("silu".into()),     output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("reshape".into()),  output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("rms_norm".into()), output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 4, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 4, to: 5, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 5, to: 6, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 6, port_in_idx: 1, wire_kind: WireKind::Move },
                Wire { from: 2, to: 6, port_in_idx: 2, wire_kind: WireKind::Move },
                Wire { from: 3, to: 6, port_in_idx: 3, wire_kind: WireKind::Move },
            ],
            output_node: 6,
            return_type: "f32".into(),
        };
        let (out, apps) = fuse_with_codegen(g, &FusionRegistry::default());
        assert_eq!(apps.len(), 1,
            "coalesce + fuse should produce exactly one fusion; got {apps:?}");
        assert_eq!(apps[0].fused_op, "fused_silu_rms_norm_f32");
        match &out.nodes[out.output_node] {
            GraphNode::Apply { op: OpKind::Custom(name), .. } =>
                assert_eq!(name, "fused_silu_rms_norm_f32"),
            other => panic!("output should be the fused Apply, got {other:?}"),
        }
    }

    // ── Phase G: matmul-epilogue graph rewriter ─────────────────────────

    /// `matmul → silu`: simplest matmul-epilogue chain. The fusion
    /// pass should detect it and collapse to one
    /// `Custom("fused_matmul_silu_f32")` node carrying (input, weight)
    /// as its two graph args.
    fn matmul_silu_graph() -> FunctionGraph {
        FunctionGraph {
            name: "matmul_silu".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(),      data_type: "f32".into() }),
                GraphNode::Param(Param { name: "weight".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("matmul".into()), output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("silu".into()),   output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Move },
                Wire { from: 2, to: 3, port_in_idx: 0, wire_kind: WireKind::Move },
            ],
            output_node: 3,
            return_type: "f32".into(),
        }
    }

    #[test]
    fn fuse_with_codegen_collapses_matmul_silu_chain() {
        let g = matmul_silu_graph();
        let (out, apps) = fuse_with_codegen(g, &FusionRegistry::default());
        assert_eq!(apps.len(), 1, "exactly one matmul-epilogue fusion; got {apps:?}");
        assert_eq!(apps[0].fused_op, "fused_matmul_silu_f32");
        // matmul + silu collapsed → 1 fused node. Plus 2 Params = 3.
        assert_eq!(out.nodes.len(), 3);
        match &out.nodes[out.output_node] {
            GraphNode::Apply { op: OpKind::Custom(name), .. } =>
                assert_eq!(name, "fused_matmul_silu_f32"),
            other => panic!("output should be fused Apply, got {other:?}"),
        }
        // Fused node's incoming wires: input at port 0, weight at port 1.
        let in_wires: Vec<&Wire> = out.wires.iter()
            .filter(|w| w.to == out.output_node).collect();
        assert_eq!(in_wires.len(), 2);
        let mut ports: Vec<i32> = in_wires.iter().map(|w| w.port_in_idx).collect();
        ports.sort();
        assert_eq!(ports, vec![0, 1]);
    }

    #[test]
    fn fuse_with_codegen_collapses_matmul_silu_gelu_chain() {
        let g = FunctionGraph {
            name: "matmul_silu_gelu".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(),      data_type: "f32".into() }),
                GraphNode::Param(Param { name: "weight".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("matmul".into()), output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("silu".into()),   output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("gelu".into()),   output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Move },
                Wire { from: 2, to: 3, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 3, to: 4, port_in_idx: 0, wire_kind: WireKind::Move },
            ],
            output_node: 4,
            return_type: "f32".into(),
        };
        let (out, apps) = fuse_with_codegen(g, &FusionRegistry::default());
        assert_eq!(apps.len(), 1, "three-node chain ⇒ one fusion");
        assert_eq!(apps[0].fused_op, "fused_matmul_silu_gelu_f32");
        assert_eq!(apps[0].original_nodes.len(), 3);
        assert_eq!(out.nodes.len(), 3, "matmul + silu + gelu collapsed to 1 + 2 params");
    }

    #[test]
    fn fuse_with_codegen_collapses_matmul_swiglu_chain_with_up_arg() {
        // matmul → swiglu pulls swiglu's `up` (port 1, a different
        // matmul's output in real FFN) into the fused node as an extra
        // arg slot. Net args on the fused node: (input, weight, up) at
        // ports 0, 1, 2 — exactly the matmul-epilogue dispatch's
        // shape-inference layout.
        let g = FunctionGraph {
            name: "matmul_swiglu".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(),      data_type: "f32".into() }),
                GraphNode::Param(Param { name: "weight".into(), data_type: "f32".into() }),
                GraphNode::Param(Param { name: "up".into(),     data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("matmul".into()), output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("swiglu".into()), output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 3, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 3, port_in_idx: 1, wire_kind: WireKind::Move },
                // matmul output is swiglu's port 0 (gate).
                Wire { from: 3, to: 4, port_in_idx: 0, wire_kind: WireKind::Move },
                // up is swiglu's port 1.
                Wire { from: 2, to: 4, port_in_idx: 1, wire_kind: WireKind::Move },
            ],
            output_node: 4,
            return_type: "f32".into(),
        };
        let (out, apps) = fuse_with_codegen(g, &FusionRegistry::default());
        assert_eq!(apps.len(), 1);
        assert_eq!(apps[0].fused_op, "fused_matmul_swiglu_f32");
        let in_wires: Vec<&Wire> = out.wires.iter()
            .filter(|w| w.to == out.output_node).collect();
        assert_eq!(in_wires.len(), 3, "fused node has (input, weight, up)");
        let mut ports: Vec<i32> = in_wires.iter().map(|w| w.port_in_idx).collect();
        ports.sort();
        assert_eq!(ports, vec![0, 1, 2]);
    }

    #[test]
    fn fuse_with_codegen_does_not_fuse_plain_matmul() {
        // matmul with no Free-fragment consumer: stays as-is. Plain
        // matmul falls through to BLAS.
        let g = FunctionGraph {
            name: "plain_matmul".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(),      data_type: "f32".into() }),
                GraphNode::Param(Param { name: "weight".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("matmul".into()), output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Move },
            ],
            output_node: 2,
            return_type: "f32".into(),
        };
        let (_, apps) = fuse_with_codegen(g, &FusionRegistry::default());
        assert!(apps.is_empty(), "plain matmul without epilogue must not fuse; got {apps:?}");
    }

    #[test]
    fn fuse_with_codegen_does_not_fuse_matmul_then_rms_norm() {
        // rms_norm carries an output-axis reduction — incompatible with
        // matmul's per-output-element threading. Phase G must skip this.
        let g = FunctionGraph {
            name: "matmul_rms".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(),       data_type: "f32".into() }),
                GraphNode::Param(Param { name: "weight".into(),  data_type: "f32".into() }),
                GraphNode::Param(Param { name: "rms_w".into(),   data_type: "f32".into() }),
                GraphNode::Param(Param { name: "ngroups".into(), data_type: "i32".into() }),
                GraphNode::Param(Param { name: "eps".into(),     data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("matmul".into()),   output_type: "f32".into() },
                GraphNode::Apply { op: OpKind::Custom("rms_norm".into()), output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 5, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 5, port_in_idx: 1, wire_kind: WireKind::Move },
                Wire { from: 5, to: 6, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 2, to: 6, port_in_idx: 1, wire_kind: WireKind::Move },
                Wire { from: 3, to: 6, port_in_idx: 2, wire_kind: WireKind::Move },
                Wire { from: 4, to: 6, port_in_idx: 3, wire_kind: WireKind::Move },
            ],
            output_node: 6,
            return_type: "f32".into(),
        };
        let (_, apps) = fuse_with_codegen(g, &FusionRegistry::default());
        // The matmul-epilogue walker rejects the chain (reduction in
        // epilogue), so matmul stays. The per-element walker sees
        // only rms_norm alone after that — length 1, doesn't fuse.
        assert!(apps.is_empty(),
            "matmul → rms_norm has reduction in epilogue; Phase G must skip; got {apps:?}");
    }

    #[test]
    fn fuse_with_codegen_runs_matmul_then_per_element() {
        // `matmul → silu` AND a separate `silu → rms_norm → gelu` chain
        // in the same graph. Both walkers fire; the matmul-rooted one
        // runs first, then the per-element walker picks up the
        // independent chain. Two fusions total.
        let g = FunctionGraph {
            name: "two_chains".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(),       data_type: "f32".into() }),
                GraphNode::Param(Param { name: "w_mm".into(),    data_type: "f32".into() }),
                GraphNode::Param(Param { name: "y".into(),       data_type: "f32".into() }),
                GraphNode::Param(Param { name: "w_rms".into(),   data_type: "f32".into() }),
                GraphNode::Param(Param { name: "ngroups".into(), data_type: "i32".into() }),
                GraphNode::Param(Param { name: "eps".into(),     data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Custom("matmul".into()),   output_type: "f32".into() }, // 6
                GraphNode::Apply { op: OpKind::Custom("silu".into()),     output_type: "f32".into() }, // 7 — matmul tail
                GraphNode::Apply { op: OpKind::Custom("silu".into()),     output_type: "f32".into() }, // 8 — independent chain head
                GraphNode::Apply { op: OpKind::Custom("rms_norm".into()), output_type: "f32".into() }, // 9
                GraphNode::Apply { op: OpKind::Custom("gelu".into()),     output_type: "f32".into() }, // 10
                GraphNode::Apply { op: OpKind::Custom("tensor_concat".into()), output_type: "f32".into() }, // 11 — joins both outputs
            ],
            wires: vec![
                Wire { from: 0, to: 6, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 6, port_in_idx: 1, wire_kind: WireKind::Move },
                Wire { from: 6, to: 7, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 2, to: 8, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 8, to: 9, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 3, to: 9, port_in_idx: 1, wire_kind: WireKind::Move },
                Wire { from: 4, to: 9, port_in_idx: 2, wire_kind: WireKind::Move },
                Wire { from: 5, to: 9, port_in_idx: 3, wire_kind: WireKind::Move },
                Wire { from: 9, to: 10, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 7, to: 11, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 10, to: 11, port_in_idx: 1, wire_kind: WireKind::Move },
            ],
            output_node: 11,
            return_type: "f32".into(),
        };
        let (_, apps) = fuse_with_codegen(g, &FusionRegistry::default());
        let names: Vec<&str> = apps.iter().map(|a| a.fused_op.as_str()).collect();
        assert!(names.contains(&"fused_matmul_silu_f32"),
            "matmul→silu must fuse; apps: {names:?}");
        assert!(names.contains(&"fused_silu_rms_norm_gelu_f32"),
            "independent silu→rms_norm→gelu chain must fuse; apps: {names:?}");
        assert_eq!(apps.len(), 2);
    }

    #[test]
    fn fuse_with_codegen_matmul_pass_is_idempotent() {
        let g = matmul_silu_graph();
        let (g1, apps1) = fuse_with_codegen(g, &FusionRegistry::default());
        assert_eq!(apps1.len(), 1);
        let (g2, apps2) = fuse_with_codegen(g1, &FusionRegistry::default());
        assert!(apps2.is_empty(), "second pass on already-fused graph is a no-op; got {apps2:?}");
        let _ = g2;
    }

    #[test]
    fn registry_extends_with_caller_supplied_pattern() {
        // Caller registers a hypothetical (yet-unimplemented) pattern.
        // The pass picks it up if the chain matches; the consumer
        // backend would need a kernel for the fused op name, but
        // that's separate from the pass's correctness.
        let mut reg = FusionRegistry::empty();
        reg.register(FusionPattern::new(vec!["silu", "rms_norm"], "my_custom_fused"));
        let g = silu_then_rms_norm_graph();
        let (out, apps) = fuse(g, &reg);
        assert_eq!(apps.len(), 1);
        assert_eq!(apps[0].fused_op, "my_custom_fused");
        match &out.nodes[out.output_node] {
            GraphNode::Apply { op: OpKind::Custom(name), .. } => {
                assert_eq!(name, "my_custom_fused");
            }
            _ => panic!("output should be fused"),
        }
    }
}
