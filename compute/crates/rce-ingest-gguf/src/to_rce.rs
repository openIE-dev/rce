//! Convert a parsed `GgufFile` into a RCE `ProgramGraph`.
//!
//! Each detected transformer block becomes a `FunctionGraph` representing
//! the layer's compute (attention + feed-forward).  A top-level "model"
//! function graph orchestrates the layer calls.
//!
//! Tensor name patterns follow the llama.cpp convention:
//! - `blk.{N}.attn_q.weight`   → query projection
//! - `blk.{N}.attn_k.weight`   → key projection
//! - `blk.{N}.attn_v.weight`   → value projection
//! - `blk.{N}.attn_output.weight` → output projection
//! - `blk.{N}.ffn_gate.weight` → gated FFN (SiLU gate)
//! - `blk.{N}.ffn_up.weight`   → FFN up projection
//! - `blk.{N}.ffn_down.weight` → FFN down projection
//! - `blk.{N}.attn_norm.weight` → pre-attention layer norm
//! - `blk.{N}.ffn_norm.weight`  → pre-FFN layer norm
//! - `token_embd.weight`       → token embedding table
//! - `output_norm.weight`      → final layer norm
//! - `output.weight`           → output head (lm_head)

use std::collections::BTreeMap;

use rce_graph::{
    FunctionGraph, GraphNode, OpKind, Param, ProgramGraph, Wire, WireKind,
};

use crate::parser::{GgufFile, GgufTensorInfo};
use crate::quant::quant_info;

// ── Block detection ────────────────────────────────────────────────────

/// Tensors belonging to a single transformer block.
#[derive(Debug, Default)]
struct BlockTensors<'a> {
    attn_q: Option<&'a GgufTensorInfo>,
    attn_k: Option<&'a GgufTensorInfo>,
    attn_v: Option<&'a GgufTensorInfo>,
    attn_output: Option<&'a GgufTensorInfo>,
    ffn_gate: Option<&'a GgufTensorInfo>,
    ffn_up: Option<&'a GgufTensorInfo>,
    ffn_down: Option<&'a GgufTensorInfo>,
    attn_norm: Option<&'a GgufTensorInfo>,
    ffn_norm: Option<&'a GgufTensorInfo>,
}

/// Parse a tensor name like `blk.5.attn_q.weight` into (block_index, suffix).
fn parse_block_tensor(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    let idx: usize = rest[..dot].parse().ok()?;
    let suffix = &rest[dot + 1..];
    Some((idx, suffix))
}

/// Group tensors into blocks by index.
fn group_blocks(tensors: &[GgufTensorInfo]) -> BTreeMap<usize, BlockTensors<'_>> {
    let mut blocks: BTreeMap<usize, BlockTensors<'_>> = BTreeMap::new();
    for t in tensors {
        if let Some((idx, suffix)) = parse_block_tensor(&t.name) {
            let blk = blocks.entry(idx).or_default();
            match suffix {
                "attn_q.weight" => blk.attn_q = Some(t),
                "attn_k.weight" => blk.attn_k = Some(t),
                "attn_v.weight" => blk.attn_v = Some(t),
                "attn_output.weight" => blk.attn_output = Some(t),
                "ffn_gate.weight" => blk.ffn_gate = Some(t),
                "ffn_up.weight" => blk.ffn_up = Some(t),
                "ffn_down.weight" => blk.ffn_down = Some(t),
                "attn_norm.weight" => blk.attn_norm = Some(t),
                "ffn_norm.weight" => blk.ffn_norm = Some(t),
                _ => {} // bias tensors, rotary, etc. — skip for now
            }
        }
    }
    blocks
}

// ── Graph builders ─────────────────────────────────────────────────────

/// Type string for a tensor: `"{rce_type}[dim0 x dim1 x ...]"`.
fn tensor_type_str(t: &GgufTensorInfo) -> String {
    let qi = quant_info(t.dtype);
    let dims: Vec<String> = t.dims.iter().map(|d| d.to_string()).collect();
    format!("{}[{}]", qi.rce_type, dims.join("x"))
}

/// Create a Param node for a weight tensor.
fn weight_param(t: &GgufTensorInfo) -> GraphNode {
    GraphNode::Param(Param {
        name: t.name.clone(),
        data_type: tensor_type_str(t),
    })
}

/// Build a FunctionGraph for one transformer block.
///
/// Layout (node indices):
/// 0: input (hidden state param)
/// 1..N: weight params for all present tensors
/// Then Apply nodes for the compute graph:
/// - attn_norm → Q/K/V matmuls → attention → attn_output matmul → residual add
/// - ffn_norm → gate matmul → silu → up matmul → element-wise mul → down matmul → residual add
fn build_block_graph(block_idx: usize, blk: &BlockTensors<'_>) -> FunctionGraph {
    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut wires: Vec<Wire> = Vec::new();

    // Node 0: input hidden state
    nodes.push(GraphNode::Param(Param {
        name: "hidden_state".into(),
        data_type: "tensor".into(),
    }));
    let input_idx = 0;

    /// Push a weight param node, return its index.
    fn push_weight(nodes: &mut Vec<GraphNode>, t: &GgufTensorInfo) -> usize {
        let idx = nodes.len();
        nodes.push(weight_param(t));
        idx
    }

    // ── Attention sub-graph ────────────────────────────────────────
    let mut attn_out_idx = input_idx;

    // Pre-attention norm
    if let Some(norm) = blk.attn_norm {
        let w_idx = push_weight(&mut nodes, norm);
        let norm_idx = nodes.len();
        nodes.push(GraphNode::Apply {
            op: OpKind::Custom("rms_norm".into()),
            output_type: "tensor".into(),
        });
        wires.push(Wire { from: input_idx, to: norm_idx, port_in_idx: 0, wire_kind: WireKind::Borrow });
        wires.push(Wire { from: w_idx, to: norm_idx, port_in_idx: 1, wire_kind: WireKind::Borrow });
        attn_out_idx = norm_idx;
    }

    let norm_out = attn_out_idx;

    // Q, K, V projections
    let mut qkv_indices = [None; 3];
    for (i, maybe_t) in [blk.attn_q, blk.attn_k, blk.attn_v].iter().enumerate() {
        if let Some(t) = maybe_t {
            let w_idx = push_weight(&mut nodes, t);
            let mm_idx = nodes.len();
            nodes.push(GraphNode::Apply {
                op: OpKind::Custom("matmul".into()),
                output_type: "tensor".into(),
            });
            wires.push(Wire { from: norm_out, to: mm_idx, port_in_idx: 0, wire_kind: WireKind::Borrow });
            wires.push(Wire { from: w_idx, to: mm_idx, port_in_idx: 1, wire_kind: WireKind::Borrow });
            qkv_indices[i] = Some(mm_idx);
        }
    }

    // Attention (Q, K, V → attended output)
    if let (Some(q_idx), Some(k_idx), Some(v_idx)) =
        (qkv_indices[0], qkv_indices[1], qkv_indices[2])
    {
        let attn_idx = nodes.len();
        nodes.push(GraphNode::Apply {
            op: OpKind::Custom("attention".into()),
            output_type: "tensor".into(),
        });
        wires.push(Wire { from: q_idx, to: attn_idx, port_in_idx: 0, wire_kind: WireKind::Move });
        wires.push(Wire { from: k_idx, to: attn_idx, port_in_idx: 1, wire_kind: WireKind::Move });
        wires.push(Wire { from: v_idx, to: attn_idx, port_in_idx: 2, wire_kind: WireKind::Move });
        attn_out_idx = attn_idx;

        // Output projection
        if let Some(t) = blk.attn_output {
            let w_idx = push_weight(&mut nodes, t);
            let proj_idx = nodes.len();
            nodes.push(GraphNode::Apply {
                op: OpKind::Custom("matmul".into()),
                output_type: "tensor".into(),
            });
            wires.push(Wire { from: attn_out_idx, to: proj_idx, port_in_idx: 0, wire_kind: WireKind::Move });
            wires.push(Wire { from: w_idx, to: proj_idx, port_in_idx: 1, wire_kind: WireKind::Borrow });
            attn_out_idx = proj_idx;
        }
    }

    // Residual add after attention
    let residual1_idx = nodes.len();
    nodes.push(GraphNode::Apply {
        op: OpKind::Add,
        output_type: "tensor".into(),
    });
    wires.push(Wire { from: input_idx, to: residual1_idx, port_in_idx: 0, wire_kind: WireKind::Borrow });
    wires.push(Wire { from: attn_out_idx, to: residual1_idx, port_in_idx: 1, wire_kind: WireKind::Move });

    // ── Feed-forward sub-graph ─────────────────────────────────────
    let mut ffn_out_idx = residual1_idx;

    // Pre-FFN norm
    if let Some(norm) = blk.ffn_norm {
        let w_idx = push_weight(&mut nodes, norm);
        let norm_idx = nodes.len();
        nodes.push(GraphNode::Apply {
            op: OpKind::Custom("rms_norm".into()),
            output_type: "tensor".into(),
        });
        wires.push(Wire { from: residual1_idx, to: norm_idx, port_in_idx: 0, wire_kind: WireKind::Borrow });
        wires.push(Wire { from: w_idx, to: norm_idx, port_in_idx: 1, wire_kind: WireKind::Borrow });
        ffn_out_idx = norm_idx;
    }

    let ffn_norm_out = ffn_out_idx;

    // Gated SiLU FFN: silu(gate(x)) * up(x), then down projection
    let has_gated_ffn = blk.ffn_gate.is_some() && blk.ffn_up.is_some() && blk.ffn_down.is_some();
    if has_gated_ffn {
        // gate matmul
        let gate_w_idx = push_weight(&mut nodes, blk.ffn_gate.unwrap());
        let gate_mm_idx = nodes.len();
        nodes.push(GraphNode::Apply {
            op: OpKind::Custom("matmul".into()),
            output_type: "tensor".into(),
        });
        wires.push(Wire { from: ffn_norm_out, to: gate_mm_idx, port_in_idx: 0, wire_kind: WireKind::Borrow });
        wires.push(Wire { from: gate_w_idx, to: gate_mm_idx, port_in_idx: 1, wire_kind: WireKind::Borrow });

        // silu activation on gate
        let silu_idx = nodes.len();
        nodes.push(GraphNode::Apply {
            op: OpKind::Custom("silu".into()),
            output_type: "tensor".into(),
        });
        wires.push(Wire { from: gate_mm_idx, to: silu_idx, port_in_idx: 0, wire_kind: WireKind::Move });

        // up matmul
        let up_w_idx = push_weight(&mut nodes, blk.ffn_up.unwrap());
        let up_mm_idx = nodes.len();
        nodes.push(GraphNode::Apply {
            op: OpKind::Custom("matmul".into()),
            output_type: "tensor".into(),
        });
        wires.push(Wire { from: ffn_norm_out, to: up_mm_idx, port_in_idx: 0, wire_kind: WireKind::Borrow });
        wires.push(Wire { from: up_w_idx, to: up_mm_idx, port_in_idx: 1, wire_kind: WireKind::Borrow });

        // element-wise multiply: silu(gate) * up
        let mul_idx = nodes.len();
        nodes.push(GraphNode::Apply {
            op: OpKind::Mul,
            output_type: "tensor".into(),
        });
        wires.push(Wire { from: silu_idx, to: mul_idx, port_in_idx: 0, wire_kind: WireKind::Move });
        wires.push(Wire { from: up_mm_idx, to: mul_idx, port_in_idx: 1, wire_kind: WireKind::Move });

        // down matmul
        let down_w_idx = push_weight(&mut nodes, blk.ffn_down.unwrap());
        let down_mm_idx = nodes.len();
        nodes.push(GraphNode::Apply {
            op: OpKind::Custom("matmul".into()),
            output_type: "tensor".into(),
        });
        wires.push(Wire { from: mul_idx, to: down_mm_idx, port_in_idx: 0, wire_kind: WireKind::Move });
        wires.push(Wire { from: down_w_idx, to: down_mm_idx, port_in_idx: 1, wire_kind: WireKind::Borrow });
        ffn_out_idx = down_mm_idx;
    }

    // Residual add after FFN
    let residual2_idx = nodes.len();
    nodes.push(GraphNode::Apply {
        op: OpKind::Add,
        output_type: "tensor".into(),
    });
    wires.push(Wire { from: residual1_idx, to: residual2_idx, port_in_idx: 0, wire_kind: WireKind::Borrow });
    wires.push(Wire { from: ffn_out_idx, to: residual2_idx, port_in_idx: 1, wire_kind: WireKind::Move });

    let output_node = residual2_idx;

    FunctionGraph {
        name: format!("block_{block_idx}"),
        nodes,
        wires,
        output_node,
        return_type: "tensor".into(),
    }
}

/// Build the top-level model graph that wires embedding → blocks → output head.
fn build_model_graph(
    gguf: &GgufFile,
    block_count: usize,
) -> FunctionGraph {
    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut wires: Vec<Wire> = Vec::new();

    // Node 0: input token IDs
    nodes.push(GraphNode::Param(Param {
        name: "input_ids".into(),
        data_type: "i32[]".into(),
    }));

    // Token embedding lookup
    let embd_tensor = gguf.tensors.iter().find(|t| t.name == "token_embd.weight");
    let mut current_idx = 0; // start with input_ids

    if let Some(embd) = embd_tensor {
        let w_idx = nodes.len();
        nodes.push(weight_param(embd));
        let lookup_idx = nodes.len();
        nodes.push(GraphNode::Apply {
            op: OpKind::Custom("embedding_lookup".into()),
            output_type: "tensor".into(),
        });
        wires.push(Wire { from: 0, to: lookup_idx, port_in_idx: 0, wire_kind: WireKind::Borrow });
        wires.push(Wire { from: w_idx, to: lookup_idx, port_in_idx: 1, wire_kind: WireKind::Borrow });
        current_idx = lookup_idx;
    }

    // Call each block function
    for blk_idx in 0..block_count {
        let call_idx = nodes.len();
        nodes.push(GraphNode::Call {
            callee: format!("block_{blk_idx}"),
            arg_types: vec!["tensor".into()],
            return_type: "tensor".into(),
        });
        wires.push(Wire {
            from: current_idx,
            to: call_idx,
            port_in_idx: 0,
            wire_kind: WireKind::Move,
        });
        current_idx = call_idx;
    }

    // Final norm
    let norm_tensor = gguf.tensors.iter().find(|t| t.name == "output_norm.weight");
    if let Some(norm) = norm_tensor {
        let w_idx = nodes.len();
        nodes.push(weight_param(norm));
        let norm_idx = nodes.len();
        nodes.push(GraphNode::Apply {
            op: OpKind::Custom("rms_norm".into()),
            output_type: "tensor".into(),
        });
        wires.push(Wire { from: current_idx, to: norm_idx, port_in_idx: 0, wire_kind: WireKind::Move });
        wires.push(Wire { from: w_idx, to: norm_idx, port_in_idx: 1, wire_kind: WireKind::Borrow });
        current_idx = norm_idx;
    }

    // Output head (lm_head)
    let output_tensor = gguf.tensors.iter().find(|t| t.name == "output.weight");
    if let Some(out) = output_tensor {
        let w_idx = nodes.len();
        nodes.push(weight_param(out));
        let mm_idx = nodes.len();
        nodes.push(GraphNode::Apply {
            op: OpKind::Custom("matmul".into()),
            output_type: "tensor".into(),
        });
        wires.push(Wire { from: current_idx, to: mm_idx, port_in_idx: 0, wire_kind: WireKind::Move });
        wires.push(Wire { from: w_idx, to: mm_idx, port_in_idx: 1, wire_kind: WireKind::Borrow });
        current_idx = mm_idx;
    }

    FunctionGraph {
        name: "model".into(),
        nodes,
        wires,
        output_node: current_idx,
        return_type: "tensor".into(),
    }
}

// ── Public API ─────────────────────────────────────────────────────────

/// Convert a parsed GGUF file into a RCE `ProgramGraph`.
///
/// - Each transformer block (detected via `blk.{N}.*` tensor names) becomes
///   its own `FunctionGraph`.
/// - A top-level `model` function wires embedding → blocks → output head.
/// - If no block tensors are detected, produces a single model graph with
///   whatever top-level tensors are available.
pub fn to_program_graph(gguf: &GgufFile) -> ProgramGraph {
    let blocks = group_blocks(&gguf.tensors);
    let mut functions: Vec<FunctionGraph> = Vec::new();

    // Build per-block function graphs
    for (&idx, blk) in &blocks {
        functions.push(build_block_graph(idx, blk));
    }

    // Build top-level model graph
    let block_count = blocks.len();
    let model = build_model_graph(gguf, block_count);
    functions.push(model);

    ProgramGraph {
        functions,
        entry: "model".into(),
        allows: Default::default(),
    }
}

/// Convert individual tensors to graph Param nodes (useful for
/// inspection without building a full model graph).
pub fn tensors_to_params(tensors: &[GgufTensorInfo]) -> Vec<GraphNode> {
    tensors.iter().map(|t| weight_param(t)).collect()
}

// ── Summary / diagnostic ───────────────────────────────────────────────

/// Summary of the model architecture extracted from GGUF.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelSummary {
    pub architecture: Option<String>,
    pub model_name: Option<String>,
    pub block_count: usize,
    pub total_tensors: usize,
    pub total_params: u64,
    pub quant_distribution: Vec<(String, usize)>,
}

/// Build a summary of the model from the parsed GGUF file.
pub fn summarize(gguf: &GgufFile) -> ModelSummary {
    let blocks = group_blocks(&gguf.tensors);

    let mut quant_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut total_params: u64 = 0;
    for t in &gguf.tensors {
        let qi = quant_info(t.dtype);
        *quant_counts.entry(qi.rce_type.to_string()).or_default() += 1;
        total_params += t.numel();
    }

    ModelSummary {
        architecture: gguf.architecture().map(String::from),
        model_name: gguf.model_name().map(String::from),
        block_count: blocks.len(),
        total_tensors: gguf.tensors.len(),
        total_params,
        quant_distribution: quant_counts.into_iter().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::*;

    /// Helper: build a gguf with llama-like tensors for N blocks.
    fn llama_like_gguf(num_blocks: usize) -> GgufFile {
        let arch_val = encode_string_value("llama");
        let name_val = encode_string_value("test-model");
        let blocks_val = encode_u32_value(num_blocks as u32);
        let embed_val = encode_u32_value(4096);

        let mut tensor_descs: Vec<(String, Vec<u64>, u32, u64)> = Vec::new();
        tensor_descs.push(("token_embd.weight".into(), vec![32000, 4096], 1, 0));
        tensor_descs.push(("output_norm.weight".into(), vec![4096], 0, 0));
        tensor_descs.push(("output.weight".into(), vec![32000, 4096], 12, 0));

        for blk in 0..num_blocks {
            let suffixes = [
                ("attn_q.weight", vec![4096u64, 4096], 12u32),
                ("attn_k.weight", vec![4096, 4096], 12),
                ("attn_v.weight", vec![4096, 4096], 12),
                ("attn_output.weight", vec![4096, 4096], 12),
                ("ffn_gate.weight", vec![11008, 4096], 12),
                ("ffn_up.weight", vec![11008, 4096], 12),
                ("ffn_down.weight", vec![4096, 11008], 12),
                ("attn_norm.weight", vec![4096], 0),
                ("ffn_norm.weight", vec![4096], 0),
            ];
            for (suffix, dims, dtype) in &suffixes {
                tensor_descs.push((format!("blk.{blk}.{suffix}"), dims.clone(), *dtype, 0));
            }
        }

        // Build the raw bytes
        let raw_tensors: Vec<(&str, &[u64], u32, u64)> = tensor_descs
            .iter()
            .map(|(n, d, dt, o)| (n.as_str(), d.as_slice(), *dt, *o))
            .collect();

        let data = build_test_gguf(
            &[
                ("general.architecture", 8, &arch_val),
                ("general.name", 8, &name_val),
                ("llama.block_count", 4, &blocks_val),
                ("llama.embedding_length", 4, &embed_val),
            ],
            &raw_tensors,
        );
        GgufFile::parse(&data).unwrap()
    }

    #[test]
    fn single_block_graph() {
        let gguf = llama_like_gguf(1);
        let prog = to_program_graph(&gguf);
        // 1 block function + 1 model function
        assert_eq!(prog.functions.len(), 2);
        assert_eq!(prog.entry, "model");
        assert_eq!(prog.functions[0].name, "block_0");
        assert_eq!(prog.functions[1].name, "model");
    }

    #[test]
    fn multi_block_graph() {
        let gguf = llama_like_gguf(3);
        let prog = to_program_graph(&gguf);
        assert_eq!(prog.functions.len(), 4); // 3 blocks + model
        assert_eq!(prog.functions[0].name, "block_0");
        assert_eq!(prog.functions[1].name, "block_1");
        assert_eq!(prog.functions[2].name, "block_2");
        assert_eq!(prog.functions[3].name, "model");
    }

    #[test]
    fn block_graph_has_params_and_ops() {
        let gguf = llama_like_gguf(1);
        let prog = to_program_graph(&gguf);
        let block = &prog.functions[0];
        // Should have: 1 input param + 9 weight params + multiple Apply nodes
        let param_count = block.nodes.iter().filter(|n| matches!(n, GraphNode::Param(_))).count();
        assert!(param_count >= 10, "expected at least 10 params (1 input + 9 weights), got {param_count}");
        let apply_count = block.nodes.iter().filter(|n| matches!(n, GraphNode::Apply { .. })).count();
        assert!(apply_count >= 5, "expected at least 5 apply nodes, got {apply_count}");
        assert!(!block.wires.is_empty());
    }

    #[test]
    fn model_graph_has_calls() {
        let gguf = llama_like_gguf(2);
        let prog = to_program_graph(&gguf);
        let model = prog.functions.last().unwrap();
        assert_eq!(model.name, "model");
        let call_count = model.nodes.iter().filter(|n| matches!(n, GraphNode::Call { .. })).count();
        assert_eq!(call_count, 2, "model should call 2 block functions");
    }

    #[test]
    fn model_graph_has_embedding_and_output() {
        let gguf = llama_like_gguf(1);
        let prog = to_program_graph(&gguf);
        let model = prog.functions.last().unwrap();

        // Check for embedding_lookup op
        let has_embd = model.nodes.iter().any(|n| matches!(n, GraphNode::Apply { op: OpKind::Custom(s), .. } if s == "embedding_lookup"));
        assert!(has_embd, "model should have embedding_lookup");

        // Check for output matmul (lm_head)
        let matmul_count = model.nodes.iter().filter(|n| matches!(n, GraphNode::Apply { op: OpKind::Custom(s), .. } if s == "matmul")).count();
        assert!(matmul_count >= 1, "model should have at least one matmul for output head");

        // Check for rms_norm (output_norm)
        let has_norm = model.nodes.iter().any(|n| matches!(n, GraphNode::Apply { op: OpKind::Custom(s), .. } if s == "rms_norm"));
        assert!(has_norm, "model should have rms_norm for output_norm");
    }

    #[test]
    fn tensors_to_params_basic() {
        let gguf = llama_like_gguf(1);
        let params = tensors_to_params(&gguf.tensors);
        assert_eq!(params.len(), gguf.tensors.len());
        for (p, t) in params.iter().zip(&gguf.tensors) {
            match p {
                GraphNode::Param(param) => {
                    assert_eq!(param.name, t.name);
                    assert!(!param.data_type.is_empty());
                }
                _ => panic!("expected Param node"),
            }
        }
    }

    #[test]
    fn param_type_includes_quant_and_dims() {
        let info = GgufTensorInfo {
            name: "blk.0.attn_q.weight".into(),
            dims: vec![4096, 4096],
            dtype: GgufDtype::Q4K,
            offset: 0,
        };
        let params = tensors_to_params(&[info]);
        match &params[0] {
            GraphNode::Param(p) => {
                assert_eq!(p.data_type, "q4_k[4096x4096]");
            }
            _ => panic!("expected Param"),
        }
    }

    #[test]
    fn empty_gguf_produces_model_only() {
        let data = build_test_gguf(&[], &[]);
        let gguf = GgufFile::parse(&data).unwrap();
        let prog = to_program_graph(&gguf);
        assert_eq!(prog.functions.len(), 1);
        assert_eq!(prog.functions[0].name, "model");
    }

    #[test]
    fn summary_reports_quant_distribution() {
        let gguf = llama_like_gguf(2);
        let s = summarize(&gguf);
        assert_eq!(s.architecture.as_deref(), Some("llama"));
        assert_eq!(s.model_name.as_deref(), Some("test-model"));
        assert_eq!(s.block_count, 2);
        assert!(s.total_tensors > 0);
        assert!(s.total_params > 0);
        // Should have at least f32 and q4_k entries
        let types: Vec<&str> = s.quant_distribution.iter().map(|(k, _)| k.as_str()).collect();
        assert!(types.contains(&"q4_k"), "should have q4_k tensors");
    }

    #[test]
    fn block_graph_wires_are_valid() {
        let gguf = llama_like_gguf(1);
        let prog = to_program_graph(&gguf);
        let block = &prog.functions[0];
        let n = block.nodes.len();
        for wire in &block.wires {
            assert!(wire.from < n, "wire.from {} out of bounds (nodes: {})", wire.from, n);
            assert!(wire.to < n, "wire.to {} out of bounds (nodes: {})", wire.to, n);
            assert!(wire.from != wire.to, "self-loop at node {}", wire.from);
        }
    }

    #[test]
    fn model_graph_wires_are_valid() {
        let gguf = llama_like_gguf(2);
        let prog = to_program_graph(&gguf);
        let model = prog.functions.last().unwrap();
        let n = model.nodes.len();
        for wire in &model.wires {
            assert!(wire.from < n, "wire.from {} out of bounds (nodes: {})", wire.from, n);
            assert!(wire.to < n, "wire.to {} out of bounds (nodes: {})", wire.to, n);
        }
    }

    #[test]
    fn program_graph_serializes_to_json() {
        let gguf = llama_like_gguf(1);
        let prog = to_program_graph(&gguf);
        let json = serde_json::to_string_pretty(&prog).unwrap();
        assert!(json.contains("\"block_0\""));
        assert!(json.contains("\"model\""));
        assert!(json.contains("matmul"));
        assert!(json.contains("attention"));
    }
}
