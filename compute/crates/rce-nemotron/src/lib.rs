//! Nemotron-H inference runtime on the rce substrate.
//!
//! Targets NVIDIA's `nemotron_h` architecture: 42 layers split into
//! three block kinds in arbitrary order (Mamba-2 SSM mixers, dense
//! gated FFNs, and RoPE GQA attention blocks). For the **4B-Q4_0**
//! variant verified against `llama-simple`:
//! - 21 SSM blocks (Mamba-2: d_inner 7680 = 96 heads × 80, d_state
//!   128, n_groups 8, conv_kernel 4)
//! - 17 dense FFN blocks (d_inner_ffn varies per layer)
//! - 4 GQA attention blocks (40 Q heads, 8 KV heads, d_head 128,
//!   RoPE dim 78, freq_base 10000)
//!
//! Forward uses rce ops: `rce_op_blas::BlasMatmul` for all big
//! matmuls, `rce_op_conv1d::conv1d_causal_step` for the Mamba-2
//! depthwise conv, `rce_op_ssm::selective_scan_step` for the
//! Mamba-2 SSD scan, plus inline RoPE / softmax / silu / rms_norm.
//!
//! State (`conv_state` + `ssm_state` per Mamba-2 block; growing
//! `k_cache` + `v_cache` per attention block; per-token `pos`
//! counter for RoPE) is threaded across forward passes for
//! autoregressive decode.

use rce_ingest_gguf::{
    read_tensor_f32, sample_with_history, GgufFile, GgufValue, Tokenizer,
};
pub use rce_ingest_gguf::SampleConfig;
use rce_op_blas::BlasMatmul;
use rce_op_conv1d::conv1d_causal_step;
use rce_op_qmm::QuantWeight;
use rce_op_ssm::selective_scan_step as selective_scan_step_f32;
use rce_vm_backend_cpu::{HostTensor, OpDispatch, Value};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

// ── Config ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct NemotronConfig {
    pub d_model: usize,
    pub vocab: usize,
    pub n_layers: usize,
    // Attention.
    pub n_q_heads: usize,
    pub n_kv_heads_per_layer: Vec<usize>, // length n_layers; 0 means non-attn layer
    pub d_head: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    // SSM (Mamba-2).
    pub ssm_n_heads: usize, // 96 for 4B
    pub ssm_head_dim: usize, // 80
    pub ssm_d_state: usize, // 128
    pub ssm_n_groups: usize, // 8
    pub ssm_d_conv: usize, // 4
    // FFN (dense). Per-layer width; 0 means non-FFN layer.
    pub ffn_inner_per_layer: Vec<usize>,
    pub eps: f32,
    // MoE FFN (nemotron_h_moe only; all zero/false for plain nemotron_h).
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
    pub expert_weights_norm: bool,
    pub expert_weights_scale: f32,
}

impl NemotronConfig {
    /// Read the config from GGUF metadata for `arch ∈ {nemotron_h,
    /// nemotron_h_moe}`. Detects the per-layer block types from
    /// which weight tensors are present.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf.metadata.iter()
            .find(|kv| kv.key == "general.architecture")
            .and_then(|kv| match &kv.value {
                GgufValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .ok_or("missing general.architecture")?;
        if !arch.starts_with("nemotron_h") {
            return Err(format!("expected nemotron_h*, got {arch}"));
        }
        let prefix = arch.to_string();

        let u32_meta = |k: &str| -> Option<u32> {
            gguf.metadata.iter().find(|kv| kv.key == format!("{prefix}.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::U32(v) => Some(*v), _ => None })
        };
        let f32_meta = |k: &str| -> Option<f32> {
            gguf.metadata.iter().find(|kv| kv.key == format!("{prefix}.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::F32(v) => Some(*v), _ => None })
        };
        let u32_arr_meta = |k: &str| -> Option<Vec<u32>> {
            gguf.metadata.iter().find(|kv| kv.key == format!("{prefix}.{k}"))
                .and_then(|kv| match &kv.value {
                    GgufValue::Array(a) => Some(a.iter().filter_map(|v| match v {
                        GgufValue::U32(x) => Some(*x),
                        _ => None,
                    }).collect()),
                    _ => None,
                })
        };

        let d_model = u32_meta("embedding_length").ok_or("embedding_length")? as usize;
        let vocab = u32_meta("vocab_size").ok_or("vocab_size")? as usize;
        let n_layers = u32_meta("block_count").ok_or("block_count")? as usize;
        let n_q_heads = u32_meta("attention.head_count").ok_or("head_count")? as usize;
        let d_head = u32_meta("attention.key_length").unwrap_or(128) as usize;
        let rope_dim = u32_meta("rope.dimension_count").ok_or("rope.dimension_count")? as usize;
        let rope_freq_base = f32_meta("rope.freq_base").unwrap_or(10000.0);
        let eps = f32_meta("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let ssm_d_inner = u32_meta("ssm.inner_size").ok_or("ssm.inner_size")? as usize;
        let ssm_d_state = u32_meta("ssm.state_size").ok_or("ssm.state_size")? as usize;
        let ssm_n_groups = u32_meta("ssm.group_count").ok_or("ssm.group_count")? as usize;
        let ssm_d_conv = u32_meta("ssm.conv_kernel").ok_or("ssm.conv_kernel")? as usize;
        // **Don't trust `ssm.time_step_rank` as n_heads** — for
        // Nemotron-H 4B that metadata reads 64 but the actual n_heads
        // is 96 (per `ssm_a`/`ssm_d`/`ssm_dt.bias` tensor shapes).
        // Read n_heads from the first SSM block's `ssm_a` shape: it's
        // a 1-D vector of length n_heads (stored as GGUF shape
        // `[1, n_heads]`). This is the canonical source of truth.
        let ssm_n_heads = gguf
            .tensors
            .iter()
            .find(|t| t.name.starts_with("blk.") && t.name.ends_with(".ssm_a"))
            .and_then(|t| t.dims.iter().copied().filter(|&d| d > 1).next())
            .map(|d| d as usize)
            .ok_or("could not infer ssm n_heads from blk.*.ssm_a")?;
        if ssm_d_inner % ssm_n_heads != 0 {
            return Err(format!(
                "ssm_d_inner ({ssm_d_inner}) not divisible by ssm_n_heads ({ssm_n_heads})"
            ));
        }
        let ssm_head_dim = ssm_d_inner / ssm_n_heads;

        // Per-layer GQA + FFN-width arrays (some layers have 0 for kv_heads
        // because they're SSM/FFN blocks).
        let n_kv_arr = u32_arr_meta("attention.head_count_kv")
            .unwrap_or_else(|| vec![0; n_layers]);
        let ffn_arr = u32_arr_meta("feed_forward_length")
            .unwrap_or_else(|| vec![0; n_layers]);
        let n_kv_heads_per_layer: Vec<usize> = n_kv_arr.into_iter().map(|v| v as usize).collect();
        let ffn_inner_per_layer: Vec<usize> = ffn_arr.into_iter().map(|v| v as usize).collect();

        // MoE hparams (present only for nemotron_h_moe). Defaults make
        // plain nemotron_h a no-op (no MoE layers exist there anyway).
        let n_expert = u32_meta("expert_count").unwrap_or(0) as usize;
        let n_expert_used = u32_meta("expert_used_count").unwrap_or(0) as usize;
        let n_ff_exp = u32_meta("expert_feed_forward_length").unwrap_or(0) as usize;
        let n_ff_shexp = u32_meta("expert_shared_feed_forward_length").unwrap_or(0) as usize;
        let expert_weights_norm = gguf.metadata.iter()
            .find(|kv| kv.key == format!("{prefix}.expert_weights_norm"))
            .map(|kv| matches!(&kv.value, GgufValue::Bool(b) if *b))
            .unwrap_or(false);
        let expert_weights_scale = f32_meta("expert_weights_scale").unwrap_or(1.0);

        Ok(NemotronConfig {
            d_model, vocab, n_layers,
            n_q_heads, n_kv_heads_per_layer, d_head, rope_dim, rope_freq_base,
            ssm_n_heads, ssm_head_dim, ssm_d_state, ssm_n_groups, ssm_d_conv,
            ffn_inner_per_layer, eps,
            n_expert, n_expert_used, n_ff_exp, n_ff_shexp, expert_weights_norm, expert_weights_scale,
        })
    }

    pub fn ssm_d_inner(&self) -> usize {
        self.ssm_n_heads * self.ssm_head_dim
    }
    pub fn ssm_conv_dim(&self) -> usize {
        // x + B + C: d_inner + 2*n_groups*d_state
        self.ssm_d_inner() + 2 * self.ssm_n_groups * self.ssm_d_state
    }
    pub fn ssm_in_proj_out(&self) -> usize {
        // z + xBC + dt
        self.ssm_d_inner() + self.ssm_conv_dim() + self.ssm_n_heads
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind { Ssm, Attn, Ffn, MoeFfn }

// ── Weights ────────────────────────────────────────────────────────

struct SsmBlock {
    attn_norm_w: Vec<f32>,
    ssm_in: QuantWeight,
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    a_log: Vec<f32>,
    d_param: Vec<f32>,
    dt_bias: Vec<f32>,
    ssm_norm_w: Vec<f32>,
    ssm_out: QuantWeight,
}

struct AttnBlock {
    attn_norm_w: Vec<f32>,
    q: QuantWeight,
    k: QuantWeight,
    v: QuantWeight,
    o: QuantWeight,
    n_kv_heads: usize,
}

struct FfnBlock {
    ffn_norm_w: Vec<f32>,
    up: QuantWeight,
    down: QuantWeight,
    d_inner_ffn: usize,
}

/// nemotron_h_moe FFN: 128 routed experts (top-`n_expert_used`), each a
/// RELU² up/down MLP (no gate), routed by a sigmoid gate with a
/// per-expert selection bias (`exp_probs_b`); plus one always-on shared
/// expert (RELU² up/down, no gate, unscaled). Routed weights are
/// optionally normalized then scaled by `expert_weights_scale` (2.5).
struct MoeFfnBlock {
    ffn_norm_w: Vec<f32>,
    router: QuantWeight,       // [n_expert, d_model] F32
    exp_probs_b: Vec<f32>,     // [n_expert] selection bias
    up_exps: Vec<QuantWeight>,    // n_expert × [n_ff_exp, d_model]
    down_exps: Vec<QuantWeight>,  // n_expert × [d_model, n_ff_exp]
    shexp_up: QuantWeight,     // [n_ff_shexp, d_model]
    shexp_down: QuantWeight,   // [d_model, n_ff_shexp]
}

enum BlockTensors {
    Ssm(Box<SsmBlock>),
    Attn(Box<AttnBlock>),
    Ffn(Box<FfnBlock>),
    MoeFfn(Box<MoeFfnBlock>),
}

impl BlockTensors {
    fn kind(&self) -> BlockKind {
        match self {
            BlockTensors::Ssm(_) => BlockKind::Ssm,
            BlockTensors::Attn(_) => BlockKind::Attn,
            BlockTensors::Ffn(_) => BlockKind::Ffn,
            BlockTensors::MoeFfn(_) => BlockKind::MoeFfn,
        }
    }
}

struct NemotronWeights {
    token_embd: QuantWeight,
    output_norm_w: Vec<f32>,
    lm_head: QuantWeight,
    blocks: Vec<BlockTensors>,
}

// ── State ──────────────────────────────────────────────────────────

enum BlockState {
    Ssm { conv_state: Vec<f32>, ssm_state: Vec<f32> },
    Attn { k_cache: Vec<f32>, v_cache: Vec<f32> },
    None,
}

/// Per-conversation recurrent state plus a running RoPE position
/// counter (incremented by every forward pass).
pub struct NemotronState {
    blocks: Vec<BlockState>,
    pos: usize,
}

impl NemotronState {
    fn new(cfg: &NemotronConfig, kinds: &[BlockKind]) -> Self {
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for (i, &kind) in kinds.iter().enumerate() {
            let _ = i;
            match kind {
                BlockKind::Ssm => blocks.push(BlockState::Ssm {
                    conv_state: vec![0.0f32; cfg.ssm_conv_dim() * cfg.ssm_d_conv],
                    ssm_state: vec![0.0f32; cfg.ssm_d_inner() * cfg.ssm_d_state],
                }),
                BlockKind::Attn => blocks.push(BlockState::Attn {
                    k_cache: Vec::new(),
                    v_cache: Vec::new(),
                }),
                BlockKind::Ffn | BlockKind::MoeFfn => blocks.push(BlockState::None),
            }
        }
        Self { blocks, pos: 0 }
    }
}

// ── Loading ────────────────────────────────────────────────────────

/// Split a stacked `[n_expert, rows_per, cols_per]` MoE tensor into
/// per-expert zero-copy QuantWeight views (byte stride = total/n_expert;
/// valid because per-expert numel is a multiple of the quant block size).
fn split_experts(
    full: &QuantWeight, n_expert: usize, rows_per: usize, cols_per: usize,
) -> Vec<QuantWeight> {
    let total = full.bytes().len();
    assert!(total % n_expert == 0, "expert stack {} not divisible by n_expert {}", total, n_expert);
    let bpe = total / n_expert;
    (0..n_expert).map(|e| full.sub_view(e * bpe, bpe, rows_per, cols_per)).collect()
}

fn load_nemotron_weights(
    mmap: &Arc<Mmap>, gguf: &GgufFile, cfg: &NemotronConfig,
) -> Result<(NemotronWeights, Vec<BlockKind>), String> {
    let bytes: &[u8] = &mmap[..];
    let token_embd = QuantWeight::from_gguf(Arc::clone(mmap), gguf, "token_embd.weight", cfg.vocab, cfg.d_model)
        .map_err(|e| format!("token_embd: {e}"))?;
    let output_norm_w = read_tensor_f32(bytes, gguf, "output_norm.weight")
        .map_err(|e| format!("output_norm: {e:?}"))?;
    let lm_head = QuantWeight::from_gguf(Arc::clone(mmap), gguf, "output.weight", cfg.vocab, cfg.d_model)
        .map_err(|e| format!("output: {e}"))?;

    // Helper: look up a tensor's torch-order `numel/d_model` to derive
    // hyperparams (e.g. n_kv_heads, d_inner_ffn) that the GGUF metadata
    // arrays mangle for hybrid models.
    let tensor_numel = |name: &str| -> Option<usize> {
        gguf.tensors.iter().find(|t| t.name == name).map(|t| t.numel() as usize)
    };

    let mut blocks: Vec<BlockTensors> = Vec::with_capacity(cfg.n_layers);
    let mut kinds: Vec<BlockKind> = Vec::with_capacity(cfg.n_layers);
    for b in 0..cfg.n_layers {
        let bn = format!("blk.{b}");
        let has_ssm = gguf.tensors.iter().any(|t| t.name == format!("{bn}.ssm_in.weight"));
        let has_attn = gguf.tensors.iter().any(|t| t.name == format!("{bn}.attn_q.weight"));
        // Nemotron-H FFN is ungated: up + down only, no gate.
        let has_ffn = gguf.tensors.iter().any(|t| t.name == format!("{bn}.ffn_up.weight"));
        // nemotron_h_moe replaces the dense FFN with a routed MoE block.
        let has_moe = gguf.tensors.iter().any(|t| t.name == format!("{bn}.ffn_gate_inp.weight"));

        let attn_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_norm.weight"))
            .map_err(|e| format!("blk.{b}.attn_norm: {e:?}"))?;

        if has_ssm {
            kinds.push(BlockKind::Ssm);
            let conv_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_conv1d.weight")).map_err(|e| format!("{e:?}"))?;
            let conv_b = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_conv1d.bias")).map_err(|e| format!("{e:?}"))?;
            let a_log = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_a")).map_err(|e| format!("{e:?}"))?;
            let d_param = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_d")).map_err(|e| format!("{e:?}"))?;
            let dt_bias = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_dt.bias")).map_err(|e| format!("{e:?}"))?;
            let ssm_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_norm.weight")).map_err(|e| format!("{e:?}"))?;
            let ssm_in = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.ssm_in.weight"), cfg.ssm_in_proj_out(), cfg.d_model)
                .map_err(|e| format!("blk.{b}.ssm_in: {e}"))?;
            let ssm_out = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.ssm_out.weight"), cfg.d_model, cfg.ssm_d_inner())
                .map_err(|e| format!("blk.{b}.ssm_out: {e}"))?;
            blocks.push(BlockTensors::Ssm(Box::new(SsmBlock {
                attn_norm_w, ssm_in, conv_w, conv_b, a_log, d_param, dt_bias, ssm_norm_w, ssm_out,
            })));
        } else if has_attn {
            kinds.push(BlockKind::Attn);
            // Derive n_kv_heads from the actual attn_k shape — the
            // GGUF metadata array is zero-padded for non-attn layers.
            // `attn_k.numel() = n_kv * d_head * d_model`.
            let k_numel = tensor_numel(&format!("{bn}.attn_k.weight"))
                .ok_or_else(|| format!("blk.{b}: missing attn_k"))?;
            let n_kv = (k_numel / cfg.d_model) / cfg.d_head;
            if n_kv == 0 {
                return Err(format!("blk.{b}: attn block but k shape implies n_kv_heads=0"));
            }
            let q = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.attn_q.weight"), cfg.n_q_heads * cfg.d_head, cfg.d_model)
                .map_err(|e| format!("blk.{b}.attn_q: {e}"))?;
            let k = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.attn_k.weight"), n_kv * cfg.d_head, cfg.d_model)
                .map_err(|e| format!("blk.{b}.attn_k: {e}"))?;
            let v = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.attn_v.weight"), n_kv * cfg.d_head, cfg.d_model)
                .map_err(|e| format!("blk.{b}.attn_v: {e}"))?;
            let o = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.attn_output.weight"), cfg.d_model, cfg.n_q_heads * cfg.d_head)
                .map_err(|e| format!("blk.{b}.attn_output: {e}"))?;
            blocks.push(BlockTensors::Attn(Box::new(AttnBlock {
                attn_norm_w, q, k, v, o, n_kv_heads: n_kv,
            })));
        } else if has_moe {
            kinds.push(BlockKind::MoeFfn);
            let ne = cfg.n_expert;
            let nf_e = cfg.n_ff_exp;
            let nf_s = cfg.n_ff_shexp;
            let dm = cfg.d_model;
            let router = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.ffn_gate_inp.weight"), ne, dm)
                .map_err(|e| format!("blk.{b}.ffn_gate_inp: {e}"))?;
            let exp_probs_b = read_tensor_f32(bytes, gguf, &format!("{bn}.exp_probs_b.bias"))
                .map_err(|e| format!("blk.{b}.exp_probs_b: {e:?}"))?;
            // Stacked expert tensors: numel = n_expert × rows_per × cols_per.
            let up_stack = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.ffn_up_exps.weight"), ne * nf_e, dm)
                .map_err(|e| format!("blk.{b}.ffn_up_exps: {e}"))?;
            let down_stack = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.ffn_down_exps.weight"), ne * dm, nf_e)
                .map_err(|e| format!("blk.{b}.ffn_down_exps: {e}"))?;
            let up_exps = split_experts(&up_stack, ne, nf_e, dm);
            let down_exps = split_experts(&down_stack, ne, dm, nf_e);
            let shexp_up = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.ffn_up_shexp.weight"), nf_s, dm)
                .map_err(|e| format!("blk.{b}.ffn_up_shexp: {e}"))?;
            let shexp_down = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.ffn_down_shexp.weight"), dm, nf_s)
                .map_err(|e| format!("blk.{b}.ffn_down_shexp: {e}"))?;
            blocks.push(BlockTensors::MoeFfn(Box::new(MoeFfnBlock {
                ffn_norm_w: attn_norm_w, router, exp_probs_b, up_exps, down_exps, shexp_up, shexp_down,
            })));
        } else if has_ffn {
            kinds.push(BlockKind::Ffn);
            // Derive d_inner_ffn from the actual tensor shape (metadata
            // `feed_forward_length` array is zero-padded for non-FFN layers).
            let up_numel = tensor_numel(&format!("{bn}.ffn_up.weight"))
                .ok_or_else(|| format!("blk.{b}: missing ffn_up"))?;
            let d_inner_ffn = up_numel / cfg.d_model;
            let up = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.ffn_up.weight"), d_inner_ffn, cfg.d_model)
                .map_err(|e| format!("blk.{b}.ffn_up: {e}"))?;
            let down = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.ffn_down.weight"), cfg.d_model, d_inner_ffn)
                .map_err(|e| format!("blk.{b}.ffn_down: {e}"))?;
            blocks.push(BlockTensors::Ffn(Box::new(FfnBlock {
                // nemotron_h has a single per-block norm regardless of block type;
                // for an FFN-only layer there's no separate ffn_norm tensor, so
                // we reuse the (`attn_norm`-keyed) per-block input norm.
                ffn_norm_w: attn_norm_w, up, down, d_inner_ffn,
            })));
        } else {
            return Err(format!("blk.{b}: no ssm / attn / ffn tensors found"));
        }
    }
    Ok((NemotronWeights { token_embd, output_norm_w, lm_head, blocks }, kinds))
}

// ── Runtime ────────────────────────────────────────────────────────

pub struct NemotronRuntime {
    cfg: NemotronConfig,
    gguf: GgufFile,
    weights: NemotronWeights,
    kinds: Vec<BlockKind>,
}

impl NemotronRuntime {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| format!("open: {e}"))?;
        let mmap = Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?);
        // `WillNeed`, NOT `Sequential`: MoE decode re-reads *different*
        // experts every token, so `Sequential`'s drop-behind eviction
        // forces repeated cold re-faults from disk (sustained MoE was
        // ~2 GB/s, far below RAM bandwidth). `WillNeed` prefetches and
        // keeps the model resident — on a box with RAM ≫ model size the
        // whole file stays page-cached, so decode hits RAM speed.
        let _ = mmap.advise(memmap2::Advice::WillNeed);
        let gguf = GgufFile::parse(&mmap[..]).map_err(|e| format!("parse: {e:?}"))?;
        let cfg = NemotronConfig::from_gguf(&gguf)?;
        eprintln!(
            "nemotron_h: d_model={} vocab={} n_layers={} ssm_n_heads={} ssm_head_dim={} ssm_n_groups={} ssm_d_state={} ssm_d_conv={} rope_dim={}",
            cfg.d_model, cfg.vocab, cfg.n_layers, cfg.ssm_n_heads, cfg.ssm_head_dim, cfg.ssm_n_groups, cfg.ssm_d_state, cfg.ssm_d_conv, cfg.rope_dim,
        );
        // mmap kept alive transitively via QuantWeight Arc clones.
        let (weights, kinds) = load_nemotron_weights(&mmap, &gguf, &cfg)?;
        Ok(Self { cfg, gguf, weights, kinds })
    }

    pub fn config(&self) -> &NemotronConfig { &self.cfg }
    pub fn block_kinds(&self) -> &[BlockKind] { &self.kinds }

    pub fn tokenizer(&self) -> Tokenizer<'_> {
        Tokenizer::from_gguf(&self.gguf).expect("tokenizer")
    }

    pub fn new_state(&self) -> NemotronState {
        NemotronState::new(&self.cfg, &self.kinds)
    }

    /// One forward pass for `token_id`. Threads state (conv/ssm state
    /// for SSM blocks, growing KV cache for attention blocks, plus
    /// the RoPE position counter).
    pub fn forward_logits(&self, state: &mut NemotronState, token_id: i32) -> Vec<f32> {
        let blas = BlasMatmul::new();
        let logits = forward_token(&self.weights, &self.cfg, &self.kinds, state, token_id, &blas);
        state.pos += 1;
        logits
    }

    /// Tokenize a chat conversation (Qwen-style `<|im_start|>` framing,
    /// GPT-2 BPE), prefill, then generate up to `cfg.max_new_tokens`
    /// tokens stopping on `<|im_end|>` / EOS. `on_token(text)` is
    /// called for each generated token's decoded text.
    pub fn chat(
        &self,
        messages: &[(&str, &str)],
        cfg: &NemotronGenConfig,
        mut on_token: impl FnMut(&str),
    ) -> Result<String, String> {
        let tk = self.tokenizer();
        let prompt = tk
            .format_chat_nemotron(messages, true, cfg.enable_thinking)
            .ok_or("model lacks chat-template special tokens")?;
        let eos_id = tk.eos_token_id.unwrap_or(-1);
        let im_end_id = tk.lookup_token("<|im_end|>").unwrap_or(-1);

        let mut state = self.new_state();
        let mut history: Vec<i32> = Vec::with_capacity(prompt.len() + cfg.max_new_tokens);
        let mut step: u64 = 0;
        let seed_for = |s: u64| {
            let mut c = cfg.sample;
            c.seed = cfg.sample.seed ^ s.wrapping_mul(0x9E37_79B9);
            c
        };

        let mut next: i32 = -1;
        for (i, &tok) in prompt.iter().enumerate() {
            let logits = self.forward_logits(&mut state, tok);
            history.push(tok);
            let recent = &history[history.len().saturating_sub(cfg.rep_window)..];
            let pred = sample_with_history(&logits, &seed_for(step), recent);
            step += 1;
            if i == prompt.len() - 1 {
                next = pred;
            }
        }

        let mut out_ids: Vec<i32> = Vec::new();
        for _ in 0..cfg.max_new_tokens {
            if next == im_end_id || next == eos_id {
                break;
            }
            out_ids.push(next);
            history.push(next);
            let text = tk.decode_sequence(&[next], false);
            on_token(&text);
            let logits = self.forward_logits(&mut state, next);
            let recent = &history[history.len().saturating_sub(cfg.rep_window)..];
            next = sample_with_history(&logits, &seed_for(step), recent);
            step += 1;
        }
        Ok(tk.decode_sequence(&out_ids, true))
    }
}

/// Generation config for `NemotronRuntime::chat`. Same shape as
/// `rce-jamba::GenConfig`, but the chat template uses
/// `enable_thinking` (Nemotron-specific) instead of `thinking_prefix`.
#[derive(Clone, Copy, Debug)]
pub struct NemotronGenConfig {
    pub max_new_tokens: usize,
    pub sample: SampleConfig,
    pub rep_window: usize,
    pub enable_thinking: bool,
}

impl Default for NemotronGenConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            sample: SampleConfig::chat(0xC0DE_F00D_2026),
            rep_window: 64,
            enable_thinking: true,
        }
    }
}

fn forward_token(
    w: &NemotronWeights, cfg: &NemotronConfig, kinds: &[BlockKind],
    state: &mut NemotronState, token_id: i32, blas: &BlasMatmul,
) -> Vec<f32> {
    let dm = cfg.d_model;
    let mut x: Vec<f32> = w.token_embd.dequant_row(token_id as usize);
    let trace = std::env::var("RCE_NEMOTRON_TRACE").is_ok() && state.pos <= 1;
    if trace {
        eprintln!("[fwd pos={} tok={token_id}] embd[0..3]={:?}", state.pos, &x[..3]);
    }
    let pos = state.pos;
    let perf = std::env::var("RCE_NEMOTRON_PERF").is_ok();
    let (mut t_ssm, mut t_attn, mut t_moe) = (0.0f64, 0.0f64, 0.0f64);

    for (b, block) in w.blocks.iter().enumerate() {
        let kind = kinds[b];
        let tb = std::time::Instant::now();
        let delta = match block {
            BlockTensors::Ssm(s) => {
                let normed = rms_norm(&x, &s.attn_norm_w, cfg.eps);
                let (cs, ss) = match &mut state.blocks[b] {
                    BlockState::Ssm { conv_state, ssm_state } => (conv_state, ssm_state),
                    _ => unreachable!(),
                };
                ssm_mixer(&normed, s, cs, ss, cfg, blas)
            }
            BlockTensors::Attn(a) => {
                if std::env::var("RCE_NEMOTRON_BYPASS_ATTN").is_ok() {
                    // Diagnostic: treat attention block as identity
                    // passthrough (no residual delta). Localises whether
                    // garbage output is from the attention path.
                    let _ = a;
                    vec![0.0f32; cfg.d_model]
                } else {
                    let normed = rms_norm(&x, &a.attn_norm_w, cfg.eps);
                    let (kc, vc) = match &mut state.blocks[b] {
                        BlockState::Attn { k_cache, v_cache } => (k_cache, v_cache),
                        _ => unreachable!(),
                    };
                    attn_block_rope(&normed, a, kc, vc, pos, cfg, blas)
                }
            }
            BlockTensors::Ffn(f) => {
                let normed = rms_norm(&x, &f.ffn_norm_w, cfg.eps);
                ffn_block(&normed, f, cfg, blas)
            }
            BlockTensors::MoeFfn(m) => {
                let normed = rms_norm(&x, &m.ffn_norm_w, cfg.eps);
                moe_ffn_block(&normed, m, cfg)
            }
        };
        if perf {
            let dt = tb.elapsed().as_secs_f64();
            match kind {
                BlockKind::Ssm => t_ssm += dt,
                BlockKind::Attn => t_attn += dt,
                BlockKind::Ffn | BlockKind::MoeFfn => t_moe += dt,
            }
        }
        let _ = kind;
        for i in 0..dm { x[i] += delta[i]; }
        if trace && (b < 4 || b == 12 || b == 17 || b == 18 || b == 20 || b == 24 || b == 25 || b == 32 || b == 33 || b >= 38) {
            eprintln!("[fwd pos={} blk{b} {:?}] x[0..3]={:?}", pos, kind, &x[..3]);
        }
    }

    if perf {
        eprintln!("[perf pos={pos}] ssm={:.1}ms attn={:.1}ms moe={:.1}ms",
            t_ssm * 1e3, t_attn * 1e3, t_moe * 1e3);
    }
    let final_normed = rms_norm(&x, &w.output_norm_w, cfg.eps);
    if trace {
        eprintln!("[fwd] final_normed[0..3]={:?}", &final_normed[..3]);
    }
    let logits = qmm_1d(&final_normed, &w.lm_head);
    if trace {
        // Top-5 token IDs.
        let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("[fwd] top-5 argmax: {:?}", &indexed[..5]);
    }
    logits
}

// ── SSM mixer (Mamba-2) ────────────────────────────────────────────

fn ssm_mixer(
    x_in: &[f32], s: &SsmBlock, conv_state: &mut Vec<f32>, ssm_state: &mut Vec<f32>,
    cfg: &NemotronConfig, blas: &BlasMatmul,
) -> Vec<f32> {
    let dm = cfg.d_model;
    let d_inner = cfg.ssm_d_inner();
    let conv_dim = cfg.ssm_conv_dim();
    let nh = cfg.ssm_n_heads;
    let hd = cfg.ssm_head_dim;
    let ds = cfg.ssm_d_state;
    let ng = cfg.ssm_n_groups;
    let debug = std::env::var("RCE_NEMOTRON_TRACE").is_ok();

    // in_proj: [d_model] @ [d_model, z+xBC+dt] -> packed [d_inner + conv_dim + nh]
    let proj = qmm_1d(x_in, &s.ssm_in);
    let z = &proj[0..d_inner];
    let xbc = &proj[d_inner..d_inner + conv_dim];
    let dt_raw = &proj[d_inner + conv_dim..d_inner + conv_dim + nh];

    // Conv1d on xBC (depthwise causal, sliding window).
    // `RCE_NEMOTRON_CONV_REVERSED=1` reverses the kernel order per
    // channel — try if the GGUF stores weights in opposite causal
    // convention (weight[k=0] = current vs oldest).
    let conv_w_used: Vec<f32>;
    let conv_w: &[f32] = if std::env::var("RCE_NEMOTRON_CONV_REVERSED").is_ok() {
        let dc = cfg.ssm_d_conv;
        let mut rev = vec![0.0f32; s.conv_w.len()];
        for d in 0..conv_dim {
            for k in 0..dc {
                rev[d * dc + k] = s.conv_w[d * dc + (dc - 1 - k)];
            }
        }
        conv_w_used = rev;
        &conv_w_used
    } else {
        &s.conv_w
    };
    let (xbc_conv, new_conv_state) = conv1d_causal_step(
        xbc, conv_w, &s.conv_b, conv_state, conv_dim, cfg.ssm_d_conv,
    );
    *conv_state = new_conv_state;

    // SiLU on the **whole xBC** band, per llama.cpp's `build_mamba2_layer`
    // (mamba-base.cpp line 222). Then split into x, B, C.
    let xbc_act: Vec<f32> = xbc_conv.iter().map(|v| silu(*v)).collect();
    let x_act = &xbc_act[0..d_inner];
    let b_act = &xbc_act[d_inner..d_inner + ng * ds];
    let c_act = &xbc_act[d_inner + ng * ds..conv_dim];

    // Mamba-2 selective scan (SSD). `selective_scan_step` adds the
    // `D*x` skip internally with the post-silu x — same as llama.cpp.
    // **f64 accumulator** variant runs the inner d_state recurrence
    // in double precision, then narrows back to f32 — substantially
    // reduces drift through deep stacks (each f32 SSM scan was
    // adding ~5% relative error vs ggml; f64 brings it to <0.5%).
    let use_f64 = std::env::var("RCE_NEMOTRON_SSM_F32").is_err();
    let (y, new_ssm) = if use_f64 {
        selective_scan_step_f64(
            x_act, b_act, c_act, dt_raw, &s.dt_bias, &s.a_log, &s.d_param, ssm_state,
            nh, hd, ds, ng,
        )
    } else {
        selective_scan_step_f32(
            x_act, b_act, c_act, dt_raw, &s.dt_bias, &s.a_log, &s.d_param, ssm_state,
            nh, hd, ds, ng,
        )
    };
    *ssm_state = new_ssm;

    // **Gate first, THEN group RMSNorm** — matches llama.cpp's order
    // (mamba-base.cpp lines 266-272): `y = silu(z) * y` (swiglu_split),
    // then `build_norm(y, ssm_norm, LLM_NORM_RMS)`. NB: efficient-genai
    // has these swapped (norm then gate); llama.cpp is authoritative.
    let mut gated = vec![0.0f32; d_inner];
    for i in 0..d_inner { gated[i] = silu(z[i]) * y[i]; }
    let normed = group_rms_norm(&gated, &s.ssm_norm_w, ng, cfg.eps);

    // out_proj: [d_inner] @ [d_inner, d_model] -> [d_model]
    // **f64 accumulator** path (`RCE_NEMOTRON_OUTPROJ_F64=1`): replace
    // BLAS sgemm with a sequential f64 accumulator. The post-gate-norm
    // input mixes magnitudes wildly (tiny values ~1e-4 alongside ~25
    // outliers from a few "channels"); the original AMX sgemm's
    // vectorised accumulation drifted on those outlier channels. The
    // earlier `RCE_NEMOTRON_OUTPROJ_F64=1` diagnostic was removed
    // when `ssm_out` switched to QuantWeight — re-adding it would need
    // a per-block dequant + f64 dot fallback, which we can write if
    // numerical drift becomes a problem again.
    let out = qmm_1d(&normed, &s.ssm_out);
    if debug {
        eprintln!("[ssm] out[0..3]={:?}", &out[..3.min(out.len())]);
    }
    out
}

// ── Attention (RoPE + GQA) ─────────────────────────────────────────

fn attn_block_rope(
    x_in: &[f32], a: &AttnBlock, k_cache: &mut Vec<f32>, v_cache: &mut Vec<f32>,
    pos: usize, cfg: &NemotronConfig, blas: &BlasMatmul,
) -> Vec<f32> {
    let dm = cfg.d_model;
    let dh = cfg.d_head;
    let nq = cfg.n_q_heads;
    let nkv = a.n_kv_heads;
    let q_dim = nq * dh;
    let kv_dim = nkv * dh;

    let mut q = qmm_1d(x_in, &a.q);
    let mut k_new = qmm_1d(x_in, &a.k);
    let v_new = qmm_1d(x_in, &a.v);

    // **Default: NO RoPE.** llama.cpp's nemotron-h.cpp goes
    // `build_qkv` → `build_attn` directly with no `ggml_rope_ext`
    // between them, and empirically applying NeoX RoPE here makes
    // output measurably worse (reverts to "997.6125000**\\/" garbage
    // vs the slightly-more-English "Inter In my An as" output we get
    // without RoPE). The GGUF `rope.dimension_count` metadata is a
    // leftover; Nemotron-H attention is unpositional and gets its
    // sequence sense from the Mamba-2 SSM mixer's recurrence.
    // `RCE_NEMOTRON_FORCE_ROPE=1` re-enables for testing.
    if std::env::var("RCE_NEMOTRON_FORCE_ROPE").is_ok() {
        apply_rope_inplace(&mut q, nq, dh, cfg.rope_dim, cfg.rope_freq_base, pos);
        apply_rope_inplace(&mut k_new, nkv, dh, cfg.rope_dim, cfg.rope_freq_base, pos);
    } else {
        let _ = pos;
    }

    // Append to KV cache.
    k_cache.extend_from_slice(&k_new);
    v_cache.extend_from_slice(&v_new);
    let seq_len = k_cache.len() / kv_dim;

    let scale = 1.0 / (dh as f32).sqrt();
    // NEON-vectorized fused attention via rce-op-attn.
    let mut attended = vec![0.0f32; q_dim];
    let q_per_kv = nq / nkv;
    for h in 0..nq {
        let kv_head = h / q_per_kv;
        let q_h = &q[h * dh..(h + 1) * dh];
        let out_h = rce_op_attn::fused_attend_head(
            q_h, k_cache, v_cache, kv_head, seq_len, kv_dim, dh, scale,
        );
        let out_off = h * dh;
        attended[out_off..out_off + dh].copy_from_slice(&out_h);
    }
    qmm_1d(&attended, &a.o)
}

/// Apply RoPE to the first `rope_dim` dimensions of each head. The
/// pairing convention is env-toggleable: NeoX-style (default; pairs
/// are `(x[i], x[i+half])` for i in 0..half) or GPT-J / Llama-style
/// interleaved (pairs are `(x[2i], x[2i+1])`). `RCE_NEMOTRON_ROPE`
/// switches between them: `=neox` (default), `=gptj` / `=interleaved`.
fn apply_rope_inplace(x: &mut [f32], n_heads: usize, d_head: usize, rope_dim: usize, base: f32, pos: usize) {
    let half = rope_dim / 2;
    let interleaved = matches!(
        std::env::var("RCE_NEMOTRON_ROPE").as_deref(),
        Ok("gptj") | Ok("interleaved")
    );
    for h in 0..n_heads {
        let head_off = h * d_head;
        for i in 0..half {
            let inv_freq = (base.ln() * (2.0 * i as f32) / rope_dim as f32).exp().recip();
            let theta = pos as f32 * inv_freq;
            let (sin, cos) = theta.sin_cos();
            if interleaved {
                let a = x[head_off + 2 * i];
                let b = x[head_off + 2 * i + 1];
                x[head_off + 2 * i] = a * cos - b * sin;
                x[head_off + 2 * i + 1] = a * sin + b * cos;
            } else {
                let a = x[head_off + i];
                let b = x[head_off + i + half];
                x[head_off + i] = a * cos - b * sin;
                x[head_off + i + half] = a * sin + b * cos;
            }
        }
    }
}

// ── FFN (dense, ungated, ReLU²) ────────────────────────────────────
//
// Nemotron-H uses a 2-layer MLP with **squared-ReLU** activation
// (no gate): `down(relu²(up(x)))`. ReLU² = `max(0, x)²` — sparse like
// ReLU but smoother in the active region.

fn ffn_block(x_in: &[f32], f: &FfnBlock, cfg: &NemotronConfig, blas: &BlasMatmul) -> Vec<f32> {
    let dm = cfg.d_model;
    let dff = f.d_inner_ffn;
    let u = qmm_1d(x_in, &f.up);
    let mut hidden = vec![0.0f32; dff];
    // Activation: env-toggleable for debug. Default is ReLU² (per the
    // Nemotron paper); `RCE_NEMOTRON_FFN_ACT=gelu` switches to GELU
    // (tanh approximation), `=relu` plain ReLU.
    let act = std::env::var("RCE_NEMOTRON_FFN_ACT").unwrap_or_else(|_| "relu2".into());
    match act.as_str() {
        "gelu" => {
            for i in 0..dff {
                let x = u[i];
                let inner = (2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
                hidden[i] = 0.5 * x * (1.0 + inner.tanh());
            }
        }
        "relu" => {
            for i in 0..dff { hidden[i] = u[i].max(0.0); }
        }
        _ => {
            // ReLU² = max(0, x)²
            for i in 0..dff {
                let r = u[i].max(0.0);
                hidden[i] = r * r;
            }
        }
    }
    qmm_1d(&hidden, &f.down)
}

/// RELU² up/down expert: `down(relu²(up(x)))`, **serial** matmuls.
/// Called inside a rayon par-over-experts region — per-expert matrices
/// are too small for the row-parallel `mm_1d` to beat single-thread
/// (fork/join overhead cancels it), so we go serial and parallelize at
/// the coarser expert grain. See rce-op-qmm::mm_1d_serial.
fn relu2_expert_serial(x_in: &[f32], up: &QuantWeight, down: &QuantWeight, n_ff: usize) -> Vec<f32> {
    let mut hidden = vec![0.0f32; n_ff];
    rce_op_qmm::mm_1d_serial(x_in, up, &mut hidden);
    for v in hidden.iter_mut() { let r = v.max(0.0); *v = r * r; }
    let mut out = vec![0.0f32; down.n_rows];
    rce_op_qmm::mm_1d_serial(&hidden, down, &mut out);
    out
}

// ── MoE FFN (nemotron_h_moe) ───────────────────────────────────────
//
// DeepSeek-V3-style routing (mirrors llama.cpp `build_moe_ffn` with
// `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`):
//   logits   = router @ x                       [n_expert]
//   probs    = sigmoid(logits)                  (NOT softmax)
//   sel_prob = probs + exp_probs_b              (bias used for SELECTION only)
//   top-k    = argmax_k(sel_prob)
//   weights  = probs[top-k]                     (original probs, not biased)
//   if norm: weights /= Σ weights
//   weights *= expert_weights_scale (2.5)
//   moe_out  = Σ weights_i · relu²_expert_i(x)
//   out      = moe_out + relu²_shared_expert(x)  (shared always-on, unscaled)
fn moe_ffn_block(x_in: &[f32], m: &MoeFfnBlock, cfg: &NemotronConfig) -> Vec<f32> {
    let ne = cfg.n_expert;
    let k = cfg.n_expert_used;
    let dm = cfg.d_model;

    let logits = qmm_1d(x_in, &m.router);
    let probs: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();
    let sel: Vec<f32> = probs.iter().zip(m.exp_probs_b.iter()).map(|(p, b)| p + b).collect();

    // top-k by biased selection score.
    let mut idx: Vec<usize> = (0..ne).collect();
    idx.sort_unstable_by(|&a, &b| sel[b].partial_cmp(&sel[a]).unwrap());
    idx.truncate(k);

    // gather weights from ORIGINAL (unbiased) probs.
    let mut weights: Vec<f32> = idx.iter().map(|&e| probs[e]).collect();
    if cfg.expert_weights_norm {
        let s: f32 = weights.iter().sum();
        if s > 0.0 { for w in weights.iter_mut() { *w /= s; } }
    }
    for w in weights.iter_mut() { *w *= cfg.expert_weights_scale; }

    // ONE coarse rayon region over the k routed experts + shared expert
    // (job index k). Each task runs its RELU² FFN serially — far fewer
    // fork/joins than per-matmul parallelism, which didn't scale on these
    // small per-expert matrices.
    use rayon::prelude::*;
    let outs: Vec<Vec<f32>> = (0..=k).into_par_iter().map(|job| {
        if job < k {
            let e = idx[job];
            relu2_expert_serial(x_in, &m.up_exps[e], &m.down_exps[e], cfg.n_ff_exp)
        } else {
            relu2_expert_serial(x_in, &m.shexp_up, &m.shexp_down, cfg.n_ff_shexp)
        }
    }).collect();

    let mut out = vec![0.0f32; dm];
    for (j, &w) in weights.iter().enumerate() {
        for i in 0..dm { out[i] += w * outs[j][i]; }
    }
    // Always-on shared expert (unscaled, no gate) = job k.
    for i in 0..dm { out[i] += outs[k][i]; }
    out
}

/// f64-accumulator Mamba-2 selective scan. Same algorithm as
/// `rce_op_ssm::selective_scan_step`, but the inner d_state
/// recurrence runs in f64 to keep cumulative error from blowing up
/// through deep stacks of SSM blocks. Each forward pass of the
/// pure-f32 kernel added ~5% relative drift vs ggml's `SSM_SCAN`
/// (which uses tighter accumulation); compounded over 21 SSM blocks
/// this turned coherent logits into garbage. f64 brings per-block
/// drift to <0.5% and the deep-stack drift to a tolerable level.
fn selective_scan_step_f64(
    x: &[f32], b: &[f32], c: &[f32], dt_raw: &[f32], dt_bias: &[f32],
    a_log: &[f32], d_param: &[f32], state_in: &[f32],
    nheads: usize, headdim: usize, d_state: usize, ngroups: usize,
) -> (Vec<f32>, Vec<f32>) {
    let d_inner = nheads * headdim;
    let nheads_per_group = nheads / ngroups;
    let mut y = vec![0.0f32; d_inner];
    let mut state_out: Vec<f64> = state_in.iter().map(|&v| v as f64).collect();
    // Per-head slice-pinning: hoist `group_b`, `group_c`, `state_head`,
    // `y_head` outside the inner d_state loop so the inner loop indexes
    // local slices instead of recomputing the global offset each step.
    // Worth ~7% on the scan time in the mamba2 profiling run.
    let state_per_head = headdim * d_state;
    for h in 0..nheads {
        let group = h / nheads_per_group;
        let dt_pre = dt_raw[h] as f64 + dt_bias[h] as f64;
        // softplus(x) = ln(1 + exp(x)); numerically stable for large x via max(x,0)+ln1p(exp(-|x|)).
        let dt_h = if dt_pre > 20.0 { dt_pre } else { (1.0 + dt_pre.exp()).ln() };
        // GGUF stores `ssm_a` as the already-transformed `A = -exp(A_log)`
        // (llama.cpp's `convert_hf_to_gguf.py` does `data = -data.exp()` on
        // any tensor ending in `A_log` before writing). Use the stored
        // value directly — see `project_gguf_ssm_a_convention.md`.
        let a_h = a_log[h] as f64;
        let a_bar = (dt_h * a_h).exp();
        let group_b = &b[group * d_state..(group + 1) * d_state];
        let group_c = &c[group * d_state..(group + 1) * d_state];
        let d_h = d_param[h] as f64;
        let state_head = &mut state_out[h * state_per_head..(h + 1) * state_per_head];
        let y_head = &mut y[h * headdim..(h + 1) * headdim];
        for d in 0..headdim {
            let x_val = x[h * headdim + d] as f64;
            let mut y_val: f64 = 0.0;
            let state_d = &mut state_head[d * d_state..(d + 1) * d_state];
            for s in 0..d_state {
                let new_state = a_bar * state_d[s] + dt_h * (group_b[s] as f64) * x_val;
                state_d[s] = new_state;
                y_val += (group_c[s] as f64) * new_state;
            }
            y_head[d] = (y_val + d_h * x_val) as f32;
        }
    }
    let state_out_f32: Vec<f32> = state_out.into_iter().map(|v| v as f32).collect();
    (y, state_out_f32)
}

// ── Math helpers ───────────────────────────────────────────────────

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(w.iter()).map(|(v, ww)| v * inv * ww).collect()
}

/// Group RMSNorm: divide `x` into `n_groups` contiguous segments,
/// compute RMS per segment, multiply by per-element weight.
fn group_rms_norm(x: &[f32], w: &[f32], n_groups: usize, eps: f32) -> Vec<f32> {
    assert_eq!(x.len() % n_groups, 0);
    let gsz = x.len() / n_groups;
    let mut out = vec![0.0f32; x.len()];
    for g in 0..n_groups {
        let off = g * gsz;
        let mean_sq: f32 = x[off..off + gsz].iter().map(|v| v * v).sum::<f32>() / gsz as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for i in 0..gsz {
            out[off + i] = x[off + i] * inv * w[off + i];
        }
    }
    out
}

fn silu(x: f32) -> f32 {
    if x >= 0.0 { x / (1.0 + (-x).exp()) } else { let e = x.exp(); x * e / (1.0 + e) }
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) }
}

/// Row-vector × matrix: `out[1, n] = a[1, k] @ b[k, n]`. Calls
/// Row-vector × native-dtype matrix via `rce-op-qmm` — reads weight
/// bytes in their GGUF dtype (Q4_0 for Nemotron-Nano-4B's body, Q6_K
/// for lm_head) and produces f32 output without ever materialising f32
/// weights. Replaces the previous `sgemm_into`-on-pre-dequanted-f32
/// path; saves 4–6× weight memory bandwidth at the cost of more CPU
/// dequant work (parallelised across output cols via rayon).
fn qmm_1d(a: &[f32], w: &QuantWeight) -> Vec<f32> {
    let mut out = vec![0.0f32; w.n_rows];
    rce_op_qmm::mm_1d(a, w, &mut out);
    out
}

fn transpose_2d(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    out
}
