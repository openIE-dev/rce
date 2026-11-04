//! Pure Mamba-2 inference runtime on the rce substrate.
//!
//! Targets the `mamba2` GGUF architecture (the original Mamba-2 paper:
//! all-SSM, no attention, no FFN). Verified with `mamba2-2.7b-q5_k_m`
//! (64 SSM blocks, d_model=2560, d_inner=5120 = nheads 80 × headdim 64,
//! d_state=128, ngroups=1, conv_kernel=4).
//!
//! The mixer is identical to Nemotron-H's Mamba-2 mixer (same SSD form,
//! same in_proj `zxBCdt` layout, same gate-then-group-RMS-then-out_proj
//! order). The only structural difference is that every block is SSM —
//! there's no block-kind dispatch. Per-block state is just `(conv_state,
//! ssm_state)`; no KV cache, no RoPE pos counter.
//!
//! Reuses [`rce_op_ssm::selective_scan_step`] (now bit-correct vs
//! llama.cpp post-`ssm_a` convention fix — see
//! `project_gguf_ssm_a_convention.md`), [`rce_op_conv1d`], and
//! [`rce_op_blas::BlasMatmul`].

use rce_ingest_gguf::{
    read_tensor_f32, sample_with_history, GgufFile, GgufValue, Tokenizer,
};
pub use rce_ingest_gguf::SampleConfig;
use rce_op_conv1d::conv1d_causal_step;
use rce_op_qmm::QuantWeight;
use memmap2::Mmap;
use std::fs::File;
use std::sync::Arc;
use std::path::Path;

// ── Config ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Mamba2Config {
    pub d_model: usize,
    pub vocab: usize,
    pub n_layers: usize,
    pub ssm_n_heads: usize,
    pub ssm_head_dim: usize,
    pub ssm_d_state: usize,
    pub ssm_n_groups: usize,
    pub ssm_d_conv: usize,
    pub eps: f32,
}

impl Mamba2Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf.metadata.iter()
            .find(|kv| kv.key == "general.architecture")
            .and_then(|kv| match &kv.value {
                GgufValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .ok_or("missing general.architecture")?;
        if arch != "mamba2" {
            return Err(format!("expected mamba2, got {arch}"));
        }

        let u32_meta = |k: &str| -> Option<u32> {
            gguf.metadata.iter().find(|kv| kv.key == format!("mamba2.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::U32(v) => Some(*v), _ => None })
        };
        let f32_meta = |k: &str| -> Option<f32> {
            gguf.metadata.iter().find(|kv| kv.key == format!("mamba2.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::F32(v) => Some(*v), _ => None })
        };

        let d_model = u32_meta("embedding_length").ok_or("embedding_length")? as usize;
        let vocab = u32_meta("vocab_size")
            .or_else(|| gguf.tensors.iter()
                .find(|t| t.name == "token_embd.weight")
                .and_then(|t| t.dims.get(1).copied())
                .map(|d| d as u32))
            .ok_or("vocab_size")? as usize;
        let n_layers = u32_meta("block_count").ok_or("block_count")? as usize;
        let ssm_d_inner = u32_meta("ssm.inner_size").ok_or("ssm.inner_size")? as usize;
        let ssm_d_state = u32_meta("ssm.state_size").ok_or("ssm.state_size")? as usize;
        let ssm_n_groups = u32_meta("ssm.group_count").unwrap_or(1) as usize;
        let ssm_d_conv = u32_meta("ssm.conv_kernel").ok_or("ssm.conv_kernel")? as usize;
        let eps = f32_meta("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);

        // n_heads from blk.0.ssm_a's non-trivial dimension.
        let ssm_n_heads = gguf
            .tensors
            .iter()
            .find(|t| t.name.starts_with("blk.") && t.name.ends_with(".ssm_a"))
            .and_then(|t| t.dims.iter().copied().find(|&d| d > 1))
            .map(|d| d as usize)
            .ok_or("could not infer ssm n_heads from blk.*.ssm_a")?;
        if ssm_d_inner % ssm_n_heads != 0 {
            return Err(format!(
                "ssm_d_inner ({ssm_d_inner}) not divisible by ssm_n_heads ({ssm_n_heads})"
            ));
        }
        let ssm_head_dim = ssm_d_inner / ssm_n_heads;

        Ok(Mamba2Config {
            d_model, vocab, n_layers,
            ssm_n_heads, ssm_head_dim, ssm_d_state, ssm_n_groups, ssm_d_conv, eps,
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

// ── Weights ────────────────────────────────────────────────────────

struct SsmBlock {
    attn_norm_w: Vec<f32>,
    /// `[ssm_in_proj_out, d_model]` row-major, native GGUF dtype (Q5_K
    /// for mamba2-2.7b-q5_k_m). Routed through `rce_op_qmm::mm_1d`.
    ssm_in: QuantWeight,
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    a_log: Vec<f32>,
    d_param: Vec<f32>,
    dt_bias: Vec<f32>,
    ssm_norm_w: Vec<f32>,
    /// `[d_model, d_inner]` row-major, native GGUF dtype.
    ssm_out: QuantWeight,
}

struct Mamba2Weights {
    token_embd: QuantWeight,
    output_norm_w: Vec<f32>,
    /// `[vocab, d_model]` row-major, native GGUF dtype (typically Q6_K
    /// for mamba2 — the largest single matmul per forward pass).
    lm_head: QuantWeight,
    blocks: Vec<SsmBlock>,
}

// ── State ──────────────────────────────────────────────────────────

struct BlockState {
    conv_state: Vec<f32>,
    ssm_state: Vec<f32>,
}

pub struct Mamba2State {
    blocks: Vec<BlockState>,
    pos: usize,
}

impl Mamba2State {
    fn new(cfg: &Mamba2Config) -> Self {
        let blocks = (0..cfg.n_layers).map(|_| BlockState {
            conv_state: vec![0.0f32; cfg.ssm_conv_dim() * cfg.ssm_d_conv],
            ssm_state: vec![0.0f32; cfg.ssm_d_inner() * cfg.ssm_d_state],
        }).collect();
        Self { blocks, pos: 0 }
    }
}

// ── Loading ────────────────────────────────────────────────────────

fn load_weights(
    mmap: &Arc<Mmap>, gguf: &GgufFile, cfg: &Mamba2Config,
) -> Result<Mamba2Weights, String> {
    // Small tensors (norms, A, D, dt_bias, conv) still dequant to f32 —
    // they're tiny and used elementwise, not as matmul weights. Big
    // matmul weights stay as zero-copy `QuantWeight` views into `mmap`.
    let bytes: &[u8] = &mmap[..];
    let token_embd = QuantWeight::from_gguf(Arc::clone(mmap), gguf, "token_embd.weight", cfg.vocab, cfg.d_model)
        .map_err(|e| format!("token_embd: {e}"))?;
    let output_norm_w = read_tensor_f32(bytes, gguf, "output_norm.weight")
        .map_err(|e| format!("output_norm: {e:?}"))?;
    // Big matmul weights: native GGUF dtype, zero-copy, no transpose.
    // GGUF stores torch's `[N_out, K_in]` flat row-major — the natural
    // layout for `out[j] = Σ a[i]·W[j, i]`. Some Mamba-2 GGUFs tie
    // lm_head to token_embd; fall back if `output.weight` is missing.
    let lm_head = QuantWeight::from_gguf(Arc::clone(mmap), gguf, "output.weight", cfg.vocab, cfg.d_model)
        .or_else(|_| QuantWeight::from_gguf(Arc::clone(mmap), gguf, "token_embd.weight", cfg.vocab, cfg.d_model))
        .map_err(|e| format!("output: {e}"))?;

    let mut blocks = Vec::with_capacity(cfg.n_layers);
    for b in 0..cfg.n_layers {
        let bn = format!("blk.{b}");
        let attn_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_norm.weight"))
            .map_err(|e| format!("blk.{b}.attn_norm: {e:?}"))?;
        let conv_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_conv1d.weight")).map_err(|e| format!("{e:?}"))?;
        let conv_b = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_conv1d.bias")).map_err(|e| format!("{e:?}"))?;
        let a_log = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_a")).map_err(|e| format!("{e:?}"))?;
        let d_param = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_d")).map_err(|e| format!("{e:?}"))?;
        let dt_bias = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_dt.bias")).map_err(|e| format!("{e:?}"))?;
        let ssm_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ssm_norm.weight")).map_err(|e| format!("{e:?}"))?;
        let ssm_in = QuantWeight::from_gguf(
            Arc::clone(mmap), gguf, &format!("{bn}.ssm_in.weight"), cfg.ssm_in_proj_out(), cfg.d_model,
        ).map_err(|e| format!("blk.{b}.ssm_in: {e}"))?;
        let ssm_out = QuantWeight::from_gguf(
            Arc::clone(mmap), gguf, &format!("{bn}.ssm_out.weight"), cfg.d_model, cfg.ssm_d_inner(),
        ).map_err(|e| format!("blk.{b}.ssm_out: {e}"))?;
        blocks.push(SsmBlock {
            attn_norm_w, ssm_in, conv_w, conv_b, a_log, d_param, dt_bias, ssm_norm_w, ssm_out,
        });
    }
    Ok(Mamba2Weights { token_embd, output_norm_w, lm_head, blocks })
}

// ── Runtime ────────────────────────────────────────────────────────

pub struct Mamba2Runtime {
    cfg: Mamba2Config,
    gguf: GgufFile,
    weights: Mamba2Weights,
    /// Prebuilt once at load **only when `RCE_MAMBA2_IR` is set**, so
    /// default users pay zero. When `Some`, every block's mixer runs
    /// through this graph via the rce VM instead of hand `ssm_mixer`.
    mixer_graph: Option<MixerGraph>,
}

impl Mamba2Runtime {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| format!("open: {e}"))?;
        let mmap = Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?);
        // Hint sequential reads during the load-time weight scan; ignore
        // errors (kernel may refuse). See rce_ingest_gguf::mmap_load_sequential.
        let _ = mmap.advise(memmap2::Advice::WillNeed);
        let gguf = GgufFile::parse(&mmap[..]).map_err(|e| format!("parse: {e:?}"))?;
        let cfg = Mamba2Config::from_gguf(&gguf)?;
        eprintln!(
            "mamba2: d_model={} vocab={} n_layers={} ssm_n_heads={} ssm_head_dim={} ssm_n_groups={} ssm_d_state={} ssm_d_conv={}",
            cfg.d_model, cfg.vocab, cfg.n_layers, cfg.ssm_n_heads, cfg.ssm_head_dim, cfg.ssm_n_groups, cfg.ssm_d_state, cfg.ssm_d_conv,
        );
        // `mmap` is NOT dropped — every QuantWeight holds an Arc clone,
        // so the mapping stays resident as a zero-copy weight store.
        let weights = load_weights(&mmap, &gguf, &cfg)?;
        let mixer_graph = if std::env::var_os("RCE_MAMBA2_IR").is_some() {
            eprintln!("mamba2: RCE_MAMBA2_IR set — mixer runs through the rce VM graph");
            Some(build_mixer_graph(&cfg))
        } else {
            None
        };
        Ok(Self { cfg, gguf, weights, mixer_graph })
    }

    pub fn config(&self) -> &Mamba2Config { &self.cfg }

    pub fn tokenizer(&self) -> Tokenizer<'_> {
        Tokenizer::from_gguf(&self.gguf).expect("tokenizer")
    }

    pub fn new_state(&self) -> Mamba2State {
        Mamba2State::new(&self.cfg)
    }

    /// One forward pass for `token_id`. Threads conv/ssm state per block.
    pub fn forward_logits(&self, state: &mut Mamba2State, token_id: i32) -> Vec<f32> {
        let logits = forward_token(
            &self.weights, &self.cfg, state, token_id, None, self.mixer_graph.as_ref(),
        );
        state.pos += 1;
        logits
    }

    /// Prefill an entire prompt in **one batched pass per block** — each
    /// quantized weight is read from RAM once and reused across all
    /// prompt tokens (the prefill lever), instead of re-streaming every
    /// weight per token. Returns logits for the **last** prompt token
    /// (all the generation loop needs); leaves `state` byte-identical to
    /// `token_ids.len()` sequential [`forward_logits`] calls (proven by
    /// `forward_prefill_plus_decode_bit_exact_vs_per_token`).
    pub fn forward_prefill_logits(
        &self, state: &mut Mamba2State, token_ids: &[i32],
    ) -> Vec<f32> {
        forward_prefill(&self.weights, &self.cfg, state, token_ids)
    }

    /// Instrumented forward — fills `Profile` with per-op timings. Use
    /// for hotspot identification before optimization passes.
    pub fn forward_logits_profiled(
        &self,
        state: &mut Mamba2State,
        token_id: i32,
        profile: &std::cell::RefCell<Profile>,
    ) -> Vec<f32> {
        let logits = forward_token(
            &self.weights, &self.cfg, state, token_id, Some(profile), self.mixer_graph.as_ref(),
        );
        state.pos += 1;
        logits
    }
}

fn forward_token(
    w: &Mamba2Weights, cfg: &Mamba2Config,
    state: &mut Mamba2State, token_id: i32,
    prof: Option<&std::cell::RefCell<Profile>>,
    mixer_graph: Option<&MixerGraph>,
) -> Vec<f32> {
    let dm = cfg.d_model;
    let t0 = std::time::Instant::now();
    let mut x: Vec<f32> = w.token_embd.dequant_row(token_id as usize);
    if let Some(p) = prof { p.borrow_mut().embd += t0.elapsed(); }

    for (b, block) in w.blocks.iter().enumerate() {
        let t1 = std::time::Instant::now();
        let normed = rms_norm(&x, &block.attn_norm_w, cfg.eps);
        if let Some(p) = prof { p.borrow_mut().block_rms_norm += t1.elapsed(); }
        let bs = &mut state.blocks[b];
        let delta = match mixer_graph {
            Some(mg) => ssm_mixer_via_graph(
                &normed, block, &mut bs.conv_state, &mut bs.ssm_state, cfg, mg,
            ),
            None => ssm_mixer(&normed, block, &mut bs.conv_state, &mut bs.ssm_state, cfg, prof),
        };
        let t2 = std::time::Instant::now();
        for i in 0..dm { x[i] += delta[i]; }
        if let Some(p) = prof { p.borrow_mut().residual_add += t2.elapsed(); }
    }

    let t3 = std::time::Instant::now();
    let final_normed = rms_norm(&x, &w.output_norm_w, cfg.eps);
    if let Some(p) = prof { p.borrow_mut().final_norm += t3.elapsed(); }
    let t4 = std::time::Instant::now();
    let logits = qmm_1d(&final_normed, &w.lm_head);
    if let Some(p) = prof { p.borrow_mut().lm_head += t4.elapsed(); }
    logits
}

/// **Prefill forward**: consume a whole prompt of `token_ids` in one
/// pass — every block's mixer runs *once* over all `T` tokens via
/// [`ssm_mixer_prefill`] instead of `T` separate single-token passes.
/// The per-block conv/ssm `state` is threaded exactly as `T` sequential
/// [`forward_token`] calls would leave it, so generation can continue
/// from `state` with a normal decode step.
///
/// **Byte-identical** to looping `forward_token` over `token_ids` and
/// keeping the last token's logits: the embed, pre-block RMS, residual
/// add, final RMS and lm_head are the same scalar ops in the same order
/// (just iterated per row), and the mixer equivalence is brick 4.
/// Only the **last** token's logits are returned — that's all the
/// generation loop needs to predict the next token (the intermediate
/// prompt logits are computed-then-discarded in the per-token path too).
///
/// Returns logits `[vocab]` for `token_ids[T-1]`.
fn forward_prefill(
    w: &Mamba2Weights, cfg: &Mamba2Config,
    state: &mut Mamba2State, token_ids: &[i32],
) -> Vec<f32> {
    let dm = cfg.d_model;
    let t_rows = token_ids.len();
    assert!(t_rows > 0, "forward_prefill: empty prompt");

    // Embed every token, row-major [T, d_model].
    let mut x_all = vec![0.0f32; t_rows * dm];
    for (t, &tok) in token_ids.iter().enumerate() {
        let e = w.token_embd.dequant_row(tok as usize);
        x_all[t * dm..(t + 1) * dm].copy_from_slice(&e);
    }

    for (b, block) in w.blocks.iter().enumerate() {
        // Pre-block RMS per token (identical to forward_token's per-token
        // rms_norm — elementwise, order-independent across tokens).
        let mut normed_all = vec![0.0f32; t_rows * dm];
        for t in 0..t_rows {
            let n = rms_norm(&x_all[t * dm..(t + 1) * dm], &block.attn_norm_w, cfg.eps);
            normed_all[t * dm..(t + 1) * dm].copy_from_slice(&n);
        }
        let bs = &mut state.blocks[b];
        let delta_all = ssm_mixer_prefill(
            &normed_all, t_rows, block, &mut bs.conv_state, &mut bs.ssm_state, cfg,
        );
        for t in 0..t_rows {
            for i in 0..dm {
                x_all[t * dm + i] += delta_all[t * dm + i];
            }
        }
    }
    state.pos += t_rows;

    // Continuation needs only the final token's logits.
    let last = &x_all[(t_rows - 1) * dm..t_rows * dm];
    let final_normed = rms_norm(last, &w.output_norm_w, cfg.eps);
    qmm_1d(&final_normed, &w.lm_head)
}

// ── SSM mixer (Mamba-2) ────────────────────────────────────────────
//
// Identical algorithm to rce-nemotron's `ssm_mixer`. The only thing
// that differs is *which* config struct's accessors get called.

fn ssm_mixer(
    x_in: &[f32], s: &SsmBlock, conv_state: &mut Vec<f32>, ssm_state: &mut Vec<f32>,
    cfg: &Mamba2Config, prof: Option<&std::cell::RefCell<Profile>>,
) -> Vec<f32> {
    let dm = cfg.d_model;
    let d_inner = cfg.ssm_d_inner();
    let conv_dim = cfg.ssm_conv_dim();
    let nh = cfg.ssm_n_heads;
    let hd = cfg.ssm_head_dim;
    let ds = cfg.ssm_d_state;
    let ng = cfg.ssm_n_groups;

    let t0 = std::time::Instant::now();
    // in_proj: [d_model] @ [d_model, z+xBC+dt] -> packed [d_inner + conv_dim + nh]
    let proj = qmm_1d(x_in, &s.ssm_in);
    if let Some(p) = prof { p.borrow_mut().in_proj += t0.elapsed(); }
    let z = &proj[0..d_inner];
    let xbc = &proj[d_inner..d_inner + conv_dim];
    let dt_raw = &proj[d_inner + conv_dim..d_inner + conv_dim + nh];

    let t1 = std::time::Instant::now();
    let (xbc_conv, new_conv_state) = conv1d_causal_step(
        xbc, &s.conv_w, &s.conv_b, conv_state, conv_dim, cfg.ssm_d_conv,
    );
    *conv_state = new_conv_state;
    if let Some(p) = prof { p.borrow_mut().conv1d += t1.elapsed(); }

    let t2 = std::time::Instant::now();
    // SiLU on the whole xBC band, then split into x, B, C.
    let xbc_act: Vec<f32> = xbc_conv.iter().map(|v| silu(*v)).collect();
    if let Some(p) = prof { p.borrow_mut().silu_split += t2.elapsed(); }
    let x_act = &xbc_act[0..d_inner];
    let b_act = &xbc_act[d_inner..d_inner + ng * ds];
    let c_act = &xbc_act[d_inner + ng * ds..conv_dim];

    let t3 = std::time::Instant::now();
    let (y, new_ssm) = selective_scan_step_f64(
        x_act, b_act, c_act, dt_raw, &s.dt_bias, &s.a_log, &s.d_param, ssm_state,
        nh, hd, ds, ng,
    );
    *ssm_state = new_ssm;
    if let Some(p) = prof { p.borrow_mut().scan += t3.elapsed(); }

    let t4 = std::time::Instant::now();
    let mut gated = vec![0.0f32; d_inner];
    for i in 0..d_inner { gated[i] = silu(z[i]) * y[i]; }
    let normed = group_rms_norm(&gated, &s.ssm_norm_w, ng, cfg.eps);
    if let Some(p) = prof { p.borrow_mut().gate_norm += t4.elapsed(); }

    let t5 = std::time::Instant::now();
    let out = qmm_1d(&normed, &s.ssm_out);
    if let Some(p) = prof { p.borrow_mut().out_proj += t5.elapsed(); }
    out
}

#[derive(Default, Clone, Copy, Debug)]
pub struct Profile {
    pub in_proj: std::time::Duration,
    pub conv1d: std::time::Duration,
    pub silu_split: std::time::Duration,
    pub scan: std::time::Duration,
    pub gate_norm: std::time::Duration,
    pub out_proj: std::time::Duration,
    pub embd: std::time::Duration,
    pub block_rms_norm: std::time::Duration,
    pub residual_add: std::time::Duration,
    pub final_norm: std::time::Duration,
    pub lm_head: std::time::Duration,
}

impl Profile {
    pub fn report(&self, label: &str) {
        let total = self.in_proj + self.conv1d + self.silu_split + self.scan + self.gate_norm
            + self.out_proj + self.embd + self.block_rms_norm + self.residual_add
            + self.final_norm + self.lm_head;
        eprintln!("─── {label} profile (total {:.2?}) ───", total);
        let pct = |d: std::time::Duration| -> f64 {
            d.as_secs_f64() / total.as_secs_f64() * 100.0
        };
        eprintln!("  in_proj      : {:>8.2?}  {:>5.1}%", self.in_proj, pct(self.in_proj));
        eprintln!("  out_proj     : {:>8.2?}  {:>5.1}%", self.out_proj, pct(self.out_proj));
        eprintln!("  scan         : {:>8.2?}  {:>5.1}%", self.scan, pct(self.scan));
        eprintln!("  conv1d       : {:>8.2?}  {:>5.1}%", self.conv1d, pct(self.conv1d));
        eprintln!("  silu+split   : {:>8.2?}  {:>5.1}%", self.silu_split, pct(self.silu_split));
        eprintln!("  gate+groupRMS: {:>8.2?}  {:>5.1}%", self.gate_norm, pct(self.gate_norm));
        eprintln!("  pre-block RMS: {:>8.2?}  {:>5.1}%", self.block_rms_norm, pct(self.block_rms_norm));
        eprintln!("  residual add : {:>8.2?}  {:>5.1}%", self.residual_add, pct(self.residual_add));
        eprintln!("  embd lookup  : {:>8.2?}  {:>5.1}%", self.embd, pct(self.embd));
        eprintln!("  final RMS    : {:>8.2?}  {:>5.1}%", self.final_norm, pct(self.final_norm));
        eprintln!("  lm_head      : {:>8.2?}  {:>5.1}%", self.lm_head, pct(self.lm_head));
    }
}

/// f64-accumulator Mamba-2 selective scan.
///
/// Identical math to `rce-op-ssm::selective_scan_step` (post-`ssm_a`-fix),
/// but the inner d_state recurrence runs in f64 to keep cumulative
/// error from blowing up through deep stacks of SSM blocks.
///
/// **Serial on purpose.** Per scan call the total work is ~550µs
/// (64 head_dim × 128 d_state × 80 heads f64 fmacs), and a forward
/// pass enters this 64 times (once per layer). Splitting across
/// rayon threads has ~10–50µs task overhead per parallel section —
/// so 64 layers × that overhead exceeds the work being parallelized.
/// Both fine-grained (per-head) and chunked (per-thread) rayon variants
/// measured *slower* than serial (regressing wall-clock from 4.42 →
/// 4.44 tok/s). Local cache-resident per-head slices + tight serial
/// loops win here. If we ever rewrite the scan as a single batched
/// kernel over many timesteps (prefill batching), revisit.
fn selective_scan_step_f64(
    x: &[f32], b: &[f32], c: &[f32], dt_raw: &[f32], dt_bias: &[f32],
    a: &[f32], d_param: &[f32], state_in: &[f32],
    nheads: usize, headdim: usize, d_state: usize, ngroups: usize,
) -> (Vec<f32>, Vec<f32>) {
    let d_inner = nheads * headdim;
    let nheads_per_group = nheads / ngroups;
    let mut y = vec![0.0f32; d_inner];
    let mut state_out: Vec<f64> = state_in.iter().map(|&v| v as f64).collect();
    for h in 0..nheads {
        let group = h / nheads_per_group;
        let dt_pre = dt_raw[h] as f64 + dt_bias[h] as f64;
        let dt_h = if dt_pre > 20.0 { dt_pre } else { (1.0 + dt_pre.exp()).ln() };
        // GGUF `ssm_a` is already `A = -exp(A_log)` (linear). Use directly.
        let a_h = a[h] as f64;
        let a_bar = (dt_h * a_h).exp();
        let group_b = &b[group * d_state..(group + 1) * d_state];
        let group_c = &c[group * d_state..(group + 1) * d_state];
        let d_h = d_param[h] as f64;
        let state_head = &mut state_out[h * headdim * d_state..(h + 1) * headdim * d_state];
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

/// `T`-token batched form of the **local** [`selective_scan_step_f64`]:
/// the single-step looped over `t_rows` tokens, threading the SSM state
/// forward. **Definitionally byte-identical** to calling the local
/// single-step `t_rows` times — same f64 accumulation, same op order.
/// Mirrors `rce_op_ssm::selective_scan_step_f64_batched` but loops
/// *this crate's* scan so the prefill path stays bit-exact vs the hand
/// `ssm_mixer` (which calls the local single-step).
///
/// Per-token inputs are row-major `[t_rows, ·]`: `x[t_rows*d_inner]`,
/// `b`/`c [t_rows*ngroups*d_state]`, `dt_raw[t_rows*nheads]`. `dt_bias`,
/// `a`, `d_param [nheads]` are shared. Returns
/// `(y[t_rows*d_inner], state_out[nheads*headdim*d_state])`.
#[allow(clippy::too_many_arguments)]
fn selective_scan_step_f64_batched(
    x: &[f32], b: &[f32], c: &[f32], dt_raw: &[f32], dt_bias: &[f32],
    a: &[f32], d_param: &[f32], state_in: &[f32],
    t_rows: usize, nheads: usize, headdim: usize, d_state: usize, ngroups: usize,
) -> (Vec<f32>, Vec<f32>) {
    let d_inner = nheads * headdim;
    let bc_per = ngroups * d_state;
    let mut y_all = vec![0.0f32; t_rows * d_inner];
    let mut state = state_in.to_vec();
    for t in 0..t_rows {
        let (yt, st) = selective_scan_step_f64(
            &x[t * d_inner..(t + 1) * d_inner],
            &b[t * bc_per..(t + 1) * bc_per],
            &c[t * bc_per..(t + 1) * bc_per],
            &dt_raw[t * nheads..(t + 1) * nheads],
            dt_bias, a, d_param, &state,
            nheads, headdim, d_state, ngroups,
        );
        y_all[t * d_inner..(t + 1) * d_inner].copy_from_slice(&yt);
        state = st;
    }
    (y_all, state)
}

/// **Prefill mixer**: one SSM block over `t_rows` prompt tokens at once,
/// **byte-identical** to looping the single-token [`ssm_mixer`] `t_rows`
/// times with conv/ssm state threaded (non-IR config).
///
/// The whole point — and the only place there is any speedup — is the
/// two `qmm` matmuls (in_proj, out_proj): [`mm_batched`] reads each
/// memory-bandwidth-bound quantized weight **once** and reuses it across
/// all `t_rows` rows, instead of re-streaming the full weight per token.
/// Everything else (conv, scan) is inherently sequential and KB-cheap;
/// it runs through the brick-2/brick-3 batched helpers purely so the
/// state threads correctly in one call — same arithmetic, same order,
/// so the result is bit-for-bit the per-token path.
///
/// `x_in` is `[t_rows, d_model]` row-major; returns the out_proj delta
/// `[t_rows, d_model]` row-major. Updates `conv_state`/`ssm_state` to
/// their post-window values (exactly as `t_rows` single steps would).
fn ssm_mixer_prefill(
    x_in: &[f32], t_rows: usize, s: &SsmBlock,
    conv_state: &mut Vec<f32>, ssm_state: &mut Vec<f32>,
    cfg: &Mamba2Config,
) -> Vec<f32> {
    let dm = cfg.d_model;
    let d_inner = cfg.ssm_d_inner();
    let conv_dim = cfg.ssm_conv_dim();
    let in_proj_out = cfg.ssm_in_proj_out();
    let nh = cfg.ssm_n_heads;
    let hd = cfg.ssm_head_dim;
    let ds = cfg.ssm_d_state;
    let ng = cfg.ssm_n_groups;

    // Batched in_proj — the prefill lever. [T,d_model] @ ssm_in -> [T, in_proj_out].
    let mut proj = vec![0.0f32; t_rows * in_proj_out];
    rce_op_qmm::mm_batched(x_in, t_rows, &s.ssm_in, &mut proj);

    // Per-token de-interleave of the packed projection into the three
    // contiguous [T,·] bands the batched conv/scan helpers consume.
    let mut z_all = vec![0.0f32; t_rows * d_inner];
    let mut xbc_all = vec![0.0f32; t_rows * conv_dim];
    let mut dtr_all = vec![0.0f32; t_rows * nh];
    for t in 0..t_rows {
        let row = &proj[t * in_proj_out..(t + 1) * in_proj_out];
        z_all[t * d_inner..(t + 1) * d_inner].copy_from_slice(&row[0..d_inner]);
        xbc_all[t * conv_dim..(t + 1) * conv_dim]
            .copy_from_slice(&row[d_inner..d_inner + conv_dim]);
        dtr_all[t * nh..(t + 1) * nh]
            .copy_from_slice(&row[d_inner + conv_dim..d_inner + conv_dim + nh]);
    }

    // Sequence conv (brick 3): bit-identical to T× conv1d_causal_step,
    // threads conv_state internally.
    let (xbc_conv_all, new_conv_state) = rce_op_conv1d::conv1d_causal_step_batched(
        &xbc_all, &s.conv_w, &s.conv_b, conv_state, t_rows, conv_dim, cfg.ssm_d_conv,
    );
    *conv_state = new_conv_state;

    // SiLU over the whole xBC band (elementwise — order-independent),
    // then split per token into x/B/C.
    let xbc_act: Vec<f32> = xbc_conv_all.iter().map(|v| silu(*v)).collect();
    let mut x_all = vec![0.0f32; t_rows * d_inner];
    let mut b_all = vec![0.0f32; t_rows * ng * ds];
    let mut c_all = vec![0.0f32; t_rows * ng * ds];
    for t in 0..t_rows {
        let row = &xbc_act[t * conv_dim..(t + 1) * conv_dim];
        x_all[t * d_inner..(t + 1) * d_inner].copy_from_slice(&row[0..d_inner]);
        b_all[t * ng * ds..(t + 1) * ng * ds]
            .copy_from_slice(&row[d_inner..d_inner + ng * ds]);
        c_all[t * ng * ds..(t + 1) * ng * ds]
            .copy_from_slice(&row[d_inner + ng * ds..conv_dim]);
    }

    // Sequence scan (brick 2): bit-identical to T× selective_scan_step_f64,
    // threads ssm_state internally.
    let (y_all, new_ssm) = selective_scan_step_f64_batched(
        &x_all, &b_all, &c_all, &dtr_all, &s.dt_bias, &s.a_log, &s.d_param, ssm_state,
        t_rows, nh, hd, ds, ng,
    );
    *ssm_state = new_ssm;

    // Per-token gate + group-RMS (identical scalar math to ssm_mixer).
    let mut normed_all = vec![0.0f32; t_rows * d_inner];
    for t in 0..t_rows {
        let z = &z_all[t * d_inner..(t + 1) * d_inner];
        let y = &y_all[t * d_inner..(t + 1) * d_inner];
        let mut gated = vec![0.0f32; d_inner];
        for i in 0..d_inner {
            gated[i] = silu(z[i]) * y[i];
        }
        let normed = group_rms_norm(&gated, &s.ssm_norm_w, ng, cfg.eps);
        normed_all[t * d_inner..(t + 1) * d_inner].copy_from_slice(&normed);
    }

    // Batched out_proj — the second prefill lever.
    let mut out = vec![0.0f32; t_rows * dm];
    rce_op_qmm::mm_batched(&normed_all, t_rows, &s.ssm_out, &mut out);
    out
}

// ── Mixer as a persistent .fg graph (RCE_MAMBA2_IR) ──────────────
//
// The whole production mixer expressed once as a `FunctionGraph` and
// executed through the rce VM. Built once at load, reused every
// block/token (state threads via the packed concat output). Proven
// byte-identical to the hand `ssm_mixer` by
// `rce-op-ssm/tests/mamba2_mixer_qmm_e2e.rs`. This is what makes
// graph-level fusion / prefill-batching / layout-per-target possible —
// the optimizations an op-by-op engine structurally cannot do.

use rce_op_conv1d::Conv1dCausalLeaf;
use rce_op_nn_prims::{ConcatLeaf, RmsNormLeaf, SiluLeaf, SplitAtLeaf};
use rce_op_qmm::QmmLeaf;
use rce_op_ssm::SsmScanF64Leaf;
use rce_vm_backend_cpu::{
    interpret_program, interpret_with_config, HostTensor, Inputs, InterpretConfig, OpDispatch,
    Value,
};
use rce_graph::graph::{
    FunctionGraph, GraphNode, LiteralValue, OpKind, Param, ProgramGraph, Wire, WireKind,
};

pub struct MixerGraph {
    graph: FunctionGraph,
    qmm_in: usize,
    qmm_out: usize,
}

fn build_mixer_graph(cfg: &Mamba2Config) -> MixerGraph {
    let d_inner = cfg.ssm_d_inner();
    let conv_dim = cfg.ssm_conv_dim();
    let nh = cfg.ssm_n_heads;
    let hd = cfg.ssm_head_dim;
    let ds = cfg.ssm_d_state;
    let ng = cfg.ssm_n_groups;
    let d_conv = cfg.ssm_d_conv;
    let eps = cfg.eps;

    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut push = |g: GraphNode, v: &mut Vec<GraphNode>| -> usize {
        v.push(g);
        v.len() - 1
    };
    let p = |n: &str| GraphNode::Param(Param { name: n.into(), data_type: "f32".into() });
    let li = |v: i32| GraphNode::Literal { value: LiteralValue::I32(v), data_type: "i32".into() };
    let lf = |v: f32| GraphNode::Literal { value: LiteralValue::F32(v), data_type: "f32".into() };
    let ac = |n: &str| GraphNode::Apply { op: OpKind::Custom(n.into()), output_type: "f32".into() };

    let n_x = push(p("x_in"), &mut nodes);
    let n_cw = push(p("conv_w"), &mut nodes);
    let n_cb = push(p("conv_b"), &mut nodes);
    let n_dtb = push(p("dt_bias"), &mut nodes);
    let n_al = push(p("a_log"), &mut nodes);
    let n_dp = push(p("d_param"), &mut nodes);
    let n_snw = push(p("ssm_norm_w"), &mut nodes);
    let n_cs = push(p("conv_state"), &mut nodes);
    let n_ss = push(p("ssm_state"), &mut nodes);

    let l0 = push(li(0), &mut nodes);
    let l_di = push(li(d_inner as i32), &mut nodes);
    let l_cd = push(li(conv_dim as i32), &mut nodes);
    let l_dto = push(li((d_inner + conv_dim) as i32), &mut nodes);
    let l_nh = push(li(nh as i32), &mut nodes);
    let l_ngds = push(li((ng * ds) as i32), &mut nodes);
    let l_di_ngds = push(li((d_inner + ng * ds) as i32), &mut nodes);
    let l_csl = push(li((conv_dim * d_conv) as i32), &mut nodes);
    let l_ssl = push(li((d_inner * ds) as i32), &mut nodes);
    let l_dc = push(li(d_conv as i32), &mut nodes);
    let l_hd = push(li(hd as i32), &mut nodes);
    let l_ds = push(li(ds as i32), &mut nodes);
    let l_ng = push(li(ng as i32), &mut nodes);
    let l_eps = push(lf(eps), &mut nodes);

    let n_proj = push(ac("qmm"), &mut nodes);
    let n_z = push(ac("split_at"), &mut nodes);
    let n_xbc = push(ac("split_at"), &mut nodes);
    let n_dtr = push(ac("split_at"), &mut nodes);
    let n_pc = push(ac("conv1d_causal_depthwise"), &mut nodes);
    let n_xc = push(ac("split_at"), &mut nodes);
    let n_nc = push(ac("split_at"), &mut nodes);
    let n_xa = push(ac("silu"), &mut nodes);
    let n_x2 = push(ac("split_at"), &mut nodes);
    let n_b2 = push(ac("split_at"), &mut nodes);
    let n_c2 = push(ac("split_at"), &mut nodes);
    let n_ps = push(ac("ssm_scan_f64"), &mut nodes);
    let n_y = push(ac("split_at"), &mut nodes);
    let n_ns = push(ac("split_at"), &mut nodes);
    let n_sz = push(ac("silu"), &mut nodes);
    let n_g = push(GraphNode::Apply { op: OpKind::Mul, output_type: "f32".into() }, &mut nodes);
    let n_nm = push(ac("rms_norm"), &mut nodes);
    let n_out = push(ac("qmm"), &mut nodes);
    let n_res = push(ac("concat"), &mut nodes);

    let w: Vec<(usize, usize, usize)> = vec![
        (n_x, n_proj, 0),
        (n_proj, n_z, 0), (l0, n_z, 1), (l_di, n_z, 2),
        (n_proj, n_xbc, 0), (l_di, n_xbc, 1), (l_cd, n_xbc, 2),
        (n_proj, n_dtr, 0), (l_dto, n_dtr, 1), (l_nh, n_dtr, 2),
        (n_xbc, n_pc, 0), (n_cw, n_pc, 1), (n_cb, n_pc, 2), (n_cs, n_pc, 3), (l_dc, n_pc, 4),
        (n_pc, n_xc, 0), (l0, n_xc, 1), (l_cd, n_xc, 2),
        (n_pc, n_nc, 0), (l_cd, n_nc, 1), (l_csl, n_nc, 2),
        (n_xc, n_xa, 0),
        (n_xa, n_x2, 0), (l0, n_x2, 1), (l_di, n_x2, 2),
        (n_xa, n_b2, 0), (l_di, n_b2, 1), (l_ngds, n_b2, 2),
        (n_xa, n_c2, 0), (l_di_ngds, n_c2, 1), (l_ngds, n_c2, 2),
        (n_x2, n_ps, 0), (n_b2, n_ps, 1), (n_c2, n_ps, 2), (n_dtr, n_ps, 3),
        (n_dtb, n_ps, 4), (n_al, n_ps, 5), (n_dp, n_ps, 6), (n_ss, n_ps, 7),
        (l_hd, n_ps, 8), (l_ds, n_ps, 9),
        (n_ps, n_y, 0), (l0, n_y, 1), (l_di, n_y, 2),
        (n_ps, n_ns, 0), (l_di, n_ns, 1), (l_ssl, n_ns, 2),
        (n_z, n_sz, 0),
        (n_sz, n_g, 0), (n_y, n_g, 1),
        (n_g, n_nm, 0), (n_snw, n_nm, 1), (l_ng, n_nm, 2), (l_eps, n_nm, 3),
        (n_nm, n_out, 0),
        (n_out, n_res, 0), (n_nc, n_res, 1), (n_ns, n_res, 2),
    ];
    let wires = w
        .into_iter()
        .map(|(from, to, port_in_idx)| Wire {
            from,
            to,
            port_in_idx: port_in_idx as i32,
            wire_kind: WireKind::Move,
        })
        .collect();

    MixerGraph {
        graph: FunctionGraph {
            name: "mamba2_mixer_qmm".into(),
            nodes,
            wires,
            output_node: n_res,
            return_type: "f32".into(),
        },
        qmm_in: n_proj,
        qmm_out: n_out,
    }
}

/// Drive the prebuilt mixer graph for one block: same dataflow as
/// `ssm_mixer`, executed through the VM. Threads conv/ssm state via the
/// packed `[out ‖ new_conv_state ‖ new_ssm_state]` output. Returns the
/// out_proj delta; updates the two state buffers in place.
fn ssm_mixer_via_graph(
    x_in: &[f32], s: &SsmBlock, conv_state: &mut Vec<f32>, ssm_state: &mut Vec<f32>,
    cfg: &Mamba2Config, mg: &MixerGraph,
) -> Vec<f32> {
    let dm = cfg.d_model;
    let conv_state_len = cfg.ssm_conv_dim() * cfg.ssm_d_conv;
    let ssm_state_len = cfg.ssm_d_inner() * cfg.ssm_d_state;

    let mut qmm = QmmLeaf::new();
    qmm.register(mg.qmm_in, s.ssm_in.clone());
    qmm.register(mg.qmm_out, s.ssm_out.clone());
    let conv = Conv1dCausalLeaf::new();
    let silu_l = SiluLeaf::new();
    let split = SplitAtLeaf::new();
    let rms = RmsNormLeaf::new();
    let scan = SsmScanF64Leaf::new();
    let cat = ConcatLeaf::new();
    let ops: Vec<&dyn OpDispatch> = vec![&qmm, &conv, &silu_l, &split, &rms, &scan, &cat];

    let t = |v: &[f32]| Value::Tensor(HostTensor::from_f32(vec![v.len()], v).unwrap());
    let mut inputs: Inputs = Inputs::new();
    inputs.insert("x_in".into(), t(x_in));
    inputs.insert("conv_w".into(), t(&s.conv_w));
    inputs.insert("conv_b".into(), t(&s.conv_b));
    inputs.insert("dt_bias".into(), t(&s.dt_bias));
    inputs.insert("a_log".into(), t(&s.a_log));
    inputs.insert("d_param".into(), t(&s.d_param));
    inputs.insert("ssm_norm_w".into(), t(&s.ssm_norm_w));
    inputs.insert("conv_state".into(), t(conv_state));
    inputs.insert("ssm_state".into(), t(ssm_state));

    let packed = match interpret_with_config(
        &mg.graph,
        &inputs,
        InterpretConfig { leaves: &[], ops: &ops },
    )
    .expect("mixer graph interpret")
    {
        Value::Tensor(tt) => tt.as_f32().expect("f32 out"),
        other => panic!("mixer graph non-tensor result {other:?}"),
    };
    debug_assert_eq!(packed.len(), dm + conv_state_len + ssm_state_len);
    let out = packed[0..dm].to_vec();
    conv_state.copy_from_slice(&packed[dm..dm + conv_state_len]);
    ssm_state.copy_from_slice(&packed[dm + conv_state_len..]);
    out
}

// ── Math helpers ───────────────────────────────────────────────────

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(w.iter()).map(|(v, ww)| v * inv * ww).collect()
}

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

/// Row-vector × native-dtype matrix: `out[1, n] = a[1, k] @ w[n, k]^T`.
/// Reads `w.bytes` in its native GGUF dtype (Q5_K, Q6_K, f16, …) and
/// produces f32 output, **never materialising the full f32 weight**.
/// This is the key perf win — weight memory bandwidth drops by the
/// dtype's per-weight byte ratio (Q5_K: ~5.8×, Q4_K: ~7.1×, F16: ~2×).
fn qmm_1d(a: &[f32], w: &QuantWeight) -> Vec<f32> {
    // `RCE_MAMBA2_IR=1` routes every matmul through the rce VM
    // (`.fg` graph + `interpret_with_config` + `QmmLeaf`) instead of a
    // direct kernel call. Bit-identical output (same kernel under the
    // leaf); the point is to prove the *real model's* forward runs
    // through the substrate's IR + dispatch end-to-end, not just in a
    // unit test. Slow (graph allocated per call) — correctness/arch
    // path, opt-in, reversible.
    if std::env::var_os("RCE_MAMBA2_IR").is_some() {
        return rce_op_qmm::mm_1d_via_vm(a, w);
    }
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

/// Generation config for `Mamba2Runtime::generate`.
#[derive(Clone, Copy, Debug)]
pub struct Mamba2GenConfig {
    pub max_new_tokens: usize,
    pub sample: SampleConfig,
    pub rep_window: usize,
}

impl Default for Mamba2GenConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            sample: SampleConfig::chat(0xC0DE_F00D_2026),
            rep_window: 64,
        }
    }
}

impl Mamba2Runtime {
    /// Continue text from `prompt` for up to `cfg.max_new_tokens`
    /// tokens. Pure base-model completion (no chat template).
    /// `on_token(text)` runs once per generated token.
    pub fn generate(
        &self,
        prompt: &str,
        cfg: &Mamba2GenConfig,
        mut on_token: impl FnMut(&str),
    ) -> Result<String, String> {
        let tk = self.tokenizer();
        // Pick the encoder by GGUF `tokenizer.ggml.model`:
        //   "gpt2" → byte-level BPE (Mamba-2 from the paper uses GPT-NeoX,
        //   which is BPE; GGUF labels it "gpt2"). Default SentencePiece
        //   for other tokenizer families.
        let is_gpt2 = self.gguf.metadata.iter()
            .find(|kv| kv.key == "tokenizer.ggml.model")
            .map(|kv| matches!(&kv.value, GgufValue::String(s) if s == "gpt2"))
            .unwrap_or(false);
        let prompt_ids = if is_gpt2 {
            tk.encode_gpt2(prompt, false)
        } else {
            tk.encode(prompt, false)
        };
        if prompt_ids.is_empty() {
            return Err("tokenizer produced empty prompt".into());
        }
        let eos_id = tk.eos_token_id.unwrap_or(-1);

        let mut state = self.new_state();
        let mut history: Vec<i32> = Vec::with_capacity(prompt_ids.len() + cfg.max_new_tokens);
        let mut step: u64 = 0;
        let seed_for = |s: u64| {
            let mut c = cfg.sample;
            c.seed = cfg.sample.seed ^ s.wrapping_mul(0x9E37_79B9);
            c
        };

        // Prompt ingestion. The per-token loop's intermediate
        // `sample_with_history` calls are discarded (only the last
        // prompt token's prediction seeds generation) and sampling is
        // stateless per call, so the *only* observable effects are the
        // final-token logits, the threaded `state`, `history`, and the
        // `step` counter. `forward_prefill_logits` reproduces all of
        // them bit-for-bit (proven) while reading each weight once for
        // the whole prompt instead of once per token — the speed win.
        //
        // Skipped when `RCE_MAMBA2_IR` is set (that path deliberately
        // routes every op through the VM graph, which prefill bypasses)
        // or when explicitly opted out for debugging.
        let use_prefill = self.mixer_graph.is_none()
            && std::env::var_os("RCE_MAMBA2_NO_PREFILL").is_none();
        let mut next: i32 = -1;
        if use_prefill {
            let logits = self.forward_prefill_logits(&mut state, &prompt_ids);
            for &tok in &prompt_ids {
                history.push(tok);
            }
            let recent = &history[history.len().saturating_sub(cfg.rep_window)..];
            next = sample_with_history(
                &logits, &seed_for(prompt_ids.len() as u64 - 1), recent,
            );
            step = prompt_ids.len() as u64;
        } else {
            for (i, &tok) in prompt_ids.iter().enumerate() {
                let logits = self.forward_logits(&mut state, tok);
                history.push(tok);
                let recent = &history[history.len().saturating_sub(cfg.rep_window)..];
                let pred = sample_with_history(&logits, &seed_for(step), recent);
                step += 1;
                if i == prompt_ids.len() - 1 {
                    next = pred;
                }
            }
        }

        let mut out_ids: Vec<i32> = Vec::new();
        for _ in 0..cfg.max_new_tokens {
            if next == eos_id { break; }
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

#[cfg(test)]
mod tests {
    use super::*;
    use rce_ingest_gguf::GgufDtype;

    /// Pack `[n_rows, n_cols]` row-major f32 into a raw-bytes `QuantWeight`
    /// (GGUF `F32` dtype). `mm_batched` takes the per-row fallback for
    /// F32, so this exercises the prefill *orchestration* (de-interleave,
    /// batched conv/scan state threading, gate, group-RMS) against the
    /// per-token path. The Q5_K *fast* path of `mm_batched` is proven
    /// bit-identical to per-row separately, in rce-op-qmm.
    fn f32_qweight(vals: &[f32], n_rows: usize, n_cols: usize) -> QuantWeight {
        assert_eq!(vals.len(), n_rows * n_cols);
        let mut bytes = Vec::with_capacity(vals.len() * 4);
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        QuantWeight::from_owned_bytes(bytes, GgufDtype::F32, n_rows, n_cols)
    }

    fn ramp(n: usize, scale: f32, bias: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * scale + bias).sin()).collect()
    }

    /// Brick 4 contract: `ssm_mixer_prefill` over `T` prompt tokens must
    /// be **byte-identical** to looping the single-token `ssm_mixer`
    /// `T` times with conv/ssm state threaded — output delta *and* both
    /// final state buffers. Same precondition as bricks 1–3: the fast
    /// (batched) path equals the per-token reference, bit-for-bit.
    #[test]
    fn ssm_mixer_prefill_bit_exact_vs_t_single_steps() {
        assert!(
            std::env::var_os("RCE_MAMBA2_IR").is_none(),
            "this test compares the native prefill path vs the native \
             per-token path — run it without RCE_MAMBA2_IR set"
        );

        // Same shape family as rce-op-ssm/tests/mamba2_mixer_qmm_e2e.rs.
        let cfg = Mamba2Config {
            d_model: 32,
            vocab: 64,
            n_layers: 1,
            ssm_n_heads: 8,
            ssm_head_dim: 8,
            ssm_d_state: 16,
            ssm_n_groups: 2,
            ssm_d_conv: 4,
            eps: 1e-5,
        };
        let dm = cfg.d_model;
        let d_inner = cfg.ssm_d_inner();          // 64
        let conv_dim = cfg.ssm_conv_dim();        // 128
        let in_proj_out = cfg.ssm_in_proj_out();  // 200
        let nh = cfg.ssm_n_heads;                 // 8
        let dc = cfg.ssm_d_conv;                  // 4
        let conv_state_len = conv_dim * dc;       // 512
        let ssm_state_len = d_inner * cfg.ssm_d_state; // 1024
        let t_rows = 5usize;

        let block = SsmBlock {
            attn_norm_w: ramp(dm, 0.03, 0.1),
            ssm_in: f32_qweight(&ramp(in_proj_out * dm, 0.0007, 0.2), in_proj_out, dm),
            conv_w: ramp(conv_dim * dc, 0.001, -0.3),
            conv_b: ramp(conv_dim, 0.002, 0.05),
            a_log: (0..nh).map(|i| -((i + 1) as f32) * 0.1).collect(),
            d_param: (0..nh).map(|i| i as f32 * 0.005).collect(),
            dt_bias: (0..nh).map(|i| i as f32 * 0.02 + 0.1).collect(),
            ssm_norm_w: ramp(d_inner, 0.004, 0.7),
            ssm_out: f32_qweight(&ramp(dm * d_inner, 0.0009, -0.4), dm, d_inner),
        };

        // T distinct prompt tokens, row-major [T, d_model].
        let mut x_all = Vec::with_capacity(t_rows * dm);
        for t in 0..t_rows {
            x_all.extend(ramp(dm, 0.05, t as f32 * 0.11 - 0.5));
        }

        // Reference: T single steps, state threaded by hand.
        let mut ref_conv = vec![0.0f32; conv_state_len];
        let mut ref_ssm = vec![0.0f32; ssm_state_len];
        let mut ref_out = Vec::with_capacity(t_rows * dm);
        for t in 0..t_rows {
            let delta = ssm_mixer(
                &x_all[t * dm..(t + 1) * dm],
                &block, &mut ref_conv, &mut ref_ssm, &cfg, None,
            );
            ref_out.extend_from_slice(&delta);
        }

        // Prefill: one batched call.
        let mut pf_conv = vec![0.0f32; conv_state_len];
        let mut pf_ssm = vec![0.0f32; ssm_state_len];
        let pf_out = ssm_mixer_prefill(
            &x_all, t_rows, &block, &mut pf_conv, &mut pf_ssm, &cfg,
        );

        assert_eq!(pf_out.len(), t_rows * dm);
        for (i, (g, w)) in pf_out.iter().zip(ref_out.iter()).enumerate() {
            assert_eq!(
                g.to_bits(), w.to_bits(),
                "out[t={}, i={}]: prefill {g} ≠ T×ssm_mixer {w} (must be bit-identical)",
                i / dm, i % dm
            );
        }
        for (i, (g, w)) in pf_conv.iter().zip(ref_conv.iter()).enumerate() {
            assert_eq!(
                g.to_bits(), w.to_bits(),
                "conv_state[{i}]: prefill {g} ≠ T×ssm_mixer {w} (must be bit-identical)"
            );
        }
        for (i, (g, w)) in pf_ssm.iter().zip(ref_ssm.iter()).enumerate() {
            assert_eq!(
                g.to_bits(), w.to_bits(),
                "ssm_state[{i}]: prefill {g} ≠ T×ssm_mixer {w} (must be bit-identical)"
            );
        }
    }

    fn make_block(cfg: &Mamba2Config, seed: f32) -> SsmBlock {
        let dm = cfg.d_model;
        let d_inner = cfg.ssm_d_inner();
        let conv_dim = cfg.ssm_conv_dim();
        let in_proj_out = cfg.ssm_in_proj_out();
        let nh = cfg.ssm_n_heads;
        let dc = cfg.ssm_d_conv;
        SsmBlock {
            attn_norm_w: ramp(dm, 0.03, 0.1 + seed),
            ssm_in: f32_qweight(&ramp(in_proj_out * dm, 0.0007, 0.2 + seed), in_proj_out, dm),
            conv_w: ramp(conv_dim * dc, 0.001, -0.3 + seed),
            conv_b: ramp(conv_dim, 0.002, 0.05 + seed),
            a_log: (0..nh).map(|i| -((i + 1) as f32) * 0.1 - seed).collect(),
            d_param: (0..nh).map(|i| i as f32 * 0.005 + seed).collect(),
            dt_bias: (0..nh).map(|i| i as f32 * 0.02 + 0.1 + seed).collect(),
            ssm_norm_w: ramp(d_inner, 0.004, 0.7 + seed),
            ssm_out: f32_qweight(&ramp(dm * d_inner, 0.0009, -0.4 + seed), dm, d_inner),
        }
    }

    /// Brick 5 — end-to-end: a `T`-token prompt consumed via
    /// [`forward_prefill`], then one decode step, must be **byte-identical**
    /// to feeding the same `T` tokens one at a time through
    /// [`forward_token`] and then the same decode token. Checks the
    /// continuation logits *and* every block's threaded conv/ssm state.
    /// Multi-block (n_layers=2) so cross-block residual + state threading
    /// is exercised. This is the wire-in proof: prefill leaves the model
    /// in exactly the state the per-token path would, so generation
    /// continues bit-for-bit identically.
    #[test]
    fn forward_prefill_plus_decode_bit_exact_vs_per_token() {
        assert!(
            std::env::var_os("RCE_MAMBA2_IR").is_none(),
            "compares native prefill vs native per-token — run without RCE_MAMBA2_IR"
        );

        let cfg = Mamba2Config {
            d_model: 32,
            vocab: 16,
            n_layers: 2,
            ssm_n_heads: 8,
            ssm_head_dim: 8,
            ssm_d_state: 16,
            ssm_n_groups: 2,
            ssm_d_conv: 4,
            eps: 1e-5,
        };
        let weights = Mamba2Weights {
            token_embd: f32_qweight(
                &ramp(cfg.vocab * cfg.d_model, 0.013, -0.2),
                cfg.vocab, cfg.d_model,
            ),
            output_norm_w: ramp(cfg.d_model, 0.005, 0.3),
            lm_head: f32_qweight(
                &ramp(cfg.vocab * cfg.d_model, 0.011, 0.15),
                cfg.vocab, cfg.d_model,
            ),
            blocks: vec![make_block(&cfg, 0.0), make_block(&cfg, 0.07)],
        };

        let prompt: [i32; 6] = [3, 7, 1, 12, 5, 9];
        let decode: i32 = 4;

        // Path A — per-token reference.
        let mut state_a = Mamba2State::new(&cfg);
        for &id in &prompt {
            let _ = forward_token(&weights, &cfg, &mut state_a, id, None, None);
        }
        let logits_a = forward_token(&weights, &cfg, &mut state_a, decode, None, None);

        // Path B — prefill the prompt, then the same decode step.
        let mut state_b = Mamba2State::new(&cfg);
        let _ = forward_prefill(&weights, &cfg, &mut state_b, &prompt);
        let logits_b = forward_token(&weights, &cfg, &mut state_b, decode, None, None);

        assert_eq!(logits_a.len(), cfg.vocab);
        assert_eq!(logits_b.len(), cfg.vocab);
        for (i, (a, b)) in logits_a.iter().zip(logits_b.iter()).enumerate() {
            assert_eq!(
                a.to_bits(), b.to_bits(),
                "logit[{i}]: per-token {a} ≠ prefill+decode {b} (must be bit-identical)"
            );
        }
        for blk in 0..cfg.n_layers {
            let (ba, bb) = (&state_a.blocks[blk], &state_b.blocks[blk]);
            for (i, (a, b)) in ba.conv_state.iter().zip(bb.conv_state.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(), b.to_bits(),
                    "block {blk} conv_state[{i}]: per-token {a} ≠ prefill {b}"
                );
            }
            for (i, (a, b)) in ba.ssm_state.iter().zip(bb.ssm_state.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(), b.to_bits(),
                    "block {blk} ssm_state[{i}]: per-token {a} ≠ prefill {b}"
                );
            }
        }
    }

    /// **Converter brick 4b — composition through the VM ProgramGraph
    /// executor.** A 2-function `ProgramGraph` whose entry `block_body`
    /// runs `rms_norm` (as a leaf) then `Call`s the **proven** mixer
    /// `FunctionGraph`, executed via `interpret_program`, must be
    /// **byte-identical** to the proven host path: host `rms_norm` →
    /// the same mixer graph via `interpret_with_config`. This proves
    /// the new executor *composes* a real op with the mixer Call with
    /// zero numeric drift (Call is FP-neutral; the only new thing vs
    /// brick 4a is an in-graph leaf feeding the Call). lm_head / embed /
    /// N-block stay host (gated on the deferred first-class-weight-IR
    /// change — the loader host-drives per block, the proven pattern).
    #[test]
    fn brick4b_block_body_programgraph_bit_exact_vs_host() {
        let cfg = Mamba2Config {
            d_model: 32, vocab: 64, n_layers: 1,
            ssm_n_heads: 8, ssm_head_dim: 8, ssm_d_state: 16,
            ssm_n_groups: 2, ssm_d_conv: 4, eps: 1e-5,
        };
        let dm = cfg.d_model;
        let conv_state_len = cfg.ssm_conv_dim() * cfg.ssm_d_conv;
        let ssm_state_len = cfg.ssm_d_inner() * cfg.ssm_d_state;
        let s = make_block(&cfg, 0.0);
        let mg = build_mixer_graph(&cfg);
        let x_in = ramp(dm, 0.05, -0.3);
        let conv_state = vec![0.0f32; conv_state_len];
        let ssm_state = vec![0.0f32; ssm_state_len];

        let t = |v: &[f32]| Value::Tensor(HostTensor::from_f32(vec![v.len()], v).unwrap());
        let mk_ops = || {
            let mut qmm = QmmLeaf::new();
            qmm.register(mg.qmm_in, s.ssm_in.clone());
            qmm.register(mg.qmm_out, s.ssm_out.clone());
            (qmm, Conv1dCausalLeaf::new(), SiluLeaf::new(), SplitAtLeaf::new(),
             RmsNormLeaf::new(), SsmScanF64Leaf::new(), ConcatLeaf::new())
        };

        // ── Reference: proven host path (host rms_norm → mixer graph) ──
        let normed = rms_norm(&x_in, &s.attn_norm_w, cfg.eps);
        let (rq, rc, rsi, rsp, rr, rsc, rcat) = mk_ops();
        let ref_ops: Vec<&dyn OpDispatch> =
            vec![&rq, &rc, &rsi, &rsp, &rr, &rsc, &rcat];
        let mut ref_in: Inputs = Inputs::new();
        ref_in.insert("x_in".into(), t(&normed));
        ref_in.insert("conv_w".into(), t(&s.conv_w));
        ref_in.insert("conv_b".into(), t(&s.conv_b));
        ref_in.insert("dt_bias".into(), t(&s.dt_bias));
        ref_in.insert("a_log".into(), t(&s.a_log));
        ref_in.insert("d_param".into(), t(&s.d_param));
        ref_in.insert("ssm_norm_w".into(), t(&s.ssm_norm_w));
        ref_in.insert("conv_state".into(), t(&conv_state));
        ref_in.insert("ssm_state".into(), t(&ssm_state));
        let packed_ref = match interpret_with_config(
            &mg.graph, &ref_in, InterpretConfig { leaves: &[], ops: &ref_ops },
        ).expect("ref mixer interpret") {
            Value::Tensor(tt) => tt.as_f32().expect("f32"),
            o => panic!("non-tensor {o:?}"),
        };

        // ── Under test: rms-leaf → Call mixer, via interpret_program ──
        let mixer_callee = {
            let mut g = mg.graph.clone();
            g.name = "mixer".into();
            g
        };
        // block_body: 0..9 params, 10 ng-literal, 11 eps-literal,
        // 12 rms-leaf, 13 Call("mixer").
        let p = |n: &str| GraphNode::Param(Param { name: n.into(), data_type: "f32".into() });
        let entry = FunctionGraph {
            name: "block_body".into(),
            nodes: vec![
                p("x_in"), p("attn_norm_w"), p("conv_w"), p("conv_b"), p("dt_bias"),
                p("a_log"), p("d_param"), p("ssm_norm_w"), p("conv_state"), p("ssm_state"),
                GraphNode::Literal { value: LiteralValue::I32(1), data_type: "i32".into() },
                GraphNode::Literal { value: LiteralValue::F32(cfg.eps), data_type: "f32".into() },
                GraphNode::Apply {
                    op: OpKind::Custom("rms_norm".into()), output_type: "f32".into(),
                },
                GraphNode::Call {
                    callee: "mixer".into(),
                    arg_types: vec!["f32".into(); 9],
                    return_type: "f32".into(),
                },
            ],
            wires: vec![
                // rms_norm(x_in, attn_norm_w, ng=1, eps)
                Wire { from: 0, to: 12, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 12, port_in_idx: 1, wire_kind: WireKind::Move },
                Wire { from: 10, to: 12, port_in_idx: 2, wire_kind: WireKind::Move },
                Wire { from: 11, to: 12, port_in_idx: 3, wire_kind: WireKind::Move },
                // Call mixer(rms_out, conv_w, conv_b, dt_bias, a_log,
                //            d_param, ssm_norm_w, conv_state, ssm_state)
                Wire { from: 12, to: 13, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 2, to: 13, port_in_idx: 1, wire_kind: WireKind::Move },
                Wire { from: 3, to: 13, port_in_idx: 2, wire_kind: WireKind::Move },
                Wire { from: 4, to: 13, port_in_idx: 3, wire_kind: WireKind::Move },
                Wire { from: 5, to: 13, port_in_idx: 4, wire_kind: WireKind::Move },
                Wire { from: 6, to: 13, port_in_idx: 5, wire_kind: WireKind::Move },
                Wire { from: 7, to: 13, port_in_idx: 6, wire_kind: WireKind::Move },
                Wire { from: 8, to: 13, port_in_idx: 7, wire_kind: WireKind::Move },
                Wire { from: 9, to: 13, port_in_idx: 8, wire_kind: WireKind::Move },
            ],
            output_node: 13,
            return_type: "f32".into(),
        };
        let prog = ProgramGraph {
            functions: vec![entry, mixer_callee],
            entry: "block_body".into(),
            allows: Default::default(),
        };
        let (gq, gc, gsi, gsp, gr, gsc, gcat) = mk_ops();
        let g_ops: Vec<&dyn OpDispatch> =
            vec![&gq, &gc, &gsi, &gsp, &gr, &gsc, &gcat];
        let mut gin: Inputs = Inputs::new();
        gin.insert("x_in".into(), t(&x_in)); // raw x — rms happens IN the graph
        gin.insert("attn_norm_w".into(), t(&s.attn_norm_w));
        gin.insert("conv_w".into(), t(&s.conv_w));
        gin.insert("conv_b".into(), t(&s.conv_b));
        gin.insert("dt_bias".into(), t(&s.dt_bias));
        gin.insert("a_log".into(), t(&s.a_log));
        gin.insert("d_param".into(), t(&s.d_param));
        gin.insert("ssm_norm_w".into(), t(&s.ssm_norm_w));
        gin.insert("conv_state".into(), t(&conv_state));
        gin.insert("ssm_state".into(), t(&ssm_state));
        let packed_got = match interpret_program(
            &prog, &gin, InterpretConfig { leaves: &[], ops: &g_ops },
        ).expect("interpret_program block_body") {
            Value::Tensor(tt) => tt.as_f32().expect("f32"),
            o => panic!("non-tensor {o:?}"),
        };

        assert_eq!(
            packed_got.len(),
            dm + conv_state_len + ssm_state_len,
            "packed length mismatch"
        );
        assert_eq!(packed_got.len(), packed_ref.len());
        for (i, (g, r)) in packed_got.iter().zip(packed_ref.iter()).enumerate() {
            assert_eq!(
                g.to_bits(), r.to_bits(),
                "packed[{i}]: ProgramGraph(rms-leaf→Call mixer) {g} ≠ \
                 host rms_norm→mixer graph {r} (must be byte-identical)"
            );
        }
    }

    /// **Brick 6c — the faithful end-to-end gate: converted-model
    /// generation vs f32, on the real mamba2-2.7b.** The capstone
    /// question the whole project ultimately answers.
    ///
    /// Two full 64-layer forwards over a real token sequence, threading
    /// conv/ssm state, identical in *every* respect except the body
    /// matmul precision: one with **fq4** body weights (the converted
    /// artifact — brick 6b proved `.fgm` stores these byte-identically,
    /// so deriving fq4 from GGUF here measures the artifact's behaviour
    /// without re-reading disk), one with **f32** body (the agreed
    /// ground truth). Both reuse the **proven mixer `FunctionGraph`**
    /// for all SSM math (scan/conv/gate/norm — not re-implemented; the
    /// only diff is `Fq4QmmLeaf` vs `QmmLeaf`+F32). `lm_head`/`embed`
    /// are the **same raw `QuantWeight`** in both (6e: identical) — so
    /// any logit divergence is purely the fq4 *body* error compounded
    /// over 64 layers. Gate: greedy next-token must agree (the model
    /// generates the same token); top-5 / logit drift reported. A
    /// failure is a real finding (body needs higher precision), not
    /// papered over — same honest stance as 6d. Skips if drive absent.
    #[test]
    fn brick6c_fq4_body_vs_f32_endtoend_generation() {
        // QmmLeaf, GgufFile, read_tensor_f32, QuantWeight, the leaves,
        // build_mixer_graph, rms_norm, qmm_1d, Mamba2Config, interpret_*
        // are all in scope via `super::*`. Only the fq4 names are new.
        use rce_op_qmm::{fq4_quantize, Fq4QmmLeaf, Fq4Tensor};

        const MODEL: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/mamba2-2.7b-gguf/mamba2-2.7b-q5_k_m.gguf";
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("brick6c: external model not mounted — skipping");
            return;
        }
        let file = std::fs::File::open(MODEL).unwrap();
        let mmap = std::sync::Arc::new(unsafe { Mmap::map(&file) }.unwrap());
        let gguf = GgufFile::parse(&mmap[..]).unwrap();
        let bytes: &[u8] = &mmap[..];
        let cfg = Mamba2Config::from_gguf(&gguf).expect("cfg");
        let dm = cfg.d_model;
        let in_po = cfg.ssm_in_proj_out();
        let d_inner = cfg.ssm_d_inner();
        let csl = cfg.ssm_conv_dim() * cfg.ssm_d_conv;
        let ssl = d_inner * cfg.ssm_d_state;
        let nl = cfg.n_layers;
        let mg = build_mixer_graph(&cfg);

        // Reconstruct weights ONCE. Body: both f32 (ref) + fq4
        // (candidate). lm_head/embed: one raw QuantWeight, shared.
        let f32_qw = |v: &[f32], r: usize, c: usize| {
            let mut b = Vec::with_capacity(v.len() * 4);
            for x in v { b.extend_from_slice(&x.to_le_bytes()); }
            QuantWeight::from_owned_bytes(b, rce_ingest_gguf::GgufDtype::F32, r, c)
        };
        let token_embd =
            QuantWeight::from_gguf(std::sync::Arc::clone(&mmap), &gguf, "token_embd.weight", cfg.vocab, dm).unwrap();
        let lm_name = if gguf.tensors.iter().any(|t| t.name == "output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let lm_head =
            QuantWeight::from_gguf(std::sync::Arc::clone(&mmap), &gguf, lm_name, cfg.vocab, dm).unwrap();
        let output_norm = read_tensor_f32(bytes, &gguf, "output_norm.weight").unwrap();

        struct Blk {
            attn_norm: Vec<f32>, conv_w: Vec<f32>, conv_b: Vec<f32>,
            a_log: Vec<f32>, d_param: Vec<f32>, dt_bias: Vec<f32>, ssm_norm: Vec<f32>,
            in_f32: QuantWeight, out_f32: QuantWeight,
            in_fq4: Fq4Tensor, out_fq4: Fq4Tensor,
        }
        let mut blks = Vec::with_capacity(nl);
        for b in 0..nl {
            let bn = format!("blk.{b}");
            let rd = |s: &str| read_tensor_f32(bytes, &gguf, &format!("{bn}.{s}")).unwrap();
            let deq = |name: &str, r: usize, c: usize| -> Vec<f32> {
                let qw = QuantWeight::from_gguf(std::sync::Arc::clone(&mmap), &gguf, name, r, c).unwrap();
                (0..r).flat_map(|row| qw.dequant_row(row)).collect()
            };
            let in_g = deq(&format!("{bn}.ssm_in.weight"), in_po, dm);
            let out_g = deq(&format!("{bn}.ssm_out.weight"), dm, d_inner);
            blks.push(Blk {
                attn_norm: rd("attn_norm.weight"),
                conv_w: rd("ssm_conv1d.weight"), conv_b: rd("ssm_conv1d.bias"),
                a_log: rd("ssm_a"), d_param: rd("ssm_d"), dt_bias: rd("ssm_dt.bias"),
                ssm_norm: rd("ssm_norm.weight"),
                in_fq4: fq4_quantize(&in_g, in_po, dm),
                out_fq4: fq4_quantize(&out_g, dm, d_inner),
                in_f32: f32_qw(&in_g, in_po, dm),
                out_f32: f32_qw(&out_g, dm, d_inner),
            });
        }

        let tv = |v: &[f32]| Value::Tensor(HostTensor::from_f32(vec![v.len()], v).unwrap());
        let prompt: [i32; 5] = [1, 318, 4842, 11, 642]; // arbitrary valid ids

        // One forward; `use_fq4` flips ONLY the body qmm leaf.
        let run = |use_fq4: bool| -> Vec<f32> {
            let mut conv: Vec<Vec<f32>> = (0..nl).map(|_| vec![0.0f32; csl]).collect();
            let mut ssm: Vec<Vec<f32>> = (0..nl).map(|_| vec![0.0f32; ssl]).collect();
            let mut last = vec![0.0f32; cfg.vocab];
            for &tok in &prompt {
                let mut x = token_embd.dequant_row(tok as usize);
                for b in 0..nl {
                    let bk = &blks[b];
                    let normed = rms_norm(&x, &bk.attn_norm, cfg.eps);
                    let conv_l = Conv1dCausalLeaf::new();
                    let silu_l = SiluLeaf::new();
                    let split = SplitAtLeaf::new();
                    let rmsl = RmsNormLeaf::new();
                    let scan = SsmScanF64Leaf::new();
                    let cat = ConcatLeaf::new();
                    let mut inp: Inputs = Inputs::new();
                    inp.insert("x_in".into(), tv(&normed));
                    inp.insert("conv_w".into(), tv(&bk.conv_w));
                    inp.insert("conv_b".into(), tv(&bk.conv_b));
                    inp.insert("dt_bias".into(), tv(&bk.dt_bias));
                    inp.insert("a_log".into(), tv(&bk.a_log));
                    inp.insert("d_param".into(), tv(&bk.d_param));
                    inp.insert("ssm_norm_w".into(), tv(&bk.ssm_norm));
                    inp.insert("conv_state".into(), tv(&conv[b]));
                    inp.insert("ssm_state".into(), tv(&ssm[b]));
                    let packed = if use_fq4 {
                        let mut q = Fq4QmmLeaf::new();
                        q.register(mg.qmm_in, bk.in_fq4.clone());
                        q.register(mg.qmm_out, bk.out_fq4.clone());
                        let ops: Vec<&dyn OpDispatch> =
                            vec![&q, &conv_l, &silu_l, &split, &rmsl, &scan, &cat];
                        interpret_with_config(&mg.graph, &inp, InterpretConfig { leaves: &[], ops: &ops })
                    } else {
                        let mut q = QmmLeaf::new();
                        q.register(mg.qmm_in, bk.in_f32.clone());
                        q.register(mg.qmm_out, bk.out_f32.clone());
                        let ops: Vec<&dyn OpDispatch> =
                            vec![&q, &conv_l, &silu_l, &split, &rmsl, &scan, &cat];
                        interpret_with_config(&mg.graph, &inp, InterpretConfig { leaves: &[], ops: &ops })
                    };
                    let packed = match packed.expect("mixer interpret") {
                        Value::Tensor(t) => t.as_f32().expect("f32"),
                        o => panic!("non-tensor {o:?}"),
                    };
                    for i in 0..dm { x[i] += packed[i]; }
                    conv[b].copy_from_slice(&packed[dm..dm + csl]);
                    ssm[b].copy_from_slice(&packed[dm + csl..]);
                }
                let fnl = rms_norm(&x, &output_norm, cfg.eps);
                last = qmm_1d(&fnl, &lm_head);
            }
            last
        };

        let lf = run(false); // f32 body — ground truth
        let lq = run(true);  // fq4 body — converted artifact
        let argmax = |v: &[f32]| {
            v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
        };
        let (af, aq) = (argmax(&lf), argmax(&lq));
        let top = |v: &[f32]| {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&i, &j| v[j].partial_cmp(&v[i]).unwrap());
            idx.truncate(5);
            idx.sort_unstable();
            idx
        };
        let (tf, tq) = (top(&lf), top(&lq));
        let ov = tf.iter().filter(|i| tq.contains(i)).count();
        let maxd = lf.iter().zip(lq.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let dot: f64 = lf.iter().zip(lq.iter()).map(|(a, b)| *a as f64 * *b as f64).sum();
        let nf: f64 = lf.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let nq: f64 = lq.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (nf * nq);
        eprintln!(
            "brick6c END-TO-END [{} layers, {} tok, greedy]: \
             f32-body argmax={af} fq4-body argmax={aq}  top5_overlap={ov}/5  \
             max_logit_abs={maxd:.4}  cosine={cos:.6}",
            nl, prompt.len()
        );
        assert_eq!(
            af, aq,
            "fq4 BODY changes the generated token vs f32 end-to-end \
             (compounded over {nl} layers) — a REAL finding: the body \
             quant is too lossy for generation; higher-precision body or \
             revisit fq4 geometry. (top5={ov}/5 cos={cos:.4})"
        );
        assert!(cos > 0.999, "fq4-body logits drifted too far: cosine {cos:.6}");
    }
}
