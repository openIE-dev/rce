//! Bamba inference runtime on the rce substrate.
//!
//! IBM's [Bamba-9B](https://huggingface.co/ibm-ai-platform/Bamba-9B-v1)
//! (May 2026) — a hybrid Mamba-2 + transformer model with the Llama-3
//! ChatML tokenizer. Architecturally similar to Nemotron-H, but
//! per-layer structure is:
//!
//! ```text
//! x = mixer(rms_norm(x, attn_norm)) + x         # SSM or Attn
//! x = SwiGLU_FFN(rms_norm(x, ffn_norm)) + x      # always
//! ```
//!
//! 32 layers split 29 SSM + 3 Attn (mixer roles). **Every layer**
//! also has a gated SwiGLU FFN — that's the Llama-derived shape, vs
//! Nemotron-H where FFN replaces the mixer in dedicated FFN-only
//! layers.
//!
//! Mamba-2 SSM dims: d_model=4096, d_inner=8192 (= 2×expansion), 128
//! heads × head_dim=64, d_state=128, ngroups=1, d_conv=4. Attention:
//! 32 Q heads, 8 KV heads (GQA 4:1), d_head=128, partial RoPE (first
//! 64 of 128 dims). Vocab = 128256 (Llama-3 BPE).

use rce_ingest_gguf::{
    read_tensor_f32, sample_with_history, GgufFile, GgufValue, Tokenizer,
};
pub use rce_ingest_gguf::SampleConfig;
use rce_op_conv1d::conv1d_causal_step;
use rce_op_qmm::QuantWeight;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

// ── Config ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct BambaConfig {
    pub d_model: usize,
    pub vocab: usize,
    pub n_layers: usize,
    // Attention.
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub d_head: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    // SSM (Mamba-2).
    pub ssm_n_heads: usize,
    pub ssm_head_dim: usize,
    pub ssm_d_state: usize,
    pub ssm_n_groups: usize,
    pub ssm_d_conv: usize,
    // FFN (SwiGLU).
    pub d_inner_ffn: usize,
    pub eps: f32,
}

impl BambaConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf.metadata.iter()
            .find(|kv| kv.key == "general.architecture")
            .and_then(|kv| match &kv.value {
                GgufValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .ok_or("missing general.architecture")?;
        // Accept Bamba and Granite-4 (IBM, May 2026) — both Mamba-2 +
        // Transformer hybrids with identical tensor naming
        // (`blk.N.ssm_*`, `blk.N.attn_*`). Granite uses 9:1 SSM:Attn
        // ratio; Bamba uses its own ratio. Layer-type detection here
        // is tensor-presence-based so both ratios just work.
        let arch_owned = arch.to_string();
        let prefix: &str = match arch_owned.as_str() {
            "bamba" => "bamba",
            "granite" | "granitehybrid" | "granitemoehybrid" => &arch_owned,
            other => return Err(format!(
                "expected bamba|granite|granitehybrid|granitemoehybrid, got {other}"
            )),
        };

        let prefix_owned = prefix.to_string();
        let u32_meta = |k: &str| -> Option<u32> {
            gguf.metadata.iter().find(|kv| kv.key == format!("{prefix_owned}.{k}"))
                .and_then(|kv| match &kv.value { GgufValue::U32(v) => Some(*v), _ => None })
        };
        let f32_meta = |k: &str| -> Option<f32> {
            gguf.metadata.iter().find(|kv| kv.key == format!("{prefix_owned}.{k}"))
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
        let n_q_heads = u32_meta("attention.head_count").ok_or("attention.head_count")? as usize;
        let n_kv_heads = u32_meta("attention.head_count_kv").unwrap_or(n_q_heads as u32) as usize;
        let d_head = u32_meta("attention.key_length")
            .or_else(|| {
                let dm = d_model as u32;
                if n_q_heads > 0 { Some(dm / n_q_heads as u32) } else { None }
            })
            .ok_or("attention.key_length")? as usize;
        let rope_dim = u32_meta("rope.dimension_count").unwrap_or(d_head as u32) as usize;
        let rope_freq_base = f32_meta("rope.freq_base").unwrap_or(500_000.0);
        let eps = f32_meta("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let d_inner_ffn = u32_meta("feed_forward_length").ok_or("feed_forward_length")? as usize;
        let ssm_d_inner = u32_meta("ssm.inner_size").unwrap_or((2 * d_model) as u32) as usize;
        let ssm_d_state = u32_meta("ssm.state_size").unwrap_or(128) as usize;
        let ssm_n_groups = u32_meta("ssm.group_count").unwrap_or(1) as usize;
        let ssm_d_conv = u32_meta("ssm.conv_kernel").unwrap_or(4) as usize;

        // Derive ssm_n_heads from `ssm_a` shape: 1-D vector of length n_heads
        // (stored as `[1, n_heads]` in GGUF dims).
        let ssm_n_heads = gguf.tensors.iter()
            .find(|t| t.name.starts_with("blk.") && t.name.ends_with(".ssm_a"))
            .and_then(|t| t.dims.iter().copied().find(|&d| d > 1))
            .map(|d| d as usize)
            .ok_or("could not infer ssm_n_heads from blk.*.ssm_a")?;
        if ssm_d_inner % ssm_n_heads != 0 {
            return Err(format!("ssm_d_inner ({ssm_d_inner}) not divisible by ssm_n_heads ({ssm_n_heads})"));
        }
        let ssm_head_dim = ssm_d_inner / ssm_n_heads;

        Ok(BambaConfig {
            d_model, vocab, n_layers,
            n_q_heads, n_kv_heads, d_head, rope_dim, rope_freq_base,
            ssm_n_heads, ssm_head_dim, ssm_d_state, ssm_n_groups, ssm_d_conv,
            d_inner_ffn, eps,
        })
    }

    pub fn ssm_d_inner(&self) -> usize { self.ssm_n_heads * self.ssm_head_dim }
    pub fn ssm_conv_dim(&self) -> usize {
        self.ssm_d_inner() + 2 * self.ssm_n_groups * self.ssm_d_state
    }
    pub fn ssm_in_proj_out(&self) -> usize {
        self.ssm_d_inner() + self.ssm_conv_dim() + self.ssm_n_heads
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixerKind { Ssm, Attn }

// ── Weights ────────────────────────────────────────────────────────

struct SsmMixer {
    ssm_in: QuantWeight,     // [ssm_in_proj_out, d_model]
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    a_log: Vec<f32>,
    d_param: Vec<f32>,
    dt_bias: Vec<f32>,
    ssm_norm_w: Vec<f32>,
    ssm_out: QuantWeight,    // [d_model, d_inner]
}

struct AttnMixer {
    q: QuantWeight,          // [n_q*d_head, d_model]
    k: QuantWeight,          // [n_kv*d_head, d_model]
    v: QuantWeight,
    o: QuantWeight,          // [d_model, n_q*d_head]
}

enum Mixer {
    Ssm(Box<SsmMixer>),
    Attn(Box<AttnMixer>),
}

struct Block {
    attn_norm_w: Vec<f32>,
    ffn_norm_w: Vec<f32>,
    gate: QuantWeight,       // [d_inner_ffn, d_model]
    up: QuantWeight,
    down: QuantWeight,       // [d_model, d_inner_ffn]
    mixer: Mixer,
}

impl Block {
    fn mixer_kind(&self) -> MixerKind {
        match self.mixer {
            Mixer::Ssm(_) => MixerKind::Ssm,
            Mixer::Attn(_) => MixerKind::Attn,
        }
    }
}

struct BambaWeights {
    token_embd: QuantWeight,
    output_norm_w: Vec<f32>,
    lm_head: QuantWeight,    // [vocab, d_model]
    blocks: Vec<Block>,
}

// ── State ──────────────────────────────────────────────────────────

enum MixerState {
    Ssm { conv_state: Vec<f32>, ssm_state: Vec<f32> },
    Attn { k_cache: Vec<f32>, v_cache: Vec<f32> },
}

pub struct BambaState {
    mixers: Vec<MixerState>,
    pos: usize,
}

impl BambaState {
    fn new(cfg: &BambaConfig, kinds: &[MixerKind]) -> Self {
        let mixers = kinds.iter().map(|kind| match kind {
            MixerKind::Ssm => MixerState::Ssm {
                conv_state: vec![0.0f32; cfg.ssm_conv_dim() * cfg.ssm_d_conv],
                ssm_state: vec![0.0f32; cfg.ssm_d_inner() * cfg.ssm_d_state],
            },
            MixerKind::Attn => MixerState::Attn {
                k_cache: Vec::new(),
                v_cache: Vec::new(),
            },
        }).collect();
        Self { mixers, pos: 0 }
    }
}

// ── Loading ────────────────────────────────────────────────────────

fn load_weights(
    mmap: &Arc<Mmap>, gguf: &GgufFile, cfg: &BambaConfig,
) -> Result<(BambaWeights, Vec<MixerKind>), String> {
    let bytes: &[u8] = &mmap[..];
    let token_embd = QuantWeight::from_gguf(Arc::clone(mmap), gguf, "token_embd.weight", cfg.vocab, cfg.d_model)
        .map_err(|e| format!("token_embd: {e}"))?;
    let output_norm_w = read_tensor_f32(bytes, gguf, "output_norm.weight")
        .map_err(|e| format!("output_norm: {e:?}"))?;
    let lm_head = QuantWeight::from_gguf(Arc::clone(mmap), gguf, "output.weight", cfg.vocab, cfg.d_model)
        .or_else(|_| QuantWeight::from_gguf(Arc::clone(mmap), gguf, "token_embd.weight", cfg.vocab, cfg.d_model))
        .map_err(|e| format!("output: {e}"))?;

    let mut blocks = Vec::with_capacity(cfg.n_layers);
    let mut kinds = Vec::with_capacity(cfg.n_layers);
    for b in 0..cfg.n_layers {
        let bn = format!("blk.{b}");
        let has_ssm = gguf.tensors.iter().any(|t| t.name == format!("{bn}.ssm_in.weight"));
        let has_attn = gguf.tensors.iter().any(|t| t.name == format!("{bn}.attn_q.weight"));

        let attn_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.attn_norm.weight"))
            .map_err(|e| format!("blk.{b}.attn_norm: {e:?}"))?;
        let ffn_norm_w = read_tensor_f32(bytes, gguf, &format!("{bn}.ffn_norm.weight"))
            .map_err(|e| format!("blk.{b}.ffn_norm: {e:?}"))?;

        // FFN (SwiGLU) — every layer. Native dtype, no transpose.
        let gate = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.ffn_gate.weight"), cfg.d_inner_ffn, cfg.d_model)
            .map_err(|e| format!("blk.{b}.ffn_gate: {e}"))?;
        let up = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.ffn_up.weight"), cfg.d_inner_ffn, cfg.d_model)
            .map_err(|e| format!("blk.{b}.ffn_up: {e}"))?;
        let down = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.ffn_down.weight"), cfg.d_model, cfg.d_inner_ffn)
            .map_err(|e| format!("blk.{b}.ffn_down: {e}"))?;

        let mixer = if has_ssm {
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
            kinds.push(MixerKind::Ssm);
            Mixer::Ssm(Box::new(SsmMixer {
                ssm_in, conv_w, conv_b, a_log, d_param, dt_bias, ssm_norm_w, ssm_out,
            }))
        } else if has_attn {
            let q = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.attn_q.weight"), cfg.n_q_heads * cfg.d_head, cfg.d_model)
                .map_err(|e| format!("blk.{b}.attn_q: {e}"))?;
            let k = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.attn_k.weight"), cfg.n_kv_heads * cfg.d_head, cfg.d_model)
                .map_err(|e| format!("blk.{b}.attn_k: {e}"))?;
            let v = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.attn_v.weight"), cfg.n_kv_heads * cfg.d_head, cfg.d_model)
                .map_err(|e| format!("blk.{b}.attn_v: {e}"))?;
            let o = QuantWeight::from_gguf(Arc::clone(mmap), gguf, &format!("{bn}.attn_output.weight"), cfg.d_model, cfg.n_q_heads * cfg.d_head)
                .map_err(|e| format!("blk.{b}.attn_output: {e}"))?;
            kinds.push(MixerKind::Attn);
            Mixer::Attn(Box::new(AttnMixer { q, k, v, o }))
        } else {
            return Err(format!("blk.{b}: neither ssm nor attn weights found"));
        };

        blocks.push(Block { attn_norm_w, ffn_norm_w, gate, up, down, mixer });
    }
    Ok((BambaWeights { token_embd, output_norm_w, lm_head, blocks }, kinds))
}

// ── Runtime ────────────────────────────────────────────────────────

pub struct BambaRuntime {
    cfg: BambaConfig,
    gguf: GgufFile,
    weights: BambaWeights,
    kinds: Vec<MixerKind>,
}

impl BambaRuntime {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| format!("open: {e}"))?;
        let mmap = Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?);
        // Hint sequential reads during the load-time weight scan; ignore
        // errors (kernel may refuse). See rce_ingest_gguf::mmap_load_sequential.
        let _ = mmap.advise(memmap2::Advice::WillNeed);
        let gguf = GgufFile::parse(&mmap[..]).map_err(|e| format!("parse: {e:?}"))?;
        let cfg = BambaConfig::from_gguf(&gguf)?;
        eprintln!(
            "bamba: d_model={} vocab={} n_layers={} n_q_heads={} n_kv_heads={} d_head={} rope_dim={} ssm_n_heads={} ssm_head_dim={} d_state={} ngroups={} d_inner_ffn={}",
            cfg.d_model, cfg.vocab, cfg.n_layers, cfg.n_q_heads, cfg.n_kv_heads, cfg.d_head, cfg.rope_dim,
            cfg.ssm_n_heads, cfg.ssm_head_dim, cfg.ssm_d_state, cfg.ssm_n_groups, cfg.d_inner_ffn,
        );
        let (weights, kinds) = load_weights(&mmap, &gguf, &cfg)?;
        let ssm_n = kinds.iter().filter(|k| **k == MixerKind::Ssm).count();
        let attn_n = kinds.iter().filter(|k| **k == MixerKind::Attn).count();
        eprintln!("bamba: {ssm_n} SSM mixers + {attn_n} attn mixers across {} layers", cfg.n_layers);
        // mmap kept alive transitively via QuantWeight Arc clones.
        Ok(Self { cfg, gguf, weights, kinds })
    }

    pub fn config(&self) -> &BambaConfig { &self.cfg }
    pub fn mixer_kinds(&self) -> &[MixerKind] { &self.kinds }
    pub fn tokenizer(&self) -> Tokenizer<'_> { Tokenizer::from_gguf(&self.gguf).expect("tokenizer") }
    pub fn new_state(&self) -> BambaState { BambaState::new(&self.cfg, &self.kinds) }

    pub fn forward_logits(&self, state: &mut BambaState, token_id: i32) -> Vec<f32> {
        let logits = forward_token(&self.weights, &self.cfg, &self.kinds, state, token_id);
        state.pos += 1;
        logits
    }
}

fn forward_token(
    w: &BambaWeights, cfg: &BambaConfig, _kinds: &[MixerKind],
    state: &mut BambaState, token_id: i32,
) -> Vec<f32> {
    let dm = cfg.d_model;
    let mut x: Vec<f32> = w.token_embd.dequant_row(token_id as usize);
    let pos = state.pos;

    let bypass_attn = std::env::var("RCE_BAMBA_BYPASS_ATTN").is_ok();
    for (b, block) in w.blocks.iter().enumerate() {
        // Mixer (SSM or Attn) with pre-norm + residual.
        let mixer_normed = rms_norm(&x, &block.attn_norm_w, cfg.eps);
        let mixer_out = match (&block.mixer, &mut state.mixers[b]) {
            (Mixer::Ssm(s), MixerState::Ssm { conv_state, ssm_state }) => {
                ssm_mixer(&mixer_normed, s, conv_state, ssm_state, cfg)
            }
            (Mixer::Attn(a), MixerState::Attn { k_cache, v_cache }) => {
                if bypass_attn {
                    // Diagnostic: attention as identity passthrough (residual only,
                    // no delta). Localises whether garbage is from attn vs SSM.
                    let _ = (a, k_cache, v_cache);
                    vec![0.0f32; cfg.d_model]
                } else {
                    attn_mixer(&mixer_normed, a, k_cache, v_cache, pos, cfg)
                }
            }
            _ => unreachable!("mixer/state kind mismatch at block {b}"),
        };
        for i in 0..dm { x[i] += mixer_out[i]; }

        // FFN (SwiGLU) with pre-norm + residual — every layer.
        let ffn_normed = rms_norm(&x, &block.ffn_norm_w, cfg.eps);
        let ffn_out = swiglu_ffn(&ffn_normed, block, cfg);
        for i in 0..dm { x[i] += ffn_out[i]; }
    }

    let final_normed = rms_norm(&x, &w.output_norm_w, cfg.eps);
    qmm_1d(&final_normed, &w.lm_head)
}

// ── SSM mixer (Mamba-2) ────────────────────────────────────────────

fn ssm_mixer(
    x_in: &[f32], s: &SsmMixer, conv_state: &mut Vec<f32>, ssm_state: &mut Vec<f32>,
    cfg: &BambaConfig,
) -> Vec<f32> {
    let dm = cfg.d_model;
    let d_inner = cfg.ssm_d_inner();
    let conv_dim = cfg.ssm_conv_dim();
    let nh = cfg.ssm_n_heads;
    let hd = cfg.ssm_head_dim;
    let ds = cfg.ssm_d_state;
    let ng = cfg.ssm_n_groups;

    let proj = qmm_1d(x_in, &s.ssm_in);
    let z = &proj[0..d_inner];
    let xbc = &proj[d_inner..d_inner + conv_dim];
    let dt_raw = &proj[d_inner + conv_dim..d_inner + conv_dim + nh];

    let (xbc_conv, new_conv_state) = conv1d_causal_step(
        xbc, &s.conv_w, &s.conv_b, conv_state, conv_dim, cfg.ssm_d_conv,
    );
    *conv_state = new_conv_state;

    let xbc_act: Vec<f32> = xbc_conv.iter().map(|v| silu(*v)).collect();
    let x_act = &xbc_act[0..d_inner];
    let b_act = &xbc_act[d_inner..d_inner + ng * ds];
    let c_act = &xbc_act[d_inner + ng * ds..conv_dim];

    let (y, new_ssm) = selective_scan_f64(
        x_act, b_act, c_act, dt_raw, &s.dt_bias, &s.a_log, &s.d_param, ssm_state,
        nh, hd, ds, ng,
    );
    *ssm_state = new_ssm;

    let mut gated = vec![0.0f32; d_inner];
    for i in 0..d_inner { gated[i] = silu(z[i]) * y[i]; }
    let normed = group_rms_norm(&gated, &s.ssm_norm_w, ng, cfg.eps);
    qmm_1d(&normed, &s.ssm_out)
}

fn selective_scan_f64(
    x: &[f32], b: &[f32], c: &[f32], dt_raw: &[f32], dt_bias: &[f32],
    a: &[f32], d_param: &[f32], state_in: &[f32],
    nheads: usize, headdim: usize, d_state: usize, ngroups: usize,
) -> (Vec<f32>, Vec<f32>) {
    let d_inner = nheads * headdim;
    let nheads_per_group = nheads / ngroups;
    let mut y = vec![0.0f32; d_inner];
    let mut state_out: Vec<f64> = state_in.iter().map(|&v| v as f64).collect();
    let state_per_head = headdim * d_state;
    for h in 0..nheads {
        let group = h / nheads_per_group;
        let dt_pre = dt_raw[h] as f64 + dt_bias[h] as f64;
        let dt_h = if dt_pre > 20.0 { dt_pre } else { (1.0 + dt_pre.exp()).ln() };
        // GGUF `ssm_a` is already `A = -exp(A_log)` — see
        // `project_gguf_ssm_a_convention.md`. Use directly.
        let a_h = a[h] as f64;
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

// ── Attention (RoPE + GQA, Llama-style) ────────────────────────────

fn attn_mixer(
    x_in: &[f32], a: &AttnMixer, k_cache: &mut Vec<f32>, v_cache: &mut Vec<f32>,
    pos: usize, cfg: &BambaConfig,
) -> Vec<f32> {
    let dm = cfg.d_model;
    let dh = cfg.d_head;
    let nq = cfg.n_q_heads;
    let nkv = cfg.n_kv_heads;
    let q_dim = nq * dh;
    let kv_dim = nkv * dh;

    let mut q = qmm_1d(x_in, &a.q);
    let mut k_new = qmm_1d(x_in, &a.k);
    let v_new = qmm_1d(x_in, &a.v);

    // Partial NeoX RoPE: rotate first `rope_dim` dims of each head.
    apply_rope_inplace(&mut q, nq, dh, cfg.rope_dim, cfg.rope_freq_base, pos);
    apply_rope_inplace(&mut k_new, nkv, dh, cfg.rope_dim, cfg.rope_freq_base, pos);

    k_cache.extend_from_slice(&k_new);
    v_cache.extend_from_slice(&v_new);
    let seq_len = k_cache.len() / kv_dim;

    let scale = 1.0 / (dh as f32).sqrt();
    let mut attended = vec![0.0f32; q_dim];
    let q_per_kv = nq / nkv;
    // NEON-vectorized fused attention via rce-op-attn.
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

fn apply_rope_inplace(x: &mut [f32], n_heads: usize, d_head: usize, rope_dim: usize, base: f32, pos: usize) {
    for h in 0..n_heads {
        let head_off = h * d_head;
        rce_op_attn::rope_inplace(&mut x[head_off..head_off + d_head], rope_dim, base, pos);
    }
}

// ── FFN (SwiGLU) ───────────────────────────────────────────────────

fn swiglu_ffn(x_in: &[f32], block: &Block, cfg: &BambaConfig) -> Vec<f32> {
    let dm = cfg.d_model;
    let dff = cfg.d_inner_ffn;
    let g = qmm_1d(x_in, &block.gate);
    let u = qmm_1d(x_in, &block.up);
    let mut h = vec![0.0f32; dff];
    for i in 0..dff { h[i] = silu(g[i]) * u[i]; }
    qmm_1d(&h, &block.down)
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

// ── Generation ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct BambaGenConfig {
    pub max_new_tokens: usize,
    pub sample: SampleConfig,
    pub rep_window: usize,
}

impl Default for BambaGenConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            sample: SampleConfig::chat(0xC0DE_F00D_2026),
            rep_window: 64,
        }
    }
}

impl BambaRuntime {
    /// Base-LM text continuation (no chat template). Use this for the
    /// Bamba-9B-v1 base model, which was pretrained on raw text and
    /// **never saw the Llama-3 ChatML template** — wrapping prompts in
    /// `<|start_header_id|>user<|end_header_id|>…` produces fluent but
    /// topically random output. The Bamba-9B-instruct variant (when it
    /// lands) should use `chat()` instead.
    pub fn generate(
        &self,
        prompt: &str,
        cfg: &BambaGenConfig,
        mut on_token: impl FnMut(&str),
    ) -> Result<String, String> {
        let tk = self.tokenizer();
        // Llama-3 BPE: prepend BOS, no chat template.
        let mut prompt_ids = vec![tk.bos_token_id.unwrap_or(128000)];
        prompt_ids.extend(tk.encode_gpt2(prompt, false));
        let eos_id = tk.eos_token_id.unwrap_or(-1);

        let mut state = self.new_state();
        let mut history: Vec<i32> = Vec::with_capacity(prompt_ids.len() + cfg.max_new_tokens);
        let mut step: u64 = 0;
        let seed_for = |s: u64| {
            let mut c = cfg.sample;
            c.seed = cfg.sample.seed ^ s.wrapping_mul(0x9E37_79B9);
            c
        };

        let mut next: i32 = -1;
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

    /// Multi-turn chat using the Llama-3 ChatML template:
    /// `<|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>...`.
    /// Stops on `<|eot_id|>` or `eos_token_id`.
    pub fn chat(
        &self,
        messages: &[(&str, &str)],
        cfg: &BambaGenConfig,
        mut on_token: impl FnMut(&str),
    ) -> Result<String, String> {
        let tk = self.tokenizer();
        let prompt = format_chat_llama3(&tk, messages)
            .ok_or("model lacks Llama-3 chat special tokens")?;
        let eos_id = tk.eos_token_id.unwrap_or(-1);
        let eot_id = tk.lookup_token("<|eot_id|>").unwrap_or(-1);

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
            if next == eot_id || next == eos_id {
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

/// Llama-3 instruct chat template:
/// ```text
/// <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>
/// <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
/// <|start_header_id|>assistant<|end_header_id|>\n\n
/// ```
fn format_chat_llama3(tk: &Tokenizer, messages: &[(&str, &str)]) -> Option<Vec<i32>> {
    let start_header = tk.lookup_token("<|start_header_id|>")?;
    let end_header = tk.lookup_token("<|end_header_id|>")?;
    let eot = tk.lookup_token("<|eot_id|>")?;
    let mut out = Vec::new();
    if let Some(bos) = tk.bos_token_id {
        out.push(bos);
    }
    for (role, content) in messages {
        out.push(start_header);
        out.extend(tk.encode_gpt2(role, false));
        out.push(end_header);
        out.extend(tk.encode_gpt2("\n\n", false));
        out.extend(tk.encode_gpt2(content, false));
        out.push(eot);
    }
    out.push(start_header);
    out.extend(tk.encode_gpt2("assistant", false));
    out.push(end_header);
    out.extend(tk.encode_gpt2("\n\n", false));
    Some(out)
}
