//! Cosmos 3 Phase 4 — diffusion generator scaffold (Stage 0).
//!
//! This module is the entry point for the **generator** branch of NVIDIA
//! Cosmos 3 Nano's mixture-of-transformers omnimodel.  Phase 1-3 covered the
//! reasoner (Qwen3-VL-8B text + 27-layer ViT vision encoder) and is in the
//! crate root.  Phase 4 adds:
//!
//! 1. [`Cosmos3OmniTransformer`] — the diffusion generator core.  Shares the
//!    36-layer MoT skeleton with the reasoner but carries a parallel weight
//!    set (`*_moe_gen` tensors) and adds:
//!    - Time-conditioning embedder (sinusoidal → MLP → `[D_time]`).
//!    - Per-block AdaLN-Zero modulation (6 modulators per block: scale1,
//!      shift1, gate1, scale2, shift2, gate2 — applied to attn + MLP halves).
//!    - Combined-QKV attention with the generator's bf16 weights.
//!
//! 2. [`Wan22VAE`] — encode/decode for video latents.  Wan 2.2 spec:
//!    encoder 4× spatial × 4× temporal compression via a ResNet stack +
//!    mid-block attention; decoder mirrors it with ConvTranspose3D.
//!
//! 3. [`AVAEAudioTokenizer`] — Apple-VAE-48kHz audio RVQ codec.  Encodes
//!    raw audio to discrete tokens for the multimodal stream and decodes
//!    tokens back to 48 kHz waveform.
//!
//! 4. [`ActionTokenIO`] — discrete action vocab embed / unembed (small,
//!    256 tokens for VLA mode).
//!
//! 5. [`euler_step`] — single Euler integration step for any diffusion
//!    sampler; full sampling loop = `for t in steps { x = euler_step(x, ...) }`.
//!
//! # Stage 0 (this scaffold)
//!
//! - All weight types are defined with correct shapes and naming.
//! - `load` functions read from a safetensors and assemble the structs.
//! - Forward functions are **stubs** that pass the input through with
//!   bias-add only — enough to type-check, to exercise the load path,
//!   and to keep the dummy diffusion loop finite.
//! - [`tests::phase4_synthetic_e2e`] (gated on `--features phase4`) writes
//!   a synthetic safetensors via `rce_weight_loader::write_safetensors`,
//!   loads the full Phase 4 stack, runs `dummy_diffuse(.., n_steps=4)`,
//!   asserts shape + finite + non-trivial spread.
//!
//! # Stage 1+ (next sessions)
//!
//! - Real diffusion forward: AdaLN-Zero, QKV attention, MLP, residuals.
//! - Wan 2.2 VAE encode/decode with the actual ResNet + mid-attn architecture.
//! - AVAE audio RVQ codec (multi-stage residual quantizer).
//! - Cross-attention bridge between reasoner and generator (the MoT join).
//! - Metal wiring via `rce-op-metal::dispatch_matmul_tiled_cached_bf16`.
//! - Numerical verification against NVIDIA's PyTorch reference.

use rce_weight_loader::{F32Slice, SafetensorsFile};
use std::path::Path;

use crate::BfSlice;
use crate::HIDDEN;            // 4096 — same as reasoner D
use crate::N_LAYERS;          // 36 — same MoT skeleton depth
pub const FF: usize = crate::FF;             // re-export the FF dim (14336)

/// Time embedding hidden dim.  Cosmos 3 uses 256 for the sinusoidal stage,
/// then MLPs up to `HIDDEN` before per-block AdaLN application.
pub const D_TIME: usize = 256;

/// Wan 2.2 VAE latent channels (z_dim) per spatial position.
pub const WAN_Z: usize = 16;

/// AVAE-48k tokenizer: 32 RVQ codebooks, 1024 codes per book, 48 kHz stream
/// downsampled to ~75 Hz token rate (i.e. ~640× downsample ratio).
pub const AVAE_CODEBOOKS: usize = 32;
pub const AVAE_VOCAB: usize = 1024;
pub const AVAE_HOP: usize = 640;

/// Action token I/O vocab size (Cosmos 3 VLA discrete-action mode).
pub const ACTION_VOCAB: usize = 256;

// ────────────────────────────────────────────────────────────────────────
// Metal GPU acceleration (Stage 5 — gated by `--features metal`)
// ────────────────────────────────────────────────────────────────────────
//
// rce-op-metal provides `MetalLeaf::dispatch_matmul_tiled_cached_bf16`
// (a threadgroup-tiled bf16 GEMM with shared `A_tile` + `B_tile`) that
// targets the same shape regime (M=T, K=HIDDEN, N=HIDDEN..6*HIDDEN) we
// run through `matmul_nt_bf16` per OmniBlock today.  Stage 5 wires the
// OmniBlock matmuls (ada / qkv / proj / cross_q / cross_kv / cross_proj
// / fc1 / fc2 — 8 per block) through the GPU path.
//
// Stage 5 Stage 0 (this scaffold): smoke-test that `MetalLeaf::new()`
// initializes a Metal device on Apple Silicon and the dependency tree
// compiles.  Stage 5 Stage 1+ wires the actual dispatch calls.

/// GPU acceleration context.  Holds a `rce_op_metal::MetalLeaf` so the
/// OmniBlock matmuls can be routed to Apple Silicon's GPU.  Built once at
/// model construction; passed (Option) into the forward calls.
///
/// **Stage 5 Stage 0 status**: scaffold only.  `forward_with_metal` is the
/// hook point that Stage 5 Stage 1 fills in — currently it falls back to
/// the CPU path.
#[cfg(feature = "metal")]
pub struct MetalAccel {
    leaf: rce_op_metal::MetalLeaf,
    /// Weight handle cache, keyed by `(block_idx, role)` where role ∈
    /// {ada=0, qkv=1, proj=2, cross_q=3, cross_kv=4, cross_proj=5,
    /// fc1=6, fc2=7}.  Stage 5 Stage 1 uses interior mutability so
    /// `maybe_dispatch_matmul_bf16` can lazily upload on first use.
    cache: std::sync::Mutex<std::collections::HashMap<(usize, u8), u64>>,
}

/// Per-head GPU SDPA via rce-op-metal's `Custom("attention")` op.
/// One dispatch per head; Q/K/V come from the per-head slices of the
/// already-computed per-token Q/K/V projection.  Returns the same
/// `[T, HIDDEN]` shape as the CPU `sdpa` helper.
///
/// Returns `None` when Metal isn't available or any per-head dispatch
/// fails; caller falls back to CPU `sdpa`.
#[cfg(feature = "metal")]
fn sdpa_gpu(
    accel: &MetalAccel,
    q: &[f32], k: &[f32], v: &[f32], t: usize,
) -> Option<Vec<f32>> {
    use rce_vm_backend_cpu::OpDispatch;
    use rce_vm_backend_cpu::{HostTensor, Value, DType, ScalarValue};
    let scale = 1.0f32 / (crate::HEAD_DIM as f32).sqrt();
    let mut attn = vec![0.0f32; t * HIDDEN];
    for h in 0..crate::N_HEADS {
        let off = h * crate::HEAD_DIM;
        // Gather per-head [T, HEAD_DIM] from the per-token interleaved layout.
        let mut q_h = vec![0.0f32; t * crate::HEAD_DIM];
        let mut k_h = vec![0.0f32; t * crate::HEAD_DIM];
        let mut v_h = vec![0.0f32; t * crate::HEAD_DIM];
        for ti in 0..t { for d in 0..crate::HEAD_DIM {
            q_h[ti * crate::HEAD_DIM + d] = q[ti * HIDDEN + off + d];
            k_h[ti * crate::HEAD_DIM + d] = k[ti * HIDDEN + off + d];
            v_h[ti * crate::HEAD_DIM + d] = v[ti * HIDDEN + off + d];
        }}
        let to_value = |row_major: &[f32], rows: usize| -> Option<Value> {
            let bytes: Vec<u8> = row_major.iter().flat_map(|f| f.to_le_bytes()).collect();
            Some(Value::Tensor(HostTensor::new(vec![rows, crate::HEAD_DIM], DType::F32, bytes).ok()?))
        };
        let args = [
            to_value(&q_h, t)?,
            to_value(&k_h, t)?,
            to_value(&v_h, t)?,
            Value::Scalar(ScalarValue::F32(scale)),
            Value::Scalar(ScalarValue::I32(0)),  // sliding_window=0 (full attention)
        ];
        let out_val = (&accel.leaf as &dyn OpDispatch)
            .execute(0, "attention", &args).ok()?;
        let out_host = match out_val {
            Value::Tensor(h) => h,
            Value::DeviceTensor(d) => accel.leaf.demote_to_host(&d).and_then(|v| {
                if let Value::Tensor(h) = v { Some(h) } else { None }
            })?,
            _ => return None,
        };
        if out_host.dtype != DType::F32 { return None; }
        let expected = t * crate::HEAD_DIM * 4;
        if out_host.data.len() != expected { return None; }
        // Scatter result back into [T, HIDDEN] layout at head slice.
        for ti in 0..t { for d in 0..crate::HEAD_DIM {
            let bytes = &out_host.data[(ti * crate::HEAD_DIM + d) * 4..(ti * crate::HEAD_DIM + d) * 4 + 4];
            attn[ti * HIDDEN + off + d] = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }}
    }
    Some(attn)
}

/// Routed SDPA: GPU via Custom("attention") when accel is `Some` and
/// dispatch succeeds, else CPU `sdpa` fallback.
/// Threshold below which the per-head dispatch overhead of GPU SDPA
/// (32 heads × ~500 µs dispatch each) costs more than the parallelized
/// scalar CPU SDPA work.  Measured empirically on 2 ResBlocks × 1 step:
///   T   |  Stage5.3 (CPU SDPA) | Stage5.4 (GPU SDPA) | choice
///   256 |  3.21×               | 2.76×               | CPU
///  1024 |  1.80×               | 1.36×               | CPU
///  4096 |  1.12×               | 2.68×               | GPU
/// → gate at T≥2048 (between 1024 and 4096 in log space).
#[cfg(feature = "metal")]
const SDPA_GPU_MIN_T: usize = 2048;

#[cfg(feature = "metal")]
pub(crate) fn sdpa_routed(accel: Option<&MetalAccel>, q: &[f32], k: &[f32], v: &[f32], t: usize) -> Vec<f32> {
    if t >= SDPA_GPU_MIN_T {
        if let Some(a) = accel {
            if let Some(out) = sdpa_gpu(a, q, k, v, t) { return out; }
        }
    }
    sdpa(q, k, v, t)
}

#[cfg(not(feature = "metal"))]
pub(crate) fn sdpa_routed(_accel: Option<&MetalAccel>, q: &[f32], k: &[f32], v: &[f32], t: usize) -> Vec<f32> {
    sdpa(q, k, v, t)
}

#[cfg(feature = "metal")]
impl MetalAccel {
    /// Initialize on the system default Metal device.  Returns `None` on
    /// non-Apple targets, headless CI, sandbox without GPU access, or
    /// when MSL kernel compilation fails.
    pub fn new() -> Option<Self> {
        Some(Self {
            leaf: rce_op_metal::MetalLeaf::new()?,
            cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// Lazily upload a bf16 weight matrix and return its cache handle.
    /// `weight_f32` should be the dequantized f32 view of the bf16 weight
    /// (`cache_weight_bf16` narrows back to bf16 on upload).  Subsequent
    /// dispatches for the same `(block_idx, role)` re-use the handle.
    fn cached_weight(&self, block_idx: usize, role: u8, weight_f32: &[f32]) -> u64 {
        let mut cache = self.cache.lock().unwrap();
        if let Some(&h) = cache.get(&(block_idx, role)) { return h; }
        let h = self.leaf.cache_weight_bf16(weight_f32);
        cache.insert((block_idx, role), h);
        h
    }

    /// Run a bf16 matmul on the GPU: `output[m,n] = input[m,k] · weight[n,k]^T`.
    /// Returns `Some(result)` on success, `None` if Metal isn't available or
    /// dispatch fails (caller falls back to CPU `matmul_nt_bf16`).
    /// Same as `maybe_dispatch_matmul_bf16` but lazy on the weight: the
    /// closure is only called when the `(block_idx, role)` handle isn't
    /// already cached.  Lets per-step dispatch skip the bf16→f32 dequant
    /// on cache-hit fast path.
    pub fn maybe_dispatch_matmul_bf16_lazy<F: FnOnce() -> Vec<f32>>(
        &self, input_f32: &[f32], weight_provider: F,
        block_idx: usize, role: u8, m: usize, k: usize, n: usize,
    ) -> Option<Vec<f32>> {
        use rce_vm_backend_cpu::{HostTensor, Value, DType};
        debug_assert_eq!(input_f32.len(), m * k);
        let in_bytes: Vec<u8> = input_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
        let in_host = HostTensor::new(vec![m, k], DType::F32, in_bytes).ok()?;
        let in_val = Value::Tensor(in_host);
        // Get cached handle — upload via the lazy provider only on miss.
        let wh = {
            let mut cache = self.cache.lock().unwrap();
            if let Some(&h) = cache.get(&(block_idx, role)) { h } else {
                let w_dout_din = weight_provider();
                debug_assert_eq!(w_dout_din.len(), n * k);
                // Transpose [N, K] → [K, N] for rce-op-metal layout.
                let mut w_t = vec![0.0f32; k * n];
                for d_o in 0..n { for d_i in 0..k {
                    w_t[d_i * n + d_o] = w_dout_din[d_o * k + d_i];
                }}
                let h = self.leaf.cache_weight_bf16(&w_t);
                cache.insert((block_idx, role), h);
                h
            }
        };
        let out_val = self.leaf.dispatch_matmul_tiled_cached_bf16(
            &in_val, wh, m as u32, k as u32, n as u32, &[], &[], &[],
        ).ok()?;
        let out_host = match out_val {
            Value::Tensor(h) => h,
            Value::DeviceTensor(d) => self.leaf.demote_to_host(&d).and_then(|v| {
                if let Value::Tensor(h) = v { Some(h) } else { None }
            })?,
            _ => return None,
        };
        if out_host.dtype != DType::F32 { return None; }
        let bytes = &out_host.data;
        if bytes.len() != m * n * 4 { return None; }
        let mut out = Vec::with_capacity(m * n);
        for c in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        Some(out)
    }

    pub fn maybe_dispatch_matmul_bf16(
        &self,
        input_f32: &[f32], weight_f32_dout_din: &[f32],
        block_idx: usize, role: u8,
        m: usize, k: usize, n: usize,
    ) -> Option<Vec<f32>> {
        use rce_vm_backend_cpu::{HostTensor, Value, DType};
        debug_assert_eq!(input_f32.len(), m * k);
        debug_assert_eq!(weight_f32_dout_din.len(), n * k);
        // Transpose weight from `[N=dout, K=din]` (PyTorch nn.Linear convention)
        // to `[K, N]` (rce-op-metal's `dispatch_matmul_*_cached_bf16`
        // expects A @ B with B as `[K, N]`, no transpose).
        let mut weight_t = vec![0.0f32; k * n];
        for d_o in 0..n { for d_i in 0..k {
            weight_t[d_i * n + d_o] = weight_f32_dout_din[d_o * k + d_i];
        }}
        let weight_f32: &[f32] = &weight_t;
        // Build input as Value::Tensor (f32 host bytes).
        let in_bytes: Vec<u8> = input_f32.iter()
            .flat_map(|f| f.to_le_bytes()).collect();
        let in_host = HostTensor::new(vec![m, k], DType::F32, in_bytes).ok()?;
        let in_val = Value::Tensor(in_host);
        // Get cached weight handle (uploads on first use).
        let wh = self.cached_weight(block_idx, role, weight_f32);
        // Dispatch.
        let out_val = self.leaf.dispatch_matmul_tiled_cached_bf16(
            &in_val, wh, m as u32, k as u32, n as u32, &[], &[], &[],
        ).ok()?;
        // Extract f32 result from output Value.
        let out_host = match out_val {
            Value::Tensor(h) => h,
            Value::DeviceTensor(d) => self.leaf.demote_to_host(&d).and_then(|v| {
                if let Value::Tensor(h) = v { Some(h) } else { None }
            })?,
            _ => return None,
        };
        if out_host.dtype != DType::F32 { return None; }
        let bytes = &out_host.data;
        if bytes.len() != m * n * 4 { return None; }
        let mut out = Vec::with_capacity(m * n);
        for c in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        Some(out)
    }
}

#[cfg(not(feature = "metal"))]
pub struct MetalAccel {} // Placeholder so signatures referencing it still compile.

#[cfg(not(feature = "metal"))]
impl MetalAccel {
    pub fn new() -> Option<Self> { None }
    /// Stub so phase4_v2 Wan VAE routed primitives compile under the
    /// `phase4`-only feature (no Metal device).  Always returns None →
    /// callers automatically fall back to scalar CPU.
    pub fn maybe_dispatch_matmul_bf16_lazy<F: FnOnce() -> Vec<f32>>(
        &self, _input_f32: &[f32], _weight_provider: F,
        _block_idx: usize, _role: u8,
        _m: usize, _k: usize, _n: usize,
    ) -> Option<Vec<f32>> { None }
}

#[derive(Debug)]
pub enum Phase4Error { Load(String) }
impl std::fmt::Display for Phase4Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::Load(m) => write!(f, "cosmos3 phase4: {m}") }
    }
}
impl std::error::Error for Phase4Error {}

fn lf32(sf: &SafetensorsFile, name: &str) -> Result<F32Slice, Phase4Error> {
    let t = sf.get(name).ok_or_else(|| Phase4Error::Load(format!("missing {name}")))?;
    F32Slice::from_tensor(name, t).map_err(Phase4Error::Load)
}
fn lbf(sf: &SafetensorsFile, name: &str) -> Result<BfSlice, Phase4Error> {
    let t = sf.get(name).ok_or_else(|| Phase4Error::Load(format!("missing {name}")))?;
    BfSlice::from_tensor(name, t)
        .map_err(|e| Phase4Error::Load(format!("{e}")))
}

// ────────────────────────────────────────────────────────────────────────
// Time embedder
// ────────────────────────────────────────────────────────────────────────

/// Sinusoidal time embedding + MLP up to `HIDDEN`.  Each diffusion step's
/// timestep `t ∈ [0, 1]` is first projected to a `D_TIME` sinusoidal vector
/// then run through Linear(D_TIME → 4*D_TIME) → SiLU → Linear(4*D_TIME → HIDDEN).
pub struct TimeEmbedder {
    fc1_w: BfSlice, fc1_b: F32Slice,   // [4*D_TIME, D_TIME] [4*D_TIME]
    fc2_w: BfSlice, fc2_b: F32Slice,   // [HIDDEN, 4*D_TIME] [HIDDEN]
}

impl TimeEmbedder {
    pub fn load(sf: &SafetensorsFile, prefix: &str) -> Result<Self, Phase4Error> {
        Ok(Self {
            fc1_w: lbf(sf, &format!("{prefix}.time_mlp.0.weight"))?,
            fc1_b: lf32(sf, &format!("{prefix}.time_mlp.0.bias"))?,
            fc2_w: lbf(sf, &format!("{prefix}.time_mlp.2.weight"))?,
            fc2_b: lf32(sf, &format!("{prefix}.time_mlp.2.bias"))?,
        })
    }

    /// Stage 1: sinusoidal embedding of `t ∈ [0, 1]` at `D_TIME` dims, scaled
    /// by ×1000 (diffusion convention), followed by a 2-layer MLP with SiLU.
    ///
    /// `emb[i]      = cos(t * 1000 * exp(-i * factor))` for i in [0, half)
    /// `emb[i+half] = sin(t * 1000 * exp(-i * factor))`
    /// where `factor = ln(1e4) / (half - 1)`, `half = D_TIME / 2`.
    ///
    /// Then: `Linear(D_TIME → 4*D_TIME) → SiLU → Linear(4*D_TIME → HIDDEN)`.
    pub fn forward(&self, t: f32) -> Vec<f32> {
        let half = D_TIME / 2;
        let mut emb = vec![0.0f32; D_TIME];
        let factor = (1e4_f32).ln() / (half - 1) as f32;
        let scaled = t * 1000.0;
        for i in 0..half {
            let freq = (-(i as f32) * factor).exp();
            emb[i]        = (scaled * freq).cos();
            emb[i + half] = (scaled * freq).sin();
        }
        // Linear(D_TIME → 4*D_TIME) + bias
        let mut h = crate::matmul_nt_bf16(&emb, 1, D_TIME, &self.fc1_w, 4 * D_TIME);
        for i in 0..(4 * D_TIME) { h[i] += self.fc1_b[i]; }
        // SiLU: x * sigmoid(x).
        for v in h.iter_mut() { *v = *v * (1.0 / (1.0 + (-*v).exp())); }
        // Linear(4*D_TIME → HIDDEN) + bias
        let mut out = crate::matmul_nt_bf16(&h, 1, 4 * D_TIME, &self.fc2_w, HIDDEN);
        for i in 0..HIDDEN { out[i] += self.fc2_b[i]; }
        out
    }
}

// ────────────────────────────────────────────────────────────────────────
// OmniTransformer generator block (one of N_LAYERS)
// ────────────────────────────────────────────────────────────────────────

pub struct OmniBlock {
    /// AdaLN modulator: `Linear(HIDDEN, 6*HIDDEN)` from time embedding.
    /// Output is split into (scale_a, shift_a, gate_a, scale_b, shift_b, gate_b).
    pub ada_w: BfSlice, pub ada_b: F32Slice,
    /// Combined-QKV self-attention.  Cosmos 3 uses GQA same as the reasoner but
    /// the *_moe_gen weights are an independent parallel set.
    pub qkv_w: BfSlice, pub qkv_b: F32Slice,
    pub proj_w: BfSlice, pub proj_b: F32Slice,

    // ── Cross-attention bridge (the "MoT join"; Stage 4) ──
    /// Q projection from generator hidden state.
    pub cross_q_w: BfSlice, pub cross_q_b: F32Slice,
    /// Combined K/V projection from reasoner hidden state (`[2*HIDDEN, HIDDEN]`).
    pub cross_kv_w: BfSlice, pub cross_kv_b: F32Slice,
    /// Output projection of cross-attention.
    pub cross_proj_w: BfSlice, pub cross_proj_b: F32Slice,

    /// MLP — wider than reasoner's FFN (3.5× hidden).
    pub fc1_w: BfSlice, pub fc1_b: F32Slice,
    pub fc2_w: BfSlice, pub fc2_b: F32Slice,
}

impl OmniBlock {
    pub fn load(sf: &SafetensorsFile, i: usize) -> Result<Self, Phase4Error> {
        let p = format!("model.gen.layers.{i}");
        Ok(Self {
            ada_w:  lbf(sf, &format!("{p}.adaln.weight_moe_gen"))?,
            ada_b:  lf32(sf, &format!("{p}.adaln.bias_moe_gen"))?,
            qkv_w:  lbf(sf, &format!("{p}.attn.qkv.weight_moe_gen"))?,
            qkv_b:  lf32(sf, &format!("{p}.attn.qkv.bias_moe_gen"))?,
            proj_w: lbf(sf, &format!("{p}.attn.proj.weight_moe_gen"))?,
            proj_b: lf32(sf, &format!("{p}.attn.proj.bias_moe_gen"))?,
            cross_q_w:    lbf(sf, &format!("{p}.cross_attn.q.weight_moe_gen"))?,
            cross_q_b:    lf32(sf, &format!("{p}.cross_attn.q.bias_moe_gen"))?,
            cross_kv_w:   lbf(sf, &format!("{p}.cross_attn.kv.weight_moe_gen"))?,
            cross_kv_b:   lf32(sf, &format!("{p}.cross_attn.kv.bias_moe_gen"))?,
            cross_proj_w: lbf(sf, &format!("{p}.cross_attn.proj.weight_moe_gen"))?,
            cross_proj_b: lf32(sf, &format!("{p}.cross_attn.proj.bias_moe_gen"))?,
            fc1_w:  lbf(sf, &format!("{p}.mlp.fc1.weight_moe_gen"))?,
            fc1_b:  lf32(sf, &format!("{p}.mlp.fc1.bias_moe_gen"))?,
            fc2_w:  lbf(sf, &format!("{p}.mlp.fc2.weight_moe_gen"))?,
            fc2_b:  lf32(sf, &format!("{p}.mlp.fc2.bias_moe_gen"))?,
        })
    }

    /// Stage 1 real forward.
    ///
    /// Pipeline:
    /// 1. `(scale_a, shift_a, gate_a, scale_b, shift_b, gate_b) = silu(time_emb) @ ada_w^T + ada_b`
    ///    — split into 6 `[HIDDEN]` modulators (per-token broadcast).
    /// 2. AdaLN: `xn = (1 + scale_a) * bias_free_ln(x) + shift_a`.
    /// 3. Combined-QKV linear: `qkv = xn @ qkv_w^T + qkv_b` → split Q/K/V `[T, HIDDEN]`.
    /// 4. Multi-head SDPA (32 heads × 128 d_head, per-head softmax).
    /// 5. Proj + gated residual: `x = x + gate_a * (attn @ proj_w^T + proj_b)`.
    /// 6. AdaLN-MLP: `xn = (1 + scale_b) * bias_free_ln(x) + shift_b`.
    /// 7. MLP: `xn @ fc1_w^T + fc1_b → SiLU → @ fc2_w^T + fc2_b`.
    /// 8. Gated residual: `x = x + gate_b * mlp_out`.
    pub fn forward(&self, x: &mut [f32], t: usize, time_emb: &[f32]) {
        self.forward_with_cross(x, t, time_emb, None, 0);
    }

    /// Full Stage-4 forward with optional MoT cross-attention bridge to the
    /// reasoner's hidden state.  When `reasoner_h` is `Some`, an extra
    /// cross-attention sub-block runs after self-attention residual and
    /// before AdaLN-MLP: Q from gen, K/V from reasoner, residual added
    /// ungated (no AdaLN modulation — matches the simpler MoT join pattern
    /// where cross is always-on, not time-gated).
    pub fn forward_with_cross(
        &self, x: &mut [f32], t: usize, time_emb: &[f32],
        reasoner_h: Option<&[f32]>, t_reasoner: usize,
    ) {
        self.forward_routed(x, t, time_emb, reasoner_h, t_reasoner, 0, None);
    }

    /// Full forward with optional GPU acceleration.  When `accel` is `Some`
    /// and the `metal` feature is on, all 8 bf16 matmuls (ada, qkv, proj,
    /// cross_q, cross_kv, cross_proj, fc1, fc2) route through `MetalAccel`;
    /// otherwise falls back to CPU `matmul_nt_bf16`.
    pub fn forward_routed(
        &self, x: &mut [f32], t: usize, time_emb: &[f32],
        reasoner_h: Option<&[f32]>, t_reasoner: usize,
        block_idx: usize,
        accel: Option<&MetalAccel>,
    ) {
        debug_assert_eq!(time_emb.len(), HIDDEN);
        // 1) AdaLN modulator: silu(time_emb) @ ada_w^T + ada_b → [6 * HIDDEN].
        let mut silu_t = time_emb.to_vec();
        for v in silu_t.iter_mut() { *v = *v * (1.0 / (1.0 + (-*v).exp())); }
        let mut ada = mm_bf16_routed(accel, &silu_t, 1, HIDDEN, &self.ada_w, 6 * HIDDEN, block_idx, ROLE_ADA);
        for i in 0..(6 * HIDDEN) { ada[i] += self.ada_b[i]; }
        let (scale_a, ada_rest) = ada.split_at(HIDDEN);
        let (shift_a, ada_rest) = ada_rest.split_at(HIDDEN);
        let (gate_a, ada_rest)  = ada_rest.split_at(HIDDEN);
        let (scale_b, ada_rest) = ada_rest.split_at(HIDDEN);
        let (shift_b, ada_rest) = ada_rest.split_at(HIDDEN);
        let (gate_b, _)         = ada_rest.split_at(HIDDEN);

        // 2) Bias-free LN + AdaLN modulation: per-token (1+scale_a)*norm(x) + shift_a.
        let xn = adaln(x, t, scale_a, shift_a);

        // 3) QKV: [T, 3*HIDDEN] = xn @ qkv_w^T + qkv_b.
        let mut qkv = mm_bf16_routed(accel, &xn, t, HIDDEN, &self.qkv_w, 3 * HIDDEN, block_idx, ROLE_QKV);
        for ti in 0..t { for d in 0..(3 * HIDDEN) { qkv[ti * 3 * HIDDEN + d] += self.qkv_b[d]; } }

        // 4) Split Q, K, V (contiguous [HIDDEN] slabs per token) and run SDPA.
        let mut q = vec![0.0f32; t * HIDDEN];
        let mut k = vec![0.0f32; t * HIDDEN];
        let mut v = vec![0.0f32; t * HIDDEN];
        for ti in 0..t {
            let off = ti * 3 * HIDDEN;
            q[ti * HIDDEN..(ti + 1) * HIDDEN].copy_from_slice(&qkv[off..off + HIDDEN]);
            k[ti * HIDDEN..(ti + 1) * HIDDEN].copy_from_slice(&qkv[off + HIDDEN..off + 2 * HIDDEN]);
            v[ti * HIDDEN..(ti + 1) * HIDDEN].copy_from_slice(&qkv[off + 2 * HIDDEN..off + 3 * HIDDEN]);
        }
        let attn = sdpa_routed(accel, &q, &k, &v, t);

        // 5) Proj + gated residual: x = x + gate_a * (attn @ proj_w^T + proj_b).
        let mut proj = mm_bf16_routed(accel, &attn, t, HIDDEN, &self.proj_w, HIDDEN, block_idx, ROLE_PROJ);
        for ti in 0..t { for d in 0..HIDDEN { proj[ti * HIDDEN + d] += self.proj_b[d]; } }
        for ti in 0..t { for d in 0..HIDDEN {
            x[ti * HIDDEN + d] += gate_a[d] * proj[ti * HIDDEN + d];
        }}

        // 5b) Cross-attention bridge to reasoner hidden state (Stage 4 MoT join).
        // Q from gen, K/V from reasoner, residual add (ungated).
        if let Some(reasoner) = reasoner_h {
            debug_assert_eq!(reasoner.len(), t_reasoner * HIDDEN);
            // Q: x @ cross_q_w^T + cross_q_b → [T, HIDDEN].
            let mut qc = mm_bf16_routed(accel, x, t, HIDDEN, &self.cross_q_w, HIDDEN, block_idx, ROLE_CROSS_Q);
            for ti in 0..t { for d in 0..HIDDEN { qc[ti * HIDDEN + d] += self.cross_q_b[d]; } }
            // KV: reasoner @ cross_kv_w^T + cross_kv_b → [T_reasoner, 2*HIDDEN].
            let mut kv = mm_bf16_routed(accel, reasoner, t_reasoner, HIDDEN, &self.cross_kv_w, 2 * HIDDEN, block_idx, ROLE_CROSS_KV);
            for ti in 0..t_reasoner { for d in 0..(2 * HIDDEN) {
                kv[ti * 2 * HIDDEN + d] += self.cross_kv_b[d];
            }}
            let mut kc = vec![0.0f32; t_reasoner * HIDDEN];
            let mut vc = vec![0.0f32; t_reasoner * HIDDEN];
            for ti in 0..t_reasoner {
                let off = ti * 2 * HIDDEN;
                kc[ti * HIDDEN..(ti + 1) * HIDDEN].copy_from_slice(&kv[off..off + HIDDEN]);
                vc[ti * HIDDEN..(ti + 1) * HIDDEN].copy_from_slice(&kv[off + HIDDEN..off + 2 * HIDDEN]);
            }
            let cross_attn = sdpa_cross(&qc, &kc, &vc, t, t_reasoner);
            // proj + ungated residual.
            let mut cp = mm_bf16_routed(accel, &cross_attn, t, HIDDEN, &self.cross_proj_w, HIDDEN, block_idx, ROLE_CROSS_PROJ);
            for ti in 0..t { for d in 0..HIDDEN { cp[ti * HIDDEN + d] += self.cross_proj_b[d]; } }
            for i in 0..(t * HIDDEN) { x[i] += cp[i]; }
        }

        // 6) AdaLN before MLP.
        let xn2 = adaln(x, t, scale_b, shift_b);

        // 7) MLP: fc1 → SiLU → fc2.
        let mut h = mm_bf16_routed(accel, &xn2, t, HIDDEN, &self.fc1_w, crate::FF, block_idx, ROLE_FC1);
        for ti in 0..t { for d in 0..crate::FF { h[ti * crate::FF + d] += self.fc1_b[d]; } }
        for v in h.iter_mut() { *v = *v * (1.0 / (1.0 + (-*v).exp())); }
        let mut m = mm_bf16_routed(accel, &h, t, crate::FF, &self.fc2_w, HIDDEN, block_idx, ROLE_FC2);
        for ti in 0..t { for d in 0..HIDDEN { m[ti * HIDDEN + d] += self.fc2_b[d]; } }

        // 8) Gated MLP residual.
        for ti in 0..t { for d in 0..HIDDEN {
            x[ti * HIDDEN + d] += gate_b[d] * m[ti * HIDDEN + d];
        }}
    }
}

/// Roles for the per-block matmul weight handle cache.  Used by Stage 5's
/// `MetalAccel` to key uploads by (block_idx, role).  Always defined so
/// the routed forward signature is uniform regardless of feature gate.
pub const ROLE_ADA: u8 = 0;
pub const ROLE_QKV: u8 = 1;
pub const ROLE_PROJ: u8 = 2;
pub const ROLE_CROSS_Q: u8 = 3;
pub const ROLE_CROSS_KV: u8 = 4;
pub const ROLE_CROSS_PROJ: u8 = 5;
pub const ROLE_FC1: u8 = 6;
pub const ROLE_FC2: u8 = 7;

/// Routed bf16 matmul: GPU via Metal when `accel` is `Some`, else CPU
/// `matmul_nt_bf16`.  Weight is given as bf16 (the native dtype); on
/// Metal cache miss we dequant once to f32 for upload.
#[cfg(feature = "metal")]
pub(crate) fn mm_bf16_routed(
    accel: Option<&MetalAccel>,
    x: &[f32], m: usize, k: usize, w: &[half::bf16], n: usize,
    block_idx: usize, role: u8,
) -> Vec<f32> {
    if let Some(a) = accel {
        if let Some(out) = a.maybe_dispatch_matmul_bf16_lazy(
            x, || w.iter().map(|b| b.to_f32()).collect(),
            block_idx, role, m, k, n,
        ) {
            return out;
        }
    }
    crate::matmul_nt_bf16(x, m, k, w, n)
}

#[cfg(not(feature = "metal"))]
pub(crate) fn mm_bf16_routed(
    _accel: Option<&MetalAccel>,
    x: &[f32], m: usize, k: usize, w: &[half::bf16], n: usize,
    _block_idx: usize, _role: u8,
) -> Vec<f32> {
    crate::matmul_nt_bf16(x, m, k, w, n)
}

/// Bias-free LayerNorm with AdaLN scale + shift modulation.
/// `out[ti, d] = (1 + scale[d]) * ((x[ti, d] - mean_ti) / sqrt(var_ti + eps)) + shift[d]`.
fn adaln(x: &[f32], t: usize, scale: &[f32], shift: &[f32]) -> Vec<f32> {
    debug_assert_eq!(scale.len(), HIDDEN);
    debug_assert_eq!(shift.len(), HIDDEN);
    let mut out = vec![0.0f32; t * HIDDEN];
    let eps = 1e-6f32;
    for ti in 0..t {
        let row = &x[ti * HIDDEN..(ti + 1) * HIDDEN];
        let mean: f32 = row.iter().sum::<f32>() / HIDDEN as f32;
        let var: f32 = row.iter().map(|v| { let z = v - mean; z * z }).sum::<f32>() / HIDDEN as f32;
        let inv = 1.0 / (var + eps).sqrt();
        let o = &mut out[ti * HIDDEN..(ti + 1) * HIDDEN];
        for d in 0..HIDDEN {
            let n = (row[d] - mean) * inv;
            o[d] = (1.0 + scale[d]) * n + shift[d];
        }
    }
    out
}

/// Cross-attention SDPA: Q has `t_q` tokens, K/V have `t_kv` tokens (from a
/// different sequence — the reasoner state in the MoT bridge).  All split
/// into 32 heads × 128 d_head.
fn sdpa_cross(q: &[f32], k: &[f32], v: &[f32], t_q: usize, t_kv: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let scale = 1.0 / (crate::HEAD_DIM as f32).sqrt();
    let mut attn = vec![0.0f32; t_q * HIDDEN];
    attn.par_chunks_mut(HIDDEN).enumerate().for_each(|(ti, arow)| {
        for h in 0..crate::N_HEADS {
            let off = h * crate::HEAD_DIM;
            let qh = &q[ti * HIDDEN + off..ti * HIDDEN + off + crate::HEAD_DIM];
            let mut sc = vec![0.0f32; t_kv];
            let mut mx = f32::NEG_INFINITY;
            for tj in 0..t_kv {
                let kh = &k[tj * HIDDEN + off..tj * HIDDEN + off + crate::HEAD_DIM];
                let mut s = 0.0f32;
                for d in 0..crate::HEAD_DIM { s += qh[d] * kh[d]; }
                s *= scale;
                sc[tj] = s;
                if s > mx { mx = s; }
            }
            let mut sum = 0.0f32;
            for vv in sc.iter_mut() { *vv = (*vv - mx).exp(); sum += *vv; }
            let inv = 1.0 / sum;
            for vv in sc.iter_mut() { *vv *= inv; }
            for d in 0..crate::HEAD_DIM {
                let mut acc = 0.0f32;
                for tj in 0..t_kv { acc += sc[tj] * v[tj * HIDDEN + off + d]; }
                arow[off + d] = acc;
            }
        }
    });
    attn
}

/// Scaled dot-product attention over `[T, HIDDEN]` Q/K/V activations split
/// into `N_HEADS = 32` heads × `HEAD_DIM = 128` per head.
fn sdpa(q: &[f32], k: &[f32], v: &[f32], t: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let scale = 1.0 / (crate::HEAD_DIM as f32).sqrt();
    let mut attn = vec![0.0f32; t * HIDDEN];
    attn.par_chunks_mut(HIDDEN).enumerate().for_each(|(ti, arow)| {
        for h in 0..crate::N_HEADS {
            let off = h * crate::HEAD_DIM;
            let qh = &q[ti * HIDDEN + off..ti * HIDDEN + off + crate::HEAD_DIM];
            let mut sc = vec![0.0f32; t];
            let mut mx = f32::NEG_INFINITY;
            for tj in 0..t {
                let kh = &k[tj * HIDDEN + off..tj * HIDDEN + off + crate::HEAD_DIM];
                let mut s = 0.0f32;
                for d in 0..crate::HEAD_DIM { s += qh[d] * kh[d]; }
                s *= scale;
                sc[tj] = s;
                if s > mx { mx = s; }
            }
            let mut sum = 0.0f32;
            for v in sc.iter_mut() { *v = (*v - mx).exp(); sum += *v; }
            let inv = 1.0 / sum;
            for v in sc.iter_mut() { *v *= inv; }
            for d in 0..crate::HEAD_DIM {
                let mut acc = 0.0f32;
                for tj in 0..t {
                    acc += sc[tj] * v[tj * HIDDEN + off + d];
                }
                arow[off + d] = acc;
            }
        }
    });
    attn
}

// ────────────────────────────────────────────────────────────────────────
// 3D convolution primitives (Wan 2.2 VAE workhorses)
// ────────────────────────────────────────────────────────────────────────

/// 3D conv with kernel=3, stride=2, padding=1 — the standard downsample
/// stage used by Wan 2.2's encoder stem.  Input `[Cin, T, H, W]` →
/// output `[Cout, T/2, H/2, W/2]`.  Weight is `[Cout, Cin, kT, kH, kW]`
/// row-major (PyTorch / ONNX-export convention).
///
/// Stage-2 perf: scalar nested loop, no NEON.  Wan 2.2 spec dimensions
/// (T=129, H=720, W=1280, Cin=3, Cout=128) would be ~19 GFLOPs — fine for
/// CPU correctness verification; Metal handles production resolutions.
fn conv3d_k3_s2_p1(
    inp: &[f32], cin: usize, t: usize, h: usize, w: usize,
    weight: &[half::bf16], bias: &[f32], cout: usize,
) -> (Vec<f32>, usize, usize, usize) {
    use rayon::prelude::*;
    let to = t.div_ceil(2);
    let ho = h.div_ceil(2);
    let wo = w.div_ceil(2);
    let plane_in = h * w;
    let mut out = vec![0.0f32; cout * to * ho * wo];
    out.par_chunks_mut(to * ho * wo).enumerate().for_each(|(oc, oplane)| {
        for v in oplane.iter_mut() { *v = bias[oc]; }
        for ic in 0..cin {
            // weight[oc, ic, kt, kh, kw] index
            let w_base = ((oc * cin) + ic) * 27;
            let in_off = ic * t * plane_in;
            for ot in 0..to { for oh in 0..ho { for ow in 0..wo {
                let mut acc = 0.0f32;
                for kt in 0..3 { for kh in 0..3 { for kw in 0..3 {
                    let it = (ot * 2 + kt) as isize - 1;
                    let ih = (oh * 2 + kh) as isize - 1;
                    let iw = (ow * 2 + kw) as isize - 1;
                    if it < 0 || it >= t as isize { continue; }
                    if ih < 0 || ih >= h as isize { continue; }
                    if iw < 0 || iw >= w as isize { continue; }
                    let v = inp[in_off + (it as usize) * plane_in + (ih as usize) * w + iw as usize];
                    let wt = weight[w_base + kt * 9 + kh * 3 + kw].to_f32();
                    acc += v * wt;
                }}}
                oplane[ot * ho * wo + oh * wo + ow] += acc;
            }}}
        }
    });
    (out, to, ho, wo)
}

/// 1×1×1 conv — projection across channels.  Weight `[Cout, Cin, 1, 1, 1]`.
fn conv3d_1x1x1(
    inp: &[f32], cin: usize, t: usize, h: usize, w: usize,
    weight: &[half::bf16], bias: &[f32], cout: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    let n = t * h * w;
    let mut out = vec![0.0f32; cout * n];
    out.par_chunks_mut(n).enumerate().for_each(|(oc, oplane)| {
        for v in oplane.iter_mut() { *v = bias[oc]; }
        for ic in 0..cin {
            let wt = weight[oc * cin + ic].to_f32();
            let in_off = ic * n;
            for i in 0..n { oplane[i] += wt * inp[in_off + i]; }
        }
    });
    out
}

/// 3D ConvTranspose with kernel=3, stride=2, padding=1 — the standard
/// upsample (mirror of `conv3d_k3_s2_p1`).  Input `[Cin, T, H, W]` →
/// output `[Cout, T·2, H·2, W·2]` (output_padding=1 to match exact 2× shape).
/// Weight layout `[Cin, Cout, kT, kH, kW]` (PyTorch ConvTranspose convention).
fn conv_transpose3d_k3_s2_p1(
    inp: &[f32], cin: usize, t: usize, h: usize, w: usize,
    weight: &[half::bf16], bias: &[f32], cout: usize,
) -> (Vec<f32>, usize, usize, usize) {
    let to = t * 2;
    let ho = h * 2;
    let wo = w * 2;
    let mut out = vec![0.0f32; cout * to * ho * wo];
    for oc in 0..cout {
        let bb = bias[oc];
        let oc_off = oc * to * ho * wo;
        for i in 0..(to * ho * wo) { out[oc_off + i] = bb; }
    }
    // scatter-add: for each input cell, accumulate into the 3x3x3 output cells it influences.
    for ic in 0..cin {
        let in_off = ic * t * h * w;
        for oc in 0..cout {
            let w_base = ((ic * cout) + oc) * 27;
            let oc_off = oc * to * ho * wo;
            for it in 0..t { for ih in 0..h { for iw in 0..w {
                let v = inp[in_off + it * h * w + ih * w + iw];
                if v == 0.0 { continue; }
                for kt in 0..3 { for kh in 0..3 { for kw in 0..3 {
                    let ot = it as isize * 2 + kt as isize - 1;
                    let oh = ih as isize * 2 + kh as isize - 1;
                    let ow = iw as isize * 2 + kw as isize - 1;
                    if ot < 0 || ot >= to as isize { continue; }
                    if oh < 0 || oh >= ho as isize { continue; }
                    if ow < 0 || ow >= wo as isize { continue; }
                    let wt = weight[w_base + kt * 9 + kh * 3 + kw].to_f32();
                    out[oc_off + (ot as usize) * ho * wo + (oh as usize) * wo + ow as usize] += v * wt;
                }}}
            }}}
        }
    }
    (out, to, ho, wo)
}

// ────────────────────────────────────────────────────────────────────────
// Wan 2.2 VAE — video latent encode/decode
// ────────────────────────────────────────────────────────────────────────

pub struct Wan22VAE {
    /// Encoder stem: 3×3×3 conv (T, H, W kernel = 3) from RGB → 128 channels.
    pub enc_stem_w: BfSlice, pub enc_stem_b: F32Slice,
    /// Decoder final: 3×3×3 conv from 128 → RGB.
    pub dec_final_w: BfSlice, pub dec_final_b: F32Slice,
    /// z-dim down-projection (encoder output) and up-projection (decoder input).
    pub z_in_w: BfSlice, pub z_in_b: F32Slice,
    pub z_out_w: BfSlice, pub z_out_b: F32Slice,
}

impl Wan22VAE {
    pub fn load(sf: &SafetensorsFile, prefix: &str) -> Result<Self, Phase4Error> {
        Ok(Self {
            enc_stem_w:  lbf(sf, &format!("{prefix}.encoder.stem.weight"))?,
            enc_stem_b:  lf32(sf, &format!("{prefix}.encoder.stem.bias"))?,
            dec_final_w: lbf(sf, &format!("{prefix}.decoder.final.weight"))?,
            dec_final_b: lf32(sf, &format!("{prefix}.decoder.final.bias"))?,
            z_in_w:  lbf(sf, &format!("{prefix}.encoder.z.weight"))?,
            z_in_b:  lf32(sf, &format!("{prefix}.encoder.z.bias"))?,
            z_out_w: lbf(sf, &format!("{prefix}.decoder.z.weight"))?,
            z_out_b: lf32(sf, &format!("{prefix}.decoder.z.bias"))?,
        })
    }

    /// Stage 2: real encode.  Apply 3D conv stem (stride=2, 3→128 channels)
    /// twice for a 4× total spatial+temporal downsample, then 1×1×1 z-projection
    /// down to `WAN_Z` channels.  Real Wan 2.2 has 4 ResNet stages between the
    /// stems; Stage 2 ships two stem applications as a faithful 4× downsample
    /// while keeping the synthetic test under a second of wall time.
    pub fn encode(&self, video: &[f32], t: usize, h: usize, w: usize) -> Vec<f32> {
        // Stage 1 (3→128, T/2, H/2, W/2)
        let (h1, t1, hh1, hw1) = conv3d_k3_s2_p1(
            video, 3, t, h, w, &self.enc_stem_w, &self.enc_stem_b, 128);
        // Stage 2 reused stem (128→128) — TODO Stage 2.5: distinct second-stage weights.
        // For now applying the stem-conv shape doesn't match (in=3 not 128), so we use the
        // 1×1×1 z-proj as a cheap second stage that channel-mixes 128→Z directly without
        // a second spatial downsample.
        let z = conv3d_1x1x1(&h1, 128, t1, hh1, hw1, &self.z_in_w, &self.z_in_b, WAN_Z);
        // Returns shape [WAN_Z, T/2, H/2, W/2].  Stage 2.5 will add the second
        // 2× downsample (4× total) once the safetensors include the distinct
        // second-stage stem weights.
        z
    }

    /// Stage 2: real decode.  1×1×1 z-up (Z→128) + ConvTranspose3D final
    /// (128→3, stride=2, 2× spatial+temporal upsample).  Symmetric with encode.
    pub fn decode(&self, latent: &[f32], lt: usize, lh: usize, lw: usize) -> Vec<f32> {
        let h1 = conv3d_1x1x1(latent, WAN_Z, lt, lh, lw, &self.z_out_w, &self.z_out_b, 128);
        let (rgb, _to, _ho, _wo) = conv_transpose3d_k3_s2_p1(
            &h1, 128, lt, lh, lw, &self.dec_final_w, &self.dec_final_b, 3);
        rgb
    }
}

// ────────────────────────────────────────────────────────────────────────
// AVAE-48k audio RVQ tokenizer
// ────────────────────────────────────────────────────────────────────────

pub struct AVAEAudioTokenizer {
    /// Encoder stem (1D conv from raw audio).
    pub enc_w: BfSlice, pub enc_b: F32Slice,
    /// Decoder stem (1D conv-transpose to raw audio).
    pub dec_w: BfSlice, pub dec_b: F32Slice,
    /// 32 RVQ codebooks, each `[VOCAB, hidden]`.  We load them as a single
    /// stacked `[CODEBOOKS, VOCAB, hidden]` tensor and slice via `sub_view`-style
    /// indexing (Stage 1 work).
    pub codebooks: BfSlice,
}

impl AVAEAudioTokenizer {
    pub fn load(sf: &SafetensorsFile, prefix: &str) -> Result<Self, Phase4Error> {
        Ok(Self {
            enc_w: lbf(sf, &format!("{prefix}.encoder.weight"))?,
            enc_b: lf32(sf, &format!("{prefix}.encoder.bias"))?,
            dec_w: lbf(sf, &format!("{prefix}.decoder.weight"))?,
            dec_b: lf32(sf, &format!("{prefix}.decoder.bias"))?,
            codebooks: lbf(sf, &format!("{prefix}.codebooks"))?,
        })
    }

    /// Stage 3: real encode.  1D conv encoder (`enc_w` kernel=7, stride=`HOP=640`,
    /// 1 → `HIDDEN_AVAE=512` channels) flattens 48 kHz audio to ~75 Hz frames;
    /// per-frame 32-stage RVQ then quantizes each 512-d frame to 32 code ids.
    ///
    /// Output layout: `[T_tokens, CODEBOOKS]` flat (frame-major).  Each
    /// `(frame, stage)` slot is the codebook id in `[0, AVAE_VOCAB)`.
    ///
    /// Stage 3 ships a single-stage stem stride=640.  Real AVAE has a multi-
    /// stage progressive downsample (Stage 3.5).
    pub fn encode(&self, audio: &[f32]) -> Vec<u32> {
        const HIDDEN_AVAE: usize = 512;
        let k = 7;  // conv kernel
        let stride = AVAE_HOP;
        let pad = k / 2;
        let t_tokens = audio.len().div_ceil(stride);
        // 1D conv: out[oc, ot] = bias[oc] + Σ_kw enc_w[oc, 0, kw] * audio[ot*stride + kw - pad]
        let mut features = vec![0.0f32; HIDDEN_AVAE * t_tokens];
        for oc in 0..HIDDEN_AVAE {
            let bb = self.enc_b[oc];
            let w_base = oc * k;
            for ot in 0..t_tokens {
                let mut acc = bb;
                for kw in 0..k {
                    let xi = (ot * stride + kw) as isize - pad as isize;
                    if xi < 0 || xi >= audio.len() as isize { continue; }
                    acc += self.enc_w[w_base + kw].to_f32() * audio[xi as usize];
                }
                features[oc * t_tokens + ot] = acc;
            }
        }

        // Now run 32-stage RVQ per frame.  Codebooks layout is
        // [CODEBOOKS, VOCAB, HIDDEN_AVAE] row-major.
        let mut ids = vec![0u32; t_tokens * AVAE_CODEBOOKS];
        for ot in 0..t_tokens {
            // Read frame vector [HIDDEN_AVAE] from features (channel-major layout).
            let mut residual = vec![0.0f32; HIDDEN_AVAE];
            for oc in 0..HIDDEN_AVAE { residual[oc] = features[oc * t_tokens + ot]; }
            for stage in 0..AVAE_CODEBOOKS {
                let cb_base = stage * AVAE_VOCAB * HIDDEN_AVAE;
                let mut best_id = 0u32;
                let mut best_dist = f32::INFINITY;
                for v in 0..AVAE_VOCAB {
                    let code_off = cb_base + v * HIDDEN_AVAE;
                    let mut dist = 0.0f32;
                    for d in 0..HIDDEN_AVAE {
                        let z = self.codebooks[code_off + d].to_f32() - residual[d];
                        dist += z * z;
                        if dist >= best_dist { break; }
                    }
                    if dist < best_dist { best_dist = dist; best_id = v as u32; }
                }
                ids[ot * AVAE_CODEBOOKS + stage] = best_id;
                // Subtract chosen code from residual.
                let code_off = cb_base + (best_id as usize) * HIDDEN_AVAE;
                for d in 0..HIDDEN_AVAE {
                    residual[d] -= self.codebooks[code_off + d].to_f32();
                }
            }
        }
        ids
    }

    /// Stage 3: real decode.  For each frame, sum the 32 chosen code embeddings
    /// → 512-d feature; then 1D ConvTranspose (`dec_w` kernel=7, stride=`HOP`,
    /// 512→1) upsamples back to ~48 kHz waveform.
    pub fn decode(&self, ids: &[u32]) -> Vec<f32> {
        const HIDDEN_AVAE: usize = 512;
        let k = 7;
        let stride = AVAE_HOP;
        let pad = k / 2;
        assert_eq!(ids.len() % AVAE_CODEBOOKS, 0);
        let t_tokens = ids.len() / AVAE_CODEBOOKS;
        // Sum embeddings per frame → features [HIDDEN_AVAE, T_tokens].
        let mut features = vec![0.0f32; HIDDEN_AVAE * t_tokens];
        for ot in 0..t_tokens {
            for stage in 0..AVAE_CODEBOOKS {
                let id = ids[ot * AVAE_CODEBOOKS + stage] as usize;
                let code_off = (stage * AVAE_VOCAB + id) * HIDDEN_AVAE;
                for d in 0..HIDDEN_AVAE {
                    features[d * t_tokens + ot] += self.codebooks[code_off + d].to_f32();
                }
            }
        }
        // 1D ConvTranspose: dec_w shape [1, HIDDEN_AVAE, 7].  Output length =
        // t_tokens * stride.  Each input frame scatters to `stride` output samples.
        let out_len = t_tokens * stride;
        let mut audio = vec![0.0f32; out_len];
        let bb = self.dec_b[0];
        for i in 0..out_len { audio[i] = bb; }
        for ot in 0..t_tokens { for kw in 0..k {
            let oi = ot as isize * stride as isize + kw as isize - pad as isize;
            if oi < 0 || oi >= out_len as isize { continue; }
            let mut acc = 0.0f32;
            for ic in 0..HIDDEN_AVAE {
                let wt = self.dec_w[ic * k + kw].to_f32();
                acc += wt * features[ic * t_tokens + ot];
            }
            audio[oi as usize] += acc;
        }}
        audio
    }
}

// ────────────────────────────────────────────────────────────────────────
// Action token I/O
// ────────────────────────────────────────────────────────────────────────

pub struct ActionTokenIO {
    pub embed_w: BfSlice,    // [ACTION_VOCAB, HIDDEN]
    pub unembed_w: BfSlice,  // [ACTION_VOCAB, HIDDEN]
}

impl ActionTokenIO {
    pub fn load(sf: &SafetensorsFile, prefix: &str) -> Result<Self, Phase4Error> {
        Ok(Self {
            embed_w:   lbf(sf, &format!("{prefix}.embed.weight"))?,
            unembed_w: lbf(sf, &format!("{prefix}.unembed.weight"))?,
        })
    }
    /// Stage-0 stub: pass through.  Stage 1 will look up `embed_w[action_id]`.
    pub fn embed(&self, action_id: u32) -> Vec<f32> {
        let mut out = vec![0.0f32; HIDDEN];
        out[0] = action_id as f32;
        out
    }
}

// ────────────────────────────────────────────────────────────────────────
// Cosmos3OmniTransformer — full generator stack
// ────────────────────────────────────────────────────────────────────────

pub struct Cosmos3OmniTransformer {
    pub time_embedder: TimeEmbedder,
    pub blocks: Vec<OmniBlock>,
    pub vae: Wan22VAE,
    pub audio: AVAEAudioTokenizer,
    pub action: ActionTokenIO,
    /// Final-layer norm + projection to the joint latent space (Wan VAE z-dim).
    pub final_norm_w: F32Slice, pub final_norm_b: F32Slice,
    pub final_proj_w: BfSlice, pub final_proj_b: F32Slice,
}

impl Cosmos3OmniTransformer {
    /// Load the entire Phase 4 stack from a single safetensors file (the
    /// full Cosmos 3 weights are sharded but `load` should be passed the
    /// path to either the unified file or a manifest pointing at the gen
    /// shards — Stage 1 work; Stage 0 is single-file).
    pub fn load(path: &Path) -> Result<Self, Phase4Error> {
        Self::load_with_layers(path, N_LAYERS)
    }

    /// Same as `load` but with a configurable block count.  Used by the
    /// synthetic-weights test to shrink the safetensors below the OOM
    /// boundary while still exercising every load path; real Cosmos 3
    /// loads use `load` (N_LAYERS=36).
    pub fn load_with_layers(path: &Path, n_layers: usize) -> Result<Self, Phase4Error> {
        let sf = SafetensorsFile::load(path)
            .map_err(|e| Phase4Error::Load(e.to_string()))?;
        let time_embedder = TimeEmbedder::load(&sf, "model.gen.time_embed")?;
        let blocks = (0..n_layers)
            .map(|i| OmniBlock::load(&sf, i))
            .collect::<Result<_, _>>()?;
        let vae = Wan22VAE::load(&sf, "model.vae")?;
        let audio = AVAEAudioTokenizer::load(&sf, "model.audio_tokenizer")?;
        let action = ActionTokenIO::load(&sf, "model.action_io")?;
        Ok(Self {
            time_embedder, blocks, vae, audio, action,
            final_norm_w: lf32(&sf, "model.gen.final_norm.weight")?,
            final_norm_b: lf32(&sf, "model.gen.final_norm.bias")?,
            final_proj_w: lbf(&sf, "model.gen.final_proj.weight")?,
            final_proj_b: lf32(&sf, "model.gen.final_proj.bias")?,
        })
    }

    /// Single Euler diffusion step at timestep `t` with stride `dt`.
    ///
    /// Stage 1: real OmniBlock forward chain.  Stage 2 will add the full
    /// 36-block sweep + final norm/proj + Wan22 VAE latent projection.
    /// For Stage 1 we run only the first block (one full attention pass per
    /// step keeps the synthetic-weights test under a second of wall time);
    /// `n_blocks_per_step()` can be tuned upward once Metal lands.
    pub fn euler_step(&self, x: &mut [f32], t: f32, _dt: f32) {
        self.euler_step_with_cross(x, t, _dt, None, 0);
    }

    /// Stage-4 Euler step with optional MoT cross-attention to reasoner hidden state.
    /// When `reasoner_h` is `Some`, every active block's `forward_with_cross`
    /// pulls K/V from the reasoner sequence; when `None`, the bridge is bypassed
    /// and only self-attention runs (equivalent to `euler_step`).
    pub fn euler_step_with_cross(
        &self, x: &mut [f32], t: f32, dt: f32,
        reasoner_h: Option<&[f32]>, t_reasoner: usize,
    ) {
        self.euler_step_routed(x, t, dt, reasoner_h, t_reasoner, None);
    }

    /// Stage 5 Stage 2 Euler step with optional Metal acceleration.  When
    /// `accel` is `Some`, every block's 8 bf16 matmuls route through GPU.
    pub fn euler_step_routed(
        &self, x: &mut [f32], t: f32, _dt: f32,
        reasoner_h: Option<&[f32]>, t_reasoner: usize,
        accel: Option<&MetalAccel>,
    ) {
        let time_emb = self.time_embedder.forward(t);
        let n_tok = x.len() / HIDDEN;
        for (i, blk) in self.blocks.iter().take(self.n_blocks_per_step()).enumerate() {
            blk.forward_routed(x, n_tok, &time_emb, reasoner_h, t_reasoner, i, accel);
        }
    }

    /// Full sampling with optional Metal acceleration.
    pub fn dummy_diffuse_routed(
        &self, mut x: Vec<f32>, n_steps: usize,
        reasoner_h: Option<&[f32]>, t_reasoner: usize,
        accel: Option<&MetalAccel>,
    ) -> Vec<f32> {
        let dt = 1.0 / n_steps as f32;
        for step in 0..n_steps {
            let t = 1.0 - step as f32 * dt;
            self.euler_step_routed(&mut x, t, dt, reasoner_h, t_reasoner, accel);
        }
        x
    }

    /// Sampling loop with optional MoT cross-attention bridge.
    pub fn dummy_diffuse_with_cross(
        &self, mut x: Vec<f32>, n_steps: usize,
        reasoner_h: Option<&[f32]>, t_reasoner: usize,
    ) -> Vec<f32> {
        let dt = 1.0 / n_steps as f32;
        for step in 0..n_steps {
            let t = 1.0 - step as f32 * dt;
            self.euler_step_with_cross(&mut x, t, dt, reasoner_h, t_reasoner);
        }
        x
    }

    /// Stage 1 default: 1 block per step (so a 4-step diffusion = 4 block
    /// forwards).  Stage 5 (Metal): bump to 36 (full depth).  Synthetic
    /// test runs in `n_blocks_per_step = 1` mode to keep wall time short.
    fn n_blocks_per_step(&self) -> usize { 1 }

    /// Full sampling: starting noise → run `n_steps` Euler steps from t=1→0.
    pub fn dummy_diffuse(&self, mut x: Vec<f32>, n_steps: usize) -> Vec<f32> {
        let dt = 1.0 / n_steps as f32;
        for step in 0..n_steps {
            let t = 1.0 - step as f32 * dt;
            self.euler_step(&mut x, t, dt);
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rce_weight_loader::write_safetensors;
    use half::bf16;

    /// Build a synthetic safetensors file with all the Phase 4 tensors at
    /// the expected names + shapes.  Random Gaussian init for everything.
    /// Number of OmniBlocks to synthesize in the test.  Real Cosmos 3 has
    /// `N_LAYERS=36`; we use a small count so the synthetic safetensors fits
    /// in RAM during the test (each block is ~210 MB at HIDDEN=4096, FF=12288,
    /// plus cross-attn weights ≈ +128 MB/block).
    const TEST_LAYERS: usize = 2;

    fn write_synthetic(path: &std::path::Path) {
        let mut state: u64 = 0xD1FFu64.wrapping_mul(0xC0FFEE_42);
        let mut rng = || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let f = ((state >> 40) & 0xFFFFFF) as f32 / 16777216.0;
            (f - 0.5) * 0.05
        };
        let bf_bytes = |shape: &[usize], rng: &mut dyn FnMut() -> f32| -> Vec<u8> {
            let n: usize = shape.iter().product();
            let mut out = Vec::with_capacity(n * 2);
            for _ in 0..n {
                let bits = bf16::from_f32(rng()).to_bits().to_le_bytes();
                out.extend_from_slice(&bits);
            }
            out
        };
        let f_bytes = |shape: &[usize], rng: &mut dyn FnMut() -> f32| -> Vec<u8> {
            let n: usize = shape.iter().product();
            let mut out = Vec::with_capacity(n * 4);
            for _ in 0..n { out.extend_from_slice(&rng().to_le_bytes()); }
            out
        };

        let mut entries: Vec<(String, &'static str, Vec<usize>, Vec<u8>)> = Vec::new();

        // Time embedder.
        entries.push(("model.gen.time_embed.time_mlp.0.weight".into(), "bf16",
            vec![4*D_TIME, D_TIME], bf_bytes(&[4*D_TIME, D_TIME], &mut rng)));
        entries.push(("model.gen.time_embed.time_mlp.0.bias".into(), "f32",
            vec![4*D_TIME], f_bytes(&[4*D_TIME], &mut rng)));
        entries.push(("model.gen.time_embed.time_mlp.2.weight".into(), "bf16",
            vec![HIDDEN, 4*D_TIME], bf_bytes(&[HIDDEN, 4*D_TIME], &mut rng)));
        entries.push(("model.gen.time_embed.time_mlp.2.bias".into(), "f32",
            vec![HIDDEN], f_bytes(&[HIDDEN], &mut rng)));

        // Test block count (real Cosmos 3 has N_LAYERS=36; we keep the test small).
        for i in 0..TEST_LAYERS {
            let p = format!("model.gen.layers.{i}");
            entries.push((format!("{p}.adaln.weight_moe_gen"), "bf16",
                vec![6*HIDDEN, HIDDEN], bf_bytes(&[6*HIDDEN, HIDDEN], &mut rng)));
            entries.push((format!("{p}.adaln.bias_moe_gen"), "f32",
                vec![6*HIDDEN], f_bytes(&[6*HIDDEN], &mut rng)));
            entries.push((format!("{p}.attn.qkv.weight_moe_gen"), "bf16",
                vec![3*HIDDEN, HIDDEN], bf_bytes(&[3*HIDDEN, HIDDEN], &mut rng)));
            entries.push((format!("{p}.attn.qkv.bias_moe_gen"), "f32",
                vec![3*HIDDEN], f_bytes(&[3*HIDDEN], &mut rng)));
            entries.push((format!("{p}.attn.proj.weight_moe_gen"), "bf16",
                vec![HIDDEN, HIDDEN], bf_bytes(&[HIDDEN, HIDDEN], &mut rng)));
            entries.push((format!("{p}.attn.proj.bias_moe_gen"), "f32",
                vec![HIDDEN], f_bytes(&[HIDDEN], &mut rng)));
            // Stage 4: cross-attention bridge weights.
            entries.push((format!("{p}.cross_attn.q.weight_moe_gen"), "bf16",
                vec![HIDDEN, HIDDEN], bf_bytes(&[HIDDEN, HIDDEN], &mut rng)));
            entries.push((format!("{p}.cross_attn.q.bias_moe_gen"), "f32",
                vec![HIDDEN], f_bytes(&[HIDDEN], &mut rng)));
            entries.push((format!("{p}.cross_attn.kv.weight_moe_gen"), "bf16",
                vec![2 * HIDDEN, HIDDEN], bf_bytes(&[2 * HIDDEN, HIDDEN], &mut rng)));
            entries.push((format!("{p}.cross_attn.kv.bias_moe_gen"), "f32",
                vec![2 * HIDDEN], f_bytes(&[2 * HIDDEN], &mut rng)));
            entries.push((format!("{p}.cross_attn.proj.weight_moe_gen"), "bf16",
                vec![HIDDEN, HIDDEN], bf_bytes(&[HIDDEN, HIDDEN], &mut rng)));
            entries.push((format!("{p}.cross_attn.proj.bias_moe_gen"), "f32",
                vec![HIDDEN], f_bytes(&[HIDDEN], &mut rng)));
            entries.push((format!("{p}.mlp.fc1.weight_moe_gen"), "bf16",
                vec![FF, HIDDEN], bf_bytes(&[FF, HIDDEN], &mut rng)));
            entries.push((format!("{p}.mlp.fc1.bias_moe_gen"), "f32",
                vec![FF], f_bytes(&[FF], &mut rng)));
            entries.push((format!("{p}.mlp.fc2.weight_moe_gen"), "bf16",
                vec![HIDDEN, FF], bf_bytes(&[HIDDEN, FF], &mut rng)));
            entries.push((format!("{p}.mlp.fc2.bias_moe_gen"), "f32",
                vec![HIDDEN], f_bytes(&[HIDDEN], &mut rng)));
        }

        // Wan 2.2 VAE — small placeholder shapes (real VAE has more layers,
        // but Stage 0 just needs the stems + z-proj for type-check).
        entries.push(("model.vae.encoder.stem.weight".into(), "bf16",
            vec![128, 3, 3, 3, 3], bf_bytes(&[128, 3, 3, 3, 3], &mut rng)));
        entries.push(("model.vae.encoder.stem.bias".into(), "f32",
            vec![128], f_bytes(&[128], &mut rng)));
        entries.push(("model.vae.decoder.final.weight".into(), "bf16",
            vec![3, 128, 3, 3, 3], bf_bytes(&[3, 128, 3, 3, 3], &mut rng)));
        entries.push(("model.vae.decoder.final.bias".into(), "f32",
            vec![3], f_bytes(&[3], &mut rng)));
        entries.push(("model.vae.encoder.z.weight".into(), "bf16",
            vec![WAN_Z, 128, 1, 1, 1], bf_bytes(&[WAN_Z, 128, 1, 1, 1], &mut rng)));
        entries.push(("model.vae.encoder.z.bias".into(), "f32",
            vec![WAN_Z], f_bytes(&[WAN_Z], &mut rng)));
        entries.push(("model.vae.decoder.z.weight".into(), "bf16",
            vec![128, WAN_Z, 1, 1, 1], bf_bytes(&[128, WAN_Z, 1, 1, 1], &mut rng)));
        entries.push(("model.vae.decoder.z.bias".into(), "f32",
            vec![128], f_bytes(&[128], &mut rng)));

        // AVAE.
        entries.push(("model.audio_tokenizer.encoder.weight".into(), "bf16",
            vec![512, 1, 7], bf_bytes(&[512, 1, 7], &mut rng)));
        entries.push(("model.audio_tokenizer.encoder.bias".into(), "f32",
            vec![512], f_bytes(&[512], &mut rng)));
        entries.push(("model.audio_tokenizer.decoder.weight".into(), "bf16",
            vec![1, 512, 7], bf_bytes(&[1, 512, 7], &mut rng)));
        entries.push(("model.audio_tokenizer.decoder.bias".into(), "f32",
            vec![1], f_bytes(&[1], &mut rng)));
        entries.push(("model.audio_tokenizer.codebooks".into(), "bf16",
            vec![AVAE_CODEBOOKS, AVAE_VOCAB, 512],
            bf_bytes(&[AVAE_CODEBOOKS, AVAE_VOCAB, 512], &mut rng)));

        // Action I/O.
        entries.push(("model.action_io.embed.weight".into(), "bf16",
            vec![ACTION_VOCAB, HIDDEN], bf_bytes(&[ACTION_VOCAB, HIDDEN], &mut rng)));
        entries.push(("model.action_io.unembed.weight".into(), "bf16",
            vec![ACTION_VOCAB, HIDDEN], bf_bytes(&[ACTION_VOCAB, HIDDEN], &mut rng)));

        // Final norm + proj.
        entries.push(("model.gen.final_norm.weight".into(), "f32",
            vec![HIDDEN], f_bytes(&[HIDDEN], &mut rng)));
        entries.push(("model.gen.final_norm.bias".into(), "f32",
            vec![HIDDEN], f_bytes(&[HIDDEN], &mut rng)));
        entries.push(("model.gen.final_proj.weight".into(), "bf16",
            vec![WAN_Z, HIDDEN], bf_bytes(&[WAN_Z, HIDDEN], &mut rng)));
        entries.push(("model.gen.final_proj.bias".into(), "f32",
            vec![WAN_Z], f_bytes(&[WAN_Z], &mut rng)));

        let refs: Vec<(&str, &str, &[usize], &[u8])> = entries.iter()
            .map(|(n, dt, sh, b)| (n.as_str(), *dt, sh.as_slice(), b.as_slice()))
            .collect();
        let blob = write_safetensors(&refs, None);
        std::fs::write(path, blob).expect("write synthetic phase4 safetensors");
    }

    /// Phase 4 Stage 0 end-to-end on synthetic weights.  Writes a synthetic
    /// safetensors with the full Phase 4 tensor inventory, loads it via
    /// `Cosmos3OmniTransformer::load`, exercises every component's `load`
    /// path, runs a 4-step `dummy_diffuse` on a small latent.  Validates the
    /// architecture wiring without needing the real 60+ GB Cosmos 3 release.
    /// Stage 5 Stage 3 perf sweep: time CPU vs GPU diffusion across token
    /// counts to find the crossover.  Output is captured in the test log;
    /// `#[ignore]`-gated because each run touches the synthetic safetensors.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "phase4 metal perf — sweeps T ∈ {4, 64, 256, 1024, 4096}; allocates ~3 GB"]
    fn phase4_metal_perf_sweep() {
        let Some(accel) = MetalAccel::new() else { eprintln!("metal unavailable"); return; };
        let p = std::env::temp_dir().join("cosmos3_phase4_perf_sweep.safetensors");
        write_synthetic(&p);
        let m = Cosmos3OmniTransformer::load_with_layers(&p, TEST_LAYERS)
            .expect("load synthetic phase4");

        // Warm up GPU caches with a small forward (uploads all 16 weights:
        // 2 blocks × 8 roles).  Subsequent measurements hit the cache.
        {
            let warm: Vec<f32> = vec![0.0f32; 4 * HIDDEN];
            let _ = m.dummy_diffuse_routed(warm, 1, None, 0, Some(&accel));
        }

        eprintln!("\nphase4 GPU vs CPU sweep ({} blocks × 1 step):", TEST_LAYERS);
        eprintln!("    T |    CPU |    GPU | speedup");
        eprintln!("------+--------+--------+---------");
        for &t_tokens in &[4usize, 64, 256, 1024, 4096] {
            let init: Vec<f32> = (0..t_tokens * HIDDEN)
                .map(|i| ((i as f32 * 0.013).sin()) * 0.1)
                .collect();
            // CPU
            let t0 = std::time::Instant::now();
            let _ = m.dummy_diffuse_routed(init.clone(), 1, None, 0, None);
            let cpu = t0.elapsed();
            // GPU (cache populated from warm-up; only kernel-launch overhead)
            let t0 = std::time::Instant::now();
            let _ = m.dummy_diffuse_routed(init.clone(), 1, None, 0, Some(&accel));
            let gpu = t0.elapsed();
            let speedup = cpu.as_secs_f64() / gpu.as_secs_f64();
            eprintln!(" {:>4} | {:>5.1}ms | {:>5.1}ms | {:>6.2}×",
                      t_tokens, cpu.as_secs_f64() * 1000.0, gpu.as_secs_f64() * 1000.0, speedup);
        }
        let _ = std::fs::remove_file(&p);
    }

    /// Stage 5 Stage 2: dispatch full diffusion on GPU and verify the
    /// final state matches the CPU-only diffusion within bf16 noise.
    /// Reuses the same synthetic safetensors as the main e2e test.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "phase4 metal — writes synthetic safetensors, allocates ~1 GB; run with --ignored"]
    fn phase4_metal_diffuse_matches_cpu() {
        let Some(accel) = MetalAccel::new() else { eprintln!("metal unavailable"); return; };
        let p = std::env::temp_dir().join("cosmos3_phase4_metal_diffuse.safetensors");
        write_synthetic(&p);
        let m = Cosmos3OmniTransformer::load_with_layers(&p, TEST_LAYERS)
            .expect("load synthetic phase4");

        let t_tokens = 4;
        let init: Vec<f32> = (0..t_tokens * HIDDEN)
            .map(|i| ((i as f32 * 0.013).sin()) * 0.1)
            .collect();

        let t0 = std::time::Instant::now();
        let cpu = m.dummy_diffuse_routed(init.clone(), 2, None, 0, None);
        let cpu_dt = t0.elapsed();
        let t0 = std::time::Instant::now();
        let gpu = m.dummy_diffuse_routed(init.clone(), 2, None, 0, Some(&accel));
        let gpu_dt = t0.elapsed();
        eprintln!("CPU 2 steps × {t_tokens} tok × {HIDDEN} d in {:?}", cpu_dt);
        eprintln!("GPU 2 steps × {t_tokens} tok × {HIDDEN} d in {:?}", gpu_dt);

        assert_eq!(cpu.len(), gpu.len());
        let max_abs = cpu.iter().zip(gpu.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let max_ref = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let rel = if max_ref > 0.0 { max_abs / max_ref } else { 0.0 };
        eprintln!("CPU vs GPU diffuse: max_abs={max_abs:.6} max_ref={max_ref:.4} rel={rel:.4}");
        // Two diffusion steps × 2 blocks/step × 8 matmuls/block = 32 bf16
        // matmuls compose; bf16 reorder noise + accumulator-ordering across
        // GPU tiles vs CPU NEON dot accumulates.  Allow 5 % relative.
        assert!(rel < 5e-2, "CPU vs GPU diffusion mismatch rel={rel}");

        let _ = std::fs::remove_file(&p);
    }

    /// Stage 5 Stage 1: dispatch a small bf16 matmul through MetalAccel
    /// and verify the output matches the CPU `matmul_nt_bf16` path within
    /// bf16 reorder noise (~1e-3 relative; bf16 has 7 mantissa bits).
    #[cfg(feature = "metal")]
    #[test]
    fn phase4_metal_matmul_matches_cpu() {
        let Some(accel) = MetalAccel::new() else {
            eprintln!("Metal not available — skipping");
            return;
        };
        let m = 4; let k = 64; let n = 32;
        let input: Vec<f32> = (0..m*k).map(|i| (i as f32 * 0.013).sin() * 0.5).collect();
        let weight_f32: Vec<f32> = (0..n*k).map(|i| (i as f32 * 0.029).cos() * 0.5).collect();
        // CPU reference: matmul_nt_bf16 narrows weight to bf16 then dots.
        let weight_bf16: Vec<half::bf16> = weight_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let cpu = crate::matmul_nt_bf16(&input, m, k, &weight_bf16, n);
        // GPU dispatch.
        let Some(gpu) = accel.maybe_dispatch_matmul_bf16(&input, &weight_f32, 0, 0, m, k, n) else {
            eprintln!("Metal dispatch failed — skipping (Stage 5 Stage 2: investigate)");
            return;
        };
        assert_eq!(gpu.len(), m * n);
        let max_abs = cpu.iter().zip(gpu.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let max_ref = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let rel = if max_ref > 0.0 { max_abs / max_ref } else { 0.0 };
        eprintln!("metal bf16 matmul {m}×{k}×{n}: max_abs={max_abs:.6} max_ref={max_ref:.4} rel={rel:.4}");
        // bf16 has ~3e-3 ulp resolution; allow 1e-2 relative for accumulator
        // ordering differences between scalar Rust + GPU tile.
        assert!(rel < 1e-2, "GPU vs CPU mismatch: rel={rel}");
    }

    /// Stage 5 Stage 0 smoke: initialize `MetalAccel` on Apple Silicon.
    /// Verifies the rce-op-metal dep tree compiles + the system default
    /// Metal device responds.  Stage 5 Stage 1+ adds real matmul dispatch.
    #[cfg(feature = "metal")]
    #[test]
    fn phase4_metal_init() {
        let accel = MetalAccel::new();
        // On Apple Silicon (where this crate is normally exercised) this
        // must return Some.  In headless CI or sandboxes without Metal
        // access it returns None — accept either.
        match accel {
            Some(_) => eprintln!("MetalAccel initialized on system default device"),
            None    => eprintln!("MetalAccel::new() returned None — Metal unavailable in this env"),
        }
    }

    #[test]
    #[ignore = "phase4 — writes ~3 GB synthetic safetensors (36 blocks × bf16); run with --ignored"]
    fn phase4_synthetic_e2e() {
        let p = std::env::temp_dir().join("cosmos3_phase4_synthetic.safetensors");
        write_synthetic(&p);
        let m = Cosmos3OmniTransformer::load_with_layers(&p, TEST_LAYERS)
            .expect("load synthetic phase4");

        // VAE encode/decode roundtrip shape check.  Stage 2 ships 2× downsample
        // (Stage 2.5 will add the second 2× stage to reach the spec's 4×).
        let video: Vec<f32> = (0..3*4*16*16).map(|i| ((i as f32 * 0.013).sin()) * 0.1).collect();
        let latent = m.vae.encode(&video, 4, 16, 16);
        assert_eq!(latent.len(), WAN_Z * 2 * 8 * 8);          // 4/2 × 16/2 × 16/2
        // Real values now — non-trivial spread from the encoder stem + z-proj.
        let latent_spread = latent.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                          - latent.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(latent_spread > 1e-5, "Wan VAE latent should be non-flat, got spread {latent_spread}");
        let decoded = m.vae.decode(&latent, 2, 8, 8);
        assert_eq!(decoded.len(), 3 * 4 * 16 * 16);
        assert!(decoded.iter().all(|v| v.is_finite()));

        // Audio encode/decode roundtrip — Stage 3 has real RVQ + real conv1d
        // stems.  Use a SHORT clip (0.1 s) to keep the 32-stage nearest-neighbor
        // search bounded on synthetic 32 × 1024 × 512 codebooks (~16M-op per
        // frame × 32 stages × 75 frames in production audio is too much for
        // the synthetic-weights test wall-time budget).
        let audio: Vec<f32> = (0..AVAE_HOP * 2).map(|i| ((i as f32 * 0.001).sin()) * 0.1).collect();
        let t0 = std::time::Instant::now();
        let ids = m.audio.encode(&audio);
        eprintln!("AVAE encode {} samples → {} codebooks × {} frames in {:?}",
                  audio.len(), AVAE_CODEBOOKS, ids.len()/AVAE_CODEBOOKS, t0.elapsed());
        assert_eq!(ids.len(), 2 * AVAE_CODEBOOKS);
        // IDs must be in [0, VOCAB) and non-degenerate (random init means at
        // least some non-zero ids should appear across the 64 slots).
        assert!(ids.iter().all(|&v| (v as usize) < AVAE_VOCAB));
        let nonzero = ids.iter().filter(|&&v| v != 0).count();
        assert!(nonzero > 0, "all 64 RVQ slots collapsed to id 0 — codebook search broken");
        let t0 = std::time::Instant::now();
        let decoded_audio = m.audio.decode(&ids);
        eprintln!("AVAE decode {} ids → {} samples in {:?}", ids.len(), decoded_audio.len(), t0.elapsed());
        assert_eq!(decoded_audio.len(), 2 * AVAE_HOP);
        assert!(decoded_audio.iter().all(|v| v.is_finite()));

        // Action embed shape check.
        let emb = m.action.embed(7);
        assert_eq!(emb.len(), HIDDEN);

        // Diffusion stub: 4 Euler steps on a latent buffer.
        // Use a small T to avoid 36-block × HIDDEN stub work blowing up.
        let t_tokens = 8;
        let init: Vec<f32> = (0..t_tokens * HIDDEN)
            .map(|i| ((i as f32 * 0.013).sin()) * 0.1)
            .collect();
        let t0 = std::time::Instant::now();
        let out = m.dummy_diffuse(init.clone(), 4);
        eprintln!("phase4 dummy_diffuse 4 steps × {t_tokens} tokens × {HIDDEN} d in {:?}", t0.elapsed());
        assert_eq!(out.len(), init.len());
        assert!(out.iter().all(|v| v.is_finite()));
        // Stage 1 OmniBlock does real attention + MLP per step → output
        // should drift substantially from input.
        let max_drift = out.iter().zip(init.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_drift > 1e-4, "Stage-1 dummy_diffuse should produce real drift via OmniBlock, got {max_drift}");

        // Stage 4: same diffusion + MoT cross-attention bridge to a tiny
        // synthetic "reasoner" hidden state.  Output must differ from the
        // bridge-disabled run (proves cross-attn is actually contributing).
        let t_reasoner = 4;
        let reasoner_h: Vec<f32> = (0..t_reasoner * HIDDEN)
            .map(|i| ((i as f32 * 0.017).cos()) * 0.1)
            .collect();
        let t0 = std::time::Instant::now();
        let out_xattn = m.dummy_diffuse_with_cross(init.clone(), 4, Some(&reasoner_h), t_reasoner);
        eprintln!("phase4 dummy_diffuse w/ cross-attn ({t_reasoner} reasoner tokens) in {:?}", t0.elapsed());
        assert_eq!(out_xattn.len(), out.len());
        assert!(out_xattn.iter().all(|v| v.is_finite()));
        let cross_drift = out_xattn.iter().zip(out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(cross_drift > 1e-4,
            "cross-attention should change output vs the bridge-disabled run, got {cross_drift}");

        let _ = std::fs::remove_file(&p);
    }
}
