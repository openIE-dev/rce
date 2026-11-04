//! `rce-op-nn-prims` — small `OpDispatch` leaves the cpu kernel
//! layer doesn't expose yet but every modern model block needs:
//!
//! - `Custom("silu")`        : x ↦ x * sigmoid(x), elementwise.
//! - `Custom("rms_norm")`    : group-aware RMSNorm.
//! - `Custom("split_at")`    : slice a 1-D tensor `[start, start+len)`.
//! - `Custom("gelu")`        : tanh-approximation GeLU (Gemma, BERT, GPT-2).
//! - `Custom("swiglu")`      : silu(gate) * up (Llama 2/3, Mistral).
//! - `Custom("layer_norm")`  : group-aware LayerNorm (mean+variance).
//!
//! Together these are the prims required to express a Mamba2 / Llama /
//! Jamba / nemotron_h mixer block as a composable `FunctionGraph`
//! (instead of one fused `Custom("mamba2_mixer")` leaf). Each has a
//! Metal-backed counterpart in `rce-op-metal::MetalLeaf`; the
//! dispatch chain picks per-platform.

use rce_vm_backend_cpu::{
    DType, HostTensor, InterpretError, OpDispatch, Result, ScalarValue, Value,
};
use tracing::trace;

// ── SiLU ───────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone, Copy)]
pub struct SiluLeaf;
impl SiluLeaf {
    pub const fn new() -> Self { Self }
}
impl OpDispatch for SiluLeaf {
    fn handles(&self) -> &[&'static str] { &["silu"] }
    fn execute(&self, _node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "silu" {
            return Err(InterpretError::Unsupported(format!("silu leaf: not `{name}`")));
        }
        if args.len() != 1 {
            return Err(InterpretError::Unsupported(format!(
                "silu: expected 1 arg, got {}", args.len()
            )));
        }
        let t = arg_tensor(&args[0], "x")?;
        if t.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "silu: f32 only, got {}", t.dtype.name()
            )));
        }
        let xs = bytes_as_f32(&t.data, "x")?;
        let mut out = Vec::with_capacity(xs.len());
        for &x in xs {
            // SiLU(x) = x * sigmoid(x). Numerically stable form:
            // for x >= 0:  x / (1 + exp(-x))
            // for x <  0:  x * exp(x) / (1 + exp(x))
            let y = if x >= 0.0 {
                x / (1.0 + (-x).exp())
            } else {
                let e = x.exp();
                x * e / (1.0 + e)
            };
            out.push(y);
        }
        let bytes: &[u8] = bytemuck::cast_slice(&out);
        HostTensor::new(t.shape.clone(), DType::F32, bytes.to_vec())
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!("silu: build out: {e:?}")))
    }
}

// ── RMSNorm (group-aware) ──────────────────────────────────────────

/// `Custom("rms_norm")` — group-aware RMSNorm.
///
/// Args:
/// 1. `x`        f32 tensor (1-D, length `total`).
/// 2. `weight`   f32 tensor of shape `(total,)` — per-element scale.
/// 3. `ngroups`  i32 scalar — split `total` into `ngroups` equal groups,
///               normalise each group independently. `ngroups=1` is the
///               standard whole-vector RMSNorm.
/// 4. `eps`      f32 scalar — additive epsilon inside the sqrt for stability.
///
/// Output: f32 tensor, same shape as `x`.
#[derive(Debug, Default, Clone, Copy)]
pub struct RmsNormLeaf;
impl RmsNormLeaf {
    pub const fn new() -> Self { Self }
}
impl OpDispatch for RmsNormLeaf {
    fn handles(&self) -> &[&'static str] { &["rms_norm"] }
    fn execute(&self, _node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "rms_norm" {
            return Err(InterpretError::Unsupported(format!("rms_norm leaf: not `{name}`")));
        }
        if args.len() != 4 {
            return Err(InterpretError::Unsupported(format!(
                "rms_norm: expected 4 args (x, weight, ngroups, eps), got {}", args.len()
            )));
        }
        let x = arg_tensor(&args[0], "x")?;
        let w = arg_tensor(&args[1], "weight")?;
        let ngroups = arg_i32(&args[2], "ngroups")?;
        let eps = arg_f32(&args[3], "eps")?;

        if x.dtype != DType::F32 || w.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "rms_norm: f32 only (x={}, weight={})", x.dtype.name(), w.dtype.name()
            )));
        }
        let total = x.numel();
        if w.numel() != total {
            return Err(InterpretError::Unsupported(format!(
                "rms_norm: weight numel {} != x numel {}", w.numel(), total
            )));
        }
        if ngroups <= 0 {
            return Err(InterpretError::Unsupported(format!(
                "rms_norm: ngroups must be positive, got {ngroups}"
            )));
        }
        let ngroups = ngroups as usize;
        if total % ngroups != 0 {
            return Err(InterpretError::Unsupported(format!(
                "rms_norm: total {total} not divisible by ngroups {ngroups}"
            )));
        }
        let group_size = total / ngroups;
        let xs = bytes_as_f32(&x.data, "x")?;
        let ws = bytes_as_f32(&w.data, "weight")?;
        let mut out = vec![0.0f32; total];

        for g in 0..ngroups {
            let start = g * group_size;
            let end = start + group_size;
            let slice = &xs[start..end];
            let rms = (slice.iter().map(|v| v * v).sum::<f32>() / group_size as f32 + eps).sqrt();
            let inv = 1.0 / rms;
            for i in 0..group_size {
                out[start + i] = xs[start + i] * inv * ws[start + i];
            }
        }

        trace!(total, ngroups, group_size, "rms_norm dispatched");
        let bytes: &[u8] = bytemuck::cast_slice(&out);
        HostTensor::new(x.shape.clone(), DType::F32, bytes.to_vec())
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!("rms_norm: build out: {e:?}")))
    }
}

// ── GeLU (tanh approximation) ──────────────────────────────────────

/// `Custom("gelu")` — tanh-approximation GeLU. CPU mirror of
/// `rce-op-metal::MetalLeaf`'s `gelu_tanh_f32` kernel.
#[derive(Debug, Default, Clone, Copy)]
pub struct GeluLeaf;
impl GeluLeaf {
    pub const fn new() -> Self { Self }
}
impl OpDispatch for GeluLeaf {
    fn handles(&self) -> &[&'static str] { &["gelu"] }
    fn execute(&self, _node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "gelu" {
            return Err(InterpretError::Unsupported(format!("gelu leaf: not `{name}`")));
        }
        if args.len() != 1 {
            return Err(InterpretError::Unsupported(format!(
                "gelu: expected 1 arg, got {}", args.len()
            )));
        }
        let t = arg_tensor(&args[0], "x")?;
        if t.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "gelu: f32 only, got {}", t.dtype.name()
            )));
        }
        let xs = bytes_as_f32(&t.data, "x")?;
        const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
        const COEFF: f32 = 0.044715;
        let mut out = Vec::with_capacity(xs.len());
        for &x in xs {
            let x3 = x * x * x;
            let mut inner = SQRT_2_OVER_PI * (x + COEFF * x3);
            if inner > 10.0 { inner = 10.0; } else if inner < -10.0 { inner = -10.0; }
            out.push(0.5 * x * (1.0 + inner.tanh()));
        }
        let bytes: &[u8] = bytemuck::cast_slice(&out);
        HostTensor::new(t.shape.clone(), DType::F32, bytes.to_vec())
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!("gelu: build out: {e:?}")))
    }
}

// ── SwiGLU ─────────────────────────────────────────────────────────

/// `Custom("swiglu")` — silu(gate) * up. CPU mirror of
/// `rce-op-metal::MetalLeaf`'s `swiglu_f32` kernel.
///
/// Args: `gate` (tensor), `up` (tensor). Both same shape and length.
#[derive(Debug, Default, Clone, Copy)]
pub struct SwigluLeaf;
impl SwigluLeaf {
    pub const fn new() -> Self { Self }
}
impl OpDispatch for SwigluLeaf {
    fn handles(&self) -> &[&'static str] { &["swiglu"] }
    fn execute(&self, _node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "swiglu" {
            return Err(InterpretError::Unsupported(format!("swiglu leaf: not `{name}`")));
        }
        if args.len() != 2 {
            return Err(InterpretError::Unsupported(format!(
                "swiglu: expected 2 args (gate, up), got {}", args.len()
            )));
        }
        let gate = arg_tensor(&args[0], "gate")?;
        let up = arg_tensor(&args[1], "up")?;
        if gate.dtype != DType::F32 || up.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "swiglu: f32 only (gate={}, up={})", gate.dtype.name(), up.dtype.name()
            )));
        }
        if gate.numel() != up.numel() {
            return Err(InterpretError::Unsupported(format!(
                "swiglu: gate ({}) and up ({}) must have same numel",
                gate.numel(), up.numel()
            )));
        }
        let gs = bytes_as_f32(&gate.data, "gate")?;
        let us = bytes_as_f32(&up.data, "up")?;
        let mut out = Vec::with_capacity(gs.len());
        for (g, u) in gs.iter().zip(us.iter()) {
            // Stable SiLU: x / (1+exp(-x)) for x>=0 else x*exp(x)/(1+exp(x)).
            let silu_g = if *g >= 0.0 {
                *g / (1.0 + (-*g).exp())
            } else {
                let e = g.exp();
                *g * e / (1.0 + e)
            };
            out.push(silu_g * *u);
        }
        let bytes: &[u8] = bytemuck::cast_slice(&out);
        HostTensor::new(gate.shape.clone(), DType::F32, bytes.to_vec())
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!("swiglu: build out: {e:?}")))
    }
}

// ── LayerNorm (group-aware) ────────────────────────────────────────

/// `Custom("layer_norm")` — group-aware standard LayerNorm
/// (mean + variance, scale + shift). CPU mirror of
/// `rce-op-metal::MetalLeaf`'s `layer_norm_f32_grouped` kernel.
///
/// Args: x, weight, bias, ngroups (i32), eps (f32).
/// `ngroups=1` recovers whole-vector LayerNorm.
#[derive(Debug, Default, Clone, Copy)]
pub struct LayerNormLeaf;
impl LayerNormLeaf {
    pub const fn new() -> Self { Self }
}
impl OpDispatch for LayerNormLeaf {
    fn handles(&self) -> &[&'static str] { &["layer_norm"] }
    fn execute(&self, _node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "layer_norm" {
            return Err(InterpretError::Unsupported(format!("layer_norm leaf: not `{name}`")));
        }
        if args.len() != 5 {
            return Err(InterpretError::Unsupported(format!(
                "layer_norm: expected 5 args (x, weight, bias, ngroups, eps), got {}",
                args.len()
            )));
        }
        let x = arg_tensor(&args[0], "x")?;
        let w = arg_tensor(&args[1], "weight")?;
        let b = arg_tensor(&args[2], "bias")?;
        let ngroups = arg_i32(&args[3], "ngroups")?;
        let eps = arg_f32(&args[4], "eps")?;
        if x.dtype != DType::F32 || w.dtype != DType::F32 || b.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "layer_norm: f32 only (x={}, w={}, b={})",
                x.dtype.name(), w.dtype.name(), b.dtype.name()
            )));
        }
        let total = x.numel();
        if w.numel() != total || b.numel() != total {
            return Err(InterpretError::Unsupported(format!(
                "layer_norm: weight ({}) / bias ({}) numel must equal x numel ({})",
                w.numel(), b.numel(), total
            )));
        }
        if ngroups <= 0 {
            return Err(InterpretError::Unsupported(format!(
                "layer_norm: ngroups must be positive, got {ngroups}"
            )));
        }
        let ngroups = ngroups as usize;
        if total % ngroups != 0 {
            return Err(InterpretError::Unsupported(format!(
                "layer_norm: total {total} not divisible by ngroups {ngroups}"
            )));
        }
        let group_size = total / ngroups;
        let xs = bytes_as_f32(&x.data, "x")?;
        let ws = bytes_as_f32(&w.data, "weight")?;
        let bs = bytes_as_f32(&b.data, "bias")?;
        let mut out = vec![0.0f32; total];

        for g in 0..ngroups {
            let s = g * group_size;
            let e = s + group_size;
            let slice = &xs[s..e];
            let mean = slice.iter().sum::<f32>() / group_size as f32;
            let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / group_size as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for i in 0..group_size {
                out[s + i] = (xs[s + i] - mean) * inv_std * ws[s + i] + bs[s + i];
            }
        }

        let bytes: &[u8] = bytemuck::cast_slice(&out);
        HostTensor::new(x.shape.clone(), DType::F32, bytes.to_vec())
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!("layer_norm: build out: {e:?}")))
    }
}

// ── Argmax ─────────────────────────────────────────────────────────

/// `Custom("argmax")` — index of the maximum element in a 1-D f32
/// vector. CPU mirror of `rce-op-metal::MetalLeaf`'s `argmax_f32`.
/// Output: i32 [1].
#[derive(Debug, Default, Clone, Copy)]
pub struct ArgmaxLeaf;
impl ArgmaxLeaf {
    pub const fn new() -> Self { Self }
}
impl OpDispatch for ArgmaxLeaf {
    fn handles(&self) -> &[&'static str] { &["argmax"] }
    fn execute(&self, _node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "argmax" {
            return Err(InterpretError::Unsupported(format!("argmax leaf: not `{name}`")));
        }
        if args.len() != 1 {
            return Err(InterpretError::Unsupported(format!(
                "argmax: expected 1 arg, got {}", args.len()
            )));
        }
        let t = arg_tensor(&args[0], "input")?;
        if t.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "argmax: f32 only, got {}", t.dtype.name()
            )));
        }
        let n = t.numel();
        if n == 0 {
            return Err(InterpretError::Unsupported("argmax: empty input".into()));
        }
        let xs = bytes_as_f32(&t.data, "input")?;
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx: i32 = 0;
        for (i, &v) in xs.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i as i32;
            }
        }
        HostTensor::from_i32(vec![1], &[max_idx])
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!("argmax: build out: {e:?}")))
    }
}

// ── Conv2d (naive NHWC + OHWI) ────────────────────────────────────

/// `Custom("conv2d")` — 2-D convolution. CPU mirror of
/// `rce-op-metal::MetalLeaf`'s `conv2d_naive_f32`.
///
/// Args: input f32 [N, H, W, C_in] (NHWC), weight f32 [C_out, K_h, K_w, C_in]
/// (OHWI), bias f32 [C_out], N i32, H i32, W i32, C_in i32, C_out i32,
/// K_h i32, K_w i32, S_h i32, S_w i32, P_h i32, P_w i32.
/// Output: f32 [N, H_out, W_out, C_out].
#[derive(Debug, Default, Clone, Copy)]
pub struct Conv2dLeaf;
impl Conv2dLeaf {
    pub const fn new() -> Self { Self }
}
impl OpDispatch for Conv2dLeaf {
    fn handles(&self) -> &[&'static str] { &["conv2d"] }
    fn execute(&self, _node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "conv2d" {
            return Err(InterpretError::Unsupported(format!("conv2d leaf: not `{name}`")));
        }
        if args.len() != 14 {
            return Err(InterpretError::Unsupported(format!(
                "conv2d: expected 14 args, got {}", args.len()
            )));
        }
        let input = arg_tensor(&args[0], "input")?;
        let weight = arg_tensor(&args[1], "weight")?;
        let bias = arg_tensor(&args[2], "bias")?;
        let n = arg_i32(&args[3], "N")? as usize;
        let h = arg_i32(&args[4], "H")? as usize;
        let w = arg_i32(&args[5], "W")? as usize;
        let c_in = arg_i32(&args[6], "C_in")? as usize;
        let c_out = arg_i32(&args[7], "C_out")? as usize;
        let k_h = arg_i32(&args[8], "K_h")? as usize;
        let k_w = arg_i32(&args[9], "K_w")? as usize;
        let s_h = arg_i32(&args[10], "S_h")? as usize;
        let s_w = arg_i32(&args[11], "S_w")? as usize;
        let p_h = arg_i32(&args[12], "P_h")?;
        let p_w = arg_i32(&args[13], "P_w")?;
        if p_h < 0 || p_w < 0 {
            return Err(InterpretError::Unsupported(format!(
                "conv2d: padding must be non-negative (got P_h={p_h}, P_w={p_w})"
            )));
        }
        let p_h = p_h as usize; let p_w = p_w as usize;
        if input.dtype != DType::F32 || weight.dtype != DType::F32 || bias.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "conv2d: f32 only (input={}, weight={}, bias={})",
                input.dtype.name(), weight.dtype.name(), bias.dtype.name()
            )));
        }
        let h_out = (h + 2 * p_h).saturating_sub(k_h) / s_h.max(1) + 1;
        let w_out = (w + 2 * p_w).saturating_sub(k_w) / s_w.max(1) + 1;

        let in_f = bytes_as_f32(&input.data, "input")?;
        let w_f = bytes_as_f32(&weight.data, "weight")?;
        let b_f = bytes_as_f32(&bias.data, "bias")?;

        let mut out = vec![0.0f32; n * h_out * w_out * c_out];
        for nn in 0..n {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    for co in 0..c_out {
                        let mut acc = b_f[co];
                        for kh in 0..k_h {
                            let h_in = ho as isize * s_h as isize + kh as isize - p_h as isize;
                            if h_in < 0 || h_in as usize >= h { continue; }
                            for kw in 0..k_w {
                                let w_in = wo as isize * s_w as isize + kw as isize - p_w as isize;
                                if w_in < 0 || w_in as usize >= w { continue; }
                                for ci in 0..c_in {
                                    let in_idx = ((nn*h + h_in as usize)*w + w_in as usize)*c_in + ci;
                                    let wi = ((co*k_h + kh)*k_w + kw)*c_in + ci;
                                    acc += in_f[in_idx] * w_f[wi];
                                }
                            }
                        }
                        out[((nn*h_out + ho)*w_out + wo)*c_out + co] = acc;
                    }
                }
            }
        }
        let bytes: &[u8] = bytemuck::cast_slice(&out);
        HostTensor::new(vec![n, h_out, w_out, c_out], DType::F32, bytes.to_vec())
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!("conv2d: build out: {e:?}")))
    }
}

// ── GroupNorm (NCHW, channel-grouped) ─────────────────────────────

/// `Custom("group_norm")` — channel-grouped image normalization.
/// CPU mirror of `rce-op-metal::MetalLeaf`'s `group_norm_f32` kernel.
///
/// Used by Stable Diffusion / SDXL UNet + VAE. Args: `x` f32 [N, C, HW]
/// (3-D), `gamma` f32 [C], `bias` f32 [C], `N` i32, `G` i32, `eps` f32.
/// Output: f32 [N, C, HW].
#[derive(Debug, Default, Clone, Copy)]
pub struct GroupNormLeaf;
impl GroupNormLeaf {
    pub const fn new() -> Self { Self }
}
impl OpDispatch for GroupNormLeaf {
    fn handles(&self) -> &[&'static str] { &["group_norm"] }
    fn execute(&self, _node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "group_norm" {
            return Err(InterpretError::Unsupported(format!(
                "group_norm leaf: not `{name}`"
            )));
        }
        if args.len() != 6 {
            return Err(InterpretError::Unsupported(format!(
                "group_norm: expected 6 args (x, gamma, bias, N, G, eps), got {}",
                args.len()
            )));
        }
        let x = arg_tensor(&args[0], "x")?;
        let gamma = arg_tensor(&args[1], "gamma")?;
        let bias = arg_tensor(&args[2], "bias")?;
        let n_arg = arg_i32(&args[3], "N")?;
        let g_arg = arg_i32(&args[4], "G")?;
        let eps = arg_f32(&args[5], "eps")?;
        if x.dtype != DType::F32 || gamma.dtype != DType::F32 || bias.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "group_norm: f32 only (x={}, gamma={}, bias={})",
                x.dtype.name(), gamma.dtype.name(), bias.dtype.name()
            )));
        }
        if x.shape.len() != 3 {
            return Err(InterpretError::Unsupported(format!(
                "group_norm: x must be 3-D [N, C, HW], got {:?}", x.shape
            )));
        }
        let n_x = x.shape[0]; let c = x.shape[1]; let hw = x.shape[2];
        if n_arg <= 0 || g_arg <= 0 {
            return Err(InterpretError::Unsupported(format!(
                "group_norm: N and G must be positive (got N={n_arg}, G={g_arg})"
            )));
        }
        if n_arg as usize != n_x {
            return Err(InterpretError::Unsupported(format!(
                "group_norm: N arg {n_arg} != x.shape[0] {n_x}"
            )));
        }
        let g = g_arg as usize;
        if c % g != 0 {
            return Err(InterpretError::Unsupported(format!(
                "group_norm: C={c} not divisible by G={g}"
            )));
        }
        if gamma.numel() != c || bias.numel() != c {
            return Err(InterpretError::Unsupported(format!(
                "group_norm: gamma/bias numel ({}, {}) != C={c}",
                gamma.numel(), bias.numel()
            )));
        }
        let xs = bytes_as_f32(&x.data, "x")?;
        let gs = bytes_as_f32(&gamma.data, "gamma")?;
        let bs = bytes_as_f32(&bias.data, "bias")?;
        let c_per_g = c / g;
        let num_elems = c_per_g * hw;
        let mut out = vec![0.0f32; n_x * c * hw];

        for nn in 0..n_x {
            for gg in 0..g {
                let base_c = gg * c_per_g;
                let mut sum = 0.0f32; let mut sum_sq = 0.0f32;
                for cl in 0..c_per_g {
                    let cc = base_c + cl;
                    for hh in 0..hw {
                        let v = xs[nn * c * hw + cc * hw + hh];
                        sum += v; sum_sq += v * v;
                    }
                }
                let mean = sum / num_elems as f32;
                let var = (sum_sq / num_elems as f32) - mean * mean;
                let inv_std = 1.0 / (var.max(0.0) + eps).sqrt();
                for cl in 0..c_per_g {
                    let cc = base_c + cl;
                    for hh in 0..hw {
                        let idx = nn * c * hw + cc * hw + hh;
                        out[idx] = (xs[idx] - mean) * inv_std * gs[cc] + bs[cc];
                    }
                }
            }
        }
        let bytes: &[u8] = bytemuck::cast_slice(&out);
        HostTensor::new(x.shape.clone(), DType::F32, bytes.to_vec())
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!("group_norm: build out: {e:?}")))
    }
}

// ── TopK mask ──────────────────────────────────────────────────────

/// `Custom("topk_mask")` — given logits and `top_k`, write a binary
/// mask (1.0 at the top-K positions, 0.0 elsewhere). The math primitive
/// for sparse-MoE evaluation at the IR level.
#[derive(Debug, Default, Clone, Copy)]
pub struct TopKMaskLeaf;
impl TopKMaskLeaf {
    pub const fn new() -> Self { Self }
}
impl OpDispatch for TopKMaskLeaf {
    fn handles(&self) -> &[&'static str] { &["topk_mask"] }
    fn execute(&self, _node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "topk_mask" {
            return Err(InterpretError::Unsupported(format!(
                "topk_mask leaf: not `{name}`"
            )));
        }
        if args.len() != 2 {
            return Err(InterpretError::Unsupported(format!(
                "topk_mask: expected 2 args (logits, top_k), got {}", args.len()
            )));
        }
        let logits = arg_tensor(&args[0], "logits")?;
        let top_k = arg_i32(&args[1], "top_k")?;
        if logits.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "topk_mask: f32 only, got {}", logits.dtype.name()
            )));
        }
        let n = logits.numel();
        if top_k <= 0 || (top_k as usize) > n {
            return Err(InterpretError::Unsupported(format!(
                "topk_mask: top_k must be 1..={n}, got {top_k}"
            )));
        }
        let top_k = top_k as usize;
        let logits_f = bytes_as_f32(&logits.data, "logits")?;
        let mut idx: Vec<usize> = (0..n).collect();
        idx.select_nth_unstable_by(top_k - 1, |&a, &b| {
            logits_f[b].partial_cmp(&logits_f[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(top_k);
        let mut mask = vec![0.0f32; n];
        for i in idx {
            mask[i] = 1.0;
        }
        let bytes: &[u8] = bytemuck::cast_slice(&mask);
        HostTensor::new(vec![n], DType::F32, bytes.to_vec())
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!("topk_mask: build out: {e:?}")))
    }
}

// ── Embedding lookup ───────────────────────────────────────────────

/// `Custom("embedding_lookup")` — gather rows from a vocabulary table.
///
/// Args:
/// 1. `embed_table` f32 tensor of shape `(vocab_size, hidden_size)` row-major.
/// 2. `token_ids`   i32 tensor of shape `(seq_len,)`.
/// 3. `hidden_size` i32 scalar.
///
/// Output: f32 tensor of shape `(seq_len, hidden_size)`. Out-of-vocab
/// token IDs (≥ vocab_size or negative) emit a row of zeros — matches
/// efficient-genai's defensive OOV behavior.
#[derive(Debug, Default, Clone, Copy)]
pub struct EmbeddingLookupLeaf;
impl EmbeddingLookupLeaf {
    pub const fn new() -> Self { Self }
}
impl OpDispatch for EmbeddingLookupLeaf {
    fn handles(&self) -> &[&'static str] { &["embedding_lookup"] }
    fn execute(&self, _node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "embedding_lookup" {
            return Err(InterpretError::Unsupported(format!(
                "embedding_lookup leaf: not `{name}`"
            )));
        }
        if args.len() != 3 {
            return Err(InterpretError::Unsupported(format!(
                "embedding_lookup: expected 3 args (embed_table, token_ids, hidden_size), got {}",
                args.len()
            )));
        }
        let table = arg_tensor(&args[0], "embed_table")?;
        let ids = arg_tensor(&args[1], "token_ids")?;
        let hidden = arg_i32(&args[2], "hidden_size")?;
        if table.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "embedding_lookup: embed_table f32 only, got {}", table.dtype.name()
            )));
        }
        if ids.dtype != DType::I32 && ids.dtype != DType::I64 {
            return Err(InterpretError::Unsupported(format!(
                "embedding_lookup: token_ids i32 or i64 only, got {}", ids.dtype.name()
            )));
        }
        if hidden <= 0 {
            return Err(InterpretError::Unsupported(format!(
                "embedding_lookup: hidden_size must be positive, got {hidden}"
            )));
        }
        let hidden = hidden as usize;
        let total = table.numel();
        if total % hidden != 0 {
            return Err(InterpretError::Unsupported(format!(
                "embedding_lookup: embed_table numel {total} not divisible by hidden_size {hidden}"
            )));
        }
        let vocab_size = total / hidden;
        let seq_len = ids.numel();
        let table_f = bytes_as_f32(&table.data, "embed_table")?;

        let token_ids: Vec<i64> = if ids.dtype == DType::I32 {
            ids.as_i32()?.into_iter().map(|x| x as i64).collect()
        } else {
            ids.as_i64()?
        };

        let mut out = vec![0.0f32; seq_len * hidden];
        for (i, &tok) in token_ids.iter().enumerate() {
            if tok < 0 || (tok as usize) >= vocab_size {
                continue; // zero row for OOV
            }
            let src = (tok as usize) * hidden;
            let dst = i * hidden;
            out[dst..dst + hidden].copy_from_slice(&table_f[src..src + hidden]);
        }
        let bytes: &[u8] = bytemuck::cast_slice(&out);
        HostTensor::new(vec![seq_len, hidden], DType::F32, bytes.to_vec())
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!("embedding_lookup: build out: {e:?}")))
    }
}

// ── MoE combine ────────────────────────────────────────────────────

/// `Custom("moe_combine")` — top-K weighted combination of expert outputs.
///
/// Args:
/// 1. `gate_logits`     f32 `(num_experts,)`. Raw router logits.
/// 2. `expert_outputs`  f32 `(num_experts, d_model)`. Pre-computed
///    forward outputs from each expert (caller is responsible — sparse
///    expert evaluation lives at the IR level, not this kernel).
/// 3. `top_k`           i32 scalar. K experts to keep.
///
/// Output: f32 `(d_model,)` = Σ_{k∈topK} softmax(gate_logits over topK)[k] * expert_outputs[k].
///
/// This is the math primitive Mixtral / nemotron_h_moe / DeepSeek MoE
/// uses to combine routed expert outputs. Sparse evaluation (only run
/// the top-K experts to begin with) is a graph-level optimisation —
/// this op accepts pre-computed dense expert outputs and does the
/// weighted reduce.
#[derive(Debug, Default, Clone, Copy)]
pub struct MoeCombineLeaf;
impl MoeCombineLeaf {
    pub const fn new() -> Self { Self }
}
impl OpDispatch for MoeCombineLeaf {
    fn handles(&self) -> &[&'static str] { &["moe_combine"] }
    fn execute(&self, _node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "moe_combine" {
            return Err(InterpretError::Unsupported(format!(
                "moe_combine leaf: not `{name}`"
            )));
        }
        if args.len() != 3 {
            return Err(InterpretError::Unsupported(format!(
                "moe_combine: expected 3 args (gate_logits, expert_outputs, top_k), got {}",
                args.len()
            )));
        }
        let logits = arg_tensor(&args[0], "gate_logits")?;
        let experts = arg_tensor(&args[1], "expert_outputs")?;
        let top_k = arg_i32(&args[2], "top_k")?;
        if logits.dtype != DType::F32 || experts.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "moe_combine: f32 only (gate_logits={}, expert_outputs={})",
                logits.dtype.name(), experts.dtype.name()
            )));
        }
        let num_experts = logits.numel();
        if num_experts == 0 {
            return Err(InterpretError::Unsupported(
                "moe_combine: num_experts == 0".into(),
            ));
        }
        if experts.numel() % num_experts != 0 {
            return Err(InterpretError::Unsupported(format!(
                "moe_combine: expert_outputs numel {} not divisible by num_experts {}",
                experts.numel(), num_experts
            )));
        }
        let d_model = experts.numel() / num_experts;
        if top_k <= 0 || (top_k as usize) > num_experts {
            return Err(InterpretError::Unsupported(format!(
                "moe_combine: top_k must be in 1..={num_experts}, got {top_k}"
            )));
        }
        let top_k = top_k as usize;

        let logits_f = bytes_as_f32(&logits.data, "gate_logits")?;
        let experts_f = bytes_as_f32(&experts.data, "expert_outputs")?;

        // Top-K selection by logits (no full sort needed; keep K best).
        let mut idx: Vec<usize> = (0..num_experts).collect();
        idx.select_nth_unstable_by(top_k - 1, |&a, &b| {
            logits_f[b].partial_cmp(&logits_f[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(top_k);

        // Softmax over the top-K logits only (typical MoE convention).
        let max_l = idx.iter().map(|&i| logits_f[i]).fold(f32::NEG_INFINITY, f32::max);
        let mut weights = vec![0.0f32; top_k];
        let mut sum = 0.0f32;
        for (slot, &i) in idx.iter().enumerate() {
            let e = (logits_f[i] - max_l).exp();
            weights[slot] = e;
            sum += e;
        }
        for w in weights.iter_mut() { *w /= sum; }

        let mut out = vec![0.0f32; d_model];
        for (slot, &expert_id) in idx.iter().enumerate() {
            let w = weights[slot];
            let base = expert_id * d_model;
            for d in 0..d_model {
                out[d] += w * experts_f[base + d];
            }
        }

        let bytes: &[u8] = bytemuck::cast_slice(&out);
        HostTensor::new(vec![d_model], DType::F32, bytes.to_vec())
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!("moe_combine: build out: {e:?}")))
    }
}

// ── SplitAt ────────────────────────────────────────────────────────

/// `Custom("split_at")` — extract `[start, start+len)` from a 1-D tensor.
///
/// Args:
/// 1. `x`     f32 tensor (1-D).
/// 2. `start` i32 scalar.
/// 3. `len`   i32 scalar.
///
/// Output: f32 1-D tensor of length `len`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SplitAtLeaf;
impl SplitAtLeaf {
    pub const fn new() -> Self { Self }
}
impl OpDispatch for SplitAtLeaf {
    fn handles(&self) -> &[&'static str] { &["split_at"] }
    fn execute(&self, _node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "split_at" {
            return Err(InterpretError::Unsupported(format!("split_at leaf: not `{name}`")));
        }
        if args.len() != 3 {
            return Err(InterpretError::Unsupported(format!(
                "split_at: expected 3 args (x, start, len), got {}", args.len()
            )));
        }
        let x = arg_tensor(&args[0], "x")?;
        let start = arg_i32(&args[1], "start")?;
        let len = arg_i32(&args[2], "len")?;
        if x.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "split_at: f32 only, got {}", x.dtype.name()
            )));
        }
        if start < 0 || len < 0 {
            return Err(InterpretError::Unsupported(format!(
                "split_at: negative start ({start}) or len ({len})"
            )));
        }
        let start = start as usize;
        let len = len as usize;
        let total = x.numel();
        if start + len > total {
            return Err(InterpretError::Unsupported(format!(
                "split_at: start({start}) + len({len}) > x.numel({total})"
            )));
        }
        let xs = bytes_as_f32(&x.data, "x")?;
        let out: Vec<f32> = xs[start..start + len].to_vec();
        let bytes: &[u8] = bytemuck::cast_slice(&out);
        HostTensor::new(vec![len], DType::F32, bytes.to_vec())
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!("split_at: build out: {e:?}")))
    }
}

// ── Concat ─────────────────────────────────────────────────────────

/// `Custom("concat")` — concatenate N 1-D f32 tensors in arg order.
/// The inverse of `split_at`; the v0.1 interpreter explicitly punts on
/// the `OpKind::Concat` primitive for tensors, so this leaf supplies it.
///
/// Needed to thread state out of a graph-routed mixer: a `FunctionGraph`
/// has one `output_node`, so the per-token forward returns a packed
/// `[out ‖ new_conv_state ‖ new_ssm_state]` the Rust driver splits.
///
/// Args: ≥1 f32 1-D tensors. Output: their concatenation.
#[derive(Debug, Default, Clone, Copy)]
pub struct ConcatLeaf;
impl ConcatLeaf {
    pub const fn new() -> Self { Self }
}
impl OpDispatch for ConcatLeaf {
    fn handles(&self) -> &[&'static str] { &["concat"] }
    fn execute(&self, _node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "concat" {
            return Err(InterpretError::Unsupported(format!("concat leaf: not `{name}`")));
        }
        if args.is_empty() {
            return Err(InterpretError::Unsupported("concat: need ≥1 arg".into()));
        }
        let mut out: Vec<f32> = Vec::new();
        for (i, v) in args.iter().enumerate() {
            let t = arg_tensor(v, "concat arg")?;
            if t.dtype != DType::F32 {
                return Err(InterpretError::Unsupported(format!(
                    "concat: arg {i} not f32 ({})", t.dtype.name()
                )));
            }
            out.extend_from_slice(bytes_as_f32(&t.data, "concat arg")?);
        }
        let bytes: &[u8] = bytemuck::cast_slice(&out);
        HostTensor::new(vec![out.len()], DType::F32, bytes.to_vec())
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!("concat: build out: {e:?}")))
    }
}

// ── Sparse-MoE runtime helper ─────────────────────────────────────

/// Sparse-MoE evaluation helper: given gate logits and a per-expert
/// evaluator closure, only invoke the closure for the top-K experts —
/// non-routed experts cost zero compute. Combines the results via the
/// same softmax-over-top-K weighting `moe_combine` uses.
///
/// **What this is and isn't:**
/// - This is a *runtime* helper: callers explicitly pass `eval_expert`
///   as a closure whose call cost is dominated by the expert's FFN.
///   Skipping the closure for non-routed experts is what makes MoE
///   sparse in practice.
/// - This is *not* an IR-level rewrite of a `FunctionGraph` — that is
///   the next deliverable. The substrate carries the sparse structure
///   today (via `Custom("topk_mask")` Apply nodes flowing into
///   `Custom("moe_combine")`); a graph-level rewriter that
///   dead-code-eliminates non-routed expert sub-graphs is documented
///   in `vertical.md` §3 as the pull-forward.
///
/// Returns `output[d_model] = Σ_{k∈topK} softmax(logits over topK)[k] * eval_expert(k)`.
pub fn sparse_moe_eval<F>(
    gate_logits: &[f32],
    top_k: usize,
    d_model: usize,
    mut eval_expert: F,
) -> Vec<f32>
where
    F: FnMut(usize) -> Vec<f32>,
{
    let num_experts = gate_logits.len();
    assert!(top_k > 0 && top_k <= num_experts, "invalid top_k {top_k} for num_experts {num_experts}");

    // Top-K + softmax weights (host-side).
    let mut idx: Vec<usize> = (0..num_experts).collect();
    idx.select_nth_unstable_by(top_k - 1, |&a, &b| {
        gate_logits[b].partial_cmp(&gate_logits[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(top_k);
    let max_l = idx.iter().map(|&i| gate_logits[i]).fold(f32::NEG_INFINITY, f32::max);
    let mut weights = vec![0.0f32; top_k];
    let mut sum = 0.0f32;
    for (slot, &i) in idx.iter().enumerate() {
        let e = (gate_logits[i] - max_l).exp();
        weights[slot] = e;
        sum += e;
    }
    for w in weights.iter_mut() { *w /= sum; }

    // Run only the top-K experts, weighted-combine.
    let mut out = vec![0.0f32; d_model];
    for (slot, &expert_id) in idx.iter().enumerate() {
        let w = weights[slot];
        let expert_out = eval_expert(expert_id);
        assert_eq!(expert_out.len(), d_model, "expert {expert_id} returned wrong size");
        for d in 0..d_model {
            out[d] += w * expert_out[d];
        }
    }
    out
}

// ── helpers ────────────────────────────────────────────────────────

fn arg_tensor<'a>(v: &'a Value, label: &str) -> Result<&'a HostTensor> {
    match v {
        Value::Tensor(t) => Ok(t),
        other => Err(InterpretError::Unsupported(format!(
            "nn-prim: {label} expected tensor, got {other:?}"
        ))),
    }
}

fn arg_i32(v: &Value, label: &str) -> Result<i32> {
    match v {
        Value::Scalar(ScalarValue::I32(x)) => Ok(*x),
        Value::Scalar(ScalarValue::I64(x)) => Ok(*x as i32),
        other => Err(InterpretError::Unsupported(format!(
            "nn-prim: {label} expected i32 scalar, got {other:?}"
        ))),
    }
}

fn arg_f32(v: &Value, label: &str) -> Result<f32> {
    match v {
        Value::Scalar(ScalarValue::F32(x)) => Ok(*x),
        Value::Scalar(ScalarValue::F64(x)) => Ok(*x as f32),
        other => Err(InterpretError::Unsupported(format!(
            "nn-prim: {label} expected f32 scalar, got {other:?}"
        ))),
    }
}

fn bytes_as_f32<'a>(bytes: &'a [u8], label: &str) -> Result<&'a [f32]> {
    bytemuck::try_cast_slice(bytes).map_err(|e| {
        InterpretError::Unsupported(format!(
            "nn-prim: {label} bytes not f32-aligned ({e})"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rce_vm_backend_cpu::HostTensor;

    fn t1(v: &[f32]) -> Value {
        Value::Tensor(HostTensor::from_f32(vec![v.len()], v).unwrap())
    }

    #[test]
    fn silu_matches_reference() {
        let xs = [-2.0f32, -0.5, 0.0, 0.5, 2.0, 5.0, -10.0, 10.0];
        // Reference SiLU values.
        let want: Vec<f32> = xs.iter().map(|&x| {
            // SiLU(x) = x / (1 + exp(-x)) (stable form).
            if x >= 0.0 { x / (1.0 + (-x).exp()) } else { let e = x.exp(); x * e / (1.0 + e) }
        }).collect();

        let leaf = SiluLeaf::new();
        let result = leaf.execute(0, "silu", &[t1(&xs)]).unwrap();
        let out = match result { Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!() };

        for (i, (a, b)) in want.iter().zip(out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-7, "silu[{i}]: want {a}, got {b}");
        }
    }

    #[test]
    fn rms_norm_ngroups_one() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![1.0f32, 1.0, 1.0, 1.0];
        let leaf = RmsNormLeaf::new();
        let args = vec![
            t1(&x), t1(&w),
            Value::Scalar(ScalarValue::I32(1)),
            Value::Scalar(ScalarValue::F32(1e-6)),
        ];
        let result = leaf.execute(0, "rms_norm", &args).unwrap();
        let out = match result { Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!() };

        let mean_sq: f32 = (1.0 + 4.0 + 9.0 + 16.0) / 4.0; // 7.5
        let rms = (mean_sq + 1e-6_f32).sqrt();
        let want: Vec<f32> = x.iter().map(|v| v / rms).collect();
        for (i, (a, b)) in want.iter().zip(out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "rms_norm[{i}]: want {a}, got {b}");
        }
    }

    #[test]
    fn rms_norm_ngroups_two() {
        // Two groups of 2 elements each, normalised independently.
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![2.0f32, 0.5, 1.0, 1.0];
        let leaf = RmsNormLeaf::new();
        let args = vec![
            t1(&x), t1(&w),
            Value::Scalar(ScalarValue::I32(2)),
            Value::Scalar(ScalarValue::F32(0.0)),
        ];
        let result = leaf.execute(0, "rms_norm", &args).unwrap();
        let out = match result { Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!() };

        // Group 0: [1, 2] → rms = sqrt(2.5) → x/rms * w
        let r0 = 2.5f32.sqrt();
        let r1 = 12.5f32.sqrt();
        let want = vec![
            1.0 / r0 * 2.0, 2.0 / r0 * 0.5,
            3.0 / r1 * 1.0, 4.0 / r1 * 1.0,
        ];
        for (i, (a, b)) in want.iter().zip(out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "rms_norm[{i}]: want {a}, got {b}");
        }
    }

    #[test]
    fn gelu_matches_inline_reference() {
        let xs = [-2.0f32, -0.5, 0.0, 0.5, 2.0, 5.0, -10.0, 10.0];
        let want: Vec<f32> = xs.iter().map(|&x| {
            let x3 = x * x * x;
            let mut inner = 0.7978845608028654f32 * (x + 0.044715f32 * x3);
            if inner > 10.0 { inner = 10.0; } else if inner < -10.0 { inner = -10.0; }
            0.5 * x * (1.0 + inner.tanh())
        }).collect();

        let leaf = GeluLeaf::new();
        let result = leaf.execute(0, "gelu", &[t1(&xs)]).unwrap();
        let out = match result { Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!() };
        for (i, (a, b)) in want.iter().zip(out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "gelu[{i}]: want {a}, got {b}");
        }
    }

    #[test]
    fn swiglu_matches_inline_reference() {
        let gate = vec![-1.0f32, 0.0, 1.0, 2.0];
        let up = vec![0.5f32, 1.0, -0.3, 0.8];
        let want: Vec<f32> = gate.iter().zip(up.iter()).map(|(g, u)| {
            let silu_g = if *g >= 0.0 {
                *g / (1.0 + (-*g).exp())
            } else {
                let e = g.exp();
                *g * e / (1.0 + e)
            };
            silu_g * *u
        }).collect();

        let leaf = SwigluLeaf::new();
        let result = leaf.execute(0, "swiglu", &[t1(&gate), t1(&up)]).unwrap();
        let out = match result { Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!() };
        for (i, (a, b)) in want.iter().zip(out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-7, "swiglu[{i}]: want {a}, got {b}");
        }
    }

    #[test]
    fn layer_norm_ngroups_one_matches_inline_reference() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let w = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let b = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let eps = 1e-6f32;

        let mean = x.iter().sum::<f32>() / 8.0;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 8.0;
        let inv_std = 1.0 / (var + eps).sqrt();
        let want: Vec<f32> = x.iter().enumerate().map(|(i, v)| {
            (v - mean) * inv_std * w[i] + b[i]
        }).collect();

        let leaf = LayerNormLeaf::new();
        let args = vec![
            t1(&x), t1(&w), t1(&b),
            Value::Scalar(ScalarValue::I32(1)),
            Value::Scalar(ScalarValue::F32(eps)),
        ];
        let result = leaf.execute(0, "layer_norm", &args).unwrap();
        let out = match result { Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!() };
        for (i, (a, b)) in want.iter().zip(out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "layer_norm[{i}]: want {a}, got {b}");
        }
    }

    #[test]
    fn layer_norm_ngroups_two_matches_inline_reference() {
        // Two groups of 4 elements normalised independently.
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let w = vec![1.0f32; 8];
        let b = vec![0.0f32; 8];
        let eps = 0.0;

        let mean0: f32 = (1.0 + 2.0 + 3.0 + 4.0) / 4.0;
        let var0: f32 = ((1.0_f32 - mean0).powi(2) + (2.0_f32 - mean0).powi(2) + (3.0_f32 - mean0).powi(2) + (4.0_f32 - mean0).powi(2)) / 4.0;
        let mean1: f32 = (10.0 + 20.0 + 30.0 + 40.0) / 4.0;
        let var1: f32 = ((10.0_f32 - mean1).powi(2) + (20.0_f32 - mean1).powi(2) + (30.0_f32 - mean1).powi(2) + (40.0_f32 - mean1).powi(2)) / 4.0;
        let inv_std0: f32 = 1.0 / var0.sqrt();
        let inv_std1: f32 = 1.0 / var1.sqrt();
        let want = vec![
            (1.0 - mean0) * inv_std0, (2.0 - mean0) * inv_std0,
            (3.0 - mean0) * inv_std0, (4.0 - mean0) * inv_std0,
            (10.0 - mean1) * inv_std1, (20.0 - mean1) * inv_std1,
            (30.0 - mean1) * inv_std1, (40.0 - mean1) * inv_std1,
        ];

        let leaf = LayerNormLeaf::new();
        let args = vec![
            t1(&x), t1(&w), t1(&b),
            Value::Scalar(ScalarValue::I32(2)),
            Value::Scalar(ScalarValue::F32(eps)),
        ];
        let result = leaf.execute(0, "layer_norm", &args).unwrap();
        let out = match result { Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!() };
        for (i, (a, b)) in want.iter().zip(out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "layer_norm[{i}]: want {a}, got {b}");
        }
    }

    #[test]
    fn sparse_moe_eval_only_invokes_top_k_experts() {
        // 8 experts, top_k=2, d_model=4. Logits picked so experts 1 and 5 win.
        let logits = vec![0.1f32, 2.0, 0.5, 0.3, -1.0, 1.5, 0.8, 0.0];
        let top_k = 2;
        let d_model = 4;

        // Expert i returns a vector of all `i as f32`. We track which
        // experts were actually evaluated to confirm only top-K are called.
        let mut called: Vec<usize> = Vec::new();
        let eval = |i: usize| -> Vec<f32> {
            called.push(i);
            vec![i as f32; d_model]
        };
        let out = sparse_moe_eval(&logits, top_k, d_model, eval);

        // Top-2 by logit: expert 1 (logit 2.0), expert 5 (logit 1.5).
        let mut called_sorted = called.clone();
        called_sorted.sort();
        assert_eq!(called_sorted, vec![1usize, 5], "expected only experts 1, 5");

        // Softmax([2.0, 1.5]) = [exp(0.5), exp(0)] / sum
        // After max-subtraction: [exp(0), exp(-0.5)] / sum
        let e1: f32 = 1.0;
        let e5: f32 = (-0.5_f32).exp();
        let s = e1 + e5;
        let w1 = e1 / s;
        let w5 = e5 / s;
        // Output[d] = w1 * 1.0 + w5 * 5.0 (constant per d).
        let expected = w1 * 1.0 + w5 * 5.0;
        for v in &out {
            assert!((*v - expected).abs() < 1e-5, "want {expected}, got {v}");
        }
    }

    #[test]
    fn argmax_basic() {
        let xs = vec![0.1f32, 0.5, -1.0, 0.7, 0.3];
        let leaf = ArgmaxLeaf::new();
        let result = leaf.execute(0, "argmax", &[t1(&xs)]).unwrap();
        let out = match result { Value::Tensor(t) => t.as_i32().unwrap(), _ => panic!() };
        assert_eq!(out, vec![3]);
    }

    #[test]
    fn conv2d_basic_3x3() {
        // Tiny 3×3 conv with single-channel identity-like weight.
        let n = 1usize; let h = 3usize; let w = 3usize;
        let c_in = 1usize; let c_out = 1usize;
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // [1, 3, 3, 1]
        // Identity 3x3 weight (all ones, no bias) → output[i] = sum(window).
        let weight = vec![1.0f32; 9]; // [1, 3, 3, 1]
        let bias = vec![0.0f32];
        let leaf = Conv2dLeaf::new();
        let args = vec![
            Value::Tensor(HostTensor::from_f32(vec![n, h, w, c_in], &input).unwrap()),
            Value::Tensor(HostTensor::from_f32(vec![c_out, 3, 3, c_in], &weight).unwrap()),
            Value::Tensor(HostTensor::from_f32(vec![c_out], &bias).unwrap()),
            Value::Scalar(ScalarValue::I32(n as i32)),
            Value::Scalar(ScalarValue::I32(h as i32)),
            Value::Scalar(ScalarValue::I32(w as i32)),
            Value::Scalar(ScalarValue::I32(c_in as i32)),
            Value::Scalar(ScalarValue::I32(c_out as i32)),
            Value::Scalar(ScalarValue::I32(3)),
            Value::Scalar(ScalarValue::I32(3)),
            Value::Scalar(ScalarValue::I32(1)),
            Value::Scalar(ScalarValue::I32(1)),
            Value::Scalar(ScalarValue::I32(1)),
            Value::Scalar(ScalarValue::I32(1)),
        ];
        let result = leaf.execute(0, "conv2d", &args).unwrap();
        let out = match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3, 3, 1]);
                t.as_f32().unwrap()
            }
            _ => panic!(),
        };
        // Center pixel (1,1) = sum of all 9 = 45.
        assert_eq!(out[1 * 3 + 1], 45.0);
        // Corner (0,0) = sum of 4 in-bounds [1,2,4,5] = 12.
        assert_eq!(out[0], 12.0);
    }

    #[test]
    fn group_norm_matches_inline_reference() {
        // Tiny example: N=1, C=4, G=2, HW=2.
        let x: Vec<f32> = vec![
            1.0, 2.0,    // c=0
            3.0, 4.0,    // c=1
            5.0, 6.0,    // c=2
            7.0, 8.0,    // c=3
        ];
        let gamma = vec![1.0f32; 4];
        let bias = vec![0.0f32; 4];
        let leaf = GroupNormLeaf::new();
        let result = leaf.execute(0, "group_norm", &[
            Value::Tensor(HostTensor::from_f32(vec![1, 4, 2], &x).unwrap()),
            Value::Tensor(HostTensor::from_f32(vec![4], &gamma).unwrap()),
            Value::Tensor(HostTensor::from_f32(vec![4], &bias).unwrap()),
            Value::Scalar(ScalarValue::I32(1)),
            Value::Scalar(ScalarValue::I32(2)),
            Value::Scalar(ScalarValue::F32(0.0)),
        ]).unwrap();
        let out = match result { Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!() };

        // Group 0: c=0,1 → values [1,2,3,4] → mean=2.5, var=1.25, inv_std=1/sqrt(1.25)
        // Group 1: c=2,3 → values [5,6,7,8] → mean=6.5, var=1.25, inv_std=1/sqrt(1.25)
        let inv0: f32 = 1.0 / 1.25_f32.sqrt();
        let want = vec![
            (1.0 - 2.5) * inv0, (2.0 - 2.5) * inv0,
            (3.0 - 2.5) * inv0, (4.0 - 2.5) * inv0,
            (5.0 - 6.5) * inv0, (6.0 - 6.5) * inv0,
            (7.0 - 6.5) * inv0, (8.0 - 6.5) * inv0,
        ];
        for (i, (a, b)) in want.iter().zip(out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "group_norm[{i}]: want {a}, got {b}");
        }
    }

    #[test]
    fn topk_mask_basic() {
        let logits = vec![0.1f32, 0.9, 0.3, 0.7, 0.2, 0.5, 0.6, 0.4];
        let leaf = TopKMaskLeaf::new();
        let result = leaf.execute(0, "topk_mask", &[
            t1(&logits),
            Value::Scalar(ScalarValue::I32(3)),
        ]).unwrap();
        let out = match result { Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!() };
        // Top-3: indices [1, 3, 6] (logits 0.9, 0.7, 0.6).
        assert_eq!(out, vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn embedding_lookup_basic() {
        // Vocab of 5 tokens, hidden_size=4.
        // table[i] = [i*4, i*4+1, i*4+2, i*4+3]
        let table: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let leaf = EmbeddingLookupLeaf::new();

        // Token IDs [2, 0, 4, 7] — last is OOV (vocab=5), should give zeros.
        let ids = vec![2i32, 0, 4, 7];
        let ids_t = Value::Tensor(HostTensor::from_i32(vec![ids.len()], &ids).unwrap());
        let table_t = Value::Tensor(HostTensor::from_f32(vec![5, 4], &table).unwrap());
        let result = leaf.execute(0, "embedding_lookup", &[
            table_t, ids_t,
            Value::Scalar(ScalarValue::I32(4)),
        ]).unwrap();
        let out = match result { Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!() };

        assert_eq!(out.len(), 16);
        // Token 2: rows [8, 9, 10, 11].
        assert_eq!(&out[0..4], &[8.0, 9.0, 10.0, 11.0]);
        // Token 0: [0, 1, 2, 3].
        assert_eq!(&out[4..8], &[0.0, 1.0, 2.0, 3.0]);
        // Token 4: [16, 17, 18, 19].
        assert_eq!(&out[8..12], &[16.0, 17.0, 18.0, 19.0]);
        // OOV: zeros.
        assert_eq!(&out[12..16], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn moe_combine_top_k_weighted() {
        // 4 experts, d_model=3, top_k=2.
        // Logits picked so experts 1 and 3 win (largest values).
        let logits = vec![0.0f32, 2.0, 0.5, 3.0];
        // Expert outputs: each row = expert_id * 10 + offset.
        let experts: Vec<f32> = vec![
            1.0, 2.0, 3.0,        // expert 0
            10.0, 20.0, 30.0,     // expert 1 (selected)
            100.0, 200.0, 300.0,  // expert 2
            1000.0, 2000.0, 3000.0, // expert 3 (selected)
        ];
        let leaf = MoeCombineLeaf::new();
        let result = leaf.execute(0, "moe_combine", &[
            Value::Tensor(HostTensor::from_f32(vec![4], &logits).unwrap()),
            Value::Tensor(HostTensor::from_f32(vec![4, 3], &experts).unwrap()),
            Value::Scalar(ScalarValue::I32(2)),
        ]).unwrap();
        let out = match result { Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!() };

        // softmax([2, 3]) = [0.2689, 0.7311]
        let max = 3.0f32;
        let e1 = (2.0 - max).exp();
        let e3 = (3.0 - max).exp();
        let sum = e1 + e3;
        let w1 = e1 / sum;
        let w3 = e3 / sum;
        let want = [
            w1 * 10.0 + w3 * 1000.0,
            w1 * 20.0 + w3 * 2000.0,
            w1 * 30.0 + w3 * 3000.0,
        ];
        for (i, (a, b)) in want.iter().zip(out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "moe_combine[{i}]: want {a}, got {b}");
        }
    }

    #[test]
    fn moe_combine_top_k_equals_num_experts_is_dense_softmax() {
        // top_k = num_experts → standard dense softmax-weighted combine.
        let logits = vec![0.5f32, 1.0, 1.5, 2.0];
        let experts: Vec<f32> = vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
        ];
        let leaf = MoeCombineLeaf::new();
        let result = leaf.execute(0, "moe_combine", &[
            Value::Tensor(HostTensor::from_f32(vec![4], &logits).unwrap()),
            Value::Tensor(HostTensor::from_f32(vec![4, 2], &experts).unwrap()),
            Value::Scalar(ScalarValue::I32(4)),
        ]).unwrap();
        let out = match result { Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!() };

        // Softmax over all 4 logits.
        let max = 2.0f32;
        let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let s = exps.iter().sum::<f32>();
        let w: Vec<f32> = exps.iter().map(|e| e / s).collect();
        let want = [
            w[0] * 1.0 + w[1] * 3.0 + w[2] * 5.0 + w[3] * 7.0,
            w[0] * 2.0 + w[1] * 4.0 + w[2] * 6.0 + w[3] * 8.0,
        ];
        for (i, (a, b)) in want.iter().zip(out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "moe_combine[{i}]: want {a}, got {b}");
        }
    }

    #[test]
    fn split_at_basic() {
        let x = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
        let leaf = SplitAtLeaf::new();
        let args = vec![
            t1(&x),
            Value::Scalar(ScalarValue::I32(1)),
            Value::Scalar(ScalarValue::I32(3)),
        ];
        let result = leaf.execute(0, "split_at", &args).unwrap();
        let out = match result { Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!() };
        assert_eq!(out, vec![20.0, 30.0, 40.0]);
    }

    #[test]
    fn split_at_rejects_oob() {
        let x = vec![10.0f32, 20.0, 30.0];
        let leaf = SplitAtLeaf::new();
        let args = vec![
            t1(&x),
            Value::Scalar(ScalarValue::I32(2)),
            Value::Scalar(ScalarValue::I32(5)),
        ];
        let r = leaf.execute(0, "split_at", &args);
        assert!(matches!(r, Err(InterpretError::Unsupported(_))), "{r:?}");
    }

    #[test]
    fn concat_is_inverse_of_split_at() {
        // concat([a, b, c]) == a ‖ b ‖ c, bit-for-bit; and splitting it
        // back by the original lengths round-trips. This is exactly how
        // the graph-routed mixer will pack [out ‖ conv_state ‖ ssm_state]
        // and the Rust driver will unpack it.
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![-4.0f32, 5.5];
        let c = vec![6.0f32, 7.0, 8.0, 9.0];
        let leaf = ConcatLeaf::new();
        let out = match leaf
            .execute(0, "concat", &[t1(&a), t1(&b), t1(&c)])
            .unwrap()
        {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let mut want = a.clone();
        want.extend_from_slice(&b);
        want.extend_from_slice(&c);
        assert_eq!(out, want, "concat must be bit-identical to ‖");

        // Round-trip via split_at (the inverse the driver relies on).
        let sp = SplitAtLeaf::new();
        let s = |start: i32, len: i32| match sp
            .execute(0, "split_at", &[t1(&out), Value::Scalar(ScalarValue::I32(start)), Value::Scalar(ScalarValue::I32(len))])
            .unwrap()
        {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        assert_eq!(s(0, 3), a);
        assert_eq!(s(3, 2), b);
        assert_eq!(s(5, 4), c);
    }
}
