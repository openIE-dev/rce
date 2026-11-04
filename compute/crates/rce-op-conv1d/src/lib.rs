//! `rce-op-conv1d` — `OpDispatch` leaf for causal depthwise 1D
//! convolution with sliding-window state. This is the conv1d gate
//! inside Mamba2 / Jamba / nemotron_h SSM layers; it runs once per
//! decode step on a single new token, sliding the existing kernel
//! window forward by one and computing the per-channel dot product.
//!
//! Math is ported from efficient-genai's
//! `src/inference/architecture/mamba2.rs::causal_conv1d` so the
//! rce path is bit-for-bit equivalent.
//!
//! ## Op contract — `Custom("conv1d_causal_depthwise")`
//!
//! Stateless. The caller threads the conv state (the last `d_conv`
//! input values per channel) through the graph as `state_in` and
//! receives `state_out` packed in the result tensor.
//!
//! Inputs (4 tensors + 1 scalar):
//!
//! 1. `x`        — f32 `(conv_dim,)`. New input frame at this timestep.
//! 2. `weight`   — f32 `(conv_dim * d_conv,)` row-major over channels-then-kernel.
//! 3. `bias`     — f32 `(conv_dim,)`. Per-channel bias (use a zero tensor if absent).
//! 4. `state_in` — f32 `(conv_dim * d_conv,)`. Sliding window of past inputs per channel.
//! 5. `d_conv`   — i32 scalar. Kernel width (typically 4).
//!
//! Output: f32 `(conv_dim + conv_dim * d_conv,)` — `[y || state_out]`
//! packed. Caller splits at offset `conv_dim`.

use rce_vm_backend_cpu::{
    DType, HostTensor, InterpretError, OpDispatch, Result, ScalarValue, Value,
};
use tracing::trace;

/// Causal-depthwise-conv1d leaf. Stateless ZST.
#[derive(Debug, Default, Clone, Copy)]
pub struct Conv1dCausalLeaf;

impl Conv1dCausalLeaf {
    pub const fn new() -> Self {
        Self
    }
}

impl OpDispatch for Conv1dCausalLeaf {
    fn handles(&self) -> &[&'static str] {
        &["conv1d_causal_depthwise"]
    }

    fn execute(&self, _node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "conv1d_causal_depthwise" {
            return Err(InterpretError::Unsupported(format!(
                "conv1d_causal_depthwise leaf does not handle `{name}`"
            )));
        }
        if args.len() != 5 {
            return Err(InterpretError::Unsupported(format!(
                "conv1d_causal_depthwise: expected 5 args (x, weight, bias, state_in, d_conv), got {}",
                args.len()
            )));
        }

        let x = arg_tensor(&args[0], "x")?;
        let w = arg_tensor(&args[1], "weight")?;
        let bias = arg_tensor(&args[2], "bias")?;
        let state_in = arg_tensor(&args[3], "state_in")?;
        let d_conv = arg_i32(&args[4], "d_conv")?;

        for (label, t) in [("x", x), ("weight", w), ("bias", bias), ("state_in", state_in)] {
            if t.dtype != DType::F32 {
                return Err(InterpretError::Unsupported(format!(
                    "conv1d_causal_depthwise: f32 only ({label} is {})",
                    t.dtype.name()
                )));
            }
        }
        if d_conv <= 0 {
            return Err(InterpretError::Unsupported(format!(
                "conv1d_causal_depthwise: d_conv must be positive, got {d_conv}"
            )));
        }
        let d_conv = d_conv as usize;

        let conv_dim = x.numel();
        if bias.numel() != conv_dim {
            return Err(InterpretError::Unsupported(format!(
                "conv1d_causal_depthwise: bias numel {} != conv_dim {}",
                bias.numel(), conv_dim
            )));
        }
        if w.numel() != conv_dim * d_conv {
            return Err(InterpretError::Unsupported(format!(
                "conv1d_causal_depthwise: weight numel {} != conv_dim*d_conv {}",
                w.numel(), conv_dim * d_conv
            )));
        }
        if state_in.numel() != conv_dim * d_conv {
            return Err(InterpretError::Unsupported(format!(
                "conv1d_causal_depthwise: state_in numel {} != conv_dim*d_conv {}",
                state_in.numel(), conv_dim * d_conv
            )));
        }

        let x_v = bytes_as_f32(&x.data, "x")?;
        let w_v = bytes_as_f32(&w.data, "weight")?;
        let bias_v = bytes_as_f32(&bias.data, "bias")?;
        let state_v = bytes_as_f32(&state_in.data, "state_in")?;

        trace!(conv_dim, d_conv, "conv1d_causal_depthwise: dispatching scalar reference");

        let (y, state_out) = conv1d_causal_step(x_v, w_v, bias_v, state_v, conv_dim, d_conv);

        let total = conv_dim + conv_dim * d_conv;
        let mut packed = Vec::with_capacity(total);
        packed.extend_from_slice(&y);
        packed.extend_from_slice(&state_out);
        let bytes: &[u8] = bytemuck::cast_slice(&packed);
        HostTensor::new(vec![total], DType::F32, bytes.to_vec())
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!(
                "conv1d_causal_depthwise: build output tensor: {e:?}"
            )))
    }
}

/// Pure scalar reference: shift the conv state left by one, append
/// the new input, then compute the depthwise dot product per channel.
///
/// Mirrors efficient-genai's `causal_conv1d` (mamba2.rs lines 574-611).
pub fn conv1d_causal_step(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    state_in: &[f32],
    conv_dim: usize,
    d_conv: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut state_out = state_in.to_vec();

    // Shift left by one and append new input at position d_conv-1.
    for d in 0..conv_dim {
        for k in 0..d_conv - 1 {
            state_out[d * d_conv + k] = state_out[d * d_conv + k + 1];
        }
        state_out[d * d_conv + d_conv - 1] = x[d];
    }

    // Depthwise dot product per channel + bias.
    let mut y = vec![0.0f32; conv_dim];
    for d in 0..conv_dim {
        let mut val = bias[d];
        for k in 0..d_conv {
            val += state_out[d * d_conv + k] * weight[d * d_conv + k];
        }
        y[d] = val;
    }

    (y, state_out)
}

/// **T-token batched** causal depthwise conv: [`conv1d_causal_step`]
/// looped over `t_rows` tokens, threading the conv ring state from one
/// token to the next. **Definitionally bit-identical** to calling
/// `conv1d_causal_step` `t_rows` times with the state carried forward —
/// same shift-append, same dot-product order, byte-for-byte.
///
/// The prefill-batching counterpart of the single-step conv. The conv is
/// inherently sequential (each token's state is the prior token's window
/// shifted by one) and trivially cheap per token, so there is no speedup
/// *here* — the prefill win is in the batched `qmm` matmuls. The value is
/// structural: the batched mixer graph invokes **one** conv leaf instead
/// of orchestrating a `t_rows`-long loop of single-step leaves.
///
/// `x` is row-major `[t_rows, conv_dim]`. `weight` `[conv_dim*d_conv]`
/// and `bias` `[conv_dim]` are shared across tokens. `state_in`
/// `[conv_dim*d_conv]` is the single pre-window ring state.
///
/// Returns `(y[t_rows*conv_dim], state_out[conv_dim*d_conv])` — all
/// token outputs stacked row-major, plus the post-window ring state.
pub fn conv1d_causal_step_batched(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    state_in: &[f32],
    t_rows: usize,
    conv_dim: usize,
    d_conv: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut y_all = vec![0.0f32; t_rows * conv_dim];
    let mut state = state_in.to_vec();
    for t in 0..t_rows {
        let xt = &x[t * conv_dim..(t + 1) * conv_dim];
        let (yt, st) = conv1d_causal_step(xt, weight, bias, &state, conv_dim, d_conv);
        y_all[t * conv_dim..(t + 1) * conv_dim].copy_from_slice(&yt);
        state = st;
    }
    (y_all, state)
}

/// `OpDispatch` leaf claiming `Custom("conv1d_causal_depthwise_batched")`.
/// Same arg order as [`Conv1dCausalLeaf`] with a trailing `t_rows`
/// scalar (6 args: x, weight, bias, state_in, d_conv, t_rows). `x` is
/// `[t_rows, conv_dim]`; `weight`/`bias`/`state_in` are the shared
/// single-token tensors. Routes to [`conv1d_causal_step_batched`] and
/// packs `[y (t_rows*conv_dim) || state_out (conv_dim*d_conv)]`.
pub struct Conv1dCausalBatchedLeaf;

impl Conv1dCausalBatchedLeaf {
    pub const fn new() -> Self {
        Self
    }
}

impl OpDispatch for Conv1dCausalBatchedLeaf {
    fn handles(&self) -> &[&'static str] {
        &["conv1d_causal_depthwise_batched"]
    }

    fn execute(&self, _node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "conv1d_causal_depthwise_batched" {
            return Err(InterpretError::Unsupported(format!(
                "conv1d_causal_depthwise_batched leaf does not handle `{name}`"
            )));
        }
        if args.len() != 6 {
            return Err(InterpretError::Unsupported(format!(
                "conv1d_causal_depthwise_batched: expected 6 args (x, weight, bias, state_in, d_conv, t_rows), got {}",
                args.len()
            )));
        }

        let x = arg_tensor(&args[0], "x")?;
        let w = arg_tensor(&args[1], "weight")?;
        let bias = arg_tensor(&args[2], "bias")?;
        let state_in = arg_tensor(&args[3], "state_in")?;
        let d_conv = arg_i32(&args[4], "d_conv")?;
        let t_rows = arg_i32(&args[5], "t_rows")?;

        for (label, t) in [("x", x), ("weight", w), ("bias", bias), ("state_in", state_in)] {
            if t.dtype != DType::F32 {
                return Err(InterpretError::Unsupported(format!(
                    "conv1d_causal_depthwise_batched: f32 only ({label} is {})",
                    t.dtype.name()
                )));
            }
        }
        if d_conv <= 0 || t_rows <= 0 {
            return Err(InterpretError::Unsupported(format!(
                "conv1d_causal_depthwise_batched: d_conv and t_rows must be positive, got {d_conv}, {t_rows}"
            )));
        }
        let d_conv = d_conv as usize;
        let t_rows = t_rows as usize;

        // bias is the shared [conv_dim] vector — fixes conv_dim.
        let conv_dim = bias.numel();
        if x.numel() != t_rows * conv_dim {
            return Err(InterpretError::Unsupported(format!(
                "conv1d_causal_depthwise_batched: x numel {} != t_rows*conv_dim {}",
                x.numel(), t_rows * conv_dim
            )));
        }
        if w.numel() != conv_dim * d_conv {
            return Err(InterpretError::Unsupported(format!(
                "conv1d_causal_depthwise_batched: weight numel {} != conv_dim*d_conv {}",
                w.numel(), conv_dim * d_conv
            )));
        }
        if state_in.numel() != conv_dim * d_conv {
            return Err(InterpretError::Unsupported(format!(
                "conv1d_causal_depthwise_batched: state_in numel {} != conv_dim*d_conv {}",
                state_in.numel(), conv_dim * d_conv
            )));
        }

        let x_v = bytes_as_f32(&x.data, "x")?;
        let w_v = bytes_as_f32(&w.data, "weight")?;
        let bias_v = bytes_as_f32(&bias.data, "bias")?;
        let state_v = bytes_as_f32(&state_in.data, "state_in")?;

        trace!(conv_dim, d_conv, t_rows, "conv1d_causal_depthwise_batched: dispatching scalar reference");

        let (y, state_out) =
            conv1d_causal_step_batched(x_v, w_v, bias_v, state_v, t_rows, conv_dim, d_conv);

        let total = t_rows * conv_dim + conv_dim * d_conv;
        let mut packed = Vec::with_capacity(total);
        packed.extend_from_slice(&y);
        packed.extend_from_slice(&state_out);
        let bytes: &[u8] = bytemuck::cast_slice(&packed);
        HostTensor::new(vec![total], DType::F32, bytes.to_vec())
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!(
                "conv1d_causal_depthwise_batched: build output tensor: {e:?}"
            )))
    }
}

fn arg_tensor<'a>(v: &'a Value, label: &str) -> Result<&'a HostTensor> {
    match v {
        Value::Tensor(t) => Ok(t),
        other => Err(InterpretError::Unsupported(format!(
            "conv1d_causal_depthwise: {label} expected tensor, got {other:?}"
        ))),
    }
}

fn arg_i32(v: &Value, label: &str) -> Result<i32> {
    match v {
        Value::Scalar(ScalarValue::I32(x)) => Ok(*x),
        Value::Scalar(ScalarValue::I64(x)) => Ok(*x as i32),
        other => Err(InterpretError::Unsupported(format!(
            "conv1d_causal_depthwise: {label} expected i32 scalar, got {other:?}"
        ))),
    }
}

fn bytes_as_f32<'a>(bytes: &'a [u8], label: &str) -> Result<&'a [f32]> {
    bytemuck::try_cast_slice(bytes).map_err(|e| {
        InterpretError::Unsupported(format!(
            "conv1d_causal_depthwise: {label} bytes not f32-aligned ({e})"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation copied verbatim from efficient-genai
    /// mamba2.rs::causal_conv1d (lines 574-611). Mutates state in place.
    fn efficient_genai_reference(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        state: &mut [f32],
        conv_dim: usize,
        d_conv: usize,
    ) -> Vec<f32> {
        for d in 0..conv_dim {
            for k in 0..d_conv - 1 {
                state[d * d_conv + k] = state[d * d_conv + k + 1];
            }
            state[d * d_conv + d_conv - 1] = x[d];
        }

        let mut output = vec![0.0f32; conv_dim];
        for d in 0..conv_dim {
            let mut val = bias[d];
            for k in 0..d_conv {
                val += state[d * d_conv + k] * weight[d * d_conv + k];
            }
            output[d] = val;
        }
        output
    }

    fn make_inputs(conv_dim: usize, d_conv: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let x: Vec<f32> = (0..conv_dim).map(|i| (i as f32 * 0.013) - 0.4).collect();
        let weight: Vec<f32> = (0..conv_dim * d_conv)
            .map(|i| (i as f32 * 0.007) - 0.2)
            .collect();
        let bias: Vec<f32> = (0..conv_dim).map(|i| (i as f32 * 0.003) + 0.1).collect();
        let state: Vec<f32> = (0..conv_dim * d_conv)
            .map(|i| ((i as f32 * 0.005) - 0.3).cos())
            .collect();
        (x, weight, bias, state)
    }

    #[test]
    fn matches_reference_small() {
        let (conv_dim, d_conv) = (16, 4);
        let (x, w, b, s0) = make_inputs(conv_dim, d_conv);

        let mut ref_state = s0.clone();
        let want_y = efficient_genai_reference(&x, &w, &b, &mut ref_state, conv_dim, d_conv);
        let (got_y, got_state) = conv1d_causal_step(&x, &w, &b, &s0, conv_dim, d_conv);

        for (i, (a, b)) in want_y.iter().zip(got_y.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "y[{i}]: want {a}, got {b}");
        }
        for (i, (a, b)) in ref_state.iter().zip(got_state.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "state[{i}]: want {a}, got {b}");
        }
    }

    #[test]
    fn matches_reference_mamba2_2_7b_shapes() {
        // mamba2-2.7b conv_dim = d_inner + 2*ngroups*d_state
        //              = 5120 + 2*1*128 = 5376; d_conv = 4.
        let (conv_dim, d_conv) = (5376, 4);
        let (x, w, b, s0) = make_inputs(conv_dim, d_conv);

        let mut ref_state = s0.clone();
        let want_y = efficient_genai_reference(&x, &w, &b, &mut ref_state, conv_dim, d_conv);
        let (got_y, got_state) = conv1d_causal_step(&x, &w, &b, &s0, conv_dim, d_conv);

        let max_y_err = want_y.iter().zip(got_y.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let max_state_err = ref_state.iter().zip(got_state.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(max_y_err < 1e-6, "max_y_err {max_y_err}");
        assert!(max_state_err < 1e-6, "max_state_err {max_state_err}");
    }

    #[test]
    fn op_dispatch_matches_free_function() {
        use rce_vm_backend_cpu::{HostTensor, ScalarValue, Value};

        let (conv_dim, d_conv) = (32, 4);
        let (x, w, b, s0) = make_inputs(conv_dim, d_conv);
        let (want_y, want_state) = conv1d_causal_step(&x, &w, &b, &s0, conv_dim, d_conv);

        let to_t = |shape: Vec<usize>, v: &[f32]| -> Value {
            Value::Tensor(HostTensor::from_f32(shape, v).unwrap())
        };
        let leaf = Conv1dCausalLeaf::new();
        let args = vec![
            to_t(vec![conv_dim], &x),
            to_t(vec![conv_dim * d_conv], &w),
            to_t(vec![conv_dim], &b),
            to_t(vec![conv_dim * d_conv], &s0),
            Value::Scalar(ScalarValue::I32(d_conv as i32)),
        ];
        let result = leaf.execute(0, "conv1d_causal_depthwise", &args).unwrap();
        let out = match result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor, got {other:?}"),
        };
        let packed = out.as_f32().unwrap();
        assert_eq!(packed.len(), conv_dim + conv_dim * d_conv);
        let got_y = &packed[..conv_dim];
        let got_state = &packed[conv_dim..];

        for (i, (a, b)) in want_y.iter().zip(got_y.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "y[{i}]: want {a}, got {b}");
        }
        for (i, (a, b)) in want_state.iter().zip(got_state.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "state[{i}]: want {a}, got {b}");
        }
    }

    /// Brick 3 contract: the T-token batched conv must be **byte-identical**
    /// to calling `conv1d_causal_step` `T` times with the ring state
    /// threaded forward — and the leaf the VM invokes must be byte-identical
    /// to the free function. Per-token inputs are perturbed by `t` so the
    /// test exercises distinct tokens + genuine ring-state threading, not a
    /// degenerate all-identical window.
    #[test]
    fn conv1d_batched_bit_exact_vs_t_single_steps_and_leaf() {
        use rce_vm_backend_cpu::{HostTensor, ScalarValue, Value};

        let (conv_dim, d_conv) = (48, 4);
        let t_rows = 7usize;
        let (x0, w, b, s0) = make_inputs(conv_dim, d_conv);

        let mut x_all = Vec::with_capacity(t_rows * conv_dim);
        for t in 0..t_rows {
            x_all.extend(x0.iter().map(|&v| v + (t as f32) * 0.011));
        }

        // Reference: T single steps, ring state threaded by hand.
        let mut ref_state = s0.clone();
        let mut ref_y = Vec::with_capacity(t_rows * conv_dim);
        for t in 0..t_rows {
            let (yt, st) = conv1d_causal_step(
                &x_all[t * conv_dim..(t + 1) * conv_dim],
                &w, &b, &ref_state, conv_dim, d_conv,
            );
            ref_y.extend_from_slice(&yt);
            ref_state = st;
        }

        // Batched free function.
        let (y_bat, st_bat) =
            conv1d_causal_step_batched(&x_all, &w, &b, &s0, t_rows, conv_dim, d_conv);
        assert_eq!(y_bat.len(), t_rows * conv_dim);
        assert_eq!(st_bat.len(), conv_dim * d_conv);
        for (i, (a, b)) in y_bat.iter().zip(ref_y.iter()).enumerate() {
            assert_eq!(a, b, "y[{i}]: batched {a} ≠ T×single-step {b} (must be bit-identical)");
        }
        for (i, (a, b)) in st_bat.iter().zip(ref_state.iter()).enumerate() {
            assert_eq!(a, b, "state[{i}]: batched {a} ≠ T×single-step {b} (must be bit-identical)");
        }

        // Through the leaf (the OpDispatch the VM invokes).
        let to_t = |v: &[f32]| Value::Tensor(HostTensor::from_f32(vec![v.len()], v).unwrap());
        let leaf = Conv1dCausalBatchedLeaf::new();
        let args = vec![
            to_t(&x_all), to_t(&w), to_t(&b), to_t(&s0),
            Value::Scalar(ScalarValue::I32(d_conv as i32)),
            Value::Scalar(ScalarValue::I32(t_rows as i32)),
        ];
        let out = match leaf
            .execute(0, "conv1d_causal_depthwise_batched", &args)
            .expect("conv1d_causal_depthwise_batched execute")
        {
            Value::Tensor(t) => t.as_f32().expect("f32 out"),
            other => panic!("non-tensor result {other:?}"),
        };
        assert_eq!(out.len(), t_rows * conv_dim + conv_dim * d_conv);
        let (y_leaf, st_leaf) = out.split_at(t_rows * conv_dim);
        for (i, (a, b)) in y_leaf.iter().zip(y_bat.iter()).enumerate() {
            assert_eq!(a, b, "y[{i}]: leaf {a} ≠ free fn {b} (must be bit-identical)");
        }
        for (i, (a, b)) in st_leaf.iter().zip(st_bat.iter()).enumerate() {
            assert_eq!(a, b, "state[{i}]: leaf {a} ≠ free fn {b} (must be bit-identical)");
        }
    }

    #[test]
    fn rejects_dtype_mismatch() {
        use rce_vm_backend_cpu::{DType, HostTensor, ScalarValue, Value};

        let leaf = Conv1dCausalLeaf::new();
        let bad_x = HostTensor::new(vec![16], DType::I32, vec![0u8; 16 * 4]).unwrap();
        let zero = |n: usize| Value::Tensor(HostTensor::from_f32(vec![n], &vec![0.0; n]).unwrap());
        let args = vec![
            Value::Tensor(bad_x),
            zero(64), zero(16), zero(64),
            Value::Scalar(ScalarValue::I32(4)),
        ];
        let result = leaf.execute(0, "conv1d_causal_depthwise", &args);
        match result {
            Err(InterpretError::Unsupported(msg)) => {
                assert!(msg.contains("f32"), "expected f32 complaint, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
