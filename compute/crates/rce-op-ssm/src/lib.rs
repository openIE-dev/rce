//! `rce-op-ssm` — `OpDispatch` leaf for Mamba2-style selective
//! state-space scan, the post-attention dominant op in 2026 hybrid
//! architectures (Mamba2, Jamba, Nemotron-H, all the variants).
//!
//! The IR/dispatch shape mirrors `rce-op-attention`: one `Custom`
//! op fuses the entire per-token SSM step (softplus / -exp / discrete
//! recurrence / readout / D-skip) into a single leaf execution.
//! Multiple leaves can claim `Custom("ssm_scan")` — CPU reference
//! here, Metal-backed in `rce-op-metal`, CUDA-backed eventually —
//! and the dispatch chain falls through on `Unsupported(_)`.
//!
//! Math is ported from efficient-genai's
//! `src/inference/architecture/mamba2.rs::forward_ssd_mixer` so the
//! rce path is bit-for-bit equivalent to the existing in-house
//! runtime. That equivalence is the dogfood: the same numerical
//! behaviour, exposed through a content-addressed `.fg` graph
//! instead of a hand-rolled Rust function, so the substrate gets
//! its IR / capabilities / energy accounting / cross-language
//! reachability "for free."
//!
//! ## Op contract — `Custom("ssm_scan")`
//!
//! Stateless. The caller threads recurrent state through the graph
//! as `state_in` (tensor input) and receives `state_out` packed in
//! the result tensor. **Single decode step** semantics — full
//! prefill is composed by the caller chaining decode steps over
//! the prompt or by a future `ssm_prefill` op.
//!
//! Inputs (8 tensors, in this order):
//!
//! 1. `x`        — f32 `(d_inner,)`. Post-conv1d, post-SiLU SSM input.
//! 2. `B`        — f32 `(ngroups * d_state,)`. Input-dependent state matrix.
//! 3. `C`        — f32 `(ngroups * d_state,)`. Input-dependent readout matrix.
//! 4. `dt_raw`   — f32 `(nheads,)`. Raw dt (pre softplus).
//! 5. `dt_bias`  — f32 `(nheads,)`. Per-head bias added before softplus.
//! 6. `A_log`    — f32 `(nheads,)`. Log of negative state-decay (`A = -exp(A_log)`).
//! 7. `D`        — f32 `(nheads,)`. Skip-connection scalar per head.
//! 8. `state_in` — f32 `(nheads * headdim * d_state,)`. Recurrent state at t-1.
//!
//! Plus 2 scalar literals as the last two args:
//!
//! 9.  `headdim` — i32. Channels per head.
//! 10. `d_state` — i32. Inner state dim per head.
//! (`nheads` and `ngroups` are derived from tensor shapes:
//!  `nheads = dt_raw.numel()`,
//!  `ngroups = B.numel() / d_state`.)
//!
//! Output: f32 `(d_inner + nheads * headdim * d_state,)` — `[y || state_out]`
//! packed. Caller splits at offset `d_inner` (which equals
//! `nheads * headdim`).

use rce_vm_backend_cpu::{
    DType, HostTensor, InterpretError, OpDispatch, Result, ScalarValue, Value,
};
use tracing::trace;

/// Selective-state-space scan leaf. Stateless ZST, claims `Custom("ssm_scan")`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SsmScanLeaf;

impl SsmScanLeaf {
    pub const fn new() -> Self {
        Self
    }
}

impl OpDispatch for SsmScanLeaf {
    fn handles(&self) -> &[&'static str] {
        &["ssm_scan"]
    }

    fn execute(&self, _node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "ssm_scan" {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan leaf does not handle `{name}`"
            )));
        }
        if args.len() != 10 {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan: expected 10 args (x, B, C, dt_raw, dt_bias, A_log, D, state_in, headdim, d_state), got {}",
                args.len()
            )));
        }

        let x = arg_tensor(&args[0], "x")?;
        let b = arg_tensor(&args[1], "B")?;
        let c = arg_tensor(&args[2], "C")?;
        let dt_raw = arg_tensor(&args[3], "dt_raw")?;
        let dt_bias = arg_tensor(&args[4], "dt_bias")?;
        let a_log = arg_tensor(&args[5], "A_log")?;
        let d_param = arg_tensor(&args[6], "D")?;
        let state_in = arg_tensor(&args[7], "state_in")?;
        let headdim = arg_i32(&args[8], "headdim")?;
        let d_state = arg_i32(&args[9], "d_state")?;

        for (label, t) in [
            ("x", x), ("B", b), ("C", c),
            ("dt_raw", dt_raw), ("dt_bias", dt_bias),
            ("A_log", a_log), ("D", d_param),
            ("state_in", state_in),
        ] {
            if t.dtype != DType::F32 {
                return Err(InterpretError::Unsupported(format!(
                    "ssm_scan: f32 only ({label} is {})",
                    t.dtype.name()
                )));
            }
        }
        if headdim <= 0 || d_state <= 0 {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan: headdim and d_state must be positive (got {headdim}, {d_state})"
            )));
        }
        let headdim = headdim as usize;
        let d_state = d_state as usize;

        let nheads = dt_raw.numel();
        if dt_bias.numel() != nheads || a_log.numel() != nheads || d_param.numel() != nheads {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan: dt_raw/dt_bias/A_log/D nheads mismatch ({}, {}, {}, {})",
                dt_raw.numel(), dt_bias.numel(), a_log.numel(), d_param.numel()
            )));
        }
        let d_inner = nheads * headdim;
        if x.numel() != d_inner {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan: x has {} elements, expected nheads*headdim = {}",
                x.numel(), d_inner
            )));
        }
        if b.numel() != c.numel() {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan: B ({}) and C ({}) must have same numel",
                b.numel(), c.numel()
            )));
        }
        if b.numel() % d_state != 0 {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan: B numel {} not divisible by d_state {d_state}",
                b.numel()
            )));
        }
        let ngroups = b.numel() / d_state;
        if nheads % ngroups != 0 {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan: nheads {nheads} not divisible by ngroups {ngroups}"
            )));
        }
        let state_size = nheads * headdim * d_state;
        if state_in.numel() != state_size {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan: state_in has {} elements, expected nheads*headdim*d_state = {}",
                state_in.numel(), state_size
            )));
        }

        let x_v = bytes_as_f32(&x.data, "x")?;
        let b_v = bytes_as_f32(&b.data, "B")?;
        let c_v = bytes_as_f32(&c.data, "C")?;
        let dt_raw_v = bytes_as_f32(&dt_raw.data, "dt_raw")?;
        let dt_bias_v = bytes_as_f32(&dt_bias.data, "dt_bias")?;
        let a_log_v = bytes_as_f32(&a_log.data, "A_log")?;
        let d_v = bytes_as_f32(&d_param.data, "D")?;
        let state_v = bytes_as_f32(&state_in.data, "state_in")?;

        trace!(
            d_inner, nheads, headdim, d_state, ngroups,
            "ssm_scan_leaf: dispatching scalar reference"
        );

        let (y, state_out) = selective_scan_step(
            x_v, b_v, c_v, dt_raw_v, dt_bias_v, a_log_v, d_v, state_v,
            nheads, headdim, d_state, ngroups,
        );

        // Pack [y (d_inner) || state_out (state_size)] into the output tensor.
        let total = d_inner + state_size;
        let mut packed = Vec::with_capacity(total);
        packed.extend_from_slice(&y);
        packed.extend_from_slice(&state_out);

        let bytes: &[u8] = bytemuck::cast_slice(&packed);
        HostTensor::new(vec![total], DType::F32, bytes.to_vec())
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!(
                "ssm_scan: build output tensor: {e:?}"
            )))
    }
}

/// Pure scalar reference selective-scan step.
///
/// Mirrors the math at
/// `efficient-genai/src/inference/architecture/mamba2.rs::forward_ssd_mixer`
/// (lines 488–534): softplus(dt+bias), discretise A_bar = exp(dt*A),
/// recurrence `state := A_bar * state + dt * B * x`, readout
/// `y = C·state + D*x`.
///
/// Returns `(y, state_out)` as fresh `Vec<f32>` so the caller can
/// thread state through the IR without aliasing concerns.
pub fn selective_scan_step(
    x: &[f32],
    b: &[f32],
    c: &[f32],
    dt_raw: &[f32],
    dt_bias: &[f32],
    a_log: &[f32],
    d_param: &[f32],
    state_in: &[f32],
    nheads: usize,
    headdim: usize,
    d_state: usize,
    ngroups: usize,
) -> (Vec<f32>, Vec<f32>) {
    let d_inner = nheads * headdim;
    let nheads_per_group = nheads / ngroups;

    // dt = softplus(dt_raw + dt_bias).
    // **A is taken from `a_log` directly** — despite the parameter name,
    // ggml/llama.cpp's `convert_hf_to_gguf.py` applies `data = -data.exp()`
    // to any `*.A_log` tensor before writing to GGUF, so the `ssm_a`
    // tensor that flows in here already holds `A = -exp(A_log)`. Applying
    // `-exp()` again here was killing all SSM decay (kept the historical
    // bug for 6 sessions; see `project_gguf_ssm_a_convention.md`).
    let mut dt = vec![0.0f32; nheads];
    let mut a = vec![0.0f32; nheads];
    for h in 0..nheads {
        dt[h] = softplus(dt_raw[h] + dt_bias[h]);
        a[h] = a_log[h];
    }

    let mut y = vec![0.0f32; d_inner];
    let mut state_out = state_in.to_vec();

    for h in 0..nheads {
        let group = h / nheads_per_group;
        let a_bar = (dt[h] * a[h]).exp();
        let dt_h = dt[h];

        for d in 0..headdim {
            let x_val = x[h * headdim + d];
            let mut y_val = 0.0f32;

            for s in 0..d_state {
                let b_val = b[group * d_state + s];
                let c_val = c[group * d_state + s];

                let state_idx = (h * headdim + d) * d_state + s;
                let new_state = a_bar * state_out[state_idx] + dt_h * b_val * x_val;
                state_out[state_idx] = new_state;

                y_val += c_val * new_state;
            }

            y[h * headdim + d] = y_val + d_param[h] * x_val;
        }
    }

    (y, state_out)
}

/// **f64-accumulator** Mamba-2 selective scan. Bit-identical math to
/// `selective_scan_step` but the inner d_state recurrence runs in f64,
/// then narrows to f32. This is the variant the production `rce-mamba2`
/// runtime uses (it chose f64 during the SSM bit-exact debugging because
/// f32 accumulation drifted ~5%/forward, garbaging logits over 64 deep
/// SSM blocks). Lifted verbatim from `rce-mamba2`'s local copy so a
/// graph-routed mixer can bit-match the hand-written forward.
///
/// `ssm_a` is already `A = -exp(A_log)` per GGUF convention — used
/// directly (see `project_gguf_ssm_a_convention.md`).
pub fn selective_scan_step_f64(
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

/// `OpDispatch` leaf claiming `Custom("ssm_scan_f64")`. Identical 10-arg
/// contract and packed `[y || state_out]` output as [`SsmScanLeaf`], but
/// routes to [`selective_scan_step_f64`] so a VM-executed mamba2 mixer
/// is bit-identical to `rce-mamba2`'s hand-written f64 forward.
pub struct SsmScanF64Leaf;

impl SsmScanF64Leaf {
    pub const fn new() -> Self {
        Self
    }
}

impl OpDispatch for SsmScanF64Leaf {
    fn handles(&self) -> &[&'static str] {
        &["ssm_scan_f64"]
    }

    fn execute(&self, _node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "ssm_scan_f64" {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_f64 leaf does not handle `{name}`"
            )));
        }
        if args.len() != 10 {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_f64: expected 10 args (x, B, C, dt_raw, dt_bias, A_log, D, state_in, headdim, d_state), got {}",
                args.len()
            )));
        }
        let x = arg_tensor(&args[0], "x")?;
        let b = arg_tensor(&args[1], "B")?;
        let c = arg_tensor(&args[2], "C")?;
        let dt_raw = arg_tensor(&args[3], "dt_raw")?;
        let dt_bias = arg_tensor(&args[4], "dt_bias")?;
        let a_log = arg_tensor(&args[5], "A_log")?;
        let d_param = arg_tensor(&args[6], "D")?;
        let state_in = arg_tensor(&args[7], "state_in")?;
        let headdim = arg_i32(&args[8], "headdim")?;
        let d_state = arg_i32(&args[9], "d_state")?;

        for (label, t) in [
            ("x", x), ("B", b), ("C", c),
            ("dt_raw", dt_raw), ("dt_bias", dt_bias),
            ("A_log", a_log), ("D", d_param),
            ("state_in", state_in),
        ] {
            if t.dtype != DType::F32 {
                return Err(InterpretError::Unsupported(format!(
                    "ssm_scan_f64: f32 only ({label} is {})",
                    t.dtype.name()
                )));
            }
        }
        if headdim <= 0 || d_state <= 0 {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_f64: headdim and d_state must be positive (got {headdim}, {d_state})"
            )));
        }
        let headdim = headdim as usize;
        let d_state = d_state as usize;
        let nheads = dt_raw.numel();
        if dt_bias.numel() != nheads || a_log.numel() != nheads || d_param.numel() != nheads {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_f64: dt_raw/dt_bias/A_log/D nheads mismatch ({}, {}, {}, {})",
                dt_raw.numel(), dt_bias.numel(), a_log.numel(), d_param.numel()
            )));
        }
        let d_inner = nheads * headdim;
        if x.numel() != d_inner {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_f64: x has {} elements, expected nheads*headdim = {}",
                x.numel(), d_inner
            )));
        }
        if b.numel() != c.numel() || b.numel() % d_state != 0 {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_f64: B ({}) / C ({}) bad vs d_state {d_state}",
                b.numel(), c.numel()
            )));
        }
        let ngroups = b.numel() / d_state;
        if nheads % ngroups != 0 {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_f64: nheads {nheads} not divisible by ngroups {ngroups}"
            )));
        }
        let state_size = nheads * headdim * d_state;
        if state_in.numel() != state_size {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_f64: state_in has {} elements, expected {}",
                state_in.numel(), state_size
            )));
        }

        let x_v = bytes_as_f32(&x.data, "x")?;
        let b_v = bytes_as_f32(&b.data, "B")?;
        let c_v = bytes_as_f32(&c.data, "C")?;
        let dt_raw_v = bytes_as_f32(&dt_raw.data, "dt_raw")?;
        let dt_bias_v = bytes_as_f32(&dt_bias.data, "dt_bias")?;
        let a_log_v = bytes_as_f32(&a_log.data, "A_log")?;
        let d_v = bytes_as_f32(&d_param.data, "D")?;
        let state_v = bytes_as_f32(&state_in.data, "state_in")?;

        let (y, state_out) = selective_scan_step_f64(
            x_v, b_v, c_v, dt_raw_v, dt_bias_v, a_log_v, d_v, state_v,
            nheads, headdim, d_state, ngroups,
        );

        let total = d_inner + state_size;
        let mut packed = Vec::with_capacity(total);
        packed.extend_from_slice(&y);
        packed.extend_from_slice(&state_out);
        let bytes: &[u8] = bytemuck::cast_slice(&packed);
        HostTensor::new(vec![total], DType::F32, bytes.to_vec())
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!(
                "ssm_scan_f64: build output tensor: {e:?}"
            )))
    }
}

/// **T-token batched** f64 selective scan: [`selective_scan_step_f64`]
/// looped over `t_rows` tokens, threading the recurrent state from one
/// token to the next. **Definitionally bit-identical** to calling
/// `selective_scan_step_f64` `t_rows` times in a row with the state
/// carried forward — same f64 accumulation, same op order, byte-for-byte.
///
/// This is the prefill-batching counterpart of the single-step scan. The
/// scan itself is sequential (each token depends on the prior state) and
/// compute-bound on KB-scale data, so there is no speedup *here* — the
/// prefill win lives entirely in the two batched `qmm` matmuls around it.
/// The value of this function is structural: the batched mixer graph
/// invokes **one** scan leaf instead of orchestrating a `t_rows`-long
/// loop of single-step leaves in the IR.
///
/// Per-token inputs are row-major `[t_rows, ·]`:
/// - `x`      `[t_rows * d_inner]`
/// - `b`,`c`  `[t_rows * ngroups * d_state]`
/// - `dt_raw` `[t_rows * nheads]`
///
/// Static (shared across all tokens): `dt_bias`, `a`, `d_param` `[nheads]`.
/// `state_in` `[nheads*headdim*d_state]` is the single pre-window state.
///
/// Returns `(y[t_rows * d_inner], state_out[nheads*headdim*d_state])` —
/// all token outputs stacked row-major, plus the post-window state.
#[allow(clippy::too_many_arguments)]
pub fn selective_scan_step_f64_batched(
    x: &[f32], b: &[f32], c: &[f32], dt_raw: &[f32], dt_bias: &[f32],
    a: &[f32], d_param: &[f32], state_in: &[f32],
    t_rows: usize, nheads: usize, headdim: usize, d_state: usize, ngroups: usize,
) -> (Vec<f32>, Vec<f32>) {
    let d_inner = nheads * headdim;
    let bc_per = ngroups * d_state;
    let mut y_all = vec![0.0f32; t_rows * d_inner];
    let mut state = state_in.to_vec();
    for t in 0..t_rows {
        let xt = &x[t * d_inner..(t + 1) * d_inner];
        let bt = &b[t * bc_per..(t + 1) * bc_per];
        let ct = &c[t * bc_per..(t + 1) * bc_per];
        let dt_t = &dt_raw[t * nheads..(t + 1) * nheads];
        let (yt, st) = selective_scan_step_f64(
            xt, bt, ct, dt_t, dt_bias, a, d_param, &state,
            nheads, headdim, d_state, ngroups,
        );
        y_all[t * d_inner..(t + 1) * d_inner].copy_from_slice(&yt);
        state = st;
    }
    (y_all, state)
}

/// `OpDispatch` leaf claiming `Custom("ssm_scan_f64_batched")`. Same
/// arg order as [`SsmScanF64Leaf`] with a trailing `t_rows` scalar
/// (11 args: x, B, C, dt_raw, dt_bias, A_log, D, state_in, headdim,
/// d_state, t_rows). Per-token tensors are `[t_rows, ·]`; `dt_bias`,
/// `A_log`, `D` are the static `[nheads]` vectors. Routes to
/// [`selective_scan_step_f64_batched`] and packs
/// `[y (t_rows*d_inner) || state_out (state_size)]`.
pub struct SsmScanF64BatchedLeaf;

impl SsmScanF64BatchedLeaf {
    pub const fn new() -> Self {
        Self
    }
}

impl OpDispatch for SsmScanF64BatchedLeaf {
    fn handles(&self) -> &[&'static str] {
        &["ssm_scan_f64_batched"]
    }

    fn execute(&self, _node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "ssm_scan_f64_batched" {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_f64_batched leaf does not handle `{name}`"
            )));
        }
        if args.len() != 11 {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_f64_batched: expected 11 args (x, B, C, dt_raw, dt_bias, A_log, D, state_in, headdim, d_state, t_rows), got {}",
                args.len()
            )));
        }
        let x = arg_tensor(&args[0], "x")?;
        let b = arg_tensor(&args[1], "B")?;
        let c = arg_tensor(&args[2], "C")?;
        let dt_raw = arg_tensor(&args[3], "dt_raw")?;
        let dt_bias = arg_tensor(&args[4], "dt_bias")?;
        let a_log = arg_tensor(&args[5], "A_log")?;
        let d_param = arg_tensor(&args[6], "D")?;
        let state_in = arg_tensor(&args[7], "state_in")?;
        let headdim = arg_i32(&args[8], "headdim")?;
        let d_state = arg_i32(&args[9], "d_state")?;
        let t_rows = arg_i32(&args[10], "t_rows")?;

        for (label, t) in [
            ("x", x), ("B", b), ("C", c),
            ("dt_raw", dt_raw), ("dt_bias", dt_bias),
            ("A_log", a_log), ("D", d_param),
            ("state_in", state_in),
        ] {
            if t.dtype != DType::F32 {
                return Err(InterpretError::Unsupported(format!(
                    "ssm_scan_f64_batched: f32 only ({label} is {})",
                    t.dtype.name()
                )));
            }
        }
        if headdim <= 0 || d_state <= 0 || t_rows <= 0 {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_f64_batched: headdim, d_state, t_rows must be positive (got {headdim}, {d_state}, {t_rows})"
            )));
        }
        let headdim = headdim as usize;
        let d_state = d_state as usize;
        let t_rows = t_rows as usize;
        // Static [nheads] vectors fix nheads; per-token tensors must scale by t_rows.
        let nheads = dt_bias.numel();
        if a_log.numel() != nheads || d_param.numel() != nheads {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_f64_batched: dt_bias/A_log/D nheads mismatch ({}, {}, {})",
                dt_bias.numel(), a_log.numel(), d_param.numel()
            )));
        }
        if dt_raw.numel() != t_rows * nheads {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_f64_batched: dt_raw has {} elements, expected t_rows*nheads = {}",
                dt_raw.numel(), t_rows * nheads
            )));
        }
        let d_inner = nheads * headdim;
        if x.numel() != t_rows * d_inner {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_f64_batched: x has {} elements, expected t_rows*nheads*headdim = {}",
                x.numel(), t_rows * d_inner
            )));
        }
        if b.numel() != c.numel() || b.numel() % (t_rows * d_state) != 0 {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_f64_batched: B ({}) / C ({}) bad vs t_rows {t_rows} d_state {d_state}",
                b.numel(), c.numel()
            )));
        }
        let ngroups = b.numel() / (t_rows * d_state);
        if nheads % ngroups != 0 {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_f64_batched: nheads {nheads} not divisible by ngroups {ngroups}"
            )));
        }
        let state_size = nheads * headdim * d_state;
        if state_in.numel() != state_size {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_f64_batched: state_in has {} elements, expected {}",
                state_in.numel(), state_size
            )));
        }

        let x_v = bytes_as_f32(&x.data, "x")?;
        let b_v = bytes_as_f32(&b.data, "B")?;
        let c_v = bytes_as_f32(&c.data, "C")?;
        let dt_raw_v = bytes_as_f32(&dt_raw.data, "dt_raw")?;
        let dt_bias_v = bytes_as_f32(&dt_bias.data, "dt_bias")?;
        let a_log_v = bytes_as_f32(&a_log.data, "A_log")?;
        let d_v = bytes_as_f32(&d_param.data, "D")?;
        let state_v = bytes_as_f32(&state_in.data, "state_in")?;

        let (y, state_out) = selective_scan_step_f64_batched(
            x_v, b_v, c_v, dt_raw_v, dt_bias_v, a_log_v, d_v, state_v,
            t_rows, nheads, headdim, d_state, ngroups,
        );

        let total = t_rows * d_inner + state_size;
        let mut packed = Vec::with_capacity(total);
        packed.extend_from_slice(&y);
        packed.extend_from_slice(&state_out);
        let bytes: &[u8] = bytemuck::cast_slice(&packed);
        HostTensor::new(vec![total], DType::F32, bytes.to_vec())
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!(
                "ssm_scan_f64_batched: build output tensor: {e:?}"
            )))
    }
}

/// Selective-state-space scan, **Mamba1 variant** — per-channel `dt`
/// + state shape `[d_inner, d_state]`. Used by Jamba, the original
/// Mamba 1 architecture, and any model that doesn't multi-head the
/// SSM state. Distinct from `selective_scan_step` (Mamba2) which has
/// per-head `dt` + state `[nheads, headdim, d_state]`.
///
/// Math (per channel d in 0..d_inner, per state s in 0..d_state):
///
/// ```text
///   dt = softplus(dt_raw + dt_bias)        // per-channel
///   A  = -exp(A_log)                       // per-(d, s)
///   for s in 0..d_state:
///       a_bar = exp(dt[d] * A[d, s])
///       state[d, s] = a_bar * state[d, s] + dt[d] * B[s] * x[d]
///       y[d] += C[s] * state[d, s]
///   y[d] += D[d] * x[d]
/// ```
///
/// Used by Jamba — see efficient-genai's `jamba.rs::forward_mamba`
/// for the reference implementation. The dt has typically already
/// been bias-added by a prior Apply node when this leaf runs (Jamba's
/// dt comes from `dt_proj(dt_layernorm(dt_raw)) + dt_proj.bias`); we
/// still accept `dt_bias` as a passthrough so single-step usage
/// matches Mamba1 papers exactly.
///
/// Inputs:
/// - `x` `(d_inner,)` — post-conv post-SiLU SSM input.
/// - `b` `(d_state,)` — input-dependent state matrix (per-step).
/// - `c` `(d_state,)` — input-dependent readout matrix.
/// - `dt_raw` `(d_inner,)` — pre-softplus dt (per-channel; in Jamba
///   this has already been projected from low-rank dt_normed and
///   bias-added).
/// - `dt_bias` `(d_inner,)` — additional bias to add before softplus.
///   Pass zeros if dt was already bias-added upstream.
/// - `a_log` `(d_inner * d_state,)` — log of negative state decay,
///   row-major over `[d_inner, d_state]`.
/// - `d_param` `(d_inner,)` — skip-connection coefficient per channel.
/// - `state_in` `(d_inner * d_state,)` — recurrent state at t-1.
///
/// Returns `(y, state_out)`.
pub fn selective_scan_mamba1_step(
    x: &[f32],
    b: &[f32],
    c: &[f32],
    dt_raw: &[f32],
    dt_bias: &[f32],
    a_log: &[f32],
    d_param: &[f32],
    state_in: &[f32],
    d_inner: usize,
    d_state: usize,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(x.len(), d_inner);
    debug_assert_eq!(b.len(), d_state);
    debug_assert_eq!(c.len(), d_state);
    debug_assert_eq!(dt_raw.len(), d_inner);
    debug_assert_eq!(dt_bias.len(), d_inner);
    debug_assert_eq!(a_log.len(), d_inner * d_state);
    debug_assert_eq!(d_param.len(), d_inner);
    debug_assert_eq!(state_in.len(), d_inner * d_state);

    // Per-channel dt = softplus(dt_raw + dt_bias).
    let mut dt = vec![0.0f32; d_inner];
    for d in 0..d_inner {
        dt[d] = softplus(dt_raw[d] + dt_bias[d]);
    }

    let mut y = vec![0.0f32; d_inner];
    let mut state_out = state_in.to_vec();

    for d in 0..d_inner {
        let x_val = x[d];
        let dt_d = dt[d];
        let mut y_val = 0.0f32;
        for s in 0..d_state {
            // GGUF stores `ssm_a` as already-transformed `A = -exp(A_log)`
            // (llama.cpp converter does `data = -data.exp()` on .A_log
            // tensors). Use the stored value directly — see
            // `project_gguf_ssm_a_convention.md`.
            let a_val = a_log[d * d_state + s];
            let a_bar = (dt_d * a_val).exp();
            let b_val = b[s];
            let c_val = c[s];

            let idx = d * d_state + s;
            let new_h = a_bar * state_out[idx] + dt_d * b_val * x_val;
            state_out[idx] = new_h;
            y_val += c_val * new_h;
        }
        y[d] = y_val + d_param[d] * x_val;
    }

    (y, state_out)
}

/// `OpDispatch` leaf claiming `Custom("ssm_scan_mamba1")`. Same args
/// as `selective_scan_mamba1_step`, plus two `i32` scalar literals
/// for `d_inner` and `d_state`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SsmScanMamba1Leaf;

impl SsmScanMamba1Leaf {
    pub const fn new() -> Self {
        Self
    }
}

impl OpDispatch for SsmScanMamba1Leaf {
    fn handles(&self) -> &[&'static str] {
        &["ssm_scan_mamba1"]
    }

    fn execute(&self, _node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "ssm_scan_mamba1" {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_mamba1 leaf does not handle `{name}`"
            )));
        }
        if args.len() != 10 {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_mamba1: expected 10 args (x, B, C, dt_raw, dt_bias, A_log, D, state_in, d_inner, d_state), got {}",
                args.len()
            )));
        }

        let x = arg_tensor(&args[0], "x")?;
        let b = arg_tensor(&args[1], "B")?;
        let c = arg_tensor(&args[2], "C")?;
        let dt_raw = arg_tensor(&args[3], "dt_raw")?;
        let dt_bias = arg_tensor(&args[4], "dt_bias")?;
        let a_log = arg_tensor(&args[5], "A_log")?;
        let d_param = arg_tensor(&args[6], "D")?;
        let state_in = arg_tensor(&args[7], "state_in")?;
        let d_inner = arg_i32(&args[8], "d_inner")?;
        let d_state = arg_i32(&args[9], "d_state")?;

        for (label, t) in [
            ("x", x), ("B", b), ("C", c),
            ("dt_raw", dt_raw), ("dt_bias", dt_bias),
            ("A_log", a_log), ("D", d_param),
            ("state_in", state_in),
        ] {
            if t.dtype != DType::F32 {
                return Err(InterpretError::Unsupported(format!(
                    "ssm_scan_mamba1: f32 only ({label} is {})",
                    t.dtype.name()
                )));
            }
        }
        if d_inner <= 0 || d_state <= 0 {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_mamba1: d_inner and d_state must be positive (got {d_inner}, {d_state})"
            )));
        }
        let d_inner = d_inner as usize;
        let d_state = d_state as usize;

        if x.numel() != d_inner {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_mamba1: x has {} elements, expected d_inner = {d_inner}",
                x.numel()
            )));
        }
        if b.numel() != d_state || c.numel() != d_state {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_mamba1: B ({}) and C ({}) must have d_state ({d_state}) elements each",
                b.numel(), c.numel()
            )));
        }
        if dt_raw.numel() != d_inner || dt_bias.numel() != d_inner {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_mamba1: dt_raw ({}) and dt_bias ({}) must have d_inner ({d_inner}) elements each",
                dt_raw.numel(), dt_bias.numel()
            )));
        }
        if a_log.numel() != d_inner * d_state {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_mamba1: A_log has {} elements, expected d_inner*d_state = {}",
                a_log.numel(), d_inner * d_state
            )));
        }
        if d_param.numel() != d_inner {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_mamba1: D has {} elements, expected d_inner = {d_inner}",
                d_param.numel()
            )));
        }
        let state_size = d_inner * d_state;
        if state_in.numel() != state_size {
            return Err(InterpretError::Unsupported(format!(
                "ssm_scan_mamba1: state_in has {} elements, expected d_inner*d_state = {state_size}",
                state_in.numel()
            )));
        }

        let x_v = bytes_as_f32(&x.data, "x")?;
        let b_v = bytes_as_f32(&b.data, "B")?;
        let c_v = bytes_as_f32(&c.data, "C")?;
        let dt_raw_v = bytes_as_f32(&dt_raw.data, "dt_raw")?;
        let dt_bias_v = bytes_as_f32(&dt_bias.data, "dt_bias")?;
        let a_log_v = bytes_as_f32(&a_log.data, "A_log")?;
        let d_v = bytes_as_f32(&d_param.data, "D")?;
        let state_v = bytes_as_f32(&state_in.data, "state_in")?;

        trace!(d_inner, d_state, "ssm_scan_mamba1: dispatching scalar reference");

        let (y, state_out) = selective_scan_mamba1_step(
            x_v, b_v, c_v, dt_raw_v, dt_bias_v, a_log_v, d_v, state_v,
            d_inner, d_state,
        );

        let total = d_inner + state_size;
        let mut packed = Vec::with_capacity(total);
        packed.extend_from_slice(&y);
        packed.extend_from_slice(&state_out);
        let bytes: &[u8] = bytemuck::cast_slice(&packed);
        HostTensor::new(vec![total], DType::F32, bytes.to_vec())
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!(
                "ssm_scan_mamba1: build output tensor: {e:?}"
            )))
    }
}

#[inline]
fn softplus(z: f32) -> f32 {
    // Stable: log1p(exp(z)) for z < ~20, else ~z.
    if z > 20.0 {
        z
    } else {
        (1.0 + z.exp()).ln()
    }
}

fn arg_tensor<'a>(v: &'a Value, label: &str) -> Result<&'a HostTensor> {
    match v {
        Value::Tensor(t) => Ok(t),
        other => Err(InterpretError::Unsupported(format!(
            "ssm_scan: {label} expected tensor, got {other:?}"
        ))),
    }
}

fn arg_i32(v: &Value, label: &str) -> Result<i32> {
    match v {
        Value::Scalar(ScalarValue::I32(x)) => Ok(*x),
        Value::Scalar(ScalarValue::I64(x)) => Ok(*x as i32),
        other => Err(InterpretError::Unsupported(format!(
            "ssm_scan: {label} expected i32 scalar, got {other:?}"
        ))),
    }
}

fn bytes_as_f32<'a>(bytes: &'a [u8], label: &str) -> Result<&'a [f32]> {
    bytemuck::try_cast_slice(bytes).map_err(|e| {
        InterpretError::Unsupported(format!(
            "ssm_scan: {label} bytes not f32-aligned ({e})"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation copied verbatim from
    /// efficient-genai/src/inference/architecture/mamba2.rs lines 488-534.
    /// Used as the numerical-equivalence ground truth.
    fn efficient_genai_reference(
        x: &[f32],
        b_conv: &[f32],
        c_conv: &[f32],
        dt_raw: &[f32],
        dt_bias: &[f32],
        a_log: &[f32],
        d_param: &[f32],
        ssm_state: &mut [f32],
        nheads: usize,
        headdim: usize,
        d_state: usize,
        ngroups: usize,
    ) -> Vec<f32> {
        let d_inner = nheads * headdim;
        let nheads_per_group = nheads / ngroups;

        let mut dt = vec![0.0f32; nheads];
        for h in 0..nheads {
            dt[h] = {
                let z = dt_raw[h] + dt_bias[h];
                if z > 20.0 { z } else { (1.0 + z.exp()).ln() }
            };
        }

        let mut a = vec![0.0f32; nheads];
        for h in 0..nheads {
            // Match production: GGUF `ssm_a` stores `A` directly, not A_log.
            a[h] = a_log[h];
        }

        let mut y = vec![0.0f32; d_inner];

        for h in 0..nheads {
            let group = h / nheads_per_group;
            let a_bar = (dt[h] * a[h]).exp();
            let dt_h = dt[h];

            for d in 0..headdim {
                let x_val = x[h * headdim + d];
                let mut y_val = 0.0f32;

                for s in 0..d_state {
                    let b_val = b_conv[group * d_state + s];
                    let c_val = c_conv[group * d_state + s];

                    let state_idx = (h * headdim + d) * d_state + s;
                    ssm_state[state_idx] =
                        a_bar * ssm_state[state_idx] + dt_h * b_val * x_val;

                    y_val += c_val * ssm_state[state_idx];
                }

                y[h * headdim + d] = y_val + d_param[h] * x_val;
            }
        }

        y
    }

    fn make_inputs(
        nheads: usize,
        headdim: usize,
        d_state: usize,
        ngroups: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let d_inner = nheads * headdim;
        let state_size = nheads * headdim * d_state;

        let x: Vec<f32> = (0..d_inner).map(|i| (i as f32 * 0.01) - 0.5).collect();
        let b: Vec<f32> = (0..ngroups * d_state).map(|i| (i as f32 * 0.013) - 0.3).collect();
        let c: Vec<f32> = (0..ngroups * d_state).map(|i| (i as f32 * 0.017) + 0.1).collect();
        let dt_raw: Vec<f32> = (0..nheads).map(|i| (i as f32 * 0.05) - 0.2).collect();
        let dt_bias: Vec<f32> = (0..nheads).map(|i| (i as f32 * 0.02) + 0.1).collect();
        let a_log: Vec<f32> = (0..nheads).map(|i| -((i + 1) as f32) * 0.1).collect();
        let d_param: Vec<f32> = (0..nheads).map(|i| (i as f32) * 0.005).collect();
        let state: Vec<f32> = (0..state_size).map(|i| ((i as f32 * 0.001) - 0.5).sin()).collect();

        (x, b, c, dt_raw, dt_bias, a_log, d_param, state)
    }

    #[test]
    fn matches_efficient_genai_reference_small() {
        let (nheads, headdim, d_state, ngroups) = (4, 8, 16, 1);
        let (x, b, c, dt_raw, dt_bias, a_log, d_param, state0) =
            make_inputs(nheads, headdim, d_state, ngroups);

        let mut ref_state = state0.clone();
        let want_y = efficient_genai_reference(
            &x, &b, &c, &dt_raw, &dt_bias, &a_log, &d_param,
            &mut ref_state, nheads, headdim, d_state, ngroups,
        );

        let (got_y, got_state) = selective_scan_step(
            &x, &b, &c, &dt_raw, &dt_bias, &a_log, &d_param, &state0,
            nheads, headdim, d_state, ngroups,
        );

        assert_eq!(want_y.len(), got_y.len(), "y len mismatch");
        for (i, (a, b)) in want_y.iter().zip(got_y.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "y[{i}] diverges: want {a}, got {b}"
            );
        }
        assert_eq!(ref_state.len(), got_state.len());
        for (i, (a, b)) in ref_state.iter().zip(got_state.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "state[{i}] diverges: want {a}, got {b}"
            );
        }
    }

    #[test]
    fn matches_reference_groups_gt_one() {
        // ngroups > 1 — exercise the per-group B/C indexing.
        let (nheads, headdim, d_state, ngroups) = (8, 4, 8, 2);
        let (x, b, c, dt_raw, dt_bias, a_log, d_param, state0) =
            make_inputs(nheads, headdim, d_state, ngroups);

        let mut ref_state = state0.clone();
        let want_y = efficient_genai_reference(
            &x, &b, &c, &dt_raw, &dt_bias, &a_log, &d_param,
            &mut ref_state, nheads, headdim, d_state, ngroups,
        );

        let (got_y, got_state) = selective_scan_step(
            &x, &b, &c, &dt_raw, &dt_bias, &a_log, &d_param, &state0,
            nheads, headdim, d_state, ngroups,
        );

        for (i, (a, b)) in want_y.iter().zip(got_y.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "y[{i}]: want {a}, got {b}");
        }
        for (i, (a, b)) in ref_state.iter().zip(got_state.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "state[{i}]: want {a}, got {b}");
        }
    }

    #[test]
    fn mamba2_2_7b_realistic_shapes() {
        // Match the actual mamba2-2.7b checkpoint shapes so we know
        // the math runs on the model the user has on disk.
        // mamba2-2.7b: d_model=2560, expand=2 → d_inner=5120,
        // headdim=64 → nheads=80, d_state=128, ngroups=1.
        let (nheads, headdim, d_state, ngroups) = (80, 64, 128, 1);
        let (x, b, c, dt_raw, dt_bias, a_log, d_param, state0) =
            make_inputs(nheads, headdim, d_state, ngroups);

        let mut ref_state = state0.clone();
        let want_y = efficient_genai_reference(
            &x, &b, &c, &dt_raw, &dt_bias, &a_log, &d_param,
            &mut ref_state, nheads, headdim, d_state, ngroups,
        );

        let (got_y, got_state) = selective_scan_step(
            &x, &b, &c, &dt_raw, &dt_bias, &a_log, &d_param, &state0,
            nheads, headdim, d_state, ngroups,
        );

        let mut max_y_err = 0.0f32;
        for (a, b) in want_y.iter().zip(got_y.iter()) {
            max_y_err = max_y_err.max((a - b).abs());
        }
        let mut max_state_err = 0.0f32;
        for (a, b) in ref_state.iter().zip(got_state.iter()) {
            max_state_err = max_state_err.max((a - b).abs());
        }
        assert!(max_y_err < 1e-5, "max_y_err {max_y_err}");
        assert!(max_state_err < 1e-5, "max_state_err {max_state_err}");
    }

    #[test]
    fn op_dispatch_matches_free_function() {
        use rce_vm_backend_cpu::{HostTensor, ScalarValue, Value};

        let (nheads, headdim, d_state, ngroups) = (4, 8, 16, 1);
        let (x, b, c, dt_raw, dt_bias, a_log, d_param, state0) =
            make_inputs(nheads, headdim, d_state, ngroups);
        let d_inner = nheads * headdim;
        let state_size = nheads * headdim * d_state;

        let (want_y, want_state) = selective_scan_step(
            &x, &b, &c, &dt_raw, &dt_bias, &a_log, &d_param, &state0,
            nheads, headdim, d_state, ngroups,
        );

        let to_tensor = |shape: Vec<usize>, v: &[f32]| -> Value {
            Value::Tensor(HostTensor::from_f32(shape, v).unwrap())
        };

        let leaf = SsmScanLeaf::new();
        let args = vec![
            to_tensor(vec![d_inner], &x),
            to_tensor(vec![ngroups * d_state], &b),
            to_tensor(vec![ngroups * d_state], &c),
            to_tensor(vec![nheads], &dt_raw),
            to_tensor(vec![nheads], &dt_bias),
            to_tensor(vec![nheads], &a_log),
            to_tensor(vec![nheads], &d_param),
            to_tensor(vec![state_size], &state0),
            Value::Scalar(ScalarValue::I32(headdim as i32)),
            Value::Scalar(ScalarValue::I32(d_state as i32)),
        ];

        let result = leaf.execute(0, "ssm_scan", &args).expect("execute");
        let out = match result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor, got {other:?}"),
        };
        let packed = out.as_f32().expect("as_f32");

        assert_eq!(packed.len(), d_inner + state_size, "packed len");
        let got_y = &packed[..d_inner];
        let got_state = &packed[d_inner..];

        for (i, (a, b)) in want_y.iter().zip(got_y.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "y[{i}]: want {a}, got {b}");
        }
        for (i, (a, b)) in want_state.iter().zip(got_state.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "state[{i}]: want {a}, got {b}");
        }
    }

    #[test]
    fn mamba1_step_matches_inline_reference() {
        // Reference computation: per-channel dt + per-(d, s) A.
        // Verifies the math primitive at small shapes.
        let d_inner = 8usize;
        let d_state = 4usize;
        let state_size = d_inner * d_state;

        let x: Vec<f32> = (0..d_inner).map(|i| (i as f32 * 0.1) - 0.3).collect();
        let b: Vec<f32> = (0..d_state).map(|i| (i as f32 * 0.05) - 0.1).collect();
        let c: Vec<f32> = (0..d_state).map(|i| (i as f32 * 0.07) + 0.2).collect();
        let dt_raw: Vec<f32> = (0..d_inner).map(|i| (i as f32 * 0.04) - 0.5).collect();
        let dt_bias: Vec<f32> = (0..d_inner).map(|i| (i as f32 * 0.02) + 0.1).collect();
        let a_log: Vec<f32> = (0..state_size).map(|i| -((i + 1) as f32) * 0.05).collect();
        let d_param: Vec<f32> = (0..d_inner).map(|i| (i as f32) * 0.005).collect();
        let state0: Vec<f32> = (0..state_size).map(|i| ((i as f32 * 0.011) - 0.4).sin()).collect();

        // Inline reference (matches efficient-genai's jamba.rs forward_mamba math).
        let mut state_ref = state0.clone();
        let mut y_ref = vec![0.0f32; d_inner];
        for d in 0..d_inner {
            let dt_d = {
                let z = dt_raw[d] + dt_bias[d];
                if z > 20.0 { z } else { (1.0 + z.exp()).ln() }
            };
            for s in 0..d_state {
                // Match production: GGUF stores A directly, not A_log.
                let a = a_log[d * d_state + s];
                let a_bar = (dt_d * a).exp();
                let idx = d * d_state + s;
                state_ref[idx] = a_bar * state_ref[idx] + dt_d * b[s] * x[d];
                y_ref[d] += c[s] * state_ref[idx];
            }
            y_ref[d] += d_param[d] * x[d];
        }

        let (y, state_out) = selective_scan_mamba1_step(
            &x, &b, &c, &dt_raw, &dt_bias, &a_log, &d_param, &state0,
            d_inner, d_state,
        );

        for (i, (a, b)) in y_ref.iter().zip(y.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "y[{i}]: want {a}, got {b}");
        }
        for (i, (a, b)) in state_ref.iter().zip(state_out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "state[{i}]: want {a}, got {b}");
        }
    }

    #[test]
    fn mamba1_jamba_realistic_shapes() {
        // Jamba-Reasoning-3B SSM shape: d_model=2560, d_inner=2*d_model=5120,
        // d_state=16. State is [d_inner, d_state] = 81920 elements.
        let d_inner = 5120usize;
        let d_state = 16usize;
        let state_size = d_inner * d_state;

        let x: Vec<f32> = (0..d_inner).map(|i| (i as f32 * 0.001).sin() * 0.5).collect();
        let b: Vec<f32> = (0..d_state).map(|i| (i as f32 * 0.01) - 0.05).collect();
        let c: Vec<f32> = (0..d_state).map(|i| (i as f32 * 0.013) + 0.1).collect();
        let dt_raw: Vec<f32> = (0..d_inner).map(|i| ((i as f32 * 0.0007) - 0.3).cos() * 0.4).collect();
        let dt_bias: Vec<f32> = (0..d_inner).map(|i| (i as f32 * 0.0003) + 0.05).collect();
        let a_log: Vec<f32> = (0..state_size).map(|i| -((i % 16 + 1) as f32) * 0.1 - 0.5).collect();
        let d_param: Vec<f32> = (0..d_inner).map(|i| (i as f32 * 0.0001) - 0.05).collect();
        let state0: Vec<f32> = (0..state_size).map(|i| ((i as f32 * 0.0001) - 0.5).sin() * 0.3).collect();

        let (y, state_out) = selective_scan_mamba1_step(
            &x, &b, &c, &dt_raw, &dt_bias, &a_log, &d_param, &state0,
            d_inner, d_state,
        );

        // Sanity: outputs should be finite and have reasonable scale.
        assert_eq!(y.len(), d_inner);
        assert_eq!(state_out.len(), state_size);
        for (i, v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}] = {v} is not finite");
        }
        for (i, v) in state_out.iter().enumerate() {
            assert!(v.is_finite(), "state_out[{i}] = {v} is not finite");
        }
        let max_y = y.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("mamba1 jamba shape (d_inner={d_inner}, d_state={d_state}): |y|_∞ = {max_y:.3e}");
    }

    #[test]
    fn mamba1_op_dispatch_matches_free_function() {
        use rce_vm_backend_cpu::{HostTensor, ScalarValue, Value};

        let d_inner = 16usize;
        let d_state = 8usize;
        let state_size = d_inner * d_state;
        let x: Vec<f32> = (0..d_inner).map(|i| (i as f32 * 0.1)).collect();
        let b: Vec<f32> = (0..d_state).map(|i| (i as f32 * 0.05)).collect();
        let c: Vec<f32> = (0..d_state).map(|i| (i as f32 * 0.07)).collect();
        let dt_raw: Vec<f32> = (0..d_inner).map(|i| (i as f32 * 0.04)).collect();
        let dt_bias: Vec<f32> = (0..d_inner).map(|i| (i as f32 * 0.02)).collect();
        let a_log: Vec<f32> = (0..state_size).map(|i| -((i + 1) as f32) * 0.05).collect();
        let d_param: Vec<f32> = (0..d_inner).map(|i| (i as f32) * 0.005).collect();
        let state0: Vec<f32> = (0..state_size).map(|i| (i as f32 * 0.001)).collect();

        let (want_y, want_state) = selective_scan_mamba1_step(
            &x, &b, &c, &dt_raw, &dt_bias, &a_log, &d_param, &state0,
            d_inner, d_state,
        );

        let to_tensor = |shape: Vec<usize>, v: &[f32]| -> Value {
            Value::Tensor(HostTensor::from_f32(shape, v).unwrap())
        };
        let leaf = SsmScanMamba1Leaf::new();
        let args = vec![
            to_tensor(vec![d_inner], &x),
            to_tensor(vec![d_state], &b),
            to_tensor(vec![d_state], &c),
            to_tensor(vec![d_inner], &dt_raw),
            to_tensor(vec![d_inner], &dt_bias),
            to_tensor(vec![state_size], &a_log),
            to_tensor(vec![d_inner], &d_param),
            to_tensor(vec![state_size], &state0),
            Value::Scalar(ScalarValue::I32(d_inner as i32)),
            Value::Scalar(ScalarValue::I32(d_state as i32)),
        ];

        let result = leaf.execute(0, "ssm_scan_mamba1", &args).expect("execute");
        let out = match result { Value::Tensor(t) => t, _ => panic!() };
        let packed = out.as_f32().unwrap();
        assert_eq!(packed.len(), d_inner + state_size);
        let got_y = &packed[..d_inner];
        let got_state = &packed[d_inner..];

        for (i, (a, b)) in want_y.iter().zip(got_y.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "y[{i}]: want {a}, got {b}");
        }
        for (i, (a, b)) in want_state.iter().zip(got_state.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "state[{i}]: want {a}, got {b}");
        }
    }

    #[test]
    fn rejects_dtype_mismatch() {
        use rce_vm_backend_cpu::{DType, HostTensor, ScalarValue, Value};

        let leaf = SsmScanLeaf::new();
        // Build a non-f32 first arg.
        let bad_x = HostTensor::new(vec![32], DType::I32, vec![0u8; 32 * 4]).unwrap();
        let zero_f32 = |n| Value::Tensor(HostTensor::from_f32(vec![n], &vec![0.0; n]).unwrap());
        let args = vec![
            Value::Tensor(bad_x),
            zero_f32(16), zero_f32(16),
            zero_f32(4), zero_f32(4), zero_f32(4), zero_f32(4),
            zero_f32(4 * 8 * 16),
            Value::Scalar(ScalarValue::I32(8)),
            Value::Scalar(ScalarValue::I32(16)),
        ];
        let result = leaf.execute(0, "ssm_scan", &args);
        match result {
            Err(InterpretError::Unsupported(msg)) => {
                assert!(msg.contains("f32"), "expected f32 complaint, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    /// `SsmScanF64Leaf` (the VM op a graph-routed mamba2 mixer will use)
    /// must be **byte-identical** to calling `selective_scan_step_f64`
    /// directly — same precondition for "real model through the VM ==
    /// hand path" that `qmm` already satisfies for matmul.
    #[test]
    fn ssm_scan_f64_leaf_matches_free_fn() {
        use rce_vm_backend_cpu::{HostTensor, ScalarValue, Value};

        let (nheads, headdim, d_state, ngroups) = (6, 8, 16, 2);
        let (x, b, c, dt_raw, dt_bias, a_log, d_param, state) =
            make_inputs(nheads, headdim, d_state, ngroups);
        let d_inner = nheads * headdim;
        let state_size = nheads * headdim * d_state;

        // Reference: call the f64 free function directly.
        let (y_ref, st_ref) = selective_scan_step_f64(
            &x, &b, &c, &dt_raw, &dt_bias, &a_log, &d_param, &state,
            nheads, headdim, d_state, ngroups,
        );

        // Through the leaf (the OpDispatch the VM invokes).
        let to_t = |v: &[f32]| Value::Tensor(HostTensor::from_f32(vec![v.len()], v).unwrap());
        let args = vec![
            to_t(&x), to_t(&b), to_t(&c), to_t(&dt_raw), to_t(&dt_bias),
            to_t(&a_log), to_t(&d_param), to_t(&state),
            Value::Scalar(ScalarValue::I32(headdim as i32)),
            Value::Scalar(ScalarValue::I32(d_state as i32)),
        ];
        let out = match SsmScanF64Leaf::new()
            .execute(0, "ssm_scan_f64", &args)
            .expect("ssm_scan_f64 execute")
        {
            Value::Tensor(t) => t.as_f32().expect("f32 out"),
            other => panic!("non-tensor result {other:?}"),
        };
        assert_eq!(out.len(), d_inner + state_size);
        let (y_leaf, st_leaf) = out.split_at(d_inner);
        for (i, (a, b)) in y_leaf.iter().zip(y_ref.iter()).enumerate() {
            assert_eq!(a, b, "y[{i}]: leaf {a} ≠ free fn {b} (must be bit-identical)");
        }
        for (i, (a, b)) in st_leaf.iter().zip(st_ref.iter()).enumerate() {
            assert_eq!(a, b, "state[{i}]: leaf {a} ≠ free fn {b} (must be bit-identical)");
        }
    }

    /// Brick 2 contract: the T-token batched scan must be **byte-identical**
    /// to calling `selective_scan_step_f64` `T` times in a row with the
    /// state threaded forward — and the leaf the VM invokes must be
    /// byte-identical to the free function. Same precondition as the
    /// single-step case (and `qmm`): "real model through the VM == hand
    /// path". Per-token inputs are perturbed by `t` so the test actually
    /// exercises distinct tokens + genuine state threading, not a degenerate
    /// all-identical-tokens window.
    #[test]
    fn ssm_scan_f64_batched_bit_exact_vs_t_single_steps_and_leaf() {
        use rce_vm_backend_cpu::{HostTensor, ScalarValue, Value};

        let (nheads, headdim, d_state, ngroups) = (6, 8, 16, 2);
        let t_rows = 5usize;
        let d_inner = nheads * headdim;
        let bc_per = ngroups * d_state;
        let state_size = nheads * headdim * d_state;

        // Base single-token inputs, then per-token perturbation so each
        // of the T tokens is distinct (state threading is observable).
        let (x0, b0, c0, dt0, dt_bias, a_log, d_param, state0) =
            make_inputs(nheads, headdim, d_state, ngroups);
        let perturb = |base: &[f32], t: usize, k: f32| -> Vec<f32> {
            base.iter()
                .map(|&v| v + (t as f32) * k)
                .collect()
        };

        let mut x_all = Vec::with_capacity(t_rows * d_inner);
        let mut b_all = Vec::with_capacity(t_rows * bc_per);
        let mut c_all = Vec::with_capacity(t_rows * bc_per);
        let mut dt_all = Vec::with_capacity(t_rows * nheads);
        for t in 0..t_rows {
            x_all.extend(perturb(&x0, t, 0.003));
            b_all.extend(perturb(&b0, t, 0.002));
            c_all.extend(perturb(&c0, t, 0.0015));
            dt_all.extend(perturb(&dt0, t, 0.004));
        }

        // Reference: T single steps, state threaded by hand.
        let mut ref_state = state0.clone();
        let mut ref_y = Vec::with_capacity(t_rows * d_inner);
        for t in 0..t_rows {
            let (yt, st) = selective_scan_step_f64(
                &x_all[t * d_inner..(t + 1) * d_inner],
                &b_all[t * bc_per..(t + 1) * bc_per],
                &c_all[t * bc_per..(t + 1) * bc_per],
                &dt_all[t * nheads..(t + 1) * nheads],
                &dt_bias, &a_log, &d_param, &ref_state,
                nheads, headdim, d_state, ngroups,
            );
            ref_y.extend_from_slice(&yt);
            ref_state = st;
        }

        // Batched free function.
        let (y_bat, st_bat) = selective_scan_step_f64_batched(
            &x_all, &b_all, &c_all, &dt_all, &dt_bias, &a_log, &d_param,
            &state0, t_rows, nheads, headdim, d_state, ngroups,
        );
        assert_eq!(y_bat.len(), t_rows * d_inner);
        assert_eq!(st_bat.len(), state_size);
        for (i, (a, b)) in y_bat.iter().zip(ref_y.iter()).enumerate() {
            assert_eq!(a, b, "y[{i}]: batched {a} ≠ T×single-step {b} (must be bit-identical)");
        }
        for (i, (a, b)) in st_bat.iter().zip(ref_state.iter()).enumerate() {
            assert_eq!(a, b, "state[{i}]: batched {a} ≠ T×single-step {b} (must be bit-identical)");
        }

        // Through the leaf (the OpDispatch the VM invokes).
        let to_t = |v: &[f32]| Value::Tensor(HostTensor::from_f32(vec![v.len()], v).unwrap());
        let args = vec![
            to_t(&x_all), to_t(&b_all), to_t(&c_all), to_t(&dt_all),
            to_t(&dt_bias), to_t(&a_log), to_t(&d_param), to_t(&state0),
            Value::Scalar(ScalarValue::I32(headdim as i32)),
            Value::Scalar(ScalarValue::I32(d_state as i32)),
            Value::Scalar(ScalarValue::I32(t_rows as i32)),
        ];
        let out = match SsmScanF64BatchedLeaf::new()
            .execute(0, "ssm_scan_f64_batched", &args)
            .expect("ssm_scan_f64_batched execute")
        {
            Value::Tensor(t) => t.as_f32().expect("f32 out"),
            other => panic!("non-tensor result {other:?}"),
        };
        assert_eq!(out.len(), t_rows * d_inner + state_size);
        let (y_leaf, st_leaf) = out.split_at(t_rows * d_inner);
        for (i, (a, b)) in y_leaf.iter().zip(y_bat.iter()).enumerate() {
            assert_eq!(a, b, "y[{i}]: leaf {a} ≠ free fn {b} (must be bit-identical)");
        }
        for (i, (a, b)) in st_leaf.iter().zip(st_bat.iter()).enumerate() {
            assert_eq!(a, b, "state[{i}]: leaf {a} ≠ free fn {b} (must be bit-identical)");
        }
    }
}
