//! `rce-op-attention` — `OpDispatch` leaf that handles
//! `Custom("attention")` as a single fused kernel: scaled
//! dot-product attention computed in one BLAS sgemm + one softmax +
//! one BLAS sgemm, with no intermediate `HostTensor` allocations
//! between sub-ops.
//!
//! ## Why fused
//!
//! Composing attention as 5 separate Apply nodes (transpose, matmul,
//! scale, softmax, matmul) through the IR walks the OpDispatch chain
//! 5 times and allocates 5 intermediate `Value::Tensor` outputs.
//! For decode-mode shapes (Q is 1×head_dim) the actual BLAS work is
//! a few microseconds and the per-Apply machinery dominates. Fusing
//! the entire computation into one Custom op collapses 5 dispatches
//! into 1.
//!
//! Same architectural shape as MLIR's `linalg.attention`, StableHLO's
//! upcoming attention primitive, FlashAttention's CUDA kernel. The
//! IR carries the *semantics* (this is attention with these
//! parameters); the leaf carries the *implementation*. Multiple
//! leaves (cpu-fused, MPS, CUDA-FlashAttention) can implement the
//! same Custom op with different kernels, dispatched via the same
//! Unsupported(_) fall-through.
//!
//! ## Op contract
//!
//! `Custom("attention")` takes 5 args in this order:
//!
//! 1. **Q**: f32 row-major tensor of shape `(batch, head_dim)`.
//!    For decode-mode `batch == 1`; for prefill `batch == seq_len`.
//! 2. **K**: f32 row-major tensor of shape `(seq_len, head_dim)`.
//! 3. **V**: f32 row-major tensor of shape `(seq_len, head_dim)`.
//! 4. **scale**: f32 scalar (typically `1/sqrt(head_dim)`).
//! 5. **sliding_window**: i32 scalar (`0` = full attention; otherwise
//!    `min(seq_len, sliding_window)` last positions are attended).
//!
//! Output: f32 row-major tensor of shape `(batch, head_dim)`.

use rce_vm_backend_cpu::{
    DType, HostTensor, InterpretError, OpDispatch, Result, ScalarValue, Value,
};
use rce_graph::graph::OpKind;
use tracing::trace;

/// Fused attention leaf. Stateless, zero-sized, claims `Custom("attention")`.
#[derive(Debug, Default, Clone, Copy)]
pub struct AttentionLeaf;

impl AttentionLeaf {
    pub const fn new() -> Self {
        Self
    }
}

impl OpDispatch for AttentionLeaf {
    fn handles(&self) -> &[&'static str] {
        &["attention"]
    }

    fn execute(&self, node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        if name != "attention" {
            return Err(InterpretError::Unsupported(format!(
                "attention leaf does not handle `{name}`"
            )));
        }
        if args.len() != 5 {
            return Err(InterpretError::Unsupported(format!(
                "attention: expected 5 args (Q, K, V, scale, sliding_window), got {}",
                args.len()
            )));
        }

        let q = arg_tensor(&args[0], "Q")?;
        let k = arg_tensor(&args[1], "K")?;
        let v = arg_tensor(&args[2], "V")?;
        let scale = arg_f32(&args[3], "scale")?;
        let sliding = arg_i32(&args[4], "sliding_window")?;

        // Validate shapes / dtypes — fall through if any constraint
        // doesn't match this leaf's coverage.
        if q.dtype != DType::F32 || k.dtype != DType::F32 || v.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "attention: f32 only (got Q={}, K={}, V={})",
                q.dtype.name(),
                k.dtype.name(),
                v.dtype.name()
            )));
        }
        if q.shape.len() != 2 || k.shape.len() != 2 || v.shape.len() != 2 {
            return Err(InterpretError::Unsupported(format!(
                "attention: rank-2 only (got Q={:?}, K={:?}, V={:?})",
                q.shape, k.shape, v.shape
            )));
        }
        let (batch, head_dim) = (q.shape[0], q.shape[1]);
        let (seq_len, k_dim) = (k.shape[0], k.shape[1]);
        let (v_seq, v_dim) = (v.shape[0], v.shape[1]);
        if k_dim != head_dim || v_dim != head_dim {
            return Err(InterpretError::Unsupported(format!(
                "attention: head_dim mismatch (Q={head_dim}, K={k_dim}, V={v_dim})"
            )));
        }
        if seq_len != v_seq {
            return Err(InterpretError::Unsupported(format!(
                "attention: K seq_len {seq_len} != V seq_len {v_seq}"
            )));
        }
        if batch == 0 || head_dim == 0 {
            return Err(InterpretError::Unsupported(
                "attention: zero-size dim, fall through".into(),
            ));
        }

        // Sliding-window mask: trim K and V to the last `sliding`
        // positions. We do this via offset + active_len rather than
        // copying because the BLAS kernels can take row-stride args.
        let (kv_start, active_len) = if sliding > 0 && (sliding as usize) < seq_len {
            let s = sliding as usize;
            (seq_len - s, s)
        } else {
            (0, seq_len)
        };
        if active_len == 0 {
            // No keys to attend to — output zeros (matches scalar
            // attention's empty-keys path).
            let zeros = vec![0u8; batch * head_dim * 4];
            return HostTensor::new(vec![batch, head_dim], DType::F32, zeros)
                .map(Value::Tensor)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "attention: build zero tensor: {e:?}"
                )));
        }

        // Zero-copy &[f32] views of input bytes.
        let q_data: &[f32] = bytemuck::try_cast_slice(&q.data).map_err(|e| {
            InterpretError::Unsupported(format!("attention: Q not aligned ({e})"))
        })?;
        let k_data: &[f32] = bytemuck::try_cast_slice(&k.data).map_err(|e| {
            InterpretError::Unsupported(format!("attention: K not aligned ({e})"))
        })?;
        let v_data: &[f32] = bytemuck::try_cast_slice(&v.data).map_err(|e| {
            InterpretError::Unsupported(format!("attention: V not aligned ({e})"))
        })?;
        // K rows in [kv_start, seq_len) — apply the sliding-window
        // offset at the data slice level.
        let k_active = &k_data[kv_start * head_dim..(kv_start + active_len) * head_dim];
        let v_active = &v_data[kv_start * head_dim..(kv_start + active_len) * head_dim];

        // ── Step 1: scores = Q @ K^T  →  (batch, active_len) ───────
        //
        // Use cblas's TransB flag instead of pre-transposing K. Saves
        // an `active_len × head_dim × 4` byte memory pass + the
        // cache-unfriendly transpose loop. For seq=2048 head_dim=64
        // that's a half-megabyte of extra writes the BLAS
        // implementation handles internally with cache blocking.
        let mut scores_buf = vec![0.0f32; batch * active_len];
        rce_op_blas::sgemm_with_trans_b_into(
            q_data,
            k_active,
            &mut scores_buf,
            batch,        // M
            head_dim,     // K (inner dim of Q and K)
            active_len,   // N (output cols == K's outer dim, transposed)
            true,         // trans_b: K is treated as K^T
        );

        // ── Step 2: scaled softmax in-place per row ─────────────
        //
        // Numerically stable: subtract per-row max, exp, divide by sum.
        // Combines the *= scale step with the softmax to save a pass.
        for row in 0..batch {
            let row_off = row * active_len;
            let row_slice = &mut scores_buf[row_off..row_off + active_len];

            // pass 1: scaled max
            let mut max_v = f32::NEG_INFINITY;
            for &x in row_slice.iter() {
                let s = x * scale;
                if s > max_v {
                    max_v = s;
                }
            }

            // pass 2: exp(scaled - max), accumulate sum
            let mut sum = 0.0f32;
            for x in row_slice.iter_mut() {
                let s = *x * scale - max_v;
                let e = s.exp();
                *x = e;
                sum += e;
            }

            // pass 3: divide
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for x in row_slice.iter_mut() {
                    *x *= inv;
                }
            }
        }

        // ── Step 3: out = scores @ V  →  (batch, head_dim) ─────────
        let mut out_bytes: Vec<u8> = Vec::with_capacity(batch * head_dim * 4);
        unsafe {
            out_bytes.set_len(batch * head_dim * 4);
            let out_ptr = out_bytes.as_mut_ptr() as *mut f32;
            let out_slice = std::slice::from_raw_parts_mut(out_ptr, batch * head_dim);
            rce_op_blas::sgemm_into(
                &scores_buf,
                v_active,
                out_slice,
                batch,
                active_len,
                head_dim,
            );
        }

        trace!(
            target: "rce::op::attention",
            node = node_id,
            batch,
            head_dim,
            seq_len,
            active_len,
            "fused attention dispatched"
        );

        HostTensor::new(vec![batch, head_dim], DType::F32, out_bytes)
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Unsupported(format!(
                "attention: build out tensor: {e:?}"
            )))
    }

    fn execute_primitive(
        &self,
        _node_id: usize,
        _op: &OpKind,
        _args: &[Value],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "attention leaf: no primitive ops".into(),
        ))
    }
}

// ── Arg-parsing helpers ─────────────────────────────────────────────

fn arg_tensor<'a>(v: &'a Value, name: &str) -> Result<&'a HostTensor> {
    match v {
        Value::Tensor(t) => Ok(t),
        other => Err(InterpretError::Unsupported(format!(
            "attention: arg `{name}` must be a host tensor, got {}",
            other.type_name()
        ))),
    }
}

fn arg_f32(v: &Value, name: &str) -> Result<f32> {
    match v {
        Value::Scalar(ScalarValue::F32(x)) => Ok(*x),
        Value::Scalar(ScalarValue::F64(x)) => Ok(*x as f32),
        Value::Scalar(ScalarValue::I32(x)) => Ok(*x as f32),
        Value::Scalar(ScalarValue::I64(x)) => Ok(*x as f32),
        other => Err(InterpretError::Unsupported(format!(
            "attention: arg `{name}` must be f32 scalar, got {}",
            other.type_name()
        ))),
    }
}

fn arg_i32(v: &Value, name: &str) -> Result<i32> {
    match v {
        Value::Scalar(ScalarValue::I32(x)) => Ok(*x),
        Value::Scalar(ScalarValue::I64(x)) => Ok(*x as i32),
        Value::Scalar(ScalarValue::F32(x)) => Ok(*x as i32),
        Value::Scalar(ScalarValue::F64(x)) => Ok(*x as i32),
        other => Err(InterpretError::Unsupported(format!(
            "attention: arg `{name}` must be int scalar, got {}",
            other.type_name()
        ))),
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_attention(
        q: &[f32],          // (batch, head_dim)
        k: &[f32],          // (seq_len, head_dim)
        v: &[f32],          // (seq_len, head_dim)
        batch: usize,
        seq_len: usize,
        head_dim: usize,
        scale: f32,
        sliding: i32,
    ) -> Vec<f32> {
        let (start, active_len) = if sliding > 0 && (sliding as usize) < seq_len {
            (seq_len - sliding as usize, sliding as usize)
        } else {
            (0, seq_len)
        };
        let mut out = vec![0.0f32; batch * head_dim];
        for b in 0..batch {
            // scores: (active_len)
            let mut scores = vec![0.0f32; active_len];
            for i in 0..active_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[b * head_dim + d] * k[(start + i) * head_dim + d];
                }
                scores[i] = dot * scale;
            }
            // softmax (stable)
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            if sum > 0.0 {
                for s in scores.iter_mut() {
                    *s /= sum;
                }
            }
            // weighted V
            for i in 0..active_len {
                for d in 0..head_dim {
                    out[b * head_dim + d] += scores[i] * v[(start + i) * head_dim + d];
                }
            }
        }
        out
    }

    fn fixture(batch: usize, seq_len: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q: Vec<f32> = (0..batch * head_dim)
            .map(|i| (i as f32) * 0.01 + 0.1)
            .collect();
        let k: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.001).collect();
        let v: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i as f32) * 0.002 + 0.5)
            .collect();
        (q, k, v)
    }

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    fn run_leaf(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        seq_len: usize,
        head_dim: usize,
        scale: f32,
        sliding: i32,
    ) -> Vec<f32> {
        let leaf = AttentionLeaf::new();
        let qt = HostTensor::from_f32(vec![batch, head_dim], q).unwrap();
        let kt = HostTensor::from_f32(vec![seq_len, head_dim], k).unwrap();
        let vt = HostTensor::from_f32(vec![seq_len, head_dim], v).unwrap();
        let args = [
            Value::Tensor(qt),
            Value::Tensor(kt),
            Value::Tensor(vt),
            Value::Scalar(ScalarValue::F32(scale)),
            Value::Scalar(ScalarValue::I32(sliding)),
        ];
        let result = leaf
            .execute(0, "attention", &args)
            .expect("attention leaf execute");
        match result {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!("expected tensor"),
        }
    }

    #[test]
    fn fused_matches_naive_decode_mode() {
        let (batch, seq_len, head_dim) = (1, 32, 16);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let (q, k, v) = fixture(batch, seq_len, head_dim);
        let want = naive_attention(&q, &k, &v, batch, seq_len, head_dim, scale, 0);
        let got = run_leaf(&q, &k, &v, batch, seq_len, head_dim, scale, 0);
        assert!(
            approx_eq(&want, &got, 1e-4),
            "decode-mode mismatch\n  want = {want:?}\n   got = {got:?}"
        );
    }

    #[test]
    fn fused_matches_naive_prefill_mode() {
        let (batch, seq_len, head_dim) = (8, 64, 16);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let (q, k, v) = fixture(batch, seq_len, head_dim);
        let want = naive_attention(&q, &k, &v, batch, seq_len, head_dim, scale, 0);
        let got = run_leaf(&q, &k, &v, batch, seq_len, head_dim, scale, 0);
        assert!(
            approx_eq(&want, &got, 1e-3),
            "prefill-mode mismatch (batch=8)"
        );
    }

    #[test]
    fn fused_matches_naive_sliding_window() {
        let (batch, seq_len, head_dim) = (1, 64, 8);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let (q, k, v) = fixture(batch, seq_len, head_dim);
        let want = naive_attention(&q, &k, &v, batch, seq_len, head_dim, scale, 16);
        let got = run_leaf(&q, &k, &v, batch, seq_len, head_dim, scale, 16);
        assert!(
            approx_eq(&want, &got, 1e-4),
            "sliding-window mismatch (window=16)"
        );
    }

    #[test]
    fn fused_handles_empty_seq() {
        // When active_len is 0, the leaf returns a zero tensor — same
        // as the scalar attention's `if seq_len == 0 { return zeros }`
        // path. This keeps callers from having to special-case empty
        // KV cache state at the start of decoding.
        let head_dim = 8;
        let q: Vec<f32> = vec![1.0; head_dim];
        let leaf = AttentionLeaf::new();
        let qt = HostTensor::from_f32(vec![1, head_dim], &q).unwrap();
        let kt = HostTensor::from_f32(vec![0, head_dim], &[]).unwrap();
        let vt = HostTensor::from_f32(vec![0, head_dim], &[]).unwrap();
        let args = [
            Value::Tensor(qt),
            Value::Tensor(kt),
            Value::Tensor(vt),
            Value::Scalar(ScalarValue::F32(0.5)),
            Value::Scalar(ScalarValue::I32(0)),
        ];
        let result = leaf.execute(0, "attention", &args).unwrap();
        let out = match result {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!("expected tensor"),
        };
        assert_eq!(out, vec![0.0; head_dim]);
    }
}
