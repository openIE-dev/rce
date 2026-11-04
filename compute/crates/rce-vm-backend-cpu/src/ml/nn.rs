//! The kernels of RCE's model operations (spec 6.13, GT-4): activations,
//! softmax, the normalisations, attention, embedding lookup, the block
//! formats Q8_0 and Q4_0, and checked integer arithmetic.
//!
//! Every kernel fixes its order of operations and is single-threaded. Its
//! arithmetic is that of the element type e (binary32 for f32, binary64
//! for f64): each `+ - * /` and `sqrt` is one IEEE 754 operation of e,
//! rounded once to nearest even, with no fused multiply-add. The
//! transcendental functions are RCE's own (6.3 F9): `E(x)`, `T(x)` and
//! `R(x)` below are `math.exp`, `math.tanh` and `math.erf` of the element
//! widened exactly to binary64 ([`crate::crmath`]), rounded once to e. So
//! a result is a function of its operands' bits, and the same on every
//! machine. Where a kernel of a host adapter computes the same operation
//! (`rce-op-nn-prims`, `rce-op-attention`, `rce-op-qmm`), the order here is
//! that kernel's portable order, so the two differ only where that kernel
//! takes `exp` or `tanh` from the platform.

use super::{arg_i32, arg_tensor};
use crate::crmath;
use crate::error::{InterpretError, Result};
use crate::tensor::{DType, HostTensor};
use crate::value::{ScalarValue, Value};

/// The arithmetic of an element type.
pub trait Real:
    Copy
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    const NEG_INF: Self;
    /// a binary64 value rounded once to this type
    fn of(x: f64) -> Self;
    /// this value as binary64, exactly
    fn wide(self) -> f64;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn read(t: &HostTensor) -> Result<Vec<Self>>;
    fn write(shape: Vec<usize>, v: &[Self]) -> Result<HostTensor>;
}

impl Real for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    const NEG_INF: f32 = f32::NEG_INFINITY;
    fn of(x: f64) -> f32 {
        x as f32
    }
    fn wide(self) -> f64 {
        self as f64
    }
    fn sqrt(self) -> f32 {
        f32::sqrt(self)
    }
    fn abs(self) -> f32 {
        f32::abs(self)
    }
    fn read(t: &HostTensor) -> Result<Vec<f32>> {
        t.as_f32()
    }
    fn write(shape: Vec<usize>, v: &[f32]) -> Result<HostTensor> {
        HostTensor::from_f32(shape, v)
    }
}

impl Real for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const NEG_INF: f64 = f64::NEG_INFINITY;
    fn of(x: f64) -> f64 {
        x
    }
    fn wide(self) -> f64 {
        self
    }
    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    fn read(t: &HostTensor) -> Result<Vec<f64>> {
        t.as_f64()
    }
    fn write(shape: Vec<usize>, v: &[f64]) -> Result<HostTensor> {
        HostTensor::from_f64(shape, v)
    }
}

/// `E(x)`: math.exp of x, rounded once to e.
pub fn e_exp<T: Real>(x: T) -> T {
    T::of(crmath::exp(x.wide()))
}

/// `T(x)`: math.tanh of x, rounded once to e.
pub fn e_tanh<T: Real>(x: T) -> T {
    T::of(crmath::tanh(x.wide()))
}

/// `R(x)`: math.erf of x, rounded once to e.
pub fn e_erf<T: Real>(x: T) -> T {
    T::of(crmath::erf(x.wide()))
}

/// `S(x)`: math.sin of x, rounded once to e.
pub fn e_sin<T: Real>(x: T) -> T {
    T::of(crmath::sin(x.wide()))
}

/// `C(x)`: math.cos of x, rounded once to e.
pub fn e_cos<T: Real>(x: T) -> T {
    T::of(crmath::cos(x.wide()))
}

/// sigmoid: x >= 0: 1 / (1 + E(-x)); x < 0: E(x) / (1 + E(x)).
pub fn sigmoid<T: Real>(x: T) -> T {
    if x >= T::ZERO {
        T::ONE / (T::ONE + e_exp(-x))
    } else {
        let e = e_exp(x);
        e / (T::ONE + e)
    }
}

/// silu: x >= 0: x / (1 + E(-x)); x < 0: (x E(x)) / (1 + E(x)).
pub fn silu<T: Real>(x: T) -> T {
    if x >= T::ZERO {
        x / (T::ONE + e_exp(-x))
    } else {
        let e = e_exp(x);
        (x * e) / (T::ONE + e)
    }
}

/// The tanh form of GELU: (0.5 x)(1 + T(c1 (x + c2 ((x x) x)))) with
/// c1 = 0.7978845608028654 and c2 = 0.044715 rounded to e.
pub fn gelu_tanh<T: Real>(x: T) -> T {
    let c1 = T::of(0.7978845608028654);
    let c2 = T::of(0.044715);
    let x3 = x * x * x;
    let inner = c1 * (x + c2 * x3);
    T::of(0.5) * x * (T::ONE + e_tanh(inner))
}

/// The erf form of GELU: (x 0.5)(1 + R(x c)) with c = 0.7071067811865476
/// (the binary64 nearest 1/sqrt 2) rounded to e.
pub fn gelu_erf<T: Real>(x: T) -> T {
    let c = T::of(0.7071067811865476);
    x * T::of(0.5) * (T::ONE + e_erf(x * c))
}

fn arity(node_id: usize, kind: &str, args: &[Value], n: &[usize]) -> Result<()> {
    if n.contains(&args.len()) {
        Ok(())
    } else {
        Err(InterpretError::ArityMismatch { node_id, kind: kind.into(), expected: n[0], got: args.len() })
    }
}

fn mismatch(node_id: usize, msg: String) -> InterpretError {
    InterpretError::TypeMismatch { node_id, msg }
}

/// A scalar argument as binary64: an f32, f64 or integer literal.
fn arg_real(node_id: usize, v: &Value) -> Result<f64> {
    match v {
        Value::Scalar(ScalarValue::F32(x)) => Ok(*x as f64),
        Value::Scalar(ScalarValue::F64(x)) => Ok(*x),
        Value::Scalar(ScalarValue::I32(x)) => Ok(*x as f64),
        Value::Scalar(ScalarValue::I64(x)) => Ok(*x as f64),
        other => Err(mismatch(node_id, format!("expected a float scalar, got {}", other.type_name()))),
    }
}

macro_rules! float_dispatch {
    ($node_id:expr, $kind:expr, $t:expr, $f:ident ( $($a:expr),* )) => {
        match $t.dtype {
            DType::F32 => $f::<f32>($($a),*),
            DType::F64 => $f::<f64>($($a),*),
            other => Err(mismatch($node_id, format!("{}: dtype {} (f32 or f64 only)", $kind, other.name()))),
        }
    };
}

fn map1<T: Real>(x: &HostTensor, f: fn(T) -> T) -> Result<Value> {
    let v: Vec<T> = T::read(x)?.into_iter().map(f).collect();
    T::write(x.shape.clone(), &v).map(Value::Tensor)
}

/// An elementwise operation by name: `exp`, `tanh`, `sigmoid`, `silu`,
/// `gelu` (the tanh form) or `gelu_erf`.
pub fn unary(node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
    arity(node_id, name, args, &[1])?;
    let x = arg_tensor(node_id, &args[0])?;
    fn pick<T: Real>(name: &str) -> fn(T) -> T {
        match name {
            "exp" => e_exp::<T>,
            "tanh" => e_tanh::<T>,
            "sigmoid" => sigmoid::<T>,
            "silu" => silu::<T>,
            "gelu" => gelu_tanh::<T>,
            _ => gelu_erf::<T>,
        }
    }
    fn go<T: Real>(x: &HostTensor, name: &str) -> Result<Value> {
        map1::<T>(x, pick::<T>(name))
    }
    float_dispatch!(node_id, name, x, go(&x, name))
}

/// `softmax(x, axis)`: along `axis`, per slice in index order: m the
/// maximum (from -inf, the first of equal elements), e_i = E(x_i - m),
/// s = sum of e_i from +0.0, y_i = e_i / s.
pub fn softmax(node_id: usize, args: &[Value]) -> Result<Value> {
    arity(node_id, "softmax", args, &[2])?;
    let x = arg_tensor(node_id, &args[0])?;
    let a = arg_i32(node_id, &args[1])?;
    let rank = x.shape.len() as i32;
    let axis = if a < 0 { rank + a } else { a };
    if rank == 0 || axis < 0 || axis >= rank {
        return Err(mismatch(node_id, format!("softmax: axis {a} for rank {rank}")));
    }
    fn go<T: Real>(x: &HostTensor, axis: usize) -> Result<Value> {
        let data = T::read(x)?;
        let outer: usize = x.shape[..axis].iter().product();
        let dim = x.shape[axis];
        let inner: usize = x.shape[axis + 1..].iter().product();
        let mut out = vec![T::ZERO; data.len()];
        for o in 0..outer {
            for j in 0..inner {
                let base = o * dim * inner + j;
                let mut m = T::NEG_INF;
                for k in 0..dim {
                    let v = data[base + k * inner];
                    if v > m {
                        m = v;
                    }
                }
                let mut s = T::ZERO;
                for k in 0..dim {
                    let e = e_exp(data[base + k * inner] - m);
                    out[base + k * inner] = e;
                    s = s + e;
                }
                for k in 0..dim {
                    out[base + k * inner] = out[base + k * inner] / s;
                }
            }
        }
        T::write(x.shape.clone(), &out).map(Value::Tensor)
    }
    float_dispatch!(node_id, "softmax", x, go(&x, axis as usize))
}

/// The groups of a normalisation: `ngroups` equal runs of x's elements in
/// row-major order, and a per-element parameter of x's element count or of
/// one group's (then the same for every group).
fn groups(node_id: usize, kind: &str, x: &HostTensor, params: &[&HostTensor], ngroups: i32) -> Result<usize> {
    let total = x.numel();
    if ngroups <= 0 || total % ngroups as usize != 0 {
        return Err(mismatch(node_id, format!("{kind}: {ngroups} groups of {total} elements")));
    }
    let g = total / ngroups as usize;
    for p in params {
        if p.dtype != x.dtype || (p.numel() != total && p.numel() != g) {
            return Err(mismatch(node_id, format!("{kind}: a parameter of {} {} for groups of {g} {}", p.numel(), p.dtype.name(), x.dtype.name())));
        }
    }
    Ok(g)
}

/// `rms_norm(x, w, ngroups, eps)`: per group of n elements, in index order:
/// ss = sum of x_i x_i from +0.0; r = sqrt(ss / n + eps); inv = 1 / r;
/// y_i = (x_i inv) w_i. eps is rounded to e.
pub fn rms_norm(node_id: usize, args: &[Value]) -> Result<Value> {
    arity(node_id, "rms_norm", args, &[4])?;
    let x = arg_tensor(node_id, &args[0])?;
    let w = arg_tensor(node_id, &args[1])?;
    let ng = arg_i32(node_id, &args[2])?;
    let eps = arg_real(node_id, &args[3])?;
    let g = groups(node_id, "rms_norm", &x, &[&w], ng)?;
    fn go<T: Real>(x: &HostTensor, w: &HostTensor, g: usize, eps: f64) -> Result<Value> {
        let xs = T::read(x)?;
        let ws = T::read(w)?;
        let (eps, n) = (T::of(eps), T::of(g as f64));
        let mut out = vec![T::ZERO; xs.len()];
        for (gi, chunk) in xs.chunks(g).enumerate() {
            let mut ss = T::ZERO;
            for v in chunk {
                ss = ss + *v * *v;
            }
            let inv = T::ONE / (ss / n + eps).sqrt();
            for (i, v) in chunk.iter().enumerate() {
                let at = gi * g + i;
                out[at] = *v * inv * ws[if ws.len() == g { i } else { at }];
            }
        }
        T::write(x.shape.clone(), &out).map(Value::Tensor)
    }
    float_dispatch!(node_id, "rms_norm", x, go(&x, &w, g, eps))
}

/// `layer_norm(x, w, b, ngroups, eps)`: per group of n elements, in index
/// order: mean = (sum of x_i from +0.0) / n; var = (sum of (x_i - mean)
/// (x_i - mean) from +0.0) / n; inv = 1 / sqrt(var + eps);
/// y_i = ((x_i - mean) inv) w_i + b_i.
pub fn layer_norm(node_id: usize, args: &[Value]) -> Result<Value> {
    arity(node_id, "layer_norm", args, &[5])?;
    let x = arg_tensor(node_id, &args[0])?;
    let w = arg_tensor(node_id, &args[1])?;
    let b = arg_tensor(node_id, &args[2])?;
    let ng = arg_i32(node_id, &args[3])?;
    let eps = arg_real(node_id, &args[4])?;
    let g = groups(node_id, "layer_norm", &x, &[&w, &b], ng)?;
    fn go<T: Real>(x: &HostTensor, w: &HostTensor, b: &HostTensor, g: usize, eps: f64) -> Result<Value> {
        let xs = T::read(x)?;
        let ws = T::read(w)?;
        let bs = T::read(b)?;
        let (eps, n) = (T::of(eps), T::of(g as f64));
        let mut out = vec![T::ZERO; xs.len()];
        for (gi, chunk) in xs.chunks(g).enumerate() {
            let mut s = T::ZERO;
            for v in chunk {
                s = s + *v;
            }
            let mean = s / n;
            let mut q = T::ZERO;
            for v in chunk {
                let d = *v - mean;
                q = q + d * d;
            }
            let inv = T::ONE / (q / n + eps).sqrt();
            for (i, v) in chunk.iter().enumerate() {
                let at = gi * g + i;
                let p = if ws.len() == g { i } else { at };
                out[at] = (*v - mean) * inv * ws[p] + bs[p];
            }
        }
        T::write(x.shape.clone(), &out).map(Value::Tensor)
    }
    float_dispatch!(node_id, "layer_norm", x, go(&x, &w, &b, g, eps))
}

/// `attention(q, k, v, scale, window)` and, with a sixth operand `true`, its
/// causal form. One head: q (sq, d), k (sk, d), v (sk, dv) to (sq, dv);
/// or heads first: q (hq, sq, d), k (hk, sk, d), v (hk, sk, dv), hq a
/// multiple of hk, query head h reading key and value head h / (hq / hk).
/// The keys a query row i attends are the last `window` keys when
/// 0 < window < sk (all of them otherwise), and, causal, only those with
/// j <= i + sk - sq. For each query row, in index order:
/// a_j = sum over t of q_t k_jt from +0.0 (t increasing); m the maximum of
/// a_j scale (from -inf); p_j = E(a_j scale - m); s = sum of p_j from +0.0;
/// p_j = p_j (1 / s) when s > 0; y_t = sum over j of p_j v_jt from +0.0
/// (j increasing). This is the order of `rce-op-attention` on the portable
/// matrix product of `rce-op-blas`.
pub fn attention(node_id: usize, args: &[Value]) -> Result<Value> {
    arity(node_id, "attention", args, &[5, 6])?;
    let q = arg_tensor(node_id, &args[0])?;
    let k = arg_tensor(node_id, &args[1])?;
    let v = arg_tensor(node_id, &args[2])?;
    let scale = arg_real(node_id, &args[3])?;
    let window = arg_i32(node_id, &args[4])?;
    let causal = match args.get(5) {
        None => false,
        Some(Value::Scalar(ScalarValue::Bool(b))) => *b,
        Some(other) => return Err(mismatch(node_id, format!("attention: causal is a bool, got {}", other.type_name()))),
    };
    let r = q.shape.len();
    let ok = (r == 2 || r == 3) && k.shape.len() == r && v.shape.len() == r && k.dtype == q.dtype && v.dtype == q.dtype;
    let (hq, hk) = if r == 3 { (q.shape[0], k.shape[0]) } else { (1, 1) };
    let (sq, d) = (q.shape[r - 2], q.shape[r - 1]);
    let (sk, kd) = (k.shape[r - 2], k.shape[r - 1]);
    let (vs, dv) = (v.shape[r - 2], v.shape[r - 1]);
    if !ok || kd != d || vs != sk || (r == 3 && (v.shape[0] != hk || hk == 0 || hq % hk != 0)) || (causal && sq > sk) {
        return Err(mismatch(node_id, format!("attention: q {:?}, k {:?}, v {:?}", q.shape, k.shape, v.shape)));
    }
    let mut shape = q.shape.clone();
    shape[r - 1] = dv;
    #[allow(clippy::too_many_arguments)]
    fn go<T: Real>(q: &HostTensor, k: &HostTensor, v: &HostTensor, dims: [usize; 7], scale: f64, window: i32, causal: bool, shape: Vec<usize>) -> Result<Value> {
        let [hq, hk, sq, sk, d, dv, _] = dims;
        let (qs, ks, vs) = (T::read(q)?, T::read(k)?, T::read(v)?);
        let scale = T::of(scale);
        let lo = if window > 0 && (window as usize) < sk { sk - window as usize } else { 0 };
        let mut out = vec![T::ZERO; hq * sq * dv];
        let mut a = vec![T::ZERO; sk];
        for h in 0..hq {
            let kh = h / (hq / hk);
            for i in 0..sq {
                let hi = if causal { i + sk - sq + 1 } else { sk };
                let qrow = &qs[(h * sq + i) * d..(h * sq + i + 1) * d];
                for j in lo..hi {
                    let krow = &ks[(kh * sk + j) * d..(kh * sk + j + 1) * d];
                    let mut acc = T::ZERO;
                    for t in 0..d {
                        acc = acc + qrow[t] * krow[t];
                    }
                    a[j] = acc;
                }
                let mut m = T::NEG_INF;
                for aj in &a[lo..hi] {
                    let s = *aj * scale;
                    if s > m {
                        m = s;
                    }
                }
                let mut s = T::ZERO;
                for aj in a[lo..hi].iter_mut() {
                    let e = e_exp(*aj * scale - m);
                    *aj = e;
                    s = s + e;
                }
                if s > T::ZERO {
                    let inv = T::ONE / s;
                    for aj in a[lo..hi].iter_mut() {
                        *aj = *aj * inv;
                    }
                }
                let orow = (h * sq + i) * dv;
                for t in 0..dv {
                    let mut acc = T::ZERO;
                    for (j, p) in a.iter().enumerate().take(hi).skip(lo) {
                        acc = acc + *p * vs[(kh * sk + j) * dv + t];
                    }
                    out[orow + t] = acc;
                }
            }
        }
        T::write(shape, &out).map(Value::Tensor)
    }
    let dims = [hq, hk, sq, sk, d, dv, 0];
    float_dispatch!(node_id, "attention", q, go(&q, &k, &v, dims, scale, window, causal, shape))
}

/// `embedding_lookup(table, ids)`: table (n, d), ids an i32 tensor of any
/// shape, to ids' shape followed by d: the rows of the table the ids name.
/// An id outside 0..n is the trap `index ID out of range`.
pub fn embedding_lookup(node_id: usize, args: &[Value]) -> Result<Value> {
    arity(node_id, "embedding_lookup", args, &[2])?;
    let t = arg_tensor(node_id, &args[0])?;
    let ids = arg_tensor(node_id, &args[1])?;
    if t.shape.len() != 2 || ids.dtype != DType::I32 {
        return Err(mismatch(node_id, format!("embedding_lookup: table {:?} {}, ids {}", t.shape, t.dtype.name(), ids.dtype.name())));
    }
    let (n, d) = (t.shape[0], t.shape[1]);
    let row = d * t.dtype.byte_size();
    let mut data = Vec::with_capacity(ids.numel() * row);
    for id in ids.as_i32()? {
        if id < 0 || id as usize >= n {
            return Err(InterpretError::Trap(node_id, format!("index {id} out of range")));
        }
        data.extend_from_slice(&t.data[id as usize * row..(id as usize + 1) * row]);
    }
    let mut shape = ids.shape.clone();
    shape.push(d);
    HostTensor::new(shape, t.dtype, data).map(Value::Tensor)
}

/// The block formats of `matmul_q8_0`, `matmul_q4_0`, `round_q8_0` and
/// `round_q4_0`: 32 values per block along the last axis.
pub const QBLOCK: usize = 32;

/// `matmul_q8_0(x, q, d)` and `matmul_q4_0(x, q, d)`: x (m, k) f32; q (n, k)
/// the weights' quantized values (i8; for Q4_0 each in -8..7, otherwise
/// the trap `Q4_0 value V outside -8..7`); d (n, k / 32) f32, each block's
/// scale; to (m, n): y_ij = sum over blocks b, increasing from +0.0, of
/// d_jb s_b, where s_b sums (q_jt as f32) x_it from +0.0 over the block's
/// 32 positions t, in increasing t for Q8_0 and in the order 0, 16, 1, 17,
/// ..., 15, 31 for Q4_0 (low nibbles, then high, as GGUF packs them). This
/// is the order of `rce-op-qmm`'s portable kernels `mm_1d_q8_0` and
/// `mm_1d_q4_0`; every product q x and d s is exact or rounded once.
pub fn matmul_q(node_id: usize, args: &[Value], q4: bool) -> Result<Value> {
    let kind = if q4 { "matmul_q4_0" } else { "matmul_q8_0" };
    arity(node_id, kind, args, &[3])?;
    let x = arg_tensor(node_id, &args[0])?;
    let q = arg_tensor(node_id, &args[1])?;
    let d = arg_tensor(node_id, &args[2])?;
    let ok = x.dtype == DType::F32 && q.dtype == DType::I32 && d.dtype == DType::F32 && x.shape.len() == 2 && q.shape.len() == 2 && d.shape.len() == 2;
    if !ok || x.shape[1] != q.shape[1] || x.shape[1] % QBLOCK != 0 || d.shape != [q.shape[0], q.shape[1] / QBLOCK] {
        return Err(mismatch(node_id, format!("{kind}: x {:?} {}, q {:?} {}, d {:?} {}", x.shape, x.dtype.name(), q.shape, q.dtype.name(), d.shape, d.dtype.name())));
    }
    let (m, k, n) = (x.shape[0], x.shape[1], q.shape[0]);
    let (xs, qs, ds) = (x.as_f32()?, q.as_i32()?, d.as_f32()?);
    let lim = if q4 { -8..=7 } else { -128..=127 };
    if let Some(bad) = qs.iter().find(|v| !lim.contains(*v)) {
        let what = if q4 { "Q4_0 value" } else { "Q8_0 value" };
        return Err(InterpretError::Trap(node_id, format!("{what} {bad} outside {}..{}", lim.start(), lim.end())));
    }
    let nb = k / QBLOCK;
    let order: Vec<usize> = if q4 { (0..16).flat_map(|t| [t, t + 16]).collect() } else { (0..QBLOCK).collect() };
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let xr = &xs[i * k..(i + 1) * k];
        for j in 0..n {
            let qr = &qs[j * k..(j + 1) * k];
            let mut sum = 0.0f32;
            for b in 0..nb {
                let mut sub = 0.0f32;
                for t in &order {
                    let at = b * QBLOCK + t;
                    sub += qr[at] as f32 * xr[at];
                }
                sum += ds[j * nb + b] * sub;
            }
            out[i * n + j] = sum;
        }
    }
    HostTensor::from_f32(vec![m, n], &out).map(Value::Tensor)
}

/// x rounded once to the nearest binary16 value, ties to even, as an f32
/// (every binary16 value is one); 65520 or more in magnitude is an
/// infinity. binary16 has 11 significant bits and exponents -14 to 15;
/// its spacing at x is 2^(e - 10) for 2^e <= |x|, e >= -14, and 2^-24
/// below 2^-14, so the one rounding is that of x / q to an integer.
pub fn round_to_f16(x: f32) -> f32 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let e = ((x.abs().to_bits() >> 23) & 0xff) as i32 - 127;
    let q = f64::from_bits((((if e >= -14 { e - 10 } else { -24 }) + 1023) as u64) << 52);
    let r = ((x as f64) / q).round_ties_even() * q;
    if r.abs() >= 65520.0 {
        return if r > 0.0 { f32::INFINITY } else { f32::NEG_INFINITY };
    }
    r as f32
}

/// The quantization of one block of 32 values, as GGUF's reference
/// quantizers (`quantize_row_q8_0_ref`, `quantize_row_q4_0_ref`) compute it:
/// the values each position then represents, q times the binary16 scale.
/// Q8_0: a = the largest |x|; d = a / 127; id = 1 / d (0 when d = 0);
/// q = x id rounded half away from zero. Q4_0: m = the first x of largest
/// magnitude; d = m / -8; id as before; q = min(15, floor(x id + 8.5)) - 8.
/// Every operation is binary32. The scale stored is d rounded to binary16.
pub fn quantize_block(x: &[f32], q4: bool) -> (Vec<i32>, f32) {
    let mut amax = 0.0f32;
    let mut max = 0.0f32;
    for v in x {
        if amax < v.abs() {
            amax = v.abs();
            max = *v;
        }
    }
    let d = if q4 { max / -8.0 } else { amax / 127.0 };
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let q = x
        .iter()
        .map(|v| {
            if q4 {
                ((*v * id + 8.5).floor() as i32).min(15) - 8
            } else {
                (*v * id).round() as i32
            }
        })
        .collect();
    (q, round_to_f16(d))
}

/// `round_q8_0(x)` and `round_q4_0(x)`: f32 x whose last length is a
/// multiple of 32, each block of 32 along the last axis replaced by the
/// values its quantization represents ([`quantize_block`]): q times the
/// binary16 scale, exact in binary32. It lowers precision (lossy).
pub fn round_q(node_id: usize, args: &[Value], q4: bool) -> Result<Value> {
    let kind = if q4 { "round_q4_0" } else { "round_q8_0" };
    arity(node_id, kind, args, &[1])?;
    let x = arg_tensor(node_id, &args[0])?;
    if x.dtype != DType::F32 || x.shape.last().is_none_or(|l| l % QBLOCK != 0) {
        return Err(mismatch(node_id, format!("{kind}: x {:?} {}", x.shape, x.dtype.name())));
    }
    let mut out = Vec::with_capacity(x.numel());
    for b in x.as_f32()?.chunks(QBLOCK) {
        let (q, d) = quantize_block(b, q4);
        out.extend(q.iter().map(|v| *v as f32 * d));
    }
    HostTensor::from_f32(x.shape.clone(), &out).map(Value::Tensor)
}

/// The element type an operation node names in its output type
/// (`i8[2,3]`, `f32[4]`), as the graph term writes it.
pub fn declared_elem(output_type: &str) -> &str {
    output_type.split('[').next().unwrap_or(output_type)
}

/// The range of an integer element type of a graph term.
pub fn int_range(elem: &str) -> Option<(i64, i64)> {
    match elem {
        "i8" => Some((i8::MIN as i64, i8::MAX as i64)),
        "i32" => Some((i32::MIN as i64, i32::MAX as i64)),
        _ => None,
    }
}

fn int_value(node_id: usize, v: i64, elem: &str) -> Result<i32> {
    match int_range(elem) {
        Some((lo, hi)) if v >= lo && v <= hi => Ok(v as i32),
        _ => Err(InterpretError::Trap(node_id, format!("overflow: {v} is outside {elem}"))),
    }
}

/// Lane-wise `+ - * /` of integer tensors with broadcasting, the result of
/// element type `elem` (i8 or i32): each element exact, then checked
/// against `elem`'s range (spec 6.3 N2: `overflow: V is outside T`);
/// `/` truncates toward zero and a zero divisor is the trap
/// `division by zero`. An i8 tensor is held in i32 lanes.
pub fn int_binary(node_id: usize, op: &rce_graph::graph::OpKind, a: &HostTensor, b: &HostTensor, elem: &str) -> Result<HostTensor> {
    use rce_graph::graph::OpKind;
    let shape = super::broadcast::broadcast_shapes(node_id, &a.shape, &b.shape)?;
    let a = super::broadcast::broadcast_to(node_id, a, &shape)?.as_i32()?;
    let b = super::broadcast::broadcast_to(node_id, b, &shape)?.as_i32()?;
    let mut out = Vec::with_capacity(a.len());
    for (x, y) in a.iter().zip(&b) {
        let (x, y) = (*x as i64, *y as i64);
        let r = match op {
            OpKind::Add => x + y,
            OpKind::Sub => x - y,
            OpKind::Mul => x * y,
            OpKind::Div => {
                if y == 0 {
                    return Err(InterpretError::Trap(node_id, "division by zero".into()));
                }
                x / y
            }
            other => return Err(mismatch(node_id, format!("{other:?} on integer tensors"))),
        };
        out.push(int_value(node_id, r, elem)?);
    }
    HostTensor::from_i32(shape, &out)
}

/// The sum of an integer tensor in row-major order from 0, each partial
/// sum checked against `elem`'s range, as RCE's `+` checks it.
pub fn int_sum(node_id: usize, x: &HostTensor, elem: &str) -> Result<HostTensor> {
    let mut s: i64 = 0;
    for v in x.as_i32()? {
        s = int_value(node_id, s + v as i64, elem)? as i64;
    }
    HostTensor::from_i32(vec![], &[s as i32])
}

/// Whether every element of an integer tensor lies in `elem`'s range.
pub fn int_check(node_id: usize, x: &HostTensor, elem: &str) -> Result<()> {
    for v in x.as_i32()? {
        int_value(node_id, v as i64, elem)?;
    }
    Ok(())
}

/// `rope(x, pos, freqs)`: the rotary position embedding of adjacent pairs.
/// x (s, ..., d) of f32 or f64, d even; pos (s) i32, the position of each
/// index of the first axis; freqs (d / 2) of x's type. For each index r of
/// the first axis and each pair i: a = (pos_r as e) freqs_i, c = C(a),
/// n = S(a); each row of d along the last axis has its pair
/// (x_2i, x_2i+1) replaced by (x_2i c - x_2i+1 n, x_2i n + x_2i+1 c).
pub fn rope(node_id: usize, args: &[Value]) -> Result<Value> {
    arity(node_id, "rope", args, &[3])?;
    let x = arg_tensor(node_id, &args[0])?;
    let pos = arg_tensor(node_id, &args[1])?;
    let fr = arg_tensor(node_id, &args[2])?;
    let r = x.shape.len();
    let d = x.shape.last().copied().unwrap_or(0);
    if r < 2 || d % 2 != 0 || pos.dtype != DType::I32 || pos.shape != [x.shape[0]] || fr.dtype != x.dtype || fr.shape != [d / 2] {
        return Err(mismatch(node_id, format!("rope: x {:?} {}, pos {:?} {}, freqs {:?} {}", x.shape, x.dtype.name(), pos.shape, pos.dtype.name(), fr.shape, fr.dtype.name())));
    }
    fn go<T: Real>(x: &HostTensor, pos: &[i32], fr: &HostTensor) -> Result<Value> {
        let (xs, fs) = (T::read(x)?, T::read(fr)?);
        let d = *x.shape.last().unwrap_or(&2);
        let per = xs.len() / x.shape[0];
        let mut out = xs.clone();
        for (ri, p) in pos.iter().enumerate() {
            let cs: Vec<(T, T)> = fs.iter().map(|f| {
                let a = T::of(*p as f64) * *f;
                (e_cos(a), e_sin(a))
            }).collect();
            for row in (ri * per..(ri + 1) * per).step_by(d) {
                for (i, (c, n)) in cs.iter().enumerate() {
                    let (a, b) = (xs[row + 2 * i], xs[row + 2 * i + 1]);
                    out[row + 2 * i] = a * *c - b * *n;
                    out[row + 2 * i + 1] = a * *n + b * *c;
                }
            }
        }
        T::write(x.shape.clone(), &out).map(Value::Tensor)
    }
    let p = pos.as_i32()?;
    float_dispatch!(node_id, "rope", x, go(&x, &p, &fr))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t32(shape: Vec<usize>, v: &[f32]) -> Value {
        Value::Tensor(HostTensor::from_f32(shape, v).unwrap())
    }
    fn f32s(v: Value) -> Vec<f32> {
        match v {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        }
    }

    #[test]
    fn constants_round_alike() {
        assert_eq!(0.7978845608028654f64 as f32, 0.797_884_6_f32);
        assert_eq!(0.044715f64 as f32, 0.044715_f32);
        assert_eq!(0.7071067811865476f64 as f32, std::f32::consts::FRAC_1_SQRT_2);
    }

    #[test]
    fn activations_at_their_fixed_points() {
        assert_eq!(sigmoid(0.0f64), 0.5);
        assert_eq!(sigmoid(-800.0f64), 0.0);
        assert_eq!(sigmoid(800.0f64), 1.0);
        assert_eq!(silu(-0.0f32).to_bits(), (-0.0f32).to_bits());
        assert_eq!(silu(-800.0f64).to_bits(), (-0.0f64).to_bits());
        assert_eq!(gelu_erf(0.0f64), 0.0);
        assert_eq!(gelu_tanh(0.0f32), 0.0);
        assert_eq!(e_exp(1.0f32), std::f32::consts::E);
        assert_eq!(e_tanh(-0.0f64).to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn softmax_sums_in_order() {
        let y = f32s(softmax(0, &[t32(vec![2, 2], &[0.0, 0.0, 1.0, 1.0]), Value::Scalar(ScalarValue::I32(1))]).unwrap());
        assert_eq!(y, vec![0.5, 0.5, 0.5, 0.5]);
        let y = f32s(softmax(0, &[t32(vec![2, 2], &[0.0, 1.0, 0.0, 1.0]), Value::Scalar(ScalarValue::I32(0))]).unwrap());
        assert_eq!(y, vec![0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn causal_attention_sees_only_earlier_keys() {
        let q = t32(vec![2, 1], &[1.0, 1.0]);
        let k = t32(vec![2, 1], &[0.0, 0.0]);
        let v = t32(vec![2, 1], &[2.0, 4.0]);
        let s = Value::Scalar(ScalarValue::F32(1.0));
        let w = Value::Scalar(ScalarValue::I32(0));
        let c = Value::Scalar(ScalarValue::Bool(true));
        let y = f32s(attention(0, &[q.clone(), k.clone(), v.clone(), s.clone(), w.clone(), c]).unwrap());
        assert_eq!(y, vec![2.0, 3.0]);
        let y = f32s(attention(0, &[q, k, v, s, w]).unwrap());
        assert_eq!(y, vec![3.0, 3.0]);
    }

    #[test]
    fn blocks_quantize_as_gguf_does() {
        let x: Vec<f32> = (0..32).map(|i| i as f32 - 16.0).collect();
        let (q, d) = quantize_block(&x, false);
        assert_eq!(d, round_to_f16(16.0 / 127.0));
        assert_eq!(q[0], -127);
        assert_eq!(q[31], (15.0f32 * (127.0 / 16.0)).round() as i32);
        let (q, d) = quantize_block(&x, true);
        assert_eq!(d, 2.0);
        assert_eq!(q[0], -8);
        assert_eq!(*q.iter().max().unwrap(), 7);
        assert_eq!(round_to_f16(65519.0), 65504.0);
        assert!(round_to_f16(65520.0).is_infinite());
        assert_eq!(round_to_f16(2f32.powi(-25)), 0.0);
        assert_eq!(round_to_f16(2f32.powi(-24)), 2f32.powi(-24));
    }

    #[test]
    fn integer_lanes_trap_outside_their_type() {
        let a = HostTensor::from_i32(vec![2], &[100, -100]).unwrap();
        let b = HostTensor::from_i32(vec![2], &[27, -28]).unwrap();
        assert_eq!(int_binary(0, &rce_graph::graph::OpKind::Add, &a, &b, "i8").unwrap().as_i32().unwrap(), vec![127, -128]);
        let c = HostTensor::from_i32(vec![2], &[28, 0]).unwrap();
        let e = int_binary(0, &rce_graph::graph::OpKind::Add, &a, &c, "i8").unwrap_err();
        assert_eq!(e.to_string(), "trap in node 0: overflow: 128 is outside i8");
        let e = int_binary(0, &rce_graph::graph::OpKind::Div, &a, &c, "i32").unwrap_err();
        assert!(e.to_string().ends_with("division by zero"));
    }
}
