//! WGSL compute shader sources.
//!
//! These are the v0.1 reference kernels. They are intentionally simple:
//! correctness > speed. SIMD/blocking, fused ops, and quantised paths
//! land in v0.2 once we have a benchmark scaffold to validate against.

/// Element-wise add over two f32 tensors of equal shape, flattened.
/// Bindings:
///   @binding(0): a (read storage)
///   @binding(1): b (read storage)
///   @binding(2): out (read_write storage)
///   @binding(3): n (uniform u32, total element count)
pub const ELEMENTWISE_ADD_F32: &str = r#"
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Dims { n: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.n) { return; }
    out[i] = a[i] + b[i];
}
"#;

/// Generalised element-wise op over equal-shape f32 tensors.
/// `dims.op` selects: 0=add, 1=sub, 2=mul, 3=div.
pub const ELEMENTWISE_F32: &str = r#"
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Dims { n: u32, op: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.n) { return; }
    let av = a[i];
    let bv = b[i];
    var r: f32 = 0.0;
    if (dims.op == 0u)      { r = av + bv; }
    else if (dims.op == 1u) { r = av - bv; }
    else if (dims.op == 2u) { r = av * bv; }
    else                    { r = av / bv; }
    out[i] = r;
}
"#;

/// Reduce along the last axis: sum / mean / max / min on f32.
/// `dims.outer` = product of leading dims, `dims.dim` = last-axis length,
/// `dims.op` = 0=sum, 1=mean, 2=max, 3=min. One thread per outer row.
pub const REDUCE_LAST_AXIS_F32: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Dims { outer: u32, dim: u32, op: u32, _pad: u32 };
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.outer) { return; }
    let base = row * dims.dim;

    var acc: f32;
    if (dims.op == 0u || dims.op == 1u) {
        acc = 0.0;
    } else if (dims.op == 2u) {
        acc = -3.4028234663852886e38;  // -infinity-ish for f32
    } else {
        acc = 3.4028234663852886e38;   // +infinity-ish for f32
    }

    for (var i: u32 = 0u; i < dims.dim; i = i + 1u) {
        let v = x[base + i];
        if (dims.op == 0u || dims.op == 1u) {
            acc = acc + v;
        } else if (dims.op == 2u) {
            if (v > acc) { acc = v; }
        } else {
            if (v < acc) { acc = v; }
        }
    }
    if (dims.op == 1u) {
        acc = acc / f32(dims.dim);
    }
    out[row] = acc;
}
"#;

/// Softmax along the last axis. `dims.outer` = product of leading dims,
/// `dims.dim` = last-axis length. One thread per outer row.
pub const SOFTMAX_LAST_AXIS_F32: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Dims { outer: u32, dim: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.outer) { return; }
    let base = row * dims.dim;

    // Numerically stable: subtract max.
    var max_v: f32 = x[base];
    for (var i: u32 = 1u; i < dims.dim; i = i + 1u) {
        let v = x[base + i];
        if (v > max_v) { max_v = v; }
    }
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < dims.dim; i = i + 1u) {
        let e = exp(x[base + i] - max_v);
        out[base + i] = e;
        sum = sum + e;
    }
    for (var i: u32 = 0u; i < dims.dim; i = i + 1u) {
        out[base + i] = out[base + i] / sum;
    }
}
"#;

/// 2-D transpose `(rows, cols) → (cols, rows)`, f32, row-major.
pub const TRANSPOSE_2D_F32: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Dims { rows: u32, cols: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= dims.rows || j >= dims.cols) { return; }
    out[j * dims.rows + i] = x[i * dims.cols + j];
}
"#;

/// Element-wise unary activations on f32. `dims.op` selects:
/// 0=relu, 1=sigmoid, 2=tanh, 3=gelu (tanh approximation).
pub const ACTIVATION_F32: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Dims { n: u32, op: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.n) { return; }
    let v = x[i];
    var r: f32 = 0.0;
    if (dims.op == 0u) {
        r = max(0.0, v);
    } else if (dims.op == 1u) {
        r = 1.0 / (1.0 + exp(-v));
    } else if (dims.op == 2u) {
        r = tanh(v);
    } else {
        // GeLU (tanh approximation, OpenAI/HuggingFace convention).
        let SQRT_2_OVER_PI: f32 = 0.7978845608;
        let inner = SQRT_2_OVER_PI * (v + 0.044715 * v * v * v);
        r = 0.5 * v * (1.0 + tanh(inner));
    }
    out[i] = r;
}
"#;

/// Elementwise binary with scalar broadcast: one of the two operands is a
/// 0-D tensor (length 1). `dims.op` is {0,1,2,3} = {add, sub, mul, div};
/// `dims.scalar_left=1` ⇒ `a` is the scalar; else `b` is the scalar.
pub const ELEMENTWISE_SCALAR_BCAST_F32: &str = r#"
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Dims { n: u32, op: u32, scalar_left: u32, _pad: u32 };
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.n) { return; }
    var av: f32;
    var bv: f32;
    if (dims.scalar_left == 1u) {
        av = a[0];
        bv = b[i];
    } else {
        av = a[i];
        bv = b[0];
    }
    var r: f32 = 0.0;
    if (dims.op == 0u)      { r = av + bv; }
    else if (dims.op == 1u) { r = av - bv; }
    else if (dims.op == 2u) { r = av * bv; }
    else                    { r = av / bv; }
    out[i] = r;
}
"#;

/// General N-D broadcast elementwise on f32 tensors. Supports rank ≤ 4.
///
/// `dims.out_shape[i]`     = output shape at axis i (right-aligned padding).
/// `dims.a_strides[i]`     = stride into `a`'s flat buffer for output axis i;
///                           0 means broadcast (input dim is 1 or absent).
/// `dims.b_strides[i]`     = same for `b`.
/// `dims.op`               = 0=add, 1=sub, 2=mul, 3=div.
/// `dims.rank`             = effective rank (≤ 4).
/// `dims.n`                = total output elements (product of out_shape[..rank]).
pub const ELEMENTWISE_BCAST_F32: &str = r#"
struct BcastDims {
    rank: u32,
    op: u32,
    n: u32,
    _pad: u32,
    out_shape: vec4<u32>,
    a_strides: vec4<u32>,
    b_strides: vec4<u32>,
};

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             dims: BcastDims;

fn at_shape(i: u32) -> u32 {
    switch (i) {
        case 0u: { return dims.out_shape.x; }
        case 1u: { return dims.out_shape.y; }
        case 2u: { return dims.out_shape.z; }
        default: { return dims.out_shape.w; }
    }
}
fn at_a_stride(i: u32) -> u32 {
    switch (i) {
        case 0u: { return dims.a_strides.x; }
        case 1u: { return dims.a_strides.y; }
        case 2u: { return dims.a_strides.z; }
        default: { return dims.a_strides.w; }
    }
}
fn at_b_stride(i: u32) -> u32 {
    switch (i) {
        case 0u: { return dims.b_strides.x; }
        case 1u: { return dims.b_strides.y; }
        case 2u: { return dims.b_strides.z; }
        default: { return dims.b_strides.w; }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lin = gid.x;
    if (lin >= dims.n) { return; }

    var rem: u32 = lin;
    var a_idx: u32 = 0u;
    var b_idx: u32 = 0u;
    // Walk axes right-to-left; the rank is bounded by 4.
    for (var step: u32 = 0u; step < 4u; step = step + 1u) {
        if (step >= dims.rank) { break; }
        let axis = dims.rank - 1u - step;
        let dim_size = at_shape(axis);
        let c = rem % dim_size;
        rem = rem / dim_size;
        a_idx = a_idx + c * at_a_stride(axis);
        b_idx = b_idx + c * at_b_stride(axis);
    }

    let av = a[a_idx];
    let bv = b[b_idx];
    var r: f32 = 0.0;
    if (dims.op == 0u)      { r = av + bv; }
    else if (dims.op == 1u) { r = av - bv; }
    else if (dims.op == 2u) { r = av * bv; }
    else                    { r = av / bv; }
    out[lin] = r;
}
"#;

/// Fused matmul + scale + softmax along the last axis.
///
/// `out = softmax((a @ b) * scale, axis=-1)`, one workgroup per output row.
/// `a` is `[M, K]`, `b` is `[K, N]`; output is `[M, N]`. The matmul phase
/// parallelises across columns; softmax across the row is serial on the
/// first thread (cheap given N ≤ 256). For larger N we fall through to
/// the unfused matmul + scale + softmax pipeline.
pub const MATMUL_SCALE_SOFTMAX_F32: &str = r#"
struct FusedDims {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
    scale: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             dims: FusedDims;

const TILE: u32 = 64u;
const MAX_N: u32 = 256u;
var<workgroup> row_buf: array<f32, 256>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg.x;
    let tid = lid.x;
    if (row >= dims.m) { return; }

    // Matmul phase: each thread fills out_row[j] for a stride of output cols.
    var j: u32 = tid;
    while (j < dims.n) {
        var acc: f32 = 0.0;
        for (var k: u32 = 0u; k < dims.k; k = k + 1u) {
            acc = acc + a[row * dims.k + k] * b[k * dims.n + j];
        }
        row_buf[j] = acc * dims.scale;
        j = j + TILE;
    }
    workgroupBarrier();

    // Softmax phase (serial on tid 0 — cheap for N ≤ 256).
    if (tid == 0u) {
        var max_v: f32 = row_buf[0];
        for (var jj: u32 = 1u; jj < dims.n; jj = jj + 1u) {
            let v = row_buf[jj];
            if (v > max_v) { max_v = v; }
        }
        var sum: f32 = 0.0;
        for (var jj: u32 = 0u; jj < dims.n; jj = jj + 1u) {
            let e = exp(row_buf[jj] - max_v);
            row_buf[jj] = e;
            sum = sum + e;
        }
        for (var jj: u32 = 0u; jj < dims.n; jj = jj + 1u) {
            out[row * dims.n + jj] = row_buf[jj] / sum;
        }
    }
}
"#;

/// Batched dense matmul: A `[B,M,K]` × B `[B,K,N]` → C `[B,M,N]`.
/// `dims.b` is the batch size; matrices share an outer batch dim.
pub const MATMUL_BATCHED_F32: &str = r#"
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Dims { batch: u32, m: u32, k: u32, n: u32 };
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let bb = gid.z;
    if (i >= dims.m || j >= dims.n || bb >= dims.batch) { return; }
    let a_off = bb * dims.m * dims.k;
    let b_off = bb * dims.k * dims.n;
    let out_off = bb * dims.m * dims.n;
    var acc: f32 = 0.0;
    for (var kk: u32 = 0u; kk < dims.k; kk = kk + 1u) {
        acc = acc + a[a_off + i * dims.k + kk] * b[b_off + kk * dims.n + j];
    }
    out[out_off + i * dims.n + j] = acc;
}
"#;

/// Dense f32 matmul: A[M,K] × B[K,N] → C[M,N], all row-major.
/// Bindings:
///   @binding(0): a
///   @binding(1): b
///   @binding(2): out
///   @binding(3): dims uniform (M, K, N, _)
pub const MATMUL_F32: &str = r#"
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= dims.m || j >= dims.n) { return; }
    var acc: f32 = 0.0;
    for (var kk: u32 = 0u; kk < dims.k; kk = kk + 1u) {
        acc = acc + a[i * dims.k + kk] * b[kk * dims.n + j];
    }
    out[i * dims.n + j] = acc;
}
"#;
