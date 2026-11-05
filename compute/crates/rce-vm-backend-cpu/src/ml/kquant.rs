//! The block formats Q4_K, Q5_K and Q6_K (spec 6.13, GT-4): their
//! products, which are exact, and their reference quantizer, which is
//! lossy.
//!
//! A super-block is 256 values along the last axis. In Q4_K and Q5_K it
//! has 8 sub-blocks of 32; the value at position p is `s q - o`, with q in
//! 0..15 (Q4_K) or 0..31 (Q5_K) and s and o the scale and offset of the
//! sub-block p / 32. In Q6_K it has 16 sub-blocks of 16; the value is
//! `s q`, q in -32..31, s the scale of the sub-block p / 16. A program
//! holds a K-quant matrix as its values q (an i8 tensor in position order)
//! and its sub-block scales s and offsets o (f32 tensors). In a GGUF file
//! s is `d sc` and o is `dmin m`, d and dmin binary16 and sc and m 6-bit
//! (Q6_K: sc an i8), so every s and o is an f32 value exactly.
use super::nn::round_to_f16;
use super::arg_tensor;
use crate::error::{InterpretError, Result};
use crate::tensor::{DType, HostTensor};
use crate::value::Value;

/// The values of a super-block.
pub const QK_K: usize = 256;

/// A K-quant format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KFmt {
    Q4K,
    Q5K,
    Q6K,
}

impl KFmt {
    /// the operation names of its product and its rounding
    pub fn names(self) -> (&'static str, &'static str) {
        match self {
            KFmt::Q4K => ("matmul_q4_k", "round_q4_k"),
            KFmt::Q5K => ("matmul_q5_k", "round_q5_k"),
            KFmt::Q6K => ("matmul_q6_k", "round_q6_k"),
        }
    }
    /// the values of a sub-block
    pub fn sub(self) -> usize {
        if self == KFmt::Q6K { 16 } else { 32 }
    }
    /// the range of q
    pub fn range(self) -> (i32, i32) {
        match self {
            KFmt::Q4K => (0, 15),
            KFmt::Q5K => (0, 31),
            KFmt::Q6K => (-32, 31),
        }
    }
    /// the bytes of a super-block in a GGUF file
    pub fn block_bytes(self) -> usize {
        match self {
            KFmt::Q4K => 144,
            KFmt::Q5K => 176,
            KFmt::Q6K => 210,
        }
    }
}

fn mismatch(node_id: usize, msg: String) -> InterpretError {
    InterpretError::TypeMismatch { node_id, msg }
}

/// `matmul_q4_k(x, q, s, o)`, `matmul_q5_k(x, q, s, o)` and
/// `matmul_q6_k(x, q, s)`: x (m, k) f32, k a multiple of 256; q (n, k) the
/// values (i8 in i32 lanes; one outside the format's range is the trap
/// `Q4_K value V outside 0..15`); s (and o) (n, k / 32) f32, or (n, k / 16)
/// for Q6_K; to (m, n). Each y_ij sums from +0.0, in increasing order:
/// - Q4_K, Q5_K: for each pair of sub-blocks (2g, 2g + 1), with
///   u = sum of q_t x_t and a = sum of x_t over each one's 32 positions in
///   increasing t from +0.0, y = y + (((s_2g u_2g - o_2g a_2g) + s_2g+1 u_2g+1) - o_2g+1 a_2g+1);
/// - Q6_K: for each half of 128 of each super-block, for l in 0..31, for
///   the positions l, l + 32, l + 64, l + 96 of the half in that order,
///   y = y + (s q) x.
///
/// This is the order of `rce-op-qmm`'s portable row kernels
/// `dot_row_q4_k`, `dot_row_q5_k` and `dot_row_q6_k`; each product and
/// sum is one binary32 operation.
pub fn matmul_k(node_id: usize, args: &[Value], f: KFmt) -> Result<Value> {
    let kind = f.names().0;
    let want = if f == KFmt::Q6K { 3 } else { 4 };
    if args.len() != want {
        return Err(InterpretError::ArityMismatch { node_id, kind: kind.into(), expected: want, got: args.len() });
    }
    let x = arg_tensor(node_id, &args[0])?;
    let q = arg_tensor(node_id, &args[1])?;
    let s = arg_tensor(node_id, &args[2])?;
    let o = if want == 4 { Some(arg_tensor(node_id, &args[3])?) } else { None };
    let sub = f.sub();
    let ok = x.dtype == DType::F32 && q.dtype == DType::I32 && s.dtype == DType::F32 && x.shape.len() == 2 && q.shape.len() == 2;
    if !ok || x.shape[1] != q.shape[1] || x.shape[1] % QK_K != 0 || s.shape != [q.shape[0], q.shape[1] / sub] || o.as_ref().is_some_and(|o| o.dtype != DType::F32 || o.shape != s.shape) {
        return Err(mismatch(node_id, format!("{kind}: x {:?} {}, q {:?} {}, s {:?} {}", x.shape, x.dtype.name(), q.shape, q.dtype.name(), s.shape, s.dtype.name())));
    }
    let (m, k, n) = (x.shape[0], x.shape[1], q.shape[0]);
    let (xs, qs, ss) = (x.as_f32()?, q.as_i32()?, s.as_f32()?);
    let os = match &o {
        Some(o) => o.as_f32()?,
        None => vec![],
    };
    let (lo, hi) = f.range();
    if let Some(bad) = qs.iter().find(|v| **v < lo || **v > hi) {
        let what = match f {
            KFmt::Q4K => "Q4_K",
            KFmt::Q5K => "Q5_K",
            KFmt::Q6K => "Q6_K",
        };
        return Err(InterpretError::Trap(node_id, format!("{what} value {bad} outside {lo}..{hi}")));
    }
    let ns = k / sub;
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let xr = &xs[i * k..(i + 1) * k];
        for j in 0..n {
            let qr = &qs[j * k..(j + 1) * k];
            let sr = &ss[j * ns..(j + 1) * ns];
            let mut y = 0.0f32;
            if f == KFmt::Q6K {
                for sb in 0..k / QK_K {
                    for half in 0..2 {
                        let base = sb * QK_K + half * 128;
                        for l in 0..32 {
                            for p in [base + l, base + l + 32, base + l + 64, base + l + 96] {
                                y += sr[p / 16] * qr[p] as f32 * xr[p];
                            }
                        }
                    }
                }
            } else {
                let or = &os[j * ns..(j + 1) * ns];
                for g in 0..k / 64 {
                    let (b1, b2) = (2 * g, 2 * g + 1);
                    let (mut u1, mut a1, mut u2, mut a2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    for t in 0..32 {
                        let (p1, p2) = (b1 * 32 + t, b2 * 32 + t);
                        u1 += qr[p1] as f32 * xr[p1];
                        a1 += xr[p1];
                        u2 += qr[p2] as f32 * xr[p2];
                        a2 += xr[p2];
                    }
                    y += sr[b1] * u1 - or[b1] * a1 + sr[b2] * u2 - or[b2] * a2;
                }
            }
            out[i * n + j] = y;
        }
    }
    HostTensor::from_f32(vec![m, n], &out).map(Value::Tensor)
}

/// One super-block quantized: the values q (position order), the 6-bit
/// sub-block scales sc and offsets mn (Q6_K: the i8 scales sc, mn empty),
/// and the binary16 values d and dmin (Q6_K: dmin 0).
#[derive(Clone, Debug, PartialEq)]
pub struct KBlock {
    pub q: Vec<i32>,
    pub sc: Vec<i32>,
    pub mn: Vec<i32>,
    pub d: f32,
    pub dmin: f32,
}

impl KBlock {
    /// the scale s and offset o of each sub-block, exactly
    pub fn scales(&self) -> (Vec<f32>, Vec<f32>) {
        (self.sc.iter().map(|c| self.d * *c as f32).collect(), self.mn.iter().map(|c| self.dmin * *c as f32).collect())
    }
    /// the values the block represents: s q - o, or s q for Q6_K
    pub fn values(&self, f: KFmt) -> Vec<f32> {
        let (s, o) = self.scales();
        let sub = f.sub();
        self.q.iter().enumerate().map(|(p, q)| if f == KFmt::Q6K { s[p / sub] * *q as f32 } else { s[p / sub] * *q as f32 - o[p / sub] }).collect()
    }
}

/// x rounded half away from zero to an integer, as `f32::round`.
fn rnd(x: f32) -> i32 {
    x.round() as i32
}

/// RCE's reference quantizer of one super-block of 256 values; every
/// operation binary32, `round` half away from zero.
/// - Q4_K, Q5_K (nmax 15, 31): per sub-block b of 32, lo = min(0, least x),
///   hi = the largest x, w_b = (hi - lo) / nmax and v_b = -lo; d and dmin are
///   (the largest w_b) / 63 and (the largest v_b) / 63 rounded to binary16;
///   sc_b = min(63, round(w_b / d)) (0 when d = 0), m_b likewise from v_b
///   and dmin; with s = d sc_b and o = dmin m_b, q = min(nmax, max(0,
///   round((x + o) / s))) (0 when s = 0).
/// - Q6_K: per sub-block b of 16, w_b = (the first x of largest magnitude)
///   / -32; d = (the largest |w_b|) / 127 rounded to binary16; sc_b =
///   min(127, max(-128, round(w_b / d))) (0 when d = 0); with s = d sc_b,
///   q = min(31, max(-32, round(x / s))) (0 when s = 0).
pub fn quantize_k(x: &[f32], f: KFmt) -> KBlock {
    assert_eq!(x.len(), QK_K);
    let sub = f.sub();
    if f == KFmt::Q6K {
        let w: Vec<f32> = x
            .chunks(sub)
            .map(|c| {
                let (mut amax, mut m) = (0.0f32, 0.0f32);
                for v in c {
                    if amax < v.abs() {
                        amax = v.abs();
                        m = *v;
                    }
                }
                m / -32.0
            })
            .collect();
        let wmax = w.iter().fold(0.0f32, |a, v| if a < v.abs() { v.abs() } else { a });
        let d = round_to_f16(wmax / 127.0);
        let sc: Vec<i32> = w.iter().map(|v| if d != 0.0 { rnd(v / d).clamp(-128, 127) } else { 0 }).collect();
        let q = x
            .iter()
            .enumerate()
            .map(|(p, v)| {
                let s = d * sc[p / sub] as f32;
                if s != 0.0 { rnd(v / s).clamp(-32, 31) } else { 0 }
            })
            .collect();
        return KBlock { q, sc, mn: vec![], d, dmin: 0.0 };
    }
    let nmax = f.range().1 as f32;
    let (mut w, mut v) = (vec![], vec![]);
    for c in x.chunks(sub) {
        let lo = c.iter().fold(0.0f32, |a, b| if *b < a { *b } else { a });
        let hi = c.iter().fold(f32::NEG_INFINITY, |a, b| if *b > a { *b } else { a });
        w.push((hi - lo) / nmax);
        v.push(-lo);
    }
    let big = |u: &[f32]| u.iter().fold(0.0f32, |a, b| if *b > a { *b } else { a });
    let d = round_to_f16(big(&w) / 63.0);
    let dmin = round_to_f16(big(&v) / 63.0);
    let sc: Vec<i32> = w.iter().map(|u| if d != 0.0 { rnd(u / d).min(63) } else { 0 }).collect();
    let mn: Vec<i32> = v.iter().map(|u| if dmin != 0.0 { rnd(u / dmin).min(63) } else { 0 }).collect();
    let q = x
        .iter()
        .enumerate()
        .map(|(p, xv)| {
            let (s, o) = (d * sc[p / sub] as f32, dmin * mn[p / sub] as f32);
            if s != 0.0 { rnd((xv + o) / s).clamp(0, nmax as i32) } else { 0 }
        })
        .collect();
    KBlock { q, sc, mn, d, dmin }
}

/// `round_q4_k(x)`, `round_q5_k(x)` and `round_q6_k(x)`: f32 x whose last
/// length is a multiple of 256, each super-block along the last axis
/// replaced by the values its quantization represents ([`quantize_k`]).
/// It lowers precision (lossy).
pub fn round_k(node_id: usize, args: &[Value], f: KFmt) -> Result<Value> {
    let kind = f.names().1;
    if args.len() != 1 {
        return Err(InterpretError::ArityMismatch { node_id, kind: kind.into(), expected: 1, got: args.len() });
    }
    let x = arg_tensor(node_id, &args[0])?;
    if x.dtype != DType::F32 || x.shape.last().is_none_or(|l| l % QK_K != 0) {
        return Err(mismatch(node_id, format!("{kind}: x {:?} {}", x.shape, x.dtype.name())));
    }
    let mut out = Vec::with_capacity(x.numel());
    for b in x.as_f32()?.chunks(QK_K) {
        out.extend(quantize_k(b, f).values(f));
    }
    HostTensor::from_f32(x.shape.clone(), &out).map(Value::Tensor)
}

fn f16_bits(x: f32) -> u16 {
    // x is a binary16 value (the quantizer rounded it)
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let a = x.abs();
    if a == 0.0 {
        return s;
    }
    if a < 2f32.powi(-14) {
        return s | (a / 2f32.powi(-24)) as u16;
    }
    let e = ((a.to_bits() >> 23) as i32) - 127 + 15;
    s | ((e as u16) << 10) | ((a.to_bits() >> 13) & 0x3ff) as u16
}

/// The 12 bytes of the 6-bit scales and offsets of Q4_K and Q5_K, as GGUF
/// packs them.
pub fn pack_scales(sc: &[i32], mn: &[i32]) -> [u8; 12] {
    let mut s = [0u8; 12];
    for j in 0..4 {
        s[j] = (sc[j] as u8 & 63) | (((sc[j + 4] as u8) >> 4) << 6);
        s[j + 4] = (mn[j] as u8 & 63) | (((mn[j + 4] as u8) >> 4) << 6);
        s[j + 8] = (sc[j + 4] as u8 & 15) | ((mn[j + 4] as u8 & 15) << 4);
    }
    s
}

/// The 6-bit scale and offset of sub-block j from the 12 packed bytes.
pub fn unpack_scale(s: &[u8], j: usize) -> (i32, i32) {
    if j < 4 {
        ((s[j] & 63) as i32, (s[j + 4] & 63) as i32)
    } else {
        (((s[j + 4] & 15) | ((s[j - 4] >> 6) << 4)) as i32, ((s[j + 4] >> 4) | ((s[j] >> 6) << 4)) as i32)
    }
}

/// A super-block as the bytes of a GGUF file.
pub fn pack(b: &KBlock, f: KFmt) -> Vec<u8> {
    let mut out = vec![];
    match f {
        KFmt::Q4K | KFmt::Q5K => {
            out.extend(f16_bits(b.d).to_le_bytes());
            out.extend(f16_bits(b.dmin).to_le_bytes());
            out.extend(pack_scales(&b.sc, &b.mn));
            if f == KFmt::Q5K {
                let mut qh = [0u8; 32];
                for g in 0..4 {
                    for l in 0..32 {
                        qh[l] |= (((b.q[g * 64 + l] >> 4) & 1) as u8) << (2 * g);
                        qh[l] |= (((b.q[g * 64 + 32 + l] >> 4) & 1) as u8) << (2 * g + 1);
                    }
                }
                out.extend(qh);
            }
            for g in 0..4 {
                for l in 0..32 {
                    out.push((b.q[g * 64 + l] & 15) as u8 | (((b.q[g * 64 + 32 + l] & 15) as u8) << 4));
                }
            }
        }
        KFmt::Q6K => {
            let (mut ql, mut qh) = ([0u8; 128], [0u8; 64]);
            for half in 0..2 {
                for l in 0..32 {
                    let u = |p: usize| (b.q[half * 128 + p] + 32) as u8;
                    let (q1, q2, q3, q4) = (u(l), u(l + 32), u(l + 64), u(l + 96));
                    ql[half * 64 + l] = (q1 & 15) | ((q3 & 15) << 4);
                    ql[half * 64 + l + 32] = (q2 & 15) | ((q4 & 15) << 4);
                    qh[half * 32 + l] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
                }
            }
            out.extend(ql);
            out.extend(qh);
            out.extend(b.sc.iter().map(|c| *c as i8 as u8));
            out.extend(f16_bits(b.d).to_le_bytes());
        }
    }
    out
}

/// A super-block read back from the bytes of a GGUF file, d and dmin as
/// f32 by `f16` (the caller's exact binary16 conversion).
pub fn unpack(bytes: &[u8], f: KFmt, f16: fn(u16) -> f32) -> KBlock {
    let h = |i: usize| f16(u16::from_le_bytes([bytes[i], bytes[i + 1]]));
    match f {
        KFmt::Q4K | KFmt::Q5K => {
            let (d, dmin) = (h(0), h(2));
            let sc_b = &bytes[4..16];
            let (mut sc, mut mn) = (vec![], vec![]);
            for j in 0..8 {
                let (a, b) = unpack_scale(sc_b, j);
                sc.push(a);
                mn.push(b);
            }
            let (qh, qs) = if f == KFmt::Q5K { (&bytes[16..48], &bytes[48..176]) } else { (&bytes[0..0], &bytes[16..144]) };
            let mut q = vec![0i32; QK_K];
            for g in 0..4 {
                for l in 0..32 {
                    let byte = qs[g * 32 + l];
                    let (mut lo, mut hi) = ((byte & 15) as i32, (byte >> 4) as i32);
                    if f == KFmt::Q5K {
                        lo |= (((qh[l] >> (2 * g)) & 1) as i32) << 4;
                        hi |= (((qh[l] >> (2 * g + 1)) & 1) as i32) << 4;
                    }
                    q[g * 64 + l] = lo;
                    q[g * 64 + 32 + l] = hi;
                }
            }
            KBlock { q, sc, mn, d, dmin }
        }
        KFmt::Q6K => {
            let (ql, qh) = (&bytes[0..128], &bytes[128..192]);
            let sc = bytes[192..208].iter().map(|b| *b as i8 as i32).collect();
            let d = h(208);
            let mut q = vec![0i32; QK_K];
            for half in 0..2 {
                for l in 0..32 {
                    let (a, b, hb) = (ql[half * 64 + l], ql[half * 64 + l + 32], qh[half * 32 + l]);
                    let at = half * 128;
                    q[at + l] = ((a & 15) | ((hb & 3) << 4)) as i32 - 32;
                    q[at + l + 32] = ((b & 15) | (((hb >> 2) & 3) << 4)) as i32 - 32;
                    q[at + l + 64] = ((a >> 4) | (((hb >> 4) & 3) << 4)) as i32 - 32;
                    q[at + l + 96] = ((b >> 4) | (((hb >> 6) & 3) << 4)) as i32 - 32;
                }
            }
            KBlock { q, sc, mn: vec![], d, dmin: 0.0 }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * 0.7
            })
            .collect()
    }

    fn f16(h: u16) -> f32 {
        half_free(h)
    }

    // binary16 to f32, exactly (a copy of the lift's, to keep this crate free of it)
    fn half_free(h: u16) -> f32 {
        let s = ((h >> 15) as u32) << 31;
        let e = ((h >> 10) & 0x1f) as i32;
        let m = (h & 0x3ff) as u32;
        let mag = if e == 0 { (m as f32) * f32::from_bits(0x3380_0000) } else { f32::from_bits((((e - 15 + 127) as u32) << 23) | (m << 13)) };
        f32::from_bits(mag.to_bits() | s)
    }

    #[test]
    fn packing_round_trips_and_values_are_close() {
        for f in [KFmt::Q4K, KFmt::Q5K, KFmt::Q6K] {
            let x = lcg(7 + f as u64, QK_K);
            let b = quantize_k(&x, f);
            let bytes = pack(&b, f);
            assert_eq!(bytes.len(), f.block_bytes());
            assert_eq!(unpack(&bytes, f, f16), b, "{f:?}");
            let v = b.values(f);
            let err = x.iter().zip(&v).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(err < 0.1, "{f:?} {err}");
        }
    }

    #[test]
    fn round_k_replaces_each_super_block_by_its_values() {
        for f in [KFmt::Q4K, KFmt::Q5K, KFmt::Q6K] {
            let x = lcg(11, 2 * QK_K);
            let v = Value::Tensor(HostTensor::from_f32(vec![2, QK_K], &x).unwrap());
            let Value::Tensor(r) = round_k(0, &[v], f).unwrap() else { panic!() };
            let want: Vec<f32> = x.chunks(QK_K).flat_map(|b| quantize_k(b, f).values(f)).collect();
            assert_eq!(r.as_f32().unwrap(), want);
            let bad = Value::Tensor(HostTensor::from_f32(vec![32], &x[..32]).unwrap());
            assert!(round_k(0, &[bad], f).is_err());
        }
    }

    #[test]
    fn a_value_outside_the_format_traps() {
        let x = Value::Tensor(HostTensor::from_f32(vec![1, QK_K], &vec![1.0; QK_K]).unwrap());
        let mut q = vec![0i32; QK_K];
        q[3] = 16;
        let q = Value::Tensor(HostTensor::from_i32(vec![1, QK_K], &q).unwrap());
        let s = Value::Tensor(HostTensor::from_f32(vec![1, 8], &[1.0; 8]).unwrap());
        let e = matmul_k(0, &[x, q, s.clone(), s], KFmt::Q4K).unwrap_err();
        assert!(format!("{e:?}").contains("Q4_K value 16 outside 0..15"), "{e:?}");
    }

    #[test]
    fn rope_rotates_adjacent_pairs_by_the_position() {
        use crate::ml::nn::{e_cos, e_sin, rope};
        let x: Vec<f32> = (0..2 * 2 * 4).map(|i| i as f32 * 0.25 - 1.0).collect();
        let fr = [1.0f32, 0.01];
        let args = [
            Value::Tensor(HostTensor::from_f32(vec![2, 2, 4], &x).unwrap()),
            Value::Tensor(HostTensor::from_i32(vec![2], &[0, 3]).unwrap()),
            Value::Tensor(HostTensor::from_f32(vec![2], &fr).unwrap()),
        ];
        let Value::Tensor(y) = rope(0, &args).unwrap() else { panic!() };
        let y = y.as_f32().unwrap();
        // position 0: the identity
        assert_eq!(&y[..8], &x[..8]);
        // position 3, head 1, pair 1: angle 3 * 0.01
        let a = 3.0f32 * 0.01;
        let (c, n) = (e_cos(a), e_sin(a));
        let at = 8 + 4 + 2;
        assert_eq!(y[at], x[at] * c - x[at + 1] * n);
        assert_eq!(y[at + 1], x[at] * n + x[at + 1] * c);
        assert_eq!(e_sin(0.5f32), (0.479425538604203f64) as f32);
    }
}
