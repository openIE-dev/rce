//! Correctly rounded `exp`, `tanh` and `erf` in binary64: the functions of
//! RCE's standard library module `math` (`stdlib/lib/math/math.rce`),
//! evaluated by the same method so that a graph term's elementwise
//! operations give the bits RCE source gives (spec 6.3 F9, 6.13 GT-4).
//!
//! Method, as in `math.rce`: each function computes an integer N with
//! |N / 2^F - f(x)| < 2^16 / 2^F in exact integer arithmetic, rounds
//! N - 2^16 and N + 2^16 to binary64, and returns when both round to the
//! same double; otherwise F grows (200, 500, 1000, 2000 bits). A
//! transcendental value of a nonzero double is never a rounding midpoint
//! (Lindemann), so the loop ends with the correctly rounded value. The
//! integers here are base 2^32 limbs where `math.rce` uses base 2^30; every
//! operation is the same floor of the same exact value, so the integers,
//! and with them the results, are the same. The constants ln 2 and
//! 2/sqrt(pi) are read from `math.rce` itself.
//!
//! A correctly rounded result is unique, so these functions equal
//! `math.exp`, `math.tanh` and `math.erf` bit for bit; the tests check them
//! against the MPFR vectors of `stdlib/tests/math` and against an oracle in
//! exact decimal arithmetic, and the checker's tests compare them with
//! `math.rce` run by the reference interpreter.
//!
//! As in `math.rce`, a result beyond the binary64 range is an infinity
//! (`exp` above about 709.78), which RCE's finite-value rule turns into the
//! trap `non-finite float` (6.3 F4). A NaN argument gives NaN; infinite
//! arguments give the IEEE limits (exp: +inf, +0; tanh and erf: +-1). RCE
//! values are finite, so a graph term never passes either.

use std::cmp::Ordering;
use std::sync::OnceLock;

/// A magnitude in base 2^32 limbs, least significant first, with no zero
/// limb at the top (zero is the empty vector).
#[derive(Clone, Debug, PartialEq, Eq)]
struct N(Vec<u32>);

impl N {
    fn zero() -> N {
        N(vec![])
    }

    fn of(v: u64) -> N {
        let mut o = vec![v as u32, (v >> 32) as u32];
        trim(&mut o);
        N(o)
    }

    fn pow2(s: i64) -> N {
        N::of(1).shl(s)
    }

    fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    fn to_u64(&self) -> u64 {
        let lo = self.0.first().copied().unwrap_or(0) as u64;
        let hi = self.0.get(1).copied().unwrap_or(0) as u64;
        lo | (hi << 32)
    }

    fn bits(&self) -> i64 {
        match self.0.last() {
            None => 0,
            Some(t) => (self.0.len() as i64 - 1) * 32 + (32 - t.leading_zeros() as i64),
        }
    }

    fn bit(&self, i: i64) -> bool {
        if i < 0 {
            return false;
        }
        let q = (i / 32) as usize;
        q < self.0.len() && (self.0[q] >> (i % 32)) & 1 == 1
    }

    /// Whether any of the low `s` bits is set.
    fn low_nonzero(&self, s: i64) -> bool {
        if s <= 0 {
            return false;
        }
        let q = (s / 32) as usize;
        let b = (s % 32) as u32;
        for (i, l) in self.0.iter().enumerate() {
            if i < q && *l != 0 {
                return true;
            }
            if i == q {
                return b > 0 && (*l & ((1u32 << b) - 1)) != 0;
            }
        }
        false
    }

    fn cmp(&self, o: &N) -> Ordering {
        if self.0.len() != o.0.len() {
            return self.0.len().cmp(&o.0.len());
        }
        for i in (0..self.0.len()).rev() {
            if self.0[i] != o.0[i] {
                return self.0[i].cmp(&o.0[i]);
            }
        }
        Ordering::Equal
    }

    fn add(&self, o: &N) -> N {
        let n = self.0.len().max(o.0.len());
        let mut out = Vec::with_capacity(n + 1);
        let mut c = 0u64;
        for i in 0..n {
            let s = c + *self.0.get(i).unwrap_or(&0) as u64 + *o.0.get(i).unwrap_or(&0) as u64;
            out.push(s as u32);
            c = s >> 32;
        }
        if c > 0 {
            out.push(c as u32);
        }
        N(out)
    }

    /// self - o, self >= o
    fn sub(&self, o: &N) -> N {
        let mut out = Vec::with_capacity(self.0.len());
        let mut br = 0i64;
        for i in 0..self.0.len() {
            let mut s = self.0[i] as i64 - br - *o.0.get(i).unwrap_or(&0) as i64;
            if s < 0 {
                s += 1 << 32;
                br = 1;
            } else {
                br = 0;
            }
            out.push(s as u32);
        }
        debug_assert_eq!(br, 0, "N::sub of a larger magnitude");
        trim(&mut out);
        N(out)
    }

    fn mul(&self, o: &N) -> N {
        if self.is_zero() || o.is_zero() {
            return N::zero();
        }
        let mut out = vec![0u32; self.0.len() + o.0.len()];
        for (i, a) in self.0.iter().enumerate() {
            let mut c = 0u64;
            for (j, b) in o.0.iter().enumerate() {
                let t = out[i + j] as u64 + (*a as u64) * (*b as u64) + c;
                out[i + j] = t as u32;
                c = t >> 32;
            }
            out[i + o.0.len()] = c as u32;
        }
        trim(&mut out);
        N(out)
    }

    fn muls(&self, s: u64) -> N {
        self.mul(&N::of(s))
    }

    /// floor(self / s), s > 0
    fn divs(&self, s: u64) -> N {
        debug_assert!(s > 0 && s < (1 << 32));
        let mut out = vec![0u32; self.0.len()];
        let mut r = 0u64;
        for i in (0..self.0.len()).rev() {
            let t = (r << 32) | self.0[i] as u64;
            out[i] = (t / s) as u32;
            r = t % s;
        }
        trim(&mut out);
        N(out)
    }

    fn shr(&self, s: i64) -> N {
        if s < 0 {
            return self.shl(-s);
        }
        let q = (s / 32) as usize;
        let b = (s % 32) as u32;
        if q >= self.0.len() {
            return N::zero();
        }
        let mut out = Vec::with_capacity(self.0.len() - q);
        for i in q..self.0.len() {
            let mut v = self.0[i] >> b;
            if b > 0 && i + 1 < self.0.len() {
                v |= self.0[i + 1] << (32 - b);
            }
            out.push(v);
        }
        trim(&mut out);
        N(out)
    }

    fn shl(&self, s: i64) -> N {
        if s < 0 {
            return self.shr(-s);
        }
        if self.is_zero() {
            return N::zero();
        }
        let q = (s / 32) as usize;
        let b = (s % 32) as u32;
        let mut out = vec![0u32; q];
        let mut c = 0u32;
        for x in &self.0 {
            out.push((x << b) | c);
            c = if b > 0 { x >> (32 - b) } else { 0 };
        }
        out.push(c);
        trim(&mut out);
        N(out)
    }

    /// floor(self * 2^f / b), b > 0 (math.rce `n_div`), by long division.
    fn div(&self, b: &N, f: i64) -> N {
        let num = self.shl(f);
        let nb = num.bits();
        let mut q = vec![0u32; ((nb + 31) / 32) as usize];
        let mut r = N::zero();
        for i in (0..nb).rev() {
            r = r.shl(1);
            if num.bit(i) {
                r = r.add(&N::of(1));
            }
            if r.cmp(b) != Ordering::Less {
                r = r.sub(b);
                q[(i / 32) as usize] |= 1 << (i % 32);
            }
        }
        trim(&mut q);
        N(q)
    }
}

fn trim(v: &mut Vec<u32>) {
    while v.last() == Some(&0) {
        v.pop();
    }
}

/// 2^k, exactly, for -1074 <= k <= 1023.
fn pow2(k: i64) -> f64 {
    if k >= -1022 {
        f64::from_bits(((k + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (k + 1074))
    }
}

/// N * 2^sh rounded to the nearest binary64, ties to even, subnormals
/// included; a value beyond the range is +inf (math.rce `n_round`).
fn n_round(n: &N, sh: i64) -> f64 {
    let l = n.bits();
    if l == 0 {
        return 0.0;
    }
    let e = l - 1 + sh;
    if e > 1023 {
        return f64::INFINITY;
    }
    let mut drop = l - 53;
    if e < -1022 {
        drop = -1074 - sh;
    }
    if drop <= 0 {
        return (n.to_u64() as f64) * pow2(sh);
    }
    let mut q = n.shr(drop).to_u64();
    if n.bit(drop - 1) && (n.low_nonzero(drop - 1) || q % 2 == 1) {
        q += 1;
    }
    let s = sh + drop;
    // q <= 2^53, so q is exact; q * 2^s is exact unless it overflows
    if s > 1023 {
        return (q as f64) * pow2(1023) * pow2(s - 1023);
    }
    (q as f64) * pow2(s)
}

/// x = M 2^E exactly, x > 0 finite: (M, E).
fn mant(x: f64) -> (u64, i64) {
    let b = x.to_bits();
    let be = ((b >> 52) & 0x7ff) as i64;
    let frac = b & ((1u64 << 52) - 1);
    if be == 0 { (frac, -1074) } else { (frac | (1u64 << 52), be - 1075) }
}

/// The exponent k of x = m 2^k, 1 <= m < 2, x > 0 finite.
fn exponent(x: f64) -> i64 {
    let (m, e) = mant(x);
    (64 - m.leading_zeros() as i64 - 1) + e
}

/// floor(|x| 2^f)
fn fix(x: f64, f: i64) -> N {
    let (m, e) = mant(x.abs());
    N::of(m).shl(e + f)
}

const ZIV_F: [i64; 4] = [200, 500, 1000, 2000];
const EB: u64 = 65536;

/// The two roundings of N -+ 2^16 at scale 2^sh.
fn ziv(n: &N, sh: i64) -> (f64, f64) {
    let eb = N::of(EB);
    if n.cmp(&eb) != Ordering::Greater {
        return (0.0, 1.0);
    }
    (n_round(&n.sub(&eb), sh), n_round(&n.add(&eb), sh))
}

/// The limbs of a constant of `math.rce`, a function `fn NAME() -> List[i64]`
/// returning base 2^30 limbs, least significant first.
fn math_constant(name: &str) -> N {
    const SRC: &str = include_str!("../../../../stdlib/lib/math/math.rce");
    let head = format!("fn {name}() -> List[i64]");
    let at = SRC.find(&head).unwrap_or_else(|| panic!("math.rce has no `{head}`"));
    let rest = &SRC[at + head.len()..];
    let open = rest.find('[').expect("a list literal");
    let close = rest.find(']').expect("a list literal");
    let mut v = N::zero();
    let limbs: Vec<u64> = rest[open + 1..close]
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u64>().expect("a limb"))
        .collect();
    for l in limbs.iter().rev() {
        v = v.shl(30).add(&N::of(*l));
    }
    v
}

/// ln 2 at 2100 fraction bits (math.rce `c_ln2`).
fn c_ln2() -> &'static N {
    static C: OnceLock<N> = OnceLock::new();
    C.get_or_init(|| math_constant("c_ln2"))
}

/// ln 2 at 4200 fraction bits (math.rce `c_ln2x`).
fn c_ln2x() -> &'static N {
    static C: OnceLock<N> = OnceLock::new();
    C.get_or_init(|| math_constant("c_ln2x"))
}

/// 2/sqrt(pi) at 4200 fraction bits (math.rce `c_2osqrtpi`).
fn c_2osqrtpi() -> &'static N {
    static C: OnceLock<N> = OnceLock::new();
    C.get_or_init(|| math_constant("c_2osqrtpi"))
}

/// exp(r) 2^f for 0 <= r < 1 given at f fraction bits (Taylor).
fn exp_fix(r: &N, f: i64) -> N {
    let mut s = N::pow2(f);
    let mut t = N::pow2(f);
    let mut k = 1u64;
    while !t.is_zero() {
        t = t.mul(r).shr(f).divs(k);
        s = s.add(&t);
        k += 1;
    }
    s
}

/// Correctly rounded e^x (math.rce `exp`).
pub fn exp(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == f64::INFINITY {
        return x;
    }
    if x == f64::NEG_INFINITY {
        return 0.0;
    }
    if x == 0.0 || x.abs() < 5.551115123125783e-17 {
        return 1.0;
    }
    if x > 709.7827128933841 {
        return f64::INFINITY;
    }
    if x < -745.2 {
        return 0.0;
    }
    for &f in &ZIV_F {
        let xf = fix(x, f);
        let l = c_ln2().shr(2100 - f);
        let mut k = (x.abs() / 0.6931471805599453).floor() as i64;
        if x < 0.0 {
            k += 1;
        }
        let mut kl = l.muls(k as u64);
        let mut i = 0;
        while x >= 0.0 && kl.cmp(&xf) == Ordering::Greater && i < 4 {
            k -= 1;
            kl = l.muls(k as u64);
            i += 1;
        }
        i = 0;
        while x < 0.0 && kl.cmp(&xf) == Ordering::Less && i < 4 {
            k += 1;
            kl = l.muls(k as u64);
            i += 1;
        }
        let r = if x >= 0.0 { xf.sub(&kl) } else { kl.sub(&xf) };
        let e = exp_fix(&r, f);
        let kk = if x >= 0.0 { k } else { -k };
        let (a, b) = ziv(&e, kk - f);
        if a == b {
            return a;
        }
    }
    f64::NAN
}

/// exp(v) = N 2^(k - f) for v = (zneg ? -1 : 1) z / 2^f (math.rce `exp_big`).
fn exp_big(zneg: bool, z: &N, f: i64) -> (i64, N) {
    let zd = n_round(z, -f);
    let v = if zneg { 0.0 - zd } else { zd };
    let mut k = (v / 0.6931471805599453).floor() as i64;
    let l = c_ln2x().shr(4200 - f);
    let mut r = N::zero();
    for _ in 0..8 {
        let kl = l.muls(k.unsigned_abs());
        let kneg = k < 0;
        let rneg;
        if zneg != kneg {
            r = z.add(&kl);
            rneg = zneg;
        } else if z.cmp(&kl) != Ordering::Less {
            r = z.sub(&kl);
            rneg = zneg;
        } else {
            r = kl.sub(z);
            rneg = !zneg;
        }
        if rneg && !r.is_zero() {
            k -= 1;
        } else if r.cmp(&l) != Ordering::Less {
            k += 1;
        } else {
            break;
        }
    }
    (k, exp_fix(&r, f))
}

/// e^|x| and e^-|x| at g bits: (k1, N1 - N2', N1 + N2') (math.rce `hyp_parts`).
fn hyp_parts(ax: f64, g: i64) -> (i64, N, N) {
    let xf = fix(ax, g);
    let (k1, p) = exp_big(false, &xf, g);
    let (k2, q) = exp_big(true, &xf, g);
    let n2 = q.shr(k1 - k2);
    (k1, p.sub(&n2), p.add(&n2))
}

fn hyp_ex(ax: f64) -> i64 {
    let e = exponent(ax);
    if e < 0 { -e } else { 0 }
}

/// Correctly rounded tanh x (math.rce `tanh`).
pub fn tanh(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let ax = x.abs();
    if ax < 7.450580596923828e-09 {
        return x;
    }
    if ax > 22.0 {
        return if x < 0.0 { -1.0 } else { 1.0 };
    }
    let ex = hyp_ex(ax);
    for &f in &ZIV_F {
        let fe = f + ex;
        let (_, d, s) = hyp_parts(ax, fe + 64);
        let t = d.div(&s, fe);
        let (a, b) = ziv(&t, -fe);
        if a == b {
            return if x < 0.0 { -a } else { a };
        }
    }
    f64::NAN
}

/// erf |x| 2^G (math.rce `erf_fix`).
fn erf_fix(ax: f64, gg: i64) -> N {
    let (m, e) = mant(ax);
    let mm = N::of(m);
    let m2 = mm.mul(&mm);
    let x2 = m2.shl(2 * e + gg);
    let mut t = mm.shl(e + gg);
    let mut s = t.clone();
    let mut n = 0u64;
    let m22 = m2.muls(2);
    while !t.is_zero() {
        n += 1;
        let p = t.mul(&m22);
        let ps = p.shl(2 * e);
        t = ps.divs(2 * n + 1);
        s = s.add(&t);
    }
    let (k, ev) = exp_big(true, &x2, gg);
    let cg = c_2osqrtpi().shr(4200 - gg);
    let p = ev.mul(&s).mul(&cg);
    p.shr(2 * gg - k)
}

/// Correctly rounded erf x (math.rce `erf`).
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let ax = x.abs();
    if ax == 0.0 {
        return x;
    }
    if ax >= 6.0 {
        return if x < 0.0 { -1.0 } else { 1.0 };
    }
    let ex = hyp_ex(ax);
    for &f in &ZIV_F {
        let gg = f + ex + 64;
        let v = erf_fix(ax, gg);
        let (a, b) = ziv(&v, -gg);
        if a == b {
            return if x < 0.0 { -a } else { a };
        }
    }
    f64::NAN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers() {
        let a = N::of(0xffff_ffff_ffff_ffff);
        assert_eq!(a.add(&N::of(1)).bits(), 65);
        assert_eq!(a.mul(&a).shr(64).to_u64(), 0xffff_ffff_ffff_fffe);
        assert_eq!(N::of(1000).divs(7).to_u64(), 142);
        assert_eq!(N::of(1000).div(&N::of(7), 4).to_u64(), 16000 / 7);
        assert!(!N::of(0b10000).low_nonzero(4) && N::of(0b1000).low_nonzero(4) && N::of(0b1001).low_nonzero(1));
        assert_eq!(N::of(5).shl(70).shr(70).to_u64(), 5);
    }

    #[test]
    fn rounding_and_constants() {
        assert_eq!(n_round(&N::of(3), -1), 1.5);
        // 2^53 + 1 ties to even 2^53
        assert_eq!(n_round(&N::of((1 << 53) + 1), 0), 9007199254740992.0);
        assert_eq!(n_round(&N::of(1), -1074), 5e-324);
        assert_eq!(n_round(&N::of(1), -1075), 0.0);
        assert_eq!(n_round(&N::of(3), -1076), 5e-324);
        assert_eq!(n_round(&c_ln2().clone(), -2100), std::f64::consts::LN_2);
        assert_eq!(n_round(&c_ln2x().clone(), -4200), std::f64::consts::LN_2);
        assert_eq!(n_round(&c_2osqrtpi().clone(), -4200), std::f64::consts::FRAC_2_SQRT_PI);
    }

    #[test]
    fn special_values() {
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(-0.0), 1.0);
        assert_eq!(exp(1.0), std::f64::consts::E);
        assert!(exp(f64::NAN).is_nan());
        assert_eq!(exp(f64::INFINITY), f64::INFINITY);
        assert_eq!(exp(f64::NEG_INFINITY).to_bits(), 0);
        assert_eq!(exp(710.0), f64::INFINITY);
        assert_eq!(exp(-746.0).to_bits(), 0);
        assert_eq!(exp(5e-324), 1.0);
        assert_eq!(tanh(-0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(tanh(5e-324), 5e-324);
        assert_eq!(tanh(30.0), 1.0);
        assert_eq!(tanh(f64::NEG_INFINITY), -1.0);
        assert!(tanh(f64::NAN).is_nan());
        assert_eq!(erf(-0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(erf(6.0), 1.0);
        assert_eq!(erf(f64::INFINITY), 1.0);
        assert!(erf(f64::NAN).is_nan());
    }
}
