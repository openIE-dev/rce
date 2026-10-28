//! The runtime functions of floats, written in the core: the exact parts of
//! a binary64 value, arithmetic on unbounded natural numbers held as lists
//! of 32-bit digits, the shortest decimal digits of a float, and the texts
//! the reference gives a float: display (6.3 N3 and F9, interp
//! `float_text`), canonical JSON (6.1.1, interp `float_json`), the display
//! of an f32 value, Rust's `{:?}` of an f64 (by which interp `json_size`
//! measures a float), and the exact integer a trap message names (interp
//! `exact_int_text`).
//!
//! The core has no bit cast, so the parts of a float are found by exact
//! scalings: multiplying a binary64 value by a power of two is exact while
//! the result stays finite. The digits follow the free-format method of
//! Steele and White as Rust's `flt2dec` writes it (its Dragon strategy),
//! which is what the reference's text is defined from: the number of digits
//! is that of the shortest decimal that reads back to the float, and the
//! digits are then the closest decimal of that length (the even one on an
//! exact tie) when it reads back, and otherwise the one that does.
//!
//! Each function is built once and is the same for every program; none is
//! charged fuel. Every loop has a bounded shape (`shape`): a measure, or a
//! count bound computed before the loop, which the comment beside it shows
//! the loop cannot reach.
use super::build::{FB, Fns, R, count_t, lit, sym, wide};
use super::ir::*;
use super::rt;

/// Every builder of this module, by name (as [`rt::BUILDERS`]).
pub const BUILDERS: &[(&str, rt::Builder)] = &[
    ("text", text), ("json", json), ("debug", debug), ("text32", text32), ("int_text", int_text), ("int_to_f32", int_to_f32),
];

/// A natural number as its base-2^32 digits, least significant first, with
/// no high zero digits (zero is the empty list).
fn big() -> Ty {
    Ty::list(rt::u32t())
}

const B32: i128 = 1 << 32;

fn f64k(x: f64) -> Atom {
    Atom::K(Ty::F64, Lit::F64(x.to_bits()))
}

fn call(b: &mut FB, f: &Fns, id: usize, args: Vec<Atom>) -> Atom {
    let r = f.ret(id);
    b.call(sym(id), r, args)
}

/// An f64 operation with its domain test; the test never holds where the
/// lowering places it, so its branch halts with an internal trap.
fn fop(b: &mut FB, op: &str, x: Atom, y: Atom) -> R<Atom> {
    let c = b.p(&format!("f64.bad.{op}"), vec![], vec![x.clone(), y.clone()])?;
    b.guard_halt(c, |fb| fb.trap_value(Atom::bytes(b"internal: float scaling")))?;
    b.p(&format!("f64.{op}"), vec![], vec![x, y])
}

// ---- natural numbers

/// `big.norm(a)`: a without its high zero digits.
fn norm(f: &mut Fns) -> R<usize> {
    f.get("rt.big.norm", vec![big()], big(), |_, mut b| {
        // each iteration that goes on pops one digit: at most n0 of them
        let n0 = b.p("list.len", vec![rt::u32t()], vec![Atom::V(0)])?;
        let x = b.var(big(), Atom::V(0));
        let cnt = b.counter();
        b.loop_(|b| {
            let n = b.p("list.len", vec![rt::u32t()], vec![Atom::V(x)])?;
            let z = b.int_cmp("int.eq", n.clone(), lit(0))?;
            b.exit_if(z)?;
            let n1 = b.sub_in(n, lit(1), Ty::Int(0, INT_MAX))?;
            let d = b.list_at(Atom::V(x), n1)?;
            let nz = b.int_cmp("int.eq", d, lit(0))?;
            b.exit_unless(nz)?;
            b.count(cnt, n0.clone())?;
            let xv = b.take(x);
            let n2 = b.p("list.len", vec![rt::u32t()], vec![xv.clone()])?;
            let ne = b.int_cmp("int.eq", n2, lit(0))?;
            b.guard_halt(ne, |fb| fb.trap_value(Atom::bytes(b"internal: norm")))?;
            let r = b.prim("list.pop", vec![rt::u32t()], vec![xv])?;
            b.set(x, r[0].clone());
            Ok(())
        })?;
        let r = b.take(x);
        b.ret(r);
        Ok(b.finish())
    })
}

/// `big.of(m)`: the digits of an integer of type u64.
fn of_u64(f: &mut Fns) -> R<usize> {
    let nm = norm(f)?;
    f.get("rt.big.of", vec![Ty::u64()], big(), move |f, mut b| {
        let (q, r) = b.divmod_lit(b.arg(0), B32, Ty::u64())?;
        let q = b.cast_in(q, rt::u32t())?;
        let r = b.cast_in(r, rt::u32t())?;
        let l = b.p("list.new", vec![rt::u32t()], vec![])?;
        let l = b.p("list.push", vec![rt::u32t()], vec![l, r])?;
        let l = b.p("list.push", vec![rt::u32t()], vec![l, q])?;
        let r = call(&mut b, f, nm, vec![l]);
        b.ret(r);
        Ok(b.finish())
    })
}

/// `big.mul(a, c)`: a times c, c below 2^32.
fn mul_small(f: &mut Fns) -> R<usize> {
    let nm = norm(f)?;
    f.get("rt.big.mul", vec![big(), rt::u32t()], big(), move |f, mut b| {
        let (a, c) = (b.arg(0), b.arg(1));
        let n = b.p("list.len", vec![rt::u32t()], vec![a.clone()])?;
        let o0 = b.p("list.new", vec![rt::u32t()], vec![])?;
        let out = b.var(big(), o0);
        let carry = b.var(rt::u32t(), lit(0));
        b.for_n(n, |b, i| {
            let d = b.list_at(a.clone(), i)?;
            let p = b.mul_in(d, c.clone(), Ty::Int(0, (B32 - 1) * (B32 - 1)))?;
            let cv = b.take(carry);
            let t = b.add_in(p, cv, Ty::u64())?;
            let (q, r) = b.divmod_lit(t, B32, Ty::u64())?;
            let q = b.cast_in(q, rt::u32t())?;
            let r = b.cast_in(r, rt::u32t())?;
            let o = b.take(out);
            let o = b.p("list.push", vec![rt::u32t()], vec![o, r])?;
            b.set(out, o);
            b.set(carry, q);
            Ok(())
        })?;
        let cv = b.take(carry);
        let nz = b.int_cmp("int.eq", cv.clone(), lit(0))?;
        b.if_(nz, |_| Ok(()), |b| {
            let o = b.take(out);
            let o = b.p("list.push", vec![rt::u32t()], vec![o, cv])?;
            b.set(out, o);
            Ok(())
        })?;
        b.drop(a)?;
        let o = b.take(out);
        let r = call(&mut b, f, nm, vec![o]);
        b.ret(r);
        Ok(b.finish())
    })
}

/// The digit i of a, or 0 when a has no digit i.
fn digit_or_zero(b: &mut FB, a: &Atom, n: &Atom, i: &Atom) -> R<Atom> {
    let x = b.var(rt::u32t(), lit(0));
    let inside = b.int_cmp("int.lt", i.clone(), n.clone())?;
    let (a, i) = (a.clone(), i.clone());
    b.when(inside, |b| {
        let d = b.list_at(a, i)?;
        b.set(x, d);
        Ok(())
    })?;
    Ok(b.take(x))
}

/// `big.add(a, b)`.
fn add(f: &mut Fns) -> R<usize> {
    let nm = norm(f)?;
    f.get("rt.big.add", vec![big(), big()], big(), move |f, mut b| {
        let (x, y) = (b.arg(0), b.arg(1));
        let nx = b.p("list.len", vec![rt::u32t()], vec![x.clone()])?;
        let ny = b.p("list.len", vec![rt::u32t()], vec![y.clone()])?;
        let n = b.p("int.max", vec![Ty::Int(0, INT_MAX)], vec![nx.clone(), ny.clone()])?;
        let o0 = b.p("list.new", vec![rt::u32t()], vec![])?;
        let out = b.var(big(), o0);
        let carry = b.var(Ty::Int(0, 1), lit(0));
        b.for_n(n, |b, i| {
            let p = digit_or_zero(b, &x, &nx, &i)?;
            let q = digit_or_zero(b, &y, &ny, &i)?;
            let s = b.add_in(p, q, Ty::Int(0, 2 * (B32 - 1)))?;
            let cv = b.take(carry);
            let s = b.add_in(s, cv, Ty::Int(0, 2 * B32 - 1))?;
            let (hi, lo) = b.divmod_lit(s, B32, Ty::Int(0, 2 * B32 - 1))?;
            let hi = b.cast_in(hi, Ty::Int(0, 1))?;
            let lo = b.cast_in(lo, rt::u32t())?;
            let o = b.take(out);
            let o = b.p("list.push", vec![rt::u32t()], vec![o, lo])?;
            b.set(out, o);
            b.set(carry, hi);
            Ok(())
        })?;
        let cv = b.take(carry);
        let one = b.int_cmp("int.eq", cv.clone(), lit(1))?;
        b.when(one, |b| {
            let o = b.take(out);
            let c = b.cast_in(cv, rt::u32t())?;
            let o = b.p("list.push", vec![rt::u32t()], vec![o, c])?;
            b.set(out, o);
            Ok(())
        })?;
        b.drop(x)?;
        b.drop(y)?;
        let o = b.take(out);
        let r = call(&mut b, f, nm, vec![o]);
        b.ret(r);
        Ok(b.finish())
    })
}

/// `big.sub(a, b)`: a - b, with a >= b.
fn sub(f: &mut Fns) -> R<usize> {
    let nm = norm(f)?;
    f.get("rt.big.sub", vec![big(), big()], big(), move |f, mut b| {
        let (x, y) = (b.arg(0), b.arg(1));
        let nx = b.p("list.len", vec![rt::u32t()], vec![x.clone()])?;
        let ny = b.p("list.len", vec![rt::u32t()], vec![y.clone()])?;
        let o0 = b.p("list.new", vec![rt::u32t()], vec![])?;
        let out = b.var(big(), o0);
        let borrow = b.var(Ty::Int(0, 1), lit(0));
        b.for_n(nx, |b, i| {
            let p = b.list_at(x.clone(), i.clone())?;
            let q = digit_or_zero(b, &y, &ny, &i)?;
            // p + 2^32 - q - borrow, in 1 .. 2^33 - 1
            let s = b.add_in(p, lit(B32), Ty::Int(B32, 2 * B32 - 1))?;
            let s = b.sub_in(s, q, Ty::Int(1, 2 * B32 - 1))?;
            let bv = b.take(borrow);
            let s = b.sub_in(s, bv, Ty::Int(0, 2 * B32 - 1))?;
            let (hi, lo) = b.divmod_lit(s, B32, Ty::Int(0, 2 * B32 - 1))?;
            let lo = b.cast_in(lo, rt::u32t())?;
            let hi = b.cast_in(hi, Ty::Int(0, 1))?;
            let nb = b.sub_in(lit(1), hi, Ty::Int(0, 1))?;
            let o = b.take(out);
            let o = b.p("list.push", vec![rt::u32t()], vec![o, lo])?;
            b.set(out, o);
            b.set(borrow, nb);
            Ok(())
        })?;
        let bv = b.take(borrow);
        let neg = b.int_cmp("int.eq", bv, lit(1))?;
        b.guard_halt(neg, |fb| fb.trap_value(Atom::bytes(b"internal: negative difference")))?;
        b.drop(x)?;
        b.drop(y)?;
        let o = b.take(out);
        let r = call(&mut b, f, nm, vec![o]);
        b.ret(r);
        Ok(b.finish())
    })
}

/// `big.cmp(a, b)`: -1, 0 or 1 as a is below, equal to or above b.
fn cmp(f: &mut Fns) -> R<usize> {
    f.get("rt.big.cmp", vec![big(), big()], Ty::Int(-1, 1), |_, mut b| {
        let (x, y) = (b.arg(0), b.arg(1));
        let nx = b.p("list.len", vec![rt::u32t()], vec![x.clone()])?;
        let ny = b.p("list.len", vec![rt::u32t()], vec![y.clone()])?;
        let r = b.var(Ty::Int(-1, 1), lit(0));
        let shorter = b.int_cmp("int.lt", nx.clone(), ny.clone())?;
        let longer = b.int_cmp("int.lt", ny, nx.clone())?;
        let (x2, y2) = (x.clone(), y.clone());
        b.if_(shorter, |b| {
            b.set(r, lit(-1));
            Ok(())
        }, |b| b.if_(longer, |b| {
            b.set(r, lit(1));
            Ok(())
        }, |b| {
            // the same length: from the most significant digit down; i
            // falls by one per iteration from nx, so at most nx iterations
            let i = b.var(Ty::Int(0, INT_MAX), nx.clone());
            let cnt = b.counter();
            b.loop_(|b| {
                let iv = b.take(i);
                let z = b.int_cmp("int.eq", iv.clone(), lit(0))?;
                b.exit_if(z)?;
                b.count(cnt, nx.clone())?;
                let i1 = b.sub_in(iv, lit(1), Ty::Int(0, INT_MAX))?;
                b.set(i, i1.clone());
                let p = b.list_at(x2.clone(), i1.clone())?;
                let q = b.list_at(y2.clone(), i1)?;
                let lt = b.int_cmp("int.lt", p.clone(), q.clone())?;
                b.when(lt, |b| {
                    b.set(r, lit(-1));
                    b.brk(0);
                    Ok(())
                })?;
                let gt = b.int_cmp("int.lt", q, p)?;
                b.when(gt, |b| {
                    b.set(r, lit(1));
                    b.brk(0);
                    Ok(())
                })
            })
        }))?;
        b.drop(x)?;
        b.drop(y)?;
        let rv = b.take(r);
        b.ret(rv);
        Ok(b.finish())
    })
}

/// `big.shl(a, n)`: a times 2^n, n below 2^16.
fn shl(f: &mut Fns) -> R<usize> {
    let ms = mul_small(f)?;
    let nm = norm(f)?;
    f.get("rt.big.shl", vec![big(), Ty::Int(0, 65535)], big(), move |f, mut b| {
        let (a, n) = (b.arg(0), b.arg(1));
        let (q, r) = b.divmod_lit(n, 32, Ty::Int(0, 65535))?;
        // 2^r by doubling
        let p = b.var(rt::u32t(), lit(1));
        b.for_n(r, |b, _| {
            let pv = b.take(p);
            let p2 = b.mul_in(pv, lit(2), rt::u32t())?;
            b.set(p, p2);
            Ok(())
        })?;
        let pv = b.take(p);
        let a2 = call(&mut b, f, ms, vec![a, pv]);
        let o0 = b.p("list.new", vec![rt::u32t()], vec![])?;
        let out = b.var(big(), o0);
        b.for_n(q, |b, _| {
            let o = b.take(out);
            let o = b.p("list.push", vec![rt::u32t()], vec![o, lit(0)])?;
            b.set(out, o);
            Ok(())
        })?;
        let n2 = b.p("list.len", vec![rt::u32t()], vec![a2.clone()])?;
        b.for_n(n2, |b, i| {
            let d = b.list_at(a2.clone(), i)?;
            let o = b.take(out);
            let o = b.p("list.push", vec![rt::u32t()], vec![o, d])?;
            b.set(out, o);
            Ok(())
        })?;
        b.drop(a2)?;
        let o = b.take(out);
        let r = call(&mut b, f, nm, vec![o]);
        b.ret(r);
        Ok(b.finish())
    })
}

/// `big.pow10(a, k)`: a times 10^k, k below 2^16.
fn mul_pow10(f: &mut Fns) -> R<usize> {
    let ms = mul_small(f)?;
    f.get("rt.big.pow10", vec![big(), Ty::Int(0, 65535)], big(), move |f, mut b| {
        let x = b.var(big(), b.arg(0));
        let k = b.var(Ty::Int(0, 65535), b.arg(1));
        // k falls by 9 per iteration from the second argument and stays
        // not negative: fewer iterations than that argument
        let cnt = b.counter();
        b.loop_(|b| {
            let kv = b.take(k);
            let more = b.int_cmp("int.le", lit(9), kv.clone())?;
            b.exit_unless(more)?;
            b.count(cnt, b.arg(1))?;
            let k2 = b.sub_in(kv, lit(9), Ty::Int(0, 65535))?;
            b.set(k, k2);
            let xv = b.take(x);
            let x2 = call(b, f, ms, vec![xv, lit(1_000_000_000)]);
            b.set(x, x2);
            Ok(())
        })?;
        let kv = b.take(k);
        let p = b.var(rt::u32t(), lit(1));
        b.for_n(kv, |b, _| {
            let pv = b.take(p);
            let p2 = b.mul_in(pv, lit(10), rt::u32t())?;
            b.set(p, p2);
            Ok(())
        })?;
        let pv = b.take(p);
        let xv = b.take(x);
        let r = call(&mut b, f, ms, vec![xv, pv]);
        b.ret(r);
        Ok(b.finish())
    })
}

/// `big.divmod(a, d)`: the quotient of a by d and the remainder, d in 1 .. 2^31.
fn divmod_small(f: &mut Fns) -> R<usize> {
    let nm = norm(f)?;
    let dt = Ty::Int(1, 1 << 31);
    let rt_ = Ty::Tup(vec![big(), Ty::Int(0, (1 << 31) - 1)]);
    f.get("rt.big.divmod", vec![big(), dt], rt_.clone(), move |f, mut b| {
        let (a, d) = (b.arg(0), b.arg(1));
        let n = b.p("list.len", vec![rt::u32t()], vec![a.clone()])?;
        let q0 = b.p("list.new", vec![rt::u32t()], vec![])?;
        let q = b.var(big(), q0);
        b.for_n(n.clone(), |b, _| {
            let o = b.take(q);
            let o = b.p("list.push", vec![rt::u32t()], vec![o, lit(0)])?;
            b.set(q, o);
            Ok(())
        })?;
        let rem = b.var(Ty::Int(0, (1 << 31) - 1), lit(0));
        // i falls by one per iteration from n: at most n iterations
        let i = b.var(Ty::Int(0, INT_MAX), n.clone());
        let cnt = b.counter();
        b.loop_(|b| {
            let iv = b.take(i);
            let z = b.int_cmp("int.eq", iv.clone(), lit(0))?;
            b.exit_if(z)?;
            b.count(cnt, n.clone())?;
            let i1 = b.sub_in(iv, lit(1), Ty::Int(0, INT_MAX))?;
            b.set(i, i1.clone());
            let x = b.list_at(a.clone(), i1.clone())?;
            let rv = b.take(rem);
            let hi = b.mul_in(rv, lit(B32), Ty::Int(0, ((1 << 31) - 1) * B32))?;
            let cur = b.add_in(hi, x, Ty::Int(0, (1 << 31) * B32 - 1))?;
            let z = b.int_cmp("int.eq", d.clone(), lit(0))?;
            b.guard_halt(z, |fb| fb.trap_value(Atom::bytes(b"internal: divisor")))?;
            let bq = b.p("int.bad.quo", vec![Ty::u64()], vec![cur.clone(), d.clone()])?;
            b.guard_halt(bq, |fb| fb.trap_value(Atom::bytes(b"internal: quotient")))?;
            let qd = b.p("int.quo", vec![Ty::u64()], vec![cur.clone(), d.clone()])?;
            let rd = b.p("int.rem", vec![Ty::u64()], vec![cur, d.clone()])?;
            let qd = b.cast_in(qd, rt::u32t())?;
            let rd = b.cast_in(rd, Ty::Int(0, (1 << 31) - 1))?;
            b.set(rem, rd);
            let qv = b.take(q);
            let ql = b.p("list.len", vec![rt::u32t()], vec![qv.clone()])?;
            let ok = b.int_cmp("int.lt", i1.clone(), ql)?;
            b.guard_halt_unless(ok, "internal: quotient digit")?;
            let qv = b.p("list.set", vec![rt::u32t()], vec![qv, i1, qd])?;
            b.set(q, qv);
            Ok(())
        })?;
        b.drop(a)?;
        let qv = b.take(q);
        let qn = call(&mut b, f, nm, vec![qv]);
        let rv = b.take(rem);
        let r = b.p("tup.mk", vec![rt_.clone()], vec![qn, rv])?;
        b.ret(r);
        Ok(b.finish())
    })
}

/// `big.dec(a)`: the decimal text of a.
fn dec(f: &mut Fns) -> R<usize> {
    let dm = divmod_small(f)?;
    let ds = rt::dec_sm(f)?;
    f.get("rt.big.dec", vec![big()], Ty::Bytes, move |f, mut b| {
        // each iteration takes nine decimal digits off a; a number of n
        // base-2^32 digits is below 2^(32n) < 10^(10n), so it has at most
        // 10n decimal digits, at most 2n groups of nine when n >= 1, and 0
        // has one group: at most 2n + 1 iterations
        let n0 = b.p("list.len", vec![rt::u32t()], vec![b.arg(0)])?;
        let n2 = b.mul_in(n0, lit(2), count_t())?;
        let most = b.add_in(n2, lit(1), count_t())?;
        let x = b.var(big(), b.arg(0));
        let acc = b.var(Ty::Bytes, Atom::bytes(b""));
        let cnt = b.counter();
        b.loop_(|b| {
            b.count(cnt, most.clone())?;
            let xv = b.take(x);
            let qr = call(b, f, dm, vec![xv, lit(1_000_000_000)]);
            let qrt = b.ty(&qr);
            let parts = b.prim("tup.open", vec![qrt], vec![qr])?;
            let (q, r) = (parts[0].clone(), parts[1].clone());
            let r = b.cast_in(r, Ty::u64())?;
            let s = call(b, f, ds, vec![Atom::bool(false), r]);
            let qn = b.p("list.len", vec![rt::u32t()], vec![q.clone()])?;
            let last = b.int_cmp("int.eq", qn, lit(0))?;
            let sv = b.var(Ty::Bytes, s);
            let more = b.not(last.clone())?;
            b.when(more, |b| {
                // a digit group below the top one has nine digits; each
                // iteration adds one, so at most nine iterations
                let cnt = b.counter();
                b.loop_(|b| {
                    let s = b.take(sv);
                    let n = b.p("bytes.len", vec![], vec![s.clone()])?;
                    let short = b.int_cmp("int.lt", n, lit(9))?;
                    b.set(sv, s.clone());
                    b.exit_unless(short)?;
                    b.count(cnt, lit(9))?;
                    let s = b.take(sv);
                    let s2 = b.cat(Atom::bytes(b"0"), s)?;
                    b.set(sv, s2);
                    Ok(())
                })
            })?;
            let s = b.take(sv);
            let a = b.take(acc);
            let a2 = b.cat(s, a)?;
            b.set(acc, a2);
            b.set(x, q);
            b.when(last, |b| {
                b.brk(0);
                Ok(())
            })
        })?;
        let xv = b.take(x);
        b.drop(xv)?;
        let a = b.take(acc);
        b.ret(a);
        Ok(b.finish())
    })
}

// ---- the parts of a float

/// The precision and least exponent of a binary format: binary64 values
/// are m * 2^e with m below 2^53 and e at least -1074; binary32 values,
/// held as binary64, with m below 2^24 and e at least -149.
#[derive(Clone, Copy)]
pub struct Fmt {
    pub p: u32,
    pub emin: i128,
    pub name: &'static str,
}

pub const F64: Fmt = Fmt { p: 53, emin: -1074, name: "f64" };
pub const F32: Fmt = Fmt { p: 24, emin: -149, name: "f32" };

fn exp_t() -> Ty {
    Ty::Int(-1200, 1200)
}

/// `parts(a)`: for a finite positive a of the format, the integers m and e
/// with a = m * 2^e, m below 2^p, and m at least 2^(p-1) unless e is the
/// least exponent.
fn parts(f: &mut Fns, fm: Fmt) -> R<usize> {
    let mt = Ty::Int(0, (1i128 << fm.p) - 1);
    let rt_ = Ty::Tup(vec![mt.clone(), exp_t()]);
    f.get(&format!("rt.{}.parts", fm.name), vec![Ty::F64], rt_.clone(), move |_, mut b| {
        let p = fm.p as i32;
        let x = b.var(Ty::F64, b.arg(0));
        let e = b.var(exp_t(), lit(0));
        let pow = |k: i32| f64k(2f64.powi(k));
        // x is a positive binary64 value, so its binary exponent lies in
        // -1074 ..= 1023, a range of 2098, and each loop below moves it by
        // `step` in one direction: at most 2098 / step + 1 iterations each
        let most = |step: i128| lit(2098 / step + 1);
        // down by 2^64 while the result keeps p bits before the point, then by 2
        for (step, k) in [(64, p + 64), (1, p)] {
            let cnt = b.counter();
            b.loop_(|b| {
                let xv = b.take(x);
                let big = b.p("f64.le", vec![], vec![pow(k), xv.clone()])?;
                b.set(x, xv.clone());
                b.exit_unless(big)?;
                b.count(cnt, most(step as i128))?;
                let x2 = fop(b, "mul", xv, pow(-step))?;
                b.set(x, x2);
                let ev = b.take(e);
                let e2 = b.add_in(ev, lit(step as i128), exp_t())?;
                b.set(e, e2);
                Ok(())
            })?;
        }
        // up by 2^64, then by 2, while below 2^(p-1) and above the least exponent
        for step in [64, 1] {
            let cnt = b.counter();
            b.loop_(|b| {
                let xv = b.take(x);
                let small = b.p("f64.lt", vec![], vec![xv.clone(), pow(p - 1 - if step == 64 { 64 } else { 0 })])?;
                b.set(x, xv.clone());
                b.exit_unless(small)?;
                let ev = b.take(e);
                let room = b.int_cmp("int.le", lit(fm.emin + step as i128), ev.clone())?;
                b.set(e, ev.clone());
                b.exit_unless(room)?;
                b.count(cnt, most(step as i128))?;
                let x2 = fop(b, "mul", xv, pow(step))?;
                b.set(x, x2);
                let e2 = b.sub_in(ev, lit(step as i128), exp_t())?;
                b.set(e, e2);
                Ok(())
            })?;
        }
        let xv = b.take(x);
        let c = b.p("f64.bad.trunc", vec![mt.clone()], vec![xv.clone()])?;
        b.guard_halt(c, |fb| fb.trap_value(Atom::bytes(b"internal: float parts")))?;
        let m = b.p("f64.trunc", vec![mt.clone()], vec![xv])?;
        let ev = b.take(e);
        let r = b.p("tup.mk", vec![rt_.clone()], vec![m, ev])?;
        b.ret(r);
        Ok(b.finish())
    })
}


/// How the digits are chosen when more than one decimal of the shortest
/// length reads back to the float.
#[derive(Clone, Copy, PartialEq)]
pub enum Pick {
    /// interp `shortest_e`: the closest decimal of that length, the even one
    /// on an exact tie, when it reads back; otherwise the one that does
    Closest,
    /// Rust's shortest form (the display of f32): Dragon's rounding, which
    /// rounds up on a tie
    Rust,
}

fn kt() -> Ty {
    Ty::Int(-400, 400)
}

fn u8list() -> Ty {
    Ty::list(Ty::u8())
}

/// `(incl and x) or (not incl and y)`.
fn sel(b: &mut FB, incl: &Atom, x: Atom, y: Atom) -> R<Atom> {
    let a = b.p("bool.and", vec![], vec![incl.clone(), x])?;
    let ni = b.not(incl.clone())?;
    let c = b.p("bool.and", vec![], vec![ni, y])?;
    b.p("bool.or", vec![], vec![a, c])
}

/// Add one to the last digit of the list `ds` (ASCII digits), carrying.
/// When every digit was 9, the digits become 1 followed by zeros and k
/// grows by one; with `extend` (Rust's `round_up`) a zero is also appended.
fn round_up(b: &mut FB, ds: u32, k: u32, extend: bool) -> R<()> {
    let n = b.p("list.len", vec![Ty::u8()], vec![Atom::V(ds)])?;
    // i falls by one per iteration from n: at most n iterations
    let i = b.var(Ty::Int(0, INT_MAX), n.clone());
    let done = b.var(Ty::Bool, Atom::bool(false));
    let cnt = b.counter();
    b.loop_(|b| {
        let iv = b.take(i);
        let z = b.int_cmp("int.eq", iv.clone(), lit(0))?;
        b.exit_if(z)?;
        b.count(cnt, n.clone())?;
        let i1 = b.sub_in(iv, lit(1), Ty::Int(0, INT_MAX))?;
        b.set(i, i1.clone());
        let d = b.list_at(Atom::V(ds), i1.clone())?;
        let nine = b.int_cmp("int.eq", d.clone(), lit(b'9' as i128))?;
        let i2 = i1.clone();
        b.if_(nine, |b| {
            let l = b.take(ds);
            let l = b.p("list.set", vec![Ty::u8()], vec![l, i1, lit(b'0' as i128)])?;
            b.set(ds, l);
            Ok(())
        }, |b| {
            let d1 = b.add_in(d, lit(1), Ty::u8())?;
            let l = b.take(ds);
            let l = b.p("list.set", vec![Ty::u8()], vec![l, i2, d1])?;
            b.set(ds, l);
            b.set(done, Atom::bool(true));
            b.brk(0);
            Ok(())
        })
    })?;
    let dv = b.take(done);
    b.if_(dv, |_| Ok(()), |b| {
        let n = b.p("list.len", vec![Ty::u8()], vec![Atom::V(ds)])?;
        let ok = b.int_cmp("int.lt", lit(0), n)?;
        b.guard_halt_unless(ok, "internal: no digits")?;
        let l = b.take(ds);
        let l = b.p("list.set", vec![Ty::u8()], vec![l, lit(0), lit(b'1' as i128)])?;
        let l = if extend { b.p("list.push", vec![Ty::u8()], vec![l, lit(b'0' as i128)])? } else { l };
        b.set(ds, l);
        let kv = b.take(k);
        let k1 = b.add_in(kv, lit(1), kt())?;
        b.set(k, k1);
        Ok(())
    })
}

/// `digits(m, e)`: the shortest decimal digits of m * 2^e (m > 0), as ASCII
/// digits, and the exponent k with value 0.d1d2… * 10^k.
fn digits(f: &mut Fns, fm: Fmt, pick: Pick) -> R<usize> {
    let mt = Ty::Int(0, (1i128 << fm.p) - 1);
    let rt_ = Ty::Tup(vec![Ty::Bytes, kt()]);
    let ofu = of_u64(f)?;
    let addf = add(f)?;
    let subf = sub(f)?;
    let cmpf = cmp(f)?;
    let shlf = shl(f)?;
    let p10 = mul_pow10(f)?;
    let ms = mul_small(f)?;
    let name = format!("rt.{}.digits.{}", fm.name, if pick == Pick::Closest { "closest" } else { "rust" });
    f.get(&name, vec![mt, exp_t()], rt_.clone(), move |f, mut b| {
        let (m, e) = (b.arg(0), b.arg(1));
        let half = 1i128 << (fm.p - 1);
        let (_, mr) = b.divmod_lit(m.clone(), 2, Ty::u64())?;
        // a decimal on the boundary of the interval reads back to this float
        // when its mantissa is even (round to nearest, ties to even)
        let incl = b.int_cmp("int.eq", mr, lit(0))?;
        let normal = b.int_cmp("int.le", lit(half), m.clone())?;
        let at_half = b.int_cmp("int.eq", m.clone(), lit(half))?;
        // Rust's `decode` takes every normal value whose mantissa is 2^(p-1)
        // to have a lower neighbour half as far as the upper one; that holds
        // except at the least normal value, whose two neighbours are as far
        let rust_edge = b.p("bool.and", vec![], vec![normal, at_half])?;
        let above = b.int_cmp("int.lt", lit(fm.emin), e.clone())?;
        let true_edge = b.p("bool.and", vec![], vec![rust_edge.clone(), above])?;
        let mant = b.var(Ty::u64(), lit(0));
        let plus = b.var(Ty::Int(1, 2), lit(1));
        let minus_true = b.var(Ty::Int(1, 2), lit(1));
        let ex = b.var(exp_t(), lit(0));
        let (m2, e2) = (m.clone(), e.clone());
        b.if_(rust_edge, |b| {
            let v = b.mul_in(m.clone(), lit(4), Ty::u64())?;
            b.set(mant, v);
            b.set(plus, lit(2));
            b.if_(true_edge, |_| Ok(()), |b| {
                b.set(minus_true, lit(2));
                Ok(())
            })?;
            let x = b.sub_in(e.clone(), lit(2), exp_t())?;
            b.set(ex, x);
            Ok(())
        }, |b| {
            let v = b.mul_in(m2, lit(2), Ty::u64())?;
            b.set(mant, v);
            let x = b.sub_in(e2, lit(1), exp_t())?;
            b.set(ex, x);
            Ok(())
        })?;
        let mant_v = b.take(mant);
        let plus_v = b.take(plus);
        let mtv = b.take(minus_true);
        let exv = b.take(ex);
        // v = r/s; v - low = mm/s and high - v = mp/s as Rust takes them; the
        // true distance to the lower boundary is mtr/s
        let r0 = call(&mut b, f, ofu, vec![mant_v.clone()]);
        let mm0 = call(&mut b, f, ofu, vec![lit(1)]);
        let mt0 = call(&mut b, f, ofu, vec![mtv]);
        let mp0 = call(&mut b, f, ofu, vec![plus_v.clone()]);
        let s0 = call(&mut b, f, ofu, vec![lit(1)]);
        let r = b.var(big(), r0);
        let mm = b.var(big(), mm0);
        let mtr = b.var(big(), mt0);
        let mp = b.var(big(), mp0);
        let s = b.var(big(), s0);
        let nonneg = b.int_cmp("int.le", lit(0), exv.clone())?;
        let exa = exv.clone();
        let exb = exv.clone();
        b.if_(nonneg, |b| {
            let sh = b.cast_in(exa, Ty::Int(0, 65535))?;
            for v in [r, mm, mtr, mp] {
                let x = b.take(v);
                let y = call(b, f, shlf, vec![x, sh.clone()]);
                b.set(v, y);
            }
            Ok(())
        }, |b| {
            let sh = b.sub_in(lit(0), exb, Ty::Int(0, 65535))?;
            let x = b.take(s);
            let y = call(b, f, shlf, vec![x, sh]);
            b.set(s, y);
            Ok(())
        })?;
        // an estimate of k from the bits of mant + plus (Rust's
        // `estimate_scaling_factor`), corrected below in both directions
        let h0 = b.add_in(mant_v, plus_v, Ty::u64())?;
        let h1 = b.sub_in(h0, lit(1), Ty::u64())?;
        let t = b.var(Ty::u64(), h1);
        let nb = b.var(Ty::Int(0, 64), lit(0));
        // t is below 2^64: at most 64 halvings bring it to 0
        let cnt = b.counter();
        b.loop_(|b| {
            let tv = b.take(t);
            let z = b.int_cmp("int.eq", tv.clone(), lit(0))?;
            b.exit_if(z)?;
            b.count(cnt, lit(64))?;
            let (q, _) = b.divmod_lit(tv, 2, Ty::u64())?;
            b.set(t, q);
            let nv = b.take(nb);
            let n1 = b.add_in(nv, lit(1), Ty::Int(0, 64))?;
            b.set(nb, n1);
            Ok(())
        })?;
        let nbv = b.take(nb);
        let sum = b.add_in(nbv, exv, Ty::Int(-1300, 1300))?;
        let wt = Ty::Int(-1300 * 1292913986, 1300 * 1292913986);
        let prod = b.mul_in(sum, lit(1292913986), wt.clone())?;
        let (q, rm) = b.divmod_lit(prod, B32, wt)?;
        let below = b.int_cmp("int.lt", rm, lit(0))?;
        let k = b.var(kt(), lit(0));
        let q1 = q.clone();
        b.if_(below, |b| {
            let q2 = b.sub_in(q1, lit(1), kt())?;
            b.set(k, q2);
            Ok(())
        }, |b| {
            let q2 = b.cast_in(q, kt())?;
            b.set(k, q2);
            Ok(())
        })?;
        // scale so that r/s = v / 10^(k-1)
        let kv = b.peek(k)?;
        let pos = b.int_cmp("int.le", lit(1), kv.clone())?;
        let (ka, kb) = (kv.clone(), kv);
        b.if_(pos, |b| {
            let n = b.sub_in(ka, lit(1), Ty::Int(0, 65535))?;
            let x = b.take(s);
            let y = call(b, f, p10, vec![x, n]);
            b.set(s, y);
            Ok(())
        }, |b| {
            let n = b.sub_in(lit(1), kb, Ty::Int(0, 65535))?;
            for v in [r, mm, mtr, mp] {
                let x = b.take(v);
                let y = call(b, f, p10, vec![x, n.clone()]);
                b.set(v, y);
            }
            Ok(())
        })?;
        // k is the least integer with high < 10^k (high <= 10^k when the
        // boundary does not read back)
        let high_vs = |b: &mut FB, f: &Fns, times10: bool| -> R<Atom> {
            let rv = b.peek(r)?;
            let mpv = b.peek(mp)?;
            let h = call(b, f, addf, vec![rv, mpv]);
            let sv = b.peek(s)?;
            let sv = if times10 { call(b, f, ms, vec![sv, lit(10)]) } else { sv };
            Ok(call(b, f, cmpf, vec![h, sv]))
        };
        // k lies in -400 ..= 400 (its type) and each iteration of the next
        // two loops moves it by one: at most 801 iterations each
        let cnt = b.counter();
        b.loop_(|b| {
            let c = high_vs(b, f, true)?;
            let ge = b.int_cmp("int.le", lit(0), c.clone())?;
            let gt = b.int_cmp("int.lt", lit(0), c)?;
            let reach = sel(b, &incl, ge, gt)?;
            b.exit_unless(reach)?;
            b.count(cnt, lit(801))?;
            let x = b.take(s);
            let y = call(b, f, ms, vec![x, lit(10)]);
            b.set(s, y);
            let kv = b.take(k);
            let k1 = b.add_in(kv, lit(1), kt())?;
            b.set(k, k1);
            Ok(())
        })?;
        let cnt = b.counter();
        b.loop_(|b| {
            let c = high_vs(b, f, false)?;
            let lt = b.int_cmp("int.lt", c.clone(), lit(0))?;
            let le = b.int_cmp("int.le", c, lit(0))?;
            let low = sel(b, &incl, lt, le)?;
            b.exit_unless(low)?;
            b.count(cnt, lit(801))?;
            for v in [r, mm, mtr, mp] {
                let x = b.take(v);
                let y = call(b, f, ms, vec![x, lit(10)]);
                b.set(v, y);
            }
            let kv = b.take(k);
            let k1 = b.sub_in(kv, lit(1), kt())?;
            b.set(k, k1);
            Ok(())
        })?;
        // the digits, until the digits so far or the next decimal up read back
        let d0 = b.p("list.new", vec![Ty::u8()], vec![])?;
        let ds = b.var(u8list(), d0);
        let down = b.var(Ty::Bool, Atom::bool(false));
        let up = b.var(Ty::Bool, Atom::bool(false));
        // iteration j writes the digit of v at 10^(k - j). v = m * 2^e with
        // e >= emin is a multiple of 10^emin (2^emin is 5^-emin / 10^-emin),
        // so once the digits reach 10^emin nothing is left (r is 0, below
        // mm) and the loop stops: at most k - emin iterations, and k <= 400
        let cnt = b.counter();
        b.loop_(|b| {
            b.count(cnt, lit(400 - fm.emin))?;
            let d = b.var(Ty::Int(0, 9), lit(0));
            // d lies in 0 ..= 9 (its type) and each iteration raises it by one
            let cnt_d = b.counter();
            b.loop_(|b| {
                let rv = b.peek(r)?;
                let sv = b.peek(s)?;
                let c = call(b, f, cmpf, vec![rv, sv]);
                let lt = b.int_cmp("int.lt", c, lit(0))?;
                b.exit_if(lt)?;
                b.count(cnt_d, lit(10))?;
                let rv = b.take(r);
                let sv = b.peek(s)?;
                let r2 = call(b, f, subf, vec![rv, sv]);
                b.set(r, r2);
                let dv = b.take(d);
                let d1 = b.add_in(dv, lit(1), Ty::Int(0, 9))?;
                b.set(d, d1);
                Ok(())
            })?;
            let dv = b.take(d);
            let ch = b.add_in(dv, lit(b'0' as i128), Ty::u8())?;
            let l = b.take(ds);
            let l = b.p("list.push", vec![Ty::u8()], vec![l, ch])?;
            b.set(ds, l);
            let rv = b.peek(r)?;
            let mv = b.peek(mm)?;
            let c1 = call(b, f, cmpf, vec![rv, mv]);
            let le = b.int_cmp("int.le", c1.clone(), lit(0))?;
            let lt = b.int_cmp("int.lt", c1, lit(0))?;
            let dn = sel(b, &incl, le, lt)?;
            let sv = b.peek(s)?;
            let rv = b.peek(r)?;
            let mpv = b.peek(mp)?;
            let h = call(b, f, addf, vec![rv, mpv]);
            let c2 = call(b, f, cmpf, vec![sv, h]);
            let le = b.int_cmp("int.le", c2.clone(), lit(0))?;
            let lt = b.int_cmp("int.lt", c2, lit(0))?;
            let u = sel(b, &incl, le, lt)?;
            b.set(down, dn.clone());
            b.set(up, u.clone());
            let stop = b.p("bool.or", vec![], vec![dn, u])?;
            b.exit_if(stop)?;
            for v in [r, mm, mtr, mp] {
                let x = b.take(v);
                let y = call(b, f, ms, vec![x, lit(10)]);
                b.set(v, y);
            }
            Ok(())
        })?;
        // 2r against s: where v lies between the digits and the next decimal up
        let rv = b.peek(r)?;
        let r2 = call(&mut b, f, ms, vec![rv, lit(2)]);
        let sv = b.peek(s)?;
        let c3 = call(&mut b, f, cmpf, vec![r2, sv]);
        let upv = b.take(up);
        let downv = b.take(down);
        match pick {
            Pick::Rust => {
                // `up && (!down || 2r >= s)`
                let ge = b.int_cmp("int.le", lit(0), c3)?;
                let nd = b.not(downv)?;
                let either = b.p("bool.or", vec![], vec![nd, ge])?;
                let go = b.p("bool.and", vec![], vec![upv, either])?;
                b.when(go, |b| round_up(b, ds, k, true))?;
            }
            Pick::Closest => {
                let first = b.list_at(Atom::V(ds), lit(0))?;
                let zero = b.int_cmp("int.eq", first, lit(b'0' as i128))?;
                let c3b = c3.clone();
                b.if_(zero, |b| {
                    // the first digit is 0 only when the next power of ten
                    // reads back; Rust then writes 1, and so does the closest form
                    let l = b.take(ds);
                    let l = b.p("list.set", vec![Ty::u8()], vec![l, lit(0), lit(b'1' as i128)])?;
                    b.set(ds, l);
                    Ok(())
                }, |b| {
                    let tie = b.int_cmp("int.eq", c3b.clone(), lit(0))?;
                    let ceil = b.var(Ty::Bool, Atom::bool(false));
                    b.if_(tie, |b| {
                        // an exact tie: the even last digit
                        let n = b.p("list.len", vec![Ty::u8()], vec![Atom::V(ds)])?;
                        let n1 = b.sub_in(n, lit(1), Ty::Int(0, INT_MAX))?;
                        let last = b.list_at(Atom::V(ds), n1)?;
                        let (_, odd) = b.divmod_lit(last, 2, Ty::u8())?;
                        let o = b.int_cmp("int.eq", odd, lit(1))?;
                        b.set(ceil, o);
                        Ok(())
                    }, |b| {
                        let near_up = b.int_cmp("int.lt", lit(0), c3b)?;
                        let rv = b.peek(r)?;
                        let mv = b.peek(mtr)?;
                        let c4 = call(b, f, cmpf, vec![rv, mv]);
                        let le = b.int_cmp("int.le", c4.clone(), lit(0))?;
                        let lt = b.int_cmp("int.lt", c4, lit(0))?;
                        let floor_ok = sel(b, &incl, le, lt)?;
                        // the closest when it reads back, else the other
                        let a = b.p("bool.and", vec![], vec![near_up.clone(), upv])?;
                        let nu = b.not(near_up)?;
                        let nf = b.not(floor_ok)?;
                        let c = b.p("bool.and", vec![], vec![nu, nf])?;
                        let go = b.p("bool.or", vec![], vec![a, c])?;
                        b.set(ceil, go);
                        Ok(())
                    })?;
                    let cv = b.take(ceil);
                    b.when(cv, |b| round_up(b, ds, k, false))
                })?;
                let _ = downv;
            }
        }
        for v in [r, mm, mtr, mp, s] {
            let x = b.take(v);
            b.drop(x)?;
        }
        let l = b.take(ds);
        let t = b.p("bytes.of_list", vec![], vec![l])?;
        let kv = b.take(k);
        let out = b.p("tup.mk", vec![rt_.clone()], vec![t, kv])?;
        b.ret(out);
        Ok(b.finish())
    })
}

// ---- texts

/// `n` zeros, n not negative.
fn zeros(b: &mut FB, n: Atom) -> R<Atom> {
    let acc = b.var(Ty::Bytes, Atom::bytes(b""));
    b.for_n(n, |b, _| {
        let a = b.take(acc);
        let a2 = b.cat(a, Atom::bytes(b"0"))?;
        b.set(acc, a2);
        Ok(())
    })?;
    Ok(b.take(acc))
}

/// `t[i..j]` with its domain tests; i is not negative by its type.
fn slice(b: &mut FB, t: Atom, i: Atom, j: Atom) -> R<Atom> {
    let n = b.p("bytes.len", vec![], vec![t.clone()])?;
    let c1 = b.int_cmp("int.le", i.clone(), j.clone())?;
    b.guard_halt_unless(c1, "internal: slice")?;
    let c2 = b.int_cmp("int.le", j.clone(), n)?;
    b.guard_halt_unless(c2, "internal: slice")?;
    b.p("bytes.slice", vec![], vec![t, i, j])
}

/// The sign, the digits and the exponent k (value 0.d1d2… * 10^k) of a
/// finite nonzero float.
fn sign_digits(b: &mut FB, f: &Fns, x: Atom, pf: usize, df: usize) -> R<(Atom, Atom, Atom)> {
    let neg = b.p("f64.lt", vec![], vec![x.clone(), f64k(0.0)])?;
    let a = b.p("f64.abs", vec![], vec![x])?;
    let me = call(b, f, pf, vec![a]);
    let met = b.ty(&me);
    let ps = b.prim("tup.open", vec![met], vec![me])?;
    let dk = call(b, f, df, vec![ps[0].clone(), ps[1].clone()]);
    let dkt = b.ty(&dk);
    let ps = b.prim("tup.open", vec![dkt], vec![dk])?;
    Ok((neg, ps[0].clone(), ps[1].clone()))
}

/// The digits placed around the point: `0.000ddd`, `ddd000` or `dd.dd`
/// (Rust's `digits_to_dec_str`), with `.0` after an integer when `point`.
fn positional(b: &mut FB, ds: Atom, k: Atom, point: bool) -> R<Atom> {
    let n = b.p("bytes.len", vec![], vec![ds.clone()])?;
    let out = b.var(Ty::Bytes, Atom::bytes(b""));
    let le0 = b.int_cmp("int.le", k.clone(), lit(0))?;
    let (ds1, ds2, k1, k2, n2) = (ds.clone(), ds.clone(), k.clone(), k.clone(), n.clone());
    b.if_(le0, |b| {
        let z = b.sub_in(lit(0), k1, Ty::Int(0, 400))?;
        let zs = zeros(b, z)?;
        let s = b.cats(vec![Atom::bytes(b"0."), zs, ds1])?;
        b.set(out, s);
        Ok(())
    }, |b| {
        let k2 = b.cast_in(k2, Ty::Int(1, 400))?;
        let inside = b.int_cmp("int.lt", k2.clone(), n2.clone())?;
        let (k3, ds3, n3) = (k2.clone(), ds2.clone(), n2.clone());
        b.if_(inside, move |b| {
            let head = slice(b, ds3.clone(), lit(0), k3.clone())?;
            let tail = slice(b, ds3, k3, n3)?;
            let s = b.cats(vec![head, Atom::bytes(b"."), tail])?;
            b.set(out, s);
            Ok(())
        }, |b| {
            let z = b.sub_in(k2, n2, Ty::Int(0, 400))?;
            let zs = zeros(b, z)?;
            let mut parts = vec![ds2, zs];
            if point {
                parts.push(Atom::bytes(b".0"));
            }
            let s = b.cats(parts)?;
            b.set(out, s);
            Ok(())
        })
    })?;
    Ok(b.take(out))
}

/// `d.ddd` from the digits: the first digit, then the point and the rest
/// when there is a rest.
fn mantissa(b: &mut FB, ds: Atom) -> R<Atom> {
    let n = b.p("bytes.len", vec![], vec![ds.clone()])?;
    let out = b.var(Ty::Bytes, ds.clone());
    let more = b.int_cmp("int.lt", lit(1), n.clone())?;
    b.when(more, |b| {
        let head = slice(b, ds.clone(), lit(0), lit(1))?;
        let tail = slice(b, ds, lit(1), n)?;
        let s = b.cats(vec![head, Atom::bytes(b"."), tail])?;
        b.set(out, s);
        Ok(())
    })?;
    Ok(b.take(out))
}

/// The display of an f64 value (interp `float_text`, 6.3 N3 and F9): the
/// shortest digits that read back, positional for 1e-5 <= |x| < 1e16 and
/// with an exponent otherwise (`1e+16`, `1e-7`); -0.0 is `-0`.
pub fn text(f: &mut Fns) -> R<usize> {
    let pf = parts(f, F64)?;
    let df = digits(f, F64, Pick::Closest)?;
    let dec = rt::dec(f)?;
    f.get("rt.f64.text", vec![Ty::F64], Ty::Bytes, move |f, mut b| {
        let x = b.arg(0);
        zero_text(&mut b, &x, b"-0", b"0")?;
        let (neg, ds, k) = sign_digits(&mut b, f, x.clone(), pf, df)?;
        let a = b.p("f64.abs", vec![], vec![x])?;
        let lo = b.p("f64.le", vec![], vec![f64k(1e-5), a.clone()])?;
        let hi = b.p("f64.lt", vec![], vec![a, f64k(1e16)])?;
        let inr = b.p("bool.and", vec![], vec![lo, hi])?;
        let body = b.var(Ty::Bytes, Atom::bytes(b""));
        let (ds2, k2) = (ds.clone(), k.clone());
        b.if_(inr, |b| {
            let s = positional(b, ds, k, false)?;
            b.set(body, s);
            Ok(())
        }, |b| {
            let m = mantissa(b, ds2)?;
            let ex = b.sub_in(k2, lit(1), Ty::Int(-401, 399))?;
            let nn = b.int_cmp("int.le", lit(0), ex.clone())?;
            let sep = b.var(Ty::Bytes, Atom::bytes(b"e"));
            b.when(nn, |b| {
                b.set(sep, Atom::bytes(b"e+"));
                Ok(())
            })?;
            let sv = b.take(sep);
            let d = call(b, f, dec, vec![ex]);
            let s = b.cats(vec![m, sv, d])?;
            b.set(body, s);
            Ok(())
        })?;
        let bv = b.take(body);
        signed(&mut b, neg, bv)?;
        Ok(b.finish())
    })
}

/// The canonical JSON form of a float (interp `float_json`, 6.1.1): the
/// shortest digits, positional with at least one digit after the point for
/// 1e-4 <= |x| < 1e16, `d.ddde-N` and `d.dddeN` otherwise; -0.0 is `-0.0`.
pub fn json(f: &mut Fns) -> R<usize> {
    point_or_exp(f, "rt.f64.json", Pick::Closest)
}

/// Rust's `{:?}` of an f64 (interp `json_size` measures a float by it): the
/// form of [`json`] with Rust's shortest digits, which round up on a tie.
pub fn debug(f: &mut Fns) -> R<usize> {
    point_or_exp(f, "rt.f64.debug", Pick::Rust)
}

/// Positional with at least one digit after the point for 1e-4 <= |x| <
/// 1e16 and for zeros, `d.ddde-N` and `d.dddeN` otherwise.
fn point_or_exp(f: &mut Fns, name: &str, pick: Pick) -> R<usize> {
    let pf = parts(f, F64)?;
    let df = digits(f, F64, pick)?;
    let dec = rt::dec(f)?;
    f.get(name, vec![Ty::F64], Ty::Bytes, move |f, mut b| {
        let x = b.arg(0);
        zero_text(&mut b, &x, b"-0.0", b"0.0")?;
        let (neg, ds, k) = sign_digits(&mut b, f, x.clone(), pf, df)?;
        let a = b.p("f64.abs", vec![], vec![x])?;
        let lo = b.p("f64.le", vec![], vec![f64k(1e-4), a.clone()])?;
        let hi = b.p("f64.lt", vec![], vec![a, f64k(1e16)])?;
        let inr = b.p("bool.and", vec![], vec![lo, hi])?;
        let body = b.var(Ty::Bytes, Atom::bytes(b""));
        let (ds2, k2) = (ds.clone(), k.clone());
        b.if_(inr, |b| {
            let s = positional(b, ds, k, true)?;
            b.set(body, s);
            Ok(())
        }, |b| {
            let m = mantissa(b, ds2)?;
            let ex = b.sub_in(k2, lit(1), Ty::Int(-401, 399))?;
            let d = call(b, f, dec, vec![ex]);
            let s = b.cats(vec![m, Atom::bytes(b"e"), d])?;
            b.set(body, s);
            Ok(())
        })?;
        let bv = b.take(body);
        signed(&mut b, neg, bv)?;
        Ok(b.finish())
    })
}

/// The display of an f32 value (Rust's `Display` of f32, which the
/// reference uses): the shortest digits of the binary32 value, always
/// positional; -0.0 is `-0`.
pub fn text32(f: &mut Fns) -> R<usize> {
    let pf = parts(f, F32)?;
    let df = digits(f, F32, Pick::Rust)?;
    f.get("rt.f32.text", vec![Ty::F64], Ty::Bytes, move |f, mut b| {
        let x = b.arg(0);
        zero_text(&mut b, &x, b"-0", b"0")?;
        let (neg, ds, k) = sign_digits(&mut b, f, x, pf, df)?;
        let s = positional(&mut b, ds, k, false)?;
        signed(&mut b, neg, s)?;
        Ok(b.finish())
    })
}

/// When x is a zero, return its text.
fn zero_text(b: &mut FB, x: &Atom, neg: &[u8], pos: &[u8]) -> R<()> {
    let z = b.p("f64.eq", vec![], vec![x.clone(), f64k(0.0)])?;
    let x = x.clone();
    b.when(z, |b| {
        let s = b.p("f64.sign", vec![], vec![x])?;
        b.if_(s, |b| {
            b.ret(Atom::bytes(neg));
            Ok(())
        }, |b| {
            b.ret(Atom::bytes(pos));
            Ok(())
        })
    })
}

/// Return the text with a minus sign when `neg`.
fn signed(b: &mut FB, neg: Atom, s: Atom) -> R<()> {
    let s2 = s.clone();
    b.if_(neg, |b| {
        let t = b.cat(Atom::bytes(b"-"), s)?;
        b.ret(t);
        Ok(())
    }, |b| {
        b.ret(s2);
        Ok(())
    })
}

/// The exact decimal text of trunc(x) for a finite x (interp
/// `exact_int_text`, and the value an overflow message names when a float
/// is cast to an integer type).
pub fn int_text(f: &mut Fns) -> R<usize> {
    let pf = parts(f, F64)?;
    let ofu = of_u64(f)?;
    let shlf = shl(f)?;
    let bdec = dec(f)?;
    let dec = rt::dec(f)?;
    f.get("rt.f64.int_text", vec![Ty::F64], Ty::Bytes, move |f, mut b| {
        let x = b.arg(0);
        let a = b.p("f64.abs", vec![], vec![x.clone()])?;
        let small = b.p("f64.lt", vec![], vec![a.clone(), f64k(9007199254740992.0)])?;
        let x2 = x.clone();
        b.when(small, |b| {
            let c = b.p("f64.bad.trunc", vec![Ty::i64()], vec![x2.clone()])?;
            b.guard_halt(c, |fb| fb.trap_value(Atom::bytes(b"internal: trunc")))?;
            let t = b.p("f64.trunc", vec![Ty::i64()], vec![x2])?;
            let s = call(b, f, dec, vec![t]);
            b.ret(s);
            Ok(())
        })?;
        // |x| >= 2^53 is an integer: m * 2^e with e >= 0
        let neg = b.p("f64.lt", vec![], vec![x, f64k(0.0)])?;
        let me = call(&mut b, f, pf, vec![a]);
        let met = b.ty(&me);
        let ps = b.prim("tup.open", vec![met], vec![me])?;
        let m = b.cast_in(ps[0].clone(), Ty::u64())?;
        let bm = call(&mut b, f, ofu, vec![m]);
        let e = b.cast_in(ps[1].clone(), Ty::Int(0, 65535))?;
        let v = call(&mut b, f, shlf, vec![bm, e]);
        let s = call(&mut b, f, bdec, vec![v]);
        signed(&mut b, neg, s)?;
        Ok(b.finish())
    })
}

/// `x as f32` for an integer x of any core type (Rust's `as`, one
/// rounding to nearest, ties to even): through binary64 when x is exact
/// there, otherwise rounded to 24 bits first, so that the value is
/// rounded once.
pub fn int_to_f32(f: &mut Fns) -> R<usize> {
    f.get("rt.int.to_f32", vec![wide()], Ty::F32, |_, mut b| {
        let v = b.arg(0);
        let (neg, mag) = rt::smag(&mut b, v.clone())?;
        let small = b.int_cmp("int.lt", mag.clone(), lit(1 << 53))?;
        let v2 = v.clone();
        b.when(small, |b| {
            let x = b.p("f64.of_int", vec![wide()], vec![v2])?;
            let c = b.p("f64.bad.to_f32", vec![], vec![x.clone()])?;
            b.guard_halt(c, |fb| fb.trap_value(Atom::bytes(b"internal: to_f32")))?;
            let y = b.p("f64.to_f32", vec![], vec![x])?;
            b.ret(y);
            Ok(())
        })?;
        // the bit length of the magnitude, and the 24 leading bits rounded
        let t = b.var(Ty::u64(), mag.clone());
        let nb = b.var(Ty::Int(0, 64), lit(0));
        // t is below 2^64: at most 64 halvings bring it to 0
        let cnt = b.counter();
        b.loop_(|b| {
            let tv = b.take(t);
            let z = b.int_cmp("int.eq", tv.clone(), lit(0))?;
            b.exit_if(z)?;
            b.count(cnt, lit(64))?;
            let (q, _) = b.divmod_lit(tv, 2, Ty::u64())?;
            b.set(t, q);
            let nv = b.take(nb);
            let n1 = b.add_in(nv, lit(1), Ty::Int(0, 64))?;
            b.set(nb, n1);
            Ok(())
        })?;
        let nbv = b.take(nb);
        let sh = b.sub_in(nbv, lit(24), Ty::Int(0, 64))?;
        // 2^sh and 2^(sh-1)
        let p = b.var(Ty::u64(), lit(1));
        b.for_n(sh.clone(), |b, _| {
            let pv = b.take(p);
            let p2 = b.mul_in(pv, lit(2), Ty::u64())?;
            b.set(p, p2);
            Ok(())
        })?;
        let pw = b.take(p);
        let nz = b.int_cmp("int.eq", pw.clone(), lit(0))?;
        b.guard_halt(nz, |fb| fb.trap_value(Atom::bytes(b"internal: shift")))?;
        let bq = b.p("int.bad.quo", vec![Ty::u64()], vec![mag.clone(), pw.clone()])?;
        b.guard_halt(bq, |fb| fb.trap_value(Atom::bytes(b"internal: quotient")))?;
        let q = b.p("int.quo", vec![Ty::u64()], vec![mag.clone(), pw.clone()])?;
        let rem = b.p("int.rem", vec![Ty::u64()], vec![mag, pw.clone()])?;
        let (halfp, _) = b.divmod_lit(pw, 2, Ty::u64())?;
        let gt = b.int_cmp("int.lt", halfp.clone(), rem.clone())?;
        let eq = b.int_cmp("int.eq", rem, halfp)?;
        let (_, odd) = b.divmod_lit(q.clone(), 2, Ty::u64())?;
        let odd = b.int_cmp("int.eq", odd, lit(1))?;
        let tie_up = b.p("bool.and", vec![], vec![eq, odd])?;
        let up = b.p("bool.or", vec![], vec![gt, tie_up])?;
        let qv = b.var(Ty::Int(0, 1 << 24), lit(0));
        let q2 = q.clone();
        b.if_(up, |b| {
            let q1 = b.add_in(q, lit(1), Ty::Int(0, 1 << 24))?;
            b.set(qv, q1);
            Ok(())
        }, |b| {
            let q1 = b.cast_in(q2, Ty::Int(0, 1 << 24))?;
            b.set(qv, q1);
            Ok(())
        })?;
        let qq = b.take(qv);
        let fq = b.p("f64.of_int", vec![Ty::Int(0, 1 << 24)], vec![qq])?;
        // times 2^sh, exact in binary64
        let x = b.var(Ty::F64, fq);
        b.for_n(sh, |b, _| {
            let xv = b.take(x);
            let x2 = fop(b, "mul", xv, f64k(2.0))?;
            b.set(x, x2);
            Ok(())
        })?;
        let xv = b.take(x);
        let xs = b.var(Ty::F64, xv.clone());
        b.when(neg, |b| {
            let y = b.p("f64.neg", vec![], vec![xv])?;
            b.set(xs, y);
            Ok(())
        })?;
        let xv = b.take(xs);
        let c = b.p("f64.bad.to_f32", vec![], vec![xv.clone()])?;
        b.guard_halt(c, |fb| fb.trap_value(Atom::bytes(b"internal: to_f32")))?;
        let y = b.p("f64.to_f32", vec![], vec![xv])?;
        b.ret(y);
        Ok(b.finish())
    })
}
