//! The runtime functions the lowering calls, written in the core.
//!
//! These are the pieces of section 5 that the lowering of step 3 needs
//! before the runtime library of ONE-16 exists: fuel (6.7), the decimal
//! text of integers and of the exact values that trap messages name (6.2),
//! SHA-256 (FIPS 180-4) for the log chain (6.1), the JSON escape of text
//! (6.1.1) and the escape of text in display (6.3 N7, ONE-18). Each is a
//! core function built here once, the same for every program; none is
//! charged fuel.
//!
//! Every loop has a bounded shape (`shape`): a measure, or a count bound
//! computed before the loop from its operands, which the loop cannot
//! reach; the comment at each count bound says why it cannot.
use super::build::{FB, Fns, R, count_t, lit, sym, wide};
use super::ir::*;

/// A builder of a runtime function: adds it, and the functions it calls,
/// to the registry, and returns its id.
pub type Builder = fn(&mut Fns) -> R<usize>;

/// Every builder of this module, by name, with each instance of a builder
/// that takes a parameter, so that a check over the runtime
/// (`tests/one_core.rs::every_runtime_loop_has_a_bounded_shape`) builds
/// every runtime function, whether or not a program calls it.
pub const BUILDERS: &[(&str, Builder)] = &[
    ("tick", tick), ("charge", charge), ("try_charge", try_charge), ("ceil_log2", ceil_log2), ("dec_sm", dec_sm),
    ("dec", dec), ("dec_exact add", |f| dec_exact(f, "add")), ("dec_exact sub", |f| dec_exact(f, "sub")),
    ("dec_exact mul", |f| dec_exact(f, "mul")), ("dec_exact neg", |f| dec_exact(f, "neg")),
    ("dec_exact abs", |f| dec_exact(f, "abs")), ("dec_exact quo", |f| dec_exact(f, "quo")), ("sha_hex", sha_hex),
    ("jstr", jstr), ("debug", debug), ("why", why), ("kept_reason", kept_reason), ("utf8_count", utf8_count),
];

pub fn u32t() -> Ty {
    Ty::Int(0, u32::MAX as i128)
}
fn u16t() -> Ty {
    Ty::Int(0, 0xffff)
}
fn bytes_list() -> Ty {
    Ty::list(Ty::u8())
}

/// The fuel global and the fuel limit global (a literal of the module, so a
/// test that needs another limit changes that literal, ONE-5).
pub fn fuel_globals(f: &mut Fns, limit: u64) -> (u32, u32) {
    let fuel = f.global("fuel", Ty::u64(), Some(Lit::Int(limit as i128)));
    let lim = f.global("fuel.limit", Ty::u64(), Some(Lit::Int(limit as i128)));
    (fuel, lim)
}

/// `tick(n)`: take n units of fuel, or trap "comptime fuel exhausted" with
/// the fuel at 0 (6.7).
pub fn tick(f: &mut Fns) -> R<usize> {
    let (g, _) = fuel_globals(f, 0);
    f.get("rt.tick", vec![Ty::u64()], Ty::Bool, |_, mut b| {
        let n = b.arg(0);
        let cur = b.let_global(g, Ty::u64());
        let c = b.int_cmp("int.lt", cur.clone(), n.clone())?;
        b.open();
        b.setg(g, lit(0));
        let h = b.trap_value(Atom::bytes(b"comptime fuel exhausted"))?;
        b.halt(h);
        let t = b.close();
        b.push(Stmt::If(c, t, vec![]));
        let r = b.sub_in(cur, n, Ty::u64())?;
        b.setg(g, r);
        b.ret(Atom::bool(true));
        Ok(b.finish())
    })
}

/// The cost counters of C.6 (interp `charge`): the units charged so far,
/// and the least `start + limit` of the budgets in force, 2^64 - 1 when
/// none is. A budget that started with `spent = start` and has limit l
/// refuses a charge of n when `spent + n > start + l`.
pub fn cost_globals(f: &mut Fns) -> (u32, u32) {
    let spent = f.global("cost.spent", Ty::u64(), Some(Lit::Int(0)));
    let deadline = f.global("cost.deadline", Ty::u64(), Some(Lit::Int(u64::MAX as i128)));
    (spent, deadline)
}

/// `charge(n)`: n units of cost, or trap "budget", charging nothing, when
/// they would take a budget in force past its limit (interp `charge`).
pub fn charge(f: &mut Fns) -> R<usize> {
    let (s, d) = cost_globals(f);
    f.get("rt.charge", vec![Ty::u64()], Ty::Bool, |_, mut b| {
        let n = b.arg(0);
        let cur = b.let_global(s, Ty::u64());
        let t = b.p("int.sadd", vec![Ty::u64()], vec![cur, n])?;
        let dl = b.let_global(d, Ty::u64());
        let over = b.int_cmp("int.lt", dl, t.clone())?;
        b.open();
        let h = b.trap_value(Atom::bytes(b"budget"))?;
        b.halt(h);
        let th = b.close();
        b.push(Stmt::If(over, th, vec![]));
        b.setg(s, t);
        b.ret(Atom::bool(true));
        Ok(b.finish())
    })
}

/// `try_charge(k)`: k charges of one unit each that stop at the first one a
/// budget refuses, without a trap: the charges of an evaluation whose
/// failure the reference ignores (interp `exec` of an assignment, which
/// evaluates the place again with `.ok()`).
pub fn try_charge(f: &mut Fns) -> R<usize> {
    let (s, d) = cost_globals(f);
    f.get("rt.try_charge", vec![Ty::u64()], Ty::Bool, |_, mut b| {
        let n = b.arg(0);
        let cur = b.let_global(s, Ty::u64());
        let t = b.p("int.sadd", vec![Ty::u64()], vec![cur.clone(), n])?;
        let dl = b.let_global(d, Ty::u64());
        // min(t, max(cur, dl))
        let m = b.p("int.max", vec![Ty::u64()], vec![cur, dl])?;
        let r = b.p("int.min", vec![Ty::u64()], vec![t, m])?;
        b.setg(s, r);
        b.ret(Atom::bool(true));
        Ok(b.finish())
    })
}

/// `ceil_log2(n)`: the least k with 2^k >= n, and 0 for n <= 1 (interp
/// `conc::ceil_log2`): the number of binary digits of n - 1.
pub fn ceil_log2(f: &mut Fns) -> R<usize> {
    f.get("rt.ceil_log2", vec![Ty::u64()], Ty::Int(0, 64), |_, mut b| {
        let n = b.arg(0);
        let k = b.var(Ty::Int(0, 64), lit(0));
        let small = b.int_cmp("int.le", n.clone(), lit(1))?;
        b.when(small, |b| {
            b.ret(lit(0));
            Ok(())
        })?;
        let m0 = b.sub_in(n, lit(1), Ty::u64())?;
        let m = b.var(Ty::u64(), m0);
        // m is below 2^64, so it reaches 0 after at most 64 halvings
        let cnt = b.counter();
        b.loop_(|b| {
            let mv = b.take(m);
            let z = b.int_cmp("int.eq", mv.clone(), lit(0))?;
            b.exit_if(z)?;
            b.count(cnt, lit(64))?;
            let (h, _) = b.divmod_lit(mv, 2, Ty::u64())?;
            b.set(m, h);
            let kv = b.take(k);
            let k1 = b.add_in(kv, lit(1), Ty::Int(0, 64))?;
            b.set(k, k1);
            Ok(())
        })?;
        let kv = b.take(k);
        b.ret(kv);
        Ok(b.finish())
    })
}

/// A one-byte string of byte `x` (an int known to lie in 0..255).
pub fn byte_str(b: &mut FB, x: Atom) -> R<Atom> {
    let x = b.cast_in(x, Ty::u8())?;
    let l = b.p("list.new", vec![Ty::u8()], vec![])?;
    let l = b.p("list.push", vec![Ty::u8()], vec![l, x])?;
    b.p("bytes.of_list", vec![], vec![l])
}

/// The sign and magnitude of an integer of any core type.
pub fn smag(b: &mut FB, v: Atom) -> R<(Atom, Atom)> {
    let neg = b.int_cmp("int.lt", v.clone(), lit(0))?;
    let c = b.p("int.bad.abs", vec![Ty::u64()], vec![v.clone()])?;
    b.guard_halt(c, |fb| fb.trap_value(Atom::bytes(b"internal: magnitude")))?;
    let m = b.p("int.abs", vec![Ty::u64()], vec![v])?;
    Ok((neg, m))
}

/// `dec_sm(neg, m)`: the decimal text of ±m.
pub fn dec_sm(f: &mut Fns) -> R<usize> {
    f.get("rt.dec_sm", vec![Ty::Bool, Ty::u64()], Ty::Bytes, |_, mut b| {
        let (neg, m) = (b.arg(0), b.arg(1));
        let z = b.int_cmp("int.eq", m.clone(), lit(0))?;
        b.open();
        b.ret(Atom::bytes(b"0"));
        let t = b.close();
        b.push(Stmt::If(z, t, vec![]));
        let acc = b.var(Ty::Bytes, Atom::bytes(b""));
        let x = b.var(Ty::u64(), m);
        // m is below 2^64 and has at most 20 digits: at most 20 iterations
        // divide it, and the 21st leaves the loop
        let cnt = b.counter();
        b.open();
        b.count(cnt, lit(21))?;
        b.open();
        {
            let xv = b.let_(Atom::V(x));
            let done = b.int_cmp("int.eq", xv.clone(), lit(0))?;
            b.open();
            b.brk(1);
            let t = b.close();
            b.push(Stmt::If(done, t, vec![]));
            let (q, r) = b.divmod_lit(xv, 10, Ty::u64())?;
            let d = b.add_in(r, lit(48), Ty::u64())?;
            let s = byte_str(&mut b, d)?;
            let a = b.let_(Atom::V(acc));
            let a2 = b.cat(s, a)?;
            b.set(acc, a2);
            b.set(x, q);
        }
        let body = b.close();
        b.push(Stmt::Block(body));
        let lp = b.close();
        b.push(Stmt::Loop(lp));
        let a = b.let_(Atom::V(acc));
        b.open();
        let s = b.cat(Atom::bytes(b"-"), a.clone())?;
        b.ret(s);
        let t = b.close();
        b.open();
        b.ret(a);
        let e = b.close();
        b.push(Stmt::If(neg, t, e));
        Ok(b.finish())
    })
}

/// `dec(v)`: the decimal text of an integer of any core type.
pub fn dec(f: &mut Fns) -> R<usize> {
    let ds = dec_sm(f)?;
    f.get("rt.dec", vec![wide()], Ty::Bytes, |_, mut b| {
        let (n, m) = smag(&mut b, Atom::V(0))?;
        let r = b.call(sym(ds), Ty::Bytes, vec![n, m]);
        b.ret(r);
        Ok(b.finish())
    })
}

/// `dec_limbs(neg, l)`: the decimal text of ±(the number whose base-2^16
/// digits are l, most significant first).
fn dec_limbs(f: &mut Fns) -> R<usize> {
    let lt = Ty::list(u16t());
    f.get("rt.dec_limbs", vec![Ty::Bool, lt.clone()], Ty::Bytes, |_, mut b| {
        let neg = b.arg(0);
        // each iteration writes one decimal digit of the number; a number
        // of n base-2^16 digits is below 65536^n < 10^(5n), so it has at
        // most 5n decimal digits, and 0 has one: at most 5n + 1 iterations
        let n0 = b.p("list.len", vec![u16t()], vec![b.arg(1)])?;
        let n5 = b.mul_in(n0, lit(5), count_t())?;
        let bound = b.add_in(n5, lit(1), count_t())?;
        let l = b.var(lt.clone(), b.arg(1));
        let acc = b.var(Ty::Bytes, Atom::bytes(b""));
        let cnt = b.counter();
        // repeat: divide l by 10 in place, prepend the remainder, until l is 0
        b.open();
        b.count(cnt, bound)?;
        b.open();
        {
            let n = b.p("list.len", vec![u16t()], vec![Atom::V(l)])?;
            let q = b.p("list.new", vec![u16t()], vec![])?;
            let qv = b.var(lt.clone(), q);
            let rem = b.var(Ty::Int(0, 9), lit(0));
            let nz = b.var(Ty::Bool, Atom::bool(false));
            let i = b.var(Ty::u64(), lit(0));
            b.open();
            {
                let iv = b.let_(Atom::V(i));
                let more = b.int_cmp("int.lt", iv.clone(), n.clone())?;
                b.open();
                let t = b.close();
                b.open();
                b.brk(1);
                let e = b.close();
                b.push(Stmt::If(more, t, e));
                let d = b.list_at(Atom::V(l), iv.clone())?;
                let rv = b.let_(Atom::V(rem));
                let hi = b.mul_in(rv, lit(65536), Ty::Int(0, 9 * 65536))?;
                let cur = b.add_in(hi, d, Ty::Int(0, 10 * 65536))?;
                let (qd, rd) = b.divmod_lit(cur, 10, Ty::Int(0, 10 * 65536))?;
                let qd = b.cast_in(qd, u16t())?;
                let rd = b.cast_in(rd, Ty::Int(0, 9))?;
                b.set(rem, rd);
                let z = b.int_cmp("int.eq", qd.clone(), lit(0))?;
                let nzv = b.let_(Atom::V(nz));
                let keep_zero = b.not(nzv.clone())?;
                let skip = b.p("bool.and", vec![], vec![z.clone(), keep_zero])?;
                b.open();
                b.open();
                let qq = b.let_(Atom::V(qv));
                let qq = b.p("list.push", vec![u16t()], vec![qq, qd])?;
                b.set(qv, qq);
                b.set(nz, Atom::bool(true));
                let push = b.close();
                b.push(Stmt::Block(push));
                let e = b.close();
                b.push(Stmt::If(skip, vec![], e));
                let i1 = b.add_in(iv, lit(1), Ty::u64())?;
                b.set(i, i1);
            }
            let inner = b.close();
            b.push(Stmt::Loop(vec![Stmt::Block(inner)]));
            let rv = b.let_(Atom::V(rem));
            let d = b.add_in(rv, lit(48), Ty::Int(0, 57))?;
            let s = byte_str(&mut b, d)?;
            let a = b.let_(Atom::V(acc));
            let a2 = b.cat(s, a)?;
            b.set(acc, a2);
            let old = b.let_(Atom::V(l));
            b.drop(old)?;
            let qn = b.let_(Atom::V(qv));
            b.set(l, qn);
            let ln = b.p("list.len", vec![u16t()], vec![Atom::V(l)])?;
            let empty = b.int_cmp("int.eq", ln, lit(0))?;
            b.open();
            b.brk(1);
            let t = b.close();
            b.push(Stmt::If(empty, t, vec![]));
        }
        let body = b.close();
        b.push(Stmt::Block(body));
        let lp = b.close();
        b.push(Stmt::Loop(lp));
        let old = b.let_(Atom::V(l));
        b.drop(old)?;
        let a = b.let_(Atom::V(acc));
        b.open();
        let s = b.cat(Atom::bytes(b"-"), a.clone())?;
        b.ret(s);
        let t = b.close();
        b.open();
        b.ret(a);
        let e = b.close();
        b.push(Stmt::If(neg, t, e));
        Ok(b.finish())
    })
}

/// The four base-2^16 digits of a u64, most significant first.
fn limbs4(b: &mut FB, m: Atom) -> R<Vec<Atom>> {
    let mut out = vec![];
    let mut x = m;
    for _ in 0..4 {
        let (q, r) = b.divmod_lit(x, 65536, Ty::u64())?;
        out.push(b.cast_in(r, u16t())?);
        x = q;
    }
    out.reverse();
    Ok(out)
}

/// The exact decimal of a + b, a - b, a * b, -a, |a| and trunc(a / b) for
/// integers of any core type, whose exact value need not fit a core type.
/// `op` is one of "add", "sub", "mul", "neg", "abs", "quo".
pub fn dec_exact(f: &mut Fns, op: &str) -> R<usize> {
    let ds = dec_sm(f)?;
    let dl = dec_limbs(f)?;
    let unary = matches!(op, "neg" | "abs");
    let params = if unary { vec![wide()] } else { vec![wide(), wide()] };
    let op = op.to_string();
    f.get(&format!("rt.dec_{op}"), params, Ty::Bytes, move |_, mut b| {
        let (na, ma) = smag(&mut b, Atom::V(0))?;
        match op.as_str() {
            "neg" => {
                let n = b.not(na)?;
                let r = b.call(sym(ds), Ty::Bytes, vec![n, ma]);
                b.ret(r);
            }
            "abs" => {
                let r = b.call(sym(ds), Ty::Bytes, vec![Atom::bool(false), ma]);
                b.ret(r);
            }
            "quo" => {
                let (nb, mb) = smag(&mut b, Atom::V(1))?;
                let z = b.int_cmp("int.eq", mb.clone(), lit(0))?;
                b.guard_halt(z, |fb| fb.trap_value(Atom::bytes(b"internal: quotient by zero")))?;
                let c = b.p("int.bad.quo", vec![Ty::u64()], vec![ma.clone(), mb.clone()])?;
                b.guard_halt(c, |fb| fb.trap_value(Atom::bytes(b"internal: quotient")))?;
                let q = b.p("int.quo", vec![Ty::u64()], vec![ma, mb])?;
                let n = b.p("bool.eq", vec![], vec![na, nb])?;
                let n = b.not(n)?;
                let r = b.call(sym(ds), Ty::Bytes, vec![n, q]);
                b.ret(r);
            }
            "add" | "sub" => {
                let (nb, mb) = smag(&mut b, Atom::V(1))?;
                let nb = if op == "sub" {
                    // a - b = a + (-b); -0 is 0, which the magnitude test below keeps unsigned
                    b.not(nb)?
                } else {
                    nb
                };
                let same = b.p("bool.eq", vec![], vec![na.clone(), nb.clone()])?;
                b.open();
                {
                    // ±(ma + mb): five base-2^16 digits
                    let la = limbs4(&mut b, ma.clone())?;
                    let lb = limbs4(&mut b, mb.clone())?;
                    let mut carry = lit(0);
                    let mut digits = vec![];
                    for k in (0..4).rev() {
                        let s = b.add_in(la[k].clone(), lb[k].clone(), Ty::Int(0, 2 * 65535))?;
                        let s = b.add_in(s, carry, Ty::Int(0, 2 * 65535 + 1))?;
                        let (q, r) = b.divmod_lit(s, 65536, Ty::Int(0, 2 * 65535 + 1))?;
                        digits.push(b.cast_in(r, u16t())?);
                        carry = q;
                    }
                    let carry = b.cast_in(carry, u16t())?;
                    digits.push(carry);
                    digits.reverse();
                    let mut l = b.p("list.new", vec![u16t()], vec![])?;
                    for d in digits {
                        l = b.p("list.push", vec![u16t()], vec![l, d])?;
                    }
                    let r = b.call(sym(dl), Ty::Bytes, vec![na.clone(), l]);
                    b.ret(r);
                }
                let t = b.close();
                b.open();
                {
                    // the signs differ: ±|ma - mb|, with the sign of the larger
                    let ge = b.int_cmp("int.le", mb.clone(), ma.clone())?;
                    b.open();
                    let d = b.sub_in(ma.clone(), mb.clone(), Ty::u64())?;
                    let r = b.call(sym(ds), Ty::Bytes, vec![na.clone(), d]);
                    b.ret(r);
                    let t2 = b.close();
                    b.open();
                    let d = b.sub_in(mb.clone(), ma.clone(), Ty::u64())?;
                    let r = b.call(sym(ds), Ty::Bytes, vec![nb.clone(), d]);
                    b.ret(r);
                    let e2 = b.close();
                    b.push(Stmt::If(ge, t2, e2));
                }
                let e = b.close();
                b.push(Stmt::If(same, t, e));
            }
            _ => {
                // mul: ±(ma * mb), eight base-2^16 digits by the schoolbook method
                let (nb, mb) = smag(&mut b, Atom::V(1))?;
                let la = limbs4(&mut b, ma)?;
                let lb = limbs4(&mut b, mb)?;
                // column k (0 least significant) sums a_i * b_j with i + j = k
                let col_t = Ty::Int(0, 4 * 65535 * 65535 + 65536 * 4);
                let mut carry = lit(0);
                let mut digits = vec![];
                for k in 0..8usize {
                    let mut s = carry;
                    for i in 0..4usize {
                        if k >= i && k - i < 4 {
                            let (ai, bj) = (la[3 - i].clone(), lb[3 - (k - i)].clone());
                            let p = b.mul_in(ai, bj, Ty::Int(0, 65535 * 65535))?;
                            s = b.add_in(s, p, col_t.clone())?;
                        }
                    }
                    let (q, r) = b.divmod_lit(s, 65536, col_t.clone())?;
                    digits.push(b.cast_in(r, u16t())?);
                    carry = q;
                }
                digits.reverse();
                let mut l = b.p("list.new", vec![u16t()], vec![])?;
                for d in digits {
                    l = b.p("list.push", vec![u16t()], vec![l, d])?;
                }
                let n = b.p("bool.eq", vec![], vec![na, nb])?;
                let n = b.not(n)?;
                let r = b.call(sym(dl), Ty::Bytes, vec![n, l]);
                b.ret(r);
            }
        }
        Ok(b.finish())
    })
}

const K256: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];
const H256: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];

fn u32lit(v: u32) -> Atom {
    lit(v as i128)
}

fn rotr(b: &mut FB, x: Atom, n: u32) -> R<Atom> {
    let a = b.p("int.bad.shift", vec![u32t()], vec![x.clone(), lit(n as i128)])?;
    b.guard_halt(a, |fb| fb.trap_value(Atom::bytes(b"internal: shift")))?;
    let r = b.p("int.shr", vec![u32t()], vec![x.clone(), lit(n as i128)])?;
    let c = b.p("int.bad.shift", vec![u32t()], vec![x.clone(), lit(32 - n as i128)])?;
    b.guard_halt(c, |fb| fb.trap_value(Atom::bytes(b"internal: shift")))?;
    let l = b.p("int.shl", vec![u32t()], vec![x, lit(32 - n as i128)])?;
    b.p("int.or", vec![u32t()], vec![r, l])
}

fn shr(b: &mut FB, x: Atom, n: u32) -> R<Atom> {
    let a = b.p("int.bad.shift", vec![u32t()], vec![x.clone(), lit(n as i128)])?;
    b.guard_halt(a, |fb| fb.trap_value(Atom::bytes(b"internal: shift")))?;
    b.p("int.shr", vec![u32t()], vec![x, lit(n as i128)])
}

fn xor3(b: &mut FB, x: Atom, y: Atom, z: Atom) -> R<Atom> {
    let t = b.p("int.xor", vec![u32t()], vec![x, y])?;
    b.p("int.xor", vec![u32t()], vec![t, z])
}

fn wadd(b: &mut FB, xs: Vec<Atom>) -> R<Atom> {
    let mut it = xs.into_iter();
    let mut acc = it.next().unwrap();
    for x in it {
        acc = b.p("int.wadd", vec![u32t()], vec![acc, x])?;
    }
    Ok(acc)
}

/// `sha_hex(b)`: the SHA-256 of b as 64 lower-case hex digits (FIPS 180-4).
pub fn sha_hex(f: &mut Fns) -> R<usize> {
    f.get("rt.sha256_hex", vec![Ty::Bytes], Ty::Bytes, |_, mut b| {
        let msg = b.arg(0);
        let len = b.p("bytes.len", vec![], vec![msg.clone()])?;
        // the padded message as a list of bytes
        let l0 = b.p("bytes.to_list", vec![], vec![msg])?;
        let l0 = b.p("list.push", vec![Ty::u8()], vec![l0, lit(0x80)])?;
        let m = b.var(bytes_list(), l0);
        // at most 63 zeros bring the length to 56 modulo 64, so at most 64
        // iterations start
        let cnt = b.counter();
        b.open();
        b.count(cnt, lit(64))?;
        b.open();
        {
            let n = b.p("list.len", vec![Ty::u8()], vec![Atom::V(m)])?;
            let (_, r) = b.divmod_lit(n, 64, Ty::u64())?;
            let done = b.int_cmp("int.eq", r, lit(56))?;
            b.open();
            b.brk(1);
            let t = b.close();
            b.push(Stmt::If(done, t, vec![]));
            let mv = b.let_(Atom::V(m));
            let mv = b.p("list.push", vec![Ty::u8()], vec![mv, lit(0)])?;
            b.set(m, mv);
        }
        let body = b.close();
        b.push(Stmt::Block(body));
        let lp = b.close();
        b.push(Stmt::Loop(lp));
        // the bit length, 8 bytes big-endian (a text is below 2^61 bytes)
        let bits = b.mul_in(len, lit(8), Ty::u64())?;
        let mut parts = vec![];
        let mut x = bits;
        for _ in 0..8 {
            let (q, r) = b.divmod_lit(x, 256, Ty::u64())?;
            parts.push(b.cast_in(r, Ty::u8())?);
            x = q;
        }
        parts.reverse();
        for p in parts {
            let mv = b.let_(Atom::V(m));
            let mv = b.p("list.push", vec![Ty::u8()], vec![mv, p])?;
            b.set(m, mv);
        }
        let hs: Vec<u32> = H256.iter().map(|h| b.var(u32t(), u32lit(*h))).collect();
        let nblocks = {
            let n = b.p("list.len", vec![Ty::u8()], vec![Atom::V(m)])?;
            b.divmod_lit(n, 64, Ty::u64())?.0
        };
        let blk = b.var(Ty::u64(), lit(0));
        b.open();
        {
            let bi = b.let_(Atom::V(blk));
            let more = b.int_cmp("int.lt", bi.clone(), nblocks.clone())?;
            b.open();
            let t = b.close();
            b.open();
            b.brk(1);
            let e = b.close();
            b.push(Stmt::If(more, t, e));
            let base = b.mul_in(bi.clone(), lit(64), Ty::u64())?;
            // the message schedule
            let w0 = b.p("list.new", vec![u32t()], vec![])?;
            let w = b.var(Ty::list(u32t()), w0);
            for t in 0..16i128 {
                let mut word = lit(0);
                for k in 0..4i128 {
                    let at = b.add_in(base.clone(), lit(4 * t + k), Ty::u64())?;
                    let byte = b.list_at(Atom::V(m), at)?;
                    let sh = b.mul_in(word, lit(256), Ty::Int(0, (1i128 << 32) - 256))?;
                    word = b.add_in(sh, byte, u32t())?;
                }
                let wv = b.let_(Atom::V(w));
                let wv = b.p("list.push", vec![u32t()], vec![wv, word])?;
                b.set(w, wv);
            }
            let ti = b.var(Ty::Int(0, 64), lit(16));
            b.open();
            {
                let tv = b.let_(Atom::V(ti));
                let more = b.int_cmp("int.lt", tv.clone(), lit(64))?;
                b.open();
                let t = b.close();
                b.open();
                b.brk(1);
                let e = b.close();
                b.push(Stmt::If(more, t, e));
                let at = |b: &mut FB, k: i128| -> R<Atom> {
                    let i = b.sub_in(tv.clone(), lit(k), Ty::Int(0, 63))?;
                    b.list_at(Atom::V(w), i)
                };
                let w2 = at(&mut b, 2)?;
                let w7 = at(&mut b, 7)?;
                let w15 = at(&mut b, 15)?;
                let w16 = at(&mut b, 16)?;
                let a1 = rotr(&mut b, w2.clone(), 17)?;
                let a2 = rotr(&mut b, w2.clone(), 19)?;
                let a3 = shr(&mut b, w2, 10)?;
                let s1 = xor3(&mut b, a1, a2, a3)?;
                let b1 = rotr(&mut b, w15.clone(), 7)?;
                let b2 = rotr(&mut b, w15.clone(), 18)?;
                let b3 = shr(&mut b, w15, 3)?;
                let s0 = xor3(&mut b, b1, b2, b3)?;
                let nw = wadd(&mut b, vec![s1, w7, s0, w16])?;
                let wv = b.let_(Atom::V(w));
                let wv = b.p("list.push", vec![u32t()], vec![wv, nw])?;
                b.set(w, wv);
                let t1 = b.add_in(tv, lit(1), Ty::Int(0, 64))?;
                b.set(ti, t1);
            }
            let sched = b.close();
            b.push(Stmt::Loop(vec![Stmt::Block(sched)]));
            // the compression
            let vs: Vec<u32> = hs.iter().map(|h| {
                let x = b.let_(Atom::V(*h));
                b.var(u32t(), x)
            }).collect();
            let ri = b.var(Ty::Int(0, 64), lit(0));
            b.open();
            {
                let rv = b.let_(Atom::V(ri));
                let more = b.int_cmp("int.lt", rv.clone(), lit(64))?;
                b.open();
                let t = b.close();
                b.open();
                b.brk(1);
                let e = b.close();
                b.push(Stmt::If(more, t, e));
                let cur: Vec<Atom> = vs.iter().map(|v| b.let_(Atom::V(*v))).collect();
                let (a, bb, c, d, e, ff, g, h) = (cur[0].clone(), cur[1].clone(), cur[2].clone(), cur[3].clone(), cur[4].clone(), cur[5].clone(), cur[6].clone(), cur[7].clone());
                let e6 = rotr(&mut b, e.clone(), 6)?;
                let e11 = rotr(&mut b, e.clone(), 11)?;
                let e25 = rotr(&mut b, e.clone(), 25)?;
                let s1 = xor3(&mut b, e6, e11, e25)?;
                let ef = b.p("int.and", vec![u32t()], vec![e.clone(), ff.clone()])?;
                let ne = b.p("int.not", vec![u32t()], vec![e.clone()])?;
                let neg = b.p("int.and", vec![u32t()], vec![ne, g.clone()])?;
                let ch = b.p("int.xor", vec![u32t()], vec![ef, neg])?;
                // K[i]: a byte string of the 64 constants, read 4 bytes at a time
                let ktab: Vec<u8> = K256.iter().flat_map(|k| k.to_be_bytes()).collect();
                let ktab = Atom::bytes(&ktab);
                let mut kw = lit(0);
                for j in 0..4i128 {
                    let i4 = b.mul_in(rv.clone(), lit(4), Ty::Int(0, 252))?;
                    let ij = b.add_in(i4, lit(j), Ty::Int(0, 255))?;
                    let kb = b.byte_at(ktab.clone(), ij)?;
                    let sh = b.mul_in(kw, lit(256), Ty::Int(0, (1i128 << 32) - 256))?;
                    kw = b.add_in(sh, kb, u32t())?;
                }
                let wi = b.list_at(Atom::V(w), rv.clone())?;
                let t1 = wadd(&mut b, vec![h, s1, ch, kw, wi])?;
                let a2 = rotr(&mut b, a.clone(), 2)?;
                let a13 = rotr(&mut b, a.clone(), 13)?;
                let a22 = rotr(&mut b, a.clone(), 22)?;
                let s0 = xor3(&mut b, a2, a13, a22)?;
                let ab = b.p("int.and", vec![u32t()], vec![a.clone(), bb.clone()])?;
                let ac = b.p("int.and", vec![u32t()], vec![a.clone(), c.clone()])?;
                let bc = b.p("int.and", vec![u32t()], vec![bb.clone(), c.clone()])?;
                let maj = xor3(&mut b, ab, ac, bc)?;
                let t2 = wadd(&mut b, vec![s0, maj])?;
                let ne = wadd(&mut b, vec![d, t1.clone()])?;
                let na = wadd(&mut b, vec![t1, t2])?;
                let next = [na, a, bb, c, ne, e, ff, g];
                for (v, x) in vs.iter().zip(next) {
                    b.set(*v, x);
                }
                let r1 = b.add_in(rv, lit(1), Ty::Int(0, 64))?;
                b.set(ri, r1);
            }
            let comp = b.close();
            b.push(Stmt::Loop(vec![Stmt::Block(comp)]));
            for (h, v) in hs.iter().zip(&vs) {
                let hv = b.let_(Atom::V(*h));
                let vv = b.let_(Atom::V(*v));
                let s = wadd(&mut b, vec![hv, vv])?;
                b.set(*h, s);
            }
            let wv = b.let_(Atom::V(w));
            b.drop(wv)?;
            let b1 = b.add_in(bi, lit(1), Ty::u64())?;
            b.set(blk, b1);
        }
        let blocks = b.close();
        b.push(Stmt::Loop(vec![Stmt::Block(blocks)]));
        let mv = b.let_(Atom::V(m));
        b.drop(mv)?;
        // 64 hex digits
        let out0 = b.p("list.new", vec![Ty::u8()], vec![])?;
        let out = b.var(bytes_list(), out0);
        let hexd = Atom::bytes(b"0123456789abcdef");
        for h in &hs {
            let mut nib = vec![];
            let mut x = b.let_(Atom::V(*h));
            for _ in 0..8 {
                let (q, r) = b.divmod_lit(x, 16, u32t())?;
                nib.push(r);
                x = q;
            }
            nib.reverse();
            for r in nib {
                let r = b.cast_in(r, Ty::Int(0, 15))?;
                let c = b.byte_at(hexd.clone(), r)?;
                let o = b.let_(Atom::V(out));
                let o = b.p("list.push", vec![Ty::u8()], vec![o, c])?;
                b.set(out, o);
            }
        }
        let o = b.let_(Atom::V(out));
        let r = b.p("bytes.of_list", vec![], vec![o])?;
        b.ret(r);
        Ok(b.finish())
    })
}

/// Push the bytes of the literal `s` onto the list variable `out`.
fn push_lit(b: &mut FB, out: u32, s: &[u8]) -> R<()> {
    for c in s {
        let o = b.let_(Atom::V(out));
        let o = b.p("list.push", vec![Ty::u8()], vec![o, lit(*c as i128)])?;
        b.set(out, o);
    }
    Ok(())
}

fn push_byte(b: &mut FB, out: u32, c: Atom) -> R<()> {
    let o = b.let_(Atom::V(out));
    let o = b.p("list.push", vec![Ty::u8()], vec![o, c])?;
    b.set(out, o);
    Ok(())
}

/// Push the lower-case hex digits of `v` (no leading zeros) onto `out`.
fn push_hex(b: &mut FB, out: u32, v: Atom, width: Option<usize>) -> R<()> {
    let hexd = Atom::bytes(b"0123456789abcdef");
    let digits = b.p("list.new", vec![Ty::u8()], vec![])?;
    let ds = b.var(bytes_list(), digits);
    let x = b.var(Ty::Int(0, 0x10ffff), v);
    let count = b.var(Ty::Int(0, 8), lit(0));
    // v is at most 0x10ffff, six hex digits, and the loop writes one digit
    // per iteration until v is written and the width reached
    let most = 6.max(width.unwrap_or(0)) as i128;
    let cnt = b.counter();
    b.open();
    b.count(cnt, lit(most))?;
    b.open();
    {
        let xv = b.let_(Atom::V(x));
        let (q, r) = b.divmod_lit(xv, 16, Ty::Int(0, 0x10ffff))?;
        let r = b.cast_in(r, Ty::Int(0, 15))?;
        let c = b.byte_at(hexd, r)?;
        let d = b.let_(Atom::V(ds));
        let d = b.p("list.push", vec![Ty::u8()], vec![d, c])?;
        b.set(ds, d);
        b.set(x, q.clone());
        let cv = b.let_(Atom::V(count));
        let c1 = b.add_in(cv, lit(1), Ty::Int(0, 8))?;
        b.set(count, c1.clone());
        let z = b.int_cmp("int.eq", q, lit(0))?;
        let stop = match width {
            Some(w) => {
                let full = b.int_cmp("int.le", lit(w as i128), c1)?;
                b.p("bool.and", vec![], vec![z, full])?
            }
            None => z,
        };
        b.open();
        b.brk(1);
        let t = b.close();
        b.push(Stmt::If(stop, t, vec![]));
    }
    let body = b.close();
    b.push(Stmt::Block(body));
    let lp = b.close();
    b.push(Stmt::Loop(lp));
    // the digits were collected least significant first; there are at most
    // `most`, one taken per iteration, and one more iteration leaves
    let n = b.p("list.len", vec![Ty::u8()], vec![Atom::V(ds)])?;
    let i = b.var(Ty::u64(), n);
    let cnt = b.counter();
    b.open();
    b.count(cnt, lit(most + 1))?;
    b.open();
    {
        let iv = b.let_(Atom::V(i));
        let z = b.int_cmp("int.eq", iv.clone(), lit(0))?;
        b.open();
        b.brk(1);
        let t = b.close();
        b.push(Stmt::If(z, t, vec![]));
        let i1 = b.sub_in(iv, lit(1), Ty::u64())?;
        b.set(i, i1.clone());
        let c = b.list_at(Atom::V(ds), i1)?;
        push_byte(b, out, c)?;
    }
    let body = b.close();
    b.push(Stmt::Block(body));
    let lp = b.close();
    b.push(Stmt::Loop(lp));
    let d = b.let_(Atom::V(ds));
    b.drop(d)?;
    Ok(())
}

/// `jstr(t)`: text as a JSON string (6.1.1): `"` and `\` and LF, CR and
/// TAB by their short escapes, every other byte below 0x20 as `\u00xx`.
pub fn jstr(f: &mut Fns) -> R<usize> {
    f.get("rt.jstr", vec![Ty::Bytes], Ty::Bytes, |_, mut b| {
        let t = b.arg(0);
        let o0 = b.p("list.new", vec![Ty::u8()], vec![])?;
        let out = b.var(bytes_list(), o0);
        push_lit(&mut b, out, b"\"")?;
        let n = b.p("bytes.len", vec![], vec![t.clone()])?;
        let i = b.var(Ty::u64(), lit(0));
        b.open();
        {
            let iv = b.let_(Atom::V(i));
            let more = b.int_cmp("int.lt", iv.clone(), n.clone())?;
            b.open();
            let tt = b.close();
            b.open();
            b.brk(1);
            let e = b.close();
            b.push(Stmt::If(more, tt, e));
            let c = b.byte_at(t.clone(), iv.clone())?;
            let i1 = b.add_in(iv, lit(1), Ty::u64())?;
            b.set(i, i1);
            // a chain of tests, each leaving the iteration when it applies
            b.open();
            for (ch, esc) in [(b'"', &b"\\\""[..]), (b'\\', b"\\\\"), (b'\n', b"\\n"), (b'\r', b"\\r"), (b'\t', b"\\t")] {
                let is = b.int_cmp("int.eq", c.clone(), lit(ch as i128))?;
                b.open();
                push_lit(&mut b, out, esc)?;
                b.brk(1);
                let tt = b.close();
                b.push(Stmt::If(is, tt, vec![]));
            }
            let ctl = b.int_cmp("int.lt", c.clone(), lit(0x20))?;
            b.open();
            push_lit(&mut b, out, b"\\u00")?;
            push_hex(&mut b, out, c.clone(), Some(2))?;
            b.brk(1);
            let tt = b.close();
            b.push(Stmt::If(ctl, tt, vec![]));
            push_byte(&mut b, out, c)?;
            let chain = b.close();
            b.push(Stmt::Block(chain));
        }
        let body = b.close();
        b.push(Stmt::Loop(vec![Stmt::Block(body)]));
        push_lit(&mut b, out, b"\"")?;
        let o = b.let_(Atom::V(out));
        let r = b.p("bytes.of_list", vec![], vec![o])?;
        b.ret(r);
        Ok(b.finish())
    })
}

/// The ranges of the display escape set (6.3 N7: `spec/data/display-escape.json`,
/// read by `interp::display_escaped`) that meet the code points `from..to`,
/// each cut to that interval. The runtime reads the same data as the
/// reference, so no display depends on the Unicode tables of a toolchain.
pub fn escape_ranges(from: u32, to: u32) -> Vec<(u32, u32)> {
    crate::interp::display_escaped()
        .iter()
        .filter(|(lo, hi)| *hi >= from && *lo < to)
        .map(|(lo, hi)| ((*lo).max(from), (*hi).min(to - 1)))
        .collect()
}

/// `esc(cp)`: whether display escapes code point cp (cp >= 0x80), by binary
/// search in a byte string of ranges (three bytes for each bound).
fn esc_cp(f: &mut Fns) -> R<usize> {
    let cpt = Ty::Int(0, 0x10ffff);
    f.get("rt.escaped", vec![cpt.clone()], Ty::Bool, move |_, mut b| {
        let cp = b.arg(0);
        let rs = escape_ranges(0x80, 0x110000);
        let mut tab = vec![];
        for (lo, hi) in &rs {
            tab.extend_from_slice(&lo.to_be_bytes()[1..]);
            tab.extend_from_slice(&hi.to_be_bytes()[1..]);
        }
        let tab = Atom::bytes(&tab);
        let n = rs.len() as i128;
        let lo = b.var(Ty::Int(0, n), lit(0));
        let hi = b.var(Ty::Int(0, n), lit(n));
        let read3 = |b: &mut FB, at: Atom| -> R<Atom> {
            let mut acc = lit(0);
            for k in 0..3 {
                let i = b.add_in(at.clone(), lit(k), Ty::Int(0, 6 * n))?;
                let x = b.byte_at(tab.clone(), i)?;
                let s = b.mul_in(acc, lit(256), Ty::Int(0, 0xffff00))?;
                acc = b.add_in(s, x, Ty::Int(0, 0xffffff))?;
            }
            Ok(acc)
        };
        // each iteration that does not return leaves at most half of the
        // hi - lo ranges still to search, so after as many iterations as n
        // has binary digits none is left, and one more iteration returns
        let most = (128 - (n as u128).leading_zeros()) as i128 + 1;
        let cnt = b.counter();
        b.open();
        b.count(cnt, lit(most))?;
        b.open();
        {
            let l = b.let_(Atom::V(lo));
            let h = b.let_(Atom::V(hi));
            let empty = b.int_cmp("int.le", h.clone(), l.clone())?;
            b.open();
            b.ret(Atom::bool(false));
            let t = b.close();
            b.push(Stmt::If(empty, t, vec![]));
            let s = b.add_in(l.clone(), h.clone(), Ty::Int(0, 2 * n))?;
            let (mid, _) = b.divmod_lit(s, 2, Ty::Int(0, 2 * n))?;
            let mid = b.cast_in(mid, Ty::Int(0, n))?;
            let at = b.mul_in(mid.clone(), lit(6), Ty::Int(0, 6 * n))?;
            let rlo = read3(&mut b, at.clone())?;
            let at3 = b.add_in(at, lit(3), Ty::Int(0, 6 * n + 3))?;
            let at3 = b.cast_in(at3, Ty::Int(0, 6 * n))?;
            let rhi = read3(&mut b, at3)?;
            let below = b.int_cmp("int.lt", cp.clone(), rlo)?;
            b.open();
            b.set(hi, mid.clone());
            b.brk(0);
            let t = b.close();
            b.push(Stmt::If(below, t, vec![]));
            let above = b.int_cmp("int.lt", rhi, cp.clone())?;
            b.open();
            let m1 = b.add_in(mid, lit(1), Ty::Int(0, n))?;
            b.set(lo, m1);
            b.brk(0);
            let t = b.close();
            b.push(Stmt::If(above, t, vec![]));
            b.ret(Atom::bool(true));
        }
        let body = b.close();
        b.push(Stmt::Block(body));
        let lp = b.close();
        b.push(Stmt::Loop(lp));
        Ok(b.finish())
    })
}

/// `debug(t)`: text as display shows it (6.3 N7, `interp::show_text`):
/// quoted, with `\t \r \n \0 \\ \"` by their short escapes and each code
/// point of the display escape set as `\u{hex}`.
pub fn debug(f: &mut Fns) -> R<usize> {
    let esc = esc_cp(f)?;
    f.get("rt.debug", vec![Ty::Bytes], Ty::Bytes, move |_, mut b| {
        let t = b.arg(0);
        let o0 = b.p("list.new", vec![Ty::u8()], vec![])?;
        let out = b.var(bytes_list(), o0);
        push_lit(&mut b, out, b"\"")?;
        let n = b.p("bytes.len", vec![], vec![t.clone()])?;
        let i = b.var(Ty::u64(), lit(0));
        // every iteration that does not leave moves i forward by at least
        // one byte, and i stays below n: at most n + 1 iterations start
        let most = b.add_in(n.clone(), lit(1), count_t())?;
        let cnt = b.counter();
        b.open();
        b.count(cnt, most)?;
        b.open();
        {
            let iv = b.let_(Atom::V(i));
            let more = b.int_cmp("int.lt", iv.clone(), n.clone())?;
            b.open();
            let tt = b.close();
            b.open();
            b.brk(1);
            let e = b.close();
            b.push(Stmt::If(more, tt, e));
            let c = b.byte_at(t.clone(), iv.clone())?;
            b.open();
            let ascii = b.int_cmp("int.lt", c.clone(), lit(0x80))?;
            b.open();
            {
                let i1 = b.add_in(iv.clone(), lit(1), Ty::u64())?;
                b.set(i, i1);
                for (ch, e) in [(b'\t', &b"\\t"[..]), (b'\r', b"\\r"), (b'\n', b"\\n"), (0u8, b"\\0"), (b'\\', b"\\\\"), (b'"', b"\\\"")] {
                    let is = b.int_cmp("int.eq", c.clone(), lit(ch as i128))?;
                    b.open();
                    push_lit(&mut b, out, e)?;
                    b.brk(1);
                    let tt = b.close();
                    b.push(Stmt::If(is, tt, vec![]));
                }
                // whether c lies in an ASCII range of the display escape set
                let mut ctl: Option<Atom> = None;
                for (lo, hi) in escape_ranges(0, 0x80) {
                    let ge = b.int_cmp("int.le", lit(lo as i128), c.clone())?;
                    let le = b.int_cmp("int.le", c.clone(), lit(hi as i128))?;
                    let inr = b.p("bool.and", vec![], vec![ge, le])?;
                    ctl = Some(match ctl {
                        Some(x) => b.p("bool.or", vec![], vec![x, inr])?,
                        None => inr,
                    });
                }
                let ctl = ctl.unwrap_or(Atom::bool(false));
                b.open();
                push_lit(&mut b, out, b"\\u{")?;
                push_hex(&mut b, out, c.clone(), None)?;
                push_lit(&mut b, out, b"}")?;
                b.brk(1);
                let tt = b.close();
                b.push(Stmt::If(ctl, tt, vec![]));
                push_byte(&mut b, out, c.clone())?;
                b.brk(0);
            }
            let ab = b.close();
            b.push(Stmt::If(ascii, ab, vec![]));
            // a lead byte: its sequence length and the code point
            let l3 = b.int_cmp("int.lt", c.clone(), lit(0xe0))?;
            let l4 = b.int_cmp("int.lt", c.clone(), lit(0xf0))?;
            let k = b.var(Ty::Int(2, 4), lit(4));
            let first = b.var(Ty::Int(0, 0xff), lit(0));
            b.open();
            b.set(k, lit(2));
            let x = b.p("int.and", vec![Ty::u8()], vec![c.clone(), lit(0x1f)])?;
            b.set(first, x);
            let t2 = b.close();
            b.open();
            b.open();
            b.set(k, lit(3));
            let x = b.p("int.and", vec![Ty::u8()], vec![c.clone(), lit(0x0f)])?;
            b.set(first, x);
            let t3 = b.close();
            b.open();
            let x = b.p("int.and", vec![Ty::u8()], vec![c.clone(), lit(0x07)])?;
            b.set(first, x);
            let t4 = b.close();
            b.push(Stmt::If(l4, t3, t4));
            let e3 = b.close();
            b.push(Stmt::If(l3, t2, e3));
            let cp = b.var(Ty::Int(0, 0x10ffff), Atom::V(first));
            let j = b.var(Ty::Int(1, 4), lit(1));
            // j runs from 1 to k, and k is at most 4: at most 4 iterations start
            let cnt = b.counter();
            b.open();
            b.count(cnt, lit(4))?;
            b.open();
            {
                let jv = b.let_(Atom::V(j));
                let kv = b.let_(Atom::V(k));
                let done = b.int_cmp("int.le", kv, jv.clone())?;
                b.open();
                b.brk(1);
                let tt = b.close();
                b.push(Stmt::If(done, tt, vec![]));
                let at = b.add_in(iv.clone(), jv.clone(), Ty::u64())?;
                let cb = b.byte_at(t.clone(), at)?;
                let six = b.p("int.and", vec![Ty::u8()], vec![cb, lit(0x3f)])?;
                let cv = b.let_(Atom::V(cp));
                let cv = b.cast_in(cv, Ty::Int(0, 0x7fff))?;
                let s = b.mul_in(cv, lit(64), Ty::Int(0, 0x1fffc0))?;
                let s = b.add_in(s, six, Ty::Int(0, 0x1fffff))?;
                let s = b.cast_in(s, Ty::Int(0, 0x10ffff))?;
                b.set(cp, s);
                let j1 = b.add_in(jv, lit(1), Ty::Int(1, 4))?;
                b.set(j, j1);
            }
            let dec = b.close();
            b.push(Stmt::Block(dec));
            let lp = b.close();
            b.push(Stmt::Loop(lp));
            let cpv = b.let_(Atom::V(cp));
            let e = b.call(sym(esc), Ty::Bool, vec![cpv.clone()]);
            let kv = b.let_(Atom::V(k));
            b.open();
            push_lit(&mut b, out, b"\\u{")?;
            push_hex(&mut b, out, cpv, None)?;
            push_lit(&mut b, out, b"}")?;
            let te = b.close();
            b.open();
            {
                let j2 = b.var(Ty::Int(0, 4), lit(0));
                // j2 runs from 0 to k, and k is at most 4: at most 5 iterations start
                let cnt = b.counter();
                b.open();
                b.count(cnt, lit(5))?;
                b.open();
                let jv = b.let_(Atom::V(j2));
                let done = b.int_cmp("int.le", kv.clone(), jv.clone())?;
                b.open();
                b.brk(1);
                let tt = b.close();
                b.push(Stmt::If(done, tt, vec![]));
                let at = b.add_in(iv.clone(), jv.clone(), Ty::u64())?;
                let cb = b.byte_at(t.clone(), at)?;
                push_byte(&mut b, out, cb)?;
                let j1 = b.add_in(jv, lit(1), Ty::Int(0, 4))?;
                b.set(j2, j1);
                let cp_body = b.close();
                b.push(Stmt::Block(cp_body));
                let lp = b.close();
                b.push(Stmt::Loop(lp));
            }
            let ee = b.close();
            b.push(Stmt::If(e, te, ee));
            let i1 = b.add_in(iv, kv, Ty::u64())?;
            b.set(i, i1);
            let chain = b.close();
            b.push(Stmt::Block(chain));
        }
        let body = b.close();
        b.push(Stmt::Block(body));
        let lp = b.close();
        b.push(Stmt::Loop(lp));
        push_lit(&mut b, out, b"\"")?;
        let o = b.let_(Atom::V(out));
        let r = b.p("bytes.of_list", vec![], vec![o])?;
        b.ret(r);
        Ok(b.finish())
    })
}

/// The text of a halt as an observation line ends: `trap: m` for a trap,
/// `failed with d` for a declared error with display d (interp `why`).
pub fn why(f: &mut Fns) -> R<usize> {
    f.get("rt.why", vec![Ty::halt()], Ty::Bytes, |_, mut b| {
        let h = b.arg(0);
        let t = b.p("sum.tag", vec![Ty::halt()], vec![h.clone()])?;
        let is0 = b.int_cmp("int.eq", t, lit(0))?;
        b.open();
        let m = b.p("sum.get", vec![Ty::halt()], vec![h.clone(), lit(0)])?;
        let r = b.cat(Atom::bytes(b"trap: "), m)?;
        b.ret(r);
        let tb = b.close();
        b.open();
        let t2 = b.p("sum.tag", vec![Ty::halt()], vec![h.clone()])?;
        let is1 = b.int_cmp("int.eq", t2, lit(1))?;
        b.open();
        let p = b.p("sum.get", vec![Ty::halt()], vec![h, lit(1)])?;
        let fs = b.prim("tup.open", vec![Ty::Tup(vec![Ty::Bytes, Ty::Bytes])], vec![p])?;
        let r = b.cat(Atom::bytes(b"failed with "), fs[1].clone())?;
        b.ret(r);
        let t1 = b.close();
        b.open();
        let hh = b.trap_value(Atom::bytes(b"internal: halt tag"))?;
        b.halt(hh);
        let e1 = b.close();
        b.push(Stmt::If(is1, t1, e1));
        let eb = b.close();
        b.push(Stmt::If(is0, tb, eb));
        Ok(b.finish())
    })
}

/// The refusal text of a step that halted (interp `region_step`): the
/// error's name for a declared error, `trap(m)` for a trap.
pub fn kept_reason(f: &mut Fns) -> R<usize> {
    f.get("rt.kept_reason", vec![Ty::halt()], Ty::Bytes, |_, mut b| {
        let h = b.arg(0);
        let t = b.p("sum.tag", vec![Ty::halt()], vec![h.clone()])?;
        let is0 = b.int_cmp("int.eq", t, lit(0))?;
        b.open();
        let m = b.p("sum.get", vec![Ty::halt()], vec![h.clone(), lit(0)])?;
        let r = b.cats(vec![Atom::bytes(b"trap("), m, Atom::bytes(b")")])?;
        b.ret(r);
        let tb = b.close();
        b.open();
        let t2 = b.p("sum.tag", vec![Ty::halt()], vec![h.clone()])?;
        let is1 = b.int_cmp("int.eq", t2, lit(1))?;
        b.open();
        let p = b.p("sum.get", vec![Ty::halt()], vec![h, lit(1)])?;
        let fs = b.prim("tup.open", vec![Ty::Tup(vec![Ty::Bytes, Ty::Bytes])], vec![p])?;
        b.ret(fs[0].clone());
        let t1 = b.close();
        b.open();
        let hh = b.trap_value(Atom::bytes(b"internal: halt tag"))?;
        b.halt(hh);
        let e1 = b.close();
        b.push(Stmt::If(is1, t1, e1));
        let eb = b.close();
        b.push(Stmt::If(is0, tb, eb));
        Ok(b.finish())
    })
}

/// `utf8_count(t)`: the number of code points of a text (its bytes that do
/// not continue a sequence).
pub fn utf8_count(f: &mut Fns) -> R<usize> {
    f.get("rt.utf8_count", vec![Ty::Bytes], Ty::u64(), |_, mut b| {
        let t = b.arg(0);
        let n = b.p("bytes.len", vec![], vec![t.clone()])?;
        let k = b.var(Ty::u64(), lit(0));
        let i = b.var(Ty::u64(), lit(0));
        b.open();
        {
            let iv = b.let_(Atom::V(i));
            let more = b.int_cmp("int.lt", iv.clone(), n.clone())?;
            b.open();
            let tt = b.close();
            b.open();
            b.brk(1);
            let e = b.close();
            b.push(Stmt::If(more, tt, e));
            let c = b.byte_at(t.clone(), iv.clone())?;
            let lo = b.int_cmp("int.lt", c.clone(), lit(0x80))?;
            let hi = b.int_cmp("int.le", lit(0xc0), c)?;
            let starts = b.p("bool.or", vec![], vec![lo, hi])?;
            b.open();
            let kv = b.let_(Atom::V(k));
            let k1 = b.add_in(kv, lit(1), Ty::u64())?;
            b.set(k, k1);
            let tt = b.close();
            b.push(Stmt::If(starts, tt, vec![]));
            let i1 = b.add_in(iv, lit(1), Ty::u64())?;
            b.set(i, i1);
        }
        let body = b.close();
        b.push(Stmt::Loop(vec![Stmt::Block(body)]));
        let r = b.let_(Atom::V(k));
        b.ret(r);
        Ok(b.finish())
    })
}
