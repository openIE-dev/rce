//! The runtime functions of text, written in the core: the methods of 5.15
//! on text (interp `builtin`), with the white space of 6.3 N7 and the UTF-8
//! rules of N7 and N8. Text is bytes holding valid UTF-8; a code point is
//! decoded where a method works on characters (`strip`, `reverse`, `chars`,
//! `words`), and every other method works on the bytes, which for valid
//! UTF-8 gives the same result.
//!
//! Each function is built once and is the same for every program; none is
//! charged fuel.
use super::build::{FB, Fns, R, lit, sym};
use super::ir::*;
use super::rt;

fn cpt() -> Ty {
    Ty::Int(0, 0x10ffff)
}
fn u8list() -> Ty {
    Ty::list(Ty::u8())
}
fn texts() -> Ty {
    Ty::list(Ty::Bytes)
}
fn idx() -> Ty {
    Ty::Int(0, INT_MAX)
}

fn call(b: &mut FB, f: &Fns, id: usize, args: Vec<Atom>) -> Atom {
    let r = f.ret(id);
    b.call(sym(id), r, args)
}

/// `t[i..j]` with its domain tests (i is not negative by its type).
fn slice(b: &mut FB, t: Atom, i: Atom, j: Atom) -> R<Atom> {
    let n = b.p("bytes.len", vec![], vec![t.clone()])?;
    let c1 = b.int_cmp("int.le", i.clone(), j.clone())?;
    b.guard_halt_unless(c1, "internal: slice")?;
    let c2 = b.int_cmp("int.le", j.clone(), n)?;
    b.guard_halt_unless(c2, "internal: slice")?;
    b.p("bytes.slice", vec![], vec![t, i, j])
}

/// Whether the byte c is in lo..=hi.
fn between(b: &mut FB, c: &Atom, lo: u8, hi: u8) -> R<Atom> {
    let a = b.int_cmp("int.le", lit(lo as i128), c.clone())?;
    let z = b.int_cmp("int.le", c.clone(), lit(hi as i128))?;
    b.p("bool.and", vec![], vec![a, z])
}

fn push_text(b: &mut FB, l: u32, x: Atom) -> R<()> {
    let v = b.take(l);
    let v = b.p("list.push", vec![Ty::Bytes], vec![v, x])?;
    b.set(l, v);
    Ok(())
}

/// `decode(t, i)`: the code point that starts at byte i of the valid UTF-8
/// text t, and its length in bytes.
pub fn decode(f: &mut Fns) -> R<usize> {
    let rt_ = Ty::Tup(vec![cpt(), Ty::Int(1, 4)]);
    f.get("rt.text.decode", vec![Ty::Bytes, Ty::u64()], rt_.clone(), move |_, mut b| {
        let (t, i) = (b.arg(0), b.arg(1));
        let b0 = b.byte_at(t.clone(), i.clone())?;
        let ascii = b.int_cmp("int.lt", b0.clone(), lit(0x80))?;
        let (b0a, rta) = (b0.clone(), rt_.clone());
        b.when(ascii, |b| {
            let r = b.p("tup.mk", vec![rta], vec![b0a, lit(1)])?;
            b.ret(r);
            Ok(())
        })?;
        // the lead byte gives the length and the high bits
        let cp = b.var(cpt(), lit(0));
        let n = b.var(Ty::Int(1, 4), lit(4));
        for (lo, hi, len) in [(0xc0u8, 0xdfu8, 2i128), (0xe0, 0xef, 3), (0xf0, 0xf7, 4)] {
            let inr = between(&mut b, &b0, lo, hi)?;
            let b0c = b0.clone();
            b.when(inr, |b| {
                let v = b.sub_in(b0c, lit(lo as i128), cpt())?;
                b.set(cp, v);
                b.set(n, lit(len));
                Ok(())
            })?;
        }
        let nv = b.take(n);
        let n1 = b.sub_in(nv.clone(), lit(1), Ty::Int(0, 3))?;
        b.for_n(n1, |b, k| {
            let k1 = b.add_in(k, lit(1), Ty::Int(1, 4))?;
            let at = b.add_in(i.clone(), k1, Ty::Int(0, INT_MAX))?;
            let c = b.byte_at(t.clone(), at)?;
            let low = b.sub_in(c, lit(0x80), Ty::Int(0, 0x3f))?;
            let v = b.take(cp);
            let v = b.mul_in(v, lit(64), Ty::Int(0, 0x10ffff * 64))?;
            let v = b.add_in(v, low, Ty::Int(0, 0x10ffff * 64 + 63))?;
            let v = b.cast_in(v, cpt())?;
            b.set(cp, v);
            Ok(())
        })?;
        let cv = b.take(cp);
        let r = b.p("tup.mk", vec![rt_.clone()], vec![cv, nv])?;
        b.ret(r);
        Ok(b.finish())
    })
}

/// The length of the UTF-8 sequence a lead byte begins.
fn n_or(b: &mut FB, b0: &Atom) -> R<Atom> {
    let n = b.var(Ty::Int(1, 4), lit(1));
    for (lo, hi, len) in [(0xc0u8, 0xdfu8, 2i128), (0xe0, 0xef, 3), (0xf0, 0xf7, 4)] {
        let inr = between(b, b0, lo, hi)?;
        b.when(inr, |b| {
            b.set(n, lit(len));
            Ok(())
        })?;
    }
    Ok(b.take(n))
}

/// `is_ws(cp)`: the 25 white-space code points of 6.3 N7 (interp `is_ws`).
pub fn is_ws(f: &mut Fns) -> R<usize> {
    f.get("rt.text.is_ws", vec![cpt()], Ty::Bool, |_, mut b| {
        let cp = b.arg(0);
        let mut acc = Atom::bool(false);
        for c in crate::interp::WHITE_SPACE {
            let e = b.int_cmp("int.eq", cp.clone(), lit(c as i128))?;
            acc = b.p("bool.or", vec![], vec![acc, e])?;
        }
        b.ret(acc);
        Ok(b.finish())
    })
}

/// The byte index where the character before byte j (j > 0) starts.
fn char_before(b: &mut FB, t: &Atom, j: Atom) -> R<Atom> {
    let s0 = b.sub_in(j, lit(1), idx())?;
    let s = b.var(idx(), s0);
    let t = t.clone();
    b.loop_(|b| {
        let sv = b.take(s);
        let c = b.byte_at(t.clone(), sv.clone())?;
        let cont = between(b, &c, 0x80, 0xbf)?;
        b.exit_unless(cont)?;
        let s1 = b.sub_in(sv, lit(1), idx())?;
        b.set(s, s1);
        Ok(())
    })?;
    Ok(b.take(s))
}

/// `strip(t)`: t without the white space at either end (interp
/// `trim_matches(is_ws)`).
pub fn strip(f: &mut Fns) -> R<usize> {
    let dec = decode(f)?;
    let ws = is_ws(f)?;
    f.get("rt.text.strip", vec![Ty::Bytes], Ty::Bytes, move |f, mut b| {
        let t = b.arg(0);
        let n = b.p("bytes.len", vec![], vec![t.clone()])?;
        let a = b.var(idx(), lit(0));
        b.loop_(|b| {
            let av = b.take(a);
            let more = b.int_cmp("int.lt", av.clone(), n.clone())?;
            b.exit_unless(more)?;
            let ai = b.cast_in(av.clone(), Ty::u64())?;
            let d = call(b, f, dec, vec![t.clone(), ai]);
            let dt = b.ty(&d);
            let ps = b.prim("tup.open", vec![dt], vec![d])?;
            let w = call(b, f, ws, vec![ps[0].clone()]);
            b.exit_unless(w)?;
            let a1 = b.add_in(av, ps[1].clone(), idx())?;
            b.set(a, a1);
            Ok(())
        })?;
        let e = b.var(idx(), n.clone());
        b.loop_(|b| {
            let ev = b.take(e);
            let av = b.take(a);
            let more = b.int_cmp("int.lt", av, ev.clone())?;
            b.exit_unless(more)?;
            let s = char_before(b, &t, ev)?;
            let si = b.cast_in(s.clone(), Ty::u64())?;
            let d = call(b, f, dec, vec![t.clone(), si]);
            let dt = b.ty(&d);
            let ps = b.prim("tup.open", vec![dt], vec![d])?;
            let w = call(b, f, ws, vec![ps[0].clone()]);
            b.exit_unless(w)?;
            b.set(e, s);
            Ok(())
        })?;
        let av = b.take(a);
        let ev = b.take(e);
        let r = slice(&mut b, t, av, ev)?;
        b.ret(r);
        Ok(b.finish())
    })
}

/// `upper(t)` and `lower(t)`: ASCII letters changed, every other byte kept.
pub fn case(f: &mut Fns, upper: bool) -> R<usize> {
    let name = if upper { "rt.text.upper" } else { "rt.text.lower" };
    f.get(name, vec![Ty::Bytes], Ty::Bytes, move |_, mut b| {
        let t = b.arg(0);
        let n = b.p("bytes.len", vec![], vec![t.clone()])?;
        let o0 = b.p("list.new", vec![Ty::u8()], vec![])?;
        let out = b.var(u8list(), o0);
        b.for_n(n, |b, i| {
            let c = b.byte_at(t.clone(), i)?;
            let (lo, hi) = if upper { (b'a', b'z') } else { (b'A', b'Z') };
            let letter = between(b, &c, lo, hi)?;
            let x = b.var(Ty::u8(), c.clone());
            b.when(letter, |b| {
                let y = if upper { b.sub_in(c, lit(32), Ty::u8())? } else { b.add_in(c, lit(32), Ty::u8())? };
                b.set(x, y);
                Ok(())
            })?;
            let xv = b.take(x);
            let o = b.take(out);
            let o = b.p("list.push", vec![Ty::u8()], vec![o, xv])?;
            b.set(out, o);
            Ok(())
        })?;
        let o = b.take(out);
        let r = b.p("bytes.of_list", vec![], vec![o])?;
        b.ret(r);
        Ok(b.finish())
    })
}

/// The characters of t as one-character texts, in order.
pub fn chars(f: &mut Fns) -> R<usize> {
    f.get("rt.text.chars", vec![Ty::Bytes], texts(), |_, mut b| {
        let t = b.arg(0);
        let n = b.p("bytes.len", vec![], vec![t.clone()])?;
        let o0 = b.p("list.new", vec![Ty::Bytes], vec![])?;
        let out = b.var(texts(), o0);
        let i = b.var(idx(), lit(0));
        b.loop_(|b| {
            let iv = b.take(i);
            let more = b.int_cmp("int.lt", iv.clone(), n.clone())?;
            b.exit_unless(more)?;
            let c = b.byte_at(t.clone(), iv.clone())?;
            let l = n_or(b, &c)?;
            let j = b.add_in(iv.clone(), l, idx())?;
            let s = slice(b, t.clone(), iv, j.clone())?;
            push_text(b, out, s)?;
            b.set(i, j);
            Ok(())
        })?;
        let o = b.take(out);
        b.ret(o);
        Ok(b.finish())
    })
}

/// `json_esc_size(t)` (interp `json_size` of text): the two quotes, then
/// each `"`, `\\`, LF, CR and tab counts 2, each other byte below 0x20
/// counts 6, and every other byte 1, so that a character counts its UTF-8
/// length.
pub fn json_esc_size(f: &mut Fns) -> R<usize> {
    f.get("rt.text.json_esc_size", vec![Ty::Bytes], idx(), |_, mut b| {
        let t = b.arg(0);
        let n = b.p("bytes.len", vec![], vec![t.clone()])?;
        let acc = b.var(idx(), lit(2));
        b.for_n(n, |b, i| {
            let c = b.byte_at(t.clone(), i)?;
            let w = b.var(Ty::Int(1, 6), lit(1));
            let low = b.int_cmp("int.lt", c.clone(), lit(0x20))?;
            b.when(low, |b| {
                b.set(w, lit(6));
                Ok(())
            })?;
            for k in [b'"', b'\\', b'\n', b'\r', b'\t'] {
                let is = b.int_cmp("int.eq", c.clone(), lit(k as i128))?;
                b.when(is, |b| {
                    b.set(w, lit(2));
                    Ok(())
                })?;
            }
            let a = b.take(acc);
            let wv = b.take(w);
            let a2 = b.add_in(a, wv, idx())?;
            b.set(acc, a2);
            Ok(())
        })?;
        let a = b.take(acc);
        b.ret(a);
        Ok(b.finish())
    })
}

/// `reverse(t)`: the characters of t in reverse order.
pub fn reverse(f: &mut Fns) -> R<usize> {
    let ch = chars(f)?;
    f.get("rt.text.reverse", vec![Ty::Bytes], Ty::Bytes, move |f, mut b| {
        let a0 = b.arg(0);
        let cs = call(&mut b, f, ch, vec![a0]);
        let n = b.p("list.len", vec![Ty::Bytes], vec![cs.clone()])?;
        let acc = b.var(Ty::Bytes, Atom::bytes(b""));
        b.for_n(n, |b, i| {
            let c = b.list_at(cs.clone(), i)?;
            let a = b.take(acc);
            let a2 = b.cat(c, a)?;
            b.set(acc, a2);
            Ok(())
        })?;
        b.drop(cs)?;
        let a = b.take(acc);
        b.ret(a);
        Ok(b.finish())
    })
}

/// `words(t)`: the maximal runs of characters that are not white space.
pub fn words(f: &mut Fns) -> R<usize> {
    let dec = decode(f)?;
    let ws = is_ws(f)?;
    f.get("rt.text.words", vec![Ty::Bytes], texts(), move |f, mut b| {
        let t = b.arg(0);
        let n = b.p("bytes.len", vec![], vec![t.clone()])?;
        let o0 = b.p("list.new", vec![Ty::Bytes], vec![])?;
        let out = b.var(texts(), o0);
        let start = b.var(idx(), lit(0));
        let i = b.var(idx(), lit(0));
        let flush = |b: &mut FB, t: &Atom, end: Atom| -> R<()> {
            let sv = b.take(start);
            let nonempty = b.int_cmp("int.lt", sv.clone(), end.clone())?;
            let t = t.clone();
            b.when(nonempty, |b| {
                let w = slice(b, t, sv, end)?;
                push_text(b, out, w)
            })
        };
        b.loop_(|b| {
            let iv = b.take(i);
            let more = b.int_cmp("int.lt", iv.clone(), n.clone())?;
            b.exit_unless(more)?;
            let ii = b.cast_in(iv.clone(), Ty::u64())?;
            let d = call(b, f, dec, vec![t.clone(), ii]);
            let dt = b.ty(&d);
            let ps = b.prim("tup.open", vec![dt], vec![d])?;
            let j = b.add_in(iv.clone(), ps[1].clone(), idx())?;
            b.set(i, j.clone());
            let w = call(b, f, ws, vec![ps[0].clone()]);
            let t2 = t.clone();
            b.when(w, |b| {
                flush(b, &t2, iv)?;
                b.set(start, j);
                Ok(())
            })
        })?;
        flush(&mut b, &t, n)?;
        let o = b.take(out);
        b.ret(o);
        Ok(b.finish())
    })
}

/// `lines(t)` (Rust's `str::lines`): split after each LF, which is removed
/// with a CR before it; a last piece without LF is kept as it is, and an
/// empty one is not a line.
pub fn lines(f: &mut Fns) -> R<usize> {
    f.get("rt.text.lines", vec![Ty::Bytes], texts(), |_, mut b| {
        let t = b.arg(0);
        let n = b.p("bytes.len", vec![], vec![t.clone()])?;
        let o0 = b.p("list.new", vec![Ty::Bytes], vec![])?;
        let out = b.var(texts(), o0);
        let start = b.var(idx(), lit(0));
        b.for_n(n.clone(), |b, i| {
            let c = b.byte_at(t.clone(), i.clone())?;
            let lf = b.int_cmp("int.eq", c, lit(b'\n' as i128))?;
            let t2 = t.clone();
            b.when(lf, |b| {
                let sv = b.take(start);
                let end = b.var(idx(), i.clone());
                let has = b.int_cmp("int.lt", sv.clone(), i.clone())?;
                let (t3, i3) = (t2.clone(), i.clone());
                b.when(has, |b| {
                    let i1 = b.sub_in(i3, lit(1), idx())?;
                    let c = b.byte_at(t3, i1.clone())?;
                    let cr = b.int_cmp("int.eq", c, lit(b'\r' as i128))?;
                    b.when(cr, |b| {
                        b.set(end, i1);
                        Ok(())
                    })
                })?;
                let ev = b.take(end);
                let l = slice(b, t2, sv, ev)?;
                push_text(b, out, l)?;
                let s1 = b.add_in(i, lit(1), idx())?;
                b.set(start, s1);
                Ok(())
            })
        })?;
        let sv = b.take(start);
        let rest = b.int_cmp("int.lt", sv.clone(), n.clone())?;
        let t2 = t.clone();
        b.when(rest, |b| {
            let l = slice(b, t2, sv, n)?;
            push_text(b, out, l)
        })?;
        let o = b.take(out);
        b.ret(o);
        Ok(b.finish())
    })
}

/// `find(t, d, from)`: the least index i >= from where d occurs in t, or -1.
pub fn find(f: &mut Fns) -> R<usize> {
    f.get("rt.text.find", vec![Ty::Bytes, Ty::Bytes, idx()], Ty::Int(-1, INT_MAX), |_, mut b| {
        let (t, d, from) = (b.arg(0), b.arg(1), b.arg(2));
        let n = b.p("bytes.len", vec![], vec![t.clone()])?;
        let m = b.p("bytes.len", vec![], vec![d.clone()])?;
        let big = b.int_cmp("int.lt", n.clone(), m.clone())?;
        b.when(big, |b| {
            b.ret(lit(-1));
            Ok(())
        })?;
        let last = b.sub_in(n, m.clone(), idx())?;
        let i = b.var(idx(), from);
        b.loop_(|b| {
            let iv = b.take(i);
            let more = b.int_cmp("int.le", iv.clone(), last.clone())?;
            b.exit_unless(more)?;
            let j = b.add_in(iv.clone(), m.clone(), idx())?;
            let piece = slice(b, t.clone(), iv.clone(), j)?;
            let same = b.p("bytes.eq", vec![], vec![piece, d.clone()])?;
            let iv2 = iv.clone();
            b.when(same, |b| {
                let r = b.cast_in(iv2, Ty::Int(-1, INT_MAX))?;
                b.ret(r);
                Ok(())
            })?;
            let i1 = b.add_in(iv, lit(1), idx())?;
            b.set(i, i1);
            Ok(())
        })?;
        b.ret(lit(-1));
        Ok(b.finish())
    })
}

/// `contains(t, d)`.
pub fn contains(f: &mut Fns) -> R<usize> {
    let fd = find(f)?;
    f.get("rt.text.contains", vec![Ty::Bytes, Ty::Bytes], Ty::Bool, move |f, mut b| {
        let (a0, a1) = (b.arg(0), b.arg(1));
        let r = call(&mut b, f, fd, vec![a0, a1, lit(0)]);
        let ok = b.int_cmp("int.le", lit(0), r)?;
        b.ret(ok);
        Ok(b.finish())
    })
}

/// `startswith(t, d)` and `endswith(t, d)`.
pub fn affix(f: &mut Fns, start: bool) -> R<usize> {
    let name = if start { "rt.text.startswith" } else { "rt.text.endswith" };
    f.get(name, vec![Ty::Bytes, Ty::Bytes], Ty::Bool, move |_, mut b| {
        let (t, d) = (b.arg(0), b.arg(1));
        let n = b.p("bytes.len", vec![], vec![t.clone()])?;
        let m = b.p("bytes.len", vec![], vec![d.clone()])?;
        let big = b.int_cmp("int.lt", n.clone(), m.clone())?;
        b.when(big, |b| {
            b.ret(Atom::bool(false));
            Ok(())
        })?;
        let piece = if start {
            slice(&mut b, t, lit(0), m)?
        } else {
            let k = b.sub_in(n.clone(), m, idx())?;
            slice(&mut b, t, k, n)?
        };
        let same = b.p("bytes.eq", vec![], vec![piece, d])?;
        b.ret(same);
        Ok(b.finish())
    })
}

/// `split(t, d)` (Rust's `str::split` with a text pattern): the pieces
/// between the occurrences of d, left to right; with an empty d, an empty
/// piece, each character, and an empty piece.
pub fn split(f: &mut Fns) -> R<usize> {
    let fd = find(f)?;
    let ch = chars(f)?;
    f.get("rt.text.split", vec![Ty::Bytes, Ty::Bytes], texts(), move |f, mut b| {
        let (t, d) = (b.arg(0), b.arg(1));
        let n = b.p("bytes.len", vec![], vec![t.clone()])?;
        let m = b.p("bytes.len", vec![], vec![d.clone()])?;
        let o0 = b.p("list.new", vec![Ty::Bytes], vec![])?;
        let out = b.var(texts(), o0);
        let empty = b.int_cmp("int.eq", m.clone(), lit(0))?;
        let t2 = t.clone();
        b.when(empty, |b| {
            push_text(b, out, Atom::bytes(b""))?;
            let cs = call(b, f, ch, vec![t2]);
            let k = b.p("list.len", vec![Ty::Bytes], vec![cs.clone()])?;
            b.for_n(k, |b, i| {
                let c = b.list_at(cs.clone(), i)?;
                push_text(b, out, c)
            })?;
            b.drop(cs)?;
            push_text(b, out, Atom::bytes(b""))?;
            let o = b.take(out);
            b.ret(o);
            Ok(())
        })?;
        let start = b.var(idx(), lit(0));
        b.loop_(|b| {
            let sv = b.take(start);
            let p = call(b, f, fd, vec![t.clone(), d.clone(), sv.clone()]);
            let none = b.int_cmp("int.lt", p.clone(), lit(0))?;
            let (t2, n2, sv2) = (t.clone(), n.clone(), sv.clone());
            b.when(none, |b| {
                let piece = slice(b, t2, sv2, n2)?;
                push_text(b, out, piece)?;
                b.brk(0);
                Ok(())
            })?;
            let p = b.cast_in(p, idx())?;
            let piece = slice(b, t.clone(), sv, p.clone())?;
            push_text(b, out, piece)?;
            let s1 = b.add_in(p, m.clone(), idx())?;
            b.set(start, s1);
            Ok(())
        })?;
        let o = b.take(out);
        b.ret(o);
        Ok(b.finish())
    })
}

/// `isdigit`, `isalpha` and `isspace`: t is not empty and every byte is an
/// ASCII digit, an ASCII letter, or ASCII white space (Rust's
/// `is_ascii_whitespace`: space, tab, LF, FF and CR).
pub fn is_class(f: &mut Fns, class: &str) -> R<usize> {
    let class = class.to_string();
    f.get(&format!("rt.text.{class}"), vec![Ty::Bytes], Ty::Bool, move |_, mut b| {
        let t = b.arg(0);
        let n = b.p("bytes.len", vec![], vec![t.clone()])?;
        let z = b.int_cmp("int.eq", n.clone(), lit(0))?;
        b.when(z, |b| {
            b.ret(Atom::bool(false));
            Ok(())
        })?;
        b.for_n(n, |b, i| {
            let c = b.byte_at(t.clone(), i)?;
            let ok = match class.as_str() {
                "isdigit" => between(b, &c, b'0', b'9')?,
                "isalpha" => {
                    let u = between(b, &c, b'A', b'Z')?;
                    let l = between(b, &c, b'a', b'z')?;
                    b.p("bool.or", vec![], vec![u, l])?
                }
                _ => {
                    let mut acc = Atom::bool(false);
                    for w in [b' ', b'\t', b'\n', 0x0c, b'\r'] {
                        let e = b.int_cmp("int.eq", c.clone(), lit(w as i128))?;
                        acc = b.p("bool.or", vec![], vec![acc, e])?;
                    }
                    acc
                }
            };
            b.if_(ok, |_| Ok(()), |b| {
                b.ret(Atom::bool(false));
                Ok(())
            })
        })?;
        b.ret(Atom::bool(true));
        Ok(b.finish())
    })
}

/// A halt with a trap text.
fn trap(b: &mut FB, msg: Atom) -> R<()> {
    let h = b.trap_value(msg)?;
    b.halt(h);
    Ok(())
}

/// `toint(s)` (interp: `s.trim_matches(is_ws)` parsed as i128, then checked
/// as i64): an optional sign and at least one decimal digit. A text that
/// is not an i128 traps "not an integer: " and the display of s; an i128
/// outside i64 traps with the overflow message.
pub fn toint(f: &mut Fns) -> R<usize> {
    let st = strip(f)?;
    let dbg = rt::debug(f)?;
    f.get("rt.text.toint", vec![Ty::Bytes], Ty::i64(), move |f, mut b| {
        let s = b.arg(0);
        let t = call(&mut b, f, st, vec![s.clone()]);
        let n = b.p("bytes.len", vec![], vec![t.clone()])?;
        let bad = |b: &mut FB, f: &Fns, s: Atom| -> R<()> {
            let d = call(b, f, dbg, vec![s]);
            let m = b.cat(Atom::bytes(b"not an integer: "), d)?;
            trap(b, m)
        };
        let empty = b.int_cmp("int.eq", n.clone(), lit(0))?;
        let s1 = s.clone();
        b.when(empty, |b| bad(b, f, s1))?;
        let c0 = b.byte_at(t.clone(), lit(0))?;
        let minus = b.int_cmp("int.eq", c0.clone(), lit(b'-' as i128))?;
        let plus = b.int_cmp("int.eq", c0, lit(b'+' as i128))?;
        let signed = b.p("bool.or", vec![], vec![minus.clone(), plus])?;
        let i0 = b.var(idx(), lit(0));
        b.when(signed, |b| {
            b.set(i0, lit(1));
            Ok(())
        })?;
        let i = b.take(i0);
        let only_sign = b.int_cmp("int.eq", i.clone(), n.clone())?;
        let s1 = s.clone();
        b.when(only_sign, |b| bad(b, f, s1))?;
        // every byte after the sign is a digit
        let k = b.var(idx(), i.clone());
        let s1 = s.clone();
        b.loop_(|b| {
            let kv = b.take(k);
            let more = b.int_cmp("int.lt", kv.clone(), n.clone())?;
            b.exit_unless(more)?;
            let c = b.byte_at(t.clone(), kv.clone())?;
            let dg = between(b, &c, b'0', b'9')?;
            let s2 = s1.clone();
            b.if_(dg, |_| Ok(()), |b| bad(b, f, s2))?;
            let k1 = b.add_in(kv, lit(1), idx())?;
            b.set(k, k1);
            Ok(())
        })?;
        // without leading zeros (one digit kept)
        let j = b.var(idx(), i);
        b.loop_(|b| {
            let jv = b.take(j);
            let n1 = b.sub_in(n.clone(), lit(1), idx())?;
            let more = b.int_cmp("int.lt", jv.clone(), n1)?;
            b.exit_unless(more)?;
            let c = b.byte_at(t.clone(), jv.clone())?;
            let z = b.int_cmp("int.eq", c, lit(b'0' as i128))?;
            b.exit_unless(z)?;
            let j1 = b.add_in(jv, lit(1), idx())?;
            b.set(j, j1);
            Ok(())
        })?;
        let jv = b.take(j);
        let ds = slice(&mut b, t.clone(), jv, n.clone())?;
        let nd = b.p("bytes.len", vec![], vec![ds.clone()])?;
        // beyond i128: more than 39 digits, or 39 digits above its bound
        let long = b.int_cmp("int.lt", lit(39), nd.clone())?;
        let s1 = s.clone();
        b.when(long, |b| bad(b, f, s1))?;
        let is39 = b.int_cmp("int.eq", nd.clone(), lit(39))?;
        let (ds2, mi, s1) = (ds.clone(), minus.clone(), s.clone());
        b.when(is39, |b| {
            let lim = b.var(Ty::Bytes, Atom::bytes(i128::MAX.to_string().as_bytes()));
            b.when(mi, |b| {
                b.set(lim, Atom::bytes(i128::MIN.unsigned_abs().to_string().as_bytes()));
                Ok(())
            })?;
            let lv = b.take(lim);
            let over = b.p("bytes.lt", vec![], vec![lv, ds2])?;
            b.when(over, |b| bad(b, f, s1))
        })?;
        // an i128: outside i64, the overflow message with its exact value
        let zero = b.p("bytes.eq", vec![], vec![ds.clone(), Atom::bytes(b"0")])?;
        let nz = b.not(zero)?;
        let neg = b.p("bool.and", vec![], vec![minus, nz])?;
        let long = b.int_cmp("int.lt", lit(19), nd.clone())?;
        let is19 = b.int_cmp("int.eq", nd, lit(19))?;
        let lim = b.var(Ty::Bytes, Atom::bytes(i64::MAX.to_string().as_bytes()));
        b.when(neg.clone(), |b| {
            b.set(lim, Atom::bytes(i64::MIN.unsigned_abs().to_string().as_bytes()));
            Ok(())
        })?;
        let over19 = b.var(Ty::Bool, Atom::bool(false));
        let ds3 = ds.clone();
        b.when(is19, |b| {
            let l = b.take(lim);
            let o = b.p("bytes.lt", vec![], vec![l, ds3])?;
            b.set(over19, o);
            Ok(())
        })?;
        let o19 = b.take(over19);
        let outside = b.p("bool.or", vec![], vec![long, o19])?;
        let (ds4, neg2) = (ds.clone(), neg.clone());
        b.when(outside, |b| {
            let sign = b.var(Ty::Bytes, Atom::bytes(b""));
            b.when(neg2, |b| {
                b.set(sign, Atom::bytes(b"-"));
                Ok(())
            })?;
            let sv = b.take(sign);
            let m = b.cats(vec![Atom::bytes(b"overflow: "), sv, ds4, Atom::bytes(b" is outside i64")])?;
            trap(b, m)
        })?;
        // the value: the digits accumulated as a magnitude, then the sign
        let nd = b.p("bytes.len", vec![], vec![ds.clone()])?;
        let mag = b.var(Ty::u64(), lit(0));
        b.for_n(nd, |b, q| {
            let c = b.byte_at(ds.clone(), q)?;
            let dv = b.sub_in(c, lit(b'0' as i128), Ty::Int(0, 9))?;
            let mv = b.take(mag);
            let m10 = b.mul_in(mv, lit(10), Ty::u64())?;
            let m1 = b.add_in(m10, dv, Ty::u64())?;
            b.set(mag, m1);
            Ok(())
        })?;
        let mv = b.take(mag);
        let r = b.var(Ty::i64(), lit(0));
        let mv2 = mv.clone();
        b.if_(neg, |b| {
            let x = b.sub_in(lit(0), mv, Ty::i64())?;
            b.set(r, x);
            Ok(())
        }, |b| {
            let x = b.cast_in(mv2, Ty::i64())?;
            b.set(r, x);
            Ok(())
        })?;
        let rv = b.take(r);
        b.ret(rv);
        Ok(b.finish())
    })
}

/// `join(sep, xs)`: the texts of xs with sep between them.
pub fn join(f: &mut Fns) -> R<usize> {
    f.get("rt.text.join", vec![Ty::Bytes, texts()], Ty::Bytes, |_, mut b| {
        let (sep, xs) = (b.arg(0), b.arg(1));
        let n = b.p("list.len", vec![Ty::Bytes], vec![xs.clone()])?;
        let acc = b.var(Ty::Bytes, Atom::bytes(b""));
        b.for_n(n, |b, i| {
            let x = b.list_at(xs.clone(), i.clone())?;
            let first = b.int_cmp("int.eq", i, lit(0))?;
            let a = b.take(acc);
            let s = b.var(Ty::Bytes, sep.clone());
            b.when(first, |b| {
                b.set(s, Atom::bytes(b""));
                Ok(())
            })?;
            let sv = b.take(s);
            let a2 = b.cats(vec![a, sv, x])?;
            b.set(acc, a2);
            Ok(())
        })?;
        b.drop(xs)?;
        let a = b.take(acc);
        b.ret(a);
        Ok(b.finish())
    })
}

/// `utf8(xs)` (interp `List.utf8`): the bytes xs as text; a value outside
/// 0..255 traps "utf8: byte outside 0..255", and bytes that are not valid
/// UTF-8 (Rust's `from_utf8`: no overlong form, no surrogate, nothing above
/// U+10FFFF) trap "utf8: invalid UTF-8".
pub fn utf8(f: &mut Fns, et: Ty) -> R<usize> {
    let lt = Ty::list(et.clone());
    let key = format!("rt.text.utf8.{et:?}");
    f.get(&key, vec![lt], Ty::Bytes, move |_, mut b| {
        let xs = b.arg(0);
        let n = b.p("list.len", vec![et.clone()], vec![xs.clone()])?;
        let o0 = b.p("list.new", vec![Ty::u8()], vec![])?;
        let out = b.var(u8list(), o0);
        b.for_n(n.clone(), |b, i| {
            let x = b.list_at(xs.clone(), i)?;
            let lo = b.int_cmp("int.lt", x.clone(), lit(0))?;
            let hi = b.int_cmp("int.lt", lit(255), x.clone())?;
            let out_of = b.p("bool.or", vec![], vec![lo, hi])?;
            b.when(out_of, |b| trap(b, Atom::bytes(b"utf8: byte outside 0..255")))?;
            let x = b.cast_in(x, Ty::u8())?;
            let o = b.take(out);
            let o = b.p("list.push", vec![Ty::u8()], vec![o, x])?;
            b.set(out, o);
            Ok(())
        })?;
        b.drop(xs)?;
        let bs = b.take(out);
        let t = b.p("bytes.of_list", vec![], vec![bs])?;
        // validate, sequence by sequence
        let n = b.p("bytes.len", vec![], vec![t.clone()])?;
        let i = b.var(idx(), lit(0));
        let invalid = |b: &mut FB| trap(b, Atom::bytes(b"utf8: invalid UTF-8"));
        b.loop_(|b| {
            let iv = b.take(i);
            let more = b.int_cmp("int.lt", iv.clone(), n.clone())?;
            b.exit_unless(more)?;
            let b0 = b.byte_at(t.clone(), iv.clone())?;
            // (lead range, length, range of the second byte)
            let len = b.var(Ty::Int(0, 4), lit(0));
            let lo2 = b.var(Ty::u8(), lit(0x80));
            let hi2 = b.var(Ty::u8(), lit(0xbf));
            for (a, z, l, l2, h2) in [
                (0x00u8, 0x7fu8, 1i128, 0x80u8, 0xbfu8),
                (0xc2, 0xdf, 2, 0x80, 0xbf),
                (0xe0, 0xe0, 3, 0xa0, 0xbf),
                (0xe1, 0xec, 3, 0x80, 0xbf),
                (0xed, 0xed, 3, 0x80, 0x9f),
                (0xee, 0xef, 3, 0x80, 0xbf),
                (0xf0, 0xf0, 4, 0x90, 0xbf),
                (0xf1, 0xf3, 4, 0x80, 0xbf),
                (0xf4, 0xf4, 4, 0x80, 0x8f),
            ] {
                let inr = between(b, &b0, a, z)?;
                b.when(inr, |b| {
                    b.set(len, lit(l));
                    b.set(lo2, lit(l2 as i128));
                    b.set(hi2, lit(h2 as i128));
                    Ok(())
                })?;
            }
            let lv = b.take(len);
            let z = b.int_cmp("int.eq", lv.clone(), lit(0))?;
            b.when(z, |b| invalid(b))?;
            let end = b.add_in(iv.clone(), lv.clone(), idx())?;
            let short = b.int_cmp("int.lt", n.clone(), end.clone())?;
            b.when(short, |b| invalid(b))?;
            let l1 = b.sub_in(lv, lit(1), Ty::Int(0, 3))?;
            b.for_n(l1, |b, k| {
                let k1 = b.add_in(k, lit(1), Ty::Int(1, 4))?;
                let at = b.add_in(iv.clone(), k1.clone(), idx())?;
                let c = b.byte_at(t.clone(), at)?;
                let second = b.int_cmp("int.eq", k1, lit(1))?;
                let lo = b.var(Ty::u8(), lit(0x80));
                let hi = b.var(Ty::u8(), lit(0xbf));
                b.when(second, |b| {
                    let l = b.take(lo2);
                    let h = b.take(hi2);
                    b.set(lo, l);
                    b.set(hi, h);
                    Ok(())
                })?;
                let l = b.take(lo);
                let h = b.take(hi);
                let a1 = b.int_cmp("int.le", l, c.clone())?;
                let a2 = b.int_cmp("int.le", c, h)?;
                let ok = b.p("bool.and", vec![], vec![a1, a2])?;
                b.if_(ok, |_| Ok(()), |b| invalid(b))
            })?;
            b.set(i, end);
            Ok(())
        })?;
        b.ret(t);
        Ok(b.finish())
    })
}

/// `boundary(t, i)` (Rust's `is_char_boundary`): i is 0, the length of t,
/// or the index of a byte that does not continue a sequence.
pub fn boundary(f: &mut Fns) -> R<usize> {
    f.get("rt.text.boundary", vec![Ty::Bytes, idx()], Ty::Bool, |_, mut b| {
        let (t, i) = (b.arg(0), b.arg(1));
        let n = b.p("bytes.len", vec![], vec![t.clone()])?;
        let z = b.int_cmp("int.eq", i.clone(), lit(0))?;
        let e = b.int_cmp("int.eq", i.clone(), n.clone())?;
        let edge = b.p("bool.or", vec![], vec![z, e])?;
        b.when(edge, |b| {
            b.ret(Atom::bool(true));
            Ok(())
        })?;
        let inside = b.int_cmp("int.lt", i.clone(), n)?;
        b.if_(inside, |_| Ok(()), |b| {
            b.ret(Atom::bool(false));
            Ok(())
        })?;
        let c = b.byte_at(t, i)?;
        let cont = between(&mut b, &c, 0x80, 0xbf)?;
        let r = b.not(cont)?;
        b.ret(r);
        Ok(b.finish())
    })
}

/// `C.encode(t)` for a context C (5.16, interp `index08::encode`): HTML
/// escapes `& < > " '`; a URL keeps the unreserved bytes and writes every
/// other byte as `%XX`; SQL quotes the text and doubles its quotes;
/// `ShellArg` and `Header` keep the text (a header with CR or LF traps,
/// F.5.7); `Json` writes a JSON string.
pub fn encode(f: &mut Fns, ctx: &str) -> R<usize> {
    let js = rt::jstr(f)?;
    let ctx = ctx.to_string();
    f.get(&format!("rt.ctx.{ctx}.encode"), vec![Ty::Bytes], Ty::Bytes, move |f, mut b| {
        let t = b.arg(0);
        match ctx.as_str() {
            "ShellArg" => {
                b.ret(t);
                return Ok(b.finish());
            }
            "Json" => {
                let r = call(&mut b, f, js, vec![t]);
                b.ret(r);
                return Ok(b.finish());
            }
            "Header" => {
                let n = b.p("bytes.len", vec![], vec![t.clone()])?;
                b.for_n(n, |b, i| {
                    let c = b.byte_at(t.clone(), i)?;
                    let cr = b.int_cmp("int.eq", c.clone(), lit(13))?;
                    let lf = b.int_cmp("int.eq", c, lit(10))?;
                    let bad = b.p("bool.or", vec![], vec![cr, lf])?;
                    b.when(bad, |b| trap(b, Atom::bytes(b"Header.encode: a header value contains CR or LF (F.5.7)")))
                })?;
                b.ret(t);
                return Ok(b.finish());
            }
            _ => {}
        }
        let n = b.p("bytes.len", vec![], vec![t.clone()])?;
        let o0 = b.p("list.new", vec![Ty::u8()], vec![])?;
        let out = b.var(u8list(), o0);
        let push = |b: &mut FB, s: &[u8]| -> R<()> {
            for c in s {
                let o = b.take(out);
                let o = b.p("list.push", vec![Ty::u8()], vec![o, lit(*c as i128)])?;
                b.set(out, o);
            }
            Ok(())
        };
        if ctx == "Sql" {
            push(&mut b, b"'")?;
        }
        let ctx2 = ctx.clone();
        b.for_n(n, |b, i| {
            let c = b.byte_at(t.clone(), i)?;
            // a chain of tests, each leaving the iteration's block when it applies
            b.open();
            let table: Vec<(u8, &[u8])> = match ctx2.as_str() {
                "Html" => vec![(b'&', b"&amp;"), (b'<', b"&lt;"), (b'>', b"&gt;"), (b'"', b"&quot;"), (b'\'', b"&#39;")],
                "Sql" => vec![(b'\'', b"''")],
                _ => vec![],
            };
            for (ch, esc) in table {
                let is = b.int_cmp("int.eq", c.clone(), lit(ch as i128))?;
                b.open();
                push(b, esc)?;
                b.brk(0);
                let th = b.close();
                b.push(Stmt::If(is, th, vec![]));
            }
            if ctx2 == "Url" {
                let d = between(b, &c, b'0', b'9')?;
                let u = between(b, &c, b'A', b'Z')?;
                let l = between(b, &c, b'a', b'z')?;
                let mut keep = b.p("bool.or", vec![], vec![d, u])?;
                keep = b.p("bool.or", vec![], vec![keep, l])?;
                for ch in *b"-._~" {
                    let e = b.int_cmp("int.eq", c.clone(), lit(ch as i128))?;
                    keep = b.p("bool.or", vec![], vec![keep, e])?;
                }
                let nk = b.not(keep)?;
                b.open();
                push(b, b"%")?;
                let hex = Atom::bytes(b"0123456789ABCDEF");
                let (q, r) = b.divmod_lit(c.clone(), 16, Ty::u8())?;
                let q = b.cast_in(q, Ty::Int(0, 15))?;
                let r = b.cast_in(r, Ty::Int(0, 15))?;
                for k in [q, r] {
                    let h = b.byte_at(hex.clone(), k)?;
                    let o = b.take(out);
                    let o = b.p("list.push", vec![Ty::u8()], vec![o, h])?;
                    b.set(out, o);
                }
                b.brk(0);
                let th = b.close();
                b.push(Stmt::If(nk, th, vec![]));
            }
            let o = b.take(out);
            let o = b.p("list.push", vec![Ty::u8()], vec![o, c])?;
            b.set(out, o);
            let chain = b.close();
            b.push(Stmt::Block(chain));
            Ok(())
        })?;
        if ctx == "Sql" {
            push(&mut b, b"'")?;
        }
        let o = b.take(out);
        let r = b.p("bytes.of_list", vec![], vec![o])?;
        b.ret(r);
        Ok(b.finish())
    })
}
