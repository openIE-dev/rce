// RCE runtime for emitted Rust (spec 10.5). Values, traps and the edge
// semantics of 6.3 as in the reference evaluator; the shape follows the
// verified emitter of T9: a uniform value type, `Result` with `?` for traps
// and refusals, checked i128 arithmetic against each integer type's range.
#![allow(dead_code, unused_mut, unused_variables, unused_parens, unreachable_code, unused_assignments, non_snake_case, clippy::all)]
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Clone)]
pub enum V {
    Int(i128, &'static str),
    F(f64),
    Bool(bool),
    Text(String),
    Unit,
    Tuple(Vec<V>),
    List(Vec<V>),
    Map(Vec<(V, V)>),
    Opt(Option<Box<V>>),
    Rec(String, Vec<(String, V)>),
    Err(Box<V>),
    Fn(String),
    Clo(Rc<dyn Fn(Vec<V>) -> R<V>>),
    Range(i128, i128, &'static str),
    F32(f64),
    Gen(Rc<dyn Fn(usize) -> (Vec<V>, bool, Option<Flow>)>),
}
#[derive(Clone)]
pub enum Flow { Fail(V), Trap(String), Stop, Ret(V) }
pub type R<T> = Result<T, Flow>;
pub fn trap<T>(m: impl Into<String>) -> R<T> { Err(Flow::Trap(m.into())) }
pub type M = HashMap<String, V>;
pub struct Call { pub v: V, pub outs: Vec<Option<V>>, pub slf: Option<V> }
#[derive(Clone)]
pub enum T { Int(&'static str), F64, F32, Opt(Box<T>), List(Box<T>), Map(Box<T>, Box<T>), Tup(Vec<T>), Union(Box<T>), Ref(Box<T>), Any }
pub fn ti(n: &'static str) -> T { T::Int(n) }
pub fn topt(t: T) -> T { T::Opt(Box::new(t)) }
pub fn tlist(t: T) -> T { T::List(Box::new(t)) }
pub fn tmap(a: T, b: T) -> T { T::Map(Box::new(a), Box::new(b)) }
pub fn tunion(t: T) -> T { T::Union(Box::new(t)) }
pub fn tref(t: T) -> T { T::Ref(Box::new(t)) }

pub fn irange(n: &str) -> (i128, i128) {
    match n {
        "i8" => (-128, 127), "i16" => (-32768, 32767), "i32" => (i32::MIN as i128, i32::MAX as i128),
        "i64" => (i64::MIN as i128, i64::MAX as i128), "u8" => (0, 255), "u16" => (0, 65535),
        "u32" => (0, u32::MAX as i128), "u64" | "usize" => (0, u64::MAX as i128),
        _ => (i64::MIN as i128, i64::MAX as i128),
    }
}
pub fn lit(v: i128) -> V { V::Int(v, "lit") }
pub fn text(s: &str) -> V { V::Text(s.to_string()) }
pub fn some(v: V) -> V { V::Opt(Some(Box::new(v))) }

fn rdebug(s: &str) -> String { format!("{s:?}") }
pub fn show(v: &V) -> String {
    let j = |xs: &[V]| xs.iter().map(show).collect::<Vec<_>>().join(", ");
    match v {
        V::Int(x, _) => x.to_string(),
        V::F(f) => f.to_string(),
        V::Bool(b) => b.to_string(),
        V::Text(t) => rdebug(t),
        V::Unit => "()".into(),
        V::Tuple(xs) => format!("({})", j(xs)),
        V::List(xs) => format!("[{}]", j(xs)),
        V::Map(ps) => format!("{{{}}}", ps.iter().map(|(a, b)| format!("{}: {}", show(a), show(b))).collect::<Vec<_>>().join(", ")),
        V::Opt(None) => "none".into(),
        V::Opt(Some(x)) => format!("some({})", show(x)),
        V::Rec(n, fs) => if fs.is_empty() { n.clone() } else { format!("{n}({})", fs.iter().map(|(a, b)| format!("{a}: {}", show(b))).collect::<Vec<_>>().join(", ")) },
        V::Err(e) => format!("error {}", show(e)),
        V::Fn(n) => n.clone(),
        V::Clo(_) => "<fn>".into(),
        V::Range(a, b, _) => format!("{a}..{b}"),
        V::F32(f) => (*f as f32).to_string(),
        V::Gen(_) => "<iter>".into(),
    }
}
pub fn eq(a: &V, b: &V) -> bool {
    use V::*;
    match (a, b) {
        (Int(x, _), Int(y, _)) => x == y,
        (F(x), F(y)) | (F32(x), F32(y)) => x == y,
        (Bool(x), Bool(y)) => x == y,
        (Text(x), Text(y)) => x == y,
        (Unit, Unit) => true,
        (Tuple(x), Tuple(y)) | (List(x), List(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| eq(p, q)),
        (Map(x), Map(y)) => x.len() == y.len() && x.iter().all(|(k, v)| y.iter().any(|(k2, v2)| eq(k, k2) && eq(v, v2))),
        (Opt(None), Opt(None)) => true,
        (Opt(Some(x)), Opt(Some(y))) => eq(x, y),
        (Rec(n, x), Rec(m, y)) => n == m && x.len() == y.len() && x.iter().zip(y).all(|((p, u), (q, w))| p == q && eq(u, w)),
        (Err(x), Err(y)) => eq(x, y),
        _ => false,
    }
}
pub fn cmp(a: &V, b: &V) -> R<std::cmp::Ordering> {
    use V::*;
    Ok(match (a, b) {
        (Int(x, _), Int(y, _)) => x.cmp(y),
        (F(x), F(y)) | (F32(x), F32(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
        (Text(x), Text(y)) => x.cmp(y),
        (Bool(x), Bool(y)) => x.cmp(y),
        (Tuple(x), Tuple(y)) | (List(x), List(y)) => {
            for (p, q) in x.iter().zip(y) {
                let c = cmp(p, q)?;
                if c.is_ne() { return Ok(c); }
            }
            x.len().cmp(&y.len())
        }
        _ => return trap("values are not ordered"),
    })
}
pub fn truthy(v: &V) -> R<bool> {
    match v { V::Bool(b) => Ok(*b), o => trap(format!("expected bool, found {}", show(o))) }
}
pub fn chk(x: i128, t: &'static str) -> R<V> {
    let tt = if t == "lit" { "i64" } else { t };
    let (lo, hi) = irange(tt);
    if x < lo || x > hi { trap(format!("overflow: {x} is outside {tt}")) } else { Ok(V::Int(x, t)) }
}
fn wide_mul(x: i128, y: i128) -> String {
    let m = x.unsigned_abs().wrapping_mul(y.unsigned_abs());
    if (x < 0) != (y < 0) && m != 0 { format!("-{m}") } else { m.to_string() }
}
fn ck(r: Option<i128>, t: &'static str) -> R<V> {
    match r { Some(x) => chk(x, t), None => trap(format!("overflow: i128 is outside {}", if t == "lit" { "i64" } else { t })) }
}
pub fn arith(op: &str, a: V, b: V) -> R<V> {
    use V::*;
    match (a, b) {
        (Int(x, s), Int(y, t)) => {
            let ty = if s == "lit" { t } else { s };
            let bits: u32 = match ty { "i8" | "u8" => 8, "i16" | "u16" => 16, "i32" | "u32" => 32, _ => 64 };
            match op {
                "+" => ck(x.checked_add(y), ty),
                "-" => ck(x.checked_sub(y), ty),
                "*" => match x.checked_mul(y) { Some(p) => chk(p, ty), None => trap(format!("overflow: {} is outside {}", wide_mul(x, y), if ty == "lit" { "i64" } else { ty })) },
                "/" => if y == 0 { trap("division by zero") } else { chk(x / y, ty) },
                "%" => if y == 0 { trap("division by zero") } else { chk(x % y, ty) },
                "&" => chk(x & y, ty),
                "|" => chk(x | y, ty),
                "^" => chk(x ^ y, ty),
                "<<" => {
                    if y < 0 || y >= bits as i128 { return trap("shift amount out of range"); }
                    let r = (x << y) & ((1i128 << bits) - 1);
                    chk(if ty.starts_with('i') && r >= 1i128 << (bits - 1) { r - (1i128 << bits) } else { r }, ty)
                }
                ">>" => {
                    if y < 0 || y >= bits as i128 { return trap("shift amount out of range"); }
                    chk(x >> y, ty)
                }
                _ => trap(format!("bad operator {op}")),
            }
        }
        (F(x), F(y)) => {
            let r = match op { "+" => x + y, "-" => x - y, "*" => x * y, "/" => x / y, _ => return trap("bad float operator") };
            if r.is_finite() { Ok(F(r)) } else { trap("non-finite float") }
        }
        (F32(x), F32(y)) => {
            let r = match op { "+" => x + y, "-" => x - y, "*" => x * y, "/" => x / y, _ => return trap("bad float operator") };
            let r = r as f32;
            if r.is_finite() { Ok(F32(r as f64)) } else { trap("non-finite float") }
        }
        (List(xs), List(ys)) if matches!(op, "+" | "-" | "*" | "/" | "%") => {
            if xs.len() != ys.len() { return trap("lane count mismatch"); }
            Ok(List(xs.into_iter().zip(ys).map(|(x, y)| arith(op, x, y)).collect::<R<_>>()?))
        }
        (F(x), Int(y, _)) => arith(op, F(x), F(y as f64)),
        (Int(x, _), F(y)) => arith(op, F(x as f64), F(y)),
        (Text(x), Text(y)) if op == "+" => Ok(Text(x + &y)),
        (a, b) => trap(format!("cannot apply {op} to {} and {}", show(&a), show(&b))),
    }
}
pub fn contains(c: &V, x: &V) -> R<bool> {
    Ok(match c {
        V::List(v) | V::Tuple(v) => v.iter().any(|y| eq(y, x)),
        V::Map(m) => m.iter().any(|(k, _)| eq(k, x)),
        V::Text(t) => match x { V::Text(s) => t.contains(s.as_str()), _ => false },
        V::Range(a, b, _) => match x { V::Int(v, _) => a <= v && v < b, _ => false },
        _ => return trap("`in` needs a collection"),
    })
}
pub fn binop(op: &str, x: V, y: V) -> R<V> {
    use std::cmp::Ordering::*;
    Ok(match op {
        "==" => V::Bool(eq(&x, &y)),
        "!=" => V::Bool(!eq(&x, &y)),
        "<" => V::Bool(cmp(&x, &y)? == Less),
        "<=" => V::Bool(cmp(&x, &y)? != Greater),
        ">" => V::Bool(cmp(&x, &y)? == Greater),
        ">=" => V::Bool(cmp(&x, &y)? != Less),
        "in" => V::Bool(contains(&y, &x)?),
        "not in" => V::Bool(!contains(&y, &x)?),
        ".." | "..=" => match (x, y) {
            (V::Int(p, s), V::Int(q, t)) => V::Range(p, if op == ".." { q } else { q + 1 }, if s == "lit" { t } else { s }),
            _ => return trap("bad range"),
        },
        _ => return arith(op, x, y),
    })
}
pub fn map_set(m: &mut Vec<(V, V)>, k: V, v: V) {
    if let Some(e) = m.iter_mut().find(|(a, _)| eq(a, &k)) { e.1 = v; } else { m.push((k, v)); }
}
pub fn mk_map(ps: Vec<(V, V)>) -> R<V> {
    let mut m = vec![];
    for (k, v) in ps { map_set(&mut m, k, v); }
    Ok(V::Map(m))
}
pub fn list_repeat(x: V, n: V) -> R<V> {
    match n { V::Int(k, _) if (0..=10_000_000).contains(&k) => Ok(V::List(vec![x; k as usize])), _ => trap("bad repeat count") }
}
pub fn index(c: V, i: V) -> R<V> {
    match (c, i) {
        (V::List(v), V::Int(k, _)) => if k >= 0 && (k as usize) < v.len() { Ok(v[k as usize].clone()) } else { trap(format!("index {k} out of range")) },
        (V::Text(t), V::Int(k, _)) => if k >= 0 && (k as usize) < t.len() { Ok(V::Int(t.as_bytes()[k as usize] as i128, "u8")) } else { trap(format!("index {k} out of range")) },
        (V::Map(m), k) => m.iter().find(|(a, _)| eq(a, &k)).map(|(_, v)| v.clone()).ok_or(Flow::Trap(format!("missing key {}", show(&k)))),
        (c, _) => trap(format!("cannot index {}", show(&c))),
    }
}
pub fn slice_check(c: &V) -> R<()> {
    match c { V::List(_) | V::Text(_) => Ok(()), o => trap(format!("cannot slice {}", show(o))) }
}
pub fn slice(c: V, a: Option<V>, b: Option<V>) -> R<V> {
    let (xs, text) = match c {
        V::List(xs) => (xs, None),
        V::Text(t) => (t.bytes().map(|b| V::Int(b as i128, "u8")).collect::<Vec<_>>(), Some(t)),
        o => return trap(format!("cannot slice {}", show(&o))),
    };
    let n = xs.len() as i128;
    let norm = |v: Option<V>, d: i128| -> R<i128> {
        match v {
            None => Ok(d),
            Some(V::Int(k, _)) => { let k = if k < 0 { n + k } else { k }; if k < 0 || k > n { trap("slice index out of range") } else { Ok(k) } }
            _ => trap("bad slice bound"),
        }
    };
    let (lo, hi) = (norm(a, 0)?, norm(b, n)?);
    if lo > hi { return trap("slice bounds reversed"); }
    if let Some(t) = text {
        let (lo, hi) = (lo as usize, hi as usize);
        if !t.is_char_boundary(lo) || !t.is_char_boundary(hi) { return trap("not a char boundary"); }
        return Ok(V::Text(t[lo..hi].to_string()));
    }
    Ok(V::List(xs[lo as usize..hi as usize].to_vec()))
}
pub fn field(v: V, m: &str) -> R<V> {
    match &v {
        V::Rec(_, fs) => fs.iter().find(|(n, _)| n == m).map(|(_, x)| x.clone()).ok_or(Flow::Trap(format!("no field {m}"))),
        _ => trap(format!("{} has no field {m}", show(&v))),
    }
}
pub fn tmember(v: V, i: usize) -> R<V> {
    match v { V::Tuple(v) if i < v.len() => Ok(v[i].clone()), _ => trap("bad tuple index") }
}
pub fn neg(v: V) -> R<V> {
    match v { V::Int(x, t) => chk(-x, t), V::F(f) => Ok(V::F(-f)), V::F32(f) => Ok(V::F32(-f)), _ => trap("bad negation") }
}
pub fn bitnot(v: V) -> R<V> {
    match v {
        V::Int(x, t) => { let (lo, hi) = irange(if t == "lit" { "i64" } else { t }); Ok(V::Int(if lo == 0 { hi - x } else { !x }, t)) }
        _ => trap("bad ~"),
    }
}
pub fn not_(v: V) -> R<V> { Ok(V::Bool(!truthy(&v)?)) }
pub fn cast(v: V, t: T) -> R<V> {
    let to_f32 = matches!(t, T::F32);
    let n = if let T::Int(n) = t { Some(n) } else { None };
    match (v, n) {
        (V::Int(v, _), None) if to_f32 => Ok(V::F32(v as f32 as f64)),
        (V::F(f), None) if to_f32 => { let r = f as f32; if r.is_finite() { Ok(V::F32(r as f64)) } else { trap("non-finite float") } }
        (V::F32(f), None) if !to_f32 => Ok(V::F(f)),
        (V::F32(f), Some(n)) => chk(f.trunc() as i128, n),
        (V::Int(v, _), Some(n)) => chk(v, n),
        (V::F(f), Some(n)) => { if !f.is_finite() { return trap("non-finite float"); } chk(f.trunc() as i128, n) }
        (V::Bool(b), Some(n)) => chk(b as i128, n),
        (V::Int(v, _), None) => Ok(V::F(v as f64)),
        (o, _) => Ok(o),
    }
}
pub fn fit(v: V, t: &T) -> R<V> {
    match (v, t) {
        (V::Int(x, _), T::Int(n)) => { let (lo, hi) = irange(n); if x < lo || x > hi { return trap(format!("overflow: {x} is outside {n}")); } Ok(V::Int(x, n)) }
        (V::Int(x, _), T::F64) => Ok(V::F(x as f64)),
        (V::Int(x, _), T::F32) => Ok(V::F32(x as f32 as f64)),
        (V::Opt(Some(x)), T::Opt(i)) => Ok(V::Opt(Some(Box::new(fit(*x, i)?)))),
        (V::List(xs), T::List(a)) => Ok(V::List(xs.into_iter().map(|x| fit(x, a)).collect::<R<_>>()?)),
        (V::Map(xs), T::Map(a, b)) => Ok(V::Map(xs.into_iter().map(|(k, v)| Ok((fit(k, a)?, fit(v, b)?))).collect::<R<_>>()?)),
        (V::Tuple(xs), T::Tup(ts)) if xs.len() == ts.len() => Ok(V::Tuple(xs.into_iter().zip(ts).map(|(x, t)| fit(x, t)).collect::<R<_>>()?)),
        (V::Err(e), _) => Ok(V::Err(e)),
        (v, T::Union(x)) | (v, T::Ref(x)) => fit(v, x),
        (v, _) => Ok(v),
    }
}
pub fn fitall(a: Vec<V>, ts: &[T]) -> R<Vec<V>> { a.into_iter().zip(ts).map(|(x, t)| fit(x, t)).collect() }
pub fn arg(a: &[V], i: usize) -> V { a.get(i).cloned().unwrap_or(V::Unit) }

fn sort_list(mut xs: Vec<V>) -> R<Vec<V>> {
    let mut err = None;
    xs.sort_by(|p, q| cmp(p, q).unwrap_or_else(|e| { if err.is_none() { err = Some(e); } std::cmp::Ordering::Equal }));
    match err { Some(e) => Err(e), None => Ok(xs) }
}
pub fn apply(f: &V, vals: Vec<V>) -> R<V> {
    match f { V::Clo(c) => c(vals), V::Fn(n) => raw_fn(n, vals), o => trap(format!("{} is not callable", show(o))) }
}
pub fn call_value(f: V, vals: Vec<V>) -> R<V> {
    match f { V::Fn(n) => Ok(user_fn(&n, None, vals)?.v), V::Clo(c) => c(vals), o => trap(format!("{} is not callable", show(&o))) }
}
pub fn builtin(r: V, m: &str, a: Vec<V>) -> R<(V, Option<V>)> {
    use V::*;
    let a0 = || a.first().cloned().unwrap_or(Unit);
    let v = match (r, m) {
        (Int(x, t), "checked_add" | "checked_sub" | "checked_mul" | "checked_div") => {
            let op = match m { "checked_add" => "+", "checked_sub" => "-", "checked_mul" => "*", _ => "/" };
            match arith(op, Int(x, t), a0()) { Ok(v) => some(v), Result::Err(Flow::Trap(_)) => Opt(None), Result::Err(f) => return Result::Err(f) }
        }
        (Int(x, t), "wrapping_add" | "wrapping_sub" | "wrapping_mul" | "saturating_add" | "saturating_sub" | "saturating_mul") => {
            let y = match a0() { Int(y, _) => y, _ => return trap(m) };
            let r = match &m[m.len() - 3..] { "add" => x + y, "sub" => x - y, _ if m.starts_with("wrapping") => x.wrapping_mul(y), _ => x.checked_mul(y).unwrap_or(if (x < 0) != (y < 0) { i128::MIN } else { i128::MAX }) };
            let (lo, hi) = irange(t);
            Int(if m.starts_with("wrapping") { r.wrapping_sub(lo).rem_euclid(hi - lo + 1) + lo } else { r.clamp(lo, hi) }, t)
        }
        (Int(x, _), "tostr") => Text(x.to_string()),
        (Int(x, t), "abs") => chk(x.abs(), t)?,
        (Int(x, t), "min") => match a0() { Int(y, _) => Int(x.min(y), t), _ => return trap("min") },
        (Int(x, t), "max") => match a0() { Int(y, _) => Int(x.max(y), t), _ => return trap("max") },
        (F(x), "sqrt") => if x < 0.0 { return trap("non-finite float") } else { F(x.sqrt()) },
        (F(x), "abs") => F(x.abs()),
        (F(x), "floor") => F(x.floor()),
        (F(x), "ceil") => F(x.ceil()),
        (F(x), "round") => F(x.round()),
        (F(x), "tostr") => Text(x.to_string()),
        (Bool(b), "tostr") => Text(b.to_string()),
        (Text(s), "len") => Int(s.len() as i128, "usize"),
        (F32(x), "abs") => F32(x.abs()),
        (F32(x), "tostr") => Text((x as f32).to_string()),
        (Gen(g), "collect") => { let (items, _, err) = g(usize::MAX); if let Some(f) = err { return Result::Err(f); } List(items) }
        (Text(s), "strip") => Text(s.trim().into()),
        (Text(s), "upper") => Text(s.to_ascii_uppercase()),
        (Text(s), "lower") => Text(s.to_ascii_lowercase()),
        (Text(s), "reverse") => Text(s.chars().rev().collect()),
        (Text(s), "tostr") => Text(s),
        (Text(s), "chars") => List(s.chars().map(|c| Text(c.to_string())).collect()),
        (Text(s), "words") => List(s.split_whitespace().map(|w| Text(w.into())).collect()),
        (Text(s), "lines") => List(s.lines().map(|w| Text(w.into())).collect()),
        (Text(s), "split") => match a0() { Text(d) => List(s.split(d.as_str()).map(|w| Text(w.into())).collect()), _ => return trap("split") },
        (Text(s), "isdigit") => Bool(!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())),
        (Text(s), "isalpha") => Bool(!s.is_empty() && s.chars().all(|c| c.is_ascii_alphabetic())),
        (Text(s), "isspace") => Bool(!s.is_empty() && s.chars().all(|c| c.is_ascii_whitespace())),
        (Text(s), "isempty") => Bool(s.is_empty()),
        (Text(s), "contains") => match a0() { Text(d) => Bool(s.contains(d.as_str())), _ => return trap("contains") },
        (Text(s), "startswith") => match a0() { Text(d) => Bool(s.starts_with(d.as_str())), _ => return trap("startswith") },
        (Text(s), "endswith") => match a0() { Text(d) => Bool(s.ends_with(d.as_str())), _ => return trap("endswith") },
        (Text(s), "toint") => s.trim().parse::<i128>().map(|v| Int(v, "i64")).map_err(|_| Flow::Trap(format!("not an integer: {s:?}")))?,
        (Text(s), "join") => match a0() { List(xs) => Text(xs.iter().map(|x| if let Text(t) = x { t.clone() } else { show(x) }).collect::<Vec<_>>().join(&s)), _ => return trap("join") },
        (List(xs), "len") => Int(xs.len() as i128, "usize"),
        (List(xs), "isempty") => Bool(xs.is_empty()),
        (List(mut xs), "append" | "push") => { xs.push(a0()); return Ok((Unit, Some(List(xs)))); }
        (List(mut xs), "pop") => { let r = xs.pop(); return Ok((Opt(r.map(Box::new)), Some(List(xs)))); }
        (List(_), "clear") => return Ok((Unit, Some(List(vec![])))),
        (List(xs), "sort") => return Ok((Unit, Some(List(sort_list(xs)?)))),
        (List(mut xs), "with") => { xs.push(a0()); List(xs) }
        (List(mut xs), "reverse") => { xs.reverse(); List(xs) }
        (List(xs), "contains") => { let y = a0(); Bool(xs.iter().any(|x| eq(x, &y))) }
        (List(xs), "get") => match a0() { Int(i, _) => Opt(if i >= 0 { xs.get(i as usize).cloned().map(Box::new) } else { None }), _ => return trap("get") },
        (List(xs), "sum") => { let mut acc = Int(0, "lit"); for x in xs { acc = arith("+", acc, x)?; } acc }
        (List(xs), "map") => { let f = a0(); let mut out = vec![]; for x in xs { out.push(apply(&f, vec![x])?); } List(out) }
        (List(xs), "filter") => { let f = a0(); let mut out = vec![]; for x in xs { if matches!(apply(&f, vec![x.clone()])?, Bool(true)) { out.push(x); } } List(out) }
        (List(xs), "fold") => { let f = a.get(1).cloned().unwrap_or(Unit); let mut acc = a0(); for x in xs { acc = apply(&f, vec![acc, x])?; } acc }
        (Map(mm), "len") => Int(mm.len() as i128, "usize"),
        (Map(mm), "get") => { let k = a0(); match mm.iter().find(|(x, _)| eq(x, &k)) { Some((_, v)) => v.clone(), None => Err(Box::new(Rec("Missing".into(), vec![]))) } }
        (Map(mm), "contains") => { let k = a0(); Bool(mm.iter().any(|(x, _)| eq(x, &k))) }
        (Map(mut mm), "insert") => { map_set(&mut mm, a0(), a.get(1).cloned().unwrap_or(Unit)); return Ok((Unit, Some(Map(mm)))); }
        (Map(mut mm), "remove") => { let k = a0(); let i = mm.iter().position(|(x, _)| eq(x, &k)); let r = i.map(|i| mm.remove(i).1); return Ok((Opt(r.map(Box::new)), Some(Map(mm)))); }
        (Map(mm), "keys") => List(mm.into_iter().map(|x| x.0).collect()),
        (Map(mm), "values") => List(mm.into_iter().map(|x| x.1).collect()),
        (Opt(x), "isnone") => Bool(x.is_none()),
        (Opt(x), "issome") => Bool(x.is_some()),
        (Opt(x), "or") => match x { Some(v) => *v, None => a0() },
        (r, _) => return trap(format!("{} has no method {m} at comptime", show(&r))),
    };
    Ok((v, None))
}
pub fn call_method(recv: V, m: &str, args: Vec<V>) -> R<Call> {
    let head = match &recv { V::Rec(n, _) => Some(kind_of(n).to_string()), V::List(_) => Some("List".to_string()), _ => None };
    if let Some(h) = head && let Some(f) = user_method(&h, m) {
        return f(Some(recv), args);
    }
    let (v, s) = builtin(recv, m, args)?;
    Ok(Call { v, outs: vec![], slf: s })
}
pub type Init = Option<Rc<dyn Fn() -> R<V>>>;
pub fn construct(name: &str, fields: Vec<(&str, T, Init)>, args: Vec<(Option<&str>, V)>) -> R<V> {
    let n = fields.len();
    let mut vals: Vec<Option<V>> = vec![None; n];
    for (i, (lb, v)) in args.into_iter().enumerate() {
        let k = match lb { Some(l) => fields.iter().position(|f| f.0 == l).unwrap_or(i), None => i };
        if k < n { vals[k] = Some(fit(v, &fields[k].1)?); }
    }
    let mut out = vec![];
    for (k, (f, t, init)) in fields.into_iter().enumerate() {
        let v = match vals[k].take() {
            Some(v) => v,
            None => match init { Some(e) => fit(e()?, &t)?, None => return trap(format!("missing field {f}")) },
        };
        out.push((f.to_string(), v));
    }
    Ok(V::Rec(name.to_string(), out))
}
pub type Th = Rc<dyn Fn() -> R<V>>;
#[derive(Clone)]
pub enum P { Wild, Bind(&'static str), Tag(&'static str), Some(Box<P>), Ctor(&'static str, Vec<(Option<&'static str>, P)>), Lit(Th), Range(Th, bool, Th), Tup(Vec<P>), Alt(Vec<P>) }
pub fn pmatch(p: &P, v: &V, b: &mut M) -> R<bool> {
    Ok(match p {
        P::Wild => true,
        P::Bind(n) => { b.insert(n.to_string(), v.clone()); true }
        P::Tag(n) => matches!(v, V::Rec(m, _) if m == n),
        P::Some(q) => match v { V::Opt(Some(x)) => pmatch(q, x, b)?, _ => false },
        P::Ctor(n, args) => match v {
            V::Rec(m, fs) if m == n => {
                for (i, (lb, q)) in args.iter().enumerate() {
                    let fv = match lb { Some(l) => fs.iter().find(|(f, _)| f == l).map(|x| x.1.clone()), None => fs.get(i).map(|x| x.1.clone()) };
                    match fv { Some(fv) => if !pmatch(q, &fv, b)? { return Ok(false); }, None => return Ok(false) }
                }
                true
            }
            _ => false,
        },
        P::Lit(e) => eq(&e()?, v),
        P::Range(a, incl, c) => {
            let (x, y) = (a()?, c()?);
            let lo = cmp(v, &x)?.is_ge();
            let hi = if *incl { cmp(v, &y)?.is_le() } else { cmp(v, &y)?.is_lt() };
            lo && hi
        }
        P::Tup(ps) => match v {
            V::Tuple(vs) if vs.len() == ps.len() => { for (q, x) in ps.iter().zip(vs) { if !pmatch(q, x, b)? { return Ok(false); } } true }
            _ => false,
        },
        P::Alt(ps) => { for q in ps { if pmatch(q, v, b)? { return Ok(true); } } false }
    })
}
pub fn pm(p: &P, v: &V) -> R<Option<M>> { let mut b = M::new(); Ok(if pmatch(p, v, &mut b)? { Some(b) } else { None }) }
pub fn mget(m: &M, n: &str) -> V { m.get(n).cloned().unwrap_or(V::Unit) }

thread_local! {
    static YIELDS: RefCell<Vec<(Vec<V>, usize)>> = const { RefCell::new(vec![]) };
    static OUT: RefCell<Vec<V>> = const { RefCell::new(vec![]) };
}
pub fn yield_(v: V) -> R<()> {
    YIELDS.with(|y| {
        let mut y = y.borrow_mut();
        let Some((buf, cap)) = y.last_mut() else { return trap("yield outside an iterator") };
        if buf.len() as i128 >= LOOP_LIMIT { return trap("loop bound"); }
        buf.push(v);
        if buf.len() >= *cap { return Err(Flow::Stop); }
        Ok(())
    })
}
pub fn gen_run(body: &dyn Fn() -> R<()>, cap: usize) -> (Vec<V>, bool, Option<Flow>) {
    YIELDS.with(|y| y.borrow_mut().push((vec![], cap)));
    let r = body();
    let items = YIELDS.with(|y| y.borrow_mut().pop().unwrap().0);
    match r { Ok(()) | Err(Flow::Ret(_)) => (items, true, None), Err(Flow::Stop) => (items, false, None), Err(f) => (items, true, Some(f)) }
}
pub fn iter_items(v: V) -> R<Vec<V>> {
    Ok(match v {
        V::List(xs) => xs,
        V::Range(a, b, t) => { if b - a > LOOP_LIMIT * 10 { return trap("loop bound"); } (a..b.max(a)).map(|i| V::Int(i, t)).collect() }
        V::Text(s) => s.chars().map(|c| V::Text(c.to_string())).collect(),
        V::Map(m) => m.into_iter().map(|(k, v)| V::Tuple(vec![k, v])).collect(),
        V::Gen(g) => { let (items, _, err) = g(usize::MAX); if let Some(f) = err { return Err(f); } items }
        o => return trap(format!("cannot iterate over {}", show(&o))),
    })
}
pub enum It { L(std::vec::IntoIter<V>), G { g: Rc<dyn Fn(usize) -> (Vec<V>, bool, Option<Flow>)>, done: usize, cap: usize, buf: std::vec::IntoIter<V>, fin: bool, err: Option<Flow>, started: bool } }
pub fn iterate(v: V) -> R<It> {
    match v { V::Gen(g) => Ok(It::G { g, done: 0, cap: 64, buf: vec![].into_iter(), fin: false, err: None, started: false }), o => Ok(It::L(iter_items(o)?.into_iter())) }
}
impl It {
    pub fn next(&mut self) -> R<Option<V>> {
        match self {
            It::L(i) => Ok(i.next()),
            It::G { g, done, cap, buf, fin, err, started } => loop {
                if let Some(x) = buf.next() { *done += 1; return Ok(Some(x)); }
                if *started {
                    if let Some(f) = err.take() { return Err(f); }
                    if *fin { return Ok(None); }
                    *cap = cap.saturating_mul(2);
                }
                *started = true;
                let (items, f, e) = g(*cap);
                *buf = items.into_iter().skip(*done).collect::<Vec<_>>().into_iter();
                *fin = f;
                *err = e;
            },
        }
    }
}
pub fn listcomp(itv: V, p: P, body: &dyn Fn(&M) -> R<V>, cond: Option<&dyn Fn(&M) -> R<V>>) -> R<V> {
    let mut out = vec![];
    for x in iter_items(itv)? {
        let mut b = M::new();
        if !pmatch(&p, &x, &mut b)? { continue; }
        if let Some(c) = cond && !truthy(&c(&b)?)? { continue; }
        out.push(body(&b)?);
    }
    Ok(V::List(out))
}
pub fn try_(x: V, els: Option<&dyn Fn() -> R<V>>) -> R<V> {
    match x {
        V::Err(e) => Err(Flow::Fail(match els { Some(f) => f()?, None => *e })),
        V::Opt(None) => match els { Some(f) => Err(Flow::Fail(f()?)), None => trap("try on none") },
        V::Opt(Some(v)) => Ok(*v),
        v => Ok(v),
    }
}
pub fn compound(op: &str, cur: V, r: V) -> R<V> {
    let t = if let V::Int(_, t) = cur { t } else { "lit" };
    let x = arith(op, cur, r)?;
    if let V::Int(x, _) = x { chk(x, t) } else { Ok(x) }
}
pub fn fixtag(cur: Option<V>, v: V) -> R<V> {
    match (cur, v) { (Some(V::Int(_, t)), V::Int(x, _)) if t != "lit" => chk(x, t), (_, v) => Ok(v) }
}
pub fn set_field(rec: V, m: &str, v: V) -> R<V> {
    match rec {
        V::Rec(n, mut fs) => { match fs.iter_mut().find(|(f, _)| f == m) { Some(f) => f.1 = v, None => return trap("no field") } Ok(V::Rec(n, fs)) }
        _ => trap("member assignment on a non-record"),
    }
}
pub fn set_tuple(t: V, i: usize, v: V) -> R<V> {
    match t { V::Tuple(mut t) if i < t.len() => { t[i] = v; Ok(V::Tuple(t)) } _ => trap("bad tuple assignment") }
}
pub fn set_index(c: V, k: V, v: V) -> R<V> {
    match (c, k) {
        (V::List(mut xs), V::Int(i, _)) => { if i < 0 || i as usize >= xs.len() { return trap(format!("index {i} out of range")); } xs[i as usize] = v; Ok(V::List(xs)) }
        (V::Map(mut m), k) => { map_set(&mut m, k, v); Ok(V::Map(m)) }
        _ => trap("bad indexed assignment"),
    }
}
pub fn loop_bound(v: V) -> i128 { if let V::Int(n, _) = v { n } else { LOOP_LIMIT } }
pub fn arena_bound(v: V) -> R<i128> { if let V::Int(n, _) = v { Ok(n) } else { trap("bad arena bound") } }
pub fn check_(v: V) -> R<()> { if !truthy(&v)? { return trap("check failed"); } Ok(()) }
pub fn json_size(v: &V) -> usize {
    fn esc(s: &str) -> usize { 2 + s.chars().map(|c| match c { '"' | '\\' | '\n' | '\r' | '\t' => 2, c if (c as u32) < 0x20 => 6, c => c.len_utf8() }).sum::<usize>() }
    let seq = |xs: &mut dyn Iterator<Item = usize>, n: usize| 2 + xs.sum::<usize>() + n.saturating_sub(1);
    match v {
        V::Int(x, _) => x.to_string().len(),
        V::F(f) | V::F32(f) => format!("{f:?}").len(),
        V::Bool(b) => if *b { 4 } else { 5 },
        V::Text(s) => esc(s),
        V::Unit | V::Opt(None) => 4,
        V::Opt(Some(x)) | V::Err(x) => json_size(x),
        V::Tuple(xs) | V::List(xs) => seq(&mut xs.iter().map(json_size), xs.len()),
        V::Map(m) => seq(&mut m.iter().map(|(k, v)| 3 + json_size(k) + json_size(v)), m.len()),
        V::Rec(n, fs) => 2 + esc(n) + 1 + seq(&mut fs.iter().map(|(a, b)| esc(a) + 1 + json_size(b)), fs.len()),
        V::Range(a, b, _) => 3 + a.to_string().len() + b.to_string().len(),
        V::Fn(n) => esc(n),
        V::Clo(_) | V::Gen(_) => 2,
    }
}
pub fn arena_check(bound: i128, vals: &[&V]) -> R<()> {
    let used: i128 = vals.iter().map(|v| json_size(v) as i128).sum();
    if used > bound { trap("size limit") } else { Ok(()) }
}
pub fn why(f: &Flow) -> String {
    match f { Flow::Trap(t) => format!("trap: {t}"), Flow::Fail(v) => format!("failed with {}", show(v)), Flow::Stop => "Stop".into(), Flow::Ret(_) => "Ret".into() }
}
fn refusal(f: &Flow) -> String {
    match f {
        Flow::Fail(V::Rec(n, _)) => n.clone(),
        Flow::Fail(_) => "?".into(),
        Flow::Trap(t) => format!("trap({t})"),
        Flow::Stop => "trap(Stop)".into(),
        Flow::Ret(_) => "trap(Ret)".into(),
    }
}
pub fn emit(v: V) -> R<()> { OUT.with(|o| o.borrow_mut().push(v)); Ok(()) }
fn take_out() -> Vec<V> { OUT.with(|o| std::mem::take(&mut *o.borrow_mut())) }
pub fn obs(label: &str, th: &dyn Fn() -> R<V>) {
    match th() { Ok(v) => println!("{label} = {}", show(&v)), Err(f) => println!("{label} {}", why(&f)) }
}
pub type State = Vec<(&'static str, Vec<(&'static str, V)>)>;
fn show_state(s: &State) -> String {
    s.iter().map(|(r, fs)| format!("{r} {{ {} }}", fs.iter().map(|(n, v)| format!("{n} = {}", show(v))).collect::<Vec<_>>().join(", "))).collect::<Vec<_>>().join("; ")
}
pub fn delta(s: &State, ev: &V) -> Result<(State, Vec<V>), String> {
    let mut next = vec![];
    let mut outs = vec![];
    for (name, fs) in s {
        let mut got = None;
        for st in steps_of(name) {
            take_out();
            match st(fs, ev) {
                Ok(Some(r)) => { got = Some((r, take_out())); break; }
                Ok(None) => {}
                Err(f) => return Err(refusal(&f)),
            }
        }
        match got { Some((r, o)) => { next.push((*name, r)); outs.extend(o); } None => next.push((*name, fs.clone())) }
    }
    Ok((next, outs))
}
pub type Ev = (&'static str, Rc<dyn Fn() -> R<Option<V>>>);
pub fn run_example(line: usize, events: Vec<Ev>) {
    println!("example {line}");
    let mut s = match initial_state() { Ok(s) => s, Err(f) => { println!("initial {}", why(&f)); return; } };
    println!("0 (initial) {}", show_state(&s));
    for (k, (name, th)) in events.iter().enumerate() {
        let k = k + 1;
        let ev = match th() { Ok(Some(v)) => v, Ok(None) => { println!("{k} {name} not an event"); return; } Err(f) => { println!("{k} {name} {}", why(&f)); return; } };
        match delta(&s, &ev) {
            Ok((s2, outs)) => {
                s = s2;
                println!("{k} {} {}", show(&ev), show_state(&s));
                for o in outs { println!("  emit {}", show(&o)); }
            }
            Err(r) => { println!("{k} {} refused: {r}", show(&ev)); return; }
        }
    }
}
pub fn run_main(f: &dyn Fn(Option<V>, Vec<V>) -> R<Call>) {
    println!("main");
    take_out();
    let r = f(None, vec![]);
    for o in take_out() { println!("emit {}", show(&o)); }
    match r { Ok(c) => println!("main = {}", show(&c.v)), Err(e) => println!("main {}", why(&e)) }
}
