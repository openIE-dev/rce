// RCE runtime for emitted Rust (spec 10.5). Values, traps and the edge
// semantics of 6.3 as in the reference evaluator; the shape follows the
// verified emitter of T9: a uniform value type, `Result` with `?` for traps
// and refusals, checked i128 arithmetic against each integer type's range.
#![allow(dead_code, unused_mut, unused_variables, unused_parens, unreachable_code, unused_assignments, non_snake_case, clippy::all)]
// `Rc` is `Arc` so values can move to task and branch threads
use std::sync::Arc as Rc;
use std::cell::RefCell;
use std::collections::HashMap;

/// SHA-256 (FIPS 180-4) of a byte string as 64 lowercase hex digits (5.15 `sha256`).
pub fn sha256_hex(m: &[u8]) -> String {
    const K: [u32; 64] = [0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
    let mut h: [u32; 8] = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    let mut d = m.to_vec();
    d.push(0x80);
    while d.len() % 64 != 56 { d.push(0); }
    d.extend_from_slice(&((m.len() as u64).wrapping_mul(8)).to_be_bytes());
    for c in d.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 { w[i] = u32::from_be_bytes([c[4 * i], c[4 * i + 1], c[4 * i + 2], c[4 * i + 3]]); }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut cc, mut dd, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let t1 = hh.wrapping_add(e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25)).wrapping_add((e & f) ^ (!e & g)).wrapping_add(K[i]).wrapping_add(w[i]);
            let t2 = (a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22)).wrapping_add((a & b) ^ (a & cc) ^ (b & cc));
            hh = g; g = f; f = e; e = dd.wrapping_add(t1); dd = cc; cc = b; b = a; a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, cc, dd, e, f, g, hh]) { *x = x.wrapping_add(y); }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

#[derive(Clone)]
pub enum V {
    Int(i128, &'static str),
    F(f64),
    Bool(bool),
    Text(String),
    Unit,
    Tuple(Vec<V>),
    List(Vec<V>),
    Set(Vec<V>),
    Map(Vec<(V, V)>),
    Opt(Option<Box<V>>),
    Rec(String, Vec<(String, V)>),
    Err(Box<V>),
    Fn(String),
    Clo(Rc<dyn Fn(Vec<V>) -> R<V> + Send + Sync>),
    Range(i128, i128, &'static str),
    F32(f64),
    Gen(Rc<dyn Fn(usize) -> (Vec<V>, bool, Option<Flow>) + Send + Sync>),
    /// lattice cell (draft 0.6b C.11.8): shared, so a copy names the same cell
    LVar(Rc<LCell>),
    /// channel (content-hash id, capacity) and join handle of the tick runtime
    Chan(String, i128),
    Task(String),
    /// actor (id, type) and session endpoint (out channel, in channel, capacity)
    Actor(String, String),
    Session(String, String, i128),
}
#[derive(Clone)]
pub enum Flow { Fail(V), Trap(String), Stop, Ret(V), Cancel(u64), Brk, Cont }
pub type R<T> = Result<T, Flow>;
pub fn trap<T>(m: impl Into<String>) -> R<T> { Err(Flow::Trap(m.into())) }
pub type M = HashMap<String, V>;
pub struct Call { pub v: V, pub outs: Vec<Option<V>>, pub slf: Option<V> }
#[derive(Clone)]
pub enum T { Int(&'static str), F64, F32, Opt(Box<T>), List(Box<T>), Map(Box<T>, Box<T>), Tup(Vec<T>), Union(Box<T>), Ref(Box<T>), Refn(Box<T>, &'static str, Box<RP>), Any }
/// Refined-type predicate (spec 5.1, W-REF-1): S = self, N = self.len(), L = literal.
#[derive(Clone)]
pub enum RTm { S, N, L(i128) }
#[derive(Clone)]
pub enum RP { C(&'static str, RTm, RTm), K(&'static str), A(Box<RP>, Box<RP>), O(Box<RP>, Box<RP>), Z(Box<RP>) }
/// F.5.2 `Prompt.encode`: the data in a block whose delimiter is derived from
/// the SHA-256 of the text (rehashed until the text does not contain it), as
/// `index08::prompt_encode`.
pub fn prompt_encode(s: &str) -> String {
    let mut h = sha256_hex(s.as_bytes());
    loop {
        let d = format!("data-{h}");
        if !s.contains(&d) {
            return format!("<<{d}\n{s}\n{d}>>");
        }
        let raw: Vec<u8> = (0..32).map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap()).collect();
        h = sha256_hex(&raw);
    }
}
/// Context encoders (spec 5.16).
pub fn ctx_lit(v: V) -> R<V> { charge(4)?; Ok(v) }
pub fn ctx_encode(c: &str, v: V) -> R<V> {
    charge(4i128.saturating_add(size_of(&v)))?;
    let V::Text(s) = v else { return trap("context text") };
    let mut o = String::new();
    match c {
        "Html" => for ch in s.chars() { match ch { '&' => o.push_str("&amp;"), '<' => o.push_str("&lt;"), '>' => o.push_str("&gt;"), '"' => o.push_str("&quot;"), '\'' => o.push_str("&#39;"), ch => o.push(ch) } },
        "Url" => for b in s.bytes() { if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') { o.push(b as char) } else { o.push_str(&format!("%{b:02X}")) } },
        "Sql" => { o.push('\''); o.push_str(&s.replace('\'', "''")); o.push('\''); }
        "ShellArg" => o.push_str(&s),
        "Prompt" => o.push_str(&prompt_encode(&s)),
        "Header" => { if s.contains('\r') || s.contains('\n') { return trap("Header.encode: a header value contains CR or LF (F.5.7)") } o.push_str(&s) }
        _ => { o.push('"'); for ch in s.chars() { match ch { '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), '\n' => o.push_str("\\n"), '\r' => o.push_str("\\r"), '\t' => o.push_str("\\t"), ch if (ch as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", ch as u32)), ch => o.push(ch) } } o.push('"'); }
    }
    Ok(V::Text(o))
}
pub fn trefn(t: T, n: &'static str, p: RP) -> T { T::Refn(Box::new(t), n, Box::new(p)) }

/// 6.3 N7: the whitespace of `strip`, `words` and `toint`, enumerated so that
/// no result depends on the Unicode tables of a toolchain: the White_Space
/// property of Unicode 18.0.0 (unchanged since Unicode 6.3), frozen here.
const WHITE_SPACE: [char; 25] = ['\t', '\n', '\u{0b}', '\u{0c}', '\r', ' ', '\u{85}', '\u{a0}', '\u{1680}', '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}', '\u{2006}', '\u{2007}', '\u{2008}', '\u{2009}', '\u{200a}', '\u{2028}', '\u{2029}', '\u{202f}', '\u{205f}', '\u{3000}'];
fn is_ws(c: char) -> bool {
    WHITE_SPACE.contains(&c)
}

fn ptm(t: &RTm, v: &V) -> Option<i128> { match (t, v) { (RTm::L(n), _) => Some(*n), (RTm::S, V::Int(x, _)) => Some(*x), (RTm::N, V::Text(s)) => Some(s.len() as i128), (RTm::N, V::List(x) | V::Set(x)) => Some(x.len() as i128), (RTm::N, V::Map(x)) => Some(x.len() as i128), _ => None } }
pub fn pholds(p: &RP, v: &V) -> bool {
    match p {
        RP::C(op, a, b) => match (ptm(a, v), ptm(b, v)) { (Some(x), Some(y)) => match *op { "<" => x < y, "<=" => x <= y, ">" => x > y, ">=" => x >= y, "==" => x == y, _ => x != y }, _ => false },
        RP::K(c) => match v { V::Text(s) => match *c { "isdigit" => !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()), "isalpha" => !s.is_empty() && s.chars().all(|c| c.is_ascii_alphabetic()), "isspace" => !s.is_empty() && s.chars().all(|c| c.is_ascii_whitespace()), _ => s.is_empty() }, _ => false },
        RP::A(a, b) => pholds(a, v) && pholds(b, v),
        RP::O(a, b) => pholds(a, v) || pholds(b, v),
        RP::Z(a) => !pholds(a, v),
    }
}
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

/// 6.1.1 rule 3: the digits of a float. The shortest round-trip digit count n
/// comes from `{:e}`; the digits are then the n-digit decimal closest to x,
/// and on an exact tie between two such decimals the one with the even last
/// digit (as ECMAScript `Number::toString` and Python `repr` do). Rust's `{:e}`
/// breaks that tie to the odd candidate. Returns (digits, exponent of the
/// first digit); x must be finite and nonzero, and only |x| is used.
pub fn shortest_e(x: f64) -> (String, i32) {
    fn split(e: &str) -> (String, i32) {
        let (m, ex) = e.split_once('e').unwrap();
        (m.chars().filter(|c| *c != '.').collect(), ex.parse().unwrap())
    }
    let a = x.abs();
    let (sd, sex) = split(&format!("{a:e}"));
    let n = sd.len();
    // The tie can only arise when the digit after the n-th is a 5; then test
    // exactness against the full expansion (a binary64 has at most 767 digits).
    let (d1, _) = split(&format!("{:.*e}", n, a));
    if d1.len() == n + 1 && d1.ends_with('5') {
        let (full, fex) = split(&format!("{:.*e}", 780, a));
        if full[n + 1..].bytes().all(|b| b == b'0') {
            // exact tie: keep the first n digits when the n-th is even, else add one
            let mut head: Vec<u8> = full.as_bytes()[..n].to_vec();
            if (head[n - 1] - b'0') % 2 == 1 {
                let mut i = n;
                loop {
                    if i == 0 {
                        head.insert(0, b'1');
                        head.truncate(n);
                        return (String::from_utf8(head).unwrap(), fex + 1);
                    }
                    i -= 1;
                    if head[i] == b'9' {
                        head[i] = b'0';
                    } else {
                        head[i] += 1;
                        break;
                    }
                }
            }
            return (String::from_utf8(head).unwrap(), fex);
        }
    }
    // Otherwise the closest n digits; it reads back to x unless it fell outside the
    // narrower side of the rounding interval, then the shortest string is used.
    let (cd, cex) = split(&format!("{:.*e}", n - 1, a));
    let back: f64 = format!("{}e{}", cd, cex - (n as i32 - 1)).parse().unwrap();
    if back == a { (cd, cex) } else { (sd, sex) }
}

/// Canonical float text (spec 6.3 N3).
pub fn float_text(x: f64) -> String {
    if x == 0.0 { return if x.is_sign_negative() { "-0".into() } else { "0".into() }; }
    let (digits, ex) = shortest_e(x);
    let sign = if x < 0.0 { "-" } else { "" };
    if (1e-5..1e16).contains(&x.abs()) {
        let p = ex + 1;
        let body = if p <= 0 { format!("0.{}{}", "0".repeat((-p) as usize), digits) }
            else if p as usize >= digits.len() { format!("{}{}", digits, "0".repeat(p as usize - digits.len())) }
            else { format!("{}.{}", &digits[..p as usize], &digits[p as usize..]) };
        return format!("{sign}{body}");
    }
    let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
    if ex >= 0 { format!("{sign}{m}e+{ex}") } else { format!("{sign}{m}e{ex}") }
}
fn rdebug(s: &str) -> String { format!("{s:?}") }
pub fn show(v: &V) -> String {
    let j = |xs: &[V]| xs.iter().map(show).collect::<Vec<_>>().join(", ");
    match v {
        V::Int(x, _) => x.to_string(),
        V::F(f) => float_text(*f),
        V::Bool(b) => b.to_string(),
        V::Text(t) => rdebug(t),
        V::Unit => "()".into(),
        V::Tuple(xs) => format!("({})", j(xs)),
        V::List(xs) | V::Set(xs) => format!("[{}]", j(xs)),
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
        V::LVar(c) => format!("lvar({})", show(&c.borrow().0)),
        V::Chan(..) => "<chan>".into(),
        V::Actor(_, t) => format!("<actor {t}>"),
        V::Session(..) => "<session>".into(),
        V::Task(_) => "<task>".into(),
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
        (Set(x), Set(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| keq(p, q)),
        (Map(x), Map(y)) => x.len() == y.len() && x.iter().all(|(k, v)| y.iter().any(|(k2, v2)| keq(k, k2) && eq(v, v2))),
        (Opt(None), Opt(None)) => true,
        (Opt(Some(x)), Opt(Some(y))) => eq(x, y),
        (Rec(n, x), Rec(m, y)) => n == m && x.len() == y.len() && x.iter().zip(y).all(|((p, u), (q, w))| p == q && eq(u, w)),
        (Err(x), Err(y)) => eq(x, y),
        (LVar(x), LVar(y)) => Rc::ptr_eq(x, y),
        (Chan(x, _), Chan(y, _)) | (Task(x), Task(y)) | (Actor(x, _), Actor(y, _)) => x == y,
        (Session(a, b, _), Session(c, d, _)) => a == c && b == d,
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
        (Tuple(x), Tuple(y)) | (List(x), List(y)) | (Set(x), Set(y)) => {
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
pub fn arith(op: &str, a: V, b: V) -> R<V> { arith_raw(op, a, b).and_then(size_ok) }
fn arith_raw(op: &str, a: V, b: V) -> R<V> {
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
                "%" => if y == 0 { trap("division by zero") } else { chk(x / y, ty)?; chk(x % y, ty) },
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
        V::Set(v) => v.iter().any(|y| keq(y, x)),
        V::Map(m) => m.iter().any(|(k, _)| keq(k, x)),
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
    if let Some(e) = m.iter_mut().find(|(a, _)| keq(a, &k)) { e.1 = v; } else { let i = m.partition_point(|(a, _)| canon_cmp(a, &k) == std::cmp::Ordering::Less); m.insert(i, (k, v)); }
}
pub fn mk_map(ps: Vec<(V, V)>) -> R<V> {
    let mut m = vec![];
    for (k, v) in ps { map_set(&mut m, k, v); }
    Ok(V::Map(m))
}
pub fn list_repeat(x: V, n: V) -> R<V> {
    match n { V::Int(k, _) if (0..=10_000_000).contains(&k) => { charge(4 + k)?; Ok(V::List(vec![x; k as usize])) } _ => trap("bad repeat count") }
}
pub fn index(c: V, i: V) -> R<V> {
    match (c, i) {
        (V::List(v), V::Int(k, _)) => if k >= 0 && (k as usize) < v.len() { Ok(v[k as usize].clone()) } else { trap(format!("index {k} out of range")) },
        (V::Text(t), V::Int(k, _)) => if k >= 0 && (k as usize) < t.len() { Ok(V::Int(t.as_bytes()[k as usize] as i128, "u8")) } else { trap(format!("index {k} out of range")) },
        (V::Map(m), k) => m.iter().find(|(a, _)| keq(a, &k)).map(|(_, v)| v.clone()).ok_or(Flow::Trap(format!("missing key {}", show(&k)))),
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
        (V::F32(f), Some(n)) => trunc_chk(f, n),
        (V::Int(v, _), Some(n)) => chk(v, n),
        (V::F(f), Some(n)) => { if !f.is_finite() { return trap("non-finite float"); } trunc_chk(f, n) }
        (V::Bool(b), Some(n)) => chk(b as i128, n),
        (V::Int(v, _), None) => Ok(V::F(v as f64)),
        (o, _) => Ok(o),
    }
}
pub fn fit(v: V, t: &T) -> R<V> {
    if let T::Refn(b, n, p) = t { let v = fit(v, b)?; return if pholds(p, &v) { Ok(v) } else { trap(format!("refinement: value is outside {n}")) }; }
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
    match f { V::Fn(n) => Ok({ fuel()?; user_fn(&n, None, vals)?.v }), V::Clo(c) => c(vals), o => trap(format!("{} is not callable", show(&o))) }
}
pub fn builtin(r: V, m: &str, a: Vec<V>) -> R<(V, Option<V>)> {
    let (v, w) = builtin_raw(r, m, a)?;
    Ok((size_ok(v)?, match w { Some(w) => Some(size_ok(w)?), None => None }))
}
fn builtin_raw(r: V, m: &str, a: Vec<V>) -> R<(V, Option<V>)> {
    use V::*;
    let a0 = || a.first().cloned().unwrap_or(Unit);
    let r_chan = r.clone();
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
        (F(x), "min") => match a0() { F(y) => F(if x <= y { x } else { y }), _ => return trap("min") },
        (F(x), "max") => match a0() { F(y) => F(if x >= y { x } else { y }), _ => return trap("max") },
        (F(x), "floor") => F(x.floor()),
        (F(x), "ceil") => F(x.ceil()),
        (F(x), "round") => F(x.round()),
        (F(x), "tostr") => Text(float_text(x)),
        (Bool(b), "tostr") => Text(b.to_string()),
        (Text(s), "len") | (Text(s), "len_bytes") => Int(s.len() as i128, "usize"),
        (Text(s), "len_chars") => Int(s.chars().count() as i128, "usize"),
        (F32(x), "abs") => F32(x.abs()),
        (F32(x), "tostr") => Text((x as f32).to_string()),
        (Gen(g), "collect") => { let (items, _, err) = g(usize::MAX); if let Some(f) = err { return Result::Err(f); } List(items) }
        (Text(s), "strip") => Text(s.trim_matches(is_ws).into()),
        (Text(s), "upper") => Text(s.to_ascii_uppercase()),
        (Text(s), "sha256") => Text(sha256_hex(s.as_bytes())),
        (Text(s), "lower") => Text(s.to_ascii_lowercase()),
        (Text(s), "reverse") => Text(s.chars().rev().collect()),
        (Text(s), "tostr") => Text(s),
        (Text(s), "raw") => Text(s),
        (Text(s), "chars") => List(s.chars().map(|c| Text(c.to_string())).collect()),
        (Text(s), "words") => List(s.split(is_ws).filter(|w| !w.is_empty()).map(|w| Text(w.into())).collect()),
        (Text(s), "lines") => List(s.lines().map(|w| Text(w.into())).collect()),
        (Text(s), "split") => match a0() { Text(d) => List(s.split(d.as_str()).map(|w| Text(w.into())).collect()), _ => return trap("split") },
        (Text(s), "isdigit") => Bool(!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())),
        (Text(s), "isalpha") => Bool(!s.is_empty() && s.chars().all(|c| c.is_ascii_alphabetic())),
        (Text(s), "isspace") => Bool(!s.is_empty() && s.chars().all(|c| c.is_ascii_whitespace())),
        (Text(s), "isempty") => Bool(s.is_empty()),
        (Text(s), "contains") => match a0() { Text(d) => Bool(s.contains(d.as_str())), _ => return trap("contains") },
        (Text(s), "startswith") => match a0() { Text(d) => Bool(s.starts_with(d.as_str())), _ => return trap("startswith") },
        (Text(s), "endswith") => match a0() { Text(d) => Bool(s.ends_with(d.as_str())), _ => return trap("endswith") },
        (Text(s), "toint") => s.trim_matches(is_ws).parse::<i128>().map_err(|_| Flow::Trap(format!("not an integer: {s:?}"))).and_then(|v| chk(v, "i64"))?,
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
        (List(xs), "utf8") => { let mut b = vec![]; for x in xs { match x { Int(v, _) if (0..=255).contains(&v) => b.push(v as u8), _ => return trap("utf8: byte outside 0..255") } } match String::from_utf8(b).ok() { Some(s) => Text(s), None => return trap("utf8: invalid UTF-8") } }
        (List(xs), "sum") => { let mut acc = Int(0, "lit"); for x in xs { acc = arith("+", acc, x)?; } acc }
        (List(xs), "map") => { let f = a0(); let mut out = vec![]; for x in xs { fuel()?; out.push(apply(&f, vec![x])?); } List(out) }
        (List(xs), "filter") => { let f = a0(); let mut out = vec![]; for x in xs { fuel()?; if matches!(apply(&f, vec![x.clone()])?, Bool(true)) { out.push(x); } } List(out) }
        (List(xs), "pscan") => { let f = a0(); let e = a.get(1).cloned().unwrap_or(Unit); pscan(&xs, &f, e, false)? }
        (List(xs), "pscan_incl") => { let f = a0(); let e = a.get(1).cloned().unwrap_or(Unit); pscan(&xs, &f, e, true)? }
        (List(xs), "fold") => { let f = a.get(1).cloned().unwrap_or(Unit); let mut acc = a0(); for x in xs { fuel()?; acc = apply(&f, vec![acc, x])?; } acc }
        (Map(mm), "len") => Int(mm.len() as i128, "usize"),
        (Map(mm), "get") => { let k = a0(); match mm.iter().find(|(x, _)| keq(x, &k)) { Some((_, v)) => v.clone(), None => Err(Box::new(Rec("Missing".into(), vec![]))) } }
        (Map(mm), "contains") => { let k = a0(); Bool(mm.iter().any(|(x, _)| keq(x, &k))) }
        (Map(mut mm), "insert") => { map_set(&mut mm, a0(), a.get(1).cloned().unwrap_or(Unit)); return Ok((Unit, Some(Map(mm)))); }
        (Map(mut mm), "remove") => { let k = a0(); let i = mm.iter().position(|(x, _)| keq(x, &k)); let r = i.map(|i| mm.remove(i).1); return Ok((Opt(r.map(Box::new)), Some(Map(mm)))); }
        (Map(mm), "keys") => List(mm.into_iter().map(|x| x.0).collect()),
        (Map(mm), "values") => List(mm.into_iter().map(|x| x.1).collect()),
        (Map(mm), "merge") => match a0() { Map(ys) => map_merge(mm, ys)?, _ => return trap("merge with a non-map") },
        // a Set[τ] holds its elements in canonical order without duplicates
        (Set(xs), "union") => merge_builtin(&Set(xs), &a0())?,
        (Set(xs), "inter") => match a0() { Set(ys) => Set(xs.into_iter().filter(|x| ys.iter().any(|y| keq(x, y))).collect()), _ => return trap("inter") },
        (Set(xs), "insert") => { let r = merge_builtin(&Set(xs), &Set(vec![a0()]))?; return Ok((Unit, Some(r))); }
        (Set(xs), "tolist") => List(xs),
        (Set(xs), "len") => Int(xs.len() as i128, "usize"),
        (Set(xs), "contains") => { let y = a0(); Bool(xs.iter().any(|x| keq(x, &y))) }
        (Opt(x), "isnone") => Bool(x.is_none()),
        (Opt(x), "issome") => Bool(x.is_some()),
        (Opt(x), "or") => match x { Some(v) => *v, None => a0() },
        (LVar(c), "put") => { lvar_put(&c, a0())?; Unit }
        (LVar(c), "wait") => lvar_wait(&c, &a0())?,
        (LVar(c), "freeze") => lvar_freeze(&c),
        (Chan(..), "close") => { comm(Op::Close(r_chan.clone()))?; Unit }
        (Session(_, inn, cap), "offer") => {
            charge(8)?;
            match comm(Op::Recv(V::Chan(inn.clone(), cap)))? {
                Rep::Val(V::Opt(Some(v))) => match *v { V::Rec(l, f) if f.is_empty() => V::Rec(format!("$label:{l}"), f), o => o },
                _ => return trap("session closed"),
            }
        }
        (Session(out, _, cap), "close") => { comm(Op::Close(V::Chan(out.clone(), cap)))?; Unit }
        (Task(id), "join") => match comm(Op::Join(id))? { Rep::Val(v) => v, _ => Unit },
        (r, _) => return trap(format!("{} has no method {m} at comptime", show(&r))),
    };
    Ok((v, None))
}
/// C.6.4 method work, as conc::method_work in the reference evaluator.
pub fn size_of(v: &V) -> i128 { match v { V::Text(s) => s.len() as i128, V::List(xs) | V::Set(xs) => xs.len() as i128, V::Map(m) => m.len() as i128, _ => 0 } }
pub fn ceil_log2(n: i128) -> i128 { if n <= 1 { 0 } else { 128 - (n - 1).leading_zeros() as i128 } }
pub fn method_work(recv: &V, m: &str, args: &[V]) -> i128 {
    let n = size_of(recv);
    let k = args.first().map(size_of).unwrap_or(0);
    let lin = match recv { V::Text(_) => matches!(m, "strip" | "upper" | "lower" | "reverse" | "tostr" | "chars" | "words" | "lines" | "split" | "isdigit" | "isalpha" | "isspace" | "contains" | "startswith" | "endswith" | "toint" | "join" | "sha256"), V::List(_) => matches!(m, "reverse" | "contains" | "sum" | "map" | "filter" | "fold" | "utf8"), V::Set(_) => matches!(m, "union" | "inter" | "insert" | "contains" | "tolist"), V::Map(_) => matches!(m, "get" | "contains" | "insert" | "remove" | "keys" | "values" | "merge"), _ => false };
    if lin { n.saturating_add(k) } else if matches!(recv, V::List(_)) && m == "sort" { n.saturating_mul(ceil_log2(n)) } else { 0 }
}
/// `sized`: the call expression is in a version 0.7 module (C.15).
pub fn call_method(recv: V, m: &str, args: Vec<V>, sized: bool) -> R<Call> {
    let head = match &recv { V::Rec(n, _) => Some(kind_of(n).to_string()), V::List(_) => Some("List".to_string()), V::Set(_) => Some("Set".to_string()), _ => None };
    if let Some(h) = head && let Some(f) = user_method(&h, m) {
        fuel()?;
        return f(Some(recv), args);
    }
    charge(4i128.saturating_add(if sized { method_work(&recv, m, &args) } else { 0 }))?;
    let (v, s) = builtin(recv, m, args)?;
    Ok(Call { v, outs: vec![], slf: s })
}
pub type Init = Option<Rc<dyn Fn() -> R<V> + Send + Sync>>;
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
pub type Th = Rc<dyn Fn() -> R<V> + Send + Sync>;
#[derive(Clone)]
pub enum P { Wild, Bind(&'static str), Tag(&'static str), Some(Box<P>), Ctor(&'static str, Vec<(Option<&'static str>, P)>), Lit(Th), Range(Th, bool, Th), Tup(Vec<P>), Alt(Vec<P>) }
pub fn pmatch(p: &P, v: &V, b: &mut M) -> R<bool> {
    Ok(match p {
        P::Wild => true,
        P::Bind(n) | P::Tag(n) if matches!(v, V::Rec(l, f) if f.is_empty() && l.starts_with("$label:")) => {
            matches!(v, V::Rec(l, _) if l["$label:".len()..] == *n.rsplit("::").next().unwrap_or(n))
        }
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
        V::List(xs) | V::Set(xs) => xs,
        V::Range(a, b, t) => { if b - a > LOOP_LIMIT * 10 { return trap("loop bound"); } (a..b.max(a)).map(|i| V::Int(i, t)).collect() }
        V::Text(s) => s.chars().map(|c| V::Text(c.to_string())).collect(),
        V::Map(m) => m.into_iter().map(|(k, v)| V::Tuple(vec![k, v])).collect(),
        V::Gen(g) => { let (items, _, err) = g(usize::MAX); if let Some(f) = err { return Err(f); } items }
        o => return trap(format!("cannot iterate over {}", show(&o))),
    })
}
pub enum It { L(std::vec::IntoIter<V>), G { g: Rc<dyn Fn(usize) -> (Vec<V>, bool, Option<Flow>) + Send + Sync>, done: usize, cap: usize, buf: std::vec::IntoIter<V>, fin: bool, err: Option<Flow>, started: bool } }
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
        fuel()?;
        let mut b = M::new();
        if !pmatch(&p, &x, &mut b)? { continue; }
        if let Some(c) = cond && !truthy(&c(&b)?)? { continue; }
        out.push(body(&b)?);
    }
    charge(4 + out.len() as i128)?;
    Ok(V::List(out))
}
pub fn try_(x: V, els: Option<&dyn Fn() -> R<V>>) -> R<V> {
    match x {
        V::Err(e) => Err(Flow::Fail(match els { Some(f) => f()?, None => *e })),
        v @ V::Opt(_) if els.is_none() => Ok(v), // ok value of `T? ! E` [E-Try-V]
        V::Opt(None) => match els { Some(f) => Err(Flow::Fail(f()?)), None => trap("try on none") },
        V::Opt(Some(v)) => Ok(*v),
        v => Ok(v),
    }
}
pub fn compound(op: &str, cur: V, r: V) -> R<V> {
    // `x op= y` charges the cost of `op` exactly as `x = x op y` (C.6.4)
    charge(op_cost(op, &cur))?;
    compound_uncharged(op, cur, r)
}
/// The combine step of a `par` reduction, which charges for itself.
fn compound_uncharged(op: &str, cur: V, r: V) -> R<V> {
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
        V::Tuple(xs) | V::List(xs) | V::Set(xs) => seq(&mut xs.iter().map(json_size), xs.len()),
        V::Map(m) => seq(&mut m.iter().map(|(k, v)| 3 + json_size(k) + json_size(v)), m.len()),
        V::Rec(n, fs) => 2 + esc(n) + 1 + seq(&mut fs.iter().map(|(a, b)| esc(a) + 1 + json_size(b)), fs.len()),
        V::Range(a, b, _) => 3 + a.to_string().len() + b.to_string().len(),
        V::Fn(n) => esc(n),
        V::Clo(_) | V::Gen(_) | V::LVar(_) | V::Chan(..) | V::Task(_) | V::Actor(..) | V::Session(..) => 2,
    }
}
pub fn arena_check(bound: i128, vals: &[&V]) -> R<()> {
    let used: i128 = vals.iter().map(|v| json_size(v) as i128).sum();
    if used > bound { trap("size limit") } else { Ok(()) }
}
pub fn why(f: &Flow) -> String {
    match f { Flow::Trap(t) => format!("trap: {t}"), Flow::Fail(v) => format!("failed with {}", show(v)), Flow::Stop => "Stop".into(), Flow::Ret(_) => "Ret".into(), Flow::Cancel(f) => format!("Cancel({f})"), Flow::Brk => "Brk".into(), Flow::Cont => "Cont".into() }
}
/// Rust `Debug` of the interpreter's `V`, so refusal reasons match its
/// `trap({other:?})` text; closures, iterators and lvars print a placeholder
pub fn dbg_v(v: &V) -> String {
    let l = |xs: &[V]| xs.iter().map(dbg_v).collect::<Vec<_>>().join(", ");
    match v {
        V::Int(n, t) => format!("Int({n}, {t:?})"),
        V::F(f) => format!("F({f:?})"),
        V::F32(f) => format!("F32({f:?})"),
        V::Bool(b) => format!("Bool({b})"),
        V::Text(t) => format!("Text({t:?})"),
        V::Unit => "Unit".into(),
        V::Tuple(xs) => format!("Tuple([{}])", l(xs)),
        V::List(xs) => format!("List([{}])", l(xs)),
        V::Set(xs) => format!("Set([{}])", l(xs)),
        V::Map(kv) => format!("Map([{}])", kv.iter().map(|(k, x)| format!("({}, {})", dbg_v(k), dbg_v(x))).collect::<Vec<_>>().join(", ")),
        V::Opt(None) => "Opt(None)".into(),
        V::Opt(Some(x)) => format!("Opt(Some({}))", dbg_v(x)),
        V::Rec(n, fs) => format!("Rec({n:?}, [{}])", fs.iter().map(|(k, x)| format!("({k:?}, {})", dbg_v(x))).collect::<Vec<_>>().join(", ")),
        V::Err(x) => format!("Err({})", dbg_v(x)),
        V::Fn(n) => format!("Fn({n:?})"),
        V::Range(a, b, t) => format!("Range({a}, {b}, {t:?})"),
        V::Chan(c, k) => format!("Chan({c:?}, {k})"),
        V::Task(t) => format!("Task({t:?})"),
        V::Actor(a, t) => format!("Actor({a:?}, {t:?})"),
        V::Session(a, b, k) => format!("Session({a:?}, {b:?}, {k})"),
        V::Clo(_) => "Closure(..)".into(),
        V::Gen(_) => "Gen(..)".into(),
        V::LVar(_) => "LVar(..)".into(),
    }
}
/// Rust `Debug` of the interpreter's `Flow`
pub fn dbg_flow(f: &Flow) -> String {
    match f {
        Flow::Ret(v) => format!("Ret({})", dbg_v(v)),
        Flow::Brk => "Brk".into(),
        Flow::Cont => "Cont".into(),
        Flow::Fail(v) => format!("Fail({})", dbg_v(v)),
        Flow::Trap(t) => format!("Trap({t:?})"),
        Flow::Stop => "Stop".into(),
        Flow::Cancel(n) => format!("Cancel({n})"),
    }
}
/// machine step refusal (interp.rs): a failure names its record, else "?"
fn refusal(f: &Flow) -> String {
    match f {
        Flow::Fail(V::Rec(n, _)) => n.clone(),
        Flow::Fail(_) => "?".into(),
        Flow::Trap(t) => format!("trap({t})"),
        o => format!("trap({})", dbg_flow(o)),
    }
}
/// actor handler refusal (conc.rs): a non-record failure is shown in full
fn actor_refusal(f: &Flow) -> String {
    match f {
        Flow::Fail(V::Rec(n, _)) => n.clone(),
        Flow::Trap(t) => format!("trap({t})"),
        o => format!("trap({})", dbg_flow(o)),
    }
}
pub fn emit(v: V) -> R<()> {
    if in_task() { comm(Op::Emit(v))?; return Ok(()); }
    OUT.with(|o| o.borrow_mut().push(v)); Ok(())
}
fn take_out() -> Vec<V> { OUT.with(|o| std::mem::take(&mut *o.borrow_mut())) }
pub fn obs(label: &str, th: &dyn Fn() -> R<V>) {
    fuel_reset();
    match th() { Ok(v) => println!("{label} = {}", show(&v)), Err(f) => println!("{label} {}", why(&f)) }
    fuel_report();
}
// Fuel (6.7, 10.5): one unit at each point where the reference evaluator
// ticks. Each observation starts with full fuel and empty constant caches
// (a constant cache entry is valid only for the generation it was filled
// in). The pool is per thread, as the reference gives every task its own
// evaluator: a task thread starts full, actor handlers share the
// scheduler thread's pool, and a `par` branch starts from its parent's
// remaining fuel and is charged back in source order (par_fuel).
thread_local! {
    static FUEL: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
    pub static KGEN: std::cell::Cell<u64> = std::cell::Cell::new(KGEN_NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst));
}
static KGEN_NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
pub fn kgen() -> u64 { KGEN.with(|g| g.get()) }
static FUEL_LIMIT_CELL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);
fn fuel_limit() -> u64 {
    use std::sync::atomic::Ordering::SeqCst;
    let l = FUEL_LIMIT_CELL.load(SeqCst);
    if l != u64::MAX { return l; }
    let l = std::env::args().find_map(|a| a.strip_prefix("--fuel=").and_then(|v| v.parse().ok())).unwrap_or(50_000_000);
    FUEL_LIMIT_CELL.store(l, SeqCst);
    l
}
/// `--fuel-batch=n1,n2,..`: run every observation once per limit in this
/// process, each pass headed by `#fuel-batch n`; observations already start
/// from full fuel and empty caches, so a pass equals a fresh run.
pub fn observe_all(f: &dyn Fn()) {
    let Some(b) = std::env::args().find_map(|a| a.strip_prefix("--fuel-batch=").map(String::from)) else { f(); return };
    for n in b.split(',') {
        FUEL_LIMIT_CELL.store(n.parse().unwrap(), std::sync::atomic::Ordering::SeqCst);
        println!("#fuel-batch {n}");
        f();
    }
}
pub fn fuel_left() -> u64 { FUEL.with(|f| f.get().unwrap_or_else(fuel_limit)) }
fn fuel_set(n: u64) { FUEL.with(|f| f.set(Some(n))); }
pub fn fuel() -> R<()> {
    match fuel_left().checked_sub(1) { Some(n) => { fuel_set(n); Ok(()) } None => trap("comptime fuel exhausted") }
}
// `bound n` (6.7): active calls of each bounded function, per thread
// (each task and par branch counts its own; W-REC-1 keeps par out of a
// cycle with a bounded function, so this agrees with the evaluator)
thread_local! { static DEPTH: RefCell<std::collections::HashMap<&'static str, u64>> = RefCell::new(std::collections::HashMap::new()); }
pub fn depth_enter(k: &'static str, n: u64) -> R<()> {
    let d = DEPTH.with(|m| m.borrow().get(k).copied().unwrap_or(0));
    if d >= n { return trap("recursion bound"); }
    DEPTH.with(|m| m.borrow_mut().insert(k, d + 1));
    Ok(())
}
pub fn depth_exit(k: &'static str) { DEPTH.with(|m| { if let Some(d) = m.borrow_mut().get_mut(k) { *d = d.saturating_sub(1); } }); }
pub fn fuel_reset() { DEPTH.with(|m| m.borrow_mut().clear()); fuel_set(fuel_limit()); KGEN.with(|g| g.set(KGEN_NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))); }
pub fn fuel_report() { if std::env::args().any(|a| a == "--fuel-trace") { println!("fuel {}", fuel_limit() - fuel_left()); } }
/// resolve one `par` branch in source order: the reference runs branches in
/// sequence on one pool (ParMode::Seq), so branch i traps when the fuel of
/// branches 0..=i exceeds what the parent had at the fork
pub fn par_fuel(cum: &mut u64, f0: u64, used: u64) -> R<()> {
    *cum = cum.saturating_add(used);
    fuel_set(f0.saturating_sub(*cum));
    if *cum > f0 { return trap("comptime fuel exhausted"); }
    Ok(())
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
pub type Ev = (&'static str, Rc<dyn Fn() -> R<Option<V>> + Send + Sync>);
pub fn run_example(line: usize, events: Vec<Ev>) {
    fuel_reset();
    run_example1(line, events);
    fuel_report();
}
fn run_example1(line: usize, events: Vec<Ev>) {
    println!("example {line}");
    let mut s = match initial_state() { Ok(s) => s, Err(f) => { println!("initial {}", why(&f)); return; } };
    println!("0 (initial) {}", show_state(&s));
    let tr = std::env::args().any(|a| a == "--trace");
    let mut d = DLog { prev: D_GENESIS.to_string(), seq: 0 };
    for (k, (name, th)) in events.iter().enumerate() {
        let k = k + 1;
        let ev = match th() { Ok(Some(v)) => v, Ok(None) => { println!("{k} {name} not an event"); break; } Err(f) => { println!("{k} {name} {}", why(&f)); break; } };
        match delta(&s, &ev) {
            Ok((s2, outs)) => {
                s = s2;
                if tr { d.add(name, &ev, &s, false); }
                println!("{k} {} {}", show(&ev), show_state(&s));
                for o in outs { println!("  emit {}", show(&o)); }
            }
            Err(r) => { if tr { d.add(name, &ev, &s, true); } println!("{k} {} refused: {r}", show(&ev)); break; }
        }
    }
    if tr { println!("trace {}", d.prev); }
}
// ---- 6.1 the log D: canonical JSON (N3) and the SHA-256 entry chain, as interp.rs writes it
const D_GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";
enum FV { Null, Bool(bool), Int(i128), Float(f64), Text(String), List(Vec<FV>), Rec(Vec<(String, FV)>) }
impl FV {
    fn rank(&self) -> u8 { match self { FV::Null => 0, FV::Bool(_) => 1, FV::Int(_) | FV::Float(_) => 2, FV::Text(_) => 3, FV::List(_) => 4, FV::Rec(_) => 5 } }
    fn num(&self) -> f64 { match self { FV::Int(i) => *i as f64, FV::Float(f) => *f, _ => 0.0 } }
    fn cmp_canon(&self, o: &FV) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        let r = self.rank().cmp(&o.rank());
        if r != Equal { return r; }
        match (self, o) {
            (FV::Bool(a), FV::Bool(b)) => a.cmp(b),
            (FV::Int(a), FV::Int(b)) => a.cmp(b),
            (a @ (FV::Int(_) | FV::Float(_)), b) => a.num().partial_cmp(&b.num()).unwrap_or(Equal),
            (FV::Text(a), FV::Text(b)) => a.chars().cmp(b.chars()),
            (FV::List(a), FV::List(b)) => { for (x, y) in a.iter().zip(b) { let c = x.cmp_canon(y); if c != Equal { return c; } } a.len().cmp(&b.len()) }
            (FV::Rec(a), FV::Rec(b)) => { for ((ka, va), (kb, vb)) in a.iter().zip(b) { let c = ka.cmp(kb).then_with(|| va.cmp_canon(vb)); if c != Equal { return c; } } a.len().cmp(&b.len()) }
            _ => Equal,
        }
    }
    fn json(&self) -> String {
        match self {
            FV::Null => "null".into(),
            FV::Bool(b) => b.to_string(),
            FV::Int(i) => i.to_string(),
            FV::Float(f) => float_json(*f),
            FV::Text(t) => jstr(t),
            FV::List(xs) => format!("[{}]", xs.iter().map(|x| x.json()).collect::<Vec<_>>().join(",")),
            FV::Rec(fs) => format!("{{{}}}", fs.iter().map(|(k, v)| format!("{}:{}", jstr(k), v.json())).collect::<Vec<_>>().join(",")),
        }
    }
}
fn frec(mut fs: Vec<(String, FV)>) -> FV { fs.sort_by(|a, b| a.0.cmp(&b.0)); FV::Rec(fs) }
fn form_value(v: &V) -> FV {
    match v {
        V::Int(i, _) => FV::Int(*i),
        V::F(f) | V::F32(f) => FV::Float(*f),
        V::Bool(b) => FV::Bool(*b),
        V::Text(t) => FV::Text(t.clone()),
        V::Unit | V::Opt(None) => FV::Null,
        V::Opt(Some(x)) => form_value(x),
        V::Tuple(xs) | V::List(xs) | V::Set(xs) => FV::List(xs.iter().map(form_value).collect()),
        V::Map(ps) => {
            let mut l: Vec<(FV, FV)> = ps.iter().map(|(k, v)| (form_value(k), form_value(v))).collect();
            l.sort_by(|a, b| a.0.cmp_canon(&b.0));
            FV::List(l.into_iter().map(|(k, v)| FV::List(vec![k, v])).collect())
        }
        V::Rec(n, fs) => {
            let mut f: Vec<(String, FV)> = fs.iter().map(|(k, x)| (k.clone(), form_value(x))).collect();
            if is_variant(n) { f.push(("$v".into(), FV::Text(n.clone()))); }
            frec(f)
        }
        V::Err(x) => frec(vec![("$err".into(), form_value(x))]),
        V::Range(a, b, _) => FV::List((*a..*b).map(FV::Int).collect()),
        V::Fn(n) => frec(vec![("$fn".into(), FV::Text(n.clone()))]),
        _ => FV::Null,
    }
}
fn float_json(x: f64) -> String {
    if x == 0.0 { return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() }; }
    let e = format!("{:e}", x.abs());
    let (m, ex) = e.split_once('e').unwrap();
    let ex: i32 = ex.parse().unwrap();
    let digits: String = m.chars().filter(|c| *c != '.').collect();
    let sign = if x < 0.0 { "-" } else { "" };
    let a = x.abs();
    if (1e-4..1e16).contains(&a) {
        let p = ex + 1;
        let body = if p <= 0 { format!("0.{}{}", "0".repeat((-p) as usize), digits) }
            else if p as usize >= digits.len() { format!("{}{}.0", digits, "0".repeat(p as usize - digits.len())) }
            else { format!("{}.{}", &digits[..p as usize], &digits[p as usize..]) };
        format!("{sign}{body}")
    } else { format!("{sign}{m}e{ex}") }
}
/// 6.1, T7: the positional embedding hashed by D (names replaced by positions).
fn form_value_pos(v: &V) -> FV {
    match v {
        V::Opt(Some(x)) => form_value_pos(x),
        V::Tuple(xs) | V::List(xs) | V::Set(xs) => FV::List(xs.iter().map(form_value_pos).collect()),
        V::Map(ps) => {
            let mut l: Vec<(FV, FV, FV)> = ps.iter().map(|(k, v)| (form_value(k), form_value_pos(k), form_value_pos(v))).collect();
            l.sort_by(|a, b| a.0.cmp_canon(&b.0));
            FV::List(l.into_iter().map(|(_, k, v)| FV::List(vec![k, v])).collect())
        }
        V::Rec(n, fs) => {
            let at = |f: &str| fs.iter().find(|(k, _)| k == f).map_or(FV::Null, |(_, x)| form_value_pos(x));
            if let Some(d) = decl_fields(n) {
                FV::List(d.iter().map(|f| at(f)).collect())
            } else if let Some((k, d)) = variant_index(n) {
                FV::List(vec![FV::Int(k), FV::List(d.iter().map(|f| at(f)).collect())])
            } else {
                form_value(v)
            }
        }
        V::Err(x) => frec(vec![("$err".into(), form_value_pos(x))]),
        _ => form_value(v),
    }
}
fn jstr(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"), '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}
struct DLog { prev: String, seq: usize }
impl DLog {
    /// Append one entry and print it (`--trace`): `d applied|kept HASH BODY`,
    /// BODY being the canonical JSON that HASH is the SHA-256 of.
    fn add(&mut self, name: &str, ev: &V, st: &State, kept: bool) {
        let ty = match event_index(name) { Some(k) => k.to_string(), None => jstr(name) };
        let evj = format!("{{\"payload\":{},\"type\":{ty}}}", form_value_pos(ev).json());
        let sj = FV::List(st.iter().map(|(_, fs)| FV::List(fs.iter().map(|(_, v)| form_value_pos(v)).collect())).collect()).json();
        let sh = sha256_hex(sj.as_bytes());
        let body = format!("{{\"event\":{evj},\"prev_hash\":\"{}\",\"seq\":{},\"state_hash\":\"{sh}\"}}", self.prev, self.seq);
        let h = sha256_hex(body.as_bytes());
        println!("d {} {h} {body}", if kept { "kept" } else { "applied" });
        self.prev = h;
        self.seq += 1;
    }
}
pub fn run_main(f: &dyn Fn(Option<V>, Vec<V>) -> R<Call>) {
    fuel_reset();
    println!("main");
    take_out();
    let r = fuel().and_then(|_| f(None, vec![]));
    for o in take_out() { println!("emit {}", show(&o)); }
    match r { Ok(c) => println!("main = {}", show(&c.v)), Err(e) => println!("main {}", why(&e)) }
    fuel_report();
}

// ---- draft 0.6 (the draft 0.6 concurrency chapter, not published here C.5.2, C.11.2)
pub fn par_ident(op: &str, b: V) -> R<V> {
    Ok(match (op, &b) {
        ("+" | "|" | "^", V::Int(_, t)) => V::Int(0, t),
        ("*", V::Int(_, t)) => V::Int(1, t),
        ("&", V::Int(_, t)) => { let t2 = if *t == "lit" { "i64" } else { t }; let (lo, hi) = irange(t2); V::Int(if lo == 0 { hi } else { -1 }, t) }
        ("+", V::F(_)) => V::F(-0.0),
        ("*", V::F(_)) => V::F(1.0),
        ("+", V::F32(_)) => V::F32(-0.0),
        ("*", V::F32(_)) => V::F32(1.0),
        ("&", V::Bool(_)) => V::Bool(true),
        ("|" | "^", V::Bool(_)) => V::Bool(false),
        _ => return trap(format!("no identity for {op}= on {}", show(&b))),
    })
}
pub fn par_combine(op: &str, a: V, b: V) -> R<V> {
    charge(op_cost(op, &a))?;
    match (op, &a, &b) {
        ("&", V::Bool(x), V::Bool(y)) => Ok(V::Bool(*x && *y)),
        ("|", V::Bool(x), V::Bool(y)) => Ok(V::Bool(*x || *y)),
        ("^", V::Bool(x), V::Bool(y)) => Ok(V::Bool(*x != *y)),
        _ => compound_uncharged(op, a, b),
    }
}
/// the fixed balanced tree T: h = ceil(m / 2)
pub fn par_tree(op: &str, c: &[V], id: &V) -> R<V> {
    match c.len() {
        0 => Ok(id.clone()),
        1 => Ok(c[0].clone()),
        m => { let h = 1usize << (m - 1).ilog2(); let l = par_tree(op, &c[..h], id)?; let r = par_tree(op, &c[h..], id)?; par_combine(op, l, r) }
    }
}
/// Blelloch scan over the same tree (exclusive, or inclusive)
pub fn pscan(xs: &[V], f: &V, e: V, inclusive: bool) -> R<V> {
    fn up(xs: &[V], lo: usize, hi: usize, f: &V, sums: &mut HashMap<(usize, usize), V>) -> R<V> {
        let v = if hi - lo == 1 { xs[lo].clone() } else {
            let h = lo + (1usize << (hi - lo - 1).ilog2());
            let l = up(xs, lo, h, f, sums)?; let r = up(xs, h, hi, f, sums)?; apply(f, vec![l, r])?
        };
        sums.insert((lo, hi), v.clone());
        Ok(v)
    }
    fn down(lo: usize, hi: usize, off: Option<V>, f: &V, sums: &HashMap<(usize, usize), V>, out: &mut Vec<Option<V>>) -> R<()> {
        if hi - lo == 1 { out[lo] = off; return Ok(()); }
        let h = lo + (1usize << (hi - lo - 1).ilog2());
        let ls = sums[&(lo, h)].clone();
        let roff = match &off { None => ls, Some(o) => apply(f, vec![o.clone(), ls])? };
        down(lo, h, off, f, sums, out)?;
        down(h, hi, Some(roff), f, sums, out)
    }
    let n = xs.len();
    if n == 0 { return Ok(V::List(vec![])); }
    let mut sums = HashMap::new();
    up(xs, 0, n, f, &mut sums)?;
    let mut out = vec![None; n];
    down(0, n, None, f, &sums, &mut out)?;
    let mut r = vec![];
    for (i, o) in out.into_iter().enumerate() {
        r.push(if inclusive { match o { None => xs[i].clone(), Some(o) => apply(f, vec![o, xs[i].clone()])? } } else { o.unwrap_or_else(|| e.clone()) });
    }
    Ok(V::List(r))
}

// ---- cost accounting for `with budget` (C.6.1, C.6.4), as interp `charge`
// `&mut` write-back on failure (6.2): a failing union function's b_ records
// its copy-out here and its u_ takes it at once, on the same thread.
thread_local! { static FAIL_OUTS: RefCell<(Vec<Option<V>>, Option<V>)> = const { RefCell::new((Vec::new(), None)) }; }
pub fn fail_outs_set(o: Vec<Option<V>>, s: Option<V>) { FAIL_OUTS.with(|f| *f.borrow_mut() = (o, s)); }
pub fn fail_outs_take() -> (Vec<Option<V>>, Option<V>) { FAIL_OUTS.with(|f| std::mem::take(&mut *f.borrow_mut())) }
thread_local! { static BUD: RefCell<Vec<(i128, i128)>> = const { RefCell::new(Vec::new()) }; }
pub fn charge(u: i128) -> R<()> {
    BUD.with(|b| {
        let mut b = b.borrow_mut();
        if b.is_empty() { return Ok(()); }
        if b.iter().any(|(s, l)| s.saturating_add(u) > *l) { return trap("budget"); }
        for x in b.iter_mut() { x.0 = x.0.saturating_add(u); }
        Ok(())
    })
}
pub struct BudgetGuard;
impl Drop for BudgetGuard { fn drop(&mut self) { BUD.with(|b| { if let Some((u, _)) = b.borrow_mut().pop() { BUD_LAST.with(|l| l.set(u)); } }); } }
thread_local! { static BUD_LAST: std::cell::Cell<i128> = const { std::cell::Cell::new(0) }; }
/// units the most recently left budget frame had spent (Exhausted.used, C.5)
pub fn budget_last() -> V { V::Int(BUD_LAST.with(|l| l.get()), "u64") }
pub fn budget_enter(n: V) -> R<BudgetGuard> {
    let n = marker_int(&n)?;
    BUD.with(|b| b.borrow_mut().push((0, n)));
    Ok(BudgetGuard)
}
pub fn op_cost(op: &str, x: &V) -> i128 {
    match x { V::F(_) | V::F32(_) => match op { "+" | "-" | "*" => 2, "/" | "%" => 8, _ => 1 }, _ => 1 }
}
pub fn pre_charge<T>(u: i128, v: impl FnOnce() -> R<T>) -> R<T> { charge(u)?; v() }
pub fn cindex(c: V, i: V) -> R<V> { charge(1)?; index(c, i) }
pub fn cfield(v: V, m: &str) -> R<V> { charge(1)?; field(v, m) }
pub fn ctmember(v: V, i: usize) -> R<V> { charge(1)?; tmember(v, i) }
pub fn cneg(v: V) -> R<V> { charge(1)?; neg(v) }
pub fn cbitnot(v: V) -> R<V> { charge(1)?; bitnot(v) }
pub fn cnot(v: V) -> R<V> { charge(1)?; not_(v) }
pub fn cbinop(op: &str, x: V, y: V) -> R<V> { charge(op_cost(op, &x))?; binop(op, x, y) }
/// a value-receiver call of a user method charges before its arguments (interp `call_fn`)
pub fn pre_method(recv: V, m: &str) -> R<V> {
    let head = match &recv { V::Rec(n, _) => Some(kind_of(n).to_string()), V::List(_) => Some("List".to_string()), V::Set(_) => Some("Set".to_string()), _ => None };
    if let Some(h) = head && user_method(&h, m).is_some() { charge(4)?; }
    Ok(recv)
}
/// calling a function value charges before its arguments; a closure charges on entry
pub fn pre_value(f: V) -> R<V> { if let V::Fn(_) = f { charge(4)?; } Ok(f) }

pub fn marker_int(v: &V) -> R<i128> {
    match v { V::Int(n, _) => Ok(*n), o => trap(format!("expected an integer, got {}", show(o))) }
}

// ---- LVars (C.11.8) and the built-in Merge instances (C.11.5)
pub fn lvar_new(a: Vec<V>) -> V {
    let mut it = a.into_iter();
    let b = it.next().unwrap_or(V::Unit);
    V::LVar(Rc::new(LCell(std::sync::Mutex::new((b, it.next(), false)))))
}
fn crank(v: &V) -> u8 {
    match v { V::Unit | V::Opt(None) => 0, V::Bool(_) => 1, V::Int(..) | V::F(_) | V::F32(_) => 2, V::Text(_) => 3, V::List(_) | V::Set(_) | V::Tuple(_) => 4, _ => 5 }
}
fn cnum(v: &V) -> f64 { match v { V::Int(x, _) => *x as f64, V::F(x) | V::F32(x) => *x, _ => 0.0 } }
/// Map and set key identity (6.1): canonical-order equality, not IEEE ==.
pub fn keq(a: &V, b: &V) -> bool { canon_cmp(a, b).is_eq() }
pub fn canon_cmp(a: &V, b: &V) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    let (ra, rb) = (crank(a), crank(b));
    if ra != rb { return ra.cmp(&rb); }
    match (a, b) {
        (V::Bool(x), V::Bool(y)) => x.cmp(y),
        (V::Int(x, _), V::Int(y, _)) => x.cmp(y),
        (V::Int(..), _) => Less, // int before float: never equal keys (6.1)
        (_, V::Int(..)) => Greater,
        (x, y) if ra == 2 => cnum(x).total_cmp(&cnum(y)), // IEEE 754 totalOrder
        (V::Text(x), V::Text(y)) => x.cmp(y),
        (V::List(x) | V::Set(x) | V::Tuple(x), V::List(y) | V::Set(y) | V::Tuple(y)) => {
            for (p, q) in x.iter().zip(y) { let c = canon_cmp(p, q); if c != Equal { return c; } }
            x.len().cmp(&y.len())
        }
        (V::Opt(Some(x)), V::Opt(Some(y))) => canon_cmp(x, y),
        (V::Rec(n, x), V::Rec(m, y)) => n.cmp(m).then_with(|| {
            for ((k1, v1), (k2, v2)) in x.iter().zip(y) { let c = k1.cmp(k2).then_with(|| canon_cmp(v1, v2)); if c != Equal { return c; } }
            x.len().cmp(&y.len())
        }),
        _ => show(a).cmp(&show(b)),
    }
}
pub fn merge_builtin(a: &V, b: &V) -> R<V> {
    Ok(match (a, b) {
        (V::Int(x, t), V::Int(y, u)) => V::Int(*x.max(y), if *t == "lit" { u } else { t }),
        (V::Bool(x), V::Bool(y)) => V::Bool(*x || *y),
        (V::List(xs), V::List(ys)) => {
            let mut v: Vec<V> = xs.iter().chain(ys).cloned().collect();
            v.sort_by(canon_cmp);
            v.dedup_by(|p, q| canon_cmp(p, q).is_eq());
            V::List(v)
        }
        (V::Set(xs), V::Set(ys)) => {
            let mut v: Vec<V> = xs.iter().chain(ys).cloned().collect();
            v.sort_by(canon_cmp);
            v.dedup_by(|p, q| canon_cmp(p, q).is_eq());
            V::Set(v)
        }
        (V::Map(xs), V::Map(ys)) => {
            let mut out = xs.clone();
            for (k, v) in ys {
                if let Some(e) = out.iter_mut().find(|(k2, _)| keq(k2, k)) { e.1 = merge_builtin(&e.1, v)?; } else { out.push((k.clone(), v.clone())); }
            }
            out.sort_by(|p, q| canon_cmp(&p.0, &q.0));
            V::Map(out)
        }
        (V::Tuple(xs), V::Tuple(ys)) if xs.len() == ys.len() => V::Tuple(xs.iter().zip(ys).map(|(p, q)| merge_builtin(p, q)).collect::<R<_>>()?),
        (V::Rec(n, xs), V::Rec(m, ys)) if n == m && xs.len() == ys.len() => V::Rec(n.clone(), xs.iter().zip(ys).map(|((k, p), (_, q))| Ok((k.clone(), merge_builtin(p, q)?))).collect::<R<_>>()?),
        (V::Opt(None), x) | (x, V::Opt(None)) => x.clone(),
        (V::Opt(Some(x)), V::Opt(Some(y))) => V::Opt(Some(Box::new(merge_builtin(x, y)?))),
        (x, y) => return trap(format!("no Merge instance for {} and {}", show(x), show(y))),
    })
}
pub fn map_merge(xs: Vec<(V, V)>, ys: Vec<(V, V)>) -> R<V> {
    let mut out = xs;
    for (k, v) in ys {
        if let Some(i) = out.iter().position(|(k2, _)| keq(k2, &k)) { let m = merge_values(&out[i].1, &v)?; out[i].1 = m; } else { out.push((k, v)); }
    }
    out.sort_by(|p, q| canon_cmp(&p.0, &q.0));
    Ok(V::Map(out))
}
pub fn set_of(v: V) -> R<V> { match merge_builtin(&v, &V::List(vec![]))? { V::List(xs) => Ok(V::Set(xs)), _ => trap("set_of of a non-list") } }
pub fn merge_values(a: &V, b: &V) -> R<V> {
    if let V::Rec(n, _) = a && let Some(f) = user_method(kind_of(n), "merge") {
        charge(4)?;
        return Ok(f(Some(a.clone()), vec![b.clone()])?.v);
    }
    merge_builtin(a, b)
}
pub fn lvar_put(c: &Rc<LCell>, v: V) -> R<()> {
    let changed = loop {
        let cur = c.borrow().0.clone();
        let nv = merge_values(&cur, &v)?;
        let mut g = c.borrow_mut();
        if !eq(&g.0, &cur) { continue; } // another branch put in between: merge again
        if let Some(top) = &g.1 && eq(&nv, top) { return trap("lvar conflict"); }
        let ch = !eq(&nv, &cur);
        if ch && g.2 { return trap("lvar put after freeze"); }
        g.0 = nv;
        break ch;
    };
    if changed {
        PUTS.with(|p| p.set(p.get() + 1));
        if let Some(cx) = PARCX.with(|p| p.borrow().clone()) { cx.bump(); }
    }
    Ok(())
}
thread_local! { static SCHED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
thread_local! { static PUTS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) }; }
/// number of puts that changed an lvar so far (progress of the restart scheduler)
pub fn puts() -> u64 { PUTS.with(|p| p.get()) }
/// `freeze()`: set the frozen bit, read the exact value
pub fn lvar_freeze(c: &Rc<LCell>) -> V { let mut g = c.borrow_mut(); g.2 = true; g.0.clone() }
/// merge for a threshold test: the program's Merge, charged nothing (interp `merge_free`)
pub fn merge_free(a: &V, b: &V) -> R<V> {
    let bud = BUD.with(|x| std::mem::take(&mut *x.borrow_mut()));
    let f = fuel_left();
    fuel_set(fuel_limit());
    let r = merge_values(a, b);
    fuel_set(f);
    BUD.with(|x| *x.borrow_mut() = bud);
    r
}
fn lvar_above(ths: &[V], cur: &V) -> R<Option<V>> {
    for t in ths { if eq(&merge_free(t, cur)?, cur) { return Ok(Some(t.clone())); } }
    Ok(None)
}
/// W-LVAR-2 at run time: distinct thresholds must join to the declared top
pub fn lvar_incomp(top: &Option<V>, ths: &[V]) -> R<()> {
    for (i, a) in ths.iter().enumerate() {
        for b in &ths[i + 1..] {
            if eq(a, b) { continue; }
            let ok = match top { Some(t) => eq(&merge_free(a, b)?, t), None => false };
            if !ok { return trap("lvar thresholds are not pairwise incompatible"); }
        }
    }
    Ok(())
}
pub fn lvar_wait(c: &Rc<LCell>, q: &V) -> R<V> {
    let V::List(ths) = q else { return trap("wait needs a list of thresholds") };
    let top = c.borrow().1.clone();
    lvar_incomp(&top, ths)?;
    let cur = c.borrow().0.clone();
    if let Some(t) = lvar_above(ths, &cur)? { return Ok(t); }
    if let Some(cx) = PARCX.with(|p| p.borrow().clone()) {
        let above = |cur: &V| lvar_above(ths, cur);
        let mut g = cx.m.lock().unwrap();
        let mut seen: Option<u64> = None;
        loop {
            if g.stuck { return trap("lvar wait cannot be satisfied"); }
            if seen != Some(g.ver) {
                let cur = c.borrow().0.clone();
                if let Some(t) = above(&cur)? { return Ok(t); }
                seen = Some(g.ver);
                g.idle += 1;
            }
            if g.idle >= g.running { g.stuck = true; cx.cv.notify_all(); return trap("lvar wait cannot be satisfied"); }
            g = cx.cv.wait(g).unwrap();
        }
    }
    if SCHED.with(|s| s.get()) > 0 { return trap("$block"); }
    trap("lvar wait cannot be satisfied")
}
// ---- par on scoped OS threads (interp `par_exec`, Threads mode)
#[derive(Default)]
pub struct ParSt { running: usize, idle: usize, ver: u64, stuck: bool }
pub struct ParCx { m: std::sync::Mutex<ParSt>, cv: std::sync::Condvar, parent: Option<Rc<ParCx>>, pub fuel0: u64 }
impl ParCx {
    fn bump(&self) {
        let mut g = self.m.lock().unwrap();
        g.ver += 1; g.idle = 0; self.cv.notify_all(); drop(g);
        if let Some(p) = &self.parent { p.bump(); }
    }
}
thread_local! { static PARCX: RefCell<Option<Rc<ParCx>>> = const { RefCell::new(None) }; }
pub fn par_cx_new(k: usize) -> Rc<ParCx> {
    Rc::new(ParCx { m: std::sync::Mutex::new(ParSt { running: k, ..ParSt::default() }), cv: std::sync::Condvar::new(), parent: PARCX.with(|p| p.borrow().clone()), fuel0: fuel_left() })
}
/// run one branch on this (fresh) thread: (result, units it charged)
pub fn par_branch(cx: Rc<ParCx>, bud: Vec<(i128, i128)>, f: impl FnOnce() -> R<()>) -> (R<()>, (i128, u64)) {
    PARCX.with(|p| *p.borrow_mut() = Some(cx.clone()));
    fuel_set(cx.fuel0);
    let s0 = bud.first().map(|b| b.0).unwrap_or(0);
    BUD.with(|b| *b.borrow_mut() = bud);
    let r = f();
    let s1 = BUD.with(|b| b.borrow().first().map(|b| b.0).unwrap_or(0));
    let mut g = cx.m.lock().unwrap();
    g.running -= 1; cx.cv.notify_all(); drop(g);
    (r, (s1 - s0, cx.fuel0 - fuel_left()))
}
/// `for ... par`: how many chunks (threads) for `n` items, and the split
pub fn par_chunks(n: usize) -> usize {
    let w = std::env::var("RCE_PAR_THREADS").ok().and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4));
    n.min(w.max(1)).max(1)
}
pub fn par_split(items: &[V], k: usize) -> Vec<Vec<V>> {
    let n = items.len();
    (0..k).map(|i| items[i * n / k..(i + 1) * n / k].to_vec()).collect()
}
pub fn bud_rem() -> Option<i128> { BUD.with(|b| b.borrow().iter().map(|(s, l)| l - s).min()) }
pub fn bud_add(n: i128) { BUD.with(|b| for x in b.borrow_mut().iter_mut() { x.0 += n; }); }
pub fn merge3(base: &V, cur: &V, new: V) -> V {
    if eq(&new, base) { return cur.clone(); }
    if eq(cur, base) { return new; }
    match (base, cur, new) {
        (V::Rec(bn, bf), V::Rec(cn, cf), V::Rec(nn, nf)) if bn == cn && *cn == nn && bf.len() == cf.len() && cf.len() == nf.len() => {
            let f = bf.iter().zip(cf).zip(nf).map(|(((k, b), (_, c)), (_, n))| (k.clone(), merge3(b, c, n))).collect();
            V::Rec(nn, f)
        }
        (V::Tuple(b), V::Tuple(c), V::Tuple(n)) if b.len() == c.len() && c.len() == n.len() => V::Tuple(b.iter().zip(c).zip(n).map(|((b, c), n)| merge3(b, c, n)).collect()),
        (V::List(b), V::List(c), V::List(n)) if b.len() == c.len() && c.len() == n.len() => V::List(b.iter().zip(c).zip(n).map(|((b, c), n)| merge3(b, c, n)).collect()),
        (_, _, n) => n,
    }
}
/// a par whose branches wait on lvars runs them by restart (see emit/stmt.rs)
pub fn par_enter() { SCHED.with(|s| s.set(s.get() + 1)); }
pub fn par_exit() { SCHED.with(|s| s.set(s.get() - 1)); }
pub fn bud_snap() -> Vec<(i128, i128)> { BUD.with(|b| b.borrow().clone()) }
pub fn bud_restore(s: Vec<(i128, i128)>) { BUD.with(|b| *b.borrow_mut() = s); }
/// a restarted par attempt (C.14) restores the budgets and the fuel
pub fn par_snap() -> (Vec<(i128, i128)>, u64) { (bud_snap(), fuel_left()) }
pub fn par_restore(s: (Vec<(i128, i128)>, u64)) { bud_restore(s.0); fuel_set(s.1); }

// ---- workgroups (C.11.2)
pub fn lanes_of(v: &V) -> R<(usize, i128)> { let n = marker_int(v)?; Ok((n.max(0) as usize, n)) }
pub fn group_rec(lane: usize, n: i128) -> V {
    V::Rec("$group".into(), vec![("lane".into(), V::Int(lane as i128, "usize")), ("lanes".into(), V::Int(n, "usize"))])
}

/// an lvar cell, shared between threads
/// (value, top, frozen bit); the bit is read and set under the same lock as the value
pub struct LCell(pub std::sync::Mutex<(V, Option<V>, bool)>);
impl LCell {
    pub fn borrow(&self) -> std::sync::MutexGuard<'_, (V, Option<V>, bool)> { self.0.lock().unwrap_or_else(|e| e.into_inner()) }
    pub fn borrow_mut(&self) -> std::sync::MutexGuard<'_, (V, Option<V>, bool)> { self.borrow() }
}

// ---- SHA-256 (FIPS 180-4), for the content-hash ids of 6.6
pub fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut m = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    m.push(0x80);
    while m.len() % 64 != 56 { m.push(0); }
    m.extend_from_slice(&bits.to_be_bytes());
    for ch in m.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 { w[i] = u32::from_be_bytes([ch[4 * i], ch[4 * i + 1], ch[4 * i + 2], ch[4 * i + 3]]); }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let chf = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(chf).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for i in 0..8 { h[i] = h[i].wrapping_add(v[i]); }
    }
    let mut out = [0u8; 32];
    for i in 0..8 { out[4 * i..4 * i + 4].copy_from_slice(&h[i].to_be_bytes()); }
    out
}
/// the id of 6.6: first 16 hex digits of SHA-256 of a name-free key
pub fn hid(key: &str) -> String { sha256(key.as_bytes()).iter().take(8).map(|b| format!("{b:02x}")).collect() }

// ---- the tick runtime for `--run` (6.8, chapter C): one OS thread per task,
// a scheduler on the main thread that admits moves tick by tick, as interp
// `conc::Sched` (staged delivery, held sends, select rotation, logged nondet
// choices, deadlines, join handles, scopes). Not covered: actors, nodes,
// partitions, supervisors, sessions, cross-actor atomic.
use std::sync::mpsc;
use std::collections::{BTreeMap, VecDeque};
pub enum SArm { Recv(V, u8), Send(V, V), After(u64), Default }
pub enum Op {
    Send(V, V), Recv(V), Close(V), Emit(V), Join(String), AwaitScope(u64),
    Select { arms: Vec<SArm>, key: (usize, u64), nondet: bool },
    Spawn { f: String, args: Vec<V>, prio: u8, site: usize, idx: u64, scope: u64 },
    Atomic(Vec<(V, V)>),
    Done(Result<V, Flow>),
}
pub enum Rep { Go, Val(V), Arm(usize, Option<V>), Spawned(String), Cancel(u64), Failed(V) }
pub struct Req { op: Op, dl: Vec<(u64, u64)> }
pub const CANCEL_TASK: u64 = u64::MAX;
struct TaskCx { tx: mpsc::Sender<Req>, rx: mpsc::Receiver<(u64, Rep)>, id: String, created: u64, ords: HashMap<String, u64>, now: u64, dl: Vec<(u64, u64)>, next_frame: u64, scopes: Vec<u64>, sel_k: HashMap<usize, u64> }
thread_local! { static TASK: RefCell<Option<TaskCx>> = const { RefCell::new(None) }; }
pub fn in_task() -> bool { TASK.with(|t| t.borrow().is_some()) }
pub fn comm(op: Op) -> R<Rep> {
    let r = TASK.with(|t| {
        let t = t.borrow();
        let Some(cx) = t.as_ref() else { return Err(Flow::Trap("concurrency is not available at comptime".into())) };
        cx.tx.send(Req { op, dl: cx.dl.clone() }).map_err(|_| Flow::Trap("runtime stopped".into()))?;
        cx.rx.recv().map_err(|_| Flow::Trap("runtime stopped".into()))
    })?;
    let (tick, rep) = r;
    TASK.with(|t| if let Some(cx) = t.borrow_mut().as_mut() { cx.now = tick; });
    match rep { Rep::Cancel(f) => Err(Flow::Cancel(f)), Rep::Failed(v) => Err(Flow::Fail(v)), r => Ok(r) }
}
fn task_fresh(kind: &str, line: usize) -> String {
    TASK.with(|t| { let mut t = t.borrow_mut(); let cx = t.as_mut().unwrap(); let idx = cx.created; cx.created += 1; hid(&format!("{kind}|{line}|{}|{idx}", cx.id)) })
}
pub fn chan_new(cap: &dyn Fn() -> R<V>, line: usize) -> R<V> {
    if !in_task() { return trap("channels are not available at comptime"); }
    let n = marker_int(&cap()?)?;
    Ok(V::Chan(task_fresh("chan", line), n))
}
thread_local! {
    /// sends buffered by an `atomic` block, and by an actor handler (C.11.9, 6.4)
    static ATOMIC: RefCell<Option<Vec<(V, V)>>> = const { RefCell::new(None) };
    static HSENDS: RefCell<Option<Vec<(V, V)>>> = const { RefCell::new(None) };
}
pub fn send(t: V, v: V) -> R<()> {
    charge(8)?;
    let t = match t { V::Session(out, _, cap) => V::Chan(out, cap), o => o };
    let pair = (t, v);
    let pair = match ATOMIC.with(|a| match a.borrow_mut().as_mut() { Some(b) => { b.push(pair); None } None => Some(pair) }) { Some(p) => p, None => return Ok(()) };
    let pair = match HSENDS.with(|a| match a.borrow_mut().as_mut() { Some(b) => { b.push(pair); None } None => Some(pair) }) { Some(p) => p, None => return Ok(()) };
    comm(Op::Send(pair.0, pair.1))?;
    Ok(())
}
pub fn recv(c: V) -> R<V> {
    charge(8)?;
    let (ch, session) = match c { V::Session(_, inn, cap) => (V::Chan(inn, cap), true), o => (o, false) };
    match comm(Op::Recv(ch))? {
        Rep::Val(V::Opt(Some(v))) if session => Ok(*v),
        Rep::Val(_) if session => trap("session closed"),
        Rep::Val(v) => Ok(v),
        _ => trap("bad receive"),
    }
}
pub fn atomic_enter() -> Option<Vec<(V, V)>> { ATOMIC.with(|a| a.borrow_mut().replace(vec![])) }
/// end of an `atomic` block: restore the outer buffer, then commit (or hand
/// the sends to the outer block)
pub fn atomic_exit(outer: Option<Vec<(V, V)>>, r: R<()>) -> R<()> {
    let buf = ATOMIC.with(|a| std::mem::replace(&mut *a.borrow_mut(), outer)).unwrap_or_default();
    r?;
    if ATOMIC.with(|a| match a.borrow_mut().as_mut() { Some(o) => { o.extend(buf.clone()); true } None => false }) { return Ok(()); }
    if in_task() { comm(Op::Atomic(buf))?; return Ok(()); }
    trap("concurrency is not available at comptime")
}
/// stable path of C.11.11: supervisor position (or type), creating task, ordinal
pub fn actor_new(ty: &str, pos: &str) -> R<V> {
    if !in_task() { return trap(format!("cannot call {ty} at comptime")); }
    let id = TASK.with(|t| { let mut t = t.borrow_mut(); let cx = t.as_mut().unwrap(); cx.created += 1; let k = cx.ords.entry(format!("actor|{pos}")).or_insert(0); *k += 1; hid(&format!("actor|{pos}|{}|{}", cx.id, *k - 1)) });
    Ok(V::Actor(id, ty.to_string()))
}
pub fn session_new(cap: &dyn Fn() -> R<V>, line: usize) -> R<V> {
    if !in_task() { return trap("channels are not available at comptime"); }
    let n = marker_int(&cap()?)?;
    let a = task_fresh("chan", line);
    let b = task_fresh("chan", line);
    Ok(V::Tuple(vec![V::Session(a.clone(), b.clone(), n), V::Session(b, a, n)]))
}
/// `c.choose(Label)`: the label is not evaluated
pub fn session_choose(r: V, label: &str) -> R<V> {
    let V::Session(out, _, cap) = r else { return trap(format!("{} has no method choose at comptime", show(&r))) };
    charge(8)?;
    comm(Op::Send(V::Chan(out, cap), V::Rec(label.to_string(), vec![])))?;
    Ok(V::Unit)
}
pub fn spawn_(f: &str, args: Vec<V>, prio: i128, site: usize) -> R<V> {
    charge(32)?;
    if !in_task() { return trap("concurrency is not available at comptime"); }
    let (idx, scope) = TASK.with(|t| { let mut t = t.borrow_mut(); let cx = t.as_mut().unwrap(); cx.created += 1; let k = cx.ords.entry("spawn".to_string()).or_insert(0); *k += 1; (*k - 1, cx.scopes.last().copied().unwrap_or(0)) });
    match comm(Op::Spawn { f: f.to_string(), args, prio: prio.clamp(0, 255) as u8, site, idx, scope })? { Rep::Spawned(id) => Ok(V::Task(id)), _ => trap("bad spawn") }
}
pub fn select_(arms: Vec<SArm>, line: usize, nondet: bool) -> R<(usize, V)> {
    let arms = arms.into_iter().map(|a| match a {
        SArm::Recv(V::Session(_, inn, cap), c) => SArm::Recv(V::Chan(inn, cap), c),
        SArm::Send(V::Session(out, _, cap), v) => SArm::Send(V::Chan(out, cap), v),
        o => o,
    }).collect();
    let k = TASK.with(|t| t.borrow_mut().as_mut().map(|cx| { let e = cx.sel_k.entry(line).or_insert(0); let k = *e; *e += 1; k }).unwrap_or(0));
    match comm(Op::Select { arms, key: (line, k), nondet })? { Rep::Arm(i, v) => Ok((i, v.unwrap_or(V::Opt(None)))), _ => trap("bad select") }
}
pub fn sarm_after(n: V) -> R<SArm> { Ok(SArm::After(marker_int(&n)?.max(0) as u64)) }
/// `with deadline n`: (frame id); the body is cancelled at a communication step after tick now + n
pub fn dl_enter(n: i128) -> u64 {
    (TASK.with(|t| { let mut t = t.borrow_mut(); match t.as_mut() { Some(cx) => { let f = cx.next_frame; cx.next_frame += 1; let e = cx.now.saturating_add(n.max(0) as u64); cx.dl.push((f, e)); f } None => u64::MAX - 1 } }))

}
pub fn dl_exit() { TASK.with(|t| if let Some(cx) = t.borrow_mut().as_mut() { cx.dl.pop(); }); }
pub fn scope_enter() -> u64 { TASK.with(|t| { let mut t = t.borrow_mut(); match t.as_mut() { Some(cx) => { let f = cx.next_frame; cx.next_frame += 1; cx.scopes.push(f); f } None => 0 } }) }
pub fn scope_exit() { TASK.with(|t| if let Some(cx) = t.borrow_mut().as_mut() { cx.scopes.pop(); }); }
pub fn await_scope(sid: u64) -> R<()> { if !in_task() { return Ok(()); } comm(Op::AwaitScope(sid)).map(|_| ()) }

#[derive(Default)]
struct ChanSt { cap: i128, buf: VecDeque<V>, held: VecDeque<(V, Option<usize>)>, closed: bool }
enum St { Ready(Rep), Pending(Req), Done }
struct Slot { id: String, prio: u8, rx: mpsc::Receiver<Req>, tx: mpsc::Sender<(u64, Rep)>, st: St, parent: Option<usize>, scope: u64, result: Option<Result<V, String>>, joined: bool, cancel: bool, holding: Option<String>, released: bool, since: u64 }
pub enum Outcome { Done, Refused(String), Deadlock }
struct ActorSt { ty: String, fields: Vec<(&'static str, V)>, mailbox: VecDeque<V>, node: usize }
type HandlerOut = Option<(Vec<(&'static str, V)>, Vec<V>, Vec<(V, V)>)>;
struct Sched {
    slots: Vec<Slot>, chans: BTreeMap<String, ChanSt>, staged: Vec<(String, V)>, outputs: Vec<V>, rng: u64, tick: u64, stop: Option<Outcome>, max_ticks: u64,
    actors: BTreeMap<String, ActorSt>, mail_staged: Vec<(String, String, V)>, txns: Vec<Vec<(String, String, V)>>, txn_staged: Vec<Vec<(String, String, V)>>,
    held_links: Vec<(usize, usize, V, V)>, parted: Vec<(usize, usize)>, nodes: Vec<&'static str>, links: Vec<(u64, bool, String, String)>, restarts: HashMap<(String, String), Vec<u64>>, rlog: Vec<(u64, String, String, String, bool, Vec<String>)>, rrep: Option<std::collections::VecDeque<(u64, String, String, String, bool, Vec<String>)>>,
    /// actor log entries the interpreter also logs: Refused, Stopped, Restart, Input, Unhandled
    alog: Vec<String>,
    /// program events (HC-6): (tick, event value)
    events: Vec<(u64, V)>,
}
impl Sched {
    fn spawn_task(&mut self, f: String, args: Vec<V>, id: String, prio: u8, parent: Option<usize>, scope: u64) {
        let (req_tx, req_rx) = mpsc::channel::<Req>();
        let (rep_tx, rep_rx) = mpsc::channel::<(u64, Rep)>();
        let tid = id.clone();
        std::thread::Builder::new().stack_size(256 << 20).spawn(move || {
            let tx = req_tx.clone();
            let first = rep_rx.recv();
            TASK.with(|t| *t.borrow_mut() = Some(TaskCx { tx: req_tx, rx: rep_rx, id: tid, created: 0, ords: HashMap::new(), now: 0, dl: vec![], next_frame: 1, scopes: vec![0], sel_k: HashMap::new() }));
            let r = match first {
                Ok((tick, Rep::Go)) => {
                    TASK.with(|t| t.borrow_mut().as_mut().unwrap().now = tick);
                    // the reference starts a task with call_fn on its arguments as names: one tick for the call and one per argument
                    match (0..=args.len()).try_for_each(|_| fuel()).and_then(|_| user_fn(&f, None, args)) { Ok(c) => await_scope(0).map(|_| c.v), Err(e) => Err(e) }
                }
                _ => Err(Flow::Cancel(CANCEL_TASK)),
            };
            let _ = tx.send(Req { op: Op::Done(r), dl: vec![] });
        }).expect("task thread");
        self.slots.push(Slot { id, prio, rx: req_rx, tx: rep_tx, st: St::Ready(Rep::Go), parent, scope, result: None, joined: false, cancel: false, holding: None, released: false, since: 0 });
    }
    fn order(&self) -> Vec<usize> {
        let mut v: Vec<usize> = (0..self.slots.len()).filter(|&i| !matches!(self.slots[i].st, St::Done)).collect();
        v.sort_by(|&a, &b| { let (x, y) = (&self.slots[a], &self.slots[b]); y.prio.cmp(&x.prio).then_with(|| x.id.cmp(&y.id)) });
        v
    }
    fn chan(&mut self, v: &V) -> Option<String> {
        if let V::Chan(id, cap) = v { self.chans.entry(id.clone()).or_insert_with(|| ChanSt { cap: *cap, ..Default::default() }); Some(id.clone()) } else { None }
    }
    fn space(&self, c: &str) -> bool {
        let st = &self.chans[c];
        let staged = self.staged.iter().filter(|(x, _)| x == c).count() as i128;
        (st.buf.len() as i128) + staged < st.cap
    }
    fn take(&mut self, c: &str) -> Option<V> {
        let st = self.chans.get_mut(c)?;
        if let Some(m) = st.buf.pop_front() {
            if let Some((h, owner)) = st.held.pop_front() { st.buf.push_back(h); if let Some(o) = owner { self.slots[o].released = true; } }
            return Some(m);
        }
        if let Some((h, owner)) = st.held.pop_front() { if let Some(o) = owner { self.slots[o].released = true; } return Some(h); }
        None
    }
    fn readable(&self, c: &str) -> (bool, bool) { let st = &self.chans[c]; let has = !st.buf.is_empty() || !st.held.is_empty(); (has, !has && st.closed) }
    fn route(&mut self, from: usize, target: V, m: V) {
        let to = match &target { V::Actor(_, ty) => node_of(ty), _ => 0 };
        let key = (from.min(to), from.max(to));
        if from != to && self.parted.contains(&key) { self.held_links.push((from, to, target, m)); return; }
        match target {
            V::Actor(id, ty) => self.mail_staged.push((id, ty, m)),
            V::Chan(..) => {
                if let Some(c) = self.chan(&target) {
                    if self.space(&c) { self.staged.push((c, m)); } else { self.chans.get_mut(&c).unwrap().held.push_back((m, None)); }
                }
            }
            _ => {}
        }
    }
    fn ensure_actor(&mut self, id: &str, ty: &str) {
        if !self.actors.contains_key(id) {
            self.actors.insert(id.to_string(), ActorSt { ty: ty.to_string(), fields: actor_init(ty), mailbox: VecDeque::new(), node: node_of(ty) });
        }
    }
    fn handler(&mut self, id: &str, msg: &V) -> Result<HandlerOut, String> {
        let st = &self.actors[id];
        let (ty, fields) = (st.ty.clone(), st.fields.clone());
        for h in actor_handlers(&ty) {
            take_out();
            HSENDS.with(|a| *a.borrow_mut() = Some(vec![]));
            let r = h(&fields, msg);
            let sends = HSENDS.with(|a| a.borrow_mut().take()).unwrap_or_default();
            match r {
                Ok(None) => continue,
                Ok(Some(fs)) => return Ok(Some((fs, take_out(), sends))),
                Err(f) => { take_out(); return Err(actor_refusal(&f)); }
            }
        }
        Ok(None)
    }
    fn on_crash(&mut self, id: &str, reason: &str) {
        // restart as input (C.11.11): record the decision, or on replay
        // take it from the logged inputs (--restart=...) without the policy
        self.alog.push(format!("Refused({reason})"));
        let ty = self.actors[id].ty.clone();
        let Some((sup, strat, tys, child, bound)) = supervisor_of(&ty) else { return };
        let (tick, w) = (self.tick, restart_window());
        let dec = if let Some(q) = self.rrep.as_mut() {
            match q.pop_front() {
                Some(r) if !sup_tree().contains(&format!("{}/{}", r.1, r.2).as_str()) => { self.stop = Some(Outcome::Refused(format!("trap: upgrade removed supervisor position {}/{}", r.1, r.2))); return; }
                Some(r) if r.3 == id && r.1 == sup => r,
                _ => { self.stop = Some(Outcome::Refused("replay diverged".into())); return; }
            }
        } else {
            let h = self.restarts.entry((sup.to_string(), child.to_string())).or_default();
            if w > 0 { h.retain(|&t| tick.saturating_sub(t) < w); }
            h.push(tick);
            let stop = h.len() as u32 > bound;
            let pos = tys.iter().position(|t| *t == ty).unwrap_or(0);
            let reset: Vec<String> = if stop { vec![] } else { self.actors.iter().filter(|(aid, a)| match strat {
                "one_for_all" => tys.contains(&a.ty.as_str()),
                "rest_for_one" => tys.iter().position(|t| *t == a.ty).is_some_and(|p| p >= pos),
                _ => aid.as_str() == id,
            }).map(|(aid, _)| aid.clone()).collect() };
            (tick, sup.to_string(), child.to_string(), id.to_string(), stop, reset)
        };
        self.rlog.push(dec.clone());
        let count = self.rlog.iter().filter(|r| r.1 == sup && r.2 == child && (w == 0 || tick.saturating_sub(r.0) < w)).count();
        if dec.4 {
            self.alog.push(format!("Stopped(supervisor {sup})"));
            self.stop = Some(Outcome::Refused(format!("supervisor {sup} stopped: child {child} restarted more than {bound} times")));
            return;
        }
        self.alog.push(format!("Restart({child}, {id}, {reason}, tick {tick}, {count})"));
        for aid in dec.5 { if let Some(t) = self.actors.get(&aid).map(|a| a.ty.clone()) { self.actors.get_mut(&aid).unwrap().fields = actor_init(&t); } }
    }
    fn commit_handler(&mut self, id: &str, r: HandlerOut) {
        let Some((fs, outs, sends)) = r else { return };
        let node = self.actors[id].node;
        self.actors.get_mut(id).unwrap().fields = fs;
        self.outputs.extend(outs);
        for (t, v) in sends { self.route(node, t, v); }
    }
    fn actors_phase(&mut self) -> bool {
        let mut progress = false;
        let mut busy = std::collections::BTreeSet::new();
        for txn in std::mem::take(&mut self.txns) {
            progress = true;
            let mut results = vec![];
            let mut refused = None;
            for (id, ty, m) in &txn {
                self.ensure_actor(id, ty);
                busy.insert(id.clone());
                match self.handler(id, m) { Ok(r) => results.push((id.clone(), r)), Err(e) => { refused = Some(e); break; } }
            }
            if let Some(e) = refused {
                let failed = results.len();
                for (k, (id, _, _)) in txn.iter().enumerate() { if k == failed { self.on_crash(id, &e); } else { self.alog.push(format!("Refused({e})")); } }
                if self.stop.is_some() { break; }
                continue;
            }
            for (id, r) in results { self.commit_handler(&id, r); }
        }
        let ids: Vec<String> = self.actors.keys().cloned().collect();
        for id in ids {
            if busy.contains(&id) { continue; }
            let Some(m) = self.actors.get_mut(&id).and_then(|a| a.mailbox.pop_front()) else { continue };
            progress = true;
            match self.handler(&id, &m) { Ok(r) => self.commit_handler(&id, r), Err(e) => self.on_crash(&id, &e) }
            if self.stop.is_some() { break; }
        }
        progress
    }
    /// HC-6: an event at this tick is logged as Input and staged to every
    /// live actor that handles it, in actor id order; else Unhandled
    fn events_phase(&mut self) -> bool {
        let evs: Vec<V> = self.events.iter().filter(|e| e.0 == self.tick).map(|e| e.1.clone()).collect();
        let any = !evs.is_empty();
        for v in evs {
            let name = match &v { V::Rec(n, _) => n.clone(), _ => String::new() };
            self.alog.push(format!("Input({})", show(&v)));
            let to: Vec<(String, String)> = self.actors.iter().filter(|(_, a)| actor_events(&a.ty).contains(&name.as_str())).map(|(id, a)| (id.clone(), a.ty.clone())).collect();
            if to.is_empty() { self.alog.push(format!("Unhandled({})", show(&v))); }
            for (id, ty) in to { self.mail_staged.push((id, ty, v.clone())); }
        }
        any
    }
    fn links_phase(&mut self) -> bool {
        let mut any = false;
        let evs: Vec<(u64, bool, String, String)> = self.links.iter().filter(|e| e.0 == self.tick).cloned().collect();
        for (_, part, a, b) in evs {
            let (Some(x), Some(y)) = (self.nodes.iter().position(|n| *n == a), self.nodes.iter().position(|n| *n == b)) else { continue };
            any = true;
            let key = (x.min(y), x.max(y));
            if part { if !self.parted.contains(&key) { self.parted.push(key); } } else {
                self.parted.retain(|k| *k != key);
                for (f, t, target, m) in std::mem::take(&mut self.held_links) {
                    if (f.min(t), f.max(t)) == key { self.route(f, target, m); } else { self.held_links.push((f, t, target, m)); }
                }
            }
        }
        any
    }
    fn task_failed_value(&self, i: usize) -> V {
        let reason = match &self.slots[i].result { Some(Err(r)) => r.clone(), _ => String::new() };
        V::Rec("TaskFailed".into(), vec![("task".into(), V::Text(self.slots[i].id.clone())), ("reason".into(), V::Text(reason))])
    }
    fn pick_select(&mut self, arms: &[SArm], key: (usize, u64), nondet: bool) -> Option<usize> {
        let mut ready = vec![];
        for (k, a) in arms.iter().enumerate() {
            let ok = match a {
                SArm::Recv(cv, cls) => match self.chan(cv) { Some(c) => { let (has, closed) = self.readable(&c); (has && *cls != 2) || (closed && *cls != 1) } None => false },
                SArm::Send(cv, _) => match self.chan(cv) { Some(c) => self.space(&c) && !self.chans[&c].closed, None => false },
                _ => false,
            };
            if ok { ready.push(k); }
        }
        if ready.is_empty() { return None; }
        if !nondet { let n = arms.len(); let s = (key.1 % n as u64) as usize; return (0..n).map(|j| (s + j) % n).find(|j| ready.contains(j)); }
        self.rng ^= self.rng << 13; self.rng ^= self.rng >> 7; self.rng ^= self.rng << 17;
        Some(ready[(self.rng % ready.len() as u64) as usize])
    }
    fn handle(&mut self, i: usize, req: &Req) -> Option<Rep> {
        if let Some((fid, _)) = req.dl.iter().find(|(_, e)| self.tick > *e) {
            if let Some(c) = self.slots[i].holding.take() { if let Some(st) = self.chans.get_mut(&c) { st.held.retain(|(_, o)| *o != Some(i)); } }
            self.slots[i].released = false;
            self.slots[i].since = 0;
            return Some(Rep::Cancel(*fid));
        }
        if self.slots[i].cancel { return Some(Rep::Cancel(CANCEL_TASK)); }
        match &req.op {
            Op::Emit(v) => { self.outputs.push(v.clone()); Some(Rep::Val(V::Unit)) }
            Op::Send(target, v) => {
                if let V::Actor(..) = target { self.route(0, target.clone(), v.clone()); return Some(Rep::Val(V::Unit)); }
                let c = self.chan(target)?;
                if self.slots[i].holding.is_some() {
                    if self.slots[i].released { self.slots[i].released = false; self.slots[i].holding = None; return Some(Rep::Val(V::Unit)); }
                    return None;
                }
                if self.chans[&c].closed { self.stop = Some(Outcome::Refused("trap: send on a closed channel".into())); return Some(Rep::Cancel(CANCEL_TASK)); }
                if self.space(&c) { self.staged.push((c, v.clone())); Some(Rep::Val(V::Unit)) } else {
                    self.chans.get_mut(&c).unwrap().held.push_back((v.clone(), Some(i)));
                    self.slots[i].holding = Some(c);
                    None
                }
            }
            Op::Recv(cv) => {
                let c = self.chan(cv)?;
                let (has, closed) = self.readable(&c);
                if has { let m = self.take(&c).unwrap(); Some(Rep::Val(V::Opt(Some(Box::new(m))))) } else if closed { Some(Rep::Val(V::Opt(None))) } else { None }
            }
            Op::Close(cv) => { let c = self.chan(cv)?; self.chans.get_mut(&c).unwrap().closed = true; Some(Rep::Val(V::Unit)) }
            Op::Select { arms, key, nondet } => {
                if let Some(k) = self.pick_select(arms, *key, *nondet) {
                    self.slots[i].since = 0;
                    return Some(match &arms[k] {
                        SArm::Recv(cv, _) => { let c = self.chan(cv).unwrap(); match self.take(&c) { Some(m) => Rep::Arm(k, Some(V::Opt(Some(Box::new(m))))), None => Rep::Arm(k, Some(V::Opt(None))) } }
                        SArm::Send(cv, v) => { let c = self.chan(cv).unwrap(); self.staged.push((c, v.clone())); Rep::Arm(k, None) }
                        _ => Rep::Arm(k, None),
                    });
                }
                if let Some(d) = arms.iter().position(|a| matches!(a, SArm::Default)) { return Some(Rep::Arm(d, None)); }
                if let Some((k, n)) = arms.iter().enumerate().find_map(|(k, a)| match a { SArm::After(n) => Some((k, *n)), _ => None }) {
                    if self.slots[i].since == 0 { self.slots[i].since = self.tick; }
                    if self.tick - self.slots[i].since >= n { self.slots[i].since = 0; return Some(Rep::Arm(k, None)); }
                }
                None
            }
            Op::Spawn { idx, .. } => Some(Rep::Spawned(hid(&format!("spawn|{}|{idx}", self.slots[i].id)))),
            Op::Join(id) => {
                let j = self.slots.iter().position(|s| &s.id == id)?;
                if !matches!(self.slots[j].st, St::Done) { return None; }
                self.slots[j].joined = true;
                Some(Rep::Val(match &self.slots[j].result { Some(Ok(v)) => v.clone(), _ => V::Err(Box::new(self.task_failed_value(j))) }))
            }
            Op::AwaitScope(sid) => {
                let kids: Vec<usize> = (0..self.slots.len()).filter(|&k| self.slots[k].parent == Some(i) && self.slots[k].scope == *sid).collect();
                if kids.iter().any(|&k| !matches!(self.slots[k].st, St::Done)) { return None; }
                for &k in &kids {
                    if matches!(self.slots[k].result, Some(Err(_))) && !self.slots[k].joined { self.slots[k].joined = true; return Some(Rep::Failed(self.task_failed_value(k))); }
                }
                Some(Rep::Val(V::Unit))
            }
            Op::Atomic(sends) => {
                let chans: Vec<String> = sends.iter().filter_map(|(t, _)| self.chan(t)).collect();
                for c in &chans {
                    let need = chans.iter().filter(|x| *x == c).count() as i128;
                    let st = &self.chans[c];
                    let staged = self.staged.iter().filter(|(x, _)| x == c).count() as i128;
                    if st.closed || st.buf.len() as i128 + staged + need > st.cap { return None; }
                }
                let mut txn = vec![];
                for (t, v) in sends {
                    match t {
                        V::Actor(id, ty) => txn.push((id.clone(), ty.clone(), v.clone())),
                        _ => { let c = self.chan(t).unwrap(); self.staged.push((c, v.clone())); }
                    }
                }
                if !txn.is_empty() { self.txn_staged.push(txn); }
                Some(Rep::Val(V::Unit))
            }
            Op::Done(_) => None,
        }
    }
    fn finish_task(&mut self, i: usize, r: &Result<V, Flow>) -> Option<Outcome> {
        let res = match r { Ok(v) => Ok(v.clone()), Err(Flow::Cancel(_)) => Err("cancelled".to_string()), Err(f) => Err(why(f)) };
        let failed = matches!(r, Err(f) if !matches!(f, Flow::Cancel(_)));
        self.slots[i].result = Some(res.clone());
        if i == 0 { if let Err(e) = res { return Some(Outcome::Refused(e)); } } else if failed {
            let (p, sc) = (self.slots[i].parent, self.slots[i].scope);
            for k in 0..self.slots.len() { if k != i && self.slots[k].parent == p && self.slots[k].scope == sc { self.slots[k].cancel = true; } }
        }
        for k in 0..self.slots.len() { if self.slots[k].parent == Some(i) { self.slots[k].cancel = true; } }
        None
    }
    fn run(&mut self) -> Outcome {
        if let Some(o) = self.stop.take() { return o; }
        loop {
            self.tick += 1;
            if self.tick > self.max_ticks { return Outcome::Refused("trap: tick limit".into()); }
            let mut progress = self.links_phase();
            progress |= self.events_phase();
            let order = self.order();
            let mut moved = vec![];
            for &i in &order {
                if let St::Ready(_) = &self.slots[i].st {
                    let St::Ready(rep) = std::mem::replace(&mut self.slots[i].st, St::Done) else { unreachable!() };
                    let _ = self.slots[i].tx.send((self.tick, rep));
                    moved.push(i);
                }
            }
            for &i in &moved {
                progress = true;
                match self.slots[i].rx.recv() { Ok(req) => self.slots[i].st = St::Pending(req), Err(_) => { self.slots[i].st = St::Done; self.slots[i].result = Some(Err("task stopped".into())); } }
            }
            progress |= self.actors_phase();
            if let Some(o) = self.stop.take() { return o; }
            for &i in &order {
                let St::Pending(req) = std::mem::replace(&mut self.slots[i].st, St::Done) else { continue };
                if let Op::Done(r) = &req.op { progress = true; if let Some(o) = self.finish_task(i, r) { return o; } continue; }
                let rep = self.handle(i, &req);
                if let Some(o) = self.stop.take() { return o; }
                let Some(rep) = rep else { self.slots[i].st = St::Pending(req); continue };
                progress = true;
                if let (Op::Spawn { f, args, prio, scope, .. }, Rep::Spawned(id)) = (&req.op, &rep) {
                    if !has_user_fn(f) { return Outcome::Refused(format!("trap: no task function {f}")); }
                    self.spawn_task(f.clone(), args.clone(), id.clone(), *prio, Some(i), *scope);
                }
                self.slots[i].st = St::Ready(rep);
            }
            for (c, m) in std::mem::take(&mut self.staged) { self.chans.get_mut(&c).unwrap().buf.push_back(m); }
            for (id, ty, m) in std::mem::take(&mut self.mail_staged) { self.ensure_actor(&id, &ty); self.actors.get_mut(&id).unwrap().mailbox.push_back(m); }
            let t = std::mem::take(&mut self.txn_staged);
            self.txns.extend(t);
            let all_done = self.slots.iter().all(|s| matches!(s.st, St::Done));
            let actors_idle = self.actors.values().all(|a| a.mailbox.is_empty()) && self.txns.is_empty() && self.mail_staged.is_empty();
            let future_events = self.events.iter().any(|e| e.0 > self.tick);
            let future_links = self.links.iter().any(|e| e.0 > self.tick) || future_events;
            if all_done && actors_idle && !future_events && (self.held_links.is_empty() || !future_links) { return Outcome::Done; }
            if !progress && actors_idle && !future_links {
                let timers = self.slots.iter().any(|s| match &s.st { St::Pending(r) => !r.dl.is_empty() || matches!(&r.op, Op::Select { arms, .. } if arms.iter().any(|a| matches!(a, SArm::After(_)))), _ => false });
                if !timers { return Outcome::Deadlock; }
            }
        }
    }
}
/// logged restart inputs (C.11.11): `--restart=TICK|SUP|CHILD|ID|STOP|RESET,...`
/// in order; `--restart-replay` alone replays an empty list
fn restart_inputs() -> Option<std::collections::VecDeque<(u64, String, String, String, bool, Vec<String>)>> {
    let v: Vec<_> = std::env::args().filter_map(|a| {
        let r = a.strip_prefix("--restart=")?.to_string();
        let p: Vec<&str> = r.split('|').collect();
        Some((p.first()?.parse().ok()?, p.get(1)?.to_string(), p.get(2)?.to_string(), p.get(3)?.to_string(), *p.get(4)? == "1",
            p.get(5).map(|x| x.split(',').filter(|y| !y.is_empty()).map(String::from).collect()).unwrap_or_default()))
    }).collect();
    if v.is_empty() && !std::env::args().any(|a| a == "--restart-replay") { None } else { Some(v.into()) }
}
/// program events (HC-6): `--event=TICK|Name|field=value,...`; a field's
/// value is a decimal integer, `true`/`false`, or text, by its declared type
fn event_inputs() -> Vec<(u64, V)> {
    let mut v = vec![];
    for a in std::env::args() {
        let Some(r) = a.strip_prefix("--event=") else { continue };
        let p: Vec<&str> = r.splitn(3, '|').collect();
        fn bad_event(r: &str, m: String) -> ! { eprintln!("bad --event {r}: {m}"); std::process::exit(2) }
        let tick: u64 = p.first().and_then(|x| x.parse().ok()).unwrap_or_else(|| bad_event(r, "tick".into()));
        let name = p.get(1).copied().unwrap_or_else(|| bad_event(r, "name".into()));
        let Some(decl) = event_fields(name) else { bad_event(r, format!("no event {name}")) };
        let given: Vec<(&str, &str)> = p.get(2).map(|x| x.split(',').filter(|y| !y.is_empty()).filter_map(|y| y.split_once('=')).collect()).unwrap_or_default();
        let mut fs = vec![];
        for (f, t) in decl {
            let Some((_, x)) = given.iter().find(|(g, _)| *g == f) else { bad_event(r, format!("missing field {f}")) };
            let val = match t {
                "bool" => V::Bool(*x == "true"),
                "text" => V::Text(x.to_string()),
                "i8" | "i16" | "i32" | "i64" | "i128" | "u8" | "u16" | "u32" | "u64" | "usize" => V::Int(x.parse().unwrap_or_else(|_| bad_event(r, format!("field {f}"))), t),
                _ => bad_event(r, format!("field {f} has type {t}, not an integer, bool or text")),
            };
            fs.push((f.to_string(), val));
        }
        v.push((tick, V::Rec(name.to_string(), fs)));
    }
    v
}
/// `--restart-tree=Sup/child,...`: the schema of the replayed restart log
fn restart_tree_arg() -> Option<Vec<String>> {
    std::env::args().find_map(|a| a.strip_prefix("--restart-tree=").map(|r| r.split(',').filter(|x| !x.is_empty()).map(String::from).collect()))
}
/// migration rule (C.11.11): every recorded position keeps its supervisor
/// and child; new ones only after the recorded ones (`conc::tree_migrates`)
fn tree_migrates(rec: &[String], cur: &[&str]) -> Result<(), String> {
    fn group<S: AsRef<str>>(v: &[S]) -> Vec<(String, Vec<String>)> {
        let mut g: Vec<(String, Vec<String>)> = vec![];
        for p in v {
            let p = p.as_ref();
            let (s, c) = p.split_once('/').unwrap_or((p, ""));
            match g.last_mut() { Some((x, cs)) if x == s => cs.push(c.to_string()), _ => g.push((s.to_string(), vec![c.to_string()])) }
        }
        g
    }
    let (r, c) = (group(rec), group(cur));
    for (k, (sup, kids)) in r.iter().enumerate() {
        for (j, kid) in kids.iter().enumerate() {
            if !c.get(k).is_some_and(|(s, cs)| s == sup && cs.get(j) == Some(kid)) {
                let p = format!("{sup}/{kid}");
                return Err(if cur.contains(&p.as_str()) { format!("trap: upgrade reordered supervisor position {p}") } else { format!("trap: upgrade removed supervisor position {p}") });
            }
        }
    }
    Ok(())
}
/// `--run`: `fn main` on the tick runtime; prints the outputs and the outcome
pub fn run_tasks(main: &str) {
    // logged link inputs: --link=TICK,partition|heal,NODE,NODE
    let links = std::env::args().filter_map(|a| {
        let r = a.strip_prefix("--link=")?.to_string();
        let p: Vec<&str> = r.split(',').collect();
        Some((p.first()?.parse().ok()?, *p.get(1)? == "partition", p.get(2)?.to_string(), p.get(3)?.to_string()))
    }).collect();
    let mut s = Sched { slots: vec![], chans: BTreeMap::new(), staged: vec![], outputs: vec![], rng: 0x5eed | 1, tick: 0, stop: None, max_ticks: 100_000,
        actors: BTreeMap::new(), mail_staged: vec![], txns: vec![], txn_staged: vec![], held_links: vec![], parted: vec![], nodes: node_names(), links, restarts: HashMap::new(), alog: vec![], rlog: vec![], rrep: restart_inputs(), events: event_inputs() };
    if s.rrep.is_some() { if let Some(rec) = restart_tree_arg() { if let Err(e) = tree_migrates(&rec, &sup_tree()) { s.stop = Some(Outcome::Refused(e)); } } }
    s.spawn_task(main.to_string(), vec![], hid("main"), 0, None, 0);
    let o = s.run();
    let tick = s.tick;
    for sl in s.slots.iter_mut() {
        while !matches!(sl.st, St::Done) {
            if let St::Pending(Req { op: Op::Done(_), .. }) = sl.st { sl.st = St::Done; continue; }
            let _ = sl.tx.send((tick, Rep::Cancel(CANCEL_TASK)));
            match sl.rx.recv() { Ok(Req { op: Op::Done(_), .. }) | Err(_) => sl.st = St::Done, Ok(r) => sl.st = St::Pending(r) }
        }
    }
    for v in &s.outputs { println!("emit {}", show(v)); }
    if std::env::args().any(|a| a == "--actor-log") { for e in &s.alog { println!("actorlog {e}"); } }
    if std::env::args().any(|a| a == "--restart-log") { println!("restartlog v1 tree={}", sup_tree().join(",")); for r in &s.rlog { println!("restartin {}|{}|{}|{}|{}|{}", r.0, r.1, r.2, r.3, u8::from(r.4), r.5.join(",")); } }
    match o { Outcome::Done => println!("outcome Done"), Outcome::Deadlock => println!("outcome Deadlock"), Outcome::Refused(e) => println!("outcome Refused: {e}") }
}

/// Decimal text of an integral finite f64, exact.
fn exact_int_text(t: f64) -> String {
    let bits = t.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let mut mant = bits & ((1u64 << 52) - 1);
    if exp != 0 { mant |= 1u64 << 52; }
    let e = if exp == 0 { -1074 } else { exp - 1075 };
    // base 1e9 limbs, little-endian
    let mut limbs: Vec<u64> = vec![mant % 1_000_000_000, mant / 1_000_000_000 % 1_000_000_000, mant / 1_000_000_000_000_000_000];
    let mut sh = e;
    while sh > 0 {
        let k = sh.min(29) as u32;
        let mut carry = 0u64;
        for l in limbs.iter_mut() {
            let v = (*l << k) + carry;
            *l = v % 1_000_000_000;
            carry = v / 1_000_000_000;
        }
        while carry > 0 { limbs.push(carry % 1_000_000_000); carry /= 1_000_000_000; }
        sh -= k as i32;
    }
    while limbs.len() > 1 && *limbs.last().unwrap() == 0 { limbs.pop(); }
    let mut s = if t < 0.0 { "-".to_string() } else { String::new() };
    s += &limbs.last().unwrap().to_string();
    for l in limbs.iter().rev().skip(1) { s += &format!("{l:09}"); }
    s
}

fn trunc_chk(f: f64, n: &'static str) -> R<V> {
    let t = f.trunc();
    if t.abs() < 1.7014118346046923e38 { return chk(t as i128, n); }
    trap(format!("overflow: {} is outside {n}", exact_int_text(t)))
}

/// N9: a list or text built by any operation has at most 1,000,000 elements.
pub fn size_ok(v: V) -> R<V> {
    match &v {
        V::List(xs) if xs.len() > 1_000_000 => trap("size limit"),
        V::Text(t) if t.len() > 1_000_000 && t.chars().count() > 1_000_000 => trap("size limit"),
        _ => Ok(v),
    }
}
