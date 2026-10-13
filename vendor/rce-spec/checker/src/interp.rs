//! Reference evaluator for admission (spec 6.2–6.4, 6.9; rules W-INT-1, W-CT-1).
//! Evaluates consts, state initializers, `check`s, comptime blocks and the
//! examples of every intent, following the edge semantics of 6.3: integer
//! results are checked against their type's range (trap), division by zero
//! traps, loops trap at their bound, and an event is refused atomically.
use crate::ast::*;
use crate::types::{INTS, int_range};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub enum V {
    Int(i128, &'static str), // "lit" = an integer literal not yet given a type
    F(f64),
    Bool(bool),
    Text(String),
    Unit,
    Tuple(Vec<V>),
    List(Vec<V>),
    Map(Vec<(V, V)>),
    Opt(Option<Box<V>>),
    Rec(String, Vec<(String, V)>), // struct, kind variant, event, error
    Err(Box<V>),                   // the error side of an error union
    Fn(String),
    Closure(Vec<Param>, Box<Expr>, Vec<HashMap<String, V>>),
    Range(i128, i128, &'static str),
    F32(f64),   // an f32 value, held exactly in an f64
    Gen(usize), // a suspended iterator call (index into Interp::gens)
}
impl PartialEq for V {
    fn eq(&self, o: &V) -> bool {
        use V::*;
        match (self, o) {
            (Int(a, _), Int(b, _)) => a == b,
            (F(a), F(b)) => a == b,
            (F32(a), F32(b)) => a == b,
            (Bool(a), Bool(b)) => a == b,
            (Text(a), Text(b)) => a == b,
            (Unit, Unit) => true,
            (Tuple(a), Tuple(b)) | (List(a), List(b)) => a == b,
            (Map(a), Map(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .all(|(k, v)| b.iter().any(|(k2, v2)| k == k2 && v == v2))
            }
            (Opt(a), Opt(b)) => a == b,
            (Rec(n, a), Rec(m, b)) => n == m && a == b,
            (Err(a), Err(b)) => a == b,
            _ => false,
        }
    }
}
impl V {
    pub fn show(&self) -> String {
        match self {
            V::Int(v, _) => v.to_string(),
            V::F(f) => f.to_string(),
            V::Bool(b) => b.to_string(),
            V::Text(t) => format!("{t:?}"),
            V::Unit => "()".into(),
            V::Tuple(v) => format!(
                "({})",
                v.iter().map(|x| x.show()).collect::<Vec<_>>().join(", ")
            ),
            V::List(v) => format!(
                "[{}]",
                v.iter().map(|x| x.show()).collect::<Vec<_>>().join(", ")
            ),
            V::Map(v) => format!(
                "{{{}}}",
                v.iter()
                    .map(|(a, b)| format!("{}: {}", a.show(), b.show()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            V::Opt(None) => "none".into(),
            V::Opt(Some(x)) => format!("some({})", x.show()),
            V::Rec(n, f) => {
                if f.is_empty() {
                    n.clone()
                } else {
                    format!(
                        "{n}({})",
                        f.iter()
                            .map(|(a, b)| format!("{a}: {}", b.show()))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            }
            V::Err(e) => format!("error {}", e.show()),
            V::Fn(n) => n.clone(),
            V::Closure(..) => "<fn>".into(),
            V::Range(a, b, _) => format!("{a}..{b}"),
            V::F32(f) => (*f as f32).to_string(),
            V::Gen(_) => "<iter>".into(),
        }
    }
    fn rec_name(&self) -> Option<&str> {
        if let V::Rec(n, _) = self {
            Some(n)
        } else {
            None
        }
    }
}

/// Non-value outcomes of evaluation.
#[derive(Debug, Clone)]
pub enum Flow {
    Ret(V),
    Brk,
    Cont,
    Fail(V),
    Trap(String),
    Stop,
}
type R<T> = Result<T, Flow>;
fn trap<T>(m: impl Into<String>) -> R<T> {
    Err(Flow::Trap(m.into()))
}

pub const LOOP_LIMIT: i128 = 1_000_000;
pub const FUEL: u64 = 50_000_000;

pub struct Interp<'a> {
    pub m: &'a Module,
    fns: HashMap<String, &'a FnDecl>,
    methods: HashMap<(String, String), &'a FnDecl>,
    types: HashMap<String, &'a Item>,
    variants: HashMap<String, Vec<(String, Ty)>>,
    consts: HashMap<String, V>,
    pub scopes: Vec<HashMap<String, V>>,
    fuel: u64,
    pub outputs: Vec<V>,
    pub loop_limit: i128,
    gens: Vec<(&'a FnDecl, HashMap<String, V>)>,
    yields: Vec<(Vec<V>, usize)>,
}

fn int_name(t: &Ty) -> Option<&'static str> {
    if let Ty::Path(p, a) = t
        && p.len() == 1
        && a.is_empty()
    {
        return INTS.iter().find(|x| **x == p[0]).copied();
    }
    None
}

impl<'a> Interp<'a> {
    pub fn new(m: &'a Module) -> Self {
        let mut it = Interp {
            m,
            fns: HashMap::new(),
            methods: HashMap::new(),
            types: HashMap::new(),
            variants: HashMap::new(),
            consts: HashMap::new(),
            scopes: vec![],
            fuel: FUEL,
            outputs: vec![],
            loop_limit: LOOP_LIMIT,
            gens: vec![],
            yields: vec![],
        };
        for i in &m.items {
            match i {
                Item::Fn(f) => {
                    it.fns.insert(f.sig.name.clone(), f);
                }
                Item::Impl { ty, members, .. } => {
                    let h = crate::check::ty_head(ty);
                    for f in members {
                        it.methods.insert((h.clone(), f.sig.name.clone()), f);
                    }
                }
                Item::Struct { name, .. }
                | Item::State { name, .. }
                | Item::Event { name, .. }
                | Item::Error { name, .. }
                | Item::Actor { name, .. } => {
                    it.types.insert(name.clone(), i);
                }
                Item::Kind { name, variants, .. } => {
                    it.types.insert(name.clone(), i);
                    for (v, p) in variants {
                        it.variants.insert(v.clone(), p.clone());
                    }
                }
                Item::TypeAlias { name, .. } => {
                    it.types.insert(name.clone(), i);
                }
                Item::Const {
                    name,
                    e: Expr::Int(n, _),
                    ..
                } if name == "LOOP_LIMIT" && *n > 0 => {
                    it.loop_limit = *n;
                }
                _ => {}
            }
        }
        it
    }
    fn alias(&self, t: &Ty) -> Ty {
        if let Ty::Path(p, _) = t
            && p.len() == 1
            && let Some(Item::TypeAlias { ty, .. }) = self.types.get(&p[0])
        {
            return self.alias(ty);
        }
        t.clone()
    }
    /// Give literals the declared type and check integer ranges (6.3).
    pub fn fit(&self, v: V, t: &Ty) -> R<V> {
        let t = self.alias(t);
        match (v, &t) {
            (V::Int(x, _), _) if int_name(&t).is_some() => {
                let n = int_name(&t).unwrap();
                let (lo, hi) = int_range(n);
                if x < lo || x > hi {
                    return trap(format!("overflow: {x} is outside {n}"));
                }
                Ok(V::Int(x, n))
            }
            (V::Int(x, _), Ty::Path(p, _)) if p.len() == 1 && p[0] == "f64" => Ok(V::F(x as f64)),
            (V::Int(x, _), Ty::Path(p, _)) if p.len() == 1 && p[0] == "f32" => {
                Ok(V::F32(x as f32 as f64))
            }
            (V::Opt(Some(x)), Ty::Opt(i)) => Ok(V::Opt(Some(Box::new(self.fit(*x, i)?)))),
            (V::List(xs), Ty::Path(p, a)) if p.len() == 1 && p[0] == "List" && a.len() == 1 => {
                Ok(V::List(
                    xs.into_iter()
                        .map(|x| self.fit(x, &a[0]))
                        .collect::<R<_>>()?,
                ))
            }
            (V::List(xs), Ty::Array(e, _)) => Ok(V::List(
                xs.into_iter().map(|x| self.fit(x, e)).collect::<R<_>>()?,
            )),
            (V::Map(xs), Ty::Path(p, a)) if p.len() == 1 && p[0] == "Map" && a.len() == 2 => {
                Ok(V::Map(
                    xs.into_iter()
                        .map(|(k, v)| Ok((self.fit(k, &a[0])?, self.fit(v, &a[1])?)))
                        .collect::<R<_>>()?,
                ))
            }
            (V::Tuple(xs), Ty::Tuple(ts)) if xs.len() == ts.len() => Ok(V::Tuple(
                xs.into_iter()
                    .zip(ts)
                    .map(|(x, t)| self.fit(x, t))
                    .collect::<R<_>>()?,
            )),
            (V::Err(e), _) => Ok(V::Err(e)),
            (v, Ty::Union(ok, _)) => self.fit(v, ok),
            (v, Ty::Ref(_, x)) | (v, Ty::Region(x, _)) => self.fit(v, x),
            (v, _) => Ok(v),
        }
    }
    /// Evaluation steps (fuel ticks) taken so far by this evaluator; the
    /// unit in which the T2 step bound is measured by `rce-c1`.
    pub fn steps_used(&self) -> u64 {
        FUEL - self.fuel
    }
    fn tick(&mut self) -> R<()> {
        if self.fuel == 0 {
            return trap("comptime fuel exhausted");
        }
        self.fuel -= 1;
        Ok(())
    }
    fn get(&self, n: &str) -> Option<V> {
        self.scopes.iter().rev().find_map(|s| s.get(n).cloned())
    }
    fn set(&mut self, n: &str, v: V) -> bool {
        for s in self.scopes.iter_mut().rev() {
            if let Some(x) = s.get_mut(n) {
                *x = v;
                return true;
            }
        }
        false
    }
    fn bind(&mut self, n: &str, v: V) {
        self.scopes.last_mut().unwrap().insert(n.to_string(), v);
    }

    pub fn const_value(&mut self, n: &str) -> R<V> {
        if let Some(v) = self.consts.get(n) {
            return Ok(v.clone());
        }
        for i in &self.m.items {
            if let Item::Const { name, ty, e, .. } = i
                && name == n
            {
                let saved = std::mem::take(&mut self.scopes);
                self.scopes.push(HashMap::new());
                let v = self.eval(e);
                self.scopes = saved;
                let v = v?;
                let v = match ty {
                    Some(t) => self.fit(v, t)?,
                    None => v,
                };
                self.consts.insert(n.to_string(), v.clone());
                return Ok(v);
            }
        }
        trap(format!("unknown name {n}"))
    }
}

fn join_ty(a: &'static str, b: &'static str) -> &'static str {
    if a == "lit" { b } else { a }
}
/// The exact decimal of x * y when it does not fit in i128 (|x|, |y| < 2^64,
/// so the magnitude fits in u128).
pub(crate) fn wide_mul(x: i128, y: i128) -> String {
    let m = x.unsigned_abs().wrapping_mul(y.unsigned_abs());
    if (x < 0) != (y < 0) && m != 0 {
        format!("-{m}")
    } else {
        m.to_string()
    }
}

fn chk(x: i128, t: &'static str) -> R<V> {
    let (lo, hi) = int_range(if t == "lit" { "i64" } else { t });
    if x < lo || x > hi {
        trap(format!(
            "overflow: {x} is outside {}",
            if t == "lit" { "i64" } else { t }
        ))
    } else {
        Ok(V::Int(x, t))
    }
}
fn tdiv(a: i128, b: i128) -> i128 {
    a / b
} // truncation toward zero (6.3)
fn trem(a: i128, b: i128) -> i128 {
    a % b
}

impl<'a> Interp<'a> {
    pub fn arith(&self, op: &str, a: V, b: V) -> R<V> {
        use V::*;
        match (a, b) {
            (Int(x, s), Int(y, t)) => {
                let ty = join_ty(s, t);
                let bits: u32 = match ty {
                    "i8" | "u8" => 8,
                    "i16" | "u16" => 16,
                    "i32" | "u32" => 32,
                    _ => 64,
                };
                match op {
                    "+" => chk(x + y, ty),
                    "-" => chk(x - y, ty),
                    // |x|, |y| < 2^64, so the product can leave i128 (u64 * u64);
                    // the trap then names the exact value, as the Python and
                    // JS runtimes do with their unbounded integers.
                    "*" => match x.checked_mul(y) {
                        Some(p) => chk(p, ty),
                        None => trap(format!(
                            "overflow: {} is outside {}",
                            wide_mul(x, y),
                            if ty == "lit" { "i64" } else { ty }
                        )),
                    },
                    "/" => {
                        if y == 0 {
                            trap("division by zero")
                        } else {
                            chk(tdiv(x, y), ty)
                        }
                    }
                    "%" => {
                        if y == 0 {
                            trap("division by zero")
                        } else {
                            chk(trem(x, y), ty)
                        }
                    }
                    "&" => chk(x & y, ty),
                    "|" => chk(x | y, ty),
                    "^" => chk(x ^ y, ty),
                    "<<" => {
                        if y < 0 || y >= bits as i128 {
                            return trap("shift amount out of range");
                        }
                        let r = (x << y) & ((1i128 << bits) - 1);
                        chk(
                            if ty.starts_with('i') && r >= 1i128 << (bits - 1) {
                                r - (1i128 << bits)
                            } else {
                                r
                            },
                            ty,
                        )
                    }
                    ">>" => {
                        if y < 0 || y >= bits as i128 {
                            return trap("shift amount out of range");
                        }
                        chk(x >> y, ty)
                    }
                    _ => trap(format!("bad operator {op}")),
                }
            }
            (F(x), F(y)) => {
                let r = match op {
                    "+" => x + y,
                    "-" => x - y,
                    "*" => x * y,
                    "/" => x / y,
                    _ => return trap("bad float operator"),
                };
                if r.is_finite() {
                    Ok(F(r))
                } else {
                    trap("non-finite float")
                }
            }
            (F32(x), F32(y)) => {
                // N7: computed in f64 and rounded once to f32; exact because 53 >= 2*24 + 2.
                let r = match op {
                    "+" => x + y,
                    "-" => x - y,
                    "*" => x * y,
                    "/" => x / y,
                    _ => return trap("bad float operator"),
                };
                let r = r as f32;
                if r.is_finite() {
                    Ok(F32(r as f64))
                } else {
                    trap("non-finite float")
                }
            }
            (List(xs), List(ys)) if matches!(op, "+" | "-" | "*" | "/" | "%") => {
                if xs.len() != ys.len() {
                    return trap("lane count mismatch");
                }
                Ok(List(
                    xs.into_iter()
                        .zip(ys)
                        .map(|(x, y)| self.arith(op, x, y))
                        .collect::<R<_>>()?,
                ))
            }
            (F(x), Int(y, _)) => self.arith(op, F(x), F(y as f64)),
            (Int(x, _), F(y)) => self.arith(op, F(x as f64), F(y)),
            (Text(x), Text(y)) if op == "+" => Ok(Text(x + &y)),
            (a, b) => trap(format!(
                "cannot apply {op} to {} and {}",
                a.show(),
                b.show()
            )),
        }
    }
    fn cmp(&self, a: &V, b: &V) -> R<std::cmp::Ordering> {
        use V::*;
        Ok(match (a, b) {
            (Int(x, _), Int(y, _)) => x.cmp(y),
            (F(x), F(y)) | (F32(x), F32(y)) => {
                x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
            }
            (Text(x), Text(y)) => x.cmp(y),
            (Bool(x), Bool(y)) => x.cmp(y),
            (Tuple(x), Tuple(y)) | (List(x), List(y)) => {
                for (p, q) in x.iter().zip(y) {
                    let c = self.cmp(p, q)?;
                    if c.is_ne() {
                        return Ok(c);
                    }
                }
                x.len().cmp(&y.len())
            }
            _ => return trap("values are not ordered"),
        })
    }
    fn truthy(&self, v: V) -> R<bool> {
        match v {
            V::Bool(b) => Ok(b),
            o => trap(format!("expected bool, found {}", o.show())),
        }
    }

    pub fn eval(&mut self, e: &Expr) -> R<V> {
        self.tick()?;
        use Expr as E;
        match e {
            E::Name(n, _) => {
                if let Some(v) = self.get(n) {
                    return Ok(v);
                }
                if self.fns.contains_key(n) {
                    return Ok(V::Fn(n.clone()));
                }
                if let Some(p) = self.variants.get(n)
                    && p.is_empty()
                {
                    return Ok(V::Rec(n.clone(), vec![]));
                }
                if let Some(Item::Event { .. } | Item::Error { .. }) = self.types.get(n) {
                    return Ok(V::Rec(n.clone(), vec![]));
                }
                self.const_value(n)
            }
            E::SelfRef(_) => self.get("self").ok_or(Flow::Trap("self".into())),
            E::Int(v, _) => Ok(V::Int(*v, "lit")),
            E::Float(f, _) => Ok(V::F(*f)),
            E::Str(s, _) => Ok(V::Text(s.clone())),
            E::Bool(b, _) => Ok(V::Bool(*b)),
            E::None(_) => Ok(V::Opt(None)),
            E::Unit(_) => Ok(V::Unit),
            E::Tuple(v, _) => Ok(V::Tuple(v.iter().map(|x| self.eval(x)).collect::<R<_>>()?)),
            E::List(v, _) => Ok(V::List(v.iter().map(|x| self.eval(x)).collect::<R<_>>()?)),
            E::ListRepeat(x, n, _) => {
                let v = self.eval(x)?;
                match self.eval(n)? {
                    V::Int(k, _) if (0..=10_000_000).contains(&k) => {
                        Ok(V::List(vec![v; k as usize]))
                    }
                    _ => trap("bad repeat count"),
                }
            }
            E::ListComp(body, p, it, cond, _) => {
                let itv = self.eval(it)?;
                let items = self.iter_items(itv)?;
                let mut out = vec![];
                for x in items {
                    self.tick()?;
                    self.scopes.push(HashMap::new());
                    let ok = self.pmatch(p, &x)?;
                    let keep = ok
                        && match cond {
                            Some(c) => {
                                let c = self.eval(c);
                                match c {
                                    Ok(c) => self.truthy(c)?,
                                    Err(f) => {
                                        self.scopes.pop();
                                        return Err(f);
                                    }
                                }
                            }
                            None => true,
                        };
                    let r = if keep { Some(self.eval(body)) } else { None };
                    self.scopes.pop();
                    if let Some(r) = r {
                        out.push(r?);
                    }
                }
                Ok(V::List(out))
            }
            E::Map(v, _) => {
                let mut m: Vec<(V, V)> = vec![];
                for (a, b) in v {
                    let k = self.eval(a)?;
                    let x = self.eval(b)?;
                    map_set(&mut m, k, x);
                }
                Ok(V::Map(m))
            }
            E::ChanNew(..) => trap("channels are not available at comptime"),
            E::Call(f, args, _) => self.call(f, args),
            E::Index(x, idx, _) => {
                let c = self.eval(x)?;
                let i = self.eval(&idx[0])?;
                index(c, i)
            }
            E::Slice(x, a, b, _) => {
                let c = self.eval(x)?;
                let (xs, text) = match c {
                    V::List(xs) => (xs, None),
                    V::Text(t) => (
                        t.bytes().map(|b| V::Int(b as i128, "u8")).collect(),
                        Some(t),
                    ),
                    o => return trap(format!("cannot slice {}", o.show())),
                };
                let n = xs.len() as i128;
                let norm = |v: Option<V>, d: i128| -> R<i128> {
                    match v {
                        None => Ok(d),
                        Some(V::Int(k, _)) => {
                            let k = if k < 0 { n + k } else { k };
                            if k < 0 || k > n {
                                trap("slice index out of range")
                            } else {
                                Ok(k)
                            }
                        }
                        _ => trap("bad slice bound"),
                    }
                };
                let av = match a {
                    Some(a) => Some(self.eval(a)?),
                    None => None,
                };
                let bv = match b {
                    Some(b) => Some(self.eval(b)?),
                    None => None,
                };
                let (lo, hi) = (norm(av, 0)?, norm(bv, n)?);
                if lo > hi {
                    return trap("slice bounds reversed");
                }
                if let Some(t) = text {
                    let (lo, hi) = (lo as usize, hi as usize);
                    if !t.is_char_boundary(lo) || !t.is_char_boundary(hi) {
                        return trap("not a char boundary");
                    }
                    return Ok(V::Text(t[lo..hi].to_string()));
                }
                Ok(V::List(xs[lo as usize..hi as usize].to_vec()))
            }
            E::Member(x, m, _) => {
                let v = self.eval(x)?;
                field(&v, m)
            }
            E::TupleMember(x, i, _) => match self.eval(x)? {
                V::Tuple(v) if *i < v.len() => Ok(v[*i].clone()),
                _ => trap("bad tuple index"),
            },
            E::Neg(x, _) => match self.eval(x)? {
                V::Int(v, t) => chk(-v, t),
                V::F(f) => Ok(V::F(-f)),
                V::F32(f) => Ok(V::F32(-f)),
                _ => trap("bad negation"),
            },
            E::BitNot(x, _) => match self.eval(x)? {
                V::Int(v, t) => {
                    let t2 = if t == "lit" { "i64" } else { t };
                    let (lo, hi) = int_range(t2);
                    Ok(V::Int(if lo == 0 { hi - v } else { !v }, t))
                }
                _ => trap("bad ~"),
            },
            E::Not(x, _) => {
                let b = self.eval(x)?;
                Ok(V::Bool(!self.truthy(b)?))
            }
            E::Borrow(_, x, _) => self.eval(x),
            E::Recv(..) => trap("channels are not available at comptime"),
            E::Comptime(x, _) => self.eval(x),
            E::Cast(x, ty, _) => {
                let v = self.eval(x)?;
                let ty = self.alias(ty);
                let to_f32 =
                    matches!(&ty, Ty::Path(p, a) if a.is_empty() && p.len() == 1 && p[0] == "f32");
                match (v, int_name(&ty)) {
                    (V::Int(v, _), None) if to_f32 => Ok(V::F32(v as f32 as f64)),
                    (V::F(f), None) if to_f32 => {
                        let r = f as f32;
                        if r.is_finite() {
                            Ok(V::F32(r as f64))
                        } else {
                            trap("non-finite float")
                        }
                    }
                    (V::F32(f), None) if !to_f32 => Ok(V::F(f)),
                    (V::F32(f), Some(n)) => chk(f.trunc() as i128, n),
                    (V::Int(v, _), Some(n)) => chk(v, n),
                    (V::F(f), Some(n)) => {
                        if !f.is_finite() {
                            return trap("non-finite float");
                        }
                        chk(f.trunc() as i128, n)
                    }
                    (V::Bool(b), Some(n)) => chk(b as i128, n),
                    (V::Int(v, _), None) => Ok(V::F(v as f64)),
                    (o, _) => Ok(o),
                }
            }
            E::Bin(op, a, b, _) => self.binary(op, a, b),
            E::Catch(x, bind, h, _) => match self.eval(x)? {
                V::Err(err) => {
                    self.scopes.push(HashMap::new());
                    if let Some(n) = bind {
                        self.bind(n, *err);
                    }
                    let r = self.eval(h);
                    self.scopes.pop();
                    r
                }
                v => Ok(v),
            },
            E::Try(x, els, _) => match self.eval(x)? {
                V::Err(err) => Err(Flow::Fail(match els {
                    Some(e) => self.eval(e)?,
                    None => *err,
                })),
                V::Opt(None) => match els {
                    Some(e) => {
                        let ev = self.eval(e)?;
                        Err(Flow::Fail(ev))
                    }
                    None => trap("try on none"),
                },
                V::Opt(Some(v)) => Ok(*v),
                v => Ok(v),
            },
            E::If(c, a, b, _) => {
                let c = self.eval(c)?;
                if self.truthy(c)? {
                    self.eval(a)
                } else {
                    self.eval(b)
                }
            }
            E::Lambda(ps, body, _) => Ok(V::Closure(ps.clone(), body.clone(), self.scopes.clone())),
        }
    }

    fn binary(&mut self, op: &str, a: &Expr, b: &Expr) -> R<V> {
        match op {
            "and" => {
                let x = self.eval(a)?;
                if !self.truthy(x)? {
                    return Ok(V::Bool(false));
                }
                let y = self.eval(b)?;
                return Ok(V::Bool(self.truthy(y)?));
            }
            "or" => {
                let x = self.eval(a)?;
                if self.truthy(x)? {
                    return Ok(V::Bool(true));
                }
                let y = self.eval(b)?;
                return Ok(V::Bool(self.truthy(y)?));
            }
            _ => {}
        }
        let x = self.eval(a)?;
        let y = self.eval(b)?;
        use std::cmp::Ordering::*;
        match op {
            "==" => Ok(V::Bool(x == y)),
            "!=" => Ok(V::Bool(x != y)),
            "<" => Ok(V::Bool(self.cmp(&x, &y)? == Less)),
            "<=" => Ok(V::Bool(self.cmp(&x, &y)? != Greater)),
            ">" => Ok(V::Bool(self.cmp(&x, &y)? == Greater)),
            ">=" => Ok(V::Bool(self.cmp(&x, &y)? != Less)),
            "in" | "not in" => {
                let r = contains(&y, &x)?;
                Ok(V::Bool(if op == "in" { r } else { !r }))
            }
            ".." | "..=" => match (x, y) {
                (V::Int(p, s), V::Int(q, t)) => Ok(V::Range(
                    p,
                    if op == ".." { q } else { q + 1 },
                    join_ty(s, t),
                )),
                _ => trap("bad range"),
            },
            _ => self.arith(op, x, y),
        }
    }
}

fn map_set(m: &mut Vec<(V, V)>, k: V, v: V) {
    if let Some(e) = m.iter_mut().find(|(a, _)| *a == k) {
        e.1 = v;
    } else {
        m.push((k, v));
    }
}
fn contains(c: &V, x: &V) -> R<bool> {
    Ok(match c {
        V::List(v) | V::Tuple(v) => v.contains(x),
        V::Map(m) => m.iter().any(|(k, _)| k == x),
        V::Text(t) => match x {
            V::Text(s) => t.contains(s.as_str()),
            _ => false,
        },
        V::Range(a, b, _) => match x {
            V::Int(v, _) => a <= v && v < b,
            _ => false,
        },
        _ => return trap("`in` needs a collection"),
    })
}
fn index(c: V, i: V) -> R<V> {
    match (c, i) {
        (V::List(v), V::Int(k, _)) => {
            if k >= 0 && (k as usize) < v.len() {
                Ok(v[k as usize].clone())
            } else {
                trap(format!("index {k} out of range"))
            }
        }
        (V::Text(t), V::Int(k, _)) => {
            if k >= 0 && (k as usize) < t.len() {
                Ok(V::Int(t.as_bytes()[k as usize] as i128, "u8"))
            } else {
                trap(format!("index {k} out of range"))
            }
        }
        (V::Map(m), k) => m
            .iter()
            .find(|(a, _)| *a == k)
            .map(|(_, v)| v.clone())
            .ok_or(Flow::Trap(format!("missing key {}", k.show()))),
        (c, _) => trap(format!("cannot index {}", c.show())),
    }
}
fn field(v: &V, m: &str) -> R<V> {
    match v {
        V::Rec(_, fs) => fs
            .iter()
            .find(|(n, _)| n == m)
            .map(|(_, x)| x.clone())
            .ok_or(Flow::Trap(format!("no field {m}"))),
        _ => trap(format!("{} has no field {m}", v.show())),
    }
}

impl<'a> Interp<'a> {
    /// Run iterator `g` until it finishes, fails, or has yielded `cap` values.
    /// Returns (values, finished, pending error). A pending error is raised only
    /// when the consumer asks for the value after the last one delivered, so the
    /// observable behaviour is that of the push model (6.8).
    fn gen_run(&mut self, g: usize, cap: usize) -> R<(Vec<V>, bool, Option<Flow>)> {
        let (fd, frame) = self.gens[g].clone();
        let body = match &fd.body {
            Body::Block(b) => b,
            Body::Arrow(_) => return trap("iterator without a block body"),
        };
        let saved = std::mem::replace(&mut self.scopes, vec![frame]);
        self.yields.push((vec![], cap));
        let r = self.exec_block(body);
        let (items, _) = self.yields.pop().unwrap();
        self.scopes = saved;
        Ok(match r {
            Ok(()) | Err(Flow::Ret(_)) => (items, true, None),
            Err(Flow::Stop) => (items, false, None),
            Err(f) => (items, true, Some(f)),
        })
    }
    fn iter_items(&mut self, v: V) -> R<Vec<V>> {
        Ok(match v {
            V::List(xs) => xs,
            V::Range(a, b, t) => {
                if b - a > self.loop_limit * 10 {
                    return trap("loop bound");
                }
                (a..b.max(a)).map(|i| V::Int(i, t)).collect()
            }
            V::Text(s) => s.chars().map(|c| V::Text(c.to_string())).collect(),
            V::Map(m) => m.into_iter().map(|(k, v)| V::Tuple(vec![k, v])).collect(),
            V::Gen(g) => {
                let (items, _, err) = self.gen_run(g, usize::MAX)?;
                if let Some(f) = err {
                    return Err(f);
                }
                items
            }
            o => return trap(format!("cannot iterate over {}", o.show())),
        })
    }

    fn assign(&mut self, place: &Expr, v: V) -> R<()> {
        match place {
            Expr::Name(n, _) => {
                if self.set(n, v) {
                    Ok(())
                } else {
                    trap(format!("assignment to unknown {n}"))
                }
            }
            Expr::SelfRef(_) => {
                self.set("self", v);
                Ok(())
            }
            Expr::Borrow(_, x, _) => self.assign(x, v),
            Expr::Member(x, m, _) => match self.eval(x)? {
                V::Rec(n, mut fs) => {
                    match fs.iter_mut().find(|(f, _)| f == m) {
                        Some(f) => f.1 = v,
                        None => return trap("no field"),
                    }
                    self.assign(x, V::Rec(n, fs))
                }
                _ => trap("member assignment on a non-record"),
            },
            Expr::TupleMember(x, i, _) => match self.eval(x)? {
                V::Tuple(mut t) if *i < t.len() => {
                    t[*i] = v;
                    self.assign(x, V::Tuple(t))
                }
                _ => trap("bad tuple assignment"),
            },
            Expr::Index(x, idx, _) => {
                let k = self.eval(&idx[0])?;
                match (self.eval(x)?, k) {
                    (V::List(mut xs), V::Int(i, _)) => {
                        if i < 0 || i as usize >= xs.len() {
                            return trap(format!("index {i} out of range"));
                        }
                        xs[i as usize] = v;
                        self.assign(x, V::List(xs))
                    }
                    (V::Map(mut m), k) => {
                        map_set(&mut m, k, v);
                        self.assign(x, V::Map(m))
                    }
                    _ => trap("bad indexed assignment"),
                }
            }
            _ => trap("not a place"),
        }
    }

    fn pmatch(&mut self, p: &Pat, v: &V) -> R<bool> {
        Ok(match p {
            Pat::Wild => true,
            Pat::Path(path) => {
                let n = &path[path.len() - 1];
                if path.len() == 1
                    && !self.variants.contains_key(n)
                    && !matches!(
                        self.types.get(n),
                        Some(Item::Event { .. } | Item::Error { .. })
                    )
                {
                    self.bind(n, v.clone());
                    true
                } else {
                    v.rec_name() == Some(n.as_str())
                }
            }
            Pat::Ctor(path, args) => {
                let n = &path[path.len() - 1];
                if n == "some" && path.len() == 1 {
                    return match v {
                        V::Opt(Some(x)) => {
                            let x = (**x).clone();
                            self.pmatch(&args[0].1, &x)
                        }
                        _ => Ok(false),
                    };
                }
                match v {
                    V::Rec(m, fs) if m == n => {
                        for (i, (lb, q)) in args.iter().enumerate() {
                            let fv = match lb {
                                Some(l) => fs.iter().find(|(f, _)| f == l).map(|x| x.1.clone()),
                                None => fs.get(i).map(|x| x.1.clone()),
                            };
                            match fv {
                                Some(fv) => {
                                    if !self.pmatch(q, &fv)? {
                                        return Ok(false);
                                    }
                                }
                                None => return Ok(false),
                            }
                        }
                        true
                    }
                    _ => false,
                }
            }
            Pat::Lit(e) => {
                let x = self.eval(e)?;
                x == *v
            }
            Pat::Range(a, incl, b) => {
                let (x, y) = (self.eval(a)?, self.eval(b)?);
                let lo = self.cmp(v, &x)?.is_ge();
                let hi = if *incl {
                    self.cmp(v, &y)?.is_le()
                } else {
                    self.cmp(v, &y)?.is_lt()
                };
                lo && hi
            }
            Pat::Tuple(ps) => match v {
                V::Tuple(vs) if vs.len() == ps.len() => {
                    for (q, x) in ps.iter().zip(vs) {
                        if !self.pmatch(q, x)? {
                            return Ok(false);
                        }
                    }
                    true
                }
                _ => false,
            },
            Pat::Alt(ps) => {
                for q in ps {
                    if self.pmatch(q, v)? {
                        return Ok(true);
                    }
                }
                false
            }
        })
    }

    pub fn construct(&mut self, n: &str, args: &[Arg]) -> R<Option<V>> {
        let fields: Vec<(String, Ty, Option<Expr>)> = if let Some(p) = self.variants.get(n) {
            p.iter()
                .map(|(a, b)| (a.clone(), b.clone(), None))
                .collect()
        } else {
            match self.types.get(n) {
                Some(Item::Struct { fields, .. }) => fields
                    .iter()
                    .map(|f| (f.name.clone(), f.ty.clone(), f.init.clone()))
                    .collect(),
                Some(Item::Event { payload, .. } | Item::Error { payload, .. }) => payload
                    .iter()
                    .map(|(a, b)| (a.clone(), b.clone(), None))
                    .collect(),
                _ => return Ok(None),
            }
        };
        let mut vals: Vec<Option<V>> = vec![None; fields.len()];
        for (i, a) in args.iter().enumerate() {
            let k = match &a.label {
                Some(l) => fields.iter().position(|f| &f.0 == l).unwrap_or(i),
                None => i,
            };
            let v = self.eval(&a.e)?;
            if k < fields.len() {
                vals[k] = Some(self.fit(v, &fields[k].1)?);
            }
        }
        let mut out = vec![];
        for (k, (f, t, init)) in fields.into_iter().enumerate() {
            let v = match vals[k].take() {
                Some(v) => v,
                None => match init {
                    Some(e) => {
                        let v = self.eval(&e)?;
                        self.fit(v, &t)?
                    }
                    None => return trap(format!("missing field {f}")),
                },
            };
            out.push((f, v));
        }
        Ok(Some(V::Rec(n.to_string(), out)))
    }

    fn call(&mut self, f: &Expr, args: &[Arg]) -> R<V> {
        match f {
            Expr::Name(n, _) if self.get(n).is_none() => {
                match n.as_str() {
                    "some" => {
                        let v = self.eval(&args[0].e)?;
                        return Ok(V::Opt(Some(Box::new(v))));
                    }
                    "copy" | "Box" => return self.eval(&args[0].e),
                    _ => {}
                }
                if let Some(fd) = self.fns.get(n).copied() {
                    return self.call_fn(fd, args, None);
                }
                if let Some(v) = self.construct(n, args)? {
                    return Ok(v);
                }
                trap(format!("cannot call {n} at comptime"))
            }
            Expr::Member(obj, m, _) => {
                if let Expr::Name(tn, _) = &**obj
                    && self.get(tn).is_none()
                    && self.types.contains_key(tn)
                {
                    return match self.methods.get(&(tn.clone(), m.clone())).copied() {
                        Some(fd) => self.call_fn(fd, args, None),
                        None => trap("no such function"),
                    };
                }
                let recv = self.eval(obj)?;
                let head = match &recv {
                    V::Rec(n, _) => Some(self.kind_of(n)),
                    V::List(_) => Some("List".to_string()),
                    _ => None,
                };
                if let Some(h) = head
                    && let Some(fd) = self.methods.get(&(h, m.clone())).copied()
                {
                    return self.call_fn(fd, args, Some((recv, obj)));
                }
                self.builtin(obj, recv, m, args)
            }
            _ => {
                let fv = self.eval(f)?;
                self.call_value(fv, args)
            }
        }
    }
    fn kind_of(&self, n: &str) -> String {
        for i in &self.m.items {
            if let Item::Kind { name, variants, .. } = i
                && variants.iter().any(|(v, _)| v == n)
            {
                return name.clone();
            }
        }
        n.to_string()
    }
    fn call_value(&mut self, fv: V, args: &[Arg]) -> R<V> {
        match fv {
            V::Fn(n) => {
                let fd = self.fns[&n];
                self.call_fn(fd, args, None)
            }
            V::Closure(ps, body, env) => {
                let vals: Vec<V> = args.iter().map(|a| self.eval(&a.e)).collect::<R<_>>()?;
                self.apply_closure(&ps, &body, env, vals)
            }
            o => trap(format!("{} is not callable", o.show())),
        }
    }
    fn apply_closure(
        &mut self,
        ps: &[Param],
        body: &Expr,
        env: Vec<HashMap<String, V>>,
        vals: Vec<V>,
    ) -> R<V> {
        let saved = std::mem::replace(&mut self.scopes, env);
        self.scopes.push(HashMap::new());
        let mut r = Ok(V::Unit);
        for (p, v) in ps.iter().zip(vals) {
            if let Param::Named { name, ty, .. } = p {
                match self.fit(v, ty) {
                    Ok(v) => self.bind(name, v),
                    Err(e) => {
                        r = Err(e);
                    }
                }
            }
        }
        if r.is_ok() {
            r = self.eval(body);
        }
        self.scopes = saved;
        r
    }
    pub fn call_fn(&mut self, fd: &'a FnDecl, args: &[Arg], recv: Option<(V, &Expr)>) -> R<V> {
        self.tick()?;
        let named: Vec<(&String, &Ty)> = fd
            .sig
            .params
            .iter()
            .filter_map(|p| {
                if let Param::Named { name, ty, .. } = p {
                    Some((name, ty))
                } else {
                    None
                }
            })
            .collect();
        let mut frame = HashMap::new();
        for ((n, t), a) in named.iter().zip(args) {
            let v = self.eval(&a.e)?;
            frame.insert((*n).clone(), self.fit(v, t)?);
        }
        if let Some(Ty::Iter(_)) = &fd.sig.ret {
            self.gens.push((fd, frame));
            return Ok(V::Gen(self.gens.len() - 1));
        }
        let self_mut = matches!(fd.sig.params.first(), Some(Param::SelfRef(true)));
        if let Some((rv, _)) = &recv {
            frame.insert("self".into(), rv.clone());
        }
        let saved = std::mem::replace(&mut self.scopes, vec![frame]);
        let r = match &fd.body {
            Body::Arrow(e) => self.eval(e),
            Body::Block(b) => match self.exec_block_tail(b) {
                Ok(Some(v)) => Ok(v),
                Ok(None) => Ok(V::Unit),
                Err(Flow::Ret(v)) => Ok(v),
                Err(f) => Err(f),
            },
        };
        let fin = std::mem::replace(&mut self.scopes, saved);
        // copy-out for `&mut` arguments and a `&mut self` receiver
        for ((n, t), a) in named.iter().zip(args) {
            if matches!(t, Ty::Ref(true, _))
                && let Some(v) = fin[0].get(*n)
            {
                self.assign(&a.e, v.clone())?;
            }
        }
        if self_mut && let (Some((_, obj)), Some(v)) = (recv, fin[0].get("self")) {
            self.assign(obj, v.clone())?;
        }
        match r {
            Ok(v) => match &fd.sig.ret {
                Some(t) => self.fit(v, t),
                None => Ok(V::Unit),
            },
            Err(Flow::Fail(e)) => match &fd.sig.ret {
                Some(Ty::Union(..)) => Ok(V::Err(Box::new(e))),
                _ => Err(Flow::Fail(e)),
            },
            Err(f) => Err(f),
        }
    }
}

impl<'a> Interp<'a> {
    fn builtin(&mut self, obj: &Expr, recv: V, m: &str, args: &[Arg]) -> R<V> {
        let mut a: Vec<V> = vec![];
        let lazy_fn = matches!(m, "map" | "filter" | "fold");
        for x in args {
            a.push(self.eval(&x.e)?);
        }
        let _ = lazy_fn;
        use V::*;
        match (recv, m) {
            (Int(x, t), "checked_add" | "checked_sub" | "checked_mul" | "checked_div") => {
                let op = match m {
                    "checked_add" => "+",
                    "checked_sub" => "-",
                    "checked_mul" => "*",
                    _ => "/",
                };
                Ok(match self.arith(op, Int(x, t), a[0].clone()) {
                    Ok(v) => Opt(Some(Box::new(v))),
                    Result::Err(Flow::Trap(_)) => Opt(None),
                    Result::Err(f) => return Result::Err(f),
                })
            }
            (
                Int(x, t),
                "wrapping_add" | "wrapping_sub" | "wrapping_mul" | "saturating_add"
                | "saturating_sub" | "saturating_mul",
            ) => {
                let y = match &a[0] {
                    Int(y, _) => *y,
                    _ => return trap(m),
                };
                // u64 * u64 can leave i128: wrapping keeps the residue mod
                // 2^128 (enough for widths up to 64), saturating keeps the sign.
                let r = match &m[m.len() - 3..] {
                    "add" => x + y,
                    "sub" => x - y,
                    _ if m.starts_with("wrapping") => x.wrapping_mul(y),
                    _ => x.checked_mul(y).unwrap_or(if (x < 0) != (y < 0) {
                        i128::MIN
                    } else {
                        i128::MAX
                    }),
                };
                let (lo, hi) = crate::types::int_range(t);
                Ok(Int(
                    if m.starts_with("wrapping") {
                        let w = hi - lo + 1;
                        r.wrapping_sub(lo).rem_euclid(w) + lo
                    } else {
                        r.clamp(lo, hi)
                    },
                    t,
                ))
            }
            (Int(x, _), "tostr") => Ok(Text(x.to_string())),
            (Int(x, t), "abs") => chk(x.abs(), t),
            (Int(x, t), "min") => match &a[0] {
                Int(y, _) => Ok(Int(x.min(*y), t)),
                _ => trap("min"),
            },
            (Int(x, t), "max") => match &a[0] {
                Int(y, _) => Ok(Int(x.max(*y), t)),
                _ => trap("max"),
            },
            (F(x), "sqrt") => {
                if x < 0.0 {
                    trap("non-finite float")
                } else {
                    Ok(F(x.sqrt()))
                }
            }
            (F(x), "abs") => Ok(F(x.abs())),
            (F(x), "floor") => Ok(F(x.floor())),
            (F(x), "ceil") => Ok(F(x.ceil())),
            (F(x), "round") => Ok(F(x.round())),
            (F(x), "tostr") => Ok(Text(x.to_string())),
            (Bool(b), "tostr") => Ok(Text(b.to_string())),
            (Text(s), "len") => Ok(Int(s.len() as i128, "usize")),
            (F32(x), "abs") => Ok(F32(x.abs())),
            (F32(x), "tostr") => Ok(Text((x as f32).to_string())),
            (Gen(g), "collect") => {
                let (items, _, err) = self.gen_run(g, usize::MAX)?;
                if let Some(f) = err {
                    return Result::Err(f);
                }
                Ok(List(items))
            }
            (Text(s), "strip") => Ok(Text(s.trim().into())),
            (Text(s), "upper") => Ok(Text(s.to_ascii_uppercase())),
            (Text(s), "lower") => Ok(Text(s.to_ascii_lowercase())),
            (Text(s), "reverse") => Ok(Text(s.chars().rev().collect())),
            (Text(s), "tostr") => Ok(Text(s)),
            (Text(s), "chars") => Ok(List(s.chars().map(|c| Text(c.to_string())).collect())),
            (Text(s), "words") => Ok(List(s.split_whitespace().map(|w| Text(w.into())).collect())),
            (Text(s), "lines") => Ok(List(s.lines().map(|w| Text(w.into())).collect())),
            (Text(s), "split") => match &a[0] {
                Text(d) => Ok(List(s.split(d.as_str()).map(|w| Text(w.into())).collect())),
                _ => trap("split"),
            },
            (Text(s), "isdigit") => {
                Ok(Bool(!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())))
            }
            (Text(s), "isalpha") => Ok(Bool(
                !s.is_empty() && s.chars().all(|c| c.is_ascii_alphabetic()),
            )),
            (Text(s), "isspace") => Ok(Bool(
                !s.is_empty() && s.chars().all(|c| c.is_ascii_whitespace()),
            )),
            (Text(s), "isempty") => Ok(Bool(s.is_empty())),
            (Text(s), "contains") => match &a[0] {
                Text(d) => Ok(Bool(s.contains(d.as_str()))),
                _ => trap("contains"),
            },
            (Text(s), "startswith") => match &a[0] {
                Text(d) => Ok(Bool(s.starts_with(d.as_str()))),
                _ => trap("startswith"),
            },
            (Text(s), "endswith") => match &a[0] {
                Text(d) => Ok(Bool(s.ends_with(d.as_str()))),
                _ => trap("endswith"),
            },
            (Text(s), "toint") => s
                .trim()
                .parse::<i128>()
                .map(|v| Int(v, "i64"))
                .map_err(|_| Flow::Trap(format!("not an integer: {s:?}"))),
            (Text(s), "join") => match &a[0] {
                List(xs) => Ok(Text(
                    xs.iter()
                        .map(|x| if let Text(t) = x { t.clone() } else { x.show() })
                        .collect::<Vec<_>>()
                        .join(&s),
                )),
                _ => trap("join"),
            },
            (List(xs), "len") => Ok(Int(xs.len() as i128, "usize")),
            (List(xs), "isempty") => Ok(Bool(xs.is_empty())),
            (List(mut xs), "append" | "push") => {
                xs.push(a[0].clone());
                self.assign(obj, List(xs))?;
                Ok(Unit)
            }
            (List(mut xs), "pop") => {
                let r = xs.pop();
                self.assign(obj, List(xs))?;
                Ok(Opt(r.map(Box::new)))
            }
            (List(_), "clear") => {
                self.assign(obj, List(vec![]))?;
                Ok(Unit)
            }
            (List(mut xs), "sort") => {
                let mut err = None;
                xs.sort_by(|p, q| {
                    self.cmp(p, q).unwrap_or_else(|e| {
                        err = Some(e);
                        std::cmp::Ordering::Equal
                    })
                });
                if let Some(e) = err {
                    return Result::Err(e);
                }
                self.assign(obj, List(xs))?;
                Ok(Unit)
            }
            (List(mut xs), "with") => {
                xs.push(a[0].clone());
                Ok(List(xs))
            }
            (List(mut xs), "reverse") => {
                xs.reverse();
                Ok(List(xs))
            }
            (List(xs), "contains") => Ok(Bool(xs.contains(&a[0]))),
            (List(xs), "get") => match &a[0] {
                Int(i, _) => Ok(Opt(if *i >= 0 {
                    xs.get(*i as usize).cloned().map(Box::new)
                } else {
                    None
                })),
                _ => trap("get"),
            },
            (List(xs), "sum") => {
                let mut acc = Int(0, "lit");
                for x in xs {
                    acc = self.arith("+", acc, x)?;
                }
                Ok(acc)
            }
            (List(xs), "map") => {
                let f = a[0].clone();
                let mut out = vec![];
                for x in xs {
                    self.tick()?;
                    out.push(self.apply(&f, vec![x])?);
                }
                Ok(List(out))
            }
            (List(xs), "filter") => {
                let f = a[0].clone();
                let mut out = vec![];
                for x in xs {
                    self.tick()?;
                    if self.apply(&f, vec![x.clone()])? == Bool(true) {
                        out.push(x);
                    }
                }
                Ok(List(out))
            }
            (List(xs), "fold") => {
                let f = a[1].clone();
                let mut acc = a[0].clone();
                for x in xs {
                    self.tick()?;
                    acc = self.apply(&f, vec![acc, x])?;
                }
                Ok(acc)
            }
            (Map(mm), "len") => Ok(Int(mm.len() as i128, "usize")),
            (Map(mm), "get") => Ok(match mm.iter().find(|(k, _)| *k == a[0]) {
                Some((_, v)) => v.clone(),
                None => Err(Box::new(Rec("Missing".into(), vec![]))),
            }),
            (Map(mm), "contains") => Ok(Bool(mm.iter().any(|(k, _)| *k == a[0]))),
            (Map(mut mm), "insert") => {
                map_set(&mut mm, a[0].clone(), a[1].clone());
                self.assign(obj, Map(mm))?;
                Ok(Unit)
            }
            (Map(mut mm), "remove") => {
                let i = mm.iter().position(|(k, _)| *k == a[0]);
                let r = i.map(|i| mm.remove(i).1);
                self.assign(obj, Map(mm))?;
                Ok(Opt(r.map(Box::new)))
            }
            (Map(mm), "keys") => Ok(List(mm.into_iter().map(|x| x.0).collect())),
            (Map(mm), "values") => Ok(List(mm.into_iter().map(|x| x.1).collect())),
            (Opt(x), "isnone") => Ok(Bool(x.is_none())),
            (Opt(x), "issome") => Ok(Bool(x.is_some())),
            (Opt(x), "or") => Ok(match x {
                Some(v) => *v,
                None => a[0].clone(),
            }),
            (r, _) => trap(format!("{} has no method {m} at comptime", r.show())),
        }
    }
    fn apply(&mut self, f: &V, vals: Vec<V>) -> R<V> {
        match f.clone() {
            V::Closure(ps, body, env) => self.apply_closure(&ps, &body, env, vals),
            V::Fn(n) => {
                let fd = self.fns[&n];
                let mut frame = HashMap::new();
                for (p, v) in fd.sig.params.iter().zip(vals) {
                    if let Param::Named { name, ty, .. } = p {
                        frame.insert(name.clone(), self.fit(v, ty)?);
                    }
                }
                let saved = std::mem::replace(&mut self.scopes, vec![frame]);
                let r = match &fd.body {
                    Body::Arrow(e) => self.eval(e),
                    Body::Block(b) => match self.exec_block_tail(b) {
                        Ok(v) => Ok(v.unwrap_or(V::Unit)),
                        Err(Flow::Ret(v)) => Ok(v),
                        Err(e) => Err(e),
                    },
                };
                self.scopes = saved;
                r
            }
            o => trap(format!("{} is not callable", o.show())),
        }
    }

    /// Runs a block; returns the tail value if the block ends in one (5.3 [Tail]).
    pub fn exec_block_tail(&mut self, b: &[Stmt]) -> R<Option<V>> {
        self.scopes.push(HashMap::new());
        let mut r = Ok(None);
        for (i, s) in b.iter().enumerate() {
            let last = i + 1 == b.len();
            r = self.exec(s, last);
            if r.is_err() {
                break;
            }
        }
        self.scopes.pop();
        r
    }
    fn exec_block(&mut self, b: &[Stmt]) -> R<()> {
        self.exec_block_tail(b).map(|_| ())
    }
    fn run_loop(
        &mut self,
        bound: i128,
        mut cond: impl FnMut(&mut Self) -> R<bool>,
        body: &[Stmt],
    ) -> R<()> {
        let mut n = 0;
        while cond(self)? {
            n += 1;
            if n > bound {
                return trap("loop bound");
            }
            match self.exec_block(body) {
                Ok(()) | Err(Flow::Cont) => {}
                Err(Flow::Brk) => break,
                Err(f) => return Err(f),
            }
        }
        Ok(())
    }

    fn exec(&mut self, s: &Stmt, tail: bool) -> R<Option<V>> {
        self.tick()?;
        match s {
            Stmt::Expr(e) => {
                let v = self.eval(e)?;
                return Ok(if tail { Some(v) } else { None });
            }
            Stmt::Assign(op, place, rhs) => {
                let r = self.eval(rhs)?;
                let v = if op == "=" {
                    r
                } else {
                    let cur = self.eval(place)?;
                    let t = if let V::Int(_, t) = cur { t } else { "lit" };
                    let x = self.arith(&op[..op.len() - 1], cur, r)?;
                    if let V::Int(x, _) = x { chk(x, t)? } else { x }
                };
                let v = match (self.eval(place).ok(), v) {
                    (Some(V::Int(_, t)), V::Int(x, _)) if t != "lit" => chk(x, t)?,
                    (_, v) => v,
                };
                self.assign(place, v)?;
            }
            Stmt::Let(p, ty, e, _) => {
                let v = self.eval(e)?;
                let v = match ty {
                    Some(t) => self.fit(v, t)?,
                    None => v,
                };
                if !self.pmatch(p, &v)? {
                    return trap("let pattern did not match");
                }
            }
            Stmt::Var(n, ty, e, _) => {
                let v = self.eval(e)?;
                let v = match ty {
                    Some(t) => self.fit(v, t)?,
                    None => v,
                };
                self.bind(n, v);
            }
            Stmt::If(arms, els) => {
                for (c, b) in arms {
                    let c = self.eval(c)?;
                    if self.truthy(c)? {
                        return self.exec_block_tail(b).map(|v| if tail { v } else { None });
                    }
                }
                if let Some(b) = els {
                    return self.exec_block_tail(b).map(|v| if tail { v } else { None });
                }
            }
            Stmt::While(c, b, body) => {
                let bound = match b {
                    Some(b) => match self.eval(b)? {
                        V::Int(n, _) => n,
                        _ => self.loop_limit,
                    },
                    None => self.loop_limit,
                };
                let c = c.clone();
                self.run_loop(
                    bound,
                    move |me| {
                        let v = me.eval(&c)?;
                        me.truthy(v)
                    },
                    body,
                )?;
            }
            Stmt::Loop(b, body) => {
                let bound = match b {
                    Some(b) => match self.eval(b)? {
                        V::Int(n, _) => n,
                        _ => self.loop_limit,
                    },
                    None => self.loop_limit,
                };
                self.run_loop(bound, |_| Ok(true), body)?;
            }
            Stmt::For(p, it, body, _) if matches!(self.eval(it), Ok(V::Gen(_))) => {
                let g = match self.eval(it)? {
                    V::Gen(g) => g,
                    _ => unreachable!(),
                };
                let (mut done, mut cap) = (0usize, 64usize);
                'outer: loop {
                    let (items, finished, err) = self.gen_run(g, cap)?;
                    for x in items.into_iter().skip(done) {
                        done += 1;
                        self.scopes.push(HashMap::new());
                        let r = match self.pmatch(p, &x) {
                            Ok(_) => self.exec_block(body),
                            Err(e) => Err(e),
                        };
                        self.scopes.pop();
                        match r {
                            Ok(()) | Err(Flow::Cont) => {}
                            Err(Flow::Brk) => break 'outer,
                            Err(f) => return Err(f),
                        }
                    }
                    if let Some(f) = err {
                        return Err(f);
                    }
                    if finished {
                        break;
                    }
                    cap = cap.saturating_mul(2);
                }
            }
            Stmt::For(p, it, body, _) => {
                let itv = self.eval(it)?;
                let items = self.iter_items(itv)?;
                for x in items {
                    self.scopes.push(HashMap::new());
                    let r = match self.pmatch(p, &x) {
                        Ok(_) => self.exec_block(body),
                        Err(e) => Err(e),
                    };
                    self.scopes.pop();
                    match r {
                        Ok(()) | Err(Flow::Cont) => {}
                        Err(Flow::Brk) => break,
                        Err(f) => return Err(f),
                    }
                }
            }
            Stmt::Match(e, arms, _) => {
                let v = self.eval(e)?;
                for a in arms {
                    self.scopes.push(HashMap::new());
                    let ok = match self.pmatch(&a.pat, &v) {
                        Ok(b) => b,
                        Err(f) => {
                            self.scopes.pop();
                            return Err(f);
                        }
                    };
                    let ok = ok
                        && match &a.guard {
                            Some(g) => match self.eval(g) {
                                Ok(x) => self.truthy(x)?,
                                Err(f) => {
                                    self.scopes.pop();
                                    return Err(f);
                                }
                            },
                            None => true,
                        };
                    if ok {
                        let r = self.exec_block_tail(&a.body);
                        self.scopes.pop();
                        return r.map(|v| if tail { v } else { None });
                    }
                    self.scopes.pop();
                }
                return trap("no match arm");
            }
            Stmt::Return(e, _) => {
                let v = match e {
                    Some(e) => self.eval(e)?,
                    None => V::Unit,
                };
                return Err(Flow::Ret(v));
            }
            Stmt::Break(_) => return Err(Flow::Brk),
            Stmt::Continue(_) => return Err(Flow::Cont),
            Stmt::Pass => {}
            Stmt::Emit(e) => {
                let v = self.eval(e)?;
                self.outputs.push(v);
            }
            Stmt::Fail(e) => {
                let v = self.eval(e)?;
                return Err(Flow::Fail(v));
            }
            Stmt::Arena(_, Some(b), body, _) | Stmt::Device(_, Some(b), body, _) => {
                let bound = match self.eval(b)? {
                    V::Int(n, _) => n,
                    _ => return trap("bad arena bound"),
                };
                // R3: the region's live bytes are the canonical-JSON sizes of the
                // values bound in it, measured after every statement of its body.
                let base = self.scopes.len();
                self.scopes.push(HashMap::new());
                let mut r = Ok(None);
                for s in body.iter() {
                    r = self.exec(s, false);
                    if r.is_err() {
                        break;
                    }
                    let used: i128 = self.scopes[base..]
                        .iter()
                        .flat_map(|sc| sc.values())
                        .map(|v| json_size(v) as i128)
                        .sum();
                    if used > bound {
                        r = trap("size limit");
                        break;
                    }
                }
                self.scopes.truncate(base);
                r?;
            }
            Stmt::Arena(_, _, body, _) | Stmt::Device(_, _, body, _) | Stmt::Comptime(body) => {
                self.exec_block(body)?;
            }
            Stmt::Yield(e, _) => {
                let v = self.eval(e)?;
                let lim = self.loop_limit;
                let (buf, cap) = match self.yields.last_mut() {
                    Some(x) => x,
                    None => return trap("yield outside an iterator"),
                };
                if buf.len() as i128 >= lim {
                    return trap("loop bound");
                }
                buf.push(v);
                if buf.len() >= *cap {
                    return Err(Flow::Stop);
                }
            }
            Stmt::Check(e) => {
                let v = self.eval(e)?;
                if !self.truthy(v)? {
                    return trap("check failed");
                }
            }
            Stmt::Send(..) | Stmt::Spawn(..) | Stmt::Select(..) => {
                return trap("concurrency is not available at comptime");
            }
        }
        Ok(None)
    }
}

pub type Region = (String, Vec<(String, V, Ty)>);

impl<'a> Interp<'a> {
    pub fn initial_state(&mut self) -> R<Vec<Region>> {
        let mut st = vec![];
        for i in &self.m.items {
            if let Item::State { name, fields, .. } = i {
                let mut fs = vec![];
                for f in fields {
                    self.scopes = vec![HashMap::new()];
                    let v = match &f.init {
                        Some(e) => self.eval(e)?,
                        None => return trap("field without initial value"),
                    };
                    fs.push((f.name.clone(), self.fit(v, &f.ty)?, f.ty.clone()));
                }
                st.push((name.clone(), fs));
            }
        }
        Ok(st)
    }

    #[allow(clippy::type_complexity)]
    fn region_step(
        &mut self,
        r: &Region,
        ev: &V,
    ) -> Result<Option<(Vec<(String, V, Ty)>, Vec<V>)>, String> {
        let m = self.m;
        for it in &m.items {
            let Item::Step {
                region,
                pat,
                guard,
                body,
                ..
            } = it
            else {
                continue;
            };
            if *region != r.0 {
                continue;
            }
            self.scopes = vec![
                r.1.iter().map(|(n, v, _)| (n.clone(), v.clone())).collect(),
                HashMap::new(),
            ];
            self.outputs.clear();
            let refuse = |f: Flow| match f {
                Flow::Fail(v) => v.rec_name().unwrap_or("?").to_string(),
                Flow::Trap(t) => format!("trap({t})"),
                other => format!("trap({other:?})"),
            };
            match self.pmatch(pat, ev) {
                Ok(true) => {}
                Ok(false) => continue,
                Err(f) => return Err(refuse(f)),
            }
            if let Some(g) = guard {
                match self.eval(g).and_then(|v| self.truthy(v)) {
                    Ok(true) => {}
                    Ok(false) => continue,
                    Err(f) => return Err(refuse(f)),
                } // S1
            }
            let res = match body {
                Body::Arrow(e) => self.eval(e).map(|v| {
                    let n = r.1[0].0.clone();
                    self.set(&n, v);
                }),
                Body::Block(b) => self.exec_block(b),
            };
            if let Err(f) = res {
                return Err(refuse(f));
            }
            let mut out = vec![];
            for (n, _, t) in &r.1 {
                let v = self.scopes[0][n].clone();
                match self.fit(v, t) {
                    Ok(v) => out.push((n.clone(), v, t.clone())),
                    Err(f) => return Err(refuse(f)),
                }
            }
            return Ok(Some((out, std::mem::take(&mut self.outputs))));
        }
        Ok(None)
    }

    /// δ (6.4): every region on the pre-state; first refusal in declaration order refuses the event atomically.
    pub fn delta(&mut self, s: &[Region], ev: &V) -> Result<(Vec<Region>, Vec<V>), String> {
        let mut next = vec![];
        let mut outs = vec![];
        for r in s {
            match self.region_step(r, ev)? {
                Some((fs, o)) => {
                    next.push((r.0.clone(), fs));
                    outs.extend(o);
                }
                None => next.push(r.clone()),
            }
        }
        Ok((next, outs))
    }
}

fn walk_comptime<'b>(b: &'b [Stmt], out: &mut Vec<&'b [Stmt]>) {
    for s in b {
        match s {
            Stmt::Comptime(x) => out.push(x),
            Stmt::If(arms, els) => {
                for (_, x) in arms {
                    walk_comptime(x, out);
                }
                if let Some(x) = els {
                    walk_comptime(x, out);
                }
            }
            Stmt::While(_, _, x)
            | Stmt::Loop(_, x)
            | Stmt::For(_, _, x, _)
            | Stmt::Arena(_, _, x, _) => walk_comptime(x, out),
            Stmt::Match(_, arms, _) => {
                for a in arms {
                    walk_comptime(&a.body, out)
                }
            }
            _ => {}
        }
    }
}

pub fn why(f: Flow) -> String {
    match f {
        Flow::Trap(t) => format!("trap: {t}"),
        Flow::Fail(v) => format!("failed with {}", v.show()),
        other => format!("{other:?}"),
    }
}

/// Admission gate (W-CT-1, W-INT-1). Called only on modules that type-check.
pub fn admit(m: &Module) -> Vec<String> {
    let mut errs = vec![];
    let mut it = Interp::new(m);
    for i in &m.items {
        if let Item::Const { name, line, .. } = i
            && let Err(f) = it.const_value(name)
        {
            errs.push(format!("line {line}: W-CT-1 const `{name}`: {}", why(f)));
        }
    }
    let s0 = match it.initial_state() {
        Ok(s) => Some(s),
        Err(f) => {
            errs.push(format!("line 0: W-CT-1 state initializer: {}", why(f)));
            None
        }
    };
    let mut run_check = |it: &mut Interp, e: &Expr, rule: &str, errs: &mut Vec<String>| {
        it.scopes = vec![HashMap::new()];
        match it.eval(e).and_then(|v| it.truthy(v)) {
            Ok(true) => {}
            Ok(false) => errs.push(format!("line {}: {rule} check is false", e.line())),
            Err(f) => errs.push(format!("line {}: {rule} check: {}", e.line(), why(f))),
        }
    };
    for i in &m.items {
        match i {
            Item::Check(e) => run_check(&mut it, e, "W-INT-1", &mut errs),
            Item::Intent(_, _, cks) => {
                for e in cks {
                    run_check(&mut it, e, "W-INT-1", &mut errs)
                }
            }
            _ => {}
        }
    }
    // comptime blocks inside function bodies
    let mut bodies: Vec<&FnDecl> = vec![];
    for i in &m.items {
        match i {
            Item::Fn(f) => bodies.push(f),
            Item::Impl { members, .. } => bodies.extend(members.iter()),
            _ => {}
        }
    }
    for f in bodies {
        if let Body::Block(b) = &f.body {
            let mut cbs = vec![];
            walk_comptime(b, &mut cbs);
            for cb in cbs {
                it.scopes = vec![HashMap::new()];
                if let Err(f2) = it.exec_block(cb) {
                    errs.push(format!(
                        "line {}: W-CT-1 comptime block in `{}`: {}",
                        f.sig.line,
                        f.sig.name,
                        why(f2)
                    ));
                }
            }
        }
    }
    // examples (6.9)
    let mut exs: Vec<&Example> = vec![];
    for i in &m.items {
        match i {
            Item::Example(x) => exs.push(x),
            Item::Intent(_, xs, _) => exs.extend(xs.iter()),
            _ => {}
        }
    }
    if let Some(s0) = s0 {
        for x in exs {
            let mut s = s0.clone();
            let n = x.events.len();
            let mut bad = None;
            for (k, (evn, args)) in x.events.iter().enumerate() {
                it.scopes = vec![HashMap::new()];
                let ev = match it.construct(evn, args) {
                    Ok(Some(v)) => v,
                    Ok(None) => {
                        bad = Some(format!("`{evn}` is not an event"));
                        break;
                    }
                    Err(f) => {
                        bad = Some(why(f));
                        break;
                    }
                };
                match it.delta(&s, &ev) {
                    Ok((s2, _)) => {
                        if k + 1 == n && x.refused.is_some() {
                            bad = Some(format!(
                                "expected refused: {}, but {evn} was applied",
                                x.refused.clone().unwrap()
                            ));
                        }
                        s = s2;
                    }
                    Err(r) => {
                        if k + 1 == n && x.refused.as_deref() == Some(r.as_str()) {
                        } else {
                            bad = Some(format!("event {} ({evn}) refused: {r}", k + 1));
                        }
                        break;
                    }
                }
            }
            if bad.is_none()
                && let Some(v) = &x.value
            {
                it.scopes = vec![
                    s.iter()
                        .flat_map(|r| r.1.iter().map(|(n, v, _)| (n.clone(), v.clone())))
                        .collect(),
                ];
                match it.eval(v) {
                    Ok(want) => {
                        let single = if s.len() == 1 && s[0].1.len() == 1 {
                            Some(s[0].1[0].1.clone())
                        } else {
                            None
                        };
                        let ok = match (&single, &want) {
                            (Some(got), w)
                                if !matches!(w, V::Bool(_)) || matches!(got, V::Bool(_)) =>
                            {
                                got == w
                            }
                            _ => want == V::Bool(true),
                        };
                        if !ok {
                            bad = Some(match single {
                                Some(g) => {
                                    format!("final value is {}, expected {}", g.show(), want.show())
                                }
                                None => "final-state condition is false".into(),
                            });
                        }
                    }
                    Err(f) => bad = Some(why(f)),
                }
            }
            if let Some(b) = bad {
                errs.push(format!(
                    "line {}: W-INT-1 example does not hold: {b}",
                    x.line
                ));
            }
        }
    }
    errs
}

/// Size in bytes of a value's canonical JSON encoding (R3 arena accounting).
pub fn json_size(v: &V) -> usize {
    fn esc(s: &str) -> usize {
        2 + s
            .chars()
            .map(|c| match c {
                '"' | '\\' | '\n' | '\r' | '\t' => 2,
                c if (c as u32) < 0x20 => 6,
                c => c.len_utf8(),
            })
            .sum::<usize>()
    }
    let seq =
        |xs: &mut dyn Iterator<Item = usize>, n: usize| 2 + xs.sum::<usize>() + n.saturating_sub(1);
    match v {
        V::Int(x, _) => x.to_string().len(),
        V::F(f) | V::F32(f) => format!("{f:?}").len(),
        V::Bool(b) => {
            if *b {
                4
            } else {
                5
            }
        }
        V::Text(s) => esc(s),
        V::Unit | V::Opt(None) => 4,
        V::Opt(Some(x)) | V::Err(x) => json_size(x),
        V::Tuple(xs) | V::List(xs) => seq(&mut xs.iter().map(json_size), xs.len()),
        V::Map(m) => seq(
            &mut m.iter().map(|(k, v)| 3 + json_size(k) + json_size(v)),
            m.len(),
        ),
        V::Rec(n, fs) => {
            2 + esc(n)
                + 1
                + seq(
                    &mut fs.iter().map(|(a, b)| esc(a) + 1 + json_size(b)),
                    fs.len(),
                )
        }
        V::Range(a, b, _) => 3 + a.to_string().len() + b.to_string().len(),
        V::Fn(n) => esc(n),
        V::Closure(..) | V::Gen(_) => 2,
    }
}

/// One example run, event by event, for display (used by the browser build).
pub struct TraceStep {
    pub event: String,
    #[allow(clippy::type_complexity)]
    pub result: Result<Vec<(String, Vec<(String, String)>)>, String>,
}

/// Run every example of a type-checked module and record the state after each event.
pub fn trace_examples(m: &Module) -> Vec<(usize, Vec<TraceStep>)> {
    let mut out = vec![];
    let mut it = Interp::new(m);
    let Ok(s0) = it.initial_state() else {
        return out;
    };
    let show = |s: &[Region]| {
        s.iter()
            .map(|r| {
                (
                    r.0.clone(),
                    r.1.iter().map(|(n, v, _)| (n.clone(), v.show())).collect(),
                )
            })
            .collect::<Vec<_>>()
    };
    let mut exs: Vec<&Example> = vec![];
    for i in &m.items {
        match i {
            Item::Example(x) => exs.push(x),
            Item::Intent(_, xs, _) => exs.extend(xs.iter()),
            _ => {}
        }
    }
    for x in exs {
        let mut s = s0.clone();
        let mut steps = vec![TraceStep {
            event: "(initial)".into(),
            result: Ok(show(&s)),
        }];
        for (evn, args) in &x.events {
            it.scopes = vec![HashMap::new()];
            let label = if args.is_empty() {
                evn.clone()
            } else {
                format!("{evn}(..)")
            };
            let ev = match it.construct(evn, args) {
                Ok(Some(v)) => v,
                Ok(None) => {
                    steps.push(TraceStep {
                        event: label,
                        result: Err("not an event".into()),
                    });
                    break;
                }
                Err(f) => {
                    steps.push(TraceStep {
                        event: label,
                        result: Err(why(f)),
                    });
                    break;
                }
            };
            let label = ev.show();
            match it.delta(&s, &ev) {
                Ok((s2, _)) => {
                    s = s2;
                    steps.push(TraceStep {
                        event: label,
                        result: Ok(show(&s)),
                    });
                }
                Err(r) => {
                    steps.push(TraceStep {
                        event: label,
                        result: Err(r),
                    });
                    break;
                }
            }
        }
        out.push((x.line, steps));
    }
    out
}

// ---------- The Form of a program as JSON (spec 6.1) ----------

fn jstr(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// A Form value (6.1): null, bool, number, text, list, record; ordered canonically.
#[derive(Clone, Debug, PartialEq)]
pub enum FV {
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    Text(String),
    List(Vec<FV>),
    Rec(Vec<(String, FV)>),
}
impl FV {
    fn rank(&self) -> u8 {
        match self {
            FV::Null => 0,
            FV::Bool(_) => 1,
            FV::Int(_) | FV::Float(_) => 2,
            FV::Text(_) => 3,
            FV::List(_) => 4,
            FV::Rec(_) => 5,
        }
    }
    fn num(&self) -> f64 {
        match self {
            FV::Int(i) => *i as f64,
            FV::Float(f) => *f,
            _ => 0.0,
        }
    }
    /// Canonical order of 6.1: null < bool < number < text < list < record.
    pub fn cmp_canon(&self, o: &FV) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        let r = self.rank().cmp(&o.rank());
        if r != Equal {
            return r;
        }
        match (self, o) {
            (FV::Bool(a), FV::Bool(b)) => a.cmp(b),
            (FV::Int(a), FV::Int(b)) => a.cmp(b),
            (a @ (FV::Int(_) | FV::Float(_)), b) => a.num().partial_cmp(&b.num()).unwrap_or(Equal),
            (FV::Text(a), FV::Text(b)) => a.chars().cmp(b.chars()),
            (FV::List(a), FV::List(b)) => {
                for (x, y) in a.iter().zip(b) {
                    let c = x.cmp_canon(y);
                    if c != Equal {
                        return c;
                    }
                }
                a.len().cmp(&b.len())
            }
            (FV::Rec(a), FV::Rec(b)) => {
                for ((ka, va), (kb, vb)) in a.iter().zip(b) {
                    let c = ka.cmp(kb).then_with(|| va.cmp_canon(vb));
                    if c != Equal {
                        return c;
                    }
                }
                a.len().cmp(&b.len())
            }
            _ => Equal,
        }
    }
    pub fn json(&self) -> String {
        match self {
            FV::Null => "null".into(),
            FV::Bool(b) => b.to_string(),
            FV::Int(i) => i.to_string(),
            FV::Float(f) => {
                let s = format!("{f:?}");
                s
            }
            FV::Text(t) => jstr(t),
            FV::List(xs) => format!(
                "[{}]",
                xs.iter().map(|x| x.json()).collect::<Vec<_>>().join(",")
            ),
            FV::Rec(fs) => format!(
                "{{{}}}",
                fs.iter()
                    .map(|(k, v)| format!("{}:{}", jstr(k), v.json()))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        }
    }
}

fn rec(mut fs: Vec<(String, FV)>) -> FV {
    fs.sort_by(|a, b| a.0.cmp(&b.0));
    FV::Rec(fs)
}

/// Embed an evaluated value into the Form's values (the table in 6.1).
pub fn form_value(v: &V, m: &Module) -> FV {
    let is_variant = |n: &str| {
        m.items.iter().any(
            |i| matches!(i, Item::Kind { variants, .. } if variants.iter().any(|(vn, _)| vn == n)),
        )
    };
    match v {
        V::Int(i, _) => FV::Int(*i),
        V::F(f) | V::F32(f) => FV::Float(*f),
        V::Bool(b) => FV::Bool(*b),
        V::Text(t) => FV::Text(t.clone()),
        V::Unit | V::Opt(None) => FV::Null,
        V::Opt(Some(x)) => form_value(x, m),
        V::Tuple(xs) | V::List(xs) => FV::List(xs.iter().map(|x| form_value(x, m)).collect()),
        V::Map(ps) => {
            let mut l: Vec<(FV, FV)> = ps
                .iter()
                .map(|(k, v)| (form_value(k, m), form_value(v, m)))
                .collect();
            l.sort_by(|a, b| a.0.cmp_canon(&b.0));
            FV::List(l.into_iter().map(|(k, v)| FV::List(vec![k, v])).collect())
        }
        V::Rec(n, fs) => {
            let mut f: Vec<(String, FV)> = fs
                .iter()
                .map(|(k, x)| (k.clone(), form_value(x, m)))
                .collect();
            if is_variant(n) {
                f.push(("$v".into(), FV::Text(n.clone())));
            }
            rec(f)
        }
        V::Err(x) => rec(vec![("$err".into(), form_value(x, m))]),
        V::Range(a, b, _) => FV::List((*a..*b).map(FV::Int).collect()),
        V::Fn(n) => rec(vec![("$fn".into(), FV::Text(n.clone()))]),
        V::Closure(..) | V::Gen(_) => FV::Null,
    }
}

pub fn ty_str(t: &Ty) -> String {
    let list = |ts: &[Ty]| ts.iter().map(ty_str).collect::<Vec<_>>().join(", ");
    match t {
        Ty::Path(p, g) if g.is_empty() => p.join("."),
        Ty::Path(p, g) => format!("{}[{}]", p.join("."), list(g)),
        Ty::Unit => "()".into(),
        Ty::Tuple(ts) => format!("({})", list(ts)),
        Ty::Slice(t) => format!("[{}]", ty_str(t)),
        Ty::Array(t, n) => match &**n {
            Expr::Int(k, _) => format!("[{}; {k}]", ty_str(t)),
            _ => format!("[{}; n]", ty_str(t)),
        },
        Ty::Chan(t) => format!("chan {}", ty_str(t)),
        Ty::Iter(t) => format!("iter {}", ty_str(t)),
        Ty::Fn(a, r) => format!("fn({}) -> {}", list(a), ty_str(r)),
        Ty::Ref(mt, t) => format!("&{}{}", if *mt { "mut " } else { "" }, ty_str(t)),
        Ty::Opt(t) => format!("{}?", ty_str(t)),
        Ty::Union(a, b) => format!("{} ! {}", ty_str(a), ty_str(b)),
        Ty::Region(t, r) => format!("{} @{r}", ty_str(t)),
    }
}

fn pat_str(p: &Pat) -> String {
    match p {
        Pat::Wild => "_".into(),
        Pat::Path(x) => x.join("."),
        Pat::Ctor(x, args) => format!(
            "{}({})",
            x.join("."),
            args.iter()
                .map(|(l, q)| match l {
                    Some(l) => format!("{l}: {}", pat_str(q)),
                    None => pat_str(q),
                })
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Pat::Lit(_) => "<literal>".into(),
        Pat::Range(..) => "<range>".into(),
        Pat::Tuple(ps) => format!(
            "({})",
            ps.iter().map(pat_str).collect::<Vec<_>>().join(", ")
        ),
        Pat::Alt(ps) => ps.iter().map(pat_str).collect::<Vec<_>>().join(" | "),
    }
}

/// The Form P = (S, E, δ, D, λ) of an admitted module (6.1), as JSON.
/// δ is given as its step table (region, pattern, guard, source line); the
/// step bodies are the program's own. D is the log of each example, with each
/// entry's outcome, outputs (λ) and the state after it.
pub fn form_json(m: &Module) -> String {
    let fields = |fs: &[Field]| {
        fs.iter()
            .map(|f| {
                format!(
                    "{{\"name\":{},\"type\":{}}}",
                    jstr(&f.name),
                    jstr(&ty_str(&f.ty))
                )
            })
            .collect::<Vec<_>>()
            .join(",")
    };
    let payload = |ps: &[(String, Ty)]| {
        ps.iter()
            .map(|(n, t)| format!("{{\"name\":{},\"type\":{}}}", jstr(n), jstr(&ty_str(t))))
            .collect::<Vec<_>>()
            .join(",")
    };
    let mut regions = vec![];
    let mut events = vec![];
    let mut errors = vec![];
    let mut steps = vec![];
    for i in &m.items {
        match i {
            Item::State {
                name, fields: fs, ..
            } => regions.push(format!(
                "{{\"name\":{},\"kind\":\"state\",\"fields\":[{}]}}",
                jstr(name),
                fields(fs)
            )),
            Item::Actor {
                name,
                fields: fs,
                handlers,
                ..
            } => {
                regions.push(format!(
                    "{{\"name\":{},\"kind\":\"actor\",\"fields\":[{}],\"mailbox\":true}}",
                    jstr(name),
                    fields(fs)
                ));
                for h in handlers {
                    steps.push(format!(
                        "{{\"region\":{},\"on\":{},\"guarded\":{},\"line\":{}}}",
                        jstr(name),
                        jstr(&pat_str(&h.pat)),
                        h.guard.is_some(),
                        h.line
                    ));
                }
            }
            Item::Event {
                name, payload: p, ..
            } => events.push(format!(
                "{{\"name\":{},\"payload\":[{}]}}",
                jstr(name),
                payload(p)
            )),
            Item::Error {
                name, payload: p, ..
            } => errors.push(format!(
                "{{\"name\":{},\"payload\":[{}]}}",
                jstr(name),
                payload(p)
            )),
            Item::Step {
                region,
                pat,
                guard,
                line,
                ..
            } => steps.push(format!(
                "{{\"region\":{},\"on\":{},\"guarded\":{},\"line\":{line}}}",
                jstr(region),
                jstr(&pat_str(pat)),
                guard.is_some()
            )),
            _ => {}
        }
    }
    events.push(
        "{\"name\":\"Tick\",\"payload\":[{\"name\":\"n\",\"type\":\"u64\"}],\"builtin\":true}"
            .into(),
    );
    let state_fv = |s: &[Region]| {
        rec(s
            .iter()
            .map(|r| {
                (
                    r.0.clone(),
                    rec(r
                        .1
                        .iter()
                        .map(|(n, v, _)| (n.clone(), form_value(v, m)))
                        .collect()),
                )
            })
            .collect())
    };
    let mut it = Interp::new(m);
    let s0 = it.initial_state().ok();
    let s0_json = s0
        .as_ref()
        .map(|s| state_fv(s).json())
        .unwrap_or_else(|| "null".into());
    let mut logs = vec![];
    if let Some(s0) = &s0 {
        let mut exs: Vec<&Example> = vec![];
        for i in &m.items {
            match i {
                Item::Example(x) => exs.push(x),
                Item::Intent(_, xs, _) => exs.extend(xs.iter()),
                _ => {}
            }
        }
        for x in exs {
            let mut s = s0.clone();
            let mut entries = vec![];
            for (evn, args) in &x.events {
                it.scopes = vec![HashMap::new()];
                let ev = match it.construct(evn, args) {
                    Ok(Some(v)) => v,
                    _ => break,
                };
                let evj = form_value(&ev, m).json();
                let evj = format!("{{\"event\":{},\"payload\":{}}}", jstr(evn), evj);
                match it.delta(&s, &ev) {
                    Ok((s2, outs)) => {
                        s = s2;
                        entries.push(format!(
                            "{{\"in\":{evj},\"outcome\":\"applied\",\"out\":[{}],\"state\":{}}}",
                            outs.iter()
                                .map(|o| form_value(o, m).json())
                                .collect::<Vec<_>>()
                                .join(","),
                            state_fv(&s).json()
                        ));
                    }
                    Err(r) => {
                        entries.push(format!(
                            "{{\"in\":{evj},\"outcome\":{{\"refused\":{}}},\"out\":[{{\"Refused\":{{\"error\":{},\"payload\":{}}}}}],\"state\":{}}}",
                            jstr(&r),
                            jstr(&r),
                            form_value(&ev, m).json(),
                            state_fv(&s).json()
                        ));
                        break;
                    }
                }
            }
            logs.push(format!(
                "{{\"example_line\":{},\"D\":[{}]}}",
                x.line,
                entries.join(",")
            ));
        }
    }
    format!(
        "{{\"S\":{{\"regions\":[{}],\"s0\":{}}},\"E\":[{}],\"errors\":[{}],\"delta\":[{}],\"logs\":[{}]}}",
        regions.join(","),
        s0_json,
        events.join(","),
        errors.join(","),
        steps.join(","),
        logs.join(",")
    )
}
