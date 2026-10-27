//! The reference evaluator of the core (one-emitter.md, order of work step 2).
//!
//! It runs a core module that has passed the well-formedness check (W1 to
//! W6, `wf.rs`) and returns the bytes the module writes to standard output.
//! Each primitive is total on its stated domain; the evaluator reports an
//! application outside the domain as an internal error, because W4 places a
//! domain test before every partial application and a well-formed module
//! never reaches one.
//!
//! Integer values are held as i128, which holds every value of every core
//! integer type (-2^63 to 2^64 - 1) and every exact sum, difference and
//! negation of two of them. A product is computed with an overflow check
//! before it is compared with its type.
use super::ir::*;
use std::rc::Rc;

/// A core value.
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    Bool(bool),
    Int(i128),
    /// f64 and f32 values (an f32 value is held exactly as binary64)
    F(f64),
    Bytes(Rc<[u8]>),
    Arr(Vec<Val>),
    List(Vec<Val>),
    Tup(Vec<Val>),
    Sum(u32, Box<Val>),
}

impl Val {
    pub fn unit() -> Val {
        Val::Tup(vec![])
    }
    pub fn bytes(b: &[u8]) -> Val {
        Val::Bytes(Rc::from(b))
    }
    fn int(&self) -> Result<i128, Flow> {
        match self {
            Val::Int(v) => Ok(*v),
            o => Err(bad(format!("expected an integer, found {o:?}"))),
        }
    }
    fn boolean(&self) -> Result<bool, Flow> {
        match self {
            Val::Bool(b) => Ok(*b),
            o => Err(bad(format!("expected a bool, found {o:?}"))),
        }
    }
    fn float(&self) -> Result<f64, Flow> {
        match self {
            Val::F(f) => Ok(*f),
            o => Err(bad(format!("expected a float, found {o:?}"))),
        }
    }
    fn byt(&self) -> Result<&Rc<[u8]>, Flow> {
        match self {
            Val::Bytes(b) => Ok(b),
            o => Err(bad(format!("expected bytes, found {o:?}"))),
        }
    }
}

/// The value of a literal.
pub fn lit_val(l: &Lit) -> Val {
    match l {
        Lit::Bool(b) => Val::Bool(*b),
        Lit::Int(v) => Val::Int(*v),
        Lit::F64(b) | Lit::F32(b) => Val::F(f64::from_bits(*b)),
        Lit::Bytes(b) => Val::bytes(b),
    }
}

/// What ends an evaluation other than a value.
#[derive(Clone, Debug)]
pub enum Flow {
    /// `halt a`, with a : Halt
    Halt(Val),
    /// an internal error: the module is not well formed, or the evaluator met
    /// a primitive applied outside its domain
    Bad(String),
}

fn bad(m: impl Into<String>) -> Flow {
    Flow::Bad(m.into())
}

enum Ctl {
    Next,
    Break(u32),
    Ret(Val),
}

/// A statement with the index of the local it binds made explicit. Locals
/// are numbered in textual order, so a binding inside a branch that is not
/// taken leaves its index unused, and a loop body binds the same indices
/// on every iteration.
enum P<'a> {
    Let(u32, &'a Atom),
    Prim(u32, &'a str, &'a [Ty], &'a [Atom]),
    Call(u32, &'a FnRef, &'a [Atom]),
    Catch(u32, &'a FnRef, &'a [Atom]),
    Var(u32, &'a Atom),
    Set(&'a Place, &'a Atom),
    If(&'a Atom, Vec<P<'a>>, Vec<P<'a>>),
    Block(Vec<P<'a>>),
    Loop(Vec<P<'a>>),
    Break(u32),
    Ret(&'a Atom),
    Halt(&'a Atom),
    Out(&'a Atom),
}

struct PFn<'a> {
    nlocals: usize,
    body: Vec<P<'a>>,
}

fn prepare<'a>(b: &'a [Stmt], next: &mut u32) -> Vec<P<'a>> {
    let mut out = Vec::with_capacity(b.len());
    for s in b {
        let at = *next;
        *next += binds(s) as u32;
        out.push(match s {
            Stmt::Let(a) => P::Let(at, a),
            Stmt::Prim(p, t, a) => P::Prim(at, p, t, a),
            Stmt::Call(f, a) => P::Call(at, f, a),
            Stmt::Catch(f, a) => P::Catch(at, f, a),
            Stmt::Var(_, a) => P::Var(at, a),
            Stmt::Set(p, a) => P::Set(p, a),
            Stmt::If(c, t, e) => {
                let t = prepare(t, next);
                let e = prepare(e, next);
                P::If(c, t, e)
            }
            Stmt::Block(x) => P::Block(prepare(x, next)),
            Stmt::Loop(x) => P::Loop(prepare(x, next)),
            Stmt::Break(d) => P::Break(*d),
            Stmt::Ret(a) => P::Ret(a),
            Stmt::Halt(a) => P::Halt(a),
            Stmt::Out(a) => P::Out(a),
        });
    }
    out
}

/// The result of running a core module: its standard output, and the halt
/// that ended it, if one did.
pub struct Run {
    pub out: Vec<u8>,
    pub halt: Option<Val>,
}

/// Run the entry function of `m` with no arguments.
pub fn run(m: &Module) -> Result<Run, String> {
    let mut ev = Eval::new(m);
    let r = ev.call_key(&m.entry.0, m.entry.1, vec![]);
    match r {
        Ok(_) => Ok(Run { out: ev.out, halt: None }),
        Err(Flow::Halt(h)) => Ok(Run { out: ev.out, halt: Some(h) }),
        Err(Flow::Bad(e)) => Err(e),
    }
}

pub struct Eval<'a> {
    m: &'a Module,
    prepared: std::collections::HashMap<(&'a str, u32), Rc<PFn<'a>>>,
    globals: Vec<Option<Val>>,
    /// the one global array of integers of `mem.get` and `mem.set`
    mem: Vec<i128>,
    pub out: Vec<u8>,
}

struct Frame<'a> {
    key: &'a str,
    /// indexed by local number; `None` before binding and after consumption
    locals: Vec<Option<Val>>,
}

impl<'a> Eval<'a> {
    pub fn new(m: &'a Module) -> Self {
        Eval {
            m,
            prepared: Default::default(),
            globals: m.globals.iter().map(|(_, l)| l.as_ref().map(lit_val)).collect(),
            mem: vec![],
            out: vec![],
        }
    }

    pub fn call_key(&mut self, key: &str, i: u32, args: Vec<Val>) -> Result<Val, Flow> {
        let m = self.m;
        let Some((k, g)) = m.defs.get_key_value(key) else {
            return Err(bad(format!("no group {key}")));
        };
        let Some(f) = g.fns.get(i as usize) else {
            return Err(bad(format!("no function {i} in group {key}")));
        };
        if f.params.len() != args.len() {
            return Err(bad("argument count"));
        }
        let k: &'a str = k.as_str();
        let pf = match self.prepared.get(&(k, i)) {
            Some(p) => p.clone(),
            None => {
                let mut next = f.params.len() as u32;
                let body = prepare(&f.body, &mut next);
                let p = Rc::new(PFn { nlocals: next as usize, body });
                self.prepared.insert((k, i), p.clone());
                p
            }
        };
        let mut locals: Vec<Option<Val>> = args.into_iter().map(Some).collect();
        locals.resize(pf.nlocals, None);
        let mut fr = Frame { key: k, locals };
        let r = stacker::maybe_grow(256 * 1024, 8 * 1024 * 1024, || self.block(&pf.body, &mut fr))?;
        match r {
            Ctl::Ret(v) => Ok(v),
            Ctl::Next => Err(bad("a function ended without ret or halt")),
            Ctl::Break(_) => Err(bad("break outside a block or loop")),
        }
    }

    fn atom(&mut self, a: &Atom, fr: &mut Frame, consume: bool) -> Result<Val, Flow> {
        match a {
            Atom::K(_, l) => Ok(lit_val(l)),
            Atom::V(i) => {
                let slot = fr.locals.get_mut(*i as usize).ok_or_else(|| bad(format!("no local v{i}")))?;
                let v = slot.as_ref().ok_or_else(|| bad(format!("v{i} was used after it was consumed")))?;
                let owned = matches!(v, Val::Arr(_) | Val::List(_) | Val::Tup(_) | Val::Sum(..));
                if consume && owned { Ok(slot.take().unwrap()) } else { Ok(v.clone()) }
            }
            Atom::G(i) => {
                let slot = self.globals.get_mut(*i as usize).ok_or_else(|| bad(format!("no global g{i}")))?;
                let v = slot.as_ref().ok_or_else(|| bad(format!("g{i} was read before it was set")))?;
                let owned = matches!(v, Val::Arr(_) | Val::List(_) | Val::Tup(_) | Val::Sum(..));
                if consume && owned { Ok(slot.take().unwrap()) } else { Ok(v.clone()) }
            }
        }
    }

    fn atoms(&mut self, a: &[Atom], fr: &mut Frame, consume: &[bool]) -> Result<Vec<Val>, Flow> {
        let mut v = Vec::with_capacity(a.len());
        for (k, x) in a.iter().enumerate() {
            v.push(self.atom(x, fr, consume.get(k).copied().unwrap_or(true))?);
        }
        Ok(v)
    }

    fn callee(&self, f: &FnRef, key: &str) -> (String, u32) {
        match f {
            FnRef::Key(k, i) => (k.clone(), *i),
            FnRef::Rec(i) => (key.to_string(), *i),
        }
    }

    fn block(&mut self, b: &[P<'a>], fr: &mut Frame) -> Result<Ctl, Flow> {
        for s in b {
            match self.stmt(s, fr)? {
                Ctl::Next => {}
                c => return Ok(c),
            }
        }
        Ok(Ctl::Next)
    }

    fn put(fr: &mut Frame, at: u32, v: Val) {
        fr.locals[at as usize] = Some(v);
    }

    fn stmt(&mut self, s: &P<'a>, fr: &mut Frame) -> Result<Ctl, Flow> {
        match s {
            P::Let(at, a) => {
                let v = self.atom(a, fr, true)?;
                Self::put(fr, *at, v);
            }
            P::Prim(at, p, ty, args) => {
                let rs = self.prim(p, ty, args, fr)?;
                for (k, v) in rs.into_iter().enumerate() {
                    Self::put(fr, *at + k as u32, v);
                }
            }
            P::Call(at, f, args) => {
                let vs = self.atoms(args, fr, &[])?;
                let (k, i) = self.callee(f, fr.key);
                let v = self.call_key(&k, i, vs)?;
                Self::put(fr, *at, v);
            }
            P::Catch(at, f, args) => {
                let vs = self.atoms(args, fr, &[])?;
                let (k, i) = self.callee(f, fr.key);
                let v = match self.call_key(&k, i, vs) {
                    Ok(v) => Val::Sum(0, Box::new(v)),
                    Err(Flow::Halt(h)) => Val::Sum(1, Box::new(h)),
                    Err(e) => return Err(e),
                };
                Self::put(fr, *at, v);
            }
            P::Var(at, a) => {
                let v = self.atom(a, fr, true)?;
                Self::put(fr, *at, v);
            }
            P::Set(p, a) => {
                let v = self.atom(a, fr, true)?;
                match p {
                    Place::V(i) => {
                        *fr.locals.get_mut(*i as usize).ok_or_else(|| bad("set of an unknown local"))? = Some(v)
                    }
                    Place::G(i) => {
                        *self.globals.get_mut(*i as usize).ok_or_else(|| bad("set of an unknown global"))? = Some(v)
                    }
                }
            }
            P::If(c, t, e) => {
                let c = self.atom(c, fr, false)?.boolean()?;
                return self.block(if c { t } else { e }, fr);
            }
            P::Block(b) => {
                return Ok(match self.block(b, fr)? {
                    Ctl::Break(0) => Ctl::Next,
                    Ctl::Break(d) => Ctl::Break(d - 1),
                    c => c,
                });
            }
            P::Loop(b) => loop {
                match self.block(b, fr)? {
                    Ctl::Next => {}
                    Ctl::Break(0) => return Ok(Ctl::Next),
                    Ctl::Break(d) => return Ok(Ctl::Break(d - 1)),
                    c => return Ok(c),
                }
            },
            P::Break(d) => return Ok(Ctl::Break(*d)),
            P::Ret(a) => return Ok(Ctl::Ret(self.atom(a, fr, true)?)),
            P::Halt(a) => return Err(Flow::Halt(self.atom(a, fr, true)?)),
            P::Out(a) => {
                let v = self.atom(a, fr, false)?;
                self.out.extend_from_slice(v.byt()?);
            }
        }
        Ok(Ctl::Next)
    }

    fn prim(&mut self, p: &str, ty: &[Ty], args: &[Atom], fr: &mut Frame) -> Result<Vec<Val>, Flow> {
        // the primitives that read an owned operand without consuming it
        // (W3) look at it in place
        if matches!(p, "list.len" | "arr.len" | "list.get" | "arr.get" | "copy" | "sum.tag" | "tup.get") {
            let k = match args.get(1) {
                Some(x) => Some(self.atom(x, fr, false)?.int()?),
                None => None,
            };
            let get = |v: &Val| -> Result<Val, Flow> {
                Ok(match (p, v) {
                    ("copy", v) => v.clone(),
                    ("list.len" | "arr.len", Val::List(xs) | Val::Arr(xs)) => Val::Int(xs.len() as i128),
                    ("list.get" | "arr.get", Val::List(xs) | Val::Arr(xs)) => {
                        let k = k.unwrap_or(-1);
                        if k < 0 || k as usize >= xs.len() {
                            return Err(bad(format!("{p} applied outside its domain: index")));
                        }
                        xs[k as usize].clone()
                    }
                    ("tup.get", Val::Tup(xs)) => xs.get(k.unwrap_or(-1) as usize).cloned().ok_or_else(|| bad("tup.get index"))?,
                    ("sum.tag", Val::Sum(t, _)) => Val::Int(*t as i128),
                    (_, o) => return Err(bad(format!("{p} of {o:?}"))),
                })
            };
            let r = match &args[0] {
                Atom::V(i) => match fr.locals.get(*i as usize) {
                    Some(Some(v)) => get(v)?,
                    _ => return Err(bad(format!("v{i} was used after it was consumed"))),
                },
                Atom::G(i) => match self.globals.get(*i as usize) {
                    Some(Some(v)) => get(v)?,
                    _ => return Err(bad(format!("g{i} was read before it was set"))),
                },
                Atom::K(_, l) => get(&lit_val(l))?,
            };
            return Ok(vec![r]);
        }
        let v = self.atoms(args, fr, &[])?;
        match p {
            "io.byte" => {
                let b = v.first().ok_or_else(|| bad("io.byte: missing operand"))?.int()?;
                if !(0..=255).contains(&b) {
                    return Err(bad("io.byte applied outside its domain"));
                }
                self.out.push(b as u8);
                Ok(vec![])
            }
            "mem.get" => {
                let k = v.first().ok_or_else(|| bad("mem.get: missing operand"))?.int()?;
                if k < 0 {
                    return Err(bad("mem.get applied outside its domain"));
                }
                Ok(vec![Val::Int(self.mem.get(k as usize).copied().unwrap_or(0))])
            }
            "mem.set" => {
                let k = v.first().ok_or_else(|| bad("mem.set: missing operand"))?.int()?;
                let x = v.get(1).ok_or_else(|| bad("mem.set: missing operand"))?.int()?;
                if !(0..=1 << 32).contains(&k) {
                    return Err(bad("mem.set applied outside its domain"));
                }
                if self.mem.len() <= k as usize {
                    self.mem.resize(k as usize + 1, 0);
                }
                self.mem[k as usize] = x;
                Ok(vec![])
            }
            _ => prim(p, ty, v),
        }
    }
}

fn int_ty(ty: &[Ty], k: usize) -> Result<(i128, i128), Flow> {
    match ty.get(k) {
        Some(Ty::Int(lo, hi)) => Ok((*lo, *hi)),
        o => Err(bad(format!("expected an integer type argument, found {o:?}"))),
    }
}

fn inr(v: i128, (lo, hi): (i128, i128)) -> bool {
    lo <= v && v <= hi
}

/// Exact a * b when it fits i128 (it always does for |a|, |b| <= 2^64 except
/// near 2^128); `None` means the exact product lies outside every core type.
fn mul(a: i128, b: i128) -> Option<i128> {
    a.checked_mul(b)
}

fn wrap(v: Option<i128>, a: i128, b: i128, op: char, (lo, hi): (i128, i128)) -> i128 {
    let w = (hi - lo + 1) as u128;
    match v {
        Some(v) => (v - lo).rem_euclid(w as i128) + lo,
        // the exact product does not fit i128: reduce with 256-bit arithmetic
        None => {
            debug_assert_eq!(op, '*');
            let neg = (a < 0) != (b < 0);
            let (x, y) = (a.unsigned_abs(), b.unsigned_abs());
            // x * y mod w, by halves (x, y < 2^65, w <= 2^65)
            let r = mulmod(x % w, y % w, w);
            let r = if neg { (w - r) % w } else { r } as i128;
            (r - lo).rem_euclid(w as i128) + lo
        }
    }
}

fn mulmod(a: u128, b: u128, m: u128) -> u128 {
    // a, b < m <= 2^65: double and add
    let (mut r, mut a, mut b) = (0u128, a % m, b);
    while b > 0 {
        if b & 1 == 1 {
            r = (r + a) % m;
        }
        a = (a << 1) % m;
        b >>= 1;
    }
    r
}

fn clamp(v: Option<i128>, a: i128, b: i128, (lo, hi): (i128, i128)) -> i128 {
    match v {
        Some(v) => v.clamp(lo, hi),
        None => {
            if (a < 0) != (b < 0) {
                lo
            } else {
                hi
            }
        }
    }
}

/// trunc(a / b) and a - b * trunc(a / b); b != 0
fn tdiv(a: i128, b: i128) -> (i128, i128) {
    (a / b, a % b)
}

fn f32_round(x: f64) -> f64 {
    (x as f32) as f64
}

/// One primitive on values. The domain of each partial primitive is the
/// complement of its `bad.` test; an application outside it is an internal
/// error.
pub fn prim(p: &str, ty: &[Ty], mut v: Vec<Val>) -> Result<Vec<Val>, Flow> {
    let one = |x: Val| Ok(vec![x]);
    let a = |k: usize, v: &Vec<Val>| v.get(k).cloned().ok_or_else(|| bad(format!("{p}: missing operand {k}")));
    let i = |k: usize, v: &Vec<Val>| -> Result<i128, Flow> { a(k, v)?.int() };
    let f = |k: usize, v: &Vec<Val>| -> Result<f64, Flow> { a(k, v)?.float() };
    let dom = |m: &str| Err(bad(format!("{p} applied outside its domain: {m}")));
    match p {
        "bool.not" => one(Val::Bool(!a(0, &v)?.boolean()?)),
        "bool.and" => one(Val::Bool(a(0, &v)?.boolean()? & a(1, &v)?.boolean()?)),
        "bool.or" => one(Val::Bool(a(0, &v)?.boolean()? | a(1, &v)?.boolean()?)),
        "bool.eq" => one(Val::Bool(a(0, &v)?.boolean()? == a(1, &v)?.boolean()?)),
        _ if p.starts_with("int.") => {
            let t = int_ty(ty, 0)?;
            let (lo, hi) = t;
            let x = i(0, &v)?;
            let y = || i(1, &v);
            let exact = |op: &str, x: i128, y: i128| -> Option<i128> {
                match op {
                    "add" => x.checked_add(y),
                    "sub" => x.checked_sub(y),
                    "mul" => mul(x, y),
                    _ => None,
                }
            };
            match &p[4..] {
                "add" | "sub" | "mul" => {
                    let y = y()?;
                    match exact(&p[4..], x, y) {
                        Some(r) if inr(r, t) => one(Val::Int(r)),
                        _ => dom("the exact result is outside the type"),
                    }
                }
                "bad.add" | "bad.sub" | "bad.mul" => {
                    let y = y()?;
                    one(Val::Bool(!matches!(exact(&p[8..], x, y), Some(r) if inr(r, t))))
                }
                "neg" => if inr(-x, t) { one(Val::Int(-x)) } else { dom("-a is outside the type") },
                "bad.neg" => one(Val::Bool(!inr(-x, t))),
                "abs" => if inr(x.abs(), t) { one(Val::Int(x.abs())) } else { dom("|a| is outside the type") },
                "bad.abs" => one(Val::Bool(!inr(x.abs(), t))),
                "quo" | "rem" => {
                    let y = y()?;
                    if y == 0 {
                        return dom("division by zero");
                    }
                    let (q, r) = tdiv(x, y);
                    if !inr(q, t) {
                        return dom("the quotient is outside the type");
                    }
                    if p == "int.quo" { one(Val::Int(q)) } else if inr(r, t) { one(Val::Int(r)) } else { dom("the remainder is outside the type") }
                }
                "bad.quo" => {
                    let y = y()?;
                    one(Val::Bool(y != 0 && !inr(tdiv(x, y).0, t)))
                }
                "eq" => one(Val::Bool(x == y()?)),
                "lt" => one(Val::Bool(x < y()?)),
                "le" => one(Val::Bool(x <= y()?)),
                "min" => one(Val::Int(x.min(y()?))),
                "max" => one(Val::Int(x.max(y()?))),
                "wadd" | "wsub" | "wmul" => {
                    let y = y()?;
                    let op = match &p[5..] { "add" => '+', "sub" => '-', _ => '*' };
                    let e = exact(match op { '+' => "add", '-' => "sub", _ => "mul" }, x, y);
                    one(Val::Int(wrap(e, x, y, op, t)))
                }
                "sadd" | "ssub" | "smul" => {
                    let y = y()?;
                    let e = exact(match &p[5..] { "add" => "add", "sub" => "sub", _ => "mul" }, x, y);
                    one(Val::Int(clamp(e, x, y, t)))
                }
                "and" => one(Val::Int(x & y()?)),
                "or" => one(Val::Int(x | y()?)),
                "xor" => one(Val::Int(x ^ y()?)),
                "not" => one(Val::Int(if lo == 0 { hi - x } else { !x })),
                "shl" | "shr" | "bad.shift" => {
                    let k = y()?;
                    let bad_shift = k < 0 || k >= bits(lo, hi) as i128;
                    if p == "int.bad.shift" {
                        return one(Val::Bool(bad_shift));
                    }
                    if bad_shift {
                        return dom("shift amount");
                    }
                    if p == "int.shr" {
                        let r = x >> k;
                        return if inr(r, t) { one(Val::Int(r)) } else { dom("the result is outside the type") };
                    }
                    // a * 2^k wrapped into the type (|x| < 2^65, k < 65)
                    let w = (hi - lo + 1) as u128;
                    let r = mulmod(x.rem_euclid(w as i128) as u128, (1u128 << k as u32) % w, w);
                    one(Val::Int((r as i128 - lo).rem_euclid(w as i128) + lo))
                }
                "cast" => if inr(x, t) { one(Val::Int(x)) } else { dom("the value is outside the type") },
                "bad.cast" => one(Val::Bool(!inr(x, t))),
                _ => Err(bad(format!("unknown primitive {p}"))),
            }
        }
        _ if p.starts_with("f64.") || p.starts_with("f32.") => {
            let is32 = p.starts_with("f32.");
            let r32 = |x: f64| if is32 { f32_round(x) } else { x };
            let op2 = |n: &str, x: f64, y: f64| -> f64 {
                r32(match n {
                    "add" => x + y,
                    "sub" => x - y,
                    "mul" => x * y,
                    _ => x / y,
                })
            };
            match &p[4..] {
                "add" | "sub" | "mul" | "div" => {
                    let r = op2(&p[4..], f(0, &v)?, f(1, &v)?);
                    if r.is_finite() { one(Val::F(r)) } else { dom("the result is not finite") }
                }
                "bad.add" | "bad.sub" | "bad.mul" | "bad.div" => one(Val::Bool(!op2(&p[8..], f(0, &v)?, f(1, &v)?).is_finite())),
                "sqrt" => {
                    let x = f(0, &v)?;
                    if x < 0.0 { dom("sqrt of a negative") } else { one(Val::F(x.sqrt())) }
                }
                "bad.sqrt" => one(Val::Bool(f(0, &v)? < 0.0)),
                "sign" => one(Val::Bool(f(0, &v)?.is_sign_negative())),
                "neg" => one(Val::F(-f(0, &v)?)),
                "abs" => one(Val::F(f(0, &v)?.abs())),
                "floor" => one(Val::F(f(0, &v)?.floor())),
                "ceil" => one(Val::F(f(0, &v)?.ceil())),
                "round" => one(Val::F(f(0, &v)?.round())),
                "min" => {
                    let (x, y) = (f(0, &v)?, f(1, &v)?);
                    one(Val::F(if x <= y { x } else { y }))
                }
                "max" => {
                    let (x, y) = (f(0, &v)?, f(1, &v)?);
                    one(Val::F(if x >= y { x } else { y }))
                }
                "eq" => one(Val::Bool(f(0, &v)? == f(1, &v)?)),
                "lt" => one(Val::Bool(f(0, &v)? < f(1, &v)?)),
                "le" => one(Val::Bool(f(0, &v)? <= f(1, &v)?)),
                "of_int" => one(Val::F(i(0, &v)? as f64)),
                "trunc" | "bad.trunc" => {
                    let t = int_ty(ty, 0)?;
                    let x = f(0, &v)?.trunc();
                    let ok = x.is_finite() && x.abs() < 1.8e19 && inr(x as i128, t);
                    if p == "f64.bad.trunc" {
                        one(Val::Bool(!ok))
                    } else if ok {
                        one(Val::Int(x as i128))
                    } else {
                        dom("the truncation is outside the type")
                    }
                }
                "to_f32" => {
                    let r = f32_round(f(0, &v)?);
                    if r.is_finite() { one(Val::F(r)) } else { dom("not finite in binary32") }
                }
                "bad.to_f32" => one(Val::Bool(!f32_round(f(0, &v)?).is_finite())),
                "to_f64" => one(Val::F(f(0, &v)?)),
                _ => Err(bad(format!("unknown primitive {p}"))),
            }
        }
        "bytes.len" => one(Val::Int(a(0, &v)?.byt()?.len() as i128)),
        "bytes.get" => {
            let b = a(0, &v)?;
            let b = b.byt()?;
            let k = i(1, &v)?;
            if k < 0 || k as usize >= b.len() {
                return dom("index");
            }
            one(Val::Int(b[k as usize] as i128))
        }
        "bytes.slice" => {
            let b = a(0, &v)?;
            let b = b.byt()?;
            let (x, y) = (i(1, &v)?, i(2, &v)?);
            if x < 0 || x > y || y as usize > b.len() {
                return dom("slice bounds");
            }
            one(Val::bytes(&b[x as usize..y as usize]))
        }
        "bytes.cat" => {
            let (x, y) = (a(0, &v)?, a(1, &v)?);
            let mut c = x.byt()?.to_vec();
            c.extend_from_slice(y.byt()?);
            one(Val::Bytes(Rc::from(c)))
        }
        "bytes.eq" => one(Val::Bool(a(0, &v)?.byt()? == a(1, &v)?.byt()?)),
        "bytes.lt" => one(Val::Bool(a(0, &v)?.byt()?[..] < a(1, &v)?.byt()?[..])),
        "bytes.of_list" => match v.pop() {
            Some(Val::List(xs)) => {
                let mut b = Vec::with_capacity(xs.len());
                for x in xs {
                    let x = x.int()?;
                    if !(0..=255).contains(&x) {
                        return dom("a byte outside 0..255");
                    }
                    b.push(x as u8);
                }
                one(Val::Bytes(Rc::from(b)))
            }
            o => Err(bad(format!("bytes.of_list of {o:?}"))),
        },
        "bytes.to_list" => one(Val::List(a(0, &v)?.byt()?.iter().map(|b| Val::Int(*b as i128)).collect())),
        "arr.new" => {
            let n = i(0, &v)?;
            let x = a(1, &v)?;
            if n < 0 {
                return dom("negative length");
            }
            one(Val::Arr(vec![x; n as usize]))
        }
        "arr.len" | "list.len" => match a(0, &v)? {
            Val::Arr(xs) | Val::List(xs) => one(Val::Int(xs.len() as i128)),
            o => Err(bad(format!("{p} of {o:?}"))),
        },
        "arr.get" | "list.get" => {
            let k = i(1, &v)?;
            match &v[0] {
                Val::Arr(xs) | Val::List(xs) => {
                    if k < 0 || k as usize >= xs.len() {
                        return dom("index");
                    }
                    one(xs[k as usize].clone())
                }
                o => Err(bad(format!("{p} of {o:?}"))),
            }
        }
        "arr.set" | "list.set" => {
            let x = v.pop().ok_or_else(|| bad("set: missing value"))?;
            let k = i(1, &v)?;
            match v.swap_remove(0) {
                Val::Arr(mut xs) | Val::List(mut xs) => {
                    if k < 0 || k as usize >= xs.len() {
                        return dom("index");
                    }
                    xs[k as usize] = x;
                    one(if p == "arr.set" { Val::Arr(xs) } else { Val::List(xs) })
                }
                o => Err(bad(format!("{p} of {o:?}"))),
            }
        }
        "list.new" => one(Val::List(vec![])),
        "list.push" => {
            let x = v.pop().ok_or_else(|| bad("push: missing value"))?;
            match v.pop() {
                Some(Val::List(mut xs)) => {
                    xs.push(x);
                    one(Val::List(xs))
                }
                o => Err(bad(format!("list.push of {o:?}"))),
            }
        }
        "list.pop" => match v.pop() {
            Some(Val::List(mut xs)) => match xs.pop() {
                Some(x) => Ok(vec![Val::List(xs), x]),
                None => dom("pop of an empty list"),
            },
            o => Err(bad(format!("list.pop of {o:?}"))),
        },
        "tup.mk" => one(Val::Tup(v)),
        "tup.get" => {
            let k = i(1, &v)?;
            match &v[0] {
                Val::Tup(xs) => xs.get(k as usize).cloned().map(|x| vec![x]).ok_or_else(|| bad("tup.get index")),
                o => Err(bad(format!("tup.get of {o:?}"))),
            }
        }
        "tup.open" => match v.pop() {
            Some(Val::Tup(xs)) => Ok(xs),
            o => Err(bad(format!("tup.open of {o:?}"))),
        },
        "sum.mk" => {
            let k = i(0, &v)?;
            let x = v.pop().ok_or_else(|| bad("sum.mk: missing payload"))?;
            one(Val::Sum(k as u32, Box::new(x)))
        }
        "sum.tag" => match &v[0] {
            Val::Sum(k, _) => one(Val::Int(*k as i128)),
            o => Err(bad(format!("sum.tag of {o:?}"))),
        },
        "sum.get" => {
            let k = i(1, &v)?;
            match v.swap_remove(0) {
                Val::Sum(t, x) if t as i128 == k => one(*x),
                Val::Sum(..) => dom("the tag differs"),
                o => Err(bad(format!("sum.get of {o:?}"))),
            }
        }
        "drop" => Ok(vec![]),
        _ => Err(bad(format!("unknown primitive {p}"))),
    }
}
