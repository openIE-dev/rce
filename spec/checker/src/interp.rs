//! Reference evaluator for admission (spec 6.2–6.4, 6.9; rules W-INT-1, W-CT-1).
//! Evaluates consts, state initializers, `check`s, comptime blocks and the
//! examples of every intent, following the edge semantics of 6.3: integer
//! results are checked against their type's range (trap), division by zero
//! traps, loops trap at their bound, and a step whose event does not apply
//! keeps the whole state (atomically).
use crate::ast::*;
use crate::types::{INTS, int_range};
use std::collections::HashMap;

/// A shared, copy-on-write value: cloning a text or list value (reading a
/// variable, passing an argument) is O(1); a write copies only when the
/// buffer is shared. Evaluation and fuel are unchanged.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Shared<T: Hollow>(pub std::sync::Arc<T>);
/// What a shared buffer holds can be moved out, leaving an empty one behind.
/// Dropping the last handle to a deep value would otherwise recurse as deep
/// as the value on the host's stack (an abort, not the defined `evaluation
/// depth` trap of 6.7); `Shared`'s drop moves the contents out and drops
/// them on a stack grown on the heap, as `eval` does.
pub trait Hollow {
    fn hollow(&mut self) -> Self;
}
impl Hollow for String {
    fn hollow(&mut self) -> Self {
        std::mem::take(self)
    }
}
impl<T> Hollow for Vec<T> {
    fn hollow(&mut self) -> Self {
        std::mem::take(self)
    }
}
impl Hollow for V {
    fn hollow(&mut self) -> Self {
        std::mem::replace(self, V::Unit)
    }
}
impl<T: Hollow> Drop for Shared<T> {
    fn drop(&mut self) {
        if let Some(t) = std::sync::Arc::get_mut(&mut self.0) {
            let t = t.hollow();
            stacker::maybe_grow(64 * 1024, 1024 * 1024, move || drop(t));
        }
    }
}
impl<T: Hollow + std::fmt::Debug> std::fmt::Debug for Shared<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        (*self.0).fmt(f)
    }
}
impl<T: Hollow> std::ops::Deref for Shared<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}
impl<T: Hollow + Clone> std::ops::DerefMut for Shared<T> {
    fn deref_mut(&mut self) -> &mut T {
        std::sync::Arc::make_mut(&mut self.0)
    }
}
impl<T: Hollow> From<T> for Shared<T> {
    fn from(t: T) -> Self {
        Shared(std::sync::Arc::new(t))
    }
}
impl From<&str> for Shared<String> {
    fn from(t: &str) -> Self {
        Shared(std::sync::Arc::new(t.to_string()))
    }
}
impl<T: Hollow + Clone> Shared<T> {
    pub fn take(mut self) -> T {
        match std::sync::Arc::get_mut(&mut self.0) {
            Some(t) => t.hollow(),
            None => (*self.0).clone(),
        }
    }
}
impl std::fmt::Display for Shared<String> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl<T> std::iter::FromIterator<T> for Shared<Vec<T>> {
    fn from_iter<I: IntoIterator<Item = T>>(i: I) -> Self {
        Shared::from(i.into_iter().collect::<Vec<T>>())
    }
}
impl<'a, T> IntoIterator for &'a Shared<Vec<T>> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
impl<T: Clone> IntoIterator for Shared<Vec<T>> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.take().into_iter()
    }
}
impl std::iter::FromIterator<char> for Shared<String> {
    fn from_iter<I: IntoIterator<Item = char>>(i: I) -> Self {
        Shared::from(i.into_iter().collect::<String>())
    }
}
impl std::iter::FromIterator<String> for Shared<String> {
    fn from_iter<I: IntoIterator<Item = String>>(i: I) -> Self {
        Shared::from(i.into_iter().collect::<String>())
    }
}
impl std::ops::Add<&Shared<String>> for Shared<String> {
    type Output = Shared<String>;
    fn add(self, o: &Shared<String>) -> Shared<String> {
        let mut s = self.take();
        s.push_str(o);
        Shared::from(s)
    }
}
impl std::borrow::Borrow<str> for Shared<String> {
    fn borrow(&self) -> &str {
        &self.0
    }
}

/// numbers the index temporaries of `let &mut` borrows (crate::borrow)
static LB_TEMPS: std::sync::Mutex<usize> = std::sync::Mutex::new(0);

/// A value. Every compound value holds its parts in a `Shared` buffer, so
/// cloning one (a move, a read, an argument) is O(1) whatever its depth, and
/// a write copies only the level it changes.
#[derive(Debug, Clone)]
pub enum V {
    Int(i128, &'static str), // "lit" = an integer literal not yet given a type
    F(f64),
    Bool(bool),
    Text(Shared<String>),
    Unit,
    Tuple(Shared<Vec<V>>),
    List(Shared<Vec<V>>),
    /// a Set[τ]: canonical order, no duplicates (7.1); shown and embedded as a list
    Set(Shared<Vec<V>>),
    Map(Shared<Vec<(V, V)>>),
    Opt(Option<Shared<V>>),
    Rec(String, Shared<Vec<(String, V)>>), // struct, kind variant, event, error
    Err(Shared<V>),                        // the error side of an error union
    Fn(String),
    Closure(Vec<Param>, Box<Expr>, Vec<HashMap<String, V>>),
    Range(i128, i128, &'static str),
    F32(f64),   // an f32 value, held exactly in an f64
    Gen(usize), // a suspended iterator call (index into Interp::gens)
    // draft 0.6 / 0.6b runtime values (Appendix C, book/14-draft06-concurrency.md)
    Chan(String, i128),                        // channel id (6.6 content hash), capacity
    Actor(String, String),                     // actor id, actor type
    Task(String),                              // join handle of a spawned task (C.11.9)
    Session(String, String, i128),             // endpoint: send channel, receive channel, capacity
    LVar(std::sync::Arc<crate::conc::LCell>),  // lattice cell (C.11.8)
}
impl PartialEq for V {
    fn eq(&self, o: &V) -> bool {
        // structural, as deep as the values: grow the stack as `eval` does
        stacker::maybe_grow(256 * 1024, 8 * 1024 * 1024, || v_eq(self, o))
    }
}
fn v_eq(a0: &V, o: &V) -> bool {
    {
        use V::*;
        match (a0, o) {
            (Int(a, _), Int(b, _)) => a == b,
            (F(a), F(b)) => a == b,
            (F32(a), F32(b)) => a == b,
            (Bool(a), Bool(b)) => a == b,
            (Text(a), Text(b)) => a == b,
            (Unit, Unit) => true,
            (Tuple(a), Tuple(b)) | (List(a), List(b)) => a == b,
            // sets are kept in canonical order, so identity is elementwise key identity
            (Set(a), Set(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| key_eq(x, y)),
            (Map(a), Map(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .all(|(k, v)| b.iter().any(|(k2, v2)| key_eq(k, k2) && v == v2))
            }
            (Opt(a), Opt(b)) => a == b,
            (Rec(n, a), Rec(m, b)) => n == m && a == b,
            (Err(a), Err(b)) => a == b,
            (Chan(a, _), Chan(b, _)) | (Task(a), Task(b)) => a == b,
            (Actor(a, _), Actor(b, _)) => a == b,
            (Session(a, b, _), Session(c, d, _)) => a == c && b == d,
            (LVar(a), LVar(b)) => std::sync::Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}
impl V {
    pub fn show(&self) -> String {
        match self {
            V::Int(v, _) => v.to_string(),
            V::F(f) => float_text(*f),
            V::Bool(b) => b.to_string(),
            V::Text(t) => show_text(t),
            V::Unit => "()".into(),
            V::Tuple(v) => format!(
                "({})",
                v.iter().map(|x| x.show()).collect::<Vec<_>>().join(", ")
            ),
            V::List(v) | V::Set(v) => format!(
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
            V::Chan(..) => "<chan>".into(),
            V::Actor(_, t) => format!("<actor {t}>"),
            V::Task(_) => "<task>".into(),
            V::Session(..) => "<session>".into(),
            V::LVar(c) => format!("lvar({})", c.peek().show()),
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
    /// Cancellation of a `with deadline` frame or of a whole task (C.6.2, C.11.9).
    Cancel(u64),
}
pub type R<T> = Result<T, Flow>;
pub(crate) fn trap<T>(m: impl Into<String>) -> R<T> {
    Err(Flow::Trap(m.into()))
}

pub const LOOP_LIMIT: i128 = 1_000_000;
/// N9: a list or text built by any operation has at most SIZE_LIMIT elements
/// (list items, text code points).
pub const SIZE_LIMIT: usize = 1_000_000;

/// N9 at the one point every built value passes: the result of an expression.
pub fn size_ok(v: V) -> R<V> {
    match &v {
        V::List(xs) if xs.len() > SIZE_LIMIT => trap("size limit"),
        V::Text(t) if t.len() > SIZE_LIMIT && t.chars().count() > SIZE_LIMIT => trap("size limit"),
        _ => Ok(v),
    }
}
pub const FUEL: u64 = 50_000_000;
/// The trap text of fuel exhaustion (6.7, E1).
pub const TRAP_FUEL: &str = "comptime fuel exhausted";

/// Why δ kept the state on an input (T8), as the host reports it (HC-21): a function
/// of the state and the input that names no declaration, so it is the same
/// under the renamings of 10.6 (T7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cause {
    /// a declared error left the step (`fail`, `try`)
    Fail,
    /// any other trap of 6.3 or 6.7
    Trap,
    /// fuel ran out (6.7, E1)
    Fuel,
    /// N9 SIZE_LIMIT
    Size,
    /// N9 VALUE_LIMIT
    Value,
    /// the evaluation depth of 6.7
    Depth,
}

impl Cause {
    pub fn as_str(self) -> &'static str {
        match self {
            Cause::Fail => "fail",
            Cause::Trap => "trap",
            Cause::Fuel => "fuel",
            Cause::Size => "size",
            Cause::Value => "value",
            Cause::Depth => "depth",
        }
    }
    pub fn parse(s: &str) -> Option<Cause> {
        [Cause::Fail, Cause::Trap, Cause::Fuel, Cause::Size, Cause::Value, Cause::Depth].into_iter().find(|c| c.as_str() == s)
    }
    /// The cause of a step left by `f`.
    pub fn of(f: &Flow) -> Cause {
        match f {
            Flow::Fail(_) => Cause::Fail,
            Flow::Trap(t) if t == TRAP_FUEL => Cause::Fuel,
            Flow::Trap(t) if t == "size limit" => Cause::Size,
            Flow::Trap(t) if t == "value limit" => Cause::Value,
            Flow::Trap(t) if t == "evaluation depth" => Cause::Depth,
            _ => Cause::Trap,
        }
    }
}

/// A kept outcome of δ (T8): its cause and the text the tools print (the
/// declared error's name, or `trap(..)`).
#[derive(Clone, Debug, PartialEq)]
pub struct KeptReason {
    pub cause: Cause,
    pub text: String,
}

impl std::fmt::Display for KeptReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text)
    }
}
/// The fuel each new evaluator starts with (6.7). It is FUEL; the tests of
/// tests/fuel.rs set it to find the trap step of small programs, and an
/// emitted program takes the same number from its `fuel N` words (10.5).
pub static FUEL_LIMIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(FUEL);
/// The nesting limit of expression evaluation (6.7 addendum, host
/// independence): the evaluation that would be the (EVAL_DEPTH + 1)-th
/// nested one traps with `evaluation depth`, on every host. The interpreter
/// grows its stack on the heap where it can (`stacker`), and the wasm32
/// build reserves a stack sized for this limit (`.cargo/config.toml`), so a
/// deep program meets this defined trap, never the host's stack size.
pub const EVAL_DEPTH: u32 = 20_000;
pub fn fuel_limit() -> u64 {
    FUEL_LIMIT.load(std::sync::atomic::Ordering::Relaxed)
}

pub struct Interp<'a> {
    pub m: &'a Module,
    pub(crate) fns: HashMap<String, &'a FnDecl>,
    pub(crate) methods: HashMap<(String, String), &'a FnDecl>,
    pub(crate) types: HashMap<String, &'a Item>,
    pub(crate) variants: HashMap<String, Vec<(String, Ty)>>,
    pub(crate) consts: HashMap<String, V>,
    pub scopes: Vec<HashMap<String, V>>,
    pub(crate) fuel: u64,
    pub outputs: Vec<V>,
    pub loop_limit: i128,
    pub(crate) gens: Vec<(&'a FnDecl, HashMap<String, V>)>,
    pub(crate) yields: Vec<(Vec<V>, usize)>,
    /// Concurrency state of draft 0.6 / 0.6b (see conc.rs).
    pub cx: crate::conc::TaskCx,
    /// active calls of each function declared with `bound n` (6.7), keyed
    /// by the declaration
    pub(crate) depth: HashMap<usize, u64>,
    /// nested `eval` calls in progress (bounded by EVAL_DEPTH)
    pub(crate) eval_depth: u32,
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
    /// Give the next input a fresh fuel budget: each input is one run of δ
    /// (HC-7a), as `form_json` and the host do before every event.
    pub fn refuel(&mut self) {
        self.fuel = fuel_limit();
    }

    pub fn new(m: &'a Module) -> Self {
        let mut it = Interp {
            m,
            fns: HashMap::new(),
            methods: HashMap::new(),
            types: HashMap::new(),
            variants: HashMap::new(),
            consts: HashMap::new(),
            scopes: vec![],
            fuel: fuel_limit(),
            outputs: vec![],
            loop_limit: LOOP_LIMIT,
            gens: vec![],
            yields: vec![],
            cx: Default::default(),
            depth: HashMap::new(),
            eval_depth: 0,
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
    pub(crate) fn alias(&self, t: &Ty) -> Ty {
        // 5.16 indexed types are erased: their values are those of the base
        if let Ty::Path(p, a) = t
            && p.len() == 1
            && !self.types.contains_key(&p[0])
        {
            match (p[0].as_str(), a.as_slice()) {
                ("Guarded", [x, _]) | ("Secret", [x]) => return self.alias(x),
                ("Text", [_]) => return Ty::Path(vec!["text".into()], vec![]),
                ("Cap", [_]) => return Ty::Unit,
                _ => {}
            }
        }
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
        // A refined type (5.1, W-REF-1) is checked where its base type is:
        // the value fits the base, then must satisfy the predicate.
        if let Ty::Path(p, _) = t
            && p.len() == 1
            && let Some(Item::TypeAlias { ty, pred: Some(e), .. }) = self.types.get(&p[0])
        {
            let v = self.fit(v, ty)?;
            if let Some(b) = crate::refine::base_of(&self.alias(ty))
                && let Ok(pr) = crate::refine::compile(e, b)
                && !crate::refine::holds(&pr, &v)
            {
                return trap(crate::refine::trap_text(&p[0]));
            }
            return Ok(v);
        }
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
            (V::Opt(Some(x)), Ty::Opt(i)) => Ok(V::Opt(Some(Shared::from(self.fit(x.take(), i)?)))),
            (V::List(xs), Ty::Path(p, a)) if p.len() == 1 && p[0] == "List" && a.len() == 1 => {
                Ok(V::List(
                    xs.into_iter()
                        .map(|x| self.fit(x, &a[0]))
                        .collect::<R<_>>()?,
                ))
            }
            (V::Set(xs), Ty::Path(p, a)) if p.len() == 1 && p[0] == "Set" && a.len() == 1 => Ok(
                crate::conc::set_of(xs.into_iter().map(|x| self.fit(x, &a[0])).collect::<R<_>>()?),
            ),
            (V::List(xs), Ty::Array(e, _)) => Ok(V::List(
                xs.into_iter().map(|x| self.fit(x, e)).collect::<R<_>>()?,
            )),
            (V::Map(xs), Ty::Path(p, a)) if p.len() == 1 && p[0] == "Map" && a.len() == 2 => {
                let mut m = vec![];
                for (k, v) in xs {
                    map_set(&mut m, self.fit(k, &a[0])?, self.fit(v, &a[1])?);
                }
                Ok(V::Map(m.into()))
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
        fuel_limit().saturating_sub(self.fuel)
    }
    pub(crate) fn tick(&mut self) -> R<()> {
        if self.fuel == 0 {
            return trap(TRAP_FUEL);
        }
        self.fuel -= 1;
        Ok(())
    }
    pub(crate) fn get(&self, n: &str) -> Option<V> {
        self.scopes.iter().rev().find_map(|s| s.get(n).cloned())
    }
    pub(crate) fn set(&mut self, n: &str, v: V) -> bool {
        for s in self.scopes.iter_mut().rev() {
            if let Some(x) = s.get_mut(n) {
                *x = v;
                return true;
            }
        }
        false
    }
    /// Before an in-place update of a value read from a plain variable,
    /// drop the variable's own reference so the update does not copy the
    /// buffer (the variable is assigned right after).
    fn release(&mut self, obj: &Expr) {
        if let Expr::Name(n, _) = obj {
            for s in self.scopes.iter_mut().rev() {
                if let Some(x) = s.get_mut(n.as_str()) {
                    *x = V::Unit;
                    return;
                }
            }
        }
    }
    pub(crate) fn bind(&mut self, n: &str, v: V) {
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

/// Canonical float text (6.3 N3): shortest round-trip digits, decimal
/// notation for 1e-5 <= |x| < 1e16, else `d.ddde+N` / `d.ddde-N`.
/// 6.3 N7: the whitespace of `strip`, `words` and `toint`, enumerated so that
/// no result depends on the Unicode tables of a toolchain: the White_Space
/// property of Unicode 18.0.0 (unchanged since Unicode 6.3), frozen here.
pub const WHITE_SPACE: [char; 25] = ['\t', '\n', '\u{0b}', '\u{0c}', '\r', ' ', '\u{85}', '\u{a0}', '\u{1680}', '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}', '\u{2006}', '\u{2007}', '\u{2008}', '\u{2009}', '\u{200a}', '\u{2028}', '\u{2029}', '\u{202f}', '\u{205f}', '\u{3000}'];
pub fn is_ws(c: char) -> bool {
    WHITE_SPACE.contains(&c)
}

/// 6.3 N7 and one-emitter.md ONE-18: the code points display writes as
/// `\u{h}`, read from `spec/data/display-escape.json`, so that no display
/// depends on the Unicode tables of a toolchain. The list is the set that
/// the Debug form of a str escapes in Rust 1.98.1, whose tables are of
/// Unicode 17.0.0, enumerated once; `tests/unicode_pin.rs` compares it with
/// the toolchain, the runtime written in the core reads the same list, and
/// the emitter writes it into each emitted runtime (`emit::base`).
pub const DISPLAY_ESCAPE_JSON: &str = include_str!("../../data/display-escape.json");

/// The ranges of [`DISPLAY_ESCAPE_JSON`], ascending and disjoint.
pub fn display_escaped() -> &'static [(u32, u32)] {
    static R: std::sync::OnceLock<Vec<(u32, u32)>> = std::sync::OnceLock::new();
    R.get_or_init(|| {
        let d: serde_json::Value = serde_json::from_str(DISPLAY_ESCAPE_JSON).expect("display-escape.json is JSON");
        let cp = |x: &serde_json::Value| u32::from_str_radix(x.as_str().expect("a code point in hexadecimal"), 16).expect("a code point in hexadecimal");
        d["ranges"].as_array().expect("the ranges").iter().map(|r| (cp(&r[0]), cp(&r[1]))).collect()
    })
}

/// Whether display writes code point c as `\u{h}`.
pub fn display_escapes(c: char) -> bool {
    let (cp, rs) = (c as u32, display_escaped());
    let i = rs.partition_point(|(_, hi)| *hi < cp);
    i < rs.len() && rs[i].0 <= cp
}

/// Text as display shows it: in double quotes, with U+0000, TAB, LF, CR,
/// `"` and `\` as `\0 \t \n \r \" \\`, each code point of the display
/// escape set as `\u{h}` (h in lower-case hexadecimal, no leading zeros),
/// and every other code point as itself.
pub fn show_text(t: &str) -> String {
    let mut o = String::with_capacity(t.len() + 2);
    o.push('"');
    for c in t.chars() {
        match c {
            '\0' => o.push_str("\\0"),
            '\t' => o.push_str("\\t"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if display_escapes(c) => o.push_str(&format!("\\u{{{:x}}}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

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

/// 6.3 N3: the one canonical JSON form of a float, used for hashes and for
/// every trace or log written as JSON. Shortest round-trip digits (from
/// `{:e}`, which is shortest-digit, Ryu-style); decimal with at least one
/// fractional digit for 1e-4 <= |x| < 1e16 (`10.0`, `0.0001`), otherwise
/// `d.ddde-N` / `d.dddeN` with no `+` (`1e16`, `5e-324`); `-0.0` keeps its
/// sign. Built here rather than taken from `{:?}` so the form is ours.
pub fn float_json(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let (digits, ex) = shortest_e(x);
    let sign = if x < 0.0 { "-" } else { "" };
    let a = x.abs();
    if (1e-4..1e16).contains(&a) {
        let p = ex + 1; // digits before the point
        let body = if p <= 0 {
            format!("0.{}{}", "0".repeat((-p) as usize), digits)
        } else if p as usize >= digits.len() {
            format!("{}{}.0", digits, "0".repeat(p as usize - digits.len()))
        } else {
            format!("{}.{}", &digits[..p as usize], &digits[p as usize..])
        };
        format!("{sign}{body}")
    } else {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
        format!("{sign}{m}e{ex}")
    }
}

/// Canonical JSON of a serde value with floats in the [`float_json`] form
/// and text through `jstr`, the writer of `FV::json` (keys are already sorted:
/// serde_json's map is ordered by key).
pub fn canon_json(v: &serde_json::Value) -> String {
    use serde_json::Value as J;
    match v {
        J::Number(n) if n.is_f64() => float_json(n.as_f64().unwrap()),
        J::String(t) => jstr(t),
        J::Array(xs) => format!("[{}]", xs.iter().map(canon_json).collect::<Vec<_>>().join(",")),
        J::Object(m) => format!(
            "{{{}}}",
            m.iter().map(|(k, x)| format!("{}:{}", jstr(k), canon_json(x))).collect::<Vec<_>>().join(",")
        ),
        other => other.to_string(),
    }
}

pub fn float_text(x: f64) -> String {
    // F9: the text reads back to the same float, so -0.0 keeps its sign (as f32 does, F6)
    if x == 0.0 {
        return if x.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    let (digits, ex) = shortest_e(x);
    let sign = if x < 0.0 { "-" } else { "" };
    if (1e-5..1e16).contains(&x.abs()) {
        let p = ex + 1;
        let body = if p <= 0 {
            format!("0.{}{}", "0".repeat((-p) as usize), digits)
        } else if p as usize >= digits.len() {
            format!("{}{}", digits, "0".repeat(p as usize - digits.len()))
        } else {
            format!("{}.{}", &digits[..p as usize], &digits[p as usize..])
        };
        return format!("{sign}{body}");
    }
    let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
    if ex >= 0 { format!("{sign}{m}e+{ex}") } else { format!("{sign}{m}e{ex}") }
}

pub(crate) fn chk(x: i128, t: &'static str) -> R<V> {
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
/// `f.trunc()` checked into `n`; the trap names the exact integer value of
/// the float (N2/N4), which can exceed i128 (up to about 1.8e308).
pub(crate) fn trunc_chk(f: f64, n: &'static str) -> R<V> {
    let t = f.trunc();
    if t.abs() < 1.7014118346046923e38 {
        return chk(t as i128, n);
    }
    trap(format!("overflow: {} is outside {n}", exact_int_text(t)))
}

/// Decimal text of an integral finite f64, exact.
pub fn exact_int_text(t: f64) -> String {
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
                            // i64.MIN % -1 traps like i64.MIN / -1 (6.3 N2,
                            // Rust checked_rem): the quotient is out of range.
                            chk(tdiv(x, y), ty)?;
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
    pub(crate) fn cmp(&self, a: &V, b: &V) -> R<std::cmp::Ordering> {
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
    pub(crate) fn truthy(&self, v: V) -> R<bool> {
        match v {
            V::Bool(b) => Ok(b),
            o => trap(format!("expected bool, found {}", o.show())),
        }
    }

    /// Every recursion of the evaluator (nested expressions, calls up to a
    /// declared `bound`, 6.7) passes through here. The engine grows its own
    /// stack on the heap when fewer than 256 KiB remain, so evaluation never
    /// depends on the caller's thread stack or on a setting (HC-14).
    pub fn eval(&mut self, e: &Expr) -> R<V> {
        if self.eval_depth >= EVAL_DEPTH {
            return trap("evaluation depth");
        }
        self.eval_depth += 1;
        let r = stacker::maybe_grow(256 * 1024, 8 * 1024 * 1024, || self.eval_inner(e));
        self.eval_depth -= 1;
        r.and_then(size_ok)
    }

    fn eval_inner(&mut self, e: &Expr) -> R<V> {
        if !matches!(e, Expr::Name(n, _) if crate::borrow::is_temp(n)) {
            self.tick()?;
        }
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
                    return Ok(V::Rec(n.clone(), vec![].into()));
                }
                if let Some(Item::Event { .. } | Item::Error { .. }) = self.types.get(n) {
                    return Ok(V::Rec(n.clone(), vec![].into()));
                }
                self.const_value(n)
            }
            E::SelfRef(_) => self.get("self").ok_or(Flow::Trap("self".into())),
            E::Int(v, _) => Ok(V::Int(*v, "lit")),
            E::Float(f, _) => Ok(V::F(*f)),
            E::Str(s, _) => Ok(V::Text(s.clone().into())),
            E::Bool(b, _) => Ok(V::Bool(*b)),
            E::None(_) => Ok(V::Opt(None)),
            E::Unit(_) => Ok(V::Unit),
            E::Tuple(v, _) => Ok(V::Tuple(v.iter().map(|x| self.eval(x)).collect::<R<_>>()?)),
            E::List(v, _) => {
                self.charge(4 + v.len() as i128)?;
                Ok(V::List(v.iter().map(|x| self.eval(x)).collect::<R<_>>()?))
            }
            E::ListRepeat(x, n, _) => {
                let v = self.eval(x)?;
                match self.eval(n)? {
                    V::Int(k, _) if (0..=10_000_000).contains(&k) => {
                        self.charge(4 + k)?;
                        Ok(V::List(vec![v; k as usize].into()))
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
                self.charge(4 + out.len() as i128)?;
                Ok(V::List(out.into()))
            }
            E::Map(v, _) => {
                self.charge(4 + v.len() as i128)?;
                let mut m: Vec<(V, V)> = vec![];
                for (a, b) in v {
                    let k = self.eval(a)?;
                    let x = self.eval(b)?;
                    map_set(&mut m, k, x);
                }
                Ok(V::Map(m.into()))
            }
            E::ChanNew(t, cap, line) => self.chan_new(t, cap.as_deref(), *line),
            E::Call(f, args, _, sz) => self.call(f, args, *sz),
            E::Index(x, idx, _) => {
                let c = self.eval(x)?;
                let i = self.eval(&idx[0])?;
                self.charge(1)?;
                index(c, i)
            }
            E::Slice(x, a, b, _) => {
                let c = self.eval(x)?;
                self.charge(1)?;
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
                    return Ok(V::Text(t[lo..hi].to_string().into()));
                }
                Ok(V::List((xs[lo as usize..hi as usize].to_vec()).into()))
            }
            E::Member(x, m, _) => {
                let v = self.eval(x)?;
                self.charge(1)?;
                field(&v, m)
            }
            E::TupleMember(x, i, _) => match self.eval(x).and_then(|v| self.charge(1).map(|_| v))? {
                V::Tuple(v) if *i < v.len() => Ok(v[*i].clone()),
                _ => trap("bad tuple index"),
            },
            E::Neg(x, _) => match self.eval(x).and_then(|v| self.charge(1).map(|_| v))? {
                V::Int(v, t) => chk(-v, t),
                V::F(f) => Ok(V::F(-f)),
                V::F32(f) => Ok(V::F32(-f)),
                _ => trap("bad negation"),
            },
            E::BitNot(x, _) => match self.eval(x).and_then(|v| self.charge(1).map(|_| v))? {
                V::Int(v, t) => {
                    let t2 = if t == "lit" { "i64" } else { t };
                    let (lo, hi) = int_range(t2);
                    Ok(V::Int(if lo == 0 { hi - v } else { !v }, t))
                }
                _ => trap("bad ~"),
            },
            E::Not(x, _) => {
                let b = self.eval(x)?;
                self.charge(1)?;
                Ok(V::Bool(!self.truthy(b)?))
            }
            E::Borrow(_, x, _) => self.eval(x),
            E::Recv(c, _) => self.conc_recv(c),
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
                    (V::F32(f), Some(n)) => trunc_chk(f, n),
                    (V::Int(v, _), Some(n)) => chk(v, n),
                    (V::F(f), Some(n)) => {
                        if !f.is_finite() {
                            return trap("non-finite float");
                        }
                        trunc_chk(f, n)
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
                        self.bind(n, err.take());
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
                    None => err.take(),
                })),
                // `try e` without `else` only types for `e : T ! E` (W-TY-1
                // otherwise), so an option here is the ok value of a
                // `T? ! E` result and is returned as it is [E-Try-V].
                v @ V::Opt(_) if els.is_none() => Ok(v),
                V::Opt(None) => match els {
                    Some(e) => {
                        let ev = self.eval(e)?;
                        Err(Flow::Fail(ev))
                    }
                    None => trap("try on none"),
                },
                V::Opt(Some(v)) => Ok(v.take()),
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
                self.charge(1)?;
                let x = self.eval(a)?;
                if !self.truthy(x)? {
                    return Ok(V::Bool(false));
                }
                let y = self.eval(b)?;
                return Ok(V::Bool(self.truthy(y)?));
            }
            "or" => {
                self.charge(1)?;
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
        self.charge(crate::conc::op_cost(op, &x))?;
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

/// A map key as a Form value, for ordering only: records always carry their
/// name, so no module is needed (6.7 erratum E-1: maps are kept in canonical
/// key order, so iteration, keys(), values() and display follow it).
fn key_form(v: &V) -> FV {
    match v {
        V::Rec(n, fs) => {
            let mut f: Vec<(String, FV)> = fs.iter().map(|(k, x)| (k.clone(), key_form(x))).collect();
            f.push(("$v".into(), FV::Text(n.clone())));
            rec(f)
        }
        V::Opt(Some(x)) => key_form(x),
        V::Err(x) => rec(vec![("$err".into(), key_form(x))]),
        V::Tuple(xs) | V::List(xs) | V::Set(xs) => FV::List(xs.iter().map(key_form).collect()),
        V::Map(ps) => FV::List(ps.iter().map(|(k, v)| FV::List(vec![key_form(k), key_form(v)])).collect()),
        V::Int(i, _) => FV::Int(*i),
        V::F(f) | V::F32(f) => FV::Float(*f),
        V::Bool(b) => FV::Bool(*b),
        V::Text(t) => FV::Text(t.to_string()),
        V::Range(a, b, _) => FV::List((*a..*b).map(FV::Int).collect()),
        V::Fn(n) => rec(vec![("$fn".into(), FV::Text(n.clone()))]),
        _ => FV::Null,
    }
}
/// Map key identity (6.1): the canonical order's equality, not IEEE `==`,
/// so `-0.0` and `0.0` (distinct states, N3 F6) are distinct keys.
pub fn key_eq(a: &V, b: &V) -> bool {
    key_form(a).cmp_canon(&key_form(b)) == std::cmp::Ordering::Equal
}
fn map_set(m: &mut Vec<(V, V)>, k: V, v: V) {
    if let Some(e) = m.iter_mut().find(|(a, _)| key_eq(a, &k)) {
        e.1 = v;
    } else {
        let kf = key_form(&k);
        let i = m.partition_point(|(a, _)| key_form(a).cmp_canon(&kf) == std::cmp::Ordering::Less);
        m.insert(i, (k, v));
    }
}
fn contains(c: &V, x: &V) -> R<bool> {
    Ok(match c {
        V::List(v) | V::Tuple(v) => v.contains(x),
        V::Set(v) => v.iter().any(|k| key_eq(k, x)),
        V::Map(m) => m.iter().any(|(k, _)| key_eq(k, x)),
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
            .find(|(a, _)| key_eq(a, &k))
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
    pub(crate) fn iter_items(&mut self, v: V) -> R<Vec<V>> {
        Ok(match v {
            V::List(xs) | V::Set(xs) => xs.take(),
            V::Range(a, b, t) => {
                if b - a > self.loop_limit * 10 {
                    return trap("loop bound");
                }
                (a..b.max(a)).map(|i| V::Int(i, t)).collect()
            }
            V::Text(s) => s.chars().map(|c| V::Text(c.to_string().into())).collect(),
            V::Map(m) => m.into_iter().map(|(k, v)| V::Tuple(vec![k, v].into())).collect(),
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

    pub(crate) fn assign(&mut self, place: &Expr, v: V) -> R<()> {
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

    pub(crate) fn pmatch(&mut self, p: &Pat, v: &V) -> R<bool> {
        Ok(match p {
            Pat::Wild => true,
            // a session offer's label (C.11.3) matches by name and binds nothing
            Pat::Path(path) if matches!(v, V::Rec(l, f) if f.is_empty() && l.starts_with("$label:")) => {
                matches!(v, V::Rec(l, _) if l["$label:".len()..] == path[path.len() - 1])
            }
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
        Ok(Some(V::Rec(n.to_string(), out.into())))
    }

    fn call(&mut self, f: &Expr, args: &[Arg], sized: bool) -> R<V> {
        // R5: `f[T](x)` instantiates generic `f`; types are erased at run time
        if let Expr::Index(g, _, _) = f
            && let Expr::Name(n, _) = &**g
            && self.fns.contains_key(n)
            && self.get(n).is_none()
        {
            return self.call(g, args, sized);
        }
        // 5.16 context text: `C.lit("…")` is its literal, `C.encode(t)` escapes
        if let Expr::Member(o, m, _) = f
            && let Expr::Name(c, _) = &**o
            && crate::index08::is_ctx(c)
            && self.get(c).is_none()
            && !self.types.contains_key(c)
            && c == "Shell"
            && m == "cmd"
            && args.len() == 2
        {
            let p = self.eval(&args[0].e)?;
            let a = self.eval(&args[1].e)?;
            self.charge(crate::conc::C_CALL.saturating_add(crate::conc::size_of(&a)))?;
            return match (p, a) {
                (V::Text(p), V::List(xs)) => {
                    let mut v = vec![];
                    for x in xs.iter() {
                        let V::Text(t) = x else { return trap("context text") };
                        v.push(t.to_string());
                    }
                    Ok(V::Text(crate::index08::shell_cmd(&p, &v).into()))
                }
                _ => trap("context text"),
            };
        }
        if let Expr::Member(o, m, _) = f
            && let Expr::Name(c, _) = &**o
            && crate::index08::is_ctx(c)
            && self.get(c).is_none()
            && !self.types.contains_key(c)
            && args.len() == 1
        {
            let v = self.eval(&args[0].e)?;
            // C.16: `lit` costs c_call, `encode` c_call plus its text (linear class)
            let w = if m == "encode" { crate::conc::size_of(&v) } else { 0 };
            self.charge(crate::conc::C_CALL.saturating_add(w))?;
            return match (m.as_str(), v) {
                ("lit", V::Text(s)) => Ok(V::Text(s)),
                ("encode", V::Text(s)) if c == "Header" && (s.contains('\r') || s.contains('\n')) => trap("Header.encode: a header value contains CR or LF (F.5.7)"),
                ("encode", V::Text(s)) => Ok(V::Text((crate::index08::encode(c, &s)).into())),
                _ => trap("context text"),
            };
        }
        match f {
            Expr::Name(n, _) if self.get(n).is_none() => {
                match n.as_str() {
                    "some" => {
                        let v = self.eval(&args[0].e)?;
                        return Ok(V::Opt(Some(Shared::from(v))));
                    }
                    "copy" | "Box" => return self.eval(&args[0].e),
                    "ct_choose" if args.len() == 3 => {
                        // branchless: both values are evaluated, then one is chosen
                        let c = self.eval(&args[0].e)?;
                        let a = self.eval(&args[1].e)?;
                        let b = self.eval(&args[2].e)?;
                        return Ok(if self.truthy(c)? { a } else { b });
                    }
                    "set_of" => {
                        let v = self.eval(&args[0].e)?;
                        let V::List(xs) = v else { return trap("set_of of a non-list") };
                        return Ok(crate::conc::set_of(xs.take()));
                    }
                    "lvar_new" => {
                        let vals: Vec<V> = args.iter().map(|a| self.eval(&a.e)).collect::<R<_>>()?;
                        return Ok(crate::conc::LCell::new_value(vals));
                    }
                    _ => {}
                }
                if let Some(fd) = self.fns.get(n).copied() {
                    return self.call_fn(fd, args, None);
                }
                if let Some(v) = self.construct(n, args)? {
                    return Ok(v);
                }
                if let Some(Item::Actor { .. }) = self.types.get(n) {
                    return self.actor_new(n, f.line());
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
                    V::Set(_) => Some("Set".to_string()),
                    _ => None,
                };
                if let Some(h) = head
                    && let Some(fd) = self.methods.get(&(h, m.clone())).copied()
                {
                    return self.call_fn(fd, args, Some((recv, obj)));
                }
                self.builtin(obj, recv, m, args, sized)
            }
            _ => {
                let fv = self.eval(f)?;
                self.call_value(fv, args)
            }
        }
    }
    pub(crate) fn kind_of(&self, n: &str) -> String {
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
        self.charge(crate::conc::C_CALL)?;
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
        self.charge(crate::conc::C_CALL)?;
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
        // `bound n` (6.7): the call that would be the (n + 1)-th active call
        // of this function traps; arguments are evaluated and fitted first
        let bkey = fd.sig.bound.map(|n| (fd as *const FnDecl as usize, n));
        if let Some((k, n)) = bkey {
            let d = self.depth.get(&k).copied().unwrap_or(0);
            if d >= n {
                return trap("recursion bound");
            }
            self.depth.insert(k, d + 1);
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
        if let Some((k, _)) = bkey
            && let Some(d) = self.depth.get_mut(&k)
        {
            *d -= 1;
        }
        let fin = std::mem::replace(&mut self.scopes, saved);
        // copy-out for `&mut` arguments and a `&mut self` receiver. In a
        // version 0.9 module a call that returns by `fail` is transactional
        // (F.4.4): the arguments keep their values from the call.
        let restore = matches!(r, Err(Flow::Fail(_)));
        for ((n, t), a) in named.iter().zip(args) {
            if !restore
                && matches!(t, Ty::Ref(true, _))
                && let Some(v) = fin[0].get(*n)
            {
                self.assign(&a.e, v.clone())?;
            }
        }
        if self_mut && !restore && let (Some((_, obj)), Some(v)) = (recv, fin[0].get("self")) {
            self.assign(obj, v.clone())?;
        }
        match r {
            Ok(v) => match &fd.sig.ret {
                Some(t) => self.fit(v, t),
                None => Ok(V::Unit),
            },
            Err(Flow::Fail(e)) => match &fd.sig.ret {
                Some(Ty::Union(..)) => Ok(V::Err(Shared::from(e))),
                _ => Err(Flow::Fail(e)),
            },
            Err(f) => Err(f),
        }
    }
}

impl<'a> Interp<'a> {
    fn builtin(&mut self, obj: &Expr, recv: V, m: &str, args: &[Arg], sized: bool) -> R<V> {
        if let Some(v) = self.conc_builtin_pre(&recv, m, args)? {
            return Ok(v);
        }
        let mut a: Vec<V> = vec![];
        let lazy_fn = matches!(m, "map" | "filter" | "fold");
        for x in args {
            a.push(self.eval(&x.e)?);
        }
        let _ = lazy_fn;
        // C.6.4, C.15: size-scaled work when the call expression is in a
        // version 0.7 module; c_call alone in 0.5 and 0.6. The version of
        // the module containing the call governs, not the caller's or the
        // program's.
        let work = if sized { crate::conc::method_work(&recv, m, &a) } else { 0 };
        self.charge(crate::conc::C_CALL.saturating_add(work))?;
        if let Some(v) = self.conc_builtin(obj, &recv, m, &a)? {
            return Ok(v);
        }
        use V::*;
        // a Set[τ] is kept in canonical order without duplicates (7.1)
        match (recv, m) {
            (Set(xs), "union") => crate::conc::merge_builtin(&Set(xs), &a[0]),
            (Set(xs), "inter") => match &a[0] {
                Set(ys) => Ok(Set(xs.into_iter().filter(|x| ys.iter().any(|y| key_eq(x, y))).collect())),
                o => trap(format!("inter with {}", o.show())),
            },
            (Set(xs), "insert") => {
                let r = crate::conc::merge_builtin(&Set(xs), &Set(vec![a[0].clone()].into()))?;
                self.assign(obj, r)?;
                Ok(Unit)
            }
            (Set(xs), "tolist") => Ok(List(xs)),
            (Set(xs), "len") => Ok(Int(xs.len() as i128, "usize")),
            (Set(xs), "contains") => Ok(Bool(xs.iter().any(|k| key_eq(k, &a[0])))),
            (Map(xs), "merge") => {
                let Map(ys) = &a[0] else { return trap("merge with a non-map") };
                let mut out = xs;
                for (k, v) in ys {
                    if let Some(i) = out.iter().position(|(k2, _)| key_eq(k2, k)) {
                        let m = self.merge_values(&out[i].1, v)?;
                        out[i].1 = m;
                    } else {
                        out.push((k.clone(), v.clone()));
                    }
                }
                out.sort_by(|p, q| crate::conc::canon_cmp(&p.0, &q.0));
                Ok(Map(out))
            }
            (Int(x, t), "checked_add" | "checked_sub" | "checked_mul" | "checked_div") => {
                let op = match m {
                    "checked_add" => "+",
                    "checked_sub" => "-",
                    "checked_mul" => "*",
                    _ => "/",
                };
                Ok(match self.arith(op, Int(x, t), a[0].clone()) {
                    Ok(v) => Opt(Some(Shared::from(v))),
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
            (Int(x, _), "tostr") => Ok(Text(x.to_string().into())),
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
            // 6.3 N5: IEEE order (total on finite values); the receiver on a tie
            (F(x), "min") => match &a[0] {
                F(y) => Ok(F(if x <= *y { x } else { *y })),
                _ => trap("min"),
            },
            (F(x), "max") => match &a[0] {
                F(y) => Ok(F(if x >= *y { x } else { *y })),
                _ => trap("max"),
            },
            (F(x), "floor") => Ok(F(x.floor())),
            (F(x), "ceil") => Ok(F(x.ceil())),
            (F(x), "round") => Ok(F(x.round())),
            (F(x), "tostr") => Ok(Text(float_text(x).into())),
            (Bool(b), "tostr") => Ok(Text(b.to_string().into())),
            (Text(s), "len") => Ok(Int(s.len() as i128, "usize")),
            (F32(x), "abs") => Ok(F32(x.abs())),
            (F32(x), "tostr") => Ok(Text(((x as f32).to_string()).into())),
            (Gen(g), "collect") => {
                let (items, _, err) = self.gen_run(g, usize::MAX)?;
                if let Some(f) = err {
                    return Result::Err(f);
                }
                Ok(List(items.into()))
            }
            (Text(s), "strip") => Ok(Text(s.trim_matches(is_ws).into())),
            (Text(s), "upper") => Ok(Text(s.to_ascii_uppercase().into())),
            (Text(s), "sha256") => {
                use sha2::{Digest, Sha256};
                Ok(Text(
                    Sha256::digest(s.as_bytes())
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect(),
                ))
            }
            (Text(s), "lower") => Ok(Text(s.to_ascii_lowercase().into())),
            (Text(s), "reverse") => Ok(Text(s.chars().rev().collect())),
            (Text(s), "tostr") => Ok(Text(s)),
            (Text(s), "len_bytes") => Ok(Int(s.len() as i128, "usize")),
            (Text(s), "len_chars") => Ok(Int(s.chars().count() as i128, "usize")),
            (Text(s), "raw") => Ok(Text(s)),
            // F.5.1 `Cap[A].narrow(sub)`: a capability is `()` at run time; the root is a static fact
            (Unit, "narrow") if a.len() == 1 => Ok(Unit),
            (Text(s), "chars") => Ok(List(s.chars().map(|c| Text(c.to_string().into())).collect())),
            (Text(s), "words") => Ok(List(s.split(is_ws).filter(|w| !w.is_empty()).map(|w| Text(w.into())).collect())),
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
                .trim_matches(is_ws)
                .parse::<i128>()
                .map_err(|_| Flow::Trap(format!("not an integer: {}", show_text(&s))))
                .and_then(|v| chk(v, "i64")),
            (Text(s), "join") => match &a[0] {
                List(xs) => Ok(Text(
                    xs.iter()
                        .map(|x| if let Text(t) = x { t.to_string() } else { x.show() })
                        .collect::<Vec<_>>()
                        .join(s.as_str())
                        .into(),
                )),
                _ => trap("join"),
            },
            (List(xs), "len") => Ok(Int(xs.len() as i128, "usize")),
            (List(xs), "isempty") => Ok(Bool(xs.is_empty())),
            (List(mut xs), "append" | "push") => {
                self.release(obj);
                xs.push(a[0].clone());
                let xs = size_ok(List(xs))?;
                self.assign(obj, xs)?;
                Ok(Unit)
            }
            (List(mut xs), "pop") => {
                self.release(obj);
                let r = xs.pop();
                self.assign(obj, List(xs))?;
                Ok(Opt(r.map(Shared::from)))
            }
            (List(_), "clear") => {
                self.assign(obj, List(vec![].into()))?;
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
                    xs.get(*i as usize).cloned().map(Shared::from)
                } else {
                    None
                })),
                _ => trap("get"),
            },
            (List(xs), "utf8") => {
                let mut b = Vec::with_capacity(xs.len());
                for x in xs {
                    match x {
                        Int(v, _) if (0..=255).contains(&v) => b.push(v as u8),
                        _ => return trap("utf8: byte outside 0..255"),
                    }
                }
                match String::from_utf8(b).ok() {
                    Some(s) => Ok(Text(s.into())),
                    None => trap("utf8: invalid UTF-8"),
                }
            }
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
                Ok(List(out.into()))
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
                Ok(List(out.into()))
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
            (Map(mm), "get") => Ok(match mm.iter().find(|(k, _)| key_eq(k, &a[0])) {
                Some((_, v)) => v.clone(),
                None => Err(Shared::from(Rec("Missing".into(), vec![].into()))),
            }),
            (Map(mm), "contains") => Ok(Bool(mm.iter().any(|(k, _)| key_eq(k, &a[0])))),
            (Map(mut mm), "insert") => {
                map_set(&mut mm, a[0].clone(), a[1].clone());
                self.assign(obj, Map(mm))?;
                Ok(Unit)
            }
            (Map(mut mm), "remove") => {
                let i = mm.iter().position(|(k, _)| key_eq(k, &a[0]));
                let r = i.map(|i| mm.remove(i).1);
                self.assign(obj, Map(mm))?;
                Ok(Opt(r.map(Shared::from)))
            }
            (Map(mm), "keys") => Ok(List(mm.into_iter().map(|x| x.0).collect())),
            (Map(mm), "values") => Ok(List(mm.into_iter().map(|x| x.1).collect())),
            (Opt(x), "isnone") => Ok(Bool(x.is_none())),
            (Opt(x), "issome") => Ok(Bool(x.is_some())),
            (Opt(x), "or") => Ok(match x {
                Some(v) => v.take(),
                None => a[0].clone(),
            }),
            (r, _) => trap(format!("{} has no method {m} at comptime", r.show())),
        }
    }
    pub(crate) fn apply(&mut self, f: &V, vals: Vec<V>) -> R<V> {
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
        // 5.6: a `let &mut` borrow names its place (crate::borrow)
        let lowered;
        let b = if b.iter().any(crate::borrow::is_let_mut) {
            lowered = crate::borrow::lower_block(b, &mut LB_TEMPS.lock().unwrap_or_else(|e| e.into_inner()));
            &lowered[..]
        } else {
            b
        };
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
    pub(crate) fn exec_block(&mut self, b: &[Stmt]) -> R<()> {
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

    /// `for` over the items of a non-iterator value
    fn for_items(&mut self, p: &Pat, itv: V, body: &[Stmt]) -> R<Option<V>> {
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
        Ok(None)
    }

    pub(crate) fn exec(&mut self, s: &Stmt, tail: bool) -> R<Option<V>> {
        // the `let` of a borrow's index temporary is not a source statement
        if !matches!(s, Stmt::Let(Pat::Path(p), None, _, _) if p.len() == 1 && crate::borrow::is_temp(&p[0])) {
            self.tick()?;
        }
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
                    // `x op= y` costs what `x = x op y` costs (C.6.4)
                    self.charge(crate::conc::op_cost(&op[..op.len() - 1], &cur))?;
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
            // the iterable is evaluated once; an iterator call runs lazily
            Stmt::For(p, it, body, _) => {
                let itv = self.eval(it)?;
                let V::Gen(g) = itv else {
                    return self.for_items(p, itv, body);
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
                self.conc_emit(v)?;
            }
            Stmt::Fail(e) => {
                let v = self.eval(e)?;
                return Err(Flow::Fail(v));
            }
            Stmt::Arena(name, None, body, line) if name.starts_with('$') => {
                return self.exec_marker(name, body, *line, tail);
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
                let lowered;
                let body = if body.iter().any(crate::borrow::is_let_mut) {
                    lowered = crate::borrow::lower_block(body, &mut LB_TEMPS.lock().unwrap_or_else(|e| e.into_inner()));
                    &lowered[..]
                } else {
                    &body[..]
                };
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
            Stmt::Check(Expr::Call(f, a, _, _)) if a.len() == 2 && matches!(&**f, Expr::Name(n, _) if n == "@contract") => {
                // F.6.2: the fixed record of a violated contract clause (the step keeps the state)
                let v = self.eval(&a[1].e)?;
                if !self.truthy(v)? {
                    let rec = if let Expr::Str(s, _) = &a[0].e { s.clone() } else { String::new() };
                    return trap(rec);
                }
            }
            Stmt::Check(e) => {
                let v = self.eval(e)?;
                if !self.truthy(v)? {
                    return trap("check failed");
                }
            }
            Stmt::Send(t, v) => self.conc_send(t, v)?,
            Stmt::Spawn(e) => {
                self.conc_spawn(e, 0)?;
            }
            Stmt::Select(arms, line) => return self.conc_select(arms, *line, false, tail),
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
    ) -> Result<Option<(Vec<(String, V, Ty)>, Vec<V>)>, KeptReason> {
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
            let reason_of = |f: Flow| {
                let cause = Cause::of(&f);
                let text = match f {
                    Flow::Fail(v) => v.rec_name().unwrap_or("?").to_string(),
                    Flow::Trap(t) => format!("trap({t})"),
                    other => format!("trap({other:?})"),
                };
                KeptReason { cause, text }
            };
            match self.pmatch(pat, ev) {
                Ok(true) => {}
                Ok(false) => continue,
                Err(f) => return Err(reason_of(f)),
            }
            if let Some(g) = guard {
                match self.eval(g).and_then(|v| self.truthy(v)) {
                    Ok(true) => {}
                    Ok(false) => continue,
                    Err(f) => return Err(reason_of(f)),
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
                return Err(reason_of(f));
            }
            let mut out = vec![];
            for (n, _, t) in &r.1 {
                let v = self.scopes[0][n].clone();
                match self.fit(v, t) {
                    Ok(v) => out.push((n.clone(), v, t.clone())),
                    Err(f) => return Err(reason_of(f)),
                }
            }
            return Ok(Some((out, std::mem::take(&mut self.outputs))));
        }
        Ok(None)
    }

    /// δ (6.4): every region on the pre-state; the first region in declaration order that keeps the state keeps it for the whole event (atomically).
    pub fn delta(&mut self, s: &[Region], ev: &V) -> Result<(Vec<Region>, Vec<V>), String> {
        self.delta_r(s, ev).map_err(|r| r.text)
    }

    /// δ with the cause of a kept outcome (HC-21).
    pub fn delta_r(&mut self, s: &[Region], ev: &V) -> Result<(Vec<Region>, Vec<V>), KeptReason> {
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
                        if k + 1 == n && x.kept.is_some() {
                            bad = Some(format!(
                                "expected kept: {}, but {evn} was applied",
                                x.kept.clone().unwrap()
                            ));
                        }
                        s = s2;
                    }
                    // 6.9: the example states the outcome of its last step;
                    // an earlier kept step keeps the state (T8) and the next
                    // event is applied to it
                    Err(r) => {
                        if k + 1 == n && x.kept.as_deref() != Some(r.as_str()) {
                            bad = Some(format!("event {} ({evn}) kept: {r}", k + 1));
                        }
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
                                // state identity (6.1, N3 F6): floats by bits,
                                // so `-0.0` and `0.0` are different results
                                got == w && key_eq(got, w)
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
    errs.extend(crate::conc::crdt_gate(m));
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
        V::Tuple(xs) | V::List(xs) | V::Set(xs) => seq(&mut xs.iter().map(json_size), xs.len()),
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
        V::Chan(..) | V::Actor(..) | V::Task(_) | V::Session(..) | V::LVar(_) => 2,
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
            it.fuel = fuel_limit(); // HC-7a: each input is one run of δ, as on the host
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
                    // a kept step keeps the state (T8); the next event is applied to it
                    steps.push(TraceStep {
                        event: label,
                        result: Err(r),
                    });
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
            FV::Int(_) => 2,
            FV::Float(_) => 3,
            FV::Text(_) => 4,
            FV::List(_) => 5,
            FV::Rec(_) => 6,
        }
    }
    /// Canonical order of 6.1: null < bool < int < float < text < list <
    /// record; floats by IEEE 754 totalOrder. It agrees with state identity:
    /// it returns Equal only for values with the same canonical JSON, so
    /// distinct observable values stay distinct map keys.
    pub fn cmp_canon(&self, o: &FV) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        let r = self.rank().cmp(&o.rank());
        if r != Equal {
            return r;
        }
        match (self, o) {
            (FV::Bool(a), FV::Bool(b)) => a.cmp(b),
            (FV::Int(a), FV::Int(b)) => a.cmp(b),
            // IEEE 754 totalOrder (-0.0 < 0.0); NaN cannot occur (N3 F4)
            (FV::Float(a), FV::Float(b)) => a.total_cmp(b),
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
            FV::Float(f) => float_json(*f),
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

/// The canonical JSON of a state (6.1), whose SHA-256 is an entry's `state_hash`.
pub fn state_json(s: &[Region], m: &Module) -> String {
    rec(s
        .iter()
        .map(|r| (r.0.clone(), rec(r.1.iter().map(|(n, v, _)| (n.clone(), form_value(v, m))).collect())))
        .collect())
    .json()
}

/// The canonical JSON of an event as D stores it: `{"payload":..,"type":..}`.
pub fn event_json(name: &str, ev: &V, m: &Module) -> String {
    format!("{{\"payload\":{},\"type\":{}}}", form_value(ev, m).json(), jstr(name))
}

/// The declared field names of a struct, event or error, in declaration order.
fn decl_fields<'a>(m: &'a Module, n: &str) -> Option<Vec<&'a str>> {
    m.items.iter().find_map(|i| match i {
        Item::Struct { name, fields, .. } if name == n => Some(fields.iter().map(|f| f.name.as_str()).collect()),
        Item::Event { name, payload, .. } | Item::Error { name, payload, .. } if name == n => {
            Some(payload.iter().map(|(f, _)| f.as_str()).collect())
        }
        _ => None,
    })
}

/// The index of a variant within its kind and its field names, in declaration order.
fn variant_index<'a>(m: &'a Module, n: &str) -> Option<(usize, Vec<&'a str>)> {
    m.items.iter().find_map(|i| match i {
        Item::Kind { variants, .. } => variants
            .iter()
            .position(|(v, _)| v == n)
            .map(|k| (k, variants[k].1.iter().map(|(f, _)| f.as_str()).collect())),
        _ => None,
    })
}

/// The index of an event among the module's declared events (6.1, T7).
pub fn event_index(m: &Module, n: &str) -> Option<usize> {
    m.items
        .iter()
        .filter(|i| matches!(i, Item::Event { .. }))
        .position(|i| matches!(i, Item::Event { name, .. } if name == n))
}

/// The positional embedding (6.1, T7): the value with every declared name
/// replaced by a position. A record of a declared struct, event or error is
/// the list of its field values in declaration order; a variant record is
/// `[k, [fields]]` where k is the variant's index in its kind; every other
/// value embeds as `form_value` does. A record whose type the module does
/// not declare (the host's `host.*` records, a label) keeps the named form.
pub fn form_value_pos(v: &V, m: &Module) -> FV {
    match v {
        V::Opt(Some(x)) => form_value_pos(x, m),
        V::Tuple(xs) | V::List(xs) | V::Set(xs) => FV::List(xs.iter().map(|x| form_value_pos(x, m)).collect()),
        V::Map(ps) => {
            let mut l: Vec<(FV, FV, FV)> = ps
                .iter()
                .map(|(k, v)| (form_value(k, m), form_value_pos(k, m), form_value_pos(v, m)))
                .collect();
            l.sort_by(|a, b| a.0.cmp_canon(&b.0));
            FV::List(l.into_iter().map(|(_, k, v)| FV::List(vec![k, v])).collect())
        }
        V::Rec(n, fs) => {
            let at = |f: &str| fs.iter().find(|(k, _)| k == f).map_or(FV::Null, |(_, x)| form_value_pos(x, m));
            if let Some(decl) = decl_fields(m, n) {
                FV::List(decl.iter().map(|f| at(f)).collect())
            } else if let Some((k, decl)) = variant_index(m, n) {
                FV::List(vec![FV::Int(k as i128), FV::List(decl.iter().map(|f| at(f)).collect())])
            } else {
                form_value(v, m)
            }
        }
        V::Err(x) => rec(vec![("$err".into(), form_value_pos(x, m))]),
        _ => form_value(v, m),
    }
}

/// The hashed form of a state (6.1, T7): the regions in declaration order,
/// each the list of its field values in declaration order. Its SHA-256 is
/// an entry's `state_hash`; `state_json` is the readable form D stores.
pub fn state_pos_json(s: &[Region], m: &Module) -> String {
    FV::List(
        s.iter()
            .map(|r| FV::List(r.1.iter().map(|(_, v, _)| form_value_pos(v, m)).collect()))
            .collect(),
    )
    .json()
}

/// The hashed form of an event (6.1, T7): `{"payload":P,"type":k}` with P
/// the positional embedding of the payload and k the event's index among
/// the module's events. The host's `host.*` events and the built-in `Tick`
/// keep their name as `type`.
pub fn event_pos_json(name: &str, ev: &V, m: &Module) -> String {
    let ty = match event_index(m, name) {
        Some(k) => k.to_string(),
        None => jstr(name),
    };
    format!("{{\"payload\":{},\"type\":{ty}}}", form_value_pos(ev, m).json())
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
        V::Text(t) => FV::Text(t.to_string()),
        V::Unit | V::Opt(None) => FV::Null,
        V::Opt(Some(x)) => form_value(x, m),
        V::Tuple(xs) | V::List(xs) | V::Set(xs) => FV::List(xs.iter().map(|x| form_value(x, m)).collect()),
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
        V::Chan(..) | V::Actor(..) | V::Task(_) | V::Session(..) | V::LVar(_) => FV::Null,
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
/// The first `prev_hash` of every log D (6.1).
pub const D_GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// SHA-256 of text, lower-case hex (the hash of D, 6.1).
pub fn d_sha(s: &str) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(s.as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
}

/// δ is given as its step table (region, pattern, guard, source line); the
/// step bodies are the program's own. D is the log of each example, with each
/// entry's outcome, outputs (λ) and the state after it.
///
/// Every entry carries the canonical D fields of 6.1 (`seq`, `event`,
/// `prev_hash`, `state_hash`, `hash`), the same chain the engine profile
/// writes (D.6): `state_hash` is the SHA-256 of the canonical JSON of the
/// state after the entry, `hash` the SHA-256 of the canonical JSON of
/// {event, prev_hash, seq, state_hash}, and the first `prev_hash` is 64
/// zeros. `outcome`, `out` and `state` are derived fields (equal to replay).
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
            let mut prev = D_GENESIS.to_string();
            for (evn, args) in &x.events {
                it.scopes = vec![HashMap::new()];
                it.fuel = fuel_limit(); // HC-7a: each input is one run of δ, as on the host
                let ev = match it.construct(evn, args) {
                    Ok(Some(v)) => v,
                    _ => break,
                };
                let evj = event_json(evn, &ev, m);
                let evp = event_pos_json(evn, &ev, m);
                let (outcome, out) = match it.delta(&s, &ev) {
                    Ok((s2, outs)) => {
                        s = s2;
                        let o = outs
                            .iter()
                            .map(|o| form_value(o, m).json())
                            .collect::<Vec<_>>()
                            .join(",");
                        ("\"applied\"".to_string(), o)
                    }
                    // T8: a kept step leaves the state unchanged, and the
                    // next event is applied to that state.
                    Err(r) => (
                        format!("{{\"kept\":{}}}", jstr(&r)),
                        format!(
                            "{{\"Kept\":{{\"error\":{},\"payload\":{}}}}}",
                            jstr(&r),
                            form_value(&ev, m).json()
                        ),
                    ),
                };
                let st = state_fv(&s).json();
                // T7: the hashes are over the positional forms; `event` and
                // `state` are the readable forms, and `hashed` repeats what
                // was hashed so a reader can recompute both hashes.
                let stp = state_pos_json(&s, m);
                let (seq, sh) = (entries.len(), d_sha(&stp));
                let body = format!(
                    "{{\"event\":{evp},\"prev_hash\":\"{prev}\",\"seq\":{seq},\"state_hash\":\"{sh}\"}}"
                );
                let h = d_sha(&body);
                entries.push(format!(
                    "{{\"event\":{evj},\"hash\":\"{h}\",\"hashed\":{{\"event\":{evp},\"state\":{stp}}},\"out\":[{out}],\"outcome\":{outcome},\"prev_hash\":\"{prev}\",\"seq\":{seq},\"state\":{st},\"state_hash\":\"{sh}\"}}"
                ));
                prev = h;
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
