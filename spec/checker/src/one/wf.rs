//! Well-formedness of a core module (one-emitter.md ONE-3, W1 to W6),
//! checked whenever a core module is read.
//!
//! - W1 Types. Every atom and application is well typed, and every `ret`
//!   matches the function's result type.
//! - W2 Paths. Every path of a function ends in `ret` or `halt`, and every
//!   `break d` names an enclosing `block` or `loop`.
//! - W3 Ownership. Every owned local is consumed exactly once on every path
//!   that returns. A path that halts ends every value it holds. An owned
//!   global is moved out only by a binding and set again before the
//!   function returns or calls; the entry function sets each owned global
//!   before its first read or call.
//! - W4 Domains. Every application of a partial primitive is dominated by
//!   the domain test the lowering placed before it. A fact about a `var`
//!   local or a global is a fact about the value it held then, never about
//!   the place: each value a place takes has its own epoch. Setting the
//!   place, a call or catch (which may set any global), the head of a loop
//!   whose body may set it or call, and the end of a block or loop whose
//!   body may, each give the place a new epoch, so a fact tested before
//!   does not cover a later value, and a local that read the earlier value
//!   keeps the facts about it. `mem.get` reads memory that `mem.set`
//!   changes, so each result of it is a value of its own.
//! - W5 Keys. A group's key is the SHA-256 of its canonical JSON, `rec i`
//!   names a member of its own group, and the graph of keys has no cycle.
//!   The members of a group are in canonical order
//!   (`json::canonical_order`): their final digests ascend strictly, so
//!   no two members are bisimilar and one group has one key.
//! - W6 Arguments. The operands of `call` and `catch` are locals or
//!   literals, and `set` names a `var` local or a global.
//!
//! Locals are lexically scoped: a local bound in a nested statement list is
//! visible in that list after its binding and in the lists nested there.
//! Integer types are ordered by inclusion: wherever a value of `int[lo, hi]`
//! is expected, a value of any integer type whose range lies inside it may
//! be given.
use super::ir::*;
use super::json::{canonical_order, group_key};
use std::collections::{BTreeMap, BTreeSet, HashMap};

type R<T> = Result<T, String>;

/// Check every rule of ONE-3 on `m`; the error names the rule and the place.
pub fn check(m: &Module) -> R<()> {
    // W5: keys, callees, canonical order, no cycle
    for (k, g) in &m.defs {
        let want = group_key(g);
        if &want != k {
            return Err(format!("W5: group {k} has the key {want}"));
        }
        if g.fns.is_empty() {
            return Err(format!("W5: group {k} has no function"));
        }
    }
    let mut graph: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (k, g) in &m.defs {
        let mut out = BTreeSet::new();
        for f in &g.fns {
            callees(&f.body, &mut out);
        }
        let mut ks = BTreeSet::new();
        for c in &out {
            match c {
                FnRef::Key(c, i) => {
                    let Some(cg) = m.defs.get(c) else {
                        return Err(format!("W5: group {k} calls the unknown group {c}"));
                    };
                    if *i as usize >= cg.fns.len() {
                        return Err(format!("W5: group {c} has no function {i}"));
                    }
                    ks.insert(c.clone());
                }
                FnRef::Rec(i) => {
                    if *i as usize >= g.fns.len() {
                        return Err(format!("W5: rec {i} outside group {k}"));
                    }
                }
            }
        }
        graph.insert(k.clone(), ks);
    }
    // every `rec i` names a member, so the canonical order is defined
    for (k, g) in &m.defs {
        let (_, class) = canonical_order(&g.fns);
        if let Some(i) = class.iter().enumerate().position(|(i, c)| *c as usize != i) {
            return Err(format!("W5: member {i} of group {k} is not in canonical order, or repeats a member"));
        }
    }
    acyclic(&graph)?;
    let Some(entry) = m.func(&m.entry.0, m.entry.1) else {
        return Err("W5: the entry names no function".into());
    };
    if !entry.params.is_empty() {
        return Err("W1: the entry function takes no parameters".into());
    }
    for (i, (t, l)) in m.globals.iter().enumerate() {
        match (t.owned(), l) {
            (true, Some(_)) => return Err(format!("W1: owned global g{i} has a literal")),
            (false, None) => return Err(format!("W1: plain global g{i} has no literal")),
            (false, Some(l)) => lit_ok(t, l).map_err(|e| format!("W1: global g{i}: {e}"))?,
            (true, None) => {}
        }
    }
    // A group's check depends only on its content, which its key names, on
    // the signatures of its callees, which their keys name, and on the
    // globals it reads; so a group found well formed once is not checked again.
    static DONE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> = std::sync::OnceLock::new();
    let done = DONE.get_or_init(Default::default);
    let gkey = super::json::canon(&serde_json::Value::Array(m.globals.iter().map(|(t, _)| super::json::ty_json(t)).collect()));
    for (k, g) in &m.defs {
        let entry_here = *k == m.entry.0;
        let memo = if g.fns.iter().any(|f| mentions_globals(&f.body)) { format!("{k} {gkey}") } else { k.clone() };
        if !entry_here && done.lock().unwrap().contains(&memo) {
            continue;
        }
        for (i, f) in g.fns.iter().enumerate() {
            let is_entry = entry_here && i as u32 == m.entry.1;
            Fun::new(m, g, f).check(is_entry).map_err(|e| format!("in {k}/{i}: {e}"))?;
        }
        if !entry_here {
            done.lock().unwrap().insert(memo);
        }
    }
    Ok(())
}

fn mentions_globals(b: &[Stmt]) -> bool {
    let g = |a: &Atom| matches!(a, Atom::G(_));
    b.iter().any(|s| match s {
        Stmt::Let(a) | Stmt::Var(_, a) | Stmt::Ret(a) | Stmt::Halt(a) | Stmt::Out(a) => g(a),
        Stmt::Prim(_, _, xs) | Stmt::Call(_, xs) | Stmt::Catch(_, xs) => xs.iter().any(g),
        Stmt::Set(p, a) => matches!(p, Place::G(_)) || g(a),
        Stmt::If(c, t, e) => g(c) || mentions_globals(t) || mentions_globals(e),
        Stmt::Block(x) | Stmt::Loop(x) => mentions_globals(x),
        Stmt::Break(_) => false,
    })
}

fn callees(b: &[Stmt], out: &mut BTreeSet<FnRef>) {
    for s in b {
        match s {
            Stmt::Call(f, _) | Stmt::Catch(f, _) => {
                out.insert(f.clone());
            }
            Stmt::If(_, t, e) => {
                callees(t, out);
                callees(e, out);
            }
            Stmt::Block(x) | Stmt::Loop(x) => callees(x, out),
            _ => {}
        }
    }
}

fn acyclic(g: &BTreeMap<String, BTreeSet<String>>) -> R<()> {
    // 1 on the stack, 2 done
    let mut st: HashMap<&str, u8> = HashMap::new();
    fn go<'a>(n: &'a str, g: &'a BTreeMap<String, BTreeSet<String>>, st: &mut HashMap<&'a str, u8>) -> R<()> {
        match st.get(n) {
            Some(1) => return Err(format!("W5: the keys form a cycle through {n}")),
            Some(2) => return Ok(()),
            _ => {}
        }
        st.insert(n, 1);
        for c in g.get(n).into_iter().flatten() {
            go(c, g, st)?;
        }
        st.insert(n, 2);
        Ok(())
    }
    for n in g.keys() {
        go(n, g, &mut st)?;
    }
    Ok(())
}

fn lit_ok(t: &Ty, l: &Lit) -> R<()> {
    match (t, l) {
        (Ty::Bool, Lit::Bool(_)) | (Ty::Bytes, Lit::Bytes(_)) => Ok(()),
        (Ty::Int(lo, hi), Lit::Int(v)) if lo <= v && v <= hi => Ok(()),
        (Ty::F64, Lit::F64(b)) if f64::from_bits(*b).is_finite() => Ok(()),
        (Ty::F32, Lit::F32(b)) => {
            let x = f64::from_bits(*b);
            if x.is_finite() && (x as f32) as f64 == x { Ok(()) } else { Err("an f32 literal must be a finite binary32 value".into()) }
        }
        _ => Err(format!("the literal {l:?} is not a value of {t:?}")),
    }
}

/// Whether a value of `a` may stand where `b` is expected.
pub fn fits(a: &Ty, b: &Ty) -> bool {
    a == b || matches!((a, b), (Ty::Int(..), Ty::Int(..)) if a.within(b))
}

fn standard(lo: i128, hi: i128) -> bool {
    (1..=64).any(|k| (lo == 0 && hi == (1i128 << k) - 1) || (lo == -(1i128 << (k - 1)) && hi == (1i128 << (k - 1)) - 1))
}

fn idx_lit(a: &Atom) -> Option<i128> {
    match a {
        Atom::K(Ty::Int(..), Lit::Int(v)) => Some(*v),
        _ => None,
    }
}

/// The result types of primitive `p` at type arguments `ts` applied to
/// operands of types `ops` (W1). `args` are the operands themselves, read
/// for the literal indices of `tup.get`, `sum.mk` and `sum.get`.
pub fn signature(p: &str, ts: &[Ty], ops: &[Ty], args: &[Atom]) -> R<Vec<Ty>> {
    let n = ops.len();
    let arity = |k: usize| if n == k { Ok(()) } else { Err(format!("{p} takes {k} operands, found {n}")) };
    let targs = |k: usize| if ts.len() == k { Ok(()) } else { Err(format!("{p} takes {k} type arguments, found {}", ts.len())) };
    let is_int = |t: &Ty| matches!(t, Ty::Int(..));
    let ints = |k: usize| -> R<()> {
        arity(k)?;
        if ops.iter().all(is_int) { Ok(()) } else { Err(format!("{p} takes integers, found {ops:?}")) }
    };
    let iota = || -> R<(i128, i128, Ty)> {
        match ts.first() {
            Some(t @ Ty::Int(lo, hi)) if ts.len() == 1 => Ok((*lo, *hi, t.clone())),
            _ => Err(format!("{p} takes one integer type argument, found {ts:?}")),
        }
    };
    let within = |k: usize, t: &Ty| -> R<()> {
        if fits(&ops[k], t) { Ok(()) } else { Err(format!("{p}: operand {k} of type {:?} is not within {t:?}", ops[k])) }
    };
    let same = |want: &[Ty]| -> R<()> {
        arity(want.len())?;
        for (k, w) in want.iter().enumerate() {
            within(k, w)?;
        }
        Ok(())
    };
    let b = || Ok(vec![Ty::Bool]);
    let len_t = || Ty::Int(0, INT_MAX);
    match p {
        "bool.not" => { targs(0)?; same(&[Ty::Bool])?; b() }
        "bool.and" | "bool.or" | "bool.eq" => { targs(0)?; same(&[Ty::Bool, Ty::Bool])?; b() }
        _ if p.starts_with("int.") => {
            let (lo, hi, t) = iota()?;
            match &p[4..] {
                "add" | "sub" | "mul" | "wadd" | "wsub" | "wmul" | "sadd" | "ssub" | "smul" | "quo" | "shl" => {
                    ints(2)?;
                    Ok(vec![t])
                }
                "rem" => {
                    ints(2)?;
                    let (alo, ahi) = ops[0].bounds().unwrap();
                    if alo.min(0) < lo || ahi.max(0) > hi {
                        return Err("int.rem: the dividend's type with 0 must lie within the type argument".into());
                    }
                    Ok(vec![t])
                }
                "bad.add" | "bad.sub" | "bad.mul" | "bad.quo" | "bad.shift" => { ints(2)?; b() }
                "neg" | "abs" | "cast" => { ints(1)?; Ok(vec![t]) }
                "bad.neg" | "bad.abs" | "bad.cast" => { ints(1)?; b() }
                "eq" | "lt" | "le" => { same(&[t.clone(), t])?; b() }
                "min" | "max" => { same(&[t.clone(), t.clone()])?; Ok(vec![t]) }
                "and" | "or" | "xor" => {
                    if !standard(lo, hi) {
                        return Err(format!("{p} needs a type of the form [0, 2^k - 1] or [-2^(k-1), 2^(k-1) - 1]"));
                    }
                    same(&[t.clone(), t.clone()])?;
                    Ok(vec![t])
                }
                "not" => {
                    if !standard(lo, hi) {
                        return Err(format!("{p} needs a type of the form [0, 2^k - 1] or [-2^(k-1), 2^(k-1) - 1]"));
                    }
                    same(std::slice::from_ref(&t))?;
                    Ok(vec![t])
                }
                "shr" => {
                    ints(2)?;
                    within(0, &t)?;
                    if lo > 0 || hi < 0 {
                        return Err("int.shr needs a type that holds 0".into());
                    }
                    Ok(vec![t])
                }
                _ => Err(format!("unknown primitive {p}")),
            }
        }
        _ if p.starts_with("f64.") => {
            let f = Ty::F64;
            match &p[4..] {
                "add" | "sub" | "mul" | "div" | "min" | "max" => { targs(0)?; same(&[f.clone(), f.clone()])?; Ok(vec![f]) }
                "bad.add" | "bad.sub" | "bad.mul" | "bad.div" | "eq" | "lt" | "le" => { targs(0)?; same(&[f.clone(), f])?; b() }
                "sqrt" | "neg" | "abs" | "floor" | "ceil" | "round" => { targs(0)?; same(std::slice::from_ref(&f))?; Ok(vec![f]) }
                "bad.sqrt" | "bad.to_f32" | "sign" => { targs(0)?; same(&[f])?; b() }
                "to_f32" => { targs(0)?; same(&[f])?; Ok(vec![Ty::F32]) }
                "of_int" => { let (_, _, t) = iota()?; same(&[t])?; Ok(vec![Ty::F64]) }
                "trunc" => { let (_, _, t) = iota()?; same(&[f])?; Ok(vec![t]) }
                "bad.trunc" => { iota()?; same(&[f])?; b() }
                _ => Err(format!("unknown primitive {p}")),
            }
        }
        _ if p.starts_with("f32.") => {
            targs(0)?;
            let f = Ty::F32;
            match &p[4..] {
                "add" | "sub" | "mul" | "div" => { same(&[f.clone(), f.clone()])?; Ok(vec![f]) }
                "bad.add" | "bad.sub" | "bad.mul" | "bad.div" => { same(&[f.clone(), f])?; b() }
                "to_f64" => { same(&[f])?; Ok(vec![Ty::F64]) }
                _ => Err(format!("unknown primitive {p}")),
            }
        }
        "bytes.len" => { targs(0)?; same(&[Ty::Bytes])?; Ok(vec![len_t()]) }
        "bytes.get" => { targs(0)?; arity(2)?; within(0, &Ty::Bytes)?; ints(2).or_else(|_| if is_int(&ops[1]) { Ok(()) } else { Err(format!("{p}: the index is not an integer")) })?; Ok(vec![Ty::u8()]) }
        "bytes.slice" => {
            targs(0)?;
            arity(3)?;
            within(0, &Ty::Bytes)?;
            if !is_int(&ops[1]) || !is_int(&ops[2]) {
                return Err("bytes.slice takes integer bounds".into());
            }
            Ok(vec![Ty::Bytes])
        }
        "bytes.cat" => { targs(0)?; same(&[Ty::Bytes, Ty::Bytes])?; Ok(vec![Ty::Bytes]) }
        "bytes.eq" | "bytes.lt" => { targs(0)?; same(&[Ty::Bytes, Ty::Bytes])?; b() }
        "bytes.of_list" => {
            targs(0)?;
            arity(1)?;
            match &ops[0] {
                Ty::List(e) if fits(e, &Ty::u8()) => Ok(vec![Ty::Bytes]),
                o => Err(format!("bytes.of_list takes a list of int[0,255]-ranged integers, found {o:?}")),
            }
        }
        "bytes.to_list" => { targs(0)?; same(&[Ty::Bytes])?; Ok(vec![Ty::list(Ty::u8())]) }
        "arr.new" | "arr.len" | "arr.get" | "arr.set" | "list.new" | "list.len" | "list.get" | "list.set" | "list.push" | "list.pop" => {
            targs(1)?;
            let e = ts[0].clone();
            let c = if p.starts_with("arr") { Ty::Arr(Box::new(e.clone())) } else { Ty::List(Box::new(e.clone())) };
            let idx_ok = |k: usize| if is_int(&ops[k]) { Ok(()) } else { Err(format!("{p}: the index is not an integer")) };
            match &p[p.find('.').unwrap() + 1..] {
                "new" if p == "arr.new" => { arity(2)?; idx_ok(0)?; within(1, &e)?; Ok(vec![c]) }
                "new" => { arity(0)?; Ok(vec![c]) }
                "len" => { same(&[c])?; Ok(vec![len_t()]) }
                "get" => { arity(2)?; within(0, &c)?; idx_ok(1)?; Ok(vec![e]) }
                "set" => { arity(3)?; within(0, &c)?; idx_ok(1)?; within(2, &e)?; Ok(vec![c]) }
                "push" => { same(&[c.clone(), e])?; Ok(vec![c]) }
                "pop" => { same(std::slice::from_ref(&c))?; Ok(vec![c, e]) }
                _ => Err(format!("unknown primitive {p}")),
            }
        }
        "tup.mk" | "tup.get" | "tup.open" => {
            targs(1)?;
            let Ty::Tup(cs) = &ts[0] else { return Err(format!("{p} takes a tuple type")) };
            match p {
                "tup.mk" => { same(cs)?; Ok(vec![ts[0].clone()]) }
                "tup.open" => { same(&[ts[0].clone()])?; Ok(cs.clone()) }
                _ => {
                    arity(2)?;
                    within(0, &ts[0])?;
                    match idx_lit(&args[1]) {
                        Some(k) if k >= 0 && (k as usize) < cs.len() => Ok(vec![cs[k as usize].clone()]),
                        _ => Err("tup.get takes a literal component index".into()),
                    }
                }
            }
        }
        "sum.mk" | "sum.tag" | "sum.get" => {
            targs(1)?;
            let Ty::Sum(cs) = &ts[0] else { return Err(format!("{p} takes a sum type")) };
            match p {
                "sum.tag" => { same(&[ts[0].clone()])?; Ok(vec![Ty::Int(0, cs.len() as i128 - 1)]) }
                "sum.mk" => {
                    arity(2)?;
                    match idx_lit(&args[0]) {
                        Some(k) if k >= 0 && (k as usize) < cs.len() => {
                            within(1, &cs[k as usize])?;
                            Ok(vec![ts[0].clone()])
                        }
                        _ => Err("sum.mk takes a literal variant index".into()),
                    }
                }
                _ => {
                    arity(2)?;
                    within(0, &ts[0])?;
                    match idx_lit(&args[1]) {
                        Some(k) if k >= 0 && (k as usize) < cs.len() => Ok(vec![cs[k as usize].clone()]),
                        _ => Err("sum.get takes a literal variant index".into()),
                    }
                }
            }
        }
        "io.byte" => { targs(0)?; same(&[Ty::u8()])?; Ok(vec![]) }
        "mem.get" => { let (_, _, t) = iota()?; ints(1)?; Ok(vec![t]) }
        "mem.set" => { let (_, _, t) = iota()?; arity(2)?; ints(2)?; within(1, &t)?; Ok(vec![]) }
        "copy" => { targs(1)?; same(&[ts[0].clone()])?; Ok(vec![ts[0].clone()]) }
        "drop" => { targs(1)?; same(&[ts[0].clone()])?; Ok(vec![]) }
        _ => Err(format!("unknown primitive {p}")),
    }
}

/// The operands a primitive reads without consuming them (W3).
pub fn reads(p: &str, k: usize) -> bool {
    match p {
        "list.len" | "arr.len" | "copy" | "sum.tag" => k == 0,
        "list.get" | "arr.get" | "tup.get" => true,
        _ => false,
    }
}

// ---- per function

#[derive(Clone, Copy, PartialEq, Debug)]
enum Own {
    Live,
    Dead,
}

#[derive(Clone)]
struct Local {
    ty: Ty,
    var: bool,
    param: bool,
}

/// A place whose value can change: a `var` local or a global.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
enum Key {
    L(u32),
    G(u32),
}

/// A fact term: an atom, or a primitive applied to terms.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Term {
    /// a local that is not a `var`, which holds one value
    V(u32),
    /// the value a `var` local or a global holds in an epoch
    At(Key, u32),
    K(Lit),
    P(String, Vec<Ty>, Vec<Term>),
}

impl Term {
    fn mentions(&self, v: &Term) -> bool {
        self == v || matches!(self, Term::P(_, _, xs) if xs.iter().any(|x| x.mentions(v)))
    }
    /// Whether the term reads local k, in any epoch when k is a `var`.
    fn mentions_local(&self, k: u32) -> bool {
        match self {
            Term::V(i) | Term::At(Key::L(i), _) => *i == k,
            Term::P(_, _, xs) => xs.iter().any(|x| x.mentions_local(k)),
            _ => false,
        }
    }
}

#[derive(Clone)]
struct St {
    /// owned locals in scope that hold a value
    own: BTreeMap<u32, Own>,
    /// owned globals
    gown: BTreeMap<u32, Own>,
    /// facts known to hold: (term, truth)
    facts: Facts,
    /// the epoch of each place's value here; a place not listed is in
    /// epoch 0, the value it had where the function or its `var` began
    epochs: BTreeMap<Key, u32>,
}

impl St {
    fn epoch(&self, k: Key) -> u32 {
        self.epochs.get(&k).copied().unwrap_or(0)
    }
}

/// A set of facts, indexed by the term and by the primitive and operands
/// without its type arguments.
#[derive(Clone, Default)]
struct Facts {
    all: std::collections::HashSet<(Term, bool)>,
    loose: HashMap<(String, Vec<Term>), Vec<bool>>,
}

impl Facts {
    fn push(&mut self, f: (Term, bool)) {
        if let Term::P(p, _, xs) = &f.0 {
            self.loose.entry((p.clone(), xs.clone())).or_default().push(f.1);
        }
        self.all.insert(f);
    }
    fn contains(&self, f: &(Term, bool)) -> bool {
        self.all.contains(f)
    }
    fn holds(&self, p: &str, xs: &[Term], truth: bool) -> bool {
        self.loose.get(&(p.to_string(), xs.to_vec())).is_some_and(|v| v.contains(&truth))
    }
    fn retain(&mut self, keep: impl std::ops::Fn(&(Term, bool)) -> bool) {
        self.all.retain(|f| keep(f));
        let all = &self.all;
        let mut loose: HashMap<(String, Vec<Term>), Vec<bool>> = HashMap::new();
        for f in all {
            if let Term::P(p, _, xs) = &f.0 {
                loose.entry((p.clone(), xs.clone())).or_default().push(f.1);
            }
        }
        self.loose = loose;
    }
}

struct Exit {
    /// the locals in scope where the construct begins
    scope: BTreeSet<u32>,
    /// the state at each `break` that leaves the construct
    breaks: Vec<St>,
}

struct Fun<'a> {
    m: &'a Module,
    g: &'a Group,
    f: &'a Fn,
    locals: HashMap<u32, Local>,
    /// the definitions of locals bound by a primitive with one result
    defs: HashMap<u32, Term>,
    next: u32,
    /// the open blocks and loops, innermost last
    exits: Vec<Exit>,
    /// the last epoch given to a place's value
    epoch: u32,
}

impl<'a> Fun<'a> {
    fn new(m: &'a Module, g: &'a Group, f: &'a Fn) -> Self {
        let mut locals = HashMap::new();
        for (i, t) in f.params.iter().enumerate() {
            locals.insert(i as u32, Local { ty: t.clone(), var: false, param: true });
        }
        Fun { m, g, f, locals, defs: HashMap::new(), next: f.params.len() as u32, exits: vec![], epoch: 0 }
    }

    /// Check the function; in the entry function the owned globals start
    /// unset, so it sets each before it reads it or calls.
    fn check(&mut self, is_entry: bool) -> R<()> {
        let mut st = St { own: BTreeMap::new(), gown: BTreeMap::new(), facts: Facts::default(), epochs: BTreeMap::new() };
        for (i, t) in self.f.params.iter().enumerate() {
            if t.owned() {
                st.own.insert(i as u32, Own::Live);
            }
        }
        for (i, (t, _)) in self.m.globals.iter().enumerate() {
            if t.owned() {
                st.gown.insert(i as u32, if is_entry { Own::Dead } else { Own::Live });
            }
        }
        let body = &self.f.body;
        if self.block(body, &mut st)? {
            return Err("W2: a path reaches the end of the function without ret or halt".into());
        }
        Ok(())
    }

    fn aty(&self, a: &Atom) -> R<Ty> {
        match a {
            Atom::V(i) => self.locals.get(i).map(|l| l.ty.clone()).ok_or_else(|| format!("W1: v{i} is not in scope")),
            Atom::G(i) => self.m.globals.get(*i as usize).map(|g| g.0.clone()).ok_or_else(|| format!("W1: no global g{i}")),
            Atom::K(t, l) => {
                if t.owned() {
                    return Err("W1: a literal of an owned type".into());
                }
                lit_ok(t, l).map_err(|e| format!("W1: {e}"))?;
                Ok(t.clone())
            }
        }
    }

    fn term(&self, st: &St, a: &Atom) -> Term {
        match a {
            Atom::V(i) if self.locals.get(i).is_some_and(|l| l.var) => Term::At(Key::L(*i), st.epoch(Key::L(*i))),
            Atom::V(i) => self.defs.get(i).cloned().unwrap_or(Term::V(*i)),
            Atom::G(i) => Term::At(Key::G(*i), st.epoch(Key::G(*i))),
            Atom::K(_, l) => Term::K(l.clone()),
        }
    }

    /// Use an atom: consume it when `consume` and it is owned (W3).
    fn use_atom(&self, a: &Atom, consume: bool, st: &mut St, binding: bool) -> R<Ty> {
        let t = self.aty(a)?;
        match a {
            Atom::V(i) if t.owned() => match st.own.get(i) {
                Some(Own::Live) => {
                    if consume {
                        st.own.insert(*i, Own::Dead);
                    }
                }
                _ => return Err(format!("W3: v{i} is used after it was consumed")),
            },
            Atom::G(i) if t.owned() => match st.gown.get(i) {
                Some(Own::Live) => {
                    if consume {
                        if !binding {
                            return Err(format!("W3: the owned global g{i} is moved out by an operation other than a binding"));
                        }
                        st.gown.insert(*i, Own::Dead);
                    }
                }
                _ => return Err(format!("W3: g{i} is read while it is moved out")),
            },
            _ => {}
        }
        Ok(t)
    }

    fn bind(&mut self, t: Ty, var: bool, st: &mut St) -> u32 {
        let i = self.next;
        self.next += 1;
        if t.owned() {
            st.own.insert(i, Own::Live);
        }
        self.locals.insert(i, Local { ty: t, var, param: false });
        i
    }

    fn globals_back(&self, st: &St, what: &str) -> R<()> {
        if let Some((g, _)) = st.gown.iter().find(|(_, o)| **o == Own::Dead) {
            return Err(format!("W3: the owned global g{g} is not set again before {what}"));
        }
        Ok(())
    }

    /// Forget local k where it goes out of scope: the facts that read it
    /// and its epochs.
    fn forget(&self, st: &mut St, k: u32) {
        if st.facts.all.iter().any(|(f, _)| f.mentions_local(k)) {
            st.facts.retain(|(f, _)| !f.mentions_local(k));
        }
        st.epochs.remove(&Key::L(k));
    }

    /// Give place k a new epoch: its value from here on may differ from
    /// every value it had before.
    fn renew(&mut self, st: &mut St, k: Key) {
        self.epoch += 1;
        st.epochs.insert(k, self.epoch);
    }

    fn known(&self, st: &St, t: &Term, truth: bool) -> bool {
        st.facts.contains(&(t.clone(), truth))
    }

    fn nonneg(&self, st: &St, a: &Atom, t: &Ty) -> bool {
        if let Ty::Int(lo, _) = t
            && *lo >= 0
        {
            return true;
        }
        let x = self.term(st, a);
        let z = Term::K(Lit::Int(0));
        st.facts.holds("int.lt", &[x.clone(), z.clone()], false) || st.facts.holds("int.le", &[z, x], true)
    }

    fn rel(&self, st: &St, op: &str, a: &Term, b: &Term) -> bool {
        st.facts.holds(op, &[a.clone(), b.clone()], true)
    }

    fn eq_known(&self, st: &St, a: &Term, b: &Term, truth: bool) -> bool {
        st.facts.holds("int.eq", &[a.clone(), b.clone()], truth) || st.facts.holds("int.eq", &[b.clone(), a.clone()], truth)
    }

    /// W4: the domain of a partial primitive is established by a dominating test.
    fn domain(&self, p: &str, ts: &[Ty], args: &[Atom], tys: &[Ty], st: &St) -> R<()> {
        let terms: Vec<Term> = args.iter().map(|a| self.term(st, a)).collect();
        let neg = |q: &str| Term::P(q.to_string(), ts.to_vec(), terms.clone());
        let need = |ok: bool, what: &str| if ok { Ok(()) } else { Err(format!("W4: {p} is not dominated by {what}")) };
        let len_of = |x: &Term, c: &Ty| {
            let lp = match c {
                Ty::Bytes => "bytes.len",
                Ty::Arr(_) => "arr.len",
                _ => "list.len",
            };
            let targs = match c {
                Ty::Arr(e) | Ty::List(e) => vec![(**e).clone()],
                _ => vec![],
            };
            Term::P(lp.to_string(), targs, vec![x.clone()])
        };
        match p {
            "int.add" | "int.sub" | "int.mul" | "int.neg" | "int.abs" | "int.cast" => {
                need(self.known(st, &neg(&format!("int.bad.{}", &p[4..])), false), &format!("its test int.bad.{}", &p[4..]))
            }
            "int.shl" | "int.shr" => need(self.known(st, &neg("int.bad.shift"), false), "its test int.bad.shift"),
            "int.quo" | "int.rem" => {
                need(self.eq_known(st, &terms[1], &Term::K(Lit::Int(0)), false), "a test of the divisor against 0")?;
                need(self.known(st, &neg("int.bad.quo"), false), "its test int.bad.quo")
            }
            "f64.add" | "f64.sub" | "f64.mul" | "f64.div" | "f64.sqrt" | "f64.trunc" | "f64.to_f32" | "f32.add" | "f32.sub" | "f32.mul" | "f32.div" => {
                let q = format!("{}.bad.{}", &p[..3], &p[4..]);
                need(self.known(st, &neg(&q), false), &format!("its test {q}"))
            }
            "bytes.get" | "list.get" | "arr.get" | "list.set" | "arr.set" => {
                need(self.nonneg(st, &args[1], &tys[1]), "a test that the index is not negative")?;
                let n = len_of(&terms[0], &tys[0]);
                need(self.rel(st, "int.lt", &terms[1], &n), "a test that the index is below the length")
            }
            "bytes.slice" => {
                need(self.nonneg(st, &args[1], &tys[1]), "a test that the start is not negative")?;
                need(self.rel(st, "int.le", &terms[1], &terms[2]), "a test that the start is not after the end")?;
                let n = len_of(&terms[0], &tys[0]);
                need(self.rel(st, "int.le", &terms[2], &n), "a test that the end is not after the length")
            }
            "list.pop" => {
                let n = len_of(&terms[0], &tys[0]);
                need(self.rel(st, "int.lt", &Term::K(Lit::Int(0)), &n) || self.eq_known(st, &n, &Term::K(Lit::Int(0)), false), "a test that the list is not empty")
            }
            "sum.get" => {
                let tag = Term::P("sum.tag".into(), ts.to_vec(), vec![terms[0].clone()]);
                need(self.eq_known(st, &tag, &terms[1], true), "a test of its tag")
            }
            "arr.new" => need(self.nonneg(st, &args[0], &tys[0]), "a test that the length is not negative"),
            "mem.get" | "mem.set" => need(self.nonneg(st, &args[0], &tys[0]), "a test that the index is not negative"),
            _ => Ok(()),
        }
    }

    /// Check a statement list; returns whether it can complete normally.
    fn block(&mut self, b: &[Stmt], st: &mut St) -> R<bool> {
        let saved: Vec<u32> = self.locals.keys().copied().collect();
        let mut falls = true;
        for s in b {
            if !falls {
                return Err("W2: a statement follows one that cannot complete".into());
            }
            falls = self.stmt(s, st)?;
        }
        // leaving the list: its own owned locals must have been consumed
        let inner: Vec<u32> = self.locals.keys().copied().filter(|k| !saved.contains(k)).collect();
        if falls {
            for k in &inner {
                if st.own.get(k) == Some(&Own::Live) {
                    return Err(format!("W3: v{k} is not consumed before the end of its statement list"));
                }
            }
        }
        for k in inner {
            self.locals.remove(&k);
            self.defs.remove(&k);
            st.own.remove(&k);
            self.forget(st, k);
        }
        Ok(falls)
    }

    fn same_own(a: &St, b: &St, what: &str) -> R<()> {
        if a.own != b.own || a.gown != b.gown {
            return Err(format!("W3: the paths that meet {what} hold different owned values ({:?} and {:?})", a.own, b.own));
        }
        Ok(())
    }

    /// The state where two paths meet: the facts both know, and a new
    /// epoch for each place whose epochs differ.
    fn meet(&mut self, a: &St, b: &St) -> St {
        let mut r = a.clone();
        r.facts.retain(|f| b.facts.contains(f));
        let keys: BTreeSet<Key> = a.epochs.keys().chain(b.epochs.keys()).copied().collect();
        for k in keys {
            if a.epoch(k) != b.epoch(k) {
                self.renew(&mut r, k);
            }
        }
        r
    }

    fn stmt(&mut self, s: &Stmt, st: &mut St) -> R<bool> {
        match s {
            Stmt::Let(a) => {
                let t = self.use_atom(a, true, st, true)?;
                let term = self.term(st, a);
                let i = self.bind(t, false, st);
                self.defs.insert(i, term);
            }
            Stmt::Prim(p, ts, args) => {
                let mut tys = vec![];
                for a in args.iter() {
                    tys.push(self.aty(a)?);
                }
                let rs = signature(p, ts, &tys, args).map_err(|e| format!("W1: {e}"))?;
                self.domain(p, ts, args, &tys, st)?;
                for (k, a) in args.iter().enumerate() {
                    self.use_atom(a, !reads(p, k), st, false)?;
                }
                // a copy is equal to its operand, so facts about the one hold of the other
                let term = if p == "copy" { self.term(st, &args[0]) } else { Term::P(p.clone(), ts.clone(), args.iter().map(|a| self.term(st, a)).collect()) };
                // `mem.get` reads memory that `mem.set` changes, so its
                // result is a value of its own, not a function of its operand
                let single = rs.len() == 1 && p != "mem.get";
                for t in rs {
                    let i = self.bind(t, false, st);
                    if single {
                        self.defs.insert(i, term.clone());
                    }
                }
            }
            Stmt::Call(f, args) | Stmt::Catch(f, args) => {
                let (params, ret) = self.sig(f)?;
                if params.len() != args.len() {
                    return Err(format!("W1: {f:?} takes {} arguments, found {}", params.len(), args.len()));
                }
                for (k, a) in args.iter().enumerate() {
                    if matches!(a, Atom::G(_)) {
                        return Err("W6: an operand of call or catch is a global".into());
                    }
                    let t = self.use_atom(a, true, st, false)?;
                    if !fits(&t, &params[k]) {
                        return Err(format!("W1: argument {k} of type {t:?} is not within {:?}", params[k]));
                    }
                }
                self.globals_back(st, "a call")?;
                // a callee may set any global
                for i in 0..self.m.globals.len() as u32 {
                    self.renew(st, Key::G(i));
                }
                let rt = if matches!(s, Stmt::Call(..)) { ret } else { Ty::Sum(vec![ret, Ty::halt()]) };
                self.bind(rt, false, st);
            }
            Stmt::Var(t, a) => {
                let at = self.use_atom(a, true, st, true)?;
                if !fits(&at, t) {
                    return Err(format!("W1: var of type {t:?} initialised with {at:?}"));
                }
                // a `var` declared in a loop body begins anew on each iteration
                let i = self.bind(t.clone(), true, st);
                self.renew(st, Key::L(i));
            }
            Stmt::Set(p, a) => {
                let at = self.use_atom(a, true, st, false)?;
                match p {
                    Place::V(i) => {
                        let Some(l) = self.locals.get(i).cloned() else {
                            return Err(format!("W1: set of v{i}, which is not in scope"));
                        };
                        if l.param || !l.var {
                            return Err(format!("W6: set names v{i}, which is not a var local"));
                        }
                        if !fits(&at, &l.ty) {
                            return Err(format!("W1: set v{i} of type {:?} to {at:?}", l.ty));
                        }
                        if l.ty.owned() {
                            if st.own.get(i) == Some(&Own::Live) {
                                return Err(format!("W3: set v{i} while it still holds a value; drop it first"));
                            }
                            st.own.insert(*i, Own::Live);
                        }
                        self.renew(st, Key::L(*i));
                    }
                    Place::G(i) => {
                        let Some((gt, _)) = self.m.globals.get(*i as usize) else {
                            return Err(format!("W1: set of the unknown global g{i}"));
                        };
                        if !fits(&at, gt) {
                            return Err(format!("W1: set g{i} of type {gt:?} to {at:?}"));
                        }
                        if gt.owned() {
                            if st.gown.get(i) == Some(&Own::Live) {
                                return Err(format!("W3: set g{i} while it still holds a value"));
                            }
                            st.gown.insert(*i, Own::Live);
                        }
                        self.renew(st, Key::G(*i));
                    }
                }
            }
            Stmt::If(c, t, e) => {
                let ct = self.use_atom(c, false, st, false)?;
                if ct != Ty::Bool {
                    return Err(format!("W1: if on {ct:?}"));
                }
                let cterm = self.term(st, c);
                let mut st1 = st.clone();
                st1.facts.push((cterm.clone(), true));
                let f1 = self.block(t, &mut st1)?;
                let mut st2 = st.clone();
                st2.facts.push((cterm, false));
                let f2 = self.block(e, &mut st2)?;
                match (f1, f2) {
                    (true, true) => {
                        Self::same_own(&st1, &st2, "after an if")?;
                        *st = self.meet(&st1, &st2);
                    }
                    (true, false) => *st = st1,
                    (false, true) => *st = st2,
                    (false, false) => return Ok(false),
                }
            }
            Stmt::Block(b) | Stmt::Loop(b) => {
                let is_loop = matches!(s, Stmt::Loop(_));
                let entry = st.clone();
                let set_in_body = self.sets(b);
                // at the head of a loop, a place the body may set holds the
                // value of any earlier iteration
                let mut st_in = st.clone();
                if is_loop {
                    for k in &set_in_body {
                        self.renew(&mut st_in, *k);
                    }
                }
                self.exits.push(Exit { scope: self.locals.keys().copied().collect(), breaks: vec![] });
                let falls = self.block(b, &mut st_in)?;
                let ex = self.exits.pop().unwrap();
                if is_loop && falls {
                    Self::same_own(&entry, &st_in, "at the head of a loop")?;
                }
                let mut outs: Vec<St> = ex.breaks;
                if !is_loop && falls {
                    outs.push(st_in.clone());
                }
                let Some(first) = outs.first().cloned() else {
                    return Ok(false);
                };
                let mut r = first;
                for o in &outs[1..] {
                    Self::same_own(&r, o, "after a block or loop")?;
                    r = self.meet(&r, o);
                }
                // facts established inside do not leave, except those of a
                // block's fall-through when nothing breaks out of it
                let mut keep = entry.clone();
                keep.own = r.own.clone();
                keep.gown = r.gown.clone();
                for k in &set_in_body {
                    self.renew(&mut keep, *k);
                }
                if !is_loop && outs.len() == 1 && falls {
                    keep.facts = r.facts;
                    keep.epochs = r.epochs;
                }
                *st = keep;
            }
            Stmt::Break(d) => {
                let n = self.exits.len();
                if (*d as usize) >= n {
                    return Err(format!("W2: break {d} names no enclosing block or loop"));
                }
                let target = n - 1 - *d as usize;
                let mut out = st.clone();
                let scope = &self.exits[target].scope;
                for (k, o) in &st.own {
                    if !scope.contains(k) {
                        if *o == Own::Live {
                            return Err(format!("W3: v{k} is not consumed before break {d}"));
                        }
                        out.own.remove(k);
                    }
                }
                self.exits[target].breaks.push(out);
                return Ok(false);
            }
            Stmt::Ret(a) => {
                let t = self.use_atom(a, true, st, false)?;
                if !fits(&t, &self.f.ret) {
                    return Err(format!("W1: ret of {t:?} in a function returning {:?}", self.f.ret));
                }
                if let Some((k, _)) = st.own.iter().find(|(_, o)| **o == Own::Live) {
                    return Err(format!("W3: v{k} is not consumed before ret"));
                }
                self.globals_back(st, "ret")?;
                return Ok(false);
            }
            Stmt::Halt(a) => {
                let t = self.use_atom(a, true, st, false)?;
                if !fits(&t, &Ty::halt()) {
                    return Err(format!("W1: halt of {t:?}"));
                }
                return Ok(false);
            }
            Stmt::Out(a) => {
                let t = self.use_atom(a, false, st, false)?;
                if t != Ty::Bytes {
                    return Err(format!("W1: out of {t:?}"));
                }
            }
        }
        Ok(true)
    }

    fn sig(&self, f: &FnRef) -> R<(Vec<Ty>, Ty)> {
        let func = match f {
            FnRef::Key(k, i) => self.m.func(k, *i),
            FnRef::Rec(i) => self.g.fns.get(*i as usize),
        };
        let func = func.ok_or_else(|| format!("W5: no function {f:?}"))?;
        Ok((func.params.clone(), func.ret.clone()))
    }
}

impl Fun<'_> {
    /// The places a statement list may set: the `var` locals and globals it
    /// sets, and every global when it calls a function, since the callee
    /// may set any of them.
    fn sets(&self, b: &[Stmt]) -> Vec<Key> {
        fn go(b: &[Stmt], out: &mut Vec<Key>, calls: &mut bool) {
            for s in b {
                match s {
                    Stmt::Set(Place::V(i), _) => out.push(Key::L(*i)),
                    Stmt::Set(Place::G(i), _) => out.push(Key::G(*i)),
                    Stmt::Call(..) | Stmt::Catch(..) => *calls = true,
                    Stmt::If(_, t, e) => {
                        go(t, out, calls);
                        go(e, out, calls);
                    }
                    Stmt::Block(x) | Stmt::Loop(x) => go(x, out, calls),
                    _ => {}
                }
            }
        }
        let (mut out, mut calls) = (vec![], false);
        go(b, &mut out, &mut calls);
        if calls {
            out.extend((0..self.m.globals.len() as u32).map(Key::G));
        }
        out.sort();
        out.dedup();
        out
    }
}
