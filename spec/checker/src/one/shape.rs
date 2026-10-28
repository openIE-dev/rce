//! The bounded shapes of a core loop: the two forms of `loop` whose number
//! of iterations is bounded by values fixed before the loop starts. They
//! are the shapes that become an Iterate of the Form; a loop with neither
//! shape has no bound a reader can state before it runs.
//!
//! - **Count.** A `var` counter n, declared before the loop and set nowhere
//!   else in it, is raised by one at the top level of the loop's statement
//!   list, and the loop halts once the raised value passes a bound B fixed
//!   before the loop:
//!   `vA = let vN; vC = int.sadd(vA, 1); set vN = vC; vO = int.lt(B, vC); if vO {… halt} else {}`
//!   (`int.add` in place of `int.sadd` is the same shape). Every iteration
//!   that completes runs these statements, so at most B iterations
//!   complete. With `int.sadd`, B's greatest value must lie below the
//!   greatest value of the counter's type, or the counter could stop at its
//!   saturation below the bound. This is the shape `lower::run_loop`
//!   writes for RCE's own loops.
//! - **Measure.** A `var` i, declared before the loop, is read at the head
//!   of the loop and tested against an end H fixed before the loop, and the
//!   loop is left unless i < H:
//!   `vA = let vI; vC = int.lt(vA, H); if vC {} else {leave}` or
//!   `vA = let vI; vC = int.le(H, vA); if vC {leave} else {}`;
//!   every path that continues the loop then sets i to `int.add(vA, 1)`
//!   exactly once, and nothing else in the loop sets i. So i rises by one
//!   per iteration and stays below H. This is the shape of `for` over
//!   ranges and lists.
//!
//! The head of a measure may also be the start of a `block` that is the
//! loop's only statement; a `break 0` in that block continues the loop.
//! "Fixed before the loop" means an integer literal, or a local bound
//! before the loop that the loop does not set. A global is never fixed,
//! because a call may set it.
use super::ir::*;
use super::wf::signature;
use std::collections::HashMap;

type R<T> = Result<T, String>;

/// The bounded shape of a loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Count,
    Measure,
}

#[derive(Clone)]
enum Def {
    Var,
    Prim(String, Vec<Atom>),
    Other,
}

struct Info<'a> {
    defs: HashMap<u32, Def>,
    tys: HashMap<u32, Ty>,
    /// the first local each binding statement binds, by the statement's address
    first: HashMap<*const Stmt, u32>,
    /// each loop in textual order: its statement list and the first local
    /// bound inside it
    loops: Vec<(&'a [Stmt], u32)>,
}

/// The shape of every loop of `f`, in textual order; a loop with neither
/// shape gives the reason. `globals` are the types of the module's globals
/// and `callee` gives the result type of a function a call names.
pub fn loops(f: &Fn, globals: &[Ty], callee: &dyn std::ops::Fn(&FnRef) -> Option<Ty>) -> R<Vec<Result<Shape, String>>> {
    let mut info = Info { defs: HashMap::new(), tys: HashMap::new(), first: HashMap::new(), loops: vec![] };
    for (i, t) in f.params.iter().enumerate() {
        info.defs.insert(i as u32, Def::Other);
        info.tys.insert(i as u32, t.clone());
    }
    let mut next = f.params.len() as u32;
    let cx = Cx { globals, callee };
    number(&cx, &f.body, &mut next, &mut info)?;
    Ok(info.loops.iter().map(|(l, lo)| classify(&info, l, *lo)).collect())
}

/// Every loop of a module that has neither shape: the group key, the
/// function's position in its group, its name from the concept record,
/// and the reason.
pub fn unbounded(m: &Module) -> R<Vec<(String, u32, String, String)>> {
    let globals: Vec<Ty> = m.globals.iter().map(|g| g.0.clone()).collect();
    let mut out = vec![];
    for (k, g) in &m.defs {
        let callee = |r: &FnRef| {
            match r {
                FnRef::Key(k, i) => m.func(k, *i),
                FnRef::Rec(i) => g.fns.get(*i as usize),
            }
            .map(|f| f.ret.clone())
        };
        for (i, f) in g.fns.iter().enumerate() {
            for r in loops(f, &globals, &callee)? {
                if let Err(why) = r {
                    let name = m.names["fns"][k][i].as_str().unwrap_or("?").to_string();
                    out.push((k.clone(), i as u32, name, why));
                }
            }
        }
    }
    Ok(out)
}

struct Cx<'c> {
    globals: &'c [Ty],
    callee: &'c dyn std::ops::Fn(&FnRef) -> Option<Ty>,
}

fn aty(cx: &Cx, info: &Info, a: &Atom) -> R<Ty> {
    match a {
        Atom::V(i) => info.tys.get(i).cloned().ok_or_else(|| format!("v{i} is not bound")),
        Atom::G(i) => cx.globals.get(*i as usize).cloned().ok_or_else(|| format!("no global g{i}")),
        Atom::K(t, _) => Ok(t.clone()),
    }
}

fn number<'a>(cx: &Cx, b: &'a [Stmt], next: &mut u32, info: &mut Info<'a>) -> R<()> {
    for s in b {
        info.first.insert(s as *const Stmt, *next);
        let mut bind = |info: &mut Info<'a>, d: Def, t: Ty| {
            info.defs.insert(*next, d);
            info.tys.insert(*next, t);
            *next += 1;
        };
        match s {
            Stmt::Let(a) => {
                let t = aty(cx, info, a)?;
                bind(info, Def::Other, t);
            }
            Stmt::Var(t, _) => bind(info, Def::Var, t.clone()),
            Stmt::Prim(p, ts, args) => {
                let tys = args.iter().map(|a| aty(cx, info, a)).collect::<R<Vec<Ty>>>()?;
                let rs = signature(p, ts, &tys, args)?;
                let one = rs.len() == 1;
                for t in rs {
                    bind(info, if one { Def::Prim(p.clone(), args.clone()) } else { Def::Other }, t);
                }
            }
            Stmt::Call(f, _) | Stmt::Catch(f, _) => {
                let ret = (cx.callee)(f).ok_or_else(|| format!("no function {f:?}"))?;
                let t = if matches!(s, Stmt::Call(..)) { ret } else { Ty::Sum(vec![ret, Ty::halt()]) };
                bind(info, Def::Other, t);
            }
            Stmt::If(_, t, e) => {
                number(cx, t, next, info)?;
                number(cx, e, next, info)?;
            }
            Stmt::Block(x) => number(cx, x, next, info)?,
            Stmt::Loop(x) => {
                info.loops.push((x, *next));
                number(cx, x, next, info)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// How many statements of `b` set local v.
fn sets_of(b: &[Stmt], v: u32) -> usize {
    b.iter()
        .map(|s| match s {
            Stmt::Set(Place::V(i), _) if *i == v => 1,
            Stmt::If(_, t, e) => sets_of(t, v) + sets_of(e, v),
            Stmt::Block(x) | Stmt::Loop(x) => sets_of(x, v),
            _ => 0,
        })
        .sum()
}

/// Whether a statement list can complete, and the constructs its breaks
/// reach, numbered from the loop being classified (1) inward; a number of
/// 1 or less leaves that loop.
fn flow(b: &[Stmt], depth: i64) -> (bool, Vec<i64>) {
    let mut out = vec![];
    for s in b {
        let (f, mut t) = match s {
            Stmt::Break(d) => (false, vec![depth - *d as i64]),
            Stmt::Ret(_) | Stmt::Halt(_) => (false, vec![]),
            Stmt::If(_, t, e) => {
                let (a, mut x) = flow(t, depth);
                let (b, y) = flow(e, depth);
                x.extend(y);
                (a || b, x)
            }
            Stmt::Block(x) | Stmt::Loop(x) => {
                let (f, t) = flow(x, depth + 1);
                let to_it = t.contains(&(depth + 1));
                let falls = to_it || (f && matches!(s, Stmt::Block(_)));
                (falls, t.into_iter().filter(|x| *x != depth + 1).collect())
            }
            _ => (true, vec![]),
        };
        out.append(&mut t);
        if !f {
            return (false, out);
        }
    }
    (true, out)
}

/// Whether a statement list at `depth` cannot complete and every break in
/// it leaves the loop.
fn leaves(b: &[Stmt], depth: i64) -> bool {
    let (f, ts) = flow(b, depth);
    !f && ts.iter().all(|t| *t <= 1)
}

fn int_lit(a: &Atom) -> Option<i128> {
    match a {
        Atom::K(Ty::Int(..), Lit::Int(v)) => Some(*v),
        _ => None,
    }
}

fn is_var(info: &Info, v: u32) -> bool {
    matches!(info.defs.get(&v), Some(Def::Var))
}

/// An integer literal, or a local bound before the loop that the loop does not set.
fn fixed(a: &Atom, l: &[Stmt], lo: u32) -> bool {
    match a {
        Atom::K(Ty::Int(..), Lit::Int(_)) => true,
        Atom::V(x) => *x < lo && sets_of(l, *x) == 0,
        _ => false,
    }
}

fn first(info: &Info, s: &Stmt) -> u32 {
    info.first[&(s as *const Stmt)]
}

fn classify(info: &Info, l: &[Stmt], lo: u32) -> Result<Shape, String> {
    if count(info, l, lo) {
        return Ok(Shape::Count);
    }
    measure(info, l, lo).map(|_| Shape::Measure)
}

fn count(info: &Info, l: &[Stmt], lo: u32) -> bool {
    (0..l.len()).any(|k| count_at(info, l, lo, k).is_some())
}

/// The counter of a count bound starting at statement k of the loop's list.
fn count_at(info: &Info, l: &[Stmt], lo: u32, k: usize) -> Option<()> {
    let Stmt::Let(Atom::V(n)) = l.get(k)? else { return None };
    let n = *n;
    if !(n < lo && is_var(info, n)) {
        return None;
    }
    let a = first(info, &l[k]);
    let one = |xs: &[Atom]| xs.len() == 2 && xs[0] == Atom::V(a) && int_lit(&xs[1]) == Some(1);
    let mut k = k + 1;
    // the domain test of `int.add`, when the counter is raised with it
    if let Some(Stmt::Prim(p, _, xs)) = l.get(k)
        && p == "int.bad.add"
        && one(xs)
    {
        let q = first(info, &l[k]);
        let Stmt::If(Atom::V(q2), t, e) = l.get(k + 1)? else { return None };
        if *q2 != q || !e.is_empty() || !leaves(t, 1) {
            return None;
        }
        k += 2;
    }
    let Stmt::Prim(op, ts, xs) = l.get(k)? else { return None };
    let Some(Ty::Int(_, chi)) = ts.first() else { return None };
    if !(op == "int.sadd" || op == "int.add") || !one(xs) {
        return None;
    }
    let c = first(info, &l[k]);
    if *l.get(k + 1)? != Stmt::Set(Place::V(n), Atom::V(c)) {
        return None;
    }
    let Stmt::Prim(lt, _, ys) = l.get(k + 2)? else { return None };
    if lt != "int.lt" || ys.len() != 2 || ys[1] != Atom::V(c) || !fixed(&ys[0], l, lo) {
        return None;
    }
    let o = first(info, &l[k + 2]);
    let Stmt::If(Atom::V(o2), t, e) = l.get(k + 3)? else { return None };
    if *o2 != o || !e.is_empty() || !leaves(t, 1) || sets_of(l, n) != 1 {
        return None;
    }
    if op == "int.sadd" {
        let most = match &ys[0] {
            Atom::V(x) => info.tys.get(x).and_then(|t| t.bounds()).map(|b| b.1),
            b => int_lit(b),
        };
        if !most.is_some_and(|b| b < *chi) {
            return None;
        }
    }
    Some(())
}

fn measure(info: &Info, l: &[Stmt], lo: u32) -> R<()> {
    let (head, hd) = match l {
        [Stmt::Block(x)] => (&x[..], 2),
        _ => (l, 1),
    };
    let no = |w: &str| Err(format!("neither a count nor a measure: {w}"));
    if head.len() < 3 {
        return no("the loop has no head test");
    }
    let Stmt::Let(Atom::V(i)) = &head[0] else { return no("the loop does not start by reading a variable") };
    let i = *i;
    if !(i < lo && is_var(info, i)) {
        return no("the loop does not start by reading a var declared before it");
    }
    let a = first(info, &head[0]);
    let Stmt::Prim(op, _, xs) = &head[1] else { return no("the head has no test") };
    let lt_form = op == "int.lt" && xs.len() == 2 && xs[0] == Atom::V(a);
    let le_form = op == "int.le" && xs.len() == 2 && xs[1] == Atom::V(a);
    if !(lt_form || le_form) {
        return no("the head does not test the variable against an end");
    }
    let end = if lt_form { &xs[1] } else { &xs[0] };
    if !fixed(end, l, lo) {
        return no("the end is not fixed before the loop");
    }
    let c = first(info, &head[1]);
    let Stmt::If(Atom::V(c2), t, e) = &head[2] else { return no("the head's test is not followed by its branch") };
    let exit = if lt_form { e } else { t };
    if *c2 != c || !leaves(exit, hd) {
        return no("the head's branch does not leave the loop");
    }
    // every path that continues raises i by one exactly once
    let mut w = Raise { info, i, a, err: None };
    let (falls, _) = w.list(l, 1, 1);
    if let Some(e) = w.err {
        return no(&e);
    }
    if falls & 1 != 0 {
        return no("a path continues the loop without raising its variable");
    }
    Ok(())
}

/// The raises of a measure's variable along the paths of a loop: a state
/// is a set of counts, bit 0 for none and bit 1 for one.
struct Raise<'b, 'a> {
    info: &'b Info<'a>,
    i: u32,
    a: u32,
    err: Option<String>,
}

impl Raise<'_, '_> {
    /// The states in which `b` completes, and the breaks it leaves by with
    /// their states.
    fn list(&mut self, b: &[Stmt], depth: i64, mut st: u8) -> (u8, Vec<(i64, u8)>) {
        let mut out = vec![];
        for s in b {
            if st == 0 {
                break;
            }
            match s {
                Stmt::Set(Place::V(v), x) if *v == self.i => {
                    let raise = match x {
                        Atom::V(x) => match self.info.defs.get(x) {
                            Some(Def::Prim(p, args)) => {
                                p == "int.add"
                                    && args.len() == 2
                                    && ((args[0] == Atom::V(self.a) && int_lit(&args[1]) == Some(1)) || (args[1] == Atom::V(self.a) && int_lit(&args[0]) == Some(1)))
                            }
                            _ => false,
                        },
                        _ => false,
                    };
                    if !raise {
                        self.err.get_or_insert("the loop sets its variable to something other than its value at the head plus one".into());
                    } else if st & 2 != 0 {
                        self.err.get_or_insert("a path may raise the variable twice in one iteration".into());
                    }
                    st = 2;
                }
                Stmt::If(_, t, e) => {
                    let (a, mut x) = self.list(t, depth, st);
                    let (b, y) = self.list(e, depth, st);
                    x.extend(y);
                    out.extend(x);
                    st = a | b;
                }
                Stmt::Block(x) => {
                    let (f, bs) = self.list(x, depth + 1, st);
                    st = f;
                    for (t, s) in bs {
                        if t == depth + 1 {
                            st |= s;
                        } else {
                            out.push((t, s));
                        }
                    }
                }
                Stmt::Loop(x) => {
                    if sets_of(x, self.i) > 0 {
                        self.err.get_or_insert("the variable is set inside a nested loop".into());
                    }
                    let (_, ts) = flow(x, depth + 1);
                    let to_it = ts.contains(&(depth + 1));
                    out.extend(ts.into_iter().filter(|t| *t != depth + 1).map(|t| (t, st)));
                    if !to_it {
                        st = 0;
                    }
                }
                Stmt::Break(d) => {
                    out.push((depth - *d as i64, st));
                    st = 0;
                }
                Stmt::Ret(_) | Stmt::Halt(_) => st = 0,
                _ => {}
            }
        }
        (st, out)
    }
}
