//! Building core functions: a statement builder that keeps the type of
//! every local and numbers locals in textual order, and a registry of
//! functions that groups them by their call graph, orders the members of
//! each group canonically and gives each group its key (W5) once every
//! function is built.
use super::ir::*;
use super::json::{fn_json_with, fref_json, group_key, sha256_hex};
use serde_json::json;
use std::collections::{BTreeMap, HashMap, HashSet};

pub type R<T> = Result<T, String>;

/// A reference to a function of the registry before keys exist.
pub fn sym(id: usize) -> FnRef {
    FnRef::Key(format!("#{id}"), 0)
}

/// The registry id a reference made by [`sym`] names.
pub fn sym_id(f: &FnRef) -> Option<usize> {
    match f {
        FnRef::Key(k, 0) => k.strip_prefix('#').and_then(|x| x.parse().ok()),
        _ => None,
    }
}

/// The least integer type holding both.
pub fn hull(a: &Ty, b: &Ty) -> Ty {
    match (a, b) {
        (Ty::Int(a, b), Ty::Int(c, d)) => Ty::Int(*a.min(c), *b.max(d)),
        _ => a.clone(),
    }
}

/// The widest core integer type.
pub fn wide() -> Ty {
    Ty::Int(INT_MIN, INT_MAX)
}

pub fn lit(v: i128) -> Atom {
    Atom::K(Ty::Int(v, v), Lit::Int(v))
}

/// The type of a loop counter.
pub fn count_t() -> Ty {
    Ty::Int(0, INT_MAX)
}

/// A statement builder for one function.
pub struct FB {
    pub params: Vec<Ty>,
    pub ret: Ty,
    next: u32,
    tys: HashMap<u32, Ty>,
    stack: Vec<Vec<Stmt>>,
    /// for each open statement list, whether it holds a statement that
    /// cannot complete; nothing after that statement is reached, so
    /// nothing more is added to the list (W2)
    dead: Vec<bool>,
    /// the last number given to a result of a statement that is left out
    ghost: u32,
    /// the fuel function, in a function that is charged fuel
    pub tick_fn: Option<FnRef>,
    /// fuel taken since the last statement that can halt or be observed;
    /// it is charged in one `call tick(n)` before the next such statement
    pending: u64,
}

impl FB {
    pub fn new(params: Vec<Ty>, ret: Ty) -> FB {
        let mut tys = HashMap::new();
        for (i, t) in params.iter().enumerate() {
            tys.insert(i as u32, t.clone());
        }
        FB { next: params.len() as u32, params, ret, tys, stack: vec![vec![]], dead: vec![false], ghost: u32::MAX, tick_fn: None, pending: 0 }
    }
    /// Charge n units of fuel at this point of the reference's evaluation.
    pub fn tick(&mut self, n: u64) {
        if self.tick_fn.is_some() {
            self.pending += n;
        }
    }
    /// Charge the pending fuel now.
    pub fn flush(&mut self) {
        if *self.dead.last().unwrap() {
            self.pending = 0;
            return;
        }
        if self.pending > 0
            && let Some(f) = self.tick_fn.clone()
        {
            let n = self.pending;
            self.pending = 0;
            self.stack.last_mut().unwrap().push(Stmt::Call(f, vec![lit(n as i128)]));
            self.fresh(Ty::Bool);
        }
    }
    pub fn arg(&self, i: usize) -> Atom {
        Atom::V(i as u32)
    }
    pub fn ty(&self, a: &Atom) -> Ty {
        match a {
            Atom::V(i) => self.tys.get(i).cloned().unwrap_or(Ty::unit()),
            Atom::K(t, _) => t.clone(),
            Atom::G(_) => Ty::unit(),
        }
    }
    pub fn local_ty(&self, i: u32) -> Ty {
        self.tys[&i].clone()
    }
    pub fn push(&mut self, s: Stmt) {
        if *self.dead.last().unwrap() {
            return;
        }
        // fuel is charged before every statement that can halt or be
        // observed; a group of pure statements never halts, so moving the
        // charge before it does not change the trace (ONE-4, fuel)
        if !matches!(s, Stmt::Let(_) | Stmt::Prim(..) | Stmt::Var(..) | Stmt::Set(..)) {
            self.flush();
        }
        let ends = !falls(std::slice::from_ref(&s));
        self.stack.last_mut().unwrap().push(s);
        if ends {
            *self.dead.last_mut().unwrap() = true;
        }
    }
    /// The next atom. A statement's results are numbered in the order of
    /// the statements, so a statement that is left out (it follows one
    /// that cannot complete) numbers its results apart, from the top of the
    /// range down, where no written statement refers to them.
    fn fresh(&mut self, t: Ty) -> Atom {
        let i = if *self.dead.last().unwrap() {
            self.ghost -= 1;
            self.ghost
        } else {
            self.next += 1;
            self.next - 1
        };
        self.tys.insert(i, t);
        Atom::V(i)
    }
    /// `vN = a`; `t` is the type of `a`.
    pub fn let_t(&mut self, a: Atom, t: Ty) -> Atom {
        self.push(Stmt::Let(a));
        self.fresh(t)
    }
    pub fn let_(&mut self, a: Atom) -> Atom {
        let t = self.ty(&a);
        self.let_t(a, t)
    }
    /// A global moved out or read into a local (its type is given).
    pub fn let_global(&mut self, g: u32, t: Ty) -> Atom {
        self.let_t(Atom::G(g), t)
    }
    pub fn prim(&mut self, p: &str, targs: Vec<Ty>, args: Vec<Atom>) -> R<Vec<Atom>> {
        let ops: Vec<Ty> = args.iter().map(|a| self.ty(a)).collect();
        let rs = super::wf::signature(p, &targs, &ops, &args).map_err(|e| format!("internal: {e}"))?;
        self.push(Stmt::Prim(p.to_string(), targs, args));
        Ok(rs.into_iter().map(|t| self.fresh(t)).collect())
    }
    /// A primitive with one result.
    pub fn p(&mut self, p: &str, targs: Vec<Ty>, args: Vec<Atom>) -> R<Atom> {
        let mut r = self.prim(p, targs, args)?;
        if r.len() != 1 {
            return Err(format!("internal: {p} binds {} results", r.len()));
        }
        Ok(r.pop().unwrap())
    }
    /// A primitive with no result.
    pub fn p0(&mut self, p: &str, targs: Vec<Ty>, args: Vec<Atom>) -> R<()> {
        let r = self.prim(p, targs, args)?;
        if !r.is_empty() {
            return Err(format!("internal: {p} binds results"));
        }
        Ok(())
    }
    pub fn call(&mut self, f: FnRef, ret: Ty, args: Vec<Atom>) -> Atom {
        self.push(Stmt::Call(f, args));
        self.fresh(ret)
    }
    pub fn catch(&mut self, f: FnRef, ret: Ty, args: Vec<Atom>) -> Atom {
        self.push(Stmt::Catch(f, args));
        self.fresh(Ty::Sum(vec![ret, Ty::halt()]))
    }
    pub fn var(&mut self, t: Ty, init: Atom) -> u32 {
        self.push(Stmt::Var(t.clone(), init));
        match self.fresh(t) {
            Atom::V(i) => i,
            _ => unreachable!(),
        }
    }
    pub fn set(&mut self, v: u32, a: Atom) {
        self.push(Stmt::Set(Place::V(v), a));
    }
    pub fn setg(&mut self, g: u32, a: Atom) {
        self.push(Stmt::Set(Place::G(g), a));
    }
    /// Begin a nested statement list.
    pub fn open(&mut self) {
        self.flush();
        self.stack.push(vec![]);
        // a list opened where nothing is reached is not reached either
        let d = *self.dead.last().unwrap();
        self.dead.push(d);
    }
    /// End a nested statement list and return it.
    pub fn close(&mut self) -> Vec<Stmt> {
        if falls(self.stack.last().unwrap()) {
            self.flush();
        }
        self.pending = 0;
        self.dead.pop();
        self.stack.pop().unwrap()
    }
    pub fn brk(&mut self, d: u32) {
        self.push(Stmt::Break(d));
    }
    pub fn ret(&mut self, a: Atom) {
        self.push(Stmt::Ret(a));
    }
    pub fn halt(&mut self, a: Atom) {
        self.push(Stmt::Halt(a));
    }
    pub fn out(&mut self, a: Atom) {
        self.push(Stmt::Out(a));
    }
    /// Whether the open statement list can still complete normally.
    pub fn falls(&self) -> bool {
        falls(self.stack.last().unwrap())
    }
    pub fn finish(mut self) -> Fn {
        self.flush();
        let body = self.stack.pop().unwrap();
        Fn { params: self.params, ret: self.ret, body }
    }

    // ---- structured statements built from closures

    /// `if c { t } else { e }`.
    pub fn if_(&mut self, c: Atom, t: impl FnOnce(&mut FB) -> R<()>, e: impl FnOnce(&mut FB) -> R<()>) -> R<()> {
        self.open();
        t(self)?;
        let tt = self.close();
        self.open();
        e(self)?;
        let ee = self.close();
        self.push(Stmt::If(c, tt, ee));
        Ok(())
    }
    /// `if c { t }`.
    pub fn when(&mut self, c: Atom, t: impl FnOnce(&mut FB) -> R<()>) -> R<()> {
        self.if_(c, t, |_| Ok(()))
    }
    /// `loop { body }`. Inside `body`, outside any block or loop it opens,
    /// `break 0` leaves this loop.
    pub fn loop_(&mut self, body: impl FnOnce(&mut FB) -> R<()>) -> R<()> {
        self.open();
        body(self)?;
        let b = self.close();
        self.push(Stmt::Loop(b));
        Ok(())
    }
    /// A loop counter, declared before the loop it counts.
    pub fn counter(&mut self) -> u32 {
        self.var(count_t(), lit(0))
    }
    /// Raise loop counter `n` by one, and halt with "loop bound" when it
    /// passes `bound`: the count bound of `shape::Count`. Written at the top
    /// level of a loop's statement list, with `n` declared and `bound`
    /// computed before the loop, it lets the loop pass this point at most
    /// `bound` times. The runtime places it where every pass is an
    /// iteration its own bound already limits, with a bound the loop cannot
    /// reach, so it never halts there.
    pub fn count(&mut self, n: u32, bound: Atom) -> R<()> {
        let nv = self.let_(Atom::V(n));
        let n1 = self.add_in(nv, lit(1), count_t())?;
        self.set(n, n1.clone());
        let over = self.int_cmp("int.lt", bound, n1)?;
        self.guard_halt(over, |fb| fb.trap_value(Atom::bytes(b"loop bound")))
    }
    /// Leave the innermost loop when `c` is false.
    pub fn exit_unless(&mut self, c: Atom) -> R<()> {
        self.if_(c, |_| Ok(()), |b| {
            b.brk(0);
            Ok(())
        })
    }
    /// Leave the innermost loop when `c` is true.
    pub fn exit_if(&mut self, c: Atom) -> R<()> {
        self.when(c, |b| {
            b.brk(0);
            Ok(())
        })
    }
    /// `for i in 0..n { body(i) }`, with i of type u64 and n known not to
    /// be negative. Inside `body`, `break 0` leaves the loop.
    pub fn for_n(&mut self, n: Atom, body: impl FnOnce(&mut FB, Atom) -> R<()>) -> R<()> {
        let i = self.var(Ty::u64(), lit(0));
        self.loop_(|b| {
            let iv = b.let_(Atom::V(i));
            let c = b.int_cmp("int.lt", iv.clone(), n)?;
            b.exit_unless(c)?;
            body(b, iv.clone())?;
            if b.falls() {
                let i1 = b.add_in(iv, lit(1), Ty::u64())?;
                b.set(i, i1);
            }
            Ok(())
        })
    }
    /// Move the value out of a `var` local.
    pub fn take(&mut self, v: u32) -> Atom {
        self.let_(Atom::V(v))
    }
    /// A copy of the value of a `var` local, which keeps its value.
    pub fn peek(&mut self, v: u32) -> R<Atom> {
        let t = self.local_ty(v);
        if t.owned() { self.p("copy", vec![t], vec![Atom::V(v)]) } else { Ok(self.let_(Atom::V(v))) }
    }
    /// Replace the value of a `var` local of an owned type, dropping the old one.
    pub fn replace(&mut self, v: u32, a: Atom) -> R<()> {
        if self.local_ty(v).owned() {
            let old = self.let_(Atom::V(v));
            self.drop(old)?;
        }
        self.set(v, a);
        Ok(())
    }

    // ---- small helpers used everywhere

    pub fn int_cmp(&mut self, op: &str, a: Atom, b: Atom) -> R<Atom> {
        let t = hull(&self.ty(&a), &self.ty(&b));
        self.p(op, vec![t], vec![a, b])
    }
    pub fn not(&mut self, a: Atom) -> R<Atom> {
        self.p("bool.not", vec![], vec![a])
    }
    pub fn cat(&mut self, a: Atom, b: Atom) -> R<Atom> {
        self.p("bytes.cat", vec![], vec![a, b])
    }
    /// The concatenation of several byte strings.
    pub fn cats(&mut self, parts: Vec<Atom>) -> R<Atom> {
        let mut it = parts.into_iter();
        let mut acc = it.next().unwrap_or(Atom::bytes(b""));
        for p in it {
            acc = self.cat(acc, p)?;
        }
        Ok(acc)
    }
    /// `if c { halt h } else {}` where `h` is built by `f` inside the branch.
    pub fn guard_halt(&mut self, c: Atom, f: impl FnOnce(&mut FB) -> R<Atom>) -> R<()> {
        self.open();
        let h = f(self)?;
        self.halt(h);
        let t = self.close();
        self.push(Stmt::If(c, t, vec![]));
        Ok(())
    }
    /// `if c {} else { halt (internal trap msg) }`: after it, c holds (W4).
    pub fn guard_halt_unless(&mut self, c: Atom, msg: &str) -> R<()> {
        self.open();
        let t = self.close();
        self.open();
        let h = self.trap_value(Atom::bytes(msg.as_bytes()))?;
        self.halt(h);
        let e = self.close();
        self.push(Stmt::If(c, t, e));
        Ok(())
    }
    /// A trap: `halt sum.mk[Halt](0, msg)`.
    pub fn trap_value(&mut self, msg: Atom) -> R<Atom> {
        self.p("sum.mk", vec![Ty::halt()], vec![lit(0), msg])
    }
    /// An atom of type t for code that follows a statement that cannot
    /// complete: such code is never reached and never written, so the atom
    /// is never defined.
    pub fn unreached(&mut self, t: Ty) -> Atom {
        debug_assert!(!self.falls());
        self.fresh(t)
    }
    pub fn unit(&mut self) -> R<Atom> {
        self.p("tup.mk", vec![Ty::unit()], vec![])
    }
    pub fn drop(&mut self, a: Atom) -> R<()> {
        let t = self.ty(&a);
        if t.owned() {
            self.p0("drop", vec![t], vec![a])?;
        }
        Ok(())
    }
    pub fn copy(&mut self, a: Atom) -> R<Atom> {
        let t = self.ty(&a);
        if t.owned() { self.p("copy", vec![t], vec![a]) } else { Ok(a) }
    }
    /// `a + b` at `t`, with its domain test; `a + b` must lie in `t`.
    pub fn add_in(&mut self, a: Atom, b: Atom, t: Ty) -> R<Atom> {
        self.arith_in("add", a, b, t)
    }
    pub fn sub_in(&mut self, a: Atom, b: Atom, t: Ty) -> R<Atom> {
        self.arith_in("sub", a, b, t)
    }
    pub fn mul_in(&mut self, a: Atom, b: Atom, t: Ty) -> R<Atom> {
        self.arith_in("mul", a, b, t)
    }
    /// An operation whose result is known to lie in `t`: the domain test is
    /// still placed (W4), and its branch halts with an internal trap that a
    /// correct lowering never reaches.
    pub fn arith_in(&mut self, op: &str, a: Atom, b: Atom, t: Ty) -> R<Atom> {
        let c = self.p(&format!("int.bad.{op}"), vec![t.clone()], vec![a.clone(), b.clone()])?;
        self.guard_halt(c, |fb| fb.trap_value(Atom::bytes(b"internal: arithmetic outside its proven range")))?;
        self.p(&format!("int.{op}"), vec![t], vec![a, b])
    }
    /// A cast to `t` of a value known to lie in `t`.
    pub fn cast_in(&mut self, a: Atom, t: Ty) -> R<Atom> {
        if self.ty(&a).within(&t) {
            return Ok(a);
        }
        let c = self.p("int.bad.cast", vec![t.clone()], vec![a.clone()])?;
        self.guard_halt(c, |fb| fb.trap_value(Atom::bytes(b"internal: cast outside its proven range")))?;
        self.p("int.cast", vec![t], vec![a])
    }
    /// The quotient and remainder of `a` by a positive literal `d`, both
    /// known to lie in `t`.
    pub fn divmod_lit(&mut self, a: Atom, d: i128, t: Ty) -> R<(Atom, Atom)> {
        let z = self.int_cmp("int.eq", lit(d), lit(0))?;
        self.guard_halt(z, |fb| fb.trap_value(Atom::bytes(b"internal: division by zero")))?;
        let c = self.p("int.bad.quo", vec![t.clone()], vec![a.clone(), lit(d)])?;
        self.guard_halt(c, |fb| fb.trap_value(Atom::bytes(b"internal: quotient outside its proven range")))?;
        let q = self.p("int.quo", vec![t.clone()], vec![a.clone(), lit(d)])?;
        let r = self.p("int.rem", vec![t], vec![a, lit(d)])?;
        Ok((q, r))
    }
    /// `bytes.get(b, i)` with its domain tests; the index must be known not
    /// to be negative by its type.
    pub fn byte_at(&mut self, b: Atom, i: Atom) -> R<Atom> {
        let n = self.p("bytes.len", vec![], vec![b.clone()])?;
        let c = self.int_cmp("int.lt", i.clone(), n)?;
        self.open();
        let t = self.close();
        self.open();
        let h = self.trap_value(Atom::bytes(b"internal: byte index"))?;
        self.halt(h);
        let e = self.close();
        self.push(Stmt::If(c, t, e));
        self.p("bytes.get", vec![], vec![b, i])
    }
    /// `list.get(l, i)` with its domain tests, `i` not negative by its type.
    pub fn list_at(&mut self, l: Atom, i: Atom) -> R<Atom> {
        let Ty::List(e) = self.ty(&l) else { return Err("internal: list_at of a non-list".into()) };
        let e = (*e).clone();
        let n = self.p("list.len", vec![e.clone()], vec![l.clone()])?;
        let c = self.int_cmp("int.lt", i.clone(), n)?;
        self.open();
        let t = self.close();
        self.open();
        let h = self.trap_value(Atom::bytes(b"internal: list index"))?;
        self.halt(h);
        let x = self.close();
        self.push(Stmt::If(c, t, x));
        self.p("list.get", vec![e], vec![l, i])
    }
}

/// Whether a statement list can complete normally.
pub fn falls(b: &[Stmt]) -> bool {
    fn breaks_to(b: &[Stmt], depth: u32) -> bool {
        b.iter().any(|s| match s {
            Stmt::Break(d) => *d == depth,
            Stmt::If(_, t, e) => breaks_to(t, depth) || breaks_to(e, depth),
            Stmt::Block(x) | Stmt::Loop(x) => breaks_to(x, depth + 1),
            _ => false,
        })
    }
    fn stmt_falls(s: &Stmt) -> bool {
        match s {
            Stmt::Break(_) | Stmt::Ret(_) | Stmt::Halt(_) => false,
            Stmt::If(_, t, e) => falls(t) || falls(e),
            Stmt::Block(x) => falls(x) || breaks_to(x, 0),
            Stmt::Loop(x) => breaks_to(x, 0),
            _ => true,
        }
    }
    b.iter().all(stmt_falls)
}

/// The functions of a module being built, named by registry ids until the
/// module is finished.
#[derive(Default)]
pub struct Fns {
    pub fns: Vec<Option<Fn>>,
    pub sigs: Vec<(Vec<Ty>, Ty)>,
    pub names: Vec<String>,
    by_name: HashMap<String, usize>,
    pub globals: Vec<(Ty, Option<Lit>, String)>,
    gnames: HashMap<String, u32>,
}

impl Fns {
    /// Declare a function; it is defined later with [`Fns::define`].
    pub fn declare(&mut self, name: &str, params: Vec<Ty>, ret: Ty) -> usize {
        let id = self.fns.len();
        self.fns.push(None);
        self.sigs.push((params, ret));
        self.names.push(name.to_string());
        id
    }
    pub fn define(&mut self, id: usize, f: Fn) {
        self.fns[id] = Some(f);
    }
    /// The id of the function named `name`, declaring it with `params` and
    /// `ret` and defining it with `build` on first use.
    pub fn get(&mut self, name: &str, params: Vec<Ty>, ret: Ty, build: impl FnOnce(&mut Fns, FB) -> R<Fn>) -> R<usize> {
        if let Some(id) = self.by_name.get(name) {
            return Ok(*id);
        }
        let id = self.declare(name, params.clone(), ret.clone());
        self.by_name.insert(name.to_string(), id);
        let f = build(self, FB::new(params, ret))?;
        self.define(id, f);
        Ok(id)
    }
    pub fn lookup(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }
    pub fn name(&mut self, name: &str, id: usize) {
        self.by_name.insert(name.to_string(), id);
    }
    pub fn ret(&self, id: usize) -> Ty {
        self.sigs[id].1.clone()
    }
    /// A global, created on first use.
    pub fn global(&mut self, name: &str, t: Ty, init: Option<Lit>) -> u32 {
        if let Some(g) = self.gnames.get(name) {
            return *g;
        }
        let g = self.globals.len() as u32;
        self.globals.push((t, init, name.to_string()));
        self.gnames.insert(name.to_string(), g);
        g
    }

    /// Group the functions reachable from `entry` by their call graph (one
    /// group per strongly connected component), give each group its key,
    /// and write the module.
    pub fn finish(self, entry: usize) -> R<Module> {
        let n = self.fns.len();
        let mut edges: Vec<Vec<usize>> = vec![vec![]; n];
        for (i, f) in self.fns.iter().enumerate() {
            let f = f.as_ref().ok_or_else(|| format!("internal: function {} is declared but not defined", self.names[i]))?;
            let mut out = HashSet::new();
            calls(&f.body, &mut out);
            let mut v: Vec<usize> = out.into_iter().collect();
            v.sort();
            edges[i] = v;
        }
        // Tarjan's algorithm from the entry; components come out callees first
        struct T<'a> {
            edges: &'a [Vec<usize>],
            index: Vec<Option<usize>>,
            low: Vec<usize>,
            on: Vec<bool>,
            stack: Vec<usize>,
            next: usize,
            comps: Vec<Vec<usize>>,
        }
        fn visit(t: &mut T, v: usize) {
            t.index[v] = Some(t.next);
            t.low[v] = t.next;
            t.next += 1;
            t.stack.push(v);
            t.on[v] = true;
            for k in 0..t.edges[v].len() {
                let w = t.edges[v][k];
                match t.index[w] {
                    None => {
                        visit(t, w);
                        t.low[v] = t.low[v].min(t.low[w]);
                    }
                    Some(iw) if t.on[w] => t.low[v] = t.low[v].min(iw),
                    _ => {}
                }
            }
            if Some(t.low[v]) == t.index[v] {
                let mut c = vec![];
                loop {
                    let w = t.stack.pop().unwrap();
                    t.on[w] = false;
                    c.push(w);
                    if w == v {
                        break;
                    }
                }
                c.sort();
                t.comps.push(c);
            }
        }
        let mut t = T { edges: &edges, index: vec![None; n], low: vec![0; n], on: vec![false; n], stack: vec![], next: 0, comps: vec![] };
        stacker::maybe_grow(256 * 1024, 16 * 1024 * 1024, || visit(&mut t, entry));
        let comps = t.comps;
        let mut place: HashMap<usize, (String, u32)> = HashMap::new();
        let mut defs = BTreeMap::new();
        let mut names = serde_json::Map::new();
        let mut fns = self.fns;
        for comp in comps {
            let pos: HashMap<usize, u32> = comp.iter().enumerate().map(|(k, id)| (*id, k as u32)).collect();
            let mut members = vec![];
            for id in &comp {
                let mut f = fns[*id].take().unwrap();
                rewrite(&mut f.body, &pos, &place)?;
                members.push(f);
            }
            // the members in canonical order, bisimilar members as one
            let (order, class) = canonical_order(&members);
            let mut g = Group { fns: vec![] };
            for &k in &order {
                let mut f = members[k].clone();
                renumber(&mut f.body, &class);
                g.fns.push(f);
            }
            let key = group_key(&g);
            for (k, id) in comp.iter().enumerate() {
                place.insert(*id, (key.clone(), class[k]));
            }
            let member_names = (0..order.len() as u32).map(|j| {
                let mut ns: Vec<&str> = comp.iter().enumerate().filter(|(k, _)| class[*k] == j).map(|(_, id)| self.names[*id].as_str()).collect();
                ns.sort();
                serde_json::Value::from(ns.join(" = "))
            });
            names.insert(key.clone(), serde_json::Value::Array(member_names.collect()));
            defs.insert(key, g);
        }
        let entry = place[&entry].clone();
        Ok(Module {
            globals: self.globals.iter().map(|(t, l, _)| (t.clone(), l.clone())).collect(),
            defs,
            entry,
            names: serde_json::json!({
                "fns": serde_json::Value::Object(names),
                "globals": self.globals.iter().map(|g| g.2.clone()).collect::<Vec<_>>(),
            }),
        })
    }
}

fn calls(b: &[Stmt], out: &mut HashSet<usize>) {
    for s in b {
        match s {
            Stmt::Call(f, _) | Stmt::Catch(f, _) => {
                if let Some(i) = sym_id(f) {
                    out.insert(i);
                }
            }
            Stmt::If(_, t, e) => {
                calls(t, out);
                calls(e, out);
            }
            Stmt::Block(x) | Stmt::Loop(x) => calls(x, out),
            _ => {}
        }
    }
}

/// The canonical order of a group's members (`rec i` naming member i of
/// `fs`): an iterated digest. In round 0 each member's digest is the
/// SHA-256 of its canonical JSON with every reference `rec i` written
/// `["rec","?"]`; in each later round, with every `rec i` written
/// `["rec", d]`, d the digest of member i in the round before. After n
/// rounds (rounds 0 to n - 1, n the number of members) two members have the
/// same digest exactly when they are bisimilar: the same function up to
/// the members they call, which are bisimilar in turn. The group's members
/// are the distinct final digests in ascending order, so its key does not
/// depend on the order in which the lowering first used them, and
/// bisimilar members are one function. Returns, for each new position, the
/// member written there (the first with that digest), and, for each member,
/// its new position.
fn canonical_order(fs: &[Fn]) -> (Vec<usize>, Vec<u32>) {
    let n = fs.len();
    if n == 1 {
        return (vec![0], vec![0]);
    }
    let digest = |f: &Fn, rec: &dyn std::ops::Fn(u32) -> serde_json::Value| {
        let rf = |r: &FnRef| match r {
            FnRef::Rec(i) => json!(["rec", rec(*i)]),
            other => fref_json(other),
        };
        sha256_hex(&fn_json_with(f, &rf))
    };
    let mut d: Vec<String> = fs.iter().map(|f| digest(f, &|_| json!("?"))).collect();
    for _ in 1..n {
        let prev = d;
        d = fs.iter().map(|f| digest(f, &|i| json!(prev[i as usize]))).collect();
    }
    let mut distinct: Vec<&String> = d.iter().collect();
    distinct.sort();
    distinct.dedup();
    let class: Vec<u32> = d.iter().map(|x| distinct.binary_search(&x).unwrap() as u32).collect();
    let order = (0..distinct.len() as u32).map(|j| class.iter().position(|c| *c == j).unwrap()).collect();
    (order, class)
}

/// Write each `rec i` of a member as `rec class[i]`, its canonical position.
fn renumber(b: &mut [Stmt], class: &[u32]) {
    for s in b {
        match s {
            Stmt::Call(FnRef::Rec(i), _) | Stmt::Catch(FnRef::Rec(i), _) => *i = class[*i as usize],
            Stmt::If(_, t, e) => {
                renumber(t, class);
                renumber(e, class);
            }
            Stmt::Block(x) | Stmt::Loop(x) => renumber(x, class),
            _ => {}
        }
    }
}

fn rewrite(b: &mut [Stmt], group: &HashMap<usize, u32>, done: &HashMap<usize, (String, u32)>) -> R<()> {
    for s in b {
        match s {
            Stmt::Call(f, _) | Stmt::Catch(f, _) => {
                if let Some(i) = sym_id(f) {
                    *f = match group.get(&i) {
                        Some(k) => FnRef::Rec(*k),
                        None => {
                            let (k, j) = done.get(&i).ok_or("internal: a callee outside the finished groups")?;
                            FnRef::Key(k.clone(), *j)
                        }
                    };
                }
            }
            Stmt::If(_, t, e) => {
                rewrite(t, group, done)?;
                rewrite(e, group, done)?;
            }
            Stmt::Block(x) | Stmt::Loop(x) => rewrite(x, group, done)?,
            _ => {}
        }
    }
    Ok(())
}
