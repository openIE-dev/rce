//! The lowering (one-emitter.md ONE-4, section 3.5; order of work step 3):
//! an admitted RCE module, read through its typed tree, written as a core
//! module whose run prints the module's trace (ONE-5).
//!
//! The lowering follows the reference interpreter (src/interp.rs) step by
//! step: the same order of evaluation, the same trap texts, and fuel taken
//! at the same points, so that evaluating the core prints the reference's
//! bytes. Each construct is lowered as the table of section 3.5 states:
//! every expression to A-normal form, every checked operation to its domain
//! test, a halt and the total primitive, `match` to a chain of tests,
//! bounded loops to a counter and a halt, `fail` to a return of the error
//! value or a halt, steps to one function per region.
//!
//! A construct this lowering does not yet cover is reported with an error
//! that names it; the module is then not covered, and nothing is printed.
//! The constructs covered are those of the counter-style programs: checked
//! integers, bools, text values, structs, events, errors, kinds, options and
//! error unions, state, steps, kept steps, `if`, `match`, bounded loops, `for`
//! over ranges and lists, functions and methods, and constants; and floats:
//! f64 and f32 arithmetic with its domain tests, the methods of 5.15 on
//! floats, casts between floats and integers, and the display and JSON of
//! floats through the runtime functions of `rtf.rs`; `for … par` with the
//! reduction tree of C.8; the methods of 5.15 on text through `rtt.rs`,
//! slices of lists and text, `in`, and the list methods `contains`, `get`,
//! `reverse`, `sum`, `pop`, `clear`, `sort` and `utf8`; list
//! comprehensions and repeated lists; maps, kept as lists of pairs in the
//! canonical order of 6.1 without duplicate keys, with their literals,
//! index, indexed assignment, iteration, display, JSON, equality and
//! methods; `<` on tuples and lists; refined types (5.1), whose fit tests
//! the predicate; context text (5.16), `C.lit` and `C.encode`; `&mut`
//! parameters and `&mut self`, by state passing (the function returns
//! their final values beside its result and the caller writes them back,
//! except after a union function's error, F.4.4); lambdas given to `map`,
//! `filter` and `fold`, defunctionalised into the loop that runs them, and
//! lambdas a `let` makes, whose body runs at each call; lane-wise
//! arithmetic on lists; generic functions and generic impls,
//! monomorphised: a call runs the instantiation of its function at its
//! arguments' types, which the typed tree types once per key (as it does a
//! call whose float arguments keep a width other than the declared one);
//! `set_of` and the set methods `tolist` and `len`; the sum of a list of
//! f64 that has an item; a call of a foreign function, which the
//! reference does not run: it traps, as arithmetic on operands no rule
//! joins (an f32 and an f64) does; and arena and device regions with a
//! bound, which add up the sizes of the canonical JSON of the values they
//! hold after each statement (R3); and iterator functions (5.14), each
//! call a frame and each run a function that the consumer calls under
//! `catch` with a cap on the items, as interp `gen_run` runs them, so that
//! a trap in a run is raised after the items it gave; and budgets (C.5),
//! each block a function called under `catch` with the budget in force,
//! with the cost of C.6 charged where the reference charges it in a module
//! that has a budget, the only place where cost is observed. Traits, which
//! hold signatures only,
//! lower to nothing, and each function is lowered when the trace first
//! reaches it. A statement that follows one that cannot complete (a halt,
//! a return, a jump) is never reached and is left out.
//!
//! Ownership (W3): an RCE variable holds its value until the end of its
//! scope, where it is dropped; every read of an owned variable is a `copy`;
//! a temporary is consumed by the operation that uses it.
use super::build::*;
use super::ir::*;
use super::rt;
use super::rtf;
use super::rtt;
use crate::ast::{self, Arg, Body, Expr, FnDecl, Item, Param, Pat, Stmt as S};
use crate::typed::{self, Binding, Callee, IntT, RType, Tables, Typed, addr};
use std::collections::HashMap;

/// Lower an admitted module to the core.
pub fn lower(m: &ast::Module) -> R<Module> {
    let t = Tables::new(m);
    let mut ty = typed::build(m);
    let insts = std::mem::take(&mut ty.insts);
    let mut lw = Lw::new(m, t, ty)?;
    lw.insts = insts;
    let entry = lw.module()?;
    lw.f.finish(entry)
}

/// A value: an atom and the type of the values it holds. A value whose core
/// type is owned is owned by whoever holds it and is consumed exactly once.
#[derive(Clone)]
struct Val {
    a: Atom,
    t: RType,
}

/// What an RCE local is in the core.
#[derive(Clone)]
enum Bound {
    /// a `var` local of the core, holding values of this type
    Var(u32, RType),
    /// an immutable value
    Val(Atom, RType),
}

struct Scope {
    /// the number of open blocks and loops where the scope begins
    depth: u32,
    /// owned values the scope ends: dropped at its end and before a jump out
    drops: Vec<Atom>,
}

struct LoopCx {
    /// the nesting index of the loop's `loop` and of its body's `block`
    brk: u32,
    cont: u32,
}

#[derive(Clone)]
enum Ret {
    /// a function of the module: its declared result type
    Fn(Option<ast::Ty>),
    /// a step body: `return` is not covered
    Step,
    /// an expression evaluated alone (a constant, a check, an event)
    None,
    /// the body of an iterator function: `return` ends the run, finished
    Gen,
}

/// The run of an iterator function (5.14, interp `gen_run`): the items go
/// to the buffer on top of the stack held by global `g` (a list of lists of
/// items), which the consumer pushes before the run and pops after it, so
/// that the items given before a trap are still there; `count` counts them
/// and `cap` is the number after which the run stops.
#[derive(Clone)]
struct GenCx {
    g: u32,
    stack_t: Ty,
    item: RType,
    count: u32,
    cap: Atom,
}

#[derive(Clone, Copy, PartialEq)]
enum FailCx {
    /// `fail e` returns the error value
    Union,
    /// `fail e` halts with the error's name and display
    Halt,
}

struct Fx {
    b: FB,
    scopes: Vec<Scope>,
    bound: HashMap<u32, Bound>,
    loops: Vec<LoopCx>,
    /// open blocks and loops
    depth: u32,
    ret: Ret,
    fail: FailCx,
    /// the core type of the union a `fail` builds (in a union function)
    union_t: Option<Ty>,
    /// the `var` locals of the `&mut` parameters (and of a `&mut self`,
    /// last), whose values the function returns beside its result
    muts: Vec<u32>,
    /// in an instantiation: the type its result is fitted to
    inst_ret: Option<RType>,
    /// in the body of an iterator function
    run: Option<GenCx>,
}

impl Fx {
    fn new(b: FB, ret: Ret, fail: FailCx) -> Fx {
        Fx { b, scopes: vec![Scope { depth: 0, drops: vec![] }], bound: HashMap::new(), loops: vec![], depth: 0, ret, fail, union_t: None, muts: vec![], inst_ret: None, run: None }
    }
    fn push_scope(&mut self) {
        self.scopes.push(Scope { depth: self.depth, drops: vec![] });
    }
    fn own(&mut self, a: Atom) {
        if self.b.ty(&a).owned() {
            self.scopes.last_mut().unwrap().drops.push(a);
        }
    }
    /// End the innermost scope, dropping what it holds (when the path falls
    /// through to its end).
    fn pop_scope(&mut self) -> R<()> {
        let s = self.scopes.pop().unwrap();
        if self.b.falls() {
            for a in s.drops.into_iter().rev() {
                self.b.drop(a)?;
            }
        }
        Ok(())
    }
    /// Drop every value held by the scopes opened inside the construct at
    /// nesting index `target` (all scopes when `target` is None).
    fn drops_to(&mut self, target: Option<u32>) -> R<()> {
        let mut ds = vec![];
        for s in self.scopes.iter().rev() {
            match target {
                Some(t) if s.depth <= t => break,
                _ => ds.extend(s.drops.iter().rev().cloned()),
            }
        }
        for a in ds {
            self.b.drop(a)?;
        }
        Ok(())
    }
    /// `break` out of the construct at nesting index `target`.
    fn break_to(&mut self, target: u32) -> R<()> {
        self.drops_to(Some(target))?;
        self.b.brk(self.depth - 1 - target);
        Ok(())
    }
    fn open_nest(&mut self) -> u32 {
        self.b.open();
        self.depth += 1;
        self.depth - 1
    }
    fn close_nest(&mut self) -> Vec<Stmt> {
        self.depth -= 1;
        self.b.close()
    }
}

/// Whether a module has a `with budget` anywhere.
fn has_budget(m: &ast::Module) -> bool {
    let mut found = false;
    let mut look = |b: &[S]| crate::check::walk_stmts(b, &mut |s| {
        if matches!(s, S::Arena(n, ..) if n == "$budget") {
            found = true;
        }
    });
    for it in &m.items {
        match it {
            Item::Fn(f) => {
                if let Body::Block(b) = &f.body {
                    look(b);
                }
            }
            Item::Impl { members, .. } => {
                for f in members {
                    if let Body::Block(b) = &f.body {
                        look(b);
                    }
                }
            }
            Item::Step { body: Body::Block(b), .. } => look(b),
            _ => {}
        }
    }
    found
}

/// The charges of evaluating a place (one for each field, component and
/// element it reads, interp `eval`).
fn place_charges(p: &Expr) -> i128 {
    match p {
        Expr::Member(x, _, _) | Expr::TupleMember(x, _, _) | Expr::Index(x, _, _) => 1 + place_charges(x),
        Expr::Borrow(_, x, _) => place_charges(x),
        _ => 0,
    }
}

/// The cost of an operator on a value of type t (interp `op_cost`, C.6.4).
fn op_cost(op: &str, t: &RType) -> i128 {
    match t {
        RType::F64 | RType::F32 => match op {
            "+" | "-" | "*" => 2,
            "/" | "%" => 8,
            _ => 1,
        },
        _ => 1,
    }
}

fn no<T>(what: impl std::fmt::Display) -> R<T> {
    Err(format!("not covered by the lowering: {what}"))
}

/// The name a trap message gives an integer type, when every name of the
/// set prints the same.
fn print_name(i: &IntT) -> R<String> {
    let mut ns: Vec<&str> = i.tags.names().into_iter().map(|n| if n == "lit" { "i64" } else { n }).collect();
    ns.dedup();
    if ns.len() == 1 { Ok(ns[0].to_string()) } else { no(format!("an integer whose type name is one of {ns:?} at run time")) }
}

/// The range a checked integer operation with this result type checks
/// against, and the name its trap message prints.
fn check_of(i: &IntT) -> R<(Ty, String)> {
    let Some((lo, hi)) = i.check_range() else {
        return no(format!("an integer operation checked against ranges that differ at run time ({:?})", i.tags.names()));
    };
    Ok((Ty::Int(lo, hi), print_name(i)?))
}

struct Lw<'m> {
    m: &'m ast::Module,
    t: Tables<'m>,
    ty: Typed,
    f: Fns,
    tick: usize,
    fuel: (u32, u32),
    out_g: u32,
    fns: HashMap<String, usize>,
    methods: HashMap<(String, String), usize>,
    /// the type of the impl of each method, by (type head, name)
    impl_tys: HashMap<(String, String), ast::Ty>,
    /// the instantiations of the typed tree not yet lowered, by key
    insts: HashMap<String, typed::Inst>,
    /// the lowered instantiations: their functions, and their parameters'
    /// and result's types
    inst_ids: HashMap<String, usize>,
    inst_sigs: HashMap<usize, (Vec<RType>, RType)>,
    /// while an instantiation is lowered: its type parameters
    gs: Vec<String>,
    /// a constant: its evaluation function, its "evaluated" flag and its value
    consts: HashMap<String, (usize, u32, u32)>,
    named: HashMap<String, Ty>,
    gens: HashMap<String, usize>,
    loop_limit: i128,
    /// the event sum: every declared event, in declaration order
    ev_t: Ty,
    /// the buffer stacks of iterator runs, by the core type of the items
    gen_stacks: HashMap<Ty, u32>,
    /// whether the module has a budget, the only place where cost (C.6)
    /// is observed: then cost is charged where the reference charges it
    costs: bool,
}

impl<'m> Lw<'m> {
    fn new(m: &'m ast::Module, t: Tables<'m>, ty: Typed) -> R<Self> {
        let mut f = Fns::default();
        let fuel = rt::fuel_globals(&mut f, crate::interp::fuel_limit());
        let tick = rt::tick(&mut f)?;
        let out_g = f.global("outputs", Ty::list(Ty::Bytes), None);
        let mut loop_limit = crate::interp::LOOP_LIMIT;
        for i in &m.items {
            if let Item::Const { name, e: Expr::Int(n, _), .. } = i
                && name == "LOOP_LIMIT"
                && *n > 0
            {
                loop_limit = *n;
            }
        }
        let mut lw = Lw { m, t, ty, f, tick, fuel, out_g, fns: HashMap::new(), methods: HashMap::new(), impl_tys: HashMap::new(), insts: HashMap::new(), inst_ids: HashMap::new(), inst_sigs: HashMap::new(), gs: vec![], consts: HashMap::new(), named: HashMap::new(), gens: HashMap::new(), loop_limit, ev_t: Ty::unit(), gen_stacks: HashMap::new(), costs: has_budget(m) };
        let mut evs = vec![];
        for e in lw.t.events.clone() {
            evs.push(lw.ct(&RType::Named(e))?);
        }
        lw.ev_t = if evs.is_empty() { Ty::Sum(vec![Ty::unit()]) } else { Ty::Sum(evs) };
        Ok(lw)
    }

    // ---- types

    fn ct(&mut self, t: &RType) -> R<Ty> {
        self.ct_d(t, 0)
    }

    fn ct_d(&mut self, t: &RType, d: usize) -> R<Ty> {
        if d > 24 {
            return no("a recursive type");
        }
        Ok(match t {
            RType::Never | RType::Closure(_) => Ty::unit(),
            // the arguments of the call, fitted: the frame each run starts from
            // (an iterator observed without being run has no frame)
            RType::Gen(key, _) if key.is_empty() => Ty::unit(),
            RType::Gen(key, _) => Ty::Tup(self.inst_params(key)?.iter().map(|p| self.ct_d(p, d + 1)).collect::<R<_>>()?),
            RType::Bool => Ty::Bool,
            RType::Int(i) => {
                if i.lo < INT_MIN || i.hi > INT_MAX {
                    return no(format!("an integer outside -2^63..2^64-1 ({}..{})", i.lo, i.hi));
                }
                Ty::Int(i.lo, i.hi)
            }
            RType::F64 => Ty::F64,
            RType::F32 => Ty::F32,
            RType::Text => Ty::Bytes,
            RType::Unit => Ty::unit(),
            RType::Tuple(ts) => Ty::Tup(ts.iter().map(|x| self.ct_d(x, d + 1)).collect::<R<_>>()?),
            RType::List(x) | RType::Set(x) => Ty::list(self.ct_d(x, d + 1)?),
            RType::Map(k, v) => Ty::list(Ty::Tup(vec![self.ct_d(k, d + 1)?, self.ct_d(v, d + 1)?])),
            RType::Opt(x) => Ty::opt(self.ct_d(x, d + 1)?),
            RType::Union(a, b) => Ty::Sum(vec![self.ct_d(a, d + 1)?, self.ct_d(b, d + 1)?]),
            RType::Named(n) => {
                if let Some(t) = self.named.get(n) {
                    return Ok(t.clone());
                }
                let r = if let Some(vs) = self.t.kinds.get(n).cloned() {
                    let mut out = vec![];
                    for (_, fs) in vs {
                        let mut ts = vec![];
                        for (_, ft) in fs {
                            let rt = self.t.of_ty(&ft);
                            ts.push(self.ct_d(&rt, d + 1)?);
                        }
                        out.push(Ty::Tup(ts));
                    }
                    Ty::Sum(out)
                } else if let Some(fs) = self.t.fields(n) {
                    let mut ts = vec![];
                    for (_, ft) in fs {
                        let rt = self.t.of_ty(&ft);
                        ts.push(self.ct_d(&rt, d + 1)?);
                    }
                    Ty::Tup(ts)
                } else if n == "Missing" {
                    // the error a map's get gives when the module declares none
                    Ty::unit()
                } else {
                    return no(format!("the type {n}"));
                };
                self.named.insert(n.clone(), r.clone());
                r
            }
            RType::Range(_) => return no("a range as a value"),
            RType::Fn(_) => return no("a function as a value"),
            RType::Unknown(r) => return no(r),
        })
    }

    fn node_ty(&self, e: &Expr) -> RType {
        self.ty.ty(e)
    }

    // ---- function frames

    fn fx(&self, params: Vec<Ty>, ret: Ty, r: Ret, fail: FailCx, ticks: bool) -> Fx {
        let mut b = FB::new(params, ret);
        if ticks {
            b.tick_fn = Some(sym(self.tick));
        }
        Fx::new(b, r, fail)
    }

    fn trap(&mut self, fx: &mut Fx, msg: Atom) -> R<()> {
        let h = fx.b.trap_value(msg)?;
        fx.b.halt(h);
        Ok(())
    }

    /// `if c { halt (trap msg) }`, the message built in the branch.
    fn guard(&mut self, fx: &mut Fx, c: Atom, msg: impl FnOnce(&mut Self, &mut Fx) -> R<Atom>) -> R<()> {
        fx.b.open();
        let m = msg(self, fx)?;
        self.trap(fx, m)?;
        let t = fx.b.close();
        fx.b.push(Stmt::If(c, t, vec![]));
        Ok(())
    }

    /// Charge cost units (an atom of an integer type) where the reference
    /// charges them (interp `charge`); only in a module with a budget.
    fn charge(&mut self, fx: &mut Fx, units: Atom) -> R<()> {
        if !self.costs {
            return Ok(());
        }
        let u = fx.b.cast_in(units, Ty::u64())?;
        let f = rt::charge(&mut self.f)?;
        fx.b.call(sym(f), Ty::Bool, vec![u]);
        Ok(())
    }

    fn charge_k(&mut self, fx: &mut Fx, k: i128) -> R<()> {
        self.charge(fx, lit(k))
    }

    /// The charge of a built-in method (interp `builtin`): c_call, and in a
    /// sized call (a version 0.7 module) the method's work by its class
    /// (C.6.4): n + k for a linear method, n * ceil(log2 n) for sort, where
    /// n is the size of the receiver and k of the first argument, both read.
    fn charge_method(&mut self, fx: &mut Fx, m: &str, recv: (Atom, RType), arg: Option<(&Atom, &RType)>, sized: bool) -> R<()> {
        if !self.costs {
            return Ok(());
        }
        let kind = match &recv.1 {
            RType::Text => "text",
            RType::List(_) => "list",
            RType::Set(_) => "set",
            RType::Map(..) => "map",
            _ => "",
        };
        let class = if sized { crate::specread::work(kind, m) } else { crate::specread::Work::Constant };
        use crate::specread::Work;
        let w = match class {
            Work::Constant => lit(0),
            Work::Linear => {
                let n = self.size_of(fx, &recv.0, &recv.1)?;
                let k = match arg {
                    Some((a, t)) => self.size_of(fx, a, t)?,
                    None => lit(0),
                };
                fx.b.p("int.sadd", vec![Ty::u64()], vec![n, k])?
            }
            Work::NLogN => {
                let n = self.size_of(fx, &recv.0, &recv.1)?;
                let g = rt::ceil_log2(&mut self.f)?;
                let l = self.call_rt(fx, g, vec![n.clone()]);
                fx.b.p("int.smul", vec![Ty::u64()], vec![n, l])?
            }
        };
        let u = fx.b.p("int.sadd", vec![Ty::u64()], vec![w, lit(crate::conc::C_CALL)])?;
        self.charge(fx, u)
    }

    /// The size of a value for the work of a method (interp `size_of`): the
    /// bytes of a text, the items of a list or set, the entries of a map,
    /// and 0 for any other value.
    fn size_of(&mut self, fx: &mut Fx, a: &Atom, t: &RType) -> R<Atom> {
        let n = match t {
            RType::Text => fx.b.p("bytes.len", vec![], vec![a.clone()])?,
            RType::List(_) | RType::Set(_) | RType::Map(..) => {
                let ct = fx.b.ty(a);
                let Ty::List(e) = ct else { return Err("internal: a list of no list type".into()) };
                fx.b.p("list.len", vec![(*e).clone()], vec![a.clone()])?
            }
            _ => lit(0),
        };
        fx.b.cast_in(n, Ty::u64())
    }

    /// `4 + n` units for a list of n items (interp `charge(4 + len)`).
    fn charge_list(&mut self, fx: &mut Fx, n: Atom) -> R<()> {
        if !self.costs {
            return Ok(());
        }
        let n = fx.b.cast_in(n, Ty::Int(0, INT_MAX))?;
        let u = fx.b.p("int.sadd", vec![Ty::u64()], vec![n, lit(4)])?;
        self.charge(fx, u)
    }

    fn call_rt(&mut self, fx: &mut Fx, id: usize, args: Vec<Atom>) -> Atom {
        let r = self.f.ret(id);
        fx.b.call(sym(id), r, args)
    }

    /// A binary64 value known to be a binary32 value, as f32.
    fn round_to_f32(&mut self, fx: &mut Fx, x: Atom) -> R<Atom> {
        let c = fx.b.p("f64.bad.to_f32", vec![], vec![x.clone()])?;
        self.guard(fx, c, |_, _| Ok(Atom::bytes(b"internal: f32")))?;
        fx.b.p("f64.to_f32", vec![], vec![x])
    }

    fn dec(&mut self, fx: &mut Fx, v: Atom) -> R<Atom> {
        let d = rt::dec(&mut self.f)?;
        Ok(self.call_rt(fx, d, vec![v]))
    }

    /// "overflow: V is outside T" with V the exact result of `op`.
    fn ovf(&mut self, fx: &mut Fx, op: &str, args: Vec<Atom>, name: &str) -> R<Atom> {
        let v = if op == "value" {
            self.dec(fx, args[0].clone())?
        } else {
            let d = rt::dec_exact(&mut self.f, op)?;
            self.call_rt(fx, d, args)
        };
        fx.b.cats(vec![Atom::bytes(b"overflow: "), v, Atom::bytes(format!(" is outside {name}").as_bytes())])
    }

    /// A default value of a core type, for a variable assigned on every path
    /// before it is read.
    fn default(&mut self, fx: &mut Fx, t: &Ty) -> R<Atom> {
        Ok(match t {
            Ty::Bool => Atom::bool(false),
            Ty::Int(lo, hi) => Atom::K(t.clone(), Lit::Int(if *lo <= 0 && 0 <= *hi { 0 } else { *lo })),
            Ty::F64 => Atom::K(Ty::F64, Lit::F64(0)),
            Ty::F32 => Atom::K(Ty::F32, Lit::F32(0)),
            Ty::Bytes => Atom::bytes(b""),
            Ty::List(e) => fx.b.p("list.new", vec![(**e).clone()], vec![])?,
            Ty::Arr(e) => {
                let x = self.default(fx, e)?;
                fx.b.p("arr.new", vec![(**e).clone()], vec![lit(0), x])?
            }
            Ty::Tup(ts) => {
                let mut xs = vec![];
                for c in ts {
                    xs.push(self.default(fx, c)?);
                }
                fx.b.p("tup.mk", vec![t.clone()], xs)?
            }
            Ty::Sum(ts) => {
                let x = self.default(fx, &ts[0])?;
                fx.b.p("sum.mk", vec![t.clone()], vec![lit(0), x])?
            }
        })
    }

    /// A variable assigned on each branch of a construct.
    fn result_var(&mut self, fx: &mut Fx, t: &Ty) -> R<u32> {
        let d = self.default(fx, t)?;
        Ok(fx.b.var(t.clone(), d))
    }

    /// Store an owned or plain value into a result variable.
    fn store_var(&mut self, fx: &mut Fx, v: u32, a: Atom) -> R<()> {
        let t = fx.b.local_ty(v);
        if t.owned() {
            let old = fx.b.let_(Atom::V(v));
            fx.b.drop(old)?;
        }
        fx.b.set(v, a);
        Ok(())
    }

    /// Read a variable as an owned value (a copy) or a plain one.
    fn read_var(&mut self, fx: &mut Fx, v: u32) -> R<Atom> {
        let t = fx.b.local_ty(v);
        if t.owned() { fx.b.p("copy", vec![t], vec![Atom::V(v)]) } else { Ok(fx.b.let_(Atom::V(v))) }
    }

    /// Move a variable's value out (the variable is then empty).
    fn take_var(&mut self, fx: &mut Fx, v: u32) -> Atom {
        fx.b.let_(Atom::V(v))
    }

    // ---- conversions

    /// Retype or fit a value of type `from` to type `to` (interp `fit` when
    /// `check`: an integer outside the target's range traps with the target
    /// type's name). Consumes `a`.
    fn conv(&mut self, fx: &mut Fx, a: Atom, from: &RType, to: &RType, check: bool) -> R<Atom> {
        let ft = fx.b.ty(&a);
        let tt = self.ct(to)?;
        if *from == RType::Never && !fx.b.falls() {
            // a value of no type, after a statement that cannot complete
            return Ok(fx.b.unreached(tt));
        }
        if ft == tt || (matches!((&ft, &tt), (Ty::Int(..), Ty::Int(..))) && ft.within(&tt)) {
            return Ok(a);
        }
        match (from, to) {
            (RType::Int(_), RType::Int(ti)) => {
                let name = print_name(ti)?;
                if !check {
                    return no(format!("a join of integer types {} and {}", from.show(), to.show()));
                }
                let c = fx.b.p("int.bad.cast", vec![tt.clone()], vec![a.clone()])?;
                let a2 = a.clone();
                self.guard(fx, c, |lw, fx| lw.ovf(fx, "value", vec![a2], &name))?;
                fx.b.p("int.cast", vec![tt], vec![a])
            }
            (RType::Int(_), RType::F64) if check => fx.b.p("f64.of_int", vec![ft], vec![a]),
            (RType::Int(_), RType::F32) if check => {
                let g = rtf::int_to_f32(&mut self.f)?;
                Ok(self.call_rt(fx, g, vec![a]))
            }
            (RType::List(x), RType::List(y)) | (RType::Set(x), RType::Set(y)) => {
                let Ty::List(te) = &tt else { unreachable!() };
                let te = (**te).clone();
                if **x == RType::Never {
                    fx.b.drop(a)?;
                    return fx.b.p("list.new", vec![te], vec![]);
                }
                let Ty::List(fe) = &ft else { return Err("internal: a list of a non-list core type".into()) };
                let fe = (**fe).clone();
                let n = fx.b.p("list.len", vec![fe.clone()], vec![a.clone()])?;
                let out0 = fx.b.p("list.new", vec![te.clone()], vec![])?;
                let out = fx.b.var(tt.clone(), out0);
                let i = fx.b.var(Ty::u64(), lit(0));
                fx.b.open();
                let iv = fx.b.let_(Atom::V(i));
                let more = fx.b.int_cmp("int.lt", iv.clone(), n)?;
                fx.b.open();
                let th = fx.b.close();
                fx.b.open();
                fx.b.brk(1);
                let el = fx.b.close();
                fx.b.push(Stmt::If(more, th, el));
                let e = fx.b.list_at(a.clone(), iv.clone())?;
                let depth0 = fx.depth;
                fx.depth += 2;
                let e2 = self.conv(fx, e, x, y, check)?;
                fx.depth = depth0;
                let o = fx.b.let_(Atom::V(out));
                let o = fx.b.p("list.push", vec![te], vec![o, e2])?;
                fx.b.set(out, o);
                let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
                fx.b.set(i, i1);
                let body = fx.b.close();
                fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
                fx.b.drop(a)?;
                Ok(fx.b.let_(Atom::V(out)))
            }
            (RType::Map(k1, v1), RType::Map(k2, v2)) => {
                let Ty::List(te) = &tt else { unreachable!() };
                let te = (**te).clone();
                if **k1 == RType::Never {
                    fx.b.drop(a)?;
                    return fx.b.p("list.new", vec![te], vec![]);
                }
                // fitting keeps the value of each key, so the order holds
                let from = RType::List(Box::new(RType::Tuple(vec![(**k1).clone(), (**v1).clone()])));
                let to = RType::List(Box::new(RType::Tuple(vec![(**k2).clone(), (**v2).clone()])));
                self.conv(fx, a, &from, &to, check)
            }
            (RType::Opt(x), RType::Opt(y)) => {
                if **x == RType::Never {
                    fx.b.drop(a)?;
                    let u = fx.b.unit()?;
                    return fx.b.p("sum.mk", vec![tt], vec![lit(0), u]);
                }
                let fx_t = ft.clone();
                let tag = fx.b.p("sum.tag", vec![fx_t.clone()], vec![a.clone()])?;
                let some = fx.b.int_cmp("int.eq", tag, lit(1))?;
                let r = self.result_var(fx, &tt)?;
                fx.b.open();
                let p = fx.b.p("sum.get", vec![fx_t.clone()], vec![a.clone(), lit(1)])?;
                let p2 = self.conv(fx, p, x, y, check)?;
                let s = fx.b.p("sum.mk", vec![tt.clone()], vec![lit(1), p2])?;
                self.store_var(fx, r, s)?;
                let th = fx.b.close();
                fx.b.open();
                fx.b.drop(a.clone())?;
                let th2 = fx.b.close();
                fx.b.push(Stmt::If(some, th, th2));
                Ok(fx.b.let_(Atom::V(r)))
            }
            (RType::Tuple(xs), RType::Tuple(ys)) if xs.len() == ys.len() => {
                let cs = fx.b.prim("tup.open", vec![ft], vec![a])?;
                let mut out = vec![];
                for ((c, x), y) in cs.into_iter().zip(xs).zip(ys) {
                    out.push(self.conv(fx, c, x, y, check)?);
                }
                fx.b.p("tup.mk", vec![tt], out)
            }
            (RType::Union(a1, e1), RType::Union(a2, e2)) => {
                let r = self.result_var(fx, &tt)?;
                let parts = [((**a1).clone(), (**a2).clone()), ((**e1).clone(), (**e2).clone())];
                self.switch(fx, &a, &ft, 2, |lw, fx, k| {
                    let p = fx.b.p("sum.get", vec![ft.clone()], vec![a.clone(), lit(k as i128)])?;
                    let p2 = lw.conv(fx, p, &parts[k].0, &parts[k].1, check)?;
                    let s = fx.b.p("sum.mk", vec![tt.clone()], vec![lit(k as i128), p2])?;
                    lw.store_var(fx, r, s)
                })?;
                Ok(fx.b.let_(Atom::V(r)))
            }
            (x, RType::Union(a2, _)) if !matches!(x, RType::Union(..)) => {
                let v = self.conv(fx, a, x, a2, check)?;
                fx.b.p("sum.mk", vec![tt], vec![lit(0), v])
            }
            _ => no(format!("a conversion of {} to {}", from.show(), to.show())),
        }
    }

    /// interp `fit`: a value of type `from` given the declared type `to`.
    fn fit(&mut self, fx: &mut Fx, a: Atom, from: &RType, to: &ast::Ty) -> R<Val> {
        // a refined type (5.1): the base type's fit, then the predicate
        if let ast::Ty::Path(p, _) = to
            && p.len() == 1
            && let Some(Item::TypeAlias { ty, pred: Some(e), .. }) = self.t.types.get(&p[0]).copied()
        {
            let name = p[0].clone();
            let v = self.fit(fx, a, from, ty)?;
            let Some(base) = crate::refine::base_of(&self.t.alias(ty)) else { return no(format!("the refined type {name}")) };
            let Ok(pr) = crate::refine::compile(e, base) else { return no(format!("the predicate of {name}")) };
            let ok = self.holds(fx, &pr, &v)?;
            let bad = fx.b.not(ok)?;
            let msg = crate::refine::trap_text(&name);
            self.guard(fx, bad, move |_, _| Ok(Atom::bytes(msg.as_bytes())))?;
            return Ok(v);
        }
        let target = self.t.fit_generic(from, to, &self.gs);
        if let RType::Unknown(r) = &target {
            return no(r);
        }
        // an error value keeps its type: fit to a union keeps the error side
        let a2 = self.conv(fx, a, from, &target, true)?;
        Ok(Val { a: a2, t: target })
    }

    /// Whether the value v (read) satisfies the predicate p (refine `holds`).
    fn holds(&mut self, fx: &mut Fx, p: &crate::refine::P, v: &Val) -> R<Atom> {
        use crate::refine::{P, Tm};
        let term = |lw: &mut Self, fx: &mut Fx, t: &Tm| -> R<Option<Atom>> {
            Ok(match (t, &v.t) {
                (Tm::Lit(n), _) => Some(lit(*n)),
                (Tm::SelfV, RType::Int(_)) => Some(v.a.clone()),
                (Tm::Len, RType::Text) => Some(fx.b.p("bytes.len", vec![], vec![v.a.clone()])?),
                (Tm::Len, RType::List(_) | RType::Set(_) | RType::Map(..)) => {
                    let ct = fx.b.ty(&v.a);
                    let Ty::List(e) = &ct else { return Err("internal: len of a non-list".into()) };
                    Some(fx.b.p("list.len", vec![(**e).clone()], vec![v.a.clone()])?)
                }
                _ => {
                    let _ = lw;
                    None
                }
            })
        };
        match p {
            P::Cmp(op, a, b) => {
                let (Some(x), Some(y)) = (term(self, fx, a)?, term(self, fx, b)?) else { return Ok(Atom::bool(false)) };
                let (p, q, neg) = match *op {
                    "<" => (x, y, ("int.lt", false)),
                    "<=" => (x, y, ("int.le", false)),
                    ">" => (y, x, ("int.lt", false)),
                    ">=" => (y, x, ("int.le", false)),
                    "==" => (x, y, ("int.eq", false)),
                    _ => (x, y, ("int.eq", true)),
                };
                let r = fx.b.int_cmp(neg.0, p, q)?;
                if neg.1 { fx.b.not(r) } else { Ok(r) }
            }
            P::Cls(c) => {
                if v.t != RType::Text {
                    return Ok(Atom::bool(false));
                }
                if *c == "isempty" {
                    let n = fx.b.p("bytes.len", vec![], vec![v.a.clone()])?;
                    return fx.b.int_cmp("int.eq", n, lit(0));
                }
                let g = rtt::is_class(&mut self.f, c)?;
                Ok(self.call_rt(fx, g, vec![v.a.clone()]))
            }
            P::And(a, b) => {
                let x = self.holds(fx, a, v)?;
                let y = self.holds(fx, b, v)?;
                fx.b.p("bool.and", vec![], vec![x, y])
            }
            P::Or(a, b) => {
                let x = self.holds(fx, a, v)?;
                let y = self.holds(fx, b, v)?;
                fx.b.p("bool.or", vec![], vec![x, y])
            }
            P::Not(a) => {
                let x = self.holds(fx, a, v)?;
                fx.b.not(x)
            }
        }
    }

    /// interp `fit` into a place whose core type is that of the declared
    /// type: a parameter, a field, a result. A float the interpreter keeps
    /// at its own width (an f64 value given to an f32 field) has no core
    /// value of the declared type, and is not covered.
    fn fit_decl(&mut self, fx: &mut Fx, a: Atom, from: &RType, to: &ast::Ty) -> R<Val> {
        let v = self.fit(fx, a, from, to)?;
        let want = self.t.of_ty(to);
        if v.t == want {
            return Ok(v);
        }
        let a = self.conv(fx, v.a, &v.t, &want, false)?;
        Ok(Val { a, t: want })
    }

    // ---- generated per-type code: display, equality, positional JSON

    fn generated(&mut self, kind: &str, rt: &RType, params: Vec<Ty>, ret: Ty, body: impl FnOnce(&mut Self, &mut Fx) -> R<()>) -> R<usize> {
        let key = format!("{kind}:{}", rt.show());
        if let Some(id) = self.gens.get(&key) {
            return Ok(*id);
        }
        let id = self.f.declare(&key, params.clone(), ret.clone());
        self.gens.insert(key, id);
        let mut fx = self.fx(params, ret, Ret::None, FailCx::Halt, false);
        body(self, &mut fx)?;
        self.f.define(id, fx.b.finish());
        Ok(id)
    }

    /// The display of a value (interp `V::show`): `show:τ(v) -> bytes`, consuming v.
    fn show_fn(&mut self, rt: &RType) -> R<usize> {
        let t = self.ct(rt)?;
        let rt2 = rt.clone();
        self.generated("show", rt, vec![t.clone()], Ty::Bytes, move |lw, fx| {
            let v = fx.b.arg(0);
            let r = lw.show_body(fx, v, &rt2, &t)?;
            fx.b.ret(r);
            Ok(())
        })
    }

    fn show(&mut self, fx: &mut Fx, v: Atom, rt: &RType) -> R<Atom> {
        let id = self.show_fn(rt)?;
        Ok(self.call_rt(fx, id, vec![v]))
    }

    fn fields_show(&mut self, fx: &mut Fx, name: &str, fs: &[(String, ast::Ty)], tup: Atom, tt: &Ty) -> R<Atom> {
        if fs.is_empty() {
            fx.b.drop(tup)?;
            return Ok(Atom::bytes(name.as_bytes()));
        }
        let cs = fx.b.prim("tup.open", vec![tt.clone()], vec![tup])?;
        let mut parts = vec![Atom::bytes(format!("{name}(").as_bytes())];
        for (k, ((f, fty), c)) in fs.iter().zip(cs).enumerate() {
            let frt = self.t.of_ty(fty);
            let s = self.show(fx, c, &frt)?;
            let sep = if k == 0 { format!("{f}: ") } else { format!(", {f}: ") };
            parts.push(Atom::bytes(sep.as_bytes()));
            parts.push(s);
        }
        parts.push(Atom::bytes(b")"));
        fx.b.cats(parts)
    }

    fn show_body(&mut self, fx: &mut Fx, v: Atom, rt: &RType, t: &Ty) -> R<Atom> {
        match rt {
            RType::Int(_) => self.dec(fx, v),
            RType::Bool => {
                let r = fx.b.var(Ty::Bytes, Atom::bytes(b"false"));
                fx.b.open();
                fx.b.set(r, Atom::bytes(b"true"));
                let th = fx.b.close();
                fx.b.push(Stmt::If(v, th, vec![]));
                Ok(fx.b.let_(Atom::V(r)))
            }
            RType::Text => {
                let d = rt::debug(&mut self.f)?;
                Ok(self.call_rt(fx, d, vec![v]))
            }
            RType::Unit | RType::Never => {
                fx.b.drop(v)?;
                Ok(Atom::bytes(b"()"))
            }
            RType::Tuple(ts) => {
                let cs = fx.b.prim("tup.open", vec![t.clone()], vec![v])?;
                let mut parts = vec![Atom::bytes(b"(")];
                for (k, (c, ct)) in cs.into_iter().zip(ts).enumerate() {
                    if k > 0 {
                        parts.push(Atom::bytes(b", "));
                    }
                    parts.push(self.show(fx, c, ct)?);
                }
                parts.push(Atom::bytes(b")"));
                fx.b.cats(parts)
            }
            RType::List(e) | RType::Set(e) => {
                let et = self.ct(e)?;
                let n = fx.b.p("list.len", vec![et.clone()], vec![v.clone()])?;
                let acc = fx.b.var(Ty::Bytes, Atom::bytes(b"["));
                let i = fx.b.var(Ty::u64(), lit(0));
                fx.b.open();
                let iv = fx.b.let_(Atom::V(i));
                let more = fx.b.int_cmp("int.lt", iv.clone(), n)?;
                fx.b.open();
                let th = fx.b.close();
                fx.b.open();
                fx.b.brk(1);
                let el = fx.b.close();
                fx.b.push(Stmt::If(more, th, el));
                let x = fx.b.list_at(v.clone(), iv.clone())?;
                let s = self.show(fx, x, e)?;
                let a = fx.b.let_(Atom::V(acc));
                let first = fx.b.int_cmp("int.eq", iv.clone(), lit(0))?;
                let sep = fx.b.var(Ty::Bytes, Atom::bytes(b", "));
                fx.b.open();
                fx.b.set(sep, Atom::bytes(b""));
                let th = fx.b.close();
                fx.b.push(Stmt::If(first, th, vec![]));
                let sp = fx.b.let_(Atom::V(sep));
                let a = fx.b.cats(vec![a, sp, s])?;
                fx.b.set(acc, a);
                let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
                fx.b.set(i, i1);
                let body = fx.b.close();
                fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
                fx.b.drop(v)?;
                let a = fx.b.let_(Atom::V(acc));
                fx.b.cat(a, Atom::bytes(b"]"))
            }
            RType::Opt(x) => {
                let tag = fx.b.p("sum.tag", vec![t.clone()], vec![v.clone()])?;
                let some = fx.b.int_cmp("int.eq", tag, lit(1))?;
                let r = fx.b.var(Ty::Bytes, Atom::bytes(b"none"));
                fx.b.open();
                let p = fx.b.p("sum.get", vec![t.clone()], vec![v.clone(), lit(1)])?;
                let s = self.show(fx, p, x)?;
                let s = fx.b.cats(vec![Atom::bytes(b"some("), s, Atom::bytes(b")")])?;
                fx.b.set(r, s);
                let th = fx.b.close();
                fx.b.open();
                fx.b.drop(v)?;
                let el = fx.b.close();
                fx.b.push(Stmt::If(some, th, el));
                Ok(fx.b.let_(Atom::V(r)))
            }
            RType::Union(a, e) => {
                let r = fx.b.var(Ty::Bytes, Atom::bytes(b""));
                let (a, e) = ((**a).clone(), (**e).clone());
                self.switch(fx, &v, t, 2, |lw, fx, k| {
                    let p = fx.b.p("sum.get", vec![t.clone()], vec![v.clone(), lit(k as i128)])?;
                    let s = if k == 0 {
                        lw.show(fx, p, &a)?
                    } else {
                        let s = lw.show(fx, p, &e)?;
                        fx.b.cat(Atom::bytes(b"error "), s)?
                    };
                    fx.b.set(r, s);
                    Ok(())
                })?;
                Ok(fx.b.let_(Atom::V(r)))
            }
            RType::Named(n) => {
                if let Some(vs) = self.t.kinds.get(n).cloned() {
                    let r = fx.b.var(Ty::Bytes, Atom::bytes(b""));
                    let Ty::Sum(cs) = t.clone() else { unreachable!() };
                    self.switch(fx, &v, t, vs.len(), |lw, fx, k| {
                        let p = fx.b.p("sum.get", vec![t.clone()], vec![v.clone(), lit(k as i128)])?;
                        let s = lw.fields_show(fx, &vs[k].0, &vs[k].1, p, &cs[k])?;
                        fx.b.set(r, s);
                        Ok(())
                    })?;
                    Ok(fx.b.let_(Atom::V(r)))
                } else {
                    let fs = self.t.fields(n).unwrap_or_default();
                    self.fields_show(fx, n, &fs, v, t)
                }
            }
            RType::Map(k, x) => {
                let (k, x) = ((**k).clone(), (**x).clone());
                let pt = Ty::Tup(vec![self.ct(&k)?, self.ct(&x)?]);
                let n = fx.b.p("list.len", vec![pt.clone()], vec![v.clone()])?;
                let acc = fx.b.var(Ty::Bytes, Atom::bytes(b"{"));
                let i = fx.b.var(Ty::u64(), lit(0));
                fx.b.open();
                let iv = fx.b.let_(Atom::V(i));
                let more = fx.b.int_cmp("int.lt", iv.clone(), n)?;
                fx.b.open();
                let th = fx.b.close();
                fx.b.open();
                fx.b.brk(1);
                let el = fx.b.close();
                fx.b.push(Stmt::If(more, th, el));
                let e = fx.b.list_at(v.clone(), iv.clone())?;
                let ps = fx.b.prim("tup.open", vec![pt], vec![e])?;
                let ks = self.show(fx, ps[0].clone(), &k)?;
                let xs = self.show(fx, ps[1].clone(), &x)?;
                let a = fx.b.let_(Atom::V(acc));
                let first = fx.b.int_cmp("int.eq", iv.clone(), lit(0))?;
                let sep = fx.b.var(Ty::Bytes, Atom::bytes(b", "));
                fx.b.open();
                fx.b.set(sep, Atom::bytes(b""));
                let th = fx.b.close();
                fx.b.push(Stmt::If(first, th, vec![]));
                let sp = fx.b.let_(Atom::V(sep));
                let a = fx.b.cats(vec![a, sp, ks, Atom::bytes(b": "), xs])?;
                fx.b.set(acc, a);
                let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
                fx.b.set(i, i1);
                let body = fx.b.close();
                fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
                fx.b.drop(v)?;
                let a = fx.b.let_(Atom::V(acc));
                fx.b.cat(a, Atom::bytes(b"}"))
            }
            RType::F64 => {
                let d = rtf::text(&mut self.f)?;
                Ok(self.call_rt(fx, d, vec![v]))
            }
            RType::F32 => {
                let x = fx.b.p("f32.to_f64", vec![], vec![v])?;
                let d = rtf::text32(&mut self.f)?;
                Ok(self.call_rt(fx, d, vec![x]))
            }
            RType::Gen(..) => {
                fx.b.drop(v)?;
                Ok(Atom::bytes(b"<iter>"))
            }
            o => no(format!("the display of {}", o.show())),
        }
    }
}

impl<'m> Lw<'m> {
    /// An arena or device region with a bound (R3, interp `exec`): the
    /// bound, then the body in a scope of its own; after each statement of
    /// the body, the sizes of the canonical JSON of the values bound in
    /// that scope are added up, and a sum above the bound traps with
    /// "size limit".
    fn bounded_region(&mut self, fx: &mut Fx, b: &Expr, body: &[S]) -> R<()> {
        let bound = self.expr(fx, b)?;
        if !matches!(bound.t, RType::Int(_)) {
            return no("an arena bound that is not an integer");
        }
        fx.push_scope();
        let mut locals: Vec<u32> = vec![];
        for s in body {
            self.stmt(fx, s, false)?;
            if !fx.b.falls() {
                break;
            }
            match s {
                S::Let(p, ..) => self.pat_locals(p, &mut locals)?,
                S::Var(..) => locals.push(self.ty.var_binds.get(&addr(s)).copied().ok_or("internal: a var without its local")?),
                _ => {}
            }
            let used = fx.b.var(Ty::Int(0, INT_MAX), lit(0));
            for id in &locals {
                let (a, rt) = match fx.bound.get(id).cloned() {
                    Some(Bound::Var(v, rt)) => (fx.b.peek(v)?, rt),
                    Some(Bound::Val(a, rt)) => (fx.b.copy(a)?, rt),
                    None => return no("a region local outside its frame"),
                };
                let z = self.jsize(fx, a, &rt)?;
                let u = fx.b.take(used);
                let u = fx.b.add_in(u, z, Ty::Int(0, INT_MAX))?;
                fx.b.set(used, u);
            }
            let u = fx.b.take(used);
            let over = fx.b.int_cmp("int.lt", bound.a.clone(), u)?;
            self.guard(fx, over, |_, _| Ok(Atom::bytes(b"size limit")))?;
        }
        fx.pop_scope()
    }

    /// The locals a `let` pattern binds.
    fn pat_locals(&self, p: &Pat, out: &mut Vec<u32>) -> R<()> {
        match p {
            Pat::Wild => Ok(()),
            Pat::Path(_) => {
                out.push(self.ty.pat_binds.get(&addr(p)).copied().ok_or("internal: a pattern without its local")?);
                Ok(())
            }
            Pat::Tuple(ps) => ps.iter().try_for_each(|q| self.pat_locals(q, out)),
            _ => no("a pattern in a region with a bound"),
        }
    }

    fn jsize(&mut self, fx: &mut Fx, v: Atom, rt: &RType) -> R<Atom> {
        let id = self.jsize_fn(rt)?;
        Ok(self.call_rt(fx, id, vec![v]))
    }

    /// The size in bytes of a value's canonical JSON (interp `json_size`):
    /// `jsize:τ(v) -> int`, consuming v. An integer counts its decimal
    /// digits and sign, a float its Rust `{:?}` text, text its escaped
    /// form in quotes, null 4, a sequence 2 plus its items and the commas
    /// between them, a record its quoted name, a colon and its fields as
    /// a sequence of `"name":value`.
    fn jsize_fn(&mut self, rt: &RType) -> R<usize> {
        let t = self.ct(rt)?;
        let rt2 = rt.clone();
        self.generated("jsize", rt, vec![t.clone()], Ty::Int(0, INT_MAX), move |lw, fx| {
            let v = fx.b.arg(0);
            let r = lw.jsize_body(fx, v, &rt2, &t)?;
            fx.b.ret(r);
            Ok(())
        })
    }

    /// `2 + sizes + (n - 1)` for a sequence of n items whose sizes add up
    /// to `sum` (n - 1 commas when n > 0).
    fn seq_size(&mut self, fx: &mut Fx, sum: Atom, n: Atom) -> R<Atom> {
        let w = Ty::Int(0, INT_MAX);
        let a = fx.b.add_in(sum, lit(2), w.clone())?;
        let r = fx.b.var(w.clone(), a.clone());
        let pos = fx.b.int_cmp("int.lt", lit(0), n.clone())?;
        fx.b.when(pos, |b| {
            let n1 = b.sub_in(n, lit(1), Ty::Int(0, INT_MAX))?;
            let a2 = b.add_in(a, n1, w)?;
            b.set(r, a2);
            Ok(())
        })?;
        Ok(fx.b.take(r))
    }

    /// The size of a record `{"name":{"f":v,…}}` given its fields' values.
    fn rec_size(&mut self, fx: &mut Fx, name: &str, fs: &[(String, ast::Ty)], tup: Atom, tt: &Ty) -> R<Atom> {
        let w = Ty::Int(0, INT_MAX);
        let cs = fx.b.prim("tup.open", vec![tt.clone()], vec![tup])?;
        let mut sum = lit(0);
        for ((f, fty), c) in fs.iter().zip(cs) {
            let frt = self.t.of_ty(fty);
            let z = self.jsize(fx, c, &frt)?;
            let z = fx.b.add_in(z, lit(f.len() as i128 + 3), w.clone())?;
            sum = fx.b.add_in(sum, z, w.clone())?;
        }
        let inner = self.seq_size(fx, sum, lit(fs.len() as i128))?;
        fx.b.add_in(inner, lit(name.len() as i128 + 5), w)
    }

    fn jsize_body(&mut self, fx: &mut Fx, v: Atom, rt: &RType, t: &Ty) -> R<Atom> {
        let w = Ty::Int(0, INT_MAX);
        match rt {
            RType::Int(_) => {
                let d = self.dec(fx, v)?;
                fx.b.p("bytes.len", vec![], vec![d])
            }
            RType::F64 | RType::F32 => {
                let x = if *rt == RType::F32 { fx.b.p("f32.to_f64", vec![], vec![v])? } else { v };
                let d = rtf::debug(&mut self.f)?;
                let s = self.call_rt(fx, d, vec![x]);
                fx.b.p("bytes.len", vec![], vec![s])
            }
            RType::Bool => {
                let r = fx.b.var(w, lit(5));
                fx.b.when(v, |b| {
                    b.set(r, lit(4));
                    Ok(())
                })?;
                Ok(fx.b.take(r))
            }
            RType::Text => {
                let g = rtt::json_esc_size(&mut self.f)?;
                Ok(self.call_rt(fx, g, vec![v]))
            }
            RType::Unit | RType::Never => {
                fx.b.drop(v)?;
                Ok(lit(4))
            }
            RType::Tuple(ts) => {
                let cs = fx.b.prim("tup.open", vec![t.clone()], vec![v])?;
                let mut sum = lit(0);
                for (c, ct) in cs.into_iter().zip(ts) {
                    let z = self.jsize(fx, c, ct)?;
                    sum = fx.b.add_in(sum, z, w.clone())?;
                }
                self.seq_size(fx, sum, lit(ts.len() as i128))
            }
            RType::List(_) | RType::Set(_) | RType::Map(..) => {
                let Ty::List(et) = t else { return Err("internal: a list of no list type".into()) };
                let et = (**et).clone();
                let n = fx.b.p("list.len", vec![et.clone()], vec![v.clone()])?;
                let sum = fx.b.var(w.clone(), lit(0));
                let rt2 = rt.clone();
                let (v2, n2) = (v.clone(), n.clone());
                let i = fx.b.var(Ty::u64(), lit(0));
                fx.b.open();
                {
                    let iv = fx.b.let_(Atom::V(i));
                    let more = fx.b.int_cmp("int.lt", iv.clone(), n2)?;
                    fx.b.open();
                    let th = fx.b.close();
                    fx.b.open();
                    fx.b.brk(1);
                    let el = fx.b.close();
                    fx.b.push(Stmt::If(more, th, el));
                    let x = fx.b.list_at(v2, iv.clone())?;
                    let z = match &rt2 {
                        RType::Map(k, x2) => {
                            let ps = fx.b.prim("tup.open", vec![et.clone()], vec![x])?;
                            let a = self.jsize(fx, ps[0].clone(), k)?;
                            let b2 = self.jsize(fx, ps[1].clone(), x2)?;
                            let ab = fx.b.add_in(a, b2, w.clone())?;
                            fx.b.add_in(ab, lit(3), w.clone())?
                        }
                        RType::List(e) | RType::Set(e) => self.jsize(fx, x, e)?,
                        _ => unreachable!(),
                    };
                    let a = fx.b.take(sum);
                    let a = fx.b.add_in(a, z, w.clone())?;
                    fx.b.set(sum, a);
                    let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
                    fx.b.set(i, i1);
                }
                let body = fx.b.close();
                fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
                fx.b.drop(v)?;
                let sv = fx.b.take(sum);
                self.seq_size(fx, sv, n)
            }
            RType::Opt(x) => {
                let r = fx.b.var(w, lit(4));
                let tag = fx.b.p("sum.tag", vec![t.clone()], vec![v.clone()])?;
                let some = fx.b.int_cmp("int.eq", tag, lit(1))?;
                fx.b.open();
                let p = fx.b.p("sum.get", vec![t.clone()], vec![v.clone(), lit(1)])?;
                let z = self.jsize(fx, p, x)?;
                fx.b.set(r, z);
                let th = fx.b.close();
                fx.b.open();
                fx.b.drop(v)?;
                let el = fx.b.close();
                fx.b.push(Stmt::If(some, th, el));
                Ok(fx.b.take(r))
            }
            RType::Union(a, e) => {
                let r = fx.b.var(w, lit(0));
                let (a, e) = ((**a).clone(), (**e).clone());
                self.switch(fx, &v, t, 2, |lw, fx, k| {
                    let p = fx.b.p("sum.get", vec![t.clone()], vec![v.clone(), lit(k as i128)])?;
                    let z = lw.jsize(fx, p, if k == 0 { &a } else { &e })?;
                    fx.b.set(r, z);
                    Ok(())
                })?;
                Ok(fx.b.take(r))
            }
            RType::Named(n) => {
                if let Some(vs) = self.t.kinds.get(n).cloned() {
                    let r = fx.b.var(w, lit(0));
                    let Ty::Sum(cs) = t.clone() else { unreachable!() };
                    self.switch(fx, &v, t, vs.len(), |lw, fx, k| {
                        let p = fx.b.p("sum.get", vec![t.clone()], vec![v.clone(), lit(k as i128)])?;
                        let z = lw.rec_size(fx, &vs[k].0, &vs[k].1, p, &cs[k])?;
                        fx.b.set(r, z);
                        Ok(())
                    })?;
                    Ok(fx.b.take(r))
                } else if let Some(fs) = self.t.fields(n) {
                    self.rec_size(fx, n, &fs, v, t)
                } else {
                    no(format!("the JSON size of {n}"))
                }
            }
            RType::Closure(_) => Ok(lit(2)),
            o => no(format!("the JSON size of {}", o.show())),
        }
    }

    /// Run `case(k)` where the tag of the sum value `v` (read, not consumed)
    /// is k; a tag outside 0..n cannot occur and halts.
    fn switch(&mut self, fx: &mut Fx, v: &Atom, t: &Ty, n: usize, mut case: impl FnMut(&mut Self, &mut Fx, usize) -> R<()>) -> R<()> {
        let tag = fx.b.p("sum.tag", vec![t.clone()], vec![v.clone()])?;
        let mut stack = vec![];
        for k in 0..n {
            let is = fx.b.int_cmp("int.eq", tag.clone(), lit(k as i128))?;
            fx.b.open();
            case(self, fx, k)?;
            let th = fx.b.close();
            fx.b.open();
            stack.push((is, th));
        }
        self.trap(fx, Atom::bytes(b"internal: tag"))?;
        while let Some((is, th)) = stack.pop() {
            let el = fx.b.close();
            fx.b.push(Stmt::If(is, th, el));
        }
        Ok(())
    }

    /// interp `V == V`: `eq:a|b(x, y) -> bool`, consuming both.
    fn eq_fn(&mut self, a: &RType, b: &RType) -> R<usize> {
        let ta = self.ct(a)?;
        let tb = self.ct(b)?;
        let key_t = RType::Tuple(vec![a.clone(), b.clone()]);
        let (a, b) = (a.clone(), b.clone());
        self.generated("eq", &key_t, vec![ta.clone(), tb.clone()], Ty::Bool, move |lw, fx| {
            let (x, y) = (fx.b.arg(0), fx.b.arg(1));
            let r = lw.eq_body(fx, x, &a, &ta, y, &b, &tb)?;
            fx.b.ret(r);
            Ok(())
        })
    }

    fn eq(&mut self, fx: &mut Fx, x: Val, y: Val) -> R<Atom> {
        // scalars inline
        match (&x.t, &y.t) {
            (RType::Int(_), RType::Int(_)) => return fx.b.int_cmp("int.eq", x.a, y.a),
            (RType::Bool, RType::Bool) => return fx.b.p("bool.eq", vec![], vec![x.a, y.a]),
            (RType::Text, RType::Text) => return fx.b.p("bytes.eq", vec![], vec![x.a, y.a]),
            (RType::F64, RType::F64) => return fx.b.p("f64.eq", vec![], vec![x.a, y.a]),
            _ => {}
        }
        // the generated function takes the core types of the RTypes; retype first
        let ta = self.ct(&x.t)?;
        let tb = self.ct(&y.t)?;
        let xa = if fx.b.ty(&x.a) == ta { x.a } else { self.conv(fx, x.a, &x.t, &x.t, false)? };
        let ya = if fx.b.ty(&y.a) == tb { y.a } else { self.conv(fx, y.a, &y.t, &y.t, false)? };
        let id = self.eq_fn(&x.t, &y.t)?;
        Ok(self.call_rt(fx, id, vec![xa, ya]))
    }

    #[allow(clippy::too_many_arguments)]
    fn eq_body(&mut self, fx: &mut Fx, x: Atom, a: &RType, ta: &Ty, y: Atom, b: &RType, tb: &Ty) -> R<Atom> {
        use RType as T;
        let fals = |lw: &mut Self, fx: &mut Fx, x: Atom, y: Atom| -> R<Atom> {
            let _ = lw;
            fx.b.drop(x)?;
            fx.b.drop(y)?;
            Ok(Atom::bool(false))
        };
        match (a, b) {
            (T::Int(_), T::Int(_)) => fx.b.int_cmp("int.eq", x, y),
            (T::Bool, T::Bool) => fx.b.p("bool.eq", vec![], vec![x, y]),
            (T::Text, T::Text) => fx.b.p("bytes.eq", vec![], vec![x, y]),
            (T::F64, T::F64) => fx.b.p("f64.eq", vec![], vec![x, y]),
            (T::F32, T::F32) => {
                let x = fx.b.p("f32.to_f64", vec![], vec![x])?;
                let y = fx.b.p("f32.to_f64", vec![], vec![y])?;
                fx.b.p("f64.eq", vec![], vec![x, y])
            }
            (T::Unit, T::Unit) => {
                fx.b.drop(x)?;
                fx.b.drop(y)?;
                Ok(Atom::bool(true))
            }
            (T::Never, _) | (_, T::Never) => fals(self, fx, x, y),
            (T::Tuple(xs), T::Tuple(ys)) if xs.len() == ys.len() => {
                let cx = fx.b.prim("tup.open", vec![ta.clone()], vec![x])?;
                let cy = fx.b.prim("tup.open", vec![tb.clone()], vec![y])?;
                let mut acc = Atom::bool(true);
                for ((p, q), (s, t)) in cx.into_iter().zip(cy).zip(xs.iter().zip(ys)) {
                    let e = self.eq(fx, Val { a: p, t: s.clone() }, Val { a: q, t: t.clone() })?;
                    acc = fx.b.p("bool.and", vec![], vec![acc, e])?;
                }
                Ok(acc)
            }
            (T::List(e1), T::List(e2)) | (T::Set(e1), T::Set(e2)) => {
                let (Ty::List(c1), Ty::List(c2)) = (ta, tb) else { unreachable!() };
                let n1 = fx.b.p("list.len", vec![(**c1).clone()], vec![x.clone()])?;
                let n2 = fx.b.p("list.len", vec![(**c2).clone()], vec![y.clone()])?;
                let r = fx.b.var(Ty::Bool, Atom::bool(false));
                let same = fx.b.int_cmp("int.eq", n1.clone(), n2)?;
                fx.b.open();
                {
                    fx.b.set(r, Atom::bool(true));
                    let i = fx.b.var(Ty::u64(), lit(0));
                    fx.b.open();
                    let iv = fx.b.let_(Atom::V(i));
                    let more = fx.b.int_cmp("int.lt", iv.clone(), n1)?;
                    fx.b.open();
                    let th = fx.b.close();
                    fx.b.open();
                    fx.b.brk(1);
                    let el = fx.b.close();
                    fx.b.push(Stmt::If(more, th, el));
                    let p = fx.b.list_at(x.clone(), iv.clone())?;
                    let q = fx.b.list_at(y.clone(), iv.clone())?;
                    let e = self.eq(fx, Val { a: p, t: (**e1).clone() }, Val { a: q, t: (**e2).clone() })?;
                    fx.b.open();
                    let th = fx.b.close();
                    fx.b.open();
                    fx.b.set(r, Atom::bool(false));
                    fx.b.brk(1);
                    let el = fx.b.close();
                    fx.b.push(Stmt::If(e, th, el));
                    let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
                    fx.b.set(i, i1);
                    let body = fx.b.close();
                    fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
                }
                let th = fx.b.close();
                fx.b.push(Stmt::If(same, th, vec![]));
                fx.b.drop(x)?;
                fx.b.drop(y)?;
                Ok(fx.b.let_(Atom::V(r)))
            }
            (T::Opt(p), T::Opt(q)) => {
                let (p, q) = ((**p).clone(), (**q).clone());
                self.sum_eq(fx, x, ta, y, tb, &[T::Unit, p], &[T::Unit, q])
            }
            (T::Union(p1, e1), T::Union(p2, e2)) => {
                let (p1, e1, p2, e2) = ((**p1).clone(), (**e1).clone(), (**p2).clone(), (**e2).clone());
                self.sum_eq(fx, x, ta, y, tb, &[p1, e1], &[p2, e2])
            }
            (T::Named(n), T::Named(m)) if n == m => {
                if let Some(vs) = self.t.kinds.get(n).cloned() {
                    let cs: Vec<RType> = vs.iter().map(|(_, fs)| RType::Tuple(fs.iter().map(|(_, t)| self.t.of_ty(t)).collect())).collect();
                    self.sum_eq(fx, x, ta, y, tb, &cs, &cs)
                } else {
                    let fs: Vec<RType> = self.t.fields(n).unwrap_or_default().iter().map(|(_, t)| self.t.of_ty(t)).collect();
                    self.eq_body(fx, x, &T::Tuple(fs.clone()), ta, y, &T::Tuple(fs), tb)
                }
            }
            (T::Map(k1, v1), T::Map(k2, v2)) => {
                // both are kept in key order without duplicates: the same
                // length, and pairwise the same key (6.1 identity) and equal values
                if k1 != k2 {
                    return no("equality of maps with keys of different types");
                }
                let (k1, v1, v2) = ((**k1).clone(), (**v1).clone(), (**v2).clone());
                let (Ty::List(c1), Ty::List(c2)) = (ta, tb) else { unreachable!() };
                let (c1, c2) = ((**c1).clone(), (**c2).clone());
                let cmpf = self.cmp_fn(&k1)?;
                let n1 = fx.b.p("list.len", vec![c1.clone()], vec![x.clone()])?;
                let n2 = fx.b.p("list.len", vec![c2.clone()], vec![y.clone()])?;
                let r = fx.b.var(Ty::Bool, Atom::bool(false));
                let same = fx.b.int_cmp("int.eq", n1.clone(), n2)?;
                fx.b.open();
                {
                    fx.b.set(r, Atom::bool(true));
                    let i = fx.b.var(Ty::u64(), lit(0));
                    fx.b.open();
                    let iv = fx.b.let_(Atom::V(i));
                    let more = fx.b.int_cmp("int.lt", iv.clone(), n1)?;
                    fx.b.open();
                    let th = fx.b.close();
                    fx.b.open();
                    fx.b.brk(1);
                    let el = fx.b.close();
                    fx.b.push(Stmt::If(more, th, el));
                    let p = fx.b.list_at(x.clone(), iv.clone())?;
                    let q = fx.b.list_at(y.clone(), iv.clone())?;
                    let ps = fx.b.prim("tup.open", vec![c1.clone()], vec![p])?;
                    let qs = fx.b.prim("tup.open", vec![c2.clone()], vec![q])?;
                    let c = self.call_rt(fx, cmpf, vec![ps[0].clone(), qs[0].clone()]);
                    let ke = fx.b.int_cmp("int.eq", c, lit(0))?;
                    let ve = self.eq(fx, Val { a: ps[1].clone(), t: v1.clone() }, Val { a: qs[1].clone(), t: v2.clone() })?;
                    let both = fx.b.p("bool.and", vec![], vec![ke, ve])?;
                    fx.b.open();
                    let th = fx.b.close();
                    fx.b.open();
                    fx.b.set(r, Atom::bool(false));
                    fx.b.brk(1);
                    let el = fx.b.close();
                    fx.b.push(Stmt::If(both, th, el));
                    let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
                    fx.b.set(i, i1);
                    let body = fx.b.close();
                    fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
                }
                let th = fx.b.close();
                fx.b.push(Stmt::If(same, th, vec![]));
                fx.b.drop(x)?;
                fx.b.drop(y)?;
                let _ = k1;
                Ok(fx.b.let_(Atom::V(r)))
            }
            (T::Map(..), _) | (_, T::Map(..)) => fals(self, fx, x, y),
            _ => fals(self, fx, x, y),
        }
    }

    /// Equality of two sums: the same tag and equal payloads.
    #[allow(clippy::too_many_arguments)]
    fn sum_eq(&mut self, fx: &mut Fx, x: Atom, ta: &Ty, y: Atom, tb: &Ty, ca: &[RType], cb: &[RType]) -> R<Atom> {
        let r = fx.b.var(Ty::Bool, Atom::bool(false));
        let tx = fx.b.p("sum.tag", vec![ta.clone()], vec![x.clone()])?;
        let ty_ = fx.b.p("sum.tag", vec![tb.clone()], vec![y.clone()])?;
        let same = fx.b.int_cmp("int.eq", tx, ty_.clone())?;
        fx.b.open();
        {
            let n = ca.len();
            let (ta2, tb2) = (ta.clone(), tb.clone());
            self.switch(fx, &x, ta, n, |lw, fx, k| {
                let p = fx.b.p("sum.get", vec![ta2.clone()], vec![x.clone(), lit(k as i128)])?;
                let isk = fx.b.int_cmp("int.eq", ty_.clone(), lit(k as i128))?;
                fx.b.open();
                let q = fx.b.p("sum.get", vec![tb2.clone()], vec![y.clone(), lit(k as i128)])?;
                let e = lw.eq(fx, Val { a: p.clone(), t: ca[k].clone() }, Val { a: q, t: cb[k].clone() })?;
                fx.b.set(r, e);
                let th = fx.b.close();
                fx.b.open();
                lw.trap(fx, Atom::bytes(b"internal: tags"))?;
                let el = fx.b.close();
                fx.b.push(Stmt::If(isk, th, el));
                Ok(())
            })?;
        }
        let th = fx.b.close();
        fx.b.open();
        fx.b.drop(x)?;
        fx.b.drop(y)?;
        let el = fx.b.close();
        fx.b.push(Stmt::If(same, th, el));
        Ok(fx.b.let_(Atom::V(r)))
    }

    /// The positional canonical JSON of a value (6.1, T7; interp
    /// `form_value_pos`), consuming it.
    fn cjp_fn(&mut self, rt: &RType) -> R<usize> {
        let t = self.ct(rt)?;
        let rt2 = rt.clone();
        self.generated("cjp", rt, vec![t.clone()], Ty::Bytes, move |lw, fx| {
            let v = fx.b.arg(0);
            let r = lw.cjp_body(fx, v, &rt2, &t)?;
            fx.b.ret(r);
            Ok(())
        })
    }

    fn cjp(&mut self, fx: &mut Fx, v: Atom, rt: &RType) -> R<Atom> {
        let id = self.cjp_fn(rt)?;
        Ok(self.call_rt(fx, id, vec![v]))
    }

    fn cjp_list(&mut self, fx: &mut Fx, cs: Vec<Atom>, ts: &[RType]) -> R<Atom> {
        let mut parts = vec![Atom::bytes(b"[")];
        for (k, (c, t)) in cs.into_iter().zip(ts).enumerate() {
            if k > 0 {
                parts.push(Atom::bytes(b","));
            }
            parts.push(self.cjp(fx, c, t)?);
        }
        parts.push(Atom::bytes(b"]"));
        fx.b.cats(parts)
    }

    fn cjp_body(&mut self, fx: &mut Fx, v: Atom, rt: &RType, t: &Ty) -> R<Atom> {
        match rt {
            RType::Int(_) => self.dec(fx, v),
            RType::Bool => {
                let r = fx.b.var(Ty::Bytes, Atom::bytes(b"false"));
                fx.b.open();
                fx.b.set(r, Atom::bytes(b"true"));
                let th = fx.b.close();
                fx.b.push(Stmt::If(v, th, vec![]));
                Ok(fx.b.let_(Atom::V(r)))
            }
            RType::Text => {
                let j = rt::jstr(&mut self.f)?;
                Ok(self.call_rt(fx, j, vec![v]))
            }
            RType::Unit | RType::Never => {
                fx.b.drop(v)?;
                Ok(Atom::bytes(b"null"))
            }
            RType::Tuple(ts) => {
                let cs = fx.b.prim("tup.open", vec![t.clone()], vec![v])?;
                self.cjp_list(fx, cs, ts)
            }
            RType::List(e) | RType::Set(e) => {
                let et = self.ct(e)?;
                let n = fx.b.p("list.len", vec![et.clone()], vec![v.clone()])?;
                let acc = fx.b.var(Ty::Bytes, Atom::bytes(b"["));
                let i = fx.b.var(Ty::u64(), lit(0));
                fx.b.open();
                let iv = fx.b.let_(Atom::V(i));
                let more = fx.b.int_cmp("int.lt", iv.clone(), n)?;
                fx.b.open();
                let th = fx.b.close();
                fx.b.open();
                fx.b.brk(1);
                let el = fx.b.close();
                fx.b.push(Stmt::If(more, th, el));
                let x = fx.b.list_at(v.clone(), iv.clone())?;
                let s = self.cjp(fx, x, e)?;
                let first = fx.b.int_cmp("int.eq", iv.clone(), lit(0))?;
                let sep = fx.b.var(Ty::Bytes, Atom::bytes(b","));
                fx.b.open();
                fx.b.set(sep, Atom::bytes(b""));
                let th = fx.b.close();
                fx.b.push(Stmt::If(first, th, vec![]));
                let a = fx.b.let_(Atom::V(acc));
                let sp = fx.b.let_(Atom::V(sep));
                let a = fx.b.cats(vec![a, sp, s])?;
                fx.b.set(acc, a);
                let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
                fx.b.set(i, i1);
                let body = fx.b.close();
                fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
                fx.b.drop(v)?;
                let a = fx.b.let_(Atom::V(acc));
                fx.b.cat(a, Atom::bytes(b"]"))
            }
            RType::Opt(x) => {
                let r = fx.b.var(Ty::Bytes, Atom::bytes(b"null"));
                let x = (**x).clone();
                self.switch(fx, &v, t, 2, |lw, fx, k| {
                    let p = fx.b.p("sum.get", vec![t.clone()], vec![v.clone(), lit(k as i128)])?;
                    if k == 0 {
                        fx.b.drop(p)?;
                    } else {
                        let s = lw.cjp(fx, p, &x)?;
                        fx.b.set(r, s);
                    }
                    Ok(())
                })?;
                Ok(fx.b.let_(Atom::V(r)))
            }
            RType::Union(a, e) => {
                let r = fx.b.var(Ty::Bytes, Atom::bytes(b""));
                let (a, e) = ((**a).clone(), (**e).clone());
                self.switch(fx, &v, t, 2, |lw, fx, k| {
                    let p = fx.b.p("sum.get", vec![t.clone()], vec![v.clone(), lit(k as i128)])?;
                    let s = if k == 0 {
                        lw.cjp(fx, p, &a)?
                    } else {
                        let s = lw.cjp(fx, p, &e)?;
                        fx.b.cats(vec![Atom::bytes(b"{\"$err\":"), s, Atom::bytes(b"}")])?
                    };
                    fx.b.set(r, s);
                    Ok(())
                })?;
                Ok(fx.b.let_(Atom::V(r)))
            }
            RType::Named(n) => {
                if let Some(vs) = self.t.kinds.get(n).cloned() {
                    let r = fx.b.var(Ty::Bytes, Atom::bytes(b""));
                    let Ty::Sum(cs) = t.clone() else { unreachable!() };
                    self.switch(fx, &v, t, vs.len(), |lw, fx, k| {
                        let p = fx.b.p("sum.get", vec![t.clone()], vec![v.clone(), lit(k as i128)])?;
                        let fts: Vec<RType> = vs[k].1.iter().map(|(_, ft)| lw.t.of_ty(ft)).collect();
                        let fsv = fx.b.prim("tup.open", vec![cs[k].clone()], vec![p])?;
                        let inner = lw.cjp_list(fx, fsv, &fts)?;
                        let s = fx.b.cats(vec![Atom::bytes(format!("[{k},").as_bytes()), inner, Atom::bytes(b"]")])?;
                        fx.b.set(r, s);
                        Ok(())
                    })?;
                    Ok(fx.b.let_(Atom::V(r)))
                } else if self.t.fields(n).is_none() {
                    // a record the module does not declare (Missing) keeps its named form
                    fx.b.drop(v)?;
                    Ok(Atom::bytes(b"{}"))
                } else {
                    let fts: Vec<RType> = self.t.fields(n).unwrap_or_default().iter().map(|(_, ft)| self.t.of_ty(ft)).collect();
                    let cs = fx.b.prim("tup.open", vec![t.clone()], vec![v])?;
                    self.cjp_list(fx, cs, &fts)
                }
            }
            RType::Map(k, x) => {
                // interp `form_value_pos`: [[k, v], …] in the canonical order of
                // the keys' Form values, which for the keys a map holds here is
                // the order it is kept in
                self.cmp_fn(k)?;
                let pair = RType::Tuple(vec![(**k).clone(), (**x).clone()]);
                self.cjp_body(fx, v, &RType::List(Box::new(pair)), t)
            }
            RType::F64 => {
                let j = rtf::json(&mut self.f)?;
                Ok(self.call_rt(fx, j, vec![v]))
            }
            RType::F32 => {
                let x = fx.b.p("f32.to_f64", vec![], vec![v])?;
                let j = rtf::json(&mut self.f)?;
                Ok(self.call_rt(fx, j, vec![x]))
            }
            o => no(format!("{} in the log", o.show())),
        }
    }
}

impl<'m> Lw<'m> {
    // ---- the canonical order of 6.1, and maps kept in it

    /// Whether values of this type have a canonical order the lowering
    /// writes: scalars, tuples, lists, options of values that are not
    /// null, structs and kinds of those.
    fn cmp_supported(&self, rt: &RType, d: usize) -> R<()> {
        if d > 16 {
            return no("a recursive type in the canonical order");
        }
        match rt {
            RType::Int(_) | RType::F64 | RType::F32 | RType::Bool | RType::Text | RType::Unit | RType::Never => Ok(()),
            RType::Tuple(ts) => ts.iter().try_for_each(|t| self.cmp_supported(t, d + 1)),
            RType::List(e) | RType::Set(e) => self.cmp_supported(e, d + 1),
            RType::Opt(x) => match &**x {
                RType::Unit | RType::Never | RType::Opt(_) => no("an option of a null value in the canonical order"),
                x => self.cmp_supported(x, d + 1),
            },
            RType::Named(n) => {
                if let Some(vs) = self.t.kinds.get(n) {
                    for (_, fs) in vs {
                        for (_, ft) in fs {
                            self.cmp_supported(&self.t.of_ty(ft), d + 1)?;
                        }
                    }
                    Ok(())
                } else if let Some(fs) = self.t.fields(n) {
                    fs.iter().try_for_each(|(_, ft)| self.cmp_supported(&self.t.of_ty(ft), d + 1))
                } else {
                    no(format!("the canonical order of {n}"))
                }
            }
            o => no(format!("the canonical order of {}", o.show())),
        }
    }

    /// The canonical order of 6.1 (interp `key_form` and `cmp_canon`):
    /// `cmp:τ(a, b) -> int[-1, 1]`, consuming both. Floats are ordered by
    /// IEEE 754 totalOrder, so -0.0 comes before 0.0; a record compares its
    /// type's name first, then its fields in the order of their names.
    fn cmp_fn(&mut self, rt: &RType) -> R<usize> {
        self.cmp_supported(rt, 0)?;
        let t = self.ct(rt)?;
        let rt2 = rt.clone();
        self.generated("cmp", rt, vec![t.clone(), t.clone()], Ty::Int(-1, 1), move |lw, fx| {
            let (a, b) = (fx.b.arg(0), fx.b.arg(1));
            let r = lw.cmp_body(fx, a, b, &rt2, &t)?;
            fx.b.ret(r);
            Ok(())
        })
    }

    /// The order of `<` and of `sort` (interp `cmp`): integers, floats (where
    /// the two zeros are equal), text, bools, and tuples and lists of those
    /// compared item by item: `ord:τ(a, b) -> int[-1, 1]`, consuming both.
    fn ord_fn(&mut self, rt: &RType) -> R<usize> {
        fn ok(rt: &RType) -> bool {
            match rt {
                RType::Int(_) | RType::F64 | RType::F32 | RType::Text | RType::Bool | RType::Never => true,
                RType::Tuple(ts) => ts.iter().all(ok),
                RType::List(e) => ok(e),
                _ => false,
            }
        }
        if !ok(rt) {
            return no(format!("the order of {}", rt.show()));
        }
        let t = self.ct(rt)?;
        let rt2 = rt.clone();
        self.generated("ord", rt, vec![t.clone(), t.clone()], Ty::Int(-1, 1), move |lw, fx| {
            let (a, b) = (fx.b.arg(0), fx.b.arg(1));
            let r = match &rt2 {
                RType::F64 | RType::F32 => {
                    let (a, b) = if rt2 == RType::F32 {
                        (fx.b.p("f32.to_f64", vec![], vec![a])?, fx.b.p("f32.to_f64", vec![], vec![b])?)
                    } else {
                        (a, b)
                    };
                    let lt = fx.b.p("f64.lt", vec![], vec![a.clone(), b.clone()])?;
                    let gt = fx.b.p("f64.lt", vec![], vec![b, a])?;
                    lw.order(fx, lt, gt)?
                }
                RType::Tuple(ts) => {
                    let xs = fx.b.prim("tup.open", vec![t.clone()], vec![a])?;
                    let ys = fx.b.prim("tup.open", vec![t.clone()], vec![b])?;
                    let r = fx.b.var(Ty::Int(-1, 1), lit(0));
                    for ((x, y), ct) in xs.into_iter().zip(ys).zip(ts.clone()) {
                        let f = lw.ord_fn(&ct)?;
                        let c = lw.call_rt(fx, f, vec![x, y]);
                        let rv = fx.b.let_(Atom::V(r));
                        let z = fx.b.int_cmp("int.eq", rv, lit(0))?;
                        fx.b.open();
                        fx.b.set(r, c);
                        let th = fx.b.close();
                        fx.b.push(Stmt::If(z, th, vec![]));
                    }
                    fx.b.let_(Atom::V(r))
                }
                RType::List(_) => {
                    // item by item, then the shorter first: as the canonical
                    // order of a list, with this order on the items
                    let RType::List(e) = &rt2 else { unreachable!() };
                    let e = (**e).clone();
                    let Ty::List(ec) = &t else { unreachable!() };
                    let ec = (**ec).clone();
                    let n1 = fx.b.p("list.len", vec![ec.clone()], vec![a.clone()])?;
                    let n2 = fx.b.p("list.len", vec![ec.clone()], vec![b.clone()])?;
                    let n = fx.b.p("int.min", vec![Ty::Int(0, INT_MAX)], vec![n1.clone(), n2.clone()])?;
                    let r = fx.b.var(Ty::Int(-1, 1), lit(0));
                    let i = fx.b.var(Ty::u64(), lit(0));
                    let f = lw.ord_fn(&e)?;
                    fx.b.open();
                    let iv = fx.b.let_(Atom::V(i));
                    let more = fx.b.int_cmp("int.lt", iv.clone(), n)?;
                    fx.b.open();
                    let th = fx.b.close();
                    fx.b.open();
                    fx.b.brk(1);
                    let el = fx.b.close();
                    fx.b.push(Stmt::If(more, th, el));
                    let x = fx.b.list_at(a.clone(), iv.clone())?;
                    let y = fx.b.list_at(b.clone(), iv.clone())?;
                    let c = lw.call_rt(fx, f, vec![x, y]);
                    let z = fx.b.int_cmp("int.eq", c.clone(), lit(0))?;
                    fx.b.open();
                    let th = fx.b.close();
                    fx.b.open();
                    fx.b.set(r, c);
                    fx.b.brk(1);
                    let el = fx.b.close();
                    fx.b.push(Stmt::If(z, th, el));
                    let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
                    fx.b.set(i, i1);
                    let body = fx.b.close();
                    fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
                    fx.b.drop(a)?;
                    fx.b.drop(b)?;
                    let rv = fx.b.let_(Atom::V(r));
                    let z = fx.b.int_cmp("int.eq", rv, lit(0))?;
                    fx.b.open();
                    let lt = fx.b.int_cmp("int.lt", n1.clone(), n2.clone())?;
                    let gt = fx.b.int_cmp("int.lt", n2, n1)?;
                    let o = lw.order(fx, lt, gt)?;
                    fx.b.set(r, o);
                    let th = fx.b.close();
                    fx.b.push(Stmt::If(z, th, vec![]));
                    fx.b.let_(Atom::V(r))
                }
                _ => lw.cmp_body(fx, a, b, &rt2, &t)?,
            };
            fx.b.ret(r);
            Ok(())
        })
    }

    /// `sort:List[τ](xs)` (interp `sort`, Rust's stable sort): an insertion
    /// sort by the order of `ord`, which moves an item only past greater
    /// ones, so equal items keep their order.
    fn sort_fn(&mut self, et: &RType) -> R<usize> {
        let f = self.ord_fn(et)?;
        self.sort_by_fn("sort", et, f)
    }

    /// `set_of:τ(xs)` (interp `set_of`): the items sorted in canonical
    /// order (6.1), then each item kept unless it equals the one before.
    /// Equal items in that order display alike, so which of them is kept
    /// is not observed. Only scalars, text, and tuples and lists of those,
    /// where the order of `conc::canon_cmp` and of `cmp` agree.
    fn set_of_fn(&mut self, et: &RType) -> R<usize> {
        fn ok(rt: &RType) -> bool {
            match rt {
                RType::Int(_) | RType::F64 | RType::F32 | RType::Text | RType::Bool | RType::Unit | RType::Never => true,
                RType::Tuple(ts) => ts.iter().all(ok),
                RType::List(e) => ok(e),
                _ => false,
            }
        }
        if !ok(et) {
            return no(format!("set_of of {}", et.show()));
        }
        let c = self.cmp_fn(et)?;
        let s = self.sort_by_fn("canon_sort", et, c)?;
        let e = self.ct(et)?;
        let lt = Ty::list(e.clone());
        self.generated("set_of", et, vec![lt.clone()], lt.clone(), move |lw, fx| {
            let a0 = fx.b.arg(0);
            let xs = lw.call_rt(fx, s, vec![a0]);
            let xs = fx.b.let_(xs);
            let l0 = fx.b.p("list.new", vec![e.clone()], vec![])?;
            let out = fx.b.var(lt.clone(), l0);
            let n = fx.b.p("list.len", vec![e.clone()], vec![xs.clone()])?;
            fx.b.for_n(n, |fx2, i| {
                let x = fx2.list_at(xs.clone(), i)?;
                let k = fx2.p("list.len", vec![e.clone()], vec![Atom::V(out)])?;
                let z = fx2.int_cmp("int.eq", k.clone(), lit(0))?;
                let keep = fx2.var(Ty::Bool, Atom::bool(true));
                fx2.open();
                let th = fx2.close();
                fx2.open();
                {
                    let k1 = fx2.sub_in(k, lit(1), Ty::u64())?;
                    let y = fx2.list_at(Atom::V(out), k1)?;
                    let xc = fx2.copy(x.clone())?;
                    let c2 = fx2.call(sym(c), Ty::Int(-1, 1), vec![y, xc]);
                    let d = fx2.int_cmp("int.eq", c2, lit(0))?;
                    let nd = fx2.not(d)?;
                    fx2.set(keep, nd);
                }
                let el = fx2.close();
                fx2.push(Stmt::If(z, th, el));
                let kv = fx2.let_(Atom::V(keep));
                fx2.open();
                {
                    let o = fx2.let_(Atom::V(out));
                    let xc = fx2.copy(x.clone())?;
                    let o = fx2.p("list.push", vec![e.clone()], vec![o, xc])?;
                    fx2.set(out, o);
                }
                let th = fx2.close();
                fx2.push(Stmt::If(kv, th, vec![]));
                fx2.drop(x)?;
                Ok(())
            })?;
            fx.b.drop(xs)?;
            let r = fx.b.let_(Atom::V(out));
            fx.b.ret(r);
            Ok(())
        })
    }

    /// A stable insertion sort of a list by the order function f
    /// (`f(a, b) -> int[-1, 1]`, consuming both), which moves an item only
    /// past greater ones, so equal items keep their order.
    fn sort_by_fn(&mut self, kind: &str, et: &RType, f: usize) -> R<usize> {
        let e = self.ct(et)?;
        let lt = Ty::list(e.clone());
        self.generated(kind, et, vec![lt.clone()], lt.clone(), move |lw, fx| {
            let a = fx.b.var(lt.clone(), fx.b.arg(0));
            let n = fx.b.p("list.len", vec![e.clone()], vec![Atom::V(a)])?;
            let i = fx.b.var(Ty::u64(), lit(1));
            fx.b.open();
            {
                let iv = fx.b.let_(Atom::V(i));
                let more = fx.b.int_cmp("int.lt", iv.clone(), n.clone())?;
                fx.b.open();
                let th = fx.b.close();
                fx.b.open();
                fx.b.brk(1);
                let el = fx.b.close();
                fx.b.push(Stmt::If(more, th, el));
                let x = fx.b.list_at(Atom::V(a), iv.clone())?;
                let j = fx.b.var(Ty::u64(), iv.clone());
                fx.b.open();
                {
                    let jv = fx.b.let_(Atom::V(j));
                    let z = fx.b.int_cmp("int.eq", jv.clone(), lit(0))?;
                    fx.b.open();
                    fx.b.brk(1);
                    let th = fx.b.close();
                    fx.b.push(Stmt::If(z, th, vec![]));
                    let j1 = fx.b.sub_in(jv.clone(), lit(1), Ty::u64())?;
                    let y = fx.b.list_at(Atom::V(a), j1.clone())?;
                    let xc = fx.b.copy(x.clone())?;
                    let c = lw.call_rt(fx, f, vec![y, xc]);
                    let gt = fx.b.int_cmp("int.lt", lit(0), c)?;
                    fx.b.open();
                    let th = fx.b.close();
                    fx.b.open();
                    fx.b.brk(1);
                    let el = fx.b.close();
                    fx.b.push(Stmt::If(gt, th, el));
                    let y2 = fx.b.list_at(Atom::V(a), j1.clone())?;
                    let n2 = fx.b.p("list.len", vec![e.clone()], vec![Atom::V(a)])?;
                    let ok = fx.b.int_cmp("int.lt", jv.clone(), n2)?;
                    fx.b.guard_halt_unless(ok, "internal: sort index")?;
                    let av = fx.b.let_(Atom::V(a));
                    let av = fx.b.p("list.set", vec![e.clone()], vec![av, jv, y2])?;
                    fx.b.set(a, av);
                    fx.b.set(j, j1);
                }
                let body = fx.b.close();
                fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
                let jv = fx.b.let_(Atom::V(j));
                let n2 = fx.b.p("list.len", vec![e.clone()], vec![Atom::V(a)])?;
                let ok = fx.b.int_cmp("int.lt", jv.clone(), n2)?;
                fx.b.guard_halt_unless(ok, "internal: sort index")?;
                let av = fx.b.let_(Atom::V(a));
                let av = fx.b.p("list.set", vec![e.clone()], vec![av, jv, x])?;
                fx.b.set(a, av);
                let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
                fx.b.set(i, i1);
            }
            let body = fx.b.close();
            fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
            let av = fx.b.let_(Atom::V(a));
            fx.b.ret(av);
            Ok(())
        })
    }

    fn cmp(&mut self, fx: &mut Fx, a: Atom, b: Atom, rt: &RType) -> R<Atom> {
        let f = self.cmp_fn(rt)?;
        Ok(self.call_rt(fx, f, vec![a, b]))
    }

    /// -1 when `lt`, 1 when `gt`, else 0.
    fn order(&mut self, fx: &mut Fx, lt: Atom, gt: Atom) -> R<Atom> {
        let r = fx.b.var(Ty::Int(-1, 1), lit(0));
        fx.b.open();
        fx.b.set(r, lit(-1));
        let th = fx.b.close();
        fx.b.open();
        fx.b.open();
        fx.b.set(r, lit(1));
        let th2 = fx.b.close();
        fx.b.push(Stmt::If(gt, th2, vec![]));
        let el = fx.b.close();
        fx.b.push(Stmt::If(lt, th, el));
        Ok(fx.b.let_(Atom::V(r)))
    }

    /// The first of the comparisons that is not 0 (each consumes its pair).
    fn first_nonzero(&mut self, fx: &mut Fx, pairs: Vec<(Atom, Atom, RType)>) -> R<Atom> {
        let r = fx.b.var(Ty::Int(-1, 1), lit(0));
        for (x, y, t) in pairs {
            let c = self.cmp(fx, x, y, &t)?;
            let rv = fx.b.let_(Atom::V(r));
            let z = fx.b.int_cmp("int.eq", rv, lit(0))?;
            fx.b.open();
            fx.b.set(r, c);
            let th = fx.b.close();
            fx.b.push(Stmt::If(z, th, vec![]));
        }
        Ok(fx.b.let_(Atom::V(r)))
    }

    /// Fields compared in the order of their names.
    fn fields_by_name(&mut self, fx: &mut Fx, a: Atom, b: Atom, tt: &Ty, fs: &[(String, ast::Ty)]) -> R<Atom> {
        let xs = fx.b.prim("tup.open", vec![tt.clone()], vec![a])?;
        let ys = fx.b.prim("tup.open", vec![tt.clone()], vec![b])?;
        let mut order: Vec<usize> = (0..fs.len()).collect();
        order.sort_by(|p, q| fs[*p].0.cmp(&fs[*q].0));
        let pairs = order.into_iter().map(|k| (xs[k].clone(), ys[k].clone(), self.t.of_ty(&fs[k].1))).collect();
        self.first_nonzero(fx, pairs)
    }

    fn cmp_body(&mut self, fx: &mut Fx, a: Atom, b: Atom, rt: &RType, t: &Ty) -> R<Atom> {
        match rt {
            RType::Int(_) => {
                let lt = fx.b.int_cmp("int.lt", a.clone(), b.clone())?;
                let gt = fx.b.int_cmp("int.lt", b, a)?;
                self.order(fx, lt, gt)
            }
            RType::F64 | RType::F32 => {
                let (a, b) = if *rt == RType::F32 {
                    (fx.b.p("f32.to_f64", vec![], vec![a])?, fx.b.p("f32.to_f64", vec![], vec![b])?)
                } else {
                    (a, b)
                };
                let lt = fx.b.p("f64.lt", vec![], vec![a.clone(), b.clone()])?;
                let gt = fx.b.p("f64.lt", vec![], vec![b.clone(), a.clone()])?;
                // equal values differ only in the sign of a zero
                let sa = fx.b.p("f64.sign", vec![], vec![a])?;
                let sb = fx.b.p("f64.sign", vec![], vec![b])?;
                let nsb = fx.b.not(sb.clone())?;
                let nsa = fx.b.not(sa.clone())?;
                let zlt = fx.b.p("bool.and", vec![], vec![sa, nsb])?;
                let zgt = fx.b.p("bool.and", vec![], vec![nsa, sb])?;
                let ngt = fx.b.not(gt.clone())?;
                let lt2 = fx.b.p("bool.and", vec![], vec![ngt, zlt])?;
                let lt = fx.b.p("bool.or", vec![], vec![lt, lt2])?;
                let nlt = fx.b.not(lt.clone())?;
                let gt2 = fx.b.p("bool.and", vec![], vec![nlt, zgt])?;
                let gt = fx.b.p("bool.or", vec![], vec![gt, gt2])?;
                self.order(fx, lt, gt)
            }
            RType::Bool => {
                let na = fx.b.not(a.clone())?;
                let nb = fx.b.not(b.clone())?;
                let lt = fx.b.p("bool.and", vec![], vec![na, b])?;
                let gt = fx.b.p("bool.and", vec![], vec![a, nb])?;
                self.order(fx, lt, gt)
            }
            RType::Text => {
                let lt = fx.b.p("bytes.lt", vec![], vec![a.clone(), b.clone()])?;
                let gt = fx.b.p("bytes.lt", vec![], vec![b, a])?;
                self.order(fx, lt, gt)
            }
            RType::Unit | RType::Never => {
                fx.b.drop(a)?;
                fx.b.drop(b)?;
                Ok(lit(0))
            }
            RType::Tuple(ts) => {
                let xs = fx.b.prim("tup.open", vec![t.clone()], vec![a])?;
                let ys = fx.b.prim("tup.open", vec![t.clone()], vec![b])?;
                let pairs = xs.into_iter().zip(ys).zip(ts.clone()).map(|((x, y), t)| (x, y, t)).collect();
                self.first_nonzero(fx, pairs)
            }
            RType::List(e) | RType::Set(e) => {
                let e = (**e).clone();
                let Ty::List(ec) = t else { unreachable!() };
                let ec = (**ec).clone();
                let n1 = fx.b.p("list.len", vec![ec.clone()], vec![a.clone()])?;
                let n2 = fx.b.p("list.len", vec![ec.clone()], vec![b.clone()])?;
                let n = fx.b.p("int.min", vec![Ty::Int(0, INT_MAX)], vec![n1.clone(), n2.clone()])?;
                let r = fx.b.var(Ty::Int(-1, 1), lit(0));
                let i = fx.b.var(Ty::u64(), lit(0));
                fx.b.open();
                {
                    let iv = fx.b.let_(Atom::V(i));
                    let more = fx.b.int_cmp("int.lt", iv.clone(), n)?;
                    fx.b.open();
                    let th = fx.b.close();
                    fx.b.open();
                    fx.b.brk(1);
                    let el = fx.b.close();
                    fx.b.push(Stmt::If(more, th, el));
                    let x = fx.b.list_at(a.clone(), iv.clone())?;
                    let y = fx.b.list_at(b.clone(), iv.clone())?;
                    let c = self.cmp(fx, x, y, &e)?;
                    let z = fx.b.int_cmp("int.eq", c.clone(), lit(0))?;
                    fx.b.open();
                    let th = fx.b.close();
                    fx.b.open();
                    fx.b.set(r, c);
                    fx.b.brk(1);
                    let el = fx.b.close();
                    fx.b.push(Stmt::If(z, th, el));
                    let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
                    fx.b.set(i, i1);
                }
                let body = fx.b.close();
                fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
                fx.b.drop(a)?;
                fx.b.drop(b)?;
                let rv = fx.b.let_(Atom::V(r));
                let z = fx.b.int_cmp("int.eq", rv, lit(0))?;
                fx.b.open();
                let lt = fx.b.int_cmp("int.lt", n1.clone(), n2.clone())?;
                let gt = fx.b.int_cmp("int.lt", n2, n1)?;
                let o = self.order(fx, lt, gt)?;
                fx.b.set(r, o);
                let th = fx.b.close();
                fx.b.push(Stmt::If(z, th, vec![]));
                Ok(fx.b.let_(Atom::V(r)))
            }
            RType::Opt(x) => {
                // none is null, which comes before every value
                let x = (**x).clone();
                let ta = fx.b.p("sum.tag", vec![t.clone()], vec![a.clone()])?;
                let tb = fx.b.p("sum.tag", vec![t.clone()], vec![b.clone()])?;
                let sa = fx.b.int_cmp("int.eq", ta.clone(), lit(1))?;
                let sb = fx.b.int_cmp("int.eq", tb.clone(), lit(1))?;
                let r = fx.b.var(Ty::Int(-1, 1), lit(0));
                let both = fx.b.p("bool.and", vec![], vec![sa.clone(), sb.clone()])?;
                fx.b.open();
                {
                    let one_a = fx.b.int_cmp("int.eq", ta, lit(1))?;
                    fx.b.guard_halt_unless(one_a, "internal: tag")?;
                    let p = fx.b.p("sum.get", vec![t.clone()], vec![a.clone(), lit(1)])?;
                    let one_b = fx.b.int_cmp("int.eq", tb, lit(1))?;
                    fx.b.guard_halt_unless(one_b, "internal: tag")?;
                    let q = fx.b.p("sum.get", vec![t.clone()], vec![b.clone(), lit(1)])?;
                    let c = self.cmp(fx, p, q, &x)?;
                    fx.b.set(r, c);
                }
                let th = fx.b.close();
                fx.b.open();
                {
                    let na = fx.b.not(sa.clone())?;
                    let nb = fx.b.not(sb.clone())?;
                    let lt = fx.b.p("bool.and", vec![], vec![na, sb])?;
                    let gt = fx.b.p("bool.and", vec![], vec![sa, nb])?;
                    let o = self.order(fx, lt, gt)?;
                    fx.b.set(r, o);
                    fx.b.drop(a)?;
                    fx.b.drop(b)?;
                }
                let el = fx.b.close();
                fx.b.push(Stmt::If(both, th, el));
                Ok(fx.b.let_(Atom::V(r)))
            }
            RType::Named(n) => {
                if let Some(vs) = self.t.kinds.get(n).cloned() {
                    // the variant's name first, then its fields by name
                    let mut names: Vec<(String, usize)> = vs.iter().enumerate().map(|(k, (v, _))| (v.clone(), k)).collect();
                    names.sort();
                    let rank: Vec<usize> = (0..vs.len()).map(|k| names.iter().position(|(_, j)| *j == k).unwrap()).collect();
                    let Ty::Sum(cs) = t.clone() else { unreachable!() };
                    let ta = fx.b.p("sum.tag", vec![t.clone()], vec![a.clone()])?;
                    let tb = fx.b.p("sum.tag", vec![t.clone()], vec![b.clone()])?;
                    let ra = fx.b.var(Ty::Int(0, vs.len() as i128), lit(0));
                    let rb = fx.b.var(Ty::Int(0, vs.len() as i128), lit(0));
                    for (k, rk) in rank.iter().enumerate() {
                        for (tg, v) in [(&ta, ra), (&tb, rb)] {
                            let is = fx.b.int_cmp("int.eq", tg.clone(), lit(k as i128))?;
                            fx.b.open();
                            fx.b.set(v, lit(*rk as i128));
                            let th = fx.b.close();
                            fx.b.push(Stmt::If(is, th, vec![]));
                        }
                    }
                    let r = fx.b.var(Ty::Int(-1, 1), lit(0));
                    let same = fx.b.int_cmp("int.eq", ta.clone(), tb.clone())?;
                    fx.b.open();
                    {
                        let tb2 = tb.clone();
                        self.switch(fx, &a, t, vs.len(), |lw, fx, k| {
                            let p = fx.b.p("sum.get", vec![t.clone()], vec![a.clone(), lit(k as i128)])?;
                            let isk = fx.b.int_cmp("int.eq", tb2.clone(), lit(k as i128))?;
                            fx.b.guard_halt_unless(isk, "internal: tags")?;
                            let q = fx.b.p("sum.get", vec![t.clone()], vec![b.clone(), lit(k as i128)])?;
                            let c = lw.fields_by_name(fx, p, q, &cs[k], &vs[k].1)?;
                            fx.b.set(r, c);
                            Ok(())
                        })?;
                    }
                    let th = fx.b.close();
                    fx.b.open();
                    {
                        let av = fx.b.let_(Atom::V(ra));
                        let bv = fx.b.let_(Atom::V(rb));
                        let lt = fx.b.int_cmp("int.lt", av.clone(), bv.clone())?;
                        let gt = fx.b.int_cmp("int.lt", bv, av)?;
                        let o = self.order(fx, lt, gt)?;
                        fx.b.set(r, o);
                        fx.b.drop(a)?;
                        fx.b.drop(b)?;
                    }
                    let el = fx.b.close();
                    fx.b.push(Stmt::If(same, th, el));
                    Ok(fx.b.let_(Atom::V(r)))
                } else {
                    let fs = self.t.fields(n).unwrap_or_default();
                    self.fields_by_name(fx, a, b, t, &fs)
                }
            }
            o => no(format!("the canonical order of {}", o.show())),
        }
    }

    /// `mfind:Map(m, k)`: the index of key k in the map m, or -1; consumes both.
    fn map_find_fn(&mut self, mt: &RType) -> R<usize> {
        let RType::Map(k, _) = mt else { return Err("internal: map_find of a non-map".into()) };
        let k = (**k).clone();
        let cmpf = self.cmp_fn(&k)?;
        let m_t = self.ct(mt)?;
        let kt = self.ct(&k)?;
        let Ty::List(pt) = m_t.clone() else { unreachable!() };
        let pt = (*pt).clone();
        self.generated("mfind", mt, vec![m_t, kt], Ty::Int(-1, INT_MAX), move |lw, fx| {
            let (m, key) = (fx.b.arg(0), fx.b.arg(1));
            let n = fx.b.p("list.len", vec![pt.clone()], vec![m.clone()])?;
            let r = fx.b.var(Ty::Int(-1, INT_MAX), lit(-1));
            let i = fx.b.var(Ty::u64(), lit(0));
            fx.b.open();
            {
                let iv = fx.b.let_(Atom::V(i));
                let more = fx.b.int_cmp("int.lt", iv.clone(), n)?;
                fx.b.open();
                let th = fx.b.close();
                fx.b.open();
                fx.b.brk(1);
                let el = fx.b.close();
                fx.b.push(Stmt::If(more, th, el));
                let e = fx.b.list_at(m.clone(), iv.clone())?;
                let ps = fx.b.prim("tup.open", vec![pt.clone()], vec![e])?;
                fx.b.drop(ps[1].clone())?;
                let kc = fx.b.copy(key.clone())?;
                let c = lw.call_rt(fx, cmpf, vec![ps[0].clone(), kc]);
                let z = fx.b.int_cmp("int.eq", c.clone(), lit(0))?;
                fx.b.open();
                let iv2 = fx.b.cast_in(iv.clone(), Ty::Int(-1, INT_MAX))?;
                fx.b.set(r, iv2);
                fx.b.brk(1);
                let th = fx.b.close();
                fx.b.push(Stmt::If(z, th, vec![]));
                // the keys are in order: past k, k is not there
                let past = fx.b.int_cmp("int.lt", lit(0), c)?;
                fx.b.open();
                fx.b.brk(1);
                let th = fx.b.close();
                fx.b.push(Stmt::If(past, th, vec![]));
                let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
                fx.b.set(i, i1);
            }
            let body = fx.b.close();
            fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
            fx.b.drop(m)?;
            fx.b.drop(key)?;
            let rv = fx.b.let_(Atom::V(r));
            fx.b.ret(rv);
            Ok(())
        })
    }

    /// `mset:Map(m, k, v)` (interp `map_set`): a key already there keeps its
    /// place and takes the value; a new key goes where the order puts it.
    fn map_set_fn(&mut self, mt: &RType) -> R<usize> {
        let RType::Map(k, x) = mt else { return Err("internal: map_set of a non-map".into()) };
        let (k, x) = ((**k).clone(), (**x).clone());
        let cmpf = self.cmp_fn(&k)?;
        let m_t = self.ct(mt)?;
        let kt = self.ct(&k)?;
        let xt = self.ct(&x)?;
        let Ty::List(pt) = m_t.clone() else { unreachable!() };
        let pt = (*pt).clone();
        self.generated("mset", mt, vec![m_t.clone(), kt, xt], m_t.clone(), move |lw, fx| {
            let (m, key, val) = (fx.b.arg(0), fx.b.arg(1), fx.b.arg(2));
            let n = fx.b.p("list.len", vec![pt.clone()], vec![m.clone()])?;
            // the first index whose key is not below k, and whether it is k
            let at = fx.b.var(Ty::Int(0, INT_MAX), n.clone());
            let found = fx.b.var(Ty::Bool, Atom::bool(false));
            let i = fx.b.var(Ty::u64(), lit(0));
            fx.b.open();
            {
                let iv = fx.b.let_(Atom::V(i));
                let more = fx.b.int_cmp("int.lt", iv.clone(), n.clone())?;
                fx.b.open();
                let th = fx.b.close();
                fx.b.open();
                fx.b.brk(1);
                let el = fx.b.close();
                fx.b.push(Stmt::If(more, th, el));
                let e = fx.b.list_at(m.clone(), iv.clone())?;
                let ps = fx.b.prim("tup.open", vec![pt.clone()], vec![e])?;
                fx.b.drop(ps[1].clone())?;
                let kc = fx.b.copy(key.clone())?;
                let c = lw.call_rt(fx, cmpf, vec![ps[0].clone(), kc]);
                let below = fx.b.int_cmp("int.lt", c.clone(), lit(0))?;
                fx.b.open();
                let th = fx.b.close();
                fx.b.open();
                let iv2 = fx.b.cast_in(iv.clone(), Ty::Int(0, INT_MAX))?;
                fx.b.set(at, iv2);
                let eq = fx.b.int_cmp("int.eq", c, lit(0))?;
                fx.b.set(found, eq);
                fx.b.brk(1);
                let el = fx.b.close();
                fx.b.push(Stmt::If(below, th, el));
                let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
                fx.b.set(i, i1);
            }
            let body = fx.b.close();
            fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
            let atv = fx.b.let_(Atom::V(at));
            let fv = fx.b.let_(Atom::V(found));
            fx.b.open();
            {
                // replace the value, keeping the key that is there
                let e = fx.b.list_at(m.clone(), atv.clone())?;
                let ps = fx.b.prim("tup.open", vec![pt.clone()], vec![e])?;
                fx.b.drop(ps[1].clone())?;
                let pair = fx.b.p("tup.mk", vec![pt.clone()], vec![ps[0].clone(), val.clone()])?;
                fx.b.drop(key.clone())?;
                let n2 = fx.b.p("list.len", vec![pt.clone()], vec![m.clone()])?;
                let ok = fx.b.int_cmp("int.lt", atv.clone(), n2)?;
                fx.b.guard_halt_unless(ok, "internal: map index")?;
                let m2 = fx.b.p("list.set", vec![pt.clone()], vec![m.clone(), atv.clone(), pair])?;
                fx.b.ret(m2);
            }
            let th = fx.b.close();
            fx.b.push(Stmt::If(fv, th, vec![]));
            // insert: the items before, the new pair, the items after
            let o0 = fx.b.p("list.new", vec![pt.clone()], vec![])?;
            let out = fx.b.var(m_t.clone(), o0);
            let j = fx.b.var(Ty::u64(), lit(0));
            fx.b.open();
            {
                let jv = fx.b.let_(Atom::V(j));
                let more = fx.b.int_cmp("int.lt", jv.clone(), n.clone())?;
                fx.b.open();
                let th = fx.b.close();
                fx.b.open();
                fx.b.brk(1);
                let el = fx.b.close();
                fx.b.push(Stmt::If(more, th, el));
                let here = fx.b.int_cmp("int.eq", jv.clone(), atv.clone())?;
                fx.b.open();
                let pair = fx.b.p("tup.mk", vec![pt.clone()], vec![key.clone(), val.clone()])?;
                let o = fx.b.let_(Atom::V(out));
                let o = fx.b.p("list.push", vec![pt.clone()], vec![o, pair])?;
                fx.b.set(out, o);
                let th = fx.b.close();
                fx.b.push(Stmt::If(here, th, vec![]));
                let e = fx.b.list_at(m.clone(), jv.clone())?;
                let o = fx.b.let_(Atom::V(out));
                let o = fx.b.p("list.push", vec![pt.clone()], vec![o, e])?;
                fx.b.set(out, o);
                let j1 = fx.b.add_in(jv, lit(1), Ty::u64())?;
                fx.b.set(j, j1);
            }
            let body = fx.b.close();
            fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
            let last = fx.b.int_cmp("int.eq", atv, n)?;
            fx.b.open();
            let pair = fx.b.p("tup.mk", vec![pt.clone()], vec![key.clone(), val.clone()])?;
            let o = fx.b.let_(Atom::V(out));
            let o = fx.b.p("list.push", vec![pt.clone()], vec![o, pair])?;
            fx.b.set(out, o);
            let th = fx.b.close();
            fx.b.open();
            fx.b.drop(key)?;
            fx.b.drop(val)?;
            let el = fx.b.close();
            fx.b.push(Stmt::If(last, th, el));
            fx.b.drop(m)?;
            let o = fx.b.let_(Atom::V(out));
            fx.b.ret(o);
            Ok(())
        })
    }

    /// `mdel:Map(m, i)`: the map without its item at index i, and that item.
    fn map_del_fn(&mut self, mt: &RType) -> R<usize> {
        let m_t = self.ct(mt)?;
        let Ty::List(pt) = m_t.clone() else { return Err("internal: map_del of a non-map".into()) };
        let pt = (*pt).clone();
        let rt_ = Ty::Tup(vec![m_t.clone(), pt.clone()]);
        self.generated("mdel", mt, vec![m_t.clone(), Ty::Int(0, INT_MAX)], rt_.clone(), move |_, fx| {
            let (m, at) = (fx.b.arg(0), fx.b.arg(1));
            let n = fx.b.p("list.len", vec![pt.clone()], vec![m.clone()])?;
            let x = fx.b.list_at(m.clone(), at.clone())?;
            let o0 = fx.b.p("list.new", vec![pt.clone()], vec![])?;
            let out = fx.b.var(m_t.clone(), o0);
            let j = fx.b.var(Ty::u64(), lit(0));
            fx.b.open();
            {
                let jv = fx.b.let_(Atom::V(j));
                let more = fx.b.int_cmp("int.lt", jv.clone(), n)?;
                fx.b.open();
                let th = fx.b.close();
                fx.b.open();
                fx.b.brk(1);
                let el = fx.b.close();
                fx.b.push(Stmt::If(more, th, el));
                let j1 = fx.b.add_in(jv.clone(), lit(1), Ty::u64())?;
                fx.b.set(j, j1);
                let here = fx.b.int_cmp("int.eq", jv.clone(), at.clone())?;
                fx.b.open();
                fx.b.brk(0);
                let th = fx.b.close();
                fx.b.push(Stmt::If(here, th, vec![]));
                let e = fx.b.list_at(m.clone(), jv)?;
                let o = fx.b.let_(Atom::V(out));
                let o = fx.b.p("list.push", vec![pt.clone()], vec![o, e])?;
                fx.b.set(out, o);
            }
            let body = fx.b.close();
            fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
            fx.b.drop(m)?;
            let o = fx.b.let_(Atom::V(out));
            let r = fx.b.p("tup.mk", vec![rt_.clone()], vec![o, x])?;
            fx.b.ret(r);
            Ok(())
        })
    }
}

/// A small expression whose evaluation cannot change a variable: a name, a
/// literal, or arithmetic, comparison and field reads of those.
fn simple(e: &Expr) -> bool {
    match e {
        Expr::Name(..) | Expr::Int(..) | Expr::Bool(..) | Expr::Str(..) | Expr::SelfRef(_) => true,
        Expr::Bin(op, a, b, _) => !matches!(op.as_str(), "and" | "or") && simple(a) && simple(b),
        Expr::Neg(x, _) | Expr::Not(x, _) | Expr::Member(x, _, _) | Expr::TupleMember(x, _, _) => simple(x),
        _ => false,
    }
}

impl<'m> Lw<'m> {
    // ---- expressions

    fn bound(&self, fx: &Fx, e: &Expr) -> Option<Bound> {
        match self.ty.node(e).and_then(|n| n.bind.clone()) {
            Some(Binding::Local(id)) => fx.bound.get(&id).cloned(),
            _ => None,
        }
    }

    /// The variable an expression names, when it names a `var` local.
    fn var_of(&self, fx: &Fx, e: &Expr) -> Option<u32> {
        match self.bound(fx, e) {
            Some(Bound::Var(v, _)) => Some(v),
            _ => None,
        }
    }

    fn expr(&mut self, fx: &mut Fx, e: &Expr) -> R<Val> {
        fx.b.tick(1);
        let rt = self.node_ty(e);
        if let RType::Unknown(r) = &rt {
            return no(r);
        }
        match e {
            Expr::Name(n, _) => self.name(fx, e, n, rt),
            Expr::SelfRef(_) => self.name(fx, e, "self", rt),
            Expr::Int(v, _) => Ok(Val { a: lit(*v), t: rt }),
            Expr::Float(f, _) => Ok(Val { a: Atom::K(Ty::F64, Lit::F64(f.to_bits())), t: rt }),
            Expr::Str(s, _) => Ok(Val { a: Atom::bytes(s.as_bytes()), t: rt }),
            Expr::Bool(b, _) => Ok(Val { a: Atom::bool(*b), t: rt }),
            Expr::None(_) => {
                let t = self.ct(&rt)?;
                let u = fx.b.unit()?;
                Ok(Val { a: fx.b.p("sum.mk", vec![t], vec![lit(0), u])?, t: rt })
            }
            Expr::Unit(_) => Ok(Val { a: fx.b.unit()?, t: rt }),
            Expr::Tuple(xs, _) => {
                let RType::Tuple(ts) = rt.clone() else { return no("a tuple of unknown type") };
                let mut vs = vec![];
                for x in xs {
                    let v = self.expr(fx, x)?;
                    vs.push(v);
                }
                let mut atoms = vec![];
                for (v, t) in vs.into_iter().zip(&ts) {
                    atoms.push(self.conv(fx, v.a, &v.t, t, false)?);
                }
                let t = self.ct(&rt)?;
                Ok(Val { a: fx.b.p("tup.mk", vec![t], atoms)?, t: rt })
            }
            Expr::List(xs, _) => {
                let RType::List(et) = rt.clone() else { return no("a list of unknown type") };
                self.charge_k(fx, 4 + xs.len() as i128)?;
                let mut vs = vec![];
                for x in xs {
                    vs.push(self.expr(fx, x)?);
                }
                let ect = self.ct(&et)?;
                let mut l = fx.b.p("list.new", vec![ect.clone()], vec![])?;
                for v in vs {
                    let a = self.conv(fx, v.a, &v.t, &et, false)?;
                    l = fx.b.p("list.push", vec![ect.clone()], vec![l, a])?;
                }
                Ok(Val { a: l, t: rt })
            }
            Expr::ListComp(body, p, it, cond, _) => self.list_comp(fx, body, p, it, cond.as_deref(), rt),
            // a lambda a `let` makes: nothing is kept, since each call runs
            // its body where it was made (closure_call)
            Expr::Lambda(..) if matches!(rt, RType::Closure(_)) => Ok(Val { a: fx.b.unit()?, t: rt }),
            Expr::ListRepeat(x, n, _) => {
                // interp: the value, the count (0 to 10,000,000), then N9's limit
                let RType::List(et) = rt.clone() else { return no("a repeated list of unknown type") };
                let v = self.expr(fx, x)?;
                let k = self.expr(fx, n)?;
                let RType::Int(_) = &k.t else { return no("a repeat count that is not an integer") };
                let lo = fx.b.int_cmp("int.lt", k.a.clone(), lit(0))?;
                let hi = fx.b.int_cmp("int.lt", lit(10_000_000), k.a.clone())?;
                let bad = fx.b.p("bool.or", vec![], vec![lo, hi])?;
                self.guard(fx, bad, |_, _| Ok(Atom::bytes(b"bad repeat count")))?;
                self.charge_list(fx, k.a.clone())?;
                let big = fx.b.int_cmp("int.lt", lit(crate::interp::SIZE_LIMIT as i128), k.a.clone())?;
                self.guard(fx, big, |_, _| Ok(Atom::bytes(b"size limit")))?;
                let ect = self.ct(&et)?;
                let va = self.conv(fx, v.a, &v.t, &et, false)?;
                let o0 = fx.b.p("list.new", vec![ect.clone()], vec![])?;
                let out = fx.b.var(Ty::list(ect.clone()), o0);
                let i = fx.b.var(Ty::u64(), lit(0));
                fx.b.open();
                {
                    let iv = fx.b.let_(Atom::V(i));
                    let more = fx.b.int_cmp("int.lt", iv.clone(), k.a.clone())?;
                    fx.b.open();
                    let th = fx.b.close();
                    fx.b.open();
                    fx.b.brk(1);
                    let el = fx.b.close();
                    fx.b.push(Stmt::If(more, th, el));
                    let c = fx.b.copy(va.clone())?;
                    let o = fx.b.let_(Atom::V(out));
                    let o = fx.b.p("list.push", vec![ect.clone()], vec![o, c])?;
                    fx.b.set(out, o);
                    let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
                    fx.b.set(i, i1);
                }
                let body = fx.b.close();
                fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
                fx.b.drop(va)?;
                Ok(Val { a: fx.b.let_(Atom::V(out)), t: rt })
            }
            Expr::Map(ps, _) => {
                // interp: each key, then its value, set into the map in turn
                let RType::Map(k, v) = rt.clone() else { return no("a map of unknown type") };
                let mt = self.ct(&rt)?;
                let Ty::List(pt) = &mt else { unreachable!() };
                self.charge_k(fx, 4 + ps.len() as i128)?;
                let mut m = fx.b.p("list.new", vec![(**pt).clone()], vec![])?;
                if ps.is_empty() {
                    return Ok(Val { a: m, t: rt });
                }
                let setf = self.map_set_fn(&rt)?;
                for (a, b) in ps {
                    let kv = self.expr(fx, a)?;
                    let vv = self.expr(fx, b)?;
                    let ka = self.conv(fx, kv.a, &kv.t, &k, false)?;
                    let va = self.conv(fx, vv.a, &vv.t, &v, false)?;
                    m = self.call_rt(fx, setf, vec![m, ka, va]);
                }
                Ok(Val { a: m, t: rt })
            }
            Expr::Call(f, args, _, _) => self.call(fx, e, f, args, rt),
            Expr::Index(x, idx, _) => self.index(fx, x, idx, rt),
            Expr::Slice(x, a, b, _) => self.slice(fx, x, a.as_deref(), b.as_deref(), rt),
            Expr::Member(x, f, _) => self.member(fx, x, f, rt),
            Expr::TupleMember(x, i, _) => {
                let v = self.expr(fx, x)?;
                self.charge_k(fx, 1)?;
                let RType::Tuple(_) = &v.t else { return no("a component of a non-tuple") };
                let t = fx.b.ty(&v.a);
                let c = fx.b.p("tup.get", vec![t], vec![v.a.clone(), lit(*i as i128)])?;
                fx.b.drop(v.a)?;
                Ok(Val { a: c, t: rt })
            }
            Expr::Neg(x, _) => {
                let v = self.expr(fx, x)?;
                self.charge_k(fx, 1)?;
                match &v.t {
                    RType::Int(i) => {
                        let (r, name) = check_of(&IntT::of_tags(i.tags))?;
                        let c = fx.b.p("int.bad.neg", vec![r.clone()], vec![v.a.clone()])?;
                        let a = v.a.clone();
                        self.guard(fx, c, |lw, fx| lw.ovf(fx, "neg", vec![a], &name))?;
                        Ok(Val { a: fx.b.p("int.neg", vec![r], vec![v.a])?, t: rt })
                    }
                    RType::F64 => Ok(Val { a: fx.b.p("f64.neg", vec![], vec![v.a])?, t: rt }),
                    RType::F32 => {
                        let x = fx.b.p("f32.to_f64", vec![], vec![v.a])?;
                        let y = fx.b.p("f64.neg", vec![], vec![x])?;
                        Ok(Val { a: self.round_to_f32(fx, y)?, t: rt })
                    }
                    o => no(format!("negation of {}", o.show())),
                }
            }
            Expr::BitNot(x, _) => {
                let v = self.expr(fx, x)?;
                self.charge_k(fx, 1)?;
                let RType::Int(i) = &v.t else { return no("~ of a non-integer") };
                let Some(n) = i.tags.single() else { return no("~ of an integer whose type name differs at run time") };
                let (lo, hi) = typed::tag_range(n);
                let r = Ty::Int(lo, hi);
                if !fx.b.ty(&v.a).within(&r) {
                    return no("~ of a value outside its type's range");
                }
                Ok(Val { a: fx.b.p("int.not", vec![r], vec![v.a])?, t: rt })
            }
            Expr::Not(x, _) => {
                let v = self.expr(fx, x)?;
                self.charge_k(fx, 1)?;
                Ok(Val { a: fx.b.not(v.a)?, t: rt })
            }
            Expr::Borrow(_, x, _) | Expr::Comptime(x, _) => self.expr(fx, x),
            Expr::Cast(x, ty, _) => self.cast(fx, x, ty, rt),
            Expr::Bin(op, a, b, _) => self.bin(fx, op, a, b, rt),
            Expr::If(c, a, b, _) => {
                let cv = self.expr(fx, c)?;
                let t = self.ct(&rt)?;
                let r = self.result_var(fx, &t)?;
                fx.b.open();
                let av = self.expr(fx, a)?;
                let aa = self.conv(fx, av.a, &av.t, &rt, false)?;
                self.store_var(fx, r, aa)?;
                let th = fx.b.close();
                fx.b.open();
                let bv = self.expr(fx, b)?;
                let ba = self.conv(fx, bv.a, &bv.t, &rt, false)?;
                self.store_var(fx, r, ba)?;
                let el = fx.b.close();
                fx.b.push(Stmt::If(cv.a, th, el));
                Ok(Val { a: fx.b.let_(Atom::V(r)), t: rt })
            }
            Expr::Catch(x, bind, h, _) => self.catch_expr(fx, e, x, bind.as_deref(), h, rt),
            Expr::Try(x, els, _) => self.try_expr(fx, x, els.as_deref(), rt),
            _ => no(format!("the expression at line {}", e.line())),
        }
    }

    fn name(&mut self, fx: &mut Fx, e: &Expr, n: &str, rt: RType) -> R<Val> {
        let node = self.ty.node(e).cloned();
        match node.and_then(|x| x.bind) {
            Some(Binding::Local(id)) => match fx.bound.get(&id).cloned() {
                Some(Bound::Var(v, _)) => Ok(Val { a: self.read_var(fx, v)?, t: rt }),
                Some(Bound::Val(a, _)) => Ok(Val { a: fx.b.copy(a)?, t: rt }),
                None => no(format!("the local {n} outside its frame")),
            },
            Some(Binding::Const(c)) => self.const_read(fx, &c, rt),
            Some(Binding::Ctor(c)) => self.ctor_value(fx, &c, vec![], rt),
            Some(Binding::Fn(_)) => no("a function used as a value"),
            None => no(format!("the name {n}")),
        }
    }

    /// The value of a constructor whose fields are already fitted.
    fn ctor_value(&mut self, fx: &mut Fx, c: &str, fields: Vec<Atom>, rt: RType) -> R<Val> {
        if let Some((kind, k, _)) = self.t.variants.get(c).cloned() {
            let kt = self.ct(&RType::Named(kind))?;
            let Ty::Sum(cs) = &kt else { unreachable!() };
            let tup = fx.b.p("tup.mk", vec![cs[k].clone()], fields)?;
            return Ok(Val { a: fx.b.p("sum.mk", vec![kt.clone()], vec![lit(k as i128), tup])?, t: rt });
        }
        let t = self.ct(&RType::Named(c.to_string()))?;
        Ok(Val { a: fx.b.p("tup.mk", vec![t], fields)?, t: rt })
    }

    fn const_slot(&mut self, c: &str) -> R<(usize, u32, u32)> {
        if let Some(s) = self.consts.get(c) {
            return Ok(*s);
        }
        let Some((decl, e)) = self.t.consts.get(c).cloned() else { return no(format!("the constant {c}")) };
        let crt = match self.ty.node(e) {
            Some(n) => match &decl {
                Some(t) => self.t.fit(&n.ty, t),
                None => n.ty.clone(),
            },
            None => return no(format!("the constant {c}")),
        };
        let t = self.ct(&crt)?;
        let id = self.f.declare(&format!("const {c}"), vec![], t.clone());
        let done = self.f.global(&format!("const {c} evaluated"), Ty::Bool, Some(Lit::Bool(false)));
        let init = if t.owned() { None } else { Some(match &t {
            Ty::Bool => Lit::Bool(false),
            Ty::Int(lo, hi) => Lit::Int(if *lo <= 0 && 0 <= *hi { 0 } else { *lo }),
            Ty::F64 => Lit::F64(0),
            Ty::F32 => Lit::F32(0),
            _ => Lit::Bytes(vec![]),
        }) };
        let val = self.f.global(&format!("const {c}"), t.clone(), init);
        self.consts.insert(c.to_string(), (id, done, val));
        // the constant's expression, evaluated in a scope of its own (interp `const_value`)
        let mut fx = self.fx(vec![], t, Ret::None, FailCx::Halt, true);
        let v = self.expr(&mut fx, e)?;
        let v = match &decl {
            Some(d) => self.fit(&mut fx, v.a, &v.t, d)?,
            None => v,
        };
        let a = self.conv(&mut fx, v.a, &v.t, &crt, false)?;
        fx.b.ret(a);
        self.f.define(id, fx.b.finish());
        Ok((id, done, val))
    }

    fn const_read(&mut self, fx: &mut Fx, c: &str, rt: RType) -> R<Val> {
        let (id, done, val) = self.const_slot(c)?;
        let t = self.f.ret(id);
        let d = fx.b.let_global(done, Ty::Bool);
        fx.b.open();
        let th = fx.b.close();
        fx.b.open();
        let v = fx.b.call(sym(id), t.clone(), vec![]);
        if t.owned() {
            let old = fx.b.let_global(val, t.clone());
            fx.b.drop(old)?;
        }
        fx.b.setg(val, v);
        fx.b.setg(done, Atom::bool(true));
        let el = fx.b.close();
        fx.b.push(Stmt::If(d, th, el));
        if t.owned() {
            let x = fx.b.let_global(val, t.clone());
            let c = fx.b.p("copy", vec![t], vec![x.clone()])?;
            fx.b.setg(val, x);
            Ok(Val { a: c, t: rt })
        } else {
            Ok(Val { a: fx.b.let_global(val, t), t: rt })
        }
    }

    /// The arguments of a call, each fitted to its parameter (interp `call_fn`).
    fn args(&mut self, fx: &mut Fx, f: &FnDecl, args: &[Arg]) -> R<Vec<Atom>> {
        let ps: Vec<ast::Ty> = f.sig.params.iter().filter_map(|p| if let Param::Named { ty, .. } = p { Some(ty.clone()) } else { None }).collect();
        if ps.len() != args.len() {
            return no("a call whose arguments do not match its parameters one to one");
        }
        let mut out = vec![];
        for (p, a) in ps.iter().zip(args) {
            let v = self.expr(fx, &a.e)?;
            let v = self.fit_decl(fx, v.a, &v.t, p)?;
            out.push(v.a);
        }
        Ok(out)
    }

    /// Call a function of the module. With `&mut` parameters its result is
    /// unpacked, and the parameters' final values are written back to the
    /// places the arguments name, `&mut self` last (interp `call_fn`'s copy
    /// out), unless the call returned a union function's error (F.4.4).
    fn call_user(&mut self, fx: &mut Fx, id: usize, fd: &FnDecl, a: Vec<Atom>, args: &[Arg], recv: Option<&Expr>) -> R<Atom> {
        let r = self.f.ret(id);
        let c = fx.b.call(sym(id), r.clone(), a);
        let muts = Self::mut_params(fd);
        if muts.is_empty() {
            return Ok(c);
        }
        let ps = fx.b.prim("tup.open", vec![r], vec![c])?;
        let res = ps[0].clone();
        let mut places: Vec<(&Expr, RType, Atom)> = vec![];
        for (j, k) in muts.iter().enumerate() {
            match &fd.sig.params[*k] {
                Param::Named { ty, .. } => {
                    let named_k = fd.sig.params[..*k].iter().filter(|p| matches!(p, Param::Named { .. })).count();
                    let Some(arg) = args.get(named_k) else { return no("a &mut parameter without its argument") };
                    let place = match &arg.e {
                        Expr::Borrow(_, x, _) => &**x,
                        e => e,
                    };
                    places.push((place, self.t.of_ty(ty), ps[j + 1].clone()));
                }
                _ => {
                    let Some(obj) = recv else { return no("a &mut self without its receiver") };
                    let st = match self.ty.params.get(&addr(fd)).and_then(|(_, s)| *s) {
                        Some(sid) => self.ty.locals[sid as usize].ty.clone(),
                        None => return no("the type of a &mut self"),
                    };
                    places.push((obj, st, ps[j + 1].clone()));
                }
            }
        }
        let write = |lw: &mut Self, fx: &mut Fx, places: Vec<(&Expr, RType, Atom)>| -> R<()> {
            for (place, t, a) in places {
                lw.store_place(fx, place, Val { a, t })?;
            }
            Ok(())
        };
        if matches!(&fd.sig.ret, Some(ast::Ty::Union(..))) {
            let rt_ = fx.b.ty(&res);
            let tag = fx.b.p("sum.tag", vec![rt_], vec![res.clone()])?;
            let ok = fx.b.int_cmp("int.eq", tag, lit(0))?;
            fx.b.open();
            let drops: Vec<Atom> = places.iter().map(|p| p.2.clone()).collect();
            write(self, fx, places)?;
            let th = fx.b.close();
            fx.b.open();
            for d in drops {
                fx.b.drop(d)?;
            }
            let el = fx.b.close();
            fx.b.push(Stmt::If(ok, th, el));
        } else {
            write(self, fx, places)?;
        }
        Ok(res)
    }

    /// The instantiation `key` of `fd` (typed::Inst), lowered on its first use.
    fn inst_fn(&mut self, key: &str, fd: &FnDecl) -> R<usize> {
        if let Some(id) = self.inst_ids.get(key) {
            return Ok(*id);
        }
        let Some(inst) = self.insts.remove(key) else { return no(format!("the instantiation {key}")) };
        let mut ps = vec![];
        for p in &inst.params {
            ps.push(self.ct(p)?);
        }
        let mut ret = self.ct(&inst.ret)?;
        let muts = Self::mut_params(fd);
        if !muts.is_empty() {
            let mut rs = vec![ret];
            for k in muts {
                rs.push(ps[k].clone());
            }
            ret = Ty::Tup(rs);
        }
        let id = self.f.declare(&format!("fn {key}"), ps.clone(), ret.clone());
        self.inst_ids.insert(key.to_string(), id);
        self.inst_sigs.insert(id, (inst.params.clone(), inst.ret.clone()));
        let self_t = if fd.sig.params.iter().any(|p| !matches!(p, Param::Named { .. })) { inst.params.first().cloned() } else { None };
        let saved = std::mem::replace(&mut self.ty, inst.typed);
        let mut gs: Vec<String> = fd.sig.generics.iter().map(|g| g.name.clone()).collect();
        if let Some(head) = key.split_once('.').map(|x| x.0.to_string()).filter(|_| self_t.is_some()) {
            gs.extend(self.t.impl_generics.get(&(head, fd.sig.name.clone())).cloned().unwrap_or_default());
        }
        let saved_gs = std::mem::replace(&mut self.gs, gs);
        let r = self.lower_fn_sig(id, fd, self_t.as_ref(), ps, ret, Some(inst.ret.clone()));
        self.ty = saved;
        self.gs = saved_gs;
        r?;
        Ok(id)
    }

    /// A call of an instantiation: each argument fitted as the reference
    /// fits it (a type parameter keeps the value), then the call.
    fn call_inst(&mut self, fx: &mut Fx, key: &str, fd: &FnDecl, args: &[Arg], recv: Option<(Atom, &Expr)>) -> R<Atom> {
        let id = self.inst_fn(key, fd)?;
        let (params, _) = self.inst_sigs[&id].clone();
        let named = Self::named_params(fd);
        if named.len() != args.len() {
            return no("a call whose arguments do not match its parameters one to one");
        }
        let mut a = vec![];
        let place = recv.as_ref().map(|(_, e)| *e);
        let skip = match recv {
            Some((r, _)) => {
                a.push(r);
                1
            }
            None => 0,
        };
        for (k, (ty, arg)) in named.iter().zip(args).enumerate() {
            let v = self.expr(fx, &arg.e)?;
            let want = &params[k + skip];
            let x = if self.t.of_ty(ty).is_known() && !matches!(ty, ast::Ty::Ref(..)) {
                let v = self.fit(fx, v.a, &v.t, ty)?;
                self.conv(fx, v.a, &v.t, want, false)?
            } else {
                self.conv(fx, v.a, &v.t, want, true)?
            };
            a.push(x);
        }
        self.call_user(fx, id, fd, a, args, place)
    }

    /// The function named `name`, lowered on its first use.
    fn user_fn(&mut self, name: &str) -> R<usize> {
        if let Some(id) = self.fns.get(name) {
            return Ok(*id);
        }
        let Some(f) = self.t.fns.get(name).copied() else { return no(format!("the function {name}")) };
        let (ps, r) = self.fn_sig(f, None)?;
        let id = self.f.declare(&format!("fn {name}"), ps, r);
        self.fns.insert(name.to_string(), id);
        self.lower_fn(id, f, None)?;
        Ok(id)
    }

    /// The method `m` of the impl of type head `h`, lowered on its first use.
    fn method_fn(&mut self, h: &str, m: &str) -> R<usize> {
        if let Some(id) = self.methods.get(&(h.to_string(), m.to_string())) {
            return Ok(*id);
        }
        let (Some(f), Some(ty)) = (self.t.methods.get(&(h.to_string(), m.to_string())).copied(), self.impl_tys.get(&(h.to_string(), m.to_string())).cloned()) else {
            return no(format!("the method {m}"));
        };
        let st = self.t.of_ty(&ty);
        let (ps, r) = self.fn_sig(f, Some(&st))?;
        let id = self.f.declare(&format!("fn {h}.{m}"), ps, r);
        self.methods.insert((h.to_string(), m.to_string()), id);
        self.lower_fn(id, f, Some(&st))?;
        Ok(id)
    }

    fn call(&mut self, fx: &mut Fx, e: &Expr, f: &Expr, args: &[Arg], rt: RType) -> R<Val> {
        let callee = self.ty.node(e).and_then(|n| n.callee.clone());
        match callee {
            Some(Callee::Some) => {
                let RType::Opt(inner) = rt.clone() else { return no("some(…) of unknown type") };
                let v = self.expr(fx, &args[0].e)?;
                let a = self.conv(fx, v.a, &v.t, &inner, false)?;
                let t = self.ct(&rt)?;
                Ok(Val { a: fx.b.p("sum.mk", vec![t], vec![lit(1), a])?, t: rt })
            }
            Some(Callee::Same) => self.expr(fx, &args[0].e),
            Some(Callee::Fn(n)) if matches!(self.t.fns[&n].sig.ret, Some(ast::Ty::Iter(_))) => self.iter_call(fx, e, &n, args, rt),
            Some(Callee::Fn(n)) if self.ty.inst_of.contains_key(&addr(e)) => {
                let fd = self.t.fns[&n];
                let key = self.ty.inst_of[&addr(e)].clone();
                fx.b.tick(1);
                self.charge_k(fx, crate::conc::C_CALL)?;
                Ok(Val { a: self.call_inst(fx, &key, fd, args, None)?, t: rt })
            }
            Some(Callee::Method(h, m)) if self.ty.inst_of.contains_key(&addr(e)) => {
                let Expr::Member(obj, _, _) = f else { return no("a method call") };
                let fd = self.t.methods[&(h.clone(), m.clone())];
                let key = self.ty.inst_of[&addr(e)].clone();
                let recv = self.expr(fx, obj)?;
                fx.b.tick(1);
                self.charge_k(fx, crate::conc::C_CALL)?;
                Ok(Val { a: self.call_inst(fx, &key, fd, args, Some((recv.a, obj)))?, t: rt })
            }
            Some(Callee::Fn(n)) => {
                let fd = self.t.fns[&n];
                let id = self.user_fn(&n)?;
                fx.b.tick(1);
                self.charge_k(fx, crate::conc::C_CALL)?;
                let a = self.args(fx, fd, args)?;
                Ok(Val { a: self.call_user(fx, id, fd, a, args, None)?, t: rt })
            }
            Some(Callee::Static(tn, m)) => {
                let fd = self.t.methods[&(tn.clone(), m.clone())];
                let id = self.method_fn(&tn, &m)?;
                fx.b.tick(1);
                self.charge_k(fx, crate::conc::C_CALL)?;
                let a = self.args(fx, fd, args)?;
                Ok(Val { a: self.call_user(fx, id, fd, a, args, None)?, t: rt })
            }
            Some(Callee::Method(h, m)) => {
                let Expr::Member(obj, _, _) = f else { return no("a method call") };
                let fd = self.t.methods[&(h.clone(), m.clone())];
                let id = self.method_fn(&h, &m)?;
                let recv = self.expr(fx, obj)?;
                fx.b.tick(1);
                self.charge_k(fx, crate::conc::C_CALL)?;
                let mut a = vec![recv.a];
                a.extend(self.args(fx, fd, args)?);
                Ok(Val { a: self.call_user(fx, id, fd, a, args, Some(obj))?, t: rt })
            }
            Some(Callee::Ctor(c)) => self.construct(fx, &c, args, rt),
            Some(Callee::Builtin(m)) => {
                let Expr::Member(obj, _, _) = f else { return no("a method call") };
                let sized = matches!(e, Expr::Call(_, _, _, true));
                self.builtin(fx, obj, &m, args, rt, sized)
            }
            Some(Callee::Ctx(c, m)) => {
                // 5.16: `C.lit(t)` is t; `C.encode(t)` escapes t for C
                let v = self.expr(fx, &args[0].e)?;
                if v.t != RType::Text {
                    return no(format!("{c}.{m} of {}", v.t.show()));
                }
                // C.16: `lit` costs c_call, `encode` c_call and its text's bytes
                if self.costs {
                    let w = if m == "encode" { fx.b.p("bytes.len", vec![], vec![v.a.clone()])? } else { lit(0) };
                    let w = fx.b.cast_in(w, Ty::Int(0, INT_MAX))?;
                    let u = fx.b.p("int.sadd", vec![Ty::u64()], vec![w, lit(crate::conc::C_CALL)])?;
                    self.charge(fx, u)?;
                }
                match (m.as_str(), c.as_str()) {
                    ("lit", "Html" | "Url" | "Sql" | "Json" | "ShellArg" | "Prompt" | "Header") => Ok(Val { a: v.a, t: rt }),
                    ("encode", "Html" | "Url" | "Sql" | "Json" | "ShellArg" | "Header") => {
                        let g = rtt::encode(&mut self.f, &c)?;
                        Ok(Val { a: self.call_rt(fx, g, vec![v.a]), t: rt })
                    }
                    _ => no(format!("{c}.{m}")),
                }
            }
            Some(Callee::SetOf) => {
                // interp `set_of`: the items in canonical order, without duplicates
                let v = self.expr(fx, &args[0].e)?;
                let RType::List(et) = &v.t else { return no(format!("set_of of {}", v.t.show())) };
                let et = (**et).clone();
                let g = self.set_of_fn(&et)?;
                let a = self.call_rt(fx, g, vec![v.a]);
                let a = self.conv(fx, a, &RType::Set(Box::new(et)), &rt, false)?;
                Ok(Val { a, t: rt })
            }
            Some(Callee::Closure(l)) => self.closure_call(fx, f, args, l, rt),
            Some(Callee::Trap(msg)) => {
                self.trap(fx, Atom::bytes(msg.as_bytes()))?;
                Ok(Val { a: fx.b.unit()?, t: RType::Never })
            }
            Some(Callee::Unknown(r)) => no(r),
            None => no(format!("the call at line {}", e.line())),
        }
    }

    /// interp `construct`: the fields in argument order, each fitted, then
    /// the missing fields from their initial values.
    fn construct(&mut self, fx: &mut Fx, c: &str, args: &[Arg], rt: RType) -> R<Val> {
        let fields: Vec<(String, ast::Ty, Option<Expr>)> = if let Some((_, _, p)) = self.t.variants.get(c) {
            p.iter().map(|(a, b)| (a.clone(), b.clone(), None)).collect()
        } else {
            match self.t.types.get(c) {
                Some(Item::Struct { fields, .. }) => fields.iter().map(|f| (f.name.clone(), f.ty.clone(), f.init.clone())).collect(),
                Some(Item::Event { payload, .. } | Item::Error { payload, .. }) => payload.iter().map(|(a, b)| (a.clone(), b.clone(), None)).collect(),
                _ => return no(format!("a constructor of {c}")),
            }
        };
        let mut vals: Vec<Option<Atom>> = vec![None; fields.len()];
        for (i, a) in args.iter().enumerate() {
            let k = match &a.label {
                Some(l) => fields.iter().position(|f| &f.0 == l).unwrap_or(i),
                None => i,
            };
            let v = self.expr(fx, &a.e)?;
            if k < fields.len() {
                let v = self.fit_decl(fx, v.a, &v.t, &fields[k].1)?;
                if let Some(old) = vals[k].take() {
                    fx.b.drop(old)?;
                }
                vals[k] = Some(v.a);
            } else {
                fx.b.drop(v.a)?;
            }
        }
        let mut out = vec![];
        for (k, (f, t, init)) in fields.iter().enumerate() {
            match vals[k].take() {
                Some(v) => out.push(v),
                None => match init {
                    Some(_) => {
                        // the initial value is evaluated here, on each construction
                        let Some(Item::Struct { fields: fs, .. }) = self.t.types.get(c) else { unreachable!() };
                        let ie = fs[k].init.as_ref().unwrap();
                        let v = self.expr(fx, ie)?;
                        let v = self.fit_decl(fx, v.a, &v.t, t)?;
                        out.push(v.a);
                    }
                    None => {
                        self.trap(fx, Atom::bytes(format!("missing field {f}").as_bytes()))?;
                        return no("a constructor that always traps");
                    }
                },
            }
        }
        self.ctor_value(fx, c, out, rt)
    }

    fn builtin(&mut self, fx: &mut Fx, obj: &Expr, m: &str, args: &[Arg], rt: RType, sized: bool) -> R<Val> {
        let ort = self.node_ty(obj);
        // a method that writes its receiver back needs the receiver to be a variable
        if matches!(m, "push" | "append") {
            let Some(v) = self.var_of(fx, obj) else { return no(format!("{m} on a receiver that is not a variable")) };
            fx.b.tick(1); // the receiver's evaluation
            let x = self.expr(fx, &args[0].e)?;
            self.charge_method(fx, m, (Atom::V(v), ort.clone()), Some((&x.a, &x.t)), sized)?;
            let RType::List(_) = &ort else { return no(format!("{m} on {}", ort.show())) };
            let lt = fx.b.local_ty(v);
            let Ty::List(et) = &lt else { return no("push onto a non-list") };
            let et = (**et).clone();
            let st = self.ty.locals.iter().map(|_| ()).count();
            let _ = st;
            let elem_rt = match self.var_rtype(fx, v) {
                Some(RType::List(e)) => *e,
                _ => return no("push onto a list of unknown type"),
            };
            let xa = self.conv(fx, x.a, &x.t, &elem_rt, false)?;
            let l = self.take_var(fx, v);
            let l2 = fx.b.p("list.push", vec![et.clone()], vec![l, xa])?;
            self.size_check(fx, &l2, &et)?;
            fx.b.set(v, l2);
            return Ok(Val { a: fx.b.unit()?, t: rt });
        }
        if matches!(m, "map" | "filter" | "fold") && matches!(&ort, RType::List(_)) {
            let fa = if m == "fold" { args.get(1) } else { args.first() };
            return match fa {
                Some(Arg { e: Expr::Lambda(ps, body, _), .. }) => self.hof(fx, obj, m, args, ps, body, rt, sized),
                _ => no(format!("{m} with a function that is not a lambda")),
            };
        }
        if matches!(m, "insert" | "remove") && matches!(&ort, RType::Map(..)) {
            let Some(v) = self.var_of(fx, obj) else { return no(format!("{m} on a receiver that is not a variable")) };
            fx.b.tick(1); // the receiver's evaluation
            let mut av = vec![];
            for a in args {
                av.push(self.expr(fx, &a.e)?);
            }
            let first = av.first().map(|x| (x.a.clone(), x.t.clone()));
            self.charge_method(fx, m, (Atom::V(v), ort.clone()), first.as_ref().map(|(a, t)| (a, t)), sized)?;
            let Some(mt @ RType::Map(..)) = self.var_rtype(fx, v) else { return no(format!("{m} on a map of unknown type")) };
            let RType::Map(kt, vt) = &mt else { unreachable!() };
            let (kt, vt) = ((**kt).clone(), (**vt).clone());
            if m == "insert" {
                if av.len() != 2 {
                    return no("insert without a key and a value");
                }
                let x = av.pop().unwrap();
                let k = av.pop().unwrap();
                let setf = self.map_set_fn(&mt)?;
                let ka = self.conv(fx, k.a, &k.t, &kt, false)?;
                let xa = self.conv(fx, x.a, &x.t, &vt, false)?;
                let l = self.take_var(fx, v);
                let l2 = self.call_rt(fx, setf, vec![l, ka, xa]);
                fx.b.set(v, l2);
                return Ok(Val { a: fx.b.unit()?, t: rt });
            }
            let k = av.remove(0);
            let RType::Opt(inner) = &rt else { return no("remove of unknown type") };
            let inner = (**inner).clone();
            let i = self.map_find(fx, &Val { a: Atom::V(v), t: mt.clone() }, k.clone())?;
            fx.b.drop(k.a)?;
            let t = self.ct(&rt)?;
            let r = self.result_var(fx, &t)?;
            let found = fx.b.int_cmp("int.le", lit(0), i.clone())?;
            fx.b.open();
            {
                let del = self.map_del_fn(&mt)?;
                let l = self.take_var(fx, v);
                let ii = fx.b.cast_in(i, Ty::Int(0, INT_MAX))?;
                let pr = self.call_rt(fx, del, vec![l, ii]);
                let prt = fx.b.ty(&pr);
                let ps = fx.b.prim("tup.open", vec![prt], vec![pr])?;
                fx.b.set(v, ps[0].clone());
                let Ty::List(pt) = self.ct(&mt)? else { unreachable!() };
                let kv = fx.b.prim("tup.open", vec![(*pt).clone()], vec![ps[1].clone()])?;
                fx.b.drop(kv[0].clone())?;
                let x = self.conv(fx, kv[1].clone(), &vt, &inner, false)?;
                let sm = fx.b.p("sum.mk", vec![t.clone()], vec![lit(1), x])?;
                self.store_var(fx, r, sm)?;
            }
            let th = fx.b.close();
            fx.b.push(Stmt::If(found, th, vec![]));
            return Ok(Val { a: fx.b.let_(Atom::V(r)), t: rt });
        }
        if m == "sort" && matches!(&ort, RType::List(_)) && args.is_empty() {
            let Some(v) = self.var_of(fx, obj) else { return no("sort on a receiver that is not a variable") };
            fx.b.tick(1); // the receiver's evaluation
            self.charge_method(fx, m, (Atom::V(v), ort.clone()), None, sized)?;
            let Some(RType::List(et)) = self.var_rtype(fx, v) else { return no("sort of a list of unknown type") };
            let sf = self.sort_fn(&et)?;
            let l = self.take_var(fx, v);
            let l2 = self.call_rt(fx, sf, vec![l]);
            fx.b.set(v, l2);
            return Ok(Val { a: fx.b.unit()?, t: rt });
        }
        if matches!(m, "pop" | "clear")
            && matches!(&ort, RType::List(_))
            && args.is_empty()
        {
            let Some(v) = self.var_of(fx, obj) else { return no(format!("{m} on a receiver that is not a variable")) };
            fx.b.tick(1); // the receiver's evaluation
            self.charge_method(fx, m, (Atom::V(v), ort.clone()), None, sized)?;
            let lt = fx.b.local_ty(v);
            let Ty::List(et) = &lt else { return no(format!("{m} on a non-list")) };
            let et = (**et).clone();
            if m == "clear" {
                let l = self.take_var(fx, v);
                fx.b.drop(l)?;
                let e = fx.b.p("list.new", vec![et], vec![])?;
                fx.b.set(v, e);
                return Ok(Val { a: fx.b.unit()?, t: rt });
            }
            let RType::Opt(inner) = &rt else { return no("pop of unknown type") };
            let inner = (**inner).clone();
            let Some(RType::List(st)) = self.var_rtype(fx, v) else { return no("pop of a list of unknown type") };
            let t = self.ct(&rt)?;
            let r = self.result_var(fx, &t)?;
            let n = fx.b.p("list.len", vec![et.clone()], vec![Atom::V(v)])?;
            let z = fx.b.int_cmp("int.eq", n, lit(0))?;
            fx.b.open();
            let th = fx.b.close();
            fx.b.open();
            {
                let l = self.take_var(fx, v);
                let ps = fx.b.prim("list.pop", vec![et.clone()], vec![l])?;
                fx.b.set(v, ps[0].clone());
                let x = self.conv(fx, ps[1].clone(), &st, &inner, false)?;
                let sm = fx.b.p("sum.mk", vec![t.clone()], vec![lit(1), x])?;
                self.store_var(fx, r, sm)?;
            }
            let el = fx.b.close();
            fx.b.push(Stmt::If(z, th, el));
            return Ok(Val { a: fx.b.let_(Atom::V(r)), t: rt });
        }
        // a read of a variable's length needs no copy
        if matches!(m, "len" | "len_bytes" | "isempty")
            && let Some(v) = self.var_of(fx, obj)
            && args.is_empty()
        {
            fx.b.tick(1);
            self.charge_method(fx, m, (Atom::V(v), ort.clone()), None, sized)?;
            let lt = fx.b.local_ty(v);
            let n = match &lt {
                Ty::List(e) => fx.b.p("list.len", vec![(**e).clone()], vec![Atom::V(v)])?,
                Ty::Bytes => {
                    let x = fx.b.let_(Atom::V(v));
                    fx.b.p("bytes.len", vec![], vec![x])?
                }
                _ => return no(format!("{m} of {}", ort.show())),
            };
            if m == "isempty" {
                return Ok(Val { a: fx.b.int_cmp("int.eq", n, lit(0))?, t: rt });
            }
            return Ok(Val { a: n, t: rt });
        }
        let recv = self.expr(fx, obj)?;
        let mut av = vec![];
        for a in args {
            av.push(self.expr(fx, &a.e)?);
        }
        let first = av.first().map(|x| (x.a.clone(), x.t.clone()));
        self.charge_method(fx, m, (recv.a.clone(), recv.t.clone()), first.as_ref().map(|(a, t)| (a, t)), sized)?;
        match (&recv.t, m) {
            (RType::Int(i), "abs") => {
                let (r, name) = check_of(&IntT::of_tags(i.tags))?;
                let c = fx.b.p("int.bad.abs", vec![r.clone()], vec![recv.a.clone()])?;
                let a = recv.a.clone();
                self.guard(fx, c, |lw, fx| lw.ovf(fx, "abs", vec![a], &name))?;
                Ok(Val { a: fx.b.p("int.abs", vec![r], vec![recv.a])?, t: rt })
            }
            (RType::Int(_), "min" | "max") => {
                let y = av.remove(0);
                let t = self.ct(&rt)?;
                Ok(Val { a: fx.b.p(&format!("int.{m}"), vec![t], vec![recv.a, y.a])?, t: rt })
            }
            (RType::Int(i), "wrapping_add" | "wrapping_sub" | "wrapping_mul" | "saturating_add" | "saturating_sub" | "saturating_mul") => {
                let y = av.remove(0);
                let Some((lo, hi)) = IntT::of_tags(i.tags).check_range() else { return no(format!("{m} on an integer whose range differs at run time")) };
                let p = format!("int.{}{}", if m.starts_with('w') { "w" } else { "s" }, &m[m.len() - 3..]);
                Ok(Val { a: fx.b.p(&p, vec![Ty::Int(lo, hi)], vec![recv.a, y.a])?, t: rt })
            }
            (RType::Int(i), "checked_add" | "checked_sub" | "checked_mul" | "checked_div") => {
                let y = av.remove(0);
                let RType::Int(j) = &y.t else { return no(format!("{m} of a non-integer")) };
                let (r, _) = check_of(&IntT::of_tags(i.tags.join_op(j.tags)))?;
                let t = self.ct(&rt)?;
                let out = self.result_var(fx, &t)?;
                let op = &m[8..];
                let bad = if op == "div" {
                    let z = fx.b.int_cmp("int.eq", y.a.clone(), lit(0))?;
                    let q = fx.b.var(Ty::Bool, Atom::bool(true));
                    fx.b.open();
                    let th = fx.b.close();
                    fx.b.open();
                    let bq = fx.b.p("int.bad.quo", vec![r.clone()], vec![recv.a.clone(), y.a.clone()])?;
                    fx.b.set(q, bq);
                    let el = fx.b.close();
                    fx.b.push(Stmt::If(z, th, el));
                    fx.b.let_(Atom::V(q))
                } else {
                    fx.b.p(&format!("int.bad.{op}"), vec![r.clone()], vec![recv.a.clone(), y.a.clone()])?
                };
                fx.b.open();
                let u = fx.b.unit()?;
                let none = fx.b.p("sum.mk", vec![t.clone()], vec![lit(0), u])?;
                self.store_var(fx, out, none)?;
                let th = fx.b.close();
                fx.b.open();
                if op == "div" {
                    // inside: the divisor is not 0 and the quotient fits
                    let z = fx.b.int_cmp("int.eq", y.a.clone(), lit(0))?;
                    self.guard(fx, z, |_, _| Ok(Atom::bytes(b"internal: divisor")))?;
                    let bq = fx.b.p("int.bad.quo", vec![r.clone()], vec![recv.a.clone(), y.a.clone()])?;
                    self.guard(fx, bq, |_, _| Ok(Atom::bytes(b"internal: quotient")))?;
                    let q = fx.b.p("int.quo", vec![r], vec![recv.a, y.a])?;
                    let s = fx.b.p("sum.mk", vec![t.clone()], vec![lit(1), q])?;
                    self.store_var(fx, out, s)?;
                } else {
                    let v = fx.b.p(&format!("int.{op}"), vec![r], vec![recv.a, y.a])?;
                    let s = fx.b.p("sum.mk", vec![t.clone()], vec![lit(1), v])?;
                    self.store_var(fx, out, s)?;
                }
                let el = fx.b.close();
                fx.b.push(Stmt::If(bad, th, el));
                Ok(Val { a: fx.b.let_(Atom::V(out)), t: rt })
            }
            (RType::Int(_), "tostr") => Ok(Val { a: self.dec(fx, recv.a)?, t: rt }),
            (RType::F64, "tostr") => {
                let d = rtf::text(&mut self.f)?;
                Ok(Val { a: self.call_rt(fx, d, vec![recv.a]), t: rt })
            }
            (RType::F32, "tostr") => {
                let x = fx.b.p("f32.to_f64", vec![], vec![recv.a])?;
                let d = rtf::text32(&mut self.f)?;
                Ok(Val { a: self.call_rt(fx, d, vec![x]), t: rt })
            }
            (RType::F64, "sqrt") => {
                // interp: a negative operand traps; sqrt(-0.0) is -0.0
                let c = fx.b.p("f64.bad.sqrt", vec![], vec![recv.a.clone()])?;
                self.guard(fx, c, |_, _| Ok(Atom::bytes(b"non-finite float")))?;
                Ok(Val { a: fx.b.p("f64.sqrt", vec![], vec![recv.a])?, t: rt })
            }
            (RType::F64, "abs" | "floor" | "ceil" | "round") => Ok(Val { a: fx.b.p(&format!("f64.{m}"), vec![], vec![recv.a])?, t: rt }),
            (RType::F64, "min" | "max") => {
                let y = av.remove(0);
                if y.t != RType::F64 {
                    return no(format!("{m} of f64 with {}", y.t.show()));
                }
                Ok(Val { a: fx.b.p(&format!("f64.{m}"), vec![], vec![recv.a, y.a])?, t: rt })
            }
            (RType::F32, "abs") => {
                let x = fx.b.p("f32.to_f64", vec![], vec![recv.a])?;
                let y = fx.b.p("f64.abs", vec![], vec![x])?;
                Ok(Val { a: self.round_to_f32(fx, y)?, t: rt })
            }
            (RType::Bool, "tostr") => {
                let r = fx.b.var(Ty::Bytes, Atom::bytes(b"false"));
                fx.b.open();
                fx.b.set(r, Atom::bytes(b"true"));
                let th = fx.b.close();
                fx.b.push(Stmt::If(recv.a, th, vec![]));
                Ok(Val { a: fx.b.let_(Atom::V(r)), t: rt })
            }
            (RType::Text, "tostr" | "raw") => Ok(recv),
            (RType::Text, "strip" | "upper" | "lower" | "reverse" | "chars" | "words" | "lines" | "isdigit" | "isalpha" | "isspace" | "toint" | "sha256" | "len_chars") => {
                let g = match m {
                    "strip" => rtt::strip(&mut self.f)?,
                    "upper" => rtt::case(&mut self.f, true)?,
                    "lower" => rtt::case(&mut self.f, false)?,
                    "reverse" => rtt::reverse(&mut self.f)?,
                    "chars" => rtt::chars(&mut self.f)?,
                    "words" => rtt::words(&mut self.f)?,
                    "lines" => rtt::lines(&mut self.f)?,
                    "toint" => rtt::toint(&mut self.f)?,
                    "sha256" => rt::sha_hex(&mut self.f)?,
                    "len_chars" => rt::utf8_count(&mut self.f)?,
                    c => rtt::is_class(&mut self.f, c)?,
                };
                Ok(Val { a: self.call_rt(fx, g, vec![recv.a]), t: rt })
            }
            (RType::Text, "split" | "contains" | "startswith" | "endswith") => {
                let d = av.remove(0);
                if d.t != RType::Text {
                    // interp: a trap names the method
                    return no(format!("{m} with {}", d.t.show()));
                }
                let g = match m {
                    "split" => rtt::split(&mut self.f)?,
                    "contains" => rtt::contains(&mut self.f)?,
                    "startswith" => rtt::affix(&mut self.f, true)?,
                    _ => rtt::affix(&mut self.f, false)?,
                };
                Ok(Val { a: self.call_rt(fx, g, vec![recv.a, d.a]), t: rt })
            }
            (RType::Text, "join") => {
                let xs = av.remove(0);
                let RType::List(et) = &xs.t else { return no(format!("join with {}", xs.t.show())) };
                let l = if **et == RType::Text {
                    xs.a
                } else {
                    // a value that is not text is joined as display shows it
                    let et = (**et).clone();
                    self.map_list(fx, xs.a, &et, Ty::Bytes, |lw, fx, x| lw.show(fx, x, &et))?
                };
                let g = rtt::join(&mut self.f)?;
                Ok(Val { a: self.call_rt(fx, g, vec![recv.a, l]), t: rt })
            }
            (RType::Map(..), "len") => {
                let t = fx.b.ty(&recv.a);
                let Ty::List(e) = &t else { unreachable!() };
                let n = fx.b.p("list.len", vec![(**e).clone()], vec![recv.a.clone()])?;
                fx.b.drop(recv.a)?;
                Ok(Val { a: n, t: rt })
            }
            (RType::Map(..), "contains") => {
                let k = av.remove(0);
                let i = self.map_find(fx, &recv, k.clone())?;
                fx.b.drop(recv.a)?;
                fx.b.drop(k.a)?;
                Ok(Val { a: fx.b.int_cmp("int.le", lit(0), i)?, t: rt })
            }
            (RType::Map(_, vt), "get") => {
                // the value as the ok side of a union, or the error Missing
                let vt = (**vt).clone();
                let k = av.remove(0);
                let RType::Union(okt, _) = &rt else { return no("get of unknown type") };
                let okt = (**okt).clone();
                let t = self.ct(&rt)?;
                let Ty::Sum(cs) = &t else { unreachable!() };
                if cs[1] != Ty::unit() {
                    return no("get of a map where the error Missing has fields");
                }
                let i = self.map_find(fx, &recv, k.clone())?;
                fx.b.drop(k.a)?;
                let r = self.result_var(fx, &t)?;
                let found = fx.b.int_cmp("int.le", lit(0), i.clone())?;
                fx.b.open();
                let x = self.map_value_at(fx, recv.clone(), i)?;
                let x = self.conv(fx, x.a, &vt, &okt, false)?;
                let sm = fx.b.p("sum.mk", vec![t.clone()], vec![lit(0), x])?;
                self.store_var(fx, r, sm)?;
                let th = fx.b.close();
                fx.b.open();
                fx.b.drop(recv.a)?;
                let u = fx.b.unit()?;
                let sm = fx.b.p("sum.mk", vec![t.clone()], vec![lit(1), u])?;
                self.store_var(fx, r, sm)?;
                let el = fx.b.close();
                fx.b.push(Stmt::If(found, th, el));
                Ok(Val { a: fx.b.let_(Atom::V(r)), t: rt })
            }
            (RType::Map(kt, vt), "keys" | "values") => {
                let (kt, vt) = ((**kt).clone(), (**vt).clone());
                let pair = RType::Tuple(vec![kt.clone(), vt.clone()]);
                let pt = self.ct(&pair)?;
                let (et, k) = if m == "keys" { (kt, 0usize) } else { (vt, 1usize) };
                let ect = self.ct(&et)?;
                let l = self.map_list(fx, recv.a, &pair, ect, move |_, fx, p| {
                    let ps = fx.b.prim("tup.open", vec![pt.clone()], vec![p])?;
                    fx.b.drop(ps[1 - k].clone())?;
                    Ok(ps[k].clone())
                })?;
                let a = self.conv(fx, l, &RType::List(Box::new(et)), &rt, false)?;
                Ok(Val { a, t: rt })
            }
            (RType::List(et), "utf8") => {
                match &**et {
                    RType::Never => {
                        fx.b.drop(recv.a)?;
                        return Ok(Val { a: Atom::bytes(b""), t: rt });
                    }
                    RType::Int(_) => {}
                    o => return no(format!("utf8 of a list of {}", o.show())),
                }
                let ect = self.ct(et)?;
                let g = rtt::utf8(&mut self.f, ect)?;
                Ok(Val { a: self.call_rt(fx, g, vec![recv.a]), t: rt })
            }
            (RType::List(et), "contains") => {
                let x = av.remove(0);
                let et = (**et).clone();
                let g = self.list_contains_fn(&et, &x.t)?;
                Ok(Val { a: self.call_rt(fx, g, vec![recv.a, x.a]), t: rt })
            }
            (RType::List(et), "reverse") => {
                let et = (**et).clone();
                let ect = self.ct(&et)?;
                let n = fx.b.p("list.len", vec![ect.clone()], vec![recv.a.clone()])?;
                let o0 = fx.b.p("list.new", vec![ect.clone()], vec![])?;
                let out = fx.b.var(Ty::list(ect.clone()), o0);
                let i = fx.b.var(Ty::Int(0, INT_MAX), n);
                fx.b.open();
                {
                    let iv = fx.b.let_(Atom::V(i));
                    let z = fx.b.int_cmp("int.eq", iv.clone(), lit(0))?;
                    fx.b.open();
                    fx.b.brk(1);
                    let th = fx.b.close();
                    fx.b.push(Stmt::If(z, th, vec![]));
                    let i1 = fx.b.sub_in(iv, lit(1), Ty::Int(0, INT_MAX))?;
                    fx.b.set(i, i1.clone());
                    let x = fx.b.list_at(recv.a.clone(), i1)?;
                    let o = fx.b.let_(Atom::V(out));
                    let o = fx.b.p("list.push", vec![ect.clone()], vec![o, x])?;
                    fx.b.set(out, o);
                }
                let body = fx.b.close();
                fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
                fx.b.drop(recv.a)?;
                Ok(Val { a: fx.b.let_(Atom::V(out)), t: recv.t.clone() })
            }
            (RType::List(et), "get") => {
                let i = av.remove(0);
                let RType::Int(_) = &i.t else { return no(format!("get with {}", i.t.show())) };
                let et = (**et).clone();
                let ect = self.ct(&et)?;
                let t = self.ct(&rt)?;
                let r = self.result_var(fx, &t)?;
                let neg = fx.b.int_cmp("int.lt", i.a.clone(), lit(0))?;
                let n = fx.b.p("list.len", vec![ect.clone()], vec![recv.a.clone()])?;
                let inside = fx.b.int_cmp("int.lt", i.a.clone(), n)?;
                let nn = fx.b.not(neg)?;
                let ok = fx.b.p("bool.and", vec![], vec![nn, inside])?;
                fx.b.open();
                let ii = fx.b.cast_in(i.a.clone(), Ty::Int(0, INT_MAX))?;
                let x = fx.b.list_at(recv.a.clone(), ii)?;
                let RType::Opt(inner) = &rt else { return no("get of unknown type") };
                let x = self.conv(fx, x, &et, inner, false)?;
                let sm = fx.b.p("sum.mk", vec![t.clone()], vec![lit(1), x])?;
                self.store_var(fx, r, sm)?;
                let th = fx.b.close();
                fx.b.push(Stmt::If(ok, th, vec![]));
                fx.b.drop(recv.a)?;
                Ok(Val { a: fx.b.let_(Atom::V(r)), t: rt })
            }
            (RType::List(et), "sum") if **et == RType::F64 && rt == RType::F64 => {
                // the list has an item (typed): the integer 0 plus the first
                // item is the float 0.0 + x0, then each item is added in
                // order, each sum finite (interp `arith`)
                let acc = fx.b.var(Ty::F64, Atom::K(Ty::F64, Lit::F64(0f64.to_bits())));
                let n = fx.b.p("list.len", vec![Ty::F64], vec![recv.a.clone()])?;
                let ra = recv.a.clone();
                fx.b.for_n(n, |b, i| {
                    let x = b.list_at(ra.clone(), i)?;
                    let a = b.let_(Atom::V(acc));
                    let bad = b.p("f64.bad.add", vec![], vec![a.clone(), x.clone()])?;
                    b.open();
                    let h = b.trap_value(Atom::bytes(b"non-finite float"))?;
                    b.halt(h);
                    let th = b.close();
                    b.push(Stmt::If(bad, th, vec![]));
                    let s2 = b.p("f64.add", vec![], vec![a, x])?;
                    b.set(acc, s2);
                    Ok(())
                })?;
                fx.b.drop(recv.a)?;
                Ok(Val { a: fx.b.let_(Atom::V(acc)), t: rt })
            }
            (RType::List(et), "sum") => {
                let et = (**et).clone();
                let RType::Int(ei) = &et else {
                    if et == RType::Never {
                        fx.b.drop(recv.a)?;
                        return Ok(Val { a: lit(0), t: rt });
                    }
                    return no(format!("sum of {}", et.show()));
                };
                let tags = typed::Tags::lit().join_op(ei.tags);
                let res = IntT::of_tags(tags);
                let (r, name) = check_of(&res)?;
                let ect = self.ct(&et)?;
                let acc = fx.b.var(hull(&r, &Ty::Int(0, 0)), lit(0));
                let n = fx.b.p("list.len", vec![ect.clone()], vec![recv.a.clone()])?;
                let i = fx.b.var(Ty::u64(), lit(0));
                fx.b.open();
                {
                    let iv = fx.b.let_(Atom::V(i));
                    let more = fx.b.int_cmp("int.lt", iv.clone(), n)?;
                    fx.b.open();
                    let th = fx.b.close();
                    fx.b.open();
                    fx.b.brk(1);
                    let el = fx.b.close();
                    fx.b.push(Stmt::If(more, th, el));
                    let x = fx.b.list_at(recv.a.clone(), iv.clone())?;
                    let a = fx.b.let_(Atom::V(acc));
                    let s2 = self.int_op(fx, "+", a, x, &r, &name, &res)?;
                    fx.b.set(acc, s2);
                    let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
                    fx.b.set(i, i1);
                }
                let body = fx.b.close();
                fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
                fx.b.drop(recv.a)?;
                Ok(Val { a: fx.b.let_(Atom::V(acc)), t: rt })
            }
            (RType::Text, "len" | "len_bytes") => Ok(Val { a: fx.b.p("bytes.len", vec![], vec![recv.a])?, t: rt }),
            (RType::Text, "isempty") => {
                let n = fx.b.p("bytes.len", vec![], vec![recv.a])?;
                Ok(Val { a: fx.b.int_cmp("int.eq", n, lit(0))?, t: rt })
            }
            (RType::Set(_), "tolist") => Ok(Val { a: recv.a, t: rt }),
            (RType::Gen(key, item), "collect") => {
                // interp: one run without a cap; a trap in it is raised
                let (key, item) = (key.clone(), (**item).clone());
                let (r, buf) = self.gen_run(fx, recv.a.clone(), &key, &item, lit(u64::MAX as i128))?;
                fx.b.drop(recv.a)?;
                let rtt = fx.b.ty(&r);
                self.switch(fx, &r, &rtt, 2, |_, fx, k| {
                    let x = fx.b.p("sum.get", vec![rtt.clone()], vec![r.clone(), lit(k as i128)])?;
                    if k == 1 {
                        fx.b.halt(x);
                    }
                    Ok(())
                })?;
                let RType::List(re) = &rt else { return no("collect of unknown type") };
                let a = self.conv(fx, buf, &RType::List(Box::new(item)), &RType::List(re.clone()), false)?;
                Ok(Val { a, t: rt })
            }
            (RType::List(_), "len" | "isempty") | (RType::Set(_), "len") => {
                let t = fx.b.ty(&recv.a);
                let Ty::List(e) = &t else { unreachable!() };
                let n = fx.b.p("list.len", vec![(**e).clone()], vec![recv.a.clone()])?;
                fx.b.drop(recv.a)?;
                if m == "isempty" {
                    return Ok(Val { a: fx.b.int_cmp("int.eq", n, lit(0))?, t: rt });
                }
                Ok(Val { a: n, t: rt })
            }
            (RType::List(_), "with") => {
                let x = av.remove(0);
                let RType::List(et) = &rt else { return no("with of unknown type") };
                let l = self.conv(fx, recv.a, &recv.t, &rt, false)?;
                let ect = self.ct(et)?;
                let xa = self.conv(fx, x.a, &x.t, et, false)?;
                let l2 = fx.b.p("list.push", vec![ect], vec![l, xa])?;
                Ok(Val { a: l2, t: rt })
            }
            (RType::Opt(_), "isnone" | "issome") => {
                let t = fx.b.ty(&recv.a);
                let tag = fx.b.p("sum.tag", vec![t], vec![recv.a.clone()])?;
                fx.b.drop(recv.a)?;
                Ok(Val { a: fx.b.int_cmp("int.eq", tag, lit(if m == "issome" { 1 } else { 0 }))?, t: rt })
            }
            (RType::Opt(x), "or") => {
                let d = av.remove(0);
                let t = self.ct(&rt)?;
                let r = self.result_var(fx, &t)?;
                let rt_t = fx.b.ty(&recv.a);
                let x = (**x).clone();
                let (da, dt) = (d.a, d.t);
                let mut d_left = Some(da);
                let rt2 = rt.clone();
                self.switch(fx, &recv.a, &rt_t, 2, |lw, fx, k| {
                    let p = fx.b.p("sum.get", vec![rt_t.clone()], vec![recv.a.clone(), lit(k as i128)])?;
                    let dd = d_left.clone().unwrap();
                    if k == 0 {
                        fx.b.drop(p)?;
                        let a = lw.conv(fx, dd, &dt, &rt2, false)?;
                        lw.store_var(fx, r, a)
                    } else {
                        fx.b.drop(dd)?;
                        let a = lw.conv(fx, p, &x, &rt2, false)?;
                        lw.store_var(fx, r, a)
                    }
                })?;
                d_left.take();
                Ok(Val { a: fx.b.let_(Atom::V(r)), t: rt })
            }
            (t, _) => no(format!("the method {m} of {}", t.show())),
        }
    }

    /// A new list of `f(x)` for each item x of the list `l` (consumed),
    /// whose items have type `et`; `f` consumes its argument.
    fn map_list(&mut self, fx: &mut Fx, l: Atom, et: &RType, out_t: Ty, mut f: impl FnMut(&mut Self, &mut Fx, Atom) -> R<Atom>) -> R<Atom> {
        let ect = self.ct(et)?;
        let n = fx.b.p("list.len", vec![ect], vec![l.clone()])?;
        let o0 = fx.b.p("list.new", vec![out_t.clone()], vec![])?;
        let out = fx.b.var(Ty::list(out_t.clone()), o0);
        let i = fx.b.var(Ty::u64(), lit(0));
        fx.b.open();
        {
            let iv = fx.b.let_(Atom::V(i));
            let more = fx.b.int_cmp("int.lt", iv.clone(), n)?;
            fx.b.open();
            let th = fx.b.close();
            fx.b.open();
            fx.b.brk(1);
            let el = fx.b.close();
            fx.b.push(Stmt::If(more, th, el));
            let x = fx.b.list_at(l.clone(), iv.clone())?;
            let y = f(self, fx, x)?;
            let o = fx.b.let_(Atom::V(out));
            let o = fx.b.p("list.push", vec![out_t.clone()], vec![o, y])?;
            fx.b.set(out, o);
            let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
            fx.b.set(i, i1);
        }
        let body = fx.b.close();
        fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
        fx.b.drop(l)?;
        Ok(fx.b.let_(Atom::V(out)))
    }

    /// `xs.contains(x)` and `x in xs` for a list (interp: `==` with each item):
    /// `contains:(List[e], x)(xs, x) -> bool`, consuming both.
    fn list_contains_fn(&mut self, et: &RType, xt: &RType) -> R<usize> {
        let lt = Ty::list(self.ct(et)?);
        let xct = self.ct(xt)?;
        let key = RType::Tuple(vec![RType::List(Box::new(et.clone())), xt.clone()]);
        let (et2, xt2) = (et.clone(), xt.clone());
        self.generated("contains", &key, vec![lt, xct], Ty::Bool, move |lw, fx| {
            let (l, x) = (fx.b.arg(0), fx.b.arg(1));
            let ect = lw.ct(&et2)?;
            let n = fx.b.p("list.len", vec![ect], vec![l.clone()])?;
            let i = fx.b.var(Ty::u64(), lit(0));
            let r = fx.b.var(Ty::Bool, Atom::bool(false));
            fx.b.open();
            {
                let iv = fx.b.let_(Atom::V(i));
                let more = fx.b.int_cmp("int.lt", iv.clone(), n)?;
                fx.b.open();
                let th = fx.b.close();
                fx.b.open();
                fx.b.brk(1);
                let el = fx.b.close();
                fx.b.push(Stmt::If(more, th, el));
                let y = fx.b.list_at(l.clone(), iv.clone())?;
                let xc = fx.b.copy(x.clone())?;
                let e = lw.eq(fx, Val { a: y, t: et2.clone() }, Val { a: xc, t: xt2.clone() })?;
                fx.b.open();
                fx.b.set(r, Atom::bool(true));
                fx.b.brk(1);
                let th = fx.b.close();
                fx.b.push(Stmt::If(e, th, vec![]));
                let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
                fx.b.set(i, i1);
            }
            let body = fx.b.close();
            fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
            fx.b.drop(l)?;
            fx.b.drop(x)?;
            let rv = fx.b.let_(Atom::V(r));
            fx.b.ret(rv);
            Ok(())
        })
    }

    /// `xs.map(f)`, `xs.filter(f)` and `xs.fold(init, f)` with a lambda f,
    /// defunctionalised (ONE-4): the lambda is not a value; its body runs
    /// once per item in a scope of its own, its parameters bound to the
    /// item (and the accumulator), each fitted to its declared type (interp
    /// `apply_closure`), one step of fuel per item. The lambda sees the
    /// variables as they are when it is made, which is where it runs, so a
    /// body that writes a variable (a `&mut` argument, push, insert, …),
    /// which in the reference writes the closure's own copy, is not covered.
    #[allow(clippy::too_many_arguments)]
    fn hof(&mut self, fx: &mut Fx, obj: &Expr, m: &str, args: &[Arg], ps: &[Param], body: &Expr, rt: RType, sized: bool) -> R<Val> {
        if self.lambda_writes(body) {
            return no(format!("a lambda given to {m} that writes a variable"));
        }
        let recv = self.expr(fx, obj)?;
        let RType::List(et) = recv.t.clone() else { return no(format!("{m} of a non-list")) };
        let et = (*et).clone();
        let ect = self.ct(&et)?;
        let init = if m == "fold" { Some(self.expr(fx, &args[0].e)?) } else { None };
        fx.b.tick(1); // the lambda
        let first = init.as_ref().map(|x| (x.a.clone(), x.t.clone()));
        self.charge_method(fx, m, (recv.a.clone(), recv.t.clone()), first.as_ref().map(|(a, t)| (a, t)), sized)?;
        // the result: a list being built, or the accumulator
        let out_t = self.ct(&rt)?;
        let out = match (m, init) {
            ("fold", Some(i)) => {
                let a = self.conv(fx, i.a, &i.t, &rt, false)?;
                fx.b.var(out_t.clone(), a)
            }
            _ => {
                let Ty::List(oe) = &out_t else { return no("map of unknown type") };
                let l = fx.b.p("list.new", vec![(**oe).clone()], vec![])?;
                fx.b.var(out_t.clone(), l)
            }
        };
        fx.own(Atom::V(out));
        let n = fx.b.p("list.len", vec![ect.clone()], vec![recv.a.clone()])?;
        let i = fx.b.var(Ty::u64(), lit(0));
        let l = fx.open_nest();
        let iv = fx.b.let_(Atom::V(i));
        let more = fx.b.int_cmp("int.lt", iv.clone(), n)?;
        fx.b.open();
        let th = fx.b.close();
        fx.b.open();
        fx.break_to(l)?;
        let el = fx.b.close();
        fx.b.push(Stmt::If(more, th, el));
        fx.b.tick(1);
        self.charge_k(fx, crate::conc::C_CALL)?; // the application
        let x = fx.b.list_at(recv.a.clone(), iv.clone())?;
        fx.push_scope();
        let mut ins = vec![];
        if m == "fold" {
            let a = self.read_var(fx, out)?;
            ins.push(Val { a, t: rt.clone() });
        }
        let item = Val { a: x, t: et.clone() };
        ins.push(item.clone());
        self.bind_lambda_params(fx, ps, ins)?;
        let r = self.expr(fx, body)?;
        match m {
            "map" => {
                let RType::List(oe) = &rt else { unreachable!() };
                let a = self.conv(fx, r.a, &r.t, oe, false)?;
                let o = fx.b.let_(Atom::V(out));
                let Ty::List(oc) = &out_t else { unreachable!() };
                let o = fx.b.p("list.push", vec![(**oc).clone()], vec![o, a])?;
                fx.b.set(out, o);
            }
            "filter" => {
                // interp: the item is kept when the result is `true`
                if r.t != RType::Bool {
                    return no("a filter whose lambda does not give a bool");
                }
                fx.b.open();
                let xc = fx.b.list_at(recv.a.clone(), iv.clone())?;
                let RType::List(oe) = &rt else { unreachable!() };
                let a = self.conv(fx, xc, &et, oe, false)?;
                let o = fx.b.let_(Atom::V(out));
                let Ty::List(oc) = &out_t else { unreachable!() };
                let o = fx.b.p("list.push", vec![(**oc).clone()], vec![o, a])?;
                fx.b.set(out, o);
                let th = fx.b.close();
                fx.b.push(Stmt::If(r.a, th, vec![]));
            }
            _ => {
                let a = self.conv(fx, r.a, &r.t, &rt, false)?;
                self.store_var(fx, out, a)?;
            }
        }
        fx.pop_scope()?;
        let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
        fx.b.set(i, i1);
        let lb = fx.close_nest();
        fx.b.push(Stmt::Loop(lb));
        fx.b.drop(recv.a)?;
        fx.disown(&Atom::V(out));
        Ok(Val { a: fx.b.let_(Atom::V(out)), t: rt })
    }

    /// The parameter types of an instantiation, lowered or not.
    fn inst_params(&self, key: &str) -> R<Vec<RType>> {
        if let Some(i) = self.insts.get(key) {
            return Ok(i.params.clone());
        }
        match self.inst_ids.get(key) {
            Some(id) => Ok(self.inst_sigs[id].0.clone()),
            None => no(format!("the instantiation {key}")),
        }
    }

    /// The global holding the buffer stack of the runs of iterators whose
    /// items have core type t.
    fn gen_stack(&mut self, t: &Ty) -> u32 {
        if let Some(g) = self.gen_stacks.get(t) {
            return *g;
        }
        let k = self.gen_stacks.len();
        let g = self.f.global(&format!("iterator buffers {k}"), Ty::list(Ty::list(t.clone())), None);
        self.gen_stacks.insert(t.clone(), g);
        g
    }

    /// A call of an iterator function (interp `call_fn`): one step, the
    /// arguments evaluated and fitted in order, and no run: the value is
    /// the frame the runs start from.
    fn iter_call(&mut self, fx: &mut Fx, e: &Expr, n: &str, args: &[Arg], rt: RType) -> R<Val> {
        let fd = self.t.fns[n];
        let RType::Gen(key, _) = &rt else { return no(format!("a call of the iterator function {n}")) };
        let params = self.inst_params(key)?;
        let named = Self::named_params(fd);
        if named.len() != args.len() || self.ty.inst_of.get(&addr(e)) != Some(key) {
            return no("a call whose arguments do not match its parameters one to one");
        }
        fx.b.tick(1);
        self.charge_k(fx, crate::conc::C_CALL)?;
        let mut parts = vec![];
        for (k, (ty, arg)) in named.iter().zip(args).enumerate() {
            let v = self.expr(fx, &arg.e)?;
            let v = self.fit(fx, v.a, &v.t, ty)?;
            parts.push(self.conv(fx, v.a, &v.t, &params[k], false)?);
        }
        let t = self.ct(&rt)?;
        Ok(Val { a: fx.b.p("tup.mk", vec![t], parts)?, t: rt })
    }

    /// One run of an iterator from its frame (borrowed), stopping after cap
    /// items: an empty buffer is pushed, the run is caught, and the buffer
    /// is popped. Gives the run's result, `sum(finished, Halt)`, and the
    /// items, both owned by the caller.
    fn gen_run(&mut self, fx: &mut Fx, frame: Atom, key: &str, item: &RType, cap: Atom) -> R<(Atom, Atom)> {
        let fname = key.strip_prefix("iter ").and_then(|k| k.split('(').next()).unwrap_or("").to_string();
        let Some(fd) = self.t.fns.get(&fname).copied() else { return no(format!("the iterator {key}")) };
        let id = self.gen_fn(key, fd)?;
        let it = self.ct(item)?;
        let g = self.gen_stack(&it);
        let st = Ty::list(Ty::list(it.clone()));
        let s0 = fx.b.let_global(g, st.clone());
        let e0 = fx.b.p("list.new", vec![it.clone()], vec![])?;
        let s1 = fx.b.p("list.push", vec![Ty::list(it.clone())], vec![s0, e0])?;
        fx.b.setg(g, s1);
        let ft = fx.b.ty(&frame);
        let fc = fx.b.copy(frame)?;
        let mut args = fx.b.prim("tup.open", vec![ft], vec![fc])?;
        args.push(cap);
        let r = fx.b.catch(sym(id), Ty::Bool, args);
        let s2 = fx.b.let_global(g, st.clone());
        let n = fx.b.p("list.len", vec![Ty::list(it.clone())], vec![s2.clone()])?;
        let some = fx.b.int_cmp("int.lt", lit(0), n)?;
        fx.b.guard_halt_unless(some, "internal: iterator buffers")?;
        let ps = fx.b.prim("list.pop", vec![Ty::list(it.clone())], vec![s2])?;
        fx.b.setg(g, ps[0].clone());
        Ok((r, ps[1].clone()))
    }

    /// `for p in f(…)` over an iterator (interp `exec` of `For`): runs of
    /// the iterator that stop after 64 items, then twice as many each time,
    /// each from the frame; after each run the items not yet taken run the
    /// body in order, then a trap of the run is raised, and a finished run
    /// ends the loop. `break` leaves the whole loop.
    fn for_gen(&mut self, fx: &mut Fx, p: &Pat, frame: Atom, key: &str, item: &RType, body: &[S]) -> R<()> {
        let ft = fx.b.ty(&frame);
        let fv = fx.b.var(ft, frame);
        fx.own(Atom::V(fv));
        let done = fx.b.var(Ty::u64(), lit(0));
        let cap = fx.b.var(Ty::u64(), lit(64));
        let l = fx.open_nest();
        fx.push_scope();
        let capv = fx.b.let_(Atom::V(cap));
        let (r, buf) = self.gen_run(fx, Atom::V(fv), key, item, capv)?;
        fx.own(r.clone());
        fx.own(buf.clone());
        let it = self.ct(item)?;
        let n = fx.b.p("list.len", vec![it.clone()], vec![buf.clone()])?;
        // the items after those already taken
        let li = fx.open_nest();
        let dv = fx.b.let_(Atom::V(done));
        let more = fx.b.int_cmp("int.lt", dv.clone(), n)?;
        fx.b.open();
        let th = fx.b.close();
        fx.b.open();
        fx.break_to(li)?;
        let el = fx.b.close();
        fx.b.push(Stmt::If(more, th, el));
        let d1 = fx.b.add_in(dv.clone(), lit(1), Ty::u64())?;
        fx.b.set(done, d1);
        let x = fx.b.list_at(buf.clone(), dv)?;
        let c = fx.open_nest();
        fx.loops.push(LoopCx { brk: l, cont: c });
        fx.push_scope();
        self.for_bind(fx, p, Val { a: x, t: item.clone() })?;
        self.block(fx, body, false)?;
        fx.pop_scope()?;
        fx.loops.pop();
        let inner = fx.close_nest();
        fx.b.push(Stmt::Block(inner));
        let lib = fx.close_nest();
        fx.b.push(Stmt::Loop(lib));
        // then the run's trap, or the end when it finished
        fx.disown(&r);
        let fin = fx.b.var(Ty::Bool, Atom::bool(false));
        let rt = fx.b.ty(&r);
        self.switch(fx, &r, &rt, 2, |_, fx, k| {
            let x = fx.b.p("sum.get", vec![rt.clone()], vec![r.clone(), lit(k as i128)])?;
            if k == 1 {
                fx.b.halt(x);
            } else {
                fx.b.set(fin, x);
            }
            Ok(())
        })?;
        let f = fx.b.let_(Atom::V(fin));
        fx.b.open();
        fx.break_to(l)?;
        let th = fx.b.close();
        fx.b.push(Stmt::If(f, th, vec![]));
        let cv = fx.b.let_(Atom::V(cap));
        let c2 = fx.b.p("int.smul", vec![Ty::u64()], vec![cv, lit(2)])?;
        fx.b.set(cap, c2);
        fx.pop_scope()?;
        let lb = fx.close_nest();
        fx.b.push(Stmt::Loop(lb));
        Ok(())
    }

    /// `yield e` in the body of an iterator (interp `exec`): the value; a
    /// run that has given the loop limit of items traps; the item goes to
    /// the buffer on top of the stack; after cap items the run stops.
    fn yield_stmt(&mut self, fx: &mut Fx, e: &Expr) -> R<()> {
        let Some(gc) = fx.run.clone() else { return no("yield outside an iterator") };
        let v = self.expr(fx, e)?;
        let a = self.conv(fx, v.a, &v.t, &gc.item, false)?;
        let n = fx.b.let_(Atom::V(gc.count));
        let over = fx.b.int_cmp("int.le", lit(self.loop_limit), n.clone())?;
        self.guard(fx, over, |_, _| Ok(Atom::bytes(b"loop bound")))?;
        let Ty::List(bt) = gc.stack_t.clone() else { unreachable!() };
        let s0 = fx.b.let_global(gc.g, gc.stack_t.clone());
        let k = fx.b.p("list.len", vec![(*bt).clone()], vec![s0.clone()])?;
        let some = fx.b.int_cmp("int.lt", lit(0), k)?;
        fx.b.guard_halt_unless(some, "internal: iterator buffers")?;
        let ps = fx.b.prim("list.pop", vec![(*bt).clone()], vec![s0])?;
        let Ty::List(it) = (*bt).clone() else { unreachable!() };
        let top = fx.b.p("list.push", vec![(*it).clone()], vec![ps[1].clone(), a])?;
        let s1 = fx.b.p("list.push", vec![(*bt).clone()], vec![ps[0].clone(), top])?;
        fx.b.setg(gc.g, s1);
        let n1 = fx.b.add_in(n, lit(1), Ty::u64())?;
        fx.b.set(gc.count, n1.clone());
        let stop = fx.b.int_cmp("int.le", gc.cap.clone(), n1)?;
        fx.b.open();
        fx.drops_to(None)?;
        fx.b.ret(Atom::bool(false));
        let th = fx.b.close();
        fx.b.push(Stmt::If(stop, th, vec![]));
        Ok(())
    }

    /// The function of an iterator's instantiation: its frame and the cap,
    /// and whether the run finished (interp `gen_run`).
    fn gen_fn(&mut self, key: &str, fd: &FnDecl) -> R<usize> {
        if let Some(id) = self.inst_ids.get(key) {
            return Ok(*id);
        }
        let Some(inst) = self.insts.remove(key) else { return no(format!("the instantiation {key}")) };
        if !Self::mut_params(fd).is_empty() {
            return no("an iterator with a &mut parameter");
        }
        let mut ps = vec![];
        for p in &inst.params {
            ps.push(self.ct(p)?);
        }
        let it = self.ct(&inst.ret)?;
        let g = self.gen_stack(&it);
        let np = ps.len();
        ps.push(Ty::u64());
        let id = self.f.declare(&format!("gen {key}"), ps.clone(), Ty::Bool);
        self.inst_ids.insert(key.to_string(), id);
        self.inst_sigs.insert(id, (inst.params.clone(), inst.ret.clone()));
        let saved = std::mem::replace(&mut self.ty, inst.typed);
        let r = (|| -> R<()> {
            let mut fx = self.fx(ps, Ty::Bool, Ret::Gen, FailCx::Halt, true);
            let count = fx.b.var(Ty::u64(), lit(0));
            fx.run = Some(GenCx { g, stack_t: Ty::list(Ty::list(it.clone())), item: inst.ret.clone(), count, cap: fx.b.arg(np) });
            let (ids, _) = self.ty.params.get(&addr(fd)).cloned().unwrap_or_default();
            for k in 0..np {
                let Some(lid) = ids.get(k).copied() else { return no("a parameter") };
                let a = fx.b.arg(k);
                let rt = self.ty.locals[lid as usize].ty.clone();
                let a = self.conv(&mut fx, a, &inst.params[k], &rt, false)?;
                fx.bound.insert(lid, Bound::Val(a.clone(), rt));
                fx.own(a);
            }
            let Body::Block(b) = &fd.body else { return no("an iterator without a block body") };
            self.block(&mut fx, b, false)?;
            if fx.b.falls() {
                fx.drops_to(None)?;
                fx.b.ret(Atom::bool(true));
            }
            self.f.define(id, fx.b.finish());
            Ok(())
        })();
        self.ty = saved;
        r?;
        Ok(id)
    }

    /// Whether the body of a lambda writes a variable: a `&mut` argument, a
    /// method that writes its receiver (push, insert, …). In the reference
    /// it writes the closure's own copy of the variables, which the
    /// lowering does not keep.
    fn lambda_writes(&self, body: &Expr) -> bool {
        let mut writes = false;
        crate::check::walk_expr(body, &mut |x| {
            if let Expr::Borrow(true, ..) = x {
                writes = true;
            }
            if let Expr::Call(f, _, _, _) = x {
                if let Expr::Member(_, mm, _) = &**f
                    && matches!(mm.as_str(), "push" | "append" | "pop" | "clear" | "sort" | "insert" | "remove")
                {
                    writes = true;
                }
                if let Some(Callee::Method(h, mm)) = self.ty.node(x).and_then(|n| n.callee.clone())
                    && let Some(fd) = self.t.methods.get(&(h, mm))
                    && matches!(fd.sig.params.first(), Some(Param::SelfRef(true)))
                {
                    writes = true;
                }
            }
        });
        writes
    }

    /// The parameters of a lambda bound to the values ins, each fitted to
    /// its declared type, in a scope the caller opened. interp
    /// `apply_closure` fits every parameter and keeps the last failure, so
    /// the fits, which have no other effect, run from the last parameter
    /// to the first and the first that fails traps.
    fn bind_lambda_params(&mut self, fx: &mut Fx, ps: &[Param], ins: Vec<Val>) -> R<()> {
        let named: Vec<(usize, ast::Ty)> = ps.iter().enumerate().filter_map(|(k, p)| if let Param::Named { ty, .. } = p { Some((k, ty.clone())) } else { None }).collect();
        if named.len() != ins.len() {
            return no("a lambda called with another number of arguments");
        }
        let mut fitted = vec![];
        for ((_, ty), v) in named.iter().zip(ins).rev() {
            fitted.push(self.fit(fx, v.a, &v.t, ty)?);
        }
        fitted.reverse();
        for ((k, _), fv) in named.iter().zip(fitted) {
            let Some(id) = self.ty.lambda_binds.get(&addr(&ps[*k])).copied() else { return no("a lambda parameter") };
            let a = fx.b.let_(fv.a);
            fx.bound.insert(id, Bound::Val(a.clone(), fv.t));
            fx.own(a);
        }
        Ok(())
    }

    /// A call of a lambda a `let` made (interp `call_value`, then
    /// `apply_closure`): one step for the name, the arguments in order,
    /// then the body in a scope of its own, its parameters bound to the
    /// fitted arguments. The body runs here, not where the lambda was
    /// made; the names it reads are the same locals, and an immutable
    /// local holds the same value here, so a body that reads a `var` (which
    /// may have changed since) or writes a variable is not covered.
    fn closure_call(&mut self, fx: &mut Fx, f: &Expr, args: &[Arg], l: usize, rt: RType) -> R<Val> {
        let Some(lam) = self.ty.lambdas.get(&l).cloned() else { return no("a lambda") };
        let Expr::Lambda(ps, body, _) = &*lam else { return no("a lambda") };
        if self.lambda_writes(body) {
            return no("a lambda that writes a variable");
        }
        let mut reads_var = false;
        crate::check::walk_expr(body, &mut |x| {
            if let Expr::Name(..) = x
                && let Some(typed::Binding::Local(id)) = self.ty.node(x).and_then(|n| n.bind.clone())
                && matches!(fx.bound.get(&id), Some(Bound::Var(..)))
            {
                reads_var = true;
            }
        });
        if reads_var {
            return no("a lambda that reads a var");
        }
        let fv = self.expr(fx, f)?;
        fx.b.drop(fv.a)?;
        let mut ins = vec![];
        for a in args {
            ins.push(self.expr(fx, &a.e)?);
        }
        self.charge_k(fx, crate::conc::C_CALL)?;
        fx.push_scope();
        self.bind_lambda_params(fx, ps, ins)?;
        let r = self.expr(fx, body)?;
        let a = self.conv(fx, r.a, &r.t, &rt, false)?;
        let out = fx.b.let_(a);
        fx.pop_scope()?;
        Ok(Val { a: out, t: rt })
    }

    /// The current type of the values of a `var` local (its storage type).
    fn var_rtype(&self, fx: &Fx, v: u32) -> Option<RType> {
        fx.bound.iter().find_map(|(id, b)| match b {
            Bound::Var(x, _) if *x == v => Some(self.ty.locals[*id as usize].ty.clone()),
            _ => None,
        })
    }

    /// N9: a list built by an operation has at most SIZE_LIMIT elements.
    fn size_check(&mut self, fx: &mut Fx, l: &Atom, et: &Ty) -> R<()> {
        let n = fx.b.p("list.len", vec![et.clone()], vec![l.clone()])?;
        let c = fx.b.int_cmp("int.lt", lit(crate::interp::SIZE_LIMIT as i128), n)?;
        self.guard(fx, c, |_, _| Ok(Atom::bytes(b"size limit")))
    }

    /// interp `index`: a list or text indexed by an integer, with the trap
    /// "index K out of range".
    fn index(&mut self, fx: &mut Fx, x: &Expr, idx: &[Expr], rt: RType) -> R<Val> {
        if let RType::Map(..) = self.node_ty(x) {
            let c = self.expr(fx, x)?;
            let kv = self.expr(fx, &idx[0])?;
            self.charge_k(fx, 1)?;
            let i = self.map_find(fx, &c, kv.clone())?;
            let missing = fx.b.int_cmp("int.lt", i.clone(), lit(0))?;
            let kv2 = kv.clone();
            self.guard(fx, missing, |lw, fx| {
                let s = lw.show(fx, kv2.a, &kv2.t)?;
                fx.b.cat(Atom::bytes(b"missing key "), s)
            })?;
            let _ = kv;
            let v = self.map_value_at(fx, c, i)?;
            return Ok(v);
        }
        // in place when the container is a variable and the index cannot change it
        let in_place = self.var_of(fx, x).filter(|_| simple(&idx[0]));
        let (c, owned) = match in_place {
            Some(v) => {
                fx.b.tick(1);
                (Atom::V(v), false)
            }
            None => (self.expr(fx, x)?.a, true),
        };
        let i = self.expr(fx, &idx[0])?;
        self.charge_k(fx, 1)?;
        let RType::Int(_) = &i.t else { return no("an index that is not an integer") };
        let ct = fx.b.ty(&c);
        let neg = fx.b.int_cmp("int.lt", i.a.clone(), lit(0))?;
        let ia = i.a.clone();
        self.guard(fx, neg, |lw, fx| {
            let d = lw.dec(fx, ia)?;
            fx.b.cats(vec![Atom::bytes(b"index "), d, Atom::bytes(b" out of range")])
        })?;
        let n = match &ct {
            Ty::List(e) => fx.b.p("list.len", vec![(**e).clone()], vec![c.clone()])?,
            Ty::Bytes => fx.b.p("bytes.len", vec![], vec![c.clone()])?,
            _ => return no("an index into a value that is not a list or a text"),
        };
        let ok = fx.b.int_cmp("int.lt", i.a.clone(), n)?;
        fx.b.open();
        let th = fx.b.close();
        fx.b.open();
        let d = self.dec(fx, i.a.clone())?;
        let msg = fx.b.cats(vec![Atom::bytes(b"index "), d, Atom::bytes(b" out of range")])?;
        self.trap(fx, msg)?;
        let el = fx.b.close();
        fx.b.push(Stmt::If(ok, th, el));
        let r = match &ct {
            Ty::List(e) => fx.b.p("list.get", vec![(**e).clone()], vec![c.clone(), i.a])?,
            _ => fx.b.p("bytes.get", vec![], vec![c.clone(), i.a])?,
        };
        if owned {
            fx.b.drop(c)?;
        }
        Ok(Val { a: r, t: rt })
    }

    /// The index of key k (a value, read) in the map m (read), or -1.
    fn map_find(&mut self, fx: &mut Fx, m: &Val, k: Val) -> R<Atom> {
        let RType::Map(kt, _) = &m.t else { return Err("internal: map_find of a non-map".into()) };
        let kt = (**kt).clone();
        let find = self.map_find_fn(&m.t)?;
        let mc = fx.b.copy(m.a.clone())?;
        let kc = fx.b.copy(k.a)?;
        let want = self.ct(&kt)?;
        if !fx.b.ty(&kc).within(&want) && fx.b.ty(&kc) != want {
            return no(format!("a map key of type {} for keys of type {}", k.t.show(), kt.show()));
        }
        let kc = self.conv(fx, kc, &k.t, &kt, false)?;
        Ok(self.call_rt(fx, find, vec![mc, kc]))
    }

    /// The value at index i (known to hold a key) of the map m (consumed).
    fn map_value_at(&mut self, fx: &mut Fx, m: Val, i: Atom) -> R<Val> {
        let RType::Map(_, vt) = &m.t else { return Err("internal: map_value_at of a non-map".into()) };
        let vt = (**vt).clone();
        let mt = fx.b.ty(&m.a);
        let Ty::List(pt) = &mt else { unreachable!() };
        let pt = (**pt).clone();
        let i = fx.b.cast_in(i, Ty::Int(0, INT_MAX))?;
        let e = fx.b.list_at(m.a.clone(), i)?;
        let ps = fx.b.prim("tup.open", vec![pt], vec![e])?;
        fx.b.drop(ps[0].clone())?;
        fx.b.drop(m.a)?;
        Ok(Val { a: ps[1].clone(), t: vt })
    }

    /// `[body for p in it if cond]` (interp `E::ListComp`): the iterable
    /// once; for each item one step of fuel, the pattern (an item it does
    /// not match is left out), the condition, and the body in a scope of
    /// their own; then N9's limit on the result.
    fn list_comp(&mut self, fx: &mut Fx, body: &Expr, p: &Pat, it: &Expr, cond: Option<&Expr>, rt: RType) -> R<Val> {
        let RType::List(et) = rt.clone() else { return no("a list comprehension of unknown type") };
        let et = (*et).clone();
        let ect = self.ct(&et)?;
        // the items: a range (with the limit of interp `iter_items`), a list or a map
        let (items, item_rt, range) = if let Expr::Bin(op, a, bx, _) = it
            && (op == ".." || op == "..=")
        {
            fx.b.tick(1);
            let x = self.expr(fx, a)?;
            let y = self.expr(fx, bx)?;
            self.charge_k(fx, 1)?; // the range
            let (RType::Int(_), RType::Int(_)) = (&x.t, &y.t) else { return no("a range of non-integers") };
            let item_t = match self.node_ty(it) {
                RType::Range(i) => i,
                _ => return no("a range of unknown type"),
            };
            let hi = if op == "..=" {
                let yt = fx.b.ty(&y.a);
                let (yl, yh) = yt.bounds().unwrap();
                if yh >= INT_MAX {
                    return no("an inclusive range up to 2^64 - 1");
                }
                fx.b.add_in(y.a, lit(1), Ty::Int(yl + 1, yh + 1))?
            } else {
                y.a
            };
            let le = fx.b.int_cmp("int.le", hi.clone(), x.a.clone())?;
            fx.b.open();
            let th = fx.b.close();
            fx.b.open();
            let lim = fx.b.p("int.sadd", vec![wide()], vec![x.a.clone(), lit(self.loop_limit * 10)])?;
            let over = fx.b.int_cmp("int.lt", lim, hi.clone())?;
            self.guard(fx, over, |_, _| Ok(Atom::bytes(b"loop bound")))?;
            let el = fx.b.close();
            fx.b.push(Stmt::If(le, th, el));
            let it_t = hull(&fx.b.ty(&x.a), &fx.b.ty(&hi));
            let (lo_b, hi_b) = it_t.bounds().unwrap();
            (None, RType::Int(IntT { tags: item_t.tags, lo: lo_b, hi: hi_b }), Some((x.a, hi, it_t)))
        } else {
            let itv = self.expr(fx, it)?;
            let item = match &itv.t {
                RType::List(e) | RType::Set(e) => (**e).clone(),
                RType::Map(k, v) => RType::Tuple(vec![(**k).clone(), (**v).clone()]),
                o => return no(format!("a list comprehension over {}", o.show())),
            };
            let lt = fx.b.ty(&itv.a);
            let Ty::List(e) = &lt else { unreachable!() };
            let lv = fx.b.var(lt.clone(), itv.a);
            fx.own(Atom::V(lv));
            (Some((lv, (**e).clone())), item, None)
        };
        let o0 = fx.b.p("list.new", vec![ect.clone()], vec![])?;
        let out = fx.b.var(Ty::list(ect.clone()), o0);
        let i = match &range {
            Some((x, _, it_t)) => fx.b.var(it_t.clone(), x.clone()),
            None => fx.b.var(Ty::u64(), lit(0)),
        };
        let l = fx.open_nest();
        let iv = fx.b.let_(Atom::V(i));
        let n = match (&range, &items) {
            (Some((_, hi, _)), _) => hi.clone(),
            (None, Some((lv, e))) => fx.b.p("list.len", vec![e.clone()], vec![Atom::V(*lv)])?,
            _ => unreachable!(),
        };
        let more = fx.b.int_cmp("int.lt", iv.clone(), n)?;
        fx.b.open();
        let th = fx.b.close();
        fx.b.open();
        fx.break_to(l)?;
        let el = fx.b.close();
        fx.b.push(Stmt::If(more, th, el));
        let item = match &items {
            Some((lv, _)) => fx.b.list_at(Atom::V(*lv), iv.clone())?,
            None => iv.clone(),
        };
        fx.b.tick(1);
        // one item: a block left when the pattern or the condition fails
        let blk = fx.open_nest();
        fx.push_scope();
        let x = fx.b.let_(item);
        fx.own(x.clone());
        self.pat(fx, p, &x, &item_rt, blk)?;
        if let Some(c) = cond {
            let cv = self.expr(fx, c)?;
            self.need(fx, cv.a, blk)?;
        }
        let v = self.expr(fx, body)?;
        let va = self.conv(fx, v.a, &v.t, &et, false)?;
        let o = fx.b.let_(Atom::V(out));
        let o = fx.b.p("list.push", vec![ect.clone()], vec![o, va])?;
        fx.b.set(out, o);
        fx.pop_scope()?;
        let inner = fx.close_nest();
        fx.b.push(Stmt::Block(inner));
        let it_t = fx.b.local_ty(i);
        let i1 = fx.b.add_in(iv, lit(1), it_t)?;
        fx.b.set(i, i1);
        let lb = fx.close_nest();
        fx.b.push(Stmt::Loop(lb));
        let r = fx.b.let_(Atom::V(out));
        if self.costs {
            let n = fx.b.p("list.len", vec![ect.clone()], vec![r.clone()])?;
            self.charge_list(fx, n)?;
        }
        self.size_check(fx, &r, &ect)?;
        Ok(Val { a: r, t: rt })
    }

    /// interp `contains(c, x)`, the operator `in`: c and x consumed.
    fn contains(&mut self, fx: &mut Fx, c: Val, x: Val) -> R<Atom> {
        match &c.t {
            RType::List(et) => {
                let et = (**et).clone();
                let g = self.list_contains_fn(&et, &x.t)?;
                Ok(self.call_rt(fx, g, vec![c.a, x.a]))
            }
            RType::Text => {
                if x.t != RType::Text {
                    fx.b.drop(c.a)?;
                    fx.b.drop(x.a)?;
                    return Ok(Atom::bool(false));
                }
                let g = rtt::contains(&mut self.f)?;
                Ok(self.call_rt(fx, g, vec![c.a, x.a]))
            }
            RType::Tuple(ts) => {
                let ts = ts.clone();
                let ct = fx.b.ty(&c.a);
                let mut acc = Atom::bool(false);
                for (k, t) in ts.iter().enumerate() {
                    let y = fx.b.p("tup.get", vec![ct.clone()], vec![c.a.clone(), lit(k as i128)])?;
                    let xc = fx.b.copy(x.a.clone())?;
                    let e = self.eq(fx, Val { a: y, t: t.clone() }, Val { a: xc, t: x.t.clone() })?;
                    acc = fx.b.p("bool.or", vec![], vec![acc, e])?;
                }
                fx.b.drop(c.a)?;
                fx.b.drop(x.a)?;
                Ok(acc)
            }
            RType::Map(..) => {
                let i = self.map_find(fx, &c, x.clone())?;
                fx.b.drop(c.a)?;
                fx.b.drop(x.a)?;
                fx.b.int_cmp("int.le", lit(0), i)
            }
            o => no(format!("`in` on {}", o.show())),
        }
    }

    /// interp `E::Slice`: `x[a:b]` of a list or a text; a negative bound
    /// counts from the end; the traps are those of the reference.
    fn slice(&mut self, fx: &mut Fx, x: &Expr, a: Option<&Expr>, b: Option<&Expr>, rt: RType) -> R<Val> {
        let c = self.expr(fx, x)?;
        self.charge_k(fx, 1)?;
        let av = match a {
            Some(e) => Some(self.expr(fx, e)?),
            None => None,
        };
        let bv = match b {
            Some(e) => Some(self.expr(fx, e)?),
            None => None,
        };
        let ct = fx.b.ty(&c.a);
        let n = match &ct {
            Ty::List(e) => fx.b.p("list.len", vec![(**e).clone()], vec![c.a.clone()])?,
            Ty::Bytes => fx.b.p("bytes.len", vec![], vec![c.a.clone()])?,
            _ => return no(format!("a slice of {}", c.t.show())),
        };
        let lo = self.slice_bound(fx, av, &n, lit(0))?;
        let hi = self.slice_bound(fx, bv, &n, n.clone())?;
        let rev = fx.b.int_cmp("int.lt", hi.clone(), lo.clone())?;
        self.guard(fx, rev, |_, _| Ok(Atom::bytes(b"slice bounds reversed")))?;
        match ct {
            Ty::Bytes => {
                let g = rtt::boundary(&mut self.f)?;
                let b1 = self.call_rt(fx, g, vec![c.a.clone(), lo.clone()]);
                let b2 = self.call_rt(fx, g, vec![c.a.clone(), hi.clone()]);
                let both = fx.b.p("bool.and", vec![], vec![b1, b2])?;
                let bad = fx.b.not(both)?;
                self.guard(fx, bad, |_, _| Ok(Atom::bytes(b"not a char boundary")))?;
                let ok = fx.b.int_cmp("int.le", lo.clone(), hi.clone())?;
                fx.b.guard_halt_unless(ok, "internal: slice")?;
                let ok = fx.b.int_cmp("int.le", hi.clone(), n)?;
                fx.b.guard_halt_unless(ok, "internal: slice")?;
                Ok(Val { a: fx.b.p("bytes.slice", vec![], vec![c.a, lo, hi])?, t: rt })
            }
            Ty::List(e) => {
                let e = (*e).clone();
                let o0 = fx.b.p("list.new", vec![e.clone()], vec![])?;
                let out = fx.b.var(Ty::list(e.clone()), o0);
                let i = fx.b.var(Ty::Int(0, INT_MAX), lo);
                fx.b.open();
                {
                    let iv = fx.b.let_(Atom::V(i));
                    let more = fx.b.int_cmp("int.lt", iv.clone(), hi)?;
                    fx.b.open();
                    let th = fx.b.close();
                    fx.b.open();
                    fx.b.brk(1);
                    let el = fx.b.close();
                    fx.b.push(Stmt::If(more, th, el));
                    let y = fx.b.list_at(c.a.clone(), iv.clone())?;
                    let o = fx.b.let_(Atom::V(out));
                    let o = fx.b.p("list.push", vec![e.clone()], vec![o, y])?;
                    fx.b.set(out, o);
                    let i1 = fx.b.add_in(iv, lit(1), Ty::Int(0, INT_MAX))?;
                    fx.b.set(i, i1);
                }
                let body = fx.b.close();
                fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
                fx.b.drop(c.a)?;
                let r = fx.b.let_(Atom::V(out));
                let a = self.conv(fx, r, &c.t, &rt, false)?;
                Ok(Val { a, t: rt })
            }
            _ => unreachable!(),
        }
    }

    /// A slice bound normalised as interp `norm` does: absent, the default;
    /// negative, counted from the end; then within 0..=n or a trap.
    fn slice_bound(&mut self, fx: &mut Fx, v: Option<Val>, n: &Atom, default: Atom) -> R<Atom> {
        let Some(v) = v else { return Ok(default) };
        let RType::Int(_) = &v.t else { return no(format!("a slice bound of type {}", v.t.show())) };
        let k = fx.b.var(wide(), v.a.clone());
        let neg = fx.b.int_cmp("int.lt", v.a.clone(), lit(0))?;
        fx.b.open();
        let s = fx.b.add_in(n.clone(), v.a, wide())?;
        fx.b.set(k, s);
        let th = fx.b.close();
        fx.b.push(Stmt::If(neg, th, vec![]));
        let kv = fx.b.let_(Atom::V(k));
        let below = fx.b.int_cmp("int.lt", kv.clone(), lit(0))?;
        let above = fx.b.int_cmp("int.lt", n.clone(), kv.clone())?;
        let out = fx.b.p("bool.or", vec![], vec![below, above])?;
        self.guard(fx, out, |_, _| Ok(Atom::bytes(b"slice index out of range")))?;
        fx.b.cast_in(kv, Ty::Int(0, INT_MAX))
    }

    fn field_index(&self, rt: &RType, f: &str) -> R<usize> {
        match rt {
            RType::Named(n) if !self.t.kinds.contains_key(n) => match self.t.fields(n).and_then(|fs| fs.iter().position(|(x, _)| x == f)) {
                Some(k) => Ok(k),
                None => no(format!("the field {f} of {n}")),
            },
            o => no(format!("the field {f} of {}", o.show())),
        }
    }

    fn member(&mut self, fx: &mut Fx, x: &Expr, f: &str, rt: RType) -> R<Val> {
        let xt = self.node_ty(x);
        let k = self.field_index(&xt, f)?;
        if let Some(v) = self.var_of(fx, x) {
            fx.b.tick(1);
            self.charge_k(fx, 1)?;
            let t = fx.b.local_ty(v);
            return Ok(Val { a: fx.b.p("tup.get", vec![t], vec![Atom::V(v), lit(k as i128)])?, t: rt });
        }
        if let Some(Bound::Val(a, _)) = self.bound(fx, x)
            && matches!(x, Expr::Name(..) | Expr::SelfRef(_))
        {
            fx.b.tick(1);
            self.charge_k(fx, 1)?;
            let t = fx.b.ty(&a);
            return Ok(Val { a: fx.b.p("tup.get", vec![t], vec![a, lit(k as i128)])?, t: rt });
        }
        let v = self.expr(fx, x)?;
        self.charge_k(fx, 1)?;
        let t = fx.b.ty(&v.a);
        let c = fx.b.p("tup.get", vec![t], vec![v.a.clone(), lit(k as i128)])?;
        fx.b.drop(v.a)?;
        Ok(Val { a: c, t: rt })
    }

    fn cast(&mut self, fx: &mut Fx, x: &Expr, ty: &ast::Ty, rt: RType) -> R<Val> {
        let v = self.expr(fx, x)?;
        let target = self.t.alias(ty);
        match (&v.t, &rt) {
            (RType::Int(_), RType::Int(ti)) => {
                let (r, name) = check_of(ti)?;
                if fx.b.ty(&v.a).within(&r) {
                    return Ok(Val { a: v.a, t: rt });
                }
                let c = fx.b.p("int.bad.cast", vec![r.clone()], vec![v.a.clone()])?;
                let a = v.a.clone();
                self.guard(fx, c, |lw, fx| lw.ovf(fx, "value", vec![a], &name))?;
                Ok(Val { a: fx.b.p("int.cast", vec![r], vec![v.a])?, t: rt })
            }
            (RType::Bool, RType::Int(_)) => {
                let t = self.ct(&rt)?;
                let r = fx.b.var(t, lit(0));
                fx.b.open();
                fx.b.set(r, lit(1));
                let th = fx.b.close();
                fx.b.push(Stmt::If(v.a, th, vec![]));
                Ok(Val { a: fx.b.let_(Atom::V(r)), t: rt })
            }
            (RType::Int(_), RType::F64) => {
                let t = fx.b.ty(&v.a);
                Ok(Val { a: fx.b.p("f64.of_int", vec![t], vec![v.a])?, t: rt })
            }
            (RType::Int(_), RType::F32) => {
                let g = rtf::int_to_f32(&mut self.f)?;
                Ok(Val { a: self.call_rt(fx, g, vec![v.a]), t: rt })
            }
            (RType::F64 | RType::F32, RType::Int(ti)) => {
                // interp `trunc_chk`: trunc(x) checked into the type, the trap
                // naming its exact value
                let x = if v.t == RType::F32 { fx.b.p("f32.to_f64", vec![], vec![v.a])? } else { v.a };
                let Some(n) = ti.tags.single() else { return no("a cast to an integer type whose name differs at run time") };
                let n = if n == "lit" { "i64" } else { n };
                let (lo, hi) = typed::tag_range(n);
                let r = Ty::Int(lo, hi);
                let c = fx.b.p("f64.bad.trunc", vec![r.clone()], vec![x.clone()])?;
                let x2 = x.clone();
                self.guard(fx, c, |lw, fx| {
                    let g = rtf::int_text(&mut lw.f)?;
                    let d = lw.call_rt(fx, g, vec![x2]);
                    fx.b.cats(vec![Atom::bytes(b"overflow: "), d, Atom::bytes(format!(" is outside {n}").as_bytes())])
                })?;
                Ok(Val { a: fx.b.p("f64.trunc", vec![r], vec![x])?, t: rt })
            }
            (RType::F64, RType::F64) | (RType::F32, RType::F32) => Ok(Val { a: v.a, t: rt }),
            (RType::F32, RType::F64) => Ok(Val { a: fx.b.p("f32.to_f64", vec![], vec![v.a])?, t: rt }),
            (RType::F64, RType::F32) => {
                let c = fx.b.p("f64.bad.to_f32", vec![], vec![v.a.clone()])?;
                self.guard(fx, c, |_, _| Ok(Atom::bytes(b"non-finite float")))?;
                Ok(Val { a: fx.b.p("f64.to_f32", vec![], vec![v.a])?, t: rt })
            }
            (a, _) => no(format!("a cast of {} to {}", a.show(), crate::interp::ty_str(&target))),
        }
    }

    fn bin(&mut self, fx: &mut Fx, op: &str, a: &Expr, b: &Expr, rt: RType) -> R<Val> {
        if op == "and" || op == "or" {
            self.charge_k(fx, 1)?;
            let x = self.expr(fx, a)?;
            let r = fx.b.var(Ty::Bool, Atom::bool(op == "or"));
            fx.b.open();
            let y = self.expr(fx, b)?;
            fx.b.set(r, y.a);
            let go = fx.b.close();
            if op == "and" {
                fx.b.push(Stmt::If(x.a, go, vec![]));
            } else {
                fx.b.push(Stmt::If(x.a, vec![], go));
            }
            return Ok(Val { a: fx.b.let_(Atom::V(r)), t: rt });
        }
        if (op == "in" || op == "not in")
            && let Expr::Bin(rop, lo, hi, _) = b
            && (rop == ".." || rop == "..=")
        {
            // `x in a..b`: the range is not built (interp `contains` of a range)
            let x = self.expr(fx, a)?;
            fx.b.tick(1);
            let l = self.expr(fx, lo)?;
            let h = self.expr(fx, hi)?;
            self.charge_k(fx, 1)?; // the range
            self.charge_k(fx, op_cost(op, &x.t))?;
            let r = match (&x.t, &l.t, &h.t) {
                (RType::Int(_), RType::Int(_), RType::Int(_)) => {
                    let ge = fx.b.int_cmp("int.le", l.a, x.a.clone())?;
                    let below = fx.b.int_cmp(if rop == ".." { "int.lt" } else { "int.le" }, x.a, h.a)?;
                    fx.b.p("bool.and", vec![], vec![ge, below])?
                }
                (_, RType::Int(_), RType::Int(_)) => {
                    fx.b.drop(x.a)?;
                    Atom::bool(false)
                }
                _ => return no("a range of non-integers"),
            };
            let r = if op == "not in" { fx.b.not(r)? } else { r };
            return Ok(Val { a: r, t: rt });
        }
        let x = self.expr(fx, a)?;
        let y = self.expr(fx, b)?;
        self.charge_k(fx, op_cost(op, &x.t))?;
        if op == "in" || op == "not in" {
            let r = self.contains(fx, y, x)?;
            let r = if op == "not in" { fx.b.not(r)? } else { r };
            return Ok(Val { a: r, t: rt });
        }
        match op {
            "==" | "!=" => {
                let e = self.eq(fx, x, y)?;
                let e = if op == "!=" { fx.b.not(e)? } else { e };
                Ok(Val { a: e, t: rt })
            }
            "<" | "<=" | ">" | ">=" => {
                let (p, q) = if op == "<" || op == "<=" { (x, y) } else { (y, x) };
                let strict = op == "<" || op == ">";
                let r = match (&p.t, &q.t) {
                    (RType::Int(_), RType::Int(_)) => fx.b.int_cmp(if strict { "int.lt" } else { "int.le" }, p.a, q.a)?,
                    (RType::F64, RType::F64) => fx.b.p(if strict { "f64.lt" } else { "f64.le" }, vec![], vec![p.a, q.a])?,
                    (RType::F32, RType::F32) => {
                        let x = fx.b.p("f32.to_f64", vec![], vec![p.a])?;
                        let y = fx.b.p("f32.to_f64", vec![], vec![q.a])?;
                        fx.b.p(if strict { "f64.lt" } else { "f64.le" }, vec![], vec![x, y])?
                    }
                    (RType::Text, RType::Text) => {
                        if strict {
                            fx.b.p("bytes.lt", vec![], vec![p.a, q.a])?
                        } else {
                            let l = fx.b.p("bytes.lt", vec![], vec![q.a, p.a])?;
                            fx.b.not(l)?
                        }
                    }
                    (RType::Bool, RType::Bool) => {
                        let np = fx.b.not(p.a)?;
                        if strict { fx.b.p("bool.and", vec![], vec![np, q.a])? } else { fx.b.p("bool.or", vec![], vec![np, q.a])? }
                    }
                    (RType::Tuple(_), RType::Tuple(_)) | (RType::List(_), RType::List(_)) if p.t == q.t => {
                        let f = self.ord_fn(&p.t)?;
                        let c = self.call_rt(fx, f, vec![p.a, q.a]);
                        fx.b.int_cmp(if strict { "int.lt" } else { "int.le" }, c, lit(0))?
                    }
                    // a value of no type is never made: the test is not reached
                    (RType::Never, _) | (_, RType::Never) => {
                        fx.b.drop(p.a)?;
                        fx.b.drop(q.a)?;
                        Atom::bool(false)
                    }
                    (s, t) => return no(format!("the order of {} and {}", s.show(), t.show())),
                };
                Ok(Val { a: r, t: rt })
            }
            ".." | "..=" => no(format!("the operator {op}")),
            _ => self.arith(fx, op, x, y, rt),
        }
    }

    /// interp `arith` on two values.
    fn arith(&mut self, fx: &mut Fx, op: &str, x: Val, y: Val, rt: RType) -> R<Val> {
        match (&x.t, &y.t) {
            (RType::Int(i), RType::Int(j)) => {
                let res = IntT::of_tags(i.tags.join_op(j.tags));
                let (r, name) = match check_of(&res) {
                    Ok(x) => x,
                    Err(e) => {
                        // the names the result may carry check different
                        // ranges; when the exact result lies in every one of
                        // them, no name makes the operation trap
                        let (Some((al, ah)), Some((bl, bh))) = (fx.b.ty(&x.a).bounds(), fx.b.ty(&y.a).bounds()) else { return Err(e) };
                        let exact = match op {
                            "+" => Some((al + bl, ah + bh)),
                            "-" => Some((al - bh, ah - bl)),
                            "*" => {
                                let ps = [al.checked_mul(bl), al.checked_mul(bh), ah.checked_mul(bl), ah.checked_mul(bh)];
                                if ps.iter().all(|p| p.is_some()) {
                                    let ps: Vec<i128> = ps.iter().map(|p| p.unwrap()).collect();
                                    Some((*ps.iter().min().unwrap(), *ps.iter().max().unwrap()))
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        };
                        let (lo, hi) = res.tags.names().into_iter().map(typed::tag_range).fold((INT_MIN, INT_MAX), |(l, h), (a, b)| (l.max(a), h.min(b)));
                        let Some((el, eh)) = exact.filter(|(el, eh)| lo <= *el && *eh <= hi) else { return Err(e) };
                        let p = match op { "+" => "add", "-" => "sub", _ => "mul" };
                        let a = fx.b.arith_in(p, x.a, y.a, Ty::Int(el, eh))?;
                        return Ok(Val { a, t: rt });
                    }
                };
                let a = self.int_op(fx, op, x.a, y.a, &r, &name, &res)?;
                Ok(Val { a, t: rt })
            }
            (RType::F64, RType::F64) | (RType::F64, RType::Int(_)) | (RType::Int(_), RType::F64) => {
                let fa = if matches!(x.t, RType::Int(_)) { let t = fx.b.ty(&x.a); fx.b.p("f64.of_int", vec![t], vec![x.a])? } else { x.a };
                let fb = if matches!(y.t, RType::Int(_)) { let t = fx.b.ty(&y.a); fx.b.p("f64.of_int", vec![t], vec![y.a])? } else { y.a };
                let name = match op { "+" => "add", "-" => "sub", "*" => "mul", "/" => "div", _ => return no(format!("the float operator {op}")) };
                let c = fx.b.p(&format!("f64.bad.{name}"), vec![], vec![fa.clone(), fb.clone()])?;
                self.guard(fx, c, |_, _| Ok(Atom::bytes(b"non-finite float")))?;
                Ok(Val { a: fx.b.p(&format!("f64.{name}"), vec![], vec![fa, fb])?, t: rt })
            }
            (RType::F32, RType::F32) => {
                let name = match op { "+" => "add", "-" => "sub", "*" => "mul", "/" => "div", _ => return no(format!("the float operator {op}")) };
                let c = fx.b.p(&format!("f32.bad.{name}"), vec![], vec![x.a.clone(), y.a.clone()])?;
                self.guard(fx, c, |_, _| Ok(Atom::bytes(b"non-finite float")))?;
                Ok(Val { a: fx.b.p(&format!("f32.{name}"), vec![], vec![x.a, y.a])?, t: rt })
            }
            (RType::List(xe), RType::List(ye)) if matches!(op, "+" | "-" | "*" | "/" | "%") => {
                // lane-wise (interp `arith` of two lists): the same length,
                // then the operation item by item, in order
                let (xe, ye) = ((**xe).clone(), (**ye).clone());
                let RType::List(re) = rt.clone() else { return no("lane-wise arithmetic of unknown type") };
                let re = (*re).clone();
                let (xc, yc, rc) = (self.ct(&xe)?, self.ct(&ye)?, self.ct(&re)?);
                let n1 = fx.b.p("list.len", vec![xc.clone()], vec![x.a.clone()])?;
                let n2 = fx.b.p("list.len", vec![yc.clone()], vec![y.a.clone()])?;
                let ne = fx.b.int_cmp("int.eq", n1.clone(), n2)?;
                let bad = fx.b.not(ne)?;
                self.guard(fx, bad, |_, _| Ok(Atom::bytes(b"lane count mismatch")))?;
                let o0 = fx.b.p("list.new", vec![rc.clone()], vec![])?;
                let out = fx.b.var(Ty::list(rc.clone()), o0);
                let i = fx.b.var(Ty::u64(), lit(0));
                fx.b.open();
                {
                    let iv = fx.b.let_(Atom::V(i));
                    let more = fx.b.int_cmp("int.lt", iv.clone(), n1)?;
                    fx.b.open();
                    let th = fx.b.close();
                    fx.b.open();
                    fx.b.brk(1);
                    let el = fx.b.close();
                    fx.b.push(Stmt::If(more, th, el));
                    let p = fx.b.list_at(x.a.clone(), iv.clone())?;
                    let q = fx.b.list_at(y.a.clone(), iv.clone())?;
                    let r = self.arith(fx, op, Val { a: p, t: xe.clone() }, Val { a: q, t: ye.clone() }, re.clone())?;
                    let r = self.conv(fx, r.a, &re, &re, false)?;
                    let o = fx.b.let_(Atom::V(out));
                    let o = fx.b.p("list.push", vec![rc.clone()], vec![o, r])?;
                    fx.b.set(out, o);
                    let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
                    fx.b.set(i, i1);
                }
                let body = fx.b.close();
                fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
                fx.b.drop(x.a)?;
                fx.b.drop(y.a)?;
                Ok(Val { a: fx.b.let_(Atom::V(out)), t: rt })
            }
            (RType::Text, RType::Text) if op == "+" => {
                let c = fx.b.cat(x.a, y.a)?;
                // N9: a text above SIZE_LIMIT bytes is checked by its code points
                let n = fx.b.p("bytes.len", vec![], vec![c.clone()])?;
                let big = fx.b.int_cmp("int.lt", lit(crate::interp::SIZE_LIMIT as i128), n)?;
                fx.b.open();
                let cnt = rt::utf8_count(&mut self.f)?;
                let k = self.call_rt(fx, cnt, vec![c.clone()]);
                let over = fx.b.int_cmp("int.lt", lit(crate::interp::SIZE_LIMIT as i128), k)?;
                self.guard(fx, over, |_, _| Ok(Atom::bytes(b"size limit")))?;
                let th = fx.b.close();
                fx.b.push(Stmt::If(big, th, vec![]));
                Ok(Val { a: c, t: rt })
            }
            (s, t) if typed::no_arith_rule(s, t) || matches!((s, t, op), (RType::Text, RType::Text, _)) => {
                // interp `arith`: no rule for these operands; the trap names
                // both values as display shows them
                let (sa, sb) = (self.show(fx, x.a, &x.t)?, self.show(fx, y.a, &y.t)?);
                let m = fx.b.cats(vec![Atom::bytes(format!("cannot apply {op} to ").as_bytes()), sa, Atom::bytes(b" and "), sb])?;
                self.trap(fx, m)?;
                let t = self.ct(&rt)?;
                Ok(Val { a: fx.b.unreached(t), t: rt })
            }
            (s, t) => no(format!("{} {op} {}", s.show(), t.show())),
        }
    }

    /// A checked integer operation with result range `r` (interp `arith`
    /// with the result's type name; `name` is that name as a trap prints it).
    #[allow(clippy::too_many_arguments)]
    fn int_op(&mut self, fx: &mut Fx, op: &str, a: Atom, b: Atom, r: &Ty, name: &str, res: &IntT) -> R<Atom> {
        let ovf = |lw: &mut Self, fx: &mut Fx, kind: &str, args: Vec<Atom>| lw.ovf(fx, kind, args, name);
        match op {
            "+" | "-" | "*" => {
                let p = match op { "+" => "add", "-" => "sub", _ => "mul" };
                let c = fx.b.p(&format!("int.bad.{p}"), vec![r.clone()], vec![a.clone(), b.clone()])?;
                let (a2, b2) = (a.clone(), b.clone());
                self.guard(fx, c, |lw, fx| ovf(lw, fx, p, vec![a2, b2]))?;
                fx.b.p(&format!("int.{p}"), vec![r.clone()], vec![a, b])
            }
            "/" | "%" => {
                let z = fx.b.int_cmp("int.eq", b.clone(), lit(0))?;
                self.guard(fx, z, |_, _| Ok(Atom::bytes(b"division by zero")))?;
                let c = fx.b.p("int.bad.quo", vec![r.clone()], vec![a.clone(), b.clone()])?;
                let (a2, b2) = (a.clone(), b.clone());
                self.guard(fx, c, |lw, fx| ovf(lw, fx, "quo", vec![a2, b2]))?;
                if op == "/" {
                    return fx.b.p("int.quo", vec![r.clone()], vec![a, b]);
                }
                // the remainder, computed where it always lies, then checked
                let at = fx.b.ty(&a);
                let bt = fx.b.ty(&b);
                let h = hull(&hull(&at, &bt), &Ty::Int(0, 0));
                let c = fx.b.p("int.bad.quo", vec![h.clone()], vec![a.clone(), b.clone()])?;
                self.guard(fx, c, |_, _| Ok(Atom::bytes(b"internal: remainder")))?;
                let rem = fx.b.p("int.rem", vec![h], vec![a, b])?;
                if fx.b.ty(&rem).within(r) {
                    return Ok(rem);
                }
                let c = fx.b.p("int.bad.cast", vec![r.clone()], vec![rem.clone()])?;
                let rm = rem.clone();
                self.guard(fx, c, |lw, fx| ovf(lw, fx, "value", vec![rm]))?;
                fx.b.p("int.cast", vec![r.clone()], vec![rem])
            }
            "&" | "|" | "^" => {
                let p = match op { "&" => "and", "|" => "or", _ => "xor" };
                let at = fx.b.ty(&a);
                let bt = fx.b.ty(&b);
                if at.within(r) && bt.within(r) {
                    return fx.b.p(&format!("int.{p}"), vec![r.clone()], vec![a, b]);
                }
                // computed at i64 or u64, then checked against the result type
                let w = [Ty::i64(), Ty::u64()].into_iter().find(|w| at.within(w) && bt.within(w));
                let Some(w) = w else { return no("a bit operation on values of i64 and u64 ranges together") };
                let v = fx.b.p(&format!("int.{p}"), vec![w], vec![a, b])?;
                let c = fx.b.p("int.bad.cast", vec![r.clone()], vec![v.clone()])?;
                let vv = v.clone();
                self.guard(fx, c, |lw, fx| ovf(lw, fx, "value", vec![vv]))?;
                fx.b.p("int.cast", vec![r.clone()], vec![v])
            }
            "<<" | ">>" => {
                let tag = res.tags.single().ok_or("not covered by the lowering: a shift whose type name differs at run time")?;
                let bits = typed::tag_bits(tag);
                let (lo, hi) = r.bounds().unwrap();
                if crate::one::ir::bits(lo, hi) != bits {
                    return no("a shift whose width differs from its range");
                }
                let c = fx.b.p("int.bad.shift", vec![r.clone()], vec![a.clone(), b.clone()])?;
                self.guard(fx, c, |_, _| Ok(Atom::bytes(b"shift amount out of range")))?;
                if op == "<<" {
                    if tag == "lit" {
                        // an untyped value: the 64 low bits, then checked as i64
                        let c = fx.b.p("int.bad.shift", vec![Ty::u64()], vec![a.clone(), b.clone()])?;
                        self.guard(fx, c, |_, _| Ok(Atom::bytes(b"internal: shift")))?;
                        let v = fx.b.p("int.shl", vec![Ty::u64()], vec![a, b])?;
                        let c = fx.b.p("int.bad.cast", vec![r.clone()], vec![v.clone()])?;
                        let vv = v.clone();
                        self.guard(fx, c, |lw, fx| ovf(lw, fx, "value", vec![vv]))?;
                        return fx.b.p("int.cast", vec![r.clone()], vec![v]);
                    }
                    return fx.b.p("int.shl", vec![r.clone()], vec![a, b]);
                }
                let at = fx.b.ty(&a);
                let w = [r.clone(), Ty::i64(), Ty::u64()].into_iter().find(|w| hull(&at, &Ty::Int(0, 0)).within(w));
                let Some(w) = w else { return no("a shift of a value outside i64 and u64") };
                if &w != r {
                    let c = fx.b.p("int.bad.shift", vec![w.clone()], vec![a.clone(), b.clone()])?;
                    self.guard(fx, c, |_, _| Ok(Atom::bytes(b"internal: shift")))?;
                }
                let v = fx.b.p("int.shr", vec![w], vec![a, b])?;
                if fx.b.ty(&v).within(r) {
                    return Ok(v);
                }
                let c = fx.b.p("int.bad.cast", vec![r.clone()], vec![v.clone()])?;
                let vv = v.clone();
                self.guard(fx, c, |lw, fx| ovf(lw, fx, "value", vec![vv]))?;
                fx.b.p("int.cast", vec![r.clone()], vec![v])
            }
            _ => no(format!("the operator {op}")),
        }
    }

    fn catch_expr(&mut self, fx: &mut Fx, e: &Expr, x: &Expr, bind: Option<&str>, h: &Expr, rt: RType) -> R<Val> {
        // `m.get(k) catch h`: the handler runs when k is not a key, and no
        // error value is needed when the handler does not bind it
        if bind.is_none()
            && let Expr::Call(f, args, _, _) = x
            && let Expr::Member(obj, mname, _) = &**f
            && mname == "get"
            && args.len() == 1
            && matches!(self.ty.node(x).and_then(|n| n.callee.clone()), Some(Callee::Builtin(_)))
            && let RType::Map(_, vt) = self.node_ty(obj)
        {
            let vt = (*vt).clone();
            fx.b.tick(1); // the call
            let m = self.expr(fx, obj)?;
            let k = self.expr(fx, &args[0].e)?;
            let sized = matches!(x, Expr::Call(_, _, _, true));
            self.charge_method(fx, "get", (m.a.clone(), m.t.clone()), Some((&k.a, &k.t)), sized)?;
            let i = self.map_find(fx, &m, k.clone())?;
            fx.b.drop(k.a)?;
            let t = self.ct(&rt)?;
            let r = self.result_var(fx, &t)?;
            let found = fx.b.int_cmp("int.le", lit(0), i.clone())?;
            fx.b.open();
            let v = self.map_value_at(fx, m.clone(), i)?;
            let a = self.conv(fx, v.a, &vt, &rt, false)?;
            self.store_var(fx, r, a)?;
            let th = fx.b.close();
            fx.b.open();
            fx.b.drop(m.a)?;
            fx.push_scope();
            let hv = self.expr(fx, h)?;
            let a = self.conv(fx, hv.a, &hv.t, &rt, false)?;
            self.store_var(fx, r, a)?;
            fx.pop_scope()?;
            let el = fx.b.close();
            fx.b.push(Stmt::If(found, th, el));
            return Ok(Val { a: fx.b.let_(Atom::V(r)), t: rt });
        }
        let v = self.expr(fx, x)?;
        let RType::Union(ok, err) = v.t.clone() else { return Ok(v) };
        let t = self.ct(&rt)?;
        let r = self.result_var(fx, &t)?;
        let vt = fx.b.ty(&v.a);
        let bind_id = self.ty.catch_binds.get(&addr(e)).copied();
        let _ = bind;
        self.switch(fx, &v.a, &vt, 2, |lw, fx, k| {
            let p = fx.b.p("sum.get", vec![vt.clone()], vec![v.a.clone(), lit(k as i128)])?;
            if k == 0 {
                let a = lw.conv(fx, p, &ok, &rt, false)?;
                lw.store_var(fx, r, a)
            } else {
                fx.push_scope();
                match bind_id {
                    Some(id) => {
                        fx.bound.insert(id, Bound::Val(p.clone(), (*err).clone()));
                        fx.own(p);
                    }
                    None => fx.b.drop(p)?,
                }
                let hv = lw.expr(fx, h)?;
                let a = lw.conv(fx, hv.a, &hv.t, &rt, false)?;
                lw.store_var(fx, r, a)?;
                fx.pop_scope()
            }
        })?;
        Ok(Val { a: fx.b.let_(Atom::V(r)), t: rt })
    }

    fn try_expr(&mut self, fx: &mut Fx, x: &Expr, els: Option<&Expr>, rt: RType) -> R<Val> {
        let v = self.expr(fx, x)?;
        match v.t.clone() {
            RType::Union(ok, err) => {
                let t = self.ct(&rt)?;
                let r = self.result_var(fx, &t)?;
                let vt = fx.b.ty(&v.a);
                self.switch(fx, &v.a, &vt, 2, |lw, fx, k| {
                    let p = fx.b.p("sum.get", vec![vt.clone()], vec![v.a.clone(), lit(k as i128)])?;
                    if k == 0 {
                        match (&*ok, els) {
                            (RType::Opt(inner), Some(e)) => {
                                let pt = fx.b.ty(&p);
                                let inner = (**inner).clone();
                                lw.switch(fx, &p, &pt, 2, |lw, fx, j| {
                                    let q = fx.b.p("sum.get", vec![pt.clone()], vec![p.clone(), lit(j as i128)])?;
                                    if j == 1 {
                                        let a = lw.conv(fx, q, &inner, &rt, false)?;
                                        lw.store_var(fx, r, a)
                                    } else {
                                        fx.b.drop(q)?;
                                        let ev = lw.expr(fx, e)?;
                                        lw.fail(fx, ev)
                                    }
                                })
                            }
                            _ => {
                                let a = lw.conv(fx, p, &ok, &rt, false)?;
                                lw.store_var(fx, r, a)
                            }
                        }
                    } else {
                        match els {
                            Some(e) => {
                                fx.b.drop(p)?;
                                let ev = lw.expr(fx, e)?;
                                lw.fail(fx, ev)
                            }
                            None => lw.fail(fx, Val { a: p, t: (*err).clone() }),
                        }
                    }
                })?;
                Ok(Val { a: fx.b.let_(Atom::V(r)), t: rt })
            }
            RType::Opt(inner) => {
                let Some(e) = els else { return Ok(v) };
                let t = self.ct(&rt)?;
                let r = self.result_var(fx, &t)?;
                let vt = fx.b.ty(&v.a);
                self.switch(fx, &v.a, &vt, 2, |lw, fx, k| {
                    let p = fx.b.p("sum.get", vec![vt.clone()], vec![v.a.clone(), lit(k as i128)])?;
                    if k == 1 {
                        let a = lw.conv(fx, p, &inner, &rt, false)?;
                        lw.store_var(fx, r, a)
                    } else {
                        fx.b.drop(p)?;
                        let ev = lw.expr(fx, e)?;
                        lw.fail(fx, ev)
                    }
                })?;
                Ok(Val { a: fx.b.let_(Atom::V(r)), t: rt })
            }
            _ => Ok(v),
        }
    }

    /// The name of the declared error a value is (interp `rec_name`).
    fn err_name(&mut self, fx: &mut Fx, v: &Atom, rt: &RType) -> R<Atom> {
        match rt {
            RType::Named(n) => {
                if let Some(vs) = self.t.kinds.get(n).cloned() {
                    let r = fx.b.var(Ty::Bytes, Atom::bytes(b""));
                    let t = fx.b.ty(v);
                    self.switch(fx, v, &t, vs.len(), |_, fx, k| {
                        fx.b.set(r, Atom::bytes(vs[k].0.as_bytes()));
                        Ok(())
                    })?;
                    Ok(fx.b.let_(Atom::V(r)))
                } else {
                    Ok(Atom::bytes(n.as_bytes()))
                }
            }
            _ => Ok(Atom::bytes(b"?")),
        }
    }

    /// `fail e`: return the error value from a union function, or halt with
    /// the error's name and display (a kept step, `main`'s failure).
    fn fail(&mut self, fx: &mut Fx, v: Val) -> R<()> {
        match fx.fail {
            FailCx::Union => {
                let ut = fx.union_t.clone().ok_or("internal: a union function without its type")?;
                let Ty::Sum(cs) = &ut else { unreachable!() };
                let RType::Union(_, err) = (match &fx.ret {
                    Ret::Fn(Some(t)) => self.t.of_ty(t),
                    _ => RType::Never,
                }) else {
                    return no("fail outside a union function");
                };
                let a = if fx.b.ty(&v.a) == cs[1] { v.a } else { self.conv(fx, v.a, &v.t, &err, false)? };
                let r = fx.b.p("sum.mk", vec![ut], vec![lit(1), a])?;
                let r = self.with_muts(fx, r)?;
                fx.drops_to(None)?;
                fx.b.ret(r);
            }
            FailCx::Halt => {
                let name = self.err_name(fx, &v.a, &v.t)?;
                let d = self.show(fx, v.a, &v.t)?;
                let p = fx.b.p("tup.mk", vec![Ty::Tup(vec![Ty::Bytes, Ty::Bytes])], vec![name, d])?;
                let h = fx.b.p("sum.mk", vec![Ty::halt()], vec![lit(1), p])?;
                fx.b.halt(h);
            }
        }
        Ok(())
    }
}

impl<'m> Lw<'m> {
    // ---- statements

    /// The type of the value a statement list leaves when it ends in a value.
    fn tail_type(&self, b: &[S]) -> Option<RType> {
        match b.last()? {
            S::Expr(e) => Some(self.node_ty(e)),
            S::If(arms, Some(els)) => {
                let mut t = self.tail_type(els)?;
                for (_, x) in arms {
                    t = t.join(&self.tail_type(x)?);
                }
                Some(t)
            }
            S::Match(_, arms, _) => {
                let mut t = RType::Never;
                for a in arms {
                    t = t.join(&self.tail_type(&a.body)?);
                }
                Some(t)
            }
            _ => None,
        }
    }

    /// A statement list in a scope of its own (interp `exec_block_tail`);
    /// with `tail`, the value it ends in.
    fn block(&mut self, fx: &mut Fx, b: &[S], tail: bool) -> R<Option<Val>> {
        fx.push_scope();
        let mut last = None;
        for (i, s) in b.iter().enumerate() {
            last = self.stmt(fx, s, tail && i + 1 == b.len())?;
            if !fx.b.falls() {
                break;
            }
        }
        fx.pop_scope()?;
        Ok(last)
    }

    fn stmt(&mut self, fx: &mut Fx, s: &S, tail: bool) -> R<Option<Val>> {
        fx.b.tick(1);
        match s {
            S::Expr(e) => {
                let v = self.expr(fx, e)?;
                if tail {
                    return Ok(Some(v));
                }
                fx.b.drop(v.a)?;
            }
            S::Assign(op, place, rhs) => self.assign(fx, s, op, place, rhs)?,
            S::Let(p, ty, e, _) => {
                let v = self.expr(fx, e)?;
                let v = match ty {
                    Some(t) => self.fit(fx, v.a, &v.t, t)?,
                    None => v,
                };
                self.let_pat(fx, p, v)?;
            }
            S::Var(n, ty, e, _) => {
                let v = self.expr(fx, e)?;
                let v = match ty {
                    Some(t) => self.fit(fx, v.a, &v.t, t)?,
                    None => v,
                };
                let Some(id) = self.ty.var_binds.get(&addr(s)).copied() else { return no(format!("the var {n}")) };
                let st = self.ty.locals[id as usize].ty.clone();
                let ct = self.ct(&st)?;
                let a = self.conv(fx, v.a, &v.t, &st, false)?;
                let x = fx.b.var(ct, a);
                fx.bound.insert(id, Bound::Var(x, st));
                fx.own(Atom::V(x));
            }
            S::If(arms, els) => return self.if_stmt(fx, arms, els.as_deref(), tail),
            S::While(c, b, body) => self.run_loop(fx, b.as_ref(), Some(c), body)?,
            S::Loop(b, body) => self.run_loop(fx, b.as_ref(), None, body)?,
            S::For(p, it, body, _) => self.for_stmt(fx, p, it, body)?,
            S::Match(e, arms, _) => return self.match_stmt(fx, e, arms, tail),
            S::Return(e, _) => {
                let v = match e {
                    Some(e) => self.expr(fx, e)?,
                    None => Val { a: fx.b.unit()?, t: RType::Unit },
                };
                match fx.ret.clone() {
                    Ret::Fn(decl) => self.finish_fn(fx, v, decl.as_ref())?,
                    Ret::Gen => {
                        // interp `gen_run`: a return ends the run, finished
                        fx.b.drop(v.a)?;
                        fx.drops_to(None)?;
                        fx.b.ret(Atom::bool(true));
                    }
                    Ret::Step => return no("return in a step"),
                    Ret::None => return no("return outside a function"),
                }
            }
            S::Break(_) => {
                let Some(l) = fx.loops.last() else { return no("break outside a loop") };
                let t = l.brk;
                fx.break_to(t)?;
            }
            S::Continue(_) => {
                let Some(l) = fx.loops.last() else { return no("continue outside a loop") };
                let t = l.cont;
                fx.break_to(t)?;
            }
            S::Pass => {}
            S::Emit(e) => {
                let v = self.expr(fx, e)?;
                let s = self.show(fx, v.a, &v.t)?;
                let o = fx.b.let_global(self.out_g, Ty::list(Ty::Bytes));
                let o = fx.b.p("list.push", vec![Ty::Bytes], vec![o, s])?;
                fx.b.setg(self.out_g, o);
            }
            S::Fail(e) => {
                let v = self.expr(fx, e)?;
                self.fail(fx, v)?;
            }
            S::Check(Expr::Call(f, _, _, _)) if matches!(&**f, Expr::Name(n, _) if n == "@contract") => return no("a contract clause"),
            S::Check(e) => {
                let v = self.expr(fx, e)?;
                let bad = fx.b.not(v.a)?;
                self.guard(fx, bad, |_, _| Ok(Atom::bytes(b"check failed")))?;
            }
            S::Comptime(b) => {
                self.block(fx, b, false)?;
            }
            // `par` (C.4, T16): the branches in source order; their results
            // are those of the sequential run, which is what the reference
            // resolves them to
            S::Arena(name, None, body, _) if name == "$forpar" => self.for_par(fx, body)?,
            S::Arena(name, None, body, _) if name == "$par" => self.par(fx, body)?,
            S::Arena(name, None, body, _) if name == "$budget" => self.budget(fx, body)?,
            S::Arena(name, None, body, _) | S::Device(name, None, body, _) if !name.starts_with('$') => {
                self.block(fx, body, false)?;
            }
            S::Arena(_, Some(b), body, _) | S::Device(_, Some(b), body, _) => self.bounded_region(fx, b, body)?,
            S::Arena(..) | S::Device(..) => return no("a marker region (a budget, a deadline, a priority or a workgroup)"),
            S::Yield(e, _) => self.yield_stmt(fx, e)?,
            S::Send(..) | S::Spawn(..) | S::Select(..) => return no("a concurrency statement"),
        }
        Ok(None)
    }

    /// Bind an irrefutable `let` pattern.
    fn let_pat(&mut self, fx: &mut Fx, p: &Pat, v: Val) -> R<()> {
        match p {
            Pat::Wild => fx.b.drop(v.a),
            Pat::Path(path) if path.len() == 1 && !self.t.is_tested_name(&path[0]) => {
                let Some(id) = self.ty.pat_binds.get(&addr(p)).copied() else { return no("a let binding") };
                fx.bound.insert(id, Bound::Val(v.a.clone(), v.t));
                fx.own(v.a);
                Ok(())
            }
            Pat::Tuple(ps) => {
                let RType::Tuple(ts) = &v.t else { return no("a tuple pattern on a non-tuple") };
                if ts.len() != ps.len() {
                    return no("a tuple pattern of another length");
                }
                let t = fx.b.ty(&v.a);
                let cs = fx.b.prim("tup.open", vec![t], vec![v.a])?;
                for ((q, c), ct) in ps.iter().zip(cs).zip(ts.clone()) {
                    self.let_pat(fx, q, Val { a: c, t: ct })?;
                }
                Ok(())
            }
            _ => no("a refutable let pattern"),
        }
    }

    /// The place an assignment writes: a variable, a field of a variable, or
    /// an element of a list variable at a simple index.
    fn assign(&mut self, fx: &mut Fx, s: &S, op: &str, place: &Expr, rhs: &Expr) -> R<()> {
        let r = self.expr(fx, rhs)?;
        let cur_t = self.node_ty(place);
        let v = if op == "=" {
            r
        } else {
            let cur = self.place_value(fx, place)?;
            // `x op= y` costs what `x = x op y` costs (C.6.4)
            self.charge_k(fx, op_cost(&op[..op.len() - 1], &cur.t))?;
            let x = self.arith(fx, &op[..op.len() - 1], cur.clone(), r, RType::Never)?;
            // `x op= y` keeps x's type name (interp: chk(x, t))
            match (&cur.t, &x.a) {
                (RType::Int(c), _) => {
                    let ct = IntT::of_tags(c.tags);
                    let target = RType::Int(ct);
                    let xt = RType::Int(IntT { tags: c.tags, lo: INT_MIN, hi: INT_MAX });
                    let a = self.conv(fx, x.a, &xt, &target, true)?;
                    Val { a, t: target }
                }
                _ => Val { a: x.a, t: self.ty.stores.get(&addr(s)).cloned().unwrap_or(RType::Never) },
            }
        };
        // the place is evaluated again for its type (interp `exec`, `eval(place).ok()`),
        // and a charge its budget does not allow is ignored there
        self.place_ticks(fx, place)?;
        if self.costs {
            let k = place_charges(place);
            if k > 0 {
                let f = rt::try_charge(&mut self.f)?;
                fx.b.call(sym(f), Ty::Bool, vec![lit(k)]);
            }
        }
        let stored = self.ty.stores.get(&addr(s)).cloned().unwrap_or(RType::Never);
        let v = match (&cur_t, &v.t) {
            (RType::Int(c), RType::Int(_)) if !c.tags.has_lit() => {
                let a = self.conv(fx, v.a, &v.t, &RType::Int(IntT::of_tags(c.tags)), true)?;
                Val { a, t: stored }
            }
            (RType::Int(c), RType::Int(_)) if c.tags.names().len() > 1 => {
                // the place's name is `lit` (no check) or another name (a
                // check against it): when the value lies in every such
                // range, neither traps
                let vt = fx.b.ty(&v.a);
                let safe = c.tags.names().into_iter().filter(|n| *n != "lit").all(|n| {
                    let (lo, hi) = typed::tag_range(n);
                    vt.within(&Ty::Int(lo, hi))
                });
                if !safe {
                    return no("an assignment to a place whose type name differs at run time");
                }
                Val { a: v.a, t: if stored == RType::Never { v.t } else { stored } }
            }
            _ => Val { a: v.a, t: if stored == RType::Never { v.t } else { stored } },
        };
        self.store_place(fx, place, v)
    }

    /// The ticks of evaluating a place whose evaluation cannot trap.
    fn place_ticks(&mut self, fx: &mut Fx, p: &Expr) -> R<()> {
        match p {
            Expr::Name(..) | Expr::SelfRef(_) => fx.b.tick(1),
            Expr::Member(x, _, _) | Expr::TupleMember(x, _, _) => {
                fx.b.tick(1);
                self.place_ticks(fx, x)?;
            }
            Expr::Index(x, idx, _) if matches!(&idx[0], Expr::Name(..) | Expr::Int(..) | Expr::Str(..) | Expr::Bool(..)) => {
                fx.b.tick(2);
                self.place_ticks(fx, x)?;
            }
            Expr::Borrow(_, x, _) => {
                fx.b.tick(1);
                self.place_ticks(fx, x)?;
            }
            _ => return no("an assignment to this place"),
        }
        Ok(())
    }

    /// The value at a place, as an owned value.
    fn place_value(&mut self, fx: &mut Fx, p: &Expr) -> R<Val> {
        self.expr(fx, p)
    }

    /// interp `assign`.
    fn store_place(&mut self, fx: &mut Fx, p: &Expr, v: Val) -> R<()> {
        match p {
            Expr::Name(..) | Expr::SelfRef(_) => {
                let Some(x) = self.var_of(fx, p) else { return no("an assignment to a name that is not a var") };
                let st = self.var_rtype(fx, x).unwrap_or(v.t.clone());
                let a = self.conv(fx, v.a, &v.t, &st, false)?;
                self.store_var(fx, x, a)
            }
            Expr::Borrow(_, x, _) => self.store_place(fx, x, v),
            Expr::Member(x, f, _) => {
                let Some(var) = self.var_of(fx, x) else { return no("an assignment to a field of a value that is not a var") };
                fx.b.tick(1); // assign evaluates the record
                let xt = self.node_ty(x);
                let k = self.field_index(&xt, f)?;
                let RType::Named(n) = &xt else { unreachable!() };
                let fty = self.t.fields(n).unwrap()[k].1.clone();
                let frt = self.t.of_ty(&fty);
                let a = self.conv(fx, v.a, &v.t, &frt, false)?;
                let t = fx.b.local_ty(var);
                let rec = self.take_var(fx, var);
                let mut cs = fx.b.prim("tup.open", vec![t.clone()], vec![rec])?;
                fx.b.drop(cs[k].clone())?;
                cs[k] = a;
                let r = fx.b.p("tup.mk", vec![t], cs)?;
                fx.b.set(var, r);
                Ok(())
            }
            Expr::Index(x, idx, _) if matches!(&idx[0], Expr::Name(..) | Expr::Int(..) | Expr::Str(..) | Expr::Bool(..)) && matches!(self.node_ty(x), RType::Map(..)) => {
                // interp `assign`: the key, then the map, then `map_set`
                let Some(var) = self.var_of(fx, x) else { return no("an assignment to an element of a map that is not a var") };
                let k = self.expr(fx, &idx[0])?;
                fx.b.tick(1);
                let Some(mt @ RType::Map(..)) = self.var_rtype(fx, var) else { return no("an indexed assignment to a map of unknown type") };
                let RType::Map(kt, vt) = &mt else { unreachable!() };
                let (kt, vt) = ((**kt).clone(), (**vt).clone());
                let setf = self.map_set_fn(&mt)?;
                let ka = self.conv(fx, k.a, &k.t, &kt, false)?;
                let va = self.conv(fx, v.a, &v.t, &vt, false)?;
                let m = self.take_var(fx, var);
                let m2 = self.call_rt(fx, setf, vec![m, ka, va]);
                fx.b.set(var, m2);
                Ok(())
            }
            Expr::Index(x, idx, _) if matches!(&idx[0], Expr::Name(..) | Expr::Int(..)) => {
                let Some(var) = self.var_of(fx, x) else { return no("an assignment to an element of a value that is not a var") };
                let i = self.expr(fx, &idx[0])?;
                fx.b.tick(1);
                let lt = fx.b.local_ty(var);
                let Ty::List(et) = &lt else { return no("an indexed assignment to a non-list") };
                let et = (**et).clone();
                let Some(RType::List(ert)) = self.var_rtype(fx, var) else { return no("an indexed assignment to a list of unknown type") };
                let neg = fx.b.int_cmp("int.lt", i.a.clone(), lit(0))?;
                let ia = i.a.clone();
                let msg = |lw: &mut Self, fx: &mut Fx, ia: Atom| -> R<Atom> {
                    let d = lw.dec(fx, ia)?;
                    fx.b.cats(vec![Atom::bytes(b"index "), d, Atom::bytes(b" out of range")])
                };
                self.guard(fx, neg, |lw, fx| msg(lw, fx, ia))?;
                let n = fx.b.p("list.len", vec![et.clone()], vec![Atom::V(var)])?;
                let ok = fx.b.int_cmp("int.lt", i.a.clone(), n)?;
                fx.b.open();
                let th = fx.b.close();
                fx.b.open();
                let m = msg(self, fx, i.a.clone())?;
                self.trap(fx, m)?;
                let el = fx.b.close();
                fx.b.push(Stmt::If(ok, th, el));
                let a = self.conv(fx, v.a, &v.t, &ert, false)?;
                let l = self.take_var(fx, var);
                let l2 = fx.b.p("list.set", vec![et], vec![l, i.a, a])?;
                fx.b.set(var, l2);
                Ok(())
            }
            _ => no("an assignment to this place"),
        }
    }

    fn if_stmt(&mut self, fx: &mut Fx, arms: &[(Expr, Vec<S>)], els: Option<&[S]>, tail: bool) -> R<Option<Val>> {
        let tt = if tail { self.tail_type_if(arms, els) } else { None };
        let res = match &tt {
            Some(t) => {
                let ct = self.ct(t)?;
                Some(self.result_var(fx, &ct)?)
            }
            None => None,
        };
        self.if_arms(fx, arms, els, res, tt.as_ref())?;
        Ok(match (res, tt) {
            (Some(r), Some(t)) => Some(Val { a: fx.b.let_(Atom::V(r)), t }),
            _ => None,
        })
    }

    fn tail_type_if(&self, arms: &[(Expr, Vec<S>)], els: Option<&[S]>) -> Option<RType> {
        let mut t = self.tail_type(els?)?;
        for (_, b) in arms {
            t = t.join(&self.tail_type(b)?);
        }
        Some(t)
    }

    fn if_arms(&mut self, fx: &mut Fx, arms: &[(Expr, Vec<S>)], els: Option<&[S]>, res: Option<u32>, tt: Option<&RType>) -> R<()> {
        let Some(((c, b), rest)) = arms.split_first() else {
            if let Some(e) = els {
                self.branch(fx, e, res, tt)?;
            }
            return Ok(());
        };
        let cv = self.expr(fx, c)?;
        fx.b.open();
        self.branch(fx, b, res, tt)?;
        let th = fx.b.close();
        fx.b.open();
        self.if_arms(fx, rest, els, res, tt)?;
        let el = fx.b.close();
        fx.b.push(Stmt::If(cv.a, th, el));
        Ok(())
    }

    /// One branch of an `if` or `match`: its block, its value stored in the
    /// result variable when there is one.
    fn branch(&mut self, fx: &mut Fx, b: &[S], res: Option<u32>, tt: Option<&RType>) -> R<()> {
        let v = self.block(fx, b, res.is_some())?;
        if let (Some(r), Some(t)) = (res, tt)
            && fx.b.falls()
        {
            let Some(v) = v else { return no("a branch without a value where the others have one") };
            let a = self.conv(fx, v.a, &v.t, t, false)?;
            self.store_var(fx, r, a)?;
        }
        Ok(())
    }

    /// interp `run_loop`: at most `bound` iterations (the loop limit when
    /// none is given); the condition is tested before each.
    fn run_loop(&mut self, fx: &mut Fx, bound: Option<&Expr>, cond: Option<&Expr>, body: &[S]) -> R<()> {
        let bnd = match bound {
            Some(b) => {
                let v = self.expr(fx, b)?;
                match v.t {
                    RType::Int(_) => v.a,
                    _ => lit(self.loop_limit),
                }
            }
            None => lit(self.loop_limit),
        };
        let n = fx.b.var(Ty::Int(0, INT_MAX), lit(0));
        let l = fx.open_nest();
        if let Some(c) = cond {
            let cv = self.expr(fx, c)?;
            fx.b.open();
            let th = fx.b.close();
            fx.b.open();
            fx.break_to(l)?;
            let el = fx.b.close();
            fx.b.push(Stmt::If(cv.a, th, el));
        }
        let nv = fx.b.let_(Atom::V(n));
        let n1 = fx.b.p("int.sadd", vec![Ty::Int(0, INT_MAX)], vec![nv, lit(1)])?;
        fx.b.set(n, n1.clone());
        let over = fx.b.int_cmp("int.lt", bnd, n1)?;
        self.guard(fx, over, |_, _| Ok(Atom::bytes(b"loop bound")))?;
        let c = fx.open_nest();
        fx.loops.push(LoopCx { brk: l, cont: c });
        self.block(fx, body, false)?;
        fx.loops.pop();
        let inner = fx.close_nest();
        fx.b.push(Stmt::Block(inner));
        let lb = fx.close_nest();
        fx.b.push(Stmt::Loop(lb));
        Ok(())
    }

    /// `for p in e`: a range (with the loop limit test of interp
    /// `iter_items`) or a list, item by item.
    fn for_stmt(&mut self, fx: &mut Fx, p: &Pat, it: &Expr, body: &[S]) -> R<()> {
        if let Expr::Bin(op, a, b, _) = it
            && (op == ".." || op == "..=")
        {
            fx.b.tick(1); // the range expression
            let x = self.expr(fx, a)?;
            let y = self.expr(fx, b)?;
            self.charge_k(fx, 1)?;
            let (RType::Int(_), RType::Int(_)) = (&x.t, &y.t) else { return no("a range of non-integers") };
            let item_t = match self.node_ty(it) {
                RType::Range(i) => i,
                _ => return no("a range of unknown type"),
            };
            let hi = if op == "..=" {
                let yt = fx.b.ty(&y.a);
                let (_, yh) = yt.bounds().unwrap();
                if yh >= INT_MAX {
                    return no("an inclusive range up to 2^64 - 1");
                }
                let (yl, yh) = yt.bounds().unwrap();
                fx.b.add_in(y.a, lit(1), Ty::Int(yl + 1, yh + 1))?
            } else {
                y.a
            };
            // more than ten times the loop limit of items traps before the first
            let le = fx.b.int_cmp("int.le", hi.clone(), x.a.clone())?;
            fx.b.open();
            let th = fx.b.close();
            fx.b.open();
            let lim = fx.b.p("int.sadd", vec![wide()], vec![x.a.clone(), lit(self.loop_limit * 10)])?;
            let over = fx.b.int_cmp("int.lt", lim, hi.clone())?;
            self.guard(fx, over, |_, _| Ok(Atom::bytes(b"loop bound")))?;
            let el = fx.b.close();
            fx.b.push(Stmt::If(le, th, el));
            let it_t = hull(&fx.b.ty(&x.a), &fx.b.ty(&hi));
            let i = fx.b.var(it_t.clone(), x.a);
            let l = fx.open_nest();
            let iv = fx.b.let_(Atom::V(i));
            let more = fx.b.int_cmp("int.lt", iv.clone(), hi.clone())?;
            fx.b.open();
            let th = fx.b.close();
            fx.b.open();
            fx.break_to(l)?;
            let el = fx.b.close();
            fx.b.push(Stmt::If(more, th, el));
            let c = fx.open_nest();
            fx.loops.push(LoopCx { brk: l, cont: c });
            fx.push_scope();
            let item_rt = RType::Int(IntT { tags: item_t.tags, lo: it_t.bounds().unwrap().0, hi: it_t.bounds().unwrap().1 });
            self.for_bind(fx, p, Val { a: iv.clone(), t: item_rt })?;
            self.block(fx, body, false)?;
            fx.pop_scope()?;
            fx.loops.pop();
            let inner = fx.close_nest();
            fx.b.push(Stmt::Block(inner));
            let i1 = fx.b.add_in(iv, lit(1), it_t)?;
            fx.b.set(i, i1);
            let lb = fx.close_nest();
            fx.b.push(Stmt::Loop(lb));
            return Ok(());
        }
        let itv = self.expr(fx, it)?;
        if let RType::Gen(key, item) = itv.t.clone() {
            return self.for_gen(fx, p, itv.a, &key, &item, body);
        }
        let item = match &itv.t {
            RType::List(e) | RType::Set(e) => (**e).clone(),
            // a map's items are its (key, value) pairs in key order
            RType::Map(k, v) => RType::Tuple(vec![(**k).clone(), (**v).clone()]),
            o => return no(format!("for over {}", o.show())),
        };
        let lt = fx.b.ty(&itv.a);
        let Ty::List(et) = &lt else { unreachable!() };
        let et = (**et).clone();
        let lv = fx.b.var(lt.clone(), itv.a);
        fx.own(Atom::V(lv));
        let n = fx.b.p("list.len", vec![et.clone()], vec![Atom::V(lv)])?;
        let i = fx.b.var(Ty::u64(), lit(0));
        let l = fx.open_nest();
        let iv = fx.b.let_(Atom::V(i));
        let more = fx.b.int_cmp("int.lt", iv.clone(), n)?;
        fx.b.open();
        let th = fx.b.close();
        fx.b.open();
        fx.break_to(l)?;
        let el = fx.b.close();
        fx.b.push(Stmt::If(more, th, el));
        let x = fx.b.list_at(Atom::V(lv), iv.clone())?;
        let c = fx.open_nest();
        fx.loops.push(LoopCx { brk: l, cont: c });
        fx.push_scope();
        self.for_bind(fx, p, Val { a: x, t: item })?;
        self.block(fx, body, false)?;
        fx.pop_scope()?;
        fx.loops.pop();
        let inner = fx.close_nest();
        fx.b.push(Stmt::Block(inner));
        let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
        fx.b.set(i, i1);
        let lb = fx.close_nest();
        fx.b.push(Stmt::Loop(lb));
        Ok(())
    }

    /// `par` (C.4, T16, interp `par_exec`): the branches in source order;
    /// their results are those of the sequential run, which is what the
    /// reference resolves them to. Under a budget each branch runs on a
    /// copy of the budgets as they are at the fork, and a branch that
    /// takes the branches so far past a budget traps after it: so each
    /// branch is charged as if the ones before it had charged nothing, and
    /// the total is checked after it.
    fn par(&mut self, fx: &mut Fx, body: &[S]) -> R<()> {
        self.charge_k(fx, crate::conc::C_FORK * body.len() as i128)?;
        let fork = if self.costs {
            let (sp, dl) = rt::cost_globals(&mut self.f);
            Some((sp, dl, fx.b.let_global(sp, Ty::u64()), fx.b.let_global(dl, Ty::u64())))
        } else {
            None
        };
        for (k, b) in body.iter().enumerate() {
            let S::Arena(_, _, x, _) = b else { return no("a par branch") };
            if let Some((sp, dl, s0, d0)) = &fork
                && k > 0
            {
                let s = fx.b.let_global(*sp, Ty::u64());
                let cum = fx.b.sub_in(s, s0.clone(), Ty::u64())?;
                let d = fx.b.p("int.sadd", vec![Ty::u64()], vec![d0.clone(), cum])?;
                fx.b.setg(*dl, d);
            }
            self.block(fx, x, false)?;
            if !fx.b.falls() {
                break;
            }
            if let Some((sp, _, _, d0)) = &fork {
                let s = fx.b.let_global(*sp, Ty::u64());
                let over = fx.b.int_cmp("int.lt", d0.clone(), s)?;
                self.guard(fx, over, |_, _| Ok(Atom::bytes(b"budget")))?;
            }
        }
        if let Some((_, dl, _, d0)) = fork
            && fx.b.falls()
        {
            fx.b.setg(dl, d0);
        }
        Ok(())
    }

    /// `with budget n Block else [as e] Else` (C.5, interp `exec_marker`):
    /// the limit, then the block in a function of its own, called under
    /// `catch` with the budget in force, which takes the variables it
    /// reads and gives back the values of the `var`s, which are written
    /// back when it completes. When its budget does not allow a charge ("budget"),
    /// nothing is written back, so the else sees the state of block entry,
    /// with the limit and the units the block spent; any other trap
    /// is raised again.
    fn budget(&mut self, fx: &mut Fx, body: &[S]) -> R<()> {
        let Some(S::Expr(ne)) = body.first() else { return no("a budget without its limit") };
        let (rest, eb) = typed::budget_parts(body);
        let nv = self.expr(fx, ne)?;
        if !matches!(nv.t, RType::Int(_)) {
            return no("a budget whose limit is not an integer");
        }
        // the block runs elsewhere: what leaves it, or yields, is not covered
        let mut bad = None;
        fn exits(b: &[S], in_loop: bool, bad: &mut Option<&'static str>) {
            for s in b {
                match s {
                    S::Return(..) => *bad = Some("return"),
                    S::Fail(_) => *bad = Some("fail"),
                    S::Yield(..) => *bad = Some("yield"),
                    S::Break(_) | S::Continue(_) if !in_loop => *bad = Some("break or continue"),
                    S::While(_, _, x) | S::Loop(_, x) | S::For(_, _, x, _) => exits(x, true, bad),
                    S::If(arms, els) => {
                        for (_, x) in arms {
                            exits(x, in_loop, bad);
                        }
                        if let Some(x) = els {
                            exits(x, in_loop, bad);
                        }
                    }
                    S::Match(_, arms, _) => {
                        for a in arms {
                            exits(&a.body, in_loop, bad);
                        }
                    }
                    S::Arena(_, _, x, _) | S::Device(_, _, x, _) | S::Comptime(x) => exits(x, in_loop, bad),
                    _ => {}
                }
            }
        }
        exits(rest, false, &mut bad);
        let mut tries = false;
        crate::check::walk_stmts(rest, &mut |s| {
            let mut look = |e: &Expr| {
                crate::check::walk_expr(e, &mut |x| {
                    if matches!(x, Expr::Try(..)) {
                        tries = true;
                    }
                })
            };
            match s {
                S::Expr(e) | S::Let(_, _, e, _) | S::Var(_, _, e, _) | S::Return(Some(e), _) => look(e),
                S::Assign(_, p, e) => {
                    look(p);
                    look(e);
                }
                _ => {}
            }
        });
        if let Some(b) = bad {
            return no(format!("{b} in the block of a budget"));
        }
        if tries {
            return no("try in the block of a budget");
        }
        // the locals of this function the block names, in order
        let mut ids: Vec<u32> = vec![];
        crate::check::walk_stmts(rest, &mut |s| {
            let mut look = |e: &Expr| {
                crate::check::walk_expr(e, &mut |x| {
                    if let Some(Binding::Local(id)) = self.ty.node(x).and_then(|n| n.bind.clone())
                        && fx.bound.contains_key(&id)
                        && !ids.contains(&id)
                    {
                        ids.push(id);
                    }
                })
            };
            match s {
                S::Expr(e) | S::Let(_, _, e, _) | S::Var(_, _, e, _) | S::Return(Some(e), _) | S::Emit(e) | S::Fail(e) | S::Check(e) => look(e),
                S::Assign(_, p, e) => {
                    look(p);
                    look(e);
                }
                S::If(arms, _) => {
                    for (c, _) in arms {
                        look(c);
                    }
                }
                S::While(c, b, _) => {
                    look(c);
                    if let Some(b) = b {
                        look(b);
                    }
                }
                S::Loop(Some(b), _) => look(b),
                S::For(_, it, _, _) => look(it),
                S::Match(e, arms, _) => {
                    look(e);
                    for a in arms {
                        if let Some(g) = &a.guard {
                            look(g);
                        }
                    }
                }
                S::Arena(_, Some(b), _, _) | S::Device(_, Some(b), _, _) => look(b),
                _ => {}
            }
        });
        ids.sort();
        // the block's function: the values of the locals, then the vars' values out
        let mut ps = vec![];
        let mut args = vec![];
        let mut vars = vec![];
        for id in &ids {
            match fx.bound.get(id).cloned() {
                Some(Bound::Val(a, rt)) => {
                    ps.push(fx.b.ty(&a));
                    args.push(fx.b.copy(a)?);
                    let _ = rt;
                }
                Some(Bound::Var(v, rt)) => {
                    ps.push(fx.b.local_ty(v));
                    args.push(fx.b.peek(v)?);
                    vars.push((*id, v, rt));
                }
                None => unreachable!(),
            }
        }
        let ret = Ty::Tup(vars.iter().map(|(_, v, _)| fx.b.local_ty(*v)).collect());
        let id = self.f.declare(&format!("budget block {}", self.f.fns.len()), ps.clone(), ret.clone());
        {
            let mut fx2 = self.fx(ps, ret.clone(), Ret::None, FailCx::Halt, true);
            let mut inner_vars = vec![];
            for (k, lid) in ids.iter().enumerate() {
                let a = fx2.b.arg(k);
                match fx.bound.get(lid).cloned() {
                    Some(Bound::Val(_, rt)) => {
                        fx2.bound.insert(*lid, Bound::Val(a.clone(), rt));
                        fx2.own(a);
                    }
                    Some(Bound::Var(_, rt)) => {
                        let t = fx2.b.ty(&a);
                        let v = fx2.b.var(t, a);
                        fx2.bound.insert(*lid, Bound::Var(v, rt));
                        fx2.own(Atom::V(v));
                        inner_vars.push(v);
                    }
                    None => unreachable!(),
                }
            }
            self.block(&mut fx2, rest, false)?;
            if fx2.b.falls() {
                let mut outs = vec![];
                for v in inner_vars {
                    outs.push(self.read_var(&mut fx2, v)?);
                }
                let r = fx2.b.p("tup.mk", vec![ret.clone()], outs)?;
                fx2.drops_to(None)?;
                fx2.b.ret(r);
            }
            self.f.define(id, fx2.b.finish());
        }
        // the budget in force: spent + units may reach start + n
        let (sp, dl) = rt::cost_globals(&mut self.f);
        let start = fx.b.let_global(sp, Ty::u64());
        let d0 = fx.b.let_global(dl, Ty::u64());
        let n = self.conv(fx, nv.a, &nv.t, &RType::Int(IntT::named("u64")), false)?;
        let lim = fx.b.p("int.sadd", vec![Ty::u64()], vec![start.clone(), n.clone()])?;
        let d1 = fx.b.p("int.min", vec![Ty::u64()], vec![d0.clone(), lim])?;
        fx.b.setg(dl, d1);
        let r = fx.b.catch(sym(id), ret.clone(), args);
        let after = fx.b.let_global(sp, Ty::u64());
        let used = fx.b.sub_in(after, start, Ty::u64())?;
        fx.b.setg(dl, d0);
        let rt_ = fx.b.ty(&r);
        let vars2 = vars.clone();
        self.switch(fx, &r, &rt_, 2, |lw, fx, k| {
            let x = fx.b.p("sum.get", vec![rt_.clone()], vec![r.clone(), lit(k as i128)])?;
            if k == 0 {
                // the block completed: its vars' values are written back
                let parts = fx.b.prim("tup.open", vec![ret.clone()], vec![x])?;
                for ((_, v, _), a) in vars2.iter().zip(parts) {
                    lw.store_var(fx, *v, a)?;
                }
                return Ok(());
            }
            // a trap: "budget" runs the else, when there is one
            let Some((ifs, eb)) = eb else {
                fx.b.halt(x);
                return Ok(());
            };
            let ht = Ty::halt();
            let tag = fx.b.p("sum.tag", vec![ht.clone()], vec![x.clone()])?;
            let is_trap = fx.b.int_cmp("int.eq", tag, lit(0))?;
            let is_budget = fx.b.var(Ty::Bool, Atom::bool(false));
            fx.b.open();
            {
                let xc = fx.b.copy(x.clone())?;
                let m = fx.b.p("sum.get", vec![ht.clone()], vec![xc, lit(0)])?;
                let e = fx.b.p("bytes.eq", vec![], vec![m, Atom::bytes(b"budget")])?;
                fx.b.set(is_budget, e);
            }
            let th = fx.b.close();
            fx.b.push(Stmt::If(is_trap, th, vec![]));
            let ib = fx.b.let_(Atom::V(is_budget));
            fx.b.open();
            {
                fx.b.drop(x.clone())?;
                // the hidden limit and spent units, then the else
                fx.push_scope();
                let (Some(bn), Some(bu)) = (lw.ty.hidden_binds.get(&addr(ifs)).copied(), lw.ty.hidden_binds.get(&(addr(ifs) + 1)).copied()) else {
                    return no("the hidden names of a budget's else");
                };
                let u = RType::Int(IntT::named("u64"));
                fx.bound.insert(bn, Bound::Val(n.clone(), u.clone()));
                fx.bound.insert(bu, Bound::Val(used.clone(), u));
                lw.block(fx, eb, false)?;
                fx.pop_scope()?;
            }
            let th = fx.b.close();
            fx.b.open();
            fx.b.halt(x);
            let el = fx.b.close();
            fx.b.push(Stmt::If(ib, th, el));
            Ok(())
        })?;
        Ok(())
    }

    /// `for p in it par` (C.5.2, interp `forpar_exec`). The variables the
    /// body updates with `+=`, `*=`, `&=`, `|=` or `^=` and does not declare
    /// are its reduction targets: each item's run starts them from the
    /// operator's identity, their values after the run are collected, and
    /// after the last item each target is its value before the loop combined
    /// with the reduction of the collected values by the tree of C.8. The
    /// items run in order, as the reference runs them.
    fn for_par(&mut self, fx: &mut Fx, body: &[S]) -> R<()> {
        let Some(S::For(p, it, b, _)) = body.first() else { return no("a for par without its loop") };
        let mut jumps = false;
        crate::check::walk_stmts(b, &mut |s| {
            if matches!(s, S::Break(_) | S::Continue(_) | S::Return(..)) {
                jumps = true;
            }
        });
        if jumps {
            return no("break, continue or return in the body of a for par");
        }
        let mut ops: Vec<(String, String, usize)> = vec![];
        crate::check::walk_stmts(b, &mut |s| {
            if let S::Assign(op, place @ Expr::Name(n, _), _) = s
                && ["+=", "*=", "&=", "|=", "^="].contains(&op.as_str())
                && !ops.iter().any(|(m, _, _)| m == n)
            {
                ops.push((n.clone(), op.clone(), addr(place)));
            }
        });
        let local = crate::conc::declared_in(b);
        ops.retain(|(n, _, _)| !local.contains(n));
        // each target: its var, the type of its values, the operator and the identity
        let mut targets: Vec<(u32, RType, String, Atom)> = vec![];
        for (n, op, pa) in &ops {
            let v = match self.ty.nodes.get(pa).and_then(|x| x.bind.clone()) {
                Some(Binding::Local(id)) => match fx.bound.get(&id) {
                    Some(Bound::Var(v, _)) => *v,
                    _ => return no(format!("the reduction target {n}, which is not a var")),
                },
                _ => return no(format!("the reduction target {n}")),
            };
            let rt = self.var_rtype(fx, v).ok_or_else(|| format!("not covered by the lowering: the type of {n}"))?;
            let sym = op[..op.len() - 1].to_string();
            let id = self.identity(&sym, &rt)?;
            targets.push((v, rt, sym, id));
        }
        // the items: a range or a list, at most the loop limit of them
        let (items, item_rt, range) = if let Expr::Bin(op, a, bx, _) = it
            && (op == ".." || op == "..=")
        {
            fx.b.tick(1);
            let x = self.expr(fx, a)?;
            let y = self.expr(fx, bx)?;
            self.charge_k(fx, 1)?; // the range
            let (RType::Int(_), RType::Int(_)) = (&x.t, &y.t) else { return no("a range of non-integers") };
            let item_t = match self.node_ty(it) {
                RType::Range(i) => i,
                _ => return no("a range of unknown type"),
            };
            let hi = if op == "..=" {
                let yt = fx.b.ty(&y.a);
                let (yl, yh) = yt.bounds().unwrap();
                if yh >= INT_MAX {
                    return no("an inclusive range up to 2^64 - 1");
                }
                fx.b.add_in(y.a, lit(1), Ty::Int(yl + 1, yh + 1))?
            } else {
                y.a
            };
            let le = fx.b.int_cmp("int.le", hi.clone(), x.a.clone())?;
            fx.b.open();
            let th = fx.b.close();
            fx.b.open();
            let lim = fx.b.p("int.sadd", vec![wide()], vec![x.a.clone(), lit(self.loop_limit)])?;
            let over = fx.b.int_cmp("int.lt", lim, hi.clone())?;
            self.guard(fx, over, |_, _| Ok(Atom::bytes(b"loop bound")))?;
            let el = fx.b.close();
            fx.b.push(Stmt::If(le, th, el));
            let it_t = hull(&fx.b.ty(&x.a), &fx.b.ty(&hi));
            let (lo_b, hi_b) = it_t.bounds().unwrap();
            (None, RType::Int(IntT { tags: item_t.tags, lo: lo_b, hi: hi_b }), Some((x.a, hi, it_t)))
        } else {
            let itv = self.expr(fx, it)?;
            let item = match &itv.t {
                RType::List(e) | RType::Set(e) => (**e).clone(),
                o => return no(format!("for par over {}", o.show())),
            };
            let lt = fx.b.ty(&itv.a);
            let Ty::List(et) = &lt else { unreachable!() };
            let n = fx.b.p("list.len", vec![(**et).clone()], vec![itv.a.clone()])?;
            let over = fx.b.int_cmp("int.lt", lit(self.loop_limit), n)?;
            self.guard(fx, over, |_, _| Ok(Atom::bytes(b"loop bound")))?;
            let lv = fx.b.var(lt.clone(), itv.a);
            fx.own(Atom::V(lv));
            (Some((lv, (**et).clone())), item, None)
        };
        // the values before the loop, and the lists of contributions
        let mut befores = vec![];
        let mut contribs = vec![];
        for (v, rt, _, _) in &targets {
            let x = self.read_var(fx, *v)?;
            befores.push(Val { a: x, t: rt.clone() });
            let et = self.ct(rt)?;
            let l = fx.b.p("list.new", vec![et.clone()], vec![])?;
            contribs.push((fx.b.var(Ty::list(et.clone()), l), et));
        }
        // the loop over the items
        let i = match &range {
            Some((x, _, it_t)) => fx.b.var(it_t.clone(), x.clone()),
            None => fx.b.var(Ty::u64(), lit(0)),
        };
        let l = fx.open_nest();
        let iv = fx.b.let_(Atom::V(i));
        let n = match (&range, &items) {
            (Some((_, hi, _)), _) => hi.clone(),
            (None, Some((lv, et))) => fx.b.p("list.len", vec![et.clone()], vec![Atom::V(*lv)])?,
            _ => unreachable!(),
        };
        let more = fx.b.int_cmp("int.lt", iv.clone(), n)?;
        fx.b.open();
        let th = fx.b.close();
        fx.b.open();
        fx.break_to(l)?;
        let el = fx.b.close();
        fx.b.push(Stmt::If(more, th, el));
        let item = match &items {
            Some((lv, _)) => fx.b.list_at(Atom::V(*lv), iv.clone())?,
            None => iv.clone(),
        };
        for (v, rt, _, id) in &targets {
            let st = self.var_rtype(fx, *v).unwrap_or(rt.clone());
            let a = self.conv(fx, id.clone(), rt, &st, false)?;
            self.store_var(fx, *v, a)?;
        }
        fx.push_scope();
        self.for_bind(fx, p, Val { a: item, t: item_rt })?;
        self.block(fx, b, false)?;
        fx.pop_scope()?;
        for ((v, _, _, _), (cl, et)) in targets.iter().zip(&contribs) {
            let x = self.read_var(fx, *v)?;
            let c = fx.b.let_(Atom::V(*cl));
            let c = fx.b.p("list.push", vec![et.clone()], vec![c, x])?;
            fx.b.set(*cl, c);
        }
        let it_t = fx.b.local_ty(i);
        let i1 = fx.b.add_in(iv, lit(1), it_t)?;
        fx.b.set(i, i1);
        let lb = fx.close_nest();
        fx.b.push(Stmt::Loop(lb));
        // each target: its value before, combined with the tree of its contributions
        for (((v, rt, sym, id), (cl, _)), before) in targets.iter().zip(&contribs).zip(befores) {
            let tree = self.tree_fn(sym, rt)?;
            let c = fx.b.let_(Atom::V(*cl));
            let t = self.call_rt(fx, tree, vec![c, id.clone()]);
            let r = self.combine(fx, sym, before, Val { a: t, t: rt.clone() })?;
            self.store_var(fx, *v, r)?;
        }
        Ok(())
    }

    /// interp `ident`: the identity of a reduction operator at a type.
    fn identity(&mut self, sym: &str, rt: &RType) -> R<Atom> {
        Ok(match (sym, rt) {
            ("+" | "|" | "^", RType::Int(_)) => lit(0),
            ("*", RType::Int(_)) => lit(1),
            ("&", RType::Int(i)) => {
                let Some(n) = i.tags.single() else { return no("an &= reduction on an integer whose type name differs at run time") };
                let (lo, hi) = typed::tag_range(n);
                lit(if lo == 0 { hi } else { -1 })
            }
            ("+", RType::F64) => Atom::K(Ty::F64, Lit::F64((-0.0f64).to_bits())),
            ("*", RType::F64) => Atom::K(Ty::F64, Lit::F64(1.0f64.to_bits())),
            ("+", RType::F32) => Atom::K(Ty::F32, Lit::F32((-0.0f64).to_bits())),
            ("*", RType::F32) => Atom::K(Ty::F32, Lit::F32(1.0f64.to_bits())),
            ("&", RType::Bool) => Atom::bool(true),
            ("|" | "^", RType::Bool) => Atom::bool(false),
            (o, t) => return no(format!("a {o}= reduction on {}", t.show())),
        })
    }

    /// interp `combine`: `a op b`, an integer result checked against the
    /// type name of a.
    fn combine(&mut self, fx: &mut Fx, sym: &str, a: Val, b: Val) -> R<Atom> {
        self.charge_k(fx, op_cost(sym, &a.t))?;
        match (&a.t, sym) {
            (RType::Bool, "&") => fx.b.p("bool.and", vec![], vec![a.a, b.a]),
            (RType::Bool, "|") => fx.b.p("bool.or", vec![], vec![a.a, b.a]),
            (RType::Bool, "^") => {
                let e = fx.b.p("bool.eq", vec![], vec![a.a, b.a])?;
                fx.b.not(e)
            }
            (RType::Int(i), _) => {
                let tags = i.tags;
                let RType::Int(j) = &b.t else { return no("a reduction of mixed types") };
                let res = tags.join_op(j.tags);
                let rt = a.t.clone();
                let x = self.arith(fx, sym, a, b, rt)?;
                if tags == typed::Tags::lit() {
                    return Ok(x.a);
                }
                if tags.single().is_none() {
                    return no("a reduction on an integer whose type name differs at run time");
                }
                let from = RType::Int(IntT { tags: res, lo: INT_MIN, hi: INT_MAX });
                self.conv(fx, x.a, &from, &RType::Int(IntT::of_tags(tags)), true)
            }
            _ => {
                let rt = a.t.clone();
                Ok(self.arith(fx, sym, a, b, rt)?.a)
            }
        }
    }

    /// The tree of C.8 over a list of contributions (interp `tree`): the
    /// left part of m values is the largest power of two below m. It is
    /// computed as the values arrive, keeping a stack of complete subtrees
    /// whose sizes are powers of two, then folding the stack from the
    /// right, which combines in the order the recursive definition does.
    fn tree_fn(&mut self, sym: &str, rt: &RType) -> R<usize> {
        let t = self.ct(rt)?;
        let lt = Ty::list(t.clone());
        let (sym2, rt2) = (sym.to_string(), rt.clone());
        self.generated(&format!("tree{sym}"), rt, vec![lt.clone(), t.clone()], t.clone(), move |lw, fx| {
            let (c, id) = (fx.b.arg(0), fx.b.arg(1));
            let val = |a: Atom| Val { a, t: rt2.clone() };
            let n = fx.b.p("list.len", vec![t.clone()], vec![c.clone()])?;
            let z = fx.b.int_cmp("int.eq", n.clone(), lit(0))?;
            fx.b.open();
            fx.b.drop(c.clone())?;
            fx.b.ret(id.clone());
            let th = fx.b.close();
            fx.b.push(Stmt::If(z, th, vec![]));
            let s0 = fx.b.p("list.new", vec![t.clone()], vec![])?;
            let st = fx.b.var(lt.clone(), s0);
            let z0 = fx.b.p("list.new", vec![Ty::u64()], vec![])?;
            let sz = fx.b.var(Ty::list(Ty::u64()), z0);
            let i = fx.b.var(Ty::u64(), lit(0));
            // for each value: merge it with the subtrees of its size
            fx.b.open();
            {
                let iv = fx.b.let_(Atom::V(i));
                let more = fx.b.int_cmp("int.lt", iv.clone(), n.clone())?;
                fx.b.open();
                let th = fx.b.close();
                fx.b.open();
                fx.b.brk(0);
                let el = fx.b.close();
                fx.b.push(Stmt::If(more, th, el));
                let x = fx.b.list_at(c.clone(), iv.clone())?;
                let cur = fx.b.var(t.clone(), x);
                let csz = fx.b.var(Ty::u64(), lit(1));
                fx.b.open();
                {
                    let sl = fx.b.p("list.len", vec![Ty::u64()], vec![Atom::V(sz)])?;
                    let e = fx.b.int_cmp("int.eq", sl.clone(), lit(0))?;
                    fx.b.open();
                    fx.b.brk(0);
                    let th = fx.b.close();
                    fx.b.push(Stmt::If(e, th, vec![]));
                    let sl1 = fx.b.sub_in(sl, lit(1), Ty::Int(0, INT_MAX))?;
                    let top = fx.b.list_at(Atom::V(sz), sl1)?;
                    let cs = fx.b.let_(Atom::V(csz));
                    let same = fx.b.int_cmp("int.eq", top, cs.clone())?;
                    fx.b.open();
                    let th = fx.b.close();
                    fx.b.open();
                    fx.b.brk(0);
                    let el = fx.b.close();
                    fx.b.push(Stmt::If(same, th, el));
                    let szv = fx.b.let_(Atom::V(sz));
                    let r = fx.b.prim("list.pop", vec![Ty::u64()], vec![szv])?;
                    fx.b.set(sz, r[0].clone());
                    let stn = fx.b.p("list.len", vec![t.clone()], vec![Atom::V(st)])?;
                    let e2 = fx.b.int_cmp("int.eq", stn, lit(0))?;
                    fx.b.guard_halt(e2, |fb| fb.trap_value(Atom::bytes(b"internal: tree stack")))?;
                    let stv = fx.b.let_(Atom::V(st));
                    let r = fx.b.prim("list.pop", vec![t.clone()], vec![stv])?;
                    fx.b.set(st, r[0].clone());
                    let curv = fx.b.let_(Atom::V(cur));
                    let m = lw.combine(fx, &sym2, val(r[1].clone()), val(curv))?;
                    fx.b.set(cur, m);
                    let c2 = fx.b.mul_in(cs, lit(2), Ty::u64())?;
                    fx.b.set(csz, c2);
                }
                let inner = fx.b.close();
                fx.b.push(Stmt::Loop(inner));
                let curv = fx.b.let_(Atom::V(cur));
                let stv = fx.b.let_(Atom::V(st));
                let stv = fx.b.p("list.push", vec![t.clone()], vec![stv, curv])?;
                fx.b.set(st, stv);
                let cs = fx.b.let_(Atom::V(csz));
                let szv = fx.b.let_(Atom::V(sz));
                let szv = fx.b.p("list.push", vec![Ty::u64()], vec![szv, cs])?;
                fx.b.set(sz, szv);
                let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
                fx.b.set(i, i1);
            }
            let body = fx.b.close();
            fx.b.push(Stmt::Loop(body));
            // fold the stack from the right
            let stn = fx.b.p("list.len", vec![t.clone()], vec![Atom::V(st)])?;
            let e = fx.b.int_cmp("int.eq", stn, lit(0))?;
            fx.b.guard_halt(e, |fb| fb.trap_value(Atom::bytes(b"internal: tree stack")))?;
            let stv = fx.b.let_(Atom::V(st));
            let r = fx.b.prim("list.pop", vec![t.clone()], vec![stv])?;
            fx.b.set(st, r[0].clone());
            let acc = fx.b.var(t.clone(), r[1].clone());
            fx.b.open();
            {
                let stn = fx.b.p("list.len", vec![t.clone()], vec![Atom::V(st)])?;
                let e = fx.b.int_cmp("int.eq", stn, lit(0))?;
                fx.b.open();
                fx.b.brk(0);
                let th = fx.b.close();
                fx.b.push(Stmt::If(e, th, vec![]));
                let stv = fx.b.let_(Atom::V(st));
                let r = fx.b.prim("list.pop", vec![t.clone()], vec![stv])?;
                fx.b.set(st, r[0].clone());
                let a = fx.b.let_(Atom::V(acc));
                let m = lw.combine(fx, &sym2, val(r[1].clone()), val(a))?;
                fx.b.set(acc, m);
            }
            let body = fx.b.close();
            fx.b.push(Stmt::Loop(body));
            for v in [st, sz] {
                let x = fx.b.let_(Atom::V(v));
                fx.b.drop(x)?;
            }
            fx.b.drop(c)?;
            fx.b.drop(id)?;
            let a = fx.b.let_(Atom::V(acc));
            fx.b.ret(a);
            Ok(())
        })
    }

    /// The pattern of a `for`: bound to each item (interp ignores a mismatch).
    fn for_bind(&mut self, fx: &mut Fx, p: &Pat, v: Val) -> R<()> {
        match p {
            Pat::Wild => fx.b.drop(v.a),
            Pat::Path(path) if path.len() == 1 && !self.t.is_tested_name(&path[0]) => {
                let Some(id) = self.ty.pat_binds.get(&addr(p)).copied() else { return no("a for binding") };
                let a = fx.b.let_(v.a);
                fx.bound.insert(id, Bound::Val(a.clone(), v.t));
                fx.own(a);
                Ok(())
            }
            Pat::Tuple(_) => self.let_pat(fx, p, v),
            _ => no("this for pattern"),
        }
    }

    fn match_stmt(&mut self, fx: &mut Fx, e: &Expr, arms: &[ast::Arm], tail: bool) -> R<Option<Val>> {
        let v = self.expr(fx, e)?;
        let tt = if tail {
            let mut t = Some(RType::Never);
            for a in arms {
                t = match (t, self.tail_type(&a.body)) {
                    (Some(x), Some(y)) => Some(x.join(&y)),
                    _ => None,
                };
            }
            t
        } else {
            None
        };
        let res = match &tt {
            Some(t) => {
                let ct = self.ct(t)?;
                Some(self.result_var(fx, &ct)?)
            }
            None => None,
        };
        fx.push_scope();
        let va = fx.b.let_(v.a);
        fx.own(va.clone());
        let mb = fx.open_nest();
        for a in arms {
            let ab = fx.open_nest();
            fx.push_scope();
            self.pat(fx, &a.pat, &va, &v.t, ab)?;
            if let Some(g) = &a.guard {
                let gv = self.expr(fx, g)?;
                fx.b.open();
                let th = fx.b.close();
                fx.b.open();
                fx.break_to(ab)?;
                let el = fx.b.close();
                fx.b.push(Stmt::If(gv.a, th, el));
            }
            self.branch(fx, &a.body, res, tt.as_ref())?;
            fx.pop_scope()?;
            if fx.b.falls() {
                fx.break_to(mb)?;
            }
            let body = fx.close_nest();
            fx.b.push(Stmt::Block(body));
        }
        // when an arm always matches, nothing follows the arms
        if fx.b.falls() {
            self.trap(fx, Atom::bytes(b"no match arm"))?;
        }
        let mbody = fx.close_nest();
        fx.b.push(Stmt::Block(mbody));
        fx.pop_scope()?;
        Ok(match (res, tt) {
            (Some(r), Some(t)) => Some(Val { a: fx.b.let_(Atom::V(r)), t }),
            _ => None,
        })
    }

    // ---- patterns

    /// Leave the construct at nesting index `target` when `c` is false.
    fn need(&mut self, fx: &mut Fx, c: Atom, target: u32) -> R<()> {
        fx.b.open();
        let th = fx.b.close();
        fx.b.open();
        fx.break_to(target)?;
        let el = fx.b.close();
        fx.b.push(Stmt::If(c, th, el));
        Ok(())
    }

    /// Test pattern `p` against the value `v` (read, not consumed) of type
    /// `rt` (interp `pmatch`), binding its names in the current scope; on a
    /// mismatch, leave the construct at nesting index `fail`.
    fn pat(&mut self, fx: &mut Fx, p: &Pat, v: &Atom, rt: &RType, fail: u32) -> R<()> {
        match p {
            Pat::Wild => Ok(()),
            Pat::Path(path) => {
                let n = &path[path.len() - 1];
                if path.len() == 1 && !self.t.is_tested_name(n) {
                    let Some(id) = self.ty.pat_binds.get(&addr(p)).copied() else { return no("a pattern binding") };
                    let a = fx.b.copy(v.clone())?;
                    let a = if fx.b.ty(&a).owned() { a } else { fx.b.let_(a) };
                    fx.bound.insert(id, Bound::Val(a.clone(), rt.clone()));
                    fx.own(a);
                    return Ok(());
                }
                self.name_test(fx, n, v, rt, fail)
            }
            Pat::Ctor(path, args) => {
                let n = &path[path.len() - 1];
                if n == "some" && path.len() == 1 {
                    let RType::Opt(inner) = rt else { return self.never(fx, fail) };
                    let t = fx.b.ty(v);
                    let tag = fx.b.p("sum.tag", vec![t.clone()], vec![v.clone()])?;
                    let is = fx.b.int_cmp("int.eq", tag, lit(1))?;
                    self.need(fx, is, fail)?;
                    let c = fx.b.p("copy", vec![t.clone()], vec![v.clone()])?;
                    let x = fx.b.p("sum.get", vec![t], vec![c, lit(1)])?;
                    let x = if fx.b.ty(&x).owned() { x } else { fx.b.let_(x) };
                    fx.own(x.clone());
                    let inner = (**inner).clone();
                    return match args.first() {
                        Some((_, q)) => self.pat(fx, q, &x, &inner, fail),
                        None => Ok(()),
                    };
                }
                let Some(fs) = self.t.ctor_fields(n) else { return self.never(fx, fail) };
                // the record's own fields: a variant's payload, or the struct itself
                let rec = match rt {
                    RType::Named(k) if self.t.kinds.contains_key(k) => {
                        let Some((kind, idx, _)) = self.t.variants.get(n).cloned() else { return self.never(fx, fail) };
                        if &kind != k {
                            return self.never(fx, fail);
                        }
                        let t = fx.b.ty(v);
                        let tag = fx.b.p("sum.tag", vec![t.clone()], vec![v.clone()])?;
                        let is = fx.b.int_cmp("int.eq", tag, lit(idx as i128))?;
                        self.need(fx, is, fail)?;
                        let c = fx.b.p("copy", vec![t.clone()], vec![v.clone()])?;
                        let x = fx.b.p("sum.get", vec![t], vec![c, lit(idx as i128)])?;
                        fx.own(x.clone());
                        x
                    }
                    RType::Named(k) if k == "$event" => {
                        let Some(idx) = self.t.events.iter().position(|e| e == n) else { return self.never(fx, fail) };
                        let t = fx.b.ty(v);
                        let tag = fx.b.p("sum.tag", vec![t.clone()], vec![v.clone()])?;
                        let is = fx.b.int_cmp("int.eq", tag, lit(idx as i128))?;
                        self.need(fx, is, fail)?;
                        let c = fx.b.p("copy", vec![t.clone()], vec![v.clone()])?;
                        let x = fx.b.p("sum.get", vec![t], vec![c, lit(idx as i128)])?;
                        fx.own(x.clone());
                        x
                    }
                    RType::Named(k) if k == n => v.clone(),
                    _ => return self.never(fx, fail),
                };
                let rt_rec = fx.b.ty(&rec);
                for (i, (lb, q)) in args.iter().enumerate() {
                    let k = match lb {
                        Some(l) => fs.iter().position(|(f, _)| f == l),
                        None => (i < fs.len()).then_some(i),
                    };
                    let Some(k) = k else { return self.never(fx, fail) };
                    let x = fx.b.p("tup.get", vec![rt_rec.clone()], vec![rec.clone(), lit(k as i128)])?;
                    let x = if fx.b.ty(&x).owned() {
                        fx.own(x.clone());
                        x
                    } else {
                        x
                    };
                    let frt = self.t.of_ty(&fs[k].1);
                    self.pat(fx, q, &x, &frt, fail)?;
                }
                Ok(())
            }
            Pat::Lit(e) => {
                let x = self.expr(fx, e)?;
                let vv = fx.b.copy(v.clone())?;
                let is = self.eq(fx, x, Val { a: vv, t: rt.clone() })?;
                self.need(fx, is, fail)
            }
            Pat::Range(a, incl, b) => {
                let x = self.expr(fx, a)?;
                let y = self.expr(fx, b)?;
                let (RType::Int(_), RType::Int(_), RType::Int(_)) = (&x.t, &y.t, rt) else { return no("a range pattern on non-integers") };
                let lo = fx.b.int_cmp("int.le", x.a, v.clone())?;
                let hi = fx.b.int_cmp(if *incl { "int.le" } else { "int.lt" }, v.clone(), y.a)?;
                let both = fx.b.p("bool.and", vec![], vec![lo, hi])?;
                self.need(fx, both, fail)
            }
            Pat::Tuple(ps) => {
                let RType::Tuple(ts) = rt else { return self.never(fx, fail) };
                if ts.len() != ps.len() {
                    return self.never(fx, fail);
                }
                let t = fx.b.ty(v);
                for (k, (q, ct)) in ps.iter().zip(ts.clone()).enumerate() {
                    let x = fx.b.p("tup.get", vec![t.clone()], vec![v.clone(), lit(k as i128)])?;
                    if fx.b.ty(&x).owned() {
                        fx.own(x.clone());
                    }
                    self.pat(fx, q, &x, &ct, fail)?;
                }
                Ok(())
            }
            Pat::Alt(ps) => {
                if ps.iter().any(binds) {
                    return no("an alternative pattern that binds names");
                }
                // the first alternative that matches
                let m = fx.b.var(Ty::Bool, Atom::bool(false));
                let outer = fx.open_nest();
                for q in ps {
                    let alt = fx.open_nest();
                    fx.push_scope();
                    self.pat(fx, q, v, rt, alt)?;
                    fx.b.set(m, Atom::bool(true));
                    fx.pop_scope()?;
                    fx.break_to(outer)?;
                    let ab = fx.close_nest();
                    fx.b.push(Stmt::Block(ab));
                }
                let ob = fx.close_nest();
                fx.b.push(Stmt::Block(ob));
                let mv = fx.b.let_(Atom::V(m));
                self.need(fx, mv, fail)
            }
        }
    }

    /// A pattern that never matches a value of this type.
    fn never(&mut self, fx: &mut Fx, fail: u32) -> R<()> {
        self.need(fx, Atom::bool(false), fail)
    }

    /// A name a pattern tests: a variant, an event or an error.
    fn name_test(&mut self, fx: &mut Fx, n: &str, v: &Atom, rt: &RType, fail: u32) -> R<()> {
        match rt {
            RType::Named(k) if self.t.kinds.contains_key(k) => match self.t.variants.get(n).cloned() {
                Some((kind, idx, _)) if &kind == k => {
                    let t = fx.b.ty(v);
                    let tag = fx.b.p("sum.tag", vec![t], vec![v.clone()])?;
                    let is = fx.b.int_cmp("int.eq", tag, lit(idx as i128))?;
                    self.need(fx, is, fail)
                }
                _ => self.never(fx, fail),
            },
            RType::Named(k) if k == "$event" => match self.t.events.iter().position(|e| e == n) {
                Some(idx) => {
                    let t = fx.b.ty(v);
                    let tag = fx.b.p("sum.tag", vec![t], vec![v.clone()])?;
                    let is = fx.b.int_cmp("int.eq", tag, lit(idx as i128))?;
                    self.need(fx, is, fail)
                }
                None => self.never(fx, fail),
            },
            RType::Named(k) if k == n => Ok(()),
            _ => self.never(fx, fail),
        }
    }
}

/// `__any`: the result type of a probe function, which the interpreter does not fit.
fn is_any(t: &ast::Ty) -> bool {
    matches!(t, ast::Ty::Path(p, a) if p.len() == 1 && p[0] == "__any" && a.is_empty())
}

fn binds(p: &Pat) -> bool {
    match p {
        Pat::Path(path) => path.len() == 1 && path[0].chars().next().is_some_and(|c| c.is_lowercase() || c == '_') && path[0] != "_",
        Pat::Ctor(_, args) => args.iter().any(|(_, q)| binds(q)),
        Pat::Tuple(ps) | Pat::Alt(ps) => ps.iter().any(binds),
        _ => false,
    }
}

impl Fx {
    /// Forget an owned value a scope holds (it was consumed).
    fn disown(&mut self, a: &Atom) {
        for s in self.scopes.iter_mut() {
            s.drops.retain(|x| x != a);
        }
    }
}

impl<'m> Lw<'m> {
    // ---- functions

    /// Fit the result of a function to its declared type and return it
    /// (interp `call_fn`).
    fn finish_fn(&mut self, fx: &mut Fx, v: Val, decl: Option<&ast::Ty>) -> R<()> {
        let a = match decl {
            None => {
                fx.b.drop(v.a)?;
                fx.b.unit()?
            }
            // `__any` (the probes of the emitter tests): the result is not fitted
            Some(t) if is_any(t) => {
                let want = fx.b.ret.clone();
                if fx.b.ty(&v.a) == want { v.a } else { return no("an unfitted result of another type") }
            }
            Some(_) if fx.inst_ret.is_some() => {
                // an instantiation: the result fitted where the declared type is
                // concrete, and kept where it is a type parameter
                let r = fx.inst_ret.clone().unwrap();
                self.conv(fx, v.a, &v.t, &r, true)?
            }
            Some(t) => self.fit_decl(fx, v.a, &v.t, t)?.a,
        };
        let a = self.with_muts(fx, a)?;
        fx.drops_to(None)?;
        fx.b.ret(a);
        Ok(())
    }

    /// The result of a function with `&mut` parameters: the result and the
    /// parameters' current values.
    fn with_muts(&mut self, fx: &mut Fx, a: Atom) -> R<Atom> {
        if fx.muts.is_empty() {
            return Ok(a);
        }
        let mut parts = vec![a];
        for v in fx.muts.clone() {
            parts.push(self.read_var(fx, v)?);
        }
        let t = fx.b.ret.clone();
        fx.b.p("tup.mk", vec![t], parts)
    }

    /// The type of the values of core parameter k (self's is the impl's type).
    fn param_rtype(&self, f: &FnDecl, k: usize, self_t: Option<&RType>) -> RType {
        match &f.sig.params[k] {
            Param::Named { ty, .. } => self.t.of_ty(ty),
            _ => self_t.cloned().unwrap_or(RType::Unit),
        }
    }

    fn named_params(f: &FnDecl) -> Vec<ast::Ty> {
        f.sig.params.iter().filter_map(|p| if let Param::Named { ty, .. } = p { Some(ty.clone()) } else { None }).collect()
    }

    fn fn_sig(&mut self, f: &FnDecl, self_t: Option<&RType>) -> R<(Vec<Ty>, Ty)> {
        if !f.sig.generics.is_empty() {
            return no(format!("the generic function {}", f.sig.name));
        }
        if f.sig.bound.is_some() {
            return no(format!("the recursion bound of {}", f.sig.name));
        }
        let mut ps = vec![];
        for p in &f.sig.params {
            match p {
                Param::Named { ty, .. } => {
                    let rt = self.t.of_ty(ty);
                    ps.push(self.ct(&rt)?);
                }
                Param::SelfVal | Param::SelfRef(_) => {
                    let st = self_t.cloned().ok_or("not covered by the lowering: self outside an impl")?;
                    ps.push(self.ct(&st)?);
                }
            }
        }
        let ret = match &f.sig.ret {
            Some(ast::Ty::Iter(_)) => return no(format!("the iterator function {}", f.sig.name)),
            _ => {
                let rt = self.ret_rtype(f);
                self.ct(&rt)?
            }
        };
        // `&mut` (form-ir-2.md section 2): the parameters' final values are
        // returned beside the result, and the caller writes them back
        let muts = Self::mut_params(f);
        if muts.is_empty() {
            return Ok((ps, ret));
        }
        let mut rs = vec![ret];
        for k in muts {
            rs.push(ps[k].clone());
        }
        Ok((ps, Ty::Tup(rs)))
    }

    /// The positions among a function's core parameters of its `&mut`
    /// parameters, in order, then of a `&mut self`: the order in which the
    /// reference copies them out (interp `call_fn`).
    fn mut_params(f: &FnDecl) -> Vec<usize> {
        let mut named = vec![];
        let mut this = None;
        for (k, p) in f.sig.params.iter().enumerate() {
            match p {
                Param::Named { ty: ast::Ty::Ref(true, _), .. } => named.push(k),
                Param::SelfRef(true) => this = Some(k),
                _ => {}
            }
        }
        named.extend(this);
        named
    }

    /// The type of a function's result: its declared type, unit when none
    /// is declared, and the type of its body's value for `__any`.
    fn ret_rtype(&self, f: &FnDecl) -> RType {
        match &f.sig.ret {
            None => RType::Unit,
            Some(t) if is_any(t) => match &f.body {
                Body::Arrow(e) => self.node_ty(e),
                Body::Block(b) => self.tail_type(b).unwrap_or(RType::Unit),
            },
            Some(t) => self.t.of_ty(t),
        }
    }

    fn lower_fn(&mut self, id: usize, f: &FnDecl, self_t: Option<&RType>) -> R<()> {
        let (ps, ret) = self.fn_sig(f, self_t)?;
        self.lower_fn_sig(id, f, self_t, ps, ret, None)
    }

    /// The function `f` with these core parameter and result types; an
    /// instantiation gives the type its result is fitted to.
    fn lower_fn_sig(&mut self, id: usize, f: &FnDecl, self_t: Option<&RType>, ps: Vec<Ty>, ret: Ty, inst_ret: Option<RType>) -> R<()> {
        if f.sig.bound.is_some() {
            return no(format!("the recursion bound of {}", f.sig.name));
        }
        let union = matches!(&f.sig.ret, Some(ast::Ty::Union(..)));
        let mut fx = self.fx(ps, ret.clone(), Ret::Fn(f.sig.ret.clone()), if union { FailCx::Union } else { FailCx::Halt }, true);
        fx.inst_ret = inst_ret;
        if union {
            // with `&mut` parameters the union is the first part of the result
            fx.union_t = Some(match &ret {
                Ty::Tup(rs) if !Self::mut_params(f).is_empty() => rs[0].clone(),
                _ => ret,
            });
        }
        let (ids, self_id) = self.ty.params.get(&addr(f)).cloned().unwrap_or_default();
        let mut k = 0usize;
        for p in &f.sig.params {
            let a = fx.b.arg(k);
            k += 1;
            let lid = match p {
                Param::Named { .. } => ids.get(k - 1 - self_id.map_or(0, |_| 1)).copied(),
                _ => self_id,
            };
            let Some(lid) = lid else { return no("a parameter") };
            let rt = self.ty.locals[lid as usize].ty.clone();
            if Self::mut_params(f).contains(&(k - 1)) {
                // a `&mut` parameter is a var of the function, written back by the caller
                let st = self.ct(&rt)?;
                let from = self.param_rtype(f, k - 1, self_t);
                let a2 = self.conv(&mut fx, a, &from, &rt, false)?;
                let v = fx.b.var(st, a2);
                fx.bound.insert(lid, Bound::Var(v, rt));
                fx.own(Atom::V(v));
                continue;
            }
            fx.bound.insert(lid, Bound::Val(a.clone(), rt));
            fx.own(a);
        }
        // the vars of the `&mut` parameters, in the order of `mut_params`
        for k in Self::mut_params(f) {
            let lid = match &f.sig.params[k] {
                Param::Named { .. } => {
                    let named_k = f.sig.params[..k].iter().filter(|p| matches!(p, Param::Named { .. })).count();
                    ids.get(named_k).copied()
                }
                _ => self_id,
            };
            match lid.and_then(|l| fx.bound.get(&l).cloned()) {
                Some(Bound::Var(v, _)) => fx.muts.push(v),
                _ => return no("a &mut parameter"),
            }
        }
        match &f.body {
            Body::Arrow(e) => {
                let v = self.expr(&mut fx, e)?;
                self.finish_fn(&mut fx, v, f.sig.ret.as_ref())?;
            }
            Body::Block(b) => {
                let v = self.block(&mut fx, b, true)?;
                if fx.b.falls() {
                    let v = match v {
                        Some(v) => v,
                        None => Val { a: fx.b.unit()?, t: RType::Unit },
                    };
                    self.finish_fn(&mut fx, v, f.sig.ret.as_ref())?;
                }
            }
        }
        self.f.define(id, fx.b.finish());
        Ok(())
    }

    /// The module's items that change what every observation means are
    /// reported here as not covered; functions and methods are lowered when the trace first
    /// reaches them (`user_fn`, `method_fn`), so that one the trace never
    /// calls does not stop the module from being covered.
    fn functions(&mut self) -> R<()> {
        let m = self.m;
        for it in &m.items {
            match it {
                Item::Impl { ty, members, .. } => {
                    let h = crate::check::ty_head(ty);
                    for f in members {
                        self.impl_tys.insert((h.clone(), f.sig.name.clone()), ty.clone());
                    }
                }
                // a trait has signatures only, which nothing runs; a foreign
                // function is reported as not covered where it is called
                Item::Trait { .. } | Item::Extern { .. } => {}
                Item::Actor { .. } | Item::Protocol { .. } | Item::Node { .. } | Item::Supervisor { .. } => return no("an actor, protocol, node or supervisor"),
                Item::State { replicated: true, .. } => return no("a replicated state"),
                Item::Step { deadline: Some(_), .. } => return no("a step deadline"),
                _ => {}
            }
        }
        Ok(())
    }

    // ---- the state, the steps and δ

    fn regions(&self) -> Vec<(&'m str, &'m [ast::Field])> {
        self.m.items.iter().filter_map(|i| match i {
            Item::State { name, fields, .. } => Some((name.as_str(), fields.as_slice())),
            _ => None,
        }).collect()
    }

    fn region_t(&mut self, fields: &[ast::Field]) -> R<(Ty, Vec<RType>)> {
        let mut ts = vec![];
        let mut rts = vec![];
        for f in fields {
            let rt = self.t.of_ty(&f.ty);
            ts.push(self.ct(&rt)?);
            rts.push(rt);
        }
        Ok((Ty::Tup(ts), rts))
    }

    fn state_t(&mut self) -> R<(Ty, RType)> {
        let mut ts = vec![];
        let mut rts = vec![];
        for (_, fs) in self.regions() {
            let (t, r) = self.region_t(fs)?;
            ts.push(t);
            rts.push(RType::Tuple(r));
        }
        Ok((Ty::Tup(ts), RType::Tuple(rts)))
    }

    /// The function of one region (ONE-4): its steps in source order, each a
    /// test of the event, the payload bindings, the guard and the body; it
    /// returns none when no step applies and halts when the step keeps the state.
    fn region_fn(&mut self, name: &str, fields: &[ast::Field]) -> R<usize> {
        let (ft, frts) = self.region_t(fields)?;
        let res_t = Ty::Sum(vec![Ty::unit(), Ty::Tup(vec![ft.clone(), Ty::list(Ty::Bytes)])]);
        let id = self.f.declare(&format!("region {name}"), vec![ft.clone(), self.ev_t.clone()], res_t.clone());
        let mut fx = self.fx(vec![ft.clone(), self.ev_t.clone()], res_t.clone(), Ret::Step, FailCx::Halt, true);
        let (fs, ev) = (fx.b.arg(0), fx.b.arg(1));
        fx.own(fs.clone());
        fx.own(ev.clone());
        let m = self.m;
        for it in &m.items {
            let Item::Step { region, pat, guard, body, .. } = it else { continue };
            if region != name {
                continue;
            }
            let sb = fx.open_nest();
            fx.push_scope();
            // interp `region_step`: the outputs are cleared before each step
            let o = fx.b.let_global(self.out_g, Ty::list(Ty::Bytes));
            fx.b.drop(o)?;
            let e = fx.b.p("list.new", vec![Ty::Bytes], vec![])?;
            fx.b.setg(self.out_g, e);
            let ids = self.ty.step_fields.get(&addr(it)).cloned().unwrap_or_default();
            let c = fx.b.copy(fs.clone())?;
            let parts = fx.b.prim("tup.open", vec![ft.clone()], vec![c])?;
            let mut vars = vec![];
            for (k, part) in parts.into_iter().enumerate() {
                let st = self.ty.locals[ids[k] as usize].ty.clone();
                let ct = self.ct(&st)?;
                let a = self.conv(&mut fx, part, &frts[k], &st, false)?;
                let x = fx.b.var(ct, a);
                fx.bound.insert(ids[k], Bound::Var(x, st.clone()));
                fx.own(Atom::V(x));
                vars.push((x, st));
            }
            fx.push_scope();
            self.pat(&mut fx, pat, &ev, &RType::Named("$event".into()), sb)?;
            if let Some(g) = guard {
                let gv = self.expr(&mut fx, g)?;
                self.need(&mut fx, gv.a, sb)?;
            }
            match body {
                Body::Arrow(e) => {
                    let v = self.expr(&mut fx, e)?;
                    let Some((x, st)) = vars.first().cloned() else { return no("a step on a state without fields") };
                    let a = self.conv(&mut fx, v.a, &v.t, &st, false)?;
                    self.store_var(&mut fx, x, a)?;
                }
                Body::Block(b) => {
                    self.block(&mut fx, b, false)?;
                }
            }
            if fx.b.falls() {
                fx.pop_scope()?;
                // each field is fitted to its declared type; a trap keeps the state
                let mut out = vec![];
                for (k, (x, st)) in vars.iter().enumerate() {
                    fx.disown(&Atom::V(*x));
                    let a = self.take_var(&mut fx, *x);
                    let v = self.fit_decl(&mut fx, a, st, &fields[k].ty)?;
                    out.push(v.a);
                }
                let nf = fx.b.p("tup.mk", vec![ft.clone()], out)?;
                let outs = fx.b.let_global(self.out_g, Ty::list(Ty::Bytes));
                let e = fx.b.p("list.new", vec![Ty::Bytes], vec![])?;
                fx.b.setg(self.out_g, e);
                let pair = fx.b.p("tup.mk", vec![Ty::Tup(vec![ft.clone(), Ty::list(Ty::Bytes)])], vec![nf, outs])?;
                let r = fx.b.p("sum.mk", vec![res_t.clone()], vec![lit(1), pair])?;
                fx.drops_to(None)?;
                fx.b.ret(r);
            } else {
                fx.scopes.pop();
            }
            fx.scopes.pop();
            let blk = fx.close_nest();
            fx.b.push(Stmt::Block(blk));
        }
        fx.drops_to(None)?;
        let u = fx.b.unit()?;
        let r = fx.b.p("sum.mk", vec![res_t], vec![lit(0), u])?;
        fx.b.ret(r);
        self.f.define(id, fx.b.finish());
        Ok(id)
    }

    /// δ (6.4): every region on the pre-state, in declaration order; the
    /// first region that keeps the state keeps it for the whole event.
    fn delta_fn(&mut self) -> R<usize> {
        let (st, _) = self.state_t()?;
        let res_t = Ty::Sum(vec![Ty::Tup(vec![st.clone(), Ty::list(Ty::Bytes)]), Ty::Bytes]);
        let mut rfs = vec![];
        for (name, fields) in self.regions() {
            let (ft, _) = self.region_t(fields)?;
            rfs.push((self.region_fn(name, fields)?, ft));
        }
        let kept_reason = rt::kept_reason(&mut self.f)?;
        let id = self.f.declare("delta", vec![st.clone(), self.ev_t.clone()], res_t.clone());
        let mut fx = self.fx(vec![st.clone(), self.ev_t.clone()], res_t.clone(), Ret::None, FailCx::Halt, false);
        let (s, ev) = (fx.b.arg(0), fx.b.arg(1));
        let parts = fx.b.prim("tup.open", vec![st.clone()], vec![s])?;
        let mut vars = vec![];
        for (p, (_, ft)) in parts.into_iter().zip(&rfs) {
            vars.push(fx.b.var(ft.clone(), p));
        }
        let o0 = fx.b.p("list.new", vec![Ty::Bytes], vec![])?;
        let outs = fx.b.var(Ty::list(Ty::Bytes), o0);
        for (k, (rf, ft)) in rfs.iter().enumerate() {
            let rc = fx.b.copy(Atom::V(vars[k]))?;
            let ec = fx.b.copy(ev.clone())?;
            let rres = self.f.ret(*rf);
            let c = fx.b.catch(sym(*rf), rres.clone(), vec![rc, ec]);
            let ct = fx.b.ty(&c);
            let kept = fx.b.p("sum.tag", vec![ct.clone()], vec![c.clone()])?;
            let is1 = fx.b.int_cmp("int.eq", kept, lit(1))?;
            fx.b.open();
            {
                let h = fx.b.p("sum.get", vec![ct.clone()], vec![c.clone(), lit(1)])?;
                let txt = self.call_rt(&mut fx, kept_reason, vec![h]);
                for v in &vars {
                    let x = fx.b.let_(Atom::V(*v));
                    fx.b.drop(x)?;
                }
                let o = fx.b.let_(Atom::V(outs));
                fx.b.drop(o)?;
                fx.b.drop(ev.clone())?;
                let r = fx.b.p("sum.mk", vec![res_t.clone()], vec![lit(1), txt])?;
                fx.b.ret(r);
            }
            let th = fx.b.close();
            fx.b.open();
            {
                let tag = fx.b.p("sum.tag", vec![ct.clone()], vec![c.clone()])?;
                let is0 = fx.b.int_cmp("int.eq", tag, lit(0))?;
                fx.b.open();
                let res = fx.b.p("sum.get", vec![ct.clone()], vec![c.clone(), lit(0)])?;
                let pair_t = Ty::Tup(vec![ft.clone(), Ty::list(Ty::Bytes)]);
                let vk = vars[k];
                self.switch(&mut fx, &res, &rres, 2, |lw, fx, j| {
                    let p = fx.b.p("sum.get", vec![rres.clone()], vec![res.clone(), lit(j as i128)])?;
                    if j == 0 {
                        return fx.b.drop(p);
                    }
                    let nf_o = fx.b.prim("tup.open", vec![pair_t.clone()], vec![p])?;
                    let old = fx.b.let_(Atom::V(vk));
                    fx.b.drop(old)?;
                    fx.b.set(vk, nf_o[0].clone());
                    lw.append_list(fx, outs, nf_o[1].clone())
                })?;
                let t0 = fx.b.close();
                fx.b.open();
                self.trap(&mut fx, Atom::bytes(b"internal: catch tag"))?;
                let e0 = fx.b.close();
                fx.b.push(Stmt::If(is0, t0, e0));
            }
            let el = fx.b.close();
            fx.b.push(Stmt::If(is1, th, el));
        }
        let mut fs = vec![];
        for v in &vars {
            fs.push(fx.b.let_(Atom::V(*v)));
        }
        let ns = fx.b.p("tup.mk", vec![st.clone()], fs)?;
        let o = fx.b.let_(Atom::V(outs));
        let pair = fx.b.p("tup.mk", vec![Ty::Tup(vec![st, Ty::list(Ty::Bytes)])], vec![ns, o])?;
        fx.b.drop(ev)?;
        let r = fx.b.p("sum.mk", vec![res_t], vec![lit(0), pair])?;
        fx.b.ret(r);
        self.f.define(id, fx.b.finish());
        Ok(id)
    }

    /// Append the items of the list `l` (consumed) to the list variable `v`.
    fn append_list(&mut self, fx: &mut Fx, v: u32, l: Atom) -> R<()> {
        let lt = fx.b.ty(&l);
        let Ty::List(et) = &lt else { return Err("internal: append of a non-list".into()) };
        let et = (**et).clone();
        let n = fx.b.p("list.len", vec![et.clone()], vec![l.clone()])?;
        let i = fx.b.var(Ty::u64(), lit(0));
        fx.b.open();
        let iv = fx.b.let_(Atom::V(i));
        let more = fx.b.int_cmp("int.lt", iv.clone(), n)?;
        fx.b.open();
        let th = fx.b.close();
        fx.b.open();
        fx.b.brk(1);
        let el = fx.b.close();
        fx.b.push(Stmt::If(more, th, el));
        let x = fx.b.list_at(l.clone(), iv.clone())?;
        let o = fx.b.let_(Atom::V(v));
        let o = fx.b.p("list.push", vec![et], vec![o, x])?;
        fx.b.set(v, o);
        let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
        fx.b.set(i, i1);
        let body = fx.b.close();
        fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
        fx.b.drop(l)
    }

    /// Print each item of a list of bytes (consumed) between `pre` and a newline.
    fn print_list(&mut self, fx: &mut Fx, l: Atom, pre: &[u8]) -> R<()> {
        let n = fx.b.p("list.len", vec![Ty::Bytes], vec![l.clone()])?;
        let i = fx.b.var(Ty::u64(), lit(0));
        fx.b.open();
        let iv = fx.b.let_(Atom::V(i));
        let more = fx.b.int_cmp("int.lt", iv.clone(), n)?;
        fx.b.open();
        let th = fx.b.close();
        fx.b.open();
        fx.b.brk(1);
        let el = fx.b.close();
        fx.b.push(Stmt::If(more, th, el));
        let x = fx.b.list_at(l.clone(), iv.clone())?;
        let line = fx.b.cats(vec![Atom::bytes(pre), x, Atom::bytes(b"\n")])?;
        fx.b.out(line);
        let i1 = fx.b.add_in(iv, lit(1), Ty::u64())?;
        fx.b.set(i, i1);
        let body = fx.b.close();
        fx.b.push(Stmt::Loop(vec![Stmt::Block(body)]));
        fx.b.drop(l)
    }

    /// The initial state (interp `initial_state`): each field's initial
    /// value in a scope of its own, fitted to the field's type.
    fn init_state_fn(&mut self) -> R<usize> {
        let (st, _) = self.state_t()?;
        let id = self.f.declare("initial state", vec![], st.clone());
        let mut fx = self.fx(vec![], st.clone(), Ret::None, FailCx::Halt, true);
        let mut rs = vec![];
        for (_, fields) in self.regions() {
            let (ft, _) = self.region_t(fields)?;
            let mut vs = vec![];
            for f in fields {
                match &f.init {
                    Some(e) => {
                        let v = self.expr(&mut fx, e)?;
                        let v = self.fit_decl(&mut fx, v.a, &v.t, &f.ty)?;
                        vs.push(v.a);
                    }
                    None => {
                        self.trap(&mut fx, Atom::bytes(b"field without initial value"))?;
                        self.f.define(id, fx.b.finish());
                        return Ok(id);
                    }
                }
            }
            rs.push(fx.b.p("tup.mk", vec![ft], vs)?);
        }
        let s = fx.b.p("tup.mk", vec![st], rs)?;
        fx.b.ret(s);
        self.f.define(id, fx.b.finish());
        Ok(id)
    }

    /// The display of a state: `R { f = v, … }`, regions joined by `; `.
    fn show_state_fn(&mut self) -> R<usize> {
        let (st, _) = self.state_t()?;
        if let Some(id) = self.f.lookup("show state") {
            return Ok(id);
        }
        let id = self.f.declare("show state", vec![st.clone()], Ty::Bytes);
        self.f.name("show state", id);
        let mut fx = self.fx(vec![st.clone()], Ty::Bytes, Ret::None, FailCx::Halt, false);
        let s = fx.b.arg(0);
        let regs: Vec<(String, Vec<ast::Field>)> = self.regions().into_iter().map(|(n, f)| (n.to_string(), f.to_vec())).collect();
        let parts = fx.b.prim("tup.open", vec![st], vec![s])?;
        let mut out = vec![];
        for (k, ((name, fields), r)) in regs.iter().zip(parts).enumerate() {
            if k > 0 {
                out.push(Atom::bytes(b"; "));
            }
            let (ft, frts) = self.region_t(fields)?;
            let fs = fx.b.prim("tup.open", vec![ft], vec![r])?;
            out.push(Atom::bytes(format!("{name} {{ ").as_bytes()));
            for (j, ((f, rt), v)) in fields.iter().zip(&frts).zip(fs).enumerate() {
                if j > 0 {
                    out.push(Atom::bytes(b", "));
                }
                out.push(Atom::bytes(format!("{} = ", f.name).as_bytes()));
                out.push(self.show(&mut fx, v, rt)?);
            }
            out.push(Atom::bytes(b" }"));
        }
        let r = fx.b.cats(out)?;
        fx.b.ret(r);
        self.f.define(id, fx.b.finish());
        Ok(id)
    }

    /// An event as the log hashes it: `{"payload":P,"type":k}` (interp `event_pos_json`).
    fn ev_json_fn(&mut self) -> R<usize> {
        if let Some(id) = self.f.lookup("event json") {
            return Ok(id);
        }
        let et = self.ev_t.clone();
        let id = self.f.declare("event json", vec![et.clone()], Ty::Bytes);
        self.f.name("event json", id);
        let mut fx = self.fx(vec![et.clone()], Ty::Bytes, Ret::None, FailCx::Halt, false);
        let ev = fx.b.arg(0);
        let r = fx.b.var(Ty::Bytes, Atom::bytes(b""));
        let evs = self.t.events.clone();
        let n = evs.len().max(1);
        self.switch(&mut fx, &ev, &et, n, |lw, fx, k| {
            let p = fx.b.p("sum.get", vec![et.clone()], vec![ev.clone(), lit(k as i128)])?;
            let Some(name) = evs.get(k) else { return fx.b.drop(p) };
            let j = lw.cjp(fx, p, &RType::Named(name.clone()))?;
            let s = fx.b.cats(vec![Atom::bytes(b"{\"payload\":"), j, Atom::bytes(format!(",\"type\":{k}}}").as_bytes())])?;
            fx.b.set(r, s);
            Ok(())
        })?;
        let rv = fx.b.let_(Atom::V(r));
        fx.b.ret(rv);
        self.f.define(id, fx.b.finish());
        Ok(id)
    }

    /// An event as display shows it.
    fn ev_show_fn(&mut self) -> R<usize> {
        if let Some(id) = self.f.lookup("event show") {
            return Ok(id);
        }
        let et = self.ev_t.clone();
        let id = self.f.declare("event show", vec![et.clone()], Ty::Bytes);
        self.f.name("event show", id);
        let mut fx = self.fx(vec![et.clone()], Ty::Bytes, Ret::None, FailCx::Halt, false);
        let ev = fx.b.arg(0);
        let r = fx.b.var(Ty::Bytes, Atom::bytes(b""));
        let evs = self.t.events.clone();
        let n = evs.len().max(1);
        self.switch(&mut fx, &ev, &et, n, |lw, fx, k| {
            let p = fx.b.p("sum.get", vec![et.clone()], vec![ev.clone(), lit(k as i128)])?;
            let Some(name) = evs.get(k) else { return fx.b.drop(p) };
            let s = lw.show(fx, p, &RType::Named(name.clone()))?;
            fx.b.set(r, s);
            Ok(())
        })?;
        let rv = fx.b.let_(Atom::V(r));
        fx.b.ret(rv);
        self.f.define(id, fx.b.finish());
        Ok(id)
    }

    // ---- observations (ONE-5)

    /// Print `label = show(v)` or `label why(h)` for a caught result.
    fn print_result(&mut self, fx: &mut Fx, c: Atom, rt: &RType, label: &str) -> R<()> {
        let ct = fx.b.ty(&c);
        let why = rt::why(&mut self.f)?;
        let rt = rt.clone();
        self.switch(fx, &c, &ct, 2, |lw, fx, k| {
            let p = fx.b.p("sum.get", vec![ct.clone()], vec![c.clone(), lit(k as i128)])?;
            let line = if k == 0 {
                let s = lw.show(fx, p, &rt)?;
                fx.b.cats(vec![Atom::bytes(format!("{label} = ").as_bytes()), s, Atom::bytes(b"\n")])?
            } else {
                let w = lw.call_rt(fx, why, vec![p]);
                fx.b.cats(vec![Atom::bytes(format!("{label} ").as_bytes()), w, Atom::bytes(b"\n")])?
            };
            fx.b.out(line);
            Ok(())
        })
    }

    /// An observation body: `body() -> bool`, printing its lines.
    fn obs_body(&mut self, name: &str, build: impl FnOnce(&mut Self, &mut Fx) -> R<()>) -> R<usize> {
        let id = self.f.declare(name, vec![], Ty::Bool);
        let mut fx = self.fx(vec![], Ty::Bool, Ret::None, FailCx::Halt, false);
        build(self, &mut fx)?;
        fx.b.ret(Atom::bool(true));
        self.f.define(id, fx.b.finish());
        Ok(id)
    }

    /// A function that evaluates one expression alone (a check), charged fuel.
    fn expr_fn(&mut self, name: &str, e: &Expr) -> R<(usize, RType)> {
        let rt = self.node_ty(e);
        let t = self.ct(&rt)?;
        let id = self.f.declare(name, vec![], t.clone());
        let mut fx = self.fx(vec![], t, Ret::None, FailCx::Halt, true);
        let v = self.expr(&mut fx, e)?;
        let a = self.conv(&mut fx, v.a, &v.t, &rt, false)?;
        fx.b.ret(a);
        self.f.define(id, fx.b.finish());
        Ok((id, rt))
    }

    fn obs_check(&mut self, e: &Expr) -> R<usize> {
        let line = e.line();
        let (f, rt) = self.expr_fn(&format!("check {line}"), e)?;
        self.obs_body(&format!("observe check {line}"), |lw, fx| {
            let r = lw.f.ret(f);
            let c = fx.b.catch(sym(f), r, vec![]);
            lw.print_result(fx, c, &rt, &format!("check {line}"))
        })
    }

    fn obs_const(&mut self, name: &str) -> R<usize> {
        let (f, _, _) = self.const_slot(name)?;
        let Some((decl, e)) = self.t.consts.get(name).cloned() else { return no("a constant") };
        let rt = match &decl {
            Some(t) => self.t.of_ty(t),
            None => self.node_ty(e),
        };
        self.obs_body(&format!("observe const {name}"), |lw, fx| {
            let r = lw.f.ret(f);
            let c = fx.b.catch(sym(f), r, vec![]);
            lw.print_result(fx, c, &rt, &format!("const {name}"))
        })
    }

    /// A call of a function of the module with no arguments, as an
    /// observation makes it (interp `call_fn`, with its tick).
    fn call_fn0(&mut self, f: &FnDecl) -> R<(usize, RType)> {
        if matches!(f.sig.ret, Some(ast::Ty::Iter(_))) {
            // a call of an iterator function takes one step and runs
            // nothing: its value is the (empty) frame, shown as `<iter>`
            let w = self.f.declare(&format!("call {}", f.sig.name), vec![], Ty::unit());
            let mut fx = self.fx(vec![], Ty::unit(), Ret::None, FailCx::Halt, true);
            fx.b.tick(1);
            self.charge_k(&mut fx, crate::conc::C_CALL)?;
            let u = fx.b.unit()?;
            fx.b.ret(u);
            self.f.define(w, fx.b.finish());
            return Ok((w, RType::Gen(String::new(), Box::new(RType::Never))));
        }
        let id = self.user_fn(&f.sig.name)?;
        let rt = self.ret_rtype(f);
        let t = self.f.ret(id);
        let w = self.f.declare(&format!("call {}", f.sig.name), vec![], t.clone());
        let mut fx = self.fx(vec![], t.clone(), Ret::None, FailCx::Halt, true);
        fx.b.tick(1);
        self.charge_k(&mut fx, crate::conc::C_CALL)?;
        let v = fx.b.call(sym(id), t, vec![]);
        fx.b.ret(v);
        self.f.define(w, fx.b.finish());
        Ok((w, rt))
    }

    fn obs_fn(&mut self, f: &FnDecl) -> R<usize> {
        let (w, rt) = self.call_fn0(f)?;
        let name = f.sig.name.clone();
        self.obs_body(&format!("observe fn {name}"), |lw, fx| {
            let r = lw.f.ret(w);
            let c = fx.b.catch(sym(w), r, vec![]);
            lw.print_result(fx, c, &rt, &format!("fn {name}"))
        })
    }

    fn obs_main(&mut self, f: &FnDecl) -> R<usize> {
        let (w, rt) = self.call_fn0(f)?;
        self.obs_body("observe main", |lw, fx| {
            fx.b.out(Atom::bytes(b"main\n"));
            let o = fx.b.let_global(lw.out_g, Ty::list(Ty::Bytes));
            fx.b.drop(o)?;
            let e = fx.b.p("list.new", vec![Ty::Bytes], vec![])?;
            fx.b.setg(lw.out_g, e);
            let r = lw.f.ret(w);
            let c = fx.b.catch(sym(w), r, vec![]);
            let outs = fx.b.let_global(lw.out_g, Ty::list(Ty::Bytes));
            let e = fx.b.p("list.new", vec![Ty::Bytes], vec![])?;
            fx.b.setg(lw.out_g, e);
            lw.print_list(fx, outs, b"emit ")?;
            lw.print_result(fx, c, &rt, "main")
        })
    }

    /// One example (interp `obs_example`): the initial state, each event
    /// with the state after it or why it kept the state, then the head of its log.
    fn obs_example(&mut self, x: &ast::Example, delta: usize, init: usize) -> R<usize> {
        let mut evs = vec![];
        for (k, (evn, args)) in x.events.iter().enumerate() {
            let Some(idx) = self.t.events.iter().position(|e| e == evn) else { return no(format!("an example event `{evn}` that is not a declared event")) };
            let id = self.f.declare(&format!("example {} event {}", x.line, k + 1), vec![], self.ev_t.clone());
            let mut fx = self.fx(vec![], self.ev_t.clone(), Ret::None, FailCx::Halt, true);
            let v = self.construct(&mut fx, evn, args, RType::Named(evn.clone()))?;
            let ev_t = self.ev_t.clone();
            let s = fx.b.p("sum.mk", vec![ev_t], vec![lit(idx as i128), v.a])?;
            fx.b.ret(s);
            self.f.define(id, fx.b.finish());
            evs.push((id, evn.clone()));
        }
        let (st, srt) = self.state_t()?;
        let show_state = self.show_state_fn()?;
        let ev_show = self.ev_show_fn()?;
        let ev_json = self.ev_json_fn()?;
        let sha = rt::sha_hex(&mut self.f)?;
        let why = rt::why(&mut self.f)?;
        let line = x.line;
        self.obs_body(&format!("observe example {line}"), |lw, fx| {
            fx.b.out(Atom::bytes(format!("example {line}\n").as_bytes()));
            let c = fx.b.catch(sym(init), st.clone(), vec![]);
            let ct = fx.b.ty(&c);
            lw.switch(fx, &c, &ct, 2, |lw, fx, k| {
                let p = fx.b.p("sum.get", vec![ct.clone()], vec![c.clone(), lit(k as i128)])?;
                if k == 1 {
                    let w = lw.call_rt(fx, why, vec![p]);
                    let l = fx.b.cats(vec![Atom::bytes(b"initial "), w, Atom::bytes(b"\n")])?;
                    fx.b.out(l);
                    return Ok(());
                }
                let s = fx.b.var(st.clone(), p);
                let sc = fx.b.copy(Atom::V(s))?;
                let txt = lw.call_rt(fx, show_state, vec![sc]);
                let l = fx.b.cats(vec![Atom::bytes(b"0 (initial) "), txt, Atom::bytes(b"\n")])?;
                fx.b.out(l);
                let prev = fx.b.var(Ty::Bytes, Atom::bytes(crate::interp::D_GENESIS.as_bytes()));
                let seq = fx.b.var(Ty::u64(), lit(0));
                let eb = fx.open_nest();
                for (k, (eid, evn)) in evs.iter().enumerate() {
                    let k = k + 1;
                    let er = fx.b.catch(sym(*eid), lw.ev_t.clone(), vec![]);
                    let ert = fx.b.ty(&er);
                    let evn = evn.clone();
                    lw.switch(fx, &er, &ert, 2, |lw, fx, j| {
                        let p = fx.b.p("sum.get", vec![ert.clone()], vec![er.clone(), lit(j as i128)])?;
                        if j == 1 {
                            let w = lw.call_rt(fx, why, vec![p]);
                            let l = fx.b.cats(vec![Atom::bytes(format!("{k} {evn} ").as_bytes()), w, Atom::bytes(b"\n")])?;
                            fx.b.out(l);
                            fx.b.brk(fx.depth - 1 - eb);
                            return Ok(());
                        }
                        let ev = p;
                        let sc = fx.b.copy(Atom::V(s))?;
                        let ec = fx.b.copy(ev.clone())?;
                        let dr = lw.f.ret(delta);
                        let d = fx.b.call(sym(delta), dr.clone(), vec![sc, ec]);
                        lw.switch(fx, &d, &dr, 2, |lw, fx, i| {
                            let q = fx.b.p("sum.get", vec![dr.clone()], vec![d.clone(), lit(i as i128)])?;
                            if i == 0 {
                                let Ty::Sum(cs) = &dr else { unreachable!() };
                                let so = fx.b.prim("tup.open", vec![cs[0].clone()], vec![q])?;
                                let old = fx.b.let_(Atom::V(s));
                                fx.b.drop(old)?;
                                fx.b.set(s, so[0].clone());
                                lw.dlog(fx, ev_json, sha, prev, seq, &ev, s)?;
                                let es = lw.call_rt(fx, ev_show, vec![ev.clone()]);
                                let sc = fx.b.copy(Atom::V(s))?;
                                let ss = lw.call_rt(fx, show_state, vec![sc]);
                                let l = fx.b.cats(vec![Atom::bytes(format!("{k} ").as_bytes()), es, Atom::bytes(b" "), ss, Atom::bytes(b"\n")])?;
                                fx.b.out(l);
                                lw.print_list(fx, so[1].clone(), b"  emit ")
                            } else {
                                lw.dlog(fx, ev_json, sha, prev, seq, &ev, s)?;
                                let es = lw.call_rt(fx, ev_show, vec![ev.clone()]);
                                let l = fx.b.cats(vec![Atom::bytes(format!("{k} ").as_bytes()), es, Atom::bytes(b" kept: "), q, Atom::bytes(b"\n")])?;
                                // a kept step keeps the state (T8); the next event is applied to it
                                fx.b.out(l);
                                Ok(())
                            }
                        })
                    })?;
                }
                let blk = fx.close_nest();
                fx.b.push(Stmt::Block(blk));
                let pv = fx.b.let_(Atom::V(prev));
                let l = fx.b.cats(vec![Atom::bytes(b"trace "), pv, Atom::bytes(b"\n")])?;
                fx.b.out(l);
                let sv = fx.b.let_(Atom::V(s));
                fx.b.drop(sv)
            })?;
            let _ = &srt;
            Ok(())
        })
    }

    /// Add an entry to the log chain (6.1): the hash of the positional event,
    /// the previous head, the sequence number and the hash of the state.
    #[allow(clippy::too_many_arguments)]
    fn dlog(&mut self, fx: &mut Fx, ev_json: usize, sha: usize, prev: u32, seq: u32, ev: &Atom, s: u32) -> R<()> {
        let (_, srt) = self.state_t()?;
        let ec = fx.b.copy(ev.clone())?;
        let evj = self.call_rt(fx, ev_json, vec![ec]);
        let sc = fx.b.copy(Atom::V(s))?;
        let sj = self.cjp(fx, sc, &srt)?;
        let sh = self.call_rt(fx, sha, vec![sj]);
        let pv = fx.b.let_(Atom::V(prev));
        let sq = fx.b.let_(Atom::V(seq));
        let sqd = self.dec(fx, sq.clone())?;
        let body = fx.b.cats(vec![
            Atom::bytes(b"{\"event\":"), evj, Atom::bytes(b",\"prev_hash\":\""), pv, Atom::bytes(b"\",\"seq\":"), sqd,
            Atom::bytes(b",\"state_hash\":\""), sh, Atom::bytes(b"\"}"),
        ])?;
        let h = self.call_rt(fx, sha, vec![body]);
        fx.b.set(prev, h);
        let s1 = fx.b.add_in(sq, lit(1), Ty::u64())?;
        fx.b.set(seq, s1);
        Ok(())
    }

    /// The whole module: functions, steps, then one observation per item in
    /// item order and `fn main` last, each from full fuel with no constant
    /// evaluated; returns the entry.
    fn module(&mut self) -> R<usize> {
        self.functions()?;
        let m = self.m;
        let has_examples = m.items.iter().any(|i| matches!(i, Item::Example(_)) || matches!(i, Item::Intent(_, xs, ..) if !xs.is_empty()));
        let (delta, init) = if has_examples { (Some(self.delta_fn()?), Some(self.init_state_fn()?)) } else { (None, None) };
        let mut bodies = vec![];
        let mut main = None;
        for it in &m.items {
            match it {
                Item::Const { name, .. } => bodies.push(self.obs_const(name)?),
                Item::Check(e) => bodies.push(self.obs_check(e)?),
                Item::Intent(_, xs, cks, ..) => {
                    for e in cks {
                        bodies.push(self.obs_check(e)?);
                    }
                    for x in xs {
                        bodies.push(self.obs_example(x, delta.unwrap(), init.unwrap())?);
                    }
                }
                Item::Example(x) => bodies.push(self.obs_example(x, delta.unwrap(), init.unwrap())?),
                Item::Fn(f) if f.sig.name == "main" && f.sig.params.is_empty() => main = Some(f),
                Item::Fn(f) if crate::emit::zero_arg(f) => bodies.push(self.obs_fn(f)?),
                _ => {}
            }
        }
        if let Some(f) = main {
            bodies.push(self.obs_main(f)?);
        }
        // each observation from full fuel and with no constant evaluated
        let flags: Vec<u32> = self.consts.values().map(|(_, d, _)| *d).collect();
        let mut owned: Vec<(u32, Ty)> = self.consts.values().filter_map(|(id, _, v)| {
            let t = self.f.ret(*id);
            t.owned().then_some((*v, t))
        }).collect();
        // the buffer stacks of iterator runs start empty
        for (t, g) in &self.gen_stacks {
            owned.push((*g, Ty::list(Ty::list(t.clone()))));
        }
        let dec = rt::dec(&mut self.f)?;
        let mut wrappers = vec![];
        for (k, b) in bodies.into_iter().enumerate() {
            let id = self.f.declare(&format!("observation {}", k + 1), vec![], Ty::Bool);
            let mut fx = self.fx(vec![], Ty::Bool, Ret::None, FailCx::Halt, false);
            let lim = fx.b.let_global(self.fuel.1, Ty::u64());
            fx.b.setg(self.fuel.0, lim);
            if self.costs {
                // and from no cost spent and no budget in force
                let (sp, dl) = rt::cost_globals(&mut self.f);
                fx.b.setg(sp, lit(0));
                fx.b.setg(dl, lit(u64::MAX as i128));
            }
            let mut fl = flags.clone();
            fl.sort();
            for g in fl {
                fx.b.setg(g, Atom::bool(false));
            }
            fx.b.call(sym(b), Ty::Bool, vec![]);
            let f = fx.b.let_global(self.fuel.0, Ty::u64());
            let lim = fx.b.let_global(self.fuel.1, Ty::u64());
            let used = fx.b.sub_in(lim, f, Ty::u64())?;
            let d = self.call_rt(&mut fx, dec, vec![used]);
            let l = fx.b.cats(vec![Atom::bytes(b"fuel "), d, Atom::bytes(b"\n")])?;
            fx.b.out(l);
            fx.b.ret(Atom::bool(true));
            self.f.define(id, fx.b.finish());
            wrappers.push(id);
        }
        let entry = self.f.declare("entry", vec![], Ty::unit());
        let mut fx = self.fx(vec![], Ty::unit(), Ret::None, FailCx::Halt, false);
        let e = fx.b.p("list.new", vec![Ty::Bytes], vec![])?;
        fx.b.setg(self.out_g, e);
        owned.sort_by_key(|x| x.0);
        for (g, t) in owned {
            let d = self.default(&mut fx, &t)?;
            fx.b.setg(g, d);
        }
        for w in wrappers {
            fx.b.call(sym(w), Ty::Bool, vec![]);
        }
        let u = fx.b.unit()?;
        fx.b.ret(u);
        self.f.define(entry, fx.b.finish());
        Ok(entry)
    }
}
