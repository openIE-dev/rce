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
//! A construct this lowering does not yet cover is refused with an error
//! that names it; the module is then not covered, and nothing is printed.
//! The constructs covered are those of the counter-style programs: checked
//! integers, bools, text values, structs, events, errors, kinds, options and
//! error unions, state, steps, refusals, `if`, `match`, bounded loops, `for`
//! over ranges and lists, functions and methods, and constants.
//!
//! Ownership (W3): an RCE variable holds its value until the end of its
//! scope, where it is dropped; every read of an owned variable is a `copy`;
//! a temporary is consumed by the operation that uses it.
use super::build::*;
use super::ir::*;
use super::rt;
use crate::ast::{self, Arg, Body, Expr, FnDecl, Item, Param, Pat, Stmt as S};
use crate::typed::{self, Binding, Callee, IntT, RType, Tables, Typed, addr};
use std::collections::HashMap;

/// Lower an admitted module to the core.
pub fn lower(m: &ast::Module) -> R<Module> {
    let t = Tables::new(m);
    let ty = typed::build(m);
    let mut lw = Lw::new(m, t, ty)?;
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
    /// a step body: `return` is refused
    Step,
    /// an expression evaluated alone (a constant, a check, an event)
    None,
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
}

impl Fx {
    fn new(b: FB, ret: Ret, fail: FailCx) -> Fx {
        Fx { b, scopes: vec![Scope { depth: 0, drops: vec![] }], bound: HashMap::new(), loops: vec![], depth: 0, ret, fail, union_t: None }
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
    /// a constant: its evaluation function, its "evaluated" flag and its value
    consts: HashMap<String, (usize, u32, u32)>,
    named: HashMap<String, Ty>,
    gens: HashMap<String, usize>,
    loop_limit: i128,
    /// the event sum: every declared event, in declaration order
    ev_t: Ty,
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
        let mut lw = Lw { m, t, ty, f, tick, fuel, out_g, fns: HashMap::new(), methods: HashMap::new(), consts: HashMap::new(), named: HashMap::new(), gens: HashMap::new(), loop_limit, ev_t: Ty::unit() };
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
            RType::Never => Ty::unit(),
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

    fn call_rt(&mut self, fx: &mut Fx, id: usize, args: Vec<Atom>) -> Atom {
        let r = self.f.ret(id);
        fx.b.call(sym(id), r, args)
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
        if self.t.refined(to) {
            return no("a refined type");
        }
        let target = self.t.of_ty(to);
        if let RType::Unknown(r) = &target {
            return no(r);
        }
        // an error value keeps its type: fit to a union keeps the error side
        let a2 = self.conv(fx, a, from, &target, true)?;
        Ok(Val { a: a2, t: target })
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
            RType::Map(..) => no("the display of a map"),
            RType::F64 | RType::F32 => no("the display of a float"),
            o => no(format!("the display of {}", o.show())),
        }
    }
}

impl<'m> Lw<'m> {
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
            (T::Map(..), _) | (_, T::Map(..)) => no("equality of maps"),
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
                } else {
                    let fts: Vec<RType> = self.t.fields(n).unwrap_or_default().iter().map(|(_, ft)| self.t.of_ty(ft)).collect();
                    let cs = fx.b.prim("tup.open", vec![t.clone()], vec![v])?;
                    self.cjp_list(fx, cs, &fts)
                }
            }
            RType::Map(..) => no("a map in the log"),
            RType::F64 | RType::F32 => no("a float in the log"),
            o => no(format!("{} in the log", o.show())),
        }
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
            Expr::Call(f, args, _, _) => self.call(fx, e, f, args, rt),
            Expr::Index(x, idx, _) => self.index(fx, x, idx, rt),
            Expr::Member(x, f, _) => self.member(fx, x, f, rt),
            Expr::TupleMember(x, i, _) => {
                let v = self.expr(fx, x)?;
                let RType::Tuple(_) = &v.t else { return no("a component of a non-tuple") };
                let t = fx.b.ty(&v.a);
                let c = fx.b.p("tup.get", vec![t], vec![v.a.clone(), lit(*i as i128)])?;
                fx.b.drop(v.a)?;
                Ok(Val { a: c, t: rt })
            }
            Expr::Neg(x, _) => {
                let v = self.expr(fx, x)?;
                match &v.t {
                    RType::Int(i) => {
                        let (r, name) = check_of(&IntT::of_tags(i.tags))?;
                        let c = fx.b.p("int.bad.neg", vec![r.clone()], vec![v.a.clone()])?;
                        let a = v.a.clone();
                        self.guard(fx, c, |lw, fx| lw.ovf(fx, "neg", vec![a], &name))?;
                        Ok(Val { a: fx.b.p("int.neg", vec![r], vec![v.a])?, t: rt })
                    }
                    RType::F64 => Ok(Val { a: fx.b.p("f64.neg", vec![], vec![v.a])?, t: rt }),
                    o => no(format!("negation of {}", o.show())),
                }
            }
            Expr::BitNot(x, _) => {
                let v = self.expr(fx, x)?;
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
            let v = self.fit(fx, v.a, &v.t, p)?;
            out.push(v.a);
        }
        Ok(out)
    }

    fn user_fn(&self, name: &str) -> R<usize> {
        self.fns.get(name).copied().ok_or_else(|| format!("not covered by the lowering: the function {name}"))
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
            Some(Callee::Fn(n)) => {
                let fd = self.t.fns[&n];
                let id = self.user_fn(&n)?;
                fx.b.tick(1);
                let a = self.args(fx, fd, args)?;
                let r = self.f.ret(id);
                Ok(Val { a: fx.b.call(sym(id), r, a), t: rt })
            }
            Some(Callee::Static(tn, m)) => {
                let fd = self.t.methods[&(tn.clone(), m.clone())];
                let id = self.methods.get(&(tn, m.clone())).copied().ok_or_else(|| format!("not covered by the lowering: the method {m}"))?;
                fx.b.tick(1);
                let a = self.args(fx, fd, args)?;
                let r = self.f.ret(id);
                Ok(Val { a: fx.b.call(sym(id), r, a), t: rt })
            }
            Some(Callee::Method(h, m)) => {
                let Expr::Member(obj, _, _) = f else { return no("a method call") };
                let fd = self.t.methods[&(h.clone(), m.clone())];
                let id = self.methods.get(&(h, m.clone())).copied().ok_or_else(|| format!("not covered by the lowering: the method {m}"))?;
                let recv = self.expr(fx, obj)?;
                fx.b.tick(1);
                let mut a = vec![recv.a];
                a.extend(self.args(fx, fd, args)?);
                let r = self.f.ret(id);
                Ok(Val { a: fx.b.call(sym(id), r, a), t: rt })
            }
            Some(Callee::Ctor(c)) => self.construct(fx, &c, args, rt),
            Some(Callee::Builtin(m)) => {
                let Expr::Member(obj, _, _) = f else { return no("a method call") };
                self.builtin(fx, obj, &m, args, rt)
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
                let v = self.fit(fx, v.a, &v.t, &fields[k].1)?;
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
                        let v = self.fit(fx, v.a, &v.t, t)?;
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

    fn builtin(&mut self, fx: &mut Fx, obj: &Expr, m: &str, args: &[Arg], rt: RType) -> R<Val> {
        let ort = self.node_ty(obj);
        // a method that writes its receiver back needs the receiver to be a variable
        if matches!(m, "push" | "append") {
            let Some(v) = self.var_of(fx, obj) else { return no(format!("{m} on a receiver that is not a variable")) };
            fx.b.tick(1); // the receiver's evaluation
            let x = self.expr(fx, &args[0].e)?;
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
        // a read of a variable's length needs no copy
        if matches!(m, "len" | "len_bytes" | "isempty")
            && let Some(v) = self.var_of(fx, obj)
            && args.is_empty()
        {
            fx.b.tick(1);
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
            (RType::Bool, "tostr") => {
                let r = fx.b.var(Ty::Bytes, Atom::bytes(b"false"));
                fx.b.open();
                fx.b.set(r, Atom::bytes(b"true"));
                let th = fx.b.close();
                fx.b.push(Stmt::If(recv.a, th, vec![]));
                Ok(Val { a: fx.b.let_(Atom::V(r)), t: rt })
            }
            (RType::Text, "tostr" | "raw") => Ok(recv),
            (RType::Text, "len" | "len_bytes") => Ok(Val { a: fx.b.p("bytes.len", vec![], vec![recv.a])?, t: rt }),
            (RType::Text, "isempty") => {
                let n = fx.b.p("bytes.len", vec![], vec![recv.a])?;
                Ok(Val { a: fx.b.int_cmp("int.eq", n, lit(0))?, t: rt })
            }
            (RType::List(_), "len" | "isempty") => {
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
            let t = fx.b.local_ty(v);
            return Ok(Val { a: fx.b.p("tup.get", vec![t], vec![Atom::V(v), lit(k as i128)])?, t: rt });
        }
        if let Some(Bound::Val(a, _)) = self.bound(fx, x)
            && matches!(x, Expr::Name(..) | Expr::SelfRef(_))
        {
            fx.b.tick(1);
            let t = fx.b.ty(&a);
            return Ok(Val { a: fx.b.p("tup.get", vec![t], vec![a, lit(k as i128)])?, t: rt });
        }
        let v = self.expr(fx, x)?;
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
        let x = self.expr(fx, a)?;
        let y = self.expr(fx, b)?;
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
                    (s, t) => return no(format!("the order of {} and {}", s.show(), t.show())),
                };
                Ok(Val { a: r, t: rt })
            }
            "in" | "not in" | ".." | "..=" => no(format!("the operator {op}")),
            _ => self.arith(fx, op, x, y, rt),
        }
    }

    /// interp `arith` on two values.
    fn arith(&mut self, fx: &mut Fx, op: &str, x: Val, y: Val, rt: RType) -> R<Val> {
        match (&x.t, &y.t) {
            (RType::Int(i), RType::Int(j)) => {
                let res = IntT::of_tags(i.tags.join_op(j.tags));
                let (r, name) = check_of(&res)?;
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
    /// the error's name and display (a step's refusal, `main`'s failure).
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
            S::Arena(name, None, body, _) if name == "$par" => {
                for b in body {
                    let S::Arena(_, _, x, _) = b else { return no("a par branch") };
                    self.block(fx, x, false)?;
                    if !fx.b.falls() {
                        break;
                    }
                }
            }
            S::Arena(name, None, body, _) | S::Device(name, None, body, _) if !name.starts_with('$') => {
                self.block(fx, body, false)?;
            }
            S::Arena(..) | S::Device(..) => return no("an arena with a bound, a device region or a marker region"),
            S::Send(..) | S::Spawn(..) | S::Select(..) | S::Yield(..) => return no("a concurrency or iterator statement"),
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
        // the place is evaluated again for its type (interp `exec`, `eval(place).ok()`)
        self.place_ticks(fx, place)?;
        let stored = self.ty.stores.get(&addr(s)).cloned().unwrap_or(RType::Never);
        let v = match (&cur_t, &v.t) {
            (RType::Int(c), RType::Int(_)) if !c.tags.has_lit() => {
                let a = self.conv(fx, v.a, &v.t, &RType::Int(IntT::of_tags(c.tags)), true)?;
                Val { a, t: stored }
            }
            (RType::Int(c), RType::Int(_)) if c.tags.names().len() > 1 => return no("an assignment to a place whose type name differs at run time"),
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
            Expr::Index(x, idx, _) if matches!(&idx[0], Expr::Name(..) | Expr::Int(..)) => {
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
        let item = match &itv.t {
            RType::List(e) | RType::Set(e) => (**e).clone(),
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
            Some(t) => self.fit(fx, v.a, &v.t, t)?.a,
        };
        fx.drops_to(None)?;
        fx.b.ret(a);
        Ok(())
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
                    if matches!(ty, ast::Ty::Ref(true, _)) {
                        return no(format!("the &mut parameter of {}", f.sig.name));
                    }
                    let rt = self.t.of_ty(ty);
                    ps.push(self.ct(&rt)?);
                }
                Param::SelfRef(true) => return no(format!("the &mut self of {}", f.sig.name)),
                Param::SelfVal | Param::SelfRef(false) => {
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
        Ok((ps, ret))
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
        let union = matches!(&f.sig.ret, Some(ast::Ty::Union(..)));
        let mut fx = self.fx(ps, ret.clone(), Ret::Fn(f.sig.ret.clone()), if union { FailCx::Union } else { FailCx::Halt }, true);
        if union {
            fx.union_t = Some(ret);
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
            fx.bound.insert(lid, Bound::Val(a.clone(), rt));
            fx.own(a);
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

    fn functions(&mut self) -> R<()> {
        let m = self.m;
        let mut todo = vec![];
        for it in &m.items {
            match it {
                Item::Fn(f) => {
                    let (ps, r) = self.fn_sig(f, None)?;
                    let id = self.f.declare(&format!("fn {}", f.sig.name), ps, r);
                    self.fns.insert(f.sig.name.clone(), id);
                    todo.push((id, f, None));
                }
                Item::Impl { ty, members, .. } => {
                    let h = crate::check::ty_head(ty);
                    let st = self.t.of_ty(ty);
                    for f in members {
                        let (ps, r) = self.fn_sig(f, Some(&st))?;
                        let id = self.f.declare(&format!("fn {h}.{}", f.sig.name), ps, r);
                        self.methods.insert((h.clone(), f.sig.name.clone()), id);
                        todo.push((id, f, Some(st.clone())));
                    }
                }
                Item::Trait { .. } => return no("a trait"),
                Item::Extern { .. } => return no("a foreign function"),
                Item::Actor { .. } | Item::Protocol { .. } | Item::Node { .. } | Item::Supervisor { .. } => return no("an actor, protocol, node or supervisor"),
                Item::TypeAlias { pred: Some(_), .. } => return no("a refined type"),
                Item::Event { foreign: Some(_), .. } | Item::Event { interrupt: Some(_), .. } => return no("a foreign or interrupt event"),
                Item::State { replicated: true, .. } => return no("a replicated state"),
                Item::Step { deadline: Some(_), .. } => return no("a step deadline"),
                _ => {}
            }
        }
        for (id, f, st) in todo {
            self.lower_fn(id, f, st.as_ref())?;
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
    /// returns none when no step applies and halts when the step refuses.
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
                // each field is fitted to its declared type; a trap refuses the event
                let mut out = vec![];
                for (k, (x, st)) in vars.iter().enumerate() {
                    fx.disown(&Atom::V(*x));
                    let a = self.take_var(&mut fx, *x);
                    let v = self.fit(&mut fx, a, st, &fields[k].ty)?;
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
    /// first refusal refuses the event and leaves the state unchanged.
    fn delta_fn(&mut self) -> R<usize> {
        let (st, _) = self.state_t()?;
        let res_t = Ty::Sum(vec![Ty::Tup(vec![st.clone(), Ty::list(Ty::Bytes)]), Ty::Bytes]);
        let mut rfs = vec![];
        for (name, fields) in self.regions() {
            let (ft, _) = self.region_t(fields)?;
            rfs.push((self.region_fn(name, fields)?, ft));
        }
        let refusal = rt::refusal(&mut self.f)?;
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
            let refused = fx.b.p("sum.tag", vec![ct.clone()], vec![c.clone()])?;
            let is1 = fx.b.int_cmp("int.eq", refused, lit(1))?;
            fx.b.open();
            {
                let h = fx.b.p("sum.get", vec![ct.clone()], vec![c.clone(), lit(1)])?;
                let txt = self.call_rt(&mut fx, refusal, vec![h]);
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
                        let v = self.fit(&mut fx, v.a, &v.t, &f.ty)?;
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
        let id = self.user_fn(&f.sig.name)?;
        let rt = self.ret_rtype(f);
        let t = self.f.ret(id);
        let w = self.f.declare(&format!("call {}", f.sig.name), vec![], t.clone());
        let mut fx = self.fx(vec![], t.clone(), Ret::None, FailCx::Halt, true);
        fx.b.tick(1);
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
    /// with the state after it or its refusal, then the head of its log.
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
                                let l = fx.b.cats(vec![Atom::bytes(format!("{k} ").as_bytes()), es, Atom::bytes(b" refused: "), q, Atom::bytes(b"\n")])?;
                                fx.b.out(l);
                                fx.b.brk(fx.depth - 1 - eb);
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
        let has_examples = m.items.iter().any(|i| matches!(i, Item::Example(_)) || matches!(i, Item::Intent(_, xs, _) if !xs.is_empty()));
        let (delta, init) = if has_examples { (Some(self.delta_fn()?), Some(self.init_state_fn()?)) } else { (None, None) };
        let mut bodies = vec![];
        let mut main = None;
        for it in &m.items {
            match it {
                Item::Const { name, .. } => bodies.push(self.obs_const(name)?),
                Item::Check(e) => bodies.push(self.obs_check(e)?),
                Item::Intent(_, xs, cks) => {
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
        let owned: Vec<(u32, Ty)> = self.consts.values().filter_map(|(id, _, v)| {
            let t = self.f.ret(*id);
            t.owned().then_some((*v, t))
        }).collect();
        let dec = rt::dec(&mut self.f)?;
        let mut wrappers = vec![];
        for (k, b) in bodies.into_iter().enumerate() {
            let id = self.f.declare(&format!("observation {}", k + 1), vec![], Ty::Bool);
            let mut fx = self.fx(vec![], Ty::Bool, Ret::None, FailCx::Halt, false);
            let lim = fx.b.let_global(self.fuel.1, Ty::u64());
            fx.b.setg(self.fuel.0, lim);
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
        let mut owned = owned;
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
