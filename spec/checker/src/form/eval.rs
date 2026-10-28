//! A Form evaluator (rce-form.md §2): the meaning of nodes, of the order
//! value and of the frame, enough for the admission vectors of FG15 and the
//! predicates of refined types. Every primitive is the core evaluator's
//! (`one::eval::prim`), so each has one implementation; only nodes, regions,
//! ω and the frame are evaluated here.
//!
//! It runs modules that passed FG1 to FG14. A case's inputs each start with
//! a fresh ω, as on a host (§5, HC-7a).
use super::ir::*;
use super::types::Tx;
use crate::one::eval::{Flow, Val, lit_val, prim};
use crate::one::ir::Ty;
use serde_json::{Value as J, json};
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

/// The reference's fuel limit (10.5).
pub const FUEL: u64 = 50_000_000;
const U64_MAX: u128 = u64::MAX as u128;
/// Node evaluations after which a run stops as an internal failure; fuel is
/// charged only where a module writes `fuel.tick`, so a loop with a large
/// count bound and no tick would otherwise run for as long as its bound.
const STEPS: u64 = 100_000_000;

enum Stop {
    Halt(Val),
    Bad(String),
}

impl From<Flow> for Stop {
    fn from(f: Flow) -> Stop {
        match f {
            Flow::Halt(v) => Stop::Halt(v),
            Flow::Bad(m) => Stop::Bad(m),
        }
    }
}

/// The working context of one input (§5): fuel, cost counters, outputs, memo cells.
struct Omega {
    fuel: u64,
    spent: u128,
    deadline: u128,
    out: Vec<J>,
    cells: BTreeMap<(bool, String, u32), Val>,
}

impl Omega {
    fn fresh() -> Omega {
        Omega { fuel: FUEL, spent: 0, deadline: U64_MAX, out: vec![], cells: BTreeMap::new() }
    }
}

fn trap(t: &str) -> Stop {
    Stop::Halt(Val::Sum(0, Box::new(Val::bytes(t.as_bytes()))))
}

fn core_ty(t: &BTy) -> Ty {
    match t {
        BTy::Bool => Ty::Bool,
        BTy::Int(lo, hi) => Ty::Int(*lo, *hi),
        BTy::F64 => Ty::F64,
        BTy::F32 => Ty::F32,
        BTy::Bytes => Ty::Bytes,
        BTy::Arr(x) => Ty::Arr(Box::new(core_ty(x))),
        BTy::List(x) => Ty::List(Box::new(core_ty(x))),
        BTy::Tup(xs) => Ty::Tup(xs.iter().map(core_ty).collect()),
        BTy::Sum(xs) => Ty::Sum(xs.iter().map(core_ty).collect()),
        // the core's primitives read only integer type arguments
        BTy::Decl(_) | BTy::Order => Ty::unit(),
    }
}

struct Evr<'a> {
    tx: Tx<'a>,
    active: HashMap<(String, u32), u128>,
    steps: u64,
}

fn omega() -> Val {
    Val::Tup(vec![])
}

impl<'a> Evr<'a> {
    fn call(&mut self, key: &str, fi: u32, args: Vec<Val>, om: &mut Omega) -> Result<Vec<Val>, Stop> {
        let f = self.tx.m.defs.get(key).and_then(|g| g.fns.get(fi as usize)).ok_or_else(|| Stop::Bad(format!("no function {key} {fi}")))?;
        let id = (key.to_string(), fi);
        if let Bound::N(n) = f.bound {
            let a = self.active.entry(id.clone()).or_default();
            if *a >= n {
                return Err(trap("recursion bound"));
            }
            *a += 1;
        }
        // deep but bounded recursion grows the stack on the heap, as the reference's does
        let r = stacker::maybe_grow(64 * 1024, 4 * 1024 * 1024, || self.region(key, &f.body, &args, om));
        if let Bound::N(_) = f.bound {
            *self.active.get_mut(&id).unwrap() -= 1;
        }
        r
    }

    fn resolve(&self, group: &str, r: &Ref) -> (String, u32) {
        match r {
            Ref::Key(k, i) => (k.clone(), *i),
            Ref::Rec(i) => (group.to_string(), *i),
        }
    }

    fn region(&mut self, group: &str, b: &Body, ins: &[Val], om: &mut Omega) -> Result<Vec<Val>, Stop> {
        let mut vals: Vec<Vec<Val>> = Vec::with_capacity(b.nodes.len());
        let get = |vals: &Vec<Vec<Val>>, s: &Src| -> Result<Val, Stop> {
            match s {
                Src::In(j) => ins.get(*j as usize).cloned(),
                Src::Out(i, p) => vals.get(*i as usize).and_then(|v| v.get(*p as usize)).cloned(),
            }
            .ok_or_else(|| Stop::Bad(format!("source {s:?} has no value")))
        };
        for n in &b.nodes {
            self.steps += 1;
            if self.steps > STEPS {
                return Err(Stop::Bad("the evaluation passed its step limit".into()));
            }
            let args = n.ins.iter().map(|s| get(&vals, s)).collect::<Result<Vec<_>, _>>()?;
            let outs = self.node(group, n, args, om)?;
            vals.push(outs);
        }
        b.out.iter().map(|s| get(&vals, s)).collect()
    }

    fn node(&mut self, group: &str, n: &Node, args: Vec<Val>, om: &mut Omega) -> Result<Vec<Val>, Stop> {
        let int = |v: &Val| match v {
            Val::Int(x) => Ok(*x),
            o => Err(Stop::Bad(format!("expected an integer, found {o:?}"))),
        };
        match &n.attrs {
            Attr::Lit(_, l) => Ok(vec![lit_val(l)]),
            Attr::Prim(name, targs, cell) => self.prim(group, name, targs, cell.as_ref(), args, om),
            Attr::Call(r) => {
                let (k, i) = self.resolve(group, r);
                self.call(&k, i, args, om)
            }
            Attr::Catch(r) => {
                let (k, i) = self.resolve(group, r);
                let f = self.tx.callee(group, r).ok_or_else(|| Stop::Bad("an unresolved catch".into()))?;
                let takes = f.outs.last() == Some(&BTy::Order);
                match self.call(&k, i, args, om) {
                    Ok(mut outs) => {
                        if takes {
                            outs.pop();
                        }
                        let mut v = vec![Val::Sum(0, Box::new(Val::Tup(outs)))];
                        if takes {
                            v.push(omega());
                        }
                        Ok(v)
                    }
                    // fuel exhaustion cannot be caught (E1)
                    Err(Stop::Halt(h)) if h == Val::Sum(0, Box::new(Val::bytes(b"comptime fuel exhausted"))) => Err(Stop::Halt(h)),
                    Err(Stop::Halt(h)) => {
                        let mut v = vec![Val::Sum(1, Box::new(h))];
                        if takes {
                            v.push(omega());
                        }
                        Ok(v)
                    }
                    Err(e) => Err(e),
                }
            }
            Attr::Cap(_) => Err(Stop::Bad("a capability cannot run here".into())),
            Attr::Branch(_, _) => {
                let k = match &args[0] {
                    Val::Bool(b) => *b as usize,
                    v => int(v)? as usize,
                };
                let r = n.regions.get(k).ok_or_else(|| Stop::Bad(format!("no arm {k}")))?;
                self.region(group, r, &args[1..], om)
            }
            Attr::Count(_) | Attr::Measure(..) => {
                let counted = matches!(n.attrs, Attr::Count(_));
                let bound = if counted { int(&args[0])? } else { 0 };
                let mut c: Vec<Val> = args[usize::from(counted)..].to_vec();
                let mut i: i128 = 0;
                let split = |mut v: Vec<Val>| -> Result<(bool, Vec<Val>), Stop> {
                    if v.is_empty() {
                        return Err(Stop::Bad("a loop region returned nothing".into()));
                    }
                    let k = v.remove(0);
                    match k {
                        Val::Bool(b) => Ok((b, v)),
                        o => Err(Stop::Bad(format!("a loop flag {o:?}"))),
                    }
                };
                loop {
                    let (k, nc) = split(self.region(group, &n.regions[0], &c, om)?)?;
                    c = nc;
                    if !k {
                        return Ok(c);
                    }
                    if counted {
                        i += 1;
                        if i > bound {
                            return Err(trap("loop bound"));
                        }
                    }
                    let (k, nc) = split(self.region(group, &n.regions[1], &c, om)?)?;
                    c = nc;
                    if !k {
                        return Ok(c);
                    }
                }
            }
            Attr::Halt(_) => Err(Stop::Halt(args.into_iter().next().unwrap_or(Val::unit()))),
        }
    }

    fn prim(&mut self, group: &str, name: &str, targs: &[TyArg], cell: Option<&Cell>, args: Vec<Val>, om: &mut Omega) -> Result<Vec<Val>, Stop> {
        let int = |k: usize| match args.get(k) {
            Some(Val::Int(x)) => Ok(*x),
            o => Err(Stop::Bad(format!("{name}: operand {k} is {o:?}"))),
        };
        let cell_id = || -> Result<(bool, String, u32), Stop> {
            let c = cell.ok_or_else(|| Stop::Bad(format!("{name} names no cell")))?;
            let (k, i) = self.resolve(group, c.r());
            Ok((matches!(c, Cell::Memo(_)), k, i))
        };
        match name {
            "emit" => {
                let Some(TyArg::Frame(t)) = targs.first() else { return Err(Stop::Bad("emit without a frame type".into())) };
                let v = to_json(self.tx.m, t, &args[0]).ok_or_else(|| Stop::Bad("emit of a value with no positional form".into()))?;
                om.out.push(v);
                Ok(vec![omega()])
            }
            "fuel.tick" => {
                let n = int(0)?;
                if (om.fuel as i128) < n {
                    om.fuel = 0;
                    return Err(trap("comptime fuel exhausted"));
                }
                om.fuel -= n as u64;
                Ok(vec![omega()])
            }
            "cost.charge" => {
                let s = (om.spent + int(0)? as u128).min(U64_MAX);
                if s > om.deadline {
                    return Err(trap("budget"));
                }
                om.spent = s;
                Ok(vec![omega()])
            }
            "cost.try_charge" => {
                if om.spent <= om.deadline {
                    om.spent = (om.spent + int(0)? as u128).min(om.deadline);
                }
                Ok(vec![omega()])
            }
            "cost.spent" => Ok(vec![Val::Int(om.spent as i128), omega()]),
            "cost.deadline" => Ok(vec![Val::Int(om.deadline as i128), omega()]),
            "cost.set_deadline" => {
                om.deadline = int(0)? as u128;
                Ok(vec![omega()])
            }
            "memo.get" => {
                let id = cell_id()?;
                Ok(vec![
                    match om.cells.remove(&id) {
                        Some(v) => Val::Sum(1, Box::new(v)),
                        None => Val::Sum(0, Box::new(Val::unit())),
                    },
                    omega(),
                ])
            }
            "memo.put" => {
                om.cells.insert(cell_id()?, args[0].clone());
                Ok(vec![omega()])
            }
            "copy" => Ok(vec![args[0].clone(), args[0].clone()]),
            _ => {
                let e = super::catalogue::entry(name).ok_or_else(|| Stop::Bad(format!("unknown entry {name}")))?;
                let tys: Vec<Ty> = targs
                    .iter()
                    .map(|a| match a {
                        TyArg::Body(t) => core_ty(t),
                        TyArg::Frame(_) => Ty::unit(),
                    })
                    .collect();
                let core = prim(name, &tys, args.clone())?;
                // the core returns no operand it reads; the Form returns it as a trailing output (§4)
                let mut out = vec![];
                let mut it = core.into_iter();
                for (of, _) in &e.results {
                    match of {
                        Some(k) if e.operands.get(*k).is_some_and(|o| o.0 == super::catalogue::Mode::ReadReturn) => out.push(args[*k].clone()),
                        _ => {
                            if e.results.len() == 1 && e.results[0].0.is_none() && matches!(e.results[0].1.get(0).and_then(|h| h.as_str()), Some("parts")) {
                                out.extend(it.by_ref());
                            } else {
                                out.push(it.next().ok_or_else(|| Stop::Bad(format!("{name} gave too few results")))?);
                            }
                        }
                    }
                }
                Ok(out)
            }
        }
    }
}

/// A positional canonical JSON value (6.1) as the value of the body type that
/// represents its frame type.
pub fn from_json(m: &Module, t: &FTy, v: &J) -> Option<Val> {
    let int = |v: &J| -> Option<i128> {
        match v {
            J::Number(n) if !n.is_f64() => n.as_i64().map(i128::from).or(n.as_u64().map(i128::from)),
            _ => None,
        }
    };
    Some(match (t, v) {
        (FTy::Bool, J::Bool(b)) => Val::Bool(*b),
        (FTy::Int(..), _) => Val::Int(int(v)?),
        (FTy::F64 | FTy::F32, J::Number(n)) if n.is_f64() => Val::F(n.as_f64()?),
        (FTy::Text, J::String(s)) => Val::bytes(s.as_bytes()),
        (FTy::Unit, J::Null) => Val::unit(),
        (FTy::Tup(xs), J::Array(a)) if xs.len() == a.len() => Val::Tup(xs.iter().zip(a).map(|(x, y)| from_json(m, x, y)).collect::<Option<_>>()?),
        (FTy::Arr(x, _), J::Array(a)) => Val::Arr(a.iter().map(|y| from_json(m, x, y)).collect::<Option<_>>()?),
        (FTy::List(x) | FTy::Set(x), J::Array(a)) => Val::List(a.iter().map(|y| from_json(m, x, y)).collect::<Option<_>>()?),
        (FTy::Map(k, w), J::Array(a)) => Val::List(
            a.iter()
                .map(|p| match p.as_array().map(|x| &x[..]) {
                    Some([x, y]) => Some(Val::Tup(vec![from_json(m, k, x)?, from_json(m, w, y)?])),
                    _ => None,
                })
                .collect::<Option<_>>()?,
        ),
        (FTy::Opt(_), J::Null) => Val::Sum(0, Box::new(Val::unit())),
        (FTy::Opt(x), _) => Val::Sum(1, Box::new(from_json(m, x, v)?)),
        (FTy::Decl(i), _) => match &m.types[*i as usize] {
            TypeDecl::Record(fs) => from_json(m, &FTy::Tup(fs.clone()), v)?,
            TypeDecl::Kind(vs) => match v.as_array().map(|a| &a[..]) {
                Some([k, J::Array(fields)]) => {
                    let k = int(k)?;
                    let fs = vs.get(usize::try_from(k).ok()?)?;
                    if fs.len() != fields.len() {
                        return None;
                    }
                    Val::Sum(k as u32, Box::new(Val::Tup(fs.iter().zip(fields).map(|(x, y)| from_json(m, x, y)).collect::<Option<_>>()?)))
                }
                _ => return None,
            },
            TypeDecl::Refined(x, _) => from_json(m, x, v)?,
        },
        _ => return None,
    })
}

/// The positional canonical JSON value (6.1) of a body value of a frame type.
pub fn to_json(m: &Module, t: &FTy, v: &Val) -> Option<J> {
    Some(match (t, v) {
        (FTy::Bool, Val::Bool(b)) => J::Bool(*b),
        (FTy::Int(..), Val::Int(x)) => {
            if *x < 0 { J::from(i64::try_from(*x).ok()?) } else { J::from(u64::try_from(*x).ok()?) }
        }
        (FTy::F64 | FTy::F32, Val::F(x)) => J::Number(serde_json::Number::from_f64(*x)?),
        (FTy::Text, Val::Bytes(b)) => J::String(String::from_utf8(b.to_vec()).ok()?),
        (FTy::Unit, Val::Tup(xs)) if xs.is_empty() => J::Null,
        (FTy::Tup(xs), Val::Tup(vs)) if xs.len() == vs.len() => {
            J::Array(xs.iter().zip(vs).map(|(x, y)| to_json(m, x, y)).collect::<Option<_>>()?)
        }
        (FTy::Arr(x, _), Val::Arr(vs)) | (FTy::List(x) | FTy::Set(x), Val::List(vs)) => J::Array(vs.iter().map(|y| to_json(m, x, y)).collect::<Option<_>>()?),
        (FTy::Map(k, w), Val::List(vs)) => J::Array(
            vs.iter()
                .map(|p| match p {
                    Val::Tup(kv) if kv.len() == 2 => Some(json!([to_json(m, k, &kv[0])?, to_json(m, w, &kv[1])?])),
                    _ => None,
                })
                .collect::<Option<_>>()?,
        ),
        (FTy::Opt(_), Val::Sum(0, _)) => J::Null,
        (FTy::Opt(x), Val::Sum(1, y)) => to_json(m, x, y)?,
        (FTy::Decl(i), _) => match (&m.types[*i as usize], v) {
            (TypeDecl::Record(fs), Val::Tup(vs)) if fs.len() == vs.len() => J::Array(fs.iter().zip(vs).map(|(x, y)| to_json(m, x, y)).collect::<Option<_>>()?),
            (TypeDecl::Kind(variants), Val::Sum(k, inner)) => {
                let fs = variants.get(*k as usize)?;
                let Val::Tup(vs) = &**inner else { return None };
                if fs.len() != vs.len() {
                    return None;
                }
                json!([k, fs.iter().zip(vs).map(|(x, y)| to_json(m, x, y)).collect::<Option<Vec<_>>>()?])
            }
            (TypeDecl::Refined(x, _), _) => to_json(m, x, v)?,
            _ => return None,
        },
        _ => return None,
    })
}

/// The predicate of a refined type on a value: its class-0 definition, run.
pub fn predicate(m: &Module, r: &Ref, t: &FTy, v: &J) -> Result<bool, String> {
    let x = from_json(m, t, v).ok_or("the value has no body representation")?;
    let Ref::Key(k, i) = r else { return Err("a predicate named by rec".into()) };
    let mut e = Evr { tx: Tx::new(m), active: HashMap::new(), steps: 0 };
    let mut om = Omega::fresh();
    match e.call(k, *i, vec![x], &mut om) {
        Ok(o) => match o.first() {
            Some(Val::Bool(b)) => Ok(*b),
            _ => Err("the predicate returned no truth value".into()),
        },
        Err(Stop::Halt(_)) => Err("the predicate halted".into()),
        Err(Stop::Bad(e)) => Err(e),
    }
}

/// The text of a halt as the kept outcome names it: a trap text, or a declared error's name.
fn halt_text(h: &Val) -> String {
    let b = match h {
        Val::Sum(0, t) => match &**t {
            Val::Bytes(b) => b.clone(),
            _ => Rc::from(&b"?"[..]),
        },
        Val::Sum(1, t) => match &**t {
            Val::Tup(xs) => match xs.first() {
                Some(Val::Bytes(b)) => b.clone(),
                _ => Rc::from(&b"?"[..]),
            },
            _ => Rc::from(&b"?"[..]),
        },
        _ => Rc::from(&b"?"[..]),
    };
    String::from_utf8_lossy(&b).into_owned()
}

/// δ for one input (§9): the next state and λ, or the kept outcome's X.
/// The next state and λ of an applied input, or the X of a kept one.
type Step = Result<(Vec<Vec<Val>>, Vec<J>), String>;

fn delta(m: &Module, state: &[Vec<Val>], ev: &Ev2, payload: &[Val]) -> Result<Step, String> {
    let mut e = Evr { tx: Tx::new(m), active: HashMap::new(), steps: 0 };
    let mut om = Omega::fresh();
    let mut next = state.to_vec();
    let run = |e: &mut Evr, r: &Ref, args: Vec<Val>, om: &mut Omega| -> Result<Result<Vec<Val>, String>, String> {
        let Ref::Key(k, i) = r else { return Err("a frame reference by rec".into()) };
        match e.call(k, *i, args, om) {
            Ok(v) => Ok(Ok(v)),
            Err(Stop::Halt(h)) => Ok(Err(halt_text(&h))),
            Err(Stop::Bad(b)) => Err(b),
        }
    };
    if let Ev2::Start = ev {
        let r = m.main.as_ref().ok_or("Start, and the module has no main")?;
        let mut args = state[0].clone();
        args.extend(payload.iter().cloned());
        args.push(omega());
        return Ok(match run(&mut e, r, args, &mut om)? {
            Ok(mut v) => {
                v.pop();
                Ok((vec![v], om.out))
            }
            Err(x) => Err(x),
        });
    }
    for (ri, fields) in state.iter().enumerate() {
        for s in &m.steps {
            if s.region as usize != ri || !ev.matches(&s.ev) {
                continue;
            }
            let mut args = fields.clone();
            args.extend(payload.iter().cloned());
            if let Some(g) = &s.guard {
                let gf = e.tx.callee("", g).ok_or("an unresolved guard")?;
                let mut ga = args.clone();
                if gf.eff >= 1 {
                    ga.push(omega());
                }
                match run(&mut e, g, ga, &mut om)? {
                    Ok(v) if v.first() == Some(&Val::Bool(true)) => {}
                    Ok(_) => continue,
                    Err(x) => return Ok(Err(x)),
                }
            }
            args.push(omega());
            match run(&mut e, &s.body, args, &mut om)? {
                Ok(mut v) => {
                    v.pop();
                    next[ri] = v;
                }
                Err(x) => return Ok(Err(x)),
            }
            break;
        }
    }
    Ok(Ok((next, om.out)))
}

enum Ev2 {
    Idx(u32),
    Tick,
    Start,
}

impl Ev2 {
    fn matches(&self, e: &Ev) -> bool {
        match (self, e) {
            (Ev2::Idx(i), Ev::Idx(j)) => i == j,
            (Ev2::Tick, Ev::Name(n)) => n == "Tick",
            _ => false,
        }
    }
}

/// The positional form of a state: the list of its regions, each the list of its field values (6.1).
fn state_json(m: &Module, s: &[Vec<Val>]) -> Option<J> {
    Some(J::Array(
        m.regions
            .iter()
            .zip(s)
            .map(|(r, vs)| r.fields.iter().zip(vs).map(|((t, _), v)| to_json(m, t, v)).collect::<Option<Vec<_>>>().map(J::Array))
            .collect::<Option<_>>()?,
    ))
}

/// FG15: the module's admission vectors (`rce-vectors/1`, §18) hold on this evaluator.
pub fn admission(m: &Module, bytes: &[u8], vectors: &[u8]) -> Result<(), (String, String)> {
    let err = |p: &str, r: String| (p.to_string(), r);
    let v: J = serde_json::from_slice(vectors).map_err(|e| err("vectors", format!("the vectors are not JSON: {e}")))?;
    if v["format"] != "rce-vectors/1" {
        return Err(err("vectors", format!("format {} is not `rce-vectors/1`", v["format"])));
    }
    let key = crate::interp::d_sha(std::str::from_utf8(bytes).unwrap_or_default());
    if v["module"] != key.as_str() {
        return Err(err("vectors", format!("the vectors name module {}, and this module's key is {key}", v["module"])));
    }
    if v["observations"].as_array().is_none_or(|o| !o.is_empty()) {
        return Err(err("vectors observations", "observations of the reference trace need the concept record, which this check does not read".into()));
    }
    let init: Vec<Vec<Val>> = m
        .regions
        .iter()
        .map(|r| r.fields.iter().map(|(t, v)| from_json(m, t, v)).collect::<Option<Vec<_>>>())
        .collect::<Option<_>>()
        .ok_or_else(|| err("regions", "an initial value has no body representation".into()))?;
    for (ci, c) in v["cases"].as_array().into_iter().flatten().enumerate() {
        let at = format!("vectors case {ci}");
        let mut state = init.clone();
        let mut outputs: Vec<J> = vec![];
        let mut outcome = "done".to_string();
        for (ii, input) in c["inputs"].as_array().into_iter().flatten().enumerate() {
            let (ev, tys) = match &input["type"] {
                J::String(n) if n == "Tick" => (Ev2::Tick, vec![FTy::Int(0, U64_MAX as i128)]),
                J::String(n) if n == "Start" && m.main.is_some() => {
                    // the payload of 4.2, or of a lifted command (§9), as main's signature says
                    let f = Tx::new(m).callee("", m.main.as_ref().unwrap()).unwrap();
                    let bytes = FTy::List(Box::new(FTy::Int(0, 255)));
                    let tys = if f.ins.len() == m.regions[0].fields.len() + 2 {
                        vec![FTy::List(Box::new(FTy::Text))]
                    } else {
                        vec![FTy::List(Box::new(bytes.clone())), FTy::List(Box::new(FTy::Tup(vec![bytes.clone(), bytes.clone()]))), bytes]
                    };
                    (Ev2::Start, tys)
                }
                x => match x.as_u64().and_then(|i| m.events.get(i as usize).map(|p| (i, p))) {
                    Some((i, p)) => (Ev2::Idx(i as u32), p.clone()),
                    None => return Err(err(&at, format!("input {ii} names no event: {x}"))),
                },
            };
            let pv = input["payload"].as_array().filter(|p| p.len() == tys.len()).ok_or_else(|| err(&at, format!("input {ii}: the payload does not match the event")))?;
            let payload = tys.iter().zip(pv).map(|(t, x)| from_json(m, t, x)).collect::<Option<Vec<_>>>().ok_or_else(|| err(&at, format!("input {ii}: the payload is not a value of the event")))?;
            match delta(m, &state, &ev, &payload).map_err(|e| err(&at, format!("input {ii}: the evaluation failed: {e}")))? {
                Ok((s, out)) => {
                    state = s;
                    outputs.extend(out);
                }
                Err(x) => {
                    outcome = format!("refused:{x}");
                    break;
                }
            }
        }
        let exp = &c["expect"];
        let got_out = J::Array(outputs);
        if exp["outputs"] != got_out {
            return Err(err(&at, format!("the outputs are {}, and the case expects {}", canon(&got_out), canon(&exp["outputs"]))));
        }
        if exp["outcome"] != outcome.as_str() {
            return Err(err(&at, format!("the outcome is {outcome}, and the case expects {}", exp["outcome"])));
        }
        if !exp["state"].is_null() {
            let sj = state_json(m, &state).ok_or_else(|| err(&at, "the state has no positional form".into()))?;
            let h = sha(&sj);
            if exp["state"] != h.as_str() {
                return Err(err(&at, format!("the state is {} with hash {h}, and the case expects {}", canon(&sj), exp["state"])));
            }
        }
    }
    Ok(())
}
