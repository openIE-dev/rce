//! The schedule and the construction (rce-form.md §13) on one slice.
//!
//! The schedule writes a canonical Form as core statements. The construction
//! reads those statements, and the same shapes when a core module uses them,
//! back into a Form. `canonical` then restores the canonical bytes, so
//! construct(schedule(F)) is F for every Form this slice schedules.
//!
//! The slice is the fragment both sides can apply and check:
//!
//! - literals, `let`, catalogue primitives (read-return and update-return
//!   included), `drop` (the schedule writes it, the construction ignores it)
//! - `call` of a non-recursive function, `if` on a bool, `var` and `set`
//! - `halt`, which makes the function take and return ω
//! - calls of `rt.tick`, `rt.charge` and `rt.try_charge`, read back as
//!   `fuel.tick`, `cost.charge` and `cost.try_charge`
//! - `loop` in the two-region shape the schedule writes: a measure, or a
//!   count bound whose test or body uses ω
//!
//! A construct outside that fragment requires change. The reason starts with
//! a tag (`global`, `out`, `carrier`, `graph`, `catch`, `recursion`, `loop`,
//! `break`, `exit`, `emit`, `memo`, `cap`, `selector`, `bound`, `branch`,
//! `omega`, `defect`).
//! This is not the production compiler: the hand emitters still walk the
//! syntax tree, and a lowered program whose loops, globals or `out` are not
//! in the slice requires change here.
use super::catalogue::{self, Mode};
use super::ir::{self as form, Attr, BTy, Body, Bound, Func, Node, Ref, Src, TyArg};
use super::types::Tx;
use crate::one::build::{self, FB};
use crate::one::ir::{self as core, Atom, FnRef, Lit, Place, Stmt, Ty};
use crate::one::json::{canonical_group, group_key};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// A construct the schedule or the construction refuses, with the place and
/// the reason.
#[derive(Clone, Debug, PartialEq)]
pub struct Change {
    /// `construction` or `schedule`
    pub phase: &'static str,
    pub at: String,
    pub reason: String,
}

impl Change {
    pub fn line(&self) -> String {
        format!(
            "requires change {} {}: {}",
            self.phase,
            if self.at.is_empty() { "-" } else { &self.at },
            self.reason
        )
    }
    /// The tag before the colon, used to group reasons.
    pub fn tag(&self) -> &str {
        self.reason
            .split(':')
            .next()
            .unwrap_or(self.reason.as_str())
    }
}

fn deny(phase: &'static str, at: &str, reason: impl Into<String>) -> Change {
    Change {
        phase,
        at: at.to_string(),
        reason: reason.into(),
    }
}

fn bty(t: &Ty) -> Result<BTy, String> {
    Ok(match t {
        Ty::Bool => BTy::Bool,
        Ty::Int(a, b) => BTy::Int(*a, *b),
        Ty::F64 => BTy::F64,
        Ty::F32 => BTy::F32,
        Ty::Bytes => BTy::Bytes,
        Ty::Arr(x) => BTy::Arr(Box::new(bty(x)?)),
        Ty::List(x) => BTy::List(Box::new(bty(x)?)),
        Ty::Tup(xs) => BTy::Tup(xs.iter().map(bty).collect::<Result<Vec<_>, _>>()?),
        Ty::Sum(xs) => BTy::Sum(xs.iter().map(bty).collect::<Result<Vec<_>, _>>()?),
    })
}

fn cty(t: &BTy) -> Result<Ty, String> {
    Ok(match t {
        BTy::Bool => Ty::Bool,
        BTy::Int(a, b) => Ty::Int(*a, *b),
        BTy::F64 => Ty::F64,
        BTy::F32 => Ty::F32,
        BTy::Bytes => Ty::Bytes,
        BTy::Arr(x) => Ty::Arr(Box::new(cty(x)?)),
        BTy::List(x) => Ty::List(Box::new(cty(x)?)),
        BTy::Tup(xs) => Ty::Tup(xs.iter().map(cty).collect::<Result<Vec<_>, _>>()?),
        BTy::Sum(xs) => Ty::Sum(xs.iter().map(cty).collect::<Result<Vec<_>, _>>()?),
        BTy::Order => return Err("order".into()),
        BTy::Decl(_) => return Err("decl".into()),
    })
}

fn ty_arg(t: &Ty) -> Result<TyArg, String> {
    Ok(TyArg::Body(bty(t)?))
}

fn fname(m: &core::Module, k: &str, i: usize) -> String {
    m.names
        .get("fns")
        .and_then(|f| f.get(k))
        .and_then(|a| a.get(i))
        .and_then(|v| v.as_str())
        .unwrap_or("?")
        .to_string()
}

/// `rt.tick`, `rt.charge`, `rt.try_charge`, including a joined concept name.
fn shim_of(name: &str) -> Option<&'static str> {
    name.split(" = ").find_map(|p| match p {
        "rt.tick" => Some("fuel.tick"),
        "rt.charge" => Some("cost.charge"),
        "rt.try_charge" => Some("cost.try_charge"),
        _ => None,
    })
}

fn has_break(b: &[Stmt], depth: u32) -> bool {
    b.iter().any(|s| match s {
        Stmt::Break(d) => *d == depth,
        Stmt::If(_, t, e) => has_break(t, depth) || has_break(e, depth),
        Stmt::Block(x) | Stmt::Loop(x) => has_break(x, depth + 1),
        _ => false,
    })
}

fn atom_locals(a: &Atom, out: &mut BTreeSet<u32>) {
    if let Atom::V(i) = a {
        out.insert(*i);
    }
}

fn stmt_locals(b: &[Stmt], out: &mut BTreeSet<u32>) {
    for s in b {
        match s {
            Stmt::Let(a) | Stmt::Var(_, a) | Stmt::Ret(a) | Stmt::Halt(a) | Stmt::Out(a) => {
                atom_locals(a, out)
            }
            Stmt::Set(Place::V(v), a) => {
                out.insert(*v);
                atom_locals(a, out);
            }
            Stmt::Set(_, a) => atom_locals(a, out),
            Stmt::Prim(_, _, xs) | Stmt::Call(_, xs) | Stmt::Catch(_, xs) | Stmt::Graph(_, xs) => {
                xs.iter().for_each(|a| atom_locals(a, out))
            }
            Stmt::If(c, t, e) => {
                atom_locals(c, out);
                stmt_locals(t, out);
                stmt_locals(e, out);
            }
            Stmt::Block(x) | Stmt::Loop(x) => stmt_locals(x, out),
            Stmt::Break(_) => {}
        }
    }
}

struct Loc {
    ty: Ty,
    var: bool,
}

struct Info {
    first: HashMap<*const Stmt, u32>,
    loc: HashMap<u32, Loc>,
    used: HashSet<u32>,
}

fn precompute(
    f: &core::Fn,
    globals: &[Ty],
    callee: &dyn Fn(&FnRef) -> Option<Ty>,
    at: &str,
) -> Result<Info, Change> {
    let mut first = HashMap::new();
    let mut loc = HashMap::new();
    for (i, t) in f.params.iter().enumerate() {
        loc.insert(i as u32, Loc {
            ty: t.clone(),
            var: false,
        });
    }
    let mut next = f.params.len() as u32;
    fn walk(
        b: &[Stmt],
        next: &mut u32,
        first: &mut HashMap<*const Stmt, u32>,
        loc: &mut HashMap<u32, Loc>,
        globals: &[Ty],
        callee: &dyn Fn(&FnRef) -> Option<Ty>,
        at: &str,
    ) -> Result<(), Change> {
        for s in b {
            first.insert(s as *const Stmt, *next);
            let bind = |loc: &mut HashMap<u32, Loc>, next: &mut u32, t: Ty, var: bool| {
                loc.insert(*next, Loc { ty: t, var });
                *next += 1;
            };
            let aty = |a: &Atom, loc: &HashMap<u32, Loc>| -> Result<Ty, Change> {
                match a {
                    Atom::V(i) => loc.get(i).map(|l| l.ty.clone()).ok_or_else(|| {
                        deny("construction", at, format!("defect: v{i} is not bound"))
                    }),
                    Atom::K(t, _) => Ok(t.clone()),
                    Atom::G(i) => globals.get(*i as usize).cloned().ok_or_else(|| {
                        deny(
                            "construction",
                            at,
                            format!("global: g{i} is not part of the working context ω holds"),
                        )
                    }),
                }
            };
            match s {
                Stmt::Let(a) => {
                    let t = aty(a, loc)?;
                    bind(loc, next, t, false);
                }
                Stmt::Var(t, _) => bind(loc, next, t.clone(), true),
                Stmt::Prim(p, ts, args) => {
                    let ops = args
                        .iter()
                        .map(|a| aty(a, loc))
                        .collect::<Result<Vec<_>, _>>()?;
                    let rs = crate::one::wf::signature(p, ts, &ops, args)
                        .map_err(|e| deny("construction", at, format!("defect: {e}")))?;
                    let n = rs.len();
                    for t in rs {
                        bind(loc, next, t, false);
                    }
                    if n != core::prim_results(p, ts) && !args.is_empty() {
                        // numbering follows prim_results; a mismatch is a defect of the walk
                    }
                    let _ = n;
                }
                Stmt::Call(r, _) | Stmt::Catch(r, _) => {
                    let t = callee(r).ok_or_else(|| {
                        deny("construction", at, "defect: a call names no function")
                    })?;
                    let t = if matches!(s, Stmt::Catch(..)) {
                        Ty::Sum(vec![t, Ty::halt()])
                    } else {
                        t
                    };
                    bind(loc, next, t, false);
                }
                Stmt::Graph(_, _) => bind(loc, next, Ty::unit(), false),
                Stmt::If(_, t, e) => {
                    walk(t, next, first, loc, globals, callee, at)?;
                    walk(e, next, first, loc, globals, callee, at)?;
                }
                Stmt::Block(x) | Stmt::Loop(x) => walk(x, next, first, loc, globals, callee, at)?,
                _ => {}
            }
        }
        Ok(())
    }
    walk(
        &f.body, &mut next, &mut first, &mut loc, globals, callee, at,
    )?;
    let mut used_set = BTreeSet::new();
    stmt_locals(&f.body, &mut used_set);
    let used = used_set.into_iter().collect();
    Ok(Info { first, loc, used })
}

#[derive(Clone)]
struct Meta {
    temp: String,
    omega: bool,
    params: Vec<BTy>,
    ret: BTy,
}

enum End {
    Fall,
    Ret(Src),
    Halt(Src),
}

struct Scope {
    nodes: Vec<Node>,
    env: HashMap<u32, Src>,
    omega: Option<Src>,
    caps: Vec<Src>,
    assigned: BTreeSet<u32>,
}

impl Scope {
    fn push(&mut self, attrs: Attr, ins: Vec<Src>, regions: Vec<Body>) -> u32 {
        let i = self.nodes.len() as u32;
        self.nodes.push(Node {
            attrs,
            ins,
            regions,
        });
        i
    }
}

struct Built {
    at: String,
    info: Info,
    /// (group, index) -> meta, for user functions
    meta: HashMap<(String, u32), Meta>,
    names: HashMap<(String, u32), String>,
    group: String,
    index: u32,
    phase: &'static str,
}

impl Built {
    fn deny(&self, reason: impl Into<String>) -> Change {
        deny(self.phase, &self.at, reason)
    }
    fn loc_ty(&self, i: u32) -> Result<Ty, Change> {
        self.info
            .loc
            .get(&i)
            .map(|l| l.ty.clone())
            .ok_or_else(|| self.deny(format!("defect: v{i} is not bound")))
    }
    fn atom_ty(&self, a: &Atom) -> Result<Ty, Change> {
        match a {
            Atom::V(i) => self.loc_ty(*i),
            Atom::K(t, _) => Ok(t.clone()),
            Atom::G(i) => Err(self.deny(format!(
                "global: g{i} is not part of the working context ω holds"
            ))),
        }
    }
    fn lit_node(&self, sc: &mut Scope, t: &Ty, l: &Lit) -> Result<Src, Change> {
        let bt = bty(t).map_err(|e| self.deny(format!("defect: {e}")))?;
        let i = sc.push(Attr::Lit(bt, l.clone()), vec![], vec![]);
        Ok(Src::Out(i, 0))
    }
    fn src_atom(&self, sc: &mut Scope, a: &Atom) -> Result<Src, Change> {
        match a {
            Atom::V(i) => sc
                .env
                .get(i)
                .copied()
                .ok_or_else(|| self.deny(format!("defect: v{i} has no source in this region"))),
            Atom::K(t, l) => self.lit_node(sc, t, l),
            Atom::G(i) => Err(self.deny(format!(
                "global: g{i} is not part of the working context ω holds"
            ))),
        }
    }
    fn seed_of(&self, arms: &[&[Stmt]], tail: &[&Stmt], parent: &Scope) -> Vec<u32> {
        let mut want = BTreeSet::new();
        for a in arms {
            stmt_locals(a, &mut want);
        }
        for s in tail {
            stmt_locals(std::slice::from_ref(*s), &mut want);
        }
        want.retain(|v| parent.env.contains_key(v));
        want.into_iter().collect()
    }

    fn need_omega(&self, sc: &mut Scope, parent: Option<Src>) -> Result<Src, Change> {
        if let Some(s) = sc.omega {
            return Ok(s);
        }
        let p = parent.ok_or_else(|| self.deny("omega: a node needs ω and the region has none"))?;
        let j = sc.caps.len() as u32;
        sc.caps.push(p);
        let s = Src::In(j);
        sc.omega = Some(s);
        Ok(s)
    }
    fn ref_name(&self, r: &FnRef) -> Result<String, Change> {
        let id = match r {
            FnRef::Key(k, i) => (k.clone(), *i),
            FnRef::Rec(i) => (self.group.clone(), *i),
        };
        if matches!(r, FnRef::Rec(_)) {
            return Err(self.deny("recursion: a recursive function needs a declared bound, and rce-core/1 does not carry one"));
        }
        self.names
            .get(&id)
            .cloned()
            .ok_or_else(|| self.deny("defect: a call names no function"))
    }
    fn exec(
        &self,
        slices: &[&[Stmt]],
        sc: &mut Scope,
        parent_omega: Option<Src>,
        ret_ty: &BTy,
    ) -> Result<End, Change> {
        let flat: Vec<&Stmt> = slices.iter().copied().flat_map(|p| p.iter()).collect();
        self.exec_refs(&flat, sc, parent_omega, ret_ty)
    }

    fn exec_refs(
        &self,
        flat: &[&Stmt],
        sc: &mut Scope,
        parent_omega: Option<Src>,
        ret_ty: &BTy,
    ) -> Result<End, Change> {
        let mut i = 0;
        while i < flat.len() {
            let s = flat[i];
            let rest = &flat[i + 1..];
            match s {
                Stmt::Let(a) => {
                    let local = self.info.first[&(s as *const Stmt)];
                    let src = self.src_atom(sc, a)?;
                    sc.env.insert(local, src);
                    i += 1;
                }
                Stmt::Var(_, a) => {
                    let local = self.info.first[&(s as *const Stmt)];
                    let src = self.src_atom(sc, a)?;
                    sc.env.insert(local, src);
                    i += 1;
                }
                Stmt::Set(Place::V(v), a) => {
                    let src = self.src_atom(sc, a)?;
                    sc.env.insert(*v, src);
                    sc.assigned.insert(*v);
                    i += 1;
                }
                Stmt::Set(Place::G(g), _) => return Err(self.deny(format!("global: g{g} is not part of the working context ω holds"))),
                Stmt::Prim(p, ts, args) if p == "drop" || p == "io.byte" || p == "mem.get" || p == "mem.set" => {
                    if p == "drop" {
                        i += 1;
                        continue;
                    }
                    return Err(self.deny(format!("carrier: `{p}` is not in the catalogue; it is a printed carrier only")));
                }
                Stmt::Prim(p, ts, args) => {
                    let local0 = self.info.first[&(s as *const Stmt)];
                    self.prim(sc, p, ts, args, local0, parent_omega)?;
                    i += 1;
                }
                Stmt::Call(r, args) | Stmt::Catch(r, args) => {
                    if matches!(s, Stmt::Catch(..)) {
                        return Err(self.deny("catch: a catch and the fuel-trap branch are not in this slice"));
                    }
                    let name = self.ref_name(r)?;
                    let local = self.info.first[&(s as *const Stmt)];
                    if let Some(entry) = shim_of(&name) {
                        if args.len() != 1 {
                            return Err(self.deny(format!("defect: {name} takes one argument")));
                        }
                        let n = self.src_atom(sc, &args[0])?;
                        let w = self.need_omega(sc, parent_omega)?;
                        let id = sc.push(Attr::Prim(entry.to_string(), vec![], None), vec![n, w], vec![]);
                        sc.omega = Some(Src::Out(id, 0));
                        if self.info.used.contains(&local) {
                            let b = self.lit_node(sc, &Ty::Bool, &Lit::Bool(true))?;
                            sc.env.insert(local, b);
                        }
                    } else {
                        let id = match r {
                            FnRef::Key(k, i) => (k.clone(), *i),
                            FnRef::Rec(_) => unreachable!(),
                        };
                        let meta = self.meta.get(&id).ok_or_else(|| self.deny("defect: a call names no user function"))?;
                        let mut ins = vec![];
                        for a in args {
                            ins.push(self.src_atom(sc, a)?);
                        }
                        if meta.omega {
                            ins.push(self.need_omega(sc, parent_omega)?);
                        }
                        let node = sc.push(Attr::Call(Ref::Key(meta.temp.clone(), 0)), ins, vec![]);
                        sc.env.insert(local, Src::Out(node, 0));
                        if meta.omega {
                            sc.omega = Some(Src::Out(node, 1));
                        }
                    }
                    i += 1;
                }
                Stmt::If(c, t, e) => {
                    if self.atom_ty(c)? != Ty::Bool {
                        return Err(self.deny("selector: the core if is not on a bool, and this slice schedules only a bool selector"));
                    }
                    let t_falls = build::falls(t);
                    let e_falls = build::falls(e);
                    let here = parent_omega.or(sc.omega);
                    if !rest.is_empty() && t_falls != e_falls {
                        let (cont, stop, cont_then) = if t_falls { (t.as_slice(), e.as_slice(), true) } else { (e.as_slice(), t.as_slice(), false) };
                        let seed = self.seed_of(&[cont, stop], rest, sc);
                        let stop_end = self.arm(stop, &seed, sc, here, ret_ty)?;
                        let cont_end = self.arm_then(cont, rest, &seed, sc, here, ret_ty)?;
                        let (e0, a0, e1, a1) = if cont_then {
                            (stop_end.0, stop_end.1, cont_end.0, cont_end.1)
                        } else {
                            (cont_end.0, cont_end.1, stop_end.0, stop_end.1)
                        };
                        let sel = self.src_atom(sc, c)?;
                        return self.finish_if(sc, sel, a0, e0, a1, e1, here, ret_ty);
                    }
                    let sel = self.src_atom(sc, c)?;
                    let seed = self.seed_of(&[t, e], &[], sc);
                    let (e1, a1) = self.arm(t, &seed, sc, here, ret_ty)?;
                    let (e0, a0) = self.arm(e, &seed, sc, here, ret_ty)?;
                    let end = self.finish_if(sc, sel, a0, e0, a1, e1, here, ret_ty)?;
                    match end {
                        End::Fall => i += 1,
                        other => return Ok(other),
                    }
                }
                Stmt::Block(b) => {
                    if has_break(b, 0) {
                        return Err(self.deny("break: a block that is not a loop, and a break out of it, are an exit this slice does not restructure"));
                    }
                    let mut nlist: Vec<&Stmt> = b.iter().collect();
                    nlist.extend(&flat[i + 1..]);
                    return self.exec_refs(&nlist, sc, parent_omega.or(sc.omega), ret_ty);
                }
                Stmt::Loop(b) => {
                    self.loop_stmt(b, s, sc, parent_omega.or(sc.omega), ret_ty)?;
                    i += 1;
                }
                Stmt::Ret(a) => {
                    let src = self.src_atom(sc, a)?;
                    return Ok(End::Ret(src));
                }
                Stmt::Halt(a) => {
                    let src = self.src_atom(sc, a)?;
                    return Ok(End::Halt(src));
                }
                Stmt::Break(_) => return Err(self.deny("break: a break outside a recognised loop is an exit this slice does not restructure")),
                Stmt::Out(_) => return Err(self.deny("out: core out is a printed carrier; emit needs the per-type runtime definition this slice does not link")),
                Stmt::Graph(_, _) => return Err(self.deny("graph: a graph term has no Form node")),
            }
        }
        Ok(End::Fall)
    }

    /// An arm. `seed` is the shared capture list, the same for both arms.
    fn arm(
        &self,
        stmts: &[Stmt],
        seed: &[u32],
        parent: &Scope,
        parent_omega: Option<Src>,
        ret_ty: &BTy,
    ) -> Result<(End, Scope), Change> {
        self.arm_then(stmts, &[], seed, parent, parent_omega, ret_ty)
    }

    fn arm_then(
        &self,
        first: &[Stmt],
        tail: &[&Stmt],
        seed: &[u32],
        parent: &Scope,
        parent_omega: Option<Src>,
        ret_ty: &BTy,
    ) -> Result<(End, Scope), Change> {
        let mut sc = Scope {
            nodes: vec![],
            env: HashMap::new(),
            omega: None,
            caps: vec![],
            assigned: BTreeSet::new(),
        };
        for (j, l) in seed.iter().enumerate() {
            sc.env.insert(*l, Src::In(j as u32));
            sc.caps.push(parent.env[l]);
        }
        // One sequence, so a domain test sinks the statements that follow it into its arm.
        let mut list: Vec<&Stmt> = first.iter().collect();
        list.extend(tail.iter().copied());
        let end = self.exec_refs(&list, &mut sc, parent_omega, ret_ty)?;
        Ok((end, sc))
    }

    fn prim(
        &self,
        sc: &mut Scope,
        p: &str,
        ts: &[Ty],
        args: &[Atom],
        local0: u32,
        parent_omega: Option<Src>,
    ) -> Result<(), Change> {
        let e = catalogue::entry(p)
            .ok_or_else(|| self.deny(format!("defect: `{p}` is not a catalogue entry")))?;
        if e.cell {
            return Err(self.deny("memo: a memo cell is not in this slice"));
        }
        if p == "emit" || p == "fuel.tick" || p.starts_with("cost.") {
            return Err(self.deny(format!("defect: `{p}` is not written as a core primitive")));
        }
        let args_ty: Vec<TyArg> = ts
            .iter()
            .map(ty_arg)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| self.deny(format!("defect: {err}")))?;
        let lits: Vec<Option<i128>> = args
            .iter()
            .map(|a| match a {
                Atom::K(Ty::Int(..), Lit::Int(v)) => Some(*v),
                _ => None,
            })
            .collect();
        let blank = empty_form();
        let tx = Tx::new(&blank);
        let ports = tx
            .operand_ports(e, &args_ty)
            .map_err(|err| self.deny(format!("defect: {p}: {err}")))?;
        let results = tx
            .results(e, &args_ty, &lits)
            .map_err(|err| self.deny(format!("defect: {p}: {err}")))?;
        let mut ins = vec![];
        for a in args {
            ins.push(self.src_atom(sc, a)?);
        }
        let id = sc.push(Attr::Prim(p.to_string(), args_ty, None), ins, vec![]);
        let mut k = 0u32;
        for (oi, (of, _)) in results.iter().enumerate() {
            let mode = of.and_then(|idx| {
                // `of` indexes the catalogue operand, which matches the spread port when no operand is `parts`
                let spread = spread_of(e, idx);
                ports.get(spread).map(|p| p.0)
            });
            match mode {
                Some(Mode::ReadReturn) => {
                    let op = (*of).unwrap();
                    if let Atom::V(v) = &args[spread_of(e, op)] {
                        sc.env.insert(*v, Src::Out(id, oi as u32));
                    }
                }
                Some(Mode::Order) => sc.omega = Some(Src::Out(id, oi as u32)),
                _ => {
                    sc.env.insert(local0 + k, Src::Out(id, oi as u32));
                    k += 1;
                }
            }
        }
        let _ = parent_omega;
        Ok(())
    }

    fn finish_if(
        &self,
        sc: &mut Scope,
        sel: Src,
        mut a0: Scope,
        e0: End,
        mut a1: Scope,
        e1: End,
        parent_omega: Option<Src>,
        ret_ty: &BTy,
    ) -> Result<End, Change> {
        let halt0 = matches!(e0, End::Halt(_));
        let halt1 = matches!(e1, End::Halt(_));
        let needs_w = halt0
            || halt1
            || a0.omega.is_some()
            || a1.omega.is_some()
            || matches!(e0, End::Ret(_)) && sc.omega.is_some() && false;
        let _ = needs_w;
        let arm_needs = |scj: &Scope, e: &End| matches!(e, End::Halt(_)) || scj.omega.is_some();
        let needs_w = arm_needs(&a0, &e0) || arm_needs(&a1, &e1);
        if needs_w {
            let p = parent_omega
                .or(sc.omega)
                .ok_or_else(|| self.deny("omega: a branch needs ω and has none"))?;
            for arm in [&mut a0, &mut a1] {
                if arm.omega.is_none() {
                    let j = arm.caps.len() as u32;
                    arm.caps.push(p);
                    arm.omega = Some(Src::In(j));
                }
            }
            if a0.caps.len() != a1.caps.len() {
                return Err(self.deny("branch: the arms capture different values, and this slice does not reconcile them"));
            }
        }
        // outputs: vars assigned in either arm, in index order, then ω
        let mut vars: BTreeSet<u32> = BTreeSet::new();
        for arm in [&a0, &a1] {
            for v in &arm.assigned {
                if sc.env.contains_key(v)
                    || self.info.loc.get(v).is_some_and(|l| l.var) && sc.env.contains_key(v)
                {
                    vars.insert(*v);
                }
            }
        }
        // an assigned local that was preseeded (in caps / env of the arm as an input) is a parent var
        vars.retain(|v| a0.env.contains_key(v) || a1.env.contains_key(v));
        vars.retain(|v| {
            // parent knows it, so it is not local to one arm only
            sc.env.contains_key(v)
        });
        let vars: Vec<u32> = vars.into_iter().collect();
        let mut ts = vec![];
        for v in &vars {
            ts.push(bty(&self.loc_ty(*v)?).map_err(|e| self.deny(format!("defect: {e}")))?);
        }
        let both_term =
            matches!(e0, End::Ret(_) | End::Halt(_)) && matches!(e1, End::Ret(_) | End::Halt(_));
        if both_term {
            // the value of a branch that ends on every path is the return, not the vars it set on the way
            ts.clear();
            ts.push(ret_ty.clone());
        }
        if needs_w {
            ts.push(BTy::Order);
        }
        if ts.is_empty()
            && !needs_w
            && vars.is_empty()
            && matches!((&e0, &e1), (End::Fall, End::Fall))
        {
            // a branch with nothing to join: keep it only when an arm has nodes that the outputs reach; otherwise drop it
            if a0.nodes.is_empty() && a1.nodes.is_empty() {
                return Ok(End::Fall);
            }
        }
        let value_out = |arm: &Scope, e: &End, v: u32| -> Result<Src, Change> {
            if arm.assigned.contains(&v) {
                arm.env
                    .get(&v)
                    .copied()
                    .ok_or_else(|| self.deny("defect: a set var has no source"))
            } else {
                arm.env
                    .get(&v)
                    .copied()
                    .ok_or_else(|| self.deny("defect: a branch output is not in the arm"))
            }
            .or_else(|_| match e {
                _ => arm
                    .env
                    .get(&v)
                    .copied()
                    .ok_or_else(|| self.deny("defect: a branch output is not in the arm")),
            })
        };
        let _ = value_out;
        let mut seal = |arm: &mut Scope, e: End| -> Result<Vec<Src>, Change> {
            match e {
                End::Halt(v) => {
                    let w = self.need_omega(arm, parent_omega.or(sc.omega))?;
                    let tys = if ts.is_empty() {
                        vec![ret_ty.clone(), BTy::Order]
                    } else {
                        ts.clone()
                    };
                    let i = arm.push(Attr::Halt(tys.clone()), vec![v, w], vec![]);
                    Ok((0..tys.len() as u32).map(|p| Src::Out(i, p)).collect())
                }
                End::Ret(src) => {
                    let mut o = vec![src];
                    if needs_w {
                        o.push(
                            arm.omega
                                .ok_or_else(|| self.deny("defect: a returning arm has no ω"))?,
                        );
                    }
                    Ok(o)
                }
                End::Fall => {
                    let mut o = vec![];
                    if vars.is_empty() && ts.first() == Some(ret_ty) {
                        return Err(
                            self.deny("exit: a branch arm falls through where the other returns")
                        );
                    }
                    for v in &vars {
                        let s = if arm.assigned.contains(v) {
                            arm.env.get(v).copied()
                        } else {
                            arm.env.get(v).copied()
                        };
                        o.push(
                            s.ok_or_else(|| self.deny("defect: a joined var is not in the arm"))?,
                        );
                    }
                    if needs_w {
                        o.push(arm.omega.ok_or_else(|| {
                            self.deny("defect: an arm of an effectful branch has no ω")
                        })?);
                    }
                    Ok(o)
                }
            }
        };
        let nout = ts.len();
        let outs0 = seal(&mut a0, e0)?;
        let outs1 = seal(&mut a1, e1)?;
        if outs0.len() != nout || outs1.len() != nout {
            return Err(self.deny(format!(
                "defect: branch arms return {}, {} values for {nout} outputs",
                outs0.len(),
                outs1.len()
            )));
        }
        if a0.caps != a1.caps {
            return Err(self.deny(
                "branch: the arms capture different values, and this slice does not reconcile them",
            ));
        }
        let caps = a0.caps.clone();
        let mut ins = vec![sel];
        ins.extend(caps);
        let regions = vec![
            Body {
                nodes: a0.nodes,
                out: outs0,
            },
            Body {
                nodes: a1.nodes,
                out: outs1,
            },
        ];
        let id = sc.push(Attr::Branch(2, ts), ins, regions);
        if !both_term {
            for (p, v) in vars.iter().enumerate() {
                sc.env.insert(*v, Src::Out(id, p as u32));
            }
        }
        if needs_w {
            sc.omega = Some(Src::Out(id, (nout - 1) as u32));
        }
        if both_term {
            return Ok(End::Ret(Src::Out(id, 0)));
        }
        Ok(End::Fall)
    }

    fn loop_stmt(
        &self,
        body: &[Stmt],
        stmt: &Stmt,
        sc: &mut Scope,
        parent_omega: Option<Src>,
        ret_ty: &BTy,
    ) -> Result<(), Change> {
        let lo = self.info.first[&(stmt as *const Stmt)];
        let Some(parts) = split_loop(body) else {
            return Err(self.deny(loop_reason(body, &self.info, lo)));
        };
        let carried = parts.carried.clone();
        let count = parse_count(&parts.middle, &carried, lo, &self.info);
        if parts.middle_looks_like_count && count.is_none() {
            return Err(
                self.deny("loop: a loop has a counter but not the count shape the schedule writes")
            );
        }
        let (mid_body, count) = match count {
            Some((c, n)) => (&parts.middle[n..], Some(c)),
            None => (parts.middle, None),
        };
        let needs_w_parent = parent_omega.is_some() || sc.omega.is_some();
        let _ = (needs_w_parent, ret_ty);
        let mut build_region = |stmts: &[Stmt],
                                flag: &Atom,
                                sets: &[(u32, Atom)]|
         -> Result<Scope, Change> {
            let mut region = Scope {
                nodes: vec![],
                env: HashMap::new(),
                omega: None,
                caps: vec![],
                assigned: BTreeSet::new(),
            };
            let mut ix = 0u32;
            // ω is threaded only when the surrounding function has it; the region receives it first
            let thread = sc.omega.is_some() || parent_omega.is_some();
            if thread {
                let p = sc
                    .omega
                    .or(parent_omega)
                    .ok_or_else(|| self.deny("defect: a loop has no ω to thread"))?;
                region.omega = Some(Src::In(0));
                region.caps.push(p);
                ix = 1;
            }
            for v in &carried {
                let p = sc.env.get(v).copied().ok_or_else(|| {
                    self.deny("defect: a carried var has no source before the loop")
                })?;
                region.env.insert(*v, Src::In(ix));
                region.caps.push(p);
                ix += 1;
            }
            let parent_w = region.omega;
            let end = self.exec(&[stmts], &mut region, parent_w, ret_ty)?;
            match end {
                End::Ret(_) => {
                    return Err(
                        self.deny("exit: a nested ret is an exit this slice does not restructure")
                    );
                }
                End::Halt(v) => {
                    let w = self.need_omega(&mut region, sc.omega.or(parent_omega))?;
                    let mut tys = vec![BTy::Bool];
                    if region.caps.first().is_some() && region.omega == Some(Src::In(0)) && thread {
                        tys.push(BTy::Order);
                    }
                    for v in &carried {
                        tys.push(
                            bty(&self.loc_ty(*v)?)
                                .map_err(|e| self.deny(format!("defect: {e}")))?,
                        );
                    }
                    let i = region.push(Attr::Halt(tys.clone()), vec![v, w], vec![]);
                    region.assigned.clear();
                    // stash the outs in assigned? store on caps' parallel by reusing env key u32::MAX
                    let _ = i;
                    region.env.insert(u32::MAX, Src::Out(i, 0));
                    return Ok(region);
                }
                End::Fall => {}
            }
            let flag_s = self.src_atom(&mut region, flag)?;
            region.env.insert(u32::MAX - 1, flag_s);
            for (k, (v, a)) in sets.iter().enumerate() {
                let s = self.src_atom(&mut region, a)?;
                region.env.insert(10_000_000 + k as u32, s);
                let _ = v;
            }
            Ok(region)
        };
        let mut test = build_region(parts.test, &parts.flag0, &parts.sets0)?;
        let mut bod = build_region(mid_body, &parts.flag1, &parts.sets1)?;
        let thread = test.omega.is_some() || bod.omega.is_some() || sc.omega.is_some();
        if thread {
            for region in [&mut test, &mut bod] {
                if region.omega.is_none() {
                    let p = sc
                        .omega
                        .or(parent_omega)
                        .ok_or_else(|| self.deny("defect: a loop region has no ω"))?;
                    // inputs were built without ω; refuse rather than renumber
                    if !region.caps.is_empty() || !region.nodes.is_empty() {
                        return Err(self.deny("bound: a count-bounded Iterate must carry ω, and this loop's regions do not both receive it"));
                    }
                    region.caps.insert(0, p);
                    region.omega = Some(Src::In(0));
                }
            }
        }
        let omega_used = |region: &Scope| -> bool {
            let Some(Src::In(0)) = region.omega else {
                return region.omega.is_some();
            };
            if region.env.contains_key(&u32::MAX) {
                return true; // sealed halt
            }
            fn touch(nodes: &[Node], inn: Src) -> bool {
                nodes
                    .iter()
                    .any(|n| n.ins.contains(&inn) || n.regions.iter().any(|r| touch(&r.nodes, inn)))
            }
            touch(&region.nodes, Src::In(0)) || region.omega != Some(Src::In(0))
        };
        if count.is_some() && !omega_used(&test) && !omega_used(&bod) {
            return Err(self.deny("bound: a count-bounded Iterate must carry ω, and C7 removes a carried ω that neither region uses, so this pure count loop has no applied form"));
        }
        let finish = |region: &mut Scope, sets: &[(u32, Atom)]| -> Result<Body, Change> {
            if let Some(h) = region.env.get(&u32::MAX).copied() {
                // halt node already pushed; its outputs are the region's outs
                let n = region.nodes.len() - 1;
                let tys = match &region.nodes[n].attrs {
                    Attr::Halt(ts) => ts.len(),
                    _ => 0,
                };
                let out = (0..tys as u32).map(|p| Src::Out(n as u32, p)).collect();
                let _ = h;
                return Ok(Body {
                    nodes: std::mem::take(&mut region.nodes),
                    out,
                });
            }
            let mut out = vec![
                *region
                    .env
                    .get(&(u32::MAX - 1))
                    .ok_or_else(|| self.deny("defect: a loop region has no flag"))?,
            ];
            if region.omega == Some(Src::In(0)) || matches!(region.omega, Some(Src::Out(..))) {
                // ω is an output when it is an input
                if region.caps.first().is_some() && matches!(region.omega, Some(_)) && thread {
                    out.push(region.omega.unwrap());
                }
            }
            for k in 0..sets.len() {
                out.push(
                    *region
                        .env
                        .get(&(10_000_000 + k as u32))
                        .ok_or_else(|| self.deny("defect: a carried result is missing"))?,
                );
            }
            Ok(Body {
                nodes: std::mem::take(&mut region.nodes),
                out,
            })
        };
        // Recompute outs with a consistent ω bit.
        let use_w = test.omega.is_some()
            && bod.omega.is_some()
            && (count.is_some() || omega_used(&test) || omega_used(&bod));
        if count.is_some() && !use_w {
            return Err(self.deny("bound: a count-bounded Iterate must carry ω, and C7 removes a carried ω that neither region uses, so this pure count loop has no applied form"));
        }
        let mut carried_ty = vec![];
        if use_w {
            carried_ty.push(BTy::Order);
        }
        for v in &carried {
            carried_ty.push(bty(&self.loc_ty(*v)?).map_err(|e| self.deny(format!("defect: {e}")))?);
        }
        let body_of =
            |region: &mut Scope, sets: &[(u32, Atom)]| -> Result<Body, Change> {
                if region.env.contains_key(&u32::MAX) {
                    let n = region.nodes.len() - 1;
                    let wide = 1 + carried_ty.len();
                    // rewrite the halt node's type list to the region's outputs
                    if let Attr::Halt(ts) = &mut region.nodes[n].attrs {
                        *ts = std::iter::once(BTy::Bool)
                            .chain(carried_ty.iter().cloned())
                            .collect();
                    }
                    let out = (0..wide as u32).map(|p| Src::Out(n as u32, p)).collect();
                    return Ok(Body {
                        nodes: std::mem::take(&mut region.nodes),
                        out,
                    });
                }
                let mut out =
                    vec![*region.env.get(&(u32::MAX - 1)).ok_or_else(|| {
                        self.deny("loop: a loop region does not produce its flag")
                    })?];
                if use_w {
                    out.push(
                        region
                            .omega
                            .ok_or_else(|| self.deny("defect: a loop region lost ω"))?,
                    );
                }
                for k in 0..sets.len() {
                    out.push(
                        *region
                            .env
                            .get(&(10_000_000 + k as u32))
                            .ok_or_else(|| self.deny("loop: a carried value is not returned"))?,
                    );
                }
                Ok(Body {
                    nodes: std::mem::take(&mut region.nodes),
                    out,
                })
            };
        let _ = finish;
        let test_b = body_of(&mut test, &parts.sets0)?;
        let body_b = body_of(&mut bod, &parts.sets1)?;
        let (attrs, mut ins) = if let Some(c) = count {
            let bound = self.src_atom(sc, &c.bound)?;
            let mut ins = vec![bound];
            if use_w {
                ins.push(
                    sc.omega
                        .or(parent_omega)
                        .ok_or_else(|| self.deny("defect: count has no ω"))?,
                );
            }
            for v in &carried {
                ins.push(*sc.env.get(v).ok_or_else(|| self.deny("defect: carried"))?);
            }
            (Attr::Count(carried_ty.clone()), ins)
        } else {
            let mut ins = vec![];
            if use_w {
                ins.push(sc.omega.unwrap());
            }
            for v in &carried {
                ins.push(sc.env[v]);
            }
            let (i, j) = measure_ij(&test_b).map_err(|m| self.deny(m))?;
            (Attr::Measure(i, j, carried_ty.clone()), ins)
        };
        let id = sc.push(attrs, std::mem::take(&mut ins), vec![test_b, body_b]);
        let base = u32::from(use_w);
        for (k, v) in carried.iter().enumerate() {
            sc.env.insert(*v, Src::Out(id, base + k as u32));
        }
        if use_w {
            sc.omega = Some(Src::Out(id, 0));
        }
        Ok(())
    }
}

fn spread_of(e: &catalogue::Entry, of: usize) -> usize {
    let mut s = 0;
    for (i, (_, p)) in e.operands.iter().enumerate() {
        if i == of {
            return s;
        }
        if p.get(0).and_then(|x| x.as_str()) == Some("parts") {
            // no read-return entry uses parts; keep the index
        }
        s += 1;
    }
    of
}

fn measure_ij(test: &Body) -> Result<(u32, u32), String> {
    let flag = *test.out.first().ok_or("measure: the test has no flag")?;
    let Src::Out(i, 0) = flag else {
        return Err("measure: the test flag is not a node".into());
    };
    let n = test
        .nodes
        .get(i as usize)
        .ok_or("measure: the flag names no node")?;
    let Attr::Prim(name, _, _) = &n.attrs else {
        return Err("measure: the test flag is not int.lt of two carried values".into());
    };
    if name == "int.lt"
        && let [Src::In(a), Src::In(b)] = n.ins[..]
    {
        return Ok((a, b));
    }
    Err("measure: the test flag is not int.lt of two carried values".into())
}

struct CountP {
    bound: Atom,
}

struct Parts<'a> {
    test: &'a [Stmt],
    middle: &'a [Stmt],
    carried: Vec<u32>,
    flag0: Atom,
    flag1: Atom,
    sets0: Vec<(u32, Atom)>,
    sets1: Vec<(u32, Atom)>,
    middle_looks_like_count: bool,
}

fn break_if(s: &Stmt) -> Option<Atom> {
    match s {
        Stmt::If(c, t, e) if t.is_empty() && matches!(e.as_slice(), [Stmt::Break(0)]) => {
            Some(c.clone())
        }
        _ => None,
    }
}

fn split_loop(body: &[Stmt]) -> Option<Parts<'_>> {
    let breaks: Vec<usize> = body
        .iter()
        .enumerate()
        .filter_map(|(i, s)| break_if(s).map(|_| i))
        .collect();
    if breaks.len() != 2 {
        return None;
    }
    let epi = |at: usize| -> Option<(usize, Atom, Vec<(u32, Atom)>)> {
        let flag = break_if(&body[at])?;
        let mut sets = vec![];
        let mut k = at;
        while k > 0 {
            match &body[k - 1] {
                Stmt::Set(Place::V(v), a) => {
                    sets.push((*v, a.clone()));
                    k -= 1;
                }
                _ => break,
            }
        }
        sets.reverse();
        if sets.is_empty() {
            return None;
        }
        Some((k, flag, sets))
    };
    let (s0, f0, sets0) = epi(breaks[0])?;
    let (s1, f1, sets1) = epi(breaks[1])?;
    let c0: Vec<u32> = sets0.iter().map(|s| s.0).collect();
    let c1: Vec<u32> = sets1.iter().map(|s| s.0).collect();
    if c0 != c1 {
        return None;
    }
    let middle = &body[breaks[0] + 1..s1];
    let looks = matches!(middle.first(), Some(Stmt::Let(Atom::V(_))));
    Some(Parts {
        test: &body[..s0],
        middle,
        carried: c0,
        flag0: f0,
        flag1: f1,
        sets0,
        sets1,
        middle_looks_like_count: looks,
    })
}

fn parse_count(middle: &[Stmt], carried: &[u32], lo: u32, info: &Info) -> Option<(CountP, usize)> {
    let mut i = 0;
    let Stmt::Let(Atom::V(n)) = middle.get(i)? else {
        return None;
    };
    let n = *n;
    if n >= lo || !info.loc.get(&n).is_some_and(|l| l.var) || carried.contains(&n) {
        return None;
    }
    let stmt = &middle[i];
    let a = info.first.get(&(stmt as *const Stmt)).copied()?;
    i += 1;
    if let Some(Stmt::Prim(p, _, xs)) = middle.get(i)
        && p == "int.bad.add"
        && xs.len() == 2
    {
        let stmt = &middle[i];
        let q = *info.first.get(&(stmt as *const Stmt))?;
        i += 1;
        match middle.get(i) {
            Some(Stmt::If(Atom::V(v), t, e)) if *v == q && e.is_empty() && !build::falls(t) => {
                i += 1
            }
            _ => return None,
        }
    }
    let Stmt::Prim(op, _, xs) = middle.get(i)? else {
        return None;
    };
    if op != "int.sadd" && op != "int.add" {
        return None;
    }
    let one = |xs: &[Atom]| {
        xs.len() == 2 && xs[0] == Atom::V(a) && matches!(xs[1], Atom::K(Ty::Int(..), Lit::Int(1)))
    };
    if !one(xs) {
        return None;
    }
    let stmt = &middle[i];
    let c = *info.first.get(&(stmt as *const Stmt))?;
    i += 1;
    if middle.get(i) != Some(&Stmt::Set(Place::V(n), Atom::V(c))) {
        return None;
    }
    i += 1;
    let Stmt::Prim(lt, _, ys) = middle.get(i)? else {
        return None;
    };
    if lt != "int.lt" || ys.len() != 2 || ys[1] != Atom::V(c) {
        return None;
    }
    let bound = ys[0].clone();
    let stmt = &middle[i];
    let o = *info.first.get(&(stmt as *const Stmt))?;
    i += 1;
    match middle.get(i)? {
        Stmt::If(Atom::V(v), t, e) if *v == o && e.is_empty() && arm_is_loop_bound(t) => {}
        _ => return None,
    }
    i += 1;
    Some((CountP { bound }, i))
}

fn arm_is_loop_bound(b: &[Stmt]) -> bool {
    !build::falls(b)
        && b.iter().any(|s| match s {
            Stmt::Halt(_) => true,
            Stmt::Prim(_, _, xs) => xs
                .iter()
                .any(|a| matches!(a, Atom::K(_, Lit::Bytes(v)) if v.as_slice() == b"loop bound")),
            Stmt::If(_, t, e) => arm_is_loop_bound(t) || arm_is_loop_bound(e),
            Stmt::Block(x) => arm_is_loop_bound(x),
            _ => false,
        })
}

fn loop_reason(_body: &[Stmt], _info: &Info, _lo: u32) -> String {
    "loop: a loop has none of the two-region shapes the schedule writes (a measure, or a count bound). A lowering loop with a different statement order requires change".into()
}

fn empty_form() -> form::Module {
    form::Module {
        catalogue: catalogue::key().into(),
        types: vec![],
        regions: vec![],
        events: vec![],
        errors: vec![],
        steps: vec![],
        main: None,
        caps: vec![],
        defs: BTreeMap::new(),
    }
}

fn direct_omega(f: &core::Fn, names: &HashMap<(String, u32), String>, group: &str) -> bool {
    fn go(b: &[Stmt], names: &HashMap<(String, u32), String>, group: &str) -> bool {
        b.iter().any(|s| match s {
            Stmt::Halt(_) => true,
            Stmt::Call(r, _) | Stmt::Catch(r, _) => {
                let id = match r {
                    FnRef::Key(k, i) => (k.clone(), *i),
                    FnRef::Rec(i) => (group.to_string(), *i),
                };
                names.get(&id).is_some_and(|n| shim_of(n).is_some())
            }
            Stmt::If(_, t, e) => go(t, names, group) || go(e, names, group),
            Stmt::Block(x) | Stmt::Loop(x) => go(x, names, group),
            _ => false,
        })
    }
    go(&f.body, names, group)
}

struct UserFn {
    group: String,
    index: u32,
    name: String,
    func: core::Fn,
}

/// Build a canonical Form from a well-formed core module, or the first
/// construct that requires change.
pub fn construct(m: &core::Module) -> Result<form::Module, Change> {
    let users = users_of(m);
    let mut names = HashMap::new();
    for u in &users {
        names.insert((u.group.clone(), u.index), u.name.clone());
    }
    // also record shim names so calls resolve
    for (k, g) in &m.defs {
        for (i, _) in g.fns.iter().enumerate() {
            names
                .entry((k.clone(), i as u32))
                .or_insert_with(|| fname(m, k, i));
        }
    }
    let mut omega: HashMap<(String, u32), bool> = HashMap::new();
    for u in &users {
        omega.insert(
            (u.group.clone(), u.index),
            direct_omega(&u.func, &names, &u.group),
        );
    }
    // calls of a function that needs ω need ω too
    let edges: Vec<((String, u32), (String, u32))> = users
        .iter()
        .flat_map(|u| {
            let mut cs = vec![];
            calls_of(&u.func.body, &mut cs);
            cs.into_iter().filter_map(move |r| match r {
                FnRef::Key(k, i) => Some(((u.group.clone(), u.index), (k, i))),
                FnRef::Rec(_) => None,
            })
        })
        .collect();
    loop {
        let mut changed = false;
        for (caller, callee) in &edges {
            if omega.get(callee).copied().unwrap_or(false)
                && !omega.get(caller).copied().unwrap_or(false)
            {
                omega.insert(caller.clone(), true);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut meta = HashMap::new();
    for (n, u) in users.iter().enumerate() {
        let ret =
            bty(&u.func.ret).map_err(|e| deny("construction", &u.name, format!("defect: {e}")))?;
        let params = u
            .func
            .params
            .iter()
            .map(bty)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| deny("construction", &u.name, format!("defect: {e}")))?;
        meta.insert((u.group.clone(), u.index), Meta {
            temp: format!("t{n:04}"),
            omega: omega[&(u.group.clone(), u.index)],
            params,
            ret,
        });
    }
    let globals: Vec<Ty> = m.globals.iter().map(|g| g.0.clone()).collect();
    let mut defs = BTreeMap::new();
    for u in &users {
        let at = u.name.clone();
        let callee = |r: &FnRef| -> Option<Ty> {
            match r {
                FnRef::Key(k, i) => m.func(k, *i).map(|f| f.ret.clone()),
                FnRef::Rec(i) => m.func(&u.group, *i).map(|f| f.ret.clone()),
            }
        };
        let info = precompute(&u.func, &globals, &callee, &at)?;
        let built = Built {
            at: at.clone(),
            info,
            meta: meta.clone(),
            names: names.clone(),
            group: u.group.clone(),
            index: u.index,
            phase: "construction",
        };
        let me = &meta[&(u.group.clone(), u.index)];
        let mut sc = Scope {
            nodes: vec![],
            env: HashMap::new(),
            omega: None,
            caps: vec![],
            assigned: BTreeSet::new(),
        };
        let mut ins = me.params.clone();
        for (i, _) in me.params.iter().enumerate() {
            sc.env.insert(i as u32, Src::In(i as u32));
        }
        if me.omega {
            let j = me.params.len() as u32;
            sc.omega = Some(Src::In(j));
            ins.push(BTy::Order);
        }
        let end = built.exec(&[&u.func.body], &mut sc, None, &me.ret)?;
        let (out, outs, eff) = match end {
            End::Ret(src) => {
                let mut out = vec![src];
                let mut outs = vec![me.ret.clone()];
                if me.omega {
                    let w = sc.omega.ok_or_else(|| {
                        deny(
                            "construction",
                            &at,
                            "defect: the function needs ω and does not return it",
                        )
                    })?;
                    out.push(w);
                    outs.push(BTy::Order);
                }
                (out, outs, u8::from(me.omega))
            }
            End::Halt(src) => {
                let w = built.need_omega(&mut sc, None)?;
                let tys = vec![me.ret.clone(), BTy::Order];
                let i = sc.push(Attr::Halt(tys.clone()), vec![src, w], vec![]);
                let out = vec![Src::Out(i, 0), Src::Out(i, 1)];
                if !ins.last().is_some_and(|t| *t == BTy::Order) {
                    ins.push(BTy::Order);
                }
                (out, tys, 1)
            }
            End::Fall => {
                return Err(deny(
                    "construction",
                    &at,
                    "exit: a path ends without ret or halt",
                ));
            }
        };
        let func = Func {
            ins,
            outs,
            eff,
            bound: Bound::None,
            body: Body {
                nodes: sc.nodes,
                out,
            },
        };
        defs.insert(me.temp.clone(), form::Group { fns: vec![func] });
    }
    let module = form::Module {
        catalogue: catalogue::key().into(),
        types: vec![],
        regions: vec![],
        events: vec![],
        errors: vec![],
        steps: vec![],
        main: None,
        caps: vec![],
        defs,
    };
    let (module, _) = super::canon::canonical(&module).map_err(|e| {
        deny(
            "construction",
            "",
            format!("defect: the canonical passes did not apply: {}", e.0),
        )
    })?;
    let bytes = form::encode(&module);
    match super::check(bytes.as_bytes(), None) {
        super::Outcome::Applied(_) => super::decode::decode(bytes.as_bytes())
            .map_err(|k| deny("construction", &k.path, format!("{}: {}", k.rule, k.reason))),
        super::Outcome::RequiresChange(k) => Err(deny(
            "construction",
            &k.path,
            format!("{}: {}", k.rule, k.reason),
        )),
    }
}

fn users_of(m: &core::Module) -> Vec<UserFn> {
    let mut v = vec![];
    for (k, g) in &m.defs {
        for (i, f) in g.fns.iter().enumerate() {
            let name = fname(m, k, i);
            if shim_of(&name).is_some() {
                continue;
            }
            v.push(UserFn {
                group: k.clone(),
                index: i as u32,
                name,
                func: f.clone(),
            });
        }
    }
    v
}

fn calls_of(b: &[Stmt], out: &mut Vec<FnRef>) {
    for s in b {
        match s {
            Stmt::Call(r, _) | Stmt::Catch(r, _) => out.push(r.clone()),
            Stmt::If(_, t, e) => {
                calls_of(t, out);
                calls_of(e, out);
            }
            Stmt::Block(x) | Stmt::Loop(x) => calls_of(x, out),
            _ => {}
        }
    }
}

/// Every user function of `m`, translated on its own closure, with the first
/// reason it requires change. A shim (`rt.tick`, `rt.charge`, `rt.try_charge`)
/// is not a user function.
pub fn functions(m: &core::Module) -> Vec<(String, Result<(), Change>)> {
    let users = users_of(m);
    let mut by_id: HashMap<(String, u32), usize> = HashMap::new();
    for (n, u) in users.iter().enumerate() {
        by_id.insert((u.group.clone(), u.index), n);
    }
    let mut need: Vec<Vec<usize>> = vec![vec![]; users.len()];
    for (n, u) in users.iter().enumerate() {
        let mut cs = vec![];
        calls_of(&u.func.body, &mut cs);
        for r in cs {
            if let FnRef::Key(k, i) = r
                && let Some(j) = by_id.get(&(k, i))
            {
                need[n].push(*j);
            }
        }
    }
    let mut out = vec![];
    for (n, u) in users.iter().enumerate() {
        let mut seen = HashSet::new();
        let mut stack = vec![n];
        let mut ids = vec![];
        while let Some(x) = stack.pop() {
            if !seen.insert(x) {
                continue;
            }
            ids.push(x);
            stack.extend(&need[x]);
        }
        ids.sort();
        let sub = submodule(m, &users, &ids);
        out.push((u.name.clone(), construct(&sub).map(|_| ())));
    }
    out
}

fn submodule(m: &core::Module, users: &[UserFn], ids: &[usize]) -> core::Module {
    let mut defs = BTreeMap::new();
    let mut names_f = serde_json::Map::new();
    for id in ids {
        let u = &users[*id];
        defs.entry(u.group.clone())
            .or_insert_with(|| core::Group { fns: vec![] });
        // functions in a group must keep their indices; copy the whole group once
    }
    let mut groups: BTreeSet<String> = BTreeSet::new();
    for id in ids {
        groups.insert(users[*id].group.clone());
    }
    for g in &groups {
        let group = m.defs.get(g).unwrap().clone();
        let mut arr = vec![];
        for (i, _) in group.fns.iter().enumerate() {
            arr.push(serde_json::Value::from(fname(m, g, i)));
        }
        names_f.insert(g.clone(), serde_json::Value::Array(arr));
        defs.insert(g.clone(), group);
    }
    // shims the closure calls
    for (k, g) in &m.defs {
        if groups.contains(k) {
            continue;
        }
        let mut take = false;
        for (i, _) in g.fns.iter().enumerate() {
            if shim_of(&fname(m, k, i)).is_some() {
                take = true;
            }
        }
        if take {
            let mut arr = vec![];
            for (i, _) in g.fns.iter().enumerate() {
                arr.push(serde_json::Value::from(fname(m, k, i)));
            }
            names_f.insert(k.clone(), serde_json::Value::Array(arr));
            defs.insert(k.clone(), g.clone());
        }
    }
    let entry = (users[ids[0]].group.clone(), users[ids[0]].index);
    core::Module {
        globals: m.globals.clone(),
        defs,
        entry,
        names: serde_json::json!({"fns": names_f, "globals": m.names.get("globals").cloned().unwrap_or(serde_json::json!([]))}),
        graphs: m.graphs.clone(),
    }
}

// ---- schedule

#[derive(Clone)]
enum Av {
    A(Atom),
    Omega,
}

struct Sch {
    next: u32,
    stmts: Vec<Stmt>,
    tys: HashMap<u32, Ty>,
    live: BTreeSet<u32>,
    tick: bool,
    charge: bool,
    try_c: bool,
    at: String,
}

impl Sch {
    fn deny(&self, reason: impl Into<String>) -> Change {
        deny("schedule", &self.at, reason)
    }
    fn fresh(&mut self, t: Ty, owned_live: bool) -> u32 {
        let i = self.next;
        self.next += 1;
        self.tys.insert(i, t.clone());
        if owned_live && t.owned() {
            self.live.insert(i);
        }
        i
    }
    fn let_atom(&mut self, a: Atom, t: Ty) -> Result<u32, Change> {
        self.consume(&a)?;
        self.stmts.push(Stmt::Let(a));
        Ok(self.fresh(t, true))
    }
    fn consume(&mut self, a: &Atom) -> Result<(), Change> {
        if let Atom::V(i) = a
            && self.tys.get(i).is_some_and(|t| t.owned())
        {
            if !self.live.remove(i) {
                return Err(self.deny(format!("defect: v{i} is used after it was consumed")));
            }
        }
        Ok(())
    }
    fn drop_except(&mut self, keep: &HashSet<u32>, from: u32) {
        let gone: Vec<u32> = self
            .live
            .iter()
            .copied()
            .filter(|i| *i >= from && !keep.contains(i))
            .collect();
        for i in gone {
            let t = self.tys[&i].clone();
            self.stmts
                .push(Stmt::Prim("drop".into(), vec![t], vec![Atom::V(i)]));
            self.live.remove(&i);
        }
    }
    fn val(&mut self, env: &HashMap<Src, Av>, s: Src) -> Result<Av, Change> {
        env.get(&s)
            .cloned()
            .ok_or_else(|| self.deny(format!("defect: {s:?} has no core value")))
    }
    fn atom(&mut self, env: &HashMap<Src, Av>, s: Src) -> Result<Atom, Change> {
        match self.val(env, s)? {
            Av::A(a) => Ok(a),
            Av::Omega => Err(self.deny("defect: ω where a core value is required")),
        }
    }
}

fn schedule_func(
    m: &form::Module,
    group: &str,
    f: &Func,
    at: &str,
) -> Result<(core::Fn, bool, bool, bool), Change> {
    let (params, ret, omega) = sig_of(f).map_err(|r| deny("schedule", at, r))?;
    if f.body
        .nodes
        .iter()
        .any(|n| matches!(n.attrs, Attr::Catch(_) | Attr::Cap(_)))
        || has_kind(&f.body, |a| matches!(a, Attr::Catch(_) | Attr::Cap(_)))
    {
        return Err(deny("schedule", at, cap_or_catch(&f.body)));
    }
    let mut sch = Sch {
        next: params.len() as u32,
        stmts: vec![],
        tys: params
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, t)| (i as u32, t))
            .collect(),
        live: params
            .iter()
            .enumerate()
            .filter(|(_, t)| t.owned())
            .map(|(i, _)| i as u32)
            .collect(),
        tick: false,
        charge: false,
        try_c: false,
        at: at.to_string(),
    };
    let mut env = HashMap::new();
    let mut pi = 0u32;
    for (j, t) in f.ins.iter().enumerate() {
        if *t == BTy::Order {
            env.insert(Src::In(j as u32), Av::Omega);
        } else {
            env.insert(Src::In(j as u32), Av::A(Atom::V(pi)));
            pi += 1;
        }
    }
    let mark = sch.next;
    schedule_region(&mut sch, m, group, &f.body, &mut env, None)?;
    let mut keep = HashSet::new();
    let mut ret_atom = None;
    for s in &f.body.out {
        match sch.val(&env, *s)? {
            Av::A(Atom::V(i)) => {
                keep.insert(i);
                ret_atom = Some(Atom::V(i));
            }
            Av::A(a) => ret_atom = Some(a),
            Av::Omega => {}
        }
    }
    sch.drop_except(&keep, mark);
    // also drop owned params that are not the result
    let gone: Vec<u32> = sch
        .live
        .iter()
        .copied()
        .filter(|i| !keep.contains(i))
        .collect();
    for i in gone {
        let t = sch.tys[&i].clone();
        sch.stmts
            .push(Stmt::Prim("drop".into(), vec![t], vec![Atom::V(i)]));
        sch.live.remove(&i);
    }
    let ret_a = ret_atom.ok_or_else(|| sch.deny("defect: the function has no core result"))?;
    if let Atom::V(i) = &ret_a {
        sch.consume(&Atom::V(*i))?;
    }
    sch.stmts.push(Stmt::Ret(ret_a));
    let _ = omega;
    Ok((
        core::Fn {
            params,
            ret,
            body: sch.stmts,
        },
        sch.tick,
        sch.charge,
        sch.try_c,
    ))
}

fn sig_of(f: &Func) -> Result<(Vec<Ty>, Ty, bool), String> {
    if f.eff >= 2 {
        return Err("emit: a function of effect class 2 or 3 is not in this slice".into());
    }
    if f.bound != Bound::None || has_rec_ref(&f.body) {
        return Err("recursion: a recursive function needs a declared bound, and rce-core/1 does not carry one".into());
    }
    let omega = f.ins.last() == Some(&BTy::Order);
    if omega && (f.eff == 0 || f.outs.last() != Some(&BTy::Order)) {
        return Err("defect: ω is not the last input and output of an effectful function".into());
    }
    let ins = if omega {
        &f.ins[..f.ins.len() - 1]
    } else {
        &f.ins[..]
    };
    let outs = if omega {
        &f.outs[..f.outs.len() - 1]
    } else {
        &f.outs[..]
    };
    if outs.len() != 1 {
        return Err(
            "exit: a function with several results has no single core return in this slice".into(),
        );
    }
    if ins.iter().any(|t| *t == BTy::Order) {
        return Err("defect: ω is not only the last input".into());
    }
    let params = ins
        .iter()
        .map(cty)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| "defect: a type the core cannot hold".to_string())?;
    let ret =
        cty(&outs[0]).map_err(|_| "defect: a result type the core cannot hold".to_string())?;
    Ok((params, ret, omega))
}

fn has_rec_ref(b: &Body) -> bool {
    b.nodes
        .iter()
        .any(|n| matches!(n.attrs.refs(), Some(Ref::Rec(_))) || n.regions.iter().any(has_rec_ref))
}

fn has_kind(b: &Body, pred: impl Fn(&Attr) -> bool + Copy) -> bool {
    b.nodes
        .iter()
        .any(|n| pred(&n.attrs) || n.regions.iter().any(|r| has_kind(r, pred)))
}

fn cap_or_catch(b: &Body) -> String {
    if has_kind(b, |a| matches!(a, Attr::Cap(_))) {
        "cap: a cap node waits for the host statement of rce-core/2".into()
    } else {
        "catch: a catch and the fuel-trap branch are not in this slice".into()
    }
}

struct Dom {
    bad: String,
    targs: Vec<Ty>,
    args: Vec<Atom>,
}

fn align_with_test(dom: Option<&Dom>, name: &str, targs: &[Ty], args: &mut [Atom]) {
    let Some(d) = dom else { return };
    let Some((fam, op)) = name.split_once('.') else {
        return;
    };
    if d.bad == format!("{fam}.bad.{op}")
        && d.targs == targs
        && args.len() == 2
        && d.args.len() == 2
        && args[0] == d.args[1]
        && args[1] == d.args[0]
    {
        args.swap(0, 1);
    }
}

fn schedule_region(
    sch: &mut Sch,
    m: &form::Module,
    group: &str,
    b: &Body,
    env: &mut HashMap<Src, Av>,
    dom: Option<&Dom>,
) -> Result<(), Change> {
    let tx = Tx::new(m);
    for (i, n) in b.nodes.iter().enumerate() {
        match &n.attrs {
            Attr::Lit(t, l) => {
                // A Bind is an atom (section 13): the literal is the operand, not a local.
                // `sum.mk` and `sum.get` require the index to be a literal atom.
                let ct =
                    cty(t).map_err(|_| sch.deny("defect: a literal type the core cannot hold"))?;
                env.insert(Src::Out(i as u32, 0), Av::A(Atom::K(ct, l.clone())));
            }
            Attr::Prim(name, args, cell) => {
                if cell.is_some() || name.starts_with("memo.") {
                    return Err(sch.deny("memo: a memo cell is not in this slice"));
                }
                if name == "emit" {
                    return Err(sch.deny(
                        "emit: emit needs the per-type runtime definition this slice does not link",
                    ));
                }
                if matches!(
                    name.as_str(),
                    "cost.spent" | "cost.deadline" | "cost.set_deadline"
                ) {
                    return Err(sch.deny(format!("emit: `{name}` is not in this slice")));
                }
                if matches!(
                    name.as_str(),
                    "fuel.tick" | "cost.charge" | "cost.try_charge"
                ) {
                    let nsrc = sch.atom(env, n.ins[0])?;
                    sch.consume(&nsrc)?;
                    let _ = sch.atom(env, n.ins[1]).ok(); // ω, no core value
                    let (flag, ret) = match name.as_str() {
                        "fuel.tick" => {
                            sch.tick = true;
                            (FnRef::Key("#tick".into(), 0), Ty::Bool)
                        }
                        "cost.charge" => {
                            sch.charge = true;
                            (FnRef::Key("#charge".into(), 0), Ty::Bool)
                        }
                        _ => {
                            sch.try_c = true;
                            (FnRef::Key("#try".into(), 0), Ty::Bool)
                        }
                    };
                    sch.stmts.push(Stmt::Call(flag, vec![nsrc]));
                    let id = sch.fresh(ret, false);
                    let _ = id;
                    env.insert(Src::Out(i as u32, 0), Av::Omega);
                    continue;
                }
                let e = catalogue::entry(name).ok_or_else(|| {
                    sch.deny(format!("defect: `{name}` is not a catalogue entry"))
                })?;
                let lits: Vec<Option<i128>> = n
                    .ins
                    .iter()
                    .map(|s| match env.get(s) {
                        Some(Av::A(Atom::K(Ty::Int(..), Lit::Int(v)))) => Some(*v),
                        _ => None,
                    })
                    .collect();
                let ports = tx
                    .operand_ports(e, args)
                    .map_err(|err| sch.deny(format!("defect: {name}: {err}")))?;
                let results = tx
                    .results(e, args, &lits)
                    .map_err(|err| sch.deny(format!("defect: {name}: {err}")))?;
                let mut core_args = vec![];
                for (k, s) in n.ins.iter().enumerate() {
                    let a = sch.atom(env, *s)?;
                    let read = crate::one::wf::reads(name, k);
                    if !read {
                        sch.consume(&a)?;
                    }
                    core_args.push(a);
                }
                let targs: Vec<Ty> = args
                    .iter()
                    .map(|a| match a {
                        TyArg::Body(t) => cty(t)
                            .map_err(|_| sch.deny("defect: a type argument the core cannot hold")),
                        TyArg::Frame(_) => {
                            Err(sch.deny("emit: a frame type argument is not in this slice"))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut produced: Vec<(usize, usize, Ty)> = vec![];
                for (oi, (of, ty)) in results.iter().enumerate() {
                    let mode = of.and_then(|idx| ports.get(spread_of(e, idx)).map(|p| p.0));
                    match mode {
                        Some(Mode::ReadReturn) => {}
                        Some(Mode::Order) => {}
                        _ => {
                            let ct = cty(ty).map_err(|_| {
                                sch.deny("defect: a result type the core cannot hold")
                            })?;
                            produced.push((oi, produced.len(), ct));
                        }
                    }
                }
                let start = sch.next;
                align_with_test(dom, name, &targs, &mut core_args);
                if !produced.is_empty() || name == "drop" {
                    sch.stmts.push(Stmt::Prim(name.clone(), targs, core_args));
                }
                for (_, k, ct) in &produced {
                    let id = sch.fresh(ct.clone(), true);
                    debug_assert_eq!(id, start + *k as u32);
                }
                let mut pk = 0usize;
                for (oi, (of, _)) in results.iter().enumerate() {
                    let mode = of.and_then(|idx| ports.get(spread_of(e, idx)).map(|p| p.0));
                    match mode {
                        Some(Mode::ReadReturn) => {
                            let op = spread_of(e, (*of).unwrap());
                            env.insert(Src::Out(i as u32, oi as u32), env[&n.ins[op]].clone());
                        }
                        Some(Mode::Order) => {
                            env.insert(Src::Out(i as u32, oi as u32), Av::Omega);
                        }
                        _ => {
                            env.insert(
                                Src::Out(i as u32, oi as u32),
                                Av::A(Atom::V(start + pk as u32)),
                            );
                            pk += 1;
                        }
                    }
                }
            }
            Attr::Call(r) => {
                let cf = tx
                    .callee(group, r)
                    .ok_or_else(|| sch.deny("defect: a call names no function"))?;
                let (cp, cr, co) = sig_of(cf).map_err(|e| sch.deny(e))?;
                let _ = cp;
                let mut args = vec![];
                let mut oi = 0;
                for (k, s) in n.ins.iter().enumerate() {
                    if cf.ins.get(k) == Some(&BTy::Order) {
                        continue;
                    }
                    let a = sch.atom(env, *s)?;
                    sch.consume(&a)?;
                    args.push(a);
                    oi += 1;
                }
                let _ = (oi, co);
                let rf = match r {
                    Ref::Key(k, i) => FnRef::Key(k.clone(), *i),
                    Ref::Rec(_) => return Err(sch.deny("recursion: a recursive function needs a declared bound, and rce-core/1 does not carry one")),
                };
                sch.stmts.push(Stmt::Call(rf, args));
                let id = sch.fresh(cr, true);
                env.insert(Src::Out(i as u32, 0), Av::A(Atom::V(id)));
                if co {
                    env.insert(Src::Out(i as u32, 1), Av::Omega);
                }
            }
            Attr::Catch(_) => {
                return Err(
                    sch.deny("catch: a catch and the fuel-trap branch are not in this slice")
                );
            }
            Attr::Cap(_) => {
                return Err(sch.deny("cap: a cap node waits for the host statement of rce-core/2"));
            }
            Attr::Branch(2, ts) => schedule_branch(sch, m, group, b, i, n, ts, env)?,
            Attr::Branch(_, _) => return Err(sch.deny(
                "selector: this slice schedules only a Branch with a bool selector and two arms",
            )),
            Attr::Count(ts) => schedule_iter(sch, m, group, i, n, ts, true, env)?,
            Attr::Measure(_, _, ts) => schedule_iter(sch, m, group, i, n, ts, false, env)?,
            Attr::Halt(_) => {
                let h = sch.atom(env, n.ins[0])?;
                sch.consume(&h)?;
                sch.stmts.push(Stmt::Halt(h));
            }
        }
    }
    Ok(())
}

fn dom_from(sch: &mut Sch, body: &Body, sel: Src, env: &HashMap<Src, Av>) -> Option<Dom> {
    let Src::Out(i, 0) = sel else { return None };
    let node = body.nodes.get(i as usize)?;
    let Attr::Prim(name, args, _) = &node.attrs else {
        return None;
    };
    if !name.contains(".bad.") {
        return None;
    }
    let mut atoms = vec![];
    for s in &node.ins {
        atoms.push(sch.atom(env, *s).ok()?);
    }
    let targs = args
        .iter()
        .map(|a| match a {
            TyArg::Body(t) => cty(t).ok(),
            TyArg::Frame(_) => None,
        })
        .collect::<Option<Vec<_>>>()?;
    Some(Dom {
        bad: name.clone(),
        targs,
        args: atoms,
    })
}

fn schedule_branch(
    sch: &mut Sch,
    m: &form::Module,
    group: &str,
    body: &Body,
    i: usize,
    n: &Node,
    ts: &[BTy],
    env: &mut HashMap<Src, Av>,
) -> Result<(), Change> {
    let sel = sch.atom(env, n.ins[0])?;
    if sch.tys_of(&sel)? != Ty::Bool && !matches!(sel, Atom::V(_)) {
        // a bool literal is fine; a non-bool local is not
    }
    if sch.tys_of(&sel)? != Ty::Bool {
        return Err(sch.deny(
            "selector: this slice schedules only a Branch with a bool selector and two arms",
        ));
    }
    let omega_at = ts.iter().position(|t| *t == BTy::Order);
    let mut vars = vec![];
    for (p, t) in ts.iter().enumerate() {
        if Some(p) == omega_at {
            continue;
        }
        let ct = cty(t).map_err(|_| sch.deny("defect: a branch output the core cannot hold"))?;
        let init = dummy(sch, t)?;
        sch.consume(&init)?;
        sch.stmts.push(Stmt::Var(ct.clone(), init));
        let id = sch.fresh(ct, true);
        vars.push((p, id));
    }
    let dom = dom_from(sch, body, n.ins[0], env);
    let mut arm = |region: &Body, dom: Option<&Dom>| -> Result<Vec<Stmt>, Change> {
        let saved = std::mem::take(&mut sch.stmts);
        let mut en = env.clone();
        for (j, s) in n.ins.iter().skip(1).enumerate() {
            en.insert(
                Src::In(j as u32),
                env.get(s)
                    .cloned()
                    .ok_or_else(|| sch.deny("defect: a capture has no value"))?,
            );
        }
        let mark = sch.next;
        schedule_region(sch, m, group, region, &mut en, dom)?;
        // A path that halts does not set the outputs (W3 lets a halt end what it holds).
        let halted = !build::falls(&sch.stmts);
        let mut sets = vec![];
        let mut keep = HashSet::new();
        if !halted {
            for (p, var) in &vars {
                let src = region.out[*p];
                match sch.val(&en, src)? {
                    Av::A(a) => {
                        if let Atom::V(id) = &a {
                            keep.insert(*id);
                        }
                        if sch.tys[var].owned() && sch.live.contains(var) {
                            sets.push(Stmt::Prim("drop".into(), vec![sch.tys[var].clone()], vec![
                                Atom::V(*var),
                            ]));
                            sch.live.remove(var);
                        }
                        sch.consume(&a)?;
                        sets.push(Stmt::Set(Place::V(*var), a));
                        if sch.tys[var].owned() {
                            sch.live.insert(*var);
                        }
                    }
                    Av::Omega => return Err(sch.deny("defect: a non-ω output is ω")),
                }
            }
            sch.drop_except(&keep, mark);
        }
        let mut stmts = std::mem::take(&mut sch.stmts);
        stmts.extend(sets);
        sch.stmts = saved;
        Ok(stmts)
    };
    // then is arm 1 (selector true), else is arm 0. A `bad` test is false in arm 0,
    // which is where the partial primitive sits, so that arm inherits the test's operand order.
    let then_s = arm(&n.regions[1], None)?;
    let else_s = arm(&n.regions[0], dom.as_ref())?;
    sch.stmts.push(Stmt::If(sel, then_s, else_s));
    for (p, var) in vars {
        env.insert(Src::Out(i as u32, p as u32), Av::A(Atom::V(var)));
    }
    if let Some(p) = omega_at {
        env.insert(Src::Out(i as u32, p as u32), Av::Omega);
    }
    Ok(())
}

impl Sch {
    fn tys_of(&self, a: &Atom) -> Result<Ty, Change> {
        match a {
            Atom::V(i) => self
                .tys
                .get(i)
                .cloned()
                .ok_or_else(|| self.deny(format!("defect: v{i}"))),
            Atom::K(t, _) => Ok(t.clone()),
            Atom::G(_) => {
                Err(self.deny("global: a global is not part of the working context ω holds"))
            }
        }
    }
}

fn dummy(sch: &mut Sch, t: &BTy) -> Result<Atom, Change> {
    Ok(match t {
        BTy::Bool => Atom::bool(false),
        BTy::Int(lo, hi) => Atom::K(Ty::Int(*lo, *hi), Lit::Int(*lo)),
        BTy::F64 => Atom::K(Ty::F64, Lit::F64(0)),
        BTy::F32 => Atom::K(Ty::F32, Lit::F32(0)),
        BTy::Bytes => Atom::bytes(b""),
        BTy::List(e) => {
            let et = cty(e).map_err(|_| sch.deny("defect: a list element the core cannot hold"))?;
            sch.stmts
                .push(Stmt::Prim("list.new".into(), vec![et.clone()], vec![]));
            let id = sch.fresh(Ty::List(Box::new(et)), true);
            Atom::V(id)
        }
        BTy::Tup(xs) => {
            let mut args = vec![];
            let mut tys = vec![];
            for x in xs {
                let a = dummy(sch, x)?;
                sch.consume(&a)?;
                tys.push(cty(x).map_err(|_| sch.deny("defect: a tuple component"))?);
                args.push(a);
            }
            let tt = Ty::Tup(tys);
            sch.stmts
                .push(Stmt::Prim("tup.mk".into(), vec![tt.clone()], args));
            let id = sch.fresh(tt, true);
            Atom::V(id)
        }
        BTy::Sum(xs) => {
            let inner = dummy(
                sch,
                xs.first().ok_or_else(|| sch.deny("defect: an empty sum"))?,
            )?;
            sch.consume(&inner)?;
            let st = cty(t).map_err(|_| sch.deny("defect: a sum"))?;
            sch.stmts
                .push(Stmt::Prim("sum.mk".into(), vec![st.clone()], vec![
                    build::lit(0),
                    inner,
                ]));
            let id = sch.fresh(st, true);
            Atom::V(id)
        }
        BTy::Arr(_) => {
            return Err(
                sch.deny("defect: an array output has no dummy initial value in this slice")
            );
        }
        BTy::Order | BTy::Decl(_) => {
            return Err(sch.deny("defect: a branch output the core cannot hold"));
        }
    })
}

fn schedule_iter(
    sch: &mut Sch,
    m: &form::Module,
    group: &str,
    i: usize,
    n: &Node,
    ts: &[BTy],
    count: bool,
    env: &mut HashMap<Src, Av>,
) -> Result<(), Change> {
    let omega_at = ts.iter().position(|t| *t == BTy::Order);
    if omega_at == Some(0) || omega_at.is_none() {
        // ω first, or absent
    } else if omega_at.is_some() {
        return Err(sch.deny("defect: ω is carried, and not as the first carried value"));
    }
    let carried_ins = if count { &n.ins[1..] } else { &n.ins[..] };
    if carried_ins.len() != ts.len() {
        return Err(sch.deny("defect: an Iterate's inputs do not match its carried values"));
    }
    let mut vars: Vec<Option<u32>> = vec![];
    for (k, t) in ts.iter().enumerate() {
        if *t == BTy::Order {
            vars.push(None);
            continue;
        }
        let ct = cty(t).map_err(|_| sch.deny("defect: a carried type the core cannot hold"))?;
        let init = sch.atom(env, carried_ins[k])?;
        sch.consume(&init)?;
        sch.stmts.push(Stmt::Var(ct.clone(), init));
        let id = sch.fresh(ct, true);
        vars.push(Some(id));
    }
    let counter = if count {
        let ct = build::count_t();
        sch.stmts.push(Stmt::Var(ct.clone(), build::lit(0)));
        Some(sch.fresh(ct, false))
    } else {
        None
    };
    let saved = std::mem::take(&mut sch.stmts);
    let run_region = |sch: &mut Sch, region: &Body| -> Result<(Vec<Stmt>, Vec<Av>), Change> {
        let mut en = HashMap::new();
        for (k, s) in carried_ins.iter().enumerate() {
            let av = if ts[k] == BTy::Order {
                Av::Omega
            } else {
                Av::A(Atom::V(vars[k].unwrap()))
            };
            en.insert(Src::In(k as u32), av);
            let _ = s;
        }
        let mark = sch.next;
        let before = std::mem::take(&mut sch.stmts);
        schedule_region(sch, m, group, region, &mut en, None)?;
        let mut outs = vec![];
        let mut keep = HashSet::new();
        for s in &region.out {
            let av = sch.val(&en, *s)?;
            if let Av::A(Atom::V(id)) = &av {
                keep.insert(*id);
            }
            outs.push(av);
        }
        sch.drop_except(&keep, mark);
        let stmts = std::mem::take(&mut sch.stmts);
        sch.stmts = before;
        Ok((stmts, outs))
    };
    let (test_s, test_o) = run_region(sch, &n.regions[0])?;
    let epi = |outs: &[Av], sch: &mut Sch| -> Result<Vec<Stmt>, Change> {
        let mut s = vec![];
        let flag = match &outs[0] {
            Av::A(a) => a.clone(),
            Av::Omega => return Err(sch.deny("defect: a loop flag is ω")),
        };
        for (k, var) in vars.iter().enumerate() {
            let Some(var) = var else { continue };
            let Av::A(a) = &outs[1 + k] else {
                return Err(sch.deny("defect: a carried result is ω"));
            };
            if sch.tys[var].owned() {
                s.push(Stmt::Prim("drop".into(), vec![sch.tys[var].clone()], vec![
                    Atom::V(*var),
                ]));
            }
            s.push(Stmt::Set(Place::V(*var), a.clone()));
        }
        s.push(Stmt::If(flag, vec![], vec![Stmt::Break(0)]));
        Ok(s)
    };
    let mut loop_body = test_s;
    loop_body.extend(epi(&test_o, sch)?);
    if let Some(cn) = counter {
        // Bind locals in the order a walker meets the statements: the let, the add, the
        // comparison, then `sum.mk` inside the true arm.
        let ct = build::count_t();
        let bound = sch.atom(env, n.ins[0])?;
        loop_body.push(Stmt::Let(Atom::V(cn)));
        let a = sch.fresh(ct.clone(), false);
        loop_body.push(Stmt::Prim("int.sadd".into(), vec![ct.clone()], vec![
            Atom::V(a),
            build::lit(1),
        ]));
        let c = sch.fresh(ct.clone(), false);
        loop_body.push(Stmt::Set(Place::V(cn), Atom::V(c)));
        loop_body.push(Stmt::Prim("int.lt".into(), vec![ct], vec![
            bound,
            Atom::V(c),
        ]));
        let o = sch.fresh(Ty::Bool, false);
        let h = sch.next;
        loop_body.push(Stmt::If(
            Atom::V(o),
            vec![
                Stmt::Prim("sum.mk".into(), vec![Ty::halt()], vec![
                    build::lit(0),
                    Atom::bytes(b"loop bound"),
                ]),
                Stmt::Halt(Atom::V(h)),
            ],
            vec![],
        ));
        let got = sch.fresh(Ty::halt(), true);
        debug_assert_eq!(got, h);
    }
    let (body_s, body_o) = run_region(sch, &n.regions[1])?;
    loop_body.extend(body_s);
    loop_body.extend(epi(&body_o, sch)?);
    sch.stmts = saved;
    sch.stmts.push(Stmt::Loop(loop_body));
    for (k, var) in vars.iter().enumerate() {
        let av = match var {
            Some(v) => Av::A(Atom::V(*v)),
            None => Av::Omega,
        };
        env.insert(Src::Out(i as u32, k as u32), av);
    }
    Ok(())
}

fn shim_tick(g: u32) -> core::Fn {
    let mut b = FB::new(vec![Ty::u64()], Ty::Bool);
    let n = b.arg(0);
    let cur = b.let_global(g, Ty::u64());
    let c = b.int_cmp("int.lt", cur.clone(), n.clone()).expect("tick");
    b.guard_halt(c, |b| b.trap_value(Atom::bytes(b"comptime fuel exhausted")))
        .expect("tick");
    let r = b.sub_in(cur, n, Ty::u64()).expect("tick");
    b.setg(g, r);
    b.ret(Atom::bool(true));
    b.finish()
}

fn shim_charge(spent: u32, deadline: u32) -> core::Fn {
    let mut b = FB::new(vec![Ty::u64()], Ty::Bool);
    let n = b.arg(0);
    let cur = b.let_global(spent, Ty::u64());
    let t = b
        .p("int.sadd", vec![Ty::u64()], vec![cur, n])
        .expect("charge");
    let dl = b.let_global(deadline, Ty::u64());
    let over = b.int_cmp("int.lt", dl, t.clone()).expect("charge");
    b.guard_halt(over, |b| b.trap_value(Atom::bytes(b"budget")))
        .expect("charge");
    b.setg(spent, t);
    b.ret(Atom::bool(true));
    b.finish()
}

fn shim_try(spent: u32, deadline: u32) -> core::Fn {
    let mut b = FB::new(vec![Ty::u64()], Ty::Bool);
    let n = b.arg(0);
    let cur = b.let_global(spent, Ty::u64());
    let t = b
        .p("int.sadd", vec![Ty::u64()], vec![cur.clone(), n])
        .expect("try");
    let dl = b.let_global(deadline, Ty::u64());
    let m = b.p("int.max", vec![Ty::u64()], vec![cur, dl]).expect("try");
    let r = b.p("int.min", vec![Ty::u64()], vec![t, m]).expect("try");
    b.setg(spent, r);
    b.ret(Atom::bool(true));
    b.finish()
}

/// The core view of a Form this slice schedules, or the first construct that
/// requires change. The result is well formed (W1 to W6).
pub fn schedule(m: &form::Module) -> Result<core::Module, Change> {
    if m.caps.iter().any(|_| true) && !m.caps.is_empty() {
        return Err(deny(
            "schedule",
            "caps",
            "cap: a capability waits for the host statement of rce-core/2",
        ));
    }
    let mut funcs: Vec<(String, core::Fn, bool, bool, bool)> = vec![];
    let mut need = [false; 3];
    for (k, g) in &m.defs {
        if g.fns.len() != 1 || g.fns.iter().any(|f| has_rec_ref(&f.body)) {
            return Err(deny(
                "schedule",
                k,
                "recursion: a recursive function needs a declared bound, and rce-core/1 does not carry one",
            ));
        }
        for (i, f) in g.fns.iter().enumerate() {
            let at = format!("{k} fn {i}");
            let (cf, tick, charge, try_c) = schedule_func(m, k, f, &at)?;
            need[0] |= tick;
            need[1] |= charge;
            need[2] |= try_c;
            funcs.push((k.clone(), cf, tick, charge, try_c));
        }
    }
    // rewrite shim placeholders once the global indices are known
    let mut globals = vec![];
    let mut gnames = vec![];
    let g_fuel = if need[0] {
        globals.push((Ty::u64(), Some(Lit::Int(super::eval::FUEL as i128))));
        gnames.push("fuel");
        Some(0u32)
    } else {
        None
    };
    let (g_spent, g_deadline) = if need[1] || need[2] {
        let s = globals.len() as u32;
        globals.push((Ty::u64(), Some(Lit::Int(0))));
        gnames.push("cost.spent");
        let d = globals.len() as u32;
        globals.push((Ty::u64(), Some(Lit::Int(u64::MAX as i128))));
        gnames.push("cost.deadline");
        (Some(s), Some(d))
    } else {
        (None, None)
    };
    let mut shims: Vec<(&str, core::Fn)> = vec![];
    if let Some(g) = g_fuel {
        shims.push(("rt.tick", shim_tick(g)));
    }
    if need[1] {
        shims.push((
            "rt.charge",
            shim_charge(g_spent.unwrap(), g_deadline.unwrap()),
        ));
    }
    if need[2] {
        shims.push((
            "rt.try_charge",
            shim_try(g_spent.unwrap(), g_deadline.unwrap()),
        ));
    }
    let tick_ref = FnRef::Key("#tick".into(), 0);
    let charge_ref = FnRef::Key("#charge".into(), 0);
    let try_ref = FnRef::Key("#try".into(), 0);
    // keys: each function its own group, shims included, placeholders rewritten after keys exist
    let mut members: Vec<(String, core::Fn)> = vec![];
    for (name, mut f, _, _, _) in funcs {
        rewrite_shim(
            &mut f.body,
            &tick_ref,
            &charge_ref,
            &try_ref,
            None,
            None,
            None,
        );
        members.push((name, f));
    }
    let shim_base = members.len();
    for (name, f) in &shims {
        members.push(((*name).to_string(), f.clone()));
    }
    // assign keys in dependency order: shims first (no calls), then users, by computing keys of shims,
    // rewriting user calls to those keys, then keying users.
    let mut shim_keys: Vec<Option<(String, u32)>> = vec![None; shims.len()];
    for (s, (name, f)) in shims.iter().enumerate() {
        let (g, _) = canonical_group(std::slice::from_ref(f));
        let key = group_key(&g);
        shim_keys[s] = Some((key, 0));
        let _ = name;
    }
    let map = |r: &FnRef| -> Option<(String, u32)> {
        match r {
            FnRef::Key(k, 0) if k == "#tick" => shim_keys.first().and_then(|x| x.clone()),
            FnRef::Key(k, 0) if k == "#charge" => {
                shim_keys.get(usize::from(need[0])).and_then(|x| x.clone())
            }
            FnRef::Key(k, 0) if k == "#try" => {
                let idx = usize::from(need[0]) + usize::from(need[1]);
                shim_keys.get(idx).and_then(|x| x.clone())
            }
            _ => None,
        }
    };
    // The placeholder indices depend on which shims exist. Rebuild from the flags.
    let mut named: Vec<(String, core::Fn)> = vec![];
    // re-schedule was already done; rewrite using the flag order
    let mut shim_pairs: Vec<(&str, String)> = vec![];
    if need[0] {
        shim_pairs.push(("rt.tick", shim_keys[0].clone().unwrap().0));
    }
    if need[1] {
        let idx = usize::from(need[0]);
        shim_pairs.push(("rt.charge", shim_keys[idx].clone().unwrap().0));
    }
    if need[2] {
        let idx = usize::from(need[0]) + usize::from(need[1]);
        shim_pairs.push(("rt.try_charge", shim_keys[idx].clone().unwrap().0));
    }
    let _ = (map, shim_base, named);
    // Simpler assembly: put every function, with calls still as #tick/#charge/#try, into groups
    // after substituting the real shim keys.
    let key_of = |ph: &str| -> Option<String> {
        shim_pairs
            .iter()
            .find(|(n, _)| {
                (*n == "rt.tick" && ph == "#tick")
                    || (*n == "rt.charge" && ph == "#charge")
                    || (*n == "rt.try_charge" && ph == "#try")
            })
            .map(|(_, k)| k.clone())
    };
    let mut defs = BTreeMap::new();
    let mut fn_names = serde_json::Map::new();
    let mut user_fs: Vec<(String, core::Fn)> = members.drain(..).collect();
    // members holds users, then shims. User calls of a shim are still `#tick` and the like;
    // user calls of a user function still name the Form group.
    let shims_here = user_fs.split_off(user_fs.len() - shims.len());
    let mut shim_core: HashMap<String, String> = HashMap::new();
    for (name, f) in &shims_here {
        let (g, _) = canonical_group(std::slice::from_ref(f));
        let key = group_key(&g);
        shim_core.insert(name.clone(), key.clone());
        fn_names.insert(key.clone(), serde_json::json!([name]));
        defs.insert(key, g);
    }
    for (name, f) in &mut user_fs {
        subst_placeholders(&mut f.body, &key_of);
        let _ = name;
    }
    // Callees before callers, so a core key is known before a caller is hashed.
    let form_keys: HashSet<String> = user_fs.iter().map(|(k, _)| k.clone()).collect();
    let mut pending: HashMap<String, core::Fn> = user_fs.into_iter().collect();
    let mut form_core: HashMap<String, String> = HashMap::new();
    while !pending.is_empty() {
        let ready: Vec<String> = pending
            .iter()
            .filter(|(_, f)| {
                form_calls(f)
                    .iter()
                    .all(|c| !form_keys.contains(c) || form_core.contains_key(c))
            })
            .map(|(k, _)| k.clone())
            .collect();
        if ready.is_empty() {
            return Err(deny(
                "schedule",
                "",
                "recursion: a recursive function needs a declared bound, and rce-core/1 does not carry one",
            ));
        }
        for k in ready {
            let mut f = pending.remove(&k).unwrap();
            subst_form_keys(&mut f.body, &form_core);
            let (g, _) = canonical_group(std::slice::from_ref(&f));
            let key = group_key(&g);
            form_core.insert(k, key.clone());
            fn_names.insert(key.clone(), serde_json::json!(["f"]));
            defs.insert(key, g);
        }
    }
    let _ = shim_core;
    let mut entry = form_core
        .values()
        .next()
        .cloned()
        .or_else(|| defs.keys().next().cloned())
        .ok_or_else(|| deny("schedule", "", "defect: a Form with no function"))?;
    // W1: the entry takes no parameters. The scheduled functions keep theirs.
    if !defs[&entry].fns[0].params.is_empty() {
        let main = core::Fn {
            params: vec![],
            ret: Ty::i64(),
            body: vec![Stmt::Ret(build::lit(0))],
        };
        let (g, _) = canonical_group(std::slice::from_ref(&main));
        let key = group_key(&g);
        fn_names.insert(key.clone(), serde_json::json!(["main"]));
        defs.insert(key.clone(), g);
        entry = key;
    }
    let module = core::Module {
        globals,
        defs,
        entry: (entry, 0),
        names: serde_json::json!({"fns": fn_names, "globals": gnames}),
        graphs: BTreeMap::new(),
    };
    crate::one::wf::check(&module).map_err(|e| {
        deny(
            "schedule",
            "",
            format!("defect: the schedule is not well formed: {e}"),
        )
    })?;
    let _ = rewrite_shim;
    Ok(module)
}

fn rewrite_shim(
    b: &mut [Stmt],
    tick: &FnRef,
    charge: &FnRef,
    try_c: &FnRef,
    tk: Option<&FnRef>,
    ck: Option<&FnRef>,
    yk: Option<&FnRef>,
) {
    let _ = (tick, charge, try_c, tk, ck, yk, b);
}

fn form_calls(f: &core::Fn) -> Vec<String> {
    let mut v = vec![];
    fn go(b: &[Stmt], v: &mut Vec<String>) {
        for s in b {
            match s {
                Stmt::Call(FnRef::Key(k, _), _) | Stmt::Catch(FnRef::Key(k, _), _)
                    if !k.starts_with('#') =>
                {
                    v.push(k.clone())
                }
                Stmt::If(_, t, e) => {
                    go(t, v);
                    go(e, v);
                }
                Stmt::Block(x) | Stmt::Loop(x) => go(x, v),
                _ => {}
            }
        }
    }
    go(&f.body, &mut v);
    v
}

fn subst_form_keys(b: &mut [Stmt], map: &HashMap<String, String>) {
    for s in b {
        match s {
            Stmt::Call(FnRef::Key(k, _), _) | Stmt::Catch(FnRef::Key(k, _), _) => {
                if let Some(n) = map.get(k) {
                    *k = n.clone();
                }
            }
            Stmt::If(_, t, e) => {
                subst_form_keys(t, map);
                subst_form_keys(e, map);
            }
            Stmt::Block(x) | Stmt::Loop(x) => subst_form_keys(x, map),
            _ => {}
        }
    }
}

fn subst_placeholders(b: &mut [Stmt], key_of: &dyn Fn(&str) -> Option<String>) {
    for s in b {
        match s {
            Stmt::Call(FnRef::Key(k, _), _) if k.starts_with('#') => {
                if let Some(real) = key_of(k) {
                    *k = real;
                }
            }
            Stmt::If(_, t, e) => {
                subst_placeholders(t, key_of);
                subst_placeholders(e, key_of);
            }
            Stmt::Block(x) | Stmt::Loop(x) => subst_placeholders(x, key_of),
            _ => {}
        }
    }
}
