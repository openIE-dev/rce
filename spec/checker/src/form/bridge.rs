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
//! - `loop` in the two-region shape the schedule writes, and the same
//!   count and measure shapes the lowering writes (`one/shape.rs`): a
//!   measure (`for` over a range or a list), or a count bound whose test
//!   or body uses ω. A `break` that names that loop and sits in the body
//!   is the body's flag (option B of `spec/proposals/break-carried-flag.md`):
//!   the break path returns false, the continue path returns true and the
//!   carried values. That rewrite is not syntactic identity. `body_break_rewrite`
//!   reports it. A break that does not name the loop still requires change.
//! - a Branch whose arms use different outer values: both arms take the
//!   union of those captures, ω included when either arm needs it
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
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

thread_local! {
    /// Set when `construct` turns a loop-body `break` into the body's flag.
    /// Option B of break-carried-flag.md: that form is outside syntactic
    /// identity. See `body_break_rewrite`.
    static BODY_BREAK: Cell<bool> = const { Cell::new(false) };
}

fn note_body_break() {
    BODY_BREAK.with(|c| c.set(true));
}

/// Whether the most recent `construct` on this thread restructured a
/// loop-body `break` into the body's flag.
///
/// Option B of `spec/proposals/break-carried-flag.md`.
/// The flag form can still satisfy `construct(schedule(F)) = F` as canonical
/// bytes. That equality is the flag encoding against its schedule. It is not
/// syntactic identity for the break: Lean `Stmt` has no break constructor, so
/// the Iterate has no preimage there. A break-restructured form is outside
/// that syntactic domain. The claim on it is three-way eval, with the flag
/// tested before the bound (O6') and the flag's nodes not reading or writing
/// ω (O7').
pub fn body_break_rewrite() -> bool {
    BODY_BREAK.with(|c| c.get())
}

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

/// Whether every statement can complete. A `break` of any depth does not.
fn refs_fall(stmts: &[&Stmt]) -> bool {
    stmts.iter().all(|s| build::falls(std::slice::from_ref(*s)))
}

/// An `if` whose arms do not both fall through. The ordinary construction
/// sinks the statements after it into the arm that falls. A loop-body break
/// in that tail has to stay with the values the tail computes.
fn sinking_if(s: &Stmt) -> bool {
    let Stmt::If(_, t, e) = s else {
        return false;
    };
    build::falls(t) != build::falls(e)
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
    /// A `break` that names the loop under construction. The path returns
    /// the flag false and does not run the statements after the break.
    Exit,
}

struct Scope {
    nodes: Vec<Node>,
    env: HashMap<u32, Src>,
    omega: Option<Src>,
    caps: Vec<Src>,
    assigned: BTreeSet<u32>,
    /// `set` targets in the order the arm stores them. Branch outputs follow
    /// this order, which is the order the schedule writes the stores back.
    set_at: Vec<u32>,
    /// Literals already built in this region, so a second use is the same source.
    lits: Vec<(Ty, Lit, Src)>,
    /// The loop flag this scope produces, when a body break was joined here.
    exit_flag: Option<Src>,
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
    /// A literal node. A second use of the same literal in this region is the
    /// same source, so a domain test and the operation it guards compare equal.
    fn ensure_lit(&self, sc: &mut Scope, t: &Ty, l: &Lit) -> Result<Src, Change> {
        if let Some(s) = find_lit(sc, t, l) {
            return Ok(s);
        }
        let s = self.lit_node(sc, t, l)?;
        sc.lits.push((t.clone(), l.clone(), s));
        Ok(s)
    }

    /// A value of `t` the test of a loop overwrites before the body reads it.
    /// The body receives the test's output, so this initial is not an operand
    /// of the computation. An owned initial is dropped when the test's result
    /// is stored, which is what the schedule writes for a carried owned value.
    fn dummy_src(&self, sc: &mut Scope, t: &Ty) -> Result<Src, Change> {
        match t {
            Ty::Bool => self.ensure_lit(sc, t, &Lit::Bool(false)),
            Ty::Int(lo, hi) => {
                let v = if *lo <= 0 && *hi >= 0 { 0 } else { *lo };
                self.ensure_lit(sc, &Ty::Int(v, v), &Lit::Int(v))
            }
            Ty::F64 => self.ensure_lit(sc, t, &Lit::F64(0)),
            Ty::F32 => self.ensure_lit(sc, t, &Lit::F32(0)),
            Ty::Bytes => self.ensure_lit(sc, t, &Lit::Bytes(vec![])),
            Ty::List(e) => {
                let arg = ty_arg(e).map_err(|e| self.deny(format!("defect: {e}")))?;
                let id = sc.push(Attr::Prim("list.new".into(), vec![arg], None), vec![], vec![]);
                Ok(Src::Out(id, 0))
            }
            Ty::Tup(xs) => {
                let mut ins = vec![];
                for x in xs {
                    ins.push(self.dummy_src(sc, x)?);
                }
                let arg = ty_arg(t).map_err(|e| self.deny(format!("defect: {e}")))?;
                let id = sc.push(Attr::Prim("tup.mk".into(), vec![arg], None), ins, vec![]);
                Ok(Src::Out(id, 0))
            }
            Ty::Sum(xs) => {
                let inner_ty = xs
                    .first()
                    .ok_or_else(|| self.deny("defect: an empty sum"))?;
                let inner = self.dummy_src(sc, inner_ty)?;
                let tag = self.ensure_lit(sc, &Ty::Int(0, 0), &Lit::Int(0))?;
                let arg = ty_arg(t).map_err(|e| self.deny(format!("defect: {e}")))?;
                let id = sc.push(
                    Attr::Prim("sum.mk".into(), vec![arg], None),
                    vec![tag, inner],
                    vec![],
                );
                Ok(Src::Out(id, 0))
            }
            Ty::Arr(_) => Err(self.deny(
                "defect: a value defined in a loop test and used in its body is an array, and this slice has no initial value for it",
            )),
        }
    }

    /// Locals the test binds on every path and the body uses, but the parent
    /// does not have yet. Each gets a dummy initial so it can be carried.
    fn seed_flow(
        &self,
        sc: &mut Scope,
        test: &[Stmt],
        body: &[Stmt],
        except: &[u32],
    ) -> Result<(), Change> {
        let mut defined = BTreeSet::new();
        straight_bound(test, &self.info, &mut defined);
        let mut used = BTreeSet::new();
        stmt_locals(body, &mut used);
        for v in defined {
            if except.contains(&v) || !used.contains(&v) || sc.env.contains_key(&v) {
                continue;
            }
            let t = self.loc_ty(v)?;
            let src = self.dummy_src(sc, &t)?;
            sc.env.insert(v, src);
        }
        Ok(())
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
            Atom::K(t, l) => self.ensure_lit(sc, t, l),
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

    /// Whether a statement list halts, ticks, or calls a function that takes ω.
    /// Both arms of a branch that needs ω capture it, so the lists match.
    fn stmts_need_order(&self, stmts: &[Stmt]) -> bool {
        stmts.iter().any(|s| self.stmt_needs_order(s))
    }

    fn stmt_needs_order(&self, s: &Stmt) -> bool {
        match s {
            Stmt::Halt(_) => true,
            Stmt::Call(r, _) | Stmt::Catch(r, _) => self.ref_needs_order(r),
            Stmt::If(_, t, e) => self.stmts_need_order(t) || self.stmts_need_order(e),
            Stmt::Block(x) | Stmt::Loop(x) => self.stmts_need_order(x),
            _ => false,
        }
    }

    fn ref_needs_order(&self, r: &FnRef) -> bool {
        let id = match r {
            FnRef::Key(k, i) => (k.clone(), *i),
            FnRef::Rec(i) => (self.group.clone(), *i),
        };
        if self.names.get(&id).is_some_and(|n| shim_of(n).is_some()) {
            return true;
        }
        self.meta.get(&id).is_some_and(|m| m.omega)
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
                    if sc.assigned.insert(*v) {
                        sc.set_at.push(*v);
                    }
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
                    // The ω this region already holds. A source from outside the
                    // region is not one of its inputs.
                    let here = sc.omega.or(parent_omega);
                    let tail_needs = rest.iter().copied().any(|s| self.stmt_needs_order(s));
                    let thread = if self.stmts_need_order(t) || self.stmts_need_order(e) || tail_needs {
                        here
                    } else {
                        None
                    };
                    // A literal the parent already built is captured, so the
                    // operation and its domain test are the same value (FG8).
                    let lits = lits_already(&sc, &[t.as_slice(), e.as_slice()], rest);
                    // A tail of only drops and stores, whose values the parent
                    // already has, is the schedule's join of a branch's outputs.
                    // It stays after this branch. Sinking it into a nested halt
                    // would put those outputs on the inner branch, and
                    // construct(schedule(F)) would differ from F.
                    if !rest.is_empty() && t_falls != e_falls && !tail_is_join(rest, sc) {
                        let (cont, stop, cont_then) = if t_falls { (t.as_slice(), e.as_slice(), true) } else { (e.as_slice(), t.as_slice(), false) };
                        let seed = self.seed_of(&[cont, stop], rest, sc);
                        let stop_end = self.arm(stop, &seed, &lits, sc, here, thread, ret_ty)?;
                        let cont_end =
                            self.arm_then(cont, rest, &seed, &lits, sc, here, thread, ret_ty)?;
                        let (e0, a0, e1, a1) = if cont_then {
                            (stop_end.0, stop_end.1, cont_end.0, cont_end.1)
                        } else {
                            (cont_end.0, cont_end.1, stop_end.0, stop_end.1)
                        };
                        let sel = self.src_atom(sc, c)?;
                        return self.finish_if(sc, sel, a0, e0, a1, e1, here, ret_ty, false);
                    }
                    let sel = self.src_atom(sc, c)?;
                    let seed = self.seed_of(&[t, e], &[], sc);
                    let (e1, a1) = self.arm(t, &seed, &lits, sc, here, thread, ret_ty)?;
                    let (e0, a0) = self.arm(e, &seed, &lits, sc, here, thread, ret_ty)?;
                    let end = self.finish_if(sc, sel, a0, e0, a1, e1, here, ret_ty, false)?;
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
                    return self.exec_refs(&nlist, sc, sc.omega.or(parent_omega), ret_ty);
                }
                Stmt::Loop(b) => {
                    self.loop_stmt(b, s, sc, sc.omega.or(parent_omega), ret_ty)?;
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
    /// `thread` is the parent's ω when either arm needs it; both arms capture
    /// it at the same index, whether or not this arm uses it.
    #[allow(clippy::too_many_arguments)]
    fn arm(
        &self,
        stmts: &[Stmt],
        seed: &[u32],
        lits: &[(Ty, Lit, Src)],
        parent: &Scope,
        parent_omega: Option<Src>,
        thread: Option<Src>,
        ret_ty: &BTy,
    ) -> Result<(End, Scope), Change> {
        self.arm_then(stmts, &[], seed, lits, parent, parent_omega, thread, ret_ty)
    }

    #[allow(clippy::too_many_arguments)]
    fn arm_then(
        &self,
        first: &[Stmt],
        tail: &[&Stmt],
        seed: &[u32],
        lits: &[(Ty, Lit, Src)],
        parent: &Scope,
        parent_omega: Option<Src>,
        thread: Option<Src>,
        ret_ty: &BTy,
    ) -> Result<(End, Scope), Change> {
        let mut sc = Scope {
            nodes: vec![],
            env: HashMap::new(),
            omega: None,
            caps: vec![],
            assigned: BTreeSet::new(),
            set_at: vec![],
            lits: vec![],
            exit_flag: None,
        };
        for (j, l) in seed.iter().enumerate() {
            sc.env.insert(*l, Src::In(j as u32));
            sc.caps.push(parent.env[l]);
        }
        for (t, l, src) in lits {
            let j = sc.caps.len() as u32;
            sc.caps.push(*src);
            sc.lits.push((t.clone(), l.clone(), Src::In(j)));
        }
        if let Some(p) = thread {
            let j = sc.caps.len() as u32;
            sc.caps.push(p);
            sc.omega = Some(Src::In(j));
        }
        // One sequence, so a domain test sinks the statements that follow it into its arm.
        let mut list: Vec<&Stmt> = first.iter().collect();
        list.extend(tail.iter().copied());
        let live = sc.omega.or(parent_omega);
        let end = self.exec_refs(&list, &mut sc, live, ret_ty)?;
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
        join_flag: bool,
    ) -> Result<End, Change> {
        if join_flag && (matches!(e0, End::Ret(_)) || matches!(e1, End::Ret(_))) {
            return Err(self.deny(
                "exit: a break path and a ret in one loop body are an exit this slice does not restructure",
            ));
        }
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
        // Both arms take the outer sources either of them captured. C7 then
        // drops a capture neither arm uses and orders the rest by digest.
        reconcile_caps(&mut a0, &mut a1);
        if needs_w {
            let p = sc
                .omega
                .or(parent_omega)
                .ok_or_else(|| self.deny("omega: a branch needs ω and has none"))?;
            if !a0.caps.contains(&p) {
                a0.caps.push(p);
                a1.caps.push(p);
            }
            let j = a0
                .caps
                .iter()
                .position(|s| *s == p)
                .ok_or_else(|| self.deny("defect: ω is not a capture of the branch"))?
                as u32;
            for arm in [&mut a0, &mut a1] {
                if arm.omega.is_none() {
                    arm.omega = Some(Src::In(j));
                }
            }
        }
        if a0.caps != a1.caps {
            return Err(self.deny(
                "branch: the arms capture different values, and this slice does not reconcile them",
            ));
        }
        // outputs: vars assigned in either arm, then ω
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
        // A read-return or a nested branch rebinds a parent local inside an
        // arm without a `set` on this arm. The new source has to leave the
        // branch, or the parent still uses the value the branch captured.
        let rebound = |arm: &Scope, v: &u32| matches!(arm.env.get(v), Some(Src::Out(_, _)));
        for v in sc.env.keys().copied() {
            if vars.contains(&v) {
                continue;
            }
            let r0 = rebound(&a0, &v);
            let r1 = rebound(&a1, &v);
            if !r0 && !r1 {
                continue;
            }
            let yields = |e: &End| matches!(e, End::Fall | End::Exit);
            let present = if yields(&e0) && yields(&e1) {
                a0.env.contains_key(&v) && a1.env.contains_key(&v)
            } else if yields(&e0) && matches!(e1, End::Halt(_)) {
                a0.env.contains_key(&v)
            } else if yields(&e1) && matches!(e0, End::Halt(_)) {
                a1.env.contains_key(&v)
            } else {
                false
            };
            if present {
                vars.insert(v);
            }
        }
        // Stores first, in the order the arms write them, then any rebound
        // local that has no store, in index order. The schedule writes the
        // stores in this same order, so the construction reads them back.
        let mut ordered = vec![];
        for v in a0.set_at.iter().chain(a1.set_at.iter()) {
            if vars.contains(v) && !ordered.contains(v) {
                ordered.push(*v);
            }
        }
        for v in &vars {
            if !ordered.contains(v) {
                ordered.push(*v);
            }
        }
        let mut vars = ordered;
        // Two locals that name one source in every arm that falls through are
        // one output. Listing the source twice is two consumers (FG6).
        let mut alias: Vec<(u32, u32)> = vec![];
        let shares = |u: u32, v: u32| {
            let same = |arm: &Scope, e: &End| match e {
                End::Halt(_) => true,
                End::Fall | End::Exit => {
                    arm.env.contains_key(&u) && arm.env.get(&u) == arm.env.get(&v)
                }
                End::Ret(_) => false,
            };
            same(&a0, &e0) && same(&a1, &e1)
        };
        let mut kept = vec![];
        for v in vars {
            if let Some(rep) = kept.iter().copied().find(|u| shares(*u, v)) {
                alias.push((v, rep));
            } else {
                kept.push(v);
            }
        }
        vars = kept;
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
        if join_flag {
            // The flag is the first output. It is a literal or a joined flag
            // in the arm, not a carried input, so C7 does not drop it (B4).
            ts.insert(0, BTy::Bool);
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
                End::Fall | End::Exit => {
                    let mut o = vec![];
                    if join_flag {
                        o.push(arm.exit_flag.ok_or_else(|| self.deny("defect: a break arm has no flag"))?);
                    }
                    if vars.is_empty() && ts.first() == Some(ret_ty) && !join_flag {
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
        let caps: Vec<Src> = ins.iter().skip(1).copied().collect();
        let id = sc.push(Attr::Branch(2, ts), ins, regions);
        if !both_term {
            let rebound = |v: u32| vars.contains(&v) || alias.iter().any(|(a, _)| *a == v);
            // A nested branch captured an owned local and did not return it
            // under that name. The name was moved. Leaving the old source in
            // the parent makes the next output a second consumer.
            let mut moved = vec![];
            for (v, src) in &sc.env {
                if rebound(*v) || !caps.contains(src) {
                    continue;
                }
                if self.loc_ty(*v).ok().is_some_and(|t| t.owned()) {
                    moved.push(*v);
                }
            }
            let base = u32::from(join_flag);
            for (p, v) in vars.iter().enumerate() {
                sc.env.insert(*v, Src::Out(id, base + p as u32));
            }
            for (v, rep) in &alias {
                let Some(p) = vars.iter().position(|u| u == rep) else {
                    continue;
                };
                sc.env.insert(*v, Src::Out(id, base + p as u32));
            }
            for v in moved {
                sc.env.remove(&v);
            }
        }
        if needs_w {
            sc.omega = Some(Src::Out(id, (nout - 1) as u32));
        }
        if join_flag {
            sc.exit_flag = Some(Src::Out(id, 0));
        }
        if both_term {
            return Ok(End::Ret(Src::Out(id, 0)));
        }
        Ok(End::Fall)
    }

    /// The body's flag. A `break` that names this loop (depth `names` at this
    /// statement list, one deeper inside a block) makes the break path return
    /// the literal false. The continue path returns true, or a flag joined
    /// from a nested break, and it still returns the carried values.
    /// Statements after the break are not on the break path.
    ///
    /// `fall` runs only when this list falls through, at depth `fall_names`.
    /// It is the statements after a block the list is nested in.
    ///
    /// Option B: this is a flag encoding, not a preimage under Lean `Stmt`.
    fn exec_body(
        &self,
        stmts: &[Stmt],
        sc: &mut Scope,
        parent_omega: Option<Src>,
        ret_ty: &BTy,
        names: u32,
    ) -> Result<Src, Change> {
        let refs: Vec<&Stmt> = stmts.iter().collect();
        self.exec_body_seq(&refs, sc, parent_omega, ret_ty, names, &[], 0)
    }

    #[allow(clippy::too_many_arguments)]
    fn exec_body_seq(
        &self,
        stmts: &[&Stmt],
        sc: &mut Scope,
        parent_omega: Option<Src>,
        ret_ty: &BTy,
        names: u32,
        fall: &[&Stmt],
        fall_names: u32,
    ) -> Result<Src, Change> {
        let hits = |s: &&Stmt| has_break(std::slice::from_ref(*s), names);
        if !stmts.iter().any(hits) {
            if !stmts.is_empty() {
                let end = self.exec_refs(stmts, sc, parent_omega, ret_ty)?;
                if !matches!(end, End::Fall) {
                    return Err(self.deny("exit: a loop body does not fall through"));
                }
            }
            if fall.is_empty() {
                return Ok(lit_bool(sc, true));
            }
            return self.exec_body_seq(fall, sc, parent_omega, ret_ty, fall_names, &[], 0);
        }
        note_body_break();
        let k = stmts.iter().position(hits).expect("a break was just found");
        // A domain test before the break sinks the tail, including the break
        // condition, into its continue arm. Leave that tail in the arm that
        // falls, so the condition is in scope where the break is joined.
        if let Some(j) = (0..k).find(|&j| sinking_if(stmts[j])) {
            if j > 0 {
                let end = self.exec_refs(&stmts[..j], sc, parent_omega, ret_ty)?;
                if !matches!(end, End::Fall) {
                    return Err(self.deny("exit: a loop body does not fall through"));
                }
            }
            let Stmt::If(c, t, e) = stmts[j] else {
                return Err(self.deny("defect: a domain test was not an if"));
            };
            return self.nest_domain(
                c,
                t,
                e,
                &stmts[j + 1..],
                fall,
                fall_names,
                sc,
                parent_omega,
                ret_ty,
                names,
            );
        }
        if k > 0 {
            let end = self.exec_refs(&stmts[..k], sc, parent_omega, ret_ty)?;
            if !matches!(end, End::Fall) {
                return Err(self.deny("exit: a loop body does not fall through"));
            }
        }
        let after = &stmts[k + 1..];
        match stmts[k] {
            Stmt::Break(d) if *d == names => Ok(lit_bool(sc, false)),
            Stmt::Break(_) => Err(self.deny(
                "break: a break that does not name the loop is an exit this slice does not restructure",
            )),
            Stmt::If(c, t, e) => self.join_break(
                c,
                t,
                e,
                after,
                fall,
                fall_names,
                sc,
                parent_omega,
                ret_ty,
                names,
            ),
            Stmt::Block(b) => {
                if has_break(b, 0) {
                    return Err(self.deny(
                        "break: a block that is not a loop, and a break out of it, are an exit this slice does not restructure",
                    ));
                }
                let inner: Vec<&Stmt> = b.iter().collect();
                if after.is_empty() {
                    self.exec_body_seq(&inner, sc, parent_omega, ret_ty, names + 1, fall, fall_names)
                } else if !refs_fall(&inner) {
                    self.exec_body_seq(&inner, sc, parent_omega, ret_ty, names + 1, &[], 0)
                } else if !fall.is_empty() {
                    Err(self.deny(
                        "break: a break inside a nested block, with statements after that block and after an outer block, is an exit this slice does not restructure",
                    ))
                } else {
                    self.exec_body_seq(&inner, sc, parent_omega, ret_ty, names + 1, after, names)
                }
            }
            Stmt::Loop(_) => Err(self.deny(
                "break: a break that leaves more than one loop is an exit this slice does not restructure",
            )),
            _ => Err(self.deny(
                "break: a break outside a recognised loop is an exit this slice does not restructure",
            )),
        }
    }

    /// One arm of a break flag. The arm's scope captures `seed` from the parent.
    /// The returned flag is a literal false on a break, or the joined flag.
    #[allow(clippy::too_many_arguments)]
    fn body_arm(
        &self,
        stmts: &[&Stmt],
        fall: &[&Stmt],
        fall_names: u32,
        seed: &[u32],
        lits: &[(Ty, Lit, Src)],
        parent: &Scope,
        parent_omega: Option<Src>,
        thread: Option<Src>,
        ret_ty: &BTy,
        names: u32,
        false_lit: Option<Src>,
    ) -> Result<(End, Scope), Change> {
        let mut sc = Scope {
            nodes: vec![],
            env: HashMap::new(),
            omega: None,
            caps: vec![],
            assigned: BTreeSet::new(),
            set_at: vec![],
            lits: vec![],
            exit_flag: None,
        };
        for (j, l) in seed.iter().enumerate() {
            sc.env.insert(*l, Src::In(j as u32));
            sc.caps.push(parent.env[l]);
        }
        for (t, l, src) in lits {
            let j = sc.caps.len() as u32;
            sc.caps.push(*src);
            sc.lits.push((t.clone(), l.clone(), Src::In(j)));
        }
        if let Some(p) = thread {
            let j = sc.caps.len() as u32;
            sc.caps.push(p);
            sc.omega = Some(Src::In(j));
        }
        // A pure break returns the parent's literal false, captured, with no
        // node in the arm. The schedule's branch dummy is that same literal,
        // so construct(schedule(F)) rebuilds this capture instead of a second
        // copy inside the arm (B4: the flag is not a carried input).
        if fall.is_empty() && stmts.len() == 1 && matches!(stmts[0], Stmt::Break(d) if *d == names)
        {
            let src =
                false_lit.ok_or_else(|| self.deny("defect: a break flag has no literal false"))?;
            let j = sc.caps.len() as u32;
            sc.caps.push(src);
            sc.exit_flag = Some(Src::In(j));
            return Ok((End::Exit, sc));
        }
        let live = sc.omega.or(parent_omega);
        let flag = self.exec_body_seq(stmts, &mut sc, live, ret_ty, names, fall, fall_names)?;
        sc.exit_flag = Some(flag);
        let end = if refs_fall(stmts) && (fall.is_empty() || refs_fall(fall)) {
            End::Fall
        } else {
            End::Exit
        };
        Ok((end, sc))
    }

    /// Join a core `if` whose arms can break out of the loop into one Branch.
    /// The first output is the flag. Carried values the arms update follow it.
    #[allow(clippy::too_many_arguments)]
    fn join_break(
        &self,
        c: &Atom,
        t: &[Stmt],
        e: &[Stmt],
        within: &[&Stmt],
        fall: &[&Stmt],
        fall_names: u32,
        sc: &mut Scope,
        parent_omega: Option<Src>,
        ret_ty: &BTy,
        names: u32,
    ) -> Result<Src, Change> {
        let t_arm_falls = build::falls(t);
        let e_arm_falls = build::falls(e);
        let mut t_stmts: Vec<&Stmt> = t.iter().collect();
        let mut e_stmts: Vec<&Stmt> = e.iter().collect();
        if t_arm_falls {
            t_stmts.extend(within.iter().copied());
        }
        if e_arm_falls {
            e_stmts.extend(within.iter().copied());
        }
        let t_fall: &[&Stmt] = if t_arm_falls && refs_fall(within) {
            fall
        } else {
            &[]
        };
        let e_fall: &[&Stmt] = if e_arm_falls && refs_fall(within) {
            fall
        } else {
            &[]
        };
        let here = sc.omega.or(parent_omega);
        let mut tail_refs: Vec<&Stmt> = within.to_vec();
        tail_refs.extend(fall.iter().copied());
        let tail_needs = tail_refs.iter().any(|s| self.stmt_needs_order(s));
        let thread = if self.stmts_need_order(t) || self.stmts_need_order(e) || tail_needs {
            here
        } else {
            None
        };
        let lits = lits_already(sc, &[t, e], &tail_refs);
        let seed = self.seed_of(&[t, e], &tail_refs, sc);
        let sel = self.src_atom(sc, c)?;
        let false_lit = if !t_arm_falls || !e_arm_falls {
            Some(self.ensure_lit(sc, &Ty::Bool, &Lit::Bool(false))?)
        } else {
            None
        };
        let (e1, a1) = self.body_arm(
            &t_stmts,
            t_fall,
            fall_names,
            &seed,
            &lits,
            sc,
            here,
            thread,
            ret_ty,
            names,
            if t_arm_falls { None } else { false_lit },
        )?;
        let (e0, a0) = self.body_arm(
            &e_stmts,
            e_fall,
            fall_names,
            &seed,
            &lits,
            sc,
            here,
            thread,
            ret_ty,
            names,
            if e_arm_falls { None } else { false_lit },
        )?;
        self.finish_if(sc, sel, a0, e0, a1, e1, here, ret_ty, true)?;
        sc.exit_flag
            .take()
            .ok_or_else(|| self.deny("defect: a break flag was not joined"))
    }

    /// A domain test (one arm halts, the other falls) in front of a body break.
    /// The falling arm runs the rest of the body, including the break. The
    /// halt arm does not return a flag; its halt node has the region's output
    /// types, the same way a loop region that halts already does.
    #[allow(clippy::too_many_arguments)]
    fn nest_domain(
        &self,
        c: &Atom,
        t: &[Stmt],
        e: &[Stmt],
        tail: &[&Stmt],
        fall: &[&Stmt],
        fall_names: u32,
        sc: &mut Scope,
        parent_omega: Option<Src>,
        ret_ty: &BTy,
        names: u32,
    ) -> Result<Src, Change> {
        let t_falls = build::falls(t);
        let (fall_arm, halt_arm, fall_then) = if t_falls { (t, e, true) } else { (e, t, false) };
        let mut cont: Vec<&Stmt> = fall_arm.iter().collect();
        cont.extend(tail.iter().copied());
        let here = sc.omega.or(parent_omega);
        let mut tail_refs: Vec<&Stmt> = cont.clone();
        tail_refs.extend(fall.iter().copied());
        for s in halt_arm {
            tail_refs.push(s);
        }
        let tail_needs = tail_refs.iter().any(|s| self.stmt_needs_order(s));
        let thread = if self.stmts_need_order(halt_arm) || tail_needs {
            here
        } else {
            None
        };
        let lits = lits_already(sc, &[fall_arm, halt_arm], &tail_refs);
        let seed = self.seed_of(&[fall_arm, halt_arm], &tail_refs, sc);
        let sel = self.src_atom(sc, c)?;
        let (fall_end, fall_sc) = self.body_arm(
            &cont, fall, fall_names, &seed, &lits, sc, here, thread, ret_ty, names, None,
        )?;
        let (halt_end, halt_sc) = self.arm(halt_arm, &seed, &lits, sc, here, thread, ret_ty)?;
        if !matches!(halt_end, End::Halt(_)) {
            return Err(self.deny("exit: a loop body does not fall through"));
        }
        let (e0, a0, e1, a1) = if fall_then {
            (halt_end, halt_sc, fall_end, fall_sc)
        } else {
            (fall_end, fall_sc, halt_end, halt_sc)
        };
        self.finish_if(sc, sel, a0, e0, a1, e1, here, ret_ty, true)?;
        sc.exit_flag
            .take()
            .ok_or_else(|| self.deny("defect: a break flag was not joined"))
    }

    /// A loop the schedule did not write, in one of the lowering's shapes
    /// (`one/shape.rs`): a count bound, or a measure over a range or a list.
    /// `Ok(false)` means the statement list is neither; the caller keeps the
    /// `loop` tag. A recognised shape that still cannot be an Iterate returns
    /// the reason (`bound`, `break`, `exit`, `global`, …).
    fn try_lowered_loop(
        &self,
        body: &[Stmt],
        lo: u32,
        sc: &mut Scope,
        parent_omega: Option<Src>,
        ret_ty: &BTy,
    ) -> Result<bool, Change> {
        let (surface, brk) = match body {
            [Stmt::Block(x)] => (x.as_slice(), 1u32),
            other => (other, 0u32),
        };
        if let Some((at, count, end)) = find_count_span(surface, lo, &self.info) {
            let Stmt::Let(Atom::V(counter)) = &surface[at] else {
                return Ok(false);
            };
            let counter = *counter;
            let prefix = &surface[..at];
            let rest = &surface[end..];
            let mut used = BTreeSet::new();
            stmt_locals(prefix, &mut used);
            stmt_locals(rest, &mut used);
            if !used.contains(&counter) {
                self.build_count(prefix, rest, &count, brk, sc, parent_omega, ret_ty)?;
                return Ok(true);
            }
        }
        if let Some(spec) = find_measure(surface, brk, lo, &self.info) {
            self.build_measure(surface, &spec, sc, parent_omega, ret_ty, brk)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn open_region(
        &self,
        sc: &Scope,
        parent_omega: Option<Src>,
        vars: &[u32],
        thread: bool,
    ) -> Result<Scope, Change> {
        let mut region = Scope {
            nodes: vec![],
            env: HashMap::new(),
            omega: None,
            caps: vec![],
            assigned: BTreeSet::new(),
            set_at: vec![],
            lits: vec![],
            exit_flag: None,
        };
        let mut ix = 0u32;
        if thread {
            let p = sc
                .omega
                .or(parent_omega)
                .ok_or_else(|| self.deny("omega: a loop needs ω and the region has none"))?;
            region.omega = Some(Src::In(0));
            region.caps.push(p);
            ix = 1;
        }
        for v in vars {
            let p =
                sc.env.get(v).copied().ok_or_else(|| {
                    self.deny("defect: a carried var has no source before the loop")
                })?;
            region.env.insert(*v, Src::In(ix));
            region.caps.push(p);
            ix += 1;
        }
        Ok(region)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_count(
        &self,
        prefix: &[Stmt],
        rest: &[Stmt],
        count: &CountP,
        brk: u32,
        sc: &mut Scope,
        parent_omega: Option<Src>,
        ret_ty: &BTy,
    ) -> Result<(), Change> {
        let (test_stmts, flag_of) = peel_exit(prefix, brk);
        // A local the test defines and the body uses is not in the parent.
        // Carry a dummy initial the test overwrites, so the body sees the
        // test's value (the Iterate passes the test's outputs to the body).
        self.seed_flow(sc, test_stmts, rest, &[])?;
        let vars = carried_vars(sc, &[test_stmts, rest]);
        let thread = (sc.omega.is_some() || parent_omega.is_some())
            && (self.stmts_need_order(test_stmts)
                || self.stmts_need_order(rest)
                || sc.omega.is_some());
        // A count Iterate may halt, so it has no applied form unless a region
        // actually consumes ω (C7 drops a carried ω that is only passed through).
        if !thread {
            return Err(self.deny(
                "bound: a count-bounded Iterate must carry ω, and C7 removes a carried ω that neither region uses, so this pure count loop has no applied form",
            ));
        }
        let mut test = self.open_region(sc, parent_omega, &vars, true)?;
        let tw = test.omega;
        let end = self.exec(&[test_stmts], &mut test, tw, ret_ty)?;
        if !matches!(end, End::Fall) {
            return Err(self.deny("exit: a loop test does not fall through"));
        }
        let flag = match &flag_of {
            None => lit_bool(&mut test, true),
            Some((atom, neg)) => {
                let s = self.src_atom(&mut test, atom)?;
                if *neg { bool_not(&mut test, s) } else { s }
            }
        };
        let mut bod = self.open_region(sc, parent_omega, &vars, true)?;
        let bw = bod.omega;
        // A body break that names this loop is the flag. A pure count still
        // has no applied form: the flag is a bool and is not a use of ω (O8).
        let bflag = self.exec_body(rest, &mut bod, bw, ret_ty, brk)?;
        if !region_uses_order(&test) && !region_uses_order(&bod) {
            // The body may have been read as a flag. A pure count is still
            // outside Dom, so that reading is not an admission (O8).
            BODY_BREAK.with(|c| c.set(false));
            return Err(self.deny(
                "bound: a count-bounded Iterate must carry ω, and C7 removes a carried ω that neither region uses, so this pure count loop has no applied form",
            ));
        }
        let test_b = Body {
            nodes: std::mem::take(&mut test.nodes),
            out: region_outs(&test, flag, &vars, true, None)
                .map_err(|m| self.deny(format!("loop: {m}")))?,
        };
        let body_b = Body {
            nodes: std::mem::take(&mut bod.nodes),
            out: region_outs(&bod, bflag, &vars, true, None)
                .map_err(|m| self.deny(format!("loop: {m}")))?,
        };
        let bound = self.src_atom(sc, &count.bound)?;
        let mut ins = vec![
            bound,
            sc.omega
                .or(parent_omega)
                .ok_or_else(|| self.deny("defect: a count loop has no ω"))?,
        ];
        let mut carried_ty = vec![BTy::Order];
        for v in &vars {
            ins.push(sc.env[v]);
            carried_ty.push(bty(&self.loc_ty(*v)?).map_err(|e| self.deny(format!("defect: {e}")))?);
        }
        let id = sc.push(Attr::Count(carried_ty), ins, vec![test_b, body_b]);
        sc.omega = Some(Src::Out(id, 0));
        for (k, v) in vars.iter().enumerate() {
            sc.env.insert(*v, Src::Out(id, 1 + k as u32));
        }
        Ok(())
    }

    fn build_measure(
        &self,
        surface: &[Stmt],
        spec: &MeasureSpec,
        sc: &mut Scope,
        parent_omega: Option<Src>,
        ret_ty: &BTy,
        brk: u32,
    ) -> Result<(), Change> {
        let head = &surface[..spec.if_at];
        let tail = &surface[spec.if_at + 1..];
        // The index let is already the measure's carried index. Other
        // locals the test defines and the body uses are carried beside it.
        self.seed_flow(sc, head, tail, &[spec.iv])?;
        let mut vars = carried_vars(sc, &[surface]);
        if !vars.contains(&spec.ivar) {
            vars.push(spec.ivar);
        }
        let end_var = match &spec.end {
            Atom::V(v) if sc.env.contains_key(v) => Some(*v),
            Atom::V(v) => {
                return Err(self.deny(format!("defect: the measure end v{v} is not in scope")));
            }
            Atom::G(g) => {
                return Err(self.deny(format!(
                    "global: g{g} is not part of the working context ω holds"
                )));
            }
            Atom::K(_, _) => None,
        };
        if let Some(v) = end_var
            && !vars.contains(&v)
        {
            vars.push(v);
        }
        vars.sort();
        vars.dedup();
        let thread = (sc.omega.is_some() || parent_omega.is_some())
            && (self.stmts_need_order(head) || self.stmts_need_order(tail));
        let i_in = u32::from(thread) + vars.iter().position(|v| *v == spec.ivar).unwrap() as u32;
        let j_in = match end_var {
            Some(v) => u32::from(thread) + vars.iter().position(|x| *x == v).unwrap() as u32,
            None => u32::from(thread) + vars.len() as u32,
        };
        let mut test = self.open_region(sc, parent_omega, &vars, thread)?;
        test.env.insert(spec.iv, Src::In(i_in));
        let tw = test.omega;
        let end = self.exec(&[&surface[..spec.cmp]], &mut test, tw, ret_ty)?;
        if !matches!(end, End::Fall) {
            return Err(self.deny("exit: a measure test does not fall through"));
        }
        let Stmt::Prim(_, ts, _) = &surface[spec.cmp] else {
            return Err(self.deny("defect: a measure test is not a comparison"));
        };
        let args_ty = ts
            .iter()
            .map(ty_arg)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| self.deny(format!("defect: {e}")))?;
        let cmp = test.push(
            Attr::Prim("int.lt".into(), args_ty, None),
            vec![Src::In(i_in), Src::In(j_in)],
            vec![],
        );
        let flag = Src::Out(cmp, 0);
        if spec.cmp + 1 < spec.if_at {
            let tw = test.omega;
            let end = self.exec(&[&surface[spec.cmp + 1..spec.if_at]], &mut test, tw, ret_ty)?;
            if !matches!(end, End::Fall) {
                return Err(self.deny("exit: a measure test does not fall through"));
            }
        }
        if test.env.get(&spec.ivar) != Some(&Src::In(i_in)) {
            return Err(self.deny("loop: a measure test changes the index it compares"));
        }
        let mut bod = self.open_region(sc, parent_omega, &vars, thread)?;
        bod.env.insert(spec.iv, Src::In(i_in));
        let bw = bod.omega;
        // The continue path still has to raise the index. A break path
        // returns false and does not have to. A body that raises the index
        // only on some continuing paths is still not this shape (B5); the
        // gate rejects it, and this function does not rewrite it into a count.
        let bflag = self.exec_body(tail, &mut bod, bw, ret_ty, brk)?;
        let lit_slot = matches!(spec.end, Atom::K(_, _));
        let test_b = Body {
            nodes: std::mem::take(&mut test.nodes),
            out: region_outs(&test, flag, &vars, thread, lit_slot.then_some(j_in))
                .map_err(|m| self.deny(format!("loop: {m}")))?,
        };
        let body_b = Body {
            nodes: std::mem::take(&mut bod.nodes),
            out: region_outs(&bod, bflag, &vars, thread, lit_slot.then_some(j_in))
                .map_err(|m| self.deny(format!("loop: {m}")))?,
        };
        let mut ins = vec![];
        let mut carried_ty = vec![];
        if thread {
            ins.push(
                sc.omega
                    .or(parent_omega)
                    .ok_or_else(|| self.deny("defect: a measure has no ω"))?,
            );
            carried_ty.push(BTy::Order);
        }
        for v in &vars {
            ins.push(sc.env[v]);
            carried_ty.push(bty(&self.loc_ty(*v)?).map_err(|e| self.deny(format!("defect: {e}")))?);
        }
        if let Atom::K(t, l) = &spec.end {
            ins.push(self.lit_node(sc, t, l)?);
            carried_ty.push(bty(t).map_err(|e| self.deny(format!("defect: {e}")))?);
        }
        let id = sc.push(Attr::Measure(i_in, j_in, carried_ty), ins, vec![
            test_b, body_b,
        ]);
        let base = u32::from(thread);
        for (k, v) in vars.iter().enumerate() {
            sc.env.insert(*v, Src::Out(id, base + k as u32));
        }
        if thread {
            sc.omega = Some(Src::Out(id, 0));
        }
        Ok(())
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
            return if self.try_lowered_loop(body, lo, sc, parent_omega, ret_ty)? {
                Ok(())
            } else {
                Err(self.deny(loop_reason(body, &self.info, lo)))
            };
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
                set_at: vec![],
                lits: vec![],
                exit_flag: None,
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
                End::Exit => {
                    return Err(self.deny(
                        "break: a break outside a recognised loop is an exit this slice does not restructure",
                    ));
                }
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

/// Give both arms the same capture list: the union, in the order the first
/// arm already uses, then any capture only the second arm has. Inputs of
/// each arm are renumbered. Nested regions keep their own inputs.
fn reconcile_caps(a0: &mut Scope, a1: &mut Scope) {
    if a0.caps == a1.caps {
        return;
    }
    let mut union = a0.caps.clone();
    for s in &a1.caps {
        if !union.contains(s) {
            union.push(*s);
        }
    }
    renumber_caps(a0, &union);
    renumber_caps(a1, &union);
}

fn renumber_caps(arm: &mut Scope, union: &[Src]) {
    let map: Vec<u32> = arm
        .caps
        .iter()
        .map(|s| union.iter().position(|u| u == s).unwrap_or(0) as u32)
        .collect();
    let fix = |s: &mut Src, map: &[u32]| {
        if let Src::In(j) = s
            && let Some(n) = map.get(*j as usize)
        {
            *j = *n;
        }
    };
    for n in &mut arm.nodes {
        for i in &mut n.ins {
            fix(i, &map);
        }
    }
    for s in arm.env.values_mut() {
        fix(s, &map);
    }
    if let Some(s) = arm.omega.as_mut() {
        fix(s, &map);
    }
    arm.caps = union.to_vec();
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

struct MeasureSpec {
    ivar: u32,
    /// the local the head `let` binds, a copy of `ivar` for this iteration
    iv: u32,
    end: Atom,
    cmp: usize,
    if_at: usize,
}

fn push_lit(a: &Atom, out: &mut Vec<(Ty, Lit)>) {
    if let Atom::K(t, l) = a
        && !out.iter().any(|(tt, ll)| tt == t && ll == l)
    {
        out.push((t.clone(), l.clone()));
    }
}

fn stmt_lits(b: &[Stmt], out: &mut Vec<(Ty, Lit)>) {
    for s in b {
        match s {
            Stmt::Let(a) | Stmt::Var(_, a) | Stmt::Ret(a) | Stmt::Halt(a) | Stmt::Out(a) => {
                push_lit(a, out)
            }
            Stmt::Set(_, a) => push_lit(a, out),
            Stmt::Prim(_, _, xs) | Stmt::Call(_, xs) | Stmt::Catch(_, xs) | Stmt::Graph(_, xs) => {
                xs.iter().for_each(|a| push_lit(a, out))
            }
            Stmt::If(c, t, e) => {
                push_lit(c, out);
                stmt_lits(t, out);
                stmt_lits(e, out);
            }
            Stmt::Block(x) | Stmt::Loop(x) => stmt_lits(x, out),
            Stmt::Break(_) => {}
        }
    }
}

/// Literals mentioned in the arms that this region has already built.
fn lits_already(sc: &Scope, arms: &[&[Stmt]], tail: &[&Stmt]) -> Vec<(Ty, Lit, Src)> {
    let mut ks = vec![];
    for a in arms {
        stmt_lits(a, &mut ks);
    }
    for s in tail {
        stmt_lits(std::slice::from_ref(s), &mut ks);
    }
    ks.into_iter()
        .filter_map(|(t, l)| find_lit(sc, &t, &l).map(|src| (t, l, src)))
        .collect()
}

fn find_lit(sc: &Scope, t: &Ty, l: &Lit) -> Option<Src> {
    if let Some((_, _, s)) = sc.lits.iter().find(|(tt, ll, _)| tt == t && ll == l) {
        return Some(*s);
    }
    let bt = bty(t).ok()?;
    sc.nodes
        .iter()
        .enumerate()
        .find_map(|(i, n)| match &n.attrs {
            Attr::Lit(ty, lit) if ty == &bt && lit == l => Some(Src::Out(i as u32, 0)),
            _ => None,
        })
}

/// Locals bound by statements that run on every path through `stmts`.
/// A binding inside `if` or `loop` does not, because the body would then
/// see the dummy initial on the path that skipped it.
/// Drops and stores of values the parent already holds: the schedule's
/// record of a branch's outputs, not a computation that belongs in the arm.
fn tail_is_join(rest: &[&Stmt], parent: &Scope) -> bool {
    if rest.is_empty() {
        return false;
    }
    let mut used = BTreeSet::new();
    for s in rest {
        match s {
            Stmt::Set(Place::V(_), a) => atom_locals(a, &mut used),
            Stmt::Prim(p, _, xs) if p == "drop" => {
                for a in xs {
                    atom_locals(a, &mut used);
                }
            }
            _ => return false,
        }
    }
    used.iter().all(|v| parent.env.contains_key(v))
}

fn straight_bound(stmts: &[Stmt], info: &Info, out: &mut BTreeSet<u32>) {
    for st in stmts {
        match st {
            Stmt::Block(x) => straight_bound(x, info, out),
            Stmt::If(_, _, _) | Stmt::Loop(_) | Stmt::Break(_) => {}
            _ => {
                let Some(&a) = info.first.get(&(st as *const Stmt)) else {
                    continue;
                };
                let n = match st {
                    Stmt::Let(_)
                    | Stmt::Var(_, _)
                    | Stmt::Call(_, _)
                    | Stmt::Catch(_, _)
                    | Stmt::Graph(_, _) => 1,
                    Stmt::Prim(p, ts, _) => core::prim_results(p, ts) as u32,
                    _ => 0,
                };
                for i in 0..n {
                    out.insert(a + i);
                }
            }
        }
    }
}

fn carried_vars(sc: &Scope, parts: &[&[Stmt]]) -> Vec<u32> {
    let mut used = BTreeSet::new();
    for p in parts {
        stmt_locals(p, &mut used);
    }
    used.retain(|v| sc.env.contains_key(v));
    used.into_iter().collect()
}

fn lit_bool(region: &mut Scope, v: bool) -> Src {
    let i = region.push(Attr::Lit(BTy::Bool, Lit::Bool(v)), vec![], vec![]);
    Src::Out(i, 0)
}

fn bool_not(region: &mut Scope, s: Src) -> Src {
    let i = region.push(Attr::Prim("bool.not".into(), vec![], None), vec![s], vec![]);
    Src::Out(i, 0)
}

fn region_uses_order(region: &Scope) -> bool {
    matches!(region.omega, Some(Src::Out(..)))
        || region.nodes.iter().any(|n| n.ins.contains(&Src::In(0)))
}

fn region_outs(
    region: &Scope,
    flag: Src,
    vars: &[u32],
    thread: bool,
    lit: Option<u32>,
) -> Result<Vec<Src>, String> {
    let mut out = vec![flag];
    if thread {
        out.push(region.omega.ok_or("a loop region lost ω")?);
    }
    for v in vars {
        out.push(*region.env.get(v).ok_or("a carried value is not returned")?);
    }
    if let Some(j) = lit {
        out.push(Src::In(j));
    }
    Ok(out)
}

fn is_break_arm(arm: &[Stmt], brk: u32) -> bool {
    matches!(arm, [Stmt::Break(d)] if *d == brk)
}

/// `(statements to run, Some((condition, negated)))`. Negated means the arm
/// breaks when the condition is true, so the continue flag is its negation.
fn peel_exit(prefix: &[Stmt], brk: u32) -> (&[Stmt], Option<(Atom, bool)>) {
    if let Some(Stmt::If(c, t, e)) = prefix.last() {
        if t.is_empty() && is_break_arm(e, brk) {
            return (&prefix[..prefix.len() - 1], Some((c.clone(), false)));
        }
        if e.is_empty() && is_break_arm(t, brk) {
            return (&prefix[..prefix.len() - 1], Some((c.clone(), true)));
        }
    }
    (prefix, None)
}

fn find_count_span(body: &[Stmt], lo: u32, info: &Info) -> Option<(usize, CountP, usize)> {
    (0..body.len()).find_map(|k| parse_count(&body[k..], &[], lo, info).map(|(c, n)| (k, c, k + n)))
}

fn stmt_key(s: &Stmt) -> *const Stmt {
    s as *const Stmt
}

/// The head of a lowering `for`: read the index, compare it with an end,
/// leave when the comparison fails. The end is the second operand of
/// `int.lt`, or the first of `int.le`.
fn find_measure(stmts: &[Stmt], brk: u32, lo: u32, info: &Info) -> Option<MeasureSpec> {
    let mut k = 0;
    while matches!(stmts.get(k), Some(Stmt::Call(..))) {
        k += 1;
    }
    let Stmt::Let(Atom::V(var)) = stmts.get(k)? else {
        return None;
    };
    let ivar = *var;
    if ivar >= lo || !info.loc.get(&ivar).is_some_and(|l| l.var) {
        return None;
    }
    let iv = *info.first.get(&stmt_key(&stmts[k]))?;
    k += 1;
    while matches!(stmts.get(k), Some(Stmt::Call(..))) {
        k += 1;
    }
    let Stmt::Prim(op, _, xs) = stmts.get(k)? else {
        return None;
    };
    if xs.len() != 2 {
        return None;
    }
    let direct = op == "int.lt" && xs[0] == Atom::V(iv);
    let swapped = op == "int.le" && xs[1] == Atom::V(iv);
    if !direct && !swapped {
        return None;
    }
    let end = if direct { xs[1].clone() } else { xs[0].clone() };
    let cmp = k;
    k += 1;
    while matches!(stmts.get(k), Some(Stmt::Call(..))) {
        k += 1;
    }
    let Stmt::If(c, t, e) = stmts.get(k)? else {
        return None;
    };
    let cmp_local = *info.first.get(&stmt_key(&stmts[cmp]))?;
    if *c != Atom::V(cmp_local) {
        return None;
    }
    let leaves = if direct {
        t.is_empty() && is_break_arm(e, brk)
    } else {
        e.is_empty() && is_break_arm(t, brk)
    };
    if !leaves || k + 1 >= stmts.len() {
        return None;
    }
    Some(MeasureSpec {
        ivar,
        iv,
        end,
        cmp,
        if_at: k,
    })
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
    BODY_BREAK.with(|c| c.set(false));
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
            set_at: vec![],
            lits: vec![],
            exit_flag: None,
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
            End::Exit => {
                return Err(deny(
                    "construction",
                    &at,
                    "break: a break outside a recognised loop is an exit this slice does not restructure",
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

/// The core module of the user function `name` and the functions it calls,
/// plus any shim those calls name. `None` when no user function has that
/// name. A joined concept name matches when one of its parts is `name`.
pub fn closure(m: &core::Module, name: &str) -> Option<core::Module> {
    let users = users_of(m);
    let mut by_id: HashMap<(String, u32), usize> = HashMap::new();
    for (n, u) in users.iter().enumerate() {
        by_id.insert((u.group.clone(), u.index), n);
    }
    let n = users
        .iter()
        .position(|u| u.name.split(" = ").any(|p| p == name))?;
    let mut need: Vec<Vec<usize>> = vec![vec![]; users.len()];
    for (i, u) in users.iter().enumerate() {
        let mut cs = vec![];
        calls_of(&u.func.body, &mut cs);
        for r in cs {
            if let FnRef::Key(k, idx) = r
                && let Some(j) = by_id.get(&(k, idx))
            {
                need[i].push(*j);
            }
        }
    }
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
    let mut sub = submodule(m, &users, &ids);
    // W1: the entry takes no parameters. A closure whose root takes some
    // gets a trivial entry that returns 0, as `schedule` does.
    let entry_ok = sub
        .func(&sub.entry.0, sub.entry.1)
        .is_some_and(|f| f.params.is_empty());
    if !entry_ok {
        let main = core::Fn {
            params: vec![],
            ret: Ty::i64(),
            body: vec![Stmt::Ret(build::lit(0))],
        };
        let (g, _) = canonical_group(std::slice::from_ref(&main));
        let key = group_key(&g);
        if let Some(map) = sub.names.get_mut("fns").and_then(|v| v.as_object_mut()) {
            map.insert(key.clone(), serde_json::json!(["main"]));
        }
        sub.defs.insert(key.clone(), g);
        sub.entry = (key, 0);
    }
    Some(sub)
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
            // Both regions store every carried var, so the construction sees
            // one list. When the region returned the var itself, `set v = v`
            // moves it onto itself; a drop first would use it twice.
            let same = matches!(a, Atom::V(id) if id == var);
            if !same && sch.tys[var].owned() && sch.live.contains(var) {
                s.push(Stmt::Prim("drop".into(), vec![sch.tys[var].clone()], vec![
                    Atom::V(*var),
                ]));
                sch.live.remove(var);
            }
            if let Atom::V(id) = a
                && *id != *var
            {
                sch.live.remove(id);
            }
            s.push(Stmt::Set(Place::V(*var), a.clone()));
            if sch.tys[var].owned() {
                sch.live.insert(*var);
            }
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
        // The sum is halted inside the arm, so it is not live after the loop.
        // Marking it live made the schedule drop it where it is not in scope.
        let got = sch.fresh(Ty::halt(), false);
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
    // HashMap iteration is not the schedule. Prefer a function that already
    // takes no parameters, and otherwise the least core key, so the same Form
    // always schedules the same entry.
    let mut entry = form_core
        .values()
        .filter(|k| defs[*k].fns[0].params.is_empty())
        .cloned()
        .min()
        .or_else(|| form_core.values().cloned().min())
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
