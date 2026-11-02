//! Mutation adequacy (spec 6.12, `spec/proposals/mutation-adequacy.md`).
//!
//! An obligation (an `example`, a `check` or a `law`) carries information
//! about a program only to the extent that it can fail when the program
//! changes. The **mutant set** of a program is every single edit of rung d
//! of `rce repair` (`repair::small_edits`: an operator swapped within its
//! class, an integer literal moved to n + 1, n - 1, 0 or 1, a boolean
//! flipped, the operands of an ordering comparison swapped, two arguments of
//! a call swapped, a condition negated) at every site of every definition of
//! the program's own text (functions, methods, steps, actors, constants and
//! state declarations), never at an example, check or law line and never in
//! an imported module. The order is fixed (definitions in source order,
//! sites in preorder, edits in the order `small_edits` lists them) and only
//! the first MUTANT_BOUND mutants are judged.
//!
//! A mutant is **in Form** when the checker gives it the reports the program
//! has (none: the program is admitted) and its constants and initial state
//! evaluate. An obligation **kills** an in-Form mutant when it fails on it.
//! A mutant no obligation kills is **equivalent** when `rce equiv` judges it
//! to have the program's behaviour on the whole explored graph, and a
//! **survivor** otherwise. Killing beats equivalence: an obligation that
//! fails on a mutant observed a difference, so a killed mutant is never
//! excluded.
//!
//! The **reach** of an obligation is the definitions its run can read: for
//! a check, the functions and constants it names and what they name in
//! turn; for an example or a law, every step, actor and state declaration
//! and what they name. A mutant outside the reach leaves the run unchanged,
//! so the obligation is not run on it. The **core** is the part of the
//! reach that measures the obligation for W-MUT-1: for a check its reach;
//! for an example the steps its events can select and what they name; for a
//! law every step and what it names.
//!
//! Each obligation runs on a mutant with a fresh fuel budget of twice what
//! it took on the program plus MUTANT_FUEL_SLACK, never more than the fuel
//! of 6.7. A run that exhausts a budget below that fuel is **undecided** for
//! that obligation: neither a kill nor a hold. A law runs on a mutant with a
//! state bound of four times the states it explored on the program (at
//! least 10,000, at most LAW_FUEL).
//!
//! Two rules use the result. W-MUT-1 (requires change): a measured
//! obligation (one that names a definition of the program, fuel at most
//! MUTANT_OBLIGATION_FUEL, a law's steps at most MUTANT_LAW_STEPS) whose
//! core has an in-Form mutant, whose reach is within the bound, and that
//! holds, decided, on every in-Form mutant of its reach. W-MUT-2
//! (recommended): a survivor.
use crate::ast::*;
use crate::interp::{FUEL, Interp, Measure, Region, TRAP_FUEL, V};
use crate::repair::{Unit, apply, callable_name, diff, mentions, sites, small_edits, unit_name, units, visit_expr, visit_unit};
use crate::unparse::{expr as show, unparse};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// The number of mutants judged, a constant of the language (6.12) beside
/// LAW_FUEL: `rce check` judges at most the first MUTANT_BOUND mutants in
/// order, so admission depends only on the program text. It is the default
/// bound of rung d of `rce repair`.
pub const MUTANT_BOUND: usize = crate::repair::DEFAULT_BOUND;

/// Fuel added to twice an obligation's own fuel for its run on a mutant.
pub const MUTANT_FUEL_SLACK: u64 = 100_000;

/// What W-MUT-1 may spend on one obligation, in fuel (6.7): an obligation
/// is **measured** when the mutants of its reach times its cost (the fuel it
/// took on the program plus MUTANT_RUN_COST) are at most this, so that even
/// a run on every mutant of its reach stays within one FUEL. `rce mutants`
/// reports what an unmeasured obligation kills, and W-MUT-1 does not apply
/// to it.
pub const MUTANT_GATE_FUEL: u64 = crate::interp::FUEL;

/// The fuel a judgement of a mutant counts beyond the run itself (the
/// checker's pass over the mutant).
pub const MUTANT_RUN_COST: u64 = 10_000;

/// For a law, the cost is the steps (δ applications) its exploration took
/// on the program, and the mutants of its reach times that is at most this.
pub const MUTANT_LAW_STEPS: u64 = 1_000_000;

/// The state bound of `rce equiv` when it compares a survivor with the
/// program (10.11's default is 100,000; a mutant is a one-site change).
pub const EQUIV_FUEL: usize = 20_000;

/// One obligation of the program's own text.
#[derive(Clone, Debug)]
pub struct Obligation {
    pub kind: &'static str,
    pub line: usize,
    pub text: String,
    what: Ob,
}

#[derive(Clone, Debug)]
enum Ob {
    Ex(Example),
    Ck(Expr),
    Law(String),
}

/// The verdict of one obligation on one program.
#[derive(Clone, Debug, PartialEq)]
enum Res {
    Holds,
    Fails,
    Undecided,
}

/// One mutant: a definition, a site in it and the edit.
#[derive(Clone, Debug)]
pub struct Mutant {
    unit: usize,
    site: usize,
    repl: Expr,
    pub definition: String,
    pub edit: String,
    pub line: usize,
}

/// What happened to one mutant.
#[derive(Clone, Debug, PartialEq)]
pub enum Fate {
    /// the checker reports it, or a constant or the initial state traps
    NotInForm(String),
    /// the obligations (by index) that fail on it
    Killed(Vec<usize>),
    /// killed by none, and `rce equiv` judges it equivalent (exact)
    Equivalent,
    /// killed by none, with an obligation undecided on it
    Undecided(Vec<usize>),
    /// killed by none: the witness (events, the program's output, the
    /// mutant's) where `rce equiv` gives one, else why it was not compared
    Survives { witness: Option<(Vec<String>, String, String)>, why: Option<String> },
}

/// An admitted program as `rce check` linked and checked it.
pub struct Program<'a> {
    /// the program with its imports linked in (4.1)
    pub linked: &'a Module,
    /// how many of its items are the program's own text (the rest are
    /// imported, appended by the linker)
    pub own: usize,
    pub imported_fns: &'a HashSet<String>,
    pub imported_types: &'a HashSet<String>,
    pub known: &'a HashSet<String>,
}

struct Prep<'a> {
    p: &'a Program<'a>,
    obs: Vec<Obligation>,
    /// fuel (or, for a law, states) each obligation took on the program
    base: Vec<u64>,
    /// its cost: fuel, or for a law the steps its exploration took
    cost: Vec<u64>,
    units: Vec<Unit>,
    reach: Vec<BTreeSet<usize>>,
    /// per obligation: the mutants (indices) that measure it for W-MUT-1
    measured: Vec<BTreeSet<usize>>,
    /// whether the obligation names a definition at all (own or imported)
    reads: Vec<bool>,
    mutants: Vec<Mutant>,
    /// single edits of the program before the bound
    total: usize,
    /// the checker's reports with all own bodies but one skipped, by the
    /// one kept
    baseline: HashMap<Option<String>, Vec<String>>,
    /// the value of every constant of the program, and per constant the
    /// own definitions its value depends on (itself included)
    consts0: HashMap<String, V>,
    const_deps: Vec<(String, BTreeSet<usize>)>,
    /// the own definitions the initial state depends on
    state_deps: BTreeSet<usize>,
    /// per mutant judged so far: whether it is in Form
    form: HashMap<usize, Result<(), String>>,
    /// the linked program, into which `flip` puts one mutant's edit at a time
    work: Option<Module>,
}

/// Every obligation of the first `own` items, in text order.
fn obligations(m: &Module, own: usize) -> Vec<Obligation> {
    let mut v = vec![];
    let ex = |x: &Example| Obligation { kind: "example", line: x.line, text: crate::unparse::example(x), what: Ob::Ex(x.clone()) };
    let ck = |e: &Expr| Obligation { kind: "check", line: e.line(), text: format!("check {}", show(e)), what: Ob::Ck(e.clone()) };
    for it in m.items.iter().take(own) {
        match it {
            Item::Example(x) => v.push(ex(x)),
            Item::Check(e) => v.push(ck(e)),
            Item::Intent(_, xs, cs) => {
                v.extend(xs.iter().map(ex));
                v.extend(cs.iter().map(ck));
            }
            Item::Law(l) => v.push(Obligation { kind: "law", line: l.line, text: format!("law {}", l.name), what: Ob::Law(l.name.clone()) }),
            _ => {}
        }
    }
    v.sort_by_key(|o| o.line);
    v
}

/// The names of every definition of a module: items, methods, state fields.
fn definition_names(m: &Module) -> BTreeSet<String> {
    let mut s = BTreeSet::new();
    for it in &m.items {
        if let Some(n) = crate::link::item_name(it) {
            s.insert(n);
        }
        match it {
            Item::Impl { members, .. } => s.extend(members.iter().map(|f| f.sig.name.clone())),
            Item::State { fields, .. } => s.extend(fields.iter().map(|f| f.name.clone())),
            _ => {}
        }
    }
    s
}

fn law_names(l: &Law) -> BTreeSet<String> {
    let mut v = vec![];
    match &l.claim {
        LawClaim::Always(p) => v.push(p.clone()),
        LawClaim::Every { guard, outcome, .. } => {
            v.extend(guard.clone());
            if let LawOutcome::Expr(e) = outcome {
                v.push(e.clone());
            }
        }
    }
    v.iter().flat_map(crate::repair::expr_names).collect()
}

/// Whether a step can be selected by one of the events `evs` (an actor:
/// always, since a handler may be).
fn selects(m: &Module, u: &Unit, evs: &BTreeSet<&str>) -> bool {
    fn names(p: &Pat) -> Option<Vec<String>> {
        match p {
            Pat::Path(q) | Pat::Ctor(q, _) => Some(q.last().cloned().into_iter().collect()),
            Pat::Alt(ps) => {
                let mut v = vec![];
                for p in ps {
                    v.extend(names(p)?);
                }
                Some(v)
            }
            _ => None,
        }
    }
    match u {
        Unit::Step(i) => match &m.items[*i] {
            Item::Step { pat, .. } => names(pat).is_none_or(|ns| ns.iter().any(|n| evs.contains(n.as_str()))),
            _ => false,
        },
        _ => true,
    }
}

/// The module that holds only the item of `u`, and `u` in it: the walkers
/// of `repair` clone the module they walk, and one item is enough.
fn slim(m: &Module, u: &Unit) -> (Module, Unit) {
    let (i, v) = match u {
        Unit::Fn(i) => (*i, Unit::Fn(0)),
        Unit::ImplFn(i, j) => (*i, Unit::ImplFn(0, *j)),
        Unit::Step(i) => (*i, Unit::Step(0)),
        Unit::Actor(i) => (*i, Unit::Actor(0)),
        Unit::Const(i) => (*i, Unit::Const(0)),
        Unit::State(i) => (*i, Unit::State(0)),
    };
    let one = Module { path: None, items: vec![m.items[i].clone()], pubs: BTreeSet::new(), v09_hygiene: vec![], x09: Default::default() };
    (one, v)
}

fn item_of(u: &Unit) -> usize {
    match u {
        Unit::Fn(i) | Unit::ImplFn(i, _) | Unit::Step(i) | Unit::Actor(i) | Unit::Const(i) | Unit::State(i) => *i,
    }
}

/// The own definitions of a linked program, in source order.
fn own_units(m: &Module, own: usize) -> Vec<Unit> {
    units(m).into_iter().filter(|u| item_of(u) < own).collect()
}

/// The expressions of the own definitions, by address: per definition in
/// the order the sites of `repair` count them (`visit_unit`), since the
/// coverage admission records (6.12) names expressions by address; and the
/// expressions inside an `emit`, whose value is an output and never part of
/// the state or the outcome an example states.
pub struct Sites {
    pub addrs: Vec<Vec<usize>>,
    pub emitted: HashSet<usize>,
}

/// The `Sites` of a linked program. Called on the module that admission
/// then evaluates, before it does.
pub fn site_addresses(lm: &mut Module, own: usize) -> Sites {
    let us = own_units(lm, own);
    let mut addrs = vec![];
    for u in &us {
        let mut v = vec![];
        let mut f = |e: &mut Expr, _: bool| {
            v.push(e as *const Expr as usize);
            false
        };
        visit_unit(lm, u, &mut f);
        addrs.push(v);
    }
    let mut emitted = HashSet::new();
    fn stmts(b: &mut [Stmt], out: &mut HashSet<usize>) {
        for s in b {
            match s {
                Stmt::Emit(e) => {
                    let mut f = |x: &mut Expr, _: bool| {
                        out.insert(x as *const Expr as usize);
                        false
                    };
                    visit_expr(e, &mut f);
                }
                Stmt::If(arms, el) => {
                    for (_, b) in arms {
                        stmts(b, out);
                    }
                    if let Some(b) = el {
                        stmts(b, out);
                    }
                }
                Stmt::While(_, _, b) | Stmt::Loop(_, b) | Stmt::For(_, _, b, _) | Stmt::Arena(_, _, b, _) | Stmt::Device(_, _, b, _) | Stmt::Comptime(b) => stmts(b, out),
                Stmt::Match(_, arms, _) => {
                    for a in arms {
                        stmts(&mut a.body, out);
                    }
                }
                Stmt::Select(arms, _) => {
                    for a in arms {
                        match a {
                            SelArm::Recv(_, _, b) | SelArm::Send(_, _, b) | SelArm::After(_, b) | SelArm::Default(b) => stmts(b, out),
                        }
                    }
                }
                _ => {}
            }
        }
    }
    let body = |b: &mut Body, out: &mut HashSet<usize>| {
        if let Body::Block(s) = b {
            stmts(s, out);
        }
    };
    for it in lm.items.iter_mut().take(own) {
        match it {
            Item::Fn(f) => body(&mut f.body, &mut emitted),
            Item::Impl { members, .. } => members.iter_mut().for_each(|f| body(&mut f.body, &mut emitted)),
            Item::Step { body: b, .. } => body(b, &mut emitted),
            Item::Actor { handlers, fns, .. } => {
                handlers.iter_mut().for_each(|h| body(&mut h.body, &mut emitted));
                fns.iter_mut().for_each(|f| body(&mut f.body, &mut emitted));
            }
            _ => {}
        }
    }
    Sites { addrs, emitted }
}

impl<'a> Prep<'a> {
    fn new(p: &'a Program<'a>, bound: usize, measure: &Measure, sites_of: &Sites) -> Prep<'a> {
        let addrs = &sites_of.addrs;
        let lm = p.linked;
        let obs = obligations(lm, p.own);
        let us = own_units(lm, p.own);
        let mut by_name: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, u) in us.iter().enumerate() {
            if let Some(n) = callable_name(lm, u) {
                by_name.entry(n).or_default().push(i);
            }
        }
        let ment: Vec<BTreeSet<String>> = us
            .iter()
            .map(|u| {
                let (one, v) = slim(lm, u);
                mentions(&v, &one)
            })
            .collect();
        let pick = |f: fn(&Unit) -> bool| -> Vec<usize> { us.iter().enumerate().filter(|(_, u)| f(u)).map(|(i, _)| i).collect() };
        let behaviour = pick(|u| matches!(u, Unit::Step(_) | Unit::Actor(_) | Unit::State(_)));
        let steps = pick(|u| matches!(u, Unit::Step(_) | Unit::Actor(_)));
        let close = |roots: &BTreeSet<String>, start: &[usize]| -> BTreeSet<usize> {
            let mut out: BTreeSet<usize> = start.iter().copied().collect();
            for n in roots {
                out.extend(by_name.get(n).into_iter().flatten().copied());
            }
            let mut todo: Vec<usize> = out.iter().copied().collect();
            while let Some(u) = todo.pop() {
                for n in &ment[u] {
                    for &v in by_name.get(n).into_iter().flatten() {
                        if out.insert(v) {
                            todo.push(v);
                        }
                    }
                }
            }
            out
        };
        let defs = definition_names(lm);
        let (mut reach, mut core, mut reads) = (vec![], vec![], vec![]);
        for o in &obs {
            match &o.what {
                Ob::Ck(e) => {
                    let names = crate::repair::expr_names(e);
                    reads.push(names.iter().any(|n| defs.contains(n)));
                    let r = close(&names, &[]);
                    core.push(r.clone());
                    reach.push(r);
                }
                Ob::Ex(x) => {
                    reads.push(true);
                    reach.push(close(&BTreeSet::new(), &behaviour));
                    let evs: BTreeSet<&str> = x.events.iter().map(|(e, _)| e.as_str()).collect();
                    let sel: Vec<usize> = steps.iter().copied().filter(|&i| selects(lm, &us[i], &evs)).collect();
                    core.push(close(&BTreeSet::new(), &sel));
                }
                Ob::Law(n) => {
                    reads.push(true);
                    let names = lm.items.iter().find_map(|i| if let Item::Law(l) = i && &l.name == n { Some(law_names(l)) } else { None }).unwrap_or_default();
                    reach.push(close(&names, &behaviour));
                    core.push(close(&names, &steps));
                }
            }
        }
        let mut mutants = vec![];
        let mut total = 0;
        for (ui, u) in us.iter().enumerate() {
            let (one, v) = slim(lm, u);
            for (site, d, r) in sites(&one, &v, &small_edits) {
                total += 1;
                if mutants.len() < bound {
                    let line = r.line();
                    mutants.push(Mutant { unit: ui, site, repl: r, definition: unit_name(lm, u), edit: d, line });
                }
            }
        }
        let mut base = vec![];
        let mut cost = vec![];
        for o in &obs {
            let (b, c) = match o.what {
                Ob::Law(_) => measure.laws.get(&o.line).map(|&(st, sp)| (st as u64, sp as u64)).unwrap_or((0, u64::MAX)),
                _ => measure.fuel.get(&o.line).map(|&f| (f, f)).unwrap_or((0, u64::MAX)),
            };
            base.push(b);
            cost.push(c);
        }
        let mut const_deps = vec![];
        for (i, u) in us.iter().enumerate() {
            if let Unit::Const(_) = u
                && let Some(n) = callable_name(lm, u)
            {
                const_deps.push((n, close(&BTreeSet::new(), &[i])));
            }
        }
        let state_deps = close(&BTreeSet::new(), &pick(|u| matches!(u, Unit::State(_))));
        let mut it = Interp::new(lm);
        for i in &lm.items {
            if let Item::Const { name, .. } = i {
                it.fuel = FUEL;
                let _ = it.const_value(name);
            }
        }
        let consts0 = it.consts.clone();
        // the mutants that measure each obligation: at a definition of its
        // core (not a constant or a state declaration), and, for a check or
        // an example, at an expression its run on the program evaluated;
        // for an example, not inside an `emit` (it states no output)
        let mut measured = vec![];
        for (o, ob) in obs.iter().enumerate() {
            let cov = match ob.what {
                Ob::Law(_) => None,
                _ => Some(measure.cover.get(&ob.line)),
            };
            let set: BTreeSet<usize> = (0..mutants.len())
                .filter(|&k| {
                    let m = &mutants[k];
                    core[o].contains(&m.unit)
                        && !matches!(us[m.unit], Unit::Const(_) | Unit::State(_))
                        && cov.is_none_or(|c| c.is_some_and(|c| addrs.get(m.unit).and_then(|v| v.get(m.site)).is_some_and(|a| c.contains(a))))
                        && !(matches!(ob.what, Ob::Ex(_)) && addrs.get(m.unit).and_then(|v| v.get(m.site)).is_some_and(|a| sites_of.emitted.contains(a)))
                })
                .collect();
            measured.push(set);
        }
        Prep { p, obs, base, cost, units: us, reach, measured, reads, mutants, total, baseline: HashMap::new(), consts0, const_deps, state_deps, form: HashMap::new(), work: Some(lm.clone()) }
    }

    /// Whether W-MUT-1 measures obligation `o` (MUTANT_GATE_FUEL).
    fn light(&self, o: usize) -> bool {
        let n = self.mutants.iter().filter(|m| self.reach[o].contains(&m.unit)).count() as u64;
        match self.obs[o].what {
            Ob::Law(_) => n.saturating_mul(self.cost[o]) <= MUTANT_LAW_STEPS,
            _ => n.saturating_mul(self.cost[o].saturating_add(MUTANT_RUN_COST)) <= MUTANT_GATE_FUEL,
        }
    }

    /// The checker's reports on a linked module with every own function
    /// body but `keep`'s skipped: a mutant changes one body, and the others
    /// get the reports they had on the program.
    fn reports(&self, lm: &Module, keep: &Option<String>) -> Vec<String> {
        let mut c = crate::check::Checker::new();
        c.imported_fns = self.p.imported_fns.clone();
        for it in lm.items.iter().take(self.p.own) {
            if let Item::Fn(f) = it
                && Some(&f.sig.name) != keep.as_ref()
            {
                c.imported_fns.insert(f.sig.name.clone());
            }
        }
        c.imported_types = self.p.imported_types.clone();
        c.check_module(lm, self.p.known);
        c.errs
    }

    fn mutant(&self, k: usize) -> Module {
        let mu = &self.mutants[k];
        apply(self.p.linked, &self.units[mu.unit], mu.site, &mu.repl)
    }

    /// Put mutant `k`'s edit into the working copy, or take it out again:
    /// the expression at its site and its replacement trade places, so two
    /// calls leave both as they were.
    fn flip(&mut self, k: usize) {
        let mut w = self.work.take().expect("working copy");
        let u = self.units[self.mutants[k].unit].clone();
        let site = self.mutants[k].site;
        let mut r = std::mem::replace(&mut self.mutants[k].repl, Expr::Unit(0));
        let mut i = 0usize;
        let mut f = |e: &mut Expr, _: bool| {
            if i == site {
                std::mem::swap(e, &mut r);
                return true;
            }
            i += 1;
            false
        };
        visit_unit(&mut w, &u, &mut f);
        self.mutants[k].repl = r;
        self.work = Some(w);
    }

    /// Whether mutant `k` is in Form (the checker's reports are the
    /// program's, and the constants and the initial state its definition
    /// feeds evaluate): `Err` is the report that puts it out of Form.
    fn in_form(&mut self, k: usize) -> Result<(), String> {
        match self.form.get(&k) {
            Some(Ok(())) => return Ok(()),
            Some(Err(e)) => return Err(e.clone()),
            None => {}
        }
        let u = self.mutants[k].unit;
        let keep = match &self.units[u] {
            Unit::Fn(i) => crate::link::item_name(&self.p.linked.items[*i]),
            _ => None,
        };
        if !self.baseline.contains_key(&keep) {
            let b = self.reports(self.p.linked, &keep);
            self.baseline.insert(keep.clone(), b);
        }
        self.flip(k);
        let ms = self.work.as_ref().expect("working copy");
        let got = self.reports(ms, &keep);
        let base = &self.baseline[&keep];
        let mut verdict = if &got != base { Err(got.iter().find(|e| !base.contains(e)).cloned().unwrap_or_else(|| "the checker's reports changed".into())) } else { Ok(()) };
        if verdict.is_ok() {
            let mut it = Interp::new(ms);
            it.consts = self.seed(u);
            for (n, deps) in &self.const_deps {
                if deps.contains(&u) {
                    it.fuel = FUEL;
                    if let Err(f) = it.const_value(n) {
                        verdict = Err(format!("W-CT-1 const `{n}`: {}", crate::interp::why(f)));
                        break;
                    }
                }
            }
            if verdict.is_ok() && self.state_deps.contains(&u) {
                it.fuel = FUEL;
                if let Err(f) = it.initial_state() {
                    verdict = Err(format!("W-CT-1 state initializer: {}", crate::interp::why(f)));
                }
            }
        }
        self.flip(k);
        self.form.insert(k, verdict.clone());
        verdict
    }

    /// The program's constants whose value does not depend on definition `u`.
    fn seed(&self, u: usize) -> HashMap<String, V> {
        let mut c = self.consts0.clone();
        for (n, deps) in &self.const_deps {
            if deps.contains(&u) {
                c.remove(n);
            }
        }
        c
    }

    /// Run the in-Form mutant `k` on the obligations `which`.
    fn run(&mut self, k: usize, which: &[usize]) -> Vec<(usize, Res)> {
        self.flip(k);
        let b = &self.base;
        let budget = |o: usize| FUEL.min(b[o].saturating_mul(2).saturating_add(MUTANT_FUEL_SLACK));
        let states = |o: usize| crate::law::LAW_FUEL.min((b[o] as usize).saturating_mul(4).max(10_000));
        let rs = evaluate(self.work.as_ref().expect("working copy"), &self.obs, which, &budget, &states, self.seed(self.mutants[k].unit));
        self.flip(k);
        which.iter().copied().zip(rs).collect()
    }

    /// The obligations whose reach holds mutant `k`'s definition.
    fn reached_by(&self, k: usize) -> Vec<usize> {
        let u = self.mutants[k].unit;
        (0..self.obs.len()).filter(|&o| self.reach[o].contains(&u)).collect()
    }

    /// The mutants (indices) at the definitions `set`, and whether all of
    /// that set's mutants are within the bound.
    fn of(&self, set: &BTreeSet<usize>) -> (Vec<usize>, bool) {
        let ks: Vec<usize> = (0..self.mutants.len()).filter(|&k| set.contains(&self.mutants[k].unit)).collect();
        let cut = self.mutants.len() < self.total;
        // the bound cut the list inside or before the definition of the
        // last mutant judged: that definition and later ones are incomplete
        let complete = !cut || self.mutants.last().is_none_or(|m| set.iter().all(|&u| u < m.unit));
        (ks, complete)
    }
}

/// Run obligations `which` on a linked module whose constants and initial
/// state evaluate (`Prep::in_form`), with the constant values `seed`: per
/// obligation its verdict.
fn evaluate(lm: &Module, obs: &[Obligation], which: &[usize], budget: &dyn Fn(usize) -> u64, states: &dyn Fn(usize) -> usize, seed: HashMap<String, V>) -> Vec<Res> {
    let mut it = Interp::new(lm);
    it.consts = seed;
    it.fuel = FUEL;
    let s0: Option<Vec<Region>> = if which.iter().any(|&o| matches!(obs[o].what, Ob::Ex(_))) { it.initial_state().ok() } else { None };
    let laws: Vec<usize> = which.iter().copied().filter(|&o| matches!(obs[o].what, Ob::Law(_))).collect();
    let decided = if laws.is_empty() { vec![] } else { crate::law::decide_with(lm, laws.iter().map(|&o| states(o)).max().unwrap_or(0)) };
    let mut out = vec![];
    for &o in which {
        let b = budget(o);
        it.fuel = b;
        let fault = match &obs[o].what {
            Ob::Ex(x) => crate::interp::example_fault(&mut it, s0.as_deref().unwrap_or(&[]), x),
            Ob::Ck(e) => crate::interp::check_fault(&mut it, e),
            Ob::Law(n) => {
                out.push(match decided.iter().find(|d| &d.name == n).map(|d| &d.verdict) {
                    Some(crate::law::Verdict::Holds { .. }) => Res::Holds,
                    Some(crate::law::Verdict::Fails { .. }) => Res::Fails,
                    _ => Res::Undecided,
                });
                continue;
            }
        };
        out.push(match fault {
            None => Res::Holds,
            Some(f) if f.contains(TRAP_FUEL) && b < FUEL => Res::Undecided,
            Some(_) => Res::Fails,
        });
    }
    out
}

/// W-MUT-1 for obligation `o`, or `None`. `in_form` counts the in-Form
/// mutants of its core it was judged on.
fn vacuity(o: &Obligation, reads: bool, in_form: usize, complete: bool, killed: bool, undecided: bool) -> Option<String> {
    if in_form > 0 && reads && complete && !killed && !undecided {
        return Some(format!(
            "line {}: W-MUT-1 `{}` holds on each of the {in_form} mutant{} {} and on every other mutant (6.12), so no single edit of the program makes it fail; strengthen it so that the program could fail it, or remove it",
            o.line,
            clip(&o.text, 80),
            if in_form == 1 { "" } else { "s" },
            if o.kind == "law" { "of the steps it quantifies over" } else { "at the expressions its run evaluates" }
        ));
    }
    None
}

/// W-MUT-1 for an admitted program (7, 6.12). `sites_of` are the
/// `site_addresses` of `p.linked` and `measure` what admission measured on
/// it. The verdict is that of `rce mutants`; the coverage only orders the
/// search: for each measured obligation, the mutants at expressions its run
/// evaluated are judged first, each on every obligation not yet killed that
/// reaches it, and the search stops at the obligation's first kill. An
/// obligation no mutant has killed is judged on every mutant of its reach
/// before it is reported.
pub fn gate(p: &Program, sites_of: &Sites, measure: &Measure) -> Vec<String> {
    let addrs = &sites_of.addrs;
    let lm = p.linked;
    if !lm.items.iter().take(p.own).any(|i| matches!(i, Item::Example(_) | Item::Check(_) | Item::Intent(..) | Item::Law(_))) {
        return vec![];
    }
    let mut q = Prep::new(p, MUTANT_BOUND, measure, sites_of);
    let n = q.obs.len();
    let mut killed = vec![false; n];
    let mut undec = vec![false; n];
    let mut core_inform: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    // (mutant, obligation) pairs run
    let mut ran: HashSet<(usize, usize)> = HashSet::new();
    let mut errs = vec![];
    let addr = |q: &Prep, k: usize| addrs.get(q.mutants[k].unit).and_then(|v| v.get(q.mutants[k].site)).copied();
    for o in 0..n {
        if !q.reads[o] || killed[o] || !q.light(o) {
            continue;
        }
        let (all, complete) = q.of(&q.reach[o]);
        if !complete || q.measured[o].is_empty() {
            continue;
        }
        let cov = measure.cover.get(&q.obs[o].line);
        let hit = |q: &Prep, k: usize| cov.is_some_and(|c| addr(q, k).is_some_and(|a| c.contains(&a)));
        // the mutants that measure it, then the others its run evaluated,
        // then the rest of its reach
        let mut order: Vec<usize> = q.measured[o].iter().copied().collect();
        order.extend(all.iter().copied().filter(|&k| hit(&q, k) && !q.measured[o].contains(&k)));
        order.extend(all.iter().copied().filter(|&k| !hit(&q, k) && !q.measured[o].contains(&k)));
        for k in order {
            if killed[o] {
                break;
            }
            if ran.contains(&(k, o)) {
                continue;
            }
            if q.in_form(k).is_err() {
                continue;
            }
            // `o` first; on a kill, the other obligations not yet killed
            // whose run evaluated the mutated expression
            let mut which = vec![o];
            loop {
                let rs = q.run(k, &which);
                let mut fails = false;
                for (r, v) in rs {
                    ran.insert((k, r));
                    if q.measured[r].contains(&k) {
                        core_inform[r].insert(k);
                    }
                    match v {
                        Res::Fails => {
                            killed[r] = true;
                            fails |= r == o;
                        }
                        Res::Undecided => undec[r] = true,
                        Res::Holds => {}
                    }
                }
                if !fails || which.len() > 1 || which[0] != o {
                    break;
                }
                let a = addr(&q, k);
                which = q
                    .reached_by(k)
                    .into_iter()
                    .filter(|&r| r != o && !killed[r] && !ran.contains(&(k, r)) && a.is_some_and(|a| measure.cover.get(&q.obs[r].line).is_some_and(|c| c.contains(&a))))
                    .collect();
                if which.is_empty() {
                    break;
                }
            }
        }
        if !killed[o] {
            // every mutant of the reach was judged while `o` was not
            // killed, so `o` was run on each of them
            errs.extend(vacuity(&q.obs[o], true, core_inform[o].len(), complete, false, undec[o]));
        }
    }
    errs
}

/// The result of `rce mutants FILE`.
pub struct Report {
    pub file: String,
    pub obligations: Vec<Obligation>,
    /// per obligation: the mutants it kills (indices into `mutants`)
    pub kills: Vec<Vec<usize>>,
    /// per obligation: the in-Form mutants of its reach it was run on, and
    /// whether its reach is within the bound
    pub reach: Vec<(usize, bool)>,
    pub reads: Vec<bool>,
    pub mutants: Vec<Mutant>,
    pub fates: Vec<Fate>,
    pub total: usize,
    pub bound: usize,
    pub definitions: usize,
    /// the canonical diff of each surviving mutant
    pub diffs: Vec<Vec<String>>,
    pub vacuous: Vec<String>,
    /// per obligation: its cost on the program (fuel, or a law's steps) and
    /// whether W-MUT-1 measures it
    pub cost: Vec<u64>,
    pub light: Vec<bool>,
}

impl Report {
    pub fn count(&self, f: fn(&Fate) -> bool) -> usize {
        self.fates.iter().filter(|x| f(x)).count()
    }
    pub fn killed(&self) -> usize {
        self.count(|f| matches!(f, Fate::Killed(_)))
    }
    pub fn survivors(&self) -> usize {
        self.count(|f| matches!(f, Fate::Survives { .. }))
    }
    pub fn non_equivalent(&self) -> usize {
        self.killed() + self.survivors()
    }
}

/// `rce mutants FILE [MODULE...]`: judge every mutant within `bound` on
/// every obligation that reaches it. `Err` is the report of a program that
/// is not admitted by the rules before W-MUT-1.
pub fn mutants(file: &str, src: &str, others: &[(String, String)], bound: usize) -> Result<Report, String> {
    let mut files = vec![(file.to_string(), src.to_string())];
    files.extend(others.iter().cloned());
    crate::check_program_with(&files, false, false).remove(0).1?;
    let m = crate::parse::parse(src)?;
    let parsed: Vec<Module> = others.iter().filter_map(|(_, s)| crate::parse::parse(s).ok()).collect();
    let mut mods: Vec<&Module> = vec![&m];
    mods.extend(parsed.iter());
    let known: HashSet<String> = mods.iter().filter_map(|x| x.path.as_ref().map(|q| q.join("."))).collect();
    let mut lk = crate::link::link(&m, &mods);
    let addrs = site_addresses(&mut lk.module, m.items.len());
    let measure = crate::interp::admit_measured(&lk.module).1;
    let p = Program { linked: &lk.module, own: m.items.len(), imported_fns: &lk.imported_fns, imported_types: &lk.imported_types, known: &known };
    let mut q = Prep::new(&p, bound, &measure, &addrs);
    let n = q.obs.len();
    let mut kills = vec![vec![]; n];
    let mut undec = vec![false; n];
    let mut inform = vec![0usize; n];
    let mut core_inform = vec![0usize; n];
    let mut fates = vec![];
    let us = q.units.clone();
    let has_steps = lk.module.items.iter().any(|i| matches!(i, Item::Step { .. } | Item::Actor { .. }));
    let behaviour = closure(&lk.module, &us, |u| matches!(u, Unit::Step(_) | Unit::Actor(_) | Unit::State(_)));
    let main = closure(&lk.module, &us, |u| matches!(u, Unit::Fn(_)));
    let main: BTreeSet<usize> = main.into_iter().filter(|&i| reaches_main(&lk.module, &us, i)).collect();
    for k in 0..q.mutants.len() {
        let which = q.reached_by(k);
        let fate = match q.in_form(k) {
            Err(e) => Fate::NotInForm(e),
            Ok(()) => {
                let rs = q.run(k, &which);
                let (mut kl, mut und) = (vec![], vec![]);
                for (o, r) in rs {
                    inform[o] += 1;
                    if q.measured[o].contains(&k) {
                        core_inform[o] += 1;
                    }
                    match r {
                        Res::Fails => {
                            kills[o].push(k);
                            kl.push(o);
                        }
                        Res::Undecided => {
                            undec[o] = true;
                            und.push(o);
                        }
                        Res::Holds => {}
                    }
                }
                let u = q.mutants[k].unit;
                if !kl.is_empty() {
                    Fate::Killed(kl)
                } else if !und.is_empty() {
                    Fate::Undecided(und)
                } else {
                    survivor(&q, k, has_steps && behaviour.contains(&u) && !main.contains(&u), has_steps)
                }
            }
        };
        fates.push(fate);
    }
    // the diff of a survivor, on the canonical text of FILE as written
    let base_text = unparse(&m);
    let diffs: Vec<Vec<String>> = (0..q.mutants.len()).map(|k| if matches!(fates[k], Fate::Survives { .. }) { source_diff(&m, &q, k, &base_text) } else { vec![] }).collect();
    let mut reach = vec![];
    let mut vacuous = vec![];
    for o in 0..n {
        let (_, complete) = q.of(&q.reach[o]);
        reach.push((inform[o], complete));
        let measured = q.light(o);
        if let Some(v) = vacuity(&q.obs[o], q.reads[o], if measured { core_inform[o] } else { 0 }, complete, !kills[o].is_empty(), undec[o]) {
            vacuous.push(v);
        }
    }
    let light = (0..n).map(|o| q.light(o)).collect();
    Ok(Report {
        file: file.to_string(),
        obligations: q.obs.clone(),
        kills,
        reach,
        reads: q.reads.clone(),
        mutants: q.mutants.clone(),
        fates,
        total: q.total,
        bound,
        definitions: q.units.len(),
        diffs,
        vacuous,
        cost: q.cost.clone(),
        light,
    })
}

/// The diff of mutant `k` on the canonical text of the source: the same
/// edit (the same index among the edits of the same definition) applied to
/// the unlinked module, whose own items are those of the linked one. Linking
/// only renames, so a definition has the same list of edits in both, unless
/// it names an import through its module; then the diff is of the linked
/// text.
fn source_diff(m: &Module, q: &Prep, k: usize, base_text: &str) -> Vec<String> {
    let mu = &q.mutants[k];
    let u = &q.units[mu.unit];
    let index = q.mutants.iter().filter(|x| x.unit == mu.unit).position(|x| x.site == mu.site && x.repl == mu.repl);
    let (one, v) = slim(m, u);
    let (lone, lv) = slim(q.p.linked, u);
    let es = sites(&one, &v, &small_edits);
    if let Some(ix) = index
        && es.len() == sites(&lone, &lv, &small_edits).len()
        && let Some((site, _, r)) = es.get(ix)
    {
        return diff(base_text, &unparse(&apply(m, u, *site, r)));
    }
    diff(&unparse(q.p.linked), &unparse(&q.mutant(k)))
}

/// The own definitions reachable by name from those `start` picks.
fn closure(lm: &Module, us: &[Unit], start: fn(&Unit) -> bool) -> BTreeSet<usize> {
    let mut by_name: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, u) in us.iter().enumerate() {
        if let Some(n) = callable_name(lm, u) {
            by_name.entry(n).or_default().push(i);
        }
    }
    let mut out: BTreeSet<usize> = us.iter().enumerate().filter(|(_, u)| start(u)).map(|(i, _)| i).collect();
    let mut todo: Vec<usize> = out.iter().copied().collect();
    while let Some(u) = todo.pop() {
        let (one, v) = slim(lm, &us[u]);
        for n in mentions(&v, &one) {
            for &v in by_name.get(&n).into_iter().flatten() {
                if out.insert(v) {
                    todo.push(v);
                }
            }
        }
    }
    out
}

/// Whether definition `i` is `fn main` or reachable from it.
fn reaches_main(lm: &Module, us: &[Unit], i: usize) -> bool {
    let Some(m) = us.iter().position(|u| matches!(u, Unit::Fn(_)) && callable_name(lm, u).as_deref() == Some("main")) else { return false };
    let mut seen = BTreeSet::from([m]);
    let mut todo = vec![m];
    while let Some(u) = todo.pop() {
        if u == i {
            return true;
        }
        let (one, v) = slim(lm, &us[u]);
        for n in mentions(&v, &one) {
            for (j, w) in us.iter().enumerate() {
                if callable_name(lm, w).as_deref() == Some(n.as_str()) && seen.insert(j) {
                    todo.push(j);
                }
            }
        }
    }
    false
}

/// A mutant no obligation kills: equivalent, or a survivor with the witness
/// `rce equiv` gives.
fn survivor(q: &Prep, k: usize, comparable: bool, has_steps: bool) -> Fate {
    if !comparable {
        let why = if has_steps {
            "no step reaches the definition, or fn main does: rce equiv compares the steps' behaviour (6.10)"
        } else {
            "the program has no step: rce equiv compares the steps' behaviour (6.10)"
        };
        return Fate::Survives { witness: None, why: Some(why.into()) };
    }
    let ms = q.mutant(k);
    match crate::dim::equiv(q.p.linked, &ms, EQUIV_FUEL) {
        Ok(e) if e.equivalent && e.exact => Fate::Equivalent,
        Ok(e) if e.witness.is_some() => Fate::Survives { witness: e.witness, why: None },
        Ok(e) if e.equivalent => Fate::Survives { witness: None, why: Some(format!("rce equiv: equivalent only to the explored bound ({EQUIV_FUEL} states)")) },
        Ok(e) if e.undecided => Fate::Survives { witness: None, why: Some(format!("rce equiv: undecided within {EQUIV_FUEL} states")) },
        Ok(e) if !e.only_a.is_empty() || !e.only_b.is_empty() => Fate::Survives { witness: None, why: Some("rce equiv: the mutant explores other event instances".into()) },
        Ok(_) => Fate::Survives { witness: None, why: Some("rce equiv: different, with no finite witness in the explored graph".into()) },
        Err(e) => Fate::Survives { witness: None, why: Some(format!("rce equiv: not compared ({e})")) },
    }
}

/// The report `rce mutants` prints.
pub fn report(r: &Report) -> String {
    let mut s = String::new();
    let nif = r.count(|f| matches!(f, Fate::NotInForm(_)));
    let eq = r.count(|f| matches!(f, Fate::Equivalent));
    let und = r.count(|f| matches!(f, Fate::Undecided(_)));
    s.push_str(&format!(
        "mutants     {} of {} single edits judged (bound {}), {} definition{}\n",
        r.mutants.len(),
        r.total,
        r.bound,
        r.definitions,
        if r.definitions == 1 { "" } else { "s" }
    ));
    s.push_str(&format!("            {nif} not in Form, {eq} equivalent (rce equiv), {und} undecided (fuel)\n"));
    s.push_str(&format!("score       {} killed of {} non-equivalent{}\n", r.killed(), r.non_equivalent(), percent(r.killed(), r.non_equivalent())));
    for (o, ob) in r.obligations.iter().enumerate() {
        let (reach, complete) = r.reach[o];
        let note = match (r.reads[o], complete, r.light[o]) {
            (false, _, _) => "  (reads no definition of the program)",
            (_, false, _) => "  (reach beyond the bound)",
            (_, true, false) => "  (heavier than W-MUT-1 measures)",
            _ => "",
        };
        s.push_str(&format!("obligation  line {:<5} kills {:>4} of {:>4}  {}{note}\n", ob.line, r.kills[o].len(), reach, clip(&ob.text, 60)));
    }
    for v in &r.vacuous {
        s.push_str(&format!("vacuous     {v}\n"));
    }
    for (k, f) in r.fates.iter().enumerate() {
        if let Fate::Survives { witness, why } = f {
            let mu = &r.mutants[k];
            s.push_str(&format!("survivor    W-MUT-2 in `{}`, {}\n", mu.definition, mu.edit));
            for l in &r.diffs[k] {
                s.push_str(&format!("            {l}\n"));
            }
            match (witness, why) {
                (Some((seq, a, b)), _) => s.push_str(&format!("            after {}: the program {}, the mutant {}\n", seq.join(", "), output(a), output(b))),
                (None, Some(w)) => s.push_str(&format!("            not compared: {w}\n")),
                _ => {}
            }
        }
    }
    let verdict = if r.vacuous.is_empty() {
        format!("ok {}: no vacuous obligation; {} surviving mutant{}", r.file, r.survivors(), if r.survivors() == 1 { "" } else { "s" })
    } else {
        format!("requires change {}: {} vacuous obligation{} (W-MUT-1)", r.file, r.vacuous.len(), if r.vacuous.len() == 1 { "" } else { "s" })
    };
    s.push_str(&format!("verdict     {verdict}\n"));
    s
}

fn output(l: &str) -> String {
    if l == "ok:" {
        "is silent".into()
    } else if let Some(x) = l.strip_prefix("ok:") {
        format!("emits {x}")
    } else if let Some(x) = l.strip_prefix("kept:") {
        format!("is kept: {x}")
    } else {
        l.to_string()
    }
}

fn clip(t: &str, n: usize) -> String {
    if t.chars().count() <= n { t.to_string() } else { format!("{}...", t.chars().take(n).collect::<String>()) }
}

/// ` (82.6%)`, by integer arithmetic, rounded down.
fn percent(a: usize, b: usize) -> String {
    if b == 0 {
        return String::new();
    }
    let p = a * 1000 / b;
    format!(" ({}.{}%)", p / 10, p % 10)
}

/// The JSON report (format `rce-mutants/1`).
pub fn report_json(r: &Report) -> String {
    let fate = |f: &Fate| match f {
        Fate::NotInForm(e) => json!({"fate": "not in Form", "report": e}),
        Fate::Killed(os) => json!({"fate": "killed", "by": os.iter().map(|&o| r.obligations[o].line).collect::<Vec<_>>()}),
        Fate::Equivalent => json!({"fate": "equivalent"}),
        Fate::Undecided(os) => json!({"fate": "undecided", "undecided_for": os.iter().map(|&o| r.obligations[o].line).collect::<Vec<_>>()}),
        Fate::Survives { witness, why } => json!({
            "fate": "survives",
            "witness": witness.as_ref().map(|(seq, a, b)| json!({"after": seq, "program": a, "mutant": b})),
            "not_compared": why,
        }),
    };
    let v = json!({
        "format": "rce-mutants/1",
        "file": r.file,
        "bound": r.bound,
        "single_edits": r.total,
        "judged": r.mutants.len(),
        "definitions": r.definitions,
        "killed": r.killed(),
        "non_equivalent": r.non_equivalent(),
        "not_in_form": r.count(|f| matches!(f, Fate::NotInForm(_))),
        "equivalent": r.count(|f| matches!(f, Fate::Equivalent)),
        "undecided": r.count(|f| matches!(f, Fate::Undecided(_))),
        "survivors": r.survivors(),
        "obligations": r.obligations.iter().enumerate().map(|(o, ob)| json!({
            "line": ob.line,
            "kind": ob.kind,
            "text": ob.text,
            "kills": r.kills[o].len(),
            "reach": r.reach[o].0,
            "reach_within_bound": r.reach[o].1,
            "reads_a_definition": r.reads[o],
            "cost": r.cost[o],
            "measured": r.light[o],
        })).collect::<Vec<_>>(),
        "vacuous": r.vacuous,
        "mutants": r.mutants.iter().zip(&r.fates).enumerate().map(|(k, (m, f))| {
            let mut x = fate(f);
            x["definition"] = json!(m.definition);
            x["edit"] = json!(m.edit);
            x["line"] = json!(m.line);
            if !r.diffs[k].is_empty() {
                x["diff"] = json!(r.diffs[k]);
            }
            x
        }).collect::<Vec<_>>(),
    });
    crate::interp::canon_json(&v)
}
