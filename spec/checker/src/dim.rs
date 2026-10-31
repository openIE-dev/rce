//! `rce dim` (spec 6.10, 10.11): the number of behaviours a program can be
//! in and the redundancy of its state.
//!
//! Exploration: breadth-first from s₀ over every declared event, with
//! payloads drawn per field from the boundary set of 6.3 of its base type
//! and the field's harvested values (`harvest`, 10.11): the literals the
//! steps compare the field with, including those of a refined type's
//! predicate. Partition refinement: Moore's iterative algorithm on the
//! explored graph with the per-step output of λ as the initial partition.
//! It is quadratic in the worst case where Hopcroft is n log n; it is kept
//! because its order of operations is obviously deterministic and the graphs
//! `fuel` admits are small. States not explored (bound reached) share one
//! class, so the count is then a lower bound on dim(P), never an upper one.
//!
//! Status: written against the interpreter API (`Interp::new`,
//! `initial_state`, `construct`, `delta`, `state_json`); not yet built or
//! run in the session that wrote it; later built and run (2026-10-23).
//!
//! `control` computes the control quotient of 6.10 on the explored
//! graph: the same states and transitions, with each step labelled only by
//! its control outcome (which regions changed, whether it kept the state
//! and why, the constructors of what it emitted) instead of the values.
//! That is a coarser output function on the same deterministic machine, so
//! its minimal size dim_c is at most dim, exactly as 6.10 states, and the
//! predicate-equivalence oracle is concrete evaluation on reach(P): exact
//! on the explored set, a lower bound when the fuel bound is reached. The
//! canonical control string numbers the quotient's states by first visit
//! in breadth-first order from s0 with events in canonical order
//! (Reis, Moreira and Almeida 2005) and `shape_ctl` is its SHA-256.

use crate::ast::{Arg, Body, Expr, FnDecl, Item, Module, Param, Pat, SelArm, Stmt, Ty};
use crate::interp::{state_json, why, Interp, Region, V};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

/// One concrete event instance: its name and the literal arguments used to
/// construct it, kept so the report can name a distinguishing event.
#[derive(Clone)]
struct EvInst {
    name: String,
    args: Vec<Arg>,
    show: String,
}

/// The result of `rce dim` on one program.
pub struct Dim {
    pub states: usize,
    pub bound_reached: bool,
    pub behaviours: usize,
    pub redundancy_bits: f64,
    pub regions: Vec<(String, usize, usize, f64)>, // name, distinct values, needed values, bits
    pub events_skipped: Vec<String>,
    /// `control`: dim_c and the hash of the canonical control string
    pub control: Option<(usize, String)>,
    /// Some explored step has a λ output (6.1): an emitted value, or the
    /// record of a kept step. A program with no such step has one behaviour by
    /// definition (6.10): its state is in D and readable, but no future
    /// output depends on it.
    pub has_output: bool,
    /// Some explored step emits a value.
    pub emits: bool,
    /// Some explored step keeps the state (its event does not apply).
    pub keeps: bool,
    /// Distinct output sequences over the explored steps (the text of a
    /// step's emitted values, or why it kept the state; a silent step is excluded): 1
    /// means every output is the same, so the program has one behaviour as
    /// well.
    pub distinct_outputs: usize,
}

fn lit_int(i: i128) -> Expr {
    Expr::Int(i, 0)
}

/// The boundary set of 6.3 for a payload field type: the values an
/// exploration tries. A type this table does not know yields `None` and the
/// event is skipped and reported.
fn boundary(t: &Ty, aliases: &HashMap<String, Ty>) -> Option<Vec<Expr>> {
    match t {
        Ty::Path(p, args) if args.is_empty() && p.len() == 1 && aliases.contains_key(&p[0]) => boundary(&aliases[&p[0]], aliases),
        Ty::Path(p, args) if args.is_empty() && p.len() == 1 => match p[0].as_str() {
            "bool" => Some(vec![Expr::Bool(false, 0), Expr::Bool(true, 0)]),
            "text" => Some(vec![Expr::Str(String::new(), 0), Expr::Str("a".into(), 0)]),
            "i8" => Some(vec![lit_int(-128), lit_int(-1), lit_int(0), lit_int(1), lit_int(127)]),
            "i16" => Some(vec![lit_int(-32768), lit_int(-1), lit_int(0), lit_int(1), lit_int(32767)]),
            "i32" => Some(vec![lit_int(-(1 << 31)), lit_int(-1), lit_int(0), lit_int(1), lit_int((1 << 31) - 1)]),
            "i64" => Some(vec![lit_int(i64::MIN as i128), lit_int(-1), lit_int(0), lit_int(1), lit_int(i64::MAX as i128)]),
            "u8" => Some(vec![lit_int(0), lit_int(1), lit_int(255)]),
            "u16" => Some(vec![lit_int(0), lit_int(1), lit_int(65535)]),
            "u32" => Some(vec![lit_int(0), lit_int(1), lit_int((1u128 << 32) as i128 - 1)]),
            "u64" | "usize" => Some(vec![lit_int(0), lit_int(1), lit_int(u64::MAX as i128)]),
            "f64" | "f32" => Some(vec![Expr::Float(-1.0, 0), Expr::Float(0.0, 0), Expr::Float(1.0, 0)]),
            _ => None,
        },
        Ty::Unit => Some(vec![Expr::Unit(0)]),
        Ty::Opt(inner) => {
            let mut v = vec![Expr::None(0)];
            for e in boundary(inner, aliases)? {
                v.push(Expr::Call(Box::new(Expr::Name("some".into(), 0)), vec![Arg { label: None, e }], 0, false));
            }
            Some(v)
        }
        Ty::Path(p, args) if p.len() == 1 && p[0] == "List" && args.len() == 1 => {
            let mut v = vec![Expr::List(vec![], 0)];
            if let Some(first) = boundary(&args[0], aliases).and_then(|b| b.into_iter().nth(1)) {
                v.push(Expr::List(vec![first], 0));
            }
            Some(v)
        }
        _ => None,
    }
}

/// The most harvested values one field adds to its boundary set (10.11):
/// the smallest ones in value order are kept.
pub const HARVEST_MAX: usize = 64;

/// A literal a step compares an event field against (10.11), as a value.
/// Ordered by kind, then by value (integers numerically, floats by their
/// total order, texts by their bytes), so a harvested set has one order.
#[derive(Clone, Debug)]
pub enum Lit {
    Int(i128),
    Float(f64),
    Text(String),
}

impl Lit {
    fn rank(&self) -> u8 {
        match self {
            Lit::Int(_) => 0,
            Lit::Float(_) => 1,
            Lit::Text(_) => 2,
        }
    }
}

impl PartialEq for Lit {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o).is_eq()
    }
}
impl Eq for Lit {}
impl PartialOrd for Lit {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Lit {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        match (self, o) {
            (Lit::Int(a), Lit::Int(b)) => a.cmp(b),
            (Lit::Float(a), Lit::Float(b)) => a.total_cmp(b),
            (Lit::Text(a), Lit::Text(b)) => a.as_bytes().cmp(b.as_bytes()),
            _ => self.rank().cmp(&o.rank()),
        }
    }
}

/// The literals harvested per event (declaration index) and payload field
/// (position): the alphabet's additions to the boundary sets (10.11).
pub type Harvest = Vec<Vec<BTreeSet<Lit>>>;

/// Where a name's value comes from: the field `(event, position)` it is
/// bound to, directly or through `copy`, `&`, a cast or a branch of an `if`.
type Field = (usize, usize);

struct Harvester<'a> {
    fns: HashMap<&'a str, &'a FnDecl>,
    consts: HashMap<&'a str, Lit>,
    out: Harvest,
    /// (function, parameter position, field): each analysed once, which
    /// bounds the walk through calls, recursive ones included
    seen: BTreeSet<(String, usize, Field)>,
}

fn lit_of(e: &Expr, consts: &HashMap<&str, Lit>) -> Option<Lit> {
    match e {
        Expr::Int(i, _) => Some(Lit::Int(*i)),
        Expr::Float(f, _) => Some(Lit::Float(*f)),
        Expr::Str(s, _) => Some(Lit::Text(s.clone())),
        Expr::Neg(x, _) => match lit_of(x, consts)? {
            Lit::Int(i) => Some(Lit::Int(-i)),
            Lit::Float(f) => Some(Lit::Float(-f)),
            Lit::Text(_) => None,
        },
        Expr::Name(n, _) => consts.get(n.as_str()).cloned(),
        _ => None,
    }
}

impl<'a> Harvester<'a> {
    fn new(m: &'a Module) -> Self {
        let mut fns = HashMap::new();
        let mut consts = HashMap::new();
        for it in &m.items {
            match it {
                Item::Fn(f) => {
                    fns.insert(f.sig.name.as_str(), f);
                }
                Item::Const { name, e, .. } => {
                    if let Some(l) = lit_of(e, &HashMap::new()) {
                        consts.insert(name.as_str(), l);
                    }
                }
                _ => {}
            }
        }
        Harvester { fns, consts, out: vec![], seen: BTreeSet::new() }
    }

    fn record(&mut self, f: Field, l: Lit) {
        self.out[f.0][f.1].insert(l);
    }

    /// The field an expression's value is, when it is a bound name or one
    /// of the direct forms around it.
    fn tie(e: &Expr, ties: &HashMap<String, Field>) -> Option<Field> {
        match e {
            Expr::Name(n, _) => ties.get(n).copied(),
            Expr::SelfRef(_) => ties.get("self").copied(),
            Expr::Borrow(_, x, _) | Expr::Cast(x, _, _) => Self::tie(x, ties),
            Expr::Call(c, args, _, _) if args.len() == 1 && matches!(&**c, Expr::Name(n, _) if n == "copy") => Self::tie(&args[0].e, ties),
            Expr::If(_, a, b, _) => Self::tie(a, ties).or_else(|| Self::tie(b, ties)),
            _ => None,
        }
    }

    fn pat_lits(&mut self, p: &Pat, f: Field) {
        match p {
            Pat::Lit(e) => {
                if let Some(l) = lit_of(e, &self.consts) {
                    self.record(f, l);
                }
            }
            Pat::Range(a, _, b) => {
                for e in [a, b] {
                    if let Some(l) = lit_of(e, &self.consts) {
                        self.record(f, l);
                    }
                }
            }
            Pat::Alt(ps) => ps.iter().for_each(|q| self.pat_lits(q, f)),
            _ => {}
        }
    }

    /// A pattern matched against a value of field `f` binds its name to `f`.
    fn pat_bind(p: &Pat, f: Option<Field>, ties: &mut HashMap<String, Field>) {
        if let Pat::Path(v) = p
            && v.len() == 1
        {
            match f {
                Some(f) => ties.insert(v[0].clone(), f),
                None => ties.remove(&v[0]),
            };
        }
    }

    fn expr(&mut self, e: &Expr, ties: &HashMap<String, Field>) {
        match e {
            Expr::Bin(op, a, b, _) => {
                if matches!(op.as_str(), "==" | "!=" | "<" | "<=" | ">" | ">=") {
                    for (x, y) in [(a, b), (b, a)] {
                        if let (Some(f), Some(l)) = (Self::tie(x, ties), lit_of(y, &self.consts)) {
                            self.record(f, l);
                        }
                    }
                }
                self.expr(a, ties);
                self.expr(b, ties);
            }
            Expr::Call(c, args, _, _) => {
                if let Expr::Name(n, _) = &**c
                    && let Some(&fd) = self.fns.get(n.as_str())
                {
                    let names: Vec<(Option<&String>, &String)> = fd
                        .sig
                        .params
                        .iter()
                        .filter_map(|p| match p {
                            Param::Named { label, name, .. } => Some((label.as_ref(), name)),
                            _ => None,
                        })
                        .collect();
                    for (i, a) in args.iter().enumerate() {
                        let Some(f) = Self::tie(&a.e, ties) else { continue };
                        let pos = match &a.label {
                            Some(l) => names.iter().position(|(lb, nm)| lb.unwrap_or(nm) == l),
                            None => Some(i),
                        };
                        if let Some(pos) = pos.filter(|&p| p < names.len())
                            && self.seen.insert((n.clone(), pos, f))
                        {
                            let mut inner = HashMap::new();
                            inner.insert(names[pos].1.clone(), f);
                            match &fd.body {
                                Body::Arrow(x) => self.expr(x, &inner),
                                Body::Block(ss) => self.stmts(ss, &mut inner),
                            }
                        }
                    }
                }
                self.expr(c, ties);
                args.iter().for_each(|a| self.expr(&a.e, ties));
            }
            Expr::Tuple(xs, _) | Expr::List(xs, _) => xs.iter().for_each(|x| self.expr(x, ties)),
            Expr::Index(x, xs, _) => {
                self.expr(x, ties);
                xs.iter().for_each(|y| self.expr(y, ties));
            }
            Expr::Map(kv, _) => kv.iter().for_each(|(k, v)| {
                self.expr(k, ties);
                self.expr(v, ties);
            }),
            Expr::ListRepeat(a, b, _) | Expr::Catch(a, _, b, _) => {
                self.expr(a, ties);
                self.expr(b, ties);
            }
            Expr::ListComp(a, _, b, c, _) => {
                self.expr(a, ties);
                self.expr(b, ties);
                if let Some(c) = c {
                    self.expr(c, ties);
                }
            }
            Expr::Slice(a, b, c, _) => {
                self.expr(a, ties);
                for x in [b, c].into_iter().flatten() {
                    self.expr(x, ties);
                }
            }
            Expr::Try(a, b, _) => {
                self.expr(a, ties);
                if let Some(b) = b {
                    self.expr(b, ties);
                }
            }
            Expr::If(c, a, b, _) => {
                self.expr(c, ties);
                self.expr(a, ties);
                self.expr(b, ties);
            }
            Expr::Member(x, _, _)
            | Expr::TupleMember(x, _, _)
            | Expr::Neg(x, _)
            | Expr::BitNot(x, _)
            | Expr::Borrow(_, x, _)
            | Expr::Recv(x, _)
            | Expr::Comptime(x, _)
            | Expr::Cast(x, _, _)
            | Expr::Not(x, _)
            | Expr::Lambda(_, x, _) => self.expr(x, ties),
            _ => {}
        }
    }

    fn stmts(&mut self, ss: &[Stmt], ties: &mut HashMap<String, Field>) {
        for s in ss {
            self.stmt(s, ties);
        }
    }

    fn stmt(&mut self, s: &Stmt, ties: &mut HashMap<String, Field>) {
        match s {
            Stmt::Let(p, _, e, _) => {
                self.expr(e, ties);
                let f = Self::tie(e, ties);
                Self::pat_bind(p, f, ties);
            }
            Stmt::Var(n, _, e, _) => {
                self.expr(e, ties);
                match Self::tie(e, ties) {
                    Some(f) => ties.insert(n.clone(), f),
                    None => ties.remove(n),
                };
            }
            Stmt::Assign(_, place, e) => {
                self.expr(e, ties);
                self.expr(place, ties);
                if let Expr::Name(n, _) = place {
                    ties.remove(n);
                }
            }
            Stmt::Match(x, arms, _) => {
                self.expr(x, ties);
                // a tuple scrutinee is matched position by position
                let parts: Vec<Option<Field>> = match x {
                    Expr::Tuple(xs, _) => xs.iter().map(|y| Self::tie(y, ties)).collect(),
                    _ => vec![Self::tie(x, ties)],
                };
                for arm in arms {
                    let mut t = ties.clone();
                    match (&arm.pat, x) {
                        (Pat::Tuple(ps), Expr::Tuple(..)) if ps.len() == parts.len() => {
                            for (q, f) in ps.iter().zip(&parts) {
                                if let Some(f) = f {
                                    self.pat_lits(q, *f);
                                }
                                Self::pat_bind(q, *f, &mut t);
                            }
                        }
                        (p, _) if parts.len() == 1 => {
                            if let Some(f) = parts[0] {
                                self.pat_lits(p, f);
                            }
                            Self::pat_bind(p, parts[0], &mut t);
                        }
                        _ => {}
                    }
                    if let Some(g) = &arm.guard {
                        self.expr(g, &t);
                    }
                    self.stmts(&arm.body, &mut t);
                }
            }
            Stmt::If(bs, els) => {
                for (c, b) in bs {
                    self.expr(c, ties);
                    self.stmts(b, &mut ties.clone());
                }
                if let Some(b) = els {
                    self.stmts(b, &mut ties.clone());
                }
            }
            Stmt::While(c, inv, b) => {
                self.expr(c, ties);
                if let Some(i) = inv {
                    self.expr(i, ties);
                }
                self.stmts(b, &mut ties.clone());
            }
            Stmt::Loop(c, b) => {
                if let Some(c) = c {
                    self.expr(c, ties);
                }
                self.stmts(b, &mut ties.clone());
            }
            Stmt::For(p, e, b, _) => {
                self.expr(e, ties);
                let mut t = ties.clone();
                Self::pat_bind(p, None, &mut t);
                self.stmts(b, &mut t);
            }
            Stmt::Arena(_, e, b, _) | Stmt::Device(_, e, b, _) => {
                if let Some(e) = e {
                    self.expr(e, ties);
                }
                self.stmts(b, &mut ties.clone());
            }
            Stmt::Comptime(b) => self.stmts(b, &mut ties.clone()),
            Stmt::Select(arms, _) => {
                for a in arms {
                    match a {
                        SelArm::Recv(_, e, b) | SelArm::After(e, b) => {
                            self.expr(e, ties);
                            self.stmts(b, &mut ties.clone());
                        }
                        SelArm::Send(x, y, b) => {
                            self.expr(x, ties);
                            self.expr(y, ties);
                            self.stmts(b, &mut ties.clone());
                        }
                        SelArm::Default(b) => self.stmts(b, &mut ties.clone()),
                    }
                }
            }
            Stmt::Expr(e) | Stmt::Emit(e) | Stmt::Fail(e) | Stmt::Spawn(e) | Stmt::Check(e) | Stmt::Yield(e, _) => self.expr(e, ties),
            Stmt::Send(a, b) => {
                self.expr(a, ties);
                self.expr(b, ties);
            }
            Stmt::Return(Some(e), _) => self.expr(e, ties),
            _ => {}
        }
    }

    /// A step (or actor handler) on event `ev`: its pattern binds fields,
    /// its literal sub-patterns are harvested, then its guard and body.
    fn step(&mut self, ev: usize, fields: &[(String, Ty)], pat: &Pat, guard: Option<&Expr>, body: &Body) {
        let mut ties = HashMap::new();
        if let Pat::Ctor(_, args) = pat {
            for (i, (lb, q)) in args.iter().enumerate() {
                let pos = match lb {
                    Some(l) => fields.iter().position(|(n, _)| n == l),
                    None => Some(i),
                };
                let Some(pos) = pos.filter(|&p| p < fields.len()) else { continue };
                self.pat_lits(q, (ev, pos));
                Self::pat_bind(q, Some((ev, pos)), &mut ties);
            }
        }
        if let Some(g) = guard {
            self.expr(g, &ties);
        }
        match body {
            Body::Arrow(x) => self.expr(x, &ties),
            Body::Block(ss) => self.stmts(ss, &mut ties),
        }
    }
}

/// Literal harvesting (10.11): for each event field, the literals that the
/// program compares it against. A value is followed from the step pattern
/// that binds the field through `let`/`var` bindings of the forms `x`,
/// `copy(x)`, `&x`, `x as T` and `if c then x else y`, into match arms of
/// that value (or of a tuple position holding it) and into the parameters
/// of the module's functions it is passed to; the literals it meets in
/// `==`, `!=`, `<`, `<=`, `>`, `>=`, in literal and range patterns, and in
/// the predicate of a refined field type (with `self` the field) are
/// recorded. A constant whose value is a literal counts as that literal.
pub fn harvest(m: &Module) -> Harvest {
    let mut h = Harvester::new(m);
    let events: Vec<(&String, &Vec<(String, Ty)>)> = m
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Event { name, payload, .. } => Some((name, payload)),
            _ => None,
        })
        .collect();
    h.out = events.iter().map(|(_, p)| vec![BTreeSet::new(); p.len()]).collect();
    let ev_index = |n: &[String]| n.last().and_then(|n| events.iter().position(|(e, _)| *e == n));
    for it in &m.items {
        match it {
            Item::Step { pat, guard, body, .. } => {
                if let Pat::Ctor(n, _) = pat
                    && let Some(ev) = ev_index(n)
                {
                    h.step(ev, events[ev].1, pat, guard.as_ref(), body);
                }
            }
            Item::Actor { handlers, .. } => {
                for hd in handlers {
                    if let Pat::Ctor(n, _) = &hd.pat
                        && let Some(ev) = ev_index(n)
                    {
                        h.step(ev, events[ev].1, &hd.pat, hd.guard.as_ref(), &hd.body);
                    }
                }
            }
            _ => {}
        }
    }
    // the predicate of a refined field type, with `self` the field
    for (ev, (_, payload)) in events.iter().enumerate() {
        for (pos, (_, t)) in payload.iter().enumerate() {
            if let Ty::Path(p, args) = t
                && args.is_empty()
                && p.len() == 1
                && let Some(Item::TypeAlias { pred: Some(e), .. }) = m.items.iter().find(|i| matches!(i, Item::TypeAlias { name, .. } if *name == p[0]))
            {
                let mut ties = HashMap::new();
                ties.insert("self".to_string(), (ev, pos));
                h.expr(e, &ties);
            }
        }
    }
    h.out
}

/// The union of two harvests, event by declaration index and field by
/// position, so `rce equiv` explores both programs over one alphabet.
fn union(a: &Harvest, b: &Harvest) -> Harvest {
    let mut out = a.clone();
    for (ev, fs) in b.iter().enumerate() {
        if ev >= out.len() {
            out.push(fs.clone());
            continue;
        }
        for (pos, s) in fs.iter().enumerate() {
            if pos >= out[ev].len() {
                out[ev].push(s.clone());
            } else {
                out[ev][pos].extend(s.iter().cloned());
            }
        }
    }
    out
}

/// The integer range of a base type, for clamping harvested neighbours.
fn int_range(t: &str) -> Option<(i128, i128)> {
    Some(match t {
        "i8" => (i8::MIN as i128, i8::MAX as i128),
        "i16" => (i16::MIN as i128, i16::MAX as i128),
        "i32" => (i32::MIN as i128, i32::MAX as i128),
        "i64" => (i64::MIN as i128, i64::MAX as i128),
        "u8" => (0, u8::MAX as i128),
        "u16" => (0, u16::MAX as i128),
        "u32" => (0, u32::MAX as i128),
        "u64" | "usize" => (0, u64::MAX as i128),
        _ => return None,
    })
}

/// The values a field's harvested literals add to its boundary set: for an
/// integer type each literal and its two neighbours within the type's
/// range; for a float type each literal and the literal ± 1; for text each
/// literal. Values already in the boundary set are not repeated; at most
/// `HARVEST_MAX` are added, the smallest first. For text, when every text
/// of the boundary set is a harvested literal, the first of "a", "aa",
/// "aaa", … that is not one is added after them.
fn harvested_values(t: &Ty, aliases: &HashMap<String, Ty>, lits: &BTreeSet<Lit>, boundary: &[Expr]) -> Vec<Expr> {
    let base = match t {
        Ty::Path(p, args) if args.is_empty() && p.len() == 1 && aliases.contains_key(&p[0]) => {
            return harvested_values(&aliases[&p[0]], aliases, lits, boundary);
        }
        Ty::Path(p, args) if args.is_empty() && p.len() == 1 => p[0].as_str(),
        _ => return vec![],
    };
    let mut vals: BTreeSet<Lit> = BTreeSet::new();
    if let Some((lo, hi)) = int_range(base) {
        for l in lits {
            if let Lit::Int(i) = l {
                for v in [i.saturating_sub(1), *i, i.saturating_add(1)] {
                    if lo <= v && v <= hi {
                        vals.insert(Lit::Int(v));
                    }
                }
            }
        }
    } else if base == "f64" || base == "f32" {
        for l in lits {
            if let Lit::Float(f) = l {
                for v in [f - 1.0, *f, f + 1.0] {
                    if v.is_finite() {
                        vals.insert(Lit::Float(v));
                    }
                }
            }
        }
    } else if base == "text" {
        vals.extend(lits.iter().filter(|l| matches!(l, Lit::Text(_))).cloned());
    }
    let mut out: Vec<Expr> = vals
        .into_iter()
        .map(|l| match l {
            Lit::Int(i) => Expr::Int(i, 0),
            Lit::Float(f) => Expr::Float(f, 0),
            Lit::Text(s) => Expr::Str(s, 0),
        })
        .filter(|e| !boundary.contains(e))
        .take(HARVEST_MAX)
        .collect();
    // a text no comparison names, so the branch every literal misses is reached
    if base == "text" && boundary.iter().all(|b| matches!(b, Expr::Str(s, _) if lits.contains(&Lit::Text(s.clone())))) {
        let mut k = 1;
        while lits.contains(&Lit::Text("a".repeat(k))) {
            k += 1;
        }
        out.push(Expr::Str("a".repeat(k), 0));
    }
    out
}

/// Every concrete instance of the declared events, in declaration order
/// (the name-free key of 6.6), payloads in the order of their boundary sets
/// followed by the field's harvested values in value order.
fn instances(m: &Module, harvest: &Harvest) -> (Vec<EvInst>, Vec<String>) {
    let mut out = vec![];
    let mut skipped = vec![];
    // type aliases, refined or not (5.1): a refined field gets its base type's
    // boundary set, and a step that keeps the state on an out-of-range value is an observed step
    let aliases: HashMap<String, Ty> = m
        .items
        .iter()
        .filter_map(|i| match i {
            Item::TypeAlias { name, generics, ty, .. } if generics.is_empty() => Some((name.clone(), ty.clone())),
            _ => None,
        })
        .collect();
    // Declaration order, never name order: the exploration order is part of
    // the canonical control string and must not change under renaming (T61).
    let decls: Vec<(&String, &Vec<(String, Ty)>)> = m
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Event { name, payload, .. } => Some((name, payload)),
            _ => None,
        })
        .collect();
    'ev: for (ev, (name, payload)) in decls.into_iter().enumerate() {
        let mut columns: Vec<Vec<Expr>> = vec![];
        for (pos, (_, t)) in payload.iter().enumerate() {
            match boundary(t, &aliases) {
                Some(mut b) => {
                    if let Some(lits) = harvest.get(ev).and_then(|f| f.get(pos)) {
                        let extra = harvested_values(t, &aliases, lits, &b);
                        b.extend(extra);
                    }
                    columns.push(b)
                }
                None => {
                    skipped.push(format!("{name}: no boundary set for a field of type {t:?}"));
                    continue 'ev;
                }
            }
        }
        // Cartesian product, capped so one wide event cannot exhaust the fuel.
        let mut rows: Vec<Vec<Expr>> = vec![vec![]];
        for col in &columns {
            let mut next = vec![];
            for r in &rows {
                for e in col {
                    let mut r2 = r.clone();
                    r2.push(e.clone());
                    next.push(r2);
                }
            }
            rows = next;
            if rows.len() > 4096 {
                rows.truncate(4096);
            }
        }
        for r in rows {
            let args: Vec<Arg> = r
                .iter()
                .zip(payload)
                .map(|(e, (f, _))| Arg { label: Some(f.clone()), e: e.clone() })
                .collect();
            let show = format!("{}({})", name, args.iter().map(|a| expr_show(&a.e)).collect::<Vec<_>>().join(", "));
            out.push(EvInst { name: name.clone(), args, show });
        }
    }
    (out, skipped)
}

fn expr_show(e: &Expr) -> String {
    match e {
        Expr::Int(i, _) => i.to_string(),
        Expr::Float(f, _) => format!("{f:?}"),
        Expr::Str(s, _) => format!("{s:?}"),
        Expr::Bool(b, _) => b.to_string(),
        Expr::None(_) => "none".into(),
        Expr::Unit(_) => "()".into(),
        Expr::List(xs, _) => format!("[{}]", xs.iter().map(expr_show).collect::<Vec<_>>().join(", ")),
        Expr::Call(_, a, _, _) => format!("some({})", a.iter().map(|x| expr_show(&x.e)).collect::<Vec<_>>().join(", ")),
        _ => "?".into(),
    }
}

/// The label of one step: the canonical text of its outputs, or why it kept
/// the state.
fn step_label(r: &Result<(Vec<Region>, Vec<V>), String>) -> String {
    match r {
        Ok((_, outs)) => format!("ok:{}", outs.iter().map(|o| o.show()).collect::<Vec<_>>().join("|")),
        Err(x) => format!("kept:{x}"),
    }
}

/// Declaration index of every named item, in declaration order: the
/// name-free key of 6.6 for control labels, so that `shape_ctl` does not
/// depend on names (T61, T7).
fn decl_index(m: &Module) -> HashMap<String, usize> {
    // index within kind: the k-th error, the k-th struct, the k-th kind
    // variant; an `intent` or a function elsewhere in the file must not shift
    // the key of an error (the labels must agree across projections and lifts)
    let mut idx = HashMap::new();
    let (mut errors, mut structs, mut variants) = (0usize, 0usize, 0usize);
    for it in &m.items {
        match it {
            Item::Error { name, .. } => {
                idx.entry(name.clone()).or_insert(errors);
                errors += 1;
            }
            Item::Struct { name, .. } => {
                idx.entry(name.clone()).or_insert(structs);
                structs += 1;
            }
            Item::Kind { variants: vs, .. } => {
                for (vn, _) in vs {
                    idx.entry(vn.clone()).or_insert(variants);
                    variants += 1;
                }
            }
            _ => {}
        }
    }
    idx
}

/// The control label of one step (6.10 control projection): the error that
/// kept the state (by declaration index) if the step was kept, else the constructors of the
/// emitted values (by declaration index, or a type class); never a data
/// value and never a name. It is a function of the step's output alone, so
/// the control quotient is a coarsening of the behaviour quotient and
/// dim_c <= dim holds by construction. (An earlier draft also recorded
/// which regions changed; a change with no output is invisible to λ, so
/// that label was not a coarsening and the pilot showed dim_c > dim on
/// example 51.)
/// The name-free output label of a step (T7): emitted values in their
/// positional form (6.1), a kept step by the error's declaration index. Two
/// programs are compared on these, so a renaming never distinguishes them.
fn pos_label(m: &Module, idx: &HashMap<String, usize>, r: &Result<(Vec<Region>, Vec<V>), String>) -> String {
    match r {
        Ok((_, outs)) => format!("ok:{}", outs.iter().map(|o| crate::interp::form_value_pos(o, m).json()).collect::<Vec<_>>().join("|")),
        Err(x) => {
            let name = x.split('(').next().unwrap_or(x).trim();
            match idx.get(name) {
                Some(i) => format!("kept:err#{i}"),
                None => format!("kept:{x}"),
            }
        }
    }
}

fn control_label(idx: &HashMap<String, usize>, r: &Result<(Vec<Region>, Vec<V>), String>) -> String {
    match r {
        Ok((_, outs)) => {
            let emitted: Vec<String> = outs
                .iter()
                .map(|o| match o {
                    V::Rec(n, _) => idx.get(n).map_or("rec".to_string(), |i| format!("rec#{i}")),
                    V::Int(..) => "int".into(),
                    V::F(_) | V::F32(_) => "float".into(),
                    V::Bool(_) => "bool".into(),
                    V::Text(_) => "text".into(),
                    _ => "value".into(),
                })
                .collect();
            format!("ok:{}", emitted.join(","))
        }
        Err(x) => {
            let name = x.split('(').next().unwrap_or(x).trim();
            match idx.get(name) {
                Some(i) => format!("kept:err#{i}"),
                None => format!("kept:{name}"), // a trap: its text carries no declared name
            }
        }
    }
}

/// Hopcroft's partition refinement on a deterministic labelled graph
/// (n log n in the number of transitions). Returns the class of each state.
/// `succ[i][j]` is the successor on event j (`None` = unexplored: one shared
/// sink, class index `n`, never split), `label[i][j]` its output label.
/// Classes are numbered in order of first state, so the result is a
/// deterministic function of the input.
fn refine(n: usize, succ: &[Vec<Option<usize>>], label: &[Vec<String>]) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }
    let k = succ[0].len();
    // initial partition: by the full output label vector
    let mut class: Vec<usize> = {
        let mut sig: BTreeMap<&Vec<String>, usize> = BTreeMap::new();
        let mut order = vec![];
        for l in label {
            if !sig.contains_key(l) {
                sig.insert(l, order.len());
                order.push(l);
            }
        }
        label.iter().map(|l| sig[l]).collect()
    };
    // blocks as index lists; the sink is block `n_blocks` conceptually and
    // has no members, so it never needs splitting
    let mut nblocks = class.iter().copied().max().map_or(0, |m| m + 1);
    let mut blocks: Vec<Vec<usize>> = vec![vec![]; nblocks];
    for (i, &c) in class.iter().enumerate() {
        blocks[c].push(i);
    }
    // inverse transitions per event
    let mut pred: Vec<Vec<Vec<usize>>> = vec![vec![vec![]; n]; k];
    for i in 0..n {
        for (j, t) in succ[i].iter().enumerate() {
            if let Some(x) = t {
                pred[j][*x].push(i);
            }
        }
    }
    let mut work: VecDeque<(usize, usize)> = VecDeque::new(); // (block, event)
    for b in 0..nblocks {
        for j in 0..k {
            work.push_back((b, j));
        }
    }
    while let Some((b, j)) = work.pop_front() {
        // states with an event-j successor in block b
        let splitter: std::collections::BTreeSet<usize> = blocks[b].iter().flat_map(|&x| pred[j][x].iter().copied()).collect();
        if splitter.is_empty() {
            continue;
        }
        // which blocks does the splitter cut?
        let mut touched: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for &s in &splitter {
            touched.entry(class[s]).or_default().push(s);
        }
        for (c, inside) in touched {
            if inside.len() == blocks[c].len() {
                continue; // whole block is in the splitter: no split
            }
            let inside_set: std::collections::BTreeSet<usize> = inside.iter().copied().collect();
            let (yes, no): (Vec<usize>, Vec<usize>) = blocks[c].iter().partition(|x| inside_set.contains(x));
            // the smaller part becomes a new block (Hopcroft's rule); its
            // members keep deterministic numbering by first state
            let (keep, new) = if yes.len() <= no.len() { (no, yes) } else { (yes, no) };
            let nb = nblocks;
            nblocks += 1;
            for &x in &new {
                class[x] = nb;
            }
            blocks[c] = keep;
            blocks.push(new);
            for e in 0..k {
                work.push_back((nb, e));
            }
        }
    }
    // renumber classes by first occurrence so the output is canonical
    let mut renum: BTreeMap<usize, usize> = BTreeMap::new();
    class
        .iter()
        .map(|&c| {
            let next = renum.len();
            *renum.entry(c).or_insert(next)
        })
        .collect()
}

/// The canonical string of a quotient machine: states renumbered by first
/// visit in breadth-first order from the class of state 0, events in their
/// canonical order; one line per (state, event) with the successor's
/// canonical number and the label. Unique for the isomorphism class.
fn canonical_string(n: usize, class: &[usize], succ: &[Vec<Option<usize>>], label: &[Vec<String>]) -> String {
    // representative concrete state per class: the first explored (BFS order) one
    let mut rep: BTreeMap<usize, usize> = BTreeMap::new();
    for i in 0..n {
        rep.entry(class[i]).or_insert(i);
    }
    let mut number: BTreeMap<usize, usize> = BTreeMap::new();
    let mut order = vec![];
    let mut queue = VecDeque::new();
    number.insert(class[0], 0);
    queue.push_back(class[0]);
    while let Some(c) = queue.pop_front() {
        order.push(c);
        let i = rep[&c];
        for t in &succ[i] {
            if let Some(j) = t {
                let cj = class[*j];
                if !number.contains_key(&cj) {
                    let k = number.len();
                    number.insert(cj, k);
                    queue.push_back(cj);
                }
            }
        }
    }
    let mut s = String::new();
    for c in order {
        let i = rep[&c];
        for (j, t) in succ[i].iter().enumerate() {
            let target = match t {
                Some(x) => number[&class[*x]].to_string(),
                None => "?".into(),
            };
            s.push_str(&format!("{} {} {} {}\n", number[&c], j, target, label[i][j]));
        }
    }
    s
}

/// Explore and refine. `fuel` bounds the number of explored states.
pub fn dim(m: &Module, fuel: usize) -> Result<Dim, String> {
    dim_with(m, fuel, false)
}

/// Breadth-first exploration of a program from its initial state over the
/// boundary instances of its events (10.11), to at most `fuel` states.
pub fn explore(m: &Module, fuel: usize) -> Result<Graph, String> {
    explore_to(m, fuel, usize::MAX, &harvest(m))
}

/// `explore` that expands only the states fewer than `max_depth` steps from
/// s₀: a state at depth `max_depth` is stored but not stepped, and its row is
/// unexplored (`None`, empty labels). `usize::MAX` is the whole exploration.
fn explore_to(m: &Module, fuel: usize, max_depth: usize, harvest: &Harvest) -> Result<Graph, String> {
    let mut it = Interp::new(m);
    let s0 = it.initial_state().map_err(why)?;
    let (evs, skipped) = instances(m, harvest);
    let has_state = m.items.iter().any(|i| matches!(i, Item::State { .. } | Item::Actor { .. }));
    if !has_state {
        return Err("a command (no state region) is one behaviour; nothing to explore".into());
    }
    if evs.is_empty() {
        return Err(format!("no event has a boundary set; nothing to explore ({})", skipped.join("; ")));
    }
    // Build the event values once; `construct` reads the scopes of a fresh frame.
    let mut evals: Vec<(EvInst, V)> = vec![];
    let mut skipped = skipped;
    for e in &evs {
        it.scopes = vec![HashMap::new()];
        match it.construct(&e.name, &e.args) {
            Ok(Some(v)) => evals.push((e.clone(), v)),
            Ok(None) => return Err(format!("`{}` is not an event", e.name)),
            // a boundary value outside a refined type traps at construction
            // (T35): that instance cannot be an event, so it is not explored
            Err(f) => skipped.push(format!("{}: {}", e.show, why(f))),
        }
    }
    if evals.is_empty() {
        return Err(format!("no event instance can be constructed ({})", skipped.join("; ")));
    }
    let k = evals.len();
    let idx = decl_index(m);

    // Breadth-first exploration. Key = canonical JSON of the state (6.1).
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut states: Vec<Vec<Region>> = vec![];
    let mut succ: Vec<Vec<Option<usize>>> = vec![]; // None = unexplored (bound reached)
    let mut label: Vec<Vec<String>> = vec![];
    let mut clabel: Vec<Vec<String>> = vec![];
    let mut plabel: Vec<Vec<String>> = vec![];
    let mut queue = VecDeque::new();
    let key0 = state_json(&s0, m);
    index.insert(key0, 0);
    states.push(s0);
    queue.push_back(0usize);
    let mut depth: Vec<usize> = vec![0];
    let mut bound_reached = false;
    while let Some(i) = queue.pop_front() {
        if depth[i] >= max_depth {
            succ.push(vec![None; k]);
            label.push(vec![String::new(); k]);
            clabel.push(vec![String::new(); k]);
            plabel.push(vec![String::new(); k]);
            continue;
        }
        let s = states[i].clone();
        let mut row = Vec::with_capacity(k);
        let mut lab = Vec::with_capacity(k);
        let mut clab = Vec::with_capacity(k);
        let mut plab = Vec::with_capacity(k);
        for (_, ev) in &evals {
            it.scopes = vec![HashMap::new()];
            let r = it.delta(&s, ev);
            lab.push(step_label(&r));
            plab.push(pos_label(m, &idx, &r));
            clab.push(control_label(&idx, &r));
            match r {
                Ok((s2, _)) => {
                    let key = state_json(&s2, m);
                    if let Some(&j) = index.get(&key) {
                        row.push(Some(j));
                    } else if states.len() < fuel {
                        let j = states.len();
                        index.insert(key, j);
                        states.push(s2);
                        depth.push(depth[i] + 1);
                        queue.push_back(j);
                        row.push(Some(j));
                    } else {
                        bound_reached = true;
                        row.push(None);
                    }
                }
                Err(_) => row.push(Some(i)), // T8: a kept step leaves the state unchanged
            }
        }
        succ.push(row);
        label.push(lab);
        clabel.push(clab);
        plabel.push(plab);
    }
    let keys = evals
        .iter()
        .map(|(e, _)| format!("{}({})", crate::interp::event_index(m, &e.name).map(|i| i.to_string()).unwrap_or_else(|| e.name.clone()), e.args.iter().map(|a| expr_show(&a.e)).collect::<Vec<_>>().join(", ")))
        .collect();
    Ok(Graph { states, succ, label, clabel, plabel, bound_reached, skipped, alphabet: evals.iter().map(|(e, _)| e.show.clone()).collect(), keys })
}

/// The explored graph of a program: states keyed by canonical JSON, one
/// successor and one output label per state and event instance.
pub struct Graph {
    pub states: Vec<Vec<Region>>,
    pub succ: Vec<Vec<Option<usize>>>,
    pub label: Vec<Vec<String>>,
    pub clabel: Vec<Vec<String>>,
    /// name-free output labels (T7), the labels `equiv` compares
    pub plabel: Vec<Vec<String>>,
    pub bound_reached: bool,
    pub skipped: Vec<String>,
    /// the event instances explored, in canonical order, as printed
    pub alphabet: Vec<String>,
    /// the same instances with the event named by its declaration index
    /// (T7): the key two programs are compared on
    pub keys: Vec<String>,
}

/// `dim`, optionally with the control quotient (`control`).
pub fn dim_with(m: &Module, fuel: usize, control: bool) -> Result<Dim, String> {
    let Graph { states, succ, label, clabel, bound_reached, skipped, .. } = explore(m, fuel)?;
    let _ = &states;
    let n = states.len();
    let has_output = label.iter().flatten().any(|l| l != "ok:");
    let emits = label.iter().flatten().any(|l| l.starts_with("ok:") && l != "ok:");
    let keeps = label.iter().flatten().any(|l| l.starts_with("kept:"));
    let distinct_outputs = label.iter().flatten().filter(|l| *l != "ok:").collect::<std::collections::BTreeSet<_>>().len();

    // Moore refinement with the full output labels (dim) and, if asked, the
    // control labels (dim_c), on the same explored graph.
    let class = refine(n, &succ, &label);
    let behaviours = class.iter().collect::<std::collections::BTreeSet<_>>().len();
    let bits = |x: usize| if x == 0 { 0.0 } else { (x as f64).log2() };
    let redundancy_bits = bits(n) - bits(behaviours);

    // Per region: distinct values seen, and distinct values among one
    // representative per behaviour class (the smallest state key in the class).
    let mut rep: BTreeMap<usize, usize> = BTreeMap::new();
    for i in 0..n {
        let c = class[i];
        let better = match rep.get(&c) {
            None => true,
            Some(&r) => state_json(&states[i], m) < state_json(&states[r], m),
        };
        if better {
            rep.insert(c, i);
        }
    }
    let region_names: Vec<String> = states[0].iter().map(|r| r.0.clone()).collect();
    let mut regions = vec![];
    for (ri, rn) in region_names.iter().enumerate() {
        let val = |s: &Vec<Region>| state_json(std::slice::from_ref(&s[ri]), m);
        let all: std::collections::BTreeSet<String> = states.iter().map(val).collect();
        let needed: std::collections::BTreeSet<String> = rep.values().map(|&i| val(&states[i])).collect();
        regions.push((rn.clone(), all.len(), needed.len(), bits(all.len()) - bits(needed.len())));
    }
    let control = if control {
        let cclass = refine(n, &succ, &clabel);
        let dim_c = cclass.iter().collect::<std::collections::BTreeSet<_>>().len();
        let cs = canonical_string(n, &cclass, &succ, &clabel);
        use sha2::{Digest, Sha256};
        let h = Sha256::digest(crate::interp::canon_json(&serde_json::Value::String(cs)).as_bytes());
        Some((dim_c, h.iter().map(|b| format!("{b:02x}")).collect::<String>()))
    } else {
        None
    };
    Ok(Dim { states: n, bound_reached, behaviours, redundancy_bits, regions, events_skipped: skipped, control, has_output, emits, keeps, distinct_outputs })
}

/// The text report of 10.11.
pub fn report(d: &Dim) -> String {
    let mut s = String::new();
    s.push_str(&format!("states explored     {:<8} bound reached: {}\n", d.states, if d.bound_reached { "yes" } else { "no" }));
    s.push_str(&format!(
        "behaviours          {:<8} dim(P) ({})\n",
        d.behaviours,
        if d.bound_reached { "lower bound" } else { "exact" }
    ));
    s.push_str(&format!("redundancy          {:<8.2} bits: log2(states) - log2(behaviours)\n", d.redundancy_bits));
    let yn = |b: bool| if b { "yes" } else { "no" };
    s.push_str(&format!("outputs             {:<8} some explored step has a λ output (emits: {}, keeps: {})\n", yn(d.has_output), yn(d.emits), yn(d.keeps)));
    s.push_str(&format!("distinct outputs    {:<8} distinct output sequences over the explored steps (silent steps excluded)\n", d.distinct_outputs));
    for (n, all, needed, b) in &d.regions {
        s.push_str(&format!("  region {:<12} {:<8.2} ({all} values, {needed} distinguish behaviours)\n", n, b));
    }
    if let Some((c, h)) = &d.control {
        s.push_str(&format!("control states      {:<8} dim_c(P) ({})\n", c, if d.bound_reached { "lower bound" } else { "exact" }));
        s.push_str(&format!("shape_ctl           sha256:{h}\n"));
    }
    for e in &d.events_skipped {
        s.push_str(&format!("skipped             {e}\n"));
    }
    s
}

/// The JSON report (`rce-dim/1`, compact, sorted keys).
pub fn report_json(d: &Dim) -> String {
    let v = serde_json::json!({
        "format": "rce-dim/1",
        "states": d.states,
        "bound_reached": d.bound_reached,
        "behaviours": d.behaviours,
        "redundancy_bits": d.redundancy_bits,
        "has_output": d.has_output,
        "emits": d.emits,
        "keeps": d.keeps,
        "distinct_outputs": d.distinct_outputs,
        "regions": d.regions.iter().map(|(n, a, b, bits)| serde_json::json!({"name": n, "values": a, "needed": b, "bits": bits})).collect::<Vec<_>>(),
        "skipped": d.events_skipped,
        "control_states": d.control.as_ref().map(|c| c.0),
        "shape_ctl": d.control.as_ref().map(|c| c.1.clone()),
    });
    crate::interp::canon_json(&v)
}

/// The result of `rce equiv A B`: behaviour equivalence of two programs on
/// the explored alphabet (6.10, T58), with a shortest distinguishing event
/// sequence when they differ.
pub struct Equiv {
    pub equivalent: bool,
    /// neither exploration reached its bound: the verdict is over the whole
    /// reachable graph of both programs on this alphabet
    pub exact: bool,
    pub states_a: usize,
    pub states_b: usize,
    pub alphabet: Vec<String>,
    /// event instances explored by one program and not the other
    pub only_a: Vec<String>,
    pub only_b: Vec<String>,
    /// (event sequence, A's output at its end, B's output at its end)
    pub witness: Option<(Vec<String>, String, String)>,
    /// an exploration reached its bound and the refinement separates the
    /// initial states with no distinguishing sequence in the explored graph:
    /// the separation may come from the unexplored frontier (10.11)
    pub undecided: bool,
}

/// Behaviour equivalence of two programs (6.10): explore both over the same
/// event instances, refine the disjoint union by output labels, and read
/// whether the two initial states fall in one class (T58: equivalence is
/// bisimilarity). The verdict is over the explored alphabet; T59 bounds what
/// trace agreement to a depth proves.
pub fn equiv(a: &Module, b: &Module, fuel: usize) -> Result<Equiv, String> {
    // one alphabet for both: the union of their harvested literals
    let h = union(&harvest(a), &harvest(b));
    let ga = explore_to(a, fuel, usize::MAX, &h)?;
    let gb = explore_to(b, fuel, usize::MAX, &h)?;
    let sa: std::collections::BTreeSet<&String> = ga.keys.iter().collect();
    let sb: std::collections::BTreeSet<&String> = gb.keys.iter().collect();
    let only_a: Vec<String> = ga.keys.iter().zip(&ga.alphabet).filter(|(k, _)| !sb.contains(k)).map(|(_, s)| s.clone()).collect();
    let only_b: Vec<String> = gb.keys.iter().zip(&gb.alphabet).filter(|(k, _)| !sa.contains(k)).map(|(_, s)| s.clone()).collect();
    if !only_a.is_empty() || !only_b.is_empty() || ga.keys != gb.keys {
        return Ok(Equiv { equivalent: false, exact: !ga.bound_reached && !gb.bound_reached, states_a: ga.states.len(), states_b: gb.states.len(), alphabet: ga.alphabet.clone(), only_a, only_b, witness: None, undecided: false });
    }
    let na = ga.states.len();
    let nb = gb.states.len();
    let mut succ = ga.succ.clone();
    succ.extend(gb.succ.iter().map(|row| row.iter().map(|x| x.map(|j| j + na)).collect::<Vec<_>>()));
    let mut label = ga.plabel.clone();
    label.extend(gb.plabel.iter().cloned());
    let class = refine(na + nb, &succ, &label);
    let equivalent = class[0] == class[na];
    // shortest distinguishing sequence: breadth-first over state pairs
    let mut witness = None;
    if !equivalent {
        let k = ga.alphabet.len();
        let mut seen = std::collections::HashSet::new();
        let mut q = VecDeque::new();
        q.push_back((0usize, 0usize, Vec::<usize>::new()));
        seen.insert((0usize, 0usize));
        'bfs: while let Some((i, j, path)) = q.pop_front() {
            for e in 0..k {
                if ga.plabel[i][e] != gb.plabel[j][e] {
                    let mut p: Vec<String> = path.iter().map(|&x| ga.alphabet[x].clone()).collect();
                    p.push(ga.alphabet[e].clone());
                    witness = Some((p, ga.label[i][e].clone(), gb.label[j][e].clone()));
                    break 'bfs;
                }
                if let (Some(i2), Some(j2)) = (ga.succ[i][e], gb.succ[j][e]) {
                    if seen.insert((i2, j2)) {
                        let mut p = path.clone();
                        p.push(e);
                        q.push_back((i2, j2, p));
                    }
                }
            }
        }
    }
    let exact = !ga.bound_reached && !gb.bound_reached;
    let undecided = !equivalent && witness.is_none() && !exact;
    Ok(Equiv { equivalent, exact, states_a: na, states_b: nb, alphabet: ga.alphabet.clone(), only_a, only_b, witness, undecided })
}

fn show_output(l: &str) -> String {
    if l == "ok:" { "silent".into() } else if let Some(x) = l.strip_prefix("ok:") { format!("emits {x}") } else if let Some(x) = l.strip_prefix("kept:") { format!("kept: {x}") } else { l.to_string() }
}

/// The text report of `rce equiv`.
pub fn equiv_report(e: &Equiv) -> String {
    let mut s = String::new();
    s.push_str(&format!("alphabet            {:<8} event instances explored\n", e.alphabet.len()));
    s.push_str(&format!("states              {} / {}\n", e.states_a, e.states_b));
    if !e.only_a.is_empty() || !e.only_b.is_empty() {
        s.push_str("verdict             different  the programs explore different event instances\n");
        for x in &e.only_a { s.push_str(&format!("  only A            {x}\n")); }
        for x in &e.only_b { s.push_str(&format!("  only B            {x}\n")); }
        return s;
    }
    if e.equivalent {
        s.push_str(&format!("verdict             equivalent  same behaviour on the explored alphabet ({})\n", if e.exact { "exact" } else { "to the explored bound" }));
    } else if let Some((seq, la, lb)) = &e.witness {
        s.push_str("verdict             different\n");
        s.push_str(&format!("after               {}\n", seq.join(", ")));
        s.push_str(&format!("  A                 {}\n", show_output(la)));
        s.push_str(&format!("  B                 {}\n", show_output(lb)));
    } else if e.undecided {
        s.push_str("verdict             undecided  the exploration reached its bound before a distinguishing event sequence or equivalence was established: raise fuel\n");
    } else {
        s.push_str("verdict             different  (no finite witness within the explored graph)\n");
    }
    s
}

/// The JSON report of `rce equiv` (format `rce-equiv/1`).
pub fn equiv_json(e: &Equiv) -> String {
    let v = serde_json::json!({
        "format": "rce-equiv/1",
        "equivalent": e.equivalent,
        "exact": e.exact,
        "undecided": e.undecided,
        "states": [e.states_a, e.states_b],
        "alphabet": e.alphabet,
        "only_a": e.only_a,
        "only_b": e.only_b,
        "witness": e.witness.as_ref().map(|(seq, la, lb)| serde_json::json!({"after": seq, "a": la, "b": lb})),
    });
    crate::interp::canon_json(&v)
}

/// The result of `rce fingerprint FILE`: a behaviour fingerprint (proposal
/// `behaviour-fingerprint.md`, BF-1).
pub struct Fingerprint {
    /// `sha256:` and 64 hex digits
    pub hash: String,
    pub depth: usize,
    /// event instances in the alphabet (as `rce equiv` explores them)
    pub alphabet: usize,
    /// states within `depth - 1` steps of s₀, each stepped on every instance
    pub states: usize,
    /// event sequences the hash covers: alphabet^1 + ... + alphabet^depth
    pub sequences: u128,
}

/// The behaviour fingerprint of a program to depth `n` (BF-1): the SHA-256
/// of the name-free outputs (the labels `rce equiv` compares) of every event
/// sequence of length 1 to `n` over the alphabet `rce equiv` explores, in the
/// canonical order of that alphabet. The trace tree is hashed bottom up
/// (h₀ = a constant; h_d(s) = SHA-256 over the instances e in order of the
/// output of e at s and h_{d-1} of its successor), so the cost is one hash
/// per (state within depth, instance), never one per sequence, and the
/// result is a function of the outputs alone: two programs with the same
/// output on every sequence of length at most `n` have the same fingerprint.
/// The alphabet's name-free keys and `n` are hashed in, so two programs that
/// `rce equiv` cannot compare (different alphabets) never share one. `fuel`
/// bounds the states stepped; a fingerprint that would need an unexplored
/// state is not computed.
pub fn fingerprint(m: &Module, depth: usize, fuel: usize) -> Result<Fingerprint, String> {
    use sha2::{Digest, Sha256};
    if depth == 0 {
        return Err("depth N: N is at least 1".into());
    }
    let g = explore_to(m, fuel, depth, &harvest(m))?;
    let k = g.keys.len();
    let n = g.states.len();
    // depth of each state (BFS from s0 over explored rows)
    let mut dist = vec![usize::MAX; n];
    dist[0] = 0;
    let mut q = VecDeque::from([0usize]);
    while let Some(i) = q.pop_front() {
        for t in g.succ[i].iter().flatten() {
            if dist[*t] == usize::MAX {
                dist[*t] = dist[i] + 1;
                q.push_back(*t);
            }
        }
    }
    let leaf: [u8; 32] = Sha256::digest(b"rce-fingerprint/1 leaf").into();
    // h[s] at remaining depth d, computed for d = 1..=depth; only states with
    // dist + d <= depth are needed
    let mut prev: Vec<Option<[u8; 32]>> = vec![Some(leaf); n];
    for d in 1..=depth {
        let mut cur: Vec<Option<[u8; 32]>> = vec![None; n];
        for i in 0..n {
            if dist[i] == usize::MAX || dist[i] + d > depth {
                continue;
            }
            let mut h = Sha256::new();
            for e in 0..k {
                let l = g.plabel[i][e].as_bytes();
                h.update((l.len() as u64).to_le_bytes());
                h.update(l);
                let next = match g.succ[i][e] {
                    Some(j) => prev[j],
                    None if d == 1 => Some(leaf),
                    None => None,
                };
                match next {
                    Some(x) => h.update(x),
                    None => return Err(format!("fuel reached before depth {depth} ({n} states); raise fuel or lower depth")),
                }
            }
            cur[i] = Some(h.finalize().into());
        }
        prev = cur;
    }
    let root = prev[0].ok_or("internal: no root hash")?;
    let mut h = Sha256::new();
    h.update(format!("rce-fingerprint/1 depth {depth} alphabet {}\n", k).as_bytes());
    for key in &g.keys {
        h.update((key.len() as u64).to_le_bytes());
        h.update(key.as_bytes());
    }
    h.update(root);
    let hash = format!("sha256:{}", h.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>());
    let stepped = (0..n).filter(|&i| dist[i] < depth).count();
    let sequences = (1..=depth as u32).fold(0u128, |acc, d| acc.saturating_add((k as u128).saturating_pow(d)));
    Ok(Fingerprint { hash, depth, alphabet: k, states: stepped, sequences })
}

/// The text report of `rce fingerprint`.
pub fn fingerprint_report(f: &Fingerprint) -> String {
    format!(
        "fingerprint         {}\ndepth               {:<8} event sequences of length 1 to {} ({} sequences)\nalphabet            {:<8} event instances explored\nstates              {:<8} states stepped\n",
        f.hash, f.depth, f.depth, f.sequences, f.alphabet, f.states
    )
}

/// The JSON report of `rce fingerprint` (format `rce-fingerprint/1`).
pub fn fingerprint_json(f: &Fingerprint) -> String {
    let v = serde_json::json!({
        "format": "rce-fingerprint/1",
        "fingerprint": f.hash,
        "depth": f.depth,
        "alphabet": f.alphabet,
        "states": f.states,
        "sequences": f.sequences.to_string(),
    });
    crate::interp::canon_json(&v)
}
