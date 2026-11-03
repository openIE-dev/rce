//! `rce repair` (10.13): a ladder of exact proposers for a program that
//! requires change. Each rung proposes candidate programs by a fixed,
//! deterministic rule; the checker (`check_program`, the same gate as
//! `rce check`, which runs every example and check) judges each one, and the
//! ladder stops at the first candidate the checker admits. No rung reads
//! anything but the program, the modules given with it and the built-in
//! library, and no rung calls a model. Rung f, after them, judges proposals
//! written outside RCE and given as a file (`proposals.rs`, RF-1 to RF-4);
//! it too calls no model.
//!
//! The rungs, in order:
//!
//! - **a, reuse**: replace one definition of the program by an existing
//!   definition with the same name-free signature (parameter types and
//!   result type), from the program itself, the modules given with it and the
//!   built-in library. Donors whose name-free projection (10.6) is equal are
//!   tried once; a donor with the same name is tried first.
//! - **b, lookup by type**: from the brick catalog (10.10) of the program,
//!   its other modules and the library modules it uses, (1) the body of a
//!   definition becomes a call of another visible function of the same type
//!   on the same parameters, and (2) the callee of a call becomes another
//!   visible function of the callee's type, in `rce search` rank order.
//! - **c, altered rules**: the text `rce fmt` writes for the rules whose
//!   method is altered (7.1, `alter.rs`: W-UNUSED-1 and W-DROP-1), every
//!   rewrite in the order `alter` writes them; one candidate, the canonical
//!   text of that program, and none when `alter` writes nothing.
//! - **d, bounded enumeration**: one small edit at one site: an operator
//!   swapped within its class, an integer literal moved to n + 1, n - 1, 0
//!   or 1, a boolean literal flipped, the operands of an ordering comparison
//!   swapped, two arguments of a call swapped, a condition negated (or its
//!   `not` removed). The edits never introduce or rename a name, so they act
//!   the same on the name-free projection. At most `bound` candidates
//!   (default 2000) are judged.
//! - **e, renames**: one name changed to another: a use of a value name (or
//!   of a name the report names) becomes another value name in scope there,
//!   or a binding becomes a name the report names. A rename carries no
//!   information about behaviour: the name-free projection (10.6) removes
//!   names, so a consistent renaming leaves it unchanged, and a use renamed
//!   to another binding is what the checker sees. With `known GOOD`, when
//!   the canonical texts of FILE and GOOD differ in identifiers only, the
//!   renaming onto GOOD is read from the two texts and judged before rung
//!   a (at most two candidates, no search). At most `bound` candidates.
//!
//! The rungs that edit the program's own text at one site (c, d, e) run in
//! order of how many alternatives an edit chooses among: c judges at most
//! one candidate and changes no behaviour; d chooses among a fixed handful
//! at each site (an operator of its class, four literals, one negation);
//! e chooses among the names in scope, a number that grows with the
//! definition.
//!
//! Sites are visited in a fixed order: definitions named by a failing line
//! first, then the definitions reachable from the failing checks and
//! examples (breadth first), then every other definition in source order;
//! within a definition, expressions in preorder. Intent, example and check
//! items, and `check`, `requires` and `ensures` statements, are never edited:
//! they are what the candidate is judged by.
//!
//! - **f, outside proposals**, only when a to e admit nothing: each proposal
//!   of the file `proposals PROPOSALS` (canonical JSON, each with its origin
//!   label and recorded by its digest) whose text parses, names FILE's
//!   module and keeps FILE's obligations (RF-2) is judged as its canonical
//!   text by the same gate. Rung f has a mode, read from the modes table of
//!   10.13 (RF-3): in `observe` every proposal is judged and each one the
//!   checker admits is reported as observed, with its diff, digest and
//!   origin, and nothing closes; in `auto` the first admitted proposal
//!   closes the program, labelled stochastic; `off` runs nothing. At most
//!   `bound` proposals are judged.
use crate::admission::Mode;
use crate::ast::*;
use crate::unparse::{expr as show, unparse};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};

pub const DEFAULT_BOUND: usize = 2000;

/// One definition of the module: the unit a rung edits.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Unit {
    Fn(usize),
    ImplFn(usize, usize),
    Step(usize),
    Actor(usize),
    Const(usize),
    State(usize),
}

/// What the ladder did.
pub struct Outcome {
    pub closed: bool,
    /// "a" to "f", or "none" when the program already complies
    pub rung: Option<String>,
    pub edit: Option<String>,
    pub diff: Vec<String>,
    pub tried: BTreeMap<String, usize>,
    pub verdict: String,
    pub attempts: Vec<Value>,
    pub skipped: Vec<String>,
    pub repaired: Option<String>,
    /// whether the canonical text of the input gets the same report as the
    /// input itself (every candidate is a canonical text)
    pub canonical_same_report: Option<bool>,
    /// the bound of rungs d, e and f
    pub bound: usize,
    /// the SHA-256 of each input file's bytes (RF-4)
    pub inputs: Value,
    /// what rung f did, when a proposals file was given
    pub proposals: Option<RungF>,
    /// closed by rung f (RF-3): a proposal written outside RCE
    pub stochastic: bool,
}

/// What rung f was given and did (RF-4).
pub struct RungF {
    pub mode: Mode,
    /// the SHA-256 of the proposals file's bytes
    pub sha256: String,
    /// whether rung f ran: rungs a to e admitted nothing, FILE parses and the
    /// mode is not `off`
    pub ran: bool,
    pub given: usize,
    pub judged: usize,
    pub admitted: usize,
    pub observed: usize,
    /// one record per proposal, in the order of the file
    pub records: Vec<Value>,
}

/// The proposals rung f judges and the mode it runs in.
pub struct Outside<'p> {
    pub proposals: &'p crate::proposals::Proposals,
    pub mode: Mode,
}

struct Ctx<'a> {
    file: String,
    others: &'a [(String, String)],
    attempts: Vec<Value>,
    tried: BTreeMap<String, usize>,
    seen: HashSet<String>,
}

impl Ctx<'_> {
    /// Judge one candidate text with the checker; true when admitted.
    fn judge(&mut self, rung: &str, unit: &str, edit: &str, text: &str) -> Option<String> {
        if !self.seen.insert(text.to_string()) {
            return None;
        }
        *self.tried.entry(rung.to_string()).or_insert(0) += 1;
        let mut files = vec![(self.file.clone(), text.to_string())];
        files.extend(self.others.iter().cloned());
        let v = crate::check_program(&files, false).remove(0).1;
        let verdict = match &v {
            Ok(()) => "admitted".to_string(),
            Err(e) => format!("requires change: {}", e.chars().take(200).collect::<String>()),
        };
        self.attempts.push(json!({"rung": rung, "definition": unit, "edit": edit, "verdict": verdict}));
        v.ok().map(|_| text.to_string())
    }

    /// Judge one proposal of rung f by the same gate (`check_program`: the
    /// admission of `rce check`, with the mutation gate W-MUT-1), with what
    /// the mutation gate measured of the admitted text.
    fn judge_measured(&mut self, unit: &str, edit: &str, text: &str) -> (String, Option<Value>) {
        self.seen.insert(text.to_string());
        *self.tried.entry("f".to_string()).or_insert(0) += 1;
        let mut files = vec![(self.file.clone(), text.to_string())];
        files.extend(self.others.iter().cloned());
        let a = crate::admission::program(crate::admission::RuleSet::in_force(), &files, true).remove(0).1;
        let (verdict, mutation) = match a {
            Err(e) => (format!("requires change: {}", e.chars().take(200).collect::<String>()), None),
            Ok(a) if a.errs.is_empty() => {
                let measured = a.mutation.iter().filter(|r| r.measured).count();
                ("admitted".to_string(), Some(json!({"obligations": a.mutation.len(), "measured": measured, "vacuous": 0})))
            }
            Ok(a) => (format!("requires change: {}", a.errs.join("; ").chars().take(200).collect::<String>()), None),
        };
        self.attempts.push(json!({"rung": "f", "definition": unit, "edit": edit, "verdict": verdict}));
        (verdict, mutation)
    }

    /// A proposal of rung f refused before admission runs (its text does
    /// not parse, or it does not keep FILE's obligations): judged, logged.
    fn refuse(&mut self, unit: &str, edit: &str, verdict: &str) {
        *self.tried.entry("f".to_string()).or_insert(0) += 1;
        self.attempts.push(json!({"rung": "f", "definition": unit, "edit": edit, "verdict": verdict}));
    }
}

/// Run the ladder on `file` (its text `src`), checked together with `others`.
pub fn repair(file: &str, src: &str, others: &[(String, String)], known: Option<&Known>, bound: usize) -> Outcome {
    repair_with(file, src, others, known, None, bound)
}

/// The ladder, with rung f judging `outside` after rungs a to e (RF-2,
/// RF-3) when they admit nothing.
pub fn repair_with(file: &str, src: &str, others: &[(String, String)], known: Option<&Known>, outside: Option<Outside>, bound: usize) -> Outcome {
    let mut files = vec![(file.to_string(), src.to_string())];
    files.extend(others.iter().cloned());
    let first = crate::check_program(&files, false).remove(0).1;
    let sha = |t: &str| crate::manifest::sha256_hex(t.as_bytes());
    let inputs = json!({
        "file": sha(src),
        "known": known.map(|k| k.sha256.clone()),
        "proposals": outside.as_ref().map(|o| o.proposals.sha256.clone()),
        "modules": others.iter().map(|(_, t)| sha(t)).collect::<Vec<_>>(),
    });
    let mut rf = outside.as_ref().map(|o| RungF {
        mode: o.mode,
        sha256: o.proposals.sha256.clone(),
        ran: false,
        given: o.proposals.list.len(),
        judged: 0,
        admitted: 0,
        observed: 0,
        records: vec![],
    });
    let mut out = Outcome {
        closed: false,
        rung: None,
        edit: None,
        diff: vec![],
        tried: BTreeMap::new(),
        verdict: String::new(),
        attempts: vec![],
        skipped: vec![],
        repaired: None,
        canonical_same_report: None,
        bound,
        inputs,
        proposals: None,
        stochastic: false,
    };
    // a proposal that rung f does not judge is recorded with the reason
    let unjudged = |rf: &mut Option<RungF>, why: &str| {
        if let (Some(r), Some(o)) = (rf.as_mut(), outside.as_ref()) {
            r.records = o.proposals.list.iter().enumerate().map(|(i, p)| record(i, p, &format!("not judged: {why}"))).collect();
        }
    };
    let err = match first {
        Ok(()) => {
            out.closed = true;
            out.rung = Some("none".into());
            out.verdict = format!("ok {file}: complies with the Form; nothing to repair");
            unjudged(&mut rf, "FILE complies with the Form");
            out.proposals = rf;
            return out;
        }
        Err(e) => e,
    };
    out.verdict = format!("requires change {file}: {err}");
    let m = match crate::parse::parse(src) {
        Ok(m) => m,
        Err(e) => {
            out.skipped.push(format!("a, b, c, d, e: the text does not parse ({e}); every rung edits the parsed module"));
            if rf.is_some() {
                out.skipped.push("f: the text does not parse; rung f compares each proposal with the obligations of the parsed FILE (RF-2)".into());
            }
            unjudged(&mut rf, "FILE does not parse");
            out.proposals = rf;
            return out;
        }
    };
    let base = unparse(&m);
    let mut bfiles = vec![(file.to_string(), base.clone())];
    bfiles.extend(others.iter().cloned());
    let same = crate::check_program(&bfiles, false).remove(0).1.err().map(|e| strip_lines(&e)) == Some(strip_lines(&err));
    out.canonical_same_report = Some(same);
    let mut cx = Ctx { file: file.to_string(), others, attempts: vec![], tried: BTreeMap::new(), seen: HashSet::new() };
    cx.seen.insert(base.clone());
    for r in ["a", "b", "c", "d", "e"] {
        cx.tried.insert(r.into(), 0);
    }
    if outside.is_some() {
        cx.tried.insert("f".into(), 0);
    }
    let units = order_units(&m, &err);
    // with `known GOOD`, the renaming onto GOOD (no search) first
    let found = known
        .and_then(|k| rung_e_exact(&base, k, &mut cx))
        .or_else(|| rung_a(&m, &units, others, &mut cx))
        .or_else(|| rung_b(&m, &units, file, src, others, &mut cx))
        .or_else(|| rung_c(src, &mut cx))
        .or_else(|| rung_d(&m, &units, bound, &mut cx))
        .or_else(|| rung_e(&m, &units, &err, bound, &mut cx));
    // rung f, only when rungs a to e admitted nothing (RF-2)
    let found = if let Some(f) = found {
        unjudged(&mut rf, &format!("rung {} closed the program", f.0));
        Some(f)
    } else if outside.as_ref().is_some_and(|o| o.mode == Mode::Off) {
        out.skipped.push("f: rung f is off (its row in the modes table of 10.13)".into());
        unjudged(&mut rf, "rung f is off");
        None
    } else if let (Some(o), Some(r)) = (outside.as_ref(), rf.as_mut()) {
        rung_f(&base, &m, o, known, bound, &mut cx, r)
    } else {
        None
    };
    out.stochastic = found.as_ref().is_some_and(|f| f.0 == "f");
    out.proposals = rf;
    out.tried = cx.tried;
    out.attempts = cx.attempts;
    if let Some((rung, edit, text)) = found {
        out.closed = true;
        out.rung = Some(rung);
        out.edit = Some(edit);
        out.diff = diff(&base, &text);
        let (e, c) = parse_counts(&text);
        out.verdict = format!("ok {file}: admitted; {e} examples and {c} checks hold");
        out.repaired = Some(text);
    }
    out
}

fn parse_counts(text: &str) -> (usize, usize) {
    crate::parse::parse(text).map(|m| crate::cli::count_tests(&m)).unwrap_or((0, 0))
}

/// The changed lines of two canonical texts: the lines between their common
/// prefix and common suffix, `- ` old and `+ ` new, with line numbers of the
/// canonical text.
pub fn diff(a: &str, b: &str) -> Vec<String> {
    let x: Vec<&str> = a.lines().collect();
    let y: Vec<&str> = b.lines().collect();
    let mut p = 0;
    while p < x.len() && p < y.len() && x[p] == y[p] {
        p += 1;
    }
    let mut s = 0;
    while s < x.len() - p && s < y.len() - p && x[x.len() - 1 - s] == y[y.len() - 1 - s] {
        s += 1;
    }
    let mut out = vec![];
    for (i, l) in x[p..x.len() - s].iter().enumerate() {
        out.push(format!("-{:>4} {l}", p + i + 1));
    }
    for (i, l) in y[p..y.len() - s].iter().enumerate() {
        out.push(format!("+{:>4} {l}", p + i + 1));
    }
    out
}

// ---------- definitions and their order ----------

pub(crate) fn units(m: &Module) -> Vec<Unit> {
    let mut v = vec![];
    for (i, it) in m.items.iter().enumerate() {
        match it {
            Item::Fn(_) => v.push(Unit::Fn(i)),
            Item::Impl { members, .. } => v.extend((0..members.len()).map(|j| Unit::ImplFn(i, j))),
            Item::Step { .. } => v.push(Unit::Step(i)),
            Item::Actor { .. } => v.push(Unit::Actor(i)),
            Item::Const { .. } => v.push(Unit::Const(i)),
            Item::State { .. } => v.push(Unit::State(i)),
            _ => {}
        }
    }
    v
}

pub(crate) fn unit_name(m: &Module, u: &Unit) -> String {
    match u {
        Unit::Fn(i) => match &m.items[*i] {
            Item::Fn(f) => f.sig.name.clone(),
            _ => unreachable!(),
        },
        Unit::ImplFn(i, j) => match &m.items[*i] {
            Item::Impl { members, ty, .. } => format!("{}.{}", crate::unparse::ty(ty), members[*j].sig.name),
            _ => unreachable!(),
        },
        Unit::Step(i) => match &m.items[*i] {
            Item::Step { region, pat, .. } => format!("step {region} on {}", crate::unparse::pat(pat)),
            _ => unreachable!(),
        },
        Unit::Actor(i) | Unit::Const(i) | Unit::State(i) => crate::link::item_name(&m.items[*i]).unwrap_or_default(),
    }
}

/// The name a call reaches this unit by (functions, methods, constants).
pub(crate) fn callable_name(m: &Module, u: &Unit) -> Option<String> {
    match u {
        Unit::Fn(i) | Unit::Const(i) => crate::link::item_name(&m.items[*i]),
        Unit::ImplFn(i, j) => match &m.items[*i] {
            Item::Impl { members, .. } => Some(members[*j].sig.name.clone()),
            _ => None,
        },
        _ => None,
    }
}

pub(crate) fn unit_line(m: &Module, u: &Unit) -> usize {
    match u {
        Unit::Fn(i) => match &m.items[*i] {
            Item::Fn(f) => f.sig.line,
            _ => 0,
        },
        Unit::ImplFn(i, j) => match &m.items[*i] {
            Item::Impl { members, .. } => members[*j].sig.line,
            _ => 0,
        },
        Unit::Step(i) | Unit::Actor(i) | Unit::Const(i) | Unit::State(i) => match &m.items[*i] {
            Item::Step { line, .. } | Item::Actor { line, .. } | Item::Const { line, .. } | Item::State { line, .. } => *line,
            _ => 0,
        },
    }
}

/// Names (bare and member) an expression tree mentions.
pub(crate) fn mentions(u: &Unit, m: &Module) -> BTreeSet<String> {
    let mut mm = m.clone();
    let mut names = BTreeSet::new();
    let mut f = |e: &mut Expr, _: bool| {
        match e {
            Expr::Name(n, _) => {
                names.insert(n.clone());
            }
            Expr::Member(_, n, _) => {
                names.insert(n.clone());
            }
            _ => {}
        }
        false
    };
    visit_unit(&mut mm, u, &mut f);
    names
}

pub(crate) fn expr_names(e: &Expr) -> BTreeSet<String> {
    let mut e = e.clone();
    let mut names = BTreeSet::new();
    let mut f = |e: &mut Expr, _: bool| {
        if let Expr::Name(n, _) | Expr::Member(_, n, _) = e {
            names.insert(n.clone());
        }
        false
    };
    let mut v = Visit { f: &mut f };
    v.expr(&mut e, false);
    names
}

/// The definitions in the order the rungs visit them (module comment).
fn order_units(m: &Module, err: &str) -> Vec<Unit> {
    let all = units(m);
    let lines: BTreeSet<usize> = regex_lines(err);
    let mut order: Vec<Unit> = vec![];
    // definitions whose span holds a line the report names (a static rule)
    let mut starts: Vec<(usize, Unit)> = all.iter().map(|u| (unit_line(m, u), u.clone())).collect();
    starts.sort_by_key(|x| x.0);
    for &l in &lines {
        if let Some((_, u)) = starts.iter().rev().find(|(s, _)| *s <= l && *s > 0)
            && !order.contains(u)
            && !is_spec_line(m, l)
        {
            order.push(u.clone());
        }
    }
    // roots: names in the failing checks; steps and actors for failing examples
    let mut roots: BTreeSet<String> = BTreeSet::new();
    let mut example_fails = false;
    for it in &m.items {
        match it {
            Item::Check(e) if lines.contains(&e.line()) => roots.extend(expr_names(e)),
            Item::Intent(_, xs, cs) => {
                for c in cs {
                    if lines.contains(&c.line()) {
                        roots.extend(expr_names(c));
                    }
                }
                if xs.iter().any(|x| lines.contains(&x.line)) {
                    example_fails = true;
                }
            }
            Item::Example(x) if lines.contains(&x.line) => example_fails = true,
            _ => {}
        }
    }
    let mut queue: VecDeque<Unit> = VecDeque::new();
    if example_fails {
        for u in &all {
            if matches!(u, Unit::Step(_) | Unit::Actor(_) | Unit::State(_)) {
                queue.push_back(u.clone());
            }
        }
    }
    for u in &all {
        if callable_name(m, u).is_some_and(|n| roots.contains(&n)) {
            queue.push_back(u.clone());
        }
    }
    let mut seen: HashSet<usize> = HashSet::new();
    while let Some(u) = queue.pop_front() {
        let k = all.iter().position(|x| *x == u).unwrap();
        if !seen.insert(k) {
            continue;
        }
        if !order.contains(&u) {
            order.push(u.clone());
        }
        let ns = mentions(&u, m);
        for v in &all {
            if callable_name(m, v).is_some_and(|n| ns.contains(&n)) {
                queue.push_back(v.clone());
            }
        }
    }
    for u in all {
        if !order.contains(&u) {
            order.push(u);
        }
    }
    order
}

fn is_spec_line(m: &Module, l: usize) -> bool {
    m.items.iter().any(|it| match it {
        Item::Check(e) => e.line() == l,
        Item::Example(x) => x.line == l,
        Item::Intent(_, xs, cs) => xs.iter().any(|x| x.line == l) || cs.iter().any(|c| c.line() == l),
        _ => false,
    })
}

/// A report without its line numbers, which the canonical layout moves.
fn strip_lines(err: &str) -> String {
    let mut o = String::new();
    let mut rest = err;
    while let Some(p) = rest.find("line ") {
        o.push_str(&rest[..p]);
        let t = &rest[p + 5..];
        let n = t.bytes().take_while(|b| b.is_ascii_digit()).count();
        if n == 0 {
            o.push_str("line ");
        }
        rest = &t[n..];
    }
    o.push_str(rest);
    o
}

fn regex_lines(err: &str) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    let b = err.as_bytes();
    let mut i = 0;
    while let Some(p) = err[i..].find("line ") {
        let s = i + p + 5;
        let mut e = s;
        while e < b.len() && b[e].is_ascii_digit() {
            e += 1;
        }
        if e > s {
            out.insert(err[s..e].parse().unwrap_or(0));
        }
        i = s;
    }
    out
}

// ---------- the expression walker ----------

/// Preorder walk over every editable expression; `f(e, is_condition)`
/// returns true to stop the walk.
struct Visit<'f> {
    f: &'f mut dyn FnMut(&mut Expr, bool) -> bool,
}

impl Visit<'_> {
    fn expr(&mut self, e: &mut Expr, cond: bool) -> bool {
        if (self.f)(e, cond) {
            return true;
        }
        use Expr as E;
        match e {
            E::Tuple(v, _) | E::List(v, _) => v.iter_mut().any(|x| self.expr(x, false)),
            E::ListRepeat(a, b, _) => self.expr(a, false) || self.expr(b, false),
            E::ListComp(a, _, s, c, _) => {
                self.expr(a, false) || self.expr(s, false) || c.as_mut().is_some_and(|c| self.expr(c, true))
            }
            E::Map(v, _) => v.iter_mut().any(|(k, x)| self.expr(k, false) || self.expr(x, false)),
            E::ChanNew(_, c, _) => c.as_mut().is_some_and(|c| self.expr(c, false)),
            E::Call(f, a, _, _) => self.expr(f, false) || a.iter_mut().any(|x| self.expr(&mut x.e, false)),
            E::Index(x, ix, _) => self.expr(x, false) || ix.iter_mut().any(|i| self.expr(i, false)),
            E::Slice(x, lo, hi, _) => {
                self.expr(x, false)
                    || lo.as_mut().is_some_and(|l| self.expr(l, false))
                    || hi.as_mut().is_some_and(|h| self.expr(h, false))
            }
            E::Member(x, _, _)
            | E::TupleMember(x, _, _)
            | E::Neg(x, _)
            | E::BitNot(x, _)
            | E::Borrow(_, x, _)
            | E::Recv(x, _)
            | E::Comptime(x, _)
            | E::Cast(x, _, _) => self.expr(x, false),
            E::Not(x, _) => self.expr(x, cond),
            E::Bin(op, a, b, _) => {
                let c = cond && (op == "and" || op == "or");
                self.expr(a, c) || self.expr(b, c)
            }
            E::Catch(a, _, h, _) => self.expr(a, false) || self.expr(h, false),
            E::Try(a, el, _) => self.expr(a, false) || el.as_mut().is_some_and(|x| self.expr(x, false)),
            E::If(c, a, b, _) => self.expr(c, true) || self.expr(a, false) || self.expr(b, false),
            E::Lambda(_, b, _) => self.expr(b, false),
            _ => false,
        }
    }

    fn stmts(&mut self, b: &mut [Stmt]) -> bool {
        b.iter_mut().any(|s| self.stmt(s))
    }

    fn stmt(&mut self, s: &mut Stmt) -> bool {
        match s {
            Stmt::Expr(e) | Stmt::Emit(e) | Stmt::Fail(e) | Stmt::Spawn(e) | Stmt::Yield(e, _) => self.expr(e, false),
            Stmt::Assign(_, a, b) => self.expr(a, false) || self.expr(b, false),
            Stmt::Send(a, b) => self.expr(a, false) || self.expr(b, false),
            Stmt::Let(_, _, e, _) | Stmt::Var(_, _, e, _) => self.expr(e, false),
            Stmt::If(arms, el) => {
                arms.iter_mut().any(|(c, b)| self.expr(c, true) || self.stmts(b)) || el.as_mut().is_some_and(|b| self.stmts(b))
            }
            Stmt::While(c, bd, b) => self.expr(c, true) || bd.as_mut().is_some_and(|x| self.expr(x, false)) || self.stmts(b),
            Stmt::Loop(bd, b) => bd.as_mut().is_some_and(|x| self.expr(x, false)) || self.stmts(b),
            Stmt::For(_, e, b, _) => self.expr(e, false) || self.stmts(b),
            Stmt::Match(e, arms, _) => {
                self.expr(e, false)
                    || arms.iter_mut().any(|a| a.guard.as_mut().is_some_and(|g| self.expr(g, true)) || self.stmts(&mut a.body))
            }
            Stmt::Return(e, _) => e.as_mut().is_some_and(|e| self.expr(e, false)),
            Stmt::Select(arms, _) => arms.iter_mut().any(|a| match a {
                SelArm::Recv(_, e, b) | SelArm::After(e, b) => self.expr(e, false) || self.stmts(b),
                SelArm::Send(x, y, b) => self.expr(x, false) || self.expr(y, false) || self.stmts(b),
                SelArm::Default(b) => self.stmts(b),
            }),
            Stmt::Arena(_, e, b, _) | Stmt::Device(_, e, b, _) => e.as_mut().is_some_and(|x| self.expr(x, false)) || self.stmts(b),
            Stmt::Comptime(b) => self.stmts(b),
            // `check`, `requires`, `ensures`: what a candidate is judged by
            Stmt::Check(_) => false,
            Stmt::Break(_) | Stmt::Continue(_) | Stmt::Pass => false,
        }
    }

    fn body(&mut self, b: &mut Body) -> bool {
        match b {
            Body::Arrow(e) => self.expr(e, false),
            Body::Block(s) => self.stmts(s),
        }
    }
}

/// Visit every editable expression of `e`, in the order of `visit_unit`.
pub(crate) fn visit_expr(e: &mut Expr, f: &mut dyn FnMut(&mut Expr, bool) -> bool) {
    let mut v = Visit { f };
    v.expr(e, false);
}

pub(crate) fn visit_unit(m: &mut Module, u: &Unit, f: &mut dyn FnMut(&mut Expr, bool) -> bool) {
    let mut v = Visit { f };
    match u {
        Unit::Fn(i) => {
            if let Item::Fn(d) = &mut m.items[*i] {
                v.body(&mut d.body);
            }
        }
        Unit::ImplFn(i, j) => {
            if let Item::Impl { members, .. } = &mut m.items[*i] {
                v.body(&mut members[*j].body);
            }
        }
        Unit::Step(i) => {
            if let Item::Step { guard, body, .. } = &mut m.items[*i] {
                let _ = guard.as_mut().is_some_and(|g| v.expr(g, true)) || v.body(body);
            }
        }
        Unit::Actor(i) => {
            if let Item::Actor { handlers, fns, .. } = &mut m.items[*i] {
                let _ = handlers.iter_mut().any(|h| h.guard.as_mut().is_some_and(|g| v.expr(g, true)) || v.body(&mut h.body))
                    || fns.iter_mut().any(|d| v.body(&mut d.body));
            }
        }
        Unit::Const(i) => {
            if let Item::Const { e, .. } = &mut m.items[*i] {
                v.expr(e, false);
            }
        }
        Unit::State(i) => {
            if let Item::State { fields, .. } = &mut m.items[*i] {
                let _ = fields.iter_mut().any(|f| f.init.as_mut().is_some_and(|x| v.expr(x, false)));
            }
        }
    }
}

/// Every (site, description, replacement) the generator `g` proposes in a
/// unit, in preorder.
pub(crate) type Gen<'g> = &'g dyn Fn(&Expr, bool) -> Vec<(String, Expr)>;

pub(crate) fn sites(m: &Module, u: &Unit, g: Gen) -> Vec<(usize, String, Expr)> {
    let mut mm = m.clone();
    let mut k = 0usize;
    let mut out = vec![];
    let mut f = |e: &mut Expr, c: bool| {
        for (d, r) in g(e, c) {
            out.push((k, d, r));
        }
        k += 1;
        false
    };
    visit_unit(&mut mm, u, &mut f);
    out
}

/// The module with site `site` of unit `u` replaced by `r`.
pub(crate) fn apply(m: &Module, u: &Unit, site: usize, r: &Expr) -> Module {
    let mut mm = m.clone();
    let mut k = 0usize;
    let mut f = |e: &mut Expr, _: bool| {
        if k == site {
            *e = r.clone();
            return true;
        }
        k += 1;
        false
    };
    visit_unit(&mut mm, u, &mut f);
    mm
}

// ---------- rung a: reuse ----------

fn fn_decl<'m>(m: &'m Module, u: &Unit) -> Option<&'m FnDecl> {
    match u {
        Unit::Fn(i) => match &m.items[*i] {
            Item::Fn(d) => Some(d),
            _ => None,
        },
        Unit::ImplFn(i, j) => match &m.items[*i] {
            Item::Impl { members, .. } => Some(&members[*j]),
            _ => None,
        },
        _ => None,
    }
}

fn fn_decl_mut<'m>(m: &'m mut Module, u: &Unit) -> Option<&'m mut FnDecl> {
    match u {
        Unit::Fn(i) => match &mut m.items[*i] {
            Item::Fn(d) => Some(d),
            _ => None,
        },
        Unit::ImplFn(i, j) => match &mut m.items[*i] {
            Item::Impl { members, .. } => Some(&mut members[*j]),
            _ => None,
        },
        _ => None,
    }
}

/// The name-free signature: generic arity, parameter types, result type.
fn sig_key(s: &FnSig) -> Option<String> {
    let mut ps = vec![];
    for p in &s.params {
        match p {
            Param::Named { ty, comptime, .. } => ps.push(format!("{}{}", if *comptime { "comptime " } else { "" }, crate::unparse::ty(ty))),
            _ => return None,
        }
    }
    let g: Vec<&str> = s.generics.iter().map(|g| g.name.as_str()).collect();
    Some(format!("[{}]({}) -> {}", g.join(","), ps.join(", "), s.ret.as_ref().map(crate::unparse::ty).unwrap_or_else(|| "()".into())))
}

fn param_names(s: &FnSig) -> Vec<String> {
    s.params.iter().filter_map(|p| if let Param::Named { name, .. } = p { Some(name.clone()) } else { None }).collect()
}

/// The name-free shape of one function (10.6), for deduplicating donors.
fn shape(d: &FnDecl) -> String {
    let mut d = d.clone();
    d.sig.name = "f".into();
    let one = Module { path: None, items: vec![Item::Fn(d)], pubs: BTreeSet::new(), v09_hygiene: vec![], x09: Default::default() };
    let t = unparse(&one);
    crate::nameless::nameless(&t).unwrap_or(t)
}

/// Rename the free uses of parameter names in a body.
fn rename_params(b: &mut Body, map: &BTreeMap<String, String>) {
    let mut f = |e: &mut Expr, _: bool| {
        if let Expr::Name(n, _) = e
            && let Some(t) = map.get(n)
        {
            *n = t.clone();
        }
        false
    };
    let mut v = Visit { f: &mut f };
    v.body(b);
}

fn rung_a(m: &Module, units: &[Unit], others: &[(String, String)], cx: &mut Ctx) -> Option<(String, String, String)> {
    // donors: the program's own module, its other modules, then the built-in library
    let mut donors: Vec<(String, FnDecl)> = vec![];
    let own_mod = m.path.as_ref().map(|q| q.join(".")).unwrap_or_default();
    for it in &m.items {
        if let Item::Fn(d) = it {
            donors.push((own_mod.clone(), d.clone()));
        }
    }
    for (p, s) in others {
        if let Ok(om) = crate::parse::parse(s) {
            let mname = om.path.as_ref().map(|q| q.join(".")).unwrap_or_else(|| p.clone());
            for it in &om.items {
                if let Item::Fn(d) = it {
                    donors.push((mname.clone(), d.clone()));
                }
            }
        }
    }
    for n in crate::stdlib::names() {
        if let Some(lm) = crate::stdlib::module(n) {
            for it in &lm.items {
                if let Item::Fn(d) = it {
                    donors.push((n.to_string(), d.clone()));
                }
            }
        }
    }
    for u in units {
        let Some(d) = fn_decl(m, u) else { continue };
        let Some(k) = sig_key(&d.sig) else { continue };
        let own = shape(d);
        let mut cands: Vec<&(String, FnDecl)> = donors.iter().filter(|(_, x)| sig_key(&x.sig).as_deref() == Some(k.as_str())).collect();
        // the same name first, then (module, name) order
        cands.sort_by_key(|(mo, x)| (x.sig.name != d.sig.name, mo.clone(), x.sig.name.clone()));
        let mut shapes: HashSet<String> = HashSet::new();
        shapes.insert(own);
        for (mo, x) in cands {
            if !shapes.insert(shape(x)) {
                continue;
            }
            let mut mm = m.clone();
            let t = fn_decl_mut(&mut mm, u).unwrap();
            let map: BTreeMap<String, String> = param_names(&x.sig).into_iter().zip(param_names(&t.sig)).filter(|(a, b)| a != b).collect();
            let mut body = x.body.clone();
            rename_params(&mut body, &map);
            t.body = body;
            let from = if mo.is_empty() { x.sig.name.clone() } else { format!("{mo}.{}", x.sig.name) };
            let edit = format!("the body of `{}` (line {}) becomes the body of `{from}`", unit_name(m, u), d.sig.line);
            let text = unparse(&mm);
            if let Some(t) = cx.judge("a", &unit_name(m, u), &edit, &text) {
                return Some(("a".into(), edit, t));
            }
        }
    }
    None
}

// ---------- rung b: lookup by type ----------

fn query(params: &[String], ret: &str) -> String {
    format!("({}) -> {}", params.join(", "), ret)
}

fn rung_b(m: &Module, units: &[Unit], file: &str, src: &str, others: &[(String, String)], cx: &mut Ctx) -> Option<(String, String, String)> {
    // the catalog of the program, its other modules and the library modules it uses
    let mut files = vec![(file.to_string(), src.to_string())];
    files.extend(others.iter().cloned());
    let own_mod = m.path.as_ref().map(|q| q.join(".")).unwrap_or_default();
    let mut alias: BTreeMap<String, String> = BTreeMap::new(); // module path -> alias
    for it in &m.items {
        if let Item::Use(p, al, _) = it {
            let s = p.join(".");
            alias.insert(s.clone(), al.clone().unwrap_or_else(|| p.last().cloned().unwrap_or_default()));
            if crate::stdlib::has(&s)
                && !others.iter().any(|(_, t)| crate::parse::parse(t).ok().and_then(|x| x.path).map(|q| q.join(".")) == Some(s.clone()))
                && let Some(t) = crate::stdlib::source(&s)
            {
                files.push((format!("{s}.rce"), t.to_string()));
            }
        }
    }
    let cat = crate::brick::index(&files);
    let entries: Vec<Value> = cat["entries"].as_array().cloned().unwrap_or_default();
    let ty_of = |module: &str, name: &str| -> Option<String> {
        entries.iter().find(|e| e["module"] == module && e["name"] == name).map(|e| {
            let ps: Vec<String> = e["params"].as_array().map(|a| a.iter().map(|p| p[1].as_str().unwrap_or("").to_string()).collect()).unwrap_or_default();
            query(&ps, e["ret"].as_str().unwrap_or("()"))
        })
    };
    // a visible callee for a catalog hit
    let callee = |module: &str, name: &str, line: usize| -> Option<Expr> {
        if module == own_mod {
            Some(Expr::Name(name.into(), line))
        } else {
            alias.get(module).map(|a| Expr::Member(Box::new(Expr::Name(a.clone(), line)), name.into(), line))
        }
    };
    let hits = |q: &str| -> Vec<(String, String)> {
        crate::brick::search(&cat, q).unwrap_or_default().into_iter().map(|(_, mo, n, _)| (mo, n)).collect()
    };
    for u in units {
        // (1) the body becomes a call of another function of its type
        if let Some(d) = fn_decl(m, u)
            && d.sig.generics.is_empty()
            && let Some(_) = sig_key(&d.sig)
        {
            let ps: Vec<String> = d.sig.params.iter().filter_map(|p| if let Param::Named { ty, .. } = p { Some(crate::unparse::ty(ty)) } else { None }).collect();
            let q = query(&ps, &d.sig.ret.as_ref().map(crate::unparse::ty).unwrap_or_else(|| "()".into()));
            for (mo, n) in hits(&q) {
                if mo == own_mod && n == d.sig.name {
                    continue;
                }
                let Some(c) = callee(&mo, &n, d.sig.line) else { continue };
                let args = param_names(&d.sig).into_iter().map(|p| Arg { label: None, e: Expr::Name(p, d.sig.line) }).collect();
                let call = Expr::Call(Box::new(c), args, d.sig.line, false);
                let mut mm = m.clone();
                fn_decl_mut(&mut mm, u).unwrap().body = Body::Block(vec![Stmt::Return(Some(call.clone()), d.sig.line)]);
                let edit = format!("the body of `{}` (line {}) becomes `return {}`", unit_name(m, u), d.sig.line, show(&call));
                let text = unparse(&mm);
                if let Some(t) = cx.judge("b", &unit_name(m, u), &edit, &text) {
                    return Some(("b".into(), edit, t));
                }
            }
        }
        // (2) the callee of a call becomes another function of its type
        let gen_ = |e: &Expr, _: bool| -> Vec<(String, Expr)> {
            let Expr::Call(f, a, l, v) = e else { return vec![] };
            let (mo, n) = match &**f {
                Expr::Name(n, _) => (own_mod.clone(), n.clone()),
                Expr::Member(x, n, _) => match &**x {
                    Expr::Name(al, _) => match alias.iter().find(|(_, a)| *a == al) {
                        Some((p, _)) => (p.clone(), n.clone()),
                        None => return vec![],
                    },
                    _ => return vec![],
                },
                _ => return vec![],
            };
            let Some(q) = ty_of(&mo, &n) else { return vec![] };
            let mut out = vec![];
            for (hm, hn) in hits(&q) {
                if hm == mo && hn == n {
                    continue;
                }
                if let Some(c) = callee(&hm, &hn, *l) {
                    let r = Expr::Call(Box::new(c), a.clone(), *l, *v);
                    out.push((format!("line {l}: `{}` becomes `{}`", show(e), show(&r)), r));
                }
            }
            out
        };
        for (site, d, r) in sites(m, u, &gen_) {
            let mm = apply(m, u, site, &r);
            let text = unparse(&mm);
            let edit = format!("in `{}`, {d}", unit_name(m, u));
            if let Some(t) = cx.judge("b", &unit_name(m, u), &edit, &text) {
                return Some(("b".into(), edit, t));
            }
        }
    }
    None
}

// ---------- rung c: altered rules ----------

/// The rewrites `rce fmt` writes (10.6) for the rules whose method is
/// altered (7.1): `alter` on the text of FILE, kept as `rce fmt` keeps it
/// (only when the canonical layout accepts the altered text). The candidate
/// is the canonical text of the altered program, so it is judged like every
/// other candidate; the edit lists the rewrites in the order `alter` wrote
/// them, each with the line `rce fmt` reports for it.
fn rung_c(src: &str, cx: &mut Ctx) -> Option<(String, String, String)> {
    let (altered, done) = crate::alter::alter(src);
    if done.is_empty() || altered == src || crate::cli::fmt_text(&altered).is_err() {
        return None;
    }
    let m = crate::parse::parse(&altered).ok()?;
    let text = unparse(&m);
    let what: Vec<String> = done.iter().map(|a| format!("line {}: {} {}", a.line, a.rule, a.what)).collect();
    let edit = format!("the rewrites of rce fmt: {}", what.join("; "));
    let unit = done.iter().map(|a| a.rule).collect::<BTreeSet<_>>().into_iter().collect::<Vec<_>>().join(", ");
    cx.judge("c", &format!("module ({unit})"), &edit, &text).map(|t| ("c".into(), edit, t))
}

// ---------- rung d: bounded enumeration ----------

const CLASSES: &[&[&str]] = &[&["+", "-", "*", "/", "%"], &["<", "<=", ">", ">="], &["==", "!="], &["and", "or"], &["&", "|", "^"], &["<<", ">>"]];

/// The small edits of one expression, in a fixed order.
pub fn small_edits(e: &Expr, cond: bool) -> Vec<(String, Expr)> {
    let mut out = vec![];
    let l = e.line();
    let mut push = |r: Expr| {
        out.push((format!("line {l}: `{}` becomes `{}`", show(e), show(&r)), r));
    };
    match e {
        Expr::Bin(op, a, b, _) => {
            if let Some(c) = CLASSES.iter().find(|c| c.contains(&op.as_str())) {
                for o in c.iter().filter(|o| **o != op) {
                    push(Expr::Bin(o.to_string(), a.clone(), b.clone(), l));
                }
            }
            if ["<", "<=", ">", ">="].contains(&op.as_str()) {
                push(Expr::Bin(op.clone(), b.clone(), a.clone(), l));
            }
        }
        Expr::Int(n, _) => {
            let mut seen = vec![*n];
            for v in [n + 1, n - 1, 0, 1] {
                if v >= 0 && !seen.contains(&v) {
                    seen.push(v);
                    push(Expr::Int(v, l));
                }
            }
        }
        Expr::Bool(b, _) => push(Expr::Bool(!b, l)),
        Expr::Call(f, a, _, v) if a.len() >= 2 && a.len() <= 4 => {
            for i in 0..a.len() {
                for j in i + 1..a.len() {
                    let mut a2 = a.clone();
                    let t = a2[i].e.clone();
                    a2[i].e = a2[j].e.clone();
                    a2[j].e = t;
                    push(Expr::Call(f.clone(), a2, l, *v));
                }
            }
        }
        _ => {}
    }
    if cond {
        match e {
            Expr::Not(x, _) => push((**x).clone()),
            Expr::Bool(..) => {}
            _ => push(Expr::Not(Box::new(e.clone()), l)),
        }
    }
    out
}

fn rung_d(m: &Module, units: &[Unit], bound: usize, cx: &mut Ctx) -> Option<(String, String, String)> {
    for u in units {
        for (site, d, r) in sites(m, u, &small_edits) {
            if cx.tried.get("d").copied().unwrap_or(0) >= bound {
                return None;
            }
            let mm = apply(m, u, site, &r);
            let text = unparse(&mm);
            let edit = format!("in `{}`, {d}", unit_name(m, u));
            if let Some(t) = cx.judge("d", &unit_name(m, u), &edit, &text) {
                return Some(("d".into(), edit, t));
            }
        }
    }
    None
}

// ---------- rung e: renames ----------

/// The names a pattern binds (a one-segment lowercase path), in order.
fn pat_binders(p: &Pat, out: &mut Vec<String>) {
    match p {
        Pat::Path(v) if v.len() == 1 && v[0] != "_" && v[0].starts_with(|c: char| c.is_ascii_lowercase() || c == '_') => {
            out.push(v[0].clone())
        }
        Pat::Ctor(_, fs) => fs.iter().for_each(|(_, q)| pat_binders(q, out)),
        Pat::Tuple(v) | Pat::Alt(v) => v.iter().for_each(|q| pat_binders(q, out)),
        _ => {}
    }
}

fn param_binders(ps: &[Param], out: &mut Vec<String>) {
    out.extend(ps.iter().filter_map(|p| if let Param::Named { name, .. } = p { Some(name.clone()) } else { None }));
}

/// A preorder walk over the names of one definition that keeps the value
/// names in scope: parameters, `let`, `var` and `for` bindings, pattern
/// bindings, the fields of a step's state or of an actor, and the module's
/// constants. `f(e, scope)` is called on every `Expr::Name` (innermost name
/// last in `scope`) and returns true to stop the walk. Like `Visit`, it never
/// enters `check`, `requires` or `ensures` statements.
struct Scoped<'f> {
    scope: Vec<String>,
    f: &'f mut dyn FnMut(&mut Expr, &[String]) -> bool,
}

impl Scoped<'_> {
    fn with<T>(&mut self, names: Vec<String>, g: impl FnOnce(&mut Self) -> T) -> T {
        let n = self.scope.len();
        self.scope.extend(names);
        let r = g(self);
        self.scope.truncate(n);
        r
    }

    fn expr(&mut self, e: &mut Expr) -> bool {
        use Expr as E;
        match e {
            E::Name(..) => (self.f)(e, &self.scope),
            E::Tuple(v, _) | E::List(v, _) => v.iter_mut().any(|x| self.expr(x)),
            E::ListRepeat(a, b, _) => self.expr(a) || self.expr(b),
            E::ListComp(a, p, s, c, _) => {
                if self.expr(s) {
                    return true;
                }
                let mut bs = vec![];
                pat_binders(p, &mut bs);
                self.with(bs, |w| w.expr(a) || c.as_mut().is_some_and(|c| w.expr(c)))
            }
            E::Map(v, _) => v.iter_mut().any(|(k, x)| self.expr(k) || self.expr(x)),
            E::ChanNew(_, c, _) => c.as_mut().is_some_and(|c| self.expr(c)),
            E::Call(f, a, _, _) => self.expr(f) || a.iter_mut().any(|x| self.expr(&mut x.e)),
            E::Index(x, ix, _) => self.expr(x) || ix.iter_mut().any(|i| self.expr(i)),
            E::Slice(x, lo, hi, _) => {
                self.expr(x) || lo.as_mut().is_some_and(|l| self.expr(l)) || hi.as_mut().is_some_and(|h| self.expr(h))
            }
            E::Member(x, _, _)
            | E::TupleMember(x, _, _)
            | E::Neg(x, _)
            | E::BitNot(x, _)
            | E::Borrow(_, x, _)
            | E::Recv(x, _)
            | E::Comptime(x, _)
            | E::Cast(x, _, _)
            | E::Not(x, _) => self.expr(x),
            E::Bin(_, a, b, _) => self.expr(a) || self.expr(b),
            E::Catch(a, n, h, _) => self.expr(a) || self.with(n.iter().cloned().collect(), |w| w.expr(h)),
            E::Try(a, el, _) => self.expr(a) || el.as_mut().is_some_and(|x| self.expr(x)),
            E::If(c, a, b, _) => self.expr(c) || self.expr(a) || self.expr(b),
            E::Lambda(ps, b, _) => {
                let mut bs = vec![];
                param_binders(ps, &mut bs);
                self.with(bs, |w| w.expr(b))
            }
            _ => false,
        }
    }

    fn block(&mut self, b: &mut [Stmt]) -> bool {
        self.with(vec![], |w| b.iter_mut().any(|s| w.stmt(s)))
    }

    fn stmt(&mut self, s: &mut Stmt) -> bool {
        match s {
            Stmt::Expr(e) | Stmt::Emit(e) | Stmt::Fail(e) | Stmt::Spawn(e) | Stmt::Yield(e, _) => self.expr(e),
            Stmt::Assign(_, a, b) | Stmt::Send(a, b) => self.expr(a) || self.expr(b),
            Stmt::Let(p, _, e, _) => {
                if self.expr(e) {
                    return true;
                }
                pat_binders(p, &mut self.scope);
                false
            }
            Stmt::Var(n, _, e, _) => {
                if self.expr(e) {
                    return true;
                }
                self.scope.push(n.clone());
                false
            }
            Stmt::If(arms, el) => arms.iter_mut().any(|(c, b)| self.expr(c) || self.block(b)) || el.as_mut().is_some_and(|b| self.block(b)),
            Stmt::While(c, bd, b) => self.expr(c) || bd.as_mut().is_some_and(|x| self.expr(x)) || self.block(b),
            Stmt::Loop(bd, b) => bd.as_mut().is_some_and(|x| self.expr(x)) || self.block(b),
            Stmt::For(p, e, b, _) => {
                if self.expr(e) {
                    return true;
                }
                let mut bs = vec![];
                pat_binders(p, &mut bs);
                self.with(bs, |w| w.block(b))
            }
            Stmt::Match(e, arms, _) => {
                self.expr(e)
                    || arms.iter_mut().any(|a| {
                        let mut bs = vec![];
                        pat_binders(&a.pat, &mut bs);
                        self.with(bs, |w| a.guard.as_mut().is_some_and(|g| w.expr(g)) || w.block(&mut a.body))
                    })
            }
            Stmt::Return(e, _) => e.as_mut().is_some_and(|e| self.expr(e)),
            Stmt::Select(arms, _) => arms.iter_mut().any(|a| match a {
                SelArm::Recv(p, e, b) => {
                    if self.expr(e) {
                        return true;
                    }
                    let mut bs = vec![];
                    pat_binders(p, &mut bs);
                    self.with(bs, |w| w.block(b))
                }
                SelArm::After(e, b) => self.expr(e) || self.block(b),
                SelArm::Send(x, y, b) => self.expr(x) || self.expr(y) || self.block(b),
                SelArm::Default(b) => self.block(b),
            }),
            Stmt::Arena(_, e, b, _) | Stmt::Device(_, e, b, _) => e.as_mut().is_some_and(|x| self.expr(x)) || self.block(b),
            Stmt::Comptime(b) => self.block(b),
            // `check`, `requires`, `ensures`: what a candidate is judged by
            Stmt::Check(_) => false,
            Stmt::Break(_) | Stmt::Continue(_) | Stmt::Pass => false,
        }
    }

    fn body(&mut self, b: &mut Body) -> bool {
        match b {
            Body::Arrow(e) => self.expr(e),
            Body::Block(s) => self.block(s),
        }
    }
}

/// The fields of a state or actor, by name.
fn region_fields(m: &Module, region: &str) -> Vec<String> {
    m.items
        .iter()
        .find_map(|it| match it {
            Item::State { name, fields, .. } | Item::Actor { name, fields, .. } if name == region => Some(fields.iter().map(|f| f.name.clone()).collect()),
            _ => None,
        })
        .unwrap_or_default()
}

/// Walk the names of one unit with their scopes (`Scoped`).
fn walk_names(m: &mut Module, u: &Unit, f: &mut dyn FnMut(&mut Expr, &[String]) -> bool) {
    let consts: Vec<String> = m.items.iter().filter_map(|it| if let Item::Const { name, .. } = it { Some(name.clone()) } else { None }).collect();
    let fields = match u {
        Unit::Step(i) => match &m.items[*i] {
            Item::Step { region, .. } => region_fields(m, region),
            _ => vec![],
        },
        Unit::Actor(i) => match &m.items[*i] {
            Item::Actor { fields, .. } => fields.iter().map(|f| f.name.clone()).collect(),
            _ => vec![],
        },
        _ => vec![],
    };
    let mut w = Scoped { scope: consts, f };
    match u {
        Unit::Fn(_) | Unit::ImplFn(..) => {
            if let Some(d) = fn_decl_mut(m, u) {
                let mut bs = vec![];
                param_binders(&d.sig.params, &mut bs);
                w.with(bs, |w| w.body(&mut d.body));
            }
        }
        Unit::Step(i) => {
            if let Item::Step { pat, guard, body, .. } = &mut m.items[*i] {
                let mut bs = fields;
                pat_binders(pat, &mut bs);
                w.with(bs, |w| guard.as_mut().is_some_and(|g| w.expr(g)) || w.body(body));
            }
        }
        Unit::Actor(i) => {
            if let Item::Actor { handlers, fns, .. } = &mut m.items[*i] {
                w.with(fields, |w| {
                    handlers.iter_mut().any(|h| {
                        let mut bs = vec![];
                        pat_binders(&h.pat, &mut bs);
                        w.with(bs, |w| h.guard.as_mut().is_some_and(|g| w.expr(g)) || w.body(&mut h.body))
                    }) || fns.iter_mut().any(|d| {
                        let mut bs = vec![];
                        param_binders(&d.sig.params, &mut bs);
                        w.with(bs, |w| w.body(&mut d.body))
                    })
                });
            }
        }
        Unit::Const(i) => {
            if let Item::Const { e, .. } = &mut m.items[*i] {
                w.expr(e);
            }
        }
        Unit::State(i) => {
            if let Item::State { fields, .. } = &mut m.items[*i] {
                let _ = fields.iter_mut().any(|f| f.init.as_mut().is_some_and(|x| w.expr(x)));
            }
        }
    }
}

/// Every name use of a unit: (index, name, line, the value names in scope
/// there, innermost first, each once).
fn name_sites(m: &Module, u: &Unit) -> Vec<(usize, String, usize, Vec<String>)> {
    let mut mm = m.clone();
    let mut out = vec![];
    let mut f = |e: &mut Expr, scope: &[String]| {
        if let Expr::Name(n, l) = e {
            let mut seen = BTreeSet::new();
            let s: Vec<String> = scope.iter().rev().filter(|x| seen.insert(x.as_str())).cloned().collect();
            out.push((out.len(), n.clone(), *l, s));
        }
        false
    };
    walk_names(&mut mm, u, &mut f);
    out
}

/// The module with name use `site` of unit `u` renamed to `to`.
fn rename_use(m: &Module, u: &Unit, site: usize, to: &str) -> Module {
    let mut mm = m.clone();
    let mut k = 0usize;
    let mut f = |e: &mut Expr, _: &[String]| {
        if let Expr::Name(n, _) = e {
            if k == site {
                *n = to.to_string();
                return true;
            }
            k += 1;
        }
        false
    };
    walk_names(&mut mm, u, &mut f);
    mm
}

/// The declarations of value names in a unit that a rename can change: the
/// parameters of a function and its `let`, `var` and `for` bindings of one
/// name, as (name, line), in preorder.
fn binder_slots(b: &mut [Stmt], f: &mut dyn FnMut(&mut String, usize) -> bool) -> bool {
    for s in b {
        let stop = match s {
            Stmt::Let(Pat::Path(v), _, _, l) if v.len() == 1 => f(&mut v[0], *l),
            Stmt::Var(n, _, _, l) => f(n, *l),
            Stmt::For(p, _, body, l) => {
                (if let Pat::Path(v) = p
                    && v.len() == 1
                {
                    f(&mut v[0], *l)
                } else {
                    false
                }) || binder_slots(body, f)
            }
            Stmt::If(arms, el) => arms.iter_mut().any(|(_, b)| binder_slots(b, f)) || el.as_mut().is_some_and(|b| binder_slots(b, f)),
            Stmt::While(_, _, b) | Stmt::Loop(_, b) | Stmt::Arena(_, _, b, _) | Stmt::Device(_, _, b, _) | Stmt::Comptime(b) => binder_slots(b, f),
            Stmt::Match(_, arms, _) => arms.iter_mut().any(|a| binder_slots(&mut a.body, f)),
            _ => false,
        };
        if stop {
            return true;
        }
    }
    false
}

fn walk_binders(m: &mut Module, u: &Unit, f: &mut dyn FnMut(&mut String, usize) -> bool) {
    if let Some(d) = fn_decl_mut(m, u) {
        let line = d.sig.line;
        for p in d.sig.params.iter_mut() {
            if let Param::Named { name, .. } = p
                && f(name, line)
            {
                return;
            }
        }
        if let Body::Block(b) = &mut d.body {
            binder_slots(b, f);
        }
    }
}

/// The identifiers a report names between backquotes, in order, each once.
fn report_names(err: &str) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    for (i, part) in err.split('`').enumerate() {
        if i % 2 == 1
            && !part.is_empty()
            && part.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
            && part.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            && !out.iter().any(|x| x == part)
        {
            out.push(part.to_string());
        }
    }
    out
}

/// The edit distance of two names (Levenshtein, by bytes: names are ASCII).
fn distance(a: &str, b: &str) -> usize {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut prev = row[0];
        row[0] = i;
        for j in 1..=b.len() {
            let t = row[j];
            row[j] = (row[j] + 1).min(row[j - 1] + 1).min(prev + usize::from(a[i - 1] != b[j - 1]));
            prev = t;
        }
    }
    row[b.len()]
}

/// The name-free projection (10.6) of a module's canonical text.
fn projection(text: &str) -> Option<String> {
    crate::nameless::nameless(text).ok()
}

/// What `known GOOD` gives rung e: the canonical text of GOOD and its
/// name-free projection.
pub struct Known {
    pub file: String,
    pub canon: String,
    pub proj: String,
    /// the SHA-256 of GOOD's bytes (RF-4)
    pub sha256: String,
}

/// Read `known GOOD`: its canonical text and projection; an error when GOOD
/// does not parse.
pub fn known(file: &str, src: &str) -> Result<Known, String> {
    let m = crate::parse::parse(src).map_err(|e| format!("{file}: {e}"))?;
    let canon = unparse(&m);
    let proj = projection(&canon).ok_or_else(|| format!("{file}: no name-free projection"))?;
    Ok(Known { file: file.to_string(), canon, proj, sha256: crate::manifest::sha256_hex(src.as_bytes()) })
}

/// The identifiers of a canonical text outside strings and comments, as
/// byte spans, and the text between them.
fn ident_spans(t: &str) -> (Vec<(usize, usize)>, Vec<&str>) {
    let b = t.as_bytes();
    let (mut spans, mut gaps) = (vec![], vec![]);
    let (mut i, mut last) = (0, 0);
    while i < b.len() {
        let c = b[i];
        if c == b'"' {
            i += 1;
            while i < b.len() && b[i] != b'"' {
                i += if b[i] == b'\\' { 2 } else { 1 };
            }
            i += 1;
        } else if c == b'#' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if c.is_ascii_digit() {
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || (b[i] == b'.' && b.get(i + 1).is_some_and(u8::is_ascii_digit))) {
                i += 1;
            }
        } else if c.is_ascii_alphabetic() || c == b'_' {
            let s = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            gaps.push(&t[last..s]);
            spans.push((s, i));
            last = i;
        } else {
            i += 1;
        }
    }
    gaps.push(&t[last.min(t.len())..]);
    (spans, gaps)
}

/// The renaming onto GOOD (part of rung e, judged before rung a): when the
/// canonical texts of FILE and GOOD differ in identifiers only (none of
/// them a keyword), the renaming is read from the two texts, with no
/// search. Each name of GOOD becomes the name FILE writes most often in its
/// place (on a tie, the first); every identifier of FILE that this renaming
/// does not give is changed to the one it gives, so the first candidate
/// keeps FILE's names and has GOOD's name-free projection (it is judged only
/// when it has). The second candidate is the canonical text of GOOD, the
/// renaming of every name back; it is the one candidate when the
/// projections of FILE and GOOD are equal.
fn rung_e_exact(base: &str, k: &Known, cx: &mut Ctx) -> Option<(String, String, String)> {
    if base == k.canon {
        return None;
    }
    let (fs, fg) = ident_spans(base);
    let (gs, gg) = ident_spans(&k.canon);
    if fs.len() != gs.len() || fg != gg {
        return None;
    }
    let reserved = |w: &str| crate::parse::KEYWORDS.contains(&w) || crate::parse::DRAFT06_KEYWORDS.contains(&w);
    let pairs: Vec<(&str, &str, usize)> =
        fs.iter().zip(&gs).map(|(f, g)| (&base[f.0..f.1], &k.canon[g.0..g.1], base[..f.0].matches('\n').count() + 1)).collect();
    if pairs.iter().any(|(f, g, _)| f != g && (reserved(f) || reserved(g))) {
        return None;
    }
    // each name of GOOD: the name FILE writes most often in its place
    let mut counts: BTreeMap<&str, Vec<(&str, usize, usize)>> = BTreeMap::new(); // g -> (f, count, first)
    for (i, (f, g, _)) in pairs.iter().enumerate() {
        let v = counts.entry(g).or_default();
        match v.iter_mut().find(|x| x.0 == *f) {
            Some(x) => x.1 += 1,
            None => v.push((f, 1, i)),
        }
    }
    let to: BTreeMap<&str, &str> = counts.iter().map(|(g, v)| (*g, v.iter().max_by_key(|x| (x.1, std::cmp::Reverse(x.2))).unwrap().0)).collect();
    let mut text = String::new();
    let mut changed: Vec<String> = vec![];
    for (i, (f, g, line)) in pairs.iter().enumerate() {
        text.push_str(fg[i]);
        let t = to[g];
        if t != *f {
            changed.push(format!("line {line}: `{f}` becomes `{t}`"));
        }
        text.push_str(t);
    }
    text.push_str(fg[pairs.len()]);
    let one_to_one = to.values().collect::<BTreeSet<_>>().len() == to.len();
    if one_to_one && !changed.is_empty() && projection(&text).as_deref() == Some(k.proj.as_str()) {
        let edit = format!("the renaming onto {} that keeps the names of FILE: {}", k.file, changed.join("; "));
        if let Some(t) = cx.judge("e", "module (renaming)", &edit, &text) {
            return Some(("e".into(), edit, t));
        }
    }
    let mut back: Vec<String> = vec![];
    for (f, g, _) in &pairs {
        let p = format!("`{f}` becomes `{g}`");
        if f != g && !back.contains(&p) {
            back.push(p);
        }
    }
    let edit = format!("the renaming onto {}: {}", k.file, back.join(", "));
    cx.judge("e", "module (renaming)", &edit, &k.canon).map(|t| ("e".into(), edit, t))
}

/// Rung e, the search: one name changed to another. For each definition in
/// the ladder's order, (1) one use of a value name (or of a name the report
/// names) becomes another value name in scope there, and (2) one binding of
/// a value name becomes a name the report names. Uses on a line the report
/// names come first; the replacing names are ordered by whether the report
/// names them, by edit distance to the name they replace, then innermost
/// first. A use renamed to another binding changes the name-free projection
/// (that is how the checker sees it), so no candidate here keeps it; one
/// with GOOD's projection would differ from GOOD in names only, which the
/// renaming onto GOOD has already judged.
fn rung_e(m: &Module, units: &[Unit], err: &str, bound: usize, cx: &mut Ctx) -> Option<(String, String, String)> {
    let named = report_names(err);
    let lines = regex_lines(err);
    let rank = |c: &str, n: &str, i: usize| (!named.iter().any(|x| x == c), distance(c, n), i, c.to_string());
    // (unit, edit, how to make the candidate), in order; a candidate's text is
    // made only when it is judged
    enum How<'u> {
        Use(&'u Unit, usize, String),
        Bind(&'u Unit, usize, String),
    }
    let make = |h: &How| -> String {
        match h {
            How::Use(u, site, to) => unparse(&rename_use(m, u, *site, to)),
            How::Bind(u, j, to) => {
                let mut mm = m.clone();
                let mut k = 0usize;
                walk_binders(&mut mm, u, &mut |n, _| {
                    if k == *j {
                        *n = to.clone();
                        return true;
                    }
                    k += 1;
                    false
                });
                unparse(&mm)
            }
        }
    };
    let mut cands: Vec<(String, String, How)> = vec![];
    for u in units {
        let un = unit_name(m, u);
        let mut sites = name_sites(m, u);
        sites.retain(|(_, n, _, s)| s.contains(n) || named.contains(n));
        sites.sort_by_key(|(k, _, l, _)| (!lines.contains(l), *k));
        for (site, n, l, scope) in &sites {
            // which use of `n` on its line, when there is more than one
            let same: Vec<usize> = sites.iter().filter(|x| x.1 == *n && x.2 == *l).map(|x| x.0).collect();
            let which = if same.len() > 1 {
                format!("use {} of `{n}` on the line", same.iter().filter(|k| **k <= *site).count())
            } else {
                format!("`{n}`")
            };
            let mut tos: Vec<(usize, &String)> = scope.iter().enumerate().filter(|(_, c)| *c != n).collect();
            tos.sort_by_key(|(i, c)| rank(c, n, *i));
            for (_, to) in tos {
                cands.push((un.clone(), format!("in `{un}`, line {l}: {which} becomes `{to}`"), How::Use(u, *site, to.clone())));
            }
        }
        // a binding renamed to a name the report names that the unit does not bind
        let mut mm = m.clone();
        let mut binders: Vec<(String, usize)> = vec![];
        walk_binders(&mut mm, u, &mut |n, l| {
            binders.push((n.clone(), l));
            false
        });
        for (j, (b, l)) in binders.iter().enumerate() {
            for to in named.iter().filter(|x| *x != b && !binders.iter().any(|(y, _)| y == *x)) {
                cands.push((un.clone(), format!("in `{un}`, line {l}: the binding `{b}` becomes `{to}`"), How::Bind(u, j, to.clone())));
            }
        }
    }
    for (u, edit, how) in cands {
        if cx.tried.get("e").copied().unwrap_or(0) >= bound {
            return None;
        }
        let text = make(&how);
        if let Some(t) = cx.judge("e", &u, &edit, &text) {
            return Some(("e".into(), edit, t));
        }
    }
    None
}

// ---------- rung f: outside proposals ----------

/// The record of one proposal (RF-4): its place in the file, its origin,
/// its digest and the verdict.
fn record(i: usize, p: &crate::proposals::Proposal, verdict: &str) -> Value {
    json!({"index": i + 1, "origin": p.origin, "sha256": p.sha256, "verdict": verdict, "admitted": false, "observed": false})
}

/// The text of one item, as the source printer writes it alone.
fn item_text(it: &Item) -> String {
    let mm = Module { path: None, items: vec![it.clone()], pubs: Default::default(), v09_hygiene: vec![], x09: Default::default() };
    unparse(&mm)
}

/// The statement blocks of a definition.
fn unit_blocks<'m>(m: &'m Module, u: &Unit) -> Vec<&'m [Stmt]> {
    let of = |b: &'m Body| match b {
        Body::Block(s) => Some(s.as_slice()),
        Body::Arrow(_) => None,
    };
    match (u, &m.items[match u {
        Unit::Fn(i) | Unit::ImplFn(i, _) | Unit::Step(i) | Unit::Actor(i) | Unit::Const(i) | Unit::State(i) => *i,
    }]) {
        (Unit::Fn(_), Item::Fn(d)) => of(&d.body).into_iter().collect(),
        (Unit::ImplFn(_, j), Item::Impl { members, .. }) => of(&members[*j].body).into_iter().collect(),
        (Unit::Step(_), Item::Step { body, .. }) => of(body).into_iter().collect(),
        (Unit::Actor(_), Item::Actor { handlers, fns, .. }) => handlers.iter().map(|h| &h.body).chain(fns.iter().map(|d| &d.body)).filter_map(of).collect(),
        _ => vec![],
    }
}

/// What a candidate is judged by (RF-2): the module it names, and the
/// canonical texts of its intent, example, check and law items and of the
/// `check`, `requires` and `ensures` statements of each definition (named
/// by the definition), sorted, so that a proposal may move a definition but
/// not change what judges it.
fn obligations(m: &Module) -> (Option<Vec<String>>, Vec<String>) {
    let check = |e: &Expr| format!("check {}", show(e));
    let mut v: Vec<String> = vec![];
    for it in &m.items {
        match it {
            Item::Intent(what, xs, cs) => {
                v.push(format!("intent {}", crate::unparse::text(what)));
                v.extend(xs.iter().map(crate::unparse::example));
                v.extend(cs.iter().map(check));
            }
            Item::Example(x) => v.push(crate::unparse::example(x)),
            Item::Check(e) => v.push(check(e)),
            Item::Law(_) => v.push(item_text(it)),
            _ => {}
        }
    }
    for u in units(m) {
        let name = unit_name(m, &u);
        for b in unit_blocks(m, &u) {
            crate::check::walk_stmts(b, &mut |s| {
                if let Stmt::Check(e) = s {
                    v.push(format!("{name}: {}", show(e)));
                }
            });
        }
    }
    v.sort();
    (m.path.clone(), v)
}

/// Why a proposal does not keep FILE's obligations (RF-2), or None.
fn kept(want: &(Option<Vec<String>>, Vec<String>), m: &Module) -> Option<String> {
    let have = obligations(m);
    if have.0 != want.0 {
        return Some("the proposal names another module than FILE".into());
    }
    if have.1 == want.1 {
        return None;
    }
    let first = |t: &str| t.lines().find(|l| !l.starts_with("# rce-version:")).unwrap_or("").trim().chars().take(80).collect::<String>();
    let mut left: Vec<&String> = have.1.iter().collect();
    let mut removed = vec![];
    for w in &want.1 {
        match left.iter().position(|h| *h == w) {
            Some(k) => {
                left.remove(k);
            }
            None => removed.push(format!("`{}`", first(w))),
        }
    }
    let added: Vec<String> = left.iter().map(|h| format!("`{}`", first(h))).collect();
    let mut why = "the proposal does not keep what FILE is judged by:".to_string();
    if !removed.is_empty() {
        why.push_str(&format!(" removed {}", removed.join(", ")));
    }
    if !added.is_empty() {
        why.push_str(&format!("{} added {}", if removed.is_empty() { "" } else { ";" }, added.join(", ")));
    }
    Some(why)
}

/// How an admitted proposal compares with GOOD (RF-4): `restored` when the
/// program hash of 10.8 is GOOD's; else, by `rce equiv` (10.11) with the
/// state bound of `rce mutants`, `behaviour-equal`, `contradicts` (with the
/// event sequence after which the outputs differ, when there is one) or
/// `undecided`; `not compared` when a program has no step.
fn against(text: &str, k: &Known) -> Value {
    let (Ok(g), Ok(p)) = (crate::parse::parse(&k.canon), crate::parse::parse(text)) else {
        return json!({"result": "not compared", "why": "a text does not parse"});
    };
    if crate::apidiff::program_hash(&[g]) == crate::apidiff::program_hash(&[p]) {
        return json!({"result": "restored"});
    }
    let (Ok(g), Ok(p)) = (crate::program(&k.canon), crate::program(text)) else {
        return json!({"result": "not compared", "why": "a program does not link"});
    };
    let steps = |m: &Module| m.items.iter().any(|i| matches!(i, Item::Step { .. } | Item::Actor { .. }));
    if !steps(&g) || !steps(&p) {
        return json!({"result": "not compared", "why": "a program with no step: rce equiv compares the steps' behaviour (6.10)"});
    }
    let fuel = crate::mutants::EQUIV_FUEL;
    match crate::dim::equiv(&g, &p, fuel) {
        Ok(e) if e.equivalent && e.exact => json!({"result": "behaviour-equal"}),
        Ok(e) if e.equivalent => json!({"result": "undecided", "why": format!("rce equiv: equivalent only to the explored bound ({fuel} states)")}),
        Ok(e) if e.undecided => json!({"result": "undecided", "why": format!("rce equiv: undecided within {fuel} states")}),
        Ok(e) => match e.witness {
            Some((seq, a, b)) => {
                let why = format!("after {}: GOOD {}, the proposal {}", seq.join(", "), crate::mutants::output(&a), crate::mutants::output(&b));
                json!({"result": "contradicts", "after": seq, "why": why})
            }
            None if !e.only_a.is_empty() || !e.only_b.is_empty() => json!({"result": "contradicts", "why": "rce equiv: the proposal and GOOD explore other event instances"}),
            None => json!({"result": "contradicts", "why": "rce equiv: different, with no finite witness in the explored graph"}),
        },
        Err(e) => json!({"result": "not compared", "why": format!("rce equiv: {e}")}),
    }
}

/// Rung f (RF-2, RF-3): each proposal, in the order of the file, whose text
/// parses, names FILE's module and keeps FILE's obligations is judged as its
/// canonical text by the gate of every rung; a candidate already judged is
/// judged once, and at most `bound` proposals are judged. In `observe`, every
/// proposal is judged and each admitted one is observed; nothing closes. In
/// `auto`, the first admitted proposal closes the program.
fn rung_f(base: &str, m: &Module, o: &Outside, known: Option<&Known>, bound: usize, cx: &mut Ctx, rf: &mut RungF) -> Option<(String, String, String)> {
    rf.ran = true;
    let want = obligations(m);
    let mut mine: BTreeMap<String, usize> = BTreeMap::new();
    let mut found: Option<(String, String, usize)> = None;
    for (i, p) in o.proposals.list.iter().enumerate() {
        let n = i + 1;
        let unit = format!("module (proposal {n})");
        let edit = format!("proposal {n} (origin `{}`, sha256 {})", p.origin, p.sha256);
        let mut rec = record(i, p, "");
        let verdict = if let Some((_, _, k)) = &found {
            format!("not judged: proposal {k} closed the program")
        } else if rf.judged >= bound {
            format!("not judged: rung f judged its bound of {bound} proposals")
        } else {
            match crate::parse::parse(&p.text) {
                Err(e) => {
                    rf.judged += 1;
                    let v = format!("requires change: {}", e.chars().take(200).collect::<String>());
                    cx.refuse(&unit, &edit, &v);
                    v
                }
                Ok(pm) => {
                    let text = unparse(&pm);
                    if text == base {
                        "not judged: the proposal is FILE itself, in its canonical text".to_string()
                    } else if let Some(k) = mine.get(&text) {
                        format!("not judged: the candidate of proposal {k}, judged once")
                    } else if cx.seen.contains(&text) {
                        "not judged: an earlier rung judged the same candidate and did not admit it".to_string()
                    } else if let Some(why) = kept(&want, &pm) {
                        rf.judged += 1;
                        mine.insert(text, n);
                        let v = format!("requires change: RF-2 {why}");
                        cx.refuse(&unit, &edit, &v);
                        v
                    } else {
                        rf.judged += 1;
                        mine.insert(text.clone(), n);
                        let (v, mutation) = cx.judge_measured(&unit, &edit, &text);
                        if v == "admitted" {
                            rf.admitted += 1;
                            rec["admitted"] = json!(true);
                            rec["diff"] = json!(diff(base, &text));
                            rec["mutation"] = mutation.unwrap_or(Value::Null);
                            if let Some(k) = known {
                                rec["known"] = against(&text, k);
                            }
                            if rf.mode == Mode::Auto {
                                found = Some((edit.clone(), text, n));
                            } else {
                                rf.observed += 1;
                                rec["observed"] = json!(true);
                            }
                        }
                        v
                    }
                }
            }
        };
        rec["verdict"] = json!(verdict);
        rf.records.push(rec);
    }
    found.map(|(e, t, _)| ("f".to_string(), e, t))
}

/// The report `rce repair` prints.
pub fn report(file: &str, o: &Outcome) -> String {
    let mut s = String::new();
    let rung = |r: &str| match r {
        "a" => "a reuse",
        "b" => "b lookup by type",
        "c" => "c altered rules",
        "d" => "d bounded enumeration",
        "e" => "e renames",
        "f" => "f outside proposal (stochastic)",
        "none" => "none (nothing to repair)",
        _ => "?",
    };
    match &o.rung {
        Some(r) => s.push_str(&format!("rung      {}\n", rung(r))),
        None => s.push_str("rung      not closed\n"),
    }
    if let Some(e) = &o.edit {
        s.push_str(&format!("edit      {e}\n"));
    }
    for (i, l) in o.diff.iter().enumerate() {
        s.push_str(&format!("{}{l}\n", if i == 0 { "diff      " } else { "          " }));
    }
    let t: Vec<String> = o.tried.iter().map(|(k, v)| format!("{k} {v}")).collect();
    if !t.is_empty() {
        s.push_str(&format!("tried     {} (candidates judged by the checker; 0 model calls)\n", t.join(", ")));
    }
    if let Some(p) = &o.proposals {
        s.push_str(&format!(
            "proposals {} given, {} judged, {} admitted, {} observed (rung f {}; sha256 {})\n",
            p.given,
            p.judged,
            p.admitted,
            p.observed,
            p.mode.text(),
            p.sha256
        ));
        for r in p.records.iter().filter(|r| r["observed"] == true) {
            let mm = &r["mutation"];
            s.push_str(&format!(
                "observed  proposal {} (origin `{}`, sha256 {}): admitted; W-MUT-1 measured {} of {} obligations; rung f is observe, so nothing closes\n",
                r["index"],
                r["origin"].as_str().unwrap_or(""),
                r["sha256"].as_str().unwrap_or(""),
                mm["measured"],
                mm["obligations"]
            ));
            for l in r["diff"].as_array().into_iter().flatten() {
                s.push_str(&format!("          {}\n", l.as_str().unwrap_or("")));
            }
            if let Some(k) = r["known"].as_object() {
                let why = k.get("why").and_then(Value::as_str).map(|w| format!(": {w}")).unwrap_or_default();
                s.push_str(&format!("          against GOOD: {}{why}\n", k["result"].as_str().unwrap_or("")));
            }
        }
    }
    for k in &o.skipped {
        s.push_str(&format!("skipped   {k}\n"));
    }
    s.push_str(&format!("verdict   {}\n", o.verdict));
    s.push_str(&format!("{} {file}\n", if o.closed { "closed" } else { "not closed" }));
    s
}

/// The JSON report of `rce repair json` (format `rce-repair/1`, RF-4): the
/// canonical JSON of 6.1.1.
pub fn report_json(file: &str, o: &Outcome) -> String {
    let proposals = o.proposals.as_ref().map(|p| {
        json!({
            "mode": p.mode.text(),
            "sha256": p.sha256,
            "ran": p.ran,
            "given": p.given,
            "judged": p.judged,
            "admitted": p.admitted,
            "observed": p.observed,
            "records": p.records,
        })
    });
    let v = json!({
        "format": "rce-repair/1",
        "file": file,
        "closed": o.closed,
        "rung": o.rung,
        "edit": o.edit,
        "diff": o.diff,
        "tried": o.tried,
        "model_calls": 0,
        "skipped": o.skipped,
        "verdict": o.verdict,
        "repaired": o.repaired,
        "canonical_same_report": o.canonical_same_report,
        "attempts": o.attempts,
        "bound": o.bound,
        "inputs": o.inputs,
        "proposals": proposals,
        "stochastic": o.stochastic,
    });
    crate::interp::canon_json(&v)
}
