//! Static rules of draft 0.6b (Appendix C, book/14-draft06-concurrency.md C.11): workgroups
//! (W-GRP-1..3), session types (W-SES-1..3), structured spawn (W-SCOPE-1),
//! cross-actor transactions (W-ATOM-1..2), supervisors (W-SUP-1..2), nodes
//! (W-NODE-1), interrupts and step deadlines (W-INTR-1, W-RT-1), LVars
//! (W-LVAR-1) and `impl Merge` (W-CRDT-1, checked by the admission gate in
//! `conc::crdt_gate`).
use crate::ast::*;
use crate::check::{stmt_exprs, walk_expr, walk_stmts, Checker, Cx};
use crate::types::T;
use std::collections::{BTreeSet, HashMap};

fn d<X: std::fmt::Display>(line: usize, rule: &str, m: X) -> String {
    format!("line {line}: {rule} {m}")
}

#[derive(Default)]
pub(crate) struct State06b {
    pub protocols: HashMap<String, Vec<ProtoStep>>,
    /// payload type of each session send / receive, by (line, endpoint)
    pub ses_send: HashMap<(usize, String), Ty>,
    pub ses_recv: HashMap<(usize, String), Ty>,
    pub scope_depth: usize,
    pub atomic_depth: usize,
    pub group_depth: usize,
    pub node_of: HashMap<String, String>,
    /// the two endpoint names bound by one `chan[session P]` (W-SES-5)
    pub ses_dual: HashMap<String, String>,
    /// W-SES-5 tree-rule findings, reported only when the program has no
    /// priorities either (`session_prio`, T28p)
    pub ses5: Vec<String>,
}

// ---------------------------------------------------------------- protocols

/// One element of a protocol state: a step still to do, or the end of an
/// unfolded `loop` body (jump back to the loop).
#[derive(Clone, Debug, PartialEq)]
enum PS {
    S(ProtoStep),
    End(Vec<ProtoStep>),
    /// the endpoint was consumed (moved, sent, passed or closed) at this
    /// line: its only state is that no operation is left (W-SES-4)
    Gone(usize, &'static str),
}
type St = Vec<PS>;

pub fn dual(steps: &[ProtoStep]) -> Vec<ProtoStep> {
    steps
        .iter()
        .map(|s| match s {
            ProtoStep::Send(t) => ProtoStep::Recv(t.clone()),
            ProtoStep::Recv(t) => ProtoStep::Send(t.clone()),
            ProtoStep::Offer(a) => ProtoStep::Choose(a.iter().map(|(l, b)| (l.clone(), dual(b))).collect()),
            ProtoStep::Choose(a) => ProtoStep::Offer(a.iter().map(|(l, b)| (l.clone(), dual(b))).collect()),
            ProtoStep::Loop(b) => ProtoStep::Loop(dual(b)),
            ProtoStep::Break => ProtoStep::Break,
        })
        .collect()
}

fn st_of(steps: &[ProtoStep]) -> St {
    steps.iter().cloned().map(PS::S).collect()
}

/// Canonical form: a pending `End(b)` is the same state as `Loop(b)`, and a
/// `break` leaves the innermost unfolded loop.
fn canon(mut s: St) -> St {
    loop {
        match s.first() {
            Some(PS::End(b)) => {
                let b = b.clone();
                s[0] = PS::S(ProtoStep::Loop(b));
            }
            Some(PS::S(ProtoStep::Break)) => {
                let k = s.iter().position(|x| matches!(x, PS::End(_)));
                match k {
                    Some(k) => {
                        s.drain(..=k);
                    }
                    None => {
                        s.remove(0);
                    }
                }
            }
            _ => return s,
        }
    }
}

/// Unfold loops at the head until the head is an operation (or the end).
fn head(s: St) -> St {
    let mut s = canon(s);
    for _ in 0..64 {
        match s.first() {
            Some(PS::S(ProtoStep::Loop(b))) => {
                let b = b.clone();
                let rest = s.split_off(1);
                s = st_of(&b);
                s.push(PS::End(b));
                s.extend(rest);
                s = canon(s);
            }
            _ => return s,
        }
    }
    s
}

fn show_head(s: &St) -> String {
    match s.first() {
        None => "the end of the protocol".into(),
        Some(PS::S(ProtoStep::Send(_))) => "a send".into(),
        Some(PS::S(ProtoStep::Recv(_))) => "a receive".into(),
        Some(PS::S(ProtoStep::Offer(_))) => "an offer".into(),
        Some(PS::S(ProtoStep::Choose(_))) => "a choose".into(),
        Some(PS::Gone(l, how)) => format!("nothing: it was {how} at line {l}"),
        Some(_) => "a loop".into(),
    }
}

fn gone(s: &St) -> Option<(usize, &'static str)> {
    match s.first() {
        Some(PS::Gone(l, h)) => Some((*l, h)),
        _ => None,
    }
}

/// One endpoint's protocol state driven by actions, the automaton the
/// session pass runs. Actions: `send`, `recv`, `choose:L`, `offer:L`
/// (the offer arm taken). `tests/lean_session.rs` drives it with traces
/// computed by `RCE.SessionRec` (`lean/RCE/SessionVectors.lean`) and
/// requires the same enabled actions at every point and the same end.
pub struct ProtoCursor(St);

impl ProtoCursor {
    pub fn new(steps: &[ProtoStep]) -> Self {
        ProtoCursor(st_of(steps))
    }
    /// the actions allowed here, sorted
    pub fn enabled(&self) -> Vec<String> {
        let s = head(self.0.clone());
        let mut v: Vec<String> = match s.first() {
            Some(PS::S(ProtoStep::Send(_))) => vec!["send".into()],
            Some(PS::S(ProtoStep::Recv(_))) => vec!["recv".into()],
            Some(PS::S(ProtoStep::Choose(a))) => a.iter().map(|(l, _)| format!("choose:{l}")).collect(),
            Some(PS::S(ProtoStep::Offer(a))) => a.iter().map(|(l, _)| format!("offer:{l}")).collect(),
            _ => vec![],
        };
        v.sort();
        v
    }
    pub fn step(&mut self, act: &str) -> bool {
        let s = head(self.0.clone());
        let rest: St = s.iter().skip(1).cloned().collect();
        let arm = |a: &Vec<(String, Vec<ProtoStep>)>, l: &str| a.iter().find(|(n, _)| n == l).map(|(_, b)| b.clone());
        let next = match (s.first(), act.split_once(':')) {
            (Some(PS::S(ProtoStep::Send(_))), None) if act == "send" => Some(vec![]),
            (Some(PS::S(ProtoStep::Recv(_))), None) if act == "recv" => Some(vec![]),
            (Some(PS::S(ProtoStep::Choose(a))), Some(("choose", l))) => arm(a, l),
            (Some(PS::S(ProtoStep::Offer(a))), Some(("offer", l))) => arm(a, l),
            _ => None,
        };
        match next {
            Some(b) => {
                let mut ns = st_of(&b);
                ns.extend(rest);
                self.0 = ns;
                true
            }
            None => false,
        }
    }
    pub fn at_end(&self) -> bool {
        head(self.0.clone()).is_empty()
    }
}

/// Simulation environment of the session pass: endpoint -> protocol state.
type Env = std::collections::BTreeMap<String, St>;

struct Flow6 {
    breaks: Vec<Env>,
    returned: bool,
}

impl Checker {
    // ------------------------------------------------------------ passes

    pub(crate) fn pre06b(&mut self, m: &Module) {
        for it in &m.items {
            if let Item::Protocol { name, steps, .. } = it {
                self.b6.protocols.insert(name.clone(), steps.clone());
            }
            if let Item::Node { name, actors, .. } = it {
                for a in actors {
                    self.b6.node_of.entry(a.clone()).or_insert(name.clone());
                }
            }
        }
        // TaskFailed (C.11.9) is built in unless the module declares it
        let declared = m.items.iter().any(|i| {
            matches!(i, Item::Error { name, .. } | Item::Struct { name, .. } | Item::Kind { name, .. } if name == "TaskFailed")
        });
        if !declared {
            let text = Ty::Path(vec!["text".into()], vec![]);
            self.types.insert(
                "TaskFailed".into(),
                crate::check::TyDef::Error {
                    payload: vec![("task".into(), text.clone()), ("reason".into(), text)],
                },
            );
        }
        let us = Ty::Path(vec!["usize".into()], vec![]);
        let f = |n: &str| Field {
            name: n.into(),
            ty: us.clone(),
            init: None,
            line: 0,
        };
        self.types.insert(
            "$Group".into(),
            crate::check::TyDef::Struct {
                generics: vec![],
                fields: vec![f("lane"), f("lanes")],
            },
        );
        // session pass over every function (fills ses_send / ses_recv)
        let mut fns: Vec<&FnDecl> = vec![];
        for it in &m.items {
            match it {
                Item::Fn(f) => fns.push(f),
                Item::Impl { members, .. } => fns.extend(members.iter()),
                Item::Actor { fns: fs, .. } => fns.extend(fs.iter()),
                _ => {}
            }
        }
        for f in fns {
            self.session_fn(f);
        }
        // W-SES-5: a session graph that is not a tree is still accepted when
        // its exchanges have priorities (T28p)
        let pending = std::mem::take(&mut self.b6.ses5);
        if !pending.is_empty() && !crate::session_prio::discharges(m, &self.b6.protocols) {
            for e in pending {
                self.e(e);
            }
        }
    }

    pub(crate) fn post06b(&mut self, m: &Module) {
        let mut seen: HashMap<String, usize> = HashMap::new();
        for it in &m.items {
            match it {
                Item::Protocol { name, steps, line } => {
                    if let Some(l) = seen.insert(format!("p:{name}"), *line) {
                        self.e(d(*line, "W-NAME-2", format!("protocol `{name}` declared twice (first at line {l})")));
                    }
                    self.check_proto(steps, 0, *line);
                }
                Item::Node { name, actors, line } => {
                    for a in actors {
                        if !matches!(self.types.get(a), Some(crate::check::TyDef::Actor { .. })) {
                            self.e(d(*line, "W-NODE-1", format!("node `{name}` places `{a}`, which is not an actor")));
                        }
                        if let Some(l) = seen.insert(format!("n:{a}"), *line) {
                            self.e(d(*line, "W-NODE-1", format!("actor `{a}` is already placed on a node (line {l})")));
                        }
                    }
                }
                Item::Supervisor {
                    name,
                    strategy,
                    children,
                    line,
                } => {
                    if !["one_for_one", "one_for_all", "rest_for_one"].contains(&strategy.as_str()) {
                        self.e(d(
                            *line,
                            "W-SUP-1",
                            format!("supervisor `{name}`: unknown strategy `{strategy}` (one_for_one, one_for_all or rest_for_one)"),
                        ));
                    }
                    let mut names = BTreeSet::new();
                    for c in children {
                        if !names.insert(c.name.clone()) {
                            self.e(d(c.line, "W-NAME-2", format!("child `{}` declared twice", c.name)));
                        }
                        if !matches!(self.types.get(&c.ty), Some(crate::check::TyDef::Actor { .. })) {
                            self.e(d(c.line, "W-SUP-2", format!("child `{}`: `{}` is not an actor", c.name, c.ty)));
                        }
                    }
                }
                Item::Event {
                    interrupt: Some(p),
                    line,
                    ..
                } => {
                    if !matches!(p, Expr::Int(n, _) if (0..=255).contains(n)) {
                        self.e(d(*line, "W-INTR-1", "an interrupt priority is an integer literal in 0..=255"));
                    }
                }
                Item::Step {
                    deadline: Some(e),
                    line,
                    ..
                } => {
                    if !matches!(e, Expr::Int(n, _) if *n > 0) {
                        self.e(d(*line, "W-RT-1", "a step deadline is a positive integer literal (ticks)"));
                    }
                }
                Item::Const { name, ty, line, .. } if name == "TICK_NS" => {
                    let ok = matches!(ty, Some(Ty::Path(p, a)) if a.is_empty() && p.len() == 1 && p[0] == "u64");
                    if !ok {
                        self.e(d(*line, "W-RT-1", "TICK_NS must be declared `const TICK_NS: u64 = n`"));
                    }
                }
                Item::Fn(f) => {
                    if let Some(r) = &f.sig.ret
                        && ty_mentions(r, "task")
                    {
                        self.e(d(f.sig.line, "W-SCOPE-1", "a join handle cannot leave its scope"));
                    }
                }
                _ => {}
            }
        }
    }

    fn check_proto(&mut self, steps: &[ProtoStep], loops: usize, line: usize) {
        for s in steps {
            match s {
                ProtoStep::Send(t) | ProtoStep::Recv(t) => {
                    let _ = self.lower(t, line);
                }
                ProtoStep::Offer(a) | ProtoStep::Choose(a) => {
                    let mut ls = BTreeSet::new();
                    for (l, b) in a {
                        if !ls.insert(l) {
                            self.e(d(line, "W-SES-3", format!("label `{l}` appears twice")));
                        }
                        self.check_proto(b, loops, line);
                    }
                }
                ProtoStep::Loop(b) => {
                    if !b.iter().any(|x| !matches!(x, ProtoStep::Loop(_) | ProtoStep::Break)) {
                        self.e(d(line, "W-SES-3", "a protocol loop needs a message before it repeats"));
                    }
                    self.check_proto(b, loops + 1, line)
                }
                ProtoStep::Break => {
                    if loops == 0 {
                        self.e(d(line, "W-SES-3", "`break` outside a protocol loop"));
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------ types

    pub(crate) fn lower06b(&mut self, k: &str, args: &[Ty], line: usize) -> T {
        if k == "session" {
            return match args.first() {
                Some(Ty::Path(p, _)) if p.len() == 1 && self.b6.protocols.contains_key(&p[0]) => {
                    T::Named("session".into(), vec![T::Named(p[0].clone(), vec![])])
                }
                Some(Ty::Path(p, _)) => {
                    self.e(d(line, "W-NAME-1", format!("unknown protocol `{}`", p.join("."))));
                    T::Never
                }
                _ => {
                    self.e(d(line, "W-SYN-1", "session needs a protocol name"));
                    T::Never
                }
            };
        }
        let inner = match args.first() {
            Some(t) => self.lower(t, line),
            None => T::Never,
        };
        if let Some(g) = self.merge_gap(&inner) {
            self.e(d(line, "W-CRDT-1", format!("lvar needs a type with Merge, found {inner} ({g})")));
        }
        T::Named("lvar".into(), vec![inner])
    }

    /// built-in Merge instances (C.11.5) or a user `impl Merge`
    pub(crate) fn mergeable(&mut self, t: &T) -> bool {
        self.merge_gap(t).is_none()
    }

    /// Why `t` has no `Merge` (C.11.5), or None when it has one: the
    /// built-in instances need a `Merge` on every component, so a struct
    /// without its own `impl Merge` needs one on every field, recursively
    /// (a struct reached again through itself is assumed, the greatest
    /// fixed point, since its merge recurses on finite values).
    pub(crate) fn merge_gap(&mut self, t: &T) -> Option<String> {
        let mut seen = vec![];
        self.merge_gap_in(t, &mut seen)
    }
    fn merge_gap_in(&mut self, t: &T, seen: &mut Vec<String>) -> Option<String> {
        match t {
            // union is a join for any element type with equality
            T::Int(_) | T::Bool | T::Never | T::Var(_) | T::Set(_) | T::List(_) => None,
            T::Opt(x) => self.merge_gap_in(x, seen),
            T::Map(_, v) => self.merge_gap_in(v, seen).map(|g| format!("its values: {g}")),
            T::Tuple(v) => v.iter().enumerate().find_map(|(i, x)| self.merge_gap_in(x, seen).map(|g| format!("component {i}: {g}"))),
            T::Named(n, args) => {
                if self.impls.iter().any(|(ty, tr)| ty == n && tr.as_deref() == Some("Merge")) || seen.contains(n) {
                    return None;
                }
                let Some(crate::check::TyDef::Struct { generics, fields }) = self.types.get(n).cloned() else {
                    return Some(format!("{t} has no impl Merge"));
                };
                seen.push(n.clone());
                let mut r = None;
                for f in &fields {
                    let ft = self.lower(&f.ty, f.line);
                    let ft = match &ft {
                        T::Named(g, a) if a.is_empty() => match generics.iter().position(|p| &p.name == g) {
                            Some(i) => args.get(i).cloned().unwrap_or(T::Never),
                            None => ft,
                        },
                        _ => ft,
                    };
                    if let Some(g) = self.merge_gap_in(&ft, seen) {
                        r = Some(format!("field `{}` of {n}: {g}", f.name));
                        break;
                    }
                }
                seen.pop();
                r
            }
            _ => Some(format!("{t} has no Merge")),
        }
    }

    pub(crate) fn lvar_new_ty(&mut self, args: &[Arg], l: usize) -> T {
        if args.is_empty() || args.len() > 2 {
            self.e(d(l, "W-TY-1", "`lvar_new` takes a bottom value and an optional top"));
            return T::Never;
        }
        let t = self.infer(&args[0].e, true);
        if args.len() == 2 {
            let u = self.infer(&args[1].e, true);
            self.expect(&u, &t, l);
        }
        let t = self.sub.resolve(&t);
        if let Some(g) = self.merge_gap(&t) {
            self.e(d(l, "W-CRDT-1", format!("lvar needs a type with Merge, found {t} ({g})")));
        }
        T::Named("lvar".into(), vec![t])
    }

    /// `chan[session P](n)`: the endpoints (session P, session dual(P)).
    /// dual(P) is named by the protocol declared equal to it, if any.
    pub(crate) fn session_pair(&mut self, a: &[T]) -> T {
        let p = match a.first() {
            Some(T::Named(p, _)) => p.clone(),
            _ => return T::Never,
        };
        let q = self.dual_name(&p);
        T::Tuple(vec![
            T::Named("session".into(), vec![T::Named(p, vec![])]),
            T::Named("session".into(), vec![T::Named(q, vec![])]),
        ])
    }

    pub(crate) fn dual_name(&self, p: &str) -> String {
        let Some(steps) = self.b6.protocols.get(p) else {
            return format!("dual {p}");
        };
        let du = dual(steps);
        let mut names: Vec<&String> = self.b6.protocols.keys().collect();
        names.sort();
        for n in names {
            if self.b6.protocols[n] == du {
                return n.clone();
            }
        }
        format!("dual {p}")
    }

    pub(crate) fn label_pat(&mut self, base: &T, label: &str, line: usize) -> bool {
        let T::Named(n, _) = self.sub.resolve(base) else {
            return false;
        };
        let Some(p) = n.strip_prefix("$labels:") else {
            return false;
        };
        let mut labels = BTreeSet::new();
        if let Some(steps) = self.b6.protocols.get(p) {
            collect_labels(steps, &mut labels);
        }
        if !labels.contains(label) {
            self.e(d(line, "W-SES-1", format!("`{label}` is not a label of protocol `{p}`")));
        }
        true
    }

    pub(crate) fn session_send_ty(&mut self, ch: &Expr, vt: &T, line: usize) {
        if let Expr::Name(c, _) = ch
            && let Some(t) = self.b6.ses_send.get(&(line, c.clone())).cloned()
        {
            let want = self.lower(&t, line);
            self.expect(vt, &want, line);
        }
    }

    pub(crate) fn session_recv_ty(&mut self, x: &Expr, l: usize) -> T {
        if let Expr::Name(c, _) = x
            && let Some(t) = self.b6.ses_recv.get(&(l, c.clone())).cloned()
        {
            return self.lower(&t, l);
        }
        self.sub.fresh()
    }

    pub(crate) fn method06b(&mut self, base: &T, m: &str, args: &[Arg], l: usize) -> Option<T> {
        let elem = match base {
            T::List(e) | T::Slice(e) | T::Array(e, _) => Some((**e).clone()),
            _ => None,
        };
        if let Some(e) = elem
            && (m == "pscan" || m == "pscan_incl")
        {
            if args.len() != 2 {
                self.e(d(l, "W-TY-1", format!("`{m}` takes a function and its identity")));
                return Some(T::Never);
            }
            let ft = self.infer(&args[0].e, true);
            self.expect(&ft, &T::Fn(vec![e.clone(), e.clone()], Box::new(e.clone())), l);
            let it = self.infer(&args[1].e, true);
            self.expect(&it, &e, l);
            return Some(T::List(Box::new(e)));
        }
        let T::Named(n, a) = base else { return None };
        let in_task = matches!(self.cx, Cx::Main | Cx::Task);
        match (n.as_str(), m) {
            ("session", "choose") => {
                if !in_task {
                    self.e(d(l, "W-EFF-1", "choose outside main or a task function"));
                }
                if !matches!(args, [Arg { e: Expr::Name(..), .. }]) {
                    self.e(d(l, "W-SES-1", "choose takes one label"));
                }
                Some(T::Unit)
            }
            ("session", "offer") => {
                if !in_task {
                    self.e(d(l, "W-EFF-1", "offer outside main or a task function"));
                }
                let p = match a.first() {
                    Some(T::Named(p, _)) => p.clone(),
                    _ => String::new(),
                };
                Some(T::Named(format!("$labels:{p}"), vec![]))
            }
            ("session", "close") => Some(T::Unit),
            ("lvar", "put") => {
                if args.len() == 1 {
                    let t = self.infer(&args[0].e, true);
                    self.expect(&t, &a[0], l);
                } else {
                    self.e(d(l, "W-TY-1", "`put` takes one value"));
                }
                Some(T::Unit)
            }
            ("lvar", "wait") => {
                if args.len() == 1 {
                    let t = self.infer(&args[0].e, true);
                    self.expect(&t, &T::List(Box::new(a[0].clone())), l);
                    // W-LVAR-2: the threshold set must be pairwise incompatible
                    // (Kuper and Newton, FHPC 2013: distinct thresholds join to
                    // top), else the threshold returned depends on the schedule.
                    // A literal list is decided at admission (`conc::lvar_gate`:
                    // two comparable thresholds require change); every `wait`
                    // checks the set against the cell's top at run time, which
                    // does not read the value, so the outcome is schedule-free.
                    if matches!(&args[0].e, Expr::List(xs, _) if xs.is_empty()) {
                        self.e(d(l, "W-LVAR-2", "`wait` needs a non-empty threshold list"));
                    }
                } else {
                    self.e(d(l, "W-TY-1", "`wait` takes a threshold list"));
                }
                Some(a[0].clone())
            }
            ("lvar", "freeze") => Some(a[0].clone()),
            ("task", "join") => {
                if !in_task {
                    self.e(d(l, "W-EFF-1", "join outside main or a task function"));
                }
                Some(T::Union(
                    Box::new(a.first().cloned().unwrap_or(T::Unit)),
                    Box::new(T::Named("TaskFailed".into(), vec![])),
                ))
            }
            _ => None,
        }
    }

    // ------------------------------------------------------------ statements

    pub(crate) fn marker06b(&mut self, r: &str, body: &[Stmt], line: usize) -> bool {
        if let Some(h) = r.strip_prefix("$spawnas:") {
            let Some(inner) = body.first() else { return false };
            self.stmt(inner);
            let mut ret = T::Unit;
            walk_stmts(body, &mut |s| {
                if let Stmt::Spawn(Expr::Call(f, _, _, _)) = s
                    && let Expr::Name(n, _) = &**f
                {
                    ret = T::Named(format!("$ret:{n}"), vec![]);
                }
            });
            if let T::Named(n, _) = &ret {
                let f = n.trim_start_matches("$ret:");
                ret = match self.fns.get(f).map(|i| i.decl.sig.ret.clone()) {
                    Some(Some(t)) => self.lower(&t, line),
                    _ => T::Unit,
                };
            }
            self.declare(h, T::Named("task".into(), vec![ret]), false, line, false);
            return false;
        }
        if let Some(g) = r.strip_prefix("$group:") {
            return self.group06b(g, body, line);
        }
        match r {
            "$barrier" => {
                self.e(d(line, "W-GRP-2", "barrier only as a statement at the top level of a group body"));
                false
            }
            "$scope" => {
                if !matches!(self.cx, Cx::Main | Cx::Task) {
                    self.e(d(line, "W-EFF-1", "scope outside main or a task function"));
                }
                self.b6.scope_depth += 1;
                let div = self.block(body);
                self.b6.scope_depth -= 1;
                div
            }
            "$atomic" => {
                if !matches!(self.cx, Cx::Main | Cx::Task | Cx::Handler) {
                    self.e(d(line, "W-EFF-1", "atomic outside main, a task function or a handler"));
                }
                let mut bad = None;
                walk_stmts(body, &mut |s| {
                    let b = match s {
                        Stmt::Select(..) => Some("select"),
                        Stmt::Spawn(_) => Some("spawn"),
                        Stmt::Arena(m, _, _, _) if m == "$scope" || m == "$nondet" => Some("scope or select"),
                        _ => None,
                    };
                    bad = bad.or(b);
                    for e in stmt_exprs(s) {
                        walk_expr(e, &mut |x| {
                            if matches!(x, Expr::Recv(..)) {
                                bad = bad.or(Some("receive"));
                            }
                        });
                    }
                });
                if let Some(b) = bad {
                    self.e(d(line, "W-ATOM-1", format!("no {b} inside atomic (the block cannot wait on itself)")));
                }
                // W-ATOM-2: the receiving actors are on one node
                let mut nodes = BTreeSet::new();
                let mut targets = vec![];
                walk_stmts(body, &mut |s| {
                    if let Stmt::Send(t, _) = s {
                        targets.push(t.clone());
                    }
                });
                for t in targets {
                    let qt = self.infer_quiet06(&t);
                    if let T::Named(a, _) = self.sub.resolve(&qt)
                        && matches!(self.types.get(&a), Some(crate::check::TyDef::Actor { .. }))
                    {
                        nodes.insert(self.b6.node_of.get(&a).cloned().unwrap_or_default());
                    }
                }
                if nodes.len() > 1 {
                    self.e(d(line, "W-ATOM-2", "an atomic block's actors must be on one node"));
                }
                self.b6.atomic_depth += 1;
                let div = self.block(body);
                self.b6.atomic_depth -= 1;
                div
            }
            _ => self.block(body),
        }
    }

    fn infer_quiet06(&mut self, e: &Expr) -> T {
        let n = self.errs.len();
        let t = self.infer(e, false);
        self.errs.truncate(n);
        t
    }

    fn group06b(&mut self, g: &str, body: &[Stmt], line: usize) -> bool {
        if self.device == 0 || self.b6.group_depth > 0 {
            self.e(d(line, "W-GRP-1", "with group only directly inside a device region"));
        }
        if let Some(Stmt::Expr(n)) = body.first() {
            let t = self.infer(n, false);
            self.expect(&t, &T::Int("usize"), line);
        }
        let body = &body[1..];
        self.b6.group_depth += 1;
        self.push_scope06();
        self.declare(g, T::Named("$Group".into(), vec![]), false, line, false);
        let mut shared = BTreeSet::new();
        for s in body {
            match s {
                Stmt::Var(n, _, _, _) => {
                    shared.insert(n.clone());
                }
                Stmt::Let(p, _, _, _) => {
                    let mut v = BTreeSet::new();
                    crate::check::pat_names_pub(p, &mut v);
                    shared.extend(v);
                }
                _ => {}
            }
        }
        for s in body {
            if !matches!(s, Stmt::Arena(m, _, _, _) if m == "$barrier") {
                self.stmt(s);
            }
        }
        self.pop_scope06();
        self.b6.group_depth -= 1;
        // W-GRP-3, per phase
        for phase in body.split(|s| matches!(s, Stmt::Arena(m, _, _, _) if m == "$barrier")) {
            let mut written = BTreeSet::new();
            let mut bad = vec![];
            walk_stmts(phase, &mut |s| {
                if let Stmt::Assign(op, pl, _) = s {
                    match pl {
                        Expr::Index(a, ix, _) => {
                            if let Expr::Name(n, _) = &**a
                                && shared.contains(n)
                            {
                                if op == "=" && ix.len() == 1 && is_lane(&ix[0], g) {
                                    written.insert(n.clone());
                                } else {
                                    bad.push(format!("lane writes `{n}` other than at [{g}.lane]"));
                                }
                            }
                        }
                        Expr::Name(n, _) if shared.contains(n) => {
                            bad.push(format!("lanes write the group-shared scalar `{n}`"));
                        }
                        _ => {}
                    }
                }
            });
            walk_stmts(phase, &mut |s| {
                let place = match s {
                    Stmt::Assign(_, pl, _) => Some(pl),
                    _ => None,
                };
                for e in stmt_exprs(s) {
                    let is_place = place.is_some_and(|p| std::ptr::eq(p, e));
                    walk_expr(e, &mut |x| {
                        if let Expr::Index(a, ix, _) = x
                            && let Expr::Name(n, _) = &**a
                            && written.contains(n)
                            && !(ix.len() == 1 && is_lane(&ix[0], g))
                            && !is_place
                        {
                            bad.push(format!("a lane reads `{n}` at another lane's index in the phase that writes it"));
                        }
                    });
                }
            });
            for b in bad {
                self.e(d(line, "W-GRP-3", b));
            }
        }
        false
    }

    // ------------------------------------------------------------ session pass

    fn session_fn(&mut self, f: &FnDecl) {
        self.b6.ses_dual.clear();
        let mut env = Env::new();
        for p in &f.sig.params {
            if let Param::Named { name, ty, .. } = p
                && let Ty::Path(q, a) = ty
                && q.len() == 1
                && q[0] == "session"
                && let Some(Ty::Path(pn, _)) = a.first()
                && let Some(steps) = self.b6.protocols.get(&pn[0])
            {
                env.insert(name.clone(), st_of(steps));
            }
        }
        let body = match &f.body {
            Body::Block(b) => b.clone(),
            Body::Arrow(e) => vec![Stmt::Expr(e.clone())],
        };
        let (env, fl) = self.ses_block(&body, env);
        if !fl.returned {
            self.ses_end(&env, f.sig.line);
        }
    }

    fn ses_end(&mut self, env: &Env, line: usize) {
        for (c, s) in env {
            let s = canon(s.clone());
            if !s.is_empty() && gone(&s).is_none() {
                self.e(d(
                    line,
                    "W-SES-2",
                    format!("session endpoint `{c}` is not at the end of its protocol (next: {})", show_head(&s)),
                ));
            }
        }
    }

    fn ses_step(&mut self, env: &mut Env, c: &str, line: usize, op: &str, label: Option<&str>) -> Option<Vec<(String, St)>> {
        let s = head(env.get(c)?.clone());
        if let Some((l0, how)) = gone(&s) {
            self.e(d(line, "W-SES-4", format!("`{c}`: {op} on an endpoint that was {how} at line {l0}")));
            return None;
        }
        self.ses_dual_held(env, c, line, op);
        let bad = |me: &mut Checker, s: &St| {
            me.e(d(line, "W-SES-1", format!("`{c}`: {op} not allowed here, the protocol expects {}", show_head(s))));
        };
        let rest: St = s.iter().skip(1).cloned().collect();
        match (s.first(), op) {
            (Some(PS::S(ProtoStep::Send(t))), "send") => {
                self.b6.ses_send.insert((line, c.into()), t.clone());
                env.insert(c.into(), rest);
            }
            (Some(PS::S(ProtoStep::Recv(t))), "receive") => {
                self.b6.ses_recv.insert((line, c.into()), t.clone());
                env.insert(c.into(), rest);
            }
            (Some(PS::S(ProtoStep::Choose(arms))), "choose") => {
                let l = label.unwrap_or("");
                match arms.iter().find(|(n, _)| n == l) {
                    Some((_, b)) => {
                        let mut ns = st_of(b);
                        ns.extend(rest);
                        env.insert(c.into(), ns);
                    }
                    None => {
                        self.e(d(line, "W-SES-1", format!("`{c}`: `{l}` is not a label of this choose")));
                        env.remove(c);
                    }
                }
            }
            (Some(PS::S(ProtoStep::Offer(arms))), "offer") => {
                return Some(
                    arms.iter()
                        .map(|(n, b)| {
                            let mut ns = st_of(b);
                            ns.extend(rest.clone());
                            (n.clone(), ns)
                        })
                        .collect(),
                );
            }
            _ => {
                bad(self, &s);
                env.remove(c);
            }
        }
        None
    }

    /// session operations inside one expression, in evaluation order
    fn ses_expr(&mut self, e: &Expr, env: &mut Env) {
        let mut ops: Vec<(String, usize, &'static str, Option<String>)> = vec![];
        let mut covered: Vec<*const Expr> = vec![];
        walk_expr(e, &mut |x| match x {
            Expr::Recv(c, l) => {
                if let Expr::Name(n, _) = &**c {
                    covered.push(&**c as *const Expr);
                    ops.push((n.clone(), *l, "receive", None));
                }
            }
            Expr::Call(f, args, l, _) => {
                if let Expr::Member(o, m, _) = &**f
                    && let Expr::Name(n, _) = &**o
                {
                    covered.push(&**o as *const Expr);
                    match m.as_str() {
                        "choose" => {
                            let lab = match args.first().map(|a| &a.e) {
                                Some(Expr::Name(l, _)) => Some(l.clone()),
                                _ => None,
                            };
                            ops.push((n.clone(), *l, "choose", lab));
                        }
                        "offer" => ops.push((n.clone(), *l, "offer-bare", None)),
                        "close" => ops.push((n.clone(), *l, "close", None)),
                        _ => {}
                    }
                }
            }
            _ => {}
        });
        // a bare use of an endpoint passes it on (delegation, a send of the
        // endpoint, a call or spawn argument, a rebinding): the owner here
        // keeps no capability, so the name is consumed (W-SES-4 on reuse)
        let mut moved = vec![];
        walk_expr(e, &mut |x| {
            if let Expr::Name(n, l) = x
                && env.contains_key(n)
                && !covered.contains(&(x as *const Expr))
            {
                moved.push((n.clone(), *l));
            }
        });
        for (c, l, op, lab) in ops {
            if !env.contains_key(&c) {
                continue;
            }
            if let Some((l0, how)) = gone(&env[&c]) {
                let what = if op == "offer-bare" { "offer" } else { op };
                self.e(d(l, "W-SES-4", format!("`{c}`: {what} on an endpoint that was {how} at line {l0}")));
                continue;
            }
            match op {
                "offer-bare" => {
                    self.e(d(l, "W-SES-1", format!("`{c}.offer()` must be the subject of a match over its labels")));
                    env.remove(&c);
                }
                "close" => {
                    self.ses_dual_held(env, &c, l, "close");
                    let s = canon(env[&c].clone());
                    if !s.is_empty() {
                        self.e(d(l, "W-SES-2", format!("`{c}` closed before the end of its protocol")));
                    }
                    env.insert(c.clone(), vec![PS::Gone(l, "closed")]);
                }
                _ => {
                    self.ses_step(env, &c, l, op, lab.as_deref());
                }
            }
        }
        for (n, l) in moved {
            let Some(s) = env.get(&n) else { continue };
            if let Some((l0, how)) = gone(s) {
                self.e(d(l, "W-SES-4", format!("`{n}` used after it was {how} at line {l0}")));
                continue;
            }
            env.insert(n, vec![PS::Gone(l, "moved")]);
        }
    }

    /// W-SES-5: a process that still holds both ends of one channel must
    /// not act on either (it would wait on itself; the session graph has a
    /// loop). It passes one end on first.
    fn ses_dual_held(&mut self, env: &Env, c: &str, line: usize, op: &str) {
        if let Some(y) = self.b6.ses_dual.get(c)
            && env.get(y).is_some_and(|s| gone(s).is_none())
        {
            self.b6.ses5.push(d(line, "W-SES-5", format!("`{c}`: {op} while this body still holds `{y}`, the other end of the same channel")));
        }
    }

    fn ses_join(&mut self, envs: Vec<Env>, line: usize) -> Env {
        let mut it = envs.into_iter();
        let Some(first) = it.next() else { return Env::new() };
        let mut out = first;
        for e in it {
            let keys: BTreeSet<String> = out.keys().chain(e.keys()).cloned().collect();
            for k in keys {
                let a = out.get(&k).map(|s| canon(s.clone()));
                let b = e.get(&k).map(|s| canon(s.clone()));
                if a != b {
                    let (ga, gb) = (a.as_ref().and_then(gone), b.as_ref().and_then(gone));
                    let live = |x: &Option<St>| x.as_ref().is_some_and(|s| gone(s).is_none() && !s.is_empty());
                    match (ga, gb) {
                        // consumed on every path (where it was used last does not matter)
                        (Some(g), Some(_)) => {
                            out.insert(k.clone(), vec![PS::Gone(g.0, g.1)]);
                            continue;
                        }
                        // consumed on one path, still owed on the other
                        (Some((l0, how)), None) | (None, Some((l0, how))) if live(&a) || live(&b) => {
                            self.e(d(
                                line,
                                "W-SES-4",
                                format!("`{k}` is {how} on one path (line {l0}) and still in use on another"),
                            ));
                        }
                        (Some(g), None) | (None, Some(g)) => {
                            out.insert(k.clone(), vec![PS::Gone(g.0, g.1)]);
                            continue;
                        }
                        (None, None) if a.is_some() && b.is_some() => {
                            self.e(d(line, "W-SES-1", format!("branches leave `{k}` in different protocol states")));
                        }
                        _ => {}
                    }
                    out.remove(&k);
                }
            }
        }
        out
    }

    fn ses_block(&mut self, b: &[Stmt], mut env: Env) -> (Env, Flow6) {
        let mut fl = Flow6 {
            breaks: vec![],
            returned: false,
        };
        if env.is_empty() && !block_has_session_new(b) {
            return (env, fl);
        }
        let outer: BTreeSet<String> = env.keys().cloned().collect();
        for s in b {
            if fl.returned {
                break;
            }
            match s {
                Stmt::Let(p, _, Expr::ChanNew(Ty::Path(q, a), cap, _), _)
                    if q.len() == 1 && q[0] == "session" =>
                {
                    if let Some(c) = cap {
                        self.ses_expr(c, &mut env);
                    }
                    if let (Pat::Tuple(ps), Some(Ty::Path(pn, _))) = (p, a.first())
                        && let Some(steps) = self.b6.protocols.get(&pn[0]).cloned()
                        && ps.len() == 2
                        && let (Pat::Path(x), Pat::Path(y)) = (&ps[0], &ps[1])
                    {
                        env.insert(x[0].clone(), st_of(&steps));
                        env.insert(y[0].clone(), st_of(&dual(&steps)));
                        self.b6.ses_dual.insert(x[0].clone(), y[0].clone());
                        self.b6.ses_dual.insert(y[0].clone(), x[0].clone());
                    }
                }
                Stmt::Let(p, _, e, _) => {
                    // `let m = x` moves the endpoint `x` to `m`: `m` owes
                    // the rest of x's protocol (SessionLin `move`)
                    let alias = match (p, e) {
                        (Pat::Path(m), Expr::Name(x, _)) if m.len() == 1 => {
                            env.get(x).filter(|s| gone(s).is_none()).cloned().map(|st| (m[0].clone(), st))
                        }
                        _ => None,
                    };
                    self.ses_expr(e, &mut env);
                    let mut v = BTreeSet::new();
                    crate::check::pat_names_pub(p, &mut v);
                    for n in v {
                        if env.get(&n).is_some_and(|s| gone(s).is_some()) {
                            env.remove(&n);
                        }
                    }
                    if let Some((m, st)) = alias {
                        if let (Expr::Name(x, _), true) = (e, true)
                            && let Some(y) = self.b6.ses_dual.get(x).cloned()
                        {
                            self.b6.ses_dual.insert(m.clone(), y.clone());
                            self.b6.ses_dual.insert(y, m.clone());
                        }
                        env.insert(m, st);
                    }
                }
                Stmt::Spawn(Expr::Call(_, args, l, _)) => {
                    let ends: Vec<&String> = args
                        .iter()
                        .filter_map(|a| match &a.e {
                            Expr::Name(x, _) if env.get(x).is_some_and(|s| gone(s).is_none()) => Some(x),
                            _ => None,
                        })
                        .collect();
                    if ends.len() > 1 {
                        self.b6.ses5.push(d(
                            *l,
                            "W-SES-5",
                            format!(
                                "spawn passes {} session endpoints ({}); at most one keeps the session graph a tree",
                                ends.len(),
                                ends.iter().map(|x| format!("`{x}`")).collect::<Vec<_>>().join(", ")
                            ),
                        ));
                    }
                    for e in stmt_exprs(s) {
                        self.ses_expr(e, &mut env);
                    }
                }
                Stmt::Send(Expr::Name(c, _), v) if env.contains_key(c) => {
                    self.ses_expr(v, &mut env);
                    self.ses_step(&mut env, c, v.line(), "send", None);
                }
                Stmt::Match(Expr::Call(f, _, l, _), arms, line)
                    if matches!(&**f, Expr::Member(o, m, _) if m == "offer" && matches!(&**o, Expr::Name(n, _) if env.contains_key(n))) =>
                {
                    let Expr::Member(o, _, _) = &**f else { unreachable!() };
                    let Expr::Name(c, _) = &**o else { unreachable!() };
                    let c = c.clone();
                    let Some(branches) = self.ses_step(&mut env, &c, *l, "offer", None) else {
                        continue;
                    };
                    let mut outs = vec![];
                    let mut covered = BTreeSet::new();
                    let mut wild = false;
                    for a in arms {
                        let lab = match &a.pat {
                            Pat::Path(p) if p.len() == 1 => Some(p[0].clone()),
                            Pat::Wild => {
                                wild = true;
                                None
                            }
                            _ => None,
                        };
                        let targets: Vec<&(String, St)> = branches
                            .iter()
                            .filter(|(n, _)| lab.as_ref().is_none_or(|l| l == n) && !covered.contains(n))
                            .collect();
                        if let Some(l) = &lab {
                            if covered.contains(l) {
                                self.e(d(a.line, "W-MATCH-2", "unreachable arm"));
                            }
                            if !branches.iter().any(|(n, _)| n == l) {
                                self.e(d(a.line, "W-SES-1", format!("`{l}` is not offered here")));
                            }
                        }
                        for (n, st) in targets {
                            covered.insert(n.clone());
                            let mut e2 = env.clone();
                            e2.insert(c.clone(), st.clone());
                            let (e3, f3) = self.ses_block(&a.body, e2);
                            fl.breaks.extend(f3.breaks);
                            if !f3.returned {
                                outs.push(e3);
                            }
                        }
                    }
                    if !wild && covered.len() < branches.len() {
                        self.e(d(*line, "W-SES-1", format!("match on `{c}.offer()` does not handle every label")));
                    }
                    env = self.ses_join(outs, *line);
                }
                Stmt::If(arms, els) => {
                    let mut outs = vec![];
                    for (c, body) in arms {
                        self.ses_expr(c, &mut env);
                        let (e2, f2) = self.ses_block(body, env.clone());
                        fl.breaks.extend(f2.breaks);
                        if !f2.returned {
                            outs.push(e2);
                        }
                    }
                    match els {
                        Some(body) => {
                            let (e2, f2) = self.ses_block(body, env.clone());
                            fl.breaks.extend(f2.breaks);
                            if !f2.returned {
                                outs.push(e2);
                            }
                        }
                        None => outs.push(env.clone()),
                    }
                    let line = stmt_line(s);
                    env = self.ses_join(outs, line);
                }
                Stmt::Match(e, arms, line) => {
                    self.ses_expr(e, &mut env);
                    let mut outs = vec![];
                    for a in arms {
                        let (e2, f2) = self.ses_block(&a.body, env.clone());
                        fl.breaks.extend(f2.breaks);
                        if !f2.returned {
                            outs.push(e2);
                        }
                    }
                    env = self.ses_join(outs, *line);
                }
                Stmt::Loop(bnd, body) | Stmt::While(_, bnd, body) => {
                    if let Stmt::While(c, _, _) = s {
                        self.ses_expr(c, &mut env);
                    }
                    let line = stmt_line(s);
                    let s0: Env = env.iter().map(|(k, v)| (k.clone(), canon(v.clone()))).collect();
                    let (e1, f1) = self.ses_block(body, s0.clone());
                    let mut ends = vec![s0.clone(), e1];
                    let mut exits = f1.breaks.clone();
                    if bnd.is_some() || matches!(s, Stmt::While(..)) {
                        exits.push(s0.clone());
                    }
                    let _ = self.ses_join(std::mem::take(&mut ends), line);
                    env = if exits.is_empty() { s0 } else { self.ses_join(exits, line) };
                    if f1.returned {
                        fl.returned = false;
                    }
                }
                Stmt::For(_, it, body, line) => {
                    self.ses_expr(it, &mut env);
                    let s0: Env = env.iter().map(|(k, v)| (k.clone(), canon(v.clone()))).collect();
                    let (e1, f1) = self.ses_block(body, s0.clone());
                    let _ = self.ses_join(vec![s0.clone(), e1], *line);
                    let mut exits = f1.breaks;
                    exits.push(s0);
                    env = self.ses_join(exits, *line);
                }
                Stmt::Break(_) => {
                    fl.breaks.push(env.clone());
                    fl.returned = true;
                }
                Stmt::Continue(_) => {
                    fl.returned = true;
                }
                Stmt::Return(e, line) => {
                    if let Some(e) = e {
                        self.ses_expr(e, &mut env);
                    }
                    self.ses_end(&env, *line);
                    fl.returned = true;
                    env.clear();
                }
                Stmt::Arena(_, bnd, body, _) | Stmt::Device(_, bnd, body, _) => {
                    if let Some(b) = bnd {
                        self.ses_expr(b, &mut env);
                    }
                    let (e2, f2) = self.ses_block(body, env);
                    env = e2;
                    fl.breaks.extend(f2.breaks);
                    fl.returned |= f2.returned;
                }
                Stmt::Comptime(body) => {
                    let (e2, _) = self.ses_block(body, env);
                    env = e2;
                }
                other => {
                    for e in stmt_exprs(other) {
                        self.ses_expr(e, &mut env);
                    }
                }
            }
        }
        // an endpoint bound in this block goes out of scope at its end: it
        // must be finished or passed on here (W-SES-2), and the enclosing
        // block does not see it
        if !fl.returned {
            let local: Env = env.iter().filter(|(k, _)| !outer.contains(*k)).map(|(k, v)| (k.clone(), v.clone())).collect();
            if !local.is_empty() {
                let line = b.last().map(stmt_line).unwrap_or(0);
                self.ses_end(&local, line);
                env.retain(|k, _| outer.contains(k));
            }
        }
        // `break` and `continue` end the block early: hand the break states up
        (env, fl)
    }

    /// W-LVAR-1: `x.freeze()` only after the par or loop that puts to x has
    /// joined (C.11.8). The rule is closed under aliasing: inside the
    /// branches of one par (or the body of a `for ... par`), a put or a
    /// freeze counts unless its receiver is an lvar the same branch made
    /// with `lvar_new` (and has not rebound), whatever the name; a call to a
    /// function or method that puts to or freezes an lvar it did not make
    /// counts as that put or freeze. If any branch may put to an outside
    /// lvar, no branch may freeze one. The frozen bit makes a missed case
    /// an error rather than a wrong answer (T29q); this rule makes it a
    /// static error, so an admitted program is deterministic.
    pub(crate) fn lvar_freeze06(&mut self, bodies: &[&[Stmt]], line: usize) {
        let sm = self.lvar_summaries();
        let uses: Vec<LvarUse> = bodies
            .iter()
            .map(|b| {
                let mut u = LvarUse::default();
                lv_scan(b, &mut BTreeSet::new(), &sm, &mut u);
                u
            })
            .collect();
        let put = uses.iter().find_map(|u| u.put.first().cloned());
        let Some(put) = put else { return };
        let mut seen = BTreeSet::new();
        for u in &uses {
            for f in &u.frz {
                if seen.insert(f.clone()) {
                    self.e(d(
                        line,
                        "W-LVAR-1",
                        format!("{f} inside a par may freeze an lvar that {put} puts to before the par has joined (an alias, a parameter or a helper can name the same lvar)"),
                    ));
                }
            }
        }
    }

    /// functions and methods that put to / freeze an lvar they did not make
    /// (a fixpoint over calls)
    fn lvar_summaries(&self) -> LvarSumm {
        let mut bodies: Vec<(bool, String, &Body)> = vec![];
        for (n, f) in &self.fns {
            bodies.push((false, n.clone(), &f.decl.body));
        }
        for (_, f) in &self.methods {
            bodies.push((true, f.decl.sig.name.clone(), &f.decl.body));
        }
        let mut sm = LvarSumm::default();
        loop {
            let mut changed = false;
            for (meth, n, b) in &bodies {
                let mut u = LvarUse::default();
                match b {
                    Body::Arrow(e) => lv_expr(e, &BTreeSet::new(), &sm, &mut u),
                    Body::Block(st) => lv_scan(st, &mut BTreeSet::new(), &sm, &mut u),
                }
                let (pset, fset) = if *meth { (&mut sm.put_m, &mut sm.frz_m) } else { (&mut sm.put_f, &mut sm.frz_f) };
                if !u.put.is_empty() && pset.insert(n.clone()) {
                    changed = true;
                }
                if !u.frz.is_empty() && fset.insert(n.clone()) {
                    changed = true;
                }
            }
            if !changed {
                return sm;
            }
        }
    }

    fn push_scope06(&mut self) {
        self.push_scope_pub();
    }
    fn pop_scope06(&mut self) {
        self.pop_scope_pub();
    }
}

#[derive(Default)]
struct LvarSumm {
    put_f: BTreeSet<String>,
    frz_f: BTreeSet<String>,
    put_m: BTreeSet<String>,
    frz_m: BTreeSet<String>,
}

/// puts and freezes whose lvar may have been made outside the block
#[derive(Default)]
struct LvarUse {
    put: Vec<String>,
    frz: Vec<String>,
}

fn lv_is_new(e: &Expr) -> bool {
    matches!(e, Expr::Call(f, _, _, _) if matches!(&**f, Expr::Name(n, _) if n == "lvar_new"))
}

fn lv_expr(e: &Expr, fresh: &BTreeSet<String>, sm: &LvarSumm, u: &mut LvarUse) {
    walk_expr(e, &mut |x| {
        let Expr::Call(f, _, _, _) = x else { return };
        match &**f {
            Expr::Member(o, m, _) if m == "put" || m == "freeze" => {
                let own = matches!(&**o, Expr::Name(n, _) if fresh.contains(n));
                if !own {
                    let who = match &**o {
                        Expr::Name(n, _) => format!("`{n}.{m}()`"),
                        _ => format!("`.{m}()`"),
                    };
                    if m == "put" { u.put.push(who) } else { u.frz.push(who) }
                }
            }
            Expr::Member(_, m, _) => {
                if sm.put_m.contains(m) {
                    u.put.push(format!("method `{m}`"));
                }
                if sm.frz_m.contains(m) {
                    u.frz.push(format!("method `{m}`"));
                }
            }
            Expr::Name(g, _) => {
                if sm.put_f.contains(g) {
                    u.put.push(format!("`{g}`"));
                }
                if sm.frz_f.contains(g) {
                    u.frz.push(format!("`{g}`"));
                }
            }
            _ => {}
        }
    });
}

/// scan a block in order; `fresh` holds the names bound here to `lvar_new`
fn lv_scan(b: &[Stmt], fresh: &mut BTreeSet<String>, sm: &LvarSumm, u: &mut LvarUse) {
    for s in b {
        for e in stmt_exprs(s) {
            lv_expr(e, fresh, sm, u);
        }
        let mut sub = |b: &[Stmt], fresh: &BTreeSet<String>, u: &mut LvarUse| lv_scan(b, &mut fresh.clone(), sm, u);
        match s {
            Stmt::Let(p, _, e, _) => {
                let mut names = vec![];
                lv_pat_names(p, &mut names);
                for n in names {
                    if lv_is_new(e) && matches!(p, Pat::Path(_)) {
                        fresh.insert(n);
                    } else {
                        fresh.remove(&n);
                    }
                }
            }
            Stmt::Var(n, _, e, _) => {
                if lv_is_new(e) {
                    fresh.insert(n.clone());
                } else {
                    fresh.remove(n);
                }
            }
            Stmt::Assign(_, Expr::Name(n, _), _) => {
                fresh.remove(n);
            }
            Stmt::If(arms, els) => {
                for (_, b) in arms {
                    sub(b, fresh, u);
                }
                if let Some(b) = els {
                    sub(b, fresh, u);
                }
            }
            Stmt::While(_, _, b)
            | Stmt::Loop(_, b)
            | Stmt::For(_, _, b, _)
            | Stmt::Arena(_, _, b, _)
            | Stmt::Device(_, _, b, _)
            | Stmt::Comptime(b) => sub(b, fresh, u),
            Stmt::Match(_, arms, _) => {
                for a in arms {
                    if let Some(g) = &a.guard {
                        lv_expr(g, fresh, sm, u);
                    }
                    sub(&a.body, fresh, u);
                }
            }
            Stmt::Select(arms, _) => {
                for a in arms {
                    match a {
                        SelArm::Recv(_, e, b) | SelArm::After(e, b) => {
                            lv_expr(e, fresh, sm, u);
                            sub(b, fresh, u);
                        }
                        SelArm::Send(a, c, b) => {
                            lv_expr(a, fresh, sm, u);
                            lv_expr(c, fresh, sm, u);
                            sub(b, fresh, u);
                        }
                        SelArm::Default(b) => sub(b, fresh, u),
                    }
                }
            }
            _ => {}
        }
    }
}

fn lv_pat_names(p: &Pat, out: &mut Vec<String>) {
    match p {
        Pat::Path(v) if v.len() == 1 => out.push(v[0].clone()),
        Pat::Ctor(_, fs) => fs.iter().for_each(|(_, q)| lv_pat_names(q, out)),
        Pat::Tuple(v) | Pat::Alt(v) => v.iter().for_each(|q| lv_pat_names(q, out)),
        _ => {}
    }
}

fn block_has_session_new(b: &[Stmt]) -> bool {
    let mut f = false;
    walk_stmts(b, &mut |s| {
        if let Stmt::Let(_, _, Expr::ChanNew(Ty::Path(q, _), _, _), _) = s
            && q.len() == 1
            && q[0] == "session"
        {
            f = true;
        }
    });
    f
}

pub(crate) fn stmt_line(s: &Stmt) -> usize {
    if let Some(e) = stmt_exprs(s).first() {
        return e.line();
    }
    let mut l = 0;
    walk_stmts(std::slice::from_ref(s), &mut |x| {
        if l == 0
            && let Some(e) = stmt_exprs(x).first()
        {
            l = e.line();
        }
    });
    l
}

fn collect_labels(steps: &[ProtoStep], out: &mut BTreeSet<String>) {
    for s in steps {
        match s {
            ProtoStep::Offer(a) | ProtoStep::Choose(a) => {
                for (l, b) in a {
                    out.insert(l.clone());
                    collect_labels(b, out);
                }
            }
            ProtoStep::Loop(b) => collect_labels(b, out),
            _ => {}
        }
    }
}

fn is_lane(e: &Expr, g: &str) -> bool {
    matches!(e, Expr::Member(o, m, _) if m == "lane" && matches!(&**o, Expr::Name(n, _) if n == g))
}

fn ty_mentions(t: &Ty, n: &str) -> bool {
    match t {
        Ty::Path(p, a) => (p.len() == 1 && p[0] == n) || a.iter().any(|x| ty_mentions(x, n)),
        Ty::Ref(_, x) | Ty::Opt(x) | Ty::Slice(x) | Ty::Chan(x) | Ty::Iter(x) => ty_mentions(x, n),
        Ty::Tuple(v) => v.iter().any(|x| ty_mentions(x, n)),
        _ => false,
    }
}
