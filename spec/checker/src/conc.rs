//! Draft 0.6 and 0.6b concurrency semantics for the reference evaluator
//! (Appendix C, book/14-draft06-concurrency.md). This module gives the evaluator:
//!
//! * the cost table of C.6.4 and `with budget` (`trap("budget")`);
//! * `par` (C.4, T16: meaning = branches in source order; the evaluator can
//!   also run the branches in a permuted order or on threads, which by T16
//!   gives the same result, and is used to test T16);
//! * `for ... par` with the fixed reduction tree T(n) of C.5.2;
//! * workgroups and barriers (C.11.2), `pscan` (C.11.6), LVars (C.11.8)
//!   and the built-in `Merge` instances (C.11.5);
//! * the tick runtime of 6.8 for `fn main`: tasks on threads that move one
//!   communication step per tick, channels with held sends, `select` with
//!   rotation or `nondet` choice logged as `Choice` (C.5), `spawn ...
//!   priority` (order (-p, id)), `with deadline ... else` cancellation at
//!   communication steps (C.6.2), actors with mailboxes, `scope` and join
//!   handles (C.11.9), `atomic` (C.11.10), supervisors (C.11.11), nodes,
//!   logged partitions and vector clocks (C.11.4), and session endpoints
//!   (C.11.3).
use crate::ast::*;
use crate::interp::{Flow, Interp, R, Shared, V, chk, trap};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, mpsc};

/// The split of the fixed reduction tree (C.8) for m >= 2 leaves: the left
/// part is the largest power of two below m, so every left subtree is a
/// complete binary tree. One definition for the reference evaluator, the
/// placement executor (`host` re-exports it) and, by the same formula,
/// every runtime. It lives here so the wasm checker needs no host module.
pub fn tree_split(m: usize) -> usize {
    debug_assert!(m >= 2);
    1usize << (m - 1).ilog2()
}

// ---------------------------------------------------------------- costs (C.6.4)

/// Cost of a call (user function, closure, builtin method).
pub const C_CALL: i128 = 4;
/// Cost of a channel send or receive.
pub const C_CHAN: i128 = 8;
/// Cost of a spawn.
pub const C_SPAWN: i128 = 32;
/// Cost of one `par` branch (c_fork).
pub const C_FORK: i128 = 16;

/// Cost of a binary operator applied to a left operand `x` (C.6.4): integer,
/// bool, text and comparison operators 1; float `+ - *` 2; float `/ %` 8.
pub fn op_cost(op: &str, x: &V) -> i128 {
    match x {
        V::F(_) | V::F32(_) => match op {
            "+" | "-" | "*" => 2,
            "/" | "%" => 8,
            _ => 1,
        },
        _ => 1,
    }
}

/// Size of a value for method costs (C.6.4): text is its UTF-8 byte
/// length, a list its length, a map its number of entries, anything
/// else 0.
pub fn size_of(v: &V) -> i128 {
    match v {
        V::Text(s) => s.len() as i128,
        V::List(xs) | V::Set(xs) => xs.len() as i128,
        V::Map(m) => m.len() as i128,
        _ => 0,
    }
}

/// ⌈log₂ n⌉ for n ≥ 1, and 0 for n ≤ 1.
pub fn ceil_log2(n: i128) -> i128 {
    if n <= 1 { 0 } else { 128 - (n - 1).leading_zeros() as i128 }
}

/// Work of a standard library method (C.6.4), by complexity class, on
/// n = size of the receiver and k = size of the first argument:
/// linear methods n + k, `sort` n·⌈log₂ n⌉, every other method 0.
/// Saturating, so it never overflows.
pub fn method_work(recv: &V, m: &str, args: &[V]) -> i128 {
    let n = size_of(recv);
    let k = args.first().map(size_of).unwrap_or(0);
    // the classes are read from spec-data.json (C.6.4, `specread`)
    let kind = match recv {
        V::Text(_) => "text",
        V::List(_) => "list",
        V::Set(_) => "set",
        V::Map(_) => "map",
        _ => "",
    };
    let class = crate::specread::work(kind, m);
    let lin = class == crate::specread::Work::Linear;
    if lin {
        n.saturating_add(k)
    } else if class == crate::specread::Work::NLogN {
        n.saturating_mul(ceil_log2(n))
    } else {
        0
    }
}

// ---------------------------------------------------------------- task context

/// How the evaluator runs the branches of a `par`. The meaning is fixed
/// (source order, C.4, T16); the modes other than `Seq` exist to test T16.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum ParMode {
    #[default]
    Seq,
    /// branches run in a permutation chosen by the seed, on copies of the
    /// store, whose writes are then merged in source order
    Permute(u64),
    /// branches run at the same time on OS threads
    Threads,
}

/// Per-evaluator concurrency state.
#[derive(Default)]
pub struct TaskCx {
    /// enclosing `with budget` blocks: (spent, limit)
    pub budgets: Vec<(i128, i128)>,
    /// total cost units charged by this evaluator (E_sim units, T20)
    pub spent: i128,
    pub par_mode: ParMode,
    pub link: Option<Link>,
    pub id: String,
    pub now: u64,
    /// active deadline frames: (frame id, last tick the body may finish in)
    pub dl: Vec<(u64, u64)>,
    pub next_frame: u64,
    pub sel_k: HashMap<usize, u64>,
    pub created: u64,
    /// per-key ordinals of named creations (actors by supervisor position,
    /// spawns by callee): the stable path of C.11.11
    pub ords: HashMap<String, u64>,
    pub scopes: Vec<u64>,
    pub atomic: Option<Vec<(V, V)>>,
    pub handler_sends: Option<Vec<(V, V)>>,
    pub par_ctx: Option<Arc<ParCtx>>,
}

/// Shared state of the branches of one threaded `par` (LVar waits).
#[derive(Default)]
pub struct ParCtx {
    m: Mutex<ParState>,
    cv: Condvar,
    /// the enclosing par's context: a put wakes the waiters of every enclosing par
    parent: Option<Arc<ParCtx>>,
}
/// Branch accounting for lvar waits (T29). `ver` counts the puts that
/// changed an lvar; `idle` counts waiting branches that have checked their
/// thresholds at the current `ver`. When every running branch is idle no
/// put can come, and the waits fail together (`stuck`). An earlier version
/// counted woken-but-not-yet-rechecked branches as blocked and could report
/// a chain of waits stuck while it was making progress.
#[derive(Default)]
struct ParState {
    running: usize,
    idle: usize,
    ver: u64,
    stuck: bool,
}
impl ParCtx {
    fn new(running: usize, parent: Option<Arc<ParCtx>>) -> ParCtx {
        ParCtx {
            m: Mutex::new(ParState { running, ..ParState::default() }),
            cv: Condvar::new(),
            parent,
        }
    }
    fn bump(&self) {
        let mut g = self.m.lock().unwrap();
        g.ver += 1;
        g.idle = 0;
        self.cv.notify_all();
        drop(g);
        if let Some(p) = &self.parent {
            p.bump();
        }
    }
}

/// A lattice cell (C.11.8): the value, the optional top, and the frozen
/// bit of Kuper, Turon, Krishnaswami and Newton (POPL 2014, Sec. 4.2). The
/// bit is read and written under the value's lock, so a put and a freeze
/// are ordered. Model: `RCE/LVarQD.lean` (T29q).
pub struct LCell {
    v: Mutex<V>,
    top: Option<V>,
    frz: std::sync::atomic::AtomicBool,
}
impl std::fmt::Debug for LCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LCell({})", self.peek().show())
    }
}
impl LCell {
    pub fn peek(&self) -> V {
        self.v.lock().map(|g| g.clone()).unwrap_or(V::Unit)
    }
    pub fn frozen(&self) -> bool {
        self.frz.load(std::sync::atomic::Ordering::SeqCst)
    }
    /// `freeze()`: set the frozen bit and read the exact value (E-FREEZE-SIMPLE)
    pub fn freeze(&self) -> V {
        let g = self.v.lock().unwrap();
        self.frz.store(true, std::sync::atomic::Ordering::SeqCst);
        g.clone()
    }
    /// `lvar_new(bottom)` or `lvar_new(bottom, top)`.
    pub fn new_value(vals: Vec<V>) -> V {
        let mut it = vals.into_iter();
        let b = it.next().unwrap_or(V::Unit);
        V::LVar(Arc::new(LCell {
            v: Mutex::new(b),
            top: it.next(),
            frz: std::sync::atomic::AtomicBool::new(false),
        }))
    }
}

/// SHA-256 based id of 6.6 (first 16 hex digits), from a name-free key.
/// Stable paths (C.11.11). An id names where an instance sits in the
/// program's structure, never a source line or a count of unrelated
/// creations, so an edit that keeps the structure keeps the ids and a
/// restart log replays across it. The key is name-free (6.6, T7): a
/// supervised actor by its tree position `i.j` (supervisor i in
/// declaration order, its child j), the creating task and its ordinal
/// there; an unsupervised actor by `-`, the task and its ordinal among
/// that task's unsupervised creations; a task by its parent and its
/// ordinal among the parent's spawns. Declared names enter only the
/// schema of a restart log (`sup_tree`).
pub fn actor_pos(m: &Module, ty: &str) -> String {
    let mut i = 0;
    for it in &m.items {
        if let Item::Supervisor { children, .. } = it {
            if let Some(j) = children.iter().position(|c| c.ty == ty) {
                return format!("{i}.{j}");
            }
            i += 1;
        }
    }
    "-".to_string()
}

pub fn ord_next(ords: &mut HashMap<String, u64>, key: &str) -> u64 {
    let k = ords.entry(key.to_string()).or_insert(0);
    *k += 1;
    *k - 1
}

pub fn stable_actor_id(pos: &str, task: &str, idx: u64) -> String {
    hid(&format!("actor|{pos}|{task}|{idx}"))
}

pub fn stable_spawn_id(parent: &str, idx: u64) -> String {
    hid(&format!("spawn|{parent}|{idx}"))
}

/// The schema of a restart log (C.11.11): every supervisor position by its
/// declared names, `Sup/child`, in declaration order.
pub fn sup_tree(m: &Module) -> Vec<String> {
    let mut v = vec![];
    for i in &m.items {
        if let Item::Supervisor { name, children, .. } = i {
            for c in children {
                v.push(format!("{name}/{}", c.name));
            }
        }
    }
    v
}

/// Migration rule of a restart log (C.11.11). Ids are keyed by tree
/// position, so a log replays on a program in which every recorded
/// position still holds the same declared supervisor and child: the
/// recorded supervisors are the first ones, in order, and each keeps its
/// recorded children as the first ones, in order. New children may follow
/// a supervisor's recorded ones and new supervisors may follow the
/// recorded ones. Otherwise the first recorded position that is gone
/// (`removed`) or now sits elsewhere (`reordered`).
pub fn tree_migrates(recorded: &[String], current: &[String]) -> Result<(), String> {
    let group = |v: &[String]| {
        let mut g: Vec<(String, Vec<String>)> = vec![];
        for p in v {
            let (s, c) = p.split_once('/').unwrap_or((p.as_str(), ""));
            match g.last_mut() {
                Some((x, cs)) if x == s => cs.push(c.to_string()),
                _ => g.push((s.to_string(), vec![c.to_string()])),
            }
        }
        g
    };
    let (r, c) = (group(recorded), group(current));
    for (k, (sup, kids)) in r.iter().enumerate() {
        for (j, kid) in kids.iter().enumerate() {
            let same = c.get(k).is_some_and(|(s, cs)| s == sup && cs.get(j) == Some(kid));
            if !same {
                let p = format!("{sup}/{kid}");
                return Err(if current.contains(&p) {
                    format!("trap: upgrade reordered supervisor position {p}")
                } else {
                    format!("trap: upgrade removed supervisor position {p}")
                });
            }
        }
    }
    Ok(())
}

pub fn hid(key: &str) -> String {
    use sha2::{Digest, Sha256};
    let h = Sha256::digest(key.as_bytes());
    h.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------- canonical order and Merge

fn rank(v: &V) -> u8 {
    match v {
        V::Unit | V::Opt(None) => 0,
        V::Bool(_) => 1,
        V::Int(..) => 2,
        V::F(_) | V::F32(_) => 3,
        V::Text(_) => 4,
        V::List(_) | V::Set(_) | V::Tuple(_) => 5,
        _ => 6,
    }
}
fn num(v: &V) -> f64 {
    match v {
        V::Int(x, _) => *x as f64,
        V::F(x) | V::F32(x) => *x,
        _ => 0.0,
    }
}
/// Canonical total order of 6.1 on values.
pub fn canon_cmp(a: &V, b: &V) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    let (ra, rb) = (rank(a), rank(b));
    if ra != rb {
        return ra.cmp(&rb);
    }
    match (a, b) {
        (V::Bool(x), V::Bool(y)) => x.cmp(y),
        (V::Int(x, _), V::Int(y, _)) => x.cmp(y),
        // IEEE 754 totalOrder (-0.0 < 0.0), ints and floats never equal:
        // the order agrees with state identity (6.1)
        (x, y) if ra == 3 => num(x).total_cmp(&num(y)),
        (V::Text(x), V::Text(y)) => x.cmp(y),
        (V::List(x) | V::Set(x) | V::Tuple(x), V::List(y) | V::Set(y) | V::Tuple(y)) => {
            for (p, q) in x.iter().zip(y) {
                let c = canon_cmp(p, q);
                if c != Equal {
                    return c;
                }
            }
            x.len().cmp(&y.len())
        }
        (V::Opt(Some(x)), V::Opt(Some(y))) => canon_cmp(x, y),
        (V::Rec(n, x), V::Rec(m, y)) => n.cmp(m).then_with(|| {
            for ((k1, v1), (k2, v2)) in x.iter().zip(y) {
                let c = k1.cmp(k2).then_with(|| canon_cmp(v1, v2));
                if c != Equal {
                    return c;
                }
            }
            x.len().cmp(&y.len())
        }),
        _ => a.show().cmp(&b.show()),
    }
}

/// The built-in `Merge` instances of C.11.5: integers under `max`, `bool`
/// under `or`, lists and sets under union (kept sorted, no duplicates),
/// maps pointwise, tuples and records componentwise, `none` as bottom.
/// `set_of(xs)`: the elements in canonical order without duplicates.
pub fn set_of(mut xs: Vec<V>) -> V {
    xs.sort_by(canon_cmp);
    xs.dedup_by(|p, q| canon_cmp(p, q).is_eq());
    V::Set(xs.into())
}

pub fn merge_builtin(a: &V, b: &V) -> R<V> {
    Ok(match (a, b) {
        (V::Int(x, t), V::Int(y, u)) => V::Int(*x.max(y), if *t == "lit" { u } else { t }),
        (V::Bool(x), V::Bool(y)) => V::Bool(*x || *y),
        (V::List(xs), V::List(ys)) => {
            let mut v: Vec<V> = xs.iter().chain(ys).cloned().collect();
            v.sort_by(canon_cmp);
            v.dedup_by(|p, q| canon_cmp(p, q).is_eq());
            V::List(v.into())
        }
        (V::Set(xs), V::Set(ys)) => {
            let mut v: Vec<V> = xs.iter().chain(ys).cloned().collect();
            v.sort_by(canon_cmp);
            v.dedup_by(|p, q| canon_cmp(p, q).is_eq());
            V::Set(v.into())
        }
        (V::Map(xs), V::Map(ys)) => {
            let mut out = xs.clone();
            for (k, v) in ys {
                if let Some(e) = out.iter_mut().find(|(k2, _)| canon_cmp(k2, k).is_eq()) {
                    e.1 = merge_builtin(&e.1, v)?;
                } else {
                    out.push((k.clone(), v.clone()));
                }
            }
            out.sort_by(|p, q| canon_cmp(&p.0, &q.0));
            V::Map(out)
        }
        (V::Tuple(xs), V::Tuple(ys)) if xs.len() == ys.len() => V::Tuple(
            xs.iter()
                .zip(ys)
                .map(|(p, q)| merge_builtin(p, q))
                .collect::<R<_>>()?,
        ),
        (V::Rec(n, xs), V::Rec(m, ys)) if n == m && xs.len() == ys.len() => V::Rec(
            n.clone(),
            xs.iter()
                .zip(ys)
                .map(|((k, p), (_, q))| Ok((k.clone(), merge_builtin(p, q)?)))
                .collect::<R<_>>()?,
        ),
        (V::Opt(None), x) | (x, V::Opt(None)) => x.clone(),
        (V::Opt(Some(x)), V::Opt(Some(y))) => V::Opt(Some(Shared::from(merge_builtin(x, y)?))),
        (x, y) => return trap(format!("no Merge instance for {} and {}", x.show(), y.show())),
    })
}

// ---------------------------------------------------------------- runtime protocol

/// The task end of the link to the scheduler.
pub struct Link {
    pub tx: mpsc::Sender<Req>,
    pub rx: mpsc::Receiver<RepMsg>,
}
pub struct Req {
    pub op: Op,
    pub dl: Vec<(u64, u64)>,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PatClass {
    Any,
    Some,
    None,
}
pub enum SArm {
    Recv(V, PatClass),
    Send(V, V),
    After(u64),
    Default,
}
pub enum Op {
    Send(V, V),
    Recv(V),
    Select {
        arms: Vec<SArm>,
        key: (usize, u64),
        nondet: bool,
    },
    Spawn {
        f: String,
        args: Vec<V>,
        prio: u8,
        site: usize,
        idx: u64,
        scope: u64,
    },
    Emit(V),
    Close(V),
    Join(String),
    AwaitScope(u64),
    Atomic(Vec<(V, V)>),
    Done(Result<V, Flow>),
}
pub enum Rep {
    Go,
    Val(V),
    Arm(usize, Option<V>),
    Spawned(String),
    Cancel(u64),
    Failed(V),
}
pub struct RepMsg {
    pub tick: u64,
    pub rep: Rep,
}

/// Cancellation of a whole task (sibling of a failed task in a scope).
pub const CANCEL_TASK: u64 = u64::MAX;

// ---------------------------------------------------------------- evaluator side

fn ident(op: &str, before: &V) -> R<V> {
    Ok(match (op, before) {
        ("+=" | "|=" | "^=", V::Int(_, t)) => V::Int(0, t),
        ("*=", V::Int(_, t)) => V::Int(1, t),
        ("&=", V::Int(_, t)) => {
            let t2 = if *t == "lit" { "i64" } else { t };
            let (lo, hi) = crate::types::int_range(t2);
            V::Int(if lo == 0 { hi } else { -1 }, t)
        }
        ("+=", V::F(_)) => V::F(-0.0),
        ("*=", V::F(_)) => V::F(1.0),
        ("+=", V::F32(_)) => V::F32(-0.0),
        ("*=", V::F32(_)) => V::F32(1.0),
        ("&=", V::Bool(_)) => V::Bool(true),
        ("|=" | "^=", V::Bool(_)) => V::Bool(false),
        (o, v) => return trap(format!("no identity for {o} on {}", v.show())),
    })
}

pub(crate) fn assign_ops(b: &[Stmt], out: &mut Vec<(String, String)>) {
    crate::check::walk_stmts(b, &mut |s| {
        if let Stmt::Assign(op, Expr::Name(n, _), _) = s
            && ["+=", "*=", "&=", "|=", "^="].contains(&op.as_str())
            && !out.iter().any(|(m, _)| m == n)
        {
            out.push((n.clone(), op.clone()));
        }
    });
}

pub(crate) fn declared_in(b: &[Stmt]) -> std::collections::BTreeSet<String> {
    let mut d = std::collections::BTreeSet::new();
    crate::check::walk_stmts(b, &mut |s| match s {
        Stmt::Var(n, ..) => {
            d.insert(n.clone());
        }
        Stmt::Let(p, ..) | Stmt::For(p, ..) => pat_binds(p, &mut d),
        _ => {}
    });
    d
}
fn pat_binds(p: &Pat, out: &mut std::collections::BTreeSet<String>) {
    match p {
        Pat::Path(v) if v.len() == 1 => {
            out.insert(v[0].clone());
        }
        Pat::Ctor(_, a) => a.iter().for_each(|(_, q)| pat_binds(q, out)),
        Pat::Tuple(v) | Pat::Alt(v) => v.iter().for_each(|q| pat_binds(q, out)),
        _ => {}
    }
}

fn has_wait(b: &[Stmt]) -> bool {
    let mut w = false;
    crate::check::walk_stmts(b, &mut |s| {
        for e in crate::check::stmt_exprs(s) {
            crate::check::walk_expr(e, &mut |x| {
                if let Expr::Call(c, _, _, _) = x
                    && matches!(&**c, Expr::Member(_, m, _) if m == "wait")
                {
                    w = true;
                }
            });
        }
    });
    w
}

fn pat_class(p: &Pat) -> PatClass {
    match p {
        Pat::Ctor(v, _) if v.len() == 1 && v[0] == "some" => PatClass::Some,
        Pat::Lit(Expr::None(_)) => PatClass::None,
        Pat::Path(v) if v.len() == 1 && v[0] == "none" => PatClass::None,
        _ => PatClass::Any,
    }
}

fn int_of(v: &V) -> R<i128> {
    match v {
        V::Int(n, _) => Ok(*n),
        o => trap(format!("expected an integer, got {}", o.show())),
    }
}

fn marker_int<'a>(it: &mut Interp<'a>, s: Option<&Stmt>) -> R<i128> {
    match s {
        Some(Stmt::Expr(e)) => {
            let v = it.eval(e)?;
            int_of(&v)
        }
        _ => trap("missing operand"),
    }
}

impl<'a> Interp<'a> {
    /// Charge `units` cost units (C.6.4); traps with `trap("budget")` when
    /// starting the operation would take any enclosing budget past its limit.
    pub fn charge(&mut self, units: i128) -> R<()> {
        if self.cx.budgets.iter().any(|(s, l)| s.saturating_add(units) > *l) {
            return trap("budget");
        }
        for b in self.cx.budgets.iter_mut() {
            b.0 = b.0.saturating_add(units);
        }
        self.cx.spent = self.cx.spent.saturating_add(units);
        Ok(())
    }

    /// A copy of this evaluator for running a `par` branch: same store, no
    /// link to a scheduler (branches have no communication, W-PAR-2).
    fn fork_copy(&self) -> Interp<'a> {
        let mut c = Interp::new(self.m);
        c.scopes = self.scopes.clone();
        c.fuel = self.fuel;
        c.loop_limit = self.loop_limit;
        c.gens = self.gens.clone();
        c.cx.budgets = self.cx.budgets.clone();
        c.cx.spent = self.cx.spent;
        c.cx.par_mode = self.cx.par_mode;
        c.cx.id = self.cx.id.clone();
        c.cx.now = self.cx.now;
        c
    }

    /// Statements encoded by the parser as marker regions (`$par`, ...).
    pub(crate) fn exec_marker(
        &mut self,
        name: &str,
        body: &[Stmt],
        _line: usize,
        tail: bool,
    ) -> R<Option<V>> {
        match name {
            "$par" => self.par_exec(body).map(|_| None),
            "$forpar" => self.forpar_exec(body).map(|_| None),
            "$budget" => {
                // with budget n Block [else [as e] Else] (C.5). Atomicity: a
                // trapped block is all or nothing, so on trap("budget") every
                // write of Block is undone before Else runs, and Else sees
                // the state of block entry, which a sequential run also
                // reaches. The work done stays charged to every enclosing
                // budget (C.6.1). `as e` binds Exhausted{limit, used}, used
                // being the units this frame spent before the trapping
                // operation. W-BUD-1 keeps Block free of effects that cannot
                // be undone (emit, channels, spawn, lvar puts, foreign calls).
                let n = marker_int(self, body.first())?;
                let (rest, else_b) = match body.last() {
                    Some(Stmt::If(arms, _)) if arms.len() == 1
                        && matches!(&arms[0].0, Expr::Bool(false, _)) =>
                        (&body[1..body.len() - 1], Some(&arms[0].1[..])),
                    _ => (&body[1..], None),
                };
                let snap = else_b.map(|_| self.scopes.clone());
                self.cx.budgets.push((0, n));
                let r = self.exec_block(rest);
                let used = self.cx.budgets.last().map(|(s, _)| *s).unwrap_or(0);
                self.cx.budgets.pop();
                match (r, else_b, snap) {
                    (Err(Flow::Trap(t)), Some(eb), Some(snap)) if t == "budget" => {
                        self.scopes = snap;
                        let mut hidden = HashMap::new();
                        hidden.insert("\u{1}__bn".to_string(), V::Int(n, "u64"));
                        hidden.insert("\u{1}__bu".to_string(), V::Int(used, "u64"));
                        self.scopes.push(hidden);
                        let r = self.exec_block(eb);
                        self.scopes.pop();
                        r.map(|_| None)
                    }
                    (r, _, _) => r.map(|_| None),
                }
            }
            "$deadline" => {
                let n = marker_int(self, body.first())?;
                let fid = self.cx.next_frame;
                self.cx.next_frame += 1;
                let exp = self.cx.now.saturating_add(n.max(0) as u64);
                self.cx.dl.push((fid, exp));
                let r = match body.get(1) {
                    Some(Stmt::Arena(_, _, b, _)) => self.exec_block(b),
                    _ => Ok(()),
                };
                self.cx.dl.pop();
                match r {
                    Err(Flow::Cancel(f)) if f == fid => {
                        if let Some(Stmt::If(arms, _)) = body.get(2) {
                            self.exec_block(&arms[0].1)?;
                        }
                        Ok(None)
                    }
                    r => r.map(|_| None),
                }
            }
            "$nondet" => match body.first() {
                Some(Stmt::Select(arms, l)) => self.conc_select(arms, *l, true, tail),
                _ => trap("bad nondet"),
            },
            "$priority" => {
                let p = marker_int(self, body.first())?;
                match body.get(1) {
                    Some(Stmt::Spawn(e)) => self.conc_spawn(e, p).map(|_| None),
                    _ => trap("bad priority"),
                }
            }
            n if n.starts_with("$spawnas:") => {
                let h = &n["$spawnas:".len()..];
                let id = match body.first() {
                    Some(Stmt::Spawn(e)) => self.conc_spawn(e, 0)?,
                    Some(Stmt::Arena(m, _, b, _)) if m == "$priority" => {
                        let p = marker_int(self, b.first())?;
                        match b.get(1) {
                            Some(Stmt::Spawn(e)) => self.conc_spawn(e, p)?,
                            _ => return trap("bad spawn"),
                        }
                    }
                    _ => return trap("bad spawn"),
                };
                self.bind(h, V::Task(id));
                Ok(None)
            }
            "$scope" => {
                let sid = self.cx.next_frame;
                self.cx.next_frame += 1;
                self.cx.scopes.push(sid);
                let r = self.exec_block_tail(body);
                self.cx.scopes.pop();
                let v = r?;
                self.await_scope(sid)?;
                Ok(if tail { v } else { None })
            }
            "$atomic" => {
                let outer = self.cx.atomic.replace(vec![]);
                let r = self.exec_block(body);
                let buf = std::mem::replace(&mut self.cx.atomic, outer).unwrap_or_default();
                r?;
                if let Some(o) = self.cx.atomic.as_mut() {
                    o.extend(buf);
                } else if self.cx.link.is_some() {
                    self.comm(Op::Atomic(buf))?;
                } else {
                    return trap("concurrency is not available at comptime");
                }
                Ok(None)
            }
            n if n.starts_with("$group:") => self.group_exec(&n["$group:".len()..], body).map(|_| None),
            "$barrier" => trap("barrier outside a group"),
            _ => self.exec_block_tail(body).map(|v| if tail { v } else { None }),
        }
    }

    // ---- par (C.4)
    fn par_exec(&mut self, branches: &[Stmt]) -> R<()> {
        let bodies: Vec<&[Stmt]> = branches
            .iter()
            .filter_map(|b| match b {
                Stmt::Arena(_, _, x, _) => Some(&x[..]),
                _ => None,
            })
            .collect();
        self.charge(C_FORK * bodies.len() as i128)?;
        let waits = bodies.iter().any(|b| has_wait(b));
        let mode = if waits { ParMode::Threads } else { self.cx.par_mode };
        if mode == ParMode::Seq {
            for b in &bodies {
                self.exec_block(b)?;
            }
            return Ok(());
        }
        let k = bodies.len();
        let mut results: Vec<Option<(R<()>, Interp<'a>)>> = (0..k).map(|_| None).collect();
        if let ParMode::Permute(seed) = mode {
            let mut order: Vec<usize> = (0..k).collect();
            let mut x = seed | 1;
            for i in (1..k).rev() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                order.swap(i, (x % (i as u64 + 1)) as usize);
            }
            for i in order {
                let mut c = self.fork_copy();
                let r = c.exec_block(bodies[i]);
                results[i] = Some((r, c));
            }
        } else {
            let ctx = Arc::new(ParCtx::new(k, self.cx.par_ctx.clone()));
            let copies: Vec<Interp<'a>> = (0..k)
                .map(|_| {
                    let mut c = self.fork_copy();
                    c.cx.par_ctx = Some(ctx.clone());
                    c
                })
                .collect();
            let outs: Vec<(R<()>, Interp<'a>)> = std::thread::scope(|s| {
                let hs: Vec<_> = copies
                    .into_iter()
                    .zip(bodies.iter().copied())
                    .map(|(mut c, b)| {
                        let ctx = ctx.clone();
                        std::thread::Builder::new()
                            .stack_size(256 << 20)
                            .spawn_scoped(s, move || {
                                let r = c.exec_block(b);
                                let mut g = ctx.m.lock().unwrap();
                                g.running -= 1;
                                ctx.cv.notify_all();
                                drop(g);
                                (r, c)
                            })
                            .expect("thread")
                    })
                    .collect();
                hs.into_iter()
                    .map(|h| h.join().expect("par branch panicked"))
                    .collect()
            });
            for (i, o) in outs.into_iter().enumerate() {
                results[i] = Some(o);
            }
        }
        // resolve in source order: the first failing branch decides (E-Par),
        // with budget exhaustion placed where the sequential run would hit it.
        let rem = self.cx.budgets.iter().map(|(s, l)| l - s).min();
        let start = self.cx.spent;
        let fuel0 = self.fuel;
        let mut fcum = 0u64;
        let mut cum = 0i128;
        let orig: Vec<_> = self.scopes.clone();
        let mut writes: Vec<(usize, String, V)> = vec![];
        for r in results.into_iter() {
            let (r, c) = r.expect("branch result");
            let cost = c.cx.spent - start;
            // Sum-checked fuel account (C.6.3): each branch ran on a copy of
            // the fuel left at the fork; charge them back in source order and
            // trap at the branch whose cumulative use passes the fork's fuel,
            // which is where the sequential run traps, whatever the schedule.
            let used = fuel0.saturating_sub(c.fuel);
            fcum = fcum.saturating_add(used);
            self.fuel = fuel0.saturating_sub(fcum);
            if fcum > fuel0 || matches!(&r, Err(Flow::Trap(t)) if t == "comptime fuel exhausted") {
                self.fuel = 0;
                return trap("comptime fuel exhausted");
            }
            if let Some(rem) = rem
                && (matches!(&r, Err(Flow::Trap(t)) if t == "budget") || cum + cost > rem)
            {
                return trap("budget");
            }
            r?;
            cum += cost;
            for (fi, fr) in self.scopes.iter().enumerate() {
                for (n, v) in &c.scopes[fi] {
                    if let Some(old) = fr.get(n)
                        && old != v
                    {
                        writes.push((fi, n.clone(), v.clone()));
                    }
                }
            }
        }
        // W-PAR-1 makes branch write paths disjoint (C.3), so per-variable
        // results are combined by a three-way merge against the fork point:
        // fields and elements a branch left unchanged keep the other's write.
        for (fi, n, v) in writes {
            let base = orig[fi].get(&n).cloned().unwrap_or(V::Unit);
            let cur = self.scopes[fi].get(&n).cloned().unwrap_or(V::Unit);
            self.scopes[fi].insert(n, merge3(&base, &cur, v));
        }
        for b in self.cx.budgets.iter_mut() {
            b.0 += cum;
        }
        self.cx.spent += cum;
        Ok(())
    }

    // ---- for ... par (C.5.2)
    fn forpar_exec(&mut self, body: &[Stmt]) -> R<()> {
        let Some(Stmt::For(p, it, b, _)) = body.first() else {
            return trap("bad for par");
        };
        let mut targets = vec![];
        assign_ops(b, &mut targets);
        let local = declared_in(b);
        targets.retain(|(n, _)| !local.contains(n));
        let itv = self.eval(it)?;
        let items = self.iter_items(itv)?;
        if items.len() as i128 > self.loop_limit {
            return trap("loop bound");
        }
        let before: Vec<V> = targets
            .iter()
            .map(|(n, _)| self.get(n).ok_or_else(|| Flow::Trap(format!("unbound {n}"))))
            .collect::<R<_>>()?;
        let ids: Vec<V> = targets
            .iter()
            .zip(&before)
            .map(|((_, op), v)| ident(op, v))
            .collect::<R<_>>()?;
        let mut contrib: Vec<Vec<V>> = vec![vec![]; targets.len()];
        let mut res = Ok(());
        for x in items {
            for ((n, _), id) in targets.iter().zip(&ids) {
                self.set(n, id.clone());
            }
            self.scopes.push(HashMap::new());
            let r = match self.pmatch(p, &x) {
                Ok(_) => self.exec_block(b),
                Err(e) => Err(e),
            };
            self.scopes.pop();
            if let Err(e) = r {
                res = Err(e);
                break;
            }
            for (k, (n, _)) in targets.iter().enumerate() {
                contrib[k].push(self.get(n).unwrap_or(V::Unit));
            }
        }
        for ((n, _), v) in targets.iter().zip(&before) {
            self.set(n, v.clone());
        }
        res?;
        for (k, (n, op)) in targets.iter().enumerate() {
            let sym = &op[..op.len() - 1];
            let t = self.tree(sym, &contrib[k], &ids[k])?;
            let v = self.combine(sym, before[k].clone(), t)?;
            self.set(n, v);
        }
        Ok(())
    }

    fn combine(&mut self, sym: &str, a: V, b: V) -> R<V> {
        self.charge(op_cost(sym, &a))?;
        let t = if let V::Int(_, t) = &a { *t } else { "lit" };
        let v = match (sym, &a, &b) {
            ("&", V::Bool(x), V::Bool(y)) => V::Bool(*x && *y),
            ("|", V::Bool(x), V::Bool(y)) => V::Bool(*x || *y),
            ("^", V::Bool(x), V::Bool(y)) => V::Bool(*x != *y),
            _ => self.arith(sym, a, b)?,
        };
        Ok(match v {
            V::Int(x, _) if t != "lit" => chk(x, t)?,
            v => v,
        })
    }

    /// The fixed tree T of C.8: the left part is the largest power of two
    /// below m (`host::tree_split`), shared with the placement executor.
    pub fn tree(&mut self, sym: &str, c: &[V], id: &V) -> R<V> {
        match c.len() {
            0 => Ok(id.clone()),
            1 => Ok(c[0].clone()),
            m => {
                let h = tree_split(m);
                let l = self.tree(sym, &c[..h], id)?;
                let r = self.tree(sym, &c[h..], id)?;
                self.combine(sym, l, r)
            }
        }
    }

    // ---- workgroups (C.11.2)
    fn group_exec(&mut self, g: &str, body: &[Stmt]) -> R<()> {
        let n = marker_int(self, body.first())?;
        let stmts = &body[1..];
        let mut phases: Vec<Vec<&Stmt>> = vec![vec![]];
        for s in stmts {
            if matches!(s, Stmt::Arena(m, _, _, _) if m == "$barrier") {
                phases.push(vec![]);
            } else {
                phases.last_mut().unwrap().push(s);
            }
        }
        let mut targets = vec![];
        assign_ops(stmts, &mut targets);
        let local = declared_in(stmts);
        targets.retain(|(t, _)| !local.contains(t));
        let before: Vec<V> = targets
            .iter()
            .map(|(t, _)| self.get(t).ok_or_else(|| Flow::Trap(format!("unbound {t}"))))
            .collect::<R<_>>()?;
        let ids: Vec<V> = targets
            .iter()
            .zip(&before)
            .map(|((_, op), v)| ident(op, v))
            .collect::<R<_>>()?;
        let lanes = n.max(0) as usize;
        let mut acc: Vec<Vec<V>> = (0..lanes).map(|_| ids.clone()).collect();
        self.scopes.push(HashMap::new()); // group-shared memory
        let shared = self.scopes.len() - 1;
        let mut res = Ok(());
        'ph: for ph in &phases {
            for (lane, acc_l) in acc.iter_mut().enumerate() {
                for ((t, _), v) in targets.iter().zip(acc_l.iter()) {
                    self.set(t, v.clone());
                }
                let mut lf = HashMap::new();
                lf.insert(
                    g.to_string(),
                    V::Rec(
                        "$group".into(),
                        vec![
                            ("lane".into(), V::Int(lane as i128, "usize")),
                            ("lanes".into(), V::Int(n, "usize")),
                        ]
                        .into(),
                    ),
                );
                self.scopes.push(lf);
                for s in ph {
                    let top_decl = matches!(s, Stmt::Var(..) | Stmt::Let(..));
                    if top_decl && lane > 0 {
                        continue;
                    }
                    if let Err(e) = self.exec(s, false) {
                        res = Err(e);
                        break;
                    }
                    if top_decl {
                        let lf = self.scopes.last_mut().unwrap();
                        let moved: Vec<(String, V)> = lf
                            .iter()
                            .filter(|(k, _)| k.as_str() != g)
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();
                        for (k, _) in &moved {
                            lf.remove(k);
                        }
                        self.scopes[shared].extend(moved);
                    }
                }
                self.scopes.pop();
                for (k, (t, _)) in targets.iter().enumerate() {
                    acc_l[k] = self.get(t).unwrap_or(V::Unit);
                }
                if res.is_err() {
                    break 'ph;
                }
            }
        }
        self.scopes.truncate(shared);
        for ((t, _), v) in targets.iter().zip(&before) {
            self.set(t, v.clone());
        }
        res?;
        for (k, (t, op)) in targets.iter().enumerate() {
            let sym = &op[..op.len() - 1];
            let col: Vec<V> = acc.iter().map(|a| a[k].clone()).collect();
            let tv = self.tree(sym, &col, &ids[k])?;
            let v = self.combine(sym, before[k].clone(), tv)?;
            self.set(t, v);
        }
        Ok(())
    }

    // ---- pscan (C.11.6): Blelloch up-sweep / down-sweep over the tree of C.5.2
    fn pscan(&mut self, xs: &[V], f: &V, e: &V, inclusive: bool) -> R<V> {
        fn up(
            it: &mut Interp<'_>,
            xs: &[V],
            lo: usize,
            hi: usize,
            f: &V,
            sums: &mut HashMap<(usize, usize), V>,
        ) -> R<V> {
            let v = if hi - lo == 1 {
                xs[lo].clone()
            } else {
                let h = lo + tree_split(hi - lo);
                let l = up(it, xs, lo, h, f, sums)?;
                let r = up(it, xs, h, hi, f, sums)?;
                it.apply(f, vec![l, r])?
            };
            sums.insert((lo, hi), v.clone());
            Ok(v)
        }
        fn down(
            it: &mut Interp<'_>,
            (lo, hi): (usize, usize),
            off: Option<V>,
            f: &V,
            sums: &HashMap<(usize, usize), V>,
            out: &mut Vec<Option<V>>,
        ) -> R<()> {
            if hi - lo == 1 {
                out[lo] = off;
                return Ok(());
            }
            let h = lo + tree_split(hi - lo);
            let ls = sums[&(lo, h)].clone();
            let roff = match &off {
                None => ls,
                Some(o) => it.apply(f, vec![o.clone(), ls])?,
            };
            down(it, (lo, h), off, f, sums, out)?;
            down(it, (h, hi), Some(roff), f, sums, out)
        }
        let n = xs.len();
        if n == 0 {
            return Ok(V::List(vec![].into()));
        }
        let mut sums = HashMap::new();
        up(self, xs, 0, n, f, &mut sums)?;
        let mut out = vec![None; n];
        down(self, (0, n), None, f, &sums, &mut out)?;
        let mut r = vec![];
        for (i, o) in out.into_iter().enumerate() {
            r.push(if inclusive {
                match o {
                    None => xs[i].clone(),
                    Some(o) => self.apply(f, vec![o, xs[i].clone()])?,
                }
            } else {
                o.unwrap_or_else(|| e.clone())
            });
        }
        Ok(V::List(r.into()))
    }

    /// merge through a user `impl Merge` or the built-in instances (C.11.5)
    pub fn merge_values(&mut self, a: &V, b: &V) -> R<V> {
        if let V::Rec(n, _) = a {
            let h = self.kind_of(n);
            if let Some(fd) = self.methods.get(&(h, "merge".to_string())).copied() {
                let saved = std::mem::replace(&mut self.scopes, vec![HashMap::new()]);
                self.bind("$a", a.clone());
                self.bind("$b", b.clone());
                let args = [Arg {
                    label: None,
                    e: Expr::Name("$b".into(), 0),
                }];
                let obj = Expr::Name("$a".into(), 0);
                let r = self.call_fn(fd, &args, Some((a.clone(), &obj)));
                self.scopes = saved;
                return r;
            }
        }
        merge_builtin(a, b)
    }

    fn lvar_put(&mut self, c: &Arc<LCell>, v: V) -> R<()> {
        let changed = loop {
            let cur = c.peek();
            let nv = self.merge_values(&cur, &v)?;
            let mut g = c.v.lock().unwrap();
            if *g != cur {
                continue; // another branch put in between: merge again
            }
            if let Some(top) = &c.top
                && nv == *top
            {
                return trap("lvar conflict");
            }
            let changed = *g != nv;
            // a put that would change a frozen lvar is an error (POPL 2014);
            // one that changes nothing is allowed
            if changed && c.frozen() {
                return trap("lvar put after freeze");
            }
            *g = nv;
            break changed;
        };
        if changed && let Some(ctx) = &self.cx.par_ctx {
            ctx.bump();
        }
        Ok(())
    }

    /// `merge` for a threshold test: the program's own `Merge` (the order
    /// `put` grows the value in), charged nothing, so the cost of a `wait`
    /// does not depend on how often a schedule rechecks it
    fn merge_free(&mut self, a: &V, b: &V) -> R<V> {
        let saved = (self.fuel, std::mem::take(&mut self.cx.budgets), self.cx.spent);
        self.fuel = crate::interp::fuel_limit();
        let r = self.merge_values(a, b);
        self.fuel = saved.0;
        self.cx.budgets = saved.1;
        self.cx.spent = saved.2;
        r
    }

    /// `t ⊑ cur` in the program's lattice: `t ⊔ cur = cur`
    fn lvar_above(&mut self, ths: &[V], cur: &V) -> R<Option<V>> {
        for t in ths {
            if self.merge_free(t, cur)? == *cur {
                return Ok(Some(t.clone()));
            }
        }
        Ok(None)
    }

    /// W-LVAR-2 at run time: a threshold set of more than one element must
    /// be pairwise incompatible, `a ⊔ b = top` for distinct a, b (the
    /// `incomp(Q)` premise of FHPC 2013 E-GETVAL). The test does not read
    /// the cell's value, so its outcome does not depend on the schedule.
    fn lvar_incomp(&mut self, c: &LCell, ths: &[V]) -> R<()> {
        for (i, a) in ths.iter().enumerate() {
            for b in &ths[i + 1..] {
                if a == b {
                    continue;
                }
                let ok = match &c.top {
                    Some(top) => self.merge_free(a, b)? == *top,
                    None => false,
                };
                if !ok {
                    return trap("lvar thresholds are not pairwise incompatible");
                }
            }
        }
        Ok(())
    }

    fn lvar_wait(&mut self, c: &Arc<LCell>, q: &V) -> R<V> {
        let V::List(ths) = q else {
            return trap("wait needs a list of thresholds");
        };
        let ths = ths.clone();
        self.lvar_incomp(c, &ths)?;
        let Some(ctx) = self.cx.par_ctx.clone() else {
            let cur = c.peek();
            return match self.lvar_above(&ths, &cur)? {
                Some(t) => Ok(t),
                None => trap("lvar wait cannot be satisfied"),
            };
        };
        let mut g = ctx.m.lock().unwrap();
        let mut seen: Option<u64> = None;
        loop {
            if g.stuck {
                return trap("lvar wait cannot be satisfied");
            }
            if seen != Some(g.ver) {
                // a put bumps `ver` after writing the cell, so a check at
                // this `ver` sees every put it counts
                let cur = c.peek();
                if let Some(t) = self.lvar_above(&ths, &cur)? {
                    return Ok(t);
                }
                seen = Some(g.ver);
                g.idle += 1;
            }
            if g.idle >= g.running {
                g.stuck = true;
                ctx.cv.notify_all();
                return trap("lvar wait cannot be satisfied");
            }
            g = ctx.cv.wait(g).unwrap();
        }
    }

    /// `c.choose(Label)` reads its argument as a label, before evaluation.
    pub(crate) fn conc_builtin_pre(&mut self, recv: &V, m: &str, args: &[Arg]) -> R<Option<V>> {
        if let (V::Session(out, _, cap), "choose") = (recv, m) {
            let l = match args.first().map(|a| &a.e) {
                Some(Expr::Name(n, _)) => n.clone(),
                _ => return trap("choose needs a label"),
            };
            self.charge(C_CHAN)?;
            self.comm(Op::Send(V::Chan(out.clone(), *cap), V::Rec(l, vec![].into())))?;
            return Ok(Some(V::Unit));
        }
        Ok(None)
    }

    /// Methods on the 0.6b runtime values, and `pscan`.
    pub(crate) fn conc_builtin(&mut self, _obj: &Expr, recv: &V, m: &str, a: &[V]) -> R<Option<V>> {
        Ok(Some(match (recv, m) {
            (V::Chan(..), "close") => {
                self.comm(Op::Close(recv.clone()))?;
                V::Unit
            }
            (V::Task(id), "join") => match self.comm(Op::Join(id.clone()))? {
                Rep::Val(v) => v,
                _ => V::Unit,
            },
            (V::Session(_, inn, cap), "offer") => {
                self.charge(C_CHAN)?;
                match self.comm(Op::Recv(V::Chan(inn.clone(), *cap)))? {
                    Rep::Val(V::Opt(Some(v))) => match v.take() {
                        V::Rec(l, f) if f.is_empty() => V::Rec(format!("$label:{l}"), f),
                        o => o,
                    },
                    _ => return trap("session closed"),
                }
            }
            (V::Session(out, _, cap), "close") => {
                self.comm(Op::Close(V::Chan(out.clone(), *cap)))?;
                V::Unit
            }
            (V::LVar(c), "put") => {
                self.lvar_put(c, a.first().cloned().unwrap_or(V::Unit))?;
                V::Unit
            }
            (V::LVar(c), "wait") => self.lvar_wait(c, a.first().unwrap_or(&V::Unit))?,
            (V::LVar(c), "freeze") => c.freeze(),
            (V::List(xs), "pscan" | "pscan_incl") if a.len() == 2 => {
                self.pscan(xs, &a[0], &a[1], m == "pscan_incl")?
            }
            _ => return Ok(None),
        }))
    }

    // ---- the task side of the tick runtime
    pub(crate) fn comm(&mut self, op: Op) -> R<Rep> {
        let Some(link) = self.cx.link.as_ref() else {
            return trap("concurrency is not available at comptime");
        };
        link.tx
            .send(Req {
                op,
                dl: self.cx.dl.clone(),
            })
            .map_err(|_| Flow::Trap("runtime stopped".into()))?;
        let RepMsg { tick, rep } = link
            .rx
            .recv()
            .map_err(|_| Flow::Trap("runtime stopped".into()))?;
        self.cx.now = tick;
        match rep {
            Rep::Cancel(f) => Err(Flow::Cancel(f)),
            Rep::Failed(v) => Err(Flow::Fail(v)),
            r => Ok(r),
        }
    }

    pub(crate) fn await_scope(&mut self, sid: u64) -> R<()> {
        if self.cx.link.is_none() {
            return Ok(());
        }
        self.comm(Op::AwaitScope(sid)).map(|_| ())
    }

    pub(crate) fn conc_emit(&mut self, v: V) -> R<()> {
        if self.cx.link.is_some() && self.cx.handler_sends.is_none() {
            self.comm(Op::Emit(v))?;
        } else {
            self.outputs.push(v);
        }
        Ok(())
    }

    pub(crate) fn conc_send(&mut self, t: &Expr, v: &Expr) -> R<()> {
        let tv = self.eval(t)?;
        let vv = self.eval(v)?;
        self.charge(C_CHAN)?;
        let tv = match tv {
            V::Session(out, _, cap) => V::Chan(out, cap),
            o => o,
        };
        if let Some(b) = self.cx.atomic.as_mut() {
            b.push((tv, vv));
        } else if let Some(b) = self.cx.handler_sends.as_mut() {
            b.push((tv, vv));
        } else {
            self.comm(Op::Send(tv, vv))?;
        }
        Ok(())
    }

    pub(crate) fn conc_recv(&mut self, c: &Expr) -> R<V> {
        let cv = self.eval(c)?;
        self.charge(C_CHAN)?;
        let (ch, session) = match cv {
            V::Session(_, inn, cap) => (V::Chan(inn, cap), true),
            o => (o, false),
        };
        match self.comm(Op::Recv(ch))? {
            Rep::Val(V::Opt(Some(v))) if session => Ok(v.take()),
            Rep::Val(_) if session => trap("session closed"),
            Rep::Val(v) => Ok(v),
            _ => trap("bad receive"),
        }
    }

    pub(crate) fn conc_spawn(&mut self, e: &Expr, prio: i128) -> R<String> {
        let Expr::Call(f, args, line, _) = e else {
            return trap("spawn needs a call");
        };
        let Expr::Name(fname, _) = &**f else {
            return trap("spawn needs a function name");
        };
        let vals: Vec<V> = args.iter().map(|a| self.eval(&a.e)).collect::<R<_>>()?;
        self.charge(C_SPAWN)?;
        self.cx.created += 1;
        let idx = ord_next(&mut self.cx.ords, "spawn");
        let scope = self.cx.scopes.last().copied().unwrap_or(0);
        match self.comm(Op::Spawn {
            f: fname.clone(),
            args: vals,
            prio: prio.clamp(0, 255) as u8,
            site: *line,
            idx,
            scope,
        })? {
            Rep::Spawned(id) => Ok(id),
            _ => trap("bad spawn"),
        }
    }

    pub(crate) fn conc_select(
        &mut self,
        arms: &[SelArm],
        line: usize,
        nondet: bool,
        tail: bool,
    ) -> R<Option<V>> {
        let mut sarms = vec![];
        for a in arms {
            sarms.push(match a {
                SelArm::Recv(p, c, _) => {
                    let cv = match self.eval(c)? {
                        V::Session(_, inn, cap) => V::Chan(inn, cap),
                        o => o,
                    };
                    SArm::Recv(cv, pat_class(p))
                }
                SelArm::Send(v, c, _) => {
                    let vv = self.eval(v)?;
                    let cv = match self.eval(c)? {
                        V::Session(out, _, cap) => V::Chan(out, cap),
                        o => o,
                    };
                    SArm::Send(cv, vv)
                }
                SelArm::After(n, _) => {
                    let v = self.eval(n)?;
                    SArm::After(int_of(&v)?.max(0) as u64)
                }
                SelArm::Default(_) => SArm::Default,
            });
        }
        let k = {
            let e = self.cx.sel_k.entry(line).or_insert(0);
            let k = *e;
            *e += 1;
            k
        };
        let (i, v) = match self.comm(Op::Select {
            arms: sarms,
            key: (line, k),
            nondet,
        })? {
            Rep::Arm(i, v) => (i, v),
            _ => return trap("bad select"),
        };
        let (pat, body) = match &arms[i] {
            SelArm::Recv(p, _, b) => (Some(p), b),
            SelArm::Send(_, _, b) | SelArm::After(_, b) | SelArm::Default(b) => (None, b),
        };
        self.scopes.push(HashMap::new());
        if let Some(p) = pat {
            let val = v.unwrap_or(V::Opt(None));
            match self.pmatch(p, &val) {
                Ok(true) => {}
                Ok(false) => {
                    self.scopes.pop();
                    return trap("select pattern did not match");
                }
                Err(e) => {
                    self.scopes.pop();
                    return Err(e);
                }
            }
        }
        let r = self.exec_block_tail(body);
        self.scopes.pop();
        r.map(|v| if tail { v } else { None })
    }

    pub(crate) fn chan_new(&mut self, t: &Ty, cap: Option<&Expr>, line: usize) -> R<V> {
        if self.cx.link.is_none() {
            return trap("channels are not available at comptime");
        }
        let n = match cap {
            Some(e) => {
                let v = self.eval(e)?;
                int_of(&v)?
            }
            None => 0,
        };
        let fresh = |me: &mut Self| {
            let idx = me.cx.created;
            me.cx.created += 1;
            hid(&format!("chan|{line}|{}|{idx}", me.cx.id))
        };
        if let Ty::Path(p, _) = t
            && p.len() == 1
            && p[0] == "session"
        {
            let a = fresh(self);
            let b = fresh(self);
            return Ok(V::Tuple(vec![
                V::Session(a.clone(), b.clone(), n),
                V::Session(b, a, n),
            ].into()));
        }
        Ok(V::Chan(fresh(self), n))
    }

    pub(crate) fn actor_new(&mut self, ty: &str, line: usize) -> R<V> {
        if self.cx.link.is_none() {
            return trap(format!("cannot call {ty} at comptime"));
        }
        let _ = line;
        self.cx.created += 1;
        let pos = actor_pos(self.m, ty);
        let idx = ord_next(&mut self.cx.ords, &format!("actor|{pos}"));
        Ok(V::Actor(stable_actor_id(&pos, &self.cx.id, idx), ty.to_string()))
    }

    /// Call `fd` on argument values (task bodies, tests).
    pub fn call_values(&mut self, fd: &'a FnDecl, vals: Vec<V>) -> R<V> {
        let mut fr = HashMap::new();
        let mut args = vec![];
        for (i, v) in vals.into_iter().enumerate() {
            let n = format!("$a{i}");
            fr.insert(n.clone(), v);
            args.push(Arg {
                label: None,
                e: Expr::Name(n, 0),
            });
        }
        let saved = std::mem::replace(&mut self.scopes, vec![fr]);
        let r = self.call_fn(fd, &args, None);
        self.scopes = saved;
        r
    }
}

// ---------------------------------------------------------------- scheduler

/// Configuration of a run of `fn main` (inputs other than the program).
#[derive(Clone, Debug)]
pub struct RunCfg {
    /// seed of the environment's choices for `select nondet` (C.5)
    pub seed: u64,
    /// replay: the arm indices of a previous run's `Choice` entries
    pub choices: Option<Vec<usize>>,
    /// logged inputs (tick, true = Partition / false = Heal, node, node)
    pub links: Vec<(u64, bool, String, String)>,
    pub max_ticks: u64,
    pub par_mode: ParMode,
    /// replay: the supervision decisions of a previous run (`RunResult.
    /// restarts`). A restart is an input (C.11.11): replay applies the
    /// logged decision and does not consult the supervisor's policy
    pub restarts: Option<Vec<RestartIn>>,
    /// schema of the replayed log (C.11.11): the supervisor tree it was
    /// recorded against (`RunResult.tree`). A log whose tree does not
    /// migrate to this program's tree (`tree_migrates`) is reported before any
    /// step instead of being misreplayed. `None` is a log without a schema
    /// header: each replayed decision's position is still checked.
    pub restart_tree: Option<Vec<String>>,
    /// program events fed to the running task (HC-6): (tick, event value).
    /// At its tick an event is logged as `Input(..)` and delivered to every
    /// live actor whose type has an `on` handler for it, in actor-id order;
    /// with no such actor it is logged as `Unhandled(..)`, never dropped
    /// silently.
    pub events: Vec<(u64, V)>,
}

/// A supervision decision, logged as an input of the run (C.11.11): at a
/// crash of actor `id` (a child of `sup`), either restart the actors in
/// `reset` to their initializers, or stop the supervisor.
#[derive(Clone, Debug, PartialEq)]
pub struct RestartIn {
    pub tick: u64,
    pub sup: String,
    pub child: String,
    pub id: String,
    pub reset: Vec<String>,
    pub stop: bool,
}
impl Default for RunCfg {
    fn default() -> Self {
        RunCfg {
            seed: 0x5eed,
            choices: None,
            links: vec![],
            max_ticks: 100_000,
            par_mode: ParMode::Seq,
            restarts: None,
            restart_tree: None,
            events: vec![],
        }
    }
}

/// One entry of the log D of a run, with its vector timestamp (C.11.4).
#[derive(Clone, Debug)]
pub struct Entry {
    pub tick: u64,
    pub node: usize,
    pub what: String,
    pub vc: Vec<u64>,
    /// for a delivery: the index of the entry that sent the message
    pub cause: Option<usize>,
    /// in a traced run (`run_main_traced`), the state of the run when the
    /// entry was written, as the canonical JSON of its positional form
    /// (10.9): `[tick, [[actor id, [field values]], ...]]`, the live actors
    /// in id order, each field value positional (6.1, T7), in declaration
    /// order; `None` in an untraced run
    pub state: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Done,
    Stopped(String),
    Deadlock,
}

#[derive(Debug)]
pub struct RunResult {
    pub outputs: Vec<V>,
    pub outcome: Outcome,
    pub log: Vec<Entry>,
    pub choices: Vec<usize>,
    /// the supervision decisions taken, in order: the run's restart inputs
    pub restarts: Vec<RestartIn>,
    /// the supervisor tree of the program (`sup_tree`): the schema a
    /// restart log of this run is recorded against
    pub tree: Vec<String>,
    pub ticks: u64,
    pub nodes: Vec<String>,
}

#[derive(Clone)]
struct Msg {
    v: V,
    vc: Vec<u64>,
    entry: usize,
}
#[derive(Default)]
struct ChanSt {
    cap: i128,
    buf: VecDeque<Msg>,
    held: VecDeque<(Msg, Option<usize>)>,
    closed: bool,
}
struct ActorSt {
    ty: String,
    fields: Vec<(String, V, Ty)>,
    mailbox: VecDeque<Msg>,
    node: usize,
}
enum St {
    Ready(Rep),
    Pending(Req),
    Done,
}
struct Slot {
    id: String,
    prio: u8,
    rx: mpsc::Receiver<Req>,
    tx: mpsc::Sender<RepMsg>,
    st: St,
    parent: Option<usize>,
    scope: u64,
    result: Option<Result<V, String>>,
    joined: bool,
    cancel: bool,
    holding: Option<String>,
    released: bool,
    since: u64,
}

type HandlerOut = Option<(Vec<(String, V, Ty)>, Vec<V>, Vec<(V, V)>)>;

struct Sched<'a> {
    m: &'a Module,
    it: Interp<'a>,
    cfg: RunCfg,
    slots: Vec<Slot>,
    chans: BTreeMap<String, ChanSt>,
    staged: Vec<(String, Msg)>,
    actors: BTreeMap<String, ActorSt>,
    mail_staged: Vec<(String, String, Msg)>,
    txns: Vec<Vec<(String, String, Msg)>>,
    txn_staged: Vec<Vec<(String, String, Msg)>>,
    held_links: Vec<(usize, usize, V, Msg)>,
    parted: Vec<(usize, usize)>,
    nodes: Vec<String>,
    node_of: HashMap<String, usize>,
    vcs: Vec<Vec<u64>>,
    log: Vec<Entry>,
    outputs: Vec<V>,
    choices: Vec<usize>,
    replay: VecDeque<usize>,
    rng: u64,
    tick: u64,
    /// restart ticks per (supervisor, child), for the intensity window (C.11.11)
    restarts: HashMap<(String, String), Vec<u64>>,
    restart_log: Vec<RestartIn>,
    restart_replay: Option<VecDeque<RestartIn>>,
    stop: Option<Outcome>,
    /// record the state of the run with every entry (`Entry::state`)
    trace: bool,
    /// set when a task thread could not be started (`NO_TASK_THREADS`)
    no_threads: Option<String>,
}

/// The start of the failure of a run of `fn main` on a host that cannot
/// start a task thread (10.3): WASI preview 1 (`wasm32-wasip1`) has no
/// threads, and the tick runtime runs each task on its own thread.
pub const NO_TASK_THREADS: &str = "fn main needs task threads, which this host does not provide";

/// Whether an error of `run_main` is the failure to start a task thread.
pub fn is_no_task_threads(e: &str) -> bool {
    e.starts_with(NO_TASK_THREADS)
}

thread_local! {
    /// how many more task threads a run started from this thread may start;
    /// `None`: no limit but the host's (`refuse_task_threads_after`)
    static TASK_THREADS_LEFT: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Let runs of `fn main` started from the calling thread start at most `n`
/// task threads, and fail to start any further one as a host without
/// threads does; `None` lifts the limit. Tests use it to check the failure
/// of `NO_TASK_THREADS` on a host that has threads.
pub fn refuse_task_threads_after(n: Option<usize>) {
    TASK_THREADS_LEFT.with(|c| c.set(n));
}

fn sup(a: &mut [u64], b: &[u64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x = (*x).max(*y);
    }
}

impl<'a> Sched<'a> {
    fn new(m: &'a Module, cfg: &RunCfg) -> Self {
        let mut nodes = vec!["main".to_string()];
        let mut node_of = HashMap::new();
        for i in &m.items {
            if let Item::Node { name, actors, .. } = i {
                nodes.push(name.clone());
                for a in actors {
                    node_of.insert(a.clone(), nodes.len() - 1);
                }
            }
        }
        let k = nodes.len();
        Sched {
            m,
            it: Interp::new(m),
            cfg: cfg.clone(),
            slots: vec![],
            chans: BTreeMap::new(),
            staged: vec![],
            actors: BTreeMap::new(),
            mail_staged: vec![],
            txns: vec![],
            txn_staged: vec![],
            held_links: vec![],
            parted: vec![],
            nodes,
            node_of,
            vcs: vec![vec![0; k]; k],
            log: vec![],
            outputs: vec![],
            choices: vec![],
            replay: cfg.choices.clone().unwrap_or_default().into(),
            rng: cfg.seed | 1,
            tick: 0,
            restarts: HashMap::new(),
            restart_log: vec![],
            restart_replay: cfg.restarts.clone().map(|v| v.into()),
            stop: match (&cfg.restarts, &cfg.restart_tree) {
                (Some(_), Some(rec)) => tree_migrates(rec, &sup_tree(m)).err().map(Outcome::Stopped),
                _ => None,
            },
            trace: false,
            no_threads: None,
        }
    }

    /// The positional state of the run now (`Entry::state`).
    fn state_json(&self) -> String {
        use crate::interp::{FV, form_value_pos};
        let actors = self
            .actors
            .iter()
            .map(|(id, a)| {
                let fs = a.fields.iter().map(|(_, v, _)| form_value_pos(v, self.m)).collect();
                FV::List(vec![FV::Text(id.clone()), FV::List(fs)])
            })
            .collect();
        FV::List(vec![FV::Int(self.tick as i128), FV::List(actors)]).json()
    }

    fn local(&mut self, node: usize, what: String) -> usize {
        self.vcs[node][node] += 1;
        let state = self.trace.then(|| self.state_json());
        self.log.push(Entry {
            tick: self.tick,
            node,
            what,
            vc: self.vcs[node].clone(),
            cause: None,
            state,
        });
        self.log.len() - 1
    }
    fn deliver_entry(&mut self, node: usize, what: String, m: &Msg) -> usize {
        sup(&mut self.vcs[node], &m.vc);
        self.vcs[node][node] += 1;
        let state = self.trace.then(|| self.state_json());
        self.log.push(Entry {
            tick: self.tick,
            node,
            what,
            vc: self.vcs[node].clone(),
            cause: Some(m.entry),
            state,
        });
        self.log.len() - 1
    }
    fn msg(&mut self, node: usize, what: String, v: V) -> Msg {
        let e = self.local(node, what);
        Msg {
            v,
            vc: self.log[e].vc.clone(),
            entry: e,
        }
    }

    /// Start the thread of a task. When the host cannot start it, the error
    /// is `NO_TASK_THREADS`, the host and the reason.
    #[allow(clippy::too_many_arguments)]
    fn spawn_task<'s, 'e>(
        &mut self,
        s: &'s std::thread::Scope<'s, 'e>,
        fd: &'a FnDecl,
        args: Vec<V>,
        id: String,
        prio: u8,
        parent: Option<usize>,
        scope: u64,
    ) -> Result<(), String>
    where
        'a: 's,
    {
        let refused = TASK_THREADS_LEFT.with(|c| match c.get() {
            Some(0) => true,
            Some(n) => {
                c.set(Some(n - 1));
                false
            }
            None => false,
        });
        let (req_tx, req_rx) = mpsc::channel::<Req>();
        let (rep_tx, rep_rx) = mpsc::channel::<RepMsg>();
        let m = self.m;
        let pm = self.cfg.par_mode;
        let tid = id.clone();
        let started = if refused {
            Err(std::io::Error::from(std::io::ErrorKind::Unsupported))
        } else {
            std::thread::Builder::new().stack_size(256 << 20).spawn_scoped(s, move || {
                let mut it = Interp::new(m);
                it.cx.id = tid;
                it.cx.par_mode = pm;
                let tx = req_tx.clone();
                it.cx.link = Some(Link {
                    tx: req_tx,
                    rx: rep_rx,
                });
                let first = it.cx.link.as_ref().unwrap().rx.recv();
                let r = match first {
                    Ok(RepMsg { tick, rep: Rep::Go }) => {
                        it.cx.now = tick;
                        it.cx.scopes = vec![0];
                        match it.call_values(fd, args) {
                            Ok(v) => it.await_scope(0).map(|_| v),
                            Err(e) => Err(e),
                        }
                    }
                    _ => Err(Flow::Cancel(CANCEL_TASK)),
                };
                let _ = tx.send(Req {
                    op: Op::Done(r),
                    dl: vec![],
                });
            })
        };
        if let Err(e) = started {
            // the target's name: on WASI `std::env::consts::OS` is empty
            let host = if cfg!(all(target_os = "wasi", target_env = "p1")) {
                "wasm32-wasip1".to_string()
            } else if cfg!(target_os = "wasi") {
                "wasm32-wasi".to_string()
            } else {
                format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
            };
            return Err(format!("{NO_TASK_THREADS} ({host}: {e})"));
        }
        self.slots.push(Slot {
            id,
            prio,
            rx: req_rx,
            tx: rep_tx,
            st: St::Ready(Rep::Go),
            parent,
            scope,
            result: None,
            joined: false,
            cancel: false,
            holding: None,
            released: false,
            since: 0,
        });
        Ok(())
    }

    fn order(&self) -> Vec<usize> {
        let mut v: Vec<usize> = (0..self.slots.len())
            .filter(|&i| !matches!(self.slots[i].st, St::Done))
            .collect();
        v.sort_by(|&a, &b| {
            let (x, y) = (&self.slots[a], &self.slots[b]);
            y.prio.cmp(&x.prio).then_with(|| x.id.cmp(&y.id))
        });
        v
    }

    fn chan(&mut self, v: &V) -> Option<String> {
        if let V::Chan(id, cap) = v {
            self.chans.entry(id.clone()).or_insert_with(|| ChanSt {
                cap: *cap,
                ..Default::default()
            });
            Some(id.clone())
        } else {
            None
        }
    }
    fn space(&self, c: &str) -> bool {
        let st = &self.chans[c];
        let staged = self.staged.iter().filter(|(x, _)| x == c).count() as i128;
        (st.buf.len() as i128) + staged < st.cap
    }
    /// take a value from channel c (buffer first, then a held send).
    fn take(&mut self, c: &str) -> Option<Msg> {
        let st = self.chans.get_mut(c)?;
        if let Some(m) = st.buf.pop_front() {
            if let Some((h, owner)) = st.held.pop_front() {
                st.buf.push_back(h);
                if let Some(o) = owner {
                    self.slots[o].released = true;
                }
            }
            return Some(m);
        }
        if let Some((h, owner)) = st.held.pop_front() {
            if let Some(o) = owner {
                self.slots[o].released = true;
            }
            return Some(h);
        }
        None
    }
    fn readable(&self, c: &str) -> (bool, bool) {
        let st = &self.chans[c];
        let has = !st.buf.is_empty() || !st.held.is_empty();
        (has, !has && st.closed)
    }

    fn route(&mut self, from: usize, target: V, m: Msg) {
        let to = match &target {
            V::Actor(_, ty) => self.node_of.get(ty).copied().unwrap_or(0),
            _ => 0,
        };
        let key = (from.min(to), from.max(to));
        if from != to && self.parted.contains(&key) {
            let what = format!("Held({} -> {})", self.nodes[from], self.nodes[to]);
            self.local(from, what);
            self.held_links.push((from, to, target, m));
            return;
        }
        match target {
            V::Actor(id, ty) => self.mail_staged.push((id, ty, m)),
            V::Chan(..) => {
                if let Some(c) = self.chan(&target) {
                    if self.space(&c) {
                        self.staged.push((c, m));
                    } else {
                        self.chans.get_mut(&c).unwrap().held.push_back((m, None));
                    }
                }
            }
            _ => {}
        }
    }

    fn task_failed_value(&self, i: usize) -> V {
        let reason = match &self.slots[i].result {
            Some(Err(r)) => r.clone(),
            _ => String::new(),
        };
        V::Rec(
            "TaskFailed".into(),
            vec![
                ("task".into(), V::Text(self.slots[i].id.clone().into())),
                ("reason".into(), V::Text(reason.into())),
            ]
            .into(),
        )
    }

    fn pick_select(&mut self, i: usize, arms: &[SArm], key: (usize, u64), nondet: bool) -> Option<usize> {
        let mut ready = vec![];
        for (k, a) in arms.iter().enumerate() {
            let ok = match a {
                SArm::Recv(cv, cls) => match self.chan(cv) {
                    Some(c) => {
                        let (has, closed) = self.readable(&c);
                        (has && *cls != PatClass::None) || (closed && *cls != PatClass::Some)
                    }
                    None => false,
                },
                SArm::Send(cv, _) => match self.chan(cv) {
                    Some(c) => self.space(&c) && !self.chans[&c].closed,
                    None => false,
                },
                _ => false,
            };
            if ok {
                ready.push(k);
            }
        }
        if ready.is_empty() {
            return None;
        }
        if !nondet {
            let n = arms.len();
            let s = (key.1 % n as u64) as usize;
            return (0..n).map(|j| (s + j) % n).find(|j| ready.contains(j));
        }
        let arm = match self.replay.pop_front() {
            Some(a) if ready.contains(&a) => a,
            Some(_) => {
                self.stop = Some(Outcome::Stopped("replay diverged".into()));
                ready[0]
            }
            None => {
                self.rng ^= self.rng << 13;
                self.rng ^= self.rng >> 7;
                self.rng ^= self.rng << 17;
                ready[(self.rng % ready.len() as u64) as usize]
            }
        };
        self.choices.push(arm);
        let what = format!(
            "Choice(task {}, select line {}, execution {}, arm {arm})",
            self.slots[i].id, key.0, key.1
        );
        self.local(0, what);
        Some(arm)
    }

    /// Try to complete task i's pending operation in this tick.
    fn handle(&mut self, i: usize, req: &Req) -> Option<Rep> {
        // cancellation checkpoint (C.6.2): the outermost expired deadline
        if let Some((fid, _)) = req.dl.iter().find(|(_, e)| self.tick > *e) {
            if let Some(c) = self.slots[i].holding.take()
                && let Some(st) = self.chans.get_mut(&c)
            {
                st.held.retain(|(_, o)| *o != Some(i));
            }
            self.slots[i].released = false;
            self.slots[i].since = 0;
            let what = format!("Cancelled(task {})", self.slots[i].id);
            self.local(0, what);
            return Some(Rep::Cancel(*fid));
        }
        if self.slots[i].cancel {
            return Some(Rep::Cancel(CANCEL_TASK));
        }
        match &req.op {
            Op::Emit(v) => {
                self.local(0, format!("Emit({})", v.show()));
                self.outputs.push(v.clone());
                Some(Rep::Val(V::Unit))
            }
            Op::Send(target, v) => {
                if let V::Actor(..) = target {
                    let m = self.msg(0, format!("Send({})", v.show()), v.clone());
                    self.route(0, target.clone(), m);
                    return Some(Rep::Val(V::Unit));
                }
                let c = self.chan(target)?;
                if self.slots[i].holding.is_some() {
                    if self.slots[i].released {
                        self.slots[i].released = false;
                        self.slots[i].holding = None;
                        return Some(Rep::Val(V::Unit));
                    }
                    return None;
                }
                if self.chans[&c].closed {
                    self.stop = Some(Outcome::Stopped("trap: send on a closed channel".into()));
                    return Some(Rep::Cancel(CANCEL_TASK));
                }
                let m = self.msg(0, format!("Send({})", v.show()), v.clone());
                if self.space(&c) {
                    self.staged.push((c, m));
                    Some(Rep::Val(V::Unit))
                } else {
                    self.local(0, "Backpressure".into());
                    self.chans.get_mut(&c).unwrap().held.push_back((m, Some(i)));
                    self.slots[i].holding = Some(c);
                    None
                }
            }
            Op::Recv(cv) => {
                let c = self.chan(cv)?;
                let (has, closed) = self.readable(&c);
                if has {
                    let m = self.take(&c).unwrap();
                    self.deliver_entry(0, format!("Recv({})", m.v.show()), &m);
                    Some(Rep::Val(V::Opt(Some(Shared::from(m.v)))))
                } else if closed {
                    self.local(0, "Recv(none)".into());
                    Some(Rep::Val(V::Opt(None)))
                } else {
                    None
                }
            }
            Op::Close(cv) => {
                let c = self.chan(cv)?;
                self.chans.get_mut(&c).unwrap().closed = true;
                self.local(0, "Close".into());
                Some(Rep::Val(V::Unit))
            }
            Op::Select { arms, key, nondet } => {
                if let Some(k) = self.pick_select(i, arms, *key, *nondet) {
                    self.slots[i].since = 0;
                    return Some(match &arms[k] {
                        SArm::Recv(cv, _) => {
                            let c = self.chan(cv).unwrap();
                            match self.take(&c) {
                                Some(m) => {
                                    self.deliver_entry(0, format!("Recv({})", m.v.show()), &m);
                                    Rep::Arm(k, Some(V::Opt(Some(Shared::from(m.v)))))
                                }
                                None => {
                                    self.local(0, "Recv(none)".into());
                                    Rep::Arm(k, Some(V::Opt(None)))
                                }
                            }
                        }
                        SArm::Send(cv, v) => {
                            let c = self.chan(cv).unwrap();
                            let m = self.msg(0, format!("Send({})", v.show()), v.clone());
                            self.staged.push((c, m));
                            Rep::Arm(k, None)
                        }
                        _ => Rep::Arm(k, None),
                    });
                }
                if let Some(d) = arms.iter().position(|a| matches!(a, SArm::Default)) {
                    return Some(Rep::Arm(d, None));
                }
                if let Some((k, n)) = arms.iter().enumerate().find_map(|(k, a)| match a {
                    SArm::After(n) => Some((k, *n)),
                    _ => None,
                }) {
                    if self.slots[i].since == 0 {
                        self.slots[i].since = self.tick;
                    }
                    if self.tick - self.slots[i].since >= n {
                        self.slots[i].since = 0;
                        return Some(Rep::Arm(k, None));
                    }
                }
                None
            }
            Op::Spawn { idx, .. } => {
                // the thread itself is started by `run` (it needs the thread scope)
                let id = stable_spawn_id(&self.slots[i].id, *idx);
                Some(Rep::Spawned(id))
            }
            Op::Join(id) => {
                let j = self.slots.iter().position(|s| &s.id == id)?;
                if !matches!(self.slots[j].st, St::Done) {
                    return None;
                }
                self.slots[j].joined = true;
                Some(Rep::Val(match &self.slots[j].result {
                    Some(Ok(v)) => v.clone(),
                    _ => V::Err(Shared::from(self.task_failed_value(j))),
                }))
            }
            Op::AwaitScope(sid) => {
                let kids: Vec<usize> = (0..self.slots.len())
                    .filter(|&k| self.slots[k].parent == Some(i) && self.slots[k].scope == *sid)
                    .collect();
                if kids.iter().any(|&k| !matches!(self.slots[k].st, St::Done)) {
                    return None;
                }
                for &k in &kids {
                    if matches!(self.slots[k].result, Some(Err(_))) && !self.slots[k].joined {
                        self.slots[k].joined = true;
                        return Some(Rep::Failed(self.task_failed_value(k)));
                    }
                }
                Some(Rep::Val(V::Unit))
            }
            Op::Atomic(sends) => {
                let chans: Vec<String> = sends.iter().filter_map(|(t, _)| self.chan(t)).collect();
                for c in &chans {
                    let need = chans.iter().filter(|x| *x == c).count() as i128;
                    let st = &self.chans[c];
                    let staged = self.staged.iter().filter(|(x, _)| x == c).count() as i128;
                    if st.closed || st.buf.len() as i128 + staged + need > st.cap {
                        return None;
                    }
                }
                let mut txn = vec![];
                for (t, v) in sends {
                    let m = self.msg(0, format!("Send({})", v.show()), v.clone());
                    match t {
                        V::Actor(id, ty) => txn.push((id.clone(), ty.clone(), m)),
                        _ => {
                            let c = self.chan(t).unwrap();
                            self.staged.push((c, m));
                        }
                    }
                }
                if !txn.is_empty() {
                    self.txn_staged.push(txn);
                }
                self.local(0, "Atomic".into());
                Some(Rep::Val(V::Unit))
            }
            Op::Done(_) => None,
        }
    }

    fn actor_init(&mut self, ty: &str) -> Vec<(String, V, Ty)> {
        let mut fs = vec![];
        for i in &self.m.items {
            if let Item::Actor { name, fields, .. } = i
                && name == ty
            {
                for f in fields {
                    self.it.scopes = vec![HashMap::new()];
                    let v = match &f.init {
                        Some(e) => self.it.eval(e).unwrap_or(V::Unit),
                        None => V::Unit,
                    };
                    let v = self.it.fit(v, &f.ty).unwrap_or(V::Unit);
                    fs.push((f.name.clone(), v, f.ty.clone()));
                }
            }
        }
        fs
    }

    /// Run actor `id`'s handler on `msg` (6.4 on the actor's region).
    fn handler(&mut self, id: &str, msg: &V) -> Result<HandlerOut, String> {
        let m = self.m;
        let st = &self.actors[id];
        let ty = st.ty.clone();
        let fields = st.fields.clone();
        let Some(Item::Actor { handlers, .. }) = m
            .items
            .iter()
            .find(|i| matches!(i, Item::Actor { name, .. } if *name == ty))
        else {
            return Ok(None);
        };
        let it = &mut self.it;
        let reason_of = |f: Flow| match f {
            Flow::Fail(V::Rec(n, _)) => n,
            Flow::Trap(t) => format!("trap({t})"),
            o => format!("trap({o:?})"),
        };
        for h in handlers {
            it.scopes = vec![
                fields.iter().map(|(n, v, _)| (n.clone(), v.clone())).collect(),
                HashMap::new(),
            ];
            it.outputs.clear();
            it.cx.handler_sends = Some(vec![]);
            match it.pmatch(&h.pat, msg) {
                Ok(true) => {}
                Ok(false) => continue,
                Err(f) => return Err(reason_of(f)),
            }
            if let Some(g) = &h.guard {
                match it.eval(g).and_then(|v| it.truthy(v)) {
                    Ok(true) => {}
                    Ok(false) => continue,
                    Err(f) => return Err(reason_of(f)),
                }
            }
            let r = match &h.body {
                Body::Arrow(e) => it.eval(e).map(|v| {
                    if let Some((n, _, _)) = fields.first() {
                        it.scopes[0].insert(n.clone(), v);
                    }
                }),
                Body::Block(b) => it.exec_block(b),
            };
            let sends = it.cx.handler_sends.take().unwrap_or_default();
            r.map_err(reason_of)?;
            let mut out = vec![];
            for (n, _, t) in &fields {
                let v = it.scopes[0][n].clone();
                out.push((n.clone(), it.fit(v, t).map_err(reason_of)?, t.clone()));
            }
            return Ok(Some((out, std::mem::take(&mut it.outputs), sends)));
        }
        Ok(None)
    }

    fn supervisor_of(&self, ty: &str) -> Option<(String, String, Vec<String>, String, u32)> {
        for i in &self.m.items {
            if let Item::Supervisor {
                name,
                strategy,
                children,
                ..
            } = i
                && let Some(c) = children.iter().find(|c| c.ty == ty)
            {
                let tys = children.iter().map(|c| c.ty.clone()).collect();
                return Some((name.clone(), strategy.clone(), tys, c.name.clone(), c.restart));
            }
        }
        None
    }

    /// The restart window W in ticks: `const RESTART_WINDOW: u64 = W` when
    /// declared, else unbounded (every restart of the run counts).
    fn restart_window(&self) -> Option<u64> {
        self.m.items.iter().find_map(|i| match i {
            Item::Const { name, e: Expr::Int(n, _), .. } if name == "RESTART_WINDOW" && *n > 0 => Some(*n as u64),
            _ => None,
        })
    }

    fn on_crash(&mut self, id: &str, reason: &str) {
        let ty = self.actors[id].ty.clone();
        let node = self.actors[id].node;
        self.local(node, format!("Kept({reason})"));
        let Some((sup, strat, tys, child, bound)) = self.supervisor_of(&ty) else {
            return;
        };
        // Restart as input. The state of a supervised run is the fold of its
        // log, so the decision taken at a crash (which actors are reset, or
        // that the supervisor stops) is an input of the run: recorded here
        // as a RestartIn, and on replay taken from the log instead of the
        // policy, so a log replays under the decisions it records even if
        // the policy (strategy, bound, RESTART_WINDOW) has changed since.
        let tick = self.tick;
        let dec = if let Some(q) = self.restart_replay.as_mut() {
            match q.pop_front() {
                Some(r) if !sup_tree(self.m).contains(&format!("{}/{}", r.sup, r.child)) => {
                    self.stop = Some(Outcome::Stopped(format!(
                        "trap: upgrade removed supervisor position {}/{}",
                        r.sup, r.child
                    )));
                    return;
                }
                Some(r) if r.id == id && r.sup == sup => r,
                _ => {
                    self.stop = Some(Outcome::Stopped("replay diverged".into()));
                    return;
                }
            }
        } else {
            // OTP-style intensity, measured in logical ticks: more than
            // `bound` restarts of this child within the last W ticks stops
            // the supervisor
            let win = self.restart_window();
            let hist = self.restarts.entry((sup.clone(), child.clone())).or_default();
            if let Some(w) = win {
                hist.retain(|&t| tick.saturating_sub(t) < w);
            }
            hist.push(tick);
            let stop = hist.len() as u32 > bound;
            let pos = tys.iter().position(|t| *t == ty).unwrap_or(0);
            let reset: Vec<String> = if stop {
                vec![]
            } else {
                self.actors
                    .iter()
                    .filter(|(aid, a)| match strat.as_str() {
                        "one_for_all" => tys.contains(&a.ty),
                        "rest_for_one" => tys.iter().position(|t| *t == a.ty).is_some_and(|p| p >= pos),
                        _ => aid.as_str() == id,
                    })
                    .map(|(aid, _)| aid.clone())
                    .collect()
            };
            RestartIn { tick, sup: sup.clone(), child: child.clone(), id: id.to_string(), reset, stop }
        };
        self.restart_log.push(dec.clone());
        // k: this child's decisions in the intensity window (all of the run
        // without a window), the one just taken included
        let win = self.restart_window();
        let count = self
            .restart_log
            .iter()
            .filter(|r| r.sup == sup && r.child == child && win.is_none_or(|w| tick.saturating_sub(r.tick) < w))
            .count();
        if dec.stop {
            self.local(node, format!("Stopped(supervisor {sup})"));
            self.stop = Some(Outcome::Stopped(format!(
                "supervisor {sup} stopped: child {child} restarted more than {bound} times"
            )));
            return;
        }
        // the log entry carries the instance id (a content hash, 6.6)
        self.local(node, format!("Restart({child}, {id}, {reason}, tick {tick}, {count})"));
        for aid in dec.reset {
            if let Some(t) = self.actors.get(&aid).map(|a| a.ty.clone()) {
                let init = self.actor_init(&t);
                self.actors.get_mut(&aid).unwrap().fields = init;
            }
        }
    }

    fn actors_phase(&mut self) -> bool {
        let mut progress = false;
        let mut busy = std::collections::BTreeSet::new();
        for txn in std::mem::take(&mut self.txns) {
            progress = true;
            let mut results = vec![];
            let mut kept = None;
            for (id, ty, m) in &txn {
                self.ensure_actor(id, ty);
                let node = self.actors[id].node;
                self.deliver_entry(node, format!("Deliver({})", m.v.show()), m);
                busy.insert(id.clone());
                match self.handler(id, &m.v) {
                    Ok(r) => results.push((id.clone(), r)),
                    Err(e) => {
                        kept = Some(e);
                        break;
                    }
                }
            }
            if let Some(e) = kept {
                // all or none (T30): no handler of the batch commits; the
                // actor whose handler failed is then supervised like any crash
                let failed = results.len();
                for (k, (id, _, _)) in txn.iter().enumerate() {
                    if k == failed {
                        self.on_crash(id, &e);
                    } else {
                        let node = self.actors.get(id).map(|a| a.node).unwrap_or(0);
                        self.local(node, format!("Kept({e})"));
                    }
                }
                if self.stop.is_some() {
                    break;
                }
                continue;
            }
            for (id, r) in results {
                self.commit_handler(&id, r);
            }
        }
        let ids: Vec<String> = self.actors.keys().cloned().collect();
        for id in ids {
            if busy.contains(&id) {
                continue;
            }
            let Some(m) = self.actors.get_mut(&id).and_then(|a| a.mailbox.pop_front()) else {
                continue;
            };
            progress = true;
            let node = self.actors[&id].node;
            self.deliver_entry(node, format!("Deliver({})", m.v.show()), &m);
            match self.handler(&id, &m.v) {
                Ok(r) => self.commit_handler(&id, r),
                Err(e) => self.on_crash(&id, &e),
            }
            if self.stop.is_some() {
                break;
            }
        }
        progress
    }

    fn commit_handler(&mut self, id: &str, r: HandlerOut) {
        let Some((fs, outs, sends)) = r else { return };
        let node = self.actors[id].node;
        self.actors.get_mut(id).unwrap().fields = fs;
        for o in outs {
            self.local(node, format!("Emit({})", o.show()));
            self.outputs.push(o);
        }
        for (t, v) in sends {
            let m = self.msg(node, format!("Send({})", v.show()), v);
            self.route(node, t, m);
        }
    }

    fn ensure_actor(&mut self, id: &str, ty: &str) {
        if !self.actors.contains_key(id) {
            let fields = self.actor_init(ty);
            let node = self.node_of.get(ty).copied().unwrap_or(0);
            self.actors.insert(
                id.to_string(),
                ActorSt {
                    ty: ty.to_string(),
                    fields,
                    mailbox: VecDeque::new(),
                    node,
                },
            );
        }
    }

    fn end_of_tick(&mut self) {
        for (c, m) in std::mem::take(&mut self.staged) {
            self.chans.get_mut(&c).unwrap().buf.push_back(m);
        }
        for (id, ty, m) in std::mem::take(&mut self.mail_staged) {
            self.ensure_actor(&id, &ty);
            self.actors.get_mut(&id).unwrap().mailbox.push_back(m);
        }
        let t = std::mem::take(&mut self.txn_staged);
        self.txns.extend(t);
    }

    /// HC-6: program events at this tick enter D as `Input(..)` and reach
    /// the actors that handle them. An input event has no addressee, so its
    /// receivers are the program's handlers of it, as the steps `on E` of a
    /// Form are; the order among receivers is the actor id order (6.6).
    fn events_phase(&mut self) -> bool {
        let evs: Vec<V> = self.cfg.events.iter().filter(|e| e.0 == self.tick).map(|e| e.1.clone()).collect();
        let mut any = false;
        for v in evs {
            any = true;
            let name = match &v {
                V::Rec(n, _) => n.clone(),
                _ => String::new(),
            };
            let entry = self.local(0, format!("Input({})", v.show()));
            let m = self.m;
            let handles = |ty: &str| {
                m.items.iter().any(|i| match i {
                    Item::Actor { name: an, handlers, .. } if an == ty => handlers.iter().any(|h| match &h.pat {
                        Pat::Ctor(p, _) | Pat::Path(p) => p.last().is_some_and(|x| *x == name),
                        _ => false,
                    }),
                    _ => false,
                })
            };
            let to: Vec<(String, String)> = self
                .actors
                .iter()
                .filter(|(_, a)| handles(&a.ty))
                .map(|(id, a)| (id.clone(), a.ty.clone()))
                .collect();
            if to.is_empty() {
                self.local(0, format!("Unhandled({})", v.show()));
            }
            for (id, ty) in to {
                let msg = Msg { v: v.clone(), vc: self.vcs[0].clone(), entry };
                self.mail_staged.push((id, ty, msg));
            }
        }
        any
    }

    fn links_phase(&mut self) -> bool {
        let mut any = false;
        let evs: Vec<(u64, bool, String, String)> =
            self.cfg.links.iter().filter(|e| e.0 == self.tick).cloned().collect();
        for (_, part, a, b) in evs {
            let (Some(x), Some(y)) = (
                self.nodes.iter().position(|n| *n == a),
                self.nodes.iter().position(|n| *n == b),
            ) else {
                continue;
            };
            any = true;
            let key = (x.min(y), x.max(y));
            if part {
                self.local(x, format!("Partition({a}, {b})"));
                if !self.parted.contains(&key) {
                    self.parted.push(key);
                }
            } else {
                self.local(x, format!("Heal({a}, {b})"));
                self.parted.retain(|k| *k != key);
                let held = std::mem::take(&mut self.held_links);
                for (f, t, target, m) in held {
                    if (f.min(t), f.max(t)) == key {
                        self.route(f, target, m);
                    } else {
                        self.held_links.push((f, t, target, m));
                    }
                }
            }
        }
        any
    }

    fn finish_task(&mut self, i: usize, r: &Result<V, Flow>) -> Option<Outcome> {
        let res = match r {
            Ok(v) => Ok(v.clone()),
            Err(Flow::Cancel(_)) => Err("cancelled".to_string()),
            Err(f) => Err(crate::interp::why(f.clone())),
        };
        let failed = matches!(r, Err(f) if !matches!(f, Flow::Cancel(_)));
        self.slots[i].result = Some(res.clone());
        let what = format!("Done(task {})", self.slots[i].id);
        self.local(0, what);
        if i == 0 {
            if let Err(e) = res {
                return Some(Outcome::Stopped(e));
            }
        } else if failed {
            // a failed task cancels its siblings in the same scope (C.11.9)
            let (p, sc) = (self.slots[i].parent, self.slots[i].scope);
            for k in 0..self.slots.len() {
                if k != i && self.slots[k].parent == p && self.slots[k].scope == sc {
                    self.slots[k].cancel = true;
                }
            }
        }
        for k in 0..self.slots.len() {
            if self.slots[k].parent == Some(i) {
                self.slots[k].cancel = true;
            }
        }
        None
    }

    fn run<'s, 'e>(&mut self, s: &'s std::thread::Scope<'s, 'e>) -> Outcome
    where
        'a: 's,
    {
        if let Some(o) = self.stop.clone() {
            return o;
        }
        loop {
            self.tick += 1;
            if self.tick > self.cfg.max_ticks {
                return Outcome::Stopped("trap: tick limit".into());
            }
            let mut progress = self.links_phase();
            progress |= self.events_phase();
            let order = self.order();
            // move phase: each task that may move runs, on its own thread and
            // at the same time as the others, to its next communication step
            let mut moved = vec![];
            for &i in &order {
                if let St::Ready(_) = &self.slots[i].st {
                    let St::Ready(rep) = std::mem::replace(&mut self.slots[i].st, St::Done) else {
                        unreachable!()
                    };
                    let _ = self.slots[i].tx.send(RepMsg { tick: self.tick, rep });
                    moved.push(i);
                }
            }
            for &i in &moved {
                progress = true;
                match self.slots[i].rx.recv() {
                    Ok(req) => self.slots[i].st = St::Pending(req),
                    Err(_) => {
                        self.slots[i].st = St::Done;
                        self.slots[i].result = Some(Err("task stopped".into()));
                    }
                }
            }
            // actors handle one message each, admitted at an earlier tick
            progress |= self.actors_phase();
            if let Some(o) = self.stop.take() {
                return o;
            }
            // delivery phase, in task order (-priority, id); all effects staged
            for &i in &order {
                let St::Pending(req) = std::mem::replace(&mut self.slots[i].st, St::Done) else {
                    continue;
                };
                if let Op::Done(r) = &req.op {
                    progress = true;
                    if let Some(o) = self.finish_task(i, r) {
                        return o;
                    }
                    continue;
                }
                let rep = self.handle(i, &req);
                if let Some(o) = self.stop.take() {
                    return o;
                }
                let Some(rep) = rep else {
                    self.slots[i].st = St::Pending(req);
                    continue;
                };
                progress = true;
                if let (Op::Spawn { f, args, prio, scope, .. }, Rep::Spawned(id)) = (&req.op, &rep) {
                    let fd = self.m.items.iter().find_map(|it| match it {
                        Item::Fn(fd) if fd.sig.name == *f => Some(fd),
                        _ => None,
                    });
                    let Some(fd) = fd else {
                        return Outcome::Stopped(format!("trap: no task function {f}"));
                    };
                    self.local(0, format!("Spawn(task {id}, priority {prio})"));
                    if let Err(e) = self.spawn_task(s, fd, args.clone(), id.clone(), *prio, Some(i), *scope) {
                        // the spawning task is cancelled when the run is wound up
                        self.slots[i].st = St::Ready(rep);
                        self.no_threads = Some(e.clone());
                        return Outcome::Stopped(e);
                    }
                }
                self.slots[i].st = St::Ready(rep);
            }
            self.end_of_tick();
            let all_done = self.slots.iter().all(|s| matches!(s.st, St::Done));
            let actors_idle = self.actors.values().all(|a| a.mailbox.is_empty())
                && self.txns.is_empty()
                && self.mail_staged.is_empty();
            let future_links = self.cfg.links.iter().any(|e| e.0 > self.tick)
                || self.cfg.events.iter().any(|e| e.0 > self.tick);
            let future_events = self.cfg.events.iter().any(|e| e.0 > self.tick);
            if all_done && actors_idle && !future_events && (self.held_links.is_empty() || !future_links) {
                return Outcome::Done;
            }
            if !progress && actors_idle {
                let timers = self.slots.iter().any(|s| match &s.st {
                    St::Pending(r) => {
                        !r.dl.is_empty()
                            || matches!(&r.op, Op::Select { arms, .. }
                                if arms.iter().any(|a| matches!(a, SArm::After(_))))
                    }
                    _ => false,
                });
                if !timers && !future_links {
                    return Outcome::Deadlock;
                }
            }
        }
    }
}

/// Run `fn main()` of `m` on the tick runtime (6.8 and chapter C).
pub fn run_main(m: &Module, cfg: &RunCfg) -> Result<RunResult, String> {
    run_main_with(m, cfg, false)
}

/// `run_main`, with the state of the run recorded in every entry of its
/// log (`Entry::state`): the run of `fn main` in the reference trace (10.9).
pub fn run_main_traced(m: &Module, cfg: &RunCfg) -> Result<RunResult, String> {
    run_main_with(m, cfg, true)
}

fn run_main_with(m: &Module, cfg: &RunCfg, trace: bool) -> Result<RunResult, String> {
    let main = m
        .items
        .iter()
        .find_map(|i| match i {
            Item::Fn(f) if f.sig.name == "main" && f.sig.params.is_empty() => Some(f),
            _ => None,
        })
        .ok_or("nothing to run: declare `fn main()`")?;
    let (outcome, sch) = std::thread::scope(|s| {
        let mut sch = Sched::new(m, cfg);
        sch.trace = trace;
        let o = match sch.spawn_task(s, main, vec![], hid("main"), 0, None, 0) {
            Ok(()) => sch.run(s),
            Err(e) => {
                sch.no_threads = Some(e.clone());
                Outcome::Stopped(e)
            }
        };
        // stop every task thread that is still waiting
        let tick = sch.tick;
        for sl in sch.slots.iter_mut() {
            while !matches!(sl.st, St::Done) {
                if matches!(sl.st, St::Ready(_)) || matches!(sl.st, St::Pending(_)) {
                    if let St::Pending(Req { op: Op::Done(_), .. }) = sl.st {
                        sl.st = St::Done;
                        continue;
                    }
                    let _ = sl.tx.send(RepMsg {
                        tick,
                        rep: Rep::Cancel(CANCEL_TASK),
                    });
                }
                match sl.rx.recv() {
                    Ok(Req { op: Op::Done(_), .. }) | Err(_) => sl.st = St::Done,
                    Ok(r) => sl.st = St::Pending(r),
                }
            }
        }
        (o, sch)
    });
    if let Some(e) = sch.no_threads {
        return Err(e);
    }
    Ok(RunResult {
        outputs: sch.outputs,
        outcome,
        log: sch.log,
        choices: sch.choices,
        restarts: sch.restart_log,
        tree: sup_tree(m),
        ticks: sch.tick,
        nodes: sch.nodes,
    })
}

// ---------------------------------------------------------------- interrupts (C.11.7)

/// Admission order of pending events (C.11.7): interrupts by (-priority,
/// arrival), then ordinary events in arrival order. Each item is
/// (arrival index, Some(priority) for an interrupt event).
pub fn admission_order(pending: &[(usize, Option<u8>)]) -> Vec<usize> {
    let mut ints: Vec<&(usize, Option<u8>)> = pending.iter().filter(|p| p.1.is_some()).collect();
    ints.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut out: Vec<usize> = ints.into_iter().map(|p| p.0).collect();
    let mut ord: Vec<usize> = pending.iter().filter(|p| p.1.is_none()).map(|p| p.0).collect();
    ord.sort();
    out.extend(ord);
    out
}

// ---------------------------------------------------------------- real-time analysis (C.11.6)

/// Liu and Layland rate-monotonic sufficient test, tasks (C, T):
/// U <= m (2^(1/m) - 1).
pub fn rm_schedulable(tasks: &[(f64, f64)]) -> bool {
    if tasks.is_empty() {
        return true;
    }
    let m = tasks.len() as f64;
    let u: f64 = tasks.iter().map(|(c, t)| c / t).sum();
    u <= m * (2f64.powf(1.0 / m) - 1.0)
}
/// Earliest deadline first: schedulable iff U <= 1 (Liu and Layland).
pub fn edf_schedulable(tasks: &[(f64, f64)]) -> bool {
    tasks.iter().map(|(c, t)| c / t).sum::<f64>() <= 1.0
}

// ---------------------------------------------------------------- replicated state (C.11.4, T31)

/// Reference model of a `replicated state` (C.11.4). Events submitted at a
/// node enter one totally ordered log when the submitting node is on a
/// majority side; submissions on a minority side are held and retried when
/// the partition heals. Every node on the majority side has applied the whole
/// committed log; a node on a minority side keeps the prefix it had. The
/// language fixes this interface and State Machine Safety, not a consensus
/// protocol, so this is a model of the contract, not an implementation of
/// Raft or Paxos.
pub struct Replicas {
    pub n: usize,
    pub log: Vec<i64>,
    pub applied: Vec<usize>,
    pub held: Vec<(usize, i64)>,
    pub side: Vec<usize>,
}
impl Replicas {
    pub fn new(n: usize) -> Self {
        Replicas {
            n,
            log: vec![],
            applied: vec![0; n],
            held: vec![],
            side: vec![0; n],
        }
    }
    /// Split the nodes into sides (side[i] in 0..k); all equal = healed.
    pub fn partition(&mut self, side: Vec<usize>) {
        self.side = side;
        let h = std::mem::take(&mut self.held);
        for (at, ev) in h {
            self.submit(at, ev);
        }
        self.sync();
    }
    fn quorum(&self, at: usize) -> bool {
        let k = self.side.iter().filter(|s| **s == self.side[at]).count();
        2 * k > self.n
    }
    pub fn submit(&mut self, at: usize, ev: i64) {
        if self.quorum(at) {
            self.log.push(ev);
        } else {
            self.held.push((at, ev));
        }
        self.sync();
    }
    fn sync(&mut self) {
        for i in 0..self.n {
            if self.quorum(i) {
                self.applied[i] = self.log.len();
            }
        }
    }
    /// the state of node i: the fold of the log prefix it has applied
    pub fn state(&self, i: usize, f: impl Fn(i64, i64) -> i64, s0: i64) -> i64 {
        self.log[..self.applied[i]].iter().fold(s0, |s, e| f(s, *e))
    }
}

// ---------------------------------------------------------------- Merge laws (C.11.5, W-CRDT-1)

/// Admission check of every `impl Merge for T` (W-CRDT-1). The laws
/// (associativity, commutativity, idempotence, `bottom()` an identity) are
/// an obligation, discharged in one of three ways:
/// * T is a kind of nullary variants: every triple is evaluated, which
///   decides the laws (`MergeLaws.laws_of_enum`);
/// * `merge` is componentwise, `T(f: max(self.f, o.f), g: self.g or o.g, ...)`
///   over integer and bool fields: the laws follow from `MergeLaws.prodSL`,
///   `intMax` and `boolOr`;
/// * otherwise the obligation is not discharged and the impl requires change.
/// Before that, a search on a finite sample (`bottom()`, the module's
/// constants of T, closed under `merge` to depth 3, at most 24 values)
/// looks for a counterexample, so the report names one when it can.
/// Types without an `impl Merge` use the built-in instances, whose laws are
/// proved in `MergeLaws.lean`.
pub fn crdt_gate(m: &Module) -> Vec<String> {
    let mut errs = vec![];
    for it in &m.items {
        let Item::Impl {
            ty: Ty::Path(tp, _),
            trait_: Some(Ty::Path(tr, _)),
            line,
            members,
            ..
        } = it
        else {
            continue;
        };
        if tr.len() != 1 || tr[0] != "Merge" || tp.len() != 1 {
            continue;
        }
        let t = tp[0].clone();
        match merge_laws(m, &t, members) {
            Ok(()) => {}
            Err(e) => errs.push(format!("line {line}: W-CRDT-1 impl Merge for {t}: {e}")),
        }
    }
    errs.extend(lvar_gate(m));
    errs
}

fn merge_laws(m: &Module, t: &str, members: &[FnDecl]) -> Result<(), String> {
    let mut it = Interp::new(m);
    let w = |f: Flow| crate::interp::why(f);
    let Some(bot) = it.methods.get(&(t.to_string(), "bottom".to_string())).copied() else {
        return Err("needs `fn bottom() -> Self`".into());
    };
    if !it.methods.contains_key(&(t.to_string(), "merge".to_string())) {
        return Err("needs `fn merge(self, other: Self) -> Self`".into());
    }
    it.scopes = vec![HashMap::new()];
    let b = it.call_fn(bot, &[], None).map_err(w)?;
    let mut sample = vec![b.clone()];
    let mut exhaustive = false;
    for i in &m.items {
        if let Item::Kind { name, variants, .. } = i
            && name == t
            && variants.iter().all(|v| v.1.is_empty())
        {
            exhaustive = true;
            for (v, _) in variants {
                let x = V::Rec(v.clone(), vec![].into());
                if !sample.contains(&x) {
                    sample.push(x);
                }
            }
        }
    }
    if !exhaustive {
        for i in &m.items {
            if let Item::Const { name, .. } = i
                && let Ok(v) = it.const_value(name)
                && matches!(&v, V::Rec(n, _) if it.kind_of(n) == t)
                && !sample.contains(&v)
            {
                sample.push(v);
            }
        }
        for _ in 0..3 {
            let cur = sample.clone();
            for a in &cur {
                for c in &cur {
                    if sample.len() >= 24 {
                        break;
                    }
                    let x = it.merge_values(a, c).map_err(w)?;
                    if !sample.contains(&x) {
                        sample.push(x);
                    }
                }
            }
        }
    }
    let s = sample;
    // on an enumeration every triple is tried, which decides the laws
    let third = if exhaustive { s.len() } else { 12 };
    for a in &s {
        if it.merge_values(a, a).map_err(w)? != *a {
            return Err(format!("not idempotent at {}", a.show()));
        }
        if it.merge_values(a, &b).map_err(w)? != *a || it.merge_values(&b, a).map_err(w)? != *a {
            return Err(format!("bottom() is not the identity at {}", a.show()));
        }
        for c in &s {
            let ac = it.merge_values(a, c).map_err(w)?;
            if ac != it.merge_values(c, a).map_err(w)? {
                return Err(format!("not commutative at {}, {}", a.show(), c.show()));
            }
            for e in s.iter().take(third) {
                let l = it.merge_values(&ac, e).map_err(w)?;
                let ce = it.merge_values(c, e).map_err(w)?;
                if l != it.merge_values(a, &ce).map_err(w)? {
                    return Err(format!(
                        "not associative at {}, {}, {}",
                        a.show(),
                        c.show(),
                        e.show()
                    ));
                }
            }
        }
    }
    let _ = members;
    if exhaustive || merge_certs(m).contains_key(t) {
        return Ok(());
    }
    Err("no W-CRDT-1 certificate: the laws hold by construction only for a `merge` built from the proven combinators (per field: `max` or `min` on integers, `or` or `and` on bools, `.merge` of a field type with a certified impl; lexicographic on an integer key: `if self.k > o.k { self } else if o.k > self.k { o } else { T(k: self.k, ...) }`), or a kind of nullary variants (decided by enumeration); a counterexample-free sample is not a proof".into())
}

/// A W-CRDT-1 certificate: the `merge` body read as a term over proven
/// combinators. Every certificate denotes a join with the three laws by
/// construction (lean/RCE/MergeCert.lean, `Cert.semi`, `cert_laws`); the
/// checker's only job is to read the body as one, which is syntactic and
/// total. `Ref` is a field whose type has its own certified `impl Merge`.
#[derive(Clone, Debug, PartialEq)]
pub enum Cert {
    Union,
    Inter,
    Map(Box<Cert>),
    MaxI,
    MinI,
    OrB,
    AndB,
    Prod(Vec<(String, Cert)>),
    Lex(String, Box<Cert>),
    Ref(String),
    Enum(String),
}

impl Cert {
    /// the Lean term (`RCE.MergeCert.Cert`); `Ref` and `Enum` stand for the
    /// certificate of the named type (an enumeration is decided by
    /// `MergeLaws.laws_of_enum`, modelled as `unit` in the term)
    pub fn to_lean(&self, env: &dyn Fn(&str) -> Option<Cert>) -> String {
        match self {
            Cert::MaxI => ".maxI".into(),
            Cert::MinI => ".minI".into(),
            Cert::OrB => ".orB".into(),
            Cert::AndB => ".andB".into(),
            Cert::Enum(_) => ".unit".into(),
            Cert::Union => ".union".into(),
            Cert::Inter => ".inter".into(),
            Cert::Map(v) => format!("(.map {})", v.to_lean(env)),
            Cert::Ref(t) => env(t).map(|c| c.to_lean(env)).unwrap_or_else(|| ".unit".into()),
            Cert::Lex(_, b) => format!("(.lex {})", b.to_lean(env)),
            Cert::Prod(fs) => match fs.split_last() {
                None => ".unit".into(),
                Some((l, [])) => l.1.to_lean(env),
                Some((l, init)) => {
                    let mut t = l.1.to_lean(env);
                    for f in init.iter().rev() {
                        t = format!("(.prod {} {t})", f.1.to_lean(env));
                    }
                    t
                }
            },
        }
    }
}

/// a field type as a key: `i64`, `Set[i64]`, `Map[text,Set[i64]]`
fn ty_key(t: &Ty) -> Option<String> {
    let Ty::Path(p, g) = t else { return None };
    if p.len() != 1 {
        return None;
    }
    if g.is_empty() {
        return Some(p[0].clone());
    }
    let gs: Vec<String> = g.iter().map(ty_key).collect::<Option<_>>()?;
    Some(format!("{}[{}]", p[0], gs.join(",")))
}

/// `v` of `Map[k,v]` (split at the top-level comma)
fn map_val(t: &str) -> Option<&str> {
    let inner = t.strip_prefix("Map[")?.strip_suffix(']')?;
    let mut depth = 0;
    for (i, c) in inner.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => depth -= 1,
            ',' if depth == 0 => return Some(&inner[i + 1..]),
            _ => {}
        }
    }
    None
}

const INTS: [&str; 12] = ["i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "isize", "usize"];

/// certificates of every `impl Merge` in the module (least fixpoint, so a
/// field may refer to another certified type but not to itself)
pub fn merge_certs(m: &Module) -> HashMap<String, Cert> {
    let mut out: HashMap<String, Cert> = HashMap::new();
    loop {
        let mut grew = false;
        for it in &m.items {
            let Item::Impl { ty: Ty::Path(tp, _), trait_: Some(Ty::Path(tr, _)), members, .. } = it else {
                continue;
            };
            if tr.len() != 1 || tr[0] != "Merge" || tp.len() != 1 || out.contains_key(&tp[0]) {
                continue;
            }
            if let Some(c) = merge_cert(m, &tp[0], members, &out) {
                out.insert(tp[0].clone(), c);
                grew = true;
            }
        }
        if !grew {
            return out;
        }
    }
}

fn merge_cert(m: &Module, t: &str, members: &[FnDecl], known: &HashMap<String, Cert>) -> Option<Cert> {
    if m.items.iter().any(|i| matches!(i, Item::Kind { name, variants, .. } if name == t && variants.iter().all(|v| v.1.is_empty()))) {
        return Some(Cert::Enum(t.to_string()));
    }
    let fields = m.items.iter().find_map(|i| match i {
        Item::Struct { name, fields, .. } if name == t => Some(fields),
        _ => None,
    })?;
    let fd = members.iter().find(|f| f.sig.name == "merge")?;
    let o = match fd.sig.params.as_slice() {
        [Param::SelfVal, Param::Named { name, .. }] => name.clone(),
        _ => return None,
    };
    let one = |b: &'_ [Stmt]| -> Option<Expr> {
        match b {
            [Stmt::Expr(e)] | [Stmt::Return(Some(e), _)] => Some(e.clone()),
            _ => None,
        }
    };
    // the lexicographic shape, from `if .. then .. else ..` or an `if` statement
    let mut lex: Option<(Expr, Expr, Expr, Expr, Expr)> = None;
    let e = match &fd.body {
        Body::Arrow(e) => e.clone(),
        Body::Block(b) => match b.as_slice() {
            [Stmt::If(arms, Some(els))] if arms.len() == 2 => {
                lex = Some((arms[0].0.clone(), one(&arms[0].1)?, arms[1].0.clone(), one(&arms[1].1)?, one(els)?));
                Expr::Unit(0)
            }
            b => one(b).unwrap_or(Expr::Unit(0)),
        },
    };
    // selection on a single integer field: `if o.k > self.k { o } else { self }` is max (min when the smaller side is kept)
    let sel: Option<(Expr, Expr, Expr)> = match (&fd.body, &e) {
        (Body::Block(b), _) if b.len() == 2 => match (&b[0], &b[1]) {
            (Stmt::If(arms, None), last) if arms.len() == 1 => {
                Some((arms[0].0.clone(), one(&arms[0].1)?, one(std::slice::from_ref(last))?))
            }
            _ => None,
        },
        (Body::Block(b), _) if b.len() == 1 => match &b[0] {
            Stmt::If(arms, Some(els)) if arms.len() == 1 => Some((arms[0].0.clone(), one(&arms[0].1)?, one(els)?)),
            _ => None,
        },
        (_, Expr::If(c, a, b2, _)) if !matches!(&**b2, Expr::If(..)) => Some(((**c).clone(), (**a).clone(), (**b2).clone())),
        _ => None,
    };
    if let Expr::If(c1, e1, rest, _) = &e
        && let Expr::If(c2, e2, e3, _) = &**rest
    {
        lex = Some(((**c1).clone(), (**e1).clone(), (**c2).clone(), (**e2).clone(), (**e3).clone()));
    }
    let fty = |k: &str| -> Option<String> { ty_key(&fields.iter().find(|f| f.name == k)?.ty) };
    let cx = Cx { t, o: &o, known };
    if let Some((c, a, b2)) = &sel
        && fields.len() == 1
    {
        let (k, hi) = cx.cmp(c)?;
        if !INTS.contains(&fty(&k)?.as_str()) || cx.side(b2)? == cx.side(a)? {
            return None;
        }
        let keep = cx.side(a)?;
        return Some(Cert::Prod(vec![(k, if keep == hi { Cert::MaxI } else { Cert::MinI })]));
    }
    // lexicographic: if self.k > o.k { self } else if o.k > self.k { o } else { T(k: self.k, ...) }
    if let Some((c1, e1, c2, e2, e3)) = &lex {
        let (k, hi1) = cx.cmp(c1)?;
        let (k2, hi2) = cx.cmp(c2)?;
        if k != k2 || hi1 == hi2 || !INTS.contains(&fty(&k)?.as_str()) || !cx.whole(e1, hi1) || !cx.whole(e2, hi2) {
            return None;
        }
        let Cert::Prod(fs) = cx.ctor(e3, fields.len(), &fty, Some(&k))? else { return None };
        return Some(Cert::Lex(k, Box::new(Cert::Prod(fs))));
    }
    cx.ctor(&e, fields.len(), &fty, None)
}

struct Cx<'a> {
    t: &'a str,
    o: &'a str,
    known: &'a HashMap<String, Cert>,
}

impl Cx<'_> {
    fn is_self(x: &Expr) -> bool {
        matches!(x, Expr::SelfRef(_)) || matches!(x, Expr::Name(n, _) if n == "self")
    }
    fn is_o(&self, x: &Expr) -> bool {
        matches!(x, Expr::Name(n, _) if n == self.o)
    }
    /// `true` = self, `false` = other
    fn side(&self, x: &Expr) -> Option<bool> {
        if Self::is_self(x) {
            Some(true)
        } else if self.is_o(x) {
            Some(false)
        } else {
            None
        }
    }
    fn fld(&self, x: &Expr) -> Option<(String, bool)> {
        match x {
            Expr::Member(r, f, _) => Some((f.clone(), self.side(r)?)),
            _ => None,
        }
    }
    /// `self.k > o.k` (or `<`): the key and which side is strictly larger
    fn cmp(&self, c: &Expr) -> Option<(String, bool)> {
        let Expr::Bin(op, x, y, _) = c else { return None };
        let (kx, sx) = self.fld(x)?;
        let (ky, sy) = self.fld(y)?;
        if kx != ky || sx == sy {
            return None;
        }
        match op.as_str() {
            ">" => Some((kx, sx)),
            "<" => Some((kx, sy)),
            _ => None,
        }
    }
    fn whole(&self, x: &Expr, side: bool) -> bool {
        self.side(x) == Some(side)
    }
    /// both operands of a binary field combinator, one per side
    fn pair(&self, x: &Expr, y: &Expr, k: &str) -> bool {
        matches!((self.fld(x), self.fld(y)), (Some((a, sa)), Some((b, sb))) if a == k && b == k && sa != sb)
    }
    fn ctor(&self, e: &Expr, n: usize, fty: &dyn Fn(&str) -> Option<String>, key: Option<&str>) -> Option<Cert> {
        let Expr::Call(f, args, _, _) = e else { return None };
        if !matches!(&**f, Expr::Name(x, _) if x == self.t) || args.len() != n {
            return None;
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut out = vec![];
        for a in args {
            let k = a.label.clone()?;
            if !seen.insert(k.clone()) {
                return None;
            }
            let ty = fty(&k)?;
            if Some(k.as_str()) == key {
                // the tie case of a lexicographic merge keeps the (equal) key
                if self.fld(&a.e).map(|p| p.0) != Some(k.clone()) {
                    return None;
                }
                continue;
            }
            out.push((k.clone(), self.field(&a.e, &k, &ty)?));
        }
        Some(Cert::Prod(out))
    }
    /// the certificate of the Merge that `merge_values` applies to a value
    /// of type `ty` (a user impl, or the built-in instance)
    fn builtin(&self, ty: &str) -> Option<Cert> {
        if let Some(_c) = self.known.get(ty) {
            return Some(Cert::Ref(ty.to_string()));
        }
        if INTS.contains(&ty) {
            Some(Cert::MaxI)
        } else if ty == "bool" {
            Some(Cert::OrB)
        } else if ty.starts_with("Set[") {
            Some(Cert::Union)
        } else if ty.starts_with("Map[") {
            Some(Cert::Map(Box::new(self.builtin(map_val(ty)?)?)))
        } else {
            None
        }
    }
    fn field(&self, e: &Expr, k: &str, ty: &str) -> Option<Cert> {
        let int = INTS.contains(&ty);
        match e {
            Expr::Call(c, xs, _, _) => match (&**c, xs.as_slice()) {
                (Expr::Name(n, _), [x, y]) if int && (n == "max" || n == "min") && self.pair(&x.e, &y.e, k) => {
                    Some(if n == "max" { Cert::MaxI } else { Cert::MinI })
                }
                (Expr::Member(x, n, _), [y]) if self.pair(x, &y.e, k) => match n.as_str() {
                    "max" if int => Some(Cert::MaxI),
                    "min" if int => Some(Cert::MinI),
                    "merge" if self.known.contains_key(ty) => Some(Cert::Ref(ty.to_string())),
                    // Map[k, v].merge: pointwise, each value by the Merge of v
                    "merge" if ty.starts_with("Map[") => Some(Cert::Map(Box::new(self.builtin(map_val(ty)?)?))),
                    "union" if ty.starts_with("Set[") => Some(Cert::Union),
                    "inter" if ty.starts_with("Set[") => Some(Cert::Inter),
                    _ => None,
                },
                _ => None,
            },
            Expr::Bin(op, x, y, _) if ty == "bool" && self.pair(x, y, k) => match op.as_str() {
                "or" | "||" => Some(Cert::OrB),
                "and" | "&&" => Some(Cert::AndB),
                _ => None,
            },
            _ => None,
        }
    }
}

/// W-LVAR-2 at admission: in a literal threshold list, two thresholds where
/// one is below the other (`a ⊔ b ∈ {a, b}` in the program's Merge) are
/// compatible for any top, so the set requires change. Other lists are checked
/// against the cell's top by every `wait` (interp `lvar_incomp`).
pub fn lvar_gate(m: &Module) -> Vec<String> {
    // per body: the parameters of a finite type (a kind of nullary variants,
    // or bool) that the body never rebinds, each with every value it can take
    let finite = |t: &Ty| -> Option<Vec<V>> {
        let Ty::Path(p, g) = t else { return None };
        if p.len() != 1 || !g.is_empty() {
            return None;
        }
        if p[0] == "bool" {
            return Some(vec![V::Bool(false), V::Bool(true)]);
        }
        m.items.iter().find_map(|i| match i {
            Item::Kind { name, variants, .. } if *name == p[0] && variants.iter().all(|v| v.1.is_empty()) => {
                Some(variants.iter().map(|v| V::Rec(v.0.clone(), vec![].into())).collect())
            }
            _ => None,
        })
    };
    let mut fns: Vec<(&Body, HashMap<String, Vec<V>>)> = vec![];
    let add = |f: &FnDecl| -> HashMap<String, Vec<V>> {
        let dbg = format!("{:?}", f.body);
        let mut env = HashMap::new();
        for p in &f.sig.params {
            if let Param::Named { name, ty, .. } = p
                && let Some(vs) = finite(ty)
                && !dbg.contains(&format!("Path([{name:?}])"))
                && !dbg.contains(&format!("Var({name:?}"))
                && !dbg.contains(&format!("Assign({name:?}"))
            {
                env.insert(name.clone(), vs);
            }
        }
        env
    };
    for i in &m.items {
        match i {
            Item::Fn(f) => {
                let env = add(f);
                fns.push((&f.body, env));
            }
            Item::Impl { members, .. } => {
                for f in members {
                    let env = add(f);
                    fns.push((&f.body, env));
                }
            }
            _ => {}
        }
    }
    let mut lists: Vec<(Vec<Expr>, usize, HashMap<String, Vec<V>>)> = vec![];
    for (b, env) in fns {
        let mut grab = |e: &Expr| {
            crate::check::walk_expr(e, &mut |x| {
                if let Expr::Call(f, args, l, _) = x
                    && let Expr::Member(_, n, _) = &**f
                    && n == "wait"
                    && let [a] = args.as_slice()
                    && let Expr::List(xs, _) = &a.e
                    && xs.len() > 1
                {
                    lists.push((xs.clone(), *l, env.clone()));
                }
            })
        };
        match b {
            Body::Arrow(e) => grab(e),
            Body::Block(st) => crate::check::walk_stmts(st, &mut |s| {
                for e in crate::check::stmt_exprs(s) {
                    grab(e)
                }
            }),
        }
    }
    let mut errs = vec![];
    if lists.is_empty() {
        return errs;
    }
    let mut it = Interp::new(m);
    'list: for (xs, l, env) in lists {
        // each threshold as the finite set of values it can take: a
        // constant, or `if c then a else b` over such (the condition is
        // abstracted, so every combination is checked)
        let mut vs: Vec<Vec<V>> = vec![];
        for x in &xs {
            match thr_values(&mut it, x, &env) {
                Some(v) => vs.push(v),
                None => continue 'list, // not finite: checked against top by every `wait` at run time
            }
        }
        for (i, sa) in vs.iter().enumerate() {
            for sb in &vs[i + 1..] {
                for a in sa {
                    for b in sb {
                        if a == b {
                            continue;
                        }
                        let Ok(j) = it.merge_values(a, b) else { continue };
                        if j == *a || j == *b {
                            let lit = sa.len() == 1 && sb.len() == 1;
                            errs.push(format!(
                                "line {l}: W-LVAR-2 thresholds {} and {} {} compatible ({} is below the other), so the threshold `wait` returns would depend on the schedule; a threshold set must be pairwise incompatible",
                                a.show(),
                                b.show(),
                                if lit { "are" } else { "can be, on some input, and are then" },
                                if j == *b { a.show() } else { b.show() }
                            ));
                            continue 'list;
                        }
                    }
                }
            }
        }
    }
    errs
}

fn thr_values(it: &mut Interp, x: &Expr, env: &HashMap<String, Vec<V>>) -> Option<Vec<V>> {
    if let Expr::Name(n, _) = x
        && let Some(vs) = env.get(n)
    {
        return Some(vs.clone());
    }
    if let Expr::If(_, a, b, _) = x {
        let mut v = thr_values(it, a, env)?;
        for y in thr_values(it, b, env)? {
            if !v.contains(&y) {
                v.push(y);
            }
        }
        return Some(v);
    }
    it.scopes = vec![HashMap::new()];
    it.eval(x).ok().map(|v| vec![v])
}

/// Three-way merge of a variable written by par branches (base = value at
/// the fork, cur = value after earlier branches, new = this branch's value).
pub(crate) fn merge3(base: &V, cur: &V, new: V) -> V {
    if &new == base {
        return cur.clone();
    }
    if cur == base {
        return new;
    }
    match (base, cur, new) {
        (V::Rec(bn, bf), V::Rec(cn, cf), V::Rec(nn, nf))
            if bn == cn && cn == &nn && bf.len() == cf.len() && cf.len() == nf.len() =>
        {
            let f = bf
                .iter()
                .zip(cf)
                .zip(nf)
                .map(|(((k, b), (_, c)), (_, n))| (k.clone(), merge3(b, c, n)))
                .collect();
            V::Rec(nn, f)
        }
        (V::Tuple(b), V::Tuple(c), V::Tuple(n)) if b.len() == c.len() && c.len() == n.len() => {
            V::Tuple(b.iter().zip(c).zip(n).map(|((b, c), n)| merge3(b, c, n)).collect())
        }
        (V::List(b), V::List(c), V::List(n)) if b.len() == c.len() && c.len() == n.len() => {
            V::List(b.iter().zip(c).zip(n).map(|((b, c), n)| merge3(b, c, n)).collect())
        }
        (_, _, n) => n,
    }
}

/// The LVar rule table: each evaluator rule or checker rule of C.11.5 and
/// C.11.8 with the trap it raises (if any) and the Lean declaration that
/// models or proves it. `tests/lvar_rules.rs` generates
/// lean/RCE/LVarRules.lean from this table (one `#check` per row, plus the
/// certificate of each certified test program as a typed term), so
/// `lake build` fails when a row names a declaration that does not exist or
/// a certificate that does not elaborate; the same test checks that every
/// trap string is raised by the evaluator and by every emitter runtime.
pub const LVAR_RULES: &[(&str, Option<&str>, &str)] = &[
    ("put: v := v ⊔ d in the program Merge", None, "RCE.LVarStore.Step.put"),
    ("put: v ⊔ d = top is an error", Some("lvar conflict"), "RCE.LVarStore.Step.put_top"),
    ("put: a frozen cell that would change is an error", Some("lvar put after freeze"), "RCE.LVarStore.Step.put_frozen"),
    ("wait: threshold set pairwise incompatible against top", Some("lvar thresholds are not pairwise incompatible"), "RCE.LVarQD.Incomp"),
    ("wait: at most one threshold is reached", None, "RCE.LVarQD.incomp_unique"),
    ("wait: answers q ⊑ v in the Merge order", None, "RCE.LVarStore.Step.get"),
    ("freeze: sets the bit, returns the exact value", None, "RCE.LVarStore.Step.freeze"),
    ("store of LVars: quasi-determinism", None, "RCE.LVarStore.store_quasi_det"),
    ("store of LVars: no error without freeze", None, "RCE.LVarStore.store_det_no_err"),
    ("one LVar: quasi-determinism", None, "RCE.LVarQD.quasi_det"),
    ("W-LVAR-1 is needed: a racing freeze errs on some schedule", None, "RCE.LVarQD.quasi_not_det"),
    ("W-CRDT-1 certificate: laws by construction", None, "RCE.MergeCert.cert_laws"),
    ("W-CRDT-1 lex needs a chain key", None, "RCE.MergeCert.lex_needs_chain"),
    ("W-CRDT-1 kind of nullary variants: decided by enumeration", None, "RCE.MergeLaws.laws_of_enum"),
    ("LVar without top: absorbing top added", None, "RCE.MergeLaws.withTop"),
    ("lvar_new: allocates a fresh location", None, "RCE.LVarAlloc.Step.new"),
    ("lvar_new: locations from a deterministic counter", None, "RCE.LVarAlloc.counter_owner"),
    ("store with allocation: quasi-determinism", None, "RCE.LVarAlloc.alloc_quasi_det"),
    ("store with allocation: no error without freeze", None, "RCE.LVarAlloc.alloc_det_no_err"),
    ("executable semantics: each step is a Step", None, "RCE.LVarAlloc.stepAt_sound"),
    ("executable semantics: each run is a run", None, "RCE.LVarAlloc.exec_sound"),
    ("executable semantics: every enabled step is taken", None, "RCE.LVarAlloc.stepAt_complete"),
    ("Set[τ]: the sorted representation is unique", None, "RCE.Canon.set_canon"),
    ("Map[k, v]: the key-sorted representation is unique", None, "RCE.Canon.map_canon"),
    ("cmp_canon: a strict total order (rank, IEEE totalOrder floats, lexicographic), Equal iff equal", None, "RCE.ValOrder.lt_total"),
    ("canonical order of values: transitive", None, "RCE.ValOrder.lt_trans"),
    ("Set[τ] of values: one sorted representation", None, "RCE.ValOrder.set_canon_val"),
    ("Map with value keys: one key-sorted representation", None, "RCE.ValOrder.map_canon_val"),
    ("W-CRDT-1 Set.union, Set.inter, Map.merge", None, "RCE.MergeCert.mapS"),
];

/// lean/RCE/LVarRules.lean for the table and the given certificates
pub fn lvar_rules_lean(certs: &[(String, String)]) -> String {
    let mut s = String::from(
        "/-\nGenerated by checker/tests/lvar_rules.rs from `conc::LVAR_RULES` and the\ncertificates the checker derives for the certified test programs. Do not\nedit: regenerate with RCE_BLESS=1 cargo test --features conc06 --test lvar_rules.\n-/\nimport RCE.LVarStore\nimport RCE.LVarAlloc\nimport RCE.Canon\nimport RCE.ValOrder\nimport RCE.MergeCert\nimport RCE.MergeLaws\n\nnamespace RCE.LVarRules\n\n",
    );
    for (rule, trap, decl) in LVAR_RULES {
        let t = trap.map(|t| format!(" (trap \"{t}\")")).unwrap_or_default();
        s.push_str(&format!("-- {rule}{t}\n#check @{decl}\n"));
    }
    s.push('\n');
    for (name, term) in certs {
        s.push_str(&format!(
            "/-- impl Merge for {name} -/\nexample : RCE.MergeCert.Semi (RCE.MergeCert.Cert.ty {term}) :=\n  RCE.MergeCert.Cert.semi {term}\n"
        ));
    }
    s.push_str("\nend RCE.LVarRules\n");
    s
}
