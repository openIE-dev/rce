//! Borrow paths and `let` borrows (spec 5.6).
//!
//! A borrowed place has a path: a root variable and a list of projections
//! (a field, a tuple element, or `[_]` for any element of a list or map).
//! Two paths overlap iff one is a prefix of the other, and `x[i]` and `x[j]`
//! always overlap (indexes are not compared statically). This is the
//! `overlaps` of the Lean model `RCE.Borrow` (theorem T6).
//!
//! A `let a = &p` or `let a = &mut p` borrow is live from its `let` to the
//! last statement of the same block that uses `a` (or a name bound from it).
//! Between the two, a use of a path overlapping p that the borrow's kind
//! forbids ([No-Alias-Write], [No-Mut-Read], [Borrow-Mut], [Borrow-Shr])
//! requires change under W-OWN-2 (`let_borrow_conflicts`).
//!
//! In an admitted program `a` denotes the place p itself: a write through
//! `a` is a write to p (`writeVia` of the Lean model). The evaluators and the
//! program emitters realise that by one shared rewrite (`lower_block`,
//! `lower_module`): every later use of `a` in its block is replaced by p,
//! with p's index expressions evaluated once, at the `let`, into
//! temporaries named `anf_lb$N`. Neither the `let` of such a temporary nor a
//! read of it is a source statement or expression, so neither ticks (6.7).
use crate::ast::*;
use crate::check::{mutating_method, place_root, stmt_direct_exprs, walk_expr, walk_stmts};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq)]
pub enum Proj {
    Field(String),
    Elem,
}

/// The root and the projections of a place (`f.a` is `f` with `[a]`).
pub fn place_path(e: &Expr) -> Option<(String, Vec<Proj>)> {
    let ext = |x: &Expr, p: Proj| {
        let (r, mut ps) = place_path(x)?;
        ps.push(p);
        Some((r, ps))
    };
    match e {
        Expr::Name(n, _) => Some((n.clone(), vec![])),
        Expr::SelfRef(_) => Some(("self".into(), vec![])),
        Expr::Member(x, f, _) => ext(x, Proj::Field(f.clone())),
        Expr::TupleMember(x, i, _) => ext(x, Proj::Field(i.to_string())),
        Expr::Index(x, _, _) | Expr::Slice(x, _, _, _) => ext(x, Proj::Elem),
        Expr::Borrow(_, x, _) => place_path(x),
        _ => None,
    }
}

/// Two paths of one root overlap iff one is a prefix of the other (5.6).
pub fn overlaps(p: &[Proj], q: &[Proj]) -> bool {
    let n = p.len().min(q.len());
    p[..n] == q[..n]
}

pub fn show_path(root: &str, p: &[Proj]) -> String {
    let mut s = root.to_string();
    for x in p {
        match x {
            Proj::Field(f) => {
                s.push('.');
                s.push_str(f);
            }
            Proj::Elem => s.push_str("[_]"),
        }
    }
    s
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Acc {
    Read,
    Write,
    Shr,
    Mut,
}

/// The children of an expression, in source order.
fn kids(e: &Expr) -> Vec<&Expr> {
    use Expr::*;
    match e {
        Tuple(v, _) | List(v, _) => v.iter().collect(),
        ListRepeat(a, b, _) | Bin(_, a, b, _) | Catch(a, _, b, _) => vec![a, b],
        ListComp(a, _, b, c, _) => {
            let mut v: Vec<&Expr> = vec![a, b];
            v.extend(c.as_deref());
            v
        }
        Map(v, _) => v.iter().flat_map(|(a, b)| [a, b]).collect(),
        ChanNew(_, Some(x), _) => vec![x],
        Call(c, args, _, _) => {
            let mut v: Vec<&Expr> = vec![c];
            v.extend(args.iter().map(|a| &a.e));
            v
        }
        Index(x, ix, _) => {
            let mut v: Vec<&Expr> = vec![x];
            v.extend(ix.iter());
            v
        }
        Slice(x, a, b, _) => {
            let mut v: Vec<&Expr> = vec![x];
            v.extend(a.as_deref());
            v.extend(b.as_deref());
            v
        }
        Member(x, _, _) | TupleMember(x, _, _) | Neg(x, _) | BitNot(x, _) | Borrow(_, x, _)
        | Recv(x, _) | Comptime(x, _) | Cast(x, _, _) | Not(x, _) | Lambda(_, x, _) => vec![x],
        Try(a, b, _) => {
            let mut v: Vec<&Expr> = vec![a];
            v.extend(b.as_deref());
            v
        }
        If(a, b, c, _) => vec![a, b, c],
        _ => vec![],
    }
}

/// Every access of a path rooted at `root` in a statement (and the blocks
/// nested in it), with its kind and line.
struct Scan<'a> {
    root: &'a str,
    mut_methods: &'a HashSet<String>,
    out: Vec<(Vec<Proj>, Acc, usize)>,
}

impl Scan<'_> {
    fn expr(&mut self, e: &Expr, k: Acc) {
        match e {
            Expr::Borrow(m, x, _) => {
                return self.expr(x, if *m { Acc::Mut } else { Acc::Shr });
            }
            Expr::Call(f, args, _, _) => {
                match &**f {
                    Expr::Member(o, m, _) => {
                        let w = mutating_method(m) || self.mut_methods.contains(m);
                        self.expr(o, if w { Acc::Write } else { Acc::Read });
                    }
                    other => self.expr(other, Acc::Read),
                }
                for a in args {
                    self.expr(&a.e, Acc::Read);
                }
                return;
            }
            _ => {}
        }
        if let Some((r, p)) = place_path(e) {
            if r == self.root {
                self.out.push((p, k, e.line()));
            }
            self.indexes(e);
            return;
        }
        for c in kids(e) {
            self.expr(c, Acc::Read);
        }
    }
    /// the index expressions inside a place are read
    fn indexes(&mut self, e: &Expr) {
        match e {
            Expr::Member(x, _, _) | Expr::TupleMember(x, _, _) | Expr::Borrow(_, x, _) => self.indexes(x),
            Expr::Index(x, ix, _) => {
                self.indexes(x);
                for i in ix {
                    self.expr(i, Acc::Read);
                }
            }
            Expr::Slice(x, a, b, _) => {
                self.indexes(x);
                for i in a.iter().chain(b.iter()) {
                    self.expr(i, Acc::Read);
                }
            }
            _ => {}
        }
    }
    fn stmt(&mut self, s: &Stmt) {
        walk_stmts(std::slice::from_ref(s), &mut |s| match s {
            Stmt::Assign(_, p, rhs) => {
                self.expr(p, Acc::Write);
                self.expr(rhs, Acc::Read);
            }
            Stmt::Match(e, arms, _) => {
                self.expr(e, Acc::Read);
                for a in arms {
                    if let Some(g) = &a.guard {
                        self.expr(g, Acc::Read);
                    }
                }
            }
            _ => {
                for e in stmt_direct_exprs(s) {
                    self.expr(e, Acc::Read);
                }
            }
        });
    }
}

fn mentions(s: &Stmt, names: &HashSet<String>) -> bool {
    let mut hit = false;
    walk_stmts(std::slice::from_ref(s), &mut |s| {
        let mut es = stmt_direct_exprs(s);
        if let Stmt::Match(_, arms, _) = s {
            es.extend(arms.iter().filter_map(|a| a.guard.as_ref()));
        }
        for e in es {
            walk_expr(e, &mut |x| {
                if let Expr::Name(n, _) = x
                    && names.contains(n)
                {
                    hit = true;
                }
            });
        }
    });
    hit
}

fn stmt_line(s: &Stmt) -> usize {
    crate::check06b::stmt_line(s)
}

/// A `let` (or `var`) that binds one name to a borrow of a place.
fn let_borrow(s: &Stmt) -> Option<(&String, bool, &Expr, usize)> {
    match s {
        Stmt::Let(Pat::Path(p), _, Expr::Borrow(m, x, _), line) if p.len() == 1 => Some((&p[0], *m, x, *line)),
        Stmt::Var(n, _, Expr::Borrow(m, x, _), line) => Some((n, *m, x, *line)),
        _ => None,
    }
}

/// A `let` borrow of one block (5.6): statement `i` makes it, statement `j`
/// is the last that uses it.
pub struct LetLoan {
    pub i: usize,
    pub j: usize,
    pub alias: String,
    pub mutable: bool,
    pub root: String,
    pub path: Vec<Proj>,
    pub line: usize,
    /// the line of statement `j`
    pub used: usize,
}

/// The `let` borrows of a block that are used after their `let`.
pub fn let_loans(b: &[Stmt]) -> Vec<LetLoan> {
    let mut out = vec![];
    for (i, s) in b.iter().enumerate() {
        let Some((a, m, place, line)) = let_borrow(s) else { continue };
        let Some((root, path)) = place_path(place) else { continue };
        if &root == a {
            continue;
        }
        // the names that denote the borrow: `a` and names bound from it
        let mut names: HashSet<String> = HashSet::from([a.clone()]);
        let mut last = None;
        for (k, t) in b.iter().enumerate().skip(i + 1) {
            if mentions(t, &names) {
                last = Some(k);
                if let Stmt::Let(Pat::Path(p), _, e, _) = t
                    && p.len() == 1
                    && place_root(e).is_some_and(|r| names.contains(&r))
                {
                    names.insert(p[0].clone());
                }
            }
        }
        let Some(j) = last else { continue };
        out.push(LetLoan { i, j, alias: a.clone(), mutable: m, root, path, line, used: stmt_line(&b[j]) });
    }
    out
}

/// W-OWN-2 for the `let` borrows of one block (5.6): `(line, message)` for
/// every use of an overlapping path that the borrow's kind forbids while
/// the borrow is live. `mut_methods` names the methods with `&mut self`.
/// (A move of the whole root under a `&` borrow is found by the checker,
/// which knows which types move.)
pub fn let_borrow_conflicts(b: &[Stmt], mut_methods: &HashSet<String>) -> Vec<(usize, String)> {
    let mut out = vec![];
    for ln in let_loans(b) {
        let LetLoan { i, j, alias: a, mutable: m, root, path, line, used } = ln;
        let shown = show_path(&root, &path);
        for t in &b[i + 1..=j] {
            let mut sc = Scan { root: &root, mut_methods, out: vec![] };
            sc.stmt(t);
            for (q, acc, l) in sc.out {
                if !overlaps(&path, &q) {
                    continue;
                }
                let what = show_path(&root, &q);
                let (verb, rule) = match (m, acc) {
                    (true, Acc::Read) => ("read", "[No-Mut-Read]"),
                    (_, Acc::Write) => ("written", "[No-Alias-Write]"),
                    (_, Acc::Mut) => ("borrowed mutably", "[Borrow-Mut]"),
                    (true, Acc::Shr) => ("borrowed", "[Borrow-Shr]"),
                    (false, Acc::Read | Acc::Shr) => continue,
                };
                let kind = if m { "&mut" } else { "&" };
                out.push((
                    l.max(1),
                    format!(
                        "`{what}` is {verb} while `{a}` (line {line}) borrows `{shown}` with `{kind}`, and `{a}` is used again at line {used}: end the borrow before this use (5.6 {rule})"
                    ),
                ));
            }
        }
    }
    out
}

// ---- the shared rewrite: a `let &mut` borrow names its place ----

/// `let a = &mut p` with p a place without a slice: the statement whose
/// later uses of `a` are replaced by p.
pub fn is_let_mut(s: &Stmt) -> bool {
    matches!(let_borrow(s), Some((a, true, p, _)) if place_path(p).is_some_and(|(r, _)| &r != a) && !has_slice(p))
}

fn has_slice(e: &Expr) -> bool {
    match e {
        Expr::Slice(..) => true,
        Expr::Member(x, _, _) | Expr::TupleMember(x, _, _) | Expr::Index(x, _, _) | Expr::Borrow(_, x, _) => has_slice(x),
        _ => false,
    }
}

/// The prefix of the temporaries that hold a `let` borrow's indexes.
pub const TEMP: &str = "anf_lb$";

pub fn is_temp(n: &str) -> bool {
    n.starts_with(TEMP)
}

/// p with every index that is not a literal bound to a temporary (in `pre`)
fn hoist(e: &Expr, n: &mut usize, pre: &mut Vec<Stmt>) -> Expr {
    match e {
        Expr::Member(x, f, l) => Expr::Member(Box::new(hoist(x, n, pre)), f.clone(), *l),
        Expr::TupleMember(x, i, l) => Expr::TupleMember(Box::new(hoist(x, n, pre)), *i, *l),
        Expr::Borrow(m, x, l) => Expr::Borrow(*m, Box::new(hoist(x, n, pre)), *l),
        Expr::Index(x, ix, l) => {
            let x = hoist(x, n, pre);
            let ix = ix
                .iter()
                .map(|i| match i {
                    Expr::Int(..) | Expr::Str(..) | Expr::Bool(..) => i.clone(),
                    _ => {
                        *n += 1;
                        let t = format!("{TEMP}{n}");
                        pre.push(Stmt::Let(Pat::Path(vec![t.clone()]), None, i.clone(), i.line()));
                        Expr::Name(t, i.line())
                    }
                })
                .collect();
            Expr::Index(Box::new(x), ix, *l)
        }
        _ => e.clone(),
    }
}

fn free_names(e: &Expr) -> HashSet<String> {
    let mut s = HashSet::new();
    walk_expr(e, &mut |x| {
        if let Expr::Name(n, _) = x {
            s.insert(n.clone());
        }
    });
    s
}

type Env = HashMap<String, Expr>;

/// Capture-avoiding replacement of names in statements: a binder of a
/// replaced name ends its replacement; a binder of a name that occurs in a
/// replacement (the borrowed root) is renamed, so the replacement keeps
/// naming the outer variable.
struct Sub<'a> {
    avoid: HashSet<String>,
    n: &'a mut usize,
}

impl Sub<'_> {
    fn bind(&mut self, x: &mut String, env: &mut Env) {
        if self.avoid.contains(x) {
            *self.n += 1;
            let f = format!("{x}$lb{}", self.n);
            env.insert(x.clone(), Expr::Name(f.clone(), 0));
            *x = f;
        } else {
            env.remove(x);
        }
    }
    fn pat(&mut self, p: &mut Pat, env: &mut Env) {
        match p {
            Pat::Path(v) if v.len() == 1 => {
                let mut x = v[0].clone();
                self.bind(&mut x, env);
                v[0] = x;
            }
            Pat::Ctor(_, fs) => fs.iter_mut().for_each(|(_, q)| self.pat(q, env)),
            Pat::Tuple(v) | Pat::Alt(v) => v.iter_mut().for_each(|q| self.pat(q, env)),
            _ => {}
        }
    }
    fn block(&mut self, b: &mut [Stmt], env: &Env) {
        let mut env = env.clone();
        for s in b {
            self.stmt(s, &mut env);
        }
    }
    fn stmt(&mut self, s: &mut Stmt, env: &mut Env) {
        match s {
            Stmt::Expr(e) | Stmt::Emit(e) | Stmt::Fail(e) | Stmt::Spawn(e) | Stmt::Check(e) | Stmt::Yield(e, _) => self.expr(e, env),
            Stmt::Assign(_, a, b) | Stmt::Send(a, b) => {
                self.expr(a, env);
                self.expr(b, env);
            }
            Stmt::Let(p, _, e, _) => {
                self.expr(e, env);
                self.pat(p, env);
            }
            Stmt::Var(x, _, e, _) => {
                self.expr(e, env);
                self.bind(x, env);
            }
            Stmt::If(arms, els) => {
                for (c, b) in arms {
                    self.expr(c, env);
                    self.block(b, env);
                }
                if let Some(b) = els {
                    self.block(b, env);
                }
            }
            Stmt::While(c, bd, b) => {
                self.expr(c, env);
                if let Some(x) = bd {
                    self.expr(x, env);
                }
                self.block(b, env);
            }
            Stmt::Loop(bd, b) => {
                if let Some(x) = bd {
                    self.expr(x, env);
                }
                self.block(b, env);
            }
            Stmt::For(p, it, b, _) => {
                self.expr(it, env);
                let mut e2 = env.clone();
                self.pat(p, &mut e2);
                self.block(b, &e2);
            }
            Stmt::Match(e, arms, _) => {
                self.expr(e, env);
                for a in arms {
                    let mut e2 = env.clone();
                    self.pat(&mut a.pat, &mut e2);
                    if let Some(g) = &mut a.guard {
                        self.expr(g, &e2);
                    }
                    self.block(&mut a.body, &e2);
                }
            }
            Stmt::Return(e, _) => {
                if let Some(e) = e {
                    self.expr(e, env);
                }
            }
            Stmt::Select(arms, _) => {
                for a in arms {
                    match a {
                        SelArm::Recv(p, e, b) => {
                            self.expr(e, env);
                            let mut e2 = env.clone();
                            self.pat(p, &mut e2);
                            self.block(b, &e2);
                        }
                        SelArm::Send(x, y, b) => {
                            self.expr(x, env);
                            self.expr(y, env);
                            self.block(b, env);
                        }
                        SelArm::After(x, b) => {
                            self.expr(x, env);
                            self.block(b, env);
                        }
                        SelArm::Default(b) => self.block(b, env),
                    }
                }
            }
            Stmt::Arena(_, bd, b, _) | Stmt::Device(_, bd, b, _) => {
                if let Some(x) = bd {
                    self.expr(x, env);
                }
                self.block(b, env);
            }
            Stmt::Comptime(b) => self.block(b, env),
            Stmt::Break(_) | Stmt::Continue(_) | Stmt::Pass => {}
        }
    }
    fn expr(&mut self, e: &mut Expr, env: &Env) {
        use Expr::*;
        if let Name(n, _) = e
            && let Some(r) = env.get(n.as_str())
        {
            *e = r.clone();
            return;
        }
        match e {
            Tuple(v, _) | List(v, _) => v.iter_mut().for_each(|x| self.expr(x, env)),
            ListRepeat(a, b, _) | Bin(_, a, b, _) => {
                self.expr(a, env);
                self.expr(b, env);
            }
            Catch(a, bind, h, _) => {
                self.expr(a, env);
                let mut e2 = env.clone();
                if let Some(x) = bind {
                    self.bind(x, &mut e2);
                }
                self.expr(h, &e2);
            }
            ListComp(body, p, it, c, _) => {
                self.expr(it, env);
                let mut e2 = env.clone();
                self.pat(p, &mut e2);
                self.expr(body, &e2);
                if let Some(c) = c {
                    self.expr(c, &e2);
                }
            }
            Map(v, _) => {
                for (a, b) in v {
                    self.expr(a, env);
                    self.expr(b, env);
                }
            }
            ChanNew(_, Some(x), _) => self.expr(x, env),
            Call(c, args, _, _) => {
                self.expr(c, env);
                args.iter_mut().for_each(|a| self.expr(&mut a.e, env));
            }
            Index(x, ix, _) => {
                self.expr(x, env);
                ix.iter_mut().for_each(|y| self.expr(y, env));
            }
            Slice(x, a, b, _) => {
                self.expr(x, env);
                if let Some(a) = a {
                    self.expr(a, env);
                }
                if let Some(b) = b {
                    self.expr(b, env);
                }
            }
            Member(x, _, _) | TupleMember(x, _, _) | Neg(x, _) | BitNot(x, _) | Borrow(_, x, _) | Recv(x, _)
            | Comptime(x, _) | Cast(x, _, _) | Not(x, _) => self.expr(x, env),
            Lambda(ps, body, _) => {
                let mut e2 = env.clone();
                for p in ps.iter_mut() {
                    if let Param::Named { name, .. } = p {
                        self.bind(name, &mut e2);
                    }
                }
                self.expr(body, &e2);
            }
            Try(a, b, _) => {
                self.expr(a, env);
                if let Some(b) = b {
                    self.expr(b, env);
                }
            }
            If(a, b, c, _) => {
                self.expr(a, env);
                self.expr(b, env);
                self.expr(c, env);
            }
            _ => {}
        }
    }
}

/// One block with its `let &mut` borrows rewritten (the blocks nested in
/// it are left for their own turn); `n` numbers the temporaries.
pub fn lower_block(b: &[Stmt], n: &mut usize) -> Vec<Stmt> {
    let mut rest: Vec<Stmt> = b.to_vec();
    let mut i = 0;
    while i < rest.len() {
        if is_let_mut(&rest[i]) {
            let (a, place, ty, line) = match &rest[i] {
                Stmt::Let(Pat::Path(p), ty, Expr::Borrow(_, x, _), line) => (p[0].clone(), (**x).clone(), ty.clone(), *line),
                Stmt::Var(a, ty, Expr::Borrow(_, x, _), line) => (a.clone(), (**x).clone(), ty.clone(), *line),
                _ => unreachable!(),
            };
            let mut pre = vec![];
            let place = hoist(&place, n, &mut pre);
            let k = pre.len();
            let new = match &rest[i] {
                Stmt::Var(..) => Stmt::Var(a.clone(), ty, Expr::Borrow(true, Box::new(place.clone()), line), line),
                _ => Stmt::Let(Pat::Path(vec![a.clone()]), ty, Expr::Borrow(true, Box::new(place.clone()), line), line),
            };
            pre.push(new);
            rest.splice(i..=i, pre);
            i += k;
            let env: Env = HashMap::from([(a, place.clone())]);
            let mut sub = Sub { avoid: free_names(&place), n };
            sub.block(&mut rest[i + 1..], &env);
        }
        i += 1;
    }
    rest
}

fn lower_nested(b: &mut Vec<Stmt>, n: &mut usize) {
    if b.iter().any(is_let_mut) {
        *b = lower_block(b, n);
    }
    for s in b.iter_mut() {
        match s {
            Stmt::If(arms, els) => {
                arms.iter_mut().for_each(|(_, b)| lower_nested(b, n));
                if let Some(b) = els {
                    lower_nested(b, n);
                }
            }
            Stmt::While(_, _, b) | Stmt::Loop(_, b) | Stmt::For(_, _, b, _) | Stmt::Arena(_, _, b, _) | Stmt::Device(_, _, b, _) | Stmt::Comptime(b) => lower_nested(b, n),
            Stmt::Match(_, arms, _) => arms.iter_mut().for_each(|a| lower_nested(&mut a.body, n)),
            Stmt::Select(arms, _) => {
                for a in arms {
                    match a {
                        SelArm::Recv(_, _, b) | SelArm::Send(_, _, b) | SelArm::After(_, b) | SelArm::Default(b) => lower_nested(b, n),
                    }
                }
            }
            _ => {}
        }
    }
}

fn lower_body(b: &mut Body, n: &mut usize) {
    if let Body::Block(b) = b {
        lower_nested(b, n);
    }
}

/// The module with every `let &mut` borrow rewritten (for the emitters).
pub fn lower_module(m: &Module) -> Module {
    let mut out = m.clone();
    let mut n = 0;
    for it in out.items.iter_mut() {
        match it {
            Item::Fn(f) => lower_body(&mut f.body, &mut n),
            Item::Impl { members, .. } => members.iter_mut().for_each(|f| lower_body(&mut f.body, &mut n)),
            Item::Actor { fns, handlers, .. } => {
                fns.iter_mut().for_each(|f| lower_body(&mut f.body, &mut n));
                handlers.iter_mut().for_each(|h| lower_body(&mut h.body, &mut n));
            }
            Item::Step { body, .. } => lower_body(body, &mut n),
            _ => {}
        }
    }
    out
}
