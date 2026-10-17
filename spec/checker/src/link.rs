//! Cross-module linking (spec 4.1). `use p.q [as r]` makes module `p.q`
//! visible under `r` (default `q`); its `pub` items are reached as `r.x`
//! in expressions, types and patterns. Checking and evaluating a module
//! works on a *linked* copy: the library items of every module it imports
//! (transitively) are appended under their own names, and each qualified
//! reference `r.x` is rewritten to `x`. The appended items are checked in
//! their own module, so the checker skips their bodies here.
//!
//! Limits (refused with a diagnostic, not silently wrong): an imported item
//! whose name clashes with an item of the importing module or of another
//! imported module (W-MOD-3; there is no renaming yet), and Form items
//! (state, event, step, actor) of an imported module, which are not linked.
use crate::ast::*;
use std::collections::{BTreeMap, BTreeSet, HashSet};

pub fn item_name(it: &Item) -> Option<String> {
    Some(match it {
        Item::Const { name, .. }
        | Item::TypeAlias { name, .. }
        | Item::Struct { name, .. }
        | Item::State { name, .. }
        | Item::Event { name, .. }
        | Item::Error { name, .. }
        | Item::Kind { name, .. }
        | Item::Trait { name, .. }
        | Item::Actor { name, .. }
        | Item::Protocol { name, .. }
        | Item::Node { name, .. }
        | Item::Supervisor { name, .. } => name.clone(),
        Item::Fn(f) => f.sig.name.clone(),
        Item::Extern { sig, .. } => sig.name.clone(),
        _ => return None,
    })
}

/// items a library import contributes
fn linkable(it: &Item) -> bool {
    matches!(
        it,
        Item::Const { .. }
            | Item::TypeAlias { .. }
            | Item::Struct { .. }
            | Item::Error { .. }
            | Item::Kind { .. }
            | Item::Trait { .. }
            | Item::Impl { .. }
            | Item::Fn(_)
            | Item::Extern { .. }
            | Item::Protocol { .. }
    )
}

pub struct Linked {
    pub module: Module,
    /// names of imported functions and impl heads (bodies checked elsewhere)
    pub imported_fns: HashSet<String>,
    pub imported_types: HashSet<String>,
    pub errs: Vec<String>,
}

fn by_path<'a>(all: &'a [&'a Module], p: &str) -> Option<&'a Module> {
    all.iter().copied().find(|m| m.path.as_ref().map(|q| q.join(".")).as_deref() == Some(p))
}

/// alias -> module path, for the user modules `m` imports
fn aliases(m: &Module, all: &[&Module]) -> BTreeMap<String, String> {
    let mut a = BTreeMap::new();
    for it in &m.items {
        if let Item::Use(p, al, _) = it {
            let s = p.join(".");
            if by_path(all, &s).is_some() {
                a.insert(al.clone().unwrap_or_else(|| p.last().cloned().unwrap_or_default()), s);
            }
        }
    }
    a
}

pub fn link(m: &Module, all: &[&Module]) -> Linked {
    link_in(m, all, &mut vec![])
}

/// `stack` holds the modules being linked on the current path; meeting one
/// of them again is an import cycle (W-MOD-2), reported instead of followed.
fn link_in(m: &Module, all: &[&Module], stack: &mut Vec<String>) -> Linked {
    let mut errs = vec![];
    let mut out = m.clone();
    let al = aliases(m, all);
    if al.is_empty() {
        return Linked { module: out, imported_fns: HashSet::new(), imported_types: HashSet::new(), errs };
    }
    // qualified references in m
    let mut rw = Rw { al: &al, all, errs: &mut errs };
    for it in &mut out.items {
        rw.item(it);
    }
    // bare use of an imported item is not in scope (4.1)
    let own: BTreeSet<String> = m.items.iter().filter_map(item_name).collect();
    // modules to append, transitively, each once
    let mut seen = BTreeSet::new();
    let mut queue: Vec<String> = al.values().cloned().collect();
    let mut extra = vec![];
    let mut origin: BTreeMap<String, String> = BTreeMap::new();
    let mut imported_fns = HashSet::new();
    let mut imported_types = HashSet::new();
    let self_path = m.path.as_ref().map(|q| q.join("."));
    stack.push(self_path.clone().unwrap_or_default());
    while let Some(p) = queue.pop() {
        if Some(&p) == self_path.as_ref() || !seen.insert(p.clone()) {
            continue;
        }
        if let Some(i) = stack.iter().position(|q| q == &p) {
            let line = m.items.iter().find_map(|it| match it {
                Item::Use(q, _, l) if q.join(".") == p => Some(*l),
                _ => None,
            }).unwrap_or(0);
            let mut cyc = stack[i..].to_vec();
            cyc.push(p.clone());
            errs.push(format!("line {line}: W-MOD-2 import cycle {}", cyc.join(" -> ")));
            continue;
        }
        let Some(dm) = by_path(all, &p) else { continue };
        let inner = link_in(dm, all, stack);
        errs.extend(inner.errs.iter().filter(|e| e.contains("W-MOD-2")).cloned());
        queue.extend(aliases(dm, all).into_values());
        for it in inner.module.items.iter().filter(|i| linkable(i)) {
            // items the inner link appended belong to other modules, handled in their turn
            if let Some(n) = item_name(it) {
                if !dm.items.iter().any(|j| item_name(j).as_deref() == Some(n.as_str())) {
                    continue;
                }
                if own.contains(&n) {
                    errs.push(format!("line 0: W-MOD-3 `{n}` of module `{p}` clashes with a declaration of this module (imported names are not renamed yet)"));
                    continue;
                }
                if let Some(o) = origin.insert(n.clone(), p.clone())
                    && o != p
                {
                    errs.push(format!("line 0: W-MOD-3 `{n}` is declared in both `{o}` and `{p}` (imported names are not renamed yet)"));
                    continue;
                }
                match it {
                    Item::Fn(_) | Item::Extern { .. } => {
                        imported_fns.insert(n.clone());
                    }
                    _ => {
                        imported_types.insert(n.clone());
                    }
                }
            } else if let Item::Impl { ty, .. } = it {
                imported_types.insert(crate::check::ty_head(ty));
            }
            extra.push(it.clone());
        }
    }
    // a bare call or constructor naming an imported item that m does not declare
    let mut locals = BTreeSet::new();
    binders(m, &mut locals);
    for it in &m.items {
        each_expr(it, &mut |x| {
            if let Expr::Call(c, _, l, _) = x
                && let Expr::Name(n, _) = &**c
                && origin.contains_key(n)
                && !own.contains(n)
                && !locals.contains(n)
            {
                let msg = format!(
                    "line {l}: W-NAME-1 `{n}` is declared in module `{}`; refer to it through its import",
                    origin[n]
                );
                if !errs.contains(&msg) {
                    errs.push(msg);
                }
            }
        });
    }
    out.items.extend(extra);
    stack.pop();
    errs.dedup();
    Linked { module: out, imported_fns, imported_types, errs }
}

fn binders(m: &Module, out: &mut BTreeSet<String>) {
    let mut fns = vec![];
    for it in &m.items {
        match it {
            Item::Fn(f) => fns.push(f),
            Item::Impl { members, .. } => fns.extend(members),
            Item::Actor { fns: fs, .. } => fns.extend(fs),
            _ => {}
        }
    }
    for f in fns {
        for p in &f.sig.params {
            if let Param::Named { name, .. } = p {
                out.insert(name.clone());
            }
        }
        if let Body::Block(b) = &f.body {
            crate::check::walk_stmts(b, &mut |s| match s {
                Stmt::Var(n, ..) => {
                    out.insert(n.clone());
                }
                Stmt::Let(p, ..) | Stmt::For(p, ..) => crate::check::pat_names_pub(p, out),
                _ => {}
            });
        }
    }
}

/// every expression of an item (bodies, guards, constants, checks)
fn each_expr(it: &Item, f: &mut dyn FnMut(&Expr)) {
    fn body(b: &Body, f: &mut dyn FnMut(&Expr)) {
        match b {
            Body::Arrow(e) => crate::check::walk_expr(e, f),
            Body::Block(s) => crate::check::walk_stmts(s, &mut |st| {
                for e in crate::check::stmt_exprs(st) {
                    crate::check::walk_expr(e, f);
                }
            }),
        }
    }
    match it {
        Item::Fn(fd) => body(&fd.body, f),
        Item::Impl { members, .. } => members.iter().for_each(|fd| body(&fd.body, f)),
        Item::Step { body: b, guard, .. } => {
            body(b, f);
            if let Some(g) = guard {
                crate::check::walk_expr(g, f);
            }
        }
        Item::Const { e, .. } | Item::Check(e) => crate::check::walk_expr(e, f),
        _ => {}
    }
}

struct Rw<'a, 'b> {
    al: &'a BTreeMap<String, String>,
    all: &'a [&'a Module],
    errs: &'b mut Vec<String>,
}

impl Rw<'_, '_> {
    /// `alias.x` -> `x`, if `x` is a pub item of the aliased module
    fn resolve(&mut self, a: &str, x: &str, line: usize) -> bool {
        let Some(p) = self.al.get(a) else { return false };
        let Some(dm) = by_path(self.all, p) else { return false };
        let declared = dm.items.iter().any(|i| item_name(i).as_deref() == Some(x));
        let variant = dm.items.iter().any(|i| matches!(i, Item::Kind { variants, .. } if variants.iter().any(|(v, _)| v == x)));
        if !declared && !variant {
            self.errs.push(format!("line {line}: W-NAME-1 module `{p}` has no item `{x}`"));
        } else if declared && !dm.pubs.contains(x) {
            self.errs.push(format!("line {line}: W-VIS-1 `{x}` is private to module `{p}`"));
        }
        true
    }
    fn ty(&mut self, t: &mut Ty) {
        match t {
            Ty::Path(p, args) => {
                if p.len() == 2 && self.resolve(&p[0].clone(), &p[1].clone(), 0) {
                    p.remove(0);
                }
                args.iter_mut().for_each(|a| self.ty(a));
            }
            Ty::Tuple(v) | Ty::Fn(v, _) => {
                v.iter_mut().for_each(|a| self.ty(a));
                if let Ty::Fn(_, r) = t {
                    self.ty(r);
                }
            }
            Ty::Slice(a) | Ty::Chan(a) | Ty::Iter(a) | Ty::Ref(_, a) | Ty::Opt(a) | Ty::Region(a, _) => self.ty(a),
            Ty::Array(a, e) => {
                self.ty(a);
                self.expr(e);
            }
            Ty::Union(a, b) => {
                self.ty(a);
                self.ty(b);
            }
            Ty::Unit => {}
        }
    }
    fn pat(&mut self, p: &mut Pat) {
        match p {
            Pat::Path(v) | Pat::Ctor(v, _) => {
                if v.len() >= 2 && self.al.contains_key(&v[0]) {
                    let (a, x) = (v[0].clone(), v[1].clone());
                    self.resolve(&a, &x, 0);
                    v.remove(0);
                }
                if let Pat::Ctor(_, args) = p {
                    args.iter_mut().for_each(|(_, q)| self.pat(q));
                }
            }
            Pat::Tuple(v) | Pat::Alt(v) => v.iter_mut().for_each(|q| self.pat(q)),
            _ => {}
        }
    }
    fn expr(&mut self, e: &mut Expr) {
        use Expr::*;
        if let Member(o, x, l) = e
            && let Name(a, _) = &**o
            && self.al.contains_key(a)
        {
            let (a, x, l) = (a.clone(), x.clone(), *l);
            self.resolve(&a, &x, l);
            *e = Name(x, l);
            return;
        }
        match e {
            Tuple(v, _) | List(v, _) => v.iter_mut().for_each(|x| self.expr(x)),
            ListRepeat(a, b, _) | Bin(_, a, b, _) | Catch(a, _, b, _) => {
                self.expr(a);
                self.expr(b)
            }
            ListComp(a, p, b, c, _) => {
                self.expr(a);
                self.pat(p);
                self.expr(b);
                if let Some(c) = c {
                    self.expr(c)
                }
            }
            Map(v, _) => v.iter_mut().for_each(|(a, b)| {
                self.expr(a);
                self.expr(b)
            }),
            ChanNew(t, x, _) => {
                self.ty(t);
                if let Some(x) = x {
                    self.expr(x)
                }
            }
            Call(c, args, _, _) => {
                self.expr(c);
                args.iter_mut().for_each(|a| self.expr(&mut a.e))
            }
            Index(x, v, _) => {
                self.expr(x);
                v.iter_mut().for_each(|y| self.expr(y))
            }
            Slice(x, a, b, _) => {
                self.expr(x);
                if let Some(a) = a {
                    self.expr(a)
                }
                if let Some(b) = b {
                    self.expr(b)
                }
            }
            Member(x, _, _) | TupleMember(x, _, _) | Neg(x, _) | BitNot(x, _) | Borrow(_, x, _) | Recv(x, _)
            | Comptime(x, _) | Not(x, _) => self.expr(x),
            Cast(x, t, _) => {
                self.expr(x);
                self.ty(t)
            }
            Lambda(ps, x, _) => {
                self.params(ps);
                self.expr(x)
            }
            Try(a, b, _) => {
                self.expr(a);
                if let Some(b) = b {
                    self.expr(b)
                }
            }
            If(a, b, c, _) => {
                self.expr(a);
                self.expr(b);
                self.expr(c)
            }
            _ => {}
        }
    }
    fn params(&mut self, ps: &mut [Param]) {
        for p in ps {
            if let Param::Named { ty, .. } = p {
                self.ty(ty);
            }
        }
    }
    fn sig(&mut self, s: &mut FnSig) {
        self.params(&mut s.params);
        if let Some(r) = &mut s.ret {
            self.ty(r);
        }
    }
    fn block(&mut self, b: &mut [Stmt]) {
        for s in b {
            self.stmt(s);
        }
    }
    fn stmt(&mut self, s: &mut Stmt) {
        match s {
            Stmt::Expr(e) | Stmt::Emit(e) | Stmt::Fail(e) | Stmt::Spawn(e) | Stmt::Check(e) | Stmt::Yield(e, _) => self.expr(e),
            Stmt::Assign(_, a, b) | Stmt::Send(a, b) => {
                self.expr(a);
                self.expr(b)
            }
            Stmt::Let(p, t, e, _) => {
                self.pat(p);
                if let Some(t) = t {
                    self.ty(t)
                }
                self.expr(e)
            }
            Stmt::Var(_, t, e, _) => {
                if let Some(t) = t {
                    self.ty(t)
                }
                self.expr(e)
            }
            Stmt::If(arms, els) => {
                for (c, b) in arms {
                    self.expr(c);
                    self.block(b)
                }
                if let Some(b) = els {
                    self.block(b)
                }
            }
            Stmt::While(c, bd, b) => {
                self.expr(c);
                if let Some(x) = bd {
                    self.expr(x)
                }
                self.block(b)
            }
            Stmt::Loop(bd, b) => {
                if let Some(x) = bd {
                    self.expr(x)
                }
                self.block(b)
            }
            Stmt::For(p, e, b, _) => {
                self.pat(p);
                self.expr(e);
                self.block(b)
            }
            Stmt::Match(e, arms, _) => {
                self.expr(e);
                for a in arms {
                    self.pat(&mut a.pat);
                    if let Some(g) = &mut a.guard {
                        self.expr(g)
                    }
                    self.block(&mut a.body)
                }
            }
            Stmt::Return(Some(e), _) => self.expr(e),
            Stmt::Select(arms, _) => {
                for a in arms {
                    match a {
                        SelArm::Recv(_, c, x) | SelArm::After(c, x) => {
                            self.expr(c);
                            self.block(x)
                        }
                        SelArm::Send(v, c, x) => {
                            self.expr(v);
                            self.expr(c);
                            self.block(x)
                        }
                        SelArm::Default(x) => self.block(x),
                    }
                }
            }
            Stmt::Arena(_, bd, b, _) | Stmt::Device(_, bd, b, _) => {
                if let Some(x) = bd {
                    self.expr(x)
                }
                self.block(b)
            }
            Stmt::Comptime(b) => self.block(b),
            _ => {}
        }
    }
    fn body(&mut self, b: &mut Body) {
        match b {
            Body::Arrow(e) => self.expr(e),
            Body::Block(s) => self.block(s),
        }
    }
    fn fields(&mut self, fs: &mut [Field]) {
        for f in fs {
            self.ty(&mut f.ty);
            if let Some(e) = &mut f.init {
                self.expr(e)
            }
        }
    }
    fn item(&mut self, it: &mut Item) {
        match it {
            Item::Fn(f) => {
                self.sig(&mut f.sig);
                self.body(&mut f.body)
            }
            Item::Impl { ty, members, .. } => {
                self.ty(ty);
                for f in members {
                    self.sig(&mut f.sig);
                    self.body(&mut f.body)
                }
            }
            Item::Const { ty, e, .. } => {
                if let Some(t) = ty {
                    self.ty(t)
                }
                self.expr(e)
            }
            Item::Check(e) => self.expr(e),
            Item::TypeAlias { ty, .. } => self.ty(ty),
            Item::Struct { fields, .. } | Item::State { fields, .. } => self.fields(fields),
            Item::Event { payload, .. } | Item::Error { payload, .. } => payload.iter_mut().for_each(|(_, t)| self.ty(t)),
            Item::Kind { variants, .. } => variants.iter_mut().for_each(|(_, p)| p.iter_mut().for_each(|(_, t)| self.ty(t))),
            Item::Step { pat, guard, body, .. } => {
                self.pat(pat);
                if let Some(g) = guard {
                    self.expr(g)
                }
                self.body(body)
            }
            Item::Actor { fields, handlers, fns, .. } => {
                self.fields(fields);
                for h in handlers {
                    self.pat(&mut h.pat);
                    if let Some(g) = &mut h.guard {
                        self.expr(g)
                    }
                    self.body(&mut h.body)
                }
                for f in fns {
                    self.sig(&mut f.sig);
                    self.body(&mut f.body)
                }
            }
            Item::Intent(_, xs, cks) => {
                cks.iter_mut().for_each(|e| self.expr(e));
                for x in xs {
                    x.events.iter_mut().for_each(|(_, a)| a.iter_mut().for_each(|a| self.expr(&mut a.e)));
                }
            }
            Item::Example(x) => x.events.iter_mut().for_each(|(_, a)| a.iter_mut().for_each(|a| self.expr(&mut a.e))),
            _ => {}
        }
    }
}
