//! Order-preserving A-normal form for write-back calls (spec 6.2, T34).
//!
//! The program emitters write back `&mut` arguments and receivers by
//! running a write-back call as a statement of its own. A write-back call
//! nested inside a larger expression (`1 + (f(&mut x) catch 0)`) is first
//! flattened by this one pass, shared by all three emitters: in the
//! strict (always evaluated) positions of the statement's expression,
//! every operand evaluated before the write-back call is bound to a
//! temporary in left-to-right order, then the call is bound, and the
//! residual expression reads the temporaries. Literals are left in place
//! (evaluating one has no effect and no cost). So reads, budget charges
//! and emits happen in the order of the original expression (6.2
//! evaluation order); `let` of a temporary and a read of it cost 0
//! (C.6.4).
//!
//! A write-back call in a conditionally evaluated position (the right of
//! `and`/`or`, a branch of `if`, a `catch` handler, a `try ... else`, a
//! lambda or comprehension body), in a `while` condition or an assignment
//! target, or with another write-back call in its own arguments, is left
//! as it is, and the emitter reports with a diagnostic that it does not
//! yet print it.
use crate::ast::*;
use std::collections::HashSet;

const MUTATING: &[&str] = &["append", "push", "pop", "clear", "sort", "insert", "remove"];

struct Cx {
    fns: HashSet<String>,
    methods: HashSet<String>,
    n: usize,
}

fn mut_params(f: &FnDecl) -> bool {
    f.sig.params.iter().any(|p| {
        matches!(p, Param::Named { ty: Ty::Ref(true, _), .. } | Param::SelfRef(true))
    })
}

/// The module with every flattenable nested write-back call flattened.
pub fn lower(m: &Module) -> Module {
    let mut cx = Cx { fns: HashSet::new(), methods: HashSet::new(), n: 0 };
    for it in &m.items {
        match it {
            Item::Fn(f) if mut_params(f) => {
                cx.fns.insert(f.sig.name.clone());
            }
            Item::Impl { members, .. } => {
                for f in members.iter().filter(|f| mut_params(f)) {
                    cx.methods.insert(f.sig.name.clone());
                }
            }
            Item::Actor { fns, .. } => {
                for f in fns.iter().filter(|f| mut_params(f)) {
                    cx.fns.insert(f.sig.name.clone());
                }
            }
            _ => {}
        }
    }
    let mut out = m.clone();
    for it in out.items.iter_mut() {
        match it {
            Item::Fn(f) => cx.fn_decl(f),
            Item::Impl { members, .. } => members.iter_mut().for_each(|f| cx.fn_decl(f)),
            Item::Actor { fns, handlers, .. } => {
                fns.iter_mut().for_each(|f| cx.fn_decl(f));
                for h in handlers {
                    if let Body::Block(b) = &mut h.body {
                        cx.block(b);
                    }
                }
            }
            Item::Step { body: Body::Block(b), .. } => cx.block(b),
            _ => {}
        }
    }
    out
}

impl Cx {
    fn is_wb(&self, e: &Expr) -> bool {
        let Expr::Call(f, _, _, _) = e else { return false };
        match &**f {
            Expr::Name(n, _) => self.fns.contains(n),
            Expr::Member(_, m, _) => MUTATING.contains(&m.as_str()) || self.methods.contains(m),
            _ => false,
        }
    }
    fn has_wb(&self, e: &Expr) -> bool {
        let mut found = false;
        walk(e, &mut |x| found |= self.is_wb(x));
        found
    }
    /// Already supported by the emitters as is: no write-back call, or one
    /// that is the whole expression, possibly under one `catch` or `try`.
    fn top_ok(&self, e: &Expr) -> bool {
        let core = match e {
            Expr::Catch(x, _, h, _) if !self.has_wb(h) => &**x,
            Expr::Try(x, None, _) => &**x,
            _ => e,
        };
        !self.has_wb(e) || (self.is_wb(core) && !self.args_have_wb(core))
    }
    fn args_have_wb(&self, call: &Expr) -> bool {
        let Expr::Call(f, args, _, _) = call else { return false };
        let recv = match &**f {
            Expr::Member(o, _, _) => self.has_wb(o),
            _ => false,
        };
        recv || args.iter().any(|a| self.has_wb(&a.e))
    }

    fn fn_decl(&mut self, f: &mut FnDecl) {
        match &mut f.body {
            Body::Block(b) => self.block(b),
            Body::Arrow(e) => {
                if self.top_ok(e) {
                    return;
                }
                let mut pre = vec![];
                if let Some(r) = self.flat(e, &mut pre) {
                    pre.push(Stmt::Expr(r));
                    f.body = Body::Block(pre);
                }
            }
        }
    }

    fn block(&mut self, b: &mut Vec<Stmt>) {
        let mut out = Vec::with_capacity(b.len());
        for mut s in std::mem::take(b) {
            self.nested(&mut s);
            let mut pre = vec![];
            let ok = match &mut s {
                Stmt::Expr(e) | Stmt::Emit(e) | Stmt::Fail(e) | Stmt::Let(_, _, e, _) | Stmt::Var(_, _, e, _) | Stmt::Return(Some(e), _) => self.rewrite(e, &mut pre),
                Stmt::Assign(_, place, rhs) if !self.has_wb(place) => self.rewrite(rhs, &mut pre),
                _ => true,
            };
            if ok {
                out.extend(pre);
            }
            out.push(s);
        }
        *b = out;
    }

    /// Rewrite `e` in place, its bindings appended to `pre`; when it
    /// cannot be rewritten, leave `e` and `pre` untouched and return false.
    fn rewrite(&mut self, e: &mut Expr, pre: &mut Vec<Stmt>) -> bool {
        if self.top_ok(e) {
            return true;
        }
        let mut p = vec![];
        match self.flat(e, &mut p) {
            Some(r) => {
                *e = r;
                pre.extend(p);
                true
            }
            None => false,
        }
    }

    fn nested(&mut self, s: &mut Stmt) {
        match s {
            Stmt::If(arms, els) => {
                arms.iter_mut().for_each(|(_, b)| self.block(b));
                if let Some(b) = els {
                    self.block(b);
                }
            }
            Stmt::While(_, _, b) | Stmt::Loop(_, b) | Stmt::For(_, _, b, _) | Stmt::Arena(_, _, b, _) | Stmt::Device(_, _, b, _) | Stmt::Comptime(b) => self.block(b),
            Stmt::Match(_, arms, _) => arms.iter_mut().for_each(|a| self.block(&mut a.body)),
            Stmt::Select(arms, _) => {
                for a in arms {
                    match a {
                        SelArm::Recv(_, _, b) | SelArm::Send(_, _, b) | SelArm::After(_, b) | SelArm::Default(b) => self.block(b),
                    }
                }
            }
            _ => {}
        }
    }

    fn tmp(&mut self, e: Expr, pre: &mut Vec<Stmt>) -> Expr {
        if inert(&e) {
            return e;
        }
        self.n += 1;
        let name = format!("anf_{}", self.n);
        let line = e.line();
        pre.push(Stmt::Let(Pat::Path(vec![name.clone()]), None, e, line));
        Expr::Name(name, line)
    }

    /// Flatten the strict children `cs` (in evaluation order): those
    /// before the last child that holds a write-back call are bound.
    fn children(&mut self, cs: Vec<Expr>, pre: &mut Vec<Stmt>) -> Option<Vec<Expr>> {
        let Some(j) = cs.iter().rposition(|c| self.has_wb(c)) else { return Some(cs) };
        let mut out = vec![];
        for (i, c) in cs.into_iter().enumerate() {
            if i < j {
                let r = if self.has_wb(&c) { self.flat(&c, pre)? } else { c };
                out.push(self.tmp(r, pre));
            } else if i == j {
                out.push(self.flat(&c, pre)?);
            } else {
                out.push(c);
            }
        }
        Some(out)
    }

    /// The residual of `e`; bindings go to `pre`. None: not flattenable.
    fn flat(&mut self, e: &Expr, pre: &mut Vec<Stmt>) -> Option<Expr> {
        if !self.has_wb(e) {
            return Some(e.clone());
        }
        let b = |x: Expr| Box::new(x);
        Some(match e {
            _ if self.is_wb(e) => {
                if self.args_have_wb(e) {
                    return None;
                }
                self.n += 1;
                let name = format!("anf_{}", self.n);
                pre.push(Stmt::Let(Pat::Path(vec![name.clone()]), None, e.clone(), e.line()));
                Expr::Name(name, e.line())
            }
            Expr::Catch(x, bind, h, l) if !self.has_wb(h) => Expr::Catch(b(self.flat(x, pre)?), bind.clone(), h.clone(), *l),
            Expr::Try(x, els, l) if els.as_ref().is_none_or(|x| !self.has_wb(x)) => Expr::Try(b(self.flat(x, pre)?), els.clone(), *l),
            Expr::Bin(op, x, y, l) if op != "and" && op != "or" => {
                let mut v = self.children(vec![(**x).clone(), (**y).clone()], pre)?;
                let y = v.pop()?;
                Expr::Bin(op.clone(), b(v.pop()?), b(y), *l)
            }
            Expr::Bin(op, x, y, l) if !self.has_wb(y) => Expr::Bin(op.clone(), b(self.flat(x, pre)?), y.clone(), *l),
            Expr::If(c, x, y, l) if !self.has_wb(x) && !self.has_wb(y) => Expr::If(b(self.flat(c, pre)?), x.clone(), y.clone(), *l),
            Expr::Neg(x, l) => Expr::Neg(b(self.flat(x, pre)?), *l),
            Expr::BitNot(x, l) => Expr::BitNot(b(self.flat(x, pre)?), *l),
            Expr::Not(x, l) => Expr::Not(b(self.flat(x, pre)?), *l),
            Expr::Cast(x, t, l) => Expr::Cast(b(self.flat(x, pre)?), t.clone(), *l),
            Expr::Member(x, m, l) => Expr::Member(b(self.flat(x, pre)?), m.clone(), *l),
            Expr::TupleMember(x, i, l) => Expr::TupleMember(b(self.flat(x, pre)?), *i, *l),
            Expr::Tuple(xs, l) => Expr::Tuple(self.children(xs.clone(), pre)?, *l),
            Expr::List(xs, l) => Expr::List(self.children(xs.clone(), pre)?, *l),
            Expr::Index(x, idx, l) => {
                let mut cs = vec![(**x).clone()];
                cs.extend(idx.iter().cloned());
                let mut v = self.children(cs, pre)?;
                let x = v.remove(0);
                Expr::Index(b(x), v, *l)
            }
            Expr::Call(f, args, l, sz) => {
                // a call that does not write back: callee receiver, then
                // arguments; borrows of places are kept as arguments
                if args.iter().any(|a| matches!(a.e, Expr::Borrow(..)) && self.has_wb(&a.e)) {
                    return None;
                }
                let (recv, m) = match &**f {
                    Expr::Member(o, m, _) => (Some((**o).clone()), Some(m.clone())),
                    Expr::Name(..) => (None, None),
                    _ => return None,
                };
                let mut cs: Vec<Expr> = recv.iter().cloned().collect();
                cs.extend(args.iter().map(|a| a.e.clone()));
                let mut v = self.children(cs, pre)?;
                let f2 = match (recv, m) {
                    (Some(_), Some(m)) => {
                        let o = v.remove(0);
                        Expr::Member(b(o), m, f.line())
                    }
                    _ => (**f).clone(),
                };
                let args2 = args.iter().zip(v).map(|(a, e)| Arg { label: a.label.clone(), e }).collect();
                Expr::Call(b(f2), args2, *l, *sz)
            }
            _ => return None,
        })
    }
}

/// Evaluating it has no effect, no cost and a value independent of time.
fn inert(e: &Expr) -> bool {
    matches!(e, Expr::Int(..) | Expr::Float(..) | Expr::Str(..) | Expr::Bool(..) | Expr::None(..) | Expr::Unit(..))
}

/// Visit every subexpression (including conditional ones).
fn walk(e: &Expr, f: &mut dyn FnMut(&Expr)) {
    f(e);
    use Expr::*;
    match e {
        Tuple(xs, _) | List(xs, _) => xs.iter().for_each(|x| walk(x, f)),
        ListRepeat(a, b, _) | Bin(_, a, b, _) => {
            walk(a, f);
            walk(b, f)
        }
        ListComp(a, _, b, c, _) => {
            walk(a, f);
            walk(b, f);
            if let Some(c) = c {
                walk(c, f)
            }
        }
        Map(ps, _) => ps.iter().for_each(|(a, b)| {
            walk(a, f);
            walk(b, f)
        }),
        ChanNew(_, Some(x), _) => walk(x, f),
        Call(c, args, _, _) => {
            walk(c, f);
            args.iter().for_each(|a| walk(&a.e, f))
        }
        Index(x, idx, _) => {
            walk(x, f);
            idx.iter().for_each(|i| walk(i, f))
        }
        Slice(x, a, b, _) => {
            walk(x, f);
            if let Some(a) = a {
                walk(a, f)
            }
            if let Some(b) = b {
                walk(b, f)
            }
        }
        Member(x, _, _) | TupleMember(x, _, _) | Neg(x, _) | BitNot(x, _) | Borrow(_, x, _) | Recv(x, _) | Comptime(x, _) | Cast(x, _, _) | Not(x, _) | Lambda(_, x, _) => walk(x, f),
        Catch(a, _, b, _) => {
            walk(a, f);
            walk(b, f)
        }
        Try(a, b, _) => {
            walk(a, f);
            if let Some(b) = b {
                walk(b, f)
            }
        }
        If(a, b, c, _) => {
            walk(a, f);
            walk(b, f);
            walk(c, f)
        }
        _ => {}
    }
}
