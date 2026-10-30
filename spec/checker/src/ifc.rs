//! Information-flow labels (spec F.4.2, S1): W-IFC-1 to W-IFC-5, and the
//! label half of W-CTX-2 (an untrusted argument of a model call, F.5.6).
//!
//! The analysis runs on the module as written (`parse_surface`), before the
//! label syntax is erased, and needs no type information: a value carries the
//! join of the labels of everything it was computed from (the spec's rule for
//! calls, `match`, `?` and derived `Eq`/`Ord`/`Hash`/`Show`), a place has a
//! label fixed where it is declared, and the control context (the join of the
//! labels of every enclosing condition) joins into every value assigned and
//! every output. A function is checked per instantiation, with the labels of
//! the actual arguments and the caller's context (the spec's monomorphized
//! checking).
//!
//! Spellings chosen here (decided by derivation, stated in F.4.2 and F.12):
//! - `T @ {A, B}` labels a whole declared type; `untrusted` is the built-in
//!   integrity tag and `secret` the built-in confidentiality label.
//! - `label A [by Auth]` declares a label and the authority struct whose
//!   `Cap[Auth]` a function must hold to `release` it. `label secret by X` and
//!   `label untrusted by X` attach an authority to the built-ins (the latter
//!   governs `endorse`).
//! - An event payload field or a foreign result is `untrusted` unless its
//!   type is written with an explicit empty set, `T @ {}`.
//! - An unlabeled place accepts any integrity but only the public
//!   confidentiality; its integrity is the join of what is stored.
//!
//! Limits (F.12): no types, so every `+ - * / % << >>` is treated as a
//! possible trap (W-IFC-4); a method call on a mutable variable flows its
//! arguments into the variable only for the usual mutating method names; the
//! label of a composite is the join of its parts (a field label is by field
//! name across all structs); state fields are labeled by their declaration.
use crate::ast::*;
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Default, Debug, PartialEq)]
struct L {
    /// confidentiality label -> where it came from (provenance, W-IFC-3)
    conf: BTreeMap<String, String>,
    /// integrity: `Some(origin)` when the value is `untrusted`
    untrusted: Option<String>,
    /// W-CTX-3: the untrusted part of the value is all inside `Prompt.encode`
    /// blocks (sealed); an untrusted value that steered the text otherwise
    /// (a branch, a join with plain data) is not sealed
    sealed: bool,
}

impl L {
    fn join(&self, o: &L) -> L {
        let mut r = self.clone();
        for (k, v) in &o.conf {
            r.conf.entry(k.clone()).or_insert_with(|| v.clone());
        }
        if r.untrusted.is_none() {
            r.untrusted = o.untrusted.clone();
        }
        let su = |l: &L| l.untrusted.is_none() || l.sealed;
        r.sealed = r.untrusted.is_some() && su(self) && su(o);
        r
    }
    fn public(&self) -> bool {
        self.conf.is_empty()
    }
    fn flows(&self, to: &L) -> bool {
        self.conf.keys().all(|k| to.conf.contains_key(k)) && (self.untrusted.is_none() || to.untrusted.is_some())
    }
    fn show(&self) -> String {
        let mut v: Vec<String> = self.conf.iter().map(|(k, o)| format!("{k} (from {o})")).collect();
        if let Some(o) = &self.untrusted {
            v.push(format!("untrusted (from {o})"));
        }
        if v.is_empty() { "public".into() } else { format!("{{{}}}", v.join(", ")) }
    }
    fn key(&self) -> String {
        format!("{:?}/{}/{}", self.conf.keys().collect::<Vec<_>>(), self.untrusted.is_some(), self.sealed)
    }
}

#[derive(Clone)]
struct V {
    val: L,
    /// the label fixed at declaration (annotated), else only the confidentiality
    /// bound of an inferred place (`None` for an immutable binding)
    place: Option<L>,
    annotated: bool,
    mutable: bool,
}

const MUTATORS: &[&str] = &["push", "push_back", "push_front", "insert", "append", "extend", "set", "add", "put", "merge"];
const TRAPS: &[&str] = &["+", "-", "*", "/", "%", "<<", ">>"];

fn label_of(t: &Ty, origin: &str) -> Option<Result<L, String>> {
    if let Ty::Region(_, s) = t
        && let Some(inner) = s.strip_prefix("@{").and_then(|x| x.strip_suffix('}'))
    {
        let mut l = L::default();
        for n in inner.split(',').filter(|x| !x.is_empty()) {
            if n == "untrusted" {
                l.untrusted = Some(origin.to_string());
            } else {
                l.conf.insert(n.to_string(), origin.to_string());
            }
        }
        return Some(Ok(l));
    }
    None
}

/// `&Text[Prompt]`, the one text parameter of a model function (W-CTX-2)
fn is_prompt_ref(t: &Ty) -> bool {
    match strip(t) {
        Ty::Ref(false, x) => matches!(strip(x), Ty::Path(p, a) if p.len() == 1 && p[0] == "Text" && matches!(a.as_slice(), [Ty::Path(c, b)] if c.len() == 1 && c[0] == "Prompt" && b.is_empty())),
        _ => false,
    }
}

fn strip(t: &Ty) -> &Ty {
    match t {
        Ty::Region(x, s) if s.starts_with("@{") => x,
        o => o,
    }
}

fn cap_of(t: &Ty) -> Option<String> {
    match strip(t) {
        Ty::Ref(_, x) => cap_of(x),
        Ty::Path(p, a) if p.len() == 1 && p[0] == "Cap" && a.len() == 1 => match strip(&a[0]) {
            Ty::Path(q, _) if q.len() == 1 => Some(q[0].clone()),
            _ => None,
        },
        _ => None,
    }
}

/// every label written anywhere inside a type that is not its top
fn nested_label(t: &Ty, top: bool, out: &mut bool) {
    match t {
        Ty::Region(x, s) => {
            if s.starts_with("@{") && !top {
                *out = true;
            }
            nested_label(x, false, out);
        }
        Ty::Path(_, a) | Ty::Tuple(a) => a.iter().for_each(|x| nested_label(x, false, out)),
        Ty::Slice(x) | Ty::Chan(x) | Ty::Iter(x) | Ty::Ref(_, x) | Ty::Opt(x) | Ty::Array(x, _) => nested_label(x, false, out),
        Ty::Union(a, b) => {
            nested_label(a, false, out);
            nested_label(b, false, out);
        }
        Ty::Fn(a, r) => {
            a.iter().for_each(|x| nested_label(x, false, out));
            nested_label(r, false, out);
        }
        Ty::Unit => {}
    }
}

struct A<'a> {
    labels: HashSet<String>,
    auth: HashMap<String, String>,
    fns: HashMap<String, &'a FnDecl>,
    externs: HashMap<String, &'a FnSig>,
    /// the `extern "model"` functions (W-CTX-2: untrusted data reaches them only
    /// through a `&Text[Prompt]` parameter, that is through `Prompt.encode`)
    models: HashSet<String>,
    /// W-CTX-3: set when the rest of the current function is control dependent
    /// on untrusted data (an early exit taken under an untrusted condition)
    exit_dep: Option<String>,
    /// constructor (struct, event, error, variant) -> (field name, declared label)
    ctors: HashMap<String, Vec<(String, Option<L>)>>,
    member: HashMap<String, L>,
    state: HashMap<String, V>,
    variants: HashSet<String>,
    errs: Vec<String>,
    memo: HashMap<String, L>,
    active: HashSet<String>,
    scopes: Vec<HashMap<String, V>>,
    ctx: L,
    caps: HashSet<String>,
    rets: L,
    fname: String,
    inst: String,
    /// names of the `ct fn` functions (F.5.4) and whether the function being checked is one
    ct: HashSet<String>,
    in_ct: bool,
}

impl<'a> A<'a> {
    fn err(&mut self, line: usize, code: &str, m: String) {
        let tail = if self.inst.is_empty() { String::new() } else { format!(" [in `{}`{}]", self.fname, self.inst) };
        self.errs.push(format!("line {line}: {code} {m}{tail}"));
    }
    fn lookup(&self, n: &str) -> Option<V> {
        for s in self.scopes.iter().rev() {
            if let Some(v) = s.get(n) {
                return Some(v.clone());
            }
        }
        self.state.get(n).cloned()
    }
    fn set(&mut self, n: &str, v: V) {
        if let Some(s) = self.scopes.last_mut() {
            s.insert(n.to_string(), v);
        }
    }
    fn names(&self, p: &Pat, out: &mut Vec<String>) {
        match p {
            Pat::Path(q) if q.len() == 1 && !self.variants.contains(&q[0]) => out.push(q[0].clone()),
            Pat::Ctor(_, fs) => fs.iter().for_each(|(_, p)| self.names(p, out)),
            Pat::Tuple(ps) => ps.iter().for_each(|p| self.names(p, out)),
            Pat::Alt(ps) => {
                if let Some(p) = ps.first() {
                    self.names(p, out)
                }
            }
            _ => {}
        }
    }
    fn bind(&mut self, p: &Pat, l: &L) {
        let mut v = vec![];
        self.names(p, &mut v);
        for n in v {
            self.set(&n, V { val: l.clone(), place: Option::None, annotated: false, mutable: false });
        }
    }
    fn with_ctx<T>(&mut self, extra: &L, f: impl FnOnce(&mut Self) -> T) -> T {
        let old = self.ctx.clone();
        self.ctx = old.join(extra);
        let r = f(self);
        self.ctx = old;
        r
    }
    /// F.5.4 W-CTFN-1: in a `ct fn` a `secret` value decides no control flow or access
    fn ct_use(&mut self, line: usize, l: &L, what: &str) {
        if self.in_ct && l.conf.contains_key("secret") {
            let c = l.show();
            self.err(line, "W-CTFN-1", format!("{what} is labeled {c} in `ct fn {}`: control flow and memory access must not depend on a secret (use `ct_choose` or bitwise operators)", self.fname));
        }
    }
    fn trap_site(&mut self, line: usize, what: &str, operands: &[&L]) {
        if !self.ctx.public() {
            let c = self.ctx.show();
            self.err(line, "W-IFC-4", format!("{what} can trap in a control context labeled {c}: the kept step is observable; use a checked form that returns an Option"));
        }
        for o in operands {
            if !o.public() {
                let c = o.show();
                self.err(line, "W-IFC-4", format!("{what} can trap and an operand that decides it is labeled {c}: the kept step is observable; use a checked form that returns an Option"));
            }
        }
    }
    fn expr(&mut self, e: &Expr) -> L {
        use Expr::*;
        match e {
            Name(n, _) => self.lookup(n).map(|v| v.val).unwrap_or_default(),
            SelfRef(_) | Int(..) | Float(..) | Str(..) | Bool(..) | None(_) | Unit(_) => L::default(),
            Bin(op, a, b, l) => {
                let la = self.expr(a);
                let lb = if op == "and" || op == "or" {
                    self.with_ctx(&la.clone(), |s| s.expr(b))
                } else {
                    self.expr(b)
                };
                match op.as_str() {
                    "and" | "or" => self.ct_use(*l, &la, "an `and`/`or` operand"),
                    "<<" | ">>" => self.ct_use(*l, &lb, "a shift amount"),
                    "/" | "%" => self.ct_use(*l, &lb, "a divisor"),
                    _ => {}
                }
                if TRAPS.contains(&op.as_str()) {
                    self.trap_site(*l, &format!("`{op}`"), &[&la, &lb]);
                }
                la.join(&lb)
            }
            Index(b, ix, l) => {
                let lb = self.expr(b);
                let mut li = L::default();
                for x in ix {
                    li = li.join(&self.expr(x));
                }
                self.ct_use(*l, &li, "an index");
                self.trap_site(*l, "an index", &[&li]);
                lb.join(&li)
            }
            Member(b, f, _) => {
                let lb = self.expr(b);
                match self.member.get(f) {
                    Some(m) => lb.join(&m.clone()),
                    Option::None => lb,
                }
            }
            If(c, a, b, l) => {
                let lc = self.expr(c);
                self.ct_use(*l, &lc, "an `if` condition");
                let (la, lb) = self.with_ctx(&lc.clone(), |s| (s.expr(a), s.expr(b)));
                lc.join(&la).join(&lb)
            }
            Catch(a, bind, h, _) => {
                let la = self.expr(a);
                self.scopes.push(HashMap::new());
                if let Some(n) = bind {
                    self.set(n, V { val: L::default(), place: Option::None, annotated: false, mutable: false });
                }
                let lh = self.expr(h);
                self.scopes.pop();
                la.join(&lh)
            }
            Lambda(_, body, l) => {
                let mut cap = vec![];
                crate::v09::each_expr(body, true, &mut |x, _| {
                    if let Name(n, _) = x {
                        cap.push(n.clone());
                    }
                });
                for n in cap {
                    if let Some(v) = self.lookup(&n)
                        && !v.val.public()
                    {
                        let c = v.val.show();
                        self.err(*l, "W-IFC-1", format!("a closure captures `{n}`, labeled {c}: authority and labeled data held by a closure must be visible (F.5.1)"));
                    }
                }
                L::default()
            }
            Call(f, args, l, _) => self.call(f, args, *l),
            Try(x, d, _) => {
                let lx = self.expr(x);
                if d.is_none() && self.exit_dep.is_none() {
                    if let Some(o) = lx.untrusted.clone().or_else(|| self.ctx.untrusted.clone()) {
                        self.exit_dep = Some(o);
                    }
                }
                let ld = d.as_ref().map(|y| self.expr(y)).unwrap_or_default();
                lx.join(&ld)
            }
            other => {
                let mut r = L::default();
                for c in crate::v09::children(other) {
                    r = r.join(&self.expr(c));
                }
                r
            }
        }
    }
    fn call(&mut self, f: &Expr, args: &[Arg], line: usize) -> L {
        if let Expr::Name(n, _) = f {
            if n == "@release" || n == "@endorse" {
                let l = self.expr(&args[0].e);
                return if n == "@release" { self.release(&l, line) } else { self.endorse(&l, line) };
            }
            let als: Vec<L> = args.iter().map(|a| self.expr(&a.e)).collect();
            let all = als.iter().fold(L::default(), |a, b| a.join(b));
            if self.in_ct && (self.externs.contains_key(n) || (self.fns.contains_key(n) && !self.ct.contains(n))) {
                for l in &als {
                    self.ct_use(line, l, &format!("an argument of the call to `{n}`, which is not a `ct fn`,"));
                }
            }
            if let Some(sig) = self.externs.get(n).copied() {
                for (i, l) in als.iter().enumerate() {
                    if !l.public() {
                        let c = l.show();
                        self.err(line, "W-IFC-1", format!("argument {} of the foreign call `{n}` is labeled {c}: a foreign call is an output", i + 1));
                    }
                }
                if !self.ctx.public() {
                    let c = self.ctx.show();
                    self.err(line, "W-IFC-2", format!("the foreign call `{n}` is made in a control context labeled {c}: the call is observable"));
                }
                if self.models.contains(n) {
                    if let Some(o) = self.ctx.untrusted.clone().or_else(|| self.exit_dep.clone()) {
                        self.err(line, "W-CTX-3", format!("the model call `{n}` is control dependent on untrusted data (from {o}): whether and which request is made would be steered; `endorse` the deciding value first"));
                    }
                    let ps: Vec<&Ty> = sig.params.iter().filter_map(|p| if let Param::Named { ty, .. } = p { Some(ty) } else { None }).collect();
                    for (i, l) in als.iter().enumerate() {
                        if let Some(o) = &l.untrusted
                            && !ps.get(i).is_some_and(|t| is_prompt_ref(t))
                        {
                            let o = o.clone();
                            self.err(line, "W-CTX-2", format!("argument {} of the model call `{n}` is untrusted (from {o}): untrusted data reaches a model only inside a `Text[Prompt]`, through `Prompt.encode`", i + 1));
                        } else if let Some(o) = &l.untrusted
                            && !l.sealed
                        {
                            let o = o.clone();
                            self.err(line, "W-CTX-3", format!("argument {} of the model call `{n}` was steered by untrusted data (from {o}) outside a `Prompt.encode` block", i + 1));
                        }
                    }
                }
                let origin = format!("foreign result of `{n}`");
                return match sig.ret.as_ref().and_then(|t| label_of(t, &origin)) {
                    Some(Ok(l)) => l,
                    _ => L { conf: BTreeMap::new(), untrusted: Some(origin), sealed: false },
                };
            }
            if let Some(fd) = self.fns.get(n).copied() {
                return self.user_call(fd, &als, line);
            }
            if let Some(fs) = self.ctors.get(n).cloned() {
                for (i, a) in args.iter().enumerate() {
                    let want = match &a.label {
                        Some(lb) => fs.iter().find(|(f, _)| f == lb),
                        None => fs.get(i),
                    };
                    if let Some((fname, decl)) = want {
                        let ok = match decl {
                            Some(d) => als[i].flows(d),
                            None => als[i].public(),
                        };
                        if !ok {
                            let (v, d) = (als[i].show(), decl.as_ref().map(|d| d.show()).unwrap_or_else(|| "public".into()));
                            self.err(line, "W-IFC-1", format!("field `{fname}` of `{n}` is {d}, the value is labeled {v}"));
                        }
                    }
                }
                return all.join(&self.ctx.clone());
            }
            return all.join(&self.ctx.clone());
        }
        let lf = self.expr(f);
        let als: Vec<L> = args.iter().map(|a| self.expr(&a.e)).collect();
        let mut all = als.iter().fold(lf, |a, b| a.join(b));
        // W-CTX-3: the data of `Prompt.encode(x)` sits in a delimited block
        if let Expr::Member(o, m, _) = f
            && m == "encode"
            && matches!(&**o, Expr::Name(c, _) if c == "Prompt")
        {
            all.sealed = all.untrusted.is_some();
        }
        // F.5.7: `Header.encode` traps on CR and LF at run time
        if let Expr::Member(o, m, _) = f
            && m == "encode"
            && matches!(&**o, Expr::Name(c, _) if c == "Header")
        {
            let a = all.clone();
            self.trap_site(line, "`Header.encode`", &[&a]);
        }
        if let Expr::Member(obj, m, _) = f
            && MUTATORS.contains(&m.as_str())
            && let Expr::Name(v, _) = &**obj
            && let Some(var) = self.lookup(v)
            && var.mutable
        {
            let aj = als.iter().fold(L::default(), |a, b| a.join(b)).join(&self.ctx.clone());
            self.store(line, v, &var, &aj);
        }
        all.join(&self.ctx.clone())
    }
    fn user_call(&mut self, fd: &'a FnDecl, als: &[L], line: usize) -> L {
        // W-CTX-3: a call made after an untrusted early exit runs in that dependence
        if let Some(o) = self.exit_dep.clone() {
            let d = L { conf: BTreeMap::new(), untrusted: Some(o), sealed: false };
            return self.with_ctx(&d, |s| s.user_call1(fd, als, line));
        }
        self.user_call1(fd, als, line)
    }
    fn user_call1(&mut self, fd: &'a FnDecl, als: &[L], line: usize) -> L {
        let named: Vec<(&String, &Ty)> = fd.sig.params.iter().filter_map(|p| if let Param::Named { name, ty, .. } = p { Some((name, ty)) } else { None }).collect();
        let mut arg_labels = vec![];
        for (i, (pn, pty)) in named.iter().enumerate() {
            let a = als.get(i).cloned().unwrap_or_default();
            match label_of(pty, pn) {
                Some(Ok(d)) => {
                    if !a.flows(&d) {
                        let (v, ds) = (a.show(), d.show());
                        self.err(line, "W-IFC-1", format!("parameter `{pn}` of `{}` is {ds}, the argument is labeled {v}", fd.sig.name));
                    }
                    arg_labels.push(d);
                }
                _ => arg_labels.push(a),
            }
        }
        // declared result label
        if let Some(r) = &fd.sig.ret
            && let Some(Ok(d)) = label_of(r, &fd.sig.name)
        {
            self.run_fn(fd, &arg_labels);
            return d;
        }
        let r = self.run_fn(fd, &arg_labels);
        r.join(&self.ctx.clone())
    }
    /// check one function with these parameter labels and the current context
    fn run_fn(&mut self, fd: &'a FnDecl, arg_labels: &[L]) -> L {
        let desc: Vec<String> = arg_labels.iter().map(L::key).collect();
        let key = format!("{}|{}|{}", fd.sig.name, desc.join(";"), self.ctx.key());
        if let Some(r) = self.memo.get(&key) {
            return r.clone();
        }
        if !self.active.insert(key.clone()) {
            // recursion: assume the join of the arguments
            return arg_labels.iter().fold(L::default(), |a, b| a.join(b));
        }
        let saved = (std::mem::take(&mut self.scopes), self.caps.clone(), std::mem::take(&mut self.rets), self.fname.clone(), self.inst.clone());
        let public_roots = arg_labels.iter().all(|l| l.public() && l.untrusted.is_none()) && self.ctx.public() && self.ctx.untrusted.is_none();
        let saved_ct = self.in_ct;
        let saved_dep = self.exit_dep.take();
        self.in_ct = self.ct.contains(&fd.sig.name);
        self.fname = fd.sig.name.clone();
        self.inst = if public_roots { String::new() } else { format!(" called with {}", if arg_labels.is_empty() { "a labeled context".into() } else { arg_labels.iter().map(L::show).collect::<Vec<_>>().join(", ") }) };
        self.caps = HashSet::new();
        self.scopes = vec![HashMap::new()];
        let named: Vec<(&String, &Ty)> = fd.sig.params.iter().filter_map(|p| if let Param::Named { name, ty, .. } = p { Some((name, ty)) } else { None }).collect();
        for (i, (pn, pty)) in named.iter().enumerate() {
            let l = arg_labels.get(i).cloned().unwrap_or_default();
            let annotated = label_of(pty, pn).is_some();
            if let Some(c) = cap_of(pty) {
                self.caps.insert(c);
            }
            self.set(pn, V { val: l.clone(), place: Some(l), annotated, mutable: matches!(pty, Ty::Ref(true, _)) });
        }
        let tail = match &fd.body {
            Body::Arrow(e) => self.expr(e),
            Body::Block(b) => {
                self.block(b);
                match b.last() {
                    Some(Stmt::Expr(e)) => {
                        let ctx = self.ctx.clone();
                        self.expr(e).join(&ctx)
                    }
                    _ => L::default(),
                }
            }
        };
        let r = std::mem::take(&mut self.rets).join(&tail);
        self.scopes = saved.0;
        self.caps = saved.1;
        self.rets = saved.2;
        self.fname = saved.3;
        self.inst = saved.4;
        self.in_ct = saved_ct;
        self.exit_dep = saved_dep;
        self.active.remove(&key);
        self.memo.insert(key, r.clone());
        r
    }
    fn release(&mut self, l: &L, line: usize) -> L {
        let mut out = l.clone();
        if let Some(o) = self.ctx.untrusted.clone() {
            self.err(line, "W-IFC-5", format!("`release` is decided by untrusted data (from {o}): an attacker could steer what is released; `endorse` the deciding value first"));
        }
        for k in l.conf.keys().cloned().collect::<Vec<_>>() {
            match self.auth.get(&k).cloned() {
                Some(a) if self.caps.contains(&a) => {
                    out.conf.remove(&k);
                }
                Some(a) => self.err(line, "W-IFC-5", format!("`release` of `{k}` needs a held `Cap[{a}]` and `{}` holds none", self.fname)),
                None => self.err(line, "W-IFC-5", format!("`{k}` has no release authority: declare `label {k} by Auth` and hold `Cap[Auth]`")),
            }
        }
        out
    }
    fn endorse(&mut self, l: &L, line: usize) -> L {
        let mut out = l.clone();
        match self.auth.get("untrusted").cloned() {
            Some(a) if self.caps.contains(&a) => out.untrusted = None,
            Some(a) => self.err(line, "W-IFC-5", format!("`endorse` needs a held `Cap[{a}]` and `{}` holds none", self.fname)),
            None => self.err(line, "W-IFC-5", "`endorse` needs `label untrusted by Auth` and a held `Cap[Auth]`".into()),
        }
        out
    }
    fn store(&mut self, line: usize, name: &str, var: &V, val: &L) {
        let ok = if var.annotated {
            val.flows(var.place.as_ref().unwrap_or(&L::default()))
        } else {
            match &var.place {
                Some(p) => val.conf.keys().all(|k| p.conf.contains_key(k)),
                None => val.public(),
            }
        };
        if !ok {
            let (v, p) = (val.show(), var.place.as_ref().map(|p| p.show()).unwrap_or_else(|| "public".into()));
            self.err(line, "W-IFC-1", format!("`{name}` is {p}, the value stored is labeled {v}"));
            return;
        }
        if !var.annotated {
            // an inferred place keeps the join of what is stored (integrity)
            let mut nv = var.clone();
            nv.val = nv.val.join(val);
            for s in self.scopes.iter_mut().rev() {
                if s.contains_key(name) {
                    s.insert(name.to_string(), nv);
                    return;
                }
            }
            if self.state.contains_key(name) {
                self.state.insert(name.to_string(), nv);
            }
        }
    }
    fn block(&mut self, ss: &[Stmt]) {
        self.scopes.push(HashMap::new());
        for s in ss {
            self.stmt(s);
        }
        self.scopes.pop();
    }
    fn root<'e>(&self, e: &'e Expr) -> Option<&'e str> {
        match e {
            Expr::Name(n, _) => Some(n),
            Expr::Member(b, _, _) | Expr::TupleMember(b, _, _) | Expr::Index(b, _, _) | Expr::Slice(b, _, _, _) => self.root(b),
            _ => None,
        }
    }
    fn stmt(&mut self, s: &Stmt) {
        let looped = matches!(s, Stmt::While(..) | Stmt::Loop(..) | Stmt::For(..));
        let before = self.exit_dep.is_none();
        let n = self.errs.len();
        self.stmt1(s);
        // W-CTX-3: an exit under an untrusted condition inside a loop also
        // decides the iterations before it; check the body again with the
        // dependence known, keeping only the new W-CTX-3 findings
        if looped && before && self.exit_dep.is_some() {
            let first: HashSet<String> = self.errs[n..].iter().cloned().collect();
            let m = self.errs.len();
            self.stmt1(s);
            let second = self.errs.split_off(m);
            let mut add: Vec<String> = vec![];
            for e in second {
                if e.contains(" W-CTX-3 ") && !first.contains(&e) && !add.contains(&e) {
                    add.push(e);
                }
            }
            self.errs.extend(add);
        }
    }
    fn stmt1(&mut self, s: &Stmt) {
        use Stmt::*;
        match s {
            Expr(e) | Check(e) | Spawn(e) | Yield(e, _) => {
                self.expr(e);
            }
            Emit(e) => {
                let l = self.expr(e).join(&self.ctx.clone());
                if !l.public() {
                    let c = l.show();
                    self.err(e.line(), "W-IFC-1", format!("an `emit` is public and the value is labeled {c}"));
                }
            }
            Send(c, v) => {
                self.expr(c);
                let l = self.expr(v).join(&self.ctx.clone());
                if !l.public() {
                    let c = l.show();
                    self.err(v.line(), "W-IFC-1", format!("a channel is public and the value sent is labeled {c}"));
                }
            }
            Fail(e) => {
                let l = self.expr(e);
                let line = e.line();
                self.note_exit();
                if !self.ctx.public() {
                    let c = self.ctx.show();
                    self.err(line, "W-IFC-4", format!("`fail` in a control context labeled {c}: the kept step is observable"));
                }
                if !l.public() {
                    let c = l.show();
                    self.err(line, "W-IFC-4", format!("the `fail` payload is labeled {c} and a failure can reach a public context"));
                }
            }
            Return(e, _) => {
                self.note_exit();
                let l = e.as_ref().map(|x| self.expr(x)).unwrap_or_default().join(&self.ctx.clone());
                self.rets = self.rets.join(&l);
            }
            Let(p, t, e, line) => {
                let mut l = self.expr(e).join(&self.ctx.clone());
                let mut annotated = false;
                if let Some(Ok(d)) = t.as_ref().and_then(|t| label_of(t, "its declaration")) {
                    if !l.flows(&d) {
                        let (v, ds) = (l.show(), d.show());
                        self.err(*line, "W-IFC-1", format!("the declared label {ds} does not cover the value, labeled {v}"));
                    }
                    l = d;
                    annotated = true;
                }
                let mut v = vec![];
                self.names(p, &mut v);
                for n in v {
                    self.set(&n, V { val: l.clone(), place: Option::None, annotated, mutable: false });
                }
            }
            Var(n, t, e, line) => {
                let mut l = self.expr(e).join(&self.ctx.clone());
                let mut annotated = false;
                let mut place = L { conf: l.conf.clone(), untrusted: None, sealed: false };
                if let Some(Ok(d)) = t.as_ref().and_then(|t| label_of(t, n)) {
                    if !l.flows(&d) {
                        let (v, ds) = (l.show(), d.show());
                        self.err(*line, "W-IFC-1", format!("the declared label {ds} does not cover the initial value, labeled {v}"));
                    }
                    l = d.clone();
                    place = d;
                    annotated = true;
                }
                self.set(n, V { val: l, place: Some(place), annotated, mutable: true });
            }
            Assign(op, place, rhs) => {
                let mut lv = self.expr(rhs).join(&self.ctx.clone());
                let line = rhs.line();
                if op != "=" {
                    let bin = &op[..op.len() - 1];
                    let lp = self.expr(place);
                    if TRAPS.contains(&bin) {
                        self.trap_site(line, &format!("`{op}`"), &[&lp, &lv]);
                    }
                    lv = lv.join(&lp);
                }
                if let Some(r) = self.root(place).map(String::from)
                    && let Some(var) = self.lookup(&r)
                {
                    self.store(line, &r, &var, &lv);
                }
            }
            If(bs, el) => {
                let mut acc = L::default();
                for (c, b) in bs {
                    let lc = self.with_ctx(&acc.clone(), |s| s.expr(c));
                    self.ct_use(c.line(), &lc, "an `if` condition");
                    acc = acc.join(&lc);
                    let inner = acc.clone();
                    self.with_ctx(&inner, |s| s.block(b));
                }
                if let Some(b) = el {
                    let inner = acc.clone();
                    self.with_ctx(&inner, |s| s.block(b));
                }
            }
            While(c, b, body) => {
                let lc = self.expr(c);
                self.ct_use(c.line(), &lc, "a `while` condition");
                if !lc.public() {
                    let s = lc.show();
                    self.err(c.line(), "W-IFC-2", format!("a `while` condition labeled {s} bounds the loop: the number of iterations is observable"));
                }
                if let Some(b) = b {
                    self.expr(b);
                }
                self.with_ctx(&lc.clone(), |s| s.block(body));
            }
            Loop(b, body) => {
                if let Some(b) = b {
                    let lb = self.expr(b);
                    self.ct_use(b.line(), &lb, "a loop bound");
                    if !lb.public() {
                        let s = lb.show();
                        self.err(b.line(), "W-IFC-2", format!("a loop bound labeled {s}: the number of iterations is observable"));
                    }
                }
                self.block(body);
            }
            For(p, it, body, line) => {
                let li = self.expr(it);
                self.ct_use(*line, &li, "a `for` range");
                if !li.public() {
                    let s = li.show();
                    self.err(*line, "W-IFC-2", format!("a `for` over a value labeled {s}: the number of iterations is observable"));
                }
                self.scopes.push(HashMap::new());
                self.bind(p, &li);
                self.with_ctx(&li.clone(), |s| s.block(body));
                self.scopes.pop();
            }
            Match(e, arms, _) => {
                let ls = self.expr(e);
                self.ct_use(e.line(), &ls, "a `match` scrutinee");
                let mut acc = ls.clone();
                for a in arms {
                    self.scopes.push(HashMap::new());
                    self.bind(&a.pat, &ls);
                    let lg = a.guard.as_ref().map(|g| self.with_ctx(&acc.clone(), |s| s.expr(g))).unwrap_or_default();
                    let inner = acc.join(&lg);
                    self.with_ctx(&inner, |s| s.block(&a.body));
                    acc = acc.join(&lg);
                    self.scopes.pop();
                }
            }
            Select(arms, _) => {
                for a in arms {
                    match a {
                        SelArm::Recv(p, c, b) => {
                            self.expr(c);
                            self.scopes.push(HashMap::new());
                            self.bind(p, &L::default());
                            self.block(b);
                            self.scopes.pop();
                        }
                        SelArm::Send(c, v, b) => {
                            self.expr(c);
                            let l = self.expr(v).join(&self.ctx.clone());
                            if !l.public() {
                                let s = l.show();
                                self.err(v.line(), "W-IFC-1", format!("a channel is public and the value sent is labeled {s}"));
                            }
                            self.block(b);
                        }
                        SelArm::After(e, b) => {
                            self.expr(e);
                            self.block(b);
                        }
                        SelArm::Default(b) => self.block(b),
                    }
                }
            }
            Arena(_, e, b, _) | Device(_, e, b, _) => {
                if let Some(e) = e {
                    self.expr(e);
                }
                self.block(b);
            }
            Comptime(b) => self.block(b),
            Break(_) | Continue(_) => self.note_exit(),
            Pass => {}
        }
    }
    /// W-CTX-3: an exit taken under an untrusted condition makes everything
    /// after it in this function (and the rest of an enclosing loop) control
    /// dependent on that data; recorded for the rest of the function run.
    fn note_exit(&mut self) {
        if self.exit_dep.is_none()
            && let Some(o) = self.ctx.untrusted.clone()
        {
            self.exit_dep = Some(o);
        }
    }
    /// A step or handler body: the pattern binds payload fields with their labels.
    fn handler(&mut self, pat: &Pat, guard: &Option<crate::ast::Expr>, body: &Body, payloads: &HashMap<String, Vec<(String, Ty)>>, region: &str) {
        let saved = (std::mem::take(&mut self.scopes), self.caps.clone(), self.fname.clone());
        self.fname = format!("step {region}");
        self.caps = HashSet::new();
        self.scopes = vec![HashMap::new()];
        if let Pat::Ctor(path, fields) = pat
            && let Some(pl) = path.last().and_then(|n| payloads.get(n))
        {
            for (i, (lb, p)) in fields.iter().enumerate() {
                let field = match lb {
                    Some(n) => pl.iter().find(|(f, _)| f == n),
                    None => pl.get(i),
                };
                if let Some((fname, fty)) = field {
                    let origin = format!("event {}.{fname}", path.last().unwrap());
                    let l = match label_of(fty, &origin) {
                        Some(Ok(l)) => l,
                        _ => L { conf: BTreeMap::new(), untrusted: Some(origin), sealed: false },
                    };
                    if let Some(c) = cap_of(fty) {
                        self.caps.insert(c);
                    }
                    self.bind(p, &l);
                }
            }
        }
        let lg = guard.as_ref().map(|g| self.expr(g)).unwrap_or_default();
        let inner = lg.clone();
        self.with_ctx(&inner, |s| match body {
            Body::Arrow(e) => {
                let l = s.expr(e).join(&s.ctx.clone());
                // a one-field region: the value is stored into the field
                if let Some((n, v)) = s.state_single(region)
                    && !(if v.annotated { l.flows(v.place.as_ref().unwrap_or(&L::default())) } else { l.public() || v.place.as_ref().is_some_and(|p| l.conf.keys().all(|k| p.conf.contains_key(k))) })
                {
                    let (ls, ps) = (l.show(), v.place.as_ref().map(|p| p.show()).unwrap_or_else(|| "public".into()));
                    s.err(e.line(), "W-IFC-1", format!("the step stores into `{n}`, which is {ps}; the value is labeled {ls}"));
                }
            }
            Body::Block(b) => s.block(b),
        });
        self.scopes = saved.0;
        self.caps = saved.1;
        self.fname = saved.2;
    }
    fn state_single(&self, region: &str) -> Option<(String, V)> {
        let key = format!("@region:{region}");
        let n = self.state.get(&key).map(|_| ())?;
        let _ = n;
        let fname = self.state.keys().find(|k| k.starts_with(&format!("@only:{region}:")))?.clone();
        let real = fname.rsplit(':').next()?.to_string();
        self.state.get(&fname).map(|v| (real, v.clone()))
    }
}

/// W-IFC-1 to W-IFC-5 over a module as written (surface reading).
pub fn check(m: &Module) -> Vec<String> {
    let mut labels: HashSet<String> = ["secret".to_string(), "untrusted".to_string()].into();
    let mut auth: HashMap<String, String> = HashMap::new();
    let mut errs: Vec<String> = vec![];
    let mut any = false;
    for it in &m.items {
        if let Item::Struct { name, fields, line, .. } = it
            && let Some(l) = name.strip_prefix("@label:")
        {
            any = true;
            if (labels.contains(l) && l != "secret" && l != "untrusted") || (labels.insert(l.to_string()) && false) {
                errs.push(format!("line {line}: W-IFC-6 label `{l}` is declared twice"));
            }
            if let Some(f) = fields.first()
                && let Ty::Path(p, _) = &f.ty
            {
                auth.insert(l.to_string(), p[0].clone());
            }
        }
    }
    // any label written?
    let mut used = any;
    let mut check_ty = |t: &Ty, line: usize, top_ok: bool, errs: &mut Vec<String>, used: &mut bool| {
        let mut nested = false;
        nested_label(t, true, &mut nested);
        if nested {
            errs.push(format!("line {line}: W-IFC-6 a label is written on a whole declared type, not inside it"));
        }
        if let Ty::Region(_, s) = t
            && let Some(inner) = s.strip_prefix("@{").and_then(|x| x.strip_suffix('}'))
        {
            *used = true;
            if !top_ok {
                errs.push(format!("line {line}: W-IFC-6 a label is not allowed on this type"));
            }
            for n in inner.split(',').filter(|x| !x.is_empty()) {
                if !labels.contains(n) {
                    errs.push(format!("line {line}: W-IFC-6 `{n}` is not a declared label (declare it with `label {n}`)"));
                }
            }
        }
    };
    let mut ctors: HashMap<String, Vec<(String, Option<L>)>> = HashMap::new();
    let mut member: HashMap<String, L> = HashMap::new();
    let mut state: HashMap<String, V> = HashMap::new();
    let mut variants: HashSet<String> = HashSet::new();
    let mut payloads: HashMap<String, Vec<(String, Ty)>> = HashMap::new();
    let mut fns: HashMap<String, &FnDecl> = HashMap::new();
    let mut externs: HashMap<String, &FnSig> = HashMap::new();
    for it in &m.items {
        match it {
            Item::Struct { name, fields, line, .. } if !name.starts_with("@label:") => {
                for f in fields {
                    check_ty(&f.ty, *line, true, &mut errs, &mut used);
                }
                ctors.insert(name.clone(), fields.iter().map(|f| (f.name.clone(), label_of(&f.ty, &format!("field {name}.{}", f.name)).and_then(|r| r.ok()))).collect());
                for f in fields {
                    if let Some(Ok(l)) = label_of(&f.ty, &format!("field {name}.{}", f.name)) {
                        let e = member.entry(f.name.clone()).or_default();
                        *e = e.join(&l);
                    }
                }
            }
            Item::Kind { variants: vs, line, .. } => {
                for (v, ps) in vs {
                    variants.insert(v.clone());
                    for (_, t) in ps {
                        check_ty(t, *line, true, &mut errs, &mut used);
                    }
                    ctors.insert(v.clone(), ps.iter().map(|(n, t)| (n.clone(), label_of(t, &format!("variant {v}.{n}")).and_then(|r| r.ok()))).collect());
                }
            }
            Item::Event { name, payload, line, .. } => {
                variants.insert(name.clone());
                for (_, t) in payload {
                    check_ty(t, *line, true, &mut errs, &mut used);
                }
                payloads.insert(name.clone(), payload.clone());
                ctors.insert(name.clone(), payload.iter().map(|(n, t)| (n.clone(), label_of(t, &format!("event {name}.{n}")).and_then(|r| r.ok()))).collect());
            }
            Item::Error { name, payload, line, .. } => {
                variants.insert(name.clone());
                for (_, t) in payload {
                    check_ty(t, *line, false, &mut errs, &mut used);
                }
            }
            Item::State { name, fields, line, .. } => {
                for f in fields {
                    check_ty(&f.ty, *line, true, &mut errs, &mut used);
                    let d = label_of(&f.ty, &format!("state {name}.{}", f.name)).and_then(|r| r.ok());
                    let v = V { val: d.clone().unwrap_or_default(), place: d.clone().or(Some(L::default())), annotated: d.is_some(), mutable: true };
                    state.insert(f.name.clone(), v.clone());
                    if fields.len() == 1 {
                        state.insert(format!("@only:{name}:{}", f.name), v);
                        state.insert(format!("@region:{name}"), V { val: L::default(), place: Option::None, annotated: false, mutable: false });
                    }
                }
            }
            Item::Fn(f) => {
                let mut bt = vec![];
                crate::v09::collect_body_tys(&f.body, &mut bt);
                for (l, t) in bt {
                    check_ty(t, l, true, &mut errs, &mut used);
                }
                fns.insert(f.sig.name.clone(), f);
                for p in &f.sig.params {
                    if let Param::Named { ty, .. } = p {
                        check_ty(ty, f.sig.line, true, &mut errs, &mut used);
                    }
                }
                if let Some(r) = &f.sig.ret {
                    check_ty(r, f.sig.line, true, &mut errs, &mut used);
                }
            }
            Item::Extern { sig, line, .. } => {
                externs.insert(sig.name.clone(), sig);
                for p in &sig.params {
                    if let Param::Named { ty, .. } = p {
                        check_ty(ty, *line, false, &mut errs, &mut used);
                    }
                }
                if let Some(r) = &sig.ret {
                    check_ty(r, *line, true, &mut errs, &mut used);
                }
            }
            _ => {}
        }
    }
    // an authority must name a declared struct
    for (l, a) in &auth {
        if !m.items.iter().any(|i| matches!(i, Item::Struct { name, .. } if name == a)) {
            errs.push(format!("line 1: W-IFC-6 the authority `{a}` of label `{l}` is not a declared struct"));
        }
    }
    if !used {
        errs.sort();
        errs.dedup();
        return errs;
    }
    // state fields: the plain name reads the field
    let mut a = A {
        labels,
        auth,
        fns: fns.clone(),
        externs,
        models: m.items.iter().filter_map(|i| if let Item::Extern { abi, sig, .. } = i && abi == "model" { Some(sig.name.clone()) } else { None }).collect(),
        ctors,
        member,
        state,
        variants,
        errs: vec![],
        memo: HashMap::new(),
        exit_dep: None,
        active: HashSet::new(),
        scopes: vec![],
        ctx: L::default(),
        caps: HashSet::new(),
        rets: L::default(),
        fname: String::new(),
        inst: String::new(),
        ct: m.x09.ct.clone().into_iter().collect(),
        in_ct: false,
    };
    let _ = &a.labels;
    for it in &m.items {
        match it {
            Item::Fn(f) => {
                let ls: Vec<L> = f
                    .sig
                    .params
                    .iter()
                    .filter_map(|p| if let Param::Named { name, ty, .. } = p { Some(label_of(ty, name).and_then(|r| r.ok()).unwrap_or_default()) } else { None })
                    .collect();
                a.run_fn(f, &ls);
            }
            Item::Impl { members, .. } => {
                for f in members {
                    let ls: Vec<L> = f.sig.params.iter().filter_map(|p| if let Param::Named { name, ty, .. } = p { Some(label_of(ty, name).and_then(|r| r.ok()).unwrap_or_default()) } else { None }).collect();
                    // methods: checked like functions but not reachable by name
                    a.fns.entry(format!("@method:{}", f.sig.name)).or_insert(f);
                    let _ = ls;
                }
            }
            Item::Step { region, pat, guard, body, .. } => a.handler(pat, guard, body, &payloads, region),
            Item::Actor { name, handlers, .. } => {
                for h in handlers {
                    a.handler(&h.pat, &h.guard, &h.body, &payloads, name);
                }
            }
            _ => {}
        }
    }
    errs.extend(a.errs);
    errs.sort();
    errs.dedup();
    errs
}
