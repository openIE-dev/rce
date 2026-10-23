//! Probes: extra zero-argument functions that call each function and method
//! of a module on fixed sample arguments, so that the emitter tests observe
//! function bodies and not only the constants, checks and examples a program
//! happens to have. Samples are deterministic, small and built from the
//! parameter types; a function whose parameter types have no sample is not
//! probed.
use crate::ast::*;

fn ex(e: Expr) -> Arg {
    Arg { label: None, e }
}
fn name(n: &str) -> Expr {
    Expr::Name(n.into(), 0)
}

struct S<'a> {
    m: &'a Module,
}

impl S<'_> {
    fn item(&self, n: &str) -> Option<&Item> {
        self.m.items.iter().find(|i| match i {
            Item::Kind { name, .. }
            | Item::Struct { name, .. }
            | Item::Event { name, .. }
            | Item::Error { name, .. }
            | Item::TypeAlias { name, .. } => name == n,
            _ => false,
        })
    }

    /// Sample `k` (0, 1 or 2) of type `t`.
    fn sample(&self, t: &Ty, k: usize, depth: usize) -> Option<Expr> {
        if depth > 4 {
            return None;
        }
        let d = depth + 1;
        Some(match t {
            Ty::Unit => Expr::Unit(0),
            Ty::Opt(x) => {
                if k == 0 {
                    Expr::None(0)
                } else {
                    Expr::Call(Box::new(name("some")), vec![ex(self.sample(x, k, d)?)], 0, false)
                }
            }
            Ty::Tuple(ts) => Expr::Tuple(
                ts.iter()
                    .map(|t| self.sample(t, k, d))
                    .collect::<Option<_>>()?,
                0,
            ),
            Ty::Ref(_, x) | Ty::Region(x, _) | Ty::Union(x, _) => self.sample(x, k, d)?,
            Ty::Slice(x) => {
                self.sample(&Ty::Path(vec!["List".into()], vec![(**x).clone()]), k, d)?
            }
            Ty::Array(x, n) => {
                let Expr::Int(n, _) = **n else { return None };
                if n > 8 {
                    return None;
                }
                Expr::List(
                    (0..n as usize)
                        .map(|i| self.sample(x, (k + i) % 3, d))
                        .collect::<Option<_>>()?,
                    0,
                )
            }
            Ty::Path(p, a) if p.len() == 1 => {
                let n = p[0].as_str();
                if a.is_empty() && crate::types::INTS.contains(&n) {
                    return Some(Expr::Int([0, 1, 3][k], 0));
                }
                match (n, a.len()) {
                    ("bool", 0) => Expr::Bool(k > 0, 0),
                    ("f64" | "f32", 0) => match k {
                        0 => Expr::Float(0.0, 0),
                        1 => Expr::Float(1.5, 0),
                        _ => Expr::Neg(Box::new(Expr::Float(2.25, 0)), 0),
                    },
                    ("text", 0) => Expr::Str(["", "ab", "h\u{e9}llo"][k].into(), 0),
                    ("List", 1) => {
                        if k == 0 {
                            Expr::List(vec![], 0)
                        } else {
                            Expr::List(
                                vec![
                                    self.sample(&a[0], 1, d)?,
                                    self.sample(&a[0], 2, d)?,
                                    self.sample(&a[0], 0, d)?,
                                ],
                                0,
                            )
                        }
                    }
                    ("Map", 2) => {
                        if k == 0 {
                            Expr::Map(vec![], 0)
                        } else {
                            Expr::Map(
                                vec![
                                    (self.sample(&a[0], 1, d)?, self.sample(&a[1], 1, d)?),
                                    (self.sample(&a[0], 2, d)?, self.sample(&a[1], 2, d)?),
                                ],
                                0,
                            )
                        }
                    }
                    _ => match self.item(n)? {
                        Item::TypeAlias { ty, generics, .. } if generics.is_empty() => {
                            self.sample(ty, k, d)?
                        }
                        Item::Kind {
                            variants, generics, ..
                        } if generics.is_empty() && !variants.is_empty() => {
                            // Prefer a variant whose payload has a sample.
                            (0..variants.len())
                                .map(|i| &variants[(k + i) % variants.len()])
                                .find_map(|(v, p)| {
                                    self.ctor(
                                        v,
                                        p.iter().map(|(a, b)| (a.clone(), b.clone())),
                                        k,
                                        d,
                                    )
                                })?
                        }
                        Item::Struct {
                            name: s,
                            fields,
                            generics,
                            ..
                        } if generics.is_empty() => self.ctor(
                            s,
                            fields.iter().map(|f| (f.name.clone(), f.ty.clone())),
                            k,
                            d,
                        )?,
                        Item::Event {
                            name: s, payload, ..
                        }
                        | Item::Error {
                            name: s, payload, ..
                        } => self.ctor(s, payload.iter().cloned(), k, d)?,
                        _ => return None,
                    },
                }
            }
            _ => return None,
        })
    }

    fn ctor(
        &self,
        n: &str,
        fields: impl Iterator<Item = (String, Ty)>,
        k: usize,
        d: usize,
    ) -> Option<Expr> {
        let args: Vec<Arg> = fields
            .map(|(f, t)| {
                Some(Arg {
                    label: Some(f),
                    e: self.sample(&t, k, d)?,
                })
            })
            .collect::<Option<_>>()?;
        Some(if args.is_empty() {
            name(n)
        } else {
            Expr::Call(Box::new(name(n)), args, 0, false)
        })
    }

    /// The body of one probe: bind samples, call, return the result and
    /// whatever the call wrote back.
    fn probe(&self, f: &FnDecl, recv: Option<&Ty>, callee: Expr, k: usize) -> Option<Vec<Stmt>> {
        let mut body = vec![];
        let mut args = vec![];
        let mut outs = vec![];
        if let Some(t) = recv {
            body.push(Stmt::Var("__s".into(), None, self.sample(t, k, 0)?, 0));
        }
        let gs: Vec<String> = f.sig.generics.iter().map(|g| g.name.clone()).collect();
        for (i, p) in f.sig.params.iter().enumerate() {
            let Param::Named { ty, .. } = p else { continue };
            let ty = &subst(ty, &gs);
            let n = format!("__a{i}");
            body.push(Stmt::Var(n.clone(), None, self.sample(ty, k, 0)?, 0));
            if matches!(ty, Ty::Ref(true, _)) {
                args.push(ex(Expr::Borrow(true, Box::new(name(&n)), 0)));
                outs.push(name(&n));
            } else {
                args.push(ex(name(&n)));
            }
        }
        let call = Expr::Call(Box::new(callee), args, 0, false);
        body.push(Stmt::Let(Pat::Path(vec!["__r".into()]), None, call, 0));
        let mut r = name("__r");
        if matches!(f.sig.ret, Some(Ty::Iter(_))) {
            r = Expr::Call(
                Box::new(Expr::Member(Box::new(r), "collect".into(), 0)),
                vec![],
                0,
                false,
            );
        }
        if recv.is_some() && matches!(f.sig.params.first(), Some(Param::SelfRef(true))) {
            outs.push(name("__s"));
        }
        let tail = if outs.is_empty() {
            r
        } else {
            outs.insert(0, r);
            Expr::Tuple(outs, 0)
        };
        body.push(Stmt::Expr(tail));
        Some(body)
    }
}

/// `t` with each generic parameter in `gs` replaced by i64.
fn subst(t: &Ty, gs: &[String]) -> Ty {
    let s = |x: &Ty| Box::new(subst(x, gs));
    match t {
        Ty::Path(p, a) if p.len() == 1 && a.is_empty() && gs.contains(&p[0]) => {
            Ty::Path(vec!["i64".into()], vec![])
        }
        Ty::Path(p, a) => Ty::Path(p.clone(), a.iter().map(|x| subst(x, gs)).collect()),
        Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(|x| subst(x, gs)).collect()),
        Ty::Slice(x) => Ty::Slice(s(x)),
        Ty::Array(x, n) => Ty::Array(s(x), n.clone()),
        Ty::Ref(m, x) => Ty::Ref(*m, s(x)),
        Ty::Opt(x) => Ty::Opt(s(x)),
        Ty::Union(a, b) => Ty::Union(s(a), s(b)),
        Ty::Region(x, r) => Ty::Region(s(x), r.clone()),
        t => t.clone(),
    }
}

/// `m` with probe functions appended: for each non-generic function with
/// parameters and each method, up to three calls on sample arguments.
pub fn with_probes(m: &Module) -> Module {
    let s = S { m };
    let mut extra = vec![];
    let mut add = |label: String, f: &FnDecl, recv: Option<&Ty>, callee: Expr| {
        for k in 0..3 {
            if let Some(b) = s.probe(f, recv, callee.clone(), k) {
                extra.push(Item::Fn(FnDecl {
                    sig: FnSig {
                        comptime: false,
                        name: format!("__probe_{label}_{k}"),
                        generics: vec![],
                        params: vec![],
                        ret: Some(Ty::Path(vec!["__any".into()], vec![])),
                        bound: None,
                        line: 0,
                    },
                    body: Body::Block(b),
                }));
            }
        }
    };
    for i in &m.items {
        match i {
            Item::Fn(f)
                if f.sig.generics.iter().all(|g| !g.comptime)
                    && !f.sig.params.is_empty()
                    && f.sig.name != "main" =>
            {
                add(f.sig.name.clone(), f, None, name(&f.sig.name));
            }
            Item::Impl {
                generics,
                ty,
                members,
                ..
            } if generics.is_empty() => {
                let h = crate::check::ty_head(ty);
                for f in members.iter().filter(|f| f.sig.generics.is_empty()) {
                    let label = format!("{}_{}", h, f.sig.name);
                    if matches!(
                        f.sig.params.first(),
                        Some(Param::SelfVal | Param::SelfRef(_))
                    ) {
                        add(
                            label,
                            f,
                            Some(ty),
                            Expr::Member(Box::new(name("__s")), f.sig.name.clone(), 0),
                        );
                    } else {
                        add(
                            label,
                            f,
                            None,
                            Expr::Member(Box::new(name(&h)), f.sig.name.clone(), 0),
                        );
                    }
                }
            }
            _ => {}
        }
    }
    // One example that sends every event three times over, with samples 1, 2, 0.
    let mut events = vec![];
    for k in [1, 2, 0] {
        for i in &m.items {
            if let Item::Event {
                name: n, payload, ..
            } = i
                && let Some(args) = payload
                    .iter()
                    .map(|(f, t)| {
                        Some(Arg {
                            label: Some(f.clone()),
                            e: s.sample(t, k, 0)?,
                        })
                    })
                    .collect::<Option<Vec<_>>>()
            {
                events.push((n.clone(), args));
            }
        }
    }
    if !events.is_empty() && m.items.iter().any(|i| matches!(i, Item::State { .. })) {
        extra.push(Item::Example(Example {
            events,
            refused: None,
            value: None,
            line: 0,
        }));
    }
    let mut out = m.clone();
    out.items.extend(extra);
    out
}
