//! Expressions: each becomes target code that evaluates its operands in the
//! reference evaluator's order and calls the runtime helper for the operator.
use super::Lang;
use super::base::*;
use crate::ast::*;
use std::collections::HashSet;

/// A call that may write back its receiver or `&mut` arguments.
pub(super) struct WriteBack {
    pub tmp: String,
    pub recv: Option<Expr>,
    pub args: Vec<Arg>,
}

impl<'a> G<'a> {
    fn args(&mut self, args: &[Arg]) -> Result<Vec<String>, String> {
        args.iter().map(|a| self.ex(&a.e)).collect()
    }
    fn opt_ex(&mut self, e: &Option<Box<Expr>>) -> Result<Option<String>, String> {
        Ok(match e {
            Some(e) => Some(self.ex(e)?),
            None => None,
        })
    }

    /// `e` with the reference evaluator's fuel tick on entry (interp `eval`
    /// ticks before it looks at the expression, 6.7). Hoisted temporaries
    /// and the A-normal-form names of `anf.rs` are not source expressions:
    /// their source expression ticked where it was evaluated.
    pub fn ex(&mut self, e: &Expr) -> E {
        let v = self.ex_raw(e)?;
        if self.no_fuel > 0 || matches!(e, Expr::Name(n, _) if n.starts_with('\u{1}') || n.starts_with("anf_") || n.starts_with("$fpchunk")) {
            return Ok(v);
        }
        Ok(self.pre_fuel(&v))
    }
    /// `v` evaluated after one fuel tick.
    pub fn pre_fuel(&self, v: &str) -> String {
        match self.lang {
            Lang::Rust => format!("{{ fuel()?; {v} }}"),
            Lang::Python => format!("(fuel() or ({v}))"),
            Lang::Js => format!("(fuel(), ({v}))"),
            Lang::Scala => format!("{{ fuel(); {v} }}"),
        }
    }
    /// `e` without a tick at any node (see `no_fuel`).
    pub fn ex_nofuel(&mut self, e: &Expr) -> E {
        self.no_fuel += 1;
        let r = self.ex(e);
        self.no_fuel -= 1;
        r
    }

    fn ex_raw(&mut self, e: &Expr) -> E {
        let l = self.lang;
        Ok(match e {
            // a hoisted temporary (ex_stmt), not a source name
            Expr::Name(n, _) if n.starts_with('\u{1}') => {
                let t = &n[1..];
                if l == Lang::Rust { format!("{t}.clone()") } else { t.to_string() }
            }
            Expr::Name(n, _) => {
                if self.local(n) {
                    self.var(n)
                } else if self.fns.contains_key(n) {
                    match l {
                        Lang::Rust => format!("V::Fn({}.to_string())", q(n)),
                        _ => self.new_("FnV", &q(n)),
                    }
                } else if self.variants.get(n).is_some_and(|p| p.1.is_empty())
                    || matches!(
                        self.types.get(n),
                        Some(Item::Event { .. } | Item::Error { .. })
                    )
                {
                    self.rec0(n)
                } else if self.consts.contains(n) {
                    match l {
                        Lang::Rust => format!("k_{}()?", ident(n)),
                        _ => format!("k_{}()", ident(n)),
                    }
                } else {
                    self.trap_e(&format!("unknown name {n}"))
                }
            }
            Expr::SelfRef(_) => {
                if self.local("self") {
                    self.var("self")
                } else {
                    self.trap_e("self")
                }
            }
            Expr::Int(v, _) => match l {
                Lang::Js => format!("lit({v}n)"),
                Lang::Scala if *v >= i64::MIN as i128 && *v <= i64::MAX as i128 => format!("lit({v}L)"),
                Lang::Scala => format!("lit(BigInt(\"{v}\"))"),
                _ => format!("lit({v})"),
            },
            Expr::Float(f, _) => match l {
                Lang::Rust => format!("V::F({f:?}_f64)"),
                _ => self.new_("F", &format!("{f:?}")),
            },
            Expr::Str(s, _) => self.text(s),
            Expr::Bool(b, _) => match l {
                Lang::Rust => format!("V::Bool({b})"),
                Lang::Python => (if *b { "True" } else { "False" }).into(),
                Lang::Js | Lang::Scala => b.to_string(),
            },
            Expr::None(_) => match l {
                Lang::Rust => "V::Opt(None)".into(),
                _ => "NONE".into(),
            },
            Expr::Unit(_) => self.unit().into(),
            Expr::Tuple(xs, _) | Expr::List(xs, _) => {
                let v = xs
                    .iter()
                    .map(|x| self.ex(x))
                    .collect::<Result<Vec<_>, _>>()?;
                let tup = matches!(e, Expr::Tuple(..));
                let lv = match l {
                    Lang::Rust => format!(
                        "V::{}(vec![{}])",
                        if tup { "Tuple" } else { "List" },
                        v.join(", ")
                    ),
                    Lang::Scala => format!(
                        "{}(V({}))",
                        if tup { "Tup" } else { "Lst" },
                        v.join(", ")
                    ),
                    _ => self.new_(
                        if tup { "Tup" } else { "Lst" },
                        &format!("[{}]", v.join(", ")),
                    ),
                };
                if tup { lv } else { self.pre_charge(4 + xs.len(), &lv) }
            }
            Expr::ListRepeat(x, n, _) => {
                let (a, b) = (self.ex(x)?, self.ex(n)?);
                self.h("list_repeat", &[a, b])
            }
            Expr::ListComp(body, p, it, cond, _) => self.listcomp(body, p, it, cond)?,
            Expr::Map(ps, _) => {
                let mut v = vec![];
                for (a, b) in ps {
                    let (x, y) = (self.ex(a)?, self.ex(b)?);
                    v.push(self.pair(&x, &y));
                }
                let mm = self.h("mk_map", &[self.vec(&v)]);
                self.pre_charge(4 + ps.len(), &mm)
            }
            Expr::ChanNew(t, cap, line) if self.tasks_on() => {
                let session = matches!(t, Ty::Path(p, _) if p.len() == 1 && p[0] == "session");
                let e = self.chan_new_e(cap.as_deref(), *line)?;
                if session { e.replacen("chan_new(", "session_new(", 1).replacen("chanNew(", "sessionNew(", 1) } else { e }
            }
            Expr::Recv(c, _) if self.tasks_on() => self.recv_e(c)?,
            Expr::ChanNew(..) | Expr::Recv(..) => {
                self.trap_e("channels are not available at comptime")
            }
            Expr::Call(f, args, _, sz) => match self.call_expr(f, args, false, *sz)? {
                Some((v, _)) => v,
                None => {
                    return unsupported(
                        "a call that writes back its receiver or arguments in a conditional position (right of and/or, a branch, a handler, a lambda or a comprehension; strict positions are flattened by the ANF pass)",
                    );
                }
            },
            Expr::Index(x, idx, _) => {
                let a = self.ex(x)?;
                let b = self.ex(&idx[0])?;
                self.h("cindex", &[a, b])
            }
            Expr::Slice(x, a, b, _) => {
                let c = self.ex(x)?;
                let av = self.opt_ex(a)?;
                let bv = self.opt_ex(b)?;
                match l {
                    Lang::Rust => {
                        let o = |x: Option<String>| {
                            x.map(|x| format!("Some({x})")).unwrap_or("None".into())
                        };
                        format!(
                            "{{ let __c = {c}; charge(1)?; slice_check(&__c)?; let __a = {}; let __b = {}; slice(__c, __a, __b)? }}",
                            o(av),
                            o(bv)
                        )
                    }
                    _ => {
                        let th = |x: Option<String>| match x {
                            Some(x) if l == Lang::Python => format!("(lambda: {x})"),
                            Some(x) => format!("(() => {x})"),
                            None => self.none().to_string(),
                        };
                        format!("{}({c}, {}, {})", self.rt("slice"), th(av), th(bv))
                    }
                }
            }
            Expr::Member(x, m, _) => {
                let a = self.ex(x)?;
                self.h("cfield", &[a, q(m)])
            }
            Expr::TupleMember(x, i, _) => {
                let a = self.ex(x)?;
                self.h("ctmember", &[a, i.to_string()])
            }
            Expr::Neg(x, _) => {
                let a = self.ex(x)?;
                self.h("cneg", &[a])
            }
            Expr::BitNot(x, _) => {
                let a = self.ex(x)?;
                self.h("cbitnot", &[a])
            }
            Expr::Not(x, _) => {
                let a = self.ex(x)?;
                self.h("cnot", &[a])
            }
            Expr::Borrow(_, x, _) | Expr::Comptime(x, _) => self.ex(x)?,
            Expr::Cast(x, t, _) => {
                let a = self.ex(x)?;
                let ty = self.cast_ty(t);
                self.h("cast", &[a, ty])
            }
            Expr::Bin(op, a, b, _) if op == "and" || op == "or" => {
                let (x, y) = (self.ex(a)?, self.ex(b)?);
                let (tx, ty) = (self.truthy(&x), self.truthy(&y));
                // interp charges 1 before the left operand (C.6.4)
                let tx = match l {
                    Lang::Rust => format!("{{ charge(1)?; {tx} }}"),
                    Lang::Python => format!("(charge(1), {tx})[1]"),
                    Lang::Js => format!("(charge(1n), {tx})"),
                    Lang::Scala => format!("{{ charge(1); {tx} }}"),
                };
                let o = if l == Lang::Python {
                    op.as_str()
                } else if op == "and" {
                    "&&"
                } else {
                    "||"
                };
                match l {
                    Lang::Rust => format!("V::Bool({tx} {o} {ty})"),
                    _ => format!("({tx} {o} {ty})"),
                }
            }
            Expr::Bin(op, a, b, _) => {
                let (x, y) = (self.ex(a)?, self.ex(b)?);
                self.h("cbinop", &[q(op), x, y])
            }
            Expr::Catch(x, bind, h, _) => {
                let xv = self.ex(x)?;
                self.scopes.push(bind.iter().cloned().collect());
                let hv = self.ex(h)?;
                self.scopes.pop();
                let p = match bind {
                    Some(n) => format!("l_{}", ident(n)),
                    None => "_e".into(),
                };
                match l {
                    Lang::Rust => format!(
                        "(match {xv} {{ V::Err(__e) => {{ #[allow(unused)] let {p} = *__e; {hv} }} __c => __c }})"
                    ),
                    Lang::Python => format!("catch_({xv}, lambda {p}: {hv})"),
                    Lang::Js => format!("catch_({xv}, ({p}) => {hv})"),
                    Lang::Scala => format!("catch_({xv}, ({p} : Any) => {hv})"),
                }
            }
            Expr::Try(x, els, _) => {
                let xv = self.ex(x)?;
                let ev = self.opt_ex(els)?;
                match (l, ev) {
                    (Lang::Rust, Some(e)) => format!(
                        "try_({xv}, Some(&(|| -> R<V> {{ Ok({e}) }}) as &dyn Fn() -> R<V>))?"
                    ),
                    (Lang::Rust, None) => format!("try_({xv}, None)?"),
                    (Lang::Python, Some(e)) => format!("try_({xv}, lambda: {e})"),
                    (Lang::Js | Lang::Scala, Some(e)) => format!("try_({xv}, () => {e})"),
                    (_, None) => format!("try_({xv}, {})", self.none()),
                }
            }
            Expr::If(c, a, b, _) => {
                let (cv, av, bv) = (self.ex(c)?, self.ex(a)?, self.ex(b)?);
                let t = self.truthy(&cv);
                match l {
                    Lang::Rust => format!("(if {t} {{ {av} }} else {{ {bv} }})"),
                    Lang::Python => format!("({av} if {t} else {bv})"),
                    Lang::Js => format!("({t} ? {av} : {bv})"),
                    Lang::Scala => format!("(if ({t}) {av} else {bv})"),
                }
            }
            Expr::Lambda(ps, body, _) => self.lambda(ps, body)?,
        })
    }

    /// The type for `cast`: only an integer type or f32 matter there.
    fn cast_ty(&self, t: &Ty) -> String {
        let mut t = t.clone();
        for _ in 0..32 {
            if let Ty::Path(p, _) = &t
                && p.len() == 1
                && let Some(Item::TypeAlias { ty, .. }) = self.types.get(&p[0])
            {
                t = ty.clone();
                continue;
            }
            break;
        }
        match &t {
            Ty::Path(p, a)
                if p.len() == 1
                    && a.is_empty()
                    && (p[0] == "f32" || crate::types::INTS.contains(&p[0].as_str())) =>
            {
                self.ty(&t)
            }
            _ => self.ty(&Ty::Unit),
        }
    }

    fn listcomp(&mut self, body: &Expr, p: &Pat, it: &Expr, cond: &Option<Box<Expr>>) -> E {
        let l = self.lang;
        let itv = self.ex(it)?;
        let ps = self.pat(p)?;
        let mut names = vec![];
        self.pat_names(p, &mut names);
        self.scopes.push(names.clone());
        let b = self.ex(body)?;
        let c = self.opt_ex(cond)?;
        self.scopes.pop();
        Ok(match l {
            Lang::Rust => {
                let binds: String = names
                    .iter()
                    .map(|n| format!("let l_{} = mget(__m, {}); ", ident(n), q(n)))
                    .collect();
                let bf = format!("&|__m: &M| -> R<V> {{ {binds}Ok({b}) }}");
                let cf = match c {
                    Some(c) => format!(
                        "Some(&(|__m: &M| -> R<V> {{ {binds}Ok({c}) }}) as &dyn Fn(&M) -> R<V>)"
                    ),
                    None => "None".into(),
                };
                format!("listcomp({itv}, {ps}, {bf}, {cf})?")
            }
            _ => {
                let params: Vec<String> = names.iter().map(|n| format!("l_{}", ident(n))).collect();
                let ns: Vec<String> = names.iter().map(|n| q(n)).collect();
                let lam = |x: &str| match l {
                    Lang::Python => format!("(lambda {}: {x})", params.join(", ")),
                    Lang::Scala => {
                        let binds: String = params.iter().enumerate().map(|(i, p)| format!("val {p} : Any = __b({i}); ")).collect();
                        format!("((__b: Vector[Any]) => {{ {binds}{x} }})")
                    }
                    _ => format!("(({}) => {x})", params.join(", ")),
                };
                let cf = c.map(|c| lam(&c)).unwrap_or(self.none().into());
                let (lo, hi) = if l == Lang::Scala { ("V(", ")") } else { ("[", "]") };
                format!(
                    "listcomp({itv}, {ps}, {lo}{}{hi}, {}, {cf})",
                    ns.join(", "),
                    lam(&b)
                )
            }
        })
    }

    fn lambda(&mut self, ps: &[Param], body: &Expr) -> E {
        let l = self.lang;
        let mut names = vec![];
        let mut tys = vec![];
        for p in ps {
            if let Param::Named { name, ty, .. } = p {
                names.push(name.clone());
                tys.push(self.ty(ty));
            }
        }
        let mut free = HashSet::new();
        expr_names(body, &mut free);
        for n in &names {
            free.remove(n);
        }
        let caps: Vec<String> = self
            .all_locals()
            .into_iter()
            .filter(|n| free.contains(n))
            .collect();
        self.scopes.push(names.clone());
        let b = self.ex(body)?;
        self.scopes.pop();
        let n = names.len();
        let params: Vec<String> = names.iter().map(|x| format!("l_{}", ident(x))).collect();
        Ok(match l {
            Lang::Rust => {
                let cap: String = caps
                    .iter()
                    .map(|c| format!("let l_{0} = l_{0}.clone(); ", ident(c)))
                    .collect();
                let binds: String = params
                    .iter()
                    .enumerate()
                    .map(|(i, x)| format!("let {x} = arg(&__a, {i}); "))
                    .collect();
                format!(
                    "{{ {cap}V::Clo(Rc::new(move |__a: Vec<V>| -> R<V> {{ charge(4)?; let __a = fitall(__a, &[{}])?; {binds}Ok({b}) }})) }}",
                    tys.join(", ")
                )
            }
            Lang::Python => {
                let mut head = vec!["__a".to_string()];
                head.extend(caps.iter().map(|c| format!("l_{0}=l_{0}", ident(c))));
                format!(
                    "Clo(lambda {}: (charge(4), (lambda {}: {b})(*(fitall(__a, [{}]) + [UNIT] * {n})[:{n}]))[1])",
                    head.join(", "),
                    params.join(", "),
                    tys.join(", ")
                )
            }
            Lang::Js => {
                let cap: Vec<String> = caps.iter().map(|c| format!("l_{}", ident(c))).collect();
                format!(
                    "(({}) => new Clo((__a) => {{ charge(4n); const [{}] = [...fitall(__a, [{}]), ...Array({n}).fill(UNIT)].slice(0, {n}); return {b}; }}))({})",
                    cap.join(", "),
                    params.join(", "),
                    tys.join(", "),
                    cap.join(", ")
                )
            }
            Lang::Scala => {
                let capd: Vec<String> = caps.iter().map(|c| format!("l_{} : Any", ident(c))).collect();
                let cap: Vec<String> = caps.iter().map(|c| format!("l_{}", ident(c))).collect();
                let binds: String = params.iter().enumerate().map(|(i, x)| format!("val {x} : Any = argat(__f, {i}); ")).collect();
                format!(
                    "(({}) => new Clo((__a: Vector[Any]) => {{ charge(4); val __f = fitall(__a, V({})); {binds}{b} }}))({})",
                    capd.join(", "),
                    tys.join(", "),
                    cap.join(", ")
                )
            }
        })
    }

    /// Fields of a constructible name: (field, type, default) in order.
    fn ctor_fields(&self, n: &str) -> Option<Vec<(String, Ty, Option<Expr>)>> {
        if let Some((_, p)) = self.variants.get(n) {
            return Some(
                p.iter()
                    .map(|(a, b)| (a.clone(), b.clone(), None))
                    .collect(),
            );
        }
        match self.types.get(n) {
            Some(Item::Struct { fields, .. }) => Some(
                fields
                    .iter()
                    .map(|f| (f.name.clone(), f.ty.clone(), f.init.clone()))
                    .collect(),
            ),
            Some(Item::Event { payload, .. } | Item::Error { payload, .. }) => Some(
                payload
                    .iter()
                    .map(|(a, b)| (a.clone(), b.clone(), None))
                    .collect(),
            ),
            _ => None,
        }
    }

    pub fn construct_code(&mut self, n: &str, args: &[Arg]) -> Result<Option<String>, String> {
        let Some(fields) = self.ctor_fields(n) else {
            return Ok(None);
        };
        let l = self.lang;
        let mut fs = vec![];
        for (f, t, init) in &fields {
            let ty = self.ty(t);
            let iv = match init {
                Some(e) => {
                    let saved = std::mem::take(&mut self.scopes);
                    let code = self.ex(e);
                    self.scopes = saved;
                    let code = code?;
                    match l {
                        Lang::Rust => format!(
                            "Some(Rc::new(|| -> R<V> {{ Ok({code}) }}) as Rc<dyn Fn() -> R<V> + Send + Sync>)"
                        ),
                        Lang::Python => format!("(lambda: {code})"),
                        Lang::Js | Lang::Scala => format!("(() => {code})"),
                    }
                }
                None => self.none().into(),
            };
            fs.push(match l {
                Lang::Js => format!("[{}, {ty}, {iv}]", q(f)),
                Lang::Scala => format!("V({}, {ty}, {iv})", q(f)),
                _ => format!("({}, {ty}, {iv})", q(f)),
            });
        }
        let mut avs = vec![];
        for a in args {
            let v = self.ex(&a.e)?;
            avs.push(self.pair(&self.opt_label(&a.label), &v));
        }
        Ok(Some(
            self.h("construct", &[q(n), self.vec(&fs), self.vec(&avs)]),
        ))
    }

    /// interp `call_fn` charges C_CALL before evaluating the arguments
    fn user_call(&self, id: &str, a: &[String]) -> String {
        let c = match self.lang {
            Lang::Rust => format!("{id}(None, vec![{}])?", a.join(", ")),
            Lang::Scala => format!("{id}(null, V({}))", a.join(", ")),
            _ => format!("{id}({}, [{}])", self.none(), a.join(", ")),
        };
        // interp `call_fn` ticks, then charges, then evaluates arguments
        let c = self.pre_charge(4, &c);
        self.pre_fuel(&c)
    }
    /// `v`, evaluated after charging `n` cost units
    pub fn pre_charge(&self, n: usize, v: &str) -> String {
        match self.lang {
            Lang::Rust => format!("{{ charge({n})?; {v} }}"),
            Lang::Python => format!("(charge({n}), {v})[1]"),
            Lang::Js => format!("(charge({n}n), {v})"),
            Lang::Scala => format!("{{ charge({n}); {v} }}"),
        }
    }
    fn val_of(&self, c: &str) -> String {
        match self.lang {
            Lang::Rust => format!("{c}.v"),
            Lang::Scala => format!("at({c}, 0)"),

            _ => format!("{c}[0]"),
        }
    }

    /// A call: (value expression, write-back). The write-back is set when
    /// the call may write back its receiver or `&mut` arguments; when `stmt`
    /// is false such a call is refused (Ok(None)).
    pub fn call_expr(
        &mut self,
        f: &Expr,
        args: &[Arg],
        stmt: bool,
        sized: bool,
    ) -> Result<Option<(String, Option<WriteBack>)>, String> {
        let l = self.lang;
        // 5.16 context text
        if let Expr::Member(o, m, _) = f
            && let Expr::Name(c, _) = &**o
            && crate::index08::CTX.contains(&c.as_str())
            && !self.local(c)
            && !self.types.contains_key(c)
            && args.len() == 1
        {
            let a = self.ex(&args[0].e)?;
            return Ok(Some((if m == "lit" { self.h("ctx_lit", &[a]) } else { self.h("ctx_encode", &[q(c), a]) }, None)));
        }
        match f {
            Expr::Name(n, _) if !self.local(n) => {
                match n.as_str() {
                    "some" if !args.is_empty() => {
                        let a = self.ex(&args[0].e)?;
                        let v = if l == Lang::Python {
                            format!("Opt({a})")
                        } else {
                            format!("some({a})")
                        };
                        return Ok(Some((v, None)));
                    }
                    "copy" | "Box" if !args.is_empty() => {
                        return Ok(Some((self.ex(&args[0].e)?, None)));
                    }
                    _ => {}
                }
                if let Some(fd) = self.fns.get(n).copied() {
                    let a = self.args(args)?;
                    let call = self.user_call(&format!("u_{}", ident(n)), &a);
                    return self.finish_user(call, has_mut_params(fd), args, stmt);
                }
                if let Some(c) = self.construct_code(n, args)? {
                    return Ok(Some((c, None)));
                }
                if self.tasks_on() && self.m.items.iter().find(|i| matches!(i, Item::Actor { name, .. } if name == n)).is_some() && !self.local(n) {
                    return Ok(Some((self.h("actor_new", &[q(n), f.line().to_string()]), None)));
                }
                if n == "lvar_new" {
                    let a = self.args(args)?;
                    let v = match l {
                        Lang::Rust => format!("lvar_new(vec![{}])", a.join(", ")),
                        Lang::Scala => format!("lvar_new(V({}))", a.join(", ")),

                        _ => format!("{}([{}])", self.rt("lvar_new"), a.join(", ")),
                    };
                    return Ok(Some((v, None)));
                }
                Ok(Some((
                    self.trap_e(&format!("cannot call {n} at comptime")),
                    None,
                )))
            }
            Expr::Member(obj, m, _) => {
                if let Expr::Name(tn, _) = &**obj
                    && !self.local(tn)
                    && self.types.contains_key(tn)
                {
                    return match self.methods.get(&(tn.clone(), m.clone())).cloned() {
                        Some((id, fd)) => {
                            let a = self.args(args)?;
                            let call = self.user_call(&format!("u_{id}"), &a);
                            self.finish_user(call, has_mut_params(fd), args, stmt)
                        }
                        None => Ok(Some((self.trap_e("no such function"), None))),
                    };
                }
                let recv = self.ex(obj)?;
                let recv = self.h("pre_method", &[recv, q(m)]);
                if self.tasks_on()
                    && m == "choose"
                    && let [Arg { e: Expr::Name(lb, _), .. }] = args
                    && !self.local(lb)
                {
                    return Ok(Some((self.h("session_choose", &[recv, q(lb)]), None)));
                }
                let a = self.args(args)?;
                // C.15: the version of the module containing this call governs its cost
                let sized_lit = match l {
                    Lang::Python => if sized { "True" } else { "False" },
                    _ => if sized { "true" } else { "false" },
                }
                .to_string();
                let call = self.h("call_method", &[recv, q(m), self.vec(&a), sized_lit]);
                if MUTATING.contains(&m.as_str()) || self.mut_method_names.contains(m) {
                    if !stmt {
                        return Ok(None);
                    }
                    let t = self.fresh("c");
                    return Ok(Some((
                        call,
                        Some(WriteBack {
                            tmp: t,
                            recv: Some((**obj).clone()),
                            args: args.to_vec(),
                        }),
                    )));
                }
                Ok(Some((self.val_of(&call), None)))
            }
            _ => {
                let fv = self.ex(f)?;
                let fv = self.h("pre_value", &[fv]);
                let a = self.args(args)?;
                Ok(Some((self.h("call_value", &[fv, self.vec(&a)]), None)))
            }
        }
    }

    /// Whether `x` is a call that writes back (as `call_expr` decides).
    pub fn is_wb_call(&self, x: &Expr) -> bool {
        let Expr::Call(f, _, _, _) = x else { return false };
        match &**f {
            Expr::Name(n, _) if !self.local(n) => {
                self.fns.get(n).is_some_and(|fd| has_mut_params(fd))
            }
            Expr::Member(obj, m, _) => {
                if let Expr::Name(tn, _) = &**obj
                    && !self.local(tn)
                    && self.types.contains_key(tn)
                {
                    return self.methods.get(&(tn.clone(), m.clone())).is_some_and(|(_, fd)| has_mut_params(fd));
                }
                MUTATING.contains(&m.as_str()) || self.mut_method_names.contains(m)
            }
            _ => false,
        }
    }

    fn finish_user(
        &mut self,
        call: String,
        muts: bool,
        args: &[Arg],
        stmt: bool,
    ) -> Result<Option<(String, Option<WriteBack>)>, String> {
        if !muts {
            return Ok(Some((self.val_of(&call), None)));
        }
        if !stmt {
            return Ok(None);
        }
        let t = self.fresh("c");
        Ok(Some((
            call,
            Some(WriteBack {
                tmp: t,
                recv: None,
                args: args.to_vec(),
            }),
        )))
    }
}
