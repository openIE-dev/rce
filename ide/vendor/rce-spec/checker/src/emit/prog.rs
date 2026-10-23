//! Whole programs: functions, constants, steps, the initial state, the
//! dispatch tables the runtime consults, and the observation driver.
use super::base::*;
use super::{Lang, zero_arg};
use crate::ast::*;

impl<'a> G<'a> {
    fn get_arg(&self, i: usize) -> String {
        match self.lang {
            Lang::Rust => format!("arg(&a, {i})"),
            Lang::Python => format!("(a[{i}] if len(a) > {i} else UNIT)"),
            Lang::Js => format!("(a.length > {i} ? a[{i}] : UNIT)"),
            Lang::Scala => format!("atOr(a, {i}, UNIT)"),
        }
    }

    /// The body (`b_`), call (`u_`) and raw (`r_`) forms of one function or method.
    fn function(&mut self, id: &str, f: &FnDecl) -> Result<(), String> {
        let l = self.lang;
        let named = named_params(f);
        let muts: Vec<Option<String>> = named
            .iter()
            .map(|(n, t)| {
                if matches!(t, Ty::Ref(true, _)) {
                    Some(n.clone())
                } else {
                    None
                }
            })
            .collect();
        let self_mut = matches!(f.sig.params.first(), Some(Param::SelfRef(true)));
        let has_self = matches!(
            f.sig.params.first(),
            Some(Param::SelfVal | Param::SelfRef(_))
        );
        let tys: Vec<String> = named.iter().map(|(_, t)| self.ty(t)).collect();
        let tl = tys.join(", ");
        if matches!(f.sig.ret, Some(Ty::Iter(_))) {
            match l {
                Lang::Rust => self.open(&format!("fn g_{id}(a: Vec<V>) -> R<()>")),
                Lang::Python => self.open(&format!("def g_{id}(a)")),
                Lang::Js => self.open(&format!("function g_{id}(a)")),
                Lang::Scala => self.open(&format!("def g_{id}(a: Vector[Any]): Unit =")),
            }
            self.scopes = vec![vec![]];
            for (i, (n, _)) in named.iter().enumerate() {
                let s = self.let_(&format!("l_{}", ident(n)), &self.get_arg(i));
                self.line(s);
                self.declare(n);
            }
            self.ctx = Ctx::Gen;
            match &f.body {
                Body::Block(b) => self.block(b, false)?,
                Body::Arrow(_) => {
                    let s = self.trap_s("iterator without a block body");
                    self.line(s);
                }
            }
            if l == Lang::Rust {
                self.line("Ok(())");
            }
            self.close();
            match l {
                Lang::Rust => {
                    self.open(&format!("fn u_{id}(slf: Option<V>, a: Vec<V>) -> R<Call>"))
                }
                Lang::Python => self.open(&format!("def u_{id}(slf, a)")),
                Lang::Js => self.open(&format!("function u_{id}(slf, a)")),
                Lang::Scala => self.open(&format!("def u_{id}(slf: Any, a: Vector[Any]): Vector[Any] =")),
            }
            match l {
                Lang::Rust => {
                    self.line(format!("let fr = fitall(a, &[{tl}])?;"));
                    self.line(format!(
                        "Ok(Call {{ v: V::Gen(Rc::new(move |cap: usize| {{ let fr = fr.clone(); gen_run(&move || g_{id}(fr.clone()), cap) }})), outs: vec![], slf: None }})"
                    ));
                }
                Lang::Python => {
                    self.line(format!("fr = fitall(a, [{tl}])"));
                    self.line(format!(
                        "return (Gen(lambda cap: gen_run(lambda: g_{id}(fr), cap)), [], None)"
                    ));
                }
                Lang::Js => {
                    self.line(format!("const fr = fitall(a, [{tl}]);"));
                    self.line(format!(
                        "return [new Gen((cap) => genRun(() => g_{id}(fr), cap)), [], null];"
                    ));
                }
                Lang::Scala => {
                    self.line(format!("val fr = fitall(a, V({tl}))"));
                    self.line(format!(
                        "V(new Gen((cap: Int) => gen_run(() => g_{id}(fr), cap)), V(), null)"
                    ));
                }
            }
            self.close();
            match l {
                Lang::Rust => {
                    self.open(&format!("fn r_{id}(a: Vec<V>) -> R<V>"));
                    self.line(format!(
                        "let a = fitall(a, &[{tl}])?; g_{id}(a)?; Ok(V::Unit)"
                    ));
                }
                Lang::Python => {
                    self.open(&format!("def r_{id}(a)"));
                    self.line(format!("g_{id}(fitall(a, [{tl}]))\nreturn UNIT"));
                }
                Lang::Js => {
                    self.open(&format!("function r_{id}(a)"));
                    self.line(format!("g_{id}(fitall(a, [{tl}])); return UNIT;"));
                }
                Lang::Scala => {
                    self.open(&format!("def r_{id}(a: Vector[Any]): Any ="));
                    self.line(format!("g_{id}(fitall(a, V({tl}))); UNIT"));
                }
            }
            self.close();
            return Ok(());
        }
        // b_: the frame of interp `call_fn`, parameters fitted. With `bound
        // n` (6.7) the body is c_ and b_ wraps it: c_ counts the call after
        // fitting the parameters, b_ uncounts it on every outcome but a trap
        // (a trap ends the observation, and each observation starts with no
        // active calls).
        let bn = if f.sig.bound.is_some() { "c" } else { "b" };
        match l {
            Lang::Rust => self.open(&format!("fn {bn}_{id}(slf: Option<V>, a: Vec<V>) -> R<Call>")),
            Lang::Python => self.open(&format!("def {bn}_{id}(slf, a)")),
            Lang::Js => self.open(&format!("function {bn}_{id}(slf, a)")),
            Lang::Scala => self.open(&format!("def {bn}_{id}(slf: Any, a: Vector[Any]): Vector[Any] =")),
        }
        self.scopes = vec![vec![]];
        for (i, (n, _)) in named.iter().enumerate() {
            let s = self.let_(
                &format!("l_{}", ident(n)),
                &self.fit(&self.get_arg(i), &tys[i]),
            );
            self.line(s);
            self.declare(n);
        }
        if has_self {
            let s = match l {
                Lang::Rust => "let mut l_self = slf.clone().unwrap_or(V::Unit);",
                Lang::Python => "l_self = slf if slf is not None else UNIT",
                Lang::Js => "let l_self = slf !== null ? slf : UNIT;",
                Lang::Scala => "var l_self: Any = if (slf != null) slf else UNIT",
            };
            self.line(s);
            self.declare("self");
        }
        if let Some(n) = f.sig.bound {
            match l {
                Lang::Rust => self.line(format!("depth_enter(\"{id}\", {n})?;")),
                Lang::Python => self.line(format!("depth_enter(\"{id}\", {n})")),
                Lang::Js => self.line(format!("depthEnter(\"{id}\", {n}n);")),
                Lang::Scala => self.line(format!("depth_enter(\"{id}\", {n}L)")),
            }
        }
        let union = matches!(f.sig.ret, Some(Ty::Union(..)));
        // `&mut` write-back on failure (6.2, T33): a union function with
        // `&mut` parameters or a `&mut self` receiver records its copy-out
        // when it fails, so the caller writes back on every outcome.
        let wb_fail = union && (muts.iter().any(|m| m.is_some()) || self_mut);
        self.ctx = Ctx::Fn { muts, self_mut };
        if wb_fail {
            match l {
                Lang::Rust => {
                    self.line("let __r: R<Call> = (|| -> R<Call> {");
                    self.ind += 1;
                }
                Lang::Python => self.open("try"),
                Lang::Js | Lang::Scala => self.open("try"),
            }
        }
        match &f.body {
            Body::Arrow(e) => {
                let v = self.ex_stmt(e)?;
                let r = self.ret_pack(&v);
                self.line(r);
            }
            Body::Block(b) => {
                self.block(b, true)?;
                let r = self.ret_pack(self.unit());
                self.line(r);
            }
        }
        if wb_fail {
            let (outs, slf) = self.outs_slf();
            match l {
                Lang::Rust => {
                    self.ind -= 1;
                    self.line("})();");
                    self.line(format!(
                        "if let Err(Flow::Fail(_)) = &__r {{ fail_outs_set(vec![{outs}], {slf}); }}"
                    ));
                    self.line("__r");
                }
                Lang::Python => {
                    self.close();
                    self.open("except Fail as __e");
                    self.line(format!("__e.outs = [{outs}]\n__e.slf = {slf}\nraise"));
                    self.close();
                }
                Lang::Js => {
                    self.ind -= 1;
                    self.line(format!(
                        "}} catch (__e) {{ if (__e instanceof Fail) {{ __e.outs = [{outs}]; __e.slf = {slf}; }} throw __e; }}"
                    ));
                }
                Lang::Scala => {
                    self.ind -= 1;
                    self.line(format!(
                        "}} catch {{ case __e: Fail => __e.outs = V({outs}); __e.slf = {slf}; throw __e }}"
                    ));
                }
            }
        }
        self.close();
        if f.sig.bound.is_some() {
            match l {
                Lang::Rust => {
                    self.open(&format!("fn b_{id}(slf: Option<V>, a: Vec<V>) -> R<Call>"));
                    self.line(format!("let r = c_{id}(slf, a); if !matches!(r, Err(Flow::Trap(_))) {{ depth_exit(\"{id}\"); }} r"));
                }
                Lang::Python => {
                    self.open(&format!("def b_{id}(slf, a)"));
                    self.line(format!("try:\n    r = c_{id}(slf, a)\nexcept Trap:\n    raise\nexcept BaseException:\n    depth_exit(\"{id}\")\n    raise\ndepth_exit(\"{id}\")\nreturn r"));
                }
                Lang::Js => {
                    self.open(&format!("function b_{id}(slf, a)"));
                    self.line(format!("let r; try {{ r = c_{id}(slf, a); }} catch (e) {{ if (!(e instanceof Trap)) depthExit(\"{id}\"); throw e; }} depthExit(\"{id}\"); return r;"));
                }
                Lang::Scala => {
                    self.open(&format!("def b_{id}(slf: Any, a: Vector[Any]): Vector[Any] ="));
                    self.line(format!("val r = try {{ c_{id}(slf, a) }} catch {{ case e: Trap => throw e; case e: Throwable => {{ depth_exit(\"{id}\"); throw e }} }}"));
                    self.line(format!("depth_exit(\"{id}\")"));
                    self.line("r");
                }
            }
            self.close();
        }
        // u_: interp `call_fn`: fit the result; a failure of a union function is a value.
        let ret = f.sig.ret.as_ref().map(|t| self.ty(t));
        let nm = named.len();
        match l {
            Lang::Rust => {
                self.open(&format!("fn u_{id}(slf: Option<V>, a: Vec<V>) -> R<Call>"));
                self.line(format!("match b_{id}(slf, a) {{"));
                match &ret {
                    Some(t) => {
                        self.line(format!("  Ok(mut c) => {{ c.v = fit(c.v, &{t})?; Ok(c) }}"))
                    }
                    None => self.line("  Ok(mut c) => { c.v = V::Unit; Ok(c) }"),
                }
                if wb_fail {
                    self.line(
                        "  Err(Flow::Fail(e)) => { let (o, s) = fail_outs_take(); Ok(Call { v: V::Err(Box::new(e)), outs: o, slf: s }) }",
                    );
                } else if union {
                    self.line(format!(
                        "  Err(Flow::Fail(e)) => Ok(Call {{ v: V::Err(Box::new(e)), outs: vec![None; {nm}], slf: None }}),"
                    ));
                }
                self.line("  Err(f) => Err(f),");
                self.line("}");
            }
            Lang::Python => {
                self.open(&format!("def u_{id}(slf, a)"));
                if wb_fail {
                    self.line(format!(
                        "try:\n    v, outs, s = b_{id}(slf, a)\nexcept Fail as e:\n    return (Err(e.v), e.outs, e.slf)"
                    ));
                } else if union {
                    self.line(format!(
                        "try:\n    v, outs, s = b_{id}(slf, a)\nexcept Fail as e:\n    return (Err(e.v), [None] * {nm}, None)"
                    ));
                } else {
                    self.line(format!("v, outs, s = b_{id}(slf, a)"));
                }
                match &ret {
                    Some(t) => self.line(format!("return (fit(v, {t}), outs, s)")),
                    None => self.line("return (UNIT, outs, s)"),
                }
            }
            Lang::Js => {
                self.open(&format!("function u_{id}(slf, a)"));
                self.line("let r;");
                if wb_fail {
                    self.line(format!(
                        "try {{ r = b_{id}(slf, a); }} catch (e) {{ if (e instanceof Fail) return [new Err(e.v), e.outs, e.slf]; throw e; }}"
                    ));
                } else if union {
                    self.line(format!(
                        "try {{ r = b_{id}(slf, a); }} catch (e) {{ if (e instanceof Fail) return [new Err(e.v), Array({nm}).fill(null), null]; throw e; }}"
                    ));
                } else {
                    self.line(format!("r = b_{id}(slf, a);"));
                }
                match &ret {
                    Some(t) => self.line(format!("return [fit(r[0], {t}), r[1], r[2]];")),
                    None => self.line("return [UNIT, r[1], r[2]];"),
                }
            }
            Lang::Scala => {
                self.open(&format!("def u_{id}(slf: Any, a: Vector[Any]): Vector[Any] ="));
                if wb_fail {
                    self.line(format!(
                        "val r = try {{ b_{id}(slf, a) }} catch {{ case e: Fail => return V(new Err(e.v), e.outs, e.slf) }}"
                    ));
                } else if union {
                    self.line(format!(
                        "val r = try {{ b_{id}(slf, a) }} catch {{ case e: Fail => return V(new Err(e.v), Vector.fill({nm})(null), null) }}"
                    ));
                } else {
                    self.line(format!("val r = b_{id}(slf, a)"));
                }
                match &ret {
                    Some(t) => self.line(format!("V(fit(r(0), {t}), r(1), r(2))")),
                    None => self.line("V(UNIT, r(1), r(2))"),
                }
            }
        }
        self.close();
        // r_: interp `apply` of a function value (no result fit; failures propagate).
        match l {
            Lang::Rust => {
                self.open(&format!("fn r_{id}(a: Vec<V>) -> R<V>"));
                self.line(format!("Ok(b_{id}(None, a)?.v)"));
            }
            Lang::Python => {
                self.open(&format!("def r_{id}(a)"));
                self.line(format!("return b_{id}(None, a)[0]"));
            }
            Lang::Js => {
                self.open(&format!("function r_{id}(a)"));
                self.line(format!("return b_{id}(null, a)[0];"));
            }
            Lang::Scala => {
                self.open(&format!("def r_{id}(a: Vector[Any]): Any ="));
                self.line(format!("b_{id}(null, a)(0)"));
            }
        }
        self.close();
        Ok(())
    }

    fn constant(&mut self, name: &str, ty: &Option<Ty>, e: &Expr) -> Result<(), String> {
        self.scopes = vec![vec![]];
        self.ctx = Ctx::Gen;
        let mut v = self.ex(e)?;
        if let Some(t) = ty {
            v = self.fit(&v, &self.ty(t));
        }
        let k = ident(name);
        match self.lang {
            Lang::Rust => {
                self.line(format!(
                    "thread_local! {{ static K_{k}: RefCell<Option<(u64, V)>> = const {{ RefCell::new(None) }}; }}"
                ));
                self.open(&format!("fn k_{k}() -> R<V>"));
                self.line(format!(
                    "if let Some((g, v)) = K_{k}.with(|c| c.borrow().clone()) && g == kgen() {{ return Ok(v); }}"
                ));
                self.line(format!("let v = {v};"));
                self.line(format!(
                    "K_{k}.with(|c| *c.borrow_mut() = Some((kgen(), v.clone())));"
                ));
                self.line("Ok(v)");
            }
            Lang::Python => {
                self.open(&format!("def k_{k}()"));
                self.line(format!("if \"{k}\" in TLS.kc: return TLS.kc[\"{k}\"]"));
                self.line(format!("v = {v}"));
                self.line(format!("TLS.kc[\"{k}\"] = v"));
                self.line("return v");
            }
            Lang::Js => {
                self.open(&format!("function k_{k}()"));
                self.line(format!("if (\"{k}\" in KC) return KC[\"{k}\"];"));
                self.line(format!("const v = {v};"));
                self.line(format!("KC[\"{k}\"] = v;"));
                self.line("return v;");
            }
            Lang::Scala => {
                self.open(&format!("def k_{k}(): Any ="));
                self.line(format!("if (KC.contains(\"{k}\")) return KC(\"{k}\")"));
                self.line(format!("val v = {v}"));
                self.line(format!("KC(\"{k}\") = v"));
                self.line("v");
            }
        }
        self.close();
        Ok(())
    }

    pub(super) fn step(
        &mut self,
        sid: &str,
        fields: &[Field],
        pat: &Pat,
        guard: &Option<Expr>,
        body: &Body,
    ) -> Result<(), String> {
        let l = self.lang;
        match l {
            Lang::Rust => self.open(&format!(
                "fn {sid}(fs: &Vec<(&'static str, V)>, ev: &V) -> R<Option<Vec<(&'static str, V)>>>"
            )),
            Lang::Python => self.open(&format!("def {sid}(fs, ev)")),
            Lang::Js => self.open(&format!("function {sid}(fs, ev)")),
            Lang::Scala => self.open(&format!("def {sid}(fs: Vector[Any], ev: Any): Any =")),
        }
        self.scopes = vec![vec![]];
        self.ctx = Ctx::Step;
        for (j, f) in fields.iter().enumerate() {
            let get = if l == Lang::Scala {
                format!("at(at(fs, {j}), 1)")
            } else if l == Lang::Rust {
                format!("fs[{j}].1.clone()")
            } else {
                format!("fs[{j}][1]")
            };
            let s = self.let_(&format!("l_{}", ident(&f.name)), &get);
            self.line(s);
            self.declare(&f.name);
        }
        self.scopes.push(vec![]);
        let ps = self.pat(pat)?;
        match l {
            Lang::Rust => self.line(format!(
                "let Some(__pm) = pm(&{ps}, ev)? else {{ return Ok(None); }};"
            )),
            Lang::Python => self.line(format!("__pm = pm({ps}, ev)\nif __pm is None: return None")),
            Lang::Js => self.line(format!(
                "const __pm = pm({ps}, ev);\nif (__pm === null) return null;"
            )),
            Lang::Scala => self.line(format!(
                "val __pm = pm({ps}, ev)\nif (__pm == null) return null"
            )),
        }
        let mut names = vec![];
        self.pat_names(pat, &mut names);
        self.bind_from_map("__pm", &names);
        if let Some(g) = guard {
            let gv = self.ex(g)?;
            let t = self.truthy(&gv);
            self.line(match l {
                Lang::Rust => format!("if !{t} {{ return Ok(None); }}"),
                Lang::Python => format!("if not {t}: return None"),
                Lang::Js => format!("if (!{t}) return null;"),
                Lang::Scala => format!("if (!{t}) return null"),
            });
        }
        match body {
            Body::Arrow(e) => {
                let v = self.ex_stmt(e)?;
                if let Some(f0) = fields.first() {
                    let s = self.set_(&format!("l_{}", ident(&f0.name)), &v);
                    self.line(s);
                }
            }
            Body::Block(b) => {
                self.open_block();
                self.block(b, false)?;
                self.close_block();
            }
        }
        let mut outs = vec![];
        for f in fields {
            let cur = if l == Lang::Rust {
                format!("l_{}.clone()", ident(&f.name))
            } else {
                format!("l_{}", ident(&f.name))
            };
            let v = self.fit(&cur, &self.ty(&f.ty));
            outs.push(self.pair(&q(&f.name), &v));
        }
        self.line(match l {
            Lang::Rust => format!("Ok(Some(vec![{}]))", outs.join(", ")),
            Lang::Python => format!("return [{}]", outs.join(", ")),
            Lang::Js => format!("return [{}];", outs.join(", ")),
            Lang::Scala => format!("V({})", outs.join(", ")),
        });
        self.close();
        Ok(())
    }
}

impl<'a> G<'a> {
    pub fn program(&mut self) -> Result<String, String> {
        let l = self.lang;
        let m = self.m;
        let mut loop_limit: i128 = crate::interp::LOOP_LIMIT;
        for i in &m.items {
            if let Item::Const {
                name,
                e: Expr::Int(n, _),
                ..
            } = i
                && name == "LOOP_LIMIT"
                && *n > 0
            {
                loop_limit = *n;
            }
        }
        match l {
            Lang::Rust => {
                self.out.push(RT_RS.to_string());
                self.line(format!("const LOOP_LIMIT: i128 = {loop_limit};"));
            }
            Lang::Python => {
                self.out.push(RT_PY.to_string());
                self.line(format!("LOOP_LIMIT = {loop_limit}"));
            }
            Lang::Js => {
                self.out.push(RT_JS.to_string());
                self.line(format!(
                    "const LOOP_LIMIT = {loop_limit};\nconst KC = {{}};"
                ));
            }
            Lang::Scala => {
                self.out.push(RT_SCALA.to_string());
                self.line(format!("val LOOP_LIMIT: BigInt = BigInt(\"{loop_limit}\")"));
            }
        }
        let mut fn_ids: Vec<(String, String)> = vec![];
        for i in &m.items {
            if let Item::Fn(f) = i {
                let id = ident(&f.sig.name);
                self.function(&id, f)?;
                fn_ids.push((f.sig.name.clone(), id));
            }
        }
        let mut ms: Vec<(String, String, String, &FnDecl)> = self
            .methods
            .iter()
            .map(|((h, n), (id, f))| (h.clone(), n.clone(), id.clone(), *f))
            .collect();
        ms.sort_by(|a, b| a.2.cmp(&b.2));
        for (_, _, id, f) in &ms {
            self.function(id, f)?;
        }
        for i in &m.items {
            if let Item::Const { name, ty, e, .. } = i {
                self.constant(name, ty, e)?;
            }
        }
        let states: Vec<(String, Vec<Field>)> = m
            .items
            .iter()
            .filter_map(|i| {
                if let Item::State { name, fields, .. } = i {
                    Some((name.clone(), fields.clone()))
                } else {
                    None
                }
            })
            .collect();
        let mut steps: Vec<(String, String)> = vec![];
        for (k, i) in m.items.iter().enumerate() {
            let Item::Step {
                region,
                pat,
                guard,
                body,
                ..
            } = i
            else {
                continue;
            };
            let Some((_, fields)) = states.iter().find(|(n, _)| n == region) else {
                continue;
            };
            let sid = format!("s_{k}");
            self.step(&sid, fields, pat, guard, body)?;
            steps.push((region.clone(), sid));
        }
        self.initial(&states)?;
        self.actor_tables()?;
        self.tables(&fn_ids, &ms, &states, &steps);
        if self.lang == Lang::Js {
            let h = std::mem::take(&mut self.hoist);
            self.out.extend(h);
            self.out.push(format!("const PAR_FNS = {{ {} }};", self.hoist_names.join(", ")));
        }
        self.driver()?;
        Ok(self.out.join("\n") + "\n")
    }

    fn initial(&mut self, states: &[(String, Vec<Field>)]) -> Result<(), String> {
        let l = self.lang;
        match l {
            Lang::Rust => self.open("fn initial_state() -> R<State>"),
            Lang::Python => self.open("def initial_state()"),
            Lang::Js => self.open("function initialState()"),
            Lang::Scala => self.open("def initial_state(): Vector[Any] ="),
        }
        self.scopes = vec![vec![]];
        self.ctx = Ctx::Gen;
        let mut regs = vec![];
        for (name, fields) in states {
            let mut fs = vec![];
            for f in fields {
                let v = match &f.init {
                    Some(e) => {
                        let v = self.ex(e)?;
                        self.fit(&v, &self.ty(&f.ty))
                    }
                    None => self.trap_e("field without initial value"),
                };
                let t = self.fresh("f");
                let s = self.let_(&t, &v);
                self.line(s);
                fs.push(self.pair(&q(&f.name), &t));
            }
            regs.push(self.pair(&q(name), &self.vec(&fs)));
        }
        let r = self.vec(&regs);
        self.line(match l {
            Lang::Rust => format!("Ok({r})"),
            Lang::Python => format!("return {r}"),
            Lang::Js => format!("return {r};"),
            Lang::Scala => r.clone(),
        });
        self.close();
        Ok(())
    }

    fn tables(
        &mut self,
        fn_ids: &[(String, String)],
        ms: &[(String, String, String, &FnDecl)],
        states: &[(String, Vec<Field>)],
        steps: &[(String, String)],
    ) {
        let mut kinds: Vec<(String, String)> = self
            .variants
            .iter()
            .map(|(a, (k, _))| (a.clone(), k.clone()))
            .collect();
        kinds.sort();
        let steps_of = |r: &str| -> Vec<String> {
            steps
                .iter()
                .filter(|s| s.0 == r)
                .map(|s| s.1.clone())
                .collect()
        };
        match self.lang {
            Lang::Rust => {
                let mut o = String::new();
                o += "fn user_fn(n: &str, slf: Option<V>, a: Vec<V>) -> R<Call> {\n  match n {\n";
                for (n, id) in fn_ids {
                    o += &format!("    {} => u_{id}(slf, a),\n", q(n));
                }
                o += "    _ => trap(format!(\"unknown function {n}\")),\n  }\n}\n";
                o += "fn has_user_fn(n: &str) -> bool {\n  matches!(n, \"\"";
                for (n, _) in fn_ids.iter() {
                    o += &format!(" | {}", q(n));
                }
                o += ")\n}\n";
                o += "fn raw_fn(n: &str, a: Vec<V>) -> R<V> {\n  match n {\n";
                for (n, id) in fn_ids {
                    o += &format!("    {} => r_{id}(a),\n", q(n));
                }
                o += "    _ => trap(format!(\"unknown function {n}\")),\n  }\n}\n";
                o += "fn user_method(h: &str, m: &str) -> Option<fn(Option<V>, Vec<V>) -> R<Call>> {\n  match (h, m) {\n";
                for (h, n, id, _) in ms {
                    o += &format!("    ({}, {}) => Some(u_{id}),\n", q(h), q(n));
                }
                o += "    _ => None,\n  }\n}\n";
                o += "fn kind_of(n: &str) -> &str {\n  match n {\n";
                for (a, k) in &kinds {
                    o += &format!("    {} => {},\n", q(a), q(k));
                }
                o += "    _ => n,\n  }\n}\n";
                o += "type StepFn = fn(&Vec<(&'static str, V)>, &V) -> R<Option<Vec<(&'static str, V)>>>;\n";
                o += "fn steps_of(r: &str) -> Vec<StepFn> {\n  match r {\n";
                for (name, _) in states {
                    let ids: Vec<String> = steps_of(name)
                        .into_iter()
                        .map(|i| format!("{i} as StepFn"))
                        .collect();
                    o += &format!("    {} => vec![{}],\n", q(name), ids.join(", "));
                }
                o += "    _ => vec![],\n  }\n}";
                self.line(o);
            }
            Lang::Scala => {
                let fns: Vec<String> = fn_ids.iter().map(|(n, id)| format!("{} -> (u_{id} _)", q(n))).collect();
                let raw: Vec<String> = fn_ids.iter().map(|(n, id)| format!("{} -> (r_{id} _)", q(n))).collect();
                let meth: Vec<String> = ms.iter().map(|(h, n, id, _)| format!("{} -> (u_{id} _)", q(&format!("{h}.{n}")))).collect();
                let kind: Vec<String> = kinds.iter().map(|(a, k)| format!("{} -> {}", q(a), q(k))).collect();
                let st: Vec<String> = states
                    .iter()
                    .map(|(n, _)| {
                        let f: Vec<String> = steps_of(n).iter().map(|s| format!("({s} _)")).collect();
                        format!("{} -> Vector[(Vector[Any], Any) => Any]({})", q(n), f.join(", "))
                    })
                    .collect();
                self.line(format!("lazy val FNS: Map[String, (Any, Vector[Any]) => Vector[Any]] = Map({})", fns.join(", ")));
                self.line(format!("lazy val RAW: Map[String, Vector[Any] => Any] = Map({})", raw.join(", ")));
                self.line(format!("lazy val METHODS: Map[String, (Any, Vector[Any]) => Vector[Any]] = Map({})", meth.join(", ")));
                self.line(format!("lazy val KIND_OF: Map[String, String] = Map({})", kind.join(", ")));
                self.line(format!("lazy val STEPS: Map[String, Vector[(Vector[Any], Any) => Any]] = Map({})", st.join(", ")));
            }
            Lang::Python | Lang::Js => {
                let py = self.lang == Lang::Python;
                let decl = if py { "" } else { "const " };
                let end = if py { "" } else { ";" };
                let fns: Vec<String> = fn_ids
                    .iter()
                    .map(|(n, id)| format!("{}: u_{id}", q(n)))
                    .collect();
                let raw: Vec<String> = fn_ids
                    .iter()
                    .map(|(n, id)| format!("{}: r_{id}", q(n)))
                    .collect();
                let meth: Vec<String> = ms
                    .iter()
                    .map(|(h, n, id, _)| {
                        if py {
                            format!("({}, {}): u_{id}", q(h), q(n))
                        } else {
                            format!("{}: u_{id}", q(&format!("{h}.{n}")))
                        }
                    })
                    .collect();
                let kind: Vec<String> = kinds
                    .iter()
                    .map(|(a, k)| format!("{}: {}", q(a), q(k)))
                    .collect();
                let st: Vec<String> = states
                    .iter()
                    .map(|(n, _)| format!("{}: [{}]", q(n), steps_of(n).join(", ")))
                    .collect();
                for (name, xs) in [
                    ("FNS", fns),
                    ("RAW", raw),
                    ("METHODS", meth),
                    ("KIND_OF", kind),
                    ("STEPS", st),
                ] {
                    self.line(format!("{decl}{name} = {{{}}}{end}", xs.join(", ")));
                }
            }
        }
    }

    fn obs_line(&mut self, label: &str, v: &str) {
        let label = q(label);
        self.line(match self.lang {
            Lang::Rust => format!("obs({label}, &|| Ok({v}));"),
            Lang::Python => format!("obs({label}, lambda: {v})"),
            Lang::Js => format!("obs({label}, () => {v});"),
            Lang::Scala => format!("obs({label}, () => {v})"),
        });
    }

    fn obs_example(&mut self, x: &Example) -> Result<(), String> {
        let l = self.lang;
        let mut evs = vec![];
        for (evn, args) in &x.events {
            let c = self.construct_code(evn, args)?;
            let n = q(evn);
            evs.push(match (l, c) {
                (Lang::Rust, c) => format!(
                    "({n}, Rc::new(|| -> R<Option<V>> {{ Ok({}) }}) as Rc<dyn Fn() -> R<Option<V>> + Send + Sync>)",
                    c.map(|c| format!("Some({c})")).unwrap_or("None".into())
                ),
                (Lang::Python, c) => format!("({n}, lambda: {})", c.unwrap_or("None".into())),
                (Lang::Js, c) => format!("[{n}, () => {}]", c.unwrap_or("null".into())),
                (Lang::Scala, c) => format!("V({n}, () => {})", c.unwrap_or("null".into())),
            });
        }
        let s = self.stmt_call("run_example", &format!("{}, {}", x.line, self.vec(&evs)));
        // run_example returns nothing: drop only the statement's own `?`
        let s = match s.strip_suffix(")?;") {
            Some(h) => format!("{h});"),
            None => s,
        };
        self.line(s);
        Ok(())
    }

    fn driver(&mut self) -> Result<(), String> {
        let l = self.lang;
        let m = self.m;
        match l {
            Lang::Rust => self.open("fn main()"),
            Lang::Python => self.open("def observe()"),
            Lang::Js => self.open("function observe()"),
            Lang::Scala => self.open("def observe(): Unit ="),
        }
        // `--run` without `fn main()` refuses as `conc::run_main` does (10.3)
        let has_main = m.items.iter().any(|i| matches!(i, Item::Fn(f) if f.sig.name == "main" && f.sig.params.is_empty()));
        let run = |r: &str, py: bool| -> String {
            match (l, has_main) {
                (_, true) => r.to_string(),
                (Lang::Python, false) if py => "(sys.stderr.write(\"nothing to run: declare `fn main()`\\n\"), sys.exit(1))".into(),
                (Lang::Js, false) => "{ process.stderr.write(\"nothing to run: declare `fn main()`\\n\"); process.exit(1); }".into(),
                _ => "{ eprintln!(\"nothing to run: declare `fn main()`\"); std::process::exit(1); }".into(),
            }
        };
        if l == Lang::Rust {
            let r = run("run_tasks(\"main\")", false);
            self.line(format!("if std::env::args().any(|a| a == \"--run\") {{ {r}; return; }}"));
        }
        self.scopes = vec![vec![]];
        self.ctx = Ctx::Gen;
        let mut main_id = None;
        for i in &m.items {
            match i {
                Item::Const { name, .. } => {
                    let v = if l == Lang::Rust {
                        format!("k_{}()?", ident(name))
                    } else {
                        format!("k_{}()", ident(name))
                    };
                    self.obs_line(&format!("const {name}"), &v);
                }
                Item::Check(e) => {
                    let v = self.ex(e)?;
                    self.obs_line(&format!("check {}", e.line()), &v);
                }
                Item::Intent(_, xs, cks) => {
                    for e in cks {
                        let v = self.ex(e)?;
                        self.obs_line(&format!("check {}", e.line()), &v);
                    }
                    for x in xs {
                        self.obs_example(x)?;
                    }
                }
                Item::Example(x) => self.obs_example(x)?,
                Item::Fn(f) if f.sig.name == "main" && f.sig.params.is_empty() => {
                    main_id = Some(ident(&f.sig.name))
                }
                Item::Fn(f) if zero_arg(f) => {
                    let id = ident(&f.sig.name);
                    let v = match l {
                        Lang::Rust => format!("u_{id}(None, vec![])?.v"),
                        Lang::Python => format!("u_{id}(None, [])[0]"),
                        Lang::Js => format!("u_{id}(null, [])[0]"),
                        Lang::Scala => format!("u_{id}(null, V())(0)"),
                    };
                    // interp `call_fn` ticks on entry (6.7)
                    let v = self.pre_fuel(&v);
                    self.obs_line(&format!("fn {}", f.sig.name), &v);
                }
                _ => {}
            }
        }
        if let Some(id) = main_id {
            self.line(match l {
                Lang::Rust => format!("run_main(&u_{id});"),
                Lang::Python => format!("run_main(u_{id})"),
                Lang::Js => format!("runMain(u_{id});"),
                Lang::Scala => format!("run_main(u_{id} _)"),
            });
        }
        self.close();
        match l {
            Lang::Rust => {}
            Lang::Python => {
                let r = run("run_tasks(\"main\")", true);
                self.line(format!("if \"--run\" in sys.argv: {r}\nelse: observe()"))
            }
            Lang::Js => {
                let r = run("runTasks(\"main\");", false);
                self.line(format!(
                    "if (!WT.isMainThread) {{ if (WT.workerData && WT.workerData.par) parMain(); else taskMain(); }}\nelse if (process.argv.includes(\"--run\")) {r}\nelse {{\n  observe();\n  process.stdout.write(LINES.map((x) => x + \"\\n\").join(\"\"));\n}}"
                ))
            }
            Lang::Scala => {
                let r = if has_main {
                    "run_tasks(\"main\")"
                } else {
                    "{ System.err.println(\"nothing to run: declare `fn main()`\"); sys.exit(1) }"
                };
                self.line(format!(
                    "@main def rce_main(args: String*): Unit = big_stack {{ () =>\n  init_args(args)\n  if (args.contains(\"--run\")) {r}\n  else {{\n    observe()\n    val o = new java.io.PrintStream(new java.io.FileOutputStream(java.io.FileDescriptor.out), false, \"UTF-8\")\n    for (x <- LINES) {{ o.print(x); o.print(\"\\n\") }}\n    o.flush()\n  }}\n}}"
                ))
            }

        }
        Ok(())
    }
}
