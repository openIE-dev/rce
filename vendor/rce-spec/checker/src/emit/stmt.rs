//! Statements: target control flow, with places rebuilt the way the
//! reference evaluator's `assign` rebuilds them (values never change in place).
use super::Lang;
use super::base::*;
use crate::ast::*;

impl<'a> G<'a> {
    /// Evaluate `e` as a statement-level expression, writing back receivers
    /// and `&mut` arguments; returns an expression for the value.
    pub fn ex_stmt(&mut self, e: &Expr) -> E {
        // `CALL catch h` and `try CALL` with a write-back call: the call is
        // the first thing the statement evaluates, so it is hoisted, its
        // copy-out written back on every outcome (6.2), and the catch or
        // try applied to its value. Evaluation order is unchanged.
        match e {
            Expr::Catch(x, bind, h, line) if self.is_wb_call(x) => {
                let v = self.ex_stmt(x)?;
                let t = self.fresh("w");
                let s = self.let_(&t, &v);
                self.line(s);
                let ne = Expr::Catch(Box::new(Expr::Name(format!("\u{1}{t}"), *line)), bind.clone(), h.clone(), *line);
                return self.ex(&ne);
            }
            Expr::Try(x, els, line) if self.is_wb_call(x) => {
                let v = self.ex_stmt(x)?;
                let t = self.fresh("w");
                let s = self.let_(&t, &v);
                self.line(s);
                let ne = Expr::Try(Box::new(Expr::Name(format!("\u{1}{t}"), *line)), els.clone(), *line);
                return self.ex(&ne);
            }
            _ => {}
        }
        let Expr::Call(f, args, _, sz) = e else {
            return self.ex(e);
        };
        let Some((call, wb)) = self.call_expr(f, args, true, *sz)? else {
            return self.ex(e);
        };
        // the call expression's own tick (interp `eval`), before its operands
        let call = if self.no_fuel > 0 { call } else { self.pre_fuel(&call) };
        let Some(wb) = wb else { return Ok(call) };
        let l = self.lang;
        let t = wb.tmp.clone();
        self.line(match l {
            Lang::Rust => format!("let {t} = {call};"),
            Lang::Python => format!("{t} = {call}"),
            Lang::Js => format!("const {t} = {call};"),
            Lang::Scala => format!("val {t} = {call}"),
        });
        // Arguments first, then the receiver (interp `call_fn`, then `builtin`).
        for (i, a) in wb.args.iter().enumerate() {
            let get = match l {
                Lang::Rust => format!("{t}.outs.get({i}).cloned().flatten()"),
                Lang::Python => format!("({t}[1][{i}] if len({t}[1]) > {i} else None)"),
                Lang::Js => format!("({t}[1].length > {i} ? {t}[1][{i}] : null)"),
                Lang::Scala => format!("atOr(at({t}, 1), {i})"),
            };
            let o = self.fresh("o");
            self.open_some(&o, &get);
            if is_place(&a.e) {
                self.assign(&a.e, &o)?;
            } else {
                let s = self.trap_s("not a place");
                self.line(s);
            }
            self.close();
        }
        if let Some(obj) = &wb.recv {
            let o = self.fresh("o");
            let get = match l {
                Lang::Rust => format!("{t}.slf.clone()"),
                Lang::Scala => format!("at({t}, 2)"),
                _ => format!("{t}[2]"),
            };
            self.open_some(&o, &get);
            if is_place(obj) {
                self.assign(obj, &o)?;
            } else {
                let s = self.trap_s("not a place");
                self.line(s);
            }
            self.close();
        }
        Ok(match l {
            Lang::Rust => format!("{t}.v.clone()"),
            Lang::Scala => format!("at({t}, 0)"),
            _ => format!("{t}[0]"),
        })
    }

    /// Assign the value in temporary `v` to `place` (interp `assign`).
    pub fn assign(&mut self, place: &Expr, v: &str) -> Result<(), String> {
        match place {
            Expr::Name(n, _) if self.local(n) => {
                let s = self.set_(&format!("l_{}", ident(n)), v);
                self.line(s);
            }
            Expr::Name(n, _) => {
                let s = self.trap_s(&format!("assignment to unknown {n}"));
                self.line(s);
            }
            Expr::SelfRef(_) => {
                if self.local("self") {
                    let s = self.set_("l_self", v);
                    self.line(s);
                }
            }
            Expr::Borrow(_, x, _) => self.assign(x, v)?,
            Expr::Member(x, m, _) => {
                let cur = self.ex(x)?;
                let t = self.fresh("t");
                let s = self.let_(&t, &self.h("set_field", &[cur, q(m), v.to_string()]));
                self.line(s);
                self.assign(x, &t)?;
            }
            Expr::TupleMember(x, i, _) => {
                let cur = self.ex(x)?;
                let t = self.fresh("t");
                let s = self.let_(
                    &t,
                    &self.h("set_tuple", &[cur, i.to_string(), v.to_string()]),
                );
                self.line(s);
                self.assign(x, &t)?;
            }
            Expr::Index(x, idx, _) => {
                let k = self.ex(&idx[0])?;
                let kt = self.fresh("k");
                let s = self.let_(&kt, &k);
                self.line(s);
                let cur = self.ex(x)?;
                let t = self.fresh("t");
                let s = self.let_(&t, &self.h("set_index", &[cur, kt, v.to_string()]));
                self.line(s);
                self.assign(x, &t)?;
            }
            _ => {
                let s = self.trap_s("not a place");
                self.line(s);
            }
        }
        Ok(())
    }

    /// `return v` in a function: (value, copy-out of `&mut` params, self).
    /// `return v` in the current context; inside a task closure (scope or
    /// deadline body) it leaves the closure as `Flow::Ret`, re-raised after it
    pub fn ret_with(&mut self, v: &str) {
        let l = self.lang;
        if !self.cl.is_empty() {
            self.line(format!("return Err(Flow::Ret({v}));"));
            return;
        }
        let s = match &self.ctx {
            Ctx::Fn { .. } => self.ret_pack(v),
            Ctx::Gen => match l {
                Lang::Rust => format!("{{ let _ = {v}; return Ok(()); }}"),
                Lang::Python => format!("{v}\nreturn"),
                Lang::Js => format!("{v}; return;"),
                Lang::Scala => format!("discard({v}); return"),
            },
            Ctx::Step => match l {
                Lang::Rust => format!("return Err(Flow::Ret({v}));"),
                Lang::Python => format!("raise Ret({v})"),
                Lang::Js => format!("throw new Ret({v});"),
                Lang::Scala => format!("throw new Ret({v})"),
            },
        };
        self.line(s);
    }
    /// The copy-out list and receiver of the current function frame.
    pub fn outs_slf(&self) -> (String, String) {
        let Ctx::Fn { muts, self_mut } = &self.ctx else {
            return (String::new(), self.none().to_string());
        };
        let l = self.lang;
        let outs: Vec<String> = muts
            .iter()
            .map(|m| match (m, l) {
                (Some(n), Lang::Rust) => format!("Some(l_{}.clone())", ident(n)),
                (Some(n), _) => format!("l_{}", ident(n)),
                (None, _) => self.none().into(),
            })
            .collect();
        let slf = match (*self_mut && self.local("self"), l) {
            (true, Lang::Rust) => "Some(l_self.clone())".to_string(),
            (true, _) => "l_self".to_string(),
            (false, _) => self.none().to_string(),
        };
        (outs.join(", "), slf)
    }
    pub fn ret_pack(&self, v: &str) -> String {
        let Ctx::Fn { muts, self_mut } = &self.ctx else {
            return v.to_string();
        };
        let l = self.lang;
        let outs: Vec<String> = muts
            .iter()
            .map(|m| match (m, l) {
                (Some(n), Lang::Rust) => format!("Some(l_{}.clone())", ident(n)),
                (Some(n), _) => format!("l_{}", ident(n)),
                (None, _) => self.none().into(),
            })
            .collect();
        let slf = match (*self_mut && self.local("self"), l) {
            (true, Lang::Rust) => "Some(l_self.clone())".to_string(),
            (true, _) => "l_self".to_string(),
            (false, _) => self.none().to_string(),
        };
        match l {
            Lang::Rust => format!(
                "return Ok(Call {{ v: {v}, outs: vec![{}], slf: {slf} }});",
                outs.join(", ")
            ),
            Lang::Python => format!("return ({v}, [{}], {slf})", outs.join(", ")),
            Lang::Js => format!("return [{v}, [{}], {slf}];", outs.join(", ")),
            Lang::Scala => format!("return V({v}, V({}), {slf})", outs.join(", ")),
        }
    }

    /// `for p in e par body` (draft 0.6 C.5.2): each iteration starts its
    /// reduction targets at the identity; the per-iteration values are
    /// combined by the fixed tree T and then with the value before the loop,
    /// exactly as the reference evaluator does (so float results agree bit
    /// for bit). Slot writes `out[i] = ...` need no transformation.
    fn forpar(&mut self, body: &[Stmt]) -> Result<(), String> {
        let Some(Stmt::For(_, _, b, _)) = body.first() else {
            return Err("bad for par".into());
        };
        let l = self.lang;
        let mut targets = vec![];
        crate::conc::assign_ops(b, &mut targets);
        let local = crate::conc::declared_in(b);
        targets.retain(|(n, _)| !local.contains(n));
        if l != Lang::Scala && super::conc06::forpar_threadable(b) {
            return self.forpar_threads(&body[0], &targets);
        }
        if targets.is_empty() {
            return self.stmt(&body[0], false);
        }
        self.open_block();
        let mut names = vec![];
        for (n, op) in &targets {
            let sym = q(&op[..op.len() - 1]);
            let (bk, ik, ck) = (self.fresh("pb"), self.fresh("pi"), self.fresh("pc"));
            let cur = self.var(n);
            let s = self.let_(&bk, &cur);
            self.line(s);
            let idv = match l {
                Lang::Rust => format!("par_ident({sym}, {bk}.clone())?"),
                _ => self.h("par_ident", &[sym.clone(), bk.clone()]),
            };
            let s = self.let_(&ik, &idv);
            self.line(s);
            let s = self.let_(&ck, &self.vec(&[]));
            self.line(s);
            names.push((n.clone(), sym, bk, ik, ck));
        }
        // the loop, with the reset and the collection around each iteration
        let Stmt::For(p, it, b, line) = &body[0] else { unreachable!() };
        let mut nb = vec![];
        for (n, _, _, ik, _) in &names {
            nb.push(Stmt::Expr(Expr::Call(
                Box::new(Expr::Name("$par_reset".into(), 0)),
                vec![
                    Arg { label: None, e: Expr::Name(n.clone(), 0) },
                    Arg { label: None, e: Expr::Name(ik.clone(), 0) },
                ],
                0,
                false,
            )));
        }
        nb.extend(b.iter().cloned());
        for (n, _, _, _, ck) in &names {
            nb.push(Stmt::Expr(Expr::Call(
                Box::new(Expr::Name("$par_collect".into(), 0)),
                vec![
                    Arg { label: None, e: Expr::Name(ck.clone(), 0) },
                    Arg { label: None, e: Expr::Name(n.clone(), 0) },
                ],
                0,
                false,
            )));
        }
        self.par_hooks = true;
        let r = self.stmt(&Stmt::For(p.clone(), it.clone(), nb, *line), false);
        self.par_hooks = false;
        r?;
        for (n, sym, bk, ik, ck) in &names {
            let v = self.fresh("pv");
            let e = match l {
                Lang::Rust => format!("par_combine({sym}, {bk}.clone(), par_tree({sym}, &{ck}, &{ik})?)?"),
                _ => self.h("par_combine", &[sym.clone(), bk.clone(), self.h("par_tree", &[sym.clone(), ck.clone(), ik.clone()])]),
            };
            let s = self.let_(&v, &e);
            self.line(s);
            self.assign(&Expr::Name(n.clone(), 0), &v)?;
        }
        self.close_block();
        Ok(())
    }

    pub fn block(&mut self, b: &[Stmt], tail: bool) -> Result<(), String> {
        self.scopes.push(vec![]);
        for (i, s) in b.iter().enumerate() {
            self.stmt(s, tail && i + 1 == b.len())?;
        }
        self.scopes.pop();
        Ok(())
    }
    pub fn open_block(&mut self) {
        if self.lang == Lang::Scala {
            // a bare `{` after a line would be read as an argument block
            self.line("locally {");
            self.ind += 1;
        } else if self.lang != Lang::Python {
            self.line("{");
            self.ind += 1;
        }
    }
    pub fn close_block(&mut self) {
        if self.lang != Lang::Python {
            self.ind -= 1;
            self.line("}");
        }
    }

    pub fn bind_from_map(&mut self, m: &str, names: &[String]) {
        for n in names {
            let get = match self.lang {
                Lang::Rust => format!("mget(&{m}, {})", q(n)),
                Lang::Python => format!("{m}.get({}, UNIT)", q(n)),
                Lang::Js => format!("({0} in {m} ? {m}[{0}] : UNIT)", q(n)),
                Lang::Scala => format!("getb({m}, {})", q(n)),
            };
            let s = self.let_(&format!("l_{}", ident(n)), &get);
            self.line(s);
            self.declare(n);
        }
    }

    fn bind_pat(&mut self, p: &Pat, v: &str) -> Result<(), String> {
        let l = self.lang;
        match p {
            Pat::Path(path) if path.len() == 1 && !self.is_tag(&path[0]) => {
                let s = self.let_(&format!("l_{}", ident(&path[0])), v);
                self.line(s);
                self.declare(&path[0]);
            }
            Pat::Wild => {
                let s = self.discard(v);
                self.line(s);
            }
            _ => {
                let ps = self.pat(p)?;
                let mt = self.fresh("m");
                match l {
                    Lang::Rust => self.line(format!(
                        "let Some({mt}) = pm(&{ps}, &{v})? else {{ return Err(Flow::Trap(\"let pattern did not match\".to_string())); }};"
                    )),
                    Lang::Python => {
                        self.line(format!("{mt} = pm({ps}, {v})"));
                        self.line(format!("if {mt} is None: trap(\"let pattern did not match\")"));
                    }
                    Lang::Js => {
                        self.line(format!("const {mt} = pm({ps}, {v});"));
                        self.line(format!("if ({mt} === null) trap(\"let pattern did not match\");"));
                    }
                    Lang::Scala => {
                        self.line(format!("val {mt} = pm({ps}, {v})"));
                        self.line(format!("if ({mt} == null) trap(\"let pattern did not match\")"));
                    }
                }
                let mut names = vec![];
                self.pat_names(p, &mut names);
                self.bind_from_map(&mt, &names);
            }
        }
        Ok(())
    }

    fn loop_counter(&mut self, bnd: &Option<Expr>) -> Result<(String, String), String> {
        let bound = match bnd {
            Some(b) => {
                let v = self.ex(b)?;
                format!("{}({v})", self.rt("loop_bound"))
            }
            None if self.lang == Lang::Js => "BigInt(LOOP_LIMIT)".into(),
            None => "LOOP_LIMIT".into(),
        };
        let (b, n) = (self.fresh("b"), self.fresh("n"));
        match self.lang {
            Lang::Rust => {
                self.line(format!("let {b}: i128 = {bound};"));
                self.line(format!("let mut {n}: i128 = 0;"));
            }
            Lang::Python => {
                self.line(format!("{b} = {bound}"));
                self.line(format!("{n} = 0"));
            }
            Lang::Js => {
                self.line(format!("const {b} = {bound};"));
                self.line(format!("let {n} = 0n;"));
            }
            Lang::Scala => {
                self.line(format!("val {b}: BigInt = {bound}"));
                self.line(format!("var {n}: BigInt = BigInt(0)"));
            }
        }
        Ok((b, n))
    }
    fn loop_tick(&mut self, b: &str, n: &str) {
        match self.lang {
            Lang::Rust => {
                self.line(format!("{n} += 1;"));
                self.line(format!(
                    "if {n} > {b} {{ return Err(Flow::Trap(\"loop bound\".to_string())); }}"
                ));
            }
            Lang::Python => {
                self.line(format!("{n} += 1"));
                self.line(format!("if {n} > {b}: trap(\"loop bound\")"));
            }
            Lang::Js => {
                self.line(format!("{n} += 1n;"));
                self.line(format!("if ({n} > {b}) trap(\"loop bound\");"));
            }
            Lang::Scala => {
                self.line(format!("{n} += 1"));
                self.line(format!("if ({n} > {b}) trap(\"loop bound\")"));
            }
        }
    }
    /// Scala has no `break` / `continue`: a loop is `try { while (true) {
    /// try { .. } catch CONT } } catch BRK`, and the statements throw.
    fn sc_loop_open(&mut self, head: &str) {
        self.open("try");
        self.open(head);
        self.open("try");
    }
    fn sc_loop_close(&mut self) {
        self.ind -= 1;
        self.line("} catch { case _: ContEx => () }");
        self.close();
        self.ind -= 1;
        self.line("} catch { case _: BrkEx => () }");
    }
    /// Close an `if` arm and open its `else`.
    fn else_(&mut self) {
        self.pad_py();
        self.ind -= 1;
        self.line(if self.lang == Lang::Python {
            "else:"
        } else {
            "} else {"
        });
        self.ind += 1;
    }

    pub fn stmt(&mut self, s: &Stmt, tail: bool) -> Result<(), String> {
        let l = self.lang;
        // interp `exec` ticks once per statement (6.7); the `let` of an
        // A-normal-form temporary is not a source statement.
        // Nor are the hooks and the per-chunk loop of `for ... par`: the
        // reference resets and collects targets without evaluating anything.
        let synthetic = matches!(s, Stmt::Let(Pat::Path(p), None, _, _) if p.len() == 1 && p[0].starts_with("anf_"))
            || matches!(s, Stmt::Expr(Expr::Call(f, _, _, _)) if matches!(&**f, Expr::Name(h, _) if h == "$par_reset" || h == "$par_collect"))
            || matches!(s, Stmt::For(_, Expr::Name(n, _), _, _) if n.starts_with("$fpchunk"));
        if !synthetic && self.no_fuel == 0 {
            let t = self.stmt_call("fuel", "");
            self.line(t);
        }
        if let Stmt::Expr(Expr::Call(f, a, _, _)) = s
            && let Expr::Name(h, _) = &**f
            && (h == "$par_reset" || h == "$par_collect")
            && let [Arg { e: Expr::Name(x, _), .. }, Arg { e: Expr::Name(y, _), .. }] = &a[..]
        {
            if h == "$par_reset" {
                let v = match l {
                    Lang::Rust => format!("{y}.clone()"),
                    _ => y.clone(),
                };
                self.assign(&Expr::Name(x.clone(), 0), &v)?;
            } else {
                let cur = self.var(y);
                self.line(match l {
                    Lang::Rust => format!("{x}.push({cur});"),
                    Lang::Python => format!("{x}.append({cur})"),
                    Lang::Scala => format!("{x} = vpush({x}, {cur})"),
                    Lang::Js => format!("{x}.push({cur});"),
                });
            }
            return Ok(());
        }
        match s {
            Stmt::Expr(e) => {
                let v = self.ex_stmt(e)?;
                let s = if tail && matches!(self.ctx, Ctx::Fn { .. }) {
                    self.ret_pack(&v)
                } else {
                    self.discard(&v)
                };
                self.line(s);
            }
            Stmt::Assign(op, place, rhs) => {
                let r = self.ex_stmt(rhs)?;
                let rt = self.fresh("r");
                let s = self.let_(&rt, &r);
                self.line(s);
                let v = if op == "=" {
                    rt
                } else {
                    let cur = self.ex(place)?;
                    self.h("compound", &[q(&op[..op.len() - 1]), cur, rt])
                };
                let pe = self.ex(place)?;
                let cur = match l {
                    Lang::Rust => format!("(|| -> R<V> {{ Ok({pe}) }})().ok()"),
                    Lang::Python => format!("try_eval(lambda: {pe})"),
                    Lang::Js => format!("tryEval(() => {pe})"),
                    Lang::Scala => format!("try_eval(() => {pe})"),
                };

                let vt = self.fresh("v");
                let s = self.let_(&vt, &self.h("fixtag", &[cur, v]));
                self.line(s);
                self.assign(place, &vt)?;
            }
            Stmt::Let(p, ty, e, _) => {
                let mut v = self.ex_stmt(e)?;
                if let Some(t) = ty {
                    v = self.fit(&v, &self.ty(t));
                }
                self.bind_pat(p, &v)?;
            }
            Stmt::Var(n, ty, e, _) => {
                let mut v = self.ex_stmt(e)?;
                if let Some(t) = ty {
                    v = self.fit(&v, &self.ty(t));
                }
                self.bind_pat(&Pat::Path(vec![n.clone()]), &v)?;
            }
            Stmt::If(arms, els) => {
                // Each later arm nests in the else of the one before, so its
                // condition is evaluated only when the earlier ones are false.
                for (i, (c, b)) in arms.iter().enumerate() {
                    if i > 0 {
                        self.else_();
                    }
                    let cv = self.ex(c)?;
                    let h = self.if_head(&self.truthy(&cv));
                    self.open(&h);
                    self.block(b, tail)?;
                }
                if let Some(b) = els {
                    self.else_();
                    self.block(b, tail)?;
                }
                for _ in 0..arms.len() {
                    self.close();
                }
            }
            Stmt::While(c, bnd, body) => {
                let (b, n) = self.loop_counter(bnd)?;
                match l {
                    Lang::Rust => self.open("loop"),
                    Lang::Python => self.open("while True"),
                    Lang::Js => self.open("for (;;)"),
                    Lang::Scala => self.sc_loop_open("while (true)"),
                }
                let cv = self.ex(c)?;
                let t = self.truthy(&cv);
                self.line(match l {
                    Lang::Rust => format!("if !{t} {{ break; }}"),
                    Lang::Python => format!("if not {t}: break"),
                    Lang::Js => format!("if (!{t}) break;"),
                    Lang::Scala => format!("if (!{t}) throw BRK"),
                });
                self.loop_tick(&b, &n);
                self.loops += 1;
                if let Some(d) = self.cl.last_mut() { *d += 1; }
                let rb = self.block(body, false);
                self.loops -= 1;
                if let Some(d) = self.cl.last_mut() { *d -= 1; }
                rb?;
                if l == Lang::Scala { self.sc_loop_close() } else { self.close() }
            }
            Stmt::Loop(bnd, body) => {
                let (b, n) = self.loop_counter(bnd)?;
                match l {
                    Lang::Rust => self.open("loop"),
                    Lang::Python => self.open("while True"),
                    Lang::Js => self.open("for (;;)"),
                    Lang::Scala => self.sc_loop_open("while (true)"),
                }
                self.loop_tick(&b, &n);
                self.loops += 1;
                if let Some(d) = self.cl.last_mut() { *d += 1; }
                let rb = self.block(body, false);
                self.loops -= 1;
                if let Some(d) = self.cl.last_mut() { *d -= 1; }
                rb?;
                if l == Lang::Scala { self.sc_loop_close() } else { self.close() }
            }
            Stmt::For(p, it, body, _) => {
                let itv = self.ex(it)?;
                let x = self.fresh("x");
                match l {
                    Lang::Rust => {
                        let i = self.fresh("it");
                        self.line(format!("let mut {i} = iterate({itv})?;"));
                        self.open(&format!("while let Some({x}) = {i}.next()?"));
                    }
                    Lang::Python => self.open(&format!("for {x} in iterate({itv})")),
                    Lang::Js => self.open(&format!("for (const {x} of iterate({itv}))")),
                    Lang::Scala => {
                        let i = self.fresh("it");
                        self.line(format!("val {i} = iterate({itv})"));
                        self.sc_loop_open(&format!("while ({i}.hasNext)"));
                        self.line(format!("val {x} = {i}.next()"));
                    }
                }
                self.scopes.push(vec![]);
                match p {
                    Pat::Path(path) if path.len() == 1 && !self.is_tag(&path[0]) => {
                        let s = self.let_(&format!("l_{}", ident(&path[0])), &x);
                        self.line(s);
                        self.declare(&path[0]);
                    }
                    Pat::Wild => {}
                    _ => {
                        let ps = self.pat(p)?;
                        let mt = self.fresh("m");
                        match l {
                            Lang::Rust => {
                                self.line(format!("let mut {mt} = M::new();"));
                                self.line(format!("pmatch(&{ps}, &{x}, &mut {mt})?;"));
                            }
                            Lang::Python => {
                                self.line(format!("{mt} = {{}}"));
                                self.line(format!("pmatch({ps}, {x}, {mt})"));
                            }
                            Lang::Js => {
                                self.line(format!("const {mt} = {{}};"));
                                self.line(format!("pmatch({ps}, {x}, {mt});"));
                            }
                            Lang::Scala => {
                                self.line(format!("val {mt} = new_binds()"));
                                self.line(format!("pmatch({ps}, {x}, {mt})"));
                            }
                        }
                        let mut names = vec![];
                        self.pat_names(p, &mut names);
                        self.bind_from_map(&mt, &names);
                    }
                }
                self.loops += 1;
                if let Some(d) = self.cl.last_mut() { *d += 1; }
                let rb = self.block(body, false);
                self.loops -= 1;
                if let Some(d) = self.cl.last_mut() { *d -= 1; }
                rb?;
                self.scopes.pop();
                if l == Lang::Scala { self.sc_loop_close() } else { self.close() }
            }
            Stmt::Match(e, arms, _) => {
                let v = self.ex(e)?;
                let vt = self.fresh("mv");
                let d = self.fresh("d");
                let s = self.let_(&vt, &v);
                self.line(s);
                self.line(match l {
                    Lang::Rust => format!("let mut {d} = false;"),
                    Lang::Python => format!("{d} = False"),
                    Lang::Js => format!("let {d} = false;"),
                    Lang::Scala => format!("var {d} = false"),
                });
                for a in arms {
                    let h = self.if_head(&self.not(&d));
                    self.open(&h);
                    self.scopes.push(vec![]);
                    let ps = self.pat(&a.pat)?;
                    let mt = self.fresh("m");
                    let get = match l {
                        Lang::Rust => format!("pm(&{ps}, &{vt})?"),
                        _ => format!("pm({ps}, {vt})"),
                    };
                    self.open_some(&mt, &get);
                    let mut names = vec![];
                    self.pat_names(&a.pat, &mut names);
                    self.bind_from_map(&mt, &names);
                    let g = match &a.guard {
                        Some(g) => {
                            let gv = self.ex(g)?;
                            self.truthy(&gv)
                        }
                        None => self.tru().into(),
                    };
                    let h = self.if_head(&g);
                    self.open(&h);
                    self.line(match l {
                        Lang::Python => format!("{d} = True"),
                        Lang::Scala => format!("{d} = true"),
                        _ => format!("{d} = true;"),
                    });
                    self.block(&a.body, tail)?;
                    self.close();
                    self.close();
                    self.scopes.pop();
                    self.close();
                }
                let h = self.if_head(&self.not(&d));
                self.open(&h);
                let s = self.trap_s("no match arm");
                self.line(s);
                self.close();
            }
            Stmt::Return(e, _) => {
                let v = match e {
                    Some(e) => self.ex_stmt(e)?,
                    None => self.unit().into(),
                };
                self.ret_with(&v);
            }
            Stmt::Break(_) if self.cl.last() == Some(&0) => self.line("return Err(Flow::Brk);"),
            Stmt::Continue(_) if self.cl.last() == Some(&0) => self.line("return Err(Flow::Cont);"),
            Stmt::Break(_) if l == Lang::Scala => self.line("throw BRK"),
            Stmt::Continue(_) if l == Lang::Scala => self.line("throw CONT"),
            Stmt::Break(_) => self.line(if l == Lang::Python { "break" } else { "break;" }),
            Stmt::Continue(_) => self.line(if l == Lang::Python {
                "continue"
            } else {
                "continue;"
            }),
            Stmt::Pass => self.line(if l == Lang::Python { "pass" } else { "" }),
            Stmt::Emit(e) => {
                let v = self.ex_stmt(e)?;
                let s = self.stmt_call("emit", &v);
                self.line(s);
            }
            Stmt::Fail(e) => {
                let v = self.ex_stmt(e)?;
                self.line(match l {
                    Lang::Rust => format!("return Err(Flow::Fail({v}));"),
                    Lang::Python => format!("raise Fail({v})"),
                    Lang::Js => format!("throw new Fail({v});"),
                    Lang::Scala => format!("throw new Fail({v})"),
                });
            }
            Stmt::Send(t, v) if self.tasks_on() => self.send_s(t, v)?,
            Stmt::Spawn(e) if self.tasks_on() => {
                let h = self.spawn_e(e, "0")?;
                let s = self.discard(&h);
                self.line(s);
            }
            Stmt::Select(arms, line) if self.tasks_on() => self.select_s(arms, *line, false, tail)?,
            Stmt::Arena(r, None, body, _) if self.tasks_on() && r == "$nondet" => match body.first() {
                Some(Stmt::Select(arms, line)) => self.select_s(arms, *line, true, tail)?,
                _ => self.line(self.trap_s("bad nondet")),
            },
            Stmt::Arena(r, None, body, _) if self.tasks_on() && r == "$deadline" => self.deadline_s(body)?,
            Stmt::Arena(r, None, body, _) if self.tasks_on() && r == "$scope" => self.scope_s(body)?,
            Stmt::Arena(r, None, body, _) if self.tasks_on() && r == "$atomic" => self.atomic_s(body)?,
            Stmt::Arena(r, None, body, _) if self.tasks_on() && (r == "$priority" || r.starts_with("$spawnas:")) => {
                self.spawn_marker(r, body)?
            }
            Stmt::Send(..) | Stmt::Spawn(..) | Stmt::Select(..) => {
                let s = self.trap_s("concurrency is not available at comptime");
                self.line(s);
            }
            Stmt::Arena(r, None, body, _) if r == "$forpar" => self.forpar(body)?,
            Stmt::Arena(r, None, body, _) if r == "$budget" => self.budget06(body)?,
            Stmt::Arena(r, None, body, _) if r == "$par" => self.par06(body)?,
            Stmt::Arena(r, None, body, _) if r == "$deadline" => {
                // Comptime (the emitters' observation model): no tick passes
                // and every communication step traps first, so the body is
                // never cancelled and the else block never runs (C.6.2).
                let n = self.marker_operand(body)?;
                let s = self.discard(&n);
                self.line(s);
                if let Some(Stmt::Arena(_, _, b, _)) = body.get(1) {
                    self.open_block();
                    self.block(b, false)?;
                    self.close_block();
                }
            }
            Stmt::Arena(r, None, body, _) if r == "$nondet" => {
                // `select nondet` at comptime: the select itself traps
                self.block(body, false)?;
            }
            Stmt::Arena(r, None, body, _) if r == "$atomic" => {
                self.atomic_depth += 1;
                self.open_block();
                let res = self.block(body, false);
                self.close_block();
                self.atomic_depth -= 1;
                res?;
                if self.atomic_depth == 0 {
                    // the buffered tells need the scheduler, which comptime lacks
                    let s = self.trap_s("concurrency is not available at comptime");
                    self.line(s);
                }
            }
            Stmt::Arena(r, None, body, _) if r.starts_with("$group:") => {
                self.group06(&r["$group:".len()..], body)?
            }
            Stmt::Arena(r, None, _, _) if r == "$barrier" => {
                let s = self.trap_s("barrier outside a group");
                self.line(s);
            }
            Stmt::Arena(_, Some(b), body, _) | Stmt::Device(_, Some(b), body, _) => {
                let bv = self.ex(b)?;
                let ab = self.fresh("ab");
                let bound = self.h("arena_bound", &[bv]);
                self.line(match l {
                    Lang::Rust => format!("let {ab}: i128 = {bound};"),
                    Lang::Python => format!("{ab} = {bound}"),
                    Lang::Js => format!("const {ab} = {bound};"),
                    Lang::Scala => format!("val {ab} = {bound}"),
                });
                self.open_block();
                self.scopes.push(vec![]);
                for s2 in body {
                    self.stmt(s2, false)?;
                    let names = self.scopes.last().cloned().unwrap_or_default();
                    let vals: Vec<String> = names
                        .iter()
                        .map(|n| {
                            if l == Lang::Rust {
                                format!("&l_{}", ident(n))
                            } else {
                                format!("l_{}", ident(n))
                            }
                        })
                        .collect();
                    let a = match l {
                        Lang::Rust => format!("{ab}, &[{}]", vals.join(", ")),
                        Lang::Scala => format!("{ab}, V({})", vals.join(", ")),
                        _ => format!("{ab}, [{}]", vals.join(", ")),
                    };
                    let s = self.stmt_call("arena_check", &a);
                    self.line(s);
                }
                self.scopes.pop();
                self.close_block();
            }
            Stmt::Arena(_, _, body, _) | Stmt::Device(_, _, body, _) | Stmt::Comptime(body) => {
                self.open_block();
                self.block(body, false)?;
                self.close_block();
            }
            Stmt::Yield(e, _) => {
                let v = self.ex_stmt(e)?;
                let s = self.stmt_call("yield_", &v);
                self.line(s);
            }
            Stmt::Check(e) => {
                let v = self.ex(e)?;
                let s = self.stmt_call("check_", &v);
                self.line(s);
            }
        }
        Ok(())
    }
}
