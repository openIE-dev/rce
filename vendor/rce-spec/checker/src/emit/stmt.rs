//! Statements: target control flow, with places rebuilt the way the
//! reference evaluator's `assign` rebuilds them (values never change in place).
use super::Lang;
use super::base::*;
use crate::ast::*;

impl<'a> G<'a> {
    /// Evaluate `e` as a statement-level expression, writing back receivers
    /// and `&mut` arguments; returns an expression for the value.
    pub fn ex_stmt(&mut self, e: &Expr) -> E {
        let Expr::Call(f, args, _) = e else {
            return self.ex(e);
        };
        let Some((call, wb)) = self.call_expr(f, args, true)? else {
            return self.ex(e);
        };
        let Some(wb) = wb else { return Ok(call) };
        let l = self.lang;
        let t = wb.tmp.clone();
        self.line(match l {
            Lang::Rust => format!("let {t} = {call};"),
            Lang::Python => format!("{t} = {call}"),
            Lang::Js => format!("const {t} = {call};"),
        });
        // Arguments first, then the receiver (interp `call_fn`, then `builtin`).
        for (i, a) in wb.args.iter().enumerate() {
            let get = match l {
                Lang::Rust => format!("{t}.outs.get({i}).cloned().flatten()"),
                Lang::Python => format!("({t}[1][{i}] if len({t}[1]) > {i} else None)"),
                Lang::Js => format!("({t}[1].length > {i} ? {t}[1][{i}] : null)"),
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
        }
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
        if self.lang != Lang::Python {
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
        }
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
                }
                let cv = self.ex(c)?;
                let t = self.truthy(&cv);
                self.line(match l {
                    Lang::Rust => format!("if !{t} {{ break; }}"),
                    Lang::Python => format!("if not {t}: break"),
                    Lang::Js => format!("if (!{t}) break;"),
                });
                self.loop_tick(&b, &n);
                self.block(body, false)?;
                self.close();
            }
            Stmt::Loop(bnd, body) => {
                let (b, n) = self.loop_counter(bnd)?;
                match l {
                    Lang::Rust => self.open("loop"),
                    Lang::Python => self.open("while True"),
                    Lang::Js => self.open("for (;;)"),
                }
                self.loop_tick(&b, &n);
                self.block(body, false)?;
                self.close();
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
                        }
                        let mut names = vec![];
                        self.pat_names(p, &mut names);
                        self.bind_from_map(&mt, &names);
                    }
                }
                self.block(body, false)?;
                self.scopes.pop();
                self.close();
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
                let s = match &self.ctx {
                    Ctx::Fn { .. } => self.ret_pack(&v),
                    Ctx::Gen => match l {
                        Lang::Rust => format!("{{ let _ = {v}; return Ok(()); }}"),
                        Lang::Python => format!("{v}\nreturn"),
                        Lang::Js => format!("{v}; return;"),
                    },
                    Ctx::Step => match l {
                        Lang::Rust => format!("return Err(Flow::Ret({v}));"),
                        Lang::Python => format!("raise Ret({v})"),
                        Lang::Js => format!("throw new Ret({v});"),
                    },
                };
                self.line(s);
            }
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
                });
            }
            Stmt::Send(..) | Stmt::Spawn(..) | Stmt::Select(..) => {
                let s = self.trap_s("concurrency is not available at comptime");
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
