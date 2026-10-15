//! Emission of the draft 0.6 / 0.6b statements (Appendix C, book/14-draft06-concurrency.md)
//! under the emitters' observation model, which is the reference
//! evaluator's comptime mode (`emit::observe`):
//!
//! * `with budget n`: every Form operation charges its C.6.4 cost through
//!   the runtime's `charge`, at the same point as interp `charge`, so the
//!   `trap("budget")` falls on the same operation on every target.
//! * `par`: branches in source order (C.4, T16), after the fork cost. A `par`
//!   whose branches wait on lvars cannot run in source order (a branch may
//!   wait for a later one), so it is run by restart: each pending branch is
//!   attempted; a wait that is not yet satisfied abandons the attempt,
//!   restoring the branch's own writes and the budget counters, and the
//!   branch is tried again after the others. Puts are joins, hence
//!   idempotent, so a repeated attempt puts nothing new (T29). When a whole
//!   round makes no progress the waiting branches trap as in the evaluator.
//! * `with group g lanes n`: phases between barriers run lane after lane,
//!   with the reduction targets reset per lane and combined by the fixed
//!   tree, and top-level declarations shared (lane 0 runs them), as in
//!   interp `group_exec`.
use super::Lang;
use super::base::*;
use crate::ast::*;

impl<'a> G<'a> {
    /// the operand of a marker region (budget, deadline, lanes), as an integer check
    pub(super) fn marker_operand(&mut self, body: &[Stmt]) -> Result<String, String> {
        let Some(Stmt::Expr(e)) = body.first() else {
            return Err("missing operand".into());
        };
        let v = self.ex(e)?;
        Ok(match self.lang {
            Lang::Rust => format!("marker_int(&{v})?"),
            Lang::Python => format!("marker_int({v})"),
            Lang::Js => format!("markerInt({v})"),
            Lang::Scala => format!("marker_int({v})"),
        })
    }

    pub(super) fn budget06(&mut self, body: &[Stmt]) -> Result<(), String> {
        let Some(Stmt::Expr(e)) = body.first() else {
            return Err("missing operand".into());
        };
        let n = self.ex(e)?;
        let (rest, eb) = match body.last() {
            Some(Stmt::If(arms, _))
                if arms.len() == 1 && matches!(&arms[0].0, Expr::Bool(false, _)) =>
            {
                (&body[1..body.len() - 1], Some(&arms[0].1[..]))
            }
            _ => (&body[1..], None),
        };
        let l = self.lang;
        let Some(eb) = eb else {
            match l {
                Lang::Rust => {
                    self.open_block();
                    self.line(format!("let __bg = budget_enter({n})?;"));
                    self.block(rest, false)?;
                    self.close_block();
                }
                Lang::Python => {
                    self.open(&format!("with budget({n})"));
                    self.block(rest, false)?;
                    self.close();
                }
                Lang::Js => {
                    self.line(format!("budgetEnter({n});"));
                    self.open("try");
                    self.block(rest, false)?;
                    self.close();
                    self.open("finally");
                    self.line("budgetExit();");
                    self.close();
                }
                Lang::Scala => {
                    self.line(format!("budget_enter({n})"));
                    self.open("try");
                    self.block(rest, false)?;
                    self.ind -= 1;
                    self.line("} finally { budget_exit() }");
                }
            }
            return Ok(());
        };
        // with an else: the block is atomic (C.5). Every local the block
        // writes (state fields and `self` are locals of the emitted step) is
        // saved at entry and put back when the budget traps, then the else
        // runs with __bn (limit) and __bu (units spent) for `as e`.
        let roots: Vec<String> = written_roots(rest).into_iter().filter(|x| self.local(x)).collect();
        let snaps: Vec<(String, String)> = roots
            .iter()
            .map(|x| (format!("l_{}", ident(x)), self.fresh("bsn")))
            .collect();
        if l != Lang::Python {
            self.open_block();
        }
        // the limit is kept under a fresh name (Python has no block scope,
        // so a nested budget must not overwrite it) and bound to __bn for
        // the else only
        let bn = self.fresh("bn");
        let s = self.let_(&bn, &n);
        self.line(s);
        for (v, sn) in &snaps {
            let e = if l == Lang::Rust { format!("{v}.clone()") } else { v.clone() };
            let s = self.let_(sn, &e);
            self.line(s);
        }
        let restore: Vec<String> = snaps.iter().map(|(v, sn)| self.set_(v, sn)).collect();
        match l {
            Lang::Rust => {
                self.line("let __r: R<()> = (|| -> R<()> {");
                self.ind += 1;
                self.line(format!("let __bg = budget_enter({bn}.clone())?;"));
                self.block(rest, false)?;
                self.line("Ok(())");
                self.ind -= 1;
                self.line("})();");
                self.open("match __r");
                self.open("Err(Flow::Trap(t)) if t == \"budget\" =>");
                for r in &restore {
                    self.line(r);
                }
                self.line(format!("let __bn = {bn}.clone(); let __bu = budget_last();"));
                self.block(eb, false)?;
                self.close();
                self.line("r => r?,");
                self.close();
                self.close_block();
            }
            Lang::Python => {
                self.open("try");
                self.open(&format!("with budget({bn})"));
                self.block(rest, false)?;
                self.close();
                self.close();
                self.open("except Trap as __t");
                self.open("if __t.m != 'budget'");
                self.line("raise");
                self.close();
                for r in &restore {
                    self.line(r);
                }
                self.line(format!("__bn = {bn}; __bu = budget_last()"));
                self.block(eb, false)?;
                self.close();
            }
            Lang::Js => {
                self.open("try");
                self.line(format!("budgetEnter({bn});"));
                self.open("try");
                self.block(rest, false)?;
                self.close();
                self.open("finally");
                self.line("budgetExit();");
                self.close();
                self.close();
                self.open("catch (__t)");
                self.open("if (!(__t instanceof Trap) || __t.m !== 'budget')");
                self.line("throw __t;");
                self.close();
                for r in &restore {
                    self.line(r);
                }
                self.line(format!("const __bn = {bn}; const __bu = budgetLast();"));
                self.block(eb, false)?;
                self.close();
                self.close_block();
            }
            Lang::Scala => {
                self.open("try");
                self.line(format!("budget_enter({bn})"));
                self.open("try");
                self.block(rest, false)?;
                self.ind -= 1;
                self.line("} finally { budget_exit() }");
                self.ind -= 1;
                self.line("} catch { case __t: Trap if __t.m == \"budget\" =>");
                self.ind += 1;
                for r in &restore {
                    self.line(r);
                }
                self.line(format!("val __bn: Any = {bn}; val __bu: Any = budget_last()"));
                self.block(eb, false)?;
                self.close();
                self.close_block();
            }
        }
        Ok(())
    }

    pub(super) fn par06(&mut self, branches: &[Stmt]) -> Result<(), String> {
        let bodies: Vec<&[Stmt]> = branches
            .iter()
            .filter_map(|b| match b {
                Stmt::Arena(_, _, x, _) => Some(&x[..]),
                _ => None,
            })
            .collect();
        let k = bodies.len();
        let fork = 16 * k;
        let l = self.lang;
        let s = self.stmt_call("charge", &format!("{fork}{}", if l == Lang::Js { "n" } else { "" }));
        if l == Lang::Scala {
            self.line(s);
            let waits = bodies.iter().any(|b| has_wait(b));
            return self.par06_seq(&bodies, waits);
        }
        self.line(s);
        let waits = bodies.iter().any(|b| has_wait(b));
        if matches!(l, Lang::Rust | Lang::Python) && k > 1 {
            return self.par_threads(&bodies);
        }
        if l == Lang::Js && k > 1 {
            return self.par_workers_js(&bodies, waits);
        }
        self.par06_seq(&bodies, waits)
    }

    /// the single-thread schedule: branches in source order, or by restart
    /// when they wait on lvars
    fn par06_seq(&mut self, bodies: &[&[Stmt]], waits: bool) -> Result<(), String> {
        let k = bodies.len();
        let l = self.lang;
        if !waits {
            for b in bodies {
                self.open_block();
                self.block(b, false)?;
                self.close_block();
            }
            return Ok(());
        }
        let (pd, pr, prog) = (self.fresh("pd"), self.fresh("pr"), self.fresh("prog"));
        if l == Lang::Scala {
            return self.par06_seq_scala(bodies, &pd, &pr, &prog);
        }
        self.open_block();
        match l {
            Lang::Rust => {
                self.line(format!("let mut {pd} = vec![false; {k}];"));
                self.line(format!("let mut {pr}: Vec<Option<Flow>> = vec![None; {k}];"));
                self.line("par_enter();");
                self.open("loop");
                self.line(format!("let mut {prog} = false;"));
                self.line("let __pc = puts();");
            }
            Lang::Python => {
                self.line(format!("{pd} = [False] * {k}"));
                self.line(format!("{pr} = [None] * {k}"));
                self.line("par_enter()");
                self.open("try");
                self.open("while True");
                self.line(format!("{prog} = False"));
                self.line("__pc = puts()");
            }
            Lang::Js | Lang::Scala => {
                self.line(format!("const {pd} = Array({k}).fill(false);"));
                self.line(format!("const {pr} = Array({k}).fill(null);"));
                self.line("parEnter();");
                self.open("try");
                self.open("for (;;)");
                self.line(format!("let {prog} = false;"));
                self.line("const __pc = puts();");
            }
        }
        for (i, b) in bodies.iter().enumerate() {
            let roots: Vec<String> = written_roots(b)
                .into_iter()
                .filter(|n| self.local(n))
                .collect();
            let snaps: Vec<(String, String)> = roots
                .iter()
                .map(|n| (format!("l_{}", ident(n)), self.fresh("sn")))
                .collect();
            let bs = self.fresh("bs");
            let head = match l {
                Lang::Js | Lang::Scala => format!("if (!{pd}[{i}])"),
                Lang::Python => format!("if not {pd}[{i}]"),
                Lang::Rust => format!("if !{pd}[{i}]"),
            };
            self.open(&head);
            for (v, sn) in &snaps {
                let e = if l == Lang::Rust { format!("{v}.clone()") } else { v.clone() };
                let s = self.let_(sn, &e);
                self.line(s);
            }
            let s = self.let_(&bs, &format!("{}()", self.rt("par_snap")));
            self.line(s);
            let restore: Vec<String> = snaps
                .iter()
                .map(|(v, sn)| self.set_(v, sn))
                .chain(std::iter::once(match l {
                    Lang::Python => format!("par_restore({bs})"),
                    Lang::Rust => format!("par_restore({bs});"),
                    Lang::Js | Lang::Scala => format!("parRestore({bs});"),
                }))
                .collect();
            match l {
                Lang::Rust => {
                    self.line("let __r: R<()> = (|| -> R<()> {");
                    self.ind += 1;
                    self.block(b, false)?;
                    self.line("Ok(())");
                    self.ind -= 1;
                    self.line("})();");
                    self.open("match __r");
                    self.open("Err(Flow::Trap(t)) if t == \"$block\" =>");
                    for r in &restore {
                        self.line(r);
                    }
                    self.close();
                    self.line(format!(
                        "r => {{ {pd}[{i}] = true; {prog} = true; if let Err(f) = r {{ {pr}[{i}] = Some(f); }} }}"
                    ));
                    self.close();
                }
                Lang::Python => {
                    self.open("try");
                    self.block(b, false)?;
                    self.line(format!("{pd}[{i}] = True; {prog} = True"));
                    self.close();
                    self.open("except (Trap, Fail) as __e");
                    self.open("if type(__e) is Trap and __e.m == \"$block\"");
                    for r in &restore {
                        self.line(r);
                    }
                    self.close();
                    self.open("else");
                    self.line(format!("{pd}[{i}] = True; {prog} = True; {pr}[{i}] = __e"));
                    self.close();
                    self.close();
                }
                Lang::Js | Lang::Scala => {
                    self.open("try");
                    self.block(b, false)?;
                    self.line(format!("{pd}[{i}] = true; {prog} = true;"));
                    self.close();
                    self.open("catch (__e)");
                    self.line("if (!isFlow(__e)) throw __e;");
                    self.open("if (__e instanceof Trap && __e.m === \"$block\")");
                    for r in &restore {
                        self.line(r);
                    }
                    self.close();
                    self.line(format!("else {{ {pd}[{i}] = true; {prog} = true; {pr}[{i}] = __e; }}"));
                    self.close();
                }
            }
            self.close();
        }
        let stuck = "lvar wait cannot be satisfied";
        match l {
            Lang::Rust => {
                self.line(format!("if {pd}.iter().all(|d| *d) {{ break; }}"));
                self.line(format!(
                    "if !{prog} && puts() == __pc {{ for j in 0..{k} {{ if !{pd}[j] {{ {pr}[j] = Some(Flow::Trap(\"{stuck}\".to_string())); }} }} break; }}"
                ));
                self.close();
                self.line("par_exit();");
                self.line(format!("for f in {pr} {{ if let Some(f) = f {{ return Err(f); }} }}"));
            }
            Lang::Python => {
                self.line(format!("if all({pd}): break"));
                self.open(&format!("if not {prog} and puts() == __pc"));
                self.line(format!("{pr} = [Trap(\"{stuck}\") if not d else e for d, e in zip({pd}, {pr})]"));
                self.line("break");
                self.close();
                self.close();
                self.close();
                self.open("finally");
                self.line("par_exit()");
                self.close();
                self.open(&format!("for __e in {pr}"));
                self.line("if __e is not None: raise __e");
                self.close();
            }
            Lang::Js => {
                self.line(format!("if ({pd}.every((d) => d)) break;"));
                self.line(format!(
                    "if (!{prog} && parStuck(__pc)) {{ for (let j = 0; j < {k}; j++) if (!{pd}[j]) {pr}[j] = new Trap(\"{stuck}\"); break; }}"
                ));
                self.close();
                self.close();
                self.open("finally");
                self.line("parExit();");
                self.close();
                self.line(format!("for (const __e of {pr}) if (__e !== null) throw __e;"));
            }
            Lang::Scala => unreachable!(),
        }
        self.close_block();
        Ok(())
    }

    /// Scala: the restart schedule, with a flag in place of `break`
    fn par06_seq_scala(&mut self, bodies: &[&[Stmt]], pd: &str, pr: &str, prog: &str) -> Result<(), String> {
        let k = bodies.len();
        let go = self.fresh("go");
        self.open_block();
        self.line(format!("val {pd} = Array.fill({k})(false)"));
        self.line(format!("val {pr} = Array.fill[Throwable]({k})(null)"));
        self.line(format!("var {go} = true"));
        self.line("par_enter()");
        self.open("try");
        self.open(&format!("while ({go})"));
        self.line(format!("var {prog} = false"));
        self.line("val __pc = puts()");
        for (i, b) in bodies.iter().enumerate() {
            let roots: Vec<String> = written_roots(b).into_iter().filter(|n| self.local(n)).collect();
            let snaps: Vec<(String, String)> = roots.iter().map(|n| (format!("l_{}", ident(n)), self.fresh("sn"))).collect();
            let bs = self.fresh("bs");
            self.open(&format!("if (!{pd}({i}))"));
            for (v, sn) in &snaps {
                self.line(format!("val {sn} = {v}"));
            }
            self.line(format!("val {bs} = par_snap()"));
            self.open("try");
            self.block(b, false)?;
            self.line(format!("{pd}({i}) = true; {prog} = true"));
            self.ind -= 1;
            self.line("} catch {");
            self.ind += 1;
            let mut rs: Vec<String> = snaps.iter().map(|(v, sn)| format!("{v} = {sn}")).collect();
            rs.push(format!("par_restore({bs})"));
            self.line(format!("case __e: Trap if __e.m == \"$block\" => {}", rs.join("; ")));
            self.line(format!("case __e: Flow if __e.isInstanceOf[Trap] || __e.isInstanceOf[Fail] => {pd}({i}) = true; {prog} = true; {pr}({i}) = __e"));
            self.close();
            self.close();
        }
        let stuck = "lvar wait cannot be satisfied";
        self.line(format!("if ({pd}.forall(d => d)) {go} = false"));
        self.line(format!(
            "else if (!{prog} && par_stuck(__pc)) {{ for (j <- 0 until {k}) if (!{pd}(j)) {pr}(j) = new Trap(\"{stuck}\"); {go} = false }}"
        ));
        self.close();
        self.ind -= 1;
        self.line("} finally { par_exit() }");
        self.line(format!("for (__e <- {pr}) if (__e != null) throw __e"));
        self.close_block();
        Ok(())
    }

    /// emit a top-level JavaScript function (for a worker) whose body sees
    /// the current locals, bound from its first argument
    fn hoist_js(&mut self, name: &str, locals: &[String], pre: &[String], body: impl FnOnce(&mut Self) -> Result<String, String>) -> Result<(), String> {
        let saved = (std::mem::take(&mut self.out), self.ind, std::mem::take(&mut self.scopes), self.loops, std::mem::take(&mut self.cl));
        self.ind = 0;
        self.scopes = vec![locals.to_vec()];
        self.loops = 0;
        self.open(&format!("function {name}(__env, __x)"));
        let names: Vec<String> = locals.iter().map(|n| format!("l_{}", ident(n))).collect();
        self.line(format!("let [{}] = __env;", names.join(", ")));
        for p in pre {
            self.line(p);
        }
        let r = body(self);
        let r = r.map(|ret| {
            self.line(ret);
            self.close();
        });
        let code = std::mem::replace(&mut self.out, saved.0);
        self.ind = saved.1;
        self.scopes = saved.2;
        self.loops = saved.3;
        self.cl = saved.4;
        r?;
        self.hoist.push(String::new());
        self.hoist.extend(code);
        self.hoist_names.push(name.to_string());
        Ok(())
    }

    /// JavaScript: every branch on its own worker (see rt.js), unless a
    /// local cannot be copied to a worker (closure, iterator), in which case
    /// the single-thread schedule runs
    fn par_workers_js(&mut self, bodies: &[&[Stmt]], waits: bool) -> Result<(), String> {
        let k = bodies.len();
        let locals = self.all_locals();
        let mut roots: Vec<Vec<String>> = vec![];
        for b in bodies {
            roots.push(written_roots(b).into_iter().filter(|n| self.local(n)).collect());
        }
        let env = self.fresh("pe");
        let ws = self.fresh("ws");
        self.open_block();
        let lv: Vec<String> = locals.iter().map(|n| format!("l_{}", ident(n))).collect();
        self.line(format!("const {env} = [{}];", lv.join(", ")));
        self.open(&format!("if (parCanWorker({env}))"));
        let mut names = vec![];
        for (i, b) in bodies.iter().enumerate() {
            let f = format!("__pb{}", self.fresh("b"));
            let ret = roots[i].iter().map(|n| format!("l_{}", ident(n))).collect::<Vec<_>>().join(", ");
            let b = *b;
            self.hoist_js(&f, &locals, &[], |g| {
                g.scopes.push(vec![]);
                let r = g.block(b, false);
                g.scopes.pop();
                r.map(|_| format!("return [{ret}];"))
            })?;
            names.push(format!("\"{f}\""));
        }
        self.line(format!("const {ws} = parResolve(parWorkers([{}], {env}, Array({k}).fill(null)));", names.join(", ")));
        for (i, rs) in roots.iter().enumerate() {
            for (j, n) in rs.iter().enumerate() {
                let id = ident(n);
                self.line(format!("if (!eq({ws}[{i}][{j}], {env}[{}])) l_{id} = merge3({env}[{}], l_{id}, {ws}[{i}][{j}]);",
                    locals.iter().position(|x| x == n).unwrap(), locals.iter().position(|x| x == n).unwrap()));
            }
        }
        self.close();
        self.open("else");
        self.par06_seq(bodies, waits)?;
        self.close();
        self.close_block();
        Ok(())
    }

    /// Rust: every branch on its own scoped OS thread, on a copy of the
    /// store and the budget counters, sharing lvars (Arc) and the wait
    /// accounting of interp `ParCtx`. Results resolve in source order exactly
    /// as interp `par_exec` in Threads mode: the first failing branch decides,
    /// budget exhaustion is placed where the sequential run would hit it,
    /// and the disjoint writes are combined by the three-way merge (C.3).
    fn par_threads(&mut self, bodies: &[&[Stmt]]) -> Result<(), String> {
        if self.lang == Lang::Python {
            return self.par_threads_py(bodies);
        }
        let k = bodies.len();
        let locals = self.all_locals();
        let mut roots: Vec<Vec<String>> = vec![];
        for b in bodies {
            roots.push(written_roots(b).into_iter().filter(|n| self.local(n)).collect());
        }
        let mut all_roots: Vec<String> = roots.iter().flatten().cloned().collect();
        all_roots.sort();
        all_roots.dedup();
        self.open_block();
        for n in &all_roots {
            self.line(format!("let __base_{0} = l_{0}.clone();", ident(n)));
        }
        self.line(format!("let __pcx = par_cx_new({k});"));
        self.line("let __bud = bud_snap();");
        self.line("let __outs: Vec<(R<()>, (i128, u64), Vec<V>)> = std::thread::scope(|__s| {");
        self.ind += 1;
        for (i, b) in bodies.iter().enumerate() {
            self.line(format!("let __h{i} = {{"));
            self.ind += 1;
            for n in &locals {
                self.line(format!("#[allow(unused_mut, unused_variables)] let mut l_{0} = l_{0}.clone();", ident(n)));
            }
            self.line("let (__pcx, __bud) = (__pcx.clone(), __bud.clone());");
            self.line("std::thread::Builder::new().stack_size(256 << 20).spawn_scoped(__s, move || {");
            self.ind += 1;
            self.line("let (__r, __c) = par_branch(__pcx, __bud, || -> R<()> {");
            self.ind += 1;
            self.scopes.push(vec![]);
            let res = self.block(b, false);
            self.scopes.pop();
            res?;
            self.line("Ok(())");
            self.ind -= 1;
            self.line("});");
            let ws: Vec<String> = roots[i].iter().map(|n| format!("l_{}", ident(n))).collect();
            self.line(format!("(__r, __c, vec![{}])", ws.join(", ")));
            self.ind -= 1;
            self.line("}).expect(\"par thread\")");
            self.ind -= 1;
            self.line("};");
        }
        let joins: Vec<String> = (0..k).map(|i| format!("__h{i}.join().expect(\"par branch panicked\")")).collect();
        self.line(format!("vec![{}]", joins.join(", ")));
        self.ind -= 1;
        self.line("});");
        self.line("let __rem = bud_rem();");
        self.line("let (__f0, mut __fcum) = (__pcx.fuel0, 0u64);");
        self.line("let mut __cum: i128 = 0;");
        self.line("let mut __ws: Vec<Vec<V>> = vec![];");
        self.open("for (__r, __c, __w) in __outs");
        self.line("par_fuel(&mut __fcum, __f0, __c.1)?;");
        self.line("if let Some(__rm) = __rem { if matches!(&__r, Err(Flow::Trap(t)) if t == \"budget\") || __cum + __c.0 > __rm { Err(Flow::Trap(\"budget\".to_string()))?; } }");
        self.line("__r?;");
        self.line("__cum += __c.0;");
        self.line("__ws.push(__w);");
        self.close();
        for (i, rs) in roots.iter().enumerate() {
            for (j, n) in rs.iter().enumerate() {
                let id = ident(n);
                self.line(format!(
                    "if !eq(&__ws[{i}][{j}], &__base_{id}) {{ l_{id} = merge3(&__base_{id}, &l_{id}, __ws[{i}][{j}].clone()); }}"
                ));
            }
        }
        self.line("bud_add(__cum);");
        self.close_block();
        Ok(())
    }

    /// Python: the same on `threading` threads (parallel under the
    /// free-threaded CPython). Each branch is a nested function whose
    /// written roots are parameters defaulted to the fork-point values.
    fn par_threads_py(&mut self, bodies: &[&[Stmt]]) -> Result<(), String> {
        let k = bodies.len();
        let mut roots: Vec<Vec<String>> = vec![];
        for b in bodies {
            roots.push(written_roots(b).into_iter().filter(|n| self.local(n)).collect());
        }
        let mut all_roots: Vec<String> = roots.iter().flatten().cloned().collect();
        all_roots.sort();
        all_roots.dedup();
        for n in &all_roots {
            self.line(format!("__base_{0} = l_{0}", ident(n)));
        }
        let mut names = vec![];
        for (i, b) in bodies.iter().enumerate() {
            let f = self.fresh("br");
            let ps: Vec<String> = roots[i].iter().map(|n| format!("l_{0}=l_{0}", ident(n))).collect();
            self.open(&format!("def {f}({})", ps.join(", ")));
            self.scopes.push(vec![]);
            let res = self.block(b, false);
            self.scopes.pop();
            res?;
            let ws: Vec<String> = roots[i].iter().map(|n| format!("l_{}", ident(n))).collect();
            self.line(format!("return [{}]", ws.join(", ")));
            self.close();
            names.push(f);
        }
        let ws = self.fresh("ws");
        self.line(format!("{ws} = par_resolve(par_threads({k}, [{}]))", names.join(", ")));
        for (i, rs) in roots.iter().enumerate() {
            for (j, n) in rs.iter().enumerate() {
                let id = ident(n);
                self.line(format!(
                    "if not eq({ws}[{i}][{j}], __base_{id}): l_{id} = merge3(__base_{id}, l_{id}, {ws}[{i}][{j}])"
                ));
            }
        }
        Ok(())
    }

    /// `for ... par` on threads: the items are split into contiguous chunks,
    /// one scoped thread per chunk on a copy of the store and the budget
    /// counters. Each iteration starts its reduction targets at the identity
    /// and records its contribution; after the join the results resolve in
    /// chunk order as in `par` (first failure decides, budget exhaustion
    /// placed where the sequential run would hit it), the contributions are
    /// concatenated in iteration order and combined by the same fixed tree
    /// T as the sequential loop, and other writes (slot writes) are merged
    /// against the fork point. Iterations are independent (W-PAR-1), so the
    /// result does not depend on the chunking.
    pub(super) fn forpar_threads(&mut self, f: &Stmt, targets: &[(String, String)]) -> Result<(), String> {
        let Stmt::For(p, it, b, line) = f else { return Err("bad for par".into()) };
        let l = self.lang;
        let tnames: Vec<String> = targets.iter().map(|(n, _)| n.clone()).collect();
        let roots: Vec<String> = written_roots(b)
            .into_iter()
            .filter(|n| self.local(n) && !tnames.contains(n))
            .collect();
        self.open_block();
        let itv = self.ex(it)?;
        let items = self.fresh("items");
        let nch = self.fresh("nch");
        match l {
            Lang::Rust => {
                self.line(format!("let {items} = iter_items({itv})?;"));
                self.line(format!("if {items}.len() as i128 > LOOP_LIMIT {{ Err(Flow::Trap(\"loop bound\".to_string()))?; }}"));
                self.line(format!("let {nch} = par_chunks({items}.len());"));
            }
            Lang::Python => {
                self.line(format!("{items} = list(iter_items({itv}))"));
                self.line(format!("if len({items}) > LOOP_LIMIT: trap(\"loop bound\")"));
                self.line(format!("{nch} = par_chunks(len({items}))"));
            }
            Lang::Js | Lang::Scala => {
                self.line(format!("const {items} = [...iterItems({itv})];"));
                self.line(format!("if ({items}.length > LOOP_LIMIT) trap(\"loop bound\");"));
                self.line(format!("const {nch} = parChunks({items}.length);"));
            }
        }
        let mut tv = vec![];
        for (n, op) in targets {
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
            tv.push((n.clone(), sym, bk, ik, ck));
        }
        // the loop over one chunk, with the reset and collection hooks
        let chunk_name = format!("$fpchunk{}", self.fresh("c"));
        let mut nb = vec![];
        for (n, _, _, ik, _) in &tv {
            nb.push(Stmt::Expr(Expr::Call(
                Box::new(Expr::Name("$par_reset".into(), 0)),
                vec![Arg { label: None, e: Expr::Name(n.clone(), 0) }, Arg { label: None, e: Expr::Name(ik.clone(), 0) }],
                0,
                false,
            )));
        }
        nb.extend(b.iter().cloned());
        for (n, _, _, _, ck) in &tv {
            nb.push(Stmt::Expr(Expr::Call(
                Box::new(Expr::Name("$par_collect".into(), 0)),
                vec![Arg { label: None, e: Expr::Name(ck.clone(), 0) }, Arg { label: None, e: Expr::Name(n.clone(), 0) }],
                0,
                false,
            )));
        }
        let lp = Stmt::For(p.clone(), Expr::Name(chunk_name.clone(), 0), nb, *line);
        let cvar = format!("l_{}", ident(&chunk_name));
        let cks: Vec<String> = tv.iter().map(|t| t.4.clone()).collect();
        let ws = self.fresh("ws");
        match l {
            Lang::Rust => {
                for n in &roots {
                    self.line(format!("let __base_{0} = l_{0}.clone();", ident(n)));
                }
                let locals = self.all_locals();
                self.line(format!("let __pcx = par_cx_new({nch});"));
                self.line("let __bud = bud_snap();");
                self.line("let __outs: Vec<(R<()>, (i128, u64), Vec<V>, Vec<Vec<V>>)> = std::thread::scope(|__s| {");
                self.ind += 1;
                self.line("let mut __hs = vec![];");
                self.open(&format!("for __ch in par_split(&{items}, {nch})"));
                for n in &locals {
                    self.line(format!("#[allow(unused_mut, unused_variables)] let mut l_{0} = l_{0}.clone();", ident(n)));
                }
                for (_, _, _, ik, _) in &tv {
                    self.line(format!("let {ik} = {ik}.clone();"));
                }
                self.line("let (__pcx, __bud) = (__pcx.clone(), __bud.clone());");
                self.line("__hs.push(std::thread::Builder::new().stack_size(256 << 20).spawn_scoped(__s, move || {");
                self.ind += 1;
                self.line(format!("#[allow(unused_mut)] let mut {cvar} = V::List(__ch);"));
                for ck in &cks {
                    self.line(format!("let mut {ck}: Vec<V> = vec![];"));
                }
                self.line("let (__r, __c) = par_branch(__pcx, __bud, || -> R<()> {");
                self.ind += 1;
                self.scopes.push(vec![chunk_name.clone()]);
                self.par_hooks = true;
                let res = self.stmt(&lp, false);
                self.par_hooks = false;
                self.scopes.pop();
                res?;
                self.line("Ok(())");
                self.ind -= 1;
                self.line("});");
                let wv: Vec<String> = roots.iter().map(|n| format!("l_{}", ident(n))).collect();
                self.line(format!("(__r, __c, vec![{}], vec![{}])", wv.join(", "), cks.join(", ")));
                self.ind -= 1;
                self.line("}).expect(\"par thread\"));");
                self.close();
                self.line("__hs.into_iter().map(|h| h.join().expect(\"par chunk panicked\")).collect()");
                self.ind -= 1;
                self.line("});");
                self.line("let __rem = bud_rem();");
        self.line("let (__f0, mut __fcum) = (__pcx.fuel0, 0u64);");
                self.line("let mut __cum: i128 = 0;");
                self.line(format!("let mut {ws}: Vec<Vec<V>> = vec![];"));
                for ck in &cks {
                    self.line(format!("let mut {ck}: Vec<V> = vec![];"));
                }
                self.open("for (__r, __c, __w, __k) in __outs");
                self.line("par_fuel(&mut __fcum, __f0, __c.1)?;");
        self.line("if let Some(__rm) = __rem { if matches!(&__r, Err(Flow::Trap(t)) if t == \"budget\") || __cum + __c.0 > __rm { Err(Flow::Trap(\"budget\".to_string()))?; } }");
                self.line("__r?;");
                self.line("__cum += __c.0;");
                self.line(format!("{ws}.push(__w);"));
                if !cks.is_empty() {
                    self.line("let mut __k = __k.into_iter();");
                    for ck in &cks {
                        self.line(format!("{ck}.extend(__k.next().unwrap());"));
                    }
                } else {
                    self.line("let _ = __k;");
                }
                self.close();
                self.open(&format!("for __w in &{ws}"));
                for (j, n) in roots.iter().enumerate() {
                    let id = ident(n);
                    self.line(format!("if !eq(&__w[{j}], &__base_{id}) {{ l_{id} = merge3(&__base_{id}, &l_{id}, __w[{j}].clone()); }}"));
                }
                self.close();
                self.line("bud_add(__cum);");
            }
            Lang::Js | Lang::Scala => {
                let locals = self.all_locals();
                let env = self.fresh("pe");
                let lv: Vec<String> = locals.iter().map(|n| format!("l_{}", ident(n))).collect();
                self.line(format!("const {env} = [{}];", lv.join(", ")));
                for ck in &cks {
                    self.line(format!("let {ck} = [];"));
                }
                let iks: Vec<String> = tv.iter().map(|t| t.3.clone()).collect();
                self.open(&format!("if ({nch} > 1 && parCanWorker({env}))"));
                let f = format!("__fp{}", self.fresh("f"));
                let mut pre = vec![format!("let {cvar} = __x[0];"), format!("const [{}] = __x[1];", iks.join(", "))];
                for ck in &cks {
                    pre.push(format!("let {ck} = [];"));
                }
                let wv: Vec<String> = roots.iter().map(|n| format!("l_{}", ident(n))).collect();
                let ret = format!("return [[{}], [{}]];", wv.join(", "), cks.join(", "));
                let cn = chunk_name.clone();
                self.hoist_js(&f, &locals, &pre, |g| {
                    g.scopes.push(vec![cn]);
                    g.par_hooks = true;
                    let r = g.stmt(&lp, false);
                    g.par_hooks = false;
                    g.scopes.pop();
                    r.map(|_| ret)
                })?;
                self.line(format!(
                    "const {ws} = parResolve(parWorkers(Array({nch}).fill(\"{f}\"), {env}, parSplit({items}, {nch}).map((c) => [new Lst(c), [{}]])));",
                    iks.join(", ")
                ));
                for (j, ck) in cks.iter().enumerate() {
                    self.line(format!("{ck} = {ws}.flatMap((w) => w[1][{j}]);"));
                }
                for (j, n) in roots.iter().enumerate() {
                    let id = ident(n);
                    let pos = locals.iter().position(|x| x == n).unwrap();
                    self.line(format!("for (const w of {ws}) if (!eq(w[0][{j}], {env}[{pos}])) l_{id} = merge3({env}[{pos}], l_{id}, w[0][{j}]);"));
                }
                self.close();
                self.open("else");
                self.line(format!("let {cvar} = new Lst({items});"));
                self.scopes.push(vec![chunk_name.clone()]);
                self.par_hooks = true;
                let r = self.stmt(&lp, false);
                self.par_hooks = false;
                self.scopes.pop();
                r?;
                self.close();
            }
            Lang::Python => {
                for n in &roots {
                    self.line(format!("__base_{0} = l_{0}", ident(n)));
                }
                let br = self.fresh("br");
                let mut ps: Vec<String> = vec![cvar.clone()];
                ps.extend(roots.iter().map(|n| format!("l_{0}=l_{0}", ident(n))));
                ps.extend(tnames.iter().map(|n| format!("l_{0}=l_{0}", ident(n))));
                self.open(&format!("def {br}({})", ps.join(", ")));
                for ck in &cks {
                    self.line(format!("{ck} = []"));
                }
                self.scopes.push(vec![chunk_name.clone()]);
                self.par_hooks = true;
                let res = self.stmt(&lp, false);
                self.par_hooks = false;
                self.scopes.pop();
                res?;
                let wv: Vec<String> = roots.iter().map(|n| format!("l_{}", ident(n))).collect();
                self.line(format!("return ([{}], [{}])", wv.join(", "), cks.join(", ")));
                self.close();
                self.line(format!(
                    "{ws} = par_resolve(par_threads({nch}, [(lambda __c=__c: {br}(Lst(__c))) for __c in par_split({items}, {nch})]))"
                ));
                for (j, ck) in cks.iter().enumerate() {
                    self.line(format!("{ck} = [x for __w in {ws} for x in __w[1][{j}]]"));
                }
                self.open(&format!("for __w in {ws}"));
                if roots.is_empty() {
                    self.line("pass");
                }
                for (j, n) in roots.iter().enumerate() {
                    let id = ident(n);
                    self.line(format!("if not eq(__w[0][{j}], __base_{id}): l_{id} = merge3(__base_{id}, l_{id}, __w[0][{j}])"));
                }
                self.close();
            }
        }
        for (n, sym, bk, ik, ck) in &tv {
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

    pub(super) fn group06(&mut self, g: &str, body: &[Stmt]) -> Result<(), String> {
        let l = self.lang;
        let Some(Stmt::Expr(ne)) = body.first() else {
            return Err("missing operand".into());
        };
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
        crate::conc::assign_ops(stmts, &mut targets);
        let local = crate::conc::declared_in(stmts);
        targets.retain(|(t, _)| !local.contains(t));
        self.open_block();
        self.scopes.push(vec![]);
        let nv = self.ex(ne)?;
        let (gn, graw) = (self.fresh("gn"), self.fresh("graw"));
        match l {
            Lang::Rust => self.line(format!("let ({gn}, {graw}) = lanes_of(&{nv})?;")),
            Lang::Python => self.line(format!("{gn}, {graw} = lanes_of({nv})")),
            Lang::Js => self.line(format!("const [{gn}, {graw}] = lanesOf({nv});")),
            Lang::Scala => self.line(format!("val ({gn}, {graw}) = lanes_of({nv})")),
        }
        // per target: value before, identity, one accumulator per lane
        let mut tv = vec![];
        for (t, op) in &targets {
            let sym = q(&op[..op.len() - 1]);
            let (bk, ik, ak) = (self.fresh("gb"), self.fresh("gi"), self.fresh("ga"));
            let cur = self.var(t);
            let s = self.let_(&bk, &cur);
            self.line(s);
            let idv = match l {
                Lang::Rust => format!("par_ident({sym}, {bk}.clone())?"),
                _ => self.h("par_ident", &[sym.clone(), bk.clone()]),
            };
            let s = self.let_(&ik, &idv);
            self.line(s);
            let acc = match l {
                Lang::Rust => format!("vec![{ik}.clone(); {gn}]"),
                Lang::Python => format!("[{ik}] * {gn}"),
                Lang::Js => format!("Array({gn}).fill({ik})"),
                Lang::Scala => format!("Vector.fill[Any]({gn})({ik})"),
            };
            let s = self.let_(&ak, &acc);
            self.line(s);
            tv.push((t.clone(), sym, bk, ik, ak));
        }
        // shared declarations: declared once, before the phases; lane 0 runs them
        let mut shared = vec![];
        for ph in &phases {
            for s in ph {
                match s {
                    Stmt::Var(n, ..) => shared.push(n.clone()),
                    Stmt::Let(Pat::Path(p), ..) if p.len() == 1 && !self.is_tag(&p[0]) => {
                        shared.push(p[0].clone())
                    }
                    Stmt::Let(..) => {
                        return Err("a destructuring let at the top of a workgroup body is not emitted".into());
                    }
                    _ => {}
                }
            }
        }
        for n in &shared {
            let s = self.let_(&format!("l_{}", ident(n)), self.unit());
            self.line(s);
            self.declare(n);
        }
        let lane = self.fresh("lane");
        for ph in &phases {
            match l {
                Lang::Rust => self.open(&format!("for {lane} in 0..{gn}")),
                Lang::Python => self.open(&format!("for {lane} in range({gn})")),
                Lang::Js => self.open(&format!("for (let {lane} = 0; {lane} < {gn}; {lane}++)")),
                Lang::Scala => self.open(&format!("for ({lane} <- 0 until {gn})")),
            }
            self.scopes.push(vec![]);
            for (t, _, _, _, ak) in &tv {
                let v = match l {
                    Lang::Rust => format!("{ak}[{lane}].clone()"),
                    Lang::Scala => format!("at({ak}, {lane})"),
                    _ => format!("{ak}[{lane}]"),
                };
                self.assign(&Expr::Name(t.clone(), 0), &v)?;
            }
            let gr = match l {
                Lang::Rust => format!("group_rec({lane}, {graw})"),
                Lang::Python => format!("group_rec({lane}, {graw})"),
                Lang::Js => format!("groupRec({lane}, {graw})"),
                Lang::Scala => format!("group_rec({lane}, {graw})"),
            };
            let s = self.let_(&format!("l_{}", ident(g)), &gr);
            self.line(s);
            self.declare(g);
            for s in ph {
                match s {
                    Stmt::Var(_, ty, e, _) | Stmt::Let(Pat::Path(_), ty, e, _) => {
                        let n = match s {
                            Stmt::Var(n, ..) => n.clone(),
                            Stmt::Let(Pat::Path(p), ..) => p[0].clone(),
                            _ => unreachable!(),
                        };
                        let head = match l {
                            Lang::Js => format!("if ({lane} === 0)"),
                            Lang::Scala => format!("if ({lane} == 0)"),
                            _ => format!("if {lane} == 0"),
                        };
                        self.open(&head);
                        let mut v = self.ex_stmt(e)?;
                        if let Some(t) = ty {
                            v = self.fit(&v, &self.ty(t));
                        }
                        let s2 = self.set_(&format!("l_{}", ident(&n)), &v);
                        self.line(s2);
                        self.close();
                        self.declare(&n);
                    }
                    _ => self.stmt(s, false)?,
                }
            }
            for (t, _, _, _, ak) in &tv {
                let cur = self.var(t);
                let s = if l == Lang::Scala {
                    format!("{ak} = {ak}.updated({lane}, {cur})")
                } else {
                    self.set_(&format!("{ak}[{lane}]"), &cur)
                };

                self.line(s);
            }
            self.scopes.pop();
            self.close();
        }
        for (t, sym, bk, ik, ak) in &tv {
            let v = self.fresh("gv");
            let e = match l {
                Lang::Rust => format!("par_combine({sym}, {bk}.clone(), par_tree({sym}, &{ak}, &{ik})?)?"),
                _ => self.h(
                    "par_combine",
                    &[sym.clone(), bk.clone(), self.h("par_tree", &[sym.clone(), ak.clone(), ik.clone()])],
                ),
            };
            let s = self.let_(&v, &e);
            self.line(s);
            self.assign(&Expr::Name(t.clone(), 0), &v)?;
        }
        self.scopes.pop();
        self.close_block();
        Ok(())
    }
}

/// does a branch wait on an lvar (interp `has_wait`)
/// a `for ... par` body runs on threads unless it waits on lvars (the
/// evaluator runs iterations in order, so a wait on a later iteration is
/// stuck there), escapes the loop, yields, or emits (outputs are ordered)
pub(super) fn forpar_threadable(b: &[Stmt]) -> bool {
    if has_wait(b) {
        return false;
    }
    let mut ok = true;
    crate::check::walk_stmts(b, &mut |s| {
        if matches!(s, Stmt::Return(..) | Stmt::Break(_) | Stmt::Continue(_) | Stmt::Yield(..) | Stmt::Emit(_) | Stmt::Spawn(_) | Stmt::Send(..) | Stmt::Select(..)) {
            ok = false;
        }
    });
    ok
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

/// roots a branch writes (assignment, mutating method, `&mut`), for the restart snapshot
fn written_roots(b: &[Stmt]) -> Vec<String> {
    let mut out = std::collections::BTreeSet::new();
    crate::check::walk_stmts(b, &mut |s| {
        if let Stmt::Assign(_, p, _) = s
            && let Some(r) = crate::check::place_root(p)
        {
            out.insert(r);
        }
        for e in crate::check::stmt_exprs(s) {
            crate::check::walk_expr(e, &mut |x| match x {
                Expr::Borrow(true, y, _) => {
                    if let Some(r) = crate::check::place_root(y) {
                        out.insert(r);
                    }
                }
                Expr::Call(c, _, _, _) => {
                    if let Expr::Member(o, m, _) = &**c
                        && crate::check::mutating_method(m)
                        && let Some(r) = crate::check::place_root(o)
                    {
                        out.insert(r);
                    }
                }
                _ => {}
            });
        }
    });
    out.into_iter().collect()
}
