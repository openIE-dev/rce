//! Emission of the 6.8 / chapter C task statements onto the Rust tick
//! runtime of `rt.rs` (`--run`): channels, send, receive, select (rotation
//! and nondet), spawn (with priority and handles), scopes and deadlines.
//! Every statement mirrors the evaluation order and charge points of interp
//! `conc_*`, so with no task context (comptime) the same trap falls on the
//! same operation as before. Python and JavaScript keep the comptime trap.
use super::Lang;
use super::base::*;
use crate::ast::*;

fn pat_class(p: &Pat) -> u8 {
    match p {
        Pat::Ctor(v, _) if v.len() == 1 && v[0] == "some" => 1,
        Pat::Lit(Expr::None(_)) => 2,
        Pat::Path(v) if v.len() == 1 && v[0] == "none" => 2,
        _ => 0,
    }
}

impl<'a> G<'a> {
    pub(super) fn tasks_on(&self) -> bool {
        true
    }
    fn js(&self) -> bool {
        self.lang == Lang::Js
    }
    fn py(&self) -> bool {
        self.lang == Lang::Python
    }
    fn scl(&self) -> bool {
        self.lang == Lang::Scala
    }

    pub(super) fn chan_new_e(&mut self, cap: Option<&Expr>, line: usize) -> E {
        let c = match cap {
            Some(e) => self.ex(e)?,
            None if self.js() => "lit(0n)".into(),
            None if self.scl() => "lit(0L)".into(),
            None => "lit(0)".into(),
        };
        if self.py() {
            return Ok(format!("chan_new(lambda: {c}, {line})"));
        }
        if self.js() {
            return Ok(format!("chanNew(() => {c}, {line})"));
        }
        if self.scl() {
            return Ok(format!("chan_new(() => {c}, {line})"));
        }
        Ok(format!("chan_new(&|| Ok({c}), {line})?"))
    }

    pub(super) fn recv_e(&mut self, c: &Expr) -> E {
        let v = self.ex(c)?;
        Ok(self.h("recv", &[v]))
    }

    pub(super) fn send_s(&mut self, t: &Expr, v: &Expr) -> Result<(), String> {
        let a = self.ex(t)?;
        let ta = self.fresh("st");
        let s = self.let_(&ta, &a);
        self.line(s);
        let b = self.ex(v)?;
        let s = self.stmt_call("send", &format!("{ta}, {b}"));
        self.line(s);
        Ok(())
    }

    /// `spawn f(args)`, as an expression yielding the handle
    pub(super) fn spawn_e(&mut self, e: &Expr, prio: &str) -> E {
        let Expr::Call(f, args, line, _) = e else { return Ok(self.trap_e("spawn needs a call")) };
        let Expr::Name(fname, _) = &**f else { return Ok(self.trap_e("spawn needs a function name")) };
        let mut vs = vec![];
        for a in args {
            let v = self.ex(&a.e)?;
            let t = self.fresh("sa");
            let s = self.let_(&t, &v);
            self.line(s);
            vs.push(t);
        }
        let a = self.vec(&vs);
        Ok(self.h("spawn_", &[q(fname), a, prio.to_string(), line.to_string()]))
    }

    pub(super) fn select_s(&mut self, arms: &[SelArm], line: usize, nondet: bool, tail: bool) -> Result<(), String> {
        let av = self.fresh("arms");
        if self.py() || self.js() || self.scl() {
            return self.select_py(&av, arms, line, nondet, tail);
        }
        self.line(format!("let mut {av}: Vec<SArm> = vec![];"));
        for a in arms {
            match a {
                SelArm::Recv(p, c, _) => {
                    let v = self.ex(c)?;
                    self.line(format!("{av}.push(SArm::Recv({v}, {}));", pat_class(p)));
                }
                SelArm::Send(v, c, _) => {
                    let x = self.ex(v)?;
                    let t = self.fresh("sv");
                    let s = self.let_(&t, &x);
                    self.line(s);
                    let cv = self.ex(c)?;
                    self.line(format!("{av}.push(SArm::Send({cv}, {t}));"));
                }
                SelArm::After(n, _) => {
                    let v = self.ex(n)?;
                    self.line(format!("{av}.push(sarm_after({v})?);"));
                }
                SelArm::Default(_) => self.line(format!("{av}.push(SArm::Default);")),
            }
        }
        let (k, sv) = (self.fresh("k"), self.fresh("sv"));
        self.line(format!("let ({k}, {sv}) = select_({av}, {line}, {nondet})?;"));
        self.line(format!("let _ = &{sv};"));
        for (i, a) in arms.iter().enumerate() {
            self.open(&format!("if {k} == {i}"));
            self.scopes.push(vec![]);
            let (pat, body) = match a {
                SelArm::Recv(p, _, b) => (Some(p), b),
                SelArm::Send(_, _, b) | SelArm::After(_, b) | SelArm::Default(b) => (None, b),
            };
            if let Some(p) = pat {
                let ps = self.pat(p)?;
                let mt = self.fresh("m");
                self.line(format!(
                    "let Some({mt}) = pm(&{ps}, &{sv})? else {{ return Err(Flow::Trap(\"select pattern did not match\".to_string())); }};"
                ));
                let mut names = vec![];
                self.pat_names(p, &mut names);
                self.bind_from_map(&mt, &names);
            }
            self.block(body, tail)?;
            self.scopes.pop();
            self.close();
        }
        Ok(())
    }

    fn select_py(&mut self, av: &str, arms: &[SelArm], line: usize, nondet: bool, tail: bool) -> Result<(), String> {
        if self.scl() {
            return self.select_scala(av, arms, line, nondet, tail);
        }
        let js = self.js();
        let sc = if js { ";" } else { "" };
        let push = if js { "push" } else { "append" };
        self.line(if js { format!("const {av} = [];") } else { format!("{av} = []") });
        for a in arms {
            match a {
                SelArm::Recv(p, c, _) => {
                    let v = self.ex(c)?;
                    self.line(if js { format!("{av}.push([\"recv\", {v}, {}]);", pat_class(p)) } else { format!("{av}.append((\"recv\", {v}, {}))", pat_class(p)) });
                }
                SelArm::Send(v, c, _) => {
                    let x = self.ex(v)?;
                    let t = self.fresh("sv");
                    self.line(if js { format!("const {t} = {x};") } else { format!("{t} = {x}") });
                    let cv = self.ex(c)?;
                    self.line(if js { format!("{av}.push([\"send\", {cv}, {t}]);") } else { format!("{av}.append((\"send\", {cv}, {t}))") });
                }
                SelArm::After(n, _) => {
                    let v = self.ex(n)?;
                    self.line(format!("{av}.{push}({}({v})){sc}", self.rt("sarm_after")));
                }
                SelArm::Default(_) => self.line(if js { format!("{av}.push([\"default\"]);") } else { format!("{av}.append((\"default\",))") }),
            }
        }
        let (k, sv) = (self.fresh("k"), self.fresh("sv"));
        let nd = match (nondet, js) { (true, true) => "true", (false, true) => "false", (true, false) => "True", (false, false) => "False" };
        self.line(if js { format!("const [{k}, {sv}] = select_({av}, {line}, {nd});") } else { format!("{k}, {sv} = select_({av}, {line}, {nd})") });
        for (i, a) in arms.iter().enumerate() {
            self.open(&if js { format!("if ({k} === {i})") } else { format!("if {k} == {i}") });
            self.scopes.push(vec![]);
            let (pat, body) = match a {
                SelArm::Recv(p, _, b) => (Some(p), b),
                SelArm::Send(_, _, b) | SelArm::After(_, b) | SelArm::Default(b) => (None, b),
            };
            if let Some(p) = pat {
                let ps = self.pat(p)?;
                let mt = self.fresh("m");
                if js {
                    self.line(format!("const {mt} = pm({ps}, {sv});"));
                    self.line(format!("if ({mt} === null) trap(\"select pattern did not match\");"));
                } else {
                    self.line(format!("{mt} = pm({ps}, {sv})"));
                    self.line(format!("if {mt} is None: trap(\"select pattern did not match\")"));
                }
                let mut names = vec![];
                self.pat_names(p, &mut names);
                self.bind_from_map(&mt, &names);
            }
            self.block(body, tail)?;
            self.scopes.pop();
            self.close();
        }
        Ok(())
    }

    fn select_scala(&mut self, av: &str, arms: &[SelArm], line: usize, nondet: bool, tail: bool) -> Result<(), String> {
        self.line(format!("val {av} = ArrayBuffer[Any]()"));
        for a in arms {
            match a {
                SelArm::Recv(p, c, _) => {
                    let v = self.ex(c)?;
                    self.line(format!("{av} += V(\"recv\", {v}, {})", pat_class(p)));
                }
                SelArm::Send(v, c, _) => {
                    let x = self.ex(v)?;
                    let t = self.fresh("sv");
                    self.line(format!("val {t} = {x}"));
                    let cv = self.ex(c)?;
                    self.line(format!("{av} += V(\"send\", {cv}, {t})"));
                }
                SelArm::After(n, _) => {
                    let v = self.ex(n)?;
                    self.line(format!("{av} += sarm_after({v})"));
                }
                SelArm::Default(_) => self.line(format!("{av} += V(\"default\")")),
            }
        }
        let (k, sv, r) = (self.fresh("k"), self.fresh("sv"), self.fresh("sr"));
        self.line(format!("val {r} = select_({av}.toVector, {line}, {nondet})"));
        self.line(format!("val {k} = {r}(0); val {sv} = {r}(1)"));
        for (i, a) in arms.iter().enumerate() {
            self.open(&format!("if ({k} == {i})"));
            self.scopes.push(vec![]);
            let (pat, body) = match a {
                SelArm::Recv(p, _, b) => (Some(p), b),
                SelArm::Send(_, _, b) | SelArm::After(_, b) | SelArm::Default(b) => (None, b),
            };
            if let Some(p) = pat {
                let ps = self.pat(p)?;
                let mt = self.fresh("m");
                self.line(format!("val {mt} = pm({ps}, {sv})"));
                self.line(format!("if ({mt} == null) trap(\"select pattern did not match\")"));
                let mut names = vec![];
                self.pat_names(p, &mut names);
                self.bind_from_map(&mt, &names);
            }
            self.block(body, tail)?;
            self.scopes.pop();
            self.close();
        }
        Ok(())
    }

    fn closure_block(&mut self, body: &[Stmt], r: &str) -> Result<(), String> {
        self.line(format!("let {r}: R<()> = (|| -> R<()> {{"));
        self.ind += 1;
        self.cl.push(0);
        let res = self.block(body, false);
        self.cl.pop();
        res?;
        self.line("Ok(())");
        self.ind -= 1;
        self.line("})();");
        Ok(())
    }

    /// re-raise a return, break or continue that left a task closure
    fn escapes_out(&mut self) {
        self.line("Err(Flow::Ret(v)) => {");
        self.ind += 1;
        self.ret_with("v");
        self.ind -= 1;
        self.line("}");
        let (b, c) = if self.loops == 0 {
            ("return trap(\"break outside a loop\");", "return trap(\"continue outside a loop\");")
        } else if self.cl.last() == Some(&0) {
            ("return Err(Flow::Brk);", "return Err(Flow::Cont);")
        } else {
            ("break;", "continue;")
        };
        self.line(format!("Err(Flow::Brk) => {{ {b} }}"));
        self.line(format!("Err(Flow::Cont) => {{ {c} }}"));
    }

    pub(super) fn scope_s(&mut self, body: &[Stmt]) -> Result<(), String> {
        let (sid, r) = (self.fresh("sid"), self.fresh("r"));
        if self.scl() {
            self.line(format!("val {sid} = scope_enter()"));
            self.open("try");
            self.block(body, false)?;
            self.ind -= 1;
            self.line("} finally { scope_exit() }");
            self.line(format!("await_scope({sid})"));
            return Ok(());
        }
        if self.py() || self.js() {
            // return / break / continue leave through `finally`, skipping the await, as interp
            let sc = if self.js() { ";" } else { "" };
            let decl = if self.js() { "const " } else { "" };
            self.line(format!("{decl}{sid} = {}(){sc}", self.rt("scope_enter")));
            self.open("try");
            self.block(body, false)?;
            self.close();
            self.open("finally");
            self.line(format!("{}(){sc}", self.rt("scope_exit")));
            self.close();
            self.line(format!("{}({sid}){sc}", self.rt("await_scope")));
            return Ok(());
        }
        self.line(format!("let {sid} = scope_enter();"));
        self.closure_block(body, &r)?;
        self.line("scope_exit();");
        self.line(format!("match {r} {{"));
        self.ind += 1;
        self.escapes_out();
        self.line(format!("r => {{ r?; await_scope({sid})?; }}"));
        self.ind -= 1;
        self.line("}");
        Ok(())
    }

    pub(super) fn deadline_s(&mut self, body: &[Stmt]) -> Result<(), String> {
        let n = self.marker_operand(body)?;
        let (f, r) = (self.fresh("fid"), self.fresh("r"));
        let b: &[Stmt] = match body.get(1) {
            Some(Stmt::Arena(_, _, b, _)) => b,
            _ => &[],
        };
        if self.scl() {
            let e = self.fresh("e");
            self.line(format!("val {f} = dl_enter({n})"));
            self.open("try");
            self.open("try");
            self.block(b, false)?;
            self.ind -= 1;
            self.line("} finally { dl_exit() }");
            self.ind -= 1;
            self.line(format!("}} catch {{ case {e}: Cancel if {e}.f == {f} =>"));
            self.ind += 1;
            if let Some(Stmt::If(arms, _)) = body.get(2) {
                self.block(&arms[0].1, false)?;
            }
            self.line("()");
            self.close();
            return Ok(());
        }
        if self.js() {
            let e = self.fresh("e");
            self.line(format!("const {f} = dlEnter({n});"));
            self.open("try");
            self.open("try");
            self.block(b, false)?;
            self.close();
            self.open("finally");
            self.line("dlExit();");
            self.close();
            self.close();
            self.open(&format!("catch ({e})"));
            self.line(format!("if (!({e} instanceof Cancel) || {e}.f !== {f}) throw {e};"));
            if let Some(Stmt::If(arms, _)) = body.get(2) {
                self.block(&arms[0].1, false)?;
            }
            self.close();
            return Ok(());
        }
        if self.py() {
            let e = self.fresh("e");
            self.line(format!("{f} = dl_enter({n})"));
            self.open("try");
            self.open("try");
            self.block(b, false)?;
            self.close();
            self.open("finally");
            self.line("dl_exit()");
            self.close();
            self.close();
            self.open(&format!("except Cancel as {e}"));
            self.line(format!("if {e}.f != {f}: raise"));
            if let Some(Stmt::If(arms, _)) = body.get(2) {
                self.block(&arms[0].1, false)?;
            }
            self.close();
            return Ok(());
        }
        self.line(format!("let {f} = dl_enter({n});"));
        self.closure_block(b, &r)?;
        self.line("dl_exit();");
        self.line(format!("match {r} {{"));
        self.ind += 1;
        self.line(format!("Err(Flow::Cancel(x)) if x == {f} => {{"));
        self.ind += 1;
        if let Some(Stmt::If(arms, _)) = body.get(2) {
            self.block(&arms[0].1, false)?;
        }
        self.ind -= 1;
        self.line("}");
        self.escapes_out();
        self.line("r => r?,");
        self.ind -= 1;
        self.line("}");
        Ok(())
    }

    /// `$priority p { spawn .. }` or `$spawnas:h { .. }`
    pub(super) fn spawn_marker(&mut self, name: &str, body: &[Stmt]) -> Result<(), String> {
        let (prio, sp) = match (name, body.first(), body.get(1)) {
            ("$priority", _, Some(Stmt::Spawn(e))) => (self.marker_operand(body)?, e),
            (_, Some(Stmt::Spawn(e)), _) => ("0".to_string(), e),
            (_, Some(Stmt::Arena(m, _, b, _)), _) if m == "$priority" => match b.get(1) {
                Some(Stmt::Spawn(e)) => (self.marker_operand(b)?, e),
                _ => return {
                    self.line(self.trap_s("bad spawn"));
                    Ok(())
                },
            },
            _ => return {
                self.line(self.trap_s("bad spawn"));
                Ok(())
            },
        };
        let pv = self.fresh("pr");
        if self.py() {
            self.line(format!("{pv} = {prio}"));
        } else if self.js() {
            self.line(format!("const {pv} = {prio};"));
        } else if self.scl() {
            self.line(format!("val {pv} : Any = {prio}"));
        } else {
            self.line(format!("let {pv}: i128 = {prio};"));
        }
        let h = self.spawn_e(sp, &pv)?;
        if let Some(hn) = name.strip_prefix("$spawnas:") {
            let s = self.let_(&format!("l_{}", ident(hn)), &h);
            self.line(s);
            self.declare(hn);
        } else {
            let s = self.discard(&h);
            self.line(s);
        }
        Ok(())
    }

    /// Rust: the actor handlers (as state steps over the actor's fields),
    /// initial fields, node placement and supervisor table read by the
    /// runtime's actors phase (interp `Sched::handler`, `actor_init`,
    /// `on_crash`, `route`).
    /// Tables the tick runtimes need for restart-log schemas and program
    /// events (C.11.11, HC-6): the supervisor tree, each event's payload
    /// fields with their type names, and the events each actor handles.
    fn conc_meta_tables(&mut self) {
        let m = self.m;
        let tree = crate::conc::sup_tree(m);
        let mut evs: Vec<(String, Vec<(String, String)>)> = vec![];
        let mut hand: Vec<(String, Vec<String>)> = vec![];
        for it in &m.items {
            match it {
                Item::Event { name, payload, .. } => {
                    let fs = payload
                        .iter()
                        .map(|(f, t)| (f.clone(), match t { Ty::Path(p, a) if p.len() == 1 && a.is_empty() => p[0].clone(), _ => "?".into() }))
                        .collect();
                    evs.push((name.clone(), fs));
                }
                Item::Actor { name, handlers, .. } => {
                    let mut ns = vec![];
                    for h in handlers {
                        if let Pat::Ctor(p, _) | Pat::Path(p) = &h.pat
                            && let Some(x) = p.last()
                            && !ns.contains(x)
                        {
                            ns.push(x.clone());
                        }
                    }
                    hand.push((name.clone(), ns));
                }
                _ => {}
            }
        }
        let qs = |v: &[String]| v.iter().map(|x| q(x)).collect::<Vec<_>>().join(", ");
        // 6.1: a variant record carries "$v" in the Form, so D needs the variant names
        let mut vars: Vec<String> = vec![];
        for it in &m.items {
            if let Item::Kind { variants, .. } = it {
                vars.extend(variants.iter().map(|(v, _)| v.clone()));
            }
        }
        // 6.1, T7: the positional embedding hashed by D needs the declared
        // field order of every struct, event and error, each variant's index
        // in its kind, and each event's index among the events
        let mut decls: Vec<(String, Vec<String>)> = vec![];
        let mut vidx: Vec<(String, usize, Vec<String>)> = vec![];
        let mut evidx: Vec<(String, usize)> = vec![];
        for it in &m.items {
            match it {
                Item::Struct { name, fields, .. } => decls.push((name.clone(), fields.iter().map(|f| f.name.clone()).collect())),
                Item::Event { name, payload, .. } => {
                    evidx.push((name.clone(), evidx.len()));
                    decls.push((name.clone(), payload.iter().map(|(f, _)| f.clone()).collect()));
                }
                Item::Error { name, payload, .. } => decls.push((name.clone(), payload.iter().map(|(f, _)| f.clone()).collect())),
                Item::Kind { variants, .. } => {
                    for (k, (v, fs)) in variants.iter().enumerate() {
                        vidx.push((v.clone(), k, fs.iter().map(|(f, _)| f.clone()).collect()));
                    }
                }
                _ => {}
            }
        }
        match self.lang {
            Lang::Rust => {
                let mut o = String::from("#[allow(unreachable_patterns)]\nfn decl_fields(n: &str) -> Option<&'static [&'static str]> {\n  match n {\n");
                for (n, fs) in &decls {
                    o += &format!("    {} => Some(&[{}]),\n", q(n), qs(fs));
                }
                o += "    _ => None,\n  }\n}\n#[allow(unreachable_patterns)]\nfn variant_index(n: &str) -> Option<(i128, &'static [&'static str])> {\n  match n {\n";
                for (v, k, fs) in &vidx {
                    o += &format!("    {} => Some(({k}, &[{}])),\n", q(v), qs(fs));
                }
                o += "    _ => None,\n  }\n}\n#[allow(unreachable_patterns)]\nfn event_index(n: &str) -> Option<usize> {\n  match n {\n";
                for (n, k) in &evidx {
                    o += &format!("    {} => Some({k}),\n", q(n));
                }
                o += "    _ => None,\n  }\n}";
                self.line(o);
            }
            Lang::Scala => {
                let d: Vec<String> = decls.iter().map(|(n, fs)| format!("{} -> Vector[String]({})", q(n), qs(fs))).collect();
                self.line(format!("lazy val DECL_FIELDS: Map[String, Vector[String]] = Map({})", d.join(", ")));
                let v: Vec<String> = vidx.iter().map(|(v, k, fs)| format!("{} -> (BigInt({k}), Vector[String]({}))", q(v), qs(fs))).collect();
                self.line(format!("lazy val VARIANT_INDEX: Map[String, (BigInt, Vector[String])] = Map({})", v.join(", ")));
                let e: Vec<String> = evidx.iter().map(|(n, k)| format!("{} -> {k}", q(n))).collect();
                self.line(format!("lazy val EVENT_INDEX: Map[String, Int] = Map({})", e.join(", ")));
            }
            _ => {
                let (decl, end) = if self.js() { ("const ", ";") } else { ("", "") };
                let d: Vec<String> = decls.iter().map(|(n, fs)| format!("{}: [{}]", q(n), qs(fs))).collect();
                self.line(format!("{decl}DECL_FIELDS = {{{}}}{end}", d.join(", ")));
                let v: Vec<String> = vidx
                    .iter()
                    .map(|(v, k, fs)| if self.js() { format!("{}: [{k}, [{}]]", q(v), qs(fs)) } else { format!("{}: ({k}, [{}])", q(v), qs(fs)) })
                    .collect();
                self.line(format!("{decl}VARIANT_INDEX = {{{}}}{end}", v.join(", ")));
                let e: Vec<String> = evidx.iter().map(|(n, k)| format!("{}: {k}", q(n))).collect();
                self.line(format!("{decl}EVENT_INDEX = {{{}}}{end}", e.join(", ")));
            }
        }
        match self.lang {
            Lang::Rust if vars.is_empty() => self.line("fn is_variant(_n: &str) -> bool { false }".to_string()),
            Lang::Rust => self.line(format!("fn is_variant(n: &str) -> bool {{ matches!(n, {}) }}", vars.iter().map(|x| q(x)).collect::<Vec<_>>().join(" | "))),
            Lang::Scala => self.line(format!("lazy val VARIANTS: Set[String] = Set({})", qs(&vars))),
            Lang::Js => self.line(format!("const VARIANTS = new Set([{}]);", qs(&vars))),
            _ => self.line(format!("VARIANTS = frozenset([{}])", qs(&vars))),
        }
        match self.lang {
            Lang::Rust => {
                let mut o = format!("fn sup_tree() -> Vec<&'static str> {{ vec![{}] }}\n", qs(&tree));
                o += "#[allow(unreachable_patterns)]\npub fn event_fields(n: &str) -> Option<Vec<(&'static str, &'static str)>> {\n  match n {\n";
                for (n, fs) in &evs {
                    let f: Vec<String> = fs.iter().map(|(a, b)| format!("({}, {})", q(a), q(b))).collect();
                    o += &format!("    {} => Some(vec![{}]),\n", q(n), f.join(", "));
                }
                o += "    _ => None,\n  }\n}\n#[allow(unreachable_patterns)]\nfn actor_events(t: &str) -> Vec<&'static str> {\n  match t {\n";
                for (t, ns) in &hand {
                    o += &format!("    {} => vec![{}],\n", q(t), qs(ns));
                }
                o += "    _ => vec![],\n  }\n}";
                self.line(o);
            }
            Lang::Scala => {
                self.line(format!("lazy val SUP_TREE: Vector[String] = Vector({})", qs(&tree)));
                let e: Vec<String> = evs.iter().map(|(n, fs)| {
                    let f: Vec<String> = fs.iter().map(|(a, b)| format!("({}, {})", q(a), q(b))).collect();
                    format!("{} -> Vector[(String, String)]({})", q(n), f.join(", "))
                }).collect();
                self.line(format!("lazy val EVENT_FIELDS: Map[String, Vector[(String, String)]] = Map({})", e.join(", ")));
                let h: Vec<String> = hand.iter().map(|(t, ns)| format!("{} -> Vector[String]({})", q(t), qs(ns))).collect();
                self.line(format!("lazy val ACTOR_EVENTS: Map[String, Vector[String]] = Map({})", h.join(", ")));
            }
            _ => {
                let (decl, end) = if self.js() { ("const ", ";") } else { ("", "") };
                self.line(format!("{decl}SUP_TREE = [{}]{end}", qs(&tree)));
                let e: Vec<String> = evs.iter().map(|(n, fs)| {
                    let f: Vec<String> = fs.iter().map(|(a, b)| format!("[{}, {}]", q(a), q(b))).collect();
                    format!("{}: [{}]", q(n), f.join(", "))
                }).collect();
                self.line(format!("{decl}EVENT_FIELDS = {{{}}}{end}", e.join(", ")));
                let h: Vec<String> = hand.iter().map(|(t, ns)| format!("{}: [{}]", q(t), qs(ns))).collect();
                self.line(format!("{decl}ACTOR_EVENTS = {{{}}}{end}", h.join(", ")));
            }
        }
    }

    pub(super) fn actor_tables(&mut self) -> Result<(), String> {
        self.conc_meta_tables();
        let m = self.m;
        let mut hs: Vec<(String, Vec<String>)> = vec![];
        let mut inits: Vec<(String, String)> = vec![];
        for (k, i) in m.items.iter().enumerate() {
            let Item::Actor { name, fields, handlers, .. } = i else { continue };
            let mut ids = vec![];
            for (j, h) in handlers.iter().enumerate() {
                let sid = format!("ah_{k}_{j}");
                self.step(&sid, fields, &h.pat, &h.guard, &h.body)?;
                ids.push(sid);
            }
            hs.push((name.clone(), ids));
            let aid = format!("ai_{k}");
            if self.scl() {
                self.open(&format!("def {aid}(): Vector[Any] ="));
                self.scopes = vec![vec![]];
                self.ctx = Ctx::Gen;
                self.line("val __fs = ArrayBuffer[Any]()");
                self.line("var __v: Any = null");
                for f in fields {
                    let v = match &f.init {
                        Some(e) => self.ex(e)?,
                        None => "UNIT".into(),
                    };
                    self.line(format!("__v = try {{ {v} }} catch {{ case _: Flow => UNIT }}"));
                    let fv = self.fit("__v", &self.ty(&f.ty));
                    self.line(format!("__v = try {{ {fv} }} catch {{ case _: Flow => UNIT }}"));
                    self.line(format!("__fs += V({}, __v)", q(&f.name)));
                }
                self.line("__fs.toVector");
                self.close();
                inits.push((name.clone(), aid));
                continue;
            }
            if self.js() {

                self.open(&format!("function {aid}()"));
                self.scopes = vec![vec![]];
                self.ctx = Ctx::Gen;
                self.line("const __fs = [];");
                self.line("let __v;");
                for f in fields {
                    self.open("try");
                    match &f.init {
                        Some(e) => {
                            let v = self.ex(e)?;
                            self.line(format!("__v = {v};"));
                        }
                        None => self.line("__v = UNIT;"),
                    }
                    self.close();
                    self.line("catch (e) { if (!isFlow(e)) throw e; __v = UNIT; }");
                    let fv = self.fit("__v", &self.ty(&f.ty));
                    self.line(format!("try {{ __v = {fv}; }} catch (e) {{ if (!isFlow(e)) throw e; __v = UNIT; }}"));
                    self.line(format!("__fs.push([{}, __v]);", q(&f.name)));
                }
                self.line("return __fs;");
                self.close();
                inits.push((name.clone(), aid));
                continue;
            }
            if self.py() {
                self.open(&format!("def {aid}()"));
                self.scopes = vec![vec![]];
                self.ctx = Ctx::Gen;
                self.line("__fs = []");
                for f in fields {
                    self.open("try");
                    match &f.init {
                        Some(e) => {
                            let v = self.ex(e)?;
                            self.line(format!("__v = {v}"));
                        }
                        None => self.line("__v = UNIT"),
                    }
                    self.close();
                    self.line("except (Trap, Fail, Stop, Ret): __v = UNIT");
                    let fv = self.fit("__v", &self.ty(&f.ty));
                    self.open("try");
                    self.line(format!("__v = {fv}"));
                    self.close();
                    self.line("except (Trap, Fail, Stop, Ret): __v = UNIT");
                    self.line(format!("__fs.append(({}, __v))", q(&f.name)));
                }
                self.line("return __fs");
                self.close();
                inits.push((name.clone(), aid));
                continue;
            }
            self.open(&format!("fn {aid}() -> Vec<(&'static str, V)>"));
            self.scopes = vec![vec![]];
            self.ctx = Ctx::Gen;
            let mut outs = vec![];
            for f in fields {
                let t = self.fresh("f");
                self.line(format!("let {t}: V = (|| -> R<V> {{"));
                self.ind += 1;
                let v = match &f.init {
                    Some(e) => {
                        self.line("let __v: V = (|| -> R<V> {");
                        self.ind += 1;
                        let v = self.ex(e)?;
                        self.line(format!("Ok({v})"));
                        self.ind -= 1;
                        self.line("})().unwrap_or(V::Unit);");
                        "__v".to_string()
                    }
                    None => "V::Unit".to_string(),
                };
                let fv = self.fit(&v, &self.ty(&f.ty));
                self.line(format!("Ok({fv})"));
                self.ind -= 1;
                self.line("})().unwrap_or(V::Unit);");
                outs.push(format!("({}, {t})", q(&f.name)));
            }
            self.line(format!("vec![{}]", outs.join(", ")));
            self.close();
            inits.push((name.clone(), aid));
        }
        if self.py() || self.js() || self.scl() {
            return self.actor_tables_py(&hs, &inits);
        }
        let mut o = String::new();
        o += "fn actor_handlers(t: &str) -> Vec<StepFn> {\n  match t {\n";
        for (n, ids) in &hs {
            let v: Vec<String> = ids.iter().map(|i| format!("{i} as StepFn")).collect();
            o += &format!("    {} => vec![{}],\n", q(n), v.join(", "));
        }
        o += "    _ => vec![],\n  }\n}\n";
        o += "fn actor_init(t: &str) -> Vec<(&'static str, V)> {\n  match t {\n";
        for (n, aid) in &inits {
            o += &format!("    {} => {aid}(),\n", q(n));
        }
        o += "    _ => vec![],\n  }\n}\n";
        let mut nodes = vec![q("main")];
        let mut place = vec![];
        let mut sups = vec![];
        for i in &m.items {
            match i {
                Item::Node { name, actors, .. } => {
                    nodes.push(q(name));
                    for a in actors {
                        place.push(format!("    {} => {},\n", q(a), nodes.len() - 1));
                    }
                }
                Item::Supervisor { name, strategy, children, .. } => {
                    let tys: Vec<String> = children.iter().map(|c| q(&c.ty)).collect();
                    for c in children {
                        if sups.iter().any(|(t, _): &(String, String)| *t == c.ty) {
                            continue;
                        }
                        sups.push((
                            c.ty.clone(),
                            format!("Some(({}, {}, vec![{}], {}, {}))", q(name), q(strategy), tys.join(", "), q(&c.name), c.restart),
                        ));
                    }
                }
                _ => {}
            }
        }
        o += &format!("fn node_names() -> Vec<&'static str> {{ vec![{}] }}\n", nodes.join(", "));
        o += "#[allow(unreachable_patterns)]\nfn node_of(t: &str) -> usize {\n  match t {\n";
        // a later `node` placement of the same actor wins, as interp's map
        for p in place.into_iter().rev() {
            o += &p;
        }
        o += "    _ => 0,\n  }\n}\n";
        o += "#[allow(unreachable_patterns)]\nfn supervisor_of(t: &str) -> Option<(&'static str, &'static str, Vec<&'static str>, &'static str, u32)> {\n  match t {\n";
        for (t, v) in sups {
            o += &format!("    {} => {v},\n", q(&t));
        }
        o += "    _ => None,\n  }\n}\n";
        o += &format!("fn restart_window() -> u64 {{ {} }}", restart_window(m));
        self.line(o);
        Ok(())
    }

    fn actor_tables_py(&mut self, hs: &[(String, Vec<String>)], inits: &[(String, String)]) -> Result<(), String> {
        let m = self.m;
        if self.scl() {
            return self.actor_tables_scala(hs, inits);
        }
        let h: Vec<String> = hs.iter().map(|(n, ids)| format!("{}: [{}]", q(n), ids.join(", "))).collect();
        let (decl, end) = if self.js() { ("const ", ";") } else { ("", "") };
        self.line(format!("{decl}ACTOR_HANDLERS = {{{}}}{end}", h.join(", ")));
        let i: Vec<String> = inits.iter().map(|(n, a)| format!("{}: {a}", q(n))).collect();
        self.line(format!("{decl}ACTOR_INIT = {{{}}}{end}", i.join(", ")));
        let mut nodes = vec![q("main")];
        let mut place = vec![];
        let mut sups: Vec<(String, String)> = vec![];
        for it in &m.items {
            match it {
                Item::Node { name, actors, .. } => {
                    nodes.push(q(name));
                    for a in actors {
                        place.push((a.clone(), nodes.len() - 1));
                    }
                }
                Item::Supervisor { name, strategy, children, .. } => {
                    let tys: Vec<String> = children.iter().map(|c| q(&c.ty)).collect();
                    for c in children {
                        if sups.iter().any(|(t, _)| *t == c.ty) {
                            continue;
                        }
                        let (lo, hi) = if self.js() { ("[", "]") } else { ("(", ")") };
                        sups.push((c.ty.clone(), format!("{lo}{}, {}, [{}], {}, {}{hi}", q(name), q(strategy), tys.join(", "), q(&c.name), c.restart)));
                    }
                }
                _ => {}
            }
        }
        self.line(format!("{decl}NODE_NAMES = [{}]{end}", nodes.join(", ")));
        let p: Vec<String> = place.iter().map(|(a, n)| format!("{}: {n}", q(a))).collect();
        self.line(format!("{decl}NODE_OF = {{{}}}{end}", p.join(", ")));
        let sv: Vec<String> = sups.iter().map(|(t, v)| format!("{}: {v}", q(t))).collect();
        self.line(format!("{decl}SUPERVISORS = {{{}}}{end}", sv.join(", ")));
        self.line(format!("{decl}RESTART_WINDOW = {}{end}", restart_window(m)));
        Ok(())
    }

    fn actor_tables_scala(&mut self, hs: &[(String, Vec<String>)], inits: &[(String, String)]) -> Result<(), String> {
        let m = self.m;
        let h: Vec<String> = hs.iter().map(|(n, ids)| {
            let f: Vec<String> = ids.iter().map(|i| format!("({i} _)")).collect();
            format!("{} -> Vector[(Vector[Any], Any) => Any]({})", q(n), f.join(", "))
        }).collect();
        self.line(format!("lazy val ACTOR_HANDLERS: Map[String, Vector[(Vector[Any], Any) => Any]] = Map({})", h.join(", ")));
        let i: Vec<String> = inits.iter().map(|(n, a)| format!("{} -> (() => {a}())", q(n))).collect();
        self.line(format!("lazy val ACTOR_INIT: Map[String, () => Vector[Any]] = Map({})", i.join(", ")));
        let mut nodes = vec![q("main")];
        let mut place: Vec<(String, usize)> = vec![];
        let mut sups: Vec<(String, String)> = vec![];
        for it in &m.items {
            match it {
                Item::Node { name, actors, .. } => {
                    nodes.push(q(name));
                    for a in actors {
                        place.retain(|(x, _)| x != a);
                        place.push((a.clone(), nodes.len() - 1));
                    }
                }
                Item::Supervisor { name, strategy, children, .. } => {
                    let tys: Vec<String> = children.iter().map(|c| q(&c.ty)).collect();
                    for c in children {
                        if sups.iter().any(|(t, _)| *t == c.ty) {
                            continue;
                        }
                        sups.push((c.ty.clone(), format!("V({}, {}, V({}), {}, {})", q(name), q(strategy), tys.join(", "), q(&c.name), c.restart)));
                    }
                }
                _ => {}
            }
        }
        self.line(format!("lazy val NODE_NAMES: Vector[String] = Vector({})", nodes.join(", ")));
        let p: Vec<String> = place.iter().map(|(a, n)| format!("{} -> {n}", q(a))).collect();
        self.line(format!("lazy val NODE_OF: Map[String, Int] = Map({})", p.join(", ")));
        let sv: Vec<String> = sups.iter().map(|(t, v)| format!("{} -> {v}", q(t))).collect();
        self.line(format!("lazy val SUPERVISORS: Map[String, Vector[Any]] = Map({})", sv.join(", ")));
        self.line(format!("lazy val RESTART_WINDOW: Long = {}L", restart_window(m)));
        Ok(())
    }


    pub(super) fn atomic_s(&mut self, body: &[Stmt]) -> Result<(), String> {
        let (o, r) = (self.fresh("ao"), self.fresh("r"));
        if self.scl() {
            let (ok, buf) = (self.fresh("ok"), self.fresh("buf"));
            self.line(format!("val {o} = atomic_enter()"));
            self.line(format!("var {ok} = false; var {buf}: ArrayBuffer[Any] = null"));
            self.open("try");
            self.block(body, false)?;
            self.line(format!("{ok} = true"));
            self.ind -= 1;
            self.line(format!("}} finally {{ {buf} = atomic_restore({o}) }}"));
            self.line(format!("if ({ok}) atomic_commit({buf})"));
            return Ok(());
        }
        if self.js() {
            let (ok, buf) = (self.fresh("ok"), self.fresh("buf"));
            self.line(format!("const {o} = atomicEnter();"));
            self.line(format!("let {ok} = false, {buf} = null;"));
            self.open("try");
            self.block(body, false)?;
            self.line(format!("{ok} = true;"));
            self.close();
            self.open("finally");
            self.line(format!("{buf} = atomicRestore({o});"));
            self.close();
            self.line(format!("if ({ok}) atomicCommit({buf});"));
            return Ok(());
        }
        if self.py() {
            let (ok, buf) = (self.fresh("ok"), self.fresh("buf"));
            self.line(format!("{o} = atomic_enter()"));
            self.line(format!("{ok} = False"));
            self.open("try");
            self.block(body, false)?;
            self.line(format!("{ok} = True"));
            self.close();
            self.open("finally");
            self.line(format!("{buf} = atomic_restore({o})"));
            self.close();
            self.line(format!("if {ok}: atomic_commit({buf})"));
            return Ok(());
        }
        self.line(format!("let {o} = atomic_enter();"));
        self.closure_block(body, &r)?;
        self.line(format!("let {r} = atomic_exit({o}, {r});"));
        self.line(format!("match {r} {{"));
        self.ind += 1;
        self.escapes_out();
        self.line("r => r?,");
        self.ind -= 1;
        self.line("}");
        Ok(())
    }
}

/// `const RESTART_WINDOW: u64 = W` (C.11.11), or 0 for an unbounded window
fn restart_window(m: &Module) -> u64 {
    m.items
        .iter()
        .find_map(|i| match i {
            Item::Const { name, e: Expr::Int(n, _), .. } if name == "RESTART_WINDOW" && *n > 0 => Some(*n as u64),
            _ => None,
        })
        .unwrap_or(0)
}
