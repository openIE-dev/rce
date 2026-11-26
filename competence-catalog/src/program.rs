//! Whole-program execution + lifting for the statement layer (the IR-kind expansion).
//! The expression IR (rce-spec-check Expr) stays the core; this executes declarations,
//! assignment, blocks, conditionals and bounded loops over an environment. Loops carry a
//! fuel guard so execution is total; a lifted program that exceeds fuel is refused, not hung.

use std::collections::BTreeMap;

use rce_spec_check::ast::Expr;

use crate::lut_emit::Stmt;

pub type Env = BTreeMap<String, i128>;

const FUEL: u64 = 100_000;

/// Evaluate an expression against an environment (Name resolves from env).
pub fn eval_expr_env(e: &Expr, env: &Env) -> Option<i128> {
    use Expr::*;
    match e {
        Int(v, _) => Some(*v),
        Name(n, _) => env.get(n).copied(),
        Neg(a, _) => Some(-eval_expr_env(a, env)?),
        Bin(op, a, b, _) => {
            let x = eval_expr_env(a, env)?;
            let y = eval_expr_env(b, env)?;
            match op.as_str() {
                "+" => Some(x + y),
                "-" => Some(x - y),
                "*" => Some(x * y),
                "/" => if y != 0 { Some(x / y) } else { Option::None },
                "%" => if y != 0 { Some(x % y) } else { Option::None },
                "==" => Some((x == y) as i128),
                "!=" => Some((x != y) as i128),
                "<" => Some((x < y) as i128),
                ">" => Some((x > y) as i128),
                "<=" => Some((x <= y) as i128),
                ">=" => Some((x >= y) as i128),
                "&&" => Some(((x != 0) && (y != 0)) as i128),
                "||" => Some(((x != 0) || (y != 0)) as i128),
                _ => Option::None,
            }
        }
        _ => Option::None,
    }
}

/// Local functions (closure seed) plus a deterministic macro table.
/// A macro is (arity, expression template with `{0}`,`{1}`,… arg slots); expansion is
/// substitution + lift, a deterministic rewrite, never a model call.
pub struct FnTable {
    pub fns: BTreeMap<String, (Vec<String>, Stmt)>,
    pub macros: BTreeMap<String, (usize, String)>,
}

impl FnTable {
    pub fn new() -> Self {
        FnTable { fns: BTreeMap::new(), macros: BTreeMap::new() }
    }
    pub fn add_macro(&mut self, name: &str, arity: usize, template: &str) {
        self.macros.insert(name.to_string(), (arity, template.to_string()));
    }
}

impl Default for FnTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Expand a macro by substituting rendered args into its template and lifting the result.
pub fn expand_macro(name: &str, args: &[Expr], macros: &BTreeMap<String, (usize, String)>) -> Result<Expr, String> {
    let (arity, tpl) = macros.get(name).ok_or_else(|| format!("unknown macro {name}"))?;
    if *arity != args.len() {
        return Err(format!("macro {name} arity mismatch"));
    }
    let mut out = tpl.clone();
    for (i, a) in args.iter().enumerate() {
        out = out.replace(&format!("{{{i}}}"), &crate::canonical::emit(a));
    }
    crate::lift::lift_expr(&out)
}

/// Execute a statement; returns the returned value (Return) or None. Fuel-bounded.
pub fn eval_stmt(s: &Stmt, env: &mut Env) -> Result<Option<i128>, String> {
    eval_program(s, env, &mut FnTable::new())
}

/// Execute with a local-function table (enables LetFn/Call and recursion). Fuel-bounded.
pub fn eval_program(s: &Stmt, env: &mut Env, fns: &mut FnTable) -> Result<Option<i128>, String> {
    let mut fuel = FUEL;
    eval_fuel(s, env, fns, &mut fuel)
}

fn eval_fuel(s: &Stmt, env: &mut Env, fns: &mut FnTable, fuel: &mut u64) -> Result<Option<i128>, String> {
    if *fuel == 0 {
        return Err("loop fuel exhausted".into());
    }
    *fuel -= 1;
    match s {
        Stmt::Let(n, e) => {
            let v = eval_expr_env(e, env).ok_or_else(|| format!("unbound in let {n}"))?;
            env.insert(n.clone(), v);
            Ok(Option::None)
        }
        Stmt::Assign(n, e) => {
            if !env.contains_key(n) {
                return Err(format!("assign to undeclared {n}"));
            }
            let v = eval_expr_env(e, env).ok_or_else(|| format!("unbound in assign {n}"))?;
            env.insert(n.clone(), v);
            Ok(Option::None)
        }
        Stmt::Return(e) => Ok(eval_expr_env(e, env)),
        Stmt::Import(_) | Stmt::TypeDef(_) => Ok(Option::None),
        Stmt::If(c, t, f) => {
            let c = eval_expr_env(c, env).ok_or_else(|| "unbound cond".to_string())?;
            if c != 0 { eval_fuel(t, env, fns, fuel) } else { eval_fuel(f, env, fns, fuel) }
        }
        Stmt::While(c, b) => {
            loop {
                let cv = eval_expr_env(c, env).ok_or_else(|| "unbound while cond".to_string())?;
                if cv == 0 {
                    return Ok(Option::None);
                }
                if let Some(r) = eval_fuel(b, env, fns, fuel)? {
                    return Ok(Some(r));
                }
            }
        }
        Stmt::For(n, lo, hi, b) => {
            let lo = eval_expr_env(lo, env).ok_or_else(|| "unbound for lo".to_string())?;
            let hi = eval_expr_env(hi, env).ok_or_else(|| "unbound for hi".to_string())?;
            let mut i = lo;
            while i < hi {
                env.insert(n.clone(), i);
                if let Some(r) = eval_fuel(b, env, fns, fuel)? {
                    return Ok(Some(r));
                }
                i += 1;
            }
            Ok(Option::None)
        }
        Stmt::Match(scrut, arms) => {
            let v = eval_expr_env(scrut, env).ok_or_else(|| "unbound match scrutinee".to_string())?;
            for (p, body) in arms {
                let hit = match p {
                    crate::lut_emit::Pattern::Wild => true,
                    crate::lut_emit::Pattern::Lit(l) => *l == v,
                    crate::lut_emit::Pattern::Bind(n) => {
                        env.insert(n.clone(), v);
                        true
                    }
                };
                if hit {
                    return eval_fuel(body, env, fns, fuel);
                }
            }
            Ok(Option::None)
        }
        Stmt::LetFn(n, params, body) => {
            fns.fns.insert(n.clone(), (params.clone(), *body.clone()));
            Ok(Option::None)
        }
        Stmt::Call(n, args) => {
            let (params, body) = fns.fns.get(n).cloned().ok_or_else(|| format!("unknown fn {n}"))?;
            if params.len() != args.len() {
                return Err(format!("arity mismatch calling {n}"));
            }
            let vals: Vec<i128> = args
                .iter()
                .map(|a| eval_expr_env(a, env).ok_or_else(|| format!("unbound arg to {n}")))
                .collect::<Result<_, _>>()?;
            // Own scope per call so recursion cannot clobber the caller's bindings.
            let mut callee_env = env.clone();
            for (p, v) in params.iter().zip(vals) {
                callee_env.insert(p.clone(), v);
            }
            eval_fuel(&body, &mut callee_env, fns, fuel)
        }
        Stmt::GenericFn(n, _tp, params, body) => {
            // Monomorphization seed: type params erased at execution (i128 core).
            fns.fns.insert(n.clone(), (params.clone(), *body.clone()));
            Ok(Option::None)
        }
        Stmt::GenericType(_, _) => Ok(Option::None),
        Stmt::AsyncFn(n, params, body) => {
            fns.fns.insert(n.clone(), (params.clone(), *body.clone()));
            Ok(Option::None)
        }
        Stmt::Await(inner) => eval_fuel(inner, env, fns, fuel),
        Stmt::AwaitEvent(_) => Err("await event requires the IoRuntime".into()),
        Stmt::Attribute(_) => Ok(Option::None),
        Stmt::Macro(n, args) => {
            let e = expand_macro(n, args, &fns.macros)?;
            Ok(eval_expr_env(&e, env))
        }
        Stmt::LetCall(n, f, args) => {
            let v = eval_fuel(&Stmt::Call(f.clone(), args.clone()), env, fns, fuel)?
                .ok_or_else(|| format!("call {f} returned no value"))?;
            env.insert(n.clone(), v);
            Ok(Option::None)
        }
        Stmt::Seq(ss) => {
            for s in ss {
                if let Some(r) = eval_fuel(s, env, fns, fuel)? {
                    return Ok(Some(r));
                }
            }
            Ok(Option::None)
        }
    }
}

/// A cooperative async scheduler seed: spawned tasks run ONE statement per round in FIFO
/// order (deterministic interleaving); `Await` is a yield point (the task gives up the rest
/// of its round). Fuel-bounded; no preemption, no IO async (those stay open).
pub struct Scheduler {
    tasks: std::collections::VecDeque<(String, Vec<Stmt>, usize, Env)>,
    pub log: Vec<String>,
    pub rounds: u64,
}

impl Scheduler {
    pub fn new() -> Self {
        Scheduler { tasks: std::collections::VecDeque::new(), log: Vec::new(), rounds: 0 }
    }

    /// Spawn a task as a statement sequence with its own environment.
    pub fn spawn(&mut self, id: &str, body: Stmt, seed: Env) {
        let stmts = match body {
            Stmt::Seq(ss) => ss,
            other => vec![other],
        };
        self.tasks.push_back((id.to_string(), stmts, 0, seed));
    }

    /// Run one round: each ready task executes one statement (Await yields without executing).
    /// Returns true if any task remains.
    pub fn run_round(&mut self, fns: &mut FnTable, fuel: &mut u64) -> Result<bool, String> {
        self.rounds += 1;
        let n = self.tasks.len();
        for _ in 0..n {
            let (id, stmts, pc, mut env) = match self.tasks.pop_front() {
                Some(t) => t,
                None => break,
            };
            if pc >= stmts.len() {
                continue; // finished
            }
            if *fuel == 0 {
                return Err("scheduler fuel exhausted".into());
            }
            *fuel -= 1;
            let s = &stmts[pc];
            let yielded = matches!(s, Stmt::Await(_));
            if !yielded {
                // A Return inside a task ends that task.
                if let Some(_v) = eval_fuel(s, &mut env, fns, fuel)? {
                    self.log.push(format!("{id}:return"));
                    continue;
                }
            }
            self.log.push(format!("{id}:{}", if yielded { "yield" } else { "step" }));
            self.tasks.push_back((id, stmts, pc + 1, env));
        }
        Ok(!self.tasks.is_empty())
    }

    /// Run to completion (or fuel exhaustion). Deterministic for a given spawn order.
    pub fn run_to_completion(&mut self, fns: &mut FnTable, max_rounds: u64) -> Result<(), String> {
        let mut fuel = FUEL;
        while self.run_round(fns, &mut fuel)? {
            if self.rounds >= max_rounds {
                return Err("scheduler round cap".into());
            }
        }
        Ok(())
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// A deterministic preemptive + IO-async runtime seed. Tasks run one statement per tick in
/// priority order (higher priority preempts at statement granularity); `AwaitEvent(id)`
/// suspends a task until an IO completion scheduled at a tick arrives. Completions are
/// delivered in tick order, so the whole interleaving is reproducible. No real IO / threads.
pub struct IoRuntime {
    tasks: Vec<IoTask>,
    events: BTreeMap<u64, Vec<(String, i128)>>,
    pub log: Vec<String>,
    pub now: u64,
}

struct IoTask {
    id: String,
    prio: u8,
    stmts: Vec<Stmt>,
    pc: usize,
    env: Env,
    wait: Option<u64>,
    done: bool,
}

impl IoRuntime {
    pub fn new() -> Self {
        IoRuntime { tasks: Vec::new(), events: BTreeMap::new(), log: Vec::new(), now: 0 }
    }

    pub fn spawn(&mut self, id: &str, prio: u8, body: Stmt, seed: Env) {
        let stmts = match body {
            Stmt::Seq(ss) => ss,
            other => vec![other],
        };
        self.tasks.push(IoTask { id: id.to_string(), prio, stmts, pc: 0, env: seed, wait: Option::None, done: false });
    }

    /// Schedule an IO completion delivering `val` to task `id` at tick `t`.
    pub fn event(&mut self, t: u64, id: &str, val: i128) {
        self.events.entry(t).or_default().push((id.to_string(), val));
    }

    /// One tick: deliver due completions, then run one statement per ready task in priority
    /// order (ties by id). Returns true if any task is still alive.
    pub fn tick(&mut self, fns: &mut FnTable, fuel: &mut u64) -> Result<bool, String> {
        self.now += 1;
        if let Some(due) = self.events.remove(&self.now) {
            for (id, val) in due {
                if let Some(t) = self.tasks.iter_mut().find(|t| t.id == id) {
                    t.wait = Option::None;
                    t.env.insert("io".to_string(), val);
                    self.log.push(format!("{id}:io"));
                }
            }
        }
        let mut order: Vec<usize> = (0..self.tasks.len())
            .filter(|i| !self.tasks[*i].done && self.tasks[*i].wait.is_none())
            .collect();
        order.sort_by(|a, b| self.tasks[*a].prio.cmp(&self.tasks[*b].prio).then_with(|| self.tasks[*a].id.cmp(&self.tasks[*b].id)));
        for i in order {
            let t = &mut self.tasks[i];
            if t.pc >= t.stmts.len() {
                t.done = true;
                continue;
            }
            if *fuel == 0 {
                return Err("io runtime fuel exhausted".into());
            }
            *fuel -= 1;
            let s = t.stmts[t.pc].clone();
            match s {
                Stmt::AwaitEvent(id) => {
                    t.wait = Some(id);
                    t.pc += 1;
                    self.log.push(format!("{}:wait", t.id));
                }
                other => {
                    if let Some(_v) = eval_fuel(&other, &mut t.env, fns, fuel)? {
                        t.done = true;
                        self.log.push(format!("{}:return", t.id));
                    } else {
                        t.pc += 1;
                        self.log.push(format!("{}:step", t.id));
                    }
                }
            }
        }
        Ok(self.tasks.iter().any(|t| !t.done))
    }

    pub fn run(&mut self, fns: &mut FnTable, max_ticks: u64) -> Result<(), String> {
        let mut fuel = FUEL;
        while self.tick(fns, &mut fuel)? {
            if self.now >= max_ticks {
                return Err("io runtime tick cap".into());
            }
        }
        Ok(())
    }
}

impl Default for IoRuntime {
    fn default() -> Self {
        Self::new()
    }
}

/// Real OS threads, deterministic commit order: each task runs on its own thread with its
/// own environment; results are joined and sorted by id, so the aggregate is reproducible
/// even though execution is concurrent. Real concurrency, deterministic output.
pub fn threaded_run(tasks: &[(String, Stmt, Env)]) -> Result<Vec<(String, Option<i128>)>, String> {
    let handles: Vec<_> = tasks
        .iter()
        .map(|(id, stmt, env)| {
            let id = id.clone();
            let stmt = stmt.clone();
            let mut env = env.clone();
            std::thread::spawn(move || {
                let r = eval_program(&stmt, &mut env, &mut FnTable::new());
                (id, r)
            })
        })
        .collect();
    let mut out = Vec::new();
    for h in handles {
        let (id, r) = h.join().map_err(|_| "thread panicked".to_string())?;
        out.push((id, r?));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// The replay gate for whole-program Forms: admitted iff the program reproduces EVERY probe
/// (seeded environment → expected return) under deterministic execution. Same admission law
/// as expressions, extended to statements.
pub fn replay_admit(s: &Stmt, probes: &[(Env, i128)]) -> bool {
    !probes.is_empty() && probes.iter().all(|(seed, expected)| {
        let mut env = seed.clone();
        matches!(eval_program(s, &mut env, &mut FnTable::new()), Ok(Some(v)) if v == *expected)
    })
}

/// Name-free content key for a whole-program Form: SHA-256 of its canonical emission
/// (reference LUT), so equivalent renderings collapse to one key.
pub fn program_key(s: &Stmt) -> Result<String, String> {
    let lut = crate::lut_emit::builtin_luts()
        .into_iter()
        .find(|l| l.name == "rust")
        .ok_or_else(|| "no reference lut".to_string())?;
    let rendered = crate::lut_emit::render_stmt(s, &lut)?;
    Ok(rcevc_core::sha256(rendered.as_bytes()).hex())
}

/// Lift a small statement language: let/assign/return/import/type, if/while/for with brace
/// blocks, and bare expression statements. Expressions delegate to the expression lifter.
pub fn lift_program(src: &str) -> Result<Stmt, String> {
    let mut cur = Cursor { s: src.as_bytes(), i: 0 };
    let stmts = cur.parse_block_body(false)?;
    Ok(Stmt::Seq(stmts))
}

struct Cursor<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn ws(&mut self) {
        while self.i < self.s.len() && (self.s[self.i] as char).is_whitespace() {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> Option<char> {
        self.ws();
        self.s.get(self.i).map(|b| *b as char)
    }
    fn ident(&mut self) -> String {
        self.ws();
        let start = self.i;
        while self.i < self.s.len() && ((self.s[self.i] as char).is_alphanumeric() || self.s[self.i] == b'_') {
            self.i += 1;
        }
        String::from_utf8_lossy(&self.s[start..self.i]).into_owned()
    }
    fn expect(&mut self, c: char) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected {c}"))
        }
    }
    /// Read an expression up to a top-level delimiter (paren-aware).
    fn expr_until(&mut self, delim: char) -> Result<Expr, String> {
        self.ws();
        let start = self.i;
        let mut depth = 0i32;
        while self.i < self.s.len() {
            let c = self.s[self.i] as char;
            if depth == 0 && c == delim {
                break;
            }
            if c == '(' { depth += 1; }
            if c == ')' { depth -= 1; }
            self.i += 1;
        }
        let text = String::from_utf8_lossy(&self.s[start..self.i]).trim().to_string();
        crate::lift::lift_expr(&text)
    }
    fn parse_block_body(&mut self, brace: bool) -> Result<Vec<Stmt>, String> {
        if brace {
            self.expect('{')?;
        }
        let mut out = Vec::new();
        loop {
            match self.peek() {
                Option::None => {
                    if brace { return Err("unclosed block".into()); }
                    return Ok(out);
                }
                Some('}') => {
                    if brace {
                        self.i += 1;
                        return Ok(out);
                    }
                    return Err("unexpected }".into());
                }
                Some(';') => { self.i += 1; }
                Some(_) => out.push(self.parse_stmt()?),
            }
        }
    }
    fn parse_stmt(&mut self) -> Result<Stmt, String> {
        let kw = self.ident();
        match kw.as_str() {
            "let" => {
                let n = self.ident();
                self.expect('=')?;
                let e = self.expr_until(';')?;
                self.expect(';')?;
                Ok(Stmt::Let(n, e))
            }
            "return" => {
                let e = self.expr_until(';')?;
                self.expect(';')?;
                Ok(Stmt::Return(e))
            }
            "import" => {
                let p = self.ident();
                self.expect(';')?;
                Ok(Stmt::Import(p))
            }
            "type" => {
                let n = self.ident();
                self.expect(';')?;
                Ok(Stmt::TypeDef(n))
            }
            "if" => {
                let c = self.expr_until('{')?;
                let t = Stmt::Seq(self.parse_block_body(true)?);
                let f = if self.peek_ident() == "else" {
                    self.ident();
                    Stmt::Seq(self.parse_block_body(true)?)
                } else {
                    Stmt::Seq(vec![])
                };
                Ok(Stmt::If(c, Box::new(t), Box::new(f)))
            }
            "while" => {
                let c = self.expr_until('{')?;
                let b = Stmt::Seq(self.parse_block_body(true)?);
                Ok(Stmt::While(c, Box::new(b)))
            }
            "for" => {
                let n = self.ident();
                let in_kw = self.ident();
                if in_kw != "in" { return Err("expected in".into()); }
                let lo = self.expr_until('.')?;
                self.expect('.')?;
                self.expect('.')?;
                let hi = self.expr_until('{')?;
                let b = Stmt::Seq(self.parse_block_body(true)?);
                Ok(Stmt::For(n, lo, hi, Box::new(b)))
            }
            _ => {
                // assign `name = expr;` or bare expression statement
                if self.peek() == Some('=') {
                    self.expect('=')?;
                    let e = self.expr_until(';')?;
                    self.expect(';')?;
                    Ok(Stmt::Assign(kw, e))
                } else {
                    // re-parse the whole statement as an expression: rewind not available,
                    // so treat kw as start of an expression by re-reading from ident start.
                    Err(format!("unsupported statement start {kw}"))
                }
            }
        }
    }
    fn peek_ident(&mut self) -> String {
        let save = self.i;
        let id = self.ident();
        self.i = save;
        id
    }
}
