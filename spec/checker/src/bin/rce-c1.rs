//! rce-c1: differential test of the transition law (C1) and the T2 step
//! bound (C5) on random well-formed programs with random event logs.
//!
//! For each program: a random module (kinds, functions, 1 or 2 state
//! regions, 1 to 3 events, guarded and block-bodied steps) and a random
//! event log are generated from a seeded splitmix64 stream. The module must
//! be admitted by the reference checker (programs it refuses are counted and
//! replaced, never compared). Then:
//!   1. `rce run` (`cli::run`) prints the state after each event;
//!   2. the log is appended as an `example`, and the reference evaluator's
//!      observations (`emit::observe`) must contain the same per-step lines;
//!   3. the Rust, Python and JS emitters' outputs are run and must print the
//!      reference observations line for line. Each per-step line is also
//!      compared by SHA-256 (equal lines give equal hashes; the hash is what
//!      a receipt would carry).
//!   4. C5: the evaluator's steps per event are compared with
//!      B(P) * (1 + size(s) + size(e)), with B(P) computed statically from
//!      the AST in the evaluator's own step unit (one tick per expression
//!      evaluation, statement execution and call; loops multiply by their
//!      literal bound). This B is this harness's, not the Lean `B` of
//!      RCE/Bound.lean, which is over a core without calls.
//!
//! Usage: rce-c1 N SEED OUTDIR [JOBS]
//! The generator code (expressions and statements) is copied from
//! rce-fuzz (src/bin/fuzz.rs), which needs the C tree-sitter port; this
//! binary is pure Rust.
use rce_spec_check::ast::{Body, Expr, FnDecl, Item, Module, Stmt};
use rce_spec_check::emit::{Lang, emit, observe};
use rce_spec_check::interp::{Interp, LOOP_LIMIT, json_size};
use rce_spec_check::{cli, parse::parse};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

const INT: [&str; 5] = ["i64", "u32", "u8", "i32", "u64"];

/// splitmix64: a small deterministic generator (the run is reproducible
/// from its seed on every machine).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn random(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in [a, b], both ends included.
    fn randint(&mut self, a: u64, b: u64) -> u64 {
        let span = (b as u128) - (a as u128) + 1;
        let x = ((self.next() as u128) << 64 | self.next() as u128) % span;
        (a as u128 + x) as u64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn choice<'a, T>(&mut self, v: &'a [T]) -> &'a T {
        &v[self.below(v.len())]
    }
}

/// name -> (type, mutable), in insertion order.
#[derive(Clone, Default)]
struct Env(Vec<(String, (String, bool))>);
impl Env {
    fn set(&mut self, k: &str, v: (String, bool)) {
        if let Some(e) = self.0.iter_mut().find(|(n, _)| n == k) {
            e.1 = v
        } else {
            self.0.push((k.to_string(), v))
        }
    }
    fn get(&self, k: &str) -> Option<&(String, bool)> {
        self.0.iter().find(|(n, _)| n == k).map(|(_, v)| v)
    }
}

type Mark = (String, String, Option<String>);

struct Gen<'r> {
    r: &'r mut Rng,
    n: usize,
    marks: Vec<(String, Vec<Mark>)>,
    fns: Vec<(String, Vec<String>, String)>,
    kinds: Vec<(String, Vec<(String, bool)>)>,
    all_fns: Vec<String>,
}

fn alnum(s: &str) -> bool {
    let t: String = s.chars().filter(|c| *c != '_').collect();
    !t.is_empty() && t.chars().all(|c| c.is_alphanumeric())
}

impl<'r> Gen<'r> {
    fn fresh(&mut self, p: &str) -> String {
        self.n += 1;
        format!("{p}{}", self.n)
    }
    /// A placeholder that is `valid` in the program and `bad` in one mutant.
    fn mark(&mut self, kind: &str, valid: &str, bad: Option<&str>) -> String {
        let total: usize = self.marks.iter().map(|(_, v)| v.len()).sum();
        let idx = match self.marks.iter().position(|(k, _)| k == kind) {
            Some(i) => i,
            None => {
                self.marks.push((kind.to_string(), vec![]));
                self.marks.len() - 1
            }
        };
        let key = format!("@@{kind}{}@@", self.marks[idx].1.len() + 1 + total);
        self.marks[idx]
            .1
            .push((key.clone(), valid.to_string(), bad.map(|s| s.to_string())));
        key
    }
    fn kind(&self, t: &str) -> Option<&Vec<(String, bool)>> {
        self.kinds.iter().find(|(k, _)| k == t).map(|(_, v)| v)
    }

    // ---------- expressions ----------
    fn lit(&mut self, t: &str) -> String {
        if t == "bool" {
            return self.r.choice(&["true", "false"]).to_string();
        }
        let v = self.r.randint(0, if t == "u8" { 9 } else { 1000 });
        match self.r.below(3) {
            1 => format!("{v:#x}"),
            _ => v.to_string(),
        }
    }
    fn expr(&mut self, t: &str, env: &Env, d: i32) -> String {
        let names: Vec<String> = env
            .0
            .iter()
            .filter(|(_, (tt, _))| tt == t)
            .map(|(n, _)| n.clone())
            .collect();
        let mut opts = vec!["lit"];
        if !names.is_empty() {
            opts.extend(["name"; 3]);
        }
        if d > 0 {
            opts.extend(["bin", "paren", "ite"]);
            if t == "bool" {
                opts.extend(["cmp", "logic", "not"])
            } else {
                opts.extend(["bits", "cast"])
            }
            if self.fns.iter().any(|f| f.2 == t) {
                opts.push("call");
            }
            if t == "i64" || t == "i32" {
                opts.push("neg");
            }
        }
        let k = *self.r.choice(&opts);
        match k {
            "lit" => self.lit(t),
            "name" => self.r.choice(&names).clone(),
            "paren" => format!("({})", self.expr(t, env, d - 1)),
            "ite" => format!(
                "(if {} then {} else {})",
                self.expr("bool", env, d - 1),
                self.expr(t, env, d - 1),
                self.expr(t, env, d - 1)
            ),
            "call" => {
                let c: Vec<_> = self.fns.iter().filter(|f| f.2 == t).cloned().collect();
                let (name, ps, _) = self.r.choice(&c).clone();
                let a: Vec<String> = ps.iter().map(|p| self.expr(p, env, d - 1)).collect();
                format!("{name}({})", a.join(", "))
            }
            "neg" => format!("-{}", self.atom(t, env, d - 1)),
            "not" => format!("not {}", self.atom("bool", env, d - 1)),
            "cast" => {
                let s = *self.r.choice(&INT);
                format!("({} as {t})", self.atom(s, env, d - 1))
            }
            "cmp" => {
                let s = *self.r.choice(&INT);
                let a = self.arith(s, env, d - 1);
                let op = *self.r.choice(&["==", "!=", "<", "<=", ">", ">="]);
                format!("{a} {op} {}", self.arith(s, env, d - 1))
            }
            "logic" => {
                let a = self.atom("bool", env, d - 1);
                let op = *self.r.choice(&["and", "or"]);
                format!("{a} {op} {}", self.atom("bool", env, d - 1))
            }
            "bits" => {
                let a = self.atom(t, env, d - 1);
                let op = *self.r.choice(&["&", "|", "^"]);
                format!("{a} {op} {}", self.atom(t, env, d - 1))
            }
            _ => {
                if t == "bool" {
                    let a = self.atom("bool", env, d - 1);
                    return format!("{a} == {}", self.atom("bool", env, d - 1));
                }
                self.arith(t, env, d)
            }
        }
    }
    fn atom(&mut self, t: &str, env: &Env, d: i32) -> String {
        let e = self.expr(t, env, d);
        if alnum(&e) { e } else { format!("({e})") }
    }
    fn arith(&mut self, t: &str, env: &Env, d: i32) -> String {
        if d <= 0 {
            return self.atom(t, env, 0);
        }
        let op = *self.r.choice(&["+", "-", "*", "/", "%"]);
        let a = self.atom(t, env, d - 1);
        let rhs = if op == "/" || op == "%" {
            self.r.randint(1, 9).to_string()
        } else {
            self.atom(t, env, d - 1)
        };
        format!("{a} {op} {rhs}")
    }

    // ---------- statements ----------
    fn block(&mut self, env: &Env, ind: usize, n: u64, lp: bool) -> Vec<String> {
        let mut out = vec![];
        let mut env = env.clone();
        for _ in 0..n {
            out.extend(self.stmt(&mut env, ind, lp));
        }
        out
    }
    fn stmt(&mut self, env: &mut Env, ind: usize, lp: bool) -> Vec<String> {
        let sp = "    ".repeat(ind);
        let mut ks = vec![
            "let", "var", "if", "for", "while", "match", "assign", "assign",
        ];
        if lp {
            ks.push("break");
        }
        match *self.r.choice(&ks) {
            "let" => {
                let mut ts: Vec<&str> = INT.to_vec();
                ts.push("bool");
                let t = *self.r.choice(&ts);
                let x = self.fresh("l");
                let ann = if self.r.random() < 0.5 || t != "i64" {
                    format!(": {t}")
                } else {
                    String::new()
                };
                let line = format!("{sp}let {x}{ann} = {}", self.expr(t, env, 2));
                env.set(&x, (t.into(), false));
                vec![line]
            }
            "var" => {
                let t = *self.r.choice(&INT);
                let x = self.fresh("v");
                let init = self.expr(t, env, 1);
                env.set(&x, (t.into(), true));
                let m = self.mark("VAR", "var", Some("let")); // mutant: an assigned binding declared with let (W-PLACE)
                let op = *self.r.choice(&["+", "-", "*", "|", "^"]);
                vec![
                    format!("{sp}{m} {x}: {t} = {init}"),
                    format!("{sp}{x} {op}= {}", self.expr(t, env, 1)),
                ]
            }
            "assign" => {
                let vs: Vec<String> = env
                    .0
                    .iter()
                    .filter(|(_, (_, m))| *m)
                    .map(|(n, _)| n.clone())
                    .collect();
                if vs.is_empty() {
                    return vec![format!("{sp}pass")];
                }
                let x = self.r.choice(&vs).clone();
                let t = env.get(&x).unwrap().0.clone();
                let e = self.expr(&t, env, 1);
                let bad = self.mark("ASG", &e, Some("true")); // mutant: bool into an integer (W-TY-1)
                vec![format!("{sp}{x} = {bad}")]
            }
            "if" => {
                let mut out = vec![format!("{sp}if {}", self.expr("bool", env, 2))];
                let n = self.r.randint(1, 2);
                out.extend(self.block(env, ind + 1, n, lp));
                if self.r.random() < 0.4 {
                    out.push(format!("{sp}elif {}", self.expr("bool", env, 1)));
                    out.extend(self.block(env, ind + 1, 1, lp));
                }
                if self.r.random() < 0.5 {
                    out.push(format!("{sp}else"));
                    out.extend(self.block(env, ind + 1, 1, lp));
                }
                out
            }
            "for" => {
                let x = self.fresh("i");
                let mut e2 = env.clone();
                e2.set(&x, ("i64".into(), false));
                let mut out = vec![format!("{sp}for {x} in 0..{}", self.r.randint(1, 5))];
                let n = self.r.randint(1, 2);
                out.extend(self.block(&e2, ind + 1, n, true));
                out
            }
            "while" => {
                let mut out = vec![format!(
                    "{sp}while {} bound {}",
                    self.expr("bool", env, 1),
                    self.r.randint(1, 100)
                )];
                out.extend(self.block(env, ind + 1, 1, true));
                out
            }
            "break" => vec![
                format!("{sp}if {}", self.expr("bool", env, 1)),
                format!("{sp}    {}", self.r.choice(&["break", "continue"])),
            ],
            _ => {
                // match
                let ks: Vec<String> = env
                    .0
                    .iter()
                    .filter(|(_, (t, _))| self.kind(t).is_some())
                    .map(|(n, _)| n.clone())
                    .collect();
                if ks.is_empty() {
                    return vec![format!("{sp}pass")];
                }
                let x = self.r.choice(&ks).clone();
                let arms = self.kind(&env.get(&x).unwrap().0).unwrap().clone();
                let mut out = vec![format!("{sp}match {x}")];
                let drop = self.r.below(arms.len());
                for (i, (v, has)) in arms.iter().enumerate() {
                    let mut arm = if *has {
                        let y = self.fresh("p");
                        let mut e2 = env.clone();
                        e2.set(&y, ("i64".into(), false));
                        format!("{v}(v: {y}) -> {}", self.inline(&e2))
                    } else {
                        format!("{v} -> {}", self.inline(env))
                    };
                    if i == drop {
                        arm = self.mark("ARM", &arm, None);
                    } // mutant: arm deleted (W-MATCH-1)
                    out.push(format!("{sp}    {arm}"));
                }
                out
            }
        }
    }
    fn inline(&mut self, env: &Env) -> String {
        let vs: Vec<String> = env
            .0
            .iter()
            .filter(|(_, (_, m))| *m)
            .map(|(n, _)| n.clone())
            .collect();
        if !vs.is_empty() && self.r.random() < 0.6 {
            let x = self.r.choice(&vs).clone();
            let t = env.get(&x).unwrap().0.clone();
            return format!("{x} += {}", self.expr(&t, env, 1));
        }
        "pass".into()
    }
}

/// Names declared at indentation 1 of the body are visible to the final return.
fn visible(env: &Env, lines: &[String]) -> Env {
    let mut e = env.clone();
    for l in lines {
        let Some(s) = l.strip_prefix("    ").filter(|s| !s.starts_with("    ")) else {
            continue;
        };
        let w: Vec<&str> = s.split_whitespace().collect();
        if w.is_empty() {
            continue;
        }
        if w[0] == "let" {
            let ty = if w[1].ends_with(':') {
                w[2].to_string()
            } else {
                "i64".to_string()
            };
            e.set(w[1].trim_end_matches(':'), (ty, false));
        } else if w[0].starts_with("@@VAR") {
            e.set(w[1].trim_end_matches(':'), (w[2].to_string(), true));
        }
    }
    e
}

fn new_gen(r: &mut Rng) -> Gen<'_> {
    Gen { r, n: 0, marks: vec![], fns: vec![], kinds: vec![], all_fns: vec![] }
}

/// One random program (raw, with marks) and one random event log.
fn program(g: &mut Gen) -> (String, String) {
    let mut items = vec![];
    for _ in 0..g.r.randint(0, 2) {
        let k = g.fresh("K");
        let mut vs = vec![];
        for _ in 0..g.r.randint(2, 4) {
            let v = g.fresh("V");
            vs.push((v, g.r.random() < 0.5));
        }
        let body: Vec<String> = vs
            .iter()
            .map(|(v, h)| format!("    {v}{}", if *h { "(v: i64)" } else { "" }))
            .collect();
        items.push(format!("kind {k}\n{}", body.join("\n")));
        g.kinds.push((k, vs));
    }
    for _ in 0..g.r.randint(1, 4) {
        let name = g.fresh("f");
        g.all_fns.push(name.clone());
        let mut rts: Vec<&str> = INT.to_vec();
        rts.push("bool");
        let ret = g.r.choice(&rts).to_string();
        let mut pts: Vec<String> = rts.iter().map(|s| s.to_string()).collect();
        pts.extend(g.kinds.iter().map(|(k, _)| k.clone()));
        let mut ps = vec![];
        for _ in 0..g.r.randint(0, 3) {
            let a = g.fresh("a");
            ps.push((a, g.r.choice(&pts).clone()));
        }
        let mut env = Env::default();
        for (p, t) in &ps {
            env.set(p, (t.clone(), false));
        }
        let body = if g.r.random() < 0.4 {
            format!(" = {}", g.expr(&ret, &env, 3))
        } else {
            let n = g.r.randint(1, 4);
            let lines = g.block(&env, 1, n, false);
            let vis = visible(&env, &lines);
            format!("\n{}\n    return {}", lines.join("\n"), g.expr(&ret, &vis, 2))
        };
        let sig: Vec<String> = ps.iter().map(|(p, t)| format!("{p}: {t}")).collect();
        items.push(format!("fn {name}({}) -> {ret}{body}", sig.join(", ")));
        if ps.iter().all(|(_, t)| INT.contains(&t.as_str()) || t == "bool") {
            g.fns.push((name, ps.iter().map(|(_, t)| t.clone()).collect(), ret));
        }
    }
    // state, events, steps
    let t = *g.r.choice(&["i64", "u64", "i32", "u32"]);
    let err = g.fresh("Err");
    items.push(format!("error {err}"));
    let nreg = g.r.randint(1, 2);
    let mut regs = vec![];
    for _ in 0..nreg {
        let st = g.fresh("S");
        let init = g.r.randint(0, 50);
        // W-STEP-3: the arrow form needs a region with exactly one field
        let single = g.r.random() < 0.4;
        if single {
            items.push(format!("state {st}\n    n: {t} = {init}"));
        } else {
            items.push(format!("state {st}\n    n: {t} = {init}\n    c: i64 = 0\n    f: bool = false"));
        }
        regs.push((st, single));
    }
    let mut evs = vec![];
    for _ in 0..g.r.randint(1, 3) {
        let e = g.fresh("E");
        items.push(format!("event {e}(k: {t})"));
        evs.push(e);
    }
    for (ri, (st, single)) in regs.iter().enumerate() {
        let single = *single;
        for e in &evs {
            if ri > 0 && g.r.random() < 0.3 {
                continue;
            }
            let mut env = Env::default();
            env.set("n", (t.into(), true));
            if !single {
                env.set("c", ("i64".into(), true));
                env.set("f", ("bool".into(), false));
            }
            env.set("k", (t.into(), false));
            let guard = if g.r.random() < 0.3 {
                format!(" if {}", g.expr("bool", &env, 1))
            } else {
                String::new()
            };
            let mut s = if single && g.r.random() < 0.5 {
                format!("step {st} on {e}(k: k){guard} -> {}", g.expr(t, &env, 2))
            } else {
                let mut lines = vec![format!("step {st} on {e}(k: k){guard}")];
                if g.r.random() < 0.3 {
                    lines.push(format!("    if {}", g.expr("bool", &env, 1)));
                    lines.push(format!("        fail {err}"));
                }
                let n = g.r.randint(1, 3);
                lines.extend(g.block(&env, 1, n, false));
                if !single && g.r.random() < 0.5 {
                    lines.push(format!("    f = {}", g.expr("bool", &env, 1)));
                }
                if !single && g.r.random() < 0.5 {
                    lines.push("    c += 1".into());
                }
                lines.join("\n")
            };
            if !guard.is_empty() {
                s += &if single {
                    format!("\nstep {st} on {e}(k: _) -> n")
                } else {
                    format!("\nstep {st} on {e}(k: _)\n    pass")
                };
            }
            items.push(s);
        }
    }
    let nev = g.r.randint(1, 12);
    let mut log = vec![];
    for _ in 0..nev {
        let e = g.r.choice(&evs).clone();
        let v: String = match g.r.below(10) {
            0 if t.starts_with('i') => format!("-{}", g.r.randint(1, 1000)),
            1 => match t {
                "u64" => u64::MAX.to_string(),
                "i64" => i64::MAX.to_string(),
                "u32" => u32::MAX.to_string(),
                _ => i32::MAX.to_string(),
            },
            _ => g.r.randint(0, 1000).to_string(),
        };
        log.push(format!("{e}(k: {v})"));
    }
    (format!("{}\n", items.join("\n\n")), log.join(", "))
}

fn materialize(src: &str, marks: &[(String, Vec<Mark>)]) -> String {
    let mut s = src.to_string();
    for (_, lst) in marks {
        for (k, valid, _) in lst {
            s = s.replace(k.as_str(), valid);
        }
    }
    s
}

// ---------- static step bound (evaluator ticks) ----------

struct Bound<'a> {
    fns: HashMap<&'a str, &'a FnDecl>,
    memo: HashMap<String, u128>,
}

impl<'a> Bound<'a> {
    fn new(m: &'a Module) -> Self {
        let mut fns = HashMap::new();
        for i in &m.items {
            if let Item::Fn(f) = i {
                fns.insert(f.sig.name.as_str(), f);
            }
        }
        Bound { fns, memo: HashMap::new() }
    }
    fn lit(e: &Expr) -> Option<i128> {
        match e {
            Expr::Int(n, _) => Some(*n),
            _ => None,
        }
    }
    fn expr(&mut self, e: &Expr) -> Option<u128> {
        use Expr as E;
        let c = match e {
            E::Name(..) | E::Int(..) | E::Bool(..) | E::Float(..) | E::Unit(_) => 0,
            E::Neg(a, _) | E::Not(a, _) | E::BitNot(a, _) | E::Cast(a, _, _) => self.expr(a)?,
            E::Bin(_, a, b, _) => self.expr(a)? + self.expr(b)?,
            E::If(c, a, b, _) => self.expr(c)? + self.expr(a)?.max(self.expr(b)?),
            E::Call(f, args, _, _) => {
                let mut s = self.expr(f)? + 1;
                for a in args {
                    s += self.expr(&a.e)?;
                }
                let E::Name(n, _) = &**f else { return None };
                s + self.fun(n)?
            }
            _ => return None,
        };
        Some(1 + c)
    }
    fn fun(&mut self, n: &str) -> Option<u128> {
        if let Some(v) = self.memo.get(n) {
            return Some(*v);
        }
        let f = *self.fns.get(n)?;
        let v = self.body(&f.body)?;
        self.memo.insert(n.to_string(), v);
        Some(v)
    }
    fn body(&mut self, b: &Body) -> Option<u128> {
        match b {
            Body::Arrow(e) => self.expr(e),
            Body::Block(b) => self.block(b),
        }
    }
    fn block(&mut self, b: &[Stmt]) -> Option<u128> {
        let mut s = 0;
        for x in b {
            s += self.stmt(x)?;
        }
        Some(s)
    }
    fn stmt(&mut self, s: &Stmt) -> Option<u128> {
        let c = match s {
            Stmt::Expr(e) | Stmt::Emit(e) | Stmt::Fail(e) => self.expr(e)?,
            Stmt::Return(e, _) => match e {
                Some(e) => self.expr(e)?,
                None => 0,
            },
            // the place is evaluated once more for `op=` and once to range-check
            Stmt::Assign(_, t, v) => 2 * self.expr(t)? + self.expr(v)?,
            Stmt::Let(_, _, e, _) | Stmt::Var(_, _, e, _) => self.expr(e)?,
            Stmt::If(arms, els) => {
                let mut cs = 0;
                let mut mx = 0;
                for (c, b) in arms {
                    cs += self.expr(c)?;
                    mx = mx.max(self.block(b)?);
                }
                if let Some(b) = els {
                    mx = mx.max(self.block(b)?);
                }
                cs + mx
            }
            Stmt::While(c, b, body) => {
                let (bb, n) = match b {
                    Some(b) => (self.expr(b)?, Self::lit(b).unwrap_or(LOOP_LIMIT)),
                    None => (0, LOOP_LIMIT),
                };
                let n = n.max(0) as u128;
                bb + (n + 1) * self.expr(c)? + n * self.block(body)?
            }
            Stmt::For(_, it, body, _) => {
                let Expr::Bin(op, a, b, _) = it else { return None };
                let (a, b) = (Self::lit(a)?, Self::lit(b)?);
                let n = match op.as_str() {
                    ".." => (b - a).max(0),
                    "..=" => (b - a + 1).max(0),
                    _ => return None,
                } as u128;
                // the evaluator evaluates the iterated expression twice
                2 * self.expr(it)? + n * self.block(body)?
            }
            Stmt::Match(e, arms, _) => {
                let mut s = self.expr(e)?;
                let mut mx = 0;
                for a in arms {
                    if let Some(g) = &a.guard {
                        s += self.expr(g)?;
                    }
                    mx = mx.max(self.block(&a.body)?);
                }
                s + mx
            }
            Stmt::Break(_) | Stmt::Continue(_) | Stmt::Pass => 0,
            _ => return None,
        };
        Some(1 + c)
    }
    /// Upper bound on the ticks of one δ: every step of every region (the
    /// evaluator stops at the first match, so the sum over-approximates).
    fn delta(&mut self, m: &'a Module) -> Option<u128> {
        let mut s = 0;
        for i in &m.items {
            if let Item::Step { guard, body, .. } = i {
                if let Some(g) = guard {
                    s += self.expr(g)?;
                }
                s += self.body(body)?;
            }
        }
        Some(s)
    }
}

fn sha(s: &str) -> String {
    let h = Sha256::digest(s.as_bytes());
    h.iter().map(|b| format!("{b:02x}")).collect()
}

fn tool_version(t: &str, a: &str) -> String {
    Command::new(t)
        .arg(a)
        .output()
        .map(|o| {
            format!(
                "{}{}",
                String::from_utf8_lossy(&o.stdout).trim(),
                String::from_utf8_lossy(&o.stderr).trim()
            )
        })
        .unwrap_or_else(|e| format!("unavailable: {e}"))
}

/// Run one emitted file; Ok(stdout) or Err(why it did not run).
fn run_target(lang: Lang, dir: &Path, stem: &str) -> Result<String, String> {
    let src = dir.join(format!("{stem}.{}", lang.ext()));
    let out = match lang {
        Lang::Rust => {
            let exe = dir.join(format!("{stem}.bin"));
            let c = Command::new("rustc")
                .args(["--edition", "2024", "-C", "opt-level=0", "-A", "warnings", "-o"])
                .arg(&exe)
                .arg(&src)
                .output()
                .map_err(|e| e.to_string())?;
            if !c.status.success() {
                return Err(format!("rustc: {}", String::from_utf8_lossy(&c.stderr)));
            }
            let o = Command::new(&exe).output();
            let _ = std::fs::remove_file(&exe);
            o
        }
        Lang::Python => Command::new("python3").arg(&src).output(),
        Lang::Js => Command::new("node").arg(&src).output(),
        Lang::Scala => Command::new("scala-cli").args(["run", "-q", "--server=false", "--jvm", "21", "--scala", "3.10.0", "--scalac-option", "-nowarn"]).arg(&src).output(),
    }
    .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "exit {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The program with its log appended as an example. `-> true` is a
/// final-state condition; the example's verdict is not part of the
/// comparison (refused and trapping logs are compared too), so admission
/// of the base program is checked without it.
fn with_example(src: &str, log: &str) -> String {
    format!("{src}\nexample {log} -> true\n")
}

/// First line where the reference and a target differ.
fn first_diff(want: &[String], got: &str) -> Option<(usize, String, String)> {
    let g: Vec<&str> = got.lines().collect();
    for i in 0..want.len().max(g.len()) {
        let w = want.get(i).map(|s| s.as_str()).unwrap_or("<missing>");
        let x = g.get(i).copied().unwrap_or("<missing>");
        if sha(w) != sha(x) {
            return Some((i + 1, w.to_string(), x.to_string()));
        }
    }
    None
}

struct Case {
    id: usize,
    src: String,
    log: String,
    want: Vec<String>,
}

/// Does `src` + `log` still show a mismatch for `lang` (used to minimize)?
fn mismatches(src: &str, log: &str, lang: Lang, dir: &Path) -> bool {
    if !cli::diagnostics(src, &HashSet::new()).is_empty() {
        return false;
    }
    let full = with_example(src, log);
    let Ok(m) = parse(&full) else { return false };
    let want = observe(&m);
    let Ok(code) = emit(&m, lang) else { return false };
    let stem = "min";
    std::fs::write(dir.join(format!("{stem}.{}", lang.ext())), code).unwrap();
    match run_target(lang, dir, stem) {
        Ok(out) => first_diff(&want, &out).is_some(),
        Err(_) => true,
    }
}

/// Greedy line and event deletion until no single deletion keeps the mismatch.
fn minimize(src: &str, log: &str, lang: Lang, dir: &Path) -> (String, String) {
    let mut lines: Vec<String> = src.lines().map(String::from).collect();
    let mut evs: Vec<String> = split_log(log);
    loop {
        let mut changed = false;
        let mut i = 0;
        while i < lines.len() {
            let mut t = lines.clone();
            t.remove(i);
            let s = t.join("\n") + "\n";
            if mismatches(&s, &evs.join(", "), lang, dir) {
                lines = t;
                changed = true;
            } else {
                i += 1;
            }
        }
        let mut j = 0;
        while evs.len() > 1 && j < evs.len() {
            let mut t = evs.clone();
            t.remove(j);
            if mismatches(&(lines.join("\n") + "\n"), &t.join(", "), lang, dir) {
                evs = t;
                changed = true;
            } else {
                j += 1;
            }
        }
        if !changed {
            break;
        }
    }
    (lines.join("\n") + "\n", evs.join(", "))
}

fn split_log(log: &str) -> Vec<String> {
    let mut out = vec![];
    let mut depth = 0;
    let mut cur = String::new();
    for ch in log.chars() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
                continue;
            }
            _ => {}
        }
        cur.push(ch);
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n: usize = a.first().and_then(|s| s.parse().ok()).unwrap_or(100);
    let seed: u64 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let out = PathBuf::from(a.get(2).cloned().unwrap_or_else(|| "/tmp/rce-c1".into()));
    let jobs: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    std::fs::create_dir_all(&out).unwrap();
    std::panic::set_hook(Box::new(|_| {}));
    let mut rnd = Rng(seed);
    let mut stats: Vec<(String, u64)> = vec![];
    let inc = |stats: &mut Vec<(String, u64)>, k: &str, by: u64| {
        if let Some(e) = stats.iter_mut().find(|(n, _)| n == k) {
            e.1 += by
        } else {
            stats.push((k.to_string(), by))
        }
    };
    let mut cases: Vec<Case> = vec![];
    let mut c5_max_ratio = 0f64;
    let mut c5_max_steps = 0u64;
    let mut c5_viol = vec![];
    let mut generated = 0usize;
    let mut fail_log = vec![];
    while cases.len() < n {
        generated += 1;
        let (raw, log, marks) = {
            let mut g = new_gen(&mut rnd);
            let (raw, log) = program(&mut g);
            (raw, log, g.marks)
        };
        let src = materialize(&raw, &marks);
        inc(&mut stats, "programs generated", 1);
        let d = cli::diagnostics(&src, &HashSet::new());
        if !d.is_empty() {
            inc(&mut stats, "refused by checker (replaced, not compared)", 1);
            let rule = d[0].1.split_whitespace().find(|w| w.starts_with("W-")).unwrap_or("other").to_string();
            inc(&mut stats, &format!("  kept: {rule}"), 1);
            if fail_log.len() < 10 {
                fail_log.push(d[0].1.clone());
            }
            continue;
        }
        inc(&mut stats, "admitted", 1);
        let id = cases.len();
        // 1. rce run
        let run = match std::panic::catch_unwind(|| cli::run(&src, Some(&log))) {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                inc(&mut stats, "rce run error (events rejected)", 1);
                fail_log.push(format!("case {id}: rce run: {e}"));
                continue;
            }
            Err(_) => {
                inc(&mut stats, "rce run panicked", 1);
                continue;
            }
        };
        if run.kept {
            inc(&mut stats, "logs refused at some event", 1);
        }
        let full = with_example(&src, &log);
        let m = parse(&full).expect("parses with example");
        let want = observe(&m);
        // 2. rce run lines must appear as the example block of observe
        let start = want.iter().position(|l| l.starts_with("example ")).expect("example");
        let ex: Vec<&String> = want[start + 1..]
            .iter()
            .take_while(|l| !l.starts_with("example ") && !l.starts_with("main"))
            .collect();
        let ok = ex.len() == run.lines.len()
            && ex.iter().zip(&run.lines).all(|(a, b)| sha(a) == sha(b));
        if !ok {
            inc(&mut stats, "MISMATCH rce run vs reference observe", 1);
            fail_log.push(format!("case {id}: rce run differs from observe"));
        }
        inc(&mut stats, "per-step lines (rce run)", run.lines.len() as u64);
        // 4. C5
        {
            let mut it = Interp::new(&m);
            let mut b = Bound::new(&m);
            match b.delta(&m) {
                None => inc(&mut stats, "C5 bound not computable (skipped)", 1),
                Some(bp) => {
                    let mut s = it.initial_state().unwrap();
                    for (evn, args) in &m.items.iter().rev().find_map(|i| match i {
                        Item::Example(x) => Some(x.events.clone()),
                        _ => None,
                    }).unwrap() {
                        it.scopes = vec![HashMap::new()];
                        let Ok(Some(ev)) = it.construct(evn, args) else { break };
                        let ss: usize = s.iter().flat_map(|r| r.1.iter().map(|f| json_size(&f.1))).sum();
                        let se = json_size(&ev);
                        let before = it.steps_used();
                        let r = it.delta(&s, &ev);
                        let used = it.steps_used() - before;
                        inc(&mut stats, "C5 events measured", 1);
                        c5_max_steps = c5_max_steps.max(used);
                        let ratio = used as f64 / bp.max(1) as f64;
                        if ratio > c5_max_ratio {
                            c5_max_ratio = ratio;
                        }
                        let lim = bp * (1 + ss as u128 + se as u128);
                        if used as u128 > lim {
                            c5_viol.push(format!("case {id}: steps {used} > B(P)*(1+{ss}+{se}) = {lim}"));
                        }
                        if used as u128 > bp {
                            inc(&mut stats, "C5 events above B(P) alone (size factor used)", 1);
                            if std::env::var("C1_DEBUG").is_ok() {
                                eprintln!("case {id}: used {used} bp {bp} ev {}\n{full}", ev.show());
                            }
                        }
                        match r {
                            Ok((s2, _)) => s = s2,
                            Err(_) => break,
                        }
                    }
                }
            }
        }
        std::fs::write(out.join(format!("{id:05}.rce")), &full).unwrap();
        cases.push(Case { id, src, log, want });
    }
    // 3. targets, in parallel
    let langs = [Lang::Rust, Lang::Python, Lang::Js];
    #[allow(clippy::type_complexity)]
    let results: std::sync::Mutex<Vec<(usize, Lang, Result<Option<(usize, String, String)>, String>)>> =
        std::sync::Mutex::new(vec![]);
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for w in 0..jobs {
            let (cases, results, next, out) = (&cases, &results, &next, &out);
            sc.spawn(move || {
                let dir = out.join(format!("w{w}"));
                std::fs::create_dir_all(&dir).unwrap();
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if i >= cases.len() * 3 {
                        break;
                    }
                    let (c, lang) = (&cases[i / 3], langs[i % 3]);
                    let m = parse(&with_example(&c.src, &c.log)).unwrap();
                    let r = match emit(&m, lang) {
                        Err(e) => Err(format!("emit refused: {e}")),
                        Ok(code) => {
                            let stem = format!("p{}", c.id);
                            std::fs::write(dir.join(format!("{stem}.{}", lang.ext())), code).unwrap();
                            let r = run_target(lang, &dir, &stem).map(|o| first_diff(&c.want, &o));
                            let _ = std::fs::remove_file(dir.join(format!("{stem}.{}", lang.ext())));
                            r
                        }
                    };
                    let mut g = results.lock().unwrap();
                    g.push((c.id, lang, r));
                    if g.len() % 1500 == 0 {
                        eprintln!("{} target runs done", g.len());
                    }
                }
            });
        }
    });
    let mut res = results.into_inner().unwrap();
    res.sort_by_key(|x| (x.0, x.1 as u8));
    let mut bad = vec![];
    for (id, lang, r) in &res {
        let k = format!("{lang:?}");
        match r {
            Ok(None) => {
                inc(&mut stats, &format!("{k}: agree on every line"), 1);
                inc(&mut stats, &format!("{k}: lines compared"), cases[*id].want.len() as u64);
            }
            Ok(Some(d)) => {
                inc(&mut stats, &format!("{k}: MISMATCH"), 1);
                bad.push((*id, *lang, format!("line {}: want `{}` got `{}`", d.0, d.1, d.2)));
            }
            Err(e) => {
                inc(&mut stats, &format!("{k}: did not run"), 1);
                bad.push((*id, *lang, e.chars().take(600).collect()));
            }
        }
    }
    // minimize up to 3 failures per language
    let mdir = out.join("min");
    std::fs::create_dir_all(&mdir).unwrap();
    let mut per: HashMap<String, usize> = HashMap::new();
    let mut mins = vec![];
    for (id, lang, why) in &bad {
        let k = per.entry(format!("{lang:?}")).or_default();
        if *k >= 3 {
            continue;
        }
        *k += 1;
        let c = &cases[*id];
        let (s, l) = minimize(&c.src, &c.log, *lang, &mdir);
        let f = mdir.join(format!("{id:05}-{lang:?}.rce"));
        std::fs::write(&f, with_example(&s, &l)).unwrap();
        mins.push(format!("case {id} {lang:?}: {why}\n  minimized: {}", f.display()));
    }
    println!("rce-c1 N={n} SEED={seed} generated={generated}");
    println!("rustc: {}", tool_version("rustc", "--version"));
    println!("python3: {}", tool_version("python3", "--version"));
    println!("node: {}", tool_version("node", "--version"));
    for (k, v) in &stats {
        println!("{k:<55} {v}");
    }
    println!("C5 max steps per event: {c5_max_steps}");
    println!("C5 max steps / B(P): {c5_max_ratio:.4}");
    println!("C5 violations of B(P)*(1+size(s)+size(e)): {}", c5_viol.len());
    for v in c5_viol.iter().take(10) {
        println!("  {v}");
    }
    println!("target failures: {}", bad.len());
    for (id, lang, why) in bad.iter().take(20) {
        println!("  case {id} {lang:?}: {why}");
    }
    for m in &mins {
        println!("{m}");
    }
    if !fail_log.is_empty() {
        println!("sample checker refusals / run errors:");
        for f in fail_log.iter().take(20) {
            println!("  {f}");
        }
    }
    std::process::exit(if bad.is_empty() && c5_viol.is_empty() { 0 } else { 1 });
}
