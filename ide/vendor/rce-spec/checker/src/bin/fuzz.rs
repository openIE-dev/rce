//! rce-fuzz: grammar-level random program generation for RCE (spec
//! sections 2 and 5). Rust port of the former fuzz/gen.py.
//!
//! Generates random, type-directed programs from the grammar's productions
//! (items: kind, error, state, event, step, fn; statements: let, var,
//! op-assign, if/elif/else, for, while-bound, match, return; expressions:
//! literals, names, unary and binary operators at every precedence level,
//! if-then-else, calls; and the draft 0.4/0.5 constructs: extern, arena
//! bound, structural recursion, f32, lane-wise array arithmetic, device
//! regions, byte-indexed text, full-range u64, select, linear structs,
//! LOOP_LIMIT and iterators). Each program must
//!   1. parse with the tree-sitter port of the grammar (generated with
//!      `conflicts: []`, so LR(1) with no unresolved conflict: one parse)
//!      and with the LALR(1) tables of spec section 3 (crate::lalr),
//!   2. parse with the Rust reference parser and pass the reference checker.
//!
//! Each program also yields targeted near-misses (one small edit that breaks
//! one named rule); every near-miss must be refused (by both parsers for
//! syntax mutants, by the checker for the others), and tree-sitter and the
//! LALR(1) tables must agree on every syntax mutant.
//!
//! Usage: rce-fuzz [N] [SEED] [STATS.json]
//! With STATS.json, counts are rewritten there every 100 programs and each
//! failure is written to STATS.json.failures/ as soon as it is found.
use rce_spec_check::{check_one, ts::TsParser};
use regex::Regex;
use std::path::Path;

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

    // ---------- items ----------
    fn program(&mut self) -> String {
        let mut items = vec![];
        if self.r.random() < 0.5 {
            items.push(format!("module m{}.gen", self.r.randint(0, 99)));
        }
        for _ in 0..self.r.randint(0, 2) {
            let k = self.fresh("K");
            let mut vs = vec![];
            for _ in 0..self.r.randint(2, 4) {
                let v = self.fresh("V");
                vs.push((v, self.r.random() < 0.5));
            }
            let body: Vec<String> = vs
                .iter()
                .map(|(v, h)| format!("    {v}{}", if *h { "(v: i64)" } else { "" }))
                .collect();
            items.push(format!("kind {k}\n{}", body.join("\n")));
            self.kinds.push((k, vs));
        }
        for _ in 0..self.r.randint(1, 4) {
            let name = self.fresh("f");
            self.all_fns.push(name.clone());
            let mut rts: Vec<&str> = INT.to_vec();
            rts.push("bool");
            let ret = self.r.choice(&rts).to_string();
            let mut pts: Vec<String> = rts.iter().map(|s| s.to_string()).collect();
            pts.extend(self.kinds.iter().map(|(k, _)| k.clone()));
            let mut ps = vec![];
            for _ in 0..self.r.randint(0, 3) {
                let a = self.fresh("a");
                ps.push((a, self.r.choice(&pts).clone()));
            }
            let mut env = Env::default();
            for (p, t) in &ps {
                env.set(p, (t.clone(), false));
            }
            let rt = self.mark("RET", &ret, Some("text")); // mutant: wrong declared return type (W-TY-1)
            let body = if self.r.random() < 0.4 {
                format!(" = {}", self.expr(&ret, &env, 3))
            } else {
                let n = self.r.randint(1, 4);
                let lines = self.block(&env, 1, n, false);
                let vis = visible(&env, &lines);
                let ret_line = format!("    return {}", self.expr(&ret, &vis, 2));
                format!("\n{}\n{ret_line}", lines.join("\n"))
            };
            let sig: Vec<String> = ps.iter().map(|(p, t)| format!("{p}: {t}")).collect();
            items.push(format!("fn {name}({}) -> {rt}{body}", sig.join(", ")));
            if ps
                .iter()
                .all(|(_, t)| INT.contains(&t.as_str()) || t == "bool")
            {
                self.fns
                    .push((name, ps.iter().map(|(_, t)| t.clone()).collect(), ret));
            }
        }
        if self.r.random() < 0.7 {
            let st = self.fresh("S");
            let t = *self.r.choice(&["i64", "u64"]);
            items.push(format!("state {st}\n    n: {t} = 0"));
            let mut evs = vec![];
            for _ in 0..self.r.randint(1, 3) {
                evs.push(self.fresh("E"));
            }
            for e in &evs {
                items.push(format!("event {e}(k: {t})"));
            }
            for e in &evs {
                let g = if self.r.random() < 0.3 {
                    " if k > 0"
                } else {
                    ""
                };
                let rhs = *self.r.choice(&["n + k", "n", "k", "n * 2"]);
                let mut step = format!("step {st} on {e}(k: k){g} -> {rhs}");
                if !g.is_empty() {
                    step += &format!("\nstep {st} on {e}(k: _) -> n");
                }
                let m = self.mark("STEP", &step, None); // mutant: event left unhandled (W-STEP-2)
                items.push(m);
            }
        }
        let v5 = self.v05_items();
        items.extend(v5);
        let g = self.fresh("g");
        let nm = self.mark(
            "NAME",
            "",
            Some(&format!("fn {g}() -> i64 = undefined_name\n\n")),
        ); // W-NAME-1
        let f0 = self.all_fns[0].clone();
        let dup = self.mark("DUP", "", Some(&format!("\n\nfn {f0}() -> bool = true"))); // W-NAME-2
        format!("{nm}{}{dup}\n", items.join("\n\n"))
    }
    /// Draft 0.4 and 0.5 constructs, each with a near-miss for its rule.
    fn v05_items(&mut self) -> Vec<String> {
        let mut out = vec![];
        if self.r.random() < 0.3 {
            let v = format!(
                "const LOOP_LIMIT: u64 = {}",
                self.r.randint(1000, 5_000_000)
            );
            out.push(self.mark("LOOP", &v, Some("const LOOP_LIMIT: i64 = 1000"))); // W-LOOP-1
        }
        if self.r.random() < 0.4 {
            // f32 (W-TY-1: f64 literal mixed in)
            let f = self.fresh("h");
            let v = format!("({}.5 as f32)", self.r.randint(0, 9));
            let m = self.mark("F32", &v, Some("1.5"));
            out.push(format!(
                "fn {f}(a: f32, b: f32) -> f32\n    return a * b + {m}"
            ));
        }
        if self.r.random() < 0.4 {
            // lane-wise arithmetic (W-SIMD-1: lengths differ)
            let f = self.fresh("h");
            let n = self.r.randint(2, 6);
            let t = *self.r.choice(&["i32", "i64", "f64"]);
            let m = self.mark("SIMD", &n.to_string(), Some(&(n + 1).to_string()));
            let op = *self.r.choice(&["+", "-", "*"]);
            out.push(format!(
                "fn {f}(a: [{t}; {n}], b: [{t}; {m}]) -> [{t}; {n}] = a {op} b"
            ));
        }
        if self.r.random() < 0.3 {
            // device region (W-DEV-1: a growable value inside)
            let f = self.fresh("h");
            let d = self.fresh("d");
            let b = self.r.randint(64, 4096);
            let m = self.mark("DEV", ": [i64; 2]", Some(": List[i64]"));
            out.push(format!("fn {f}(a: i64) -> i64\n    var s: i64 = 0\n    with device {d} bound {b}\n        let x{m} = [a, a]\n        for y in x\n            s += y\n    return s"));
        }
        if self.r.random() < 0.3 {
            // byte-indexed text (W-TY-1: a byte is not text)
            let f = self.fresh("h");
            let m = self.mark("TXT", "u8", Some("text"));
            out.push(format!("fn {f}(s: &text, i: usize) -> {m} = s[i]"));
        }
        if self.r.random() < 0.3 {
            // full-range u64
            let f = self.fresh("h");
            let k = self.r.randint(1u64 << 63, u64::MAX);
            out.push(format!(
                "fn {f}(a: u64) -> u64 = a / 2 + {}\n\ncheck {f}(0) == {}",
                k / 2,
                k / 2
            ));
        }
        if self.r.random() < 0.3 {
            // iterators (W-ITER-1: an iterator stored in a let)
            let g = self.fresh("it");
            let f = self.fresh("h");
            out.push(format!("fn {g}(n: u64) -> iter u64\n    var i: u64 = 0\n    while i < n bound 100\n        yield i\n        i += 1"));
            let m = self.mark("ITER", "", Some(&format!("    let t = {g}(n)\n")));
            let lim = self.r.randint(0, 50);
            out.push(format!("fn {f}(n: u64) -> u64\n    var s: u64 = 0\n{m}    for x in {g}(n)\n        if x > {lim}\n            break\n        s += x\n    return s"));
        }
        if self.r.random() < 0.3 {
            // linear handle (W-LIN-1: never consumed)
            let h = self.fresh("H");
            let c = self.fresh("c_close");
            let f = self.fresh("t");
            out.push(format!("linear struct {h}\n    fd: i32"));
            out.push(format!("extern \"C\" fn {c}(h: {h}) -> i32"));
            let m = self.mark("LIN", &format!("{c}(h)"), Some("h.fd"));
            out.push(format!(
                "fn {f}(h: {h}, out: chan i32)\n    let r = {m}\n    out <- r"
            ));
        }
        if self.r.random() < 0.3 {
            // select with the rotation rule (W-TY-1: after needs u64)
            let f = self.fresh("t");
            let b = self.r.randint(1, 100);
            let after = self.r.randint(1, 9).to_string();
            let m = self.mark("SEL", &after, Some("\"soon\""));
            out.push(format!("fn {f}(a: chan u64, b: chan u64, out: chan u64)\n    loop bound {b}\n        select\n            recv some(x) from a -> out <- x\n            recv some(y) from b -> out <- y\n            recv none from a -> break\n            after {m} -> pass"));
        }
        if self.r.random() < 0.2 {
            // arena with a byte bound (0.4)
            let f = self.fresh("h");
            let b = self.r.randint(64, 4096);
            out.push(format!("fn {f}(n: u64) -> u64\n    var k: u64 = 0\n    with arena a{f} bound {b}\n        let xs: List[u64] @a{f} = [n, n]\n        k = xs.len() as u64\n    return k"));
        }
        if self.r.random() < 0.2 {
            // structural recursion (0.4; W-REC-1: recursing on the same value)
            let k = self.fresh("T");
            let lf = self.fresh("Lf");
            let nd = self.fresh("Nd");
            let f = self.fresh("h");
            out.push(format!(
                "kind {k}\n    {lf}(v: i64)\n    {nd}(l: Box[{k}], r: Box[{k}])"
            ));
            let m = self.mark("REC", "l", Some("t"));
            out.push(format!("fn {f}(t: &{k}) -> i64\n    match t\n        {lf}(v: v) -> return v\n        {nd}(l: l, r: r) -> return {f}({m}) + {f}(r)"));
        }
        if self.r.random() < 0.2 {
            // extern (0.4; W-EFF-1: called from a pure fn)
            let c = self.fresh("c_fn");
            let f = self.fresh("t");
            out.push(format!("extern \"C\" fn {c}(n: u32) -> u32"));
            let m = self.mark("FFI", ", out: chan u32", Some(""));
            out.push(format!("fn {f}(n: u32{m}) -> u32 = {c}(n)"));
        }
        out
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

fn materialize(src: &str, marks: &[(String, Vec<Mark>)], which: Option<&str>) -> String {
    let mut s = src.to_string();
    for (_, lst) in marks {
        for (k, valid, bad) in lst {
            let v = if Some(k.as_str()) != which {
                valid.as_str()
            } else {
                bad.as_deref().unwrap_or("")
            };
            s = s.replace(k.as_str(), v);
        }
    }
    s
}

fn syntax_mutants(s: &str) -> Vec<(&'static str, Option<String>)> {
    let ind = Regex::new(r"(\n    \S[^\n]*\n)    (\S)").unwrap();
    let fname = Regex::new(r"\bfn f\d+\(").unwrap();
    let kw = ["yield", "iter", "linear", "device"][s.chars().count() % 4];
    vec![
        (
            "bad_indent",
            ind.is_match(s)
                .then(|| ind.replacen(s, 1, "${1}   ${2}").into_owned()),
        ),
        (
            "keyword_as_name",
            fname
                .is_match(s)
                .then(|| fname.replacen(s, 1, "fn match(").into_owned()),
        ),
        (
            "new_keyword_as_name",
            fname.is_match(s).then(|| {
                fname
                    .replacen(s, 1, format!("fn {kw}(").as_str())
                    .into_owned()
            }),
        ),
        (
            "missing_arrow",
            s.contains(") -> ").then(|| s.replacen(") -> ", ") ", 1)),
        ),
        (
            "unclosed_paren",
            s.contains('(').then(|| s.replacen('(', "((", 1)),
        ),
    ]
}

#[derive(Default)]
struct Stats(Vec<(String, u64)>);
impl Stats {
    fn inc(&mut self, k: &str) {
        if let Some(e) = self.0.iter_mut().find(|(n, _)| n == k) {
            e.1 += 1
        } else {
            self.0.push((k.to_string(), 1))
        }
    }
    fn json(&self) -> serde_json::Value {
        serde_json::Value::Object(
            self.0
                .iter()
                .map(|(k, v)| (k.clone(), (*v).into()))
                .collect(),
        )
    }
}

/// The checker's verdict; a panic in the checker counts as a refusal, as
/// a crashed `rce-check` process did for the former driver.
fn checks(src: &str) -> Result<(), String> {
    match std::panic::catch_unwind(|| check_one(src)) {
        Ok(r) => r,
        Err(_) => Err("checker panicked".into()),
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n: usize = a.first().and_then(|s| s.parse().ok()).unwrap_or(200);
    let seed: u64 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let statsf = a.get(2).cloned();
    std::panic::set_hook(Box::new(|_| {}));
    let mut rnd = Rng(seed);
    let fd = match &statsf {
        Some(s) => format!("{s}.failures"),
        None => std::env::temp_dir()
            .join(format!("rce-fuzz-{}-failures", std::process::id()))
            .display()
            .to_string(),
    };
    std::fs::create_dir_all(&fd).unwrap();
    let mut stats = Stats::default();
    let mut fails: Vec<(String, String)> = vec![];
    let fail = |fails: &mut Vec<(String, String)>, why: String, src: String| {
        std::fs::write(
            format!("{fd}/{:05}.rce", fails.len()),
            format!("# {}\n{src}", why.replace('\n', " ")),
        )
        .unwrap();
        fails.push((why, src));
    };
    let dump = |i: usize, stats: &Stats, nf: usize| {
        if let Some(s) = &statsf {
            let j =
                serde_json::json!({"seed": seed, "done": i, "stats": stats.json(), "failures": nf});
            std::fs::write(
                format!("{s}.tmp"),
                serde_json::to_string_pretty(&j).unwrap(),
            )
            .unwrap();
            std::fs::rename(format!("{s}.tmp"), s).unwrap();
        }
    };
    let mut ts = TsParser::new();
    let lalr = {
        let md = std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../book/03-syntax.md"),
        )
        .expect("book/03-syntax.md");
        rce_spec_check::lalr::Lalr::build(
            rce_spec_check::lalr::book_grammar(&md).expect("section 3 grammar"),
        )
    };
    assert!(
        lalr.conflicts.is_empty(),
        "LALR(1) conflicts: {:?}",
        lalr.conflicts
    );
    for i in 0..n {
        if i % 100 == 0 {
            dump(i, &stats, fails.len());
        }
        let (raw, marks) = {
            let mut g = Gen {
                r: &mut rnd,
                n: 0,
                marks: vec![],
                fns: vec![],
                kinds: vec![],
                all_fns: vec![],
            };
            let raw = g.program();
            (raw, g.marks)
        };
        let src = materialize(&raw, &marks, None);
        stats.inc("programs");
        if !ts.accepts(&src) {
            fail(&mut fails, "tree-sitter refused".into(), src);
            continue;
        }
        stats.inc("tree-sitter parsed (LR(1), no conflicts: one parse)");
        if !lalr.accepts(&src) {
            fail(&mut fails, "LALR(1) (section 3) refused".into(), src);
            continue;
        }
        stats.inc("LALR(1) from section 3 parsed");
        if let Err(e) = rce_spec_check::parse::parse(&src) {
            fail(&mut fails, format!("Rust parser refused: {e}"), src);
            continue;
        }
        stats.inc("Rust parser parsed");
        if let Err(e) = checks(&src) {
            fail(
                &mut fails,
                format!("checker refused valid program: {e}"),
                src,
            );
            continue;
        }
        stats.inc("rce-check accepted");
        for (kind, lst) in &marks {
            if (kind == "NAME" || kind == "DUP") && rnd.random() < 0.5 {
                continue;
            }
            for (k, _, _) in lst.iter().take(2) {
                let m = materialize(&raw, &marks, Some(k));
                stats.inc("near-misses (rule mutants)");
                if checks(&m).is_ok() {
                    fail(&mut fails, format!("mutant {kind} accepted"), m);
                } else {
                    stats.inc(&format!("  refused: {kind}"));
                }
            }
        }
        for (name, m) in syntax_mutants(&src) {
            let Some(m) = m else { continue };
            stats.inc("near-misses (syntax mutants)");
            let t_ok = ts.accepts(&m);
            let l_ok = lalr.accepts(&m);
            if t_ok != l_ok {
                fail(
                    &mut fails,
                    format!("syntax mutant {name}: tree-sitter={t_ok} lalr={l_ok} disagree"),
                    m.clone(),
                );
                continue;
            }
            stats.inc("  tree-sitter and LALR(1) agree");
            let r_ok = checks(&m).is_ok();
            if !t_ok && !r_ok {
                stats.inc(&format!("  refused by tree-sitter and rce-check: {name}"));
            } else {
                fail(
                    &mut fails,
                    format!("syntax mutant {name} accepted (tree-sitter={t_ok} rust={r_ok})"),
                    m,
                );
            }
        }
    }
    dump(n, &stats, fails.len());
    for (k, v) in &stats.0 {
        println!("{k:<55} {v}");
    }
    println!("failures: {}", fails.len());
    if !fails.is_empty() {
        println!("failing programs written to {fd}");
    }
    for (why, s) in fails.iter().take(5) {
        println!("--- {why}\n{s}");
    }
    std::process::exit(if fails.is_empty() { 0 } else { 1 });
}
