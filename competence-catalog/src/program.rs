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

/// Local functions (closure seed): name → (params, body).
pub type FnTable = BTreeMap<String, (Vec<String>, Stmt)>;

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
            fns.insert(n.clone(), (params.clone(), *body.clone()));
            Ok(Option::None)
        }
        Stmt::Call(n, args) => {
            let (params, body) = fns.get(n).cloned().ok_or_else(|| format!("unknown fn {n}"))?;
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
            fns.insert(n.clone(), (params.clone(), *body.clone()));
            Ok(Option::None)
        }
        Stmt::GenericType(_, _) => Ok(Option::None),
        Stmt::AsyncFn(n, params, body) => {
            fns.insert(n.clone(), (params.clone(), *body.clone()));
            Ok(Option::None)
        }
        Stmt::Await(inner) => eval_fuel(inner, env, fns, fuel),
        Stmt::Attribute(_) => Ok(Option::None),
        Stmt::Macro(_, _) => Err("macro expansion is an open frontier".into()),
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
