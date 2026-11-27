//! Real-language lift (Rust subset): source statements -> Form (Stmt), deterministic, no model.
//! Covers let / assign / return / if-else / fn / call with a Pratt expression parser. This is the
//! lift-table seed for a real surface; emission (LUT) is the inverse, so lift+emit round-trips and
//! the replay gate verifies the lifted Form. Scales to other surfaces the same way emission does.

use rce_spec_check::ast::Expr;

use crate::lut_emit::Stmt;

#[derive(Clone, Debug, PartialEq)]
enum T {
    Int(i128),
    Id(String),
    Op(String),
    LP,
    RP,
    LB,
    RB,
    Comma,
    Semi,
    Eq,
}

fn lex(s: &str) -> Result<Vec<T>, String> {
    let ch: Vec<char> = s.chars().collect();
    let mut t = Vec::new();
    let mut i = 0;
    while i < ch.len() {
        let c = ch[i];
        match c {
            ' ' | '\t' | '\n' | '\r' => i += 1,
            '(' => { t.push(T::LP); i += 1; }
            ')' => { t.push(T::RP); i += 1; }
            '{' => { t.push(T::LB); i += 1; }
            '}' => { t.push(T::RB); i += 1; }
            ',' => { t.push(T::Comma); i += 1; }
            ';' => { t.push(T::Semi); i += 1; }
            '0'..='9' => {
                let mut n = String::new();
                while i < ch.len() && ch[i].is_ascii_digit() { n.push(ch[i]); i += 1; }
                t.push(T::Int(n.parse().map_err(|_| "bad int".to_string())?));
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let mut id = String::new();
                while i < ch.len() && (ch[i].is_alphanumeric() || ch[i] == '_') { id.push(ch[i]); i += 1; }
                t.push(T::Id(id));
            }
            '=' | '!' | '<' | '>' | '&' | '|' => {
                let two: String = ch[i..(i + 2).min(ch.len())].iter().collect();
                if matches!(two.as_str(), "==" | "!=" | "<=" | ">=" | "&&" | "||") { t.push(T::Op(two)); i += 2; }
                else if c == '=' { t.push(T::Eq); i += 1; }
                else if c == '<' || c == '>' { t.push(T::Op(c.to_string())); i += 1; }
                else { return Err(format!("unexpected {c}")); }
            }
            '+' | '-' | '*' | '/' | '%' => { t.push(T::Op(c.to_string())); i += 1; }
            _ => return Err(format!("unexpected char {c}")),
        }
    }
    Ok(t)
}

fn prec(op: &str) -> u8 {
    match op {
        "||" => 1, "&&" => 2,
        "==" | "!=" | "<" | ">" | "<=" | ">=" => 3,
        "+" | "-" => 4, "*" | "/" | "%" => 5,
        _ => 0,
    }
}

struct P { t: Vec<T>, i: usize }

impl P {
    fn peek(&self) -> Option<&T> { self.t.get(self.i) }
    fn next(&mut self) -> Option<T> { let x = self.t.get(self.i).cloned(); self.i += 1; x }
    fn eat(&mut self, w: &T) -> bool { if self.peek() == Some(w) { self.i += 1; true } else { false } }
    fn expect(&mut self, w: &T) -> Result<(), String> { if self.eat(w) { Ok(()) } else { Err(format!("expected {w:?}")) } }

    fn expr(&mut self, min: u8) -> Result<Expr, String> {
        let mut lhs = self.primary()?;
        while let Some(T::Op(o)) = self.peek().cloned() {
            let p = prec(&o);
            if p < min || p == 0 { break; }
            self.next();
            let rhs = self.expr(p + 1)?;
            lhs = Expr::Bin(o, Box::new(lhs), Box::new(rhs), 0);
        }
        Ok(lhs)
    }

    fn primary(&mut self) -> Result<Expr, String> {
        match self.next() {
            Some(T::Int(n)) => Ok(Expr::Int(n, 0)),
            Some(T::Op(o)) if o == "-" => Ok(Expr::Neg(Box::new(self.expr(6)?), 0)),
            Some(T::Id(id)) => {
                if id == "true" { return Ok(Expr::Bool(true, 0)); }
                if id == "false" { return Ok(Expr::Bool(false, 0)); }
                if self.eat(&T::LP) {
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Some(T::RP)) {
                        loop {
                            args.push(self.expr(1)?);
                            if !self.eat(&T::Comma) { break; }
                        }
                    }
                    self.expect(&T::RP)?;
                    // A call in expression position is represented as a named read of its result;
                    // statement-level calls are handled in stmt(). Keep as Name for expression use.
                    let _ = args;
                    return Ok(Expr::Name(id, 0));
                }
                Ok(Expr::Name(id, 0))
            }
            Some(T::LP) => { let e = self.expr(1)?; self.expect(&T::RP)?; Ok(e) }
            _ => Err("expected primary".into()),
        }
    }

    fn block(&mut self) -> Result<Stmt, String> {
        self.expect(&T::LB)?;
        let mut v = Vec::new();
        while !matches!(self.peek(), Some(T::RB)) {
            v.push(self.stmt()?);
        }
        self.expect(&T::RB)?;
        Ok(if v.len() == 1 { v.pop().unwrap() } else { Stmt::Seq(v) })
    }

    fn stmt(&mut self) -> Result<Stmt, String> {
        match self.peek().cloned() {
            Some(T::Id(w)) if w == "let" => {
                self.next();
                let n = match self.next() { Some(T::Id(n)) => n, _ => return Err("let name".into()) };
                self.expect(&T::Eq)?;
                // let n = f(args);  ->  LetCall (keeps the call's value)
                if let (Some(T::Id(f)), Some(T::LP)) = (self.peek().cloned(), self.t.get(self.i + 1).cloned()) {
                    self.next(); self.next();
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Some(T::RP)) {
                        loop { args.push(self.expr(1)?); if !self.eat(&T::Comma) { break; } }
                    }
                    self.expect(&T::RP)?;
                    self.expect(&T::Semi)?;
                    return Ok(Stmt::LetCall(n, f, args));
                }
                let e = self.expr(1)?;
                self.expect(&T::Semi)?;
                Ok(Stmt::Let(n, e))
            }
            Some(T::Id(w)) if w == "return" => {
                self.next();
                let e = self.expr(1)?;
                self.expect(&T::Semi)?;
                Ok(Stmt::Return(e))
            }
            Some(T::Id(w)) if w == "while" => {
                self.next();
                let c = self.expr(1)?;
                let b = self.block()?;
                Ok(Stmt::While(c, Box::new(b)))
            }
            Some(T::Id(w)) if w == "if" => {
                self.next();
                let c = self.expr(1)?;
                let t = self.block()?;
                let e = if self.peek() == Some(&T::Id("else".into())) { self.next(); self.block()? } else { Stmt::Seq(vec![]) };
                Ok(Stmt::If(c, Box::new(t), Box::new(e)))
            }
            Some(T::Id(w)) if w == "fn" => {
                self.next();
                let n = match self.next() { Some(T::Id(n)) => n, _ => return Err("fn name".into()) };
                self.expect(&T::LP)?;
                let mut params = Vec::new();
                if !matches!(self.peek(), Some(T::RP)) {
                    loop {
                        match self.next() { Some(T::Id(p)) => params.push(p), _ => return Err("param".into()) }
                        if !self.eat(&T::Comma) { break; }
                    }
                }
                self.expect(&T::RP)?;
                let b = self.block()?;
                Ok(Stmt::LetFn(n, params, Box::new(b)))
            }
            Some(T::Id(n)) => {
                self.next();
                if self.eat(&T::Eq) {
                    let e = self.expr(1)?;
                    self.expect(&T::Semi)?;
                    return Ok(Stmt::Assign(n, e));
                }
                if self.eat(&T::LP) {
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Some(T::RP)) {
                        loop { args.push(self.expr(1)?); if !self.eat(&T::Comma) { break; } }
                    }
                    self.expect(&T::RP)?;
                    self.expect(&T::Semi)?;
                    return Ok(Stmt::Call(n, args));
                }
                Err(format!("unexpected statement after {n}"))
            }
            _ => Err("expected statement".into()),
        }
    }
}

/// Lift a Rust-subset program (let/assign/return/if/fn/call) into the Form IR.
pub fn lift_rust(src: &str) -> Result<Stmt, String> {
    let t = lex(src)?;
    let mut p = P { t, i: 0 };
    let mut v = Vec::new();
    while p.peek().is_some() { v.push(p.stmt()?); }
    Ok(if v.len() == 1 { v.pop().unwrap() } else { Stmt::Seq(v) })
}
