//! Full-language lift (seed beyond arithmetic): comparisons, logic, conditionals, floats,
//! booleans, and prefix (Lisp) surfaces, lifted into the rich Form IR (If/Not/Bin/Float/Bool).
//!
//! The arithmetic lifter (`lift.rs`) is the C/Python shared core; this extends the grammar table
//! to the constructs that make lift "full-language": `== != < > <= >=`, `&& || and or not`,
//! `if c then t else e`, floats, `true/false`, and prefix applicative `(+ a b)`. Per-language
//! tree-sitter grammars (in `rce/tree-sitter/`) scale this to every surface the same way emission
//! scales; this is the deterministic grammar-table seed, not a model call.

use rce_spec_check::ast::Expr;

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Int(i128),
    Float(f64),
    Ident(String),
    Op(String),
    LParen,
    RParen,
}

fn lex(src: &str) -> Result<Vec<Tok>, String> {
    let mut toks = Vec::new();
    let mut ch: Vec<char> = src.chars().collect();
    let mut i = 0;
    while i < ch.len() {
        let c = ch[i];
        match c {
            ' ' | '\t' | '\n' | '\r' => i += 1,
            '(' => { toks.push(Tok::LParen); i += 1; }
            ')' => { toks.push(Tok::RParen); i += 1; }
            '0'..='9' => {
                let mut n = String::new();
                while i < ch.len() && (ch[i].is_ascii_digit() || ch[i] == '.') { n.push(ch[i]); i += 1; }
                if n.contains('.') {
                    toks.push(Tok::Float(n.parse().map_err(|_| "bad float".to_string())?));
                } else {
                    toks.push(Tok::Int(n.parse().map_err(|_| "bad int".to_string())?));
                }
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let mut id = String::new();
                while i < ch.len() && (ch[i].is_alphanumeric() || ch[i] == '_') { id.push(ch[i]); i += 1; }
                toks.push(Tok::Ident(id));
            }
            '=' | '!' | '<' | '>' | '&' | '|' => {
                let two: String = ch[i..(i + 2).min(ch.len())].iter().collect();
                if matches!(two.as_str(), "==" | "!=" | "<=" | ">=" | "&&" | "||") {
                    toks.push(Tok::Op(two)); i += 2;
                } else if c == '<' || c == '>' {
                    toks.push(Tok::Op(c.to_string())); i += 1;
                } else {
                    return Err(format!("unexpected {c}"));
                }
            }
            '+' | '-' | '*' | '/' | '%' => { toks.push(Tok::Op(c.to_string())); i += 1; }
            _ => return Err(format!("unexpected character {c}")),
        }
    }
    Ok(toks)
}

fn norm_op(op: &str) -> &str {
    match op { "and" => "&&", "or" => "||", _ => op }
}

fn precedence(op: &str) -> u8 {
    match norm_op(op) {
        "||" => 1,
        "&&" => 2,
        "==" | "!=" | "<" | ">" | "<=" | ">=" => 3,
        "+" | "-" => 4,
        "*" | "/" | "%" => 5,
        _ => 0,
    }
}

struct P { toks: Vec<Tok>, pos: usize }

impl P {
    fn peek(&self) -> Option<&Tok> { self.toks.get(self.pos) }
    fn next(&mut self) -> Option<Tok> { let t = self.toks.get(self.pos).cloned(); self.pos += 1; t }

    fn expr(&mut self, min: u8) -> Result<Expr, String> {
        let mut lhs = self.primary()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Op(o)) => o.clone(),
                Some(Tok::Ident(w)) if w == "and" || w == "or" => w.clone(),
                _ => break,
            };
            let prec = precedence(&op);
            if prec < min || prec == 0 { break; }
            let op = norm_op(&op).to_string();
            self.next();
            let rhs = self.expr(prec + 1)?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs), 0);
        }
        Ok(lhs)
    }

    fn primary(&mut self) -> Result<Expr, String> {
        match self.next() {
            Some(Tok::Int(n)) => Ok(Expr::Int(n, 0)),
            Some(Tok::Float(f)) => Ok(Expr::Float(f, 0)),
            Some(Tok::Ident(id)) => match id.as_str() {
                "true" => Ok(Expr::Bool(true, 0)),
                "false" => Ok(Expr::Bool(false, 0)),
                "not" => Ok(Expr::Not(Box::new(self.expr(6)?), 0)),
                "if" => {
                    let c = self.expr(1)?;
                    expect_ident(self, "then")?;
                    let t = self.expr(1)?;
                    expect_ident(self, "else")?;
                    let e = self.expr(1)?;
                    Ok(Expr::If(Box::new(c), Box::new(t), Box::new(e), 0))
                }
                _ => Ok(Expr::Name(id, 0)),
            },
            Some(Tok::Op(op)) if op == "-" => Ok(Expr::Neg(Box::new(self.expr(6)?), 0)),
            Some(Tok::LParen) => {
                // Prefix (Lisp) applicative form: (+ a b ...)
                if let Some(Tok::Op(op)) = self.peek().cloned() {
                    if precedence(&op) > 0 {
                        self.next();
                        let mut args = Vec::new();
                        while !matches!(self.peek(), Some(Tok::RParen)) {
                            args.push(self.expr(1)?);
                        }
                        self.next(); // )
                        if args.is_empty() { return Err("empty prefix form".into()); }
                        if op == "-" && args.len() == 1 { return Ok(Expr::Neg(Box::new(args.remove(0)), 0)); }
                        let mut it = args.into_iter();
                        let mut acc = it.next().unwrap();
                        for a in it { acc = Expr::Bin(norm_op(&op).to_string(), Box::new(acc), Box::new(a), 0); }
                        return Ok(acc);
                    }
                }
                let e = self.expr(1)?;
                match self.next() { Some(Tok::RParen) => Ok(e), _ => Err("expected )".into()) }
            }
            _ => Err("expected a primary".into()),
        }
    }
}

fn expect_ident(p: &mut P, w: &str) -> Result<(), String> {
    match p.next() {
        Some(Tok::Ident(id)) if id == w => Ok(()),
        _ => Err(format!("expected {w}")),
    }
}

/// Lift a broad target-language expression (arithmetic + logic + comparison + if + prefix).
pub fn lift_broad(src: &str) -> Result<Expr, String> {
    let toks = lex(src)?;
    let mut p = P { toks, pos: 0 };
    let e = p.expr(1)?;
    if p.pos != p.toks.len() { return Err("trailing tokens".into()); }
    Ok(e)
}