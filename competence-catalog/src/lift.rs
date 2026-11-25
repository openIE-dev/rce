//! Lift — the reverse direction (language → Form), the other half of "one
//! complete language". A deterministic parser reads a target-language expression
//! into the canonical Form IR. Like emission, this is data-driven in spirit: the
//! grammar is a small table of operator precedences, and the parse is a
//! deterministic operation, not a model call.
//!
//! This is the arithmetic-fragment lifter (numbers, variables, `+ - * / %`,
//! unary `-`, parentheses) — shared syntax across the C-family/Python. It is the
//! seed of the universal importer: lift any language's constructs into Form, then
//! the competence economy owns them. Full grammars per language (tree-sitter as
//! data) scale this the same way emission scales.

use rce_spec_check::ast::Expr;

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Int(i128),
    Ident(String),
    Op(String),
    LParen,
    RParen,
}

fn lex(src: &str) -> Result<Vec<Tok>, String> {
    let mut toks = Vec::new();
    let mut chars = src.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' | '\n' | '\r' => { chars.next(); }
            '(' => { chars.next(); toks.push(Tok::LParen); }
            ')' => { chars.next(); toks.push(Tok::RParen); }
            '0'..='9' => {
                let mut n = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_digit() { n.push(d); chars.next(); } else { break; }
                }
                toks.push(Tok::Int(n.parse().map_err(|_| "bad integer".to_string())?));
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let mut id = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_alphanumeric() || d == '_' { id.push(d); chars.next(); } else { break; }
                }
                toks.push(Tok::Ident(id));
            }
            '+' | '-' | '*' | '/' | '%' => {
                chars.next();
                toks.push(Tok::Op(c.to_string()));
            }
            '<' | '>' => {
                chars.next();
                if chars.peek() == Some(&'=') { chars.next(); toks.push(Tok::Op(format!("{c}="))); }
                else { toks.push(Tok::Op(c.to_string())); }
            }
            '=' => {
                chars.next();
                if chars.peek() == Some(&'=') { chars.next(); toks.push(Tok::Op("==".into())); }
                else { return Err("expected ==".into()); }
            }
            '!' => {
                chars.next();
                if chars.peek() == Some(&'=') { chars.next(); toks.push(Tok::Op("!=".into())); }
                else { return Err("expected !=".into()); }
            }
            '&' => {
                chars.next();
                if chars.peek() == Some(&'&') { chars.next(); toks.push(Tok::Op("&&".into())); }
                else { return Err("expected &&".into()); }
            }
            '|' => {
                chars.next();
                if chars.peek() == Some(&'|') { chars.next(); toks.push(Tok::Op("||".into())); }
                else { return Err("expected ||".into()); }
            }
            _ => return Err(format!("unexpected character {c}")),
        }
    }
    Ok(toks)
}

/// Operator precedence (the grammar table).
fn precedence(op: &str) -> u8 {
    match op {
        "||" => 1,
        "&&" => 2,
        "==" | "!=" | "<" | ">" | "<=" | ">=" => 3,
        "+" | "-" => 4,
        "*" | "/" | "%" => 5,
        _ => 0,
    }
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        self.pos += 1;
        t
    }

    /// Pratt parse: a primary, then operators while precedence binds tighter.
    fn expr(&mut self, min_prec: u8) -> Result<Expr, String> {
        let mut lhs = self.primary()?;
        while let Some(Tok::Op(op)) = self.peek() {
            let prec = precedence(op);
            if prec < min_prec || prec == 0 {
                break;
            }
            let op = op.clone();
            self.next();
            let rhs = self.expr(prec + 1)?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs), 0);
        }
        Ok(lhs)
    }

    fn primary(&mut self) -> Result<Expr, String> {
        match self.next() {
            Some(Tok::Int(n)) => Ok(Expr::Int(n, 0)),
            Some(Tok::Ident(id)) => Ok(Expr::Name(id, 0)),
            Some(Tok::Op(op)) if op == "-" => {
                let e = self.expr(6)?;
                Ok(Expr::Neg(Box::new(e), 0))
            }
            Some(Tok::LParen) => {
                let e = self.expr(1)?;
                match self.next() {
                    Some(Tok::RParen) => Ok(e),
                    _ => Err("expected )".into()),
                }
            }
            _ => Err("expected a primary expression".into()),
        }
    }
}

/// Lift a target-language arithmetic expression into the Form IR.
pub fn lift_expr(src: &str) -> Result<Expr, String> {
    let toks = lex(src)?;
    let mut p = Parser { toks, pos: 0 };
    let e = p.expr(1)?;
    if p.pos != p.toks.len() {
        return Err("trailing tokens".into());
    }
    Ok(e)
}

/// Lift and render to canonical Form source (round-trips through the IR).
pub fn lift_to_form(src: &str) -> Result<String, String> {
    let e = lift_expr(src)?;
    Ok(crate::canonical::emit(&e))
}
