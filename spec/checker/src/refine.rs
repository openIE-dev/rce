//! Refined types (spec 5.1 "Refined types", W-REF-1; CORE-PROPERTIES.md P2).
//! `type R = τ where p`: the values of τ that satisfy p. p is in a decidable
//! fragment, compiled here once and shared by the checker, the reference
//! evaluator and the emitters, so every implementation decides it alike.
//! Lean: RCE/Refine.lean (holds_total, fit_sound, refine_preserved).
use crate::ast::{Expr, Ty};
use crate::interp::V;

#[derive(Debug, Clone, PartialEq)]
pub enum Tm {
    SelfV,
    Len,
    Lit(i128),
}
#[derive(Debug, Clone, PartialEq)]
pub enum P {
    Cmp(&'static str, Tm, Tm),
    Cls(&'static str),
    And(Box<P>, Box<P>),
    Or(Box<P>, Box<P>),
    Not(Box<P>),
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Base {
    Int,
    Text,
    Coll,
}

pub fn base_of(t: &Ty) -> Option<Base> {
    match t {
        Ty::Path(p, a) if p.len() == 1 => {
            let n = p[0].as_str();
            if a.is_empty() && crate::types::INTS.contains(&n) {
                Some(Base::Int)
            } else if a.is_empty() && n == "text" {
                Some(Base::Text)
            } else if matches!(n, "List" | "Map" | "Set") {
                Some(Base::Coll)
            } else {
                None
            }
        }
        Ty::Array(..) => Some(Base::Coll),
        _ => None,
    }
}

const CMPS: &[&str] = &["<", "<=", ">", ">=", "==", "!="];
const CLS: &[&str] = &["isdigit", "isalpha", "isspace", "isempty"];

fn term(e: &Expr, b: Base) -> Result<Tm, String> {
    match e {
        Expr::SelfRef(_) if b == Base::Int => Ok(Tm::SelfV),
        Expr::SelfRef(_) => Err("only `self.len()` of a text or collection may be compared".into()),
        Expr::Int(n, _) => Ok(Tm::Lit(*n)),
        Expr::Neg(x, _) => match &**x {
            Expr::Int(n, _) => Ok(Tm::Lit(-*n)),
            _ => Err("a predicate compares with integer literals only".into()),
        },
        Expr::Call(f, args, _, _) if args.is_empty() => match &**f {
            Expr::Member(r, m, _) if m == "len" && matches!(**r, Expr::SelfRef(_)) && b != Base::Int => Ok(Tm::Len),
            _ => Err("a predicate term is `self`, `self.len()` or an integer literal".into()),
        },
        _ => Err("a predicate term is `self`, `self.len()` or an integer literal".into()),
    }
}

/// Compile a predicate, or say why it is outside the fragment (W-REF-1).
pub fn compile(e: &Expr, b: Base) -> Result<P, String> {
    match e {
        Expr::Bin(op, x, y, _) if op == "and" || op == "or" => {
            let (x, y) = (Box::new(compile(x, b)?), Box::new(compile(y, b)?));
            Ok(if op == "and" { P::And(x, y) } else { P::Or(x, y) })
        }
        Expr::Bin(op, x, y, _) => match CMPS.iter().find(|c| **c == op.as_str()) {
            Some(c) => Ok(P::Cmp(c, term(x, b)?, term(y, b)?)),
            None => Err(format!("operator `{op}` is not in the predicate fragment")),
        },
        Expr::Not(x, _) => Ok(P::Not(Box::new(compile(x, b)?))),
        Expr::Call(f, args, _, _) if args.is_empty() => match &**f {
            Expr::Member(r, m, _) if matches!(**r, Expr::SelfRef(_)) && b == Base::Text => {
                match CLS.iter().find(|c| **c == m.as_str()) {
                    Some(c) => Ok(P::Cls(c)),
                    None => Err(format!("`self.{m}()` is not in the predicate fragment")),
                }
            }
            _ => Err("only character classes of a text are predicate calls".into()),
        },
        _ => Err("a predicate is a comparison, a character class, `and`, `or` or `not`".into()),
    }
}

fn tv(t: &Tm, v: &V) -> Option<i128> {
    match (t, v) {
        (Tm::Lit(n), _) => Some(*n),
        (Tm::SelfV, V::Int(x, _)) => Some(*x),
        (Tm::Len, V::Text(s)) => Some(s.len() as i128),
        (Tm::Len, V::List(xs) | V::Set(xs)) => Some(xs.len() as i128),
        (Tm::Len, V::Map(xs)) => Some(xs.len() as i128),
        _ => None,
    }
}

/// Whether v satisfies p. Total on values of the base type.
pub fn holds(p: &P, v: &V) -> bool {
    match p {
        P::Cmp(op, a, b) => match (tv(a, v), tv(b, v)) {
            (Some(x), Some(y)) => match *op {
                "<" => x < y,
                "<=" => x <= y,
                ">" => x > y,
                ">=" => x >= y,
                "==" => x == y,
                _ => x != y,
            },
            _ => false,
        },
        P::Cls(c) => match v {
            V::Text(s) => match *c {
                "isdigit" => !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()),
                "isalpha" => !s.is_empty() && s.chars().all(|c| c.is_ascii_alphabetic()),
                "isspace" => !s.is_empty() && s.chars().all(|c| c.is_ascii_whitespace()),
                _ => s.is_empty(),
            },
            _ => false,
        },
        P::And(a, b) => holds(a, v) && holds(b, v),
        P::Or(a, b) => holds(a, v) || holds(b, v),
        P::Not(a) => !holds(a, v),
    }
}

/// The trap text when a value is outside a refined type (all runtimes).
pub fn trap_text(name: &str) -> String {
    format!("refinement: value is outside {name}")
}

/// A predicate as a runtime descriptor: Rust constructors or a Python/JS literal.
pub fn desc(p: &P, rust: bool, py: bool) -> String {
    let lit = |n: i128| {
        if rust {
            format!("RTm::L({n})")
        } else if py {
            format!("(\"lit\", {n})")
        } else {
            format!("[\"lit\", {n}n]")
        }
    };
    let tm = |t: &Tm| match t {
        Tm::SelfV => if rust { "RTm::S".into() } else if py { "(\"self\",)".into() } else { "[\"self\"]".into() },
        Tm::Len => if rust { "RTm::N".into() } else if py { "(\"len\",)".into() } else { "[\"len\"]".into() },
        Tm::Lit(n) => lit(*n),
    };
    let (o, c) = if py { ("(", ")") } else { ("[", "]") };
    match p {
        P::Cmp(op, a, b) => if rust { format!("RP::C(\"{op}\", {}, {})", tm(a), tm(b)) } else { format!("{o}\"cmp\", \"{op}\", {}, {}{c}", tm(a), tm(b)) },
        P::Cls(k) => if rust { format!("RP::K(\"{k}\")") } else { format!("{o}\"cls\", \"{k}\"{c}") },
        P::And(a, b) | P::Or(a, b) => {
            let k = if matches!(p, P::And(..)) { "and" } else { "or" };
            let (a, b) = (desc(a, rust, py), desc(b, rust, py));
            if rust { format!("RP::{}(Box::new({a}), Box::new({b}))", if k == "and" { "A" } else { "O" }) } else { format!("{o}\"{k}\", {a}, {b}{c}") }
        }
        P::Not(a) => { let a = desc(a, rust, py); if rust { format!("RP::Z(Box::new({a}))") } else { format!("{o}\"not\", {a}{c}") } }
    }
}
