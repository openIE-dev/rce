//! Canonicalization-on-save — the self-organizing core (No Accidental Software /
//! Conal Elliott ingested into RCE).
//!
//! The accidental-representation tax is the entropy of expressing the same
//! behavior many equivalent ways. A deterministic normalizer rewrites a Form
//! expression into a canonical shape — constant folding, identity elimination,
//! commutative ordering, double-negation collapse, idempotence, constant
//! conditionals — so accidentally-different source collapses to one canonical
//! form. The rewrite is behavior-preserving (verified by replay over a domain),
//! and the canonical key is invariant under accidental representation changes:
//! two equivalent programs share a key, so the corpus stops paying to re-derive
//! them. This is the "file reorganizes into canonical structure on save" watcher.

use rce_spec_check::ast::Expr;

/// Parse and canonicalize an expression to its canonical source string.
pub fn canonicalize(src: &str) -> Result<String, String> {
    let toks = rce_spec_check::parse::lex(src).map_err(|e| e.to_string())?;
    let mut p = rce_spec_check::parse::Parser::new(toks);
    let e = p.expr().map_err(|e| format!("{e:?}"))?;
    Ok(emit(&normalize(e)))
}

/// The name-free canonical key: SHA-256 of the canonical form. Invariant under
/// accidental representation differences (operand order, identities, folding).
pub fn canonical_key(src: &str) -> Result<String, String> {
    Ok(rcevc_core::sha256(canonicalize(src)?.as_bytes()).hex())
}

fn is_commutative(op: &str) -> bool {
    matches!(op, "+" | "*" | "==" | "!=" | "and" | "or" | "&" | "|")
}

/// Behavior-preserving normalization to canonical shape.
pub fn normalize(e: Expr) -> Expr {
    use Expr::*;
    match e {
        Bin(op, a, b, l) => {
            let a = normalize(*a);
            let b = normalize(*b);
            // Constant folding (integer arithmetic / comparison).
            if let (Int(x, _), Int(y, _)) = (&a, &b) {
                let folded = match op.as_str() {
                    "+" => Some(Int(x.wrapping_add(*y), l)),
                    "-" => Some(Int(x.wrapping_sub(*y), l)),
                    "*" => Some(Int(x.wrapping_mul(*y), l)),
                    "==" => Some(Bool(x == y, l)),
                    "!=" => Some(Bool(x != y, l)),
                    "<" => Some(Bool(x < y, l)),
                    ">" => Some(Bool(x > y, l)),
                    "<=" => Some(Bool(x <= y, l)),
                    ">=" => Some(Bool(x >= y, l)),
                    _ => Option::None,
                };
                if let Some(f) = folded {
                    return f;
                }
            }
            // Identity / annihilator elimination.
            match op.as_str() {
                "+" => {
                    if matches!(b, Int(0, _)) { return a; }
                    if matches!(a, Int(0, _)) { return b; }
                }
                "-" => {
                    if matches!(b, Int(0, _)) { return a; }
                }
                "*" => {
                    if matches!(b, Int(1, _)) { return a; }
                    if matches!(a, Int(1, _)) { return b; }
                    if matches!(b, Int(0, _)) || matches!(a, Int(0, _)) { return Int(0, l); }
                }
                "and" => {
                    if matches!(b, Bool(true, _)) { return a; }
                    if matches!(a, Bool(true, _)) { return b; }
                    if matches!(b, Bool(false, _)) || matches!(a, Bool(false, _)) { return Bool(false, l); }
                    if emit(&a) == emit(&b) { return a; } // idempotence
                }
                "or" => {
                    if matches!(b, Bool(false, _)) { return a; }
                    if matches!(a, Bool(false, _)) { return b; }
                    if matches!(b, Bool(true, _)) || matches!(a, Bool(true, _)) { return Bool(true, l); }
                    if emit(&a) == emit(&b) { return a; }
                }
                _ => {}
            }
            // Commutative canonical operand order.
            if is_commutative(&op) && emit(&a) > emit(&b) {
                return Bin(op, Box::new(b), Box::new(a), l);
            }
            Bin(op, Box::new(a), Box::new(b), l)
        }
        Neg(a, l) => {
            let a = normalize(*a);
            if let Neg(inner, _) = a { return *inner; }
            if let Int(x, _) = a { return Int(-x, l); }
            Neg(Box::new(a), l)
        }
        Not(a, l) => {
            let a = normalize(*a);
            if let Not(inner, _) = a { return *inner; }
            if let Bool(b, _) = a { return Bool(!b, l); }
            Not(Box::new(a), l)
        }
        If(c, t, f, l) => {
            let c = normalize(*c);
            if matches!(c, Bool(true, _)) { return normalize(*t); }
            if matches!(c, Bool(false, _)) { return normalize(*f); }
            If(Box::new(c), Box::new(normalize(*t)), Box::new(normalize(*f)), l)
        }
        List(xs, l) => List(xs.into_iter().map(normalize).collect(), l),
        Tuple(xs, l) => Tuple(xs.into_iter().map(normalize).collect(), l),
        other => other,
    }
}

/// Canonical pretty-print (deterministic; commutative operands already ordered).
pub fn emit(e: &Expr) -> String {
    use Expr::*;
    match e {
        Int(n, _) => n.to_string(),
        Float(f, _) => format!("{f}"),
        Bool(b, _) => b.to_string(),
        Str(s, _) => format!("\"{s}\""),
        Name(n, _) => n.clone(),
        None(_) => "none".into(),
        Unit(_) => "()".into(),
        Bin(op, a, b, _) => format!("({} {} {})", emit(a), op, emit(b)),
        Neg(a, _) => format!("-{}", emit(a)),
        BitNot(a, _) => format!("~{}", emit(a)),
        Not(a, _) => format!("!{}", emit(a)),
        If(c, t, f, _) => format!("(if {} then {} else {})", emit(c), emit(t), emit(f)),
        List(xs, _) => format!("[{}]", xs.iter().map(emit).collect::<Vec<_>>().join(", ")),
        Tuple(xs, _) => format!("({})", xs.iter().map(emit).collect::<Vec<_>>().join(", ")),
        Index(a, idx, _) => format!("{}[{}]", emit(a), idx.iter().map(emit).collect::<Vec<_>>().join(", ")),
        Member(a, m, _) => format!("{}.{}", emit(a), m),
        Call(f, args, _, _) => format!("{}({})", emit(f), args.iter().map(|a| emit(&a.e)).collect::<Vec<_>>().join(", ")),
        Borrow(mut_, a, _) => format!("{}{}", if *mut_ { "&mut " } else { "&" }, emit(a)),
        Cast(a, t, _) => format!("({} as {:?})", emit(a), t),
        _ => "<opaque>".into(),
    }
}
