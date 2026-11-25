//! The repair ladder — deterministic repair/correction of defective code.
//!
//! The world's repos are not just ingested; they are *repaired and corrected*.
//! Given a defective program and a spec it fails, the repair engine searches
//! deterministic mutations (operator substitution, operand swap, constant
//! adjustment), and admits the first candidate that **replays** to satisfy the
//! spec. Repair is propose-and-check: the mutation proposes, the replay gate
//! disposes — the same admission law as crystallization, applied to correction.
//!
//! This is the arithmetic-fragment repair ladder (the seed of real-defect repair
//! closure). It is a LUT-style deterministic operation: a table of mutation
//! operators + a deterministic search + replay verification, not a model call.

use rce_spec_check::ast::Expr;

use crate::canonical::emit;
use crate::lift::lift_expr;

/// A verified repair: the original, the corrected source, and the mutation kind.
pub struct Repair {
    /// The defective source.
    pub original: String,
    /// The corrected source (replays to the spec).
    pub repaired: String,
    /// The mutation that fixed it.
    pub mutation: String,
}

/// Evaluate a closed arithmetic expression (no free variables).
pub fn eval_expr(e: &Expr) -> Option<i128> {
    use Expr::*;
    match e {
        Int(n, _) => Some(*n),
        Neg(a, _) => Some(-eval_expr(a)?),
        Bin(op, a, b, _) => {
            let x = eval_expr(a)?;
            let y = eval_expr(b)?;
            match op.as_str() {
                "+" => Some(x.wrapping_add(y)),
                "-" => Some(x.wrapping_sub(y)),
                "*" => Some(x.wrapping_mul(y)),
                "/" => { if y == 0 { Option::None } else { Some(x.wrapping_div(y)) } }
                "%" => { if y == 0 { Option::None } else { Some(x.wrapping_rem(y)) } }
                _ => Option::None,
            }
        }
        _ => Option::None,
    }
}

/// The mutation operators (the repair LUT): alternative arithmetic operators.
const OP_MUTATIONS: &[&str] = &["+", "-", "*", "/", "%"];

/// Generate deterministic mutations of an expression: operator substitution at
/// each binary node, operand swap, and constant adjustment at each leaf.
pub fn mutations(e: &Expr) -> Vec<(Expr, String)> {
    use Expr::*;
    let mut out = Vec::new();
    match e {
        Bin(op, a, b, l) => {
            // Substitute the operator.
            for &new_op in OP_MUTATIONS {
                if new_op != op.as_str() {
                    out.push((Bin(new_op.into(), a.clone(), b.clone(), *l),
                        format!("operator {op}→{new_op}")));
                }
            }
            // Swap operands.
            out.push((Bin(op.clone(), b.clone(), a.clone(), *l), "operand swap".into()));
            // Recurse into children.
            for (ma, ka) in mutations(a) {
                out.push((Bin(op.clone(), Box::new(ma), b.clone(), *l), format!("left: {ka}")));
            }
            for (mb, kb) in mutations(b) {
                out.push((Bin(op.clone(), a.clone(), Box::new(mb), *l), format!("right: {kb}")));
            }
        }
        Int(n, l) => {
            for d in [-2i128, -1, 1, 2] {
                out.push((Int(n + d, *l), format!("constant {n}→{}", n + d)));
            }
        }
        Neg(a, l) => {
            out.push((a.as_ref().clone(), "remove negation".into()));
            for (ma, ka) in mutations(a) {
                out.push((Neg(Box::new(ma), *l), format!("neg: {ka}")));
            }
        }
        _ => {}
    }
    out
}

/// Repair a defective closed expression to hit a target value: search mutations,
/// admit the first that evaluates to the target (replay verification). Returns
/// None if the expression already hits the target (not defective) or no mutation
/// in the ladder reaches it.
pub fn repair_to_value(defective: &str, target: i128) -> Option<Repair> {
    let e = lift_expr(defective).ok()?;
    if eval_expr(&e) == Some(target) {
        return None; // not defective
    }
    for (candidate, kind) in mutations(&e) {
        if eval_expr(&candidate) == Some(target) {
            return Some(Repair {
                original: defective.to_string(),
                repaired: emit(&candidate),
                mutation: kind,
            });
        }
    }
    None
}

/// Repair against a behavioral spec (input→output probes) for a closed expression:
/// the target is the single expected output. (Multi-input specs lift to functions.)
pub fn repair_to_spec(defective: &str, expected: i128) -> Option<Repair> {
    repair_to_value(defective, expected)
}
