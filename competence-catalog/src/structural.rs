//! Frontier 3 — structural search/rewrite over the Form (ast-grep / comby / tree-sitter discipline).
//!
//! The mainstream uses tree-sitter/ast-grep/comby for lint and search; RCE uses the same structural
//! discipline as a *lift/conform* substrate: parse any surface to a tree (tree-sitter grammars live
//! in `rce/tree-sitter/`), then rewrite structurally, name-free. This module is the rewrite half:
//! bottom-up algebraic structural rules (identity, annihilator, involution) applied over the Form,
//! reducing description length without changing behaviour — the equality-saturation-lite used by
//! canonicalization and conform.

use rce_spec_check::ast::Expr;

use crate::canonical::emit;

/// Number of nodes in a Form tree (description length).
pub fn node_count(e: &Expr) -> usize {
    use Expr::*;
    match e {
        Int(_, _) | Name(_, _) => 1,
        Neg(a, _) => 1 + node_count(a),
        Bin(_, l, r, _) => 1 + node_count(l) + node_count(r),
        _ => 1,
    }
}

fn is_int(e: &Expr, v: i128) -> bool {
    matches!(e, Expr::Int(n, _) if *n == v)
}

/// Bottom-up structural rewrite: x+0→x, x*1→x, x*0→0, x−x→0, −(−x)→x.
/// Behaviour-preserving; reduces node count. The structural (not textual) rewrite discipline.
pub fn simplify(e: &Expr) -> Expr {
    use Expr::*;
    match e {
        Neg(a, sp) => {
            let a = simplify(a);
            if let Neg(b, _) = &a {
                return (**b).clone();
            }
            Neg(Box::new(a), sp.clone())
        }
        Bin(op, l, r, sp) => {
            let l = simplify(l);
            let r = simplify(r);
            match op.as_str() {
                "+" => {
                    if is_int(&l, 0) { return r; }
                    if is_int(&r, 0) { return l; }
                }
                "*" => {
                    if is_int(&l, 0) || is_int(&r, 0) { return Int(0, sp.clone()); }
                    if is_int(&l, 1) { return r; }
                    if is_int(&r, 1) { return l; }
                }
                "-" => {
                    if is_int(&r, 0) { return l; }
                    if emit(&l) == emit(&r) { return Int(0, sp.clone()); }
                }
                _ => {}
            }
            Bin(op.clone(), Box::new(l), Box::new(r), sp.clone())
        }
        _ => e.clone(),
    }
}