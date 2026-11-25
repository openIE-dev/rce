//! Cross-language semantics — the nuance/gotchas in going between languages.
//!
//! Syntactic LUT emission is not enough: languages differ in *semantics* for the same operator.
//! Integer division rounds differently (Python/Haskell floor vs Rust/Fortran/OCaml trunc) and
//! modulo takes a different sign (Python/Haskell/Ada `mod` follow the divisor; Rust/Fortran/OCaml
//! `rem` follow the dividend). A naive python→rust port of `-7 % 2` silently changes the result.
//! This module models each language's arithmetic semantics, exposes the gotchas, and provides
//! `preserve` — a semantic-aware rewrite so a ported expression evaluates identically in the
//! target language. This is the stress test of lift/emission across real language differences.

use rce_spec_check::ast::Expr;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lang { Python, Rust, Fortran, Ada, Haskell, Ocaml }

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Round { Trunc, Floor }

/// Integer-division rounding per language.
pub fn div_kind(l: Lang) -> Round {
    match l {
        Lang::Python | Lang::Haskell => Round::Floor,
        _ => Round::Trunc,
    }
}

/// Modulo sign per language (floor-mod follows divisor; trunc-rem follows dividend).
pub fn mod_kind(l: Lang) -> Round {
    match l {
        Lang::Python | Lang::Haskell | Lang::Ada => Round::Floor,
        _ => Round::Trunc,
    }
}

fn tdiv(a: i128, b: i128) -> i128 { a / b }
fn trem(a: i128, b: i128) -> i128 { a % b }
fn fdiv(a: i128, b: i128) -> i128 {
    let (q, r) = (a / b, a % b);
    if r != 0 && ((r < 0) != (b < 0)) { q - 1 } else { q }
}
fn fmod(a: i128, b: i128) -> i128 {
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) { r + b } else { r }
}

pub fn div(a: i128, b: i128, l: Lang) -> i128 {
    if b == 0 { return 0; }
    match div_kind(l) { Round::Floor => fdiv(a, b), Round::Trunc => tdiv(a, b) }
}
pub fn modulo(a: i128, b: i128, l: Lang) -> i128 {
    if b == 0 { return 0; }
    match mod_kind(l) { Round::Floor => fmod(a, b), Round::Trunc => trem(a, b) }
}

/// Evaluate a closed integer expression under a language's semantics.
pub fn eval(e: &Expr, l: Lang) -> Option<i128> {
    use Expr::*;
    match e {
        Int(n, _) => Some(*n),
        Neg(a, _) => eval(a, l).map(|v| -v),
        Bin(op, a, b, _) => {
            let (x, y) = (eval(a, l)?, eval(b, l)?);
            Some(match op.as_str() {
                "+" => x + y,
                "-" => x - y,
                "*" => x * y,
                "/" => div(x, y, l),
                "%" => modulo(x, y, l),
                "<" => (x < y) as i128,
                ">" => (x > y) as i128,
                "<=" => (x <= y) as i128,
                ">=" => (x >= y) as i128,
                "==" => (x == y) as i128,
                "!=" => (x != y) as i128,
                "and" => ((x != 0) && (y != 0)) as i128,
                "or" => ((x != 0) || (y != 0)) as i128,
                _ => return Option::None,
            })
        }
        If(c, t, f, _) => {
            if eval(c, l)? != 0 { eval(t, l) } else { eval(f, l) }
        }
        _ => Option::None,
    }
}

fn bin(op: &str, a: Expr, b: Expr) -> Expr { Expr::Bin(op.into(), Box::new(a), Box::new(b), 0) }
fn int(n: i128) -> Expr { Expr::Int(n, 0) }

/// The correction term `c = (r != 0 && (a<0)!=(b<0)) ? 1 : 0` where r = a % b (target semantics).
/// Trunc and floor results differ exactly when the operands have opposite signs and the
/// remainder is nonzero; the adjustment magnitude is the divisor.
fn adj(a: &Expr, b: &Expr) -> Expr {
    let r = bin("%", a.clone(), b.clone());
    let cond = bin("and",
        bin("!=", r.clone(), int(0)),
        bin("!=", bin("<", a.clone(), int(0)), bin("<", b.clone(), int(0))));
    Expr::If(Box::new(cond), Box::new(int(1)), Box::new(int(0)), 0)
}

/// Semantic-aware rewrite: make `e` (source semantics) evaluate identically under `to`.
/// Compensates division/modulo rounding differences using the target's native operators.
pub fn preserve(e: &Expr, from: Lang, to: Lang) -> Expr {
    use Expr::*;
    match e {
        Neg(a, s) => Neg(Box::new(preserve(a, from, to)), *s),
        Bin(op, a, b, s) => {
            let a = preserve(a, from, to);
            let b = preserve(b, from, to);
            match op.as_str() {
                "/" if div_kind(from) != div_kind(to) => {
                    let q = bin("/", a.clone(), b.clone());
                    let c = adj(&a, &b);
                    if div_kind(from) == Round::Floor { bin("-", q, c) } else { bin("+", q, c) }
                }
                "%" if mod_kind(from) != mod_kind(to) => {
                    let r = bin("%", a.clone(), b.clone());
                    let c = bin("*", adj(&a, &b), b.clone());
                    if mod_kind(from) == Round::Floor { bin("+", r, c) } else { bin("-", r, c) }
                }
                _ => Bin(op.clone(), Box::new(a), Box::new(b), *s),
            }
        }
        _ => e.clone(),
    }
}

// ---- Next stress layers: index base, layout, overflow, short-circuit ----

/// Array index base: Fortran/Ada 1-based; C-family/Python 0-based.
pub fn index_base(l: Lang) -> i128 {
    match l { Lang::Fortran | Lang::Ada => 1, _ => 0 }
}

/// Column-major (Fortran) vs row-major (C-family) linearization.
pub fn col_major(l: Lang) -> bool {
    matches!(l, Lang::Fortran)
}

/// Linear offset of (i,j) in a rows×cols matrix under the language's layout (0-based i,j).
pub fn linearize(i: i128, j: i128, rows: i128, cols: i128, l: Lang) -> i128 {
    if col_major(l) { j * rows + i } else { i * cols + j }
}

/// Shift index expressions by the base difference so a port addresses the same element.
pub fn preserve_index(e: &Expr, from: Lang, to: Lang) -> Expr {
    use Expr::*;
    match e {
        Index(a, idx, s) => {
            let shift = index_base(to) - index_base(from);
            let idx2 = idx.iter().map(|i| {
                if shift != 0 { bin("+", i.clone(), int(shift)) } else { i.clone() }
            }).collect();
            Index(a.clone(), idx2, *s)
        }
        _ => e.clone(),
    }
}

/// Fixed-width wrap (C/Rust/Ada/OCaml/Fortran integers) vs arbitrary precision (Python/Haskell).
pub fn wrap_bits(l: Lang) -> Option<u32> {
    match l { Lang::Python | Lang::Haskell => None, _ => Some(64) }
}

fn wrap(v: i128, bits: u32) -> i128 {
    let m = 1i128 << bits;
    let r = v.rem_euclid(m);
    if r >= (m >> 1) { r - m } else { r }
}

/// Evaluate with the language's overflow semantics (wrap at fixed width, or unbounded).
pub fn eval_wrapped(e: &Expr, l: Lang) -> Option<i128> {
    let v = eval(e, l)?;
    Some(match wrap_bits(l) { Some(b) => wrap(v, b), None => v })
}

/// Short-circuit `and`/`or` (Python/Rust/C) vs eager evaluation: a dead `1/0` must not be evaluated.
/// Strict division-by-zero yields None here so the gotcha is observable.
pub fn eval_sc(e: &Expr, l: Lang, short_circuit: bool) -> Option<i128> {
    use Expr::*;
    match e {
        Int(n, _) => Some(*n),
        Neg(a, _) => eval_sc(a, l, short_circuit).map(|v| -v),
        Bin(op, a, b, _) => {
            let x = eval_sc(a, l, short_circuit)?;
            match op.as_str() {
                "and" if short_circuit => {
                    if x == 0 { return Some(0); }
                    eval_sc(b, l, short_circuit).map(|y| (y != 0) as i128)
                }
                "or" if short_circuit => {
                    if x != 0 { return Some(1); }
                    eval_sc(b, l, short_circuit).map(|y| (y != 0) as i128)
                }
                "/" => {
                    let y = eval_sc(b, l, short_circuit)?;
                    if y == 0 { return Option::None; }
                    Some(div(x, y, l))
                }
                _ => {
                    let y = eval_sc(b, l, short_circuit)?;
                    Some(match op.as_str() {
                        "+" => x + y, "-" => x - y, "*" => x * y, "%" => modulo(x, y, l),
                        "and" => ((x != 0) && (y != 0)) as i128,
                        "or" => ((x != 0) || (y != 0)) as i128,
                        _ => return Option::None,
                    })
                }
            }
        }
        _ => Option::None,
    }
}