//! Cross-language stress test: the real nuance/gotchas in python→rust, rust→fortran,
//! fortran→ada, rust→haskell, rust→ocaml. Naive syntactic ports diverge on division/modulo
//! semantics; `semantics::preserve` must restore behavior preservation.

use competence_catalog::lift::lift_expr;
use competence_catalog::semantics::{eval, preserve, Lang};

const BATTERY: &[&str] = &[
    "-7 / 2", "-7 % 2", "7 / -2", "7 % -2", "-7 / -2", "-7 % -2", "7 / 2", "7 % 2",
];

fn exprs() -> Vec<rce_spec_check::ast::Expr> {
    BATTERY.iter().map(|s| lift_expr(s).unwrap()).collect()
}

fn naive_mismatches(from: Lang, to: Lang) -> usize {
    exprs().iter().filter(|e| eval(e, from) != eval(e, to)).count()
}

#[test]
fn gotchas_exposed_where_semantics_differ() {
    // python (floor) -> rust (trunc): division & modulo diverge on negatives.
    assert!(naive_mismatches(Lang::Python, Lang::Rust) > 0);
    // fortran (trunc-rem) -> ada (floor-mod): modulo diverges.
    assert!(naive_mismatches(Lang::Fortran, Lang::Ada) > 0);
    // rust (trunc) -> haskell (floor): diverges.
    assert!(naive_mismatches(Lang::Rust, Lang::Haskell) > 0);
}

#[test]
fn no_gotcha_where_semantics_agree() {
    // rust & fortran & ocaml all trunc/trunc: a naive port is already correct.
    assert_eq!(naive_mismatches(Lang::Rust, Lang::Fortran), 0);
    assert_eq!(naive_mismatches(Lang::Rust, Lang::Ocaml), 0);
}

#[test]
fn preserve_restores_behavior_across_all_pairs() {
    let pairs = [
        (Lang::Python, Lang::Rust),
        (Lang::Rust, Lang::Fortran),
        (Lang::Fortran, Lang::Ada),
        (Lang::Rust, Lang::Haskell),
        (Lang::Rust, Lang::Ocaml),
        (Lang::Python, Lang::Haskell),
        (Lang::Ada, Lang::Rust),
    ];
    for (from, to) in pairs {
        for e in exprs() {
            let want = eval(&e, from);
            let got = eval(&preserve(&e, from, to), to);
            assert_eq!(got, want, "{from:?}->{to:?} port changed behavior");
        }
    }
}