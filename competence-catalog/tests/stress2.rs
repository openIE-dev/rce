//! Stress layer 2: index base, column-major layout, overflow, short-circuit/laziness.
//! These are the next real gotchas in cross-language ports; exposed and (where value-affecting
//! and compensable) preserved.

use competence_catalog::lift::lift_expr;
use competence_catalog::semantics::{
    col_major, eval_sc, eval_wrapped, index_base, linearize, preserve_index, Lang,
};
use rce_spec_check::ast::Expr;

fn int(n: i128) -> Expr { Expr::Int(n, 0) }
fn bin(op: &str, a: Expr, b: Expr) -> Expr { Expr::Bin(op.into(), Box::new(a), Box::new(b), 0) }

#[test]
fn index_base_shift_preserves_element() {
    let a = Expr::Index(Box::new(Expr::Name("a".into(), 0)), vec![int(0)], 0);
    let to_fortran = preserve_index(&a, Lang::Rust, Lang::Fortran);
    match to_fortran {
        Expr::Index(_, idx, _) => {
            assert!(matches!(&idx[0], Expr::Bin(op, l, r, _) if op=="+" && matches!(**l,Expr::Int(0,_)) && matches!(**r,Expr::Int(1,_))));
        }
        _ => panic!("expected index"),
    }
    assert_eq!(index_base(Lang::Fortran), 1);
    assert_eq!(index_base(Lang::Rust), 0);
}

#[test]
fn column_major_vs_row_major_differ() {
    // (i=1,j=2) in 3x4: Fortran col-major = 2*3+1 = 7; Rust row-major = 1*4+2 = 6.
    assert!(col_major(Lang::Fortran));
    assert!(!col_major(Lang::Rust));
    assert_eq!(linearize(1, 2, 3, 4, Lang::Fortran), 7);
    assert_eq!(linearize(1, 2, 3, 4, Lang::Rust), 6);
    assert_ne!(linearize(1, 2, 3, 4, Lang::Fortran), linearize(1, 2, 3, 4, Lang::Rust));
}

#[test]
fn overflow_gotcha_unbounded_vs_wrap() {
    let e = lift_expr("9223372036854775807 + 1").unwrap();
    let py = eval_wrapped(&e, Lang::Python).unwrap();   // arbitrary precision
    let rs = eval_wrapped(&e, Lang::Rust).unwrap();     // i64 wrap
    assert_eq!(py, 9223372036854775808);
    assert_eq!(rs, -9223372036854775808);
    assert_ne!(py, rs); // a naive port changes the value; compensation needs bigint (open)
}

#[test]
fn short_circuit_avoids_dead_division() {
    let e = bin("and", int(0), bin("/", int(1), int(0)));
    assert_eq!(eval_sc(&e, Lang::Python, true), Some(0));   // short-circuit: dead 1/0 not evaluated
    assert_eq!(eval_sc(&e, Lang::Python, false), None);     // eager: hits div-by-zero
}