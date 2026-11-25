//! Stress layer 3: bigint overflow compensation, type/borrow legality, trained-proposer seed.

use competence_catalog::borrow::{port_structural, Ownership};
use competence_catalog::lift::lift_expr;
use competence_catalog::semantics::{eval_big, needs_bigint, overflow_plan, Lang};
use competence_catalog::train::{features, Scorer};

#[test]
fn bigint_overflow_compensation() {
    let e = lift_expr("9223372036854775807 + 1").unwrap();
    let big = eval_big(&e).unwrap();
    assert_eq!(big.to_string(), "9223372036854775808"); // unbounded, no wrap
    assert!(needs_bigint(Lang::Python, Lang::Rust));
    assert!(!needs_bigint(Lang::Rust, Lang::Python));
    assert!(overflow_plan(Lang::Python, Lang::Rust).contains("bigint"));
}

#[test]
fn borrow_rules_enforced() {
    let mut o = Ownership::new();
    o.declare("x");
    assert!(o.mut_borrow("x").is_ok());
    assert!(o.mut_borrow("x").is_err());      // second mutable borrow
    assert!(o.imm_borrow("x").is_err());      // imm while mut
    o.release_mut("x");
    assert!(o.imm_borrow("x").is_ok());
    assert!(o.imm_borrow("x").is_ok());       // many immutable
    assert!(o.mut_borrow("x").is_err());      // mut while imm
    assert!(o.move_val("x").is_err());        // cannot move while borrowed
}

#[test]
fn move_invalidates_source() {
    let mut o = Ownership::new();
    o.declare("y");
    assert!(o.move_val("y").is_ok());
    assert!(o.use_after_move("y"));
    assert!(o.imm_borrow("y").is_err());
}

#[test]
fn port_structural_directions() {
    assert!(port_structural(true, false).contains("drop"));
    assert!(port_structural(false, true).contains("introduce"));
}

#[test]
fn trained_scorer_ranks_admitted_higher() {
    // receipts: larger Forms admitted, tiny ones refused (linearly separable on node count).
    let samples: Vec<(Vec<f64>, bool)> = (1..8).map(|i| {
        let e = lift_expr(&"1 + ".repeat(i).trim_end_matches(" + ").to_string()).unwrap();
        (features(&e), i >= 4)
    }).collect();
    let s = Scorer::train(&samples, 300, 0.5);
    let hi = features(&lift_expr("1 + 1 + 1 + 1 + 1 + 1 + 1").unwrap());
    let lo = features(&lift_expr("1").unwrap());
    assert!(s.score(&hi) > s.score(&lo), "trained scorer must rank admitted higher");
}