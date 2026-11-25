//! Falsifiers for the four open edges: full-language lift, full intent→Form synthesis,
//! production JIT (native compile-cache), and wall-metered fleet energy.

use competence_catalog::fleet::{aggregate, measure_local};
use competence_catalog::fullsynth::synthesize_full;
use competence_catalog::jit::Jit;
use competence_catalog::lift::lift_expr;
use competence_catalog::lift_broad::lift_broad;
use competence_catalog::unsilo::{best_fabric, classes, project_all};
use rce_spec_check::ast::Expr;

#[test]
fn lift_broad_handles_logic_conditional_prefix_float() {
    assert!(matches!(lift_broad("if (1 < 2) then (3 + 4) else (5 - 1)").unwrap(), Expr::If(..)));
    assert!(matches!(lift_broad("true and false").unwrap(), Expr::Bin(op, ..) if op == "&&"));
    assert!(matches!(lift_broad("1 < 2").unwrap(), Expr::Bin(op, ..) if op == "<"));
    assert!(matches!(lift_broad("(+ 1 2)").unwrap(), Expr::Bin(op, ..) if op == "+"));
    assert!(matches!(lift_broad("1.5 + 2.5").unwrap(), Expr::Bin(..)));
    assert!(matches!(lift_broad("not true").unwrap(), Expr::Not(..)));
}

#[test]
fn full_synthesis_reaches_goal_with_library() {
    let corpus: Vec<_> = (0..3).map(|_| lift_expr("(a * b) + c").unwrap()).collect();
    let s = synthesize_full("2 + 3", 6, &corpus, 16, 3);
    let p = s.plan.expect("a verified plan");
    assert!(p.verified);
    assert_eq!(p.form, "6");
    assert!(!s.library.is_empty(), "library learned from corpus");
}

#[test]
fn jit_compiles_once_then_runs_native_from_cache() {
    let mut j = Jit::new();
    let a = j.run("2 * 3 + 4").unwrap();
    assert_eq!(a, "10");
    assert_eq!(j.compiles, 1);
    let b = j.run("2 * 3 + 4").unwrap();
    assert_eq!(b, "10");
    assert_eq!(j.hits, 1);
    assert_eq!(j.cache_len(), 1); // compiled once, reused
}

#[test]
fn fleet_energy_is_wall_metered() {
    let r = measure_local("node-a", 500_000).expect("hardware meter");
    assert!(!r.source.is_empty());
    assert!(r.uj > 0, "bare-metal delta must be positive, got {}", r.uj);
    let (tot, acts, mean) = aggregate(&[r]);
    assert_eq!(acts, 500_000);
    assert!(tot > 0 && mean > 0.0);
}

#[test]
fn unsilo_one_competence_projects_to_every_class() {
    let cs = classes();
    assert_eq!(cs.len(), 4); // functional, web, systems, math-hpc (a lens, not the limit)
    let ps = project_all("(a * b) + c").unwrap();
    assert!(ps.len() >= 40, "across the board, got {}", ps.len());
    let langs: Vec<&str> = ps.iter().map(|p| p.lang.as_str()).collect();
    assert!(langs.contains(&"haskell") && langs.contains(&"js")
        && langs.contains(&"rust") && langs.contains(&"c"));
    assert!(ps.iter().all(|p| !p.code.is_empty() && !p.origin.is_empty()));
    assert!(best_fabric().is_some());
}