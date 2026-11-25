//! Repair-ladder falsifiers: defective code is corrected by deterministic
//! mutation + replay verification — propose-and-check, the admission law applied
//! to correction. The repaired program provably hits the spec.

use competence_catalog::lift::lift_expr;
use competence_catalog::repair::{eval_expr, repair_to_value};

fn reference_eval(expr: &str) -> String {
    let prog = format!(
        "module m\n\nstate Out\n    v: i64 = 0\n\nevent Calc\n\nstep Out on Calc\n    v = {expr}\n"
    );
    let out = rce_spec_check::cli::run(&prog, Some("Calc")).expect("reference run");
    let last = out.lines.iter().rev().find(|l| l.contains("v = ")).cloned().unwrap_or_default();
    last.split("v = ").nth(1).unwrap_or("").trim_end_matches(" }").to_string()
}

#[test]
fn repairs_an_operator_defect() {
    // 2 + 3 = 5, but the spec wants 6 → repair to 2 * 3.
    let r = repair_to_value("2 + 3", 6).expect("repairable");
    assert_eq!(eval_expr(&lift_expr(&r.repaired).unwrap()), Some(6));
    assert!(r.mutation.contains("operator"), "{}", r.mutation);
}

#[test]
fn repairs_a_constant_defect() {
    // 2 * 3 = 6, spec wants 8 → adjust a constant.
    let r = repair_to_value("2 * 3", 8).expect("repairable");
    assert_eq!(eval_expr(&lift_expr(&r.repaired).unwrap()), Some(8));
}

#[test]
fn repairs_by_operand_swap() {
    // 2 - 3 = -1, spec wants 1 → swap to 3 - 2.
    let r = repair_to_value("2 - 3", 1).expect("repairable");
    assert_eq!(eval_expr(&lift_expr(&r.repaired).unwrap()), Some(1));
}

#[test]
fn non_defective_is_not_repaired() {
    assert!(repair_to_value("2 + 3", 5).is_none(), "already correct");
}

#[test]
fn repair_is_verified_by_the_reference() {
    // The repaired program replays to the spec under the reference evaluator.
    for (defective, target) in [("2 + 3", 6), ("2 * 3", 8), ("10 - 4", 6), ("7 % 3", 1)] {
        if let Some(r) = repair_to_value(defective, target) {
            assert_eq!(reference_eval(&r.repaired), target.to_string(),
                "repair of {defective} must replay to {target}");
        }
    }
}
