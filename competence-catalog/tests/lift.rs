//! Lift falsifiers: the reverse direction (language → Form) parses target
//! expressions into the canonical IR, and lift→emit round-trips preserve
//! behavior (verified by execution against the reference).

use competence_catalog::canonical;
use competence_catalog::lift::{lift_expr, lift_to_form};
use competence_catalog::lut_emit::{builtin_luts, emit_expr};

#[test]
fn lift_parses_to_canonical_ir() {
    assert_eq!(lift_to_form("(a + b) * c").unwrap(), "((a + b) * c)");
    assert_eq!(lift_to_form("a + b * c").unwrap(), "(a + (b * c))");
    assert_eq!(lift_to_form("-5 + 8").unwrap(), "(-5 + 8)");
}

#[test]
fn lift_respects_precedence() {
    // Multiplication binds tighter than addition.
    let e = lift_expr("2 + 3 * 4").unwrap();
    assert_eq!(canonical::emit(&e), "(2 + (3 * 4))");
}

fn reference_eval(expr: &str) -> String {
    let prog = format!(
        "module m\n\nstate Out\n    v: i64 = 0\n\nevent Calc\n\nstep Out on Calc\n    v = {expr}\n"
    );
    let out = rce_spec_check::cli::run(&prog, Some("Calc")).expect("reference run");
    let last = out.lines.iter().rev().find(|l| l.contains("v = ")).cloned().unwrap_or_default();
    last.split("v = ").nth(1).unwrap_or("").trim_end_matches(" }").to_string()
}

#[test]
fn lift_emit_roundtrip_preserves_behavior() {
    let py = builtin_luts().into_iter().find(|l| l.name == "python").unwrap();
    for src in ["2 + 3 * 4", "(2 + 3) * 4", "10 - 2 - 3", "7 % 3", "2 * 3 + 4 * 5"] {
        let form = lift_to_form(src).unwrap();           // language → Form
        let emitted = emit_expr(&form, &py).unwrap();     // Form → Python
        let out = std::process::Command::new("python3")
            .arg("-c").arg(format!("print({emitted})"))
            .output().unwrap();
        let python = String::from_utf8(out.stdout).unwrap().trim().to_string();
        assert_eq!(reference_eval(&form), python, "round-trip mismatch for {src}");
    }
}
