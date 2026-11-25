//! Behavior-identity verification: emitted code is not just syntactically
//! plausible — it is *run* and compared against the RCE reference evaluator.
//! This is the "one complete language" guarantee: emission preserves behavior.
//! Emitted Python is executed with python3 and must agree with `cli::run`.

use competence_catalog::lut_emit::{builtin_luts, emit_expr};

fn python_lut() -> competence_catalog::lut_emit::LanguageLut {
    builtin_luts().into_iter().find(|l| l.name == "python").unwrap()
}

/// Evaluate a closed integer expression with the RCE reference interpreter.
fn reference_eval(expr: &str) -> String {
    let prog = format!(
        "module m\n\nstate Out\n    v: i64 = 0\n\nevent Calc\n\nstep Out on Calc\n    v = {expr}\n"
    );
    let out = rce_spec_check::cli::run(&prog, Some("Calc")).expect("reference run");
    let last = out.lines.iter().rev().find(|l| l.contains("v = ")).cloned().unwrap_or_default();
    last.split("v = ").nth(1).unwrap_or("").trim_end_matches(" }").to_string()
}

/// Emit the expression to Python and execute it with python3.
fn python_eval(expr: &str) -> String {
    let py = emit_expr(expr, &python_lut()).expect("emit to python");
    let script = format!("print({py})");
    let out = std::process::Command::new("python3")
        .arg("-c").arg(&script)
        .output().expect("python3 runs");
    assert!(out.status.success(), "python failed: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

#[test]
fn emitted_python_matches_reference_behavior() {
    // +, -, *, % share integer semantics between RCE i64 and Python int.
    for expr in [
        "2 + 3 * 4",
        "(2 + 3) * 4",
        "10 - 2 - 3",
        "7 % 3",
        "2 * 3 + 4 * 5",
        "100 - 7 * 8",
        "(1 + 2) * (3 + 4)",
        "17 % 5 + 2 * 3",
    ] {
        let r = reference_eval(expr);
        let p = python_eval(expr);
        assert_eq!(r, p, "behavior mismatch for {expr}: reference={r} python={p}");
    }
}

#[test]
fn emitted_python_negation_matches() {
    for expr in ["-(2 + 3)", "-5 + 8", "-(3 * 4) + 10"] {
        assert_eq!(reference_eval(expr), python_eval(expr), "{expr}");
    }
}
