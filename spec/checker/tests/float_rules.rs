//! 6.3 N3 (F1 to F9): the float rules derived from IEEE 754-2019 and
//! totality, checked on the interpreter. The value vectors also live in
//! examples/57_float_vectors.rce, whose trace hash is in the cross-target
//! parity set; the trap vectors are here because an example cannot state a
//! trap.
use std::process::Command;

fn run(body: &str, name: &str) -> String {
    let p = std::env::temp_dir().join(format!("rce-float-rules-{}-{name}.rce", std::process::id()));
    std::fs::write(&p, format!("fn main()\n{body}")).unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_rce")).arg("run").arg(&p).output().unwrap();
    let _ = std::fs::remove_file(&p);
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}

fn emits(exprs: &[&str], name: &str) -> Vec<String> {
    let body: String = exprs.iter().map(|e| format!("    emit ({e}).tostr()\n")).collect();
    run(&body, name)
        .lines()
        .filter_map(|l| l.strip_prefix("emit \"").map(|r| r.trim_end_matches('"').to_string()))
        .collect()
}

#[test]
fn f1_basic_operations_are_correctly_rounded() {
    // each expected value is the correctly rounded (ties-to-even) result,
    // compared with the host's own IEEE arithmetic
    let cases: [(&str, f64); 7] = [
        ("0.1 + 0.2", 0.1 + 0.2),
        ("1.0 / 3.0", 1.0 / 3.0),
        ("(2.0).sqrt()", 2.0f64.sqrt()),
        ("1e16 + 1.0", 1e16),
        ("1e16 + 3.0", 1e16 + 4.0),
        ("9007199254740993 as f64", 9007199254740992.0),
        ("1.7976931348623157e308 + 1e291", f64::MAX),
    ];
    let got = emits(&cases.map(|c| c.0), "f1");
    for ((e, want), g) in cases.iter().zip(&got) {
        assert_eq!(g.parse::<f64>().unwrap().to_bits(), want.to_bits(), "{e}: {g}");
    }
    assert_eq!(got.len(), cases.len(), "{got:?}");
}

#[test]
fn f2_no_contraction() {
    // fused: 5.551115123125783e-17; two roundings: 0
    assert_eq!(0.1f64.mul_add(10.0, -1.0), 5.551115123125783e-17);
    let got = emits(&["0.1 * 10.0 - 1.0"], "f2");
    assert_eq!(got, ["0"]);
}

#[test]
fn f3_subnormals_are_preserved() {
    let got = emits(&["5e-324 * 3.0", "2.2250738585072014e-308 * 0.5", "5e-324 * 0.5", "2.2250738585072014e-308 / 4503599627370496.0"], "f3");
    assert_eq!(got, ["1.5e-323", "1.1125369292536007e-308", "0", "5e-324"]);
}

#[test]
fn f4_non_finite_results_trap() {
    for (i, e) in ["1.7976931348623157e308 * 10.0", "1.7976931348623157e308 + 1e292", "1.0 / 0.0", "-1.0 / 0.0", "0.0 / 0.0", "(-1.0).sqrt()", "(1e300 * 1e10) - (1e300 * 1e10)"].iter().enumerate() {
        let out = run(&format!("    emit ({e}).tostr()\n"), &format!("f4-{i}"));
        assert!(out.contains("trap: non-finite float"), "{e}: {out}");
    }
}

#[test]
fn f6_signed_zero_is_unobservable() {
    let got = emits(&["-0.0 * 1.0", "0.0 * -1.0", "(-0.0).sqrt()"], "f6");
    assert_eq!(got, ["0", "0", "0"]);
    let out = run("    emit (1.0 / (0.0 * -1.0)).tostr()\n", "f6-div");
    assert!(out.contains("trap: non-finite float"), "{out}");
}

#[test]
fn f8_rounding_to_integral_is_exact() {
    let got = emits(&["(2.5).round()", "(-2.5).round()", "(0.5).round()", "(-1.5).floor()", "(-1.5).ceil()", "(4503599627370495.5).round()"], "f8");
    assert_eq!(got, ["3", "-3", "1", "-2", "-1", "4503599627370496"]);
}

#[test]
fn f9_no_transcendentals() {
    for m in ["exp", "ln", "log", "sin", "cos", "tan", "pow", "atan2"] {
        let out = run(&format!("    emit (2.0).{m}().tostr()\n"), &format!("f9-{m}"));
        assert!(!out.contains("emit \""), "{m} exists: {out}");
    }
}
