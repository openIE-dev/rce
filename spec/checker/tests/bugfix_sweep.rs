//! Regressions from the bug sweep (BUGS.md ids in each test name).
use rce_spec_check::cli::diagnostics;
use std::collections::HashSet;

fn diag(src: &str) -> String {
    diagnostics(src, &HashSet::new())
        .into_iter()
        .map(|d| d.1)
        .collect::<Vec<_>>()
        .join("; ")
}

#[test]
fn b001_imin_rem_minus_one_traps() {
    let d = diag("const M: i64 = -9223372036854775807 - 1\ncheck M % -1 == 0\n");
    assert!(d.contains("trap"), "{d}");
    let ok = diag("const M: i64 = -9223372036854775807 - 1\ncheck M % 2 == 0\ncheck -7 % -1 == 0\n");
    assert!(ok.is_empty(), "{ok}");
    let i8 = diag("const a: i8 = -128\nconst m: i8 = -1\ncheck a % m == 0\n");
    assert!(i8.contains("trap"), "{i8}");
}

#[test]
fn b002_toint_out_of_range_traps() {
    let d = diag("check \"99999999999999999999\".toint() == 0\n");
    assert!(d.contains("trap"), "{d}");
    assert!(diag("check \" 12 \".toint() == 12\n").is_empty());
}

#[test]
fn b003_canonical_float_text() {
    let d = diag(
        "check 1e300.tostr() == \"1e+300\"\ncheck 1.5e-7.tostr() == \"1.5e-7\"\ncheck 1e16.tostr() == \"1e+16\"\ncheck 0.00001.tostr() == \"0.00001\"\ncheck 0.30000000000000004.tostr() == \"0.30000000000000004\"\ncheck 0.0.tostr() == \"0\"\n",
    );
    assert!(d.is_empty(), "{d}");
}

fn run(src: &str) -> String {
    let d = std::env::temp_dir().join(format!("rce-sweep-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let f = d.join("b.rce");
    std::fs::write(&f, src).unwrap();
    let a = vec!["run".to_string(), f.to_string_lossy().into_owned()];
    let (mut out, mut err) = (vec![], vec![]);
    rce_spec_check::cli::main(&a, &mut &b""[..], &mut out, &mut err);
    String::from_utf8(out).unwrap() + &String::from_utf8(err).unwrap()
}

#[test]
fn b004_compound_assignment_is_charged_like_binary_op() {
    let o = run("# rce-version: 0.7\nfn main()\n    with budget 5\n        var x = 0\n        for _i in 0..10\n            x += 1\n        emit x\n");
    assert!(o.contains("trap: budget"), "{o}");
    let o = run("# rce-version: 0.7\nfn main()\n    with budget 11\n        var x = 0\n        for _i in 0..10\n            x += 1\n        emit x\n");
    assert!(o.contains("emit 10"), "{o}");
}

#[test]
fn b006_b007_indices_are_usize_and_out_of_range_traps() {
    // an i64 index where usize is expected is an implicit narrowing (negative values are lost): W-TY-3
    assert!(diag("const xs: List[i64] = [1]\nconst k: i64 = 0\ncheck xs[k] == 1\n").contains("W-TY-3"));
    assert!(diag("const xs: List[i64] = [1, 2, 3]\ncheck xs[1:9].len() == 2\n").contains("slice index out of range"));
    assert!(diag("const xs: List[i64] = [1, 2, 3]\ncheck xs[2:1].len() == 0\n").contains("slice bounds reversed"));
    assert!(diag("const xs: List[i64] = [1, 2, 3]\ncheck xs[3] == 0\n").contains("index 3 out of range"));
}

#[test]
fn b008_float_rem_reports_fit() {
    assert!(diag("check 7.5 % 2.0 == 1.5\n").contains("`%` needs integers"));
}

#[test]
fn b009_f64_min_max_evaluate_and_pow_reports_fit() {
    assert!(diag("check 3.0.min(1.5) == 1.5\ncheck 3.0.max(1.5) == 3.0\ncheck 2.0.min(2.0) == 2.0\n").is_empty());
    assert!(diag("check 2.0.pow(2.0) == 4.0\n").contains("W-NAME-1"));
}

#[test]
fn b010_foreign_function_is_not_a_value() {
    let x = "extern \"C\" fn ext(w: &World, x: u32) -> u32\n";
    let d = diag(&format!("{x}fn ap(f: fn(u32) -> u32, v: u32) -> u32\n    return f(v)\nfn main(w: &World)\n    emit ap(ext, 1)\n"));
    assert!(d.contains("W-FFI-1"), "{d}");
    let d = diag(&format!("{x}fn main(w: &World)\n    let g = ext\n    emit g(1)\n"));
    assert!(d.contains("W-FFI-1"), "{d}");
    assert!(diag(&format!("{x}fn main(w: &World)\n    emit ext(w, 1)\n")).is_empty(), "a direct call from main stays valid");
}

#[test]
fn b012_to_b014_diagnostics() {
    assert_eq!(diag("fn f() -> i64\n    return 1\n\tx\n"), "line 3: W-LEX-1 tab character");
    let d = diag("fn f() -> i64\n    let actor = 1\n    return 1\n");
    assert!(d.contains("(found `actor`)") && !d.contains("Id("), "{d}");
    let d = diag("event Go(c: chan i64)\nstate S\n    n: i64 = 0\nstep S on Go(c: c) -> n\n");
    assert!(d.contains("line 1: W-TY-2 event `Go` carries a channel") && !d.contains("line 0"), "{d}");
}

#[test]
fn b015_help_after_a_subcommand() {
    for a in [vec!["fmt", "--help"], vec!["check", "-h"], vec!["--help"], vec!["-h"]] {
        let a: Vec<String> = a.into_iter().map(String::from).collect();
        let (mut out, mut err) = (vec![], vec![]);
        let c = rce_spec_check::cli::main(&a, &mut &b""[..], &mut out, &mut err);
        assert_eq!(c, 0, "{a:?}");
        assert!(String::from_utf8(out).unwrap().contains("usage:"), "{a:?}");
    }
}

#[test]
fn b014_no_diagnostic_on_line_0() {
    let d = diag("fn f() -> i64\n    return 1\n    let _x = 2\n");
    assert!(d.starts_with("line 3: W-MATCH-2"), "{d}");
    let d = diag("use shop.pricing\nfn f() -> i64\n    return pricing.p()\n");
    assert!(d.starts_with("line 1: W-NAME-1 unknown module"), "{d}");
    let d = diag("state S\n    n: i64 = 0\n\nevent Inc\nevent Dec\n\nstep S on Inc -> n + 1\n");
    assert!(d.contains("line 5: W-STEP-2"), "{d}");
}
