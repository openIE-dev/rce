//! W-CON-1 at a call (F.6.2): when every argument of a call is a literal, the
//! callee's `requires` clauses are decided with the parameters bound to those
//! literals, and a call that makes them false requires change. A call with a
//! non-literal argument is left to the run-time check (the contract record).
use rce_spec_check::cli;

fn check(name: &str, src: &str) -> (i32, String) {
    let d = std::env::temp_dir().join(format!("rce-wcon1-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let f = d.join(name);
    std::fs::write(&f, src).unwrap();
    let a = vec!["check".to_string(), f.to_string_lossy().into_owned()];
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (code, String::from_utf8(out).unwrap() + &String::from_utf8(err).unwrap())
}

const HALF: &str = "state S\n    n: i64 = 0\nevent Go(x: i64)\nfn half(x: i64) -> i64\n    requires x > 0\n    x / 2\nstep S on Go(x: v)\n    n = CALL + v\n";

#[test]
fn a_literal_argument_that_breaks_requires_is_w_con_1() {
    for arg in ["0", "-1", "-7"] {
        let (c, o) = check("bad.rce", &HALF.replace("CALL", &format!("half({arg})")));
        assert_ne!(c, 0, "half({arg}): {o}");
        assert!(o.contains("W-CON-1") && o.contains("this call of `half`"), "half({arg}): {o}");
    }
}

#[test]
fn a_literal_argument_that_meets_requires_is_accepted() {
    for arg in ["1", "4", "100"] {
        let (c, o) = check("ok.rce", &HALF.replace("CALL", &format!("half({arg})")));
        assert_eq!(c, 0, "half({arg}): {o}");
        assert!(!o.contains("W-CON-1"), "half({arg}): {o}");
    }
}

#[test]
fn a_non_literal_argument_is_left_to_the_run_time_check() {
    let (c, o) = check("var.rce", &HALF.replace("CALL", "half(v)"));
    assert_eq!(c, 0, "{o}");
    assert!(!o.contains("W-CON-1"), "{o}");
}

#[test]
fn every_parameter_is_bound_for_a_two_argument_call() {
    let src = "fn withdraw(balance: i64, amount: i64) -> i64\n    requires amount > 0\n    requires amount <= balance\n    return balance - amount\n\nfn main()\n    emit withdraw(10, ARG).tostr()\n";
    let (c, o) = check("w30.rce", &src.replace("ARG", "30"));
    assert_ne!(c, 0, "{o}");
    assert!(o.contains("W-CON-1"), "{o}");
    let (c, o) = check("w3.rce", &src.replace("ARG", "3"));
    assert_eq!(c, 0, "{o}");
}
