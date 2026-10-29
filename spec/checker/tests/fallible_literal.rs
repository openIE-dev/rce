//! Regression: a literal returned from a fallible function is its success
//! value (`return 0` in `-> i64 ! Boom`), and the range and type checks
//! still apply to it, so no ill-typed program is newly accepted.
use rce_spec_check::cli;

fn check(name: &str, src: &str) -> (i32, String) {
    let d = std::env::temp_dir().join(format!("rce-flit-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let f = d.join(name);
    std::fs::write(&f, src).unwrap();
    let a = vec!["run".to_string(), f.to_string_lossy().into_owned()];
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (code, String::from_utf8(out).unwrap() + &String::from_utf8(err).unwrap())
}

#[test]
fn literal_is_the_success_value_of_a_fallible_return() {
    let (c, o) = check("ok.rce", "error Boom\n\nfn f(b: bool) -> i64 ! Boom\n    if b\n        fail Boom\n    return 0\n\nfn g() -> List[i64] ! Boom\n    return [1, 2]\n\nfn main()\n    emit f(false) catch -1\n    emit f(true) catch -1\n    emit (g() catch []).len()\n");
    assert_eq!(c, 0, "{o}");
    assert_eq!(o, "emit 0\nemit -1\nemit 2\n");
}

#[test]
fn soundness_is_unchanged() {
    for (n, src) in [
        ("range.rce", "error Boom\n\nfn f() -> u8 ! Boom\n    return 300\n\nfn main()\n    emit f() catch 0\n"),
        ("text.rce", "error Boom\n\nfn f() -> i64 ! Boom\n    return \"x\"\n\nfn main()\n    emit f() catch 0\n"),
        ("neg.rce", "error Boom\n\nfn f() -> u8 ! Boom\n    return -1\n\nfn main()\n    emit f() catch 0\n"),
        ("list.rce", "error Boom\n\nfn f() -> i64 ! Boom\n    return [1]\n\nfn main()\n    emit f() catch 0\n"),
    ] {
        let (c, o) = check(n, src);
        assert_ne!(c, 0, "{n} must require change: {o}");
    }
}
