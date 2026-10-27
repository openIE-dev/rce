//! Declared recursion bounds (spec 6.7, `bound n`, W-REC-1; Lean
//! `RCE/Mutual.lean`, `bump_slack`, `bound_dec`). The programs in
//! tests/bound/: `ok_*` are admitted (their `check` items hold at
//! admission) and every emitted target prints the reference's observations,
//! including the `recursion bound` trap of the probes; `bad_*` are refused
//! with W-REC-1.
use rce_spec_check::emit::{Lang, emit, observe};
use rce_spec_check::parse::parse;
use std::process::Command;

#[path = "common/targets.rs"]
mod targets;

fn progs(prefix: &str) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = std::fs::read_dir("tests/bound")
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with(prefix))
        .map(|p| (p.file_stem().unwrap().to_string_lossy().into_owned(), std::fs::read_to_string(&p).unwrap()))
        .collect();
    v.sort();
    v
}

fn verdict(src: &str) -> Result<(), String> {
    let d = rce_spec_check::check_program(&[("t.rce".into(), src.to_string())], false);
    d[0].1.clone()
}

#[test]
fn admitted_and_refused() {
    let ok = progs("ok_");
    let bad = progs("bad_");
    assert!(ok.len() >= 2 && bad.len() >= 3);
    for (n, s) in &ok {
        assert!(verdict(s).is_ok(), "{n}: {:?}", verdict(s));
    }
    for (n, s) in &bad {
        let e = verdict(s).expect_err(n);
        assert!(e.contains("W-REC-1"), "{n}: {e}");
    }
}

#[test]
fn emitted_targets_agree() {
    for (n, src) in progs("ok_") {
        let m = parse(&src).unwrap();
        let want = observe(&m);
        assert!(!want.is_empty(), "{n}");
        let d = std::env::temp_dir().join(format!("rce-bound-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        for lang in targets::covered(&targets::ALL, targets::oracle) {
            let code = emit(&m, lang).unwrap_or_else(|e| panic!("{n} {lang:?}: {e}"));
            let f = d.join(format!("{n}.{}", lang.ext()));
            std::fs::write(&f, code).unwrap();
            let out = match lang {
                Lang::Rust => {
                    let exe = d.join(&n);
                    let c = Command::new(targets::oracle(Lang::Rust)).args(["--edition", "2024", "-O", "-o"]).arg(&exe).arg(&f).output();
                    let c = c.unwrap_or_else(|e| panic!("{n}: rustc does not run: {e}"));
                    assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));
                    Command::new(&exe).output().unwrap()
                }
                Lang::Python => match Command::new(targets::oracle(Lang::Python)).arg(&f).output() {
                    Ok(o) => o,
                    Err(e) => panic!("{n}: python does not run: {e}"),
                },
                Lang::Js => match Command::new(targets::oracle(Lang::Js)).arg(&f).output() {
                    Ok(o) => o,
                    Err(e) => panic!("{n}: node does not run: {e}"),
                },
                Lang::Scala => match Command::new(targets::oracle(Lang::Scala))
                    .args(["run", "-q", "--server=false", "--jvm", "21", "--scala", "3.10.0", "--scalac-option", "-nowarn"])
                    .arg(&f)
                    .output()
                {
                    Ok(o) => o,
                    Err(e) => panic!("{n}: scala-cli does not run: {e}"),
                },
            };
            let got: Vec<String> = String::from_utf8(out.stdout).unwrap().lines().map(String::from).collect();
            assert_eq!(got, want, "{n} {lang:?}");
        }
    }
}

#[test]
fn trap_at_the_bound_in_every_target() {
    let src = std::fs::read_to_string("tests/bound/ok_int.rce").unwrap();
    let m = parse(&src).unwrap();
    let want = observe(&m);
    assert!(want.iter().any(|l| l.contains("recursion bound")), "{want:?}");
}
