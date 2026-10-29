//! rce-check conformance for draft 0.6 (Appendix C, book/14-draft06-concurrency.md C.8):
//! every program in spec/conc06/accept/ is accepted by the reference
//! checker; every program in spec/conc06/ill_typed/ requires change and the
//! report names the rule in its `# expect: RULE` line (after the version line);
//! every program in spec/conc06/change/ requires change in the Rust parser;
//! and `08_run_main.rce` runs with the meaning of C.4.2 and C.5.2.
use rce_spec_check::{check_program, cli};
use std::path::{Path, PathBuf};

fn dir(s: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("conc06").join(s)
}
fn files(d: &Path) -> Vec<(String, String)> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(d)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "rce"))
        .collect();
    v.sort();
    v.into_iter()
        .map(|p| (p.display().to_string(), std::fs::read_to_string(&p).unwrap()))
        .collect()
}

#[test]
fn accept_programs_check() {
    for (n, s) in files(&dir("accept")) {
        let r = check_program(&[(n.clone(), s)], false);
        assert!(r[0].1.is_ok(), "{n}: {:?}", r[0].1);
    }
}

#[test]
fn ill_typed_programs_require_change_by_rule() {
    let fs = files(&dir("ill_typed"));
    assert!(fs.len() >= 16);
    for (n, s) in fs {
        let want = s.lines().find(|l| l.starts_with("# expect: ")).unwrap().trim_start_matches("# expect: ").trim().to_string();
        let r = check_program(&[(n.clone(), s)], false);
        match &r[0].1 {
            Ok(()) => panic!("{n} accepted, expected {want}"),
            Err(e) => assert!(e.contains(&want), "{n}: expected {want}, got {e}"),
        }
    }
}

#[test]
fn syntax_fixtures_require_change_by_parser() {
    for (n, s) in files(&dir("change")) {
        assert!(rce_spec_check::parse::parse(&s).is_err(), "{n} parsed");
    }
}

#[test]
fn par_program_runs_sequential_meaning() {
    let src = std::fs::read_to_string(dir("accept").join("08_run_main.rce")).unwrap();
    let r = cli::run(&src, None).expect("admitted");
    assert!(!r.kept, "{:?}", r.lines);
    let out = r.lines.join("\n");
    println!("{out}");
    assert!(out.contains("emit 42"), "{out}");
    assert!(out.contains("emit 45"), "{out}");
    assert!(out.contains("emit [0, 1, 4, 9]"), "{out}");
}
