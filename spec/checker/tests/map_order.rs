//! Erratum E-1 to spec 6.7: maps iterate in canonical key order. The
//! reference evaluator admits tests/maporder/ok_* (their `check` items hold)
//! and every emitted target prints the reference's observations.
use rce_spec_check::emit::{Lang, emit, observe};
use rce_spec_check::parse::parse;
use std::process::Command;

fn progs(prefix: &str) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = std::fs::read_dir("tests/maporder")
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
fn admitted() {
    let ok = progs("ok_");
    assert!(!ok.is_empty());
    for (n, s) in &ok {
        assert!(verdict(s).is_ok(), "{n}: {:?}", verdict(s));
    }
}

#[test]
fn emitted_targets_agree() {
    for (n, src) in progs("ok_") {
        let m = parse(&src).unwrap();
        let want = observe(&m);
        assert!(!want.is_empty(), "{n}");
        let d = std::env::temp_dir().join(format!("rce-maporder-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        for lang in [Lang::Rust, Lang::Python, Lang::Js, Lang::Scala] {
            let code = emit(&m, lang).unwrap_or_else(|e| panic!("{n} {lang:?}: {e}"));
            let f = d.join(format!("{n}.{}", lang.ext()));
            std::fs::write(&f, code).unwrap();
            let out = match lang {
                Lang::Rust => {
                    let exe = d.join(&n);
                    let c = Command::new(std::env::var("RCE_RUSTC").unwrap_or("rustc".into())).args(["--edition", "2024", "-O", "-o"]).arg(&exe).arg(&f).output();
                    let Ok(c) = c else { eprintln!("SKIPPED rustc (not a pass)"); continue };
                    assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));
                    Command::new(&exe).output().unwrap()
                }
                Lang::Python => match Command::new(std::env::var("RCE_PYTHON").unwrap_or("python3".into())).arg(&f).output() {
                    Ok(o) => o,
                    Err(_) => { eprintln!("SKIPPED python (not a pass)"); continue }
                },
                Lang::Js => match Command::new(std::env::var("RCE_NODE").unwrap_or("node".into())).arg(&f).output() {
                    Ok(o) => o,
                    Err(_) => { eprintln!("SKIPPED node (not a pass)"); continue }
                },
                Lang::Scala => match Command::new(std::env::var("RCE_SCALA").unwrap_or("scala-cli".into()))
                    .args(["run", "-q", "--server=false", "--jvm", "21", "--scala", "3.10.0", "--scalac-option", "-nowarn"])
                    .arg(&f)
                    .output()
                {
                    Ok(o) => o,
                    Err(_) => { eprintln!("SKIPPED scala-cli (not a pass)"); continue }
                },
            };
            let got: Vec<String> = String::from_utf8(out.stdout).unwrap().lines().map(String::from).collect();
            assert_eq!(got, want, "{n} {lang:?}");
        }
    }
}

