//! Emitters on the draft 0.6 constructs (spec 10.5, Appendix C, book/14-draft06-concurrency.md):
//! `tests/emit06.rce` is emitted to Rust, Python and JavaScript, run, and
//! compared line for line with the reference evaluator, including
//! `for .. par` float reductions over the fixed tree T and `pscan`.
//! The other 0.6 / 0.6b constructs are covered by `tests/emit06b.rs`.
use rce_spec_check::emit::{Lang, emit, observe, with_probes};
use rce_spec_check::parse::parse;
use std::process::Command;

#[path = "common/targets.rs"]
mod targets;

fn tool(lang: Lang) -> &'static str {
    match lang {
        Lang::Rust => "rustc",
        Lang::Python => "python3",
        Lang::Js => "node",
        Lang::Scala => "scala-cli",
    }
}

fn run(lang: Lang, code: &str) -> Result<String, String> {
    let dir = std::env::temp_dir().join(format!("rce-emit06-{lang:?}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join(format!("p.{}", lang.ext()));
    std::fs::write(&src, code).unwrap();
    let out = match lang {
        Lang::Rust => {
            let exe = dir.join("p");
            let c = Command::new("rustc")
                .args(["--edition", "2024", "-O", "-o"])
                .arg(&exe)
                .arg(&src)
                .output()
                .map_err(|e| e.to_string())?;
            if !c.status.success() {
                return Err(String::from_utf8_lossy(&c.stderr).to_string());
            }
            Command::new(&exe).output()
        }
        _ => Command::new(tool(lang)).arg(&src).output(),
    }
    .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_dir_all(&dir);
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).to_string());
    }
    Ok(String::from_utf8(out.stdout).unwrap())
}

#[test]
fn emitters_agree_on_0_6_constructs() {
    let src = std::fs::read_to_string("tests/emit06.rce").unwrap();
    let d = rce_spec_check::check_program(&[("e.rce".into(), src.clone())], false);
    assert!(d[0].1.is_ok(), "{:?}", d[0].1);
    let m = with_probes(&parse(&src).unwrap());
    let want = observe(&m);
    assert!(want.iter().any(|l| l == "fn par_two = 42"), "{want:?}");
    assert!(want.iter().any(|l| l.starts_with("fn overflow trap: overflow")), "{want:?}");
    let mut ran = 0;
    for lang in targets::covered(&[Lang::Rust, Lang::Python, Lang::Js], |l| tool(l).to_string()) {
        let code = emit(&m, lang).unwrap();
        let got = run(lang, &code).unwrap_or_else(|e| panic!("{lang:?} did not run: {e}"));
        assert_eq!(got.lines().collect::<Vec<_>>(), want, "{lang:?}");
        ran += 1;
    }
    eprintln!("emit06: {} observation lines agree on {ran} targets", want.len());
}
