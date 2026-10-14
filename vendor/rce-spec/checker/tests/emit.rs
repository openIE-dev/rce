//! `rce emit rust|python|js` (spec 10.5): every emitted program is run and its
//! output compared line for line with the reference evaluator's observations
//! (`emit::observe`) on all example programs. rustc, python3 and node serve
//! only as oracles here. When one is missing the test prints why and skips
//! that target; it never counts a skipped program as passed.
use rce_spec_check::emit::{Lang, emit, observe, with_probes};
use rce_spec_check::parse::parse;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Programs in spec/examples (51 through draft 0.7; 51_refined_input adds 5.1 refined types, 52-54 the 0.8 indexed values of 5.16).
const CORPUS: usize = 56;

fn examples() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir("../examples")
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "rce"))
        .collect();
    // the modules of the package of example 26 (4.1), each emitted alone
    v.extend(rce_spec_check::cli::package_files(Path::new("../examples/26_use_module")).unwrap().into_iter().map(PathBuf::from));
    v.sort();
    v
}

/// The interpreter used as the oracle for `lang`: `RCE_PYTHON` / `RCE_NODE`
/// when set, else `python3` / `node` on PATH. Rust uses `rustc` on PATH.
fn tool(lang: Lang) -> String {
    let (var, def) = match lang {
        Lang::Rust => ("RCE_RUSTC", "rustc"),
        Lang::Python => ("RCE_PYTHON", "python3"),
        Lang::Js => ("RCE_NODE", "node"),
        Lang::Scala => ("RCE_SCALA", "scala-cli"),
    };
    std::env::var(var)
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or(def.to_string())
}

fn have(tool: &str, arg: &str) -> Option<String> {
    let o = Command::new(tool).arg(arg).output().ok()?;
    o.status.success().then(|| {
        String::from_utf8_lossy(&o.stdout).trim().to_string()
            + String::from_utf8_lossy(&o.stderr).trim()
    })
}

fn scratch(lang: Lang) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "rce-emit-{:?}-{}-{:?}",
        lang,
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Run one emitted program; Ok(stdout) or Err(why it did not run).
fn run(lang: Lang, dir: &Path, stem: &str, code: &str) -> Result<String, String> {
    let src = dir.join(format!("{stem}.{}", lang.ext()));
    std::fs::write(&src, code).unwrap();
    let out = match lang {
        Lang::Rust => {
            let exe = dir.join(stem);
            let c = Command::new(tool(Lang::Rust))
                .args(["--edition", "2024", "-O", "-o"])
                .arg(&exe)
                .arg(&src)
                .output()
                .map_err(|e| e.to_string())?;
            if !c.status.success() {
                return Err(format!("rustc: {}", String::from_utf8_lossy(&c.stderr)));
            }
            Command::new(&exe).output()
        }
        Lang::Python => Command::new(tool(Lang::Python)).arg(&src).output(),
        Lang::Js => Command::new(tool(Lang::Js)).arg(&src).output(),
        // Scala 3.10 on the JVM: scala-cli compiles and runs one file
        Lang::Scala => Command::new(tool(Lang::Scala))
            .args(["run", "-q", "--server=false", "--scala", "3.10.0", "--scalac-option", "-nowarn"])
            .arg(&src)
            .output(),

    }
    .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "exit {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8(out.stdout).unwrap())
}

/// Emit, run and compare every example; returns (agreeing, refused) and
/// panics on any disagreement or failed run.
fn check_all(lang: Lang) -> (usize, Vec<String>) {
    let dir = scratch(lang);
    let progs: Vec<(String, Vec<String>, Result<String, String>)> = examples()
        .iter()
        .map(|p| {
            let stem = p.file_stem().unwrap().to_string_lossy().replace('-', "_");
            let m = with_probes(&parse(&std::fs::read_to_string(p).unwrap()).unwrap());
            (stem, observe(&m), emit(&m, lang))
        })
        .collect();
    assert_eq!(progs.len(), CORPUS, "the example corpus has {CORPUS} programs");
    type Ran = (String, Vec<String>, Result<Result<String, String>, String>);
    let results: Vec<Ran> = std::thread::scope(|s| {
        let hs: Vec<_> = progs
            .chunks(progs.len().div_ceil(6))
            .map(|ch| {
                let dir = &dir;
                s.spawn(move || {
                    ch.iter()
                        .map(|(stem, want, code)| {
                            (
                                stem.clone(),
                                want.clone(),
                                code.clone().map(|c| run(lang, dir, stem, &c)),
                            )
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let (mut ok, mut refused, mut bad) = (0, vec![], vec![]);
    for (stem, want, r) in results {
        match r {
            Err(why) => refused.push(format!("{stem}: {why}")),
            Ok(Err(why)) => bad.push(format!(
                "{stem}: did not run: {}",
                &why[..why.len().min(3000)]
            )),
            Ok(Ok(got)) => {
                let got: Vec<&str> = got.lines().collect();
                if got == want {
                    ok += 1;
                } else {
                    let i = (0..want.len().max(got.len()))
                        .find(|&i| want.get(i).map(|s| s.as_str()) != got.get(i).copied())
                        .unwrap();
                    bad.push(format!(
                        "{stem}: line {}: want {:?}, got {:?}",
                        i + 1,
                        want.get(i),
                        got.get(i)
                    ));
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        bad.is_empty(),
        "{:?}: {} disagree:\n{}",
        lang,
        bad.len(),
        bad.join("\n")
    );
    let lines: usize = progs.iter().map(|p| p.1.len()).sum();
    let empty: Vec<&str> = progs
        .iter()
        .filter(|p| p.1.is_empty())
        .map(|p| p.0.as_str())
        .collect();
    eprintln!("{lang:?}: {lines} observation lines; programs with none: {empty:?}");
    eprintln!(
        "{lang:?}: {ok} of {CORPUS} agree with the reference; {} refused: {refused:?}",
        refused.len()
    );
    (ok, refused)
}

#[test]
fn rust_emitter_agrees_with_reference() {
    let Some(v) = have(&tool(Lang::Rust), "--version") else {
        eprintln!(
            "SKIPPED rust emitter test: {} not found (not a pass)",
            tool(Lang::Rust)
        );
        return;
    };
    eprintln!("rust oracle: {v}");
    let (ok, refused) = check_all(Lang::Rust);
    assert_eq!(ok + refused.len(), CORPUS);
}

#[test]
fn python_emitter_agrees_with_reference() {
    let Some(v) = have(&tool(Lang::Python), "--version") else {
        eprintln!(
            "SKIPPED python emitter test: {} not found (not a pass)",
            tool(Lang::Python)
        );
        return;
    };
    eprintln!("python oracle: {v}");
    let (ok, refused) = check_all(Lang::Python);
    assert_eq!(ok + refused.len(), CORPUS);
}

#[test]
fn js_emitter_agrees_with_reference() {
    let Some(v) = have(&tool(Lang::Js), "--version") else {
        eprintln!(
            "SKIPPED js emitter test: {} not found (not a pass)",
            tool(Lang::Js)
        );
        return;
    };
    eprintln!("node oracle: {v}");
    let (ok, refused) = check_all(Lang::Js);
    assert_eq!(ok + refused.len(), CORPUS);
}

#[test]
fn scala_emitter_agrees_with_reference() {
    let Some(v) = have(&tool(Lang::Scala), "--version") else {
        eprintln!(
            "SKIPPED scala emitter test: {} not found (not a pass)",
            tool(Lang::Scala)
        );
        return;
    };
    eprintln!("scala oracle: {v}");
    let (ok, refused) = check_all(Lang::Scala);
    assert_eq!(ok + refused.len(), CORPUS);
}

/// Edge cases beyond the corpus: integer range traps, loop bounds, `&mut`
/// write-back, closures, float and text display, builtin mutation.
#[test]
fn edge_program_agrees_on_every_available_target() {
    let src = std::fs::read_to_string("tests/emit_edge.rce").unwrap();
    let m = with_probes(&parse(&src).unwrap());
    let want = observe(&m);
    assert!(
        want.iter()
            .any(|l| l == "fn over trap: overflow: 400 is outside u8"),
        "{want:?}"
    );
    assert!(want.iter().any(|l| l == "fn spin trap: loop bound"));
    assert!(want.iter().any(|l| l == "fn caller = 21"));
    // u64 * u64 leaves i128: the trap names the exact product (found by rce-c1)
    assert!(want.iter().any(|l| l
        == "fn wide trap: overflow: 340282366920938463426481119284349108225 is outside u64"));
    assert!(want.iter().any(|l| l == "fn wide_sat = 18446744073709551615"));
    assert!(want.iter().any(|l| l == "fn wide_wrap = 1"));
    let mut ran = 0;
    for lang in [Lang::Rust, Lang::Python, Lang::Js, Lang::Scala] {
        let t = tool(lang);
        if have(&t, "--version").is_none() {
            eprintln!("SKIPPED {lang:?} on the edge program: {t} not found (not a pass)");
            continue;
        }
        let dir = scratch(lang);
        let got = run(lang, &dir, "edge", &emit(&m, lang).unwrap()).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(got.lines().collect::<Vec<_>>(), want, "{lang:?}");
        ran += 1;
    }
    eprintln!("edge program: {ran} of 4 targets ran and agreed");

}
