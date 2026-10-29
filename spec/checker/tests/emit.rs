//! `rce emit rust|python|js` (spec 10.5): every emitted program is run and its
//! output compared line for line with the reference evaluator's observations
//! (`emit::observe`) on all example programs; then it is run again with
//! `--trace --fuel-trace` and compared line for line with
//! `emit::observe_trace`, the whole trace of every example (each entry of D,
//! the states, outputs, kept reasons and fuel). rustc, python3, node and
//! scala-cli serve only as oracles here. The targets a run covers are named
//! in tests/emit_targets.txt (tests/common/targets.rs): a covered target
//! whose toolchain does not run fails its test, and a target the file does
//! not name is reported as not covered, never counted as passed.
use rce_spec_check::emit::{Lang, emit, observe, observe_trace, with_probes};
use rce_spec_check::parse::parse;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/targets.rs"]
mod targets;

/// Programs in spec/examples (51 through draft 0.7; 51_refined_input adds 5.1 refined types, 52-54 the 0.8 indexed values of 5.16).
const CORPUS: usize = 62;

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

/// The version line of the oracle of `lang`, for a test of that target
/// alone (`targets::require` fails the test when the run does not cover
/// `lang` or its oracle does not run).
fn oracle_version(lang: Lang) -> String {
    targets::require(lang, tool);
    let o = Command::new(tool(lang)).arg("--version").output().unwrap();
    String::from_utf8_lossy(&o.stdout).trim().to_string() + String::from_utf8_lossy(&o.stderr).trim()
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

/// Run one emitted program with no argument; Ok(stdout) or Err(why it did not run).
fn run(lang: Lang, dir: &Path, stem: &str, code: &str) -> Result<String, String> {
    run_args(lang, dir, stem, code, &[&[]]).map(|mut v| v.remove(0))
}

/// Build one emitted program once and run it once per argument list;
/// Ok(stdout of each run) or Err(why it did not build or run).
fn run_args(lang: Lang, dir: &Path, stem: &str, code: &str, runs: &[&[&str]]) -> Result<Vec<String>, String> {
    let src = dir.join(format!("{stem}.{}", lang.ext()));
    std::fs::write(&src, code).unwrap();
    let exe = dir.join(stem);
    if lang == Lang::Rust {
        let c = Command::new(tool(Lang::Rust))
            .args(["--edition", "2024", "-O", "-o"])
            .arg(&exe)
            .arg(&src)
            .output()
            .map_err(|e| e.to_string())?;
        if !c.status.success() {
            return Err(format!("rustc: {}", String::from_utf8_lossy(&c.stderr)));
        }
    }
    let mut outs = vec![];
    for args in runs {
        let out = match lang {
            Lang::Rust => Command::new(&exe).args(*args).output(),
            Lang::Python => Command::new(tool(Lang::Python)).arg(&src).args(*args).output(),
            Lang::Js => Command::new(tool(Lang::Js)).arg(&src).args(*args).output(),
            // Scala 3.10 on the JVM: scala-cli compiles and runs one file
            Lang::Scala => Command::new(tool(Lang::Scala))
                .args(["run", "-q", "--server=false", "--jvm", "21", "--scala", "3.10.0", "--scalac-option", "-nowarn"])
                .arg(&src)
                .arg("--")
                .args(*args)
                .output(),
        }
        .map_err(|e| e.to_string())?;
        if !out.status.success() {
            return Err(format!(
                "exit {:?} ({args:?}): {}",
                out.status.code(),
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        outs.push(String::from_utf8(out.stdout).unwrap());
    }
    Ok(outs)
}

/// The arguments of a run that prints the whole trace of every example.
const TRACE: &[&str] = &["--trace", "--fuel-trace"];

/// The first line where `got` differs from `want`, as a message.
fn first_diff(stem: &str, mode: &str, want: &[String], got: &str) -> Option<String> {
    let got: Vec<&str> = got.lines().collect();
    if got == want {
        return None;
    }
    let i = (0..want.len().max(got.len()))
        .find(|&i| want.get(i).map(|s| s.as_str()) != got.get(i).copied())
        .unwrap();
    Some(format!("{stem} ({mode}): line {}: want {:?}, got {:?}", i + 1, want.get(i), got.get(i)))
}

/// Emit, run and compare every example, plainly and with its whole trace;
/// returns (agreeing, refused) and panics on any disagreement or failed run.
fn check_all(lang: Lang) -> (usize, Vec<String>) {
    let dir = scratch(lang);
    type Prog = (String, (Vec<String>, Vec<String>), Result<String, String>);
    let progs: Vec<Prog> = examples()
        .iter()
        .map(|p| {
            let stem = p.file_stem().unwrap().to_string_lossy().replace('-', "_");
            let m = with_probes(&parse(&std::fs::read_to_string(p).unwrap()).unwrap());
            (stem, (observe(&m), observe_trace(&m)), emit(&m, lang))
        })
        .collect();
    assert_eq!(progs.len(), CORPUS, "the example corpus has {CORPUS} programs");
    type Ran = (String, (Vec<String>, Vec<String>), Result<Result<Vec<String>, String>, String>);
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
                                code.clone().map(|c| run_args(lang, dir, stem, &c, &[&[], TRACE])),
                            )
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let (mut ok, mut kept, mut bad) = (0, vec![], vec![]);
    for (stem, want, r) in results {
        match r {
            Err(why) => kept.push(format!("{stem}: {why}")),
            Ok(Err(why)) => bad.push(format!(
                "{stem}: did not run: {}",
                &why[..why.len().min(3000)]
            )),
            Ok(Ok(got)) => {
                let d = first_diff(&stem, "plain", &want.0, &got[0]).or_else(|| first_diff(&stem, "--trace --fuel-trace", &want.1, &got[1]));
                match d {
                    None => ok += 1,
                    Some(d) => bad.push(d),
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
    let lines: usize = progs.iter().map(|p| p.1.0.len()).sum();
    let traced: usize = progs.iter().map(|p| p.1.1.len()).sum();
    let empty: Vec<&str> = progs
        .iter()
        .filter(|p| p.1.0.is_empty())
        .map(|p| p.0.as_str())
        .collect();
    eprintln!("{lang:?}: {lines} observation lines, {traced} whole-trace lines; programs with none: {empty:?}");
    eprintln!(
        "{lang:?}: {ok} of {CORPUS} agree with the reference; {} kept: {kept:?}",
        kept.len()
    );
    (ok, kept)
}

#[test]
#[cfg_attr(not(rce_emit = "rust"), ignore = "tests/emit_targets.txt does not name rust")]
fn rust_emitter_agrees_with_reference() {
    let v = oracle_version(Lang::Rust);
    eprintln!("rust oracle: {v}");
    let (ok, kept) = check_all(Lang::Rust);
    assert_eq!(ok + kept.len(), CORPUS);
}

#[test]
#[cfg_attr(not(rce_emit = "python"), ignore = "tests/emit_targets.txt does not name python")]
fn python_emitter_agrees_with_reference() {
    let v = oracle_version(Lang::Python);
    eprintln!("python oracle: {v}");
    let (ok, kept) = check_all(Lang::Python);
    assert_eq!(ok + kept.len(), CORPUS);
}

#[test]
#[cfg_attr(not(rce_emit = "js"), ignore = "tests/emit_targets.txt does not name js")]
fn js_emitter_agrees_with_reference() {
    let v = oracle_version(Lang::Js);
    eprintln!("node oracle: {v}");
    let (ok, kept) = check_all(Lang::Js);
    assert_eq!(ok + kept.len(), CORPUS);
}

#[test]
#[cfg_attr(not(rce_emit = "scala"), ignore = "tests/emit_targets.txt does not name scala")]
fn scala_emitter_agrees_with_reference() {
    let v = oracle_version(Lang::Scala);
    eprintln!("scala oracle: {v}");
    let (ok, kept) = check_all(Lang::Scala);
    assert_eq!(ok + kept.len(), CORPUS);
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
    assert!(want.iter().any(|l| l == "fn zindex = 12"), "{want:?}");
    assert!(want.iter().any(|l| l == "fn zkeys = [\"2\", \"-0\", \"1\", \"-0\"]"), "{want:?}");
    assert!(want.iter().any(|l| l.starts_with("fn f32texts = [\"0.000000000000000000000000000000000000000000001\"")), "{want:?}");
    let mut ran = 0;
    for lang in targets::covered(&targets::ALL, tool) {
        let dir = scratch(lang);
        let got = run(lang, &dir, "edge", &emit(&m, lang).unwrap()).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(got.lines().collect::<Vec<_>>(), want, "{lang:?}");
        ran += 1;
    }
    eprintln!("edge program: {ran} covered targets ran and agreed");

}

/// F.5.2 and F.5.7 context text on every target: `Prompt.encode` is the
/// SHA-256 delimiter block of `index08::prompt_encode`, `+` concatenates, and
/// `Header.encode` traps on CR or LF; each emitted program agrees with the
/// reference evaluator line for line.
#[test]
fn prompt_and_header_contexts_agree_on_every_available_target() {
    let src = std::fs::read_to_string("tests/emit_contexts.rce").unwrap();
    assert_eq!(rce_spec_check::check_one(&src), Ok(()));
    let m = with_probes(&parse(&src).unwrap());
    let want = observe(&m);
    let enc = rce_spec_check::index08::prompt_encode("hi");
    assert!(want.iter().any(|l| *l == format!("fn penc = {:?}", enc)) || want.iter().any(|l| l.starts_with("fn penc = ") && l.contains("data-")), "{want:?}");
    assert!(want.iter().any(|l| l.starts_with("fn hcr trap")), "{want:?}");
    assert!(want.iter().any(|l| l.starts_with("fn hlf trap")), "{want:?}");
    let mut ran = 0;
    for lang in targets::covered(&targets::ALL, tool) {
        let dir = scratch(lang);
        let got = run(lang, &dir, "contexts", &emit(&m, lang).unwrap()).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(got.lines().collect::<Vec<_>>(), want, "{lang:?}");
        ran += 1;
    }
    eprintln!("context program: {ran} covered targets ran and agreed");
}

/// 6.3 N7 on every target: display writes as `\u{h}` exactly the code points
/// of `spec/data/display-escape.json`. For each range the program displays
/// its two ends and the code points just outside them, between letters, so
/// a runtime whose escape set differs from the data, or that takes it from
/// its toolchain's Unicode tables, disagrees with the reference here.
#[test]
fn display_escape_set_agrees_on_every_available_target() {
    let mut cps = vec![];
    for &(lo, hi) in rce_spec_check::interp::display_escaped() {
        for cp in [lo.wrapping_sub(1), lo, hi, hi + 1] {
            if char::from_u32(cp).is_some() && !cps.contains(&cp) {
                cps.push(cp);
            }
        }
    }
    let mut src = String::new();
    for (i, ch) in cps.chunks(64).enumerate() {
        let body: String = ch.iter().map(|cp| format!("\\u{{{cp:x}}}a")).collect();
        src += &format!("fn esc{i}() -> text = \"a{body}\"\n");
    }
    // the trap message of toint names its text as display writes it
    src += "fn shy_int() -> i64 = \"1\\u{ad}\".toint()\n";
    let m = with_probes(&parse(&src).unwrap());
    let want = observe(&m);
    assert!(want.iter().any(|l| l.contains("a\\u{ad}a")), "{want:?}");
    assert!(want.iter().any(|l| l.starts_with("fn shy_int trap") && l.contains("\\u{ad}")), "{want:?}");
    let mut ran = 0;
    for lang in targets::covered(&targets::ALL, tool) {
        let dir = scratch(lang);
        let got = run(lang, &dir, "display_escape", &emit(&m, lang).unwrap()).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(got.lines().collect::<Vec<_>>(), want, "{lang:?}");
        ran += 1;
    }
    eprintln!("display escape program: {ran} covered targets ran and agreed");
}
