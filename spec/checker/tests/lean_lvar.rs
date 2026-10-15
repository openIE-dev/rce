//! Lean-to-evaluator link for T29a: vectors computed by the executable
//! LVar semantics `stepAt` (lean/RCE/LVarAlloc.lean, proved sound against
//! the step relation by `stepAt_sound`) are replayed on the evaluator and
//! the emitters. A vector lists the operations in the order the schedule
//! ran them, with their results, and the outcome (`ok`, or the error).
//! Each becomes a `fn main` doing the same operations on cells
//! `lvar_new(0, 6)` (the lattice 0..6 under max, top 6): every `wait` and
//! `freeze` result must equal Lean's, and the run must end with Lean's
//! error or none. When `lake` is on PATH the vectors are regenerated from
//! Lean and must equal the checked-in file (RCE_BLESS=1 rewrites it).
#![cfg(feature = "conc06")]
use rce_spec_check::emit::{Lang, emit, observe, with_probes};
use rce_spec_check::parse::parse;
use serde_json::Value;
use std::path::Path;
use std::process::Command;

fn spec() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn tool(lang: Lang) -> &'static str {
    match lang {
        Lang::Rust => "rustc",
        Lang::Python => "python3",
        Lang::Js => "node",
        Lang::Scala => "scala-cli",
    }
}

fn run(lang: Lang, tag: &str, code: &str) -> Result<String, String> {
    let dir = std::env::temp_dir().join(format!("rce-emit06b-{tag}-{lang:?}-{}", std::process::id()));
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
        return Err(format!("{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)));
    }
    Ok(String::from_utf8(out.stdout).unwrap())
}


#[test]
fn lean_lvar_vectors_are_current() {
    let file = spec().join("checker/tests/vectors/lvar-lean.jsonl");
    let Ok(out) = Command::new("lake").args(["env", "lean", "--run", "RCE/LVarVectors.lean"]).current_dir(spec().join("lean")).output() else {
        eprintln!("lake not on PATH: checking the checked-in vectors only");
        return;
    };
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let got = String::from_utf8(out.stdout).unwrap();
    if std::env::var("RCE_BLESS").is_ok() {
        std::fs::write(&file, &got).unwrap();
    }
    assert_eq!(std::fs::read_to_string(&file).unwrap(), got, "lvar-lean.jsonl is stale: RCE_BLESS=1");
}

/// the vector as a program, and the lines `observe` must print
fn program(v: &Value) -> (String, Vec<String>) {
    let mut src = String::from("fn main()\n");
    let mut want = vec!["main".to_string()];
    for s in v["steps"].as_array().unwrap() {
        let op = s[1].as_str().unwrap();
        let c = format!("c_{}_{}", s[2][0], s[2][1]);
        let (arg, res) = (s[3].as_u64().unwrap(), s[4].as_u64().unwrap());
        match op {
            "new" => src.push_str(&format!("    let {c} = lvar_new(0, 6)\n")),
            "put" => src.push_str(&format!("    {c}.put({arg})\n")),
            "get" => {
                src.push_str(&format!("    emit {c}.wait([{arg}])\n"));
                want.push(format!("emit {res}"));
            }
            "freeze" => {
                src.push_str(&format!("    emit {c}.freeze()\n"));
                want.push(format!("emit {res}"));
            }
            o => panic!("op {o}"),
        }
    }
    match v["out"].as_str().unwrap() {
        "ok" => want.push("main = ()".into()),
        e => want.push(format!("main trap: {e}")),
    }
    (src, want)
}

#[test]
fn evaluator_and_emitters_agree_with_lean() {
    let text = std::fs::read_to_string(spec().join("checker/tests/vectors/lvar-lean.jsonl")).unwrap();
    let mut n = 0;
    for line in text.lines() {
        let v: Value = serde_json::from_str(line).unwrap();
        let (src, want) = program(&v);
        let d = rce_spec_check::check_program(&[("v.rce".into(), src.clone())], false);
        assert!(d[0].1.is_ok(), "vector {n}: {:?}\n{src}", d[0].1);
        let m = parse(&src).unwrap();
        let got = observe(&m);
        assert_eq!(got, want, "evaluator, vector {n} (seed {}):\n{src}", v["seed"]);
        let langs: &[Lang] = if n < 4 { &[Lang::Rust, Lang::Python, Lang::Js] } else { &[Lang::Python, Lang::Js] };
        let pm = with_probes(&m);
        let want_p = observe(&pm);
        for &lang in langs {
            if Command::new(tool(lang)).arg("--version").output().is_err() {
                eprintln!("SKIPPED {lang:?}: {} not found (not a pass)", tool(lang));
                continue;
            }
            let code = emit(&pm, lang).unwrap();
            let out = run(lang, &format!("v{n}"), &code).unwrap_or_else(|e| panic!("vector {n} {lang:?}: {e}"));
            let out: Vec<&str> = out.lines().collect();
            assert_eq!(out, want_p, "vector {n} {lang:?}");
        }
        n += 1;
    }
    assert_eq!(n, 60);
}
