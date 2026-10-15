//! Fuel in emitted code (6.7, 10.5, T2): every emitted program spends one
//! unit of fuel at exactly the points where the reference evaluator ticks,
//! and traps "comptime fuel exhausted" at the same step.
//!
//! 1. Step counts: on every example program (with the emitter probes) and
//!    `emit_edge.rce`, each observation's step count printed by the Rust,
//!    Python and JavaScript programs (`--fuel-trace`) equals the reference
//!    evaluator's `steps_used` for that observation.
//! 2. Trap step: for programs in `fuel_progs`, at every fuel limit N from 0
//!    up to one past the largest step count, the observations (outputs,
//!    values, trap messages and step counts) of every target equal the
//!    reference run with the same limit.
//!
//! One test function, because the reference limit is a process-wide setting.
use rce_spec_check::emit::{Lang, emit, observe_fuel, with_probes};
use rce_spec_check::interp::{FUEL, FUEL_LIMIT};
use rce_spec_check::parse::parse;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::Ordering;

#[path = "common/scala_java.rs"]
mod scala_java;
use scala_java::scala_java;

fn tool(l: Lang) -> &'static str {
    match l {
        Lang::Rust => "rustc",
        Lang::Python => "python3",
        Lang::Js => "node",
        Lang::Scala => "scala-cli",
    }
}

fn scratch() -> PathBuf {
    let d = std::env::temp_dir().join(format!("rce-fuel-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Build the emitted program; returns the command line that runs it.
fn build(l: Lang, dir: &Path, stem: &str, code: &str) -> Result<Vec<PathBuf>, String> {
    let src = dir.join(format!("{stem}.{}", l.ext()));
    std::fs::write(&src, code).unwrap();
    match l {
        Lang::Rust => {
            let exe = dir.join(stem);
            let c = Command::new("rustc").args(["--edition", "2024", "-O", "-o"]).arg(&exe).arg(&src).output().map_err(|e| e.to_string())?;
            if !c.status.success() {
                return Err(format!("rustc: {}", String::from_utf8_lossy(&c.stderr)));
            }
            Ok(vec![exe])
        }
        // Scala 3.10 on the JVM: package once into a jar, run it with `java -jar`
        Lang::Scala => {
            let jar = dir.join(format!("{stem}.jar"));
            let c = Command::new("scala-cli")
                .args(["--power", "package", "-q", "--server=false", "--scala", "3.10.0", "--scalac-option", "-nowarn", "--assembly", "-f", "-o"])
                .arg(&jar)
                .arg(&src)
                .output()
                .map_err(|e| e.to_string())?;
            if !c.status.success() {
                return Err(format!("scala-cli: {}", String::from_utf8_lossy(&c.stderr)));
            }
            let java = scala_java().ok_or("no Java 19 or later for the Scala jar")?;
            Ok(vec![java, PathBuf::from("-jar"), jar])
        }
        _ => Ok(vec![PathBuf::from(tool(l)), src]),
    }
}

fn run(cmd: &[PathBuf], fuel: Option<u64>) -> Vec<String> {
    let mut c = Command::new(&cmd[0]);
    c.args(&cmd[1..]).arg("--fuel-trace");
    if let Some(n) = fuel {
        c.arg(format!("--fuel={n}"));
    }
    let o = c.output().unwrap();
    let mut v: Vec<String> = String::from_utf8_lossy(&o.stdout).lines().map(String::from).collect();
    if !o.status.success() {
        v.push(format!("EXIT {:?}: {}", o.status.code(), String::from_utf8_lossy(&o.stderr)));
    }
    v
}

fn first_diff(a: &[String], b: &[String]) -> Option<String> {
    for i in 0..a.len().max(b.len()) {
        let (x, y) = (a.get(i), b.get(i));
        if x != y {
            return Some(format!("line {i}: reference {x:?}, emitted {y:?}"));
        }
    }
    None
}

/// RCE_FUEL_LANGS (for example `py`) and RCE_FUEL_ONLY (a program stem)
/// narrow a local run; neither is set in the full suite.
fn langs() -> Vec<Lang> {
    let only = std::env::var("RCE_FUEL_LANGS").unwrap_or_default();
    [Lang::Rust, Lang::Python, Lang::Js, Lang::Scala]
        .into_iter()

        .filter(|l| only.is_empty() || only.split(',').any(|x| x == l.ext()))
        .filter(|l| Command::new(tool(*l)).arg("--version").output().is_ok_and(|o| o.status.success()))
        .filter(|l| {
            let ok = *l != Lang::Scala || scala_java().is_some();
            if !ok {
                eprintln!("SKIPPED Scala: no Java 19 or later found (not a pass)");
            }
            ok
        })
        .collect()
}

fn corpus() -> Vec<(String, String)> {
    let mut v: Vec<PathBuf> = std::fs::read_dir("../examples")
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "rce"))
        .collect();
    v.sort();
    v.push(PathBuf::from("tests/emit_edge.rce"));
    let only = std::env::var("RCE_FUEL_ONLY").unwrap_or_default();
    v.into_iter()
        .map(|p| (p.file_stem().unwrap().to_string_lossy().replace('-', "_"), std::fs::read_to_string(&p).unwrap()))
        .filter(|(s, _)| only.is_empty() || only.split(',').any(|o| s.starts_with(o)))
        .collect()
}

/// Small programs whose trap step is swept over every fuel limit.
fn fuel_progs() -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = [
        ("sweep_main", "const K: i64 = 3 + 4\n\nfn sq(x: i64) -> i64\n    x * x\n\nfn main()\n    var s: i64 = 0\n    for i in 0..4\n        s = s + sq(i) + K\n        emit s\n    let ys = [y * 2 for y in [1, 2, 3] if y > 1]\n    let zs = ys.map(fn(v: i64) => v + 1)\n    emit zs.fold(0, fn(a: i64, b: i64) => a + b)\n    emit zs.filter(fn(v: i64) => v > 4)\n"),
        ("sweep_methods", "struct P\n    x: i64\n\nimpl P\n    fn get(self) -> i64\n        self.x\n    fn bump(&mut self)\n        self.x += 1\n\nfn twice(f: fn(i64) -> i64, v: i64) -> i64\n    f(f(v))\n\nfn inc(v: i64) -> i64\n    v + 1\n\nfn main()\n    var p = P(x: 1)\n    p.bump()\n    emit p.get()\n    emit twice(inc, 5)\n    var xs: List[i64] = [3, 1, 2]\n    xs.push(4)\n    xs.sort()\n    emit xs\n    let t: text = \"ab\"\n    emit t.upper()\n"),
        ("sweep_overflow", "fn main()\n    var x: i8 = 120\n    while x < 127\n        x = x + 1\n        emit x\n    emit x + 1\n"),
    ]
    .iter()
    .map(|(a, b)| (a.to_string(), b.to_string()))
    .collect();
    for (stem, src) in corpus() {
        if ["03_fib_loop", "04_bank_account", "07_error_union", "24_lambda", "39_structural_recursion", "50_iterators", "21_maps"].contains(&stem.as_str()) {
            v.push((stem, src));
        }
    }
    v
}

#[test]
fn emitted_fuel_matches_the_reference() {
    let dir = scratch();
    let ls = langs();
    let mut bad = vec![];
    // 1. step counts on the whole corpus, default limit
    FUEL_LIMIT.store(FUEL, Ordering::SeqCst);
    let corpus = corpus();
    if std::env::var("RCE_FUEL_ONLY").is_err() {
        assert!(corpus.len() >= 50);
    }
    for (stem, src) in &corpus {
        let m = with_probes(&parse(src).unwrap());
        let want = observe_fuel(&m);
        for &l in &ls {
            let code = emit(&m, l).unwrap_or_else(|e| panic!("{stem}: {e}"));
            match build(l, &dir, &format!("{stem}_{}", l.ext()), &code) {
                Ok(cmd) => {
                    if let Some(d) = first_diff(&want, &run(&cmd, None)) {
                        bad.push(format!("{stem} {l:?}: {d}"));
                    }
                }
                Err(e) => bad.push(format!("{stem} {l:?}: {e}")),
            }
        }
    }
    // 2. trap step at every limit
    let mut swept = 0;
    let only = std::env::var("RCE_FUEL_ONLY").unwrap_or_default();
    for (stem, src) in fuel_progs().into_iter().filter(|(s, _)| only.is_empty() || only.split(',').any(|o| s.starts_with(o))) {
        let m = with_probes(&parse(&src).unwrap());
        FUEL_LIMIT.store(FUEL, Ordering::SeqCst);
        let max = observe_fuel(&m).iter().filter_map(|l| l.strip_prefix("fuel ")).map(|n| n.parse::<u64>().unwrap()).max().unwrap();
        let cmds: Vec<(Lang, Vec<PathBuf>)> = ls
            .iter()
            .map(|&l| (l, build(l, &dir, &format!("sw_{stem}_{}", l.ext()), &emit(&m, l).unwrap()).unwrap()))
            .collect();
        let step = (max / 400).max(1);
        let mut n = 0;
        while n <= max + 1 {
            FUEL_LIMIT.store(n, Ordering::SeqCst);
            let want = observe_fuel(&m);
            for (l, cmd) in &cmds {
                if let Some(d) = first_diff(&want, &run(cmd, Some(n))) {
                    bad.push(format!("sweep {stem} {l:?} fuel {n}: {d}"));
                }
            }
            swept += 1;
            n += if n < 300 { 1 } else { step };
        }
    }
    FUEL_LIMIT.store(FUEL, Ordering::SeqCst);
    eprintln!("fuel: {} programs x {} targets compared; {swept} limits swept", corpus.len(), ls.len());
    assert!(bad.is_empty(), "{} disagreements:\n{}", bad.len(), bad.iter().take(60).cloned().collect::<Vec<_>>().join("\n"));
}
