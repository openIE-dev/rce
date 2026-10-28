//! Fuel in `--run` (6.7, 10.5): the emitted Rust, Python and JavaScript
//! runtimes run `fn main` with the reference's fuel pools (each task its own
//! pool, actor handlers on the scheduler's pool, `par` branches charged back
//! in source order) and trap at the same point as `conc::run_main` at every
//! limit where the reference's result changes, and at the limit before it.
//! Its own test binary: it sets the process-wide `interp::FUEL_LIMIT`.
use rce_spec_check::conc::{Outcome, RunCfg, run_main};
use rce_spec_check::emit::{Lang, emit};
use rce_spec_check::interp::FUEL_LIMIT;
use rce_spec_check::parse::parse;
use std::process::Command;
use std::sync::atomic::Ordering::SeqCst;

fn reference(src: &str, fuel: u64) -> Vec<String> {
    FUEL_LIMIT.store(fuel, SeqCst);
    let m = parse(src).unwrap();
    let r = run_main(&m, &RunCfg::default()).unwrap();
    let mut v: Vec<String> = r.outputs.iter().map(|x| format!("emit {}", x.show())).collect();
    v.push(match r.outcome {
        Outcome::Done => "outcome Done".into(),
        Outcome::Deadlock => "outcome Deadlock".into(),
        Outcome::Stopped(e) => format!("outcome Stopped: {e}"),
    });
    v
}

fn tool(var: &str, dflt: &str) -> String {
    std::env::var(var).ok().filter(|v| !v.is_empty()).unwrap_or(dflt.into())
}

/// (interpreter or "" for a binary, path)
fn build(lang: Lang, stem: &str, src: &str) -> Option<(String, std::path::PathBuf)> {
    let code = emit(&parse(src).unwrap(), lang).unwrap_or_else(|e| panic!("{stem}: {e}"));
    let d = std::env::temp_dir().join("rce-fuel-run");
    std::fs::create_dir_all(&d).unwrap();
    let f = d.join(format!("{stem}.{}", lang.ext()));
    std::fs::write(&f, code).unwrap();
    match lang {
        Lang::Rust => {
            let exe = d.join(stem);
            let c = Command::new(tool("RCE_RUSTC", "rustc")).args(["--edition", "2024", "-O", "-o"]).arg(&exe).arg(&f).output().ok()?;
            assert!(c.status.success(), "{stem}: {}", String::from_utf8_lossy(&c.stderr));
            Some((String::new(), exe))
        }
        Lang::Python => Some((tool("RCE_PYTHON_RUN", &tool("RCE_PYTHON", "python3")), f)),
        Lang::Js => Some((tool("RCE_NODE", "node"), f)),
        // the Scala fuel run is tests/fuel.rs (one jar per program)
        Lang::Scala => None,
    }
}

fn run(exe: &(String, std::path::PathBuf), fuel: u64) -> Vec<String> {
    let mut c = Command::new("timeout");
    c.arg("120");
    if !exe.0.is_empty() {
        c.arg(&exe.0);
    }
    let o = c.arg(&exe.1).arg("--run").arg(format!("--fuel={fuel}")).output().unwrap();
    assert!(o.status.success(), "{:?} fuel {fuel}: {}", exe.1, String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout).unwrap().lines().map(String::from).collect()
}

fn programs() -> Vec<(String, String)> {
    let mut v = vec![];
    for n in ["14_select", "15_actor", "34_chan_types", "47_select_fair"] {
        let ex = std::fs::read_to_string(format!("../examples/{n}.rce")).unwrap();
        let main = std::fs::read_to_string(format!("tests/drivers/{n}.main.rce")).unwrap();
        v.push((format!("d{n}"), ex + &main));
    }
    v.push((
        "tasks".into(),
        "fn sq(x: i64) -> i64\n    x * x\n\nfn w(c: chan i64, k: i64)\n    for i in 0..3\n        c <- sq(k) + i\n    c.close()\n\nfn main()\n    let a = chan[i64](1)\n    let b = chan[i64](1)\n    spawn w(a, 2)\n    spawn w(b, 5)\n    var s: i64 = 0\n    loop bound 10\n        match <- a\n            some(v) -> s += v\n            none -> break\n    loop bound 10\n        match <- b\n            some(v) -> s += sq(v)\n            none -> break\n    emit s\n".into(),
    ));
    v.push((
        "par".into(),
        "fn sq(x: i64) -> i64\n    x * x\n\nfn main()\n    var a: i64 = 0\n    var b: i64 = 0\n    par\n        branch one\n            for i in 0..4\n                a += sq(i)\n        branch two\n            for i in 0..3\n                b += i\n    emit a + b\n".into(),
    ));
    v.push((
        "forpar".into(),
        "fn sq(x: i64) -> i64\n    x * x\n\nfn main()\n    var s: i64 = 0\n    for i in 0..6 par\n        s += sq(i as i64)\n    emit s\n".into(),
    ));
    // a par whose branches wait on lvars: the reference runs it on threads,
    // each branch on a copy of the fork's fuel, and charges the branches back
    // in source order (C.6.3); the emitters run it by restart and restore the
    // fuel of an abandoned attempt. This was the untested case of 10.5.
    v.push((
        "lvarpar".into(),
        "fn sq(x: i64) -> i64\n    x * x\n\nfn main()\n    let a = lvar_new(0)\n    let b = lvar_new(0)\n    var x: i64 = 0\n    var y: i64 = 0\n    par\n        branch one\n            for i in 0..3\n                x += sq(i)\n            x += a.wait([2])\n        branch two\n            y = b.wait([3])\n            for i in 0..4\n                y += sq(i)\n            a.put(2)\n        branch three\n            b.put(3)\n    emit x * 100 + y\n".into(),
    ));
    v.push((
        "lvartrap".into(),
        "fn sq(x: i64) -> i64\n    x * x\n\nfn main()\n    let a = lvar_new(0)\n    let z: i64 = 0\n    var x: i64 = 0\n    var y: i64 = 0\n    par\n        branch one\n            for i in 0..5\n                x += sq(i)\n            a.put(2)\n        branch two\n            y = a.wait([2])\n            for i in 0..5\n                y += sq(i)\n            y = y / z\n    emit x + y\n".into(),
    ));
    v
}

#[test]
fn concurrent_fuel_matches_the_reference() {
    let langs: Vec<Lang> = [Lang::Rust, Lang::Python, Lang::Js]
        .into_iter()
        .filter(|l| std::env::var("RCE_FUEL_LANGS").map_or(true, |v| v.contains(l.ext())))
        .collect();
    let mut bad = vec![];
    let mut checked = 0;
    for (stem, src) in programs() {
        if std::env::var("RCE_FUEL_ONLY").is_ok_and(|o| !stem.starts_with(&o)) {
            continue;
        }
        let full = reference(&src, 50_000_000);
        // every program ends Done except `lvartrap`, which ends with a trap
        // late in a branch, so the fuel trap must win when the branches'
        // summed fuel runs out before that point (source order, C.6.3)
        let want = if stem == "lvartrap" { "outcome Stopped: trap: division by zero" } else { "outcome Done" };
        assert_eq!(full.last().unwrap(), want, "{stem}: {full:?}");
        // the smallest limit from which the run equals the unlimited run
        let mut top = 1u64;
        while reference(&src, top) != full {
            top *= 2;
        }
        let refs: Vec<Vec<String>> = (0..=top).map(|n| reference(&src, n)).collect();
        let mut limits: Vec<u64> = (1..=top as usize).filter(|&n| refs[n] != refs[n - 1]).flat_map(|n| [n as u64 - 1, n as u64]).collect();
        limits.push(top);
        limits.sort();
        limits.dedup();
        eprintln!("{stem}: limits 0..={top}, {} change points", limits.len());
        for lang in &langs {
            let Some(exe) = build(*lang, &stem, &src) else {
                eprintln!("SKIPPED {lang:?}: compiler not found (not a pass)");
                continue;
            };
            for &n in &limits {
                let got = run(&exe, n);
                checked += 1;
                if got != refs[n as usize] {
                    bad.push(format!("{stem} {lang:?} fuel {n}:\n want {:?}\n got  {got:?}", refs[n as usize]));
                    break;
                }
            }
        }
    }
    FUEL_LIMIT.store(rce_spec_check::interp::FUEL, SeqCst);
    assert!(bad.is_empty(), "{} disagreements:\n{}", bad.len(), bad.join("\n"));
    eprintln!("fuel_run: {checked} runs agree with run_main");
}
