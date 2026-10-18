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
    let Ok(out) = Command::new("lake").args(["env", "lean", "--run", "LVarVectorsMain.lean"]).current_dir(spec().join("lean")).output() else {
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

/// Every vector as a threaded `par` on OS threads: each Lean thread
/// becomes a branch running its own puts and gets in program order, the
/// cells are allocated before the fork (allocation commutes with every
/// other step), and each `freeze` is moved after the join, where W-LVAR-1
/// accepts it (a freeze racing puts inside the `par` is refused). After
/// the join every put has happened, so a freeze reads the join of all the
/// puts to its cell (the lattice 0..6 under max); the run errs with
/// `lvar conflict` exactly when some cell reaches the top 6, and otherwise
/// every wait gives Lean's result (`store_quasi_det`). A frozen value Lean
/// read mid-run is at most that join. Every schedule, rather than the one
/// the OS picks, is covered by `every_schedule_of_small_vectors` below.
fn threaded(v: &Value) -> (String, Result<Vec<String>, String>, Vec<(u64, u64)>) {
    let (news, th) = thread_ops(v);
    let mut src = String::from("fn main()\n");
    for c in &news {
        src.push_str(&format!("    let {c} = lvar_new(0, 6)\n"));
    }
    let mut join: std::collections::BTreeMap<String, u64> = news.iter().map(|c| (c.clone(), 0)).collect();
    for o in th.iter().flatten() {
        if o.0 == "put" {
            let e = join.get_mut(&o.1).unwrap();
            *e = (*e).max(o.2);
        }
    }
    let mut body = String::new();
    let (mut want, mut frz, mut lean_frz) = (vec![], vec![], vec![]);
    for (t, ops) in th.iter().enumerate() {
        body.push_str(&format!("        branch t{t}\n            var j: i64 = 0\n"));
        for (i, (op, c, arg, res)) in ops.iter().enumerate() {
            match op.as_str() {
                "put" => body.push_str(&format!("            {c}.put({arg})\n")),
                "get" => {
                    src.push_str(&format!("    var r{t}_{i}: i64 = 0\n"));
                    body.push_str(&format!("            r{t}_{i} = {c}.wait([{arg}])\n"));
                    want.push((format!("r{t}_{i}"), res.to_string()));
                }
                "freeze" => {
                    frz.push(c.clone());
                    lean_frz.push((*res, join[c]));
                }
                o => panic!("op {o}"),
            }
        }
        body.push_str("            j += 0\n");
    }
    src.push_str("    par\n");
    src.push_str(&body);
    let mut out = vec![];
    for (r, res) in want {
        src.push_str(&format!("    emit {r}\n"));
        out.push(res);
    }
    for c in frz {
        src.push_str(&format!("    emit {c}.freeze()\n"));
        out.push(join[&c].to_string());
    }
    let res = if join.values().any(|&x| x >= 6) { Err("lvar conflict".to_string()) } else { Ok(out) };
    (src, res, lean_frz)
}

#[test]
fn threaded_scheduler_agrees_with_lean() {
    use rce_spec_check::conc::{Outcome, ParMode, RunCfg, run_main};
    let text = std::fs::read_to_string(spec().join("checker/tests/vectors/lvar-lean.jsonl")).unwrap();
    let (mut n, mut with_freeze) = (0, 0);
    for (k, line) in text.lines().enumerate() {
        let v: Value = serde_json::from_str(line).unwrap();
        let (src, want, lean_frz) = threaded(&v);
        for (read, join) in lean_frz {
            assert!(read <= join, "vector {k}: Lean froze {read} above the join {join}");
        }
        if v["steps"].as_array().unwrap().iter().any(|s| s[1] == "freeze") {
            with_freeze += 1;
        }
        let d = rce_spec_check::check_program(&[("v.rce".into(), src.clone())], false);
        assert!(d[0].1.is_ok(), "vector {k}: {:?}\n{src}", d[0].1);
        if v["out"] == "ok" && !src.contains(".freeze()") {
            assert!(want.is_ok(), "vector {k}: Lean completed but the puts reach the top");
        }
        let r = run_main(&parse(&src).unwrap(), &RunCfg { par_mode: ParMode::Threads, ..RunCfg::default() }).unwrap();
        match want {
            Ok(w) => {
                assert_eq!(r.outcome, Outcome::Done, "vector {k}:\n{src}");
                let got: Vec<String> = r.outputs.iter().map(|x| x.show()).collect();
                assert_eq!(got, w, "vector {k}:\n{src}");
            }
            Err(e) => assert!(matches!(&r.outcome, Outcome::Refused(m) if m.contains(&e)), "vector {k}: want {e}, got {:?}\n{src}", r.outcome),
        }
        n += 1;
    }
    eprintln!("threaded replay: {n} vectors, {with_freeze} with freeze after the join");
    assert_eq!(n, 60);
    assert!(with_freeze >= 10, "{with_freeze}");
}

/// One operation of a Lean thread: (op, cell, argument, Lean's result).
type Op = (String, String, u64, u64);

fn thread_ops(v: &Value) -> (Vec<String>, Vec<Vec<Op>>) {
    let k = v["threads"].as_u64().unwrap() as usize;
    let mut news = vec![];
    let mut th: Vec<Vec<Op>> = vec![vec![]; k];
    for s in v["steps"].as_array().unwrap() {
        let c = format!("c_{}_{}", s[2][0], s[2][1]);
        let op = s[1].as_str().unwrap().to_string();
        if op == "new" {
            news.push(c);
        } else {
            th[s[0].as_u64().unwrap() as usize].push((op, c, s[3].as_u64().unwrap(), s[4].as_u64().unwrap()));
        }
    }
    (news, th)
}

/// The end of one schedule on the evaluator: `Ok(results of the gets and
/// freezes, by (thread, index))`, `Err(trap)`, or None when the schedule
/// is not one (its last operation is a wait that cannot be satisfied yet).
fn run_schedule(news: &[String], th: &[Vec<Op>], order: &[usize]) -> Option<Result<Vec<((usize, usize), String)>, String>> {
    use rce_spec_check::conc::{Outcome, RunCfg, run_main};
    let mut src = String::from("fn main()\n");
    for c in news {
        src.push_str(&format!("    let {c} = lvar_new(0, 6)\n"));
    }
    let mut pos = vec![0; th.len()];
    let mut tags = vec![];
    for &t in order {
        let (op, c, arg, _) = &th[t][pos[t]];
        match op.as_str() {
            "put" => src.push_str(&format!("    {c}.put({arg})\n")),
            "get" => { src.push_str(&format!("    emit {c}.wait([{arg}])\n")); tags.push((t, pos[t])); }
            "freeze" => { src.push_str(&format!("    emit {c}.freeze()\n")); tags.push((t, pos[t])); }
            o => panic!("op {o}"),
        }
        pos[t] += 1;
    }
    let r = run_main(&parse(&src).unwrap(), &RunCfg::default()).unwrap();
    match r.outcome {
        Outcome::Done => Some(Ok(tags.into_iter().zip(r.outputs.iter().map(|x| x.show())).collect())),
        Outcome::Refused(m) if m.contains("lvar wait cannot be satisfied") => None,
        Outcome::Refused(m) => Some(Err(m)),
        o => panic!("{o:?}\n{src}"),
    }
}

/// Every schedule of the threads' operations (every interleaving that
/// respects program order and in which each wait is enabled when it runs),
/// explored depth first with the evaluator's deterministic sequential
/// run as the oracle for enabledness, results and errors.
fn explore(news: &[String], th: &[Vec<Op>], order: &mut Vec<usize>, leaves: &mut Vec<Result<Vec<((usize, usize), String)>, String>>) {
    let r = run_schedule(news, th, order);
    let Some(r) = r else { return };
    let total: usize = th.iter().map(|t| t.len()).sum();
    if r.is_err() || order.len() == total {
        leaves.push(r);
        return;
    }
    for t in 0..th.len() {
        if order.iter().filter(|&&x| x == t).count() < th[t].len() {
            order.push(t);
            explore(news, th, order, leaves);
            order.pop();
        }
    }
}

/// T29a over every schedule, not a sample: for each vector with at most
/// 9 operations, all schedules are run on the evaluator. With no freeze, a
/// vector Lean completes has no erring schedule and every schedule gives
/// Lean's results (`store_det_no_err`, `store_quasi_det`), and a vector
/// Lean ends in an error errs on every schedule (the same puts join to the
/// top whatever the order). With freeze, every schedule that completes
/// gives the same results (`alloc_quasi_det`), and Lean's own outcome is
/// one of the schedules' outcomes.
#[test]
fn every_schedule_of_small_vectors() {
    let text = std::fs::read_to_string(spec().join("checker/tests/vectors/lvar-lean.jsonl")).unwrap();
    let (mut vecs, mut scheds, mut skipped) = (0, 0, 0);
    for (n, line) in text.lines().enumerate() {
        let v: Value = serde_json::from_str(line).unwrap();
        let (news, th) = thread_ops(&v);
        let total: usize = th.iter().map(|t| t.len()).sum();
        if total > 9 {
            skipped += 1;
            continue;
        }
        let mut leaves = vec![];
        explore(&news, &th, &mut vec![], &mut leaves);
        assert!(!leaves.is_empty(), "vector {n}: no schedule");
        let frz = th.iter().flatten().any(|o| o.0 == "freeze");
        let lean: Result<Vec<((usize, usize), String)>, String> = match v["out"].as_str().unwrap() {
            "ok" => {
                let mut want = vec![];
                for (t, ops) in th.iter().enumerate() {
                    for (i, o) in ops.iter().enumerate() {
                        if o.0 != "put" {
                            want.push(((t, i), o.3.to_string()));
                        }
                    }
                }
                Ok(want)
            }
            e => Err(e.to_string()),
        };
        let norm = |r: &Result<Vec<((usize, usize), String)>, String>| match r {
            Ok(x) => { let mut x = x.clone(); x.sort(); Ok(x) }
            Err(e) => Err(e.clone()),
        };
        let lean = norm(&lean);
        let done: Vec<_> = leaves.iter().filter(|r| r.is_ok()).map(&norm).collect();
        for d in &done {
            assert_eq!(d, &done[0], "vector {n}: two complete schedules differ");
        }
        if !frz {
            match &lean {
                Ok(_) => for l in &leaves { assert_eq!(norm(l), lean, "vector {n}: a schedule differs from Lean"); },
                Err(e) => for l in &leaves { assert!(matches!(l, Err(m) if m.contains(e.as_str())), "vector {n}: a schedule did not err: {l:?}"); },
            }
        } else {
            match &lean {
                Ok(_) => assert!(done.iter().all(|d| d == &lean), "vector {n}: complete schedules differ from Lean"),
                Err(e) => assert!(leaves.iter().any(|l| matches!(l, Err(m) if m.contains(e.as_str()))), "vector {n}: no schedule has Lean's error"),
            }
            assert!(leaves.iter().any(|l| norm(l) == lean || matches!((l, &lean), (Err(m), Err(e)) if m.contains(e.as_str()))), "vector {n}: Lean's outcome is not among the schedules");
        }
        vecs += 1;
        scheds += leaves.len();
    }
    eprintln!("every schedule: {vecs} vectors, {scheds} schedules, {skipped} vectors over 9 operations");
    assert!(vecs >= 40, "{vecs}");
}
