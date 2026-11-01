//! The emitted Rust program under the word `run` runs `fn main` on its own tick
//! runtime (one OS thread per task, a scheduler admitting moves tick by
//! tick). Its outputs and outcome are compared with the reference
//! `conc::run_main` on a corpus of task programs, and a stress pass reruns
//! each program and a many-task program repeatedly: every run must give the
//! same result as the evaluator, whatever the OS does with the threads.
use rce_spec_check::conc::{Outcome, RunCfg, run_main};
use rce_spec_check::emit::{Lang, emit};
use rce_spec_check::parse::parse;
use std::path::PathBuf;
use std::process::Command;

#[path = "common/targets.rs"]
mod targets;

fn rustc() -> String {
    std::env::var("RCE_RUSTC").ok().filter(|v| !v.is_empty()).unwrap_or("rustc".into())
}

fn node() -> String {
    std::env::var("RCE_NODE").ok().filter(|v| !v.is_empty()).unwrap_or("node".into())
}

/// the program that runs target `l` here
fn oracle(l: Lang) -> String {
    match l {
        Lang::Rust => rustc(),
        Lang::Python => python(),
        Lang::Js => node(),
        Lang::Scala => "scala-cli".into(),
    }
}

/// for a test of target `l` alone: fails unless this run covers `l`
/// (tests/emit_targets.txt) and its toolchain runs; such a test is marked
/// ignored, through `rce_emit`, when the file does not name `l`
fn require(l: Lang) {
    targets::require(l, oracle);
}

type Links = Vec<(u64, bool, String, String)>;

fn want(src: &str) -> Vec<String> {
    want_links(src, &vec![])
}

fn want_links(src: &str, links: &Links) -> Vec<String> {
    let d = rce_spec_check::check_program(&[("t.rce".into(), src.to_string())], false);
    assert!(d[0].1.is_ok(), "not admitted: {:?}\n{src}", d[0].1);
    let m = parse(src).unwrap();
    let r = run_main(&m, &RunCfg { links: links.clone(), ..RunCfg::default() }).unwrap();
    let mut v: Vec<String> = r.outputs.iter().map(|x| format!("emit {}", x.show())).collect();
    v.push(match r.outcome {
        Outcome::Done => "outcome Done".into(),
        Outcome::Deadlock => "outcome Deadlock".into(),
        Outcome::Stopped(e) => format!("outcome Stopped: {e}"),
    });
    v
}

/// the Python oracle for `run`: RCE_PYTHON_RUN, else RCE_PYTHON, else a
/// free-threaded 3.14 (python3.14t on PATH, or one found by `uv python
/// find 3.14t`), else python3
fn python() -> String {
    for v in ["RCE_PYTHON_RUN", "RCE_PYTHON"] {
        if let Ok(p) = std::env::var(v) && !p.is_empty() {
            return p;
        }
    }
    let gil_off = |p: &str| {
        Command::new(p)
            .args(["-c", "import sys; print(sys._is_gil_enabled())"])
            .output()
            .is_ok_and(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).trim() == "False")
    };
    if gil_off("python3.14t") {
        return "python3.14t".into();
    }
    if let Ok(o) = Command::new("uv").args(["python", "find", "3.14t"]).output()
        && o.status.success()
    {
        let p = String::from_utf8_lossy(&o.stdout).trim().to_string();
        if !p.is_empty() && gil_off(&p) {
            return p;
        }
    }
    "python3".into()
}

/// an emitted program ready to run: (interpreter or "" for a binary, path)
type Exe = (String, PathBuf);

fn build_lang(lang: Lang, stem: &str, src: &str) -> Exe {
    if lang == Lang::Rust {
        return (String::new(), build(stem, src));
    }
    let m = parse(src).unwrap();
    let code = emit(&m, lang).unwrap_or_else(|e| panic!("{stem}: not emitted: {e}"));
    let d = std::env::temp_dir().join("rce-emit-run");
    std::fs::create_dir_all(&d).unwrap();
    let f = d.join(format!("{stem}.{}", lang.ext()));
    std::fs::write(&f, code).unwrap();
    if lang == Lang::Scala {
        // one assembly per program, then a launcher that runs it with the arguments
        let jar = d.join(format!("{stem}.jar"));
        let c = Command::new("scala-cli")
            .args(["--power", "package", "-q", "--server=false", "--jvm", "21", "--scala", "3.10.0", "--scalac-option", "-nowarn", "--assembly", "-f", "-o"])
            .arg(&jar)
            .arg(&f)
            .output()
            .unwrap();
        assert!(c.status.success(), "{stem}: scala-cli: {}", String::from_utf8_lossy(&c.stderr));
        let sh = d.join(format!("{stem}.sh"));
        let java = java_bin().expect("a JVM");
        std::fs::write(&sh, format!("#!/bin/sh\nexec '{}' -jar '{}' \"$@\"\n", java.display(), jar.display())).unwrap();
        return ("sh".into(), sh);
    }
    (oracle(lang), f)
}

/// a JVM for the Scala assemblies: RCE_JAVA, else `java` when it runs,
/// else the JDK 21 that scala-cli fetched into the Coursier cache
fn java_bin() -> Option<PathBuf> {
    if let Ok(j) = std::env::var("RCE_JAVA") && !j.is_empty() {
        return Some(PathBuf::from(j));
    }
    if Command::new("java").arg("-version").output().is_ok_and(|o| o.status.success()) {
        return Some(PathBuf::from("java"));
    }
    fn walk(d: &std::path::Path, depth: u32, out: &mut Option<PathBuf>) {
        if depth == 0 || out.is_some() {
            return;
        }
        let Ok(rd) = std::fs::read_dir(d) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.ends_with("bin/java") && p.to_string_lossy().contains("21") {
                *out = Some(p);
                return;
            }
            if p.is_dir() {
                walk(&p, depth - 1, out);
            }
        }
    }
    let home = std::env::var("HOME").ok()?;
    let mut out = None;
    for c in ["Library/Caches/Coursier/arc", ".cache/coursier/arc"] {
        walk(&PathBuf::from(&home).join(c), 10, &mut out);
    }
    out
}

/// the targets with a tick runtime that this run covers
/// (tests/emit_targets.txt); a covered Scala also needs a JVM for its
/// assemblies
fn run_targets() -> Vec<Lang> {
    let v = targets::covered(&targets::ALL, oracle);
    assert!(!v.contains(&Lang::Scala) || java_bin().is_some(), "scala: no JVM for the assemblies; {} names scala", targets::FILE);
    v
}

fn run_exe(exe: &Exe, links: &Links) -> Vec<String> {
    // `timeout` guards against a hung runtime (a deadlock in the runtime itself)
    let mut c = Command::new("timeout");
    c.arg("120");
    if !exe.0.is_empty() {
        c.arg(&exe.0);
    }
    c.arg(&exe.1);
    c.arg("run");
    for (t, p, a, b) in links {
        c.arg("link").arg(format!("{t},{},{a},{b}", if *p { "partition" } else { "heal" }));
    }
    let o = c.output().unwrap();
    assert!(o.status.success(), "{:?}: {}", exe.1, String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout).unwrap().lines().map(String::from).collect()
}

fn build(stem: &str, src: &str) -> PathBuf {
    let m = parse(src).unwrap();
    let code = emit(&m, Lang::Rust).unwrap_or_else(|e| panic!("{stem}: not emitted: {e}"));
    let d = std::env::temp_dir().join("rce-emit-run");
    std::fs::create_dir_all(&d).unwrap();
    let rs = d.join(format!("{stem}.rs"));
    std::fs::write(&rs, code).unwrap();
    let exe = d.join(stem);
    let c = Command::new(rustc()).args(["--edition", "2024", "-O", "-o"]).arg(&exe).arg(&rs).output().unwrap();
    assert!(c.status.success(), "{stem}: rustc: {}", String::from_utf8_lossy(&c.stderr));
    exe
}

fn got(exe: &PathBuf) -> Vec<String> {
    got_links(exe, &vec![])
}

fn got_links(exe: &PathBuf, links: &Links) -> Vec<String> {
    let mut c = Command::new(exe);
    c.arg("run");
    for (t, p, a, b) in links {
        c.arg("link").arg(format!("{t},{},{a},{b}", if *p { "partition" } else { "heal" }));
    }
    let o = c.output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout).unwrap().lines().map(String::from).collect()
}

const NONDET: &str = "fn feed(c: chan i64, base: i64)
    for i in 0..5
        c <- base + i
fn main()
    let a = chan[i64](8)
    let b = chan[i64](8)
    spawn feed(a, 100)
    spawn feed(b, 200)
    loop bound 20
        select nondet
            recv v from a -> emit v
            recv w from b -> emit w
            after 3 -> break
";

const ROTATE: &str = "fn feed(c: chan i64, base: i64)
    for i in 0..4
        c <- base + i
fn main()
    let a = chan[i64](1)
    let b = chan[i64](1)
    spawn feed(a, 10)
    spawn feed(b, 20)
    loop bound 20
        select
            recv v from a -> emit v
            recv w from b -> emit w
            after 2 -> break
";

const DEADLINE: &str = "fn main()
    let c = chan[i64](1)
    with deadline 4
        let v = <- c
        emit v
    else
        emit 0
    emit 9
";

const DEADLINE_OK: &str = "fn put(c: chan i64)\n    c <- 5\nfn main()\n    let c = chan[i64](1)\n    spawn put(c)\n    with deadline 10\n        match <- c\n            some(v) -> emit v\n            none -> emit 1\n    else\n        emit 0\n";

const DEADLOCK: &str = "fn main()\n    let c = chan[i64](1)\n    let v = <- c\n    emit v\n";

const BACKPRESSURE: &str = "fn prod(c: chan i64)
    for i in 0..6
        c <- i
        emit i * 100
    c.close()
fn main()
    let c = chan[i64](1)
    spawn prod(c)
    loop
        match <- c
            some(v) -> emit v
            none -> break
";

const SCOPE_FAIL: &str = "error Boom
fn bad(_c: chan i64) -> i64 ! Boom
    fail Boom
fn slow(c: chan i64)
    loop bound 50
        c <- 2
fn main()
    let c = chan[i64](4)
    scope
        spawn bad(c)
        spawn slow(c)
    emit 7
";

const SCOPE: &str = "fn worker(c: chan i64, k: i64) -> i64
    c <- k
    k * 10

fn bad(_c: chan i64) -> i64
    let xs = [1, 2]
    xs[5]

fn main()
    let c = chan[i64](8)
    scope
        spawn worker(c, 1) -> h1
        spawn worker(c, 2)
        spawn worker(c, 3)
        emit h1.join()
    emit 99
    scope
        spawn bad(c) -> h3
        emit h3.join()
    emit 100
    scope
        spawn worker(c, 4)
        spawn bad(c)
    emit 101
";

/// the budget-exhaustion edge case inside a task: the trap falls on the same
/// operation (the send whose C_CHAN charge crosses the bound), so the outputs
/// before it and the outcome agree with the evaluator
const BUDGET_TASK: &str = "fn worker(c: chan i64)
    with budget 30
        for i in 0..10
            c <- i
            emit i
fn main()
    let c = chan[i64](16)
    scope
        spawn worker(c)
    loop bound 20
        select
            recv v from c -> emit v
            default -> break
";

/// return, break and continue leaving scope and deadline bodies
const ESCAPES: &str = "fn put(c: chan i64, k: i64)
    c <- k
fn first(c: chan i64) -> i64
    scope
        spawn put(c, 4)
        if k0(c)
            return 40
    0
fn k0(_c: chan i64) -> bool
    true
fn timed(c: chan i64) -> i64
    with deadline 50
        match <- c
            some(v) -> return v + 1
            none -> return 0
    else
        if k0(c)
            return 99
    7
fn main()
    let c = chan[i64](4)
    emit first(c)
    emit timed(c)
    var n: i64 = 0
    for i in 0..6
        scope
            if i == 1
                continue
            if i == 4
                break
            spawn put(c, i)
        n += 1
    emit n
    loop bound 10
        with deadline 3
            match <- c
                some(v) -> emit v
                none -> break
        else
            break
    emit 1000
";

const WAIT_CHAIN: &str = "fn main()
    let a = lvar_new(0)
    let b = lvar_new(0)
    let c = lvar_new(0)
    var x: i64 = 0
    var y: i64 = 0
    par
        branch one
            x = a.wait([2]) + 1
        branch two
            y = b.wait([3])
            a.put(2)
        branch three
            c.put(1)
            b.put(3)
    emit x * 10 + y
    emit c.freeze()
";

const WAIT_STUCK: &str = "fn main()
    let a = lvar_new(0)
    let b = lvar_new(0)
    var x: i64 = 0
    var y: i64 = 0
    par
        branch one
            x = a.wait([1])
            b.put(1)
        branch two
            y = b.wait([1])
            a.put(1)
    emit x + y
";

const PAR_MERGE: &str = "struct P
    x: i64
    y: i64
fn main()
    var p = P(x: 0, y: 0)
    var xs = [0, 0, 0]
    var s: i64 = 0
    par
        branch one
            p.x = 5
            xs[0] = 1
        branch two
            p.y = 7
            xs[2] = 3
        branch three
            for i in 0..10
                s += i
    emit p
    emit xs
    emit s
";

/// `with budget n ... else ...`: the innermost budget with an else catches a
/// budget trap of its body; writes before the trap stay; with an outer limit
/// passed the else's own charges trap again and the outer else catches
fn budget_else(b: i64) -> String {
    format!(
        "fn main()
    var a: i64 = 0
    var r: i64 = 0
    var xs = [0, 0]
    var u: u64 = 0
    var k = Exhausted(limit: 0, used: 0)
    with budget {b}
        with budget 40
            for _i in 0..100
                a += 1
                xs[0] = a
        else as e
            r = 1
            u = e.used
            k = e
        for _i in 0..20
            a += 1
            xs[1] = a
    else as e2
        r += 10
        k = e2
    emit a
    emit r
    emit xs
    emit u
    emit k
    with budget 5
        for _i in 0..10
            a += 1
    else
        emit 7
    emit a
"
    )
}

/// par budget exhaustion: the branches run at the same time on their own
/// threads, but the trap is placed where the sequential run would hit it
fn par_budget(b: i64) -> String {
    format!(
        "fn main()
    var a: i64 = 0
    var c: i64 = 0
    var d: i64 = 0
    emit 1
    with budget {b}
        par
            branch one
                for i in 0..6
                    a += i
            branch two
                for i in 0..6
                    c += i * 2
            branch three
                for _i in 0..6
                    d += 1
    emit a + c + d
"
    )
}

/// `for ... par` on threads: float and integer reductions (the fixed tree
/// makes the float sum bit-exact), slot writes, a product, nested in a par
const FORPAR: &str = "error Bad

fn sq(i: i64) -> i64
    i * i

fn main()
    var s: i64 = 0
    var f: f64 = 0.0
    var m: i64 = 1
    var out: [i64; 500] = [0; 500]
    for i in 0..500 par
        s += sq(i as i64)
        f += 1.0 / (i as f64 + 1.0)
    for i in 0..500 par
        out[i] = (i as i64) * 3
    for i in 1..12 par
        m *= i
    emit s
    emit f
    emit m
    emit out[0] + out[499] + out[250]
    var a: i64 = 0
    var b: i64 = 0
    par
        branch one
            for i in 0..100 par
                a += i
        branch two
            for i in 0..100 par
                b += 2 * i
    emit a + b
";

/// a failing iteration: the first failure in iteration order decides
const FORPAR_FAIL: &str = "error Bad(i: i64)

fn chk(i: i64) -> i64 ! Bad
    if i == 333 or i == 777
        fail Bad(i: i)
    i

fn run() -> i64 ! Bad
    var s: i64 = 0
    for i in 0..1000 par
        s += try chk(i)
    s

fn main()
    emit run() catch |_e| -1
    emit run() catch |e| e.i
";

/// `for ... par` budget exhaustion, placed as in the sequential run
fn forpar_budget(b: i64) -> String {
    format!(
        "fn main()
    var s: i64 = 0
    emit 1
    with budget {b}
        for i in 0..40 par
            s += i * i
    emit s
"
    )
}

const ACTORS: &str = "event Ping(k: i64)

actor A
    n: i64 = 0
    on Ping(k: k)
        n += k
        emit k

actor B
    n: i64 = 0
    on Ping(k: k)
        emit k + 1000

node east { A }
node west { B }

fn main()
    let a = A()
    let b = B()
    for i in 0..4
        a <- Ping(k: i)
        b <- Ping(k: i)
";

const ATOMIC_ACTORS: &str = "event Add(k: i64)
event Show
error Neg

actor Acc
    n: i64 = 0
    on Add(k: k)
        if k < 0
            fail Neg
        n += k
    on Show
        emit n

fn main()
    let a = Acc()
    let b = Acc()
    atomic
        a <- Add(k: 5)
        b <- Add(k: -1)
    a <- Show
    b <- Show
    atomic
        a <- Add(k: 2)
        b <- Add(k: 3)
    a <- Show
    b <- Show
";

const SUPERVISOR: &str = "event Hit(k: i64)
event Show
error Boom

actor Worker
    n: i64 = 0
    on Hit(k: k)
        n += k
        if k == 0
            fail Boom
    on Show
        emit n

supervisor Root strategy one_for_one
    child w: Worker restart bound 2

fn main()
    let w = Worker()
    w <- Hit(k: 5)
    w <- Show
    w <- Hit(k: 0)
    w <- Show
    w <- Hit(k: 0)
    w <- Hit(k: 0)
    w <- Show
";

/// actors that send to each other and to a channel read by a task
const ACTOR_PIPE: &str = "event Num(k: i64)

actor Doubler
    seen: i64 = 0
    on Num(k: k)
        seen += 1
        emit k * 2

fn feeder(d: Doubler, c: chan i64)
    for i in 0..5
        d <- Num(k: i)
        c <- i
    c.close()

fn main()
    let d = Doubler()
    let c = chan[i64](2)
    spawn feeder(d, c)
    var s: i64 = 0
    loop
        match <- c
            some(v) -> s += v
            none -> break
    emit s
";

const SESSION: &str = "protocol Lookup
    send text
    offer
        Found
            recv i64
        Missing
    loop
        choose
            More
                send text
                recv i64
            Done
                break

protocol Serve
    recv text
    choose
        Found
            send i64
        Missing
    loop
        offer
            More
                recv text
                send i64
            Done
                break

fn server(s: session Serve)
    let k = <- s
    if k == \"key\"
        s.choose(Found)
        s <- 7
    else
        s.choose(Missing)
    loop
        match s.offer()
            More
                let _q = <- s
                s <- 1
            Done -> break

fn main()
    let (c, s) = chan[session Lookup](1)
    spawn server(s)
    c <- \"key\"
    match c.offer()
        Found
            let v = <- c
            emit v
        Missing -> emit 0
    for _i in 0..3
        c.choose(More)
        c <- \"q\"
        let w = <- c
        emit w
    c.choose(Done)
";

fn many(n: usize) -> String {
    format!(
        "fn w(c: chan i64, k: i64)
    for i in 0..3
        c <- k * 10 + i
fn main()
    let c = chan[i64](2)
    for k in 0..{n}
        spawn w(c, k)
    var s: i64 = 0
    loop bound {}
        match <- c
            some(v) -> s += v
            none -> break
        emit s
    emit s
",
        n * 3
    )
}

fn corpus() -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = vec![
        ("nondet".into(), NONDET.into()),
        ("rotate".into(), ROTATE.into()),
        ("deadline".into(), DEADLINE.into()),
        ("deadline_ok".into(), DEADLINE_OK.into()),
        ("deadlock".into(), DEADLOCK.into()),
        ("backpressure".into(), BACKPRESSURE.into()),
        ("scope_fail".into(), SCOPE_FAIL.into()),
        ("scope".into(), SCOPE.into()),
        ("budget_task".into(), BUDGET_TASK.into()),
        ("budget_else_big".into(), budget_else(100000)),
        ("budget_else_mid".into(), budget_else(60)),
        ("budget_else_small".into(), budget_else(20)),
        ("many".into(), many(24)),
        ("escapes".into(), ESCAPES.into()),
        ("wait_chain".into(), WAIT_CHAIN.into()),
        ("wait_stuck".into(), WAIT_STUCK.into()),
        ("par_merge".into(), PAR_MERGE.into()),
        ("actors".into(), ACTORS.into()),
        ("atomic_actors".into(), ATOMIC_ACTORS.into()),
        ("supervisor".into(), SUPERVISOR.into()),
        ("actor_pipe".into(), ACTOR_PIPE.into()),
        ("session".into(), SESSION.into()),
    ];
    v.push(("forpar".into(), FORPAR.into()));
    v.push(("forpar_fail".into(), FORPAR_FAIL.into()));
    for b in [5, 40, 78, 79, 80, 81, 82, 83, 84, 1000] {
        v.push((format!("forpar_budget_{b}"), forpar_budget(b)));
    }
    for b in [10, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60] {
        v.push((format!("par_budget_{b}"), par_budget(b)));
    }
    for f in ["../examples/13_spawn_chan.rce", "../examples/02_hello_main.rce", "../conc06/accept/05_priority_spawn.rce", "../conc06/accept/08_run_main.rce", "../conc06b/accept/04_scope_atomic.rce"] {
        let stem = PathBuf::from(f).file_stem().unwrap().to_string_lossy().replace('-', "_");
        v.push((format!("x{stem}"), std::fs::read_to_string(f).unwrap()));
    }
    v.extend(drivers());
    v
}

/// the concurrency examples 14, 15, 34 and 47 declare no `fn main`; each is
/// run with a driver from tests/drivers/ appended to its unchanged text
fn drivers() -> Vec<(String, String)> {
    ["14_select", "15_actor", "34_chan_types", "47_select_fair"]
        .iter()
        .map(|n| {
            let ex = std::fs::read_to_string(format!("../examples/{n}.rce")).unwrap();
            let main = std::fs::read_to_string(format!("tests/drivers/{n}.main.rce")).unwrap();
            assert!(!ex.contains("fn main"), "{n} gained a main; drop its driver");
            (format!("d{n}"), ex + &main)
        })
        .collect()
}

#[test]
#[cfg_attr(not(rce_emit = "rust"), ignore = "tests/emit_targets.txt does not name rust")]
fn emitted_runtime_agrees_with_run_main() {
    require(Lang::Rust);
    let progs = corpus();
    let results: Vec<(String, Vec<String>, PathBuf)> = std::thread::scope(|s| {
        let hs: Vec<_> = progs.iter().map(|(n, src)| s.spawn(move || (n.clone(), want(src), build(n, src)))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut bad = vec![];
    for (n, w, exe) in &results {
        eprintln!("{n}: {}", w.join(" | "));
        // stress: every rerun must agree (thread interleavings vary per run)
        for k in 0..8 {
            let g = got(exe);
            if &g != w {
                bad.push(format!("{n} (run {k}):\n want {w:?}\n got  {g:?}"));
                break;
            }
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
    eprintln!("{} task programs agree with run_main, 8 runs each", results.len());
}

/// stress: the lvar wait chain and the stuck pair on real threads, 100 runs
/// each; every run must equal the evaluator's result
#[test]
#[cfg_attr(not(rce_emit = "rust"), ignore = "tests/emit_targets.txt does not name rust")]
fn par_waits_stress() {
    require(Lang::Rust);
    for (n, src) in [("stress_chain", WAIT_CHAIN), ("stress_stuck", WAIT_STUCK)] {
        let w = want(src);
        let exe = build(n, src);
        for k in 0..100 {
            assert_eq!(got(&exe), w, "{n} run {k}");
        }
    }
}

/// partitions (C.11.4): logged link inputs hold messages across the link and
/// Heal delivers them in order, as in run_main with the same inputs
#[test]
#[cfg_attr(not(rce_emit = "rust"), ignore = "tests/emit_targets.txt does not name rust")]
fn partitions_agree() {
    require(Lang::Rust);
    let exe = build("partitions", ACTORS);
    let cases: Vec<Links> = vec![
        vec![],
        vec![(1, true, "main".into(), "west".into()), (9, false, "main".into(), "west".into())],
        vec![(2, true, "main".into(), "west".into()), (6, false, "main".into(), "west".into())],
        vec![(1, true, "main".into(), "east".into())],
        vec![(1, true, "main".into(), "west".into()), (3, true, "main".into(), "east".into()), (5, false, "main".into(), "west".into()), (7, false, "main".into(), "east".into())],
    ];
    for l in &cases {
        let w = want_links(ACTORS, l);
        eprintln!("{l:?}: {}", w.join(" | "));
        for _ in 0..4 {
            assert_eq!(got_links(&exe, l), w, "{l:?}");
        }
    }
}

fn agree_lang(lang: Lang, runs: usize) {
    let progs = corpus();
    let results: Vec<(String, Vec<String>, Exe)> = std::thread::scope(|s| {
        let hs: Vec<_> = progs.iter().map(|(n, src)| s.spawn(move || (n.clone(), want(src), build_lang(lang, n, src)))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut bad = vec![];
    for (n, w, exe) in &results {
        for k in 0..runs {
            let g = run_exe(exe, &vec![]);
            if &g != w {
                bad.push(format!("{n} (run {k}):\n want {w:?}\n got  {g:?}"));
                break;
            }
        }
    }
    assert!(bad.is_empty(), "{lang:?}:\n{}", bad.join("\n"));
    let exe = build_lang(lang, "partitions", ACTORS);
    for l in partition_cases() {
        assert_eq!(run_exe(&exe, &l), want_links(ACTORS, &l), "{lang:?} {l:?}");
    }
    eprintln!("{lang:?}: {} task programs agree with run_main, {runs} runs each, plus partitions", results.len());
}

fn partition_cases() -> Vec<Links> {
    vec![
        vec![],
        vec![(1, true, "main".into(), "west".into()), (9, false, "main".into(), "west".into())],
        vec![(2, true, "main".into(), "west".into()), (6, false, "main".into(), "west".into())],
        vec![(1, true, "main".into(), "east".into())],
        vec![(1, true, "main".into(), "west".into()), (3, true, "main".into(), "east".into()), (5, false, "main".into(), "west".into()), (7, false, "main".into(), "east".into())],
    ]
}

#[test]
#[cfg_attr(not(rce_emit = "python"), ignore = "tests/emit_targets.txt does not name python")]
fn python_runtime_agrees_with_run_main() {
    eprintln!("python oracle: {}", python());
    require(Lang::Python);
    let o = Command::new(python()).args(["-c", "import sys; print(sys.version, getattr(sys, '_is_gil_enabled', lambda: True)())"]).output().unwrap();
    eprintln!("python: {}", String::from_utf8_lossy(&o.stdout).trim());
    agree_lang(Lang::Python, 3);
}

#[test]
#[cfg_attr(not(rce_emit = "js"), ignore = "tests/emit_targets.txt does not name js")]
fn js_runtime_agrees_with_run_main() {
    require(Lang::Js);
    let o = Command::new(node()).arg("--version").output().unwrap();
    eprintln!("node: {}", String::from_utf8_lossy(&o.stdout).trim());
    agree_lang(Lang::Js, 2);
}

/// kept handler steps of every kind: a record failure, traps (overflow,
/// division by zero, index), an atomic transaction that keeps the state, and a supervisor
/// that gives up
const KEPT_STEPS: &str = "event Hit(k: i64)
event Div(k: i64)
event Idx(k: i64)
event Show
error Boom

actor Worker
    n: i64 = 0
    xs: List[i64] = [1, 2]
    on Hit(k: k)
        n += k
        if k == 0
            fail Boom
    on Div(k: k)
        n = 10 / k
    on Idx(k: k)
        n = xs[k as usize]
    on Show
        emit n

actor Big
    n: i64 = 9223372036854775800
    on Hit(k: k)
        n += k
    on Show
        emit n

supervisor Root strategy one_for_all
    child w: Worker restart bound 3

fn main()
    let w = Worker()
    let b = Big()
    w <- Hit(k: 5)
    w <- Div(k: 0)
    w <- Show
    w <- Idx(k: 7)
    b <- Hit(k: 100)
    b <- Show
    atomic
        b <- Hit(k: 1)
        w <- Hit(k: 0)
    w <- Hit(k: 0)
    w <- Hit(k: 0)
    w <- Show
";

fn want_log(src: &str) -> Vec<String> {
    let m = parse(src).unwrap();
    let r = run_main(&m, &RunCfg::default()).unwrap();
    let mut v: Vec<String> = r.outputs.iter().map(|x| format!("emit {}", x.show())).collect();
    for e in &r.log {
        if ["Kept(", "Stopped(", "Restart(", "Input(", "Unhandled("].iter().any(|p| e.what.starts_with(p)) {
            v.push(format!("actorlog {}", e.what));
        }
    }
    v.push(match r.outcome {
        Outcome::Done => "outcome Done".into(),
        Outcome::Deadlock => "outcome Deadlock".into(),
        Outcome::Stopped(e) => format!("outcome Stopped: {e}"),
    });
    v
}

fn run_log(exe: &Exe) -> Vec<String> {
    let mut c = Command::new("timeout");
    c.arg("120");
    if !exe.0.is_empty() {
        c.arg(&exe.0);
    }
    let o = c.arg(&exe.1).args(["run", "actor-log"]).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout).unwrap().lines().map(String::from).collect()
}

#[test]
fn actor_kept_outcomes_match() {
    let d = rce_spec_check::check_program(&[("t.rce".into(), KEPT_STEPS.to_string())], false);
    assert!(d[0].1.is_ok(), "not admitted: {:?}", d[0].1);
    let w = want_log(KEPT_STEPS);
    eprintln!("{w:#?}");
    assert!(w.iter().filter(|l| l.starts_with("actorlog Kept")).count() >= 4, "{w:?}");
    // the same program with an intensity window (C.11.11): restarts older
    // than RESTART_WINDOW ticks no longer count toward the bound
    let windowed = format!("const RESTART_WINDOW: u64 = 2\n{KEPT_STEPS}");
    let ww = want_log(&windowed);
    assert_ne!(ww, w, "the window must change the outcome of this program");
    // a crash inside `atomic` is supervised: the failing actor restarts
    assert!(w.iter().filter(|l| l.starts_with("actorlog Restart(")).count() >= 2, "{w:?}");
    for lang in targets::covered(&[Lang::Rust, Lang::Python, Lang::Js], oracle) {
        let exe = build_lang(lang, &format!("kept_{lang:?}"), KEPT_STEPS);
        assert_eq!(run_log(&exe), w, "{lang:?}");
        let exe = build_lang(lang, &format!("kept_win_{lang:?}"), &windowed);
        assert_eq!(run_log(&exe), ww, "windowed {lang:?}");
    }
}

fn fmt_restart(r: &rce_spec_check::conc::RestartIn) -> String {
    format!("{}|{}|{}|{}|{}|{}", r.tick, r.sup, r.child, r.id, u8::from(r.stop), r.reset.join(","))
}

fn want_log_cfg(src: &str, cfg: &RunCfg) -> Vec<String> {
    let m = parse(src).unwrap();
    let r = run_main(&m, cfg).unwrap();
    let mut v: Vec<String> = r.outputs.iter().map(|x| format!("emit {}", x.show())).collect();
    for e in &r.log {
        if ["Kept(", "Stopped(", "Restart(", "Input(", "Unhandled("].iter().any(|p| e.what.starts_with(p)) {
            v.push(format!("actorlog {}", e.what));
        }
    }
    v.push(match r.outcome {
        Outcome::Done => "outcome Done".into(),
        Outcome::Deadlock => "outcome Deadlock".into(),
        Outcome::Stopped(e) => format!("outcome Stopped: {e}"),
    });
    v
}

fn run_args(exe: &Exe, extra: &[String]) -> Vec<String> {
    let mut c = Command::new("timeout");
    c.arg("120");
    if !exe.0.is_empty() {
        c.arg(&exe.0);
    }
    let o = c.arg(&exe.1).args(["run", "actor-log"]).args(extra).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout).unwrap().lines().map(String::from).collect()
}

/// restart as input on every target (C.11.11): each runtime records the
/// same RestartIn decisions as the reference (`restart-log`), and replays
/// a logged list (`restart ...`) under a changed policy exactly as the
/// reference does, the logged decisions governing; a decision for another
/// actor is `replay diverged`
#[test]
fn restart_inputs_replay_on_every_target() {
    let m = parse(KEPT_STEPS).unwrap();
    let rec = run_main(&m, &RunCfg::default()).unwrap();
    assert!(rec.restarts.len() >= 2);
    let logged: Vec<String> = rec.restarts.iter().map(fmt_restart).collect();
    // a changed policy (ids are stable paths, C.11.11): a looser bound
    let windowed = KEPT_STEPS.replace("restart bound 3", "restart bound 9");
    assert_ne!(windowed, KEPT_STEPS);
    let want_replay = want_log_cfg(&windowed, &RunCfg { restarts: Some(rec.restarts.clone()), ..RunCfg::default() });
    let w0 = want_log(KEPT_STEPS);
    // the stop message names the bound in the source; the decision is the log's
    let plain = |v: &[String]| {
        v.iter()
            .filter(|l| !l.starts_with("actorlog Restart("))
            .map(|l| l.split(" restarted more than").next().unwrap().to_string())
            .collect::<Vec<_>>()
    };
    assert_eq!(plain(&want_replay), plain(&w0), "logged decisions govern under the windowed policy");
    let mut bad = rec.restarts.clone();
    bad[0].id = "nobody".into();
    let want_bad = want_log_cfg(KEPT_STEPS, &RunCfg { restarts: Some(bad.clone()), ..RunCfg::default() });
    assert!(want_bad.last().unwrap().ends_with("replay diverged"), "{want_bad:?}");
    for lang in targets::covered(&[Lang::Rust, Lang::Python, Lang::Js], oracle) {
        let exe = build_lang(lang, &format!("rin_{lang:?}"), KEPT_STEPS);
        let got: Vec<String> = run_args(&exe, &["restart-log".into()])
            .into_iter()
            .filter_map(|l| l.strip_prefix("restartin ").map(String::from))
            .collect();
        assert_eq!(got, logged, "{lang:?} records the reference's decisions");
        let args: Vec<String> = logged.iter().flat_map(|l| ["restart".to_string(), l.to_string()]).collect();
        let exe_w = build_lang(lang, &format!("rin_win_{lang:?}"), &windowed);
        assert_eq!(run_args(&exe_w, &args), want_replay, "{lang:?} replays the logged decisions");
        let bad_args: Vec<String> = bad.iter().flat_map(|r| ["restart".to_string(), fmt_restart(r)]).collect();
        assert_eq!(run_args(&exe, &bad_args), want_bad, "{lang:?} diverges on a mismatched decision");
    }
}

/// `run` on a module without `fn main()` fails as `rce run` does
/// (10.3): status 1 and the reference's message, in every target, rather
/// than starting a scheduler with no task
#[test]
fn run_without_main_requires_change() {
    for n in ["14_select", "15_actor", "34_chan_types", "47_select_fair"] {
        let src = std::fs::read_to_string(format!("../examples/{n}.rce")).unwrap();
        let m = parse(&src).unwrap();
        let want = run_main(&m, &RunCfg::default()).err().expect("no main");
        for lang in [Lang::Rust, Lang::Python, Lang::Js] {
            let exe = build_lang(lang, &format!("nomain_{n}"), &src);
            let mut c = Command::new("timeout");
            c.arg("60");
            if !exe.0.is_empty() {
                c.arg(&exe.0);
            }
            let o = c.arg(&exe.1).arg("run").output().unwrap();
            assert_eq!(o.status.code(), Some(1), "{n} {lang:?}: {}", String::from_utf8_lossy(&o.stderr));
            assert!(o.stdout.is_empty(), "{n} {lang:?}");
            assert_eq!(String::from_utf8_lossy(&o.stderr).trim(), want, "{n} {lang:?}");
        }
    }
}

const EVENTS: &str = "event Hit(k: i64)
event Tag(flag: bool, who: text)
event Show
event Nobody

actor Acc
    n: i64 = 0
    on Hit(k: k)
        n += k
    on Tag(flag: f, who: w)
        if f
            emit w
    on Show
        emit n

actor Echo
    on Hit(k: k)
        emit k

actor Ghost
    on Nobody
        emit 9

fn main()
    let a = Acc()
    let _e = Echo()
    a <- Show
";

/// HC-6 on every target: program events given as `event` words reach
/// the live actors that handle them in actor id order, are logged as
/// `Input(..)`, and an event no actor handles is logged `Unhandled(..)`;
/// each runtime gives the reference's outputs, log and outcome
#[test]
fn program_events_agree_on_every_target() {
    use rce_spec_check::interp::V;
    let int = |k: i64| V::Int(k as i128, "i64");
    let rec = |n: &str, fs: Vec<(&str, V)>| V::Rec(n.into(), fs.into_iter().map(|(a, b)| (a.to_string(), b)).collect());
    let cfg = RunCfg {
        events: vec![
            (5, rec("Hit", vec![("k", int(3))])),
            (6, rec("Hit", vec![("k", int(-4))])),
            (7, rec("Tag", vec![("flag", V::Bool(true)), ("who", V::Text("ann".into()))])),
            (7, rec("Tag", vec![("flag", V::Bool(false)), ("who", V::Text("bob".into()))])),
            (8, rec("Show", vec![])),
            (12, rec("Nobody", vec![])),
        ],
        ..RunCfg::default()
    };
    let want = want_log_cfg(EVENTS, &cfg);
    assert!(want.iter().any(|l| l == "actorlog Unhandled(Nobody)"), "{want:?}");
    assert!(want.iter().any(|l| l == "emit -1"), "{want:?}");
    assert_eq!(want.last().unwrap(), "outcome Done");
    let args: Vec<String> = ["5|Hit|k=3", "6|Hit|k=-4", "7|Tag|flag=true,who=ann", "7|Tag|flag=false,who=bob", "8|Show", "12|Nobody"]
        .iter()
        .flat_map(|e| ["event".to_string(), e.to_string()])
        .collect();
    for lang in run_targets() {
        let exe = build_lang(lang, &format!("ev_{lang:?}"), EVENTS);
        assert_eq!(run_args(&exe, &args), want, "{lang:?} takes program events as the reference does");
        // a malformed event is an input error (status 2), not a program trap
        let mut c = Command::new("timeout");
        c.arg("60");
        if !exe.0.is_empty() {
            c.arg(&exe.0);
        }
        let o = c.arg(&exe.1).args(["run", "event", "3|Hit|k=x"]).output().unwrap();
        assert_eq!(o.status.code(), Some(2), "{lang:?}");
    }
}

/// C.11.11 stable ids: actor and task ids are name-free paths (tree
/// position, creating task, ordinal; a task by parent and ordinal), so a
/// restart log replays on an upgraded source that shifts every line, adds
/// a function, an actor type and a later supervisor; an upgrade that
/// removes or moves a recorded position stops before any step with a
/// defined trap (schema check),
/// and a log without a schema header traps at the first decision whose
/// position is gone, on every target as in the reference
#[test]
fn restart_replay_survives_upgrades_and_traps_on_tree_changes() {
    let m = parse(KEPT_STEPS).unwrap();
    let rec = run_main(&m, &RunCfg::default()).unwrap();
    assert!(rec.restarts.len() >= 2);
    assert_eq!(rec.tree, vec!["Root/w".to_string()]);
    let logged: Vec<String> = rec.restarts.iter().map(fmt_restart).collect();
    let upgraded = format!(
        "const UNUSED: i64 = 7\n\nfn helper(x: i64) -> i64\n    x + 1\n\nactor Zed\n    on Show\n        emit 0\n\n{}\nsupervisor Spare strategy one_for_one\n    child z: Zed restart bound 1\n",
        KEPT_STEPS
    );
    let up = parse(&upgraded).unwrap();
    assert_eq!(rec.restarts, run_main(&up, &RunCfg::default()).unwrap().restarts, "ids survive the upgrade");
    let replay = |src: &str, tree: Option<Vec<String>>| {
        want_log_cfg(src, &RunCfg { restarts: Some(rec.restarts.clone()), restart_tree: tree, ..RunCfg::default() })
    };
    let w_up = replay(&upgraded, Some(rec.tree.clone()));
    assert_eq!(w_up, want_log(KEPT_STEPS), "the log replays on the upgraded source");
    let renamed = KEPT_STEPS.replace("child w: Worker", "child w2: Worker");
    let w_ren = replay(&renamed, Some(rec.tree.clone()));
    assert_eq!(w_ren, vec!["outcome Stopped: trap: upgrade removed supervisor position Root/w".to_string()]);
    let w_ren0 = replay(&renamed, None);
    assert!(w_ren0.last().unwrap().ends_with("trap: upgrade removed supervisor position Root/w"), "{w_ren0:?}");
    let reord = vec!["Spare/z".to_string(), "Root/w".to_string()];
    let w_reord = replay(&upgraded, Some(reord.clone()));
    assert_eq!(w_reord, vec!["outcome Stopped: trap: upgrade reordered supervisor position Spare/z".to_string()]);
    let rargs = |tree: Option<&[String]>| {
        let mut a: Vec<String> = logged.iter().flat_map(|l| ["restart".to_string(), l.to_string()]).collect();
        if let Some(t) = tree {
            a.extend(["restart-tree".to_string(), t.join(",")]);
        }
        a
    };
    for lang in targets::covered(&[Lang::Rust, Lang::Python, Lang::Js], oracle) {
        let exe = build_lang(lang, &format!("rup_rec_{lang:?}"), KEPT_STEPS);
        let log = run_args(&exe, &["restart-log".into()]);
        assert!(log.contains(&"restartlog v1 tree=Root/w".to_string()), "{lang:?} {log:?}");
        let exe_up = build_lang(lang, &format!("rup_up_{lang:?}"), &upgraded);
        assert_eq!(run_args(&exe_up, &rargs(Some(&rec.tree))), w_up, "{lang:?} replays across the upgrade");
        assert_eq!(run_args(&exe_up, &rargs(Some(&reord))), w_reord, "{lang:?} reorder trap");
        let exe_ren = build_lang(lang, &format!("rup_ren_{lang:?}"), &renamed);
        assert_eq!(run_args(&exe_ren, &rargs(Some(&rec.tree))), w_ren, "{lang:?} removal trap");
        assert_eq!(run_args(&exe_ren, &rargs(None)), w_ren0, "{lang:?} per-decision position check");
    }
}

/// the Scala tick runtime (C.11.11, HC-6, 6.6): actor programs give the
/// reference's outputs, actor log and outcome, as the other targets do:
/// kept handler steps, arithmetic and index traps, supervision with restarts
/// and a stop, atomic batches, transactions across actors, and tasks with
/// channels
#[test]
#[cfg_attr(not(rce_emit = "scala"), ignore = "tests/emit_targets.txt does not name scala")]
fn scala_actor_programs_agree_with_the_reference() {
    require(Lang::Scala);
    assert!(java_bin().is_some(), "scala: no JVM for the assemblies");
    let exe = build_lang(Lang::Scala, "sc_kept", KEPT_STEPS);
    assert_eq!(run_log(&exe), want_log(KEPT_STEPS), "kept steps and supervision");
    let rec = run_args(&exe, &["restart-log".into()]);
    let m = parse(KEPT_STEPS).unwrap();
    let r = run_main(&m, &RunCfg::default()).unwrap();
    let want_rin: Vec<String> = r.restarts.iter().map(fmt_restart).collect();
    let got_rin: Vec<String> = rec.iter().filter_map(|l| l.strip_prefix("restartin ").map(String::from)).collect();
    assert_eq!(got_rin, want_rin, "restart inputs");
    for n in ["37_actor_spawn"] {
        let p = format!("../examples/{n}.rce");
        let Ok(src) = std::fs::read_to_string(&p) else { continue };
        if !src.contains("fn main()") {
            continue;
        }
        let exe = build_lang(Lang::Scala, &format!("sc_{n}"), &src);
        assert_eq!(run_log(&exe), want_log(&src), "{n}");
    }
    for (n, src) in [
        ("actors", ACTORS),
        ("atomic_actors", ATOMIC_ACTORS),
        ("supervisor", SUPERVISOR),
        ("actor_pipe", ACTOR_PIPE),
        ("scope_fail", SCOPE_FAIL),
        ("backpressure", BACKPRESSURE),
        ("deadline_ok", DEADLINE_OK),
        ("deadlock", DEADLOCK),
        ("nondet", NONDET),
        ("session", SESSION),
    ] {
        let exe = build_lang(Lang::Scala, &format!("sc_{n}"), src);
        assert_eq!(run_log(&exe), want_log(src), "{n}");
    }
}

/// The trace of `fn main` (10.9) as an emitted program prints it under
/// `run trace`: one `d applied|kept HASH BODY` line per step of the
/// reference's trace (`cli::main_trace`), BODY the canonical JSON its hash
/// is over, then `trace HEAD` and the outcome line of `run`
fn want_main_trace(src: &str, cfg: &RunCfg) -> Vec<String> {
    use rce_spec_check::interp::{D_GENESIS, canon_json};
    let d = rce_spec_check::check_program(&[("t.rce".into(), src.to_string())], false);
    assert!(d[0].1.is_ok(), "not admitted: {:?}\n{src}", d[0].1);
    let t = rce_spec_check::cli::main_trace(&parse(src).unwrap(), cfg).unwrap();
    let mut v = vec![];
    let mut head = D_GENESIS.to_string();
    for s in t["steps"].as_array().unwrap() {
        let body = canon_json(&serde_json::json!({
            "event": s["hashed"]["event"], "prev_hash": s["prev_hash"], "seq": s["seq"], "state_hash": s["state_hash"],
        }));
        head = s["hash"].as_str().unwrap().to_string();
        v.push(format!("d {} {head} {body}", if s["outcome"] == "applied" { "applied" } else { "kept" }));
    }
    v.push(format!("trace {head}"));
    v.push(match &t["outcome"] {
        o if o == "done" => "outcome Done".into(),
        o if o == "deadlock" => "outcome Deadlock".into(),
        o => format!("outcome Stopped: {}", o["stopped"].as_str().unwrap()),
    });
    v
}

/// the trace of `fn main` that an emitted program prints with `run trace`
fn got_main_trace(exe: &Exe, extra: &[String]) -> Vec<String> {
    let mut c = Command::new("timeout");
    c.arg("120");
    if !exe.0.is_empty() {
        c.arg(&exe.0);
    }
    let o = c.arg(&exe.1).args(["run", "trace"]).args(extra).output().unwrap();
    assert!(o.status.success(), "{:?}: {}", exe.1, String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout)
        .unwrap()
        .lines()
        .filter(|l| l.starts_with("d ") || l.starts_with("trace ") || l.starts_with("outcome "))
        .map(String::from)
        .collect()
}

/// The trace of `fn main` (10.9) on every covered target: for every task
/// and actor program of this file, the spec examples with `fn main()`, the
/// partition cases and the program events, each runtime prints, under
/// `run trace`, the reference's steps (entry, state hash, chain hash),
/// head and outcome, line for line. The programs give the reference steps
/// of every kind the tick runtime writes; each kind is checked to occur.
/// The runs of `fn main` the trace tests compare: (name, program, inputs,
/// the same inputs as arguments of an emitted program's `run trace`).
fn main_trace_cases() -> Vec<(String, String, RunCfg, Vec<String>)> {
    let mut progs = corpus();
    progs.push(("kept".into(), KEPT_STEPS.into()));
    for f in ["../examples/37_actor_spawn.rce", "../conc06b/accept/07_session_tree.rce", "../conc06b/accept/08_session_priorities.rce"] {
        let stem = PathBuf::from(f).file_stem().unwrap().to_string_lossy().into_owned();
        progs.push((format!("x{stem}"), std::fs::read_to_string(f).unwrap()));
    }
    let none = RunCfg::default();
    let mut cases: Vec<(String, String, RunCfg, Vec<String>)> = progs.into_iter().map(|(n, s)| (n, s, none.clone(), vec![])).collect();
    for (k, l) in partition_cases().into_iter().enumerate() {
        let args = l.iter().flat_map(|(t, p, a, b)| ["link".to_string(), format!("{t},{},{a},{b}", if *p { "partition" } else { "heal" })]).collect();
        cases.push((format!("partitions_{k}"), ACTORS.into(), RunCfg { links: l, ..RunCfg::default() }, args));
    }
    {
        use rce_spec_check::interp::V;
        let rec = |n: &str, fs: Vec<(&str, V)>| V::Rec(n.into(), fs.into_iter().map(|(a, b)| (a.to_string(), b)).collect());
        let cfg = RunCfg {
            events: vec![(5, rec("Hit", vec![("k", V::Int(3, "i64"))])), (8, rec("Show", vec![])), (12, rec("Nobody", vec![]))],
            ..RunCfg::default()
        };
        let args = ["5|Hit|k=3", "8|Show", "12|Nobody"].iter().flat_map(|e| ["event".to_string(), e.to_string()]).collect();
        cases.push(("events".into(), EVENTS.into(), cfg, args));
    }
    cases
}

#[test]
fn main_traces_agree_on_every_target() {
    let cases = main_trace_cases();
    let want: Vec<Vec<String>> = cases.iter().map(|(_, s, cfg, _)| want_main_trace(s, cfg)).collect();
    let kinds = [
        "Emit", "Send", "Recv", "Backpressure", "Close", "Atomic", "Deliver", "Kept", "Restart", "Stopped", "Input", "Unhandled",
        "Partition", "Heal", "Held", "Done", "Spawn", "Choice", "Cancelled",
    ];
    for k in kinds {
        let tag = format!("\"type\":\"{k}\"");
        assert!(want.iter().flatten().any(|l| l.contains(&tag)), "no reference step of kind {k}");
    }
    for lang in run_targets() {
        let exes: Vec<Exe> = std::thread::scope(|s| {
            let hs: Vec<_> = cases.iter().map(|(n, src, _, _)| s.spawn(move || build_lang(lang, &format!("mt_{n}"), src))).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut bad = vec![];
        for ((n, _, _, args), (exe, w)) in cases.iter().zip(exes.iter().zip(&want)) {
            let g = got_main_trace(exe, args);
            if &g != w {
                let k = g.iter().zip(w).position(|(a, b)| a != b).unwrap_or(g.len().min(w.len()));
                bad.push(format!("{n}: first difference at line {k}:\n want {:?}\n got  {:?}", w.get(k), g.get(k)));
            }
        }
        assert!(bad.is_empty(), "{lang:?}:\n{}", bad.join("\n"));
        eprintln!("{lang:?}: the trace of fn main agrees with the reference on {} programs", cases.len());
    }
}

/// The tick runtime without task threads (10.3): on every run of
/// `main_trace_cases`, a run whose tasks are state machines of the
/// scheduler (`Tasks::Machines`, as on WASI preview 1) writes the log of
/// the reference run, whose tasks are threads, entry for entry (tick,
/// node, text, vector clock, cause and state), with the same outputs,
/// choices, ticks and outcome, and so the same trace of `fn main`.
/// Runs of `fn main` in which an lvar passes from one task to another
/// (10.3): (name, program, the outputs of the reference run, its outcome).
/// The handle passes by move (a `spawn` argument, a message to a channel
/// or an actor, a task's result) or is shared by a closure that captures
/// it, so two tasks hold it at once; `late_fill` and `owner` fill it after
/// the task that waits on it has started.
const LVAR_PASSING: &[(&str, &str, &[&str], &str)] = &[
    (
        "spawn_arg",
        "fn reader(l: lvar i64, out: chan i64)
    l.put(2)
    out <- l.freeze()

fn main()
    let out = chan[i64](1)
    let l = lvar_new(0)
    l.put(1)
    spawn reader(l, out)
    match <- out
        some(v) -> emit v
        none -> emit 0
",
        &["2"],
        "Done",
    ),
    (
        "message",
        "fn adder(c: chan lvar i64, done: chan i64)
    match <- c
        some(l) -> done <- adder_put(l)
        none -> done <- 0

fn adder_put(l: lvar i64) -> i64
    l.put(3)
    l.put(9)
    l.freeze()

fn main()
    let c = chan[lvar i64](1)
    let done = chan[i64](1)
    let l = lvar_new(0)
    spawn adder(c, done)
    l.put(1)
    c <- l
    match <- done
        some(v) -> emit v
        none -> emit 0
",
        &["9"],
        "Done",
    ),
    (
        "result",
        "fn maker(c: chan i64, n: i64) -> lvar i64
    let l = lvar_new(0)
    l.put(n)
    c <- n
    l

fn main()
    let c = chan[i64](1)
    scope
        spawn maker(c, 4) -> h
        let l = h.join() catch lvar_new(100)
        l.put(6)
        emit l.freeze()
",
        &["6"],
        "Done",
    ),
    (
        "actor",
        "event Fill(l: lvar i64, n: i64, back: chan lvar i64)

actor Filler
    seen: i64 = 0
    on Fill(l: l, n: n, back: b)
        seen += 1
        l.put(n)
        b <- l

fn main()
    let a = Filler()
    let back = chan[lvar i64](1)
    let l = lvar_new(0)
    l.put(1)
    a <- Fill(l: l, n: 4, back: back)
    match <- back
        some(m) -> emit m.wait([4])
        none -> emit 0
",
        &["4"],
        "Done",
    ),
    (
        "late_fill",
        "fn reader(f: fn(i64) -> i64, out: chan i64)
    emit 1
    out <- f(5) * 2

fn main()
    let l = lvar_new(0)
    let out = chan[i64](1)
    let gap = chan[i64](4)
    let f = fn(x: i64) => l.wait([x])
    spawn reader(f, out)
    gap <- 1
    gap <- 2
    match <- gap
        some(v) -> emit v
        none -> emit 0
    match <- gap
        some(v) -> emit v
        none -> emit 0
    l.put(5)
    match <- out
        some(v) -> emit v
        none -> emit 0
    emit l.freeze()
",
        &["1", "1", "2", "10", "5"],
        "Done",
    ),
    (
        "owner",
        "fn owner(back: chan fn(i64) -> (), out: chan i64)
    let l = lvar_new(0)
    let g = fn(x: i64) => l.put(x)
    back <- g
    let t = l.wait([7])
    out <- t + 1
    emit l.freeze()

fn main()
    let back = chan[fn(i64) -> ()](1)
    let out = chan[i64](1)
    spawn owner(back, out)
    match <- back
        some(g) -> g(7)
        none -> emit 0
    match <- out
        some(v) -> emit v
        none -> emit 0
",
        &["7", "8"],
        "Done",
    ),
    (
        "par_in_task",
        "fn worker(l: lvar i64, out: chan i64)
    let m = lvar_new(0)
    var x: i64 = 0
    par
        branch one
            x = m.wait([2]) + l.wait([4])
        branch two
            m.put(2)
    out <- x
    emit m.freeze()

fn main()
    let out = chan[i64](1)
    let l = lvar_new(0)
    l.put(4)
    spawn worker(l, out)
    match <- out
        some(v) -> emit v
        none -> emit 0
",
        &["2", "6"],
        "Done",
    ),
    (
        "never_filled",
        "fn reader(f: fn(i64) -> i64, out: chan i64)
    out <- f(5)

fn main()
    let l = lvar_new(0)
    let out = chan[i64](1)
    let f = fn(x: i64) => l.wait([x])
    spawn reader(f, out)
    l.put(2)
    emit l.freeze()
    match <- out
        some(v) -> emit v
        none -> emit 0
",
        &["2"],
        "Deadlock",
    ),
];

/// An lvar passed from one task to another (10.3): with every task a state
/// machine, with every task a thread, and with only `fn main` on a thread,
/// each run of `LVAR_PASSING` writes the same log, entry for entry, and
/// the outputs and outcome the reference gives. Twenty threaded runs of
/// each give that log too: once an lvar has passed, the tasks move one
/// after another, so the threaded run does not depend on the host's
/// interleaving.
#[test]
fn lvar_passing_traces_without_threads_equal_the_reference() {
    use rce_spec_check::conc::{Tasks, refuse_task_threads_after, run_main_traced};
    for (n, src, outs, outcome) in LVAR_PASSING {
        let m = parse(src).unwrap();
        let threads = RunCfg { tasks: Tasks::Threads, ..RunCfg::default() };
        let machines = RunCfg { tasks: Tasks::Machines, ..RunCfg::default() };
        let a = run_main_traced(&m, &threads).unwrap_or_else(|e| panic!("{n}: {e}"));
        let shown: Vec<String> = a.outputs.iter().map(|o| o.show()).collect();
        assert_eq!(shown, *outs, "{n}: outputs");
        assert_eq!(format!("{:?}", a.outcome), *outcome, "{n}: outcome");
        let log = |r: &rce_spec_check::conc::RunResult| r.log.iter().map(|e| format!("{e:?}")).collect::<Vec<_>>();
        let want = log(&a);
        let mut runs = vec![("machines", run_main_traced(&m, &machines))];
        refuse_task_threads_after(Some(1));
        runs.push(("main on a thread", run_main_traced(&m, &threads)));
        refuse_task_threads_after(None);
        for _ in 0..20 {
            runs.push(("threads", run_main_traced(&m, &threads)));
        }
        for (how, b) in runs {
            let b = b.unwrap_or_else(|e| panic!("{n} {how}: {e}"));
            assert_eq!(format!("{:?}", a.outcome), format!("{:?}", b.outcome), "{n} {how}");
            let got = log(&b);
            assert_eq!(want.len(), got.len(), "{n} {how}: log length");
            for (k, (x, y)) in want.iter().zip(&got).enumerate() {
                assert_eq!(x, y, "{n} {how}: entry {k}");
            }
            assert_eq!(format!("{:?}", a.outputs), format!("{:?}", b.outputs), "{n} {how}");
            assert_eq!((&a.choices, a.ticks), (&b.choices, b.ticks), "{n} {how}");
        }
        let ta = rce_spec_check::cli::main_trace(&m, &threads).unwrap();
        let tb = rce_spec_check::cli::main_trace(&m, &machines).unwrap();
        assert_eq!(ta, tb, "{n}");
    }
}

#[test]
fn main_traces_without_threads_equal_the_reference() {
    use rce_spec_check::conc::{Tasks, run_main_traced};
    let cases = main_trace_cases();
    for (n, src, cfg, _) in &cases {
        let m = parse(src).unwrap();
        let threads = RunCfg { tasks: Tasks::Threads, ..cfg.clone() };
        let machines = RunCfg { tasks: Tasks::Machines, ..cfg.clone() };
        let a = run_main_traced(&m, &threads).unwrap();
        let b = run_main_traced(&m, &machines).unwrap_or_else(|e| panic!("{n}: {e}"));
        assert_eq!(format!("{:?}", a.outcome), format!("{:?}", b.outcome), "{n}");
        assert_eq!(a.log.len(), b.log.len(), "{n}: log length");
        for (k, (x, y)) in a.log.iter().zip(&b.log).enumerate() {
            assert_eq!(format!("{x:?}"), format!("{y:?}"), "{n}: entry {k}");
        }
        assert_eq!(format!("{:?}", a.outputs), format!("{:?}", b.outputs), "{n}");
        assert_eq!((a.choices, a.ticks), (b.choices, b.ticks), "{n}");
        let ta = rce_spec_check::cli::main_trace(&m, &threads).unwrap();
        let tb = rce_spec_check::cli::main_trace(&m, &machines).unwrap();
        assert_eq!(ta, tb, "{n}");
    }
    assert_eq!(cases.len(), 66);
    eprintln!("the run without task threads equals the reference on {} runs", cases.len());
}
