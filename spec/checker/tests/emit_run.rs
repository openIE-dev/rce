//! The emitted Rust program in `--run` mode runs `fn main` on its own tick
//! runtime (one OS thread per task, a scheduler admitting moves tick by
//! tick). Its outputs and outcome are compared with the reference
//! `conc::run_main` on a corpus of task programs, and a stress pass reruns
//! each program and a many-task program repeatedly: every run must give the
//! same result as the evaluator, whatever the OS does with the threads.
#![cfg(feature = "conc06")]
use rce_spec_check::conc::{Outcome, RunCfg, run_main};
use rce_spec_check::emit::{Lang, emit};
use rce_spec_check::parse::parse;
use std::path::PathBuf;
use std::process::Command;

fn rustc() -> String {
    std::env::var("RCE_RUSTC").ok().filter(|v| !v.is_empty()).unwrap_or("rustc".into())
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
        Outcome::Refused(e) => format!("outcome Refused: {e}"),
    });
    v
}

/// the Python oracle for `--run`: RCE_PYTHON_RUN, else RCE_PYTHON, else a
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
    let tool = if lang == Lang::Python { python() } else { std::env::var("RCE_NODE").ok().filter(|v| !v.is_empty()).unwrap_or("node".into()) };
    (tool, f)
}

fn run_exe(exe: &Exe, links: &Links) -> Vec<String> {
    // `timeout` guards against a hung runtime (a deadlock in the runtime itself)
    let mut c = Command::new("timeout");
    c.arg("120");
    if !exe.0.is_empty() {
        c.arg(&exe.0);
    }
    c.arg(&exe.1);
    c.arg("--run");
    for (t, p, a, b) in links {
        c.arg(format!("--link={t},{},{a},{b}", if *p { "partition" } else { "heal" }));
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
    c.arg("--run");
    for (t, p, a, b) in links {
        c.arg(format!("--link={t},{},{a},{b}", if *p { "partition" } else { "heal" }));
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
fn bad(c: chan i64) -> i64 ! Boom
    c <- 1
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

fn bad(c: chan i64) -> i64
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
fn k0(c: chan i64) -> bool
    true
fn timed(c: chan i64) -> i64
    with deadline 50
        match <- c
            some(v) -> return v + 1
            none -> return 0
    else
        if true
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
                for i in 0..6
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
    emit run() catch |e| -1
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
                let q = <- s
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
    for i in 0..3
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
fn emitted_runtime_agrees_with_run_main() {
    if Command::new(rustc()).arg("--version").output().is_err() {
        eprintln!("SKIPPED: rustc not found (not a pass)");
        return;
    }
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
fn par_waits_stress() {
    if Command::new(rustc()).arg("--version").output().is_err() {
        eprintln!("SKIPPED: rustc not found (not a pass)");
        return;
    }
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
fn partitions_agree() {
    if Command::new(rustc()).arg("--version").output().is_err() {
        eprintln!("SKIPPED: rustc not found (not a pass)");
        return;
    }
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
fn python_runtime_agrees_with_run_main() {
    eprintln!("python oracle: {}", python());
    let Ok(o) = Command::new(python()).args(["-c", "import sys; print(sys.version, getattr(sys, '_is_gil_enabled', lambda: True)())"]).output() else {
        eprintln!("SKIPPED: {} not found (not a pass)", python());
        return;
    };
    eprintln!("python: {}", String::from_utf8_lossy(&o.stdout).trim());
    agree_lang(Lang::Python, 3);
}

#[test]
fn js_runtime_agrees_with_run_main() {
    let node = std::env::var("RCE_NODE").ok().filter(|v| !v.is_empty()).unwrap_or("node".into());
    let Ok(o) = Command::new(&node).arg("--version").output() else {
        eprintln!("SKIPPED: {node} not found (not a pass)");
        return;
    };
    eprintln!("node: {}", String::from_utf8_lossy(&o.stdout).trim());
    agree_lang(Lang::Js, 2);
}

/// handler refusals of every kind: a record failure, traps (overflow,
/// division by zero, index), a refused atomic transaction, and a supervisor
/// that gives up
const REFUSALS: &str = "event Hit(k: i64)
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
        if ["Refused(", "Stopped(", "Restart("].iter().any(|p| e.what.starts_with(p)) {
            v.push(format!("actorlog {}", e.what));
        }
    }
    v.push(match r.outcome {
        Outcome::Done => "outcome Done".into(),
        Outcome::Deadlock => "outcome Deadlock".into(),
        Outcome::Refused(e) => format!("outcome Refused: {e}"),
    });
    v
}

fn run_log(exe: &Exe) -> Vec<String> {
    let mut c = Command::new("timeout");
    c.arg("120");
    if !exe.0.is_empty() {
        c.arg(&exe.0);
    }
    let o = c.arg(&exe.1).args(["--run", "--actor-log"]).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout).unwrap().lines().map(String::from).collect()
}

#[test]
fn actor_refusals_match() {
    let d = rce_spec_check::check_program(&[("t.rce".into(), REFUSALS.to_string())], false);
    assert!(d[0].1.is_ok(), "not admitted: {:?}", d[0].1);
    let w = want_log(REFUSALS);
    eprintln!("{w:#?}");
    assert!(w.iter().filter(|l| l.starts_with("actorlog Refused")).count() >= 4, "{w:?}");
    let node = std::env::var("RCE_NODE").ok().filter(|v| !v.is_empty()).unwrap_or("node".into());
    let has_node = Command::new(&node).arg("--version").output().is_ok();
    for lang in [Lang::Rust, Lang::Python, Lang::Js] {
        if lang == Lang::Js && !has_node {
            eprintln!("SKIPPED: js (not a pass)");
            continue;
        }
        let exe = build_lang(lang, &format!("refusals_{lang:?}"), REFUSALS);
        assert_eq!(run_log(&exe), w, "{lang:?}");
    }
}

/// `--run` on a module without `fn main()` refuses as `rce run` does
/// (10.3): status 1 and the reference's message, in every target, rather
/// than starting a scheduler with no task
#[test]
fn run_without_main_refuses() {
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
            let o = c.arg(&exe.1).arg("--run").output().unwrap();
            assert_eq!(o.status.code(), Some(1), "{n} {lang:?}: {}", String::from_utf8_lossy(&o.stderr));
            assert!(o.stdout.is_empty(), "{n} {lang:?}");
            assert_eq!(String::from_utf8_lossy(&o.stderr).trim(), want, "{n} {lang:?}");
        }
    }
}
