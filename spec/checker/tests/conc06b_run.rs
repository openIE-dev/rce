//! Run-time meaning of draft 0.6b (Appendix C, book/14-draft06-concurrency.md C.11) and
//! property tests for theorems T23 to T31. Each test names the theorem it
//! exercises; these are tests on generated or fixed inputs, not proofs.
use rce_spec_check::conc::{
    admission_order, edf_schedulable, rm_schedulable, run_main, Outcome, ParMode, Replicas,
    RunCfg, RunResult,
};
use rce_spec_check::parse::parse;

fn run_cfg(src: &str, cfg: &RunCfg) -> RunResult {
    let d = rce_spec_check::check_program(&[("t.rce".into(), src.to_string())], false);
    assert!(d[0].1.is_ok(), "not admitted: {:?}\n{src}", d[0].1);
    run_main(&parse(src).unwrap(), cfg).unwrap()
}
fn run(src: &str) -> RunResult {
    run_cfg(src, &RunCfg::default())
}
fn outs(r: &RunResult) -> Vec<String> {
    r.outputs.iter().map(|v| v.show()).collect()
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn int(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo) as u64) as i64
    }
}
fn modes(g: &mut Rng) -> Vec<RunCfg> {
    [ParMode::Seq, ParMode::Permute(g.next()), ParMode::Threads]
        .into_iter()
        .map(|p| RunCfg {
            par_mode: p,
            ..RunCfg::default()
        })
        .collect()
}

/// T23: a workgroup kernel gives the phase-then-lane result.
#[test]
fn t23_workgroup() {
    let mut g = Rng(23);
    for _ in 0..12 {
        let n = g.int(1, 17) as usize;
        let xs: Vec<i64> = (0..n).map(|_| g.int(-50, 50)).collect();
        let lit = format!("{xs:?}");
        let src = format!(
            "fn main()
    let x: [i64; {n}] = {lit}
    var total: i64 = 0
    with device d bound 4096
        with group g lanes {n}
            var tile: [i64; {n}] = [0; {n}]
            tile[g.lane] = x[g.lane] * 3
            barrier
            total += tile[{m} - g.lane] * (g.lane as i64 + 1)
    emit total
",
            m = n - 1
        );
        let want: i64 = (0..n).map(|l| xs[n - 1 - l] * 3 * (l as i64 + 1)).sum();
        for cfg in modes(&mut g) {
            assert_eq!(outs(&run_cfg(&src, &cfg)), vec![want.to_string()], "{src}");
        }
    }
}

/// C.11.2: pscan is the Blelloch exclusive scan; for + it equals the prefix sums.
#[test]
fn pscan_matches_prefix_sums() {
    let mut g = Rng(2);
    for _ in 0..20 {
        let n = g.int(1, 30) as usize;
        let xs: Vec<i64> = (0..n).map(|_| g.int(-99, 99)).collect();
        let src = format!(
            "fn main()\n    let xs = {xs:?}\n    emit xs.pscan(fn(a: i64, b: i64) => a + b, 0)\n    emit xs.pscan_incl(fn(a: i64, b: i64) => a + b, 0)\n"
        );
        let mut ex = vec![];
        let mut inc = vec![];
        let mut s = 0;
        for x in &xs {
            ex.push(s);
            s += x;
            inc.push(s);
        }
        assert_eq!(outs(&run(&src)), vec![format!("{ex:?}"), format!("{inc:?}")]);
    }
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

/// T24: for log entries a, b, a happened before b (program order on a
/// node plus send-to-delivery causes, transitively) iff V(a) < V(b).
#[test]
fn t24_vector_clocks() {
    let cfg = RunCfg {
        links: vec![(2, true, "main".into(), "west".into()), (6, false, "main".into(), "west".into())],
        ..RunCfg::default()
    };
    for r in [run(ACTORS), run_cfg(ACTORS, &cfg)] {
        assert_eq!(r.outcome, Outcome::Done);
        let n = r.log.len();
        assert!(n > 10);
        let mut hb = vec![vec![false; n]; n];
        for b in 0..n {
            for a in 0..b {
                if r.log[a].node == r.log[b].node {
                    hb[a][b] = true;
                }
            }
            if let Some(c) = r.log[b].cause {
                hb[c][b] = true;
            }
        }
        for k in 0..n {
            for i in 0..n {
                if hb[i][k] {
                    for j in 0..n {
                        if hb[k][j] {
                            hb[i][j] = true;
                        }
                    }
                }
            }
        }
        let lt = |a: &[u64], b: &[u64]| a.iter().zip(b).all(|(x, y)| x <= y) && a != b;
        for a in 0..n {
            for b in 0..n {
                assert_eq!(
                    hb[a][b],
                    lt(&r.log[a].vc, &r.log[b].vc),
                    "{:?} / {:?}",
                    r.log[a],
                    r.log[b]
                );
            }
        }
    }
}

/// C.11.4: a partition holds messages across the link and Heal delivers
/// them in order; nothing is dropped.
#[test]
fn partition_holds_then_heals() {
    let cfg = RunCfg {
        links: vec![(1, true, "main".into(), "west".into()), (9, false, "main".into(), "west".into())],
        ..RunCfg::default()
    };
    let r = run_cfg(ACTORS, &cfg);
    let base = run(ACTORS);
    let pick = |r: &RunResult, big: bool| -> Vec<String> {
        outs(r).into_iter().filter(|s| (s.len() >= 4) == big).collect()
    };
    assert!(r.log.iter().any(|e| e.what.starts_with("Held")));
    assert!(r.log.iter().any(|e| e.what.starts_with("Heal")));
    assert_eq!(pick(&r, true), pick(&base, true));
    assert_eq!(pick(&r, false), pick(&base, false));
}

/// T25: interrupts are admitted by (-priority, arrival) before ordinary
/// events; an interrupt of priority p waits only for pending interrupts of
/// priority >= p.
#[test]
fn t25_interrupt_admission() {
    let mut g = Rng(25);
    for _ in 0..200 {
        let n = g.int(0, 12) as usize;
        let pend: Vec<(usize, Option<u8>)> = (0..n)
            .map(|i| (i, if g.int(0, 2) == 0 { None } else { Some(g.int(0, 4) as u8) }))
            .collect();
        let ord = admission_order(&pend);
        assert_eq!(ord.len(), n);
        for (pos, &i) in ord.iter().enumerate() {
            if let Some(p) = pend[i].1 {
                let ahead = pend.iter().filter(|q| q.0 != i && q.1.is_some_and(|x| x > p || (x == p && q.0 < i))).count();
                assert_eq!(pos, ahead);
            } else {
                assert!(ord[pos..].iter().all(|&j| pend[j].1.is_none()));
            }
        }
    }
}

/// C.11.6: the Liu and Layland bounds.
#[test]
fn rt_schedulability() {
    assert!(rm_schedulable(&[(1.0, 4.0), (1.0, 5.0), (1.0, 10.0)]));
    assert!(!rm_schedulable(&[(2.0, 4.0), (2.0, 5.0)]));
    assert!(edf_schedulable(&[(2.0, 4.0), (2.0, 5.0)]));
    assert!(!edf_schedulable(&[(3.0, 4.0), (2.0, 5.0)]));
    // the RM bound tends to ln 2
    let many: Vec<(f64, f64)> = (0..50).map(|_| (0.69 / 50.0, 1.0)).collect();
    assert!(rm_schedulable(&many));
}

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

/// T26: when a scope completes no task it spawned is running, and an
/// unjoined failure is re-raised in the parent.
#[test]
fn t26_structured_spawn() {
    let r = run(SCOPE);
    let o = outs(&r);
    assert_eq!(o[0], "10");
    assert_eq!(o[1], "99");
    assert!(o[2].contains("TaskFailed"), "{o:?}");
    assert_eq!(o[3], "100");
    assert_eq!(o.len(), 4, "101 must not be emitted: {o:?}");
    assert!(matches!(&r.outcome, Outcome::Stopped(e) if e.contains("TaskFailed")));
    // every task spawned before an `emit 99` finished before it
    let emit99 = r.log.iter().position(|e| e.what == "Emit(99)").unwrap();
    let spawned: Vec<String> = r.log[..emit99]
        .iter()
        .filter_map(|e| e.what.strip_prefix("Spawn(task ").map(|s| s.split(',').next().unwrap().to_string()))
        .collect();
    assert_eq!(spawned.len(), 3);
    for id in spawned {
        let done = r.log.iter().position(|e| e.what == format!("Done(task {id})")).unwrap();
        assert!(done < emit99);
    }
}

/// T27: an actor handles its messages one per tick in mailbox order, so
/// its history is the sequential history of the sends.
#[test]
fn t27_linearizable_actor() {
    let r = run(ACTORS);
    let small: Vec<String> = outs(&r).into_iter().filter(|s| s.len() < 4).collect();
    assert_eq!(small, vec!["0", "1", "2", "3"]);
    let big: Vec<String> = outs(&r).into_iter().filter(|s| s.len() >= 4).collect();
    assert_eq!(big, vec!["1000", "1001", "1002", "1003"]);
}

/// T28: a well-typed session runs to the end without deadlock. (A bounded
/// `loop bound n` around the server's offer is refused with W-SES-1: it can
/// exit with the protocol unfinished.)
#[test]
fn t28_session_fidelity() {
    let src = "protocol Lookup
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
    let r = run(src);
    assert_eq!(r.outcome, Outcome::Done, "{:?}", outs(&r));
    assert_eq!(outs(&r), vec!["7", "1", "1", "1"]);
}

/// T29: lvar puts from par branches and for..par iterations give the same
/// frozen value, or the same conflict trap, in every schedule.
#[test]
fn t29_lvar_determinism() {
    let mut g = Rng(29);
    for _ in 0..10 {
        let n = g.int(1, 9);
        let vals: Vec<i64> = (0..n).map(|_| g.int(0, 100)).collect();
        let mut branches = String::new();
        for (i, v) in vals.iter().enumerate() {
            branches += &format!("        branch b{i}\n            x.put({v})\n");
        }
        let src = format!(
            "fn main()\n    let x = lvar_new(0)\n    par\n{branches}    emit x.freeze()\n    let s = lvar_new([0])\n    for i in 0..{n} par\n        s.put([i * 3 % 7])\n    emit s.freeze()\n"
        );
        let mx = vals.iter().max().unwrap();
        let mut set: Vec<i64> = (0..n).map(|i| i * 3 % 7).chain([0]).collect();
        set.sort();
        set.dedup();
        for cfg in modes(&mut g) {
            assert_eq!(outs(&run_cfg(&src, &cfg)), vec![mx.to_string(), format!("{set:?}")]);
        }
    }
    let conflict = "fn main()\n    let x = lvar_new(0, 10)\n    par\n        branch a\n            x.put(3)\n        branch b\n            x.put(10)\n    emit x.freeze()\n";
    let mut g = Rng(9);
    for cfg in modes(&mut g) {
        let r = run_cfg(conflict, &cfg);
        assert!(matches!(&r.outcome, Outcome::Stopped(e) if e.contains("lvar conflict")), "{:?}", r.outcome);
    }
}

/// T30: after the tick of an atomic block either every receiving actor
/// applied its message or none changed state.
#[test]
fn t30_atomic_across_actors() {
    let src = "event Add(k: i64)
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
    assert_eq!(outs(&run(src)), vec!["0", "0", "2", "3"]);
}

/// C.11.11: a crashing child restarts with its initial state; restart
/// n + 1 stops the supervisor.
#[test]
fn supervisor_restarts_then_stops() {
    let src = "event Hit(k: i64)
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
    let r = run(src);
    assert_eq!(outs(&r), vec!["5", "0"]);
    // Restart{child, instance, reason, height}: an entry of the run log
    assert!(r.log.iter().any(|e| e.what.starts_with("Restart(w, ") && e.what.ends_with(", 1)") && e.what.contains("tick ")), "{:?}", r.log);
    // the log, restarts included, is a function of the program: a second run
    // gives the same entries with the same vector clocks
    let r2 = run(src);
    let show = |l: &[rce_spec_check::conc::Entry]| l.iter().map(|e| format!("{} {} {:?}", e.tick, e.what, e.vc)).collect::<Vec<_>>();
    assert_eq!(show(&r.log), show(&r2.log));
    assert!(matches!(&r.outcome, Outcome::Stopped(e) if e.contains("supervisor Root stopped")));
}

/// T31: with partitions, every node's state is a fold of a prefix of one
/// common log, and minority submissions are held, never applied out of order.
#[test]
fn t31_replicated_safety() {
    let mut g = Rng(31);
    for _ in 0..200 {
        let n = g.int(1, 6) as usize;
        let mut r = Replicas::new(n);
        let mut history: Vec<Vec<i64>> = vec![];
        for _ in 0..30 {
            if g.int(0, 5) == 0 {
                let side: Vec<usize> = (0..n).map(|_| g.int(0, 2) as usize).collect();
                r.partition(side);
            } else {
                r.submit(g.int(0, n as i64) as usize, g.int(0, 1000));
            }
            // the log only grows, so every earlier log is a prefix of it
            for h in &history {
                assert_eq!(&r.log[..h.len()], &h[..]);
            }
            history.push(r.log.clone());
            for i in 0..n {
                assert!(r.applied[i] <= r.log.len());
                let s = r.state(i, |a, b| a.wrapping_mul(31).wrapping_add(b), 7);
                let want = r.log[..r.applied[i]].iter().fold(7i64, |a, b| a.wrapping_mul(31).wrapping_add(*b));
                assert_eq!(s, want);
            }
        }
        r.partition(vec![0; n]);
        assert!(r.held.is_empty());
        assert!(r.applied.iter().all(|&a| a == r.log.len()));
    }
}

/// T29 with threshold waits: a chain of waits (each branch waits for the
/// one after it) completes on every run under threads. Regression test for
/// the earlier stuck detection, which counted woken branches as blocked.
#[test]
fn t29_wait_chain_is_not_reported_stuck() {
    let src = "fn main()
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
";
    for _ in 0..200 {
        let r = run_cfg(src, &RunCfg { par_mode: ParMode::Threads, ..RunCfg::default() });
        assert_eq!(r.outcome, Outcome::Done, "{:?}", r.outcome);
        assert_eq!(outs(&r), vec!["33"]);
    }
}

/// C.11.11 restart as input: the supervision decisions of a run are inputs
/// (`RunResult.restarts`). Replaying them reproduces the run; the logged
/// decision governs even when the policy in the source has changed; a
/// decision is applied as logged (an empty reset keeps the actor's state);
/// a decision for another actor is a divergence.
#[test]
fn restart_is_an_input_replay_consumes() {
    let src = |bound: u32| format!("event Hit(k: i64)
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
    child w: Worker restart bound {bound}

fn main()
    let w = Worker()
    w <- Hit(k: 5)
    w <- Show
    w <- Hit(k: 0)
    w <- Show
    w <- Hit(k: 7)
    w <- Hit(k: 0)
    w <- Hit(k: 0)
    w <- Show
");
    let rec = run(&src(2));
    assert_eq!(rec.restarts.len(), 3);
    assert!(rec.restarts[2].stop && !rec.restarts[0].stop);
    let replay = |s: &str, rs: Vec<rce_spec_check::conc::RestartIn>| {
        run_cfg(s, &RunCfg { restarts: Some(rs), ..RunCfg::default() })
    };
    let show = |l: &[rce_spec_check::conc::Entry]| l.iter().map(|e| format!("{} {}", e.tick, e.what)).collect::<Vec<_>>();
    // replay reproduces the run
    let r = replay(&src(2), rec.restarts.clone());
    assert_eq!(show(&r.log), show(&rec.log));
    assert_eq!(r.outcome, rec.outcome);
    // the logged decisions govern under a changed policy (bound 9)
    let free = run(&src(9));
    assert_eq!(free.outcome, Outcome::Done, "the new policy alone would not stop");
    let r = replay(&src(9), rec.restarts.clone());
    assert!(matches!(&r.outcome, Outcome::Stopped(e) if e.contains("supervisor Root stopped")), "{:?}", r.outcome);
    // a decision is applied as logged: no reset keeps n
    let mut keep = rec.restarts.clone();
    keep[0].reset.clear();
    keep.truncate(1);
    let mut rs = keep.clone();
    let mut two = rec.restarts[1].clone();
    two.reset.clear();
    two.stop = false;
    rs.push(two.clone());
    let mut three = two.clone();
    three.tick = rec.restarts[2].tick;
    rs.push(three);
    let r = replay(&src(9), rs);
    assert_eq!(outs(&r), vec!["5", "5", "12"], "{:?}", r.log);
    // a decision for another actor diverges
    let mut bad = rec.restarts.clone();
    bad[0].id = "nobody".into();
    let r = replay(&src(2), bad);
    assert!(matches!(&r.outcome, Outcome::Stopped(e) if e == "replay diverged"), "{:?}", r.outcome);
}

/// HC-6: program events feed the running task. An event at tick t enters
/// the run log as `Input(..)` and reaches every live actor that handles it
/// in actor-id order; one nobody handles is logged `Unhandled(..)`. The log
/// is a function of the program and the timed inputs.
#[test]
fn program_events_feed_the_running_task() {
    let src = "event Hit(k: i64)
event Show
event Nobody

actor Acc
    n: i64 = 0
    on Hit(k: k)
        n += k
    on Show
        emit n

fn main()
    let a = Acc()
    a <- Show
";
    let m = parse(src).unwrap();
    let ev = |n: &str, k: Option<i64>| {
        rce_spec_check::interp::V::Rec(n.into(), k.map(|k| vec![("k".to_string(), rce_spec_check::interp::V::Int(k as i128, "i64"))]).unwrap_or_default())
    };
    let cfg = RunCfg {
        events: vec![(5, ev("Hit", Some(3))), (6, ev("Hit", Some(4))), (8, ev("Show", None)), (9, ev("Nobody", None))],
        ..RunCfg::default()
    };
    let r = run_main(&m, &cfg).unwrap();
    assert_eq!(outs(&r), vec!["0", "7"], "{:?}", r.log.iter().map(|e| &e.what).collect::<Vec<_>>());
    assert_eq!(r.outcome, Outcome::Done);
    assert!(r.log.iter().any(|e| e.what == "Input(Hit(k: 3))" && e.tick == 5), "{:?}", r.log.iter().map(|e| (&e.what, e.tick)).collect::<Vec<_>>());
    assert!(r.log.iter().any(|e| e.what.starts_with("Unhandled(Nobody")));
    let r2 = run_main(&m, &cfg).unwrap();
    let show = |l: &[rce_spec_check::conc::Entry]| l.iter().map(|e| format!("{} {} {:?}", e.tick, e.what, e.vc)).collect::<Vec<_>>();
    assert_eq!(show(&r.log), show(&r2.log));
}

/// C.11.11 migration rule of a restart log, on the schema alone: recorded
/// positions keep their supervisor and child, new ones only after them
#[test]
fn restart_log_schema_migration_rule() {
    use rce_spec_check::conc::tree_migrates;
    let v = |xs: &[&str]| xs.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let rec = v(&["A/x", "A/y", "B/z"]);
    assert!(tree_migrates(&rec, &rec).is_ok());
    assert!(tree_migrates(&rec, &v(&["A/x", "A/y", "A/new", "B/z", "B/w", "C/q"])).is_ok(), "appended children and supervisors");
    assert!(tree_migrates(&[], &rec).is_ok());
    assert_eq!(tree_migrates(&rec, &v(&["A/x", "B/z"])).unwrap_err(), "trap: upgrade removed supervisor position A/y");
    assert_eq!(tree_migrates(&rec, &v(&["A/y", "A/x", "B/z"])).unwrap_err(), "trap: upgrade reordered supervisor position A/x");
    assert_eq!(tree_migrates(&rec, &v(&["C/q", "A/x", "A/y", "B/z"])).unwrap_err(), "trap: upgrade reordered supervisor position A/x");
    assert_eq!(tree_migrates(&rec, &v(&["A/x", "A/new", "A/y", "B/z"])).unwrap_err(), "trap: upgrade reordered supervisor position A/y");
}

/// C.11.11 stable ids: an edit that keeps the program's structure (a
/// constant, a function and blank lines before everything, a renamed
/// local) keeps every actor and task id, so the run log is the same
#[test]
fn actor_and_task_ids_survive_structure_preserving_edits() {
    let src = "event Hit(k: i64)
error Boom
actor Acc
    n: i64 = 0
    on Hit(k: k)
        if k == 0
            fail Boom
        n += k
        emit n
supervisor S strategy one_for_one
    child a: Acc restart bound 2
fn work(c: chan i64)
    c <- 1
fn main()
    let a = Acc()
    let b = Acc()
    let c = chan[i64](1)
    spawn work(c)
    match <- c
        some(v) -> a <- Hit(k: v)
        none -> emit 0
    b <- Hit(k: 2)
    b <- Hit(k: 0)
";
    let edited = format!("const K: i64 = 3\n\nfn helper(x: i64) -> i64\n    x * K\n\n\n{}", src.replace("some(v) -> a <- Hit(k: v)", "some(got) -> a <- Hit(k: got)"));
    let ids = |s: &str| {
        let r = run(s);
        r.log.iter().map(|e| format!("{} {}", e.tick, e.what)).collect::<Vec<_>>()
    };
    let a = ids(src);
    assert!(a.iter().any(|l| l.contains("Restart(a, ")), "{a:?}");
    assert_eq!(a, ids(&edited));
}
