//! HC-6: `fn main` runs as logged scheduler steps in D, replayed exactly.
//! HC-12: placement from D's host entries, executed on the fabrics this
//! host has, with the same bits on every fabric.
use rce_spec_check::host::{Entry, Host, MemStore, fixed_tree_sum, fixed_tree_sum_threads, replay};
use rce_spec_check::parse::parse;

const NONDET: &str = "# rce-version: 0.6
module sched

fn feed(c: chan i64, base: i64)
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

#[test]
fn main_run_is_logged_and_replays_exactly() {
    let m = parse(NONDET).unwrap();
    let mut seen = std::collections::BTreeSet::new();
    for seed in [1u64, 2, 3, 7, 11] {
        let mut st = MemStore::default();
        let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
        let r = h.run_main(seed, vec![], 100_000).unwrap();
        assert_eq!(r.outcome, "done");
        assert_eq!(r.outputs.len(), 10);
        assert!(r.steps > 0);
        seen.insert(r.outputs.clone());
        drop(h);
        let es: Vec<Entry> = st.lines.iter().map(|l| Entry::parse_line(l).unwrap()).collect();
        assert_eq!(es.iter().filter(|e| e.kind() == "host.Step").count(), r.steps);
        let text = st.lines.join("\n") + "\n";
        replay(&m, &text).expect("replay re-derives the logged run");
        // Replay reports a tampered step (the chain is re-linked so
        // only the content differs).
        let mut evs: Vec<String> = es.iter().map(|e| e.event.clone()).collect();
        let k = evs.iter().position(|e| e.contains("host.Step")).unwrap();
        evs[k] = evs[k].replace("\"tick\":", "\"tick\":9").replace("\"tick\":9", "\"tick\":99");
        let mut prev = rce_spec_check::interp::D_GENESIS.to_string();
        let mut lines = vec![];
        for (i, (ev, e)) in evs.into_iter().zip(&es).enumerate() {
            let ne = Entry::new(&m, i as u64, ev, &prev, &e.state_hash);
            prev = ne.hash.clone();
            lines.push(ne.line());
        }
        assert!(replay(&m, &(lines.join("\n") + "\n")).is_err(), "tampered step accepted");
    }
    assert!(seen.len() > 1, "different seeds should schedule differently");
}

#[test]
fn fixed_tree_bits_do_not_depend_on_the_fabric() {
    let xs: Vec<f64> = (0..1000).map(|i| 1.0 / (i as f64 + 1.0) * if i % 3 == 0 { -1e8 } else { 1.0 }).collect();
    let a = fixed_tree_sum(&xs);
    for d in 0..6 {
        assert_eq!(fixed_tree_sum_threads(&xs, d).to_bits(), a.to_bits());
    }
}

#[test]
fn placement_executor_follows_d_and_replays() {
    let m = parse("# rce-version: 0.6\nmodule p\nstate S\n    n: i64 = 0\n").unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    let small = vec![0.5f64; 8];
    let big: Vec<f64> = (0..100_000).map(|i| (i as f64).sqrt()).collect();
    // No calibration: everything stays on the CPU.
    assert_eq!(h.place_sum("s0", &big).unwrap().0, "cpu");
    h.calibrate("cpu.threads", "sum.f64", 400.0, 6.0).unwrap();
    // Calibrated but not probed exact: still the CPU (HC-13).
    assert_eq!(h.place_sum("s1", &big).unwrap().0, "cpu");
    assert!(h.probe("cpu.threads", "sum.f64", "d1", "d1").unwrap());
    let (f, r) = h.place_sum("s2", &big).unwrap();
    assert_eq!(f, "cpu.threads");
    assert_eq!(r.to_bits(), fixed_tree_sum(&big).to_bits());
    assert_eq!(h.place_sum("s3", &small).unwrap().0, "cpu", "small work stays on the CPU");
    // A faster fabric whose probe differs is never chosen.
    h.calibrate("gpu0", "sum.f64", 10.0, 1000.0).unwrap();
    assert!(!h.probe("gpu0", "sum.f64", "x", "y").unwrap());
    assert_eq!(h.place_sum("s4", &big).unwrap().0, "cpu.threads");
    drop(h);
    let text = st.lines.join("\n") + "\n";
    replay(&m, &text).expect("placements re-derive from host entries");
    // Change a logged placement: replay reports that it differs.
    let es: Vec<Entry> = st.lines.iter().map(|l| Entry::parse_line(l).unwrap()).collect();
    let mut prev = rce_spec_check::interp::D_GENESIS.to_string();
    let mut lines = vec![];
    for (i, e) in es.iter().enumerate() {
        let ev = if e.event.contains("\"op\":\"s2\"") { e.event.replace("cpu.threads", "cpu") } else { e.event.clone() };
        let ne = Entry::new(&m, i as u64, ev, &prev, &e.state_hash);
        prev = ne.hash.clone();
        lines.push(ne.line());
    }
    assert!(replay(&m, &(lines.join("\n") + "\n")).is_err());
}

const MIXED: &str = "# rce-version: 0.6
module mixed

event Key(c: text)

state S
    seen: text = \"\"

step S on Key(c: c)
    seen = seen + c

fn feed(c: chan i64, base: i64)
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

fn relink(m: &rce_spec_check::ast::Module, es: &[Entry], keep: impl Fn(usize) -> bool) -> String {
    let mut prev = rce_spec_check::interp::D_GENESIS.to_string();
    let mut lines = vec![];
    let mut i = 0u64;
    for (k, e) in es.iter().enumerate() {
        if !keep(k) {
            continue;
        }
        let ne = Entry::new(m, i, e.event.clone(), &prev, &e.state_hash);
        prev = ne.hash.clone();
        lines.push(ne.line());
        i += 1;
    }
    lines.join("\n") + "\n"
}

/// HC-6: the task advances one logged step at a time, program inputs
/// interleave with its steps, every entry's state_hash covers the task, and
/// a host rebuilt from any prefix of D continues to the same run.
#[test]
fn main_steps_interleave_with_inputs_and_state_is_the_fold_of_d() {
    let m = parse(MIXED).unwrap();
    let whole = {
        let mut st = MemStore::default();
        let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
        h.run_main(7, vec![], 100_000).unwrap()
    };
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    h.main_begin(7, vec![], 100_000).unwrap();
    let mut n = 0;
    let keys = ["a", "b", "c", "d"];
    while h.main_step().unwrap().is_some() {
        n += 1;
        if n % 5 == 0 && n / 5 <= keys.len() {
            h.admit(MIXED, &format!("Key(c: \"{}\")", keys[n / 5 - 1])).unwrap();
        }
    }
    assert!(h.snapshot().is_none(), "no snapshot while a task is live");
    let r = h.main_end().unwrap();
    assert_eq!(r.outputs, whole.outputs);
    assert_eq!(r.choices, whole.choices);
    drop(h);
    let es: Vec<Entry> = st.lines.iter().map(|l| Entry::parse_line(l).unwrap()).collect();
    let text = st.lines.join("\n") + "\n";
    replay(&m, &text).expect("interleaved run replays");
    // Program entries sit between steps.
    let first_step = es.iter().position(|e| e.kind() == "host.Step").unwrap();
    let last_step = es.iter().rposition(|e| e.kind() == "host.Step").unwrap();
    assert!(es[first_step..last_step].iter().filter(|e| !e.is_host()).count() >= 3);
    // Each step changes the folded state (the task is part of it).
    for w in es[first_step..=last_step].windows(2) {
        if w[1].kind() == "host.Step" {
            assert_ne!(w[0].state_hash, w[1].state_hash);
        }
    }
    // Rebuild from a prefix cut in the middle of the task, then finish it.
    let cut = (first_step + last_step) / 2;
    let prefix = st.lines[..cut].join("\n") + "\n";
    let mut st2 = MemStore::default();
    let mut h2 = Host::resume(&m, &prefix, Box::new(&mut st2)).unwrap();
    assert!(h2.main_steps().is_some());
    h2.take_role(1, "h2").unwrap();
    let r2 = h2.main_end().unwrap();
    assert_eq!(r2.outputs, whole.outputs);
    // Replay reports a dropped step.
    assert!(replay(&m, &relink(&m, &es, |k| k != first_step + 1)).is_err());
}

/// HC-6: an input for the live task is accepted only after the last logged
/// step; an earlier one would rewrite D's past and is not logged.
#[test]
fn main_input_after_the_last_step_only() {
    let m = parse(MIXED).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    h.main_begin(3, vec![], 100_000).unwrap();
    let mut last = 0;
    for _ in 0..6 {
        last = h.main_step().unwrap().unwrap()["tick"].as_u64().unwrap();
    }
    let before = h.len();
    assert!(h.main_input(last, true, "x", "y").is_err());
    assert_eq!(h.len(), before, "a late input is not logged");
    h.main_input(last + 1, true, "x", "y").unwrap();
    h.main_input(last + 2, false, "x", "y").unwrap();
    h.main_end().unwrap();
    drop(h);
    replay(&m, &(st.lines.join("\n") + "\n")).expect("run with inputs replays");
}

const ACTOR_FEED: &str = "# rce-version: 0.6
module feed

event Hit(k: i64)
event Show

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

/// HC-6: a program event fed to the live `fn main` task (`main_event`)
/// enters D as `host.MainInput` {event, tick}, reaches the actor that
/// handles it through the scheduler, and replays from D.
#[test]
fn program_events_feed_the_live_task_and_replay() {
    use rce_spec_check::ast::{Arg, Expr};
    let m = parse(ACTOR_FEED).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    h.main_begin(1, vec![], 100_000).unwrap();
    let k = |v: i128| vec![Arg { label: Some("k".into()), e: Expr::Int(v, 0) }];
    h.main_event(5, "Hit", k(3)).unwrap();
    h.main_event(6, "Hit", k(4)).unwrap();
    h.main_event(8, "Show", vec![]).unwrap();
    let r = h.main_end().unwrap();
    assert_eq!(r.outputs, vec!["0", "7"]);
    assert!(h.main_event(9, "Show", vec![]).is_err(), "no task is live after MainEnd");
    drop(h);
    assert!(st.lines.iter().any(|l| l.contains("host.MainInput") && l.contains("Hit")));
    let text = st.lines.join("\n") + "\n";
    replay(&m, &text).expect("the fed events re-derive from D");
}
