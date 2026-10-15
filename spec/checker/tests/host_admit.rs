//! E.2a canonical admission and E.2b admitter failover.
use rce_spec_check::ast::{Arg, Expr, Module};
use rce_spec_check::host::{Entry, Host, MemStore, Sourced, replay};
use rce_spec_check::parse::parse;
use rce_spec_check::repl::{Cluster, LeaderStore};

fn module() -> Module {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/host/priority.rce")).unwrap();
    parse(&src).unwrap()
}
fn key(src: &str, seq: u64, tick: u64, c: &str) -> Sourced {
    Sourced::new(src, seq, tick, "Key", vec![Arg { label: Some("c".into()), e: Expr::Str(c.into(), 0) }])
}
fn alarm(src: &str, seq: u64, tick: u64) -> Sourced {
    Sourced::new(src, seq, tick, "Alarm", vec![Arg { label: Some("n".into()), e: Expr::Int(seq as i128, 0) }])
}
/// Inputs of three sources over ticks 0..12, with retransmissions.
fn inputs() -> Vec<Sourced> {
    let mut v = vec![];
    for t in 0..12u64 {
        v.push(key("kbd", t + 1, t, &format!("{}", (b'a' + t as u8) as char)));
        if t % 3 == 0 {
            v.push(alarm("irq", t / 3 + 1, t));
        }
        if t % 2 == 1 {
            v.push(key("net", t / 2 + 1, t, &format!("{}", (b'A' + t as u8) as char)));
        }
    }
    v
}
/// A deterministic permutation (xorshift) standing in for arrival timing.
fn shuffle(v: &mut [Sourced], mut s: u64) {
    for i in (1..v.len()).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v.swap(i, (s % (i as u64 + 1)) as usize);
    }
}
fn run(m: &Module, arrivals: Vec<Sourced>, close_every: usize) -> (Vec<String>, String) {
    let mut st = MemStore::default();
    let mut h = Host::start(m, Box::new(&mut st)).unwrap();
    // Windows close at ticks 3, 7, 11. Arrival order and arrival batching
    // differ per host; an input arrives before the window holding its tick closes.
    let mut by_window: Vec<Vec<Sourced>> = vec![vec![], vec![], vec![]];
    for x in arrivals {
        let w = if x.tick <= 3 { 0 } else if x.tick <= 7 { 1 } else { 2 };
        by_window[w].push(x);
    }
    for (w, xs) in by_window.into_iter().enumerate() {
        for (i, x) in xs.into_iter().enumerate() {
            h.offer(x.clone()).unwrap();
            if i % close_every == 0 {
                h.offer(x).unwrap(); // a retransmission
            }
        }
        let a = h.close_window([3, 7, 11][w]).unwrap();
        assert!(a.held.is_empty() && a.rejected.is_empty(), "{a:?}");
    }
    let hash = h.state_hash().to_string();
    drop(h);
    (st.lines, hash)
}

#[test]
fn two_hosts_different_arrival_orders_give_byte_identical_d() {
    let m = module();
    let mut a = inputs();
    let mut b = inputs();
    shuffle(&mut a, 0x9e3779b97f4a7c15);
    shuffle(&mut b, 0x2545f4914f6cdd1d);
    b.reverse();
    let (da, ha) = run(&m, a, 2);
    let (db, hb) = run(&m, b, 3);
    assert_eq!(da, db, "D differs between hosts");
    assert_eq!(ha, hb);
    let text = da.join("\n") + "\n";
    let (_, rh, _) = replay(&m, &text).unwrap();
    assert_eq!(rh, ha);
    // Every (source, seq) is in D exactly once.
    let ins: Vec<String> = da.iter().map(|l| Entry::parse_line(l).unwrap()).filter(|e| e.kind() == "host.Input").map(|e| e.event).collect();
    let mut u = ins.clone();
    u.sort();
    u.dedup();
    assert_eq!(u.len(), ins.len());
    assert_eq!(ins.len(), inputs().len());
}

#[test]
fn window_order_is_priority_then_source_then_seq() {
    let m = module();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    for x in [key("z", 1, 0, "z"), key("a", 2, 0, "b"), key("a", 1, 0, "a"), alarm("q", 1, 0)] {
        h.offer(x).unwrap();
    }
    let a = h.close_window(0).unwrap();
    assert_eq!(a.acks, vec![("q".into(), 1), ("a".into(), 1), ("a".into(), 2), ("z".into(), 1)]);
    drop(h);
    let kinds: Vec<String> = st.lines.iter().map(|l| Entry::parse_line(l).unwrap().kind()).filter(|k| !k.starts_with("host.")).collect();
    assert_eq!(kinds, ["Alarm", "Key", "Key", "Key"]);
}

#[test]
fn gap_is_held_and_admitted_once_filled() {
    let m = module();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    h.offer(key("s", 2, 0, "b")).unwrap();
    let a = h.close_window(0).unwrap();
    assert_eq!(a.held, vec![("s".into(), 2)]);
    assert!(a.acks.is_empty());
    h.offer(key("s", 1, 1, "a")).unwrap();
    let a = h.close_window(1).unwrap();
    assert_eq!(a.acks, vec![("s".into(), 1), ("s".into(), 2)]);
    h.offer(key("s", 1, 2, "a")).unwrap();
    let a = h.close_window(2).unwrap();
    assert_eq!(a.dups, vec![("s".into(), 1)]);
    assert_eq!(h.admitted_upto("s"), 2);
}

#[test]
fn no_ack_without_a_durable_append() {
    let m = module();
    let mut st = MemStore { fail_barrier_at: Some(2), ..Default::default() };
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    h.offer(key("s", 1, 0, "a")).unwrap();
    assert!(h.close_window(0).is_err());
    assert!(h.stopped());
    h.offer(key("s", 2, 1, "b")).unwrap();
    assert!(h.close_window(1).is_err());
}

#[test]
fn dedupe_state_survives_replay() {
    let m = module();
    let mut st = MemStore::default();
    {
        let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
        h.offer(key("s", 1, 0, "a")).unwrap();
        h.close_window(0).unwrap();
    }
    let text = st.lines.join("\n") + "\n";
    let mut h = Host::resume(&m, &text, Box::new(MemStore::default())).unwrap();
    assert_eq!(h.admitted_upto("s"), 1);
    assert_eq!(h.next_window(), 1);
    h.offer(key("s", 1, 1, "a")).unwrap();
    let a = h.close_window(1).unwrap();
    assert_eq!(a.dups, vec![("s".into(), 1)]);
}

/// E.2b: leader A (term 1) admits two windows and is partitioned while
/// closing a third; B wins term 2 from a quorum, logs `host.Term`, sources
/// retransmit what A never acked, and D has one history with every input
/// exactly once. A's late barrier is fenced and A stops.
#[test]
fn admitter_failover_keeps_one_d_and_exactly_once() {
    let m = module();
    let c = Cluster::new(3);
    let log = c.borrow_mut().elect(1).unwrap();
    let mut a = Host::resume(&m, "", Box::new(LeaderStore::new(c.clone(), 1, log))).unwrap();
    a.take_role(1, "A").unwrap();
    let all = inputs();
    let mut acked: Vec<(String, u64)> = vec![];
    for (w, upto) in [3u64, 7].into_iter().enumerate() {
        for x in all.iter().filter(|x| x.tick <= upto && (w == 0 || x.tick > 3)) {
            a.offer(x.clone()).unwrap();
        }
        acked.extend(a.close_window(upto).unwrap().acks);
    }
    let committed_before = c.borrow().commit;
    // Partition: A can reach only replica 0. B is elected by replicas 1 and 2.
    c.borrow_mut().replicas[0].up = false;
    let log_b = c.borrow_mut().elect(2).unwrap();
    c.borrow_mut().replicas[0].up = true;
    c.borrow_mut().replicas[1].up = false;
    c.borrow_mut().replicas[2].up = false;
    for x in all.iter().filter(|x| x.tick > 7) {
        a.offer(x.clone()).unwrap();
    }
    assert!(a.close_window(11).is_err(), "deposed admitter must not commit");
    assert!(a.stopped());
    c.borrow_mut().replicas[1].up = true;
    c.borrow_mut().replicas[2].up = true;
    let text_b: String = log_b.iter().map(|l| format!("{l}\n")).collect();
    let mut b = Host::resume(&m, &text_b, Box::new(LeaderStore::new(c.clone(), 2, log_b.clone()))).unwrap();
    assert!(b.take_role(1, "B").is_err(), "term must grow");
    b.take_role(2, "B").unwrap();
    // Sources retransmit everything not acked, plus some acked inputs (lost acks).
    let mut retrans: Vec<Sourced> = all.iter().filter(|x| !acked.contains(&(x.source.clone(), x.seq))).cloned().collect();
    retrans.extend(all.iter().filter(|x| x.tick == 5).cloned());
    shuffle(&mut retrans, 7);
    for x in retrans {
        b.offer(x).unwrap();
    }
    let r = b.close_window(11).unwrap();
    assert!(!r.dups.is_empty());
    acked.extend(r.acks);
    let mut sorted = acked.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), acked.len(), "an input was acked twice");
    assert_eq!(acked.len(), all.len(), "an input was lost");
    let cl = c.borrow();
    let text = cl.committed_text();
    let lines: Vec<&str> = text.split_terminator('\n').collect();
    // No fork: what was committed under term 1 is a prefix of D.
    assert_eq!(&log_b[..committed_before], &lines[..committed_before]);
    let (_, rh, _) = replay(&m, &text).unwrap();
    assert_eq!(rh, b.state_hash());
    let terms: Vec<u64> = lines.iter().map(|l| Entry::parse_line(l).unwrap()).filter(|e| e.kind() == "host.Term").map(|e| serde_json::from_str::<serde_json::Value>(&e.event).unwrap()["payload"]["term"].as_u64().unwrap()).collect();
    assert_eq!(terms, [1, 2]);
    for r in cl.replicas.iter() {
        assert!(r.log.len() <= lines.len());
    }
}

/// HC-9 without source ids: the order is a function of the set of inputs.
#[test]
fn simultaneous_order_does_not_depend_on_caller_order() {
    let m = module();
    let ev = |n: &str, c: &str| (n.to_string(), vec![Arg { label: Some(if n == "Key" { "c" } else { "n" }.into()), e: if n == "Key" { Expr::Str(c.into(), 0) } else { Expr::Int(1, 0) } }]);
    let mk = || vec![ev("Key", "b"), ev("Alarm", ""), ev("Key", "a"), ev("Key", "c")];
    let mut logs = vec![];
    for rot in 0..4 {
        let mut st = MemStore::default();
        let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
        let mut v = mk();
        v.rotate_left(rot);
        if rot % 2 == 1 {
            v.reverse();
        }
        h.admit_simultaneous(v).unwrap();
        drop(h);
        logs.push(st.lines);
    }
    assert!(logs.windows(2).all(|w| w[0] == w[1]));
}

fn ticked() -> Module {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/host/ticked.rce")).unwrap();
    parse(&src).unwrap()
}
fn tick(seq: u64, n: u64) -> Sourced {
    Sourced::new("clock", seq, n, "Tick", vec![Arg { label: Some("n".into()), e: Expr::Int(n as i128, 0) }])
}

/// HC-15: a logged Tick closes the window; lateness is a fact of D.
#[test]
fn logged_tick_closes_the_window_and_decides_lateness() {
    let m = ticked();
    let run = |order: &[usize]| -> Vec<String> {
        let mut st = MemStore::default();
        let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
        let early = [key("a", 1, 0, "x"), key("b", 1, 1, "y"), key("a", 2, 1, "z")];
        for &i in order {
            assert!(h.offer(early[i].clone()).unwrap().is_none(), "no window closes without a Tick");
        }
        let w = h.offer(tick(1, 1)).unwrap().expect("Tick(1) closes window 0");
        assert_eq!(w.acks.last().unwrap(), &("clock".to_string(), 1));
        assert_eq!(h.closed_tick(), 1);
        // Stamped at tick 1 but offered after Tick(1) is in D: late.
        h.offer(key("b", 2, 1, "w")).unwrap();
        h.offer(key("a", 3, 2, "v")).unwrap();
        let w = h.offer(tick(2, 2)).unwrap().unwrap();
        assert_eq!(w.acks.len(), 3);
        // A retransmitted Tick does not close anything.
        assert!(h.offer(tick(2, 2)).unwrap().is_none());
        drop(h);
        st.lines
    };
    let a = run(&[0, 1, 2]);
    let b = run(&[2, 1, 0]);
    assert_eq!(a, b);
    let late: Vec<Entry> = a.iter().map(|l| Entry::parse_line(l).unwrap()).filter(|e| e.event.contains("\"late\":true")).collect();
    assert_eq!(late.len(), 1, "exactly b/2 is late");
    assert!(late[0].event.contains("\"source\":\"b\"") && late[0].event.contains("\"seq\":2"));
    let text = a.join("\n") + "\n";
    let h = Host::resume(&m, &text, Box::new(MemStore::default())).unwrap();
    assert_eq!(h.closed_tick(), 2);
    assert_eq!(h.next_window(), 2);
}
