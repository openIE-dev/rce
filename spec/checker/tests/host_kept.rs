//! HC-21: a kept step is a fact of the run, so it is an input. The
//! admitter appends the kept step's entry with the state unchanged (T8), then
//! `host.StepKept` {cause, seq}, then, when the module declares
//! `event StepKept(seq: u64, cause: text)`, the input `StepKept` with the
//! same payload. Replay re-derives every kept step and its cause and reports
//! as a host fault a D whose entries for kept steps are missing, extra or
//! different.
use rce_spec_check::host::{Entry, FileStore, Host, MemStore, replay, replay_report};
use rce_spec_check::interp::{Cause, FUEL_LIMIT, Flow, V};
use rce_spec_check::parse::parse;
use std::path::Path;
use std::sync::atomic::Ordering;

/// Every test of this file runs with the same fuel, small enough that an
/// input that runs out of it does so quickly (as tests/fuel.rs does).
const FUEL: u64 = 2_000_000;

fn src() -> String {
    FUEL_LIMIT.store(FUEL, Ordering::Relaxed);
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/step_kept.rce")).unwrap()
}

fn field(h: &Host, name: &str) -> V {
    h.state().iter().flat_map(|(_, fs)| fs.iter()).find(|(n, _, _)| n == name).map(|(_, v, _)| v.clone()).unwrap()
}

fn int(v: &V) -> i64 {
    if let V::Int(i, _) = v { *i as i64 } else { panic!("{v:?}") }
}

fn text(st: &MemStore) -> String {
    st.lines.iter().map(|l| format!("{l}\n")).collect()
}

fn events(t: &str) -> Vec<String> {
    t.lines().map(|l| Entry::parse_line(l).unwrap().event).collect()
}

/// The same entries, with seq, prev_hash and hash recomputed: a log that
/// verifies (HS-4) but whose content was changed, so only replay can tell.
fn rechain(m: &rce_spec_check::ast::Module, es: Vec<Entry>) -> String {
    let mut prev = rce_spec_check::interp::D_GENESIS.to_string();
    let mut out = String::new();
    for (i, e) in es.into_iter().enumerate() {
        let n = Entry::new(m, i as u64, e.event, &prev, &e.state_hash);
        prev = n.hash.clone();
        out.push_str(&n.line());
        out.push('\n');
    }
    out
}

fn parsed(t: &str) -> Vec<Entry> {
    t.lines().map(|l| Entry::parse_line(l).unwrap()).collect()
}

#[test]
fn kept_step_is_followed_by_host_step_kept_and_the_bound_event() {
    let s = src();
    let m = parse(&s).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    let before = h.state_hash().to_string();
    let adm = h.admit(&s, "Fail").unwrap();
    // the input, then the bound StepKept it caused
    assert_eq!(adm.len(), 2, "{adm:?}");
    assert_eq!((adm[0].cause, adm[0].outcome.clone()), (Some("fail"), Err("Bad".to_string())));
    assert_eq!(adm[1].cause, None);
    let k = adm[0].seq;
    assert_eq!(adm[1].seq, k + 2);
    assert_eq!(int(&field(&h, "kept_outcomes")), 1);
    assert_eq!(int(&field(&h, "last_seq")), k as i64);
    assert_ne!(h.state_hash(), before, "the StepKept input changed the state");
    let live = h.state_hash().to_string();
    drop(h);
    let t = text(&st);
    let es = parsed(&t);
    assert!(es[0..k as usize].iter().any(|e| e.event.contains(r#""event":"StepKept","kind":"host.StepKept""#)), "host.Bind names StepKept");
    let fail = &es[k as usize];
    assert_eq!(fail.event, r#"{"payload":{},"type":"Fail"}"#);
    assert_eq!(fail.state_hash, es[k as usize - 1].state_hash, "T8: the kept step's entry leaves the state unchanged");
    assert_eq!(es[k as usize + 1].event, format!(r#"{{"payload":{{"cause":"fail","seq":{k}}},"type":"host.StepKept"}}"#));
    assert_eq!(es[k as usize + 2].event, format!(r#"{{"payload":{{"cause":"fail","seq":{k}}},"type":"StepKept"}}"#));
    assert!(!t.contains("Bad"), "no text of a kept step is in D (T7): the text names a declaration");
    let r = replay_report(&m, &t).unwrap();
    assert_eq!(r.state_hash, live);
    assert_eq!(r.kept_outcomes, vec![(k, "fail", "Bad".to_string())], "replay re-derives the kept step and its text");
}

#[test]
fn each_cause_is_classified() {
    let s = src();
    let m = parse(&s).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    let mut got = vec![];
    for ev in ["Fail", "Divide", "Burn", "Big", "Deep"] {
        let a = h.admit(&s, ev).unwrap();
        got.push(a[0].cause.unwrap_or("applied"));
    }
    assert_eq!(got, vec!["fail", "trap", "fuel", "size", "depth"]);
    assert_eq!(int(&field(&h, "kept_outcomes")), 5);
    assert_eq!(h.admit(&s, "Ok").unwrap()[0].cause, None);
    drop(h);
    let t = text(&st);
    for c in ["fail", "trap", "fuel", "size", "depth"] {
        assert!(t.contains(&format!(r#"{{"payload":{{"cause":"{c}","seq":"#)), "no host.StepKept with cause {c}");
    }
    let r = replay_report(&m, &t).unwrap();
    let causes: Vec<&str> = r.kept_outcomes.iter().map(|x| x.1).collect();
    assert_eq!(causes, vec!["fail", "trap", "fuel", "size", "depth"]);
    assert_eq!(r.kept_outcomes[1].2, "trap(division by zero)");
    // N9 VALUE_LIMIT and the remaining flows, by the trap texts of 6.3 and 6.7
    assert_eq!(Cause::of(&Flow::Trap("value limit".into())), Cause::Value);
    assert_eq!(Cause::of(&Flow::Trap("size limit".into())), Cause::Size);
    assert_eq!(Cause::of(&Flow::Trap("evaluation depth".into())), Cause::Depth);
    assert_eq!(Cause::of(&Flow::Trap(rce_spec_check::interp::TRAP_FUEL.into())), Cause::Fuel);
    assert_eq!(Cause::of(&Flow::Trap("loop bound".into())), Cause::Trap);
    assert_eq!(Cause::of(&Flow::Brk), Cause::Trap);
    for c in ["fail", "trap", "fuel", "size", "value", "depth"] {
        assert_eq!(Cause::parse(c).map(Cause::as_str), Some(c));
    }
}

#[test]
fn a_kept_step_kept_input_is_not_redelivered() {
    let s = src();
    let m = parse(&s).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    h.admit(&s, "Poison").unwrap();
    let a = h.admit(&s, "Fail").unwrap();
    // Fail kept; its StepKept kept in turn; nothing more
    assert_eq!(a.iter().map(|x| x.cause).collect::<Vec<_>>(), vec![Some("fail"), Some("trap")]);
    let after = h.admit(&s, "Ok").unwrap();
    assert_eq!(after[0].cause, None, "the next input is admitted");
    assert_eq!(int(&field(&h, "applied")), 1);
    assert_eq!(int(&field(&h, "kept_outcomes")), 0);
    let live = h.state_hash().to_string();
    drop(h);
    let t = text(&st);
    let ev = events(&t);
    assert_eq!(ev.iter().filter(|e| e.ends_with(r#""type":"StepKept"}"#)).count(), 1);
    assert_eq!(ev.iter().filter(|e| e.ends_with(r#""type":"host.StepKept"}"#)).count(), 2);
    let r = replay_report(&m, &t).unwrap();
    assert_eq!(r.state_hash, live);
    assert_eq!(r.kept_outcomes.len(), 2);
}

/// HC-7a: the bound StepKept is an ordinary input with the full fuel, even
/// when the input it reports ran out of fuel.
#[test]
fn step_kept_handler_has_full_fuel() {
    let s = src();
    let m = parse(&s).unwrap();
    // the largest power-of-two amount of work one input can do
    let mut h = Host::start(&m, Box::new(MemStore::default())).unwrap();
    let mut n = 1024u64;
    loop {
        let a = h.admit(&s, &format!("Spin(n: {})", n * 2)).unwrap();
        if a[0].cause.is_some() {
            assert_eq!(a[0].cause, Some("fuel"));
            break;
        }
        n *= 2;
    }
    assert!(n > 1024);
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    h.admit(&s, &format!("Busy(n: {n})")).unwrap();
    let a = h.admit(&s, "Burn").unwrap();
    assert_eq!(a[0].cause, Some("fuel"));
    assert_eq!(a[1].cause, None, "the StepKept handler did work that needs more than half of the fuel: {a:?}");
    assert_eq!(int(&field(&h, "work_done")), n as i64);
    let live = h.state_hash().to_string();
    drop(h);
    assert_eq!(replay(&m, &text(&st)).unwrap().1, live);
}

fn failed_log(m: &rce_spec_check::ast::Module, s: &str) -> (String, u64) {
    let mut st = MemStore::default();
    let mut h = Host::start(m, Box::new(&mut st)).unwrap();
    h.admit(s, "Ok").unwrap();
    let k = h.admit(s, "Fail").unwrap()[0].seq;
    h.admit(s, "Ok").unwrap();
    drop(h);
    (text(&st), k)
}

#[test]
fn replay_detects_a_log_without_host_step_kept() {
    let s = src();
    let m = parse(&s).unwrap();
    let (t, k) = failed_log(&m, &s);
    assert!(replay(&m, &t).is_ok());
    let mut es = parsed(&t);
    es.remove(k as usize + 1);
    let cut = rechain(&m, es);
    let e = replay(&m, &cut).unwrap_err();
    assert!(e.contains("host fault") && e.contains("host.StepKept"), "{e}");
    // without the bound StepKept
    let mut es = parsed(&t);
    es.remove(k as usize + 2);
    let e = replay(&m, &rechain(&m, es)).unwrap_err();
    assert!(e.contains("host fault"), "{e}");
}

#[test]
fn replay_detects_a_forged_host_step_kept() {
    let s = src();
    let m = parse(&s).unwrap();
    let (t, k) = failed_log(&m, &s);
    let es = parsed(&t);
    // a host.StepKept after an entry replay applies (the first Ok)
    let mut forged = es.clone();
    let ok = (k - 1) as usize;
    assert!(forged[ok].event.contains("\"type\":\"Ok\""));
    let mut fake = forged[ok].clone();
    fake.event = format!(r#"{{"payload":{{"cause":"trap","seq":{ok}}},"type":"host.StepKept"}}"#);
    forged.insert(ok + 1, fake);
    let e = replay(&m, &rechain(&m, forged)).unwrap_err();
    assert!(e.contains("host fault") && e.contains("follows an entry that replay applies"), "{e}");
    // the right place, the wrong cause
    let mut wrong = es.clone();
    wrong[k as usize + 1].event = format!(r#"{{"payload":{{"cause":"fuel","seq":{k}}},"type":"host.StepKept"}}"#);
    let e = replay(&m, &rechain(&m, wrong)).unwrap_err();
    assert!(e.contains("host fault"), "{e}");
    // a bound StepKept with a different payload
    let mut bound = es.clone();
    bound[k as usize + 2].event = format!(r#"{{"payload":{{"cause":"fail","seq":{}}},"type":"StepKept"}}"#, k + 1);
    let e = replay(&m, &rechain(&m, bound)).unwrap_err();
    assert!(e.contains("host fault"), "{e}");
    // a second StepKept after the delivered one, neither delivered nor typed
    let mut extra = es;
    let dup = extra[k as usize + 2].clone();
    extra.insert(k as usize + 3, dup);
    let e = replay(&m, &rechain(&m, extra)).unwrap_err();
    assert!(e.contains("host fault") && e.contains("neither that delivery nor typed"), "{e}");
}

/// HC-10b: `StepKept` typed on the input stream is admitted like any input.
/// D marks it with `host.Typed` {seq}, so replay tells it from the delivery
/// that follows `host.StepKept`. It starts no descent: when its own step is
/// kept, that kept step is delivered as a `StepKept`, which in turn is not
/// redelivered (HC-21). Replay gives the live state, the run validates,
/// and the same entry without its mark is a host fault.
#[test]
fn typed_step_kept_is_admitted_and_marked() {
    let s = src();
    let m = parse(&s).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    let a = h.admit(&s, r#"StepKept(seq: 7, cause: "fail")"#).unwrap();
    assert_eq!((a.len(), a[0].cause), (1, None), "{a:?}");
    let k = a[0].seq as usize;
    assert_eq!((int(&field(&h, "kept_outcomes")), int(&field(&h, "last_seq"))), (1, 7));
    h.admit(&s, "Poison").unwrap();
    let b = h.admit(&s, r#"StepKept(seq: 8, cause: "trap")"#).unwrap();
    assert_eq!(b.iter().map(|x| x.cause).collect::<Vec<_>>(), vec![Some("trap"), Some("trap")], "the typed input kept, then its delivered StepKept kept: {b:?}");
    let j = b[0].seq as usize;
    let live = h.state_hash().to_string();
    drop(h);
    let t = text(&st);
    let es = parsed(&t);
    assert_eq!(es[k - 1].event, format!(r#"{{"payload":{{"seq":{k}}},"type":"host.Typed"}}"#));
    assert_eq!(es[k].event, r#"{"payload":{"cause":"fail","seq":7},"type":"StepKept"}"#);
    let want = [
        format!(r#"{{"payload":{{"seq":{j}}},"type":"host.Typed"}}"#),
        r#"{"payload":{"cause":"trap","seq":8},"type":"StepKept"}"#.to_string(),
        format!(r#"{{"payload":{{"cause":"trap","seq":{j}}},"type":"host.StepKept"}}"#),
        format!(r#"{{"payload":{{"cause":"trap","seq":{j}}},"type":"StepKept"}}"#),
        format!(r#"{{"payload":{{"cause":"trap","seq":{}}},"type":"host.StepKept"}}"#, j + 2),
    ];
    assert_eq!(es[j - 1..j + 4].iter().map(|e| e.event.clone()).collect::<Vec<_>>(), want);
    let r = replay_report(&m, &t).unwrap();
    assert_eq!(r.state_hash, live);
    assert_eq!(r.kept_outcomes.len(), 2);
    // the same entry without its mark is not a fact of the run
    let mut bare = es;
    bare.remove(k - 1);
    let e = replay(&m, &rechain(&m, bare)).unwrap_err();
    assert!(e.contains("host fault") && e.contains("neither that delivery nor typed"), "{e}");
    // the host loop: a typed line naming StepKept is admitted and the run validates
    let dir = std::env::temp_dir().join(format!("rce-typed-kept-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let log = dir.join("d.log");
    let program = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/step_kept.rce");
    let mut s = serve(&program, &log);
    {
        use std::io::Write;
        s.stdin.write_all(b"StepKept(seq: 3, cause: \"fuel\")\nOk\n").unwrap();
        s.stdin.flush().unwrap();
    }
    let t0 = std::time::Instant::now();
    while !std::fs::read_to_string(&log).unwrap_or_default().contains("\"type\":\"Ok\"") {
        assert!(t0.elapsed() < std::time::Duration::from_secs(20), "the typed lines were not admitted");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let (ok, err) = s.stop();
    assert!(ok && err.contains("validated") && !err.contains("input not admitted"), "{err}");
    let t = std::fs::read_to_string(&log).unwrap();
    assert!(t.contains(r#""payload":{"cause":"fuel","seq":3},"type":"StepKept""#) && t.contains("\"type\":\"host.Typed\""), "{t}");
    replay(&m, &t).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

/// A host that stopped right after the kept step's entry (or after its
/// `host.StepKept`) left a D that verifies; opening it completes the
/// entries owed for the kept step before anything else is appended.
#[test]
fn open_completes_a_trailing_kept_step() {
    let s = src();
    let m = parse(&s).unwrap();
    let dir = std::env::temp_dir().join(format!("rce-kept-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for keep_host_step_kept in [false, true] {
        let log = dir.join(format!("d-{keep_host_step_kept}.log"));
        let _ = std::fs::remove_file(&log);
        let (fs, _) = FileStore::open(&log).unwrap();
        let mut h = Host::start(&m, Box::new(fs)).unwrap();
        let k = h.admit(&s, "Fail").unwrap()[0].seq;
        drop(h);
        // cut D after the kept step's entry, or after its host.StepKept
        let t = std::fs::read_to_string(&log).unwrap();
        let keep = k as usize + 1 + keep_host_step_kept as usize;
        let cut: String = t.lines().take(keep).map(|l| format!("{l}\n")).collect();
        std::fs::write(&log, &cut).unwrap();
        assert!(replay(&m, &cut).is_ok(), "a D that ends owing the entries of a kept step is not a fault");
        let h = Host::open_file(&m, &log).unwrap();
        assert_eq!(int(&field(&h, "kept_outcomes")), 1, "the bound StepKept was admitted on open");
        let live = h.state_hash().to_string();
        drop(h);
        let t = std::fs::read_to_string(&log).unwrap();
        let es = parsed(&t);
        assert_eq!(es[k as usize + 1].event, format!(r#"{{"payload":{{"cause":"fail","seq":{k}}},"type":"host.StepKept"}}"#));
        assert_eq!(es[k as usize + 2].event, format!(r#"{{"payload":{{"cause":"fail","seq":{k}}},"type":"StepKept"}}"#));
        assert!(es[k as usize + 3].event.contains("host.StorageCaps"), "completed before anything else: {}", es[k as usize + 3].event);
        assert_eq!(replay(&m, &t).unwrap().1, live);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// HD-4 (E.4c): a `StepKept` of another shape is matched by field name.
/// Declared as `StepKept(seq: u64)` it is delivered with the seq, and
/// `host.StepKept` before it already holds the cause. Declared with a field the host has no
/// value for, it cannot be built: the host starts, says so, and after each
/// `host.StepKept` records `host.Kept` with the reason, which replay
/// requires and a host that stopped between the two completes on open.
#[test]
fn a_step_kept_of_another_shape_is_matched_by_field_name() {
    let s = src().replace("event StepKept(seq: u64, cause: text)", "event StepKept(seq: u64)").replace("step S on StepKept(seq: q, cause: c)", "step S on StepKept(seq: q)").replace("    last_cause = copy(c)\n", "");
    let m = parse(&s).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    assert!(h.delivery_notes().is_empty(), "the cause is in host.StepKept: {:?}", h.delivery_notes());
    let a = h.admit(&s, "Fail").unwrap();
    let k = a[0].seq;
    assert_eq!((a.len(), int(&field(&h, "kept_outcomes")), int(&field(&h, "last_seq"))), (2, 1, k as i64));
    let live = h.state_hash().to_string();
    drop(h);
    let t = text(&st);
    let es = parsed(&t);
    assert_eq!(es[k as usize + 2].event, format!(r#"{{"payload":{{"seq":{k}}},"type":"StepKept"}}"#));
    assert_eq!(replay(&m, &t).unwrap().1, live);
    // a field the host has no value for
    let s = src().replace("event StepKept(seq: u64, cause: text)", "event StepKept(seq: u64, why: text)").replace("step S on StepKept(seq: q, cause: c)", "step S on StepKept(seq: q, why: c)");
    let m = parse(&s).unwrap();
    let dir = std::env::temp_dir().join(format!("rce-kept-shape-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let log = dir.join("d.log");
    let _ = std::fs::remove_file(&log);
    let (fs, _) = FileStore::open(&log).unwrap();
    let mut h = Host::start(&m, Box::new(fs)).unwrap();
    assert!(h.delivery_notes().iter().any(|n| n.starts_with("event StepKept:") && n.contains("field 2 (why) has no value from the host") && n.contains("never delivered")), "{:?}", h.delivery_notes());
    let a = h.admit(&s, "Fail").unwrap();
    assert_eq!((a.len(), int(&field(&h, "kept_outcomes"))), (1, 0), "{a:?}");
    let k = a[0].seq as usize;
    drop(h);
    let t = std::fs::read_to_string(&log).unwrap();
    let es = parsed(&t);
    let kept = format!(r#"{{"payload":{{"event":{{"payload":{{"cause":"fail","seq":{k}}},"type":"StepKept"}},"why":"field 2 has no value from the host"}},"type":"host.Kept"}}"#);
    assert_eq!(es[k + 2].event, kept);
    replay(&m, &t).unwrap();
    // stopped after host.StepKept: opening completes the host.Kept first
    let cut: String = t.lines().take(k + 2).map(|l| format!("{l}\n")).collect();
    std::fs::write(&log, &cut).unwrap();
    drop(Host::open_file(&m, &log).unwrap());
    let t = std::fs::read_to_string(&log).unwrap();
    assert_eq!(parsed(&t)[k + 2].event, kept);
    replay(&m, &t).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

fn refloop() -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/refloop.rce")).unwrap()
}

/// HC-21: a `StepKept` step that requests an effect whose result step is
/// kept in turn. The result descends from the `StepKept`, so its
/// kept step gets `host.StepKept` only, and the admission ends instead of
/// redelivering `StepKept` forever (the audit's `refloop.rce`).
#[test]
fn a_kept_step_whose_effect_is_kept_is_not_redelivered() {
    let s = refloop();
    let m = parse(&s).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start_with(&m, Box::new(&mut st), vec![Box::new(rce_spec_check::adapters::RandomAdapter)]).unwrap();
    let a = h.admit(&s, "Go").unwrap();
    // Go kept; its StepKept applied; the result of the StepKept step's
    // request kept, with no StepKept after it
    assert_eq!(a.iter().map(|x| (x.event.split('(').next().unwrap().to_string(), x.cause)).collect::<Vec<_>>(), vec![("Go".to_string(), Some("trap")), ("StepKept".to_string(), None), ("RandomDone".to_string(), Some("trap"))], "{a:?}");
    assert_eq!(int(&field(&h, "kept_outcomes")), 1);
    assert_eq!(int(&field(&h, "req")), 1);
    // the next input is admitted as before, and its kept step is delivered
    let b = h.admit(&s, "Go").unwrap();
    assert_eq!(b.iter().filter(|x| x.event.starts_with("StepKept")).count(), 1);
    let live = h.state_hash().to_string();
    drop(h);
    let t = text(&st);
    let ev = events(&t);
    assert_eq!(ev.iter().filter(|e| e.ends_with(r#""type":"StepKept"}"#)).count(), 2, "one StepKept per Go");
    assert_eq!(ev.iter().filter(|e| e.ends_with(r#""type":"host.StepKept"}"#)).count(), 4, "every kept step has its host.StepKept");
    // replay derives the same descent from D (with no adapter bound)
    let r = replay_report(&m, &t).unwrap();
    assert_eq!(r.state_hash, live);
    assert_eq!(r.kept_outcomes.len(), 4);
    // a result that does not descend from a StepKept still gets one
    let mut h = Host::start(&m, Box::new(MemStore::default())).unwrap();
    let a = h.admit(&s, "RandomDone(req: 7, status: \"failed\", data: none, err: \"x\")").unwrap();
    assert_eq!(a.iter().filter(|x| x.event.starts_with("StepKept")).count(), 1, "{a:?}");
}

/// The `rce host serve` child, its stdin and its stderr.
struct Serving {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    err: std::sync::Arc<std::sync::Mutex<String>>,
}

fn serve(program: &Path, log: &Path) -> Serving {
    use std::io::Read;
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_rce"))
        .args(["host", "serve"])
        .arg(program)
        .arg(log)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let stdin = child.stdin.take().unwrap();
    let err = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let (mut e, sink) = (child.stderr.take().unwrap(), err.clone());
    std::thread::spawn(move || {
        let mut b = [0u8; 4096];
        while let Ok(n) = e.read(&mut b) {
            if n == 0 {
                break;
            }
            sink.lock().unwrap().push_str(&String::from_utf8_lossy(&b[..n]));
        }
    });
    Serving { child, stdin, err }
}

impl Serving {
    /// SIGTERM, then the exit within 20 s (else the child is killed and
    /// the test fails): (success, stderr). stdin stays open, so only the
    /// stop can end the loop.
    fn stop(mut self) -> (bool, String) {
        use std::time::{Duration, Instant};
        // SAFETY: a signal to our own child process.
        unsafe { libc::kill(self.child.id() as i32, libc::SIGTERM) };
        let t = Instant::now();
        let st = loop {
            if let Some(st) = self.child.try_wait().unwrap() {
                break st;
            }
            if t.elapsed() > Duration::from_secs(20) {
                let _ = self.child.kill();
                panic!("the host loop did not stop within 20 s of SIGTERM: {}", self.err.lock().unwrap());
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        std::thread::sleep(Duration::from_millis(50));
        drop(self.stdin);
        (st.success(), self.err.lock().unwrap().clone())
    }
}

/// HC-21 and HC-24 on the host loop: the kept `StepKept` chain of
/// `refloop.rce` ends; an effect chain that never ends (each result
/// requests the next effect) is still stopped by SIGTERM, because a drain
/// performs at most DRAIN_MAX requests and the loop reads its stop between
/// drains; and so is the same chain continued by the restart's
/// `interrupted` result when the host opens that D again.
#[test]
fn serve_stops_while_effects_keep_coming() {
    use std::io::Write;
    use std::time::{Duration, Instant};
    let program = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/refloop.rce");
    let dir = std::env::temp_dir().join(format!("rce-refloop-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let m = parse(&refloop()).unwrap();
    // the kept StepKept chain ends on its own
    let log = dir.join("kept.log");
    let mut s = serve(&program, &log);
    s.stdin.write_all(b"Go\n").unwrap();
    s.stdin.flush().unwrap();
    let t = Instant::now();
    while std::fs::read_to_string(&log).unwrap_or_default().matches("\"type\":\"host.StepKept\"").count() < 2 {
        assert!(t.elapsed() < Duration::from_secs(20), "the kept step of RandomDone was not logged");
        std::thread::sleep(Duration::from_millis(20));
    }
    std::thread::sleep(Duration::from_millis(300));
    let (ok, err) = s.stop();
    assert!(ok && err.contains("validated") && err.contains("2 kept"), "{err}");
    let t = std::fs::read_to_string(&log).unwrap();
    let ev = events(&t);
    assert_eq!(ev.iter().filter(|e| e.ends_with(r#""type":"StepKept"}"#)).count(), 1, "no second StepKept");
    assert_eq!(ev.iter().filter(|e| e.ends_with(r#""type":"host.StepKept"}"#)).count(), 2);
    assert_eq!(ev.iter().filter(|e| e.ends_with(r#""type":"RandomDone"}"#)).count(), 1);
    replay(&m, &t).unwrap();
    // an effect chain that never ends, stopped
    let log = dir.join("endless.log");
    let mut s = serve(&program, &log);
    s.stdin.write_all(b"Endless\n").unwrap();
    s.stdin.flush().unwrap();
    let t = Instant::now();
    while std::fs::read_to_string(&log).unwrap_or_default().matches("\"type\":\"RandomDone\"").count() < 3 {
        assert!(t.elapsed() < Duration::from_secs(20), "the chain did not start");
        std::thread::sleep(Duration::from_millis(20));
    }
    let (ok, err) = s.stop();
    assert!(ok && err.contains("validated"), "{err}");
    let before = std::fs::read_to_string(&log).unwrap();
    replay(&m, &before).unwrap();
    // opened again: the outstanding request is interrupted, its result
    // requests the next one, and the stop still ends the loop
    let s = serve(&program, &log);
    let t = Instant::now();
    while std::fs::read_to_string(&log).unwrap().lines().count() < before.lines().count() + 8 {
        assert!(t.elapsed() < Duration::from_secs(20), "the chain did not continue after the restart");
        std::thread::sleep(Duration::from_millis(20));
    }
    let (ok, err) = s.stop();
    assert!(ok && err.contains("validated"), "{err}");
    let after = std::fs::read_to_string(&log).unwrap();
    assert!(after.contains("\"type\":\"host.Restarted\"") && after.contains(r#""err":"interrupted""#), "the restart interrupted the outstanding request");
    replay(&m, &after).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

/// HC-24: a line of the input stream that is not events of the module is
/// held, not turned away. With no `Input` declared it is recorded in D as
/// `host.Kept` and the loop goes on; with `Input(data: text)` declared it
/// is delivered as that input. Every run validates, and `rce host run`
/// holds such a line the same way.
#[test]
fn a_line_that_is_not_an_event_is_held() {
    use std::io::Write;
    let dir = std::env::temp_dir().join(format!("rce-held-line-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let program = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/step_kept.rce");
    let wait_for = |log: &Path, what: &str| {
        let t0 = std::time::Instant::now();
        while !std::fs::read_to_string(log).unwrap_or_default().contains(what) {
            assert!(t0.elapsed() < std::time::Duration::from_secs(20), "{what} was not logged");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    };
    let log = dir.join("held.log");
    let mut s = serve(&program, &log);
    s.stdin.write_all(b"hello there\nNope\nOk\n").unwrap();
    s.stdin.flush().unwrap();
    wait_for(&log, "\"type\":\"Ok\"");
    let (ok, err) = s.stop();
    assert!(ok && err.contains("validated") && !err.contains("input not admitted") && err.contains("line recorded in D"), "{err}");
    let t = std::fs::read_to_string(&log).unwrap();
    assert!(t.contains(r#"{"payload":{"event":{"payload":{"data":"hello there\n"},"type":"Input"},"why":"not declared"},"type":"host.Kept"}"#), "{t}");
    assert!(t.contains(r#"{"payload":{"event":{"payload":{"data":"Nope\n"},"type":"Input"},"why":"not declared"},"type":"host.Kept"}"#), "{t}");
    let m = parse(&src()).unwrap();
    replay(&m, &t).unwrap();
    // with Input declared, the line is that input
    let with_input = format!("{}\nevent Input(data: text)\n\nstep S on Input(data: _d)\n    applied += 10\n", src());
    let p = dir.join("with_input.rce");
    std::fs::write(&p, &with_input).unwrap();
    let log = dir.join("input.log");
    let mut s = serve(&p, &log);
    s.stdin.write_all(b"hello there\n").unwrap();
    s.stdin.flush().unwrap();
    wait_for(&log, "\"type\":\"Input\"");
    let (ok, err) = s.stop();
    assert!(ok && err.contains("validated"), "{err}");
    let t = std::fs::read_to_string(&log).unwrap();
    assert!(t.contains(r#"{"payload":{"data":"hello there\n"},"type":"Input"}"#) && !t.contains("host.Kept"), "{t}");
    // rce host run
    let log = dir.join("run.log");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_rce")).args(["host", "run"]).arg(&program).arg(&log).arg("Nope").output().unwrap();
    let (so, se) = (String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(so.contains("host.Kept: the line is not events of the module") && se.contains("validated"), "{so}{se}");
    assert_eq!(out.status.code(), Some(1), "the line was not applied");
    let _ = std::fs::remove_dir_all(&dir);
}

/// HC-24: a line is held whole. A line one of whose events cannot be built
/// (`Ok, Nope, Ok`) appends none of them: the whole line is recorded in D,
/// and the next line is admitted. A line that is not UTF-8 is held as its
/// bytes, never read as the end of the input. `rce host serve` and `rce
/// host run` hold them the same way, and `Input(data: List[i64])` takes
/// any line as its bytes (HC-10a). Every run validates.
#[test]
fn a_line_is_held_whole_and_as_bytes() {
    use std::io::Write;
    let dir = std::env::temp_dir().join(format!("rce-held-whole-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let program = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/step_kept.rce");
    let m = parse(&src()).unwrap();
    let input: &[u8] = b"Ok, Nope, Ok\n\xff\xfe\nOk\n";
    let mixed = r#"{"payload":{"event":{"payload":{"data":"Ok, Nope, Ok\n"},"type":"Input"},"why":"not declared"},"type":"host.Kept"}"#;
    let bytes = r#"{"payload":{"event":{"payload":{"data":[255,254,10]},"type":"Input"},"why":"not declared"},"type":"host.Kept"}"#;
    let check = |t: &str| {
        assert!(t.contains(mixed) && t.contains(bytes), "{t}");
        assert_eq!(t.matches(r#"{"payload":{},"type":"Ok"}"#).count(), 1, "only the last line's Ok is applied: {t}");
        assert!(t.find(bytes).unwrap() < t.find(r#""type":"Ok"}"#).unwrap(), "the line after the bytes is admitted after them");
        replay(&m, t).unwrap();
    };
    // rce host serve
    let log = dir.join("serve.log");
    let mut s = serve(&program, &log);
    s.stdin.write_all(input).unwrap();
    s.stdin.flush().unwrap();
    let t0 = std::time::Instant::now();
    while !std::fs::read_to_string(&log).unwrap_or_default().contains(r#""type":"Ok"}"#) {
        assert!(t0.elapsed() < std::time::Duration::from_secs(20), "the last line was not admitted");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let (ok, err) = s.stop();
    assert!(ok && err.contains("validated"), "{err}");
    check(&std::fs::read_to_string(&log).unwrap());
    // rce host run
    let log = dir.join("run.log");
    let mut c = std::process::Command::new(env!("CARGO_BIN_EXE_rce"))
        .args(["host", "run"])
        .arg(&program)
        .arg(&log)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    c.stdin.take().unwrap().write_all(input).unwrap();
    let out = c.wait_with_output().unwrap();
    let se = String::from_utf8_lossy(&out.stderr);
    assert!(se.contains("validated"), "{se}");
    assert_eq!(out.status.code(), Some(1), "two lines were not delivered");
    check(&std::fs::read_to_string(&log).unwrap());
    // Input(data: List[i64]): every line is delivered as its bytes
    let with_input = format!("{}\nevent Input(data: List[i64])\n\nstep S on Input(data: _d)\n    applied += 10\n", src());
    let p = dir.join("bytes_input.rce");
    std::fs::write(&p, &with_input).unwrap();
    let log = dir.join("bytes.log");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_rce")).args(["host", "run"]).arg(&p).arg(&log).arg("Nope").output().unwrap();
    assert!(String::from_utf8_lossy(&out.stderr).contains("validated") && out.status.code() == Some(0), "{out:?}");
    let t = std::fs::read_to_string(&log).unwrap();
    assert!(t.contains(r#"{"payload":{"data":[78,111,112,101,10]},"type":"Input"}"#) && !t.contains("host.Kept"), "{t}");
    let _ = std::fs::remove_dir_all(&dir);
}
