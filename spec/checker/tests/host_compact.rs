//! HC-37: log compaction by an exact checkpoint. `rce host compact FILE D
//! N` replaces the first N entries of D by `host.Checkpoint`, which carries
//! the state at N in its positional form and every piece of host state the
//! replay of the entries after N reads, and is admitted only when the replay
//! of D' from it equals the full replay of D. These tests compact logs at
//! every N, logs with kept steps, typed inputs, windows, restarts, effects
//! in flight, a live `fn main` task, HTTP inputs and RAP invocations and
//! results among them, and require of every D' what the rule says: the same
//! states, outputs, kept steps, RAP results and HTTP inputs as D, byte for
//! byte, and, once a host opens it, the same entries appended as on D. A
//! forged checkpoint fails closed, and a torn tail is not compacted.
use rce_spec_check::ast::{Arg, Module};
use rce_spec_check::host::{self, Adapter, EffectResult, Entry, FileStore, Host, MemStore, Sourced, compact_text, http_inputs, prove_checkpoint, replay_outputs, replay_rap, replay_report};
use rce_spec_check::interp::{D_GENESIS, d_sha};
use rce_spec_check::parse::parse;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::Command;

const LEDGER: &str = "module compacted

struct Echo
    req: u64
    x: i64

event Add(x: i64)
event Ping(x: i64)
event EchoDone(req: u64, status: text, data: i64?, err: text)
event StepKept(seq: u64, cause: text)
event Restarted(below: u64)

error Neg

state S
    n: i64 = 0
    next: u64 = 0
    echoed: i64 = 0
    failed: u64 = 0
    kept: u64 = 0
    restarts: u64 = 0

step S on Add(x: x)
    if n + x < 0
        fail Neg
    n += x

step S on Ping(x: x)
    next += 1
    emit Echo(req: next, x: x)

step S on EchoDone(req: _r, status: s, data: d, err: _e)
    if s == \"delivered\"
        echoed += d.or(0)
    else
        failed += 1

step S on StepKept(seq: _q, cause: _c)
    kept += 1

step S on Restarted(below: _b)
    restarts += 1
";

const TASK: &str = "# rce-version: 0.6
module sched

state S
    n: i64 = 0

event Add(x: i64)

step S on Add(x: x)
    n += x

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

/// `Echo` performed in the process: delivered with x as its data.
struct EchoAdapter;
impl Adapter for EchoAdapter {
    fn kind(&self) -> &'static str {
        "Echo"
    }
    fn perform(&mut self, r: &Value) -> EffectResult {
        EffectResult::ok(json!(r["x"].as_i64().unwrap_or(0)))
    }
}

/// `HttpRespond` performed in the process: handed to the connection.
struct Responder;
impl Adapter for Responder {
    fn kind(&self) -> &'static str {
        "HttpRespond"
    }
    fn perform(&mut self, r: &Value) -> EffectResult {
        EffectResult::ok(json!(r["body"].as_str().unwrap_or("").len()))
    }
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rce-compact-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn args(src: &str, call: &str) -> (String, Vec<Arg>) {
    let probe = parse(&format!("{src}\nexample {call} -> 0\n")).unwrap();
    match probe.items.last() {
        Some(rce_spec_check::ast::Item::Example(x)) => x.events[0].clone(),
        _ => panic!("{call}"),
    }
}

fn adapters(kind: &str) -> Vec<Box<dyn Adapter>> {
    match kind {
        "echo" => vec![Box::new(EchoAdapter)],
        "http" => vec![Box::new(Responder)],
        _ => vec![],
    }
}

fn seqs(text: &str) -> (u64, u64) {
    let es: Vec<Entry> = text.lines().map(|l| Entry::parse_line(l).unwrap()).collect();
    (es[0].seq, es.last().unwrap().seq)
}

/// The refusals the rule allows: D owes entries after N, or N falls
/// between `host.Input` and the input it announces.
fn allowed_refusal(why: &str) -> bool {
    why.contains("D owes entries") || why.contains("announces an input")
}

/// Compact `text` at every N and check each D' against D. Returns the N
/// that were refused, with why.
fn every_n(m: &Module, text: &str, kind: &str, name: &str) -> Vec<(u64, String)> {
    let full = replay_report(m, text).unwrap();
    let outs = replay_outputs(m, text).unwrap();
    let rap = replay_rap(m, text).unwrap();
    let https = http_inputs(m, text).unwrap();
    let (lo, hi) = seqs(text);
    let low = if text.contains("\"type\":\"host.Checkpoint\"") { lo + 2 } else { 1 };
    let d = dir(name);
    let mut refused = vec![];
    for n in low..=hi + 1 {
        let c = match compact_text(m, text, n) {
            Ok(c) => c,
            Err(why) => {
                assert!(allowed_refusal(&why), "{name}: N {n} refused for a reason the rule does not give: {why}");
                refused.push((n, why));
                continue;
            }
        };
        prove_checkpoint(m, &c.text, &c.archive).unwrap_or_else(|e| panic!("{name}: N {n}: the archive does not prove the checkpoint: {e}"));
        assert_eq!(c.archive.lines().count() as u64 + c.kept as u64, text.lines().count() as u64);
        assert!(text.ends_with(c.text.split_once('\n').unwrap().1), "{name}: N {n}: D' keeps the entries after N byte for byte");
        let r = replay_report(m, &c.text).unwrap_or_else(|e| panic!("{name}: N {n}: D' does not replay: {e}"));
        assert_eq!((r.entries, &r.state_hash, format!("{:?}", r.state)), (full.entries, &full.state_hash, format!("{:?}", full.state)), "{name}: N {n}");
        let kept: Vec<_> = full.kept_outcomes.iter().filter(|k| k.0 >= n).cloned().collect();
        assert_eq!(r.kept_outcomes, kept, "{name}: N {n}: the kept steps after N and their texts");
        let o: Vec<_> = outs.iter().filter(|x| x.0 >= n).cloned().collect();
        assert_eq!(replay_outputs(m, &c.text).unwrap(), o, "{name}: N {n}: the outputs after N");
        assert_eq!(replay_rap(m, &c.text).unwrap(), rap, "{name}: N {n}: every RAP result, byte for byte");
        let hs = http_inputs(m, &c.text).unwrap();
        assert!(hs.iter().all(|x| https.contains(x)) && https.iter().filter(|x| x.id >= n).all(|x| hs.contains(x)), "{name}: N {n}: the HTTP inputs");
        // a host opened on D and one opened on D' append the same entries,
        // byte for byte: the restart below is the same seq (HC-22)
        let (a, b) = (d.join(format!("a{n}.d")), d.join(format!("b{n}.d")));
        std::fs::write(&a, text).unwrap();
        std::fs::write(&b, &c.text).unwrap();
        let ha = Host::open_file_with(m, &a, adapters(kind)).unwrap();
        let hb = Host::open_file_with(m, &b, adapters(kind)).unwrap();
        let same = ha.state_hash() == hb.state_hash();
        drop((ha, hb));
        let (ta, tb) = (std::fs::read_to_string(&a).unwrap(), std::fs::read_to_string(&b).unwrap());
        if !same || ta[text.len()..] != tb[c.text.len()..] {
            for (x, y) in ta[text.len()..].lines().zip(tb[c.text.len()..].lines()) {
                if x != y {
                    eprintln!("A {x}\nB {y}");
                }
            }
        }
        assert!(same, "{name}: N {n}: the states after the restart");
        assert_eq!(&ta[text.len()..], &tb[c.text.len()..], "{name}: N {n}: the entries a restart appends");
        // compacting D' again stands for more of D
        if n <= hi {
            let n2 = n + 1 + (hi - n) / 2;
            match compact_text(m, &c.text, n2) {
                Ok(c2) => {
                    prove_checkpoint(m, &c2.text, &c2.archive).unwrap();
                    assert_eq!(replay_report(m, &c2.text).unwrap().state_hash, full.state_hash, "{name}: N {n} then {n2}");
                    assert_eq!(replay_rap(m, &c2.text).unwrap(), rap);
                    // the same checkpoint as compacting D at n2 at once, except the archive it names
                    let once = compact_text(m, text, n2).unwrap();
                    let strip = |t: &str| -> Value {
                        let e = Entry::parse_line(t.lines().next().unwrap()).unwrap();
                        let mut v: Value = serde_json::from_str(&e.event).unwrap();
                        v["payload"]["archive"] = Value::Null;
                        v
                    };
                    assert_eq!(strip(&c2.text), strip(&once.text), "{name}: N {n} then {n2}: the checkpoint is the fold at n2 either way");
                }
                Err(why) => assert!(allowed_refusal(&why), "{name}: {why}"),
            }
        }
    }
    let _ = std::fs::remove_dir_all(&d);
    refused
}

/// A log with kept steps (their `host.StepKept` and the bound `StepKept`),
/// typed inputs, a window of sourced inputs, effects in flight across
/// entries and across a restart, placement entries and wall time.
fn ledger_log(d: &Path) -> (Module, String) {
    let m = parse(LEDGER).unwrap();
    let log = d.join("ledger.d");
    let (fs, _) = FileStore::open(&log).unwrap();
    let mut h = Host::start_with(&m, Box::new(fs), adapters("echo")).unwrap();
    let ev = |c: &str| args(LEDGER, c);
    for c in ["Add(x: 5)", "Add(x: -10)"] {
        let (n, a) = ev(c);
        h.admit_event(&n, a).unwrap();
    }
    // two pings and an add, with the echoes in flight until the settle
    for c in ["Ping(x: 3)", "Add(x: 1)", "Ping(x: 4)"] {
        let (n, a) = ev(c);
        h.apply_input(&n, a).unwrap();
    }
    h.settle().unwrap();
    h.wall_time(1_000_000_000).unwrap();
    // a typed StepKept and a typed result (HC-10b)
    h.admit(LEDGER, "StepKept(seq: 1, cause: \"fail\")").unwrap();
    h.admit(LEDGER, "EchoDone(req: 9, status: \"delivered\", data: some(1), err: \"\")").unwrap();
    // a window of sourced inputs, one of them kept (HC-15 to HC-17)
    for (q, c) in [(1, "Add(x: 2)"), (2, "Add(x: -50)"), (3, "Add(x: 7)")] {
        let (n, a) = ev(c);
        h.offer(Sourced::new("client", q, 1, &n, a)).unwrap();
    }
    h.close_window(1).unwrap();
    h.calibrate("cpu.threads", "sum.f64", 400.0, 6.0).unwrap();
    h.probe("cpu.threads", "sum.f64", "d1", "d1").unwrap();
    h.place_sum("s0", &(0..100_000).map(|_| 0.5f64).collect::<Vec<_>>()).unwrap();
    // a ping in flight when the process stops: the restart interrupts it
    let (n, a) = ev("Ping(x: 11)");
    h.apply_input(&n, a).unwrap();
    drop(h);
    let h = Host::open_file_with(&m, &log, adapters("echo")).unwrap();
    drop(h);
    let mut h = Host::open_file_with(&m, &log, adapters("echo")).unwrap();
    for c in ["Ping(x: 20)", "Add(x: 4)"] {
        let (n, a) = ev(c);
        h.admit_event(&n, a).unwrap();
    }
    let (n, a) = ev("Ping(x: 5)");
    h.apply_input(&n, a).unwrap();
    drop(h);
    (m, std::fs::read_to_string(&log).unwrap())
}

#[test]
fn every_compaction_of_a_ledger_replays_equal_to_the_original() {
    let d = dir("ledger");
    let (m, text) = ledger_log(&d);
    let refused = every_n(&m, &text, "echo", "ledger");
    let (_, hi) = seqs(&text);
    // the cuts between a kept entry and its host.StepKept, and between
    // host.StepKept and the bound StepKept, are refused; most are not
    assert!(!refused.is_empty() && refused.len() < (hi as usize) / 3, "{refused:?}");
    let es: Vec<Entry> = text.lines().map(|l| Entry::parse_line(l).unwrap()).collect();
    let k = es.iter().find(|e| e.kind() == "host.StepKept").unwrap().seq;
    assert!(refused.iter().any(|(n, w)| *n == k && w.contains("D owes entries")), "{refused:?}");
    let i = es.iter().find(|e| e.kind() == "host.Input").unwrap().seq;
    assert!(refused.iter().any(|(n, w)| *n == i + 1 && w.contains("announces an input")), "{refused:?}");
    // a checkpoint can stand while an effect is in flight, and carries it
    let q = es.iter().position(|e| e.event.contains("\"type\":\"Ping\"")).unwrap() as u64 + 1;
    let c = compact_text(&m, &text, q).unwrap();
    assert!(c.text.lines().next().unwrap().contains("\"queue\":[[\"Echo\",1,"), "{}", c.text.lines().next().unwrap());
    let _ = std::fs::remove_dir_all(&d);
}

/// A log of RAP invocations (HC-35, HC-36): conforming, not conforming,
/// duplicated, kept, and results settled and not, across a restart.
fn rap_log(d: &Path) -> (Module, String) {
    let src = std::fs::read_to_string(root().join("../../programs/money/account.rce")).unwrap();
    let m = rce_spec_check::program(&src).unwrap();
    let log = d.join("rap.d");
    let (fs, _) = FileStore::open(&log).unwrap();
    let mut h = Host::start_with(&m, Box::new(fs), vec![]).unwrap();
    let cb = "127.0.0.1:9";
    let granted = [cb.to_string()];
    let inv = |op: &str, cents: Value, id: &str, call: Value| json!({"operation": op, "arguments": {"cents": cents}, "id": id, "call_id": call, "callback_url": format!("http://{cb}/cb"), "group_id": "g"}).to_string().into_bytes();
    let invoke = |h: &mut Host, b: Vec<u8>| {
        h.rap_invoke(rce_spec_check::rap::envelope(&b, &granted).unwrap()).unwrap();
    };
    invoke(&mut h, inv("Deposit", json!(1000), "a", Value::Null));
    invoke(&mut h, inv("Withdraw", json!(5000), "b", json!("c-b")));
    h.rap_commit().unwrap();
    let (_, outs) = h.rap_steps().unwrap();
    h.settle().unwrap();
    h.rap_sent(outs[0].call, 200).unwrap();
    h.rap_commit().unwrap();
    // a duplicate of the kept one and of the first, an invalid one
    invoke(&mut h, inv("Withdraw", json!(5000), "b", json!("c-b2")));
    invoke(&mut h, inv("Deposit", json!("ten"), "c", Value::Null));
    invoke(&mut h, inv("Withdraw", json!(100), "d", Value::Null));
    h.rap_commit().unwrap();
    drop(h);
    // the host stopped after acknowledging, before stepping: a new one steps them
    let mut h = Host::open_file_with(&m, &log, vec![]).unwrap();
    let (_, outs) = h.rap_steps().unwrap();
    h.settle().unwrap();
    for o in h.rap_unsent() {
        if o.call != outs[0].call {
            h.rap_sent(o.call, 200).unwrap();
        }
    }
    h.rap_commit().unwrap();
    invoke(&mut h, inv("Deposit", json!(7), "e", Value::Null));
    invoke(&mut h, inv("Deposit", json!(1000), "a", json!("again")));
    h.rap_commit().unwrap();
    h.rap_steps().unwrap();
    h.settle().unwrap();
    drop(h);
    (m, std::fs::read_to_string(&log).unwrap())
}

#[test]
fn every_compaction_of_a_rap_log_keeps_every_result() {
    let d = dir("rap");
    let (m, text) = rap_log(&d);
    assert!(text.contains("host.RapDuplicate") && text.contains("\"cause\":\"fail\"") && text.contains("host.RapSent"), "the log has duplicates, a kept step and settled results");
    let refused = every_n(&m, &text, "", "rap");
    let (_, hi) = seqs(&text);
    assert!(refused.len() < hi as usize / 3, "{refused:?}");
    // after the last step: every call is carried, the kept one with the
    // state it was kept on and no text, and a duplicate of it after the
    // checkpoint is answered with the text derived again
    let c = compact_text(&m, &text, hi + 1).unwrap();
    let first = c.text.lines().next().unwrap();
    assert!(first.contains("\"kept\":{\"cause\":\"fail\"") && !first.contains("Insufficient"), "the kept text is not in D': {first}");
    let mut st = MemStore::default();
    let mut h = Host::resume(&m, &c.text, Box::new(&mut st)).unwrap();
    let granted = ["127.0.0.1:9".to_string()];
    let b = json!({"operation": "Withdraw", "arguments": {"cents": 5000}, "id": "b", "call_id": "late", "callback_url": "http://127.0.0.1:9/cb", "group_id": "g"}).to_string().into_bytes();
    h.rap_invoke(rce_spec_check::rap::envelope(&b, &granted).unwrap()).unwrap();
    let (_, outs) = h.rap_steps().unwrap();
    assert!(outs[0].body.contains("Insufficient") && outs[0].body.contains("\"call_id\":\"late\""), "{}", outs[0].body);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn every_compaction_of_an_http_log_keeps_the_open_requests() {
    let d = dir("http");
    let src = std::fs::read_to_string(root().join("../../programs/network/notes_http.rce")).unwrap();
    let m = rce_spec_check::program(&src).unwrap();
    let log = d.join("http.d");
    let (fs, _) = FileStore::open(&log).unwrap();
    let mut h = Host::start_with(&m, Box::new(fs), adapters("http")).unwrap();
    let req = |method: &str, target: &str, version: &str, body: &[u8]| {
        let mut p = serde_json::Map::new();
        p.insert("body".into(), json!(body));
        p.insert("method".into(), method.into());
        p.insert("names".into(), json!(["Host", "Connection"]));
        p.insert("target".into(), target.into());
        p.insert("values".into(), json!(["notes.test", "keep-alive"]));
        p.insert("version".into(), version.into());
        p
    };
    // two requests admitted before their responses are performed
    h.http_input("HttpRequest", req("POST", "/notes", "HTTP/1.1", b"milk")).unwrap();
    h.http_input("HttpRequest", req("HEAD", "/health", "HTTP/1.0", b"")).unwrap();
    h.settle().unwrap();
    h.http_input("HttpRequest", req("GET", "/notes", "HTTP/1.1", b"")).unwrap();
    h.http_input("HttpRequest", req("POST", "/notes", "HTTP/1.1", &[0xff])).unwrap();
    h.settle().unwrap();
    h.http_input("HttpRequest", req("DELETE", "/notes/1", "HTTP/1.1", b"")).unwrap();
    drop(h);
    let text = std::fs::read_to_string(&log).unwrap();
    let refused = every_n(&m, &text, "http", "http");
    assert!(refused.len() < text.lines().count() / 3, "{refused:?}");
    // between a request and its response the request is carried open
    let es: Vec<Entry> = text.lines().map(|l| Entry::parse_line(l).unwrap()).collect();
    let r = es.iter().find(|e| e.kind() == "HttpRequest").unwrap().seq;
    let c = compact_text(&m, &text, r + 2).unwrap();
    assert!(c.text.lines().next().unwrap().contains(&format!("\"http\":[[{r},false,false],")), "{}", c.text.lines().next().unwrap());
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn every_compaction_of_a_live_task_replays_equal() {
    let m = parse(TASK).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    h.main_begin(7, vec![], 100_000).unwrap();
    for i in 0..6 {
        h.main_step().unwrap();
        if i % 2 == 0 {
            let (n, a) = args(TASK, &format!("Add(x: {i})"));
            h.admit_event(&n, a).unwrap();
        }
    }
    while h.main_step().unwrap().is_some() {}
    h.main_end().unwrap();
    drop(h);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    let refused = every_n(&m, &text, "", "task");
    assert!(refused.is_empty(), "{refused:?}");
    // while the task is live the checkpoint carries it
    let es: Vec<Entry> = text.lines().map(|l| Entry::parse_line(l).unwrap()).collect();
    let k = es.iter().position(|e| e.kind() == "host.Step").unwrap() as u64 + 2;
    assert!(compact_text(&m, &text, k).unwrap().text.lines().next().unwrap().contains("\"task\":{\"digest\""));
}

/// The first line of D' rebuilt with `f` applied to its payload, its hash
/// recomputed (`rehash`) or left as it was.
fn forge(m: &Module, text: &str, f: impl Fn(&mut Value), state_hash: Option<String>, rehash: bool) -> String {
    let (first, rest) = text.split_once('\n').unwrap();
    let e = Entry::parse_line(first).unwrap();
    let mut v: Value = serde_json::from_str(&e.event).unwrap();
    f(&mut v["payload"]);
    let event = format!("{{\"payload\":{},\"type\":\"host.Checkpoint\"}}", rce_spec_check::interp::canon_json(&v["payload"]));
    let sh = state_hash.unwrap_or(e.state_hash.clone());
    let line = if rehash {
        Entry::new(m, e.seq, event, D_GENESIS, &sh).line()
    } else {
        Entry { event, state_hash: sh, ..e }.line()
    };
    format!("{line}\n{rest}")
}

/// A forged checkpoint fails closed: replay fails, and a host does not
/// open D, cuts nothing and appends nothing.
fn fails_closed(m: &Module, forged: &str, d: &Path, what: &str) {
    assert!(replay_report(m, forged).is_err(), "{what}: replay accepted a forged checkpoint");
    let p = d.join("forged.d");
    std::fs::write(&p, forged).unwrap();
    assert!(Host::open_file(m, &p).is_err(), "{what}: a host opened a forged checkpoint");
    assert_eq!(std::fs::read_to_string(&p).unwrap(), forged, "{what}: D changed");
}

#[test]
fn a_forged_checkpoint_fails_closed() {
    let d = dir("forged");
    let (m, text) = ledger_log(&d);
    let es: Vec<Entry> = text.lines().map(|l| Entry::parse_line(l).unwrap()).collect();
    // a checkpoint followed by a program entry, so a consistent forgery of
    // the state is found by the step after it
    let n = es.iter().position(|e| e.kind() == "Add").unwrap() as u64;
    let c = compact_text(&m, &text, n).unwrap();
    assert!(replay_report(&m, &c.text).is_ok());
    let bump = |p: &mut Value| p["state"] = Value::String(p["state"].as_str().unwrap().replacen("\"n\":0", "\"n\":1", 1));
    // the content changed and the hash not: HS-4
    fails_closed(&m, &forge(&m, &c.text, bump, None, false), &d, "state, hash kept");
    // the hash recomputed: the state does not hash to the state hash
    fails_closed(&m, &forge(&m, &c.text, bump, None, true), &d, "state, hash recomputed");
    // the state and its hash both forged: the entries after it disagree
    let st = forge(&m, &c.text, bump, None, true);
    let p: Value = serde_json::from_str(&Entry::parse_line(st.lines().next().unwrap()).unwrap().event).unwrap();
    // the positional form of the forged state: its fields in declaration order
    let named: Value = serde_json::from_str(p["payload"]["state"].as_str().unwrap()).unwrap();
    let pos: Vec<String> = ["n", "next", "echoed", "failed", "kept", "restarts"].iter().map(|f| named["S"][f].to_string()).collect();
    let sh = d_sha(&format!("[[{}]]", pos.join(",")));
    fails_closed(&m, &forge(&m, &c.text, bump, Some(sh), true), &d, "state and state hash");
    // a head the next entry does not name: D is not cut back to the checkpoint
    fails_closed(&m, &forge(&m, &c.text, |p| p["head"] = json!("ab".repeat(32)), None, true), &d, "head");
    // a carried value replay does not read back the same
    fails_closed(&m, &forge(&m, &c.text, |p| p["extra"] = json!(1), None, true), &d, "an extra field");
    fails_closed(&m, &forge(&m, &c.text, |p| p["term"] = json!("1"), None, true), &d, "a term that is not a count");
    let q = es.iter().position(|e| e.event.contains("\"type\":\"Ping\"")).unwrap() as u64 + 1;
    let cq = compact_text(&m, &text, q).unwrap();
    // an item the entries after N do not read back, here an effect in
    // flight dropped, is found by proving the checkpoint from its archive
    let dropped = forge(&m, &cq.text, |p| p["queue"] = json!([]), None, true);
    assert!(prove_checkpoint(&m, &dropped, &cq.archive).is_err(), "an effect in flight dropped");
    // a RAP result whose digest is not the one derived
    let (rm, rt) = rap_log(&d);
    let (_, hi) = seqs(&rt);
    let rc = compact_text(&rm, &rt, hi + 1).unwrap();
    fails_closed(&rm, &forge(&rm, &rc.text, |p| p["rap"][0]["answered"]["sha256"] = json!("00".repeat(32)), None, true), &d, "a RAP result digest");
    fails_closed(&rm, &forge(&rm, &rc.text, |p| p["rap"][1]["outcome"]["kept"]["cause"] = json!("fuel"), None, true), &d, "a kept cause");
    // a checkpoint anywhere but at the start of D
    let mut lines: Vec<String> = text.lines().map(String::from).collect();
    let mid = Entry::new(&m, 3, c.text.lines().next().map(|l| Entry::parse_line(l).unwrap().event).unwrap(), &Entry::parse_line(&lines[2]).unwrap().hash, &Entry::parse_line(&lines[2]).unwrap().state_hash);
    lines[3] = mid.line();
    assert!(replay_report(&m, &(lines[..4].join("\n") + "\n")).is_err(), "a checkpoint after the start of D");
    // an archive that is not the one the checkpoint names
    let short: String = c.archive.lines().take(c.archive.lines().count() - 1).map(|l| format!("{l}\n")).collect();
    assert!(prove_checkpoint(&m, &c.text, &short).is_err(), "an archive that is not the one named");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn compaction_of_a_torn_tail_is_refused() {
    let d = dir("torn");
    let (m, text) = ledger_log(&d);
    let torn = format!("{text}{{\"event\":{{\"payload\":{{\"torn");
    let why = compact_text(&m, &torn, 3).unwrap_err();
    assert!(why.contains("torn"), "{why}");
    // a whole last line without its line end is torn as well
    assert!(compact_text(&m, text.trim_end(), 3).unwrap_err().contains("torn"));
    // a corrupted entry in the middle
    let bad = text.replacen("\"x\":5", "\"x\":6", 1);
    assert!(compact_text(&m, &bad, 3).unwrap_err().contains("does not verify"));
    // the file is left as it was, and no archive is written
    let log = d.join("torn.d");
    std::fs::write(&log, &torn).unwrap();
    assert!(host::compact_file(&m, &log, 3).is_err());
    assert_eq!(std::fs::read_to_string(&log).unwrap(), torn);
    assert!(std::fs::read_dir(&d).unwrap().all(|e| !e.unwrap().file_name().to_string_lossy().starts_with("torn.d.archive")));
    let _ = std::fs::remove_dir_all(&d);
}

fn rce(a: &[&str]) -> (i32, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_rce")).args(a).output().unwrap();
    (o.status.code().unwrap_or(-1), String::from_utf8_lossy(&o.stdout).into_owned() + &String::from_utf8_lossy(&o.stderr))
}

/// `rce host compact FILE D N` writes D' and the archive, `rce host replay`
/// proves the checkpoint from the archive, a refusal is exit status 2 and
/// leaves D as it was, and `archive none` is refused (HS-1, HS-9, HC-20).
#[test]
fn the_command_compacts_proves_and_refuses() {
    let d = dir("cli");
    let file = d.join("ledger.rce");
    std::fs::write(&file, LEDGER).unwrap();
    let log = d.join("d.log");
    let (f, l) = (file.to_str().unwrap(), log.to_str().unwrap());
    let (code, out) = rce(&["host", "run", f, l, "Add(x: 5), Add(x: 2), Ping(x: 1), Add(x: 3), Add(x: 1)"]);
    assert_eq!(code, 0, "{out}");
    let text = std::fs::read_to_string(&log).unwrap();
    let (_, before) = rce(&["host", "replay", f, l]);
    let (code, out) = rce(&["host", "compact", "archive", "none", f, l, "4"]);
    assert_eq!(code, 2, "{out}");
    assert!(out.contains("HS-1") && out.contains("HC-20"), "{out}");
    assert_eq!(std::fs::read_to_string(&log).unwrap(), text);
    let (code, out) = rce(&["host", "compact", f, l, "0"]);
    assert_eq!(code, 2, "{out}");
    let (code, out) = rce(&["host", "compact", f, l, "4"]);
    assert_eq!(code, 0, "{out}");
    let c = std::fs::read_to_string(&log).unwrap();
    assert!(c.starts_with("{\"event\":{\"payload\":{\"admitted\""));
    let sha = d_sha(&text.lines().take(4).map(|x| format!("{x}\n")).collect::<String>());
    assert!(host::archive_path(&log, &sha).exists(), "{out}");
    let (code, after) = rce(&["host", "replay", f, l]);
    assert_eq!(code, 0, "{after}");
    assert!(after.contains("proven from archive"), "{after}");
    let tail = |t: &str| t.lines().next().map(|l| l.split(" state_hash ").nth(1).map(String::from)).unwrap_or_default();
    assert_eq!(tail(&before), tail(&after), "the same state hash: {before} / {after}");
    assert_eq!(before.lines().last(), after.lines().last(), "the same state");
    // a host continues on D' and replay still proves it
    let (code, out) = rce(&["host", "run", f, l, "Add(x: 1)"]);
    assert_eq!(code, 0, "{out}");
    let (code, out) = rce(&["host", "replay", f, l]);
    assert_eq!(code, 0, "{out}");
    // an archive that does not prove the checkpoint fails replay
    std::fs::write(host::archive_path(&log, &sha), text.lines().take(3).map(|x| format!("{x}\n")).collect::<String>()).unwrap();
    let (code, out) = rce(&["host", "replay", f, l]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("REPLAY FAILED"), "{out}");
    let _ = std::fs::remove_dir_all(&d);
}
