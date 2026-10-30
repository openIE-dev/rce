//! W1b: typed effect requests (HC-3, E-x). A request is an emitted struct
//! with `req: u64`; the admitter answers each one with exactly one result
//! event `<Kind>Done(req, status, data: T?, err)`, logged in D, so replay
//! needs no adapter (HC-8), and a request cut off by a crash is answered
//! `failed`/`interrupted` on the next open.
use rce_spec_check::host::{Adapter, EffectResult, Entry, Host, MemStore, replay};
use rce_spec_check::parse::parse;
use serde_json::{Value, json};
use std::path::Path;

struct Echo {
    calls: std::rc::Rc<std::cell::Cell<u32>>,
}
impl Adapter for Echo {
    fn kind(&self) -> &'static str {
        "Echo"
    }
    fn perform(&mut self, r: &Value) -> EffectResult {
        self.calls.set(self.calls.get() + 1);
        if r["msg"] == "boom" { EffectResult::failed("not performed by adapter") } else { EffectResult::ok(r["msg"].clone()) }
    }
}

fn src() -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/echo.rce")).unwrap()
}
fn kinds(lines: &[String]) -> Vec<String> {
    lines.iter().map(|l| Entry::parse_line(l).unwrap().kind()).collect()
}

#[test]
fn each_request_gets_exactly_one_result_and_replays_without_adapters() {
    let s = src();
    let m = parse(&s).unwrap();
    let calls = std::rc::Rc::new(std::cell::Cell::new(0));
    let mut st = MemStore::default();
    let mut h = Host::start_with(&m, Box::new(&mut st), vec![Box::new(Echo { calls: calls.clone() })]).unwrap();
    let out = h.admit(&s, r#"Say(msg: "a"), Say(msg: "boom"), Say(msg: "c")"#).unwrap();
    assert_eq!(out.len(), 6, "{out:?}");
    assert!(out.iter().all(|a| a.outcome.is_ok()), "{out:?}");
    let live = h.state_hash().to_string();
    let n = h.len();
    drop(h);
    assert_eq!(calls.get(), 3);
    let k = kinds(&st.lines);
    assert!(k.contains(&"host.Bind".to_string()));
    // request, then its one result, in emit order
    let prog: Vec<&String> = k.iter().filter(|x| !x.starts_with("host.")).collect();
    assert_eq!(prog, ["Say", "EchoDone", "Say", "EchoDone", "Say", "EchoDone"]);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert!(text.contains(r#""data":null,"err":"not performed by adapter","req":2,"status":"failed""#), "{text}");
    let (k2, h2, state) = replay(&m, &text).unwrap();
    assert_eq!((k2, h2), (n, live));
    let shown = format!("{state:?}");
    assert!(shown.contains("\"a\"") && shown.contains("\"c\""), "{shown}");
}

#[test]
fn interrupted_request_is_answered_failed_on_open() {
    let s = src();
    let m = parse(&s).unwrap();
    let p = std::env::temp_dir().join(format!("rce-fx-{}.log", std::process::id()));
    let _ = std::fs::remove_file(&p);
    let calls = std::rc::Rc::new(std::cell::Cell::new(0));
    {
        let mut h = Host::open_file_with(&m, &p, vec![Box::new(Echo { calls: calls.clone() })]).unwrap();
        h.admit(&s, r#"Say(msg: "a")"#).unwrap();
    }
    // crash after the request's entry and before its result: cut D there
    let text = std::fs::read_to_string(&p).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    let say = lines.iter().position(|l| l.contains("\"type\":\"Say\"")).unwrap();
    std::fs::write(&p, lines[..=say].iter().map(|l| format!("{l}\n")).collect::<String>()).unwrap();
    let before = calls.get();
    {
        let h = Host::open_file_with(&m, &p, vec![Box::new(Echo { calls: calls.clone() })]).unwrap();
        drop(h);
    }
    assert_eq!(calls.get(), before, "the effect is not re-performed after a crash");
    let text = std::fs::read_to_string(&p).unwrap();
    let done: Vec<&str> = text.lines().filter(|l| l.contains("\"type\":\"EchoDone\"")).collect();
    assert_eq!(done.len(), 1, "{text}");
    assert!(done[0].contains("\"err\":\"interrupted\"") && done[0].contains("\"status\":\"failed\""));
    // reopening again answers nothing twice
    drop(Host::open_file_with(&m, &p, vec![Box::new(Echo { calls: calls.clone() })]).unwrap());
    let text = std::fs::read_to_string(&p).unwrap();
    assert_eq!(text.lines().filter(|l| l.contains("\"type\":\"EchoDone\"")).count(), 1);
    replay(&m, &text).unwrap();
    let _ = std::fs::remove_file(&p);
}

/// HD-1 (E.4c): a result event declared with fewer fields is matched by
/// field name. The host starts, says so in one line, performs the effect and
/// delivers the fields the module declares; `host.Kept` keeps in D the
/// fields the event does not take, and replay plans the same.
#[test]
fn result_event_of_another_shape_is_matched_by_field_name() {
    let s = src()
        .replace("event EchoDone(req: u64, status: text, data: text?, err: text)", "event EchoDone(req: u64, status: text)")
        .replace("step S on EchoDone(req: _r, status: st, data: d, err: _e)", "step S on EchoDone(req: _r, status: st)")
        .replace("        got.push(d.or(\"\"))\n", "        got.push(\"\")\n")
        .replace(r#"EchoDone(req: 1, status: "failed", data: none, err: "x")"#, r#"EchoDone(req: 1, status: "failed")"#);
    let m = parse(&s).unwrap();
    let calls = std::rc::Rc::new(std::cell::Cell::new(0));
    let mut st = MemStore::default();
    let mut h = Host::start_with(&m, Box::new(&mut st), vec![Box::new(Echo { calls: calls.clone() })]).unwrap();
    assert!(h.delivery_notes().iter().any(|n| n.starts_with("event EchoDone:") && n.contains("data, err are not delivered")), "{:?}", h.delivery_notes());
    let out = h.admit(&s, r#"Say(msg: "a"), Say(msg: "boom")"#).unwrap();
    assert_eq!((calls.get(), out.len()), (2, 4), "{out:?}");
    assert!(out.iter().all(|a| a.outcome.is_ok()), "{out:?}");
    let live = h.state_hash().to_string();
    drop(h);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert!(text.contains(r#"{"payload":{"event":{"payload":{"data":"a","err":"","req":1,"status":"delivered"},"type":"EchoDone"},"why":"not declared: data, err"},"type":"host.Kept"}"#), "{text}");
    assert!(text.contains(r#"{"payload":{"req":1,"status":"delivered"},"type":"EchoDone"}"#), "{text}");
    assert!(text.contains(r#""payload":{"data":null,"err":"not performed by adapter","req":2,"status":"failed"},"type":"EchoDone"},"why":"not declared: err""#), "{text}");
    assert_eq!(replay(&m, &text).unwrap().1, live);
    // replay plans each delivery again: a host.Kept with another reason is
    // a fault
    let es: Vec<Entry> = st.lines.iter().map(|l| Entry::parse_line(l).unwrap()).collect();
    let mut prev = rce_spec_check::interp::D_GENESIS.to_string();
    let mut forged = String::new();
    for e in es {
        let n = Entry::new(&m, e.seq, e.event.replace("not declared: data, err", "not declared: data"), &prev, &e.state_hash);
        prev = n.hash.clone();
        forged.push_str(&n.line());
        forged.push('\n');
    }
    let e = replay(&m, &forged).unwrap_err();
    assert!(e.contains("host fault") && e.contains("host.Kept"), "{e}");
}

/// HD-1 (E.4c): a request struct whose result event the module does not
/// declare is still performed (fire and forget). Its result is recorded in
/// D as `host.Kept` and not delivered; it answers the request, so opening
/// D again interrupts nothing.
#[test]
fn bound_effect_without_its_result_event_is_performed_and_recorded() {
    let s = src();
    let s = s[..s.find("step S on EchoDone").unwrap()]
        .replace("event EchoDone(req: u64, status: text, data: text?, err: text)\n", "")
        .replace(r#"    example Say(msg: "hi"), EchoDone(req: 1, status: "failed", data: none, err: "x") -> failed == 1"#, r#"    example Say(msg: "hi") -> next == 1"#);
    let m = parse(&s).unwrap();
    let p = std::env::temp_dir().join(format!("rce-fx-ff-{}.log", std::process::id()));
    let _ = std::fs::remove_file(&p);
    let calls = std::rc::Rc::new(std::cell::Cell::new(0));
    {
        let mut h = Host::open_file_with(&m, &p, vec![Box::new(Echo { calls: calls.clone() })]).unwrap();
        assert!(h.delivery_notes().iter().any(|n| n.contains("struct Echo: event EchoDone is not declared")), "{:?}", h.delivery_notes());
        let out = h.admit(&s, r#"Say(msg: "a")"#).unwrap();
        assert_eq!(out.len(), 1, "only the Say step: {out:?}");
    }
    assert_eq!(calls.get(), 1, "the effect is performed");
    drop(Host::open_file_with(&m, &p, vec![Box::new(Echo { calls: calls.clone() })]).unwrap());
    assert_eq!(calls.get(), 1);
    let text = std::fs::read_to_string(&p).unwrap();
    let kept: Vec<&str> = text.lines().filter(|l| l.contains("\"type\":\"host.Kept\"")).collect();
    assert_eq!(kept.len(), 1, "one record, and no interruption on open: {text}");
    assert!(kept[0].contains(r#"{"event":{"payload":{"data":"a","err":"","req":1,"status":"delivered"},"type":"EchoDone"},"why":"not declared"}"#), "{}", kept[0]);
    assert!(text.lines().all(|l| Entry::parse_line(l).unwrap().kind() != "EchoDone"), "nothing is delivered");
    replay(&m, &text).unwrap();
    let _ = std::fs::remove_file(&p);
    let _ = json!(null);
}

/// HS-2: the request's entry is durable (a barrier ran) before the adapter
/// performs the effect.
#[test]
fn request_is_durable_before_its_effect() {
    struct Probe<'s> {
        barriers_at_call: &'s std::cell::Cell<usize>,
        store: *const MemStore,
    }
    impl Adapter for Probe<'_> {
        fn kind(&self) -> &'static str {
            "Echo"
        }
        fn perform(&mut self, r: &Value) -> EffectResult {
            // SAFETY: the store outlives the host in this test and is only read.
            self.barriers_at_call.set(unsafe { (*self.store).barriers });
            EffectResult::ok(r["msg"].clone())
        }
    }
    let s = src();
    let m = parse(&s).unwrap();
    let mut st = MemStore::default();
    let at = std::cell::Cell::new(0);
    let ptr: *const MemStore = &st;
    let mut h = Host::start_with(&m, Box::new(&mut st), vec![Box::new(Probe { barriers_at_call: &at, store: ptr })]).unwrap();
    h.admit(&s, r#"Say(msg: "a")"#).unwrap();
    drop(h);
    assert_eq!(at.get(), 2, "start barrier, then the Say barrier, before the effect");
}

/// HD-1 with HC-3: a result whose `req` is not delivered as an integer
/// (`req: u64?`) is recorded by `host.Kept` before it, so replay matches
/// it to its request by the req that record holds. Opening D again answers
/// no request twice, and a D cut between the record and the delivery is
/// completed on open with the one result, not interrupted as well.
#[test]
fn a_result_whose_req_is_not_an_integer_answers_its_request_once() {
    let s = src()
        .replace("event EchoDone(req: u64, status: text, data: text?, err: text)", "event EchoDone(req: u64?, status: text, data: text?, err: text)")
        .replace(r#"EchoDone(req: 1, status: "failed", data: none, err: "x")"#, r#"EchoDone(req: some(1), status: "failed", data: none, err: "x")"#);
    let m = parse(&s).unwrap();
    let p = std::env::temp_dir().join(format!("rce-fx-reqopt-{}.log", std::process::id()));
    let _ = std::fs::remove_file(&p);
    let calls = std::rc::Rc::new(std::cell::Cell::new(0));
    let open = |p: &Path| drop(Host::open_file_with(&m, p, vec![Box::new(Echo { calls: calls.clone() })]).unwrap());
    {
        let mut h = Host::open_file_with(&m, &p, vec![Box::new(Echo { calls: calls.clone() })]).unwrap();
        h.admit(&s, r#"Say(msg: "a")"#).unwrap();
    }
    open(&p);
    let text = std::fs::read_to_string(&p).unwrap();
    let done: Vec<&str> = text.lines().filter(|l| l.contains(r#""type":"EchoDone"}"#)).collect();
    assert_eq!(done.len(), 2, "the record and the delivery of one result: {text}");
    assert!(done[0].contains(r#""why":"req: not delivered as an integer""#) && !text.contains("interrupted"), "{text}");
    replay(&m, &text).unwrap();
    // cut between host.Kept and the delivery it records: completed on open
    let lines: Vec<&str> = text.lines().collect();
    let k = lines.iter().position(|l| l.contains(r#""type":"host.Kept""#)).unwrap();
    let q = std::env::temp_dir().join(format!("rce-fx-reqopt-cut-{}.log", std::process::id()));
    std::fs::write(&q, lines[..=k].iter().map(|l| format!("{l}\n")).collect::<String>()).unwrap();
    open(&q);
    let text = std::fs::read_to_string(&q).unwrap();
    assert_eq!(text.lines().filter(|l| l.contains(r#""type":"EchoDone"}"#) && !l.contains("host.Kept")).count(), 1, "one delivery: {text}");
    assert!(!text.contains("interrupted"), "the completed result is not interrupted as well: {text}");
    replay(&m, &text).unwrap();
    // the same for a result declared with fewer fields (no err)
    let s2 = src()
        .replace("event EchoDone(req: u64, status: text, data: text?, err: text)", "event EchoDone(req: u64, status: text, data: text?)")
        .replace("step S on EchoDone(req: _r, status: st, data: d, err: _e)", "step S on EchoDone(req: _r, status: st, data: d)")
        .replace(r#"EchoDone(req: 1, status: "failed", data: none, err: "x")"#, r#"EchoDone(req: 1, status: "failed", data: none)"#);
    let m2 = parse(&s2).unwrap();
    let r = std::env::temp_dir().join(format!("rce-fx-noerr-cut-{}.log", std::process::id()));
    let _ = std::fs::remove_file(&r);
    {
        let mut h = Host::open_file_with(&m2, &r, vec![Box::new(Echo { calls: calls.clone() })]).unwrap();
        h.admit(&s2, r#"Say(msg: "a")"#).unwrap();
    }
    let text = std::fs::read_to_string(&r).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    let k = lines.iter().position(|l| l.contains(r#""type":"host.Kept""#)).unwrap();
    std::fs::write(&r, lines[..=k].iter().map(|l| format!("{l}\n")).collect::<String>()).unwrap();
    drop(Host::open_file_with(&m2, &r, vec![Box::new(Echo { calls: calls.clone() })]).unwrap());
    let text = std::fs::read_to_string(&r).unwrap();
    assert!(!text.contains("interrupted"), "{text}");
    replay(&m2, &text).unwrap();
    for f in [&p, &q, &r] {
        let _ = std::fs::remove_file(f);
    }
}

/// HC-10b: whether a typed input is marked depends on the module alone,
/// not on the adapters a run is granted. A typed `EchoDone` is marked by
/// `host.Typed` whether or not the Echo adapter is bound.
#[test]
fn a_typed_result_is_marked_without_its_adapter() {
    let s = src();
    let m = parse(&s).unwrap();
    for adapters in [true, false] {
        let mut st = MemStore::default();
        let a: Vec<Box<dyn Adapter>> = if adapters { vec![Box::new(Echo { calls: Default::default() })] } else { vec![] };
        let mut h = Host::start_with(&m, Box::new(&mut st), a).unwrap();
        h.admit(&s, r#"EchoDone(req: 5, status: "failed", data: none, err: "x")"#).unwrap();
        drop(h);
        let k = kinds(&st.lines);
        let i = k.iter().position(|x| x == "EchoDone").unwrap();
        assert_eq!(k[i - 1], "host.Typed", "adapters bound: {adapters}: {k:?}");
    }
}
