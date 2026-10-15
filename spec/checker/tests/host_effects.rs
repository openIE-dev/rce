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
        if r["msg"] == "boom" { EffectResult::failed("refused by adapter") } else { EffectResult::ok(r["msg"].clone()) }
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
    assert!(text.contains(r#""data":null,"err":"refused by adapter","req":2,"status":"failed""#), "{text}");
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

#[test]
fn bound_effect_without_its_result_event_is_refused() {
    let s = src().replace("event EchoDone(req: u64, status: text, data: text?, err: text)", "event EchoDone(req: u64, status: text)").replace(", data: d, err: e)", ")").replace("        got.push(d.or(\"\"))\n", "        got.push(\"\")\n").replace(r#"EchoDone(req: 1, status: "failed", data: none, err: "x")"#, r#"EchoDone(req: 1, status: "failed")"#);
    let m = parse(&s).unwrap();
    let mut st = MemStore::default();
    let e = Host::start_with(&m, Box::new(&mut st), vec![Box::new(Echo { calls: Default::default() })]).err().unwrap();
    assert!(e.contains("W-HOST-1"), "{e}");
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
