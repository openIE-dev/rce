//! HC-34 to HC-36: an RCE program served as a tool to agent runtimes that
//! speak RAP, the Reactive Agent Protocol (`rce host serve rap HOST:PORT
//! callback HOST:PORT FILE D`). The toolset is derived from the program's
//! events, each invocation is logged before it is acknowledged and stepped
//! after, a duplicate id is answered from D, and every result is a function
//! of D that `rce host replay rap` reproduces byte for byte. The tests run
//! the `rce` binary with an in-process callback receiver on loopback; no
//! RAP implementation and no network beyond loopback is used. The message
//! shapes are checked against examples copied from the RAP specification
//! (`tests/host/rap/NOTICE.md` names their source and license).
use rce_spec_check::host::{Entry, Host, replay_rap, replay_report};
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const BANK: &str = "../../spec/examples/04_bank_account.rce";
const TYPES: &str = "tests/host/rap/types.rce";

fn app(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn module(rel: &str) -> rce_spec_check::ast::Module {
    rce_spec_check::program(&std::fs::read_to_string(app(rel)).unwrap()).unwrap()
}

/// A loopback address no other test of this process is handed, from a
/// counter below every platform's ephemeral range (as `tests/host_http.rs`).
fn free_addr() -> String {
    static NEXT: AtomicU16 = AtomicU16::new(0);
    let base = 23000 + (std::process::id() % 1000) as u16 * 8;
    loop {
        let port = base + NEXT.fetch_add(1, Ordering::Relaxed);
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return format!("127.0.0.1:{port}");
        }
    }
}

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rce-rap-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

// ---------------------------------------------------------------- callback receiver

/// What the receiver does with one POST: answer with a status, or close the
/// connection without answering (a lost callback).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Answer {
    Status(u16),
    Drop,
}

#[derive(Default)]
struct Got {
    /// (path, body, what the receiver did) per POST, in arrival order
    posts: Vec<(String, Vec<u8>, Answer)>,
    /// what to do with the next POSTs, front first; then `Status(200)`
    plan: Vec<Answer>,
}

/// An in-process RAP runtime's callback endpoint on loopback.
struct Receiver {
    addr: String,
    got: Arc<Mutex<Got>>,
}

impl Receiver {
    /// On a port the system chooses, so no other listener can hold it.
    fn start() -> Receiver {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let got = Arc::new(Mutex::new(Got::default()));
        let g = got.clone();
        std::thread::spawn(move || {
            for s in l.incoming() {
                let Ok(mut s) = s else { continue };
                let g = g.clone();
                std::thread::spawn(move || {
                    s.set_read_timeout(Some(Duration::from_secs(10))).ok();
                    let mut buf = vec![];
                    let mut b = [0u8; 4096];
                    let (head, body) = loop {
                        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            let h = String::from_utf8_lossy(&buf[..i]).into_owned();
                            let n: usize = h.lines().find_map(|l| l.strip_prefix("Content-Length: ")).and_then(|v| v.trim().parse().ok()).unwrap_or(0);
                            if buf.len() >= i + 4 + n {
                                break (h, buf[i + 4..i + 4 + n].to_vec());
                            }
                        }
                        match s.read(&mut b) {
                            Ok(0) | Err(_) => return,
                            Ok(k) => buf.extend_from_slice(&b[..k]),
                        }
                    };
                    let path = head.split(' ').nth(1).unwrap_or("").to_string();
                    let a = {
                        let mut g = g.lock().unwrap();
                        let a = if g.plan.is_empty() { Answer::Status(200) } else { g.plan.remove(0) };
                        g.posts.push((path, body, a));
                        a
                    };
                    if let Answer::Status(c) = a {
                        let _ = s.write_all(format!("HTTP/1.1 {c} X\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes());
                    }
                });
            }
        });
        Receiver { addr, got }
    }
    fn url(&self) -> String {
        format!("http://{}/callback", self.addr)
    }
    fn plan(&self, a: &[Answer]) {
        self.got.lock().unwrap().plan.extend_from_slice(a);
    }
    /// The bodies answered 200, once there are `n`.
    fn delivered(&self, n: usize) -> Vec<Value> {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let v: Vec<Value> = self.got.lock().unwrap().posts.iter().filter(|p| p.2 == Answer::Status(200)).map(|p| serde_json::from_slice(&p.1).unwrap()).collect();
            if v.len() >= n {
                return v;
            }
            assert!(Instant::now() < deadline, "{} of {n} results delivered", v.len());
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    fn posts(&self) -> Vec<(String, Vec<u8>, Answer)> {
        self.got.lock().unwrap().posts.clone()
    }
}

// ---------------------------------------------------------------- the host

struct Served {
    child: Child,
    addr: String,
    log: PathBuf,
    err: Arc<Mutex<String>>,
}

fn serve_at(program: &str, log: &Path, addr: &str, callbacks: &[&str]) -> Served {
    let mut args: Vec<String> = vec!["host".into(), "serve".into(), "rap".into(), addr.into()];
    for c in callbacks {
        args.extend(["callback".into(), c.to_string()]);
    }
    let mut child = Command::new(env!("CARGO_BIN_EXE_rce")).args(&args).arg(app(program)).arg(log).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped()).spawn().unwrap();
    let err = Arc::new(Mutex::new(String::new()));
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
    let deadline = Instant::now() + Duration::from_secs(20);
    while !err.lock().unwrap().contains("rap listening on") {
        assert!(Instant::now() < deadline, "the host did not listen: {}", err.lock().unwrap());
        std::thread::sleep(Duration::from_millis(10));
    }
    Served { child, addr: addr.to_string(), log: log.to_path_buf(), err }
}

impl Served {
    /// SIGTERM, then wait for the exit: (success, stderr).
    fn stop(mut self) -> (bool, String) {
        // SAFETY: a signal to our own child process.
        unsafe { libc::kill(self.child.id() as i32, libc::SIGTERM) };
        let st = self.child.wait().unwrap();
        std::thread::sleep(Duration::from_millis(50));
        (st.success(), self.err.lock().unwrap().clone())
    }
    /// kill -9: D stays as the process left it.
    fn kill(mut self) {
        self.child.kill().unwrap();
        self.child.wait().unwrap();
    }
    fn d(&self) -> String {
        std::fs::read_to_string(&self.log).unwrap()
    }
}

/// One request on its own connection: (status, body).
fn request(addr: &str, method: &str, target: &str, body: &[u8]) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    let mut r = format!("{method} {target} HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", body.len()).into_bytes();
    r.extend_from_slice(body);
    s.write_all(&r).unwrap();
    let mut out = vec![];
    s.read_to_end(&mut out).unwrap();
    let t = String::from_utf8_lossy(&out).into_owned();
    let status = t.split(' ').nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    (status, t.split_once("\r\n\r\n").map(|x| x.1.to_string()).unwrap_or_default())
}

fn invoke(addr: &str, inv: &Value) -> (u16, String) {
    request(addr, "POST", "/invoke", inv.to_string().as_bytes())
}

fn inv(op: &str, args: Value, id: &str, cb: &str) -> Value {
    json!({"operation": op, "arguments": args, "id": id, "call_id": null, "callback_url": cb, "group_id": "thread_1"})
}

fn kinds(d: &str, kind: &str) -> usize {
    d.lines().filter(|l| Entry::parse_line(l).is_some_and(|e| e.kind() == kind)).count()
}

/// `rce host replay rap FILE D`: each result as (call, status, body).
fn replayed(program: &str, log: &Path) -> Vec<(u64, String, String)> {
    let o = Command::new(env!("CARGO_BIN_EXE_rce")).args(["host", "replay", "rap"]).arg(app(program)).arg(log).output().unwrap();
    let t = String::from_utf8_lossy(&o.stdout).into_owned();
    assert!(o.status.success() && t.starts_with("replay ok"), "{t}");
    t.lines()
        .filter_map(|l| l.strip_prefix("rap "))
        .map(|l| {
            let mut p = l.splitn(3, ' ');
            (p.next().unwrap().parse().unwrap(), p.next().unwrap().to_string(), p.next().unwrap().to_string())
        })
        .collect()
}

fn text(r: &Value) -> String {
    r["text"].as_str().unwrap().to_string()
}

// ---------------------------------------------------------------- HC-34

/// HC-34: each event the program accepts from outside is one operation,
/// its fields' types the JSON Schema with `additionalProperties: false`;
/// events the host delivers are not operations; the toolset is a function
/// of the module and the endpoint, and meets RAP's toolset validation.
#[test]
fn toolset_is_derived_from_the_declared_events() {
    let m = module(TYPES);
    let ts: Value = serde_json::from_str(&rce_spec_check::rap::toolset(&m, "http://127.0.0.1:1/invoke").unwrap()).unwrap();
    assert_eq!(ts["name"], "rce-rap.types");
    assert_eq!(ts["endpoint"], "http://127.0.0.1:1/invoke");
    let names: Vec<&str> = ts["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["Put", "Ping"], "Tick, StepKept and Restarted are delivered by the host");
    let put = &ts["tools"][0]["inputSchema"];
    assert_eq!(
        *put,
        json!({
            "additionalProperties": false,
            "properties": {
                "at": {"additionalProperties": false, "properties": {"x": {"maximum": i64::MAX, "minimum": i64::MIN, "type": "integer"}, "y": {"maximum": i64::MAX, "minimum": i64::MIN, "type": "integer"}}, "required": ["x", "y"], "type": "object"},
                "flag": {"type": "boolean"},
                "name": {"type": "string"},
                "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "ratio": {"type": "number"},
                "small": {"maximum": 255, "minimum": 0, "type": "integer"},
                "tags": {"items": {"type": "string"}, "type": "array"}
            },
            "required": ["small", "ratio", "flag", "name", "tags", "at"],
            "type": "object"
        })
    );
    assert_eq!(ts["tools"][1]["inputSchema"], json!({"additionalProperties": false, "properties": {}, "required": [], "type": "object"}));
    // RAP's toolset validation: a non-empty name of at most 128 characters,
    // an endpoint, at least one tool, each with a name of letters, digits,
    // `_` and `-`, a description and an object inputSchema
    let n = ts["name"].as_str().unwrap();
    assert!(!n.is_empty() && n.len() <= 128);
    for t in ts["tools"].as_array().unwrap() {
        let tn = t["name"].as_str().unwrap();
        assert!(!tn.is_empty() && tn.len() <= 128 && tn.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'));
        assert!(t["description"].as_str().is_some_and(|d| !d.is_empty()));
        assert_eq!(t["inputSchema"]["type"], "object");
    }
    // a function of the module and the endpoint: the same bytes twice
    assert_eq!(rce_spec_check::rap::toolset(&m, "http://h/invoke").unwrap(), rce_spec_check::rap::toolset(&m, "http://h/invoke").unwrap());
}

/// HC-34: the host serves the toolset at RAP's discovery path, records its
/// digest in `host.RapListen`, and answers other paths and methods as HTTP does.
#[test]
fn the_toolset_is_served_at_the_discovery_path() {
    let d = dir("toolset");
    let rx = Receiver::start();
    let s = serve_at(BANK, &d.join("d.log"), &free_addr(), &[&rx.addr]);
    let (st, body) = request(&s.addr, "GET", "/.well-known/rap-toolset", b"");
    assert_eq!(st, 200, "{body}");
    let want = rce_spec_check::rap::toolset(&module(BANK), &format!("http://{}/invoke", s.addr)).unwrap();
    assert_eq!(body, want);
    assert_eq!(request(&s.addr, "POST", "/.well-known/rap-toolset", b"").0, 405);
    assert_eq!(request(&s.addr, "GET", "/invoke", b"").0, 405);
    assert_eq!(request(&s.addr, "GET", "/nothing", b"").0, 404);
    let t = s.d();
    let listen = t.lines().filter_map(Entry::parse_line).find(|e| e.kind() == "host.RapListen").unwrap();
    let p: Value = serde_json::from_str(&listen.event).unwrap();
    assert_eq!(p["payload"]["toolset"], rce_spec_check::interp::d_sha(&want));
    let (ok, err) = s.stop();
    assert!(ok, "{err}");
    let _ = std::fs::remove_dir_all(&d);
}

// ---------------------------------------------------------------- HC-35, HC-36

/// HC-35, HC-36: an invocation is logged, acknowledged with 200, stepped,
/// and answered at its callback URL: the outputs of an applied step, an
/// error naming the cause of a kept step, and an error for arguments that
/// do not conform, which are acknowledged too and never stepped.
#[test]
fn an_invocation_is_acked_stepped_and_answered() {
    let d = dir("answer");
    let rx = Receiver::start();
    let s = serve_at(BANK, &d.join("d.log"), &free_addr(), &[&rx.addr]);
    let cb = rx.url();
    let mut withdraw = inv("Withdraw", json!({"amount": 20}), "c2", &cb);
    withdraw["call_id"] = json!("model-7");
    for i in [inv("Deposit", json!({"amount": 50}), "c1", &cb), withdraw, inv("Withdraw", json!({"amount": 1000}), "c3", &cb)] {
        assert_eq!(invoke(&s.addr, &i).0, 200);
        rx.delivered(1);
    }
    let bad = [
        (inv("Transfer", json!({"amount": 1}), "b1", &cb), "unknown operation Transfer; the toolset has Deposit, Withdraw"),
        (inv("Deposit", json!({}), "b2", &cb), "arguments.amount is missing"),
        (inv("Deposit", json!({"amount": 1, "memo": "x"}), "b3", &cb), "arguments.memo is not a field (additionalProperties is false)"),
        (inv("Deposit", json!({"amount": "1"}), "b4", &cb), "arguments.amount is text where i64 is declared"),
        (inv("Deposit", json!({"amount": 1.5}), "b5", &cb), "arguments.amount is a number where i64 is declared"),
        (inv("Deposit", json!([1]), "b6", &cb), "the arguments of Deposit are not a JSON object"),
    ];
    for (i, _) in &bad {
        assert_eq!(invoke(&s.addr, i).0, 200, "a bad invocation is acknowledged (RAP: validation errors are results)");
    }
    let got = rx.delivered(3 + bad.len());
    let by = |id: &str| got.iter().find(|r| r["id"] == id).unwrap_or_else(|| panic!("no result for {id}")).clone();
    let r1 = by("c1");
    assert_eq!(r1["type"], "tool_result");
    assert_eq!(r1["group_id"], "thread_1");
    assert_eq!(r1["call_id"], Value::Null);
    let t1: Value = serde_json::from_str(&text(&r1)).unwrap();
    assert_eq!(t1["outputs"], json!([]));
    let r2 = by("c2");
    assert_eq!(r2["call_id"], "model-7", "call_id is echoed");
    let t2: Value = serde_json::from_str(&text(&r2)).unwrap();
    assert_eq!(t2["outputs"], json!([{"payload": {"amount": 20}, "type": "Withdrawn"}]));
    assert!(t2["seq"].as_u64().unwrap() > t1["seq"].as_u64().unwrap());
    let t3 = text(&by("c3"));
    assert!(t3.starts_with("Error: the step of Withdraw at seq ") && t3.contains("was kept (cause fail): Insufficient"), "{t3}");
    for (i, why) in &bad {
        let t = text(&by(i["id"].as_str().unwrap()));
        assert!(t.starts_with("Error: ") && t.contains(why), "{t} lacks {why}");
    }
    let (ok, err) = s.stop();
    assert!(ok, "{err}");
    let t = std::fs::read_to_string(d.join("d.log")).unwrap();
    assert_eq!(kinds(&t, "host.RapInvoke"), 9);
    assert_eq!(kinds(&t, "host.RapStep"), 3, "only the invocations that conform are stepped");
    assert_eq!(kinds(&t, "host.RapResult"), 9);
    assert_eq!(kinds(&t, "host.RapSent"), 9);
    // every invocation is durable before its acknowledgement: a barrier
    // (host.Synced) follows each host.RapInvoke before its host.RapStep
    let es: Vec<Entry> = t.lines().filter_map(Entry::parse_line).collect();
    for (i, e) in es.iter().enumerate().filter(|(_, e)| e.kind() == "host.RapInvoke") {
        let step = es[i..].iter().position(|x| x.kind() == "host.RapStep" || x.kind() == "host.RapResult").unwrap();
        assert!(es[i..i + step].iter().any(|x| x.kind() == "host.Synced"), "invocation at {} stepped before a barrier", e.seq);
    }
    // 50 deposited, 20 withdrawn; the kept step and the bad invocations changed nothing
    let r = replay_report(&module(BANK), &t).unwrap();
    assert!(format!("{:?}", r.state).contains("Int(30"), "{:?}", r.state);
    let _ = std::fs::remove_dir_all(&d);
}

/// HC-35: an invocation that could not be answered (no id or group_id, no
/// `http://` callback URL, one to an address not granted, a body that is not
/// JSON or nests too deep) is refused with 400, and nothing enters D. A
/// cancellation and a thread closure are acknowledged with 200 and change
/// nothing, since each step is applied when it is admitted.
#[test]
fn unanswerable_invocations_are_refused_and_notifications_change_nothing() {
    let d = dir("refuse");
    let rx = Receiver::start();
    let s = serve_at(BANK, &d.join("d.log"), &free_addr(), &[&rx.addr]);
    let before = s.d().lines().count();
    let cb = rx.url();
    let refused: Vec<Vec<u8>> = vec![
        json!({"operation": "Deposit", "arguments": {"amount": 1}, "group_id": "g", "callback_url": cb}).to_string().into_bytes(),
        json!({"operation": "Deposit", "arguments": {"amount": 1}, "id": "x", "callback_url": cb}).to_string().into_bytes(),
        json!({"operation": "Deposit", "arguments": {"amount": 1}, "id": "x", "group_id": "g"}).to_string().into_bytes(),
        json!({"operation": "Deposit", "arguments": {"amount": 1}, "id": "x", "group_id": "g", "callback_url": format!("https://{}/cb", rx.addr)}).to_string().into_bytes(),
        json!({"operation": "Deposit", "arguments": {"amount": 1}, "id": "x", "group_id": "g", "callback_url": "http://127.0.0.1:9/cb"}).to_string().into_bytes(),
        b"not json".to_vec(),
        format!("{}{}", "[".repeat(200), "]".repeat(200)).into_bytes(),
    ];
    for b in &refused {
        let (st, body) = request(&s.addr, "POST", "/invoke", b);
        assert_eq!(st, 400, "{}", String::from_utf8_lossy(b));
        assert!(serde_json::from_str::<Value>(&body).unwrap()["error"].is_string());
    }
    assert_eq!(request(&s.addr, "POST", "/cancel_tool_call", br#"{"thread_id":"thread_1","tool_call_id":"c1"}"#).0, 200);
    assert_eq!(request(&s.addr, "POST", "/close_thread", br#"{"thread_id":"thread_1"}"#).0, 200);
    assert_eq!(s.d().lines().count(), before, "nothing entered D");
    let (ok, err) = s.stop();
    assert!(ok, "{err}");
    assert!(err.contains("not granted (callback HOST:PORT, HC-5)"), "{err}");
    assert!(rx.posts().is_empty());
    let _ = std::fs::remove_dir_all(&d);
}

/// HC-35: RAP delivers invocations at least once. A second invocation with
/// a (group_id, id) D holds is `host.RapDuplicate`: not stepped again, and
/// answered from the log with the same text, its own call_id echoed.
#[test]
fn a_duplicate_id_is_answered_from_the_log_and_not_stepped_twice() {
    let d = dir("dup");
    let rx = Receiver::start();
    let s = serve_at(BANK, &d.join("d.log"), &free_addr(), &[&rx.addr]);
    let first = inv("Deposit", json!({"amount": 50}), "c1", &rx.url());
    assert_eq!(invoke(&s.addr, &first).0, 200);
    rx.delivered(1);
    let mut again = first.clone();
    again["call_id"] = json!("retry");
    for _ in 0..2 {
        assert_eq!(invoke(&s.addr, &again).0, 200);
    }
    let got = rx.delivered(3);
    assert!(got.iter().all(|r| r["id"] == "c1" && r["text"] == got[0]["text"]), "{got:?}");
    assert_eq!(got.iter().filter(|r| r["call_id"] == "retry").count(), 2);
    let (ok, err) = s.stop();
    assert!(ok, "{err}");
    let t = std::fs::read_to_string(d.join("d.log")).unwrap();
    assert_eq!(kinds(&t, "host.RapInvoke"), 1);
    assert_eq!(kinds(&t, "host.RapDuplicate"), 2);
    assert_eq!(kinds(&t, "Deposit"), 1, "stepped once");
    assert!(format!("{:?}", replay_report(&module(BANK), &t).unwrap().state).contains("Int(50"));
    let _ = std::fs::remove_dir_all(&d);
}

/// HC-36: replay with every adapter disconnected reproduces every result
/// byte for byte, from D alone.
#[test]
fn replay_reproduces_every_result_byte_for_byte() {
    let d = dir("replay");
    let rx = Receiver::start();
    let s = serve_at(BANK, &d.join("d.log"), &free_addr(), &[&rx.addr]);
    let cb = rx.url();
    let mut n = 0;
    for (k, (op, a)) in [("Deposit", 70), ("Withdraw", 5), ("Withdraw", 500), ("Deposit", 3), ("Bogus", 1), ("Withdraw", 68)].iter().enumerate() {
        assert_eq!(invoke(&s.addr, &inv(op, json!({"amount": a}), &format!("r{k}"), &cb)).0, 200);
        n += 1;
    }
    assert_eq!(invoke(&s.addr, &inv("Deposit", json!({"amount": 70}), "r0", &cb)).0, 200);
    rx.delivered(n + 1);
    let (ok, err) = s.stop();
    assert!(ok, "{err}");
    let log = d.join("d.log");
    let rep = replayed(BANK, &log);
    assert_eq!(rep.len(), n + 1);
    let mut wire: Vec<Vec<u8>> = rx.posts().into_iter().map(|p| p.1).collect();
    let mut again: Vec<Vec<u8>> = rep.iter().map(|r| r.2.as_bytes().to_vec()).collect();
    wire.sort();
    again.sort();
    assert_eq!(wire, again, "the bytes the runtime received are the bytes replay derives");
    assert!(rep.iter().all(|r| r.1 == "200"));
    // the library function gives the same bytes
    let t = std::fs::read_to_string(&log).unwrap();
    let lib: Vec<String> = replay_rap(&module(BANK), &t).unwrap().into_iter().map(|o| o.body).collect();
    assert_eq!(lib, rep.iter().map(|r| r.2.clone()).collect::<Vec<_>>());
    let _ = std::fs::remove_dir_all(&d);
}

/// HC-36: a result whose POST is lost (the connection closes without an
/// answer) or answered 5xx is sent again until an answer settles it, and
/// `host.RapSent` records that answer once.
#[test]
fn a_lost_callback_is_retried_until_it_settles() {
    let d = dir("lost");
    let rx = Receiver::start();
    rx.plan(&[Answer::Drop, Answer::Status(503), Answer::Drop]);
    let s = serve_at(BANK, &d.join("d.log"), &free_addr(), &[&rx.addr]);
    assert_eq!(invoke(&s.addr, &inv("Deposit", json!({"amount": 9}), "c1", &rx.url())).0, 200);
    let got = rx.delivered(1);
    let posts = rx.posts();
    assert_eq!(posts.len(), 4, "three lost attempts and the one that settled");
    assert!(posts.iter().all(|p| p.1 == posts[0].1), "every attempt carries the same bytes");
    assert_eq!(got[0]["id"], "c1");
    // a 4xx answer settles a delivery without a retry (RAP)
    rx.plan(&[Answer::Status(404)]);
    assert_eq!(invoke(&s.addr, &inv("Deposit", json!({"amount": 1}), "c2", &rx.url())).0, 200);
    let deadline = Instant::now() + Duration::from_secs(20);
    while kinds(&s.d(), "host.RapSent") < 2 {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(10));
    }
    let (ok, err) = s.stop();
    assert!(ok, "{err}");
    assert_eq!(rx.posts().len(), 5);
    let rep = replayed(BANK, &d.join("d.log"));
    assert_eq!(rep.iter().map(|r| r.1.as_str()).collect::<Vec<_>>(), ["200", "404"]);
    let _ = std::fs::remove_dir_all(&d);
}

/// HC-36 with kill -9: a result D holds and whose delivery never settled
/// is sent again by the next host on D, once, and a settled one never is.
#[test]
fn a_restart_sends_an_unsettled_result_once_more() {
    let d = dir("resend");
    let log = d.join("d.log");
    let addr = free_addr();
    let rx = Receiver::start();
    let s = serve_at(BANK, &log, &addr, &[&rx.addr]);
    assert_eq!(invoke(&s.addr, &inv("Deposit", json!({"amount": 5}), "settled", &rx.url())).0, 200);
    rx.delivered(1);
    // every attempt for the next result is answered 503, then the process dies
    rx.plan(&[Answer::Status(503); 1000]);
    assert_eq!(invoke(&s.addr, &inv("Deposit", json!({"amount": 7}), "open", &rx.url())).0, 200);
    let deadline = Instant::now() + Duration::from_secs(20);
    while rx.posts().len() < 2 {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(10));
    }
    s.kill();
    let t = std::fs::read_to_string(&log).unwrap();
    assert_eq!(kinds(&t, "host.RapResult"), 2);
    assert_eq!(kinds(&t, "host.RapSent"), 1, "the second result is not settled when the process dies");
    rx.got.lock().unwrap().plan.clear();
    let before = rx.posts().len();
    let s = serve_at(BANK, &log, &addr, &[&rx.addr]);
    let deadline = Instant::now() + Duration::from_secs(20);
    while kinds(&s.d(), "host.RapSent") < 2 {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(10));
    }
    let (ok, err) = s.stop();
    assert!(ok, "{err}");
    let after: Vec<Value> = rx.posts()[before..].iter().map(|p| serde_json::from_slice(&p.1).unwrap()).collect();
    assert_eq!(after.len(), 1, "one resend: {after:?}");
    assert_eq!(after[0]["id"], "open");
    let t = std::fs::read_to_string(&log).unwrap();
    assert_eq!(kinds(&t, "host.RapResult"), 2, "a resend appends no second result");
    assert_eq!(kinds(&t, "Deposit"), 2);
    replayed(BANK, &log);
    let _ = std::fs::remove_dir_all(&d);
}

/// HC-35 with kill -9 between the acknowledgement and the step: an
/// invocation D holds durably and that was not stepped is stepped by the
/// next host on D, and its result delivered.
#[test]
fn an_acknowledged_invocation_is_stepped_after_a_kill() {
    let d = dir("acked");
    let log = d.join("d.log");
    let rx = Receiver::start();
    let m = module(BANK);
    {
        // the process that acknowledged: the invocation is durable, then it dies
        let mut h = Host::open_file(&m, &log).unwrap();
        let p = rce_spec_check::rap::envelope(inv("Deposit", json!({"amount": 11}), "k1", &rx.url()).to_string().as_bytes(), std::slice::from_ref(&rx.addr)).unwrap();
        h.rap_invoke(p).unwrap();
        h.rap_commit().unwrap();
    }
    let t = std::fs::read_to_string(&log).unwrap();
    assert_eq!((kinds(&t, "host.RapInvoke"), kinds(&t, "host.RapStep")), (1, 0));
    let s = serve_at(BANK, &log, &free_addr(), &[&rx.addr]);
    let got = rx.delivered(1);
    assert_eq!(got[0]["id"], "k1");
    assert_eq!(serde_json::from_str::<Value>(&text(&got[0])).unwrap()["outputs"], json!([]));
    let (ok, err) = s.stop();
    assert!(ok, "{err}");
    let t = std::fs::read_to_string(&log).unwrap();
    assert_eq!((kinds(&t, "host.RapStep"), kinds(&t, "Deposit"), kinds(&t, "host.RapSent")), (1, 1, 1));
    // the step comes after the restart that found the invocation
    let es: Vec<Entry> = t.lines().filter_map(Entry::parse_line).collect();
    let at = |k: &str| es.iter().position(|e| e.kind() == k).unwrap();
    assert!(at("host.Restarted") < at("host.RapStep"));
    replayed(BANK, &log);
    let _ = std::fs::remove_dir_all(&d);
}

/// HC-36: replay checks each RAP entry against what it derives; a forged
/// result digest, a step of an invocation that does not conform and a second
/// settled delivery are host faults.
#[test]
fn replay_detects_forged_rap_entries() {
    let d = dir("forge");
    let rx = Receiver::start();
    let s = serve_at(BANK, &d.join("d.log"), &free_addr(), &[&rx.addr]);
    assert_eq!(invoke(&s.addr, &inv("Deposit", json!({"amount": 3}), "c1", &rx.url())).0, 200);
    assert_eq!(invoke(&s.addr, &inv("Deposit", json!({"amount": "x"}), "c2", &rx.url())).0, 200);
    rx.delivered(2);
    let (ok, err) = s.stop();
    assert!(ok, "{err}");
    let m = module(BANK);
    let t = std::fs::read_to_string(d.join("d.log")).unwrap();
    assert!(replay_report(&m, &t).is_ok());
    let es: Vec<Entry> = t.lines().filter_map(Entry::parse_line).collect();
    // rewrite the events `f` names and re-chain every entry after them
    let forge = |f: &dyn Fn(&Entry) -> Option<String>| -> String {
        let mut prev = rce_spec_check::interp::D_GENESIS.to_string();
        let mut out = String::new();
        for e in &es {
            let n = Entry::new(&m, e.seq, f(e).unwrap_or(e.event.clone()), &prev, &e.state_hash);
            prev = n.hash.clone();
            out.push_str(&n.line());
            out.push('\n');
        }
        out
    };
    let wrong = forge(&|e| (e.kind() == "host.RapResult").then(|| e.event.replace("\"length\":", "\"length\":1")));
    let r = replay_report(&m, &wrong).unwrap_err();
    assert!(r.contains("host fault") && r.contains("differs from the result replay derives"), "{r}");
    let first = es.iter().find(|e| e.kind() == "host.RapInvoke").unwrap().seq;
    let invalid = es.iter().find(|e| e.kind() == "host.RapInvoke" && e.event.contains("\"amount\":\"x\"")).unwrap().seq;
    let step = forge(&|e| (e.kind() == "host.RapSent" && e.event.contains(&format!("\"call\":{invalid},"))).then(|| format!("{{\"payload\":{{\"call\":{invalid}}},\"type\":\"host.RapStep\"}}")));
    let r = replay_report(&m, &step).unwrap_err();
    assert!(r.contains("host fault") && r.contains("no invocation that conforms"), "{r}");
    let last = es.last().unwrap().seq;
    assert_eq!(es.last().unwrap().kind(), "host.Synced");
    let twice = forge(&|e| (e.seq == last).then(|| format!("{{\"payload\":{{\"call\":{first},\"status\":200}},\"type\":\"host.RapSent\"}}")));
    let r = replay_report(&m, &twice).unwrap_err();
    assert!(r.contains("host fault") && r.contains("already settled"), "{r}");
    let _ = std::fs::remove_dir_all(&d);
}

// ---------------------------------------------------------------- RAP's shapes

/// The shapes `rce host serve rap` reads and writes, against examples
/// copied from the RAP specification (`tests/host/rap/NOTICE.md`): the
/// host keeps every field of the spec's invocation, refuses its `https://`
/// callback (this host POSTs over plain HTTP only), writes a result with
/// exactly the fields and JSON types of the spec's results, and a toolset
/// and tools with the spec's fields.
#[test]
fn rap_messages_match_the_spec_examples() {
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(app("tests/host/rap/spec_examples.json")).unwrap()).unwrap();
    let ex = |k: &str| fx[k]["message"].clone();
    // the invocation: every field of the spec's table is kept, by name
    let i = ex("invocation");
    let refused = rce_spec_check::rap::envelope(i.to_string().as_bytes(), &["agent.example.com:443".into()]).unwrap_err();
    assert!(refused.contains("is not an http:// URL"), "{refused}");
    let mut plain = i.clone();
    plain["callback_url"] = json!("http://agent.example.com/callback");
    let p = rce_spec_check::rap::envelope(plain.to_string().as_bytes(), &["agent.example.com:80".into()]).unwrap();
    let mut keys: Vec<&String> = i.as_object().unwrap().keys().collect();
    keys.sort();
    assert_eq!(p.keys().collect::<Vec<_>>(), keys, "the host's value holds exactly the spec's invocation fields");
    for (k, v) in i.as_object().unwrap() {
        if k != "callback_url" {
            assert_eq!(&p[k], v, "{k}");
        }
    }
    // the result: the spec's fields and JSON types, call_id echoed
    let o = rce_spec_check::rap::Outcome::Applied { event: "E".into(), seq: 7, outputs: vec![], state_hash: "ab".into() };
    let ours: Value = serde_json::from_str(&rce_spec_check::rap::result_body(&p, &o)).unwrap();
    for spec in [ex("tool_result"), ex("error_result")] {
        for (k, v) in spec.as_object().unwrap() {
            assert_eq!(std::mem::discriminant(&ours[k]), std::mem::discriminant(v), "field {k}");
        }
        assert_eq!(ours["group_id"], spec["group_id"]);
        assert_eq!(ours["id"], spec["id"]);
    }
    assert_eq!(ours["type"], "tool_result");
    assert_eq!(ours["call_id"], i["call_id"]);
    let mut ok: Vec<&str> = ours.as_object().unwrap().keys().map(String::as_str).collect();
    ok.sort();
    assert_eq!(ok, ["call_id", "group_id", "id", "text", "type"]);
    let err = rce_spec_check::rap::result_text(&rce_spec_check::rap::Outcome::Invalid("x".into()));
    assert!(err.starts_with("Error: ") && ex("error_result")["text"].as_str().unwrap().starts_with("Error: "), "the spec's error convention");
    // the toolset and a tool: the spec's fields, and nothing it does not define
    let ts: Value = serde_json::from_str(&rce_spec_check::rap::toolset(&module(BANK), "http://h:1/invoke").unwrap()).unwrap();
    let spec_ts = ex("toolset");
    for k in ["name", "description", "endpoint", "tools"] {
        assert_eq!(std::mem::discriminant(&ts[k]), std::mem::discriminant(&spec_ts[k]), "toolset field {k}");
    }
    assert!(ts.as_object().unwrap().keys().all(|k| spec_ts.get(k).is_some()));
    let spec_tool = ex("tool");
    for t in ts["tools"].as_array().unwrap() {
        assert!(t.as_object().unwrap().keys().all(|k| spec_tool.get(k).is_some()), "{t}");
        assert_eq!(t["inputSchema"]["type"], spec_tool["inputSchema"]["type"]);
        assert!(t["inputSchema"]["required"].is_array() && t["inputSchema"]["properties"].is_object());
    }
    // the notifications parse as the spec writes them
    assert!(ex("cancel")["tool_call_id"].is_string() && ex("close_thread")["thread_id"].is_string());
    assert_eq!(rce_spec_check::rap::CANCEL, "/cancel_tool_call");
    assert_eq!(rce_spec_check::rap::CLOSE, "/close_thread");
    assert_eq!(rce_spec_check::rap::DISCOVERY, "/.well-known/rap-toolset");
}
