//! HC-22: adapter resources belong to one host process. Opening a D that
//! holds entries, or taking the admitter role, appends `host.Restarted`
//! {below} (below its own seq), admits the bound `Restarted(below)`, and
//! then interrupts the requests outstanding in D. Ids are issued from below
//! upward, so a handle from before the restart never reaches a new resource.
use rce_spec_check::adapters::net::{NetAdapter, net_adapters};
use rce_spec_check::host::{Adapter, Entry, FileStore, Host, MemStore, replay};
use rce_spec_check::interp::V;
use rce_spec_check::parse::parse;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn read(rel: &str) -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)).unwrap()
}

fn free_addr() -> String {
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    format!("127.0.0.1:{port}")
}

fn boxed(xs: Vec<NetAdapter>) -> Vec<Box<dyn Adapter>> {
    xs.into_iter().map(|a| Box::new(a) as Box<dyn Adapter>).collect()
}

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rce-restart-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn field(h: &Host, name: &str) -> V {
    h.state().iter().flat_map(|(_, fs)| fs.iter()).find(|(n, _, _)| n == name).map(|(_, v, _)| v.clone()).unwrap()
}

fn int(v: &V) -> i64 {
    if let V::Int(i, _) = v { *i as i64 } else { panic!("{v:?}") }
}

fn entries(log: &Path) -> Vec<Entry> {
    std::fs::read_to_string(log).unwrap().lines().map(|l| Entry::parse_line(l).unwrap()).collect()
}

fn position(es: &[Entry], from: usize, f: impl Fn(&Entry) -> bool) -> usize {
    from + es[from..].iter().position(f).unwrap_or_else(|| panic!("not found after {from}"))
}

/// A request outstanding when the process stopped gets its one result
/// (`interrupted`) after `host.Restarted` and the bound `Restarted`.
#[test]
fn restart_appends_restarted_then_interrupts() {
    let src = read("tests/host/net.rce");
    let m = parse(&src).unwrap();
    let d = dir("order");
    let log = d.join("d.log");
    let addr = free_addr();
    let (fs, _) = FileStore::open(&log).unwrap();
    let mut h = Host::start_with(&m, Box::new(fs), boxed(net_adapters(std::slice::from_ref(&addr)))).unwrap();
    let k = h.admit(&src, &format!(r#"Listen(addr: "{addr}")"#)).unwrap()[0].seq;
    drop(h);
    // the process stopped after the request was in D and before its result
    let t = std::fs::read_to_string(&log).unwrap();
    let cut: String = t.lines().take(k as usize + 1).map(|l| format!("{l}\n")).collect();
    std::fs::write(&log, &cut).unwrap();
    let h = Host::open_file_with(&m, &log, boxed(net_adapters(std::slice::from_ref(&addr)))).unwrap();
    assert_eq!(int(&field(&h, "restarts")), 1);
    let live = h.state_hash().to_string();
    drop(h);
    let es = entries(&log);
    let r = position(&es, k as usize + 1, |e| e.event.contains("\"type\":\"host.Restarted\""));
    assert_eq!(es[r].event, format!(r#"{{"payload":{{"below":{r}}},"type":"host.Restarted"}}"#), "below is the entry's own seq");
    assert_eq!(es[r + 1].event, format!(r#"{{"payload":{{"below":{r}}},"type":"Restarted"}}"#), "the bound Restarted follows");
    let i = position(&es, r + 2, |e| e.event.contains("\"type\":\"NetListenDone\""));
    assert!(es[i].event.contains(r#""err":"interrupted""#) && es[i].event.contains(r#""status":"failed""#), "{}", es[i].event);
    assert!(es[..r].iter().all(|e| !e.event.contains("\"type\":\"NetListenDone\"")), "the result comes after the restart");
    assert_eq!(replay(&m, &std::fs::read_to_string(&log).unwrap()).unwrap().1, live);
    let _ = std::fs::remove_dir_all(&d);
}

/// HC-22: ids continue from below, so a stale id fails instead of reaching
/// a resource of the new process.
#[test]
fn ids_are_not_reused_after_restart() {
    let src = read("tests/host/net.rce");
    let m = parse(&src).unwrap();
    let d = dir("ids");
    let log = d.join("d.log");
    let addr = free_addr();
    let (fs, _) = FileStore::open(&log).unwrap();
    let mut h = Host::start_with(&m, Box::new(fs), boxed(net_adapters(std::slice::from_ref(&addr)))).unwrap();
    h.admit(&src, &format!(r#"Listen(addr: "{addr}")"#)).unwrap();
    assert_eq!(int(&field(&h, "last")), 1, "the first listener is id 1");
    drop(h);
    let mut h = Host::open_file_with(&m, &log, boxed(net_adapters(std::slice::from_ref(&addr)))).unwrap();
    let es = entries(&log);
    let below = es.iter().rev().find(|e| e.event.contains("\"type\":\"host.Restarted\"")).unwrap().seq;
    h.admit(&src, &format!(r#"Listen(addr: "{addr}")"#)).unwrap();
    let id = int(&field(&h, "last"));
    assert_eq!(id as u64, below, "the new listener's id is below, above every id issued before");
    // the old handle names nothing now: `no listener 1`, not the new socket
    h.admit(&src, "Accept(id: 1)").unwrap();
    let live = h.state_hash().to_string();
    drop(h);
    let t = std::fs::read_to_string(&log).unwrap();
    assert!(t.lines().last().is_some());
    let accept = entries(&log).into_iter().rev().find(|e| e.event.contains("NetAcceptDone")).unwrap();
    assert!(accept.event.contains(r#""err":"no listener 1""#), "{}", accept.event);
    assert_eq!(replay(&m, &t).unwrap().1, live);
    let _ = std::fs::remove_dir_all(&d);
}

/// HC-22: the host loop's Tick(n) continues from the last n in D.
#[test]
fn tick_numbers_continue() {
    let src = read("tests/host/ticked.rce");
    let m = parse(&src).unwrap();
    let d = dir("tick");
    let log = d.join("d.log");
    let (fs, _) = FileStore::open(&log).unwrap();
    let mut h = Host::start(&m, Box::new(fs)).unwrap();
    h.admit(&src, "Tick(n: 1), Tick(n: 2), Tick(n: 7)").unwrap();
    assert_eq!(h.last_tick(), 7);
    drop(h);
    let h = Host::open_file(&m, &log).unwrap();
    assert_eq!(h.last_tick(), 7, "refolded from D");
    drop(h);
    // end to end: two runs of the host loop, each for a few frames
    let rce = env!("CARGO_BIN_EXE_rce");
    for _ in 0..2 {
        let mut c = Command::new(rce).args(["host", "serve"]).arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/ticked.rce")).arg("--log").arg(&log).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
        std::thread::sleep(Duration::from_millis(300));
        drop(c.stdin.take());
        let o = c.wait_with_output().unwrap();
        assert!(String::from_utf8_lossy(&o.stderr).contains("validated"), "{}", String::from_utf8_lossy(&o.stderr));
    }
    let ns: Vec<u64> = entries(&log)
        .iter()
        .filter_map(|e| {
            let v: serde_json::Value = serde_json::from_str(&e.event).ok()?;
            (v["type"] == "Tick").then(|| v["payload"]["n"].as_u64()).flatten()
        })
        .collect();
    assert!(ns.len() > 5, "{ns:?}");
    assert!(ns.windows(2).all(|w| w[0] < w[1]), "Tick numbers rise across restarts: {ns:?}");
    let _ = std::fs::remove_dir_all(&d);
}

/// E.2b with HC-22: the new admitter's adapters are fresh.
#[test]
fn take_role_appends_restarted() {
    let src = read("tests/host/net.rce");
    let m = parse(&src).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    h.admit(&src, "Roll(n: 4)").unwrap();
    h.take_role(1, "a").unwrap();
    assert_eq!(int(&field(&h, "restarts")), 1);
    let live = h.state_hash().to_string();
    drop(h);
    let es: Vec<Entry> = st.lines.iter().map(|l| Entry::parse_line(l).unwrap()).collect();
    let t = es.iter().position(|e| e.event.contains("host.Term")).unwrap();
    assert_eq!(es[t + 1].event, format!(r#"{{"payload":{{"below":{}}},"type":"host.Restarted"}}"#, t + 1));
    assert_eq!(es[t + 2].event, format!(r#"{{"payload":{{"below":{}}},"type":"Restarted"}}"#, t + 1));
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert_eq!(replay(&m, &text).unwrap().1, live);
    // replay checks below and the bound payload
    let mut forged = es.clone();
    forged[t + 1].event = format!(r#"{{"payload":{{"below":{}}},"type":"host.Restarted"}}"#, t);
    let mut prev = rce_spec_check::interp::D_GENESIS.to_string();
    let mut f = String::new();
    for (i, e) in forged.into_iter().enumerate() {
        let n = Entry::new(&m, i as u64, e.event, &prev, &e.state_hash);
        prev = n.hash.clone();
        f.push_str(&n.line());
        f.push('\n');
    }
    let e = replay(&m, &f).unwrap_err();
    assert!(e.contains("host fault") && e.contains("below"), "{e}");
}

fn get(addr: &str, req: &str) -> Option<String> {
    let mut s = TcpStream::connect(addr).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    s.write_all(req.as_bytes()).ok()?;
    let mut out = vec![];
    s.read_to_end(&mut out).ok()?;
    let t = String::from_utf8_lossy(&out).into_owned();
    (!t.is_empty()).then_some(t)
}

fn get_until(addr: &str, req: &str) -> String {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if let Some(r) = get(addr, req) {
            return r;
        }
        assert!(Instant::now() < deadline, "no answer from {addr}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Audit defect 4: `rce host serve` on an existing D refolded to a listener
/// that did not exist in the new process, logged "no listener 1" every
/// Tick and never bound the port. With `Restarted` the program forgets its
/// sockets and listens again.
#[test]
fn serve_on_an_existing_log_listens_again() {
    let d = dir("serve");
    let log = d.join("d.log");
    let addr = free_addr();
    let app = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/notes_net.rce");
    let rce = env!("CARGO_BIN_EXE_rce");
    for round in 0..2 {
        let mut c = Command::new(rce).args(["host", "serve"]).arg(&app).arg("--log").arg(&log).arg("--net").arg(&addr).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
        let mut i = c.stdin.take().unwrap();
        if round == 0 {
            writeln!(i, "Start(addr: \"{addr}\")").unwrap();
            let r = get_until(&addr, "POST /notes HTTP/1.1\r\nHost: x\r\nContent-Length: 4\r\n\r\nmilk");
            assert!(r.starts_with("HTTP/1.1 201") && r.contains(r#"{"id":1,"text":"milk"}"#), "{r}");
        } else {
            // nothing written: the address is in D, and the restart
            // handler makes the next Tick listen again
            let r = get_until(&addr, "GET /notes HTTP/1.1\r\nHost: x\r\n\r\n");
            assert!(r.starts_with("HTTP/1.1 200") && r.contains(r#"[{"id":1,"text":"milk"}]"#), "{r}");
        }
        drop(i);
        let o = c.wait_with_output().unwrap();
        let err = String::from_utf8_lossy(&o.stderr);
        assert!(err.contains("validated"), "{err}");
    }
    let t = std::fs::read_to_string(&log).unwrap();
    assert!(!t.contains("no listener"), "no request named a listener of the old process");
    assert_eq!(t.matches("\"type\":\"host.Restarted\"").count(), 1);
    let _ = std::fs::remove_dir_all(&d);
}
