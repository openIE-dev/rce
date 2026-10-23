//! Declared network capabilities: allow, deny and undeclared, against a
//! local mock server, through the proxy directly, through a wasm guest, and
//! through the Linux jail's Landlock TCP rules.
use ide_sandbox::net::*;
use ide_sandbox::wasi::{FuelLimit, Sandbox};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Mock HTTP server: records each request, answers `200 ok`.
struct Mock {
    port: u16,
    seen: Arc<Mutex<Vec<String>>>,
}

fn mock() -> Mock {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(vec![]));
    let s2 = seen.clone();
    std::thread::spawn(move || {
        for c in l.incoming() {
            let Ok(mut c) = c else { continue };
            let mut b = Vec::new();
            let _ = c.read_to_end(&mut b);
            s2.lock()
                .unwrap()
                .push(String::from_utf8_lossy(&b).into_owned());
            let _ = c.write_all(b"HTTP/1.0 200 OK\r\nContent-Length: 2\r\n\r\nok");
        }
    });
    Mock { port, seen }
}

struct Counting(Arc<AtomicUsize>, bool);
impl Approver for Counting {
    fn approve(&mut self, _: &NetCap) -> bool {
        self.0.fetch_add(1, Ordering::SeqCst);
        self.1
    }
}

fn manifest(port: u16) -> NetManifest {
    NetManifest::parse(&format!(
        r#"{{"net":[{{"host":"127.0.0.1","port":{port},"proto":"http","reason":"test api"}}]}}"#
    ))
    .unwrap()
}

fn proxy(port: u16, yes: bool) -> (NetProxy, Arc<AtomicUsize>) {
    let n = Arc::new(AtomicUsize::new(0));
    let mut creds = Credentials::default();
    creds.set(
        &endpoint_key(Proto::Http, "127.0.0.1", port),
        "Authorization",
        "Bearer SUPERVISOR",
    );
    let p = NetProxy::new(
        manifest(port),
        Approvals::default(),
        Box::new(Counting(n.clone(), yes)),
        creds,
    );
    (p, n)
}

const REQ: &[u8] =
    b"GET /v1/x HTTP/1.1\r\nHost: evil\r\nAuthorization: Bearer GUEST\r\nX-A: 1\r\n\r\n";

#[test]
fn allow_injects_credentials_and_counts_bytes() {
    let m = mock();
    let (mut p, asked) = proxy(m.port, true);
    assert_eq!(p.pending().len(), 1);
    let r = p.request(Proto::Http, "127.0.0.1", m.port, REQ).unwrap();
    assert!(r.ends_with(b"ok"));
    p.request(Proto::Http, "127.0.0.1", m.port, REQ).unwrap();
    assert_eq!(
        asked.load(Ordering::SeqCst),
        1,
        "asked once, then remembered"
    );
    assert!(p.pending().is_empty());
    let seen = m.seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert!(seen[0].contains("Authorization: Bearer SUPERVISOR"));
    assert!(!seen[0].contains("GUEST") && !seen[0].contains("evil"));
    assert!(seen[0].contains(&format!("Host: 127.0.0.1:{}", m.port)) && seen[0].contains("X-A: 1"));
    let e = &p.events[0];
    assert_eq!(e.decision, Decision::Allowed);
    assert!(e.credential_injected && e.bytes_out > 0 && e.bytes_in > 0);
    assert_eq!(
        p.bytes(),
        p.events
            .iter()
            .map(|e| e.bytes_in + e.bytes_out)
            .sum::<u64>()
    );
    // credentials never serialized
    let j = serde_json::to_string(&p.events).unwrap();
    assert!(!j.contains("SUPERVISOR"));
}

#[test]
fn declared_but_denied_never_connects() {
    let m = mock();
    let (mut p, asked) = proxy(m.port, false);
    let e = p
        .request(Proto::Http, "127.0.0.1", m.port, REQ)
        .unwrap_err();
    assert!(matches!(e, NetError::Denied(_)));
    p.request(Proto::Http, "127.0.0.1", m.port, REQ)
        .unwrap_err();
    assert_eq!(asked.load(Ordering::SeqCst), 1);
    std::thread::sleep(std::time::Duration::from_millis(50));
    assert!(m.seen.lock().unwrap().is_empty());
    assert_eq!(p.events[0].decision, Decision::Denied);
    assert_eq!(p.bytes(), 0);
}

#[test]
fn undeclared_is_denied_without_asking() {
    let m = mock();
    let other = mock();
    let (mut p, asked) = proxy(m.port, true);
    let e = p
        .request(Proto::Http, "127.0.0.1", other.port, REQ)
        .unwrap_err();
    assert!(matches!(e, NetError::Undeclared(_)));
    // same port, different protocol: also undeclared
    let e = p
        .request(Proto::Tcp, "127.0.0.1", m.port, b"x")
        .unwrap_err();
    assert!(matches!(e, NetError::Undeclared(_)));
    assert_eq!(asked.load(Ordering::SeqCst), 0);
    assert!(other.seen.lock().unwrap().is_empty() && m.seen.lock().unwrap().is_empty());
    assert!(p.events.iter().all(|e| e.decision == Decision::Undeclared));
}

#[test]
fn approvals_persist_and_manifest_validates() {
    let t = tempfile::tempdir().unwrap();
    let path = t.path().join(".rce-ide/net-approvals.json");
    let mut a = Approvals::default();
    a.decided.insert("http://h:1".into(), true);
    a.save(&path).unwrap();
    assert_eq!(Approvals::load(&path), a);
    assert!(NetManifest::parse(r#"{"net":[{"host":"a/b","port":1,"proto":"tcp"}]}"#).is_err());
    assert!(NetManifest::parse(r#"{"net":[{"host":"a","port":0,"proto":"tcp"}]}"#).is_err());
    assert_eq!(NetManifest::load(t.path()).unwrap(), NetManifest::default());
    let _ = Fixed(BTreeMap::new());
}

/// A guest that calls `rce_net.request` once and exits 0 if it got a
/// response, else `-code` (1 undeclared, 2 denied).
fn guest(port: u16) -> Vec<u8> {
    let req = "GET / HTTP/1.1\\0d\\0a\\0d\\0a";
    wat::parse_str(format!(
        r#"(module
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (import "rce_net" "request" (func $req (param i32 i32 i32 i32 i32 i32 i32 i32) (result i64)))
  (memory (export "memory") 1)
  (data (i32.const 0) "127.0.0.1")
  (data (i32.const 16) "{req}")
  (func (export "_start") (local $r i64)
    (local.set $r (call $req (i32.const 0) (i32.const 0) (i32.const 9) (i32.const {port})
                             (i32.const 16) (i32.const 18) (i32.const 1024) (i32.const 4096)))
    (if (i64.gt_s (local.get $r) (i64.const 0))
      (then (call $exit (i32.const 0)))
      (else (call $exit (i32.wrap_i64 (i64.sub (i64.const 0) (local.get $r))))))))"#
    ))
    .unwrap()
}

#[test]
fn wasm_guest_gets_exactly_the_declared_endpoint() {
    let m = mock();
    let other = mock();
    let ws = tempfile::tempdir().unwrap();
    let (mut p, _) = proxy(m.port, true);
    let r = Sandbox::new(&guest(m.port))
        .unwrap()
        .run_with_net(ws.path(), &[], FuelLimit::Fuel(1_000_000), &mut p)
        .unwrap();
    assert_eq!(r.exit_code, Some(0), "{r:?}");
    assert!(r.net_bytes > 0 && r.features().net_bytes == r.net_bytes);
    assert_eq!(r.net.len(), 1);
    // The run's bytes reach the simulator features and a signed receipt.
    let mut d = ide_receipt::Draft::new(ide_receipt::ActionKind::Run, "sandbox", "p");
    d.detail = r.receipt_detail();
    let mut led = ide_receipt::chain::Ledger::new(ide_receipt::generate_key().unwrap());
    let rc = led.append(d).unwrap();
    assert_eq!(rc.body.draft.detail["net_bytes"], r.net_bytes);
    assert_eq!(rc.body.draft.detail["net"][0]["decision"], "allowed");
    let r = Sandbox::new(&guest(other.port))
        .unwrap()
        .run_with_net(ws.path(), &[], FuelLimit::Fuel(1_000_000), &mut p)
        .unwrap();
    assert_eq!(r.exit_code, Some(1), "undeclared");
    assert_eq!(r.net_bytes, 0);
    let (mut d, _) = proxy(m.port, false);
    let r = Sandbox::new(&guest(m.port))
        .unwrap()
        .run_with_net(ws.path(), &[], FuelLimit::Fuel(1_000_000), &mut d)
        .unwrap();
    assert_eq!(r.exit_code, Some(2), "denied");
    // default run: no network at all
    let r = Sandbox::new(&guest(m.port))
        .unwrap()
        .run(ws.path(), &[], FuelLimit::Fuel(1_000_000))
        .unwrap();
    assert_eq!(r.exit_code, Some(1));
    assert_eq!(m.seen.lock().unwrap().len(), 1);
}

#[cfg(target_os = "linux")]
#[test]
fn landlock_allows_only_declared_tcp_ports() {
    use ide_sandbox::jail::{JailPolicy, platform_jail};
    use std::path::{Path, PathBuf};
    use std::process::Command;
    if !Path::new("/bin/bash").exists() {
        eprintln!("SKIP: no /bin/bash for /dev/tcp");
        return;
    }
    let m = mock();
    let other = mock();
    let pol = JailPolicy {
        rw: vec![],
        ro: ["/usr", "/bin", "/lib", "/lib64", "/etc"]
            .iter()
            .map(PathBuf::from)
            .filter(|p| p.exists())
            .collect(),
        allow_net: false,
        tcp_connect: manifest(m.port).ports(),
    };
    let j = platform_jail();
    if let Err(e) = j.probe(&pol) {
        eprintln!("SKIP: jail unavailable here: {e}");
        return;
    }
    let conn = |port: u16| {
        let mut c = Command::new("/bin/bash");
        c.args(["-c", &format!("exec 3<>/dev/tcp/127.0.0.1/{port}")]);
        j.run(c, &pol).unwrap()
    };
    assert!(conn(m.port).status.success(), "declared port");
    let o = conn(other.port);
    assert!(!o.status.success(), "undeclared port");
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("not permitted") || err.contains("denied"),
        "{err}"
    );
    // UDP is refused by seccomp even though sockets are allowed.
    let mut c = Command::new("/bin/bash");
    c.args(["-c", &format!("echo x > /dev/udp/127.0.0.1/{}", m.port)]);
    assert!(!j.run(c, &pol).unwrap().status.success(), "udp");
}
