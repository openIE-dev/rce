//! End to end: the rce-mcp binary over Streamable HTTP.
use ide_receipt::chain::read_jsonl;
use ide_receipt::{GENESIS, SigningKey, verify_chain};
use rce_mcp::http::{HttpClient, HttpConfig, HttpServer};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

const COUNTER: &str =
    include_str!("../../../../spec/examples/01_counter.rce");

struct Srv {
    child: Child,
    url: String,
    token: String,
    root: PathBuf,
    _t: tempfile::TempDir,
}

impl Drop for Srv {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start(extra: &[&str]) -> Srv {
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("ws");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("counter.rce"), COUNTER).unwrap();
    let tf = t.path().join("token");
    let mut child = Command::new(env!("CARGO_BIN_EXE_rce-mcp"))
        .args(["no-memory", "http", "127.0.0.1:0", "token-file"])
        .arg(&tf)
        .args(extra)
        // The HTTP self-jail is Linux-only; elsewhere HTTP mode needs an explicit opt-out.
        .args(if cfg!(target_os = "linux") { &[][..] } else { &["no-jail"][..] })
        .arg(&root)
        .stdin(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut err = BufReader::new(child.stderr.take().unwrap());
    let mut url = String::new();
    for _ in 0..10 {
        let mut l = String::new();
        err.read_line(&mut l).unwrap();
        if let Some(u) = l.trim().strip_prefix("rce-mcp: listening on ") {
            url = u.to_string();
            break;
        }
    }
    assert!(url.starts_with("http://127.0.0.1:"), "{url}");
    std::thread::spawn(move || for _ in err.lines() {});
    let token = std::fs::read_to_string(&tf).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&tf).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    Srv {
        child,
        url,
        token,
        root,
        _t: t,
    }
}

fn tool(c: &mut HttpClient, name: &str, args: Value) -> Value {
    let v = c
        .request("tools/call", json!({"name": name, "arguments": args}))
        .unwrap();
    v["result"].clone()
}

fn session_flow(s: &Srv) {
    let mut c = HttpClient::new(&s.url, &s.token).unwrap();
    let init = c.initialize("test").unwrap();
    assert_eq!(init["result"]["serverInfo"]["name"], "rce-mcp");
    assert!(c.session.as_ref().is_some_and(|x| x.len() == 32));
    assert_eq!(c.protocol_version.as_deref(), Some("2025-11-25"));
    let tools = c.request("tools/list", json!({})).unwrap();
    assert!(
        tools["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["name"] == "propose_edit")
    );
    let r = tool(
        &mut c,
        "propose_edit",
        json!({"path": "counter.rce", "edits": [
            {"op": "insert_item", "after": "event Dec", "text": "event Reset"},
            {"op": "insert_item", "text": "step Counter on Reset -> 0"}]}),
    );
    assert_eq!(r["isError"], json!(false), "{r}");
    assert!(
        std::fs::read_to_string(s.root.join("counter.rce"))
            .unwrap()
            .contains("event Reset")
    );
    let bad = tool(
        &mut c,
        "propose_edit",
        json!({"path": "counter.rce", "edits": [{"op": "insert_item", "text": "step Counter on Nope -> 0"}]}),
    );
    assert_eq!(bad["isError"], json!(true));
    // Receipts: signed and chained, same as over stdio.
    let key: [u8; 32] = std::fs::read(s.root.join(".rce-ide/ed25519.key"))
        .unwrap()
        .try_into()
        .unwrap();
    let rs = read_jsonl(&s.root.join(".rce-ide/receipts.jsonl")).unwrap();
    assert_eq!(rs.len(), 2);
    verify_chain(&rs, &SigningKey::from_bytes(&key).verifying_key(), GENESIS).unwrap();
    assert!(c.bytes_in > 0 && c.bytes_out > 0);
    // End the session; it is gone afterwards.
    let sid = c.session.clone();
    assert_eq!(c.close().unwrap(), 204);
    c.session = sid;
    match c.request("tools/list", json!({})) {
        Err(rce_mcp::http::HttpError::Status(404, _)) => {}
        r => panic!("{:?}", r.map(|v| v.to_string())),
    }
}

#[test]
fn json_replies_session_tools_receipts() {
    session_flow(&start(&[]));
}

#[test]
fn sse_replies() {
    session_flow(&start(&["sse"]));
}

#[test]
fn auth_origin_accept_session_and_version_are_enforced() {
    let s = start(&["allow-origin", "http://127.0.0.1:8099"]);
    let mut c = HttpClient::new(&s.url, &s.token).unwrap();
    let init = json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}).to_string();
    let good = format!("Bearer {}", s.token);
    let base = |auth: &'static str| -> Vec<(&'static str, &'static str)> {
        vec![
            ("Authorization", auth),
            ("Content-Type", "application/json"),
            ("Accept", "application/json, text/event-stream"),
        ]
    };
    let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
    let g = leak(good);
    // No token, wrong token.
    let r = c.raw("POST", &base("")[1..], init.as_bytes()).unwrap();
    assert_eq!(r.status, 401);
    assert_eq!(r.header("www-authenticate"), Some("Bearer"));
    let wrong = leak(format!("Bearer {}", "0".repeat(64)));
    assert_eq!(
        c.raw("POST", &base(wrong), init.as_bytes()).unwrap().status,
        401
    );
    // A request with a foreign Origin (DNS rebinding) is not performed; the allowed one passes.
    let mut h = base(g);
    h.push(("Origin", "http://evil.example"));
    assert_eq!(c.raw("POST", &h, init.as_bytes()).unwrap().status, 403);
    let mut h = base(g);
    h.push(("Origin", "http://127.0.0.1:8099"));
    let r = c.raw("POST", &h, init.as_bytes()).unwrap();
    assert_eq!(r.status, 200);
    let sid = leak(r.header("mcp-session-id").unwrap().to_string());
    // Accept must list both types; JSON content type required.
    let mut h = base(g);
    h[2] = ("Accept", "application/json");
    assert_eq!(c.raw("POST", &h, init.as_bytes()).unwrap().status, 406);
    let mut h = base(g);
    h[1] = ("Content-Type", "text/plain");
    assert_eq!(c.raw("POST", &h, init.as_bytes()).unwrap().status, 415);
    let list = json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}).to_string();
    // No session, unknown session, bad protocol version, then the good one.
    assert_eq!(
        c.raw("POST", &base(g), list.as_bytes()).unwrap().status,
        400
    );
    let mut h = base(g);
    h.push(("MCP-Session-Id", "deadbeef"));
    assert_eq!(c.raw("POST", &h, list.as_bytes()).unwrap().status, 404);
    let mut h = base(g);
    h.push(("MCP-Session-Id", sid));
    h.push(("MCP-Protocol-Version", "1999-01-01"));
    assert_eq!(c.raw("POST", &h, list.as_bytes()).unwrap().status, 400);
    let mut h = base(g);
    h.push(("MCP-Session-Id", sid));
    h.push(("MCP-Protocol-Version", "2025-11-25"));
    assert_eq!(c.raw("POST", &h, list.as_bytes()).unwrap().status, 200);
    // Notifications: 202, no body. Batches: 400. Bad JSON: 400 parse error.
    let n = json!({"jsonrpc":"2.0","method":"notifications/initialized"}).to_string();
    let r = c.raw("POST", &h, n.as_bytes()).unwrap();
    assert_eq!((r.status, r.body.len()), (202, 0));
    assert_eq!(
        c.raw("POST", &h, format!("[{list}]").as_bytes())
            .unwrap()
            .status,
        400
    );
    let r = c.raw("POST", &h, b"{nope").unwrap();
    assert_eq!(r.status, 400);
    assert!(String::from_utf8_lossy(&r.body).contains("-32700"));
    // GET: no server stream (405). Wrong path: 404.
    let r = c
        .raw(
            "GET",
            &[("Authorization", g), ("Accept", "text/event-stream")],
            b"",
        )
        .unwrap();
    assert_eq!(r.status, 405);
    let mut other = HttpClient::new(&s.url.replace("/mcp", "/other"), &s.token).unwrap();
    assert_eq!(
        other.raw("POST", &base(g), init.as_bytes()).unwrap().status,
        404
    );
}

#[test]
fn binds_loopback_only_and_needs_a_real_token() {
    assert!(HttpServer::bind("0.0.0.0:0", HttpConfig::new("x".repeat(64))).is_err());
    assert!(HttpServer::bind("127.0.0.1:0", HttpConfig::new("short")).is_err());
    let s = HttpServer::bind("127.0.0.1:0", HttpConfig::new("x".repeat(64))).unwrap();
    assert!(s.url().ends_with("/mcp"));
}

#[test]
fn an_idle_connection_does_not_block_other_clients() {
    let s = start(&[]);
    let addr = s
        .url
        .trim_start_matches("http://")
        .split('/')
        .next()
        .unwrap()
        .to_string();
    // Open connections that never send a request (they'd hold a
    // one-at-a-time server for its whole read timeout).
    let idle: Vec<_> = (0..3)
        .map(|_| std::net::TcpStream::connect(&addr).unwrap())
        .collect();
    let t0 = std::time::Instant::now();
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let (url, token) = (s.url.clone(), s.token.clone());
            std::thread::spawn(move || {
                let mut c = HttpClient::new(&url, &token).unwrap();
                c.initialize("par").unwrap();
                c.request("tools/list", json!({})).unwrap()
            })
        })
        .collect();
    for t in threads {
        assert!(t.join().unwrap()["result"]["tools"].is_array());
    }
    assert!(
        t0.elapsed() < std::time::Duration::from_secs(10),
        "{:?}",
        t0.elapsed()
    );
    drop(idle);
}

#[cfg(target_os = "linux")]
#[test]
fn http_mode_jails_itself_and_does_not_reach_undeclared_hosts() {
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("ws");
    std::fs::create_dir_all(&root).unwrap();
    let run = |bind: &str| {
        let mut c = Command::new(env!("CARGO_BIN_EXE_rce-mcp"));
        c.args(["no-memory", "http", bind, "token-file"]).arg(t.path().join("tok")).arg(&root)
            .stdin(Stdio::null()).stderr(Stdio::piped());
        c
    };
    let o = run("0.0.0.0:0").output().unwrap();
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("not a declared host"));
    let mut child = run("127.0.0.1:0").spawn().unwrap();
    let mut err = BufReader::new(child.stderr.take().unwrap());
    let mut seen = String::new();
    for _ in 0..12 {
        let mut l = String::new();
        if err.read_line(&mut l).unwrap() == 0 { break }
        seen.push_str(&l);
        if l.contains("jail ") { break }
    }
    let _ = child.kill();
    let _ = child.wait();
    assert!(seen.contains("rce-mcp: jail landlock+seccomp"), "{seen}");
}
