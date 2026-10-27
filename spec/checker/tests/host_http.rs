//! HC-24 to HC-26: the HTTP event host (`rce host serve --http HOST:PORT`).
//! Connections are served concurrently, each request is one logged input
//! (`HttpRequest` or `HttpMalformed`), the program answers with the
//! `HttpRespond` effect, limits and deadlines apply before a byte enters D,
//! and every byte written is a function of D. These tests run the `rce`
//! binary on programs/network/notes_http.rce and tests/host/http_echo.rce.
use rce_spec_check::host::{Entry, replay_outputs};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const NOTES: &str = "../../programs/network/notes_http.rce";
const ECHO: &str = "tests/host/http_echo.rce";

fn app(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// A loopback address that no other test of this process is handed. Ports
/// come from a counter below every platform's ephemeral range (Linux starts
/// at 32768, macOS and Windows at 49152), so neither a concurrent test's
/// `bind("127.0.0.1:0")` nor a connection's local port can take the port
/// between this probe and the bind of the host under test.
fn free_addr() -> String {
    static NEXT: AtomicU16 = AtomicU16::new(0);
    let base = 20000 + (std::process::id() % 1000) as u16 * 10;
    loop {
        let port = base + NEXT.fetch_add(1, Ordering::Relaxed);
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return format!("127.0.0.1:{port}");
        }
    }
}

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rce-http-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct Served {
    child: Child,
    addr: String,
    log: PathBuf,
    err: Arc<Mutex<String>>,
}

fn serve_at(program: &str, log: &Path, addr: &str) -> Served {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rce"))
        .args(["host", "serve"])
        .arg(app(program))
        .arg("--log")
        .arg(log)
        .arg("--http")
        .arg(addr)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
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
    while !err.lock().unwrap().contains("http listening on") {
        assert!(Instant::now() < deadline, "the host did not listen: {}", err.lock().unwrap());
        std::thread::sleep(Duration::from_millis(10));
    }
    Served { child, addr: addr.to_string(), log: log.to_path_buf(), err }
}

fn serve(program: &str, d: &Path) -> Served {
    serve_at(program, &d.join("d.log"), &free_addr())
}

impl Served {
    /// SIGTERM, then wait for the exit: (success, stderr).
    fn stop(mut self) -> (bool, String) {
        // SAFETY: a signal to our own child process.
        unsafe { libc::kill(self.child.id() as i32, libc::SIGTERM) };
        let st = self.child.wait().unwrap();
        std::thread::sleep(Duration::from_millis(50));
        let e = self.err.lock().unwrap().clone();
        (st.success(), e)
    }
    fn lines(&self) -> usize {
        std::fs::read_to_string(&self.log).unwrap().lines().count()
    }
}

/// One request on its own connection; the whole answer up to the close.
fn exchange(addr: &str, req: &[u8]) -> String {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    s.write_all(req).unwrap();
    let mut out = vec![];
    s.read_to_end(&mut out).unwrap();
    String::from_utf8_lossy(&out).into_owned()
}

fn get(addr: &str, target: &str) -> String {
    exchange(addr, format!("GET {target} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").as_bytes())
}

fn post(addr: &str, target: &str, body: &[u8]) -> String {
    let mut r = format!("POST {target} HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Length: {}\r\n\r\n", body.len()).into_bytes();
    r.extend_from_slice(body);
    exchange(addr, &r)
}

/// One response from a kept-alive connection: (status, fields, body).
fn read_response(s: &mut TcpStream, buf: &mut Vec<u8>, head: bool) -> (u16, Vec<(String, String)>, Vec<u8>) {
    let mut b = [0u8; 65536];
    loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let h = String::from_utf8_lossy(&buf[..i]).into_owned();
            let mut ls = h.split("\r\n");
            let status: u16 = ls.next().unwrap().split(' ').nth(1).unwrap().parse().unwrap();
            let fs: Vec<(String, String)> = ls.map(|l| l.split_once(": ").map(|(a, b)| (a.to_string(), b.to_string())).unwrap()).collect();
            let n: usize = if head { 0 } else { fs.iter().find(|(k, _)| k == "Content-Length").map(|(_, v)| v.parse().unwrap()).unwrap_or(0) };
            if buf.len() >= i + 4 + n {
                let body = buf[i + 4..i + 4 + n].to_vec();
                buf.drain(..i + 4 + n);
                return (status, fs, body);
            }
        }
        let k = s.read(&mut b).unwrap();
        assert!(k > 0, "closed before a whole response: {:?}", String::from_utf8_lossy(buf));
        buf.extend_from_slice(&b[..k]);
    }
}

fn entries(log: &Path) -> Vec<Entry> {
    std::fs::read_to_string(log).unwrap().lines().map(|l| Entry::parse_line(l).unwrap()).collect()
}

fn replays(program: &str, log: &Path) {
    let r = Command::new(env!("CARGO_BIN_EXE_rce")).args(["host", "replay"]).arg(app(program)).arg("--log").arg(log).output().unwrap();
    let out = String::from_utf8_lossy(&r.stdout);
    assert!(out.starts_with("replay ok"), "{out}");
}

/// Audit defect 3: one idle client blocked every other. Now B to E are
/// answered while A has sent half a head, and A gets 408 after HEAD_MS.
#[test]
fn concurrent_clients_are_served_while_one_stalls() {
    let d = dir("stall");
    let s = serve(NOTES, &d);
    let mut a = TcpStream::connect(&s.addr).unwrap();
    a.write_all(b"GET /health HTTP/1.1\r\nHo").unwrap();
    let t0 = Instant::now();
    for _ in 0..4 {
        let r = get(&s.addr, "/health");
        assert!(r.starts_with("HTTP/1.1 200 OK") && r.ends_with(r#"{"ok":true}"#), "{r}");
    }
    assert!(t0.elapsed() < Duration::from_secs(5), "the others waited for the stalled client: {:?}", t0.elapsed());
    a.set_read_timeout(Some(Duration::from_secs(15))).unwrap();
    let mut out = vec![];
    a.read_to_end(&mut out).unwrap();
    let r = String::from_utf8_lossy(&out);
    assert!(r.starts_with("HTTP/1.1 408 Request Timeout") && r.contains(r#"{"error":"head timed out"}"#) && r.contains("Connection: close"), "{r}");
    assert!(t0.elapsed() < Duration::from_secs(11), "408 after {:?}", t0.elapsed());
    let log = s.log.clone();
    let (ok, err) = s.stop();
    assert!(ok && err.contains("validated"), "{err}");
    let t = std::fs::read_to_string(&log).unwrap();
    assert!(t.contains(r#""status":408,"why":"head timed out""#), "HttpMalformed is logged");
    replays(NOTES, &log);
    let _ = std::fs::remove_dir_all(&d);
}

/// No Tick, no polling: an idle server appends nothing.
#[test]
fn idle_server_appends_nothing() {
    let d = dir("idle");
    let s = serve(NOTES, &d);
    assert!(get(&s.addr, "/health").starts_with("HTTP/1.1 200"));
    std::thread::sleep(Duration::from_millis(100));
    let before = s.lines();
    std::thread::sleep(Duration::from_secs(2));
    assert_eq!(s.lines(), before, "entries appended while idle");
    let (ok, err) = s.stop();
    assert!(ok, "{err}");
    let _ = std::fs::remove_dir_all(&d);
}

/// RFC 9112 9.3: requests on one connection, pipelined in one write, are
/// admitted one at a time and answered in order.
#[test]
fn keep_alive_and_pipelined_requests_answer_in_order() {
    let d = dir("pipe");
    let s = serve(NOTES, &d);
    let mut c = TcpStream::connect(&s.addr).unwrap();
    c.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    c.write_all(b"POST /notes HTTP/1.1\r\nHost: x\r\nContent-Length: 1\r\n\r\naPOST /notes HTTP/1.1\r\nHost: x\r\nContent-Length: 1\r\n\r\nbGET /notes HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
    let mut buf = vec![];
    let r1 = read_response(&mut c, &mut buf, false);
    let r2 = read_response(&mut c, &mut buf, false);
    let r3 = read_response(&mut c, &mut buf, false);
    assert_eq!((r1.0, String::from_utf8_lossy(&r1.2).into_owned()), (201, r#"{"id":1,"text":"a"}"#.to_string()));
    assert_eq!((r2.0, String::from_utf8_lossy(&r2.2).into_owned()), (201, r#"{"id":2,"text":"b"}"#.to_string()));
    assert_eq!((r3.0, String::from_utf8_lossy(&r3.2).into_owned()), (200, r#"[{"id":1,"text":"a"},{"id":2,"text":"b"}]"#.to_string()));
    assert!(r3.1.iter().all(|(k, _)| k != "Connection"), "the connection stays open: {:?}", r3.1);
    // the same connection answers again
    c.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
    assert_eq!(read_response(&mut c, &mut buf, false).0, 200);
    drop(c);
    let (ok, err) = s.stop();
    assert!(ok, "{err}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn head_has_no_body() {
    let d = dir("head");
    let s = serve(NOTES, &d);
    let mut c = TcpStream::connect(&s.addr).unwrap();
    c.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    c.write_all(b"HEAD /health HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
    let mut buf = vec![];
    let (st, fs, body) = read_response(&mut c, &mut buf, true);
    assert_eq!(st, 200);
    assert!(fs.contains(&("Content-Length".to_string(), "11".to_string())), "{fs:?}");
    assert!(body.is_empty());
    // the next response on the connection starts right after the head
    c.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
    let (st, _, body) = read_response(&mut c, &mut buf, false);
    assert_eq!((st, body), (200, br#"{"ok":true}"#.to_vec()));
    drop(c);
    let (ok, err) = s.stop();
    assert!(ok, "{err}");
    let _ = std::fs::remove_dir_all(&d);
}

/// Audit wedge: a 1.5 MB body. It is refused from its head (413) without
/// being read, the connection closes, and the service goes on serving.
#[test]
fn oversized_body_is_413_and_closes() {
    let d = dir("big");
    let s = serve(NOTES, &d);
    let r = exchange(&s.addr, b"POST /notes HTTP/1.1\r\nHost: x\r\nContent-Length: 1000001\r\n\r\n");
    assert!(r.starts_with("HTTP/1.1 413 Content Too Large") && r.contains("Connection: close") && r.ends_with(r#"{"error":"body over 1000000 bytes"}"#), "{r}");
    let addr = s.addr.clone();
    let sender = std::thread::spawn(move || {
        let mut c = TcpStream::connect(&addr).unwrap();
        c.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
        c.write_all(b"POST /notes HTTP/1.1\r\nHost: x\r\nContent-Length: 1500000\r\n\r\n").unwrap();
        let _ = c.write_all(&vec![b'a'; 1_500_000]);
        let mut out = vec![];
        let _ = c.read_to_end(&mut out);
        String::from_utf8_lossy(&out).into_owned()
    });
    let r = sender.join().unwrap();
    assert!(r.starts_with("HTTP/1.1 413"), "{r}");
    assert!(get(&s.addr, "/health").starts_with("HTTP/1.1 200"), "still serving");
    // a body of exactly the note limit plus one is the program's 413
    let r = post(&s.addr, "/notes", &vec![b'a'; 4097]);
    assert!(r.starts_with("HTTP/1.1 413") && r.ends_with(r#"{"error":"note over 4096 bytes"}"#), "{r}");
    let log = s.log.clone();
    let (ok, err) = s.stop();
    assert!(ok && err.contains("validated"), "{err}");
    replays(NOTES, &log);
    let _ = std::fs::remove_dir_all(&d);
}

/// HC-10a end to end: a lone 0xFF, and a body that is not UTF-8, arrive
/// exactly and get the program's 400; a character split across two
/// writes is one note.
#[test]
fn invalid_body_bytes_are_exact_and_get_400() {
    let d = dir("bytes");
    let s = serve(NOTES, &d);
    let r = post(&s.addr, "/notes", &[0xFF, 0xFE]);
    assert!(r.starts_with("HTTP/1.1 400") && r.ends_with(r#"{"error":"body is not UTF-8"}"#), "{r}");
    let r = post(&s.addr, "/notes", &[0xFF]);
    assert!(r.starts_with("HTTP/1.1 400"), "{r}");
    let mut c = TcpStream::connect(&s.addr).unwrap();
    c.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    c.write_all(b"POST /notes HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Length: 3\r\n\r\n\xE2\x82").unwrap();
    std::thread::sleep(Duration::from_millis(200));
    c.write_all(b"\xAC").unwrap();
    let mut out = vec![];
    c.read_to_end(&mut out).unwrap();
    let r = String::from_utf8_lossy(&out);
    assert!(r.starts_with("HTTP/1.1 201") && r.ends_with("{\"id\":1,\"text\":\"\u{20ac}\"}"), "{r}");
    let log = s.log.clone();
    let (ok, err) = s.stop();
    assert!(ok, "{err}");
    let t = std::fs::read_to_string(&log).unwrap();
    assert!(t.contains(r#""body":[255,254]"#) && t.contains(r#""body":[255]"#) && t.contains(r#""body":[226,130,172]"#), "D holds the bytes received");
    replays(NOTES, &log);
    let _ = std::fs::remove_dir_all(&d);
}

/// HC-21 with HC-24: the id of a request is the seq of its entry, so the
/// program answers a refused request from `Refused(seq, cause)`.
#[test]
fn refused_request_gets_the_programs_status() {
    let d = dir("refused");
    let s = serve(ECHO, &d);
    let r = get(&s.addr, "/trap");
    assert!(r.starts_with("HTTP/1.1 500 Internal Server Error"), "{r}");
    let mut c = TcpStream::connect(&s.addr).unwrap();
    c.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    c.write_all(b"POST /echo HTTP/1.1\r\nHost: x\r\nContent-Length: 3\r\n\r\n\x00\xFF\n").unwrap();
    let (st, _, body) = read_response(&mut c, &mut vec![], false);
    assert_eq!((st, body), (200, vec![0, 255, 10]), "the program echoes the bytes received");
    drop(c);
    let log = s.log.clone();
    let (ok, err) = s.stop();
    assert!(ok && err.contains("refused (trap)") && err.contains("1 refused"), "{err}");
    let es = entries(&log);
    let k = es.iter().find(|e| e.event.contains("\"target\":\"/trap\"")).unwrap().seq;
    assert!(es[k as usize + 1].event == format!(r#"{{"payload":{{"cause":"trap","seq":{k}}},"type":"host.Refused"}}"#));
    replays(ECHO, &log);
    let _ = std::fs::remove_dir_all(&d);
}

/// HC-26: a response that fails the checks fails and leaves the request
/// open, so the program can answer again; a second answer fails.
#[test]
fn invalid_response_fails_and_the_id_stays_open() {
    let d = dir("invalid");
    let s = serve(ECHO, &d);
    let r = get(&s.addr, "/twice");
    assert!(r.starts_with("HTTP/1.1 200") && r.ends_with("ok"), "{r}");
    let r = get(&s.addr, "/late");
    assert!(r.starts_with("HTTP/1.1 200") && r.ends_with("one"), "{r}");
    let log = s.log.clone();
    let (ok, err) = s.stop();
    assert!(ok, "{err}");
    let t = std::fs::read_to_string(&log).unwrap();
    assert!(t.contains(r#""err":"status 99 is outside 200 to 599","req":"#), "{t}");
    assert!(t.contains(r#""err":"no request "#), "the second answer fails: {t}");
    replays(ECHO, &log);
    let _ = std::fs::remove_dir_all(&d);
}

/// The raw bytes of one response read from a connection (`head`: the
/// response to HEAD, so the body is not sent).
fn read_raw(s: &mut TcpStream, buf: &mut Vec<u8>, head: bool) -> Vec<u8> {
    let mut b = [0u8; 65536];
    loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let h = String::from_utf8_lossy(&buf[..i]).into_owned();
            let n: usize = if head { 0 } else { h.split("\r\n").find_map(|l| l.strip_prefix("Content-Length: ")).map_or(0, |v| v.parse().unwrap()) };
            if buf.len() >= i + 4 + n {
                return buf.drain(..i + 4 + n).collect();
            }
        }
        let k = s.read(&mut b).unwrap();
        assert!(k > 0, "closed before a whole response: {:?}", String::from_utf8_lossy(buf));
        buf.extend_from_slice(&b[..k]);
    }
}

/// Whether the host closed the connection: EOF at once, or nothing within
/// 300 ms (a kept-alive connection waits IDLE_MS before it closes).
fn closed(s: &mut TcpStream) -> bool {
    s.set_read_timeout(Some(Duration::from_millis(300))).unwrap();
    let mut b = [0u8; 16];
    match s.read(&mut b) {
        Ok(0) => true,
        Ok(n) => panic!("bytes after the last response: {:?}", String::from_utf8_lossy(&b[..n])),
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock || e.kind() == std::io::ErrorKind::TimedOut => false,
        Err(_) => true,
    }
}

/// One request: its head, and its body sent after the interim
/// `100 Continue` when `expect`, else in the same write.
struct Req {
    head: &'static [u8],
    body: &'static [u8],
    expect: bool,
}

fn req(head: &'static [u8]) -> Req {
    Req { head, body: b"", expect: false }
}

const CONTINUE: &[u8] = b"HTTP/1.1 100 Continue\r\n\r\n";

/// HC-26: send each connection's requests one at a time and record, per
/// request, the final response's bytes and whether the connection closed
/// after it; then make every response again from D alone, with the
/// framing read from the request's entry (`http::framing`) and the
/// `HttpRespond` replay re-derives, and require the same bytes and the
/// same closes. Returns D's HttpRequest and HttpMalformed payloads.
fn check_function_of_d(program: &str, name: &str, conns: Vec<Vec<Req>>) -> Vec<serde_json::Value> {
    let d = dir(name);
    let s = serve(program, &d);
    let mut got: Vec<(Vec<u8>, bool)> = vec![];
    for conn in &conns {
        let mut c = TcpStream::connect(&s.addr).unwrap();
        c.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
        let mut buf = vec![];
        for (i, r) in conn.iter().enumerate() {
            if r.expect {
                c.write_all(r.head).unwrap();
                while buf.len() < CONTINUE.len() {
                    let mut b = [0u8; 64];
                    let k = c.read(&mut b).unwrap();
                    assert!(k > 0, "closed before 100 Continue");
                    buf.extend_from_slice(&b[..k]);
                }
                let interim: Vec<u8> = buf.drain(..CONTINUE.len()).collect();
                assert_eq!(String::from_utf8_lossy(&interim), String::from_utf8_lossy(CONTINUE), "the interim response comes before the body is sent");
                c.write_all(r.body).unwrap();
            } else {
                c.write_all(&[r.head, r.body].concat()).unwrap();
            }
            let bytes = read_raw(&mut c, &mut buf, r.head.starts_with(b"HEAD "));
            assert!(buf.is_empty(), "bytes after the response: {:?}", String::from_utf8_lossy(&buf));
            let last = i + 1 == conn.len();
            got.push((bytes, last && closed(&mut c)));
            c.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
        }
    }
    let log = s.log.clone();
    let (ok, err) = s.stop();
    assert!(ok && err.contains("validated"), "{err}");
    let text = std::fs::read_to_string(&log).unwrap();
    let outs = replay_outputs(&rce_spec_check::program(&std::fs::read_to_string(app(program)).unwrap()).unwrap(), &text).unwrap();
    let mut answers = vec![];
    let mut inputs = vec![];
    for e in entries(&log) {
        let v: serde_json::Value = serde_json::from_str(&e.event).unwrap();
        let Some((head, close)) = rce_spec_check::adapters::http::framing(&v) else { continue };
        let resp = outs.iter().find(|(q, _)| *q == e.seq).map(|(_, o)| o[0].clone()).unwrap();
        answers.push(rce_spec_check::adapters::http::response_bytes(&resp, head, close).unwrap());
        inputs.push(v["payload"].clone());
    }
    assert_eq!(answers.len(), got.len(), "one entry per request");
    for ((a, closes), (g, was_closed)) in answers.iter().zip(&got) {
        assert_eq!(String::from_utf8_lossy(a), String::from_utf8_lossy(g));
        assert_eq!(closes, was_closed, "D says close = {closes} for {:?}", String::from_utf8_lossy(g));
    }
    replays(program, &log);
    let _ = std::fs::remove_dir_all(&d);
    inputs
}

/// HC-26: the host writes only bytes that are a function of D. Every
/// response, made again from D (the `HttpRespond` replay re-derives and
/// the framing read from the request's entry), equals the bytes the
/// client received, and the connection closed exactly when D says so.
#[test]
fn responses_are_a_function_of_d() {
    check_function_of_d(
        NOTES,
        "fn-of-d",
        vec![
            vec![req(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")],
            vec![Req { head: b"POST /notes HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Length: 8\r\n\r\n", body: b"buy milk", expect: false }],
            vec![req(b"HEAD /notes HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")],
            vec![req(b"DELETE /notes/1 HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")],
            vec![req(b"PUT /notes HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")],
            vec![req(b"GET /x HTTP/1.1\nHost: x\n\n")],
        ],
    );
}

/// HC-24, HC-26: the version is in D. The same GET over HTTP/1.0 and over
/// HTTP/1.1 differs in its entry only by `version` and id, and gets
/// `Connection: close` and a close for 1.0, a kept-alive connection for
/// 1.1; HTTP/1.0 with `keep-alive` and HTTP/1.1 each answer two requests
/// on one connection. Every byte and every close is made again from D.
#[test]
fn http_1_0_and_keep_alive_are_framed_from_d() {
    let inputs = check_function_of_d(
        NOTES,
        "version",
        vec![
            vec![req(b"GET /health HTTP/1.0\r\nHost: x\r\n\r\n")],
            vec![req(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")],
            vec![req(b"GET /health HTTP/1.0\r\nConnection: keep-alive\r\n\r\n"), req(b"GET /notes HTTP/1.0\r\nConnection: keep-alive\r\n\r\n")],
            vec![Req { head: b"POST /notes HTTP/1.1\r\nHost: x\r\nContent-Length: 8\r\n\r\n", body: b"buy milk", expect: false }, req(b"GET /notes/1 HTTP/1.1\r\nHost: x\r\n\r\n")],
            vec![req(b"GET /health HTTP/1.0\r\nConnection: keep-alive\r\n\r\n"), req(b"GET /health HTTP/1.0\r\n\r\n")],
        ],
    );
    let (mut a, mut b) = (inputs[0].clone(), inputs[1].clone());
    assert_eq!((a["version"].as_str(), b["version"].as_str()), (Some("HTTP/1.0"), Some("HTTP/1.1")));
    for x in [&mut a, &mut b] {
        x.as_object_mut().unwrap().remove("id");
        x.as_object_mut().unwrap().remove("version");
    }
    assert_eq!(a, b, "the two entries differ only by id and version");
}

/// HC-24, HC-26: `Expect: 100-continue` gets the interim response once
/// the head is accepted, then the body is read and the final response is
/// made from D. A client that closes after the interim response, before
/// its body, leaves nothing in D: the interim response is transport.
#[test]
fn expect_continue_is_transport() {
    let inputs = check_function_of_d(
        NOTES,
        "continue",
        vec![
            vec![Req { head: b"POST /notes HTTP/1.1\r\nHost: x\r\nExpect: 100-continue\r\nContent-Length: 4\r\n\r\n", body: b"eggs", expect: true }, req(b"GET /notes HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")],
            vec![Req { head: b"POST /notes HTTP/1.1\r\nHost: x\r\nExpect: 100-continue\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n", body: b"4\r\nmilk\r\n0\r\n\r\n", expect: true }],
        ],
    );
    assert_eq!(inputs.len(), 3);
    assert_eq!(inputs[0]["body"], serde_json::json!([101, 103, 103, 115]));
    // the interim response for a request whose body never comes
    let d = dir("continue-gone");
    let s = serve(NOTES, &d);
    let mut c = TcpStream::connect(&s.addr).unwrap();
    c.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    c.write_all(b"POST /notes HTTP/1.1\r\nHost: x\r\nExpect: 100-continue\r\nContent-Length: 4\r\n\r\n").unwrap();
    let mut b = vec![0u8; CONTINUE.len()];
    c.read_exact(&mut b).unwrap();
    assert_eq!(b, CONTINUE);
    drop(c);
    std::thread::sleep(Duration::from_millis(200));
    let log = s.log.clone();
    let (ok, err) = s.stop();
    assert!(ok, "{err}");
    let t = std::fs::read_to_string(&log).unwrap();
    assert!(!t.contains("\"type\":\"HttpRequest\"") && !t.contains("\"type\":\"HttpMalformed\""), "a request that never completed is not in D");
    let _ = std::fs::remove_dir_all(&d);
}

/// 16 clients × 50 requests at once; every answer is right and D replays.
#[test]
fn parallel_load_replays_exactly() {
    let d = dir("load");
    let s = serve(NOTES, &d);
    let hs: Vec<_> = (0..16)
        .map(|t| {
            let addr = s.addr.clone();
            std::thread::spawn(move || {
                let mut c = TcpStream::connect(&addr).unwrap();
                c.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
                let mut buf = vec![];
                let mut created = 0;
                for i in 0..50 {
                    if i % 2 == 0 {
                        let body = format!("note {t}-{i}");
                        c.write_all(format!("POST /notes HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes()).unwrap();
                        let (st, _, b) = read_response(&mut c, &mut buf, false);
                        assert_eq!(st, 201);
                        assert!(String::from_utf8_lossy(&b).contains(&body));
                        created += 1;
                    } else {
                        c.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
                        assert_eq!(read_response(&mut c, &mut buf, false).0, 200);
                    }
                }
                created
            })
        })
        .collect();
    let created: usize = hs.into_iter().map(|h| h.join().unwrap()).sum();
    assert_eq!(created, 400);
    let r = get(&s.addr, "/notes");
    assert_eq!(r.matches("\"id\":").count(), 400, "every note is in the state");
    let log = s.log.clone();
    let (ok, err) = s.stop();
    assert!(ok && err.contains("validated"), "{err}");
    let t = std::fs::read_to_string(&log).unwrap();
    assert_eq!(t.matches("\"type\":\"HttpRequest\"").count(), 801);
    replays(NOTES, &log);
    let _ = std::fs::remove_dir_all(&d);
}

/// SIGTERM stops accepting, lets an open request finish, takes the final
/// barrier and validates the run.
#[test]
fn sigterm_drains_and_validates() {
    let d = dir("term");
    let mut s = serve(NOTES, &d);
    let mut c = TcpStream::connect(&s.addr).unwrap();
    c.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    c.write_all(b"POST /notes HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Length: 4\r\n\r\nmi").unwrap();
    std::thread::sleep(Duration::from_millis(200));
    // SAFETY: a signal to our own child process.
    unsafe { libc::kill(s.child.id() as i32, libc::SIGTERM) };
    std::thread::sleep(Duration::from_millis(300));
    c.write_all(b"lk").unwrap();
    let mut out = vec![];
    c.read_to_end(&mut out).unwrap();
    let r = String::from_utf8_lossy(&out);
    assert!(r.starts_with("HTTP/1.1 201") && r.ends_with(r#"{"id":1,"text":"milk"}"#), "{r}");
    let st = s.child.wait().unwrap();
    std::thread::sleep(Duration::from_millis(50));
    let err = s.err.lock().unwrap().clone();
    assert!(st.success() && err.contains("validated"), "{err}");
    replays(NOTES, &s.log);
    let _ = std::fs::remove_dir_all(&d);
}

/// HC-22 with HC-24: the listener is a grant the host binds again at start.
#[test]
fn restart_binds_again() {
    let d = dir("rebind");
    let log = d.join("d.log");
    let addr = free_addr();
    let s = serve_at(NOTES, &log, &addr);
    assert!(post(&s.addr, "/notes", b"milk").starts_with("HTTP/1.1 201"));
    let (ok, err) = s.stop();
    assert!(ok, "{err}");
    let s = serve_at(NOTES, &log, &addr);
    let r = get(&s.addr, "/notes/1");
    assert!(r.starts_with("HTTP/1.1 200") && r.ends_with(r#"{"id":1,"text":"milk"}"#), "{r}");
    let (ok, err) = s.stop();
    assert!(ok, "{err}");
    let t = std::fs::read_to_string(&log).unwrap();
    assert_eq!(t.matches("\"type\":\"host.HttpListen\"").count(), 2);
    assert_eq!(t.matches("\"type\":\"host.Restarted\"").count(), 1);
    replays(NOTES, &log);
    let _ = std::fs::remove_dir_all(&d);
}

/// The 16 ms Tick capped the net service at 62.5 requests a second. The
/// HTTP host admits each request when it arrives; the bound here is far
/// below what it does (see the ledger) and far above the old cap.
#[test]
fn sequential_requests_are_not_paced_by_frames() {
    let d = dir("rate");
    let s = serve(NOTES, &d);
    let t0 = Instant::now();
    for _ in 0..300 {
        assert!(get(&s.addr, "/health").starts_with("HTTP/1.1 200"));
    }
    let rate = 300.0 / t0.elapsed().as_secs_f64();
    eprintln!("{rate:.0} sequential requests a second");
    assert!(rate > 100.0, "{rate:.0} requests a second");
    let (ok, err) = s.stop();
    assert!(ok, "{err}");
    let _ = std::fs::remove_dir_all(&d);
}

/// W-HOST-2 and W-HOST-3 (HC-24, HC-10a): a module served with `--http`
/// declares the HTTP inputs, the response and its result with exactly the
/// shapes of HC-24, the body as bytes; otherwise the host refuses to start
/// and appends nothing.
#[test]
fn http_declarations_are_checked_at_start() {
    use rce_spec_check::host::{Host, MemStore};
    let src = std::fs::read_to_string(app(ECHO)).unwrap();
    // the result of starting, and the number of entries appended
    let start = |s: &str| -> (Result<(), String>, usize) {
        let m = rce_spec_check::parse::parse(s).unwrap();
        let server = rce_spec_check::adapters::http::Server::bind("127.0.0.1:0");
        let mut st = MemStore::default();
        let r = Host::start_with(&m, Box::new(&mut st), vec![Box::new(server.responder())]).map(|_| ());
        (r, st.lines.len())
    };
    let (ok, n) = start(&src);
    assert!(ok.is_ok() && n > 0, "{ok:?}");
    let shapes = [
        ("event HttpMalformed(id: u64, status: u64, why: text)", "event HttpMalformed(id: u64, status: text, why: text)", "W-HOST-2"),
        ("event HttpRespondDone(req: u64, status: text, data: u64?, err: text)", "event HttpRespondDone(req: u64, status: text, data: text?, err: text)", "W-HOST-2"),
        ("    names: List[text]\n    values: List[text]\n    body: List[i64]\n", "    names: List[text]\n    body: List[i64]\n", "W-HOST-2"),
        ("values: List[text], body: List[i64])", "values: List[text], body: text)", "W-HOST-3"),
    ];
    for (from, to, rule) in shapes {
        assert!(src.contains(from), "{from}");
        let (r, n) = start(&src.replace(from, to));
        let e = r.err().unwrap_or_else(|| panic!("{to} was accepted"));
        assert!(e.contains(rule) && n == 0, "{to}: {e}");
    }
}
