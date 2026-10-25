//! HC-24 to HC-26: the HTTP event host (`rce host serve --http HOST:PORT`).
//! Connections are served concurrently, each request is one logged input
//! (`HttpRequest` or `HttpMalformed`), the program answers with the
//! `HttpRespond` effect, limits and deadlines apply before a byte enters D,
//! and every byte written is a function of D. These tests run the `rce`
//! binary on programs/network/notes.rce and tests/host/http_echo.rce.
use rce_spec_check::host::{Entry, replay_outputs};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const NOTES: &str = "../../programs/network/notes.rce";
const ECHO: &str = "tests/host/http_echo.rce";

fn app(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn free_addr() -> String {
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    format!("127.0.0.1:{port}")
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

/// HC-26: the host writes only bytes that are a function of D. Every
/// response, re-serialized from the outputs replay re-derives, equals the
/// bytes the client received.
#[test]
fn responses_are_a_function_of_d() {
    let d = dir("fn-of-d");
    let s = serve(NOTES, &d);
    let reqs: Vec<Vec<u8>> = vec![
        b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n".to_vec(),
        b"POST /notes HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Length: 8\r\n\r\nbuy milk".to_vec(),
        b"HEAD /notes HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n".to_vec(),
        b"DELETE /notes/1 HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n".to_vec(),
        b"PUT /notes HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n".to_vec(),
        b"GET /x HTTP/1.1\nHost: x\n\n".to_vec(),
    ];
    let mut got = vec![];
    for r in &reqs {
        let mut c = TcpStream::connect(&s.addr).unwrap();
        c.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
        c.write_all(r).unwrap();
        let mut out = vec![];
        c.read_to_end(&mut out).unwrap();
        got.push(out);
    }
    let log = s.log.clone();
    let (ok, err) = s.stop();
    assert!(ok, "{err}");
    let text = std::fs::read_to_string(&log).unwrap();
    let outs = replay_outputs(&rce_spec_check::program(&std::fs::read_to_string(app(NOTES)).unwrap()).unwrap(), &text).unwrap();
    let es = entries(&log);
    let mut answers = vec![];
    for e in &es {
        let v: serde_json::Value = serde_json::from_str(&e.event).unwrap();
        let ty = v["type"].as_str().unwrap();
        if ty != "HttpRequest" && ty != "HttpMalformed" {
            continue;
        }
        let head = v["payload"]["method"] == "HEAD";
        let resp = outs.iter().find(|(q, _)| *q == e.seq).map(|(_, o)| o[0].clone()).unwrap();
        let (bytes, _) = rce_spec_check::adapters::http::response_bytes(&resp, head, true).unwrap();
        answers.push(bytes);
    }
    assert_eq!(answers.len(), got.len());
    for (a, g) in answers.iter().zip(&got) {
        assert_eq!(String::from_utf8_lossy(a), String::from_utf8_lossy(g));
    }
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
