//! MCP Streamable HTTP transport (protocol 2025-11-25, "Transports"), on
//! std sockets: no async runtime, no C.
//!
//! Server ([`HttpServer`]):
//! - one endpoint (default `/mcp`), bound to loopback only;
//! - every request needs `Authorization: Bearer <token>` (401 otherwise);
//! - an `Origin` header, when present, must be in the allow list (403);
//! - POST carries one JSON-RPC message; `Accept` must list both
//!   `application/json` and `text/event-stream` (406); requests get either
//!   one JSON body or (with [`HttpConfig::sse`]) an SSE stream holding the
//!   response; notifications and responses get 202;
//! - `initialize` opens a session (`MCP-Session-Id`); later requests must
//!   carry it (400 without, 404 for an unknown or ended one); a
//!   `MCP-Protocol-Version` header must be a supported version (400);
//! - GET returns 405 (no server-initiated stream); DELETE ends the session.
//!
//! Each connection gets its own thread (one request per connection,
//! `Connection: close`), so a slow or idle client does not hold up others.
//! The JSON-RPC dispatch runs on the serving thread, which owns the
//! [`Server`] (connections hand it messages over a channel): the workspace,
//! checker and ledger stay strictly ordered, and nothing else waits on it.
//!
//! Client ([`HttpClient`]): POSTs with the headers above, follows the
//! session id and negotiated version, reads JSON or SSE replies.
use crate::{PROTOCOL_VERSIONS, Server};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const MAX_HEAD: usize = 64 * 1024;
const MAX_BODY: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct HttpConfig {
    pub path: String,
    /// Bearer token every request must present.
    pub token: String,
    /// Allowed `Origin` values (requests without `Origin` are allowed:
    /// they are not from a browser page).
    pub allowed_origins: Vec<String>,
    /// Answer requests with an SSE stream instead of one JSON body.
    pub sse: bool,
}

impl HttpConfig {
    pub fn new(token: impl Into<String>) -> Self {
        HttpConfig {
            path: "/mcp".into(),
            token: token.into(),
            allowed_origins: vec![],
            sse: false,
        }
    }
}

/// A random bearer token (32 bytes, hex).
pub fn new_token() -> String {
    let mut b = [0u8; 32];
    getrandom::fill(&mut b).expect("os randomness");
    hex::encode(b)
}

fn session_id() -> String {
    let mut b = [0u8; 16];
    getrandom::fill(&mut b).expect("os randomness");
    hex::encode(b)
}

/// Constant-time byte comparison.
fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[derive(Debug, Default)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

fn bad(m: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string())
}

/// Read one HTTP/1.1 request head and its `Content-Length` body.
pub fn read_request(r: &mut impl BufRead) -> std::io::Result<Request> {
    let mut head = vec![];
    loop {
        let n = r.read_until(b'\n', &mut head)?;
        if n == 0 {
            return Err(bad("connection closed"));
        }
        if head.len() > MAX_HEAD {
            return Err(bad("header too large"));
        }
        if head.ends_with(b"\r\n\r\n") || head.ends_with(b"\n\n") {
            break;
        }
    }
    let text = String::from_utf8(head).map_err(|_| bad("header not utf-8"))?;
    let mut lines = text.lines();
    let first = lines.next().unwrap_or("");
    let mut p = first.split(' ');
    let method = p.next().unwrap_or("").to_string();
    let path = p.next().unwrap_or("").to_string();
    let mut headers = vec![];
    for l in lines.filter(|l| !l.is_empty()) {
        let (k, v) = l.split_once(':').ok_or_else(|| bad("bad header"))?;
        headers.push((k.trim().to_string(), v.trim().to_string()));
    }
    let mut req = Request {
        method,
        path,
        headers,
        body: vec![],
    };
    if req.header("transfer-encoding").is_some() {
        return Err(bad("chunked bodies are not supported"));
    }
    let len: usize = req
        .header("content-length")
        .map(|v| v.parse().map_err(|_| bad("bad content-length")))
        .transpose()?
        .unwrap_or(0);
    if len > MAX_BODY {
        return Err(bad("body too large"));
    }
    req.body.resize(len, 0);
    r.read_exact(&mut req.body)?;
    Ok(req)
}

pub struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Response {
    fn new(status: u16) -> Self {
        Response {
            status,
            headers: vec![],
            body: vec![],
        }
    }
    fn json(status: u16, v: &Value) -> Self {
        let mut r = Response::new(status);
        r.headers
            .push(("Content-Type".into(), "application/json".into()));
        r.body = v.to_string().into_bytes();
        r
    }
    /// An HTTP error with a JSON-RPC error body that has no id.
    fn error(status: u16, msg: &str) -> Self {
        Response::json(
            status,
            &json!({"jsonrpc":"2.0","error":{"code":-32600,"message":msg}}),
        )
    }
    fn header(mut self, k: &str, v: &str) -> Self {
        self.headers.push((k.into(), v.into()));
        self
    }
    pub fn write_to(&self, w: &mut impl Write) -> std::io::Result<()> {
        let reason = match self.status {
            200 => "OK",
            202 => "Accepted",
            204 => "No Content",
            400 => "Bad Request",
            401 => "Unauthorized",
            403 => "Forbidden",
            404 => "Not Found",
            405 => "Method Not Allowed",
            406 => "Not Acceptable",
            415 => "Unsupported Media Type",
            _ => "Error",
        };
        let mut h = format!("HTTP/1.1 {} {reason}\r\n", self.status);
        for (k, v) in &self.headers {
            h.push_str(&format!("{k}: {v}\r\n"));
        }
        h.push_str(&format!(
            "Content-Length: {}\r\nConnection: close\r\n\r\n",
            self.body.len()
        ));
        w.write_all(h.as_bytes())?;
        w.write_all(&self.body)?;
        w.flush()
    }
}

/// A message for the dispatching thread and where to send its reply.
pub type Job = (Value, std::sync::mpsc::Sender<Option<Value>>);
pub type Dispatch = std::sync::mpsc::Sender<Job>;

pub struct HttpServer {
    listener: TcpListener,
    pub cfg: HttpConfig,
    /// Session id -> negotiated protocol version.
    sessions: Mutex<HashMap<String, String>>,
    /// Requests answered, by status (for the IDE and tests).
    pub stats: Mutex<HashMap<u16, u64>>,
}

impl HttpServer {
    /// Bind to a loopback address (`127.0.0.1:0` picks a port). Other
    /// addresses are an error: this server is for the local IDE and agents.
    pub fn bind(addr: &str, cfg: HttpConfig) -> std::io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        if !listener.local_addr()?.ip().is_loopback() {
            return Err(bad("rce-mcp over HTTP binds to loopback only"));
        }
        if cfg.token.len() < 16 {
            return Err(bad("bearer token too short"));
        }
        Ok(HttpServer {
            listener,
            cfg,
            sessions: Mutex::default(),
            stats: Mutex::default(),
        })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.listener.local_addr().expect("bound")
    }

    pub fn url(&self) -> String {
        format!("http://{}{}", self.local_addr(), self.cfg.path)
    }

    /// Serve until `stop` is set. Accepting and connection I/O run on
    /// their own threads; this thread owns `server` and dispatches.
    pub fn serve(&self, server: &mut Server, stop: Arc<AtomicBool>) -> std::io::Result<()> {
        self.listener.set_nonblocking(true)?;
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        std::thread::scope(|sc| {
            let acceptor = sc.spawn(|| {
                while !stop.load(Ordering::SeqCst) {
                    match self.listener.accept() {
                        Ok((s, _)) => {
                            let tx = tx.clone();
                            sc.spawn(move || {
                                let _ = s.set_nonblocking(false);
                                let _ = s.set_read_timeout(Some(Duration::from_secs(30)));
                                let _ = self.connection(&tx, s);
                            });
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(5))
                        }
                        Err(e) => return Err(e),
                    }
                }
                Ok(())
            });
            while !acceptor.is_finished() {
                if let Ok((msg, back)) = rx.recv_timeout(Duration::from_millis(20)) {
                    let _ = back.send(server.handle(&msg));
                }
            }
            acceptor.join().unwrap_or(Ok(()))
        })
    }

    fn connection(&self, server: &Dispatch, s: TcpStream) -> std::io::Result<()> {
        let mut r = BufReader::new(s.try_clone()?);
        let resp = match read_request(&mut r) {
            Ok(req) => self.handle(server, &req),
            Err(e) => Response::error(400, &e.to_string()),
        };
        *self
            .stats
            .lock()
            .expect("stats")
            .entry(resp.status)
            .or_default() += 1;
        let mut w = s;
        resp.write_to(&mut w)
    }

    /// Handle one parsed request (public for tests without sockets).
    pub fn handle(&self, server: &Dispatch, req: &Request) -> Response {
        if req.path.split('?').next() != Some(self.cfg.path.as_str()) {
            return Response::error(404, "no such endpoint");
        }
        if let Some(o) = req.header("origin")
            && !self.cfg.allowed_origins.iter().any(|a| a == o)
        {
            return Response::error(403, "origin not allowed");
        }
        let auth = req.header("authorization").unwrap_or("");
        let tok = auth.strip_prefix("Bearer ").unwrap_or("");
        if !same(tok.as_bytes(), self.cfg.token.as_bytes()) {
            return Response::error(401, "missing or bad bearer token")
                .header("WWW-Authenticate", "Bearer");
        }
        let sid = req.header("mcp-session-id").map(str::to_string);
        match req.method.as_str() {
            "GET" => {
                return Response::error(405, "no server-initiated stream")
                    .header("Allow", "POST, DELETE");
            }
            "DELETE" => {
                return match sid {
                    None => Response::error(400, "MCP-Session-Id required"),
                    Some(s) if self.sessions.lock().expect("sessions").remove(&s).is_some() => {
                        Response::new(204)
                    }
                    Some(_) => Response::error(404, "unknown session"),
                };
            }
            "POST" => {}
            _ => {
                return Response::error(405, "method not allowed").header("Allow", "POST, DELETE");
            }
        }
        let accept = req.header("accept").unwrap_or("");
        if !(accept.contains("application/json") && accept.contains("text/event-stream")) {
            return Response::error(
                406,
                "Accept must list application/json and text/event-stream",
            );
        }
        if !req
            .header("content-type")
            .unwrap_or("")
            .starts_with("application/json")
        {
            return Response::error(415, "Content-Type must be application/json");
        }
        let Ok(msg) = serde_json::from_slice::<Value>(&req.body) else {
            return Response::json(
                400,
                &json!({"jsonrpc":"2.0","error":{"code":-32700,"message":"parse error"}}),
            );
        };
        if !msg.is_object() {
            return Response::error(400, "one JSON-RPC message per POST (no batches)");
        }
        let is_init = msg["method"] == json!("initialize") && msg.get("id").is_some();
        if !is_init {
            match &sid {
                None => return Response::error(400, "MCP-Session-Id required"),
                Some(s) if !self.sessions.lock().expect("sessions").contains_key(s) => {
                    return Response::error(404, "unknown or ended session");
                }
                _ => {}
            }
            if let Some(v) = req.header("mcp-protocol-version")
                && !PROTOCOL_VERSIONS.contains(&v)
            {
                return Response::error(400, "unsupported MCP-Protocol-Version");
            }
        }
        let is_request = msg.get("method").is_some() && msg.get("id").is_some();
        let (btx, brx) = std::sync::mpsc::channel();
        if server.send((msg.clone(), btx)).is_err() {
            return Response::error(503, "server stopping");
        }
        let Ok(reply) = brx.recv() else {
            return Response::error(503, "server stopping");
        };
        if !is_request {
            return Response::new(202);
        }
        let reply = reply.unwrap_or(Value::Null);
        let mut resp = if self.cfg.sse {
            // Prime event (id, empty data), then the response, then end.
            let eid = format!("{}-{}", sid.as_deref().unwrap_or("init"), msg["id"]);
            let mut r = Response::new(200);
            r.headers
                .push(("Content-Type".into(), "text/event-stream".into()));
            r.headers.push(("Cache-Control".into(), "no-cache".into()));
            r.body =
                format!("id: {eid}-0\ndata:\n\nid: {eid}-1\nevent: message\ndata: {reply}\n\n")
                    .into_bytes();
            r
        } else {
            Response::json(200, &reply)
        };
        if is_init {
            let id = session_id();
            let v = reply["result"]["protocolVersion"]
                .as_str()
                .unwrap_or(PROTOCOL_VERSIONS[0])
                .to_string();
            self.sessions
                .lock()
                .expect("sessions")
                .insert(id.clone(), v);
            resp = resp.header("MCP-Session-Id", &id);
        }
        resp
    }

    pub fn sessions(&self) -> usize {
        self.sessions.lock().expect("sessions").len()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum HttpError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("url: {0}")]
    Url(String),
    #[error("HTTP {0}: {1}")]
    Status(u16, String),
    #[error("bad reply: {0}")]
    Reply(String),
}

/// A Streamable HTTP MCP client (plain HTTP to a local server).
pub struct HttpClient {
    addr: String,
    host: String,
    path: String,
    token: String,
    pub session: Option<String>,
    pub protocol_version: Option<String>,
    next: u64,
    pub bytes_out: u64,
    pub bytes_in: u64,
}

pub struct Reply {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Reply {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

impl HttpClient {
    pub fn new(url: &str, token: &str) -> Result<Self, HttpError> {
        let rest = url
            .strip_prefix("http://")
            .ok_or_else(|| HttpError::Url("only http:// (local servers)".into()))?;
        let (hostport, path) = rest.split_once('/').unwrap_or((rest, ""));
        Ok(HttpClient {
            addr: hostport.to_string(),
            host: hostport.to_string(),
            path: format!("/{path}"),
            token: token.into(),
            session: None,
            protocol_version: None,
            next: 0,
            bytes_out: 0,
            bytes_in: 0,
        })
    }

    /// Send one raw HTTP request (also for tests of the error paths).
    pub fn raw(
        &mut self,
        method: &str,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> Result<Reply, HttpError> {
        let mut s = TcpStream::connect(&self.addr)?;
        s.set_read_timeout(Some(Duration::from_secs(60)))?;
        let mut h = format!(
            "{method} {} HTTP/1.1\r\nHost: {}\r\nContent-Length: {}\r\nConnection: close\r\n",
            self.path,
            self.host,
            body.len()
        );
        for (k, v) in headers {
            h.push_str(&format!("{k}: {v}\r\n"));
        }
        h.push_str("\r\n");
        s.write_all(h.as_bytes())?;
        s.write_all(body)?;
        self.bytes_out += (h.len() + body.len()) as u64;
        let mut buf = vec![];
        s.read_to_end(&mut buf)?;
        self.bytes_in += buf.len() as u64;
        let end = buf
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or_else(|| HttpError::Reply("no header end".into()))?;
        let head = String::from_utf8_lossy(&buf[..end]).to_string();
        let mut lines = head.lines();
        let status = lines
            .next()
            .and_then(|l| l.split(' ').nth(1))
            .and_then(|c| c.parse().ok())
            .ok_or_else(|| HttpError::Reply("bad status line".into()))?;
        let headers = lines
            .filter_map(|l| l.split_once(':'))
            .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
            .collect();
        Ok(Reply {
            status,
            headers,
            body: buf[end + 4..].to_vec(),
        })
    }

    fn post(&mut self, msg: &Value) -> Result<Reply, HttpError> {
        let auth = format!("Bearer {}", self.token);
        let mut h: Vec<(&str, &str)> = vec![
            ("Authorization", &auth),
            ("Content-Type", "application/json"),
            ("Accept", "application/json, text/event-stream"),
        ];
        let (sid, ver) = (self.session.clone(), self.protocol_version.clone());
        if let Some(s) = &sid {
            h.push(("MCP-Session-Id", s));
        }
        if let Some(v) = &ver {
            h.push(("MCP-Protocol-Version", v));
        }
        self.raw("POST", &h, msg.to_string().as_bytes())
    }

    /// The JSON-RPC message in a JSON or SSE reply with this id.
    fn message(r: &Reply, id: &Value) -> Result<Value, HttpError> {
        if r.status != 200 {
            return Err(HttpError::Status(
                r.status,
                String::from_utf8_lossy(&r.body).to_string(),
            ));
        }
        let ct = r.header("content-type").unwrap_or("");
        if ct.starts_with("text/event-stream") {
            let text = String::from_utf8_lossy(&r.body);
            for ev in text.split("\n\n") {
                let data: Vec<&str> = ev
                    .lines()
                    .filter_map(|l| l.strip_prefix("data:"))
                    .map(str::trim_start)
                    .collect();
                if data.is_empty() || data.iter().all(|d| d.is_empty()) {
                    continue;
                }
                let v: Value = serde_json::from_str(&data.join("\n"))
                    .map_err(|e| HttpError::Reply(e.to_string()))?;
                if &v["id"] == id {
                    return Ok(v);
                }
            }
            Err(HttpError::Reply("no response in the event stream".into()))
        } else {
            serde_json::from_slice(&r.body).map_err(|e| HttpError::Reply(e.to_string()))
        }
    }

    /// A JSON-RPC request; returns the whole response message.
    pub fn request(&mut self, method: &str, params: Value) -> Result<Value, HttpError> {
        self.next += 1;
        let id = json!(self.next);
        let m = json!({"jsonrpc":"2.0","id":id,"method":method,"params":params});
        let r = self.post(&m)?;
        if method == "initialize"
            && let Some(s) = r.header("mcp-session-id")
        {
            self.session = Some(s.to_string());
        }
        let v = Self::message(&r, &id)?;
        if method == "initialize" {
            self.protocol_version = v["result"]["protocolVersion"].as_str().map(str::to_string);
        }
        Ok(v)
    }

    pub fn notify(&mut self, method: &str, params: Value) -> Result<u16, HttpError> {
        let m = json!({"jsonrpc":"2.0","method":method,"params":params});
        Ok(self.post(&m)?.status)
    }

    /// `initialize` + `notifications/initialized`.
    pub fn initialize(&mut self, client: &str) -> Result<Value, HttpError> {
        let v = self.request(
            "initialize",
            json!({"protocolVersion": PROTOCOL_VERSIONS[0], "capabilities": {}, "clientInfo": {"name": client, "version": "0"}}),
        )?;
        self.notify("notifications/initialized", json!({}))?;
        Ok(v)
    }

    /// End the session (HTTP DELETE).
    pub fn close(&mut self) -> Result<u16, HttpError> {
        let auth = format!("Bearer {}", self.token);
        let sid = self.session.take().unwrap_or_default();
        Ok(self
            .raw(
                "DELETE",
                &[("Authorization", &auth), ("MCP-Session-Id", &sid)],
                b"",
            )?
            .status)
    }
}
