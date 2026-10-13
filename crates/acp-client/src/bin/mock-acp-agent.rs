//! A mock ACP v1 agent for tests. Speaks stdio JSON-RPC, connects to the
//! first stdio MCP server it is given, and acts on directive lines in the
//! prompt text:
//!
//! - `mock:mcp <tool> <json args>`: ask permission (title `mcp__<srv>__<tool>`,
//!   kind `edit`), then call the tool over MCP if allowed.
//! - `mock:ask <title> <kind>`: ask permission for an arbitrary tool, do nothing.
//! - `mock:fs_write <abs path>` / `mock:fs_read <abs path>`: call the client's fs methods.
//! - `mock:terminal`: call `terminal/create`.
//! - `mock:rogue <rel path> <text>`: write a workspace file directly.
//! - `mock:hang`: never answer until `session/cancel`.
//! - `mock:read <abs path>`: read a file directly (a built-in tool that
//!   never asks); reports `read ok <n>` or `read err <errno>`.
//! - `mock:write_abs <abs path> <text>`: write anywhere directly; reports.
//! - `mock:connect <port>`: TCP connect to 127.0.0.1:port; reports.
//! - `mock:fetch <url>`: GET `url` through `HTTP_PROXY` (absolute form,
//!   proxy token from the URL's userinfo); reports `fetch <status> <body>`.
//!
//! An `http` MCP server (Streamable HTTP, `headers` sent as given) is used
//! when it is the first one offered; the agent advertises
//! `mcpCapabilities.http`.
//!
//! Env `MOCK_ACP_VERSION` overrides the protocol version it answers.
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

enum Mcp {
    Stdio(StdioMcp),
    Http(HttpMcp),
}

impl Mcp {
    fn call(&mut self, method: &str, params: Value) -> Value {
        match self {
            Mcp::Stdio(m) => m.call(method, params),
            Mcp::Http(m) => m.call(method, params),
        }
    }
}

struct HttpMcp {
    addr: String,
    path: String,
    headers: Vec<(String, String)>,
    session: Option<String>,
    id: u64,
}

/// Minimal HTTP/1.1 exchange: returns (status, headers, body).
fn http(addr: &str, head: &str, body: &[u8]) -> std::io::Result<(u16, String, String)> {
    use std::io::Read;
    let mut s = std::net::TcpStream::connect(addr)?;
    s.write_all(head.as_bytes())?;
    s.write_all(body)?;
    let mut out = Vec::new();
    s.read_to_end(&mut out)?;
    let t = String::from_utf8_lossy(&out).to_string();
    let (h, b) = t.split_once("\r\n\r\n").unwrap_or((&t, ""));
    let code = h
        .split(' ')
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or(0);
    Ok((code, h.to_string(), b.to_string()))
}

impl HttpMcp {
    fn call(&mut self, method: &str, params: Value) -> Value {
        self.id += 1;
        let m = json!({"jsonrpc":"2.0","id":self.id,"method":method,"params":params}).to_string();
        let mut head = format!(
            "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n",
            self.path,
            self.addr,
            m.len()
        );
        for (k, v) in &self.headers {
            head += &format!("{k}: {v}\r\n");
        }
        if let Some(sid) = &self.session {
            head += &format!("MCP-Session-Id: {sid}\r\nMCP-Protocol-Version: 2025-11-25\r\n");
        }
        head += "\r\n";
        let Ok((_, h, b)) = http(&self.addr, &head, m.as_bytes()) else {
            return json!({"error": "mcp http failed"});
        };
        for l in h.lines() {
            if let Some((k, v)) = l.split_once(':')
                && k.trim().eq_ignore_ascii_case("mcp-session-id")
            {
                self.session = Some(v.trim().to_string());
            }
        }
        serde_json::from_str(&b).unwrap_or_else(|_| json!({"error": b}))
    }
}

fn b64(data: &[u8]) -> String {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut o = String::new();
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            if i <= c.len() {
                o.push(T[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                o.push('=');
            }
        }
    }
    o
}

/// GET `url` through `HTTP_PROXY` (`http://user:token@host:port`).
fn fetch(url: &str) -> String {
    let Ok(p) = std::env::var("HTTP_PROXY") else {
        return "fetch noproxy ".into();
    };
    let rest = p.trim_start_matches("http://");
    let (auth, addr) = rest.rsplit_once('@').unwrap_or(("", rest));
    let host = url
        .split("://")
        .nth(1)
        .and_then(|r| r.split('/').next())
        .unwrap_or("");
    let head = format!(
        "GET {url} HTTP/1.1\r\nHost: {host}\r\nProxy-Authorization: Basic {}\r\nConnection: close\r\n\r\n",
        b64(auth.as_bytes())
    );
    match http(addr, &head, b"") {
        Ok((c, _, b)) => format!("fetch {c} {} ", b.replace(['\r', '\n'], "|")),
        Err(e) => format!("fetch err {} ", e.raw_os_error().unwrap_or(-1)),
    }
}

struct StdioMcp {
    _child: Child,
    w: ChildStdin,
    r: BufReader<ChildStdout>,
    id: u64,
}

impl StdioMcp {
    fn call(&mut self, method: &str, params: Value) -> Value {
        self.id += 1;
        let m = json!({"jsonrpc":"2.0","id":self.id,"method":method,"params":params});
        writeln!(self.w, "{m}").unwrap();
        self.w.flush().unwrap();
        let mut l = String::new();
        loop {
            l.clear();
            if self.r.read_line(&mut l).unwrap_or(0) == 0 {
                return json!({"error": "mcp closed"});
            }
            let v: Value = serde_json::from_str(&l).unwrap_or_default();
            if v["id"] == json!(self.id) {
                return v;
            }
        }
    }
}

struct Agent {
    out: std::io::Stdout,
    lines: std::io::Lines<std::io::StdinLock<'static>>,
    next: u64,
    cwd: String,
    mcp: Option<(String, Mcp)>,
}

impl Agent {
    fn send(&mut self, v: Value) {
        let mut o = self.out.lock();
        writeln!(o, "{v}").unwrap();
        o.flush().unwrap();
    }
    fn update(&mut self, sid: &str, u: Value) {
        self.send(json!({"jsonrpc":"2.0","method":"session/update","params":{"sessionId":sid,"update":u}}));
    }
    /// Request to the client; returns its response (serving nothing else).
    fn ask(&mut self, method: &str, params: Value) -> Value {
        self.next += 1;
        let id = format!("a{}", self.next);
        self.send(json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}));
        while let Some(Ok(l)) = self.lines.next() {
            let v: Value = serde_json::from_str(&l).unwrap_or_default();
            if v["id"] == json!(id) && v.get("method").is_none() {
                return v;
            }
        }
        json!({"error":"closed"})
    }
    fn permit(&mut self, sid: &str, call_id: &str, title: &str, kind: &str) -> bool {
        self.update(sid, json!({"sessionUpdate":"tool_call","toolCallId":call_id,"title":title,"kind":kind,"status":"pending"}));
        let r = self.ask(
            "session/request_permission",
            json!({"sessionId":sid,
            "toolCall":{"toolCallId":call_id,"title":title,"kind":kind},
            "options":[{"optionId":"allow-once","name":"Allow","kind":"allow_once"},
                       {"optionId":"allow-always","name":"Always","kind":"allow_always"},
                       {"optionId":"reject-once","name":"Reject","kind":"reject_once"}]}),
        );
        let ok = r["result"]["outcome"]["optionId"] == json!("allow-once")
            || r["result"]["outcome"]["optionId"] == json!("allow-always");
        let st = if ok { "in_progress" } else { "failed" };
        self.update(
            sid,
            json!({"sessionUpdate":"tool_call_update","toolCallId":call_id,"status":st}),
        );
        ok
    }
}

fn main() {
    let stdin: &'static std::io::Stdin = Box::leak(Box::new(std::io::stdin()));
    let mut a = Agent {
        out: std::io::stdout(),
        lines: stdin.lock().lines(),
        next: 0,
        cwd: String::new(),
        mcp: None,
    };
    let version: Value = std::env::var("MOCK_ACP_VERSION")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map_or(json!(1), |v| json!(v));
    while let Some(Ok(l)) = a.lines.next() {
        let v: Value = match serde_json::from_str(&l) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let id = v["id"].clone();
        let p = v["params"].clone();
        match v["method"].as_str().unwrap_or("") {
            "initialize" => {
                assert_eq!(p["clientCapabilities"]["terminal"], json!(false));
                a.send(
                    json!({"jsonrpc":"2.0","id":id,"result":{"protocolVersion":version,
                    "agentCapabilities":{"loadSession":false,"promptCapabilities":{},"mcpCapabilities":{"http":true,"sse":false}},
                    "agentInfo":{"name":"mock-acp-agent","version":"0.1.0"},"authMethods":[]}}),
                );
            }
            "session/new" => {
                a.cwd = p["cwd"].as_str().unwrap_or("").to_string();
                let servers = p["mcpServers"].as_array().cloned().unwrap_or_default();
                if let Some(s) = servers.first().filter(|s| s["type"] == json!("http")) {
                    let url = s["url"].as_str().unwrap_or("");
                    let rest = url.trim_start_matches("http://");
                    let (addr, path) = rest.split_once('/').unwrap_or((rest, ""));
                    let headers = s["headers"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .map(|h| {
                            (
                                h["name"].as_str().unwrap_or("").to_string(),
                                h["value"].as_str().unwrap_or("").to_string(),
                            )
                        })
                        .collect();
                    let mut m = Mcp::Http(HttpMcp {
                        addr: addr.into(),
                        path: format!("/{path}"),
                        headers,
                        session: None,
                        id: 0,
                    });
                    m.call("initialize", json!({"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"mock","version":"0"}}));
                    a.mcp = Some((s["name"].as_str().unwrap_or("").to_string(), m));
                } else if let Some(s) = servers.first() {
                    let args: Vec<String> = s["args"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect();
                    let mut c = Command::new(s["command"].as_str().unwrap_or(""))
                        .args(&args)
                        .arg("--no-memory")
                        .stdin(Stdio::piped())
                        .stdout(Stdio::piped())
                        .stderr(Stdio::null())
                        .spawn()
                        .expect("spawn mcp");
                    let w = c.stdin.take().unwrap();
                    let r = BufReader::new(c.stdout.take().unwrap());
                    let mut m = Mcp::Stdio(StdioMcp {
                        _child: c,
                        w,
                        r,
                        id: 0,
                    });
                    m.call("initialize", json!({"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"mock","version":"0"}}));
                    a.mcp = Some((s["name"].as_str().unwrap_or("").to_string(), m));
                }
                a.send(json!({"jsonrpc":"2.0","id":id,"result":{"sessionId":"sess_mock_1","servers":servers.len()}}));
            }
            "session/cancel" => {}
            "session/prompt" => {
                let sid = p["sessionId"].as_str().unwrap_or("").to_string();
                let text: String = p["prompt"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|b| b["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                let mut stop = "end_turn";
                let mut n = 0;
                for line in text.lines() {
                    let Some(d) = line.trim().strip_prefix("mock:") else {
                        continue;
                    };
                    n += 1;
                    let cid = format!("call_{n}");
                    let (cmd, rest) = d.split_once(' ').unwrap_or((d, ""));
                    match cmd {
                        "mcp" => {
                            let (tool, args) = rest.split_once(' ').unwrap_or((rest, "{}"));
                            let srv = a.mcp.as_ref().map(|m| m.0.clone()).unwrap_or_default();
                            if a.permit(&sid, &cid, &format!("mcp__{srv}__{tool}"), "edit") {
                                let args: Value = serde_json::from_str(args).unwrap_or_default();
                                let r = a
                                    .mcp
                                    .as_mut()
                                    .map(|m| {
                                        m.1.call(
                                            "tools/call",
                                            json!({"name":tool,"arguments":args}),
                                        )
                                    })
                                    .unwrap_or_default();
                                let st = if r["result"]["isError"] == json!(true) {
                                    "failed"
                                } else {
                                    "completed"
                                };
                                a.update(&sid, json!({"sessionUpdate":"tool_call_update","toolCallId":cid,"status":st}));
                            }
                        }
                        "ask" => {
                            let (t, k) = rest.split_once(' ').unwrap_or((rest, "other"));
                            a.permit(&sid, &cid, t, k);
                        }
                        "fs_write" => {
                            let r = a.ask(
                                "fs/write_text_file",
                                json!({"sessionId":sid,"path":rest,"content":"x"}),
                            );
                            a.update(&sid, json!({"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":format!("fs_write error={} ", r["error"]["code"])}}));
                        }
                        "fs_read" => {
                            let r =
                                a.ask("fs/read_text_file", json!({"sessionId":sid,"path":rest}));
                            a.update(&sid, json!({"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":format!("fs_read error={} ", r["error"]["code"])}}));
                        }
                        "terminal" => {
                            a.ask(
                                "terminal/create",
                                json!({"sessionId":sid,"command":"sh","args":["-c","id"]}),
                            );
                        }
                        "rogue" => {
                            let (path, t) = rest.split_once(' ').unwrap_or((rest, ""));
                            std::fs::write(std::path::Path::new(&a.cwd).join(path), t).unwrap();
                        }
                        "read" | "write_abs" | "connect" => {
                            let r = match cmd {
                                "read" => std::fs::read(rest).map(|b| b.len()),
                                "write_abs" => {
                                    let (path, t) = rest.split_once(' ').unwrap_or((rest, ""));
                                    std::fs::write(path, t).map(|_| t.len())
                                }
                                _ => std::net::TcpStream::connect((
                                    "127.0.0.1",
                                    rest.parse::<u16>().unwrap_or(0),
                                ))
                                .map(|_| 0),
                            };
                            let msg = match r {
                                Ok(n) => format!("{cmd} ok {n} "),
                                Err(e) => format!("{cmd} err {} ", e.raw_os_error().unwrap_or(-1)),
                            };
                            a.update(&sid, json!({"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":msg}}));
                        }
                        "fetch" => {
                            let msg = fetch(rest);
                            a.update(&sid, json!({"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":msg}}));
                        }
                        "hang" => {
                            while let Some(Ok(l)) = a.lines.next() {
                                let v: Value = serde_json::from_str(&l).unwrap_or_default();
                                if v["method"] == json!("session/cancel") {
                                    break;
                                }
                            }
                            stop = "cancelled";
                        }
                        _ => {}
                    }
                }
                a.update(&sid, json!({"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"done"}}));
                a.update(
                    &sid,
                    json!({"sessionUpdate":"usage_update","used":1200,"size":200000}),
                );
                a.send(json!({"jsonrpc":"2.0","id":id,"result":{"stopReason":stop}}));
            }
            m if !id.is_null() => {
                a.send(json!({"jsonrpc":"2.0","id":id,"error":{"code":-32601,"message":format!("unknown {m}")}}));
            }
            _ => {}
        }
    }
}
