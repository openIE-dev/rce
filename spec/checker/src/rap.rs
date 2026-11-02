//! HC-34 to HC-36: an RCE program served as a tool to agent runtimes that
//! speak RAP, the Reactive Agent Protocol of the Infinity project
//! (github.com/hydro-project/infinity, Apache-2.0). This file holds what is
//! a function of the module and of bytes alone: the toolset a program
//! publishes, the check of an invocation's envelope and arguments, the
//! bytes of a tool result, and the HTTP client that POSTs a result to its
//! callback URL. What enters D, and how replay checks it, is in `host.rs`
//! (`Host::rap_invoke`, `Host::rap_steps`, `Host::rap_sent`); the listener
//! and the loop are in `cli.rs`.
//!
//! - HC-34: each event the program accepts from outside is one operation,
//!   named as the event, whose `inputSchema` is the JSON Schema of its
//!   fields with `additionalProperties: false` ([`toolset`]). The toolset
//!   is served at `/.well-known/rap-toolset`; its `endpoint` is
//!   `http://ADDR/invoke`.
//! - HC-35: an invocation needs `id`, `group_id` and an `http://`
//!   `callback_url` to a granted address ([`envelope`]); without them it
//!   cannot be answered and is refused with 400, and nothing enters D.
//!   Everything else (an unknown operation, arguments that do not conform) is
//!   acknowledged and answered by an error result ([`arguments`]).
//! - HC-36: the result is `{call_id, group_id, id, text, type}` as
//!   canonical JSON ([`result_body`]), its text the step's outputs, or
//!   `Error: ...` when the invocation does not conform or its step is kept.
use crate::ast::{Arg, Item, Module, Ty};
use serde_json::{Map, Value, json};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// The discovery path of RAP's toolsets spec (`{endpoint_base}/.well-known/rap-toolset`).
pub const DISCOVERY: &str = "/.well-known/rap-toolset";
/// The invocation endpoint the toolset names.
pub const INVOKE: &str = "/invoke";
/// RAP's best-effort notifications (tool cancellation, thread closure):
/// acknowledged with 200 and not acted on (HC-35).
pub const CANCEL: &str = "/cancel_tool_call";
pub const CLOSE: &str = "/close_thread";
/// The deepest JSON nesting an invocation may have: past it the body is
/// refused before it is parsed, so no body can exhaust the parser's stack.
pub const DEPTH_MAX: usize = 64;
/// The longest pause between two attempts to deliver one result.
pub const RETRY_MAX_MS: u64 = 30_000;
/// The first pause after an attempt that failed (doubled each time).
pub const RETRY_FIRST_MS: u64 = 100;
/// Connecting to a callback URL, and reading its status line.
pub const CONNECT_MS: u64 = 5_000;
pub const ANSWER_MS: u64 = 10_000;

/// One operation of the toolset: an event the program accepts from outside.
#[derive(Clone, Debug)]
pub struct Tool {
    pub name: String,
    pub fields: Vec<(String, Ty)>,
}

/// The events the host delivers itself, which no runtime may invoke: the
/// host inputs and results of HC-10b, the frame tick and the held line.
fn host_named(m: &Module) -> Vec<String> {
    let mut v = crate::host::host_named_events(m);
    v.extend(["Tick", "Input"].map(String::from));
    v
}

/// The operations of a module, in declaration order, and one note for each
/// declared event left out of the toolset, saying why (HC-34).
pub fn tools(m: &Module) -> (Vec<Tool>, Vec<String>) {
    let host = host_named(m);
    let (mut tools, mut notes) = (vec![], vec![]);
    for i in &m.items {
        let Item::Event { name, payload, foreign, .. } = i else { continue };
        if host.contains(name) {
            notes.push(format!("rap: event {name} is delivered by the host, so it is not an operation"));
            continue;
        }
        if foreign.is_some() {
            notes.push(format!("rap: event {name} carries a foreign runtime's answers (extern), so it is not an operation"));
            continue;
        }
        match payload.iter().map(|(_, t)| schema(t, m, 0)).collect::<Result<Vec<_>, _>>() {
            Ok(_) => tools.push(Tool { name: name.clone(), fields: payload.clone() }),
            Err(why) => notes.push(format!("rap: event {name} is not an operation: {why}")),
        }
    }
    (tools, notes)
}

fn int_name(t: &Ty) -> Option<&str> {
    match t {
        Ty::Path(p, g) if p.len() == 1 && g.is_empty() && crate::types::INTS.contains(&p[0].as_str()) => Some(p[0].as_str()),
        _ => None,
    }
}

fn simple(t: &Ty, name: &str) -> bool {
    matches!(t, Ty::Path(p, g) if p.len() == 1 && g.is_empty() && p[0] == name)
}

fn struct_fields<'m>(m: &'m Module, n: &str) -> Option<&'m [crate::ast::Field]> {
    m.items.iter().find_map(|i| match i {
        Item::Struct { name, fields, generics, .. } if name == n && generics.is_empty() => Some(fields.as_slice()),
        _ => None,
    })
}

/// The JSON Schema (draft 2020-12) of a declared type, or why it has none
/// here: integers with their range, `f64` and `f32` as numbers, `bool`,
/// `text`, `List[T]`, `T?` (the value or null) and structs (an object of
/// their fields, every field that is not `?` required, no other property).
pub fn schema(t: &Ty, m: &Module, depth: usize) -> Result<Value, String> {
    if depth > 16 {
        return Err("a type nested deeper than 16 levels".into());
    }
    let t = crate::host::unalias_ty(t, m);
    if let Some(n) = int_name(t) {
        let (lo, hi) = crate::types::int_range(n);
        let num = |x: i128| if x < 0 { json!(x as i64) } else { json!(x as u64) };
        return Ok(json!({"maximum": num(hi), "minimum": num(lo), "type": "integer"}));
    }
    if simple(t, "f64") || simple(t, "f32") {
        return Ok(json!({"type": "number"}));
    }
    if simple(t, "bool") {
        return Ok(json!({"type": "boolean"}));
    }
    if simple(t, "text") {
        return Ok(json!({"type": "string"}));
    }
    match t {
        Ty::Opt(x) => Ok(json!({"anyOf": [schema(x, m, depth + 1)?, {"type": "null"}]})),
        Ty::Path(p, g) if p.len() == 1 && p[0] == "List" && g.len() == 1 => Ok(json!({"items": schema(&g[0], m, depth + 1)?, "type": "array"})),
        Ty::Path(p, g) if p.len() == 1 && g.is_empty() => {
            let fs = struct_fields(m, &p[0]).ok_or_else(|| format!("type {} has no JSON Schema here", p[0]))?;
            let fields: Vec<(String, Ty)> = fs.iter().map(|f| (f.name.clone(), f.ty.clone())).collect();
            object(&fields, m, depth + 1)
        }
        other => Err(format!("type {} has no JSON Schema here", crate::host::ty_name(other))),
    }
}

fn is_opt(t: &Ty, m: &Module) -> bool {
    matches!(crate::host::unalias_ty(t, m), Ty::Opt(_))
}

/// The schema of an object with these fields, `additionalProperties: false`.
fn object(fields: &[(String, Ty)], m: &Module, depth: usize) -> Result<Value, String> {
    let mut props = Map::new();
    let mut req = vec![];
    for (k, t) in fields {
        props.insert(k.clone(), schema(t, m, depth)?);
        if !is_opt(t, m) {
            req.push(Value::String(k.clone()));
        }
    }
    Ok(json!({"additionalProperties": false, "properties": props, "required": req, "type": "object"}))
}

/// The name of a module's toolset: its path, or `program` when it has none.
pub fn name(m: &Module) -> String {
    match &m.path {
        Some(p) if !p.is_empty() => format!("rce-{}", p.join(".")),
        _ => "rce-program".into(),
    }
}

/// The toolset of a module served at `endpoint` (HC-34), as canonical JSON.
/// It is a function of the module and the endpoint, so the host records
/// its digest in `host.RapListen` and replay re-derives it.
pub fn toolset(m: &Module, endpoint: &str) -> Result<String, String> {
    let (tools, _) = tools(m);
    if tools.is_empty() {
        return Err("rap: the program declares no event a runtime can invoke, and a toolset holds at least one tool".into());
    }
    let mut ts = vec![];
    for t in &tools {
        let sig: Vec<String> = t.fields.iter().map(|(k, ty)| format!("{k}: {}", crate::host::ty_name(ty))).collect();
        ts.push(json!({
            "description": format!("Admit the event {}({}) to the RCE program as one input of its durable log. The result's text is the step's outputs as JSON, or starts with \"Error: \" when the arguments do not conform or the step is kept (the state is then unchanged).", t.name, sig.join(", ")),
            "inputSchema": object(&t.fields, m, 0)?,
            "name": t.name,
        }));
    }
    let v = json!({
        "description": format!("The events of the RCE program {}, served by rce host serve rap. Each invocation is one input of the program's durable log, applied once per (group_id, id).", name(m)),
        "endpoint": endpoint,
        "name": name(m),
        "tools": ts,
    });
    Ok(crate::interp::canon_json(&v))
}

/// The SHA-256 of the toolset served at `endpoint`, as `host.RapListen` holds it.
pub fn toolset_digest(m: &Module, endpoint: &str) -> Result<String, String> {
    Ok(crate::interp::d_sha(&toolset(m, endpoint)?))
}

/// A callback URL split: (address `host:port` as a grant names it, host,
/// port, path with query). Only `http://` URLs without user information.
pub fn http_url(u: &str) -> Result<(String, String, u16, String), String> {
    let rest = u.strip_prefix("http://").ok_or_else(|| format!("callback_url {u} is not an http:// URL"))?;
    let rest = rest.split('#').next().unwrap_or("");
    let cut = rest.find(['/', '?']).unwrap_or(rest.len());
    let (auth, path) = rest.split_at(cut);
    if auth.is_empty() || auth.contains('@') || !auth.bytes().all(|b| (33..=126).contains(&b)) || !path.bytes().all(|b| (33..=126).contains(&b)) {
        return Err(format!("callback_url {u} has no plain host and port"));
    }
    let path = if path.is_empty() { "/".to_string() } else if path.starts_with('?') { format!("/{path}") } else { path.to_string() };
    // [v6]:port or host:port, the port 80 when absent
    let (host, port) = match auth.strip_prefix('[') {
        Some(r) => {
            let (h, p) = r.split_once(']').ok_or_else(|| format!("callback_url {u}: unclosed ["))?;
            (format!("[{h}]"), p.strip_prefix(':').unwrap_or(""))
        }
        None => match auth.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), p),
            None => (auth.to_string(), ""),
        },
    };
    let port: u16 = if port.is_empty() { 80 } else { port.parse().map_err(|_| format!("callback_url {u}: port {port}"))? };
    Ok((format!("{host}:{port}"), host, port, path))
}

/// Whether the JSON text nests deeper than [`DEPTH_MAX`] (strings skipped).
fn too_deep(b: &[u8]) -> bool {
    let (mut d, mut s, mut esc) = (0usize, false, false);
    for &c in b {
        if s {
            match (esc, c) {
                (true, _) => esc = false,
                (false, b'\\') => esc = true,
                (false, b'"') => s = false,
                _ => {}
            }
            continue;
        }
        match c {
            b'"' => s = true,
            b'[' | b'{' => {
                d += 1;
                if d > DEPTH_MAX {
                    return true;
                }
            }
            b']' | b'}' => d = d.saturating_sub(1),
            _ => {}
        }
    }
    false
}

/// HC-35: the host's whole value of one invocation, the body of `host.RapInvoke`:
/// the eight fields of RAP's tool invocation, each `null` when absent, other
/// fields left out. `Err` when the invocation cannot be answered: the body
/// is not a JSON object, `id` or `group_id` is not a string, or
/// `callback_url` is not an `http://` URL to an address in `granted`
/// (`callback HOST:PORT`, HC-5). Such an invocation is refused with 400.
pub fn envelope(body: &[u8], granted: &[String]) -> Result<Map<String, Value>, String> {
    if too_deep(body) {
        return Err(format!("the body nests deeper than {DEPTH_MAX} levels"));
    }
    let v: Value = serde_json::from_slice(body).map_err(|e| format!("the body is not JSON: {e}"))?;
    let o = v.as_object().ok_or("the body is not a JSON object")?;
    for k in ["id", "group_id", "callback_url"] {
        match o.get(k) {
            Some(Value::String(s)) if !s.is_empty() => {}
            _ => return Err(format!("{k} is not a non-empty string, so no result could be delivered")),
        }
    }
    let url = o["callback_url"].as_str().unwrap_or("");
    let (addr, ..) = http_url(url)?;
    if !granted.contains(&addr) {
        return Err(format!("callback_url {url} names {addr}, which is not granted (callback HOST:PORT, HC-5)"));
    }
    let mut p = Map::new();
    for k in ["arguments", "call_id", "callback_url", "group_id", "id", "operation", "thread_ancestors", "user_id"] {
        p.insert(k.into(), o.get(k).cloned().unwrap_or(Value::Null));
    }
    Ok(p)
}

/// Check a JSON value against a declared type exactly as [`schema`] states
/// it, naming the path of the first value that does not conform.
fn conforms(v: &Value, t: &Ty, m: &Module, at: &str, depth: usize) -> Result<(), String> {
    if depth > 16 {
        return Err(format!("{at} is nested deeper than 16 levels"));
    }
    let t = crate::host::unalias_ty(t, m);
    let bad = |want: &str| Err(format!("{at} is {} where {want} is declared", crate::host::json_kind_of(v)));
    if let Some(n) = int_name(t) {
        let (lo, hi) = crate::types::int_range(n);
        let x = v.as_i64().map(i128::from).or_else(|| v.as_u64().map(i128::from));
        return match x {
            Some(x) if x >= lo && x <= hi => Ok(()),
            Some(x) => Err(format!("{at} is {x}, outside {n} ({lo} to {hi})")),
            None => bad(n),
        };
    }
    if simple(t, "f64") || simple(t, "f32") {
        return if v.is_number() { Ok(()) } else { bad("a number") };
    }
    if simple(t, "bool") {
        return if v.is_boolean() { Ok(()) } else { bad("bool") };
    }
    if simple(t, "text") {
        return if v.is_string() { Ok(()) } else { bad("text") };
    }
    match t {
        Ty::Opt(x) => {
            if v.is_null() {
                Ok(())
            } else {
                conforms(v, x, m, at, depth + 1)
            }
        }
        Ty::Path(p, g) if p.len() == 1 && p[0] == "List" && g.len() == 1 => match v.as_array() {
            Some(xs) => xs.iter().enumerate().try_for_each(|(i, x)| conforms(x, &g[0], m, &format!("{at}[{i}]"), depth + 1)),
            None => bad("a list"),
        },
        Ty::Path(p, g) if p.len() == 1 && g.is_empty() => {
            let fs = struct_fields(m, &p[0]).ok_or_else(|| format!("{at}: type {} has no JSON Schema here", p[0]))?;
            let fields: Vec<(String, Ty)> = fs.iter().map(|f| (f.name.clone(), f.ty.clone())).collect();
            match v.as_object() {
                Some(o) => conforms_object(o, &fields, m, at, depth + 1),
                None => bad(&format!("struct {}", p[0])),
            }
        }
        other => Err(format!("{at}: type {} has no JSON Schema here", crate::host::ty_name(other))),
    }
}

fn conforms_object(o: &Map<String, Value>, fields: &[(String, Ty)], m: &Module, at: &str, depth: usize) -> Result<(), String> {
    let join = |k: &str| if at.is_empty() { k.to_string() } else { format!("{at}.{k}") };
    if let Some(k) = o.keys().find(|k| !fields.iter().any(|(f, _)| f == *k)) {
        return Err(format!("{} is not a field (additionalProperties is false)", join(k)));
    }
    for (k, t) in fields {
        match o.get(k) {
            Some(v) => conforms(v, t, m, &join(k), depth)?,
            None if is_opt(t, m) => {}
            None => return Err(format!("{} is missing", join(k))),
        }
    }
    Ok(())
}

/// HC-35: the event an invocation names and its arguments, built for the
/// module's declaration, or why the invocation does not conform: the operation
/// is not one of the toolset's, or the arguments do not conform its
/// `inputSchema`. The caller then checks the event can be constructed (a
/// refined type, 5.1) and answers a failure with an error result.
pub fn arguments(m: &Module, inv: &Map<String, Value>) -> Result<(String, Vec<Arg>), String> {
    let (tools, _) = tools(m);
    let op = match inv.get("operation") {
        Some(Value::String(s)) => s.clone(),
        _ => return Err("operation is not a string".into()),
    };
    let Some(t) = tools.iter().find(|t| t.name == op) else {
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        return Err(format!("unknown operation {op}; the toolset has {}", names.join(", ")));
    };
    let args = match inv.get("arguments") {
        Some(Value::Object(o)) => o,
        _ => return Err(format!("the arguments of {op} are not a JSON object")),
    };
    conforms_object(args, &t.fields, m, "arguments", 0).map_err(|w| format!("{op}: {w}"))?;
    let mut out = vec![];
    for (k, ty) in &t.fields {
        let v = args.get(k).cloned().unwrap_or(Value::Null);
        out.push(Arg { label: Some(k.clone()), e: crate::host::convert_value(&v, ty, m).map_err(|w| format!("{op}: arguments.{k}: {w}"))? });
    }
    Ok((op, out))
}

/// How the step of one invocation ended, as replay re-derives it (HC-36).
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// the invocation does not conform: why
    Invalid(String),
    /// the step applied: the event's name, its entry's seq, each output as
    /// canonical JSON `{"payload":..,"type":..}` (6.1), and the state hash
    /// after it
    Applied { event: String, seq: u64, outputs: Vec<String>, state_hash: String },
    /// the step was kept (HC-21): the event's name, its seq, the cause, the
    /// text replay re-derives
    Kept { event: String, seq: u64, cause: String, text: String },
}

/// The text of a result (HC-36): the outputs of an applied step as
/// `{"outputs":[...],"seq":N,"state_hash":H}`, or an error that starts with
/// `Error: `, as RAP's error convention asks.
pub fn result_text(o: &Outcome) -> String {
    match o {
        Outcome::Invalid(why) => format!("Error: {why}"),
        Outcome::Applied { seq, outputs, state_hash, .. } => format!("{{\"outputs\":[{}],\"seq\":{seq},\"state_hash\":\"{state_hash}\"}}", outputs.join(",")),
        Outcome::Kept { event, seq, cause, text } => format!("Error: the step of {event} at seq {seq} was kept (cause {cause}): {text}; the state is unchanged"),
    }
}

/// The bytes of the tool result for an invocation (HC-36): canonical JSON
/// with `call_id` echoed (null when the invocation had none), `group_id`
/// and `id` of the invocation, the text of [`result_text`] and `type`.
pub fn result_body(inv: &Map<String, Value>, o: &Outcome) -> String {
    let mut r = Map::new();
    r.insert("call_id".into(), inv.get("call_id").cloned().unwrap_or(Value::Null));
    r.insert("group_id".into(), inv.get("group_id").cloned().unwrap_or(Value::Null));
    r.insert("id".into(), inv.get("id").cloned().unwrap_or(Value::Null));
    r.insert("text".into(), Value::String(result_text(o)));
    r.insert("type".into(), Value::String("tool_result".into()));
    crate::interp::canon_json(&Value::Object(r))
}

/// POST `body` to an `http://` URL once: the status of the answer, or why
/// there was none. The request closes its connection.
pub fn post(url: &str, body: &[u8]) -> Result<u16, String> {
    let (addr, host, port, path) = http_url(url)?;
    let sa = (host.trim_start_matches('[').trim_end_matches(']'), port).to_socket_addrs().map_err(|e| format!("{addr}: {e}"))?.next().ok_or_else(|| format!("{addr}: no address"))?;
    let mut s = TcpStream::connect_timeout(&sa, Duration::from_millis(CONNECT_MS)).map_err(|e| format!("{addr}: {e}"))?;
    s.set_write_timeout(Some(Duration::from_millis(ANSWER_MS))).ok();
    s.set_read_timeout(Some(Duration::from_millis(ANSWER_MS))).ok();
    let mut req = format!("POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).into_bytes();
    req.extend_from_slice(body);
    s.write_all(&req).and_then(|_| s.flush()).map_err(|e| format!("{addr}: {e}"))?;
    let mut got = vec![];
    let mut b = [0u8; 1024];
    while !got.windows(2).any(|w| w == b"\r\n") {
        match s.read(&mut b) {
            Ok(0) => break,
            Ok(n) => got.extend_from_slice(&b[..n]),
            Err(e) => return Err(format!("{addr}: {e}")),
        }
        if got.len() > 8192 {
            break;
        }
    }
    let line = String::from_utf8_lossy(&got);
    let line = line.split("\r\n").next().unwrap_or("");
    let mut ws = line.split(' ');
    match (ws.next(), ws.next().and_then(|c| c.parse::<u16>().ok())) {
        (Some(v), Some(c)) if v.starts_with("HTTP/1.") => Ok(c),
        _ => Err(format!("{addr}: no HTTP status line in the answer")),
    }
}

/// Whether an answer ends a delivery: any status but 5xx. RAP retries on
/// 5xx and on a failed connection, and not on 4xx, which says the result
/// was rejected for a reason that will not change.
pub fn settles(status: u16) -> bool {
    !(500..=599).contains(&status)
}

/// The deliveries of results in flight (HC-36): each one POSTs its result
/// until an answer settles it, pausing between attempts from
/// [`RETRY_FIRST_MS`] doubling to [`RETRY_MAX_MS`], then reports the
/// status. A stopped host abandons what is not settled; the next host on
/// the same D sends it again.
#[derive(Clone, Default)]
pub struct Callbacks {
    stop: Arc<AtomicBool>,
    pending: Arc<AtomicUsize>,
}

impl Callbacks {
    /// Deliver one result on its own thread; `done(call, status)` when settled.
    pub fn send(&self, call: u64, url: String, body: String, done: Arc<dyn Fn(u64, u16) + Send + Sync>) {
        let (stop, pending) = (self.stop.clone(), self.pending.clone());
        pending.fetch_add(1, Ordering::SeqCst);
        std::thread::spawn(move || {
            let mut pause = RETRY_FIRST_MS;
            loop {
                if let Ok(s) = post(&url, body.as_bytes())
                    && settles(s)
                {
                    done(call, s);
                    break;
                }
                let until = Instant::now() + Duration::from_millis(pause);
                while Instant::now() < until && !stop.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(20));
                }
                if stop.load(Ordering::SeqCst) {
                    break;
                }
                pause = (pause * 2).min(RETRY_MAX_MS);
            }
            pending.fetch_sub(1, Ordering::SeqCst);
        });
    }
    /// Results not yet settled.
    pub fn pending(&self) -> usize {
        self.pending.load(Ordering::SeqCst)
    }
    /// Stop retrying.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_split_into_a_grant_and_a_path() {
        assert_eq!(http_url("http://127.0.0.1:9000/cb?t=1").unwrap(), ("127.0.0.1:9000".into(), "127.0.0.1".into(), 9000, "/cb?t=1".into()));
        assert_eq!(http_url("http://agent.example.com").unwrap().0, "agent.example.com:80");
        assert_eq!(http_url("http://[::1]:7/x").unwrap().0, "[::1]:7");
        assert!(http_url("https://agent.example.com/cb").is_err());
        assert!(http_url("http://user@host:1/").is_err());
    }

    #[test]
    fn deep_bodies_are_refused_before_parsing() {
        let deep = format!("{}{}", "[".repeat(DEPTH_MAX + 1), "]".repeat(DEPTH_MAX + 1));
        assert!(envelope(deep.as_bytes(), &[]).unwrap_err().contains("nests deeper"));
        assert!(!too_deep(br#"{"a":"[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[["}"#));
    }
}
