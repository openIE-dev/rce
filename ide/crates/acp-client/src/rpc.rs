//! Newline-delimited JSON-RPC 2.0 over a child's stdio (ACP stdio transport).
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, ChildStdout};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::time::{Duration, Instant};

/// One message from the peer.
#[derive(Debug, Clone, PartialEq)]
pub enum Incoming {
    Response {
        id: Value,
        result: Result<Value, Value>,
    },
    Request {
        id: Value,
        method: String,
        params: Value,
    },
    Notification {
        method: String,
        params: Value,
    },
    /// A line that is not a JSON-RPC message (counted, never acted on).
    Garbage(String),
}

pub fn classify(v: Value) -> Incoming {
    let method = v.get("method").and_then(Value::as_str).map(str::to_string);
    let id = v.get("id").cloned();
    match (method, id) {
        (Some(m), Some(id)) => Incoming::Request {
            id,
            method: m,
            params: v.get("params").cloned().unwrap_or(Value::Null),
        },
        (Some(m), None) => Incoming::Notification {
            method: m,
            params: v.get("params").cloned().unwrap_or(Value::Null),
        },
        (None, Some(id)) => {
            let result = match (v.get("result"), v.get("error")) {
                (_, Some(e)) => Err(e.clone()),
                (Some(r), None) => Ok(r.clone()),
                (None, None) => Err(json!({"code": -32603, "message": "response without result"})),
            };
            Incoming::Response { id, result }
        }
        (None, None) => Incoming::Garbage(v.to_string()),
    }
}

pub struct Conn {
    w: ChildStdin,
    rx: Receiver<(Incoming, u64)>,
    next_id: u64,
    /// Bytes written to and read from the transport.
    pub bytes_out: u64,
    pub bytes_in: u64,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum RpcError {
    #[error("agent closed the connection")]
    Closed,
    #[error("timed out")]
    Timeout,
    #[error("io: {0}")]
    Io(String),
}

impl Conn {
    pub fn new(w: ChildStdin, r: ChildStdout) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(r).lines() {
                let Ok(line) = line else { break };
                if line.trim().is_empty() {
                    continue;
                }
                let n = line.len() as u64 + 1;
                let m = match serde_json::from_str::<Value>(&line) {
                    Ok(v) => classify(v),
                    Err(_) => Incoming::Garbage(line),
                };
                if tx.send((m, n)).is_err() {
                    break;
                }
            }
        });
        Conn {
            w,
            rx,
            next_id: 0,
            bytes_out: 0,
            bytes_in: 0,
        }
    }

    fn write(&mut self, v: &Value) -> Result<(), RpcError> {
        let mut s = serde_json::to_string(v).map_err(|e| RpcError::Io(e.to_string()))?;
        s.push('\n');
        self.bytes_out += s.len() as u64;
        self.w
            .write_all(s.as_bytes())
            .and_then(|_| self.w.flush())
            .map_err(|e| RpcError::Io(e.to_string()))
    }

    /// Send a request; returns its id.
    pub fn send_request(&mut self, method: &str, params: Value) -> Result<Value, RpcError> {
        let id = json!(self.next_id);
        self.next_id += 1;
        self.write(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))?;
        Ok(id)
    }

    pub fn notify(&mut self, method: &str, params: Value) -> Result<(), RpcError> {
        self.write(&json!({"jsonrpc": "2.0", "method": method, "params": params}))
    }

    pub fn respond(
        &mut self,
        id: Value,
        result: Result<Value, (i64, String)>,
    ) -> Result<(), RpcError> {
        let v = match result {
            Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
            Err((code, message)) => {
                json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
            }
        };
        self.write(&v)
    }

    /// Next message, waiting until `deadline`.
    pub fn recv(&mut self, deadline: Instant) -> Result<Incoming, RpcError> {
        let left = deadline.saturating_duration_since(Instant::now());
        match self.rx.recv_timeout(left.max(Duration::from_millis(1))) {
            Ok((m, n)) => {
                self.bytes_in += n;
                Ok(m)
            }
            Err(RecvTimeoutError::Timeout) => Err(RpcError::Timeout),
            Err(RecvTimeoutError::Disconnected) => Err(RpcError::Closed),
        }
    }
}
