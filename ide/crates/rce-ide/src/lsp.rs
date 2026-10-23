//! Language-server client for checker diagnostics.
//!
//! Speaks LSP (`initialize`, `textDocument/didOpen|didChange`, and the
//! server's `textDocument/publishDiagnostics`) to the RCE language server.
//! Two transports carry the same messages: [`Lsp::stdio`] spawns an
//! external server process (`rce-check lsp`, Content-Length framing), and
//! [`Lsp::in_process`] drives `rce_spec_check::cli::Server` directly, for
//! the browser build and for hosts with no server binary.
use serde_json::{Value, json};
use std::io::{BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Diagnostic {
    /// 1-based line.
    pub line: usize,
    pub message: String,
}

enum Transport {
    InProcess(Box<rce_spec_check::cli::Server>),
    Stdio {
        child: Child,
        w: ChildStdin,
        r: BufReader<ChildStdout>,
    },
}

pub struct Lsp {
    t: Transport,
    next_id: u64,
    version: i64,
    pub server_name: String,
}

fn diags_of(params: &Value) -> Vec<Diagnostic> {
    params["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|d| Diagnostic {
            line: d["range"]["start"]["line"].as_u64().unwrap_or(0) as usize + 1,
            message: d["message"].as_str().unwrap_or("").to_string(),
        })
        .collect()
}

impl Lsp {
    pub fn in_process() -> Self {
        let mut l = Lsp {
            t: Transport::InProcess(Box::default()),
            next_id: 0,
            version: 0,
            server_name: String::new(),
        };
        l.initialize();
        l
    }

    /// Spawn `cmd args...` as an LSP server over stdio.
    pub fn stdio(cmd: &std::path::Path, args: &[&str]) -> std::io::Result<Self> {
        let mut child = Command::new(cmd)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let w = child.stdin.take().expect("piped");
        let r = BufReader::new(child.stdout.take().expect("piped"));
        let mut l = Lsp {
            t: Transport::Stdio { child, w, r },
            next_id: 0,
            version: 0,
            server_name: String::new(),
        };
        l.initialize();
        if l.server_name.is_empty() {
            return Err(std::io::Error::other(
                "language server did not answer initialize",
            ));
        }
        Ok(l)
    }

    /// Send one message, return everything the server sends back up to
    /// (and including) the response to it, or the first diagnostics push.
    fn exchange(&mut self, m: Value) -> Vec<Value> {
        match &mut self.t {
            Transport::InProcess(s) => s.handle(&m).0,
            Transport::Stdio { w, r, .. } => {
                let s = m.to_string();
                if write!(w, "Content-Length: {}\r\n\r\n{s}", s.len())
                    .and_then(|_| w.flush())
                    .is_err()
                {
                    return vec![];
                }
                let want_id = m.get("id").cloned();
                let mut out = vec![];
                while let Some(v) = rce_spec_check::cli::read_msg(r) {
                    let done = match &want_id {
                        Some(id) => v.get("id") == Some(id),
                        None => v["method"] == "textDocument/publishDiagnostics",
                    };
                    out.push(v);
                    if done {
                        break;
                    }
                }
                out
            }
        }
    }

    fn initialize(&mut self) {
        self.next_id += 1;
        let r = self.exchange(
            json!({"jsonrpc": "2.0", "id": self.next_id, "method": "initialize",
            "params": {"processId": null, "rootUri": null, "capabilities": {}}}),
        );
        self.server_name = r
            .iter()
            .find_map(|v| {
                v["result"]["serverInfo"]["name"]
                    .as_str()
                    .map(str::to_string)
            })
            .unwrap_or_default();
        self.exchange_notify(json!({"jsonrpc": "2.0", "method": "initialized", "params": {}}));
    }

    fn exchange_notify(&mut self, m: Value) {
        // `initialized` gets no reply; do not wait for one on stdio.
        match &mut self.t {
            Transport::InProcess(s) => {
                s.handle(&m);
            }
            Transport::Stdio { w, .. } => {
                let s = m.to_string();
                let _ = write!(w, "Content-Length: {}\r\n\r\n{s}", s.len()).and_then(|_| w.flush());
            }
        }
    }

    fn diagnostics_from(out: Vec<Value>, uri: &str) -> Vec<Diagnostic> {
        out.iter()
            .rev()
            .find(|v| v["method"] == "textDocument/publishDiagnostics" && v["params"]["uri"] == uri)
            .map(|v| diags_of(&v["params"]))
            .unwrap_or_default()
    }

    pub fn open(&mut self, uri: &str, text: &str) -> Vec<Diagnostic> {
        self.version = 1;
        let out = self.exchange(json!({"jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": {"textDocument": {"uri": uri, "languageId": "rce", "version": 1, "text": text}}}));
        Self::diagnostics_from(out, uri)
    }

    pub fn change(&mut self, uri: &str, text: &str) -> Vec<Diagnostic> {
        self.version += 1;
        let out = self.exchange(json!({"jsonrpc": "2.0", "method": "textDocument/didChange",
            "params": {"textDocument": {"uri": uri, "version": self.version}, "contentChanges": [{"text": text}]}}));
        Self::diagnostics_from(out, uri)
    }

    pub fn is_stdio(&self) -> bool {
        matches!(self.t, Transport::Stdio { .. })
    }
}

impl Drop for Lsp {
    fn drop(&mut self) {
        if let Transport::Stdio { child, .. } = &mut self.t {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
