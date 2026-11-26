//! Minimal LSP server backed by the competence-backed editor loop — the live-editor integration
//! of the zero-error hypothesis. Diagnostics are produced on `textDocument/didChange` by lifting
//! the document and replay-gating it against registered probes: an error is reported AT EDIT TIME
//! and never reaches runtime. Completion serves admitted competence / LUT surfaces, not guesses.
//!
//! Prototype transport: line-delimited JSON-RPC over stdio (example `lsp_stdio`). The message
//! handler is a pure function so tests drive it without a subprocess. Wiring to a real editor's
//! stdio/LSP client is the integration step; the semantics here are the real gate.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use crate::editor_loop::{EditorLoop, EditOutcome};
use crate::program::{lift_program, replay_admit, Env};

pub struct LspState {
    pub loop_: EditorLoop,
    pub docs: BTreeMap<String, String>,
    pub probes: BTreeMap<String, Vec<(Env, i128)>>,
}

impl LspState {
    pub fn new() -> Self {
        LspState { loop_: EditorLoop::rust(), docs: BTreeMap::new(), probes: BTreeMap::new() }
    }
}

fn diag(uri: &str, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
        "params": { "uri": uri, "diagnostics": [ {
            "range": { "start": {"line":0,"character":0}, "end": {"line":0,"character":1} },
            "severity": 1, "source": "rce-replay-gate", "message": message
        } ] }
    })
}

fn clear(uri: &str) -> Value {
    json!({ "jsonrpc":"2.0", "method":"textDocument/publishDiagnostics",
            "params": { "uri": uri, "diagnostics": [] } })
}

/// Handle one JSON-RPC message; return responses + notifications to send.
pub fn handle(state: &mut LspState, msg: &Value) -> Vec<Value> {
    let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let id = msg.get("id").cloned();
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    let reply = |result: Value| -> Vec<Value> {
        match &id { Some(i) => vec![json!({"jsonrpc":"2.0","id":i,"result":result})], None => vec![] }
    };
    match method {
        "initialize" => reply(json!({
            "capabilities": {
                "textDocumentSync": 1,
                "completionProvider": { "triggerCharacters": ["."] },
                "diagnosticProvider": { "interFileDependencies": false, "workspaceDiagnostics": false }
            },
            "serverInfo": { "name": "rce-competence-lsp", "version": "0.1.0" }
        })),
        "textDocument/didOpen" | "textDocument/didChange" => {
            let uri = params["textDocument"]["uri"].as_str().unwrap_or("").to_string();
            let text = if method == "textDocument/didOpen" {
                params["textDocument"]["text"].as_str().unwrap_or("").to_string()
            } else {
                params["contentChanges"][0]["text"].as_str().unwrap_or("").to_string()
            };
            state.docs.insert(uri.clone(), text.clone());
            // Edit-time gate: parse errors are diagnostics immediately.
            match lift_program(&text) {
                Err(e) => vec![diag(&uri, &format!("parse: {e}"))],
                Ok(stmt) => {
                    if let Some(probes) = state.probes.get(&uri) {
                        if !probes.is_empty() && !replay_admit(&stmt, probes) {
                            return vec![diag(&uri, "replay divergence: document does not reproduce its probes")];
                        }
                    }
                    vec![clear(&uri)]
                }
            }
        }
        "rce/setProbes" => {
            let uri = params["uri"].as_str().unwrap_or("").to_string();
            let mut probes: Vec<(Env, i128)> = Vec::new();
            if let Some(arr) = params["probes"].as_array() {
                for p in arr {
                    let mut env = Env::new();
                    if let Some(m) = p[0].as_object() {
                        for (k, v) in m { if let Some(n) = v.as_i64() { env.insert(k.clone(), n as i128); } }
                    }
                    if let Some(e) = p[1].as_i64() { probes.push((env, e as i128)); }
                }
            }
            state.probes.insert(uri, probes);
            reply(json!({ "ok": true }))
        }
        "rce/admit" => {
            let uri = params["uri"].as_str().unwrap_or("").to_string();
            let doc = state.docs.get(&uri).cloned().unwrap_or_default();
            let probes = state.probes.get(&uri).cloned().unwrap_or_default();
            match state.loop_.edit(&doc, &probes) {
                EditOutcome::Served { key, rust, hit } => {
                    let mut out = reply(json!({ "served": true, "hit": hit, "key": key, "rust": rust }));
                    out.push(clear(&uri));
                    out
                }
                EditOutcome::Rejected { diagnostic } => {
                    let mut out = reply(json!({ "served": false, "diagnostic": diagnostic }));
                    out.push(diag(&uri, &diagnostic));
                    out
                }
            }
        }
        "textDocument/completion" => {
            let items: Vec<Value> = state.loop_.lut.templates.iter().map(|(k, t)| {
                json!({ "label": k, "kind": 3, "insertText": t, "detail": "admitted competence surface" })
            }).collect();
            reply(json!({ "isIncomplete": false, "items": items }))
        }
        "shutdown" => reply(Value::Null),
        _ => vec![],
    }
}
