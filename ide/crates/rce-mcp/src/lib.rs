//! `rce-mcp`: a stdio MCP server over the RCE Form.
//!
//! JSON-RPC 2.0, one message per line (the MCP stdio transport). Methods:
//! `initialize`, `notifications/initialized`, `ping`, `tools/list`,
//! `tools/call`. Tools: `read_form`, `propose_edit`, `rce_check`, `rce_test`,
//! `emit`. The checker is the RCE reference checker (`rce-spec-check`), the
//! same code as `rce check`, linked in-process. Every tool call writes one
//! signed receipt. No model and no network are used.
pub mod http;
pub mod jail;
pub mod items;
pub mod ws;

use ide_energy::meter::{Meter, now_ns};
pub use ide_energy::shared::SharedMeter;
use ide_energy::sim::{Features, LinearSim, Simulator};
use ide_receipt::{ActionKind, Draft, Ledger, Measured, MeterSource, Verdict, policy_hash};
use rce_memory::redb_engine::RedbEngine;
use rce_memory::service::{Memory, RecallQuery, ReceiptCtx, Remember};
use rce_spec_check::{check_program, cli};
use serde_json::{Value, json};
use std::io::{BufRead, Write};
use ws::Workspace;

pub const SERVER_NAME: &str = "rce-mcp";
pub const PROTOCOL_VERSIONS: &[&str] = &["2025-11-25", "2025-06-18", "2025-03-26"];

pub struct Server {
    pub ws: Workspace,
    pub ledger: Ledger,
    pub sim: LinearSim,
    /// Shared so overlapping actions split the counter (`apportion`).
    pub meter: Option<SharedMeter>,
    /// MEI memory over this workspace; memory tools are listed only when set.
    pub memory: Option<Memory<RedbEngine>>,
    policy_hash: String,
}

/// Result of one tool: a JSON payload, an error flag, and what the receipt needs.
struct Out {
    payload: Value,
    is_error: bool,
    verdict: Verdict,
    feats: Features,
}

fn tool_specs() -> Value {
    let path = json!({"type": "string", "description": "path relative to the workspace root"});
    let paths = json!({"type": "array", "items": {"type": "string"}, "description": "files to report; default all .rce files"});
    json!([
        {"name": "read_form", "description": "Read one .rce file: its source, its top-level items, and its Form JSON when rce check admits it (diagnostics otherwise).",
         "inputSchema": {"type": "object", "properties": {"path": path}, "required": ["path"]}},
        {"name": "propose_edit", "description": "Propose a structured change to one .rce file. Edits address whole top-level items by name. The change is written only if rce check admits the whole workspace program afterwards; otherwise it is rejected with diagnostics and nothing is written.",
         "inputSchema": {"type": "object", "properties": {"path": path, "edits": {"type": "array", "items": {"type": "object", "properties": {
             "op": {"type": "string", "enum": ["replace_item", "insert_item", "delete_item", "set_file"]},
             "name": {"type": "string"}, "after": {"type": "string"}, "text": {"type": "string"}}, "required": ["op"]}}},
          "required": ["path", "edits"]}},
        {"name": "rce_check", "description": "rce check over the workspace program.", "inputSchema": {"type": "object", "properties": {"paths": paths}}},
        {"name": "rce_test", "description": "rce test: check every file and count its examples and checks.", "inputSchema": {"type": "object", "properties": {"paths": paths}}},
        {"name": "graph_query", "description": "Query the Form code graph: callers NAME | callees NAME | reach NAME | path A B | kind KIND [joules > J] [divergence > D] | refused | uses-hw NAME. Returns nodes with file, line, kind and energy.",
         "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
        {"name": "graph_scip", "description": "Export the Form code graph as a SCIP index (definitions and call references), base64 protobuf, for code-intelligence tools.",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "emit", "description": "rce emit TARGET FILE. Targets: form, rust, python, js and scala (scala does not yet agree with the reference).",
         "inputSchema": {"type": "object", "properties": {"path": path, "target": {"type": "string", "enum": ["form", "rust", "python", "js", "scala"]}}, "required": ["path", "target"]}},
    ])
}

fn count_nodes(v: &Value) -> u64 {
    match v {
        Value::Array(a) => 1 + a.iter().map(count_nodes).sum::<u64>(),
        Value::Object(o) => 1 + o.values().map(count_nodes).sum::<u64>(),
        _ => 1,
    }
}

fn form_nodes(src: &str) -> u64 {
    cli::emit("form", src)
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .map(|v| count_nodes(&v))
        .unwrap_or(0)
}

fn feats(nodes: u64, passes: u64) -> Features {
    Features {
        form_nodes: nodes,
        checker_passes: passes,
        gear: "checker".into(),
        ..Default::default()
    }
}

fn rejected(reason: String) -> Out {
    Out {
        payload: json!({"error": reason}),
        is_error: true,
        verdict: Verdict::Rejected { reason },
        feats: feats(0, 0),
    }
}

fn str_arg<'a>(a: &'a Value, k: &str) -> Result<&'a str, String> {
    a.get(k)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string argument `{k}`"))
}

impl Server {
    pub fn new(
        ws: Workspace,
        ledger: Ledger,
        sim: LinearSim,
        meter: Option<Box<dyn Meter>>,
    ) -> Self {
        let policy_hash = policy_hash(&ws.policy());
        Server {
            ws,
            ledger,
            sim,
            meter: meter.map(SharedMeter::new),
            memory: None,
            policy_hash,
        }
    }

    /// Attach memory (memory tools appear in `tools/list`).
    pub fn with_memory(mut self, m: Memory<RedbEngine>) -> Self {
        self.memory = Some(m);
        self
    }

    pub fn policy_hash(&self) -> &str {
        &self.policy_hash
    }

    /// Handle one JSON-RPC message. `None` for notifications.
    pub fn handle(&mut self, m: &Value) -> Option<Value> {
        let id = m.get("id").cloned();
        let method = m.get("method").and_then(Value::as_str).unwrap_or("");
        let Some(id) = id else {
            return None; // notification (e.g. notifications/initialized): no reply
        };
        if m.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
            return Some(err(id, -32600, "invalid request: jsonrpc must be \"2.0\""));
        }
        let params = m.get("params").cloned().unwrap_or(json!({}));
        Some(match method {
            "initialize" => {
                let asked = params
                    .get("protocolVersion")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let v = if PROTOCOL_VERSIONS.contains(&asked) {
                    asked
                } else {
                    PROTOCOL_VERSIONS[0]
                };
                ok(
                    id,
                    json!({
                        "protocolVersion": v,
                        "capabilities": {"tools": {"listChanged": false}},
                        "serverInfo": {"name": SERVER_NAME, "version": env!("CARGO_PKG_VERSION")},
                        "instructions": "Every change goes through propose_edit and is applied only if rce check admits it. Paths are relative to the workspace. Each call writes a signed receipt.",
                    }),
                )
            }
            "ping" => ok(id, json!({})),
            "tools/list" => {
                let mut t = tool_specs();
                if self.memory.is_some()
                    && let (Some(a), Value::Array(m)) = (t.as_array_mut(), memory_specs())
                {
                    a.extend(m);
                }
                ok(id, json!({"tools": t}))
            }
            "tools/call" => {
                let Some(name) = params.get("name").and_then(Value::as_str) else {
                    return Some(err(id, -32602, "tools/call needs `name`"));
                };
                let args = params.get("arguments").cloned().unwrap_or(json!({}));
                match self.call(name, &args) {
                    Ok(v) => ok(id, v),
                    Err((code, msg)) => err(id, code, &msg),
                }
            }
            _ => err(id, -32601, &format!("method not found: {method}")),
        })
    }


    /// Run one tool and write its receipt.
    pub fn call(&mut self, name: &str, args: &Value) -> Result<Value, (i64, String)> {
        if MEMORY_TOOLS.contains(&name) {
            return self.memory_call(name, args);
        }
        let kind = match name {
            "read_form" => ActionKind::ReadForm,
            "propose_edit" => ActionKind::ProposeEdit,
            "rce_check" => ActionKind::Check,
            "rce_test" => ActionKind::Test,
            "emit" => ActionKind::Emit,
            "graph_query" | "graph_scip" => ActionKind::Other(name.to_string()),
            _ => return Err((-32602, format!("unknown tool: {name}"))),
        };
        let t0 = now_ns();
        let span = self.meter.as_ref().and_then(|m| m.begin());
        let out = match name {
            "read_form" => self.read_form(args),
            "propose_edit" => self.propose_edit(args),
            "rce_check" => self.check(args, false),
            "rce_test" => self.check(args, true),
            "graph_query" => self.graph_query(args),
            "graph_scip" => self.graph_scip(),
            _ => self.emit(args),
        };
        let joules = match (self.meter.as_ref(), span) {
            (Some(m), Some(s)) => m.end(s),
            _ => None,
        };
        let t1 = now_ns();
        let source = self.meter.as_ref().map(|m| m.source());
        let mut d = Draft::new(kind, "checker", self.policy_hash.clone());
        d.t_start_ns = t0;
        d.t_end_ns = t1;
        let pred = self.sim.predict(&out.feats);
        d.e_sim_j = pred.map(|p| p.joules);
        if let (Some(j), Some(src)) = (joules, source) {
            let tag = match src {
                "rapl" => Some(MeterSource::Rapl),
                "nvml" => Some(MeterSource::Nvml),
                "powermetrics" => Some(MeterSource::Powermetrics),
                "ioreport" => Some(MeterSource::Ioreport),
                "smc" => Some(MeterSource::Smc),
                "hwmon" => Some(MeterSource::Hwmon),
                "power-supply" => Some(MeterSource::PowerSupply),
                "jetson" => Some(MeterSource::Jetson),
                _ => None,
            };
            if let Some(tag) = tag {
                d.e_meas = Some(Measured {
                    joules: j,
                    source: tag,
                });
            }
        }
        d.verdict = out.verdict;
        d.detail = json!({"tool": name, "args": args, "features": out.feats});
        let anomaly = match (pred, d.e_meas) {
            (Some(p), Some(m)) => Some(ide_energy::anomaly::is_anomaly(
                m.joules - p.joules,
                p.sigma,
                3.0,
            )),
            _ => None,
        };
        let r = self
            .ledger
            .append(d)
            .map_err(|e| (-32603, format!("receipt write failed: {e}")))?;
        let mut payload = out.payload;
        payload["receipt"] = json!({"seq": r.body.seq, "hash": r.hash, "e_sim_j": r.body.draft.e_sim_j,
            "e_meas": r.body.draft.e_meas, "delta_j": r.body.delta_j, "anomaly": anomaly});
        Ok(json!({
            "content": [{"type": "text", "text": serde_json::to_string_pretty(&payload).expect("json")}],
            "structuredContent": payload,
            "isError": out.is_error,
        }))
    }

    fn read_form(&self, a: &Value) -> Out {
        let rel = match str_arg(a, "path") {
            Ok(s) => s,
            Err(e) => return rejected(e),
        };
        let p = match self.ws.resolve(rel, false) {
            Ok(p) => p,
            Err(e) => return rejected(e),
        };
        let src = match std::fs::read_to_string(&p) {
            Ok(s) => s,
            Err(e) => return rejected(format!("`{rel}`: {e}")),
        };
        let names: Vec<String> = items::items(&src).into_iter().map(|i| i.name).collect();
        let (form, diags) = match cli::emit("form", &src) {
            Ok(s) => (serde_json::from_str::<Value>(&s).ok(), vec![]),
            Err(e) => (None, vec![e]),
        };
        let nodes = form.as_ref().map(count_nodes).unwrap_or(0);
        Out {
            payload: json!({"path": rel, "source": src, "items": names, "form": form, "diagnostics": diags}),
            is_error: false,
            verdict: Verdict::NotRun,
            feats: feats(nodes, 1),
        }
    }

    fn propose_edit(&self, a: &Value) -> Out {
        let rel = match str_arg(a, "path") {
            Ok(s) => s,
            Err(e) => return rejected(e),
        };
        let edits: Vec<items::Edit> = match a.get("edits").cloned().map(serde_json::from_value) {
            Some(Ok(e)) => e,
            Some(Err(e)) => return rejected(format!("edits: {e}")),
            None => return rejected("missing `edits`".into()),
        };
        if edits.is_empty() {
            return rejected("no edits".into());
        }
        let p = match self.ws.resolve(rel, true) {
            Ok(p) => p,
            Err(e) => return rejected(e),
        };
        let old = std::fs::read_to_string(&p).unwrap_or_default();
        let new = match items::apply(&old, &edits) {
            Ok(s) => s,
            Err(e) => return rejected(e),
        };
        // Check the whole program with the proposed file in place.
        let rel_key = p
            .strip_prefix(&self.ws.root)
            .unwrap_or(&p)
            .display()
            .to_string();
        let mut prog = self.ws.program();
        prog.retain(|(r, _)| *r != rel_key);
        prog.push((rel_key.clone(), new.clone()));
        prog.sort();
        let verdicts = check_program(&prog, false);
        let refusals: Vec<String> = verdicts
            .iter()
            .filter_map(|(f, v)| v.as_ref().err().map(|e| format!("{f}: {e}")))
            .collect();
        let f = feats(form_nodes(&new), prog.len() as u64);
        if !refusals.is_empty() {
            return Out {
                payload: json!({"applied": false, "path": rel, "diagnostics": refusals}),
                is_error: true,
                verdict: Verdict::Refused {
                    diagnostics: refusals,
                },
                feats: f,
            };
        }
        if let Err(e) = self.ws.write_atomic(&p, &new) {
            return rejected(format!("`{rel}`: write failed: {e}"));
        }
        Out {
            payload: json!({"applied": true, "path": rel, "bytes": new.len(),
                "before_blake3": blake3::hash(old.as_bytes()).to_hex().to_string(),
                "after_blake3": blake3::hash(new.as_bytes()).to_hex().to_string()}),
            is_error: false,
            verdict: Verdict::Pass,
            feats: f,
        }
    }

    fn check(&self, a: &Value, test: bool) -> Out {
        let prog = self.ws.program();
        let want: Option<Vec<String>> = a.get("paths").and_then(Value::as_array).map(|v| {
            v.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        });
        if let Some(w) = &want {
            for r in w {
                if let Err(e) = self.ws.resolve(r, false) {
                    return rejected(e);
                }
            }
        }
        let verdicts = check_program(&prog, false);
        let mut rows = vec![];
        let mut refusals = vec![];
        let (mut pass, mut fail, mut nodes) = (0u64, 0u64, 0u64);
        for ((f, v), (_, src)) in verdicts.iter().zip(&prog) {
            if want.as_ref().is_some_and(|w| !w.contains(f)) {
                continue;
            }
            match v {
                Ok(()) => {
                    pass += 1;
                    nodes += form_nodes(src);
                    let mut row = json!({"path": f, "ok": true});
                    if test {
                        let (e, c) = rce_spec_check::parse::parse(src)
                            .map(|m| cli::count_tests(&m))
                            .unwrap_or((0, 0));
                        row["examples"] = json!(e);
                        row["checks"] = json!(c);
                    }
                    rows.push(row);
                }
                Err(e) => {
                    fail += 1;
                    refusals.push(format!("{f}: {e}"));
                    rows.push(json!({"path": f, "ok": false, "diagnostic": e}));
                }
            }
        }
        Out {
            payload: json!({"files": rows, "passed": pass, "failed": fail}),
            is_error: fail > 0,
            verdict: if fail == 0 {
                Verdict::Pass
            } else {
                Verdict::Refused {
                    diagnostics: refusals,
                }
            },
            feats: feats(nodes, prog.len() as u64),
        }
    }

    fn graph_query(&self, a: &Value) -> Out {
        let q = match str_arg(a, "query") {
            Ok(q) => q,
            Err(e) => return rejected(e),
        };
        let g = rce_graph3d::native::code_graph_of(&self.ws.program());
        match rce_graph3d::query::run(&g, q) {
            Ok(v) => {
                let nodes: Vec<Value> = v.iter().map(|&i| {
                    let n = &g.nodes[i];
                    json!({"label": n.label, "kind": n.kind.as_str(), "file": n.file, "line": n.line,
                           "joules": n.energy.joules, "refused": n.refused})
                }).collect();
                Out { payload: json!({"query": q, "nodes": nodes}), is_error: false, verdict: Verdict::Pass,
                      feats: feats(g.nodes.len() as u64, 0) }
            }
            Err(e) => rejected(e),
        }
    }

    fn graph_scip(&self) -> Out {
        let g = rce_graph3d::native::code_graph_of(&self.ws.program());
        let b = rce_graph3d::scip::export(&g, &format!("file://{}", self.ws.root.display()));
        const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut s = String::new();
        for c in b.chunks(3) {
            let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
            for k in 0..4 {
                if k <= c.len() { s.push(T[(n >> (18 - 6 * k) & 63) as usize] as char) } else { s.push('=') }
            }
        }
        Out { payload: json!({"format": "scip", "bytes": b.len(), "base64": s}), is_error: false,
              verdict: Verdict::Pass, feats: feats(g.nodes.len() as u64, 0) }
    }

    fn emit(&self, a: &Value) -> Out {
        let (rel, target) = match (str_arg(a, "path"), str_arg(a, "target")) {
            (Ok(p), Ok(t)) => (p, t),
            (Err(e), _) | (_, Err(e)) => return rejected(e),
        };
        let p = match self.ws.resolve(rel, false) {
            Ok(p) => p,
            Err(e) => return rejected(e),
        };
        let src = match std::fs::read_to_string(&p) {
            Ok(s) => s,
            Err(e) => return rejected(format!("`{rel}`: {e}")),
        };
        match cli::emit(target, &src) {
            Ok(text) => Out {
                payload: json!({"path": rel, "target": target, "text": text}),
                is_error: false,
                verdict: Verdict::Pass,
                feats: feats(form_nodes(&src), 1),
            },
            Err(e) => {
                let v = if e.contains("not implemented") || e.contains("unknown emit target") {
                    Verdict::Rejected { reason: e.clone() }
                } else {
                    Verdict::Refused {
                        diagnostics: vec![e.clone()],
                    }
                };
                Out {
                    payload: json!({"path": rel, "target": target, "error": e}),
                    is_error: true,
                    verdict: v,
                    feats: feats(0, 1),
                }
            }
        }
    }
}

pub const MEMORY_TOOLS: &[&str] = &["recall", "remember", "forget", "repo_map", "memory_sync"];

fn memory_specs() -> Value {
    json!([
        {"name": "recall", "description": "Recall from memory through cheap gears in order: exact Form lookup, graph walk, BM25 keyword, vector (only if an index is installed). Returns hits and the gears tried; escalate=true means nothing answered and a model gear may be tried. Notes are synced first (files win). Writes a receipt.",
         "inputSchema": {"type": "object", "properties": {
             "query": {"type": "string"}, "valid_at": {"type": "integer", "description": "world time, Unix ms; default now"},
             "known_at": {"type": "integer", "description": "system time, Unix ms; default now"},
             "exclude_model": {"type": "boolean"}, "limit": {"type": "integer"}}, "required": ["query"]}},
        {"name": "remember", "description": "Store a bitemporal fact pinned to the current Form hash. A different object for the same subject and predicate (multi=false) ends the old fact; nothing is deleted. note=true also writes a Markdown note under notes/. Writes a receipt whose hash the fact carries.",
         "inputSchema": {"type": "object", "properties": {
             "subject": {"type": "string"}, "predicate": {"type": "string"}, "object": {"type": "string"},
             "text": {"type": "string"}, "multi": {"type": "boolean"},
             "valid_from": {"type": "integer"}, "valid_to": {"type": "integer"},
             "provenance": {"type": "string", "enum": ["extracted", "inferred", "model"]},
             "note": {"type": "boolean"}}, "required": ["subject", "predicate", "object"]}},
        {"name": "forget", "description": "Forget as expiry: the row stops being believed (system time). With ended_at, record that it stopped being true at that world time instead. Nothing is deleted.",
         "inputSchema": {"type": "object", "properties": {"id": {"type": "integer"}, "ended_at": {"type": "integer"}}, "required": ["id"]}},
        {"name": "repo_map", "description": "Ranked map of Form definitions (PageRank over references and calls), personalized to focus names, cut to a character budget.",
         "inputSchema": {"type": "object", "properties": {"focus": {"type": "array", "items": {"type": "string"}}, "budget_chars": {"type": "integer"}}}},
        {"name": "memory_sync", "description": "Make memory match the notes/ Markdown files (files win).", "inputSchema": {"type": "object", "properties": {}}},
    ])
}

impl Server {
    fn memory_call(&mut self, name: &str, args: &Value) -> Result<Value, (i64, String)> {
        let Some(mem) = self.memory.as_mut() else {
            return Err((-32602, format!("memory is not enabled: {name}")));
        };
        let program = self.ws.program();
        let mut ctx = ReceiptCtx {
            ledger: &mut self.ledger,
            sim: &self.sim,
            meter: self.meter.as_ref(),
            policy_hash: &self.policy_hash,
        };
        let now = rce_memory::now_ms();
        let bad = |e: serde_json::Error| (-32602, format!("{name}: {e}"));
        let (payload, r, is_error) = match name {
            "recall" => {
                let q: RecallQuery = serde_json::from_value(args.clone()).map_err(bad)?;
                let (rc, r) = mem.recall(&mut ctx, &program, &q, now).map_err(mem_err)?;
                (json!(rc), r, false)
            }
            "remember" => {
                let q: Remember = serde_json::from_value(args.clone()).map_err(bad)?;
                let (p, r) = mem.remember(&mut ctx, &program, &q, now).map_err(mem_err)?;
                let rejected = matches!(r.body.draft.verdict, Verdict::Rejected { .. });
                (
                    json!({"stored": p.new_id, "duplicate_of": p.duplicate_of,
                    "expired": p.expires, "rows": p.inserts.iter().map(|f| f.id).collect::<Vec<_>>(),
                    "error": if rejected { r.body.draft.verdict.clone() } else { Verdict::NotRun }}),
                    r,
                    rejected,
                )
            }
            "forget" => {
                let id = args
                    .get("id")
                    .and_then(Value::as_u64)
                    .ok_or((-32602, "forget needs integer `id`".to_string()))?;
                let ended = args.get("ended_at").and_then(Value::as_i64);
                let (p, r) = mem.forget(&mut ctx, id, ended, now).map_err(mem_err)?;
                let rejected = matches!(r.body.draft.verdict, Verdict::Rejected { .. });
                (
                    json!({"expired": p.expires, "successor": p.new_id, "verdict": r.body.draft.verdict}),
                    r,
                    rejected,
                )
            }
            "repo_map" => {
                let focus: Vec<String> = args
                    .get("focus")
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(bad)?
                    .unwrap_or_default();
                let budget = args
                    .get("budget_chars")
                    .and_then(Value::as_u64)
                    .unwrap_or(4000) as usize;
                let (text, names, r) = mem
                    .repo_map(&mut ctx, &program, &focus, budget)
                    .map_err(mem_err)?;
                (json!({"map": text, "defs": names}), r, false)
            }
            _ => {
                let (rep, r) = mem.sync_notes(&mut ctx, &program, now).map_err(mem_err)?;
                (json!(rep), r, false)
            }
        };
        let mut payload = payload;
        payload["receipt"] = json!({"seq": r.body.seq, "hash": r.hash, "e_sim_j": r.body.draft.e_sim_j,
            "e_meas": r.body.draft.e_meas, "delta_j": r.body.delta_j, "gear": r.body.draft.gear});
        Ok(json!({
            "content": [{"type": "text", "text": serde_json::to_string_pretty(&payload).expect("json")}],
            "structuredContent": payload,
            "isError": is_error,
        }))
    }
}

fn mem_err(e: rce_memory::service::ServiceError) -> (i64, String) {
    (-32603, format!("memory: {e}"))
}

fn ok(id: Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

fn err(id: Value, code: i64, msg: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": msg}})
}

/// Serve newline-delimited JSON-RPC until EOF.
pub fn serve(server: &mut Server, r: impl BufRead, mut w: impl Write) -> std::io::Result<()> {
    for line in r.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Err(e) => Some(err(Value::Null, -32700, &format!("parse error: {e}"))),
            Ok(Value::Array(_)) => Some(err(Value::Null, -32600, "batches are not supported")),
            Ok(m) => server.handle(&m),
        };
        if let Some(v) = reply {
            serde_json::to_writer(&mut w, &v)?;
            w.write_all(b"\n")?;
            w.flush()?;
        }
    }
    Ok(())
}
