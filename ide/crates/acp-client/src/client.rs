//! The ACP client: spawn, initialize, one session, prompt turns.
use crate::PROTOCOL_VERSION;
use crate::rpc::{Conn, Incoming, RpcError};
use crate::snapshot::Snapshot;
use ide_energy::meter::now_ns;
use ide_receipt::{
    ActionKind, Draft, Ledger, Measured, MeterSource, Receipt, Verdict, chain::read_jsonl,
};
use ide_sandbox::net::NetProxy;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The one MCP server the agent gets (ACP `McpServerStdio`).
#[derive(Debug, Clone, PartialEq)]
pub struct McpServer {
    pub name: String,
    pub command: PathBuf,
    pub args: Vec<String>,
}

/// Tool and capability policy.
#[derive(Debug, Clone, PartialEq)]
pub struct Policy {
    /// Tool names an agent may run (bare rce-mcp names).
    pub tools: BTreeSet<String>,
    /// ACP tool kinds never permitted, even for an allowed name.
    pub denied_kinds: BTreeSet<String>,
    /// Revert workspace changes not backed by a passing rce-mcp receipt.
    pub revert_unmediated: bool,
}

impl Default for Policy {
    fn default() -> Self {
        let tools = [
            "read_form",
            "propose_edit",
            "rce_check",
            "rce_test",
            "emit",
            "recall",
            "remember",
            "forget",
            "repo_map",
            "memory_sync",
        ];
        Policy {
            tools: tools.iter().map(|s| s.to_string()).collect(),
            denied_kinds: ["execute", "delete", "move", "fetch"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            revert_unmediated: true,
        }
    }
}

impl Policy {
    /// The rce-mcp tool a call refers to, if any. Agents label MCP tool calls
    /// differently; accepted forms: `mcp__<server>__<tool>`, `<server>:<tool>`,
    /// `<server>/<tool>`, `<server>.<tool>`, or `rawInput`/`_meta` naming the
    /// server and tool. A bare tool name is not enough (an agent's own
    /// built-in tool could share it).
    pub fn rce_tool(&self, server: &str, call: &Value) -> Option<String> {
        let cands = [
            call["title"].as_str(),
            call["_meta"]["toolName"].as_str(),
            call["rawInput"]["tool"].as_str(),
        ];
        for c in cands.into_iter().flatten() {
            let c = c.trim();
            for pre in [
                format!("mcp__{server}__"),
                format!("{server}:"),
                format!("{server}/"),
                format!("{server}."),
            ] {
                if let Some(t) = c.strip_prefix(&pre)
                    && self.tools.contains(t)
                {
                    return Some(t.to_string());
                }
            }
        }
        let srv = call["rawInput"]["server"]
            .as_str()
            .or(call["_meta"]["server"].as_str());
        let tool = call["rawInput"]["name"].as_str();
        match (srv, tool) {
            (Some(s), Some(t)) if s == server && self.tools.contains(t) => Some(t.to_string()),
            _ => None,
        }
    }

    /// Decide a `session/request_permission`: `Some(tool)` to allow.
    pub fn permit(&self, server: &str, call: &Value) -> Result<String, String> {
        let kind = call["kind"].as_str().unwrap_or("other");
        if self.denied_kinds.contains(kind) {
            return Err(format!("tool kind `{kind}` is not permitted"));
        }
        self.rce_tool(server, call).ok_or_else(|| {
            format!(
                "`{}` is not an rce-mcp tool",
                call["title"].as_str().unwrap_or("?")
            )
        })
    }
}

pub struct AcpConfig {
    /// The agent command line.
    pub command: PathBuf,
    pub args: Vec<String>,
    /// Extra environment for the agent (e.g. its own API key, supplied by the user).
    pub env: Vec<(String, String)>,
    /// Workspace root (absolute; becomes the session `cwd`).
    pub workspace: PathBuf,
    /// Path of the `rce-mcp` binary.
    pub rce_mcp: PathBuf,
    pub policy: Policy,
    pub turn_timeout: Duration,
    pub init_timeout: Duration,
    /// Run the agent (and the rce-mcp it starts) in the platform jail.
    /// `Some` by default; `start` fails closed if the jail cannot be
    /// enforced. `None` runs it unconfined (tests' control case).
    pub jail: Option<AgentJail>,
    /// How the agent reaches rce-mcp. `Stdio`: the agent spawns it (inside
    /// its jail). `Http`: the client runs it on loopback (Streamable HTTP,
    /// bearer token) and the jail grants only that port.
    pub mcp_transport: McpTransport,
    /// Extra arguments for the client-run (HTTP) rce-mcp.
    pub mcp_args: Vec<String>,
    /// Route the agent's network through this proxy: a loopback CONNECT /
    /// absolute-form listener is started for the session, its port is the
    /// only other one the jail grants, and `HTTPS_PROXY`/`HTTP_PROXY` point
    /// at it (with its per-session token). Host filtering, approvals and
    /// credentials are the proxy's.
    pub net: Option<Arc<Mutex<NetProxy>>>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum McpTransport {
    #[default]
    Stdio,
    Http,
}

/// What a jailed agent may reach beyond the workspace. By default: read
/// and write the workspace, read/execute its own binary and rce-mcp, the
/// dynamic loader and shared libraries; nothing else (no $HOME, no /etc, no
/// other projects) and no network.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentJail {
    /// Extra read-only paths (an interpreter such as node and its modules,
    /// the agent's own config).
    pub extra_ro: Vec<PathBuf>,
    /// Extra read-write paths (a cache directory).
    pub extra_rw: Vec<PathBuf>,
    /// TCP ports the agent may connect to (its model API, through the
    /// network proxy); none by default.
    pub tcp_connect: Vec<u16>,
}

impl AgentJail {
    pub fn policy(&self, ws: &Path, exes: &[PathBuf]) -> ide_sandbox::jail::JailPolicy {
        let mut p = ide_sandbox::jail::JailPolicy::for_program(ws, exes);
        p.ro.extend(self.extra_ro.iter().cloned());
        p.rw.extend(self.extra_rw.iter().cloned());
        p.tcp_connect = self.tcp_connect.clone();
        p
    }
}

impl AcpConfig {
    pub fn new(
        command: impl Into<PathBuf>,
        workspace: impl Into<PathBuf>,
        rce_mcp: impl Into<PathBuf>,
    ) -> Self {
        AcpConfig {
            command: command.into(),
            args: vec![],
            env: vec![],
            workspace: workspace.into(),
            rce_mcp: rce_mcp.into(),
            policy: Policy::default(),
            turn_timeout: Duration::from_secs(600),
            init_timeout: Duration::from_secs(30),
            jail: Some(AgentJail::default()),
            mcp_transport: McpTransport::Stdio,
            mcp_args: vec![],
            net: None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AcpError {
    #[error("spawn {0}: {1}")]
    Spawn(String, std::io::Error),
    #[error("jail: {0}")]
    Jail(String),
    #[error("rpc: {0}")]
    Rpc(#[from] RpcError),
    #[error("agent error on {0}: {1}")]
    Agent(String, Value),
    #[error("agent chose protocol version {0}; this client speaks only {PROTOCOL_VERSION}")]
    Version(Value),
    #[error("agent requires authentication ({0}); not supported here")]
    Auth(String),
    #[error("workspace: {0}")]
    Workspace(String),
    #[error("agent does not support MCP over HTTP (mcpCapabilities.http)")]
    NoMcpHttp,
    #[error("rce-mcp (http): {0}")]
    McpHttp(String),
    #[error("receipt: {0}")]
    Receipt(String),
}

/// What one prompt turn did.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct TurnReport {
    pub stop_reason: String,
    pub agent_text: String,
    /// `toolCallId → (title, kind, last status)`.
    pub tool_calls: Vec<(String, String, String, String)>,
    pub permissions_granted: Vec<String>,
    pub permissions_denied: Vec<(String, String)>,
    /// Client methods the agent called that it was never given.
    pub refused_methods: Vec<String>,
    /// rce-mcp receipts written during the turn.
    pub mcp_receipts: Vec<String>,
    pub mcp_ledger_head: Option<String>,
    /// Files changed through passing `propose_edit` (or memory) receipts.
    pub mediated: Vec<String>,
    /// Files changed outside rce-mcp; reverted when the policy says so.
    pub unmediated: Vec<String>,
    pub reverted: Vec<String>,
    /// Last `usage_update` (context tokens used/size, cost) if any.
    pub usage: Option<Value>,
    pub bytes_out: u64,
    /// Proxy decisions during the turn (endpoint, allowed/denied, bytes,
    /// whether a supervisor credential was injected).
    pub net: Vec<ide_sandbox::net::NetEvent>,
    pub bytes_in: u64,
    pub receipt: Option<String>,
}

pub struct AcpClient {
    pub cfg: AcpConfig,
    child: Child,
    conn: Conn,
    pub session_id: String,
    pub agent_info: Value,
    pub agent_caps: Value,
    pub mcp_ledger: PathBuf,
    mcp_name: String,
    ledger: Ledger,
    policy_hash: String,
    /// The jail the agent runs in (`null` when unconfined).
    pub jail: Value,
    /// Shared with any in-process server so overlaps are split (`apportion`).
    pub meter: Option<ide_energy::shared::SharedMeter>,
    /// The client-run rce-mcp (HTTP transport) and its URL.
    mcp_http: Option<(Child, String)>,
    /// The session's proxy listener (`net`).
    proxy: Option<ide_sandbox::proxy::ProxyListener>,
}

/// Start rce-mcp on loopback; returns the child, URL and bearer token.
fn start_mcp_http(
    cfg: &AcpConfig,
    args: &[String],
    token_file: &Path,
) -> Result<(Child, String, String), AcpError> {
    use std::io::BufRead;
    let mut child = Command::new(&cfg.rce_mcp)
        .args(args)
        .args(&cfg.mcp_args)
        .args(["--http", "127.0.0.1:0", "--token-file"])
        .arg(token_file)
        .current_dir(&cfg.workspace)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| AcpError::Spawn(cfg.rce_mcp.display().to_string(), e))?;
    let mut err = std::io::BufReader::new(child.stderr.take().expect("piped"));
    let mut url = None;
    for _ in 0..20 {
        let mut l = String::new();
        if err.read_line(&mut l).unwrap_or(0) == 0 {
            break;
        }
        if let Some(u) = l.trim().strip_prefix("rce-mcp: listening on ") {
            url = Some(u.to_string());
            break;
        }
    }
    std::thread::spawn(move || for _ in err.lines() {});
    let Some(url) = url else {
        let _ = child.kill();
        let _ = child.wait();
        return Err(AcpError::McpHttp("did not start listening".into()));
    };
    let token = std::fs::read_to_string(token_file)
        .map_err(|e| AcpError::McpHttp(e.to_string()))?
        .trim()
        .to_string();
    Ok((child, url, token))
}

fn url_port(url: &str) -> Option<u16> {
    let rest = url.split_once("://")?.1;
    let hp = rest.split('/').next()?;
    hp.rsplit_once(':')?.1.parse().ok()
}

fn rand_hex() -> String {
    let mut b = [0u8; 8];
    getrandom::fill(&mut b).expect("os randomness");
    hex::encode(b)
}

impl AcpClient {
    /// Spawn the agent, initialize (v1), and open one session whose only
    /// MCP server is rce-mcp. `ledger` receives the per-turn receipts.
    pub fn start(cfg: AcpConfig, ledger: Ledger) -> Result<Self, AcpError> {
        let ws = cfg
            .workspace
            .canonicalize()
            .map_err(|e| AcpError::Workspace(e.to_string()))?;
        let state = ws.join(".rce-ide/acp");
        std::fs::create_dir_all(&state).map_err(|e| AcpError::Workspace(e.to_string()))?;
        let tag = rand_hex();
        let mcp_ledger = state.join(format!("rce-mcp-{tag}.jsonl"));
        let mcp = McpServer {
            name: "rce".into(),
            command: cfg.rce_mcp.clone(),
            args: vec![
                "--root".into(),
                ws.display().to_string(),
                "--ledger".into(),
                mcp_ledger.display().to_string(),
                "--key".into(),
                state.join("rce-mcp.key").display().to_string(),
            ],
        };
        let mut cmd = Command::new(&cfg.command);
        cmd.args(&cfg.args)
            .current_dir(&ws)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        for (k, v) in &cfg.env {
            cmd.env(k, v);
        }
        // MCP over HTTP: the client runs rce-mcp; the agent gets a URL.
        let mut mcp_http = None;
        let mut grant: Vec<u16> = vec![];
        let mcp_entry = match cfg.mcp_transport {
            McpTransport::Stdio => {
                json!({"name": mcp.name, "command": mcp.command.display().to_string(), "args": mcp.args, "env": []})
            }
            McpTransport::Http => {
                let (child, url, token) =
                    start_mcp_http(&cfg, &mcp.args, &state.join(format!("rce-mcp-{tag}.token")))?;
                grant.extend(url_port(&url));
                let e = json!({"type": "http", "name": mcp.name, "url": url,
                    "headers": [{"name": "Authorization", "value": format!("Bearer {token}")}]});
                mcp_http = Some((child, url));
                e
            }
        };
        let mut proxy = None;
        if let Some(np) = &cfg.net {
            let l = ide_sandbox::proxy::listen(np.clone(), "127.0.0.1:0")
                .map_err(|e| AcpError::Jail(format!("proxy: {e}")))?;
            grant.push(l.addr.port());
            let u = l.proxy_url();
            for k in [
                "HTTPS_PROXY",
                "HTTP_PROXY",
                "https_proxy",
                "http_proxy",
                "ALL_PROXY",
            ] {
                cmd.env(k, &u);
            }
            // Only the MCP endpoint (if any) bypasses the proxy.
            for k in ["NO_PROXY", "no_proxy"] {
                cmd.env(k, if mcp_http.is_some() { "127.0.0.1" } else { "" });
            }
            proxy = Some(l);
        }
        let (mut child, mut jail) = match &cfg.jail {
            Some(j) => {
                let exe = |p: &Path| {
                    p.canonicalize()
                        .map_err(|e| AcpError::Spawn(p.display().to_string(), e))
                };
                let exes = vec![exe(&cfg.command)?, exe(&cfg.rce_mcp)?];
                let mut pol = j.policy(&ws, &exes);
                pol.tcp_connect.extend(grant.iter().copied());
                let jl = ide_sandbox::jail::platform_jail();
                let child = jl.spawn(cmd, &pol).map_err(|e| match e {
                    ide_sandbox::jail::JailError::Spawn(e) => {
                        AcpError::Spawn(cfg.command.display().to_string(), e)
                    }
                    e => AcpError::Jail(e.to_string()),
                })?;
                let desc = json!({
                    "backend": jl.name(),
                    "rw": pol.rw.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
                    "ro": pol.ro.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
                    "tcp_connect": pol.tcp_connect,
                });
                (child, desc)
            }
            None => (
                cmd.spawn()
                    .map_err(|e| AcpError::Spawn(cfg.command.display().to_string(), e))?,
                Value::Null,
            ),
        };
        if let Some(o) = jail.as_object_mut() {
            o.insert(
                "mcp".into(),
                json!(match &mcp_http {
                    Some((_, u)) => format!("http {u}"),
                    None => "stdio".into(),
                }),
            );
            o.insert(
                "proxy".into(),
                json!(proxy.as_ref().map(|l| l.addr.to_string())),
            );
        }
        let conn = Conn::new(
            child.stdin.take().expect("piped"),
            child.stdout.take().expect("piped"),
        );
        let policy_hash = blake3::hash(format!("acp-client/v1 tools={:?} denied_kinds={:?} revert={} fs=none terminal=none mcp=rce-only jail={jail}",
            cfg.policy.tools, cfg.policy.denied_kinds, cfg.policy.revert_unmediated).as_bytes()).to_hex().to_string();
        let mut c = AcpClient {
            child,
            conn,
            session_id: String::new(),
            agent_info: Value::Null,
            agent_caps: Value::Null,
            mcp_ledger,
            mcp_name: mcp.name.clone(),
            ledger,
            policy_hash,
            jail,
            meter: None,
            mcp_http,
            proxy,
            cfg,
        };
        let init = c.call(
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "clientCapabilities": {"fs": {"readTextFile": false, "writeTextFile": false}, "terminal": false},
                "clientInfo": {"name": "rce-ide", "title": "RCE IDE", "version": env!("CARGO_PKG_VERSION")}
            }),
            c.cfg.init_timeout,
            &mut TurnReport::default(),
        )?;
        if init["protocolVersion"] != json!(PROTOCOL_VERSION) {
            c.shutdown();
            return Err(AcpError::Version(init["protocolVersion"].clone()));
        }
        c.agent_info = init["agentInfo"].clone();
        c.agent_caps = init["agentCapabilities"].clone();
        if c.mcp_http.is_some() && c.agent_caps["mcpCapabilities"]["http"] != json!(true) {
            c.shutdown();
            return Err(AcpError::NoMcpHttp);
        }
        let ses = c.call(
            "session/new",
            json!({
                "cwd": ws.display().to_string(),
                "mcpServers": [mcp_entry]
            }),
            c.cfg.init_timeout,
            &mut TurnReport::default(),
        );
        let ses = match ses {
            Err(AcpError::Agent(_, e)) if e["code"] == json!(-32000) => {
                let m = init["authMethods"].to_string();
                c.shutdown();
                return Err(AcpError::Auth(m));
            }
            r => r?,
        };
        c.session_id = ses["sessionId"].as_str().unwrap_or_default().to_string();
        c.cfg.workspace = ws;
        Ok(c)
    }

    /// Send a request and serve the agent's requests until its response.
    fn call(
        &mut self,
        method: &str,
        params: Value,
        timeout: Duration,
        rep: &mut TurnReport,
    ) -> Result<Value, AcpError> {
        let id = self.conn.send_request(method, params)?;
        let deadline = Instant::now() + timeout;
        loop {
            let m = match self.conn.recv(deadline) {
                Err(RpcError::Timeout) if method == "session/prompt" => {
                    // ACP cancellation: notify, then wait for the `cancelled` stop reason.
                    self.conn
                        .notify("session/cancel", json!({"sessionId": self.session_id}))?;
                    return self.drain_after_cancel(id, rep);
                }
                r => r?,
            };
            match m {
                Incoming::Response { id: rid, result } if rid == id => {
                    return result.map_err(|e| AcpError::Agent(method.into(), e));
                }
                Incoming::Response { .. } | Incoming::Garbage(_) => {}
                Incoming::Notification { method: nm, params } => {
                    self.on_notification(&nm, &params, rep)
                }
                Incoming::Request {
                    id: rid,
                    method: rm,
                    params,
                } => {
                    let r = self.on_request(&rm, &params, rep, false);
                    self.conn.respond(rid, r)?;
                }
            }
        }
    }

    fn drain_after_cancel(&mut self, id: Value, rep: &mut TurnReport) -> Result<Value, AcpError> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match self.conn.recv(deadline) {
                Err(RpcError::Timeout) => {
                    return Ok(json!({"stopReason": "cancelled", "clientTimeout": true}));
                }
                Err(e) => return Err(e.into()),
                Ok(Incoming::Response { id: rid, result }) if rid == id => {
                    return result.map_err(|e| AcpError::Agent("session/prompt".into(), e));
                }
                Ok(Incoming::Notification { method, params }) => {
                    self.on_notification(&method, &params, rep)
                }
                Ok(Incoming::Request {
                    id: rid,
                    method,
                    params,
                }) => {
                    let r = self.on_request(&method, &params, rep, true);
                    self.conn.respond(rid, r)?;
                }
                Ok(_) => {}
            }
        }
    }

    fn on_notification(&mut self, method: &str, p: &Value, rep: &mut TurnReport) {
        if method != "session/update" {
            return;
        }
        let u = &p["update"];
        match u["sessionUpdate"].as_str().unwrap_or("") {
            "agent_message_chunk" => {
                if let Some(t) = u["content"]["text"].as_str() {
                    rep.agent_text.push_str(t);
                }
            }
            "tool_call" => rep.tool_calls.push((
                u["toolCallId"].as_str().unwrap_or("").into(),
                u["title"].as_str().unwrap_or("").into(),
                u["kind"].as_str().unwrap_or("other").into(),
                u["status"].as_str().unwrap_or("pending").into(),
            )),
            "tool_call_update" => {
                let id = u["toolCallId"].as_str().unwrap_or("");
                if let Some(t) = rep.tool_calls.iter_mut().find(|t| t.0 == id) {
                    if let Some(s) = u["status"].as_str() {
                        t.3 = s.into();
                    }
                    if let Some(s) = u["title"].as_str() {
                        t.1 = s.into();
                    }
                }
            }
            "usage_update" => rep.usage = Some(u.clone()),
            _ => {}
        }
    }

    fn on_request(
        &mut self,
        method: &str,
        p: &Value,
        rep: &mut TurnReport,
        cancelled: bool,
    ) -> Result<Value, (i64, String)> {
        match method {
            "session/request_permission" => {
                if cancelled {
                    return Ok(json!({"outcome": {"outcome": "cancelled"}}));
                }
                let call = &p["toolCall"];
                let title = call["title"].as_str().unwrap_or("?").to_string();
                let opts = p["options"].as_array().cloned().unwrap_or_default();
                let pick = |kinds: &[&str]| {
                    kinds.iter().find_map(|k| {
                        opts.iter()
                            .find(|o| o["kind"] == json!(k))
                            .and_then(|o| o["optionId"].as_str().map(str::to_string))
                    })
                };
                match self.cfg.policy.permit(&self.mcp_name, call) {
                    Ok(tool) => match pick(&["allow_once"]) {
                        Some(id) => {
                            rep.permissions_granted.push(tool);
                            Ok(json!({"outcome": {"outcome": "selected", "optionId": id}}))
                        }
                        None => {
                            rep.permissions_denied
                                .push((title, "no allow_once option (never allow_always)".into()));
                            Ok(json!({"outcome": {"outcome": "cancelled"}}))
                        }
                    },
                    Err(why) => {
                        rep.permissions_denied.push((title, why));
                        match pick(&["reject_once", "reject_always"]) {
                            Some(id) => {
                                Ok(json!({"outcome": {"outcome": "selected", "optionId": id}}))
                            }
                            None => Ok(json!({"outcome": {"outcome": "cancelled"}})),
                        }
                    }
                }
            }
            m => {
                // fs/*, terminal/*, elicitation and anything else: not advertised, not served.
                rep.refused_methods.push(m.to_string());
                Err((
                    -32601,
                    format!(
                        "method not available: {m} (this client grants only the rce-mcp tools)"
                    ),
                ))
            }
        }
    }

    /// One prompt turn, then one signed receipt.
    pub fn prompt(&mut self, text: &str) -> Result<TurnReport, AcpError> {
        let mut rep = TurnReport::default();
        let ws = self.cfg.workspace.clone();
        let before = Snapshot::take(&ws).map_err(|e| AcpError::Workspace(e.to_string()))?;
        let mcp_before = self.mcp_receipts().len();
        let (b_out, b_in) = (self.conn.bytes_out, self.conn.bytes_in);
        let net0 = self.net_events().len();
        let t0 = now_ns();
        let span = self.meter.as_ref().and_then(|m| m.begin());
        let res = self.call(
            "session/prompt",
            json!({"sessionId": self.session_id, "prompt": [{"type": "text", "text": text}]}),
            self.cfg.turn_timeout,
            &mut rep,
        );
        let jm = match (self.meter.as_ref(), span) {
            (Some(m), Some(sp)) => m.end(sp),
            _ => None,
        };
        let t1 = now_ns();
        let (verdict, stop) = match &res {
            Ok(r) => (
                None,
                r["stopReason"].as_str().unwrap_or("unknown").to_string(),
            ),
            Err(e) => (Some(e.to_string()), "error".to_string()),
        };
        rep.stop_reason = stop;
        rep.bytes_out = self.conn.bytes_out - b_out;
        rep.net = self.net_events().into_iter().skip(net0).collect();
        rep.bytes_in = self.conn.bytes_in - b_in;
        // What rce-mcp did this turn.
        let mcp: Vec<Receipt> = self.mcp_receipts().into_iter().skip(mcp_before).collect();
        rep.mcp_receipts = mcp.iter().map(|r| r.hash.clone()).collect();
        rep.mcp_ledger_head = self.mcp_receipts().last().map(|r| r.hash.clone());
        let mut ok_paths: BTreeSet<String> = BTreeSet::new();
        let mut memory_wrote = false;
        for r in &mcp {
            let d = &r.body.draft;
            if d.action == ActionKind::ProposeEdit
                && d.verdict == Verdict::Pass
                && let Some(p) = d.detail["args"]["path"].as_str()
            {
                ok_paths.insert(p.trim_start_matches("./").to_string());
            }
            if matches!(
                d.detail["tool"].as_str(),
                Some("remember" | "forget" | "memory_sync")
            ) {
                memory_wrote = true;
            }
        }
        let after = Snapshot::take(&ws).map_err(|e| AcpError::Workspace(e.to_string()))?;
        for f in before.changed(&after) {
            if ok_paths.contains(&f) || (memory_wrote && f.starts_with("notes/")) {
                rep.mediated.push(f);
            } else {
                rep.unmediated.push(f);
            }
        }
        if self.cfg.policy.revert_unmediated {
            for f in &rep.unmediated {
                if before.restore(&ws, f).unwrap_or(false) {
                    rep.reverted.push(f.clone());
                }
            }
        }
        let mut d = Draft::new(
            ActionKind::Other("acp.turn".into()),
            "g4",
            self.policy_hash.clone(),
        );
        d.model = self.agent_info["name"].as_str().map(|n| {
            format!(
                "acp:{n}@{}",
                self.agent_info["version"].as_str().unwrap_or("?")
            )
        });
        d.t_start_ns = t0;
        d.t_end_ns = t1;
        // The meter sees this machine during the turn: the agent process and
        // rce-mcp, not a remote model behind the agent.
        if let (Some(j), Some(src)) = (jm, self.meter.as_ref().map(|m| m.source())) {
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
            d.e_meas = tag.map(|source| Measured {
                joules: j,
                source,
            });
        }
        d.verdict = if let Some(e) = verdict {
            Verdict::Rejected { reason: e }
        } else if !rep.unmediated.is_empty() {
            Verdict::Rejected {
                reason: format!(
                    "writes outside rce-mcp: {:?} (reverted: {:?})",
                    rep.unmediated, rep.reverted
                ),
            }
        } else {
            Verdict::NotRun
        };
        d.detail = json!({
            "protocol_version": PROTOCOL_VERSION,
            "session_id": self.session_id,
            "agent": self.agent_info,
            "turn": rep,
            "jail": self.jail,
            "mcp_ledger": self.mcp_ledger.strip_prefix(&ws).unwrap_or(&self.mcp_ledger).display().to_string(),
            "features": {"gear": "g4", "net_bytes": rep.bytes_in + rep.bytes_out},
        });
        let r = self
            .ledger
            .append(d)
            .map_err(|e| AcpError::Receipt(e.to_string()))?;
        rep.receipt = Some(r.hash);
        res.map(|_| rep)
    }

    fn mcp_receipts(&self) -> Vec<Receipt> {
        read_jsonl(&self.mcp_ledger).unwrap_or_default()
    }

    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// Close stdin and stop the agent.
    fn net_events(&self) -> Vec<ide_sandbox::net::NetEvent> {
        self.cfg
            .net
            .as_ref()
            .and_then(|p| p.lock().ok().map(|g| g.events.clone()))
            .unwrap_or_default()
    }

    pub fn shutdown(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some((c, _)) = self.mcp_http.as_mut() {
            let _ = c.kill();
            let _ = c.wait();
        }
        if let Some(l) = self.proxy.as_mut() {
            l.stop();
        }
    }

    pub fn workspace(&self) -> &Path {
        &self.cfg.workspace
    }
}

impl Drop for AcpClient {
    fn drop(&mut self) {
        self.shutdown();
    }
}
