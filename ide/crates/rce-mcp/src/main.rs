//! `rce-mcp [ledger FILE] [key FILE] [calibration FILE] [no-memory]
//!          [http 127.0.0.1:PORT] [token-file FILE] [sse] [allow-origin O]...
//!          [allow-host HOST]... [no-jail] ROOT`, and `rce-mcp help`
//! (spec/proposals/positional-cli.md: words, then the operand ROOT; no
//! argument starts with `-`, and a removed flag names its new spelling).
//!
//! In HTTP mode the server jails itself after binding (see `rce_mcp::jail`):
//! declared hosts only, workspace-only files, fail closed. `no-jail` opts
//! out explicitly and is logged.
//!
//! Serves MCP over stdin/stdout, or with `http` over Streamable HTTP on a
//! loopback address (bearer token from `$RCE_MCP_TOKEN` or `token-file`;
//! a missing token file is created with a fresh token, mode 0600). Logs go to stderr. Defaults: ledger
//! `DIR/.rce-ide/receipts.jsonl`, key `DIR/.rce-ide/ed25519.key` (32 raw
//! bytes, created with mode 0600 if missing). The RAPL meter is used when
//! /sys/class/powercap is readable; otherwise receipts carry no measurement.
//! Memory (`DIR/.rce-ide/memory.redb`, notes in `DIR/notes/`) is on unless
//! `no-memory`.
use ide_energy::meter::{Meter, RaplMeter};
use rce_spec_check::args::{Command, Fate, Removed, Word};
use ide_energy::sim::LinearSim;
use ide_receipt::{Ledger, SigningKey, generate_key};
use rce_mcp::{Server, serve, ws::Workspace};
use std::path::PathBuf;

fn load_key(p: &std::path::Path) -> Result<SigningKey, String> {
    if p.exists() {
        let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
        let arr: [u8; 32] = b
            .try_into()
            .map_err(|_| format!("{}: expected 32 bytes", p.display()))?;
        return Ok(SigningKey::from_bytes(&arr));
    }
    let k = generate_key().map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(p)
            .map_err(|e| format!("{}: {e}", p.display()))?;
        f.write_all(&k.to_bytes()).map_err(|e| e.to_string())?;
    }
    #[cfg(not(unix))]
    std::fs::write(p, k.to_bytes()).map_err(|e| e.to_string())?;
    Ok(k)
}

fn write_private(p: &std::path::Path, b: &[u8]) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(p)
            .and_then(|mut f| f.write_all(b))
            .map_err(|e| format!("{}: {e}", p.display()))
    }
    #[cfg(not(unix))]
    std::fs::write(p, b).map_err(|e| format!("{}: {e}", p.display()))
}

/// The command line of rce-mcp, as data (positional-cli.md PC-8).
const TABLE: &[Command] = &[
    Command {
        path: &[],
        words: &[
            Word::set("ledger", &["FILE"]),
            Word::set("key", &["FILE"]),
            Word::set("calibration", &["FILE"]),
            Word::mode("no-memory"),
            Word::set("http", &["127.0.0.1:PORT"]),
            Word::set("token-file", &["FILE"]),
            Word::mode("sse"),
            Word::many("allow-origin", &["ORIGIN"]),
            Word::many("allow-host", &["HOST"]),
            Word::mode("no-jail"),
        ],
        operands: "ROOT",
        min: 1,
        max: 1,
        help: "serve MCP for the workspace ROOT over stdin/stdout, or over HTTP with http",
    },
    Command { path: &["help"], words: &[], operands: "", min: 0, max: 0, help: "this text" },
];

/// The flags rce-mcp had (positional-cli.md section 1.4).
const REMOVED: &[Removed] = &[
    Removed { flag: "--root", paths: &[], values: 1, fate: Fate::Operand(0) },
    Removed { flag: "--ledger", paths: &[], values: 1, fate: Fate::Word },
    Removed { flag: "--key", paths: &[], values: 1, fate: Fate::Word },
    Removed { flag: "--calibration", paths: &[], values: 1, fate: Fate::Word },
    Removed { flag: "--no-memory", paths: &[], values: 0, fate: Fate::Word },
    Removed { flag: "--http", paths: &[], values: 1, fate: Fate::Word },
    Removed { flag: "--token-file", paths: &[], values: 1, fate: Fate::Word },
    Removed { flag: "--sse", paths: &[], values: 0, fate: Fate::Word },
    Removed { flag: "--allow-origin", paths: &[], values: 1, fate: Fate::Word },
    Removed { flag: "--allow-host", paths: &[], values: 1, fate: Fate::Word },
    Removed { flag: "--no-jail", paths: &[], values: 0, fate: Fate::Word },
    Removed { flag: "--help", paths: &[], values: 0, fate: Fate::Command("help") },
    Removed { flag: "-h", paths: &[], values: 0, fate: Fate::Command("help") },
];

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (c, p) = rce_spec_check::args::read_or_exit("rce-mcp", TABLE, REMOVED, &args);
    if c.path == ["help"] {
        println!("{}", rce_spec_check::args::usage("rce-mcp", TABLE));
        return Ok(());
    }
    let file = |w: &str| p.value(w).map(PathBuf::from);
    let (ledger, key, calib) = (file("ledger"), file("key"), file("calibration"));
    let memory = !p.has("no-memory");
    let (http, token_file, sse) = (p.value("http").map(String::from), file("token-file"), p.has("sse"));
    let origins: Vec<String> = p.all("allow-origin").iter().map(|v| v[0].clone()).collect();
    let hosts: Vec<String> = p.all("allow-host").iter().map(|v| v[0].clone()).collect();
    let jail = !p.has("no-jail");
    let root = PathBuf::from(&p.operands[0]);
    let ws = Workspace::new(&root).map_err(|e| format!("{}: {e}", root.display()))?;
    let state = ws.root.join(".rce-ide");
    if ledger.is_none() || key.is_none() {
        std::fs::create_dir_all(&state).map_err(|e| e.to_string())?;
    }
    let key_path = key.unwrap_or_else(|| state.join("ed25519.key"));
    let key = load_key(&key_path)?;
    let ledger_path = ledger.unwrap_or_else(|| state.join("receipts.jsonl"));
    let ledger = Ledger::open(&ledger_path, key).map_err(|e| e.to_string())?;
    let sim = match calib {
        Some(p) => serde_json::from_str::<LinearSim>(
            &std::fs::read_to_string(&p).map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("{}: {e}", p.display()))?,
        None => LinearSim::default(),
    };
    let meter: Option<Box<dyn Meter>> = match RaplMeter::open() {
        Ok(m) => Some(Box::new(m)),
        Err(e) => {
            eprintln!("rce-mcp: no hardware meter: {e}");
            None
        }
    };
    let mut server = Server::new(ws, ledger, sim, meter);
    if memory {
        let m = rce_memory::service::Memory::open_default(&server.ws.root)
            .map_err(|e| format!("memory: {e}"))?;
        server = server.with_memory(m);
    }
    eprintln!(
        "rce-mcp: root {} policy {}",
        server.ws.root.display(),
        server.policy_hash()
    );
    if let Some(addr) = http {
        let token = match (std::env::var("RCE_MCP_TOKEN"), &token_file) {
            (Ok(t), _) => t,
            (_, Some(f)) if f.exists() => std::fs::read_to_string(f)
                .map_err(|e| format!("{}: {e}", f.display()))?
                .trim()
                .to_string(),
            (_, Some(f)) => {
                let t = rce_mcp::http::new_token();
                write_private(f, t.as_bytes())?;
                t
            }
            _ => return Err("http needs $RCE_MCP_TOKEN or token-file FILE".into()),
        };
        let mut cfg = rce_mcp::http::HttpConfig::new(token);
        cfg.sse = sse;
        cfg.allowed_origins = origins;
        rce_mcp::jail::check_bind(&addr, &hosts)?;
        let h = rce_mcp::http::HttpServer::bind(&addr, cfg).map_err(|e| format!("{addr}: {e}"))?;
        if jail {
            let ports: Vec<u16> = ide_sandbox::net::NetManifest::load(&server.ws.root)
                .map(|m| m.ports())
                .unwrap_or_default();
            let policy = rce_mcp::jail::http_policy(&server.ws.root, &[ledger_path.clone(), key_path.clone()], &ports);
            let name = ide_sandbox::jail::restrict_self(&policy)
                .map_err(|e| format!("{e} (give the word no-jail to serve unconfined)"))?;
            eprintln!("rce-mcp: jail {name}: rw {:?}, tcp connect ports {:?}", policy.rw, policy.tcp_connect);
        } else {
            eprintln!("rce-mcp: WARNING: no-jail, serving unconfined");
        }
        eprintln!("rce-mcp: listening on {}", h.url());
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        return h.serve(&mut server, stop).map_err(|e| e.to_string());
    }
    let stdin = std::io::stdin();
    serve(&mut server, stdin.lock(), std::io::stdout().lock()).map_err(|e| e.to_string())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("rce-mcp: {e}");
        std::process::exit(2);
    }
}
