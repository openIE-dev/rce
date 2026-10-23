//! `rce-mcp --root DIR [--ledger FILE] [--key FILE] [--calibration FILE]
//!          [--http 127.0.0.1:PORT --token-file FILE [--sse] [--allow-origin O]..
//!           [--allow-host HOST].. [--no-jail]]`
//!
//! In HTTP mode the server jails itself after binding (see `rce_mcp::jail`):
//! declared hosts only, workspace-only files, fail closed. `--no-jail` opts
//! out explicitly and is logged.
//!
//! Serves MCP over stdin/stdout, or with `--http` over Streamable HTTP on a
//! loopback address (bearer token from `$RCE_MCP_TOKEN` or `--token-file`;
//! a missing token file is created with a fresh token, mode 0600). Logs go to stderr. Defaults: ledger
//! `DIR/.rce-ide/receipts.jsonl`, key `DIR/.rce-ide/ed25519.key` (32 raw
//! bytes, created with mode 0600 if missing). The RAPL meter is used when
//! /sys/class/powercap is readable; otherwise receipts carry no measurement.
//! Memory (`DIR/.rce-ide/memory.redb`, notes in `DIR/notes/`) is on unless
//! `--no-memory`.
use ide_energy::meter::{Meter, RaplMeter};
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

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let (mut root, mut ledger, mut key, mut calib) = (None, None, None, None);
    let mut memory = true;
    let (mut http, mut token_file, mut sse, mut origins) = (None, None, false, vec![]);
    let (mut hosts, mut jail) = (Vec::<String>::new(), true);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--root" => root = args.next().map(PathBuf::from),
            "--ledger" => ledger = args.next().map(PathBuf::from),
            "--key" => key = args.next().map(PathBuf::from),
            "--calibration" => calib = args.next().map(PathBuf::from),
            "--no-memory" => memory = false,
            "--http" => http = args.next(),
            "--token-file" => token_file = args.next().map(PathBuf::from),
            "--sse" => sse = true,
            "--allow-host" => hosts.extend(args.next()),
            "--no-jail" => jail = false,
            "--allow-origin" => origins.extend(args.next()),
            "-h" | "--help" => {
                eprintln!(
                    "usage: rce-mcp --root DIR [--ledger FILE] [--key FILE] [--calibration FILE] [--no-memory] [--http 127.0.0.1:PORT --token-file FILE [--sse] [--allow-origin ORIGIN].. [--allow-host HOST].. [--no-jail]]"
                );
                return Ok(());
            }
            x => return Err(format!("unknown argument `{x}`")),
        }
    }
    let root = root.ok_or("--root DIR is required")?;
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
            _ => return Err("--http needs $RCE_MCP_TOKEN or --token-file FILE".into()),
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
                .map_err(|e| format!("{e} (pass --no-jail to serve unconfined)"))?;
            eprintln!("rce-mcp: jail {name}: rw {:?}, tcp connect ports {:?}", policy.rw, policy.tcp_connect);
        } else {
            eprintln!("rce-mcp: WARNING: --no-jail, serving unconfined");
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
