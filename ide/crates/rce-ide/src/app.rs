//! Opening a workspace: the rce-mcp server, ledger and language server the
//! IDE runs on. Shared by the desktop binary and the tests.
use crate::lsp::Lsp;
use crate::vm::Ide;
use ide_receipt::{Ledger, SigningKey, generate_key};
use rce_mcp::{Server, ws::Workspace};
use std::path::{Path, PathBuf};

pub struct Options {
    pub root: PathBuf,
    pub file: String,
    /// External language server command (e.g. `rce-check lsp`); in-process otherwise.
    pub lsp: Option<Vec<String>>,
    pub memory: bool,
}

fn key(p: &Path) -> Result<SigningKey, String> {
    if let Ok(b) = std::fs::read(p) {
        let a: [u8; 32] = b
            .try_into()
            .map_err(|_| format!("{}: expected 32 bytes", p.display()))?;
        return Ok(SigningKey::from_bytes(&a));
    }
    let k = generate_key().map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(p)
            .and_then(|mut f| f.write_all(&k.to_bytes()))
            .map_err(|e| e.to_string())?;
    }
    #[cfg(not(unix))]
    std::fs::write(p, k.to_bytes()).map_err(|e| e.to_string())?;
    Ok(k)
}

pub fn open(o: &Options) -> Result<Ide, String> {
    let ws = Workspace::new(&o.root).map_err(|e| format!("{}: {e}", o.root.display()))?;
    let state = ws.root.join(".rce-ide");
    std::fs::create_dir_all(&state).map_err(|e| e.to_string())?;
    let ledger_path = state.join("receipts.jsonl");
    let ledger =
        Ledger::open(&ledger_path, key(&state.join("ed25519.key"))?).map_err(|e| e.to_string())?;
    let meter = ide_energy::meter::RaplMeter::open()
        .ok()
        .map(|m| Box::new(m) as Box<dyn ide_energy::meter::Meter>);
    // A saved calibration (`rce-ide telemetry calibrate`), if any; without
    // one, E_sim stays absent rather than invented.
    let sim = crate::calibrate::default_path()
        .filter(|p| p.exists())
        .map(|p| ide_energy::calib::load_sim(&p))
        .transpose()?
        .unwrap_or_default();
    let mut server = Server::new(ws, ledger, sim, meter);
    if o.memory {
        let m = rce_memory::service::Memory::open_default(&server.ws.root)
            .map_err(|e| format!("memory: {e}"))?;
        server = server.with_memory(m);
    }
    let lsp = match &o.lsp {
        Some(cmd) if !cmd.is_empty() => {
            let args: Vec<&str> = cmd[1..].iter().map(String::as_str).collect();
            Lsp::stdio(Path::new(&cmd[0]), &args)
                .map_err(|e| format!("language server {}: {e}", cmd[0]))?
        }
        _ => Lsp::in_process(),
    };
    Ok(Ide::new(server, lsp, ledger_path, &o.file))
}
