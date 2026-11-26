//! rce-lsp — the shipped competence-backed LSP server binary that RCEdit plugins spawn.
//! Line-delimited JSON-RPC over stdio; diagnostics-on-change = edit-time replay gate.
use std::io::{BufRead, Write};

use competence_catalog::lsp_server::{handle, LspState};

fn main() {
    let mut state = LspState::new();
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() { continue; }
        let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
        for out in handle(&mut state, &msg) {
            let _ = writeln!(stdout, "{}", serde_json::to_string(&out).unwrap());
            let _ = stdout.flush();
        }
        if msg.get("method").and_then(|m| m.as_str()) == Some("exit") { break; }
    }
}
