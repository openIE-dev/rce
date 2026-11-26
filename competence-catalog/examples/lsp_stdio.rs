//! Line-delimited JSON-RPC LSP stdio transport for the competence-backed editor loop.
//! Run: cargo run --example lsp_stdio  (one JSON message per line on stdin; responses on stdout).
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
