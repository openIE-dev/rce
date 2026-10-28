//! Demo data for screenshots: every panel populated through the real
//! services (rce-mcp tools, the checker, memory, the ledger, the network
//! manifest and approvals). Nothing here is synthesized; each row on screen
//! is a receipt the server actually wrote.
use crate::buffer::{EditOp, Motion};
use crate::vm::{Action, Ide};
use rce_mcp::items::Edit;
use serde_json::json;

pub fn populate(ide: &mut Ide, query: &str) -> Result<(), String> {
    let root = ide.server.ws.root.clone();
    // A manifest declaring two endpoints; one approved, one left pending.
    let manifest = root.join("rce.json");
    if !manifest.exists() {
        std::fs::write(
            &manifest,
            serde_json::to_string_pretty(&json!({"net": [
                {"host": "127.0.0.1", "port": 8787, "proto": "http", "reason": "local metrics sink"},
                {"host": "api.example.com", "port": 443, "proto": "tcp", "reason": "push counter snapshots"}
            ]}))
            .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    }
    ide.refresh();
    ide.apply(Action::Net {
        key: "http://127.0.0.1:8787".into(),
        allow: true,
    })?;
    let call = |ide: &mut Ide, tool: &str, args: serde_json::Value| {
        ide.server.call(tool, &args).map(|_| ()).map_err(|(_, m)| m)
    };
    call(ide, "rce_check", json!({}))?;
    call(ide, "rce_test", json!({}))?;
    call(
        ide,
        "emit",
        json!({"target": "python", "path": ide.editor.path.clone()}),
    )?;
    let path = ide.editor.path.clone();
    let ok = ide.propose(
        &path,
        "mei g2 (checker-guided)",
        "add a Reset event",
        serde_json::from_value::<Vec<Edit>>(json!([
            {"op": "insert_item", "after": "event Dec", "text": "event Reset"},
            {"op": "insert_item", "text": "step Counter on Reset -> 0"}
        ]))
        .map_err(|e| e.to_string())?,
    );
    let bad = ide.propose(
        &path,
        "acp agent (g4)",
        "step on an undeclared event",
        serde_json::from_value::<Vec<Edit>>(json!([
            {"op": "insert_item", "text": "step Counter on Nope -> n + 2"}
        ]))
        .map_err(|e| e.to_string())?,
    );
    let rej = ide.propose(
        &path,
        "mei g3",
        "rename Underflow",
        serde_json::from_value::<Vec<Edit>>(json!([
            {"op": "replace_item", "name": "error Underflow", "text": "error BelowZero"}
        ]))
        .map_err(|e| e.to_string())?,
    );
    ide.propose(
        &path,
        "mei g2",
        "document Inc",
        serde_json::from_value::<Vec<Edit>>(json!([
            {"op": "replace_item", "name": "event Inc", "text": "event Inc  -- add one"}
        ]))
        .map_err(|e| e.to_string())?,
    );
    ide.apply(Action::Accept(ok))?;
    ide.apply(Action::Accept(bad))?;
    ide.apply(Action::Decline(rej))?;
    ide.apply(Action::Recall(query.into()))?;
    // An unsaved edit in the buffer: a typo the language server flags,
    // with the typo selected.
    ide.apply(Action::Edit(EditOp::Move {
        to: Motion::DocEnd,
        select: false,
    }))?;
    for op in [
        EditOp::Newline,
        EditOp::Insert("step Counter on Incr -> n + 1".into()),
        EditOp::Move {
            to: Motion::WordLeft,
            select: false,
        },
        EditOp::Move {
            to: Motion::WordLeft,
            select: false,
        },
        EditOp::Move {
            to: Motion::WordLeft,
            select: false,
        },
        EditOp::Move {
            to: Motion::WordRight,
            select: true,
        },
    ] {
        ide.apply(Action::Edit(op))?;
    }
    Ok(())
}

/// The panels as JSON for the browser build (`WebIde::set_model`), plus the
/// editor's file and buffer.
pub fn export(ide: &Ide) -> serde_json::Value {
    let saved = ide
        .server
        .ws
        .resolve(&ide.editor.path, false)
        .ok()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .unwrap_or_default();
    json!({
        "proposals": ide.proposals,
        "receipts": ide.receipts,
        "memory": ide.memory,
        "net": ide.net,
        "editor": {"path": ide.editor.path, "saved": saved, "buffer": ide.buffer.text()},
    })
}
