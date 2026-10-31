//! Memory tools through the MCP server, in-process and over stdio.
use ide_energy::sim::LinearSim;
use ide_receipt::chain::read_jsonl;
use ide_receipt::{GENESIS, Ledger, SigningKey, verify_chain};
use rce_mcp::{Server, ws::Workspace};
use rce_memory::service::Memory;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

const BANK: &str =
    include_str!("../../../../spec/examples/04_bank_account.rce");
const TREE: &str = include_str!(
    "../../../../spec/examples/39_structural_recursion.rce"
);

fn server(root: &std::path::Path) -> Server {
    std::fs::write(root.join("bank.rce"), BANK).unwrap();
    std::fs::write(root.join("tree.rce"), TREE).unwrap();
    let ws = Workspace::new(root).unwrap();
    let mem = Memory::open_default(&ws.root).unwrap();
    Server::new(
        ws,
        Ledger::new(SigningKey::from_bytes(&[1; 32])),
        LinearSim::default(),
        None,
    )
    .with_memory(mem)
}

fn call(s: &mut Server, name: &str, args: Value) -> Value {
    let r = s.handle(&json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}})).unwrap();
    assert!(r["result"].is_object(), "{r}");
    r["result"].clone()
}

#[test]
fn memory_tools_listed_only_when_enabled() {
    let t = tempfile::tempdir().unwrap();
    let mut s = server(t.path());
    let l = s
        .handle(&json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}))
        .unwrap();
    let names: Vec<&str> = l["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    for n in [
        "read_form",
        "recall",
        "remember",
        "forget",
        "repo_map",
        "memory_sync",
    ] {
        assert!(names.contains(&n), "{n} in {names:?}");
    }
    s.memory = None;
    let l = s
        .handle(&json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
        .unwrap();
    assert_eq!(l["result"]["tools"].as_array().unwrap().len(), 7);
    let r = s.handle(&json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "recall", "arguments": {"query": "x"}}})).unwrap();
    assert!(r["error"].is_object());
}

#[test]
fn remember_recall_forget_repo_map_in_process() {
    let t = tempfile::tempdir().unwrap();
    let mut s = server(t.path());
    let r = call(
        &mut s,
        "remember",
        json!({"subject": "tree_sum", "predicate": "complexity", "object": "linear", "provenance": "inferred"}),
    );
    assert_eq!(r["isError"], false, "{r}");
    let id = r["structuredContent"]["stored"].as_u64().unwrap();
    let hash = r["structuredContent"]["receipt"]["hash"]
        .as_str()
        .unwrap()
        .to_string();
    let r = call(&mut s, "recall", json!({"query": "tree_sum"}));
    let sc = &r["structuredContent"];
    assert_eq!(sc["found"], "exact");
    let fact = sc["hits"]
        .as_array()
        .unwrap()
        .iter()
        .find(|h| h["fact"].is_object())
        .unwrap();
    assert_eq!(fact["fact"]["receipt_id"], json!(hash));
    let r = call(&mut s, "recall", json!({"query": "who calls tree_sum"}));
    assert_eq!(r["structuredContent"]["found"], "graph");
    let r = call(&mut s, "recall", json!({"query": "qqqzz"}));
    assert_eq!(r["structuredContent"]["escalate"], true);
    let r = call(
        &mut s,
        "repo_map",
        json!({"focus": ["sample_sum"], "budget_chars": 300}),
    );
    assert!(
        r["structuredContent"]["map"]
            .as_str()
            .unwrap()
            .contains("tree.rce:")
    );
    let r = call(&mut s, "forget", json!({"id": id}));
    assert_eq!(r["isError"], false, "{r}");
    let r = call(&mut s, "forget", json!({"id": id}));
    assert_eq!(r["isError"], true, "second forget not performed");
    let r = call(
        &mut s,
        "remember",
        json!({"subject": "x", "predicate": "p", "object": "o", "note": true}),
    );
    assert_eq!(r["isError"], false);
    assert!(t.path().join("notes/x-p.md").exists());
    let r = call(
        &mut s,
        "remember",
        json!({"subject": "AGENTS", "predicate": "", "object": "o", "note": true}),
    );
    assert_eq!(r["isError"], true, "empty predicate not performed");
    assert!(!t.path().join("AGENTS.md").exists() && !t.path().join("notes/agents.md").exists());
    let r = call(&mut s, "memory_sync", json!({}));
    assert_eq!(r["structuredContent"]["unchanged"], 1, "{r}");
}

#[test]
fn memory_over_stdio_joins_the_receipt_chain() {
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("ws");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("tree.rce"), TREE).unwrap();
    let ledger = t.path().join("r.jsonl");
    let key = t.path().join("k");
    let mut child = Command::new(env!("CARGO_BIN_EXE_rce-mcp"))
        .args([
            "ledger",
            ledger.to_str().unwrap(),
            "key",
            key.to_str().unwrap(),
            root.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    let mut send = |id: u64, name: &str, args: Value| -> Value {
        let i = child.stdin.as_mut().unwrap();
        writeln!(i, "{}", json!({"jsonrpc": "2.0", "id": id, "method": "tools/call", "params": {"name": name, "arguments": args}})).unwrap();
        i.flush().unwrap();
        let mut line = String::new();
        out.read_line(&mut line).unwrap();
        serde_json::from_str(&line).unwrap()
    };
    let a = send(1, "rce_check", json!({}));
    assert_eq!(a["result"]["isError"], false, "{a}");
    let b = send(
        2,
        "remember",
        json!({"subject": "Tree", "predicate": "shape", "object": "binary"}),
    );
    assert_eq!(b["result"]["isError"], false, "{b}");
    let c = send(3, "recall", json!({"query": "Tree"}));
    assert_eq!(c["result"]["structuredContent"]["found"], "exact");
    drop(child.stdin.take());
    child.wait().unwrap();
    let rs = read_jsonl(&ledger).unwrap();
    assert_eq!(rs.len(), 3, "checker and memory receipts share one chain");
    let k: [u8; 32] = std::fs::read(&key).unwrap().try_into().unwrap();
    verify_chain(&rs, &SigningKey::from_bytes(&k).verifying_key(), GENESIS).unwrap();
    assert!(root.join(".rce-ide/memory.redb").exists());
}
