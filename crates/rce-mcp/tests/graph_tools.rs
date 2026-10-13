//! Graph tools (graph_query, graph_scip) through the MCP server.
use ide_energy::sim::LinearSim;
use ide_receipt::{Ledger, SigningKey};
use rce_mcp::{Server, ws::Workspace};
use serde_json::{Value, json};

const TREE: &str = include_str!(
    "../../../vendor/rce-spec/examples/39_structural_recursion.rce"
);

fn call(s: &mut Server, name: &str, args: Value) -> Value {
    let r = s.handle(&json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}})).unwrap();
    r["result"].clone()
}

#[test]
fn query_and_scip() {
    let t = tempfile::tempdir().unwrap();
    std::fs::write(t.path().join("tree.rce"), TREE).unwrap();
    let ws = Workspace::new(t.path()).unwrap();
    let mut s = Server::new(ws, Ledger::new(SigningKey::from_bytes(&[1; 32])), LinearSim::default(), None);
    let r = call(&mut s, "graph_query", json!({"query": "kind fn"}));
    let text = r["content"][0]["text"].as_str().unwrap_or("").to_string();
    let v: Value = serde_json::from_str(&text).unwrap_or(r.clone());
    let n = v["nodes"].as_array().map(|a| a.len()).unwrap_or(0);
    assert!(n > 0, "{r}");
    let r = call(&mut s, "graph_query", json!({"query": "frob"}));
    assert_eq!(r["isError"], true, "{r}");
    let r = call(&mut s, "graph_scip", json!({}));
    let text = r["content"][0]["text"].as_str().unwrap_or("").to_string();
    let v: Value = serde_json::from_str(&text).unwrap_or(r.clone());
    assert!(v["bytes"].as_u64().unwrap() > 20, "{r}");
    assert!(v["base64"].as_str().unwrap().starts_with('C'));
}
