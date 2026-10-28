//! End to end: spawn the rce-mcp binary and talk MCP over its stdio.
use ide_receipt::chain::read_jsonl;
use ide_receipt::{GENESIS, SigningKey, verify_chain};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

const COUNTER: &str =
    include_str!("../../../../spec/examples/01_counter.rce");

struct Client {
    child: std::process::Child,
    out: BufReader<std::process::ChildStdout>,
    next: u64,
}

impl Client {
    fn send(&mut self, v: &Value) {
        let i = self.child.stdin.as_mut().unwrap();
        writeln!(i, "{v}").unwrap();
        i.flush().unwrap();
    }
    fn req(&mut self, method: &str, params: Value) -> Value {
        self.next += 1;
        self.send(&json!({"jsonrpc": "2.0", "id": self.next, "method": method, "params": params}));
        let mut line = String::new();
        self.out.read_line(&mut line).unwrap();
        let v: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["id"], json!(self.next));
        v
    }
    fn tool(&mut self, name: &str, args: Value) -> Value {
        let v = self.req("tools/call", json!({"name": name, "arguments": args}));
        let r = v["result"].clone();
        assert!(r.is_object(), "{v}");
        // text content mirrors structuredContent
        let text: Value = serde_json::from_str(r["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(text, r["structuredContent"]);
        r
    }
}

#[test]
fn mcp_over_stdio() {
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("ws");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("counter.rce"), COUNTER).unwrap();
    let ledger = t.path().join("receipts.jsonl");
    let key = t.path().join("k");
    let mut child = Command::new(env!("CARGO_BIN_EXE_rce-mcp"))
        .args([
            "--root",
            root.to_str().unwrap(),
            "--ledger",
            ledger.to_str().unwrap(),
            "--key",
            key.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let out = BufReader::new(child.stdout.take().unwrap());
    let mut c = Client {
        child,
        out,
        next: 0,
    };

    let init = c.req("initialize", json!({"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}));
    assert_eq!(init["result"]["protocolVersion"], "2025-11-25");
    assert_eq!(init["result"]["serverInfo"]["name"], "rce-mcp");
    c.send(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"})); // no reply expected
    assert_eq!(c.req("ping", json!({}))["result"], json!({}));

    let list = c.req("tools/list", json!({}));
    let names: Vec<&str> = list["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "read_form",
            "propose_edit",
            "rce_check",
            "rce_test",
            "graph_query",
            "graph_scip",
            "emit",
            "recall",
            "remember",
            "forget",
            "repo_map",
            "memory_sync"
        ]
    );

    // read_form: source, items, and the Form JSON
    let r = c.tool("read_form", json!({"path": "counter.rce"}));
    assert_eq!(r["isError"], false);
    assert!(r["structuredContent"]["form"].is_object());
    assert_eq!(r["structuredContent"]["items"][1], "state Counter");

    // check and test
    let r = c.tool("rce_check", json!({}));
    assert_eq!(r["structuredContent"]["passed"], 1);
    let r = c.tool("rce_test", json!({}));
    assert_eq!(r["structuredContent"]["files"][0]["examples"], 2);

    // A change the checker reports (`fit`): step on an undeclared event. Nothing is written.
    let r = c.tool(
        "propose_edit",
        json!({"path": "counter.rce", "edits": [
        {"op": "insert_item", "text": "step Counter on Reset -> 0"}]}),
    );
    assert_eq!(r["isError"], true);
    assert_eq!(r["structuredContent"]["applied"], false);
    assert!(
        !r["structuredContent"]["diagnostics"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        std::fs::read_to_string(root.join("counter.rce")).unwrap(),
        COUNTER
    );

    // The same change with its event declared and an example: admitted and written.
    let r = c.tool("propose_edit", json!({"path": "counter.rce", "edits": [
        {"op": "insert_item", "after": "event Dec", "text": "event Reset"},
        {"op": "insert_item", "text": "step Counter on Reset -> 0"},
        {"op": "replace_item", "name": "intent \"a counter that never goes negative\"", "text":
            "intent \"a counter that never goes negative\"\n    example Inc, Inc, Dec -> 1\n    example Dec -> kept: Underflow\n    example Inc, Inc, Reset -> 0"}]}));
    assert_eq!(r["isError"], false, "{r}");
    assert_eq!(r["structuredContent"]["applied"], true);
    let now = std::fs::read_to_string(root.join("counter.rce")).unwrap();
    assert!(now.contains("event Reset") && now.contains("on Reset -> 0"));
    let r = c.tool("rce_test", json!({"paths": ["counter.rce"]}));
    assert_eq!(r["structuredContent"]["files"][0]["examples"], 3);

    // A new file
    let r = c.tool("propose_edit", json!({"path": "hello.rce", "edits": [{"op": "set_file", "text": "fn main()\n    emit \"hi\"\n"}]}));
    assert_eq!(r["isError"], false, "{r}");

    // Policy: escapes, hooks and config files are not performed; the checker does not run.
    for p in [
        "../x.rce",
        ".git/hooks/pre-commit",
        "Cargo.toml",
        "/tmp/x.rce",
    ] {
        let r = c.tool(
            "propose_edit",
            json!({"path": p, "edits": [{"op": "set_file", "text": "event A"}]}),
        );
        assert_eq!(r["isError"], true, "{p}");
    }
    assert!(!root.join(".git").exists() && !root.join("Cargo.toml").exists());

    // emit: form, and (since checker 893bbf5) python; unknown targets reported.
    let r = c.tool("emit", json!({"path": "counter.rce", "target": "form"}));
    assert_eq!(r["isError"], false);
    let r = c.tool("emit", json!({"path": "counter.rce", "target": "python"}));
    assert_eq!(r["isError"], false, "{r}");
    assert!(!r["structuredContent"]["text"].as_str().unwrap().is_empty());
    let r = c.tool("emit", json!({"path": "counter.rce", "target": "cobol"}));
    assert_eq!(r["isError"], true);

    // Protocol errors
    assert_eq!(c.req("nope", json!({}))["error"]["code"], -32601);
    assert_eq!(
        c.req("tools/call", json!({"name": "rm_rf", "arguments": {}}))["error"]["code"],
        -32602
    );

    // Close stdin; the server exits cleanly.
    drop(c.child.stdin.take());
    assert!(c.child.wait().unwrap().success());

    // One receipt per tool call, chained, signed by the key on disk.
    let rs = read_jsonl(&ledger).unwrap();
    assert_eq!(rs.len(), 14); // 14 tool calls; protocol errors write none
    let kb: [u8; 32] = std::fs::read(&key).unwrap().try_into().unwrap();
    verify_chain(&rs, &SigningKey::from_bytes(&kb).verifying_key(), GENESIS).unwrap();
    // No simulator calibration was given: E_sim is absent, not invented.
    assert!(rs.iter().all(|r| r.body.draft.e_sim_j.is_none()));
    let fit = rs
        .iter()
        .filter(|r| matches!(r.body.draft.verdict, ide_receipt::Verdict::Fit { .. }))
        .count();
    assert_eq!(fit, 1);
}
