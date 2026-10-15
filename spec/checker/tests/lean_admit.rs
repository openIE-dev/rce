//! Lean-to-Rust link for T40: vectors computed by `RCE.Admit.window`
//! (`spec/lean/RCE/AdmitVectors.lean`) are checked against host.rs. Each
//! vector's inputs are offered in its arrival order (a retransmission
//! included) and the window's `host.Input` order must equal Lean's.
//! When `lake` is on PATH the vectors are regenerated from Lean and must
//! equal the checked-in file (RCE_BLESS=1 rewrites it).
use rce_spec_check::ast::{Arg, Expr};
use rce_spec_check::host::{Entry, Host, MemStore, Sourced};
use rce_spec_check::parse::parse;
use serde_json::Value;
use std::path::Path;

fn spec() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

#[test]
fn lean_vectors_are_current() {
    let file = spec().join("checker/tests/vectors/admit-lean.jsonl");
    let Ok(out) = std::process::Command::new("lake").args(["env", "lean", "--run", "RCE/AdmitVectors.lean"]).current_dir(spec().join("lean")).output() else {
        eprintln!("lake not on PATH: checking the checked-in vectors only");
        return;
    };
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let got = String::from_utf8(out.stdout).unwrap();
    if std::env::var("RCE_BLESS").is_ok() {
        std::fs::write(&file, &got).unwrap();
    }
    assert_eq!(std::fs::read_to_string(&file).unwrap(), got, "admit-lean.jsonl is stale: RCE_BLESS=1");
}

#[test]
fn host_window_order_equals_lean() {
    let src = std::fs::read_to_string(spec().join("checker/tests/host/priority.rce")).unwrap();
    let m = parse(&src).unwrap();
    let text = std::fs::read_to_string(spec().join("checker/tests/vectors/admit-lean.jsonl")).unwrap();
    let mut n = 0;
    for line in text.lines() {
        let v: Value = serde_json::from_str(line).unwrap();
        let mut st = MemStore::default();
        let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
        for x in v["arrival"].as_array().unwrap() {
            let f: Vec<u64> = x.as_array().unwrap().iter().map(|y| y.as_u64().unwrap()).collect();
            let (prio, s, q, pay) = (f[0], f[1], f[2], f[3]);
            let source = format!("s{s:02}");
            let inp = if prio == 200 {
                Sourced::new(&source, q, 0, "Alarm", vec![Arg { label: Some("n".into()), e: Expr::Int(pay as i128, 0) }])
            } else {
                Sourced::new(&source, q, 0, "Key", vec![Arg { label: Some("c".into()), e: Expr::Str(format!("p{pay:03}"), 0) }])
            };
            h.offer(inp).unwrap();
        }
        h.close_window(0).unwrap();
        drop(h);
        let got: Vec<(String, u64)> = st
            .lines
            .iter()
            .map(|l| Entry::parse_line(l).unwrap())
            .filter(|e| e.kind() == "host.Input")
            .map(|e| {
                let p: Value = serde_json::from_str(&e.event).unwrap();
                (p["payload"]["source"].as_str().unwrap().to_string(), p["payload"]["seq"].as_u64().unwrap())
            })
            .collect();
        let want: Vec<(String, u64)> = v["window"].as_array().unwrap().iter().map(|w| (format!("s{:02}", w[0].as_u64().unwrap()), w[1].as_u64().unwrap())).collect();
        assert_eq!(got, want, "vector {n}");
        n += 1;
    }
    assert_eq!(n, 40);
}
