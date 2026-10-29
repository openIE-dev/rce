//! Lean-to-Rust link for T28 (sessions): vectors computed by
//! `RCE.SessionRec` through the encoding of `RCE.SessionEnc`
//! (`spec/lean/RCE/SessionVectors.lean`) are checked against the
//! checker's session automaton (`check06b`). For each vector: both
//! protocols parse; the checker's `dual` of A equals B (Lean `sdual`);
//! and along the joint run under `Red`, the checker's cursor for each
//! endpoint enables exactly the actions Lean's `PStep` enables, accepts
//! the action taken, and is at the end exactly when Lean's endpoint is
//! `done`. When `lake` is on PATH the vectors are regenerated from Lean
//! and must equal the checked-in file (RCE_BLESS=1 rewrites it).
use rce_spec_check::ast::{Item, ProtoStep};
use rce_spec_check::check06b::{dual, ProtoCursor};
use rce_spec_check::parse::parse;
use serde_json::Value;
use std::path::Path;

fn spec() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn lake() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let elan = Path::new(&home).join(".elan/bin/lake");
    if elan.exists() { Some(elan) } else { Some("lake".into()) }
}

#[test]
fn lean_vectors_are_current() {
    let file = spec().join("checker/tests/vectors/session-lean.jsonl");
    let Ok(out) = std::process::Command::new(lake().unwrap()).args(["env", "lean", "--run", "RCE/SessionVectors.lean"]).current_dir(spec().join("lean")).output() else {
        eprintln!("lake not on PATH: checking the checked-in vectors only");
        return;
    };
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let got = String::from_utf8(out.stdout).unwrap();
    if std::env::var("RCE_BLESS").is_ok() {
        std::fs::write(&file, &got).unwrap();
    }
    assert_eq!(std::fs::read_to_string(&file).unwrap(), got, "session-lean.jsonl is stale: RCE_BLESS=1");
}

fn protocols(a: &str, b: &str) -> (Vec<ProtoStep>, Vec<ProtoStep>) {
    let src = format!("# rce-version: 0.6\n{a}\n\n{b}\n");
    let m = parse(&src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    let mut ps = m.items.iter().filter_map(|i| match i {
        Item::Protocol { steps, .. } => Some(steps.clone()),
        _ => None,
    });
    (ps.next().unwrap(), ps.next().unwrap())
}

fn strs(v: &Value) -> Vec<String> {
    let mut x: Vec<String> = v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect();
    x.sort();
    x
}

fn dual_act(a: &str) -> String {
    match a.split_once(':') {
        Some(("choose", l)) => format!("offer:{l}"),
        Some(("offer", l)) => format!("choose:{l}"),
        _ if a == "send" => "recv".into(),
        _ => "send".into(),
    }
}

#[test]
fn checker_automaton_equals_lean() {
    let text = std::fs::read_to_string(spec().join("checker/tests/vectors/session-lean.jsonl")).unwrap();
    let mut n = 0;
    for line in text.lines() {
        let v: Value = serde_json::from_str(line).unwrap();
        let (pa, pb) = protocols(v["a"].as_str().unwrap(), v["b"].as_str().unwrap());
        assert_eq!(dual(&pa), pb, "vector {n}: checker dual differs from Lean sdual");
        let (mut ca, mut cb) = (ProtoCursor::new(&pa), ProtoCursor::new(&pb));
        for (i, st) in v["run"].as_array().unwrap().iter().enumerate() {
            assert_eq!(ca.enabled(), strs(&st["a"]), "vector {n} step {i}: A enabled");
            assert_eq!(cb.enabled(), strs(&st["b"]), "vector {n} step {i}: B enabled");
            assert!(!ca.at_end() && !cb.at_end(), "vector {n} step {i}: ended early");
            let t = st["take"].as_str().unwrap();
            assert!(ca.step(t), "vector {n} step {i}: A cannot take {t}");
            assert!(cb.step(&dual_act(t)), "vector {n} step {i}: B cannot take the dual of {t}");
        }
        let done = v["done"].as_bool().unwrap();
        assert_eq!(ca.at_end() && cb.at_end(), done, "vector {n}: end differs from Lean");
        if !done {
            // fuel ran out mid-protocol: both still have an action (progress)
            assert!(!ca.enabled().is_empty() && !cb.enabled().is_empty(), "vector {n}: stuck");
        }
        n += 1;
    }
    assert_eq!(n, 60);
}
