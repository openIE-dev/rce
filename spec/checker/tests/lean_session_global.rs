//! Lean-to-Rust link for multiparty global types (T28g): vectors computed by
//! `spec/lean/RCE/SessionGlobalVectors.lean` give random global types and
//! their projections by `RCE.SessionGlobal.proj` (null where plain merge
//! fails); `mpst::proj` must agree on every role. For a global type of
//! messages only, the priorities of `SessionGlobal.global_prio` must pass
//! `session_prio::decide`. When `lake` is on PATH the vectors are regenerated
//! and must equal the checked-in file (RCE_BLESS=1 rewrites it).
use rce_spec_check::{mpst, session_prio};
use serde_json::Value;
use std::path::Path;

fn spec() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn file() -> std::path::PathBuf {
    spec().join("checker/tests/vectors/session-global-lean.jsonl")
}

#[test]
fn lean_global_vectors_are_current() {
    let lake = std::env::var("HOME")
        .map(|h| Path::new(&h).join(".elan/bin/lake"))
        .ok()
        .filter(|p| p.exists())
        .unwrap_or_else(|| "lake".into());
    let Ok(out) = std::process::Command::new(lake)
        .args(["env", "lean", "--run", "RCE/SessionGlobalVectors.lean"])
        .current_dir(spec().join("lean"))
        .output()
    else {
        eprintln!("lake not on PATH: checking the checked-in vectors only");
        return;
    };
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let got = String::from_utf8(out.stdout).unwrap();
    if std::env::var("RCE_BLESS").is_ok() {
        std::fs::write(file(), &got).unwrap();
    }
    assert_eq!(std::fs::read_to_string(file()).unwrap(), got, "session-global-lean.jsonl is stale: RCE_BLESS=1");
}

#[test]
fn checker_projection_equals_lean() {
    let text = std::fs::read_to_string(file()).unwrap();
    let (mut total, mut undefined, mut prio) = (0, 0, 0);
    for (n, line) in text.lines().enumerate() {
        let v: Value = serde_json::from_str(line).unwrap();
        let g = mpst::parse_g(v["g"].as_str().unwrap()).unwrap_or_else(|| panic!("vector {n}: bad g"));
        assert!(mpst::wf(&g), "vector {n}");
        for (r, want) in v["proj"].as_array().unwrap().iter().enumerate() {
            let got = mpst::proj(&g, r).map(|l| mpst::show_l(&l));
            assert_eq!(got.as_deref(), want.as_str(), "vector {n} role {r}: {}", v["g"]);
            total += 1;
            undefined += want.is_null() as usize;
        }
        if let Some(evs) = mpst::msgs(&g) {
            assert!(session_prio::decide(&mpst::msgs_net(&evs, 4)), "vector {n}: global_prio priorities do not hold");
            prio += 1;
        }
    }
    assert!(undefined > 0 && undefined < total && prio > 0, "undefined {undefined} of {total}, message-only {prio}");
}
