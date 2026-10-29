//! Lean-to-Rust link for linear endpoints (W-SES-2, W-SES-4): vectors
//! computed by `RCE.SessionLin.chkBody` (`spec/lean/RCE/SessionLinVectors.lean`)
//! are checked against the checker. `chkBody` is proved sound for the
//! linear typing judgment `RCE.SessionLin.Ty`, and `Ty` is proved safe
//! (no use of a gone endpoint, no unfinished close, every endpoint gone
//! or done at the end, on every path). Each vector is an RCE body; the
//! checker must accept exactly the bodies Lean accepts and, for each body
//! Lean refuses, report the rule Lean names. When `lake` is on PATH the
//! vectors are regenerated and must equal the checked-in file
//! (RCE_BLESS=1 rewrites it).
use rce_spec_check::check_program;
use serde_json::Value;
use std::path::Path;

fn spec() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// These vectors compare the verdict of one property (session linearity, priorities, the parallel
/// semantics) with the Lean model. Hygiene findings (unread names, dead stores, discarded values)
/// are a different property of the same one language, so they are set aside before comparing.
fn property_verdict(r: &Result<(), String>) -> Result<(), String> {
    let Err(e) = r else { return Ok(()) };
    let kept: Vec<&str> = e
        .split("; line ")
        .filter(|f| !["W-UNUSED-1", "W-UNUSED-2", "W-ASSIGN-1", "W-DROP-1"].iter().any(|h| f.contains(h)))
        .collect();
    if kept.is_empty() { Ok(()) } else { Err(kept.join("; line ")) }
}

fn file() -> std::path::PathBuf {
    spec().join("checker/tests/vectors/session-lin-lean.jsonl")
}

#[test]
fn lean_lin_vectors_are_current() {
    let lake = std::env::var("HOME")
        .map(|h| Path::new(&h).join(".elan/bin/lake"))
        .ok()
        .filter(|p| p.exists())
        .unwrap_or_else(|| "lake".into());
    let Ok(out) = std::process::Command::new(lake)
        .args(["env", "lean", "--run", "RCE/SessionLinVectors.lean"])
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
    assert_eq!(std::fs::read_to_string(file()).unwrap(), got, "session-lin-lean.jsonl is stale: RCE_BLESS=1");
}

#[test]
fn checker_linearity_equals_lean() {
    let text = std::fs::read_to_string(file()).unwrap();
    let (mut ok, mut changes) = (0, std::collections::BTreeMap::<String, usize>::new());
    let mut bad = vec![];
    for (n, line) in text.lines().enumerate() {
        let v: Value = serde_json::from_str(line).unwrap();
        let src = format!("# rce-version: 0.6\n{}", v["src"].as_str().unwrap());
        let want = v["expect"].as_str().unwrap();
        let r = check_program(&[(format!("lin{n}.rce"), src.clone())], false);
        match (&property_verdict(&r[0].1), want) {
            (Ok(()), "ok") => ok += 1,
            (Err(e), "ok") => bad.push(format!("vector {n}: Lean accepts, checker refuses: {e}\n{src}")),
            (Ok(()), w) => bad.push(format!("vector {n}: Lean refuses with {w}, checker accepts\n{src}")),
            (Err(e), w) if !e.contains(w) => bad.push(format!("vector {n}: Lean refuses with {w}, checker: {e}\n{src}")),
            (Err(_), w) => *changes.entry(w.to_string()).or_default() += 1,
        }
    }
    assert!(bad.is_empty(), "{} of {} vectors differ:\n{}", bad.len(), text.lines().count(), bad.join("\n\n"));
    assert!(ok > 0, "no accepted vector");
    for w in ["W-SES-1", "W-SES-2", "W-SES-4"] {
        assert!(changes.contains_key(w), "no vector refused with {w}");
    }
}
