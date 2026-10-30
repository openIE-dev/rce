//! Lean-to-Rust link for W-SES-5 by priorities (T28p): vectors computed by
//! `spec/lean/RCE/SessionPrioVectors.lean` are checked against the checker.
//! Each vector is an RCE program (`main` and the workers it spawns over
//! session channels, cyclic session graphs included). Lean accepts it when
//! it has W-SES-5's tree shape or when `RCE.SessionPrio.decideP` finds
//! priorities, which `RCE.SessionPrio.decideP_sound` proves deadlock free
//! for every reachable state. The checker must accept exactly the programs
//! Lean accepts and report each of the others as requiring change with W-SES-5. When `lake` is on PATH
//! the vectors are regenerated and must equal the checked-in file
//! (RCE_BLESS=1 rewrites it).
use rce_spec_check::check_program;
use serde_json::Value;
use std::path::Path;

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

fn spec() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn file() -> std::path::PathBuf {
    spec().join("checker/tests/vectors/session-prio-lean.jsonl")
}

#[test]
fn lean_prio_vectors_are_current() {
    let lake = std::env::var("HOME")
        .map(|h| Path::new(&h).join(".elan/bin/lake"))
        .ok()
        .filter(|p| p.exists())
        .unwrap_or_else(|| "lake".into());
    let Ok(out) = std::process::Command::new(lake)
        .args(["env", "lean", "--run", "RCE/SessionPrioVectors.lean"])
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
    assert_eq!(std::fs::read_to_string(file()).unwrap(), got, "session-prio-lean.jsonl is stale: RCE_BLESS=1");
}

#[test]
fn checker_priorities_equal_lean() {
    let text = std::fs::read_to_string(file()).unwrap();
    let (mut ok, mut cyclic_ok, mut changes) = (0, 0, 0);
    let mut bad = vec![];
    for (n, line) in text.lines().enumerate() {
        let v: Value = serde_json::from_str(line).unwrap();
        let src = format!("# rce-version: 0.6\n{}", v["src"].as_str().unwrap());
        let want = v["expect"].as_str().unwrap();
        let r = check_program(&[(format!("prio{n}.rce"), src.clone())], false);
        match (&property_verdict(&r[0].1), want) {
            (Ok(()), "ok") => {
                ok += 1;
                if v["tree"] == Value::Bool(false) {
                    cyclic_ok += 1;
                }
            }
            (Err(e), "ok") => bad.push(format!("vector {n}: Lean accepts, checker requires change: {e}\n{src}")),
            (Ok(()), w) => bad.push(format!("vector {n}: Lean requires change with {w}, checker accepts\n{src}")),
            (Err(e), w) if !e.contains(w) || e.lines().any(|l| l.contains("W-") && !l.contains(w)) => {
                bad.push(format!("vector {n}: Lean requires change with {w}, checker: {e}\n{src}"))
            }
            (Err(_), _) => changes += 1,
        }
    }
    assert!(bad.is_empty(), "{} of {} vectors differ:\n{}", bad.len(), text.lines().count(), bad.join("\n\n"));
    assert!(ok > 0 && changes > 0 && cyclic_ok > 0, "ok {ok}, cyclic ok {cyclic_ok}, requires change {changes}");
}
