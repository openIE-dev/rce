//! Lean-to-Rust link for multiparty global types with full merge and
//! asynchronous delivery (T28a): vectors computed by
//! `spec/lean/RCE/SessionAsyncVectors.lean` give random static global types
//! and their projections by `RCE.SessionAsync.proj` (null where full merge
//! fails); `mpst_async::proj` must agree on every role. Where every role
//! projects, the asynchronous runs from the projections with empty queues
//! never get stuck, and every receiver can take its queue head
//! (`SessionAsync.deadlock_free`). When `lake` is on PATH the vectors are
//! regenerated and must equal the checked-in file (RCE_BLESS=1 rewrites it).
use rce_spec_check::mpst_async as ma;
use serde_json::Value;
use std::path::Path;

fn spec() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn file() -> std::path::PathBuf {
    spec().join("checker/tests/vectors/session-async-lean.jsonl")
}

#[test]
fn lean_async_vectors_are_current() {
    let lake = std::env::var("HOME")
        .map(|h| Path::new(&h).join(".elan/bin/lake"))
        .ok()
        .filter(|p| p.exists())
        .unwrap_or_else(|| "lake".into());
    let Ok(out) = std::process::Command::new(lake)
        .args(["env", "lean", "--run", "RCE/SessionAsyncVectors.lean"])
        .current_dir(spec().join("lean"))
        .output()
    else {
        eprintln!("lake not on PATH: checking the checked-in vectors only");
        return;
    };
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let got = String::from_utf8(out.stdout).unwrap();
    if std::env::var("RCE_BLESS").is_ok() {
        std::fs::write(file(), &got).unwrap();
    }
    assert_eq!(
        std::fs::read_to_string(file()).unwrap(),
        got,
        "session-async-lean.jsonl is stale: RCE_BLESS=1"
    );
}

#[test]
fn checker_full_merge_equals_lean_and_runs_never_stick() {
    let text = std::fs::read_to_string(file()).unwrap();
    let (mut total, mut undefined, mut explored, mut wider) = (0, 0, 0, 0);
    for (n, line) in text.lines().enumerate() {
        let v: Value = serde_json::from_str(line).unwrap();
        let g =
            ma::parse_g(v["g"].as_str().unwrap()).unwrap_or_else(|| panic!("vector {n}: bad g"));
        assert!(ma::wf(&g), "vector {n}");
        let mut locals = vec![];
        for (r, want) in v["proj"].as_array().unwrap().iter().enumerate() {
            let got = ma::proj(&g, r);
            assert_eq!(
                got.as_ref().map(ma::show_l).as_deref(),
                want.as_str(),
                "vector {n} role {r}: {}",
                v["g"]
            );
            total += 1;
            undefined += want.is_null() as usize;
            locals.extend(got);
        }
        if locals.len() == 4 {
            // a role that only merges receives may accept more than one branch needs
            wider += locals.iter().any(|l| ma::show_l(l).contains("1 1")) as usize;
            match ma::explore(ma::start(locals), 200_000) {
                Ok(k) => explored += k,
                Err(k) => panic!("vector {n}: stuck at {k:?}"),
            }
        }
    }
    assert!(
        undefined > 0 && undefined < total && explored > 0 && wider > 0,
        "undefined {undefined} of {total}, explored {explored}"
    );
}

#[test]
fn async_explorer_finds_a_stuck_configuration() {
    // role 1 offers only label 1 from 0, but 0 sends label 0: stuck at the head
    let c = vec![
        ma::L::Snd(1, false, true, Box::new(ma::L::Done), Box::new(ma::L::Done)),
        ma::L::Rcv(0, true, false, Box::new(ma::L::Done), Box::new(ma::L::Done)),
    ];
    assert!(ma::explore(ma::start(c), 100).is_err());
    // two roles both waiting to receive first: stuck
    let c = vec![
        ma::L::Rcv(1, true, false, Box::new(ma::L::Done), Box::new(ma::L::Done)),
        ma::L::Rcv(0, true, false, Box::new(ma::L::Done), Box::new(ma::L::Done)),
    ];
    assert!(ma::explore(ma::start(c), 100).is_err());
}
