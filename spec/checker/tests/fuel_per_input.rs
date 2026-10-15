//! HC-7a: fuel is per admitted input. Each input (one application of δ)
//! gets the full fuel of 6.7, on the host and in `rce trace` alike, so a
//! session never starves and trace and host give the same D.
//! Its own test binary: it sets the process-wide `interp::FUEL_LIMIT`.
use rce_spec_check::cli::trace;
use rce_spec_check::host::{Entry, Host, MemStore};
use rce_spec_check::interp::{FUEL, FUEL_LIMIT};
use rce_spec_check::parse::parse;
use serde_json::Value;
use std::sync::atomic::Ordering::SeqCst;

const SRC: &str = "state Counter\n    n: u64 = 0\n\nevent Inc\n\nstep Counter on Inc -> n + 1\n";

fn trace_steps(evs: &str) -> Vec<Value> {
    match trace("t.rce", SRC, Some(evs)) {
        Ok(t) => serde_json::from_str::<Value>(&t).unwrap()["traces"][0]["steps"].as_array().cloned().unwrap_or_default(),
        Err(_) => vec![],
    }
}

#[test]
fn trace_and_host_charge_fuel_per_input() {
    // smallest limit under which one Inc is applied
    let one = (1..10_000u64)
        .find(|&l| {
            FUEL_LIMIT.store(l, SeqCst);
            trace_steps("Inc").first().is_some_and(|s| s["outcome"] == "applied")
        })
        .expect("one Inc fits in some fuel limit");
    FUEL_LIMIT.store(one, SeqCst);
    let evs = "Inc, Inc, Inc, Inc, Inc, Inc, Inc, Inc";
    let steps = trace_steps(evs);
    assert_eq!(steps.len(), 8, "trace starved: a budget was shared across inputs");
    assert!(steps.iter().all(|s| s["outcome"] == "applied"));
    let want: Vec<String> = steps.iter().map(|s| s["state_hash"].as_str().unwrap().to_string()).collect();

    let m = parse(SRC).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    for _ in 0..8 {
        h.admit(SRC, "Inc").unwrap();
    }
    drop(h);
    let got: Vec<String> = st
        .lines
        .iter()
        .map(|l| Entry::parse_line(l).unwrap())
        .filter(|e| !e.is_host())
        .map(|e| e.state_hash)
        .collect();
    assert_eq!(got, want, "host and trace disagree under per-input fuel");

    // one unit less and the single input is refused in both
    FUEL_LIMIT.store(one - 1, SeqCst);
    assert!(trace_steps("Inc").first().is_none_or(|s| s["outcome"] != "applied"));
    FUEL_LIMIT.store(FUEL, SeqCst);
}
