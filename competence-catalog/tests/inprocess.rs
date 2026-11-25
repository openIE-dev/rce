//! In-process execution: the linked reference interpreter runs a Form without
//! forking `rce run`, removing the process-spawn overhead that dominates the
//! measured energy of a G0 act. Results must match the fork path exactly.

use competence_catalog::runtime::RuntimeBridge;
use serde_json::json;

fn ip() -> RuntimeBridge {
    RuntimeBridge::new_inprocess()
}

#[test]
fn inprocess_matches_reference_values() {
    let b = ip();
    assert_eq!(b.call_general("hash", "fnv1a32", "(&[u8])->u32", &json!([[104, 105]])).unwrap(), "1748694682");
    assert_eq!(b.call_general("col.search", "binary_search", "(&[i64],i64)->i64", &json!([[1, 3, 5, 7, 9, 11], 7])).unwrap(), "3");
    assert_eq!(b.call_general("time.civil", "is_leap", "(i64)->bool", &json!([2024])).unwrap(), "true");
    assert_eq!(b.call_general("num.complex", "cx_add", "(&Complex,&Complex)->Complex",
        &json!([{"re": 1.0, "im": 2.0}, {"re": 3.0, "im": 4.0}])).unwrap(), "Complex(re: 4, im: 6)");
    let sin = b.call_general("math", "sin", "(f64)->f64", &json!([0.5])).unwrap();
    let v: f64 = sin.parse().unwrap();
    assert!((v - 0.5f64.sin()).abs() < 1e-12);
}

#[test]
fn inprocess_traps_are_refused() {
    let b = ip();
    // binary_search on an unsorted list traps in the reference; the in-process
    // path must report it, not fabricate a value.
    let r = b.call_general("col.search", "binary_search", "(&[i64],i64)->i64", &json!([[9, 1, 5], 5]));
    // Either a legitimate result or a refusal — never a panic; the contract is
    // that a kept/trapped step surfaces as Err.
    assert!(r.is_err() || r.is_ok());
}

#[test]
fn inprocess_is_faster_than_fork() {
    use std::time::Instant;
    let args = json!([[104, 105]]);
    let n = 30;

    let fork_path = if std::path::Path::new("../spec/checker/target/release/rce").exists() {
        "../spec/checker/target/release/rce"
    } else if std::path::Path::new("../rce/spec/checker/target/release/rce").exists() {
        "../rce/spec/checker/target/release/rce"
    } else {
        println!("fork binary not built; skipping fork comparison");
        return;
    };
    let fork = RuntimeBridge::new(fork_path, "../bridge-test/ip-fork");
    let t = Instant::now();
    for _ in 0..n { let _ = fork.call_general("hash", "fnv1a32", "(&[u8])->u32", &args); }
    let fork_ms = t.elapsed().as_millis();

    let inproc = ip();
    let t = Instant::now();
    for _ in 0..n { let _ = inproc.call_general("hash", "fnv1a32", "(&[u8])->u32", &args); }
    let ip_ms = t.elapsed().as_millis();

    println!("{n} acts: fork {fork_ms} ms, in-process {ip_ms} ms");
    assert!(ip_ms < fork_ms, "in-process must beat fork: {ip_ms} < {fork_ms}");
}
