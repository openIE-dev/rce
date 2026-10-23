//! The shipped `rce` binary: plain `rce test` and the energy-budget gate.
use std::process::Command;

const COUNTER: &str = include_str!("../../../../spec/examples/01_counter.rce");

fn rce(args: &[&str], f: &std::path::Path) -> (i32, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_rce")).args(args).arg(f).output().unwrap();
    (o.status.code().unwrap_or(-1), String::from_utf8_lossy(&o.stdout).into_owned())
}

#[test]
fn energy_budget_gate() {
    let d = tempfile::tempdir().unwrap();
    let f = d.path().join("c.rce");
    std::fs::write(&f, COUNTER).unwrap();
    let (c, out) = rce(&["test"], &f);
    assert_eq!(c, 0, "{out}");
    assert!(out.contains("ok "));
    let (c, out) = rce(&["test", "--energy-budget", "1e9"], &f);
    assert_eq!(c, 0, "{out}");
    assert!(out.contains("energy ok") && out.contains("simulated"), "{out}");
    let (c, out) = rce(&["test", "--energy-budget", "1", "--measured", "2"], &f);
    assert_eq!(c, 1, "{out}");
    assert!(out.contains("energy OVER budget") && out.contains("measured"));
    let (c, out) = rce(&["test", "--energy-budget", "1e-300"], &f);
    assert_eq!(c, 1, "{out}");
}
