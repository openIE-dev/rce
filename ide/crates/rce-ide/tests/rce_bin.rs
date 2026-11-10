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
    let (c, out) = rce(&["test", "energy-budget", "1e9"], &f);
    assert_eq!(c, 0, "{out}");
    assert!(out.contains("energy simulated") && !out.contains("energy ok") && !out.contains("OVER"), "{out}");
    // a typed figure is a D.11 reading only with its provenance
    let prov = d.path().join("provenance.json");
    std::fs::write(&prov, r#"{"boundary":"rail:VDD","counter":"shunt meter energy register","end_unix_ns":2000000000,"idle_baseline_w":null,"interval_s":1.0,"machine":"bench host","sampler":"bench logger","start_unix_ns":1000000000,"wraps":0}"#).unwrap();
    let pv = prov.to_str().unwrap();
    let (c, out) = rce(&["test", "energy-budget", "1", "measured", "2", pv], &f);
    assert_eq!(c, 1, "{out}");
    assert!(out.contains(r#"energy receipt: {"amortization":{"kind":"per_call"},"basis":"measured","j":2.0,"provenance":{"boundary":"rail:VDD","#), "{out}");
    assert!(out.contains("energy OVER budget: 2.000e0 J > 1.000e0 J (measured: wall or shunt meter, with its provenance; basis measured)"), "{out}");
    // without one it is refused (D.11 rule 1): status 2, nothing run
    std::fs::write(&prov, "{}").unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_rce")).args(["test", "energy-budget", "1", "measured", "2", pv]).arg(&f).output().unwrap();
    assert_eq!(o.status.code(), Some(2));
    assert!(o.stdout.is_empty());
    assert!(String::from_utf8_lossy(&o.stderr).contains("D.11 rule 1"), "{}", String::from_utf8_lossy(&o.stderr));
    let (c, out) = rce(&["test", "energy-budget", "1e-300"], &f);
    assert_eq!(c, 0, "{out}");
    assert!(out.contains("energy simulated") && !out.contains("OVER"), "{out}");
    // positional-cli.md decision 2: a removed flag names its new spelling, exits 2, runs nothing
    let (c, out) = rce(&["test", "--energy-budget", "1e9"], &f);
    assert_eq!((c, out.as_str()), (2, ""));
    let o = Command::new(env!("CARGO_BIN_EXE_rce")).arg("--version").output().unwrap();
    assert_eq!(o.status.code(), Some(2));
    assert_eq!(String::from_utf8_lossy(&o.stderr), "--version was removed; write: rce version\n");
    let o = Command::new(env!("CARGO_BIN_EXE_rce")).arg("version").output().unwrap();
    assert!(o.status.success() && String::from_utf8_lossy(&o.stdout).contains("rce-ide"));
}
