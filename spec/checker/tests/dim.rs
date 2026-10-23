//! `rce dim` (spec 6.10, 10.11): behaviours and redundancy on the three
//! dimension examples, which state their expected counts in their intent.
//! Not yet run in the session that wrote it (no toolchain there).
use rce_spec_check::cli;

fn rce(args: &[&str]) -> (i32, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (code, String::from_utf8(out).unwrap())
}

fn dim(file: &str) -> serde_json::Value {
    let (code, out) = rce(&["dim", "--json", file]);
    assert_eq!(code, 0, "{out}");
    serde_json::from_str(&out).unwrap()
}

#[test]
fn redundant_field_is_one_bit() {
    let d = dim("../examples/59_dim_redundant_field.rce");
    assert_eq!(d["states"], 8);
    assert_eq!(d["behaviours"], 4);
    assert_eq!(d["bound_reached"], false);
    assert!((d["redundancy_bits"].as_f64().unwrap() - 1.0).abs() < 1e-9);
}

#[test]
fn hidden_period_is_log2_of_three() {
    let d = dim("../examples/60_dim_hidden_period.rce");
    assert_eq!(d["states"], 6);
    assert_eq!(d["behaviours"], 2);
    assert!((d["redundancy_bits"].as_f64().unwrap() - 3f64.log2()).abs() < 1e-9);
}

#[test]
fn minimal_door_has_no_redundancy() {
    let d = dim("../examples/61_dim_minimal.rce");
    assert_eq!(d["states"], 3); // (open, locked) = (true, true) is unreachable
    assert_eq!(d["behaviours"], 3);
    assert_eq!(d["redundancy_bits"], 0.0);
}

#[test]
fn unbounded_counter_reports_a_lower_bound() {
    let (code, out) = rce(&["dim", "--fuel", "50", "../examples/01_counter.rce"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("bound reached: yes"), "{out}");
    assert!(out.contains("lower bound"), "{out}");
}

#[test]
fn control_is_reserved() {
    let (code, out) = rce(&["dim", "--control", "../examples/61_dim_minimal.rce"]);
    assert_eq!(code, 1);
    assert!(out.contains("reserved, not implemented"), "{out}");
}

#[test]
fn dim_is_deterministic() {
    let a = rce(&["dim", "--json", "../examples/59_dim_redundant_field.rce"]).1;
    let b = rce(&["dim", "--json", "../examples/59_dim_redundant_field.rce"]).1;
    assert_eq!(a, b);
}
