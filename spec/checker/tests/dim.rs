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
fn control_quotient_is_coarser_than_behaviour() {
    // the door: three control-distinct states (which steps fire differs)
    let (code, out) = rce(&["dim", "--control", "--json", "../examples/61_dim_minimal.rce"]);
    assert_eq!(code, 0, "{out}");
    let d: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(d["control_states"], 3);
    assert_eq!(d["shape_ctl"].as_str().unwrap().len(), 64);
    // the mod-4 counter and the mod-6 phase: every event fires the same step
    // with the same control outcome, so one control state each
    for f in ["../examples/59_dim_redundant_field.rce", "../examples/60_dim_hidden_period.rce"] {
        let (code, out) = rce(&["dim", "--control", "--json", f]);
        assert_eq!(code, 0, "{out}");
        let d: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(d["control_states"], 1, "{f}");
        assert!(d["control_states"].as_u64() <= d["behaviours"].as_u64(), "dim_c <= dim: {f}");
    }
}

#[test]
fn shape_ctl_is_name_free() {
    // T61: renaming leaves the control hash unchanged
    let (_, a) = rce(&["dim", "--control", "--json", "../examples/61_dim_minimal.rce"]);
    let (_, text) = rce(&["fmt", "--nameless", "../examples/61_dim_minimal.rce"]);
    let d = std::env::temp_dir().join(format!("rce-dim-ctl-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join("door.rce");
    std::fs::write(&p, text).unwrap();
    let (_, b) = rce(&["dim", "--control", "--json", p.to_str().unwrap()]);
    let a: serde_json::Value = serde_json::from_str(&a).unwrap();
    let b: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert_eq!(a["control_states"], b["control_states"]);
    assert_eq!(a["shape_ctl"], b["shape_ctl"], "T61: shape_ctl is name-free");
}

#[test]
fn dim_is_deterministic() {
    let a = rce(&["dim", "--json", "../examples/59_dim_redundant_field.rce"]).1;
    let b = rce(&["dim", "--json", "../examples/59_dim_redundant_field.rce"]).1;
    assert_eq!(a, b);
}

fn tmp(name: &str, text: &str) -> String {
    let p = std::env::temp_dir().join(format!("rce_equiv_{}_{name}.rce", std::process::id()));
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

const COUNTER: &str = "intent \"a counter\"\n    example Inc, Inc, Dec -> 1\n    example Dec -> refused: Underflow\n\nstate Counter\n    n: u64 = 0\n\nevent Inc\nevent Dec\n\nerror Underflow\n\nstep Counter on Inc -> n + 1\nstep Counter on Dec -> try n.checked_sub(1) else Underflow\n";

#[test]
fn equiv_is_name_free() {
    let a = tmp("a", COUNTER);
    let (_, nameless) = rce(&["fmt", "--nameless", &a]);
    let b = tmp("b", &nameless);
    let (status, out) = rce(&["equiv", "--fuel", "500", &a, &b]);
    assert_eq!(status, 0, "{out}");
    assert!(out.contains("verdict             equivalent"), "{out}");
}

#[test]
fn equiv_finds_the_shortest_distinguishing_sequence() {
    let a = tmp("c", COUNTER);
    let b = tmp("d", &COUNTER.replace("checked_sub(1)", "checked_sub(2)"));
    let (status, out) = rce(&["equiv", "--fuel", "500", &a, &b]);
    assert_eq!(status, 1, "{out}");
    assert!(out.contains("after               Inc(), Dec()"), "{out}");
    assert!(out.contains("A                 silent") && out.contains("B                 refused: Underflow"), "{out}");
    let (_, json) = rce(&["equiv", "--json", "--fuel", "500", &a, &b]);
    let v: serde_json::Value = serde_json::from_str(json.trim()).unwrap();
    assert_eq!(v["format"], "rce-equiv/1");
    assert_eq!(v["equivalent"], false);
    assert_eq!(v["witness"]["after"], serde_json::json!(["Inc()", "Dec()"]));
}

#[test]
fn change_classifies_erratum_and_meaning_change() {
    let a = tmp("e", COUNTER);
    let same = tmp("f", &COUNTER.replace("Underflow", "Below").replace("Counter", "Count"));
    let (status, out) = rce(&["change", "--fuel", "500", &a, &same]);
    assert_eq!(status, 0, "{out}");
    assert!(out.contains("verdict             same program"), "{out}");
    let bad = tmp("g", &COUNTER.replace("checked_sub(1)", "checked_sub(2)"));
    let (status, out) = rce(&["change", "--fuel", "500", &a, &bad]);
    assert_eq!(status, 1, "{out}");
    assert!(out.contains("verdict             meaning change"), "{out}");
    let same_beh = tmp("h", &COUNTER.replace("-> n + 1", "-> 1 + n"));
    let (status, out) = rce(&["change", "--fuel", "500", &a, &same_beh]);
    assert_eq!(status, 0, "{out}");
    assert!(out.contains("verdict             erratum"), "{out}");
}
