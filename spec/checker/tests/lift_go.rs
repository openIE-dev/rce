//! The Go lift probe (spec/lift/go): the committed Form lifted from door.go
//! by lift_go.py (tree-sitter + a semantics table, no model) is accepted,
//! has the behaviour of the hand-written door, and the same shape_ctl; and
//! the committed report for Go the probe does not carry names a rule and a line.
use rce_spec_check::cli;

fn rce(args: &[&str]) -> (i32, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (code, String::from_utf8(out).unwrap())
}

#[test]
fn go_door_lifts_to_the_hand_written_door() {
    let lifted = "../lift/go/door.lifted.rce";
    let hand = "../examples/61_dim_minimal.rce";
    assert_eq!(rce(&["check", lifted]).0, 0);
    let (_, a) = rce(&["dim", "control", "json", lifted]);
    let (_, b) = rce(&["dim", "control", "json", hand]);
    let a: serde_json::Value = serde_json::from_str(&a).unwrap();
    let b: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert_eq!(a["behaviours"], 3);
    assert_eq!(a["control_states"], b["control_states"]);
    assert_eq!(a["shape_ctl"], b["shape_ctl"], "the lifted Go door and the hand-written door have one canonical control form");
}

/// fixtures/counter.go holds Go constructs the probe does not carry. Its
/// committed report (written by `lift_go.py lift counter.go`, and checked
/// byte for byte against the probe by spec/lift/go/test_lift_go.py) names
/// a rule of the probe and a line of the file for each of them, so nothing
/// the probe cannot carry is dropped without a word.
#[test]
fn go_constructs_the_probe_does_not_carry_require_change() {
    let probe = std::fs::read_to_string("../lift/go/lift_go.py").unwrap();
    let src = std::fs::read_to_string("../lift/go/fixtures/counter.go").unwrap();
    let report = std::fs::read_to_string("../lift/go/fixtures/counter.requires-change").unwrap();
    let mut rules = std::collections::BTreeSet::new();
    for line in report.lines() {
        let rest = line.strip_prefix("requires change counter.go: line ").unwrap_or_else(|| panic!("{line}"));
        let (n, rest) = rest.split_once(": ").unwrap();
        let n: usize = n.parse().unwrap();
        assert!(n >= 1 && n <= src.lines().count(), "{line}");
        let (rule, reason) = rest.split_once(": ").unwrap();
        assert!(rule.starts_with("L-GO-") && probe.contains(&format!("\"{rule}\":")), "{line}: not a rule of lift_go.py");
        assert!(!reason.is_empty(), "{line}");
        rules.insert(rule);
    }
    for want in ["L-GO-FUNC", "L-GO-FOR", "L-GO-SWITCH", "L-GO-IF", "L-GO-CALL", "L-GO-ASSERT", "L-GO-COLLECTION", "L-GO-IMPORT"] {
        assert!(rules.contains(want), "the report names {want}");
    }
}
