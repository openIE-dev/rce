//! The Go lift probe (spec/lift/go): the committed Form lifted from door.go
//! by lift_go.py (tree-sitter + a semantics table, no model) is accepted,
//! has the behaviour of the hand-written door, and the same shape_ctl.
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
    let (_, a) = rce(&["dim", "--control", "--json", lifted]);
    let (_, b) = rce(&["dim", "--control", "--json", hand]);
    let a: serde_json::Value = serde_json::from_str(&a).unwrap();
    let b: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert_eq!(a["behaviours"], 3);
    assert_eq!(a["control_states"], b["control_states"]);
    assert_eq!(a["shape_ctl"], b["shape_ctl"], "the lifted Go door and the hand-written door have one canonical control form");
}
