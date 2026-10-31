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

const COUNTER: &str = "intent \"a counter\"\n    example Inc, Inc, Dec -> 1\n    example Dec -> kept: Underflow\n\nstate Counter\n    n: u64 = 0\n\nevent Inc\nevent Dec\n\nerror Underflow\n\nstep Counter on Inc -> n + 1\nstep Counter on Dec -> try n.checked_sub(1) else Underflow\n";

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
    assert!(out.contains("A                 silent") && out.contains("B                 kept: Underflow"), "{out}");
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

fn fingerprint(depth: &str, file: &str) -> String {
    let (status, out) = rce(&["fingerprint", "--json", "--depth", depth, "--fuel", "500", file]);
    assert_eq!(status, 0, "{out}");
    let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
    assert_eq!(v["format"], "rce-fingerprint/1");
    v["fingerprint"].as_str().unwrap().to_string()
}

#[test]
fn fingerprint_is_shared_by_equivalent_programs_of_another_shape() {
    // BF-1: a renaming keeps the program hash and a reshaping (`1 + n` for
    // `n + 1`) changes it; both have the original's behaviour, so both have
    // its fingerprint
    let a = tmp("fa", COUNTER);
    let (_, nameless) = rce(&["fmt", "--nameless", &a]);
    let b = tmp("fb", &nameless);
    let c = tmp("fc", &COUNTER.replace("-> n + 1", "-> 1 + n"));
    let (_, ha) = rce(&["hash", &a]);
    let (_, hc) = rce(&["hash", &c]);
    assert_ne!(ha.lines().last(), hc.lines().last(), "the reshaping has another program hash");
    for d in ["1", "4", "8"] {
        let fa = fingerprint(d, &a);
        assert_eq!(fa, fingerprint(d, &b), "renaming, depth {d}");
        assert_eq!(fa, fingerprint(d, &c), "reshaping, depth {d}");
        assert_eq!(fa.len(), "sha256:".len() + 64);
    }
    assert_ne!(fingerprint("4", &a), fingerprint("8", &a), "the depth is part of the fingerprint");
}

#[test]
fn fingerprint_collision_at_a_short_depth_is_resolved_by_a_longer_one() {
    // checked_sub(2) differs only after Inc, Dec (a sequence of length 2)
    let a = tmp("fd", COUNTER);
    let b = tmp("fe", &COUNTER.replace("checked_sub(1)", "checked_sub(2)"));
    assert_eq!(fingerprint("1", &a), fingerprint("1", &b), "every sequence of length 1 has the same output");
    assert_ne!(fingerprint("2", &a), fingerprint("2", &b));
    let (status, _) = rce(&["equiv", "--fuel", "500", &a, &b]);
    assert_eq!(status, 1, "rce equiv resolves the depth-1 collision");
}

#[test]
fn fingerprint_reports_fuel_and_lists_many_files() {
    let a = tmp("ff", COUNTER);
    let (status, out) = rce(&["fingerprint", "--depth", "8", "--fuel", "3", &a]);
    assert_eq!(status, 1, "{out}");
    assert!(out.contains("not fingerprinted") && out.contains("fuel reached before depth 8"), "{out}");
    let (status, out) = rce(&["fingerprint", "--depth", "3", &a, &a]);
    assert_eq!(status, 0, "{out}");
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0], lines[1]);
    assert!(lines[0].starts_with("sha256:"));
    let (status, out) = rce(&["fingerprint", "--depth", "3", &a]);
    assert_eq!(status, 0);
    assert!(out.contains("event sequences of length 1 to 3 (14 sequences)"), "{out}");
}

#[test]
fn equiv_is_undecided_when_only_the_frontier_separates() {
    // A counter modulo 7 (explored whole) and an unbounded counter (cut at
    // the bound): both silent on every event, so equivalent; the cut graph
    // separates them only at its frontier, with no distinguishing sequence.
    // That is `undecided` (status 3), not `different` (10.11).
    let d = std::env::temp_dir().join(format!("rce-equiv-undecided-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let a = d.join("cyc.rce");
    let b = d.join("grow.rce");
    std::fs::write(&a, "state C\n    n: u64 = 0\n\nevent Tick\n\nstep C on Tick -> (n + 1) % 7\n").unwrap();
    std::fs::write(&b, "state C\n    n: u64 = 0\n\nevent Tick\n\nstep C on Tick -> n + 1\n").unwrap();
    let (a, b) = (a.to_string_lossy().into_owned(), b.to_string_lossy().into_owned());
    let (code, out) = rce(&["equiv", "--fuel", "50", &a, &b]);
    assert_eq!(code, 3, "{out}");
    assert!(out.contains("verdict             undecided"), "{out}");
    let (code, out) = rce(&["equiv", "--fuel", "50", "--json", &a, &b]);
    assert_eq!(code, 3, "{out}");
    let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
    assert_eq!(v["undecided"], true);
    assert_eq!(v["equivalent"], false);
    let (code, out) = rce(&["change", "--fuel", "50", &a, &b]);
    assert_eq!(code, 3, "{out}");
    assert!(out.contains("verdict             undecided"), "{out}");
}

// The explored alphabet of 10.11 harvests the literals a step compares an
// event field with. In notes_http a wrong method's status is reached only on
// a target the step matches ("/health", "/notes"), which the boundary set of
// text ("", "a") never produces.
const NOTES: &str = "../../programs/network/notes_http.rce";

#[test]
fn change_of_an_unexplored_status_is_a_meaning_change() {
    let src = std::fs::read_to_string(NOTES).unwrap();
    let a = tmp("notes_v1", &src);
    let b = tmp("notes_v2", &src.replace("405", "501"));
    assert_eq!(rce(&["check", &b]).0, 0, "the changed program, examples included, is accepted");
    let (status, out) = rce(&["change", &a, &b]);
    assert_eq!(status, 1, "{out}");
    assert!(out.contains("verdict             meaning change"), "{out}");
    assert!(out.contains("behaviour after               HttpRequest(0, \"\", \"/health\""), "{out}");
    assert!(out.contains("status: 405") && out.contains("status: 501"), "{out}");
}

#[test]
fn layout_and_reshaping_of_notes_http_keep_its_behaviour() {
    let src = std::fs::read_to_string(NOTES).unwrap();
    let a = tmp("notes_l1", &src);
    // layout only: a comment and blank lines; the program hash is unchanged
    let layout = tmp("notes_l2", &src.replace("state Server\n", "# the server's state\n\n\nstate Server\n"));
    let (status, out) = rce(&["change", &a, &layout]);
    assert_eq!(status, 0, "{out}");
    assert!(out.contains("verdict             same program"), "{out}");
    // a reshaping with the same behaviour: a new program hash, an erratum
    let reshaped = tmp("notes_l3", &src.replace("if c == \"fuel\" then", "if \"fuel\" == c then"));
    let (status, out) = rce(&["change", &a, &reshaped]);
    assert_eq!(status, 0, "{out}");
    assert!(out.contains("verdict             erratum  every export compatible"), "{out}");
}

#[test]
fn equiv_joins_the_literals_of_both_programs() {
    // `x < 5` harvests 4, 5, 6 and `x <= 4` harvests 3, 4, 5; both programs
    // are explored over the union, so they are compared on the same values
    let p = "state S\n    n: u64 = 0\n\nevent Put(x: u64)\n\nstep S on Put(x: x)\n    emit x < 5\n";
    let a = tmp("lt", p);
    let b = tmp("le", &p.replace("x < 5", "x <= 4"));
    let (status, out) = rce(&["equiv", &a, &b]);
    assert_eq!(status, 0, "{out}");
    let c = tmp("le3", &p.replace("x < 5", "x <= 3"));
    let (status, out) = rce(&["equiv", &a, &c]);
    assert_eq!(status, 1, "{out}");
    assert!(out.contains("Put(4)"), "{out}");
}
