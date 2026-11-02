//! Mutation adequacy (spec 6.12, W-MUT-1 and W-MUT-2, `rce mutants`,
//! `spec/proposals/mutation-adequacy.md`).
//!
//! * `tests/mutants/vacuous.rce`: a vacuous example and a vacuous law
//!   require change under W-MUT-1 in `rce check` and in `rce mutants`, with
//!   the same reports; four mutants survive, each with a diff and the events
//!   after which `rce equiv` tells it apart.
//! * `tests/mutants/equivalent.rce`: `n + 0` becomes `n - 0`, a mutant
//!   `rce equiv` judges equivalent; it is counted apart, never as a
//!   survivor or in the score.
//! * `tests/mutants/unmeasured.rce`: an example that no edit can reach and
//!   a check that reads no definition are not measured, so not refused.
//! * determinism, and the bound: `bound N` judges the first N mutants, and
//!   an obligation whose reach goes beyond the bound is not refused.
use rce_spec_check::{cli, mutants};
use serde_json::Value;

fn rce(args: &[&str]) -> (i32, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (code, String::from_utf8(out).unwrap() + &String::from_utf8(err).unwrap())
}

const VACUOUS: &str = "tests/mutants/vacuous.rce";
const EQUIVALENT: &str = "tests/mutants/equivalent.rce";
const UNMEASURED: &str = "tests/mutants/unmeasured.rce";

fn json(args: &[&str]) -> Value {
    let mut a = vec!["mutants", "json"];
    a.extend_from_slice(args);
    let (_, out) = rce(&a);
    serde_json::from_str(&out).unwrap_or_else(|e| panic!("{e}: {out}"))
}

#[test]
fn vacuous_example_and_law_require_change() {
    let (code, out) = rce(&["check", VACUOUS]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("line 7: W-MUT-1 `example Reset -> n >= (0)` holds on each of the 1 mutant at the expressions its run evaluates"), "{out}");
    assert!(out.contains("line 9: W-MUT-1 `law small` holds on each of the 22 mutants of the steps it quantifies over"), "{out}");
    assert!(out.contains("[requires change]"), "{out}");
    // the other obligations kill a mutant each, so only these two
    assert_eq!(out.matches("W-MUT-1").count(), 2, "{out}");
}

#[test]
fn rce_mutants_reports_scores_vacuity_and_survivors() {
    let (code, out) = rce(&["mutants", VACUOUS]);
    assert_eq!(code, 1, "{out}");
    assert!(out.starts_with("mutants     23 of 23 single edits judged (bound 2000), 5 definitions\n"), "{out}");
    assert!(out.contains("score       19 killed of 23 non-equivalent (82.6%)\n"), "{out}");
    assert!(out.contains("obligation  line 5     kills   13 of   23  example Inc, Inc -> 2\n"), "{out}");
    assert!(out.contains("obligation  line 7     kills    0 of   23  example Reset -> n >= (0)\n"), "{out}");
    assert!(out.contains("obligation  line 8     kills    6 of    7  check double(2) == (4)\n"), "{out}");
    // a survivor: its edit, the canonical diff and the witness of rce equiv
    assert!(out.contains("survivor    W-MUT-2 in `step Counter on Inc`, line 26: `6` becomes `7`\n            -  25 step Counter on Inc -> (n + (1)) % (6)\n            +  25 step Counter on Inc -> (n + (1)) % (7)\n            after Inc(), Inc(), Inc(), Inc(), Inc(), Inc(), Show(): the program emits Shown(value: 0), the mutant emits Shown(value: 12)\n"), "{out}");
    assert!(out.contains("survivor    W-MUT-2 in `double`, line 24: `x * (2)` becomes `x + (2)`"), "{out}");
    assert_eq!(out.matches("survivor ").count(), 4, "{out}");
    assert!(out.ends_with("verdict     requires change tests/mutants/vacuous.rce: 2 vacuous obligations (W-MUT-1)\n"), "{out}");
}

#[test]
fn check_and_mutants_give_the_same_w_mut_1_reports() {
    for f in [VACUOUS, EQUIVALENT, UNMEASURED] {
        let (_, check) = rce(&["check", f]);
        let v = json(&[f]);
        for r in v["vacuous"].as_array().unwrap() {
            assert!(check.contains(r.as_str().unwrap()), "{f}: {r} not in {check}");
        }
        assert_eq!(check.matches("W-MUT-1").count(), v["vacuous"].as_array().unwrap().len(), "{f}: {check}");
    }
}

#[test]
fn the_mutant_set_is_deterministic() {
    let (a, b) = (rce(&["mutants", "json", VACUOUS]), rce(&["mutants", "json", VACUOUS]));
    assert_eq!(a, b);
    let (a, b) = (rce(&["mutants", EQUIVALENT]), rce(&["mutants", EQUIVALENT]));
    assert_eq!(a, b);
    let (a, b) = (rce(&["check", VACUOUS]), rce(&["check", VACUOUS]));
    assert_eq!(a, b);
    // the edits come in rung d's order: definitions in source order,
    // expressions in preorder
    let v = json(&[VACUOUS]);
    let lines: Vec<u64> = v["mutants"].as_array().unwrap().iter().map(|m| m["line"].as_u64().unwrap()).collect();
    let mut sorted = lines.clone();
    sorted.sort();
    assert_eq!(lines, sorted, "{lines:?}");
}

#[test]
fn an_equivalent_mutant_is_excluded_and_counted_apart() {
    let v = json(&[EQUIVALENT]);
    let ms = v["mutants"].as_array().unwrap();
    let eq: Vec<&Value> = ms.iter().filter(|m| m["fate"] == "equivalent").collect();
    assert_eq!(eq.len(), 1, "{v}");
    assert!(eq[0]["edit"].as_str().unwrap().contains("`n + (0)` becomes `n - (0)`"), "{v}");
    assert_eq!(v["equivalent"], 1);
    let (k, s) = (v["killed"].as_u64().unwrap(), v["survivors"].as_u64().unwrap());
    assert_eq!(v["non_equivalent"].as_u64().unwrap(), k + s);
    assert_eq!(k + s + 1, v["judged"].as_u64().unwrap(), "{v}");
    let (_, out) = rce(&["mutants", EQUIVALENT]);
    assert!(out.contains("1 equivalent (rce equiv)"), "{out}");
    assert!(!out.contains("becomes `n - (0)`"), "an equivalent mutant is not a survivor: {out}");
    assert!(out.contains(&format!("score       {k} killed of {} non-equivalent", k + s)), "{out}");
}

#[test]
fn an_obligation_the_edits_cannot_reach_is_not_refused() {
    let (code, out) = rce(&["check", UNMEASURED]);
    assert_eq!(code, 0, "SetVolume runs `-> p`, which no edit changes, and 1 + 1 reads no definition: {out}");
    let (_, out) = rce(&["mutants", UNMEASURED]);
    assert!(out.contains("check ((1) + (1)) == (2)  (reads no definition of the program)"), "{out}");
    let v = json(&[UNMEASURED]);
    let o = &v["obligations"][0];
    assert_eq!((o["line"].as_u64(), o["kills"].as_u64()), (Some(6), Some(0)));
}

#[test]
fn the_bound_cuts_the_mutant_set_in_order() {
    let full = json(&[VACUOUS]);
    let v = json(&["bound", "5", VACUOUS]);
    assert_eq!(v["bound"], 5);
    assert_eq!(v["judged"], 5);
    assert_eq!(v["single_edits"], full["single_edits"]);
    let (a, b) = (v["mutants"].as_array().unwrap(), full["mutants"].as_array().unwrap());
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x["edit"], y["edit"]);
    }
    // the law's reach holds every step, beyond the first five mutants, so
    // it is not refused at this bound
    assert!(!v["vacuous"].as_array().unwrap().iter().any(|r| r.as_str().unwrap().contains("law small")), "{v}");
    let law = v["obligations"].as_array().unwrap().iter().find(|o| o["kind"] == "law").unwrap();
    assert_eq!(law["reach_within_bound"], false);
    let (_, out) = rce(&["mutants", "bound", "5", VACUOUS]);
    assert!(out.starts_with("mutants     5 of 23 single edits judged (bound 5)"), "{out}");
    assert!(out.contains("(reach beyond the bound)"), "{out}");
    // rce check judges at most MUTANT_BOUND, rung d's default bound
    assert_eq!(mutants::MUTANT_BOUND, rce_spec_check::repair::DEFAULT_BOUND);
    assert_eq!(full["bound"], 2000);
}

#[test]
fn a_program_that_requires_change_is_not_judged() {
    let d = std::env::temp_dir().join(format!("rce-mutants-test-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join("bad.rce");
    std::fs::write(&p, "fn f(x: i64) -> i64 = x + 1\n\ncheck f(1) == 3\n").unwrap();
    let (code, out) = rce(&["mutants", p.to_str().unwrap()]);
    assert_eq!(code, 1);
    assert!(out.starts_with("requires change ") && out.contains("W-INT-1"), "{out}");
}
