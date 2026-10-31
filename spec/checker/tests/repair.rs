//! `rce repair` (spec 10.13): one fixture per rung of the exact-proposer
//! ladder, a program the ladder does not close, a program that already
//! complies, the bound of rung d, and the positional words.
use rce_spec_check::cli;

fn rce(args: &[&str]) -> (i32, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (code, String::from_utf8(out).unwrap() + &String::from_utf8(err).unwrap())
}

fn repair(args: &[&str]) -> (i32, serde_json::Value) {
    let mut a = vec!["repair", "json"];
    a.extend_from_slice(args);
    let (code, out) = rce(&a);
    (code, serde_json::from_str(&out).unwrap_or_else(|e| panic!("{e}: {out}")))
}

/// The repaired text is admitted by `rce check`, as the ladder said.
fn admitted(text: &str) {
    assert_eq!(rce_spec_check::check_one(text), Ok(()), "{text}");
}

#[test]
fn reuse_closes_with_a_definition_of_the_same_signature() {
    let (code, r) = repair(&["tests/repair/reuse.rce"]);
    assert_eq!(code, 0, "{r}");
    assert_eq!(r["rung"], "a");
    assert_eq!(r["model_calls"], 0);
    assert!(r["edit"].as_str().unwrap().contains("becomes the body of `twice`"), "{r}");
    assert!(r["diff"].as_array().unwrap().iter().any(|l| l.as_str().unwrap().contains("return n + n")), "{r}");
    admitted(r["repaired"].as_str().unwrap());
}

#[test]
fn lookup_by_type_swaps_a_callee_of_the_same_type() {
    let (code, r) = repair(&["tests/repair/lookup.rce"]);
    assert_eq!(code, 0, "{r}");
    assert_eq!(r["rung"], "b");
    assert!(r["edit"].as_str().unwrap().contains("`add(w, h)` becomes `mul(w, h)`"), "{r}");
    assert!(r["tried"]["a"].as_u64().unwrap() > 0, "rung a ran first: {r}");
    admitted(r["repaired"].as_str().unwrap());
}

#[test]
fn enumeration_flips_one_comparison() {
    let (code, r) = repair(&["tests/repair/enumerate.rce"]);
    assert_eq!(code, 0, "{r}");
    assert_eq!(r["rung"], "d");
    assert_eq!(r["tried"]["c"], 0);
    let edit = r["edit"].as_str().unwrap();
    assert!(edit.contains("in `larger`"), "{r}");
    admitted(r["repaired"].as_str().unwrap());
    // the same run gives the same answer, candidate for candidate
    let (_, r2) = repair(&["tests/repair/enumerate.rce"]);
    assert_eq!(r, r2);
}

#[test]
fn enumeration_respects_its_bound() {
    let (code, r) = repair(&["bound", "0", "tests/repair/enumerate.rce"]);
    assert_eq!(code, 1, "{r}");
    assert_eq!(r["closed"], false);
    assert_eq!(r["tried"]["d"], 0);
}

#[test]
fn a_string_the_edits_cannot_reach_is_not_closed() {
    let (code, r) = repair(&["tests/repair/notclosed.rce"]);
    assert_eq!(code, 1, "{r}");
    assert_eq!(r["closed"], false);
    assert!(r["rung"].is_null());
    assert!(r["verdict"].as_str().unwrap().contains("W-INT-1"), "{r}");
    // every attempt is logged with the checker's verdict
    assert!(r["attempts"].as_array().unwrap().iter().all(|a| a["verdict"].as_str().unwrap().starts_with("requires change")));
}

#[test]
fn a_program_that_complies_needs_nothing() {
    let (code, r) = repair(&["tests/repair/complies.rce"]);
    assert_eq!(code, 0);
    assert_eq!(r["rung"], "none");
    assert!(r["attempts"].as_array().unwrap().is_empty());
}

#[test]
fn text_report_and_words() {
    let (code, out) = rce(&["repair", "tests/repair/enumerate.rce"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.starts_with("rung      d bounded enumeration\n"), "{out}");
    assert!(out.contains("0 model calls"), "{out}");
    assert!(out.contains("skipped   c:"), "{out}");
    assert!(out.ends_with("closed tests/repair/enumerate.rce\n"), "{out}");
    let (code, out) = rce(&["repair", "bound", "x", "tests/repair/enumerate.rce"]);
    assert_eq!(code, 2, "{out}");
    let (code, out) = rce(&["repair", "--json", "tests/repair/enumerate.rce"]);
    assert_eq!(code, 2, "{out}");
}
