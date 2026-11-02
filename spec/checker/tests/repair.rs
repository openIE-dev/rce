//! `rce repair` (spec 10.13): one fixture per rung of the exact-proposer
//! ladder, the renaming onto a known-good version, a program the ladder
//! does not close, a program that already complies, the bounds of rungs d
//! and e, and the positional words.
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
fn altered_rules_close_with_the_rewrites_of_rce_fmt() {
    let (code, r) = repair(&["tests/repair/altered.rce"]);
    assert_eq!(code, 0, "{r}");
    assert_eq!(r["rung"], "c");
    assert_eq!(r["model_calls"], 0);
    // rungs a and b ran first and closed nothing; rungs d and e were not reached
    assert!(r["tried"]["a"].as_u64().unwrap() + r["tried"]["b"].as_u64().unwrap() > 0, "{r}");
    assert_eq!(r["tried"]["c"], 1, "{r}");
    assert_eq!(r["tried"]["d"], 0, "{r}");
    assert_eq!(r["tried"]["e"], 0, "{r}");
    let edit = r["edit"].as_str().unwrap();
    assert!(edit.contains("W-UNUSED-1 `side` is named `_side`"), "{r}");
    assert!(edit.contains("W-DROP-1"), "{r}");
    let fixed = r["repaired"].as_str().unwrap();
    assert!(fixed.contains("let _side = r.w") && fixed.contains("let _ = doubled(&r)"), "{fixed}");
    admitted(fixed);
    // the candidate is the canonical text of what `rce fmt` writes
    let src = std::fs::read_to_string("tests/repair/altered.rce").unwrap();
    let (altered, _) = rce_spec_check::alter::alter(&src);
    let canon = rce_spec_check::unparse::unparse(&rce_spec_check::parse::parse(&altered).unwrap());
    assert_eq!(fixed, canon);
    // only rung c closes it: no attempt of rung a or b was admitted
    let attempts = r["attempts"].as_array().unwrap();
    assert!(attempts.iter().filter(|a| a["rung"] != "c").all(|a| a["verdict"].as_str().unwrap().starts_with("requires change")), "{r}");
    // deterministic
    let (_, r2) = repair(&["tests/repair/altered.rce"]);
    assert_eq!(r, r2);
    let (code, out) = rce(&["repair", "tests/repair/altered.rce"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.starts_with("rung      c altered rules\n"), "{out}");
}

#[test]
fn enumeration_flips_one_comparison() {
    let (code, r) = repair(&["tests/repair/enumerate.rce"]);
    assert_eq!(code, 0, "{r}");
    assert_eq!(r["rung"], "d");
    // nothing of an altered rule is reported, so rung c has no candidate
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
    assert_eq!(r["tried"]["e"], 0);
}

/// The original of tests/repair/rename.rce: one use of `h` was renamed to `w`.
const RENAME_ORIGINAL: &str = "intent \"the perimeter of a rectangle of width w and height h\"\n    check perimeter(2, 3) == 10\n    check perimeter(1, 5) == 12\n\nfn perimeter(w: u64, h: u64) -> u64\n    let side = w + h\n    return side * 2\n";

fn canonical(text: &str) -> String {
    rce_spec_check::unparse::unparse(&rce_spec_check::parse::parse(text).unwrap())
}

#[test]
fn renames_close_what_no_other_rung_closes() {
    let (code, r) = repair(&["tests/repair/rename.rce"]);
    assert_eq!(code, 0, "{r}");
    assert_eq!(r["rung"], "e");
    assert_eq!(r["model_calls"], 0);
    // the report names `h` (W-UNUSED-1), so `h` replaces a use of `w` first
    assert!(r["edit"].as_str().unwrap().contains("use 1 of `w` on the line becomes `h`"), "{r}");
    assert_eq!(r["tried"]["e"], 1, "{r}");
    // rung c ran first (it names `h` as `_h`, and the checks still fail),
    // and no single edit of rung d holds
    assert_eq!(r["tried"]["c"], 1, "{r}");
    assert!(r["tried"]["d"].as_u64().unwrap() > 0, "{r}");
    let attempts = r["attempts"].as_array().unwrap();
    assert!(attempts.iter().filter(|a| a["rung"] != "e").all(|a| a["verdict"].as_str().unwrap().starts_with("requires change")), "{r}");
    let fixed = r["repaired"].as_str().unwrap();
    assert!(fixed.contains("let side = h + w"), "{fixed}");
    admitted(fixed);
    // the same run gives the same answer, candidate for candidate
    let (_, r2) = repair(&["tests/repair/rename.rce"]);
    assert_eq!(r, r2);
    let (code, out) = rce(&["repair", "tests/repair/rename.rce"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.starts_with("rung      e renames\n"), "{out}");
    assert!(out.contains(", e 1 (candidates judged by the checker; 0 model calls)"), "{out}");
}

#[test]
fn renames_stay_within_their_bound() {
    let (code, r) = repair(&["bound", "0", "tests/repair/rename.rce"]);
    assert_eq!(code, 1, "{r}");
    assert_eq!(r["tried"]["d"], 0);
    assert_eq!(r["tried"]["e"], 0);
}

#[test]
fn renames_never_touch_what_a_candidate_is_judged_by() {
    // the check names values the body could be renamed to, and no rename of
    // the body closes it; every candidate of rung e edits the body only
    let dir = std::env::temp_dir().join(format!("rce-repair-spec-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("greet.rce");
    std::fs::write(&file, "intent \"a greeting\"\n    check greet(\"a\", \"b\") == \"hello\"\n\nfn greet(x: text, y: text) -> text\n    let z = x + y\n    return z + y\n").unwrap();
    let (code, r) = repair(&[file.to_str().unwrap()]);
    assert_eq!(code, 1, "{r}");
    let d: Vec<&str> = r["attempts"].as_array().unwrap().iter().filter(|a| a["rung"] == "e").map(|a| a["edit"].as_str().unwrap()).collect();
    assert!(d.len() >= 4, "{r}");
    for e in &d {
        assert!(e.contains("line 5:") || e.contains("line 6:"), "{e}");
    }
    assert_eq!(r["tried"]["e"].as_u64().unwrap() as usize, d.len());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_renaming_onto_a_known_good_version_restores_the_original() {
    // rename_good.rce has other names (`width`, `height`, `half`) and the
    // original's name-free projection: the renaming onto it keeps the names
    // of rename.rce and changes the one use that breaks it
    let (code, r) = repair(&["known", "tests/repair/rename_good.rce", "tests/repair/rename.rce"]);
    assert_eq!(code, 0, "{r}");
    assert_eq!(r["rung"], "e");
    let edit = r["edit"].as_str().unwrap();
    assert!(edit.contains("keeps the names of FILE: line 7: `w` becomes `h`"), "{r}");
    // judged before rung a, with no search
    assert_eq!(r["tried"]["a"], 0, "{r}");
    assert_eq!(r["tried"]["c"], 0, "{r}");
    assert_eq!(r["tried"]["d"], 0, "{r}");
    assert_eq!(r["tried"]["e"], 1, "{r}");
    let fixed = r["repaired"].as_str().unwrap();
    assert_eq!(fixed, canonical(RENAME_ORIGINAL), "restored, names and all");
    admitted(fixed);
    let (_, r2) = repair(&["known", "tests/repair/rename_good.rce", "tests/repair/rename.rce"]);
    assert_eq!(r, r2);
    // a GOOD that does not exist or does not parse is a usage error
    let (code, out) = rce(&["repair", "known", "tests/repair/none.rce", "tests/repair/rename.rce"]);
    assert_eq!(code, 2, "{out}");
}

#[test]
fn a_renaming_that_is_not_one_to_one_falls_back_to_the_known_good_text() {
    // FILE writes `x` where GOOD writes `a` and where it writes `b`, so no
    // one-to-one renaming keeps FILE's names: the candidate is GOOD's text
    let dir = std::env::temp_dir().join(format!("rce-repair-known-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let good = dir.join("good.rce");
    let file = dir.join("file.rce");
    std::fs::write(&good, "intent \"the sum\"\n    check add(2, 3) == 5\n\nfn add(a: u64, b: u64) -> u64\n    return a + b\n").unwrap();
    std::fs::write(&file, "intent \"the sum\"\n    check add(2, 3) == 5\n\nfn add(x: u64, x: u64) -> u64\n    return x + x\n").unwrap();
    let (g, f) = (good.to_str().unwrap(), file.to_str().unwrap());
    let (code, r) = repair(&["known", g, f]);
    assert_eq!(code, 0, "{r}");
    assert_eq!(r["rung"], "e");
    let edit = r["edit"].as_str().unwrap();
    assert!(edit.contains("`x` becomes `a`, `x` becomes `b`") && !edit.contains("keeps the names"), "{r}");
    assert_eq!(r["tried"]["a"], 0, "{r}");
    assert_eq!(r["tried"]["e"], 1, "{r}");
    assert_eq!(r["repaired"].as_str().unwrap(), canonical(&std::fs::read_to_string(&good).unwrap()));
    std::fs::remove_dir_all(&dir).unwrap();
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
    assert!(!out.contains("skipped"), "every rung runs: {out}");
    assert!(out.ends_with("closed tests/repair/enumerate.rce\n"), "{out}");
    let (code, out) = rce(&["repair", "bound", "x", "tests/repair/enumerate.rce"]);
    assert_eq!(code, 2, "{out}");
    let (code, out) = rce(&["repair", "--json", "tests/repair/enumerate.rce"]);
    assert_eq!(code, 2, "{out}");
}
