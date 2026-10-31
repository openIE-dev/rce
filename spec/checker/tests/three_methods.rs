//! The three methods by which a static rule guides a program into Form
//! (spec 7.1): altered, recommended, requires change.
//!
//! * The table of 7.1 (`w_methods` in spec-data.json) gives every rule of
//!   section 7, Appendix C and Appendix F exactly one method, and names no
//!   other rule.
//! * The altered rows are the rules `rce fmt` rewrites (`alter::RULES`).
//! * For every fixture `X.rce` of `spec/examples-0.9/altered/`: `X` requires
//!   change under the rule on its second line, `rce check` names the method
//!   `altered by rce fmt`, `rce fmt X` gives `X.altered.rce`, that text is
//!   admitted, `rce fmt` writes nothing more on it, and the two texts have
//!   the same reference trace (the outputs, outcomes and state hashes of
//!   every const, example and `fn main`).
use rce_spec_check::{alter, cli};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

const METHODS: [&str; 3] = ["altered", "recommended", "requires change"];

fn spec_data() -> Value {
    serde_json::from_str(&std::fs::read_to_string("../data/spec-data.json").unwrap()).unwrap()
}

fn rce(args: &[&str]) -> (i32, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (code, String::from_utf8(out).unwrap())
}

#[test]
fn every_rule_has_exactly_one_method() {
    let d = spec_data();
    let mut rules: BTreeSet<String> = BTreeSet::new();
    for k in ["w_rules", "w_rules_06", "w_rules_09"] {
        let o = d[k]["rules"].as_object().unwrap_or_else(|| panic!("{k}"));
        assert!(!o.is_empty(), "{k}");
        rules.extend(o.keys().cloned());
    }
    let mut rows: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for r in d["w_methods"]["table"].as_array().expect("w_methods") {
        let (rule, m) = (r["rule"].as_str().unwrap(), r["method"].as_str().unwrap());
        assert!(METHODS.contains(&m), "{rule}: method `{m}`");
        assert!(!r["reason"].as_str().unwrap().is_empty(), "{rule}: no reason");
        rows.entry(rule.to_string()).or_default().push(m.to_string());
    }
    for r in &rules {
        let ms = rows.get(r).cloned().unwrap_or_default();
        assert_eq!(ms.len(), 1, "{r} has {} methods in 7.1: {ms:?}", ms.len());
    }
    for r in rows.keys() {
        assert!(rules.contains(r), "7.1 names {r}, which no rule list defines");
    }
}

#[test]
fn the_altered_rules_are_the_rules_fmt_rewrites() {
    let d = spec_data();
    let table: BTreeSet<&str> = d["w_methods"]["table"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["method"] == "altered")
        .map(|r| r["rule"].as_str().unwrap())
        .collect();
    let fmt: BTreeSet<&str> = alter::RULES.iter().copied().collect();
    assert_eq!(table, fmt);
    // the table compiled into the checker agrees
    for r in alter::RULES {
        assert_eq!(rce_spec_check::specread::rule_method(r), Some("altered"), "{r}");
    }
    assert_eq!(rce_spec_check::specread::rule_method("W-INT-2"), Some("recommended"));
    assert_eq!(rce_spec_check::specread::rule_method("W-TY-1"), Some("requires change"));
}

fn fixtures() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir("../examples-0.9/altered")
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "rce") && !p.to_string_lossy().ends_with(".altered.rce"))
        .collect();
    v.sort();
    v
}

/// The text `rce fmt` writes.
fn fmt(src: &str) -> String {
    cli::fmt_text(&alter::alter(src).0).unwrap()
}

#[test]
fn altered_fixtures_are_rewritten_admitted_idempotent_and_keep_their_meaning() {
    let fs = fixtures();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for p in &fs {
        let src = std::fs::read_to_string(p).unwrap();
        let want = src.lines().nth(1).and_then(|l| l.strip_prefix("# expect: ")).unwrap_or_else(|| panic!("{p:?}: no `# expect:` line")).trim();
        seen.insert(want.to_string());
        let e = rce_spec_check::check_one(&src).expect_err(&format!("{p:?} must require change"));
        assert!(e.contains(want), "{p:?}: {e}");
        let (code, out) = rce(&["check", p.to_str().unwrap()]);
        assert_eq!(code, 1, "{out}");
        assert!(out.contains(&format!("{want} ")) && out.contains("[altered by rce fmt]"), "{p:?}: {out}");
        let exp_path = p.with_extension("altered.rce");
        let expected = std::fs::read_to_string(&exp_path).unwrap_or_else(|_| panic!("{exp_path:?}"));
        // (a) the output is the expected text and is admitted
        let got = fmt(&src);
        assert_eq!(got, expected, "{p:?}");
        assert_eq!(rce_spec_check::check_one(&got), Ok(()), "{p:?}");
        // (b) idempotent
        assert_eq!(fmt(&got), got, "{p:?}");
        assert!(alter::alter(&got).1.is_empty(), "{p:?}");
        let (code, out) = rce(&["fmt", "check", exp_path.to_str().unwrap()]);
        assert_eq!((code, out.as_str()), (0, ""), "{exp_path:?}");
        // (c) the meaning is the same: the reference trace of the text as
        // written (parsed and linked; its reports are of naming and dropping
        // only) equals that of the altered text
        let before = rce_spec_check::one::trace::reference(&rce_spec_check::program(&src).unwrap());
        let after = rce_spec_check::one::trace::reference(&rce_spec_check::program(&got).unwrap());
        assert!(!after.is_empty(), "{p:?}: empty trace");
        assert_eq!(before, after, "{p:?}");
    }
    let fmt_rules: BTreeSet<String> = alter::RULES.iter().map(|s| s.to_string()).collect();
    assert_eq!(seen, fmt_rules, "every altered rule has a fixture");
}

#[test]
fn fmt_check_lists_each_rewrite_and_writes_nothing() {
    let p = "../examples-0.9/altered/unused_param.rce";
    let before = std::fs::read_to_string(p).unwrap();
    let (code, out) = rce(&["fmt", "check", p]);
    assert_eq!(code, 1);
    assert_eq!(out, format!("would alter {p}: line 3: W-UNUSED-1 `b` is named `_b`\n"));
    assert_eq!(std::fs::read_to_string(p).unwrap(), before);
}

#[test]
fn a_new_name_is_one_the_text_does_not_use() {
    // `_x` is taken, so the unread `x` is named `_x1`
    let src = "fn g(_x: i64) -> i64\n    return 1\n\nfn main()\n    let x = 1\n    emit g(2).tostr()\n";
    let (out, done) = alter::alter(src);
    assert_eq!(done.len(), 1, "{done:?}");
    assert!(out.contains("let _x1 = 1"), "{out}");
    assert_eq!(rce_spec_check::check_one(&out), Ok(()));
}

#[test]
fn requires_change_reports_name_their_method() {
    let p = std::env::temp_dir().join(format!("rce-three-methods-{}.rce", std::process::id()));
    std::fs::write(&p, "fn f() -> i64\n    let y = 1\n    return \"text\"\n").unwrap();
    let (code, out) = rce(&["check", p.to_str().unwrap()]);
    assert_eq!(code, 1);
    assert!(out.contains("W-TY-1") && out.contains("[requires change]"), "{out}");
    assert!(!out.contains("[] ") && out.trim_end().ends_with(']'), "{out}");
}
