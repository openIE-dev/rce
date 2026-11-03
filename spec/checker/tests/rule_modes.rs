//! The modes of the altered rules and the attention line (spec 7.1, RM-1 to
//! RM-3, AC-1; 10.2, 10.6; proposal `spec/proposals/admission-receipts.md`).
//!
//! * RM-1: the modes table of 7.1 names exactly the altered rules, and no
//!   rule set gives a mode to a rule that requires change.
//! * RM-2: in observe mode `rce fmt` records exactly the rewrites that auto
//!   mode writes, and writes nothing; applying the recorded diffs in order
//!   gives the text auto mode writes. The report of a rule in observe mode
//!   does not block admission; a rule that is off is not judged.
//! * AC-1: every run of `rce check` and `rce fmt` ends with the attention
//!   line on standard error, and only findings that require change make the
//!   status non-zero.
use rce_spec_check::admission::{self, Attention, Mode, RuleSet};
use rce_spec_check::{alter, cli};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::PathBuf;

fn rce(args: &[&str]) -> (i32, String, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (code, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
}

fn scratch(name: &str, text: &str) -> String {
    let d = std::env::temp_dir().join(format!("rce-rule-modes-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p: PathBuf = d.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

/// The rule set in force with every altered rule in mode `m`.
fn all(m: Mode) -> RuleSet {
    alter::RULES.iter().fold(RuleSet::in_force().clone(), |s, r| s.with_mode(r, m))
}

/// The rule set in force with only `rule` judged among the altered rules, in mode `m`.
fn only(rule: &str, m: Mode) -> RuleSet {
    alter::RULES.iter().fold(RuleSet::in_force().clone(), |s, r| s.with_mode(r, if *r == rule { m } else { Mode::Off }))
}

fn fixtures() -> Vec<(String, String)> {
    let mut v: Vec<PathBuf> = std::fs::read_dir("../examples-0.9/altered")
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "rce") && !p.to_string_lossy().ends_with(".altered.rce"))
        .collect();
    v.sort();
    let mut out: Vec<(String, String)> = v.iter().map(|p| (p.to_string_lossy().into_owned(), std::fs::read_to_string(p).unwrap())).collect();
    // both rules on one text, several reports of each
    out.push((
        "both.rce".into(),
        "fn g(x: i64) -> i64\n    return 1\n\nfn h(a: i64, b: i64) -> i64\n    let c = 3\n    return a\n\nfn main()\n    g(1)\n    h(1, 2)\n    emit g(2).tostr()\n".into(),
    ));
    out
}

#[test]
fn the_modes_table_names_exactly_the_altered_rules() {
    let d: Value = serde_json::from_str(&std::fs::read_to_string("../data/spec-data.json").unwrap()).unwrap();
    let altered: BTreeSet<&str> = d["w_methods"]["table"].as_array().unwrap().iter().filter(|r| r["method"] == "altered").map(|r| r["rule"].as_str().unwrap()).collect();
    let moded: BTreeSet<&str> = d["w_modes"]["table"].as_array().expect("w_modes").iter().map(|r| r["rule"].as_str().unwrap()).collect();
    assert_eq!(altered, moded, "the modes table of 7.1 names exactly the altered rules");
    for r in d["w_modes"]["table"].as_array().unwrap() {
        assert!(Mode::parse(r["mode"].as_str().unwrap()).is_some(), "{r}");
        assert!(!r["reason"].as_str().unwrap().is_empty(), "{r}: no reason");
    }
    // the rule set compiled into the checker: today's altered rules are auto,
    // so what `rce fmt` writes is unchanged
    let s = RuleSet::in_force();
    for r in alter::RULES {
        assert_eq!(s.mode(r), Some(Mode::Auto), "{r}");
    }
    // a rule that requires change has no mode, and none can be given to it
    assert_eq!(s.mode("W-TY-1"), None);
    assert_eq!(s.with_mode("W-TY-1", Mode::Off), *s);
    assert_eq!(s.with_mode("W-INT-2", Mode::Off), *s);
    let row = |v: &str| RuleSet::from_table(&serde_json::from_str(v).unwrap());
    assert!(row(r#"[["W-TY-1","requires change","off"]]"#).is_none());
    assert!(row(r#"[["W-INT-2","recommended","observe"]]"#).is_none());
    assert!(row(r#"[["W-UNUSED-1","altered"]]"#).is_none());
    assert!(row(r#"[["W-UNUSED-1","altered","observe"],["W-TY-1","requires change"]]"#).is_some());
}

#[test]
fn observe_records_exactly_what_auto_writes() {
    for (p, src) in fixtures() {
        // every altered rule in observe, and each rule alone
        let mut sets = vec![(all(Mode::Observe), all(Mode::Auto))];
        for r in alter::RULES {
            sets.push((only(r, Mode::Observe), only(r, Mode::Auto)));
        }
        for (obs, auto) in sets {
            let o = alter::alter_modes(&obs, &src);
            let a = alter::alter_modes(&auto, &src);
            assert_eq!(o.text, src, "{p}: observe writes nothing");
            assert!(o.applied.is_empty(), "{p}");
            assert!(a.observed.is_empty(), "{p}");
            let seen: Vec<(usize, &str, &str)> = o.observed.iter().map(|x| (x.line, x.rule, x.what.as_str())).collect();
            let done: Vec<(usize, &str, &str)> = a.applied.iter().map(|x| (x.line, x.rule, x.what.as_str())).collect();
            assert_eq!(seen, done, "{p}: observe records exactly what auto writes");
            // applying the recorded diffs in order gives auto's text
            let mut t = src.clone();
            for x in &o.observed {
                assert!(!x.diff.is_empty(), "{p}: {x:?}");
                t = alter::apply_diff(&t, &x.diff).unwrap_or_else(|| panic!("{p}: {x:?} does not apply"));
            }
            assert_eq!(t, a.text, "{p}");
            // through `rce fmt`: the observe run writes no rewrite, prints each
            // with its diff, and its diffs then the layout give what auto writes
            let (lines, f) = cli::fmt_file(&obs, &p, &src, false);
            let (_, g) = cli::fmt_file(&auto, &p, &src, false);
            assert_eq!(f.attention.observed, done.len(), "{p}: {lines:?}");
            assert_eq!(f.attention.altered, 0, "{p}");
            assert_eq!(f.attention.requires_change, 0, "{p}: {lines:?}");
            assert_eq!(lines.iter().filter(|l| l.starts_with("observed ")).count(), done.len(), "{p}");
            assert!(f.write.is_none() || f.write == cli::fmt_text(&src).ok(), "{p}: observe changed the text");
            let auto_written = g.write.clone().unwrap_or_else(|| src.clone());
            assert_eq!(cli::fmt_text(&t).unwrap(), auto_written, "{p}");
            // deterministic
            assert_eq!(cli::fmt_file(&obs, &p, &src, false).0, lines, "{p}");
        }
    }
}

#[test]
fn the_observe_report_is_a_canonical_diff_with_its_rule() {
    let p = "../examples-0.9/altered/unused_param.rce";
    let src = std::fs::read_to_string(p).unwrap();
    let (lines, f) = cli::fmt_file(&all(Mode::Observe), p, &src, false);
    assert_eq!(
        lines,
        vec![
            format!("observed {p}: line 3: W-UNUSED-1 `b` is named `_b`"),
            "    -   3 fn f(a: i64, b: i64) -> i64".to_string(),
            "    +   3 fn f(a: i64, _b: i64) -> i64".to_string(),
        ]
    );
    assert_eq!(f.write, None);
    // with `check` as without: a recorded rewrite is never written and
    // never requires change
    let (lines2, f2) = cli::fmt_file(&all(Mode::Observe), p, &src, true);
    assert_eq!(lines2, lines);
    assert_eq!(f2.attention, Attention { requires_change: 0, recommended: 0, altered: 0, observed: 1 });
    // off: nothing
    let (lines3, f3) = cli::fmt_file(&all(Mode::Off), p, &src, true);
    assert!(lines3.is_empty(), "{lines3:?}");
    assert_eq!(f3.attention, Attention::default());
}

#[test]
fn observed_reports_do_not_block_admission() {
    let src = std::fs::read_to_string("../examples-0.9/altered/unused_param.rce").unwrap();
    let files = vec![("p.rce".to_string(), src.clone())];
    let one = |rules: &RuleSet| admission::program(rules, &files, true).remove(0).1.unwrap();
    // auto: requires change until `rce fmt` writes the rewrite
    let a = one(RuleSet::in_force());
    assert_eq!(a.errs.len(), 1, "{:?}", a.errs);
    assert!(a.errs[0].contains("W-UNUSED-1"), "{:?}", a.errs);
    // observe: admitted, the report recorded
    let o = one(&all(Mode::Observe));
    assert!(o.errs.is_empty(), "{:?}", o.errs);
    assert_eq!(o.observed.len(), 1);
    assert!(o.observed[0].contains("W-UNUSED-1"), "{:?}", o.observed);
    // off: not judged
    let f = one(&all(Mode::Off));
    assert!(f.errs.is_empty() && f.observed.is_empty());
    // a report that requires change stays one in every mode
    let bad = vec![("q.rce".to_string(), "fn f(b: i64) -> i64\n    return \"text\"\n".to_string())];
    for m in [Mode::Off, Mode::Observe, Mode::Auto] {
        let r = admission::program(&all(m), &bad, true).remove(0).1.unwrap();
        assert!(r.errs.iter().any(|e| e.contains("W-TY-1")), "{m:?}: {:?}", r.errs);
    }
    // every tool shares the one admission: `rce check`, `check_program` and
    // the language server's diagnostics agree under the rules in force
    assert!(rce_spec_check::check_one(&src).is_err());
    assert_eq!(cli::diagnostics(&src, &Default::default()).len(), 1);
}

#[test]
fn every_check_and_fmt_run_ends_with_the_attention_line() {
    // an admitted program with one recommended change (W-INT-2)
    let (code, out, err) = rce(&["check", "../examples/01_counter.rce"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("recommended ") && out.contains("W-INT-2"), "{out}");
    assert_eq!(err, "attention: 0 requires change, 1 recommended, 0 altered, 0 observed\n");
    // a program that requires change: one finding per report
    let two = scratch("two.rce", "fn f(b: i64) -> i64\n    let y = 1\n    return \"text\"\n");
    let (code, out, err) = rce(&["check", &two]);
    assert_eq!(code, 1, "{out}");
    let n = out.matches(" [requires change]").count() + out.matches(" [altered by rce fmt]").count();
    assert!(n >= 2, "{out}");
    assert_eq!(err, format!("attention: {n} requires change, 0 recommended, 0 altered, 0 observed\n"));
    // fmt check: a rewrite not yet written requires change
    let p = scratch("unused_param.rce", &std::fs::read_to_string("../examples-0.9/altered/unused_param.rce").unwrap());
    let (code, _, err) = rce(&["fmt", "check", &p]);
    assert_eq!((code, err.as_str()), (1, "attention: 1 requires change, 0 recommended, 0 altered, 0 observed\n"));
    // fmt: the rewrite is written, and nothing is left for a person
    let (code, out, err) = rce(&["fmt", &p]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(err, "attention: 0 requires change, 0 recommended, 1 altered, 0 observed\n");
    let (code, out, err) = rce(&["fmt", "check", &p]);
    assert_eq!((code, out.as_str(), err.as_str()), (0, "", "attention: 0 requires change, 0 recommended, 0 altered, 0 observed\n"));
    // only a finding that requires change makes the status non-zero
    assert_eq!(Attention { requires_change: 0, recommended: 3, altered: 2, observed: 5 }.status(), 0);
    assert_eq!(Attention { requires_change: 1, ..Default::default() }.status(), 1);
}
