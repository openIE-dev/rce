//! Admission receipts (spec 10.15, AR-1 to AR-5, RM-3; proposal
//! `spec/proposals/admission-receipts.md`).
//!
//! * AR-1: a receipt records the name-free program hash, the checker, the
//!   rule set with its hash, every obligation with its fuel or its counts
//!   and result, what the mutation gate measured, and the attestations.
//! * AR-2: the same program, rules and checker give the same bytes; a
//!   renaming or a layout change of the program does too.
//! * AR-3, AR-4: revalidation recomputes admission and reports each case:
//!   the receipt holds, the program changed, the rule set changed (with the
//!   rules added, removed and changed, and whether the program is still
//!   admitted), the checker changed, or the record differs.
//! * AR-5: a receipt that disagrees with itself is reported as invalid.
//! * RM-3: promoting an altered rule changes the rule-set hash, so a receipt
//!   written before the promotion reports it.
use rce_spec_check::admission::{Mode, RuleSet, Row};
use rce_spec_check::receipt::{self, Outcome};
use rce_spec_check::{apidiff, cli, interp, parse};
use serde_json::{Value, json};
use std::path::PathBuf;

fn rce(args: &[&str]) -> (i32, String, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (code, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
}

fn scratch(name: &str, text: &str) -> String {
    let d = std::env::temp_dir().join(format!("rce-receipts-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p: PathBuf = d.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

fn one(name: &str, text: &str) -> Vec<(String, String)> {
    vec![(name.to_string(), text.to_string())]
}

/// `tests/law/deposit.rce` with its undecided law attested at the current cover.
fn deposit() -> String {
    let src = std::fs::read_to_string("tests/law/deposit.rce").unwrap();
    let (_, out, _) = rce(&["check", "tests/law/deposit.rce"]);
    let i = out.find("WHO;YYYY-MM-DD;").expect("the report names the cover") + "WHO;YYYY-MM-DD;".len();
    let cover = &out[i..i + 64];
    src.replace("        every step -> total >= old(total)\n", &format!("        every step -> total >= old(total)\n        attest \"A. Person;2026-11-01;{cover}\"\n"))
}

fn keys() -> String {
    std::fs::read_to_string("tests/law/keys.rce").unwrap()
}

/// The receipt `v` with its rule table, rule-set hash and binding made
/// consistent again after an edit, as canonical bytes.
fn reseal(mut v: Value) -> String {
    let rules = RuleSet::from_table(&v["rules"]["table"]).unwrap();
    v["rules"]["sha256"] = json!(rules.sha256());
    v["binding"] = json!(format!("{};{};{}", v["program"].as_str().unwrap(), rules.sha256(), v["checker"]["commit"].as_str().unwrap()));
    interp::canon_json(&v) + "\n"
}

#[test]
fn a_receipt_records_program_rules_checker_obligations_and_attestations() {
    let rules = RuleSet::in_force();
    // a law attested, an example
    let src = deposit();
    let r = receipt::write(rules, &one("d.rce", &src)).unwrap();
    let v: Value = serde_json::from_str(&r).unwrap();
    assert_eq!(v["format"], "rce-admission-receipt/1");
    assert_eq!(v["program"], json!(apidiff::program_hash(&[parse::parse(&src).unwrap()])));
    assert_eq!(v["checker"], receipt::checker());
    assert_eq!(v["checker"]["commit"], json!(rce_spec_check::GIT_COMMIT));
    assert_eq!(v["rules"]["sha256"], json!(rules.sha256()));
    let table = v["rules"]["table"].as_array().unwrap();
    assert_eq!(table.len(), rules.rows.len());
    assert!(table.contains(&json!(["W-UNUSED-1", "altered", "auto"])), "{table:?}");
    assert!(table.contains(&json!(["W-TY-1", "requires change"])));
    assert!(table.contains(&json!(["W-INT-2", "recommended"])));
    let obs = v["obligations"].as_array().unwrap();
    assert_eq!(obs.len(), 2, "{obs:?}");
    let ex = obs.iter().find(|o| o["kind"] == "example").unwrap();
    assert_eq!(ex["result"], "holds");
    assert!(ex["fuel"].as_u64().unwrap() > 0, "{ex}");
    assert_eq!(ex["sha256"].as_str().unwrap().len(), 64);
    assert!(ex["mutation"]["reach"].as_u64().is_some(), "{ex}");
    let law = obs.iter().find(|o| o["kind"] == "law").unwrap();
    assert_eq!(law["result"], "attested");
    let at = law["attestation"].as_str().unwrap();
    assert!(at.starts_with("A. Person;2026-11-01;") && at.len() == "A. Person;2026-11-01;".len() + 64, "{at}");
    assert_eq!(v["attestations"], json!([at]));
    assert_eq!(v["binding"], json!(format!("{};{};{}", v["program"].as_str().unwrap(), rules.sha256(), rce_spec_check::GIT_COMMIT)));
    assert!(v["mutation"]["total"].as_u64().unwrap() >= v["mutation"]["judged"].as_u64().unwrap());
    // laws that hold: their states and steps (6.11), and a check's fuel
    let k = keys() + "\ncheck 2 + 2 == 4\n";
    let v: Value = serde_json::from_str(&receipt::write(rules, &one("k.rce", &k)).unwrap()).unwrap();
    let laws: Vec<&Value> = v["obligations"].as_array().unwrap().iter().filter(|o| o["kind"] == "law").collect();
    assert_eq!(laws.len(), 4, "{v}");
    for l in laws {
        assert_eq!(l["result"], "holds");
        assert!(l["states"].as_u64().unwrap() > 0 && l["steps"].as_u64().unwrap() > 0, "{l}");
    }
    assert!(v["obligations"].as_array().unwrap().iter().any(|o| o["kind"] == "check" && o["fuel"].is_u64()), "{v}");
    assert_eq!(v["attestations"], json!([]));
    // a program that requires change has no receipt
    let e = receipt::write(rules, &one("bad.rce", "fn f() -> i64\n    return \"text\"\n")).unwrap_err();
    assert!(e.contains("W-TY-1"), "{e}");
}

#[test]
fn a_receipt_is_byte_stable() {
    let rules = RuleSet::in_force();
    let src = deposit();
    let a = receipt::write(rules, &one("d.rce", &src)).unwrap();
    let b = receipt::write(rules, &one("other-name.rce", &src)).unwrap();
    assert_eq!(a, b, "the file name is not in the receipt");
    assert!(a.ends_with("}\n") && !a[..a.len() - 1].contains('\n'));
    assert_eq!(interp::canon_json(&serde_json::from_str(&a).unwrap()) + "\n", a);
    // the receipt `rce check write` writes has the same bytes
    let p = scratch("d.rce", &src);
    let out_path = scratch("d.receipt", "");
    let (code, out, _) = rce(&["check", "write", &out_path, &p]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(std::fs::read_to_string(&out_path).unwrap(), a);
    assert!(out.contains(&format!("receipt {out_path}: {}", rce_spec_check::manifest::sha256_hex(a.as_bytes()))), "{out}");
    // a layout change and a comment: the same receipt
    let laid = src.replace("fn add(", "# a comment\n\nfn add(");
    assert_eq!(receipt::write(rules, &one("d.rce", &laid)).unwrap(), a);
    // the name-free projection (10.6): the same receipt
    let nameless = rce_spec_check::nameless::nameless(&keys()).unwrap();
    assert_eq!(receipt::write(rules, &one("k.rce", &nameless)).unwrap(), receipt::write(rules, &one("k.rce", &keys())).unwrap());
    // a changed obligation changes it
    let other = src.replace("-> total == 12", "-> total >= 12");
    assert_ne!(receipt::write(rules, &one("d.rce", &other)).unwrap(), a);
}

#[test]
fn revalidation_reports_each_case() {
    let rules = RuleSet::in_force();
    let src = deposit();
    let files = one("d.rce", &src);
    let r = receipt::write(rules, &files).unwrap();
    let v: Value = serde_json::from_str(&r).unwrap();

    // 1. the receipt holds
    assert_eq!(receipt::revalidate(rules, &r, &files), Ok(Outcome::Holds));

    // 2. the program changed, still admitted (a definition no obligation covers)
    let other = src.replace("fn unused(a: u64) -> u64\n    return a\n", "fn unused(a: u64) -> u64\n    return a + 1\n");
    match receipt::revalidate(rules, &r, &one("d.rce", &other)).unwrap() {
        Outcome::ProgramChanged { old, new, admitted } => {
            assert_eq!(old, v["program"].as_str().unwrap());
            assert_ne!(old, new);
            assert!(admitted);
        }
        o => panic!("{o:?}"),
    }
    // ... and no longer admitted (the attestation lapses)
    let lapsed = src.replace("    return a + b\n", "    return a + b + 0\n");
    assert!(matches!(receipt::revalidate(rules, &r, &one("d.rce", &lapsed)), Ok(Outcome::ProgramChanged { admitted: false, .. })));

    // 3. the rule set changed: a rule added, one removed and one whose method changed
    let mut old = rules.clone();
    old.rows.remove("W-TY-1");
    old.rows.insert("W-OLD-1".into(), Row { method: "requires change".into(), mode: None });
    old.rows.insert("W-INT-2".into(), Row { method: "requires change".into(), mode: None });
    let r_old = receipt::write(&old, &files).unwrap();
    match receipt::revalidate(rules, &r_old, &files).unwrap() {
        Outcome::RulesChanged { old: o, new, added, removed, changed, admitted } => {
            assert_eq!(o, old.sha256());
            assert_eq!(new, rules.sha256());
            assert_eq!(added, vec!["W-TY-1 requires change"]);
            assert_eq!(removed, vec!["W-OLD-1 requires change"]);
            assert_eq!(changed, vec!["W-INT-2 requires change -> recommended"]);
            assert!(admitted);
        }
        o => panic!("{o:?}"),
    }
    // ... a promotion (RM-3), under which a program admitted before now requires change
    let unread = std::fs::read_to_string("../examples-0.9/altered/unused_param.rce").unwrap();
    let observing = rules.with_mode("W-UNUSED-1", Mode::Observe);
    let r_obs = receipt::write(&observing, &one("u.rce", &unread)).unwrap();
    let vo: Value = serde_json::from_str(&r_obs).unwrap();
    assert_eq!(vo["observed"], json!({"W-UNUSED-1": 1}));
    assert!(receipt::write(rules, &one("u.rce", &unread)).is_err(), "auto: requires change");
    assert_eq!(
        receipt::revalidate(rules, &r_obs, &one("u.rce", &unread)),
        Ok(Outcome::RulesChanged {
            old: observing.sha256(),
            new: rules.sha256(),
            added: vec![],
            removed: vec![],
            changed: vec!["W-UNUSED-1 altered observe -> altered auto".into()],
            admitted: false
        })
    );

    // 4. the checker changed: the same record otherwise
    let mut vc = v.clone();
    vc["checker"]["commit"] = json!("000000000000");
    let r_chk = reseal(vc);
    assert!(matches!(receipt::revalidate(rules, &r_chk, &files), Ok(Outcome::CheckerChanged { admitted: true, same: true, .. })));
    // ... and a record that differs too
    let mut vd = v.clone();
    vd["checker"]["draft"] = json!("0.8");
    vd["mutation"]["judged"] = json!(0);
    assert!(matches!(receipt::revalidate(rules, &reseal(vd), &files), Ok(Outcome::CheckerChanged { admitted: true, same: false, .. })));

    // 5. program, rules and checker the receipt's, and a record that is not
    let mut vf = v.clone();
    vf["obligations"][0]["result"] = json!("fails");
    assert_eq!(receipt::revalidate(rules, &reseal(vf), &files), Ok(Outcome::Differs { admitted: true }));

    // the command line: one line per case, status 0 only when the receipt holds
    let p = scratch("d.rce", &src);
    let rp = scratch("d.receipt", &r);
    let (code, out, err) = rce(&["check", "receipt", &rp, &p]);
    assert_eq!(code, 0, "{out}");
    assert!(out.starts_with(&format!("ok {p}\n")), "{out}");
    assert!(out.contains(&format!("receipt holds {rp}: program {}", v["program"].as_str().unwrap())), "{out}");
    assert_eq!(err, "attention: 0 requires change, 0 recommended, 0 altered, 0 observed\n");
    let po = scratch("other.rce", &other);
    let (code, out, err) = rce(&["check", "receipt", &rp, &po]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains(&format!("receipt does not hold {rp}: the program changed (program ")), "{out}");
    assert!(out.contains("under the rules in force the program is admitted"), "{out}");
    assert_eq!(err, "attention: 1 requires change, 0 recommended, 0 altered, 0 observed\n");
    let pu = scratch("u.rce", &unread);
    let ro = scratch("u.receipt", &r_obs);
    let (code, out, _) = rce(&["check", "receipt", &ro, &pu]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("the rule set changed (rules "), "{out}");
    assert!(out.contains("under the new set the program requires change\n    changed W-UNUSED-1 altered observe -> altered auto\n"), "{out}");
    let rc = scratch("c.receipt", &r_chk);
    let (code, out, _) = rce(&["check", "receipt", &rc, &p]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("the checker changed (checker 000000000000 (draft 0.9"), "{out}");
    assert!(out.contains("the program is admitted, and the record it gives is the receipt's"), "{out}");
}

#[test]
fn revalidation_recomputes_admission() {
    // a receipt never admits: given with a program that requires change, the
    // verdict is the recomputed one, whatever the receipt says
    let rules = RuleSet::in_force();
    let src = deposit();
    let r = receipt::write(rules, &one("d.rce", &src)).unwrap();
    let rp = scratch("r.receipt", &r);
    let lapsed = scratch("lapsed.rce", &src.replace("    return a + b\n", "    return a + b + 0\n"));
    let (code, out, _) = rce(&["check", "receipt", &rp, &lapsed]);
    assert_eq!(code, 1, "{out}");
    assert!(out.starts_with(&format!("requires change {lapsed}: ")) && out.contains("W-LAW-1"), "{out}");
    // a receipt edited to say what the program does not do is not believed
    let mut v: Value = serde_json::from_str(&r).unwrap();
    v["obligations"].as_array_mut().unwrap().push(json!({"kind": "check", "result": "holds", "fuel": 1, "sha256": "0".repeat(64)}));
    assert_eq!(receipt::revalidate(rules, &reseal(v), &one("d.rce", &src)), Ok(Outcome::Differs { admitted: true }));
}

#[test]
fn an_invalid_receipt_is_reported() {
    let rules = RuleSet::in_force();
    let src = deposit();
    let files = one("d.rce", &src);
    let r = receipt::write(rules, &files).unwrap();
    let v: Value = serde_json::from_str(&r).unwrap();
    let mut extra = v.clone();
    extra["note"] = json!("x");
    let mut wrong_hash = v.clone();
    wrong_hash["rules"]["table"][0][1] = json!("recommended");
    let mut wrong_binding = v.clone();
    wrong_binding["binding"] = json!("a;b;c");
    let mut bad_row = v.clone();
    bad_row["rules"]["table"][0] = json!(["W-TY-1", "requires change", "off"]);
    let cases = [
        "not json".to_string(),
        "{}\n".to_string(),
        interp::canon_json(&extra) + "\n",
        serde_json::to_string_pretty(&v).unwrap(),
        r.trim_end().to_string(),
        interp::canon_json(&wrong_hash) + "\n",
        interp::canon_json(&wrong_binding) + "\n",
        interp::canon_json(&bad_row) + "\n",
    ];
    for c in &cases {
        let e = receipt::revalidate(rules, c, &files).unwrap_err();
        assert!(e.starts_with("AR-5 "), "{e}");
    }
    let p = scratch("d.rce", &src);
    let rp = scratch("bad.receipt", &cases[5]);
    let (code, out, err) = rce(&["check", "receipt", &rp, &p]);
    assert_eq!(code, 2, "{out}");
    assert!(err.contains(&format!("{rp}: AR-5 the receipt's rule-set hash is not the hash of the rules it lists")), "{err}");
}

#[test]
fn a_promotion_changes_the_rule_set_hash() {
    let s = RuleSet::in_force();
    let observing = s.with_mode("W-UNUSED-1", Mode::Observe);
    let off = s.with_mode("W-UNUSED-1", Mode::Off);
    assert_ne!(s.sha256(), observing.sha256());
    assert_ne!(s.sha256(), off.sha256());
    assert_ne!(observing.sha256(), off.sha256());
    // promoting back gives the set in force, and its hash
    assert_eq!(observing.with_mode("W-UNUSED-1", Mode::Auto).sha256(), s.sha256());
    // a rule that requires change has no mode to change (RM-1)
    assert_eq!(s.with_mode("W-TY-1", Mode::Off).sha256(), s.sha256());
    // the diff names the change of mode
    assert_eq!(s.diff(&observing), (vec![], vec![], vec!["W-UNUSED-1 altered observe -> altered auto".to_string()]));
    // the hash is that of the canonical table under rce-rules/1
    let want = rce_spec_check::manifest::sha256_hex(format!("rce-rules/1\n{}", interp::canon_json(&s.table())).as_bytes());
    assert_eq!(s.sha256(), want);
}
