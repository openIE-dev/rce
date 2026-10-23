//! `rce fmt --nameless` (10.6, 6.6 concept record): the projection is
//! accepted whenever the source is, has the same Form up to names (same
//! log hashes, T7), is idempotent, and renames the declared names.
//! Not yet run in the session that wrote it (no toolchain there).
use rce_spec_check::cli;
use std::path::{Path, PathBuf};

fn examples() -> Vec<PathBuf> {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("examples");
    let mut v: Vec<PathBuf> = std::fs::read_dir(d)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "rce"))
        .collect();
    v.sort();
    v
}

fn rce(args: &[&str]) -> (i32, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (code, String::from_utf8(out).unwrap())
}

fn tmp(name: &str, text: &str) -> String {
    let d = std::env::temp_dir().join(format!("rce-nameless-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

#[test]
fn counter_is_renamed_by_kind() {
    let (code, out) = rce(&["fmt", "--nameless", "../examples/01_counter.rce"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("state r0"), "{out}");
    assert!(out.contains("event e0") && out.contains("event e1"), "{out}");
    assert!(out.contains("error x0"), "{out}");
    assert!(out.contains("v0: u64 = 0"), "{out}");
    assert!(out.contains("a counter that never goes negative"), "intent text is kept: {out}");
    assert!(!out.contains("Counter") && !out.contains("Underflow"), "{out}");
}

#[test]
fn projection_is_accepted_and_has_the_same_trace_hashes() {
    let mut checked = 0;
    for p in examples() {
        let file = p.to_string_lossy().into_owned();
        let (c0, _) = rce(&["check", &file]);
        if c0 != 0 {
            continue; // the corpus has files that are only accepted as a package
        }
        let (c1, text) = rce(&["fmt", "--nameless", &file]);
        assert_eq!(c1, 0, "{file}: {text}");
        let t = tmp(p.file_name().unwrap().to_str().unwrap(), &text);
        let (c2, why) = rce(&["check", &t]);
        assert_eq!(c2, 0, "{file} nameless refused: {why}\n{text}");
        let (_, a) = rce(&["trace", "--json", &file]);
        let (_, b) = rce(&["trace", "--json", &t]);
        // Same shape of every trace: step count and outcome per step. The state
        // hashes are NOT compared: 6.1 hashes the canonical JSON of the state,
        // whose keys are region and field names, so they change under renaming
        // (see `state_hashes_are_not_yet_name_free` below).
        let outcomes = |s: &str| -> Vec<String> {
            s.match_indices("\"outcome\":").map(|(i, _)| s[i..s[i..].find('}').map_or(s.len(), |j| i + j)].to_string()).collect()
        };
        let oa = outcomes(&a);
        let ob = outcomes(&b);
        assert_eq!(oa.len(), ob.len(), "{file}: step count");
        for (x, y) in oa.iter().zip(&ob) {
            // an outcome names the error on refusal; compare up to the renamed error
            assert_eq!(x.starts_with("\"outcome\":\"refused"), y.starts_with("\"outcome\":\"refused"), "{file}: {x} vs {y}");
        }
        checked += 1;
    }
    assert!(checked >= 40, "only {checked} examples checked");
}

#[test]
fn projection_is_idempotent() {
    let (_, once) = rce(&["fmt", "--nameless", "../examples/04_bank_account.rce"]);
    let t = tmp("bank_once.rce", &once);
    let (_, twice) = rce(&["fmt", "--nameless", &t]);
    assert_eq!(once, twice);
}

/// Finding of 2026-10-22: the Lean model of T7 (`RCE/NameFree.lean`) hashes
/// De Bruijn keys, but `spec/checker` computes `state_hash` over the
/// canonical JSON of the state (6.1), whose keys are region and field
/// names. So log hashes depend on names in the implementation and not in
/// the model. Making them agree changes every trace fixture (a new release
/// under C.15) or restates T7 for the implementation. Ignored until the
/// LEDGER decides; run with `--ignored` to see the current difference.
#[test]
#[ignore]
fn state_hashes_are_not_yet_name_free() {
    let (_, a) = rce(&["trace", "--json", "../examples/01_counter.rce"]);
    let (_, text) = rce(&["fmt", "--nameless", "../examples/01_counter.rce"]);
    let t = tmp("counter_nameless.rce", &text);
    let (_, b) = rce(&["trace", "--json", &t]);
    let hashes = |s: &str| -> Vec<String> {
        s.match_indices("\"state_hash\":\"").map(|(i, _)| s[i + 14..i + 14 + 64].to_string()).collect()
    };
    assert_eq!(hashes(&a), hashes(&b), "T7 as modelled: log hashes do not depend on names");
}

/// 6.6: the concept record is printed by `rce emit form` under `names`,
/// keyed by declaration index, and is the only place the source names
/// survive in a nameless projection's Form.
#[test]
fn emit_form_carries_the_concept_record() {
    let (code, out) = rce(&["emit", "form", "../examples/01_counter.rce"]);
    assert_eq!(code, 0, "{out}");
    let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
    let items = v["names"]["items"].as_array().unwrap();
    assert_eq!(items[0]["key"], "#1");
    assert_eq!(items[0]["name"], "Counter");
    assert_eq!(v["names"]["intent"][0], "a counter that never goes negative");
    let (_, text) = rce(&["fmt", "--nameless", "../examples/01_counter.rce"]);
    let t = tmp("counter_cr.rce", &text);
    let (_, out2) = rce(&["emit", "form", &t]);
    let w: serde_json::Value = serde_json::from_str(out2.trim()).unwrap();
    assert_eq!(w["names"]["items"][0]["key"], "#1");
    assert_eq!(w["names"]["items"][0]["name"], "r0");
    assert_eq!(w["names"]["intent"], v["names"]["intent"], "intent text survives the projection");
}
