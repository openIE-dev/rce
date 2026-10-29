//! The gate on `rce-form/1` (`spec/proposals/rce-form.md` §10 and §11) on the
//! fixtures of `spec/examples/form/`. Each fixture's expected outcome is in
//! `expect.json`, written by hand from the rules as the proposal states them
//! (`make.py` holds the table and builds the fixtures with an encoder of its
//! own); none comes from the checker. Each of C1 to C11 and FG1 to FG15 names
//! a fixture it applies and one it keeps, except where the README says why a
//! rule cannot keep a module on its own.
use rce_spec_check::cli;
use rce_spec_check::form::{self, Outcome, decode, ir};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples/form")
}

fn read(name: &str) -> Vec<u8> {
    std::fs::read(dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn expect() -> Value {
    serde_json::from_slice(&read("expect.json")).expect("expect.json is JSON")
}

fn run(f: &Value) -> Outcome {
    let form = read(f["form"].as_str().unwrap());
    let vectors = f["vectors"].as_str().map(read);
    form::check(&form, vectors.as_deref())
}

#[test]
fn every_fixture_has_its_expected_outcome() {
    let e = expect();
    assert_eq!(e["format"], "rce-form-fixtures/1");
    let fixtures = e["fixtures"].as_array().unwrap();
    assert!(fixtures.len() >= 30, "only {} fixtures", fixtures.len());
    for f in fixtures {
        let name = format!("{} {}", f["form"], f["vectors"]);
        match (run(f), f["outcome"].as_str().unwrap()) {
            (Outcome::Applied(_), "applied") => {}
            (Outcome::Kept(k), "kept") => {
                assert_eq!(k.rule, f["rule"].as_str().unwrap(), "{name}: {}", k.line());
                if let Some(why) = f["names"].as_str() {
                    assert!(k.reason.contains(why), "{name}: the reason does not name {why}: {}", k.line());
                }
            }
            (o, want) => panic!("{name}: expected {want}, got {o:?}"),
        }
    }
}

#[test]
fn every_rule_has_an_applied_and_a_kept_fixture() {
    let e = expect();
    // a fixture's name is its vectors file's, or its module file's when it has no vectors
    let outcome = |name: &str| -> Option<&Value> {
        e["fixtures"].as_array().unwrap().iter().find(|f| f["vectors"].as_str().or(f["form"].as_str()).unwrap().trim_end_matches(".json") == name)
    };
    let rules = e["rules"].as_object().unwrap();
    let mut names: Vec<String> = (1..=11).map(|i| format!("C{i}")).chain((1..=15).map(|i| format!("FG{i}"))).collect();
    names.sort();
    let mut have: Vec<String> = rules.keys().cloned().collect();
    have.sort();
    assert_eq!(have, names, "expect.json names each of C1 to C11 and FG1 to FG15");
    // the two rules that cannot keep a module on their own (README.md)
    let alone = ["C2", "FG9"];
    for (r, v) in rules {
        let a = v["applied"].as_str().unwrap_or_else(|| panic!("{r}: no applied fixture"));
        let fa = outcome(a).unwrap_or_else(|| panic!("{r}: {a} is not a fixture"));
        assert_eq!(fa["outcome"], "applied", "{r}: {a}");
        match v["kept"].as_str() {
            Some(k) => {
                let fk = outcome(k).unwrap_or_else(|| panic!("{r}: {k} is not a fixture"));
                assert_eq!(fk["outcome"], "kept", "{r}: {k}");
                // the rule itself is named: as the outcome's rule, or, for a canonical
                // rule, in the reason of FG1 (C1, C10) or of FG3 (C2 to C11)
                assert!(fk["rule"] == r.as_str() || fk["names"] == r.as_str(), "{r}: {k} does not name {r}");
            }
            None => assert!(alone.contains(&r.as_str()), "{r}: no kept fixture"),
        }
    }
}

/// FG3 on every applied fixture: decoding then encoding again gives the same
/// bytes, and the module key is the SHA-256 of those bytes.
#[test]
fn applied_fixtures_encode_to_their_own_bytes() {
    let e = expect();
    for f in e["fixtures"].as_array().unwrap() {
        if f["outcome"] != "applied" {
            continue;
        }
        let name = f["form"].as_str().unwrap();
        let bytes = read(name);
        let m = decode::decode(&bytes).unwrap_or_else(|k| panic!("{name}: {}", k.line()));
        assert_eq!(ir::encode(&m).as_bytes(), &bytes[..], "{name}");
        let key = rce_spec_check::interp::d_sha(std::str::from_utf8(&bytes).unwrap());
        assert_eq!(form::canon::module_key(&m), key, "{name}");
        assert_eq!(run(f), Outcome::Applied(key), "{name}");
        // every group key is the SHA-256 of the group's canonical JSON
        for (k, g) in &m.defs {
            assert_eq!(&ir::group_key(g), k, "{name}");
        }
    }
}

/// The module names the catalogue the checker carries by its key, the SHA-256
/// of `spec/data/form-catalogue.json`.
#[test]
fn the_catalogue_key_is_the_digest_of_the_catalogue_file() {
    let bytes = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/form-catalogue.json")).unwrap();
    assert_eq!(form::catalogue::key(), rce_spec_check::interp::d_sha(&bytes));
    let m: Value = serde_json::from_slice(&read("counter.json")).unwrap();
    assert_eq!(m["catalogue"], form::catalogue::key());
}

/// A changed byte never makes the gate fail by a panic: every one-digit change
/// of an applied fixture is applied or kept.
#[test]
fn a_changed_digit_is_applied_or_kept() {
    for name in ["counter.json", "loops.json", "group.json", "types.json"] {
        let bytes = read(name);
        let mut seen = 0;
        for i in 0..bytes.len() {
            if !bytes[i].is_ascii_digit() {
                continue;
            }
            let mut b = bytes.clone();
            b[i] = if b[i] == b'9' { b'0' } else { b[i] + 1 };
            let _ = form::check(&b, None);
            seen += 1;
        }
        assert!(seen > 20, "{name}: only {seen} digits changed");
    }
}

/// `rce form-check FORM [VECTORS]`: positional arguments only; exit 0 when the
/// module is applied, 1 when it is kept, 2 for a usage error.
#[test]
fn form_check_command() {
    let rce = |args: &[&str]| -> (i32, String) {
        let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let (mut out, mut err) = (vec![], vec![]);
        let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
        (code, String::from_utf8(out).unwrap())
    };
    let p = |n: &str| dir().join(n).to_string_lossy().into_owned();
    let (c, out) = rce(&["form-check", &p("counter.json")]);
    assert_eq!(c, 0);
    assert!(out.starts_with("applied "), "{out}");
    let (c, out) = rce(&["form-check", &p("counter.json"), &p("counter.vectors.json")]);
    assert_eq!((c, out.starts_with("applied ")), (0, true), "{out}");
    let (c, out) = rce(&["form-check", &p("counter-fg6.json")]);
    assert_eq!(c, 1);
    assert!(out.starts_with("kept FG6 "), "{out}");
    let (c, out) = rce(&["form-check", &p("counter.json"), &p("counter.wrong-vectors.json")]);
    assert_eq!(c, 1);
    assert!(out.starts_with("kept FG15 "), "{out}");
    assert_eq!(rce(&["form-check"]).0, 2);
    assert_eq!(rce(&["form-check", &p("no-such-file.json")]).0, 2);
    assert_eq!(rce(&["form-check", &p("counter.json"), &p("counter.vectors.json"), "x"]).0, 2);
}
