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
        assert_eq!(c2, 0, "{file} nameless requires change: {why}\n{text}");
        let (_, a) = rce(&["trace", "--json", &file]);
        let (_, b) = rce(&["trace", "--json", &t]);
        // Same shape of every trace: step count and outcome per step, and the
        // same state hashes (T7: 6.1 hashes the positional form of the state).
        let hashes = |s: &str| -> Vec<String> {
            s.match_indices("\"state_hash\":\"").map(|(i, _)| s[i + 14..i + 14 + 64].to_string()).collect()
        };
        assert_eq!(hashes(&a), hashes(&b), "{file}: state hashes differ under renaming");
        let outcomes = |s: &str| -> Vec<String> {
            s.match_indices("\"outcome\":").map(|(i, _)| s[i..s[i..].find('}').map_or(s.len(), |j| i + j)].to_string()).collect()
        };
        let oa = outcomes(&a);
        let ob = outcomes(&b);
        assert_eq!(oa.len(), ob.len(), "{file}: step count");
        for (x, y) in oa.iter().zip(&ob) {
            // the outcome of a kept step names the error; compare up to the renamed error
            assert_eq!(x.starts_with("\"outcome\":\"kept"), y.starts_with("\"outcome\":\"kept"), "{file}: {x} vs {y}");
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

/// T7 as modelled (`RCE/NameFree.lean` hashes De Bruijn keys): since the
/// LEDGER decision of 2026-10-23 (resolution A), `state_hash` and the entry
/// hash are over the positional forms of 6.1, so a renamed program has the
/// same log hashes. Before that decision this test was ignored and failing
/// (finding of 2026-10-22: the hashes were over name-keyed JSON).
#[test]
fn state_hashes_are_name_free() {
    let (_, a) = rce(&["trace", "--json", "../examples/01_counter.rce"]);
    let (_, text) = rce(&["fmt", "--nameless", "../examples/01_counter.rce"]);
    let t = tmp("counter_nameless.rce", &text);
    let (_, b) = rce(&["trace", "--json", &t]);
    let hashes = |s: &str| -> Vec<String> {
        s.match_indices("\"state_hash\":\"").map(|(i, _)| s[i + 14..i + 14 + 64].to_string()).collect()
    };
    assert_eq!(hashes(&a), hashes(&b), "T7 as modelled: log hashes do not depend on names");
    let entry_hashes = |s: &str| -> Vec<String> {
        s.match_indices("\"hash\":\"").map(|(i, _)| s[i + 8..i + 8 + 64].to_string()).collect()
    };
    assert_eq!(entry_hashes(&a), entry_hashes(&b), "the entry hashes cover the positional event");
    assert!(!entry_hashes(&a).is_empty());
}

/// HC-21 under T7: the host's entries for kept steps name no declaration (the
/// cause, not the error's name, is in D), so a renamed program run on the
/// host gives a D with the same hashes, kept steps included.
#[test]
fn kept_entries_are_name_free() {
    use rce_spec_check::ast::Item;
    use rce_spec_check::host::{Host, MemStore};
    let src = "state Counter\n    n: u64 = 0\n\nevent Inc\nevent Dec\n\nerror Underflow\n\nstep Counter on Inc -> n + 1\nstep Counter on Dec -> try n.checked_sub(1) else Underflow\n";
    let f = tmp("kept_names.rce", src);
    let (_, projected) = rce(&["fmt", "--nameless", &f]);
    assert!(!projected.contains("Underflow"), "{projected}");
    let run = |text: &str| -> (Vec<String>, String) {
        let m = rce_spec_check::parse::parse(text).unwrap();
        let names: Vec<String> = m.items.iter().filter_map(|i| if let Item::Event { name, .. } = i { Some(name.clone()) } else { None }).collect();
        let (inc, dec) = (names[0].clone(), names[1].clone());
        let mut st = MemStore::default();
        let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
        let mut causes = vec![];
        for ev in [&dec, &inc, &dec, &dec] {
            causes.extend(h.admit_event(ev, vec![]).unwrap().iter().filter_map(|a| a.cause));
        }
        assert_eq!(causes, vec!["fail", "fail"]);
        drop(h);
        let hashes = st.lines.iter().map(|l| rce_spec_check::host::Entry::parse_line(l).unwrap().hash).collect();
        (hashes, st.lines.join("\n"))
    };
    let (a, da) = run(src);
    let (b, db) = run(&projected);
    assert!(da.contains(r#"{"payload":{"cause":"fail","seq":2},"type":"host.StepKept"}"#), "{da}");
    assert!(db.contains(r#"{"payload":{"cause":"fail","seq":2},"type":"host.StepKept"}"#), "{db}");
    assert_eq!(a, b, "D hashes do not depend on names, entries of kept steps included");
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

#[test]
fn a_field_named_like_a_builtin_method_does_not_rename_the_method() {
    // T7 sweep finding (apps/chat.rce, 24 October 2026): a state field `keys`
    // renamed the Map method `.keys()` and the projection required change
    let src = "intent \"keys\"\n    example Put(k: \"a\") -> n == 1\n\nstate S\n    keys: Map[text, u64] = {}\n    n: usize = 0\n\nevent Put(k: text)\n\nstep S on Put(k)\n    keys[k] = 1\n    n = keys.keys().len()\n";
    let f = tmp("keys_src", src);
    let (st, out) = rce(&["check", &f]);
    assert_eq!(st, 0, "{out}");
    let (_, projected) = rce(&["fmt", "--nameless", &f]);
    assert!(projected.contains(".keys()"), "{projected}");
    let g = tmp("keys_nameless", &projected);
    let (st, out) = rce(&["check", &g]);
    assert_eq!(st, 0, "{out}");
    let (st, out) = rce(&["equiv", "--fuel", "200", &f, &g]);
    assert_eq!(st, 0, "{out}");
}

#[test]
fn a_name_like_an_exponent_marker_leaves_float_literals_untouched() {
    // behaviour-index finding (stdlib/lib/num/special.rce): a declared `e`
    // renamed the exponent of `9.332636185032189e-302` to `…v10-302` and the
    // projection required change under W-SYN-1
    let src = "intent \"exponent\"\n    example Add(k: 2) -> n == 2\n\nconst e: f64 = 2.5e-3\n\nstate S\n    n: u64 = 0\n    x: f64 = 0.0\n\nevent Add(k: u64)\n\nstep S on Add(k)\n    let E: f64 = 1.5E2\n    x = x + e * 1e3 + E * 9.332636185032189e-302 + 0x1e as f64\n    n = n + k\n";
    let f = tmp("exponent_src", src);
    let (st, out) = rce(&["check", &f]);
    assert_eq!(st, 0, "{out}");
    let (st, projected) = rce(&["fmt", "--nameless", &f]);
    assert_eq!(st, 0, "{projected}");
    for lit in ["2.5e-3", "1.5E2", "1e3", "9.332636185032189e-302", "0x1e"] {
        assert!(projected.contains(lit), "{lit} changed: {projected}");
    }
    assert!(!projected.contains("const e:") && !projected.contains("let E:"), "{projected}");
    let g = tmp("exponent_nameless", &projected);
    let (st, out) = rce(&["check", &g]);
    assert_eq!(st, 0, "{out}\n{projected}");
    let (st, out) = rce(&["equiv", "--fuel", "200", &f, &g]);
    assert_eq!(st, 0, "{out}");
}
