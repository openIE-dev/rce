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
        let hashes = |s: &str| -> Vec<String> {
            s.match_indices("\"state_hash\":\"").map(|(i, _)| s[i + 14..i + 14 + 64].to_string()).collect()
        };
        assert_eq!(hashes(&a), hashes(&b), "{file}: T7 says the log hashes do not depend on names");
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
