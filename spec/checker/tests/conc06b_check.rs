//! Draft 0.6b static rules (Appendix C, book/14-draft06-concurrency.md C.11): every program in
//! `spec/conc06b/accept/` is admitted, every one in `ill_typed/` requires
//! change, with the rule named on its `# expect:` line, and `reject/` does not parse.
use rce_spec_check::check_program;
use std::path::{Path, PathBuf};

fn dir(s: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("conc06b").join(s)
}
fn files(d: &Path) -> Vec<(String, String)> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(d)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "rce"))
        .collect();
    v.sort();
    v.into_iter()
        .map(|p| (p.display().to_string(), std::fs::read_to_string(&p).unwrap()))
        .collect()
}

#[test]
fn accept_programs_check() {
    let fs = files(&dir("accept"));
    assert_eq!(fs.len(), 8);
    for (n, s) in fs {
        let r = check_program(&[(n.clone(), s)], false);
        assert!(r[0].1.is_ok(), "{n}: {:?}", r[0].1);
    }
}

#[test]
fn ill_typed_programs_require_change_by_rule() {
    let fs = files(&dir("ill_typed"));
    assert!(fs.len() >= 21);
    let mut rules = std::collections::BTreeSet::new();
    for (n, s) in fs {
        let want = s.lines().find(|l| l.starts_with("# expect: ")).unwrap().trim_start_matches("# expect: ").trim().to_string();
        let r = check_program(&[(n.clone(), s)], false);
        match &r[0].1 {
            Ok(()) => panic!("{n} accepted, expected {want}"),
            Err(e) => assert!(e.contains(&want), "{n}: expected {want}, got {e}"),
        }
        rules.insert(want);
    }
    for r in [
        "W-GRP-1", "W-GRP-2", "W-GRP-3", "W-SES-1", "W-SES-2", "W-SES-3", "W-SES-4", "W-SES-5", "W-SCOPE-1", "W-ATOM-1",
        "W-SUP-1", "W-SUP-2", "W-NODE-1", "W-INTR-1", "W-RT-1", "W-CRDT-1", "W-LVAR-1", "W-LVAR-2",
    ] {
        assert!(rules.contains(r), "no ill-typed program for {r}");
    }
}

#[test]
fn syntax_fixtures_require_change_by_parser() {
    for (n, s) in files(&dir("change")) {
        assert!(rce_spec_check::parse::parse(&s).is_err(), "{n} parsed");
    }
}

/// a componentwise merge discharges the Merge laws (MergeLaws.prodSL);
/// lvar_freeze.rs covers the forms that require change
#[test]
fn lawful_merge_is_admitted() {
    let src = "struct Hi
    v: i64

impl Merge for Hi
    fn merge(self, other: Hi) -> Hi
        Hi(v: self.v.max(other.v))
    fn bottom() -> Hi
        Hi(v: 0)

const A: Hi = Hi(v: 1)
const B: Hi = Hi(v: 7)
";
    let r = check_program(&[("m.rce".into(), src.into())], false);
    assert!(r[0].1.is_ok(), "{:?}", r[0].1);
}

#[test]
fn session_dual_endpoint_types() {
    let src = "protocol Ask
    send i64
    recv i64

protocol Answer
    recv i64
    send i64

fn server(s: session Answer)
    let x = <- s
    s <- x * 2

fn main()
    let (c, s) = chan[session Ask](1)
    spawn server(s)
    c <- 21
    let r = <- c
    emit r
";
    let r = check_program(&[("s.rce".into(), src.into())], false);
    assert!(r[0].1.is_ok(), "{:?}", r[0].1);
    // passing the client end where the server end is expected is a type error
    let bad = src.replace("spawn server(s)", "spawn server(c)");
    let r = check_program(&[("s.rce".into(), bad)], false);
    assert!(r[0].1.is_err());
}
