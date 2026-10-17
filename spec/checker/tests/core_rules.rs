//! Keeps lean/RCE/Core/Rules.lean equal to the table in `core_rules` and
//! the table complete against book 5.3.
use rce_spec_check::core_rules::{CORE_TY_RULES, book_rules, core_rules_lean};

fn spec() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

#[test]
fn every_book_rule_is_in_the_table() {
    let ch = std::fs::read_to_string(spec().join("book/05-types-ownership.md")).unwrap();
    let rules = book_rules(&ch);
    assert!(rules.len() >= 10, "{rules:?}");
    for r in &rules {
        assert!(CORE_TY_RULES.iter().any(|(n, _)| n == r), "book 5.3 [{r}] has no row in core_rules::CORE_TY_RULES");
    }
    for (n, _) in CORE_TY_RULES {
        assert!(rules.iter().any(|r| r == n), "core_rules row [{n}] is not a rule of book 5.3");
    }
}

#[test]
fn lean_rule_table_is_current() {
    let want = core_rules_lean();
    let p = spec().join("lean/RCE/Core/Rules.lean");
    if std::env::var("RCE_BLESS").is_ok() {
        std::fs::write(&p, &want).unwrap();
    }
    let have = std::fs::read_to_string(&p).unwrap_or_default();
    assert_eq!(have, want, "lean/RCE/Core/Rules.lean is stale: rerun with RCE_BLESS=1, then lake build");
    let root = std::fs::read_to_string(spec().join("lean/RCE.lean")).unwrap();
    assert!(root.contains("import RCE.Core.Rules"), "RCE.lean must import RCE.Core.Rules");
}
