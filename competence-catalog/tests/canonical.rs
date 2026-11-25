//! Canonicalization falsifiers (No Accidental Software / Conal ingested):
//! accidentally-different source collapses to one canonical key; distinct
//! behavior stays distinct; and normalization preserves behavior (replay).

use competence_catalog::canonical::{canonicalize, canonical_key};

#[test]
fn accidental_representation_collapses() {
    // Commutativity, identities, folding — the accidental-representation tax.
    assert_eq!(canonical_key("a + b").unwrap(), canonical_key("b + a").unwrap());
    assert_eq!(canonical_key("a * b").unwrap(), canonical_key("b * a").unwrap());
    assert_eq!(canonical_key("x + 0").unwrap(), canonical_key("x").unwrap());
    assert_eq!(canonical_key("0 + x").unwrap(), canonical_key("x").unwrap());
    assert_eq!(canonical_key("1 * x").unwrap(), canonical_key("x").unwrap());
    assert_eq!(canonicalize("2 + 3").unwrap(), "5");
    assert_eq!(canonical_key("2 * 3 + a").unwrap(), canonical_key("6 + a").unwrap());
}

#[test]
fn distinct_behavior_stays_distinct() {
    assert_ne!(canonical_key("a + b").unwrap(), canonical_key("a - b").unwrap());
    assert_ne!(canonical_key("a + b").unwrap(), canonical_key("a * b").unwrap());
    assert_ne!(canonical_key("a + 1").unwrap(), canonical_key("a + 2").unwrap());
}

fn eval(expr: &str, a: i64, b: i64) -> String {
    // Bind only the variables the expression uses (avoid W-UNUSED-1).
    let uses_b = expr.contains('b');
    let (sig, pat, event) = if uses_b {
        ("Calc(a: i64, b: i64)", "Calc(a: a, b: b)", format!("Calc(a: {a}, b: {b})"))
    } else {
        ("Calc(a: i64)", "Calc(a: a)", format!("Calc(a: {a})"))
    };
    let prog = format!(
        "module canon\n\nstate Out\n    v: i64 = 0\n\nevent {sig}\n\nstep Out on {pat}\n    v = {expr}\n"
    );
    let out = rce_spec_check::cli::run(&prog, Some(&event)).expect("run");
    let last = out.lines.iter().rev().find(|l| l.contains("v = ")).cloned().unwrap_or_default();
    let rest = last.split("v = ").nth(1).unwrap_or("");
    rest.trim_end_matches(" }").to_string()
}

#[test]
fn normalization_preserves_behavior() {
    // Original vs canonical agree over a domain — the rewrite is sound.
    for (orig, canon) in [("a + b", "b + a"), ("a * 1", "a"), ("a + 0", "a"), ("2 * 3 + a", "6 + a")] {
        for a in [-3, 0, 5] {
            for b in [-2, 1, 4] {
                assert_eq!(eval(orig, a, b), eval(canon, a, b), "{orig} vs {canon} at ({a},{b})");
            }
        }
    }
}
