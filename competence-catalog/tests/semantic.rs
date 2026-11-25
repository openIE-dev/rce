//! The semantic intent key falsifiers: an intent is resolved by *behavior*, not
//! name. A novel surface form of an owned behavior hits G0 (I → 0); novel
//! behavior misses (escalate). Equivalent specs share a name-free key.

use competence_catalog::runtime::RuntimeBridge;
use competence_catalog::semantic::{semantic_key, Probe, SemanticResolver};
use competence_catalog::{intent_key, CompetenceRecord, Provenance};
use serde_json::json;

fn entries() -> Vec<(String, String, String, String)> {
    vec![
        ("time.civil.is_leap".into(), "time.civil".into(), "is_leap".into(), "(i64)->bool".into()),
        ("hash.fnv1a32".into(), "hash".into(), "fnv1a32".into(), "(&[u8])->u32".into()),
    ]
}

fn resolver() -> SemanticResolver {
    let mut r = SemanticResolver::new(RuntimeBridge::new_inprocess(), &entries());
    // Index is_leap as an owned competence (the corpus covers leap-year behavior).
    r.index(&CompetenceRecord {
        input_key: intent_key("time.civil.is_leap", "(i64)->bool"),
        handler_kind: "time.civil.is_leap".into(),
        probe_input: "[2024]".into(),
        probe_output: "true".into(),
        result: String::new(),
        cost_uj: 1,
        provenance: Provenance::Reported,
    });
    r
}

fn leap_spec() -> Vec<Probe> {
    vec![
        Probe::new(json!([2024]), "true"),
        Probe::new(json!([1900]), "false"),
        Probe::new(json!([2000]), "true"),
    ]
}

#[test]
fn novel_name_same_behavior_hits_g0() {
    let r = resolver();
    // The intent never names is_leap — it only states the behavior (a spec).
    let hit = r.resolve("(i64)->bool", &leap_spec());
    assert!(hit.is_some(), "an owned behavior is resolved by spec, not name");
    assert_eq!(hit.unwrap().handler, "time.civil.is_leap");
}

#[test]
fn novel_behavior_misses() {
    let r = resolver();
    // A spec no owned competence reproduces (2024 is a leap year, not false).
    let novel = vec![Probe::new(json!([2024]), "false")];
    assert!(r.resolve("(i64)->bool", &novel).is_none(), "novel behavior escalates");
}

#[test]
fn shape_discriminates() {
    let r = resolver();
    // Right behavior shape but wrong type signature: no candidate.
    assert!(r.resolve("(f64)->bool", &leap_spec()).is_none());
}

#[test]
fn equivalent_specs_share_a_name_free_key() {
    let a = semantic_key("(i64)->bool", &leap_spec());
    // Same probes, different order → same canonical key.
    let mut reordered = leap_spec();
    reordered.reverse();
    let b = semantic_key("(i64)->bool", &reordered);
    assert_eq!(a, b, "the semantic key is order-independent (name-free identity)");
    // Different behavior → different key.
    let c = semantic_key("(i64)->bool", &[Probe::new(json!([2024]), "false")]);
    assert_ne!(a, c, "novel behavior has a distinct intent identity");
    // Different shape → different key.
    let d = semantic_key("(f64)->bool", &leap_spec());
    assert_ne!(a, d);
}

#[test]
fn fingerprint_groups_equivalent_behavior_beyond_probes() {
    let entries = vec![
        ("time.civil.is_leap".to_string(), "time.civil".to_string(), "is_leap".to_string(), "(i64)->bool".to_string()),
        ("time.is_leap".to_string(), "time".to_string(), "is_leap".to_string(), "(i64)->bool".to_string()),
        ("os.proc.success".to_string(), "os.proc".to_string(), "success".to_string(), "(i64)->bool".to_string()),
    ];
    let r = SemanticResolver::new(RuntimeBridge::new_inprocess(), &entries);
    let f1 = r.fingerprint("time.civil.is_leap").unwrap();
    let f2 = r.fingerprint("time.is_leap").unwrap();
    let f3 = r.fingerprint("os.proc.success").unwrap();
    assert_eq!(f1, f2, "the same leap-year behavior shares a canonical fingerprint");
    assert_ne!(f1, f3, "exit-code-success is a different behavior");
}

#[test]
fn coverage_is_the_I_measure() {
    let r = resolver();
    // I = H_task - H_corpus: covered behavior → I = 0 (a G0 hit exists).
    let covered = r.resolve("(i64)->bool", &leap_spec()).is_some();
    let novel = r.resolve("(i64)->bool", &[Probe::new(json!([2024]), "false")]).is_some();
    assert!(covered && !novel, "coverage by replay is the corpus-independent I measure");
}

#[test]
fn bounded_exhaustive_equivalence() {
    let entries = vec![
        ("time.civil.is_leap".to_string(), "time.civil".to_string(), "is_leap".to_string(), "(i64)->bool".to_string()),
        ("time.is_leap".to_string(), "time".to_string(), "is_leap".to_string(), "(i64)->bool".to_string()),
        ("os.proc.success".to_string(), "os.proc".to_string(), "success".to_string(), "(i64)->bool".to_string()),
    ];
    let r = SemanticResolver::new(RuntimeBridge::new_inprocess(), &entries);
    // Agreement over the entire bounded domain [-50,50] — full equivalence there
    // (bounded translation validation, deterministic, no SMT).
    assert!(r.bounded_equivalent("time.civil.is_leap", "time.is_leap"));
    assert!(!r.bounded_equivalent("time.civil.is_leap", "os.proc.success"));
}

#[test]
fn finite_domain_is_enumerated_completely() {
    // bool is finite: the exhaustive domain is its whole truth table.
    let inputs = competence_catalog::semantic::exhaustive_inputs("(bool)->bool");
    assert_eq!(inputs.len(), 2, "complete enumeration of the bool domain");
}
