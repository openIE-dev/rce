//! Per-record replay admission: a crystallized competence is admitted only when
//! the reference runtime re-executes its declared spectest and reproduces the
//! declared output. A forged or negatively-transferring record is refused, not
//! ranked. This is the trustworthy gate of the Law of Coding, made real for any
//! tractable operation (here a hash and an error-union stats function).

use competence_catalog::runtime::{GeneralReplayChecker, RuntimeBridge};
use competence_catalog::{intent_key, Catalog, CompetenceRecord, Provenance};
use rcevc_core::Gate;

fn catalog() -> Catalog {
    let bridge = RuntimeBridge::new_inprocess();
    let entries = vec![
        ("hash.fnv1a32".to_string(), "hash".to_string(), "fnv1a32".to_string(), "(&[u8])->u32".to_string()),
        ("num.stats.mean".to_string(), "num.stats".to_string(), "mean".to_string(), "(&[f64])->f64!StatsError".to_string()),
    ];
    let checker = GeneralReplayChecker::from_index(bridge, &entries);
    Catalog::with_gate(Gate::new().with(checker)).unwrap()
}

fn record(op: &str, shape: &str, probe_input: &str, probe_output: &str) -> CompetenceRecord {
    CompetenceRecord {
        input_key: intent_key(op, shape),
        handler_kind: op.to_string(),
        probe_input: probe_input.to_string(),
        probe_output: probe_output.to_string(),
        result: String::new(),
        cost_uj: 5_000_000,
        provenance: Provenance::Reported,
    }
}

#[test]
fn correct_spectest_is_admitted_by_replay() {
    let mut cat = catalog();
    // FNV-1a 32 of "hi" = 1748694682 (independently computed).
    let out = cat.admit(record("hash.fnv1a32", "(&[u8])->u32", "[[104,105]]", "1748694682"));
    assert!(out.is_accepted(), "a reproducing spectest must be admitted");
    assert_eq!(cat.meter().catalog_size, 1);
}

#[test]
fn forged_spectest_is_refused() {
    let mut cat = catalog();
    // Declares the wrong output for FNV-1a 32 of "hi": replay must refuse it.
    let out = cat.admit(record("hash.fnv1a32", "(&[u8])->u32", "[[104,105]]", "999"));
    assert!(!out.is_accepted(), "a non-reproducing spectest must be refused, not ranked");
    assert_eq!(cat.meter().catalog_size, 0);
}

#[test]
fn error_union_competence_admitted_by_replay() {
    let mut cat = catalog();
    // mean([1,2,3]) = 2; the ok value of the error union.
    let out = cat.admit(record("num.stats.mean", "(&[f64])->f64!StatsError", "[[1.0,2.0,3.0]]", "2"));
    assert!(out.is_accepted(), "an error-union spectest that reproduces is admitted");
    assert_eq!(cat.meter().catalog_size, 1);
}

#[test]
fn unknown_operation_is_refused() {
    let mut cat = catalog();
    let out = cat.admit(record("not.in.basis", "(i64)->i64", "[1]", "1"));
    assert!(!out.is_accepted(), "an operation with no signature cannot be replay-verified");
    assert_eq!(cat.meter().catalog_size, 0);
}
