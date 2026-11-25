//! Corpus-ingestion falsifiers: the world's repos made ingestable — lift to Form,
//! repair if defective, content-address as competence; clones deduplicate.

use competence_catalog::corpus::{ingest, ingest_batch};

#[test]
fn ingest_lifts_to_form() {
    let r = ingest("(a + b) * c", None).unwrap();
    assert_eq!(r.form, "((a + b) * c)");
    assert!(r.repaired.is_none());
    assert!(!r.key.is_empty());
}

#[test]
fn ingest_repairs_a_defective_snippet() {
    // The snippet computes 5 but its spec wants 6 → ingestion repairs it.
    let r = ingest("2 + 3", Some(6)).unwrap();
    assert!(r.repaired.is_some(), "defect corrected during ingestion");
    assert_eq!(r.repaired.unwrap(), "(2 * 3)");
}

#[test]
fn same_competence_deduplicates_across_repos() {
    // The same behavior written differently collapses to one content key.
    let batch = ingest_batch(&[
        ("a + b", None),
        ("b + a", None),   // canonicalizes to the same form
        ("(a + b) * c", None),
    ]);
    assert_eq!(batch.len(), 2, "a+b and b+a are one competence");
}

#[test]
fn content_key_is_name_free_and_stable() {
    let a = ingest("a + b", None).unwrap();
    let b = ingest("b + a", None).unwrap();
    assert_eq!(a.key, b.key, "canonical behavior → one key");
}
