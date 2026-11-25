//! Intent→form bridge falsifiers: RCE-SPEC as the bridge from intent to form.
//! Acceptance criteria are mechanically enforced (not reviewed); requirement→form
//! is content-addressed (traceability); non-compliance is flagged, never faked.

use competence_catalog::spec_bridge::{bridge_requirement, bridge_spec, compliance, Requirement};

fn req(id: &str, source: &str, target: i128) -> Requirement {
    Requirement { id: id.into(), source: source.into(), target }
}

#[test]
fn compliant_intent_bridges_verified() {
    let b = bridge_requirement(&req("R1", "2 + 3", 5)).unwrap();
    assert!(b.verified);
    assert!(!b.repaired);
    assert_eq!(b.id, "R1");
    assert!(!b.key.is_empty());
}

#[test]
fn defective_intent_is_repaired_to_meet_spec() {
    // The intent fails its acceptance criterion; the bridge repairs it.
    let b = bridge_requirement(&req("R2", "2 + 3", 6)).unwrap();
    assert!(b.verified, "the repaired form meets the spec");
    assert!(b.repaired);
    assert_eq!(b.form, "6", "canonical form constant-folds 2*3");
}

#[test]
fn unenforceable_spec_is_flagged_not_faked() {
    // No repair reaches 999: verification fails honestly.
    let b = bridge_requirement(&req("R3", "2 + 3", 999)).unwrap();
    assert!(!b.verified, "non-compliance is flagged, never silently passed");
}

#[test]
fn compliance_is_enforced_fraction() {
    let bridged = bridge_spec(&[
        req("R1", "2 + 3", 5),   // compliant
        req("R2", "2 + 3", 6),   // repaired → compliant
        req("R3", "2 + 3", 999), // unenforceable
    ]);
    assert_eq!(bridged.len(), 3);
    assert!((compliance(&bridged) - 2.0 / 3.0).abs() < 1e-9);
}

#[test]
fn traceability_is_content_addressed() {
    // Same behavior under different requirement IDs → same content key.
    let a = bridge_requirement(&req("R1", "a + b", 0)).unwrap();
    let b = bridge_requirement(&req("R2", "b + a", 0)).unwrap();
    assert_eq!(a.key, b.key, "name-free content key links requirement→form");
    assert_ne!(a.id, b.id, "but requirement IDs stay distinct (traceability)");
}
