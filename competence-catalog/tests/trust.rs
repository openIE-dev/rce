//! Trust/compliance falsifiers: SOC2 control map, tamper-evident audit, verified-only admission,
//! honest gap list.

use competence_catalog::trust::{Criterion, TrustLayer};

#[test]
fn audit_chain_is_tamper_evident() {
    let mut t = TrustLayer::new();
    t.record("admit:hash.fnv1a32", true, "Measured");
    t.record("admit:time.is_leap", true, "Reported");
    assert!(t.chain_intact());
    t.log[0].hash = "tampered".into();
    assert!(!t.chain_intact());
}

#[test]
fn verified_only_admission_and_report() {
    let mut t = TrustLayer::new();
    t.record("admit:a", true, "Measured");
    t.record("refused:forged", false, "Reported");
    let rep = t.report();
    let pi = rep.iter().find(|(c, _)| *c == Criterion::ProcessingIntegrity).unwrap();
    assert!(pi.1, "processing integrity satisfied by verified-only admission + intact chain");
    assert!(t.chain_intact());
}

#[test]
fn honest_compliance_gaps_listed() {
    let t = TrustLayer::new();
    let gaps = t.gaps();
    assert!(gaps.iter().any(|c| c.id == "C1.2")); // encryption at rest
    assert!(gaps.iter().any(|c| c.id == "CC6.3")); // RBAC
}

#[test]
fn soc2_map_covers_all_criteria() {
    let cs = competence_catalog::trust::soc2_controls();
    for c in [Criterion::Security, Criterion::Availability, Criterion::ProcessingIntegrity,
              Criterion::Confidentiality, Criterion::Privacy] {
        assert!(cs.iter().any(|x| x.criterion == c), "missing criterion {c:?}");
    }
}