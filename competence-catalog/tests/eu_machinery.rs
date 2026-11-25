use competence_catalog::trust::{eu_machinery_controls, TrustLayer};

#[test]
fn eu_map_has_built_and_open_controls() {
    let cs = eu_machinery_controls();
    assert_eq!(cs.len(), 6);
    assert_eq!(cs.iter().filter(|c| c.built).count(), 2);
}

#[test]
fn eu_gaps_are_the_honest_process_edges() {
    let t = TrustLayer::new();
    let gaps = t.eu_gaps();
    assert_eq!(gaps.len(), 4);
    assert!(gaps.iter().any(|c| c.id == "EU-1.3"));
    assert!(gaps.iter().any(|c| c.id == "IEC-61508"));
}

#[test]
fn five_year_log_chain_intact_and_soc2_unweakened() {
    let mut t = TrustLayer::new();
    t.record("safety-update", true, "HwShunt");
    t.record("refused:unverified", false, "Estimator");
    assert!(t.chain_intact());
    assert!(t.report().iter().all(|(_, ok)| *ok));
    assert!(t.gaps().is_empty());
}
