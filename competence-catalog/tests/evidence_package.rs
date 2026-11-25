use competence_catalog::trust::TrustLayer;

#[test]
fn evidence_bundle_carries_maps_chain_and_open_gaps() {
    let mut t = TrustLayer::new();
    t.record("safety-update", true, "HwShunt");
    t.record("refused:unverified", false, "Estimator");
    let p = t.evidence_package();
    assert_eq!(p["schema"], "rce-evidence/1");
    assert_eq!(p["chain_intact"], true);
    assert_eq!(p["entries"], 2);
    assert_eq!(p["soc2"].as_array().unwrap().len(), 10);
    let gaps: Vec<&str> = p["open_gaps"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert!(gaps.contains(&"EU-1.3"));
    assert!(gaps.contains(&"IEC-61508"));
}

#[test]
fn tampered_chain_shows_in_the_bundle() {
    let mut t = TrustLayer::new();
    t.record("a", true, "HwShunt");
    t.log[0].action = "edited".into();
    let p = t.evidence_package();
    assert_eq!(p["chain_intact"], false);
}
