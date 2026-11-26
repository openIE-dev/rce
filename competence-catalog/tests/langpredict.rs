use competence_catalog::langpredict::{self, FABRICS};
use competence_catalog::structure::CONSTRUCTS;

#[test]
fn every_fabric_has_a_real_capability_gap() {
    for f in FABRICS {
        let g = langpredict::gap(f);
        assert!(!g.is_empty(), "{} should have missing native constructs", f.id);
        // the gap is genuine: none of these are in the measured taxonomy yet
        for c in &g {
            assert!(!CONSTRUCTS.contains(c), "{c} must not already be in CONSTRUCTS");
        }
    }
}

#[test]
fn predicted_language_carries_the_missing_capability() {
    let q = FABRICS.iter().find(|f| f.id == "quantum").unwrap();
    let p = langpredict::predict(q);
    assert_eq!(p.name, "Superpose");
    assert!(p.missing_constructs.contains(&"qubit".to_string()));
    assert!(p.missing_constructs.contains(&"entangle".to_string()));
    assert!(p.rationale.contains("impedance tax"));
    let t = FABRICS.iter().find(|f| f.id == "thermodynamic").unwrap();
    let pt = langpredict::predict(t);
    assert!(pt.missing_constructs.contains(&"landauer".to_string()));
}

#[test]
fn prediction_does_not_mutate_the_measured_taxonomy() {
    // analysis only: the 113-construct matrix stays truthful
    assert_eq!(CONSTRUCTS.len(), 113);
    let all = langpredict::predict_all();
    assert_eq!(all.len(), FABRICS.len());
}
