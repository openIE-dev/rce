//! RCE-JEPA falsifiers: prediction happens in representation space (not tokens),
//! generalizes across accidental surface, abstains when too novel, and is always
//! replay-gated — the model proposes, the gate disposes.

use competence_catalog::jepa::{energy, latent, latent_collapse, predict_verified, verify, Jepa};

#[test]
fn latent_collapses_accidental_surface() {
    // Equivalent behavior lands on one latent point (representation, not tokens).
    assert_eq!(latent("3 + 2").unwrap(), latent("2 + 3").unwrap());
    assert_eq!(latent("2 * 3").unwrap(), latent("6").unwrap());
    assert_eq!(energy(&latent("2 + 3").unwrap(), &latent("3 + 2").unwrap()), 0.0);
}

#[test]
fn collapse_map_groups_equivalent_surfaces() {
    let by = latent_collapse(&["2 + 3", "3 + 2", "2 * 3", "6"]);
    // Two latent points: {2+3, 3+2} and {2*3, 6}.
    assert_eq!(by.len(), 2);
    assert!(by.values().all(|v| v.len() == 2));
}

#[test]
fn predict_generalizes_in_latent_space() {
    let mut m = Jepa::new();
    m.observe("2 + 3", "5").unwrap();
    // A novel surface with the same latent structure: the predictor proposes the
    // folded competence, and the gate confirms it replays to the intent's value.
    let p = m.predict("4 + 1").unwrap().expect("a prediction");
    assert_eq!(p.form, "5");
    assert_eq!(p.energy, 0.0);
    assert!(verify("4 + 1", &p.form));
    assert_eq!(predict_verified(&m, "4 + 1").as_deref(), Some("5"));
}

#[test]
fn prediction_is_gated_not_trusted() {
    let mut m = Jepa::new();
    m.observe("2 + 3", "5").unwrap();
    // "2 * 3" wants 6; the nearest competence is "5". JEPA may propose it, but the
    // gate refuses — a wrong prediction is never used.
    assert!(!verify("2 * 3", "5"));
    assert_eq!(predict_verified(&m, "2 * 3"), None);
}

#[test]
fn abstains_on_novel_intent() {
    let mut m = Jepa::new();
    m.observe("2 + 3", "5").unwrap();
    // A structurally distant intent: energy above threshold → abstain, don't guess.
    assert!(m.predict("a * b * c").unwrap().is_none());
    assert_eq!(predict_verified(&m, "a * b * c"), None);
}

#[test]
fn empty_model_predicts_nothing() {
    let m = Jepa::new();
    assert!(m.is_empty());
    assert!(m.predict("2 + 3").unwrap().is_none());
}

#[test]
fn observations_grow_the_joint_embedding() {
    let mut m = Jepa::new();
    m.observe("2 + 3", "5").unwrap();
    m.observe("4 * 5", "20").unwrap();
    assert_eq!(m.len(), 2);
    // Each intent now retrieves its own folded competence.
    assert_eq!(predict_verified(&m, "1 + 4").as_deref(), Some("5"));
    assert_eq!(predict_verified(&m, "2 * 10").as_deref(), Some("20"));
}