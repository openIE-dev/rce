use competence_catalog::twin::{identify, simulate, Plant};

#[test]
fn identifies_the_true_dt() {
    let acts = vec![1.0, 1.0, 1.0];
    let rec = simulate(&Plant { dt: 0.1 }, &acts);
    let (dt, d) = identify(&acts, &rec, &[0.05, 0.1, 0.2], 1e-9).unwrap();
    assert_eq!(dt, 0.1);
    assert!(d < 1e-9);
}

#[test]
fn no_explaining_candidate_returns_none() {
    let acts = vec![1.0, 1.0, 1.0];
    let rec = simulate(&Plant { dt: 0.1 }, &acts);
    assert!(identify(&acts, &rec, &[0.05, 0.2], 1e-9).is_none());
    assert!(identify(&acts, &rec, &[0.1], f64::NAN).is_none());
}
