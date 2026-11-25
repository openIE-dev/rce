use competence_catalog::train::{features, HierPredictor};
use competence_catalog::lift::lift_expr;

fn pairs() -> Vec<(Vec<f64>, Vec<f64>)> {
    let srcs = ["1", "1+2", "1+2+3", "2*3", "2*3+4", "(1+2)*3"];
    srcs
        .iter()
        .map(|s| {
            let e = lift_expr(s).unwrap();
            (features(&e), features(&e))
        })
        .collect()
}

#[test]
fn hierarchical_predictor_shapes_and_energy() {
    let ps = pairs();
    let h = HierPredictor::train(&ps, 3, 300, 0.05);
    let x = &ps[2].0;
    let y = &ps[2].1;
    assert_eq!(h.predict(x).len(), y.len());
    let match_e = h.energy(x, y);
    let other_e = h.energy(x, &ps[1].1);
    assert!(match_e < other_e, "matched {match_e} vs other {other_e}");
}

#[test]
fn hierarchical_training_deterministic() {
    let ps = pairs();
    let a = HierPredictor::train(&ps, 3, 100, 0.05);
    let b = HierPredictor::train(&ps, 3, 100, 0.05);
    assert_eq!(a.predict(&ps[0].0), b.predict(&ps[0].0));
}
