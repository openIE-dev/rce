use competence_catalog::train::{canonical_pairs, HierPredictor};

#[test]
fn corpus_pairs_include_changed_normalizations() {
    let p = canonical_pairs(500);
    assert!(p.len() > 400);
    assert!(p.iter().any(|(a, b)| a != b));
}

#[test]
fn learned_inverse_beats_raw_representation() {
    let p = canonical_pairs(800);
    let changed: Vec<&(Vec<f64>, Vec<f64>)> = p.iter().filter(|(a, b)| a != b).collect();
    assert!(!changed.is_empty());
    let h = HierPredictor::train(&p, 5, 300, 0.05);
    let (mut e_pred, mut e_raw) = (0.0, 0.0);
    for (x, y) in &changed {
        e_pred += h.energy(x, y);
        e_raw += x.iter().zip(y.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f64>();
    }
    assert!(e_pred < e_raw, "predicted {e_pred} vs raw {e_raw}");
}
