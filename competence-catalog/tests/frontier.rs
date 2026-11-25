//! Frontier seeds: trained latent (JEPA-style) inverse predictor + tiered JIT.

use competence_catalog::jit::Jit;
use competence_catalog::train::{features, LatentPredictor};
use competence_catalog::lift::lift_expr;

#[test]
fn latent_predictor_learns_the_inverse() {
    // verified pairs: target representation = 2 * intent representation (linear, learnable).
    let pairs: Vec<(Vec<f64>, Vec<f64>)> = (1..6).map(|i| {
        let x = vec![i as f64, (i + 1) as f64];
        let y = vec![2.0 * i as f64, 2.0 * (i + 1) as f64];
        (x, y)
    }).collect();
    let p = LatentPredictor::train(&pairs, 400, 0.05);
    let pred = p.predict(&[3.0, 4.0]);
    assert!((pred[0] - 6.0).abs() < 0.5, "pred {pred:?}");
    assert!((pred[1] - 8.0).abs() < 0.5);
    // energy ranks the correct target below a wrong one
    assert!(p.energy(&[3.0, 4.0], &[6.0, 8.0]) < p.energy(&[3.0, 4.0], &[0.0, 0.0]));
}

#[test]
fn jit_tiers_interpret_cold_compile_hot() {
    let mut j = Jit::new();
    j.threshold = 2;
    assert_eq!(j.run_tiered("2 * 3 + 4").unwrap(), "10"); // cold: interpret
    assert_eq!(j.run_tiered("2 * 3 + 4").unwrap(), "10"); // cold: interpret
    assert_eq!(j.compiles, 0);
    assert_eq!(j.run_tiered("2 * 3 + 4").unwrap(), "10"); // hot: compile native
    assert_eq!(j.compiles, 1);
}