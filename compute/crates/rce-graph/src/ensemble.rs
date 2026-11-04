//! Ensemble methods: bagging, boosting helpers, voting, stacking,
//! model averaging, diversity measures, out-of-bag estimation.

// ═══════════════════════════════════════════════════════════════════
// VOTING / AVERAGING
// ═══════════════════════════════════════════════════════════════════

/// Hard voting (majority vote) from multiple classifier predictions.
pub fn hard_vote(predictions: &[Vec<usize>]) -> Vec<usize> {
    let n = predictions[0].len();
    (0..n).map(|i| {
        let mut counts = std::collections::HashMap::new();
        for pred in predictions { *counts.entry(pred[i]).or_insert(0usize) += 1; }
        *counts.iter().max_by_key(|(_, c)| **c).unwrap().0
    }).collect()
}

/// Soft voting (average probabilities) from multiple classifier probability outputs.
/// probs: k models × n samples × c classes. Returns n × c averaged probabilities.
pub fn soft_vote(probs: &[Vec<f64>], n: usize, n_classes: usize) -> Vec<f64> {
    let k = probs.len() as f64;
    let mut avg = vec![0.0; n * n_classes];
    for model_probs in probs {
        for i in 0..n * n_classes { avg[i] += model_probs[i]; }
    }
    for v in avg.iter_mut() { *v /= k; }
    avg
}

/// Weighted voting.
pub fn weighted_vote(predictions: &[Vec<usize>], weights: &[f64]) -> Vec<usize> {
    let n = predictions[0].len();
    (0..n).map(|i| {
        let mut scores = std::collections::HashMap::new();
        for (pi, pred) in predictions.iter().enumerate() {
            *scores.entry(pred[i]).or_insert(0.0f64) += weights[pi];
        }
        *scores.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    }).collect()
}

/// Average regression predictions from multiple models.
pub fn average_predictions(predictions: &[Vec<f64>]) -> Vec<f64> {
    let n = predictions[0].len();
    let k = predictions.len() as f64;
    (0..n).map(|i| predictions.iter().map(|p| p[i]).sum::<f64>() / k).collect()
}

/// Weighted average regression predictions.
pub fn weighted_average_predictions(predictions: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    let n = predictions[0].len();
    let w_sum: f64 = weights.iter().sum();
    (0..n).map(|i| predictions.iter().zip(weights).map(|(p, &w)| p[i] * w).sum::<f64>() / w_sum).collect()
}

/// Median prediction (robust averaging).
pub fn median_predictions(predictions: &[Vec<f64>]) -> Vec<f64> {
    let n = predictions[0].len();
    (0..n).map(|i| {
        let mut vals: Vec<f64> = predictions.iter().map(|p| p[i]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = vals.len();
        if m % 2 == 0 { (vals[m/2 - 1] + vals[m/2]) / 2.0 } else { vals[m/2] }
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// BAGGING HELPERS
// ═══════════════════════════════════════════════════════════════════

/// Generate bootstrap sample indices (with replacement).
pub fn bootstrap_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut state = seed_state(seed);
    (0..n).map(|_| prng_next(&mut state) as usize % n).collect()
}

/// Out-of-bag indices (those NOT in bootstrap sample).
pub fn oob_indices(n: usize, bootstrap: &[usize]) -> Vec<usize> {
    let in_bag: std::collections::HashSet<usize> = bootstrap.iter().copied().collect();
    (0..n).filter(|i| !in_bag.contains(i)).collect()
}

/// Generate multiple bootstrap samples.
pub fn multi_bootstrap(n: usize, n_bags: usize, seed: u64) -> Vec<Vec<usize>> {
    (0..n_bags).map(|i| bootstrap_indices(n, seed.wrapping_add(i as u64))).collect()
}

/// Feature subsampling indices (random subset of features).
pub fn feature_subsample(n_features: usize, max_features: usize, seed: u64) -> Vec<usize> {
    let mut state = seed_state(seed);
    let mut indices: Vec<usize> = (0..n_features).collect();
    for i in (1..n_features).rev() {
        let j = prng_next(&mut state) as usize % (i + 1);
        indices.swap(i, j);
    }
    indices.truncate(max_features);
    indices.sort();
    indices
}

/// Recommended max_features for classification (sqrt(n_features)).
pub fn max_features_classification(n_features: usize) -> usize {
    (n_features as f64).sqrt().round() as usize
}

/// Recommended max_features for regression (n_features / 3).
pub fn max_features_regression(n_features: usize) -> usize {
    (n_features / 3).max(1)
}

// ═══════════════════════════════════════════════════════════════════
// BOOSTING HELPERS
// ═══════════════════════════════════════════════════════════════════

/// AdaBoost sample weights update.
/// Returns (new_weights, alpha) where alpha is the model weight.
pub fn adaboost_update_weights(weights: &[f64], y_true: &[usize], y_pred: &[usize]) -> (Vec<f64>, f64) {
    let _n = weights.len();
    let err: f64 = weights.iter().zip(y_true.iter().zip(y_pred))
        .map(|(&w, (&t, &p))| if t != p { w } else { 0.0 }).sum();
    let err = err.max(1e-10).min(1.0 - 1e-10);
    let alpha = 0.5 * ((1.0 - err) / err).ln();

    let mut new_weights: Vec<f64> = weights.iter().zip(y_true.iter().zip(y_pred))
        .map(|(&w, (&t, &p))| {
            if t != p { w * alpha.exp() } else { w * (-alpha).exp() }
        }).collect();
    let sum: f64 = new_weights.iter().sum();
    for w in new_weights.iter_mut() { *w /= sum; }
    (new_weights, alpha)
}

/// Gradient boosting residuals for regression (MSE loss).
pub fn gradient_boost_residuals(y_true: &[f64], y_pred: &[f64]) -> Vec<f64> {
    y_true.iter().zip(y_pred).map(|(t, p)| t - p).collect()
}

/// Gradient boosting residuals for binary classification (log loss).
pub fn gradient_boost_residuals_logistic(y_true: &[f64], y_pred_logits: &[f64]) -> Vec<f64> {
    y_true.iter().zip(y_pred_logits).map(|(&t, &logit)| {
        let p = 1.0 / (1.0 + (-logit).exp());
        t - p
    }).collect()
}

/// Learning rate schedule for boosting: shrinkage.
pub fn shrinkage_rate(base_lr: f64, iteration: usize, decay: f64) -> f64 {
    base_lr / (1.0 + decay * iteration as f64)
}

// ═══════════════════════════════════════════════════════════════════
// DIVERSITY MEASURES
// ═══════════════════════════════════════════════════════════════════

/// Disagreement measure between two classifiers.
pub fn disagreement(pred1: &[usize], pred2: &[usize]) -> f64 {
    let n = pred1.len() as f64;
    pred1.iter().zip(pred2).filter(|(a, b)| a != b).count() as f64 / n
}

/// Q-statistic between two classifiers.
pub fn q_statistic(pred1: &[usize], pred2: &[usize], y_true: &[usize]) -> f64 {
    let _n = y_true.len() as f64;
    let mut a = 0.0; // both correct
    let mut b = 0.0; // 1 correct, 2 wrong
    let mut c = 0.0; // 1 wrong, 2 correct
    let mut d = 0.0; // both wrong
    for i in 0..y_true.len() {
        let c1 = pred1[i] == y_true[i];
        let c2 = pred2[i] == y_true[i];
        match (c1, c2) {
            (true, true) => a += 1.0,
            (true, false) => b += 1.0,
            (false, true) => c += 1.0,
            (false, false) => d += 1.0,
        }
    }
    let num = a * d - b * c;
    let den = a * d + b * c;
    if (den as f64).abs() < 1e-15 { 0.0 } else { num / den }
}

/// Correlation diversity: pairwise disagreement averaged over all pairs.
pub fn ensemble_diversity(predictions: &[Vec<usize>]) -> f64 {
    let k = predictions.len();
    if k < 2 { return 0.0; }
    let mut sum = 0.0;
    let mut count = 0;
    for i in 0..k {
        for j in i+1..k {
            sum += disagreement(&predictions[i], &predictions[j]);
            count += 1;
        }
    }
    sum / count as f64
}

/// Kohavi-Wolpert variance (interrater agreement).
pub fn kw_variance(predictions: &[Vec<usize>], n_classes: usize) -> f64 {
    let k = predictions.len() as f64;
    let n = predictions[0].len();
    let mut var = 0.0;
    for i in 0..n {
        for c in 0..n_classes {
            let votes = predictions.iter().filter(|p| p[i] == c).count() as f64;
            var += votes * (k - votes);
        }
    }
    var / (n as f64 * k * k)
}

// ═══════════════════════════════════════════════════════════════════
// STACKING HELPERS
// ═══════════════════════════════════════════════════════════════════

/// Build stacking meta-features: concatenate out-of-fold predictions.
/// fold_preds: k folds × m models × n_fold_samples. Returns n × m matrix.
pub fn build_meta_features(fold_preds: &[(Vec<usize>, Vec<Vec<f64>>)], n: usize, n_models: usize) -> Vec<f64> {
    let mut meta = vec![0.0; n * n_models];
    for (test_indices, model_preds) in fold_preds {
        for (mi, preds) in model_preds.iter().enumerate() {
            for (fi, &idx) in test_indices.iter().enumerate() {
                if fi < preds.len() { meta[idx * n_models + mi] = preds[fi]; }
            }
        }
    }
    meta
}

fn prng_next(s: &mut [u64; 4]) -> u64 {
    let r = (s[0].wrapping_add(s[3])).rotate_left(23).wrapping_add(s[0]);
    let t = s[1] << 17; s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = s[3].rotate_left(45); r
}
fn seed_state(seed: u64) -> [u64; 4] {
    let mut s = [seed, seed.wrapping_mul(6364136223846793005).wrapping_add(1),
                 seed.wrapping_mul(1442695040888963407).wrapping_add(3),
                 seed.wrapping_mul(2862933555777941757).wrapping_add(7)];
    for _ in 0..20 { prng_next(&mut s); } s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hard_vote() {
        let preds = vec![vec![0, 1, 1], vec![1, 1, 0], vec![0, 1, 1]];
        assert_eq!(hard_vote(&preds), vec![0, 1, 1]);
    }

    #[test]
    fn test_average_predictions() {
        let preds = vec![vec![1.0, 2.0, 3.0], vec![3.0, 2.0, 1.0]];
        assert_eq!(average_predictions(&preds), vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_median_predictions() {
        let preds = vec![vec![1.0], vec![3.0], vec![2.0]];
        assert_eq!(median_predictions(&preds), vec![2.0]);
    }

    #[test]
    fn test_bootstrap_indices() {
        let idx = bootstrap_indices(100, 42);
        assert_eq!(idx.len(), 100);
        assert!(idx.iter().all(|&i| i < 100));
    }

    #[test]
    fn test_oob_indices() {
        let boot = vec![0, 1, 0, 2, 1];
        let oob = oob_indices(5, &boot);
        assert!(oob.contains(&3));
        assert!(oob.contains(&4));
    }

    #[test]
    fn test_feature_subsample() {
        let feat = feature_subsample(10, 3, 42);
        assert_eq!(feat.len(), 3);
        assert!(feat.iter().all(|&f| f < 10));
    }

    #[test]
    fn test_adaboost_weights() {
        let weights = vec![0.2; 5];
        let y_true = vec![0, 1, 0, 1, 1];
        let y_pred = vec![0, 1, 1, 1, 1]; // one mistake
        let (new_w, alpha) = adaboost_update_weights(&weights, &y_true, &y_pred);
        assert!(alpha > 0.0); // better than random
        assert!((new_w.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_residuals() {
        let res = gradient_boost_residuals(&[1.0, 2.0, 3.0], &[0.5, 2.5, 2.5]);
        assert_eq!(res, vec![0.5, -0.5, 0.5]);
    }

    #[test]
    fn test_disagreement() {
        assert!((disagreement(&[0, 1, 0, 1], &[0, 0, 0, 1]) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_ensemble_diversity() {
        let preds = vec![vec![0, 1, 0], vec![1, 0, 1]]; // maximum disagreement
        assert!((ensemble_diversity(&preds) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_features() {
        assert_eq!(max_features_classification(100), 10);
        assert_eq!(max_features_regression(30), 10);
    }
}
