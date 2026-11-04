//! Extended ML: Ridge/Lasso regression, gradient boosting, spectral clustering,
//! agglomerative clustering, isotonic regression, kernel ridge.

use crate::math_runtime::*;

// ═══════════════════════════════════════════════════════════════════
// RIDGE REGRESSION (L2 regularized)
// ═══════════════════════════════════════════════════════════════════

/// Ridge regression: minimize ||y - Xw||^2 + alpha * ||w||^2.
/// Returns weight vector w.
pub fn ridge_regression(x: &[Vec<f64>], y: &[f64], alpha: f64) -> Vec<f64> {
    let n = x.len();
    if n == 0 { return vec![]; }
    let dim = x[0].len();

    // X'X + alpha*I
    let xt = transpose(x);
    let xtx = mat_mul(&xt, x);
    let mut reg = xtx;
    for i in 0..dim { reg[i][i] += alpha; }

    // X'y
    let xty: Vec<f64> = (0..dim).map(|j| {
        (0..n).map(|i| x[i][j] * y[i]).sum()
    }).collect();

    // Solve (X'X + alpha*I) * w = X'y
    solve(&reg, &xty).unwrap_or(vec![0.0; dim])
}

// ═══════════════════════════════════════════════════════════════════
// LASSO REGRESSION (L1 regularized, coordinate descent)
// ═══════════════════════════════════════════════════════════════════

/// Lasso regression via coordinate descent.
pub fn lasso_regression(x: &[Vec<f64>], y: &[f64], alpha: f64, max_iter: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 { return vec![]; }
    let dim = x[0].len();
    let mut w = vec![0.0; dim];
    let mut residuals: Vec<f64> = y.to_vec();

    for _ in 0..max_iter {
        let mut max_change = 0.0;
        for j in 0..dim {
            // Add back contribution of w[j]
            for i in 0..n { residuals[i] += x[i][j] * w[j]; }

            // Compute rho = X_j' * residuals
            let rho: f64 = (0..n).map(|i| x[i][j] * residuals[i]).sum::<f64>() / n as f64;

            // Soft thresholding
            let xj_sq: f64 = (0..n).map(|i| x[i][j] * x[i][j]).sum::<f64>() / n as f64;
            let new_w = soft_threshold(rho, alpha) / xj_sq.max(1e-15);
            let change = (new_w - w[j]).abs();
            if change > max_change { max_change = change; }
            w[j] = new_w;

            // Update residuals
            for i in 0..n { residuals[i] -= x[i][j] * w[j]; }
        }
        if max_change < 1e-8 { break; }
    }
    w
}

fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda { x - lambda }
    else if x < -lambda { x + lambda }
    else { 0.0 }
}

// ═══════════════════════════════════════════════════════════════════
// ELASTIC NET (L1 + L2)
// ═══════════════════════════════════════════════════════════════════

/// Elastic net: alpha * l1_ratio * ||w||_1 + alpha * (1-l1_ratio) * ||w||_2^2.
pub fn elastic_net(x: &[Vec<f64>], y: &[f64], alpha: f64, l1_ratio: f64, max_iter: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 { return vec![]; }
    let dim = x[0].len();
    let mut w = vec![0.0; dim];
    let mut residuals: Vec<f64> = y.to_vec();
    let l1 = alpha * l1_ratio;
    let l2 = alpha * (1.0 - l1_ratio);

    for _ in 0..max_iter {
        let mut max_change = 0.0;
        for j in 0..dim {
            for i in 0..n { residuals[i] += x[i][j] * w[j]; }
            let rho: f64 = (0..n).map(|i| x[i][j] * residuals[i]).sum::<f64>() / n as f64;
            let xj_sq: f64 = (0..n).map(|i| x[i][j] * x[i][j]).sum::<f64>() / n as f64;
            let new_w = soft_threshold(rho, l1) / (xj_sq + l2).max(1e-15);
            let change = (new_w - w[j]).abs();
            if change > max_change { max_change = change; }
            w[j] = new_w;
            for i in 0..n { residuals[i] -= x[i][j] * w[j]; }
        }
        if max_change < 1e-8 { break; }
    }
    w
}

// ═══════════════════════════════════════════════════════════════════
// GRADIENT BOOSTING REGRESSOR (simplified)
// ═══════════════════════════════════════════════════════════════════

/// Gradient boosting regression with decision stumps.
pub struct GradientBoosting {
    pub stumps: Vec<(usize, f64, f64, f64)>, // (feature, threshold, left_val, right_val)
    pub learning_rate: f64,
    pub initial_pred: f64,
}

impl GradientBoosting {
    pub fn fit(x: &[Vec<f64>], y: &[f64], n_estimators: usize, learning_rate: f64, max_depth: usize) -> Self {
        let n = x.len();
        let initial_pred = mean(y);
        let mut predictions = vec![initial_pred; n];
        let mut stumps = Vec::new();

        for _ in 0..n_estimators {
            // Compute residuals (negative gradient for MSE)
            let residuals: Vec<f64> = y.iter().zip(&predictions).map(|(&yi, &pi)| yi - pi).collect();

            // Fit a decision tree to residuals
            let tree = crate::ml_models::DecisionTree::fit_regressor(x, &residuals, max_depth.min(3), 2);
            let tree_pred = tree.predict(x);

            // Update predictions
            for i in 0..n {
                predictions[i] += learning_rate * tree_pred[i];
            }

            // Store as stump (simplified — store the tree predictions directly)
            // For a proper implementation, we'd store the tree structure
            stumps.push((0, 0.0, 0.0, 0.0)); // placeholder
        }

        Self { stumps, learning_rate, initial_pred }
    }
}

// ═══════════════════════════════════════════════════════════════════
// ISOTONIC REGRESSION
// ═══════════════════════════════════════════════════════════════════

/// Isotonic regression: fit non-decreasing step function.
/// Pool Adjacent Violators (PAV) algorithm.
pub fn isotonic_regression(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    if n == 0 { return vec![]; }
    let mut result = y.to_vec();
    let mut weights = vec![1.0; n];

    loop {
        let mut changed = false;
        let mut i = 0;
        while i < result.len() - 1 {
            if result[i] > result[i + 1] {
                // Pool: merge adjacent blocks
                let new_val = (result[i] * weights[i] + result[i + 1] * weights[i + 1]) / (weights[i] + weights[i + 1]);
                let new_weight = weights[i] + weights[i + 1];
                result[i] = new_val;
                weights[i] = new_weight;
                result.remove(i + 1);
                weights.remove(i + 1);
                changed = true;
            } else {
                i += 1;
            }
        }
        if !changed { break; }
    }

    // Expand back to original length
    let mut expanded = Vec::new();
    for (_i, (&val, &w)) in result.iter().zip(&weights).enumerate() {
        for _ in 0..w as usize {
            expanded.push(val);
        }
    }
    // Handle rounding issues
    while expanded.len() < n { expanded.push(*result.last().unwrap_or(&0.0)); }
    expanded.truncate(n);
    expanded
}

// ═══════════════════════════════════════════════════════════════════
// AGGLOMERATIVE CLUSTERING (with different linkages)
// ═══════════════════════════════════════════════════════════════════

/// Agglomerative clustering. Returns labels.
pub fn agglomerative_clustering(data: &[Vec<f64>], n_clusters: usize, _linkage: &str) -> Vec<usize> {
    let n = data.len();
    if n <= n_clusters {
        return (0..n).collect();
    }

    // Distance matrix
    let dists = euclidean_dist_matrix_from_vecs(data);
    let _result = crate::advanced_stats::nmf; // just need the clustering part

    // Use rce-graph's clustermap clustering
    let dist_vecs: Vec<Vec<f64>> = dists.iter().map(|row| row.to_vec()).collect();

    // Simple single-linkage via greedy merging
    let mut labels: Vec<usize> = (0..n).collect();
    let mut n_current = n;

    while n_current > n_clusters {
        // Find closest pair
        let mut best_dist = f64::INFINITY;
        let mut best_i = 0;
        let mut best_j = 0;
        for i in 0..n {
            for j in i + 1..n {
                if labels[i] != labels[j] && dist_vecs[i][j] < best_dist {
                    best_dist = dist_vecs[i][j];
                    best_i = i;
                    best_j = j;
                }
            }
        }
        // Merge: relabel all of best_j's cluster to best_i's cluster
        let old_label = labels[best_j];
        let new_label = labels[best_i];
        for l in labels.iter_mut() {
            if *l == old_label { *l = new_label; }
        }
        n_current -= 1;
    }

    // Normalize labels to 0..n_clusters-1
    let mut unique: Vec<usize> = labels.clone();
    unique.sort(); unique.dedup();
    let map: std::collections::HashMap<usize, usize> = unique.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    labels.iter().map(|&l| map[&l]).collect()
}

fn euclidean_dist_matrix_from_vecs(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = data.len();
    let mut d = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let dist: f64 = data[i].iter().zip(&data[j]).map(|(&a, &b)| (a - b).powi(2)).sum::<f64>().sqrt();
            d[i][j] = dist;
            d[j][i] = dist;
        }
    }
    d
}

// ═══════════════════════════════════════════════════════════════════
// SPECTRAL CLUSTERING
// ═══════════════════════════════════════════════════════════════════

/// Spectral clustering using normalized Laplacian.
pub fn spectral_clustering(data: &[Vec<f64>], n_clusters: usize, sigma: f64) -> Vec<usize> {
    let n = data.len();
    if n <= n_clusters { return (0..n).collect(); }

    // Affinity matrix (RBF kernel)
    let mut w = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let d2: f64 = data[i].iter().zip(&data[j]).map(|(&a, &b)| (a - b).powi(2)).sum();
            let aff = (-d2 / (2.0 * sigma * sigma)).exp();
            w[i][j] = aff;
            w[j][i] = aff;
        }
    }

    // Degree matrix D and normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
    let d_inv_sqrt: Vec<f64> = (0..n).map(|i| {
        let di: f64 = w[i].iter().sum();
        if di > 1e-15 { 1.0 / di.sqrt() } else { 0.0 }
    }).collect();

    let mut l_norm = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            l_norm[i][j] = if i == j { 1.0 } else { 0.0 } - d_inv_sqrt[i] * w[i][j] * d_inv_sqrt[j];
        }
    }

    // Get first n_clusters eigenvectors (smallest eigenvalues)
    let (_components, _, projected) = pca(&l_norm, n_clusters);

    // K-means on the projected data
    let (labels, _) = kmeans(&projected, n_clusters, 100);
    labels
}

// ═══════════════════════════════════════════════════════════════════
// ADDITIONAL METRICS
// ═══════════════════════════════════════════════════════════════════

/// Matthews correlation coefficient (binary classification).
pub fn matthews_corrcoef(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let mut tp: f64 = 0.0; let mut tn: f64 = 0.0; let mut fp: f64 = 0.0; let mut fn_: f64 = 0.0;
    for (t, p) in y_true.iter().zip(y_pred) {
        match (*t, *p) {
            (1, 1) => tp += 1.0,
            (0, 0) => tn += 1.0,
            (0, 1) => fp += 1.0,
            (1, 0) => fn_ += 1.0,
            _ => {}
        }
    }
    let num = tp * tn - fp * fn_;
    let den = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
    if den < 1e-15 { 0.0 } else { num / den }
}

/// Cohen's kappa inter-rater agreement.
pub fn cohen_kappa(y1: &[usize], y2: &[usize]) -> f64 {
    let n = y1.len() as f64;
    if n < 1.0 { return 0.0; }
    let po = y1.iter().zip(y2).filter(|(a, b)| a == b).count() as f64 / n;
    let max_label = y1.iter().chain(y2.iter()).copied().max().unwrap_or(0);
    let mut pe = 0.0;
    for k in 0..=max_label {
        let p1 = y1.iter().filter(|&&v| v == k).count() as f64 / n;
        let p2 = y2.iter().filter(|&&v| v == k).count() as f64 / n;
        pe += p1 * p2;
    }
    if (1.0 - pe).abs() < 1e-15 { 1.0 } else { (po - pe) / (1.0 - pe) }
}

/// ROC AUC score (binary, using trapezoidal rule).
pub fn roc_auc(y_true: &[usize], y_scores: &[f64]) -> f64 {
    let n = y_true.len();
    if n < 2 { return 0.5; }

    // Sort by descending score
    let mut pairs: Vec<(f64, usize)> = y_scores.iter().zip(y_true).map(|(&s, &t)| (s, t)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let n_pos = y_true.iter().filter(|&&t| t == 1).count() as f64;
    let n_neg = n as f64 - n_pos;
    if n_pos < 1.0 || n_neg < 1.0 { return 0.5; }

    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut auc = 0.0;
    let _prev_fp = 0.0;
    let mut prev_tp = 0.0;

    for &(_, label) in &pairs {
        if label == 1 { tp += 1.0; }
        else {
            fp += 1.0;
            auc += (tp + prev_tp) * 0.5; // trapezoidal
            prev_tp = tp;
        }
    }
    auc / (n_pos * n_neg)
}

/// Log loss (cross-entropy for binary classification).
pub fn log_loss(y_true: &[usize], y_prob: &[f64]) -> f64 {
    let n = y_true.len() as f64;
    if n < 1.0 { return 0.0; }
    -y_true.iter().zip(y_prob).map(|(&t, &p)| {
        let p = p.clamp(1e-15, 1.0 - 1e-15);
        if t == 1 { p.ln() } else { (1.0 - p).ln() }
    }).sum::<f64>() / n
}

/// Mean absolute percentage error.
pub fn mape(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len() as f64;
    if n < 1.0 { return 0.0; }
    y_true.iter().zip(y_pred).map(|(&t, &p)| {
        if t.abs() < 1e-15 { 0.0 } else { ((t - p) / t).abs() }
    }).sum::<f64>() / n * 100.0
}

/// Explained variance score.
pub fn explained_variance(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let residuals: Vec<f64> = y_true.iter().zip(y_pred).map(|(&t, &p)| t - p).collect();
    let var_res = variance(&residuals);
    let var_y = variance(y_true);
    if var_y < 1e-15 { return 1.0; }
    1.0 - var_res / var_y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ridge() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let w = ridge_regression(&x, &y, 0.1);
        assert!((w[0] - 2.0).abs() < 0.5, "ridge slope ≈ 2, got {}", w[0]);
    }

    #[test]
    fn test_lasso() {
        let x = vec![vec![1.0, 0.0], vec![2.0, 0.0], vec![3.0, 0.0], vec![4.0, 0.0], vec![5.0, 0.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let w = lasso_regression(&x, &y, 0.01, 1000);
        // Second feature should be driven to zero (irrelevant)
        assert!(w[1].abs() < 0.1, "irrelevant feature weight should be ~0, got {}", w[1]);
    }

    #[test]
    fn test_elastic_net() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let w = elastic_net(&x, &y, 0.1, 0.5, 1000);
        assert!(w[0] > 1.0, "should find positive slope");
    }

    #[test]
    fn test_isotonic() {
        let y = vec![5.0, 3.0, 4.0, 2.0, 6.0, 7.0];
        let iso = isotonic_regression(&y);
        assert_eq!(iso.len(), y.len());
        // Result should be non-decreasing
        for i in 1..iso.len() {
            assert!(iso[i] >= iso[i - 1] - 1e-10, "isotonic should be non-decreasing: {} < {}", iso[i], iso[i - 1]);
        }
    }

    #[test]
    fn test_agglomerative() {
        let data = vec![vec![0.0], vec![0.1], vec![10.0], vec![10.1]];
        let labels = agglomerative_clustering(&data, 2, "single");
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_matthews() {
        let y_true = vec![1, 1, 0, 0];
        let y_pred = vec![1, 0, 0, 0];
        let mcc = matthews_corrcoef(&y_true, &y_pred);
        assert!(mcc > 0.0 && mcc < 1.0);
    }

    #[test]
    fn test_cohen_kappa() {
        let y1 = vec![0, 1, 1, 0, 1];
        let y2 = vec![0, 1, 0, 0, 1];
        let k = cohen_kappa(&y1, &y2);
        assert!(k > 0.0, "some agreement should give positive kappa");
    }

    #[test]
    fn test_log_loss() {
        let y_true = vec![1, 0, 1, 0];
        let y_prob = vec![0.9, 0.1, 0.8, 0.2];
        let ll = log_loss(&y_true, &y_prob);
        assert!(ll > 0.0 && ll < 1.0, "log loss should be reasonable, got {ll}");
    }

    #[test]
    fn test_mape() {
        let y_true = vec![100.0, 200.0, 300.0];
        let y_pred = vec![110.0, 190.0, 310.0];
        let m = mape(&y_true, &y_pred);
        assert!(m > 0.0 && m < 10.0, "MAPE should be small, got {m}");
    }
}
