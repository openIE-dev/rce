//! Extended regression: quantile regression, robust regression (Huber, RANSAC),
//! polynomial regression, kernel regression, spline regression,
//! generalized linear models (logistic, Poisson), stepwise selection.

// ═══════════════════════════════════════════════════════════════════
// POLYNOMIAL REGRESSION
// ═══════════════════════════════════════════════════════════════════

/// Polynomial regression: fit y = β₀ + β₁x + β₂x² + ... + βₙxⁿ.
/// Returns coefficients [β₀, β₁, ..., βₙ].
pub fn polyregress(x: &[f64], y: &[f64], degree: usize) -> Vec<f64> {
    let n = x.len();
    let k = degree + 1;
    // Build Vandermonde matrix
    let mut xmat = vec![0.0; n * k];
    for i in 0..n {
        let mut xp = 1.0;
        for j in 0..k {
            xmat[i * k + j] = xp;
            xp *= x[i];
        }
    }
    crate::linalg_decomp::qr_solve(&xmat, y, n, k)
}

/// Predict from polynomial coefficients.
pub fn polyregress_predict(coeffs: &[f64], x: &[f64]) -> Vec<f64> {
    x.iter().map(|&xi| {
        let mut val = 0.0;
        let mut xp = 1.0;
        for &c in coeffs { val += c * xp; xp *= xi; }
        val
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// QUANTILE REGRESSION (iteratively reweighted least squares)
// ═══════════════════════════════════════════════════════════════════

/// Quantile regression. x: n×k (with intercept), y: n, quantile: τ ∈ (0,1).
/// Returns coefficients.
pub fn quantile_regression(x: &[f64], y: &[f64], n: usize, k: usize, quantile: f64, max_iter: usize) -> Vec<f64> {
    // Start with OLS
    let mut coeffs = crate::linalg_decomp::qr_solve(x, y, n, k);
    for _ in 0..max_iter {
        // Compute residuals
        let mut weights = vec![0.0; n];
        for i in 0..n {
            let pred: f64 = (0..k).map(|j| x[i * k + j] * coeffs[j]).sum();
            let resid = y[i] - pred;
            let eps = 1e-6;
            weights[i] = if resid > 0.0 { quantile / resid.abs().max(eps) }
                         else { (1.0 - quantile) / resid.abs().max(eps) };
        }
        // WLS step
        let mut xw = vec![0.0; n * k];
        let mut yw = vec![0.0; n];
        for i in 0..n {
            let w = weights[i].sqrt();
            yw[i] = y[i] * w;
            for j in 0..k { xw[i * k + j] = x[i * k + j] * w; }
        }
        coeffs = crate::linalg_decomp::qr_solve(&xw, &yw, n, k);
    }
    coeffs
}

// ═══════════════════════════════════════════════════════════════════
// ROBUST REGRESSION
// ═══════════════════════════════════════════════════════════════════

/// Huber regression (M-estimator). delta = Huber threshold.
pub fn huber_regression(x: &[f64], y: &[f64], n: usize, k: usize, delta: f64, max_iter: usize) -> Vec<f64> {
    let mut coeffs = crate::linalg_decomp::qr_solve(x, y, n, k);
    for _ in 0..max_iter {
        let mut weights = vec![1.0; n];
        for i in 0..n {
            let pred: f64 = (0..k).map(|j| x[i * k + j] * coeffs[j]).sum();
            let resid = (y[i] - pred).abs();
            if resid > delta { weights[i] = delta / resid; }
        }
        let mut xw = vec![0.0; n * k];
        let mut yw = vec![0.0; n];
        for i in 0..n {
            let w = weights[i].sqrt();
            yw[i] = y[i] * w;
            for j in 0..k { xw[i * k + j] = x[i * k + j] * w; }
        }
        coeffs = crate::linalg_decomp::qr_solve(&xw, &yw, n, k);
    }
    coeffs
}

/// RANSAC regression (random sample consensus).
pub fn ransac_regression(x: &[f64], y: &[f64], n: usize, k: usize, n_iter: usize, threshold: f64, seed: u64) -> Vec<f64> {
    let mut state = seed_state(seed);
    let mut best_coeffs = vec![0.0; k];
    let mut best_inliers = 0;

    for _ in 0..n_iter {
        // Random sample of k points
        let mut sample_idx: Vec<usize> = Vec::new();
        while sample_idx.len() < k {
            let idx = prng_next(&mut state) as usize % n;
            if !sample_idx.contains(&idx) { sample_idx.push(idx); }
        }
        // Fit on sample
        let mut xs = vec![0.0; k * k];
        let mut ys = vec![0.0; k];
        for (si, &idx) in sample_idx.iter().enumerate() {
            for j in 0..k { xs[si * k + j] = x[idx * k + j]; }
            ys[si] = y[idx];
        }
        let coeffs = crate::linalg_decomp::lu_solve(&xs, &ys, k);
        // Count inliers
        let inliers: usize = (0..n).filter(|&i| {
            let pred: f64 = (0..k).map(|j| x[i * k + j] * coeffs[j]).sum();
            (y[i] - pred).abs() < threshold
        }).count();
        if inliers > best_inliers { best_inliers = inliers; best_coeffs = coeffs; }
    }
    best_coeffs
}

/// Theil-Sen estimator (median of pairwise slopes, 1D).
pub fn theil_sen(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len();
    let mut slopes = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            if (x[j] - x[i]).abs() > 1e-15 { slopes.push((y[j] - y[i]) / (x[j] - x[i])); }
        }
    }
    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let slope = if slopes.is_empty() { 0.0 }
        else if slopes.len() % 2 == 0 { (slopes[slopes.len()/2 - 1] + slopes[slopes.len()/2]) / 2.0 }
        else { slopes[slopes.len()/2] };
    let mut intercepts: Vec<f64> = x.iter().zip(y).map(|(&xi, &yi)| yi - slope * xi).collect();
    intercepts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let intercept = if intercepts.len() % 2 == 0 { (intercepts[intercepts.len()/2 - 1] + intercepts[intercepts.len()/2]) / 2.0 }
        else { intercepts[intercepts.len()/2] };
    (intercept, slope)
}

// ═══════════════════════════════════════════════════════════════════
// KERNEL REGRESSION (Nadaraya-Watson)
// ═══════════════════════════════════════════════════════════════════

/// Nadaraya-Watson kernel regression with Gaussian kernel.
pub fn kernel_regression(x_train: &[f64], y_train: &[f64], x_pred: &[f64], bandwidth: f64) -> Vec<f64> {
    x_pred.iter().map(|&xp| {
        let mut num = 0.0;
        let mut den = 0.0;
        for (&xi, &yi) in x_train.iter().zip(y_train) {
            let k = (-0.5 * ((xp - xi) / bandwidth).powi(2)).exp();
            num += k * yi;
            den += k;
        }
        if den > 1e-15 { num / den } else { 0.0 }
    }).collect()
}

/// LOWESS (Locally Weighted Scatterplot Smoothing).
pub fn lowess(x: &[f64], y: &[f64], x_pred: &[f64], frac: f64) -> Vec<f64> {
    let n = x.len();
    let window = (frac * n as f64).ceil() as usize;
    x_pred.iter().map(|&xp| {
        let mut dists: Vec<(usize, f64)> = (0..n).map(|i| (i, (x[i] - xp).abs())).collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let max_dist = dists[window.min(n) - 1].1.max(1e-15);
        let mut sw = 0.0;
        let mut swx = 0.0;
        let mut swy = 0.0;
        let mut swxx = 0.0;
        let mut swxy = 0.0;
        for &(i, d) in dists.iter().take(window) {
            let u = d / max_dist;
            let w = if u < 1.0 { (1.0 - u * u * u).powi(3) } else { 0.0 };
            sw += w;
            swx += w * x[i];
            swy += w * y[i];
            swxx += w * x[i] * x[i];
            swxy += w * x[i] * y[i];
        }
        let det = sw * swxx - swx * swx;
        if det.abs() < 1e-15 { return swy / sw.max(1e-15); }
        let beta0 = (swxx * swy - swx * swxy) / det;
        let beta1 = (sw * swxy - swx * swy) / det;
        beta0 + beta1 * xp
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// LOGISTIC REGRESSION (GLM)
// ═══════════════════════════════════════════════════════════════════

/// Logistic regression via IRLS. x: n×k (with intercept), y: binary {0,1}.
/// Returns coefficients.
pub fn logistic_regression(x: &[f64], y: &[f64], n: usize, k: usize, max_iter: usize) -> Vec<f64> {
    let mut beta = vec![0.0; k];
    for _ in 0..max_iter {
        let mut mu = vec![0.0; n];
        for i in 0..n {
            let z: f64 = (0..k).map(|j| x[i * k + j] * beta[j]).sum();
            mu[i] = 1.0 / (1.0 + (-z).exp());
        }
        // IRLS weights and adjusted dependent
        let mut xw = vec![0.0; n * k];
        let mut z_adj = vec![0.0; n];
        for i in 0..n {
            let w = (mu[i] * (1.0 - mu[i])).max(1e-10).sqrt();
            let eta: f64 = (0..k).map(|j| x[i * k + j] * beta[j]).sum();
            z_adj[i] = (eta + (y[i] - mu[i]) / (mu[i] * (1.0 - mu[i])).max(1e-10)) * w;
            for j in 0..k { xw[i * k + j] = x[i * k + j] * w; }
        }
        let new_beta = crate::linalg_decomp::qr_solve(&xw, &z_adj, n, k);
        let diff: f64 = beta.iter().zip(&new_beta).map(|(a, b)| (a - b).abs()).sum();
        beta = new_beta;
        if diff < 1e-8 { break; }
    }
    beta
}

/// Predict probabilities from logistic regression.
pub fn logistic_predict_proba(x: &[f64], coeffs: &[f64], n: usize, k: usize) -> Vec<f64> {
    (0..n).map(|i| {
        let z: f64 = (0..k).map(|j| x[i * k + j] * coeffs[j]).sum();
        1.0 / (1.0 + (-z).exp())
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// POISSON REGRESSION (GLM)
// ═══════════════════════════════════════════════════════════════════

/// Poisson regression via IRLS. y: non-negative counts.
pub fn poisson_regression(x: &[f64], y: &[f64], n: usize, k: usize, max_iter: usize) -> Vec<f64> {
    let mut beta = vec![0.0; k];
    for _ in 0..max_iter {
        let mut mu = vec![0.0; n];
        for i in 0..n {
            let z: f64 = (0..k).map(|j| x[i * k + j] * beta[j]).sum();
            mu[i] = z.exp().min(1e10);
        }
        let mut xw = vec![0.0; n * k];
        let mut z_adj = vec![0.0; n];
        for i in 0..n {
            let w = mu[i].max(1e-10).sqrt();
            let eta: f64 = (0..k).map(|j| x[i * k + j] * beta[j]).sum();
            z_adj[i] = (eta + (y[i] - mu[i]) / mu[i].max(1e-10)) * w;
            for j in 0..k { xw[i * k + j] = x[i * k + j] * w; }
        }
        let new_beta = crate::linalg_decomp::qr_solve(&xw, &z_adj, n, k);
        let diff: f64 = beta.iter().zip(&new_beta).map(|(a, b)| (a - b).abs()).sum();
        beta = new_beta;
        if diff < 1e-8 { break; }
    }
    beta
}

// ═══════════════════════════════════════════════════════════════════
// STEPWISE SELECTION
// ═══════════════════════════════════════════════════════════════════

/// Forward stepwise feature selection based on AIC.
/// x: n×p (no intercept), y: n. Returns selected feature indices.
pub fn forward_stepwise(x: &[f64], y: &[f64], n: usize, p: usize, max_features: usize) -> Vec<usize> {
    let mut selected: Vec<usize> = Vec::new();
    let mut remaining: Vec<usize> = (0..p).collect();
    let mut best_aic = f64::INFINITY;

    for _ in 0..max_features.min(p) {
        let mut best_feat = None;
        let mut best_feat_aic = best_aic;

        for &feat in &remaining {
            let mut trial = selected.clone();
            trial.push(feat);
            let k = trial.len() + 1; // +1 for intercept
            let x_trial = build_design(x, n, p, &trial);
            let result = crate::econometrics::ols(&x_trial, y, n, k);
            if result.aic < best_feat_aic {
                best_feat_aic = result.aic;
                best_feat = Some(feat);
            }
        }

        if let Some(feat) = best_feat {
            if best_feat_aic < best_aic - 2.0 { // AIC improvement > 2
                selected.push(feat);
                remaining.retain(|&f| f != feat);
                best_aic = best_feat_aic;
            } else { break; }
        } else { break; }
    }
    selected
}

fn build_design(x: &[f64], n: usize, p: usize, features: &[usize]) -> Vec<f64> {
    let k = features.len() + 1;
    let mut result = vec![0.0; n * k];
    for i in 0..n {
        result[i * k] = 1.0; // intercept
        for (j, &f) in features.iter().enumerate() {
            result[i * k + j + 1] = x[i * p + f];
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// REGRESSION DIAGNOSTICS
// ═══════════════════════════════════════════════════════════════════

/// Cook's distance for each observation.
pub fn cooks_distance(x: &[f64], residuals: &[f64], n: usize, k: usize) -> Vec<f64> {
    let xtx = {
        let mut m = vec![0.0; k * k];
        for i in 0..k { for j in 0..k { for s in 0..n { m[i*k+j] += x[s*k+i] * x[s*k+j]; } } }
        m
    };
    let xtx_inv = crate::linalg_decomp::matrix_inverse(&xtx, k);
    let mse = residuals.iter().map(|r| r * r).sum::<f64>() / (n - k) as f64;
    (0..n).map(|i| {
        // Leverage: h_ii = x_i' (X'X)^{-1} x_i
        let mut h = 0.0;
        for a in 0..k { for b in 0..k { h += x[i*k+a] * xtx_inv[a*k+b] * x[i*k+b]; } }
        let r = residuals[i];
        r * r * h / (k as f64 * mse * (1.0 - h).powi(2))
    }).collect()
}

/// Leverage (hat matrix diagonal).
pub fn leverage(x: &[f64], n: usize, k: usize) -> Vec<f64> {
    let xtx = {
        let mut m = vec![0.0; k * k];
        for i in 0..k { for j in 0..k { for s in 0..n { m[i*k+j] += x[s*k+i] * x[s*k+j]; } } }
        m
    };
    let xtx_inv = crate::linalg_decomp::matrix_inverse(&xtx, k);
    (0..n).map(|i| {
        let mut h = 0.0;
        for a in 0..k { for b in 0..k { h += x[i*k+a] * xtx_inv[a*k+b] * x[i*k+b]; } }
        h
    }).collect()
}

/// Studentized residuals.
pub fn studentized_residuals(residuals: &[f64], leverage: &[f64], n: usize, k: usize) -> Vec<f64> {
    let mse = residuals.iter().map(|r| r * r).sum::<f64>() / (n - k) as f64;
    residuals.iter().zip(leverage).map(|(r, h)| {
        let se = (mse * (1.0 - h)).sqrt();
        if se > 1e-15 { r / se } else { 0.0 }
    }).collect()
}

/// Variance Inflation Factor for each predictor.
pub fn vif(x: &[f64], n: usize, k: usize) -> Vec<f64> {
    // k includes intercept; compute VIF for columns 1..k
    (1..k).map(|j| {
        let y_j: Vec<f64> = (0..n).map(|i| x[i * k + j]).collect();
        let mut x_others = vec![0.0; n * (k - 1)];
        for i in 0..n {
            let mut col = 0;
            for jj in 0..k {
                if jj == j { continue; }
                x_others[i * (k - 1) + col] = x[i * k + jj];
                col += 1;
            }
        }
        let result = crate::econometrics::ols(&x_others, &y_j, n, k - 1);
        1.0 / (1.0 - result.r_squared).max(1e-15)
    }).collect()
}

fn prng_next(state: &mut [u64; 4]) -> u64 {
    let result = (state[0].wrapping_add(state[3])).rotate_left(23).wrapping_add(state[0]);
    let t = state[1] << 17;
    state[2] ^= state[0]; state[3] ^= state[1]; state[1] ^= state[2]; state[0] ^= state[3];
    state[2] ^= t; state[3] = state[3].rotate_left(45);
    result
}

fn seed_state(seed: u64) -> [u64; 4] {
    let mut s = [seed, seed.wrapping_mul(6364136223846793005).wrapping_add(1),
                 seed.wrapping_mul(1442695040888963407).wrapping_add(3),
                 seed.wrapping_mul(2862933555777941757).wrapping_add(7)];
    for _ in 0..20 { prng_next(&mut s); }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polyregress() {
        // y = 1 + 2x + 3x²
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + 3.0 * xi * xi).collect();
        let coeffs = polyregress(&x, &y, 2);
        assert!((coeffs[0] - 1.0).abs() < 0.1);
        assert!((coeffs[1] - 2.0).abs() < 0.1);
        assert!((coeffs[2] - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_theil_sen() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let (intercept, slope) = theil_sen(&x, &y);
        assert!((slope - 2.0).abs() < 1e-10);
        assert!((intercept - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_regression() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let pred = kernel_regression(&x, &y, &[2.5], 1.0);
        assert!(pred[0] > 5.0 && pred[0] < 8.0); // near x²(2.5) = 6.25
    }

    #[test]
    fn test_logistic_regression() {
        // Simple separable data
        let x = vec![1.0, -2.0, 1.0, -1.0, 1.0, 0.5, 1.0, 1.5, 1.0, 2.0]; // n=5, k=2
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        let coeffs = logistic_regression(&x, &y, 5, 2, 50);
        // Coefficient for x should be positive (higher x → class 1)
        assert!(coeffs[1] > 0.0);
    }

    #[test]
    fn test_lowess() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + (xi * 0.3).sin()).collect();
        let pred = lowess(&x, &y, &[10.0], 0.3);
        assert!((pred[0] - 20.0).abs() < 5.0);
    }

    #[test]
    fn test_cooks_distance() {
        let x = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let result = crate::econometrics::ols(&x, &y, 4, 2);
        let cd = cooks_distance(&x, &result.residuals, 4, 2);
        assert_eq!(cd.len(), 4);
    }
}
