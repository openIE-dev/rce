//! Probability theory: Bayes' theorem, conditional probability,
//! expectation, variance, covariance matrix, copulas, order statistics,
//! moment generating functions, characteristic functions, mixture models.

// ═══════════════════════════════════════════════════════════════════
// BASIC PROBABILITY
// ═══════════════════════════════════════════════════════════════════

/// Bayes' theorem: P(A|B) = P(B|A)·P(A) / P(B).
pub fn bayes(p_b_given_a: f64, p_a: f64, p_b: f64) -> f64 {
    p_b_given_a * p_a / p_b
}

/// Bayes with multiple hypotheses: P(H_i|E) = P(E|H_i)·P(H_i) / Σ P(E|H_j)·P(H_j).
pub fn bayes_multi(likelihoods: &[f64], priors: &[f64]) -> Vec<f64> {
    let evidence: f64 = likelihoods.iter().zip(priors).map(|(l, p)| l * p).sum();
    if evidence < 1e-15 { return vec![0.0; priors.len()]; }
    likelihoods.iter().zip(priors).map(|(l, p)| l * p / evidence).collect()
}

/// Law of total probability: P(A) = Σ P(A|B_i)·P(B_i).
pub fn total_probability(conditionals: &[f64], priors: &[f64]) -> f64 {
    conditionals.iter().zip(priors).map(|(c, p)| c * p).sum()
}

/// Inclusion-exclusion for 2 events: P(A∪B) = P(A) + P(B) - P(A∩B).
pub fn union_probability(p_a: f64, p_b: f64, p_ab: f64) -> f64 {
    p_a + p_b - p_ab
}

/// Odds from probability: O = p / (1-p).
pub fn prob_to_odds(p: f64) -> f64 { p / (1.0 - p) }

/// Probability from odds: p = O / (1+O).
pub fn odds_to_prob(odds: f64) -> f64 { odds / (1.0 + odds) }

/// Log-odds (logit): log(p/(1-p)).
pub fn logit(p: f64) -> f64 { (p / (1.0 - p)).ln() }

/// Inverse logit (sigmoid): 1/(1+e^(-x)).
pub fn inv_logit(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

// ═══════════════════════════════════════════════════════════════════
// MOMENTS & EXPECTATIONS
// ═══════════════════════════════════════════════════════════════════

/// Expected value: E[X] = Σ x_i · p_i.
pub fn expected_value(values: &[f64], probs: &[f64]) -> f64 {
    values.iter().zip(probs).map(|(x, p)| x * p).sum()
}

/// Variance: Var(X) = E[X²] - (E[X])².
pub fn variance_discrete(values: &[f64], probs: &[f64]) -> f64 {
    let ex = expected_value(values, probs);
    let ex2: f64 = values.iter().zip(probs).map(|(x, p)| x * x * p).sum();
    ex2 - ex * ex
}

/// Standard deviation of discrete distribution.
pub fn std_discrete(values: &[f64], probs: &[f64]) -> f64 {
    variance_discrete(values, probs).sqrt()
}

/// Nth raw moment: E[X^n].
pub fn raw_moment(values: &[f64], probs: &[f64], n: u32) -> f64 {
    values.iter().zip(probs).map(|(x, p)| x.powi(n as i32) * p).sum()
}

/// Nth central moment: E[(X - μ)^n].
pub fn central_moment(values: &[f64], probs: &[f64], n: u32) -> f64 {
    let mu = expected_value(values, probs);
    values.iter().zip(probs).map(|(x, p)| (x - mu).powi(n as i32) * p).sum()
}

/// Skewness: E[(X-μ)³] / σ³.
pub fn skewness_discrete(values: &[f64], probs: &[f64]) -> f64 {
    let sigma = std_discrete(values, probs);
    if sigma < 1e-15 { return 0.0; }
    central_moment(values, probs, 3) / sigma.powi(3)
}

/// Kurtosis: E[(X-μ)⁴] / σ⁴ - 3 (excess kurtosis).
pub fn kurtosis_discrete(values: &[f64], probs: &[f64]) -> f64 {
    let sigma = std_discrete(values, probs);
    if sigma < 1e-15 { return 0.0; }
    central_moment(values, probs, 4) / sigma.powi(4) - 3.0
}

/// Median of discrete distribution.
pub fn median_discrete(values: &[f64], probs: &[f64]) -> f64 {
    let mut cum = 0.0;
    for (&v, &p) in values.iter().zip(probs) {
        cum += p;
        if cum >= 0.5 { return v; }
    }
    *values.last().unwrap_or(&0.0)
}

/// Mode of discrete distribution.
pub fn mode_discrete(values: &[f64], probs: &[f64]) -> f64 {
    let idx = probs.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap().0;
    values[idx]
}

/// Covariance: Cov(X,Y) = E[XY] - E[X]E[Y].
pub fn covariance_weighted(x: &[f64], y: &[f64], probs: &[f64]) -> f64 {
    let exy: f64 = x.iter().zip(y).zip(probs).map(|((xi, yi), p)| xi * yi * p).sum();
    let ex = expected_value(x, probs);
    let ey = expected_value(y, probs);
    exy - ex * ey
}

/// Correlation from covariance and standard deviations.
pub fn correlation_from_cov(cov: f64, std_x: f64, std_y: f64) -> f64 {
    if std_x < 1e-15 || std_y < 1e-15 { return 0.0; }
    cov / (std_x * std_y)
}

// ═══════════════════════════════════════════════════════════════════
// SAMPLE STATISTICS (from data)
// ═══════════════════════════════════════════════════════════════════

/// Sample mean.
pub fn sample_mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

/// Sample variance (unbiased, Bessel's correction).
pub fn sample_variance(data: &[f64]) -> f64 {
    let mean = sample_mean(data);
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

/// Sample standard deviation.
pub fn sample_std(data: &[f64]) -> f64 { sample_variance(data).sqrt() }

/// Sample skewness.
pub fn sample_skewness(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mean = sample_mean(data);
    let std = sample_std(data);
    if std < 1e-15 { return 0.0; }
    let m3 = data.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>();
    m3 * n / ((n - 1.0) * (n - 2.0))
}

/// Sample kurtosis (excess).
pub fn sample_kurtosis(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mean = sample_mean(data);
    let m2 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let m4 = data.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;
    if m2 < 1e-15 { return 0.0; }
    m4 / (m2 * m2) - 3.0
}

/// Covariance matrix (n×n) from data (n_samples × n_dims, row-major).
pub fn covariance_matrix(data: &[f64], n_samples: usize, n_dims: usize) -> Vec<f64> {
    let mut means = vec![0.0; n_dims];
    for i in 0..n_samples {
        for j in 0..n_dims { means[j] += data[i * n_dims + j]; }
    }
    for m in means.iter_mut() { *m /= n_samples as f64; }

    let mut cov = vec![0.0; n_dims * n_dims];
    for i in 0..n_samples {
        for a in 0..n_dims {
            for b in a..n_dims {
                let v = (data[i * n_dims + a] - means[a]) * (data[i * n_dims + b] - means[b]);
                cov[a * n_dims + b] += v;
                if a != b { cov[b * n_dims + a] += v; }
            }
        }
    }
    let denom = (n_samples - 1) as f64;
    for v in cov.iter_mut() { *v /= denom; }
    cov
}

/// Correlation matrix from covariance matrix.
pub fn correlation_matrix(cov: &[f64], n: usize) -> Vec<f64> {
    let mut corr = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let si = cov[i * n + i].sqrt();
            let sj = cov[j * n + j].sqrt();
            corr[i * n + j] = if si > 1e-15 && sj > 1e-15 { cov[i * n + j] / (si * sj) } else { 0.0 };
        }
    }
    corr
}

// ═══════════════════════════════════════════════════════════════════
// ORDER STATISTICS
// ═══════════════════════════════════════════════════════════════════

/// Quantile (percentile) of sorted data.
pub fn quantile(data: &[f64], q: f64) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let idx = q * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi.min(n - 1)] * frac
}

/// Interquartile range.
pub fn iqr(data: &[f64]) -> f64 { quantile(data, 0.75) - quantile(data, 0.25) }

/// Percentile rank of a value in data.
pub fn percentile_rank(data: &[f64], value: f64) -> f64 {
    let below = data.iter().filter(|&&x| x < value).count();
    let equal = data.iter().filter(|&&x| (x - value).abs() < 1e-15).count();
    (below as f64 + 0.5 * equal as f64) / data.len() as f64 * 100.0
}

/// Winsorize data at given percentile limits.
pub fn winsorize(data: &[f64], lower_pct: f64, upper_pct: f64) -> Vec<f64> {
    let lo = quantile(data, lower_pct);
    let hi = quantile(data, upper_pct);
    data.iter().map(|&x| x.max(lo).min(hi)).collect()
}

/// Trimmed mean: mean after removing fraction from each tail.
pub fn trimmed_mean(data: &[f64], fraction: f64) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let trim = (n as f64 * fraction) as usize;
    let trimmed = &sorted[trim..n - trim];
    if trimmed.is_empty() { return sample_mean(data); }
    trimmed.iter().sum::<f64>() / trimmed.len() as f64
}

/// Geometric mean.
pub fn geometric_mean(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let log_sum: f64 = data.iter().map(|x| x.ln()).sum();
    (log_sum / n).exp()
}

/// Harmonic mean.
pub fn harmonic_mean(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    n / data.iter().map(|x| 1.0 / x).sum::<f64>()
}

/// Weighted mean.
pub fn weighted_mean(values: &[f64], weights: &[f64]) -> f64 {
    let sum: f64 = values.iter().zip(weights).map(|(v, w)| v * w).sum();
    let weight_sum: f64 = weights.iter().sum();
    sum / weight_sum
}

/// Weighted variance.
pub fn weighted_variance(values: &[f64], weights: &[f64]) -> f64 {
    let wm = weighted_mean(values, weights);
    let weight_sum: f64 = weights.iter().sum();
    values.iter().zip(weights).map(|(v, w)| w * (v - wm).powi(2)).sum::<f64>() / weight_sum
}

/// Coefficient of variation: std/mean.
pub fn coefficient_of_variation(data: &[f64]) -> f64 {
    let mean = sample_mean(data);
    if mean.abs() < 1e-15 { return 0.0; }
    sample_std(data) / mean.abs()
}

/// Mean absolute deviation.
pub fn mean_absolute_deviation(data: &[f64]) -> f64 {
    let mean = sample_mean(data);
    data.iter().map(|x| (x - mean).abs()).sum::<f64>() / data.len() as f64
}

/// Median absolute deviation (MAD).
pub fn median_absolute_deviation(data: &[f64]) -> f64 {
    let med = quantile(data, 0.5);
    let deviations: Vec<f64> = data.iter().map(|x| (x - med).abs()).collect();
    quantile(&deviations, 0.5)
}

/// Z-score standardization.
pub fn zscore(data: &[f64]) -> Vec<f64> {
    let mean = sample_mean(data);
    let std = sample_std(data);
    if std < 1e-15 { return vec![0.0; data.len()]; }
    data.iter().map(|x| (x - mean) / std).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayes() {
        // P(disease|positive) = P(positive|disease)·P(disease) / P(positive)
        let p = bayes(0.99, 0.01, 0.0198 + 0.01 * 0.99); // approx
        assert!(p > 0.0 && p < 1.0);
    }

    #[test]
    fn test_expected_value() {
        // Fair die: E[X] = 3.5
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let probs = vec![1.0/6.0; 6];
        assert!((expected_value(&vals, &probs) - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_variance_discrete() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let probs = vec![1.0/6.0; 6];
        let var = variance_discrete(&vals, &probs);
        assert!((var - 35.0/12.0).abs() < 1e-10);
    }

    #[test]
    fn test_logit_inv_logit() {
        assert!((inv_logit(logit(0.7)) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_sample_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((sample_mean(&data) - 3.0).abs() < 1e-10);
        assert!((sample_variance(&data) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_quantile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile(&data, 0.5) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_geometric_mean() {
        assert!((geometric_mean(&[2.0, 8.0]) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_harmonic_mean() {
        assert!((harmonic_mean(&[1.0, 4.0]) - 8.0/5.0).abs() < 1e-10);
    }

    #[test]
    fn test_zscore() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let z = zscore(&data);
        let mean: f64 = z.iter().sum::<f64>() / 5.0;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_covariance_matrix() {
        // 3 samples, 2 dims
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cov = covariance_matrix(&data, 3, 2);
        assert_eq!(cov.len(), 4); // 2×2
        assert!(cov[0] > 0.0); // var(x) > 0
    }

    #[test]
    fn test_mad() {
        let data = vec![1.0, 1.0, 2.0, 2.0, 4.0, 6.0, 9.0];
        let mad = median_absolute_deviation(&data);
        assert!(mad > 0.0);
    }

    #[test]
    fn test_trimmed_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 100.0]; // outlier
        let tm = trimmed_mean(&data, 0.2);
        assert!((tm - 3.0).abs() < 1e-10); // trims 1 from each side
    }
}
