//! Sampling design: sample size calculation, power analysis,
//! survey sampling methods, design effects, margin of error.


// ═══════════════════════════════════════════════════════════════════
// SAMPLE SIZE CALCULATIONS
// ═══════════════════════════════════════════════════════════════════

/// Sample size for estimating a proportion with given margin of error.
/// p = estimated proportion, e = margin of error, z = z-score for confidence.
pub fn sample_size_proportion(p: f64, margin: f64, z: f64) -> usize {
    let n = z * z * p * (1.0 - p) / (margin * margin);
    n.ceil() as usize
}

/// Sample size for estimating a mean with given margin of error.
/// sigma = population std dev, e = margin of error.
pub fn sample_size_mean(sigma: f64, margin: f64, z: f64) -> usize {
    let n = (z * sigma / margin).powi(2);
    n.ceil() as usize
}

/// Sample size for two-sample t-test.
/// d = effect size (Cohen's d), alpha = significance, power = desired power.
pub fn sample_size_ttest(d: f64, alpha: f64, power: f64) -> usize {
    let z_alpha = z_score(1.0 - alpha / 2.0);
    let z_beta = z_score(power);
    let n = 2.0 * ((z_alpha + z_beta) / d).powi(2);
    n.ceil() as usize
}

/// Sample size for chi-squared test.
/// w = effect size, df = degrees of freedom.
pub fn sample_size_chi2(w: f64, alpha: f64, power: f64, _df: usize) -> usize {
    let z_alpha = z_score(1.0 - alpha / 2.0);
    let z_beta = z_score(power);
    let n = ((z_alpha + z_beta) / w).powi(2);
    n.ceil() as usize
}

/// Sample size with finite population correction.
pub fn sample_size_finite(n_infinite: usize, population: usize) -> usize {
    let ni = n_infinite as f64;
    let pop = population as f64;
    let n = ni * pop / (ni + pop - 1.0);
    n.ceil() as usize
}

// ═══════════════════════════════════════════════════════════════════
// POWER ANALYSIS
// ═══════════════════════════════════════════════════════════════════

/// Statistical power for two-sample t-test.
pub fn power_ttest(n_per_group: usize, d: f64, alpha: f64) -> f64 {
    let z_alpha = z_score(1.0 - alpha / 2.0);
    let ncp = d * (n_per_group as f64 / 2.0).sqrt(); // non-centrality parameter
    let z_beta = ncp - z_alpha;
    normal_cdf(z_beta)
}

/// Power for one-sample proportion test.
pub fn power_proportion(n: usize, p0: f64, p1: f64, alpha: f64) -> f64 {
    let z_alpha = z_score(1.0 - alpha / 2.0);
    let se0 = (p0 * (1.0 - p0) / n as f64).sqrt();
    let se1 = (p1 * (1.0 - p1) / n as f64).sqrt();
    let z = (p1 - p0).abs() / se1 - z_alpha * se0 / se1;
    normal_cdf(z)
}

/// Minimum detectable effect size given n, alpha, power.
pub fn min_detectable_effect(n_per_group: usize, alpha: f64, power: f64) -> f64 {
    let z_alpha = z_score(1.0 - alpha / 2.0);
    let z_beta = z_score(power);
    (z_alpha + z_beta) * (2.0 / n_per_group as f64).sqrt()
}

// ═══════════════════════════════════════════════════════════════════
// MARGIN OF ERROR & CONFIDENCE
// ═══════════════════════════════════════════════════════════════════

/// Margin of error for a proportion.
pub fn margin_of_error_proportion(p: f64, n: usize, z: f64) -> f64 {
    z * (p * (1.0 - p) / n as f64).sqrt()
}

/// Margin of error for a mean.
pub fn margin_of_error_mean(sigma: f64, n: usize, z: f64) -> f64 {
    z * sigma / (n as f64).sqrt()
}

/// Confidence interval for a proportion (Wald).
pub fn ci_proportion(p: f64, n: usize, z: f64) -> (f64, f64) {
    let me = margin_of_error_proportion(p, n, z);
    ((p - me).max(0.0), (p + me).min(1.0))
}

/// Confidence interval for a mean.
pub fn ci_mean(mean: f64, sigma: f64, n: usize, z: f64) -> (f64, f64) {
    let me = margin_of_error_mean(sigma, n, z);
    (mean - me, mean + me)
}

/// Wilson score interval for a proportion (better for small n or extreme p).
pub fn ci_wilson(successes: usize, n: usize, z: f64) -> (f64, f64) {
    let p_hat = successes as f64 / n as f64;
    let z2 = z * z;
    let denom = 1.0 + z2 / n as f64;
    let center = (p_hat + z2 / (2.0 * n as f64)) / denom;
    let half_width = z * (p_hat * (1.0 - p_hat) / n as f64 + z2 / (4.0 * n as f64 * n as f64)).sqrt() / denom;
    ((center - half_width).max(0.0), (center + half_width).min(1.0))
}

/// Clopper-Pearson exact confidence interval for a proportion.
pub fn ci_clopper_pearson(successes: usize, n: usize, alpha: f64) -> (f64, f64) {
    // Uses beta distribution quantiles (approximation via normal)
    let lo = if successes == 0 { 0.0 }
             else {
                 let z = z_score(alpha / 2.0);
                 let p = successes as f64 / n as f64;
                 (p + z * (p * (1.0 - p) / n as f64).sqrt()).max(0.0)
             };
    let hi = if successes == n { 1.0 }
             else {
                 let z = z_score(1.0 - alpha / 2.0);
                 let p = successes as f64 / n as f64;
                 (p + z * (p * (1.0 - p) / n as f64).sqrt()).min(1.0)
             };
    (lo, hi)
}

// ═══════════════════════════════════════════════════════════════════
// DESIGN EFFECTS
// ═══════════════════════════════════════════════════════════════════

/// Design effect for cluster sampling: DEFF = 1 + (m-1)·ρ.
/// m = cluster size, rho = intra-cluster correlation.
pub fn design_effect_cluster(cluster_size: f64, icc: f64) -> f64 {
    1.0 + (cluster_size - 1.0) * icc
}

/// Effective sample size: n_eff = n / DEFF.
pub fn effective_sample_size(n: usize, deff: f64) -> f64 {
    n as f64 / deff
}

/// Required sample size with design effect.
pub fn sample_size_with_deff(n_srs: usize, deff: f64) -> usize {
    (n_srs as f64 * deff).ceil() as usize
}

// ═══════════════════════════════════════════════════════════════════
// SURVEY SAMPLING
// ═══════════════════════════════════════════════════════════════════

/// Systematic sampling: select every k-th element.
pub fn systematic_sample(n: usize, k: usize, start: usize) -> Vec<usize> {
    (start..n).step_by(k).collect()
}

/// Proportional allocation for stratified sampling.
/// stratum_sizes: population size per stratum, total_n: total sample size.
pub fn proportional_allocation(stratum_sizes: &[usize], total_n: usize) -> Vec<usize> {
    let total_pop: usize = stratum_sizes.iter().sum();
    let mut alloc: Vec<usize> = stratum_sizes.iter()
        .map(|&s| (s as f64 / total_pop as f64 * total_n as f64).round() as usize)
        .collect();
    // Adjust for rounding
    let sum: usize = alloc.iter().sum();
    if sum < total_n && !alloc.is_empty() { alloc[0] += total_n - sum; }
    else if sum > total_n && !alloc.is_empty() {
        let diff = sum - total_n;
        for i in 0..diff.min(alloc.len()) { if alloc[i] > 0 { alloc[i] -= 1; } }
    }
    alloc
}

/// Neyman (optimal) allocation for stratified sampling.
/// stratum_sizes and stratum_stds: population params per stratum.
pub fn neyman_allocation(stratum_sizes: &[usize], stratum_stds: &[f64], total_n: usize) -> Vec<usize> {
    let weighted: Vec<f64> = stratum_sizes.iter().zip(stratum_stds)
        .map(|(&n, &s)| n as f64 * s).collect();
    let total_weighted: f64 = weighted.iter().sum();
    if total_weighted < 1e-15 { return proportional_allocation(stratum_sizes, total_n); }
    weighted.iter().map(|&w| (w / total_weighted * total_n as f64).round() as usize).collect()
}

/// Response rate required to achieve target effective sample size.
pub fn required_response_rate(target_n: usize, initial_n: usize) -> f64 {
    target_n as f64 / initial_n as f64
}

/// Oversample to account for expected non-response.
pub fn oversample_size(target_n: usize, expected_response_rate: f64) -> usize {
    (target_n as f64 / expected_response_rate).ceil() as usize
}

// ═══════════════════════════════════════════════════════════════════
// Z-SCORES & NORMAL CDF
// ═══════════════════════════════════════════════════════════════════

/// Z-score for given cumulative probability (inverse normal CDF, Abramowitz & Stegun).
pub fn z_score(p: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    if (p - 0.5).abs() < 1e-15 { return 0.0; }
    let t = if p < 0.5 { (-2.0 * p.ln()).sqrt() } else { (-2.0 * (1.0 - p).ln()).sqrt() };
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;
    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    if p < 0.5 { -z } else { z }
}

fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let r = 1.0 - poly * (-x * x).exp();
    if x < 0.0 { -r } else { r }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_size_proportion() {
        // 95% CI, p=0.5, margin=0.05 → ~385
        let n = sample_size_proportion(0.5, 0.05, 1.96);
        assert!(n >= 380 && n <= 390);
    }

    #[test]
    fn test_sample_size_mean() {
        // σ=10, margin=1, z=1.96 → ~385
        let n = sample_size_mean(10.0, 1.0, 1.96);
        assert!(n >= 380 && n <= 390);
    }

    #[test]
    fn test_sample_size_ttest() {
        // Medium effect d=0.5, alpha=0.05, power=0.8 → ~64 per group
        let n = sample_size_ttest(0.5, 0.05, 0.8);
        assert!(n >= 50 && n <= 80);
    }

    #[test]
    fn test_power() {
        let p = power_ttest(100, 0.5, 0.05);
        assert!(p > 0.8); // should have good power
    }

    #[test]
    fn test_margin_of_error() {
        let me = margin_of_error_proportion(0.5, 1000, 1.96);
        assert!((me - 0.031).abs() < 0.005);
    }

    #[test]
    fn test_ci_proportion() {
        let (lo, hi) = ci_proportion(0.5, 100, 1.96);
        assert!(lo < 0.5 && hi > 0.5);
    }

    #[test]
    fn test_wilson() {
        let (lo, hi) = ci_wilson(50, 100, 1.96);
        assert!(lo < 0.5 && hi > 0.5);
    }

    #[test]
    fn test_design_effect() {
        assert!((design_effect_cluster(10.0, 0.05) - 1.45).abs() < 1e-10);
    }

    #[test]
    fn test_systematic() {
        let s = systematic_sample(20, 5, 2);
        assert_eq!(s, vec![2, 7, 12, 17]);
    }

    #[test]
    fn test_proportional_alloc() {
        let alloc = proportional_allocation(&[500, 300, 200], 100);
        assert_eq!(alloc.iter().sum::<usize>(), 100);
    }

    #[test]
    fn test_oversample() {
        assert_eq!(oversample_size(100, 0.8), 125);
    }

    #[test]
    fn test_z_score() {
        assert!((z_score(0.975) - 1.96).abs() < 0.01);
        assert!((z_score(0.5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_finite_correction() {
        let n = sample_size_finite(384, 1000);
        assert!(n < 384); // smaller than infinite case
    }
}
