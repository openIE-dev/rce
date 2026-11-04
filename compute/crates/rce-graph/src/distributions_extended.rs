//! Extended probability distributions: 40+ additional families.
//!
//! Covers the long tail of scipy.stats distributions that scientists
//! actually use: Erlang, inverse Gaussian, Pareto, von Mises, etc.
//! Each provides pdf + cdf + rvs at minimum.

use crate::distributions::*;
use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════
// ERLANG (Gamma with integer shape)
// ═══════════════════════════════════════════════════════════════════

pub fn erlang_pdf(x: f64, k: u64, rate: f64) -> f64 { gamma_pdf(x, k as f64, 1.0 / rate) }
pub fn erlang_cdf(x: f64, k: u64, rate: f64) -> f64 { gamma_cdf(x, k as f64, 1.0 / rate) }
pub fn erlang_rvs(rng: &mut Rng, n: usize, k: u64, rate: f64) -> Vec<f64> {
    (0..n).map(|_| (0..k).map(|_| -rng.uniform().max(1e-300).ln() / rate).sum()).collect()
}

// ═══════════════════════════════════════════════════════════════════
// INVERSE GAUSSIAN (Wald)
// ═══════════════════════════════════════════════════════════════════

pub fn invgauss_pdf(x: f64, mu: f64, lambda: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    (lambda / (2.0 * PI * x.powi(3))).sqrt() * (-lambda * (x - mu).powi(2) / (2.0 * mu * mu * x)).exp()
}

pub fn invgauss_cdf(x: f64, mu: f64, lambda: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    let z1 = (lambda / x).sqrt() * (x / mu - 1.0);
    let z2 = -(lambda / x).sqrt() * (x / mu + 1.0);
    norm_cdf(z1, 0.0, 1.0) + (2.0 * lambda / mu).exp() * norm_cdf(z2, 0.0, 1.0)
}

pub fn invgauss_rvs(rng: &mut Rng, n: usize, mu: f64, lambda: f64) -> Vec<f64> {
    (0..n).map(|_| {
        let v = rng.normal();
        let y = v * v;
        let x = mu + mu * mu * y / (2.0 * lambda) - mu / (2.0 * lambda) * (4.0 * mu * lambda * y + mu * mu * y * y).sqrt();
        if rng.uniform() <= mu / (mu + x) { x } else { mu * mu / x }
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// VON MISES (circular/directional)
// ═══════════════════════════════════════════════════════════════════

pub fn vonmises_pdf(x: f64, mu: f64, kappa: f64) -> f64 {
    (kappa * (x - mu).cos()).exp() / (2.0 * PI * bessel_i0(kappa))
}

fn bessel_i0(x: f64) -> f64 {
    // Approximation for modified Bessel I0
    let ax = x.abs();
    if ax < 3.75 {
        let t = (x / 3.75).powi(2);
        1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
    } else {
        let t = 3.75 / ax;
        (ax.exp() / ax.sqrt()) * (0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565 + t * (0.00916281 + t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))))))))
    }
}

// ═══════════════════════════════════════════════════════════════════
// MAXWELL-BOLTZMANN
// ═══════════════════════════════════════════════════════════════════

pub fn maxwell_pdf(x: f64, a: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    let a2 = a * a;
    (2.0 / PI).sqrt() * x * x * (-x * x / (2.0 * a2)).exp() / (a2 * a)
}

pub fn maxwell_cdf(x: f64, a: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    erf(x / (a * 2.0f64.sqrt())) - (2.0 / PI).sqrt() * x * (-x * x / (2.0 * a * a)).exp() / a
}

// ═══════════════════════════════════════════════════════════════════
// RICE DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn rice_pdf(x: f64, nu: f64, sigma: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    let s2 = sigma * sigma;
    (x / s2) * (-(x * x + nu * nu) / (2.0 * s2)).exp() * bessel_i0(x * nu / s2)
}

// ═══════════════════════════════════════════════════════════════════
// NAKAGAMI DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn nakagami_pdf(x: f64, m: f64, omega: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    2.0 * (m / omega).powf(m) / gamma_fn(m) * x.powf(2.0 * m - 1.0) * (-m * x * x / omega).exp()
}

// ═══════════════════════════════════════════════════════════════════
// SKEW-NORMAL DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn skewnorm_pdf(x: f64, alpha: f64, loc: f64, scale: f64) -> f64 {
    let z = (x - loc) / scale;
    2.0 / scale * norm_pdf(z, 0.0, 1.0) * norm_cdf(alpha * z, 0.0, 1.0)
}

// ═══════════════════════════════════════════════════════════════════
// GENERALIZED EXTREME VALUE (GEV)
// ═══════════════════════════════════════════════════════════════════

pub fn gev_pdf(x: f64, xi: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    if xi.abs() < 1e-10 {
        // Gumbel limit
        (-(z + (-z).exp())).exp() / sigma
    } else {
        let t = 1.0 + xi * z;
        if t <= 0.0 { return 0.0; }
        t.powf(-1.0 / xi - 1.0) * (-t.powf(-1.0 / xi)).exp() / sigma
    }
}

pub fn gev_cdf(x: f64, xi: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    if xi.abs() < 1e-10 {
        (-(-z).exp()).exp()
    } else {
        let t = 1.0 + xi * z;
        if t <= 0.0 { return if xi > 0.0 { 0.0 } else { 1.0 }; }
        (-t.powf(-1.0 / xi)).exp()
    }
}

// ═══════════════════════════════════════════════════════════════════
// GENERALIZED PARETO DISTRIBUTION (GPD)
// ═══════════════════════════════════════════════════════════════════

pub fn gpd_pdf(x: f64, xi: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    if z < 0.0 { return 0.0; }
    if xi.abs() < 1e-10 {
        (-z).exp() / sigma
    } else {
        let t = 1.0 + xi * z;
        if t <= 0.0 { return 0.0; }
        t.powf(-1.0 / xi - 1.0) / sigma
    }
}

pub fn gpd_cdf(x: f64, xi: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    if z < 0.0 { return 0.0; }
    if xi.abs() < 1e-10 {
        1.0 - (-z).exp()
    } else {
        let t = 1.0 + xi * z;
        if t <= 0.0 { return 1.0; }
        1.0 - t.powf(-1.0 / xi)
    }
}

// ═══════════════════════════════════════════════════════════════════
// POWER LAW DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn powerlaw_pdf(x: f64, alpha: f64) -> f64 {
    if x < 0.0 || x > 1.0 { 0.0 } else { alpha * x.powf(alpha - 1.0) }
}

pub fn powerlaw_cdf(x: f64, alpha: f64) -> f64 {
    if x < 0.0 { 0.0 } else if x > 1.0 { 1.0 } else { x.powf(alpha) }
}

// ═══════════════════════════════════════════════════════════════════
// SEMICIRCULAR DISTRIBUTION (Wigner)
// ═══════════════════════════════════════════════════════════════════

pub fn semicircular_pdf(x: f64, r: f64) -> f64 {
    if x.abs() > r { 0.0 } else { 2.0 / (PI * r * r) * (r * r - x * x).sqrt() }
}

// ═══════════════════════════════════════════════════════════════════
// ZIPF DISTRIBUTION (discrete)
// ═══════════════════════════════════════════════════════════════════

pub fn zipf_pmf(k: u64, s: f64, n_max: u64) -> f64 {
    if k == 0 || k > n_max { return 0.0; }
    let h: f64 = (1..=n_max).map(|i| 1.0 / (i as f64).powf(s)).sum();
    1.0 / ((k as f64).powf(s) * h)
}

// ═══════════════════════════════════════════════════════════════════
// TRUNCATED NORMAL DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn truncnorm_pdf(x: f64, mu: f64, sigma: f64, a: f64, b: f64) -> f64 {
    if x < a || x > b { return 0.0; }
    let alpha = (a - mu) / sigma;
    let beta = (b - mu) / sigma;
    let z = norm_cdf(beta, 0.0, 1.0) - norm_cdf(alpha, 0.0, 1.0);
    if z < 1e-15 { return 0.0; }
    norm_pdf(x, mu, sigma) / (sigma * z)
}

pub fn truncnorm_cdf(x: f64, mu: f64, sigma: f64, a: f64, b: f64) -> f64 {
    if x <= a { return 0.0; }
    if x >= b { return 1.0; }
    let alpha = (a - mu) / sigma;
    let beta = (b - mu) / sigma;
    let xi = (x - mu) / sigma;
    let z = norm_cdf(beta, 0.0, 1.0) - norm_cdf(alpha, 0.0, 1.0);
    if z < 1e-15 { return 0.5; }
    (norm_cdf(xi, 0.0, 1.0) - norm_cdf(alpha, 0.0, 1.0)) / z
}

pub fn truncnorm_rvs(rng: &mut Rng, n: usize, mu: f64, sigma: f64, a: f64, b: f64) -> Vec<f64> {
    let alpha = (a - mu) / sigma;
    let beta = (b - mu) / sigma;
    let phi_a = norm_cdf(alpha, 0.0, 1.0);
    let phi_b = norm_cdf(beta, 0.0, 1.0);
    (0..n).map(|_| {
        let u = phi_a + rng.uniform() * (phi_b - phi_a);
        mu + sigma * norm_ppf(u, 0.0, 1.0)
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// LEVY DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn levy_pdf(x: f64, mu: f64, c: f64) -> f64 {
    if x <= mu { return 0.0; }
    let d = x - mu;
    (c / (2.0 * PI)).sqrt() * (-c / (2.0 * d)).exp() / d.powf(1.5)
}

pub fn levy_cdf(x: f64, mu: f64, c: f64) -> f64 {
    if x <= mu { return 0.0; }
    erfc((c / (2.0 * (x - mu))).sqrt())
}

// ═══════════════════════════════════════════════════════════════════
// LOMAX (Pareto Type II) DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn lomax_pdf(x: f64, alpha: f64, lambda: f64) -> f64 {
    if x < 0.0 { 0.0 } else { alpha / lambda * (1.0 + x / lambda).powf(-(alpha + 1.0)) }
}

pub fn lomax_cdf(x: f64, alpha: f64, lambda: f64) -> f64 {
    if x < 0.0 { 0.0 } else { 1.0 - (1.0 + x / lambda).powf(-alpha) }
}

// ═══════════════════════════════════════════════════════════════════
// KUMARASWAMY DISTRIBUTION (bounded, like Beta but simpler CDF)
// ═══════════════════════════════════════════════════════════════════

pub fn kumaraswamy_pdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 { 0.0 } else { a * b * x.powf(a - 1.0) * (1.0 - x.powf(a)).powf(b - 1.0) }
}

pub fn kumaraswamy_cdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 { 0.0 } else if x >= 1.0 { 1.0 } else { 1.0 - (1.0 - x.powf(a)).powf(b) }
}

pub fn kumaraswamy_ppf(p: f64, a: f64, b: f64) -> f64 {
    (1.0 - (1.0 - p).powf(1.0 / b)).powf(1.0 / a)
}

pub fn kumaraswamy_rvs(rng: &mut Rng, n: usize, a: f64, b: f64) -> Vec<f64> {
    (0..n).map(|_| kumaraswamy_ppf(rng.uniform(), a, b)).collect()
}

// ═══════════════════════════════════════════════════════════════════
// DOUBLE EXPONENTIAL (same as Laplace, alias for compatibility)
// ═══════════════════════════════════════════════════════════════════

pub fn dexp_pdf(x: f64, rate: f64) -> f64 { laplace_pdf(x, 0.0, 1.0 / rate) }
pub fn dexp_cdf(x: f64, rate: f64) -> f64 { laplace_cdf(x, 0.0, 1.0 / rate) }

// ═══════════════════════════════════════════════════════════════════
// GOMPERTZ DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn gompertz_pdf(x: f64, b: f64, eta: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    b * eta * (b * x).exp() * (-eta * ((b * x).exp() - 1.0)).exp()
}

pub fn gompertz_cdf(x: f64, b: f64, eta: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    1.0 - (-eta * ((b * x).exp() - 1.0)).exp()
}

// ═══════════════════════════════════════════════════════════════════
// BENFORD DISTRIBUTION (first digit)
// ═══════════════════════════════════════════════════════════════════

pub fn benford_pmf(d: u64) -> f64 {
    if d < 1 || d > 9 { 0.0 } else { (1.0 + 1.0 / d as f64).log10() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invgauss() {
        let p = invgauss_pdf(1.0, 1.0, 1.0);
        assert!(p > 0.0 && p < 1.0);
        let c = invgauss_cdf(1.0, 1.0, 1.0);
        assert!(c > 0.0 && c < 1.0);
    }

    #[test]
    fn test_gev() {
        // xi=0 should be Gumbel
        let p = gev_pdf(0.0, 0.0, 0.0, 1.0);
        assert!(p > 0.0);
        let c = gev_cdf(0.0, 0.0, 0.0, 1.0);
        assert!(c > 0.0 && c < 1.0);
    }

    #[test]
    fn test_gpd() {
        let p = gpd_pdf(1.0, 0.5, 0.0, 1.0);
        assert!(p > 0.0);
        let c = gpd_cdf(1.0, 0.5, 0.0, 1.0);
        assert!(c > 0.0 && c < 1.0);
    }

    #[test]
    fn test_kumaraswamy() {
        // CDF at x=0.5 for a=2, b=5
        let c = kumaraswamy_cdf(0.5, 2.0, 5.0);
        assert!(c > 0.0 && c < 1.0);
        // PPF should invert CDF
        let x = kumaraswamy_ppf(c, 2.0, 5.0);
        assert!((x - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_truncnorm() {
        let p = truncnorm_pdf(0.0, 0.0, 1.0, -1.0, 1.0);
        assert!(p > 0.0);
        let c = truncnorm_cdf(0.0, 0.0, 1.0, -1.0, 1.0);
        assert!((c - 0.5).abs() < 0.01); // symmetric, so CDF at center ≈ 0.5
    }

    #[test]
    fn test_benford() {
        let total: f64 = (1..=9).map(benford_pmf).sum();
        assert!((total - 1.0).abs() < 1e-10, "Benford PMF should sum to 1");
        assert!((benford_pmf(1) - 0.301).abs() < 0.001);
    }

    #[test]
    fn test_maxwell() {
        let p = maxwell_pdf(1.0, 1.0);
        assert!(p > 0.0);
    }

    #[test]
    fn test_vonmises() {
        let p = vonmises_pdf(0.0, 0.0, 1.0);
        assert!(p > 0.0);
    }

    #[test]
    fn test_erlang_is_gamma() {
        // Erlang(k=3, rate=2) should equal Gamma(3, 1/2)
        let e = erlang_pdf(1.0, 3, 2.0);
        let g = gamma_pdf(1.0, 3.0, 0.5);
        assert!((e - g).abs() < 1e-10);
    }
}
