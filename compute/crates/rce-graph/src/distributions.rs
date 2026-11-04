//! Probability distributions: PDF, CDF, quantile (PPF), random sampling.
//!
//! Pure Rust. No external dependencies. Covers the core distributions
//! from R (dnorm/pnorm/qnorm/rnorm), SciPy (scipy.stats), and Julia (Distributions.jl).
//!
//! Each distribution provides:
//! - `pdf(x, ...)` — probability density function
//! - `cdf(x, ...)` — cumulative distribution function
//! - `ppf(p, ...)` — percent point function (quantile, inverse CDF)
//! - `rvs(n, ...)` — random variates (requires PRNG state)

use std::f64::consts::{PI, SQRT_2};

// ═══════════════════════════════════════════════════════════════════
// PRNG — xoshiro256++ (fast, high quality, reproducible)
// ═══════════════════════════════════════════════════════════════════

pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // SplitMix64 to initialize state from single seed
        let mut z = seed;
        let mut s = [0u64; 4];
        for i in 0..4 {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            s[i] = z ^ (z >> 31);
        }
        Self { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform [0, 1)
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Gamma(alpha, 1) via Marsaglia-Tsang
    pub fn gamma(&mut self, alpha: f64) -> f64 {
        if alpha < 1.0 {
            return self.gamma(alpha + 1.0) * self.uniform().powf(1.0 / alpha);
        }
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v > 0.0 {
                let u = self.uniform();
                if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                    return d * v;
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// HELPER: error function and gamma function
// ═══════════════════════════════════════════════════════════════════

/// Error function (Abramowitz & Stegun approximation, max error 1.5e-7)
pub fn erf(x: f64) -> f64 {
    if x == 0.0 { return 0.0; }
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = 1.0 - poly * (-a * a).exp();
    if x > 0.0 { result } else { -result }
}

/// Complementary error function
pub fn erfc(x: f64) -> f64 { 1.0 - erf(x) }

/// Inverse error function (rational approximation)
pub fn erfinv(p: f64) -> f64 {
    if p <= -1.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    if p == 0.0 { return 0.0; }
    let a = p.abs();
    let t = if a <= 0.7 {
        let x = a * a;
        let r = a * ((((-0.140543331 * x + 0.914624893) * x - 1.645349621) * x + 0.886226899) /
            ((((0.012229801 * x - 0.329097515) * x + 1.442710462) * x - 2.118377725) * x + 1.0));
        r
    } else {
        let x = (-((1.0 - a) / 2.0).ln()).sqrt();
        let r = (((1.641345311 * x + 3.429567803) * x - 1.624906493) * x - 1.970840454) /
            ((1.637067800 * x + 3.543889200) * x + 1.0);
        r
    };
    if p < 0.0 { -t } else { t }
}

/// Log-gamma function (Stirling's approximation for large x, Lanczos for small)
pub fn lgamma(x: f64) -> f64 {
    if x <= 0.0 { return f64::INFINITY; }
    if x < 0.5 {
        return (PI / (PI * x).sin()).ln() - lgamma(1.0 - x);
    }
    // Lanczos approximation (g=7)
    let coeffs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let z = x - 1.0;
    let mut sum = coeffs[0];
    for i in 1..9 {
        sum += coeffs[i] / (z + i as f64);
    }
    let t = z + 7.5;
    0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + sum.ln()
}

/// Gamma function
pub fn gamma_fn(x: f64) -> f64 { lgamma(x).exp() }

/// Beta function B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b)
pub fn beta_fn(a: f64, b: f64) -> f64 {
    (lgamma(a) + lgamma(b) - lgamma(a + b)).exp()
}

/// Regularized incomplete beta function I_x(a,b) via continued fraction
pub fn betainc(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - betainc(1.0 - x, b, a);
    }
    let lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - lbeta).exp() / a;
    // Lentz's continued fraction
    let mut f = 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < 1e-30 { d = 1e-30; }
    d = 1.0 / d;
    f = d;
    for m in 1..200 {
        let m = m as f64;
        // Even step
        let num = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        d = 1.0 + num * d; if d.abs() < 1e-30 { d = 1e-30; } d = 1.0 / d;
        c = 1.0 + num / c; if c.abs() < 1e-30 { c = 1e-30; }
        f *= c * d;
        // Odd step
        let num = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        d = 1.0 + num * d; if d.abs() < 1e-30 { d = 1e-30; } d = 1.0 / d;
        c = 1.0 + num / c; if c.abs() < 1e-30 { c = 1e-30; }
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-12 { break; }
    }
    front * f
}

/// Regularized incomplete gamma function P(a,x)
pub fn gammainc(a: f64, x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x > a + 1.0 {
        return 1.0 - gammainc_upper(a, x);
    }
    // Series expansion
    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;
    for n in 1..200 {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < 1e-14 * sum.abs() { break; }
    }
    sum * (-x + a * x.ln() - lgamma(a)).exp()
}

fn gammainc_upper(a: f64, x: f64) -> f64 {
    // Continued fraction for upper incomplete gamma
    let mut f = 1.0;
    let mut c = 1.0;
    let mut d = x + 1.0 - a;
    if d.abs() < 1e-30 { d = 1e-30; }
    d = 1.0 / d;
    f = d;
    for i in 1..200 {
        let an = -(i as f64) * (i as f64 - a);
        let bn = x + 2.0 * i as f64 + 1.0 - a;
        d = bn + an * d; if d.abs() < 1e-30 { d = 1e-30; } d = 1.0 / d;
        c = bn + an / c; if c.abs() < 1e-30 { c = 1e-30; }
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-12 { break; }
    }
    f * (-x + a * x.ln() - lgamma(a)).exp()
}

// ═══════════════════════════════════════════════════════════════════
// NORMAL DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn norm_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    (-0.5 * z * z).exp() / (sigma * (2.0 * PI).sqrt())
}

pub fn norm_cdf(x: f64, mu: f64, sigma: f64) -> f64 {
    0.5 * erfc(-(x - mu) / (sigma * SQRT_2))
}

pub fn norm_ppf(p: f64, mu: f64, sigma: f64) -> f64 {
    mu + sigma * SQRT_2 * erfinv(2.0 * p - 1.0)
}

pub fn norm_rvs(rng: &mut Rng, n: usize, mu: f64, sigma: f64) -> Vec<f64> {
    (0..n).map(|_| mu + sigma * rng.normal()).collect()
}

// ═══════════════════════════════════════════════════════════════════
// UNIFORM DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn uniform_pdf(x: f64, a: f64, b: f64) -> f64 {
    if x >= a && x <= b { 1.0 / (b - a) } else { 0.0 }
}

pub fn uniform_cdf(x: f64, a: f64, b: f64) -> f64 {
    if x < a { 0.0 } else if x > b { 1.0 } else { (x - a) / (b - a) }
}

pub fn uniform_ppf(p: f64, a: f64, b: f64) -> f64 { a + p * (b - a) }

pub fn uniform_rvs(rng: &mut Rng, n: usize, a: f64, b: f64) -> Vec<f64> {
    (0..n).map(|_| a + (b - a) * rng.uniform()).collect()
}

// ═══════════════════════════════════════════════════════════════════
// EXPONENTIAL DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn expon_pdf(x: f64, rate: f64) -> f64 {
    if x < 0.0 { 0.0 } else { rate * (-rate * x).exp() }
}

pub fn expon_cdf(x: f64, rate: f64) -> f64 {
    if x < 0.0 { 0.0 } else { 1.0 - (-rate * x).exp() }
}

pub fn expon_ppf(p: f64, rate: f64) -> f64 { -(1.0 - p).ln() / rate }

pub fn expon_rvs(rng: &mut Rng, n: usize, rate: f64) -> Vec<f64> {
    (0..n).map(|_| -rng.uniform().max(1e-300).ln() / rate).collect()
}

// ═══════════════════════════════════════════════════════════════════
// GAMMA DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn gamma_pdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    ((alpha - 1.0) * x.ln() - x / beta - alpha * beta.ln() - lgamma(alpha)).exp()
}

pub fn gamma_cdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if x <= 0.0 { 0.0 } else { gammainc(alpha, x / beta) }
}

pub fn gamma_ppf(p: f64, alpha: f64, beta: f64) -> f64 {
    // Newton's method on CDF
    let mut x = alpha * beta; // initial guess at mean
    for _ in 0..50 {
        let f = gamma_cdf(x, alpha, beta) - p;
        let fprime = gamma_pdf(x, alpha, beta);
        if fprime.abs() < 1e-15 { break; }
        x -= f / fprime;
        x = x.max(1e-300);
    }
    x
}

pub fn gamma_rvs(rng: &mut Rng, n: usize, alpha: f64, beta: f64) -> Vec<f64> {
    (0..n).map(|_| rng.gamma(alpha) * beta).collect()
}

// ═══════════════════════════════════════════════════════════════════
// BETA DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn beta_pdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 { return 0.0; }
    ((a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - lgamma(a) - lgamma(b) + lgamma(a + b)).exp()
}

pub fn beta_cdf(x: f64, a: f64, b: f64) -> f64 { betainc(x, a, b) }

pub fn beta_ppf(p: f64, a: f64, b: f64) -> f64 {
    let mut x = 0.5;
    for _ in 0..50 {
        let f = beta_cdf(x, a, b) - p;
        let fprime = beta_pdf(x, a, b);
        if fprime.abs() < 1e-15 { break; }
        x -= f / fprime;
        x = x.clamp(1e-15, 1.0 - 1e-15);
    }
    x
}

pub fn beta_rvs(rng: &mut Rng, n: usize, a: f64, b: f64) -> Vec<f64> {
    (0..n).map(|_| {
        let x = rng.gamma(a);
        let y = rng.gamma(b);
        x / (x + y)
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// CHI-SQUARED DISTRIBUTION (special case of Gamma)
// ═══════════════════════════════════════════════════════════════════

pub fn chi2_pdf(x: f64, df: f64) -> f64 { gamma_pdf(x, df / 2.0, 2.0) }
pub fn chi2_cdf(x: f64, df: f64) -> f64 { gamma_cdf(x, df / 2.0, 2.0) }
pub fn chi2_ppf(p: f64, df: f64) -> f64 { gamma_ppf(p, df / 2.0, 2.0) }
pub fn chi2_rvs(rng: &mut Rng, n: usize, df: f64) -> Vec<f64> { gamma_rvs(rng, n, df / 2.0, 2.0) }

// ═══════════════════════════════════════════════════════════════════
// STUDENT'S T DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn t_pdf(x: f64, df: f64) -> f64 {
    let coeff = gamma_fn((df + 1.0) / 2.0) / ((df * PI).sqrt() * gamma_fn(df / 2.0));
    coeff * (1.0 + x * x / df).powf(-(df + 1.0) / 2.0)
}

pub fn t_cdf(x: f64, df: f64) -> f64 {
    let xt = df / (df + x * x);
    if x >= 0.0 {
        1.0 - 0.5 * betainc(xt, df / 2.0, 0.5)
    } else {
        0.5 * betainc(xt, df / 2.0, 0.5)
    }
}

pub fn t_ppf(p: f64, df: f64) -> f64 {
    // Newton's method
    let mut x = norm_ppf(p, 0.0, 1.0); // initial guess from normal
    for _ in 0..50 {
        let f = t_cdf(x, df) - p;
        let fprime = t_pdf(x, df);
        if fprime.abs() < 1e-15 { break; }
        x -= f / fprime;
    }
    x
}

pub fn t_rvs(rng: &mut Rng, n: usize, df: f64) -> Vec<f64> {
    (0..n).map(|_| {
        let z = rng.normal();
        let v = rng.gamma(df / 2.0) * 2.0 / df;
        z / v.sqrt()
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// F DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn f_pdf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    let num = ((d1 * x).powf(d1) * d2.powf(d2) / (d1 * x + d2).powf(d1 + d2)).sqrt();
    num / (x * beta_fn(d1 / 2.0, d2 / 2.0))
}

pub fn f_cdf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    betainc(d1 * x / (d1 * x + d2), d1 / 2.0, d2 / 2.0)
}

pub fn f_ppf(p: f64, d1: f64, d2: f64) -> f64 {
    let mut x = 1.0;
    for _ in 0..50 {
        let f = f_cdf(x, d1, d2) - p;
        let fprime = f_pdf(x, d1, d2);
        if fprime.abs() < 1e-15 { break; }
        x -= f / fprime;
        x = x.max(1e-300);
    }
    x
}

pub fn f_rvs(rng: &mut Rng, n: usize, d1: f64, d2: f64) -> Vec<f64> {
    (0..n).map(|_| {
        let x1 = rng.gamma(d1 / 2.0) * 2.0 / d1;
        let x2 = rng.gamma(d2 / 2.0) * 2.0 / d2;
        x1 / x2
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// BINOMIAL DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn binom_pmf(k: u64, n: u64, p: f64) -> f64 {
    if k > n { return 0.0; }
    let log_coeff = lgamma(n as f64 + 1.0) - lgamma(k as f64 + 1.0) - lgamma((n - k) as f64 + 1.0);
    (log_coeff + k as f64 * p.ln() + (n - k) as f64 * (1.0 - p).ln()).exp()
}

pub fn binom_cdf(k: u64, n: u64, p: f64) -> f64 {
    (0..=k).map(|i| binom_pmf(i, n, p)).sum()
}

pub fn binom_rvs(rng: &mut Rng, count: usize, n: u64, p: f64) -> Vec<f64> {
    (0..count).map(|_| {
        (0..n).filter(|_| rng.uniform() < p).count() as f64
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// POISSON DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn poisson_pmf(k: u64, lambda: f64) -> f64 {
    (k as f64 * lambda.ln() - lambda - lgamma(k as f64 + 1.0)).exp()
}

pub fn poisson_cdf(k: u64, lambda: f64) -> f64 {
    (0..=k).map(|i| poisson_pmf(i, lambda)).sum()
}

pub fn poisson_rvs(rng: &mut Rng, n: usize, lambda: f64) -> Vec<f64> {
    // Knuth algorithm for small lambda
    (0..n).map(|_| {
        let l = (-lambda).exp();
        let mut k = 0u64;
        let mut p = 1.0;
        loop {
            k += 1;
            p *= rng.uniform();
            if p <= l { break; }
        }
        (k - 1) as f64
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// LOGNORMAL DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn lognorm_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    let z = (x.ln() - mu) / sigma;
    (-0.5 * z * z).exp() / (x * sigma * (2.0 * PI).sqrt())
}

pub fn lognorm_cdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if x <= 0.0 { 0.0 } else { norm_cdf(x.ln(), mu, sigma) }
}

pub fn lognorm_ppf(p: f64, mu: f64, sigma: f64) -> f64 {
    norm_ppf(p, mu, sigma).exp()
}

pub fn lognorm_rvs(rng: &mut Rng, n: usize, mu: f64, sigma: f64) -> Vec<f64> {
    (0..n).map(|_| (mu + sigma * rng.normal()).exp()).collect()
}

// ═══════════════════════════════════════════════════════════════════
// WEIBULL DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn weibull_pdf(x: f64, k: f64, lambda: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    (k / lambda) * (x / lambda).powf(k - 1.0) * (-(x / lambda).powf(k)).exp()
}

pub fn weibull_cdf(x: f64, k: f64, lambda: f64) -> f64 {
    if x < 0.0 { 0.0 } else { 1.0 - (-(x / lambda).powf(k)).exp() }
}

pub fn weibull_ppf(p: f64, k: f64, lambda: f64) -> f64 {
    lambda * (-(1.0 - p).ln()).powf(1.0 / k)
}

pub fn weibull_rvs(rng: &mut Rng, n: usize, k: f64, lambda: f64) -> Vec<f64> {
    (0..n).map(|_| weibull_ppf(rng.uniform(), k, lambda)).collect()
}

// ═══════════════════════════════════════════════════════════════════
// CAUCHY DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn cauchy_pdf(x: f64, x0: f64, gamma: f64) -> f64 {
    1.0 / (PI * gamma * (1.0 + ((x - x0) / gamma).powi(2)))
}

pub fn cauchy_cdf(x: f64, x0: f64, gamma: f64) -> f64 {
    0.5 + ((x - x0) / gamma).atan() / PI
}

pub fn cauchy_ppf(p: f64, x0: f64, gamma: f64) -> f64 {
    x0 + gamma * (PI * (p - 0.5)).tan()
}

pub fn cauchy_rvs(rng: &mut Rng, n: usize, x0: f64, gamma: f64) -> Vec<f64> {
    (0..n).map(|_| cauchy_ppf(rng.uniform(), x0, gamma)).collect()
}

// ═══════════════════════════════════════════════════════════════════
// LAPLACE DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn laplace_pdf(x: f64, mu: f64, b: f64) -> f64 {
    (-(x - mu).abs() / b).exp() / (2.0 * b)
}

pub fn laplace_cdf(x: f64, mu: f64, b: f64) -> f64 {
    if x < mu { 0.5 * ((x - mu) / b).exp() }
    else { 1.0 - 0.5 * (-(x - mu) / b).exp() }
}

pub fn laplace_ppf(p: f64, mu: f64, b: f64) -> f64 {
    if p < 0.5 { mu + b * (2.0 * p).ln() }
    else { mu - b * (2.0 * (1.0 - p)).ln() }
}

pub fn laplace_rvs(rng: &mut Rng, n: usize, mu: f64, b: f64) -> Vec<f64> {
    (0..n).map(|_| laplace_ppf(rng.uniform(), mu, b)).collect()
}

// ═══════════════════════════════════════════════════════════════════
// LOGISTIC DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn logistic_pdf(x: f64, mu: f64, s: f64) -> f64 {
    let z = (-(x - mu) / s).exp();
    z / (s * (1.0 + z).powi(2))
}

pub fn logistic_cdf(x: f64, mu: f64, s: f64) -> f64 {
    1.0 / (1.0 + (-(x - mu) / s).exp())
}

pub fn logistic_ppf(p: f64, mu: f64, s: f64) -> f64 {
    mu + s * (p / (1.0 - p)).ln()
}

pub fn logistic_rvs(rng: &mut Rng, n: usize, mu: f64, s: f64) -> Vec<f64> {
    (0..n).map(|_| logistic_ppf(rng.uniform(), mu, s)).collect()
}

// ═══════════════════════════════════════════════════════════════════
// PARETO DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn pareto_pdf(x: f64, alpha: f64, xm: f64) -> f64 {
    if x < xm { 0.0 } else { alpha * xm.powf(alpha) / x.powf(alpha + 1.0) }
}

pub fn pareto_cdf(x: f64, alpha: f64, xm: f64) -> f64 {
    if x < xm { 0.0 } else { 1.0 - (xm / x).powf(alpha) }
}

pub fn pareto_ppf(p: f64, alpha: f64, xm: f64) -> f64 {
    xm / (1.0 - p).powf(1.0 / alpha)
}

pub fn pareto_rvs(rng: &mut Rng, n: usize, alpha: f64, xm: f64) -> Vec<f64> {
    (0..n).map(|_| pareto_ppf(rng.uniform(), alpha, xm)).collect()
}

// ═══════════════════════════════════════════════════════════════════
// GEOMETRIC DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn geom_pmf(k: u64, p: f64) -> f64 {
    if k == 0 { return 0.0; }
    p * (1.0 - p).powi((k - 1) as i32)
}

pub fn geom_cdf(k: u64, p: f64) -> f64 {
    1.0 - (1.0 - p).powi(k as i32)
}

pub fn geom_rvs(rng: &mut Rng, n: usize, p: f64) -> Vec<f64> {
    (0..n).map(|_| (rng.uniform().max(1e-300).ln() / (1.0 - p).ln()).ceil()).collect()
}

// ═══════════════════════════════════════════════════════════════════
// NEGATIVE BINOMIAL DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn nbinom_pmf(k: u64, r: f64, p: f64) -> f64 {
    let log_coeff = lgamma(k as f64 + r) - lgamma(k as f64 + 1.0) - lgamma(r);
    (log_coeff + r * p.ln() + k as f64 * (1.0 - p).ln()).exp()
}

pub fn nbinom_cdf(k: u64, r: f64, p: f64) -> f64 {
    (0..=k).map(|i| nbinom_pmf(i, r, p)).sum()
}

pub fn nbinom_rvs(rng: &mut Rng, n: usize, r: f64, p: f64) -> Vec<f64> {
    (0..n).map(|_| {
        let lambda = rng.gamma(r) * (1.0 - p) / p;
        // Poisson with rate lambda
        let l = (-lambda).exp();
        let mut k = 0u64;
        let mut pr = 1.0;
        loop { k += 1; pr *= rng.uniform(); if pr <= l { break; } }
        (k - 1) as f64
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// HYPERGEOMETRIC DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn hypergeom_pmf(k: u64, big_n: u64, big_k: u64, n: u64) -> f64 {
    if k > big_k || k > n || n - k > big_n - big_k { return 0.0; }
    let log_p = lgamma(big_k as f64 + 1.0) - lgamma(k as f64 + 1.0) - lgamma((big_k - k) as f64 + 1.0)
        + lgamma((big_n - big_k) as f64 + 1.0) - lgamma((n - k) as f64 + 1.0) - lgamma((big_n - big_k - n + k) as f64 + 1.0)
        - lgamma(big_n as f64 + 1.0) + lgamma(n as f64 + 1.0) + lgamma((big_n - n) as f64 + 1.0);
    log_p.exp()
}

pub fn hypergeom_cdf(k: u64, big_n: u64, big_k: u64, n: u64) -> f64 {
    (0..=k).map(|i| hypergeom_pmf(i, big_n, big_k, n)).sum()
}

// ═══════════════════════════════════════════════════════════════════
// RAYLEIGH DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn rayleigh_pdf(x: f64, sigma: f64) -> f64 {
    if x < 0.0 { 0.0 } else { x / (sigma * sigma) * (-x * x / (2.0 * sigma * sigma)).exp() }
}

pub fn rayleigh_cdf(x: f64, sigma: f64) -> f64 {
    if x < 0.0 { 0.0 } else { 1.0 - (-x * x / (2.0 * sigma * sigma)).exp() }
}

pub fn rayleigh_ppf(p: f64, sigma: f64) -> f64 {
    sigma * (-2.0 * (1.0 - p).ln()).sqrt()
}

pub fn rayleigh_rvs(rng: &mut Rng, n: usize, sigma: f64) -> Vec<f64> {
    (0..n).map(|_| rayleigh_ppf(rng.uniform(), sigma)).collect()
}

// ═══════════════════════════════════════════════════════════════════
// GUMBEL DISTRIBUTION (extreme value type I)
// ═══════════════════════════════════════════════════════════════════

pub fn gumbel_pdf(x: f64, mu: f64, beta: f64) -> f64 {
    let z = (x - mu) / beta;
    (-(z + (-z).exp())).exp() / beta
}

pub fn gumbel_cdf(x: f64, mu: f64, beta: f64) -> f64 {
    let z = (x - mu) / beta;
    (-(-z).exp()).exp()
}

pub fn gumbel_ppf(p: f64, mu: f64, beta: f64) -> f64 {
    mu - beta * (-p.ln()).ln()
}

pub fn gumbel_rvs(rng: &mut Rng, n: usize, mu: f64, beta: f64) -> Vec<f64> {
    (0..n).map(|_| gumbel_ppf(rng.uniform(), mu, beta)).collect()
}

// ═══════════════════════════════════════════════════════════════════
// TRIANGULAR DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

pub fn triangular_pdf(x: f64, a: f64, b: f64, c: f64) -> f64 {
    if x < a || x > b { 0.0 }
    else if x < c { 2.0 * (x - a) / ((b - a) * (c - a)) }
    else if x == c { 2.0 / (b - a) }
    else { 2.0 * (b - x) / ((b - a) * (b - c)) }
}

pub fn triangular_cdf(x: f64, a: f64, b: f64, c: f64) -> f64 {
    if x <= a { 0.0 }
    else if x <= c { (x - a).powi(2) / ((b - a) * (c - a)) }
    else if x < b { 1.0 - (b - x).powi(2) / ((b - a) * (b - c)) }
    else { 1.0 }
}

pub fn triangular_ppf(p: f64, a: f64, b: f64, c: f64) -> f64 {
    let fc = (c - a) / (b - a);
    if p < fc { a + ((b - a) * (c - a) * p).sqrt() }
    else { b - ((b - a) * (b - c) * (1.0 - p)).sqrt() }
}

pub fn triangular_rvs(rng: &mut Rng, n: usize, a: f64, b: f64, c: f64) -> Vec<f64> {
    (0..n).map(|_| triangular_ppf(rng.uniform(), a, b, c)).collect()
}

// ═══════════════════════════════════════════════════════════════════
// DIRICHLET DISTRIBUTION (multivariate)
// ═══════════════════════════════════════════════════════════════════

pub fn dirichlet_rvs(rng: &mut Rng, alpha: &[f64]) -> Vec<f64> {
    let samples: Vec<f64> = alpha.iter().map(|&a| rng.gamma(a)).collect();
    let total: f64 = samples.iter().sum();
    if total < 1e-300 { return vec![1.0 / alpha.len() as f64; alpha.len()]; }
    samples.iter().map(|&s| s / total).collect()
}

// ═══════════════════════════════════════════════════════════════════
// MULTIVARIATE NORMAL DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════

/// Sample from multivariate normal: N(mean, cov).
/// Uses Cholesky decomposition: X = mean + L * Z where L * L' = cov.
pub fn mvnorm_rvs(rng: &mut Rng, mean: &[f64], cov: &[Vec<f64>]) -> Vec<f64> {
    let dim = mean.len();
    // Cholesky of covariance
    let l = match crate::math_runtime::cholesky(cov) {
        Some(l) => l,
        None => return mean.to_vec(), // fallback if not PD
    };
    let z: Vec<f64> = (0..dim).map(|_| rng.normal()).collect();
    (0..dim).map(|i| {
        mean[i] + (0..=i).map(|j| l[i][j] * z[j]).sum::<f64>()
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::E;

    #[test]
    fn test_erf() {
        assert!((erf(0.0)).abs() < 1e-10);
        assert!((erf(1.0) - 0.8427).abs() < 0.001);
        assert!((erf(-1.0) + 0.8427).abs() < 0.001);
    }

    #[test]
    fn test_lgamma() {
        assert!((lgamma(1.0)).abs() < 1e-10); // 0! = 1, ln(1) = 0
        assert!((lgamma(5.0) - (24.0f64).ln()).abs() < 1e-10); // 4! = 24
    }

    #[test]
    fn test_norm_cdf_ppf() {
        assert!((norm_cdf(0.0, 0.0, 1.0) - 0.5).abs() < 1e-10);
        assert!((norm_cdf(1.96, 0.0, 1.0) - 0.975).abs() < 0.001);
        assert!((norm_ppf(0.975, 0.0, 1.0) - 1.96).abs() < 0.01);
    }

    #[test]
    fn test_norm_pdf_integrates() {
        // Numerical integration of PDF should be ~1
        let n = 10000;
        let dx = 10.0 / n as f64;
        let integral: f64 = (-n/2..n/2).map(|i| {
            let x = i as f64 * dx;
            norm_pdf(x, 0.0, 1.0) * dx
        }).sum();
        assert!((integral - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_t_cdf_symmetry() {
        assert!((t_cdf(0.0, 10.0) - 0.5).abs() < 1e-10);
        let p = t_cdf(2.228, 10.0);
        assert!((p - 0.975).abs() < 0.005);
    }

    #[test]
    fn test_chi2_cdf() {
        // chi2(df=1) at x=3.841 should be ~0.95
        assert!((chi2_cdf(3.841, 1.0) - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_f_cdf() {
        // F(1, 1) at x=1 should be ~0.5
        assert!((f_cdf(1.0, 1.0, 1.0) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_beta_cdf() {
        assert!((beta_cdf(0.5, 1.0, 1.0) - 0.5).abs() < 1e-10); // Uniform
        assert!((beta_cdf(0.5, 2.0, 2.0) - 0.5).abs() < 0.01); // Symmetric
    }

    #[test]
    fn test_binom_pmf() {
        // C(10,5) * 0.5^10 = 252/1024
        assert!((binom_pmf(5, 10, 0.5) - 252.0 / 1024.0).abs() < 1e-10);
    }

    #[test]
    fn test_poisson_pmf() {
        // P(k=0 | lambda=1) = e^-1
        assert!((poisson_pmf(0, 1.0) - (-1.0f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_exponential() {
        assert!((expon_cdf(1.0, 1.0) - (1.0 - E.powf(-1.0))).abs() < 1e-10);
        assert!((expon_ppf(0.5, 1.0) - 2.0f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_weibull() {
        // Weibull(k=1, lambda=1) = Exponential(1)
        assert!((weibull_cdf(1.0, 1.0, 1.0) - expon_cdf(1.0, 1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cauchy() {
        assert!((cauchy_cdf(0.0, 0.0, 1.0) - 0.5).abs() < 1e-10);
        assert!((cauchy_ppf(0.75, 0.0, 1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rng_uniform_range() {
        let mut rng = Rng::new(42);
        for _ in 0..1000 {
            let u = rng.uniform();
            assert!(u >= 0.0 && u < 1.0);
        }
    }

    #[test]
    fn test_rng_normal_mean() {
        let mut rng = Rng::new(42);
        let samples: Vec<f64> = (0..10000).map(|_| rng.normal()).collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.05, "normal mean should be ~0, got {mean}");
    }

    #[test]
    fn test_gamma_rvs_positive() {
        let mut rng = Rng::new(42);
        let samples = gamma_rvs(&mut rng, 100, 2.0, 1.0);
        assert!(samples.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_beta_rvs_unit_interval() {
        let mut rng = Rng::new(42);
        let samples = beta_rvs(&mut rng, 100, 2.0, 5.0);
        assert!(samples.iter().all(|&x| x > 0.0 && x < 1.0));
    }

    #[test]
    fn test_betainc() {
        // I_0.5(1,1) = 0.5 (uniform)
        assert!((betainc(0.5, 1.0, 1.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gammainc() {
        // P(1, 1) = 1 - e^-1
        assert!((gammainc(1.0, 1.0) - (1.0 - E.powf(-1.0))).abs() < 1e-6);
    }
}
