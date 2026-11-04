//! Bayesian inference: conjugate priors, posterior updates, credible intervals,
//! Bayesian hypothesis testing, MCMC basics, predictive distributions.

// ═══════════════════════════════════════════════════════════════════
// CONJUGATE PRIOR UPDATES
// ═══════════════════════════════════════════════════════════════════

/// Beta-Binomial conjugate: update Beta(α, β) prior with k successes in n trials.
/// Returns posterior (α', β').
pub fn beta_binomial_update(alpha: f64, beta: f64, successes: usize, trials: usize) -> (f64, f64) {
    (alpha + successes as f64, beta + (trials - successes) as f64)
}

/// Beta posterior mean: E[θ] = α / (α + β).
pub fn beta_mean(alpha: f64, beta: f64) -> f64 { alpha / (alpha + beta) }

/// Beta posterior mode: (α - 1) / (α + β - 2) for α, β > 1.
pub fn beta_mode(alpha: f64, beta: f64) -> f64 {
    if alpha > 1.0 && beta > 1.0 { (alpha - 1.0) / (alpha + beta - 2.0) }
    else { beta_mean(alpha, beta) }
}

/// Beta posterior variance.
pub fn beta_variance(alpha: f64, beta: f64) -> f64 {
    alpha * beta / ((alpha + beta).powi(2) * (alpha + beta + 1.0))
}

/// Beta credible interval (equal-tailed, using normal approximation).
pub fn beta_credible_interval(alpha: f64, beta: f64, confidence: f64) -> (f64, f64) {
    let mean = beta_mean(alpha, beta);
    let std = beta_variance(alpha, beta).sqrt();
    let z = crate::sampling_design::z_score((1.0 + confidence) / 2.0);
    ((mean - z * std).max(0.0), (mean + z * std).min(1.0))
}

/// Normal-Normal conjugate: update N(μ₀, σ₀²) prior with data mean x̄, n observations, known σ².
/// Returns posterior (μ', σ'²).
pub fn normal_normal_update(mu0: f64, sigma0_sq: f64, x_bar: f64, n: usize, sigma_sq: f64) -> (f64, f64) {
    let precision0 = 1.0 / sigma0_sq;
    let precision_data = n as f64 / sigma_sq;
    let precision_post = precision0 + precision_data;
    let mu_post = (precision0 * mu0 + precision_data * x_bar) / precision_post;
    let sigma_post_sq = 1.0 / precision_post;
    (mu_post, sigma_post_sq)
}

/// Gamma-Poisson conjugate: update Gamma(α, β) with n observations summing to s.
pub fn gamma_poisson_update(alpha: f64, beta: f64, n: usize, sum: f64) -> (f64, f64) {
    (alpha + sum, beta + n as f64)
}

/// Gamma posterior mean: E[λ] = α / β.
pub fn gamma_mean(alpha: f64, beta: f64) -> f64 { alpha / beta }

/// Normal-Inverse-Gamma update for unknown mean and variance.
/// Prior: NIG(μ₀, κ₀, α₀, β₀), data: n observations with mean x̄ and sum of squares ss.
pub fn nig_update(mu0: f64, kappa0: f64, alpha0: f64, beta0: f64,
                  n: usize, x_bar: f64, sample_var: f64) -> (f64, f64, f64, f64) {
    let nf = n as f64;
    let kappa_n = kappa0 + nf;
    let mu_n = (kappa0 * mu0 + nf * x_bar) / kappa_n;
    let alpha_n = alpha0 + nf / 2.0;
    let beta_n = beta0 + 0.5 * nf * sample_var + kappa0 * nf * (x_bar - mu0).powi(2) / (2.0 * kappa_n);
    (mu_n, kappa_n, alpha_n, beta_n)
}

/// Dirichlet-Multinomial conjugate: update Dirichlet(α) with counts.
pub fn dirichlet_multinomial_update(alpha: &[f64], counts: &[usize]) -> Vec<f64> {
    alpha.iter().zip(counts).map(|(&a, &c)| a + c as f64).collect()
}

/// Dirichlet posterior mean: E[θ_i] = α_i / Σα.
pub fn dirichlet_mean(alpha: &[f64]) -> Vec<f64> {
    let sum: f64 = alpha.iter().sum();
    alpha.iter().map(|a| a / sum).collect()
}

// ═══════════════════════════════════════════════════════════════════
// BAYESIAN HYPOTHESIS TESTING
// ═══════════════════════════════════════════════════════════════════

/// Bayes factor for binomial test: H0: p = p0, H1: p ~ Beta(α, β).
/// Returns BF10 (evidence for H1 over H0).
pub fn bayes_factor_binomial(successes: usize, trials: usize, p0: f64, alpha: f64, beta: f64) -> f64 {
    let k = successes as f64;
    let n = trials as f64;
    // P(data | H0) = C(n,k) p0^k (1-p0)^{n-k}
    let log_p_h0 = k * p0.ln() + (n - k) * (1.0 - p0).ln();
    // P(data | H1) = B(k+α, n-k+β) / B(α, β)
    let log_p_h1 = lbeta(k + alpha, n - k + beta) - lbeta(alpha, beta);
    (log_p_h1 - log_p_h0).exp()
}

/// Interpret Bayes factor strength (Jeffreys' scale).
pub fn interpret_bayes_factor(bf: f64) -> &'static str {
    if bf > 100.0 { "Decisive" }
    else if bf > 30.0 { "Very strong" }
    else if bf > 10.0 { "Strong" }
    else if bf > 3.0 { "Substantial" }
    else if bf > 1.0 { "Barely worth mentioning" }
    else if bf > 1.0 / 3.0 { "Barely worth mentioning (against)" }
    else if bf > 1.0 / 10.0 { "Substantial (against)" }
    else if bf > 1.0 / 30.0 { "Strong (against)" }
    else { "Decisive (against)" }
}

/// Posterior probability of H1 given Bayes factor and prior odds.
pub fn posterior_probability(bf: f64, prior_prob_h1: f64) -> f64 {
    let prior_odds = prior_prob_h1 / (1.0 - prior_prob_h1);
    let posterior_odds = bf * prior_odds;
    posterior_odds / (1.0 + posterior_odds)
}

// ═══════════════════════════════════════════════════════════════════
// MCMC BASICS (Metropolis-Hastings)
// ═══════════════════════════════════════════════════════════════════

/// Metropolis-Hastings sampler for 1D distribution.
/// log_target: log of target density, proposal_std: std of Gaussian proposal.
/// Returns chain of samples.
pub fn metropolis_hastings(log_target: impl Fn(f64) -> f64, initial: f64, proposal_std: f64,
                           n_samples: usize, seed: u64) -> Vec<f64> {
    let mut state = seed_state(seed);
    let mut current = initial;
    let mut log_current = log_target(current);
    let mut chain = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let proposal = current + prng_normal(&mut state) * proposal_std;
        let log_proposal = log_target(proposal);
        let log_alpha = log_proposal - log_current;
        let u = prng_f64(&mut state);
        if u.ln() < log_alpha {
            current = proposal;
            log_current = log_proposal;
        }
        chain.push(current);
    }
    chain
}

/// Acceptance rate of MCMC chain.
pub fn mcmc_acceptance_rate(chain: &[f64]) -> f64 {
    if chain.len() < 2 { return 1.0; }
    let changes = chain.windows(2).filter(|w| (w[1] - w[0]).abs() > 1e-15).count();
    changes as f64 / (chain.len() - 1) as f64
}

/// Effective sample size (ESS) of MCMC chain.
pub fn mcmc_ess(chain: &[f64]) -> f64 {
    let n = chain.len();
    let mean = chain.iter().sum::<f64>() / n as f64;
    let var = chain.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var < 1e-15 { return n as f64; }
    let mut rho_sum = 0.0;
    for lag in 1..n / 2 {
        let rho: f64 = chain.windows(lag + 1).map(|w| (w[0] - mean) * (w[lag] - mean)).sum::<f64>() / (n as f64 * var);
        if rho < 0.0 { break; }
        rho_sum += rho;
    }
    n as f64 / (1.0 + 2.0 * rho_sum)
}

/// Geweke diagnostic: compare first 10% and last 50% of chain.
/// Returns z-score; |z| < 2 indicates convergence.
pub fn geweke_diagnostic(chain: &[f64]) -> f64 {
    let n = chain.len();
    let first_end = n / 10;
    let last_start = n / 2;
    let first = &chain[..first_end];
    let last = &chain[last_start..];
    let mean1 = first.iter().sum::<f64>() / first.len() as f64;
    let mean2 = last.iter().sum::<f64>() / last.len() as f64;
    let var1 = first.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / first.len() as f64;
    let var2 = last.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / last.len() as f64;
    let se = (var1 / first.len() as f64 + var2 / last.len() as f64).sqrt();
    if se < 1e-15 { return 0.0; }
    (mean1 - mean2) / se
}

/// Highest Posterior Density interval (HPD) from MCMC samples.
pub fn hpd_interval(samples: &[f64], confidence: f64) -> (f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let interval_size = (n as f64 * confidence).ceil() as usize;
    let mut min_width = f64::INFINITY;
    let mut best = (sorted[0], sorted[interval_size - 1]);
    for i in 0..n - interval_size + 1 {
        let width = sorted[i + interval_size - 1] - sorted[i];
        if width < min_width { min_width = width; best = (sorted[i], sorted[i + interval_size - 1]); }
    }
    best
}

// ═══════════════════════════════════════════════════════════════════
// PREDICTIVE DISTRIBUTIONS
// ═══════════════════════════════════════════════════════════════════

/// Beta-Binomial predictive probability: P(k successes in m future trials).
pub fn beta_binomial_predictive(alpha: f64, beta: f64, k: usize, m: usize) -> f64 {
    let log_binom = crate::combinatorics::log_binomial(m as u64, k as u64);
    let log_beta_num = lbeta(k as f64 + alpha, (m - k) as f64 + beta);
    let log_beta_den = lbeta(alpha, beta);
    (log_binom + log_beta_num - log_beta_den).exp()
}

/// Posterior predictive mean for Normal-Normal model.
pub fn normal_predictive_mean(mu_post: f64) -> f64 { mu_post }

/// Posterior predictive variance for Normal-Normal model.
pub fn normal_predictive_variance(sigma_post_sq: f64, sigma_sq: f64) -> f64 {
    sigma_post_sq + sigma_sq
}

fn lbeta(a: f64, b: f64) -> f64 {
    crate::special_functions::lgamma(a) + crate::special_functions::lgamma(b) - crate::special_functions::lgamma(a + b)
}

fn seed_state(seed: u64) -> [u64; 4] {
    let mut s = [seed, seed.wrapping_mul(6364136223846793005).wrapping_add(1),
                 seed.wrapping_mul(1442695040888963407).wrapping_add(3),
                 seed.wrapping_mul(2862933555777941757).wrapping_add(7)];
    for _ in 0..20 { prng_next(&mut s); }
    s
}
fn prng_next(s: &mut [u64; 4]) -> u64 {
    let r = (s[0].wrapping_add(s[3])).rotate_left(23).wrapping_add(s[0]);
    let t = s[1] << 17; s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = s[3].rotate_left(45);
    r
}
fn prng_f64(s: &mut [u64; 4]) -> f64 { (prng_next(s) >> 11) as f64 / (1u64 << 53) as f64 }
fn prng_normal(s: &mut [u64; 4]) -> f64 {
    let u1 = prng_f64(s).max(1e-15);
    let u2 = prng_f64(s);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_binomial_update() {
        let (a, b) = beta_binomial_update(1.0, 1.0, 7, 10); // uniform prior + 7/10
        assert!((a - 8.0).abs() < 1e-10);
        assert!((b - 4.0).abs() < 1e-10);
        assert!((beta_mean(a, b) - 8.0/12.0).abs() < 1e-10);
    }

    #[test]
    fn test_normal_normal_update() {
        let (mu, sigma2) = normal_normal_update(0.0, 1.0, 5.0, 10, 1.0);
        // Posterior should be between prior and data
        assert!(mu > 0.0 && mu < 5.0);
        assert!(sigma2 < 1.0); // more precise than prior
    }

    #[test]
    fn test_bayes_factor() {
        let bf = bayes_factor_binomial(7, 10, 0.5, 1.0, 1.0);
        assert!(bf.is_finite());
    }

    #[test]
    fn test_posterior_probability() {
        // BF=1, prior=0.5 → posterior=0.5
        assert!((posterior_probability(1.0, 0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_metropolis_hastings() {
        // Sample from standard normal
        let chain = metropolis_hastings(|x| -0.5 * x * x, 0.0, 1.0, 5000, 42);
        assert_eq!(chain.len(), 5000);
        let mean = chain.iter().sum::<f64>() / 5000.0;
        assert!(mean.abs() < 0.5); // should be near 0
    }

    #[test]
    fn test_mcmc_acceptance() {
        let chain = metropolis_hastings(|x| -0.5 * x * x, 0.0, 1.0, 1000, 42);
        let rate = mcmc_acceptance_rate(&chain);
        assert!(rate > 0.1 && rate < 0.9);
    }

    #[test]
    fn test_hpd() {
        let samples: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
        let (lo, hi) = hpd_interval(&samples, 0.95);
        assert!(hi - lo < 1.0);
    }

    #[test]
    fn test_dirichlet_update() {
        let prior = vec![1.0, 1.0, 1.0];
        let counts = vec![5, 3, 2];
        let post = dirichlet_multinomial_update(&prior, &counts);
        assert_eq!(post, vec![6.0, 4.0, 3.0]);
        let mean = dirichlet_mean(&post);
        assert!((mean.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_geweke() {
        // Constant chain → z = 0
        let chain = vec![1.0; 100];
        assert!(geweke_diagnostic(&chain).abs() < 1e-10);
    }

    #[test]
    fn test_credible_interval() {
        let (lo, hi) = beta_credible_interval(10.0, 10.0, 0.95);
        assert!(lo < 0.5 && hi > 0.5);
    }
}
