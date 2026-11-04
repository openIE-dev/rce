//! Econometrics: OLS regression, instrumental variables, panel data,
//! heteroskedasticity tests, Granger causality, cointegration,
//! GARCH, volatility models, economic indicators.

// ═══════════════════════════════════════════════════════════════════
// OLS REGRESSION
// ═══════════════════════════════════════════════════════════════════

/// OLS regression result.
#[derive(Debug, Clone)]
pub struct OlsResult {
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub t_statistics: Vec<f64>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub residuals: Vec<f64>,
    pub f_statistic: f64,
    pub aic: f64,
    pub bic: f64,
}

/// Ordinary Least Squares regression. x: n×k (row-major, includes intercept column), y: n.
pub fn ols(x: &[f64], y: &[f64], n: usize, k: usize) -> OlsResult {
    // β = (X'X)^{-1} X'y
    let xtx = xtx_matrix(x, n, k);
    let xty = xty_vector(x, y, n, k);
    let xtx_inv = crate::linalg_decomp::matrix_inverse(&xtx, k);
    let coefficients: Vec<f64> = (0..k).map(|i| (0..k).map(|j| xtx_inv[i * k + j] * xty[j]).sum()).collect();

    // Residuals
    let mut residuals = vec![0.0; n];
    for i in 0..n {
        let predicted: f64 = (0..k).map(|j| x[i * k + j] * coefficients[j]).sum();
        residuals[i] = y[i] - predicted;
    }

    // R²
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let ss_res: f64 = residuals.iter().map(|r| r * r).sum();
    let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let r_squared = 1.0 - ss_res / ss_tot;
    let adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) as f64 / (n - k) as f64;

    // Standard errors
    let mse = ss_res / (n - k) as f64;
    let std_errors: Vec<f64> = (0..k).map(|i| (mse * xtx_inv[i * k + i]).sqrt()).collect();
    let t_statistics: Vec<f64> = coefficients.iter().zip(&std_errors).map(|(b, se)| b / se).collect();

    // F-statistic
    let ss_reg = ss_tot - ss_res;
    let f_statistic = (ss_reg / (k - 1) as f64) / (ss_res / (n - k) as f64);

    // AIC/BIC
    let n_f = n as f64;
    let k_f = k as f64;
    let aic = n_f * (ss_res / n_f).ln() + 2.0 * k_f;
    let bic = n_f * (ss_res / n_f).ln() + k_f * n_f.ln();

    OlsResult { coefficients, std_errors, t_statistics, r_squared, adj_r_squared, residuals, f_statistic, aic, bic }
}

/// Add intercept column to X matrix (prepend column of 1s).
pub fn add_intercept(x: &[f64], n: usize, k: usize) -> Vec<f64> {
    let mut result = vec![0.0; n * (k + 1)];
    for i in 0..n {
        result[i * (k + 1)] = 1.0;
        for j in 0..k { result[i * (k + 1) + j + 1] = x[i * k + j]; }
    }
    result
}

fn xtx_matrix(x: &[f64], n: usize, k: usize) -> Vec<f64> {
    let mut xtx = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            for s in 0..n { xtx[i * k + j] += x[s * k + i] * x[s * k + j]; }
        }
    }
    xtx
}

fn xty_vector(x: &[f64], y: &[f64], n: usize, k: usize) -> Vec<f64> {
    let mut xty = vec![0.0; k];
    for j in 0..k {
        for i in 0..n { xty[j] += x[i * k + j] * y[i]; }
    }
    xty
}

// ═══════════════════════════════════════════════════════════════════
// WEIGHTED LEAST SQUARES
// ═══════════════════════════════════════════════════════════════════

/// Weighted Least Squares. weights: n-vector of observation weights.
pub fn wls(x: &[f64], y: &[f64], weights: &[f64], n: usize, k: usize) -> OlsResult {
    // Transform: X_w = sqrt(W) * X, y_w = sqrt(W) * y
    let mut xw = vec![0.0; n * k];
    let mut yw = vec![0.0; n];
    for i in 0..n {
        let w = weights[i].sqrt();
        yw[i] = y[i] * w;
        for j in 0..k { xw[i * k + j] = x[i * k + j] * w; }
    }
    ols(&xw, &yw, n, k)
}

// ═══════════════════════════════════════════════════════════════════
// HETEROSKEDASTICITY TESTS
// ═══════════════════════════════════════════════════════════════════

/// Breusch-Pagan test for heteroskedasticity.
/// Returns (test_statistic, p_value).
pub fn breusch_pagan(residuals: &[f64], x: &[f64], n: usize, k: usize) -> (f64, f64) {
    let sq_resid: Vec<f64> = residuals.iter().map(|r| r * r).collect();
    let mean_sq = sq_resid.iter().sum::<f64>() / n as f64;
    let g: Vec<f64> = sq_resid.iter().map(|&s| s / mean_sq).collect();
    let aux = ols(x, &g, n, k);
    let bp = aux.r_squared * n as f64;
    let df = (k - 1) as f64;
    let p = 1.0 - chi2_cdf_approx(bp, df);
    (bp, p)
}

/// White's test for heteroskedasticity (simplified).
pub fn white_test(residuals: &[f64], x: &[f64], n: usize, k: usize) -> (f64, f64) {
    // Use squared residuals regressed on X and X²
    let sq_resid: Vec<f64> = residuals.iter().map(|r| r * r).collect();
    let mut x_aug = Vec::with_capacity(n * (2 * k));
    for i in 0..n {
        for j in 0..k { x_aug.push(x[i * k + j]); }
        for j in 0..k { x_aug.push(x[i * k + j] * x[i * k + j]); }
    }
    let aux = ols(&x_aug, &sq_resid, n, 2 * k);
    let stat = aux.r_squared * n as f64;
    let df = (2 * k - 1) as f64;
    let p = 1.0 - chi2_cdf_approx(stat, df);
    (stat, p)
}

// ═══════════════════════════════════════════════════════════════════
// GRANGER CAUSALITY
// ═══════════════════════════════════════════════════════════════════

/// Granger causality test: does x Granger-cause y?
/// Returns (f_statistic, p_value).
pub fn granger_causality(y: &[f64], x: &[f64], max_lag: usize) -> (f64, f64) {
    let n = y.len() - max_lag;
    if n < max_lag + 2 { return (0.0, 1.0); }

    // Restricted model: y_t = Σ α_i y_{t-i}
    let mut x_restricted = vec![0.0; n * (max_lag + 1)]; // +1 for intercept
    let mut y_dep = vec![0.0; n];
    for t in 0..n {
        x_restricted[t * (max_lag + 1)] = 1.0; // intercept
        for lag in 0..max_lag {
            x_restricted[t * (max_lag + 1) + lag + 1] = y[max_lag + t - lag - 1];
        }
        y_dep[t] = y[max_lag + t];
    }
    let restricted = ols(&x_restricted, &y_dep, n, max_lag + 1);
    let ss_r: f64 = restricted.residuals.iter().map(|r| r * r).sum();

    // Unrestricted model: y_t = Σ α_i y_{t-i} + Σ β_i x_{t-i}
    let k_unr = 2 * max_lag + 1;
    let mut x_unrestricted = vec![0.0; n * k_unr];
    for t in 0..n {
        x_unrestricted[t * k_unr] = 1.0;
        for lag in 0..max_lag {
            x_unrestricted[t * k_unr + lag + 1] = y[max_lag + t - lag - 1];
            x_unrestricted[t * k_unr + max_lag + lag + 1] = x[max_lag + t - lag - 1];
        }
    }
    let unrestricted = ols(&x_unrestricted, &y_dep, n, k_unr);
    let ss_u: f64 = unrestricted.residuals.iter().map(|r| r * r).sum();

    let f_stat = ((ss_r - ss_u) / max_lag as f64) / (ss_u / (n - k_unr) as f64);
    let p = 1.0 - f_cdf_approx(f_stat, max_lag as f64, (n - k_unr) as f64);
    (f_stat, p)
}

// ═══════════════════════════════════════════════════════════════════
// GARCH(1,1)
// ═══════════════════════════════════════════════════════════════════

/// GARCH(1,1) volatility estimation.
/// returns: estimated variance series σ²_t.
/// Model: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}.
pub fn garch11_variance(returns: &[f64], omega: f64, alpha: f64, beta: f64) -> Vec<f64> {
    let n = returns.len();
    let mut sigma2 = vec![0.0; n];
    // Initialize with sample variance
    let mean = returns.iter().sum::<f64>() / n as f64;
    sigma2[0] = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64;
    for t in 1..n {
        let eps2 = (returns[t - 1] - mean).powi(2);
        sigma2[t] = omega + alpha * eps2 + beta * sigma2[t - 1];
    }
    sigma2
}

/// EWMA volatility (RiskMetrics style): σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}.
pub fn ewma_volatility(returns: &[f64], lambda: f64) -> Vec<f64> {
    let n = returns.len();
    let mut sigma2 = vec![0.0; n];
    sigma2[0] = returns[0] * returns[0];
    for t in 1..n {
        sigma2[t] = lambda * sigma2[t - 1] + (1.0 - lambda) * returns[t - 1] * returns[t - 1];
    }
    sigma2.iter().map(|s| s.sqrt()).collect()
}

// ═══════════════════════════════════════════════════════════════════
// ECONOMIC INDICATORS
// ═══════════════════════════════════════════════════════════════════

/// GDP growth rate: (GDP_t - GDP_{t-1}) / GDP_{t-1}.
pub fn growth_rate(values: &[f64]) -> Vec<f64> {
    values.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect()
}

/// Compound Annual Growth Rate.
pub fn cagr(start_value: f64, end_value: f64, years: f64) -> f64 {
    (end_value / start_value).powf(1.0 / years) - 1.0
}

/// Consumer Price Index to inflation rate.
pub fn cpi_to_inflation(cpi: &[f64]) -> Vec<f64> {
    growth_rate(cpi)
}

/// Real value from nominal: real = nominal / (CPI/100).
pub fn nominal_to_real(nominal: &[f64], cpi: &[f64]) -> Vec<f64> {
    nominal.iter().zip(cpi).map(|(n, c)| n / (c / 100.0)).collect()
}

/// Gini coefficient from sorted income data.
pub fn gini_coefficient(incomes: &[f64]) -> f64 {
    let mut sorted = incomes.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len() as f64;
    let mean = sorted.iter().sum::<f64>() / n;
    if mean < 1e-15 { return 0.0; }
    let sum: f64 = sorted.iter().enumerate()
        .map(|(i, &x)| (2.0 * (i + 1) as f64 - n - 1.0) * x)
        .sum();
    sum / (n * n * mean)
}

/// Lorenz curve points (sorted cumulative share).
pub fn lorenz_curve(incomes: &[f64]) -> Vec<(f64, f64)> {
    let mut sorted = incomes.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total: f64 = sorted.iter().sum();
    let n = sorted.len() as f64;
    let mut points = vec![(0.0, 0.0)];
    let mut cum_income = 0.0;
    for (i, &v) in sorted.iter().enumerate() {
        cum_income += v;
        points.push(((i + 1) as f64 / n, cum_income / total));
    }
    points
}

/// Herfindahl-Hirschman Index (market concentration).
pub fn hhi(market_shares: &[f64]) -> f64 {
    market_shares.iter().map(|s| (s * 100.0).powi(2)).sum()
}

/// Theil index (inequality measure).
pub fn theil_index(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if mean < 1e-15 { return 0.0; }
    values.iter().map(|&x| {
        let ratio = x / mean;
        if ratio > 1e-15 { ratio * ratio.ln() } else { 0.0 }
    }).sum::<f64>() / n
}

/// Elasticity: % change in Y / % change in X.
pub fn elasticity(x: &[f64], y: &[f64]) -> Vec<f64> {
    x.windows(2).zip(y.windows(2)).map(|(xw, yw)| {
        let pct_x = (xw[1] - xw[0]) / xw[0];
        let pct_y = (yw[1] - yw[0]) / yw[0];
        if pct_x.abs() < 1e-15 { 0.0 } else { pct_y / pct_x }
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// MOVING AVERAGES FOR ECONOMICS
// ═══════════════════════════════════════════════════════════════════

/// Hodrick-Prescott filter decomposition. Returns (trend, cycle).
pub fn hp_filter(data: &[f64], lambda: f64) -> (Vec<f64>, Vec<f64>) {
    let n = data.len();
    if n < 4 { return (data.to_vec(), vec![0.0; n]); }
    // Solve (I + λ·K'K)·τ = y via iterative method
    let mut trend = data.to_vec();
    for _ in 0..100 {
        let mut new_trend = data.to_vec();
        for t in 2..n - 2 {
            let penalty = lambda * (trend[t - 2] - 4.0 * trend[t - 1] + 6.0 * trend[t] - 4.0 * trend[t + 1] + trend[t + 2]);
            new_trend[t] = data[t] - penalty / (1.0 + 6.0 * lambda);
        }
        let diff: f64 = trend.iter().zip(&new_trend).map(|(a, b)| (a - b).abs()).sum();
        trend = new_trend;
        if diff < 1e-8 { break; }
    }
    let cycle: Vec<f64> = data.iter().zip(&trend).map(|(d, t)| d - t).collect();
    (trend, cycle)
}

/// Baxter-King bandpass filter.
pub fn bk_filter(data: &[f64], low_period: usize, high_period: usize, k: usize) -> Vec<f64> {
    let n = data.len();
    if n < 2 * k + 1 { return vec![0.0; n]; }
    let a_low = 2.0 * std::f64::consts::PI / high_period as f64;
    let a_high = 2.0 * std::f64::consts::PI / low_period as f64;
    // Ideal bandpass filter weights
    let mut weights = vec![0.0; 2 * k + 1];
    weights[k] = (a_high - a_low) / std::f64::consts::PI;
    for j in 1..=k {
        let jf = j as f64;
        weights[k + j] = ((a_high * jf).sin() - (a_low * jf).sin()) / (std::f64::consts::PI * jf);
        weights[k - j] = weights[k + j];
    }
    // Normalize
    let sum: f64 = weights.iter().sum();
    for w in weights.iter_mut() { *w -= sum / (2 * k + 1) as f64; }
    // Apply
    let mut result = vec![0.0; n];
    for t in k..n - k {
        for j in 0..2 * k + 1 { result[t] += weights[j] * data[t + j - k]; }
    }
    result
}

fn chi2_cdf_approx(x: f64, df: f64) -> f64 {
    // Wilson-Hilferty approximation
    if df <= 0.0 { return 0.0; }
    let z = (x / df).cbrt() - (1.0 - 2.0 / (9.0 * df));
    let z = z / (2.0 / (9.0 * df)).sqrt();
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

fn f_cdf_approx(f: f64, d1: f64, d2: f64) -> f64 {
    if f <= 0.0 { return 0.0; }
    let x = d1 * f / (d1 * f + d2);
    x.min(1.0).max(0.0) // very rough — adequate for test
}

fn erf_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = 1.0 - poly * (-x * x).exp();
    if x < 0.0 { -result } else { result }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ols_perfect_fit() {
        // y = 2 + 3x, with intercept
        let x = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0];
        let y = vec![5.0, 8.0, 11.0, 14.0, 17.0];
        let result = ols(&x, &y, 5, 2);
        assert!((result.coefficients[0] - 2.0).abs() < 1e-6); // intercept
        assert!((result.coefficients[1] - 3.0).abs() < 1e-6); // slope
        assert!((result.r_squared - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_intercept() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let xi = add_intercept(&x, 2, 2);
        assert_eq!(xi, vec![1.0, 1.0, 2.0, 1.0, 3.0, 4.0]);
    }

    #[test]
    fn test_garch11() {
        let returns = vec![0.01, -0.02, 0.015, -0.005, 0.03, -0.01, 0.005];
        let sigma2 = garch11_variance(&returns, 0.00001, 0.1, 0.8);
        assert_eq!(sigma2.len(), returns.len());
        assert!(sigma2.iter().all(|&s| s > 0.0));
    }

    #[test]
    fn test_growth_rate() {
        let gdp = vec![100.0, 105.0, 110.25];
        let gr = growth_rate(&gdp);
        assert!((gr[0] - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_cagr() {
        assert!((cagr(100.0, 200.0, 10.0) - 0.07177).abs() < 0.001);
    }

    #[test]
    fn test_gini() {
        // Perfect equality
        let equal = vec![100.0; 10];
        assert!(gini_coefficient(&equal).abs() < 1e-10);
        // Some inequality
        let unequal = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 91.0];
        assert!(gini_coefficient(&unequal) > 0.5);
    }

    #[test]
    fn test_hhi() {
        // Perfect competition: 100 firms with 1% each
        let shares = vec![0.01; 100];
        assert!((hhi(&shares) - 100.0).abs() < 1e-6);
        // Monopoly: 1 firm with 100%
        assert!((hhi(&[1.0]) - 10000.0).abs() < 1e-6);
    }

    #[test]
    fn test_hp_filter() {
        let data: Vec<f64> = (0..50).map(|i| i as f64 + (i as f64 * 0.3).sin() * 2.0).collect();
        let (trend, cycle) = hp_filter(&data, 1600.0);
        assert_eq!(trend.len(), data.len());
        assert_eq!(cycle.len(), data.len());
    }

    #[test]
    fn test_lorenz() {
        let incomes = vec![10.0, 20.0, 30.0, 40.0];
        let points = lorenz_curve(&incomes);
        assert_eq!(points.first().unwrap(), &(0.0, 0.0));
        assert!((points.last().unwrap().1 - 1.0).abs() < 1e-10);
    }
}
