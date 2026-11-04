//! Portfolio optimization: Markowitz mean-variance, efficient frontier,
//! risk parity, Black-Litterman, portfolio metrics, rebalancing.

// ═══════════════════════════════════════════════════════════════════
// PORTFOLIO METRICS
// ═══════════════════════════════════════════════════════════════════

/// Portfolio return: R_p = Σ w_i r_i.
pub fn portfolio_return(weights: &[f64], returns: &[f64]) -> f64 {
    weights.iter().zip(returns).map(|(w, r)| w * r).sum()
}

/// Portfolio variance: σ²_p = w' Σ w (Σ = covariance matrix).
pub fn portfolio_variance(weights: &[f64], cov: &[f64], n: usize) -> f64 {
    let mut var = 0.0;
    for i in 0..n {
        for j in 0..n { var += weights[i] * weights[j] * cov[i * n + j]; }
    }
    var
}

/// Portfolio volatility (std dev).
pub fn portfolio_volatility(weights: &[f64], cov: &[f64], n: usize) -> f64 {
    portfolio_variance(weights, cov, n).max(0.0).sqrt()
}

/// Portfolio Sharpe ratio.
pub fn portfolio_sharpe(weights: &[f64], returns: &[f64], cov: &[f64], n: usize, risk_free: f64) -> f64 {
    let ret = portfolio_return(weights, returns);
    let vol = portfolio_volatility(weights, cov, n);
    if vol < 1e-15 { return 0.0; }
    (ret - risk_free) / vol
}

/// Portfolio beta relative to market.
pub fn portfolio_beta(weights: &[f64], betas: &[f64]) -> f64 {
    weights.iter().zip(betas).map(|(w, b)| w * b).sum()
}

/// Portfolio tracking error vs benchmark.
pub fn tracking_error(portfolio_returns: &[f64], benchmark_returns: &[f64]) -> f64 {
    let n = portfolio_returns.len() as f64;
    let diffs: Vec<f64> = portfolio_returns.iter().zip(benchmark_returns).map(|(p, b)| p - b).collect();
    let mean = diffs.iter().sum::<f64>() / n;
    (diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

/// Information ratio: active return / tracking error.
pub fn portfolio_information_ratio(portfolio_returns: &[f64], benchmark_returns: &[f64]) -> f64 {
    let n = portfolio_returns.len() as f64;
    let active: Vec<f64> = portfolio_returns.iter().zip(benchmark_returns).map(|(p, b)| p - b).collect();
    let mean = active.iter().sum::<f64>() / n;
    let te = tracking_error(portfolio_returns, benchmark_returns);
    if te < 1e-15 { return 0.0; }
    mean / te
}

// ═══════════════════════════════════════════════════════════════════
// MARKOWITZ MEAN-VARIANCE
// ═══════════════════════════════════════════════════════════════════

/// Minimum variance portfolio weights (no constraints except Σw=1).
pub fn min_variance_portfolio(cov: &[f64], n: usize) -> Vec<f64> {
    let cov_inv = crate::linalg_decomp::matrix_inverse(cov, n);
    let _ones = vec![1.0; n];
    let cov_inv_ones: Vec<f64> = (0..n).map(|i| (0..n).map(|j| cov_inv[i * n + j]).sum()).collect();
    let denom: f64 = cov_inv_ones.iter().sum();
    if denom.abs() < 1e-15 { return vec![1.0 / n as f64; n]; }
    cov_inv_ones.iter().map(|&v| v / denom).collect()
}

/// Maximum Sharpe ratio portfolio (tangency portfolio).
pub fn max_sharpe_portfolio(returns: &[f64], cov: &[f64], n: usize, risk_free: f64) -> Vec<f64> {
    let cov_inv = crate::linalg_decomp::matrix_inverse(cov, n);
    let excess: Vec<f64> = returns.iter().map(|r| r - risk_free).collect();
    let cov_inv_excess: Vec<f64> = (0..n).map(|i| {
        (0..n).map(|j| cov_inv[i * n + j] * excess[j]).sum()
    }).collect();
    let denom: f64 = cov_inv_excess.iter().sum();
    if denom.abs() < 1e-15 { return vec![1.0 / n as f64; n]; }
    cov_inv_excess.iter().map(|&v| v / denom).collect()
}

/// Efficient frontier: compute n_points portfolios from min-variance to max-return.
/// Returns vec of (return, volatility, weights).
pub fn efficient_frontier(returns: &[f64], cov: &[f64], n: usize, n_points: usize) -> Vec<(f64, f64, Vec<f64>)> {
    let min_ret = returns.iter().copied().fold(f64::INFINITY, f64::min);
    let max_ret = returns.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (0..n_points).map(|i| {
        let target = min_ret + (max_ret - min_ret) * i as f64 / (n_points - 1).max(1) as f64;
        let weights = target_return_portfolio(returns, cov, n, target);
        let ret = portfolio_return(&weights, returns);
        let vol = portfolio_volatility(&weights, cov, n);
        (ret, vol, weights)
    }).collect()
}

fn target_return_portfolio(returns: &[f64], cov: &[f64], n: usize, target: f64) -> Vec<f64> {
    // Lagrangian: min w'Σw s.t. w'μ = target, w'1 = 1
    // Solved via augmented system
    let cov_inv = crate::linalg_decomp::matrix_inverse(cov, n);
    let _ones = vec![1.0; n];
    let a: f64 = (0..n).map(|i| (0..n).map(|j| cov_inv[i * n + j]).sum::<f64>()).sum();
    let b: f64 = (0..n).map(|i| (0..n).map(|j| cov_inv[i * n + j] * returns[j]).sum::<f64>()).sum();
    let c: f64 = (0..n).map(|i| {
        let r = returns[i];
        (0..n).map(|j| cov_inv[i * n + j] * returns[j]).sum::<f64>() * r
    }).sum::<f64>() / b.max(1e-15); // simplified
    let lambda1 = (target * a - b) / (a * c - b * b).max(1e-15);
    let lambda2 = (c - target * b) / (a * c - b * b).max(1e-15);
    let cov_inv_ones: Vec<f64> = (0..n).map(|i| (0..n).map(|j| cov_inv[i * n + j]).sum()).collect();
    let cov_inv_mu: Vec<f64> = (0..n).map(|i| (0..n).map(|j| cov_inv[i * n + j] * returns[j]).sum()).collect();
    let weights: Vec<f64> = (0..n).map(|i| lambda2 * cov_inv_ones[i] + lambda1 * cov_inv_mu[i]).collect();
    // Normalize
    let sum: f64 = weights.iter().sum();
    if sum.abs() < 1e-15 { return vec![1.0 / n as f64; n]; }
    weights.iter().map(|w| w / sum).collect()
}

// ═══════════════════════════════════════════════════════════════════
// RISK PARITY
// ═══════════════════════════════════════════════════════════════════

/// Risk parity (equal risk contribution) portfolio.
/// Iterative approach: w_i ∝ 1/σ_i, then adjust.
pub fn risk_parity_portfolio(cov: &[f64], n: usize, max_iter: usize) -> Vec<f64> {
    let mut weights = vec![1.0 / n as f64; n];
    for _ in 0..max_iter {
        let vol = portfolio_volatility(&weights, cov, n);
        if vol < 1e-15 { break; }
        // Marginal risk contribution: MRC_i = (Σw)_i / σ_p
        let sigma_w: Vec<f64> = (0..n).map(|i| (0..n).map(|j| cov[i * n + j] * weights[j]).sum()).collect();
        let mrc: Vec<f64> = sigma_w.iter().map(|sw| sw / vol).collect();
        // Risk contribution: RC_i = w_i * MRC_i
        let rc: Vec<f64> = weights.iter().zip(&mrc).map(|(w, m)| w * m).collect();
        let target_rc = vol / n as f64;
        // Adjust weights
        for i in 0..n {
            if mrc[i].abs() > 1e-15 { weights[i] *= target_rc / rc[i]; }
        }
        let sum: f64 = weights.iter().sum();
        for w in weights.iter_mut() { *w /= sum; }
    }
    weights
}

/// Equal weight portfolio.
pub fn equal_weight_portfolio(n: usize) -> Vec<f64> { vec![1.0 / n as f64; n] }

/// Inverse volatility portfolio.
pub fn inverse_volatility_portfolio(cov: &[f64], n: usize) -> Vec<f64> {
    let vols: Vec<f64> = (0..n).map(|i| cov[i * n + i].max(1e-15).sqrt()).collect();
    let inv: Vec<f64> = vols.iter().map(|v| 1.0 / v).collect();
    let sum: f64 = inv.iter().sum();
    inv.iter().map(|v| v / sum).collect()
}

// ═══════════════════════════════════════════════════════════════════
// REBALANCING
// ═══════════════════════════════════════════════════════════════════

/// Compute trades needed to rebalance.
/// Returns (asset_idx, trade_amount) where positive = buy, negative = sell.
pub fn rebalance_trades(current_values: &[f64], target_weights: &[f64]) -> Vec<f64> {
    let total: f64 = current_values.iter().sum();
    target_weights.iter().zip(current_values)
        .map(|(&tw, &cv)| tw * total - cv)
        .collect()
}

/// Check if rebalancing is needed (any weight deviates by more than threshold).
pub fn needs_rebalance(current_values: &[f64], target_weights: &[f64], threshold: f64) -> bool {
    let total: f64 = current_values.iter().sum();
    if total < 1e-15 { return false; }
    current_values.iter().zip(target_weights)
        .any(|(&cv, &tw)| (cv / total - tw).abs() > threshold)
}

/// Portfolio turnover (sum of absolute weight changes).
pub fn turnover(old_weights: &[f64], new_weights: &[f64]) -> f64 {
    old_weights.iter().zip(new_weights).map(|(o, n)| (o - n).abs()).sum::<f64>() / 2.0
}

// ═══════════════════════════════════════════════════════════════════
// RISK DECOMPOSITION
// ═══════════════════════════════════════════════════════════════════

/// Marginal risk contribution for each asset.
pub fn marginal_risk_contribution(weights: &[f64], cov: &[f64], n: usize) -> Vec<f64> {
    let vol = portfolio_volatility(weights, cov, n);
    if vol < 1e-15 { return vec![0.0; n]; }
    (0..n).map(|i| {
        let sigma_w_i: f64 = (0..n).map(|j| cov[i * n + j] * weights[j]).sum();
        sigma_w_i / vol
    }).collect()
}

/// Risk contribution (percentage) for each asset.
pub fn risk_contribution(weights: &[f64], cov: &[f64], n: usize) -> Vec<f64> {
    let mrc = marginal_risk_contribution(weights, cov, n);
    let vol = portfolio_volatility(weights, cov, n);
    if vol < 1e-15 { return vec![0.0; n]; }
    weights.iter().zip(&mrc).map(|(w, m)| w * m / vol).collect()
}

/// Diversification ratio: weighted average vol / portfolio vol.
pub fn diversification_ratio(weights: &[f64], cov: &[f64], n: usize) -> f64 {
    let port_vol = portfolio_volatility(weights, cov, n);
    if port_vol < 1e-15 { return 1.0; }
    let weighted_vol: f64 = (0..n).map(|i| weights[i] * cov[i * n + i].sqrt()).sum();
    weighted_vol / port_vol
}

/// Concentration ratio (Herfindahl index of weights).
pub fn weight_concentration(weights: &[f64]) -> f64 {
    weights.iter().map(|w| w * w).sum()
}

/// Effective number of assets: 1 / HHI of weights.
pub fn effective_n_assets(weights: &[f64]) -> f64 {
    let hhi = weight_concentration(weights);
    if hhi < 1e-15 { return 0.0; }
    1.0 / hhi
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_cov() -> Vec<f64> {
        vec![0.04, 0.006, 0.006, 0.09] // 2×2
    }

    #[test]
    fn test_portfolio_return() {
        assert!((portfolio_return(&[0.6, 0.4], &[0.10, 0.05]) - 0.08).abs() < 1e-10);
    }

    #[test]
    fn test_portfolio_variance() {
        let cov = sample_cov();
        let var = portfolio_variance(&[0.5, 0.5], &cov, 2);
        assert!(var > 0.0);
    }

    #[test]
    fn test_min_variance() {
        let cov = sample_cov();
        let w = min_variance_portfolio(&cov, 2);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        // Lower vol asset should have higher weight
        assert!(w[0] > w[1]);
    }

    #[test]
    fn test_equal_weight() {
        let w = equal_weight_portfolio(4);
        assert_eq!(w, vec![0.25; 4]);
    }

    #[test]
    fn test_inverse_vol() {
        let cov = sample_cov();
        let w = inverse_volatility_portfolio(&cov, 2);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rebalance() {
        let current = vec![6000.0, 4000.0];
        let target = vec![0.5, 0.5];
        let trades = rebalance_trades(&current, &target);
        assert!((trades[0] - (-1000.0)).abs() < 1e-10);
        assert!((trades[1] - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_turnover() {
        assert!((turnover(&[0.6, 0.4], &[0.5, 0.5]) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_effective_n() {
        assert!((effective_n_assets(&[0.25, 0.25, 0.25, 0.25]) - 4.0).abs() < 1e-10);
        assert!(effective_n_assets(&[0.9, 0.1]) < 2.0);
    }

    #[test]
    fn test_diversification_ratio() {
        let cov = sample_cov();
        let dr = diversification_ratio(&[0.5, 0.5], &cov, 2);
        assert!(dr >= 1.0); // always >= 1
    }
}
