//! Financial mathematics: option pricing, portfolio theory,
//! risk metrics, bond pricing, time value of money, technical indicators.

use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════
// TIME VALUE OF MONEY
// ═══════════════════════════════════════════════════════════════════

/// Future value: FV = PV × (1 + r)^n.
pub fn future_value(pv: f64, rate: f64, periods: f64) -> f64 {
    pv * (1.0 + rate).powf(periods)
}

/// Present value: PV = FV / (1 + r)^n.
pub fn present_value(fv: f64, rate: f64, periods: f64) -> f64 {
    fv / (1.0 + rate).powf(periods)
}

/// Net Present Value of cash flows. cashflows[0] = initial investment (negative).
pub fn npv(rate: f64, cashflows: &[f64]) -> f64 {
    let base: f64 = 1.0 + rate;
    cashflows.iter().enumerate()
        .map(|(i, cf)| cf / base.powi(i as i32))
        .sum()
}

/// Internal Rate of Return (Newton's method).
pub fn irr(cashflows: &[f64]) -> Option<f64> {
    let mut r = 0.1;
    for _ in 0..200 {
        let base: f64 = 1.0 + r;
        let f: f64 = cashflows.iter().enumerate()
            .map(|(i, cf)| cf / base.powi(i as i32))
            .sum();
        let fp: f64 = cashflows.iter().enumerate()
            .map(|(i, cf)| -(i as f64) * cf / base.powi(i as i32 + 1))
            .sum();
        if fp.abs() < 1e-15 { return None; }
        let new_r = r - f / fp;
        if (new_r - r).abs() < 1e-10 { return Some(new_r); }
        r = new_r;
    }
    Some(r)
}

/// Annuity payment: PMT = PV × r / (1 - (1+r)^{-n}).
pub fn annuity_payment(pv: f64, rate: f64, periods: f64) -> f64 {
    if rate.abs() < 1e-15 { return pv / periods; }
    pv * rate / (1.0 - (1.0 + rate).powf(-periods))
}

/// Annuity present value: PV = PMT × (1 - (1+r)^{-n}) / r.
pub fn annuity_pv(pmt: f64, rate: f64, periods: f64) -> f64 {
    if rate.abs() < 1e-15 { return pmt * periods; }
    pmt * (1.0 - (1.0 + rate).powf(-periods)) / rate
}

/// Perpetuity: PV = PMT / r.
pub fn perpetuity(pmt: f64, rate: f64) -> f64 {
    pmt / rate
}

/// Growing perpetuity: PV = PMT / (r - g).
pub fn growing_perpetuity(pmt: f64, rate: f64, growth: f64) -> f64 {
    pmt / (rate - growth)
}

// ═══════════════════════════════════════════════════════════════════
// OPTION PRICING (Black-Scholes)
// ═══════════════════════════════════════════════════════════════════

/// Standard normal CDF.
fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0f64.sqrt()))
}

fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
        + t * (-1.453152027 + t * 1.061405429))));
    let result = 1.0 - poly * (-x * x).exp();
    if x < 0.0 { -result } else { result }
}

/// Standard normal PDF.
fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Black-Scholes call option price.
/// S = spot, K = strike, T = time to expiry, r = risk-free rate, sigma = volatility.
pub fn bs_call(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    let d1 = ((s / k).ln() + (r + sigma * sigma / 2.0) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    s * norm_cdf(d1) - k * (-r * t).exp() * norm_cdf(d2)
}

/// Black-Scholes put option price.
pub fn bs_put(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    let d1 = ((s / k).ln() + (r + sigma * sigma / 2.0) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    k * (-r * t).exp() * norm_cdf(-d2) - s * norm_cdf(-d1)
}

/// Black-Scholes Greeks for a call option.
/// Returns (delta, gamma, theta, vega, rho).
pub fn bs_greeks_call(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> (f64, f64, f64, f64, f64) {
    let d1 = ((s / k).ln() + (r + sigma * sigma / 2.0) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    let delta = norm_cdf(d1);
    let gamma = norm_pdf(d1) / (s * sigma * t.sqrt());
    let theta = -(s * norm_pdf(d1) * sigma / (2.0 * t.sqrt()))
        - r * k * (-r * t).exp() * norm_cdf(d2);
    let vega = s * norm_pdf(d1) * t.sqrt();
    let rho = k * t * (-r * t).exp() * norm_cdf(d2);
    (delta, gamma, theta, vega, rho)
}

/// Implied volatility from call price (Newton-Raphson).
pub fn implied_volatility_call(price: f64, s: f64, k: f64, t: f64, r: f64) -> Option<f64> {
    let mut sigma = 0.3; // initial guess
    for _ in 0..100 {
        let p = bs_call(s, k, t, r, sigma);
        let d1 = ((s / k).ln() + (r + sigma * sigma / 2.0) * t) / (sigma * t.sqrt());
        let vega = s * norm_pdf(d1) * t.sqrt();
        if vega.abs() < 1e-15 { return None; }
        let new_sigma = sigma - (p - price) / vega;
        if (new_sigma - sigma).abs() < 1e-8 { return Some(new_sigma); }
        sigma = new_sigma.max(0.001);
    }
    Some(sigma)
}

// ═══════════════════════════════════════════════════════════════════
// BOND PRICING
// ═══════════════════════════════════════════════════════════════════

/// Bond price: sum of discounted coupon payments + par value.
pub fn bond_price(par: f64, coupon_rate: f64, ytm: f64, periods: usize) -> f64 {
    let coupon = par * coupon_rate;
    let mut price = 0.0;
    for t in 1..=periods {
        price += coupon / (1.0 + ytm).powi(t as i32);
    }
    price += par / (1.0 + ytm).powi(periods as i32);
    price
}

/// Macaulay duration.
pub fn macaulay_duration(par: f64, coupon_rate: f64, ytm: f64, periods: usize) -> f64 {
    let coupon = par * coupon_rate;
    let price = bond_price(par, coupon_rate, ytm, periods);
    let mut weighted = 0.0;
    for t in 1..=periods {
        weighted += t as f64 * coupon / (1.0 + ytm).powi(t as i32);
    }
    weighted += periods as f64 * par / (1.0 + ytm).powi(periods as i32);
    weighted / price
}

/// Modified duration.
pub fn modified_duration(par: f64, coupon_rate: f64, ytm: f64, periods: usize) -> f64 {
    macaulay_duration(par, coupon_rate, ytm, periods) / (1.0 + ytm)
}

/// Bond convexity.
pub fn bond_convexity(par: f64, coupon_rate: f64, ytm: f64, periods: usize) -> f64 {
    let coupon = par * coupon_rate;
    let price = bond_price(par, coupon_rate, ytm, periods);
    let mut cv = 0.0;
    for t in 1..=periods {
        cv += t as f64 * (t as f64 + 1.0) * coupon / (1.0 + ytm).powi(t as i32 + 2);
    }
    cv += periods as f64 * (periods as f64 + 1.0) * par / (1.0 + ytm).powi(periods as i32 + 2);
    cv / price
}

/// Yield to maturity (Newton-Raphson).
pub fn yield_to_maturity(price: f64, par: f64, coupon_rate: f64, periods: usize) -> f64 {
    let mut ytm = coupon_rate;
    for _ in 0..200 {
        let p = bond_price(par, coupon_rate, ytm, periods);
        let dp = -modified_duration(par, coupon_rate, ytm, periods) * p;
        if dp.abs() < 1e-15 { break; }
        let new_ytm = ytm - (p - price) / dp;
        if (new_ytm - ytm).abs() < 1e-10 { return new_ytm; }
        ytm = new_ytm.max(0.0001);
    }
    ytm
}

// ═══════════════════════════════════════════════════════════════════
// RISK METRICS
// ═══════════════════════════════════════════════════════════════════

/// Value at Risk (historical simulation).
/// Returns the loss at the given confidence level (e.g., 0.95 for 95% VaR).
pub fn var_historical(returns: &[f64], confidence: f64) -> f64 {
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
    -sorted[idx.min(sorted.len() - 1)]
}

/// Conditional VaR (Expected Shortfall / CVaR).
pub fn cvar(returns: &[f64], confidence: f64) -> f64 {
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cutoff = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
    let tail: f64 = sorted[..=cutoff.min(sorted.len() - 1)].iter().sum();
    -(tail / (cutoff + 1) as f64)
}

/// Sharpe ratio: (mean_return - risk_free) / std_dev.
pub fn sharpe_ratio(returns: &[f64], risk_free: f64) -> f64 {
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let std = (returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    if std < 1e-15 { return 0.0; }
    (mean - risk_free) / std
}

/// Sortino ratio: (mean_return - risk_free) / downside_deviation.
pub fn sortino_ratio(returns: &[f64], risk_free: f64) -> f64 {
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let downside: f64 = returns.iter()
        .filter(|&&r| r < risk_free)
        .map(|r| (r - risk_free).powi(2))
        .sum();
    let dd = (downside / n).sqrt();
    if dd < 1e-15 { return 0.0; }
    (mean - risk_free) / dd
}

/// Maximum drawdown.
pub fn max_drawdown(prices: &[f64]) -> f64 {
    let mut peak = prices[0];
    let mut max_dd = 0.0;
    for &p in prices {
        if p > peak { peak = p; }
        let dd = (peak - p) / peak;
        if dd > max_dd { max_dd = dd; }
    }
    max_dd
}

/// Calmar ratio: annualized return / max drawdown.
pub fn calmar_ratio(prices: &[f64], trading_days: f64) -> f64 {
    let n = prices.len();
    if n < 2 { return 0.0; }
    let total_return = prices[n - 1] / prices[0] - 1.0;
    let years = n as f64 / trading_days;
    let annual_return = (1.0 + total_return).powf(1.0 / years) - 1.0;
    let mdd = max_drawdown(prices);
    if mdd < 1e-15 { return 0.0; }
    annual_return / mdd
}

/// Beta: covariance(asset, market) / variance(market).
pub fn beta(asset_returns: &[f64], market_returns: &[f64]) -> f64 {
    let n = asset_returns.len() as f64;
    let ma = asset_returns.iter().sum::<f64>() / n;
    let mm = market_returns.iter().sum::<f64>() / n;
    let cov: f64 = asset_returns.iter().zip(market_returns).map(|(a, m)| (a - ma) * (m - mm)).sum();
    let var_m: f64 = market_returns.iter().map(|m| (m - mm).powi(2)).sum();
    if var_m < 1e-15 { return 0.0; }
    cov / var_m
}

/// Alpha: excess return not explained by beta.
pub fn alpha(asset_returns: &[f64], market_returns: &[f64], risk_free: f64) -> f64 {
    let n = asset_returns.len() as f64;
    let ma = asset_returns.iter().sum::<f64>() / n;
    let mm = market_returns.iter().sum::<f64>() / n;
    let b = beta(asset_returns, market_returns);
    ma - (risk_free + b * (mm - risk_free))
}

/// Information ratio.
pub fn information_ratio(asset_returns: &[f64], benchmark_returns: &[f64]) -> f64 {
    let excess: Vec<f64> = asset_returns.iter().zip(benchmark_returns).map(|(a, b)| a - b).collect();
    let n = excess.len() as f64;
    let mean = excess.iter().sum::<f64>() / n;
    let std = (excess.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    if std < 1e-15 { return 0.0; }
    mean / std
}

// ═══════════════════════════════════════════════════════════════════
// TECHNICAL INDICATORS
// ═══════════════════════════════════════════════════════════════════

/// Relative Strength Index (RSI).
pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    if n < period + 1 { return vec![]; }
    let changes: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();

    let mut result = vec![f64::NAN; period];
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;

    for i in 0..period {
        if changes[i] > 0.0 { avg_gain += changes[i]; }
        else { avg_loss -= changes[i]; }
    }
    avg_gain /= period as f64;
    avg_loss /= period as f64;

    let rs = if avg_loss < 1e-15 { 100.0 } else { avg_gain / avg_loss };
    result.push(100.0 - 100.0 / (1.0 + rs));

    for i in period..changes.len() {
        let gain = if changes[i] > 0.0 { changes[i] } else { 0.0 };
        let loss = if changes[i] < 0.0 { -changes[i] } else { 0.0 };
        avg_gain = (avg_gain * (period - 1) as f64 + gain) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + loss) / period as f64;
        let rs = if avg_loss < 1e-15 { 100.0 } else { avg_gain / avg_loss };
        result.push(100.0 - 100.0 / (1.0 + rs));
    }
    result
}

/// MACD (Moving Average Convergence Divergence).
/// Returns (macd_line, signal_line, histogram).
pub fn macd(prices: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ema_fast = ema(prices, fast);
    let ema_slow = ema(prices, slow);
    let macd_line: Vec<f64> = ema_fast.iter().zip(&ema_slow).map(|(f, s)| f - s).collect();
    let signal_line = ema(&macd_line, signal);
    let hist: Vec<f64> = macd_line.iter().zip(&signal_line).map(|(m, s)| m - s).collect();
    (macd_line, signal_line, hist)
}

/// Exponential Moving Average.
pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
    let alpha = 2.0 / (period + 1) as f64;
    let mut result = Vec::with_capacity(data.len());
    if data.is_empty() { return result; }
    result.push(data[0]);
    for i in 1..data.len() {
        result.push(alpha * data[i] + (1.0 - alpha) * result[i - 1]);
    }
    result
}

/// Simple Moving Average.
pub fn sma(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period { return vec![]; }
    let mut result = vec![f64::NAN; period - 1];
    let mut sum: f64 = data[..period].iter().sum();
    result.push(sum / period as f64);
    for i in period..data.len() {
        sum += data[i] - data[i - period];
        result.push(sum / period as f64);
    }
    result
}

/// Average True Range (ATR).
pub fn atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = high.len();
    if n < 2 { return vec![]; }
    let mut tr = vec![high[0] - low[0]];
    for i in 1..n {
        let t = (high[i] - low[i])
            .max((high[i] - close[i - 1]).abs())
            .max((low[i] - close[i - 1]).abs());
        tr.push(t);
    }
    ema(&tr, period)
}

/// Stochastic Oscillator (%K, %D).
pub fn stochastic(high: &[f64], low: &[f64], close: &[f64], k_period: usize, d_period: usize) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    if n < k_period { return (vec![], vec![]); }
    let mut k_line = vec![f64::NAN; k_period - 1];
    for i in k_period - 1..n {
        let highest = high[i + 1 - k_period..=i].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lowest = low[i + 1 - k_period..=i].iter().copied().fold(f64::INFINITY, f64::min);
        let range = highest - lowest;
        k_line.push(if range < 1e-15 { 50.0 } else { (close[i] - lowest) / range * 100.0 });
    }
    let d_line = sma(&k_line.iter().filter(|v| !v.is_nan()).copied().collect::<Vec<f64>>(), d_period);
    (k_line, d_line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_future_present_value() {
        let fv = future_value(1000.0, 0.05, 10.0);
        let pv = present_value(fv, 0.05, 10.0);
        assert!((pv - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_npv() {
        let cf = vec![-1000.0, 500.0, 400.0, 300.0];
        let v = npv(0.05, &cf);
        assert!(v > 0.0); // positive NPV at 5%
    }

    #[test]
    fn test_irr() {
        let cf = vec![-1000.0, 500.0, 400.0, 300.0];
        let r = irr(&cf).unwrap();
        // NPV at IRR should be ~0
        assert!(npv(r, &cf).abs() < 0.01);
    }

    #[test]
    fn test_annuity() {
        let pmt = annuity_payment(100000.0, 0.005, 360.0); // 30-year mortgage at 0.5% monthly
        assert!(pmt > 0.0 && pmt < 1000.0);
    }

    #[test]
    fn test_bs_put_call_parity() {
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.2;
        let call = bs_call(s, k, t, r, sigma);
        let put = bs_put(s, k, t, r, sigma);
        // Put-call parity: C - P = S - K*e^{-rT}
        let parity = call - put - (s - k * (-r * t).exp());
        assert!(parity.abs() < 1e-8);
    }

    #[test]
    fn test_implied_vol() {
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.25;
        let price = bs_call(s, k, t, r, sigma);
        let iv = implied_volatility_call(price, s, k, t, r).unwrap();
        assert!((iv - sigma).abs() < 1e-4);
    }

    #[test]
    fn test_bond_price() {
        let p = bond_price(1000.0, 0.05, 0.05, 10);
        assert!((p - 1000.0).abs() < 0.01); // at par when coupon = yield
    }

    #[test]
    fn test_sharpe() {
        let returns = vec![0.01, 0.02, -0.01, 0.03, 0.01];
        let sr = sharpe_ratio(&returns, 0.0);
        assert!(sr > 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        let prices = vec![100.0, 110.0, 105.0, 120.0, 90.0, 100.0];
        let mdd = max_drawdown(&prices);
        assert!((mdd - 0.25).abs() < 1e-10); // 120 → 90 = 25%
    }

    #[test]
    fn test_rsi() {
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let r = rsi(&prices, 14);
        // period NaN + 1 initial + (changes.len()-period) = changes.len() + 1 = prices.len()
        // Wait, let's just check it runs and has valid values
        assert!(r.len() >= 14);
        assert!(r.last().unwrap().is_finite());
    }

    #[test]
    fn test_ema() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let e = ema(&data, 3);
        assert_eq!(e.len(), 5);
    }

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = sma(&data, 3);
        assert_eq!(s.len(), 5);
        assert!((s[2] - 2.0).abs() < 1e-10); // (1+2+3)/3
    }
}
