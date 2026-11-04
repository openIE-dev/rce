//! Time series analysis: autocorrelation, ARIMA, exponential smoothing,
//! seasonal decomposition, stationarity tests, change point detection,
//! forecasting, resampling, lag features, differencing.

// ═══════════════════════════════════════════════════════════════════
// AUTOCORRELATION & CROSS-CORRELATION
// ═════════════════════════════════��════════════════════════════════��

/// Autocorrelation function (ACF) for lags 0..max_lag.
pub fn acf(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();
    if var < 1e-15 { return vec![1.0; max_lag + 1]; }
    (0..=max_lag).map(|lag| {
        let cov: f64 = (0..n - lag).map(|i| (data[i] - mean) * (data[i + lag] - mean)).sum();
        cov / var
    }).collect()
}

/// Partial autocorrelation function (PACF) using Levinson-Durbin.
pub fn pacf(data: &[f64], max_lag: usize) -> Vec<f64> {
    let acf_vals = acf(data, max_lag);
    let mut result = vec![1.0];
    if max_lag == 0 { return result; }

    let mut phi = vec![vec![0.0; max_lag + 1]; max_lag + 1];
    phi[1][1] = acf_vals[1];
    result.push(acf_vals[1]);

    for k in 2..=max_lag {
        let num: f64 = acf_vals[k] - (1..k).map(|j| phi[k-1][j] * acf_vals[k - j]).sum::<f64>();
        let den: f64 = 1.0 - (1..k).map(|j| phi[k-1][j] * acf_vals[j]).sum::<f64>();
        phi[k][k] = num / den;
        for j in 1..k {
            phi[k][j] = phi[k-1][j] - phi[k][k] * phi[k-1][k - j];
        }
        result.push(phi[k][k]);
    }
    result
}

/// Cross-correlation between two signals at given lags.
pub fn cross_correlation(x: &[f64], y: &[f64], max_lag: usize) -> Vec<(i64, f64)> {
    let nx = x.len();
    let ny = y.len();
    let mx = x.iter().sum::<f64>() / nx as f64;
    let my = y.iter().sum::<f64>() / ny as f64;
    let sx = (x.iter().map(|v| (v - mx).powi(2)).sum::<f64>()).sqrt();
    let sy = (y.iter().map(|v| (v - my).powi(2)).sum::<f64>()).sqrt();
    let norm = sx * sy;
    if norm < 1e-15 { return vec![]; }

    let mut result = Vec::new();
    let max_l = max_lag.min(nx.max(ny) - 1);
    for lag in -(max_l as i64)..=(max_l as i64) {
        let mut sum = 0.0;
        let n = nx.min(ny);
        for i in 0..n {
            let j = i as i64 + lag;
            if j >= 0 && (j as usize) < ny {
                sum += (x[i] - mx) * (y[j as usize] - my);
            }
        }
        result.push((lag, sum / norm));
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// DIFFERENCING
// ═══════════════════════════════════════════════════════════════════

/// First difference: y[i] = x[i+1] - x[i].
pub fn diff(data: &[f64]) -> Vec<f64> {
    data.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Nth order differencing.
pub fn diff_n(data: &[f64], n: usize) -> Vec<f64> {
    let mut result = data.to_vec();
    for _ in 0..n {
        result = diff(&result);
    }
    result
}

/// Seasonal differencing: y[i] = x[i] - x[i - period].
pub fn seasonal_diff(data: &[f64], period: usize) -> Vec<f64> {
    if period >= data.len() { return vec![]; }
    (period..data.len()).map(|i| data[i] - data[i - period]).collect()
}

/// Cumulative sum (inverse of diff).
pub fn cumsum(data: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    let mut sum = 0.0;
    for &v in data {
        sum += v;
        result.push(sum);
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// EXPONENTIAL SMOOTHING
// ═══════════════════════════════════════════════════════════════════

/// Simple Exponential Smoothing (SES).
pub fn ses(data: &[f64], alpha: f64) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    if data.is_empty() { return result; }
    result.push(data[0]);
    for i in 1..data.len() {
        let prev = result[i - 1];
        result.push(alpha * data[i] + (1.0 - alpha) * prev);
    }
    result
}

/// Double Exponential Smoothing (Holt's method).
pub fn holt(data: &[f64], alpha: f64, beta: f64) -> Vec<f64> {
    if data.len() < 2 { return data.to_vec(); }
    let mut level = data[0];
    let mut trend = data[1] - data[0];
    let mut result = vec![level];
    for i in 1..data.len() {
        let new_level = alpha * data[i] + (1.0 - alpha) * (level + trend);
        let new_trend = beta * (new_level - level) + (1.0 - beta) * trend;
        level = new_level;
        trend = new_trend;
        result.push(level + trend);
    }
    result
}

/// Triple Exponential Smoothing (Holt-Winters, additive).
pub fn holt_winters(data: &[f64], alpha: f64, beta: f64, gamma: f64, period: usize) -> Vec<f64> {
    let n = data.len();
    if n < 2 * period { return data.to_vec(); }

    // Initialize: level, trend, seasonal
    let mut seasonal = vec![0.0; period];
    let first_season_mean: f64 = data[..period].iter().sum::<f64>() / period as f64;
    for i in 0..period {
        seasonal[i] = data[i] - first_season_mean;
    }
    let mut level = first_season_mean;
    let mut trend = (data[period..2*period].iter().sum::<f64>() - data[..period].iter().sum::<f64>()) / (period * period) as f64;
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let si = i % period;
        if i == 0 {
            result.push(level + trend + seasonal[si]);
            continue;
        }
        let new_level = alpha * (data[i] - seasonal[si]) + (1.0 - alpha) * (level + trend);
        let new_trend = beta * (new_level - level) + (1.0 - beta) * trend;
        seasonal[si] = gamma * (data[i] - new_level) + (1.0 - gamma) * seasonal[si];
        level = new_level;
        trend = new_trend;
        result.push(level + trend + seasonal[(i + 1) % period]);
    }
    result
}

// ═════════════════════════════════════════════��═════════════════════
// MOVING STATISTICS
// ═══════════════════════════════════════════════════════════════════

/// Exponentially weighted moving average (EWMA).
pub fn ewma(data: &[f64], span: f64) -> Vec<f64> {
    let alpha = 2.0 / (span + 1.0);
    ses(data, alpha)
}

/// Rolling window function (generic).
pub fn rolling_apply(data: &[f64], window: usize, f: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    if window > data.len() { return vec![]; }
    (0..=data.len() - window).map(|i| f(&data[i..i + window])).collect()
}

/// Rolling min.
pub fn rolling_min(data: &[f64], window: usize) -> Vec<f64> {
    rolling_apply(data, window, |w| w.iter().copied().fold(f64::INFINITY, f64::min))
}

/// Rolling max.
pub fn rolling_max(data: &[f64], window: usize) -> Vec<f64> {
    rolling_apply(data, window, |w| w.iter().copied().fold(f64::NEG_INFINITY, f64::max))
}

/// Rolling median.
pub fn rolling_median(data: &[f64], window: usize) -> Vec<f64> {
    rolling_apply(data, window, |w| {
        let mut sorted = w.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        if n % 2 == 0 { (sorted[n/2 - 1] + sorted[n/2]) / 2.0 } else { sorted[n/2] }
    })
}

/// Rolling variance.
pub fn rolling_var(data: &[f64], window: usize) -> Vec<f64> {
    rolling_apply(data, window, |w| {
        let mean = w.iter().sum::<f64>() / w.len() as f64;
        w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (w.len() - 1) as f64
    })
}

/// Rolling z-score.
pub fn rolling_zscore(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    if window > n { return vec![]; }
    let mut result = Vec::with_capacity(n - window + 1);
    for i in 0..=n - window {
        let w = &data[i..i + window];
        let mean = w.iter().sum::<f64>() / window as f64;
        let std = (w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (window - 1) as f64).sqrt();
        if std < 1e-15 { result.push(0.0); }
        else { result.push((data[i + window - 1] - mean) / std); }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// SEASONAL DECOMPOSITION
// ═══════════════════════════════��════════════════════════════════���══

/// Additive seasonal decomposition: data = trend + seasonal + residual.
/// Returns (trend, seasonal, residual).
pub fn seasonal_decompose(data: &[f64], period: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = data.len();
    // Trend: centered moving average
    let mut trend = vec![f64::NAN; n];
    let half = period / 2;
    for i in half..n.saturating_sub(half) {
        let start = if i >= half { i - half } else { 0 };
        let end = (i + half + 1).min(n);
        let sum: f64 = data[start..end].iter().sum();
        trend[i] = sum / (end - start) as f64;
    }
    // Detrended
    let detrended: Vec<f64> = data.iter().zip(&trend).map(|(d, t)| if t.is_nan() { 0.0 } else { d - t }).collect();
    // Seasonal: average of detrended values at each phase
    let mut seasonal = vec![0.0; n];
    for phase in 0..period {
        let vals: Vec<f64> = (phase..n).step_by(period)
            .filter(|&i| !trend[i].is_nan())
            .map(|i| detrended[i]).collect();
        let avg = if vals.is_empty() { 0.0 } else { vals.iter().sum::<f64>() / vals.len() as f64 };
        for i in (phase..n).step_by(period) { seasonal[i] = avg; }
    }
    // Residual
    let residual: Vec<f64> = (0..n).map(|i| {
        if trend[i].is_nan() { 0.0 } else { data[i] - trend[i] - seasonal[i] }
    }).collect();
    (trend, seasonal, residual)
}

// ══════════════════════════════════════════════════════���════════════
// LAG FEATURES
// ═════════════════════════════════��════════════════════════════════���

/// Create lag feature: shift data by k positions (NaN-filled).
pub fn lag(data: &[f64], k: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; data.len()];
    for i in k..data.len() { result[i] = data[i - k]; }
    result
}

/// Create lead feature: shift data by k positions forward.
pub fn lead(data: &[f64], k: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; data.len()];
    for i in 0..data.len().saturating_sub(k) { result[i] = data[i + k]; }
    result
}

/// Percent change: (x[i] - x[i-k]) / x[i-k].
pub fn pct_change(data: &[f64], k: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; data.len()];
    for i in k..data.len() {
        if data[i - k].abs() > 1e-15 {
            result[i] = (data[i] - data[i - k]) / data[i - k];
        }
    }
    result
}

/// Log return: ln(x[i] / x[i-1]).
pub fn log_returns(data: &[f64]) -> Vec<f64> {
    let mut result = vec![f64::NAN];
    for i in 1..data.len() {
        if data[i - 1] > 0.0 && data[i] > 0.0 {
            result.push((data[i] / data[i - 1]).ln());
        } else {
            result.push(f64::NAN);
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// STATIONARITY & CHANGE POINT
// ═══════════════════════════════════════════════��═══════════════════

/// Augmented Dickey-Fuller test statistic (simplified).
/// Tests H0: unit root (non-stationary). More negative = more stationary.
pub fn adf_statistic(data: &[f64]) -> f64 {
    let dy = diff(data);
    let n = dy.len();
    if n < 3 { return 0.0; }
    // Simple regression: Δy_t = α + β·y_{t-1} + ε_t
    let y_lag: Vec<f64> = data[..n].to_vec();
    let mx = y_lag.iter().sum::<f64>() / n as f64;
    let my = dy.iter().sum::<f64>() / n as f64;
    let sxy: f64 = y_lag.iter().zip(&dy).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = y_lag.iter().map(|x| (x - mx).powi(2)).sum();
    if sxx < 1e-15 { return 0.0; }
    let beta = sxy / sxx;
    let alpha = my - beta * mx;
    let residuals: Vec<f64> = dy.iter().zip(&y_lag).map(|(d, y)| d - alpha - beta * y).collect();
    let sse: f64 = residuals.iter().map(|r| r * r).sum();
    let se_beta = (sse / (n as f64 - 2.0) / sxx).sqrt();
    if se_beta < 1e-15 { return 0.0; }
    beta / se_beta
}

/// CUSUM (cumulative sum) for change point detection.
/// Returns cumulative sum of deviations from mean.
pub fn cusum(data: &[f64]) -> Vec<f64> {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let mut cs = vec![0.0];
    for &v in data {
        cs.push(cs.last().unwrap() + (v - mean));
    }
    cs
}

/// Detect change point using CUSUM. Returns index of maximum absolute deviation.
pub fn cusum_change_point(data: &[f64]) -> usize {
    let cs = cusum(data);
    cs.iter().enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Bollinger Bands: (middle, upper, lower).
pub fn bollinger_bands(data: &[f64], window: usize, num_std: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = data.len();
    let pad = window - 1;
    let mut middle = vec![f64::NAN; pad];
    let mut upper = vec![f64::NAN; pad];
    let mut lower = vec![f64::NAN; pad];

    for i in 0..=n - window {
        let w = &data[i..i + window];
        let mean = w.iter().sum::<f64>() / window as f64;
        let std = (w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64).sqrt();
        middle.push(mean);
        upper.push(mean + num_std * std);
        lower.push(mean - num_std * std);
    }
    (middle, upper, lower)
}

// ═══════════════════════════════════════════════════════════════════
// RESAMPLING
// ════════════════════════════════════��═════════════════════════════��

/// Downsample by factor (take every nth sample).
pub fn downsample(data: &[f64], factor: usize) -> Vec<f64> {
    data.iter().step_by(factor).copied().collect()
}

/// Upsample by factor (zero-insert between samples).
pub fn upsample(data: &[f64], factor: usize) -> Vec<f64> {
    let mut result = vec![0.0; data.len() * factor];
    for (i, &v) in data.iter().enumerate() {
        result[i * factor] = v;
    }
    result
}

/// Upsample with linear interpolation.
pub fn upsample_linear(data: &[f64], factor: usize) -> Vec<f64> {
    if data.len() < 2 { return data.to_vec(); }
    let new_len = (data.len() - 1) * factor + 1;
    let mut result = Vec::with_capacity(new_len);
    for i in 0..data.len() - 1 {
        for j in 0..factor {
            let t = j as f64 / factor as f64;
            result.push(data[i] * (1.0 - t) + data[i + 1] * t);
        }
    }
    result.push(*data.last().unwrap());
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_acf() {
        // Sine wave should have periodic ACF
        let data: Vec<f64> = (0..100).map(|i| (2.0 * PI * i as f64 / 20.0).sin()).collect();
        let a = acf(&data, 20);
        assert!((a[0] - 1.0).abs() < 1e-10); // lag 0 = 1
        assert!(a[20] > 0.8); // period = 20, ACF near 1
    }

    #[test]
    fn test_diff() {
        assert_eq!(diff(&[1.0, 3.0, 6.0, 10.0]), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_cumsum() {
        assert_eq!(cumsum(&[1.0, 2.0, 3.0]), vec![1.0, 3.0, 6.0]);
    }

    #[test]
    fn test_ses() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let smoothed = ses(&data, 0.5);
        assert_eq!(smoothed.len(), 5);
        assert_eq!(smoothed[0], 1.0);
        assert!(smoothed[4] > 3.0 && smoothed[4] < 5.0);
    }

    #[test]
    fn test_ewma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ewma(&data, 3.0);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_rolling_median() {
        let data = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let med = rolling_median(&data, 3);
        assert_eq!(med.len(), 3);
        assert_eq!(med[0], 2.0); // median of [1,3,2]
    }

    #[test]
    fn test_seasonal_decompose() {
        // Trend + seasonal
        let data: Vec<f64> = (0..40).map(|i| {
            let trend = i as f64 * 0.5;
            let seasonal = (2.0 * PI * i as f64 / 4.0).sin();
            trend + seasonal
        }).collect();
        let (trend, seasonal, _residual) = seasonal_decompose(&data, 4);
        assert_eq!(trend.len(), 40);
        assert_eq!(seasonal.len(), 40);
    }

    #[test]
    fn test_lag_lead() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let lagged = lag(&data, 2);
        assert!(lagged[0].is_nan());
        assert!(lagged[1].is_nan());
        assert_eq!(lagged[2], 1.0);

        let led = lead(&data, 2);
        assert_eq!(led[0], 3.0);
        assert!(led[3].is_nan());
    }

    #[test]
    fn test_bollinger_bands() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0];
        let (mid, upper, lower) = bollinger_bands(&data, 3, 2.0);
        assert_eq!(mid.len(), data.len());
        // Upper should be above middle, lower below
        for i in 2..mid.len() {
            assert!(upper[i] > mid[i]);
            assert!(lower[i] < mid[i]);
        }
    }

    #[test]
    fn test_downsample() {
        assert_eq!(downsample(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2), vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_cusum() {
        // Constant data: cusum should stay near 0
        let data = vec![5.0; 10];
        let cs = cusum(&data);
        assert!((cs.last().unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_pct_change() {
        let data = vec![100.0, 110.0, 105.0];
        let pc = pct_change(&data, 1);
        assert!((pc[1] - 0.1).abs() < 1e-10); // 10% increase
    }
}
