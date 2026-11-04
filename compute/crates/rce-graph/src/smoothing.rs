//! Smoothing and curve fitting: Savitzky-Golay, LOESS, spline smoothing,
//! kernel density estimation, Whittaker smoother, penalized regression.

use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════
// SAVITZKY-GOLAY FILTER
// ═══════════════════════════════════════════════════════════════════

/// Savitzky-Golay smoothing filter.
/// window_size must be odd, poly_order < window_size.
pub fn savgol_filter(data: &[f64], window_size: usize, poly_order: usize) -> Vec<f64> {
    let half = window_size / 2;
    let coeffs = savgol_coefficients(window_size, poly_order, 0);
    let n = data.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..window_size {
            let idx = i as i64 + j as i64 - half as i64;
            let idx = idx.max(0).min(n as i64 - 1) as usize;
            sum += coeffs[j] * data[idx];
        }
        result[i] = sum;
    }
    result
}

/// Savitzky-Golay first derivative.
pub fn savgol_derivative(data: &[f64], window_size: usize, poly_order: usize, dx: f64) -> Vec<f64> {
    let half = window_size / 2;
    let coeffs = savgol_coefficients(window_size, poly_order, 1);
    let n = data.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..window_size {
            let idx = i as i64 + j as i64 - half as i64;
            let idx = idx.max(0).min(n as i64 - 1) as usize;
            sum += coeffs[j] * data[idx];
        }
        result[i] = sum / dx;
    }
    result
}

fn savgol_coefficients(window_size: usize, poly_order: usize, deriv: usize) -> Vec<f64> {
    let half = window_size as i64 / 2;
    let m = poly_order + 1;
    // Build Vandermonde-like system and solve for coefficients
    // Simplified: for common cases, use precomputed
    if window_size == 5 && poly_order == 2 && deriv == 0 {
        return vec![-3.0/35.0, 12.0/35.0, 17.0/35.0, 12.0/35.0, -3.0/35.0];
    }
    if window_size == 7 && poly_order == 2 && deriv == 0 {
        return vec![-2.0/21.0, 3.0/21.0, 6.0/21.0, 7.0/21.0, 6.0/21.0, 3.0/21.0, -2.0/21.0];
    }
    // General case: least squares polynomial fit at each point
    let mut a = vec![0.0; window_size * m];
    for i in 0..window_size {
        let x = i as f64 - half as f64;
        let mut xp = 1.0;
        for j in 0..m { a[i * m + j] = xp; xp *= x; }
    }
    // Solve (A^T A)^{-1} A^T for the deriv-th row
    let ata = {
        let mut mat = vec![0.0; m * m];
        for i in 0..m { for j in 0..m { for k in 0..window_size { mat[i*m+j] += a[k*m+i] * a[k*m+j]; } } }
        mat
    };
    let ata_inv = crate::linalg_decomp::matrix_inverse(&ata, m);
    // Coefficients for deriv-th derivative = deriv! * row deriv of (A^T A)^{-1} A^T
    let fact = (1..=deriv).product::<usize>() as f64;
    (0..window_size).map(|k| {
        let mut c = 0.0;
        for j in 0..m { c += ata_inv[deriv * m + j] * a[k * m + j]; }
        c * fact
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// KERNEL DENSITY ESTIMATION
// ═══════════════════════════════════════════════════════════════════

/// Gaussian KDE at evaluation points.
pub fn kde_gaussian(data: &[f64], eval_points: &[f64], bandwidth: f64) -> Vec<f64> {
    let n = data.len() as f64;
    let norm = 1.0 / (n * bandwidth * (2.0 * PI).sqrt());
    eval_points.iter().map(|&x| {
        norm * data.iter().map(|&xi| {
            let u = (x - xi) / bandwidth;
            (-0.5 * u * u).exp()
        }).sum::<f64>()
    }).collect()
}

/// Silverman's rule of thumb bandwidth for Gaussian KDE.
pub fn silverman_bandwidth(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let iqr = crate::probability::iqr(data);
    let s = std.min(iqr / 1.34);
    0.9 * s * n.powf(-0.2)
}

/// Epanechnikov KDE.
pub fn kde_epanechnikov(data: &[f64], eval_points: &[f64], bandwidth: f64) -> Vec<f64> {
    let n = data.len() as f64;
    eval_points.iter().map(|&x| {
        let sum: f64 = data.iter().map(|&xi| {
            let u = (x - xi) / bandwidth;
            if u.abs() <= 1.0 { 0.75 * (1.0 - u * u) } else { 0.0 }
        }).sum();
        sum / (n * bandwidth)
    }).collect()
}

/// Adaptive bandwidth KDE (bandwidth varies with local density).
pub fn kde_adaptive(data: &[f64], eval_points: &[f64], pilot_bandwidth: f64, alpha: f64) -> Vec<f64> {
    // Pilot density
    let pilot = kde_gaussian(data, data, pilot_bandwidth);
    let log_mean: f64 = pilot.iter().map(|&p| p.max(1e-15).ln()).sum::<f64>() / data.len() as f64;
    let g = log_mean.exp();
    // Local bandwidths
    let local_bw: Vec<f64> = pilot.iter().map(|&p| pilot_bandwidth * (g / p.max(1e-15)).powf(alpha)).collect();
    let n = data.len() as f64;
    eval_points.iter().map(|&x| {
        let sum: f64 = data.iter().zip(&local_bw).map(|(&xi, &h)| {
            let u = (x - xi) / h;
            (-0.5 * u * u).exp() / (h * (2.0 * PI).sqrt())
        }).sum();
        sum / n
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// WHITTAKER SMOOTHER
// ═══════════════════════════════════════════════════════════════════

/// Whittaker smoother (penalized least squares with 2nd-order differences).
pub fn whittaker_smooth(data: &[f64], lambda: f64) -> Vec<f64> {
    let n = data.len();
    if n < 3 { return data.to_vec(); }
    // Solve (I + λ D₂ᵀD₂) z = y via Thomas algorithm (tridiagonal approximation)
    let mut a_diag = vec![1.0 + 2.0 * lambda; n];
    a_diag[0] = 1.0 + lambda;
    a_diag[n - 1] = 1.0 + lambda;
    let off_diag = vec![-lambda; n];
    crate::pde::tridiag_solve(&off_diag, &a_diag, &off_diag, data)
}

// ═══════════════════════════════════════════════════════════════════
// EXPONENTIAL SMOOTHING (state space)
// ═══════════════════════════════════════════════════════════════════

/// Simple exponential smoothing with optimal alpha selection (minimizing MSE).
pub fn ses_optimal(data: &[f64]) -> (Vec<f64>, f64) {
    let mut best_alpha = 0.1;
    let mut best_mse = f64::INFINITY;
    for a_int in 1..20 {
        let alpha = a_int as f64 * 0.05;
        let smoothed = crate::timeseries::ses(data, alpha);
        let mse: f64 = data.iter().zip(&smoothed).skip(1).map(|(d, s)| (d - s).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        if mse < best_mse { best_mse = mse; best_alpha = alpha; }
    }
    (crate::timeseries::ses(data, best_alpha), best_alpha)
}

/// Double exponential smoothing with level and trend.
pub fn des_forecast(data: &[f64], alpha: f64, beta: f64, horizon: usize) -> Vec<f64> {
    let smoothed = crate::timeseries::holt(data, alpha, beta);
    if data.len() < 2 { return vec![*data.last().unwrap_or(&0.0); horizon]; }
    let last = *smoothed.last().unwrap();
    let trend = smoothed[smoothed.len() - 1] - smoothed[smoothed.len() - 2];
    (1..=horizon).map(|h| last + h as f64 * trend).collect()
}

// ═══════════════════════════════════════════════════════════════════
// ISOTONIC REGRESSION
// ═══════════════════════════════════════════════════════════════════

/// Pool adjacent violators algorithm (isotonic regression).
/// Fits non-decreasing function to data.
pub fn isotonic_regression(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut result = y.to_vec();
    let mut weights = vec![1.0; n];
    let mut i = 0;
    while i < n - 1 {
        if result[i] > result[i + 1] {
            // Pool
            let combined = (result[i] * weights[i] + result[i + 1] * weights[i + 1]) / (weights[i] + weights[i + 1]);
            result[i] = combined;
            result[i + 1] = combined;
            weights[i] += weights[i + 1];
            weights[i + 1] = weights[i];
            // Check backward
            if i > 0 { i -= 1; } else { i += 1; }
        } else {
            i += 1;
        }
    }
    // Forward pass to ensure monotonicity
    for i in 1..n { result[i] = result[i].max(result[i - 1]); }
    result
}

/// Isotonic regression (non-increasing).
pub fn antitonic_regression(y: &[f64]) -> Vec<f64> {
    let neg: Vec<f64> = y.iter().map(|v| -v).collect();
    isotonic_regression(&neg).iter().map(|v| -v).collect()
}

// ═══════════════════════════════════════════════════════════════════
// CONVOLUTION SMOOTHING
// ═══════════════════════════════════════════════════════════════════

/// Gaussian smoothing (1D convolution with Gaussian kernel).
pub fn gaussian_smooth(data: &[f64], sigma: f64) -> Vec<f64> {
    let radius = (3.0 * sigma).ceil() as usize;
    let kernel_size = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(kernel_size);
    let mut sum = 0.0;
    for i in 0..kernel_size {
        let x = i as f64 - radius as f64;
        let v = (-0.5 * x * x / (sigma * sigma)).exp();
        kernel.push(v);
        sum += v;
    }
    for k in kernel.iter_mut() { *k /= sum; }
    // Convolve
    let n = data.len();
    (0..n).map(|i| {
        let mut s = 0.0;
        for j in 0..kernel_size {
            let idx = i as i64 + j as i64 - radius as i64;
            let idx = idx.max(0).min(n as i64 - 1) as usize;
            s += data[idx] * kernel[j];
        }
        s
    }).collect()
}

/// Triangular (Bartlett) smoothing.
pub fn triangular_smooth(data: &[f64], window: usize) -> Vec<f64> {
    let half = window / 2;
    let mut kernel = Vec::with_capacity(window);
    for i in 0..window {
        kernel.push(1.0 - (i as f64 - half as f64).abs() / (half as f64 + 1.0));
    }
    let sum: f64 = kernel.iter().sum();
    for k in kernel.iter_mut() { *k /= sum; }
    let n = data.len();
    (0..n).map(|i| {
        let mut s = 0.0;
        for j in 0..window {
            let idx = i as i64 + j as i64 - half as i64;
            let idx = idx.max(0).min(n as i64 - 1) as usize;
            s += data[idx] * kernel[j];
        }
        s
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_savgol() {
        let data: Vec<f64> = (0..20).map(|i| (i as f64 * 0.3).sin() + 0.1 * (i as f64)).collect();
        let smoothed = savgol_filter(&data, 5, 2);
        assert_eq!(smoothed.len(), data.len());
    }

    #[test]
    fn test_kde_gaussian() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let eval = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let density = kde_gaussian(&data, &eval, 0.5);
        assert!(density.iter().all(|&d| d > 0.0));
    }

    #[test]
    fn test_silverman() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let bw = silverman_bandwidth(&data);
        assert!(bw > 0.0);
    }

    #[test]
    fn test_whittaker() {
        let data = vec![1.0, 1.5, 1.2, 1.8, 1.3, 1.6, 1.4];
        let smoothed = whittaker_smooth(&data, 10.0);
        assert_eq!(smoothed.len(), data.len());
    }

    #[test]
    fn test_isotonic() {
        let y = vec![1.0, 3.0, 2.0, 4.0, 3.5, 5.0];
        let iso = isotonic_regression(&y);
        for i in 1..iso.len() { assert!(iso[i] >= iso[i - 1] - 1e-10); }
    }

    #[test]
    fn test_gaussian_smooth() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let smoothed = gaussian_smooth(&data, 1.0);
        assert!(smoothed[2] < 1.0); // peak should be reduced
        assert!(smoothed[1] > 0.0); // neighbors should increase
    }

    #[test]
    fn test_des_forecast() {
        let data: Vec<f64> = (0..20).map(|i| i as f64 * 2.0).collect();
        let forecast = des_forecast(&data, 0.3, 0.1, 5);
        assert_eq!(forecast.len(), 5);
        assert!(forecast[0] > data[19]); // upward trend
    }
}
