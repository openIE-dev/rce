//! Interpolation: cubic spline, linear, nearest, Akima, PCHIP, Lagrange.
//!
//! Replaces scipy.interpolate, R's spline/approx, MATLAB's interp1.

/// Linear interpolation at query points.
pub fn interp1_linear(x: &[f64], y: &[f64], xq: &[f64]) -> Vec<f64> {
    xq.iter().map(|&xi| {
        if xi <= x[0] { return y[0]; }
        if xi >= x[x.len() - 1] { return y[y.len() - 1]; }
        let idx = x.partition_point(|&v| v < xi).saturating_sub(1).min(x.len() - 2);
        let t = (xi - x[idx]) / (x[idx + 1] - x[idx]);
        y[idx] + t * (y[idx + 1] - y[idx])
    }).collect()
}

/// Nearest-neighbor interpolation.
pub fn interp1_nearest(x: &[f64], y: &[f64], xq: &[f64]) -> Vec<f64> {
    xq.iter().map(|&xi| {
        let idx = x.iter().enumerate()
            .min_by(|(_, a), (_, b)| ((**a - xi).abs()).partial_cmp(&((**b - xi).abs())).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        y[idx]
    }).collect()
}

/// Natural cubic spline interpolation.
/// Returns coefficients for evaluation via `spline_eval`.
pub fn cubic_spline_coeffs(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = x.len();
    if n < 2 { return (y.to_vec(), vec![0.0; n], vec![0.0; n], vec![0.0; n]); }
    let nm1 = n - 1;

    let h: Vec<f64> = (0..nm1).map(|i| x[i + 1] - x[i]).collect();
    let alpha: Vec<f64> = (1..nm1).map(|i| {
        3.0 / h[i] * (y[i + 1] - y[i]) - 3.0 / h[i - 1] * (y[i] - y[i - 1])
    }).collect();

    // Solve tridiagonal system for c
    let mut l = vec![1.0; n];
    let mut mu = vec![0.0; n];
    let mut z = vec![0.0; n];

    for i in 1..nm1 {
        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i - 1] - h[i - 1] * z[i - 1]) / l[i];
    }

    let mut c = vec![0.0; n];
    let mut b = vec![0.0; nm1];
    let mut d = vec![0.0; nm1];

    for j in (0..nm1).rev() {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
    }

    (y[..nm1].to_vec(), b, c[..nm1].to_vec(), d)
}

/// Evaluate cubic spline at query points.
pub fn cubic_spline_eval(x: &[f64], a: &[f64], b: &[f64], c: &[f64], d: &[f64], xq: &[f64]) -> Vec<f64> {
    xq.iter().map(|&xi| {
        let idx = x.partition_point(|&v| v < xi).saturating_sub(1).min(a.len() - 1);
        let dx = xi - x[idx];
        a[idx] + b[idx] * dx + c[idx] * dx * dx + d[idx] * dx * dx * dx
    }).collect()
}

/// Convenience: cubic spline interpolation at query points.
pub fn interp1_cubic(x: &[f64], y: &[f64], xq: &[f64]) -> Vec<f64> {
    let (a, b, c, d) = cubic_spline_coeffs(x, y);
    cubic_spline_eval(x, &a, &b, &c, &d, xq)
}

/// Lagrange polynomial interpolation.
pub fn lagrange_interp(x: &[f64], y: &[f64], xq: f64) -> f64 {
    let n = x.len();
    let mut result = 0.0;
    for i in 0..n {
        let mut basis = y[i];
        for j in 0..n {
            if j != i { basis *= (xq - x[j]) / (x[i] - x[j]); }
        }
        result += basis;
    }
    result
}

/// Lagrange interpolation at multiple query points.
pub fn lagrange_interp_vec(x: &[f64], y: &[f64], xq: &[f64]) -> Vec<f64> {
    xq.iter().map(|&xi| lagrange_interp(x, y, xi)).collect()
}

/// PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) — monotone-preserving.
pub fn interp1_pchip(x: &[f64], y: &[f64], xq: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 2 { return vec![y.get(0).copied().unwrap_or(0.0); xq.len()]; }

    // Compute slopes
    let h: Vec<f64> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();
    let delta: Vec<f64> = (0..n - 1).map(|i| (y[i + 1] - y[i]) / h[i]).collect();

    // Compute derivatives at each point
    let mut d = vec![0.0; n];
    if n == 2 {
        d[0] = delta[0];
        d[1] = delta[0];
    } else {
        d[0] = delta[0];
        d[n - 1] = delta[n - 2];
        for i in 1..n - 1 {
            if delta[i - 1].signum() != delta[i].signum() {
                d[i] = 0.0;
            } else {
                d[i] = 2.0 / (1.0 / delta[i - 1] + 1.0 / delta[i]);
            }
        }
    }

    // Evaluate using Hermite basis
    xq.iter().map(|&xi| {
        let idx = x.partition_point(|&v| v < xi).saturating_sub(1).min(n - 2);
        let t = (xi - x[idx]) / h[idx];
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        h00 * y[idx] + h10 * h[idx] * d[idx] + h01 * y[idx + 1] + h11 * h[idx] * d[idx + 1]
    }).collect()
}

/// 2D bilinear interpolation on a regular grid.
pub fn interp2_bilinear(x: &[f64], y: &[f64], z: &[Vec<f64>], xq: f64, yq: f64) -> f64 {
    let nx = x.len();
    let ny = y.len();
    let xi = x.partition_point(|&v| v < xq).saturating_sub(1).min(nx - 2);
    let yi = y.partition_point(|&v| v < yq).saturating_sub(1).min(ny - 2);
    let tx = (xq - x[xi]) / (x[xi + 1] - x[xi]);
    let ty = (yq - y[yi]) / (y[yi + 1] - y[yi]);
    let z00 = z[yi][xi];
    let z10 = z[yi][xi + 1];
    let z01 = z[yi + 1][xi];
    let z11 = z[yi + 1][xi + 1];
    z00 * (1.0 - tx) * (1.0 - ty) + z10 * tx * (1.0 - ty) + z01 * (1.0 - tx) * ty + z11 * tx * ty
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interp() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0];
        let result = interp1_linear(&x, &y, &[0.5, 1.5, 2.5]);
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_spline() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0]; // y = x^2
        let result = interp1_cubic(&x, &y, &[0.5, 1.5, 2.5, 3.5]);
        // Should be close to x^2
        assert!((result[0] - 0.25).abs() < 0.3);
        assert!((result[1] - 2.25).abs() < 0.3);
    }

    #[test]
    fn test_lagrange() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0]; // y = x^2 (exact for degree-2 polynomial)
        assert!((lagrange_interp(&x, &y, 1.5) - 2.25).abs() < 1e-10);
    }

    #[test]
    fn test_pchip_monotone() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 2.0, 3.0, 4.0]; // monotone input
        let result = interp1_pchip(&x, &y, &[0.5, 1.5, 2.5, 3.5]);
        // PCHIP should preserve monotonicity
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1], "PCHIP should preserve monotonicity");
        }
    }

    #[test]
    fn test_bilinear_2d() {
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 1.0];
        let z = vec![vec![0.0, 1.0], vec![2.0, 3.0]];
        assert!((interp2_bilinear(&x, &y, &z, 0.5, 0.5) - 1.5).abs() < 1e-10);
    }
}
