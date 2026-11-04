//! Numerical methods: root finding, numerical integration (quadrature),
//! finite differences, numerical differentiation, Richardson extrapolation,
//! Romberg integration, adaptive Simpson, Gauss quadrature.


// ═══════════════════════════════════════════════════════════════════
// ROOT FINDING
// ═══════════════════════════════════════════════════════════════════

/// Bisection method. Finds root of f in [a, b].
pub fn bisect(f: impl Fn(f64) -> f64, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> Option<f64> {
    let mut fa = f(a);
    if fa * f(b) > 0.0 { return None; }
    for _ in 0..max_iter {
        let mid = (a + b) / 2.0;
        let fm = f(mid);
        if fm.abs() < tol || (b - a) / 2.0 < tol { return Some(mid); }
        if fa * fm < 0.0 { b = mid; } else { a = mid; fa = fm; }
    }
    Some((a + b) / 2.0)
}

/// Newton-Raphson method. f and f' (derivative).
pub fn newton(f: impl Fn(f64) -> f64, fp: impl Fn(f64) -> f64, mut x: f64, tol: f64, max_iter: usize) -> Option<f64> {
    for _ in 0..max_iter {
        let fx = f(x);
        let fpx = fp(x);
        if fpx.abs() < 1e-15 { return None; }
        let x_new = x - fx / fpx;
        if (x_new - x).abs() < tol { return Some(x_new); }
        x = x_new;
    }
    Some(x)
}

/// Secant method. Two initial guesses x0, x1.
pub fn secant(f: impl Fn(f64) -> f64, mut x0: f64, mut x1: f64, tol: f64, max_iter: usize) -> Option<f64> {
    let mut f0 = f(x0);
    for _ in 0..max_iter {
        let f1 = f(x1);
        if (f1 - f0).abs() < 1e-15 { return None; }
        let x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        if (x2 - x1).abs() < tol { return Some(x2); }
        x0 = x1; f0 = f1; x1 = x2;
    }
    Some(x1)
}

/// Brent's method (combines bisection, secant, inverse quadratic).
pub fn brentq(f: impl Fn(f64) -> f64, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> Option<f64> {
    let mut fa = f(a);
    let mut fb = f(b);
    if fa * fb > 0.0 { return None; }
    if fa.abs() < fb.abs() { std::mem::swap(&mut a, &mut b); std::mem::swap(&mut fa, &mut fb); }
    let mut c = a;
    let mut fc = fa;
    let mut mflag = true;
    let mut d = 0.0;

    for _ in 0..max_iter {
        if fb.abs() < tol { return Some(b); }
        if fa.abs() < tol { return Some(a); }

        let s = if (fa - fc).abs() > 1e-15 && (fb - fc).abs() > 1e-15 {
            // Inverse quadratic interpolation
            a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
        } else {
            // Secant
            b - fb * (b - a) / (fb - fa)
        };

        let cond1 = if a < b { s < (3.0 * a + b) / 4.0 || s > b } else { s > (3.0 * a + b) / 4.0 || s < b };
        let cond2 = mflag && (s - b).abs() >= (b - c).abs() / 2.0;
        let cond3 = !mflag && (s - b).abs() >= (c - d).abs() / 2.0;

        if cond1 || cond2 || cond3 {
            // Fall back to bisection: shadow the secant `s` above with
            // the bisection midpoint.
            let s = (a + b) / 2.0;
            mflag = true;
            let fs = f(s);
            d = c; c = b; fc = fb;
            if fa * fs < 0.0 { b = s; fb = fs; } else { a = s; fa = fs; }
        } else {
            mflag = false;
            let fs = f(s);
            d = c; c = b; fc = fb;
            if fa * fs < 0.0 { b = s; fb = fs; } else { a = s; fa = fs; }
        }

        if fa.abs() < fb.abs() { std::mem::swap(&mut a, &mut b); std::mem::swap(&mut fa, &mut fb); }
    }
    Some(b)
}

/// Fixed-point iteration: x = g(x).
pub fn fixed_point(g: impl Fn(f64) -> f64, mut x: f64, tol: f64, max_iter: usize) -> Option<f64> {
    for _ in 0..max_iter {
        let x_new = g(x);
        if (x_new - x).abs() < tol { return Some(x_new); }
        x = x_new;
    }
    Some(x)
}

/// Halley's method (cubic convergence).
pub fn halley(f: impl Fn(f64) -> f64, fp: impl Fn(f64) -> f64, fpp: impl Fn(f64) -> f64,
              mut x: f64, tol: f64, max_iter: usize) -> Option<f64> {
    for _ in 0..max_iter {
        let fx = f(x);
        let fpx = fp(x);
        let fppx = fpp(x);
        let denom = 2.0 * fpx * fpx - fx * fppx;
        if denom.abs() < 1e-15 { return None; }
        let x_new = x - 2.0 * fx * fpx / denom;
        if (x_new - x).abs() < tol { return Some(x_new); }
        x = x_new;
    }
    Some(x)
}

// ═══════════════════════════════════════════════════════════════════
// NUMERICAL INTEGRATION
// ═══════════════════════════════════════════════════════════════════

/// Trapezoidal rule.
pub fn trapz(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let mut sum = (f(a) + f(b)) / 2.0;
    for i in 1..n { sum += f(a + i as f64 * h); }
    sum * h
}

/// Simpson's rule.
pub fn simps(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let n = if n % 2 != 0 { n + 1 } else { n };
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += if i % 2 == 0 { 2.0 * f(x) } else { 4.0 * f(x) };
    }
    sum * h / 3.0
}

/// Adaptive Simpson's rule (automatic error control).
pub fn adaptive_simps(f: impl Fn(f64) -> f64, a: f64, b: f64, tol: f64) -> f64 {
    adaptive_simps_rec(&f, a, b, tol, f(a), f(b), f((a + b) / 2.0), 20)
}

fn adaptive_simps_rec(f: &impl Fn(f64) -> f64, a: f64, b: f64, tol: f64,
                      fa: f64, fb: f64, fc: f64, depth: usize) -> f64 {
    let h = b - a;
    let mid = (a + b) / 2.0;
    let d = (a + mid) / 2.0;
    let e = (mid + b) / 2.0;
    let fd = f(d);
    let fe = f(e);

    let s1 = h / 6.0 * (fa + 4.0 * fc + fb);
    let s2 = h / 12.0 * (fa + 4.0 * fd + 2.0 * fc + 4.0 * fe + fb);

    if depth == 0 || (s2 - s1).abs() < 15.0 * tol {
        s2 + (s2 - s1) / 15.0
    } else {
        adaptive_simps_rec(f, a, mid, tol / 2.0, fa, fc, fd, depth - 1)
            + adaptive_simps_rec(f, mid, b, tol / 2.0, fc, fb, fe, depth - 1)
    }
}

/// Romberg integration.
pub fn romberg(f: impl Fn(f64) -> f64, a: f64, b: f64, max_order: usize) -> f64 {
    let mut r = vec![vec![0.0; max_order]; max_order];
    r[0][0] = (b - a) / 2.0 * (f(a) + f(b));

    for i in 1..max_order {
        let n = 1usize << i;
        let h = (b - a) / n as f64;
        let mut sum = 0.0;
        for k in 0..n {
            let x = a + (k as f64 + 0.5) * h;
            sum += f(x);
        }
        r[i][0] = (r[i - 1][0] + h * sum) / 2.0; // wrong: this should be the refined trapezoidal

        for j in 1..=i {
            let factor = 4.0f64.powi(j as i32);
            r[i][j] = (factor * r[i][j - 1] - r[i - 1][j - 1]) / (factor - 1.0);
        }
    }
    r[max_order - 1][max_order - 1]
}

/// Gauss-Legendre quadrature (5-point).
pub fn gauss_legendre_5(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    let nodes = [-0.9061798459386640, -0.5384693101056831, 0.0,
                  0.5384693101056831, 0.9061798459386640];
    let weights = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                   0.4786286704993665, 0.2369268850561891];
    let mid = (a + b) / 2.0;
    let half = (b - a) / 2.0;
    let mut sum = 0.0;
    for i in 0..5 {
        sum += weights[i] * f(mid + half * nodes[i]);
    }
    sum * half
}

/// Numerical integration of discrete data (trapezoid rule on arrays).
pub fn trapz_array(y: &[f64], x: &[f64]) -> f64 {
    y.windows(2).zip(x.windows(2))
        .map(|(yw, xw)| (yw[0] + yw[1]) / 2.0 * (xw[1] - xw[0]))
        .sum()
}

/// Numerical integration with uniform spacing.
pub fn trapz_uniform(y: &[f64], dx: f64) -> f64 {
    if y.len() < 2 { return 0.0; }
    let mut sum = (y[0] + y[y.len() - 1]) / 2.0;
    for v in &y[1..y.len() - 1] { sum += v; }
    sum * dx
}

/// Simpson's rule on discrete data (uniform spacing).
pub fn simps_array(y: &[f64], dx: f64) -> f64 {
    let n = y.len();
    if n < 3 { return trapz_uniform(y, dx); }
    let n = if n % 2 == 0 { n - 1 } else { n }; // use odd number of points
    let mut sum = y[0] + y[n - 1];
    for i in 1..n - 1 {
        sum += if i % 2 == 0 { 2.0 * y[i] } else { 4.0 * y[i] };
    }
    sum * dx / 3.0
}

// ═══════════════════════════════════════════════════════════════════
// NUMERICAL DIFFERENTIATION
// ═══════════════════════════════════════════════════════════════════

/// Central difference approximation of f'(x).
pub fn central_diff(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// Second derivative via central difference.
pub fn central_diff2(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
}

/// Forward difference.
pub fn forward_diff(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x)) / h
}

/// Backward difference.
pub fn backward_diff(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x) - f(x - h)) / h
}

/// Five-point stencil for f'(x) (higher accuracy).
pub fn five_point_diff(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (-f(x + 2.0*h) + 8.0*f(x + h) - 8.0*f(x - h) + f(x - 2.0*h)) / (12.0 * h)
}

/// Richardson extrapolation for derivative.
pub fn richardson_diff(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    let d1 = central_diff(&f, x, h);
    let d2 = central_diff(&f, x, h / 2.0);
    (4.0 * d2 - d1) / 3.0
}

/// Gradient of f: R^n → R at point x (central differences).
pub fn gradient(f: impl Fn(&[f64]) -> f64, x: &[f64], h: f64) -> Vec<f64> {
    let n = x.len();
    let mut grad = Vec::with_capacity(n);
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();
    for i in 0..n {
        x_plus[i] = x[i] + h;
        x_minus[i] = x[i] - h;
        grad.push((f(&x_plus) - f(&x_minus)) / (2.0 * h));
        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }
    grad
}

/// Jacobian of f: R^n → R^m at point x.
/// f takes &[f64] (n-dim) and returns Vec<f64> (m-dim).
pub fn jacobian(f: impl Fn(&[f64]) -> Vec<f64>, x: &[f64], h: f64) -> Vec<f64> {
    let n = x.len();
    let f0 = f(x);
    let m = f0.len();
    let mut jac = vec![0.0; m * n]; // row-major m×n
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();
    for j in 0..n {
        x_plus[j] = x[j] + h;
        x_minus[j] = x[j] - h;
        let fp = f(&x_plus);
        let fm = f(&x_minus);
        for i in 0..m {
            jac[i * n + j] = (fp[i] - fm[i]) / (2.0 * h);
        }
        x_plus[j] = x[j];
        x_minus[j] = x[j];
    }
    jac
}

/// Hessian of f: R^n → R at point x. Returns n×n matrix.
pub fn hessian(f: impl Fn(&[f64]) -> f64, x: &[f64], h: f64) -> Vec<f64> {
    let n = x.len();
    let mut hess = vec![0.0; n * n];
    let f0 = f(x);
    let mut xi = x.to_vec();
    let _xj = x.to_vec();
    let mut xij = x.to_vec();
    for i in 0..n {
        for j in i..n {
            if i == j {
                xi[i] = x[i] + h;
                let fp = f(&xi);
                xi[i] = x[i] - h;
                let fm = f(&xi);
                xi[i] = x[i];
                hess[i * n + j] = (fp - 2.0 * f0 + fm) / (h * h);
            } else {
                xij[i] = x[i] + h; xij[j] = x[j] + h;
                let fpp = f(&xij);
                xij[i] = x[i] + h; xij[j] = x[j] - h;
                let fpm = f(&xij);
                xij[i] = x[i] - h; xij[j] = x[j] + h;
                let fmp = f(&xij);
                xij[i] = x[i] - h; xij[j] = x[j] - h;
                let fmm = f(&xij);
                xij[i] = x[i]; xij[j] = x[j];
                hess[i * n + j] = (fpp - fpm - fmp + fmm) / (4.0 * h * h);
                hess[j * n + i] = hess[i * n + j];
            }
        }
    }
    hess
}

// ═══════════════════════════════════════════════════════════════════
// FINITE DIFFERENCE OPERATORS ON ARRAYS
// ═══════════════════════════════════════════════════════════════════

/// Forward difference on array.
pub fn diff_forward(y: &[f64], h: f64) -> Vec<f64> {
    y.windows(2).map(|w| (w[1] - w[0]) / h).collect()
}

/// Central difference on array (shorter by 2).
pub fn diff_central(y: &[f64], h: f64) -> Vec<f64> {
    if y.len() < 3 { return vec![]; }
    (1..y.len() - 1).map(|i| (y[i + 1] - y[i - 1]) / (2.0 * h)).collect()
}

/// Second derivative on array.
pub fn diff2_array(y: &[f64], h: f64) -> Vec<f64> {
    if y.len() < 3 { return vec![]; }
    let h2 = h * h;
    (1..y.len() - 1).map(|i| (y[i + 1] - 2.0 * y[i] + y[i - 1]) / h2).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::PI;

    #[test]
    fn test_bisect() {
        let root = bisect(|x| x * x - 2.0, 1.0, 2.0, 1e-10, 100).unwrap();
        assert!((root - 2.0f64.sqrt()).abs() < 1e-8);
    }

    #[test]
    fn test_newton() {
        let root = newton(|x| x * x - 2.0, |x| 2.0 * x, 1.5, 1e-10, 100).unwrap();
        assert!((root - 2.0f64.sqrt()).abs() < 1e-8);
    }

    #[test]
    fn test_secant() {
        let root = secant(|x| x * x - 2.0, 1.0, 2.0, 1e-10, 100).unwrap();
        assert!((root - 2.0f64.sqrt()).abs() < 1e-8);
    }

    #[test]
    fn test_brentq() {
        let root = brentq(|x| x.powi(3) - x - 2.0, 1.0, 2.0, 1e-10, 100).unwrap();
        let check = root.powi(3) - root - 2.0;
        assert!(check.abs() < 1e-6);
    }

    #[test]
    fn test_trapz() {
        // ∫₀¹ x² dx = 1/3
        let result = trapz(|x| x * x, 0.0, 1.0, 1000);
        assert!((result - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_simps() {
        let result = simps(|x| x * x, 0.0, 1.0, 100);
        assert!((result - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_simps() {
        let result = adaptive_simps(|x| x.sin(), 0.0, PI, 1e-10);
        assert!((result - 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_gauss_legendre() {
        let result = gauss_legendre_5(|x| x * x, 0.0, 1.0);
        assert!((result - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_central_diff() {
        // d/dx(x²) = 2x at x=3 → 6
        let d = central_diff(|x| x * x, 3.0, 1e-5);
        assert!((d - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_gradient() {
        // f(x,y) = x² + y². grad = (2x, 2y)
        let g = gradient(|x| x[0] * x[0] + x[1] * x[1], &[3.0, 4.0], 1e-5);
        assert!((g[0] - 6.0).abs() < 1e-4);
        assert!((g[1] - 8.0).abs() < 1e-4);
    }

    #[test]
    fn test_hessian() {
        // f(x,y) = x² + y². Hessian = [[2,0],[0,2]]
        let h = hessian(|x| x[0] * x[0] + x[1] * x[1], &[1.0, 1.0], 1e-4);
        assert!((h[0] - 2.0).abs() < 1e-3);
        assert!((h[3] - 2.0).abs() < 1e-3);
        assert!(h[1].abs() < 1e-3);
    }

    #[test]
    fn test_trapz_array() {
        let x = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let y: Vec<f64> = x.iter().map(|&t| t * t).collect();
        let result = trapz_array(&y, &x);
        assert!((result - 1.0 / 3.0).abs() < 0.05); // coarse grid
    }

    #[test]
    fn test_diff_forward() {
        let y = vec![0.0, 1.0, 4.0, 9.0]; // x²
        let dy = diff_forward(&y, 1.0);
        assert_eq!(dy.len(), 3);
        assert!((dy[0] - 1.0).abs() < 1e-10);
    }
}
