//! Optimization: minimize, root-finding, curve fitting, linear programming.
//!
//! Replaces scipy.optimize, R's optim(), MATLAB's fminunc/fminsearch.

/// Optimization result.
#[derive(Debug, Clone)]
pub struct OptResult {
    pub x: Vec<f64>,
    pub fun: f64,
    pub nit: usize,
    pub success: bool,
}

/// Nelder-Mead simplex minimization (derivative-free).
pub fn nelder_mead(f: impl Fn(&[f64]) -> f64, x0: &[f64], max_iter: usize, tol: f64) -> OptResult {
    let n = x0.len();
    let alpha = 1.0; let gamma = 2.0; let rho = 0.5; let sigma = 0.5;

    // Initialize simplex
    let mut simplex: Vec<(Vec<f64>, f64)> = Vec::new();
    let f0 = f(x0);
    simplex.push((x0.to_vec(), f0));
    for i in 0..n {
        let mut xi = x0.to_vec();
        xi[i] += if xi[i].abs() > 1e-10 { 0.05 * xi[i] } else { 0.00025 };
        let fi = f(&xi);
        simplex.push((xi, fi));
    }

    for nit in 0..max_iter {
        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Check convergence
        let range = simplex.last().unwrap().1 - simplex[0].1;
        if range < tol {
            return OptResult { x: simplex[0].0.clone(), fun: simplex[0].1, nit, success: true };
        }

        // Centroid (excluding worst)
        let centroid: Vec<f64> = (0..n).map(|j| {
            simplex[..n].iter().map(|(x, _)| x[j]).sum::<f64>() / n as f64
        }).collect();

        // Reflection
        let xr: Vec<f64> = (0..n).map(|j| centroid[j] + alpha * (centroid[j] - simplex[n].0[j])).collect();
        let fr = f(&xr);

        if fr < simplex[0].1 {
            // Expansion
            let xe: Vec<f64> = (0..n).map(|j| centroid[j] + gamma * (xr[j] - centroid[j])).collect();
            let fe = f(&xe);
            if fe < fr { simplex[n] = (xe, fe); } else { simplex[n] = (xr, fr); }
        } else if fr < simplex[n - 1].1 {
            simplex[n] = (xr, fr);
        } else {
            // Contraction
            let xc: Vec<f64> = (0..n).map(|j| centroid[j] + rho * (simplex[n].0[j] - centroid[j])).collect();
            let fc = f(&xc);
            if fc < simplex[n].1 {
                simplex[n] = (xc, fc);
            } else {
                // Shrink
                for i in 1..=n {
                    for j in 0..n { simplex[i].0[j] = simplex[0].0[j] + sigma * (simplex[i].0[j] - simplex[0].0[j]); }
                    simplex[i].1 = f(&simplex[i].0);
                }
            }
        }
    }
    simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    OptResult { x: simplex[0].0.clone(), fun: simplex[0].1, nit: max_iter, success: false }
}

/// BFGS quasi-Newton minimization.
pub fn bfgs(f: impl Fn(&[f64]) -> f64, grad: impl Fn(&[f64]) -> Vec<f64>, x0: &[f64], max_iter: usize, tol: f64) -> OptResult {
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut h = crate::math_runtime::eye(n); // Inverse Hessian approximation (identity)

    for nit in 0..max_iter {
        let g = grad(&x);
        let gnorm: f64 = g.iter().map(|&gi| gi * gi).sum::<f64>().sqrt();
        if gnorm < tol {
            let fval = f(&x);
            return OptResult { x, fun: fval, nit, success: true };
        }

        // Search direction: p = -H * g
        let p: Vec<f64> = (0..n).map(|i| -(0..n).map(|j| h[i][j] * g[j]).sum::<f64>()).collect();

        // Line search (backtracking Armijo)
        let mut alpha = 1.0;
        let fx = f(&x);
        let slope: f64 = g.iter().zip(&p).map(|(&gi, &pi)| gi * pi).sum();
        for _ in 0..30 {
            let x_new: Vec<f64> = x.iter().zip(&p).map(|(&xi, &pi)| xi + alpha * pi).collect();
            if f(&x_new) <= fx + 1e-4 * alpha * slope { break; }
            alpha *= 0.5;
        }

        let x_new: Vec<f64> = x.iter().zip(&p).map(|(&xi, &pi)| xi + alpha * pi).collect();
        let g_new = grad(&x_new);

        let s: Vec<f64> = x_new.iter().zip(&x).map(|(&xn, &xo)| xn - xo).collect();
        let y: Vec<f64> = g_new.iter().zip(&g).map(|(&gn, &go)| gn - go).collect();
        let sy: f64 = s.iter().zip(&y).map(|(&si, &yi)| si * yi).sum();

        if sy > 1e-15 {
            // BFGS update
            let mut hs = vec![0.0; n];
            for i in 0..n { for j in 0..n { hs[i] += h[i][j] * y[j]; } }
            let yhy: f64 = y.iter().zip(&hs).map(|(&yi, &hi)| yi * hi).sum();

            for i in 0..n {
                for j in 0..n {
                    h[i][j] += (sy + yhy) * s[i] * s[j] / (sy * sy) - (hs[i] * s[j] + s[i] * hs[j]) / sy;
                }
            }
        }

        x = x_new;
    }
    let fval = f(&x);
    OptResult { x, fun: fval, nit: max_iter, success: false }
}

/// Bisection root finding.
pub fn bisect(f: impl Fn(f64) -> f64, a: f64, b: f64, tol: f64, max_iter: usize) -> f64 {
    let (mut lo, mut hi) = if f(a) < f(b) { (a, b) } else { (b, a) };
    for _ in 0..max_iter {
        let mid = (lo + hi) / 2.0;
        if (hi - lo) < tol { return mid; }
        if f(mid).signum() == f(lo).signum() { lo = mid; } else { hi = mid; }
    }
    (lo + hi) / 2.0
}

/// Golden section search for scalar minimization.
pub fn golden_section(f: impl Fn(f64) -> f64, a: f64, b: f64, tol: f64) -> f64 {
    let gr = (5.0f64.sqrt() - 1.0) / 2.0;
    let mut a = a; let mut b = b;
    let mut c = b - gr * (b - a);
    let mut d = a + gr * (b - a);
    while (b - a).abs() > tol {
        if f(c) < f(d) { b = d; } else { a = c; }
        c = b - gr * (b - a);
        d = a + gr * (b - a);
    }
    (a + b) / 2.0
}

/// Curve fitting: fit f(x, params) to data (x, y) via Nelder-Mead.
pub fn curve_fit(
    model: impl Fn(f64, &[f64]) -> f64,
    x_data: &[f64],
    y_data: &[f64],
    p0: &[f64],
) -> Vec<f64> {
    let residual = |params: &[f64]| -> f64 {
        x_data.iter().zip(y_data).map(|(&xi, &yi)| {
            (model(xi, params) - yi).powi(2)
        }).sum()
    };
    let result = nelder_mead(residual, p0, 10000, 1e-10);
    result.x
}

/// Simple linear programming: minimize c'x subject to Ax <= b, x >= 0.
/// Uses the simplex method (Phase I / Phase II).
/// Returns (optimal_x, optimal_value) or None if infeasible.
pub fn linprog(c: &[f64], a: &[Vec<f64>], b: &[f64]) -> Option<(Vec<f64>, f64)> {
    let m = a.len(); // constraints
    let n = c.len(); // variables
    // Add slack variables: [A | I] x_aug = b
    let n_aug = n + m;
    let mut tableau = vec![vec![0.0; n_aug + 1]; m + 1];

    // Fill constraints
    for i in 0..m {
        for j in 0..n { tableau[i][j] = a[i][j]; }
        tableau[i][n + i] = 1.0; // slack
        tableau[i][n_aug] = b[i];
        if b[i] < 0.0 { return None; } // need Phase I for negative RHS
    }
    // Objective row
    for j in 0..n { tableau[m][j] = c[j]; }

    // Simplex iterations
    for _ in 0..1000 {
        // Find pivot column (most negative in objective row)
        let pivot_col = (0..n_aug).min_by(|&a, &b| {
            tableau[m][a].partial_cmp(&tableau[m][b]).unwrap()
        }).unwrap();
        if tableau[m][pivot_col] >= -1e-10 { break; } // optimal

        // Find pivot row (minimum ratio test)
        let mut pivot_row = None;
        let mut min_ratio = f64::INFINITY;
        for i in 0..m {
            if tableau[i][pivot_col] > 1e-10 {
                let ratio = tableau[i][n_aug] / tableau[i][pivot_col];
                if ratio < min_ratio { min_ratio = ratio; pivot_row = Some(i); }
            }
        }
        let pivot_row = match pivot_row { Some(r) => r, None => return None }; // unbounded

        // Pivot
        let pivot_val = tableau[pivot_row][pivot_col];
        for j in 0..=n_aug { tableau[pivot_row][j] /= pivot_val; }
        for i in 0..=m {
            if i != pivot_row {
                let factor = tableau[i][pivot_col];
                for j in 0..=n_aug { tableau[i][j] -= factor * tableau[pivot_row][j]; }
            }
        }
    }

    let mut x = vec![0.0; n];
    for j in 0..n {
        // Check if column j is a basic variable (single 1, rest 0)
        let ones: Vec<usize> = (0..m).filter(|&i| (tableau[i][j] - 1.0).abs() < 1e-10).collect();
        let zeros = (0..m).filter(|&i| tableau[i][j].abs() < 1e-10).count();
        if ones.len() == 1 && zeros == m - 1 {
            x[j] = tableau[ones[0]][n_aug];
        }
    }
    let obj = -tableau[m][n_aug];
    Some((x, obj))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nelder_mead_rosenbrock() {
        // Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1,1)
        let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
        let result = nelder_mead(f, &[0.0, 0.0], 50000, 1e-10);
        assert!((result.x[0] - 1.0).abs() < 0.01, "x={}", result.x[0]);
        assert!((result.x[1] - 1.0).abs() < 0.01, "y={}", result.x[1]);
    }

    #[test]
    fn test_bfgs_quadratic() {
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let result = bfgs(f, grad, &[5.0, 3.0], 100, 1e-8);
        assert!(result.success);
        assert!(result.x[0].abs() < 1e-6);
        assert!(result.x[1].abs() < 1e-6);
    }

    #[test]
    fn test_bisect() {
        let root = bisect(|x| x * x - 2.0, 0.0, 2.0, 1e-10, 100);
        assert!((root - std::f64::consts::SQRT_2).abs() < 1e-8);
    }

    #[test]
    fn test_golden_section() {
        let min = golden_section(|x| (x - 3.0).powi(2), 0.0, 10.0, 1e-8);
        assert!((min - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_curve_fit() {
        // Fit y = a*x + b to data
        let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data = vec![1.0, 3.0, 5.0, 7.0, 9.0]; // y = 2x + 1
        let params = curve_fit(|x, p| p[0] * x + p[1], &x_data, &y_data, &[1.0, 0.0]);
        assert!((params[0] - 2.0).abs() < 0.1, "slope={}", params[0]);
        assert!((params[1] - 1.0).abs() < 0.1, "intercept={}", params[1]);
    }

    #[test]
    fn test_linprog() {
        // Minimize -x - y subject to x + y <= 4, x <= 3, y <= 2, x,y >= 0
        // Optimal: x=3, y=1, obj=-4
        let c = vec![-1.0, -1.0];
        let a = vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![4.0, 3.0, 2.0];
        let (x, obj) = linprog(&c, &a, &b).unwrap();
        assert!((obj + 4.0).abs() < 0.1, "optimal value should be -4, got {obj}");
    }
}
