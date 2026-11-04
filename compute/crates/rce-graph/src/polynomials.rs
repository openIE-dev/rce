//! Polynomial operations: fit, evaluate, roots, derivative, integral,
//! Chebyshev, Legendre, Hermite, Laguerre.

// ═══════════════════════════════════════════════════════════════════
// BASIC POLYNOMIAL OPS
// ═══════════════════════════════════════════════════════════════════

/// Polynomial fit (least squares): y = c0 + c1*x + c2*x^2 + ... + cn*x^n.
/// Returns coefficients [c0, c1, ..., cn].
pub fn polyfit(x: &[f64], y: &[f64], degree: usize) -> Vec<f64> {
    let n = x.len().min(y.len());
    let m = degree + 1;
    if n < m { return vec![0.0; m]; }
    let mut ata = vec![vec![0.0; m]; m];
    let mut aty = vec![0.0; m];
    for i in 0..n {
        let mut xi_pow = vec![1.0; m];
        for j in 1..m { xi_pow[j] = xi_pow[j - 1] * x[i]; }
        for j in 0..m {
            aty[j] += xi_pow[j] * y[i];
            for k in 0..m { ata[j][k] += xi_pow[j] * xi_pow[k]; }
        }
    }
    gauss_solve(&mut ata, &mut aty)
}

/// Evaluate polynomial at x: c0 + c1*x + c2*x^2 + ...
pub fn polyval(coeffs: &[f64], x: f64) -> f64 {
    let mut result = 0.0;
    let mut xp = 1.0;
    for &c in coeffs {
        result += c * xp;
        xp *= x;
    }
    result
}

/// Evaluate polynomial at multiple points.
pub fn polyval_vec(coeffs: &[f64], xs: &[f64]) -> Vec<f64> {
    xs.iter().map(|&x| polyval(coeffs, x)).collect()
}

/// Polynomial derivative: d/dx (c0 + c1*x + c2*x^2 + ...) = [c1, 2*c2, 3*c3, ...].
pub fn polyder(coeffs: &[f64]) -> Vec<f64> {
    if coeffs.len() <= 1 { return vec![0.0]; }
    coeffs.iter().enumerate().skip(1).map(|(i, &c)| i as f64 * c).collect()
}

/// Polynomial integral: integral (c0 + c1*x + ...) = [C, c0, c1/2, c2/3, ...].
pub fn polyint(coeffs: &[f64], constant: f64) -> Vec<f64> {
    let mut result = vec![constant];
    for (i, &c) in coeffs.iter().enumerate() {
        result.push(c / (i + 1) as f64);
    }
    result
}

/// Add two polynomials.
pub fn polyadd(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len().max(b.len());
    (0..n).map(|i| a.get(i).unwrap_or(&0.0) + b.get(i).unwrap_or(&0.0)).collect()
}

/// Subtract two polynomials.
pub fn polysub(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len().max(b.len());
    (0..n).map(|i| a.get(i).unwrap_or(&0.0) - b.get(i).unwrap_or(&0.0)).collect()
}

/// Multiply two polynomials.
pub fn polymul(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() { return vec![0.0]; }
    let n = a.len() + b.len() - 1;
    let mut result = vec![0.0; n];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

/// Polynomial division: returns (quotient, remainder).
pub fn polydiv(num: &[f64], den: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = num.len();
    let d = den.len();
    if d == 0 || den.iter().all(|&x| x.abs() < 1e-15) { return (vec![], num.to_vec()); }
    if n < d { return (vec![0.0], num.to_vec()); }
    let mut rem = num.to_vec();
    let mut quot = vec![0.0; n - d + 1];
    let lead = *den.last().unwrap();
    for i in (0..n - d + 1).rev() {
        let idx = i + d - 1;
        if idx < rem.len() {
            quot[i] = rem[idx] / lead;
            for (j, &dj) in den.iter().enumerate() {
                rem[i + j] -= quot[i] * dj;
            }
        }
    }
    rem.truncate(d - 1);
    (quot, rem)
}

/// Find roots of polynomial via companion matrix eigenvalues.
pub fn polyroots(coeffs: &[f64]) -> Vec<f64> {
    let n = coeffs.len();
    if n <= 1 { return vec![]; }
    if n == 2 { return vec![-coeffs[0] / coeffs[1]]; }
    if n == 3 {
        // Quadratic formula
        let (a, b, c) = (coeffs[2], coeffs[1], coeffs[0]);
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 { return vec![]; }
        let sq = disc.sqrt();
        return vec![(-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a)];
    }
    // Companion matrix method
    let comp = crate::linalg_extended::companion(coeffs);
    crate::math_runtime::eigenvalues_symmetric(&comp, 200)
}

/// Polynomial from roots: (x - r1)(x - r2)...
pub fn poly_from_roots(roots: &[f64]) -> Vec<f64> {
    let mut p = vec![1.0];
    for &r in roots {
        p = polymul(&p, &[-r, 1.0]);
    }
    p
}

fn gauss_solve(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let mut max_row = col;
        for row in col + 1..n {
            if a[row][col].abs() > a[max_row][col].abs() { max_row = row; }
        }
        a.swap(col, max_row);
        b.swap(col, max_row);
        let pivot = a[col][col];
        if pivot.abs() < 1e-15 { continue; }
        for row in col + 1..n {
            let factor = a[row][col] / pivot;
            for k in col..n { a[row][k] -= factor * a[col][k]; }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in i + 1..n { sum -= a[i][j] * x[j]; }
        x[i] = if a[i][i].abs() < 1e-15 { 0.0 } else { sum / a[i][i] };
    }
    x
}

// ═══════════════════════════════════════════════════════════════════
// ORTHOGONAL POLYNOMIALS
// ═══════════════════════════════════════════════════════════════════

/// Evaluate Chebyshev polynomial T_n(x) of the first kind.
pub fn chebyshev_t(n: u32, x: f64) -> f64 {
    if n == 0 { return 1.0; }
    if n == 1 { return x; }
    let mut t0 = 1.0;
    let mut t1 = x;
    for _ in 2..=n {
        let t2 = 2.0 * x * t1 - t0;
        t0 = t1;
        t1 = t2;
    }
    t1
}

/// Evaluate Chebyshev polynomial U_n(x) of the second kind.
pub fn chebyshev_u(n: u32, x: f64) -> f64 {
    if n == 0 { return 1.0; }
    if n == 1 { return 2.0 * x; }
    let mut u0 = 1.0;
    let mut u1 = 2.0 * x;
    for _ in 2..=n {
        let u2 = 2.0 * x * u1 - u0;
        u0 = u1;
        u1 = u2;
    }
    u1
}

/// Evaluate Legendre polynomial P_n(x).
pub fn legendre_p(n: u32, x: f64) -> f64 {
    if n == 0 { return 1.0; }
    if n == 1 { return x; }
    let mut p0 = 1.0;
    let mut p1 = x;
    for k in 2..=n {
        let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
        p0 = p1;
        p1 = p2;
    }
    p1
}

/// Evaluate Hermite polynomial H_n(x) (physicist's convention).
pub fn hermite_h(n: u32, x: f64) -> f64 {
    if n == 0 { return 1.0; }
    if n == 1 { return 2.0 * x; }
    let mut h0 = 1.0;
    let mut h1 = 2.0 * x;
    for k in 2..=n {
        let h2 = 2.0 * x * h1 - 2.0 * (k - 1) as f64 * h0;
        h0 = h1;
        h1 = h2;
    }
    h1
}

/// Evaluate Laguerre polynomial L_n(x).
pub fn laguerre_l(n: u32, x: f64) -> f64 {
    if n == 0 { return 1.0; }
    if n == 1 { return 1.0 - x; }
    let mut l0 = 1.0;
    let mut l1 = 1.0 - x;
    for k in 2..=n {
        let l2 = ((2 * k - 1) as f64 - x) * l1 / k as f64 - (k - 1) as f64 * l0 / k as f64;
        l0 = l1;
        l1 = l2;
    }
    l1
}

/// Gauss-Legendre quadrature nodes and weights for n points.
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = Vec::new();
    let mut weights = Vec::new();
    for i in 0..n {
        // Initial guess via Chebyshev
        let mut x = ((i as f64 + 0.75) / (n as f64 + 0.5) * std::f64::consts::PI).cos();
        // Newton refinement
        for _ in 0..20 {
            let p = legendre_p(n as u32, x);
            let pp = n as f64 * (x * p - legendre_p(n as u32 - 1, x)) / (x * x - 1.0);
            let dx = p / pp;
            x -= dx;
            if dx.abs() < 1e-15 { break; }
        }
        nodes.push(x);
        let pp = n as f64 * (x * legendre_p(n as u32, x) - legendre_p(n as u32 - 1, x)) / (x * x - 1.0);
        weights.push(2.0 / ((1.0 - x * x) * pp * pp));
    }
    (nodes, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polyfit_linear() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0]; // y = 1 + 2x
        let c = polyfit(&x, &y, 1);
        assert!((c[0] - 1.0).abs() < 0.01);
        assert!((c[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_polyval() {
        let c = vec![1.0, 2.0, 3.0]; // 1 + 2x + 3x^2
        assert!((polyval(&c, 0.0) - 1.0).abs() < 1e-10);
        assert!((polyval(&c, 1.0) - 6.0).abs() < 1e-10);
        assert!((polyval(&c, 2.0) - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_polyder() {
        let c = vec![1.0, 2.0, 3.0]; // 1 + 2x + 3x^2
        let d = polyder(&c); // 2 + 6x
        assert!((d[0] - 2.0).abs() < 1e-10);
        assert!((d[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_polyint() {
        let c = vec![2.0, 6.0]; // 2 + 6x
        let i = polyint(&c, 1.0); // 1 + 2x + 3x^2
        assert!((i[0] - 1.0).abs() < 1e-10);
        assert!((i[1] - 2.0).abs() < 1e-10);
        assert!((i[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_polymul() {
        let a = vec![1.0, 1.0]; // 1 + x
        let b = vec![1.0, 1.0]; // 1 + x
        let c = polymul(&a, &b); // 1 + 2x + x^2
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[1] - 2.0).abs() < 1e-10);
        assert!((c[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_poly_from_roots() {
        let p = poly_from_roots(&[1.0, 2.0]); // (x-1)(x-2) = 2 - 3x + x^2
        assert!((p[0] - 2.0).abs() < 1e-10);
        assert!((p[1] + 3.0).abs() < 1e-10);
        assert!((p[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_polyroots_quadratic() {
        let c = vec![-6.0, 1.0, 1.0]; // x^2 + x - 6 = (x+3)(x-2)
        let mut roots = polyroots(&c);
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((roots[0] + 3.0).abs() < 0.1);
        assert!((roots[1] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_chebyshev() {
        // T_0(x) = 1, T_1(x) = x
        assert!((chebyshev_t(0, 0.5) - 1.0).abs() < 1e-10);
        assert!((chebyshev_t(1, 0.5) - 0.5).abs() < 1e-10);
        // T_2(x) = 2x^2 - 1
        assert!((chebyshev_t(2, 0.5) - (2.0 * 0.25 - 1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_legendre() {
        assert!((legendre_p(0, 0.5) - 1.0).abs() < 1e-10);
        assert!((legendre_p(1, 0.5) - 0.5).abs() < 1e-10);
        // P_2(x) = (3x^2 - 1) / 2
        assert!((legendre_p(2, 0.5) - (3.0 * 0.25 - 1.0) / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_hermite() {
        assert!((hermite_h(0, 1.0) - 1.0).abs() < 1e-10);
        assert!((hermite_h(1, 1.0) - 2.0).abs() < 1e-10);
        // H_2(x) = 4x^2 - 2
        assert!((hermite_h(2, 1.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_laguerre() {
        assert!((laguerre_l(0, 1.0) - 1.0).abs() < 1e-10);
        assert!((laguerre_l(1, 1.0) - 0.0).abs() < 1e-10);
        // L_2(x) = (x^2 - 4x + 2) / 2
        assert!((laguerre_l(2, 1.0) - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_gauss_legendre() {
        let (nodes, weights) = gauss_legendre(3);
        assert_eq!(nodes.len(), 3);
        // Weights should sum to 2 (integral of 1 over [-1,1])
        let wsum: f64 = weights.iter().sum();
        assert!((wsum - 2.0).abs() < 0.01);
    }
}
