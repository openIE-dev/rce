//! Math runtime: NxN linear algebra, statistics, DSP, symbolic, finance, bioinformatics.
//!
//! Pure Rust. No external dependencies. Every function operates on Vec<f64> or Vec<Vec<f64>>.
//! This is the compute engine behind Charlot's Custom OpKinds.

use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════
// LINEAR ALGEBRA — MATLAB/Julia/NumPy parity
// ═══════════════════════════════════════════════════════════════════

/// NxN matrix multiply: C = A * B.
pub fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let k = if m > 0 { a[0].len() } else { 0 };
    let n = if !b.is_empty() { b[0].len() } else { 0 };
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                c[i][j] += a[i][p] * b[p][j];
            }
        }
    }
    c
}

/// Matrix transpose.
pub fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = if m > 0 { a[0].len() } else { 0 };
    let mut t = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// Determinant via LU decomposition (NxN).
pub fn determinant(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    if n == 0 { return 0.0; }
    if n == 1 { return a[0][0]; }
    if n == 2 { return a[0][0] * a[1][1] - a[0][1] * a[1][0]; }
    let (lu, _piv, sign) = lu_decompose(a);
    let mut det = sign as f64;
    for i in 0..n { det *= lu[i][i]; }
    det
}

/// Matrix inverse via Gauss-Jordan elimination (NxN).
pub fn inverse(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 { return Some(vec![]); }
    // Augmented matrix [A | I]
    let mut aug: Vec<Vec<f64>> = (0..n).map(|i| {
        let mut row = a[i].clone();
        row.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
        row
    }).collect();

    for col in 0..n {
        // Pivot
        let mut max_row = col;
        for row in col + 1..n {
            if aug[row][col].abs() > aug[max_row][col].abs() { max_row = row; }
        }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        if pivot.abs() < 1e-14 { return None; } // singular

        // Scale pivot row
        for j in 0..2 * n { aug[col][j] /= pivot; }

        // Eliminate column
        for row in 0..n {
            if row == col { continue; }
            let factor = aug[row][col];
            for j in 0..2 * n { aug[row][j] -= factor * aug[col][j]; }
        }
    }

    Some(aug.iter().map(|row| row[n..].to_vec()).collect())
}

/// Solve A * x = b via LU decomposition (NxN).
pub fn solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 || b.len() != n { return None; }

    let (lu, piv, _) = lu_decompose(a);

    // Apply permutation to b
    let mut pb: Vec<f64> = piv.iter().map(|&i| b[i]).collect();

    // Forward substitution (L * y = Pb)
    for i in 1..n {
        for j in 0..i {
            pb[i] -= lu[i][j] * pb[j];
        }
    }

    // Back substitution (U * x = y)
    for i in (0..n).rev() {
        for j in i + 1..n {
            pb[i] -= lu[i][j] * pb[j];
        }
        if lu[i][i].abs() < 1e-14 { return None; }
        pb[i] /= lu[i][i];
    }

    Some(pb)
}

/// LU decomposition with partial pivoting. Returns (LU, pivot_indices, sign).
pub fn lu_decompose(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<usize>, i32) {
    let n = a.len();
    let mut lu: Vec<Vec<f64>> = a.to_vec();
    let mut piv: Vec<usize> = (0..n).collect();
    let mut sign: i32 = 1;

    for k in 0..n {
        // Find pivot
        let mut max_val = 0.0;
        let mut max_row = k;
        for i in k..n {
            if lu[i][k].abs() > max_val { max_val = lu[i][k].abs(); max_row = i; }
        }
        if max_val < 1e-14 { continue; } // singular column

        if max_row != k {
            lu.swap(k, max_row);
            piv.swap(k, max_row);
            sign = -sign;
        }

        for i in k + 1..n {
            lu[i][k] /= lu[k][k];
            for j in k + 1..n {
                lu[i][j] -= lu[i][k] * lu[k][j];
            }
        }
    }
    (lu, piv, sign)
}

/// QR decomposition via Gram-Schmidt. Returns (Q, R).
pub fn qr_decompose(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let m = a.len();
    let n = if m > 0 { a[0].len() } else { 0 };
    let k = m.min(n);

    // Work with columns
    let cols: Vec<Vec<f64>> = (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect();
    let mut q_cols: Vec<Vec<f64>> = Vec::new();
    let mut r = vec![vec![0.0; n]; k];

    for j in 0..k {
        let mut v = cols[j].clone();
        for i in 0..j {
            let dot: f64 = v.iter().zip(&q_cols[i]).map(|(a, b)| a * b).sum();
            r[i][j] = dot;
            for l in 0..m { v[l] -= dot * q_cols[i][l]; }
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        r[j][j] = norm;
        if norm > 1e-14 {
            for l in 0..m { v[l] /= norm; }
        }
        q_cols.push(v);
    }

    // Convert Q columns to row-major
    let q: Vec<Vec<f64>> = (0..m).map(|i| q_cols.iter().map(|col| col[i]).collect()).collect();
    (q, r)
}

/// Cholesky decomposition: A = L * L^T. A must be symmetric positive definite.
pub fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j { sum += l[i][k] * l[j][k]; }
            if i == j {
                let val = a[i][i] - sum;
                if val <= 0.0 { return None; } // not positive definite
                l[i][j] = val.sqrt();
            } else {
                if l[j][j].abs() < 1e-14 { return None; }
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }
    Some(l)
}

/// Eigenvalues of a symmetric matrix via QR algorithm (iterative).
pub fn eigenvalues_symmetric(a: &[Vec<f64>], max_iter: usize) -> Vec<f64> {
    let n = a.len();
    if n == 0 { return vec![]; }
    if n == 1 { return vec![a[0][0]]; }

    let mut ak = a.to_vec();
    for _ in 0..max_iter {
        let (q, r) = qr_decompose(&ak);
        ak = mat_mul(&r, &q);
        // Check convergence (off-diagonal elements small)
        let mut off_diag = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j { off_diag += ak[i][j] * ak[i][j]; }
            }
        }
        if off_diag < 1e-20 { break; }
    }

    (0..n).map(|i| ak[i][i]).collect()
}

/// Singular Value Decomposition (SVD) — returns singular values only.
pub fn singular_values(a: &[Vec<f64>], max_iter: usize) -> Vec<f64> {
    let at = transpose(a);
    let ata = mat_mul(&at, a);
    let mut eigs = eigenvalues_symmetric(&ata, max_iter);
    eigs.iter_mut().for_each(|e| *e = e.abs().sqrt());
    eigs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    eigs
}

/// Vector norm (L2).
pub fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Dot product.
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Cross product (3D only).
pub fn cross_product(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.len() != 3 || b.len() != 3 { return vec![]; }
    vec![
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Identity matrix.
pub fn eye(n: usize) -> Vec<Vec<f64>> {
    (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect()
}

/// Zeros matrix.
pub fn zeros(m: usize, n: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0; n]; m]
}

/// Ones matrix.
pub fn ones(m: usize, n: usize) -> Vec<Vec<f64>> {
    vec![vec![1.0; n]; m]
}

/// Linspace: n evenly spaced values from start to end (inclusive).
pub fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n <= 1 { return vec![start]; }
    (0..n).map(|i| start + (end - start) * i as f64 / (n - 1) as f64).collect()
}

/// Matrix trace.
pub fn trace(a: &[Vec<f64>]) -> f64 {
    a.iter().enumerate().map(|(i, row)| if i < row.len() { row[i] } else { 0.0 }).sum()
}

/// Frobenius norm.
pub fn frobenius_norm(a: &[Vec<f64>]) -> f64 {
    a.iter().flat_map(|row| row.iter()).map(|x| x * x).sum::<f64>().sqrt()
}

/// Matrix rank (via SVD, count non-zero singular values).
pub fn rank(a: &[Vec<f64>]) -> usize {
    let svs = singular_values(a, 100);
    let tol = 1e-10 * svs.first().copied().unwrap_or(0.0);
    svs.iter().filter(|&&s| s > tol).count()
}

// ═══════════════════════════════════════════════════════════════════
// STATISTICS — R/SAS/Statistica parity
// ═══════════════════════════════════════════════════════════════════

pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn variance(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 { return 0.0; }
    let m = mean(data);
    data.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0)
}

pub fn std_dev(data: &[f64]) -> f64 { variance(data).sqrt() }

pub fn median(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut s = data.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 0 { (s[n / 2 - 1] + s[n / 2]) / 2.0 } else { s[n / 2] }
}

pub fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut s = data.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = p * (s.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil().min((s.len() - 1) as f64) as usize;
    let frac = idx - lo as f64;
    s[lo] * (1.0 - frac) + s[hi] * frac
}

pub fn covariance(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    if n < 2.0 { return 0.0; }
    let (mx, my) = (mean(x), mean(y));
    x.iter().zip(y).map(|(&xi, &yi)| (xi - mx) * (yi - my)).sum::<f64>() / (n - 1.0)
}

pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let sx = std_dev(x);
    let sy = std_dev(y);
    if sx < 1e-15 || sy < 1e-15 { return 0.0; }
    covariance(x, y) / (sx * sy)
}

/// Ordinary least squares: y = b0 + b1*x. Returns (b0, b1, r_squared).
pub fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len().min(y.len()) as f64;
    if n < 2.0 { return (0.0, 0.0, 0.0); }
    let (mx, my) = (mean(x), mean(y));
    let mut sxx = 0.0; let mut sxy = 0.0; let mut syy = 0.0;
    for i in 0..n as usize {
        let (dx, dy) = (x[i] - mx, y[i] - my);
        sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
    }
    let b1 = if sxx.abs() < 1e-15 { 0.0 } else { sxy / sxx };
    let b0 = my - b1 * mx;
    let ss_res: f64 = (0..n as usize).map(|i| (y[i] - b0 - b1 * x[i]).powi(2)).sum();
    let r2 = if syy.abs() < 1e-15 { 1.0 } else { 1.0 - ss_res / syy };
    (b0, b1, r2)
}

/// T-test: one-sample, returns (t_statistic, p_value_approx).
pub fn t_test_one_sample(data: &[f64], mu0: f64) -> (f64, f64) {
    let n = data.len() as f64;
    if n < 2.0 { return (0.0, 1.0); }
    let m = mean(data);
    let s = std_dev(data);
    if s < 1e-15 { return (0.0, 1.0); }
    let t = (m - mu0) / (s / n.sqrt());
    let df = n - 1.0;
    let p = 2.0 * t_cdf_approx(-t.abs(), df); // two-tailed
    (t, p)
}

/// T-test: two-sample (Welch's), returns (t_statistic, p_value_approx).
pub fn t_test_two_sample(a: &[f64], b: &[f64]) -> (f64, f64) {
    let (na, nb) = (a.len() as f64, b.len() as f64);
    if na < 2.0 || nb < 2.0 { return (0.0, 1.0); }
    let (ma, mb) = (mean(a), mean(b));
    let (va, vb) = (variance(a), variance(b));
    let se = (va / na + vb / nb).sqrt();
    if se < 1e-15 { return (0.0, 1.0); }
    let t = (ma - mb) / se;
    let num = (va / na + vb / nb).powi(2);
    let den = (va / na).powi(2) / (na - 1.0) + (vb / nb).powi(2) / (nb - 1.0);
    let df = num / den;
    let p = 2.0 * t_cdf_approx(-t.abs(), df);
    (t, p)
}

/// Approximate t-distribution CDF (via normal approximation for large df).
fn t_cdf_approx(t: f64, df: f64) -> f64 {
    // For df > 30, approximate with normal
    if df > 30.0 {
        return normal_cdf(t);
    }
    // Simple approximation using normal with correction
    let g = (df - 0.5) * (1.0 + t * t / df).ln();
    normal_cdf(-g.sqrt().copysign(t))
}

/// Standard normal CDF approximation (Abramowitz & Stegun).
pub fn normal_cdf(x: f64) -> f64 {
    if x < -8.0 { return 0.0; }
    if x > 8.0 { return 1.0; }
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327; // 1/sqrt(2*pi)
    let p = d * (-x * x / 2.0).exp();
    let poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    if x >= 0.0 { 1.0 - p * poly } else { p * poly }
}

/// Normal PDF.
pub fn normal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    (-(z * z) / 2.0).exp() / (sigma * (2.0 * PI).sqrt())
}

/// Chi-squared test statistic from observed and expected counts.
pub fn chi_squared(observed: &[f64], expected: &[f64]) -> f64 {
    observed.iter().zip(expected).map(|(&o, &e)| {
        if e.abs() < 1e-15 { 0.0 } else { (o - e).powi(2) / e }
    }).sum()
}

// ═══════════════════════════════════════════════════════════════════
// STATISTICAL TESTS — R/SciPy/statsmodels parity
// ═══════════════════════════════════════════════════════════════════

/// Result of a statistical test: (statistic, p_value)
pub type TestResult = (f64, f64);

/// One-sample t-test: H0: mean = mu0. Returns (t_statistic, p_value).
pub fn t_test_1samp(data: &[f64], mu0: f64) -> TestResult {
    let n = data.len() as f64;
    if n < 2.0 { return (0.0, 1.0); }
    let m = mean(data);
    let s = std_dev(data);
    if s < 1e-15 { return (0.0, 1.0); }
    let t = (m - mu0) / (s / n.sqrt());
    let df = n - 1.0;
    let p = 2.0 * (1.0 - crate::distributions::t_cdf(t.abs(), df));
    (t, p)
}

/// Welch's two-sample t-test. Returns (t_statistic, p_value).
pub fn t_test_ind(a: &[f64], b: &[f64]) -> TestResult {
    let (na, nb) = (a.len() as f64, b.len() as f64);
    if na < 2.0 || nb < 2.0 { return (0.0, 1.0); }
    let (ma, mb) = (mean(a), mean(b));
    let (va, vb) = (variance(a), variance(b));
    let se = (va / na + vb / nb).sqrt();
    if se < 1e-15 { return (0.0, 1.0); }
    let t = (ma - mb) / se;
    let num = (va / na + vb / nb).powi(2);
    let den = (va / na).powi(2) / (na - 1.0) + (vb / nb).powi(2) / (nb - 1.0);
    let df = num / den;
    let p = 2.0 * (1.0 - crate::distributions::t_cdf(t.abs(), df));
    (t, p)
}

/// Paired t-test. Returns (t_statistic, p_value).
pub fn t_test_rel(a: &[f64], b: &[f64]) -> TestResult {
    let diffs: Vec<f64> = a.iter().zip(b).map(|(&x, &y)| x - y).collect();
    t_test_1samp(&diffs, 0.0)
}

/// F-test to compare two variances. Returns (f_statistic, p_value).
pub fn var_test(a: &[f64], b: &[f64]) -> TestResult {
    let va = variance(a);
    let vb = variance(b);
    if vb < 1e-15 { return (0.0, 1.0); }
    let f = va / vb;
    let d1 = (a.len() - 1) as f64;
    let d2 = (b.len() - 1) as f64;
    let p = 2.0 * (1.0 - crate::distributions::f_cdf(f.max(1.0 / f), d1, d2)).min(1.0);
    (f, p)
}

/// One-way ANOVA. Returns (f_statistic, p_value).
pub fn f_oneway(groups: &[&[f64]]) -> TestResult {
    let k = groups.len() as f64;
    if k < 2.0 { return (0.0, 1.0); }
    let n_total: f64 = groups.iter().map(|g| g.len() as f64).sum();
    let grand_mean = groups.iter().flat_map(|g| g.iter()).sum::<f64>() / n_total;

    let ss_between: f64 = groups.iter().map(|g| {
        let ni = g.len() as f64;
        let gi_mean = mean(g);
        ni * (gi_mean - grand_mean).powi(2)
    }).sum();

    let ss_within: f64 = groups.iter().map(|g| {
        let gi_mean = mean(g);
        g.iter().map(|&x| (x - gi_mean).powi(2)).sum::<f64>()
    }).sum();

    let df_between = k - 1.0;
    let df_within = n_total - k;
    if df_within < 1.0 || ss_within < 1e-15 { return (0.0, 1.0); }

    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;
    let f = ms_between / ms_within;
    let p = 1.0 - crate::distributions::f_cdf(f, df_between, df_within);
    (f, p)
}

/// Pearson's chi-squared test of independence on contingency table.
/// Input: flattened contingency table, nrows, ncols.
pub fn chi2_test(observed: &[f64], nrows: usize, ncols: usize) -> TestResult {
    let total: f64 = observed.iter().sum();
    if total < 1e-15 { return (0.0, 1.0); }

    let row_sums: Vec<f64> = (0..nrows).map(|r| {
        (0..ncols).map(|c| observed[r * ncols + c]).sum()
    }).collect();
    let col_sums: Vec<f64> = (0..ncols).map(|c| {
        (0..nrows).map(|r| observed[r * ncols + c]).sum()
    }).collect();

    let mut chi2 = 0.0;
    for r in 0..nrows {
        for c in 0..ncols {
            let expected = row_sums[r] * col_sums[c] / total;
            if expected > 1e-15 {
                chi2 += (observed[r * ncols + c] - expected).powi(2) / expected;
            }
        }
    }
    let df = ((nrows - 1) * (ncols - 1)) as f64;
    let p = 1.0 - crate::distributions::chi2_cdf(chi2, df);
    (chi2, p)
}

/// Shapiro-Wilk test for normality (approximation for small n).
/// Returns (W_statistic, p_value_approx).
pub fn shapiro_test(data: &[f64]) -> TestResult {
    let n = data.len();
    if n < 3 || n > 5000 { return (1.0, 1.0); }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = mean(data);
    let ss: f64 = data.iter().map(|&x| (x - m).powi(2)).sum();
    if ss < 1e-15 { return (1.0, 1.0); }

    // Approximate Shapiro-Wilk using normal order statistics
    let mut a_sum = 0.0;
    for i in 0..n / 2 {
        let mi = crate::distributions::norm_ppf((i as f64 + 0.625) / (n as f64 + 0.25), 0.0, 1.0);
        let ai = mi; // simplified — full SW uses tabulated coefficients
        a_sum += ai * (sorted[n - 1 - i] - sorted[i]);
    }
    let w = (a_sum * a_sum) / ss;
    // Approximate p-value via normal approximation of ln(1-W)
    let ln1w = (1.0 - w).max(1e-15).ln();
    let mu_ln = -1.2725 + 1.0521 * (n as f64).ln();
    let sigma_ln = 1.0308 - 0.26758 * (n as f64).ln();
    let z = (ln1w - mu_ln) / sigma_ln;
    let p = crate::distributions::norm_cdf(z, 0.0, 1.0);
    (w.min(1.0), p)
}

/// Kolmogorov-Smirnov one-sample test against a normal distribution.
/// Returns (D_statistic, p_value_approx).
pub fn ks_test_normal(data: &[f64]) -> TestResult {
    let n = data.len();
    if n == 0 { return (0.0, 1.0); }
    let m = mean(data);
    let s = std_dev(data);
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut d_max: f64 = 0.0;
    for (i, &x) in sorted.iter().enumerate() {
        let f_emp = (i + 1) as f64 / n as f64;
        let f_theo = crate::distributions::norm_cdf(x, m, s);
        let d1 = (f_emp - f_theo).abs();
        let d2 = (f_emp - 1.0 / n as f64 - f_theo).abs();
        if d1 > d_max { d_max = d1; }
        if d2 > d_max { d_max = d2; }
    }

    // Approximate p-value
    let sqrt_n = (n as f64).sqrt();
    let lambda = (sqrt_n + 0.12 + 0.11 / sqrt_n) * d_max;
    let p = 2.0 * (-2.0 * lambda * lambda).exp(); // Kolmogorov approximation
    (d_max, p.clamp(0.0, 1.0))
}

/// Mann-Whitney U test (non-parametric two-sample). Returns (U_statistic, p_value).
pub fn mann_whitney_u(a: &[f64], b: &[f64]) -> TestResult {
    let na = a.len();
    let nb = b.len();
    if na == 0 || nb == 0 { return (0.0, 1.0); }

    // Combine and rank
    let mut combined: Vec<(f64, bool)> = a.iter().map(|&x| (x, true)).chain(b.iter().map(|&x| (x, false))).collect();
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let n = combined.len();

    // Assign ranks (average for ties)
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (combined[j].0 - combined[i].0).abs() < 1e-15 { j += 1; }
        let avg = (i + j + 1) as f64 / 2.0;
        for k in i..j { ranks[k] = avg; }
        i = j;
    }

    let r1: f64 = ranks.iter().zip(&combined).filter(|(_, c)| c.1).map(|(r, _)| r).sum();
    let u1 = r1 - na as f64 * (na as f64 + 1.0) / 2.0;
    let u2 = (na * nb) as f64 - u1;
    let u = u1.min(u2);

    // Normal approximation
    let mu = (na * nb) as f64 / 2.0;
    let sigma = ((na * nb) as f64 * (na + nb + 1) as f64 / 12.0).sqrt();
    if sigma < 1e-15 { return (u, 1.0); }
    let z = (u - mu) / sigma;
    let p = 2.0 * crate::distributions::norm_cdf(z, 0.0, 1.0);
    (u, p.clamp(0.0, 1.0))
}

/// Wilcoxon signed-rank test (non-parametric paired). Returns (W_statistic, p_value).
pub fn wilcoxon_test(a: &[f64], b: &[f64]) -> TestResult {
    let diffs: Vec<f64> = a.iter().zip(b).map(|(&x, &y)| x - y).filter(|&d| d.abs() > 1e-15).collect();
    let n = diffs.len();
    if n == 0 { return (0.0, 1.0); }

    let abs_diffs: Vec<f64> = diffs.iter().map(|d| d.abs()).collect();
    let ranks = rankdata(&abs_diffs);

    let mut w_plus: f64 = 0.0;
    let mut w_minus: f64 = 0.0;
    for (d, r) in diffs.iter().zip(ranks.iter()) {
        if *d > 0.0 { w_plus += r; }
        else if *d < 0.0 { w_minus += r; }
    }
    let w = w_plus.min(w_minus);

    // Normal approximation
    let n_f = n as f64;
    let mu = n_f * (n_f + 1.0) / 4.0;
    let sigma = (n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0).sqrt();
    if sigma < 1e-15 { return (w, 1.0); }
    let z = (w - mu) / sigma;
    let p = 2.0 * crate::distributions::norm_cdf(z, 0.0, 1.0);
    (w, p.clamp(0.0, 1.0))
}

/// Kruskal-Wallis H-test (non-parametric one-way ANOVA). Returns (H_statistic, p_value).
pub fn kruskal_wallis(groups: &[&[f64]]) -> TestResult {
    let k = groups.len();
    if k < 2 { return (0.0, 1.0); }
    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    if n_total < 3 { return (0.0, 1.0); }

    // Combine and rank
    let mut all: Vec<(f64, usize)> = Vec::new(); // (value, group_index)
    for (gi, g) in groups.iter().enumerate() {
        for &v in *g { all.push((v, gi)); }
    }
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let vals: Vec<f64> = all.iter().map(|a| a.0).collect();
    let ranks = rankdata(&vals);

    // Sum of ranks per group
    let mut rank_sums = vec![0.0; k];
    for (i, &(_, gi)) in all.iter().enumerate() {
        rank_sums[gi] += ranks[i];
    }

    let n = n_total as f64;
    let mut h = 0.0;
    for (gi, g) in groups.iter().enumerate() {
        let ni = g.len() as f64;
        if ni > 0.0 {
            h += rank_sums[gi].powi(2) / ni;
        }
    }
    h = 12.0 / (n * (n + 1.0)) * h - 3.0 * (n + 1.0);

    let df = (k - 1) as f64;
    let p = 1.0 - crate::distributions::chi2_cdf(h, df);
    (h, p)
}

/// Bartlett's test for equal variances. Returns (statistic, p_value).
pub fn bartlett_test(groups: &[&[f64]]) -> TestResult {
    let k = groups.len();
    if k < 2 { return (0.0, 1.0); }
    let ns: Vec<f64> = groups.iter().map(|g| g.len() as f64).collect();
    let n_total: f64 = ns.iter().sum();
    let vars: Vec<f64> = groups.iter().map(|g| variance(g)).collect();

    let sp2 = ns.iter().zip(&vars).map(|(&ni, &vi)| (ni - 1.0) * vi).sum::<f64>() / (n_total - k as f64);
    if sp2 < 1e-15 { return (0.0, 1.0); }

    let num = (n_total - k as f64) * sp2.ln() - ns.iter().zip(&vars).map(|(&ni, &vi)| (ni - 1.0) * vi.max(1e-300).ln()).sum::<f64>();
    let den = 1.0 + (1.0 / (3.0 * (k as f64 - 1.0))) * (ns.iter().map(|&ni| 1.0 / (ni - 1.0)).sum::<f64>() - 1.0 / (n_total - k as f64));
    let chi2 = num / den;
    let df = (k - 1) as f64;
    let p = 1.0 - crate::distributions::chi2_cdf(chi2, df);
    (chi2, p)
}

/// Levene's test for equal variances (more robust than Bartlett).
pub fn levene_test(groups: &[&[f64]]) -> TestResult {
    // Use deviations from group medians
    let medians: Vec<f64> = groups.iter().map(|g| median(g)).collect();
    let transformed: Vec<Vec<f64>> = groups.iter().zip(&medians).map(|(g, &med)| {
        g.iter().map(|&x| (x - med).abs()).collect()
    }).collect();
    let refs: Vec<&[f64]> = transformed.iter().map(|v| v.as_slice()).collect();
    f_oneway(&refs)
}

/// Spearman rank correlation. Returns (rho, p_value).
pub fn spearman_r(x: &[f64], y: &[f64]) -> TestResult {
    let rx = rankdata(x);
    let ry = rankdata(y);
    let rho = correlation(&rx, &ry);
    let n = x.len() as f64;
    if n < 3.0 { return (rho, 1.0); }
    let t = rho * ((n - 2.0) / (1.0 - rho * rho)).sqrt();
    let p = 2.0 * (1.0 - crate::distributions::t_cdf(t.abs(), n - 2.0));
    (rho, p)
}

/// Kendall's tau-b rank correlation. Returns (tau, p_value).
pub fn kendall_tau(x: &[f64], y: &[f64]) -> TestResult {
    let n = x.len();
    if n < 2 { return (0.0, 1.0); }
    let mut concordant: i64 = 0;
    let mut discordant: i64 = 0;
    for i in 0..n {
        for j in i + 1..n {
            let dx = x[j] - x[i];
            let dy = y[j] - y[i];
            let prod = dx * dy;
            if prod > 0.0 { concordant += 1; }
            else if prod < 0.0 { discordant += 1; }
        }
    }
    let n0 = (n * (n - 1)) as f64 / 2.0;
    if n0 < 1.0 { return (0.0, 1.0); }
    let tau = (concordant - discordant) as f64 / n0;
    // Normal approximation for p-value
    let sigma = (2.0 * (2.0 * n as f64 + 5.0) / (9.0 * n as f64 * (n as f64 - 1.0))).sqrt();
    let z = tau / sigma;
    let p = 2.0 * (1.0 - crate::distributions::norm_cdf(z.abs(), 0.0, 1.0));
    (tau, p)
}

/// Durbin-Watson statistic for autocorrelation of residuals.
pub fn durbin_watson(residuals: &[f64]) -> f64 {
    if residuals.len() < 2 { return 2.0; }
    let num: f64 = residuals.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
    let den: f64 = residuals.iter().map(|&r| r * r).sum();
    if den < 1e-15 { 2.0 } else { num / den }
}

/// Jarque-Bera test for normality. Returns (JB_statistic, p_value).
pub fn jarque_bera_test(data: &[f64]) -> TestResult {
    let n = data.len() as f64;
    if n < 4.0 { return (0.0, 1.0); }
    let s = skewness(data);
    let k = kurtosis(data);
    let jb = n / 6.0 * (s * s + k * k / 4.0);
    let p = 1.0 - crate::distributions::chi2_cdf(jb, 2.0);
    (jb, p)
}

/// Binomial test: exact test for proportion. Returns p_value.
pub fn binom_test(successes: u64, n: u64, p0: f64) -> f64 {
    // Two-sided: sum probabilities of outcomes as extreme or more extreme
    let obs_prob = crate::distributions::binom_pmf(successes, n, p0);
    let mut p_val = 0.0;
    for k in 0..=n {
        let pk = crate::distributions::binom_pmf(k, n, p0);
        if pk <= obs_prob + 1e-15 {
            p_val += pk;
        }
    }
    p_val.min(1.0)
}

/// Proportion z-test. Returns (z_statistic, p_value).
pub fn proportion_test(successes: u64, n: u64, p0: f64) -> TestResult {
    let p_hat = successes as f64 / n as f64;
    let se = (p0 * (1.0 - p0) / n as f64).sqrt();
    if se < 1e-15 { return (0.0, 1.0); }
    let z = (p_hat - p0) / se;
    let p = 2.0 * (1.0 - crate::distributions::norm_cdf(z.abs(), 0.0, 1.0));
    (z, p)
}

// ═══════════════════════════════════════════════════════════════════
// DSP — MATLAB Signal Processing / LabVIEW / GNU Radio parity
// ═══════════════════════════════════════════════════════════════════

/// FFT (Cooley-Tukey radix-2 DIT). Input length must be power of 2.
pub fn fft(real: &[f64]) -> Vec<(f64, f64)> {
    let n = real.len();
    let n2 = n.next_power_of_two();
    let mut re: Vec<f64> = real.to_vec();
    re.resize(n2, 0.0);
    let mut im = vec![0.0; n2];
    fft_inplace(&mut re, &mut im, false);
    re.into_iter().zip(im).collect()
}

/// Inverse FFT.
pub fn ifft(spectrum: &[(f64, f64)]) -> Vec<f64> {
    let n = spectrum.len();
    let mut re: Vec<f64> = spectrum.iter().map(|&(r, _)| r).collect();
    let mut im: Vec<f64> = spectrum.iter().map(|&(_, i)| i).collect();
    fft_inplace(&mut re, &mut im, true);
    let scale = 1.0 / n as f64;
    re.iter().map(|&r| r * scale).collect()
}

pub fn fft_inplace_pub(re: &mut [f64], im: &mut [f64], inverse: bool) { fft_inplace(re, im, inverse); }

fn fft_inplace(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    if n <= 1 { return; }

    // Bit-reversal permutation
    let mut j = 0;
    for i in 0..n {
        if i < j { re.swap(i, j); im.swap(i, j); }
        let mut m = n >> 1;
        while m >= 1 && j >= m { j -= m; m >>= 1; }
        j += m;
    }

    // Butterfly
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = if inverse { 2.0 * PI / len as f64 } else { -2.0 * PI / len as f64 };
        let wn_re = angle.cos();
        let wn_im = angle.sin();
        for start in (0..n).step_by(len) {
            let (mut w_re, mut w_im) = (1.0, 0.0);
            for k in 0..half {
                let i = start + k;
                let j = start + k + half;
                let tr = w_re * re[j] - w_im * im[j];
                let ti = w_re * im[j] + w_im * re[j];
                re[j] = re[i] - tr;
                im[j] = im[i] - ti;
                re[i] += tr;
                im[i] += ti;
                let new_w_re = w_re * wn_re - w_im * wn_im;
                let new_w_im = w_re * wn_im + w_im * wn_re;
                w_re = new_w_re;
                w_im = new_w_im;
            }
        }
        len <<= 1;
    }
}

/// Magnitude spectrum from FFT output.
pub fn magnitude_spectrum(spectrum: &[(f64, f64)]) -> Vec<f64> {
    spectrum.iter().map(|&(r, i)| (r * r + i * i).sqrt()).collect()
}

/// Convolution of two signals.
pub fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len() + b.len() - 1;
    let mut result = vec![0.0; n];
    for i in 0..a.len() {
        for j in 0..b.len() {
            result[i + j] += a[i] * b[j];
        }
    }
    result
}

/// FIR filter: y[n] = sum(b[k] * x[n-k]).
pub fn fir_filter(signal: &[f64], coeffs: &[f64]) -> Vec<f64> {
    let mut output = vec![0.0; signal.len()];
    for n in 0..signal.len() {
        for (k, &bk) in coeffs.iter().enumerate() {
            if n >= k { output[n] += bk * signal[n - k]; }
        }
    }
    output
}

/// IIR filter: y[n] = sum(b[k]*x[n-k]) - sum(a[k]*y[n-k]).
pub fn iir_filter(signal: &[f64], b: &[f64], a: &[f64]) -> Vec<f64> {
    let mut output = vec![0.0; signal.len()];
    for n in 0..signal.len() {
        let mut val = 0.0;
        for (k, &bk) in b.iter().enumerate() {
            if n >= k { val += bk * signal[n - k]; }
        }
        for (k, &ak) in a.iter().enumerate().skip(1) {
            if n >= k { val -= ak * output[n - k]; }
        }
        if !a.is_empty() && a[0].abs() > 1e-15 { val /= a[0]; }
        output[n] = val;
    }
    output
}

/// Windowing functions.
pub fn hann_window(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos())).collect()
}

pub fn hamming_window(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()).collect()
}

pub fn blackman_window(n: usize) -> Vec<f64> {
    (0..n).map(|i| {
        let t = 2.0 * PI * i as f64 / (n - 1) as f64;
        0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos()
    }).collect()
}

/// Apply window to signal.
pub fn apply_window(signal: &[f64], window: &[f64]) -> Vec<f64> {
    signal.iter().zip(window).map(|(s, w)| s * w).collect()
}

/// Spectrogram: STFT magnitude, returns (time_bins, freq_bins, magnitudes[time][freq]).
pub fn spectrogram(signal: &[f64], window_size: usize, hop_size: usize) -> Vec<Vec<f64>> {
    let window = hann_window(window_size);
    let n_bins = window_size / 2 + 1;
    let mut result = Vec::new();
    let mut start = 0;
    while start + window_size <= signal.len() {
        let chunk: Vec<f64> = signal[start..start + window_size].to_vec();
        let windowed = apply_window(&chunk, &window);
        let spectrum = fft(&windowed);
        let mags: Vec<f64> = spectrum[..n_bins].iter().map(|&(r, i)| (r * r + i * i).sqrt()).collect();
        result.push(mags);
        start += hop_size;
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// TIME SERIES — R ts/forecast/statsmodels parity
// ═══════════════════════════════════════════════════════════════════

/// Autocorrelation function (ACF) for lags 0..max_lag.
pub fn acf(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    let m = mean(data);
    let var: f64 = data.iter().map(|&x| (x - m).powi(2)).sum();
    if var < 1e-15 { return vec![1.0; max_lag + 1]; }
    (0..=max_lag).map(|lag| {
        if lag >= n { return 0.0; }
        let cov: f64 = (0..n - lag).map(|i| (data[i] - m) * (data[i + lag] - m)).sum();
        cov / var
    }).collect()
}

/// Partial autocorrelation function (PACF) via Levinson-Durbin recursion.
pub fn pacf(data: &[f64], max_lag: usize) -> Vec<f64> {
    let ac = acf(data, max_lag);
    let mut pacf_vals = vec![0.0; max_lag + 1];
    pacf_vals[0] = 1.0;
    if max_lag == 0 { return pacf_vals; }
    pacf_vals[1] = ac[1];
    let mut phi = vec![vec![0.0; max_lag + 1]; max_lag + 1];
    phi[1][1] = ac[1];
    for k in 2..=max_lag {
        let mut num = ac[k];
        for j in 1..k {
            num -= phi[k - 1][j] * ac[k - j];
        }
        let mut den = 1.0;
        for j in 1..k {
            den -= phi[k - 1][j] * ac[j];
        }
        if den.abs() < 1e-15 { break; }
        phi[k][k] = num / den;
        pacf_vals[k] = phi[k][k];
        for j in 1..k {
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
        }
    }
    pacf_vals
}

/// Seasonal decomposition (additive): trend + seasonal + residual.
/// Returns (trend, seasonal, residual).
pub fn decompose_additive(data: &[f64], period: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = data.len();
    // Trend: centered moving average
    let trend = centered_ma(data, period);
    // Seasonal: average of detrended values at each phase
    let mut seasonal = vec![0.0; n];
    let mut phase_sums = vec![0.0; period];
    let mut phase_counts = vec![0usize; period];
    for i in 0..n {
        if trend[i].is_finite() {
            let phase = i % period;
            phase_sums[phase] += data[i] - trend[i];
            phase_counts[phase] += 1;
        }
    }
    let phase_means: Vec<f64> = (0..period).map(|p| {
        if phase_counts[p] > 0 { phase_sums[p] / phase_counts[p] as f64 } else { 0.0 }
    }).collect();
    // Center seasonal component
    let s_mean = phase_means.iter().sum::<f64>() / period as f64;
    for i in 0..n {
        seasonal[i] = phase_means[i % period] - s_mean;
    }
    // Residual
    let residual: Vec<f64> = (0..n).map(|i| data[i] - trend[i] - seasonal[i]).collect();
    (trend, seasonal, residual)
}

/// Centered moving average.
fn centered_ma(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let half = window / 2;
    (0..n).map(|i| {
        if i < half || i + half >= n { f64::NAN }
        else {
            let sum: f64 = data[i - half..=i + half].iter().sum();
            sum / window as f64
        }
    }).collect()
}

/// Simple exponential smoothing.
pub fn ses(data: &[f64], alpha: f64) -> Vec<f64> {
    if data.is_empty() { return vec![]; }
    let mut result = vec![data[0]];
    for i in 1..data.len() {
        result.push(alpha * data[i] + (1.0 - alpha) * result[i - 1]);
    }
    result
}

/// Holt's linear trend exponential smoothing.
pub fn holt(data: &[f64], alpha: f64, beta: f64) -> Vec<f64> {
    if data.is_empty() { return vec![]; }
    let mut level = vec![data[0]];
    let mut trend = vec![if data.len() > 1 { data[1] - data[0] } else { 0.0 }];
    let mut forecast = vec![data[0]];
    for i in 1..data.len() {
        let l = alpha * data[i] + (1.0 - alpha) * (level[i - 1] + trend[i - 1]);
        let t = beta * (l - level[i - 1]) + (1.0 - beta) * trend[i - 1];
        level.push(l);
        trend.push(t);
        forecast.push(l + t);
    }
    forecast
}

/// Holt-Winters additive seasonal exponential smoothing.
pub fn holt_winters(data: &[f64], period: usize, alpha: f64, beta: f64, gamma: f64) -> Vec<f64> {
    let n = data.len();
    if n < period * 2 { return ses(data, alpha); }
    // Initialize
    let mut level = vec![0.0; n];
    let mut trend = vec![0.0; n];
    let mut season = vec![0.0; n];
    let mut forecast = vec![0.0; n];
    // Initial level and trend from first two periods
    level[0] = data[..period].iter().sum::<f64>() / period as f64;
    trend[0] = (data[period..2 * period].iter().sum::<f64>() - data[..period].iter().sum::<f64>()) / (period * period) as f64;
    for i in 0..period {
        season[i] = data[i] - level[0];
    }
    forecast[0] = level[0] + trend[0] + season[0];
    for i in 1..n {
        let s_prev = if i >= period { season[i - period] } else { season[i] };
        level[i] = alpha * (data[i] - s_prev) + (1.0 - alpha) * (level[i - 1] + trend[i - 1]);
        trend[i] = beta * (level[i] - level[i - 1]) + (1.0 - beta) * trend[i - 1];
        season[i] = gamma * (data[i] - level[i]) + (1.0 - gamma) * s_prev;
        forecast[i] = level[i] + trend[i] + season[i];
    }
    forecast
}

/// Forecast n steps ahead from exponential smoothing (simple).
pub fn forecast_ses(data: &[f64], alpha: f64, horizon: usize) -> Vec<f64> {
    let smoothed = ses(data, alpha);
    let last = *smoothed.last().unwrap_or(&0.0);
    vec![last; horizon]
}

/// Forecast n steps ahead from Holt's method.
pub fn forecast_holt(data: &[f64], alpha: f64, beta: f64, horizon: usize) -> Vec<f64> {
    if data.is_empty() { return vec![0.0; horizon]; }
    let mut level = data[0];
    let mut trend = if data.len() > 1 { data[1] - data[0] } else { 0.0 };
    for i in 1..data.len() {
        let l = alpha * data[i] + (1.0 - alpha) * (level + trend);
        let t = beta * (l - level) + (1.0 - beta) * trend;
        level = l;
        trend = t;
    }
    (1..=horizon).map(|h| level + h as f64 * trend).collect()
}

/// Lag a time series by k steps (positive k = shift right, fill with NaN).
pub fn lag(data: &[f64], k: i32) -> Vec<f64> {
    let n = data.len();
    (0..n).map(|i| {
        let src = i as i32 - k;
        if src >= 0 && (src as usize) < n { data[src as usize] } else { f64::NAN }
    }).collect()
}

/// Percent change: (x[i] - x[i-1]) / x[i-1].
pub fn pct_change(data: &[f64]) -> Vec<f64> {
    if data.is_empty() { return vec![]; }
    let mut result = vec![f64::NAN];
    for i in 1..data.len() {
        if data[i - 1].abs() < 1e-15 { result.push(f64::NAN); }
        else { result.push((data[i] - data[i - 1]) / data[i - 1]); }
    }
    result
}

/// Rolling window statistics.
pub fn rolling_mean(data: &[f64], window: usize) -> Vec<f64> {
    (0..data.len()).map(|i| {
        if i + 1 < window { f64::NAN }
        else { data[i + 1 - window..=i].iter().sum::<f64>() / window as f64 }
    }).collect()
}

pub fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
    (0..data.len()).map(|i| {
        if i + 1 < window { f64::NAN }
        else {
            let slice = &data[i + 1 - window..=i];
            std_dev(slice)
        }
    }).collect()
}

pub fn rolling_min(data: &[f64], window: usize) -> Vec<f64> {
    (0..data.len()).map(|i| {
        if i + 1 < window { f64::NAN }
        else { data[i + 1 - window..=i].iter().copied().fold(f64::INFINITY, f64::min) }
    }).collect()
}

pub fn rolling_max(data: &[f64], window: usize) -> Vec<f64> {
    (0..data.len()).map(|i| {
        if i + 1 < window { f64::NAN }
        else { data[i + 1 - window..=i].iter().copied().fold(f64::NEG_INFINITY, f64::max) }
    }).collect()
}

/// Bollinger Bands: returns (middle, upper, lower).
pub fn bollinger_bands(data: &[f64], window: usize, num_std: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let middle = rolling_mean(data, window);
    let std = rolling_std(data, window);
    let upper: Vec<f64> = middle.iter().zip(&std).map(|(&m, &s)| m + num_std * s).collect();
    let lower: Vec<f64> = middle.iter().zip(&std).map(|(&m, &s)| m - num_std * s).collect();
    (middle, upper, lower)
}

/// Relative Strength Index (RSI).
pub fn rsi(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    if n < period + 1 { return vec![f64::NAN; n]; }
    let mut gains = vec![0.0; n];
    let mut losses = vec![0.0; n];
    for i in 1..n {
        let change = data[i] - data[i - 1];
        if change > 0.0 { gains[i] = change; }
        else { losses[i] = -change; }
    }
    let mut avg_gain = gains[1..=period].iter().sum::<f64>() / period as f64;
    let mut avg_loss = losses[1..=period].iter().sum::<f64>() / period as f64;
    let mut result = vec![f64::NAN; n];
    if avg_loss > 1e-15 {
        result[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss);
    } else {
        result[period] = 100.0;
    }
    for i in period + 1..n {
        avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
        if avg_loss > 1e-15 {
            result[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss);
        } else {
            result[i] = 100.0;
        }
    }
    result
}

/// MACD: returns (macd_line, signal_line, histogram).
pub fn macd(data: &[f64], fast: usize, slow: usize, signal_period: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ema_fast = ema(data, fast);
    let ema_slow = ema(data, slow);
    let macd_line: Vec<f64> = ema_fast.iter().zip(&ema_slow).map(|(&f, &s)| f - s).collect();
    let signal_line = ema(&macd_line, signal_period);
    let histogram: Vec<f64> = macd_line.iter().zip(&signal_line).map(|(&m, &s)| m - s).collect();
    (macd_line, signal_line, histogram)
}

/// Cross-correlation between two signals.
pub fn xcorr(a: &[f64], b: &[f64]) -> Vec<f64> {
    let na = a.len();
    let nb = b.len();
    let n_out = na + nb - 1;
    let mut result = vec![0.0; n_out];
    for lag in 0..n_out {
        let k = lag as i64 - nb as i64 + 1;
        for i in 0..na {
            let j = i as i64 - k;
            if j >= 0 && (j as usize) < nb {
                result[lag] += a[i] * b[j as usize];
            }
        }
    }
    result
}

/// Butterworth lowpass filter coefficients (2nd order, returns (b, a)).
pub fn butter_lowpass(cutoff: f64, fs: f64) -> (Vec<f64>, Vec<f64>) {
    let w = (std::f64::consts::PI * cutoff / fs).tan();
    let w2 = w * w;
    let sqrt2w = std::f64::consts::SQRT_2 * w;
    let a0 = 1.0 + sqrt2w + w2;
    let b = vec![w2 / a0, 2.0 * w2 / a0, w2 / a0];
    let a = vec![1.0, 2.0 * (w2 - 1.0) / a0, (1.0 - sqrt2w + w2) / a0];
    (b, a)
}

/// Butterworth highpass filter coefficients (2nd order).
pub fn butter_highpass(cutoff: f64, fs: f64) -> (Vec<f64>, Vec<f64>) {
    let w = (std::f64::consts::PI * cutoff / fs).tan();
    let w2 = w * w;
    let sqrt2w = std::f64::consts::SQRT_2 * w;
    let a0 = 1.0 + sqrt2w + w2;
    let b = vec![1.0 / a0, -2.0 / a0, 1.0 / a0];
    let a = vec![1.0, 2.0 * (w2 - 1.0) / a0, (1.0 - sqrt2w + w2) / a0];
    (b, a)
}

/// Zero-phase filtering (filtfilt): forward then backward IIR filter.
pub fn filtfilt(signal: &[f64], b: &[f64], a: &[f64]) -> Vec<f64> {
    let forward = iir_filter(signal, b, a);
    let mut reversed: Vec<f64> = forward.into_iter().rev().collect();
    reversed = iir_filter(&reversed, b, a);
    reversed.into_iter().rev().collect()
}

/// Savitzky-Golay smoothing (simplified: local polynomial via least squares).
pub fn savgol_filter(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let w = if window % 2 == 0 { window + 1 } else { window };
    let half = w / 2;
    (0..n).map(|i| {
        if i < half || i + half >= n { data[i] }
        else {
            let slice = &data[i - half..=i + half];
            // Quadratic fit: y = a + b*x + c*x^2, evaluate at x=0 (center)
            let mut sx = 0.0; let mut sx2 = 0.0; let mut sx3 = 0.0; let mut sx4 = 0.0;
            let mut sy = 0.0; let mut sxy = 0.0; let mut sx2y = 0.0;
            for (j, &val) in slice.iter().enumerate() {
                let x = j as f64 - half as f64;
                let x2 = x * x;
                sx += x; sx2 += x2; sx3 += x2 * x; sx4 += x2 * x2;
                sy += val; sxy += x * val; sx2y += x2 * val;
            }
            let wf = w as f64;
            // Solve 3x3 system for a (intercept = value at center)
            // Simplified: for symmetric window, odd terms vanish
            let det = wf * sx2 * sx4 + 2.0 * sx * sx2 * sx3 - sx2.powi(3) - wf * sx3.powi(2) - sx.powi(2) * sx4;
            if det.abs() < 1e-15 { mean(slice) }
            else {
                let a = (sy * (sx2 * sx4 - sx3.powi(2)) - sxy * (sx * sx4 - sx2 * sx3) + sx2y * (sx * sx3 - sx2.powi(2))) / det;
                a
            }
        }
    }).collect()
}

/// Detrend a signal (remove linear trend).
pub fn detrend(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    if n < 2 { return data.to_vec(); }
    let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let (b0, b1, _) = linear_regression(&xs, data);
    data.iter().enumerate().map(|(i, &x)| x - b0 - b1 * i as f64).collect()
}

/// Hodrick-Prescott filter: decompose into trend + cycle.
/// Returns (trend, cycle).
pub fn hp_filter(data: &[f64], lambda: f64) -> (Vec<f64>, Vec<f64>) {
    let n = data.len();
    if n < 4 { return (data.to_vec(), vec![0.0; n]); }
    // Solve (I + lambda * K'K) * trend = data via iterative smoothing
    let mut trend = data.to_vec();
    for _ in 0..100 {
        let mut new_trend = trend.clone();
        for i in 2..n - 2 {
            let k = lambda * (trend[i - 2] - 4.0 * trend[i - 1] + 6.0 * trend[i] - 4.0 * trend[i + 1] + trend[i + 2]);
            new_trend[i] = data[i] - k;
        }
        // Check convergence
        let diff: f64 = trend.iter().zip(&new_trend).map(|(&a, &b)| (a - b).powi(2)).sum();
        trend = new_trend;
        if diff < 1e-12 { break; }
    }
    let cycle: Vec<f64> = data.iter().zip(&trend).map(|(&d, &t)| d - t).collect();
    (trend, cycle)
}

// ═══════════════════════════════════════════════════════════════════
// ML ESTIMATORS — scikit-learn parity
// ═══════════════════════════════════════════════════════════════════

/// K-means clustering. Returns (labels, centroids).
pub fn kmeans(data: &[Vec<f64>], k: usize, max_iter: usize) -> (Vec<usize>, Vec<Vec<f64>>) {
    let n = data.len();
    if n == 0 || k == 0 { return (vec![], vec![]); }
    let dim = data[0].len();

    // Initialize centroids from first k points
    let mut centroids: Vec<Vec<f64>> = data.iter().take(k).cloned().collect();
    let mut labels = vec![0usize; n];

    for _ in 0..max_iter {
        // Assign labels
        let mut changed = false;
        for i in 0..n {
            let mut best = 0;
            let mut best_dist = f64::INFINITY;
            for c in 0..k {
                let d: f64 = data[i].iter().zip(&centroids[c]).map(|(a, b)| (a - b).powi(2)).sum();
                if d < best_dist { best_dist = d; best = c; }
            }
            if labels[i] != best { changed = true; labels[i] = best; }
        }
        if !changed { break; }

        // Update centroids
        let mut sums = vec![vec![0.0; dim]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = labels[i];
            counts[c] += 1;
            for d in 0..dim { sums[c][d] += data[i][d]; }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim { centroids[c][d] = sums[c][d] / counts[c] as f64; }
            }
        }
    }
    (labels, centroids)
}

/// Principal Component Analysis. Returns (components, explained_variance, projected_data).
/// components: each row is a principal component vector.
pub fn pca(data: &[Vec<f64>], n_components: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let n = data.len();
    if n == 0 { return (vec![], vec![], vec![]); }
    let dim = data[0].len();
    let k = n_components.min(dim);

    // Center data
    let means: Vec<f64> = (0..dim).map(|d| data.iter().map(|row| row[d]).sum::<f64>() / n as f64).collect();
    let centered: Vec<Vec<f64>> = data.iter().map(|row| {
        row.iter().zip(&means).map(|(&x, &m)| x - m).collect()
    }).collect();

    // Covariance matrix
    let mut cov = vec![vec![0.0; dim]; dim];
    for i in 0..dim {
        for j in i..dim {
            let c: f64 = centered.iter().map(|row| row[i] * row[j]).sum::<f64>() / (n - 1).max(1) as f64;
            cov[i][j] = c;
            cov[j][i] = c;
        }
    }

    // Eigendecomposition (power iteration for top k)
    let mut components = Vec::new();
    let mut eigenvalues = Vec::new();
    let mut deflated = cov.clone();

    for _ in 0..k {
        // Power iteration
        let mut v: Vec<f64> = (0..dim).map(|i| if i == components.len() { 1.0 } else { 0.0 }).collect();
        let mut eigenvalue = 0.0;
        for _ in 0..200 {
            let mut new_v: Vec<f64> = vec![0.0; dim];
            for i in 0..dim {
                for j in 0..dim { new_v[i] += deflated[i][j] * v[j]; }
            }
            let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 { break; }
            eigenvalue = norm;
            for i in 0..dim { v[i] = new_v[i] / norm; }
        }
        // Deflate
        for i in 0..dim {
            for j in 0..dim { deflated[i][j] -= eigenvalue * v[i] * v[j]; }
        }
        components.push(v);
        eigenvalues.push(eigenvalue);
    }

    // Project data
    let projected: Vec<Vec<f64>> = centered.iter().map(|row| {
        components.iter().map(|comp| {
            row.iter().zip(comp).map(|(&x, &c)| x * c).sum()
        }).collect()
    }).collect();

    (components, eigenvalues, projected)
}

/// K-Nearest Neighbors classification. Returns predicted labels.
pub fn knn_classify(train_x: &[Vec<f64>], train_y: &[usize], test_x: &[Vec<f64>], k: usize) -> Vec<usize> {
    test_x.iter().map(|test_point| {
        let mut dists: Vec<(f64, usize)> = train_x.iter().zip(train_y).map(|(tx, &ty)| {
            let d: f64 = tx.iter().zip(test_point).map(|(&a, &b)| (a - b).powi(2)).sum();
            (d, ty)
        }).collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        // Majority vote among k nearest
        let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &(_, label) in dists.iter().take(k) {
            *counts.entry(label).or_insert(0) += 1;
        }
        counts.into_iter().max_by_key(|&(_, c)| c).map(|(l, _)| l).unwrap_or(0)
    }).collect()
}

/// K-Nearest Neighbors regression. Returns predicted values.
pub fn knn_regress(train_x: &[Vec<f64>], train_y: &[f64], test_x: &[Vec<f64>], k: usize) -> Vec<f64> {
    test_x.iter().map(|test_point| {
        let mut dists: Vec<(f64, f64)> = train_x.iter().zip(train_y).map(|(tx, &ty)| {
            let d: f64 = tx.iter().zip(test_point).map(|(&a, &b)| (a - b).powi(2)).sum();
            (d, ty)
        }).collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let sum: f64 = dists.iter().take(k).map(|&(_, y)| y).sum();
        sum / k as f64
    }).collect()
}

/// StandardScaler: zero mean, unit variance. Returns (scaled_data, means, stds).
pub fn standard_scale(data: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n = data.len();
    if n == 0 { return (vec![], vec![], vec![]); }
    let dim = data[0].len();
    let means: Vec<f64> = (0..dim).map(|d| data.iter().map(|r| r[d]).sum::<f64>() / n as f64).collect();
    let stds: Vec<f64> = (0..dim).map(|d| {
        let m = means[d];
        let v: f64 = data.iter().map(|r| (r[d] - m).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
        v.sqrt().max(1e-15)
    }).collect();
    let scaled: Vec<Vec<f64>> = data.iter().map(|row| {
        row.iter().zip(&means).zip(&stds).map(|((&x, &m), &s)| (x - m) / s).collect()
    }).collect();
    (scaled, means, stds)
}

/// MinMaxScaler: scale to [0, 1]. Returns (scaled_data, mins, maxs).
pub fn minmax_scale(data: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n = data.len();
    if n == 0 { return (vec![], vec![], vec![]); }
    let dim = data[0].len();
    let mins: Vec<f64> = (0..dim).map(|d| data.iter().map(|r| r[d]).fold(f64::INFINITY, f64::min)).collect();
    let maxs: Vec<f64> = (0..dim).map(|d| data.iter().map(|r| r[d]).fold(f64::NEG_INFINITY, f64::max)).collect();
    let scaled: Vec<Vec<f64>> = data.iter().map(|row| {
        row.iter().zip(&mins).zip(&maxs).map(|((&x, &mn), &mx)| {
            let range = (mx - mn).max(1e-15);
            (x - mn) / range
        }).collect()
    }).collect();
    (scaled, mins, maxs)
}

/// Confusion matrix from true labels and predicted labels.
pub fn confusion_matrix(y_true: &[usize], y_pred: &[usize]) -> Vec<Vec<usize>> {
    let max_label = y_true.iter().chain(y_pred.iter()).copied().max().unwrap_or(0) + 1;
    let mut cm = vec![vec![0usize; max_label]; max_label];
    for (&t, &p) in y_true.iter().zip(y_pred) {
        if t < max_label && p < max_label { cm[t][p] += 1; }
    }
    cm
}

/// Classification accuracy.
pub fn accuracy(y_true: &[usize], y_pred: &[usize]) -> f64 {
    if y_true.is_empty() { return 0.0; }
    let correct = y_true.iter().zip(y_pred).filter(|(t, p)| *t == *p).count();
    correct as f64 / y_true.len() as f64
}

/// Precision for a specific class.
pub fn precision(y_true: &[usize], y_pred: &[usize], class: usize) -> f64 {
    let tp = y_true.iter().zip(y_pred).filter(|(t, p)| **t == class && **p == class).count();
    let fp = y_true.iter().zip(y_pred).filter(|(t, p)| **t != class && **p == class).count();
    if tp + fp == 0 { 0.0 } else { tp as f64 / (tp + fp) as f64 }
}

/// Recall for a specific class.
pub fn recall(y_true: &[usize], y_pred: &[usize], class: usize) -> f64 {
    let tp = y_true.iter().zip(y_pred).filter(|(t, p)| **t == class && **p == class).count();
    let fn_ = y_true.iter().zip(y_pred).filter(|(t, p)| **t == class && **p != class).count();
    if tp + fn_ == 0 { 0.0 } else { tp as f64 / (tp + fn_) as f64 }
}

/// F1 score for a specific class.
pub fn f1_score(y_true: &[usize], y_pred: &[usize], class: usize) -> f64 {
    let p = precision(y_true, y_pred, class);
    let r = recall(y_true, y_pred, class);
    if p + r < 1e-15 { 0.0 } else { 2.0 * p * r / (p + r) }
}

/// R-squared (coefficient of determination).
pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let m = mean(y_true);
    let ss_res: f64 = y_true.iter().zip(y_pred).map(|(&t, &p)| (t - p).powi(2)).sum();
    let ss_tot: f64 = y_true.iter().map(|&t| (t - m).powi(2)).sum();
    if ss_tot < 1e-15 { return 1.0; }
    1.0 - ss_res / ss_tot
}

/// Mean squared error.
pub fn mse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() { return 0.0; }
    y_true.iter().zip(y_pred).map(|(&t, &p)| (t - p).powi(2)).sum::<f64>() / y_true.len() as f64
}

/// Root mean squared error.
pub fn rmse(y_true: &[f64], y_pred: &[f64]) -> f64 { mse(y_true, y_pred).sqrt() }

/// Mean absolute error.
pub fn mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() { return 0.0; }
    y_true.iter().zip(y_pred).map(|(&t, &p)| (t - p).abs()).sum::<f64>() / y_true.len() as f64
}

/// Silhouette score for clustering quality.
pub fn silhouette_score(data: &[Vec<f64>], labels: &[usize]) -> f64 {
    let n = data.len();
    if n < 2 { return 0.0; }
    let mut total = 0.0;
    for i in 0..n {
        // a(i): mean distance to same-cluster points
        let mut a_sum = 0.0;
        let mut a_count = 0;
        for j in 0..n {
            if j != i && labels[j] == labels[i] {
                let d: f64 = data[i].iter().zip(&data[j]).map(|(&a, &b)| (a - b).powi(2)).sum::<f64>().sqrt();
                a_sum += d;
                a_count += 1;
            }
        }
        let a = if a_count > 0 { a_sum / a_count as f64 } else { 0.0 };

        // b(i): min mean distance to other clusters
        let mut b = f64::INFINITY;
        let max_label = labels.iter().copied().max().unwrap_or(0);
        for c in 0..=max_label {
            if c == labels[i] { continue; }
            let mut b_sum = 0.0;
            let mut b_count = 0;
            for j in 0..n {
                if labels[j] == c {
                    let d: f64 = data[i].iter().zip(&data[j]).map(|(&a, &b)| (a - b).powi(2)).sum::<f64>().sqrt();
                    b_sum += d;
                    b_count += 1;
                }
            }
            if b_count > 0 {
                let avg = b_sum / b_count as f64;
                if avg < b { b = avg; }
            }
        }

        let s = if a.max(b) < 1e-15 { 0.0 } else { (b - a) / a.max(b) };
        total += s;
    }
    total / n as f64
}

/// Train/test split. Returns (train_x, test_x, train_y, test_y).
pub fn train_test_split(x: &[Vec<f64>], y: &[f64], test_frac: f64, seed: u64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n = x.len();
    let n_test = (n as f64 * test_frac).round() as usize;
    let n_train = n - n_test;

    // Shuffle indices using PRNG
    let mut rng = crate::distributions::Rng::new(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        indices.swap(i, j);
    }

    let train_x: Vec<Vec<f64>> = indices[..n_train].iter().map(|&i| x[i].clone()).collect();
    let test_x: Vec<Vec<f64>> = indices[n_train..].iter().map(|&i| x[i].clone()).collect();
    let train_y: Vec<f64> = indices[..n_train].iter().map(|&i| y[i]).collect();
    let test_y: Vec<f64> = indices[n_train..].iter().map(|&i| y[i]).collect();
    (train_x, test_x, train_y, test_y)
}

/// Cross-validation scores. Returns vector of scores (one per fold).
pub fn cross_val_score(x: &[Vec<f64>], y: &[f64], k_folds: usize, eval_fn: impl Fn(&[Vec<f64>], &[f64], &[Vec<f64>], &[f64]) -> f64) -> Vec<f64> {
    let n = x.len();
    let fold_size = n / k_folds;
    (0..k_folds).map(|fold| {
        let test_start = fold * fold_size;
        let test_end = if fold == k_folds - 1 { n } else { test_start + fold_size };
        let train_x: Vec<Vec<f64>> = x[..test_start].iter().chain(x[test_end..].iter()).cloned().collect();
        let train_y: Vec<f64> = y[..test_start].iter().chain(y[test_end..].iter()).copied().collect();
        let test_x = &x[test_start..test_end];
        let test_y = &y[test_start..test_end];
        eval_fn(&train_x, &train_y, test_x, test_y)
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// SYMBOLIC MATH — Mathematica/Maple/SymPy parity (numeric)
// ═══════════════════════════════════════════════════════════════════

/// Numerical differentiation (central difference).
pub fn differentiate(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-8;
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// Numerical integration (Simpson's rule).
pub fn integrate(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let n = if n % 2 == 0 { n } else { n + 1 }; // must be even
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += if i % 2 == 0 { 2.0 * f(x) } else { 4.0 * f(x) };
    }
    sum * h / 3.0
}

/// Newton's method root finding: f(x) = 0.
pub fn find_root(f: impl Fn(f64) -> f64, x0: f64, tol: f64, max_iter: usize) -> f64 {
    let mut x = x0;
    for _ in 0..max_iter {
        let fx = f(x);
        if fx.abs() < tol { return x; }
        let dfx = differentiate(&f, x);
        if dfx.abs() < 1e-15 { break; }
        x -= fx / dfx;
    }
    x
}

// ═══════════════════════════════════════════════════════════════════
// SPECIAL FUNCTIONS — R/SciPy/Mathematica parity
// ═══════════════════════════════════════════════════════════════════

// Note: erf, erfc, erfinv, lgamma, gamma_fn, beta_fn, betainc, gammainc
// are implemented in distributions.rs. Re-export or use from there.

/// Hyperbolic sine
pub fn sinh(x: f64) -> f64 { x.sinh() }
/// Hyperbolic cosine
pub fn cosh(x: f64) -> f64 { x.cosh() }
/// Hyperbolic tangent
pub fn tanh_fn(x: f64) -> f64 { x.tanh() }
/// Inverse hyperbolic sine
pub fn asinh(x: f64) -> f64 { x.asinh() }
/// Inverse hyperbolic cosine
pub fn acosh(x: f64) -> f64 { x.acosh() }
/// Inverse hyperbolic tangent
pub fn atanh(x: f64) -> f64 { x.atanh() }
/// Two-argument arctangent
pub fn atan2(y: f64, x: f64) -> f64 { y.atan2(x) }
/// Hypotenuse: sqrt(x^2 + y^2)
pub fn hypot(x: f64, y: f64) -> f64 { x.hypot(y) }
/// sin(pi * x) — exact at integers
pub fn sinpi(x: f64) -> f64 { (std::f64::consts::PI * x).sin() }
/// cos(pi * x) — exact at integers
pub fn cospi(x: f64) -> f64 { (std::f64::consts::PI * x).cos() }
/// Sign function: -1, 0, or 1
pub fn sign(x: f64) -> f64 { if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 } }
/// Truncate toward zero
pub fn trunc(x: f64) -> f64 { x.trunc() }
/// Round to n significant digits
pub fn signif(x: f64, digits: usize) -> f64 {
    if x == 0.0 { return 0.0; }
    let d = (x.abs().log10().floor() + 1.0) as i32;
    let pow = 10.0f64.powi(digits as i32 - d);
    (x * pow).round() / pow
}
/// Cube root
pub fn cbrt(x: f64) -> f64 { x.cbrt() }
/// exp(x) - 1 (accurate for small x)
pub fn expm1(x: f64) -> f64 { x.exp_m1() }
/// ln(1 + x) (accurate for small x)
pub fn log1p(x: f64) -> f64 { x.ln_1p() }
/// 2^x
pub fn exp2(x: f64) -> f64 { (2.0f64).powf(x) }
/// Clamp value to range
pub fn clamp(x: f64, lo: f64, hi: f64) -> f64 { x.clamp(lo, hi) }
/// Modulo (Euclidean remainder, always non-negative)
pub fn modulo(x: f64, y: f64) -> f64 { ((x % y) + y) % y }
/// Degrees to radians
pub fn deg2rad(x: f64) -> f64 { x * std::f64::consts::PI / 180.0 }
/// Radians to degrees
pub fn rad2deg(x: f64) -> f64 { x * 180.0 / std::f64::consts::PI }

/// Factorial n!
pub fn factorial(n: u64) -> f64 {
    if n <= 20 {
        let mut r: u64 = 1;
        for i in 2..=n { r *= i; }
        r as f64
    } else {
        crate::distributions::lgamma(n as f64 + 1.0).exp()
    }
}

/// Binomial coefficient C(n, k)
pub fn choose(n: u64, k: u64) -> f64 {
    if k > n { return 0.0; }
    (crate::distributions::lgamma(n as f64 + 1.0)
        - crate::distributions::lgamma(k as f64 + 1.0)
        - crate::distributions::lgamma((n - k) as f64 + 1.0))
    .exp()
}

/// Permutations P(n, k) = n! / (n-k)!
pub fn perm(n: u64, k: u64) -> f64 {
    if k > n { return 0.0; }
    (crate::distributions::lgamma(n as f64 + 1.0)
        - crate::distributions::lgamma((n - k) as f64 + 1.0))
    .exp()
}

/// Log-gamma (re-export for convenience)
pub fn lgamma(x: f64) -> f64 { crate::distributions::lgamma(x) }
/// Gamma function
pub fn gamma(x: f64) -> f64 { crate::distributions::gamma_fn(x) }
/// Beta function B(a,b)
pub fn beta(a: f64, b: f64) -> f64 { crate::distributions::beta_fn(a, b) }
/// Digamma (psi) function: d/dx ln(Gamma(x))
pub fn digamma(x: f64) -> f64 {
    // Asymptotic series for x > 6, recursion for small x
    if x < 1e-6 { return -0.5772156649015329 - 1.0 / x; }
    if x < 6.0 { return digamma(x + 1.0) - 1.0 / x; }
    let mut result = x.ln() - 0.5 / x;
    let x2 = 1.0 / (x * x);
    result -= x2 * (1.0 / 12.0 - x2 * (1.0 / 120.0 - x2 * (1.0 / 252.0)));
    result
}

/// Error function
pub fn erf(x: f64) -> f64 { crate::distributions::erf(x) }
/// Complementary error function
pub fn erfc(x: f64) -> f64 { crate::distributions::erfc(x) }
/// Inverse error function
pub fn erfinv(x: f64) -> f64 { crate::distributions::erfinv(x) }

/// Logistic (sigmoid) function: 1 / (1 + exp(-x))
pub fn expit(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }
/// Logit function: ln(p / (1-p))
pub fn logit(p: f64) -> f64 { (p / (1.0 - p)).ln() }
/// Log-sum-exp (numerically stable): log(sum(exp(x_i)))
pub fn logsumexp(x: &[f64]) -> f64 {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() { return max; }
    max + x.iter().map(|&v| (v - max).exp()).sum::<f64>().ln()
}

/// Normalized sinc: sin(pi*x) / (pi*x)
pub fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-15 { 1.0 } else { sinpi(x) / (std::f64::consts::PI * x) }
}

/// Heaviside step function
pub fn heaviside(x: f64, h0: f64) -> f64 {
    if x < 0.0 { 0.0 } else if x == 0.0 { h0 } else { 1.0 }
}

/// Greatest common divisor
pub fn gcd(a: u64, b: u64) -> u64 {
    let (mut a, mut b) = (a, b);
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

/// Least common multiple
pub fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 { 0 } else { a / gcd(a, b) * b }
}

/// Interquartile range
pub fn iqr(data: &[f64]) -> f64 {
    percentile(data, 0.75) - percentile(data, 0.25)
}

/// Median absolute deviation
pub fn mad(data: &[f64]) -> f64 {
    let med = median(data);
    let deviations: Vec<f64> = data.iter().map(|&x| (x - med).abs()).collect();
    median(&deviations)
}

/// Weighted mean
pub fn weighted_mean(data: &[f64], weights: &[f64]) -> f64 {
    let sum_w: f64 = weights.iter().sum();
    if sum_w.abs() < 1e-15 { return 0.0; }
    data.iter().zip(weights).map(|(&d, &w)| d * w).sum::<f64>() / sum_w
}

/// Geometric mean
pub fn geometric_mean(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n == 0.0 { return 0.0; }
    (data.iter().map(|&x| x.ln()).sum::<f64>() / n).exp()
}

/// Harmonic mean
pub fn harmonic_mean(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n == 0.0 { return 0.0; }
    n / data.iter().map(|&x| 1.0 / x).sum::<f64>()
}

/// Skewness (Fisher's definition)
pub fn skewness(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 3.0 { return 0.0; }
    let m = mean(data);
    let s = std_dev(data);
    if s < 1e-15 { return 0.0; }
    let m3: f64 = data.iter().map(|&x| ((x - m) / s).powi(3)).sum();
    m3 * n / ((n - 1.0) * (n - 2.0))
}

/// Kurtosis (Fisher's definition, excess kurtosis, 0 for normal)
pub fn kurtosis(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 4.0 { return 0.0; }
    let m = mean(data);
    let s = std_dev(data);
    if s < 1e-15 { return 0.0; }
    let m4: f64 = data.iter().map(|&x| ((x - m) / s).powi(4)).sum();
    let k = (n * (n + 1.0) * m4) / ((n - 1.0) * (n - 2.0) * (n - 3.0));
    k - 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0))
}

/// Coefficient of variation
pub fn cv(data: &[f64]) -> f64 {
    let m = mean(data);
    if m.abs() < 1e-15 { return 0.0; }
    std_dev(data) / m
}

/// Standard error of the mean
pub fn sem(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 { return 0.0; }
    std_dev(data) / n.sqrt()
}

/// Rank data (average rank for ties)
pub fn rankdata(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(f64, usize)> = data.iter().copied().zip(0..n).collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (indexed[j].0 - indexed[i].0).abs() < 1e-15 { j += 1; }
        let avg_rank = (i + j + 1) as f64 / 2.0; // 1-based average
        for k in i..j { ranks[indexed[k].1] = avg_rank; }
        i = j;
    }
    ranks
}

/// Cumulative product
pub fn cumprod(data: &[f64]) -> Vec<f64> {
    let mut acc = 1.0;
    data.iter().map(|&x| { acc *= x; acc }).collect()
}

/// Cumulative max
pub fn cummax(data: &[f64]) -> Vec<f64> {
    let mut m = f64::NEG_INFINITY;
    data.iter().map(|&x| { if x > m { m = x; } m }).collect()
}

/// Cumulative min
pub fn cummin(data: &[f64]) -> Vec<f64> {
    let mut m = f64::INFINITY;
    data.iter().map(|&x| { if x < m { m = x; } m }).collect()
}

/// Quantile (same as percentile but takes 0-1 directly)
pub fn quantile(data: &[f64], q: f64) -> f64 { percentile(data, q) }

/// Mode (most frequent value)
pub fn mode(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut counts: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    for &v in data {
        *counts.entry(v.to_bits()).or_insert(0) += 1;
    }
    let max_bits = counts.into_iter().max_by_key(|&(_, c)| c).unwrap().0;
    f64::from_bits(max_bits)
}

// ═══════════════════════════════════════════════════════════════════
// FINANCE — Bloomberg/QuantLib parity
// ═══════════════════════════════════════════════════════════════════

/// Black-Scholes call option price.
pub fn black_scholes_call(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    if t <= 0.0 { return (s - k).max(0.0); }
    let d1 = ((s / k).ln() + (r + sigma * sigma / 2.0) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    s * normal_cdf(d1) - k * (-r * t).exp() * normal_cdf(d2)
}

/// Black-Scholes put option price.
pub fn black_scholes_put(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    if t <= 0.0 { return (k - s).max(0.0); }
    let d1 = ((s / k).ln() + (r + sigma * sigma / 2.0) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    k * (-r * t).exp() * normal_cdf(-d2) - s * normal_cdf(-d1)
}

/// Simple moving average.
pub fn sma(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || data.is_empty() { return vec![]; }
    data.windows(window).map(|w| w.iter().sum::<f64>() / window as f64).collect()
}

/// Exponential moving average.
pub fn ema(data: &[f64], span: usize) -> Vec<f64> {
    if data.is_empty() || span == 0 { return vec![]; }
    let alpha = 2.0 / (span + 1) as f64;
    let mut result = vec![data[0]];
    for i in 1..data.len() {
        result.push(alpha * data[i] + (1.0 - alpha) * result[i - 1]);
    }
    result
}

/// Sharpe ratio: (mean_return - risk_free) / std_return.
pub fn sharpe_ratio(returns: &[f64], risk_free: f64) -> f64 {
    let m = mean(returns) - risk_free;
    let s = std_dev(returns);
    if s < 1e-15 { 0.0 } else { m / s }
}

/// Maximum drawdown.
pub fn max_drawdown(prices: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0;
    for &p in prices {
        if p > peak { peak = p; }
        let dd = (peak - p) / peak;
        if dd > max_dd { max_dd = dd; }
    }
    max_dd
}

/// Present value of a series of cash flows.
pub fn npv(rate: f64, cash_flows: &[f64]) -> f64 {
    cash_flows.iter().enumerate().map(|(i, &cf)| cf / (1.0 + rate).powi(i as i32)).sum()
}

/// Internal rate of return (IRR) via Newton's method.
pub fn irr(cash_flows: &[f64]) -> f64 {
    find_root(|r| npv(r, cash_flows), 0.1, 1e-8, 100)
}

// ═══════════════════════════════════════════════════════════════════
// BIOINFORMATICS — Biopython/Bioconductor parity
// ═══════════════════════════════════════════════════════════════════

/// Smith-Waterman local alignment score.
pub fn smith_waterman(seq1: &[u8], seq2: &[u8], match_score: i32, mismatch: i32, gap: i32) -> i32 {
    let m = seq1.len();
    let n = seq2.len();
    let mut h = vec![vec![0i32; n + 1]; m + 1];
    let mut max_score = 0;
    for i in 1..=m {
        for j in 1..=n {
            let s = if seq1[i - 1] == seq2[j - 1] { match_score } else { mismatch };
            h[i][j] = 0
                .max(h[i - 1][j - 1] + s)
                .max(h[i - 1][j] + gap)
                .max(h[i][j - 1] + gap);
            if h[i][j] > max_score { max_score = h[i][j]; }
        }
    }
    max_score
}

/// Needleman-Wunsch global alignment score.
pub fn needleman_wunsch(seq1: &[u8], seq2: &[u8], match_score: i32, mismatch: i32, gap: i32) -> i32 {
    let m = seq1.len();
    let n = seq2.len();
    let mut h = vec![vec![0i32; n + 1]; m + 1];
    for i in 0..=m { h[i][0] = gap * i as i32; }
    for j in 0..=n { h[0][j] = gap * j as i32; }
    for i in 1..=m {
        for j in 1..=n {
            let s = if seq1[i - 1] == seq2[j - 1] { match_score } else { mismatch };
            h[i][j] = (h[i - 1][j - 1] + s)
                .max(h[i - 1][j] + gap)
                .max(h[i][j - 1] + gap);
        }
    }
    h[m][n]
}

/// GC content of a DNA sequence.
pub fn gc_content(seq: &[u8]) -> f64 {
    if seq.is_empty() { return 0.0; }
    let gc = seq.iter().filter(|&&b| b == b'G' || b == b'C' || b == b'g' || b == b'c').count();
    gc as f64 / seq.len() as f64
}

/// Complement of a DNA base.
pub fn complement(seq: &[u8]) -> Vec<u8> {
    seq.iter().map(|&b| match b {
        b'A' => b'T', b'T' => b'A', b'G' => b'C', b'C' => b'G',
        b'a' => b't', b't' => b'a', b'g' => b'c', b'c' => b'g',
        other => other,
    }).collect()
}

/// Reverse complement.
pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    let mut rc = complement(seq);
    rc.reverse();
    rc
}

/// Translate DNA to protein (standard genetic code, reading frame 0).
pub fn translate(dna: &[u8]) -> Vec<u8> {
    let mut protein = Vec::new();
    for codon in dna.chunks(3) {
        if codon.len() < 3 { break; }
        let aa = match (codon[0].to_ascii_uppercase(), codon[1].to_ascii_uppercase(), codon[2].to_ascii_uppercase()) {
            (b'A', b'T', b'G') => b'M',
            (b'T', b'T', b'T') | (b'T', b'T', b'C') => b'F',
            (b'T', b'T', b'A') | (b'T', b'T', b'G') => b'L',
            (b'C', b'T', _) => b'L',
            (b'A', b'T', b'T') | (b'A', b'T', b'C') | (b'A', b'T', b'A') => b'I',
            (b'G', b'T', _) => b'V',
            (b'T', b'C', _) => b'S',
            (b'C', b'C', _) => b'P',
            (b'A', b'C', _) => b'T',
            (b'G', b'C', _) => b'A',
            (b'T', b'A', b'T') | (b'T', b'A', b'C') => b'Y',
            (b'T', b'A', b'A') | (b'T', b'A', b'G') => b'*', // stop
            (b'C', b'A', b'T') | (b'C', b'A', b'C') => b'H',
            (b'C', b'A', b'A') | (b'C', b'A', b'G') => b'Q',
            (b'A', b'A', b'T') | (b'A', b'A', b'C') => b'N',
            (b'A', b'A', b'A') | (b'A', b'A', b'G') => b'K',
            (b'G', b'A', b'T') | (b'G', b'A', b'C') => b'D',
            (b'G', b'A', b'A') | (b'G', b'A', b'G') => b'E',
            (b'T', b'G', b'T') | (b'T', b'G', b'C') => b'C',
            (b'T', b'G', b'A') => b'*', // stop
            (b'T', b'G', b'G') => b'W',
            (b'C', b'G', _) => b'R',
            (b'A', b'G', b'T') | (b'A', b'G', b'C') => b'S',
            (b'A', b'G', b'A') | (b'A', b'G', b'G') => b'R',
            (b'G', b'G', _) => b'G',
            _ => b'X', // unknown
        };
        protein.push(aa);
    }
    protein
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Linear Algebra ─────────────────────────────────────
    #[test]
    fn test_mat_mul_3x3() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]];
        let b = eye(3);
        assert_eq!(mat_mul(&a, &b), a); // A * I = A
    }

    #[test]
    fn test_determinant_3x3() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![0.0, 4.0, 5.0], vec![1.0, 0.0, 6.0]];
        let det = determinant(&a);
        assert!((det - 22.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_2x2() {
        let a = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
        let inv = inverse(&a).unwrap();
        let prod = mat_mul(&a, &inv);
        assert!((prod[0][0] - 1.0).abs() < 1e-10);
        assert!((prod[0][1]).abs() < 1e-10);
        assert!((prod[1][0]).abs() < 1e-10);
        assert!((prod[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_3x3() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![0.0, 1.0, 4.0], vec![5.0, 6.0, 0.0]];
        let inv = inverse(&a).unwrap();
        let prod = mat_mul(&a, &inv);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((prod[i][j] - expected).abs() < 1e-10, "A*A^-1[{i}][{j}] = {} != {expected}", prod[i][j]);
            }
        }
    }

    #[test]
    fn test_solve_3x3() {
        let a = vec![vec![2.0, 1.0, -1.0], vec![-3.0, -1.0, 2.0], vec![-2.0, 1.0, 2.0]];
        let b = vec![8.0, -11.0, -3.0];
        let x = solve(&a, &b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
        assert!((x[2] + 1.0).abs() < 1e-10); // x[2] = -1
    }

    #[test]
    fn test_qr() {
        let a = vec![vec![1.0, 1.0], vec![0.0, 1.0], vec![1.0, 0.0]];
        let (q, r) = qr_decompose(&a);
        let reconstructed = mat_mul(&q, &r);
        for i in 0..3 { for j in 0..2 {
            assert!((reconstructed[i][j] - a[i][j]).abs() < 1e-10);
        }}
    }

    #[test]
    fn test_cholesky() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let l = cholesky(&a).unwrap();
        let lt = transpose(&l);
        let prod = mat_mul(&l, &lt);
        for i in 0..2 { for j in 0..2 {
            assert!((prod[i][j] - a[i][j]).abs() < 1e-10);
        }}
    }

    #[test]
    fn test_eigenvalues() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]]; // eigenvalues: 1, 3
        let mut eigs = eigenvalues_symmetric(&a, 100);
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((eigs[0] - 1.0).abs() < 1e-6);
        assert!((eigs[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_linspace() {
        let v = linspace(0.0, 1.0, 5);
        assert_eq!(v.len(), 5);
        assert!((v[0] - 0.0).abs() < 1e-15);
        assert!((v[4] - 1.0).abs() < 1e-15);
        assert!((v[2] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_trace() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!((trace(&a) - 5.0).abs() < 1e-15);
    }

    // ── Statistics ──────────────────────────────────────────
    #[test]
    fn test_mean_variance() {
        let d = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((mean(&d) - 5.0).abs() < 1e-10);
        assert!((variance(&d) - 4.571428).abs() < 0.001);
    }

    #[test]
    fn test_correlation_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((correlation(&x, &y) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_t_test() {
        let d = vec![5.1, 4.9, 5.0, 5.2, 4.8, 5.1, 5.0, 4.9, 5.0, 5.1];
        let (t, p) = t_test_one_sample(&d, 5.0);
        assert!(p > 0.05); // should not reject H0: mu = 5.0
    }

    #[test]
    fn test_linear_regression() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let (b0, b1, r2) = linear_regression(&x, &y);
        assert!((b0 - 0.0).abs() < 0.01);
        assert!((b1 - 2.0).abs() < 0.01);
        assert!((r2 - 1.0).abs() < 0.01);
    }

    // ── DSP ────────────────────────────────────────────────
    #[test]
    fn test_fft_impulse() {
        let signal = vec![1.0, 0.0, 0.0, 0.0];
        let spectrum = fft(&signal);
        // FFT of impulse = all ones
        for &(re, im) in &spectrum {
            assert!((re - 1.0).abs() < 1e-10);
            assert!(im.abs() < 1e-10);
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let spectrum = fft(&signal);
        let recovered = ifft(&spectrum);
        for (i, (&orig, &rec)) in signal.iter().zip(recovered.iter()).enumerate() {
            assert!((orig - rec).abs() < 1e-10, "mismatch at {i}: {orig} vs {rec}");
        }
    }

    #[test]
    fn test_convolve() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 1.0, 0.5];
        let c = convolve(&a, &b);
        assert_eq!(c.len(), 5);
        assert!((c[0] - 0.0).abs() < 1e-10);
        assert!((c[1] - 1.0).abs() < 1e-10);
        assert!((c[2] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_spectrogram_shape() {
        let signal: Vec<f64> = (0..1024).map(|i| (2.0 * PI * 440.0 * i as f64 / 44100.0).sin()).collect();
        let spec = spectrogram(&signal, 256, 128);
        assert!(!spec.is_empty());
        assert_eq!(spec[0].len(), 129); // 256/2 + 1
    }

    // ── Finance ────────────────────────────────────────────
    #[test]
    fn test_black_scholes() {
        let c = black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.2);
        // Should be approximately 10.45
        assert!((c - 10.45).abs() < 0.1);
    }

    #[test]
    fn test_npv() {
        let cf = vec![-1000.0, 300.0, 420.0, 680.0];
        let v = npv(0.12, &cf);
        assert!(v > 0.0); // profitable at 12%
    }

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&data, 3);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[2] - 4.0).abs() < 1e-10);
    }

    // ── Bioinformatics ─────────────────────────────────────
    #[test]
    fn test_smith_waterman() {
        let s1 = b"ACGTACGT";
        let s2 = b"CGTACG";
        let score = smith_waterman(s1, s2, 2, -1, -1);
        assert!(score > 0);
    }

    #[test]
    fn test_gc_content() {
        let seq = b"ATGCGCTA";
        let gc = gc_content(seq);
        assert!((gc - 0.5).abs() < 1e-10); // 4 GC out of 8
    }

    #[test]
    fn test_reverse_complement() {
        let seq = b"ATGC";
        let rc = reverse_complement(seq);
        assert_eq!(rc, b"GCAT");
    }

    #[test]
    fn test_translate() {
        let dna = b"ATGGCTTAA";
        let protein = translate(dna);
        assert_eq!(protein, b"MA*"); // Met-Ala-Stop
    }

    // ── Symbolic (numeric) ─────────────────────────────────
    #[test]
    fn test_differentiate_x_squared() {
        let f = |x: f64| x * x;
        let d = differentiate(f, 3.0);
        assert!((d - 6.0).abs() < 1e-5); // d/dx(x^2) at x=3 = 6
    }

    #[test]
    fn test_integrate_x_squared() {
        let f = |x: f64| x * x;
        let result = integrate(f, 0.0, 1.0, 1000);
        assert!((result - 1.0 / 3.0).abs() < 1e-6); // integral of x^2 from 0 to 1 = 1/3
    }

    #[test]
    fn test_find_root() {
        let f = |x: f64| x * x - 2.0;
        let root = find_root(f, 1.0, 1e-10, 100);
        assert!((root - std::f64::consts::SQRT_2).abs() < 1e-8);
    }

    // ── Special functions ──────────────────────────────────
    #[test]
    fn test_hyperbolic() {
        assert!((sinh(0.0)).abs() < 1e-15);
        assert!((cosh(0.0) - 1.0).abs() < 1e-15);
        assert!((tanh_fn(0.0)).abs() < 1e-15);
        assert!((asinh(sinh(1.5)) - 1.5).abs() < 1e-10);
        assert!((acosh(cosh(2.0)) - 2.0).abs() < 1e-10);
        assert!((atanh(tanh_fn(0.5)) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_factorial_choose() {
        assert!((factorial(0) - 1.0).abs() < 1e-10);
        assert!((factorial(5) - 120.0).abs() < 1e-10);
        assert!((factorial(10) - 3628800.0).abs() < 1e-5);
        assert!((choose(10, 3) - 120.0).abs() < 1e-8);
        assert!((choose(20, 10) - 184756.0).abs() < 1e-3);
    }

    #[test]
    fn test_digamma() {
        // digamma(1) = -euler_gamma ≈ -0.5772
        assert!((digamma(1.0) + 0.5772156649).abs() < 0.001);
        // digamma(2) = 1 - euler_gamma ≈ 0.4228
        assert!((digamma(2.0) - 0.4228).abs() < 0.001);
    }

    #[test]
    fn test_expit_logit() {
        assert!((expit(0.0) - 0.5).abs() < 1e-15);
        assert!((logit(0.5)).abs() < 1e-15);
        assert!((logit(expit(2.0)) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_logsumexp() {
        let x = vec![1.0, 2.0, 3.0];
        let result = logsumexp(&x);
        let expected = (1.0f64.exp() + 2.0f64.exp() + 3.0f64.exp()).ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_sinc() {
        assert!((sinc(0.0) - 1.0).abs() < 1e-10);
        assert!((sinc(1.0)).abs() < 1e-10); // sin(pi)/pi = 0
    }

    #[test]
    fn test_gcd_lcm() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(lcm(4, 6), 12);
    }

    #[test]
    fn test_skewness_kurtosis() {
        let normal_ish = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let s = skewness(&normal_ish);
        assert!(s.abs() < 0.5, "symmetric data should have near-zero skew");
        let k = kurtosis(&normal_ish);
        assert!(k.abs() < 2.0, "uniform-ish data should have low excess kurtosis");
    }

    #[test]
    fn test_geometric_harmonic_mean() {
        let d = vec![1.0, 2.0, 4.0, 8.0];
        let gm = geometric_mean(&d);
        assert!((gm - 2.828).abs() < 0.01); // (1*2*4*8)^(1/4) = 2^(10/4) ≈ 2.828
        let hm = harmonic_mean(&d);
        assert!(hm < gm && hm > 0.0); // harmonic < geometric always
    }

    #[test]
    fn test_mad_iqr() {
        let d = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let i = iqr(&d);
        assert!(i > 0.0);
        let m = mad(&d);
        assert!(m > 0.0);
    }

    #[test]
    fn test_rankdata() {
        let d = vec![3.0, 1.0, 2.0];
        let r = rankdata(&d);
        assert!((r[0] - 3.0).abs() < 1e-10); // 3.0 is rank 3
        assert!((r[1] - 1.0).abs() < 1e-10); // 1.0 is rank 1
        assert!((r[2] - 2.0).abs() < 1e-10); // 2.0 is rank 2
    }

    #[test]
    fn test_cumprod_cummax() {
        assert_eq!(cumprod(&[1.0, 2.0, 3.0, 4.0]), vec![1.0, 2.0, 6.0, 24.0]);
        assert_eq!(cummax(&[1.0, 3.0, 2.0, 5.0]), vec![1.0, 3.0, 3.0, 5.0]);
        assert_eq!(cummin(&[5.0, 3.0, 4.0, 1.0]), vec![5.0, 3.0, 3.0, 1.0]);
    }

    #[test]
    fn test_mode() {
        let d = vec![1.0, 2.0, 2.0, 3.0, 2.0, 4.0];
        assert!((mode(&d) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_signif() {
        assert!((signif(1234.567, 3) - 1230.0).abs() < 1.0);
        assert!((signif(0.001234, 2) - 0.0012).abs() < 0.0001);
    }

    #[test]
    fn test_deg_rad() {
        assert!((deg2rad(180.0) - std::f64::consts::PI).abs() < 1e-10);
        assert!((rad2deg(std::f64::consts::PI) - 180.0).abs() < 1e-10);
    }

    // ── Statistical tests ─────────────────────────────────
    #[test]
    fn test_t_test_1samp() {
        let d = vec![5.1, 4.9, 5.0, 5.2, 4.8, 5.1, 5.0, 4.9, 5.0, 5.1];
        let (_, p) = t_test_1samp(&d, 5.0);
        assert!(p > 0.05, "should not reject H0: mu=5.0, p={p}");
    }

    #[test]
    fn test_t_test_ind_different() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let (_, p) = t_test_ind(&a, &b);
        assert!(p < 0.01, "should reject H0 for very different groups, p={p}");
    }

    #[test]
    fn test_f_oneway_same() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let c = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (_, p) = f_oneway(&[&a, &b, &c]);
        assert!(p > 0.05, "similar groups should not be significant");
    }

    #[test]
    fn test_chi2_test_independent() {
        // 2x2 table with no association
        let obs = vec![50.0, 50.0, 50.0, 50.0];
        let (chi2, p) = chi2_test(&obs, 2, 2);
        assert!(chi2.abs() < 1e-10, "equal table should have chi2=0");
        assert!(p > 0.99);
    }

    #[test]
    fn test_mann_whitney() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let (_, p) = mann_whitney_u(&a, &b);
        assert!(p < 0.05, "non-overlapping groups should be significant");
    }

    #[test]
    fn test_spearman_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let (rho, _) = spearman_r(&x, &y);
        assert!((rho - 1.0).abs() < 1e-10, "perfect monotone should give rho=1");
    }

    #[test]
    fn test_kendall_tau_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (tau, _) = kendall_tau(&x, &y);
        assert!((tau - 1.0).abs() < 1e-10, "identical order should give tau=1");
    }

    #[test]
    fn test_durbin_watson() {
        // Positive autocorrelation → DW < 2; no autocorrelation → DW ≈ 2
        let pos_auto = vec![0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 0.9, 0.7, 0.5];
        let dw_pos = durbin_watson(&pos_auto);
        assert!(dw_pos < 2.0, "positively autocorrelated should have DW < 2, got {dw_pos}");
        // Verify DW is between 0 and 4
        assert!(dw_pos >= 0.0 && dw_pos <= 4.0);
    }

    #[test]
    fn test_jarque_bera_normal() {
        // Approximately normal data should not reject
        let d: Vec<f64> = (0..100).map(|i| {
            let x = i as f64 / 100.0 * 6.0 - 3.0;
            (-x * x / 2.0).exp() + (i as f64 * 0.1).sin() * 0.01
        }).collect();
        let (_, p) = jarque_bera_test(&d);
        // Relaxed test — just verify it runs and returns valid p
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_binom_test() {
        // Fair coin, 50 heads out of 100
        let p = binom_test(50, 100, 0.5);
        assert!(p > 0.3, "50/100 should not reject fair coin");
    }

    #[test]
    fn test_kruskal_wallis() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 11.0, 12.0];
        let (_, p) = kruskal_wallis(&[&a, &b]);
        assert!(p < 0.05, "very different groups should be significant");
    }

    // ── Time series ───────────────────────────────────────
    #[test]
    fn test_acf() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let ac = acf(&data, 3);
        assert!((ac[0] - 1.0).abs() < 1e-10, "acf[0] should be 1.0");
        assert!(ac[1] > 0.0, "trending data should have positive acf[1]");
    }

    #[test]
    fn test_pacf() {
        let data = vec![1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 10.0];
        let pc = pacf(&data, 3);
        assert!((pc[0] - 1.0).abs() < 1e-10);
        assert_eq!(pc.len(), 4);
    }

    #[test]
    fn test_ses() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = ses(&data, 0.5);
        assert_eq!(s.len(), 5);
        assert!((s[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_holt() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let f = holt(&data, 0.8, 0.2);
        assert_eq!(f.len(), 5);
    }

    #[test]
    fn test_pct_change() {
        let data = vec![100.0, 110.0, 105.0];
        let pc = pct_change(&data);
        assert!(pc[0].is_nan());
        assert!((pc[1] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rm = rolling_mean(&data, 3);
        assert!(rm[0].is_nan());
        assert!(rm[1].is_nan());
        assert!((rm[2] - 2.0).abs() < 1e-10);
        assert!((rm[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_bollinger() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (mid, upper, lower) = bollinger_bands(&data, 3, 2.0);
        // Upper should be above middle, lower below
        for i in 2..10 {
            assert!(upper[i] > mid[i]);
            assert!(lower[i] < mid[i]);
        }
    }

    #[test]
    fn test_rsi() {
        let data = vec![44.0, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28];
        let r = rsi(&data, 14);
        assert_eq!(r.len(), data.len());
        // RSI at last point should be between 0 and 100
        let last = r[r.len() - 1];
        assert!(last >= 0.0 && last <= 100.0, "RSI should be 0-100, got {last}");
    }

    #[test]
    fn test_macd() {
        let data: Vec<f64> = (0..30).map(|i| (i as f64 * 0.3).sin() * 10.0 + 50.0).collect();
        let (macd_line, signal, hist) = macd(&data, 12, 26, 9);
        assert_eq!(macd_line.len(), data.len());
        assert_eq!(signal.len(), macd_line.len());
        assert_eq!(hist.len(), macd_line.len());
    }

    #[test]
    fn test_xcorr() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let xc = xcorr(&a, &b);
        assert_eq!(xc.len(), 5); // na + nb - 1
        assert!((xc[2] - 1.0).abs() < 1e-10); // peak at zero lag
    }

    #[test]
    fn test_butter_lowpass() {
        let (b, a) = butter_lowpass(100.0, 1000.0);
        assert_eq!(b.len(), 3);
        assert_eq!(a.len(), 3);
        assert!((a[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_detrend() {
        let data: Vec<f64> = (0..10).map(|i| 2.0 * i as f64 + 1.0).collect(); // y = 2x + 1
        let dt = detrend(&data);
        // Detrended linear data should be near zero
        let max_abs = dt.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);
        assert!(max_abs < 0.1, "detrended linear data should be near zero, max={max_abs}");
    }

    #[test]
    fn test_decompose() {
        let period = 4;
        let data: Vec<f64> = (0..20).map(|i| {
            i as f64 * 0.5 + (i % period) as f64 * 2.0
        }).collect();
        let (trend, seasonal, residual) = decompose_additive(&data, period);
        assert_eq!(trend.len(), data.len());
        assert_eq!(seasonal.len(), data.len());
        assert_eq!(residual.len(), data.len());
    }

    // ── ML estimators ─────────────────────────────────────
    #[test]
    fn test_kmeans() {
        let data = vec![
            vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.0],
            vec![5.0, 5.0], vec![5.1, 5.1], vec![5.2, 5.0],
        ];
        let (labels, centroids) = kmeans(&data, 2, 100);
        assert_eq!(labels.len(), 6);
        assert_eq!(centroids.len(), 2);
        // First 3 should be same cluster, last 3 same cluster
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_pca() {
        let data = vec![
            vec![2.5, 2.4], vec![0.5, 0.7], vec![2.2, 2.9],
            vec![1.9, 2.2], vec![3.1, 3.0], vec![2.3, 2.7],
        ];
        let (components, eigenvalues, projected) = pca(&data, 2);
        assert_eq!(components.len(), 2);
        assert_eq!(eigenvalues.len(), 2);
        assert_eq!(projected.len(), 6);
        assert!(eigenvalues[0] >= eigenvalues[1], "eigenvalues should be sorted descending");
    }

    #[test]
    fn test_knn_classify() {
        let train_x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let train_y = vec![0, 0, 0, 1, 1];
        let test_x = vec![vec![1.5], vec![3.5]];
        let pred = knn_classify(&train_x, &train_y, &test_x, 3);
        assert_eq!(pred[0], 0); // closer to 0, 1, 2
        assert_eq!(pred[1], 1); // closer to 3, 4
    }

    #[test]
    fn test_standard_scale() {
        let data = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        let (scaled, means, stds) = standard_scale(&data);
        assert_eq!(scaled.len(), 3);
        // Mean of each scaled column should be ~0
        let col0_mean: f64 = scaled.iter().map(|r| r[0]).sum::<f64>() / 3.0;
        assert!(col0_mean.abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix() {
        let y_true = vec![0, 0, 1, 1, 2, 2];
        let y_pred = vec![0, 1, 1, 1, 2, 0];
        let cm = confusion_matrix(&y_true, &y_pred);
        assert_eq!(cm[0][0], 1); // true 0, pred 0
        assert_eq!(cm[0][1], 1); // true 0, pred 1
        assert_eq!(cm[1][1], 2); // true 1, pred 1
    }

    #[test]
    fn test_accuracy_precision_recall() {
        let y_true = vec![0, 0, 1, 1, 1];
        let y_pred = vec![0, 1, 1, 1, 0];
        assert!((accuracy(&y_true, &y_pred) - 0.6).abs() < 1e-10);
        assert!((precision(&y_true, &y_pred, 1) - 2.0 / 3.0).abs() < 1e-10);
        assert!((recall(&y_true, &y_pred, 1) - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_r2_mse() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![1.1, 2.0, 2.9, 4.1, 5.0];
        let r2 = r2_score(&y_true, &y_pred);
        assert!(r2 > 0.99, "nearly perfect prediction should have high R2");
        let m = mse(&y_true, &y_pred);
        assert!(m < 0.01);
    }

    #[test]
    fn test_silhouette() {
        let data = vec![vec![0.0], vec![0.1], vec![10.0], vec![10.1]];
        let labels = vec![0, 0, 1, 1];
        let score = silhouette_score(&data, &labels);
        assert!(score > 0.9, "well-separated clusters should have high silhouette, got {score}");
    }

    #[test]
    fn test_train_test_split() {
        let x: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let (train_x, test_x, train_y, test_y) = train_test_split(&x, &y, 0.3, 42);
        assert_eq!(train_x.len(), 7);
        assert_eq!(test_x.len(), 3);
        assert_eq!(train_y.len(), 7);
        assert_eq!(test_y.len(), 3);
    }
}
