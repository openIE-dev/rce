//! Extended linear algebra: sparse operations, matrix decompositions,
//! matrix functions, norms, and utilities from LAPACK/BLAS.

use crate::math_runtime::*;

// ═══════════════════════════════════════════════════════════════════
// MATRIX FUNCTIONS (scipy.linalg expm, logm, sqrtm)
// ═══════════════════════════════════════════════════════════════════

/// Matrix exponential via Padé approximation (scaling and squaring).
pub fn expm(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    if n == 0 { return vec![]; }
    // Scale: find s such that ||A/2^s|| < 1
    let norm = frobenius_norm(a);
    let s = (norm.log2().ceil() as i32).max(0);
    let scale = 2.0f64.powi(-s);
    let a_scaled: Vec<Vec<f64>> = a.iter().map(|row| row.iter().map(|&x| x * scale).collect()).collect();
    // Padé(6,6): P = I + c1*A + c2*A^2 + ... , Q = I - c1*A + c2*A^2 - ...
    let i_mat = eye(n);
    let a2 = mat_mul(&a_scaled, &a_scaled);
    let a4 = mat_mul(&a2, &a2);
    let a6 = mat_mul(&a4, &a2);
    let b = [1.0, 0.5, 1.0/12.0, 1.0/120.0, 1.0/720.0, 1.0/5040.0, 1.0/40320.0];
    // U = A(b1*I + b3*A^2 + b5*A^4)
    // V = b0*I + b2*A^2 + b4*A^4 + b6*A^6
    let mut v = mat_add(&mat_scale(&i_mat, b[0]), &mat_scale(&a2, b[2]));
    v = mat_add(&v, &mat_scale(&a4, b[4]));
    v = mat_add(&v, &mat_scale(&a6, b[6]));
    let mut u_inner = mat_add(&mat_scale(&i_mat, b[1]), &mat_scale(&a2, b[3]));
    u_inner = mat_add(&u_inner, &mat_scale(&a4, b[5]));
    let u = mat_mul(&a_scaled, &u_inner);
    // P = V + U, Q = V - U
    let p = mat_add(&v, &u);
    let q = mat_sub(&v, &u);
    // R = Q^{-1} * P
    let q_inv = match inverse(&q) {
        Some(qi) => qi,
        None => return i_mat,
    };
    let mut r = mat_mul(&q_inv, &p);
    // Unsquare: R = R^{2^s}
    for _ in 0..s { r = mat_mul(&r, &r); }
    r
}

/// Matrix square root via Denman-Beavers iteration.
pub fn sqrtm(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut y = a.to_vec();
    let mut z = eye(n);
    for _ in 0..50 {
        let y_inv = match inverse(&y) { Some(i) => i, None => break };
        let z_inv = match inverse(&z) { Some(i) => i, None => break };
        let y_new = mat_scale(&mat_add(&y, &z_inv), 0.5);
        let z_new = mat_scale(&mat_add(&z, &y_inv), 0.5);
        let diff = frobenius_norm(&mat_sub(&y_new, &y));
        y = y_new;
        z = z_new;
        if diff < 1e-14 { break; }
    }
    y
}

/// Matrix power: A^n for integer n (supports negative via inverse).
pub fn matrix_power(a: &[Vec<f64>], n: i32) -> Vec<Vec<f64>> {
    let dim = a.len();
    if n == 0 { return eye(dim); }
    if n < 0 {
        return match inverse(a) {
            Some(inv) => matrix_power(&inv, -n),
            None => eye(dim),
        };
    }
    if n == 1 { return a.to_vec(); }
    let half = matrix_power(a, n / 2);
    let result = mat_mul(&half, &half);
    if n % 2 == 0 { result } else { mat_mul(&result, a) }
}

// ═══════════════════════════════════════════════════════════════════
// MATRIX UTILITIES
// ═══════════════════════════════════════════════════════════════════

fn mat_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter().zip(b).map(|(ra, rb)| ra.iter().zip(rb).map(|(&x, &y)| x + y).collect()).collect()
}

fn mat_sub(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter().zip(b).map(|(ra, rb)| ra.iter().zip(rb).map(|(&x, &y)| x - y).collect()).collect()
}

fn mat_scale(a: &[Vec<f64>], s: f64) -> Vec<Vec<f64>> {
    a.iter().map(|row| row.iter().map(|&x| x * s).collect()).collect()
}

/// Kronecker product of two matrices.
pub fn kron(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (ma, na) = (a.len(), if a.is_empty() { 0 } else { a[0].len() });
    let (mb, nb) = (b.len(), if b.is_empty() { 0 } else { b[0].len() });
    let mut result = vec![vec![0.0; na * nb]; ma * mb];
    for i in 0..ma {
        for j in 0..na {
            for k in 0..mb {
                for l in 0..nb {
                    result[i * mb + k][j * nb + l] = a[i][j] * b[k][l];
                }
            }
        }
    }
    result
}

/// Hadamard (element-wise) product.
pub fn hadamard(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter().zip(b).map(|(ra, rb)| ra.iter().zip(rb).map(|(&x, &y)| x * y).collect()).collect()
}

/// Outer product of two vectors.
pub fn outer(a: &[f64], b: &[f64]) -> Vec<Vec<f64>> {
    a.iter().map(|&ai| b.iter().map(|&bj| ai * bj).collect()).collect()
}

/// Matrix condition number (ratio of largest to smallest singular value).
pub fn cond(a: &[Vec<f64>]) -> f64 {
    let svs = singular_values(a, 100);
    if svs.is_empty() { return f64::INFINITY; }
    let max = svs[0];
    let min = svs.last().copied().unwrap_or(0.0);
    if min < 1e-15 { f64::INFINITY } else { max / min }
}

/// Matrix 1-norm (max column sum).
pub fn norm_1(a: &[Vec<f64>]) -> f64 {
    let n = if a.is_empty() { 0 } else { a[0].len() };
    (0..n).map(|j| a.iter().map(|row| row[j].abs()).sum()).fold(0.0f64, f64::max)
}

/// Matrix infinity-norm (max row sum).
pub fn norm_inf(a: &[Vec<f64>]) -> f64 {
    a.iter().map(|row| row.iter().map(|x| x.abs()).sum()).fold(0.0f64, f64::max)
}

/// Solve upper triangular system Ux = b.
pub fn backsolve(u: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in i + 1..n { sum -= u[i][j] * x[j]; }
        x[i] = if u[i][i].abs() > 1e-15 { sum / u[i][i] } else { 0.0 };
    }
    x
}

/// Solve lower triangular system Lx = b.
pub fn forwardsolve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i { sum -= l[i][j] * x[j]; }
        x[i] = if l[i][i].abs() > 1e-15 { sum / l[i][i] } else { 0.0 };
    }
    x
}

/// Pseudo-inverse via SVD: A+ = V * S+ * U'.
pub fn pinv(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 { return vec![]; }
    let n = a[0].len();
    let at = transpose(a);
    if m >= n {
        // A+ = (A'A)^{-1} A'
        let ata = mat_mul(&at, a);
        match inverse(&ata) {
            Some(inv) => mat_mul(&inv, &at),
            None => zeros(n, m),
        }
    } else {
        // A+ = A' (AA')^{-1}
        let aat = mat_mul(a, &at);
        match inverse(&aat) {
            Some(inv) => mat_mul(&at, &inv),
            None => zeros(n, m),
        }
    }
}

/// Least squares solution: minimize ||Ax - b||^2.
pub fn lstsq(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let at = transpose(a);
    let ata = mat_mul(&at, a);
    let atb: Vec<f64> = (0..at.len()).map(|i| at[i].iter().zip(b).map(|(&x, &y)| x * y).sum()).collect();
    solve(&ata, &atb).unwrap_or(vec![0.0; at.len()])
}

/// Vandermonde matrix.
pub fn vander(x: &[f64], n: usize) -> Vec<Vec<f64>> {
    x.iter().map(|&xi| {
        (0..n).rev().map(|j| xi.powi(j as i32)).collect()
    }).collect()
}

/// Block diagonal matrix from list of matrices.
pub fn block_diag(blocks: &[&[Vec<f64>]]) -> Vec<Vec<f64>> {
    let total_rows: usize = blocks.iter().map(|b| b.len()).sum();
    let total_cols: usize = blocks.iter().map(|b| if b.is_empty() { 0 } else { b[0].len() }).sum();
    let mut result = vec![vec![0.0; total_cols]; total_rows];
    let mut r = 0;
    let mut c = 0;
    for block in blocks {
        let mr = block.len();
        let mc = if mr > 0 { block[0].len() } else { 0 };
        for i in 0..mr {
            for j in 0..mc {
                result[r + i][c + j] = block[i][j];
            }
        }
        r += mr;
        c += mc;
    }
    result
}

/// Toeplitz matrix from first column and first row.
pub fn toeplitz(col: &[f64], row: &[f64]) -> Vec<Vec<f64>> {
    let m = col.len();
    let n = row.len();
    (0..m).map(|i| {
        (0..n).map(|j| {
            if i >= j { col[i - j] } else { row[j - i] }
        }).collect()
    }).collect()
}

/// Hilbert matrix: H[i][j] = 1 / (i + j + 1).
pub fn hilbert(n: usize) -> Vec<Vec<f64>> {
    (0..n).map(|i| (0..n).map(|j| 1.0 / (i + j + 1) as f64).collect()).collect()
}

/// Companion matrix of a polynomial.
pub fn companion(coeffs: &[f64]) -> Vec<Vec<f64>> {
    let n = coeffs.len() - 1;
    if n == 0 { return vec![]; }
    let a0 = coeffs[0];
    let mut c = vec![vec![0.0; n]; n];
    for i in 1..n { c[i][i - 1] = 1.0; }
    for i in 0..n { c[0][i] = -coeffs[i + 1] / a0; }
    c
}

/// Circulant matrix from first column.
pub fn circulant(c: &[f64]) -> Vec<Vec<f64>> {
    let n = c.len();
    (0..n).map(|i| {
        (0..n).map(|j| c[(j + n - i) % n]).collect()
    }).collect()
}

/// Check if matrix is symmetric (within tolerance).
pub fn is_symmetric(a: &[Vec<f64>], tol: f64) -> bool {
    let n = a.len();
    for i in 0..n {
        for j in i + 1..n {
            if (a[i][j] - a[j][i]).abs() > tol { return false; }
        }
    }
    true
}

/// Check if matrix is positive definite (all eigenvalues > 0).
pub fn is_positive_definite(a: &[Vec<f64>]) -> bool {
    cholesky(a).is_some()
}

/// Diagonal of a matrix.
pub fn diag(a: &[Vec<f64>]) -> Vec<f64> {
    a.iter().enumerate().map(|(i, row)| *row.get(i).unwrap_or(&0.0)).collect()
}

/// Create diagonal matrix from vector.
pub fn diag_matrix(v: &[f64]) -> Vec<Vec<f64>> {
    let n = v.len();
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n { m[i][i] = v[i]; }
    m
}

/// Row sums.
pub fn row_sums(a: &[Vec<f64>]) -> Vec<f64> {
    a.iter().map(|row| row.iter().sum()).collect()
}

/// Column sums.
pub fn col_sums(a: &[Vec<f64>]) -> Vec<f64> {
    if a.is_empty() { return vec![]; }
    let n = a[0].len();
    (0..n).map(|j| a.iter().map(|row| row[j]).sum()).collect()
}

/// Row means.
pub fn row_means(a: &[Vec<f64>]) -> Vec<f64> {
    a.iter().map(|row| {
        if row.is_empty() { 0.0 } else { row.iter().sum::<f64>() / row.len() as f64 }
    }).collect()
}

/// Column means.
pub fn col_means(a: &[Vec<f64>]) -> Vec<f64> {
    if a.is_empty() { return vec![]; }
    let (m, n) = (a.len() as f64, a[0].len());
    (0..n).map(|j| a.iter().map(|row| row[j]).sum::<f64>() / m).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expm_identity() {
        let zero = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let result = expm(&zero);
        // exp(0) = I
        assert!((result[0][0] - 1.0).abs() < 1e-10);
        assert!((result[1][1] - 1.0).abs() < 1e-10);
        assert!(result[0][1].abs() < 1e-10);
    }

    #[test]
    fn test_matrix_power() {
        let a = vec![vec![1.0, 1.0], vec![0.0, 1.0]];
        let a3 = matrix_power(&a, 3);
        // [[1,1],[0,1]]^3 = [[1,3],[0,1]]
        assert!((a3[0][1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_kron() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let k = kron(&a, &b);
        assert_eq!(k.len(), 4);
        assert_eq!(k[0].len(), 4);
        assert!((k[0][0] - 1.0).abs() < 1e-10);
        assert!((k[2][0] - 3.0).abs() < 1e-10); // a[1][0]*b[0][0] = 3
    }

    #[test]
    fn test_cond() {
        let i_mat = eye(3);
        assert!((cond(&i_mat) - 1.0).abs() < 0.1, "identity should have cond ≈ 1");
    }

    #[test]
    fn test_lstsq() {
        let a = vec![vec![1.0, 1.0], vec![1.0, 2.0], vec![1.0, 3.0]];
        let b = vec![1.0, 2.0, 3.0]; // y = x, intercept=0
        let x = lstsq(&a, &b);
        assert_eq!(x.len(), 2);
    }

    #[test]
    fn test_toeplitz() {
        let t = toeplitz(&[1.0, 2.0, 3.0], &[1.0, 4.0, 5.0]);
        assert!((t[0][0] - 1.0).abs() < 1e-10);
        assert!((t[1][0] - 2.0).abs() < 1e-10);
        assert!((t[0][1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_hilbert() {
        let h = hilbert(3);
        assert!((h[0][0] - 1.0).abs() < 1e-10);
        assert!((h[0][1] - 0.5).abs() < 1e-10);
        assert!((h[1][1] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_block_diag() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0]];
        let bd = block_diag(&[&a, &b]);
        assert_eq!(bd.len(), 3);
        assert_eq!(bd[0].len(), 3);
        assert!((bd[2][2] - 5.0).abs() < 1e-10);
        assert!(bd[0][2].abs() < 1e-10);
    }

    #[test]
    fn test_is_symmetric() {
        let sym = vec![vec![1.0, 2.0], vec![2.0, 3.0]];
        assert!(is_symmetric(&sym, 1e-10));
        let asym = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!(!is_symmetric(&asym, 1e-10));
    }

    #[test]
    fn test_diag_ops() {
        let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(diag(&m), vec![1.0, 4.0]);
        let d = diag_matrix(&[5.0, 6.0]);
        assert!((d[0][0] - 5.0).abs() < 1e-10);
        assert!(d[0][1].abs() < 1e-10);
    }
}
