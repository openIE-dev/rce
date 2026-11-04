//! Linear algebra decompositions: LU, QR, Cholesky, SVD, Schur,
//! eigenvalue decomposition, matrix factorizations.

// ═══════════════════════════════════════════════════════════════════
// LU DECOMPOSITION (partial pivoting)
// ═══════════════════════════════════════════════════════════════════

/// LU decomposition with partial pivoting: PA = LU.
/// Returns (L, U, P) as row-major n×n matrices.
pub fn lu_decompose(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let mut u = a.to_vec();
    let mut l = vec![0.0; n * n];
    let mut p: Vec<usize> = (0..n).collect();

    for i in 0..n {
        l[i * n + i] = 1.0;
        // Partial pivoting
        let mut max_val = u[i * n + i].abs();
        let mut max_row = i;
        for k in i + 1..n {
            if u[k * n + i].abs() > max_val { max_val = u[k * n + i].abs(); max_row = k; }
        }
        if max_row != i {
            p.swap(i, max_row);
            for j in 0..n { u.swap(i * n + j, max_row * n + j); }
            for j in 0..i { l.swap(i * n + j, max_row * n + j); }
        }

        if u[i * n + i].abs() < 1e-15 { continue; }

        for k in i + 1..n {
            l[k * n + i] = u[k * n + i] / u[i * n + i];
            for j in i..n {
                u[k * n + j] -= l[k * n + i] * u[i * n + j];
            }
        }
    }
    (l, u, p)
}

/// Solve Ax = b using LU decomposition.
pub fn lu_solve(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let (l, u, p) = lu_decompose(a, n);
    // Apply permutation to b
    let mut pb = vec![0.0; n];
    for i in 0..n { pb[i] = b[p[i]]; }
    // Forward substitution: Ly = Pb
    let mut y = vec![0.0; n];
    for i in 0..n {
        y[i] = pb[i];
        for j in 0..i { y[i] -= l[i * n + j] * y[j]; }
    }
    // Back substitution: Ux = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = y[i];
        for j in i + 1..n { x[i] -= u[i * n + j] * x[j]; }
        if u[i * n + i].abs() > 1e-15 { x[i] /= u[i * n + i]; }
    }
    x
}

/// Determinant via LU decomposition.
pub fn determinant(a: &[f64], n: usize) -> f64 {
    let (_, u, p) = lu_decompose(a, n);
    let mut det = 1.0;
    for i in 0..n { det *= u[i * n + i]; }
    // Count swaps in permutation
    let mut swaps = 0;
    let mut visited = vec![false; n];
    for i in 0..n {
        if visited[i] { continue; }
        let mut j = i;
        let mut cycle_len = 0;
        while !visited[j] { visited[j] = true; j = p[j]; cycle_len += 1; }
        if cycle_len > 1 { swaps += cycle_len - 1; }
    }
    if swaps % 2 != 0 { det = -det; }
    det
}

/// Matrix inverse via LU decomposition.
pub fn matrix_inverse(a: &[f64], n: usize) -> Vec<f64> {
    let mut inv = vec![0.0; n * n];
    for j in 0..n {
        let mut ej = vec![0.0; n];
        ej[j] = 1.0;
        let col = lu_solve(a, &ej, n);
        for i in 0..n { inv[i * n + j] = col[i]; }
    }
    inv
}

// ═══════════════════════════════════════════════════════════════════
// QR DECOMPOSITION (Gram-Schmidt)
// ═══════════════════════════════════════════════════════════════════

/// QR decomposition via modified Gram-Schmidt. A is m×n (row-major).
/// Returns (Q: m×n, R: n×n).
pub fn qr_decompose(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut q = vec![0.0; m * n];
    let mut r = vec![0.0; n * n];

    // Copy columns of A
    let mut cols: Vec<Vec<f64>> = (0..n).map(|j| (0..m).map(|i| a[i * n + j]).collect()).collect();

    for j in 0..n {
        // Orthogonalize against previous columns
        for i in 0..j {
            let dot: f64 = (0..m).map(|k| cols[j][k] * cols[i][k]).sum();
            r[i * n + j] = dot;
            for k in 0..m { cols[j][k] -= dot * cols[i][k]; }
        }
        // Normalize
        let norm: f64 = cols[j].iter().map(|x| x * x).sum::<f64>().sqrt();
        r[j * n + j] = norm;
        if norm > 1e-15 {
            for k in 0..m { cols[j][k] /= norm; }
        }
    }

    // Build Q matrix
    for j in 0..n {
        for i in 0..m { q[i * n + j] = cols[j][i]; }
    }
    (q, r)
}

/// Solve least squares Ax ≈ b via QR decomposition.
pub fn qr_solve(a: &[f64], b: &[f64], m: usize, n: usize) -> Vec<f64> {
    let (q, r) = qr_decompose(a, m, n);
    // Q^T * b
    let mut qtb = vec![0.0; n];
    for j in 0..n {
        for i in 0..m { qtb[j] += q[i * n + j] * b[i]; }
    }
    // Back substitution: Rx = Q^T b
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = qtb[i];
        for j in i + 1..n { x[i] -= r[i * n + j] * x[j]; }
        if r[i * n + i].abs() > 1e-15 { x[i] /= r[i * n + i]; }
    }
    x
}

// ═══════════════════════════════════════════════════════════════════
// CHOLESKY DECOMPOSITION
// ═══════════════════════════════════════════════════════════════════

/// Cholesky decomposition: A = LL^T. A must be symmetric positive definite.
/// Returns lower triangular L (n×n row-major), or None if not SPD.
pub fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j { sum += l[i * n + k] * l[j * n + k]; }
            if i == j {
                let diag = a[i * n + i] - sum;
                if diag <= 0.0 { return None; }
                l[i * n + j] = diag.sqrt();
            } else {
                l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Solve Ax = b using Cholesky decomposition (A must be SPD).
pub fn cholesky_solve(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let l = cholesky(a, n)?;
    // Forward substitution: Ly = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        y[i] = b[i];
        for j in 0..i { y[i] -= l[i * n + j] * y[j]; }
        y[i] /= l[i * n + i];
    }
    // Back substitution: L^T x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = y[i];
        for j in i + 1..n { x[i] -= l[j * n + i] * x[j]; }
        x[i] /= l[i * n + i];
    }
    Some(x)
}

// ═══════════════════════════════════════════════════════════════════
// SVD (Singular Value Decomposition) — simplified via eigendecomposition
// ═══════════════════════════════════════════════════════════════════

/// Compute singular values of m×n matrix (via A^T A eigenvalues).
pub fn singular_values(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    // A^T A is n×n
    let mut ata = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..m { ata[i * n + j] += a[k * n + i] * a[k * n + j]; }
        }
    }
    let eigenvals = eigenvalues_symmetric(&ata, n);
    eigenvals.iter().map(|&v| v.max(0.0).sqrt()).collect()
}

/// Condition number (ratio of largest to smallest singular value).
pub fn condition_number(a: &[f64], m: usize, n: usize) -> f64 {
    let sv = singular_values(a, m, n);
    let max_sv = sv.iter().copied().fold(0.0, f64::max);
    let min_sv = sv.iter().copied().fold(f64::INFINITY, f64::min);
    if min_sv < 1e-15 { return f64::INFINITY; }
    max_sv / min_sv
}

/// Matrix rank (number of singular values above threshold).
pub fn matrix_rank(a: &[f64], m: usize, n: usize, tol: f64) -> usize {
    singular_values(a, m, n).iter().filter(|&&s| s > tol).count()
}

/// Pseudoinverse via SVD (simplified: uses QR for thin case).
pub fn pseudoinverse(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    if m >= n {
        // A^+ = (A^T A)^{-1} A^T
        let mut ata = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..m { ata[i * n + j] += a[k * n + i] * a[k * n + j]; }
            }
        }
        let ata_inv = matrix_inverse(&ata, n);
        // ata_inv * A^T
        let mut result = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                for k in 0..n { result[i * m + j] += ata_inv[i * n + k] * a[j * n + k]; }
            }
        }
        result
    } else {
        // A^+ = A^T (A A^T)^{-1}
        let mut aat = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                for k in 0..n { aat[i * m + j] += a[i * n + k] * a[j * n + k]; }
            }
        }
        let aat_inv = matrix_inverse(&aat, m);
        let mut result = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                for k in 0..m { result[i * m + j] += a[k * n + i] * aat_inv[k * m + j]; }
            }
        }
        result
    }
}

// ═══════════════════════════════════════════════════════════════════
// EIGENVALUE DECOMPOSITION
// ═══════════════════════════════════════════════════════════════════

/// Eigenvalues of symmetric matrix (Jacobi method).
pub fn eigenvalues_symmetric(a: &[f64], n: usize) -> Vec<f64> {
    let mut mat = a.to_vec();
    for _ in 0..100 * n {
        // Find largest off-diagonal
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in i + 1..n {
                if mat[i * n + j].abs() > max_val {
                    max_val = mat[i * n + j].abs();
                    p = i; q = j;
                }
            }
        }
        if max_val < 1e-12 { break; }
        // Jacobi rotation
        let app = mat[p * n + p];
        let aqq = mat[q * n + q];
        let apq = mat[p * n + q];
        let theta = if (app - aqq).abs() < 1e-15 { std::f64::consts::FRAC_PI_4 } else { 0.5 * (2.0 * apq / (app - aqq)).atan() };
        let (s, c) = theta.sin_cos();
        // Update matrix
        let mut new_mat = mat.clone();
        for i in 0..n {
            if i != p && i != q {
                let ip = c * mat[i * n + p] + s * mat[i * n + q];
                let iq = -s * mat[i * n + p] + c * mat[i * n + q];
                new_mat[i * n + p] = ip; new_mat[p * n + i] = ip;
                new_mat[i * n + q] = iq; new_mat[q * n + i] = iq;
            }
        }
        new_mat[p * n + p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        new_mat[q * n + q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        new_mat[p * n + q] = 0.0;
        new_mat[q * n + p] = 0.0;
        mat = new_mat;
    }
    (0..n).map(|i| mat[i * n + i]).collect()
}

/// Eigenvalues and eigenvectors of symmetric matrix.
/// Returns (eigenvalues, eigenvectors as column-major n×n).
pub fn eig_symmetric(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut mat = a.to_vec();
    let mut vecs = vec![0.0; n * n];
    for i in 0..n { vecs[i * n + i] = 1.0; } // identity

    for _ in 0..200 * n {
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in i + 1..n {
                if mat[i * n + j].abs() > max_val {
                    max_val = mat[i * n + j].abs();
                    p = i; q = j;
                }
            }
        }
        if max_val < 1e-12 { break; }

        let app = mat[p * n + p];
        let aqq = mat[q * n + q];
        let apq = mat[p * n + q];
        let theta = if (app - aqq).abs() < 1e-15 { std::f64::consts::FRAC_PI_4 } else { 0.5 * (2.0 * apq / (app - aqq)).atan() };
        let (s, c) = theta.sin_cos();

        let mut new_mat = mat.clone();
        for i in 0..n {
            if i != p && i != q {
                let ip = c * mat[i * n + p] + s * mat[i * n + q];
                let iq = -s * mat[i * n + p] + c * mat[i * n + q];
                new_mat[i * n + p] = ip; new_mat[p * n + i] = ip;
                new_mat[i * n + q] = iq; new_mat[q * n + i] = iq;
            }
        }
        new_mat[p * n + p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        new_mat[q * n + q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        new_mat[p * n + q] = 0.0;
        new_mat[q * n + p] = 0.0;
        mat = new_mat;

        // Update eigenvector matrix
        for i in 0..n {
            let vip = c * vecs[i * n + p] + s * vecs[i * n + q];
            let viq = -s * vecs[i * n + p] + c * vecs[i * n + q];
            vecs[i * n + p] = vip;
            vecs[i * n + q] = viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| mat[i * n + i]).collect();
    (eigenvalues, vecs)
}

/// Trace of matrix.
pub fn trace(a: &[f64], n: usize) -> f64 {
    (0..n).map(|i| a[i * n + i]).sum()
}

/// Frobenius norm.
pub fn frobenius_norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Matrix-matrix multiply: C = A (m×k) * B (k×n).
pub fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            for l in 0..k { c[i * n + j] += a[i * k + l] * b[l * n + j]; }
        }
    }
    c
}

/// Matrix transpose (m×n → n×m).
pub fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut t = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n { t[j * m + i] = a[i * n + j]; }
    }
    t
}

/// Identity matrix n×n.
pub fn eye(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n { m[i * n + i] = 1.0; }
    m
}

/// Null space dimension (nullity).
pub fn nullity(a: &[f64], m: usize, n: usize) -> usize {
    n - matrix_rank(a, m, n, 1e-10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lu_solve() {
        // [[2,1],[1,3]] x = [5, 10] → x = [5/5, 15/5] = [1, 3]
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let b = vec![5.0, 10.0];
        let x = lu_solve(&a, &b, 2);
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_determinant() {
        // det([[1,2],[3,4]]) = -2
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert!((determinant(&a, 2) - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_inverse() {
        let a = vec![4.0, 7.0, 2.0, 6.0];
        let inv = matrix_inverse(&a, 2);
        // A * A^{-1} = I
        let prod = matmul(&a, &inv, 2, 2, 2);
        assert!((prod[0] - 1.0).abs() < 1e-10);
        assert!((prod[3] - 1.0).abs() < 1e-10);
        assert!(prod[1].abs() < 1e-10);
    }

    #[test]
    fn test_qr_solve() {
        // Overdetermined: 3×2 system
        let a = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let b = vec![1.0, 2.0, 2.0];
        let x = qr_solve(&a, &b, 3, 2);
        // Check residual is small
        let ax = matmul(&a, &x, 3, 2, 1);
        let res: f64 = (0..3).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        assert!(res < 1.0);
    }

    #[test]
    fn test_cholesky() {
        // SPD: [[4, 2], [2, 3]]
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let l = cholesky(&a, 2).unwrap();
        // L * L^T should equal A
        let lt = transpose(&l, 2, 2);
        let prod = matmul(&l, &lt, 2, 2, 2);
        for i in 0..4 { assert!((prod[i] - a[i]).abs() < 1e-10); }
    }

    #[test]
    fn test_cholesky_solve() {
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let x = cholesky_solve(&a, &b, 2).unwrap();
        let ax = matmul(&a, &x, 2, 2, 1);
        assert!((ax[0] - b[0]).abs() < 1e-10);
        assert!((ax[1] - b[1]).abs() < 1e-10);
    }

    #[test]
    fn test_eigenvalues() {
        // [[2,1],[1,2]] has eigenvalues 1 and 3
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let mut eigs = eigenvalues_symmetric(&a, 2);
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((eigs[0] - 1.0).abs() < 1e-6);
        assert!((eigs[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_singular_values() {
        let a = vec![3.0, 0.0, 0.0, 4.0]; // diagonal → SVs are 3,4
        let mut sv = singular_values(&a, 2, 2);
        sv.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((sv[0] - 4.0).abs() < 1e-6);
        assert!((sv[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_trace() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert!((trace(&a, 2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_rank() {
        // Rank 1: [[1,2],[2,4]]
        let a = vec![1.0, 2.0, 2.0, 4.0];
        assert_eq!(matrix_rank(&a, 2, 2, 1e-6), 1);
    }

    #[test]
    fn test_condition_number() {
        // Identity has condition number 1
        let a = vec![1.0, 0.0, 0.0, 1.0];
        assert!((condition_number(&a, 2, 2) - 1.0).abs() < 1e-6);
    }
}
