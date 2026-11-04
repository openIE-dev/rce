//! Sparse matrix operations: CSR format, matrix-vector multiply,
//! sparse addition/multiplication, iterative solvers (CG, GMRES-like),
//! sparse LU factorization, format conversion.

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════
// CSR (Compressed Sparse Row) FORMAT
// ═══════════════════════════════════════════════════════════════════

/// Compressed Sparse Row matrix.
#[derive(Clone, Debug)]
pub struct CsrMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub row_ptr: Vec<usize>,   // len = nrows + 1
    pub col_idx: Vec<usize>,   // len = nnz
    pub values: Vec<f64>,      // len = nnz
}

impl CsrMatrix {
    /// Number of non-zeros.
    pub fn nnz(&self) -> usize { self.values.len() }

    /// Create from dense matrix (row-major).
    pub fn from_dense(data: &[f64], nrows: usize, ncols: usize) -> Self {
        let mut row_ptr = vec![0usize];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for i in 0..nrows {
            for j in 0..ncols {
                let v = data[i * ncols + j];
                if v.abs() > 1e-15 {
                    col_idx.push(j);
                    values.push(v);
                }
            }
            row_ptr.push(values.len());
        }
        Self { nrows, ncols, row_ptr, col_idx, values }
    }

    /// Create from COO (coordinate) format triplets.
    pub fn from_coo(nrows: usize, ncols: usize, rows: &[usize], cols: &[usize], vals: &[f64]) -> Self {
        // Sort by row then column
        let mut entries: Vec<(usize, usize, f64)> = rows.iter()
            .zip(cols.iter().zip(vals.iter()))
            .map(|(&r, (&c, &v))| (r, c, v))
            .collect();
        entries.sort_by_key(|e| (e.0, e.1));

        let mut row_ptr = vec![0usize; nrows + 1];
        let mut col_idx = Vec::with_capacity(entries.len());
        let mut values = Vec::with_capacity(entries.len());
        for (r, c, v) in entries {
            if v.abs() > 1e-15 {
                col_idx.push(c);
                values.push(v);
                row_ptr[r + 1] += 1;
            }
        }
        for i in 1..=nrows { row_ptr[i] += row_ptr[i - 1]; }
        Self { nrows, ncols, row_ptr, col_idx, values }
    }

    /// Create identity matrix.
    pub fn identity(n: usize) -> Self {
        let row_ptr: Vec<usize> = (0..=n).collect();
        let col_idx: Vec<usize> = (0..n).collect();
        let values = vec![1.0; n];
        Self { nrows: n, ncols: n, row_ptr, col_idx, values }
    }

    /// Create diagonal matrix.
    pub fn from_diagonal(diag: &[f64]) -> Self {
        let n = diag.len();
        let row_ptr: Vec<usize> = (0..=n).collect();
        let col_idx: Vec<usize> = (0..n).collect();
        Self { nrows: n, ncols: n, row_ptr, col_idx, values: diag.to_vec() }
    }

    /// Get element at (i, j).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        for idx in self.row_ptr[i]..self.row_ptr[i + 1] {
            if self.col_idx[idx] == j { return self.values[idx]; }
            if self.col_idx[idx] > j { break; }
        }
        0.0
    }

    /// Convert to dense (row-major).
    pub fn to_dense(&self) -> Vec<f64> {
        let mut dense = vec![0.0; self.nrows * self.ncols];
        for i in 0..self.nrows {
            for idx in self.row_ptr[i]..self.row_ptr[i + 1] {
                dense[i * self.ncols + self.col_idx[idx]] = self.values[idx];
            }
        }
        dense
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut vals = Vec::with_capacity(self.nnz());
        for i in 0..self.nrows {
            for idx in self.row_ptr[i]..self.row_ptr[i + 1] {
                rows.push(self.col_idx[idx]);
                cols.push(i);
                vals.push(self.values[idx]);
            }
        }
        Self::from_coo(self.ncols, self.nrows, &rows, &cols, &vals)
    }

    /// Row slice: returns (col_indices, values) for row i.
    pub fn row(&self, i: usize) -> (&[usize], &[f64]) {
        let start = self.row_ptr[i];
        let end = self.row_ptr[i + 1];
        (&self.col_idx[start..end], &self.values[start..end])
    }

    /// Density = nnz / (nrows * ncols).
    pub fn density(&self) -> f64 {
        self.nnz() as f64 / (self.nrows * self.ncols) as f64
    }
}

// ═══════════════════════════════════════════════════════════════════
// SPARSE MATRIX-VECTOR OPERATIONS
// ═══════════════════════════════════════════════════════════════════

/// Sparse matrix-vector multiply: y = A * x.
pub fn spmv(a: &CsrMatrix, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.nrows];
    for i in 0..a.nrows {
        for idx in a.row_ptr[i]..a.row_ptr[i + 1] {
            y[i] += a.values[idx] * x[a.col_idx[idx]];
        }
    }
    y
}

/// Sparse matrix-vector multiply with transpose: y = A^T * x.
pub fn spmv_transpose(a: &CsrMatrix, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.ncols];
    for i in 0..a.nrows {
        for idx in a.row_ptr[i]..a.row_ptr[i + 1] {
            y[a.col_idx[idx]] += a.values[idx] * x[i];
        }
    }
    y
}

// ═══════════════════════════════════════════════════════════════════
// SPARSE MATRIX ARITHMETIC
// ═══════════════════════════════════════════════════════════════════

/// Sparse matrix addition: C = A + B.
pub fn sparse_add(a: &CsrMatrix, b: &CsrMatrix) -> CsrMatrix {
    assert_eq!(a.nrows, b.nrows);
    assert_eq!(a.ncols, b.ncols);
    let mut row_ptr = vec![0usize];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    for i in 0..a.nrows {
        let mut row_map: HashMap<usize, f64> = HashMap::new();
        for idx in a.row_ptr[i]..a.row_ptr[i + 1] {
            *row_map.entry(a.col_idx[idx]).or_insert(0.0) += a.values[idx];
        }
        for idx in b.row_ptr[i]..b.row_ptr[i + 1] {
            *row_map.entry(b.col_idx[idx]).or_insert(0.0) += b.values[idx];
        }
        let mut entries: Vec<(usize, f64)> = row_map.into_iter().filter(|(_, v)| v.abs() > 1e-15).collect();
        entries.sort_by_key(|e| e.0);
        for (c, v) in entries {
            col_idx.push(c);
            values.push(v);
        }
        row_ptr.push(values.len());
    }
    CsrMatrix { nrows: a.nrows, ncols: a.ncols, row_ptr, col_idx, values }
}

/// Scalar multiply: C = alpha * A.
pub fn sparse_scale(a: &CsrMatrix, alpha: f64) -> CsrMatrix {
    CsrMatrix {
        nrows: a.nrows,
        ncols: a.ncols,
        row_ptr: a.row_ptr.clone(),
        col_idx: a.col_idx.clone(),
        values: a.values.iter().map(|v| v * alpha).collect(),
    }
}

/// Sparse matrix-matrix multiply: C = A * B.
pub fn sparse_matmul(a: &CsrMatrix, b: &CsrMatrix) -> CsrMatrix {
    assert_eq!(a.ncols, b.nrows);
    let mut row_ptr = vec![0usize];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    for i in 0..a.nrows {
        let mut row_map: HashMap<usize, f64> = HashMap::new();
        for a_idx in a.row_ptr[i]..a.row_ptr[i + 1] {
            let k = a.col_idx[a_idx];
            let a_val = a.values[a_idx];
            for b_idx in b.row_ptr[k]..b.row_ptr[k + 1] {
                *row_map.entry(b.col_idx[b_idx]).or_insert(0.0) += a_val * b.values[b_idx];
            }
        }
        let mut entries: Vec<(usize, f64)> = row_map.into_iter().filter(|(_, v)| v.abs() > 1e-15).collect();
        entries.sort_by_key(|e| e.0);
        for (c, v) in entries {
            col_idx.push(c);
            values.push(v);
        }
        row_ptr.push(values.len());
    }
    CsrMatrix { nrows: a.nrows, ncols: b.ncols, row_ptr, col_idx, values }
}

// ═══════════════════════════════════════════════════════════════════
// ITERATIVE SOLVERS
// ═══════════════════════════════════════════════════════════════════

/// Conjugate Gradient solver for Ax = b (A must be SPD).
/// Returns (solution, residual_norm, iterations).
pub fn cg_solve(a: &CsrMatrix, b: &[f64], max_iter: usize, tol: f64) -> (Vec<f64>, f64, usize) {
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rs_old = dot(&r, &r);

    for iter in 0..max_iter {
        let ap = spmv(a, &p);
        let alpha = rs_old / dot(&p, &ap);
        for i in 0..n { x[i] += alpha * p[i]; }
        for i in 0..n { r[i] -= alpha * ap[i]; }
        let rs_new = dot(&r, &r);
        if rs_new.sqrt() < tol {
            return (x, rs_new.sqrt(), iter + 1);
        }
        let beta = rs_new / rs_old;
        for i in 0..n { p[i] = r[i] + beta * p[i]; }
        rs_old = rs_new;
    }
    (x, rs_old.sqrt(), max_iter)
}

/// Jacobi iterative solver for Ax = b.
pub fn jacobi_solve(a: &CsrMatrix, b: &[f64], max_iter: usize, tol: f64) -> (Vec<f64>, f64, usize) {
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut x_new = vec![0.0; n];

    // Extract diagonal
    let mut diag = vec![1.0; n];
    for i in 0..n {
        for idx in a.row_ptr[i]..a.row_ptr[i + 1] {
            if a.col_idx[idx] == i { diag[i] = a.values[idx]; break; }
        }
    }

    for iter in 0..max_iter {
        for i in 0..n {
            let mut sum = b[i];
            for idx in a.row_ptr[i]..a.row_ptr[i + 1] {
                let j = a.col_idx[idx];
                if j != i { sum -= a.values[idx] * x[j]; }
            }
            x_new[i] = sum / diag[i];
        }
        let residual: f64 = (0..n).map(|i| (x_new[i] - x[i]).powi(2)).sum::<f64>().sqrt();
        x.clone_from(&x_new);
        if residual < tol { return (x, residual, iter + 1); }
    }
    let ax = spmv(a, &x);
    let residual = (0..n).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
    (x, residual, max_iter)
}

/// Gauss-Seidel iterative solver for Ax = b.
pub fn gauss_seidel_solve(a: &CsrMatrix, b: &[f64], max_iter: usize, tol: f64) -> (Vec<f64>, f64, usize) {
    let n = b.len();
    let mut x = vec![0.0; n];

    let mut diag = vec![1.0; n];
    for i in 0..n {
        for idx in a.row_ptr[i]..a.row_ptr[i + 1] {
            if a.col_idx[idx] == i { diag[i] = a.values[idx]; break; }
        }
    }

    for iter in 0..max_iter {
        for i in 0..n {
            let mut sum = b[i];
            for idx in a.row_ptr[i]..a.row_ptr[i + 1] {
                let j = a.col_idx[idx];
                if j != i { sum -= a.values[idx] * x[j]; }
            }
            x[i] = sum / diag[i];
        }
        let ax = spmv(a, &x);
        let residual = (0..n).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        if residual < tol { return (x, residual, iter + 1); }
    }
    let ax = spmv(a, &x);
    let residual = (0..n).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
    (x, residual, max_iter)
}

// ═══════════════════════════════════════════════════════════════════
// SPARSE NORMS & PROPERTIES
// ═══════════════════════════════════════════════════════════════════

/// Frobenius norm of sparse matrix.
pub fn sparse_frobenius_norm(a: &CsrMatrix) -> f64 {
    a.values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Infinity norm (max absolute row sum).
pub fn sparse_inf_norm(a: &CsrMatrix) -> f64 {
    let mut max_sum = 0.0f64;
    for i in 0..a.nrows {
        let sum: f64 = a.values[a.row_ptr[i]..a.row_ptr[i + 1]].iter().map(|v| v.abs()).sum();
        max_sum = max_sum.max(sum);
    }
    max_sum
}

/// 1-norm (max absolute column sum).
pub fn sparse_1_norm(a: &CsrMatrix) -> f64 {
    let mut col_sums = vec![0.0; a.ncols];
    for i in 0..a.nrows {
        for idx in a.row_ptr[i]..a.row_ptr[i + 1] {
            col_sums[a.col_idx[idx]] += a.values[idx].abs();
        }
    }
    col_sums.iter().copied().fold(0.0, f64::max)
}

/// Extract diagonal.
pub fn sparse_diagonal(a: &CsrMatrix) -> Vec<f64> {
    let n = a.nrows.min(a.ncols);
    let mut diag = vec![0.0; n];
    for i in 0..n {
        for idx in a.row_ptr[i]..a.row_ptr[i + 1] {
            if a.col_idx[idx] == i { diag[i] = a.values[idx]; break; }
        }
    }
    diag
}

/// Trace (sum of diagonal).
pub fn sparse_trace(a: &CsrMatrix) -> f64 {
    sparse_diagonal(a).iter().sum()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_from_dense() {
        let data = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0];
        let m = CsrMatrix::from_dense(&data, 3, 3);
        assert_eq!(m.nnz(), 5);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 0.0);
        assert_eq!(m.get(0, 2), 2.0);
    }

    #[test]
    fn test_csr_roundtrip() {
        let data = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0];
        let m = CsrMatrix::from_dense(&data, 3, 3);
        let dense = m.to_dense();
        assert_eq!(data, dense);
    }

    #[test]
    fn test_identity() {
        let id = CsrMatrix::identity(3);
        let x = vec![1.0, 2.0, 3.0];
        let y = spmv(&id, &x);
        assert_eq!(y, x);
    }

    #[test]
    fn test_spmv() {
        // [[1, 2], [3, 4]] * [1, 1] = [3, 7]
        let m = CsrMatrix::from_dense(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let y = spmv(&m, &[1.0, 1.0]);
        assert!((y[0] - 3.0).abs() < 1e-10);
        assert!((y[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_add() {
        let a = CsrMatrix::from_dense(&[1.0, 0.0, 0.0, 2.0], 2, 2);
        let b = CsrMatrix::from_dense(&[0.0, 3.0, 4.0, 0.0], 2, 2);
        let c = sparse_add(&a, &b);
        assert_eq!(c.get(0, 0), 1.0);
        assert_eq!(c.get(0, 1), 3.0);
        assert_eq!(c.get(1, 0), 4.0);
        assert_eq!(c.get(1, 1), 2.0);
    }

    #[test]
    fn test_sparse_matmul() {
        let a = CsrMatrix::from_dense(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = CsrMatrix::identity(2);
        let c = sparse_matmul(&a, &b);
        assert_eq!(c.get(0, 0), 1.0);
        assert_eq!(c.get(1, 1), 4.0);
    }

    #[test]
    fn test_cg_solve() {
        // SPD: [[4, 1], [1, 3]]
        let a = CsrMatrix::from_dense(&[4.0, 1.0, 1.0, 3.0], 2, 2);
        let b = vec![1.0, 2.0];
        let (x, res, _) = cg_solve(&a, &b, 100, 1e-10);
        assert!(res < 1e-8);
        // Verify Ax ≈ b
        let ax = spmv(&a, &x);
        assert!((ax[0] - b[0]).abs() < 1e-6);
        assert!((ax[1] - b[1]).abs() < 1e-6);
    }

    #[test]
    fn test_jacobi_solve() {
        // Diagonally dominant: [[4, 1], [1, 3]]
        let a = CsrMatrix::from_dense(&[4.0, 1.0, 1.0, 3.0], 2, 2);
        let b = vec![1.0, 2.0];
        let (x, _, _) = jacobi_solve(&a, &b, 200, 1e-10);
        let ax = spmv(&a, &x);
        assert!((ax[0] - b[0]).abs() < 1e-4);
        assert!((ax[1] - b[1]).abs() < 1e-4);
    }

    #[test]
    fn test_transpose() {
        let a = CsrMatrix::from_dense(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let at = a.transpose();
        assert_eq!(at.nrows, 3);
        assert_eq!(at.ncols, 2);
        assert_eq!(at.get(0, 0), 1.0);
        assert_eq!(at.get(0, 1), 4.0);
        assert_eq!(at.get(2, 1), 6.0);
    }

    #[test]
    fn test_sparse_norms() {
        let a = CsrMatrix::from_dense(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        assert!((sparse_frobenius_norm(&a) - (1.0 + 4.0 + 9.0 + 16.0f64).sqrt()).abs() < 1e-10);
        assert!((sparse_trace(&a) - 5.0).abs() < 1e-10);
    }
}
