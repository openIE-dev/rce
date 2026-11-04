//! Matrix I/O: CSV, Matrix Market, dense/sparse serialization,
//! pretty printing, format conversion.

// ═══════════════════════════════════════════════════════════════════
// CSV FORMAT
// ═══════════════════════════════════════════════════════════════════

/// Parse matrix from CSV string (row-major). Returns (data, n_rows, n_cols).
pub fn matrix_from_csv(csv: &str) -> (Vec<f64>, usize, usize) {
    let rows: Vec<&str> = csv.lines().filter(|l| !l.trim().is_empty()).collect();
    let n_rows = rows.len();
    if n_rows == 0 { return (vec![], 0, 0); }
    let mut data = Vec::new();
    let mut n_cols = 0;
    for row in &rows {
        let vals: Vec<f64> = row.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        if n_cols == 0 { n_cols = vals.len(); }
        data.extend(vals);
    }
    (data, n_rows, n_cols)
}

/// Convert matrix to CSV string.
pub fn matrix_to_csv(data: &[f64], n_rows: usize, n_cols: usize) -> String {
    let mut csv = String::new();
    for i in 0..n_rows {
        for j in 0..n_cols {
            if j > 0 { csv.push(','); }
            csv.push_str(&data[i * n_cols + j].to_string());
        }
        csv.push('\n');
    }
    csv
}

/// Parse vector from CSV string (single column or single row).
pub fn vector_from_csv(csv: &str) -> Vec<f64> {
    csv.split(|c: char| c == ',' || c == '\n')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

/// Vector to CSV string.
pub fn vector_to_csv(data: &[f64]) -> String {
    data.iter().map(|v| v.to_string()).collect::<Vec<String>>().join(",")
}

// ═══════════════════════════════════════════════════════════════════
// MATRIX MARKET FORMAT (simplified)
// ═══════════════════════════════════════════════════════════════════

/// Parse Matrix Market coordinate format.
/// Returns (data, row_indices, col_indices, n_rows, n_cols).
pub fn matrix_market_parse(text: &str) -> (Vec<f64>, Vec<usize>, Vec<usize>, usize, usize) {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let mut n_rows = 0;
    let mut n_cols = 0;
    let mut header_done = false;

    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('%') { continue; }
        if !header_done {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                n_rows = parts[0].parse().unwrap_or(0);
                n_cols = parts[1].parse().unwrap_or(0);
            }
            header_done = true;
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let r: usize = parts[0].parse().unwrap_or(1) - 1; // 1-indexed
            let c: usize = parts[1].parse().unwrap_or(1) - 1;
            let v: f64 = parts[2].parse().unwrap_or(0.0);
            rows.push(r); cols.push(c); vals.push(v);
        }
    }
    (vals, rows, cols, n_rows, n_cols)
}

/// Convert to Matrix Market coordinate format string.
pub fn matrix_market_format(data: &[f64], rows: &[usize], cols: &[usize], n_rows: usize, n_cols: usize) -> String {
    let mut mm = format!("%%MatrixMarket matrix coordinate real general\n{} {} {}\n", n_rows, n_cols, data.len());
    for i in 0..data.len() {
        mm.push_str(&format!("{} {} {}\n", rows[i] + 1, cols[i] + 1, data[i]));
    }
    mm
}

/// Dense matrix to Matrix Market.
pub fn dense_to_matrix_market(data: &[f64], n_rows: usize, n_cols: usize) -> String {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..n_rows {
        for j in 0..n_cols {
            let v = data[i * n_cols + j];
            if v.abs() > 1e-15 { rows.push(i); cols.push(j); vals.push(v); }
        }
    }
    matrix_market_format(&vals, &rows, &cols, n_rows, n_cols)
}

// ═══════════════════════════════════════════════════════════════════
// PRETTY PRINTING
// ═══════════════════════════════════════════════════════════════════

/// Pretty print a matrix.
pub fn matrix_to_string(data: &[f64], n_rows: usize, n_cols: usize, precision: usize) -> String {
    let mut s = String::new();
    for i in 0..n_rows {
        s.push('[');
        for j in 0..n_cols {
            if j > 0 { s.push_str(", "); }
            s.push_str(&format!("{:.*}", precision, data[i * n_cols + j]));
        }
        s.push_str("]\n");
    }
    s
}

/// Pretty print a vector.
pub fn vector_to_string(data: &[f64], precision: usize) -> String {
    let inner: Vec<String> = data.iter().map(|v| format!("{:.*}", precision, v)).collect();
    format!("[{}]", inner.join(", "))
}

/// Matrix shape string.
pub fn shape_string(n_rows: usize, n_cols: usize) -> String {
    format!("({} × {})", n_rows, n_cols)
}

// ═══════════════════════════════════════════════════════════════════
// FORMAT CONVERSION
// ═══════════════════════════════════════════════════════════════════

/// Dense to COO (coordinate) format. Returns (values, rows, cols).
pub fn dense_to_coo(data: &[f64], n_rows: usize, n_cols: usize) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    let mut vals = Vec::new();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for i in 0..n_rows {
        for j in 0..n_cols {
            let v = data[i * n_cols + j];
            if v.abs() > 1e-15 { vals.push(v); rows.push(i); cols.push(j); }
        }
    }
    (vals, rows, cols)
}

/// COO to dense format.
pub fn coo_to_dense(vals: &[f64], rows: &[usize], cols: &[usize], n_rows: usize, n_cols: usize) -> Vec<f64> {
    let mut data = vec![0.0; n_rows * n_cols];
    for i in 0..vals.len() { data[rows[i] * n_cols + cols[i]] = vals[i]; }
    data
}

/// Diagonal to dense matrix.
pub fn diag_to_dense(diag: &[f64]) -> Vec<f64> {
    let n = diag.len();
    let mut data = vec![0.0; n * n];
    for i in 0..n { data[i * n + i] = diag[i]; }
    data
}

/// Extract diagonal from dense matrix.
pub fn dense_to_diag(data: &[f64], n: usize) -> Vec<f64> {
    (0..n).map(|i| data[i * n + i]).collect()
}

/// Lower triangle of matrix.
pub fn tril(data: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n * n];
    for i in 0..n { for j in 0..=i { result[i * n + j] = data[i * n + j]; } }
    result
}

/// Upper triangle of matrix.
pub fn triu(data: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n * n];
    for i in 0..n { for j in i..n { result[i * n + j] = data[i * n + j]; } }
    result
}

/// Check if matrix is symmetric.
pub fn is_symmetric(data: &[f64], n: usize, tol: f64) -> bool {
    for i in 0..n { for j in i + 1..n { if (data[i * n + j] - data[j * n + i]).abs() > tol { return false; } } }
    true
}

/// Check if matrix is upper triangular.
pub fn is_upper_triangular(data: &[f64], n: usize, tol: f64) -> bool {
    for i in 1..n { for j in 0..i { if data[i * n + j].abs() > tol { return false; } } }
    true
}

/// Check if matrix is lower triangular.
pub fn is_lower_triangular(data: &[f64], n: usize, tol: f64) -> bool {
    for i in 0..n { for j in i + 1..n { if data[i * n + j].abs() > tol { return false; } } }
    true
}

/// Check if matrix is diagonal.
pub fn is_diagonal(data: &[f64], n: usize, tol: f64) -> bool {
    is_upper_triangular(data, n, tol) && is_lower_triangular(data, n, tol)
}

/// Check if matrix is identity.
pub fn is_identity(data: &[f64], n: usize, tol: f64) -> bool {
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            if (data[i * n + j] - expected).abs() > tol { return false; }
        }
    }
    true
}

/// Matrix sparsity (fraction of zeros).
pub fn sparsity(data: &[f64], tol: f64) -> f64 {
    let zeros = data.iter().filter(|v| v.abs() < tol).count();
    zeros as f64 / data.len() as f64
}

/// Bandwidth of a banded matrix (max distance from diagonal with nonzero).
pub fn matrix_bandwidth(data: &[f64], n: usize, tol: f64) -> usize {
    let mut bw = 0;
    for i in 0..n {
        for j in 0..n {
            if data[i * n + j].abs() > tol {
                bw = bw.max(if i > j { i - j } else { j - i });
            }
        }
    }
    bw
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let csv = matrix_to_csv(&data, 2, 3);
        let (parsed, nr, nc) = matrix_from_csv(&csv);
        assert_eq!(nr, 2);
        assert_eq!(nc, 3);
        assert_eq!(parsed, data);
    }

    #[test]
    fn test_mm_parse() {
        let mm = "%%MatrixMarket matrix coordinate real general\n2 2 3\n1 1 1.0\n1 2 2.0\n2 2 3.0\n";
        let (vals, rows, cols, nr, nc) = matrix_market_parse(mm);
        assert_eq!(nr, 2);
        assert_eq!(nc, 2);
        assert_eq!(vals.len(), 3);
    }

    #[test]
    fn test_pretty_print() {
        let s = matrix_to_string(&[1.0, 2.0, 3.0, 4.0], 2, 2, 1);
        assert!(s.contains("1.0"));
        assert!(s.contains("4.0"));
    }

    #[test]
    fn test_coo_roundtrip() {
        let data = vec![1.0, 0.0, 0.0, 2.0];
        let (vals, rows, cols) = dense_to_coo(&data, 2, 2);
        let recovered = coo_to_dense(&vals, &rows, &cols, 2, 2);
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_tril_triu() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let l = tril(&data, 2);
        assert_eq!(l[1], 0.0); // upper off-diagonal zeroed
        let u = triu(&data, 2);
        assert_eq!(u[2], 0.0); // lower off-diagonal zeroed
    }

    #[test]
    fn test_is_symmetric() {
        assert!(is_symmetric(&[1.0, 2.0, 2.0, 3.0], 2, 1e-10));
        assert!(!is_symmetric(&[1.0, 2.0, 3.0, 4.0], 2, 1e-10));
    }

    #[test]
    fn test_sparsity() {
        let data = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        assert!((sparsity(&data, 1e-10) - 6.0/9.0).abs() < 1e-10);
    }
}
