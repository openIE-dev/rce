//! ML kernel functions: RBF, polynomial, sigmoid, linear, Laplacian,
//! chi-squared, histogram intersection, kernel matrix computation,
//! kernel PCA, SVM-related utilities.

// ═══════════════════════════════════════════════════════════════════
// KERNEL FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

/// Linear kernel: K(x, y) = x · y.
pub fn kernel_linear(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

/// Polynomial kernel: K(x, y) = (γ·x·y + c)^d.
pub fn kernel_polynomial(x: &[f64], y: &[f64], gamma: f64, coef0: f64, degree: u32) -> f64 {
    (gamma * kernel_linear(x, y) + coef0).powi(degree as i32)
}

/// RBF (Gaussian) kernel: K(x, y) = exp(-γ·‖x-y‖²).
pub fn kernel_rbf(x: &[f64], y: &[f64], gamma: f64) -> f64 {
    let sq_dist: f64 = x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum();
    (-gamma * sq_dist).exp()
}

/// Sigmoid (hyperbolic tangent) kernel: K(x, y) = tanh(γ·x·y + c).
pub fn kernel_sigmoid(x: &[f64], y: &[f64], gamma: f64, coef0: f64) -> f64 {
    (gamma * kernel_linear(x, y) + coef0).tanh()
}

/// Laplacian kernel: K(x, y) = exp(-γ·‖x-y‖₁).
pub fn kernel_laplacian(x: &[f64], y: &[f64], gamma: f64) -> f64 {
    let l1_dist: f64 = x.iter().zip(y).map(|(a, b)| (a - b).abs()).sum();
    (-gamma * l1_dist).exp()
}

/// Chi-squared kernel: K(x, y) = exp(-γ · Σ (x_i - y_i)² / (x_i + y_i)).
pub fn kernel_chi_squared(x: &[f64], y: &[f64], gamma: f64) -> f64 {
    let chi2: f64 = x.iter().zip(y).map(|(a, b)| {
        let sum = a + b;
        if sum.abs() > 1e-15 { (a - b).powi(2) / sum } else { 0.0 }
    }).sum();
    (-gamma * chi2).exp()
}

/// Histogram intersection kernel: K(x, y) = Σ min(x_i, y_i).
pub fn kernel_histogram_intersection(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a.min(*b)).sum()
}

/// Cosine similarity kernel: K(x, y) = (x · y) / (‖x‖ · ‖y‖).
pub fn kernel_cosine(x: &[f64], y: &[f64]) -> f64 {
    let dot = kernel_linear(x, y);
    let nx = x.iter().map(|a| a * a).sum::<f64>().sqrt();
    let ny = y.iter().map(|a| a * a).sum::<f64>().sqrt();
    if nx < 1e-15 || ny < 1e-15 { 0.0 } else { dot / (nx * ny) }
}

/// Rational quadratic kernel: K(x, y) = 1 - ‖x-y‖² / (‖x-y‖² + c).
pub fn kernel_rational_quadratic(x: &[f64], y: &[f64], c: f64) -> f64 {
    let sq_dist: f64 = x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum();
    1.0 - sq_dist / (sq_dist + c)
}

/// Multiquadric kernel: K(x, y) = sqrt(‖x-y‖² + c²).
pub fn kernel_multiquadric(x: &[f64], y: &[f64], c: f64) -> f64 {
    let sq_dist: f64 = x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum();
    (sq_dist + c * c).sqrt()
}

/// Inverse multiquadric kernel.
pub fn kernel_inverse_multiquadric(x: &[f64], y: &[f64], c: f64) -> f64 {
    1.0 / kernel_multiquadric(x, y, c)
}

/// Matern kernel (ν = 1/2, equivalent to Laplacian).
pub fn kernel_matern_12(x: &[f64], y: &[f64], length_scale: f64) -> f64 {
    let dist: f64 = x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
    (-dist / length_scale).exp()
}

/// Matern kernel (ν = 3/2).
pub fn kernel_matern_32(x: &[f64], y: &[f64], length_scale: f64) -> f64 {
    let dist: f64 = x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
    let r = 3.0f64.sqrt() * dist / length_scale;
    (1.0 + r) * (-r).exp()
}

/// Matern kernel (ν = 5/2).
pub fn kernel_matern_52(x: &[f64], y: &[f64], length_scale: f64) -> f64 {
    let dist: f64 = x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
    let r = 5.0f64.sqrt() * dist / length_scale;
    (1.0 + r + r * r / 3.0) * (-r).exp()
}

// ═══════════════════════════════════════════════════════════════════
// KERNEL MATRIX COMPUTATION
// ═══════════════════════════════════════════════════════════════════

/// Compute kernel matrix K[i,j] = kernel(x_i, x_j).
/// data: n×d row-major. Returns n×n kernel matrix.
pub fn kernel_matrix(data: &[f64], n: usize, d: usize, kernel_fn: impl Fn(&[f64], &[f64]) -> f64) -> Vec<f64> {
    let mut k = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let val = kernel_fn(&data[i * d..(i + 1) * d], &data[j * d..(j + 1) * d]);
            k[i * n + j] = val;
            k[j * n + i] = val;
        }
    }
    k
}

/// Compute kernel matrix between two datasets.
/// a: na×d, b: nb×d. Returns na×nb matrix.
pub fn kernel_matrix_cross(a: &[f64], na: usize, b: &[f64], nb: usize, d: usize, kernel_fn: impl Fn(&[f64], &[f64]) -> f64) -> Vec<f64> {
    let mut k = vec![0.0; na * nb];
    for i in 0..na {
        for j in 0..nb {
            k[i * nb + j] = kernel_fn(&a[i * d..(i + 1) * d], &b[j * d..(j + 1) * d]);
        }
    }
    k
}

/// Center a kernel matrix: K_c = K - 1_n K - K 1_n + 1_n K 1_n.
pub fn center_kernel_matrix(k: &[f64], n: usize) -> Vec<f64> {
    let row_means: Vec<f64> = (0..n).map(|i| k[i * n..(i + 1) * n].iter().sum::<f64>() / n as f64).collect();
    let col_means: Vec<f64> = (0..n).map(|j| (0..n).map(|i| k[i * n + j]).sum::<f64>() / n as f64).collect();
    let grand_mean = row_means.iter().sum::<f64>() / n as f64;
    let mut kc = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            kc[i * n + j] = k[i * n + j] - row_means[i] - col_means[j] + grand_mean;
        }
    }
    kc
}

// ═══════════════════════════════════════════════════════════════════
// KERNEL PCA
// ═══════════════════════════════════════════════════════════════════

/// Kernel PCA. Returns projected data (n × n_components).
pub fn kernel_pca(data: &[f64], n: usize, d: usize, n_components: usize, gamma: f64) -> Vec<f64> {
    let k = kernel_matrix(data, n, d, |x, y| kernel_rbf(x, y, gamma));
    let kc = center_kernel_matrix(&k, n);
    let (eigenvalues, eigenvectors) = crate::linalg_decomp::eig_symmetric(&kc, n);
    // Sort by eigenvalue descending
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());
    let mut projected = vec![0.0; n * n_components];
    for c in 0..n_components {
        let idx = indices[c];
        let scale = if eigenvalues[idx] > 1e-10 { 1.0 / eigenvalues[idx].sqrt() } else { 0.0 };
        for i in 0..n { projected[i * n_components + c] = eigenvectors[i * n + idx] * scale; }
    }
    projected
}

// ═══════════════════════════════════════════════════════════════════
// SVM UTILITIES
// ═══════════════════════════════════════════════════════════════════

/// Decision function for SVM: f(x) = Σ α_i y_i K(x_i, x) + b.
pub fn svm_decision(x: &[f64], support_vectors: &[f64], n_sv: usize, d: usize,
                    alphas: &[f64], y_sv: &[f64], bias: f64,
                    kernel_fn: impl Fn(&[f64], &[f64]) -> f64) -> f64 {
    let mut sum = bias;
    for i in 0..n_sv {
        sum += alphas[i] * y_sv[i] * kernel_fn(&support_vectors[i * d..(i + 1) * d], x);
    }
    sum
}

/// Predict class from SVM decision function.
pub fn svm_predict(x: &[f64], support_vectors: &[f64], n_sv: usize, d: usize,
                   alphas: &[f64], y_sv: &[f64], bias: f64,
                   kernel_fn: impl Fn(&[f64], &[f64]) -> f64) -> f64 {
    if svm_decision(x, support_vectors, n_sv, d, alphas, y_sv, bias, kernel_fn) >= 0.0 { 1.0 } else { -1.0 }
}

/// Kernel alignment: measure similarity between two kernels.
pub fn kernel_alignment(k1: &[f64], k2: &[f64], n: usize) -> f64 {
    let mut dot = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;
    for i in 0..n * n {
        dot += k1[i] * k2[i];
        norm1 += k1[i] * k1[i];
        norm2 += k2[i] * k2[i];
    }
    dot / (norm1.sqrt() * norm2.sqrt())
}

/// Ideal kernel matrix from labels (K_ideal[i,j] = y_i * y_j).
pub fn ideal_kernel_matrix(labels: &[f64]) -> Vec<f64> {
    let n = labels.len();
    let mut k = vec![0.0; n * n];
    for i in 0..n { for j in 0..n { k[i * n + j] = labels[i] * labels[j]; } }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_linear() {
        assert!((kernel_linear(&[1.0, 2.0], &[3.0, 4.0]) - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_rbf() {
        // Same point → 1.0
        assert!((kernel_rbf(&[1.0, 2.0], &[1.0, 2.0], 1.0) - 1.0).abs() < 1e-10);
        // Different points → < 1.0
        assert!(kernel_rbf(&[0.0, 0.0], &[1.0, 1.0], 1.0) < 1.0);
    }

    #[test]
    fn test_kernel_polynomial() {
        // (1·(1·2 + 2·3) + 0)^2 = 64
        assert!((kernel_polynomial(&[1.0, 2.0], &[2.0, 3.0], 1.0, 0.0, 2) - 64.0).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_matrix() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let k = kernel_matrix(&data, 3, 2, |x, y| kernel_rbf(x, y, 1.0));
        assert_eq!(k.len(), 9);
        assert!((k[0] - 1.0).abs() < 1e-10); // self-similarity
        assert!(k[0] >= k[1]); // self > cross
    }

    #[test]
    fn test_kernel_cosine() {
        let s = kernel_cosine(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(s.abs() < 1e-10); // orthogonal
        let s2 = kernel_cosine(&[1.0, 1.0], &[1.0, 1.0]);
        assert!((s2 - 1.0).abs() < 1e-10); // identical
    }

    #[test]
    fn test_matern() {
        // At distance 0, all Matern kernels = 1
        assert!((kernel_matern_12(&[0.0], &[0.0], 1.0) - 1.0).abs() < 1e-10);
        assert!((kernel_matern_32(&[0.0], &[0.0], 1.0) - 1.0).abs() < 1e-10);
        assert!((kernel_matern_52(&[0.0], &[0.0], 1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_center_kernel() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let k = kernel_matrix(&data, 4, 1, |x, y| kernel_rbf(x, y, 0.5));
        let kc = center_kernel_matrix(&k, 4);
        // Centered kernel should have zero column/row means
        let col_mean: f64 = (0..4).map(|i| kc[i * 4]).sum::<f64>() / 4.0;
        assert!(col_mean.abs() < 1e-10);
    }

    #[test]
    fn test_kernel_alignment() {
        let k = vec![1.0, 0.0, 0.0, 1.0];
        assert!((kernel_alignment(&k, &k, 2) - 1.0).abs() < 1e-10);
    }
}
