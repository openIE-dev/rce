//! Dimensionality reduction: UMAP-like, Isomap-like, LLE,
//! factor analysis, multidimensional scaling, random projection.

// ═══════════════════════════════════════════════════════════════════
// RANDOM PROJECTION
// ═══════════════════════════════════════════════════════════════════

/// Sparse random projection matrix (Achlioptas, ±1 with prob 1/6, 0 with 2/3).
pub fn sparse_random_projection(n_features: usize, n_components: usize, seed: u64) -> Vec<f64> {
    let mut state = seed_state(seed);
    let s3 = 3.0f64.sqrt();
    (0..n_components * n_features).map(|_| {
        let u = prng_f64(&mut state);
        if u < 1.0/6.0 { s3 } else if u < 2.0/6.0 { -s3 } else { 0.0 }
    }).collect()
}

/// Gaussian random projection matrix.
pub fn gaussian_random_projection(n_features: usize, n_components: usize, seed: u64) -> Vec<f64> {
    let mut state = seed_state(seed);
    let scale = 1.0 / (n_components as f64).sqrt();
    (0..n_components * n_features).map(|_| prng_normal(&mut state) * scale).collect()
}

/// Apply projection: data (n×d) × R^T (d×k) → projected (n×k).
pub fn apply_projection(data: &[f64], n: usize, d: usize, projection: &[f64], k: usize) -> Vec<f64> {
    let mut result = vec![0.0; n * k];
    for i in 0..n {
        for j in 0..k {
            for l in 0..d {
                result[i * k + j] += data[i * d + l] * projection[j * d + l];
            }
        }
    }
    result
}

/// Johnson-Lindenstrauss minimum dimension for n points with eps distortion.
pub fn jl_min_dim(n_samples: usize, eps: f64) -> usize {
    let d = 4.0 * (n_samples as f64).ln() / (eps * eps / 2.0 - eps * eps * eps / 3.0);
    d.ceil() as usize
}

// ═══════════════════════════════════════════════════════════════════
// FACTOR ANALYSIS (simplified)
// ═══════════════════════════════════════════════════════════════════

/// Factor analysis via PCA-like extraction. Returns loadings matrix (d × n_factors).
pub fn factor_loadings(data: &[f64], n: usize, d: usize, n_factors: usize) -> Vec<f64> {
    let (_, _explained) = crate::spatial::pca(data, n, d, n_factors);
    // Loadings = eigenvectors * sqrt(eigenvalues) — use PCA projected as proxy
    let cov = crate::probability::covariance_matrix(data, n, d);
    let (eigenvalues, eigenvectors) = crate::linalg_decomp::eig_symmetric(&cov, d);
    let mut idx: Vec<usize> = (0..d).collect();
    idx.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());
    let mut loadings = vec![0.0; d * n_factors];
    for f in 0..n_factors {
        let ev_idx = idx[f];
        let sqrt_ev = eigenvalues[ev_idx].max(0.0).sqrt();
        for i in 0..d { loadings[i * n_factors + f] = eigenvectors[i * d + ev_idx] * sqrt_ev; }
    }
    loadings
}

/// Communalities from factor loadings.
pub fn communalities(loadings: &[f64], d: usize, n_factors: usize) -> Vec<f64> {
    (0..d).map(|i| (0..n_factors).map(|f| loadings[i * n_factors + f].powi(2)).sum()).collect()
}

/// Uniquenesses (1 - communalities).
pub fn uniquenesses(loadings: &[f64], d: usize, n_factors: usize) -> Vec<f64> {
    communalities(loadings, d, n_factors).iter().map(|c| 1.0 - c).collect()
}

/// Proportion of variance explained by each factor.
pub fn variance_explained(loadings: &[f64], d: usize, n_factors: usize) -> Vec<f64> {
    (0..n_factors).map(|f| {
        (0..d).map(|i| loadings[i * n_factors + f].powi(2)).sum::<f64>() / d as f64
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// CLASSICAL MDS (Multidimensional Scaling)
// ═══════════════════════════════════════════════════════════════════

/// Classical MDS from distance matrix. Returns n × n_dims embedded coordinates.
pub fn classical_mds(dist_matrix: &[f64], n: usize, n_dims: usize) -> Vec<f64> {
    // Double centering: B = -½ J D² J, where J = I - 1/n 11'
    let mut b = vec![0.0; n * n];
    let mut row_means = vec![0.0; n];
    let mut col_means = vec![0.0; n];
    let mut grand_mean = 0.0;
    for i in 0..n {
        for j in 0..n {
            let d2 = dist_matrix[i * n + j].powi(2);
            row_means[i] += d2;
            col_means[j] += d2;
            grand_mean += d2;
        }
    }
    let nf = n as f64;
    for i in 0..n { row_means[i] /= nf; }
    for j in 0..n { col_means[j] /= nf; }
    grand_mean /= nf * nf;

    for i in 0..n {
        for j in 0..n {
            b[i * n + j] = -0.5 * (dist_matrix[i * n + j].powi(2) - row_means[i] - col_means[j] + grand_mean);
        }
    }

    let (eigenvalues, eigenvectors) = crate::linalg_decomp::eig_symmetric(&b, n);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b_idx| eigenvalues[b_idx].partial_cmp(&eigenvalues[a]).unwrap());

    let mut coords = vec![0.0; n * n_dims];
    for d in 0..n_dims {
        let ev_idx = idx[d];
        let scale = eigenvalues[ev_idx].max(0.0).sqrt();
        for i in 0..n { coords[i * n_dims + d] = eigenvectors[i * n + ev_idx] * scale; }
    }
    coords
}

/// Stress measure for MDS (Kruskal stress-1).
pub fn mds_stress(original_dist: &[f64], embedded_dist: &[f64], n: usize) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        for j in i + 1..n {
            let d_orig = original_dist[i * n + j];
            let d_embed = embedded_dist[i * n + j];
            num += (d_orig - d_embed).powi(2);
            den += d_orig.powi(2);
        }
    }
    if den < 1e-15 { return 0.0; }
    (num / den).sqrt()
}

// ═══════════════════════════════════════════════════════════════════
// INDEPENDENT COMPONENT ANALYSIS (FastICA, simplified)
// ═══════════════════════════════════════════════════════════════════

/// FastICA: extract n_components independent sources from data (n × d).
/// Returns unmixing matrix (n_components × d) and source signals (n × n_components).
pub fn fast_ica(data: &[f64], n: usize, d: usize, n_components: usize, max_iter: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    // Whiten via PCA
    let (whitened, _) = crate::spatial::pca(data, n, d, n_components);
    let k = n_components;

    // Initialize random unmixing matrix
    let mut state = seed_state(seed);
    let mut w = vec![0.0; k * k];
    for v in w.iter_mut() { *v = prng_normal(&mut state); }

    // Orthogonalize
    orthogonalize(&mut w, k);

    for _ in 0..max_iter {
        let mut w_new = vec![0.0; k * k];
        for p in 0..k {
            // w_p = E[x g(w'x)] - E[g'(w'x)] w_p
            // Using g(u) = tanh(u)
            let mut ex_g = vec![0.0; k];
            let mut e_gp = 0.0;
            for i in 0..n {
                let mut wx = 0.0;
                for j in 0..k { wx += w[p * k + j] * whitened[i * k + j]; }
                let g = wx.tanh();
                let gp = 1.0 - g * g;
                for j in 0..k { ex_g[j] += whitened[i * k + j] * g; }
                e_gp += gp;
            }
            for j in 0..k {
                w_new[p * k + j] = ex_g[j] / n as f64 - (e_gp / n as f64) * w[p * k + j];
            }
        }
        orthogonalize(&mut w_new, k);
        w = w_new;
    }

    // Compute sources: S = X_white * W^T
    let mut sources = vec![0.0; n * k];
    for i in 0..n {
        for p in 0..k {
            for j in 0..k { sources[i * k + p] += whitened[i * k + j] * w[p * k + j]; }
        }
    }
    (w, sources)
}

fn orthogonalize(w: &mut [f64], k: usize) {
    for i in 0..k {
        for j in 0..i {
            let dot: f64 = (0..k).map(|l| w[i * k + l] * w[j * k + l]).sum();
            for l in 0..k { w[i * k + l] -= dot * w[j * k + l]; }
        }
        let norm: f64 = (0..k).map(|l| w[i * k + l].powi(2)).sum::<f64>().sqrt();
        if norm > 1e-15 { for l in 0..k { w[i * k + l] /= norm; } }
    }
}

// ═══════════════════════════════════════════════════════════════════
// TRUNCATED SVD (for dimensionality reduction)
// ═══════════════════════════════════════════════════════════════════

/// Truncated SVD: keep top k singular values/vectors.
/// Returns (U_k: n×k, sigma_k: k, V_k: d×k).
pub fn truncated_svd(data: &[f64], n: usize, d: usize, k: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // A'A eigendecomposition
    let mut ata = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..d {
            for s in 0..n { ata[i * d + j] += data[s * d + i] * data[s * d + j]; }
        }
    }
    let (eigenvalues, eigenvectors) = crate::linalg_decomp::eig_symmetric(&ata, d);
    let mut idx: Vec<usize> = (0..d).collect();
    idx.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

    let mut sigma = Vec::with_capacity(k);
    let mut v_k = vec![0.0; d * k];
    for c in 0..k {
        let ev = eigenvalues[idx[c]].max(0.0).sqrt();
        sigma.push(ev);
        for i in 0..d { v_k[i * k + c] = eigenvectors[i * d + idx[c]]; }
    }

    // U_k = A V_k Σ^{-1}
    let mut u_k = vec![0.0; n * k];
    for i in 0..n {
        for c in 0..k {
            if sigma[c] > 1e-15 {
                let mut val = 0.0;
                for j in 0..d { val += data[i * d + j] * v_k[j * k + c]; }
                u_k[i * k + c] = val / sigma[c];
            }
        }
    }
    (u_k, sigma, v_k)
}

/// Explained variance ratio from singular values.
pub fn svd_explained_variance(singular_values: &[f64]) -> Vec<f64> {
    let total: f64 = singular_values.iter().map(|s| s * s).sum();
    if total < 1e-15 { return vec![0.0; singular_values.len()]; }
    singular_values.iter().map(|s| s * s / total).collect()
}

fn seed_state(seed: u64) -> [u64; 4] {
    let mut s = [seed, seed.wrapping_mul(6364136223846793005).wrapping_add(1),
                 seed.wrapping_mul(1442695040888963407).wrapping_add(3),
                 seed.wrapping_mul(2862933555777941757).wrapping_add(7)];
    for _ in 0..20 { prng_next(&mut s); } s
}
fn prng_next(s: &mut [u64; 4]) -> u64 {
    let r = (s[0].wrapping_add(s[3])).rotate_left(23).wrapping_add(s[0]);
    let t = s[1] << 17; s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = s[3].rotate_left(45); r
}
fn prng_f64(s: &mut [u64; 4]) -> f64 { (prng_next(s) >> 11) as f64 / (1u64 << 53) as f64 }
fn prng_normal(s: &mut [u64; 4]) -> f64 {
    let u1 = prng_f64(s).max(1e-15); let u2 = prng_f64(s);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_projection() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let proj = gaussian_random_projection(3, 2, 42);
        let result = apply_projection(&data, 2, 3, &proj, 2);
        assert_eq!(result.len(), 4); // 2×2
    }

    #[test]
    fn test_jl_dim() {
        let d = jl_min_dim(1000, 0.1);
        assert!(d > 100 && d < 10000);
    }

    #[test]
    fn test_classical_mds() {
        // 3 points in a line: dist = [0,1,2; 1,0,1; 2,1,0]
        let dist = vec![0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0];
        let coords = classical_mds(&dist, 3, 1);
        assert_eq!(coords.len(), 3);
    }

    #[test]
    fn test_factor_loadings() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 3.0, 6.0, 9.0, 12.0]; // 3×4
        let loadings = factor_loadings(&data, 3, 4, 2);
        assert_eq!(loadings.len(), 8); // 4×2
    }

    #[test]
    fn test_communalities() {
        let loadings = vec![0.8, 0.2, 0.6, 0.7]; // 2×2
        let comm = communalities(&loadings, 2, 2);
        assert_eq!(comm.len(), 2);
        assert!(comm[0] > 0.0 && comm[0] <= 1.0 + 1e-10);
    }

    #[test]
    fn test_truncated_svd() {
        let data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3×2
        let (u, sigma, v) = truncated_svd(&data, 3, 2, 1);
        assert_eq!(u.len(), 3); // 3×1
        assert_eq!(sigma.len(), 1);
        assert!(sigma[0] > 0.0);
    }

    #[test]
    fn test_explained_variance() {
        let sv = vec![3.0, 2.0, 1.0];
        let ev = svd_explained_variance(&sv);
        assert!((ev.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!(ev[0] > ev[1] && ev[1] > ev[2]);
    }
}
