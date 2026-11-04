//! Distance and similarity metrics: comprehensive collection for vectors,
//! sets, strings, distributions, time series.

// ═══════════════════════════════════════════════════════════════════
// VECTOR DISTANCES
// ═══════════════════════════════════════════════════════════════════

pub fn euclidean(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x,y)| (x-y).powi(2)).sum::<f64>().sqrt() }
pub fn sqeuclidean(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x,y)| (x-y).powi(2)).sum() }
pub fn manhattan(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x,y)| (x-y).abs()).sum() }
pub fn chebyshev(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x,y)| (x-y).abs()).fold(0.0, f64::max) }
pub fn minkowski(a: &[f64], b: &[f64], p: f64) -> f64 { a.iter().zip(b).map(|(x,y)| (x-y).abs().powf(p)).sum::<f64>().powf(1.0/p) }
pub fn cosine_dist(a: &[f64], b: &[f64]) -> f64 { 1.0 - cosine_sim(a, b) }
pub fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x,y)| x*y).sum();
    let na = a.iter().map(|x| x*x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x*x).sum::<f64>().sqrt();
    if na < 1e-15 || nb < 1e-15 { 0.0 } else { d / (na * nb) }
}
pub fn canberra(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x,y)| { let d = x.abs() + y.abs(); if d > 1e-15 { (x-y).abs() / d } else { 0.0 } }).sum()
}
pub fn braycurtis(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x,y)| (x-y).abs()).sum();
    let den: f64 = a.iter().zip(b).map(|(x,y)| x.abs() + y.abs()).sum();
    if den < 1e-15 { 0.0 } else { num / den }
}
pub fn correlation_dist(a: &[f64], b: &[f64]) -> f64 { 1.0 - crate::array_ops::correlation(a, b) }
pub fn mahalanobis(x: &[f64], mu: &[f64], cov_inv: &[f64], d: usize) -> f64 {
    let diff: Vec<f64> = x.iter().zip(mu).map(|(a,b)| a - b).collect();
    let mut result = 0.0;
    for i in 0..d { for j in 0..d { result += diff[i] * cov_inv[i*d+j] * diff[j]; } }
    result.max(0.0).sqrt()
}

// ═══════════════════════════════════════════════════════════════════
// DISTRIBUTION DISTANCES
// ═══════════════════════════════════════════════════════════════════

pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 { crate::info_theory::kl_divergence(p, q) }
pub fn js_divergence(p: &[f64], q: &[f64]) -> f64 { crate::info_theory::js_divergence(p, q) }
pub fn hellinger(p: &[f64], q: &[f64]) -> f64 { crate::info_theory::hellinger_distance(p, q) }
pub fn total_variation(p: &[f64], q: &[f64]) -> f64 { crate::info_theory::total_variation(p, q) }
pub fn wasserstein_1d(p: &[f64], q: &[f64]) -> f64 { crate::info_theory::wasserstein_1d(p, q) }
pub fn chi_squared(p: &[f64], q: &[f64]) -> f64 { crate::info_theory::chi_squared_divergence(p, q) }
pub fn bhattacharyya(p: &[f64], q: &[f64]) -> f64 {
    let bc: f64 = p.iter().zip(q).map(|(a,b)| (a * b).sqrt()).sum();
    if bc > 0.0 { -bc.ln() } else { f64::INFINITY }
}

// ═══════════════════════════════════════════════════════════════════
// SET DISTANCES
// ═══════════════════════════════════════════════════════════════════

pub fn jaccard_dist(a: &[f64], b: &[f64]) -> f64 { 1.0 - crate::sets::jaccard_index(a, b) }
pub fn dice_dist(a: &[f64], b: &[f64]) -> f64 { 1.0 - crate::sets::dice_coefficient(a, b) }
pub fn overlap_dist(a: &[f64], b: &[f64]) -> f64 { 1.0 - crate::sets::overlap_coefficient(a, b) }
pub fn hamming_binary(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).filter(|(x,y)| (x.signum() - y.signum()).abs() > 0.5).count() as f64 / a.len() as f64
}

// ═══════════════════════════════════════════════════════════════════
// STRING DISTANCES (delegating to existing modules)
// ═══════════════════════════════════════════════════════════════════

pub fn levenshtein(a: &str, b: &str) -> usize { crate::text_analysis::levenshtein(a, b) }
pub fn damerau_levenshtein(a: &str, b: &str) -> usize { crate::text_analysis::damerau_levenshtein(a, b) }
pub fn jaro(a: &str, b: &str) -> f64 { crate::string_algorithms::jaro_similarity(a, b) }
pub fn jaro_winkler(a: &str, b: &str) -> f64 { crate::string_algorithms::jaro_winkler(a, b) }
pub fn normalized_lev(a: &str, b: &str) -> f64 { crate::string_algorithms::normalized_levenshtein(a, b) }

// ═══════════════════════════════════════════════════════════════════
// TIME SERIES DISTANCES
// ═══════════════════════════════════════════════════════════════════

/// Dynamic Time Warping distance.
pub fn dtw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();
    let mut dp = vec![f64::INFINITY; (n + 1) * (m + 1)];
    dp[0] = 0.0;
    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i-1] - b[j-1]).abs();
            dp[i * (m+1) + j] = cost + dp[(i-1)*(m+1)+j].min(dp[i*(m+1)+j-1]).min(dp[(i-1)*(m+1)+j-1]);
        }
    }
    dp[n * (m+1) + m]
}

/// Frechet distance (discrete).
pub fn frechet(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();
    let mut dp = vec![f64::NEG_INFINITY; n * m];
    dp[0] = (a[0] - b[0]).abs();
    for i in 1..n { dp[i * m] = dp[(i-1)*m].max((a[i] - b[0]).abs()); }
    for j in 1..m { dp[j] = dp[j-1].max((a[0] - b[j]).abs()); }
    for i in 1..n {
        for j in 1..m {
            let prev = dp[(i-1)*m+j].min(dp[i*m+j-1]).min(dp[(i-1)*m+j-1]);
            dp[i*m+j] = prev.max((a[i] - b[j]).abs());
        }
    }
    dp[n*m - 1]
}

/// Longest Common Subsequence distance for time series (using threshold).
pub fn lcss_distance(a: &[f64], b: &[f64], epsilon: f64) -> f64 {
    let n = a.len();
    let m = b.len();
    let mut dp = vec![0usize; (n+1)*(m+1)];
    for i in 1..=n {
        for j in 1..=m {
            if (a[i-1] - b[j-1]).abs() <= epsilon { dp[i*(m+1)+j] = dp[(i-1)*(m+1)+j-1] + 1; }
            else { dp[i*(m+1)+j] = dp[(i-1)*(m+1)+j].max(dp[i*(m+1)+j-1]); }
        }
    }
    1.0 - dp[n*(m+1)+m] as f64 / n.max(m) as f64
}

/// Cross-correlation distance: 1 - max(normalized cross-correlation).
pub fn xcorr_distance(a: &[f64], b: &[f64]) -> f64 {
    let ma = crate::array_ops::mean(a);
    let mb = crate::array_ops::mean(b);
    let sa = crate::array_ops::std_dev(a);
    let sb = crate::array_ops::std_dev(b);
    if sa < 1e-15 || sb < 1e-15 { return 1.0; }
    let n = a.len();
    let mut max_corr = f64::NEG_INFINITY;
    for lag in -(n as i64 / 2)..=(n as i64 / 2) {
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..n {
            let j = i as i64 + lag;
            if j >= 0 && (j as usize) < b.len() {
                sum += (a[i] - ma) * (b[j as usize] - mb);
                count += 1;
            }
        }
        if count > 0 { max_corr = max_corr.max(sum / (count as f64 * sa * sb)); }
    }
    1.0 - max_corr.max(0.0)
}

// ═══════════════════════════════════════════════════════════════════
// PAIRWISE DISTANCE MATRIX
// ═══════════════════════════════════════════════════════════════════

/// Pairwise distance matrix for n×d data.
pub fn pdist(data: &[f64], n: usize, d: usize, metric: impl Fn(&[f64], &[f64]) -> f64) -> Vec<f64> {
    let mut mat = vec![0.0; n * n];
    for i in 0..n {
        for j in i+1..n {
            let dist = metric(&data[i*d..(i+1)*d], &data[j*d..(j+1)*d]);
            mat[i*n+j] = dist; mat[j*n+i] = dist;
        }
    }
    mat
}

/// Condensed distance vector (upper triangle, n*(n-1)/2 elements).
pub fn pdist_condensed(data: &[f64], n: usize, d: usize, metric: impl Fn(&[f64], &[f64]) -> f64) -> Vec<f64> {
    let mut dists = Vec::with_capacity(n * (n-1) / 2);
    for i in 0..n { for j in i+1..n { dists.push(metric(&data[i*d..(i+1)*d], &data[j*d..(j+1)*d])); } }
    dists
}

/// Convert condensed to square distance matrix.
pub fn squareform(condensed: &[f64], n: usize) -> Vec<f64> {
    let mut mat = vec![0.0; n * n];
    let mut idx = 0;
    for i in 0..n { for j in i+1..n { mat[i*n+j] = condensed[idx]; mat[j*n+i] = condensed[idx]; idx += 1; } }
    mat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean() { assert!((euclidean(&[0.0,0.0], &[3.0,4.0]) - 5.0).abs() < 1e-10); }

    #[test]
    fn test_cosine() {
        assert!((cosine_sim(&[1.0,0.0], &[1.0,0.0]) - 1.0).abs() < 1e-10);
        assert!(cosine_sim(&[1.0,0.0], &[0.0,1.0]).abs() < 1e-10);
    }

    #[test]
    fn test_dtw() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        assert!(dtw(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn test_dtw_shifted() {
        let a = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        assert!(dtw(&a, &b) < euclidean(&a, &b)); // DTW handles shift better
    }

    #[test]
    fn test_canberra() {
        assert!((canberra(&[1.0, 2.0], &[1.0, 2.0])).abs() < 1e-10);
    }

    #[test]
    fn test_braycurtis() {
        assert!((braycurtis(&[1.0, 2.0], &[1.0, 2.0])).abs() < 1e-10);
    }

    #[test]
    fn test_frechet() {
        let a = vec![0.0, 1.0, 2.0];
        let b = vec![0.0, 1.0, 2.0];
        assert!(frechet(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn test_pdist() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let mat = pdist(&data, 3, 2, euclidean);
        assert_eq!(mat.len(), 9);
        assert!((mat[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_squareform() {
        let condensed = vec![1.0, 2.0, 3.0]; // 3 points
        let sq = squareform(&condensed, 3);
        assert_eq!(sq[1], 1.0); // (0,1)
        assert_eq!(sq[3], 1.0); // (1,0)
    }

    #[test]
    fn test_bhattacharyya() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        assert!(bhattacharyya(&p, &q).abs() < 1e-10);
    }
}
