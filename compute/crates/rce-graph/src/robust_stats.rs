//! Robust statistics: trimmed estimators, M-estimators, breakdown point,
//! influence functions, robust covariance, outlier detection.

// ═══════════════════════════════════════════════════════════════════
// ROBUST LOCATION ESTIMATORS
// ═══════════════════════════════════════════════════════════════════

/// Trimmed mean (remove fraction from each tail).
pub fn trimmed_mean(data: &[f64], fraction: f64) -> f64 {
    let mut s = data.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let trim = (s.len() as f64 * fraction) as usize;
    let trimmed = &s[trim..s.len() - trim];
    if trimmed.is_empty() { return s.iter().sum::<f64>() / s.len() as f64; }
    trimmed.iter().sum::<f64>() / trimmed.len() as f64
}

/// Winsorized mean (replace extremes with boundary values).
pub fn winsorized_mean(data: &[f64], fraction: f64) -> f64 {
    let mut s = data.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let trim = (s.len() as f64 * fraction) as usize;
    let lo = s[trim];
    let hi = s[s.len() - 1 - trim];
    let winsorized: Vec<f64> = s.iter().map(|&v| v.max(lo).min(hi)).collect();
    winsorized.iter().sum::<f64>() / winsorized.len() as f64
}

/// Hodges-Lehmann estimator: median of pairwise averages.
pub fn hodges_lehmann(data: &[f64]) -> f64 {
    let n = data.len();
    let mut avgs = Vec::with_capacity(n * (n + 1) / 2);
    for i in 0..n {
        for j in i..n { avgs.push((data[i] + data[j]) / 2.0); }
    }
    avgs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = avgs.len();
    if m % 2 == 0 { (avgs[m/2 - 1] + avgs[m/2]) / 2.0 } else { avgs[m/2] }
}

/// Biweight midcorrelation (robust correlation).
pub fn biweight_midcorrelation(x: &[f64], y: &[f64]) -> f64 {
    let med_x = median(x);
    let med_y = median(y);
    let mad_x = mad(x).max(1e-15);
    let mad_y = mad(y).max(1e-15);
    let u: Vec<f64> = x.iter().map(|&v| (v - med_x) / (9.0 * mad_x)).collect();
    let v: Vec<f64> = y.iter().map(|&v| (v - med_y) / (9.0 * mad_y)).collect();
    let a: Vec<f64> = x.iter().zip(&u).map(|(&xi, &ui)| {
        let w = if ui.abs() < 1.0 { (1.0 - ui * ui).powi(2) } else { 0.0 };
        (xi - med_x) * w
    }).collect();
    let b: Vec<f64> = y.iter().zip(&v).map(|(&yi, &vi)| {
        let w = if vi.abs() < 1.0 { (1.0 - vi * vi).powi(2) } else { 0.0 };
        (yi - med_y) * w
    }).collect();
    let num: f64 = a.iter().zip(&b).map(|(ai, bi)| ai * bi).sum();
    let den_a: f64 = a.iter().map(|ai| ai * ai).sum::<f64>().sqrt();
    let den_b: f64 = b.iter().map(|bi| bi * bi).sum::<f64>().sqrt();
    if den_a < 1e-15 || den_b < 1e-15 { return 0.0; }
    num / (den_a * den_b)
}

// ═══════════════════════════════════════════════════════════════════
// ROBUST SCALE ESTIMATORS
// ═══════════════════════════════════════════════════════════════════

/// Median Absolute Deviation (MAD).
pub fn mad(data: &[f64]) -> f64 {
    let med = median(data);
    let deviations: Vec<f64> = data.iter().map(|x| (x - med).abs()).collect();
    median(&deviations)
}

/// MAD-based standard deviation estimate: σ̂ = 1.4826 × MAD.
pub fn mad_std(data: &[f64]) -> f64 { 1.4826 * mad(data) }

/// Interquartile range.
pub fn iqr(data: &[f64]) -> f64 {
    crate::probability::quantile(data, 0.75) - crate::probability::quantile(data, 0.25)
}

/// Qn estimator (Rousseeuw & Croux): first quartile of |x_i - x_j|.
pub fn qn_scale(data: &[f64]) -> f64 {
    let n = data.len();
    let mut diffs = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in i + 1..n { diffs.push((data[i] - data[j]).abs()); }
    }
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = diffs.len() / 4; // approximate first quartile
    2.2219 * diffs[k] // consistency factor for normal
}

/// Sn estimator (Rousseeuw & Croux): median of median |x_i - x_j|.
pub fn sn_scale(data: &[f64]) -> f64 {
    let n = data.len();
    let medians: Vec<f64> = (0..n).map(|i| {
        let mut diffs: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| (data[i] - data[j]).abs()).collect();
        diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        median(&diffs)
    }).collect();
    1.1926 * median(&medians) // consistency factor
}

// ═══════════════════════════════════════════════════════════════════
// M-ESTIMATORS
// ═══════════════════════════════════════════════════════════════════

/// Huber M-estimator of location.
pub fn huber_location(data: &[f64], k: f64, max_iter: usize) -> f64 {
    let mut mu = median(data);
    let sigma = mad_std(data).max(1e-15);
    for _ in 0..max_iter {
        let mut num = 0.0;
        let mut den = 0.0;
        for &x in data {
            let u = (x - mu) / sigma;
            let w = if u.abs() <= k { 1.0 } else { k / u.abs() };
            num += w * x;
            den += w;
        }
        let new_mu = num / den;
        if (new_mu - mu).abs() < 1e-10 { break; }
        mu = new_mu;
    }
    mu
}

/// Tukey biweight M-estimator of location.
pub fn biweight_location(data: &[f64], c: f64, max_iter: usize) -> f64 {
    let mut mu = median(data);
    let sigma = mad_std(data).max(1e-15);
    for _ in 0..max_iter {
        let mut num = 0.0;
        let mut den = 0.0;
        for &x in data {
            let u = (x - mu) / (c * sigma);
            if u.abs() < 1.0 {
                let w = (1.0 - u * u).powi(2);
                num += w * x;
                den += w;
            }
        }
        if den < 1e-15 { break; }
        let new_mu = num / den;
        if (new_mu - mu).abs() < 1e-10 { break; }
        mu = new_mu;
    }
    mu
}

// ═══════════════════════════════════════════════════════════════════
// OUTLIER DETECTION
// ═══════════════════════════════════════════════════════════════════

/// Z-score outlier detection. Returns indices of outliers.
pub fn zscore_outliers(data: &[f64], threshold: f64) -> Vec<usize> {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64).sqrt();
    if std < 1e-15 { return vec![]; }
    (0..data.len()).filter(|&i| ((data[i] - mean) / std).abs() > threshold).collect()
}

/// Modified Z-score outlier detection (using median and MAD).
pub fn modified_zscore_outliers(data: &[f64], threshold: f64) -> Vec<usize> {
    let med = median(data);
    let mad_val = mad(data);
    if mad_val < 1e-15 { return vec![]; }
    (0..data.len()).filter(|&i| {
        let mz = 0.6745 * (data[i] - med) / mad_val;
        mz.abs() > threshold
    }).collect()
}

/// IQR-based outlier detection (Tukey's fences).
pub fn iqr_outliers(data: &[f64], k: f64) -> Vec<usize> {
    let q1 = crate::probability::quantile(data, 0.25);
    let q3 = crate::probability::quantile(data, 0.75);
    let iqr = q3 - q1;
    let lo = q1 - k * iqr;
    let hi = q3 + k * iqr;
    (0..data.len()).filter(|&i| data[i] < lo || data[i] > hi).collect()
}

/// Grubbs test statistic for single outlier.
pub fn grubbs_statistic(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    if std < 1e-15 { return 0.0; }
    data.iter().map(|&x| ((x - mean) / std).abs()).fold(0.0, f64::max)
}

/// Dixon's Q test statistic (for smallest dataset).
pub fn dixon_q(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n < 3 { return 0.0; }
    let range = sorted[n - 1] - sorted[0];
    if range < 1e-15 { return 0.0; }
    let q_lo = (sorted[1] - sorted[0]) / range;
    let q_hi = (sorted[n - 1] - sorted[n - 2]) / range;
    q_lo.max(q_hi)
}

/// DBSCAN-based outlier detection (points not in any cluster).
pub fn dbscan_outliers(data: &[f64], eps: f64, min_pts: usize) -> Vec<usize> {
    let n = data.len();
    let mut labels = vec![-1i32; n]; // -1 = noise
    let mut cluster = 0i32;
    for i in 0..n {
        if labels[i] != -1 { continue; }
        let neighbors: Vec<usize> = (0..n).filter(|&j| (data[i] - data[j]).abs() <= eps).collect();
        if neighbors.len() < min_pts { continue; }
        labels[i] = cluster;
        let mut queue = std::collections::VecDeque::from(neighbors);
        while let Some(j) = queue.pop_front() {
            if labels[j] == -1 {
                labels[j] = cluster;
                let j_neighbors: Vec<usize> = (0..n).filter(|&k| (data[j] - data[k]).abs() <= eps).collect();
                if j_neighbors.len() >= min_pts { queue.extend(j_neighbors); }
            }
        }
        cluster += 1;
    }
    (0..n).filter(|&i| labels[i] == -1).collect()
}

// ═══════════════════════════════════════════════════════════════════
// ROBUST REGRESSION DIAGNOSTICS
// ═══════════════════════════════════════════════════════════════════

/// Breakdown point of an estimator: fraction of data that can be corrupted.
pub fn breakdown_point_median() -> f64 { 0.5 }
pub fn breakdown_point_mean() -> f64 { 0.0 }
pub fn breakdown_point_trimmed_mean(fraction: f64) -> f64 { fraction }

/// Influence function of the mean at point x.
pub fn influence_mean(x: f64, _mu: f64) -> f64 { x }

/// Influence function of the median (sign function).
pub fn influence_median(x: f64, theta: f64) -> f64 {
    if x > theta { 0.5 } else if x < theta { -0.5 } else { 0.0 }
}

/// Leverage points: observations with high influence on fit.
/// Returns indices where hat diagonal > 2p/n.
pub fn leverage_outliers(hat_diag: &[f64], n: usize, p: usize) -> Vec<usize> {
    let threshold = 2.0 * p as f64 / n as f64;
    (0..n).filter(|&i| hat_diag[i] > threshold).collect()
}

fn median(data: &[f64]) -> f64 {
    let mut s = data.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 0 { (s[n/2 - 1] + s[n/2]) / 2.0 } else { s[n/2] }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trimmed_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 100.0]; // outlier
        let tm = trimmed_mean(&data, 0.2);
        assert!((tm - 3.0).abs() < 1e-10); // trims 1 from each side
    }

    #[test]
    fn test_mad() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = mad(&data);
        assert!(m > 0.0);
    }

    #[test]
    fn test_huber_location() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // outlier
        let mu = huber_location(&data, 1.345, 50);
        assert!(mu < 10.0); // not pulled by outlier
    }

    #[test]
    fn test_zscore_outliers() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let outliers = zscore_outliers(&data, 2.0);
        assert!(outliers.contains(&5)); // index of 100.0
    }

    #[test]
    fn test_iqr_outliers() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let outliers = iqr_outliers(&data, 1.5);
        assert!(outliers.contains(&5));
    }

    #[test]
    fn test_hodges_lehmann() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let hl = hodges_lehmann(&data);
        assert!((hl - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_grubbs() {
        let data = vec![1.0, 2.0, 3.0, 100.0];
        let g = grubbs_statistic(&data);
        assert!(g > 1.0); // outlier should give large G
    }

    #[test]
    fn test_dixon_q() {
        let data = vec![1.0, 2.0, 3.0, 100.0];
        let q = dixon_q(&data);
        assert!(q > 0.0 && q <= 1.0);
    }

    #[test]
    fn test_biweight_location() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let mu = biweight_location(&data, 9.0, 50);
        assert!(mu < 10.0);
    }
}
