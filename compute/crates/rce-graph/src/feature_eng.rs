//! Feature engineering: encoding, scaling, imputation, feature selection,
//! binning, interaction features, target encoding, date features.

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════
// SCALING / NORMALIZATION
// ═══════════════════════════════════════════════════════════════════

/// Min-max scaling to [0, 1].
pub fn minmax_scale(data: &[f64]) -> Vec<f64> {
    let min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if range < 1e-15 { return vec![0.0; data.len()]; }
    data.iter().map(|&x| (x - min) / range).collect()
}

/// Min-max scaling to [a, b].
pub fn minmax_scale_range(data: &[f64], a: f64, b: f64) -> Vec<f64> {
    let scaled = minmax_scale(data);
    scaled.iter().map(|&x| a + x * (b - a)).collect()
}

/// Standard scaling (z-score): (x - mean) / std.
pub fn standard_scale(data: &[f64]) -> Vec<f64> {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    if std < 1e-15 { return vec![0.0; data.len()]; }
    data.iter().map(|&x| (x - mean) / std).collect()
}

/// Robust scaling: (x - median) / IQR.
pub fn robust_scale(data: &[f64]) -> Vec<f64> {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let median = if n % 2 == 0 { (sorted[n/2 - 1] + sorted[n/2]) / 2.0 } else { sorted[n/2] };
    let q1 = sorted[n / 4];
    let q3 = sorted[3 * n / 4];
    let iqr = q3 - q1;
    if iqr < 1e-15 { return vec![0.0; data.len()]; }
    data.iter().map(|&x| (x - median) / iqr).collect()
}

/// Max-abs scaling: x / max(|x|).
pub fn maxabs_scale(data: &[f64]) -> Vec<f64> {
    let max_abs = data.iter().map(|x| x.abs()).fold(0.0, f64::max);
    if max_abs < 1e-15 { return vec![0.0; data.len()]; }
    data.iter().map(|&x| x / max_abs).collect()
}

/// L2 normalization (unit norm).
pub fn l2_normalize(data: &[f64]) -> Vec<f64> {
    let norm = data.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-15 { return vec![0.0; data.len()]; }
    data.iter().map(|&x| x / norm).collect()
}

/// L1 normalization.
pub fn l1_normalize(data: &[f64]) -> Vec<f64> {
    let norm: f64 = data.iter().map(|x| x.abs()).sum();
    if norm < 1e-15 { return vec![0.0; data.len()]; }
    data.iter().map(|&x| x / norm).collect()
}

/// Log transform: ln(1 + x).
pub fn log_transform(data: &[f64]) -> Vec<f64> {
    data.iter().map(|&x| (1.0 + x).ln()).collect()
}

/// Box-Cox transform (requires positive data, lambda ≠ 0).
pub fn box_cox(data: &[f64], lambda: f64) -> Vec<f64> {
    if lambda.abs() < 1e-15 {
        data.iter().map(|&x| x.ln()).collect()
    } else {
        data.iter().map(|&x| (x.powf(lambda) - 1.0) / lambda).collect()
    }
}

/// Yeo-Johnson transform (handles negative values).
pub fn yeo_johnson(data: &[f64], lambda: f64) -> Vec<f64> {
    data.iter().map(|&x| {
        if x >= 0.0 {
            if lambda.abs() < 1e-15 { (x + 1.0).ln() }
            else { ((x + 1.0).powf(lambda) - 1.0) / lambda }
        } else {
            if (2.0 - lambda).abs() < 1e-15 { -((-x + 1.0).ln()) }
            else { -((-x + 1.0).powf(2.0 - lambda) - 1.0) / (2.0 - lambda) }
        }
    }).collect()
}

/// Power transform.
pub fn power_transform(data: &[f64], power: f64) -> Vec<f64> {
    data.iter().map(|&x| x.signum() * x.abs().powf(power)).collect()
}

// ═══════════════════════════════════════════════════════════════════
// ENCODING
// ═══════════════════════════════════════════════════════════════════

/// One-hot encode integer labels. Returns n × n_classes matrix (row-major).
pub fn one_hot_encode(labels: &[usize], n_classes: usize) -> Vec<f64> {
    let n = labels.len();
    let mut encoded = vec![0.0; n * n_classes];
    for (i, &l) in labels.iter().enumerate() {
        if l < n_classes { encoded[i * n_classes + l] = 1.0; }
    }
    encoded
}

/// Label encode strings to integers.
pub fn label_encode(labels: &[String]) -> (Vec<usize>, Vec<String>) {
    let mut mapping = Vec::new();
    let mut encoded = Vec::new();
    for label in labels {
        let idx = mapping.iter().position(|x| x == label).unwrap_or_else(|| {
            mapping.push(label.clone());
            mapping.len() - 1
        });
        encoded.push(idx);
    }
    (encoded, mapping)
}

/// Ordinal encode with given order.
pub fn ordinal_encode(labels: &[String], order: &[String]) -> Vec<f64> {
    labels.iter().map(|l| {
        order.iter().position(|o| o == l).map(|p| p as f64).unwrap_or(f64::NAN)
    }).collect()
}

/// Target encode (mean of target per category).
pub fn target_encode(categories: &[String], targets: &[f64]) -> Vec<f64> {
    let mut sums: HashMap<&String, (f64, usize)> = HashMap::new();
    for (cat, &target) in categories.iter().zip(targets) {
        let entry = sums.entry(cat).or_insert((0.0, 0));
        entry.0 += target;
        entry.1 += 1;
    }
    let means: HashMap<&String, f64> = sums.iter()
        .map(|(k, (s, c))| (*k, s / *c as f64)).collect();
    categories.iter().map(|c| *means.get(c).unwrap_or(&0.0)).collect()
}

/// Frequency encode (replace category with its frequency).
pub fn frequency_encode(categories: &[String]) -> Vec<f64> {
    let n = categories.len() as f64;
    let mut counts: HashMap<&String, usize> = HashMap::new();
    for c in categories { *counts.entry(c).or_insert(0) += 1; }
    categories.iter().map(|c| *counts.get(c).unwrap_or(&0) as f64 / n).collect()
}

// ═══════════════════════════════════════════════════════════════════
// BINNING / DISCRETIZATION
// ═══════════════════════════════════════════════════════════════════

/// Equal-width binning. Returns bin indices (0-based).
pub fn equal_width_bins(data: &[f64], n_bins: usize) -> Vec<usize> {
    let min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let width = (max - min) / n_bins as f64;
    if width < 1e-15 { return vec![0; data.len()]; }
    data.iter().map(|&x| ((x - min) / width).floor() as usize).map(|b| b.min(n_bins - 1)).collect()
}

/// Equal-frequency binning (quantile-based). Returns bin indices.
pub fn equal_freq_bins(data: &[f64], n_bins: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let bin_size = (data.len() as f64 / n_bins as f64).ceil() as usize;
    let mut bins = vec![0; data.len()];
    for (rank, (orig_idx, _)) in indexed.iter().enumerate() {
        bins[*orig_idx] = (rank / bin_size).min(n_bins - 1);
    }
    bins
}

// ═══════════════════════════════════════════════════════════════════
// IMPUTATION
// ═══════════════════════════════════════════════════════════════════

/// Fill NaN values with mean.
pub fn impute_mean(data: &[f64]) -> Vec<f64> {
    let valid: Vec<f64> = data.iter().filter(|x| x.is_finite()).copied().collect();
    let mean = if valid.is_empty() { 0.0 } else { valid.iter().sum::<f64>() / valid.len() as f64 };
    data.iter().map(|&x| if x.is_finite() { x } else { mean }).collect()
}

/// Fill NaN values with median.
pub fn impute_median(data: &[f64]) -> Vec<f64> {
    let mut valid: Vec<f64> = data.iter().filter(|x| x.is_finite()).copied().collect();
    valid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if valid.is_empty() { 0.0 }
    else if valid.len() % 2 == 0 { (valid[valid.len()/2 - 1] + valid[valid.len()/2]) / 2.0 }
    else { valid[valid.len()/2] };
    data.iter().map(|&x| if x.is_finite() { x } else { median }).collect()
}

/// Fill NaN with constant.
pub fn impute_constant(data: &[f64], value: f64) -> Vec<f64> {
    data.iter().map(|&x| if x.is_finite() { x } else { value }).collect()
}

/// Forward fill NaN values.
pub fn impute_ffill(data: &[f64]) -> Vec<f64> {
    let mut result = data.to_vec();
    let mut last_valid = f64::NAN;
    for v in result.iter_mut() {
        if v.is_finite() { last_valid = *v; }
        else if last_valid.is_finite() { *v = last_valid; }
    }
    result
}

/// Backward fill NaN values.
pub fn impute_bfill(data: &[f64]) -> Vec<f64> {
    let mut result = data.to_vec();
    let mut next_valid = f64::NAN;
    for v in result.iter_mut().rev() {
        if v.is_finite() { next_valid = *v; }
        else if next_valid.is_finite() { *v = next_valid; }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// INTERACTION FEATURES
// ═══════════════════════════════════════════════════════════════════

/// Polynomial features of degree 2 for a feature vector.
/// [x1, x2] → [x1, x2, x1², x1*x2, x2²].
pub fn polynomial_features(features: &[f64], degree: usize) -> Vec<f64> {
    if degree < 2 { return features.to_vec(); }
    let mut result = features.to_vec();
    let d = features.len();
    // Degree 2
    for i in 0..d {
        for j in i..d {
            result.push(features[i] * features[j]);
        }
    }
    result
}

/// Interaction features (pairwise products).
pub fn interaction_features(features: &[f64]) -> Vec<f64> {
    let d = features.len();
    let mut result = Vec::new();
    for i in 0..d {
        for j in i+1..d {
            result.push(features[i] * features[j]);
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// FEATURE SELECTION
// ═══════════════════════════════════════════════════════════════════

/// Variance threshold: select features with variance > threshold.
/// data: n × d (row-major). Returns indices of selected features.
pub fn variance_threshold(data: &[f64], n: usize, d: usize, threshold: f64) -> Vec<usize> {
    (0..d).filter(|&j| {
        let mean = (0..n).map(|i| data[i * d + j]).sum::<f64>() / n as f64;
        let var = (0..n).map(|i| (data[i * d + j] - mean).powi(2)).sum::<f64>() / n as f64;
        var > threshold
    }).collect()
}

/// Mutual information between feature and binary target (simplified).
pub fn mutual_info_score(feature: &[f64], target: &[bool]) -> f64 {
    let n = feature.len() as f64;
    let median = {
        let mut sorted = feature.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };

    let mut counts = [[0.0; 2]; 2]; // [feature_bin][target]
    for i in 0..feature.len() {
        let fb = if feature[i] >= median { 1 } else { 0 };
        let tb = if target[i] { 1 } else { 0 };
        counts[fb][tb] += 1.0;
    }

    let mut mi = 0.0;
    let p_f = [counts[0][0] + counts[0][1], counts[1][0] + counts[1][1]];
    let p_t = [counts[0][0] + counts[1][0], counts[0][1] + counts[1][1]];
    for fb in 0..2 {
        for tb in 0..2 {
            let p_joint = counts[fb][tb] / n;
            let p_marginal = p_f[fb] / n * p_t[tb] / n;
            if p_joint > 1e-15 && p_marginal > 1e-15 {
                mi += p_joint * (p_joint / p_marginal).ln();
            }
        }
    }
    mi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minmax_scale() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let scaled = minmax_scale(&data);
        assert!((scaled[0] - 0.0).abs() < 1e-10);
        assert!((scaled[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_standard_scale() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let scaled = standard_scale(&data);
        let mean: f64 = scaled.iter().sum::<f64>() / 5.0;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_one_hot() {
        let labels = vec![0, 1, 2, 1];
        let encoded = one_hot_encode(&labels, 3);
        assert_eq!(encoded.len(), 12); // 4 × 3
        assert_eq!(encoded[0], 1.0); // first row, class 0
        assert_eq!(encoded[4], 1.0); // second row, class 1
    }

    #[test]
    fn test_label_encode() {
        let labels: Vec<String> = vec!["cat", "dog", "cat", "bird"].into_iter().map(String::from).collect();
        let (encoded, mapping) = label_encode(&labels);
        assert_eq!(encoded[0], encoded[2]); // both "cat"
        assert_ne!(encoded[0], encoded[1]); // cat ≠ dog
    }

    #[test]
    fn test_equal_width_bins() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let bins = equal_width_bins(&data, 5);
        assert_eq!(bins[0], 0);
        assert_eq!(bins[9], 4);
    }

    #[test]
    fn test_impute_mean() {
        let data = vec![1.0, 2.0, f64::NAN, 4.0];
        let imputed = impute_mean(&data);
        assert!((imputed[2] - 7.0 / 3.0).abs() < 1e-10); // mean of 1,2,4
    }

    #[test]
    fn test_polynomial_features() {
        let f = vec![2.0, 3.0];
        let poly = polynomial_features(&f, 2);
        // [x1, x2, x1², x1*x2, x2²] = [2, 3, 4, 6, 9]
        assert_eq!(poly, vec![2.0, 3.0, 4.0, 6.0, 9.0]);
    }

    #[test]
    fn test_l2_normalize() {
        let data = vec![3.0, 4.0];
        let norm = l2_normalize(&data);
        assert!((norm[0] - 0.6).abs() < 1e-10);
        assert!((norm[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_log_transform() {
        let data = vec![0.0, 1.0, 9.0];
        let t = log_transform(&data);
        assert!((t[0] - 0.0).abs() < 1e-10); // ln(1) = 0
        assert!((t[1] - 2.0f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_frequency_encode() {
        let cats: Vec<String> = vec!["a", "b", "a", "a"].into_iter().map(String::from).collect();
        let encoded = frequency_encode(&cats);
        assert!((encoded[0] - 0.75).abs() < 1e-10); // "a" appears 3/4
    }
}
