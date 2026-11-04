//! Data validation: type checking, range checking, missing data handling,
//! schema validation, data quality metrics, anomaly flags.

// ═══════════════════════════════════════════════════════════════════
// MISSING DATA
// ═══════════════════════════════════════════════════════════════════

/// Count NaN values.
pub fn count_nan(data: &[f64]) -> usize { data.iter().filter(|x| x.is_nan()).count() }

/// Count non-NaN values.
pub fn count_valid(data: &[f64]) -> usize { data.iter().filter(|x| x.is_finite()).count() }

/// Missing data fraction.
pub fn missing_fraction(data: &[f64]) -> f64 { count_nan(data) as f64 / data.len() as f64 }

/// Indices of NaN values.
pub fn nan_indices(data: &[f64]) -> Vec<usize> { (0..data.len()).filter(|&i| data[i].is_nan()).collect() }

/// Indices of valid (finite) values.
pub fn valid_indices(data: &[f64]) -> Vec<usize> { (0..data.len()).filter(|&i| data[i].is_finite()).collect() }

/// Drop NaN values, return clean data.
pub fn dropna(data: &[f64]) -> Vec<f64> { data.iter().filter(|x| x.is_finite()).copied().collect() }

/// Fill NaN with a value.
pub fn fillna(data: &[f64], value: f64) -> Vec<f64> { data.iter().map(|&x| if x.is_finite() { x } else { value }).collect() }

/// Forward fill NaN.
pub fn ffill(data: &[f64]) -> Vec<f64> { crate::feature_eng::impute_ffill(data) }

/// Backward fill NaN.
pub fn bfill(data: &[f64]) -> Vec<f64> { crate::feature_eng::impute_bfill(data) }

/// Linear interpolation of NaN values.
pub fn interpolate_nan(data: &[f64]) -> Vec<f64> {
    let mut result = data.to_vec();
    let n = result.len();
    for i in 0..n {
        if !result[i].is_nan() { continue; }
        let mut left = None;
        let mut right = None;
        for j in (0..i).rev() { if result[j].is_finite() { left = Some((j, result[j])); break; } }
        for j in i+1..n { if result[j].is_finite() { right = Some((j, result[j])); break; } }
        result[i] = match (left, right) {
            (Some((li, lv)), Some((ri, rv))) => lv + (rv - lv) * (i - li) as f64 / (ri - li) as f64,
            (Some((_, lv)), None) => lv,
            (None, Some((_, rv))) => rv,
            (None, None) => 0.0,
        };
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// RANGE VALIDATION
// ═══════════════════════════════════════════════════════════════════

/// Check all values in range [lo, hi].
pub fn in_range(data: &[f64], lo: f64, hi: f64) -> bool { data.iter().all(|&x| x >= lo && x <= hi) }

/// Indices of out-of-range values.
pub fn out_of_range(data: &[f64], lo: f64, hi: f64) -> Vec<usize> { (0..data.len()).filter(|&i| data[i] < lo || data[i] > hi).collect() }

/// Clamp all values to [lo, hi].
pub fn clamp_range(data: &[f64], lo: f64, hi: f64) -> Vec<f64> { data.iter().map(|&x| x.max(lo).min(hi)).collect() }

/// Check for infinite values.
pub fn has_inf(data: &[f64]) -> bool { data.iter().any(|x| x.is_infinite()) }

/// Check for NaN values.
pub fn has_nan(data: &[f64]) -> bool { data.iter().any(|x| x.is_nan()) }

/// Replace infinite with NaN.
pub fn inf_to_nan(data: &[f64]) -> Vec<f64> { data.iter().map(|&x| if x.is_infinite() { f64::NAN } else { x }).collect() }

/// Replace negative values with zero.
pub fn clip_negative(data: &[f64]) -> Vec<f64> { data.iter().map(|&x| x.max(0.0)).collect() }

// ═══════════════════════════════════════════════════════════════════
// DATA QUALITY METRICS
// ═══════════════════════════════════════════════════════════════════

/// Completeness: fraction of non-missing values.
pub fn completeness(data: &[f64]) -> f64 { 1.0 - missing_fraction(data) }

/// Uniqueness: fraction of distinct values.
pub fn uniqueness(data: &[f64]) -> f64 {
    let valid = dropna(data);
    if valid.is_empty() { return 0.0; }
    crate::sets::nunique(&valid) as f64 / valid.len() as f64
}

/// Consistency check: is data monotonically increasing?
pub fn is_monotone_increasing(data: &[f64]) -> bool { data.windows(2).all(|w| w[1] >= w[0] || w[0].is_nan() || w[1].is_nan()) }

/// Consistency check: is data monotonically decreasing?
pub fn is_monotone_decreasing(data: &[f64]) -> bool { data.windows(2).all(|w| w[1] <= w[0] || w[0].is_nan() || w[1].is_nan()) }

/// Count duplicate values.
pub fn count_duplicates(data: &[f64]) -> usize {
    let valid = dropna(data);
    valid.len() - crate::sets::nunique(&valid)
}

/// Entropy of value distribution (higher = more diverse).
pub fn data_entropy(data: &[f64]) -> f64 {
    let counts = crate::sets::value_counts(&dropna(data));
    let n = dropna(data).len() as f64;
    counts.iter().map(|(_, c)| {
        let p = *c as f64 / n;
        if p > 0.0 { -p * p.log2() } else { 0.0 }
    }).sum()
}

/// Coefficient of variation (std / mean), measure of relative variability.
pub fn cv(data: &[f64]) -> f64 {
    let valid = dropna(data);
    let m = crate::array_ops::mean(&valid);
    if m.abs() < 1e-15 { return 0.0; }
    crate::array_ops::std_dev(&valid) / m.abs()
}

// ═══════════════════════════════════════════════════════════════════
// SUMMARY STATISTICS
// ═══════════════════════════════════════════════════════════════════

/// Five-number summary: (min, Q1, median, Q3, max).
pub fn five_number_summary(data: &[f64]) -> (f64, f64, f64, f64, f64) {
    let valid = dropna(data);
    if valid.is_empty() { return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN); }
    (
        crate::array_ops::min_val(&valid),
        crate::probability::quantile(&valid, 0.25),
        crate::probability::quantile(&valid, 0.5),
        crate::probability::quantile(&valid, 0.75),
        crate::array_ops::max_val(&valid),
    )
}

/// Describe: count, mean, std, min, 25%, 50%, 75%, max.
pub fn describe(data: &[f64]) -> Vec<(String, f64)> {
    let valid = dropna(data);
    let n = valid.len();
    if n == 0 { return vec![("count".into(), 0.0)]; }
    vec![
        ("count".into(), n as f64),
        ("mean".into(), crate::array_ops::mean(&valid)),
        ("std".into(), crate::array_ops::std_dev(&valid)),
        ("min".into(), crate::array_ops::min_val(&valid)),
        ("25%".into(), crate::probability::quantile(&valid, 0.25)),
        ("50%".into(), crate::probability::quantile(&valid, 0.50)),
        ("75%".into(), crate::probability::quantile(&valid, 0.75)),
        ("max".into(), crate::array_ops::max_val(&valid)),
    ]
}

/// Value range.
pub fn data_range(data: &[f64]) -> f64 {
    let valid = dropna(data);
    if valid.is_empty() { return 0.0; }
    crate::array_ops::max_val(&valid) - crate::array_ops::min_val(&valid)
}

/// Skewness.
pub fn skewness(data: &[f64]) -> f64 { crate::probability::sample_skewness(&dropna(data)) }

/// Kurtosis.
pub fn kurtosis(data: &[f64]) -> f64 { crate::probability::sample_kurtosis(&dropna(data)) }

/// Check if approximately normal (Jarque-Bera p-value > 0.05).
pub fn is_approximately_normal(data: &[f64]) -> bool {
    let valid = dropna(data);
    if valid.len() < 8 { return false; }
    let result = crate::hypothesis::jarque_bera(&valid);
    result.p_value > 0.05
}

// ═══════════════════════════════════════════════════════════════════
// ANOMALY FLAGS
// ═══════════════════════════════════════════════════════════════════

/// Flag values beyond n standard deviations from mean.
pub fn sigma_flags(data: &[f64], n_sigma: f64) -> Vec<bool> {
    let valid = dropna(data);
    let m = crate::array_ops::mean(&valid);
    let s = crate::array_ops::std_dev(&valid);
    data.iter().map(|&x| x.is_finite() && (x - m).abs() > n_sigma * s).collect()
}

/// Flag sudden jumps (absolute change > threshold).
pub fn jump_flags(data: &[f64], threshold: f64) -> Vec<bool> {
    let mut flags = vec![false; data.len()];
    for i in 1..data.len() {
        if data[i].is_finite() && data[i-1].is_finite() && (data[i] - data[i-1]).abs() > threshold { flags[i] = true; }
    }
    flags
}

/// Flag flat/stuck values (same value repeated n times).
pub fn stuck_flags(data: &[f64], min_repeat: usize) -> Vec<bool> {
    let mut flags = vec![false; data.len()];
    let mut count = 1;
    for i in 1..data.len() {
        if (data[i] - data[i-1]).abs() < 1e-15 { count += 1; }
        else { if count >= min_repeat { for j in i-count..i { flags[j] = true; } } count = 1; }
    }
    if count >= min_repeat { for j in data.len()-count..data.len() { flags[j] = true; } }
    flags
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_nan() {
        assert_eq!(count_nan(&[1.0, f64::NAN, 3.0, f64::NAN]), 2);
        assert_eq!(count_valid(&[1.0, f64::NAN, 3.0]), 2);
    }

    #[test]
    fn test_dropna() {
        assert_eq!(dropna(&[1.0, f64::NAN, 3.0]), vec![1.0, 3.0]);
    }

    #[test]
    fn test_fillna() {
        let filled = fillna(&[1.0, f64::NAN, 3.0], 0.0);
        assert_eq!(filled, vec![1.0, 0.0, 3.0]);
    }

    #[test]
    fn test_interpolate_nan() {
        let data = vec![1.0, f64::NAN, 3.0];
        let interp = interpolate_nan(&data);
        assert!((interp[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_in_range() {
        assert!(in_range(&[1.0, 2.0, 3.0], 0.0, 5.0));
        assert!(!in_range(&[1.0, 2.0, 10.0], 0.0, 5.0));
    }

    #[test]
    fn test_five_number() {
        let (min, q1, med, q3, max) = five_number_summary(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
        assert!((med - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_describe() {
        let desc = describe(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(desc[0].1, 5.0); // count
    }

    #[test]
    fn test_sigma_flags() {
        let data = vec![1.0, 2.0, 1.5, 100.0, 1.8, 2.1, 1.9, 2.0, 1.7, 2.2];
        let flags = sigma_flags(&data, 2.0);
        assert!(flags[3]); // 100 is outlier
    }

    #[test]
    fn test_stuck_flags() {
        let data = vec![1.0, 1.0, 1.0, 2.0, 3.0];
        let flags = stuck_flags(&data, 3);
        assert!(flags[0] && flags[1] && flags[2]);
        assert!(!flags[3]);
    }

    #[test]
    fn test_completeness() {
        assert!((completeness(&[1.0, f64::NAN, 3.0]) - 2.0/3.0).abs() < 1e-10);
    }
}
