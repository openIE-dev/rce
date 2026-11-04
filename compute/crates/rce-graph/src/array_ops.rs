//! Array operations: element-wise math, reductions, transformations,
//! broadcasting helpers, accumulation, windowing, reshaping.

// ═══════════════════════════════════════════════════════════════════
// ELEMENT-WISE MATH
// ═══════════════════════════════════════════════════════════════════

pub fn add(a: &[f64], b: &[f64]) -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x + y).collect() }
pub fn sub(a: &[f64], b: &[f64]) -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x - y).collect() }
pub fn mul(a: &[f64], b: &[f64]) -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x * y).collect() }
pub fn div(a: &[f64], b: &[f64]) -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x / y).collect() }
pub fn add_scalar(a: &[f64], s: f64) -> Vec<f64> { a.iter().map(|x| x + s).collect() }
pub fn mul_scalar(a: &[f64], s: f64) -> Vec<f64> { a.iter().map(|x| x * s).collect() }
pub fn pow_scalar(a: &[f64], p: f64) -> Vec<f64> { a.iter().map(|x| x.powf(p)).collect() }
pub fn negate(a: &[f64]) -> Vec<f64> { a.iter().map(|x| -x).collect() }
pub fn reciprocal(a: &[f64]) -> Vec<f64> { a.iter().map(|x| 1.0 / x).collect() }
pub fn abs_vec(a: &[f64]) -> Vec<f64> { a.iter().map(|x| x.abs()).collect() }
pub fn sqrt_vec(a: &[f64]) -> Vec<f64> { a.iter().map(|x| x.sqrt()).collect() }
pub fn exp_vec(a: &[f64]) -> Vec<f64> { a.iter().map(|x| x.exp()).collect() }
pub fn ln_vec(a: &[f64]) -> Vec<f64> { a.iter().map(|x| x.ln()).collect() }
pub fn log2_vec(a: &[f64]) -> Vec<f64> { a.iter().map(|x| x.log2()).collect() }
pub fn log10_vec(a: &[f64]) -> Vec<f64> { a.iter().map(|x| x.log10()).collect() }
pub fn sin_vec(a: &[f64]) -> Vec<f64> { a.iter().map(|x| x.sin()).collect() }
pub fn cos_vec(a: &[f64]) -> Vec<f64> { a.iter().map(|x| x.cos()).collect() }
pub fn tan_vec(a: &[f64]) -> Vec<f64> { a.iter().map(|x| x.tan()).collect() }
pub fn floor_vec(a: &[f64]) -> Vec<f64> { a.iter().map(|x| x.floor()).collect() }
pub fn ceil_vec(a: &[f64]) -> Vec<f64> { a.iter().map(|x| x.ceil()).collect() }
pub fn round_vec(a: &[f64]) -> Vec<f64> { a.iter().map(|x| x.round()).collect() }
pub fn sign_vec(a: &[f64]) -> Vec<f64> { a.iter().map(|x| x.signum()).collect() }
pub fn clamp_vec(a: &[f64], lo: f64, hi: f64) -> Vec<f64> { a.iter().map(|x| x.max(lo).min(hi)).collect() }
pub fn maximum(a: &[f64], b: &[f64]) -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x.max(*y)).collect() }
pub fn minimum(a: &[f64], b: &[f64]) -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x.min(*y)).collect() }
pub fn fma(a: &[f64], b: &[f64], c: &[f64]) -> Vec<f64> { a.iter().zip(b).zip(c).map(|((x,y),z)| x * y + z).collect() }

// ═══════════════════════════════════════════════════════════════════
// REDUCTIONS
// ═══════════════════════════════════════════════════════════════════

pub fn sum(a: &[f64]) -> f64 { a.iter().sum() }
pub fn product(a: &[f64]) -> f64 { a.iter().product() }
pub fn mean(a: &[f64]) -> f64 { sum(a) / a.len() as f64 }
pub fn min_val(a: &[f64]) -> f64 { a.iter().copied().fold(f64::INFINITY, f64::min) }
pub fn max_val(a: &[f64]) -> f64 { a.iter().copied().fold(f64::NEG_INFINITY, f64::max) }
pub fn argmin(a: &[f64]) -> usize { a.iter().enumerate().min_by(|x, y| x.1.partial_cmp(y.1).unwrap()).map(|(i,_)| i).unwrap_or(0) }
pub fn argmax(a: &[f64]) -> usize { a.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).map(|(i,_)| i).unwrap_or(0) }
pub fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x,y)| x * y).sum() }
pub fn norm_l1(a: &[f64]) -> f64 { a.iter().map(|x| x.abs()).sum() }
pub fn norm_l2(a: &[f64]) -> f64 { a.iter().map(|x| x * x).sum::<f64>().sqrt() }
pub fn norm_inf(a: &[f64]) -> f64 { a.iter().map(|x| x.abs()).fold(0.0, f64::max) }
pub fn norm_lp(a: &[f64], p: f64) -> f64 { a.iter().map(|x| x.abs().powf(p)).sum::<f64>().powf(1.0/p) }
pub fn variance(a: &[f64]) -> f64 { let m = mean(a); a.iter().map(|x| (x-m).powi(2)).sum::<f64>() / (a.len()-1) as f64 }
pub fn std_dev(a: &[f64]) -> f64 { variance(a).sqrt() }
pub fn any_positive(a: &[f64]) -> bool { a.iter().any(|x| *x > 0.0) }
pub fn all_positive(a: &[f64]) -> bool { a.iter().all(|x| *x > 0.0) }
pub fn count_nonzero(a: &[f64]) -> usize { a.iter().filter(|x| x.abs() > 1e-15).count() }
pub fn count_positive(a: &[f64]) -> usize { a.iter().filter(|x| **x > 0.0).count() }
pub fn count_negative(a: &[f64]) -> usize { a.iter().filter(|x| **x < 0.0).count() }

// ═══════════════════════════════════════════════════════════════════
// ACCUMULATION
// ═══════════════════════════════════════════════════════════════════

pub fn cumsum(a: &[f64]) -> Vec<f64> { let mut s = 0.0; a.iter().map(|&x| { s += x; s }).collect() }
pub fn cumprod(a: &[f64]) -> Vec<f64> { let mut p = 1.0; a.iter().map(|&x| { p *= x; p }).collect() }
pub fn cummax(a: &[f64]) -> Vec<f64> { let mut m = f64::NEG_INFINITY; a.iter().map(|&x| { m = m.max(x); m }).collect() }
pub fn cummin(a: &[f64]) -> Vec<f64> { let mut m = f64::INFINITY; a.iter().map(|&x| { m = m.min(x); m }).collect() }
pub fn diff(a: &[f64]) -> Vec<f64> { a.windows(2).map(|w| w[1] - w[0]).collect() }
pub fn pct_change(a: &[f64]) -> Vec<f64> { a.windows(2).map(|w| if w[0].abs() > 1e-15 { (w[1] - w[0]) / w[0] } else { 0.0 }).collect() }

// ═══════════════════════════════════════════════════════════════════
// TRANSFORMATIONS
// ═══════════════════════════════════════════════════════════════════

pub fn reverse(a: &[f64]) -> Vec<f64> { a.iter().rev().copied().collect() }
pub fn rotate(a: &[f64], k: usize) -> Vec<f64> { let k = k % a.len(); [&a[k..], &a[..k]].concat() }
pub fn flatten(nested: &[Vec<f64>]) -> Vec<f64> { nested.iter().flat_map(|v| v.iter()).copied().collect() }
pub fn repeat_each(a: &[f64], n: usize) -> Vec<f64> { a.iter().flat_map(|&v| std::iter::repeat(v).take(n)).collect() }
pub fn interleave(a: &[f64], b: &[f64]) -> Vec<f64> { a.iter().zip(b).flat_map(|(&x, &y)| vec![x, y]).collect() }
pub fn take(a: &[f64], indices: &[usize]) -> Vec<f64> { indices.iter().map(|&i| a[i]).collect() }
pub fn put(a: &mut [f64], indices: &[usize], values: &[f64]) { for (&i, &v) in indices.iter().zip(values) { a[i] = v; } }
pub fn where_mask(a: &[f64], mask: &[bool]) -> Vec<f64> { a.iter().zip(mask).filter(|(_, m)| **m).map(|(v, _)| *v).collect() }
pub fn replace_where(a: &[f64], mask: &[bool], value: f64) -> Vec<f64> { a.iter().zip(mask).map(|(&v, &m)| if m { value } else { v }).collect() }
pub fn clip(a: &[f64], lo: f64, hi: f64) -> Vec<f64> { clamp_vec(a, lo, hi) }
pub fn unique(a: &[f64]) -> Vec<f64> { crate::sets::to_set(a) }
pub fn union(a: &[f64], b: &[f64]) -> Vec<f64> { crate::sets::set_union(a, b) }
pub fn intersect(a: &[f64], b: &[f64]) -> Vec<f64> { crate::sets::set_intersection(a, b) }
pub fn setdiff(a: &[f64], b: &[f64]) -> Vec<f64> { crate::sets::set_difference(a, b) }

// ═══════════════════════════════════════════════════════════════════
// WINDOWING
// ═══════════════════════════════════════════════════════════════════

pub fn rolling_sum(a: &[f64], w: usize) -> Vec<f64> { if a.len() < w { return vec![]; } let mut s: f64 = a[..w].iter().sum(); let mut r = vec![s]; for i in w..a.len() { s += a[i] - a[i-w]; r.push(s); } r }
pub fn rolling_mean(a: &[f64], w: usize) -> Vec<f64> { rolling_sum(a, w).iter().map(|s| s / w as f64).collect() }
pub fn rolling_max(a: &[f64], w: usize) -> Vec<f64> { crate::queue_stack::sliding_window_max(a, w) }
pub fn rolling_min(a: &[f64], w: usize) -> Vec<f64> { crate::queue_stack::sliding_window_min(a, w) }

// ═══════════════════════════════════════════════════════════════════
// CREATION
// ═══════════════════════════════════════════════════════════════════

pub fn zeros_vec(n: usize) -> Vec<f64> { vec![0.0; n] }
pub fn ones_vec(n: usize) -> Vec<f64> { vec![1.0; n] }
pub fn full_vec(n: usize, val: f64) -> Vec<f64> { vec![val; n] }
pub fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> { let mut v = Vec::new(); let mut x = start; while x < stop { v.push(x); x += step; } v }
pub fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> { crate::matrix_creation::linspace(start, stop, n) }
pub fn logspace(start: f64, stop: f64, n: usize) -> Vec<f64> { crate::matrix_creation::logspace(start, stop, n) }

// ═══════════════════════════════════════════════════════════════════
// COMPARISON
// ═══════════════════════════════════════════════════════════════════

pub fn equal(a: &[f64], b: &[f64]) -> Vec<bool> { a.iter().zip(b).map(|(x, y)| (x - y).abs() < 1e-15).collect() }
pub fn greater(a: &[f64], b: &[f64]) -> Vec<bool> { a.iter().zip(b).map(|(x, y)| x > y).collect() }
pub fn less(a: &[f64], b: &[f64]) -> Vec<bool> { a.iter().zip(b).map(|(x, y)| x < y).collect() }
pub fn greater_equal(a: &[f64], b: &[f64]) -> Vec<bool> { a.iter().zip(b).map(|(x, y)| x >= y).collect() }
pub fn less_equal(a: &[f64], b: &[f64]) -> Vec<bool> { a.iter().zip(b).map(|(x, y)| x <= y).collect() }
pub fn is_nan(a: &[f64]) -> Vec<bool> { a.iter().map(|x| x.is_nan()).collect() }
pub fn is_finite(a: &[f64]) -> Vec<bool> { a.iter().map(|x| x.is_finite()).collect() }
pub fn is_inf(a: &[f64]) -> Vec<bool> { a.iter().map(|x| x.is_infinite()).collect() }

// ═══════════════════════════════════════════════════════════════════
// LOGICAL OPS ON BOOL ARRAYS
// ═══════════════════════════════════════════════════════════════════

pub fn logical_and(a: &[bool], b: &[bool]) -> Vec<bool> { a.iter().zip(b).map(|(&x, &y)| x && y).collect() }
pub fn logical_or(a: &[bool], b: &[bool]) -> Vec<bool> { a.iter().zip(b).map(|(&x, &y)| x || y).collect() }
pub fn logical_not(a: &[bool]) -> Vec<bool> { a.iter().map(|&x| !x).collect() }
pub fn logical_xor(a: &[bool], b: &[bool]) -> Vec<bool> { a.iter().zip(b).map(|(&x, &y)| x ^ y).collect() }
pub fn count_true(a: &[bool]) -> usize { a.iter().filter(|&&x| x).count() }
pub fn count_false(a: &[bool]) -> usize { a.iter().filter(|&&x| !x).count() }
pub fn any_true(a: &[bool]) -> bool { a.iter().any(|&x| x) }
pub fn all_true(a: &[bool]) -> bool { a.iter().all(|&x| x) }

// ═══════════════════════════════════════════════════════════════════
// STATISTICAL HELPERS
// ═══════════════════════════════════════════════════════════════════

pub fn covariance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = mean(a);
    let mb = mean(b);
    a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum::<f64>() / (n - 1.0)
}

pub fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let c = covariance(a, b);
    let sa = std_dev(a);
    let sb = std_dev(b);
    if sa < 1e-15 || sb < 1e-15 { 0.0 } else { c / (sa * sb) }
}

pub fn zscore(a: &[f64]) -> Vec<f64> {
    let m = mean(a);
    let s = std_dev(a);
    if s < 1e-15 { return vec![0.0; a.len()]; }
    a.iter().map(|x| (x - m) / s).collect()
}

pub fn normalize_minmax(a: &[f64]) -> Vec<f64> { crate::feature_eng::minmax_scale(a) }
pub fn normalize_l2(a: &[f64]) -> Vec<f64> { crate::feature_eng::l2_normalize(a) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ops() {
        assert_eq!(add(&[1.0, 2.0], &[3.0, 4.0]), vec![4.0, 6.0]);
        assert_eq!(mul_scalar(&[1.0, 2.0], 3.0), vec![3.0, 6.0]);
    }

    #[test]
    fn test_reductions() {
        assert!((sum(&[1.0, 2.0, 3.0]) - 6.0).abs() < 1e-10);
        assert!((mean(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-10);
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn test_cumsum() {
        assert_eq!(cumsum(&[1.0, 2.0, 3.0]), vec![1.0, 3.0, 6.0]);
    }

    #[test]
    fn test_diff() {
        assert_eq!(diff(&[1.0, 3.0, 6.0]), vec![2.0, 3.0]);
    }

    #[test]
    fn test_reverse_rotate() {
        assert_eq!(reverse(&[1.0, 2.0, 3.0]), vec![3.0, 2.0, 1.0]);
        assert_eq!(rotate(&[1.0, 2.0, 3.0, 4.0], 2), vec![3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_where_mask() {
        assert_eq!(where_mask(&[1.0, 2.0, 3.0], &[true, false, true]), vec![1.0, 3.0]);
    }

    #[test]
    fn test_arange() {
        assert_eq!(arange(0.0, 5.0, 1.0), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_logical() {
        assert_eq!(logical_and(&[true, false], &[true, true]), vec![true, false]);
        assert_eq!(count_true(&[true, false, true]), 2);
    }

    #[test]
    fn test_norms() {
        assert!((norm_l2(&[3.0, 4.0]) - 5.0).abs() < 1e-10);
        assert!((norm_l1(&[1.0, -2.0, 3.0]) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((correlation(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_interleave() {
        assert_eq!(interleave(&[1.0, 2.0], &[3.0, 4.0]), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_rolling_mean() {
        let rm = rolling_mean(&[1.0, 2.0, 3.0, 4.0, 5.0], 3);
        assert!((rm[0] - 2.0).abs() < 1e-10);
    }
}
