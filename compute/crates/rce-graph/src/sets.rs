//! Set operations on sorted arrays: union, intersection, difference,
//! symmetric difference, subset, Jaccard/Dice/overlap coefficients,
//! multiset operations, power set, Cartesian product.

// ═══════════════════════════════════════════════════════════════════
// SET OPERATIONS (on sorted f64 slices)
// ═══════════════════════════════════════════════════════════════════

/// Sort and deduplicate.
pub fn to_set(data: &[f64]) -> Vec<f64> {
    let mut s = data.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    s
}

/// Union of two sorted sets.
pub fn set_union(a: &[f64], b: &[f64]) -> Vec<f64> {
    let (mut i, mut j) = (0, 0);
    let mut result = Vec::new();
    while i < a.len() && j < b.len() {
        if (a[i] - b[j]).abs() < 1e-15 {
            result.push(a[i]);
            i += 1; j += 1;
        } else if a[i] < b[j] {
            result.push(a[i]); i += 1;
        } else {
            result.push(b[j]); j += 1;
        }
    }
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

/// Intersection of two sorted sets.
pub fn set_intersection(a: &[f64], b: &[f64]) -> Vec<f64> {
    let (mut i, mut j) = (0, 0);
    let mut result = Vec::new();
    while i < a.len() && j < b.len() {
        if (a[i] - b[j]).abs() < 1e-15 {
            result.push(a[i]);
            i += 1; j += 1;
        } else if a[i] < b[j] { i += 1; }
        else { j += 1; }
    }
    result
}

/// Difference A \ B (elements in A not in B).
pub fn set_difference(a: &[f64], b: &[f64]) -> Vec<f64> {
    let (mut i, mut j) = (0, 0);
    let mut result = Vec::new();
    while i < a.len() && j < b.len() {
        if (a[i] - b[j]).abs() < 1e-15 { i += 1; j += 1; }
        else if a[i] < b[j] { result.push(a[i]); i += 1; }
        else { j += 1; }
    }
    result.extend_from_slice(&a[i..]);
    result
}

/// Symmetric difference (elements in A or B but not both).
pub fn set_symmetric_difference(a: &[f64], b: &[f64]) -> Vec<f64> {
    let (mut i, mut j) = (0, 0);
    let mut result = Vec::new();
    while i < a.len() && j < b.len() {
        if (a[i] - b[j]).abs() < 1e-15 { i += 1; j += 1; }
        else if a[i] < b[j] { result.push(a[i]); i += 1; }
        else { result.push(b[j]); j += 1; }
    }
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

/// Is A a subset of B?
pub fn is_subset(a: &[f64], b: &[f64]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if (a[i] - b[j]).abs() < 1e-15 { i += 1; j += 1; }
        else if a[i] < b[j] { return false; }
        else { j += 1; }
    }
    i == a.len()
}

/// Is A a proper subset of B?
pub fn is_proper_subset(a: &[f64], b: &[f64]) -> bool {
    a.len() < b.len() && is_subset(a, b)
}

/// Are sets equal?
pub fn sets_equal(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-15)
}

/// Are sets disjoint?
pub fn is_disjoint(a: &[f64], b: &[f64]) -> bool {
    set_intersection(a, b).is_empty()
}

// ═══════════════════════════════════════════════════════════════════
// SIMILARITY COEFFICIENTS
// ═══════════════════════════════════════════════════════════════════

/// Jaccard index: |A ∩ B| / |A ∪ B|.
pub fn jaccard_index(a: &[f64], b: &[f64]) -> f64 {
    let inter = set_intersection(a, b).len() as f64;
    let union = set_union(a, b).len() as f64;
    if union < 1e-15 { return 0.0; }
    inter / union
}

/// Dice coefficient: 2|A ∩ B| / (|A| + |B|).
pub fn dice_coefficient(a: &[f64], b: &[f64]) -> f64 {
    let inter = set_intersection(a, b).len() as f64;
    let total = (a.len() + b.len()) as f64;
    if total < 1e-15 { return 0.0; }
    2.0 * inter / total
}

/// Overlap coefficient: |A ∩ B| / min(|A|, |B|).
pub fn overlap_coefficient(a: &[f64], b: &[f64]) -> f64 {
    let inter = set_intersection(a, b).len() as f64;
    let min_len = a.len().min(b.len()) as f64;
    if min_len < 1e-15 { return 0.0; }
    inter / min_len
}

// ═══════════════════════════════════════════════════════════════════
// MULTISET OPERATIONS (sorted arrays with repeats)
// ═══════════════════════════════════════════════════════════════════

/// Multiset union (max multiplicity).
pub fn multiset_union(a: &[f64], b: &[f64]) -> Vec<f64> {
    let (mut i, mut j) = (0, 0);
    let mut result = Vec::new();
    while i < a.len() && j < b.len() {
        if (a[i] - b[j]).abs() < 1e-15 {
            result.push(a[i]); i += 1; j += 1;
        } else if a[i] < b[j] {
            result.push(a[i]); i += 1;
        } else {
            result.push(b[j]); j += 1;
        }
    }
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

/// Multiset intersection (min multiplicity).
pub fn multiset_intersection(a: &[f64], b: &[f64]) -> Vec<f64> {
    let (mut i, mut j) = (0, 0);
    let mut result = Vec::new();
    while i < a.len() && j < b.len() {
        if (a[i] - b[j]).abs() < 1e-15 {
            result.push(a[i]); i += 1; j += 1;
        } else if a[i] < b[j] { i += 1; }
        else { j += 1; }
    }
    result
}

/// Multiset sum (sum of multiplicities).
pub fn multiset_sum(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] <= b[j] { result.push(a[i]); i += 1; }
        else { result.push(b[j]); j += 1; }
    }
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

// ═══════════════════════════════════════════════════════════════════
// CARTESIAN PRODUCT & CROSS JOIN
// ═══════════════════════════════════════════════════════════════════

/// Cartesian product A × B. Returns pairs.
pub fn cartesian_product(a: &[f64], b: &[f64]) -> Vec<(f64, f64)> {
    let mut result = Vec::with_capacity(a.len() * b.len());
    for &x in a {
        for &y in b {
            result.push((x, y));
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// FREQUENCY COUNTING
// ═══════════════════════════════════════════════════════════════════

/// Count occurrences of each unique value (sorted input).
pub fn value_counts(data: &[f64]) -> Vec<(f64, usize)> {
    if data.is_empty() { return vec![]; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut counts = Vec::new();
    let mut current = sorted[0];
    let mut count = 1;
    for &v in &sorted[1..] {
        if (v - current).abs() < 1e-15 {
            count += 1;
        } else {
            counts.push((current, count));
            current = v;
            count = 1;
        }
    }
    counts.push((current, count));
    counts
}

/// Mode: most frequent value(s).
pub fn mode(data: &[f64]) -> Vec<f64> {
    let counts = value_counts(data);
    let max_count = counts.iter().map(|(_, c)| *c).max().unwrap_or(0);
    counts.iter().filter(|(_, c)| *c == max_count).map(|(v, _)| *v).collect()
}

/// Unique values.
pub fn unique(data: &[f64]) -> Vec<f64> {
    to_set(data)
}

/// Number of unique values.
pub fn nunique(data: &[f64]) -> usize {
    to_set(data).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];
        assert_eq!(set_union(&a, &b), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_intersection() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];
        assert_eq!(set_intersection(&a, &b), vec![2.0, 3.0]);
    }

    #[test]
    fn test_difference() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];
        assert_eq!(set_difference(&a, &b), vec![1.0]);
    }

    #[test]
    fn test_symmetric_difference() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];
        assert_eq!(set_symmetric_difference(&a, &b), vec![1.0, 4.0]);
    }

    #[test]
    fn test_subset() {
        assert!(is_subset(&[1.0, 2.0], &[1.0, 2.0, 3.0]));
        assert!(!is_subset(&[1.0, 4.0], &[1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_jaccard() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];
        assert!((jaccard_index(&a, &b) - 0.5).abs() < 1e-10); // 2/4
    }

    #[test]
    fn test_value_counts() {
        let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        let vc = value_counts(&data);
        assert_eq!(vc, vec![(1.0, 1), (2.0, 2), (3.0, 3)]);
    }

    #[test]
    fn test_mode() {
        let data = vec![1.0, 2.0, 2.0, 3.0];
        assert_eq!(mode(&data), vec![2.0]);
    }

    #[test]
    fn test_cartesian_product() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let cp = cartesian_product(&a, &b);
        assert_eq!(cp.len(), 4);
        assert_eq!(cp[0], (1.0, 3.0));
    }
}
