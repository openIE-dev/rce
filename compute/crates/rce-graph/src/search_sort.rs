//! Search and sort algorithms: binary search variants, interpolation search,
//! stable/unstable sorts, partial sorts, selection algorithms.

// ═══════════════════════════════════════════════════════════════════
// BINARY SEARCH VARIANTS
// ═══════════════════════════════════════════════════════════════════

/// Binary search for exact value in sorted array. Returns index or None.
pub fn binary_search(data: &[f64], target: f64) -> Option<usize> {
    let mut lo = 0;
    let mut hi = data.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if (data[mid] - target).abs() < 1e-15 { return Some(mid); }
        if data[mid] < target { lo = mid + 1; } else { hi = mid; }
    }
    None
}

/// Lower bound: first index where data[i] >= target.
pub fn lower_bound(data: &[f64], target: f64) -> usize {
    let mut lo = 0;
    let mut hi = data.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if data[mid] < target { lo = mid + 1; } else { hi = mid; }
    }
    lo
}

/// Upper bound: first index where data[i] > target.
pub fn upper_bound(data: &[f64], target: f64) -> usize {
    let mut lo = 0;
    let mut hi = data.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if data[mid] <= target { lo = mid + 1; } else { hi = mid; }
    }
    lo
}

/// Count occurrences of value in sorted array.
pub fn count_in_sorted(data: &[f64], target: f64) -> usize {
    upper_bound(data, target) - lower_bound(data, target)
}

/// Binary search for insertion point (bisect_left equivalent).
pub fn bisect_left(data: &[f64], target: f64) -> usize { lower_bound(data, target) }

/// Binary search for insertion point (bisect_right equivalent).
pub fn bisect_right(data: &[f64], target: f64) -> usize { upper_bound(data, target) }

/// Search in sorted array for nearest value. Returns index.
pub fn search_nearest(data: &[f64], target: f64) -> usize {
    if data.is_empty() { return 0; }
    let pos = lower_bound(data, target);
    if pos == 0 { return 0; }
    if pos == data.len() { return data.len() - 1; }
    if (data[pos] - target).abs() < (data[pos - 1] - target).abs() { pos } else { pos - 1 }
}

/// Interpolation search (for uniformly distributed sorted data).
pub fn interpolation_search(data: &[f64], target: f64) -> Option<usize> {
    if data.is_empty() { return None; }
    let mut lo = 0;
    let mut hi = data.len() - 1;
    while lo <= hi && target >= data[lo] && target <= data[hi] {
        if (data[hi] - data[lo]).abs() < 1e-15 {
            if (data[lo] - target).abs() < 1e-15 { return Some(lo); }
            return None;
        }
        let pos = lo + ((target - data[lo]) * (hi - lo) as f64 / (data[hi] - data[lo])) as usize;
        if (data[pos] - target).abs() < 1e-15 { return Some(pos); }
        if data[pos] < target { lo = pos + 1; }
        else { if pos == 0 { return None; } hi = pos - 1; }
    }
    None
}

/// Exponential search (for unbounded/very large sorted arrays).
pub fn exponential_search(data: &[f64], target: f64) -> Option<usize> {
    if data.is_empty() { return None; }
    if (data[0] - target).abs() < 1e-15 { return Some(0); }
    let mut bound = 1;
    while bound < data.len() && data[bound] < target { bound *= 2; }
    let lo = bound / 2;
    let hi = bound.min(data.len());
    for i in lo..hi {
        if (data[i] - target).abs() < 1e-15 { return Some(i); }
    }
    None
}

// ═══════════════════════════════════════════════════════════════════
// SORTING ALGORITHMS
// ═══════════════════════════════════════════════════════════════════

/// Insertion sort (stable, good for small arrays).
pub fn insertion_sort(data: &mut [f64]) {
    for i in 1..data.len() {
        let key = data[i];
        let mut j = i;
        while j > 0 && data[j - 1] > key { data[j] = data[j - 1]; j -= 1; }
        data[j] = key;
    }
}

/// Shell sort.
pub fn shell_sort(data: &mut [f64]) {
    let n = data.len();
    let mut gap = n / 2;
    while gap > 0 {
        for i in gap..n {
            let temp = data[i];
            let mut j = i;
            while j >= gap && data[j - gap] > temp { data[j] = data[j - gap]; j -= gap; }
            data[j] = temp;
        }
        gap /= 2;
    }
}

/// Counting sort for non-negative integers (represented as f64).
pub fn counting_sort(data: &[f64]) -> Vec<f64> {
    if data.is_empty() { return vec![]; }
    let max = data.iter().copied().fold(0.0, f64::max) as usize;
    let mut counts = vec![0usize; max + 1];
    for &v in data { counts[v as usize] += 1; }
    let mut result = Vec::with_capacity(data.len());
    for (val, &count) in counts.iter().enumerate() {
        for _ in 0..count { result.push(val as f64); }
    }
    result
}

/// Bucket sort (for uniformly distributed data in [0, 1]).
pub fn bucket_sort(data: &[f64], n_buckets: usize) -> Vec<f64> {
    let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); n_buckets];
    for &v in data {
        let idx = (v * n_buckets as f64).floor() as usize;
        buckets[idx.min(n_buckets - 1)].push(v);
    }
    for bucket in buckets.iter_mut() { bucket.sort_by(|a, b| a.partial_cmp(b).unwrap()); }
    buckets.into_iter().flatten().collect()
}

/// Partial sort: find and sort first k smallest elements.
pub fn partial_sort(data: &[f64], k: usize) -> Vec<f64> {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.truncate(k);
    sorted
}

/// Argsort: return indices that would sort the array.
pub fn argsort(data: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..data.len()).collect();
    indices.sort_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap());
    indices
}

/// Argsort descending.
pub fn argsort_desc(data: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..data.len()).collect();
    indices.sort_by(|&a, &b| data[b].partial_cmp(&data[a]).unwrap());
    indices
}

/// Check if array is sorted (ascending).
pub fn is_sorted(data: &[f64]) -> bool {
    data.windows(2).all(|w| w[0] <= w[1])
}

/// Check if array is sorted descending.
pub fn is_sorted_desc(data: &[f64]) -> bool {
    data.windows(2).all(|w| w[0] >= w[1])
}

/// Stable partition: elements satisfying predicate first, others second.
pub fn stable_partition(data: &[f64], pred: impl Fn(f64) -> bool) -> Vec<f64> {
    let mut yes: Vec<f64> = data.iter().filter(|&&v| pred(v)).copied().collect();
    let no: Vec<f64> = data.iter().filter(|&&v| !pred(v)).copied().collect();
    yes.extend(no);
    yes
}

/// Nth element (like std::nth_element): partially sort so data[k] is correct.
pub fn nth_element(data: &mut [f64], k: usize) {
    data.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
}

/// Top-k elements (not sorted among themselves).
pub fn top_k(data: &[f64], k: usize) -> Vec<f64> {
    let mut d = data.to_vec();
    let n = d.len();
    if k >= n { d.sort_by(|a, b| b.partial_cmp(a).unwrap()); return d; }
    d.select_nth_unstable_by(n - k, |a, b| a.partial_cmp(b).unwrap());
    let mut result = d[n - k..].to_vec();
    result.sort_by(|a, b| b.partial_cmp(a).unwrap());
    result
}

/// Bottom-k elements.
pub fn bottom_k(data: &[f64], k: usize) -> Vec<f64> {
    let mut d = data.to_vec();
    if k >= d.len() { d.sort_by(|a, b| a.partial_cmp(b).unwrap()); return d; }
    d.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
    let mut result = d[..k].to_vec();
    result.sort_by(|a, b| a.partial_cmp(b).unwrap());
    result
}

/// Unique sorted: sort and remove duplicates.
pub fn unique_sorted(data: &[f64]) -> Vec<f64> {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    sorted
}

/// Set difference of two sorted arrays.
pub fn sorted_difference(a: &[f64], b: &[f64]) -> Vec<f64> {
    crate::sets::set_difference(a, b)
}

/// Merge two sorted arrays.
pub fn sorted_merge(a: &[f64], b: &[f64]) -> Vec<f64> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_search() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(binary_search(&data, 3.0), Some(2));
        assert_eq!(binary_search(&data, 3.5), None);
    }

    #[test]
    fn test_lower_upper_bound() {
        let data = vec![1.0, 2.0, 2.0, 3.0, 4.0];
        assert_eq!(lower_bound(&data, 2.0), 1);
        assert_eq!(upper_bound(&data, 2.0), 3);
        assert_eq!(count_in_sorted(&data, 2.0), 2);
    }

    #[test]
    fn test_search_nearest() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        assert_eq!(search_nearest(&data, 4.0), 1); // nearest to 3.0
        assert_eq!(search_nearest(&data, 6.0), 2); // nearest to 5.0
    }

    #[test]
    fn test_insertion_sort() {
        let mut data = vec![5.0, 3.0, 1.0, 4.0, 2.0];
        insertion_sort(&mut data);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_shell_sort() {
        let mut data = vec![5.0, 3.0, 1.0, 4.0, 2.0];
        shell_sort(&mut data);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_argsort() {
        let data = vec![30.0, 10.0, 20.0];
        assert_eq!(argsort(&data), vec![1, 2, 0]);
    }

    #[test]
    fn test_is_sorted() {
        assert!(is_sorted(&[1.0, 2.0, 3.0]));
        assert!(!is_sorted(&[1.0, 3.0, 2.0]));
    }

    #[test]
    fn test_top_k() {
        let data = vec![5.0, 1.0, 3.0, 4.0, 2.0];
        assert_eq!(top_k(&data, 3), vec![5.0, 4.0, 3.0]);
    }

    #[test]
    fn test_unique_sorted() {
        assert_eq!(unique_sorted(&[3.0, 1.0, 2.0, 1.0, 3.0]), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_stable_partition() {
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0];
        let result = stable_partition(&data, |x| x < 3.5);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_bucket_sort() {
        let data = vec![0.3, 0.1, 0.9, 0.5, 0.7];
        let sorted = bucket_sort(&data, 5);
        assert!(is_sorted(&sorted));
    }
}
