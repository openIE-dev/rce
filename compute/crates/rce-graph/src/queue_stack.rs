//! Data structures: priority queue, circular buffer, deque operations,
//! stack algorithms, queue algorithms, monotonic structures, LRU cache.

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════
// PRIORITY QUEUE (min-heap)
// ═══════════════════════════════════════════════════════════════════

/// Min-heap priority queue.
pub struct MinHeap {
    data: Vec<f64>,
}

impl MinHeap {
    pub fn new() -> Self { Self { data: Vec::new() } }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }

    pub fn push(&mut self, val: f64) {
        self.data.push(val);
        self.sift_up(self.data.len() - 1);
    }

    pub fn pop(&mut self) -> Option<f64> {
        if self.data.is_empty() { return None; }
        let n = self.data.len();
        self.data.swap(0, n - 1);
        let val = self.data.pop();
        if !self.data.is_empty() { self.sift_down(0); }
        val
    }

    pub fn peek(&self) -> Option<f64> { self.data.first().copied() }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.data[idx] < self.data[parent] { self.data.swap(idx, parent); idx = parent; }
            else { break; }
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        let n = self.data.len();
        loop {
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut smallest = idx;
            if left < n && self.data[left] < self.data[smallest] { smallest = left; }
            if right < n && self.data[right] < self.data[smallest] { smallest = right; }
            if smallest != idx { self.data.swap(idx, smallest); idx = smallest; }
            else { break; }
        }
    }
}

/// Build heap from array. Returns sorted (heap sort).
pub fn heap_sort(data: &[f64]) -> Vec<f64> {
    let mut heap = MinHeap::new();
    for &v in data { heap.push(v); }
    let mut result = Vec::with_capacity(data.len());
    while let Some(v) = heap.pop() { result.push(v); }
    result
}

/// Find k smallest elements.
pub fn k_smallest(data: &[f64], k: usize) -> Vec<f64> {
    let mut heap = MinHeap::new();
    for &v in data { heap.push(v); }
    let mut result = Vec::with_capacity(k);
    for _ in 0..k.min(data.len()) {
        if let Some(v) = heap.pop() { result.push(v); }
    }
    result
}

/// Find k largest elements.
pub fn k_largest(data: &[f64], k: usize) -> Vec<f64> {
    let neg: Vec<f64> = data.iter().map(|v| -v).collect();
    k_smallest(&neg, k).iter().map(|v| -v).collect()
}

/// Merge k sorted arrays into one sorted array.
pub fn merge_k_sorted(arrays: &[Vec<f64>]) -> Vec<f64> {
    let mut heap = MinHeap::new();
    // Simple: push all, pop all
    for arr in arrays { for &v in arr { heap.push(v); } }
    let mut result = Vec::new();
    while let Some(v) = heap.pop() { result.push(v); }
    result
}

// ═══════════════════════════════════════════════════════════════════
// CIRCULAR BUFFER
// ═══════════════════════════════════════════════════════════════════

/// Fixed-size circular buffer.
pub struct CircularBuffer {
    data: Vec<f64>,
    capacity: usize,
    head: usize,
    len: usize,
}

impl CircularBuffer {
    pub fn new(capacity: usize) -> Self {
        Self { data: vec![0.0; capacity], capacity, head: 0, len: 0 }
    }

    pub fn push(&mut self, val: f64) {
        let idx = (self.head + self.len) % self.capacity;
        self.data[idx] = val;
        if self.len < self.capacity { self.len += 1; }
        else { self.head = (self.head + 1) % self.capacity; }
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_full(&self) -> bool { self.len == self.capacity }
    pub fn is_empty(&self) -> bool { self.len == 0 }

    pub fn get(&self, idx: usize) -> Option<f64> {
        if idx >= self.len { return None; }
        Some(self.data[(self.head + idx) % self.capacity])
    }

    pub fn to_vec(&self) -> Vec<f64> {
        (0..self.len).map(|i| self.data[(self.head + i) % self.capacity]).collect()
    }

    pub fn mean(&self) -> f64 {
        if self.len == 0 { return 0.0; }
        self.to_vec().iter().sum::<f64>() / self.len as f64
    }
}

// ═══════════════════════════════════════════════════════════════════
// MONOTONIC STACK/QUEUE
// ═══════════════════════════════════════════════════════════════════

/// Next greater element for each position. Returns index or usize::MAX.
pub fn next_greater_element(data: &[f64]) -> Vec<usize> {
    let n = data.len();
    let mut result = vec![usize::MAX; n];
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..n {
        while let Some(&top) = stack.last() {
            if data[i] > data[top] { result[top] = i; stack.pop(); }
            else { break; }
        }
        stack.push(i);
    }
    result
}

/// Next smaller element for each position.
pub fn next_smaller_element(data: &[f64]) -> Vec<usize> {
    let n = data.len();
    let mut result = vec![usize::MAX; n];
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..n {
        while let Some(&top) = stack.last() {
            if data[i] < data[top] { result[top] = i; stack.pop(); }
            else { break; }
        }
        stack.push(i);
    }
    result
}

/// Sliding window maximum (deque-based, O(n)).
pub fn sliding_window_max(data: &[f64], window: usize) -> Vec<f64> {
    if data.len() < window { return vec![]; }
    let mut deque: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    let mut result = Vec::with_capacity(data.len() - window + 1);
    for i in 0..data.len() {
        while deque.back().map_or(false, |&j| data[j] <= data[i]) { deque.pop_back(); }
        deque.push_back(i);
        if deque.front() == Some(&(i.wrapping_sub(window))) { deque.pop_front(); }
        if i >= window - 1 { result.push(data[*deque.front().unwrap()]); }
    }
    result
}

/// Sliding window minimum.
pub fn sliding_window_min(data: &[f64], window: usize) -> Vec<f64> {
    if data.len() < window { return vec![]; }
    let mut deque: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    let mut result = Vec::with_capacity(data.len() - window + 1);
    for i in 0..data.len() {
        while deque.back().map_or(false, |&j| data[j] >= data[i]) { deque.pop_back(); }
        deque.push_back(i);
        if deque.front() == Some(&(i.wrapping_sub(window))) { deque.pop_front(); }
        if i >= window - 1 { result.push(data[*deque.front().unwrap()]); }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// LRU CACHE
// ═══════════════════════════════════════════════════════════════════

/// Simple LRU cache (string keys, f64 values).
pub struct LruCache {
    capacity: usize,
    map: HashMap<String, (f64, usize)>, // value, access_time
    time: usize,
}

impl LruCache {
    pub fn new(capacity: usize) -> Self {
        Self { capacity, map: HashMap::new(), time: 0 }
    }

    pub fn get(&mut self, key: &str) -> Option<f64> {
        self.time += 1;
        if let Some(entry) = self.map.get_mut(key) {
            entry.1 = self.time;
            Some(entry.0)
        } else { None }
    }

    pub fn put(&mut self, key: String, value: f64) {
        self.time += 1;
        if self.map.contains_key(&key) {
            self.map.insert(key, (value, self.time));
            return;
        }
        if self.map.len() >= self.capacity {
            let lru_key = self.map.iter().min_by_key(|(_, v)| v.1).map(|(k, _)| k.clone());
            if let Some(k) = lru_key { self.map.remove(&k); }
        }
        self.map.insert(key, (value, self.time));
    }

    pub fn len(&self) -> usize { self.map.len() }
    pub fn contains(&self, key: &str) -> bool { self.map.contains_key(key) }
}

// ═══════════════════════════════════════════════════════════════════
// SORTING ALGORITHMS
// ═══════════════════════════════════════════════════════════════════

/// Merge sort (stable).
pub fn merge_sort(data: &[f64]) -> Vec<f64> {
    if data.len() <= 1 { return data.to_vec(); }
    let mid = data.len() / 2;
    let left = merge_sort(&data[..mid]);
    let right = merge_sort(&data[mid..]);
    merge(&left, &right)
}

fn merge(a: &[f64], b: &[f64]) -> Vec<f64> {
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

/// Quick select: find kth smallest element (0-indexed, average O(n)).
pub fn quick_select(data: &[f64], k: usize) -> f64 {
    let mut arr = data.to_vec();
    quick_select_impl(&mut arr, k)
}

fn quick_select_impl(arr: &mut [f64], k: usize) -> f64 {
    if arr.len() == 1 { return arr[0]; }
    let pivot_idx = arr.len() / 2;
    let pivot = arr[pivot_idx];
    let less: Vec<f64> = arr.iter().filter(|&&v| v < pivot).copied().collect();
    let equal: Vec<f64> = arr.iter().filter(|&&v| (v - pivot).abs() < 1e-15).copied().collect();
    let greater: Vec<f64> = arr.iter().filter(|&&v| v > pivot).copied().collect();
    if k < less.len() { quick_select(&less, k) }
    else if k < less.len() + equal.len() { pivot }
    else { quick_select(&greater, k - less.len() - equal.len()) }
}

/// Counting inversions in array (merge sort based).
pub fn count_inversions(data: &[f64]) -> usize {
    if data.len() <= 1 { return 0; }
    let mid = data.len() / 2;
    let left_inv = count_inversions(&data[..mid]);
    let right_inv = count_inversions(&data[mid..]);
    let left = merge_sort(&data[..mid]);
    let right = merge_sort(&data[mid..]);
    let (mut i, mut j) = (0, 0);
    let mut split_inv = 0;
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] { i += 1; }
        else { split_inv += left.len() - i; j += 1; }
    }
    left_inv + right_inv + split_inv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_heap() {
        let mut h = MinHeap::new();
        h.push(3.0); h.push(1.0); h.push(2.0);
        assert_eq!(h.pop(), Some(1.0));
        assert_eq!(h.pop(), Some(2.0));
        assert_eq!(h.pop(), Some(3.0));
    }

    #[test]
    fn test_heap_sort() {
        assert_eq!(heap_sort(&[3.0, 1.0, 4.0, 1.0, 5.0]), vec![1.0, 1.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_k_smallest() {
        assert_eq!(k_smallest(&[5.0, 3.0, 1.0, 4.0, 2.0], 3), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_circular_buffer() {
        let mut buf = CircularBuffer::new(3);
        buf.push(1.0); buf.push(2.0); buf.push(3.0); buf.push(4.0);
        assert_eq!(buf.to_vec(), vec![2.0, 3.0, 4.0]); // oldest dropped
    }

    #[test]
    fn test_next_greater() {
        let nge = next_greater_element(&[4.0, 2.0, 3.0, 5.0, 1.0]);
        assert_eq!(nge[0], 3); // 4 → 5 at index 3
        assert_eq!(nge[1], 2); // 2 → 3 at index 2
    }

    #[test]
    fn test_sliding_window_max() {
        let result = sliding_window_max(&[1.0, 3.0, -1.0, -3.0, 5.0, 3.0, 6.0, 7.0], 3);
        assert_eq!(result, vec![3.0, 3.0, 5.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_lru_cache() {
        let mut cache = LruCache::new(2);
        cache.put("a".to_string(), 1.0);
        cache.put("b".to_string(), 2.0);
        assert_eq!(cache.get("a"), Some(1.0));
        cache.put("c".to_string(), 3.0); // evicts "b" (LRU)
        assert_eq!(cache.get("b"), None);
    }

    #[test]
    fn test_merge_sort() {
        assert_eq!(merge_sort(&[3.0, 1.0, 2.0]), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_quick_select() {
        assert!((quick_select(&[5.0, 3.0, 1.0, 4.0, 2.0], 2) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_count_inversions() {
        assert_eq!(count_inversions(&[2.0, 4.0, 1.0, 3.0, 5.0]), 3);
    }

    #[test]
    fn test_merge_k_sorted() {
        let arrays = vec![vec![1.0, 4.0, 7.0], vec![2.0, 5.0, 8.0], vec![3.0, 6.0, 9.0]];
        let merged = merge_k_sorted(&arrays);
        assert_eq!(merged, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }
}
