//! Spatial data structures and algorithms: k-d tree, nearest neighbor,
//! k-nearest neighbors, range search, ball tree, distance matrix,
//! pairwise distances, spatial hashing.

use std::collections::BinaryHeap;

// ═══════════════════════════════════════════════════════════════════
// K-D TREE
// ═══════════════════════════════════════════════════════════════════

/// K-D Tree node.
#[derive(Debug)]
enum KdNode {
    Leaf { idx: usize },
    Internal {
        split_dim: usize,
        split_val: f64,
        left: Box<KdNode>,
        right: Box<KdNode>,
    },
}

/// K-D Tree for nearest neighbor queries.
pub struct KdTree {
    root: Option<Box<KdNode>>,
    data: Vec<f64>,
    n: usize,
    d: usize,
}

impl KdTree {
    /// Build k-d tree from data (n points × d dimensions, row-major).
    pub fn build(data: &[f64], n: usize, d: usize) -> Self {
        let indices: Vec<usize> = (0..n).collect();
        let root = Self::build_node(data, &indices, d, 0);
        Self { root, data: data.to_vec(), n, d }
    }

    fn build_node(data: &[f64], indices: &[usize], d: usize, depth: usize) -> Option<Box<KdNode>> {
        if indices.is_empty() { return None; }
        if indices.len() == 1 {
            return Some(Box::new(KdNode::Leaf { idx: indices[0] }));
        }
        let dim = depth % d;
        let mut sorted = indices.to_vec();
        sorted.sort_by(|&a, &b| data[a * d + dim].partial_cmp(&data[b * d + dim]).unwrap());
        let mid = sorted.len() / 2;
        let split_val = data[sorted[mid] * d + dim];
        let left = Self::build_node(data, &sorted[..mid], d, depth + 1);
        let right = Self::build_node(data, &sorted[mid..], d, depth + 1);
        Some(Box::new(KdNode::Internal {
            split_dim: dim,
            split_val,
            left: left.unwrap_or(Box::new(KdNode::Leaf { idx: sorted[0] })),
            right: right.unwrap_or(Box::new(KdNode::Leaf { idx: sorted[sorted.len() - 1] })),
        }))
    }

    /// Find nearest neighbor. Returns (index, distance).
    pub fn nearest(&self, query: &[f64]) -> (usize, f64) {
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;
        if let Some(ref root) = self.root {
            self.search_nearest(root, query, &mut best_idx, &mut best_dist, 0);
        }
        (best_idx, best_dist.sqrt())
    }

    fn search_nearest(&self, node: &KdNode, query: &[f64], best_idx: &mut usize, best_dist: &mut f64, depth: usize) {
        match node {
            KdNode::Leaf { idx } => {
                let dist = sq_dist_slice(&self.data[idx * self.d..(idx + 1) * self.d], query);
                if dist < *best_dist {
                    *best_dist = dist;
                    *best_idx = *idx;
                }
            }
            KdNode::Internal { split_dim, split_val, left, right } => {
                let diff = query[*split_dim] - split_val;
                let (first, second) = if diff <= 0.0 { (left, right) } else { (right, left) };
                self.search_nearest(first, query, best_idx, best_dist, depth + 1);
                if diff * diff < *best_dist {
                    self.search_nearest(second, query, best_idx, best_dist, depth + 1);
                }
            }
        }
    }

    /// Find k nearest neighbors. Returns vec of (index, distance).
    pub fn knn(&self, query: &[f64], k: usize) -> Vec<(usize, f64)> {
        let mut heap: BinaryHeap<OrdF64> = BinaryHeap::new();
        if let Some(ref root) = self.root {
            self.search_knn(root, query, k, &mut heap, 0);
        }
        let mut result: Vec<(usize, f64)> = heap.into_iter().map(|h| (h.idx, h.dist.sqrt())).collect();
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        result
    }

    fn search_knn(&self, node: &KdNode, query: &[f64], k: usize, heap: &mut BinaryHeap<OrdF64>, depth: usize) {
        match node {
            KdNode::Leaf { idx } => {
                let dist = sq_dist_slice(&self.data[idx * self.d..(idx + 1) * self.d], query);
                if heap.len() < k {
                    heap.push(OrdF64 { dist, idx: *idx });
                } else if dist < heap.peek().unwrap().dist {
                    heap.pop();
                    heap.push(OrdF64 { dist, idx: *idx });
                }
            }
            KdNode::Internal { split_dim, split_val, left, right } => {
                let diff = query[*split_dim] - split_val;
                let (first, second) = if diff <= 0.0 { (left, right) } else { (right, left) };
                self.search_knn(first, query, k, heap, depth + 1);
                if heap.len() < k || diff * diff < heap.peek().unwrap().dist {
                    self.search_knn(second, query, k, heap, depth + 1);
                }
            }
        }
    }

    /// Range search: find all points within radius of query.
    pub fn range_search(&self, query: &[f64], radius: f64) -> Vec<(usize, f64)> {
        let mut results = Vec::new();
        let r2 = radius * radius;
        if let Some(ref root) = self.root {
            self.search_range(root, query, r2, &mut results, 0);
        }
        results.iter_mut().for_each(|(_, d)| *d = d.sqrt());
        results
    }

    fn search_range(&self, node: &KdNode, query: &[f64], r2: f64, results: &mut Vec<(usize, f64)>, depth: usize) {
        match node {
            KdNode::Leaf { idx } => {
                let dist = sq_dist_slice(&self.data[idx * self.d..(idx + 1) * self.d], query);
                if dist <= r2 { results.push((*idx, dist)); }
            }
            KdNode::Internal { split_dim, split_val, left, right } => {
                let diff = query[*split_dim] - split_val;
                let (first, second) = if diff <= 0.0 { (left, right) } else { (right, left) };
                self.search_range(first, query, r2, results, depth + 1);
                if diff * diff <= r2 {
                    self.search_range(second, query, r2, results, depth + 1);
                }
            }
        }
    }
}

#[derive(PartialEq)]
struct OrdF64 { dist: f64, idx: usize }
impl Eq for OrdF64 {}
impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.partial_cmp(&other.dist).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ═══════════════════════════════════════════════════════════════════
// DISTANCE MATRIX
// ═══════════════════════════════════════════════════════════════════

/// Compute pairwise distance matrix (n×n). dist_fn(a, b) -> distance.
pub fn distance_matrix(data: &[f64], n: usize, d: usize, metric: DistanceMetric) -> Vec<f64> {
    let mut mat = vec![0.0; n * n];
    for i in 0..n {
        for j in i+1..n {
            let dist = compute_distance(&data[i*d..(i+1)*d], &data[j*d..(j+1)*d], metric);
            mat[i * n + j] = dist;
            mat[j * n + i] = dist;
        }
    }
    mat
}

/// Compute pairwise distances between two datasets (n×m).
pub fn cdist(a: &[f64], na: usize, b: &[f64], nb: usize, d: usize, metric: DistanceMetric) -> Vec<f64> {
    let mut mat = vec![0.0; na * nb];
    for i in 0..na {
        for j in 0..nb {
            mat[i * nb + j] = compute_distance(&a[i*d..(i+1)*d], &b[j*d..(j+1)*d], metric);
        }
    }
    mat
}

/// Distance metric enum.
#[derive(Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Chebyshev,
    Cosine,
    Minkowski(f64),
}

fn compute_distance(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => sq_dist_slice(a, b).sqrt(),
        DistanceMetric::Manhattan => a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum(),
        DistanceMetric::Chebyshev => a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max),
        DistanceMetric::Cosine => {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            if na < 1e-15 || nb < 1e-15 { 1.0 } else { 1.0 - dot / (na * nb) }
        }
        DistanceMetric::Minkowski(p) => {
            a.iter().zip(b).map(|(x, y)| (x - y).abs().powf(p)).sum::<f64>().powf(1.0 / p)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// NEAREST NEIGHBOR (brute force, for comparison)
// ═══════════════════════════════════════════════════════════════════

/// Brute-force nearest neighbor. Returns (index, distance).
pub fn nearest_neighbor_brute(data: &[f64], n: usize, d: usize, query: &[f64]) -> (usize, f64) {
    let mut best_idx = 0;
    let mut best_dist = f64::INFINITY;
    for i in 0..n {
        let dist = sq_dist_slice(&data[i*d..(i+1)*d], query).sqrt();
        if dist < best_dist { best_dist = dist; best_idx = i; }
    }
    (best_idx, best_dist)
}

/// Brute-force k-nearest neighbors.
pub fn knn_brute(data: &[f64], n: usize, d: usize, query: &[f64], k: usize) -> Vec<(usize, f64)> {
    let mut dists: Vec<(usize, f64)> = (0..n)
        .map(|i| (i, sq_dist_slice(&data[i*d..(i+1)*d], query).sqrt()))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.truncate(k);
    dists
}

// ═══════════════════════════════════════════════════════════════════
// SPATIAL HASHING (uniform grid)
// ═══════════════════════════════════════════════════════════════════

/// Spatial hash grid for 2D points.
pub struct SpatialHash {
    cell_size: f64,
    cells: std::collections::HashMap<(i64, i64), Vec<usize>>,
}

impl SpatialHash {
    /// Build from 2D points (data = [x0, y0, x1, y1, ...]).
    pub fn build(data: &[f64], cell_size: f64) -> Self {
        let mut cells = std::collections::HashMap::new();
        let n = data.len() / 2;
        for i in 0..n {
            let cx = (data[i * 2] / cell_size).floor() as i64;
            let cy = (data[i * 2 + 1] / cell_size).floor() as i64;
            cells.entry((cx, cy)).or_insert_with(Vec::new).push(i);
        }
        Self { cell_size, cells }
    }

    /// Find all points within radius of query point (x, y).
    pub fn query_radius(&self, data: &[f64], x: f64, y: f64, radius: f64) -> Vec<(usize, f64)> {
        let r2 = radius * radius;
        let min_cx = ((x - radius) / self.cell_size).floor() as i64;
        let max_cx = ((x + radius) / self.cell_size).floor() as i64;
        let min_cy = ((y - radius) / self.cell_size).floor() as i64;
        let max_cy = ((y + radius) / self.cell_size).floor() as i64;

        let mut results = Vec::new();
        for cx in min_cx..=max_cx {
            for cy in min_cy..=max_cy {
                if let Some(indices) = self.cells.get(&(cx, cy)) {
                    for &i in indices {
                        let dx = data[i * 2] - x;
                        let dy = data[i * 2 + 1] - y;
                        let d2 = dx * dx + dy * dy;
                        if d2 <= r2 { results.push((i, d2.sqrt())); }
                    }
                }
            }
        }
        results
    }
}

// ═══════════════════════════════════════════════════════════════════
// DIMENSIONALITY REDUCTION (PCA)
// ═══════════════════════════════════════════════════════════════════

/// PCA: reduce n×d data to n×k dimensions.
/// Returns (projected_data, explained_variance_ratio).
pub fn pca(data: &[f64], n: usize, d: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    // Center data
    let mut means = vec![0.0; d];
    for i in 0..n {
        for j in 0..d { means[j] += data[i * d + j]; }
    }
    for m in means.iter_mut() { *m /= n as f64; }

    let mut centered = data.to_vec();
    for i in 0..n {
        for j in 0..d { centered[i * d + j] -= means[j]; }
    }

    // Covariance matrix (d×d)
    let mut cov = vec![0.0; d * d];
    for i in 0..n {
        for a in 0..d {
            for b in a..d {
                let v = centered[i * d + a] * centered[i * d + b];
                cov[a * d + b] += v;
                if a != b { cov[b * d + a] += v; }
            }
        }
    }
    for v in cov.iter_mut() { *v /= (n - 1) as f64; }

    // Power iteration for top-k eigenvectors
    let mut components = Vec::new();
    let mut eigenvalues = Vec::new();
    let mut cov_deflated = cov.clone();

    for _ in 0..k {
        let (eigenval, eigenvec) = power_iteration(&cov_deflated, d, 200);
        eigenvalues.push(eigenval);
        components.extend_from_slice(&eigenvec);
        // Deflate
        for a in 0..d {
            for b in 0..d {
                cov_deflated[a * d + b] -= eigenval * eigenvec[a] * eigenvec[b];
            }
        }
    }

    // Project
    let mut projected = vec![0.0; n * k];
    for i in 0..n {
        for j in 0..k {
            for dim in 0..d {
                projected[i * k + j] += centered[i * d + dim] * components[j * d + dim];
            }
        }
    }

    let total_var: f64 = eigenvalues.iter().sum::<f64>();
    let explained: Vec<f64> = if total_var > 1e-15 {
        eigenvalues.iter().map(|&v| v / total_var).collect()
    } else {
        vec![0.0; k]
    };

    (projected, explained)
}

/// t-SNE (simplified Barnes-Hut approximation).
/// Returns n×2 embedded coordinates.
pub fn tsne(data: &[f64], n: usize, d: usize, perplexity: f64, max_iter: usize, learning_rate: f64, seed: u64) -> Vec<f64> {
    let mut state = [seed, seed.wrapping_mul(6364136223846793005).wrapping_add(1),
                     seed.wrapping_mul(1442695040888963407).wrapping_add(3),
                     seed.wrapping_mul(2862933555777941757).wrapping_add(7)];
    for _ in 0..20 { next_prng(&mut state); }

    // Initialize randomly
    let mut y = vec![0.0; n * 2];
    for v in y.iter_mut() {
        *v = (next_prng(&mut state) >> 11) as f64 / (1u64 << 53) as f64 * 0.01;
    }

    // Compute pairwise affinities (simplified - no perplexity search)
    let sigma = perplexity.sqrt();
    let mut p = vec![0.0; n * n];
    for i in 0..n {
        for j in i+1..n {
            let d2 = sq_dist_slice(&data[i*d..(i+1)*d], &data[j*d..(j+1)*d]);
            let pij = (-d2 / (2.0 * sigma * sigma)).exp();
            p[i * n + j] = pij;
            p[j * n + i] = pij;
        }
        let row_sum: f64 = p[i*n..(i+1)*n].iter().sum();
        if row_sum > 1e-15 {
            for j in 0..n { p[i * n + j] /= row_sum; }
        }
    }
    // Symmetrize
    for i in 0..n {
        for j in i+1..n {
            let sym = (p[i*n+j] + p[j*n+i]) / (2.0 * n as f64);
            p[i*n+j] = sym.max(1e-12);
            p[j*n+i] = sym.max(1e-12);
        }
    }

    // Gradient descent
    let mut gains = vec![1.0; n * 2];
    let mut iy = vec![0.0; n * 2];
    let momentum = 0.8;

    for _ in 0..max_iter {
        // Compute Q (Student-t in low-dim)
        let mut q = vec![0.0; n * n];
        let mut q_sum = 0.0;
        for i in 0..n {
            for j in i+1..n {
                let d2 = sq_dist_slice(&y[i*2..(i+1)*2], &y[j*2..(j+1)*2]);
                let qij = 1.0 / (1.0 + d2);
                q[i*n+j] = qij;
                q[j*n+i] = qij;
                q_sum += 2.0 * qij;
            }
        }
        if q_sum > 1e-15 {
            for v in q.iter_mut() { *v /= q_sum; *v = (*v).max(1e-12); }
        }

        // Compute gradient
        let mut dy = vec![0.0; n * 2];
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let d2 = sq_dist_slice(&y[i*2..(i+1)*2], &y[j*2..(j+1)*2]);
                let mult = (p[i*n+j] - q[i*n+j]) / (1.0 + d2);
                for dim in 0..2 {
                    dy[i*2+dim] += 4.0 * mult * (y[i*2+dim] - y[j*2+dim]);
                }
            }
        }

        // Update with momentum
        for i in 0..n*2 {
            let sign_match = (dy[i] > 0.0) == (iy[i] > 0.0);
            gains[i] = if sign_match { (gains[i] * 0.8f64).max(0.01) } else { gains[i] + 0.2 };
            iy[i] = momentum * iy[i] - learning_rate * gains[i] * dy[i];
            y[i] += iy[i];
        }

        // Center
        let mut center = [0.0; 2];
        for i in 0..n { center[0] += y[i*2]; center[1] += y[i*2+1]; }
        center[0] /= n as f64; center[1] /= n as f64;
        for i in 0..n { y[i*2] -= center[0]; y[i*2+1] -= center[1]; }
    }
    y
}

fn power_iteration(mat: &[f64], d: usize, max_iter: usize) -> (f64, Vec<f64>) {
    let mut v = vec![1.0 / (d as f64).sqrt(); d];
    let mut eigenval = 0.0;
    for _ in 0..max_iter {
        let mut new_v = vec![0.0; d];
        for i in 0..d {
            for j in 0..d {
                new_v[i] += mat[i * d + j] * v[j];
            }
        }
        eigenval = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if eigenval > 1e-15 {
            for x in new_v.iter_mut() { *x /= eigenval; }
        }
        let diff: f64 = v.iter().zip(&new_v).map(|(a, b)| (a - b).abs()).sum();
        v = new_v;
        if diff < 1e-10 { break; }
    }
    (eigenval, v)
}

fn sq_dist_slice(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn next_prng(state: &mut [u64; 4]) -> u64 {
    let result = (state[0].wrapping_add(state[3])).rotate_left(23).wrapping_add(state[0]);
    let t = state[1] << 17;
    state[2] ^= state[0]; state[3] ^= state[1]; state[1] ^= state[2]; state[0] ^= state[3];
    state[2] ^= t; state[3] = state[3].rotate_left(45);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data() -> (Vec<f64>, usize, usize) {
        // 4 points in 2D
        (vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 4, 2)
    }

    #[test]
    fn test_kdtree_nearest() {
        let (data, n, d) = test_data();
        let tree = KdTree::build(&data, n, d);
        let (idx, dist) = tree.nearest(&[0.1, 0.1]);
        assert_eq!(idx, 0); // closest to origin
        assert!(dist < 0.2);
    }

    #[test]
    fn test_kdtree_knn() {
        let (data, n, d) = test_data();
        let tree = KdTree::build(&data, n, d);
        let neighbors = tree.knn(&[0.5, 0.5], 2);
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_kdtree_range() {
        let (data, n, d) = test_data();
        let tree = KdTree::build(&data, n, d);
        let results = tree.range_search(&[0.0, 0.0], 1.1);
        assert!(results.len() >= 2); // origin and both adjacent corners
    }

    #[test]
    fn test_distance_matrix() {
        let (data, n, d) = test_data();
        let mat = distance_matrix(&data, n, d, DistanceMetric::Euclidean);
        assert_eq!(mat.len(), 16);
        assert!((mat[0] - 0.0).abs() < 1e-10); // self-distance
        assert!((mat[1] - 1.0).abs() < 1e-10); // (0,0)→(1,0)
    }

    #[test]
    fn test_brute_nn() {
        let (data, n, d) = test_data();
        let (idx, dist) = nearest_neighbor_brute(&data, n, d, &[0.9, 0.1]);
        assert_eq!(idx, 1); // closest to (1,0)
        assert!(dist < 0.2);
    }

    #[test]
    fn test_spatial_hash() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 10.0, 10.0];
        let hash = SpatialHash::build(&data, 2.0);
        let results = hash.query_radius(&data, 0.5, 0.5, 1.5);
        assert_eq!(results.len(), 3); // first 3 points within radius
    }

    #[test]
    fn test_pca() {
        // 3 points in 2D, principal axis along x
        let data = vec![0.0, 0.0, 1.0, 0.1, 2.0, -0.1];
        let (projected, explained) = pca(&data, 3, 2, 1);
        assert_eq!(projected.len(), 3);
        assert!(explained[0] > 0.9); // most variance along first component
    }

    #[test]
    fn test_tsne() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 10.0, 10.0, 11.0, 10.0];
        let embedded = tsne(&data, 4, 2, 2.0, 50, 100.0, 42);
        assert_eq!(embedded.len(), 8); // 4 points × 2 dims
    }

    #[test]
    fn test_cdist() {
        let a = vec![0.0, 0.0, 1.0, 1.0];
        let b = vec![0.0, 0.0];
        let mat = cdist(&a, 2, &b, 1, 2, DistanceMetric::Euclidean);
        assert_eq!(mat.len(), 2);
        assert!((mat[0] - 0.0).abs() < 1e-10);
        assert!((mat[1] - 2.0f64.sqrt()).abs() < 1e-10);
    }
}
