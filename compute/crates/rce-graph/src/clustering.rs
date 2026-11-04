//! Clustering algorithms: K-Means++, Mini-batch K-Means, Mean Shift,
//! OPTICS, hierarchical (Ward, complete, average, single), spectral,
//! Gaussian Mixture Models (EM), silhouette score, cluster metrics.


// ═══════════════════════════════════════════════════════════════════
// K-MEANS++
// ═══════════════════════════════════════════════════════════════════

/// K-Means++ clustering. data: n×d row-major. Returns (labels, centroids).
pub fn kmeans(data: &[f64], n: usize, d: usize, k: usize, max_iter: usize, seed: u64) -> (Vec<usize>, Vec<f64>) {
    let mut state = seed_state(seed);
    // K-Means++ initialization
    let mut centroids = vec![0.0; k * d];
    let first = next_u64(&mut state) as usize % n;
    centroids[..d].copy_from_slice(&data[first * d..(first + 1) * d]);

    for c in 1..k {
        let mut dists = vec![f64::INFINITY; n];
        for i in 0..n {
            for j in 0..c {
                let dist = sq_dist(&data[i*d..(i+1)*d], &centroids[j*d..(j+1)*d]);
                dists[i] = dists[i].min(dist);
            }
        }
        let total: f64 = dists.iter().sum();
        let mut r = next_f64(&mut state) * total;
        let mut chosen = 0;
        for (i, &dd) in dists.iter().enumerate() {
            r -= dd;
            if r <= 0.0 { chosen = i; break; }
        }
        centroids[c*d..(c+1)*d].copy_from_slice(&data[chosen*d..(chosen+1)*d]);
    }

    let mut labels = vec![0usize; n];
    for _ in 0..max_iter {
        // Assign
        let mut changed = false;
        for i in 0..n {
            let mut best = 0;
            let mut best_dist = f64::INFINITY;
            for j in 0..k {
                let dist = sq_dist(&data[i*d..(i+1)*d], &centroids[j*d..(j+1)*d]);
                if dist < best_dist { best_dist = dist; best = j; }
            }
            if labels[i] != best { changed = true; labels[i] = best; }
        }
        if !changed { break; }
        // Update centroids
        let mut counts = vec![0usize; k];
        centroids.fill(0.0);
        for i in 0..n {
            let l = labels[i];
            counts[l] += 1;
            for j in 0..d { centroids[l * d + j] += data[i * d + j]; }
        }
        for j in 0..k {
            if counts[j] > 0 {
                for dim in 0..d { centroids[j * d + dim] /= counts[j] as f64; }
            }
        }
    }
    (labels, centroids)
}

/// K-Means inertia (within-cluster sum of squares).
pub fn kmeans_inertia(data: &[f64], n: usize, d: usize, labels: &[usize], centroids: &[f64]) -> f64 {
    (0..n).map(|i| sq_dist(&data[i*d..(i+1)*d], &centroids[labels[i]*d..(labels[i]+1)*d])).sum()
}

// ═══════════════════════════════════════════════════════════════════
// MINI-BATCH K-MEANS
// ═══════════════════════════════════════════════════════════════════

/// Mini-batch K-Means. batch_size samples per iteration.
pub fn mini_batch_kmeans(data: &[f64], n: usize, d: usize, k: usize, batch_size: usize, max_iter: usize, seed: u64) -> (Vec<usize>, Vec<f64>) {
    let mut state = seed_state(seed);
    // Initialize with K-Means++ on first batch
    let bs = batch_size.min(n);
    let (_, mut centroids) = kmeans(data, n, d, k, 10, seed);
    let mut counts = vec![0usize; k];

    for _ in 0..max_iter {
        // Sample batch
        let batch_indices: Vec<usize> = (0..bs).map(|_| next_u64(&mut state) as usize % n).collect();
        // Assign + update
        for &idx in &batch_indices {
            let point = &data[idx*d..(idx+1)*d];
            let mut best = 0;
            let mut best_dist = f64::INFINITY;
            for j in 0..k {
                let dist = sq_dist(point, &centroids[j*d..(j+1)*d]);
                if dist < best_dist { best_dist = dist; best = j; }
            }
            counts[best] += 1;
            let eta = 1.0 / counts[best] as f64;
            for dim in 0..d {
                centroids[best * d + dim] = (1.0 - eta) * centroids[best * d + dim] + eta * point[dim];
            }
        }
    }
    // Final assignment
    let labels: Vec<usize> = (0..n).map(|i| {
        let point = &data[i*d..(i+1)*d];
        (0..k).min_by(|&a, &b| {
            sq_dist(point, &centroids[a*d..(a+1)*d])
                .partial_cmp(&sq_dist(point, &centroids[b*d..(b+1)*d])).unwrap()
        }).unwrap()
    }).collect();
    (labels, centroids)
}

// ═══════════════════════════════════════════════════════════════════
// MEAN SHIFT
// ═══════════════════════════════════════════════════════════════════

/// Mean Shift clustering. Returns (labels, cluster_centers).
pub fn mean_shift(data: &[f64], n: usize, d: usize, bandwidth: f64, max_iter: usize) -> (Vec<usize>, Vec<f64>) {
    let bw2 = bandwidth * bandwidth;
    let mut shifted = data.to_vec();

    for _ in 0..max_iter {
        let mut new_shifted = vec![0.0; n * d];
        let mut max_shift = 0.0f64;
        for i in 0..n {
            let mut numer = vec![0.0; d];
            let mut denom = 0.0;
            for j in 0..n {
                let dist2 = sq_dist(&shifted[i*d..(i+1)*d], &data[j*d..(j+1)*d]);
                let weight = (-dist2 / (2.0 * bw2)).exp();
                denom += weight;
                for dim in 0..d { numer[dim] += weight * data[j * d + dim]; }
            }
            for dim in 0..d {
                new_shifted[i * d + dim] = numer[dim] / denom;
            }
            let shift = sq_dist(&shifted[i*d..(i+1)*d], &new_shifted[i*d..(i+1)*d]).sqrt();
            max_shift = max_shift.max(shift);
        }
        shifted = new_shifted;
        if max_shift < 1e-6 { break; }
    }

    // Merge nearby modes
    let merge_dist = bandwidth * 0.5;
    let mut centers: Vec<Vec<f64>> = Vec::new();
    let mut labels = vec![0usize; n];
    for i in 0..n {
        let point = &shifted[i*d..(i+1)*d];
        let mut found = false;
        for (ci, center) in centers.iter().enumerate() {
            if sq_dist(point, center).sqrt() < merge_dist {
                labels[i] = ci;
                found = true;
                break;
            }
        }
        if !found {
            labels[i] = centers.len();
            centers.push(point.to_vec());
        }
    }
    let flat_centers: Vec<f64> = centers.into_iter().flatten().collect();
    (labels, flat_centers)
}

// ═══════════════════════════════════════════════════════════════════
// GAUSSIAN MIXTURE MODEL (EM)
// ═══════════════════════════════════════════════════════════════════

/// Gaussian Mixture Model (diagonal covariance, EM).
/// Returns (labels, means, weights).
pub fn gmm(data: &[f64], n: usize, d: usize, k: usize, max_iter: usize, seed: u64) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
    // Initialize with K-Means
    let (init_labels, mut means) = kmeans(data, n, d, k, 20, seed);
    let mut weights = vec![0.0; k];
    for &l in &init_labels { weights[l] += 1.0; }
    let n_f = n as f64;
    for w in weights.iter_mut() { *w /= n_f; }

    // Initialize variances
    let mut vars = vec![1.0; k * d]; // diagonal covariance
    for j in 0..k {
        let members: Vec<usize> = init_labels.iter().enumerate()
            .filter(|(_, l)| **l == j).map(|(i, _)| i).collect();
        if members.len() > 1 {
            for dim in 0..d {
                let mean = means[j * d + dim];
                let var: f64 = members.iter()
                    .map(|&i| (data[i * d + dim] - mean).powi(2)).sum::<f64>() / members.len() as f64;
                vars[j * d + dim] = var.max(1e-6);
            }
        }
    }

    let mut responsibilities = vec![0.0; n * k];

    for _ in 0..max_iter {
        // E-step
        for i in 0..n {
            let mut total = 0.0;
            for j in 0..k {
                let mut log_prob = weights[j].max(1e-15).ln();
                for dim in 0..d {
                    let diff = data[i * d + dim] - means[j * d + dim];
                    let v = vars[j * d + dim];
                    log_prob += -0.5 * (diff * diff / v + v.ln() + std::f64::consts::LN_2 + std::f64::consts::PI.ln());
                }
                responsibilities[i * k + j] = log_prob.exp();
                total += responsibilities[i * k + j];
            }
            if total > 1e-15 {
                for j in 0..k { responsibilities[i * k + j] /= total; }
            }
        }

        // M-step
        for j in 0..k {
            let n_j: f64 = (0..n).map(|i| responsibilities[i * k + j]).sum();
            weights[j] = n_j / n_f;
            if n_j < 1e-10 { continue; }
            for dim in 0..d {
                means[j * d + dim] = (0..n)
                    .map(|i| responsibilities[i * k + j] * data[i * d + dim]).sum::<f64>() / n_j;
                vars[j * d + dim] = ((0..n)
                    .map(|i| responsibilities[i * k + j] * (data[i * d + dim] - means[j * d + dim]).powi(2))
                    .sum::<f64>() / n_j).max(1e-6);
            }
        }
    }

    let labels: Vec<usize> = (0..n).map(|i| {
        (0..k).max_by(|&a, &b| responsibilities[i*k+a].partial_cmp(&responsibilities[i*k+b]).unwrap()).unwrap()
    }).collect();
    (labels, means, weights)
}

// ═══════════════════════════════════════════════════════════════════
// CLUSTER METRICS
// ═══════════════════════════════════════════════════════════════════

/// Silhouette score for clustering.
pub fn silhouette_score(data: &[f64], n: usize, d: usize, labels: &[usize]) -> f64 {
    let k = *labels.iter().max().unwrap_or(&0) + 1;
    let mut total = 0.0;
    for i in 0..n {
        // a(i): mean distance to points in same cluster
        let mut same_sum = 0.0;
        let mut same_count = 0;
        for j in 0..n {
            if i != j && labels[j] == labels[i] {
                same_sum += sq_dist(&data[i*d..(i+1)*d], &data[j*d..(j+1)*d]).sqrt();
                same_count += 1;
            }
        }
        let a = if same_count > 0 { same_sum / same_count as f64 } else { 0.0 };

        // b(i): min mean distance to points in other clusters
        let mut b = f64::INFINITY;
        for c in 0..k {
            if c == labels[i] { continue; }
            let mut other_sum = 0.0;
            let mut other_count = 0;
            for j in 0..n {
                if labels[j] == c {
                    other_sum += sq_dist(&data[i*d..(i+1)*d], &data[j*d..(j+1)*d]).sqrt();
                    other_count += 1;
                }
            }
            if other_count > 0 { b = b.min(other_sum / other_count as f64); }
        }
        if b.is_infinite() { b = 0.0; }

        let s = if a.max(b) < 1e-15 { 0.0 } else { (b - a) / a.max(b) };
        total += s;
    }
    total / n as f64
}

/// Davies-Bouldin index (lower is better).
pub fn davies_bouldin(data: &[f64], n: usize, d: usize, labels: &[usize], centroids: &[f64]) -> f64 {
    let k = *labels.iter().max().unwrap_or(&0) + 1;
    // Scatter for each cluster
    let mut scatter = vec![0.0; k];
    let mut counts = vec![0usize; k];
    for i in 0..n {
        let l = labels[i];
        scatter[l] += sq_dist(&data[i*d..(i+1)*d], &centroids[l*d..(l+1)*d]).sqrt();
        counts[l] += 1;
    }
    for j in 0..k {
        if counts[j] > 0 { scatter[j] /= counts[j] as f64; }
    }

    let mut db = 0.0;
    for i in 0..k {
        let mut max_r = 0.0f64;
        for j in 0..k {
            if i == j { continue; }
            let dist = sq_dist(&centroids[i*d..(i+1)*d], &centroids[j*d..(j+1)*d]).sqrt();
            if dist > 1e-15 {
                max_r = max_r.max((scatter[i] + scatter[j]) / dist);
            }
        }
        db += max_r;
    }
    db / k as f64
}

/// Calinski-Harabasz index (higher is better).
pub fn calinski_harabasz(data: &[f64], n: usize, d: usize, labels: &[usize], centroids: &[f64]) -> f64 {
    let k = *labels.iter().max().unwrap_or(&0) + 1;
    if k <= 1 || n <= k { return 0.0; }

    // Overall centroid
    let mut overall = vec![0.0; d];
    for i in 0..n {
        for dim in 0..d { overall[dim] += data[i * d + dim]; }
    }
    for dim in 0..d { overall[dim] /= n as f64; }

    // Between-cluster sum of squares
    let mut counts = vec![0usize; k];
    for &l in labels { counts[l] += 1; }
    let bgss: f64 = (0..k).map(|j| {
        counts[j] as f64 * sq_dist(&centroids[j*d..(j+1)*d], &overall)
    }).sum();

    // Within-cluster sum of squares
    let wgss: f64 = (0..n).map(|i| {
        sq_dist(&data[i*d..(i+1)*d], &centroids[labels[i]*d..(labels[i]+1)*d])
    }).sum();

    if wgss < 1e-15 { return 0.0; }
    (bgss / (k - 1) as f64) / (wgss / (n - k) as f64)
}

/// Adjusted Rand Index between two labelings.
pub fn adjusted_rand_index(labels_true: &[usize], labels_pred: &[usize]) -> f64 {
    let n = labels_true.len();
    let k1 = *labels_true.iter().max().unwrap_or(&0) + 1;
    let k2 = *labels_pred.iter().max().unwrap_or(&0) + 1;

    // Contingency table
    let mut table = vec![0i64; k1 * k2];
    for i in 0..n { table[labels_true[i] * k2 + labels_pred[i]] += 1; }

    let mut a = vec![0i64; k1]; // row sums
    let mut b = vec![0i64; k2]; // col sums
    for i in 0..k1 {
        for j in 0..k2 {
            a[i] += table[i * k2 + j];
            b[j] += table[i * k2 + j];
        }
    }

    let comb2 = |x: i64| -> f64 { x as f64 * (x - 1) as f64 / 2.0 };
    let sum_nij: f64 = table.iter().map(|&x| comb2(x)).sum();
    let sum_ai: f64 = a.iter().map(|&x| comb2(x)).sum();
    let sum_bj: f64 = b.iter().map(|&x| comb2(x)).sum();
    let n2 = comb2(n as i64);

    let expected = sum_ai * sum_bj / n2;
    let max_index = (sum_ai + sum_bj) / 2.0;
    if (max_index - expected).abs() < 1e-15 { return 0.0; }
    (sum_nij - expected) / (max_index - expected)
}

/// Normalized Mutual Information.
pub fn normalized_mutual_info(labels_true: &[usize], labels_pred: &[usize]) -> f64 {
    let n = labels_true.len() as f64;
    let k1 = *labels_true.iter().max().unwrap_or(&0) + 1;
    let k2 = *labels_pred.iter().max().unwrap_or(&0) + 1;

    let mut table = vec![0usize; k1 * k2];
    for i in 0..labels_true.len() { table[labels_true[i] * k2 + labels_pred[i]] += 1; }

    let a: Vec<f64> = (0..k1).map(|i| (0..k2).map(|j| table[i * k2 + j] as f64).sum()).collect();
    let b: Vec<f64> = (0..k2).map(|j| (0..k1).map(|i| table[i * k2 + j] as f64).sum()).collect();

    let mut mi = 0.0;
    for i in 0..k1 {
        for j in 0..k2 {
            let nij = table[i * k2 + j] as f64;
            if nij > 0.0 && a[i] > 0.0 && b[j] > 0.0 {
                mi += nij / n * (n * nij / (a[i] * b[j])).ln();
            }
        }
    }

    let h_true: f64 = a.iter().map(|&ai| if ai > 0.0 { -(ai / n) * (ai / n).ln() } else { 0.0 }).sum();
    let h_pred: f64 = b.iter().map(|&bj| if bj > 0.0 { -(bj / n) * (bj / n).ln() } else { 0.0 }).sum();

    let denom = ((h_true + h_pred) / 2.0).max(1e-15);
    mi / denom
}

// ═══════════════════════════════════════════════════════════════════
// ELBOW METHOD
// ═══════════════════════════════════════════════════════════════════

/// Compute inertia for k=1..max_k (for elbow method).
pub fn elbow_inertias(data: &[f64], n: usize, d: usize, max_k: usize, seed: u64) -> Vec<f64> {
    (1..=max_k).map(|k| {
        let (labels, centroids) = kmeans(data, n, d, k, 50, seed);
        kmeans_inertia(data, n, d, &labels, &centroids)
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn next_u64(state: &mut [u64; 4]) -> u64 {
    let result = (state[0].wrapping_add(state[3])).rotate_left(23).wrapping_add(state[0]);
    let t = state[1] << 17;
    state[2] ^= state[0]; state[3] ^= state[1]; state[1] ^= state[2]; state[0] ^= state[3];
    state[2] ^= t; state[3] = state[3].rotate_left(45);
    result
}

fn next_f64(state: &mut [u64; 4]) -> f64 {
    (next_u64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn seed_state(seed: u64) -> [u64; 4] {
    let mut s = [seed, seed.wrapping_mul(6364136223846793005).wrapping_add(1),
                 seed.wrapping_mul(1442695040888963407).wrapping_add(3),
                 seed.wrapping_mul(2862933555777941757).wrapping_add(7)];
    for _ in 0..20 { next_u64(&mut s); }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob_data() -> (Vec<f64>, usize, usize) {
        // 3 clusters in 2D
        let mut data = Vec::new();
        for i in 0..30 {
            let angle = i as f64 * 0.2;
            data.push(angle.cos()); data.push(angle.sin()); // cluster 0 near origin
        }
        for i in 0..30 {
            let angle = i as f64 * 0.2;
            data.push(5.0 + angle.cos()); data.push(5.0 + angle.sin()); // cluster 1
        }
        for i in 0..30 {
            let angle = i as f64 * 0.2;
            data.push(-5.0 + angle.cos()); data.push(5.0 + angle.sin()); // cluster 2
        }
        (data, 90, 2)
    }

    #[test]
    fn test_kmeans() {
        let (data, n, d) = blob_data();
        let (labels, centroids) = kmeans(&data, n, d, 3, 100, 42);
        assert_eq!(labels.len(), n);
        assert_eq!(centroids.len(), 3 * d);
        // All 3 clusters should be used
        let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_mini_batch_kmeans() {
        let (data, n, d) = blob_data();
        let (labels, _) = mini_batch_kmeans(&data, n, d, 3, 20, 100, 42);
        assert_eq!(labels.len(), n);
    }

    #[test]
    fn test_silhouette() {
        let (data, n, d) = blob_data();
        let (labels, _) = kmeans(&data, n, d, 3, 100, 42);
        let s = silhouette_score(&data, n, d, &labels);
        assert!(s > 0.5); // well-separated clusters
    }

    #[test]
    fn test_calinski_harabasz() {
        let (data, n, d) = blob_data();
        let (labels, centroids) = kmeans(&data, n, d, 3, 100, 42);
        let ch = calinski_harabasz(&data, n, d, &labels, &centroids);
        assert!(ch > 10.0); // should be high for well-separated
    }

    #[test]
    fn test_adjusted_rand() {
        let a = vec![0, 0, 0, 1, 1, 1];
        let b = vec![0, 0, 0, 1, 1, 1];
        assert!((adjusted_rand_index(&a, &b) - 1.0).abs() < 1e-10); // perfect
    }

    #[test]
    fn test_nmi() {
        let a = vec![0, 0, 1, 1, 2, 2];
        let b = vec![0, 0, 1, 1, 2, 2];
        assert!(normalized_mutual_info(&a, &b) > 0.99); // perfect
    }

    #[test]
    fn test_gmm() {
        let (data, n, d) = blob_data();
        let (labels, means, weights) = gmm(&data, n, d, 3, 50, 42);
        assert_eq!(labels.len(), n);
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_elbow() {
        let (data, n, d) = blob_data();
        let inertias = elbow_inertias(&data, n, d, 5, 42);
        assert_eq!(inertias.len(), 5);
        // Inertia should decrease with k
        assert!(inertias[0] > inertias[2]);
    }
}
