//! Information theory: entropy, mutual information, divergences,
//! channel capacity, coding theory, compression metrics.

// ═══════════════════════════════════════════════════════════════════
// ENTROPY
// ═══════════════════════════════════════════════════════════════════

/// Shannon entropy H(X) in bits.
pub fn entropy(probs: &[f64]) -> f64 {
    probs.iter().map(|&p| if p > 1e-15 { -p * p.log2() } else { 0.0 }).sum()
}

/// Shannon entropy in nats (base e).
pub fn entropy_nats(probs: &[f64]) -> f64 {
    probs.iter().map(|&p| if p > 1e-15 { -p * p.ln() } else { 0.0 }).sum()
}

/// Joint entropy H(X,Y) from joint probability table (row-major).
pub fn joint_entropy(joint: &[f64]) -> f64 {
    joint.iter().map(|&p| if p > 1e-15 { -p * p.log2() } else { 0.0 }).sum()
}

/// Conditional entropy H(X|Y) = H(X,Y) - H(Y).
pub fn conditional_entropy(joint: &[f64], n_x: usize, n_y: usize) -> f64 {
    let h_joint = joint_entropy(joint);
    // Marginal Y
    let py: Vec<f64> = (0..n_y).map(|j| (0..n_x).map(|i| joint[i * n_y + j]).sum()).collect();
    let h_y = entropy(&py);
    h_joint - h_y
}

/// Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y).
pub fn mutual_information(joint: &[f64], n_x: usize, n_y: usize) -> f64 {
    let px: Vec<f64> = (0..n_x).map(|i| (0..n_y).map(|j| joint[i * n_y + j]).sum()).collect();
    let py: Vec<f64> = (0..n_y).map(|j| (0..n_x).map(|i| joint[i * n_y + j]).sum()).collect();
    entropy(&px) + entropy(&py) - joint_entropy(joint)
}

/// Normalized mutual information: NMI = 2*I(X;Y) / (H(X) + H(Y)).
pub fn normalized_mi(joint: &[f64], n_x: usize, n_y: usize) -> f64 {
    let px: Vec<f64> = (0..n_x).map(|i| (0..n_y).map(|j| joint[i * n_y + j]).sum()).collect();
    let py: Vec<f64> = (0..n_y).map(|j| (0..n_x).map(|i| joint[i * n_y + j]).sum()).collect();
    let hx = entropy(&px);
    let hy = entropy(&py);
    let mi = mutual_information(joint, n_x, n_y);
    if hx + hy < 1e-15 { return 0.0; }
    2.0 * mi / (hx + hy)
}

/// Cross entropy H(p,q) = -Σ p(x) log q(x).
pub fn cross_entropy(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).map(|(&pi, &qi)| {
        if pi > 1e-15 { -pi * qi.max(1e-15).log2() } else { 0.0 }
    }).sum()
}

/// Rényi entropy of order α.
pub fn renyi_entropy(probs: &[f64], alpha: f64) -> f64 {
    if (alpha - 1.0).abs() < 1e-10 { return entropy(probs); }
    let sum: f64 = probs.iter().map(|&p| p.powf(alpha)).sum();
    (1.0 / (1.0 - alpha)) * sum.log2()
}

/// Tsallis entropy of order q.
pub fn tsallis_entropy(probs: &[f64], q: f64) -> f64 {
    if (q - 1.0).abs() < 1e-10 { return entropy_nats(probs); }
    let sum: f64 = probs.iter().map(|&p| p.powf(q)).sum();
    (1.0 - sum) / (q - 1.0)
}

/// Min-entropy H_∞(X) = -log2(max(p)).
pub fn min_entropy(probs: &[f64]) -> f64 {
    let max_p = probs.iter().copied().fold(0.0, f64::max);
    if max_p < 1e-15 { return 0.0; }
    -max_p.log2()
}

/// Max-entropy H_0(X) = log2(|support|).
pub fn max_entropy(probs: &[f64]) -> f64 {
    let support = probs.iter().filter(|&&p| p > 1e-15).count();
    (support as f64).log2()
}

// ═══════════════════════════════════════════════════════════════════
// DIVERGENCES
// ═══════════════════════════════════════════════════════════════════

/// KL divergence D_KL(P || Q) in bits.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).map(|(&pi, &qi)| {
        if pi > 1e-15 { pi * (pi / qi.max(1e-15)).log2() } else { 0.0 }
    }).sum()
}

/// KL divergence in nats.
pub fn kl_divergence_nats(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).map(|(&pi, &qi)| {
        if pi > 1e-15 { pi * (pi / qi.max(1e-15)).ln() } else { 0.0 }
    }).sum()
}

/// Jensen-Shannon divergence (symmetric, bounded).
pub fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let m: Vec<f64> = p.iter().zip(q).map(|(&pi, &qi)| (pi + qi) / 2.0).collect();
    (kl_divergence(p, &m) + kl_divergence(q, &m)) / 2.0
}

/// Total variation distance: TV(P,Q) = ½ Σ |p(x) - q(x)|.
pub fn total_variation(p: &[f64], q: &[f64]) -> f64 {
    0.5 * p.iter().zip(q).map(|(&pi, &qi)| (pi - qi).abs()).sum::<f64>()
}

/// Hellinger distance: H(P,Q) = sqrt(1 - Σ sqrt(p·q)).
pub fn hellinger_distance(p: &[f64], q: &[f64]) -> f64 {
    let bc: f64 = p.iter().zip(q).map(|(&pi, &qi)| (pi * qi).sqrt()).sum();
    (1.0 - bc).max(0.0).sqrt()
}

/// Chi-squared divergence.
pub fn chi_squared_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).map(|(&pi, &qi)| {
        if qi > 1e-15 { (pi - qi).powi(2) / qi } else { 0.0 }
    }).sum()
}

/// Wasserstein distance (1D, Earth mover's distance).
pub fn wasserstein_1d(p: &[f64], q: &[f64]) -> f64 {
    let mut cum_p = 0.0;
    let mut cum_q = 0.0;
    let mut dist = 0.0;
    for (&pi, &qi) in p.iter().zip(q) {
        cum_p += pi;
        cum_q += qi;
        dist += (cum_p - cum_q).abs();
    }
    dist
}

/// f-divergence with generator function f.
pub fn f_divergence(p: &[f64], q: &[f64], f: impl Fn(f64) -> f64) -> f64 {
    p.iter().zip(q).map(|(&pi, &qi)| {
        if qi > 1e-15 { qi * f(pi / qi) } else { 0.0 }
    }).sum()
}

// ═══════════════════════════════════════════════════════════════════
// CHANNEL CAPACITY
// ═══════════════════════════════════════════════════════════════════

/// Binary symmetric channel capacity: C = 1 - H(p).
pub fn bsc_capacity(error_prob: f64) -> f64 {
    1.0 - binary_entropy(error_prob)
}

/// Binary erasure channel capacity: C = 1 - ε.
pub fn bec_capacity(erasure_prob: f64) -> f64 {
    1.0 - erasure_prob
}

/// Binary entropy function H(p).
pub fn binary_entropy(p: f64) -> f64 {
    if p < 1e-15 || p > 1.0 - 1e-15 { return 0.0; }
    -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
}

/// AWGN channel capacity: C = ½ log2(1 + SNR).
pub fn awgn_capacity(snr: f64) -> f64 {
    0.5 * (1.0 + snr).log2()
}

/// AWGN capacity with bandwidth: C = B·log2(1 + S/N).
pub fn shannon_hartley(bandwidth: f64, snr: f64) -> f64 {
    bandwidth * (1.0 + snr).log2()
}

// ═══════════════════════════════════════════════════════════════════
// CODING THEORY METRICS
// ═══════════════════════════════════════════════════════════════════

/// Compression ratio: original_size / compressed_size.
pub fn compression_ratio(original: usize, compressed: usize) -> f64 {
    original as f64 / compressed.max(1) as f64
}

/// Space savings: 1 - compressed/original.
pub fn space_savings(original: usize, compressed: usize) -> f64 {
    1.0 - compressed as f64 / original.max(1) as f64
}

/// Entropy rate (bits per symbol) from frequency counts.
pub fn entropy_rate(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    if total == 0 { return 0.0; }
    let probs: Vec<f64> = counts.iter().map(|&c| c as f64 / total as f64).collect();
    entropy(&probs)
}

/// Perplexity: 2^H(X).
pub fn perplexity(probs: &[f64]) -> f64 {
    2.0f64.powf(entropy(probs))
}

/// Pointwise mutual information: PMI(x,y) = log2(p(x,y) / (p(x)·p(y))).
pub fn pmi(p_xy: f64, p_x: f64, p_y: f64) -> f64 {
    if p_xy < 1e-15 || p_x < 1e-15 || p_y < 1e-15 { return 0.0; }
    (p_xy / (p_x * p_y)).log2()
}

/// Normalized PMI: NPMI = PMI / -log2(p(x,y)).
pub fn npmi(p_xy: f64, p_x: f64, p_y: f64) -> f64 {
    if p_xy < 1e-15 { return -1.0; }
    let pmi_val = pmi(p_xy, p_x, p_y);
    pmi_val / (-p_xy.log2())
}

/// Surprise (information content): I(x) = -log2(p(x)).
pub fn surprise(p: f64) -> f64 {
    if p < 1e-15 { return f64::INFINITY; }
    -p.log2()
}

// ═══════════════════════════════════════════════════════════════════
// RATE-DISTORTION
// ═══════════════════════════════════════════════════════════════════

/// Rate-distortion function for Gaussian source: R(D) = ½ log2(σ²/D).
pub fn rate_distortion_gaussian(variance: f64, distortion: f64) -> f64 {
    if distortion >= variance { return 0.0; }
    0.5 * (variance / distortion).log2()
}

/// Rate-distortion for binary source: R(D) = H(p) - H(D) for D < min(p, 1-p).
pub fn rate_distortion_binary(p: f64, distortion: f64) -> f64 {
    let max_d = p.min(1.0 - p);
    if distortion >= max_d { return 0.0; }
    binary_entropy(p) - binary_entropy(distortion)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_uniform() {
        // 8 equally likely outcomes → 3 bits
        let probs = vec![0.125; 8];
        assert!((entropy(&probs) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_deterministic() {
        let probs = vec![1.0, 0.0, 0.0];
        assert!(entropy(&probs).abs() < 1e-10);
    }

    #[test]
    fn test_binary_entropy() {
        assert!((binary_entropy(0.5) - 1.0).abs() < 1e-10);
        assert!(binary_entropy(0.0).abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        assert!(kl_divergence(&p, &q).abs() < 1e-10);
    }

    #[test]
    fn test_js_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        assert!(js_divergence(&p, &q).abs() < 1e-10);
        // JS is symmetric
        let a = vec![0.9, 0.1];
        let b = vec![0.1, 0.9];
        assert!((js_divergence(&a, &b) - js_divergence(&b, &a)).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_info() {
        // Independent: MI = 0
        let joint = vec![0.25, 0.25, 0.25, 0.25]; // 2×2 uniform
        assert!(mutual_information(&joint, 2, 2).abs() < 1e-10);
    }

    #[test]
    fn test_cross_entropy() {
        let p = vec![1.0, 0.0];
        let q = vec![0.9, 0.1];
        let ce = cross_entropy(&p, &q);
        assert!(ce > 0.0);
    }

    #[test]
    fn test_bsc_capacity() {
        assert!((bsc_capacity(0.0) - 1.0).abs() < 1e-10);
        assert!((bsc_capacity(0.5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_awgn_capacity() {
        // SNR = 1 (0 dB): C = ½ log2(1+1) = 0.5 bits/sample
        assert!((awgn_capacity(1.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_perplexity() {
        let probs = vec![0.25; 4];
        assert!((perplexity(&probs) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_surprise() {
        assert!((surprise(0.5) - 1.0).abs() < 1e-10);
        assert!((surprise(0.25) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_hellinger() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        assert!(hellinger_distance(&p, &q).abs() < 1e-10);
    }

    #[test]
    fn test_total_variation() {
        let p = vec![1.0, 0.0];
        let q = vec![0.0, 1.0];
        assert!((total_variation(&p, &q) - 1.0).abs() < 1e-10);
    }
}
