//! Random sampling and shuffling: reservoir sampling, bootstrap,
//! permutation tests, stratified sampling, Latin hypercube,
//! Sobol sequences, random walks, Markov chains.

// ═══════════════════════════════════════════════════════════════════
// PRNG (reuse xoshiro256++ from distributions)
// ═══════════════════════════════════════════════════════════════════

fn next_u64(state: &mut [u64; 4]) -> u64 {
    let result = (state[0].wrapping_add(state[3])).rotate_left(23).wrapping_add(state[0]);
    let t = state[1] << 17;
    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];
    state[2] ^= t;
    state[3] = state[3].rotate_left(45);
    result
}

fn next_f64(state: &mut [u64; 4]) -> f64 {
    (next_u64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn seed_state(seed: u64) -> [u64; 4] {
    let mut s = [seed, seed.wrapping_mul(6364136223846793005).wrapping_add(1),
                 seed.wrapping_mul(1442695040888963407).wrapping_add(3),
                 seed.wrapping_mul(2862933555777941757).wrapping_add(7)];
    for _ in 0..20 { next_u64(&mut s); } // warm up
    s
}

// ═══════════════════════════════════════════════════════════════════
// BASIC SAMPLING
// ═══════════════════════════════════════════════════════════════════

/// Sample k elements from data without replacement (Fisher-Yates).
pub fn sample_without_replacement(data: &[f64], k: usize, seed: u64) -> Vec<f64> {
    let n = data.len();
    let k = k.min(n);
    let mut arr = data.to_vec();
    let mut state = seed_state(seed);
    for i in 0..k {
        let j = i + (next_u64(&mut state) as usize % (n - i));
        arr.swap(i, j);
    }
    arr[..k].to_vec()
}

/// Sample k elements from data with replacement.
pub fn sample_with_replacement(data: &[f64], k: usize, seed: u64) -> Vec<f64> {
    let n = data.len();
    let mut state = seed_state(seed);
    (0..k).map(|_| data[next_u64(&mut state) as usize % n]).collect()
}

/// Weighted sampling without replacement.
pub fn weighted_sample(data: &[f64], weights: &[f64], k: usize, seed: u64) -> Vec<f64> {
    let n = data.len();
    let k = k.min(n);
    let mut state = seed_state(seed);
    // Efraimidis-Spirakis: key = random^(1/weight), pick top k
    let mut keyed: Vec<(f64, usize)> = weights.iter().enumerate().map(|(i, &w)| {
        let u = next_f64(&mut state).max(1e-15);
        (u.powf(1.0 / w.max(1e-15)), i)
    }).collect();
    keyed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    keyed[..k].iter().map(|(_, i)| data[*i]).collect()
}

/// Shuffle array in place (Fisher-Yates).
pub fn shuffle(data: &mut [f64], seed: u64) {
    let n = data.len();
    let mut state = seed_state(seed);
    for i in (1..n).rev() {
        let j = next_u64(&mut state) as usize % (i + 1);
        data.swap(i, j);
    }
}

// ═══════════════════════════════════════════════════════════════════
// RESERVOIR SAMPLING
// ═══════════════════════════════════════════════════════════════════

/// Reservoir sampling: sample k elements from a stream of unknown length.
pub fn reservoir_sample(stream: &[f64], k: usize, seed: u64) -> Vec<f64> {
    let mut state = seed_state(seed);
    let mut reservoir = Vec::with_capacity(k);
    for (i, &val) in stream.iter().enumerate() {
        if i < k {
            reservoir.push(val);
        } else {
            let j = next_u64(&mut state) as usize % (i + 1);
            if j < k {
                reservoir[j] = val;
            }
        }
    }
    reservoir
}

// ═══════════════════════════════════════════════════════════════════
// BOOTSTRAP
// ═══════════════════════════════════════════════════════════════════

/// Bootstrap resampling. Returns n_boot bootstrap samples of the statistic.
pub fn bootstrap(data: &[f64], statistic: fn(&[f64]) -> f64, n_boot: usize, seed: u64) -> Vec<f64> {
    let n = data.len();
    let mut state = seed_state(seed);
    (0..n_boot).map(|_| {
        let sample: Vec<f64> = (0..n).map(|_| data[next_u64(&mut state) as usize % n]).collect();
        statistic(&sample)
    }).collect()
}

/// Bootstrap confidence interval (percentile method).
pub fn bootstrap_ci(data: &[f64], statistic: fn(&[f64]) -> f64, n_boot: usize, confidence: f64, seed: u64) -> (f64, f64) {
    let mut stats = bootstrap(data, statistic, n_boot, seed);
    stats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let alpha = (1.0 - confidence) / 2.0;
    let lo = stats[(alpha * n_boot as f64) as usize];
    let hi_idx = ((1.0 - alpha) * n_boot as f64) as usize;
    let hi = stats[hi_idx.min(n_boot - 1)];
    (lo, hi)
}

// ═══════════════════════════════════════════════════════════════════
// PERMUTATION TEST
// ═══════════════════════════════════════════════════════════════════

/// Permutation test for difference in means.
/// Returns (observed_diff, p_value).
pub fn permutation_test_mean(a: &[f64], b: &[f64], n_perms: usize, seed: u64) -> (f64, f64) {
    let na = a.len();
    let mut combined: Vec<f64> = a.iter().chain(b.iter()).copied().collect();
    let n = combined.len();
    let obs_diff = (a.iter().sum::<f64>() / na as f64) - (b.iter().sum::<f64>() / b.len() as f64);

    let mut state = seed_state(seed);
    let mut count = 0usize;
    for _ in 0..n_perms {
        // Shuffle combined
        for i in (1..n).rev() {
            let j = next_u64(&mut state) as usize % (i + 1);
            combined.swap(i, j);
        }
        let mean_a = combined[..na].iter().sum::<f64>() / na as f64;
        let mean_b = combined[na..].iter().sum::<f64>() / (n - na) as f64;
        if (mean_a - mean_b).abs() >= obs_diff.abs() { count += 1; }
    }
    (obs_diff, count as f64 / n_perms as f64)
}

// ═══════════════════════════════════════════════════════════════════
// STRATIFIED SAMPLING
// ═══════════════════════════════════════════════════════════════════

/// Stratified sampling: sample proportionally from each stratum.
/// strata: indices for each group. k: total sample size.
pub fn stratified_sample(data: &[f64], strata: &[Vec<usize>], k: usize, seed: u64) -> Vec<f64> {
    let n_total: usize = strata.iter().map(|s| s.len()).sum();
    let mut state = seed_state(seed);
    let mut result = Vec::with_capacity(k);
    for stratum in strata {
        let n_sample = ((stratum.len() as f64 / n_total as f64) * k as f64).round() as usize;
        let n_sample = n_sample.min(stratum.len());
        let mut indices = stratum.clone();
        for i in 0..n_sample {
            let j = i + (next_u64(&mut state) as usize % (indices.len() - i));
            indices.swap(i, j);
        }
        for i in 0..n_sample {
            result.push(data[indices[i]]);
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// LATIN HYPERCUBE SAMPLING
// ═══════════════════════════════════════════════════════════════════

/// Latin Hypercube Sampling in d dimensions, n samples.
/// Returns n×d matrix (row-major) with values in [0, 1].
pub fn latin_hypercube(n: usize, d: usize, seed: u64) -> Vec<f64> {
    let mut state = seed_state(seed);
    let mut result = vec![0.0; n * d];
    for dim in 0..d {
        // Create permutation of 0..n
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = next_u64(&mut state) as usize % (i + 1);
            perm.swap(i, j);
        }
        for i in 0..n {
            let u = next_f64(&mut state);
            result[i * d + dim] = (perm[i] as f64 + u) / n as f64;
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// RANDOM WALKS
// ═══════════════════════════════════════════════════════════════════

/// 1D random walk: n steps, each ±step_size with equal probability.
pub fn random_walk_1d(n: usize, step_size: f64, seed: u64) -> Vec<f64> {
    let mut state = seed_state(seed);
    let mut path = vec![0.0];
    for _ in 0..n {
        let step = if next_u64(&mut state) % 2 == 0 { step_size } else { -step_size };
        path.push(path.last().unwrap() + step);
    }
    path
}

/// 2D random walk. Returns (xs, ys).
pub fn random_walk_2d(n: usize, step_size: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut state = seed_state(seed);
    let mut xs = vec![0.0];
    let mut ys = vec![0.0];
    for _ in 0..n {
        let angle = next_f64(&mut state) * 2.0 * std::f64::consts::PI;
        xs.push(xs.last().unwrap() + step_size * angle.cos());
        ys.push(ys.last().unwrap() + step_size * angle.sin());
    }
    (xs, ys)
}

/// Geometric Brownian Motion: S(t+dt) = S(t) * exp((μ - σ²/2)dt + σ√dt · Z).
pub fn geometric_brownian_motion(s0: f64, mu: f64, sigma: f64, dt: f64, n_steps: usize, seed: u64) -> Vec<f64> {
    let mut state = seed_state(seed);
    let mut path = vec![s0];
    let drift = (mu - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    for _ in 0..n_steps {
        // Box-Muller for standard normal
        let u1 = next_f64(&mut state).max(1e-15);
        let u2 = next_f64(&mut state);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let s = path.last().unwrap() * (drift + vol * z).exp();
        path.push(s);
    }
    path
}

// ═══════════════════════════════════════════════════════════════════
// MARKOV CHAINS
// ═══════════════════════════════════════════════════════════════════

/// Simulate a Markov chain. transition_matrix is n×n (row-major, rows sum to 1).
/// Returns sequence of state indices.
pub fn markov_chain(transition: &[f64], n_states: usize, initial_state: usize, n_steps: usize, seed: u64) -> Vec<usize> {
    let mut state = seed_state(seed);
    let mut current = initial_state;
    let mut chain = vec![current];
    for _ in 0..n_steps {
        let u = next_f64(&mut state);
        let row = &transition[current * n_states..(current + 1) * n_states];
        let mut cum = 0.0;
        for (j, &p) in row.iter().enumerate() {
            cum += p;
            if u <= cum {
                current = j;
                break;
            }
        }
        chain.push(current);
    }
    chain
}

/// Estimate stationary distribution from transition matrix (power iteration).
pub fn stationary_distribution(transition: &[f64], n_states: usize, max_iter: usize) -> Vec<f64> {
    let mut pi = vec![1.0 / n_states as f64; n_states];
    for _ in 0..max_iter {
        let mut new_pi = vec![0.0; n_states];
        for j in 0..n_states {
            for i in 0..n_states {
                new_pi[j] += pi[i] * transition[i * n_states + j];
            }
        }
        let diff: f64 = pi.iter().zip(&new_pi).map(|(a, b)| (a - b).abs()).sum();
        pi = new_pi;
        if diff < 1e-10 { break; }
    }
    pi
}

// ═══════════════════════════════════════════════════════════════════
// QUASI-RANDOM SEQUENCES
// ═══════════════════════════════════════════════════════════════════

/// Halton sequence in 1D (base b).
pub fn halton_sequence(n: usize, base: usize) -> Vec<f64> {
    (1..=n).map(|i| {
        let mut f = 1.0;
        let mut r = 0.0;
        let mut idx = i;
        while idx > 0 {
            f /= base as f64;
            r += f * (idx % base) as f64;
            idx /= base;
        }
        r
    }).collect()
}

/// Halton sequence in d dimensions (bases = first d primes).
pub fn halton_sequence_nd(n: usize, d: usize) -> Vec<Vec<f64>> {
    let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
    (0..d.min(primes.len())).map(|dim| halton_sequence(n, primes[dim])).collect()
}

/// Van der Corput sequence (base 2).
pub fn van_der_corput(n: usize) -> Vec<f64> {
    halton_sequence(n, 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_without_replacement() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let s = sample_without_replacement(&data, 5, 42);
        assert_eq!(s.len(), 5);
        // All unique
        let mut sorted = s.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 1..sorted.len() { assert!(sorted[i] > sorted[i-1]); }
    }

    #[test]
    fn test_shuffle() {
        let mut data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        shuffle(&mut data, 42);
        // Should still contain all elements
        let mut sorted = data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sorted, (0..10).map(|i| i as f64).collect::<Vec<f64>>());
    }

    #[test]
    fn test_reservoir_sample() {
        let stream: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let s = reservoir_sample(&stream, 10, 42);
        assert_eq!(s.len(), 10);
    }

    #[test]
    fn test_bootstrap() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        fn mean(d: &[f64]) -> f64 { d.iter().sum::<f64>() / d.len() as f64 }
        let stats = bootstrap(&data, mean, 100, 42);
        assert_eq!(stats.len(), 100);
        // Bootstrap means should be near original mean
        let boot_mean = stats.iter().sum::<f64>() / 100.0;
        assert!((boot_mean - 3.0).abs() < 1.0);
    }

    #[test]
    fn test_bootstrap_ci() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        fn mean(d: &[f64]) -> f64 { d.iter().sum::<f64>() / d.len() as f64 }
        let (lo, hi) = bootstrap_ci(&data, mean, 1000, 0.95, 42);
        assert!(lo < 5.5 && hi > 5.5); // true mean = 5.5
    }

    #[test]
    fn test_latin_hypercube() {
        let lhs = latin_hypercube(10, 3, 42);
        assert_eq!(lhs.len(), 30); // 10 samples × 3 dims
        for &v in &lhs { assert!(v >= 0.0 && v <= 1.0); }
    }

    #[test]
    fn test_random_walk_1d() {
        let walk = random_walk_1d(100, 1.0, 42);
        assert_eq!(walk.len(), 101);
        assert_eq!(walk[0], 0.0);
    }

    #[test]
    fn test_gbm() {
        let path = geometric_brownian_motion(100.0, 0.05, 0.2, 1.0 / 252.0, 252, 42);
        assert_eq!(path.len(), 253);
        assert!(path[0] == 100.0);
        assert!(path.last().unwrap() > &0.0); // should stay positive
    }

    #[test]
    fn test_markov_chain() {
        // Simple 2-state chain
        let transition = vec![0.9, 0.1, 0.3, 0.7];
        let chain = markov_chain(&transition, 2, 0, 100, 42);
        assert_eq!(chain.len(), 101);
        assert!(chain.iter().all(|&s| s < 2));
    }

    #[test]
    fn test_stationary_distribution() {
        let transition = vec![0.9, 0.1, 0.3, 0.7];
        let pi = stationary_distribution(&transition, 2, 1000);
        // Stationary: π = π·P → π₀ = 0.75, π₁ = 0.25
        assert!((pi[0] - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_halton() {
        let h = halton_sequence(10, 2);
        assert_eq!(h.len(), 10);
        // First few: 1/2, 1/4, 3/4, 1/8, 5/8, ...
        assert!((h[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_permutation_test() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (diff, p) = permutation_test_mean(&a, &b, 100, 42);
        assert!(diff.abs() < 1e-10); // same data, diff = 0
        assert!(p > 0.9); // not significant
    }
}
