//! Matrix creation utilities: special matrices, random matrices,
//! structured matrices, test matrices, sparse generators.

// ═══════════════════════════════════════════════════════════════════
// SPECIAL MATRICES
// ═══════════════════════════════════════════════════════════════════

/// Zeros matrix n×m.
pub fn zeros(n: usize, m: usize) -> Vec<f64> { vec![0.0; n * m] }

/// Ones matrix n×m.
pub fn ones(n: usize, m: usize) -> Vec<f64> { vec![1.0; n * m] }

/// Full matrix n×m filled with value.
pub fn full(n: usize, m: usize, val: f64) -> Vec<f64> { vec![val; n * m] }

/// Identity matrix n×n.
pub fn eye(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n { m[i * n + i] = 1.0; }
    m
}

/// Diagonal matrix from vector.
pub fn diag(d: &[f64]) -> Vec<f64> {
    let n = d.len();
    let mut m = vec![0.0; n * n];
    for i in 0..n { m[i * n + i] = d[i]; }
    m
}

/// Vandermonde matrix: V[i,j] = x[i]^j.
pub fn vandermonde(x: &[f64], degree: usize) -> Vec<f64> {
    let n = x.len();
    let m = degree + 1;
    let mut v = vec![0.0; n * m];
    for i in 0..n {
        let mut xp = 1.0;
        for j in 0..m { v[i * m + j] = xp; xp *= x[i]; }
    }
    v
}

/// Hilbert matrix H[i,j] = 1/(i+j+1).
pub fn hilbert(n: usize) -> Vec<f64> {
    let mut h = vec![0.0; n * n];
    for i in 0..n { for j in 0..n { h[i * n + j] = 1.0 / (i + j + 1) as f64; } }
    h
}

/// Toeplitz matrix from first row and column.
pub fn toeplitz(col: &[f64], row: &[f64]) -> Vec<f64> {
    let n = col.len();
    let m = row.len();
    let mut t = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            t[i * m + j] = if i <= j { row[j - i] } else { col[i - j] };
        }
    }
    t
}

/// Hankel matrix from first column and last row.
pub fn hankel(col: &[f64], row: &[f64]) -> Vec<f64> {
    let n = col.len();
    let m = row.len();
    let mut h = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            let idx = i + j;
            h[i * m + j] = if idx < n { col[idx] } else { row[idx - n + 1] };
        }
    }
    h
}

/// Companion matrix for polynomial coefficients [c0, c1, ..., cn-1] (monic).
pub fn companion(coeffs: &[f64]) -> Vec<f64> {
    let n = coeffs.len();
    let mut c = vec![0.0; n * n];
    for i in 0..n - 1 { c[(i + 1) * n + i] = 1.0; }
    for i in 0..n { c[i * n + n - 1] = -coeffs[i]; }
    c
}

/// Circulant matrix from first row.
pub fn circulant(row: &[f64]) -> Vec<f64> {
    let n = row.len();
    let mut c = vec![0.0; n * n];
    for i in 0..n { for j in 0..n { c[i * n + j] = row[(j + n - i) % n]; } }
    c
}

/// Hadamard matrix of order n (must be power of 2).
pub fn hadamard(n: usize) -> Vec<f64> {
    if n == 1 { return vec![1.0]; }
    let half = hadamard(n / 2);
    let h = n / 2;
    let mut result = vec![0.0; n * n];
    for i in 0..h {
        for j in 0..h {
            let v = half[i * h + j];
            result[i * n + j] = v;
            result[i * n + j + h] = v;
            result[(i + h) * n + j] = v;
            result[(i + h) * n + j + h] = -v;
        }
    }
    result
}

/// DFT matrix (discrete Fourier transform matrix).
pub fn dft_matrix(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; 2 * n * n]; // interleaved real/imag
    let w = -2.0 * std::f64::consts::PI / n as f64;
    for i in 0..n {
        for j in 0..n {
            let angle = w * i as f64 * j as f64;
            m[2 * (i * n + j)] = angle.cos();
            m[2 * (i * n + j) + 1] = angle.sin();
        }
    }
    m
}

/// Exchange (anti-diagonal identity) matrix.
pub fn exchange(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n { m[i * n + n - 1 - i] = 1.0; }
    m
}

/// Permutation matrix from permutation vector.
pub fn permutation_matrix(perm: &[usize]) -> Vec<f64> {
    let n = perm.len();
    let mut m = vec![0.0; n * n];
    for (i, &p) in perm.iter().enumerate() { m[i * n + p] = 1.0; }
    m
}

/// Block diagonal matrix from blocks.
pub fn block_diag(blocks: &[&[f64]], sizes: &[usize]) -> Vec<f64> {
    let total: usize = sizes.iter().sum();
    let mut m = vec![0.0; total * total];
    let mut offset = 0;
    for (&block, &size) in blocks.iter().zip(sizes) {
        for i in 0..size {
            for j in 0..size {
                m[(offset + i) * total + offset + j] = block[i * size + j];
            }
        }
        offset += size;
    }
    m
}

// ═══════════════════════════════════════════════════════════════════
// RANDOM MATRICES
// ═══════════════════════════════════════════════════════════════════

/// Random matrix with uniform [0, 1] entries.
pub fn rand_uniform(n: usize, m: usize, seed: u64) -> Vec<f64> {
    let mut state = seed_state(seed);
    (0..n * m).map(|_| prng_f64(&mut state)).collect()
}

/// Random matrix with standard normal entries.
pub fn rand_normal(n: usize, m: usize, seed: u64) -> Vec<f64> {
    let mut state = seed_state(seed);
    (0..n * m).map(|_| prng_normal(&mut state)).collect()
}

/// Random symmetric positive definite matrix.
pub fn rand_spd(n: usize, seed: u64) -> Vec<f64> {
    let a = rand_normal(n, n, seed);
    let mut result = vec![0.0; n * n];
    // A^T A + n*I
    for i in 0..n {
        for j in 0..n {
            for k in 0..n { result[i * n + j] += a[k * n + i] * a[k * n + j]; }
        }
        result[i * n + i] += n as f64;
    }
    result
}

/// Random orthogonal matrix (via QR of random).
pub fn rand_orthogonal(n: usize, seed: u64) -> Vec<f64> {
    let a = rand_normal(n, n, seed);
    let (q, _) = crate::linalg_decomp::qr_decompose(&a, n, n);
    q
}

/// Random sparse matrix with given density.
pub fn rand_sparse(n: usize, m: usize, density: f64, seed: u64) -> Vec<f64> {
    let mut state = seed_state(seed);
    (0..n * m).map(|_| {
        if prng_f64(&mut state) < density { prng_normal(&mut state) } else { 0.0 }
    }).collect()
}

/// Random correlation matrix.
pub fn rand_correlation(n: usize, seed: u64) -> Vec<f64> {
    let spd = rand_spd(n, seed);
    let diag: Vec<f64> = (0..n).map(|i| spd[i * n + i].sqrt()).collect();
    let mut corr = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            corr[i * n + j] = spd[i * n + j] / (diag[i] * diag[j]);
        }
    }
    corr
}

// ═══════════════════════════════════════════════════════════════════
// STRUCTURED CREATION
// ═══════════════════════════════════════════════════════════════════

/// Linspace vector: n values from start to stop inclusive.
pub fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    if n <= 1 { return vec![start]; }
    let step = (stop - start) / (n - 1) as f64;
    (0..n).map(|i| start + i as f64 * step).collect()
}

/// Logspace vector: n values from 10^start to 10^stop.
pub fn logspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    linspace(start, stop, n).iter().map(|&v| 10.0f64.powf(v)).collect()
}

/// Geomspace: n values geometrically spaced from start to stop.
pub fn geomspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    let log_start = start.ln();
    let log_stop = stop.ln();
    linspace(log_start, log_stop, n).iter().map(|&v| v.exp()).collect()
}

/// Repeat a value n times.
pub fn repeat(val: f64, n: usize) -> Vec<f64> { vec![val; n] }

/// Tile (repeat) a vector n times.
pub fn tile(data: &[f64], n: usize) -> Vec<f64> {
    data.iter().copied().cycle().take(data.len() * n).collect()
}

/// Meshgrid from two 1D arrays.
pub fn meshgrid(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let nx = x.len();
    let ny = y.len();
    let mut xg = Vec::with_capacity(ny * nx);
    let mut yg = Vec::with_capacity(ny * nx);
    for &yi in y { for &xi in x { xg.push(xi); yg.push(yi); } }
    (xg, yg)
}

fn seed_state(seed: u64) -> [u64; 4] {
    let mut s = [seed, seed.wrapping_mul(6364136223846793005).wrapping_add(1),
                 seed.wrapping_mul(1442695040888963407).wrapping_add(3),
                 seed.wrapping_mul(2862933555777941757).wrapping_add(7)];
    for _ in 0..20 { prng_next(&mut s); } s
}
fn prng_next(s: &mut [u64; 4]) -> u64 {
    let r = (s[0].wrapping_add(s[3])).rotate_left(23).wrapping_add(s[0]);
    let t = s[1] << 17; s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = s[3].rotate_left(45); r
}
fn prng_f64(s: &mut [u64; 4]) -> f64 { (prng_next(s) >> 11) as f64 / (1u64 << 53) as f64 }
fn prng_normal(s: &mut [u64; 4]) -> f64 {
    let u1 = prng_f64(s).max(1e-15); let u2 = prng_f64(s);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eye() {
        let i = eye(3);
        assert_eq!(i[0], 1.0); assert_eq!(i[1], 0.0); assert_eq!(i[4], 1.0);
    }

    #[test]
    fn test_hilbert() {
        let h = hilbert(3);
        assert!((h[0] - 1.0).abs() < 1e-10);
        assert!((h[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_vandermonde() {
        let v = vandermonde(&[1.0, 2.0, 3.0], 2);
        assert_eq!(v, vec![1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 1.0, 3.0, 9.0]);
    }

    #[test]
    fn test_circulant() {
        let c = circulant(&[1.0, 2.0, 3.0]);
        assert_eq!(c[0], 1.0); assert_eq!(c[1], 2.0); assert_eq!(c[2], 3.0);
        assert_eq!(c[3], 3.0); assert_eq!(c[4], 1.0); assert_eq!(c[5], 2.0);
    }

    #[test]
    fn test_hadamard() {
        let h = hadamard(4);
        assert_eq!(h.len(), 16);
        // Rows should be orthogonal: H * H^T = n * I
        for i in 0..4 {
            for j in 0..4 {
                let dot: f64 = (0..4).map(|k| h[i * 4 + k] * h[j * 4 + k]).sum();
                if i == j { assert!((dot - 4.0).abs() < 1e-10); }
                else { assert!(dot.abs() < 1e-10); }
            }
        }
    }

    #[test]
    fn test_exchange() {
        let e = exchange(3);
        assert_eq!(e[2], 1.0); // (0,2)
        assert_eq!(e[4], 1.0); // (1,1)
        assert_eq!(e[6], 1.0); // (2,0)
    }

    #[test]
    fn test_linspace() {
        let l = linspace(0.0, 1.0, 5);
        assert_eq!(l, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn test_logspace() {
        let l = logspace(0.0, 2.0, 3);
        assert!((l[0] - 1.0).abs() < 1e-10);
        assert!((l[2] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_rand_spd() {
        let m = rand_spd(3, 42);
        // Should be symmetric
        assert!(crate::matrix_io::is_symmetric(&m, 3, 1e-10));
    }

    #[test]
    fn test_rand_sparse() {
        let m = rand_sparse(10, 10, 0.1, 42);
        let nnz = m.iter().filter(|&&v| v.abs() > 1e-15).count();
        assert!(nnz < 50); // sparse
    }

    #[test]
    fn test_tile() {
        assert_eq!(tile(&[1.0, 2.0], 3), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_block_diag() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2×2
        let b = vec![5.0]; // 1×1
        let bd = block_diag(&[&a, &b], &[2, 1]);
        assert_eq!(bd.len(), 9); // 3×3
        assert_eq!(bd[8], 5.0); // bottom-right
    }
}
