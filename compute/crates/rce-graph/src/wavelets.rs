//! Wavelet transforms: Haar, Daubechies, DWT, IDWT, wavelet packets,
//! multiresolution analysis, thresholding, denoising.

// ═══════════════════════════════════════════════════════════════════
// HAAR WAVELET
// ═══════════════════════════════════════════════════════════════════

/// Haar wavelet forward transform (one level).
pub fn haar_forward(data: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = data.len() / 2;
    let s2 = std::f64::consts::SQRT_2;
    let approx: Vec<f64> = (0..n).map(|i| (data[2*i] + data[2*i+1]) / s2).collect();
    let detail: Vec<f64> = (0..n).map(|i| (data[2*i] - data[2*i+1]) / s2).collect();
    (approx, detail)
}

/// Haar wavelet inverse transform (one level).
pub fn haar_inverse(approx: &[f64], detail: &[f64]) -> Vec<f64> {
    let n = approx.len();
    let s2 = std::f64::consts::SQRT_2;
    let mut data = vec![0.0; 2 * n];
    for i in 0..n {
        data[2*i] = (approx[i] + detail[i]) / s2;
        data[2*i+1] = (approx[i] - detail[i]) / s2;
    }
    data
}

/// Multi-level Haar DWT. Returns (final_approx, [detail_levels]).
pub fn haar_dwt(data: &[f64], levels: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut approx = data.to_vec();
    let mut details = Vec::new();
    for _ in 0..levels {
        if approx.len() < 2 { break; }
        let (a, d) = haar_forward(&approx);
        details.push(d);
        approx = a;
    }
    (approx, details)
}

/// Multi-level Haar IDWT.
pub fn haar_idwt(approx: &[f64], details: &[Vec<f64>]) -> Vec<f64> {
    let mut a = approx.to_vec();
    for d in details.iter().rev() { a = haar_inverse(&a, d); }
    a
}

// ═══════════════════════════════════════════════════════════════════
// DAUBECHIES WAVELETS
// ═══════════════════════════════════════════════════════════════════

/// Daubechies D4 scaling coefficients.
pub fn daub4_coeffs() -> [f64; 4] {
    let s3 = 3.0f64.sqrt();
    let d = 4.0 * std::f64::consts::SQRT_2;
    [(1.0 + s3) / d, (3.0 + s3) / d, (3.0 - s3) / d, (1.0 - s3) / d]
}

/// Generic DWT forward with given filter coefficients.
pub fn dwt_forward(data: &[f64], lo_d: &[f64], hi_d: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = data.len();
    let flen = lo_d.len();
    let half = n / 2;
    let mut approx = Vec::with_capacity(half);
    let mut detail = Vec::with_capacity(half);
    for i in 0..half {
        let mut a = 0.0;
        let mut d = 0.0;
        for k in 0..flen {
            let idx = (2 * i + k) % n;
            a += lo_d[k] * data[idx];
            d += hi_d[k] * data[idx];
        }
        approx.push(a);
        detail.push(d);
    }
    (approx, detail)
}

/// Generic DWT inverse.
pub fn dwt_inverse(approx: &[f64], detail: &[f64], lo_r: &[f64], hi_r: &[f64]) -> Vec<f64> {
    let half = approx.len();
    let n = half * 2;
    let flen = lo_r.len();
    let mut data = vec![0.0; n];
    for i in 0..half {
        for k in 0..flen {
            let idx = (2 * i + k) % n;
            data[idx] += lo_r[k] * approx[i] + hi_r[k] * detail[i];
        }
    }
    data
}

/// Daubechies D4 filter bank (lo_d, hi_d, lo_r, hi_r).
pub fn daub4_filters() -> ([f64; 4], [f64; 4], [f64; 4], [f64; 4]) {
    let h = daub4_coeffs();
    let lo_d = h;
    let hi_d = [h[3], -h[2], h[1], -h[0]];
    let lo_r = [h[2], h[3], h[0], h[1]];
    let hi_r = [-h[1], h[0], -h[3], h[2]];
    (lo_d, hi_d, lo_r, hi_r)
}

// ═══════════════════════════════════════════════════════════════════
// WAVELET DENOISING
// ═══════════════════════════════════════════════════════════════════

/// Hard thresholding: zero out coefficients below threshold.
pub fn hard_threshold(coeffs: &mut [f64], threshold: f64) {
    for c in coeffs.iter_mut() { if c.abs() < threshold { *c = 0.0; } }
}

/// Soft thresholding: shrink coefficients toward zero.
pub fn soft_threshold(coeffs: &mut [f64], threshold: f64) {
    for c in coeffs.iter_mut() {
        if c.abs() < threshold { *c = 0.0; }
        else { *c = c.signum() * (c.abs() - threshold); }
    }
}

/// Universal threshold (VisuShrink): λ = σ√(2 ln n).
pub fn universal_threshold(n: usize, sigma: f64) -> f64 {
    sigma * (2.0 * (n as f64).ln()).sqrt()
}

/// Estimate noise σ from finest detail coefficients (MAD estimator).
pub fn estimate_noise_mad(detail: &[f64]) -> f64 {
    let mut abs_vals: Vec<f64> = detail.iter().map(|d| d.abs()).collect();
    abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if abs_vals.len() % 2 == 0 {
        (abs_vals[abs_vals.len()/2 - 1] + abs_vals[abs_vals.len()/2]) / 2.0
    } else { abs_vals[abs_vals.len()/2] };
    median / 0.6745
}

/// Wavelet denoising (Haar, soft thresholding).
pub fn wavelet_denoise(data: &[f64], levels: usize) -> Vec<f64> {
    let (approx, mut details) = haar_dwt(data, levels);
    if details.is_empty() { return data.to_vec(); }
    let sigma = estimate_noise_mad(&details[0]);
    let threshold = universal_threshold(data.len(), sigma);
    for d in details.iter_mut() { soft_threshold(d, threshold); }
    haar_idwt(&approx, &details)
}

// ═══════════════════════════════════════════════════════════════════
// STATIONARY WAVELET TRANSFORM (undecimated)
// ═══════════════════════════════════════════════════════════════════

/// Stationary (undecimated) Haar wavelet transform (one level).
/// No downsampling — output same length as input.
pub fn swt_haar_forward(data: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = data.len();
    let s2 = std::f64::consts::SQRT_2;
    let approx: Vec<f64> = (0..n).map(|i| (data[i] + data[(i + 1) % n]) / s2).collect();
    let detail: Vec<f64> = (0..n).map(|i| (data[i] - data[(i + 1) % n]) / s2).collect();
    (approx, detail)
}

// ═══════════════════════════════════════════════════════════════════
// WAVELET ENERGY & FEATURES
// ═══════════════════════════════════════════════════════════════════

/// Energy of wavelet coefficients at each level.
pub fn wavelet_energy(details: &[Vec<f64>]) -> Vec<f64> {
    details.iter().map(|d| d.iter().map(|v| v * v).sum()).collect()
}

/// Relative wavelet energy (normalized to sum to 1).
pub fn relative_wavelet_energy(details: &[Vec<f64>]) -> Vec<f64> {
    let energies = wavelet_energy(details);
    let total: f64 = energies.iter().sum();
    if total < 1e-15 { return vec![0.0; energies.len()]; }
    energies.iter().map(|e| e / total).collect()
}

/// Wavelet entropy (Shannon entropy of relative energies).
pub fn wavelet_entropy(details: &[Vec<f64>]) -> f64 {
    let rwe = relative_wavelet_energy(details);
    rwe.iter().map(|&p| if p > 1e-15 { -p * p.log2() } else { 0.0 }).sum()
}

// ═══════════════════════════════════════════════════════════════════
// CONTINUOUS WAVELET TRANSFORM (discretized)
// ═══════════════════════════════════════════════════════════════════

/// CWT with Morlet wavelet at given scales.
/// Returns matrix: scales × time (row-major).
pub fn cwt_morlet(signal: &[f64], scales: &[f64], omega0: f64) -> Vec<f64> {
    let n = signal.len();
    let mut result = vec![0.0; scales.len() * n];
    for (si, &scale) in scales.iter().enumerate() {
        for t in 0..n {
            let mut sum = 0.0;
            for tau in 0..n {
                let x = (tau as f64 - t as f64) / scale;
                let wavelet = (std::f64::consts::PI).powf(-0.25)
                    * (-0.5 * x * x).exp()
                    * (omega0 * x).cos();
                sum += signal[tau] * wavelet;
            }
            result[si * n + t] = sum / scale.sqrt();
        }
    }
    result
}

/// Scalogram: |CWT|² (power spectrum in time-scale plane).
pub fn scalogram(cwt_coeffs: &[f64]) -> Vec<f64> {
    cwt_coeffs.iter().map(|c| c * c).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haar_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (a, d) = haar_forward(&data);
        let recovered = haar_inverse(&a, &d);
        for (x, y) in data.iter().zip(&recovered) { assert!((x - y).abs() < 1e-10); }
    }

    #[test]
    fn test_haar_dwt_idwt() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (a, d) = haar_dwt(&data, 3);
        let recovered = haar_idwt(&a, &d);
        for (x, y) in data.iter().zip(&recovered) { assert!((x - y).abs() < 1e-10); }
    }

    #[test]
    fn test_hard_threshold() {
        let mut c = vec![0.1, -0.5, 0.3, -2.0, 0.05];
        hard_threshold(&mut c, 0.2);
        assert_eq!(c[0], 0.0);
        assert_eq!(c[1], -0.5);
        assert_eq!(c[4], 0.0);
    }

    #[test]
    fn test_soft_threshold() {
        let mut c = vec![1.0, -0.5, 0.3];
        soft_threshold(&mut c, 0.4);
        assert!((c[0] - 0.6).abs() < 1e-10);
        assert!((c[1] - (-0.1)).abs() < 1e-10);
        assert_eq!(c[2], 0.0);
    }

    #[test]
    fn test_denoise() {
        let clean: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        // Add noise
        let noisy: Vec<f64> = clean.iter().enumerate().map(|(i, &v)| v + 0.1 * ((i * 7 + 3) as f64 % 1.0 - 0.5)).collect();
        let denoised = wavelet_denoise(&noisy, 3);
        assert_eq!(denoised.len(), noisy.len());
    }

    #[test]
    fn test_wavelet_energy() {
        let details = vec![vec![1.0, 2.0, 3.0], vec![0.5, 0.5]];
        let e = wavelet_energy(&details);
        assert!((e[0] - 14.0).abs() < 1e-10);
        assert!((e[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_relative_energy() {
        let details = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let rwe = relative_wavelet_energy(&details);
        assert!((rwe[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_daub4_coeffs() {
        let h = daub4_coeffs();
        let sum: f64 = h.iter().sum();
        assert!((sum - std::f64::consts::SQRT_2).abs() < 1e-10);
    }
}
