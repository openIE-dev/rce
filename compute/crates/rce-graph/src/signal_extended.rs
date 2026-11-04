//! Extended signal processing: filter design, wavelets, peak detection,
//! Hilbert transform, cepstrum, zero-crossing rate, spectral features.

use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════
// PEAK DETECTION (scipy.signal.find_peaks)
// ═══════════════════════════════════════════════════════════════════

/// Find peaks in a signal. Returns indices of peaks.
pub fn find_peaks(data: &[f64], min_height: Option<f64>, min_distance: Option<usize>) -> Vec<usize> {
    let n = data.len();
    if n < 3 { return vec![]; }
    let mut peaks = Vec::new();
    for i in 1..n - 1 {
        if data[i] > data[i - 1] && data[i] > data[i + 1] {
            if let Some(h) = min_height {
                if data[i] < h { continue; }
            }
            peaks.push(i);
        }
    }
    // Enforce minimum distance
    if let Some(dist) = min_distance {
        let mut filtered = Vec::new();
        let mut last: Option<usize> = None;
        for &p in &peaks {
            if let Some(l) = last {
                if p - l < dist {
                    // Keep the taller peak
                    if data[p] > data[l] {
                        filtered.pop();
                        filtered.push(p);
                        last = Some(p);
                    }
                    continue;
                }
            }
            filtered.push(p);
            last = Some(p);
        }
        return filtered;
    }
    peaks
}

/// Peak prominences: how much a peak stands above surrounding terrain.
pub fn peak_prominences(data: &[f64], peaks: &[usize]) -> Vec<f64> {
    peaks.iter().map(|&p| {
        let left_min = data[..p].iter().copied().fold(data[p], f64::min);
        let right_min = data[p + 1..].iter().copied().fold(data[p], f64::min);
        data[p] - left_min.max(right_min)
    }).collect()
}

/// Peak widths at half prominence.
pub fn peak_widths(data: &[f64], peaks: &[usize]) -> Vec<f64> {
    let proms = peak_prominences(data, peaks);
    peaks.iter().zip(&proms).map(|(&p, &prom)| {
        let half_height = data[p] - prom / 2.0;
        let mut left = p;
        while left > 0 && data[left] > half_height { left -= 1; }
        let mut right = p;
        while right < data.len() - 1 && data[right] > half_height { right += 1; }
        (right - left) as f64
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// HILBERT TRANSFORM (analytic signal)
// ═══════════════════════════════════════════════════════════════════

/// Compute analytic signal via Hilbert transform.
/// Returns (instantaneous_amplitude, instantaneous_phase).
pub fn hilbert_transform(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len().next_power_of_two();
    let mut re: Vec<f64> = signal.to_vec();
    re.resize(n, 0.0);
    let mut im = vec![0.0; n];

    // FFT
    fft_inplace(&mut re, &mut im, false);

    // Zero negative frequencies, double positive
    re[0] *= 1.0; im[0] *= 1.0; // DC unchanged
    for i in 1..n / 2 { re[i] *= 2.0; im[i] *= 2.0; }
    // n/2 unchanged
    for i in n / 2 + 1..n { re[i] = 0.0; im[i] = 0.0; }

    // IFFT
    fft_inplace(&mut re, &mut im, true);
    let scale = 1.0 / n as f64;
    for i in 0..n { re[i] *= scale; im[i] *= scale; }

    let amp: Vec<f64> = (0..signal.len()).map(|i| (re[i] * re[i] + im[i] * im[i]).sqrt()).collect();
    let phase: Vec<f64> = (0..signal.len()).map(|i| im[i].atan2(re[i])).collect();
    (amp, phase)
}

fn fft_inplace(re: &mut [f64], im: &mut [f64], inverse: bool) {
    crate::math_runtime::fft_inplace_pub(re, im, inverse);
}

// ═══════════════════════════════════════════════════════════════════
// ZERO-CROSSING RATE
// ═══════════════════════════════════════════════════════════════════

/// Zero-crossing rate of a signal.
pub fn zero_crossing_rate(signal: &[f64]) -> f64 {
    if signal.len() < 2 { return 0.0; }
    let crossings = signal.windows(2)
        .filter(|w| (w[0] >= 0.0 && w[1] < 0.0) || (w[0] < 0.0 && w[1] >= 0.0))
        .count();
    crossings as f64 / (signal.len() - 1) as f64
}

// ═══════════════════════════════════════════════════════════════════
// SPECTRAL FEATURES
// ═══════════════════════════════════════════════════════════════════

/// Spectral centroid: center of mass of the spectrum.
pub fn spectral_centroid(magnitudes: &[f64], sample_rate: f64) -> f64 {
    let n = magnitudes.len();
    if n == 0 { return 0.0; }
    let total: f64 = magnitudes.iter().sum();
    if total < 1e-15 { return 0.0; }
    let weighted: f64 = magnitudes.iter().enumerate()
        .map(|(i, &m)| i as f64 * sample_rate / (2.0 * n as f64) * m)
        .sum();
    weighted / total
}

/// Spectral bandwidth: spread around the centroid.
pub fn spectral_bandwidth(magnitudes: &[f64], sample_rate: f64) -> f64 {
    let centroid = spectral_centroid(magnitudes, sample_rate);
    let n = magnitudes.len();
    let total: f64 = magnitudes.iter().sum();
    if total < 1e-15 { return 0.0; }
    let var: f64 = magnitudes.iter().enumerate()
        .map(|(i, &m)| {
            let freq = i as f64 * sample_rate / (2.0 * n as f64);
            (freq - centroid).powi(2) * m
        })
        .sum::<f64>() / total;
    var.sqrt()
}

/// Spectral rolloff: frequency below which X% of energy is contained.
pub fn spectral_rolloff(magnitudes: &[f64], sample_rate: f64, percent: f64) -> f64 {
    let total: f64 = magnitudes.iter().map(|&m| m * m).sum();
    let threshold = total * percent;
    let n = magnitudes.len();
    let mut cumsum = 0.0;
    for (i, &m) in magnitudes.iter().enumerate() {
        cumsum += m * m;
        if cumsum >= threshold {
            return i as f64 * sample_rate / (2.0 * n as f64);
        }
    }
    sample_rate / 2.0
}

/// Spectral flatness: Wiener entropy. 1 = white noise, 0 = tonal.
pub fn spectral_flatness(magnitudes: &[f64]) -> f64 {
    let n = magnitudes.len() as f64;
    if n < 1.0 { return 0.0; }
    let log_mean = magnitudes.iter().map(|&m| m.max(1e-15).ln()).sum::<f64>() / n;
    let arith_mean = magnitudes.iter().sum::<f64>() / n;
    if arith_mean < 1e-15 { return 0.0; }
    log_mean.exp() / arith_mean
}

// ═══════════════════════════════════════════════════════════════════
// MEL SCALE
// ═══════════════════════════════════════════════════════════════════

/// Convert frequency in Hz to Mel scale.
pub fn hz_to_mel(hz: f64) -> f64 { 2595.0 * (1.0 + hz / 700.0).log10() }

/// Convert Mel scale to Hz.
pub fn mel_to_hz(mel: f64) -> f64 { 700.0 * (10.0f64.powf(mel / 2595.0) - 1.0) }

/// Generate n_filters mel-spaced filter bank center frequencies.
pub fn mel_filter_centers(n_filters: usize, fmin: f64, fmax: f64) -> Vec<f64> {
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    (0..n_filters + 2).map(|i| {
        mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_filters + 1) as f64)
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// REAL CEPSTRUM
// ═══════════════════════════════════════════════════════════════════

/// Real cepstrum: IFFT(log(|FFT(x)|)).
pub fn rceps(signal: &[f64]) -> Vec<f64> {
    let spectrum = crate::math_runtime::fft(signal);
    let log_mag: Vec<f64> = spectrum.iter().map(|&(r, i)| (r * r + i * i).sqrt().max(1e-15).ln()).collect();
    // IFFT of log magnitude
    let n = log_mag.len();
    let mut re = log_mag;
    let mut im = vec![0.0; n];
    crate::math_runtime::fft_inplace_pub(&mut re, &mut im, true);
    let scale = 1.0 / n as f64;
    re.iter().map(|&r| r * scale).collect()
}

// ═══════════════════════════════════════════════════════════════════
// WINDOW FUNCTIONS (extended)
// ═══════════════════════════════════════════════════════════════════

/// Kaiser window.
pub fn kaiser_window(n: usize, beta: f64) -> Vec<f64> {
    let denom = crate::special_functions::bessel_i0(beta);
    (0..n).map(|i| {
        let t = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
        crate::special_functions::bessel_i0(beta * (1.0 - t * t).sqrt()) / denom
    }).collect()
}

/// Bartlett (triangular) window.
pub fn bartlett_window(n: usize) -> Vec<f64> {
    (0..n).map(|i| {
        1.0 - (2.0 * i as f64 / (n - 1) as f64 - 1.0).abs()
    }).collect()
}

/// Flat-top window (maximum amplitude accuracy).
pub fn flattop_window(n: usize) -> Vec<f64> {
    (0..n).map(|i| {
        let t = 2.0 * PI * i as f64 / (n - 1) as f64;
        0.21557895 - 0.41663158 * t.cos() + 0.277263158 * (2.0 * t).cos()
            - 0.083578947 * (3.0 * t).cos() + 0.006947368 * (4.0 * t).cos()
    }).collect()
}

/// Tukey (tapered cosine) window.
pub fn tukey_window(n: usize, alpha: f64) -> Vec<f64> {
    (0..n).map(|i| {
        let x = i as f64 / (n - 1) as f64;
        if x < alpha / 2.0 {
            0.5 * (1.0 + (2.0 * PI * x / alpha - PI).cos())
        } else if x > 1.0 - alpha / 2.0 {
            0.5 * (1.0 + (2.0 * PI * (x - 1.0) / alpha + PI).cos())
        } else {
            1.0
        }
    }).collect()
}

/// Gaussian window.
pub fn gaussian_window(n: usize, sigma: f64) -> Vec<f64> {
    let center = (n - 1) as f64 / 2.0;
    (0..n).map(|i| {
        let t = (i as f64 - center) / (sigma * center);
        (-0.5 * t * t).exp()
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// MORLET WAVELET
// ═══════════════════════════════════════════════════════════════════

/// Morlet wavelet at given width and points.
pub fn morlet_wavelet(n: usize, w: f64) -> Vec<(f64, f64)> {
    let center = (n - 1) as f64 / 2.0;
    (0..n).map(|i| {
        let t = (i as f64 - center) / (n as f64 / 8.0);
        let envelope = (-t * t / 2.0).exp();
        let re = envelope * (w * t).cos();
        let im = envelope * (w * t).sin();
        (re, im)
    }).collect()
}

/// Continuous wavelet transform (CWT) with Ricker wavelet.
pub fn cwt_ricker(signal: &[f64], widths: &[f64]) -> Vec<Vec<f64>> {
    widths.iter().map(|&width| {
        let wlen = (10.0 * width).ceil() as usize;
        let half = wlen / 2;
        let wavelet: Vec<f64> = (0..wlen).map(|i| {
            let t = (i as f64 - half as f64) / width;
            let t2 = t * t;
            (2.0 / (3.0 * width * PI.powf(0.25))) * (1.0 - t2) * (-t2 / 2.0).exp()
        }).collect();
        // Convolve signal with wavelet
        crate::math_runtime::convolve(signal, &wavelet)
            .into_iter()
            .skip(half)
            .take(signal.len())
            .collect()
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_peaks() {
        let data = vec![0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0];
        let peaks = find_peaks(&data, None, None);
        assert_eq!(peaks, vec![1, 3, 5]);
    }

    #[test]
    fn test_find_peaks_min_height() {
        let data = vec![0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0];
        let peaks = find_peaks(&data, Some(2.5), None);
        assert_eq!(peaks, vec![3]); // only the 3.0 peak
    }

    #[test]
    fn test_zero_crossing_rate() {
        let signal = vec![1.0, -1.0, 1.0, -1.0]; // crosses every sample
        let zcr = zero_crossing_rate(&signal);
        assert!((zcr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_centroid() {
        // Equal magnitudes = centroid at center frequency
        let mags = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let sc = spectral_centroid(&mags, 1000.0);
        assert!(sc > 0.0);
    }

    #[test]
    fn test_mel_scale() {
        assert!((hz_to_mel(0.0)).abs() < 1e-10);
        assert!((mel_to_hz(0.0)).abs() < 1e-10);
        // Roundtrip
        let hz = 1000.0;
        assert!((mel_to_hz(hz_to_mel(hz)) - hz).abs() < 1e-6);
    }

    #[test]
    fn test_kaiser_window() {
        let w = kaiser_window(64, 5.0);
        assert_eq!(w.len(), 64);
        // Should be symmetric
        for i in 0..32 {
            assert!((w[i] - w[63 - i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_tukey_window() {
        let w = tukey_window(64, 0.5);
        assert_eq!(w.len(), 64);
        assert!((w[32] - 1.0).abs() < 1e-10); // center should be 1.0
    }

    #[test]
    fn test_cwt_ricker() {
        let signal: Vec<f64> = (0..100).map(|i| (2.0 * PI * i as f64 / 20.0).sin()).collect();
        let result = cwt_ricker(&signal, &[5.0, 10.0]);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), signal.len());
    }
}
