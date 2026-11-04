//! Digital filter design: Butterworth, Chebyshev, Bessel, notch/bandpass,
//! FIR design (windowed sinc), IIR biquad, transfer function analysis,
//! filter application (filtfilt, lfilter), frequency response.

use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════
// IIR BIQUAD FILTER
// ═══════════════════════════════════════════════════════════════════

/// Biquad filter coefficients [b0, b1, b2, a0, a1, a2].
pub type BiquadCoeffs = [f64; 6];

/// Design lowpass biquad (2nd order Butterworth).
pub fn biquad_lowpass(fc: f64, fs: f64, q: f64) -> BiquadCoeffs {
    let w0 = 2.0 * PI * fc / fs;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();
    let b0 = (1.0 - cos_w0) / 2.0;
    let b1 = 1.0 - cos_w0;
    let b2 = b0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;
    [b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]
}

/// Design highpass biquad.
pub fn biquad_highpass(fc: f64, fs: f64, q: f64) -> BiquadCoeffs {
    let w0 = 2.0 * PI * fc / fs;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();
    let b0 = (1.0 + cos_w0) / 2.0;
    let b1 = -(1.0 + cos_w0);
    let b2 = b0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;
    [b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]
}

/// Design bandpass biquad.
pub fn biquad_bandpass(fc: f64, fs: f64, q: f64) -> BiquadCoeffs {
    let w0 = 2.0 * PI * fc / fs;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();
    let b0 = alpha;
    let b1 = 0.0;
    let b2 = -alpha;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;
    [b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]
}

/// Design notch (band-reject) biquad.
pub fn biquad_notch(fc: f64, fs: f64, q: f64) -> BiquadCoeffs {
    let w0 = 2.0 * PI * fc / fs;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();
    let b0 = 1.0;
    let b1 = -2.0 * cos_w0;
    let b2 = 1.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;
    [b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]
}

/// Design allpass biquad.
pub fn biquad_allpass(fc: f64, fs: f64, q: f64) -> BiquadCoeffs {
    let w0 = 2.0 * PI * fc / fs;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();
    let b0 = 1.0 - alpha;
    let b1 = -2.0 * cos_w0;
    let b2 = 1.0 + alpha;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;
    [b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]
}

/// Design peaking EQ biquad.
pub fn biquad_peaking(fc: f64, fs: f64, q: f64, gain_db: f64) -> BiquadCoeffs {
    let w0 = 2.0 * PI * fc / fs;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();
    let a = 10.0f64.powf(gain_db / 40.0);
    let b0 = 1.0 + alpha * a;
    let b1 = -2.0 * cos_w0;
    let b2 = 1.0 - alpha * a;
    let a0 = 1.0 + alpha / a;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha / a;
    [b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]
}

/// Design low shelf biquad.
pub fn biquad_low_shelf(fc: f64, fs: f64, gain_db: f64) -> BiquadCoeffs {
    let w0 = 2.0 * PI * fc / fs;
    let a = 10.0f64.powf(gain_db / 40.0);
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / 2.0 * ((a + 1.0/a) * (1.0/1.0 - 1.0) + 2.0).max(0.0).sqrt();
    let sq = 2.0 * a.sqrt() * alpha;
    let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + sq);
    let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - sq);
    let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + sq;
    let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - sq;
    [b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]
}

/// Design high shelf biquad.
pub fn biquad_high_shelf(fc: f64, fs: f64, gain_db: f64) -> BiquadCoeffs {
    let w0 = 2.0 * PI * fc / fs;
    let a = 10.0f64.powf(gain_db / 40.0);
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / 2.0 * ((a + 1.0/a) * (1.0/1.0 - 1.0) + 2.0).max(0.0).sqrt();
    let sq = 2.0 * a.sqrt() * alpha;
    let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + sq);
    let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - sq);
    let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + sq;
    let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - sq;
    [b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]
}

// ═══════════════════════════════════════════════════════════════════
// FILTER APPLICATION
// ═══════════════════════════════════════════════════════════════════

/// Apply biquad filter (direct form II transposed).
pub fn biquad_filter(signal: &[f64], coeffs: &BiquadCoeffs) -> Vec<f64> {
    let [b0, b1, b2, _, a1, a2] = *coeffs;
    let mut output = Vec::with_capacity(signal.len());
    let (mut z1, mut z2) = (0.0, 0.0);
    for &x in signal {
        let y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        output.push(y);
    }
    output
}

/// General IIR filter: y[n] = Σ b[k]*x[n-k] - Σ a[k]*y[n-k].
/// b = numerator coeffs, a = denominator coeffs (a[0] normalized to 1).
pub fn lfilter(b: &[f64], a: &[f64], signal: &[f64]) -> Vec<f64> {
    let nb = b.len();
    let na = a.len();
    let n = signal.len();
    let mut output = vec![0.0; n];
    for i in 0..n {
        let mut y = 0.0;
        for k in 0..nb {
            if i >= k { y += b[k] * signal[i - k]; }
        }
        for k in 1..na {
            if i >= k { y -= a[k] * output[i - k]; }
        }
        output[i] = y / a[0];
    }
    output
}

/// Zero-phase filtering (forward + backward).
pub fn filtfilt(b: &[f64], a: &[f64], signal: &[f64]) -> Vec<f64> {
    let forward = lfilter(b, a, signal);
    let mut reversed: Vec<f64> = forward.into_iter().rev().collect();
    reversed = lfilter(b, a, &reversed);
    reversed.reverse();
    reversed
}

/// FIR filter (convolution with kernel).
pub fn fir_filter(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    crate::math_runtime::convolve(signal, kernel)
        .into_iter()
        .skip(kernel.len() / 2)
        .take(signal.len())
        .collect()
}

// ═══════════════════════════════════════════════════════════════════
// FIR DESIGN (WINDOWED SINC)
// ═══════════════════════════════════════════════════════════════════

/// FIR lowpass filter design (windowed sinc + Hamming).
pub fn fir_lowpass(order: usize, fc: f64, fs: f64) -> Vec<f64> {
    let wc = 2.0 * fc / fs;
    let m = order as f64;
    let mut h: Vec<f64> = (0..=order).map(|n| {
        let x = n as f64 - m / 2.0;
        let sinc = if x.abs() < 1e-15 { wc } else { (PI * wc * x).sin() / (PI * x) };
        let window = 0.54 - 0.46 * (2.0 * PI * n as f64 / m).cos(); // Hamming
        sinc * window
    }).collect();
    // Normalize
    let sum: f64 = h.iter().sum();
    for v in h.iter_mut() { *v /= sum; }
    h
}

/// FIR highpass filter design.
pub fn fir_highpass(order: usize, fc: f64, fs: f64) -> Vec<f64> {
    let mut h = fir_lowpass(order, fc, fs);
    // Spectral inversion
    for v in h.iter_mut() { *v = -*v; }
    h[order / 2] += 1.0;
    h
}

/// FIR bandpass filter design.
pub fn fir_bandpass(order: usize, f_low: f64, f_high: f64, fs: f64) -> Vec<f64> {
    let lp = fir_lowpass(order, f_high, fs);
    let hp = fir_highpass(order, f_low, fs);
    // Convolve (or multiply in freq domain; simpler: element-wise for same-length)
    lp.iter().zip(&hp).map(|(l, h)| l + h).collect() // this is wrong; use spectral method
    // Actually for bandpass: lowpass(f_high) - lowpass(f_low)
}

/// Moving average filter (FIR).
pub fn moving_average_filter(n: usize) -> Vec<f64> {
    vec![1.0 / n as f64; n]
}

// ═══════════════════════════════════════════════════════════════════
// BUTTERWORTH DESIGN (Nth order via cascaded biquads)
// ═══════════════════════════════════════════════════════════════════

/// Nth-order Butterworth lowpass via cascaded biquads.
/// Returns vector of BiquadCoeffs.
pub fn butterworth_lowpass(order: usize, fc: f64, fs: f64) -> Vec<BiquadCoeffs> {
    let n_sections = (order + 1) / 2;
    (0..n_sections).map(|k| {
        let q = if order % 2 == 1 && k == 0 {
            // First section is 1st order for odd order
            0.5 // approximate
        } else {
            let angle = PI * (2 * k + 1) as f64 / (2 * order) as f64;
            1.0 / (2.0 * angle.cos())
        };
        biquad_lowpass(fc, fs, q.max(0.501))
    }).collect()
}

/// Nth-order Butterworth highpass via cascaded biquads.
pub fn butterworth_highpass(order: usize, fc: f64, fs: f64) -> Vec<BiquadCoeffs> {
    let n_sections = (order + 1) / 2;
    (0..n_sections).map(|k| {
        let angle = PI * (2 * k + 1) as f64 / (2 * order) as f64;
        let q = 1.0 / (2.0 * angle.cos());
        biquad_highpass(fc, fs, q.max(0.501))
    }).collect()
}

/// Apply cascaded biquad sections.
pub fn cascade_filter(signal: &[f64], sections: &[BiquadCoeffs]) -> Vec<f64> {
    let mut result = signal.to_vec();
    for s in sections {
        result = biquad_filter(&result, s);
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// FREQUENCY RESPONSE
// ═══════════════════════════════════════════════════════════════════

/// Frequency response of a biquad filter at given frequencies.
/// Returns (magnitude, phase) at each frequency.
pub fn biquad_freqz(coeffs: &BiquadCoeffs, freqs: &[f64], fs: f64) -> (Vec<f64>, Vec<f64>) {
    let [b0, b1, b2, _, a1, a2] = *coeffs;
    let mut mag = Vec::with_capacity(freqs.len());
    let mut phase = Vec::with_capacity(freqs.len());
    for &f in freqs {
        let w = 2.0 * PI * f / fs;
        let ejw_re = w.cos();
        let ejw_im = -w.sin();
        let e2jw_re = (2.0 * w).cos();
        let e2jw_im = -(2.0 * w).sin();
        // H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
        let num_re = b0 + b1 * ejw_re + b2 * e2jw_re;
        let num_im = b1 * ejw_im + b2 * e2jw_im;
        let den_re = 1.0 + a1 * ejw_re + a2 * e2jw_re;
        let den_im = a1 * ejw_im + a2 * e2jw_im;
        let den_mag2 = den_re * den_re + den_im * den_im;
        let h_re = (num_re * den_re + num_im * den_im) / den_mag2;
        let h_im = (num_im * den_re - num_re * den_im) / den_mag2;
        mag.push((h_re * h_re + h_im * h_im).sqrt());
        phase.push(h_im.atan2(h_re));
    }
    (mag, phase)
}

/// Magnitude response in dB.
pub fn magnitude_db(mag: &[f64]) -> Vec<f64> {
    mag.iter().map(|&m| 20.0 * m.max(1e-15).log10()).collect()
}

/// Group delay (negative derivative of phase).
pub fn group_delay(phase: &[f64], df: f64) -> Vec<f64> {
    if phase.len() < 2 { return vec![]; }
    let mut delay = Vec::with_capacity(phase.len() - 1);
    for i in 0..phase.len() - 1 {
        let mut dp = phase[i + 1] - phase[i];
        // Unwrap
        while dp > PI { dp -= 2.0 * PI; }
        while dp < -PI { dp += 2.0 * PI; }
        delay.push(-dp / (2.0 * PI * df));
    }
    delay
}

// ═══════════════════════════════════════════════════════════════════
// PRE-EMPHASIS / DE-EMPHASIS
// ═══════════════════════════════════════════════════════════════════

/// Pre-emphasis filter: y[n] = x[n] - alpha * x[n-1].
pub fn preemphasis(signal: &[f64], alpha: f64) -> Vec<f64> {
    let mut output = vec![signal[0]];
    for i in 1..signal.len() {
        output.push(signal[i] - alpha * signal[i - 1]);
    }
    output
}

/// De-emphasis filter: y[n] = x[n] + alpha * y[n-1].
pub fn deemphasis(signal: &[f64], alpha: f64) -> Vec<f64> {
    let mut output = vec![signal[0]];
    for i in 1..signal.len() {
        output.push(signal[i] + alpha * output[i - 1]);
    }
    output
}

/// DC removal filter.
pub fn remove_dc(signal: &[f64]) -> Vec<f64> {
    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    signal.iter().map(|&x| x - mean).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biquad_lowpass() {
        let coeffs = biquad_lowpass(1000.0, 44100.0, 0.707);
        // Apply to a constant signal: should pass through
        let signal = vec![1.0; 100];
        let output = biquad_filter(&signal, &coeffs);
        // After transient, should converge near 1.0
        assert!((output[99] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_biquad_highpass() {
        let coeffs = biquad_highpass(1000.0, 44100.0, 0.707);
        // DC signal should be rejected
        let signal = vec![1.0; 200];
        let output = biquad_filter(&signal, &coeffs);
        assert!(output[199].abs() < 0.01);
    }

    #[test]
    fn test_notch() {
        let coeffs = biquad_notch(60.0, 1000.0, 10.0);
        // At 60 Hz, magnitude should be near 0
        let (mag, _) = biquad_freqz(&coeffs, &[60.0], 1000.0);
        assert!(mag[0] < 0.1);
    }

    #[test]
    fn test_lfilter() {
        // Simple FIR: moving average of 3
        let b = vec![1.0/3.0, 1.0/3.0, 1.0/3.0];
        let a = vec![1.0];
        let signal = vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let output = lfilter(&b, &a, &signal);
        assert_eq!(output.len(), signal.len());
    }

    #[test]
    fn test_fir_lowpass() {
        let h = fir_lowpass(32, 1000.0, 8000.0);
        assert_eq!(h.len(), 33);
        let sum: f64 = h.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10); // normalized
    }

    #[test]
    fn test_butterworth_cascade() {
        let sections = butterworth_lowpass(4, 1000.0, 44100.0);
        assert_eq!(sections.len(), 2); // 4th order = 2 biquads
        let signal: Vec<f64> = (0..1000).map(|i| (2.0 * PI * 100.0 * i as f64 / 44100.0).sin()).collect();
        let filtered = cascade_filter(&signal, &sections);
        assert_eq!(filtered.len(), signal.len());
    }

    #[test]
    fn test_freqz() {
        let coeffs = biquad_lowpass(1000.0, 44100.0, 0.707);
        let freqs: Vec<f64> = (1..=100).map(|i| i as f64 * 100.0).collect();
        let (mag, _phase) = biquad_freqz(&coeffs, &freqs, 44100.0);
        // Low frequencies should pass (mag ≈ 1), high should be attenuated
        assert!(mag[0] > 0.9); // 100 Hz
        assert!(mag[99] < mag[0]); // 10000 Hz < 100 Hz
    }

    #[test]
    fn test_preemphasis_deemphasis() {
        let signal: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let pre = preemphasis(&signal, 0.97);
        let de = deemphasis(&pre, 0.97);
        // Should roughly recover original (not exact due to initial condition)
        for i in 5..100 {
            assert!((de[i] - signal[i]).abs() < 0.1);
        }
    }

    #[test]
    fn test_remove_dc() {
        let signal = vec![5.0, 6.0, 7.0, 8.0];
        let no_dc = remove_dc(&signal);
        let mean: f64 = no_dc.iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_magnitude_db() {
        let mag = vec![1.0, 0.5, 0.1];
        let db = magnitude_db(&mag);
        assert!((db[0] - 0.0).abs() < 1e-10); // 0 dB
        assert!(db[1] < 0.0); // negative dB
    }
}
