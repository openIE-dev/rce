//! Audio processing: MFCC, pitch detection, spectral features,
//! onset detection, chromagram, tempo estimation, audio effects.

use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════
// MFCC (Mel-Frequency Cepstral Coefficients)
// ═══════════════════════════════════════════════════════════════════

/// Compute MFCCs from audio signal.
/// Returns n_mfcc coefficients per frame.
pub fn mfcc(signal: &[f64], sample_rate: f64, n_mfcc: usize, n_fft: usize,
            hop_length: usize, n_mels: usize) -> Vec<Vec<f64>> {
    let frames = frame_signal(signal, n_fft, hop_length);
    let mel_fb = mel_filterbank(n_mels, n_fft, sample_rate);

    frames.iter().map(|frame| {
        // Apply Hann window
        let windowed: Vec<f64> = frame.iter().enumerate()
            .map(|(i, &x)| x * 0.5 * (1.0 - (2.0 * PI * i as f64 / (n_fft - 1) as f64).cos()))
            .collect();

        // FFT magnitude
        let spec = power_spectrum(&windowed);

        // Apply mel filterbank
        let mel_energies: Vec<f64> = mel_fb.iter().map(|filter| {
            filter.iter().zip(&spec).map(|(f, s)| f * s).sum::<f64>().max(1e-10).ln()
        }).collect();

        // DCT (type-II) to get MFCCs
        dct2(&mel_energies, n_mfcc)
    }).collect()
}

/// Frame a signal into overlapping windows.
pub fn frame_signal(signal: &[f64], frame_length: usize, hop_length: usize) -> Vec<Vec<f64>> {
    let mut frames = Vec::new();
    let mut start = 0;
    while start + frame_length <= signal.len() {
        frames.push(signal[start..start + frame_length].to_vec());
        start += hop_length;
    }
    frames
}

/// Power spectrum (|FFT|²).
pub fn power_spectrum(frame: &[f64]) -> Vec<f64> {
    let n = frame.len().next_power_of_two();
    let mut re = frame.to_vec();
    re.resize(n, 0.0);
    let mut im = vec![0.0; n];
    crate::math_runtime::fft_inplace_pub(&mut re, &mut im, false);
    let half = n / 2 + 1;
    (0..half).map(|i| re[i] * re[i] + im[i] * im[i]).collect()
}

/// Mel filterbank matrix (n_mels × (n_fft/2+1)).
pub fn mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: f64) -> Vec<Vec<f64>> {
    let n_bins = n_fft / 2 + 1;
    let fmin = 0.0;
    let fmax = sample_rate / 2.0;
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    let mel_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();

    let bin_points: Vec<usize> = mel_points.iter()
        .map(|&f| ((f / sample_rate * n_fft as f64) as usize).min(n_bins - 1))
        .collect();

    (0..n_mels).map(|m| {
        let mut filter = vec![0.0; n_bins];
        let start = bin_points[m];
        let center = bin_points[m + 1];
        let end = bin_points[m + 2];
        for k in start..center {
            if center > start {
                filter[k] = (k - start) as f64 / (center - start) as f64;
            }
        }
        for k in center..=end {
            if end > center {
                filter[k] = (end - k) as f64 / (end - center) as f64;
            }
        }
        filter
    }).collect()
}

fn hz_to_mel(hz: f64) -> f64 { 2595.0 * (1.0 + hz / 700.0).log10() }
fn mel_to_hz(mel: f64) -> f64 { 700.0 * (10.0f64.powf(mel / 2595.0) - 1.0) }

/// DCT type-II.
fn dct2(x: &[f64], n_out: usize) -> Vec<f64> {
    let n = x.len();
    (0..n_out).map(|k| {
        x.iter().enumerate().map(|(i, &xi)| {
            xi * (PI * k as f64 * (2.0 * i as f64 + 1.0) / (2.0 * n as f64)).cos()
        }).sum()
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// PITCH DETECTION
// ═══════════════════════════════════════════════════════════════════

/// Autocorrelation-based pitch detection.
/// Returns estimated fundamental frequency in Hz (or 0 if unvoiced).
pub fn pitch_autocorrelation(frame: &[f64], sample_rate: f64, min_freq: f64, max_freq: f64) -> f64 {
    let n = frame.len();
    let min_lag = (sample_rate / max_freq) as usize;
    let max_lag = (sample_rate / min_freq).min(n as f64) as usize;

    let mut best_lag = 0;
    let mut best_corr = 0.0;

    for lag in min_lag..max_lag.min(n) {
        let mut corr = 0.0;
        for i in 0..n - lag {
            corr += frame[i] * frame[i + lag];
        }
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    // Threshold for voicing
    let energy: f64 = frame.iter().map(|x| x * x).sum();
    if best_lag == 0 || best_corr < 0.3 * energy { return 0.0; }

    sample_rate / best_lag as f64
}

/// YIN pitch detection (simplified).
pub fn pitch_yin(frame: &[f64], sample_rate: f64, threshold: f64) -> f64 {
    let n = frame.len();
    let half = n / 2;
    let mut d = vec![0.0; half];

    // Difference function
    for tau in 1..half {
        for j in 0..half {
            let diff = frame[j] - frame[j + tau];
            d[tau] += diff * diff;
        }
    }

    // Cumulative mean normalized difference
    let mut dn = vec![1.0; half];
    let mut running = 0.0;
    for tau in 1..half {
        running += d[tau];
        dn[tau] = if running > 1e-15 { d[tau] * tau as f64 / running } else { 1.0 };
    }

    // Find first dip below threshold
    for tau in 2..half {
        if dn[tau] < threshold {
            // Parabolic interpolation
            if tau > 0 && tau < half - 1 {
                let a = dn[tau - 1];
                let b = dn[tau];
                let c = dn[tau + 1];
                let delta = 0.5 * (a - c) / (a - 2.0 * b + c);
                return sample_rate / (tau as f64 + delta);
            }
            return sample_rate / tau as f64;
        }
    }
    0.0 // unvoiced
}

// ═══════════════════════════════════════════════════════════════════
// SPECTRAL FEATURES (per-frame)
// ═══════════════════════════════════════════════════════════════════

/// Spectral centroid of a magnitude spectrum.
pub fn spectral_centroid(magnitudes: &[f64], sample_rate: f64) -> f64 {
    crate::signal_extended::spectral_centroid(magnitudes, sample_rate)
}

/// Spectral flux: sum of positive differences between consecutive spectra.
pub fn spectral_flux(prev_spec: &[f64], curr_spec: &[f64]) -> f64 {
    prev_spec.iter().zip(curr_spec)
        .map(|(p, c)| (c - p).max(0.0))
        .sum()
}

/// Spectral contrast: difference between peaks and valleys in frequency bands.
pub fn spectral_contrast(magnitudes: &[f64], n_bands: usize) -> Vec<f64> {
    let n = magnitudes.len();
    let band_size = n / n_bands;
    (0..n_bands).map(|b| {
        let start = b * band_size;
        let end = ((b + 1) * band_size).min(n);
        let band = &magnitudes[start..end];
        let mut sorted = band.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let alpha = 0.2;
        let n_alpha = (sorted.len() as f64 * alpha) as usize;
        let peak = sorted[sorted.len() - 1 - n_alpha..].iter().sum::<f64>() / (n_alpha + 1) as f64;
        let valley = sorted[..n_alpha + 1].iter().sum::<f64>() / (n_alpha + 1) as f64;
        peak.max(1e-15).ln() - valley.max(1e-15).ln()
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// ONSET DETECTION
// ═══════════════════════════════════════════════════════════════════

/// Energy-based onset detection. Returns onset frame indices.
pub fn onset_detect_energy(signal: &[f64], frame_length: usize, hop_length: usize, threshold: f64) -> Vec<usize> {
    let frames = frame_signal(signal, frame_length, hop_length);
    let energies: Vec<f64> = frames.iter()
        .map(|f| f.iter().map(|x| x * x).sum::<f64>() / f.len() as f64)
        .collect();

    let mut onsets = Vec::new();
    for i in 1..energies.len() {
        let diff = energies[i] - energies[i - 1];
        if diff > threshold { onsets.push(i); }
    }
    onsets
}

/// Spectral flux onset detection.
pub fn onset_detect_flux(signal: &[f64], n_fft: usize, hop_length: usize, threshold: f64) -> Vec<usize> {
    let frames = frame_signal(signal, n_fft, hop_length);
    let spectra: Vec<Vec<f64>> = frames.iter().map(|f| power_spectrum(f)).collect();

    let mut onsets = Vec::new();
    for i in 1..spectra.len() {
        let flux = spectral_flux(&spectra[i - 1], &spectra[i]);
        if flux > threshold { onsets.push(i); }
    }
    onsets
}

// ═══════════════════════════════════════════════════════════════════
// CHROMAGRAM
// ═══════════════════════════════════════════════════════════════════

/// Chromagram: 12-dimensional pitch class profile.
pub fn chromagram(signal: &[f64], sample_rate: f64, n_fft: usize, hop_length: usize) -> Vec<[f64; 12]> {
    let frames = frame_signal(signal, n_fft, hop_length);
    frames.iter().map(|frame| {
        let spec = power_spectrum(frame);
        let mut chroma = [0.0; 12];
        let freq_per_bin = sample_rate / n_fft as f64;
        for (i, &mag) in spec.iter().enumerate().skip(1) {
            let freq = i as f64 * freq_per_bin;
            if freq > 0.0 && freq < sample_rate / 2.0 {
                let midi = 12.0 * (freq / 440.0).log2() + 69.0;
                let pitch_class = ((midi.round() as i64 % 12 + 12) % 12) as usize;
                chroma[pitch_class] += mag;
            }
        }
        // Normalize
        let max = chroma.iter().copied().fold(0.0, f64::max);
        if max > 1e-15 { for c in chroma.iter_mut() { *c /= max; } }
        chroma
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// TEMPO ESTIMATION
// ═══════════════════════════════════════════════════════════════════

/// Estimate tempo (BPM) from onset strength envelope.
pub fn estimate_tempo(signal: &[f64], sample_rate: f64, n_fft: usize, hop_length: usize) -> f64 {
    let frames = frame_signal(signal, n_fft, hop_length);
    let energies: Vec<f64> = frames.iter()
        .map(|f| f.iter().map(|x| x * x).sum())
        .collect();

    // Onset strength: positive first difference
    let onset: Vec<f64> = energies.windows(2)
        .map(|w| (w[1] - w[0]).max(0.0))
        .collect();

    if onset.len() < 2 { return 120.0; } // default

    // Autocorrelation of onset envelope
    let n = onset.len();
    let mean = onset.iter().sum::<f64>() / n as f64;
    let frame_rate = sample_rate / hop_length as f64;
    let min_bpm = 60.0;
    let max_bpm = 240.0;
    let min_lag = (frame_rate * 60.0 / max_bpm) as usize;
    let max_lag = (frame_rate * 60.0 / min_bpm).min(n as f64) as usize;

    let mut best_lag = min_lag;
    let mut best_corr = 0.0;
    for lag in min_lag..max_lag.min(n) {
        let corr: f64 = (0..n - lag).map(|i| (onset[i] - mean) * (onset[i + lag] - mean)).sum();
        if corr > best_corr { best_corr = corr; best_lag = lag; }
    }

    if best_lag > 0 { frame_rate * 60.0 / best_lag as f64 } else { 120.0 }
}

// ═══════════════════════════════════════════════════════════════════
// AUDIO EFFECTS
// ═══════════════════════════════════════════════════════════════════

/// Apply gain (volume change in dB).
pub fn apply_gain(signal: &[f64], gain_db: f64) -> Vec<f64> {
    let factor = 10.0f64.powf(gain_db / 20.0);
    signal.iter().map(|&s| s * factor).collect()
}

/// Normalize audio to [-1, 1].
pub fn normalize_audio(signal: &[f64]) -> Vec<f64> {
    let max = signal.iter().map(|s| s.abs()).fold(0.0, f64::max);
    if max < 1e-15 { return signal.to_vec(); }
    signal.iter().map(|&s| s / max).collect()
}

/// Simple delay/echo effect.
pub fn delay_effect(signal: &[f64], delay_samples: usize, decay: f64) -> Vec<f64> {
    let mut output = signal.to_vec();
    for i in delay_samples..output.len() {
        output[i] += decay * output[i - delay_samples];
    }
    output
}

/// Fade in.
pub fn fade_in(signal: &[f64], n_samples: usize) -> Vec<f64> {
    let mut output = signal.to_vec();
    for i in 0..n_samples.min(output.len()) {
        output[i] *= i as f64 / n_samples as f64;
    }
    output
}

/// Fade out.
pub fn fade_out(signal: &[f64], n_samples: usize) -> Vec<f64> {
    let mut output = signal.to_vec();
    let n = output.len();
    for i in 0..n_samples.min(n) {
        output[n - 1 - i] *= i as f64 / n_samples as f64;
    }
    output
}

/// Mix two signals.
pub fn mix(a: &[f64], b: &[f64], ratio: f64) -> Vec<f64> {
    let n = a.len().max(b.len());
    (0..n).map(|i| {
        let va = if i < a.len() { a[i] } else { 0.0 };
        let vb = if i < b.len() { b[i] } else { 0.0 };
        ratio * va + (1.0 - ratio) * vb
    }).collect()
}

/// RMS (root mean square) level.
pub fn rms_level(signal: &[f64]) -> f64 {
    (signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64).sqrt()
}

/// RMS level in dB.
pub fn rms_db(signal: &[f64]) -> f64 {
    let rms = rms_level(signal);
    if rms < 1e-15 { return -120.0; }
    20.0 * rms.log10()
}

/// Clipping: limit signal to [-threshold, threshold].
pub fn clip(signal: &[f64], threshold: f64) -> Vec<f64> {
    signal.iter().map(|&s| s.max(-threshold).min(threshold)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_signal(freq: f64, sample_rate: f64, duration: f64) -> Vec<f64> {
        let n = (sample_rate * duration) as usize;
        (0..n).map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin()).collect()
    }

    #[test]
    fn test_frame_signal() {
        let signal = vec![0.0; 100];
        let frames = frame_signal(&signal, 20, 10);
        assert_eq!(frames.len(), 9); // (100 - 20) / 10 + 1
    }

    #[test]
    fn test_power_spectrum() {
        let signal = sine_signal(440.0, 8000.0, 0.01);
        let spec = power_spectrum(&signal);
        assert!(spec.len() > 0);
    }

    #[test]
    fn test_mfcc() {
        let signal = sine_signal(440.0, 16000.0, 0.5);
        let coeffs = mfcc(&signal, 16000.0, 13, 512, 256, 26);
        assert!(coeffs.len() > 0);
        assert_eq!(coeffs[0].len(), 13);
    }

    #[test]
    fn test_pitch_autocorrelation() {
        let signal = sine_signal(440.0, 16000.0, 0.1);
        let f0 = pitch_autocorrelation(&signal, 16000.0, 80.0, 1000.0);
        assert!((f0 - 440.0).abs() < 20.0); // within 20 Hz
    }

    #[test]
    fn test_chromagram() {
        let signal = sine_signal(440.0, 16000.0, 0.5);
        let chroma = chromagram(&signal, 16000.0, 1024, 512);
        assert!(chroma.len() > 0);
        // A = 440 Hz → pitch class 9
        // Chroma bin 9 (A) should be dominant in at least some frames
    }

    #[test]
    fn test_onset_energy() {
        // Impulse at sample 1000
        let mut signal = vec![0.0; 4000];
        for i in 1000..1100 { signal[i] = 1.0; }
        let onsets = onset_detect_energy(&signal, 256, 128, 0.01);
        assert!(onsets.len() >= 1);
    }

    #[test]
    fn test_normalize_audio() {
        let signal = vec![0.5, -0.3, 0.8, -0.2];
        let norm = normalize_audio(&signal);
        assert!((norm.iter().map(|s| s.abs()).fold(0.0, f64::max) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_gain() {
        let signal = vec![1.0; 10];
        let gained = apply_gain(&signal, 6.0); // ~double amplitude
        assert!((gained[0] - 1.995).abs() < 0.01);
    }

    #[test]
    fn test_rms() {
        let signal = vec![1.0; 100];
        assert!((rms_level(&signal) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip() {
        let signal = vec![2.0, -3.0, 0.5];
        let clipped = clip(&signal, 1.0);
        assert_eq!(clipped, vec![1.0, -1.0, 0.5]);
    }

    #[test]
    fn test_fade() {
        let signal = vec![1.0; 100];
        let faded = fade_in(&signal, 50);
        assert_eq!(faded[0], 0.0);
        assert_eq!(faded[50], 1.0);
    }
}
