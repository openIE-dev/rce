//! Advanced statistics: NMF, survival analysis, copulas, power analysis,
//! control systems, and extended data operations.

use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════
// NON-NEGATIVE MATRIX FACTORIZATION (NMF)
// ═══════════════════════════════════════════════════════════════════

/// NMF: V ≈ W * H where V, W, H are all non-negative.
/// Returns (W, H) where W is n×k and H is k×m.
pub fn nmf(v: &[Vec<f64>], k: usize, max_iter: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = v.len();
    let m = if n > 0 { v[0].len() } else { return (vec![], vec![]); };
    let mut rng = crate::distributions::Rng::new(42);

    let mut w: Vec<Vec<f64>> = (0..n).map(|_| (0..k).map(|_| rng.uniform() + 0.1).collect()).collect();
    let mut h: Vec<Vec<f64>> = (0..k).map(|_| (0..m).map(|_| rng.uniform() + 0.1).collect()).collect();

    for _ in 0..max_iter {
        // Update H: H *= (W'V) / (W'WH)
        let wt_v = mat_mul_t_a(&w, v);
        let wt_w = mat_mul_t_a(&w, &w);
        let wt_wh = mat_mul_ab(&wt_w, &h);
        for i in 0..k {
            for j in 0..m {
                h[i][j] *= wt_v[i][j] / (wt_wh[i][j] + 1e-15);
            }
        }

        // Update W: W *= (VH') / (WHH')
        let vh_t = mat_mul_a_bt(v, &h);
        let wh = mat_mul_ab(&w, &h);
        let whh_t = mat_mul_a_bt(&wh, &h);
        for i in 0..n {
            for j in 0..k {
                w[i][j] *= vh_t[i][j] / (whh_t[i][j] + 1e-15);
            }
        }
    }
    (w, h)
}

fn mat_mul_ab(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    crate::math_runtime::mat_mul(a, b)
}
fn mat_mul_t_a(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let at = crate::math_runtime::transpose(a);
    crate::math_runtime::mat_mul(&at, b)
}
fn mat_mul_a_bt(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let bt = crate::math_runtime::transpose(b);
    crate::math_runtime::mat_mul(a, &bt)
}

// ═══════════════════════════════════════════════════════════════════
// MULTIDIMENSIONAL SCALING (MDS)
// ═══════════════════════════════════════════════════════════════════

/// Classical MDS: embed distance matrix into n_components dimensions.
pub fn mds(dist_matrix: &[Vec<f64>], n_components: usize) -> Vec<Vec<f64>> {
    let n = dist_matrix.len();
    if n == 0 { return vec![]; }

    // Double centering: B = -0.5 * J * D^2 * J where J = I - (1/n)*11'
    let mut b = vec![vec![0.0; n]; n];
    let mut d2 = vec![vec![0.0; n]; n];
    for i in 0..n { for j in 0..n { d2[i][j] = dist_matrix[i][j].powi(2); } }

    let row_means: Vec<f64> = (0..n).map(|i| d2[i].iter().sum::<f64>() / n as f64).collect();
    let col_means: Vec<f64> = (0..n).map(|j| (0..n).map(|i| d2[i][j]).sum::<f64>() / n as f64).collect();
    let grand_mean: f64 = d2.iter().flat_map(|r| r.iter()).sum::<f64>() / (n * n) as f64;

    for i in 0..n {
        for j in 0..n {
            b[i][j] = -0.5 * (d2[i][j] - row_means[i] - col_means[j] + grand_mean);
        }
    }

    // Eigendecomposition of B
    let _eigs = crate::math_runtime::eigenvalues_symmetric(&b, 200);
    // Use PCA-like extraction
    let (components, _eigenvalues, _) = crate::math_runtime::pca(&b.iter().map(|r| r.clone()).collect::<Vec<_>>(), n_components);

    // Scale by sqrt of eigenvalues
    b.iter().enumerate().map(|(_i, row)| {
        (0..n_components.min(components.len())).map(|c| {
            row.iter().zip(&components[c]).map(|(&x, &v)| x * v).sum::<f64>()
        }).collect()
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// KAPLAN-MEIER SURVIVAL ANALYSIS
// ═══════════════════════════════════════════════════════════════════

/// Kaplan-Meier survival curve.
/// times: event or censoring times. events: true if event occurred, false if censored.
/// Returns (time_points, survival_probabilities).
pub fn kaplan_meier(times: &[f64], events: &[bool]) -> (Vec<f64>, Vec<f64>) {
    let mut data: Vec<(f64, bool)> = times.iter().zip(events).map(|(&t, &e)| (t, e)).collect();
    data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut t_points = vec![0.0];
    let mut surv = vec![1.0];
    let mut n_at_risk = data.len() as f64;
    let mut current_surv = 1.0;

    let mut i = 0;
    while i < data.len() {
        let t = data[i].0;
        let mut d = 0.0; // deaths at time t
        let mut c = 0.0; // censored at time t
        while i < data.len() && (data[i].0 - t).abs() < 1e-15 {
            if data[i].1 { d += 1.0; } else { c += 1.0; }
            i += 1;
        }
        if d > 0.0 {
            current_surv *= 1.0 - d / n_at_risk;
            t_points.push(t);
            surv.push(current_surv);
        }
        n_at_risk -= d + c;
    }
    (t_points, surv)
}

/// Log-rank test comparing two survival curves. Returns (chi2, p_value).
pub fn logrank_test(times1: &[f64], events1: &[bool], times2: &[f64], events2: &[bool]) -> (f64, f64) {
    // Combine all unique event times
    let mut all_times: Vec<f64> = times1.iter().chain(times2.iter()).copied().collect();
    all_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_times.dedup();

    let mut o1 = 0.0; let mut e1 = 0.0;
    let mut n1 = times1.len() as f64;
    let mut n2 = times2.len() as f64;

    for &t in &all_times {
        let d1: f64 = times1.iter().zip(events1).filter(|(ti, ei)| (**ti - t).abs() < 1e-15 && **ei).count() as f64;
        let d2: f64 = times2.iter().zip(events2).filter(|(ti, ei)| (**ti - t).abs() < 1e-15 && **ei).count() as f64;
        let d = d1 + d2;
        let n = n1 + n2;
        if n > 0.0 && d > 0.0 {
            o1 += d1;
            e1 += n1 * d / n;
        }
        let c1: f64 = times1.iter().zip(events1).filter(|(ti, _)| (**ti - t).abs() < 1e-15).count() as f64;
        let c2: f64 = times2.iter().zip(events2).filter(|(ti, _)| (**ti - t).abs() < 1e-15).count() as f64;
        n1 -= c1;
        n2 -= c2;
    }

    let var = e1.max(1e-15); // simplified variance
    let chi2 = (o1 - e1).powi(2) / var;
    let p = 1.0 - crate::distributions::chi2_cdf(chi2, 1.0);
    (chi2, p)
}

// ═══════════════════════════════════════════════════════════════════
// COPULAS
// ═══════════════════════════════════════════════════════════════════

/// Gaussian copula: sample n points in [0,1]^d with given correlation matrix.
pub fn gaussian_copula(rng: &mut crate::distributions::Rng, correlation: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let dim = correlation.len();
    let mean = vec![0.0; dim];
    (0..n).map(|_| {
        let z = crate::distributions::mvnorm_rvs(rng, &mean, correlation);
        z.iter().map(|&zi| crate::distributions::norm_cdf(zi, 0.0, 1.0)).collect()
    }).collect()
}

/// Clayton copula: sample n points in [0,1]^2 with parameter theta > 0.
pub fn clayton_copula(rng: &mut crate::distributions::Rng, theta: f64, n: usize) -> Vec<Vec<f64>> {
    (0..n).map(|_| {
        let u1 = rng.uniform();
        let v = rng.uniform();
        let u2 = (u1.powf(-theta) * (v.powf(-theta / (1.0 + theta)) - 1.0) + 1.0).powf(-1.0 / theta);
        vec![u1, u2.clamp(0.0, 1.0)]
    }).collect()
}

/// Frank copula: sample n points in [0,1]^2 with parameter theta != 0.
pub fn frank_copula(rng: &mut crate::distributions::Rng, theta: f64, n: usize) -> Vec<Vec<f64>> {
    (0..n).map(|_| {
        let u1 = rng.uniform();
        let v = rng.uniform();
        let u2 = -((1.0 + (-theta).exp_m1() / (v.recip() * (-theta * u1).exp_m1() + 1.0)).ln()) / theta;
        vec![u1, u2.clamp(0.0, 1.0)]
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// POWER ANALYSIS
// ═══════════════════════════════════════════════════════════════════

/// Power of a two-sample t-test.
pub fn power_t_test(n: f64, effect_size: f64, alpha: f64) -> f64 {
    let se = (2.0 / n).sqrt();
    let t_crit = crate::distributions::t_ppf(1.0 - alpha / 2.0, 2.0 * n - 2.0);
    let ncp = effect_size / se; // non-centrality parameter
    // Approximate power via normal approximation
    let power = 1.0 - crate::distributions::norm_cdf(t_crit - ncp, 0.0, 1.0)
        + crate::distributions::norm_cdf(-t_crit - ncp, 0.0, 1.0);
    power
}

/// Required sample size for two-sample t-test.
pub fn sample_size_t_test(effect_size: f64, alpha: f64, power: f64) -> f64 {
    // Binary search for n
    let mut lo = 2.0;
    let mut hi = 100000.0;
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if power_t_test(mid, effect_size, alpha) < power {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi.ceil()
}

// ═══════════════════════════════════════════════════════════════════
// CONTROL SYSTEMS (basic)
// ═══════════════════════════════════════════════════════════════════

/// Transfer function representation: H(s) = num(s) / den(s).
#[derive(Debug, Clone)]
pub struct TransferFunction {
    pub num: Vec<f64>, // numerator coefficients [b0, b1, ..., bm]
    pub den: Vec<f64>, // denominator coefficients [a0, a1, ..., an]
}

impl TransferFunction {
    pub fn new(num: Vec<f64>, den: Vec<f64>) -> Self { Self { num, den } }

    /// Evaluate frequency response at angular frequency w.
    /// Returns (magnitude, phase_radians).
    pub fn freqresp(&self, w: f64) -> (f64, f64) {
        let (nr, ni) = eval_poly_jw(&self.num, w);
        let (dr, di) = eval_poly_jw(&self.den, w);
        let denom = dr * dr + di * di;
        if denom < 1e-30 { return (f64::INFINITY, 0.0); }
        let hr = (nr * dr + ni * di) / denom;
        let hi = (ni * dr - nr * di) / denom;
        let mag = (hr * hr + hi * hi).sqrt();
        let phase = hi.atan2(hr);
        (mag, phase)
    }

    /// Bode plot data: (frequencies, magnitudes_db, phases_deg).
    pub fn bode(&self, w_start: f64, w_end: f64, n_points: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ws: Vec<f64> = (0..n_points).map(|i| {
            let t = i as f64 / (n_points - 1) as f64;
            (w_start.ln() + t * (w_end.ln() - w_start.ln())).exp()
        }).collect();
        let mut mags = Vec::new();
        let mut phases = Vec::new();
        for &w in &ws {
            let (mag, phase) = self.freqresp(w);
            mags.push(20.0 * mag.max(1e-30).log10());
            phases.push(phase * 180.0 / PI);
        }
        (ws, mags, phases)
    }

    /// DC gain (frequency response at w=0).
    pub fn dcgain(&self) -> f64 {
        let num_dc = *self.num.last().unwrap_or(&0.0);
        let den_dc = *self.den.last().unwrap_or(&1.0);
        if den_dc.abs() < 1e-15 { f64::INFINITY } else { num_dc / den_dc }
    }

    /// Poles of the transfer function (roots of denominator).
    pub fn poles(&self) -> Vec<(f64, f64)> {
        poly_roots_companion(&self.den)
    }

    /// Zeros of the transfer function (roots of numerator).
    pub fn zeros(&self) -> Vec<(f64, f64)> {
        poly_roots_companion(&self.num)
    }

    /// Step response at time points.
    pub fn step_response(&self, t: &[f64]) -> Vec<f64> {
        // Simulate via ODE: x' = A*x + B*u, y = C*x + D*u with u(t)=1
        // For simple TF, use numerical simulation
        let dt = if t.len() > 1 { t[1] - t[0] } else { 0.01 };
        let n = self.den.len() - 1; // system order
        if n == 0 { return vec![self.dcgain(); t.len()]; }

        // State space via controllable canonical form
        let a0 = self.den[0];
        let mut state = vec![0.0; n];
        let mut output = Vec::new();

        for &_ti in t {
            // Output: weighted sum of state + direct feedthrough
            let y: f64 = (0..n.min(self.num.len())).map(|i| {
                self.num.get(self.num.len().saturating_sub(n) + i).unwrap_or(&0.0) * state[i]
            }).sum::<f64>();
            output.push(y / a0);

            // State update (Euler): companion form
            let mut new_state = vec![0.0; n];
            if n > 1 {
                for i in 0..n - 1 { new_state[i] = state[i] + dt * state[i + 1]; }
            }
            let input = 1.0; // step input
            new_state[n - 1] = state[n - 1] + dt * (-state.iter().zip(self.den.iter().skip(1)).map(|(&s, &a)| a * s / a0).sum::<f64>() + input);
            state = new_state;
        }
        output
    }
}

/// Evaluate polynomial at s = jw (purely imaginary).
/// Returns (real_part, imag_part).
fn eval_poly_jw(coeffs: &[f64], w: f64) -> (f64, f64) {
    // coeffs = [c_n, c_{n-1}, ..., c_0] (highest degree first)
    // s = jw: (jw)^k = w^k * j^k
    let n = coeffs.len();
    let mut re = 0.0;
    let mut im = 0.0;
    for (i, &c) in coeffs.iter().enumerate() {
        let power = n - 1 - i;
        let w_pow = w.powi(power as i32);
        match power % 4 {
            0 => re += c * w_pow,
            1 => im += c * w_pow,
            2 => re -= c * w_pow,
            3 => im -= c * w_pow,
            _ => {}
        }
    }
    (re, im)
}

/// Find roots of polynomial via companion matrix eigenvalues (approximate).
fn poly_roots_companion(coeffs: &[f64]) -> Vec<(f64, f64)> {
    let n = coeffs.len();
    if n <= 1 { return vec![]; }
    // Normalize by leading coefficient
    let a0 = coeffs[0];
    if a0.abs() < 1e-15 { return vec![]; }
    let norm: Vec<f64> = coeffs.iter().map(|&c| c / a0).collect();
    let deg = n - 1;
    // Companion matrix
    let mut comp = vec![vec![0.0; deg]; deg];
    for i in 0..deg - 1 { comp[i + 1][i] = 1.0; }
    for i in 0..deg { comp[0][i] = -norm[i + 1]; }
    // Eigenvalues (real parts only for now)
    let eigs = crate::math_runtime::eigenvalues_symmetric(&comp, 200);
    eigs.iter().map(|&e| (e, 0.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nmf() {
        let v = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]];
        let (w, h) = nmf(&v, 2, 100);
        assert_eq!(w.len(), 3);
        assert_eq!(h.len(), 2);
        // W and H should be non-negative
        assert!(w.iter().all(|r| r.iter().all(|&x| x >= 0.0)));
        assert!(h.iter().all(|r| r.iter().all(|&x| x >= 0.0)));
    }

    #[test]
    fn test_kaplan_meier() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![true, false, true, true, false];
        let (t, s) = kaplan_meier(&times, &events);
        assert!(!t.is_empty());
        assert_eq!(s[0], 1.0); // survival starts at 1
        assert!(*s.last().unwrap() < 1.0); // decreases
        assert!(s.windows(2).all(|w| w[0] >= w[1])); // monotone non-increasing
    }

    #[test]
    fn test_power_t_test() {
        let power = power_t_test(30.0, 0.5, 0.05);
        assert!(power > 0.3 && power < 1.0, "power should be reasonable, got {power}");
    }

    #[test]
    fn test_sample_size() {
        let n = sample_size_t_test(0.5, 0.05, 0.8);
        assert!(n > 10.0 && n < 200.0, "sample size should be reasonable, got {n}");
    }

    #[test]
    fn test_transfer_function_dc() {
        // H(s) = 2 / (s + 1) → DC gain = 2/1 = 2
        let tf = TransferFunction::new(vec![2.0], vec![1.0, 1.0]);
        assert!((tf.dcgain() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_bode() {
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0]); // 1/(s+1)
        let (ws, mags, phases) = tf.bode(0.01, 100.0, 50);
        assert_eq!(ws.len(), 50);
        assert_eq!(mags.len(), 50);
        // At low freq, magnitude should be ~0 dB
        assert!(mags[0].abs() < 1.0, "DC gain should be ~0 dB, got {}", mags[0]);
        // At high freq, magnitude should be negative (rolloff)
        assert!(mags[49] < -10.0, "high freq should roll off, got {}", mags[49]);
    }

    #[test]
    fn test_copula_gaussian() {
        let mut rng = crate::distributions::Rng::new(42);
        let corr = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let samples = gaussian_copula(&mut rng, &corr, 100);
        assert_eq!(samples.len(), 100);
        // All values should be in [0, 1]
        assert!(samples.iter().all(|s| s.iter().all(|&v| v >= 0.0 && v <= 1.0)));
    }
}
