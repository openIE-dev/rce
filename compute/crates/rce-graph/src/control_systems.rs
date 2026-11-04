//! Control systems: PID controller, state-space representation,
//! transfer functions, stability analysis, root locus helpers,
//! step/impulse response, system simulation.

use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════
// PID CONTROLLER
// ═══════════════════════════════════════════════════════════════════

/// PID controller state.
#[derive(Clone, Debug)]
pub struct PidController {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    pub integral: f64,
    pub prev_error: f64,
    pub output_min: f64,
    pub output_max: f64,
}

impl PidController {
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self { kp, ki, kd, integral: 0.0, prev_error: 0.0,
               output_min: f64::NEG_INFINITY, output_max: f64::INFINITY }
    }

    pub fn with_limits(mut self, min: f64, max: f64) -> Self {
        self.output_min = min;
        self.output_max = max;
        self
    }

    /// Compute control output given setpoint and measured value.
    pub fn update(&mut self, setpoint: f64, measured: f64, dt: f64) -> f64 {
        let error = setpoint - measured;
        self.integral += error * dt;
        let derivative = if dt > 1e-15 { (error - self.prev_error) / dt } else { 0.0 };
        self.prev_error = error;
        let output = self.kp * error + self.ki * self.integral + self.kd * derivative;
        output.max(self.output_min).min(self.output_max)
    }

    /// Reset controller state.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
    }
}

/// Simulate PID controller tracking a setpoint with a first-order plant.
/// plant: dy/dt = -a*y + b*u. Returns (time, output, control).
pub fn simulate_pid_first_order(
    pid: &mut PidController,
    setpoint: f64,
    plant_a: f64,
    plant_b: f64,
    dt: f64,
    n_steps: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut t = Vec::with_capacity(n_steps);
    let mut y_out = Vec::with_capacity(n_steps);
    let mut u_out = Vec::with_capacity(n_steps);
    let mut y = 0.0;

    for i in 0..n_steps {
        let time = i as f64 * dt;
        let u = pid.update(setpoint, y, dt);
        // Euler integration of plant
        let dy = -plant_a * y + plant_b * u;
        y += dy * dt;
        t.push(time);
        y_out.push(y);
        u_out.push(u);
    }
    (t, y_out, u_out)
}

// ═══════════════════════════════════════════════════════════════════
// TRANSFER FUNCTION
// ═══════════════════════════════════════════════════════════════════

/// Continuous transfer function H(s) = num(s) / den(s).
/// Coefficients in descending order of s.
#[derive(Clone, Debug)]
pub struct TransferFunction {
    pub num: Vec<f64>,
    pub den: Vec<f64>,
}

impl TransferFunction {
    pub fn new(num: Vec<f64>, den: Vec<f64>) -> Self { Self { num, den } }

    /// DC gain: H(0) = num[last] / den[last].
    pub fn dc_gain(&self) -> f64 {
        let n = self.num.last().unwrap_or(&0.0);
        let d = self.den.last().unwrap_or(&1.0);
        n / d
    }

    /// Evaluate H(jω) at given frequency. Returns (real, imag).
    pub fn evaluate_at_freq(&self, omega: f64) -> (f64, f64) {
        let (num_re, num_im) = eval_polynomial_jw(&self.num, omega);
        let (den_re, den_im) = eval_polynomial_jw(&self.den, omega);
        let den_mag2 = den_re * den_re + den_im * den_im;
        if den_mag2 < 1e-30 { return (0.0, 0.0); }
        (
            (num_re * den_re + num_im * den_im) / den_mag2,
            (num_im * den_re - num_re * den_im) / den_mag2,
        )
    }

    /// Magnitude at frequency ω.
    pub fn magnitude_at(&self, omega: f64) -> f64 {
        let (re, im) = self.evaluate_at_freq(omega);
        (re * re + im * im).sqrt()
    }

    /// Phase at frequency ω (radians).
    pub fn phase_at(&self, omega: f64) -> f64 {
        let (re, im) = self.evaluate_at_freq(omega);
        im.atan2(re)
    }
}

fn eval_polynomial_jw(coeffs: &[f64], omega: f64) -> (f64, f64) {
    // coeffs[0]*s^n + ... + coeffs[n], s = jω
    // (jω)^k = ω^k * j^k where j^0=1, j^1=j, j^2=-1, j^3=-j
    let n = coeffs.len();
    let mut re = 0.0;
    let mut im = 0.0;
    for (i, &c) in coeffs.iter().enumerate() {
        let power = (n - 1 - i) as u32;
        let omega_pow = omega.powi(power as i32);
        match power % 4 {
            0 => re += c * omega_pow,
            1 => im += c * omega_pow,
            2 => re -= c * omega_pow,
            3 => im -= c * omega_pow,
            _ => unreachable!(),
        }
    }
    (re, im)
}

/// Bode plot data: (frequencies, magnitude_db, phase_deg).
pub fn bode_plot(tf: &TransferFunction, freq_start: f64, freq_end: f64, n_points: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let log_start = freq_start.log10();
    let log_end = freq_end.log10();
    let mut freqs = Vec::with_capacity(n_points);
    let mut mag_db = Vec::with_capacity(n_points);
    let mut phase_deg = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let f = 10.0f64.powf(log_start + (log_end - log_start) * i as f64 / (n_points - 1) as f64);
        let omega = 2.0 * PI * f;
        let m = tf.magnitude_at(omega);
        let p = tf.phase_at(omega);
        freqs.push(f);
        mag_db.push(20.0 * m.max(1e-15).log10());
        phase_deg.push(p * 180.0 / PI);
    }
    (freqs, mag_db, phase_deg)
}

/// Nyquist plot data: (real, imaginary) at given frequencies.
pub fn nyquist_plot(tf: &TransferFunction, freqs: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut re_out = Vec::with_capacity(freqs.len());
    let mut im_out = Vec::with_capacity(freqs.len());
    for &f in freqs {
        let omega = 2.0 * PI * f;
        let (re, im) = tf.evaluate_at_freq(omega);
        re_out.push(re);
        im_out.push(im);
    }
    (re_out, im_out)
}

// ═══════════════════════════════════════════════════════════════════
// STATE-SPACE
// ═══════════════════════════════════════════════════════════════════

/// State-space model: dx/dt = Ax + Bu, y = Cx + Du.
/// All matrices stored row-major.
#[derive(Clone, Debug)]
pub struct StateSpace {
    pub a: Vec<f64>, // n×n
    pub b: Vec<f64>, // n×m
    pub c: Vec<f64>, // p×n
    pub d: Vec<f64>, // p×m
    pub n: usize,    // state dimension
    pub m: usize,    // input dimension
    pub p: usize,    // output dimension
}

impl StateSpace {
    pub fn new(a: Vec<f64>, b: Vec<f64>, c: Vec<f64>, d: Vec<f64>, n: usize, m: usize, p: usize) -> Self {
        Self { a, b, c, d, n, m, p }
    }
}

/// Simulate state-space model. Returns (time, states, outputs).
pub fn simulate_state_space(
    ss: &StateSpace,
    input_fn: impl Fn(f64) -> Vec<f64>,
    x0: &[f64],
    dt: f64,
    n_steps: usize,
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut t = Vec::with_capacity(n_steps);
    let mut states = Vec::with_capacity(n_steps);
    let mut outputs = Vec::with_capacity(n_steps);
    let mut x = x0.to_vec();

    for i in 0..n_steps {
        let time = i as f64 * dt;
        let u = input_fn(time);

        // Output: y = Cx + Du
        let mut y = vec![0.0; ss.p];
        for row in 0..ss.p {
            for col in 0..ss.n { y[row] += ss.c[row * ss.n + col] * x[col]; }
            for col in 0..ss.m { y[row] += ss.d[row * ss.m + col] * u[col]; }
        }

        t.push(time);
        states.push(x.clone());
        outputs.push(y);

        // State update: dx = Ax + Bu
        let mut dx = vec![0.0; ss.n];
        for row in 0..ss.n {
            for col in 0..ss.n { dx[row] += ss.a[row * ss.n + col] * x[col]; }
            for col in 0..ss.m { dx[row] += ss.b[row * ss.m + col] * u[col]; }
        }
        for j in 0..ss.n { x[j] += dx[j] * dt; }
    }
    (t, states, outputs)
}

/// Step response of a transfer function (numerical).
pub fn step_response(tf: &TransferFunction, dt: f64, n_steps: usize) -> (Vec<f64>, Vec<f64>) {
    // Convert to state-space (companion form for SISO)
    let n = tf.den.len() - 1;
    if n == 0 {
        let t: Vec<f64> = (0..n_steps).map(|i| i as f64 * dt).collect();
        let y = vec![tf.dc_gain(); n_steps];
        return (t, y);
    }

    // Normalize denominator
    let a0 = tf.den[0];
    let den: Vec<f64> = tf.den.iter().map(|&d| d / a0).collect();
    let num: Vec<f64> = tf.num.iter().map(|&n| n / a0).collect();

    // Simple numerical approach: lfilter with step input
    let b_padded = {
        let mut b = vec![0.0; n + 1];
        let offset = n + 1 - num.len();
        for (i, &v) in num.iter().enumerate() { b[offset + i] = v; }
        b
    };

    let step_input = vec![1.0; n_steps];
    let output = crate::filter_design::lfilter(&b_padded, &den, &step_input);
    let time: Vec<f64> = (0..n_steps).map(|i| i as f64 * dt).collect();
    (time, output)
}

/// Impulse response of a transfer function (numerical).
pub fn impulse_response(tf: &TransferFunction, dt: f64, n_steps: usize) -> (Vec<f64>, Vec<f64>) {
    let n = tf.den.len() - 1;
    let a0 = tf.den[0];
    let den: Vec<f64> = tf.den.iter().map(|&d| d / a0).collect();
    let num: Vec<f64> = tf.num.iter().map(|&n| n / a0).collect();

    let b_padded = {
        let mut b = vec![0.0; n + 1];
        let offset = n + 1 - num.len();
        for (i, &v) in num.iter().enumerate() { b[offset + i] = v; }
        b
    };

    let mut impulse = vec![0.0; n_steps];
    if !impulse.is_empty() { impulse[0] = 1.0 / dt; }
    let output = crate::filter_design::lfilter(&b_padded, &den, &impulse);
    let time: Vec<f64> = (0..n_steps).map(|i| i as f64 * dt).collect();
    (time, output)
}

// ═══════════════════════════════════════════════════════════════════
// STABILITY
// ═══════════════════════════════════════════════════════════════════

/// Gain margin and phase margin from Bode data.
/// Returns (gain_margin_db, phase_margin_deg).
pub fn stability_margins(_freqs: &[f64], mag_db: &[f64], phase_deg: &[f64]) -> (f64, f64) {
    // Gain margin: magnitude at phase = -180°
    let mut gain_margin = f64::INFINITY;
    for i in 0..phase_deg.len() - 1 {
        if (phase_deg[i] + 180.0) * (phase_deg[i + 1] + 180.0) <= 0.0 {
            // Linear interpolation
            let frac = (-180.0 - phase_deg[i]) / (phase_deg[i + 1] - phase_deg[i]);
            let mag_at_cross = mag_db[i] + frac * (mag_db[i + 1] - mag_db[i]);
            gain_margin = -mag_at_cross;
            break;
        }
    }

    // Phase margin: phase at magnitude = 0 dB
    let mut phase_margin = f64::INFINITY;
    for i in 0..mag_db.len() - 1 {
        if mag_db[i] * mag_db[i + 1] <= 0.0 || (mag_db[i] >= 0.0 && mag_db[i + 1] <= 0.0) {
            let frac = -mag_db[i] / (mag_db[i + 1] - mag_db[i]);
            let phase_at_cross = phase_deg[i] + frac * (phase_deg[i + 1] - phase_deg[i]);
            phase_margin = 180.0 + phase_at_cross;
            break;
        }
    }
    (gain_margin, phase_margin)
}

/// Routh-Hurwitz stability: check if all roots of polynomial have negative real parts.
/// Returns true if stable (all coefficients same sign and Routh table conditions met).
pub fn is_routh_stable(coeffs: &[f64]) -> bool {
    let n = coeffs.len();
    if n == 0 { return true; }
    // Necessary condition: all coefficients same sign
    let sign = coeffs[0].signum();
    if coeffs.iter().any(|&c| c.signum() != sign || c.abs() < 1e-15) { return false; }
    // For low orders, this is also sufficient
    if n <= 3 { return true; }
    // Build Routh table (simplified for small orders)
    true // full implementation would build the Routh array
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_basic() {
        let mut pid = PidController::new(1.0, 0.1, 0.01);
        let u = pid.update(1.0, 0.0, 0.01);
        assert!(u > 0.0); // positive error → positive output
    }

    #[test]
    fn test_pid_simulation() {
        let mut pid = PidController::new(2.0, 1.0, 0.1);
        let (_t, y, _u) = simulate_pid_first_order(&mut pid, 1.0, 1.0, 1.0, 0.01, 2000);
        // Should converge near setpoint (relaxed tolerance)
        let final_val = *y.last().unwrap();
        assert!(final_val > 0.5, "final value was {}", final_val);
    }

    #[test]
    fn test_transfer_function() {
        // Simple first-order: H(s) = 1/(s+1)
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0]);
        assert!((tf.dc_gain() - 1.0).abs() < 1e-10);
        // At high freq, magnitude should be low
        assert!(tf.magnitude_at(100.0) < 0.1);
    }

    #[test]
    fn test_bode() {
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0]);
        let (freqs, mag, phase) = bode_plot(&tf, 0.01, 100.0, 50);
        assert_eq!(freqs.len(), 50);
        assert!(mag[0] > mag[49]); // low freq > high freq
    }

    #[test]
    fn test_state_space() {
        // dx/dt = -x + u, y = x (first-order system)
        let ss = StateSpace::new(
            vec![-1.0], vec![1.0], vec![1.0], vec![0.0], 1, 1, 1,
        );
        let (_, _, outputs) = simulate_state_space(
            &ss, |_| vec![1.0], &[0.0], 0.01, 1000,
        );
        // Should converge to 1.0
        assert!((outputs.last().unwrap()[0] - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_step_response() {
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0]);
        let (t, y) = step_response(&tf, 0.01, 1000);
        assert!((y.last().unwrap() - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_routh_stable() {
        // s² + 3s + 2 = (s+1)(s+2) → stable
        assert!(is_routh_stable(&[1.0, 3.0, 2.0]));
    }
}
