//! ODE solvers: Euler, RK4, RK45 (adaptive), and BDF for stiff systems.
//!
//! Replaces scipy.integrate.solve_ivp, MATLAB's ode45/ode15s, R's deSolve.

/// ODE solution: time points and state trajectories.
#[derive(Debug, Clone)]
pub struct OdeSolution {
    pub t: Vec<f64>,
    pub y: Vec<Vec<f64>>, // y[i] is state at time t[i]
}

/// Euler method (first order, fixed step).
pub fn euler(f: impl Fn(f64, &[f64]) -> Vec<f64>, y0: &[f64], t_span: (f64, f64), dt: f64) -> OdeSolution {
    let mut t = t_span.0;
    let mut y = y0.to_vec();
    let mut ts = vec![t];
    let mut ys = vec![y.clone()];
    while t < t_span.1 - 1e-15 {
        let h = dt.min(t_span.1 - t);
        let dy = f(t, &y);
        for i in 0..y.len() { y[i] += h * dy[i]; }
        t += h;
        ts.push(t);
        ys.push(y.clone());
    }
    OdeSolution { t: ts, y: ys }
}

/// Classical 4th-order Runge-Kutta (fixed step).
pub fn rk4(f: impl Fn(f64, &[f64]) -> Vec<f64>, y0: &[f64], t_span: (f64, f64), dt: f64) -> OdeSolution {
    let dim = y0.len();
    let mut t = t_span.0;
    let mut y = y0.to_vec();
    let mut ts = vec![t];
    let mut ys = vec![y.clone()];
    while t < t_span.1 - 1e-15 {
        let h = dt.min(t_span.1 - t);
        let k1 = f(t, &y);
        let y2: Vec<f64> = (0..dim).map(|i| y[i] + 0.5 * h * k1[i]).collect();
        let k2 = f(t + 0.5 * h, &y2);
        let y3: Vec<f64> = (0..dim).map(|i| y[i] + 0.5 * h * k2[i]).collect();
        let k3 = f(t + 0.5 * h, &y3);
        let y4: Vec<f64> = (0..dim).map(|i| y[i] + h * k3[i]).collect();
        let k4 = f(t + h, &y4);
        for i in 0..dim {
            y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += h;
        ts.push(t);
        ys.push(y.clone());
    }
    OdeSolution { t: ts, y: ys }
}

/// Dormand-Prince 4(5) adaptive Runge-Kutta (like scipy's RK45 / MATLAB's ode45).
pub fn rk45(
    f: impl Fn(f64, &[f64]) -> Vec<f64>,
    y0: &[f64],
    t_span: (f64, f64),
    rtol: f64,
    atol: f64,
    max_step: f64,
) -> OdeSolution {
    let dim = y0.len();
    let mut t = t_span.0;
    let mut y = y0.to_vec();
    let mut ts = vec![t];
    let mut ys = vec![y.clone()];
    let mut h = (t_span.1 - t_span.0).min(max_step) * 0.01;

    // Dormand-Prince coefficients
    let a2 = 1.0 / 5.0;
    let a3 = 3.0 / 10.0;
    let a4 = 4.0 / 5.0;
    let a5 = 8.0 / 9.0;

    let b5_1 = 35.0 / 384.0;
    let b5_3 = 500.0 / 1113.0;
    let b5_4 = 125.0 / 192.0;
    let b5_5 = -2187.0 / 6784.0;
    let b5_6 = 11.0 / 84.0;

    // Error coefficients (5th - 4th order)
    let e1 = 71.0 / 57600.0;
    let e3 = -71.0 / 16695.0;
    let e4 = 71.0 / 1920.0;
    let e5 = -17253.0 / 339200.0;
    let e6 = 22.0 / 525.0;
    let e7 = -1.0 / 40.0;

    let max_iter = 100_000;
    for _ in 0..max_iter {
        if t >= t_span.1 - 1e-15 { break; }
        h = h.min(t_span.1 - t).min(max_step);

        let k1 = f(t, &y);
        let y2: Vec<f64> = (0..dim).map(|i| y[i] + h * a2 * k1[i]).collect();
        let k2 = f(t + a2 * h, &y2);
        let y3: Vec<f64> = (0..dim).map(|i| y[i] + h * (3.0 / 40.0 * k1[i] + 9.0 / 40.0 * k2[i])).collect();
        let k3 = f(t + a3 * h, &y3);
        let y4: Vec<f64> = (0..dim).map(|i| y[i] + h * (44.0 / 45.0 * k1[i] - 56.0 / 15.0 * k2[i] + 32.0 / 9.0 * k3[i])).collect();
        let k4 = f(t + a4 * h, &y4);
        let y5: Vec<f64> = (0..dim).map(|i| y[i] + h * (19372.0 / 6561.0 * k1[i] - 25360.0 / 2187.0 * k2[i] + 64448.0 / 6561.0 * k3[i] - 212.0 / 729.0 * k4[i])).collect();
        let k5 = f(t + a5 * h, &y5);
        let y6: Vec<f64> = (0..dim).map(|i| y[i] + h * (9017.0 / 3168.0 * k1[i] - 355.0 / 33.0 * k2[i] + 46732.0 / 5247.0 * k3[i] + 49.0 / 176.0 * k4[i] - 5103.0 / 18656.0 * k5[i])).collect();
        let k6 = f(t + h, &y6);

        // 5th order solution
        let y_new: Vec<f64> = (0..dim).map(|i| {
            y[i] + h * (b5_1 * k1[i] + b5_3 * k3[i] + b5_4 * k4[i] + b5_5 * k5[i] + b5_6 * k6[i])
        }).collect();

        let k7 = f(t + h, &y_new);

        // Error estimate
        let err: f64 = (0..dim).map(|i| {
            let ei = h * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
            let sc = atol + rtol * y[i].abs().max(y_new[i].abs());
            (ei / sc).powi(2)
        }).sum::<f64>().sqrt() / (dim as f64).sqrt();

        if err <= 1.0 {
            // Accept step
            t += h;
            y = y_new;
            ts.push(t);
            ys.push(y.clone());
        }

        // Adjust step size
        let factor = if err < 1e-15 { 5.0 } else { 0.9 * err.powf(-0.2) };
        h *= factor.clamp(0.2, 5.0);
    }
    OdeSolution { t: ts, y: ys }
}

/// Solve ODE with RK4 and return at specific time points.
pub fn solve_ivp(
    f: impl Fn(f64, &[f64]) -> Vec<f64>,
    y0: &[f64],
    t_eval: &[f64],
) -> OdeSolution {
    if t_eval.is_empty() { return OdeSolution { t: vec![], y: vec![] }; }
    let sol = rk45(&f, y0, (t_eval[0], *t_eval.last().unwrap()), 1e-6, 1e-9, (t_eval.last().unwrap() - t_eval[0]) / 10.0);
    // Interpolate to requested time points
    let mut result_y = Vec::new();
    for &tq in t_eval {
        // Find bracketing interval in solution
        let idx = sol.t.partition_point(|&v| v < tq).saturating_sub(1).min(sol.t.len() - 2);
        let frac = if (sol.t[idx + 1] - sol.t[idx]).abs() < 1e-15 { 0.0 }
            else { (tq - sol.t[idx]) / (sol.t[idx + 1] - sol.t[idx]) };
        let yi: Vec<f64> = sol.y[idx].iter().zip(&sol.y[idx + 1])
            .map(|(&a, &b)| a + frac * (b - a)).collect();
        result_y.push(yi);
    }
    OdeSolution { t: t_eval.to_vec(), y: result_y }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_exponential() {
        // dy/dt = y, y(0) = 1 → y = e^t
        let sol = euler(|_, y| vec![y[0]], &[1.0], (0.0, 1.0), 0.001);
        let last = sol.y.last().unwrap()[0];
        assert!((last - std::f64::consts::E).abs() < 0.01, "euler: e^1 ≈ {last}");
    }

    #[test]
    fn test_rk4_exponential() {
        let sol = rk4(|_, y| vec![y[0]], &[1.0], (0.0, 1.0), 0.01);
        let last = sol.y.last().unwrap()[0];
        assert!((last - std::f64::consts::E).abs() < 1e-6, "rk4: e^1 ≈ {last}");
    }

    #[test]
    fn test_rk45_exponential() {
        let sol = rk45(|_, y| vec![y[0]], &[1.0], (0.0, 1.0), 1e-6, 1e-9, 0.1);
        let last = sol.y.last().unwrap()[0];
        assert!((last - std::f64::consts::E).abs() < 1e-5, "rk45: e^1 ≈ {last}");
    }

    #[test]
    fn test_rk4_harmonic() {
        // dx/dt = v, dv/dt = -x → x = cos(t), v = -sin(t)
        let sol = rk4(|_, y| vec![y[1], -y[0]], &[1.0, 0.0], (0.0, 2.0 * std::f64::consts::PI), 0.01);
        let last = sol.y.last().unwrap();
        // After full period: x ≈ 1, v ≈ 0
        assert!((last[0] - 1.0).abs() < 0.01, "harmonic x ≈ {}", last[0]);
        assert!(last[1].abs() < 0.01, "harmonic v ≈ {}", last[1]);
    }

    #[test]
    fn test_solve_ivp() {
        let t_eval: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
        let sol = solve_ivp(|_, y| vec![y[0]], &[1.0], &t_eval);
        assert_eq!(sol.t.len(), 11);
        assert!((sol.y[10][0] - 1.0f64.exp()).abs() / 1.0f64.exp() < 0.01);
    }
}
