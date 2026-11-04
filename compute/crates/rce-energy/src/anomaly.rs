//! The anomaly test on delta = E_meas - E_sim.

/// True when |delta| > k * sigma. A zero sigma (a perfect fit) flags any
/// nonzero delta; a non-finite input is never flagged as normal.
pub fn is_anomaly(delta_j: f64, sigma_j: f64, k: f64) -> bool {
    if !delta_j.is_finite() || !sigma_j.is_finite() {
        return true;
    }
    delta_j.abs() > k * sigma_j.max(0.0)
}
