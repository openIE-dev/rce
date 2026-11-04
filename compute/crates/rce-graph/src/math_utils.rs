//! Math utilities: common numeric helpers, special values,
//! rounding modes, comparison with tolerance, clamping, mapping.

// ═══════════════════════════════════════════════════════════════════
// COMPARISON & TOLERANCE
// ═══════════════════════════════════════════════════════════════════

/// Approximate equality within tolerance.
pub fn approx_eq(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }

/// Relative equality: |a-b| / max(|a|,|b|) < tol.
pub fn relative_eq(a: f64, b: f64, tol: f64) -> bool {
    let max_abs = a.abs().max(b.abs());
    if max_abs < 1e-15 { return (a - b).abs() < tol; }
    (a - b).abs() / max_abs < tol
}

/// Assert approximate equality (returns error message if not).
pub fn assert_approx(a: f64, b: f64, tol: f64) -> Result<(), String> {
    if approx_eq(a, b, tol) { Ok(()) }
    else { Err(format!("{} != {} (diff = {}, tol = {})", a, b, (a-b).abs(), tol)) }
}

/// Maximum of two values.
pub fn fmax(a: f64, b: f64) -> f64 { a.max(b) }

/// Minimum of two values.
pub fn fmin(a: f64, b: f64) -> f64 { a.min(b) }

/// Clamp to [lo, hi].
pub fn fclamp(x: f64, lo: f64, hi: f64) -> f64 { x.max(lo).min(hi) }

/// Safe division (returns default on divide-by-zero).
pub fn safe_div(a: f64, b: f64, default: f64) -> f64 { if b.abs() < 1e-15 { default } else { a / b } }

/// Safe log (returns -inf for non-positive).
pub fn safe_log(x: f64) -> f64 { if x > 0.0 { x.ln() } else { f64::NEG_INFINITY } }

/// Safe sqrt (returns 0 for negative).
pub fn safe_sqrt(x: f64) -> f64 { if x >= 0.0 { x.sqrt() } else { 0.0 } }

// ═══════════════════════════════════════════════════════════════════
// ROUNDING
// ═══════════════════════════════════════════════════════════════════

/// Round to nearest integer.
pub fn round(x: f64) -> f64 { x.round() }

/// Round toward zero (truncate).
pub fn trunc(x: f64) -> f64 { x.trunc() }

/// Round to n decimal places.
pub fn round_n(x: f64, n: u32) -> f64 { let f = 10.0f64.powi(n as i32); (x * f).round() / f }

/// Round to nearest multiple of step.
pub fn round_to_multiple(x: f64, step: f64) -> f64 { (x / step).round() * step }

/// Ceiling division (integer).
pub fn ceildiv(a: i64, b: i64) -> i64 { (a + b - 1) / b }

/// Floor modulo (always non-negative for positive divisor).
pub fn floormod(a: f64, b: f64) -> f64 { ((a % b) + b) % b }

// ═══════════════════════════════════════════════════════════════════
// SPECIAL VALUES
// ═══════════════════════════════════════════════════════════════════

pub fn is_nan(x: f64) -> bool { x.is_nan() }
pub fn is_inf(x: f64) -> bool { x.is_infinite() }
pub fn is_finite(x: f64) -> bool { x.is_finite() }
pub fn is_positive(x: f64) -> bool { x > 0.0 }
pub fn is_negative(x: f64) -> bool { x < 0.0 }
pub fn is_zero(x: f64) -> bool { x.abs() < 1e-15 }
pub fn is_integer(x: f64) -> bool { x.is_finite() && (x - x.round()).abs() < 1e-10 }
pub fn is_even(x: f64) -> bool { is_integer(x) && (x.round() as i64) % 2 == 0 }
pub fn is_odd(x: f64) -> bool { is_integer(x) && (x.round() as i64) % 2 != 0 }

// ═══════════════════════════════════════════════════════════════════
// MAPPING & INTERPOLATION
// ═══════════════════════════════════════════════════════════════════

/// Linear map from [a1, b1] to [a2, b2].
pub fn linmap(x: f64, a1: f64, b1: f64, a2: f64, b2: f64) -> f64 {
    a2 + (x - a1) / (b1 - a1) * (b2 - a2)
}

/// Linear interpolation.
pub fn lerp(a: f64, b: f64, t: f64) -> f64 { a + t * (b - a) }

/// Inverse lerp.
pub fn inv_lerp(a: f64, b: f64, x: f64) -> f64 { if (b - a).abs() < 1e-15 { 0.0 } else { (x - a) / (b - a) } }

/// Smooth step.
pub fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = fclamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Step function.
pub fn step(edge: f64, x: f64) -> f64 { if x >= edge { 1.0 } else { 0.0 } }

/// Heaviside step (0.5 at zero).
pub fn heaviside(x: f64) -> f64 { if x > 0.0 { 1.0 } else if x < 0.0 { 0.0 } else { 0.5 } }

/// Rectangular pulse: 1 if |x| < width/2, else 0.
pub fn rect(x: f64, width: f64) -> f64 { if x.abs() < width / 2.0 { 1.0 } else { 0.0 } }

/// Triangle function: max(0, 1 - |x|).
pub fn tri(x: f64) -> f64 { (1.0 - x.abs()).max(0.0) }

/// Sinc function: sin(πx)/(πx), sinc(0) = 1.
pub fn sinc(x: f64) -> f64 { if x.abs() < 1e-15 { 1.0 } else { let px = std::f64::consts::PI * x; px.sin() / px } }

/// Gaussian function: exp(-x²/(2σ²)).
pub fn gaussian(x: f64, sigma: f64) -> f64 { (-0.5 * (x / sigma).powi(2)).exp() }

/// Logistic function: 1/(1 + exp(-x)).
pub fn logistic(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

/// Softplus: ln(1 + exp(x)).
pub fn softplus(x: f64) -> f64 { if x > 20.0 { x } else { (1.0 + x.exp()).ln() } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approx_eq() {
        assert!(approx_eq(1.0, 1.0 + 1e-11, 1e-10));
        assert!(!approx_eq(1.0, 2.0, 0.5));
    }

    #[test]
    fn test_safe_div() {
        assert_eq!(safe_div(10.0, 2.0, 0.0), 5.0);
        assert_eq!(safe_div(10.0, 0.0, -1.0), -1.0);
    }

    #[test]
    fn test_round_n() {
        assert!((round_n(3.14159, 2) - 3.14).abs() < 1e-10);
    }

    #[test]
    fn test_round_multiple() {
        assert!((round_to_multiple(7.3, 0.5) - 7.5).abs() < 1e-10);
    }

    #[test]
    fn test_is_checks() {
        assert!(is_integer(3.0));
        assert!(!is_integer(3.5));
        assert!(is_even(4.0));
        assert!(is_odd(3.0));
    }

    #[test]
    fn test_linmap() {
        assert!((linmap(5.0, 0.0, 10.0, 0.0, 100.0) - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_sinc() {
        assert!((sinc(0.0) - 1.0).abs() < 1e-10);
        assert!(sinc(1.0).abs() < 1e-10);
    }

    #[test]
    fn test_heaviside() {
        assert_eq!(heaviside(1.0), 1.0);
        assert_eq!(heaviside(-1.0), 0.0);
        assert_eq!(heaviside(0.0), 0.5);
    }

    #[test]
    fn test_floormod() {
        assert!((floormod(-1.0, 3.0) - 2.0).abs() < 1e-10);
    }
}
