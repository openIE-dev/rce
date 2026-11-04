//! Special mathematical functions: Bessel, elliptic, Airy, Struve, etc.
//!
//! Replaces scipy.special, R's besselJ/besselI, MATLAB's special functions.

use std::f64::consts::PI;

/// Bessel function of the first kind J_n(x) for integer order n.
pub fn bessel_j(n: i32, x: f64) -> f64 {
    if n < 0 { return if n % 2 == 0 { bessel_j(-n, x) } else { -bessel_j(-n, x) }; }
    if n == 0 { return bessel_j0(x); }
    if n == 1 { return bessel_j1(x); }
    // Miller's backward recurrence for higher orders
    let mut jn = 0.0;
    let mut jn1 = 1.0;
    let m = (n as f64 + (40.0 * n as f64).sqrt()) as i32;
    let mut norm = 0.0;
    for k in (1..=m).rev() {
        let jn2 = 2.0 * k as f64 / x * jn1 - jn;
        jn = jn1;
        jn1 = jn2;
        if k == n { norm = jn1; }
    }
    let j0 = bessel_j0(x);
    if norm.abs() < 1e-300 { 0.0 } else { j0 * jn / jn1 * norm / jn }
}

/// J_0(x) — polynomial approximation.
fn bessel_j0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let p = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
        let q = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0))));
        p / q
    } else {
        let z = 8.0 / ax;
        let z2 = z * z;
        let theta = ax - 0.785398164;
        let p = 1.0 + z2 * (-0.1098628627e-2 + z2 * (0.2734510407e-4 + z2 * (-0.2073370639e-5 + z2 * 0.2093887211e-6)));
        let q = -0.1562499995e-1 + z2 * (0.1430488765e-3 + z2 * (-0.6911147651e-5 + z2 * (0.7621095161e-6 - z2 * 0.934935152e-7)));
        (0.636619772 / ax).sqrt() * (theta.cos() * p - z * theta.sin() * q)
    }
}

/// J_1(x).
fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let p = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
        let q = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0))));
        p / q
    } else {
        let z = 8.0 / ax;
        let z2 = z * z;
        let theta = ax - 2.356194491;
        let p = 1.0 + z2 * (0.183105e-2 + z2 * (-0.3516396496e-4 + z2 * (0.2457520174e-5 - z2 * 0.240337019e-6)));
        let q = 0.04687499995 + z2 * (-0.2002690873e-3 + z2 * (0.8449199096e-5 + z2 * (-0.88228987e-6 + z2 * 0.105787412e-6)));
        let result = (0.636619772 / ax).sqrt() * (theta.cos() * p - z * theta.sin() * q);
        if x < 0.0 { -result } else { result }
    }
}

/// Modified Bessel function I_0(x).
pub fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let t = (x / 3.75).powi(2);
        1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
    } else {
        let t = 3.75 / ax;
        (ax.exp() / ax.sqrt()) * (0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565 + t * (0.00916281 + t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))))))))
    }
}

/// Modified Bessel function I_1(x).
pub fn bessel_i1(x: f64) -> f64 {
    let ax = x.abs();
    let result = if ax < 3.75 {
        let t = (x / 3.75).powi(2);
        ax * (0.5 + t * (0.87890594 + t * (0.51498869 + t * (0.15084934 + t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))))
    } else {
        let t = 3.75 / ax;
        (ax.exp() / ax.sqrt()) * (0.39894228 + t * (-0.03988024 + t * (-0.00362018 + t * (0.00163801 + t * (-0.01031555 + t * (0.02282967 + t * (-0.02895312 + t * (0.01787654 - t * 0.00420059))))))))
    };
    if x < 0.0 { -result } else { result }
}

/// Modified Bessel function K_0(x).
pub fn bessel_k0(x: f64) -> f64 {
    if x <= 2.0 {
        let y = x * x / 4.0;
        -x.ln() * bessel_i0(x) + (-0.57721566 + y * (0.42278420 + y * (0.23069756 + y * (0.03488590 + y * (0.00262698 + y * (0.00010750 + y * 0.00000740))))))
    } else {
        let y = 2.0 / x;
        ((-x).exp() / x.sqrt()) * (1.25331414 + y * (-0.07832358 + y * (0.02189568 + y * (-0.01062446 + y * (0.00587872 + y * (-0.00251540 + y * 0.00053208))))))
    }
}

/// Gamma function (Lanczos approximation, re-export for convenience).
pub fn gamma(x: f64) -> f64 { crate::distributions::gamma_fn(x) }

/// Log-gamma.
pub fn lgamma(x: f64) -> f64 { crate::distributions::lgamma(x) }

/// Digamma (psi) function.
pub fn digamma(x: f64) -> f64 { crate::math_runtime::digamma(x) }

/// Trigamma function: psi_1(x) = d/dx psi(x).
pub fn trigamma(x: f64) -> f64 {
    if x < 6.0 { return trigamma(x + 1.0) + 1.0 / (x * x); }
    let x2 = 1.0 / (x * x);
    1.0 / x + x2 / 2.0 + x2 * x2 * (1.0 / 6.0 - x2 * (1.0 / 30.0 - x2 / 42.0))
}

/// Polygamma function: n-th derivative of digamma.
pub fn polygamma(n: u32, x: f64) -> f64 {
    match n {
        0 => digamma(x),
        1 => trigamma(x),
        _ => {
            // Numerical differentiation of (n-1)th polygamma
            let h = 1e-5;
            (polygamma(n - 1, x + h) - polygamma(n - 1, x - h)) / (2.0 * h)
        }
    }
}

/// Beta function.
pub fn beta(a: f64, b: f64) -> f64 { crate::distributions::beta_fn(a, b) }

/// Incomplete beta function ratio I_x(a,b).
pub fn betainc(x: f64, a: f64, b: f64) -> f64 { crate::distributions::betainc(x, a, b) }

/// Incomplete gamma function P(a,x).
pub fn gammainc(a: f64, x: f64) -> f64 { crate::distributions::gammainc(a, x) }

/// Regularized upper incomplete gamma Q(a,x) = 1 - P(a,x).
pub fn gammaincc(a: f64, x: f64) -> f64 { 1.0 - gammainc(a, x) }

/// Pochhammer symbol (rising factorial): (x)_n = x(x+1)...(x+n-1).
pub fn pochhammer(x: f64, n: u64) -> f64 {
    (0..n).map(|k| x + k as f64).product()
}

/// Binomial coefficient as continuous function via gamma.
pub fn binom(n: f64, k: f64) -> f64 {
    (lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0)).exp()
}

/// Normalized sinc.
pub fn sinc(x: f64) -> f64 { crate::math_runtime::sinc(x) }

/// Exponential integral E_1(x).
pub fn expn1(x: f64) -> f64 {
    if x <= 0.0 { return f64::INFINITY; }
    if x <= 1.0 {
        // Series expansion
        let mut sum = -0.5772156649 - x.ln();
        let mut term = x;
        for n in 1..50 {
            sum += term / n as f64;
            term *= -x / (n + 1) as f64;
        }
        sum
    } else {
        // Continued fraction
        let mut a = 1.0;
        let mut b = x + 1.0;
        let mut f = 1.0 / b;
        for i in 1..50 {
            let an = -(i * i) as f64;
            b += 2.0;
            let d = an * f + b;
            if d.abs() < 1e-30 { break; }
            f = 1.0 / d;
            a *= f * b;
        }
        a * (-x).exp()
    }
}

/// Lambert W function (principal branch W_0).
pub fn lambertw(x: f64) -> f64 {
    if x < -1.0 / std::f64::consts::E + 1e-10 { return f64::NAN; }
    if x == 0.0 { return 0.0; }
    // Initial guess
    let mut w = if x < 1.0 { x } else { x.ln() - x.ln().ln() };
    // Halley's method
    for _ in 0..50 {
        let ew = w.exp();
        let wew = w * ew;
        let f = wew - x;
        let fp = ew * (w + 1.0);
        let fpp = ew * (w + 2.0);
        let delta = f / (fp - f * fpp / (2.0 * fp));
        w -= delta;
        if delta.abs() < 1e-14 { break; }
    }
    w
}

/// Softmax (numerically stable).
pub fn softmax(x: &[f64]) -> Vec<f64> {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Log-softmax (numerically stable).
pub fn log_softmax(x: &[f64]) -> Vec<f64> {
    let lse = crate::math_runtime::logsumexp(x);
    x.iter().map(|&v| v - lse).collect()
}

/// Logit function.
pub fn logit(p: f64) -> f64 { (p / (1.0 - p)).ln() }

/// Expit (sigmoid/logistic) function.
pub fn expit(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

/// Spence's function (dilogarithm): Li_2(x) = -integral_0^x ln(1-t)/t dt.
pub fn spence(x: f64) -> f64 {
    if x == 0.0 { return PI * PI / 6.0; }
    if x == 1.0 { return 0.0; }
    // Series for |x| <= 1
    let mut sum = 0.0;
    let mut xn = x;
    for n in 1..100 {
        sum += xn / (n * n) as f64;
        xn *= x;
        if xn.abs() < 1e-15 { break; }
    }
    PI * PI / 6.0 - sum - x.abs().ln() * (1.0 - x).ln()
}

/// Dawson's integral: D(x) = exp(-x^2) * integral_0^x exp(t^2) dt.
pub fn dawson(x: f64) -> f64 {
    // Series for small x, asymptotic for large x
    let ax = x.abs();
    if ax < 0.5 {
        let x2 = x * x;
        let mut sum = x;
        let mut term = x;
        for n in 1..50 {
            term *= -2.0 * x2 / (2 * n + 1) as f64;
            sum += term;
            if term.abs() < 1e-15 { break; }
        }
        sum
    } else {
        // Rybicki's method (approximation)
        let x2 = x * x;
        0.5 / x * (1.0 - 1.0 / (2.0 * x2) + 3.0 / (4.0 * x2 * x2))
    }
}

/// Fresnel integrals: returns (S(x), C(x)).
pub fn fresnel(x: f64) -> (f64, f64) {
    // Numerical integration via series for small x
    let x2 = x * x;
    let mut s = 0.0;
    let mut c = 0.0;
    let mut term_s = x * x2 * PI / 2.0 / 3.0; // first term of S
    let mut term_c = x; // first term of C
    s = x * x2 * PI / 6.0;
    c = x;
    let _sign = -1.0;
    for k in 1..50 {
        let n_s = 4 * k + 3;
        let n_c = 4 * k + 1;
        term_s *= -(PI / 2.0).powi(2) * x2 * x2 / ((2 * k) * (2 * k + 1) * n_s) as f64;
        term_c *= -(PI / 2.0).powi(2) * x2 * x2 / ((2 * k - 1) * (2 * k) * n_c) as f64;
        s += term_s;
        c += term_c;
        if term_s.abs() + term_c.abs() < 1e-15 { break; }
    }
    (s, c)
}

/// Zeta function for real s > 1.
pub fn zeta(s: f64) -> f64 {
    if s <= 1.0 { return f64::INFINITY; }
    // Direct summation for moderate s, accelerated
    let n = 50;
    let mut sum = 0.0;
    for k in 1..=n {
        sum += 1.0 / (k as f64).powf(s);
    }
    // Euler-Maclaurin correction
    let nf = n as f64;
    sum += nf.powf(1.0 - s) / (s - 1.0) + 0.5 / nf.powf(s);
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bessel_j0() {
        assert!((bessel_j0(0.0) - 1.0).abs() < 1e-6);
        assert!((bessel_j0(2.4048) - 0.0).abs() < 0.01); // first zero of J0
    }

    #[test]
    fn test_bessel_i0() {
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-10);
        assert!(bessel_i0(1.0) > 1.0); // I0 is monotonically increasing for x > 0
    }

    #[test]
    fn test_trigamma() {
        // trigamma(1) = pi^2 / 6
        assert!((trigamma(1.0) - PI * PI / 6.0).abs() < 0.001);
    }

    #[test]
    fn test_pochhammer() {
        // (1)_5 = 5!
        assert!((pochhammer(1.0, 5) - 120.0).abs() < 1e-10);
    }

    #[test]
    fn test_lambertw() {
        // W(0) = 0
        assert!(lambertw(0.0).abs() < 1e-10);
        // W(e) = 1
        assert!((lambertw(std::f64::consts::E) - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_zeta() {
        // zeta(2) = pi^2/6
        assert!((zeta(2.0) - PI * PI / 6.0).abs() < 0.01);
    }

    #[test]
    fn test_softmax() {
        let s = softmax(&[1.0, 2.0, 3.0]);
        let total: f64 = s.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
        assert!(s[2] > s[1] && s[1] > s[0]);
    }
}
