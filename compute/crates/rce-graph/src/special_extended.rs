//! Extended special functions: elliptic integrals, hypergeometric,
//! Airy functions, Struve, spherical harmonics, Bernoulli/Euler numbers,
//! polylogarithm, Clausen function, Debye functions.

use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════
// ELLIPTIC INTEGRALS
// ═══════════════════════════════════════════════════════════════════

/// Complete elliptic integral of the first kind K(k) (AGM method).
pub fn elliptic_k(k: f64) -> f64 {
    if k.abs() >= 1.0 { return f64::INFINITY; }
    let mut a = 1.0;
    let mut b = (1.0 - k * k).sqrt();
    for _ in 0..50 {
        let a_new = (a + b) / 2.0;
        let b_new = (a * b).sqrt();
        if (a_new - b_new).abs() < 1e-15 { return PI / (2.0 * a_new); }
        a = a_new; b = b_new;
    }
    PI / (2.0 * a)
}

/// Complete elliptic integral of the second kind E(k).
pub fn elliptic_e(k: f64) -> f64 {
    if k.abs() >= 1.0 { return 1.0; }
    // Arithmetic-geometric mean method
    let mut a = 1.0;
    let mut b = (1.0 - k * k).sqrt();
    let mut c = k;
    let mut sum = 1.0 - k * k / 2.0;
    let mut pow2 = 0.5;
    for _ in 0..50 {
        let a_new = (a + b) / 2.0;
        let b_new = (a * b).sqrt();
        c = (a - b) / 2.0;
        pow2 *= 2.0;
        sum -= pow2 * c * c;
        if c.abs() < 1e-15 { break; }
        a = a_new; b = b_new;
    }
    sum * PI / (2.0 * a)
}

/// Incomplete elliptic integral of the first kind F(φ, k).
pub fn elliptic_f(phi: f64, k: f64) -> f64 {
    // Numerical integration (Gauss-Legendre)
    let n = 20;
    let mut sum = 0.0;
    let half = phi / 2.0;
    for i in 0..n {
        let t = half * (2.0 * i as f64 + 1.0) / n as f64;
        let sint = t.sin();
        sum += 1.0 / (1.0 - k * k * sint * sint).sqrt();
    }
    sum * phi / n as f64
}

/// Incomplete elliptic integral of the second kind E(φ, k).
pub fn elliptic_e_incomplete(phi: f64, k: f64) -> f64 {
    let n = 20;
    let mut sum = 0.0;
    let half = phi / 2.0;
    for i in 0..n {
        let t = half * (2.0 * i as f64 + 1.0) / n as f64;
        let sint = t.sin();
        sum += (1.0 - k * k * sint * sint).sqrt();
    }
    sum * phi / n as f64
}

// ═══════════════════════════════════════════════════════════════════
// HYPERGEOMETRIC
// ═══════════════════════════════════════════════════════════════════

/// Generalized hypergeometric 2F1(a, b; c; z) via series expansion.
pub fn hyp2f1(a: f64, b: f64, c: f64, z: f64) -> f64 {
    if z.abs() >= 1.0 { return f64::NAN; } // diverges at |z| >= 1
    let mut term = 1.0;
    let mut sum = 1.0;
    for n in 0..200 {
        let nf = n as f64;
        term *= (a + nf) * (b + nf) / ((c + nf) * (nf + 1.0)) * z;
        sum += term;
        if term.abs() < 1e-15 * sum.abs() { break; }
    }
    sum
}

/// Confluent hypergeometric 1F1(a; b; z) (Kummer's function).
pub fn hyp1f1(a: f64, b: f64, z: f64) -> f64 {
    let mut term = 1.0;
    let mut sum = 1.0;
    for n in 0..300 {
        let nf = n as f64;
        term *= (a + nf) / ((b + nf) * (nf + 1.0)) * z;
        sum += term;
        if term.abs() < 1e-15 * sum.abs() { break; }
    }
    sum
}

/// Hypergeometric 0F1(; b; z).
pub fn hyp0f1(b: f64, z: f64) -> f64 {
    let mut term = 1.0;
    let mut sum = 1.0;
    for n in 0..200 {
        let nf = n as f64;
        term *= z / ((b + nf) * (nf + 1.0));
        sum += term;
        if term.abs() < 1e-15 * sum.abs() { break; }
    }
    sum
}

// ═══════════════════════════════════════════════════════════════════
// AIRY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

/// Airy function Ai(x) (power series for small x, asymptotic otherwise).
pub fn airy_ai(x: f64) -> f64 {
    if x > 5.0 {
        // Asymptotic: Ai(x) ~ exp(-2/3 x^{3/2}) / (2√π x^{1/4})
        let xi = 2.0 / 3.0 * x.powf(1.5);
        (-xi).exp() / (2.0 * PI.sqrt() * x.powf(0.25))
    } else if x < -5.0 {
        let xi = 2.0 / 3.0 * (-x).powf(1.5);
        (xi - PI / 4.0).sin() / (PI.sqrt() * (-x).powf(0.25))
    } else {
        // Taylor series
        let c1 = 0.3550280538878172;  // Ai(0)
        let c2 = -0.2588194037928068; // Ai'(0)
        let mut f = c1;
        let mut g = c2;
        let mut a_term = c1;
        let mut b_term = c2;
        for n in 1..100 {
            let nf = n as f64;
            a_term *= x.powi(3) / ((3.0 * nf - 1.0) * (3.0 * nf));
            b_term *= x.powi(3) / ((3.0 * nf) * (3.0 * nf + 1.0));
            f += a_term;
            g += b_term;
            if a_term.abs() + b_term.abs() < 1e-15 { break; }
        }
        f
    }
}

/// Airy function Bi(x).
pub fn airy_bi(x: f64) -> f64 {
    if x > 5.0 {
        let xi = 2.0 / 3.0 * x.powf(1.5);
        xi.exp() / (PI.sqrt() * x.powf(0.25))
    } else {
        let c1 = 0.6149266274460007;  // Bi(0)
        let _c2 = 0.4482883573538264;  // Bi'(0)
        let mut f = c1;
        let mut a_term = c1;
        for n in 1..100 {
            let nf = n as f64;
            a_term *= x.powi(3) / ((3.0 * nf - 1.0) * (3.0 * nf));
            f += a_term;
            if a_term.abs() < 1e-15 { break; }
        }
        f
    }
}

// ═══════════════════════════════════════════════════════════════════
// BERNOULLI & EULER NUMBERS
// ═══════════════════════════════════════════════════════════════════

/// Bernoulli number B_n (via Akiyama-Tanigawa algorithm).
pub fn bernoulli_number(n: usize) -> f64 {
    if n == 1 { return -0.5; }
    if n % 2 != 0 && n > 1 { return 0.0; }
    let mut a = Vec::with_capacity(n + 1);
    for m in 0..=n {
        a.push(1.0 / (m + 1) as f64);
        for j in (1..=m).rev() {
            a[j - 1] = j as f64 * (a[j - 1] - a[j]);
        }
    }
    a[0]
}

/// Bernoulli polynomial B_n(x).
pub fn bernoulli_polynomial(n: usize, x: f64) -> f64 {
    let mut sum = 0.0;
    for k in 0..=n {
        sum += crate::combinatorics::binomial(n as u64, k as u64) * bernoulli_number(k) * x.powi((n - k) as i32);
    }
    sum
}

// ═══════════════════════════════════════════════════════════════════
// POLYLOGARITHM & CLAUSEN
// ═══════════════════════════════════════════════════════════════════

/// Polylogarithm Li_s(z) for |z| < 1.
pub fn polylog(s: f64, z: f64) -> f64 {
    if z.abs() >= 1.0 { return f64::NAN; }
    let mut sum = 0.0;
    for k in 1..500 {
        let term = z.powi(k as i32) / (k as f64).powf(s);
        sum += term;
        if term.abs() < 1e-15 * sum.abs().max(1e-15) { break; }
    }
    sum
}

/// Dilogarithm Li_2(z).
pub fn dilog(z: f64) -> f64 { polylog(2.0, z) }

/// Clausen function Cl_2(θ) = -∫₀^θ ln|2sin(t/2)| dt.
pub fn clausen(theta: f64) -> f64 {
    let mut sum = 0.0;
    for k in 1..200 {
        sum += (k as f64 * theta).sin() / (k as f64).powi(2);
    }
    sum
}

// ═══════════════════════════════════════════════════════════════════
// DEBYE FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

/// Debye function D_n(x) = (n/x^n) ∫₀^x t^n/(e^t - 1) dt.
pub fn debye(n: u32, x: f64) -> f64 {
    if x.abs() < 1e-10 { return 1.0; }
    let steps = 100;
    let h = x / steps as f64;
    let mut sum = 0.0;
    for i in 1..steps {
        let t = i as f64 * h;
        let et = t.exp();
        if et > 1.0 + 1e-15 { sum += t.powi(n as i32) / (et - 1.0); }
    }
    sum *= h;
    let nf = n as f64;
    nf / x.powi(n as i32) * sum
}

// ═══════════════════════════════════════════════════════════════════
// SPHERICAL HARMONICS (real form)
// ═══════════════════════════════════════════════════════════════════

/// Associated Legendre polynomial P_l^m(x) (unnormalized).
pub fn assoc_legendre(l: u32, m: u32, x: f64) -> f64 {
    if m > l { return 0.0; }
    let mut pmm = 1.0;
    if m > 0 {
        let somx2 = (1.0 - x * x).sqrt();
        let mut fact = 1.0;
        for _ in 0..m { pmm *= -fact * somx2; fact += 2.0; }
    }
    if l == m { return pmm; }
    let mut pmmp1 = x * (2 * m + 1) as f64 * pmm;
    if l == m + 1 { return pmmp1; }
    let mut pll = 0.0;
    for ll in m + 2..=l {
        pll = (x * (2 * ll - 1) as f64 * pmmp1 - (ll + m - 1) as f64 * pmm) / (ll - m) as f64;
        pmm = pmmp1;
        pmmp1 = pll;
    }
    pll
}

/// Real spherical harmonic Y_l^m(theta, phi).
pub fn spherical_harmonic(l: u32, m: i32, theta: f64, phi: f64) -> f64 {
    let am = m.unsigned_abs();
    if am > l { return 0.0; }
    let norm = {
        let num = (2 * l + 1) as f64 * crate::combinatorics::factorial((l - am) as u64);
        let den = 4.0 * PI * crate::combinatorics::factorial((l + am) as u64);
        (num / den).sqrt()
    };
    let plm = assoc_legendre(l, am, theta.cos());
    if m > 0 { norm * plm * (am as f64 * phi).cos() * std::f64::consts::SQRT_2 }
    else if m < 0 { norm * plm * (am as f64 * phi).sin() * std::f64::consts::SQRT_2 }
    else { norm * plm }
}

// ═══════════════════════════════════════════════════════════════════
// MISCELLANEOUS SPECIAL FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

/// Struve function H_0(x) (series).
pub fn struve_h0(x: f64) -> f64 {
    let mut sum = 0.0;
    for k in 0..50 {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let term = sign * (x / 2.0).powi(2 * k as i32 + 1)
            / (crate::combinatorics::factorial(k as u64) * crate::special_functions::gamma(k as f64 + 1.5));
        // Simplified: Γ(k + 3/2) = (2k+1)!!/(2^{k+1}) * √π
        sum += term;
        if term.abs() < 1e-15 { break; }
    }
    2.0 / PI * sum
}

/// Exponential integral Ei(x) for x > 0.
pub fn exponential_integral(x: f64) -> f64 {
    if x <= 0.0 { return f64::NEG_INFINITY; }
    if x < 40.0 {
        let euler = 0.5772156649015329;
        let mut sum = euler + x.ln();
        let mut term = x;
        for n in 1..100 {
            sum += term / (n * n) as f64;
            term *= x / (n + 1) as f64;
            if term.abs() < 1e-15 { break; }
        }
        sum
    } else {
        // Asymptotic: Ei(x) ~ e^x/x * Σ n!/x^n
        let mut sum = 0.0;
        let mut term = 1.0;
        for n in 0..20 {
            sum += term;
            term *= (n + 1) as f64 / x;
            if term.abs() < 1e-10 { break; }
        }
        x.exp() / x * sum
    }
}

/// Sine integral Si(x) = ∫₀^x sin(t)/t dt.
pub fn sine_integral(x: f64) -> f64 {
    let mut sum = 0.0;
    let mut term = x;
    for n in 0..100 {
        let k = 2 * n + 1;
        sum += term / k as f64;
        term *= -x * x / ((k + 1) * (k + 2)) as f64;
        if term.abs() < 1e-15 { break; }
    }
    sum
}

/// Cosine integral Ci(x) = γ + ln(x) + ∫₀^x (cos(t)-1)/t dt.
pub fn cosine_integral(x: f64) -> f64 {
    let euler = 0.5772156649015329;
    let mut sum = euler + x.ln();
    let mut term = -x * x / 4.0;
    sum += term;
    for n in 2..100 {
        let k = 2 * n;
        term *= -x * x / ((k - 1) * k) as f64;
        sum += term / k as f64;
        if term.abs() < 1e-15 { break; }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elliptic_k() {
        // K(0) = π/2
        assert!((elliptic_k(0.0) - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_elliptic_e() {
        // E(0) = π/2
        assert!((elliptic_e(0.0) - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyp2f1() {
        // 2F1(1, 1; 2; z) = -ln(1-z)/z
        let z = 0.5;
        let result = hyp2f1(1.0, 1.0, 2.0, z);
        let expected = -(1.0 - z).ln() / z;
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_hyp1f1() {
        // 1F1(1; 1; z) = e^z
        let z = 1.0;
        assert!((hyp1f1(1.0, 1.0, z) - z.exp()).abs() < 1e-6);
    }

    #[test]
    fn test_bernoulli() {
        assert!((bernoulli_number(0) - 1.0).abs() < 1e-10);
        assert!((bernoulli_number(1) - (-0.5)).abs() < 1e-10);
        assert!((bernoulli_number(2) - 1.0/6.0).abs() < 1e-6);
    }

    #[test]
    fn test_dilog() {
        // Li_2(0) = 0
        assert!(dilog(0.0).abs() < 1e-10);
        // Li_2(1) = π²/6 ≈ 1.6449
        assert!((polylog(2.0, 0.999) - PI * PI / 6.0).abs() < 0.01);
    }

    #[test]
    fn test_debye() {
        // D_1(0) = 1
        assert!((debye(1, 0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_assoc_legendre() {
        // P_0^0(x) = 1
        assert!((assoc_legendre(0, 0, 0.5) - 1.0).abs() < 1e-10);
        // P_1^0(x) = x
        assert!((assoc_legendre(1, 0, 0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sine_integral() {
        // Si(0) = 0
        assert!(sine_integral(0.0).abs() < 1e-10);
        // Si(π) ≈ 1.8519
        assert!((sine_integral(PI) - 1.8519).abs() < 0.01);
    }

    #[test]
    fn test_airy() {
        // Ai(0) ≈ 0.3550
        assert!((airy_ai(0.0) - 0.3550).abs() < 0.01);
    }
}
