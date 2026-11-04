//! Mathematical and scientific constants: pi, e, golden ratio,
//! physical constants, mathematical sequences, conversion factors.

// ═══════════════════════════════════════════════════════════════════
// MATHEMATICAL CONSTANTS
// ═══════════════════════════════════════════════════════════════════

pub const PI: f64 = std::f64::consts::PI;
pub const TAU: f64 = 2.0 * std::f64::consts::PI;
pub const E: f64 = std::f64::consts::E;
pub const SQRT_2: f64 = std::f64::consts::SQRT_2;
pub const SQRT_3: f64 = 1.7320508075688772;
pub const LN_2: f64 = std::f64::consts::LN_2;
pub const LN_10: f64 = std::f64::consts::LN_10;
pub const LOG2_E: f64 = std::f64::consts::LOG2_E;
pub const LOG10_E: f64 = std::f64::consts::LOG10_E;
pub const GOLDEN_RATIO: f64 = 1.618033988749895;
pub const SILVER_RATIO: f64 = 2.414213562373095;
pub const EULER_MASCHERONI: f64 = 0.5772156649015329;
pub const CATALAN: f64 = 0.9159655941772190;
pub const APERY: f64 = 1.2020569031595942; // ζ(3)
pub const KHINCHIN: f64 = 2.685452001065306;
pub const FEIGENBAUM_DELTA: f64 = 4.669201609102990;
pub const FEIGENBAUM_ALPHA: f64 = 2.502907875095892;
pub const PLASTIC_NUMBER: f64 = 1.324717957244746;
pub const GLAISHER_KINKELIN: f64 = 1.2824271291006226;
pub const INF: f64 = f64::INFINITY;
pub const NEG_INF: f64 = f64::NEG_INFINITY;
pub const NAN: f64 = f64::NAN;
pub const EPSILON: f64 = f64::EPSILON;

// ═══════════════════════════════════════════════════════════════════
// CONSTANT FUNCTIONS (for runtime access)
// ═══════════════════════════════════════════════════════════════════

pub fn pi() -> f64 { PI }
pub fn tau() -> f64 { TAU }
pub fn euler() -> f64 { E }
pub fn golden_ratio() -> f64 { GOLDEN_RATIO }
pub fn euler_mascheroni() -> f64 { EULER_MASCHERONI }
pub fn sqrt2() -> f64 { SQRT_2 }
pub fn sqrt3() -> f64 { SQRT_3 }
pub fn ln2() -> f64 { LN_2 }
pub fn ln10() -> f64 { LN_10 }
pub fn infinity() -> f64 { INF }
pub fn neg_infinity() -> f64 { NEG_INF }
pub fn nan() -> f64 { NAN }
pub fn epsilon() -> f64 { EPSILON }
pub fn catalan_constant() -> f64 { CATALAN }
pub fn apery_constant() -> f64 { APERY }

// ═══════════════════════════════════════════════════════════════════
// PHYSICAL CONSTANTS (see physics.rs for more)
// ═══════════════════════════════════════════════════════════════════

pub fn speed_of_light() -> f64 { crate::physics::C }
pub fn planck_constant() -> f64 { crate::physics::H }
pub fn boltzmann() -> f64 { crate::physics::KB }
pub fn avogadro() -> f64 { crate::physics::NA }
pub fn gas_constant() -> f64 { crate::physics::R_GAS }
pub fn elementary_charge() -> f64 { crate::physics::E_CHARGE }
pub fn electron_mass() -> f64 { crate::physics::M_ELECTRON }
pub fn proton_mass() -> f64 { crate::physics::M_PROTON }
pub fn gravitational_constant() -> f64 { crate::physics::G_GRAV }
pub fn standard_gravity() -> f64 { crate::physics::G_EARTH }
pub fn vacuum_permittivity() -> f64 { crate::physics::EPSILON_0 }
pub fn vacuum_permeability() -> f64 { crate::physics::MU_0 }
pub fn stefan_boltzmann() -> f64 { crate::physics::SIGMA_SB }
pub fn fine_structure_constant() -> f64 { crate::physics::ALPHA_FINE }
pub fn bohr_radius() -> f64 { crate::physics::A_BOHR }
pub fn atomic_mass_unit() -> f64 { crate::physics::AMU }

// ═══════════════════════════════════════════════════════════════════
// CONVERSION CONSTANTS
// ═══════════════════════════════════════════════════════════════════

pub fn deg_to_rad_factor() -> f64 { PI / 180.0 }
pub fn rad_to_deg_factor() -> f64 { 180.0 / PI }
pub fn km_per_mile() -> f64 { 1.609344 }
pub fn lb_per_kg() -> f64 { 2.20462 }
pub fn inch_per_cm() -> f64 { 0.393701 }
pub fn gallon_per_liter() -> f64 { 0.264172 }
pub fn atm_in_pa() -> f64 { 101325.0 }
pub fn cal_in_joule() -> f64 { 4.184 }
pub fn ev_in_joule() -> f64 { 1.602176634e-19 }
pub fn c_in_kelvin() -> f64 { 273.15 }
pub fn light_year_in_meters() -> f64 { 9.461e15 }
pub fn au_in_meters() -> f64 { 1.496e11 }
pub fn parsec_in_meters() -> f64 { 3.086e16 }
pub fn earth_radius_m() -> f64 { 6_371_000.0 }
pub fn earth_mass_kg() -> f64 { 5.972e24 }
pub fn sun_mass_kg() -> f64 { 1.989e30 }
pub fn sun_luminosity_w() -> f64 { 3.828e26 }
pub fn day_in_seconds() -> f64 { 86400.0 }
pub fn year_in_seconds() -> f64 { 365.25 * 86400.0 }
pub fn byte_in_bits() -> f64 { 8.0 }
pub fn kb_in_bytes() -> f64 { 1024.0 }
pub fn mb_in_bytes() -> f64 { 1048576.0 }
pub fn gb_in_bytes() -> f64 { 1073741824.0 }

// ═══════════════════════════════════════════════════════════════════
// COMMON SEQUENCES
// ═══════════════════════════════════════════════════════════════════

/// First n Fibonacci numbers.
pub fn fibonacci_seq(n: usize) -> Vec<f64> {
    if n == 0 { return vec![]; }
    let mut fib = vec![0.0; n];
    if n >= 1 { fib[0] = 0.0; }
    if n >= 2 { fib[1] = 1.0; }
    for i in 2..n { fib[i] = fib[i-1] + fib[i-2]; }
    fib
}

/// First n prime numbers.
pub fn primes_seq(n: usize) -> Vec<f64> {
    let limit = if n < 6 { 15 } else { (n as f64 * (n as f64).ln() * 1.3) as usize };
    crate::combinatorics::sieve_primes(limit).into_iter().take(n).map(|p| p as f64).collect()
}

/// First n triangular numbers: T(k) = k(k+1)/2.
pub fn triangular_numbers(n: usize) -> Vec<f64> {
    (1..=n).map(|k| k as f64 * (k + 1) as f64 / 2.0).collect()
}

/// First n factorial numbers: 0!, 1!, 2!, ...
pub fn factorial_seq(n: usize) -> Vec<f64> {
    (0..n).map(|k| crate::combinatorics::factorial(k as u64)).collect()
}

/// First n Catalan numbers.
pub fn catalan_seq(n: usize) -> Vec<f64> {
    (0..n).map(|k| crate::combinatorics::catalan(k as u64)).collect()
}

/// First n Bell numbers.
pub fn bell_seq(n: usize) -> Vec<f64> {
    (0..n).map(|k| crate::combinatorics::bell(k)).collect()
}

/// First n powers of 2: 1, 2, 4, 8, ...
pub fn powers_of_2(n: usize) -> Vec<f64> {
    (0..n).map(|k| 2.0f64.powi(k as i32)).collect()
}

/// Harmonic series partial sums: H_1, H_2, ..., H_n.
pub fn harmonic_series(n: usize) -> Vec<f64> {
    let mut sums = Vec::with_capacity(n);
    let mut s = 0.0;
    for k in 1..=n { s += 1.0 / k as f64; sums.push(s); }
    sums
}

/// Bernoulli numbers B_0, B_1, ..., B_n.
pub fn bernoulli_seq(n: usize) -> Vec<f64> {
    (0..n).map(|k| crate::special_extended::bernoulli_number(k)).collect()
}

/// Look up constant by name.
pub fn constant_by_name(name: &str) -> Option<f64> {
    match name.to_lowercase().as_str() {
        "pi" => Some(PI),
        "tau" => Some(TAU),
        "e" | "euler" => Some(E),
        "phi" | "golden" | "golden_ratio" => Some(GOLDEN_RATIO),
        "sqrt2" => Some(SQRT_2),
        "sqrt3" => Some(SQRT_3),
        "ln2" => Some(LN_2),
        "ln10" => Some(LN_10),
        "gamma" | "euler_mascheroni" => Some(EULER_MASCHERONI),
        "catalan" => Some(CATALAN),
        "apery" => Some(APERY),
        "c" | "speed_of_light" => Some(crate::physics::C),
        "h" | "planck" => Some(crate::physics::H),
        "kb" | "boltzmann" => Some(crate::physics::KB),
        "na" | "avogadro" => Some(crate::physics::NA),
        "g" | "gravity" => Some(crate::physics::G_EARTH),
        "epsilon0" => Some(crate::physics::EPSILON_0),
        "mu0" => Some(crate::physics::MU_0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert!((pi() - std::f64::consts::PI).abs() < 1e-15);
        assert!((golden_ratio() - 1.618).abs() < 0.001);
    }

    #[test]
    fn test_fibonacci() {
        let f = fibonacci_seq(8);
        assert_eq!(f, vec![0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0]);
    }

    #[test]
    fn test_primes() {
        let p = primes_seq(5);
        assert_eq!(p, vec![2.0, 3.0, 5.0, 7.0, 11.0]);
    }

    #[test]
    fn test_triangular() {
        assert_eq!(triangular_numbers(4), vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_harmonic() {
        let h = harmonic_series(3);
        assert!((h[0] - 1.0).abs() < 1e-10);
        assert!((h[1] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_lookup() {
        assert_eq!(constant_by_name("pi"), Some(PI));
        assert_eq!(constant_by_name("unknown"), None);
    }

    #[test]
    fn test_physical() {
        assert!(speed_of_light() > 2.99e8);
        assert!(planck_constant() > 0.0);
    }

    #[test]
    fn test_powers() {
        assert_eq!(powers_of_2(4), vec![1.0, 2.0, 4.0, 8.0]);
    }
}
