//! Physics constants and computations: SI units, mechanics,
//! thermodynamics, electromagnetism, optics, quantum, relativity.

use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════
// FUNDAMENTAL CONSTANTS (SI units, CODATA 2018)
// ═══════════════════════════════════════════════════════════════════

/// Speed of light in vacuum (m/s).
pub const C: f64 = 299_792_458.0;
/// Planck constant (J·s).
pub const H: f64 = 6.62607015e-34;
/// Reduced Planck constant ℏ (J·s).
pub const HBAR: f64 = 1.054571817e-34;
/// Boltzmann constant (J/K).
pub const KB: f64 = 1.380649e-23;
/// Avogadro constant (1/mol).
pub const NA: f64 = 6.02214076e23;
/// Gas constant (J/(mol·K)).
pub const R_GAS: f64 = 8.314462618;
/// Elementary charge (C).
pub const E_CHARGE: f64 = 1.602176634e-19;
/// Electron mass (kg).
pub const M_ELECTRON: f64 = 9.1093837015e-31;
/// Proton mass (kg).
pub const M_PROTON: f64 = 1.67262192369e-27;
/// Gravitational constant (m³/(kg·s²)).
pub const G_GRAV: f64 = 6.67430e-11;
/// Standard gravity (m/s²).
pub const G_EARTH: f64 = 9.80665;
/// Vacuum permittivity ε₀ (F/m).
pub const EPSILON_0: f64 = 8.8541878128e-12;
/// Vacuum permeability μ₀ (H/m).
pub const MU_0: f64 = 1.25663706212e-6;
/// Stefan-Boltzmann constant (W/(m²·K⁴)).
pub const SIGMA_SB: f64 = 5.670374419e-8;
/// Coulomb constant k_e (N·m²/C²).
pub const K_COULOMB: f64 = 8.9875517923e9;
/// Bohr radius (m).
pub const A_BOHR: f64 = 5.29177210903e-11;
/// Fine-structure constant α.
pub const ALPHA_FINE: f64 = 7.2973525693e-3;
/// Atomic mass unit (kg).
pub const AMU: f64 = 1.66053906660e-27;

// ═══════════════════════════════════════════════════════════════════
// UNIT CONVERSIONS
// ═══════════════════════════════════════════════════════════════════

/// Celsius to Kelvin.
pub fn celsius_to_kelvin(c: f64) -> f64 { c + 273.15 }
/// Kelvin to Celsius.
pub fn kelvin_to_celsius(k: f64) -> f64 { k - 273.15 }
/// Fahrenheit to Celsius.
pub fn fahrenheit_to_celsius(f: f64) -> f64 { (f - 32.0) * 5.0 / 9.0 }
/// Celsius to Fahrenheit.
pub fn celsius_to_fahrenheit(c: f64) -> f64 { c * 9.0 / 5.0 + 32.0 }
/// Electronvolts to Joules.
pub fn ev_to_joules(ev: f64) -> f64 { ev * E_CHARGE }
/// Joules to electronvolts.
pub fn joules_to_ev(j: f64) -> f64 { j / E_CHARGE }
/// Angstroms to meters.
pub fn angstrom_to_meters(a: f64) -> f64 { a * 1e-10 }
/// Meters to Angstroms.
pub fn meters_to_angstrom(m: f64) -> f64 { m * 1e10 }
/// Atmospheres to Pascals.
pub fn atm_to_pascals(atm: f64) -> f64 { atm * 101325.0 }
/// Pascals to atmospheres.
pub fn pascals_to_atm(pa: f64) -> f64 { pa / 101325.0 }
/// Calories to Joules.
pub fn cal_to_joules(cal: f64) -> f64 { cal * 4.184 }
/// Joules to calories.
pub fn joules_to_cal(j: f64) -> f64 { j / 4.184 }

// ═══════════════════════════════════════════════════════════════════
// MECHANICS
// ═══════════════════════════════════════════════════════════════════

/// Kinetic energy: KE = ½mv².
pub fn kinetic_energy(mass: f64, velocity: f64) -> f64 { 0.5 * mass * velocity * velocity }

/// Gravitational potential energy: PE = mgh.
pub fn gravitational_pe(mass: f64, height: f64) -> f64 { mass * G_EARTH * height }

/// Projectile range: R = v²sin(2θ)/g.
pub fn projectile_range(v: f64, angle: f64) -> f64 { v * v * (2.0 * angle).sin() / G_EARTH }

/// Projectile max height: H = v²sin²(θ)/(2g).
pub fn projectile_max_height(v: f64, angle: f64) -> f64 { v * v * angle.sin().powi(2) / (2.0 * G_EARTH) }

/// Projectile time of flight: T = 2v·sin(θ)/g.
pub fn projectile_time_of_flight(v: f64, angle: f64) -> f64 { 2.0 * v * angle.sin() / G_EARTH }

/// Orbital velocity: v = sqrt(GM/r).
pub fn orbital_velocity(m_central: f64, radius: f64) -> f64 { (G_GRAV * m_central / radius).sqrt() }

/// Orbital period: T = 2π·sqrt(r³/(GM)).
pub fn orbital_period(m_central: f64, radius: f64) -> f64 { 2.0 * PI * (radius.powi(3) / (G_GRAV * m_central)).sqrt() }

/// Escape velocity: v = sqrt(2GM/r).
pub fn escape_velocity(mass: f64, radius: f64) -> f64 { (2.0 * G_GRAV * mass / radius).sqrt() }

/// Centripetal acceleration: a = v²/r.
pub fn centripetal_acceleration(v: f64, r: f64) -> f64 { v * v / r }

/// Hooke's law spring force: F = -kx.
pub fn spring_force(k: f64, x: f64) -> f64 { -k * x }

/// Spring potential energy: PE = ½kx².
pub fn spring_pe(k: f64, x: f64) -> f64 { 0.5 * k * x * x }

/// Simple harmonic motion period: T = 2π·sqrt(m/k).
pub fn shm_period(mass: f64, k: f64) -> f64 { 2.0 * PI * (mass / k).sqrt() }

/// Pendulum period: T = 2π·sqrt(L/g).
pub fn pendulum_period(length: f64) -> f64 { 2.0 * PI * (length / G_EARTH).sqrt() }

// ═══════════════════════════════════════════════════════════════════
// THERMODYNAMICS
// ═══════════════════════════════════════════════════════════════════

/// Ideal gas: PV = nRT. Solve for one variable (pass 0.0 for unknown).
pub fn ideal_gas_pressure(n: f64, t: f64, v: f64) -> f64 { n * R_GAS * t / v }
pub fn ideal_gas_volume(n: f64, t: f64, p: f64) -> f64 { n * R_GAS * t / p }
pub fn ideal_gas_temperature(p: f64, v: f64, n: f64) -> f64 { p * v / (n * R_GAS) }

/// Heat transfer: Q = mcΔT.
pub fn heat_transfer(mass: f64, specific_heat: f64, delta_t: f64) -> f64 { mass * specific_heat * delta_t }

/// Carnot efficiency: η = 1 - T_cold/T_hot.
pub fn carnot_efficiency(t_hot: f64, t_cold: f64) -> f64 { 1.0 - t_cold / t_hot }

/// Stefan-Boltzmann radiation: P = σεAT⁴.
pub fn blackbody_power(emissivity: f64, area: f64, temperature: f64) -> f64 { SIGMA_SB * emissivity * area * temperature.powi(4) }

/// Wien's displacement law: λ_max = b/T.
pub fn wien_wavelength(temperature: f64) -> f64 { 2.897771955e-3 / temperature }

/// Planck spectral radiance B(ν,T).
pub fn planck_radiance(frequency: f64, temperature: f64) -> f64 {
    let x = H * frequency / (KB * temperature);
    if x > 500.0 { return 0.0; }
    2.0 * H * frequency.powi(3) / (C * C * (x.exp() - 1.0))
}

// ═══════════════════════════════════════════════════════════════════
// ELECTROMAGNETISM
// ═══════════════════════════════════════════════════════════════════

/// Coulomb's law: F = k·q1·q2/r².
pub fn coulomb_force(q1: f64, q2: f64, r: f64) -> f64 { K_COULOMB * q1 * q2 / (r * r) }

/// Electric field from point charge: E = k·q/r².
pub fn electric_field_point(q: f64, r: f64) -> f64 { K_COULOMB * q / (r * r) }

/// Capacitor energy: E = ½CV².
pub fn capacitor_energy(capacitance: f64, voltage: f64) -> f64 { 0.5 * capacitance * voltage * voltage }

/// Inductor energy: E = ½LI².
pub fn inductor_energy(inductance: f64, current: f64) -> f64 { 0.5 * inductance * current * current }

/// Ohm's law: V = IR.
pub fn ohms_law_voltage(current: f64, resistance: f64) -> f64 { current * resistance }
pub fn ohms_law_current(voltage: f64, resistance: f64) -> f64 { voltage / resistance }
pub fn ohms_law_resistance(voltage: f64, current: f64) -> f64 { voltage / current }

/// Power: P = IV.
pub fn electrical_power(current: f64, voltage: f64) -> f64 { current * voltage }

/// Resistors in series.
pub fn resistors_series(resistances: &[f64]) -> f64 { resistances.iter().sum() }

/// Resistors in parallel.
pub fn resistors_parallel(resistances: &[f64]) -> f64 {
    1.0 / resistances.iter().map(|r| 1.0 / r).sum::<f64>()
}

/// RC time constant.
pub fn rc_time_constant(r: f64, c: f64) -> f64 { r * c }

/// LC resonant frequency.
pub fn lc_resonant_frequency(l: f64, c: f64) -> f64 { 1.0 / (2.0 * PI * (l * c).sqrt()) }

// ═══════════════════════════════════════════════════════════════════
// OPTICS & WAVES
// ═══════════════════════════════════════════════════════════════════

/// Snell's law: n1·sin(θ1) = n2·sin(θ2). Returns θ2.
pub fn snell_law(n1: f64, n2: f64, theta1: f64) -> f64 { (n1 * theta1.sin() / n2).asin() }

/// Critical angle for total internal reflection.
pub fn critical_angle(n1: f64, n2: f64) -> f64 { (n2 / n1).asin() }

/// Thin lens equation: 1/f = 1/do + 1/di. Solve for image distance.
pub fn lens_image_distance(focal_length: f64, object_distance: f64) -> f64 {
    1.0 / (1.0 / focal_length - 1.0 / object_distance)
}

/// Magnification: M = -di/do.
pub fn lens_magnification(object_distance: f64, image_distance: f64) -> f64 { -image_distance / object_distance }

/// Wavelength from frequency: λ = c/f.
pub fn wavelength_from_freq(frequency: f64) -> f64 { C / frequency }

/// Frequency from wavelength: f = c/λ.
pub fn freq_from_wavelength(wavelength: f64) -> f64 { C / wavelength }

/// Doppler shift (source moving toward observer): f' = f·c/(c - v_s).
pub fn doppler_frequency(f0: f64, v_source: f64) -> f64 { f0 * C / (C - v_source) }

/// Diffraction grating: d·sin(θ) = mλ. Returns angle.
pub fn diffraction_angle(d: f64, wavelength: f64, order: i32) -> f64 { (order as f64 * wavelength / d).asin() }

// ═══════════════════════════════════════════════════════════════════
// QUANTUM / RELATIVITY
// ═══════════════════════════════════════════════════════════════════

/// Photon energy: E = hf.
pub fn photon_energy(frequency: f64) -> f64 { H * frequency }

/// De Broglie wavelength: λ = h/(mv).
pub fn de_broglie_wavelength(mass: f64, velocity: f64) -> f64 { H / (mass * velocity) }

/// Lorentz factor: γ = 1/sqrt(1 - v²/c²).
pub fn lorentz_factor(v: f64) -> f64 { 1.0 / (1.0 - (v / C).powi(2)).sqrt() }

/// Time dilation: Δt' = γΔt.
pub fn time_dilation(proper_time: f64, v: f64) -> f64 { lorentz_factor(v) * proper_time }

/// Length contraction: L' = L/γ.
pub fn length_contraction(proper_length: f64, v: f64) -> f64 { proper_length / lorentz_factor(v) }

/// Relativistic kinetic energy: KE = (γ-1)mc².
pub fn relativistic_ke(mass: f64, v: f64) -> f64 { (lorentz_factor(v) - 1.0) * mass * C * C }

/// Mass-energy equivalence: E = mc².
pub fn mass_energy(mass: f64) -> f64 { mass * C * C }

/// Schwarzschild radius: r_s = 2GM/c².
pub fn schwarzschild_radius(mass: f64) -> f64 { 2.0 * G_GRAV * mass / (C * C) }

/// Hydrogen atom energy levels: E_n = -13.6 eV / n².
pub fn hydrogen_energy_level(n: u32) -> f64 { -13.6 / (n * n) as f64 }

/// Rydberg formula: 1/λ = R_H (1/n1² - 1/n2²).
pub fn rydberg_wavelength(n1: u32, n2: u32) -> f64 {
    let r_h = 1.0973731568539e7; // Rydberg constant (1/m)
    1.0 / (r_h * (1.0 / (n1 * n1) as f64 - 1.0 / (n2 * n2) as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert!((C - 299_792_458.0).abs() < 1.0);
        assert!(H > 0.0 && H < 1.0);
    }

    #[test]
    fn test_temp_conversions() {
        assert!((celsius_to_kelvin(0.0) - 273.15).abs() < 1e-10);
        assert!((fahrenheit_to_celsius(32.0) - 0.0).abs() < 1e-10);
        assert!((celsius_to_fahrenheit(100.0) - 212.0).abs() < 1e-10);
    }

    #[test]
    fn test_kinetic_energy() {
        assert!((kinetic_energy(1.0, 10.0) - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_ideal_gas() {
        // 1 mol at 273.15 K, 1 atm ≈ 22.4 L
        let v = ideal_gas_volume(1.0, 273.15, 101325.0);
        assert!((v - 0.02241).abs() < 0.001);
    }

    #[test]
    fn test_carnot() {
        assert!((carnot_efficiency(400.0, 300.0) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_coulomb() {
        let f = coulomb_force(1e-6, 1e-6, 1.0);
        assert!((f - K_COULOMB * 1e-12).abs() < 1e-6);
    }

    #[test]
    fn test_snell() {
        // Normal incidence: θ1 = 0 → θ2 = 0
        assert!(snell_law(1.0, 1.5, 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_photon_energy() {
        // Visible light ~5e14 Hz
        let e = photon_energy(5e14);
        assert!(e > 0.0);
    }

    #[test]
    fn test_lorentz() {
        assert!((lorentz_factor(0.0) - 1.0).abs() < 1e-10);
        assert!(lorentz_factor(0.9 * C) > 2.0);
    }

    #[test]
    fn test_mass_energy() {
        // 1 kg → huge energy
        let e = mass_energy(1.0);
        assert!(e > 8e16);
    }

    #[test]
    fn test_hydrogen_energy() {
        assert!((hydrogen_energy_level(1) - (-13.6)).abs() < 1e-10);
        assert!((hydrogen_energy_level(2) - (-3.4)).abs() < 1e-10);
    }

    #[test]
    fn test_pendulum() {
        // 1m pendulum ≈ 2.006 s period
        let t = pendulum_period(1.0);
        assert!((t - 2.006).abs() < 0.01);
    }

    #[test]
    fn test_resistors() {
        assert!((resistors_series(&[100.0, 200.0]) - 300.0).abs() < 1e-10);
        assert!((resistors_parallel(&[100.0, 100.0]) - 50.0).abs() < 1e-10);
    }
}
