//! Unit conversions: length, mass, time, temperature, energy, pressure,
//! speed, area, volume, data storage, angle, force.

// ═══════════════════════════════════════════════════════════════════
// LENGTH
// ═══════════════════════════════════════════════════════════════════

pub fn meters_to_feet(m: f64) -> f64 { m * 3.28084 }
pub fn feet_to_meters(ft: f64) -> f64 { ft / 3.28084 }
pub fn meters_to_inches(m: f64) -> f64 { m * 39.3701 }
pub fn inches_to_meters(inches: f64) -> f64 { inches / 39.3701 }
pub fn meters_to_miles(m: f64) -> f64 { m / 1609.344 }
pub fn miles_to_meters(mi: f64) -> f64 { mi * 1609.344 }
pub fn meters_to_km(m: f64) -> f64 { m / 1000.0 }
pub fn km_to_meters(km: f64) -> f64 { km * 1000.0 }
pub fn meters_to_yards(m: f64) -> f64 { m * 1.09361 }
pub fn yards_to_meters(yd: f64) -> f64 { yd / 1.09361 }
pub fn meters_to_nautical_miles(m: f64) -> f64 { m / 1852.0 }
pub fn nautical_miles_to_meters(nm: f64) -> f64 { nm * 1852.0 }
pub fn meters_to_cm(m: f64) -> f64 { m * 100.0 }
pub fn cm_to_meters(cm: f64) -> f64 { cm / 100.0 }
pub fn meters_to_mm(m: f64) -> f64 { m * 1000.0 }
pub fn mm_to_meters(mm: f64) -> f64 { mm / 1000.0 }
pub fn meters_to_um(m: f64) -> f64 { m * 1e6 }
pub fn um_to_meters(um: f64) -> f64 { um * 1e-6 }
pub fn meters_to_nm(m: f64) -> f64 { m * 1e9 }
pub fn nm_to_meters(nm: f64) -> f64 { nm * 1e-9 }
pub fn light_years_to_meters(ly: f64) -> f64 { ly * 9.461e15 }
pub fn meters_to_light_years(m: f64) -> f64 { m / 9.461e15 }
pub fn au_to_meters(au: f64) -> f64 { au * 1.496e11 }
pub fn meters_to_au(m: f64) -> f64 { m / 1.496e11 }
pub fn parsec_to_meters(pc: f64) -> f64 { pc * 3.086e16 }
pub fn meters_to_parsec(m: f64) -> f64 { m / 3.086e16 }

// ═══════════════════════════════════════════════════════════════════
// MASS
// ═══════════════════════════════════════════════════════════════════

pub fn kg_to_pounds(kg: f64) -> f64 { kg * 2.20462 }
pub fn pounds_to_kg(lb: f64) -> f64 { lb / 2.20462 }
pub fn kg_to_ounces(kg: f64) -> f64 { kg * 35.274 }
pub fn ounces_to_kg(oz: f64) -> f64 { oz / 35.274 }
pub fn kg_to_grams(kg: f64) -> f64 { kg * 1000.0 }
pub fn grams_to_kg(g: f64) -> f64 { g / 1000.0 }
pub fn kg_to_mg(kg: f64) -> f64 { kg * 1e6 }
pub fn mg_to_kg(mg: f64) -> f64 { mg * 1e-6 }
pub fn kg_to_tons(kg: f64) -> f64 { kg / 1000.0 }
pub fn tons_to_kg(t: f64) -> f64 { t * 1000.0 }
pub fn kg_to_stones(kg: f64) -> f64 { kg / 6.35029 }
pub fn stones_to_kg(st: f64) -> f64 { st * 6.35029 }

// ═══════════════════════════════════════════════════════════════════
// TIME
// ═══════════════════════════════════════════════════════════════════

pub fn seconds_to_minutes(s: f64) -> f64 { s / 60.0 }
pub fn minutes_to_seconds(m: f64) -> f64 { m * 60.0 }
pub fn seconds_to_hours(s: f64) -> f64 { s / 3600.0 }
pub fn hours_to_seconds(h: f64) -> f64 { h * 3600.0 }
pub fn seconds_to_days(s: f64) -> f64 { s / 86400.0 }
pub fn days_to_seconds(d: f64) -> f64 { d * 86400.0 }
pub fn seconds_to_ms(s: f64) -> f64 { s * 1000.0 }
pub fn ms_to_seconds(ms: f64) -> f64 { ms / 1000.0 }
pub fn seconds_to_us(s: f64) -> f64 { s * 1e6 }
pub fn us_to_seconds(us: f64) -> f64 { us * 1e-6 }
pub fn seconds_to_ns(s: f64) -> f64 { s * 1e9 }
pub fn ns_to_seconds(ns: f64) -> f64 { ns * 1e-9 }
pub fn days_to_years(d: f64) -> f64 { d / 365.25 }
pub fn years_to_days(y: f64) -> f64 { y * 365.25 }

// ═══════════════════════════════════════════════════════════════════
// SPEED
// ═══════════════════════════════════════════════════════════════════

pub fn mps_to_kph(mps: f64) -> f64 { mps * 3.6 }
pub fn kph_to_mps(kph: f64) -> f64 { kph / 3.6 }
pub fn mps_to_mph(mps: f64) -> f64 { mps * 2.23694 }
pub fn mph_to_mps(mph: f64) -> f64 { mph / 2.23694 }
pub fn mps_to_knots(mps: f64) -> f64 { mps * 1.94384 }
pub fn knots_to_mps(kn: f64) -> f64 { kn / 1.94384 }
pub fn mps_to_mach(mps: f64) -> f64 { mps / 343.0 }
pub fn mach_to_mps(mach: f64) -> f64 { mach * 343.0 }

// ═══════════════════════════════════════════════════════════════════
// ENERGY
// ═══════════════════════════════════════════════════════════════════

pub fn joules_to_kwh(j: f64) -> f64 { j / 3.6e6 }
pub fn kwh_to_joules(kwh: f64) -> f64 { kwh * 3.6e6 }
pub fn joules_to_btu(j: f64) -> f64 { j / 1055.06 }
pub fn btu_to_joules(btu: f64) -> f64 { btu * 1055.06 }
pub fn joules_to_erg(j: f64) -> f64 { j * 1e7 }
pub fn erg_to_joules(erg: f64) -> f64 { erg * 1e-7 }
pub fn watts_to_hp(w: f64) -> f64 { w / 745.7 }
pub fn hp_to_watts(hp: f64) -> f64 { hp * 745.7 }

// ═══════════════════════════════════════════════════════════════════
// PRESSURE
// ═══════════════════════════════════════════════════════════════════

pub fn pa_to_bar(pa: f64) -> f64 { pa / 1e5 }
pub fn bar_to_pa(bar: f64) -> f64 { bar * 1e5 }
pub fn pa_to_psi(pa: f64) -> f64 { pa / 6894.76 }
pub fn psi_to_pa(psi: f64) -> f64 { psi * 6894.76 }
pub fn pa_to_mmhg(pa: f64) -> f64 { pa / 133.322 }
pub fn mmhg_to_pa(mmhg: f64) -> f64 { mmhg * 133.322 }
pub fn pa_to_atm(pa: f64) -> f64 { pa / 101325.0 }
pub fn atm_to_pa(atm: f64) -> f64 { atm * 101325.0 }

// ═══════════════════════════════════════════════════════════════════
// AREA
// ═══════════════════════════════════════════════════════════════════

pub fn sqm_to_sqft(m2: f64) -> f64 { m2 * 10.7639 }
pub fn sqft_to_sqm(ft2: f64) -> f64 { ft2 / 10.7639 }
pub fn sqm_to_acres(m2: f64) -> f64 { m2 / 4046.86 }
pub fn acres_to_sqm(ac: f64) -> f64 { ac * 4046.86 }
pub fn sqm_to_hectares(m2: f64) -> f64 { m2 / 10000.0 }
pub fn hectares_to_sqm(ha: f64) -> f64 { ha * 10000.0 }

// ═══════════════════════════════════════════════════════════════════
// VOLUME
// ═══════════════════════════════════════════════════════════════════

pub fn liters_to_gallons(l: f64) -> f64 { l / 3.78541 }
pub fn gallons_to_liters(gal: f64) -> f64 { gal * 3.78541 }
pub fn liters_to_cubic_meters(l: f64) -> f64 { l / 1000.0 }
pub fn cubic_meters_to_liters(m3: f64) -> f64 { m3 * 1000.0 }
pub fn liters_to_ml(l: f64) -> f64 { l * 1000.0 }
pub fn ml_to_liters(ml: f64) -> f64 { ml / 1000.0 }
pub fn liters_to_cups(l: f64) -> f64 { l * 4.22675 }
pub fn cups_to_liters(cups: f64) -> f64 { cups / 4.22675 }
pub fn liters_to_pints(l: f64) -> f64 { l * 2.11338 }
pub fn pints_to_liters(pt: f64) -> f64 { pt / 2.11338 }

// ═══════════════════════════════════════════════════════════════════
// DATA STORAGE
// ═══════════════════════════════════════════════════════════════════

pub fn bytes_to_kb(b: f64) -> f64 { b / 1024.0 }
pub fn kb_to_bytes(kb: f64) -> f64 { kb * 1024.0 }
pub fn bytes_to_mb(b: f64) -> f64 { b / (1024.0 * 1024.0) }
pub fn mb_to_bytes(mb: f64) -> f64 { mb * 1024.0 * 1024.0 }
pub fn bytes_to_gb(b: f64) -> f64 { b / (1024.0 * 1024.0 * 1024.0) }
pub fn gb_to_bytes(gb: f64) -> f64 { gb * 1024.0 * 1024.0 * 1024.0 }
pub fn bytes_to_tb(b: f64) -> f64 { b / (1024.0 * 1024.0 * 1024.0 * 1024.0) }
pub fn tb_to_bytes(tb: f64) -> f64 { tb * 1024.0 * 1024.0 * 1024.0 * 1024.0 }
pub fn bits_to_bytes(bits: f64) -> f64 { bits / 8.0 }
pub fn bytes_to_bits(bytes: f64) -> f64 { bytes * 8.0 }

// ═══════════════════════════════════════════════════════════════════
// ANGLE
// ═══════════════════════════════════════════════════════════════════

pub fn degrees_to_radians(deg: f64) -> f64 { deg * std::f64::consts::PI / 180.0 }
pub fn radians_to_degrees(rad: f64) -> f64 { rad * 180.0 / std::f64::consts::PI }
pub fn degrees_to_gradians(deg: f64) -> f64 { deg * 10.0 / 9.0 }
pub fn gradians_to_degrees(grad: f64) -> f64 { grad * 9.0 / 10.0 }
pub fn degrees_to_turns(deg: f64) -> f64 { deg / 360.0 }
pub fn turns_to_degrees(turns: f64) -> f64 { turns * 360.0 }

// ═══════════════════════════════════════════════════════════════════
// FORCE
// ═══════════════════════════════════════════════════════════════════

pub fn newtons_to_lbf(n: f64) -> f64 { n / 4.44822 }
pub fn lbf_to_newtons(lbf: f64) -> f64 { lbf * 4.44822 }
pub fn newtons_to_dyne(n: f64) -> f64 { n * 1e5 }
pub fn dyne_to_newtons(dyne: f64) -> f64 { dyne * 1e-5 }
pub fn newtons_to_kgf(n: f64) -> f64 { n / 9.80665 }
pub fn kgf_to_newtons(kgf: f64) -> f64 { kgf * 9.80665 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_length_roundtrip() {
        let m = 100.0;
        assert!((feet_to_meters(meters_to_feet(m)) - m).abs() < 0.01);
        assert!((miles_to_meters(meters_to_miles(m)) - m).abs() < 0.01);
    }

    #[test]
    fn test_mass_roundtrip() {
        let kg = 75.0;
        assert!((pounds_to_kg(kg_to_pounds(kg)) - kg).abs() < 0.01);
    }

    #[test]
    fn test_speed() {
        assert!((mps_to_kph(1.0) - 3.6).abs() < 0.01);
    }

    #[test]
    fn test_energy() {
        assert!((kwh_to_joules(joules_to_kwh(1e6)) - 1e6).abs() < 1.0);
    }

    #[test]
    fn test_data() {
        assert!((gb_to_bytes(1.0) - 1073741824.0).abs() < 1.0);
    }

    #[test]
    fn test_angle() {
        assert!((degrees_to_radians(180.0) - std::f64::consts::PI).abs() < 1e-10);
    }
}
