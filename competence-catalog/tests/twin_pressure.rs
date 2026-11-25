use competence_catalog::twin::{contact_force, contact_pressure, force_limit_ok, pressure_limit_ok, Plant};

#[test]
fn pressure_is_force_over_area() {
    let p = Plant { dt: 0.1 };
    let f = contact_force(&p, &[1.0, 1.0, 1.0], 0.02, 100.0, 10.0);
    let pr = contact_pressure(&f, 0.0001).unwrap(); // 1 cm^2
    assert!((pr[2] - f[2] / 0.0001).abs() < 1e-9);
    assert!(pressure_limit_ok(&pr, pr[2] + 1.0));
    assert!(!pressure_limit_ok(&pr, pr[2] - 1.0));
}

#[test]
fn bad_area_refuses() {
    assert!(contact_pressure(&[1.0], 0.0).is_none());
    assert!(contact_pressure(&[1.0], -1.0).is_none());
    assert!(contact_pressure(&[1.0], f64::NAN).is_none());
}

#[test]
fn force_and_pressure_pair_gates_together() {
    let p = Plant { dt: 0.1 };
    let f = contact_force(&p, &[1.0, 1.0, 1.0], 0.02, 100.0, 10.0);
    let pr = contact_pressure(&f, 0.0001).unwrap();
    assert!(force_limit_ok(&f, 100.0) && pressure_limit_ok(&pr, 1_000_000.0));
    assert!(!pressure_limit_ok(&pr, 10.0));
    assert!(!pressure_limit_ok(&[f64::NAN], 10.0));
}
