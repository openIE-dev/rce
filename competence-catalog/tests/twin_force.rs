use competence_catalog::twin::{contact_force, force_limit_ok, Plant};

#[test]
fn pressing_past_the_wall_produces_force() {
    let p = Plant { dt: 0.1 };
    let f = contact_force(&p, &[1.0, 1.0, 1.0], 0.02, 100.0, 10.0);
    assert!(f[2] > 0.0);
    assert_eq!(f[0], 0.0); // not yet in contact
}

#[test]
fn force_limit_gate_admits_and_refuses() {
    let p = Plant { dt: 0.1 };
    let f = contact_force(&p, &[1.0, 1.0, 1.0], 0.02, 100.0, 10.0);
    assert!(force_limit_ok(&f, f[2] + 1.0));
    assert!(!force_limit_ok(&f, f[2] - 0.001));
    assert!(!force_limit_ok(&[f64::NAN], 100.0));
    assert!(!force_limit_ok(&[1.0], -1.0));
}

#[test]
fn free_motion_never_contacts() {
    let p = Plant { dt: 0.1 };
    let f = contact_force(&p, &[1.0, 1.0], 10.0, 100.0, 10.0);
    assert!(f.iter().all(|x| *x == 0.0));
    assert!(force_limit_ok(&f, 0.0));
}
