use competence_catalog::twin::{simulate_body, Body2, State2};

fn b(rest: f64, fric: f64) -> Body2 {
    Body2 { dt: 0.01, g: 9.81, radius: 0.1, lo: (0.0, 0.0), hi: (2.0, 2.0), rest, fric }
}

fn s0() -> State2 {
    State2 { x: 1.0, y: 1.5, vx: 0.3, vy: 0.0, spin: 0.0 }
}

#[test]
fn body_stays_inside_the_box() {
    let tr = simulate_body(&b(0.9, 0.0), &s0(), &[(0.0, 0.0); 2000]);
    assert!(tr.iter().all(|s| s.x >= 0.1 && s.x <= 1.9 && s.y >= 0.1 && s.y <= 1.9));
}

#[test]
fn restitution_below_one_dissipates_energy() {
    let tr = simulate_body(&b(0.5, 0.0), &s0(), &[(0.0, 0.0); 2000]);
    let speed = |s: &State2| (s.vx * s.vx + s.vy * s.vy).sqrt();
    let early = speed(&tr[100]);
    let late = speed(&tr[1999]);
    assert!(late < early);
}

#[test]
fn friction_induces_spin_on_contact() {
    let tr = simulate_body(&b(0.9, 0.2), &s0(), &[(0.0, 0.0); 2000]);
    assert!(tr.iter().any(|s| s.spin.abs() > 1e-9));
    let no_fric = simulate_body(&b(0.9, 0.0), &s0(), &[(0.0, 0.0); 2000]);
    assert!(no_fric.iter().all(|s| s.spin == 0.0));
}
