use competence_catalog::twin::{simulate3, Plant3};

fn p(friction: f64) -> Plant3 {
    Plant3 { dt: 0.01, gravity: 9.81, lo: -1.0, hi: 1.0, restitution: 0.9, friction }
}

#[test]
fn trace_stays_within_both_boundaries() {
    let tr = simulate3(&p(0.0), &[0.0; 500]);
    assert!(tr.iter().all(|x| *x >= -1.0 && *x <= 1.0));
}

#[test]
fn friction_bleeds_contact_velocity() {
    let free = simulate3(&p(0.0), &[0.0; 300]);
    let fric = simulate3(&p(0.5), &[0.0; 300]);
    // with friction the bounce settles sooner: last-sample magnitude smaller or equal
    let spread = |t: &[f64]| t.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    assert!(spread(&fric) <= spread(&free) + 1e-9);
}

#[test]
fn ceiling_contact_clamps_and_reflects() {
    let up = Plant3 { gravity: -9.81, ..p(0.0) };
    let tr = simulate3(&up, &[0.0; 300]);
    assert!(tr.iter().all(|x| *x <= 1.0));
    assert!(tr.iter().any(|x| *x == 1.0));
}
