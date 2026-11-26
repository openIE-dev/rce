use competence_catalog::scale::reuse_curve;
use competence_catalog::twin::{collide_obb, Ball, Obb};

#[test]
fn oblique_contact_induces_spin_and_orientation_integrates() {
    let mut b = Ball::new(0.4, -0.4, 1.0, -1.0, 0.5, 1.0);
    let o = Obb { cx: 1.0, cy: -1.0, hx: 0.5, hy: 0.5, angle: 0.3 };
    collide_obb(&mut b, &o, 0.5, 0.2);
    assert!(b.omega.abs() > 1e-9, "spin {}", b.omega);
    let before = b.theta;
    b.theta += b.omega * 0.01;
    assert_ne!(b.theta, before);
}

#[test]
fn oriented_box_normal_follows_orientation() {
    // ball approaching a rotated box from above; after resolution it must be outside
    let mut b = Ball::new(0.0, 0.0, 0.0, -1.0, 0.5, 1.0);
    let o = Obb { cx: 0.0, cy: -0.6, hx: 2.0, hy: 0.2, angle: 0.5 };
    collide_obb(&mut b, &o, 0.5, 0.0);
    // distance from box center along local y must exceed hy + r - eps
    let (s, c) = (o.angle.sin(), o.angle.cos());
    let dx = b.x - o.cx;
    let dy = b.y - o.cy;
    let ly = -dx * s + dy * c;
    assert!(ly >= o.hy + b.r - 1e-6, "ly {ly}");
}

#[test]
fn reuse_curve_cost_falls_as_competence_grows() {
    let owned: Vec<String> = (0..100).map(|i| format!("k{i}")).collect();
    // Zipf-ish task stream over the first 100 keys
    let tasks: Vec<String> = (0..1000).map(|i| format!("k{}", (i * 7) % 100)).collect();
    let curve = reuse_curve(&owned, &tasks, &[0, 10, 50, 100]);
    assert_eq!(curve.len(), 4);
    assert!(curve[0].1 < curve[3].1, "hit rate rises");
    assert!(curve[3].2 < curve[0].2, "cost/act falls");
    assert_eq!(curve[3].1, 1.0);
    assert_eq!(curve[3].2, 1.0);
}

#[test]
fn tracing_jit_compiles_once_and_reuses() {
    let mut j = competence_catalog::jit::TracingJit::new();
    let mut last = String::new();
    for _ in 0..500 {
        let (v, _) = j.run("2+3*4").unwrap();
        last = v;
    }
    assert_eq!(last, "14");
    assert_eq!(j.subtree_compiles, 1, "compile once, reuse 500x");
}
