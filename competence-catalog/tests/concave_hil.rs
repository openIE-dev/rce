use competence_catalog::hil::{certify, SafetyProps};
use competence_catalog::lift::lift_expr;
use competence_catalog::lut_emit::Stmt;
use competence_catalog::twin::{circle_concave, triangulate, Ball, Plant2};

fn l_shape() -> Vec<(f64, f64)> {
    vec![(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)]
}

#[test]
fn concave_triangulation_covers_n_minus_two() {
    let tris = triangulate(&l_shape());
    assert_eq!(tris.len(), 4); // 6 verts -> 4 triangles
}

#[test]
fn circle_in_concave_notch_resolves_outside() {
    let verts = l_shape();
    // ball inside the missing notch (1.5,1.5) should be pushed out of the L
    let mut b = Ball::new(1.5, 1.5, 0.0, 0.0, 0.2, 1.0);
    circle_concave(&mut b, &verts, 0.5, 0.0);
    let inside = {
        // point-in-polygon (ray cast)
        let mut ins = false;
        let n = verts.len();
        for i in 0..n {
            let (x1, y1) = verts[i];
            let (x2, y2) = verts[(i + 1) % n];
            if (y1 > b.y) != (y2 > b.y) {
                let xin = x1 + (b.y - y1) * (x2 - x1) / (y2 - y1);
                if b.x < xin {
                    ins = !ins;
                }
            }
        }
        ins
    };
    assert!(!inside, "ball still inside concave polygon");
}

fn props() -> SafetyProps {
    SafetyProps {
        lo: -10.0, hi: 10.0, wall: 0.0, k: 100.0, c: 10.0, area_m2: 0.0001,
        max_force_n: 1000.0, max_pressure_pa: 1e7, deadline_us: 1000,
    }
}

fn plant() -> Plant2 {
    Plant2 { dt: 0.01, gravity: 9.81, drag: 0.0, wall: -100.0, restitution: 0.5 }
}

#[test]
fn safe_controller_certifies() {
    // policy: act = 0 (free fall within bounds)
    let c = Stmt::Return(lift_expr("0").unwrap());
    let r = certify(&c, &plant(), &[(0.0, 0.0), (5.0, 0.0)], 100, &props());
    assert_eq!(r.episodes, 2);
    assert_eq!(r.passed, 2);
    assert_eq!(r.failed, 0);
    assert!(r.violations.is_empty(), "{:?}", r.violations);
    assert_eq!(r.trace_digest.len(), 64);
}

#[test]
fn unsafe_controller_fails_bounds() {
    // policy: act = +2000 (strong upward thrust escapes hi=10)
    let c = Stmt::Return(lift_expr("2000").unwrap());
    let r = certify(&c, &plant(), &[(0.0, 0.0)], 200, &props());
    assert_eq!(r.failed, 1);
    assert!(r.violations.iter().any(|v| v.contains("bounds")));
}

#[test]
fn deadline_violation_is_named() {
    let c = Stmt::Return(lift_expr("0").unwrap());
    let mut p = props();
    p.deadline_us = 1; // tighter than the G0 gear wcet (5us)
    let r = certify(&c, &plant(), &[(0.0, 0.0)], 5, &p);
    assert!(r.violations.iter().any(|v| v.contains("deadline")));
}
