use competence_catalog::scale::federated_hier;
use competence_catalog::train::HierPredictor;
use competence_catalog::twin::{circle_poly, resolve_poly, sat_overlap, Ball, Poly};

fn square(cx: f64, cy: f64, h: f64) -> Poly {
    Poly {
        v: vec![(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)],
        m: 1.0, vx: 0.0, vy: 0.0, theta: 0.0, omega: 0.0,
    }
}

#[test]
fn circle_vs_triangle_resolves_outside() {
    let tri = Poly {
        v: vec![(0.0, 0.0), (2.0, 0.0), (1.0, 1.5)],
        m: 1.0, vx: 0.0, vy: 0.0, theta: 0.0, omega: 0.0,
    };
    let mut b = Ball::new(1.0, 0.2, 0.0, -1.0, 0.3, 1.0);
    circle_poly(&mut b, &tri, 0.5, 0.1);
    // after resolution the ball center must be outside the triangle
    let inside = {
        let mut sign = 0.0;
        let v = &tri.v;
        for i in 0..3 {
            let a = v[i];
            let c = v[(i + 1) % 3];
            let cross = (c.0 - a.0) * (b.y - a.1) - (c.1 - a.1) * (b.x - a.0);
            if cross.abs() > 1e-12 {
                if sign == 0.0 { sign = cross; } else if sign * cross < 0.0 { sign = f64::NAN; }
            }
        }
        !sign.is_nan()
    };
    assert!(!inside, "ball still inside triangle");
}

#[test]
fn sat_separates_overlapping_squares_and_detects_disjoint() {
    let a = square(0.0, 0.0, 1.0);
    let b = square(1.5, 0.0, 1.0);
    assert!(sat_overlap(&a, &b).is_some());
    let c = square(5.0, 0.0, 1.0);
    assert!(sat_overlap(&a, &c).is_none());
}

#[test]
fn resolve_poly_removes_overlap() {
    let mut a = square(0.0, 0.0, 1.0);
    let mut b = square(1.5, 0.0, 1.0);
    resolve_poly(&mut a, &mut b, 0.5);
    assert!(sat_overlap(&a, &b).is_none(), "still overlapping after resolve");
}

#[test]
fn world_model_rollout_tracks_linear_dynamics() {
    // learn y = 2x on 1-dim latents; rollout from 1 should double each step
    let pairs: Vec<(Vec<f64>, Vec<f64>)> = (1..200).map(|i| {
        let x = i as f64 / 100.0;
        (vec![x], vec![2.0 * x])
    }).collect();
    let h = HierPredictor::train(&pairs, 1, 300, 0.05);
    let roll = h.rollout(&[0.5], 3);
    assert!((roll[0][0] - 1.0).abs() < 0.1, "{roll:?}");
    assert!((roll[1][0] - 2.0).abs() < 0.3, "{roll:?}");
    assert!((roll[2][0] - 4.0).abs() < 0.8, "{roll:?}");
}

#[test]
fn federated_world_model_matches_centralized_direction() {
    let pairs: Vec<(Vec<f64>, Vec<f64>)> = (0..4000).map(|i| {
        let x = (i + 1) as f64 / 4000.0;
        (vec![x, x * x], vec![2.0 * x, x])
    }).collect();
    let shards: Vec<Vec<(Vec<f64>, Vec<f64>)>> = (0..4).map(|k| pairs[k * 1000..(k + 1) * 1000].to_vec()).collect();
    let fed = federated_hier(&shards, 2, 150, 0.05);
    let cent = HierPredictor::train(&pairs, 2, 150, 0.05);
    let x = vec![0.7, 0.49];
    let y = vec![1.4, 0.7];
    let ef = fed.energy(&x, &y);
    let ec = cent.energy(&x, &y);
    let wrong = vec![0.1, 0.9];
    println!("fed energy {ef:.4} cent {ec:.4} wrong {:.4}", fed.energy(&x, &wrong));
    assert!(ef < fed.energy(&x, &wrong), "fed should prefer true target");
    assert!(ef < 1.0 && ec < 1.0, "fed {ef} cent {ec}");
}
