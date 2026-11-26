use competence_catalog::twin::{simulate_n, Ball, WorldN};

fn w(rest: f64) -> WorldN {
    WorldN { dt: 0.001, g: 0.0, rest, lo: (0.0, 0.0), hi: (10.0, 10.0) }
}

fn head_on() -> Vec<Ball> {
    vec![
        Ball { x: 2.0, y: 5.0, vx: 1.0, vy: 0.0, r: 0.5, m: 1.0 },
        Ball { x: 4.0, y: 5.0, vx: -1.0, vy: 0.0, r: 0.5, m: 1.0 },
    ]
}

#[test]
fn equal_mass_head_on_exchanges_velocity() {
    let tr = simulate_n(&w(1.0), &head_on(), 1000);
    let last = tr.last().unwrap();
    // after elastic head-on collision equal masses swap velocities
    assert!(last[0].vx < 0.0, "{:?}", last[0]);
    assert!(last[1].vx > 0.0, "{:?}", last[1]);
}

#[test]
fn bodies_never_overlap_after_resolution() {
    let tr = simulate_n(&w(0.5), &head_on(), 3000);
    for frame in &tr {
        let dx = frame[1].x - frame[0].x;
        let dy = frame[1].y - frame[0].y;
        let dist = (dx * dx + dy * dy).sqrt();
        assert!(dist >= frame[0].r + frame[1].r - 1e-6, "overlap {dist}");
    }
}

#[test]
fn restitution_below_one_dissipates_kinetic_energy() {
    let ke = |f: &[Ball]| f.iter().map(|b| 0.5 * b.m * (b.vx * b.vx + b.vy * b.vy)).sum::<f64>();
    let elastic = simulate_n(&w(1.0), &head_on(), 3000);
    let lossy = simulate_n(&w(0.3), &head_on(), 3000);
    assert!(ke(lossy.last().unwrap()) < ke(elastic.last().unwrap()));
}

#[test]
fn three_body_pile_stays_in_bounds() {
    let s0 = vec![
        Ball { x: 3.0, y: 3.0, vx: 0.5, vy: 0.2, r: 0.4, m: 1.0 },
        Ball { x: 3.6, y: 3.2, vx: -0.3, vy: 0.1, r: 0.4, m: 2.0 },
        Ball { x: 3.3, y: 3.8, vx: 0.1, vy: -0.4, r: 0.4, m: 1.5 },
    ];
    let tr = simulate_n(&w(0.6), &s0, 2000);
    for f in &tr {
        for b in f {
            assert!(b.x >= 0.4 && b.x <= 9.6 && b.y >= 0.4 && b.y <= 9.6);
        }
    }
}
