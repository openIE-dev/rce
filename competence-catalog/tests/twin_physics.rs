use competence_catalog::twin::{divergence, identify2, simulate2, Plant2};

fn base() -> Plant2 {
    Plant2 { dt: 0.01, gravity: 9.81, drag: 0.0, wall: -1.0, restitution: 0.5 }
}

#[test]
fn free_fall_accelerates_downward() {
    let p = Plant2 { drag: 0.0, wall: -100.0, ..base() };
    let trace = simulate2(&p, &[0.0; 10]);
    assert!(trace[9] < trace[4] && trace[4] < trace[0]);
}

#[test]
fn drag_limits_terminal_velocity() {
    let no_drag = Plant2 { drag: 0.0, wall: -1000.0, ..base() };
    let with_drag = Plant2 { drag: 5.0, wall: -1000.0, ..base() };
    let a = simulate2(&no_drag, &[0.0; 200]);
    let b = simulate2(&with_drag, &[0.0; 200]);
    assert!(b[199] > a[199]); // drag slows the fall (less negative)
}

#[test]
fn restitution_bounces_off_the_floor() {
    let p = Plant2 { wall: -1.0, restitution: 0.9, ..base() };
    let trace = simulate2(&p, &[0.0; 300]);
    assert!(trace.iter().all(|x| *x >= -1.0)); // never below floor
    let mut rose = false;
    for w in trace.windows(2) {
        if w[1] > w[0] { rose = true; }
    }
    assert!(rose); // bounced upward after impact
}

#[test]
fn zero_restitution_sticks_to_floor() {
    let p = Plant2 { wall: -1.0, restitution: 0.0, ..base() };
    let trace = simulate2(&p, &[0.0; 300]);
    assert!(trace[299] == -1.0);
}

#[test]
fn identify2_recovers_gravity_and_drag() {
    let truth = Plant2 { dt: 0.01, gravity: 9.81, drag: 1.0, wall: -1000.0, restitution: 0.5 };
    let acts = vec![0.0; 50];
    let rec = simulate2(&truth, &acts);
    let candidates = vec![
        Plant2 { gravity: 3.0, drag: 0.0, ..truth },
        truth,
        Plant2 { gravity: 9.81, drag: 4.0, ..truth },
    ];
    let (p, d) = identify2(&acts, &rec, &candidates, 1e-9).unwrap();
    assert_eq!(p.gravity, 9.81);
    assert_eq!(p.drag, 1.0);
    assert!(d < 1e-9);
}

#[test]
fn identify2_none_when_unexplained() {
    let truth = base();
    let acts = vec![0.0; 20];
    let rec = simulate2(&truth, &acts);
    let wrong = vec![Plant2 { gravity: 1.0, ..truth }];
    assert!(identify2(&acts, &rec, &wrong, 1e-9).is_none());
    assert_eq!(divergence(&rec, &rec), Some(0.0));
}
