use competence_catalog::twin::{admit_controller, divergence, simulate, Plant};

#[test]
fn double_integrator_trace_is_exact() {
    let p = Plant { dt: 0.1 };
    let trace = simulate(&p, &[1.0, 1.0, 1.0]);
    let expect = [0.01, 0.03, 0.06];
    assert!(trace.iter().zip(expect.iter()).all(|(a, b)| (a - b).abs() < 1e-12));
}

#[test]
fn coherent_controller_admitted() {
    let p = Plant { dt: 0.1 };
    let acts = vec![1.0, 1.0, 1.0];
    let recorded = simulate(&p, &acts);
    assert!(admit_controller(&p, &acts, &recorded, 1e-9));
}

#[test]
fn divergent_controller_refused() {
    let p = Plant { dt: 0.1 };
    let acts = vec![1.0, 1.0, 1.0];
    let mut recorded = simulate(&p, &acts);
    recorded[2] += 0.5;
    assert!(!admit_controller(&p, &acts, &recorded, 1e-3));
}

#[test]
fn length_mismatch_and_bad_tol_refuse() {
    let p = Plant { dt: 0.1 };
    assert!(!admit_controller(&p, &[1.0], &[0.01, 0.03], 1e-9));
    assert!(!admit_controller(&p, &[1.0], &[0.01], f64::NAN));
    assert_eq!(divergence(&[0.0], &[0.0, 1.0]), None);
}
