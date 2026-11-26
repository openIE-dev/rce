use competence_catalog::lang_synth::{run_native, synthesize_for_capability, KT_LN2_300K};

#[test]
fn quantum_measure_deterministic_and_entangle_correlates() {
    let ops = vec![("superpose", vec![0.0]), ("entangle", vec![0.0, 1.0]), ("measure", vec![0.0])];
    let a = run_native("quantum", &ops, 42);
    let b = run_native("quantum", &ops, 42);
    assert_eq!(a.outputs, b.outputs, "seeded Born measurement reproducible");
    assert!(a.ok);
    assert_eq!(a.outputs[1], 1.0, "entangle yields correlation");
    // theta=0 => p0=1 => measure always 0
    assert_eq!(a.outputs[2], 0.0);
}

#[test]
fn quantum_no_clone_refused() {
    let r = run_native("quantum", &[("clone", vec![0.0])], 1);
    assert!(!r.ok);
    assert!(r.err.unwrap().contains("no-clone"));
}

#[test]
fn thermodynamic_landauer_bound_and_reverse() {
    let r = run_native("thermodynamic", &[("landauer", vec![3.0])], 0);
    assert!(r.energy_j >= 3.0 * KT_LN2_300K * 0.999, "Landauer bound respected");
    let rev = run_native("thermodynamic", &[("landauer", vec![3.0]), ("reverse", vec![])], 0);
    assert!(rev.energy_j < r.energy_j, "reversible computing reclaims dissipation");
    let over = run_native("thermodynamic", &[("landauer", vec![1e22]), ("entropy_budget", vec![1.0])], 0);
    assert!(!over.ok, "budget exceeded refused");
}

#[test]
fn spatial_route_latency_and_neuromorphic_spikes() {
    let sp = run_native("spatial", &[("place", vec![2.0]), ("route", vec![5.0])], 0);
    assert_eq!(sp.outputs, vec![2.0, 5.0]);
    let nm = run_native("neuromorphic", &[("spike", vec![10.0]), ("spike", vec![20.0]), ("plasticity", vec![10.0])], 0);
    assert_eq!(nm.events.len(), 2);
    assert!(nm.outputs[2] < 1.0, "STDP decay");
}

#[test]
fn probabilistic_reproducible_and_analog_mac() {
    let p1 = run_native("probabilistic", &[("sample", vec![0.7]), ("sample", vec![0.7])], 7);
    let p2 = run_native("probabilistic", &[("sample", vec![0.7]), ("sample", vec![0.7])], 7);
    assert_eq!(p1.outputs, p2.outputs);
    let an = run_native("analog", &[("mac", vec![1.0, 2.0, 3.0, 4.0])], 0);
    assert_eq!(an.outputs[0], 1.0 * 2.0 + 3.0 * 4.0);
}

#[test]
fn gap_driven_synthesis_provides_missing_capability() {
    let q = synthesize_for_capability("entangle");
    assert_eq!(q.fabric, "quantum");
    assert!(q.constructs.contains(&"entangle".to_string()));
    assert_eq!(q.coverage(), 1.0);
    assert!(q.sample_replays());
    let t = synthesize_for_capability("landauer");
    assert_eq!(t.fabric, "thermodynamic");
    let a = synthesize_for_capability("mac");
    assert_eq!(a.fabric, "analog");
}
