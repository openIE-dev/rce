use competence_catalog::realtime::{measure_lookup, select, Decision, Loop};
use competence_catalog::Provenance;

#[test]
fn measured_lookup_is_a_reading_not_a_zero() {
    let m = measure_lookup(200);
    assert_eq!(m.provenance, Provenance::Measured);
    assert_eq!(m.samples, 200);
    assert!(m.gear.wcet_us >= 1);
}

#[test]
fn measured_gear_admits_at_one_khz() {
    let m = measure_lookup(100);
    let lp = Loop { period_us: 1000, deadline_us: 1000, memory_kib: 4096 };
    let d = select(&lp, &[m.gear]);
    match d {
        Decision::Admit { gear, .. } => assert_eq!(gear, "g0-lookup-measured"),
        other => panic!("expected admit, got {:?}", other),
    }
}

#[test]
fn zero_samples_still_measures_once() {
    let m = measure_lookup(0);
    assert_eq!(m.samples, 1);
}
