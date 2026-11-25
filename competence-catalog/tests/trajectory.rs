use competence_catalog::trajectory::{content_key, dispute, ingest, Tick, Trajectory};

fn ep(source: &str, ticks: Vec<(f64, f64)>) -> Trajectory {
    Trajectory { source: source.into(), ticks: ticks.into_iter().map(|(o, a)| Tick { obs: o, act: a }).collect() }
}

#[test]
fn same_content_two_sources_dedups_to_one() {
    let r = ingest(&[ep("rig-a", vec![(0.0, 1.0), (1.0, 1.0)]), ep("rig-b", vec![(0.0, 1.0), (1.0, 1.0)])]);
    assert_eq!((r.offered, r.admitted, r.dedup, r.refused), (2, 1, 1, 0));
    assert_eq!(r.keys.len(), 1);
}

#[test]
fn content_key_ignores_source() {
    let a = ep("rig-a", vec![(0.0, 1.0)]);
    let b = ep("sim-farm", vec![(0.0, 1.0)]);
    assert_eq!(content_key(&a), content_key(&b));
}

#[test]
fn malformed_episodes_refused() {
    let r = ingest(&[ep("bad", vec![(f64::NAN, 1.0)]), ep("empty", vec![]), ep("ok", vec![(0.0, 0.0)])]);
    assert_eq!((r.admitted, r.refused), (1, 2));
}

#[test]
fn dispute_answers_the_size_claim() {
    let r = ingest(&[ep("a", vec![(0.0, 1.0)]), ep("b", vec![(1.0, 2.0)])]);
    assert_eq!(dispute(1_000_000, &r), (2, 999_998));
    assert_eq!(dispute(1, &r), (2, 0));
}
