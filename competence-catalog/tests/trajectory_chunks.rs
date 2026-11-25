use competence_catalog::trajectory::{ingest_chunks, Tick, Trajectory};
use competence_catalog::twin::{simulate, Plant};

fn ep(acts: &[f64], rec: &[f64]) -> Trajectory {
    Trajectory {
        source: "rig".into(),
        ticks: acts.iter().zip(rec.iter()).map(|(a, o)| Tick { obs: *o, act: *a }).collect(),
    }
}

#[test]
fn chunks_share_one_dedup_set_and_receipt() {
    let p = Plant { dt: 0.1 };
    let a1 = vec![1.0, 1.0];
    let r1 = simulate(&p, &a1);
    let a2 = vec![2.0, 2.0];
    let r2 = simulate(&p, &a2);
    let chunks = vec![vec![ep(&a1, &r1)], vec![ep(&a1, &r1), ep(&a2, &r2)]];
    let r = ingest_chunks(&chunks, &p, 1e-9);
    assert_eq!((r.offered, r.admitted, r.dedup), (3, 2, 1));
    assert_eq!(r.keys.len(), 2);
}

#[test]
fn twin_refusals_accumulate_across_chunks() {
    let p = Plant { dt: 0.1 };
    let acts = vec![1.0, 1.0];
    let mut bad = simulate(&p, &acts);
    bad[1] += 1.0;
    let chunks = vec![vec![ep(&acts, &bad)], vec![ep(&acts, &bad)]];
    let r = ingest_chunks(&chunks, &p, 1e-3);
    assert_eq!((r.admitted, r.refused_twin), (0, 2));
}
