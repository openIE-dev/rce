use competence_catalog::trajectory::{ingest, ingest_gated, Tick, Trajectory};
use competence_catalog::twin::{simulate, Plant};

fn ep(source: &str, acts: &[f64], recorded: &[f64]) -> Trajectory {
    Trajectory {
        source: source.into(),
        ticks: acts.iter().zip(recorded.iter()).map(|(a, o)| Tick { obs: *o, act: *a }).collect(),
    }
}

#[test]
fn coherent_episode_admitted_through_the_twin_gate() {
    let p = Plant { dt: 0.1 };
    let acts = vec![1.0, 1.0, 1.0];
    let rec = simulate(&p, &acts);
    let r = ingest_gated(&[ep("rig", &acts, &rec)], &p, 1e-9);
    assert_eq!((r.admitted, r.refused_twin, r.refused), (1, 0, 0));
}

#[test]
fn divergent_episode_refused_by_the_twin() {
    let p = Plant { dt: 0.1 };
    let acts = vec![1.0, 1.0, 1.0];
    let mut rec = simulate(&p, &acts);
    rec[2] += 0.5;
    let r = ingest_gated(&[ep("rig", &acts, &rec)], &p, 1e-3);
    assert_eq!((r.admitted, r.refused_twin), (0, 1));
}

#[test]
fn gated_dedup_still_content_addressed() {
    let p = Plant { dt: 0.1 };
    let acts = vec![1.0, 1.0];
    let rec = simulate(&p, &acts);
    let r = ingest_gated(&[ep("a", &acts, &rec), ep("b", &acts, &rec)], &p, 1e-9);
    assert_eq!((r.admitted, r.dedup), (1, 1));
}

#[test]
fn ungated_ingest_reports_zero_twin_refusals() {
    let p = Plant { dt: 0.1 };
    let acts = vec![1.0];
    let r = ingest(&[ep("a", &acts, &[99.0])]);
    assert_eq!(r.refused_twin, 0);
    assert_eq!(r.admitted, 1);
}
