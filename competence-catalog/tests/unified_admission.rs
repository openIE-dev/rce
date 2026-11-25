use competence_catalog::repo::ingest_episodes;
use competence_catalog::sovereignty::Policy;
use competence_catalog::trajectory::{Tick, Trajectory};
use competence_catalog::twin::{simulate, Plant};

fn ep(acts: &[f64], rec: &[f64]) -> Trajectory {
    Trajectory {
        source: "fleet".into(),
        ticks: acts.iter().zip(rec.iter()).map(|(a, o)| Tick { obs: *o, act: *a }).collect(),
    }
}

#[test]
fn sovereign_policy_admits_and_audits_each_episode() {
    let p = Plant { dt: 0.1 };
    let acts = vec![1.0, 1.0];
    let rec = simulate(&p, &acts);
    let a = ingest_episodes(&[ep(&acts, &rec)], Policy::default(), &p, 1e-9).unwrap();
    assert_eq!(a.report.admitted, 1);
    assert_eq!(a.audit_len, 1);
}

#[test]
fn non_sovereign_policy_refuses_the_whole_path() {
    let p = Plant { dt: 0.1 };
    let acts = vec![1.0];
    let rec = simulate(&p, &acts);
    let mut pol = Policy::default();
    pol.allow_model = true;
    assert!(ingest_episodes(&[ep(&acts, &rec)], pol, &p, 1e-9).is_err());
}

#[test]
fn divergent_episode_refused_but_path_still_sovereign() {
    let p = Plant { dt: 0.1 };
    let acts = vec![1.0, 1.0];
    let mut rec = simulate(&p, &acts);
    rec[1] += 1.0;
    let a = ingest_episodes(&[ep(&acts, &rec)], Policy::default(), &p, 1e-3).unwrap();
    assert_eq!((a.report.admitted, a.report.refused_twin, a.audit_len), (0, 1, 0));
}
