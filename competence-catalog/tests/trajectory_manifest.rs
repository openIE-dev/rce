use competence_catalog::trajectory::{ingest, manifest, Tick, Trajectory};

#[test]
fn manifest_carries_counts_and_name_free_keys() {
    let r = ingest(&[
        Trajectory { source: "a".into(), ticks: vec![Tick { obs: 0.0, act: 1.0 }] },
        Trajectory { source: "b".into(), ticks: vec![Tick { obs: 0.0, act: 1.0 }] },
        Trajectory { source: "c".into(), ticks: vec![] },
    ]);
    let m = manifest(&r);
    assert_eq!(m["schema"], "rce-episode-manifest/1");
    assert_eq!(m["offered"], 3);
    assert_eq!(m["admitted"], 1);
    assert_eq!(m["dedup"], 1);
    assert_eq!(m["refused"], 1);
    assert_eq!(m["keys"].as_array().unwrap().len(), 1);
    assert_eq!(m["keys"][0], r.keys[0]);
}
