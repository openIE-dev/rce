//! The competence-before-model cascade, over real stdlib execution: a seeded
//! basis turns novel instances into G0 hits at the interpreter floor; a novel
//! operation misses once, crystallizes by replay, then decays to the floor. The
//! ecology meter is the Law of Coding falsifier — reuse rises, cost/act falls.

use competence_catalog::foreman::Foreman;
use competence_catalog::runtime::RuntimeBridge;
use serde_json::json;

fn entries() -> Vec<(String, String, String, String)> {
    vec![
        ("hash.fnv1a32".into(), "hash".into(), "fnv1a32".into(), "(&[u8])->u32".into()),
        ("time.civil.is_leap".into(), "time.civil".into(), "is_leap".into(), "(i64)->bool".into()),
        ("col.search.binary_search".into(), "col.search".into(), "binary_search".into(), "(&[i64],i64)->i64".into()),
    ]
}

fn foreman() -> Foreman {
    let bridge = RuntimeBridge::new_inprocess();
    Foreman::new(bridge, &entries()).unwrap()
}

#[test]
fn seeded_basis_hits_novel_instances_at_the_floor() {
    let mut f = foreman();
    assert!(f.seed("hash.fnv1a32", "(&[u8])->u32", &json!([[104, 105]]), "1748694682").is_accepted());
    // Novel inputs (never the probe) still hit G0 and execute correctly.
    for bs in [[97u8], [98], [99], [100]] {
        let args = json!([bs.to_vec()]);
        let r = f.resolve("hash.fnv1a32", "(&[u8])->u32", &args).unwrap();
        assert!(r.hit, "a seeded intent hits for any novel input");
        assert_eq!(r.cost_uj, 1);
    }
    let m = f.meter();
    assert_eq!(m.misses, 0);
    assert_eq!(m.reuse_rate, 1.0);
    assert_eq!(m.cost_per_act_uj, 1.0);
}

#[test]
fn novel_operation_crystallizes_once_then_decays() {
    let mut f = foreman();
    // is_leap is not seeded: the first resolve misses and pays the proposer.
    let first = f.resolve("time.civil.is_leap", "(i64)->bool", &json!([2024])).unwrap();
    assert!(!first.hit, "a novel operation misses first");
    assert_eq!(first.result, "true");
    assert!(first.cost_uj >= 5_000_000, "the first miss pays the proposer");
    assert_eq!(f.corpus_size(), 1, "the corpus grew by one crystallized competence");

    // Subsequent novel years hit G0 at the floor.
    for y in [2025, 2026, 2027, 2028, 1900, 2000] {
        let r = f.resolve("time.civil.is_leap", "(i64)->bool", &json!([y])).unwrap();
        assert!(r.hit, "after crystallization, novel instances hit");
        assert_eq!(r.cost_uj, 1);
    }
    let m = f.meter();
    assert_eq!(m.misses, 1, "only the first novel operation missed");
    assert!(m.reuse_rate > 0.85, "exponential decay toward the floor");
}

#[test]
fn cost_per_act_falls_as_the_corpus_grows() {
    let mut f = foreman();
    f.seed("hash.fnv1a32", "(&[u8])->u32", &json!([[104, 105]]), "1748694682");
    // First act: a novel operation pays the proposer.
    f.resolve("time.civil.is_leap", "(i64)->bool", &json!([2024])).unwrap();
    let early = f.meter().cost_per_act_uj;
    // Then many G0 hits over the seeded basis.
    for i in 0..50u8 {
        f.resolve("hash.fnv1a32", "(&[u8])->u32", &json!([[i]])).unwrap();
    }
    let late = f.meter().cost_per_act_uj;
    assert!(late < early, "cost per competent act must fall as reuse grows: {late} < {early}");
    assert!(late < 100_000.0, "approaching the interpreter floor");
}
