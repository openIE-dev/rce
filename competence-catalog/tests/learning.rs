//! The public learning loop falsifiers: hindsight recalls verified resolutions,
//! promotion reranks/prunes/distills by reuse, and the engine learns — all
//! without bypassing the replay gate. No private engine.

use competence_catalog::engine::{probe, Engine, NoProposer};
use competence_catalog::learning::{Hindsight, Promotion};
use competence_catalog::runtime::RuntimeBridge;
use competence_catalog::semantic::semantic_key;
use serde_json::json;

fn entries() -> Vec<(String, String, String, String)> {
    vec![
        ("time.civil.is_leap".into(), "time.civil".into(), "is_leap".into(), "(i64)->bool".into()),
        ("hash.fnv1a32".into(), "hash".into(), "fnv1a32".into(), "(&[u8])->u32".into()),
    ]
}

fn leap() -> Vec<competence_catalog::semantic::Probe> {
    vec![probe(json!([2024]), "true"), probe(json!([1900]), "false"), probe(json!([2000]), "true")]
}

#[test]
fn hindsight_recalls_a_verified_resolution() {
    let mut h = Hindsight::new();
    let key = semantic_key("(i64)->bool", &leap());
    assert!(h.recall(&key).is_none(), "nothing learned yet");
    h.record(&key, "(i64)->bool", "time.civil.is_leap", 2);
    let exp = h.recall(&key).expect("hindsight recalls the verified resolution");
    assert_eq!(exp.handler, "time.civil.is_leap");
    assert_eq!(h.reuse("time.civil.is_leap"), 1);
}

#[test]
fn promotion_reranks_hot_competence_first() {
    let mut h = Hindsight::new();
    for _ in 0..5 { h.record("k1", "s", "hot", 0); }
    h.record("k2", "s", "cold", 2);
    let ranked = Promotion::rerank(&["cold".into(), "hot".into()], &h);
    assert_eq!(ranked[0], "hot", "reuse promotes the hot competence");
}

#[test]
fn promotion_prunes_the_cold_tail() {
    let mut h = Hindsight::new();
    for _ in 0..3 { h.record("k1", "s", "hot", 0); }
    h.record("k2", "s", "cold", 2);
    let kept = Promotion::prune(&["hot".into(), "cold".into()], &h, 2);
    assert_eq!(kept, vec!["hot".to_string()], "cold competence is pruned");
}

#[test]
fn distill_merges_behavioral_duplicates() {
    let mut h = Hindsight::new();
    for _ in 0..4 { h.record("k", "s", "time.civil.is_leap", 0); }
    h.record("k", "s", "time.is_leap", 0);
    // Same behavioral fingerprint: keep the higher-reuse representative.
    let by_fp = vec![
        ("time.civil.is_leap".to_string(), "fp_leap".to_string()),
        ("time.is_leap".to_string(), "fp_leap".to_string()),
        ("os.proc.success".to_string(), "fp_success".to_string()),
    ];
    let kept = Promotion::distill(&by_fp, &h);
    assert_eq!(kept.len(), 2, "one representative per behavior");
    assert!(kept.contains(&"time.civil.is_leap".to_string()));
    assert!(!kept.contains(&"time.is_leap".to_string()), "the duplicate is distilled away");
}

#[test]
fn engine_learns_and_recalls_at_g0() {
    let mut e = Engine::new(RuntimeBridge::new_inprocess(), &entries(), vec![], Box::new(NoProposer));
    let spec = leap();
    let first = e.resolve("(i64)->bool", &spec).unwrap();
    assert_eq!(first.gear, 2, "first coverage extracts");
    assert!(!e.hindsight.is_empty(), "the engine learned from the verified resolution");
    // The learned recall resolves the repeat at G0.
    let second = e.resolve("(i64)->bool", &spec).unwrap();
    assert_eq!(second.gear, 0);
    assert!(e.hindsight.reuse(&first.handler) >= 2, "reuse accumulates");
}
