//! Cross-engine competence circulation (rung 8): competence moves between
//! catalogs by a CALM monotone merge, and every imported record is re-verified
//! by replay at the border — circulation never bypasses the admission law.

use competence_catalog::foreman::{merge_feeds, Foreman};
use competence_catalog::runtime::RuntimeBridge;
use competence_catalog::{intent_key, CompetenceRecord, Provenance};
use serde_json::json;

fn entries() -> Vec<(String, String, String, String)> {
    vec![
        ("hash.fnv1a32".into(), "hash".into(), "fnv1a32".into(), "(&[u8])->u32".into()),
        ("time.civil.is_leap".into(), "time.civil".into(), "is_leap".into(), "(i64)->bool".into()),
    ]
}

fn engine(tag: &str) -> Foreman {
    let _ = tag;
    let bridge = RuntimeBridge::new_inprocess();
    Foreman::new(bridge, &entries()).unwrap()
}

#[test]
fn competence_circulates_and_hits_in_the_receiving_engine() {
    let mut a = engine("a");
    // Engine A crystallizes is_leap by resolving a novel instance.
    a.resolve("time.civil.is_leap", "(i64)->bool", &json!([2024])).unwrap();
    assert_eq!(a.corpus_size(), 1);

    // Engine B starts empty, imports A's feed (replay-verified at the border).
    let mut b = engine("b");
    assert_eq!(b.corpus_size(), 0);
    let report = b.import(&a.export());
    assert_eq!(report.accepted, 1, "the spectest reproduces, so B admits it");
    assert_eq!(b.corpus_size(), 1);

    // B now hits G0 for a novel instance of the circulated intent.
    let r = b.resolve("time.civil.is_leap", "(i64)->bool", &json!([2025])).unwrap();
    assert!(r.hit, "circulated competence hits in the receiving engine");
    assert_eq!(r.cost_uj, 1);
}

#[test]
fn import_is_idempotent() {
    let mut a = engine("a");
    a.resolve("time.civil.is_leap", "(i64)->bool", &json!([2024])).unwrap();
    let feed = a.export();
    let mut b = engine("b");
    b.import(&feed);
    let again = b.import(&feed);
    assert_eq!(again.skipped, 1, "re-import skips already-owned intents");
    assert_eq!(again.accepted, 0);
    assert_eq!(b.corpus_size(), 1, "idempotent: no growth on re-import");
}

#[test]
fn merge_is_calm() {
    let mut a = engine("a");
    a.resolve("time.civil.is_leap", "(i64)->bool", &json!([2024])).unwrap();
    let mut b = engine("b");
    b.seed("hash.fnv1a32", "(&[u8])->u32", &json!([[104, 105]]), "1748694682");
    let (fa, fb) = (a.export(), b.export());

    let ab = merge_feeds(&fa, &fb);
    let ba = merge_feeds(&fb, &fa);
    assert_eq!(ab.len(), 2);
    // Commutative: same intent keys regardless of order.
    let mut ka: Vec<_> = ab.iter().map(|r| r.input_key.clone()).collect();
    let mut kb: Vec<_> = ba.iter().map(|r| r.input_key.clone()).collect();
    ka.sort(); kb.sort();
    assert_eq!(ka, kb);
    // Idempotent: merging a feed with itself adds nothing.
    assert_eq!(merge_feeds(&fa, &fa).len(), fa.len());
}

#[test]
fn forged_record_is_refused_at_the_border() {
    let mut b = engine("b");
    let forged = CompetenceRecord {
        input_key: intent_key("hash.fnv1a32", "(&[u8])->u32"),
        handler_kind: "hash.fnv1a32".into(),
        probe_input: "[[104,105]]".into(),
        probe_output: "999".into(), // wrong: FNV-1a 32 of "hi" is 1748694682
        result: String::new(),
        cost_uj: 5_000_000,
        provenance: Provenance::Reported,
    };
    let report = b.import(&[forged]);
    assert_eq!(report.refused, 1, "a non-reproducing record is refused at the border");
    assert_eq!(b.corpus_size(), 0, "negative transfer never enters the corpus");
}
