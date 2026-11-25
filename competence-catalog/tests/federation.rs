//! Multi-node federation (rung 8→9): nodes with different corpora exchange feeds
//! and converge to the union (CALM), with every imported record replay-verified
//! at the border. This guards the convergence property the live `/sync` relies
//! on; the network transport itself is exercised by the two-node `/sync` demo.

use competence_catalog::foreman::{merge_feeds, Foreman};
use competence_catalog::runtime::RuntimeBridge;
use serde_json::json;

fn entries() -> Vec<(String, String, String, String)> {
    vec![
        ("hash.fnv1a32".into(), "hash".into(), "fnv1a32".into(), "(&[u8])->u32".into()),
        ("time.civil.is_leap".into(), "time.civil".into(), "is_leap".into(), "(i64)->bool".into()),
    ]
}

fn node(tag: &str) -> Foreman {
    let bridge = RuntimeBridge::new_inprocess();
    let _ = tag;
    Foreman::new(bridge, &entries()).unwrap()
}

fn keys(f: &Foreman) -> Vec<String> {
    let mut k: Vec<String> = f.export().iter().map(|r| r.input_key.clone()).collect();
    k.sort();
    k
}

#[test]
fn nodes_converge_to_the_union() {
    let mut a = node("a");
    let mut b = node("b");
    // Each node crystallizes a different competence.
    a.resolve("time.civil.is_leap", "(i64)->bool", &json!([2024])).unwrap();
    b.seed("hash.fnv1a32", "(&[u8])->u32", &json!([[104, 105]]), "1748694682");
    assert_eq!(keys(&a).len(), 1);
    assert_eq!(keys(&b).len(), 1);
    assert_ne!(keys(&a), keys(&b));

    // Mutual sync: each imports the other's feed.
    b.import(&a.export());
    a.import(&b.export());

    // Both converge to the union (CALM).
    assert_eq!(keys(&a), keys(&b));
    assert_eq!(keys(&a).len(), 2);

    // The circulated competence executes at G0 in the receiving node.
    let r = b.resolve("time.civil.is_leap", "(i64)->bool", &json!([2025])).unwrap();
    assert!(r.hit);
    assert_eq!(r.result, "false");
}

#[test]
fn repeated_sync_is_idempotent() {
    let mut a = node("a");
    let mut b = node("b");
    a.resolve("time.civil.is_leap", "(i64)->bool", &json!([2024])).unwrap();
    let feed = a.export();
    b.import(&feed);
    let size_after_first = b.corpus_size();
    // Re-syncing the same feed (and a merged feed) adds nothing.
    let report = b.import(&merge_feeds(&feed, &b.export()));
    assert_eq!(report.accepted, 0);
    assert_eq!(b.corpus_size(), size_after_first);
}
