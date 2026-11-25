//! Store-layer falsifiers (Blaze ingested): hot/warm/cold tiers with LRU demotion,
//! promotion on access, and trust-by-verification (sampled hits recomputed).

use competence_catalog::store::{Tier, TieredStore};

#[test]
fn access_promotes_to_hot() {
    let mut s: TieredStore<i64> = TieredStore::new(2, 2);
    s.put("a", 1);
    s.put("b", 2);
    s.put("c", 3); // hot cap 2 → oldest (a) demotes to warm
    assert_eq!(s.tier_of("a"), Some(Tier::Warm));
    s.get("a"); // access promotes a back to hot
    assert_eq!(s.tier_of("a"), Some(Tier::Hot));
}

#[test]
fn capacity_demotes_hot_to_warm_to_cold() {
    let mut s: TieredStore<i64> = TieredStore::new(1, 1);
    s.put("a", 1);
    s.put("b", 2); // a → warm
    s.put("c", 3); // b → warm, a → cold (warm cap 1)
    assert_eq!(s.count_in(Tier::Hot), 1);
    assert_eq!(s.count_in(Tier::Warm), 1);
    assert_eq!(s.count_in(Tier::Cold), 1);
    assert_eq!(s.tier_of("a"), Some(Tier::Cold));
}

#[test]
fn sampled_recompute_verifies_truth() {
    let mut s: TieredStore<i64> = TieredStore::new(4, 4);
    s.put("k", 42);
    // Honest recompute agrees.
    assert_eq!(s.sample_verify("k", |_| Some(42)), Some(true));
    assert_eq!(s.mismatches, 0);
    // A drifting/corrupted recompute is detected, not trusted.
    assert_eq!(s.sample_verify("k", |_| Some(99)), Some(false));
    assert_eq!(s.mismatches, 1);
    assert_eq!(s.recomputes, 2);
}

#[test]
fn same_keys_same_store() {
    // A second store with the same keys holds the same content (remote cache).
    let mut a: TieredStore<i64> = TieredStore::new(4, 4);
    let mut b: TieredStore<i64> = TieredStore::new(4, 4);
    a.put("x", 7);
    b.put("x", 7);
    assert_eq!(a.get("x"), b.get("x"));
}
