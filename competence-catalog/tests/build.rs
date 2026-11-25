//! Build-layer falsifiers (Blaze ingested): content-addressed incremental builds,
//! name-free keys ("a rename/reformat compiles nothing"), and the monotone
//! evidence ladder (tested → simulated → model-checked → proved).

use competence_catalog::build::{BuildCache, Evidence, IncrementalBuild};

#[test]
fn name_free_key_means_reformat_compiles_nothing() {
    let mut c = BuildCache::new();
    let r1 = c.build("a + b");
    let r2 = c.build("b + a"); // accidental representation change
    assert_eq!(r1.key, r2.key, "canonical key is name-free");
    assert_eq!(c.misses, 1, "only the first build compiled");
    assert_eq!(c.hits, 1, "the reformat was a cache hit — nothing recompiled");
    assert_eq!(c.len(), 1);
}

#[test]
fn behavior_change_recompiles() {
    let mut c = BuildCache::new();
    c.build("a + b");
    c.build("a - b"); // genuine behavior change
    assert_eq!(c.misses, 2);
    assert_eq!(c.len(), 2);
}

#[test]
fn evidence_ladder_is_monotone() {
    let mut c = BuildCache::new();
    let r = c.build("a + b");
    assert_eq!(r.evidence, Evidence::Tested);
    assert_eq!(c.promote(&r.key, Evidence::Proved), Some(Evidence::Proved));
    // Cannot downgrade assurance.
    assert_eq!(c.promote(&r.key, Evidence::Tested), Some(Evidence::Proved));
    assert_eq!(c.promote(&r.key, Evidence::ModelChecked), Some(Evidence::Proved));
}

#[test]
fn module_build_checks_and_caches() {
    let mut c = BuildCache::new();
    let src = "module m\n\nfn add(a: i64, b: i64) -> i64\n    return a + b\n";
    let r = c.build(src);
    assert!(r.ok, "a valid module checks: {}", r.diagnostic);
    let again = c.build(src);
    assert_eq!(again.key, r.key);
    assert_eq!(c.hits, 1, "rebuilding identical content is a hit");
}

#[test]
fn hit_rate_rises_with_reuse() {
    let mut c = BuildCache::new();
    c.build("a + b");
    for _ in 0..9 { c.build("b + a"); }
    assert!(c.hit_rate() >= 0.9, "reuse of compilation work: {}", c.hit_rate());
}

#[test]
fn item_grain_reruns_only_dependents() {
    let mut b = IncrementalBuild::new();
    b.set_item("a", "a + 1", &[]);
    b.set_item("b", "b + 1", &["a"]);
    b.set_item("c", "c + 1", &["b"]);
    // A body edit reruns that body's tasks and nothing else (DBSP).
    assert_eq!(b.change("a", "a + 2"), vec!["a", "b", "c"]);
    assert_eq!(b.change("b", "b + 2"), vec!["b", "c"]); // a untouched
    assert_eq!(b.change("c", "c + 2"), vec!["c"]);
}

#[test]
fn why_explains_from_recorded_reads() {
    let mut b = IncrementalBuild::new();
    b.set_item("a", "a + 1", &[]);
    b.set_item("b", "b + 1", &["a"]);
    assert!(b.why("b").contains("reads a"), "{}", b.why("b"));
    assert!(b.why("a").contains("reads nothing"), "{}", b.why("a"));
}
