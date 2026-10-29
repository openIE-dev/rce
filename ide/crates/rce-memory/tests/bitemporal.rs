//! Bitemporal semantics in depth (reference engine).
use rce_memory::conformance::nf;
use rce_memory::engine::{MemEngine, MemoryEngine};
use rce_memory::fact::AsOf;

fn objs(e: &MemEngine, valid_at: i64, known_at: i64) -> Vec<String> {
    let mut v: Vec<String> = e
        .as_of(AsOf { valid_at, known_at })
        .unwrap()
        .into_iter()
        .map(|f| f.object)
        .collect();
    v.sort();
    v
}

#[test]
fn three_successive_contradictions_keep_a_full_timeline() {
    let mut e = MemEngine::default();
    e.store(&nf("Account.balance", "type", "i32", 0), 10)
        .unwrap();
    e.store(&nf("Account.balance", "type", "i64", 100), 20)
        .unwrap();
    e.store(&nf("Account.balance", "type", "u128", 200), 30)
        .unwrap();
    // world timeline as known now
    assert_eq!(objs(&e, 50, 99), ["i32"]);
    assert_eq!(objs(&e, 150, 99), ["i64"]);
    assert_eq!(objs(&e, 250, 99), ["u128"]);
    // what memory believed at each system time about world 250
    assert_eq!(objs(&e, 250, 15), ["i32"]);
    assert_eq!(objs(&e, 250, 25), ["i64"]);
    assert_eq!(objs(&e, 250, 35), ["u128"]);
    // before anything was recorded
    assert!(objs(&e, 250, 5).is_empty());
    // nothing deleted: 3 originals + 2 successors
    assert_eq!(e.scan().unwrap().len(), 5);
    assert_eq!(e.list_live().unwrap().len(), 3);
}

#[test]
fn bounded_new_fact_inside_open_old_fact() {
    let mut e = MemEngine::default();
    e.store(&nf("svc", "owner", "ana", 0), 1).unwrap();
    let mut n = nf("svc", "owner", "ben", 10);
    n.valid_to = Some(20);
    e.store(&n, 2).unwrap();
    // old fact truncated at 10; the new one holds [10,20). After 20 nobody is
    // recorded: memory does not invent that ana resumed.
    assert_eq!(objs(&e, 5, 9), ["ana"]);
    assert_eq!(objs(&e, 15, 9), ["ben"]);
    assert!(objs(&e, 25, 9).is_empty());
}

#[test]
fn non_overlapping_different_object_is_not_a_contradiction() {
    let mut e = MemEngine::default();
    let mut a = nf("x", "p", "1", 0);
    a.valid_to = Some(10);
    e.store(&a, 1).unwrap();
    let p = e.store(&nf("x", "p", "2", 10), 2).unwrap();
    assert!(p.expires.is_empty());
    assert_eq!(objs(&e, 9, 5), ["1"]);
    assert_eq!(objs(&e, 10, 5), ["2"]);
}

#[test]
fn invalid_requests_change_nothing() {
    let mut e = MemEngine::default();
    let mut n = nf("x", "p", "1", 10);
    n.valid_to = Some(10);
    assert!(e.store(&n, 1).is_err());
    assert!(e.store(&nf("", "p", "1", 0), 1).is_err());
    let id = e.store(&nf("x", "p", "1", 10), 1).unwrap().new_id.unwrap();
    assert!(e.forget(id, Some(5), 2).is_err(), "end before start");
    assert!(
        e.fetch(id).unwrap().live(),
        "refused forget changed nothing"
    );
}

#[test]
fn half_open_boundaries() {
    let mut e = MemEngine::default();
    e.store(&nf("x", "p", "1", 10), 100).unwrap();
    assert!(objs(&e, 9, 100).is_empty());
    assert_eq!(objs(&e, 10, 100), ["1"]);
    assert!(objs(&e, 10, 99).is_empty());
}
