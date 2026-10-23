use rce_memory::conformance::{self, nf};
use rce_memory::engine::{MemEngine, MemoryEngine};
use rce_memory::fact::AsOf;
use rce_memory::redb_engine::RedbEngine;

#[test]
fn reference_engine_passes() {
    conformance::run(MemEngine::default);
}

#[test]
fn redb_engine_passes() {
    let dir = tempfile::tempdir().unwrap();
    let mut n = 0;
    conformance::run(|| {
        n += 1;
        RedbEngine::open(&dir.path().join(format!("m{n}.redb"))).unwrap()
    });
}

#[test]
fn redb_persists_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("m.redb");
    let (a, b);
    {
        let mut e = RedbEngine::open(&p).unwrap();
        a = e
            .store(&nf("f", "returns", "u64", 10), 100)
            .unwrap()
            .new_id
            .unwrap();
        b = e
            .store(&nf("f", "returns", "i64", 50), 200)
            .unwrap()
            .new_id
            .unwrap();
    }
    let mut e = RedbEngine::open(&p).unwrap();
    assert_eq!(e.health().unwrap().rows, 3);
    assert_eq!(e.fetch(a).unwrap().expired_at, Some(200));
    assert_eq!(e.as_of(AsOf::now(300)).unwrap()[0].id, b);
    // the id counter survived too
    assert!(e.next_id().unwrap() > b);
}

#[test]
fn both_engines_agree_on_a_mixed_history() {
    let dir = tempfile::tempdir().unwrap();
    let mut m = MemEngine::default();
    let mut r = RedbEngine::open(&dir.path().join("x.redb")).unwrap();
    fn script<E: MemoryEngine>(e: &mut E) {
        e.store(&nf("a", "p", "1", 0), 1).unwrap();
        e.store(&nf("a", "p", "2", 10), 2).unwrap();
        e.store(&nf("a", "p", "0", -5), 3).unwrap();
        let id = e.store(&nf("b", "p", "x", 0), 4).unwrap().new_id.unwrap();
        e.forget(id, Some(7), 5).unwrap();
    }
    script(&mut m);
    script(&mut r);
    assert_eq!(m.scan().unwrap(), r.scan().unwrap());
    for t in [-10, -5, 0, 5, 9, 10, 100] {
        for k in [1, 2, 3, 4, 5, 6] {
            let q = AsOf {
                valid_at: t,
                known_at: k,
            };
            assert_eq!(m.as_of(q).unwrap(), r.as_of(q).unwrap(), "{q:?}");
        }
    }
}
