//! Conformance suite for [`MemoryEngine`] backends. A backend passes when
//! [`run`] returns without panicking; each check names itself on failure.
//! `make` must return a fresh, empty engine each call.
use crate::engine::{MemoryEngine, Plan};
use crate::fact::{AsOf, Fact, NewFact, Provenance};

pub fn nf(s: &str, p: &str, o: &str, from: i64) -> NewFact {
    NewFact {
        subject: s.into(),
        predicate: p.into(),
        object: o.into(),
        text: String::new(),
        multi: false,
        valid_from: from,
        valid_to: None,
        form_hash: "h0".into(),
        provenance: Provenance::Extracted,
        source: None,
    }
}

fn objects(v: &[Fact]) -> Vec<String> {
    let mut o: Vec<String> = v.iter().map(|f| f.object.clone()).collect();
    o.sort();
    o
}

pub const CHECKS: &[&str] = &[
    "store_fetch",
    "duplicate_is_noop",
    "contradiction_truncates_older",
    "late_older_fact_is_closed",
    "multi_valued_no_contradiction",
    "forget_retracts_but_keeps_history",
    "forget_with_world_end",
    "forget_twice_changes_nothing",
    "commit_is_atomic",
    "ids_unique_monotonic",
    "provenance_and_receipt_round_trip",
    "different_subjects_independent",
];

pub fn run<E: MemoryEngine>(mut make: impl FnMut() -> E) {
    for name in CHECKS {
        let mut e = make();
        check(name, &mut e);
    }
}

fn check<E: MemoryEngine>(name: &str, e: &mut E) {
    let ctx = |m: &str| format!("conformance `{name}`: {m}");
    match name {
        "store_fetch" => {
            let p = e.store(&nf("f", "returns", "u64", 10), 100).unwrap();
            let id = p.new_id.unwrap();
            let f = e.fetch(id).unwrap();
            assert_eq!(f.object, "u64", "{}", ctx("object"));
            assert_eq!(
                (f.lineage, f.recorded_at, f.expired_at),
                (id, 100, None),
                "{}",
                ctx("times")
            );
            assert_eq!(e.health().unwrap().rows, 1, "{}", ctx("rows"));
            assert!(
                e.fetch(id + 999).is_err(),
                "{}",
                ctx("missing id must error")
            );
        }
        "duplicate_is_noop" => {
            let a = e.store(&nf("f", "returns", "u64", 10), 100).unwrap();
            let b = e.store(&nf("f", "returns", "u64", 20), 200).unwrap();
            assert!(b.is_empty(), "{}", ctx("dup writes nothing"));
            assert_eq!(
                b.duplicate_of,
                a.new_id,
                "{}",
                ctx("dup points at original")
            );
            assert_eq!(e.health().unwrap().rows, 1, "{}", ctx("rows"));
        }
        "contradiction_truncates_older" => {
            let a = e
                .store(&nf("f", "returns", "u64", 10), 100)
                .unwrap()
                .new_id
                .unwrap();
            let p = e.store(&nf("f", "returns", "i64", 50), 200).unwrap();
            assert_eq!(p.expires, vec![(a, 200)], "{}", ctx("old row expired"));
            let old = e.fetch(a).unwrap();
            assert_eq!(old.expired_at, Some(200), "{}", ctx("expired_at"));
            assert_eq!(
                old.valid_to,
                None,
                "{}",
                ctx("old row itself unchanged in world time")
            );
            // Today: u64 held [10,50), i64 from 50.
            assert_eq!(
                objects(&e.as_of(AsOf::now(300)).unwrap()),
                ["i64"],
                "{}",
                ctx("now")
            );
            let then = AsOf {
                valid_at: 30,
                known_at: 300,
            };
            assert_eq!(
                objects(&e.as_of(then).unwrap()),
                ["u64"],
                "{}",
                ctx("world 30")
            );
            // What memory believed at system time 150 about world 60: u64.
            let past = AsOf {
                valid_at: 60,
                known_at: 150,
            };
            assert_eq!(
                objects(&e.as_of(past).unwrap()),
                ["u64"],
                "{}",
                ctx("system 150")
            );
            let h = e.history(a).unwrap();
            assert_eq!(h.len(), 2, "{}", ctx("lineage keeps both rows"));
            assert_eq!(h[1].valid_to, Some(50), "{}", ctx("successor closed at 50"));
            assert_eq!(h[1].supersedes, Some(a), "{}", ctx("supersedes"));
            assert_eq!(e.health().unwrap().rows, 3, "{}", ctx("nothing deleted"));
        }
        "late_older_fact_is_closed" => {
            e.store(&nf("f", "returns", "i64", 50), 100).unwrap();
            let p = e.store(&nf("f", "returns", "u64", 10), 200).unwrap();
            assert!(p.expires.is_empty(), "{}", ctx("newer claim stays"));
            let n = e.fetch(p.new_id.unwrap()).unwrap();
            assert_eq!(n.valid_to, Some(50), "{}", ctx("older claim closed"));
            let at = |t| {
                objects(
                    &e.as_of(AsOf {
                        valid_at: t,
                        known_at: 300,
                    })
                    .unwrap(),
                )
            };
            assert_eq!(at(20), ["u64"], "{}", ctx("20"));
            assert_eq!(at(60), ["i64"], "{}", ctx("60"));
        }
        "multi_valued_no_contradiction" => {
            let mut a = nf("f", "calls", "g", 0);
            a.multi = true;
            let mut b = nf("f", "calls", "h", 0);
            b.multi = true;
            e.store(&a, 1).unwrap();
            let p = e.store(&b, 2).unwrap();
            assert!(p.expires.is_empty(), "{}", ctx("no expiry"));
            assert_eq!(
                objects(&e.as_of(AsOf::now(5)).unwrap()),
                ["g", "h"],
                "{}",
                ctx("both")
            );
        }
        "forget_retracts_but_keeps_history" => {
            let id = e.store(&nf("x", "is", "y", 0), 10).unwrap().new_id.unwrap();
            e.forget(id, None, 20).unwrap();
            assert!(e.list_live().unwrap().is_empty(), "{}", ctx("not live"));
            assert!(
                e.as_of(AsOf::now(30)).unwrap().is_empty(),
                "{}",
                ctx("not now")
            );
            let before = AsOf {
                valid_at: 30,
                known_at: 15,
            };
            assert_eq!(
                e.as_of(before).unwrap().len(),
                1,
                "{}",
                ctx("still known at 15")
            );
            assert_eq!(e.health().unwrap().rows, 1, "{}", ctx("not deleted"));
        }
        "forget_with_world_end" => {
            let id = e.store(&nf("x", "is", "y", 0), 10).unwrap().new_id.unwrap();
            let p = e.forget(id, Some(25), 20).unwrap();
            let s = e.fetch(p.new_id.unwrap()).unwrap();
            assert_eq!(
                (s.valid_to, s.lineage),
                (Some(25), id),
                "{}",
                ctx("closed successor")
            );
            let at = |t| {
                e.as_of(AsOf {
                    valid_at: t,
                    known_at: 99,
                })
                .unwrap()
                .len()
            };
            assert_eq!((at(24), at(25)), (1, 0), "{}", ctx("half-open end"));
        }
        "forget_twice_changes_nothing" => {
            let id = e.store(&nf("x", "is", "y", 0), 10).unwrap().new_id.unwrap();
            e.forget(id, None, 20).unwrap();
            assert!(e.forget(id, None, 30).is_err(), "{}", ctx("second forget"));
            assert!(e.forget(12345, None, 30).is_err(), "{}", ctx("unknown id"));
        }
        "commit_is_atomic" => {
            let id = e.store(&nf("x", "is", "y", 0), 10).unwrap().new_id.unwrap();
            let mut ghost = e.fetch(id).unwrap();
            ghost.id = e.next_id().unwrap();
            let bad = Plan {
                inserts: vec![ghost.clone()],
                expires: vec![(id, 20), (999_999, 20)],
                ..Default::default()
            };
            assert!(e.commit(&bad).is_err(), "{}", ctx("bad plan refused"));
            assert!(
                e.get(ghost.id).unwrap().is_none(),
                "{}",
                ctx("insert rolled back")
            );
            assert!(e.fetch(id).unwrap().live(), "{}", ctx("expire rolled back"));
        }
        "ids_unique_monotonic" => {
            let a = e.next_id().unwrap();
            let b = e.next_id().unwrap();
            let c = e.store(&nf("q", "r", "s", 0), 1).unwrap().new_id.unwrap();
            assert!(a < b && b < c, "{}", ctx("monotonic"));
        }
        "provenance_and_receipt_round_trip" => {
            let mut n = nf("m", "guess", "z", 0);
            n.provenance = Provenance::Model;
            n.text = "a model said so".into();
            n.source = Some("notes/m.md".into());
            let mut p = e.plan_store(&n, 5).unwrap();
            p.stamp("abc123");
            e.commit(&p).unwrap();
            let f = e.fetch(p.new_id.unwrap()).unwrap();
            assert_eq!(f.provenance, Provenance::Model, "{}", ctx("provenance"));
            assert_eq!(
                f.receipt_id.as_deref(),
                Some("abc123"),
                "{}",
                ctx("receipt")
            );
            assert_eq!(f.source.as_deref(), Some("notes/m.md"), "{}", ctx("source"));
            assert_eq!(f.form_hash, "h0", "{}", ctx("form hash"));
        }
        "different_subjects_independent" => {
            e.store(&nf("a", "is", "1", 0), 1).unwrap();
            let p = e.store(&nf("b", "is", "2", 0), 2).unwrap();
            assert!(p.expires.is_empty(), "{}", ctx("independent"));
        }
        other => panic!("unknown check {other}"),
    }
}
