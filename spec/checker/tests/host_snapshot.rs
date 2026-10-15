//! HS-5: a snapshot is (seq, state_hash, bytes), a cache used only when the
//! hash of its bytes equals D's state_hash at seq; otherwise discarded.
//! Loading one changes no meaning.
use rce_spec_check::host::{Host, Snapshot, replay};
use rce_spec_check::parse::parse;
use std::path::Path;

fn src() -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("examples/55_rcedb.rce")).unwrap()
}

#[test]
fn snapshot_open_equals_full_refold_and_bad_snapshots_are_discarded() {
    let src = src();
    let m = parse(&src).unwrap();
    let p = std::env::temp_dir().join(format!("rce-host-snap-{}.log", std::process::id()));
    let _ = std::fs::remove_file(&p);
    let snap: Snapshot;
    {
        let mut h = Host::open_file(&m, &p).unwrap();
        h.admit(&src, r#"Put(key: "b", val: "2"), Put(key: "a", val: "1")"#).unwrap();
        snap = h.snapshot().expect("no outstanding request");
        h.admit(&src, r#"Delete(key: "a"), Put(key: "c", val: "3")"#).unwrap();
    }
    let text = std::fs::read_to_string(&p).unwrap();
    let mut got = vec![];
    let mut bad_bytes = snap.clone();
    bad_bytes.bytes = bad_bytes.bytes.replace("\"2\"", "\"9\"");
    let mut bad_seq = snap.clone();
    bad_seq.seq += 1;
    for s in [Some(&snap), Some(&bad_bytes), Some(&bad_seq), None] {
        let copy = p.with_extension(format!("c{}", s.map(|x| x.seq).unwrap_or(0)));
        std::fs::write(&copy, &text).unwrap();
        let h = Host::open_file_snap(&m, &copy, vec![], s).unwrap();
        let after = std::fs::read_to_string(&copy).unwrap();
        assert_eq!(h.from_snapshot.is_some(), s == Some(&snap), "only the valid snapshot is used");
        let live = h.state_hash().to_string();
        drop(h);
        // live state after open is the fold of the log it leaves (T37)
        assert_eq!(replay(&m, &after).unwrap().1, live);
        got.push((live, after));
        let _ = std::fs::remove_file(&copy);
    }
    // with a valid, a corrupted, a misplaced or no snapshot: the same D and state
    assert!(got.windows(2).all(|w| w[0] == w[1]), "snapshot changed meaning");
    let _ = std::fs::remove_file(&p);
}
