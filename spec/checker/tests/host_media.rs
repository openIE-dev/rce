//! W1i media: capture and playback adapters. Frames are blobs addressed by
//! digest; D holds only (track, seq, digest) and the frame's Tick, so replay
//! with every adapter disconnected reaches the same state without
//! re-capturing (HC-3, HC-8, HC-10), and the playback sink refuses a frame
//! whose bytes do not match its digest.
use rce_spec_check::adapters::media::media_adapters;
use rce_spec_check::host::{Adapter, Host, MemStore, replay};
use rce_spec_check::parse::parse;
use std::path::{Path, PathBuf};

fn setup(tag: &str) -> (String, PathBuf) {
    let src = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/media.rce")).unwrap();
    let root = std::env::temp_dir().join(format!("rce-media-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("capture/cam")).unwrap();
    (src, root)
}

fn run(src: &str, root: &Path, evs: &str) -> (Vec<String>, String, String) {
    let m = parse(src).unwrap();
    let ads: Vec<Box<dyn Adapter>> = media_adapters(root).into_iter().map(|a| Box::new(a) as Box<dyn Adapter>).collect();
    let mut st = MemStore::default();
    let mut h = Host::start_with(&m, Box::new(&mut st), ads).unwrap();
    h.admit(src, evs).unwrap();
    let live = h.state_hash().to_string();
    let s = format!("{:?}", h.state());
    drop(h);
    (st.lines.clone(), live, s)
}

#[test]
fn captured_frames_are_digests_in_d_and_replay_without_capture() {
    let (src, root) = setup("cap");
    let frames: Vec<Vec<u8>> = (0..3u8).map(|i| (0..200u16).map(|p| (p as u8).wrapping_mul(i + 1) ^ 0x5a).collect()).collect();
    for (i, f) in frames.iter().enumerate() {
        std::fs::write(root.join(format!("capture/cam/{i:04}.raw")), f).unwrap();
    }
    let (lines, live, s) = run(&src, &root, r#"Open(source: "dir:cam"), Tick(n: 1), Tick(n: 2), Tick(n: 3), Tick(n: 4), Tick(n: 5)"#);
    assert!(s.contains("ended\", Bool(true"), "{s}");
    assert!(s.contains("played\", Int(3"), "{s}");
    assert!(s.contains("bytes\", Int(600"), "{s}");
    // each frame is stored once by digest, and D names it only by digest
    let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
    use sha2::{Digest, Sha256};
    for (i, f) in frames.iter().enumerate() {
        let d: String = Sha256::digest(f).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(std::fs::read(root.join(&d)).unwrap(), *f);
        assert!(text.contains(&format!("{}:{d}", i + 1)), "D logs seq:digest");
        assert!(s.contains(&format!("1/{}/{d}@{}", i + 1, i + 1)), "frame stamped with its Tick: {s}");
    }
    // the playback sink got every frame in order
    assert_eq!(std::fs::read(root.join("play/loopback")).unwrap(), frames.concat());
    // replay reads only D: remove every blob and capture first
    std::fs::remove_dir_all(&root).unwrap();
    let m = parse(&src).unwrap();
    assert_eq!(replay(&m, &text).unwrap().1, live);
}

#[test]
fn pattern_source_is_deterministic_and_bad_requests_fail() {
    let (src, root) = setup("pat");
    let evs = r#"Open(source: "pattern:4"), Tick(n: 1), Tick(n: 2), Tick(n: 3), Tick(n: 4), Tick(n: 5)"#;
    let (_, live1, s1) = run(&src, &root, evs);
    let (_, live2, _) = run(&src, &root, evs);
    assert_eq!(live1, live2, "a synthetic source gives the same D");
    assert!(s1.contains("played\", Int(4"), "{s1}");
    // a capture name that leaves the root, and an unknown source, fail
    let (_, _, s) = run(&src, &root, r#"Open(source: "dir:../x"), Open(source: "mic")"#);
    assert!(s.contains("errors\", Int(2"), "{s}");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn playback_stops_at_a_frame_that_does_not_match_its_digest() {
    let (src, root) = setup("bad");
    let (_, _, s) = run(&src, &root, r#"Open(source: "pattern:1"), Tick(n: 1)"#);
    let d = s.split("/1/").nth(1).unwrap()[..64].to_string();
    std::fs::write(root.join(&d), b"tampered").unwrap();
    let mut ads = media_adapters(&root);
    let r = ads[2].perform(&serde_json::json!({"req": 1, "track": "x", "digest": d}));
    assert!(r.error.contains("does not match"), "{}", r.error);
    let r = ads[2].perform(&serde_json::json!({"req": 2, "track": "../x", "digest": d}));
    assert!(!r.error.is_empty());
    let _ = std::fs::remove_dir_all(&root);
}
