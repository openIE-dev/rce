//! SFU forwarding for chat calls (apps/sfu.rce) over P4 remote channels:
//! two subscribers and a publisher on loopback; the SFU forwards each frame
//! unreliably to every other subscriber of the track, drops stale frames,
//! holds only (track, seq, digest) metadata, and every log replays with no
//! network.
use rce_spec_check::adapters::chan::chan_adapters;
use rce_spec_check::host::{Adapter, Host, MemStore, replay};
use rce_spec_check::parse::parse;
use std::path::Path;

fn port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

#[test]
fn sfu_forwards_frames_to_subscribers_and_replays() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let sfu = std::fs::read_to_string(dir.join("apps/sfu.rce")).unwrap();
    let peer = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/chan.rce")).unwrap();
    let (ms, mp) = (parse(&sfu).unwrap(), parse(&peer).unwrap());
    let (s, a, b, c) = (format!("127.0.0.1:{}", port()), format!("127.0.0.1:{}", port()), format!("127.0.0.1:{}", port()), format!("127.0.0.1:{}", port()));
    let ad = |me: &str, peers: Vec<String>| -> Vec<Box<dyn Adapter>> { chan_adapters(me, &peers).unwrap().into_iter().map(|x| Box::new(x) as Box<dyn Adapter>).collect() };
    let (mut ss, mut sa, mut sb, mut sc) = (MemStore::default(), MemStore::default(), MemStore::default(), MemStore::default());
    let mut hs = Host::start_with(&ms, Box::new(&mut ss), ad(&s, vec![a.clone(), b.clone(), c.clone()])).unwrap();
    let mut ha = Host::start_with(&mp, Box::new(&mut sa), ad(&a, vec![s.clone()])).unwrap();
    let mut hb = Host::start_with(&mp, Box::new(&mut sb), ad(&b, vec![s.clone()])).unwrap();
    let mut hc = Host::start_with(&mp, Box::new(&mut sc), ad(&c, vec![s.clone()])).unwrap();
    hs.admit(&sfu, r#"Setup(me: "sfu")"#).unwrap();
    // a and b watch the video track of room r; c publishes it
    ha.admit(&peer, &format!(r#"Send(peer: "{s}", body: "sub\u{{1f}}r\u{{1f}}video")"#)).unwrap();
    hb.admit(&peer, &format!(r#"Send(peer: "{s}", body: "sub\u{{1f}}r\u{{1f}}video")"#)).unwrap();
    let poll_until = |h: &mut Host, src: &str, want: &dyn Fn(&str) -> bool| {
        for _ in 0..200 {
            h.admit(src, "Poll").unwrap();
            if want(&format!("{:?}", h.state())) {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        false
    };
    assert!(poll_until(&mut hs, &sfu, &|x| x.matches("video\\u{1f}").count() >= 2), "{:?}", hs.state());
    let d1 = "11".repeat(32);
    let d2 = "22".repeat(32);
    for (q, d) in [(1, &d1), (2, &d2)] {
        hc.admit(&peer, &format!(r#"Send(peer: "{s}", body: "frame\u{{1f}}r\u{{1f}}video\u{{1f}}{q}\u{{1f}}{d}\u{{1f}}e2ee:ab{q}")"#)).unwrap();
    }
    // a stale frame (seq 1 again) is dropped by the SFU
    hc.admit(&peer, &format!(r#"Send(peer: "{s}", body: "frame\u{{1f}}r\u{{1f}}video\u{{1f}}1\u{{1f}}{d1}\u{{1f}}e2ee:old")"#)).unwrap();
    assert!(poll_until(&mut hs, &sfu, &|x| x.contains("stale\", Int(1")), "{:?}", hs.state());
    let st = format!("{:?}", hs.state());
    assert!(st.contains("forwarded\", Int(4"), "{st}");
    assert!(!st.contains("e2ee:"), "the SFU folds no payload into state: {st}");
    assert!(st.contains(&d2), "last digest per publisher is kept: {st}");
    for h in [&mut ha, &mut hb] {
        assert!(poll_until(h, &peer, &|x| x.contains("e2ee:ab2") && x.contains("e2ee:ab1")), "{:?}", h.state());
    }
    let sc_state = format!("{:?}", hc.state());
    assert!(!sc_state.contains("e2ee:ab"), "the publisher is not sent its own frames");
    let lives: Vec<String> = [&hs.state_hash(), &ha.state_hash(), &hb.state_hash(), &hc.state_hash()].iter().map(|x| x.to_string()).collect();
    drop((hs, ha, hb, hc));
    let t = |st: &MemStore| st.lines.iter().map(|l| format!("{l}\n")).collect::<String>();
    assert_eq!(replay(&ms, &t(&ss)).unwrap().1, lives[0]);
    assert_eq!(replay(&mp, &t(&sa)).unwrap().1, lives[1]);
    assert_eq!(replay(&mp, &t(&sb)).unwrap().1, lives[2]);
    assert_eq!(replay(&mp, &t(&sc)).unwrap().1, lives[3]);
}
