//! P4 remote channels: two hosts exchange messages; a reliable send is
//! delivered only after the peer's ack, an ungranted or dead peer gives one
//! failed/unreachable outcome, and both logs replay with no network.
use rce_spec_check::adapters::chan::chan_adapters;
use rce_spec_check::host::{Adapter, Host, MemStore, replay};
use rce_spec_check::parse::parse;
use std::path::Path;

fn port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

#[test]
fn two_hosts_exchange_messages_and_both_logs_replay() {
    let src = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/chan.rce")).unwrap();
    let m = parse(&src).unwrap();
    let (a, b, dead) = (format!("127.0.0.1:{}", port()), format!("127.0.0.1:{}", port()), format!("127.0.0.1:{}", port()));
    let ad = |me: &str, peers: Vec<String>| -> Vec<Box<dyn Adapter>> { chan_adapters(me, &peers).unwrap().into_iter().map(|x| Box::new(x) as Box<dyn Adapter>).collect() };
    let (mut sa, mut sb) = (MemStore::default(), MemStore::default());
    let mut ha = Host::start_with(&m, Box::new(&mut sa), ad(&a, vec![b.clone(), dead.clone()])).unwrap();
    let mut hb = Host::start_with(&m, Box::new(&mut sb), ad(&b, vec![a.clone()])).unwrap();
    ha.admit(&src, &format!(r#"Send(peer: "{b}", body: "hello b"), Ping(peer: "{b}"), Send(peer: "{b}", body: "ciphertext:9f2a")"#)).unwrap();
    hb.admit(&src, &format!(r#"Send(peer: "{a}", body: "hello a")"#)).unwrap();
    let mut got = false;
    for _ in 0..100 {
        hb.admit(&src, "Poll").unwrap();
        let s = format!("{:?}", hb.state());
        if s.contains("hello b") && s.contains("ciphertext:9f2a") && s.contains("typing") {
            got = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(got, "{:?}", hb.state());
    ha.admit(&src, &format!(r#"Poll, Send(peer: "127.0.0.1:1", body: "x"), Send(peer: "{dead}", body: "x")"#)).unwrap();
    let s = format!("{:?}", ha.state());
    assert!(s.contains("sent\", Int(3") && s.contains("lost\", Int(2"), "{s}");
    let sb_state = format!("{:?}", hb.state());
    assert_eq!(sb_state.matches("hello b").count(), 1, "each message once");
    let (la, lb) = (ha.state_hash().to_string(), hb.state_hash().to_string());
    drop(ha);
    drop(hb);
    let t = |st: &MemStore| st.lines.iter().map(|l| format!("{l}\n")).collect::<String>();
    assert!(t(&sa).contains("unreachable"));
    assert_eq!(replay(&m, &t(&sa)).unwrap().1, la);
    assert_eq!(replay(&m, &t(&sb)).unwrap().1, lb);
}
