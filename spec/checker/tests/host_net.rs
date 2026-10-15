//! W1i adapters: TCP (the P4 transport), random bytes and content-addressed
//! blobs. Every result is logged, so replay with every adapter
//! disconnected reaches the same state (HC-3, HC-5, HC-8, HC-10).
use rce_spec_check::adapters::{RandomAdapter, blob_adapters, net::net_adapters};
use rce_spec_check::host::{Adapter, Host, MemStore, replay};
use rce_spec_check::parse::parse;
use std::path::Path;

#[test]
fn tcp_loopback_random_and_blobs_replay() {
    let src = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/net.rce")).unwrap();
    let m = parse(&src).unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let addr = format!("127.0.0.1:{port}");
    let root = std::env::temp_dir().join(format!("rce-blob-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let mut ads: Vec<Box<dyn Adapter>> = net_adapters(std::slice::from_ref(&addr)).into_iter().map(|a| Box::new(a) as Box<dyn Adapter>).collect();
    ads.push(Box::new(RandomAdapter));
    ads.extend(blob_adapters(&root).into_iter().map(|a| Box::new(a) as Box<dyn Adapter>));
    let mut st = MemStore::default();
    let mut h = Host::start_with(&m, Box::new(&mut st), ads).unwrap();
    // listener is id 1, client id 2, accepted server side id 3
    h.admit(&src, &format!(r#"Listen(addr: "{addr}"), Connect(addr: "{addr}")"#)).unwrap();
    let mut accepted = false;
    for _ in 0..200 {
        h.admit(&src, "Accept(id: 1)").unwrap();
        if format!("{:?}", h.state()).contains("\"last\", Int(3") {
            accepted = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(accepted, "{:?}", h.state());
    h.admit(&src, r#"Send(id: 2, data: "hello over tcp")"#).unwrap();
    let mut ok = false;
    for _ in 0..200 {
        h.admit(&src, "Recv(id: 3)").unwrap();
        if format!("{:?}", h.state()).contains("hello over tcp") {
            ok = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(ok, "{:?}", h.state());
    // an address that was not granted fails: authority is explicit (HC-5)
    h.admit(&src, r#"Connect(addr: "127.0.0.1:9"), Close(id: 2), Roll(n: 16), Put(data: "a file body")"#).unwrap();
    let s = format!("{:?}", h.state());
    let digest = s.split("\"digest\", Text(\"").nth(1).unwrap().split('"').next().unwrap().to_string();
    assert_eq!(digest.len(), 64);
    h.admit(&src, &format!(r#"Get(digest: "{digest}"), Get(digest: "../etc")"#)).unwrap();
    let s = format!("{:?}", h.state());
    assert!(s.contains("\"blob\", Text(\"a file body\")"), "{s}");
    assert!(s.contains("errors\", Int(2"), "{s}");
    let rand = s.split("\"rand\", Text(\"").nth(1).unwrap().split('"').next().unwrap();
    assert_eq!(rand.len(), 32);
    let live = h.state_hash().to_string();
    drop(h);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert_eq!(replay(&m, &text).unwrap().1, live);
    let _ = std::fs::remove_dir_all(&root);
}
