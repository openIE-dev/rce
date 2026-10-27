//! W1i adapters: TCP (the P4 transport), random bytes and content-addressed
//! blobs. Every result is logged, so replay with every adapter
//! disconnected reaches the same state (HC-3, HC-5, HC-8, HC-10). The net
//! adapter delivers the bytes it received exactly, as `List[i64]` (HC-10a),
//! and its sends never wait for the peer (HC-23).
use rce_spec_check::adapters::{RandomAdapter, blob_adapters, net::net_adapters};
use rce_spec_check::host::{Adapter, Host, MemStore, replay};
use rce_spec_check::interp::V;
use rce_spec_check::parse::parse;
use serde_json::json;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicU16, Ordering};
use std::time::Duration;

fn net_src() -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/net.rce")).unwrap()
}

/// A loopback address that no other test of this process is handed. Ports
/// come from a counter below every platform's ephemeral range (Linux starts
/// at 32768, macOS and Windows at 49152), so neither a concurrent test's
/// `bind("127.0.0.1:0")` nor a connection's local port can take the port
/// between this probe and the bind of the host under test.
fn free_addr() -> String {
    static NEXT: AtomicU16 = AtomicU16::new(0);
    let base = 20000 + (std::process::id() % 1000) as u16 * 10;
    loop {
        let port = base + NEXT.fetch_add(1, Ordering::Relaxed);
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return format!("127.0.0.1:{port}");
        }
    }
}

fn boxed(xs: Vec<rce_spec_check::adapters::net::NetAdapter>) -> Vec<Box<dyn Adapter>> {
    xs.into_iter().map(|a| Box::new(a) as Box<dyn Adapter>).collect()
}

/// The value of state field `name`.
fn field(h: &Host, name: &str) -> V {
    h.state().iter().flat_map(|(_, fs)| fs.iter()).find(|(n, _, _)| n == name).map(|(_, v, _)| v.clone()).unwrap()
}

fn ints(v: &V) -> Vec<i64> {
    match v {
        V::List(xs) => xs.iter().map(|x| if let V::Int(i, _) = x { *i as i64 } else { panic!("{x:?}") }).collect(),
        _ => panic!("{v:?}"),
    }
}

fn int(v: &V) -> i64 {
    if let V::Int(i, _) = v { *i as i64 } else { panic!("{v:?}") }
}

fn log_text(st: &MemStore) -> String {
    st.lines.iter().map(|l| format!("{l}\n")).collect()
}

#[test]
fn tcp_loopback_random_and_blobs_replay() {
    let src = net_src();
    let m = parse(&src).unwrap();
    let addr = free_addr();
    let root = std::env::temp_dir().join(format!("rce-blob-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let mut ads = boxed(net_adapters(std::slice::from_ref(&addr)));
    ads.push(Box::new(RandomAdapter));
    ads.extend(blob_adapters(&root).into_iter().map(|a| Box::new(a) as Box<dyn Adapter>));
    let mut st = MemStore::default();
    let mut h = Host::start_with(&m, Box::new(&mut st), ads).unwrap();
    // listener is id 1, client id 2, accepted server side id 3
    h.admit(&src, &format!(r#"Listen(addr: "{addr}"), Connect(addr: "{addr}")"#)).unwrap();
    let mut accepted = false;
    for _ in 0..200 {
        h.admit(&src, "Accept(id: 1)").unwrap();
        if int(&field(&h, "last")) == 3 {
            accepted = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(accepted, "{:?}", h.state());
    h.admit(&src, r#"Send(id: 2, data: "hello over tcp")"#).unwrap();
    let mut ok = false;
    for _ in 0..200 {
        h.admit(&src, "Recv(id: 3)").unwrap();
        if ints(&field(&h, "got")) == b"hello over tcp".iter().map(|b| *b as i64).collect::<Vec<_>>() {
            ok = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
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
    assert_eq!(replay(&m, &log_text(&st)).unwrap().1, live);
    let _ = std::fs::remove_dir_all(&root);
}

/// HC-10a: a read delivers exactly the bytes that arrived, so a character
/// split across two writes, and a byte that is not UTF-8, arrive as sent;
/// D records the lists, and replay rebuilds them.
#[test]
fn recv_delivers_the_bytes_received() {
    let src = net_src();
    let m = parse(&src).unwrap();
    let addr = free_addr();
    let mut st = MemStore::default();
    let mut h = Host::start_with(&m, Box::new(&mut st), boxed(net_adapters(std::slice::from_ref(&addr)))).unwrap();
    h.admit(&src, &format!(r#"Listen(addr: "{addr}")"#)).unwrap();
    let mut client = TcpStream::connect(&addr).unwrap();
    for _ in 0..200 {
        h.admit(&src, "Accept(id: 1)").unwrap();
        if int(&field(&h, "last")) == 2 {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(int(&field(&h, "last")), 2);
    let recv_until = |h: &mut Host, n: usize| {
        for _ in 0..400 {
            h.admit(&src, "Recv(id: 2)").unwrap();
            if ints(&field(h, "got")).len() >= n {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("received {:?}", field(h, "got"));
    };
    // 0xFF is never UTF-8; E2 82 AC is the euro sign, split after two bytes
    client.write_all(&[255, 226, 130]).unwrap();
    recv_until(&mut h, 3);
    client.write_all(&[172]).unwrap();
    recv_until(&mut h, 4);
    assert_eq!(ints(&field(&h, "got")), vec![255, 226, 130, 172]);
    let live = h.state_hash().to_string();
    drop(h);
    let text = log_text(&st);
    assert!(text.contains(r#""data":[255,226,130]"#), "{text}");
    assert!(text.contains(r#""data":[172]"#), "{text}");
    assert!(!text.contains("\\ufffd") && !text.contains('\u{fffd}'), "no byte was replaced");
    assert_eq!(replay(&m, &text).unwrap().1, live);
}

/// HF-3 (E.4c): received bytes bound to a `text` result. The host starts
/// and says how it fits them: bytes that are UTF-8 are delivered as text;
/// a character split across two writes waits in the adapter for the next
/// read, so it is never split or replaced; a read that is not UTF-8, and a
/// stream that ends inside a character, are delivered as failed with the
/// offset, while `host.Unfit` keeps the bytes in D, so nothing received is
/// lost. Replay gives the live state.
#[test]
fn recv_into_a_text_field_is_text_when_utf8() {
    let src = net_src()
        .replace("event NetRecvDone(req: u64, status: text, data: List[i64]?, err: text)", "event NetRecvDone(req: u64, status: text, data: text?, err: text)")
        .replace("step Net on NetRecvDone(req: _r, status: s, data: d, err: _e)\n    if s == \"delivered\"\n        for x in d.or([])\n            got.push(x)\n", "step Net on NetRecvDone(req: _r, status: s, data: d, err: e)\n    if s == \"delivered\"\n        rand = rand + d.or(\"\")\n    else\n        blob = copy(e)\n        errors += 1\n");
    assert!(src.contains("rand = rand + d.or"));
    let m = parse(&src).unwrap();
    let addr = free_addr();
    let mut st = MemStore::default();
    let mut h = Host::start_with(&m, Box::new(&mut st), boxed(net_adapters(std::slice::from_ref(&addr)))).unwrap();
    assert!(h.fitting().iter().any(|n| n.starts_with("event NetRecvDone:") && n.contains("delivered as text when they are UTF-8")), "{:?}", h.fitting());
    h.admit(&src, &format!(r#"Listen(addr: "{addr}")"#)).unwrap();
    let mut client = TcpStream::connect(&addr).unwrap();
    for _ in 0..200 {
        h.admit(&src, "Accept(id: 1)").unwrap();
        if int(&field(&h, "last")) == 2 {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let text_of = |v: V| if let V::Text(t) = v { t.to_string() } else { panic!("{v:?}") };
    let recv_until = |h: &mut Host, done: &dyn Fn(&Host) -> bool| {
        for _ in 0..400 {
            h.admit(&src, "Recv(id: 2)").unwrap();
            if done(h) {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("received {:?}", field(h, "rand"));
    };
    client.write_all("h\u{e9}".as_bytes()).unwrap();
    recv_until(&mut h, &|h| text_of(field(h, "rand")) == "h\u{e9}");
    client.write_all(&[0xFF]).unwrap();
    recv_until(&mut h, &|h| int(&field(h, "errors")) == 1);
    assert_eq!(text_of(field(&h, "blob")), "result does not fit the declared type: not UTF-8 at offset 0");
    // E2 82 AC is the euro sign, written in two parts
    client.write_all(&[0xE2, 0x82]).unwrap();
    std::thread::sleep(Duration::from_millis(50));
    for _ in 0..5 {
        h.admit(&src, "Recv(id: 2)").unwrap();
    }
    assert_eq!((text_of(field(&h, "rand")), int(&field(&h, "errors"))), ("h\u{e9}".to_string(), 1), "the first two bytes wait for the third");
    client.write_all(&[0xAC]).unwrap();
    recv_until(&mut h, &|h| text_of(field(h, "rand")) == "h\u{e9}\u{20ac}");
    // the stream ends inside a character: delivered as it is, and failed
    client.write_all(&[0xC3]).unwrap();
    client.shutdown(std::net::Shutdown::Write).unwrap();
    recv_until(&mut h, &|h| int(&field(h, "errors")) >= 2);
    assert_eq!(text_of(field(&h, "blob")), "result does not fit the declared type: incomplete UTF-8 character at offset 0");
    let live = h.state_hash().to_string();
    drop(h);
    let text = log_text(&st);
    assert!(text.contains(r#""data":"h\u00e9""#) || text.contains("\"data\":\"h\u{e9}\""), "{text}");
    assert!(text.contains(r#""data":[255]"#) && text.contains(r#""why":"data: not UTF-8 at offset 0""#), "D keeps the bytes: {text}");
    assert!(text.contains(r#""data":[195]"#) && text.contains(r#""why":"data: incomplete UTF-8 character at offset 0""#), "D keeps the bytes: {text}");
    assert!(text.contains("\"data\":\"\u{20ac}\"") || text.contains(r#""data":"\u20ac""#), "the euro sign is one delivery: {text}");
    assert_eq!(replay(&m, &text).unwrap().1, live);
}

/// HF-3: taken as text, a read does not end inside a character even when
/// max is smaller than the character: it reads up to 3 bytes past max to
/// complete it, so a reader of one byte at a time still gets each
/// character and the end of the stream. A character whose bytes have not
/// arrived waits for the next read, and a `NetClose` of a connection that
/// keeps bytes lists them in its err, so D keeps them.
#[test]
fn recv_as_text_completes_a_character_past_max() {
    let addr = free_addr();
    let mut a = net_adapters(std::slice::from_ref(&addr));
    for x in a.iter_mut() {
        x.text();
    }
    let mut call = |k: &str, r: serde_json::Value| a.iter_mut().find(|x| x.kind() == k).unwrap().perform(&r);
    assert_eq!(call("NetListen", json!({"req": 1, "addr": addr})).status.as_str(), "delivered");
    let mut peer = TcpStream::connect(&addr).unwrap();
    let mut conn = None;
    for _ in 0..200 {
        let r = call("NetAccept", json!({"req": 2, "id": 1}));
        if r.status.as_str() == "delivered" {
            conn = r.data.and_then(|d| d.as_u64());
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let id = conn.expect("accepted");
    let recv = |call: &mut dyn FnMut(&str, serde_json::Value) -> rce_spec_check::host::EffectResult| -> rce_spec_check::host::EffectResult {
        let t0 = std::time::Instant::now();
        loop {
            let r = call("NetRecv", json!({"req": 3, "id": id, "max": 1}));
            if r.data.as_ref().and_then(|d| d.as_array()).is_none_or(|d| !d.is_empty()) || t0.elapsed() > Duration::from_secs(10) {
                return r;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    };
    peer.write_all("\u{e9}a".as_bytes()).unwrap();
    assert_eq!(recv(&mut call).data, Some(json!([0xC3, 0xA9])), "two bytes for a max of one: the whole character");
    assert_eq!(recv(&mut call).data, Some(json!([0x61])));
    // the first byte of a character whose rest has not arrived waits
    peer.write_all(&[0xE2]).unwrap();
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(call("NetRecv", json!({"req": 4, "id": id, "max": 1})).data, Some(json!([])));
    let c = call("NetClose", json!({"req": 5, "id": id}));
    assert_eq!((c.status.as_str(), c.data.clone()), ("delivered", Some(json!(0))));
    assert!(c.error.contains("[226]"), "the kept byte is in the result: {c:?}");
}

/// HC-23: NetSend takes text (its UTF-8) or bytes; an element outside
/// 0..255 fails the request and nothing is sent.
#[test]
fn send_takes_text_or_bytes() {
    let addr = free_addr();
    let mut a = net_adapters(std::slice::from_ref(&addr));
    let mut call = |k: &str, r: serde_json::Value| a.iter_mut().find(|x| x.kind() == k).unwrap().perform(&r);
    let l = call("NetListen", json!({"req": 1, "addr": addr}));
    assert_eq!(l.status.as_str(), "delivered", "{l:?}");
    let mut peer = TcpStream::connect(&addr).unwrap();
    let mut conn = None;
    for _ in 0..200 {
        let r = call("NetAccept", json!({"req": 2, "id": 1}));
        if r.status.as_str() == "delivered" {
            conn = r.data.and_then(|d| d.as_u64());
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let c = conn.expect("accepted");
    let t = call("NetSend", json!({"req": 3, "id": c, "data": "hi €"}));
    assert_eq!(t.data, Some(json!(6)), "{t:?}");
    let b = call("NetSend", json!({"req": 4, "id": c, "data": [0, 255, 10]}));
    assert_eq!(b.data, Some(json!(3)), "{b:?}");
    let bad = call("NetSend", json!({"req": 5, "id": c, "data": [1, 256]}));
    assert_eq!((bad.status.as_str(), bad.error.as_str()), ("failed", "byte outside 0..255"));
    let neg = call("NetSend", json!({"req": 6, "id": c, "data": [-1]}));
    assert_eq!(neg.error, "byte outside 0..255");
    let none = call("NetSend", json!({"req": 7, "id": 99, "data": "x"}));
    assert_eq!(none.error, "no connection 99");
    call("NetClose", json!({"req": 8, "id": c}));
    let mut got = vec![];
    peer.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    peer.read_to_end(&mut got).unwrap();
    let mut want = "hi €".as_bytes().to_vec();
    want.extend([0, 255, 10]);
    assert_eq!(got, want, "the bytes handed over are written before the close");
}

/// HC-10a, N9: a read never delivers more than SIZE_LIMIT bytes, so the
/// result can always be built and the service keeps serving.
#[test]
fn recv_is_capped_at_the_size_limit() {
    let src = r#"module bigrecv

struct NetListen
    req: u64
    addr: text

struct NetAccept
    req: u64
    id: u64

struct NetRecv
    req: u64
    id: u64
    max: u64

event Listen(addr: text)
event Accept(id: u64)
event Recv(id: u64)
event NetListenDone(req: u64, status: text, data: u64?, err: text)
event NetAcceptDone(req: u64, status: text, data: u64?, err: text)
event NetRecvDone(req: u64, status: text, data: List[i64]?, err: text)

state S
    next: u64 = 0
    conn: u64 = 0
    total: usize = 0
    largest: usize = 0
    reads: u64 = 0

step S on Listen(addr: a)
    next += 1
    emit NetListen(req: next, addr: a)

step S on Accept(id: i)
    next += 1
    emit NetAccept(req: next, id: i)

step S on Recv(id: i)
    next += 1
    emit NetRecv(req: next, id: i, max: 2000000)

step S on NetListenDone(_, _, _, _)
    pass

step S on NetAcceptDone(req: _r, status: s, data: d, err: _e)
    if s == "delivered"
        conn = d.or(0)

step S on NetRecvDone(req: _r, status: s, data: d, err: _e)
    if s == "delivered"
        let n = d.or([]).len()
        total += n
        reads += 1
        if n > largest
            largest = n
"#;
    let m = parse(src).unwrap();
    let addr = free_addr();
    let mut st = MemStore::default();
    let mut h = Host::start_with(&m, Box::new(&mut st), boxed(net_adapters(std::slice::from_ref(&addr)))).unwrap();
    h.admit(src, &format!(r#"Listen(addr: "{addr}")"#)).unwrap();
    let target = addr.clone();
    let sender = std::thread::spawn(move || {
        let mut c = TcpStream::connect(&target).unwrap();
        c.write_all(&vec![7u8; 1_500_000]).unwrap();
        c
    });
    for _ in 0..400 {
        h.admit(src, "Accept(id: 1)").unwrap();
        if int(&field(&h, "conn")) != 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let conn = int(&field(&h, "conn"));
    assert!(conn != 0);
    let _client = sender.join().unwrap();
    for _ in 0..400 {
        h.admit(src, &format!("Recv(id: {conn})")).unwrap();
        if int(&field(&h, "total")) >= 1_500_000 {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(int(&field(&h, "total")), 1_500_000, "every byte arrived and no step was refused");
    let largest = int(&field(&h, "largest"));
    assert!(largest <= 1_000_000 && largest > 0, "largest read {largest}");
    let live = h.state_hash().to_string();
    drop(h);
    assert_eq!(replay(&m, &log_text(&st)).unwrap().1, live);
}
