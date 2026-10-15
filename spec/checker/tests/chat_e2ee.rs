//! W4 chat E2EE: two replicas with the logged Random adapter (HC-10) and
//! P4 channels. Each makes its keys, signs its prekey with Ed25519 and
//! broadcasts the bundle; each verifies the other's signature; the
//! initiator runs X3DH; DMs then flow both ways through one Double Ratchet
//! per peer. Only ciphertext crosses the channel; Show decrypts with
//! `.utf8()`; each D replays with no adapters. apps/chat.rce must equal
//! scripts/build-chat.sh's output.
use rce_spec_check::adapters::{RandomAdapter, chan::chan_adapters};
use rce_spec_check::ast::{Arg, Expr};
use rce_spec_check::host::{Adapter, Host, MemStore, replay};
use rce_spec_check::interp::V;
use rce_spec_check::parse::parse;
use std::path::Path;

fn port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn show(h: &mut Host, conv: &str) -> Vec<String> {
    let a = |l: &str, v: &str| Arg { label: Some(l.into()), e: Expr::Str(v.into(), 0) };
    h.query("Show", vec![a("conv", conv), a("topic", "")]).unwrap().into_iter().filter_map(|o| match o {
        V::Rec(_, fs) => fs.into_iter().find_map(|(f, v)| match v { V::Text(t) if f == "data" => Some(t), _ => None }),
        _ => None,
    }).collect()
}

fn region(h: &Host, field: &str) -> String {
    format!("{:?}", h.state()[0].1.iter().find(|(n, _, _)| n == field).unwrap().1)
}

#[test]
fn generated_chat_is_in_sync() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let app = std::fs::read_to_string(root.join("apps/chat-app.rce")).unwrap();
    let lib = std::fs::read_to_string(root.join("lib/crypto.rce")).unwrap();
    let mut want = app;
    want.push_str("\n# ---------------------------------------------------------------- lib/crypto.rce (generated, do not edit here)\n");
    for l in lib.lines().filter(|l| *l != "module crypto" && !l.starts_with("check ")) {
        want.push_str(l);
        want.push('\n');
    }
    assert!(std::fs::read_to_string(root.join("apps/chat.rce")).unwrap() == want, "apps/chat.rce is stale: run scripts/build-chat.sh");
}

#[test]
fn dms_are_end_to_end_encrypted_and_replay() {
    let src = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../apps/chat.rce")).unwrap();
    let m = parse(&src).unwrap();
    let addrs: Vec<String> = (0..2).map(|_| format!("127.0.0.1:{}", port())).collect();
    let mut stores: Vec<MemStore> = (0..2).map(|_| MemStore::default()).collect();
    let mut hosts = vec![];
    for (i, st) in stores.iter_mut().enumerate() {
        let peer = addrs[1 - i].clone();
        let mut ads: Vec<Box<dyn Adapter>> = chan_adapters(&addrs[i], &[peer.clone()]).unwrap().into_iter().map(|x| Box::new(x) as Box<dyn Adapter>).collect();
        ads.push(Box::new(RandomAdapter));
        let mut h = Host::start_with(&m, Box::new(st), ads).unwrap();
        h.admit(&src, &format!(r#"Setup(me: "{}", peers: ["{peer}"])"#, ["ann", "bob"][i])).unwrap();
        hosts.push(h);
    }
    // queued before any session exists: sent once the ratchet is ready
    hosts[0].admit(&src, r#"Do(op: "dm", conv: "bob", topic: "", body: "hi bob, café ☕")"#).unwrap();
    let conv = "dm:ann,bob";
    let mut replied = false;
    for _ in 0..400 {
        for h in hosts.iter_mut() {
            h.admit(&src, "Poll").unwrap();
        }
        let b = show(&mut hosts[1], conv);
        if !replied && b.iter().any(|l| l.contains("café")) {
            hosts[1].admit(&src, r#"Do(op: "dm", conv: "ann", topic: "", body: "hi ann")"#).unwrap();
            replied = true;
        }
        if replied && show(&mut hosts[0], conv).iter().any(|l| l.contains("hi ann")) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let a = show(&mut hosts[0], conv);
    let b = show(&mut hosts[1], conv);
    assert!(a.contains(&"ann: hi bob, café ☕\n".to_string()) && a.contains(&"bob: hi ann\n".to_string()), "ann sees {a:?}");
    assert!(b.contains(&"ann: hi bob, café ☕\n".to_string()) && b.contains(&"bob: hi ann\n".to_string()), "bob sees {b:?}");
    for h in &hosts {
        assert!(region(h, "rejected").starts_with("Int(0"), "a bundle or message was rejected");
        assert!(region(h, "bundles").contains("Text"), "peer bundle verified and kept");
    }
    let lives: Vec<String> = hosts.iter().map(|h| h.state_hash().to_string()).collect();
    drop(hosts);
    // no plaintext crossed the channel: every logged received line is ciphertext
    for st in &stores {
        for l in st.lines.iter().filter(|l| l.contains("ChanRecvDone")) {
            assert!(!l.contains("hi ann") && !l.contains("hi bob"), "plaintext on the wire: {l}");
        }
    }
    for (st, live) in stores.iter().zip(lives) {
        let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
        assert_eq!(replay(&m, &text).unwrap().1, live);
    }
}
