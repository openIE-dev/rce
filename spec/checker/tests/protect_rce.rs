//! MLS message protection in pure RCE (crates/rce/lib/protect.rce) against
//! the official RFC 9420 vectors for cipher suite 1: secret-tree.json (all 3
//! trees of 1, 8 and 32 leaves: sender data key and nonce, and the
//! application and handshake key and nonce of every leaf at every listed
//! generation) and message-protection.json (the application, proposal and
//! commit PrivateMessages open to the expected content with a valid
//! FramedContentTBS signature, and the proposal and commit PublicMessages
//! carry a valid signature and membership tag). A changed ciphertext or
//! membership key is refused. One computation per host input; the log replays.
use rce_spec_check::host::{Host, MemStore, replay};
use rce_spec_check::parse::parse;
use serde_json::Value;
use std::path::Path;

fn combined() -> String {
    let mut s = String::from("module protect_steps\n");
    for f in ["lib/crypto.rce", "lib/hpke.rce", "lib/mls.rce", "lib/treekem.rce", "lib/framing.rce", "lib/protect.rce"] {
        for l in std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(f)).unwrap().lines() {
            if l.starts_with("module ") || l.starts_with("check ") {
                continue;
            }
            s.push_str(l);
            s.push('\n');
        }
    }
    s + APP
}

const APP: &str = r##"
event LeafKey(es: text, n: i64, i: i64, label: text, g: i64)
event SdKey(sds: text, ct: text)
event Open(msg: text, es: text, sds: text, n: i64, gc: text, pk: text)
event Pub(msg: text, gc: text, pk: text, mk: text)
event Gc(gid: text, ep: i64, th: text, cth: text)

state S
    out: List[text] = []
    seq: i64 = 0

step S on LeafKey(es: es, n: n, i: i, label: label, g: g)
    seq += 1
    out = ["O:" + seq.tostr() + "#" + st_key_nonce(st_leaf_secret(es, n, i), label, g)]

step S on SdKey(sds: sds, ct: ct)
    seq += 1
    out = ["O:" + seq.tostr() + "#" + sd_key_nonce(sds, ct)]

step S on Open(msg: msg, es: es, sds: sds, n: n, gc: gc, pk: pk)
    seq += 1
    let o = priv_open(copy(msg), es, sds, n)
    if o.len() == 0
        out = ["O:" + seq.tostr() + "#refused"]
    else
        let ok = priv_signature_ok(msg, copy(o), gc, pk)
        out = ["O:" + seq.tostr() + "#" + copy(o[0]) + "/" + copy(o[1]) + "/" + copy(o[3]) + "/" + ok.tostr()]

step S on Pub(msg: msg, gc: gc, pk: pk, mk: mk)
    seq += 1
    out = ["O:" + seq.tostr() + "#" + pub_ok(msg, gc, pk, mk).tostr()]

step S on Gc(gid: gid, ep: ep, th: th, cth: cth)
    seq += 1
    out = ["O:" + seq.tostr() + "#" + group_context(gid, ep, th, cth)]
"##;

fn vectors(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/mls").join(name)).unwrap()).unwrap()
}

#[test]
fn official_secret_tree_and_message_protection_vectors() {
    let src = combined();
    let m = parse(&src).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    let mut n = 0u64;
    let mut step = |ev: String| -> String {
        n += 1;
        h.admit(&src, &ev).unwrap();
        let s = format!("{:?}", h.state());
        let o = s.split("Text(\"O:").nth(1).unwrap().split('"').next().unwrap().to_string();
        let (k, r) = o.split_once('#').unwrap();
        assert_eq!(k.parse::<u64>().unwrap(), n, "step admitted within its fuel: {ev:.60}");
        r.to_string()
    };
    let mut keys = 0;
    for (e, t) in vectors("secret-tree-suite1.json").as_array().unwrap().iter().enumerate() {
        let sd = &t["sender_data"];
        assert_eq!(step(format!(r#"SdKey(sds: "{}", ct: "{}")"#, sd["sender_data_secret"].as_str().unwrap(), sd["ciphertext"].as_str().unwrap())), format!("{}.{}", sd["key"].as_str().unwrap(), sd["nonce"].as_str().unwrap()), "secret-tree {e}: sender data");
        let leaves = t["leaves"].as_array().unwrap();
        let es = t["encryption_secret"].as_str().unwrap();
        for (i, gens) in leaves.iter().enumerate() {
            for g in gens.as_array().unwrap() {
                for label in ["application", "handshake"] {
                    let want = format!("{}.{}", g[format!("{label}_key")].as_str().unwrap(), g[format!("{label}_nonce")].as_str().unwrap());
                    assert_eq!(step(format!(r#"LeafKey(es: "{es}", n: {}, i: {i}, label: "{label}", g: {})"#, leaves.len(), g["generation"])), want, "secret-tree {e}: leaf {i} {label}");
                    keys += 1;
                }
            }
        }
    }
    assert_eq!(keys, (1 + 8 + 32) * 2 * 2);
    let mp = &vectors("message-protection-suite1.json")[0];
    let s = |k: &str| mp[k].as_str().map(|x| x.to_string()).unwrap_or_else(|| mp[k].to_string());
    let gc = step(format!(r#"Gc(gid: "{}", ep: {}, th: "{}", cth: "{}")"#, s("group_id"), s("epoch"), s("tree_hash"), s("confirmed_transcript_hash")));
    for (kind, ct) in [("application", "01"), ("proposal", "02"), ("commit", "03")] {
        // application content is application_data<V>: one length byte below 64
        let content = if kind == "application" {
            assert!(s(kind).len() / 2 < 64);
            format!("{:02x}{}", s(kind).len() / 2, s(kind))
        } else {
            s(kind)
        };
        let got = step(format!(r#"Open(msg: "{}", es: "{}", sds: "{}", n: 2, gc: "{gc}", pk: "{}")"#, s(&format!("{kind}_priv")), s("encryption_secret"), s("sender_data_secret"), s("signature_pub")));
        assert_eq!(got, format!("00000001/{ct}/{content}/true"), "message-protection: {kind}_priv");
    }
    for kind in ["proposal", "commit"] {
        assert_eq!(step(format!(r#"Pub(msg: "{}", gc: "{gc}", pk: "{}", mk: "{}")"#, s(&format!("{kind}_pub")), s("signature_pub"), s("membership_key"))), "true", "message-protection: {kind}_pub");
    }
    // negatives: a flipped ciphertext byte, a wrong membership key
    let mut bad = s("application_priv");
    let last = bad.len() - 2;
    let flipped = format!("{:02x}", u8::from_str_radix(&bad[last..], 16).unwrap() ^ 1);
    bad.replace_range(last.., &flipped);
    assert_eq!(step(format!(r#"Open(msg: "{bad}", es: "{}", sds: "{}", n: 2, gc: "{gc}", pk: "{}")"#, s("encryption_secret"), s("sender_data_secret"), s("signature_pub"))), "refused");
    assert_eq!(step(format!(r#"Pub(msg: "{}", gc: "{gc}", pk: "{}", mk: "{}")"#, s("commit_pub"), s("signature_pub"), "00".repeat(32))), "false");
    let live = h.state_hash().to_string();
    drop(h);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert_eq!(replay(&m, &text).unwrap().1, live);
}
