//! MLS message framing in pure RCE (crates/rce/lib/framing.rce) against the
//! official RFC 9420 messages.json (all 300 entries): every PublicMessage
//! (application, proposal, commit) and PrivateMessage decodes into its fields
//! and encodes back to the same bytes, with the expected wire format and
//! content type, and every other structure in the file (KeyPackage, Welcome,
//! GroupInfo, GroupSecrets, ratchet tree, the seven proposal bodies, Commit)
//! spans exactly its bytes. Truncated or extended messages do not decode. One
//! structure per host input; the log replays.
use rce_spec_check::host::{Host, MemStore, replay};
use rce_spec_check::parse::parse;
use serde_json::Value;
use std::path::Path;
#[path = "support/flat.rs"]
mod flat;

fn combined() -> String {
    let mut s = String::from("# rce-version: 0.5\nmodule framing_steps\n");
    s.push_str(&flat::flat(&["crypto", "hpke", "mls", "treekem", "framing"], false));
    s + APP
}

const APP: &str = r##"
event Msg(h: text)
event Walk(what: text, h: text)

state S
    out: List[text] = []
    seq: i64 = 0

step S on Msg(h: h)
    seq += 1
    let f = mls_decode(copy(h))
    if f.len() == 0
        out = ["O:" + seq.tostr() + "#kept"]
    else
        let ok = mls_encode(copy(f)) == h
        let ctext = if f.len() == 10 then copy(f[5]) else copy(f[3])
        out = ["O:" + seq.tostr() + "#" + ok.tostr() + "/" + copy(f[0]) + ctext]

step S on Walk(what: k, h: h)
    seq += 1
    out = ["O:" + seq.tostr() + "#" + walk_ok(k, h).tostr()]
"##;

#[test]
fn official_messages_vectors() {
    let src = combined();
    let m = parse(&src).unwrap();
    let v: Value = serde_json::from_str(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/mls/messages.json")).unwrap()).unwrap();
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
    let walks = [
        ("add", "add_proposal"),
        ("update", "update_proposal"),
        ("remove", "remove_proposal"),
        ("psk", "pre_shared_key_proposal"),
        ("reinit", "re_init_proposal"),
        ("external_init", "external_init_proposal"),
        ("gce", "group_context_extensions_proposal"),
        ("commit", "commit"),
        ("key_package", "mls_key_package"),
        ("group_info", "mls_group_info"),
        ("welcome", "mls_welcome"),
        ("group_secrets", "group_secrets"),
        ("ratchet_tree", "ratchet_tree"),
    ];
    let mut count = 0;
    for (i, t) in v.as_array().unwrap().iter().enumerate() {
        for (field, want) in [("public_message_application", "true/0101"), ("public_message_proposal", "true/0102"), ("public_message_commit", "true/0103")] {
            assert_eq!(step(format!(r#"Msg(h: "{}")"#, t[field].as_str().unwrap())), want, "messages {i}: {field}");
        }
        let pm = step(format!(r#"Msg(h: "{}")"#, t["private_message"].as_str().unwrap()));
        assert!(pm.starts_with("true/02"), "messages {i}: private_message {pm}");
        for (kind, field) in walks {
            assert_eq!(step(format!(r#"Walk(what: "{kind}", h: "{}")"#, t[field].as_str().unwrap())), "true", "messages {i}: {field}");
        }
        count += 1;
    }
    assert_eq!(count, 300);
    // malformed: one byte short, one byte extra, unknown wire format
    let pa = v[0]["public_message_application"].as_str().unwrap();
    for bad in [&pa[..pa.len() - 2], &format!("{pa}00"), &format!("0001000f{}", &pa[8..])] {
        assert_eq!(step(format!(r#"Msg(h: "{bad}")"#)), "kept");
    }
    let c = v[0]["commit"].as_str().unwrap();
    assert_eq!(step(format!(r#"Walk(what: "commit", h: "{}")"#, &c[..c.len() - 2])), "false");
    let live = h.state_hash().to_string();
    drop(h);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert_eq!(replay(&m, &text).unwrap().1, live);
}
