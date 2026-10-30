//! Joining an MLS group from a Welcome in pure RCE (stdlib/lib/welcome.rce
//! on stdlib/lib/crypto.rce, stdlib/lib/hpke.rce, stdlib/lib/mls.rce and stdlib/lib/treekem.rce) against
//! the official RFC 9420 vectors for cipher suite 1: welcome.json (GroupSecrets
//! opened with the KeyPackage init key, GroupInfo opened, its signature and
//! confirmation tag verified) and passive-client-welcome.json (all 8 entries,
//! external PSKs included: the ratchet tree, from the vector or the GroupInfo
//! extension, hashes to the GroupContext tree hash, the signer's leaf verifies
//! the GroupInfo, and the epoch authenticator matches). One step per input;
//! a wrong PSK does not open the GroupInfo; the log replays.
use rce_spec_check::host::{Host, MemStore, replay};
use rce_spec_check::parse::parse;
use serde_json::Value;
use std::path::Path;
#[path = "support/flat.rs"]
mod flat;

fn combined() -> String {
    let mut s = String::from("# rce-version: 0.5\nmodule welcome_steps\n");
    s.push_str(&flat::flat(&["crypto", "hpke", "mls", "treekem", "welcome"], false));
    s + APP
}

const APP: &str = r#"
event WLoad(t: text)
event WOpen(kp: text, init: text)
event WInfo(psks: text)
event WEpoch(z: i64)
event WSigPub(pk: text)
event TLoad(t: text)
event TFromGi(z: i64)
event TStart(z: i64)
event TStep(k: i64)
event THash(z: i64)
event WSigTree(z: i64)

state S
    out: List[text] = []
    wx: text = ""
    gsx: text = ""
    egi: text = ""
    gi: List[i64] = []
    mem: text = ""
    tb: List[i64] = []
    pos: i64 = 0
    tend: i64 = 0
    nodes: List[text] = []

step S on WLoad(t: t)
    wx = wx + t
    out.push("O:ok")

step S on WOpen(kp: kp, init: init)
    let r = welcome_secrets(copy(wx), kp, init)
    wx = ""
    if r == ""
        out.push("O:none")
    else
        let parts = r.split(".")
        gsx = copy(parts[0])
        egi = copy(parts[1])
        out.push("O:" + (gsx != "").tostr())

step S on WInfo(psks: psks)
    let parts = welcome_group_info(copy(gsx), copy(egi), psks).split(".")
    gi = unhex(copy(parts[0]))
    mem = copy(parts[1])
    out.push("O:" + (gi.len() > 1).tostr())

step S on WEpoch(z: z)
    let parts = welcome_epoch(copy(gi), copy(mem)).split(".")
    out.push("O:" + tohex(derive_secret(unhex(copy(parts[0])), "authentication")) + "/" + copy(parts[1]))

step S on WSigPub(pk: pk)
    out.push("O:" + gi_signature_ok(copy(gi), pk).tostr())

step S on TLoad(t: t)
    for x in unhex(t)
        tb.push(x)
    out.push("O:ok")

step S on TFromGi(z: z)
    tb = unhex(gi_ratchet_tree(copy(gi)))
    out.push("O:" + tb.len().tostr())

step S on TStart(z: z)
    let r = rd_v(copy(tb), 0)
    pos = r[1]
    tend = r[0] + r[1]
    nodes = []
    out.push("O:started")

step S on TStep(k: k)
    var j: i64 = 0
    while j < k and pos < tend bound 1024
        let e = rt_node_end(copy(tb), pos as usize)
        nodes.push(rt_node(copy(tb), pos as usize, e))
        pos = e as i64
        j += 1
    if pos < tend
        out.push("O:more")
    else
        nodes = rt_pad(copy(nodes))
        tb = []
        out.push("O:done")

step S on THash(z: z)
    let th = copy(tree_hashes(copy(nodes))[(nodes.len() / 2) as usize])
    out.push("O:" + (th == gi_tree_hash(copy(gi))).tostr())

step S on WSigTree(z: z)
    out.push("O:" + gi_signature_ok(copy(gi), leaf_sig_key(copy(nodes), gi_signer(copy(gi)))).tostr())
"#;

fn vectors(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/mls").join(name)).unwrap()).unwrap()
}

#[test]
fn official_welcome_vectors() {
    let src = combined();
    let m = parse(&src).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    let mut step = |ev: String| -> String {
        let outs = |s: String| -> Vec<String> { s.split("Text(\"O:").skip(1).map(|x| x.split('"').next().unwrap().to_string()).collect() };
        let n0 = outs(format!("{:?}", h.state())).len();
        h.admit(&src, &ev).unwrap();
        let o = outs(format!("{:?}", h.state()));
        assert_eq!(o.len(), n0 + 1, "step admitted within its fuel: {ev:.60}");
        o[n0].clone()
    };
    let join = |t: &Value, step: &mut dyn FnMut(String) -> String| -> String {
        for c in t["welcome"].as_str().unwrap().as_bytes().chunks(2048) {
            step(format!(r#"WLoad(t: "{}")"#, std::str::from_utf8(c).unwrap()));
        }
        assert_eq!(step(format!(r#"WOpen(kp: "{}", init: "{}")"#, t["key_package"].as_str().unwrap(), t["init_priv"].as_str().unwrap())), "true", "our GroupSecrets found and opened");
        let psks: String = t.get("external_psks").and_then(|x| x.as_array()).map(|a| a.iter().map(|e| format!("{}:{},", e["psk_id"].as_str().unwrap(), e["psk"].as_str().unwrap())).collect()).unwrap_or_default();
        assert_eq!(step(format!(r#"WInfo(psks: "{psks}")"#)), "true", "GroupInfo opened");
        step("WEpoch(z: 0)".into())
    };
    // welcome.json
    let w = &vectors("welcome-suite1.json")[0];
    let (_, tag) = join(w, &mut step).split_once('/').map(|(a, b)| (a.to_string(), b.to_string())).unwrap();
    assert_eq!(tag, "true", "welcome.json: confirmation tag");
    assert_eq!(step(format!(r#"WSigPub(pk: "{}")"#, w["signer_pub"].as_str().unwrap())), "true", "welcome.json: GroupInfo signature");
    // passive-client-welcome.json
    let p = vectors("passive-client-welcome-suite1.json");
    let mut n = 0;
    for (i, t) in p.as_array().unwrap().iter().enumerate() {
        let got = join(t, &mut step);
        assert_eq!(got, format!("{}/true", t["initial_epoch_authenticator"].as_str().unwrap()), "passive-client-welcome {i}: epoch authenticator and confirmation tag");
        match t["ratchet_tree"].as_str() {
            Some(rt) => {
                for c in rt.as_bytes().chunks(2048) {
                    step(format!(r#"TLoad(t: "{}")"#, std::str::from_utf8(c).unwrap()));
                }
            }
            None => {
                assert_ne!(step("TFromGi(z: 0)".into()), "0", "passive-client-welcome {i}: ratchet_tree extension present");
            }
        }
        step("TStart(z: 0)".into());
        while step("TStep(k: 8)".into()) == "more" {}
        assert_eq!(step("THash(z: 0)".into()), "true", "passive-client-welcome {i}: tree hash matches the GroupContext");
        assert_eq!(step("WSigTree(z: 0)".into()), "true", "passive-client-welcome {i}: GroupInfo signed by the signer's leaf");
        n += 1;
    }
    assert_eq!(n, 8);
    // a wrong external PSK: the GroupInfo does not open
    let t = &p[2];
    assert!(t["external_psks"].as_array().unwrap().len() == 1);
    for c in t["welcome"].as_str().unwrap().as_bytes().chunks(2048) {
        step(format!(r#"WLoad(t: "{}")"#, std::str::from_utf8(c).unwrap()));
    }
    assert_eq!(step(format!(r#"WOpen(kp: "{}", init: "{}")"#, t["key_package"].as_str().unwrap(), t["init_priv"].as_str().unwrap())), "true");
    let bad = format!("{}:{},", t["external_psks"][0]["psk_id"].as_str().unwrap(), "00".repeat(32));
    assert_eq!(step(format!(r#"WInfo(psks: "{bad}")"#)), "false", "a wrong PSK does not open the GroupInfo");
    let live = h.state_hash().to_string();
    drop(h);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert_eq!(replay(&m, &text).unwrap().1, live);
}
