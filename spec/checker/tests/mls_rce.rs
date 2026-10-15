//! MLS groups in pure RCE (crates/rce/lib/mls.rce on top of lib/crypto.rce)
//! against the official RFC 9420 test vectors of mlswg/mls-implementations
//! (tests/vectors/mls): tree-math.json for every tree in the file, and
//! key-schedule.json for cipher suite 1 (every epoch: joiner, welcome and
//! every derived secret, the next init secret, the exporter (its label is
//! used as the text it is written as in the file) and the
//! external public key), and crypto-basics.json for suite 1
//! (expand_with_label, derive_secret, derive_tree_secret, ref_hash,
//! sign_with_label and its verification). Each operation is one host input inside the fuel
//! of HC-7a, and the log replays to the same state hash.
use rce_spec_check::host::{Host, MemStore, replay};
use rce_spec_check::parse::parse;
use serde_json::Value;
use std::path::Path;

fn combined(checks: bool) -> String {
    let mut s = String::from("module mls_steps\n");
    for f in ["lib/crypto.rce", "lib/mls.rce"] {
        for l in std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(f)).unwrap().lines() {
            if l.starts_with("module ") || (!checks && l.starts_with("check ")) {
                continue;
            }
            s.push_str(l);
            s.push('\n');
        }
    }
    s
}

fn vectors(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/mls").join(name)).unwrap()).unwrap()
}

const APP: &str = r#"
event Tree(n: i64, what: text)
event Epoch(init: text, commit: text, psk: text, ctx: text)
event Export(secret: text, label: text, context: text, n: u64)
event ExtPub(secret: text)
event Ewl(secret: text, label: text, context: text, n: u64)
event Ds(secret: text, label: text)
event Dts(secret: text, label: text, generation: i64, n: u64)
event Ref(label: text, value: text)
event Swl(seed: text, label: text, content: text)
event Vwl(pk: text, label: text, content: text, sig: text)

state G
    out: List[text] = []

step G on Tree(n: n, what: w)
    let width = tm_width(n)
    let r = tm_root(n)
    var parts = ""
    for x in 0..width
        var v = "-"
        if w == "left" and tm_level(x) > 0
            v = tm_left(x).tostr()
        if w == "right" and tm_level(x) > 0
            v = tm_right(x, n).tostr()
        if w == "parent" and x != r
            v = tm_parent(x, n).tostr()
        if w == "sibling" and x != r
            v = tm_sibling(x, n).tostr()
        parts = if x == 0 then v else parts + "," + v
    out.push(r.tostr() + ":" + parts)

step G on Epoch(init: i, commit: c, psk: p, ctx: x)
    out.push(mls_epoch(i, c, p, x))

step G on Export(secret: s, label: l, context: c, n: n)
    out.push(mls_export(s, l, c, n as usize))

step G on ExtPub(secret: s)
    out.push(mls_external_pub(s))

step G on Ewl(secret: s, label: l, context: c, n: n)
    out.push(tohex(expand_label(unhex(s), l, unhex(c), n as usize)))

step G on Ds(secret: s, label: l)
    out.push(tohex(derive_secret(unhex(s), l)))

step G on Dts(secret: s, label: l, generation: g, n: n)
    out.push(derive_tree_secret(s, l, g, n as usize))

step G on Ref(label: l, value: v)
    out.push(ref_hash(l, v))

step G on Swl(seed: k, label: l, content: c)
    out.push(sign_with_label(k, l, c))

step G on Vwl(pk: p, label: l, content: c, sig: g)
    out.push(verify_with_label(p, l, c, g).tostr())
"#;

struct Run<'a> {
    src: String,
    h: Host<'a>,
}

impl Run<'_> {
    fn step(&mut self, ev: String) -> String {
        let before = self.outs().len();
        self.h.admit(&self.src, &ev).unwrap();
        let o = self.outs();
        assert_eq!(o.len(), before + 1, "step admitted within its fuel: {ev:.80}");
        o[before].clone()
    }
    fn outs(&self) -> Vec<String> {
        let s = format!("{:?}", self.h.state());
        s.split("Text(\"").skip(1).map(|x| x.split('"').next().unwrap().to_string()).collect()
    }
}

#[test]
fn mls_module_checks_pass() {
    let p = std::env::temp_dir().join(format!("rce-mls-{}.rce", std::process::id()));
    std::fs::write(&p, combined(true)).unwrap();
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_rce")).arg("test").arg(&p).output().unwrap();
    let out = String::from_utf8_lossy(&o.stdout);
    let _ = std::fs::remove_file(&p);
    assert!(o.status.success() && out.contains("10 checks"), "{out}");
}

#[test]
fn official_tree_math_and_key_schedule_vectors() {
    let src = combined(false) + APP;
    let m = parse(&src).unwrap();
    let mut st = MemStore::default();
    let h = Host::start(&m, Box::new(&mut st)).unwrap();
    let mut r = Run { src: src.clone(), h };
    // tree-math.json: every tree in the file
    let tm = vectors("tree-math.json");
    let mut trees = 0;
    for t in tm.as_array().unwrap() {
        let n = t["n_leaves"].as_i64().unwrap();
        for what in ["left", "right", "parent", "sibling"] {
            let got = r.step(format!(r#"Tree(n: {n}, what: "{what}")"#));
            let want: Vec<String> = t[what].as_array().unwrap().iter().map(|v| v.as_i64().map(|x| x.to_string()).unwrap_or("-".into())).collect();
            assert_eq!(got, format!("{}:{}", t["root"], want.join(",")), "tree-math n_leaves {n} {what}");
        }
        trees += 1;
    }
    assert_eq!(trees, 10);
    // key-schedule.json, cipher suite 1
    let ks = vectors("key-schedule.json");
    let c = ks.as_array().unwrap().iter().find(|c| c["cipher_suite"] == 1).unwrap();
    let mut init = c["initial_init_secret"].as_str().unwrap().to_string();
    let names = ["joiner_secret", "welcome_secret", "", "sender_data_secret", "encryption_secret", "exporter_secret", "external_secret", "confirmation_key", "membership_key", "resumption_psk", "epoch_authenticator", "init_secret"];
    let mut epochs = 0;
    for e in c["epochs"].as_array().unwrap() {
        let s = |k: &str| e[k].as_str().unwrap().to_string();
        let got = r.step(format!(r#"Epoch(init: "{init}", commit: "{}", psk: "{}", ctx: "{}")"#, s("commit_secret"), s("psk_secret"), s("group_context")));
        let f: Vec<&str> = got.split('.').collect();
        for (i, k) in names.iter().enumerate() {
            if !k.is_empty() {
                assert_eq!(f[i], s(k), "key-schedule epoch {epochs} {k}");
            }
        }
        let x = &e["exporter"];
        let got = r.step(format!(r#"Export(secret: "{}", label: "{}", context: "{}", n: {})"#, s("exporter_secret"), x["label"].as_str().unwrap(), x["context"].as_str().unwrap(), x["length"]));
        assert_eq!(got, x["secret"].as_str().unwrap(), "key-schedule epoch {epochs} exporter");
        assert_eq!(r.step(format!(r#"ExtPub(secret: "{}")"#, s("external_secret"))), s("external_pub"), "key-schedule epoch {epochs} external_pub");
        init = s("init_secret");
        epochs += 1;
    }
    assert_eq!(epochs, 5);
    // crypto-basics.json, suite 1 (EncryptWithLabel is in tests/hpke_rce.rs)
    let cb = &vectors("crypto-basics-suite1.json")[0];
    let g = |a: &str, k: &str| cb[a][k].as_str().map(String::from).unwrap_or_else(|| cb[a][k].to_string());
    assert_eq!(r.step(format!(r#"Ewl(secret: "{}", label: "{}", context: "{}", n: {})"#, g("expand_with_label", "secret"), g("expand_with_label", "label"), g("expand_with_label", "context"), g("expand_with_label", "length"))), g("expand_with_label", "out"), "crypto-basics expand_with_label");
    assert_eq!(r.step(format!(r#"Ds(secret: "{}", label: "{}")"#, g("derive_secret", "secret"), g("derive_secret", "label"))), g("derive_secret", "out"), "crypto-basics derive_secret");
    assert_eq!(r.step(format!(r#"Dts(secret: "{}", label: "{}", generation: {}, n: {})"#, g("derive_tree_secret", "secret"), g("derive_tree_secret", "label"), g("derive_tree_secret", "generation"), g("derive_tree_secret", "length"))), g("derive_tree_secret", "out"), "crypto-basics derive_tree_secret");
    assert_eq!(r.step(format!(r#"Ref(label: "{}", value: "{}")"#, g("ref_hash", "label"), g("ref_hash", "value"))), g("ref_hash", "out"), "crypto-basics ref_hash");
    assert_eq!(r.step(format!(r#"Swl(seed: "{}", label: "{}", content: "{}")"#, g("sign_with_label", "priv"), g("sign_with_label", "label"), g("sign_with_label", "content"))), g("sign_with_label", "signature"), "crypto-basics sign_with_label");
    assert_eq!(r.step(format!(r#"Vwl(pk: "{}", label: "{}", content: "{}", sig: "{}")"#, g("sign_with_label", "pub"), g("sign_with_label", "label"), g("sign_with_label", "content"), g("sign_with_label", "signature"))), "true", "crypto-basics verify_with_label");
    let live = r.h.state_hash().to_string();
    drop(r);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert_eq!(replay(&m, &text).unwrap().1, live);
}
