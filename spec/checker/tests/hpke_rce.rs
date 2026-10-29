//! AES-128-GCM and HPKE base mode in pure RCE (crates/rce/lib/hpke.rce on
//! top of lib/crypto.rce). The FIPS 197 and SP 800-38D checks of the module;
//! the official RFC 9180 vector for DHKEM(X25519, HKDF-SHA256), HKDF-SHA256,
//! AES-128-GCM in base mode (key pair derivation, seal at sequence numbers
//! 0, 1, 2, 255 and 256, open, a tampered ciphertext that does not open); and the RFC
//! 9420 crypto-basics.json EncryptWithLabel vector for cipher suite 1
//! (decrypt the official ciphertext, then a fresh seal and open). One
//! operation per host input inside the fuel of HC-7a; the log replays.
use rce_spec_check::host::{Host, MemStore, replay};
use rce_spec_check::parse::parse;
use serde_json::Value;
use std::path::Path;

fn combined(checks: bool) -> String {
    let mut s = String::from("# rce-version: 0.5\nmodule hpke_steps\n");
    for f in ["lib/crypto.rce", "lib/hpke.rce"] {
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
    serde_json::from_str(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors").join(name)).unwrap()).unwrap()
}

const APP: &str = r#"
event Derive(ikm: text)
event Seal(pkr: text, ske: text, info: text, aad: text, pt: text, seq: i64)
event Open(skr: text, enc: text, info: text, aad: text, ctext: text, seq: i64)
event Dwl(sk: text, lbl: text, context: text, kem: text, ctext: text)
event Ewl(pk: text, ske: text, lbl: text, context: text, pt: text)

state S
    out: List[text] = []

step S on Derive(ikm: i)
    out.push(hp_derive_sk(i))

step S on Seal(pkr: p, ske: e, info: i, aad: a, pt: t, seq: n)
    out.push(hpke_seal(p, e, i, a, t, n))

step S on Open(skr: k, enc: e, info: i, aad: a, ctext: c, seq: n)
    out.push(hpke_open(k, e, i, a, c, n))

step S on Dwl(sk: k, lbl: l, context: c, kem: e, ctext: t)
    out.push(decrypt_with_label(k, l, c, e, t))

step S on Ewl(pk: p, ske: e, lbl: l, context: c, pt: t)
    out.push(encrypt_with_label(p, e, l, c, t))
"#;

#[test]
fn hpke_module_checks_pass() {
    let p = std::env::temp_dir().join(format!("rce-hpke-{}.rce", std::process::id()));
    std::fs::write(&p, combined(true)).unwrap();
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_rce")).arg("test").arg(&p).output().unwrap();
    let out = String::from_utf8_lossy(&o.stdout);
    let _ = std::fs::remove_file(&p);
    assert!(o.status.success() && out.contains("12 checks"), "{out}");
}

#[test]
fn official_rfc9180_and_mls_encrypt_with_label_vectors() {
    let src = combined(false) + APP;
    let m = parse(&src).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    let mut step = |ev: String| -> String {
        let n0 = format!("{:?}", h.state()).matches("Text(\"").count();
        h.admit(&src, &ev).unwrap();
        let s = format!("{:?}", h.state());
        let o: Vec<String> = s.split("Text(\"").skip(1).map(|x| x.split('"').next().unwrap().to_string()).collect();
        assert_eq!(o.len(), n0 + 1, "step admitted within its fuel: {ev:.80}");
        o[n0].clone()
    };
    // RFC 9180 A.1-style vector (base mode, X25519, SHA-256, AES-128-GCM)
    let v = &vectors("hpke-base-x25519-sha256-aes128gcm.json")[0];
    let s = |k: &str| v[k].as_str().unwrap().to_string();
    assert_eq!(step(format!(r#"Derive(ikm: "{}")"#, s("ikmE"))), s("skEm"), "DeriveKeyPair(ikmE)");
    assert_eq!(step(format!(r#"Derive(ikm: "{}")"#, s("ikmR"))), s("skRm"), "DeriveKeyPair(ikmR)");
    let encs = v["encryptions"].as_array().unwrap();
    for i in [0usize, 1, 2, 255, 256] {
        let e = &encs[i];
        let g = |k: &str| e[k].as_str().unwrap().to_string();
        let got = step(format!(r#"Seal(pkr: "{}", ske: "{}", info: "{}", aad: "{}", pt: "{}", seq: {i})"#, s("pkRm"), s("skEm"), s("info"), g("aad"), g("pt")));
        assert_eq!(got, format!("{}.{}", s("enc"), g("ct")), "RFC 9180 seal, sequence {i}");
        let pt = step(format!(r#"Open(skr: "{}", enc: "{}", info: "{}", aad: "{}", ctext: "{}", seq: {i})"#, s("skRm"), s("enc"), s("info"), g("aad"), g("ct")));
        assert_eq!(pt, g("pt"), "RFC 9180 open, sequence {i}");
    }
    let ct0 = encs[0]["ct"].as_str().unwrap();
    let bad = format!("{}{:02x}", &ct0[..ct0.len() - 2], u8::from_str_radix(&ct0[ct0.len() - 2..], 16).unwrap() ^ 1);
    let pt = step(format!(r#"Open(skr: "{}", enc: "{}", info: "{}", aad: "{}", ctext: "{bad}", seq: 0)"#, s("skRm"), s("enc"), s("info"), encs[0]["aad"].as_str().unwrap()));
    assert_eq!(pt, "", "a tampered tag does not open");
    // RFC 9420 crypto-basics.json, suite 1, EncryptWithLabel
    let c = &vectors("mls/crypto-basics-suite1.json")[0]["encrypt_with_label"];
    let g = |k: &str| c[k].as_str().unwrap().to_string();
    let pt = step(format!(r#"Dwl(sk: "{}", lbl: "{}", context: "{}", kem: "{}", ctext: "{}")"#, g("priv"), g("label"), g("context"), g("kem_output"), g("ciphertext")));
    assert_eq!(pt, g("plaintext"), "crypto-basics DecryptWithLabel");
    let sealed = step(format!(r#"Ewl(pk: "{}", ske: "{}", lbl: "{}", context: "{}", pt: "{}")"#, g("pub"), "77".repeat(32), g("label"), g("context"), g("plaintext")));
    let (kem, ct) = sealed.split_once('.').unwrap();
    let pt = step(format!(r#"Dwl(sk: "{}", lbl: "{}", context: "{}", kem: "{kem}", ctext: "{ct}")"#, g("priv"), g("label"), g("context")));
    assert_eq!(pt, g("plaintext"), "EncryptWithLabel then DecryptWithLabel");
    let live = h.state_hash().to_string();
    drop(h);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert_eq!(replay(&m, &text).unwrap().1, live);
}

/// All 257 encryptions of the official RFC 9180 vector, seal and open.
/// About 15 s of interpreter time per encryption (GHASH runs bit by bit in
/// RCE), so it is ignored in the default suite; run it with
/// `cargo test --release --test hpke_rce -- --ignored`.
#[test]
#[ignore]
fn all_257_rfc9180_encryptions() {
    let src = combined(false) + APP;
    let m = parse(&src).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    let v = &vectors("hpke-base-x25519-sha256-aes128gcm.json")[0];
    let s = |k: &str| v[k].as_str().unwrap().to_string();
    let encs = v["encryptions"].as_array().unwrap();
    assert_eq!(encs.len(), 257);
    let mut n = 0;
    for (i, e) in encs.iter().enumerate() {
        let g = |k: &str| e[k].as_str().unwrap().to_string();
        h.admit(&src, &format!(r#"Seal(pkr: "{}", ske: "{}", info: "{}", aad: "{}", pt: "{}", seq: {i})"#, s("pkRm"), s("skEm"), s("info"), g("aad"), g("pt"))).unwrap();
        h.admit(&src, &format!(r#"Open(skr: "{}", enc: "{}", info: "{}", aad: "{}", ctext: "{}", seq: {i})"#, s("skRm"), s("enc"), s("info"), g("aad"), g("ct"))).unwrap();
        let st = format!("{:?}", h.state());
        let o: Vec<&str> = st.split("Text(\"").skip(1).map(|x| x.split('"').next().unwrap()).collect();
        assert_eq!(o[2 * i], format!("{}.{}", s("enc"), g("ct")), "seal, sequence {i}");
        assert_eq!(o[2 * i + 1], g("pt"), "open, sequence {i}");
        n += 1;
        eprintln!("rfc9180 encryption {i} ok");
    }
    assert_eq!(n, 257);
    let live = h.state_hash().to_string();
    drop(h);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert_eq!(replay(&m, &text).unwrap().1, live);
}
