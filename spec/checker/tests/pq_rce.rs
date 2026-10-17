//! Post-quantum key agreement in pure RCE (crates/rce/lib/pq.rce on top of
//! lib/crypto.rce): the FIPS 202 checks of the module; ML-KEM-768 key
//! generation, encapsulation, decapsulation and implicit rejection as host
//! steps, against the RustCrypto ml-kem 0.2.3 crate for d = 01*32,
//! z = 02*32, m = 03*32 (digests of ek, dk and ct, the shared secrets in
//! full); then PQXDH between two parties (DHs, encapsulation and the KDF as separate steps) with an Ed25519-signed ML-KEM
//! prekey, a forged KEM ciphertext giving a different key, and the whole log
//! replaying to the same state hash. One operation per input (HC-7a).
use rce_spec_check::host::{Host, MemStore, replay};
use rce_spec_check::parse::parse;
use sha2::{Digest, Sha256};
use std::path::Path;

const EK_SHA: &str = "34de5514432a9f2a59b397b800a06ea4eb377aae027389ecb0010ee004e212bb";
const DK_SHA: &str = "e8e7516fd7cb2bbd98e3ff48ca645dc92b40e3a0f64f2c742b9183e7af01f367";
const CT_SHA: &str = "564feedaf9dc8b3d426dec15ab4168fa238b94f29c5c989e2f865714948829a5";
const SS: &str = "f39b95557ee52af1954cd59f19febcb39f127e4abefc6f90546e7b8139ce94ba";
const SS_BAD: &str = "87064adde027ac33989b425f2dbee160629c1cf40fa28c5423707924b843c6fe";

const APP: &str = r#"
event Kg(d: text, z: text)
event En(ek: text, m: text)
event De(dk: text, c: text)
event Pub(k: text)
event Sign(seed: text, msg: text)
event Verify(pk: text, msg: text, sig: text)
event DhI(ika: text, eka: text, ikb: text, spkb: text)
event DhR(ikb: text, spkb: text, ika: text, eka: text)
event Sk(dh: text, ss: text)

state S
    out: List[text] = []

step S on Kg(d: d, z: z)
    out.push(mlkem_keygen(d, z))

step S on En(ek: e, m: m)
    out.push(mlkem_encaps(e, m))

step S on De(dk: k, c: c)
    out.push(mlkem_decaps(k, c))

step S on Pub(k: k)
    out.push(pubkey(k))

step S on Sign(seed: s, msg: m)
    out.push(ed_sign(s, unhex(m)))

step S on Verify(pk: p, msg: m, sig: g)
    out.push(ed_verify(p, unhex(m), g).tostr())

step S on DhI(ika: a, eka: e, ikb: b, spkb: s)
    out.push(pqxdh_dh_init(a, e, b, s))

step S on DhR(ikb: b, spkb: s, ika: a, eka: e)
    out.push(pqxdh_dh_resp(b, s, a, e))

step S on Sk(dh: d, ss: s)
    out.push(pqxdh_sk(d, s))
"#;

fn root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// lib/crypto.rce and lib/pq.rce as one module (checks kept or dropped)
fn combined(checks: bool) -> String {
    let mut s = String::from("# rce-version: 0.5\nmodule pq_steps\n");
    for f in ["lib/crypto.rce", "lib/pq.rce"] {
        for l in std::fs::read_to_string(root().join(f)).unwrap().lines() {
            if l.starts_with("module ") || (!checks && l.starts_with("check ")) {
                continue;
            }
            s.push_str(l);
            s.push('\n');
        }
    }
    s
}

fn sha(s: &str) -> String {
    Sha256::digest(s.as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn pq_module_checks_pass() {
    let p = std::env::temp_dir().join(format!("rce-pq-{}.rce", std::process::id()));
    let src = combined(true).replace("module pq_steps", "module pq_checks");
    std::fs::write(&p, src).unwrap();
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_rce")).arg("test").arg(&p).output().unwrap();
    let out = String::from_utf8_lossy(&o.stdout);
    let _ = std::fs::remove_file(&p);
    assert!(o.status.success() && out.contains("14 checks"), "{out}");
}

#[test]
fn mlkem768_and_pqxdh_as_host_steps() {
    let src = combined(false) + APP;
    let m = parse(&src).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    let out = |h: &Host| -> Vec<String> {
        let s = format!("{:?}", h.state());
        s.split("Text(\"").skip(1).map(|x| x.split('"').next().unwrap().to_string()).collect()
    };
    let (d, z, mm) = ("01".repeat(32), "02".repeat(32), "03".repeat(32));
    // ML-KEM-768 against the reference
    h.admit(&src, &format!(r#"Kg(d: "{d}", z: "{z}")"#)).unwrap();
    let kp = out(&h)[0].clone();
    let (ek, dk) = kp.split_once('.').unwrap();
    assert_eq!((ek.len(), dk.len()), (2368, 4800));
    assert_eq!(sha(ek), EK_SHA);
    assert_eq!(sha(dk), DK_SHA);
    h.admit(&src, &format!(r#"En(ek: "{ek}", m: "{mm}")"#)).unwrap();
    let cs = out(&h)[1].clone();
    let (ct, ss) = cs.split_once('.').unwrap();
    assert_eq!(sha(ct), CT_SHA);
    assert_eq!(ss, SS);
    h.admit(&src, &format!(r#"De(dk: "{dk}", c: "{ct}")"#)).unwrap();
    assert_eq!(out(&h)[2], SS);
    let bad = format!("{:02x}{}", u8::from_str_radix(&ct[..2], 16).unwrap() ^ 1, &ct[2..]);
    h.admit(&src, &format!(r#"De(dk: "{dk}", c: "{bad}")"#)).unwrap();
    assert_eq!(out(&h)[3], SS_BAD, "implicit rejection: J(z || c)");
    // PQXDH: bob signs his ML-KEM prekey with his Ed25519 identity
    let (ika, eka, ikb, spkb, edb) = ("11".repeat(32), "22".repeat(32), "33".repeat(32), "44".repeat(32), "55".repeat(32));
    h.admit(&src, &format!(r#"Pub(k: "{ika}")"#)).unwrap();
    h.admit(&src, &format!(r#"Pub(k: "{eka}")"#)).unwrap();
    h.admit(&src, &format!(r#"Pub(k: "{ikb}")"#)).unwrap();
    h.admit(&src, &format!(r#"Pub(k: "{spkb}")"#)).unwrap();
    h.admit(&src, &format!(r#"Sign(seed: "{edb}", msg: "{ek}")"#)).unwrap();
    let o = out(&h);
    let (ika_p, eka_p, ikb_p, spkb_p, sig) = (o[4].clone(), o[5].clone(), o[6].clone(), o[7].clone(), o[8].clone());
    // ann checks bob's signature over the prekey with his Ed25519 public key
    h.admit(&src, &format!(r#"Verify(pk: "{}", msg: "{ek}", sig: "{sig}")"#, ed_pub(&edb))).unwrap();
    assert_eq!(out(&h)[9], "true", "the prekey signature verifies");
    // ann: the three DHs and an encapsulation to bob's signed ML-KEM prekey
    let n = |h: &Host| out(h).len();
    let step = |h: &mut Host, ev: String| -> String {
        let before = n(h);
        h.admit(&src, &ev).unwrap();
        let o = out(h);
        assert_eq!(o.len(), before + 1, "step admitted within its fuel: {ev:.60}");
        o[before].clone()
    };
    let dha = step(&mut h, format!(r#"DhI(ika: "{ika}", eka: "{eka}", ikb: "{ikb_p}", spkb: "{spkb_p}")"#));
    let enc = step(&mut h, format!(r#"En(ek: "{ek}", m: "{mm}")"#));
    let (pct, ssa) = enc.split_once('.').unwrap();
    assert_eq!(pct, ct, "the KEM ciphertext is ML-KEM's");
    let ska = step(&mut h, format!(r#"Sk(dh: "{dha}", ss: "{ssa}")"#));
    // bob: the same DHs from his side and a decapsulation
    let dhb = step(&mut h, format!(r#"DhR(ikb: "{ikb}", spkb: "{spkb}", ika: "{ika_p}", eka: "{eka_p}")"#));
    assert_eq!(dha, dhb);
    let ssb = step(&mut h, format!(r#"De(dk: "{dk}", c: "{pct}")"#));
    let skb = step(&mut h, format!(r#"Sk(dh: "{dhb}", ss: "{ssb}")"#));
    assert_eq!(ska, skb, "both sides derive the same session key");
    assert_eq!(ska.len(), 64);
    // a forged KEM ciphertext: implicit rejection gives another key
    let ssx = step(&mut h, format!(r#"De(dk: "{dk}", c: "{bad}")"#));
    let skx = step(&mut h, format!(r#"Sk(dh: "{dhb}", ss: "{ssx}")"#));
    assert_ne!(skx, ska, "a forged KEM ciphertext gives another key");
    let live = h.state_hash().to_string();
    drop(h);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert_eq!(replay(&m, &text).unwrap().1, live);
}

fn ed_pub(seed: &str) -> String {
    // the Ed25519 public key, computed by the RCE library itself
    let src = combined(false) + "\nevent P(s: text)\n\nstate T\n    k: text = \"\"\n\nstep T on P(s: s)\n    k = ed_pubkey(s)\n";
    let m = parse(&src).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    h.admit(&src, &format!(r#"P(s: "{seed}")"#)).unwrap();
    let s = format!("{:?}", h.state());
    s.split("Text(\"").nth(1).unwrap().split('"').next().unwrap().to_string()
}
