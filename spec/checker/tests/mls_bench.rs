//! Ignored timing probe for the MLS hot path (one host input per operation).
use rce_spec_check::host::{Host, MemStore};
use rce_spec_check::parse::parse;
use std::path::Path;

const APP: &str = r##"
event U(h: text)
event Sg(seed: text, m: text)
event Vf(z: i64)
event Hm(h: text)
state S
    out: List[text] = []
    pk: text = ""
    sg: text = ""
    m: text = ""
step S on U(h: h)
    out = [tohex(unhex(h))]
step S on Sg(seed: seed, m: mm)
    pk = ed_pubkey(copy(seed))
    sg = ed_sign(seed, unhex(copy(mm)))
    m = mm
    out = ["s"]
step S on Vf(z: z)
    out = [if ed_verify(copy(pk), unhex(copy(m)), copy(sg)) then "t" else "f"]
step S on Hm(h: h)
    out = [tohex(hmac(unhex(copy(h)), unhex(h)))]
"##;

#[test]
#[ignore]
fn mls_hot_ops() {
    let mut s = String::from("module bench\n");
    for f in ["lib/crypto.rce"] {
        for l in std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(f)).unwrap().lines() {
            if l.starts_with("module ") || l.starts_with("check ") { continue; }
            s.push_str(l); s.push('\n');
        }
    }
    s += APP;
    let m = parse(&s).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    let hx = |n: usize| "ab".repeat(n);
    let mut t = |name: &str, ev: String| { let t0 = std::time::Instant::now(); h.admit(&s, &ev).unwrap(); eprintln!("{name}: {:.3}s", t0.elapsed().as_secs_f64()); };
    for n in [256, 1024, 4096] { t(&format!("unhex+tohex {n} B"), format!(r#"U(h: "{}")"#, hx(n))); }
    t("sign 1 KB", format!(r#"Sg(seed: "{}", m: "{}")"#, "11".repeat(32), hx(1024)));
    t("verify 1 KB", "Vf(z: 0)".into());
    for n in [256, 1024] { t(&format!("hmac {n} B"), format!(r#"Hm(h: "{}")"#, hx(n))); }
}
