//! Ed25519 (RFC 8032) written in pure RCE (stdlib/lib/crypto.rce): the
//! RFC 8032 7.1 test vectors 1 to 3 as host steps, one operation per input
//! so each stays inside the per-input fuel (HC-7a); a changed message, a
//! changed signature, S + L (malleability) and a non-canonical public key
//! are rejected; the log replays to the same state hash.
use rce_spec_check::host::{Host, MemStore, replay};
use rce_spec_check::parse::parse;
#[path = "support/flat.rs"]
mod flat;

const V: [(&str, &str, &str, &str); 3] = [
    ("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60", "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", "", "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"),
    ("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb", "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c", "72", "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"),
    ("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7", "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025", "af82", "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"),
];

const APP: &str = r#"
event Pk(seed: text)
event Sign(seed: text, msg: text)
event Verify(pk: text, msg: text, sig: text)

state S
    out: List[text] = []

step S on Pk(seed: s)
    out.push(ed_pubkey(s))

step S on Sign(seed: s, msg: m)
    out.push(ed_sign(s, unhex(m)))

step S on Verify(pk: p, msg: m, sig: g)
    out.push(ed_verify(p, unhex(m), g).tostr())
"#;

#[test]
fn rfc8032_vectors_and_rejections_as_host_steps() {
    let src = format!("module ed_steps\n{}", flat::flat(&["crypto"], true)) + APP;
    let m = parse(&src).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    let mut evs = vec![];
    let mut want: Vec<String> = vec![];
    for (sk, pk, msg, sig) in V {
        evs.push(format!(r#"Pk(seed: "{sk}")"#));
        want.push(pk.into());
        evs.push(format!(r#"Sign(seed: "{sk}", msg: "{msg}")"#));
        want.push(sig.into());
        evs.push(format!(r#"Verify(pk: "{pk}", msg: "{msg}", sig: "{sig}")"#));
        want.push("true".into());
    }
    let (_, pk2, _, sig2) = V[1];
    let (_, pk1, _, sig1) = V[0];
    // message 0x73 instead of 0x72
    evs.push(format!(r#"Verify(pk: "{pk2}", msg: "73", sig: "{sig2}")"#));
    // first byte of R changed
    evs.push(format!(r#"Verify(pk: "{pk2}", msg: "72", sig: "93{}")"#, &sig2[2..]));
    // S + L: same point, rejected by the S < L rule
    evs.push(format!(r#"Verify(pk: "{pk1}", msg: "", sig: "{}4c8c7872aa064e049dbb3013fbf29380d25bf5f0595bbe24655141438e7a101b")"#, &sig1[..64]));
    // public key with y >= p (non-canonical) does not decode
    evs.push(format!(r#"Verify(pk: "{}", msg: "", sig: "{sig1}")"#, "edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f"));
    want.extend(["false", "false", "false", "false"].map(String::from));
    for e in &evs {
        for x in h.admit(&src, e).unwrap() {
            assert!(x.outcome.is_ok(), "{} kept: {:?}", x.event, x.outcome);
        }
    }
    let s = format!("{:?}", h.state());
    for (i, w) in want.iter().enumerate() {
        assert!(s.contains(&format!("{w:?}")) || (w == "true" || w == "false"), "{i}: {w} missing");
    }
    let outs: Vec<String> = s.split("Text(\"").skip(1).map(|t| t.split('"').next().unwrap().to_string()).collect();
    assert_eq!(outs, want);
    let live = h.state_hash().to_string();
    drop(h);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert_eq!(replay(&m, &text).unwrap().1, live);
}
