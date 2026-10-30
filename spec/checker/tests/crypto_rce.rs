//! Chat E2EE primitives written in RCE (stdlib/lib/crypto.rce): the
//! RFC/FIPS vectors are `check`s of the module (rce test); here X3DH and the
//! Double Ratchet run as host steps, one cryptographic operation per input so
//! each step stays inside the fuel of 6.7, with randomness supplied as
//! logged inputs; out-of-order delivery, a forged message and a group
//! (sender-key) message, and the whole log replays.
use rce_spec_check::host::{Host, MemStore, replay};
use rce_spec_check::parse::parse;
use std::path::Path;
#[path = "support/flat.rs"]
mod flat;

fn main_src() -> String {
    format!("module crypto\n{}", flat::flat(&["crypto"], true))
}

#[test]
fn crypto_module_vectors_pass() {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../stdlib/lib/crypto.rce");
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_rce")).arg("test").arg(&p).output().unwrap();
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(o.status.success() && out.contains("8 checks"), "{out}");
}

#[test]
fn x3dh_double_ratchet_and_sender_keys_as_host_steps() {
    let app = r#"
event Keys
event Keys2
event RespB
event X3dh
event InitA(fresh: text)
event InitB
event ASend(msg: text)
event BRecv(i: u64, fresh: text)
event BSend(msg: text)
event ARecv(i: u64, fresh: text)
event Forge(i: u64, fresh: text)
event Group

state S
    pk: Map[text, text] = {}
    ska: text = ""
    skb: text = ""
    a: List[Ratchet] = []
    b: List[Ratchet] = []
    to_b: List[Sealed] = []
    to_a: List[Sealed] = []
    got_b: List[text] = []
    got_a: List[text] = []
    rejected: u64 = 0
    group_ok: bool = false

const IKA: text = "1111111111111111111111111111111111111111111111111111111111111111"
const EKA: text = "2222222222222222222222222222222222222222222222222222222222222222"
const IKB: text = "3333333333333333333333333333333333333333333333333333333333333333"
const SPKB: text = "4444444444444444444444444444444444444444444444444444444444444444"

fn hexs(b: List[i64]) -> text = tohex(b)

step S on Keys
    pk["ika"] = pubkey(IKA)
    pk["eka"] = pubkey(EKA)
    pk["ikb"] = pubkey(IKB)

step S on Keys2
    pk["spkb"] = pubkey(SPKB)

step S on X3dh
    ska = x3dh_init(IKA, EKA, pk.get("ikb") catch "", pk.get("spkb") catch "")

step S on RespB
    skb = x3dh_resp(IKB, SPKB, pk.get("ika") catch "", pk.get("eka") catch "")

step S on InitB
    b.push(ratchet_init_bob(copy(skb), SPKB))

step S on InitA(fresh: f)
    a.push(ratchet_init_alice(copy(ska), pk.get("spkb") catch "", f))

step S on ASend(msg: m)
    let s = ratchet_encrypt(copy(a[0]), tobytes(m))
    a[0] = copy(s.st)
    to_b.push(s)

step S on BSend(msg: m)
    let s = ratchet_encrypt(copy(b[0]), tobytes(m))
    b[0] = copy(s.st)
    to_a.push(s)

step S on BRecv(i: i, fresh: f)
    let s = copy(to_b[i as usize])
    let o = ratchet_decrypt(copy(b[0]), copy(s.header), copy(s.body), f)
    b[0] = copy(o.st)
    if o.ok
        got_b.push(hexs(copy(o.plain)))
    else
        rejected += 1

step S on ARecv(i: i, fresh: f)
    let s = copy(to_a[i as usize])
    let o = ratchet_decrypt(copy(a[0]), copy(s.header), copy(s.body), f)
    a[0] = copy(o.st)
    if o.ok
        got_a.push(hexs(copy(o.plain)))
    else
        rejected += 1

step S on Forge(i: i, fresh: f)
    let s = copy(to_a[i as usize])
    let o = ratchet_decrypt(copy(a[0]), copy(s.header), "00" + copy(s.body), f)
    if not o.ok
        rejected += 1

step S on Group
    group_ok = group_roundtrip()
"#;
    let src = main_src().replace("module crypto", "module e2ee_steps") + app;
    let m = parse(&src).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    let r = |n: u8| format!("{:02x}", n).repeat(32);
    let evs = format!(
        r#"Keys, Keys2, X3dh, RespB, InitB, InitA(fresh: "{}"), ASend(msg: "hi bob"), ASend(msg: "second"), BRecv(i: 1, fresh: "{}"), BRecv(i: 0, fresh: "{}"), BSend(msg: "hi alice"), ARecv(i: 0, fresh: "{}"), Forge(i: 0, fresh: "{}"), Group"#,
        r(0x55), r(0x66), r(0x77), r(0x88), r(0x99)
    );
    for x in h.admit(&src, &evs).unwrap() {
        assert!(x.outcome.is_ok(), "{} kept: {:?}", x.event, x.outcome);
    }
    let s = format!("{:?}", h.state());
    let hx = |t: &str| t.bytes().map(|b| format!("{b:02x}")).collect::<String>();
    let field = |n: &str| s.split(&format!("(\"{n}\", ")).nth(1).unwrap().split("), (\"").next().unwrap().to_string();
    assert!(field("ska") == field("skb"), "X3DH secrets agree");
    let gb = field("got_b");
    assert!(gb.find(&hx("second")).unwrap() < gb.find(&hx("hi bob")).unwrap(), "out of order: {gb}");
    assert!(field("got_a").contains(&hx("hi alice")));
    assert!(field("rejected").contains("Int(1"), "forgery rejected: {}", field("rejected"));
    assert!(field("group_ok").contains("true"));
    let live = h.state_hash().to_string();
    drop(h);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert_eq!(replay(&m, &text).unwrap().1, live);
}
