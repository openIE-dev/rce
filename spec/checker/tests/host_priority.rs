//! HC-9 / T25: the admitter orders simultaneous inputs by (priority,
//! arrival), highest first, and D's order is the order it chose, so replay
//! with adapters disconnected re-derives the same state (HC-8).
use rce_spec_check::ast::{Arg, Expr};
use rce_spec_check::host::{Entry, Host, MemStore, replay};
use rce_spec_check::parse::parse;

fn arg(l: &str, e: Expr) -> Vec<Arg> {
    vec![Arg { label: Some(l.into()), e }]
}

#[test]
fn admitter_orders_simultaneous_inputs_by_priority_then_arrival() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/host/priority.rce")).unwrap();
    let m = parse(&src).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    assert_eq!(h.priority("Alarm"), 200);
    assert_eq!(h.priority("Key"), 0);
    let ins = vec![
        ("Key".to_string(), arg("c", Expr::Str("a".into(), 0))),
        ("Key".to_string(), arg("c", Expr::Str("b".into(), 0))),
        ("Alarm".to_string(), arg("n", Expr::Int(1, 0))),
    ];
    let out = h.admit_simultaneous(ins).unwrap();
    assert!(out[0].event.starts_with("Alarm"), "{:?}", out);
    assert!(out[1].event.contains("\"a\"") && out[2].event.contains("\"b\""), "{:?}", out);
    let live = h.state_hash().to_string();
    drop(h);
    let kinds: Vec<String> = st.lines.iter().map(|l| Entry::parse_line(l).unwrap().kind()).filter(|k| !k.starts_with("host.")).collect();
    assert_eq!(kinds, ["Alarm", "Key", "Key"]);
    let (_, rh, _) = replay(&m, &(st.lines.join("\n") + "\n")).unwrap();
    assert_eq!(rh, live);
}

/// HC-11: `host.FabricCaps` is logged at start with the E.7 fields, and as a
/// host entry it leaves the state hash unchanged (δ is the identity on it).
#[test]
fn fabric_caps_logged_at_start_and_leave_state_unchanged() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/host/priority.rce")).unwrap();
    let m = parse(&src).unwrap();
    let mut st = MemStore::default();
    drop(Host::start(&m, Box::new(&mut st)).unwrap());
    let es: Vec<Entry> = st.lines.iter().map(|l| Entry::parse_line(l).unwrap()).collect();
    let f = es.iter().find(|e| e.kind() == "host.FabricCaps").expect("FabricCaps at start");
    let v: serde_json::Value = serde_json::from_str(&f.event).unwrap();
    for k in ["counters", "cpu", "gpus", "memory", "npus"] {
        assert!(v["payload"].get(k).is_some(), "{k}");
    }
    assert!(es.iter().all(|e| e.state_hash == es[0].state_hash));
}
