//! P3 / HC-4: foreign code as an isolated WebAssembly instance per call,
//! named by digest, under fuel; result or trap logged; replay with the
//! adapter disconnected reaches the same state (T32, T37).
use rce_spec_check::adapters::wasm_adapter;
use rce_spec_check::host::{Adapter, Host, MemStore, replay};
use rce_spec_check::interp::d_sha;
use rce_spec_check::parse::parse;
use std::path::Path;

const WAT: &str = r#"(module
  (func (export "add") (param i64 i64) (result i64) local.get 0 local.get 1 i64.add)
  (func (export "spin") (param i32) (result i32) (loop br 0) i32.const 0)
  (func (export "div") (param i32 i32) (result i32) local.get 0 local.get 1 i32.div_s))"#;

#[test]
fn wasm_calls_are_logged_fuel_bounded_and_replayed() {
    let src = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/foreign.rce")).unwrap();
    let m = parse(&src).unwrap();
    let root = std::env::temp_dir().join(format!("rce-wasm-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let dg = d_sha(WAT);
    std::fs::write(root.join(&dg), WAT).unwrap();
    let mut st = MemStore::default();
    let ads: Vec<Box<dyn Adapter>> = vec![Box::new(wasm_adapter(&root))];
    let mut h = Host::start_with(&m, Box::new(&mut st), ads).unwrap();
    let evs = format!(
        r#"Call(code: "{dg}", func: "add", args: [40, 2]), Call(code: "{dg}", func: "spin", args: [0]), Call(code: "{dg}", func: "div", args: [1, 0]), Call(code: "{dg}", func: "nope", args: []), Call(code: "{}", func: "add", args: [1, 1])"#,
        "0".repeat(64)
    );
    h.admit(&src, &evs).unwrap();
    let s = format!("{:?}", h.state());
    assert!(s.contains("Int(42"), "{s}");
    assert!(s.contains("traps\", Int(4"), "{s}");
    let live = h.state_hash().to_string();
    drop(h);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert!(text.contains("fuel") || text.contains("trap"), "the trap reason is logged");
    // replay: no adapter, no module file
    let _ = std::fs::remove_dir_all(&root);
    assert_eq!(replay(&m, &text).unwrap().1, live);
}
