//! P3 floats under HC-13: the probe digest, NaN canonicalization, float
//! parameters and results as bit patterns, logged and replayed.
use rce_spec_check::adapters::wasm_adapter;
use rce_spec_check::host::{Adapter, Host, MemStore, replay};
use rce_spec_check::interp::d_sha;
use rce_spec_check::parse::parse;
use rce_spec_check::wasm_float::{self, CANON_F64, REFERENCE};
use std::path::Path;

#[test]
fn probe_matches_the_reference_on_cpu_and_engine() {
    let cpu = wasm_float::reference_digest();
    eprintln!("PROBE cpu {cpu}");
    let mut c = wasmi::Config::default();
    c.consume_fuel(true);
    let eng = wasm_float::engine_digest(&wasmi::Engine::new(&c)).unwrap();
    eprintln!("PROBE engine {eng}");
    assert_eq!(cpu, REFERENCE, "this CPU is not IEEE on the probe table");
    assert_eq!(eng, REFERENCE, "the engine with canonical NaNs differs from the CPU reference");
}

const WAT: &str = r#"(module
  (func (export "nan") (result i64) f64.const 0 f64.const 0 f64.div i64.reinterpret_f64)
  (func (export "nan32") (result i32) f32.const 0 f32.const 0 f32.div i32.reinterpret_f32)
  (func (export "hyp") (param f64 f64) (result f64) local.get 0 local.get 0 f64.mul local.get 1 local.get 1 f64.mul f64.add f64.sqrt)
  (func (export "raw") (param f64) (result f64) local.get 0 f64.const 1 f64.mul)
  (func (export "half") (param f32) (result f32) local.get 0 f32.const 0.5 f32.mul))"#;

#[test]
fn nan_bits_are_canonical_and_floats_cross_as_bits() {
    let src = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/foreign.rce")).unwrap();
    let m = parse(&src).unwrap();
    let root = std::env::temp_dir().join(format!("rce-wasmf-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let dg = d_sha(WAT);
    std::fs::write(root.join(&dg), WAT).unwrap();
    let wa = wasm_adapter(&root);
    assert!(wa.exact.contains("wasm.f64"), "floats admitted: {}", wa.exact);
    let mut st = MemStore::default();
    let ads: Vec<Box<dyn Adapter>> = vec![Box::new(wa)];
    let mut h = Host::start_with(&m, Box::new(&mut st), ads).unwrap();
    let neg_nan = 0xfff8_0000_0000_0001u64 as i64;
    let evs = format!(
        r#"Call(code: "{dg}", func: "nan", args: []), Call(code: "{dg}", func: "nan32", args: []), Call(code: "{dg}", func: "hyp", args: [{}, {}]), Call(code: "{dg}", func: "raw", args: [{neg_nan}]), Call(code: "{dg}", func: "half", args: [{}])"#,
        3f64.to_bits() as i64,
        4f64.to_bits() as i64,
        1f32.to_bits() as i64
    );
    h.admit(&src, &evs).unwrap();
    let s = format!("{:?}", h.state());
    for want in [CANON_F64 as i64, 0x7fc0_0000, 5f64.to_bits() as i64, CANON_F64 as i64, 0.5f32.to_bits() as i64] {
        assert!(s.contains(&format!("Int({want}")), "{want:x} in {s}");
    }
    let live = h.state_hash().to_string();
    drop(h);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    let _ = std::fs::remove_dir_all(&root);
    assert_eq!(replay(&m, &text).unwrap().1, live);
}

#[test]
fn canonicalize_keeps_integer_modules_and_rejects_garbage() {
    let w = r#"(module (func (export "f") (param i32 i64) (result i64) (local f32) local.get 1))"#;
    let b = wasm_float::canonicalize(w.as_bytes()).unwrap();
    assert_eq!(&b[..4], b"\0asm");
    assert!(wasm_float::canonicalize(b"not wasm").is_err());
}
