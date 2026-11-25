//! The energy-floor lever, measured: fork-per-call vs in-process execution of
//! the same G0 competence, under the hardware meter. In-process removes the
//! process-spawn overhead that dominates the measured cost of a floor act.

use competence_catalog::energy::EnergyMeter;
use competence_catalog::runtime::RuntimeBridge;
use serde_json::json;

fn main() {
    let mut meter = match EnergyMeter::open() {
        Ok(m) => m,
        Err(e) => { println!("no hardware meter: {e}"); return; }
    };
    let args = json!([[104, 105]]);
    let n = 100;

    let fork = RuntimeBridge::new("../rce/spec/checker/target/release/rce", "../bridge-test/ip-energy");
    let inproc = RuntimeBridge::new_inprocess();

    // Warm both paths.
    let _ = fork.call_general("hash", "fnv1a32", "(&[u8])->u32", &args);
    let _ = inproc.call_general("hash", "fnv1a32", "(&[u8])->u32", &args);

    let fork_uj = meter.measure_mean(n, || { let _ = fork.call_general("hash", "fnv1a32", "(&[u8])->u32", &args); }).ok();
    let ip_uj = meter.measure_mean(n, || { let _ = inproc.call_general("hash", "fnv1a32", "(&[u8])->u32", &args); }).ok();

    println!("meter: {} (Measured), {n}-act batches of fnv1a32", meter.source());
    println!("  fork-per-call : {:?} µJ/act", fork_uj);
    println!("  in-process    : {:?} µJ/act", ip_uj);
    println!("  modelled floor: 1 µJ/act");
    if let (Some(f), Some(i)) = (fork_uj, ip_uj) {
        if i > 0 { println!("  in-process is {:.1}× lower energy than fork", f as f64 / i as f64); }
    }
}
