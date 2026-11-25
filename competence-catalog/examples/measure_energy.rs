//! measured_j: read the hardware meter and measure the real energy of G0
//! competence execution. The cost becomes `Measured` (D.11 HwShunt/OsCounter),
//! not `Modelled`. A batch is averaged because a single interpreter-floor act
//! is below the meter's resolution. If no meter can measure here, it says so —
//! never a fabricated number.

use competence_catalog::energy::EnergyMeter;
use competence_catalog::runtime::RuntimeBridge;
use serde_json::json;

fn main() {
    let mut meter = match EnergyMeter::open() {
        Ok(m) => m,
        Err(e) => {
            println!("no hardware meter: {e}");
            println!("cost stays Modelled (the ≈1 µJ G0 floor); no Measured figure is claimed.");
            return;
        }
    };
    println!("meter source: {} (provenance: Measured)", meter.source());

    let bridge = RuntimeBridge::new(
        "../rce/spec/checker/target/release/rce",
        "../bridge-test/energy",
    );
    let args = json!([[104, 105]]);

    // Warm one execution, then measure a batch of G0 hits.
    let _ = bridge.call_general("hash", "fnv1a32", "(&[u8])->u32", &args);
    let n = 50;
    match meter.measure_mean(n, || {
        let _ = bridge.call_general("hash", "fnv1a32", "(&[u8])->u32", &args);
    }) {
        Ok(uj) => {
            println!("measured cost per G0 act (fnv1a32, {n}-act batch): {uj} µJ");
            println!("modelled G0 floor: 1 µJ — the measured figure includes process spawn + whole-package draw.");
        }
        Err(e) => println!("measurement failed: {e}"),
    }
}
