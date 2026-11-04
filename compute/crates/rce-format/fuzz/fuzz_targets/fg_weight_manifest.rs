//! Fuzz the `WeightManifest` CBOR decode + `validate_layout` invariants.
//! Bytes flow into ciborium → WeightManifest → layout check; any input
//! must produce a typed error rather than a panic or OOM.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 4 * 1024 * 1024 {
        return;
    }
    if let Ok(m) = rce_format::section::decode_cbor::<rce_format::weight::WeightManifest>(data)
    {
        // Bound the payload assertion: validate_layout is `O(entries × log entries)`
        // due to the sort, so cap the input length we hand it.
        let _ = m.validate_layout(u64::MAX / 2);
    }
});
