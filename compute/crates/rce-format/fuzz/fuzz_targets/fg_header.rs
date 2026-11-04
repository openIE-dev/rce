//! Fuzz `FgHeader::decode` directly with 64-byte slices. Decoder must
//! handle every byte pattern at the header offset without panicking.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = rce_format::FgHeader::decode(data);
});
