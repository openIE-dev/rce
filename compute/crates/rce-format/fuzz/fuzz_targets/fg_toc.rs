//! Fuzz `Toc::decode` with arbitrary bytes. Catches:
//! - bad count prefix overruns
//! - unknown section kinds inside vendor-reserved range vs hard-error range
//! - off/length combinations that overflow on add

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = rce_format::Toc::decode(data);
});
