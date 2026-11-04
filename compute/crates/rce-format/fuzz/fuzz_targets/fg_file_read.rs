//! Top-level fuzz target: feed arbitrary bytes to `FgFile::read`.
//!
//! Reading a `.fg` file from untrusted bytes is the most exposed surface
//! in the format crate. Any input — whether malformed, truncated,
//! adversarial, or random — must produce a typed `FgError` rather than
//! a panic, OOM, or stack overflow.
//!
//! Acceptance criterion (M0 plan): 24h libFuzzer at this target with
//! zero crashes.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    // Bound input size — libFuzzer drives infinite-length inputs at low
    // probability; cap so we don't spend cycles on TBs of zeros.
    if data.len() > 16 * 1024 * 1024 {
        return;
    }
    let cursor = Cursor::new(data);
    let _ = rce_format::FgFile::read(cursor);
});
