#!/bin/sh
# Translation validation of the trusted base: 300 random core step bodies x 10
# inputs, emitted to Rust by the verified emitter, compiled with rustc, and
# compared against the Lean reference `exec`.
set -e
cd "$(dirname "$0")/.."
lake build && lake env lean emit-test/Gen.lean
rustc --edition 2021 -C opt-level=0 emit-test/main.rs -o /tmp/emit_t
/tmp/emit_t
