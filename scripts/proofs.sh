#!/usr/bin/env bash
# Formal proofs gate: Kani harnesses of the kernel (certificate written
# only if all pass), then the Z3 rewrite proofs and handler proofs.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f proofs/kani.cert.new
# Kani's bundled nightly is older than the rust-version rce pins, so the
# harnesses run from proofs/kernel-kani, which compiles src/kernel.rs
# unchanged (via #[path]) with no dependencies and no rust-version pin.
( cd proofs/kernel-kani && CARGO_TARGET_DIR="${RCE_KANI_TARGET:-${TMPDIR:-/tmp}/rce-kani-target}" \
    cargo kani --harness kernel:: ) > proofs/kani.log 2>&1 || true
grep -aE 'Checking harness|VERIFICATION:|Complete' proofs/kani.log
grep -aq 'Complete - 10 successfully verified harnesses, 0 failures, 10 total' proofs/kani.log
sha256sum src/kernel.rs | cut -d' ' -f1 > proofs/kani.cert.new
grep -a 'Checking harness\|VERIFICATION' proofs/kani.log >> proofs/kani.cert.new
mv proofs/kani.cert.new proofs/kani.cert
cargo test --release -q --test smt_rewrites --test kernel_proofs
cargo test --release -q --lib handler_proof
