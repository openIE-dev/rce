#!/usr/bin/env bash
# Build the kernel library of `rce emit wasm` (spec 10.18, EW-2) and pin it.
#
# The library is the crate compute/crates/rce-wasm-kernels for
# wasm32-unknown-unknown, release, with the source paths remapped so the
# bytes do not name the machine that built them. It is written to
# compute/crates/rce-emit-wasm/kernels/rce-kernels.wasm and its BLAKE3
# digest to rce-kernels.wasm.blake3 beside it; `rce` embeds both.
#
#   scripts/build-wasm-kernels.sh         build and pin
#   scripts/build-wasm-kernels.sh check   fail when the pinned digest is not the file's
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/compute/crates/rce-emit-wasm/kernels"
digest() { (cd "$root/compute" && cargo run -q -p rce-wasm-kernels --example digest -- "$1"); }
if [ "${1:-}" = check ]; then
  got="$(digest "$out/rce-kernels.wasm")"
  want="$(tr -d '\n' < "$out/rce-kernels.wasm.blake3")"
  [ "$got" = "$want" ] || { echo "rce-kernels.wasm hashes to $got, the pin is $want" >&2; exit 1; }
  echo "rce-kernels.wasm $got"
  exit 0
fi
cd "$root/compute"
RUSTFLAGS="--remap-path-prefix=$root=rce --remap-path-prefix=$HOME/.cargo=cargo --remap-path-prefix=$HOME/.rustup=rustup" \
  cargo build -q -p rce-wasm-kernels --target wasm32-unknown-unknown --release --target-dir "$root/compute/target/wasm-kernels"
mkdir -p "$out"
cp "$root/compute/target/wasm-kernels/wasm32-unknown-unknown/release/rce_wasm_kernels.wasm" "$out/rce-kernels.wasm"
digest "$out/rce-kernels.wasm" | tr -d '\n' > "$out/rce-kernels.wasm.blake3"
echo "rce-kernels.wasm $(cat "$out/rce-kernels.wasm.blake3") ($(wc -c < "$out/rce-kernels.wasm") bytes, $(rustc --version))"
