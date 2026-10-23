#!/bin/sh
# Slow vector sets kept out of push CI (scripts/check.sh): the BLAKE3
# 102400-byte cases, one mode per file (each near one comptime fuel budget).
set -e
RCE=${RCE:-rce}
cd "$(dirname "$0")/.."
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
for c in tests/hash/blake3_nightly_*.rce; do
  { cat lib/hash/blake3.rce; echo; cat "$c"; } > "$tmp/$(basename "$c")"
  "$RCE" test "$tmp/$(basename "$c")"
done
