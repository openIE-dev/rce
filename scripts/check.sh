#!/bin/sh
# Run the `check` lines of every library. pq, hpke and mls build on
# crypto, so each is tested with a `#!lib crypto` line in front (the rce
# binary appends its embedded crypto, without crypto's own checks).
# Set RCE to the rce binary (default: rce on PATH).
set -e
RCE=${RCE:-rce}
cd "$(dirname "$0")/.."
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
"$RCE" test lib/crypto.rce
for n in pq hpke mls; do
  { echo "module ${n}_checks"; echo "#!lib crypto"; grep -v '^module ' "lib/$n.rce"; } > "$tmp/$n.rce"
  "$RCE" test "$tmp/$n.rce"
done
# Foundation modules under lib/<area>/: each is tested alone (it must admit
# as a standalone module) and then with its checks file appended (vectors
# and property sweeps live in tests/<area>/<name>_checks[_N].rce so a library's
# own comptime fuel stays small for programs that use it).
for f in lib/*/*.rce; do
  "$RCE" check "$f"
  t="tests/${f#lib/}"; t="${t%.rce}_checks"
  # one comptime fuel budget per file: heavy vector sets are sharded
  for c in "$t".rce "$t"_*.rce; do
    [ -f "$c" ] || continue
    o="$tmp/$(basename "$c")"
    { cat "$f"; echo; cat "$c"; } > "$o"
    "$RCE" test "$o"
  done
done
