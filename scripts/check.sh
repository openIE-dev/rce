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
# and property sweeps live in tests/<area>/<name>_checks.rce so a library's
# own comptime fuel stays small for programs that use it).
for f in lib/*/*.rce; do
  "$RCE" check "$f"
  t="tests/${f#lib/}"; t="${t%.rce}_checks.rce"
  if [ -f "$t" ]; then
    { cat "$f"; echo; cat "$t"; } > "$tmp/$(basename "$f")"
    "$RCE" test "$tmp/$(basename "$f")"
  fi
done
