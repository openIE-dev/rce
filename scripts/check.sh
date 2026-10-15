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
# A module with a "# uses: a.b, c.d" line is checked as a package together
# with lib/a/b.rce and lib/c/d.rce. scripts/uses.py parses that marker
# strictly: a malformed or unknown entry stops the run with an error. A tests/<area>/<name>_host.rce file
# holds checks too heavy for comptime fuel: it is flattened with the module
# and its first checks file (scripts/flatten.py), its "# events:" are fed
# to `rce host run` one per input line, and the final state must read
# "good = N, bad = 0" for the N events.
python3 scripts/uses.py --selftest
python3 scripts/uses.py --all
for f in lib/*/*.rce; do
  uses=$(python3 scripts/uses.py "$f") || { echo "check.sh: bad dependency marker in $f" >&2; exit 1; }
  pk="$tmp/pkg"; rm -rf "$pk"; mkdir "$pk"
  for u in $uses; do cp "lib/$u.rce" "$pk/"; done
  if [ -n "$uses" ]; then cp "$f" "$pk/"; "$RCE" check "$pk"; else "$RCE" check "$f"; fi
  t="tests/${f#lib/}"; t="${t%.rce}_checks"
  # one comptime fuel budget per file: heavy vector sets are sharded
  for c in "$t".rce "$t"_*.rce; do
    [ -f "$c" ] || continue
    o="$pk/$(basename "$f")"
    { cat "$f"; echo; cat "$c"; } > "$o"
    if [ -n "$uses" ]; then "$RCE" test "$pk"; else "$RCE" test "$o"; fi
  done
  h="${t%_checks}_host.rce"
  if [ -f "$h" ]; then
    python3 scripts/flatten.py "$f" "$t.rce" "$h" > "$tmp/host.rce"
    # "# events: A, B" lists events without arguments; "# event: E" lines
    # give one event each (with arguments). "# host-flags: --fs" grants a
    # fresh empty file root.
    if grep -q "^# event: " "$h"; then ev=$(sed -n "s/^# event: //p" "$h"); else ev=$(sed -n "s/^# events: //p" "$h" | tr -d ' ' | tr ',' '\n'); fi
    hf=""; if grep -q "^# host-flags: --fs" "$h"; then rm -rf "$tmp/fs"; mkdir "$tmp/fs"; hf="--fs $tmp/fs"; fi
    n=$(printf '%s\n' "$ev" | grep -c .)
    res=$(printf '%s\n' "$ev" | "$RCE" host run "$tmp/host.rce" --log "$tmp/hostlog" $hf)
    printf '%s\n' "$res" | tail -3
    printf '%s\n' "$res" | grep -q "good = $n, bad = 0 }" || { echo "FAIL host checks $h"; exit 1; }
    rm -rf "$tmp/hostlog"*
  fi
done
# A tests/<area>/<name>_probes.py script checks the rce checker itself
# against a rule table of the spec (e.g. the effect contexts of 5.2).
for p in tests/*/*_probes.py; do
  [ -f "$p" ] || continue
  RCE="$RCE" python3 "$p" | tail -1
  RCE="$RCE" python3 "$p" > /dev/null || { echo "FAIL probes $p"; exit 1; }
done
