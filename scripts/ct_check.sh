#!/bin/sh
# Checks for the constant-time work on crypto, pq and hpke.
#
#   tests/crypto/*_checks.rce   `check` lines (official vectors and properties).
#   tests/crypto/*_fuel.rce     probe functions, see below.
#
# The first line of each file may read `# needs: crypto pq hpke`: those
# library files (their own `check` lines removed) are put in front of the
# test file, and the result is run as one program. Each file has its own
# comptime fuel budget, so heavy vector sets are separate files.
#
# Fuel tests. A `_fuel.rce` file defines zero-parameter functions named
# `probe_<group>_<n>`. The file is emitted as Python by `rce emit python`
# and run with `--fuel-trace` (spec 26.11: one tick per evaluated
# expression, statement and call). Every probe of one group must use
# exactly the same fuel. A group is called with different secret inputs, so
# equal fuel means the number of evaluated steps does not depend on them.
# This is SOURCE-LEVEL, constant-fuel evidence only. It says nothing about
# the latency of the host's arithmetic, list indexing or cache behaviour,
# so it does not show wall-clock constant time.
#
# Set RCE to the rce binary. Set ONLY to a substring to select test files.
set -e
RCE=${RCE:-rce}
cd "$(dirname "$0")/.."
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
pass=0; fail=0; groups=0
for f in tests/crypto/*_checks.rce tests/crypto/*_fuel.rce; do
  [ -f "$f" ] || continue
  case "$f" in *"${ONLY:-}"*) ;; *) continue ;; esac
  needs=$(sed -n '1s/^# needs: //p' "$f")
  o="$tmp/$(basename "$f")"
  { echo "module t"; for n in $needs; do grep -v '^module \|^check ' "lib/$n.rce"; echo; done; cat "$f"; } > "$o"
  case "$f" in
  *_checks.rce)
    if "$RCE" test "$o" > "$o.out" 2>&1; then
      echo "ok   $f: $(grep -o '[0-9]* checks' "$o.out" | head -1)"; pass=$((pass+1))
    else
      echo "FAIL $f"; cat "$o.out"; fail=$((fail+1))
    fi;;
  *_fuel.rce)
    "$RCE" emit python "$o" > "$o.py"
    python3 "$o.py" --fuel-trace > "$o.out" 2>&1 || { echo "FAIL $f: run error"; tail -5 "$o.out"; fail=$((fail+1)); continue; }
    if python3 - "$o.out" "$f" <<'PY'
import sys, re, collections
g = collections.defaultdict(list)
name = None
for line in open(sys.argv[1]):
    m = re.match(r'fn (probe_(\w+?)_\d+) = ', line)
    if m: name = (m.group(2), m.group(1)); continue
    m = re.match(r'fuel (\d+)', line)
    if m and name: g[name[0]].append((name[1], int(m.group(1)))); name = None
bad = 0
for grp, v in sorted(g.items()):
    fs = {x[1] for x in v}
    if len(v) < 2 or len(fs) != 1:
        bad += 1; print("  GROUP %s: fuel %s" % (grp, [(n, k) for n, k in v]))
    else:
        print("  group %s: %d probes, fuel %d each" % (grp, len(v), fs.pop()))
sys.exit(1 if bad or not g else 0)
PY
    then echo "ok   $f"; pass=$((pass+1)); else echo "FAIL $f"; fail=$((fail+1)); fi;;
  esac
done
echo "ct_check: $pass files passed, $fail failed"
[ "$fail" = 0 ]
