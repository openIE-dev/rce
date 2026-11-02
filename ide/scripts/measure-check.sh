#!/usr/bin/env bash
# measure-check.sh RCE: run `RCE test measure` (the IDE build of rce) on a
# small program and check its energy line. Two outcomes pass, and the
# script prints which one it saw:
#   - a D.11 receipt term: basis `reported`, amortization `per_call`, and a
#     provenance with exactly the nine fields of D.11 rule 1, valid by it;
#   - the line `energy: unmeasured: no counter available (REASON)`, on a
#     machine whose counters cannot be opened (GitHub's macOS runners are
#     virtual machines and may not expose IOReport's energy channels).
# Anything else (no energy line, a malformed receipt, a failed read, a
# counter that did not move) fails.
set -euo pipefail
rce=${1:?usage: measure-check.sh RCE}
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
printf 'fn f(n: i64) -> i64\n    return n + 1\n' > "$d/a.rce"
set +e
"$rce" test measure "$d/a.rce" > "$d/out" 2> "$d/err"
status=$?
set -e
echo "rce test measure: status $status"
cat "$d/out"
cat "$d/err" >&2
[ "$status" -eq 0 ] || { echo "measure-check: status $status, not 0"; exit 1; }
python3 -I - "$d/out" <<'EOF'
import json, sys
lines = [l.rstrip("\n") for l in open(sys.argv[1]) if l.startswith("energy")]
receipt = [l for l in lines if l.startswith("energy receipt: ")]
unmeasured = [l for l in lines if l.startswith("energy: unmeasured: ")]
FIELDS = ["boundary", "counter", "end_unix_ns", "idle_baseline_w", "interval_s",
          "machine", "sampler", "start_unix_ns", "wraps"]

def fail(why):
    print(f"measure-check: FAIL: {why}")
    sys.exit(1)

if len(receipt) == 1 and not unmeasured:
    t = json.loads(receipt[0][len("energy receipt: "):])
    if sorted(t) != ["amortization", "basis", "j", "provenance"]:
        fail(f"term keys {sorted(t)}")
    if t["basis"] != "reported" or t["amortization"] != {"kind": "per_call"}:
        fail(f"basis {t['basis']!r}, amortization {t['amortization']!r}")
    j, p = t["j"], t["provenance"]
    if not (isinstance(j, (int, float)) and j >= 0):
        fail(f"j {j!r}")
    if list(p) != FIELDS:
        fail(f"provenance fields {list(p)}, want the nine of D.11 rule 1 in byte order")
    for k in ("sampler", "counter", "machine", "boundary"):
        if not (isinstance(p[k], str) and p[k]):
            fail(f"{k} {p[k]!r}")
    ints = all(isinstance(p[k], int) and not isinstance(p[k], bool) and p[k] >= 0
               for k in ("start_unix_ns", "end_unix_ns", "wraps"))
    if not ints or p["start_unix_ns"] >= p["end_unix_ns"]:
        fail("start_unix_ns, end_unix_ns, wraps")
    if not (p["interval_s"] > 0 and p["idle_baseline_w"] >= 0):
        fail("interval_s, idle_baseline_w")
    print(f"OUTCOME: reported receipt, {j} J over {p['interval_s']} s, "
          f"boundary {p['boundary']}, sampler {p['sampler']!r}, counter {p['counter']!r}")
elif len(unmeasured) == 1 and not receipt:
    u = unmeasured[0]
    if not (u.startswith("energy: unmeasured: no counter available (") and u.endswith(")")):
        fail(f"not the no-counter line: {u}")
    print(f"OUTCOME: unmeasured, {u}")
else:
    fail(f"want one receipt or one unmeasured line, got {lines}")
EOF
