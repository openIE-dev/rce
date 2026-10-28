#!/usr/bin/env bash
# T7 on every program: a program and its name-free projection (rce fmt
# --nameless, 10.6) have the same behaviour (rce equiv, 10.11). Programs
# with no state region or no explorable event are skipped and counted.
#   scripts/t7-equiv.sh [RCE] [DIR...]      default: spec/checker/target/release/rce over spec/examples, programs, apps
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RCE="${1:-$ROOT/spec/checker/target/release/rce}"; shift || true
DIRS=("$@"); [ ${#DIRS[@]} -gt 0 ] || DIRS=("$ROOT/spec/examples" "$ROOT/programs" "$ROOT/apps")
FUEL="${T7_FUEL:-2000}"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
ok=0; skipped=0; different=0; bad=()
while IFS= read -r f; do
  "$RCE" check "$f" >/dev/null 2>&1 || { skipped=$((skipped+1)); continue; }
  "$RCE" fmt --nameless "$f" > "$tmp/n.rce" 2>/dev/null || { skipped=$((skipped+1)); continue; }
  out=$("$RCE" equiv --fuel "$FUEL" "$f" "$tmp/n.rce" 2>&1); st=$?
  if [ $st -eq 0 ]; then ok=$((ok+1))
  elif echo "$out" | grep -q "^verdict             different"; then different=$((different+1)); bad+=("$f"); echo "DIFFERENT $f"; echo "$out" | sed 's/^/  /'
  else skipped=$((skipped+1)); fi
done < <(find "${DIRS[@]}" -name '*.rce' -not -path '*/fit/*' -not -path '*/ill_typed/*' -not -path '*/inadmissible/*' | sort)
echo "t7-equiv: $ok equivalent, $skipped skipped (no Form, no explorable event, or reported standalone), $different different"
[ $different -eq 0 ]
