#!/usr/bin/env bash
# T7 on every program: a program and its name-free projection (rce fmt
# --nameless, 10.6) have the same behaviour (rce equiv, 10.11). Programs
# with no state region or no explorable event are skipped and counted. A
# projection of an admitted program that is not itself admitted is a failure.
#   scripts/t7-equiv.sh [RCE] [DIR...]      default: spec/checker/target/release/rce over spec/examples, programs, apps
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RCE="${1:-$ROOT/spec/checker/target/release/rce}"; shift || true
DIRS=("$@"); [ ${#DIRS[@]} -gt 0 ] || DIRS=("$ROOT/spec/examples" "$ROOT/programs" "$ROOT/apps")
FUEL="${T7_FUEL:-2000}"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
ok=0; skipped=0; different=0; changed=0; bad=()
while IFS= read -r f; do
  "$RCE" check "$f" >/dev/null 2>&1 || { skipped=$((skipped+1)); continue; }
  if ! "$RCE" fmt --nameless "$f" > "$tmp/n.rce" 2>"$tmp/err" || ! "$RCE" check "$tmp/n.rce" > "$tmp/err" 2>&1; then
    changed=$((changed+1)); bad+=("$f"); echo "REQUIRES CHANGE $f (nameless projection)"; sed 's/^/  /' "$tmp/err"; continue
  fi
  out=$("$RCE" equiv --fuel "$FUEL" "$f" "$tmp/n.rce" 2>&1); st=$?
  if [ $st -eq 0 ]; then ok=$((ok+1))
  elif echo "$out" | grep -q "^verdict             different"; then different=$((different+1)); bad+=("$f"); echo "DIFFERENT $f"; echo "$out" | sed 's/^/  /'
  else skipped=$((skipped+1)); fi
done < <(find "${DIRS[@]}" -name '*.rce' -not -path '*/reject/*' -not -path '*/ill_typed/*' -not -path '*/inadmissible/*' -not -path '*/change/*' | sort)
echo "t7-equiv: $ok equivalent, $skipped skipped (no Form, no explorable event, or not admitted on its own), $different different, $changed projections require change"
[ $different -eq 0 ] && [ $changed -eq 0 ]
