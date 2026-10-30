#!/usr/bin/env bash
# Check every program in listings/ with the real RCE checker.
#
#   scripts/check-listings.sh           check everything
#   scripts/check-listings.sh --bless   also rewrite the expected .out files
#
# Rules:
#   * NAME.rce outside a change/ directory must pass `rce test`.
#     A directory with a program.txt is one program: its files are tested together.
#   * change/NAME.rce must require change under `rce check`, and the output
#     must contain the text in change/NAME.expect. change/NAME.with lists
#     extra files (relative to change/) that belong to the same program.
#   * NAME.cmd holds the arguments of one `rce` command, run from its own
#     directory. Its output (stdout, then the exit status) must equal NAME.out.
#
# Set RCE to the rce binary to use (default: `rce` on PATH).
set -u
RCE="${RCE:-rce}"
BLESS=0
[ "${1:-}" = "--bless" ] && BLESS=1
cd "$(dirname "$0")/../listings" || exit 2
fail=0
pass=0

ok()  { pass=$((pass + 1)); }
bad() { fail=$((fail + 1)); echo "FAIL $*"; }

# 1. programs that must pass
while IFS= read -r f; do
    d=$(dirname "$f")
    if [ -f "$d/program.txt" ]; then continue; fi
    if (cd "$d" && "$RCE" test "$(basename "$f")" >/dev/null); then ok; else bad "$f"; (cd "$d" && "$RCE" test "$(basename "$f")"); fi
done < <(find . -name '*.rce' -not -path '*/change/*' | sort)

while IFS= read -r p; do
    d=$(dirname "$p")
    # shellcheck disable=SC2046
    if (cd "$d" && "$RCE" test $(cat program.txt) >/dev/null); then ok; else bad "$d (program.txt)"; fi
done < <(find . -name program.txt | sort)

# 2. programs that must require change
while IFS= read -r f; do
    base="${f%.rce}"
    expect=$(cat "$base.expect")
    extra=()
    if [ -f "$base.with" ]; then mapfile -t extra < "$base.with"; fi
    out=$(cd "$(dirname "$f")" && "$RCE" check "${extra[@]}" "$(basename "$f")")
    status=$?
    if [ $status -eq 1 ] && printf '%s' "$out" | grep -qF -- "$expect"; then ok; else bad "$f (expected "requires change" containing: $expect)"; echo "$out"; fi
done < <(find . -path '*/change/*.rce' | sort)

# 3. command transcripts
while IFS= read -r c; do
    d=$(dirname "$c")
    n=$(basename "${c%.cmd}")
    got=$(cd "$d" && eval "\"$RCE\" $(cat "$n.cmd")" 2>&1; echo "exit status: $?")
    if [ $BLESS -eq 1 ]; then printf '%s\n' "$got" > "$d/$n.out"; fi
    if [ "$got" = "$(cat "$d/$n.out" 2>/dev/null)" ]; then ok; else bad "$c (output differs from $n.out)"; diff <(printf '%s\n' "$got") "$d/$n.out"; fi
done < <(find . -name '*.cmd' | sort)

echo "$pass passed, $fail failed"
[ $fail -eq 0 ]
