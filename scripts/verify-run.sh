#!/usr/bin/env bash
# Run the verification checks that RCE paper section 3.7 quotes, keep the log
# of each, and write the facts file from them with scripts/claim-counts.py.
#
#   scripts/verify-run.sh [DIR]          DIR defaults to verify-out
#
# Steps (each log and exit status goes to DIR):
#   tests                      spec/checker: the test binaries of cargo test --release, run by
#                              scripts/verify-tests.py from the files of one build (hard links,
#                              and the target directory's lock while a test runs target/release/rce),
#                              with TMPDIR private to the run
#   spec_check                 spec/checker: rce-spec-check, built and run the same way
#   spec_check_three_parsers   the same with --features tree-sitter
#   wasm                       spec/wasm-test: npm ci && node test.mjs
#   lean                       scripts/lean-audit.sh when lake is on PATH; otherwise
#                              the log of a public CI job of .github/workflows/lean.yml:
#                              RCE_LEAN_CI_LOG=FILE (a saved job log), or RCE_LEAN_CI_JOB=ID
#                              fetched with gh; RCE_LEAN_CI_RUN=ID names its run
# Then: claim-counts.py --runs DIR --write DIR/rce-facts.json, and --check on it.
# RCE_VERIFY_BASE=REV records whether the tested trees equal those of REV.
# A step's outcome is its own exit status; this script runs every step and
# exits 1 if any step or the final check did not pass.
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:-verify-out}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"

git -C "$root" rev-parse HEAD >"$out/run_commit"
git -C "$root" status --porcelain --untracked-files=no >"$out/worktree_status"

step() { # id cwd command...
  local id="$1" cwd="$2"; shift 2
  echo "verify-run: $id: $*" >&2
  (cd "$root/$cwd" && "$@") >"$out/$id.log" 2>&1
  local st=$?
  echo "$st" >"$out/$id.status"
  echo "verify-run: $id: exit $st" >&2
}

for s in "tests tests $out" "spec_check bin rce-spec-check $out/spec_check.log" \
         "spec_check_three_parsers bin rce-spec-check tree-sitter $out/spec_check_three_parsers.log"; do
  set -- $s
  id="$1"; shift
  echo "verify-run: $id: scripts/verify-tests.py $*" >&2
  python3 "$root/scripts/verify-tests.py" "$@" 2>>"$out/verify-tests.stderr"
  echo "verify-run: $id: exit $(cat "$out/$id.status" 2>/dev/null)" >&2
done
step wasm spec/wasm-test bash -c 'npm ci --no-audit --no-fund && node test.mjs'

if command -v lake >/dev/null 2>&1 || [ -x "$HOME/.elan/bin/lake" ]; then
  step lean . bash scripts/lean-audit.sh
else
  ci_log="${RCE_LEAN_CI_LOG:-}"
  if [ -z "$ci_log" ] && [ -n "${RCE_LEAN_CI_JOB:-}" ] && command -v gh >/dev/null 2>&1; then
    gh api "repos/openIE-dev/rce/actions/jobs/${RCE_LEAN_CI_JOB}/logs" >"$out/lean-audit.ci.log" 2>/dev/null \
      && ci_log="$out/lean-audit.ci.log"
  fi
  if [ -n "$ci_log" ] && [ -f "$ci_log" ]; then
    [ "$ci_log" -ef "$out/lean-audit.ci.log" ] || cp "$ci_log" "$out/lean-audit.ci.log"
    printf '{"repo": "https://github.com/openIE-dev/rce", "workflow": ".github/workflows/lean.yml", "job_name": "proofs", "run_id": %s, "job_id": %s}\n' \
      "${RCE_LEAN_CI_RUN:-null}" "${RCE_LEAN_CI_JOB:-null}" >"$out/lean-audit.ci.json"
    echo "verify-run: lean: lake not found; the audit is read from the CI job log $ci_log" >&2
  else
    echo "verify-run: lean: lake not found and no CI job log given; the Lean audit is recorded as not run" >&2
  fi
fi

base_args=()
[ -n "${RCE_VERIFY_BASE:-}" ] && base_args=(--base "$RCE_VERIFY_BASE")
python3 "$root/scripts/claim-counts.py" --runs "$out" "${base_args[@]}" --write "$out/rce-facts.json"
w=$?
python3 "$root/scripts/claim-counts.py" --check --runs "$out" "${base_args[@]}" --facts "$out/rce-facts.json"
c=$?
[ "$w" -eq 0 ] && [ "$c" -eq 0 ]
