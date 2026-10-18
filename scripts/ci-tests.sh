#!/usr/bin/env bash
# Run the checker's tests in groups so CI can run the slow vector suites in
# parallel jobs, each with its own timeout.
#   scripts/ci-tests.sh fast         the library tests and every test binary not listed as slow
#   scripts/ci-tests.sh bin NAME     one slow test binary (NAME is in SLOW below)
#   scripts/ci-tests.sh slow-list    print the slow binaries (the CI matrix must match)
# RCE_FEATURES, if set, is passed to cargo as --features (for example tree-sitter).
# A name in SLOW that has no tests/NAME.rs is an error, so the list cannot drift.
set -euo pipefail
cd "$(dirname "$0")/../spec/checker"
SLOW="dst_chat emit framing_rce fuel golden_apps group_rce hpke_rce mls_rce pq_rce protect_rce treekem_rce welcome_rce"
for n in $SLOW; do [ -f "tests/$n.rs" ] || { echo "ci-tests: tests/$n.rs does not exist; update SLOW" >&2; exit 1; }; done
feat=()
if [ -n "${RCE_FEATURES:-}" ]; then feat=(--features "$RCE_FEATURES"); fi
case "${1:-}" in
  slow-list) echo $SLOW; exit 0 ;;
  fast)
    args=(--lib)
    for f in tests/*.rs; do
      n=$(basename "$f" .rs)
      case " $SLOW " in *" $n "*) continue ;; esac
      args+=(--test "$n")
    done ;;
  bin)
    n="${2:?test binary name}"
    case " $SLOW " in *" $n "*) ;; *) echo "ci-tests: $n is not in SLOW" >&2; exit 1 ;; esac
    args=(--test "$n") ;;
  *) echo "usage: ci-tests.sh fast | bin NAME | slow-list" >&2; exit 2 ;;
esac
exec cargo test --release "${feat[@]}" "${args[@]}" -- --skip tree_sitter_agrees_with_rust_parser
