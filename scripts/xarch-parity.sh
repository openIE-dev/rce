#!/usr/bin/env bash
# Replay-hash parity: run scripts/xarch-hashes.sh on this target (RCE_BIN and
# RCE_RUNNER as there) and compare every line with the committed golden set
# scripts/xarch-hashes.x86_64-linux.txt. Exit 1 on any difference, printing the diff;
# the produced set is kept in $XARCH_OUT (default xarch-hashes.out.txt) for
# the CI artefact.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
golden="$here/scripts/xarch-hashes.x86_64-linux.txt"
out="${XARCH_OUT:-xarch-hashes.out.txt}"
# XARCH_SKIP: an optional extended regex of golden lines this target declares
# unsupported (e.g. the host run under WASI); they are listed, never hidden.
skip="${XARCH_SKIP:-^\$}"
bash "$here/scripts/xarch-hashes.sh" | tr -d '\r' > "$out"
want() { grep -v '^#' "$golden" | grep -Ev "$skip" || true; }
got() { grep -Ev "$skip" "$out" || true; }
if [ -n "${XARCH_SKIP:-}" ]; then
  echo "declared unsupported on this target (not compared): $(grep -v '^#' "$golden" | grep -Ec "$skip" || true) golden line(s) matching /$XARCH_SKIP/"
fi
n=$(want | wc -l | tr -d ' ')
if diff <(want) <(got); then
  echo "parity ok: $n of $n hashes equal the golden set (${XARCH_TARGET:-$(uname -sm)})"
else
  echo "parity FAILED against $golden (${XARCH_TARGET:-$(uname -sm)})" >&2
  exit 1
fi
