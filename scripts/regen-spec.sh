#!/usr/bin/env bash
# The spec's derived artefacts, regenerated in one command:
#   spec/RCE-SPEC.md, spec/book/INDEX.md            from the chapters (rce-book build)
#   spec/data/spec-data.json, spec/data/spec-tables.json  from the book (rce-spec-data)
# Edit only spec/book/NN-*.md and never hand-edit these four files; run this
# script and commit its output with the chapter change.
#   scripts/regen-spec.sh          regenerate all four
#   scripts/regen-spec.sh --check  fail if any is stale (.githooks/pre-push runs this)
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
checker="$here/spec/checker"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$checker/target}"
( cd "$checker" && cargo build -q --release --bin rce-book --bin rce-spec-data )
book="$CARGO_TARGET_DIR/release/rce-book"
data="$CARGO_TARGET_DIR/release/rce-spec-data"
if [ "${1:-}" = "--check" ]; then
  "$book" build --check
  "$data" --check
else
  "$book" build
  "$data"
fi
