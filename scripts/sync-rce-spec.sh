#!/usr/bin/env bash
# Refresh vendor/rce-spec from a checkout of https://github.com/openIE-dev/rce.
#
#   scripts/sync-rce-spec.sh /path/to/rce
#
# Copies the reference checker (spec/checker, without its build output and
# without the tests that need files outside the checker: the draft 0.6
# fixtures, the Lean vectors and sources and the tree-sitter cross-check),
# the specification book (read by the grammar and cost tests) and the
# example programs, and records the source commit in vendor/rce-spec/SOURCE.
set -eu
SRC="${1:?usage: scripts/sync-rce-spec.sh /path/to/rce}"
cd "$(dirname "$0")/.."
DST=vendor/rce-spec
rm -rf "$DST/checker" "$DST/examples" "$DST/book"
mkdir -p "$DST"
cp -a "$SRC/spec/checker" "$DST/checker"
rm -rf "$DST/checker/target"
cp -a "$SRC/spec/examples" "$DST/examples"
cp -a "$SRC/spec/book" "$DST/book"
for t in conc06 conc06_check conc06_run conc06b_check conc06b_run emit06 emit06b \
         emit_run lean_models lean_vectors par_paths theorem_index tree_sitter_sync \
         traces version surface_status; do
    rm -f "$DST/checker/tests/$t.rs"
done
rm -f "$DST/checker/tests/emit06.rce" "$DST/checker/Cargo.lock"
# rce-ide builds its own `rce` (crates/rce-ide), so the checker's standalone
# `rce` binary is dropped here to avoid two binaries with one name.
rm -f "$DST/checker/src/bin/rce.rs"
python3 - "$DST/checker/Cargo.toml" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
s = s.replace('[[bin]]\nname = "rce"\npath = "src/bin/rce.rs"\n\n', '')
open(p, 'w').write(s)
PY
commit=$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)
printf 'https://github.com/openIE-dev/rce\ncommit %s\n' "$commit" > "$DST/SOURCE"
echo "vendor/rce-spec synced from openIE-dev/rce $commit"
