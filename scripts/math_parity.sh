#!/bin/sh
# Cross-target parity for lib/math: emit the module with a vector shard per
# function (and the derived checks) to Rust, Python and JavaScript, run each
# program and require every check true and the transcripts identical byte
# for byte. Needs rustc (edition 2024), python3 and node; Scala 3 too when
# scala-cli is on PATH (it is skipped otherwise, and says so).
# Usage: RCE=path/to/rce scripts/math_parity.sh [SHARD]   (default 0)
set -e
RCE=${RCE:-rce}
n=${1:-0}
cd "$(dirname "$0")/.."
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
for c in tests/math/math_checks.rce tests/math/math_checks_eft.rce tests/math/math_checks_ties.rce tests/math/math_checks_*_"$n".rce; do
  b=$(basename "$c" .rce)
  { cat lib/math/math.rce; echo; cat "$c"; } > "$tmp/$b.rce"
  "$RCE" emit rust "$tmp/$b.rce" > "$tmp/$b.rs"
  "$RCE" emit python "$tmp/$b.rce" > "$tmp/$b.py"
  "$RCE" emit js "$tmp/$b.rce" > "$tmp/$b.js"
  rustc --edition 2024 -O -o "$tmp/$b.bin" "$tmp/$b.rs" 2>/dev/null
  "$tmp/$b.bin" > "$tmp/$b.rs.out"
  python3 "$tmp/$b.py" > "$tmp/$b.py.out"
  node "$tmp/$b.js" > "$tmp/$b.js.out"
  cmp "$tmp/$b.rs.out" "$tmp/$b.py.out"
  cmp "$tmp/$b.rs.out" "$tmp/$b.js.out"
  sc=""
  if command -v scala-cli >/dev/null; then
    "$RCE" emit scala "$tmp/$b.rce" > "$tmp/$b.scala"
    scala-cli run --server=false -q "$tmp/$b.scala" > "$tmp/$b.sc.out" 2>/dev/null
    cmp "$tmp/$b.rs.out" "$tmp/$b.sc.out"; sc=" = scala"
  fi
  t=$(grep -c '^check .* = true$' "$tmp/$b.rs.out")
  f=$(grep '^check ' "$tmp/$b.rs.out" | grep -vc ' = true$' || true)
  [ "$f" = 0 ] || { echo "$b: $f checks false"; exit 1; }
  echo "$b: rust = python = js$sc, $t checks true"
done
