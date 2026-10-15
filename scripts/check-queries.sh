#!/usr/bin/env bash
# Compiles every query in queries/ against the grammar and runs it over the
# examples and highlight tests, then checks the version comment gate.
set -eu
cd "$(dirname "$0")/.."
lib="$(mktemp -d)/rce.so"
tree-sitter build -o "$lib" . >/dev/null
files=$(ls examples/*.rce test/highlight/*.rce 2>/dev/null)
status=0
for q in queries/*.scm; do
    if out=$(tree-sitter query -l "$lib" --lang-name rce -q "$q" $files 2>&1) && ! grep -qi 'error' <<<"$out"; then
        echo "ok     $q"
    else
        echo "FAILED $q"; echo "$out" | head -5; status=1
    fi
done
# `par` is a keyword only after `# rce-version: 0.6` on the first line (spec 2.7).
tmp=$(mktemp -d)
printf '# rce-version: 0.6\nfn f()\n    par\n        branch a\n            pass\n' > "$tmp/v06.rce"
if tree-sitter parse -l "$lib" --lang-name rce "$tmp/v06.rce" | grep -q par_stmt; then echo "ok     version 0.6 gate"; else echo "FAILED version 0.6 gate"; status=1; fi
printf 'fn f()\n    let par = 1\n' > "$tmp/v05.rce"
if tree-sitter parse -l "$lib" --lang-name rce "$tmp/v05.rce" | grep -q ERROR; then echo "FAILED par as a name before 0.6"; status=1; else echo "ok     par is a name without the version comment"; fi
exit $status
