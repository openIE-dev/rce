#!/usr/bin/env bash
# Compiles every query in queries/ against the grammar and runs it over the
# examples and highlight tests, then checks that `par` is reserved.
# A `# rce-version:` line is a comment, not a language gate (spec 2.7).
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
# `par` is a reserved word of the one language (spec 2.4): a name may not be `par`,
# and `for ... in ... par` is a parallel loop.
tmp=$(mktemp -d)
printf 'fn f()\n    let par = 1\n' > "$tmp/name.rce"
if tree-sitter parse -l "$lib" --lang-name rce "$tmp/name.rce" | grep -q ERROR; then echo "ok     par is reserved"; else echo "FAILED par accepted as a name"; status=1; fi
printf 'fn f()\n    for x in xs par\n        pass\n' > "$tmp/loop.rce"
if tree-sitter parse -l "$lib" --lang-name rce "$tmp/loop.rce" | grep -q "PAR\|for_stmt"; then echo "ok     for ... par"; else echo "FAILED for ... par"; status=1; fi
exit $status
