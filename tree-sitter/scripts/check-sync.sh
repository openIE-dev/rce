#!/usr/bin/env bash
# Check that this grammar is the same as the copy in the rce repository,
# which the reference checker uses as one of its three parsers.
#
#   scripts/check-sync.sh /path/to/rce
set -eu
RCE_REPO="${1:?usage: scripts/check-sync.sh /path/to/rce}"
cd "$(dirname "$0")/.."
status=0
for f in grammar.js src/scanner.c src/grammar.json src/node-types.json; do
    if cmp -s "$f" "$RCE_REPO/spec/tree-sitter/$f"; then
        echo "same   $f"
    else
        echo "DIFFER $f"
        status=1
    fi
done
exit $status
