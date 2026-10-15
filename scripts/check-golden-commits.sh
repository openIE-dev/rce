#!/usr/bin/env bash
# W6 E1: a changed golden needs its own commit saying why.
# For every commit in RANGE (default origin/rce/host..HEAD) that modifies or
# deletes a committed golden (crates/rce/spec/traces/**), every file the
# commit touches must be a golden, and the message must have a body (the
# why) after the subject. Adding a new golden with its new example is fine.
set -euo pipefail
range="${1:-origin/rce/host..HEAD}"
golden='^crates/rce/spec/traces/'
bad=0
for c in $(git rev-list --no-merges "$range"); do
  changed=$(git diff-tree --no-commit-id -r --name-status "$c")
  if echo "$changed" | grep -E "^[MD]	" | cut -f2 | grep -qE "$golden"; then
    others=$(echo "$changed" | cut -f2- | tr '\t' '\n' | grep -vE "$golden" || true)
    body=$(git log -1 --format=%b "$c" | grep -v '^\s*$' | grep -v '^Co-authored-by:' || true)
    if [ -n "$others" ]; then
      echo "$(git log -1 --format='%h %s' "$c"): changes a golden together with other files:"; echo "$others" | sed 's/^/  /'; bad=1
    fi
    if [ -z "$body" ]; then
      echo "$(git log -1 --format='%h %s' "$c"): changes a golden without saying why (no message body)"; bad=1
    fi
  fi
done
[ $bad -eq 0 ] && echo "golden commits ok ($range)"
exit $bad
