#!/bin/sh
# apps/chat.rce = apps/chat-app.rce + lib/crypto.rce without its module line
# and its checks (the vectors run in `rce test lib/crypto.rce`). A single
# file, so `rce host serve apps/chat.rce` needs no module search path.
set -e
here=$(cd "$(dirname "$0")/.." && pwd)
{
  cat "$here/apps/chat-app.rce"
  printf '\n# ---------------------------------------------------------------- lib/crypto.rce (generated, do not edit here)\n'
  grep -v '^module crypto$' "$here/lib/crypto.rce" | grep -v '^check '
} > "$here/apps/chat.rce"
