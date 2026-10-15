#!/usr/bin/env bash
# Cross-architecture bit identity: print one line per deterministic artefact
# (reference traces of every example, the RCEdb and chat host runs, the
# crypto vectors) with the SHA-256 of what it produced. Run on two machines
# and diff the outputs: any difference is a determinism bug (HC-7, T37).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
rce="$here/spec/checker/target/release/rce"
sha() { if command -v sha256sum >/dev/null; then sha256sum | cut -c1-64; else shasum -a 256 | cut -c1-64; fi; }
for f in "$here"/spec/examples/*.rce; do
  printf '%s trace %s\n' "$("$rce" trace --json "$f" 2>&1 | sed "s#$here##g" | sha)" "$(basename "$f")"
done
tmp="$(mktemp -d)"
"$rce" host run "$here/spec/examples/55_rcedb.rce" --log "$tmp/db.log" 'Put(key: "b", val: "2"), Put(key: "a", val: "1"), Scan(start: ""), Delete(key: "a")' 2>/dev/null | grep -o 'state_hash [0-9a-f]*' | sed 's/^/host-rcedb /'
# chat: the Random result is supplied as an event (a live host draws fresh
# bytes, so its state hash differs on every run); the Polls run key
# generation (X25519, Ed25519 in RCE) as queued jobs
rnd=$(printf '5a%.0s' $(seq 1 96))
ev="Setup(me: \"ann\", peers: []), RandomDone(req: 1, status: \"delivered\", data: some(\"$rnd\"), err: \"\"), Poll, Poll, Poll, Do(op: \"post\", conv: \"c\", topic: \"t\", body: \"hi\"), Do(op: \"react\", conv: \"c\", topic: \"+1\", body: \"\"), Do(op: \"dm\", conv: \"bob\", topic: \"\", body: \"secret\")"
printf '%s trace apps/chat.rce (keygen, post, react, dm)\n' "$("$rce" trace --json "$here/apps/chat.rce" "$ev" 2>&1 | sha)"
"$rce" test "$here/lib/crypto.rce" | head -1 | sed 's/^/crypto /;s#'"$here"'##'
rm -rf "$tmp"
