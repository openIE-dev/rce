#!/usr/bin/env bash
# W5 partner demo: one engine, one log, replay with every adapter disconnected.
#   1. terminal: a real shell on a PTY through the RCE host loop; ANSI screen
#      model and scrollback queried from RCE state
#   2. chat: two hosts over P4 channels; posts, edits, deletes, unread
#      markers, and an end-to-end encrypted DM (X3DH + Double Ratchet in RCE)
#   3. crash: kill -9 a host mid-session, reopen (verify on open), replay
#   4. RCEdb: typed schema, secondary index, lookup, as-of, SQL emit
#   5. cross-architecture: SHA-256 of every deterministic artefact, diffed
#      against the reference from x86_64 Linux (live with RCE_XARCH_HOST=host)
# Every host step ends with `rce host replay`, which reads only the log.
# Usage: crates/rce/scripts/partner-demo.sh   (from the repo root or anywhere)
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
checker="$here/spec/checker"
( cd "$checker" && cargo build --release -q --bin rce )
rce="$checker/target/release/rce"
work="$(mktemp -d "${TMPDIR:-/tmp}/rce-demo.XXXXXX")"
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
pause() { sleep "${1:-0.4}"; }
replay() { "$rce" host replay "$1" --log "$2" | sed -n 1p; }

say "1. terminal app on the RCE host (apps/terminal.rce)"
# The PTY transcript and the program's Screen and History answers share
# stdout; byte offsets taken before each query split them for printing.
tout="$work/term.out"
mark() { sleep 0.8; wc -c < "$tout" | tr -d ' ' > "$work/term.$1"; }
{
  pause 0.6; echo 'Resize(rows: 8, cols: 80)'
  pause; echo 'echo hello from a shell inside RCE: $((6*7))'
  pause; echo 'seq 1 12'
  pause; echo "printf '\\033[2;10H\\033[1;31mpainted by CSI\\033[0m\\033[8;1H'"
  mark screen; echo 'Screen'
  mark history; echo 'History(start: 0, count: 4)'
  mark exit; echo 'exit'
  pause 1
} | "$rce" host serve "$here/apps/terminal.rce" --log "$work/term.log" --pty > "$tout" 2>&1
clean() { perl -pe 's/\e\[[0-9;?]*[A-Za-z]//g; s/\r//g'; }
part() { head -c "$2" "$tout" | tail -c "+$(( $1 + 1 ))"; }
s=$(cat "$work/term.screen"); h=$(cat "$work/term.history"); x=$(cat "$work/term.exit")
echo "-- shell transcript (ANSI stripped):"
part 0 "$s" | clean | grep -v '^$' | sed 's/^/   /'
echo "-- Screen query: the 8x80 screen model held in RCE state, one row per line:"
part "$s" "$h" | sed '$!s/^/  |/; $s/^/   /'
echo "-- History(start: 0, count: 4): the oldest scrollback lines:"
part "$h" "$x" | sed 's/^/  |/'
echo "-- host:"
tail -c "+$(( x + 1 ))" "$tout" | grep -E '^(durable|rce host)' | sed 's/^/   /'
say "replay the terminal session with no PTY"
replay "$here/apps/terminal.rce" "$work/term.log"

say "2. chat between two hosts over P4 remote channels"
pa=$(( 20000 + RANDOM % 20000 )); pb=$(( pa + 1 ))
A="127.0.0.1:$pa"; B="127.0.0.1:$pb"
mkfifo "$work/a.in" "$work/b.in"
"$rce" host serve "$here/apps/chat.rce" --log "$work/b.log" --chan "$B" --peer "$A" < "$work/b.in" > "$work/b.out" 2>"$work/b.err" &
bpid=$!
exec 4> "$work/b.in"
"$rce" host serve "$here/apps/chat.rce" --log "$work/a.log" --chan "$A" --peer "$B" < "$work/a.in" > "$work/a.out" 2>"$work/a.err" &
apid=$!
exec 3> "$work/a.in"
ann() { echo "$1" >&3; }
bob() { echo "$1" >&4; }
polls() { for _ in $(seq 1 "$1"); do ann Poll; bob Poll; sleep 0.1; done; }
bob "Setup(me: \"bob\", peers: [\"$A\"])"
ann "Setup(me: \"ann\", peers: [\"$B\"])"
pause 0.5
ann 'Do(op: "ws", conv: "acme", topic: "", body: "Acme")'
ann 'Do(op: "post", conv: "acme/general", topic: "launch", body: "ship it?")'
ann 'Do(op: "post", conv: "acme/general", topic: "launch", body: "draft notes in the doc")'
ann 'Do(op: "read", conv: "acme/general", topic: "launch", body: "")'
polls 10
bob 'Do(op: "post", conv: "acme/general", topic: "launch", body: "yes, shipping")'
bob 'Do(op: "read", conv: "acme/general", topic: "launch", body: "")'
polls 10
# ann edits her first message and deletes her second (only the author may)
ann 'Do(op: "edit", conv: "acme/general\u{1f}launch\u{1f}000000000002\u{1f}ann", topic: "", body: "ship it today?")'
ann 'Do(op: "delete", conv: "acme/general\u{1f}launch\u{1f}000000000003\u{1f}ann", topic: "", body: "")'
# bob tries to edit ann's message: the author rule leaves it unchanged
bob 'Do(op: "edit", conv: "acme/general\u{1f}launch\u{1f}000000000002\u{1f}ann", topic: "", body: "hijacked")'
bob 'Do(op: "post", conv: "acme/general", topic: "launch", body: "tagging v1.0 now")'
# an end-to-end encrypted DM: key agreement and ratchet run as queued jobs, one per Poll
bob 'Do(op: "dm", conv: "ann", topic: "", body: "the release key is in the vault")'
for _ in $(seq 1 300); do
  polls 1
  ann 'Show(conv: "dm:ann,bob", topic: "")'; sleep 0.1
  grep -q 'vault' "$work/a.out" && break
done
polls 5
ann 'Show(conv: "acme/general", topic: "launch")'; ann 'Unread'
bob 'Show(conv: "acme/general", topic: "launch")'; pause 1
exec 3>&-; wait "$apid" || true
echo "ann's thread (new marker after her read marker, the edit, the delete):"
awk '/^-- new --|^[a-z]+: /' "$work/a.out" | grep -v vault | tail -5
echo "ann's unread counts:"; grep 'unread$' "$work/a.out"
echo "ann's DM thread (decrypted on her replica with .utf8()):"; grep -m1 vault "$work/a.out"
n=$(grep -c 'the release key is in the vault' "$work/a.log" || true)
echo "plaintext of the DM in ann's log D: $n occurrences (only ratchet headers and ciphertext crossed the wire)"
grep -q hijacked "$work/a.out" "$work/b.out" && { echo "FAIL: a non-author edit was applied"; exit 1; } || echo "bob's edit of ann's message: not applied (author rule)"

say "3. crash: kill -9 bob's host mid-session"
kill -9 "$bpid" 2>/dev/null || true
exec 4>&-
wait "$bpid" 2>/dev/null || true
echo "bob's view before the crash:"; awk '/^[a-z]+: /' "$work/b.out" | tail -3
say "reopen both logs: verify on open (HS-4), replay with every adapter disconnected"
replay "$here/apps/chat.rce" "$work/b.log"
replay "$here/apps/chat.rce" "$work/a.log"
# The replayed state holds this device's private keys and ratchet state
# (they stay in its log D, which never leaves the device); print them as
# short digests only.
secrets=keys,sessions,held
echo "ann's replayed key material (rce host replay --redact $secrets):"
"$rce" host replay "$here/apps/chat.rce" --log "$work/a.log" --redact "$secrets" | sed -n 2p \
  | grep -oE '(keys|sessions|held) = <redacted sha256:[0-9a-f]+>' | sed 's/^/  /'

say "4. RCEdb: the database is a log D (rce db, serverless, in-process)"
db="$work/db.log"
run() { echo "\$ rce db D $*"; "$rce" db "$db" "$@" 2>&1 | grep -v '^durable' | sed 's/^/  /'; }
run schema users 'name:text,team:text,age:i64'
run insert users u1 $'Ann\tinfra\t41'
run insert users u2 $'Bob\tapps\t35'
run index users team
echo "\$ rce db D insert users u3 Cy/infra/29"
asof=$("$rce" db "$db" insert users u3 $'Cy\tinfra\t29' | awk '/^ok seq/ {print $3}')
echo "  ok seq $asof"
run lookup users team infra
run delete users u1
run lookup users team infra
run describe
echo "(as of seq $asof, before the delete:)"
run as-of "$asof" scan users
run emit sqlite > "$work/db.sql"; sed -n 1,8p "$work/db.sql"
replay "$here/rcedb/rcedb.rce" "$db"

say "5. cross-architecture bit identity"
bash "$here/scripts/xarch-hashes.sh" > "$work/xarch.local"
ref="$here/scripts/xarch-hashes.x86_64-linux.txt"
if [ -n "${RCE_XARCH_HOST:-}" ]; then
  echo "running on $RCE_XARCH_HOST ..."
  rsync -a --delete --exclude target --exclude '.git' "$here/" "$RCE_XARCH_HOST:rce-xarch/rce/"
  ssh "$RCE_XARCH_HOST" 'cd rce-xarch/rce/spec/checker && ~/.cargo/bin/cargo build --release -q --bin rce && cd ../.. && uname -sm && bash scripts/xarch-hashes.sh' > "$work/xarch.remote"
  sed -n 1p "$work/xarch.remote"; sed 1d "$work/xarch.remote" > "$work/xarch.ref"; ref="$work/xarch.ref"
fi
echo "$(wc -l < "$work/xarch.local" | tr -d ' ') artefacts on $(uname -sm); reference: $(basename "$ref")"
if diff <(grep -v "^#" "$ref") "$work/xarch.local" > "$work/xarch.diff"; then
  echo "identical: every trace, host state hash and crypto vector agrees bit for bit"
else
  echo "DIFFERENT:"; cat "$work/xarch.diff"; exit 1
fi
echo; echo "logs kept in $work"
