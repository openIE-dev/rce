#!/bin/sh
# Run the ecosystem examples on the host runtime (Appendix E) and check
# their results. Usage: ./run.sh [hpke|pqxdh|mls|sfu|rcedb|all]
# RCE is the rce binary (default: rce on PATH). Each run writes a durable
# log under a temporary directory, and `rce host replay` re-derives the
# state with every adapter disconnected (HC-8).
set -e
here=$(cd "$(dirname "$0")" && pwd)
RCE=${RCE:-rce}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0
# These programs use the `#!lib NAME` directive. The reference checker
# embeds the standard library (specification 4.1) and honours the directive,
# as the engine's rce does. The probe stays for an older binary: it says so
# instead of failing.
printf 'module lib_probe\n#!lib crypto\nfn f() -> List[i64]\n    return cat([1], [2])\n' > "$tmp/lib_probe.rce"
if ! "$RCE" check "$tmp/lib_probe.rce" > /dev/null 2>&1; then
  echo "ecosystem/run.sh: this rce has no #!lib directive (the engine's rce or an rce-ide release is needed); skipped"
  exit 0
fi
# the SFU app: next to this script in rce-examples, apps/sfu.rce in the engine tree
SFU=${SFU:-$here/sfu.rce}
[ -f "$SFU" ] || SFU="$here/../../apps/sfu.rce"
want() { # NAME OUTPUT-FILE NEEDLE...
  n=$1; f=$2; shift 2
  for s in "$@"; do
    if ! grep -qF -- "$s" "$f"; then echo "FAIL $n: missing $s"; fail=1; return; fi
  done
  echo "ok $n"
}
r32() { printf "%0.s$1" $(seq 32); }

hpke() {
  "$RCE" test "$here/hpke.rce" >/dev/null
  "$RCE" host run "$here/hpke.rce" --log "$tmp/hpke" 'KeyR(ikm: "6db9df30aa07dd42ee5e8181afdb977e538f5e1fec8a06223f33f7013e525037"), PubR, Seal(ske: "52c4a758a802cd8b936eceea314432798d5baf2d7e9235dc084ab1b9cfa2f736", info: "4f6465206f6e2061204772656369616e2055726e", aad: "436f756e742d30", pt: "4265617574792069732074727574682c20747275746820626561757479", seq: 0), Open(info: "4f6465206f6e2061204772656369616e2055726e", aad: "436f756e742d30", seq: 0), Open(info: "4f6465206f6e2061204772656369616e2055726e", aad: "436f756e742d31", seq: 0)' > "$tmp/hpke.out" 2>&1
  # RFC 9180 A.1 base mode, sequence 0; the second Open uses another AAD (additional authenticated data) and is refused
  want hpke "$tmp/hpke.out" \
    'pkr = "3948cfe0ad1ddb695d780e59077195da6c56506b027329794ab02bca80815c4d"' \
    'sealed = "37fda3567bdbd628e88668c3c8d7e97d1d1253b6d4ea6d44c150f741f1bf4431.f938558b5d72f1a23810b4be2ab4f84331acc02fc97babc53a52ae8218a355a96d8770ac83d07bea87e13c512a"' \
    'opened = ""' 'failures = 1' 'replay with adapters disconnected gives the live state_hash'
}

pqxdh() {
  "$RCE" test "$here/pqxdh.rce" >/dev/null
  ev="Keys(ika: \"$(r32 11)\", eka: \"$(r32 22)\", ikb: \"$(r32 33)\", spkb: \"$(r32 44)\"), Pub(who: \"ika\"), Pub(who: \"eka\"), Pub(who: \"ikb\"), Pub(who: \"spkb\"), Kem(d: \"$(r32 01)\", z: \"$(r32 02)\"), DhI, Encaps(m: \"$(r32 03)\"), SkI, DhR, Decaps, SkR"
  "$RCE" host run "$here/pqxdh.rce" --log "$tmp/pq" "$ev" > "$tmp/pq.out" 2>&1
  "$RCE" host replay "$here/pqxdh.rce" --log "$tmp/pq" --redact priv,ek,dk,ct,dh_i,dh_r > "$tmp/pq.rep" 2>&1
  ski=$(sed -n 's/.*sk_i = "\([0-9a-f]*\)".*/\1/p' "$tmp/pq.out")
  # ML-KEM-768 shared secret for d = 01..01, z = 02..02, m = 03..03 (RustCrypto ml-kem 0.2.3)
  want pqxdh "$tmp/pq.out" 'ss_i = "f39b95557ee52af1954cd59f19febcb39f127e4abefc6f90546e7b8139ce94ba"' \
    "ss_r = \"f39b95557ee52af1954cd59f19febcb39f127e4abefc6f90546e7b8139ce94ba\"" "sk_r = \"$ski\"" 'replay with adapters disconnected'
  want pqxdh-replay-redacted "$tmp/pq.rep" 'replay ok' 'priv = <redacted sha256:' 'dk = <redacted sha256:'
  [ ${#ski} -eq 64 ] || { echo "FAIL pqxdh: session key length"; fail=1; }
}

mls() {
  "$RCE" test "$here/mls_key_schedule.rce" > "$tmp/mls.out" 2>&1
  want mls "$tmp/mls.out" 'ok ' '3 examples'
}

sfu() {
  # apps/sfu.rce offline: the wire inputs a P4 channel would deliver are given as ChanRecvDone events
  a='ChanRecvDone(req: 1, status: "delivered", data: some(["a\t1\tsub\u{1f}r\u{1f}video"]), err: "")'
  b='ChanRecvDone(req: 2, status: "delivered", data: some(["b\t1\tsub\u{1f}r\u{1f}video"]), err: "")'
  f2='ChanRecvDone(req: 3, status: "delivered", data: some(["c\t1\tframe\u{1f}r\u{1f}video\u{1f}2\u{1f}d2\u{1f}e2ee:ab2"]), err: "")'
  f1='ChanRecvDone(req: 4, status: "delivered", data: some(["c\t2\tframe\u{1f}r\u{1f}video\u{1f}1\u{1f}d1\u{1f}e2ee:old"]), err: "")'
  bad='ChanRecvDone(req: 5, status: "delivered", data: some(["c\t3\tframe\u{1f}r\u{1f}video\u{1f}x\u{1f}d3\u{1f}"]), err: "")'
  "$RCE" host run "$SFU" --log "$tmp/sfu" "Setup(me: \"sfu\"), $a, $b, $f2, $f1, $bad" > "$tmp/sfu.out" 2>&1
  "$RCE" host replay "$SFU" --log "$tmp/sfu" --redact digests > "$tmp/sfu.rep" 2>&1
  # frame 2 goes to a and b; frame 1 is stale; a non-numeric SEQ is malformed; no payload is kept
  want sfu "$tmp/sfu.out" 'forwarded = 2' 'stale = 1' 'malformed = 1' 'replay with adapters disconnected'
  if grep -q 'e2ee:' "$tmp/sfu.rep"; then echo "FAIL sfu: payload in state"; fail=1; fi
  want sfu-replay-redacted "$tmp/sfu.rep" 'replay ok' 'digests = <redacted sha256:'
}

rcedb() {
  D="$tmp/db"
  "$RCE" db "$D" schema users name:text,age:i64 >/dev/null
  "$RCE" db "$D" index users age >/dev/null
  "$RCE" db "$D" insert users ann "$(printf 'Ann\t41')" >/dev/null
  n=$("$RCE" db "$D" insert users bob "$(printf 'Bob\t29')" | sed -n 's/^ok seq //p')
  "$RCE" db "$D" link ann knows bob >/dev/null
  "$RCE" db "$D" delete users bob >/dev/null
  { echo "-- scan"; "$RCE" db "$D" scan users
    echo "-- as-of"; "$RCE" db "$D" as-of "$((n + 1))" scan users
    echo "-- lookup"; "$RCE" db "$D" lookup users age 41
    echo "-- edges"; "$RCE" db "$D" edges ann
    echo "-- describe"; "$RCE" db "$D" describe
    echo "-- sqlite"; "$RCE" db "$D" emit sqlite; } > "$tmp/db.out" 2>&1
  want rcedb "$tmp/db.out" 'ann' 'bob' 'knows' 'users'
  # rows are tab-separated cells in schema order; bob is deleted now but present as of the entries up to its insert (time travel, HS-9)
  if sed -n '/^-- scan/,/^-- as-of/p' "$tmp/db.out" | grep -q '^bob'; then echo "FAIL rcedb: bob after delete"; fail=1; fi
  sed -n '/^-- as-of/,/^-- lookup/p' "$tmp/db.out" | grep -q '^bob' || { echo "FAIL rcedb: as-of"; fail=1; }
}

case "${1:-all}" in
  all) hpke; pqxdh; mls; sfu; rcedb ;;
  *) "$1" ;;
esac
[ $fail -eq 0 ] && echo "all ecosystem examples ok" || { cat "$tmp"/*.out >&2; exit 1; }
