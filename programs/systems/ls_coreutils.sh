#!/usr/bin/env bash
# Differential test of ls.rce on the RCE host against GNU coreutils ls, on Linux.
#
#   RCE=/path/to/rce programs/systems/ls_coreutils.sh WORKDIR
#
# Makes WORKDIR/t/d (WORKDIR/t removed first) with regular files, a
# dotfile, a hard link, a set-user-ID file, a named pipe, a subdirectory, a
# sticky directory, a symbolic link, a dangling one and one whose target
# leaves the granted root, two names that differ only in bytes that are not
# UTF-8, and (when run as root) a file whose owner and group have no name;
# modes set with chmod and times pinned with touch in a zone with summer
# time. Then the program runs on the host,
#   rce host run fs WORKDIR/t names ls.rce D
# once for each listing of `d` (short and long, with and without all): every
# name, record, link text, owner and group name, clock reading and UTC
# offset comes from a host request (Appendix E, HC-30 to HC-33). The bytes
# of its Display output are compared with those of coreutils `ls` in the C
# locale, byte for byte. The expected bytes come from coreutils, never from
# the program.
set -u
RCE="${RCE:-rce}"
case $# in
  1) W=$1 ;;
  *) echo "usage: programs/systems/ls_coreutils.sh WORKDIR" >&2; exit 2 ;;
esac
P="$(cd "$(dirname "$0")" && pwd)/ls.rce"
# the C locale, and a zone whose offset differs between the files' times
export LC_ALL=C TZ='EST5EDT,M3.2.0,M11.1.0'
mkdir -p "$W"
W="$(cd "$W" && pwd)"
T="$W/t"
D="$T/d"
rm -rf "$T" "$W"/log-*
mkdir -p "$D"
(
  set -e
  cd "$D"
  printf 'hello' > a.txt; chmod 644 a.txt
  head -c 123456 /dev/zero > big.bin; chmod 600 big.bin
  : > .dot; chmod 664 .dot
  printf '#!/bin/sh\n' > run.sh; chmod 4755 run.sh
  mkdir sub sub/x sub/y; chmod 755 sub
  mkdir tmpd; chmod 1777 tmpd
  mkfifo pipe; chmod 640 pipe
  ln a.txt hard
  ln -s a.txt link
  ln -s sub/missing dangling
  ln -s /etc/hostname outside
  : > "$(printf 'x\376')"; : > "$(printf 'x\377')"; : > "$(printf 'caf\351')"
  if [ "$(id -u)" = 0 ]; then : > anon; chown 4242:4243 anon; fi
  touch -d '2026-10-15 12:34:00' a.txt pipe .dot "$(printf 'x\376')" "$(printf 'x\377')" "$(printf 'caf\351')"
  [ -e anon ] && touch -d '2026-10-15 12:34:00' anon
  touch -d '2026-10-20 08:05:00' big.bin
  touch -d '2020-01-02 03:04:05' run.sh
  touch --no-dereference -d '2026-10-16 09:00:00' link dangling outside
  touch -d '2026-10-17 23:59:00' sub/x sub/y sub tmpd
  touch -d '2026-10-18 00:00:00' .
  touch -d '2026-10-01 00:00:00' "$T"
) || { echo "could not make $D" >&2; exit 2; }

# the bytes of every `emit Display(data: ...)` line of a host run: a list
# of bytes (HC-30) or a text
display() { python3 -I -c '
import sys, re, json
out = sys.stdout.buffer
for l in sys.stdin:
    m = re.match(r"\s*emit Display\(data: (.*)\)$", l)
    if m:
        v = json.loads(m.group(1))
        out.write(bytes(v) if isinstance(v, list) else v.encode())'; }

fail=0
same() {
  if cmp -s "$2" "$3"; then echo "MATCH $1 ($(wc -l < "$2") lines)"
  else echo "DIFFER $1"; diff "$2" "$3"; fail=1; fi
}

for long in false true; do
  for all in false true; do
    k="$long-$all"
    echo "Ls(path: \"d\", all: $all, long: $long)" | "$RCE" host run fs "$T" names "$P" "$W/log-$k" 2>"$W/err-$k" | display > "$W/got-$k"
    grep -q validated "$W/err-$k" || { echo "the host run did not validate:"; cat "$W/err-$k"; fail=1; }
    case $k in
      false-false) ls "$D" ;; false-true) ls -a "$D" ;;
      true-false) ls -l "$D" ;; true-true) ls -la "$D" ;;
    esac > "$W/want-$k"
    flags=$(case $k in false-false) echo "" ;; false-true) echo " -a" ;; true-false) echo " -l" ;; true-true) echo " -la" ;; esac)
    same "ls$flags" "$W/want-$k" "$W/got-$k"
  done
done
# the log replays with every adapter disconnected (HC-8)
"$RCE" host replay "$P" "$W/log-true-true" | sed -n 1p
[ $fail -eq 0 ]
