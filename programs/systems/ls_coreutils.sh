#!/usr/bin/env bash
# Differential test of ls.rce against GNU coreutils ls, on Linux.
#
#   RCE=/path/to/rce programs/systems/ls_coreutils.sh WORKDIR
#
# Makes WORKDIR/d (removed first) with regular files, a dotfile, a hard
# link, a set-user-ID file, a named pipe, a subdirectory, a sticky directory, a
# symbolic link and a dangling one, modes set with chmod and times pinned
# with touch, all in UTC. Then:
#  1. the short listing, with and without `all`, through the host's FileList
#     under `fs WORKDIR/d`, against `ls -1` and `ls -1a`;
#  2. the long listing: GNU stat and readlink give each entry's record, the
#     program's pure core makes the rows, and its Display text is compared
#     byte for byte with `ls -la` and with `ls -l`.
# The expected bytes come from coreutils, never from the program.
set -u
RCE="${RCE:-rce}"
case $# in
  1) W=$1 ;;
  *) echo "usage: programs/systems/ls_coreutils.sh WORKDIR" >&2; exit 2 ;;
esac
P="$(cd "$(dirname "$0")" && pwd)/ls.rce"
export LC_ALL=C TZ=UTC
mkdir -p "$W"
W="$(cd "$W" && pwd)"
D="$W/d"
rm -rf "$D" "$W"/log-*
mkdir "$D"
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
  touch -d '2026-10-15 12:34:00' a.txt pipe .dot
  touch -d '2026-10-20 08:05:00' big.bin
  touch -d '2020-01-02 03:04:05' run.sh
  touch --no-dereference -d '2026-10-16 09:00:00' link dangling
  touch -d '2026-10-17 23:59:00' sub/x sub/y sub tmpd
  touch -d '2026-10-18 00:00:00' .
) || { echo "could not make $D" >&2; exit 2; }

# the text of every `emit Display(data: "...")` line of an rce run
display() { python3 -I -c '
import sys, re, json
for l in sys.stdin:
    m = re.match(r"\s*emit Display\(data: (\".*\")\)$", l)
    if m:
        sys.stdout.write(json.loads(m.group(1)))'; }

# a Stats event for the directory: one Entry per name, from stat and readlink
stats() {
  local all=$1 now=$2 es=() n mode nl u g sz mt bl t
  for n in . .. $(cd "$D" && ls -A); do
    IFS='|' read -r mode nl u g sz mt bl < <(cd "$D" && stat -c '%f|%h|%U|%G|%s|%Y|%b' -- "$n")
    t=""; [ -L "$D/$n" ] && t=$(readlink -- "$D/$n")
    es+=("Entry(name: \"$n\", mode: $((16#$mode)), nlink: $nl, owner: \"$u\", group_name: \"$g\", size: $sz, mtime: $mt, target: \"$t\", blocks: $bl)")
  done
  local j; j=$(IFS='|'; echo "${es[*]}"); j=${j//|/, }
  echo "Stats(all: $all, now: $now, offset_min: 0, entries: [$j])"
}

fail=0
same() {
  if [ "$2" = "$3" ]; then echo "MATCH $1 ($(printf '%s\n' "$2" | wc -l) lines)"
  else echo "DIFFER $1"; diff <(printf '%s\n' "$2") <(printf '%s\n' "$3"); fail=1; fi
}

for all in false true; do
  got=$(echo "Ls(path: \"\", all: $all)" | "$RCE" host run fs "$D" "$P" "$W/log-$all" 2>/dev/null | display)
  if [ $all = true ]; then want=$(ls -1a "$D"); else want=$(ls -1 "$D"); fi
  same "short listing, all=$all" "$want" "$got"
done
now=$(date +%s)
for all in true false; do
  got=$("$RCE" run "$P" "$(stats $all "$now")" | display)
  if [ $all = true ]; then want=$(ls -la "$D"); else want=$(ls -l "$D"); fi
  same "long listing, all=$all" "$want" "$got"
done
[ $fail -eq 0 ]
