#!/bin/sh
# Build corpus evidence from the public Form bridge and seal it.
# Usage: scripts/acceptance/drive.sh OUT
set -eu
if [ "$#" -ne 1 ] || [ "${1#-}" != "$1" ]; then
  echo "usage: scripts/acceptance/drive.sh OUT" >&2
  exit 2
fi
root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
out=$1
case $out in
  /*) ;;
  *) out=$root/$out ;;
esac
tmp=$out.unsealed
cargo +1.98.1 run --release --manifest-path "$root/scripts/acceptance/drive/Cargo.toml" -- "$tmp"
python3 "$root/scripts/acceptance/accept.py" seal "$tmp" "$out"
rm -f "$tmp"
echo "acceptance-drive: sealed $out"
