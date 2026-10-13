#!/bin/sh
# Package release archives from already-built targets. Publishes nothing.
set -eu
cd "$(dirname "$0")/.."
V=0.1.1
OUT=release/dist
rm -rf "$OUT"; mkdir -p "$OUT"
python3 release/third_party.py > release/THIRD-PARTY-LICENSES.txt
for T in x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu aarch64-apple-darwin x86_64-pc-windows-gnu; do
  B=target/$T/release
  E=""; case $T in *windows*) E=.exe;; esac
  [ -f "$B/rce-ide$E" ] || { echo "skip $T (not built)"; continue; }
  N=rce-ide-$V-$T
  D=$OUT/$N; mkdir -p "$D"
  for b in rce rce-ide rce-lsp rce-mcp; do cp "$B/$b$E" "$D/"; done
  cp README.md LICENSE-MIT LICENSE-APACHE COPYRIGHT release/LICENSE-STATUS.md release/THIRD-PARTY-LICENSES.txt \
     release/NOTICE-webpki-roots-CDLA-Permissive-2.0.txt release/PHOTON-SOURCE-OFFER.md "$D/"
  case $T in
    *windows*) (cd "$OUT" && python3 -c "import shutil,sys; shutil.make_archive(sys.argv[1],'zip','.',sys.argv[1])" "$N");;
    *) tar -C "$OUT" -czf "$OUT/$N.tar.gz" "$N";;
  esac
  rm -rf "$D"
done
# photon MPL-2.0 source, as its own archive.
tar -C vendor -czf "$OUT/photon-source-mpl-2.0.tar.gz" photon
(cd "$OUT" && sha256sum *.tar.gz *.zip > SHA256SUMS)
cat "$OUT/SHA256SUMS"
