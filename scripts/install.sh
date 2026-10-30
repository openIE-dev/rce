#!/bin/sh
# Install the rce toolchain from the release of the newest main commit that
# passed ci (the rolling release rce-latest), or of one commit.
#
#   curl -fsSL https://raw.githubusercontent.com/openIE-dev/rce/main/scripts/install.sh | sh
#   sh install.sh [COMMIT]          COMMIT: the first 12 hex digits, as in the release rce-<commit>
#
# RCE_INSTALL_DIR chooses the directory (default ~/.local/bin); RCE_RELEASE_BASE
# replaces the release download URL (a mirror, or a test). The archive is
# checked against the release's SHA256SUMS before anything is installed.
set -eu

repo="openIE-dev/rce"
tag="rce-${1:-latest}"
dir="${RCE_INSTALL_DIR:-$HOME/.local/bin}"

os=$(uname -s)
arch=$(uname -m)
case "$os/$arch" in
  Linux/x86_64) name=linux-x86_64 ;;
  Linux/aarch64 | Linux/arm64) name=linux-aarch64 ;;
  Darwin/arm64) name=macos-aarch64 ;;
  MINGW*/x86_64 | MSYS*/x86_64 | CYGWIN*/x86_64) name=windows-x86_64 ;;
  *)
    echo "rce install: no release build for $os/$arch; build from source with" >&2
    echo "  git clone https://github.com/$repo && cd rce/spec/checker && cargo build --release --bin rce" >&2
    exit 1 ;;
esac

fetch() {
  if command -v curl >/dev/null 2>&1; then curl -fsSL "$1" -o "$2"
  else wget -qO "$2" "$1"; fi
}
sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
base="${RCE_RELEASE_BASE:-https://github.com/$repo/releases/download/$tag}"
archive="rce-$name.tar.gz"
fetch "$base/$archive" "$tmp/$archive"
fetch "$base/SHA256SUMS" "$tmp/SHA256SUMS"
want=$(grep " $archive\$" "$tmp/SHA256SUMS" | cut -d' ' -f1)
got=$(sha256 "$tmp/$archive")
if [ -z "$want" ] || [ "$want" != "$got" ]; then
  echo "rce install: $archive does not match SHA256SUMS of $tag; nothing installed" >&2
  exit 1
fi
tar -xzf "$tmp/$archive" -C "$tmp"
mkdir -p "$dir"
exe=rce
[ "$name" = windows-x86_64 ] && exe=rce.exe
cp "$tmp/rce/$exe" "$dir/$exe"
chmod +x "$dir/$exe"
"$dir/$exe" --version
case ":$PATH:" in
  *":$dir:"*) ;;
  *) echo "add $dir to PATH to run rce" ;;
esac
