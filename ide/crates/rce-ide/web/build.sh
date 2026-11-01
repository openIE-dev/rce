#!/bin/sh
# Build the browser IDE into crates/rce-ide/web/dist and (optionally) serve it.
#   web/build.sh [serve [PORT]]
# Needs: rustup target wasm32-unknown-unknown, wasm-bindgen-cli matching the
# crate's wasm-bindgen (0.2.129). The model comes from the native IDE's
# the words demo and export-model over a scratch copy of the counter example.
set -eu
# positional (spec/proposals/positional-cli.md PC-1): the one word is `serve`
case "${1:-}" in
  ""|serve) ;;
  --serve) echo "--serve was removed; write: web/$(basename "$0") serve ${2:-}" >&2; exit 2 ;;
  *) echo "usage: web/$(basename "$0") [serve [PORT]]" >&2; exit 2 ;;
esac
here=$(cd "$(dirname "$0")" && pwd)
ws=$(cd "$here/../../.." && pwd)
dist="$here/dist"
tgt="${CARGO_TARGET_DIR:-$ws/target}"
mkdir -p "$dist"
cd "$ws"
cargo build --release -p rce-ide --lib --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --out-dir "$dist/pkg" \
  "$tgt/wasm32-unknown-unknown/release/rce_ide.wasm"
cargo build --release -p rce-ide --bin rce-ide
demo=$(mktemp -d)
cp "${RCE_DEMO_FILE:-$ws/../spec/examples/01_counter.rce}" "$demo/counter.rce"
(cd "$demo" && "$tgt/release/rce-ide" demo Counter export-model "$dist/model.json" . counter.rce)
cp "${RCE_FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf}" "$dist/font.ttf"
cp "$here/index.html" "$dist/index.html"
ls -la "$dist" "$dist/pkg"
if [ "${1:-}" = "serve" ]; then
  cd "$dist" && exec python3 -m http.server "${2:-8099}" --bind 127.0.0.1
fi
