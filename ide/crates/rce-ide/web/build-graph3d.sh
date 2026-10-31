#!/bin/sh
# Build the browser 3D graph view into crates/rce-ide/web/dist (graph3d.html).
# The model comes from the native shots example (real demo workspace,
# SYNTHETIC energy marked).   web/build-graph3d.sh [serve [PORT]]
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
wasm-bindgen --target web --no-typescript --out-dir "$dist/pkg" "$tgt/wasm32-unknown-unknown/release/rce_ide.wasm"
cargo run --release -p rce-ide --example graph3d_shots -- "$dist/shots"
cp "$dist/shots/graph3d-model.json" "$dist/graph3d-model.json"
cp "${RCE_FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf}" "$dist/font.ttf"
cp "$here/graph3d.html" "$dist/graph3d.html"
if [ "${1:-}" = "serve" ]; then
  cd "$dist" && exec python3 -m http.server "${2:-8099}" --bind 127.0.0.1
fi
