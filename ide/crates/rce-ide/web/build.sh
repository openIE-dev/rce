#!/bin/sh
# Build the browser IDE into crates/rce-ide/web/dist and (optionally) serve it.
#   web/build.sh [--serve PORT]
# Needs: rustup target wasm32-unknown-unknown, wasm-bindgen-cli matching the
# crate's wasm-bindgen (0.2.129). The model comes from the native IDE's
# --demo/--export-model over a scratch copy of the counter example.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
ws=$(cd "$here/../../.." && pwd)
dist="$here/dist"
mkdir -p "$dist"
cd "$ws"
cargo build --release -p rce-ide --lib --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --out-dir "$dist/pkg" \
  target/wasm32-unknown-unknown/release/rce_ide.wasm
cargo build --release -p rce-ide --bin rce-ide
demo=$(mktemp -d)
cp "${RCE_DEMO_FILE:-$ws/vendor/rce-spec/examples/01_counter.rce}" "$demo/counter.rce"
(cd "$demo" && "$ws/target/release/rce-ide" --root . --file counter.rce --demo Counter --export-model "$dist/model.json")
cp "${RCE_FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf}" "$dist/font.ttf"
cp "$here/index.html" "$dist/index.html"
ls -la "$dist" "$dist/pkg"
if [ "${1:-}" = "--serve" ]; then
  cd "$dist" && exec python3 -m http.server "${2:-8099}" --bind 127.0.0.1
fi
