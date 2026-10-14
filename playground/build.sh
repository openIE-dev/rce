#!/bin/sh
# Assemble the playground site in playground/_site from the committed
# checker-wasm build and the spec example corpus. Publishes nothing.
set -eu
cd "$(dirname "$0")/.."
OUT=playground/_site
rm -rf "$OUT"; mkdir -p "$OUT/pkg"
cp playground/index.html playground/app.js playground/style.css "$OUT/"
cp spec/checker-wasm/pkg/rce_spec_check_wasm.js spec/checker-wasm/pkg/rce_spec_check_wasm_bg.wasm "$OUT/pkg/"
python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
d = "spec/examples"
ex = [{"name": f[:-4], "src": open(os.path.join(d, f)).read()}
      for f in sorted(os.listdir(d)) if f.endswith(".rce") and f != "26_use_module.rce"]
json.dump(ex, open(os.path.join(out, "examples.json"), "w"))
print(len(ex), "examples")
PY
echo "checker $(git rev-parse --short HEAD), specification draft 0.7" > "$OUT/version.txt"
