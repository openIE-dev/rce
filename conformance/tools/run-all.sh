#!/usr/bin/env bash
# Run the suite on every shipped adapter; results land in reports/.
set -u
cd "$(dirname "$0")/.."
for a in reference emit-python emit-js emit-rust emit-scala wasm-playground; do
  [ -f adapters/$a.json ] || continue
  j=8; [ $a = emit-scala ] && j=4
  python3 harness/run.py --adapter adapters/$a.json --jobs $j --quiet > reports/log-$a.txt 2>&1
  tail -1 reports/log-$a.txt
done
