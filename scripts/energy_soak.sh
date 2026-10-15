#!/usr/bin/env bash
# Energy-receipt soak = Track C. Runs the preregistered soak in
# bench/track-c.prereg.json (5 example specs x {mock, geared} x seed 42,
# 20 rounds, no API keys, no network) and keeps every artifact:
# persist dirs, energy CSVs, cassettes, handler library, logs,
# manifest.json (with the prereg SHA-256) and REPORT.md.
# measured_j stays empty: nothing here reads a power counter.
#
#   scripts/energy_soak.sh [outdir] [prereg]
set -euo pipefail
OUT=${1:-soak-energy}
PREREG=${2:-bench/track-c.prereg.json}
RCE=${RCE:-./target/debug/rce}
"$RCE" bench track-c "$PREREG" --out "$OUT"
