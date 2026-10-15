#!/usr/bin/env bash
# Standing check: build the Lean proofs and fail if any RCE theorem uses
# sorry or an axiom beyond propext, Quot.sound, Classical.choice.
set -euo pipefail
cd "$(dirname "$0")/../spec/lean"
command -v lake >/dev/null || export PATH="$HOME/.elan/bin:$PATH"
command -v lake >/dev/null || { echo "lean audit: lake not found (install elan)"; exit 1; }
lake build
out=$(lake env lean audit/Audit.lean 2>&1) || { echo "$out"; exit 1; }
echo "$out"
echo "$out" | grep -q "standard axioms only"
