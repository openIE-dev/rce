#!/usr/bin/env bash
# Standing check: build the Lean proofs, check the stated obligations, run the
# in-process axiom audit, and replay the exported library through all 6 kernels
# (leanchecker-paranoid, lean4lean, nanoda, con-leche, con-ron, Lean default).
set -euo pipefail
cd "$(dirname "$0")/../spec/lean"
command -v lake >/dev/null || export PATH="$HOME/.elan/bin:$PATH"
command -v lake >/dev/null || { echo "lean audit: lake not found (install elan)"; exit 1; }
lake build
lake env lean obligations/Obligations.lean
out=$(lake env lean audit/Audit.lean 2>&1) || { echo "$out"; exit 1; }
echo "$out"
echo "$out" | grep -q "standard axioms only"
if [ "$(uname -s)" = "Linux" ] && command -v bwrap >/dev/null 2>&1 && bwrap --ro-bind / / true >/dev/null 2>&1; then
  echo "lean audit: lake check --paranoid (sandbox: Linux bwrap)"
  lake check --paranoid
else
  echo "lean audit: lake check --paranoid (sandbox: disabled via --inadvisably-no-sandbox; bwrap unavailable)"
  lake check --paranoid --inadvisably-no-sandbox
fi
