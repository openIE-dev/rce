#!/bin/bash
# Installs, in a Claude Code cloud session, the tools the CI workflows use,
# so checks run in the session as they run in CI. Each step installs only
# what is missing, so a cached container skips it.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

root="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
have() { command -v "$1" >/dev/null 2>&1; }
bin=/usr/local/bin

# apt: zstd (Lean archive), cross compilers and qemu for the parity jobs
pkgs=()
have zstd || pkgs+=(zstd)
have s390x-linux-gnu-gcc || pkgs+=(gcc-s390x-linux-gnu libc6-dev-s390x-cross)
have qemu-s390x-static || pkgs+=(qemu-user-static)
have x86_64-w64-mingw32-gcc || pkgs+=(gcc-mingw-w64-x86-64)
if [ ${#pkgs[@]} -gt 0 ]; then
  apt-get update -qq
  apt-get install -y -qq "${pkgs[@]}"
  apt-get clean
fi

# Lean, the version of spec/lean/lean-toolchain, from its GitHub release
# (elan's download host is not reachable from cloud sessions)
lean_v=$(sed 's/.*:v//' "$root/spec/lean/lean-toolchain")
if ! lean --version 2>/dev/null | grep -q "version $lean_v,"; then
  mkdir -p /opt/lean
  curl -sSfL "https://github.com/leanprover/lean4/releases/download/v$lean_v/lean-$lean_v-linux.tar.zst" | zstd -dc | tar x -C /opt/lean
  ln -sf "/opt/lean/lean-$lean_v-linux/bin/lean" "/opt/lean/lean-$lean_v-linux/bin/lake" "$bin/"
fi

# scala-cli (the Scala emitter's oracle, spec 10.5)
if ! scala-cli version 2>/dev/null | grep -q '1.18.0'; then
  curl -sSfL https://github.com/VirtusLab/scala-cli/releases/download/v1.18.0/scala-cli-x86_64-pc-linux.gz | gunzip > "$bin/scala-cli"
  chmod +x "$bin/scala-cli"
fi

# wasmtime (wasm32-wasip1 parity)
if ! have wasmtime; then
  curl -sSfL https://github.com/bytecodealliance/wasmtime/releases/download/v26.0.1/wasmtime-v26.0.1-x86_64-linux.tar.xz | tar -xJ -C /opt
  ln -sf /opt/wasmtime-v26.0.1-x86_64-linux/wasmtime "$bin/"
fi

# mdbook (the book job)
if ! mdbook --version 2>/dev/null | grep -q '0.4.52'; then
  curl -sSfL https://github.com/rust-lang/mdBook/releases/download/v0.4.52/mdbook-v0.4.52-x86_64-unknown-linux-gnu.tar.gz | tar -xz -C "$bin"
fi

# zig (cross builds of the IDE release) and weasyprint (PDFs)
python3 -c 'import ziglang' 2>/dev/null || pip install -q --no-cache-dir ziglang
have weasyprint || pip install -q --no-cache-dir weasyprint

# Rust: the toolchain of spec/checker/rust-toolchain.toml and the parity targets
rust_v=$(sed -n 's/^channel = "\(.*\)"/\1/p' "$root/spec/checker/rust-toolchain.toml")
rustup toolchain list | grep -q "^$rust_v" || rustup toolchain install "$rust_v" --profile minimal -c clippy -c rustfmt
for t in wasm32-wasip1 s390x-unknown-linux-gnu x86_64-pc-windows-gnu; do
  rustup target list --installed --toolchain "$rust_v" | grep -qx "$t" || rustup target add --toolchain "$rust_v" "$t"
done
