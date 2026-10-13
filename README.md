# rce-ide: the RCE IDE agent layer (steps 1 to 3)

Rust only, toolchain 1.98.1. No model and no network in any crate here.

- `crates/ide-receipt`: one receipt per action (action kind, gear, model, tokens in/out,
  E_sim, E_meas with source tag `rapl|nvml|powermetrics|jetson|estimated-remote`, delta,
  checker verdict, policy hash), BLAKE3 hash chain, ed25519 signature. JSONL ledger,
  verified in full on open. Mappers from `mol.receipt.v1` and mei-studio `mei-receipt` JSON.
- `crates/ide-energy`: `Simulator` trait and a linear simulator with per-gear/model
  coefficients fitted by least squares; `Meter` trait with Linux RAPL (powercap
  `energy_uj`, wraparound handled), NVML behind `--features nvml`, and macOS / Jetson
  backends that return `Unsupported`; time-window attribution; k-sigma anomaly test.
- `crates/rce-mcp`: stdio MCP server (JSON-RPC 2.0, newline-delimited). Tools
  `read_form`, `propose_edit`, `rce_check`, `rce_test`, `emit`. Uses the RCE reference
  checker (`rce-spec-check`, path dependency on `vendor/rce-spec/checker` (vendored RCE reference checker)).

Honesty rules: an uncalibrated simulator predicts `None`; without a readable meter
`e_meas` is `None`; `delta_j` exists only for a hardware measurement and a prediction.

Run: `cargo run -p rce-mcp -- --root DIR [--ledger FILE] [--key FILE] [--calibration FILE]`.

## Install (prebuilt binaries)

Release archives contain four programs: `rce-ide` (the desktop IDE, wgpu on
Vulkan, Metal, DX12 or GL), `rce` (the RCE checker CLI with the
`rce test --energy-budget J` gate), `rce-lsp` (language server) and `rce-mcp`
(MCP server for agents).

1. Download the archive for your platform and `SHA256SUMS` from the release page.
2. Verify: `sha256sum -c SHA256SUMS --ignore-missing` (Linux), or
   `Get-FileHash rce-ide-*.zip -Algorithm SHA256` (Windows PowerShell) and compare.
3. Unpack and put the binaries on your `PATH`:

   ```sh
   tar xzf rce-ide-0.1.0-x86_64-unknown-linux-gnu.tar.gz
   cd rce-ide-0.1.0-x86_64-unknown-linux-gnu
   install -m 755 rce rce-ide rce-lsp rce-mcp ~/.local/bin/
   ```

4. Open a folder of `.rce` files: `rce-ide --root DIR --file main.rce [--history 50]`.

Platforms and requirements:

| Archive | Requirement |
|---|---|
| `x86_64-unknown-linux-gnu` | glibc 2.28 or newer; a Vulkan or GL driver (Mesa works, including llvmpipe); X11 or Wayland |
| `aarch64-unknown-linux-gnu` | as above, on 64-bit Arm |
| `x86_64-pc-windows-gnu` | Windows 10 or newer, DX12 or Vulkan |
| `aarch64-apple-darwin` | macOS on Apple silicon; Metal. Ad-hoc signed, not notarized: clear quarantine with `xattr -d com.apple.quarantine rce*` or allow in Privacy and Security |

Notes:

- `rce-mcp --http` jails itself after binding (Landlock + seccomp; loopback or
  `--allow-host` only; outbound only to ports declared in `.rce-ide/net.json`).
  The jail exists on Linux only (kernel 6.7+ for Landlock ABI 4). On Windows and
  macOS HTTP mode refuses to start unless you pass `--no-jail`; stdio mode is
  unaffected.
- `rce-ide telemetry probe` lists the energy meters and the GPU launch source
  found on this machine. Without a readable meter, energy is reported as
  simulated or missing, never guessed.
- Build from source: Rust 1.98.1, `cargo build --release -p rce-ide -p rce-mcp`.
  The checker is a path dependency on `vendor/rce-spec/checker` (vendored RCE reference checker).

Graph view keys (pointer over the graph): F3 view, F4 layout, F6/F7/F8 colour,
size and glow metric (or click the chips), Esc folds expanded files, Shift-drag
a definition onto another file to move it, Delete removes the selected
definition, N adds a function. Every graph edit is written only if `rce check`
admits the program.

## License

Copyright (c) 2026 OpenIE. Licensed under either of Apache License 2.0
(LICENSE-APACHE) or MIT (LICENSE-MIT), at your option. The vendored photon
library is MPL-2.0 (source shipped as `photon-source-mpl-2.0.tar.gz`), and
webpki-roots is CDLA-Permissive-2.0; see release/LICENSE-STATUS.md.
