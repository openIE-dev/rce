# RCE IDE 0.1.0 (DRAFT, not published)

A desktop IDE for RCE programs: text and a 3D Form graph as two views of one program, every agent action on a signed receipt, and energy shown as simulated or measured, never guessed.

## Downloads

| Platform | File | Size | SHA-256 |
|---|---|---|---|
| photon source (MPL-2.0) | `photon-source-mpl-2.0.tar.gz` | 0.1 MB | `5b55c4df444e7104d987a59f630e643905110190ee151be19b2cc0033e213778` |
| Linux aarch64 (glibc 2.28+) | `rce-ide-0.1.0-aarch64-unknown-linux-gnu.tar.gz` | 10.0 MB | `3713a2e5d4588d87884fb51bdccfcdb59f277080e84d6d6b1310b175569acd3c` |
| Linux x86_64 (glibc 2.28+) | `rce-ide-0.1.0-x86_64-unknown-linux-gnu.tar.gz` | 10.9 MB | `54be02680ecc0a5a59cb21354b2818fd8a3f0f069103c3285e7668815c2e12e7` |
| Windows x86_64 (untested build) | `rce-ide-0.1.0-x86_64-pc-windows-gnu.zip` | 37.6 MB | `5242cc978373877da48c89714cb74b08f0fd0465ff0d70a5e37f4f0e9d1963b8` |

Each archive contains `rce-ide`, `rce`, `rce-lsp`, `rce-mcp`, the README, LICENSE-MIT, LICENSE-APACHE, COPYRIGHT, LICENSE-STATUS.md, THIRD-PARTY-LICENSES.txt, the webpki-roots CDLA notice and PHOTON-SOURCE-OFFER.md. Verify with `sha256sum -c SHA256SUMS --ignore-missing`.

macOS: no binary in this release. It has to be built and signed on a Mac.

## What is new

- Graph edits: Shift-drag a definition onto another file to move it, Delete to remove, N to add a function. Each edit is written only if `rce check` admits the whole program.
- Review badges: green for a human review that still matches the code, orange once the code changed.
- Metric picker: chips (or F6/F7/F8) choose the metric for colour, size and glow.
- `--history N`: replay the last N commits that changed `.rce` files on the time scrubber.
- Semantic zoom: click a folded file to open it in place; Esc folds it again.
- GPU launch timing for the IDE's own wgpu work: GPU timestamp queries when the adapter has them, else host-timed, else none with the reason (`rce-ide telemetry probe`).
- `rce test --energy-budget J [--measured J | --measure] [--strict]`: fail a test run that exceeds an energy budget. Without a measurement the gate uses the static simulated cost and says so.
- `rce-mcp --http` jails itself on Linux (Landlock + seccomp): loopback or declared hosts only, outbound only to declared ports, files limited to the workspace. Other platforms refuse HTTP mode unless `--no-jail`.

## Known limits

- Windows binaries were cross-built and not run on Windows. They are larger than the Linux ones because the cross linker kept unused code.
- The aarch64 Linux build was cross-built and not run on Arm hardware.
- The Linux x86_64 build was smoke-tested on a machine with no physical GPU (software Vulkan, llvmpipe).
- Energy measurement needs a readable meter (RAPL on Linux; NVML behind a build feature). Desktop calibration has not been run on real hardware for this release.

## Licenses

rce-ide and the RCE checker: MIT OR Apache-2.0, Copyright (c) 2026 OpenIE. The vendored photon library is MPL-2.0; its full source is the `photon-source-mpl-2.0.tar.gz` download above. webpki-roots is CDLA-Permissive-2.0 (notice included). Other dependencies are permissive; see THIRD-PARTY-LICENSES.txt.
