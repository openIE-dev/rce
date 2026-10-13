# RCE IDE 0.1.0

A desktop IDE for RCE programs: text and a 3D Form graph as two views of one program, every agent action on a signed receipt, and energy shown as simulated or measured, never guessed.

Source: https://github.com/openIE-dev/rce-ide

## Downloads

| Platform | File | Size | SHA-256 |
|---|---|---|---|
| photon source (MPL-2.0) | `photon-source-mpl-2.0.tar.gz` | 0.1 MB | `363ddaf4d7eacfc8fb5b6fe82d775c02d68adc956045f5c96fb6420915146664` |
| macOS Apple silicon (ad-hoc signed, not notarized) | `rce-ide-0.1.0-aarch64-apple-darwin.tar.gz` | 9.3 MB | `1fbbcbd7e98d367e5c24aea5ba16fbfb9a436ba879a679e5febbfd0667043849` |
| Linux aarch64 (glibc 2.28+) | `rce-ide-0.1.0-aarch64-unknown-linux-gnu.tar.gz` | 10.0 MB | `0e3a2eafbe61ae423286e49ceefee8aa9a6dfe88b855b1d9991e2ea1bb38d453` |
| Linux x86_64 (glibc 2.28+) | `rce-ide-0.1.0-x86_64-unknown-linux-gnu.tar.gz` | 10.9 MB | `90384c02125ee46978c6ea86c219e7b3f863b0a7294b0677b7c74be5bd582523` |
| Windows x86_64 (untested build) | `rce-ide-0.1.0-x86_64-pc-windows-gnu.zip` | 37.6 MB | `6df64bd972a2f781c2d5b61c05af9133b0aee5bc6267271c42ee0406dfc1abb6` |

Each archive contains `rce-ide`, `rce`, `rce-lsp`, `rce-mcp`, the README, LICENSE-MIT, LICENSE-APACHE, COPYRIGHT, LICENSE-STATUS.md, THIRD-PARTY-LICENSES.txt, the webpki-roots CDLA notice and PHOTON-SOURCE-OFFER.md. Verify with `sha256sum -c SHA256SUMS --ignore-missing`. Downloads: https://github.com/openIE-dev/rce-ide/releases/tag/v0.1.0

macOS: built and smoke-tested on an Apple M5 Max (macOS 27), ad-hoc signed, not notarized. On first launch Gatekeeper may block it; run `xattr -d com.apple.quarantine rce rce-ide rce-lsp rce-mcp` in the unpacked folder, or allow it in System Settings, Privacy and Security.

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

rce-ide and the RCE checker: MIT OR Apache-2.0, Copyright (c) 2026 Open Interface Engineering, Inc. The vendored photon library is MPL-2.0; its full source is the `photon-source-mpl-2.0.tar.gz` download above and https://github.com/openIE-dev/rce-ide/tree/main/vendor/photon. webpki-roots is CDLA-Permissive-2.0 (notice included). Other dependencies are permissive; see THIRD-PARTY-LICENSES.txt.
