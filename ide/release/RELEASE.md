# RCE IDE 0.1.2

A desktop IDE for RCE programs: text and a 3D Form graph as two views of one program, every agent action on a signed receipt, and energy shown as simulated or measured, never guessed.

Source: https://github.com/openIE-dev/rce/tree/main/ide. Try the checker without installing anything in the [RCE playground](https://openie-dev.github.io/rce/).

## Downloads

| Platform | File |
|---|---|
| macOS Apple silicon (ad-hoc signed, not notarized) | `rce-ide-0.1.2-aarch64-apple-darwin.tar.gz` |
| Linux x86_64 (glibc 2.28+) | `rce-ide-0.1.2-x86_64-unknown-linux-gnu.tar.gz` |
| Linux aarch64 (glibc 2.28+) | `rce-ide-0.1.2-aarch64-unknown-linux-gnu.tar.gz` |
| Windows x86_64 (untested build) | `rce-ide-0.1.2-x86_64-pc-windows-gnu.zip` |
| photon source (MPL-2.0) | `photon-source-mpl-2.0.tar.gz` |

Each archive contains `rce-ide`, `rce`, `rce-lsp`, `rce-mcp`, the README and the license files. Verify with `sha256sum -c SHA256SUMS --ignore-missing`. This release was built by the release workflow of this repository from the tagged commit.

macOS: on first launch Gatekeeper may block the binaries; run `xattr -d com.apple.quarantine rce rce-ide rce-lsp rce-mcp` in the unpacked folder, or allow them in System Settings, Privacy and Security.

## What is new

- The checker tracks RCE specification draft 0.7 (openIE-dev/rce): language version 0.7 with size-scaled standard library method costs (C.6.4, C.15), the cost of a method call fixed by the version of the module where the call is written, and `rce trace --json` reference traces (10.9).
- `rce --version` prints the release, the specification draft and the accepted language versions (0.5, 0.6, 0.7).
- Release archives are now built in GitHub Actions; every push is tested by CI.

## Known limits

- The Windows binaries are cross-built and not run on Windows.
- The aarch64 Linux build is cross-built and not run on Arm hardware.
- Energy measurement needs a readable meter (RAPL on Linux; NVML behind a build feature). Desktop calibration has not been run on real hardware for this release.

## Licenses

rce-ide and the RCE checker: MIT OR Apache-2.0, Copyright (c) 2026 Open Interface Engineering, Inc. The vendored photon library is MPL-2.0; its full source is the `photon-source-mpl-2.0.tar.gz` download above and https://github.com/openIE-dev/rce/tree/main/ide/vendor/photon. webpki-roots is CDLA-Permissive-2.0 (notice included). Other dependencies are permissive; see THIRD-PARTY-LICENSES.txt.
