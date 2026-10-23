# RCE IDE 0.1.3

Binary-only release: `rce`, `rce-ide`, `rce-lsp` and `rce-mcp`, built from the private RCE engine. 0.1.2 and later ship binaries; the source in this repository is the 0.1.2 snapshot.

## What is new

- `rce db D CMD`: RCEdb, serverless and in-process (schema, index, insert, link, scan, lookup, as-of, emit sqlite|postgres|duckdb|cypher).
- `rce host run|serve --media DIR`: media capture and playback adapters; frames are blobs addressed by SHA-256.
- `rce host replay --redact F,...`: print secret fields as digests.
- Standard host libraries embedded in `rce`: a `#!lib NAME` line appends `crypto`, `pq` (ML-KEM-768, PQXDH), `hpke` (AES-128-GCM, RFC 9180) or `mls` (RFC 9420 suite 1). Source and test vectors: https://github.com/openIE-dev/rce-stdlib
- Runnable examples: https://github.com/openIE-dev/rce-examples/tree/main/ecosystem. Book: https://openie-dev.github.io/rce-book/

## Downloads

| Platform | File |
|---|---|
| macOS Apple silicon (ad-hoc signed, not notarized) | `rce-ide-0.1.3-aarch64-apple-darwin.tar.gz` |
| Linux x86_64 (glibc 2.39 or later) | `rce-ide-0.1.3-x86_64-unknown-linux-gnu.tar.gz` |
| photon source (MPL-2.0) | `photon-source-mpl-2.0.tar.gz` |

Linux arm64 and Windows are not built for 0.1.3; use 0.1.2 there.

## SHA-256

```
dee57c47e3f525dde678b2364ad576da60de43e949b7f6ecaa786c6604370c8c  photon-source-mpl-2.0.tar.gz
6b76de8e19a741c7edabb5f9ab630025e41531fa1d4d72f5f915e2696224310c  rce-ide-0.1.3-aarch64-apple-darwin.tar.gz
4c55b98b02a3cebd8f60444ba3fc73e1596379d3514a7952264e0df22d74dbed  rce-ide-0.1.3-x86_64-unknown-linux-gnu.tar.gz
```

Verify: `shasum -a 256 -c SHA256SUMS --ignore-missing`.
