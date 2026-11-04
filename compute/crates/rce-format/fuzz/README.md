# rce-format fuzz harness

libFuzzer targets for the `.fg` parser. The M0 acceptance criterion is
**24h with zero crashes** on `fg_file_read`.

## Targets

| Target | Surface | Why |
|--------|---------|-----|
| `fg_file_read` | `FgFile::read(Cursor<&[u8]>)` | The headline target: full `.fg` parse pipeline including header, TOC, sections, content-hash check. |
| `fg_header` | `FgHeader::decode(&[u8])` | 64-byte fixed header decode. Catches bad magic / version / hash-alg / TOC-pointer bytes. |
| `fg_toc` | `Toc::decode(&[u8])` | TOC decode with arbitrary count + entries. Catches off/length overflow and unknown section kinds. |
| `fg_weight_manifest` | `WeightManifest` CBOR + `validate_layout` | Catches bad CBOR + adversarial offset/length combos. |

## Corpus

`corpus/fg_file_read/` is seeded with real `.fg` files generated from the
CLI (`make-sample`, `make-sample-mlp`, `make-sample-attention`,
`make-sample-wasm`, `make-sample-wasm-tensor`, `fx-import`). libFuzzer
mutates these and persists interesting inputs back to the corpus
automatically.

## Run

Requires a nightly Rust toolchain (libFuzzer hooks rely on nightly's
sanitiser support).

```bash
# Smoke run (1 minute) — verifies the harness compiles and produces no
# immediate crashes. Run this in CI on every PR.
cargo +nightly fuzz run fg_file_read -- -max_total_time=60

# Long run (24h) — the M0 acceptance gate.
cargo +nightly fuzz run fg_file_read --release -- -max_total_time=86400 -workers=$(nproc) -jobs=$(nproc)

# Replay a crash artifact after a failure.
cargo +nightly fuzz run fg_file_read fuzz/artifacts/fg_file_read/crash-XXXX

# Run all targets one after another.
for t in fg_file_read fg_header fg_toc fg_weight_manifest; do
    cargo +nightly fuzz run "$t" -- -max_total_time=300 || break
done
```

## CI integration

A weekly cron job runs `fg_file_read` for 6 hours and surfaces any new
crashes. The 24h gate is run before each minor release.

## What we expect

Every input must produce one of:
- `Ok(FgFile)` — for the rare random hit that happens to round-trip
- `Err(FgError::*)` — typed error, well-formed message
- (silent return) — for intentionally over-large inputs the target rejects

**Not allowed:** panics, abort, segfault, OOM (excessive allocation),
stack overflow, infinite loop.

If libFuzzer flags any of those, the target dumps the offending bytes to
`fuzz/artifacts/<target>/<crash-hash>` and the run aborts. Reproduce
deterministically by feeding the artifact path back to the same target.
