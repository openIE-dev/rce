# Test vectors

Official test vectors, copied unchanged; each remains under the terms of
its upstream repository.

- `hpke-base-x25519-sha256-aes128gcm.json`: the entries of the RFC 9180
  test vectors (https://github.com/cfrg/draft-irtf-cfrg-hpke,
  `test-vectors.json`) for mode 0 (base), KEM 0x0020 (DHKEM(X25519,
  HKDF-SHA256)), KDF 0x0001 (HKDF-SHA256) and AEAD 0x0001 (AES-128-GCM).
- `mls/tree-math.json` and `mls/key-schedule.json`: RFC 9420 test vectors
  from https://github.com/mlswg/mls-implementations (`test-vectors/`,
  branch main, fetched 2026-10-15). `key-schedule.json` is checked for
  cipher suite 1 (MLS_128_DHKEMX25519_AES128GCM_SHA256_Ed25519).
- `mls/crypto-basics-suite1.json`: the cipher suite 1 entry of
  `test-vectors/crypto-basics.json` from the same repository.

ML-KEM-768 results are checked against the RustCrypto `ml-kem` crate
(version 0.2.3); the FIPS 197, SP 800-38D and FIPS 202 vectors are the
`check` lines inside the library files.

## core.exact (`lib/core/exact.rce`)

- `core/exact/dectest-selected.txt`: test cases selected, unchanged, from
  the General Decimal Arithmetic test suite by Mike Cowlishaw
  (https://speleotrove.com/decimal/dectest.zip, version 2.62):
  `add.decTest`, `subtract.decTest` and `multiply.decTest` cases with exact
  results (no conditions), and `quantize.decTest` cases whose only
  conditions are Inexact/Rounded, under each case's rounding directive.
  Cases with NaN, infinities, negative zero, clamping, or coefficients over
  60 digits are not selected (core.exact has no special values, no signed
  zero and unbounded precision). Each line became one `check` in
  `tests/core/exact_checks.rce` (generator: `scripts/gen/exact_gen.py`).
- Big integer and rational cross-checks (`big_mul`, floor and truncating
  division, `gcd`, `isqrt`, rational add/div) were computed with CPython
  3 `int`, `math` and `fractions` from seed 20261015; they are derived
  vectors, not official ones.

## hash.blake3 (`lib/hash/blake3.rce`)

- `hash/blake3/test_vectors.json`: the official BLAKE3 test vectors,
  `test_vectors/test_vectors.json` from https://github.com/BLAKE3-team/BLAKE3
  (commit f55849f89cd85083c1c9daa2c5de20766f309c00, fetched 2026-10-15),
  copied unchanged. All 35 input lengths (0 to 102400) are checked in all
  three modes (hash, keyed_hash, derive_key) at the full 131-byte extended
  output: 105 checks, sharded over `tests/hash/blake3_checks*.rce`.
