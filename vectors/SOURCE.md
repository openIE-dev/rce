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
  output: 105 checks, sharded over `tests/hash/blake3_checks*.rce`; the three
  102400-byte cases run nightly (`tests/hash/blake3_nightly_*.rce`,
  `scripts/nightly.sh`, `.github/workflows/nightly.yml`).

## core.bytes (`lib/core/bytes.rce`)

No official vector set applies; `tests/core/bytes_checks.rce` holds hand
vectors and a seeded property sweep (200 cases). Needs spec: a built-in
to build `text` from UTF-8 bytes (none in 5.15), and a native byte type
(bytes stay `List[i64]` until one exists). Option and result types are
the language's own `T?` and error unions (`T ! E`), so no core.option or
core.result library is added.

## col (`lib/col/col.rce`)

FNV-1a 32 values for "", "a" and "foobar" are from the FNV authors' test
suite (http://www.isthe.com/chongo/tech/comp/fnv/). Deque, heap, ordered
map/set and fixed-hash map are checked by seeded model-based sweeps
against List and the built-in Map (`tests/col/col_checks.rce`).
Needs engine fix (rce 0.1.3): a generic struct's field types are not
resolved on direct access outside generic code (`m.keys` is "unknown type
K"), and a generic constructor with no arguments cannot infer its type;
col exposes accessor functions (`om_keys`, `hm_keys`, `dq_to_list`).

## enc (`lib/enc/enc.rce`)

Official vectors, transcribed as `check` lines in `tests/enc/enc_checks.rce`:
- RFC 4648 section 10: all seven base64 vectors (encode and strict decode).
- RFC 8949 Appendix A: the 43 items within the enc model (integers in
  i64, simple values, tags 0/1/23/24/32, byte and text strings, definite
  arrays and maps); encoding is checked for all 43, decoding round trip for
  the 39 ASCII-text ones. Not taken: floats, 2^64-range integers and
  bignums, and the indefinite-length items (non-deterministic by 4.2.1;
  the decoder rejects them). Also the 4.2.1 map key-order example.
- RFC 8785: the 3.2.2.2 string-escaping example and the 3.2.3 key-sorting
  example (UTF-16 code-unit order, including a supplementary character).
- Varint and zigzag: values from the protobuf encoding guide
  (https://protobuf.dev/programming-guides/encoding/) and the LEB128
  example 624485 -> e5 8e 26.

Needs spec: f64 bit access (CBOR floats and JCS/ECMAScript number
formatting), text from UTF-8 bytes (decoding non-ASCII CBOR text, JSON
\u escapes in a parser). CBOR integers beyond i64 need core.exact
integration (planned, not a spec gap).
