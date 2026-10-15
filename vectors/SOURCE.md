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

## text.utf8 (`lib/text/utf8.rce`)

- `text/UTF-8-test.txt`: Markus Kuhn's "UTF-8 decoder capability and
  stress test" (https://www.cl.cam.ac.uk/~mgk25/ucs/examples/UTF-8-test.txt,
  fetched 2026-10-15, sha256 b51cfe9a8d2689c90b10a13a3624092d546e0837c6ff835b6e5d713c5749c8c6),
  copied unchanged. All 83 lines holding a non-ASCII byte (68 ill-formed)
  and the whole file are checked: validity, first-error offset, strict
  decode, and lossy decode. Expected code points come from CPython 3
  `bytes.decode('utf-8', 'replace')`, which substitutes one U+FFFD per
  maximal subpart, the Unicode 3.9 recommended practice.
- Unicode 15.0 section 3.9, Table 3-8 (U+FFFD substitution example) and
  the Table 3-7 boundaries, as hand checks.

Needs spec for the rest of M9: building text from code points or UTF-8
bytes (no built-in in 5.15), and a pinned UCD (normalization data,
grapheme break and case-folding properties) as part of the content
address. Normalization (NFC/NFD over code-point lists) is feasible in pure
RCE once UCD tables are generated into the library, but its result could
not be turned back into text without that built-in.

## time (`lib/time/time.rce`)

- Howard Hinnant, "chrono-Compatible Low-Level Date Algorithms"
  (https://howardhinnant.github.io/date_algorithms.html): the algorithms
  days_from_civil, civil_from_days and weekday_from_days, his round-trip
  and consecutive-day test (run over windows at the 1970, 2000, -400 and
  2100 boundaries plus a 2001-point stride over about +-10^8 days), and
  his fixed points (1970-01-01 = 0 and Thursday; 0000-03-01 = -719468).
- RFC 3339 section 5.8: the four examples (the leap-second one is
  refused: instants are POSIX time), plus rejections of malformed text.
- ISO 8601 week dates: the year-boundary cases 2008-12-29 (2009-W01-1)
  and 2010-01-03 (2009-W53-7).
- 60 random dates in years 1..9999 and 20 random instants with offsets,
  checked against CPython `datetime` (seed 20261015,
  `scripts/gen/time_gen.py`); derived vectors, not official ones.

Needs spec or data: named time zones need the IANA tz database as a
pinned, content-addressed input (or a logged host result); only fixed UTC
offsets are supported. Leap seconds would need a pinned leap-second table.

## rand.det (`lib/rand/det.rce`)

- `rand/rfc8439.txt`: RFC 8439, "ChaCha20 and Poly1305 for IETF
  Protocols" (https://www.rfc-editor.org/rfc/rfc8439.txt, fetched
  2026-10-15, sha256 25bef70fbf7a07ff45c2fe4cb7c6ce954eac687413d8610603268b4e4415324c),
  copied unchanged. `scripts/gen/rand_gen.py` extracts the 11 ChaCha20
  vectors from it (2.1.1 quarter round, 2.3.2 block, 2.4.2 encryption,
  Appendix A.1 #1-#5 block, A.2 #1-#3 encryption), checks them against a
  pure-Python ChaCha20 and prints them as checks.
- Derived, not official: 30 generator vectors (word streams, bits, below,
  range, bytes, shuffle, split, counter carry past 2^32) from the Python
  model in the same script; distribution checks over fixed seeds
  (chi-square of below(10) x 5000 and a die x 3000 under the p = 0.001
  critical values, bit balance within 4 sigma over 6400 bits, range
  bounds and endpoints over 40 spans). `scripts/mut/rand_det.py` is a
  mutation test: each mutant of the module must fail the checks.

Needs spec: a `Cap` is an opaque token and cannot carry data, so the
seed bytes travel beside it in the event and nothing ties those bytes to
the host's Random adapter; `from_seed` (needed for replay and vectors)
accepts any bytes. A host-issued value type (bytes that only an adapter
can mint, e.g. `Guarded`-style provenance on a payload field) would close
this. `Guarded[τ, A]` cannot hold a generator, since every draw writes it
and so every drawing event would need the capability. Example syntax
writes a capability argument as `()`.
