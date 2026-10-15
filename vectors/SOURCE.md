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

## fmt (`lib/fmt/fmt.rce`)

No official vector files. All checks are derived, with CPython 3 as the
oracle (`scripts/gen/fmt_gen.py`, seed 20261015):
- 46 integers (both i64 extremes, random values) in radixes 2..36 and
  back; 7 strict parses and 15 refusals (empty, bare sign, whitespace,
  "0x", "_", overflow by one in either direction, bad digits);
- `format(v, ',')` and `format(v, '0Nd')` for grouping and zero padding;
- 8 templates against `str.format` and 6 that it refuses;
- a 34-pattern x 41-input grid (1394 checks), full and partial matching
  against `re` with `re.ASCII` (the library's `$` is checked as `\Z`), and
  16 malformed patterns;
- linear-time cases (`(a?){25}a{25}` on a^25, `(x+x+)+y` on x^40) that
  are exponential for backtracking engines.
`scripts/mut/fmt.py`: each of 25 mutants of the module fails the checks.

Left for M17 part 2: parser combinators (function values exist, e.g.
`fn(f: fn(i64) -> i64, ...)` and `fn(x) => ...` admit under rce 0.1.3,
so this needs design, not spec; an earlier note here said otherwise) and
regex captures (a match result type). Regex classes are ASCII ranges until a
pinned UCD (M9) gives Unicode classes.

## energy (`lib/energy/energy.rce`)

- JCP v0.3 draft (the JCP specification text, section 8 "The receipt",
  sections 3 and 8.1-8.2; read 2026-10-15).
- JCP conformance/v1, official: `vectors/energy/jcp-conformance-v1/`
  (`vectors.json`, `legacy_jcp1.json`, `dual_ceilings.json`, `meta.json`),
  copied from the JCP reference repository at commit
  72aec523d2dc9a6c93ea6838773a322a2e378860 (jcp.transaction.science is
  OpenIE's own; taken from a local working tree, not a public
  repository). sha256 prefixes: vectors.json 1e82c7ea88cd4ce5,
  legacy_jcp1.json 00a46131f2d248b7, dual_ceilings.json 8ec04771502d8956,
  meta.json 01ba3099d2ec373a. Seven receipts (two jcp/2, two jcp/1,
  three dual-ceiling incl. funds-exceeded and currency-mismatch denies):
  each envelope opens, its payload is the published signed core, its
  fields equal the published wire values, and each jcp/2 core re-mints
  to the same bytes (7 comptime checks); every envelope verifies under
  meta.json `pubkey_hex` and a one-bit flip does not (8 host checks).
  They fix the COSE layout: protected header {1: -8, 4: kid} with the kid
  as the steward id's UTF-8 bytes, empty unprotected map; a steward need
  not be a did:key (`verify_with` takes the key). `action_receipt.json`
  (ar/1-draft) and `standing.json` are not used yet.
- draft-msporny-base58-03 section 5: the three base58btc examples
  ("Hello World!", the quick brown fox, 0x0000287fb4cd), official.
- Derived with `scripts/gen/energy_gen.py` (cbor2, blake3 and
  cryptography in Python; RFC 8032 test 1 and 2 seeds): did:key ids of
  both keys; six jcp/2 receipts (allow, deny, Partial coverage, budget
  deny, two chained debits) as canonical cores, blake3 ids,
  Sig_structures and COSE_Sign1 envelopes, decoded back; replay and
  per-(grant, subject) tallies of that stream; a jcp/1 core that opens;
  18 malformed envelopes refused (missing or extra payload_hash, unknown
  profile or field, negative or text cost, Whole written out, allow with
  a reason, deny without, no steward, bad trust, non-canonical head, wrong
  kid, tag 98, alg -7, kid label 5, kid unprotected, unprotected not
  empty). 55 comptime checks in all.
- Host checks (`tests/energy/energy_host.rce`, one Ed25519 operation per
  input under `rce host run`): seal matches cryptography's bytes; two
  envelopes verify; a payload substitution with the old signature fails
  (JCP C23); a receipt signed by another key fails (C5); the 8
  conformance checks above; sealing with a seed that is not the
  steward's is refused. 14 of 14.
- `scripts/mut/energy.py`: 28 of 28 mutants killed.

Needs spec or data: whether `chain_ids` belongs in the signed jcp/2 core
(taken as yes, when the chain has more than one link; no conformance
vector has a chain); and the shape of `host.Energy`
results (E.5), so receipts can be built from them directly (here
cost_uj and its provenance are parameters). Module imports do not rename
names yet (W-MOD-3), so `enc` and `hash.blake3` private helpers were
renamed (`enc_cat`, `b3_g`) to coexist with `crypto`.

## io (`lib/io/io.rce`)

- No official vectors exist for a reader/writer library over RCE host
  adapters. The reference is the host itself: RCE spec Appendix E (HC-3:
  one result per request, delivered/failed/timed_out/unreachable; HC-10)
  and the rce 0.1.3 file adapter (`rce host run --fs`), whose path rule
  (relative, not empty except for FileList, normal names and "." only)
  `path_ok` mirrors.
- Derived with `scripts/gen/io_gen.py` (a Python model of each part,
  seed 18): status names and result shapes (12 cases, each also through
  `delivered`); 30 request-table scripts (ids, tags, outstanding,
  settled, stray: unknown ids, second results, wrong kinds); 51 line
  reader runs (CRLF, lone CR, multibyte text cut anywhere, with and
  without eof) plus reads after eof; 40 writer runs (partial acks on
  character boundaries, failed writes resent, max 0, a character longer
  than max) plus refused acks (beyond the chunk, inside a character,
  nothing in flight); 40 path cases. 195 comptime checks.
- Host checks (`tests/io/io_host.rce`, flattened with the module, run
  under `rce host run --fs` on an empty root): 9 requests (writes, reads
  back through the line reader, a non-ASCII name, `..`, an absolute path,
  `a/../a.txt`, listings); each result pairs with its request, has the
  predicted outcome and the expected data, and the host's replay with
  adapters disconnected re-derives the state. 9 of 9. `scripts/check.sh`
  now reads `# event:` lines (events with arguments) and `# host-flags:
  --fs`.
- `scripts/mut/io.py`: 29 of 29 mutants killed (three equivalent ones
  were removed by simplifying the code).

Needs spec: a generic function cannot take a borrowed option (`&T?`
does not unify with `&text?` in rce 0.1.3), so `result_ok` and
`delivered` take the data by value; a library cannot declare the
request structs and `<Kind>Done` events (W-HOST-1 looks for them in the
program), so each program repeats them; the file adapter reports a
missing file and a refused path both as `failed` with free text, so a
program cannot tell them apart without parsing `err`; no adapter yet
reads a file in chunks, so the line reader is exercised over one
FileRead and over arbitrary chunkings in the comptime checks.
