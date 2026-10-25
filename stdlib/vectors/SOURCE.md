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

## effect (`lib/effect/effect.rce`)

- RCE spec 5.2, the effect-context table and the call rules (task and
  foreign functions only from main and tasks; W-EFF-1, W-SPAWN-1), and
  5.11. `tests/effect/effect_probes.py` writes one probe program per
  table cell (6 contexts x their effects, 43 cells; comptime `fail` is a
  compile error and not probed) and per call pair (10), runs `rce check`
  on each, and requires admission where the table allows and refusal
  naming W-EFF-1 or W-SPAWN-1 where it does not: 53 of 53 agree.
- Official corpus, `vectors/effect/rce-examples/` (MANIFEST): 15 programs
  copied unchanged from github.com/openIE-dev/rce at 24b9e4c (spec/examples
  13, 14, 15, 37, 40, 47 admitted; ill_typed pure_send, extern_in_pure,
  extern_spawn, iter_effect, device_emit, device_extern, spawn_borrow,
  conc06 W-ND-1_step, conc06b W-SCOPE-1_spawn_outside_scope refused with
  the named rule): 15 of 15. `scripts/check.sh` now runs every
  `tests/*/*_probes.py`.
- Canonical JSON and oracle keys: rce src/canon.rs (bytewise key order,
  serde_json strings and integers), copied unchanged into
  `scripts/ref/canon` (cargo); `scripts/gen/effect_gen.py --ref` confirms
  every one of its 60 values (20 fixed: escapes, U+2028, U+E000 against
  astral keys, i64 limits; 40 random, seed 19) against it. The oracle
  miss text is the engine's (`form/ir.rs`).
- Derived (`scripts/gen/effect_gen.py`): the table (63 `allows` and 56
  `may_call`/`may_spawn` cells, unknown names included), canonical JSON
  and SHA-256 of the 60 values, one oracle
  run (key order does not change a key; re-recording replaces), 35
  backoff delays, 30 retry runs over HC-3 outcomes, 7 deadline cases.
  318 comptime checks with the hand-written ones (plans, byte order).
- `scripts/mut/effect.py`: 35 of 35 mutants killed.

Needs spec: 5.2 says a step may send "to actors and declared channels",
but in rce 0.1.3 a step can reach neither (an event handled by a state
step may not carry a channel, W-TY-2; actors start only in main or a
task, W-EFF-1); a channel the step makes itself is admitted, so the probe
uses that. How an RCE module is used as a fit (D.7) is open, so the
recorded handler is a library value, not an engine fit; the engine's
canonical JSON is not RFC 8785 (key order, integer range), and the two
are not reconciled in the spec. Handlers in the Koka/OCaml sense
(resumable effects) are not expressible: effects are fixed by context,
so this module offers effects as data instead. W-HOST-1 result events
must be declared by the program (see io).

## net (`lib/net/url.rce`, `lib/net/idna.rce`, `lib/net/http.rce`, `lib/net/dns.rce`)

Pure codecs only (survey M25). Anything that moves bytes needs a host
binding and is not here: there is no UDP adapter in rce 0.1.3 (DNS
resolution), the TCP adapter (`NetConnect`/`NetSend`/`NetRecv`) is used
through the io request/result pattern by the program, and TLS is the
host's (`--tls`). Connection management (keep-alive, pipelining,
timeouts) is program state over those events and is left to a later
module once the bindings are specified.

- **net.url, official:** `vectors/net/urltestdata.json`, web-platform-tests/wpt
  at c48d58747e1f211527fb695fd60548a997fae617 (2026-09-19),
  url/resources/urltestdata.json, sha256 81e85fd3c199c08e.... All 896
  tests become exact checks (href and the nine getters, or failure),
  generated by `scripts/gen/url_gen.py` into `tests/net/url_checks.rce`;
  the 80 that involve non-ASCII, `xn--` or a percent-encoded non-ASCII
  byte go through net.idna. All 896 pass, none refused (none hold lone
  surrogates). Plus 2 derived IPv4 checks. 898 checks.
- **net.idna, official:** Unicode 16.0.0 data, each file pinned by sha256
  (`scripts/gen/idna_data.py` refuses a file whose digest differs; the
  large UCD files are fetched from https://www.unicode.org/Public/16.0.0/
  into `~/.cache/rce-ucd`, the others are in `vectors/idna/`):

  | file | sha256 |
  |---|---|
  | idna/16.0.0/IdnaMappingTable.txt | 6db2ef4ed35f3b3de74ebc2e00404a9607f76d499f576b8d4043cf14f1ed175c |
  | idna/16.0.0/IdnaTestV2.txt | ffcf59f6e97d765caa5ac8315e9511ba3b42955a554cbc431e4c016f20d25ca9 |
  | ucd/UnicodeData.txt (fetched) | ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f |
  | ucd/DerivedNormalizationProps.txt (fetched) | 4d4c03892dea9146d674b686e495df2d55a28d071ac474041d73518f887abddc |
  | ucd/CompositionExclusions.txt | 89e83cf9cc8bef6c1f8bf77e42cf6f0341dfa42e66261f4dbe9b492e7a23c8ee |
  | ucd/extracted/DerivedBidiClass.txt | 71ed943a49c58568d8d92e80ecc2ba2f06e62aee9c8ebb0e6e8bd2c3ed8b180e |
  | ucd/extracted/DerivedJoiningType.txt | 6bd08b97da66b70ccfdab105a352de2984e02625239ec5695422c99b33d854f0 |
  | ucd/NormalizationTest.txt (fetched) | d811971453e7075e1ad56fb1b301eece5aa80757b81f6156e74a1bfb3ae5ceb1 |
  | RFC 3492 (rfc-editor.org/rfc/rfc3492.txt) | in `vectors/idna/rfc3492.txt` |

  The data section of `lib/net/idna.rce` (status, mapping, ccc, Mark,
  bidi class, joining type, decompositions, composites) is generated from
  them as binary decision trees (rce 0.1.3 copies a const list on each
  access, which made table lookups about 6 ms each). Checks, generated by
  `scripts/gen/idna_gen.py` into `tests/net/idna_checks*.rce` (16 shards):
  every line of IdnaTestV2.txt (6389, including the 2 with lone
  surrogates, since inputs are code point lists): toUnicode, toAsciiN
  and toAsciiT each checked for the error flag and the output code
  points, also on error lines (19167 checks; status codes are not
  compared, as the file allows; X4_2 alone counts as no error, since
  section 4 records none for an empty label); every line of
  NormalizationTest.txt (19965, the NFC and NFD invariants of c1..c5);
  the 19 RFC 3492 section 7.1 sample strings decoded exactly and encoded
  (delta case ignored, as the RFC prints it in mixed case). 39170 checks,
  all pass.
- `scripts/mut/idna.py`: 41 of 41 mutants killed (Hangul, reordering,
  composition, Punycode adapt/threshold/overflow/surrogates, the six
  bidi rules, ContextJ, hyphens, leading mark, status, STD3, transitional
  mapping, xn-- handling, DNS lengths).
- **net.http, derived and cross-checked:** `scripts/gen/http_gen.py`, a
  Python model of the module's RFC 9112 rules (seed 25): 47 fixed cases
  (framing, chunked with extensions and trailers, every strict refusal,
  1xx/204/304/HEAD, close-delimited bodies, partial input) and 60 random
  ones; each accepted message is also parsed by h11 0.16.0, which agrees
  on 68 and differs on 4 where h11 is stricter (it rejects a leading
  empty line, "Content-Length: 5, 5", a non-chunked response coding, and
  whitespace before a chunk extension; RFC 9112 2.2, 6.3 and 7.1.1 allow
  them). 7 serializations, 5 chunkings, header joining, refused
  serializations (bad method, status 99, CRLF in a value). 121 checks.
- **net.dns, derived and cross-checked:** `scripts/gen/dns_gen.py` with
  dnspython 2.9.0 (seed 53): 40 random messages (A, AAAA, NS, CNAME,
  PTR, MX, SOA, TXT, SRV; names with escapes and specials) written by
  dnspython with compression, decoded to dnspython's own reading of the
  wire, and re-encoded byte for byte to an uncompressed form dnspython
  reads back as the same message; 10 names each way, 7 refused names, 14
  malformed wires dnspython also refuses, 7 pointer and length edge
  cases (forward pointer, 128 vs 130 hops, label types 01/10, 255 vs 256
  bytes), flags, a query and rdata builders. 133 checks.
- `scripts/mut/net.py`: 54 of 54 mutants killed (4 added for domain to
  ASCII) (four equivalent
  mutants removed: "#" in the query set, "0X", "@" vs "%40" in userinfo,
  the port overflow guard).

Needs spec or data: building text from arbitrary
bytes, so HTTP field values with obs-text (bytes 0x80-0xFF) are refused
and DNS label bytes are kept as \DDD; the WHATWG URL setters
(setters_tests.json) and application/x-www-form-urlencoded are not done
yet; host bindings for UDP (DNS) and a connection/stream binding for
HTTP over TCP and TLS.

## os.path (`lib/os/path.rce`)

Pure and lexical (survey M34); symbolic links, the current directory and
the file system's case folding belong to the `Fs` capability (M38).

- **official inputs, reference outputs:** `vectors/os/test_posixpath.py`
  (sha256 34ada45f0e4f0e2b8326404b7e7dbf1416a079081f6e0a1f90ab9d691e9a8be8)
  and `vectors/os/test_ntpath.py` (sha256
  53f7f0b39c1746c1553db4fc3d80d6f75780f0448b15fa478f11d66946e41089),
  CPython v3.13.5 Lib/test. `scripts/gen/path_gen.py` takes every
  argument tuple of a call to split, splitext, splitdrive, splitroot,
  basename, dirname, isabs, join, normpath, relpath, commonpath and
  isreserved in those files (assertEqual, tester(), NORMPATH_CASES,
  splitextTest), and every string literal of the files as a single path
  (984), and gets the expected value from CPython 3.13.5 posixpath and
  ntpath (os.getcwd fixed to "/home/user/bar" and "C:\Users\u"). The
  314 values the tests state themselves are asserted to agree (ntpath up
  to "/" versus "\", as tester() compares). Plus 150 seeded random
  join, relpath and commonpath cases per flavour, 16 derived join inputs,
  11 derived wasi_resolve and 8 derived wasi_preopen cases (the latter
  modelled on wasi-libc's __wasilibc_find_relpath: longest component
  prefix, later preopen on a tie). 16907 checks in
  `tests/os/path_checks.rce`.
- `scripts/mut/os.py`: 27 of 27 mutants killed (three equivalent mutants
  removed).

Needs spec: case-insensitive comparison in win_relpath and win_normcase
folds ASCII only (Windows uses the invariant-locale table, which a host
must supply); wasi_resolve is lexical, the host resolves symlinks under
the same never-leave rule.

## os.env (`lib/os/env.rce`)

Pure (survey M35). Reading the process environment and argv is the
host's: it admits one snapshot (granted keys only) as a logged event.

- **official:** `vectors/os/rust_args_windows_tests.rs` (rust-lang/rust
  8a9dd7abdf013ec10cfeaaa3685feba18ef1c90c,
  library/std/src/sys/args/windows/tests.rs, sha256
  d9c74038d366cad88aecd746cb979309009cdbd7ad9e66614ce0f77bce2b4210): all
  50 `chk` cases of the command line parser, including the Microsoft and
  daviddeley.com examples; `vectors/os/rust_process_windows_tests.rs`
  (same commit, library/std/src/sys/process/windows/tests.rs, sha256
  da3ef3b3b774153074c095da8aa94a446f71480936a313775323781baef46e9a): the
  12 assertions of test_make_command_line; `vectors/os/go_os_env_test.go`
  (golang/go 0b67e71bfae522f3405ba534d9649c7677456613, src/os/env_test.go,
  sha256 6c9d3d1fc4c4008831ccf2b0eb38be2354201a9d32fc1c4a4ec6e665090f037a):
  the 18 expandTests with its testGetenv table.
- **derived:** 120 seeded round trips (parse of the built command line
  gives back the program and arguments) and 27 cases for entries, lookup,
  set/unset, key validity and snapshots. 227 checks in
  `tests/os/env_checks.rce`, generated by `scripts/gen/env_gen.py`.
- `scripts/mut/os.py` (env part): 21 of 21 mutants killed (two
  equivalent mutants removed).

Needs spec: Windows key comparison is ASCII case-insensitive here
(Windows uses its own upper-case table: Rust's windows_env_unicode_case
pairs need that table from the host); environment values are text, so a
POSIX value that is not UTF-8 needs bytes-to-text in the language (same
item as net); the snapshot event and the grant's key list are named in
the survey's grant table item.


## os.perm (`lib/os/perm.rce`)

Pure (survey M36). The host enforces; this module decides. It covers
mode bits, chmod symbolic modes, the POSIX access check, Windows SIDs and
DACL access checks, and Landlock rights, plus the grant-to-rights mapping.

- **official/reference:** `vectors/os/landlock.h` (Linux v6.17,
  include/uapi/linux/landlock.h, sha256 853bf03489c58d284cc9e1315c18138963cd7618514ce7badd28ad1a178d2499): the 20 access-right and
  scope constants and the rights per ABI. chmod is a port of gnulib
  modechange.c as used by GNU coreutils 9.7, and it is checked against
  live `chmod` 9.7 runs: 1,400 seeded cases (52 invalid), including the
  "new permissions are" diagnosis when a mode starting with `-` is passed
  as an option, plus 7 directory cases. `filemode` is checked against
  CPython 3.13 `stat.filemode` (4,544 modes).
- **derived:** 69 cases for the POSIX access check, SID text and binary
  forms (MS-DTYP 2.4.2), DACL evaluation order (MS-DTYP 2.5.3.2),
  Landlock layers, and the grant mapping. 6,033 checks in
  `tests/os/perm_checks.rce`, generated by `scripts/gen/perm_gen.py` (with
  `scripts/gen/perm_derived.rce`).
- `scripts/mut/os.py` (perm part): 30 of 30 mutants killed (two
  equivalent mutants removed).

Needs spec: the grant names table (what `fs.read` and similar names map to)
belongs in the survey's grant table item; Windows SDDL strings, ACE
inheritance, owner rights, and integrity levels are not modelled; and
AppContainer capability SIDs are needed for the Windows mapping of grants.

## os.clock (`lib/os/clock.rce`)

Pure (survey M37). A program never reads a clock. The host admits each
monotonic or wall reading, and each timer firing, as a logged event in D
order; this module holds what a program does with those values. Dur secs
are u64 as in Rust, so the full `Duration` range holds and nothing is
remapped. `as_nanos`, `as_micros` and `as_millis` return exact decimal
text (u128 in Rust), and `from_nanos_text` takes a u128 as decimal digits.

- **official:** `vectors/os/rust_core_time_tests.rs` (rust-lang/rust
  8a9dd7abdf013ec10cfeaaa3685feba18ef1c90c,
  library/coretests/tests/time.rs, sha256
  e43ac73c5c17279b6185071a1703e8d28ce90a5f635308529eab2a3533fb25af):
  every integer assertion and `#[should_panic]` test of `Duration` is
  translated mechanically with its original u64::MAX values: 234 checks,
  all passing. Not yet covered: the 15 float tests (`div_duration_f32`,
  `div_duration_f64`, `duration_const`, `from_neg_zero`, and the
  `duration_fp_*` tests). They need NaN, infinity and -0.0, and rce 0.1.3
  traps on non-finite floats; see the open item below.
  `vectors/os/rust_std_time_tests.rs` (same commit,
  library/std/tests/time.rs, sha256
  dab3fa3fda7c92e542c68e5b51c4ec67a99c39342c59484f4399bd5995b3715a): its
  deterministic Instant and SystemTime assertions are restated in the
  derived file. These are saturating and checked `duration_since`,
  SystemTime MAX/MIN with the 1 ns unix interval, and MAX - MIN ==
  Duration::MAX. `vectors/os/wasi_monotonic-clock.wit` and
  `vectors/os/wasi_wall-clock.wit` (WebAssembly/wasi-clocks
  71e486b1b44a49687dbe17d5b979d6da6112c7f2, wasi:clocks@0.2.8, sha256
  854db80e062a173aab086569d6ced88b83eecfa6ac2239e8b53197c64eb0062a and
  c63feef78e2ffe115f193f3eae8f63efe698a34829ea8fd51155572569199bec)
  define the u64 instant and duration and the datetime record.
- **live reference:** `scripts/gen/clock_ref.rs`, built with rustc 1.99.0,
  run on 1,500 seeded rows (300 of them with secs from 2^63 to
  u64::MAX). Its output is saved as `vectors/os/rust_duration_live.tsv`
  (sha256 2d5d058b7e8cc8328c0cea8be0f3178398d3704e2777977a0fa6ed655c33fc4e).
  It covers checked add/sub/mul/div, abs_diff, 17 Debug format specs
  (precision, width, alignment, sign), and SystemTime epoch ± d, add/sub
  and `duration_since` in both directions: 42,310 checks in
  `tests/os/clock_checks_live.rce`, all exact, with no result remapped.
- **derived:** 89 checks for the Timespec range edges (i64 ± u64 exactly,
  as Rust's checked_add_unsigned and checked_sub_unsigned), exact
  as_nanos/as_micros/as_millis, u128 from_nanos, wasi value mapping, Mono
  admission and the timer queue. `tests/os/clock_checks.rce` has 323
  checks; both shards are generated by `scripts/gen/clock_gen.py` (with
  `scripts/gen/clock_derived.rce`).
- `scripts/mut/os.py` (clock part): 48 of 48 mutants killed (two
  equivalent mutants removed). The mutation test runs every shard.

Needs spec (engine): the host admits timer firings as events in the order
`fire` gives, at tick boundaries; the wasi pollable maps to a timer id.
Wall-clock resolution and clock identity are carried by the event. Open:
the float Duration API (`from_secs_f64`/`f32`, `mul_f64`, `div_f64`,
`div_duration_f64`, `as_secs_f64` and the f32 forms) and its 15 official
tests. NaN, infinity and -0.0 inputs and results cannot exist as rce
values (non-finite floats trap), so these tests need either language
support or a bit-exact soft-float over u64 bit patterns.

## data.events (`lib/data/events.rce`)

Pure (survey M60; the module is `data.events` because `event` is an rce
keyword). CloudEvents 1.0 as the common envelope of change feeds.

- **official:** `vectors/data/cloudevents/http-protocol-binding.feature`
  and `kafka-protocol-binding.feature` (cloudevents/conformance
  80f89d1d7ee71fbc998fad39ca48c95ff403c44f, `features/`, sha256
  d24dc8f28cb529b77377216a4f2015dbb93644980a09dfd8f3bdd1705a23c40c and
  a3e6f08bde13e4b219f13036ef4da168f2e797e525c05f244c80767939251865): all
  7 scenario rows (HTTP binary x2, HTTP structured x2, Kafka binary,
  Kafka structured x2), attributes and data compared;
  `json-format.md` and `http-protocol-binding.md` (cloudevents/spec
  2ed3806b4ad8fda35813263cfefb2d73098b7655, sha256
  d1ad3b0df968d4d436d2c38144042fe4502517171ad570e7d58b85eeaca8ef4a and
  a8ef45eb29b3a380c3a7388b8e6eb657eddf21169ca5ca7d070f64a751fcbb4a): the
  6 example events of 3.2 with their HTTP binary re-encodings (headers
  and body), the batch example of 4.3 and the empty batch, the
  percent-encoding example of 3.1.3.2 ("Euro € 😀") both ways, the
  overlong "%C0%A0" rejection, lower-case and unnecessary escapes. The
  placeholder "... base64 encoded string ..." of two examples is not
  base64 and is replaced by a real base64 string (the rest of the example
  is unchanged). 27 checks.
- **derived:** the CloudEvents Python SDK 2.2.0 (reference
  implementation; `vectors/data/cloudevents_sdk_py.json` keeps its
  outputs): 150 seeded events with non-ASCII, emoji, quotes and percent
  signs in attributes, string/integer/boolean extensions and JSON, text,
  binary or no data, each encoded by the SDK as structured JSON, HTTP
  binary (the SDK percent-encodes every non-unreserved character) and
  Kafka binary, and decoded here to the event the SDK decodes; our HTTP
  binary encoding (the spec's minimal escaping) is checked to decode in
  the SDK to the same attributes. Where the SDK departs from the spec the
  spec is followed and the case says so: the SDK defaults a missing `id`
  and `specversion` (REQUIRED in spec.md 3.1, refused here), accepts
  control characters (spec.md type system forbids U+0000..U+001F,
  U+007F..U+009F) and returns a binary-mode body as text when it is
  UTF-8 (http binding 3.1.1: the body is the data's bytes). Plus 117
  restated cases: RFC 8259 grammar (leading zeros, fractions, escapes,
  surrogate pairs, trailing data), RFC 3339 times (leap years, leap
  second, offsets), attribute names, media types, quoted-string header
  values, integer range of extensions, duplicates. 702 checks.
- 729 checks in `tests/data/events_checks*.rce` (7 shards), generated by
  `scripts/gen/events_gen.py`; `scripts/gen/cptab.py` writes the code
  point table. `scripts/mut/data.py` (events part): 33 of 33 mutants
  killed (one equivalent mutant removed).

Needs spec: rce 0.1.3 has no text-from-code-point or text-from-UTF-8
built-in, so `\u` escapes and percent-decoded UTF-8 go through a table
of the BMP and U+1F000..U+1FFFF; other supplementary code points are
refused. JSON numbers are kept and compared as lexemes (no f64 bit
access). A `T? ! E` result loses its `some()` wrapper in the 0.1.3
comptime evaluator; the module avoids that shape.

## ui.text (`lib/ui/text.rce`)

Pure (survey M47), on Unicode 16.0.0: UAX #29 revision 45, UAX #14
revision 53, UAX #9 revision 50, without tailoring. Code points are i64;
`cps` decodes UTF-8 through `text/utf8`. The property tables (gcb, wb,
sb, lb, ea, gc, bidi class, Extended_Pictographic, InCB, bidi brackets)
are if-trees written after the `# ---- data` marker by
`scripts/gen/text_data.py`, which checks every UCD file against its pin.

- **official:** every line of the Unicode 16.0.0 test files passes:
  GraphemeBreakTest.txt 1,093, WordBreakTest.txt 1,826,
  SentenceBreakTest.txt 512 (sha256 ee2b9354…, ad985d57…, 0aef8403…, in
  vectors/ui), LineBreakTest.txt 16,672 (910759a6…), BidiTest.txt 490,846
  cases in all their paragraph directions (93e5eb9d…), and
  BidiCharacterTest.txt 91,707 (d04a51a9…). The last three and
  DerivedCoreProperties.txt (39d35161…) are fetched from unicode.org into
  ~/.cache/rce-ucd/16 and checked against their pins; DerivedBidiClass.txt
  (71ed943a…) is the copy in vectors/idna. Property files in vectors/ui:
  GraphemeBreakProperty c29360bd…, WordBreakProperty 476464e7…,
  SentenceBreakProperty 20aab5ec…, LineBreak e97e4259…, EastAsianWidth
  43adc76c…, emoji-data f1365a51…, DerivedGeneralCategory 7676ab75…,
  BidiBrackets b8f32554…. Shards are written by `scripts/gen/text_gen.py`:
  `tests/ui/text_checks.rce` (segmentation and derived),
  `text_checks_line*`, `text_checks_bidi*` and `text_checks_bidichar*`.
- **live reference:** ICU 76.1 (libicuuc.so.76, Unicode 16.0, root
  locale) through ctypes, `scripts/gen/text_icu.py`. Its character, word,
  sentence and line break iterators are run on 15,000 seeded sequences
  drawn from every property value plus the characters the rules name. The
  output is saved as `vectors/ui/icu76_live.tsv` (sha256
  56c71bee8f8c95d3e7b735a58e573045d5bf095b487a3750b9739aec253e622e), and
  all 15,000 match in `text_checks_icu*`. The pools leave out what ICU
  tailors: lb=SA, and for words Han, Hiragana, Katakana, Hangul syllables
  (dictionary segmentation), plus the colons CLDR root removes from
  MidLetter. ubidi is not used as a reference: ICU 76 resolves an AN
  right after L in an LTR paragraph to level 0, while UAX 9 I1 (and
  BidiTest) give 2.
- **derived:** 36 checks in `scripts/gen/text_derived.rce`: empty input,
  `segments`, `line_kinds` and `line_mandatory`, paragraph level, and
  cell width. Width is not a Unicode algorithm. It is derived as
  wcwidth-like: 0 for Mn, Me, Cf, Cc, ZWSP and Hangul medial and final
  jamo; 2 for East Asian W and F; 1 otherwise. A cluster takes its first
  non-zero width, and becomes 2 when a pictographic base is followed by
  VS16. The derived checks also include LB25 PR/PO OP IS NU, confirmed
  with ICU.
- `scripts/mut/ui.py`: 65 of 65 mutants killed. Three were removed as
  equivalent; the reasons are in the script header.

Needs spec: tailoring (dictionary segmentation for SA, CJK and Hangul,
and locale word rules) and the bidi paragraph API above one paragraph
(the caller splits at B). Width follows no standard, so the terminal
host must agree to it.

## data.feed (`lib/data/feed.rce`)

770 checks in `tests/data/feed_checks*.rce` (7 shards), generated by `scripts/gen/feed_gen.py`: 353 official, 417 derived.

- Official: mongodb/specifications (commit ca7d473e) change-streams unified tests `change-streams-resume-allowlist.json` (sha256 c8a3ba12…), `change-streams-resume-errorLabels.json` (05644e91…), `change-streams-errors.json` (6cdf3b08…), `change-streams.json` (39b13699…), replayed through a scripted server at several wire versions; the Resumable Error table and prose tests 1-4, 11, 13, 14, 17-19 from `change-streams.md` (0039cb9e…) and `change-streams-tests-README.md` (2c701acf…).
- Derived: hash chains from `rce trace --json` (rce 0.1.3) saved in `vectors/data/rce_trace_feed.json` (0bc872f7…); PouchDB 9.0.0 checkpointer compare/update cases from `scripts/gen/feed_pouch.js`, saved in `vectors/data/couchdb/pouchdb_checkpointer_9.0.0.json` (ecdc535f…); admission scripts.
- Mutation testing: `scripts/mut/data.py feed`, 26 of 26 mutants killed (one documented equivalent mutant excluded).

## ui.input (`lib/ui/input.rce`)

Pure (survey M46). The host reads terminal bytes or window or DOM input
and admits each read as a logged event. This module decodes and maps the
input.

The terminal decoder is a port of crossterm 0.29.0's Unix parser
(`src/event/sys/unix/parse.rs`) and of its `Parser::advance` stream loop.
Results follow release builds: u16 coordinates wrap, so a 0 coordinate
gives 65535. `raw` stands for crossterm's raw-mode query.
`is_upper` is Rust's `char::is_uppercase` (rustc 1.99.0, Unicode 17.0.0:
DerivedCoreProperties-17.0.0.txt, sha256
24c7fed1195c482faaefd5c1e7eb821c5ee1fb6de07ecdbaa64b56a99da22c08,
fetched into ~/.cache/rce-ucd/17 by `scripts/gen/input_data.py`).

- **official:**
  - `vectors/ui/crossterm_parse.rs` is the crate's parse.rs (crate sha256
    d8b9f2e4c67f833b660cdb0a3523065869fb35570177239812ed4c905aeff87b,
    file 689b765d67be62703f7935abfa1475d98994987461e413836b54c7936966ce2b).
    `scripts/ref/input_ref` runs its 30 unit tests, and each `assert_eq!`
    and `assert!(is_err)` also records its value in canonical form. That
    gives 75 assertions in `vectors/ui/crossterm_official.tsv` (sha256
    5381a85a…), all passing.
  - UI Events KeyboardEvent code values (`vectors/ui/uievents-code.html`,
    w3c/uievents-code b201684d1de0af90bc403814bbdee6aa96647130, sha256
    d0c8f75d…): 172 values with their Required flag.
  - KeyboardEvent key values (`vectors/ui/uievents-key.html`,
    w3c/uievents-key 140cae88d6039b3fb1a51787978678115111c433, sha256
    702040ce…): 284 named values with Required.
  - The implicit "F"/"Soft" + index names that the spec text gives
    ("F13" to "F15", "F24", "Soft8").
  - Pointer Events `buttons` table (`vectors/ui/pointerevents.html`,
    w3c/pointerevents 7affe62fa4332543f8b13c56292104e4e9dfa517, sha256
    9231bce9…): 7 rows.
  - Total: 539 checks, all passing.
- **live reference:** the same harness (`cargo run --release`) runs 20,000
  seeded single `parse_event` calls and 5,000 multi-sequence streams
  through `Parser::advance`. These cover legacy, SS3 and CSI keys, CSI u,
  all three mouse encodings, paste, DA1, flags, partial and invalid
  UTF-8, both raw modes, and input_available. The results are saved as
  `vectors/ui/crossterm_live.tsv` (sha256 ddd8620b…), and all 25,000
  match in `tests/ui/input_checks_live*.rce`. `python3
  scripts/ref/input_ref/make.py` rebuilds the patched parse.rs, and the
  harness reproduces both TSVs byte for byte.
- **derived:** 61 checks in `scripts/gen/input_derived.rce`, covering:
  - DOM mapping: every crossterm key code maps to a named UI Events key;
    location; repeat/up; modifier and lock flags.
  - Key-string rules.
  - Pointer button bitmask state across down/drag/up.
  - Wheel deltas in lines.
  - Composition order.
  - is_upper spot checks.
  - Split reads.
  - Edge cases confirmed with the harness's `one` mode (a final byte
    above 126, a leading "+" in numbers, invalid shifted code points,
    invalid UTF-8 in an SGR field, keypad Enter).
- `scripts/mut/input.py`: 64 of 64 mutants killed.

Not used: xterm ctlseqs (patch #411 was read for the encodings, but it
has no machine-checkable cases), and esctest, which tests terminal output
(M55). Derived and not from a spec: `dom_key` (terminals report no
physical key, so there is no `code`; crossterm Media Reverse and
IsoLevel5Shift become "Unidentified"; SUPER and META both set metaKey),
`dom_pointer` (mouse is pointerId 1 and primary, coordinates in cells),
and `dom_wheel`.

Needs spec (engine): host input adapters emit these values as logged
events in read order. A read carries its bytes and whether more input is
available. The terminal host records raw mode.

## data.cdc (`lib/data/cdc.rce`)

Official vectors (20 checks): the worked examples in the Debezium 3.0.0.Final
connector docs (`postgresql.adoc`, `mongodb.adoc`, debezium/debezium tag v3.0.0.Final commit 2450021e), extracted by
`scripts/gen/cdc_debezium_extract.py` into `examples_3.0.0.json` (11 examples).
The doc examples are not all valid JSON as printed; the extractor records each fix
(trailing commas, `...` elisions, comments) next to the example. CouchDB 3.4.2
`changes.rst` (apache/couchdb-documentation commit 1df690da) normal and continuous feed examples.

Derived vectors (174 checks), produced by live reference servers on the box, no sudo:
- PostgreSQL 17.11 (conda-forge) with wal2json 2.6 (eulerto/wal2json 75629c2e):
  `scripts/gen/cdc_pg.py` records raw pgoutput protocol v1 messages
  (pg_logical_slot_peek_binary_changes, proto_version 1) and the wal2json
  format-1 and format-2 output of the same transactions (6 cases: insert/update/delete,
  REPLICA IDENTITY FULL, unchanged TOAST, truncate, logical message, NULL/NaN/Infinity/bytea).
  Every pgoutput case decodes to exactly the wal2json v2 change list.
- MongoDB 8.3.7 replica set (pymongo): `cdc_mongo.py`, 17 change events with
  pre/post images (`whenAvailable`), updateDescription incl. truncatedArrays;
  `apply_update` reproduces the server post-image from pre-image + updateDescription.
- SQLite changesets/patchsets from APSW 3.54.0.0 Session (`cdc_sqlite.py`), incl. a 9-byte varint.
- Redis streams (RESP2/RESP3 raw replies, `cdc_redis.py`), NATS 2.10.22 JetStream
  (HMSG frames, $JS.ACK v1/v2 subjects, PubAck; `cdc_nats.py`),
  PouchDB 6.0.0 pouchdb-memory changes feed (`cdc_pouch.js`).

Mutation testing: `scripts/mut/data.py cdc` 41/41 killed (equivalent mutants removed are listed in its header).

- `vectors/data/couchdb/changes.rst` sha256 `3c04f9bdd9e046fa916c6f26aaf7f500a04a8dc4c1117362a9304e813e6765e3`
- `vectors/data/couchdb/pouchdb_changes.json` sha256 `893374b23008354fa0577dde26abbfa9ecdf049524fb6f90bba049dd0bf7a29c`
- `vectors/data/debezium/examples_3.0.0.json` sha256 `6a1cf25b0fded9da8195a5d25622ad06479c69bb5e00f9a03c960eec8da73750`
- `vectors/data/debezium/mongodb.adoc` sha256 `75b243246512d4c6a477b6e3efac9dafdbcd25536b015886229b278c67b8a289`
- `vectors/data/debezium/postgresql.adoc` sha256 `3fbbf683a44763c0a9ae1e7448c4d9877041e2ab1f792ae539a2bbb639ae8598`
- `vectors/data/mongodb/live_events.json` sha256 `334ca38720f4dd33368d6b9d604a7f5c25fccc00f92ba60d9622e967eadbd7f1`
- `vectors/data/nats/jetstream.json` sha256 `1bd1cecd0f13e8e1f89d4664b2c3282b8ab1da5f9efaa402761870d7271accdf`
- `vectors/data/postgres/pg_live.json` sha256 `92c74ef35ac4ed3af1915177b1e8e446eeb8d354ba408077cbb72cbc96b9ce6e`
- `vectors/data/redis/streams.json` sha256 `74a61edda5c37dd0f8f07e31ac4664d7e94181cda62d974122efb7d4c3a5c283`
- `vectors/data/sqlite/changesets.json` sha256 `37b4c5174afd875073bc7b819dc672b38b635c26ac46a9106b6ed18e50d634d2`

## ui.view (`lib/ui/view.rce`)

Pure (survey M49). A view is a flat arena of nodes (tag, key, sorted
attributes, text, parent, ordered children); `diff(a, b)` lists the
operations that turn a into b and `apply` replays them. The frame digest
is BLAKE3 over deterministic CBOR (RFC 8949 4.2.1): each node is
`[tag, key, {attrs}, text, [children]]`, and a frame record is
`["rce.ui.frame/1", seq, h'view digest']`.

The child-list algorithm is derived from its laws, not copied:
- Matching: nodes are the same when tag and key agree. A common prefix
  and suffix are matched in place; in the middle, keyed nodes match by
  key (and tag) and an unkeyed old node takes the first free unkeyed new
  node of its tag, as in Vue 3.
- Removals and creations are forced by the matching (each unmatched node
  once). Moves: a node never moved keeps its order relative to the other
  unmoved nodes, so the unmoved nodes are an increasing subsequence of
  old positions taken in new order; hence moves >= n - LIS. Visiting new
  nodes right to left and moving exactly the matched nodes off one LIS
  (each before its already-placed right neighbour) reaches the target
  with n - LIS moves, so that count is the least possible. The LIS is
  the patience-sorting one (Vue's `getSequence`, Inferno).
- Checked against a breadth-first search over all single-node moves:
  every permutation of up to 6 keys and 600 random lists with insertions
  and removals (`tests/ui/view_checks_min*.rce`, 1,474 checks).

- **official:** `vectors/ui/vue_rendererChildren.spec.ts` (vuejs/core
  v3.5.44, commit bf5f0dd877dd3c23ce85c2cbc1dd771567e84c8c, sha256
  f8a3acaf503ff8c27f1f832a1f6cd4e716eb02682a592cd6f00b90cb91d07a4c).
  Each test becomes an old/new child list (`scripts/gen/view_gen.py`
  `OFFICIAL`); the test's final children are checked by replaying our
  operations and comparing the trees, and the container operation log is
  compared with Vue's. `random shuffle` (Math.random) uses 5 seeded
  shuffles, also as trees with the per-position `style` attribute.
  `should warn with duplicate keys`: such a list is rejected by
  `view_ok`. `h('div', 'hello')` text children are a text node child.
  `vectors/ui/vue_official.tsv` (sha256 7cb247e8…): 41 cases. 40 logs
  equal Vue's. `children with the same key but with different tag`
  differs by design: Vue matches by key alone, replaces the node in
  place and then moves the replacements (3 moves); here a key match
  needs the same tag, so the nodes are removed and created (1 move). The
  final children are the spec's in both.
- **live reference:** `scripts/ref/view_ref/vue_ref.js` runs
  @vue/runtime-core 3.5.44 with a recording renderer (`npm install` in
  that directory). 6,000 seeded lists (keyed, keyed with other tags,
  unkeyed, text, comments, relabelled unkeyed nodes, retagged keys) in
  `vectors/ui/vue_live.tsv` (sha256 a64e1862…). 5,779 logs equal Vue's
  exactly; the other 221 all have a key reused with another tag, and in
  each our move count is no larger than Vue's (asserted by the
  generator). `--refresh` reproduces both TSVs byte for byte.
  `scripts/ref/view_ref/model.py` is the Python statement of the rules
  used to state our expected logs.
- **digest reference:** 400 random trees and edits: the CBOR bytes equal
  Python cbor2 5.6.5 `dumps(canonical=True)` of nested lists, and the
  view digest after `apply(a, b, diff(a, b))` and the frame digest equal
  the Python `blake3` package's (`tests/ui/view_checks_tree*.rce`).
- **derived:** `scripts/gen/view_derived.rce` (LIS cases, exact
  operation lists, CBOR key order, well-formedness, frame record bytes).
- **optimal matching (`diff`):** `match_opt` keeps a longest common
  subsequence (under same tag and key) in place, then matches every
  remaining same pair. Container operations are |old| + |new| - |M| -
  LIS(M): |M| is the most the tags and keys allow, and LIS(M) <= LCS
  for any matching, with equality here, so no matching does better.
  Checked against a breadth-first search over removals, creations and
  moves on 1,500 distinct random list pairs of up to 5 children (keyed
  spans, unkeyed `<s>`/`<p>`, text nodes): `tests/ui/view_checks_opt*.rce`,
  all equal. Vue's first-free matching (`diff_vue`, the mode the Vue
  vectors use) costs more on 23 of them.
- `scripts/mut/view.py`: 48 of 48 mutants killed. Three equivalent
  mutants were removed: dropping `not put` in `attach` (sibling ids are
  distinct, so the anchor occurs at most once), dropping match_opt's
  common-suffix loop (a common suffix lies on some LCS, so the count is
  unchanged; this mode specifies the count, not the sequence), and
  `and` for `or` in its empty-side test.

Divergences from Vue, by design: key matching requires the same tag (see
above); operations are ordered breadth first (a node's attributes and
text, its child list, then its matched children), where Vue patches
children depth first while scanning. In `diff_vue` minimality holds for the moves given Vue's matching;
`diff` is optimal over all matchings.

## ui.states (`lib/ui/states.rce`), ui.ecma (`lib/ui/ecma.rce`), text.xml (`lib/text/xml.rce`)

Pure (survey M53). `ui.states` is an SCXML interpreter that follows the
W3C SCXML Recommendation Appendix D algorithm. It reads SCXML documents
(through `text.xml`) with the ECMAScript data model (through `ui.ecma`, a
subset of ECMAScript that covers what SCXML documents use), and it also
reads a compact chart notation. It covers send/cancel/invoke (with
autoforward and finalize), delayed events, donedata, late binding and the
Basic HTTP Event I/O Processor (requests are passed in as logged inputs).
`run_mode(src, evs, true)` is an XState 5 compatibility mode (event
selection, parallel regions and transition domain as XState computes
them).

- **official:** the W3C SCXML 1.0 Implementation Report Plan test suite
  (https://www.w3.org/Voice/2013/scxml-irp/, `manifest.xml` sha256
  5974da27…, `confEcma.xsl` sha256 f93de8fb…, both matching the W3C copy).
  It is vendored in `vectors/ui/scxml-irp-full`. `scripts/gen/irp_xslt.py`
  applies the suite's own `confEcma.xsl` with Saxon-HE (the stylesheet
  is XSLT 2.0), and `scripts/gen/irp_gen.py` writes
  `tests/ui/states_checks_irp*.rce`. All 200 tests pass: 166 mandatory
  and 34 optional, including the 9 manual tests (each is checked against
  what its description asks the tester to observe) and test 513 (Basic
  HTTP). `text.xml` reads all 206 IRP documents the same way lxml does
  (`tests/text/xml_checks_irp*.rce`, 206 checks).
- **live reference:** XState 5.33.2 (npm). `scripts/ref/states_ref` runs
  300 generated charts with event sequences, and the final
  configurations, entry/exit traces and data all match
  (`tests/ui/states_checks_live*.rce`). SCION 4.3.28 deviates from
  Appendix D, so it is used only as a cross-check after
  `patch_scion.py`. For `ui.ecma`, Node v20.19.2 (V8) evaluates
  495 expressions and scripts (`scripts/ref/ecma_ref`), and all of them
  agree (`tests/ui/ecma_checks*.rce`).
- **derived:** 37 compact-notation charts from IRP section 3
  (`tests/ui/states_checks.rce`). From the Appendix D text: across
  parallel regions, the LCCA is the nearest compound ancestor, and states
  exit in reverse document order. From ECMA-262: abstract equality with
  null/undefined, precedence of `&&` and of relational over equality, and
  string ordering. From XML 1.0 WFC Legal Character: `&#0;`.
- **mutation:** `scripts/mut/states.py` 16/16, `scripts/mut/ecma.py`
  10/10, `scripts/mut/xml.py` 8/8 killed.

## math (lib/math/math.rce)

Worst cases from CORE-MATH (https://gitlab.inria.fr/core-math/core-math,
commit dae5dd8c0d5bd2691b45de28c9edbe12f067c3d3, MIT; Copyright (c)
2021-2022 Alexei Sibidanov, Paul Zimmermann and the CORE-MATH authors),
`src/binary64/<fn>/<fn>.wc`, copied unchanged with their license header.
Expected values are not taken from any libm: `scripts/gen/math_vec.py`
computes each with MPFR 4.2.2 (gmpy2, ieee(64) context, subnormalized,
round to nearest even), so each is the correctly rounded binary64 result,
and checks compare exact bits (no tolerance). Inputs whose result
overflows (a trap in RCE) and non-finite inputs are skipped. Random cases:
uniform 64-bit patterns and uniform [-4, 4], seed 20261015 + arity; edge
cases: zeros, subnormal and normal boundaries, max, small integers, all
sign combinations. pow adds exact results and midpoints (integer y up to
64, y = n/2^k with x's odd part a perfect 2^k-th power, powers of two to
2^-1075) and random x in [0, 4] with y in [-300, 300].

| fn | CORE-MATH worst cases | random | edge | total |
| --- | --- | --- | --- | --- |
| cbrt | 106,244 | 2,999 | 12 | 109,255 |
| hypot | 32,917 | 3,000 | 496 | 36,413 |
| exp (push CI sample) | 4,005 of 1,129,105 | 2,504 | 122 | 6,631 |
| log (push CI sample) | 4,089 of 134,941 | 1,484 | 70 | 5,643 |
| log (full, off-repo run) | 134,941 | 1,483 | 7 | 136,431 |
| log2 (push CI sample) | 4,155 of 33,279 | 1,483 | 94 | 5,732 |
| log10 (push CI sample) | 4,095 of 65,553 | 1,484 | 238 | 5,817 |
| sin (push CI sample) | 4,005 of 1,971,056 | 3,000 | 126 | 7,131 |
| cos (push CI sample) | 4,008 of 1,090,196 | 3,000 | 126 | 7,134 |
| tan (push CI sample) | 4,006 of 1,077,395 | 3,000 | 140 | 7,146 |
| atan2 (push CI sample) | 3,991 of 28,151 | 3,000 | 603 | 7,594 |
| cos (full, off-repo run) | 1,088,192 | 2,999 | 107 | 1,091,298 |
| sin (full, off-repo run) | 1,971,056 | 2,999 | 105 | 1,974,160 |
| exp (full, off-repo run) | 1,129,105 | 2,503 | 99 | 1,131,707 |
| pow (push CI sample) | 7,229 of 1,003,179 | 5,428 | 2,188 | 14,845 |

The large CORE-MATH files (exp 25 MB, log 3 MB, sin 43 MB, cos 24 MB) are
not copied: push CI runs an evenly spaced sample (`--sample 4000`), and the
full files are run with `math_vec.py FN --cm core-math/src/binary64 --out DIR`
and the same harness (results below, every case exact). Constants (ln 2,
pi/2 to 2100 bits, 2/pi to 4200 bits) come from `scripts/gen/math_const.py`
(MPFR, 9000 bits; `--check lib/math/math.rce`).

Derived (`tests/math/math_checks.rce`, `math_checks_eft.rce`,
`math_checks_ties.rce`): exact hypot midpoint ties (300, a = t(t+1) 2^e,
b = (2t+1)/2 2^e), TwoProduct and TwoSum error terms and frexp1 against
exact rationals, small cases.
Cross-target: `scripts/math_parity.sh` emits the module with the derived
checks and one shard per function to Rust, Python and JavaScript and
requires identical transcripts, plus Scala 3 when scala-cli is installed
(run on the box: Scala 3.9.0, JVM 17; all four identical).
Mutation testing: `scripts/mut/math.py` 42/42 killed (pow: 13/13) (equivalent mutants
removed are listed in its header).

- `vectors/math/cbrt.wc` sha256 `c0487c6527aafea0eb741a4464c745bd545c49fc2a00ff7dbb89ee882b16047a`
- `vectors/math/hypot.wc` sha256 `a7f24bd5b59d371aa1e1a2385dd9d526e21c6b4f751f735471f72d505103fac7`
## data.sync (`lib/data/sync.rce`)

Official vectors (127 checks):
- json-patch/json-patch-tests commit 2a928f90 (2025-02-21), `tests.json` and
  `spec_tests.json` copied verbatim: all 112 cases, the 4 marked disabled
  included (two `op` members in one operation, a scalar root); duplicate
  member names are kept as in the source text when the case is rendered.
- RFC 7396 Appendix A, the 15 merge-patch examples, transcribed into
  `rfc7396_appendix_a.json` (rfc-editor.org text, sha256 545e1d61...c6).

Derived vectors (1145 checks):
- fast-json-patch 3.1.1 (`scripts/gen/sync_fjp.js`, seed 20261015): 150
  `compare` diffs applied by `patch_apply`, and `diff` checked by its law
  `patch_apply(a, diff(a, b)) == b` on the same pairs; 199 random patches
  applied with validation. Skipped where fast-json-patch departs from the
  RFCs: `~2` in a pointer (RFC 6901 allows only `~0`/`~1`), root targets and
  a move from the root (later ops on a scalar root are ignored), and 8
  TypeError/undefined results; RFC rules for those are checked directly.
- @automerge/automerge 2.2.9 (`scripts/gen/sync_automerge.js`, seed 4242):
  60 sessions of concurrent map puts/deletes on three replicas, 478 ops in
  shuffled order; the merged document and `getConflicts` per key (82 keys
  with several values) against `map_state` / `map_conflicts`; each session
  also merged from two overlapping halves in both orders.
- PostgreSQL 17.11 with wal2json 2.6 (`scripts/gen/sync_pg.py`, seed 6262):
  30 random transactions on a REPLICA IDENTITY FULL table, decoded with
  format-version 2; `table_apply` must give the server's table, and for 7
  subscription queries `sub_update` must give the server's own
  `(new WHERE q) EXCEPT (old WHERE q)` and the reverse (text compared
  COLLATE "C").
- Laws, stated in `scripts/gen/sync_gen.py`: PN-counter, OR-set (add wins)
  and LWW merge are commutative, associative and idempotent and give the
  expected value; the mutation log (Replicache push/pull: dedupe by client
  and mutation id, a gap refused, replay(log) == db, pending mutations
  rebased on pull, all views equal the server after a settle round).

Mutation testing: `scripts/mut/data.py sync` 48/48 killed (one equivalent
mutant removed, noted in its header).

- `vectors/data/jsonpatch/rfc7396_appendix_a.json` sha256 `9b5fd611d7564a0e926756d912a50a118e4b9f32b84f0a8dc13ad70058488ad6`
- `vectors/data/jsonpatch/spec_tests.json` sha256 `a26b050292207033e5cccc5d6102b7bd6f8add7db0d0680e5d46a7ecf40a8c7b`
- `vectors/data/jsonpatch/tests.json` sha256 `de3dce3d0d5029fed83007e50b54607750dd3d1478d3c59ca35fdc18fb1a04ae`
- `vectors/data/sync/automerge_map.json` sha256 `6c1a591d2304a154e80a50c7b8eb0239b4d4278a7ef063c9a76e3ab37f0c3617`
- `vectors/data/sync/fast_json_patch.json` sha256 `59e159c6000c9218fb48cf6df94d1316955916bc64ddf62e40cc3414e55aa1ad`
- `vectors/data/sync/pg_subscriptions.json` sha256 `134cb8582f750b1496ed7297ed798929197cc190e8c5008adc769175dcb10751`


## data.csv (`lib/data/csv.rce`)

Pure. Reading follows CPython 3.13's `Modules/_csv.c` and the Sniffer
follows `Lib/csv.py` of CPython 3.13.16, the interpreter the vectors were
generated with. 1968 checks in `tests/data/csv_checks*.rce` (29 files),
all generated by `scripts/gen/csv_gen.py` (`python3 -I`, seed 20261025),
which also writes the Unicode tables at the end of the module.

- **official:** csv-spectrum 2.0.0 (npm `csv-spectrum`,
  https://github.com/maxogden/csv-spectrum, gitHead
  d30e80f8b99d2eecb3778f1d7b9ed1cb425502ec, tarball sha256
  002a6a938bf51ceb26af7b760f5c1b54baca4820b0e9142f135619210c93d021;
  BSD-2-Clause as its `package.json` states), copied unchanged to
  `vectors/data/csv-spectrum/` (`csvs/*.csv`, `json/*.json`,
  `package.json`): all 12 files are read as text by `csv_table` and as
  UTF-8 bytes in seeded chunks by `csv_feed_bytes` and `csv_records`, and
  the records are compared with the JSON (24 checks). The JSON of
  `location_coordinates.csv` gives the phone number 1234567890 where the
  CSV holds 2095257564; the check takes the CSV's value, which CPython
  reads, and every other field of that record equals the JSON.
- **derived (CPython 3.13.16 `csv`):** the generator holds a Python model
  of the module's reader and writer and checks it on every vector against
  CPython: each row csv.reader yields, `csv.reader.line_num` after it, and
  each csv.Error or ValueError (the reader is called again after an error,
  as a caller that catches it does); each `csv.writer.writerow` output or
  error. The model supplies what CPython does not report: the column of a
  reader error and the field of a writer error.
  - reader, 810 checks, each read three ways that must agree (whole,
    through the character state machine alone, and in seeded chunks):
    RFC 4180 section 2 examples and hand cases (line ends LF, CRLF and a
    lone CR, blank lines, unterminated quotes, NUL, U+FEFF, escape
    characters, a closing quote without doublequote, lines that end only
    at LF as `has_header` reads them, `csv.field_size_limit(4)`); seeded
    random input under 17 dialects (excel, excel strict, excel_tab, unix,
    skipinitialspace, escapechar with and without doublequote, QUOTE_NONE
    with and without a quote character, QUOTE_NONNUMERIC, QUOTE_STRINGS,
    QUOTE_NOTNULL, a space delimiter with skipinitialspace, `;` with `'`,
    strict with escapechar, a non-ASCII delimiter and quote, `|`), with a
    field limit of 3 on a quarter of them; 200 lines full of quote
    characters under 8 dialects; tables written by csv.writer read back.
  - writer, 384 checks: `writerow` hand cases and seeded random records
    of text, numbers and None under 25 dialects (all quoting modes,
    escapechar, no doublequote, line terminators "\n", "!" and "ab", a
    space delimiter with skipinitialspace), with every csv.Error; 10
    `writerows` texts.
  - dialect validation, 95 checks: the error of
    `csv.reader(io.StringIO(""), **dialect)` for bad and good delimiter,
    quote and escape characters, quoting values and line terminators.
  - Sniffer, 491 checks: `sniff` (258, 28 of them with a `delimiters`
    argument) and `has_header` (233) on hand samples, 150 seeded tables
    written by csv.writer (9 delimiters, both quote characters, three
    quoting modes, three line ends, spaces after delimiters, truncated
    samples) and 60 seeded random texts; outcomes include "Could not
    determine delimiter", a sniffed delimiter equal to the quote character
    and the error a lone CR gives when has_header reads the sample.
  - `float()` and `complex()` of a str, 48 checks over 565 strings
    (underscores, exponents, inf/nan, brackets, Unicode digits and
    spaces).
  - header rows, 49 checks: `csv_table` against `csv.DictReader` on the
    rows of the header's width, ragged rows reported with row, line and
    column.
  - bytes, 48 checks: UTF-8 cut at seeded byte positions, byte-order
    marks, ill-formed sequences; the decoding is checked against
    `bytes.decode("utf-8-sig", "replace")`.
- **hand:** 19 checks of the public functions (`csv_read`, `csv_reader`,
  streaming with a CR at a chunk end, `csv_cells`, `csv_records`, a field
  longer than one text fed in chunks, 52,000 records over three chunks,
  the writer's limits at one text).

Mutation testing: `scripts/mut/data.py csv` 35/35 killed.

- `vectors/data/csv-spectrum/package.json` sha256 `d3f24d28af40679704bcb13e0c99078ecf0860619325f1cb27c858d72c64a5c2`
- `vectors/data/csv-spectrum/csvs/*.csv` and `json/*.json`: 24 files, unchanged from the tarball above

## num.linalg, num.special, num.stats (`lib/num/`)

Floating point: rce 0.1.3 `f64` is IEEE 754 binary64 with correctly
rounded + - * / and sqrt and no FMA contraction (spec 6.3 N3). Every
reduction runs in one fixed, documented order, so a result is the same
bits on every target.

num.linalg (327 checks, `scripts/gen/linalg_gen.py`, `linalg_gen2.py`):
- 244 checks bit for bit against Debian Reference BLAS/LAPACK 3.12.1-6
  (`libblas3`, `liblapack3`, loaded by ctypes in `scripts/ref/reflapack.py`,
  sha256 pinned; LAPACK sources from https://github.com/Reference-LAPACK/lapack
  tag v3.12.1): ddot, daxpy, dnrm2, dasum, idamax, dgemv, dger, dtrsm,
  dgemm, dgetf2 and dgetrs, dpotf2 and dpotrs, dgeqr2, dorg2r, dorm2r, dgels.
  For n <= 32 the blocked drivers (dgetrf, dpotrf, dgeqrf, dgehrd, dhseqr)
  take these unblocked paths, so the comparison is exact.
- 83 eigen and SVD checks: 30 dgeev eigenvalue sets bit for bit, and
  dsyev/dgesvd style tests with the LAPACK test ratios (residual and
  orthogonality) below 30, the LAPACK pass threshold.

num.special (3,421 checks, `scripts/gen/special_gen.py`): exp and log are
the correctly rounded `math.exp` and `math.log`, re-exported as `cr_exp` and
`cr_log` (521 checks bit for bit against mpmath 1.4.1, with the subnormal
results rounded exactly on the 2^-1074 grid, since mpmath's float() rounds
twice there). The FDLIBM ports stay as `exp_fdlibm` and `log_fdlibm` under
erf, erfc and lgamma: 1,897 checks bit for bit
against FDLIBM 5.3 (netlib, sources and SHA256SUMS in `scripts/ref/fdlibm`,
built with gcc -O0 -ffp-contract=off) and 1,524 within 1 ulp of the
correctly rounded value from mpmath 1.4.1 at 200 bits (exp and log: equal). FDLIBM notice:
"Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
Developed at SunSoft, a Sun Microsystems, Inc. business. Permission to use,
copy, modify, and distribute this software is freely granted, provided that
this notice is preserved."

num.stats (1,901 checks):
- NIST Statistical Reference Datasets (https://www.itl.nist.gov/div898/strd/),
  all 31 sets copied unchanged into `vectors/num/nist` (SHA256SUMS there):
  univariate (PiDigits, Lottery, Lew, Mavro, Michelso, NumAcc1-4), linear
  least squares (Norris, Pontius, NoInt1-2, Filip, Longley, Wampler1-5),
  one-way ANOVA (SiRstv, SmLs01-09, AtmWtAg). `scripts/gen/stats_nist_gen.py`
  writes 546 checks. Each certified value is checked twice by log relative
  error (LRE): against the exact answer for the data as rounded to binary64
  (mpmath at 400 bits), LRE >= 12, and against the NIST certified value,
  LRE >= min(12, L - 0.5) where L is the LRE of that binary64-exact answer
  (NumAcc3/4, Filip and SmLs04-09 lose up to 9 digits in the decimal to
  binary rounding of their inputs alone). Measured: every value has LRE
  >= 13.7 against the binary64-exact answer, including all Filip
  coefficients and standard errors.
- SciPy 1.18.1 and NumPy 2.5.4 (`scripts/gen/stats_scipy_gen.py`, 1,355
  checks): pdf, cdf, sf and quantiles of the normal (referenced to mpmath
  1.4.1 erfc at 60 digits, since `scipy.stats.norm.sf` has 5.7e-14 relative
  error at x = 30), t, chi2, gamma, beta, F, exponential, binomial and
  Poisson families with relative tolerances per family; `numpy.quantile`
  for all nine methods bit for bit; `ttest_1samp`, `ttest_ind` (pooled and
  Welch), `ttest_rel`, `chisquare`, `chi2_contingency` (with and without
  Yates), `pearsonr`, `f_oneway`, `skew`, `kurtosis` to 1e-12.
- Variates: the draws of a `rand.det` stream computed by an independent
  Python ChaCha20 (checked against the RFC 8439 2.3.2 block vector); 80
  uniforms bit for bit, normal and exponential variates against
  `scipy.special.ndtri` and `numpy.log` of the same uniforms. The variate
  functions take the integer `det.bits(r, k)`, not the generator, because
  `math` and `rand.det` both declare a public `split` and so cannot share a
  package yet.
- Algorithms and their sources: C. Loader, "Fast and Accurate Computation
  of Binomial Probabilities" (2000) for stirlerr, bd0 and the saddle-point
  densities; M. J. Wichura, Algorithm AS 241 (Applied Statistics 37, 1988)
  for the normal quantile (coefficients from the MIT-licensed `asa241.c`);
  A. Bjorck, "Iterative refinement of linear least squares solutions I"
  (BIT 7, 1967) for least squares; J. R. Shewchuk (1997) exact summation;
  T. J. Dekker (1971) exact products; Hyndman and Fan (1996) quantile types.

Mutation testing: `scripts/mut/num.py` 12/12 killed (the two left-out equivalent mutants are listed in its header).

Open gaps: eigenvectors of nonsymmetric matrices are not provided; above
n = 32 the results stay deterministic but are no longer bit-identical to
the blocked LAPACK drivers; log1p, expm1, erf, erfc and lgamma are FDLIBM
(faithful, within 1 ulp), not correctly rounded, until `math` has them; far tails
with large parameters (t with df = 1e5, binomial and Poisson pmf near 1e-300)
reach relative errors of 3e-12 and 3e-13, from the rounding of exp arguments
near 700.
