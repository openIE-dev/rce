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

