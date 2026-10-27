# RCE standard host libraries

Cryptography written in pure RCE, for programs that run on the RCE host
runtime. Every function is deterministic: randomness never comes from
the libraries, but enters a program as logged `Random` results, so a
run replays bit for bit.

| File | Contents |
|---|---|
| `lib/crypto.rce` | SHA-256, HMAC, HKDF, X25519, Ed25519, ChaCha20 with an HMAC-SHA-256 tag (AEAD), byte and hex helpers |
| `lib/pq.rce` | Keccak (SHA3-256, SHA3-512, SHAKE128, SHAKE256 of FIPS 202), ML-KEM-768 (FIPS 203) and PQXDH |
| `lib/hpke.rce` | AES-128-GCM (FIPS 197, SP 800-38D) and HPKE base mode (RFC 9180) with DHKEM(X25519, HKDF-SHA256) |
| `lib/mls.rce` | MLS (RFC 9420) cipher suite 1: ratchet tree math, the key schedule, the exporter, crypto basics |
| `lib/core/exact.rce` | `core.exact`: big integers (base 2^24 limbs), rationals, unbounded decimals with explicit rounding modes, binary fixed-point |
| `lib/core/bytes.rce` | `core.bytes`: byte buffers as `List[i64]`: big/little-endian integers, strict hex, slice, compare, xor, find |
| `lib/col/col.rce` | `col`: deque, binary heap, ordered map and set (sorted, canonical order), FNV-1a fixed-hash map with insertion-order iteration |
| `lib/col/cache.rce` | `col.cache`: LRU, LFU and TTL caches over text keys (cachetools semantics; TTL expiry by an instant the caller passes in, never a clock read), memoization keyed by the SHA-256 of the canonical JSON (6.1.1) of the arguments (functools.lru_cache counters), and `CVal`, Form values with canonical JSON (whole or in parts), hash and canonical order |
| `lib/enc/enc.rce` | `enc`: base64 (RFC 4648), LEB128 varints and zigzag, deterministic CBOR encode/strict decode (RFC 8949 4.2.1), JCS (RFC 8785, integer numbers) |
| `lib/text/utf8.rce` | `text.utf8`: UTF-8 validation (Unicode Table 3-7), strict and U+FFFD-substituting decode, encode, code points and UTF-16 of text |
| `lib/time/time.rce` | `time`: durations and instants (i64 ns, logged time only), proleptic Gregorian civil dates (Hinnant), weekdays, ISO weeks, month arithmetic, RFC 3339 format/parse with fixed offsets |
| `lib/hash/blake3.rce` | `hash.blake3`: BLAKE3 hash, keyed hash, derive_key and extendable output (SHA-2 stays in `crypto`, SHA-3 in `pq`) |
| `lib/rand/det.rce` | `rand.det`: seeded ChaCha20 (RFC 8439) streams: words, bits, unbiased `below`/`range`, bytes, shuffle, split; fresh seeds only through `from_entropy`, which takes a `Cap[Entropy]` (`# rce-version: 0.8`) |
| `lib/fmt/fmt.rce` | `fmt`: locale-free integer text in radixes 2..36 (strict parse, overflow refused), padding and digit grouping, `{}`/`{N}` templates, and a linear-time regex (Thompson NFA run as a Pike VM: classes, `\d\w\s`, `* + ? {m,n}`, alternation, groups, anchors) |
| `lib/energy/energy.rce` | `energy`: JCP v0.3 `jcp/2` receipts (JCR-1): canonical core, blake3 content id, COSE_Sign1 (EdDSA) seal and verify (checked against the JCP conformance/v1 vectors), did:key for Ed25519, strict decoding under the profile a receipt names (`jcp/1` opens, never minted), replay of a receipt stream into budget state and per-worker tallies; uses `enc`, `hash.blake3` and `crypto` |
| `lib/io/io.rce` | `io`: request/result patterns over host adapters (HC-3, HC-10): result well-formedness, a request table pairing each `<Kind>Done` with the one request it answers (strays counted), a line reader over received chunks (CRLF, eof), a writer that cuts at character boundaries and resends after partial writes, and the `--fs` adapter path rule; no effects of its own |
| `lib/effect/effect.rce` | `effect`: the effect-context table of spec 5.2 as data (`allows`, `may_call`, `may_spawn`; checked against the rce checker by `tests/effect/effect_probes.py`), effects as data (`Plan[C]` built by pure code, emitted by the step), a recorded handler (Appendix D.4: results keyed by SHA-256 of the engine's canonical JSON, misses escalate with the engine's reason), backoff, retry and deadlines on logical ticks; uses `enc` |
| `lib/net/url.rce` | `net.url`: the WHATWG URL basic parser (IPv4, IPv6, opaque and ASCII-domain hosts, file URLs, percent-encode sets), serialization and the URL API getters; non-ASCII domains through net.idna (domain to ASCII) |
| `lib/net/idna.rce` | `net.idna`: UTS #46 (Unicode 16.0.0) ToASCII and ToUnicode: IdnaMappingTable mapping, NFC/NFD (UAX #15), ContextJ (RFC 5892 A.1, A.2), bidi rules (RFC 5893), Punycode (RFC 3492); data generated from pinned, sha256-checked Unicode files |
| `lib/net/http.rce` | `net.http`: HTTP/1.1 (RFC 9112) request and response parse and serialize over bytes, message body framing, chunked coding with trailers; strict on bare CR/LF, obs-fold, CL with TE and conflicting lengths |
| `lib/net/dns.rce` | `net.dns`: DNS messages (RFC 1035): encode and decode, presentation-form names with escapes, compression pointers followed only backwards, record data with names expanded, query builder |
| `lib/os/path.rce` | `os.path`: lexical path algebra for POSIX and Windows (drives, UNC, `\\?\` device paths; split, splitext, splitroot, join, normpath, relpath with an explicit cwd, commonpath, isreserved) matching CPython 3.13 posixpath and ntpath, and WASI preopen-relative resolution that never leaves the granted directory |
| `lib/os/env.rce` | `os.env`: environment entries and lookup with POSIX and Windows rules (first match wins, `=C:` keys, case-insensitive Windows keys), pure set/unset, the granted and canonical snapshot a host admits as one logged event, Go `os.Expand`, and the Windows command line codec (CommandLineToArgvW parsing, CreateProcessW quoting) |
| `lib/os/perm.rce` | `os.perm`: mode bits and `ls`-style filemode, GNU chmod symbolic and octal modes (gnulib modechange port, including umask and the directory setuid/setgid rules), the POSIX access check, Windows SIDs (text and binary) and DACL access checks, Landlock rights per ABI with layered rule evaluation, and the grant-to-mode/Landlock mapping |
| `lib/os/clock.rce` | `os.clock`: Rust `Duration` at full u64 range (constructors, checked and saturating arithmetic, Debug formatting with precision, width and alignment), the unix `Instant`/`SystemTime` point type with checked arithmetic and `duration_since`, wasi:clocks@0.2.8 instant and datetime values, monotonic admission of logged readings, and a pure timer queue that fires due timers in deadline then subscribe order |
| `lib/data/events.rce` | `data.events` (survey M60 `data.event`; `event` is an rce keyword): CloudEvents 1.0: the JSON event format and batches, the HTTP binding (binary and structured modes, percent-encoded header values, quoted-string unescaping) and the Kafka binding (`ce_` headers, `partitionkey` key mapping); validation of attributes; an RFC 8259 JSON parser (`Jv`, numbers kept as lexemes) shared by the other `data` modules |
| `lib/ui/text.rce` | `ui.text`: Unicode 16.0 text layout data and rules: grapheme, word and sentence boundaries (UAX #29), line break opportunities with mandatory breaks (UAX #14), bidi embedding levels and visual order (UAX #9, with isolates and paired brackets), and terminal cell width per cluster; property tables generated from the pinned UCD files |
| `lib/data/feed.rce` | `data.feed` (survey M59): a change feed as a hash-chained log (D log spec 6.1 entry hash, resume token = height + digest, verify, resume_at); the MongoDB change-stream driver rules (resumable errors, cached resume token, resume position, split events); CouchDB/PouchDB replication checkpoints (compareReplicationLogs, history of 5); an admission inbox (ack only after admission, dedupe inside a window, budget and overflow) |
| `lib/ui/input.rce` | `ui.input`: input event values and the terminal byte decoder. Covers keys with modifiers, kind and lock state, X10/rxvt/SGR mouse, focus, bracketed paste, cursor position and kitty keyboard flags (a port of crossterm 0.29's Unix parser and its stream loop), and the UI Events `code` and `key` value tables. Also maps decoded keys to DOM KeyboardEvent values, and mouse to PointerEvent (button/buttons) and WheelEvent. Includes composition order, resize and clipboard records |
| `lib/data/cdc.rce` | `data.cdc` (survey M61): one `Change` record from Postgres pgoutput v1 (binary decoder, transactions, TOAST, truncate, messages) and wal2json formats 1/2, Debezium envelopes, MongoDB change events (Extended JSON, resume-token time, updateDescription application), CouchDB `_changes` (normal and continuous), SQLite changesets/patchsets, Redis streams over RESP2/3 and NATS JetStream frames, acks and PubAck |
| `lib/ui/view.rce` | `ui.view` (survey M49): the view tree as a flat arena (tag, key, sorted attributes, text, ordered children); `diff`/`apply` between trees with the fewest child removals, creations and moves (LCS matching, LIS moves; checked against breadth-first search) and `diff_vue`, Vue 3/Inferno reconciliation for parity with Vue's operation logs; deterministic CBOR encoding, BLAKE3 view digest and frame record digest for backend parity |
| `lib/math/math.rce` | `math`: correctly rounded binary64 `cbrt`, `hypot`, `exp`, `log`, `log2`, `log10`, `sin`, `cos`, `tan`, `atan2`, `pow`, `atan`, `asin`, `acos`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `expm1`, `log1p`, `exp2`, `exp10`, `sinpi`, `cospi`, `tanpi`, `erf`, `erfc`, `tgamma` and `lgamma` (full range: Payne-Hanek reduction with 4200 bits of 2/pi) (plus `sqrt`), written in RCE with only + - * / and sqrt and integers, so every emit target gives the same bits. Error-free transformations (TwoSum, Dekker/Veltkamp TwoProduct) and Shewchuk expansions decide algebraic roundings exactly; transcendentals use Ziv's strategy over exact integers; algorithms and worst cases follow CORE-MATH (MIT). `scripts/math_parity.sh` checks Rust = Python = JS |
| `lib/data/sync.rce` | `data.sync` (survey M62): JSON Pointer (RFC 6901), JSON Patch (RFC 6902) and Merge Patch (RFC 7396), snapshot diff to a patch; CRDTs (Automerge map registers with Lamport op ids and conflicts, G/PN-counter, LWW register, add-wins OR-set); a replayable mutation log with Replicache push/pull (dedupe by client and mutation id, optimistic pending mutations rebased on pull); subscriptions that push only the changed rows of a filtered table (delete + insert) |
| `lib/text/xml.rce` | `text.xml`: a non-validating XML 1.0 reader (elements, attributes, namespaces, character and entity references, CDATA, comments and PIs skipped) into a flat node arena, with text content, tag search and serialization (with or without namespace declarations); reads every SCXML IRP document as lxml does |
| `lib/ui/ecma.rce` | `ui.ecma`: the ECMAScript subset of SCXML's ECMAScript data model (ECMA-262 5.1 semantics on that subset; integer numbers plus NaN). Covers literals, functions and scopes, member/index/call/new, the operator set the IRP uses, abstract and strict equality, var/function/return/if statements, Array and String built-ins, literal-character regex search, and DOM nodes from `text.xml` documents. Agrees with Node V8 on 495 cases |
| `lib/ui/states.rce` | `ui.states` (survey M53; `state` is an rce keyword): statecharts as the SCXML Appendix D interpreter: compound, parallel and final states, shallow and deep history, eventless and internal transitions, executable content, send/cancel/raise, delays, invoke with autoforward/finalize, donedata, early and late binding, Basic HTTP Event I/O; passes the whole W3C SCXML IRP suite (200 tests), and an XState 5 mode matches XState 5.33.2 on 300 live charts |
| `lib/num/linalg.rce` | `num.linalg`: dense column-major matrices; ports of Reference BLAS/LAPACK 3.12.1 (BLAS 1/2/3 kernels, dgetf2/dgetrs LU, dpotf2 Cholesky, dgeqr2/dorg2r/dorm2r QR, the dgels least squares path, dgebal/dgehd2/dlahqr eigenvalues as dgeev), cyclic Jacobi `eigh` and one-sided Jacobi `svd`; fixed reduction order, no FMA, so results are bit-identical on every target |
| `lib/num/special.rce` | `num.special`: binary64 bit access in pure RCE, correctly rounded exp and log from `math` (`cr_exp`, `cr_log`), FDLIBM 5.3 ports of log1p, expm1, erf, erfc and lgamma |
| `lib/num/stats.rce` | `num.stats`: exact-sum descriptive statistics, NumPy quantile methods 1-9, normal/t/chi2/F/gamma/beta/exponential/binomial/Poisson pdf, cdf, sf and quantile (Loader saddle-point densities, AS 241), uniform/normal/exponential variates from `rand.det` draws, least squares with Bjorck iterative refinement, one-way ANOVA, t-tests, chi-square tests, Pearson test |

Foundation modules live under `lib/<area>/` (a module that needs others names them on one `# uses: a.b, c.d` line, parsed strictly by `scripts/uses.py`, and is checked as a package with them); their vectors and property
sweeps are in `tests/<area>/<name>_checks.rce`, which `scripts/check.sh`
appends to the module. A program uses a foundation module by name alone
(`use core.exact`): every module is built into `rce`, as the next sections
state.

## The library is pinned by hash

`lib/MANIFEST.json` is the manifest of specification F.5.3: every module of
`lib/` with the SHA-256 of its canonical text, written by `rce manifest lib`
and verified by `rce manifest-check lib/MANIFEST.json` in `scripts/check.sh`.
There is no version number: a module is named by its hash, a change to a
module changes the manifest in the same commit. The checker carries this
manifest with the library: `rce lib` ends with the SHA-256 of the manifest's
canonical text and reports any module whose text differs from its pin, and a
program's manifest pins the library as the one dependency `stdlib` by that
hash (`rce manifest` writes it; `rce manifest-check` verifies it).

## Built into the checker

Every module of `lib/` is compiled into `rce` (specification 4.1): a program
writes `use enc` or `use data.events` and the checker has it, in `check`,
`run`, `test`, `trace`, `emit`, `dim` and `host`; `rce lib` lists the
modules and `rce lib NAME` prints one. The `#!lib NAME` directive line is
honoured as well (the library's items in front, unqualified). Nothing is
installed or selected: the library is part of the language, and a
change to it is a new hash in `lib/MANIFEST.json` in the same commit.

## What the checks need

| Check | Needs | On the reference checker |
|---|---|---|
| `lib/crypto.rce`, `enc`, `hash`, num, math, text, xml, utf8 and the other foundation modules | nothing | pass |
| `pq` (6 checks), `hpke` (3), `mls` (2) | `crypto` in front of the module, by the `#!lib crypto` line | pass |
| `energy` library (`lib/energy/energy.rce`, 55 checks) | `crypto` in front, as above | pass |
| `tests/energy/energy_host.rce` (15 events) | nothing | pass; `Seal0` seals a receipt and `Seal0Check` compares it with the expected envelope |
| `tests/os/pkg_checks_10.rce` | nothing | pass; the long-identifier row uses 560,000 characters because a text is limited to 1,000,000 elements (spec N9) |

- `rce host run` and `rce host serve` run the host steps, and every host run is validated by replay with the adapters disconnected.
- `scripts/check.sh` probes for the `#!lib NAME` directive. A checker without it gets `lib/crypto.rce` (without its `module` and `check` lines) appended by `scripts/libexpand.py` to the modules that name it. Set `RCE_NO_LIB=1` or `0` to force either path.
- `ed_pubkey`, `ed_sign` and `ed_verify` are ordinary functions in `lib/crypto.rce`. Each input has the fuel of 6.7 on every host, and HC-7 requires a host to charge fuel as the reference does; proposal XF-6 (`spec/proposals/one-library-every-fabric.md`) states the same for native library code, so that a binary with native `ed_*` and the reference write the same log for the same inputs.

`pq`, `hpke` and `mls` build on `crypto`. The libraries are reference
code for correctness and replay. They are **not constant time**.

## Use

A foundation module is used by name (`use data.events`). `crypto`, `pq`,
`hpke` and `mls` declare no `pub` items and are written to be read in front
of a program, so a program names them on a directive line:

```
module my_app
#!lib hpke
```

`rce` then places `crypto` and `hpke` in front of the program, without their `module` and
`check` lines. `scripts/check.sh` runs every library's own checks
(`ONLY="col core"` restricts it to those `lib/<area>/` directories, with
`crypto` naming the top-level crypto, pq, hpke and mls modules; CI splits
the libraries over parallel jobs this way). Runnable programs that use them are in
[programs/ecosystem](../programs/ecosystem),
and the specification describes them in Appendix F of the
[RCE book](https://openie-dev.github.io/rce-book/ch17-media-calls-e2ee.html).

Public-key steps (X25519, ML-KEM) are heavy: run one per input on the
host (`rce host run`), where each input has its own fuel.

## Test vectors

`vectors/` holds the official vectors the libraries are checked against;
see `vectors/SOURCE.md`.

## License

MIT OR Apache-2.0; see `LICENSE-MIT`, `LICENSE-APACHE` and `COPYRIGHT`.
The vectors keep their upstream terms.

## Renamed names (one language)

rce has one language and one globally reserved word list, so names that the list reserves cannot be declared. `scripts/rename-reserved.py` with `scripts/rename-reserved.map` renamed them in this repository, in one deterministic pass over word tokens outside strings and comments:

| Was | Is | Where |
|---|---|---|
| `Json` (kind) | `JsonDoc` | `enc` |
| `Url` (struct) | `UrlParts` | `net.url` |
| `label`, `node`, `group`, `child`, `budget`, `restart`, `replicated` | `lbl`, `nd`, `grp`, `chd`, `bud`, `rst`, `repl` | parameters, fields and locals |
| `session`, `protocol`, `par`, `offer`, `choose` | `sess`, `proto`, `parl`, `offr`, `chse` | names |
| `ct` | `ctext` | names |

Field names and keyword arguments were renamed together with their uses. Programs that import `enc.Json` or `net.url.Url` must use the new names.

## Generators not re-run for the one-language migration (unverified)

The one-language pass edited the generated files and the templates in
`scripts/gen/` together, and `scripts/one-language.py` verifies the committed
output. These generators were not re-run end to end after the pass, so their
output is checked only by `scripts/check-all.sh` and by the verifier, not by
regeneration: blake3, cdc*, clock, dns, http, energy, events, exact, feed, fmt,
fs, idna_data, input, irp, math, pkg_lock, rand, signal (needs a gcc reference),
svc, target, text_icu, time, url, view, xml. Regenerate each on Linux (kernel
6.12 matches the committed headers) before relying on a template change.
