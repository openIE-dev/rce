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

Foundation modules live under `lib/<area>/` (a module that needs others names them on a `# uses:` line and is checked as a package with them); their vectors and property
sweeps are in `tests/<area>/<name>_checks.rce`, which `scripts/check.sh`
appends to the module. `rce` 0.1.3 embeds only `crypto`, `pq`, `hpke` and
`mls` for `#!lib`; until a release embeds the foundation modules, give the
file to `rce` together with the program (`rce check lib/core/exact.rce app.rce`
with `use core.exact`).

`pq`, `hpke` and `mls` build on `crypto`. The libraries are reference
code for correctness and replay. They are **not constant time**.

## Use

The `rce` binary of [rce-ide](https://github.com/openIE-dev/rce-ide/releases)
0.1.3 and later embeds these files. A program names the ones it uses in a
directive line (a comment to every other tool):

```
module my_app
#!lib hpke
```

`rce` then appends `crypto` and `hpke`, without their `module` and
`check` lines. `scripts/check.sh` runs every library's own checks. Runnable programs that use them are in
[rce-examples/ecosystem](https://github.com/openIE-dev/rce-examples/tree/main/ecosystem),
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
