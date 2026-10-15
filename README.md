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

Foundation modules live under `lib/<area>/`; their vectors and property
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
