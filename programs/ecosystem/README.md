# Ecosystem examples

Runnable examples for the libraries and apps of Appendix F of the spec
book and for RCEdb (10.10). `./run.sh all` runs each one on the host
runtime, checks its result and replays its log with every adapter
disconnected; set `RCE` to the `rce` binary.

| Example | What it shows | Checked against |
|---|---|---|
| `hpke.rce` (`run.sh hpke`) | HPKE (Hybrid Public Key Encryption, RFC 9180) base mode: key derivation, seal, open, a wrong AAD (additional authenticated data) refused | the RFC 9180 base-mode vector |
| `pqxdh.rce` (`run.sh pqxdh`) | PQXDH: three X25519 exchanges and ML-KEM-768 on both sides, one session key; redacted replay | RustCrypto `ml-kem` shared secret |
| `mls_key_schedule.rce` (`run.sh mls`, or `rce test`) | MLS (Messaging Layer Security, RFC 9420) key schedule and tree math | RFC 9420 `key-schedule.json` epoch 0, `tree-math.json` |
| `run.sh sfu` | `apps/sfu.rce` offline: subscribe, forward, stale and malformed frames; redacted replay | the SFU (selective forwarding unit) wire rules |
| `run.sh rcedb` | `rce db`: schema, index, insert, link, delete, scan, lookup, as-of, emit sqlite | time travel to before a delete |

`hpke.rce`, `pqxdh.rce` and `mls_key_schedule.rce` import their libraries
with `use` (`use hpke`, `use pq`, `use mls`, and `use crypto` where they
call `crypto` themselves) and call them by qualified name, such as
`hpke.hpke_seal`; every module of `stdlib/lib` is built into `rce`, so no
library file is needed. Their `example` lines are light ones,
because `rce test` gives a whole example one fuel budget; the
public-key steps run on the host, where each input has its own fuel
(HC-7a). Keys and seeds are constants here; a real program takes them
from logged `Random` results (HC-10).
