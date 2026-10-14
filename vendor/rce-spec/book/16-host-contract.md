## Appendix E. The host contract (normative)

This appendix states what a host must do to run an RCE program. A *host*
is everything outside δ and λ: the code that admits events, appends the
log D (6.1), runs λ's outputs as effects, answers foreign calls and talks
to devices. The language defines meaning; the host is conforming when
every run it performs is the fold of its own log (T3) and the log it
leaves on storage can be replayed with every adapter disconnected (T37).
The appendix applies to every version (C.15): it constrains hosts and the
format of D, and it gives no valid program a different meaning.

The rules were derived from a survey of operating systems, embedded
kernels, browsers, language runtimes, replay systems, storage engines and
compute fabrics. A primitive is in the contract only when (a) every
surveyed target can supply it and (b) its result is a pure function of
logged data or is itself logged. What fails (b), because it depends on
timing, ordering, hardware or another process, enters as an event, as
5.15 already requires. Every target supplies: pure computation on bytes
with integer operations and binary64 round-to-nearest-even, append-only
storage, SHA-256 as pure code, one input queue and one output queue, and
a monotonic counter read only by the admitter. Everything else is an
adapter.

### E.1 Terms

- **Admitter.** The one component that orders inputs into E and appends
  entries to D.
- **Adapter.** Host code that turns one outside source (a keyboard, a
  socket, a file, a clock, a GPU, a foreign library) into events, or one
  output request into an effect and a result event.
- **Host entry.** An entry of D whose event type starts with `host.`
  (E.5). No RCE identifier contains a dot, so no program declares such an
  event, and δ is the identity on it: δ(s, e) = (s, []) and the entry's
  `state_hash` equals the previous entry's (or that of s₀ for the first).
  The fold law of 6.1 is unchanged.
- **Durable.** An entry is durable when the barrier of E.3 for its
  storage returned success.

### E.2 Admission, time and authority (HC-1 to HC-9)

- **HC-1 One admitter.** A host has exactly one admitter per program
  instance. Every input reaches δ only after the admitter appended it to
  D, in D's order.
- **HC-2 Only the admitter reads clocks.** Wall time and monotonic time
  enter only as events (`Tick(n)`, and `host.WallTime` when an adapter
  supplies it). Time inside the language is logical (6.3 N11, C.11.12).
- **HC-3 Typed output requests.** Each effect requested by λ (a write, a
  send, a draw, a foreign call) has exactly one result event: delivered,
  failed, timed out or unreachable. A race inside the adapter (for
  example a file checked and then opened) is resolved by the adapter;
  the program sees one atomic outcome.
- **HC-4 Foreign code.** Foreign code runs only through 5.11, with its
  arguments copied or borrowed for the call only, under fuel, and its
  result or trap logged as `Foreign(f, arguments, result)`. An isolated
  component boundary (a WebAssembly component per call) is the
  recommended adapter; it does not change the rule.
- **HC-5 Capabilities.** Authority comes only from capabilities minted by
  logged events (5.16 `Cap[A]` in a version 0.8 module). The host holds no
  ambient authority on the program's behalf and enters the strictest
  sandbox its platform offers at start (Landlock, Capsicum `cap_enter`,
  seccomp, an app sandbox, a WASI world). The program loses nothing,
  since 5.15 gives it no ambient authority.
- **HC-6 Tick scheduler.** Concurrency runs under the tick semantics of
  6.8. A host may execute tasks in parallel only when the result equals
  the tick semantics (T1, T21).
- **HC-7 Numerics and limits.** The host reproduces 6.3 bit for bit and
  enforces fuel, SIZE_LIMIT and VALUE_LIMIT (6.7, N9) as the reference
  does, on every run.
- **HC-8 Replay.** A host offers a replay mode that reads D, disconnects
  every adapter and re-derives every state hash. Replay reads no clock,
  file, network, device or environment (T37).
- **HC-9 Priority admission.** Interrupt sources carry a priority in
  0..255 (T25, C.11.12). The admitter orders simultaneous inputs by (priority,
  arrival) and logs the order it chose. A real-time bound holds only on a
  kernel with a priority scheduler; a general-purpose OS gives none.

### E.3 Storage: durable append (HS-1 to HS-5)

- **HS-1 D is the base truth.** A host stores one primary artefact: D.
  Snapshots and indexes are caches derived from it (HS-4, E.6).
- **HS-2 Durable append.** An entry is durable only after the barrier for
  its storage succeeded. The admitter reports no output of a step as
  durable before the step's entry is durable.

| Storage | Append barrier | New file or rename barrier |
|---|---|---|
| Linux | `write` then `fdatasync` (or `fsync`) on the segment | `fsync` of the directory after create or rename |
| macOS, iOS | `fcntl(F_FULLFSYNC)`; plain `fsync` is not a barrier, since the drive may reorder and delay writes | `F_FULLFSYNC` on the directory |
| Windows | `FILE_FLAG_WRITE_THROUGH` with `FILE_FLAG_NO_BUFFERING`, or `FlushFileBuffers` | `MoveFileEx` with `MOVEFILE_WRITE_THROUGH` |
| FreeBSD, other POSIX | `fsync` on the segment | `fsync` of the directory |
| WASI, browser, no file system | none | the host declares `volatile` in `host.StorageCaps` (E.5); no entry is reported durable |

  A host that cannot meet its row declares `volatile`. A setting that
  weakens a barrier (SQLite `synchronous=OFF`, RocksDB `sync=false`,
  LMDB `MDB_NOSYNC`) is not a conforming D store.
- **HS-3 A failed barrier is fatal.** After a failed barrier the file
  system may already have discarded the failed pages and marked them
  clean, so a retry can report success for data that is lost. The
  admitter therefore stops: it admits nothing more, retries nothing,
  emits `host.StorageFault` to the outside when it still can, and the
  next start recovers by HS-4. It never retries and continues (T39).
- **HS-4 Verify on open.** On open, the host checks the chain of D from
  the genesis hash (64 zeros): each entry's `prev_hash` is the previous
  entry's `hash` and its `hash` is the SHA-256 of the canonical JSON of
  {event, prev_hash, seq, state_hash} (6.1). D is cut after the longest
  prefix that verifies, and the host logs `host.Truncated` with the number
  of entries dropped. A torn tail after a crash, a corrupted entry and a
  misdirected write are detected, never folded into state (T38). Replay
  of the kept prefix then re-derives each `state_hash` (HC-8); a mismatch
  is a host fault (HS-3).
- **HS-5 Snapshots are caches.** A snapshot is (seq, state_hash, bytes).
  It is used only when the hash of its bytes equals the `state_hash` of D
  at seq; otherwise it is discarded. Loading one changes no meaning.

### E.4 Inputs are events (HC-10)

**HC-10.** Every value that enters a run is an event or a logged answer
in D. This includes key, pointer, resize and focus input; `Tick` and
`host.WallTime`; every file, network, PTY, audio and GPU readback result;
foreign results and traps; `select nondet` choices; Delivered, TimedOut
and Unreachable; capability minting; lifecycle (suspend, resume, process
kill); random bytes; and the host entries of E.5. An input that is not
in D does not exist for the program.

### E.5 Host entries

A host entry is {type: `host.<Name>`, payload}. Its payload is canonical
JSON. The host entries are:

| Type | Payload | When |
|---|---|---|
| `host.StorageCaps` | barrier (E.3 row name, or `volatile`) | at open |
| `host.Truncated` | dropped (u64) | at open, when HS-4 cut D |
| `host.Synced` | upto (seq) | after a barrier succeeded; every entry up to seq is durable |
| `host.StorageFault` | seq, error text | output only: written to the outside, never appended, since the admitter has stopped |
| `host.FabricCaps` | the fabric record of E.7 | at start and on hot-plug |
| `host.Calibrated` | fabric id, op class, overhead, throughput | after a calibration run |
| `host.Exact` | fabric id, op class, probe digest, reference digest | after a probe (HC-12) |
| `host.Placed` | op id, fabric id | when placement is decided (HC-11) |
| `host.WallTime` | nanoseconds since the Unix epoch | when an adapter supplies wall time |
| `host.Energy` | source, basis, joules | when a counter is read (D.11) |
| `host.Bind` | host result kind, program event name | at start (E.8) |

A program receives the content of a host entry only through a binding
(E.8): the admitter then appends, after the host entry, an ordinary event
the program declares, with the same payload. δ itself never reads a host
entry.

### E.6 Fabrics and placement (HC-11 to HC-14)

- **HC-11 Fabric discovery as events.** At start and on hot-plug the host
  appends `host.FabricCaps`: ISA flags, each GPU adapter's features and
  float controls (rounding mode, denormal preservation, signed zero, inf
  and NaN preservation per width), NPU receipts, memory topology and
  which counters can be read. Replay reads the same record, so every
  decision taken from it is a function of D.
- **HC-12 Automatic placement.** The engine places each lane-wise array
  operation, `for … par` body and reduction itself, from the cost table
  of C.6.4 extended with per-fabric overhead and throughput (time =
  overhead + work / throughput), whose calibration results are
  `host.Calibrated` entries. A program needs no annotation. `with device`
  (5.13) and device kernels stay hints that never change meaning.
  Placement is a function of the host entries of D, so it is the same on
  replay (T37).
- **HC-13 Bit-exactness gate.** A fabric computes a value that reaches
  state only for an operation on its logged exact list: integer
  operations; IEEE add, sub, mul, div and sqrt with round-to-nearest-even,
  no contraction and no flush to zero; and reductions in the fixed tree of
  T18. An operation enters the list by a probe whose digest equals the
  CPU reference digest, logged as `host.Exact`. Everything else
  (transcendentals, vendor BLAS, f16, NPU, any WGSL float operation, whose
  rounding WGSL does not specify) is either output only (λ sinks) or
  verified: recomputed on the reference, or checked against a logged
  digest per device.
- **HC-14 No performance flags.** No user-visible setting chooses
  performance. The engine derives the build profile (optimization, link
  time optimization, codegen units, target ISA from `host.FabricCaps`),
  the allocator and collection policy and the I/O backend. A knob is
  admissible only when it cannot change meaning (decision 38; 10.1). Fast
  math, floating contraction, excess precision and hash-seed
  randomization do not exist under the contract. Capabilities whose use
  changes no result are on by default; input capabilities (hardware
  random numbers, clocks, energy and thermal counters) are events; those
  that change bits are gated by HC-13.

### E.7 The fabric record

`host.FabricCaps` has the payload {cpu, gpus, npus, memory, counters}.
`cpu` lists ISA feature names; each GPU and NPU entry has an id, an API
name and the float controls as booleans; `memory` lists nodes with their
size in bytes; `counters` names the readable energy and performance
counters. Field order is canonical (6.1). A host that cannot query a
field writes `null` for it; placement then treats that fabric as absent.

### E.8 RCEdb: storage and queries written in RCE

The database of an RCE system is written in RCE. The host supplies only
durable append (E.3) and sync result events (`host.Synced`, delivered by a
binding); it supplies no query engine, index or cache whose behaviour
reaches δ.

- **HS-6 Bindings.** At start the host appends one `host.Bind` per
  result kind it delivers, naming an event the program declares (for
  example `host.Synced` to `event Synced(upto: u64)`). The program's
  checker types the bound event as any other; a payload that does not fit
  is refused at admission (5.1).
- **HS-7 Indexes are projections.** An index is a pure function of a
  prefix of D and the index definition, rebuilt by folding D and checked
  by a digest per seq. Its structure (B-tree, LSM, hash) is a placement
  choice with no effect on meaning, so a foreign store (SQLite, RocksDB,
  LMDB) may hold an index, and its durability settings do not matter for
  correctness: a lost index is refolded from D.
- **HS-8 Queries.** A query is an RCE term over state using keyed reads,
  keyed joins and scans in canonical key order (6.1, D.5). Results are in
  canonical order, never unspecified. A query states in its type whether
  it is a set or a bag. Float aggregation uses the fixed tree of T18.
  Recursive queries are not in D.5.
- **HS-9 Time travel.** The state as of seq is the fold of D up to seq
  (T3), so any past state is queryable with no storage beyond snapshots.

Example 55 (`examples/55_rcedb.rce`) is a keyed store in RCE: puts are
state writes, a scan sorts the keys (`keys()` then `sort()`) so its
order is canonical, and a bound `Synced` event moves the durable mark.

### E.9 Existing ecosystems

Code in other languages and engines stays reachable as foreign calls
(5.11, HC-4): a SQL engine, a numerical library, an operating system
service or a model runtime is called from `fn main` or a task function,
and its result is logged. Its order, its timing and its numerics are
therefore recorded and replayed, never re-executed (T32). Text sent to a
foreign SQL, HTML, URL or JSON sink is `Text[C]` (5.16), so no query or
markup is built from data.

### E.10 Theorems and conformance

- **T37 Host replay.** For every adapter oracle, replaying D with every
  adapter disconnected gives the live run's state after every entry, host
  entries leave the state unchanged, and two runs that logged the same
  host entries make the same placements.
- **T38 Verify on open.** The verified prefix of a log is a prefix of it,
  is a valid chain, equals the log when the log is valid, and a valid log
  followed by any entry that does not link is cut back to the valid log.
- **T39 Fatal barrier.** After a failed barrier the admitter appends
  nothing, whatever arrives, and the log of an admitter still running is
  exactly the entries it was given: a failed barrier is never followed by
  further appends.

Lean: `RCE/Host.lean`. Conformance: `spec/checker/tests/host_contract.rs`
checks every example log against HS-4 (intact, torn tail, a flipped
byte, a swapped pair), HC-8 and T37 (each example log re-folded with host
entries interleaved, every state hash equal, every host entry's state
hash equal to its predecessor's), HC-12 (placement from host entries
only), HS-6 to HS-9 on example 55, and that every rule of this appendix
is cited by a test.

| Rule | Source in the survey | Test |
|---|---|---|
| HC-1 to HC-3 | single admitter, clocks, typed outputs | `admitter_order_is_d_order` |
| HC-4, E.9 | foreign calls logged | `foreign_and_host_entries_replay` |
| HC-5 | capabilities minted by events | `capability_examples_check` |
| HC-6, HC-7 | tick scheduler, numerics | existing T1, T9 and 6.3 tests |
| HC-8, HC-10, T37 | replay with adapters disconnected | `foreign_and_host_entries_replay` |
| HC-9 | priority admission | `admitter_order_is_d_order` |
| HS-1 to HS-5, T38, T39 | durable append, fatal barrier, verify on open | `verify_on_open_cuts_bad_suffix`, `fatal_barrier_stops_admission` |
| HC-11 to HC-14 | fabrics, placement, exactness, no flags | `placement_is_a_function_of_host_entries` |
| HS-6 to HS-9 | RCEdb in RCE | `rcedb_example_runs` |

**Read receipt.** Which fcntl command is the append barrier on macOS? (for example F_SETFL)

<!-- body-sha256: e7177eb45e23c4e7c520c6b06164bfefe2a0991c1d6a57ee600ce63ff519ea03 -->
