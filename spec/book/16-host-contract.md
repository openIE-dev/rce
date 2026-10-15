## Appendix E. The host contract (normative)

This appendix states what a host must do to run an RCE program. A *host*
is everything outside δ and λ: the code that admits events, appends the
log D (6.1), runs λ's outputs as effects, answers foreign calls and talks
to devices. The language defines meaning; the host is conforming when
every run it performs is the fold of its own log (T3) and the log it
leaves on storage can be replayed with every adapter disconnected (T37).
The appendix applies to every version (C.15): it constrains hosts and the
format of D, and it gives no valid program a different meaning.

![Host contract: five primitives, adapters, one admitter, log D, replay](https://raw.githubusercontent.com/openIE-dev/rce/main/docs/figures/host-contract.svg)

*Figure (informative). The five primitives, adapters, the single admitter (HC-1), typed results (HC-3) and replay with adapters disconnected (HC-8, T37). Source: [docs/figures](https://github.com/openIE-dev/rce/tree/main/docs/figures).*

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

- **HC-1 One admitter per term.** At any log height a program instance
  has exactly one admitter, the holder of the current term (E.2b). Every
  input reaches δ only after the admitter appended it to D, in D's order.
  The admitter is a role, not a machine: when it fails, another host takes
  the role for a higher term and continues the same D (E.2b). A host
  without replication holds term 0 for its lifetime, which is the single
  admitter of earlier versions.
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
  ambient authority on the program's behalf. This is the sandbox, and its
  contract is independent of any OS: the checker admits only programs whose
  effects go through granted capabilities, and the deterministic host
  refuses any effect without one, before any OS layer is involved. The
  same contract then has enforcement targets underneath the host, entered
  at start as strictly as the target allows: WASM (the module imports only
  the granted host functions), bare metal (no OS; the engine is the only
  host and capabilities are the whole I/O surface), Linux (Landlock and
  seccomp), macOS (Seatbelt, `sandbox_init`), Windows (AppContainer, to
  do) and other OSes (FreeBSD Capsicum `cap_enter`, OpenBSD
  pledge/unveil, iOS, Android: to do). The contract is planned to conform
  to OpenIE's Sandbox v0.1 standard (https://sandbox.transaction.science:
  S0-S7 tiers, §5 Grants, §6 energy and §7 attestation receipts) and to
  JCP v0.3 (https://jcp.transaction.science); see REMAINING-PLAN W1h-std.
  The program loses nothing,
  since 5.15 gives it no ambient authority.
- **HC-6 Concurrency and the host.** The concurrency of `fn main`
  (tasks, actors, supervisors; C.5, C.13) runs under the tick semantics
  of 6.8 in the reference run (`conc::run_main`) and its replay, which
  re-executes with the recorded `select nondet` choices. The reference
  host does not run `fn main`: it admits inputs to δ, and none of the
  task, actor or restart steps of a `fn main` run is an entry of D. A
  host that runs `fn main` must admit each scheduling decision as an
  input (a tick, a crash, a restart) so that it is in D; the reference
  host has no such scheduler, and no host may execute tasks in parallel
  unless the result equals the tick semantics (T1, T21).
- **HC-7 Numerics and limits.** The host reproduces 6.3 bit for bit and
  enforces fuel, SIZE_LIMIT and VALUE_LIMIT (6.7, N9) as the reference
  does, on every run.
- **HC-7a Fuel per input.** A host is event driven, so fuel is charged per
  admitted input: each input (one application of δ, including constructing
  its event) starts with the full fuel of 6.7, and no budget is shared
  across inputs. Every input has its own bound and a session never starves
  however long it runs. The reference trace (`rce trace`), the host and
  replay all reset fuel per input, so a refusal for fuel is a function of
  that input and the state before it, and D is the same on each.
- **HC-8 Replay.** A host offers a replay mode that reads D, disconnects
  every adapter and re-derives every state hash. Replay reads no clock,
  file, network, device or environment (T37).
- **HC-9 Priority admission.** Interrupt sources carry a priority in
  0..255 (T25, C.11.12). Inputs admitted together are ordered by priority,
  highest first. Inputs with a source and sequence number are ordered by
  the window key of HC-15. Inputs given together without source identity
  (`admit_simultaneous`) are ordered by priority, then by the bytes of
  their canonical event JSON, so the order is a function of the set of
  inputs and not of the order the caller passed them in. Either way the
  order chosen is D's order. A real-time bound holds only on a kernel with
  a priority scheduler; a general-purpose OS gives none.

### E.2a Canonical admission (HC-15 to HC-17)

The order of D must not depend on when inputs arrive. Arrival timing may
decide only which window an input lands in, and the window boundary is
defined by logical time, never by a wall clock.

- **HC-15 Windows.** Every input carries its source id (text), a
  per-source sequence number (1, 2, ... with no reuse) and the logical
  tick its source stamped on it. When the module declares
  `event Tick(n: u64)`, a window closes when a `Tick(n)` input not yet in
  D is offered: window k gets `upto = n`, holds every offered input with
  tick at most `upto` that is not already in D, and the `Tick(n)` is its
  last input. The boundary is therefore a logged fact, never a wall clock
  and never a choice of the caller. An input whose tick is at or below the
  `upto` of the last window in D is late: it is admitted in the next
  window and its `host.Input` carries `"late":true`, so whether an input
  was late is decided by D. A host whose module declares no `Tick` closes
  windows by `close_window(upto)` from its own logical time. Apart from the closing `Tick`, the order
  inside a window is the total key (priority descending, source id ascending, sequence ascending).
  So two admitters given the same window contents append the same entries
  in the same order whatever the arrival order, and two hosts given the
  same inputs per window produce byte-identical D.
- **HC-16 Exactly once.** The dedupe key is (source, seq). A (source, seq)
  already in D, or offered twice, is a duplicate and is not appended
  again. Two copies of one (source, seq) with different payloads are
  ordered by their event bytes, and the lower one is admitted, so the
  choice does not depend on arrival. An input after a gap in its source's
  sequence is held for a later window, so each source is admitted in its
  own order and nothing is skipped. A source retransmits until acked, and
  the admitter acks an input only after the barrier (HS-2) that covers
  its window succeeded. A crash before the ack leads to a retransmission,
  which HC-16 turns into a duplicate when the input did reach D.
- **HC-17 Window entries.** Closing a window appends `host.Window` {count,
  id, term, upto}, then, before each admitted input, `host.Input` {late (only when
  true), seq, source, window}. The dedupe state is the fold of these entries: on open
  and on replay the admitter rebuilds, per source, the highest seq in D. A
  `host.Input` counts only when the program entry it announces follows it,
  so a cut between the two (HS-4) admits nothing. These are host entries:
  δ never reads them and they leave the state hash unchanged.

### E.2b Replicated D and admitter failover (HC-18 to HC-20)

A host may keep D on a replicated log so that losing the admitter's
machine neither stops the program for good nor forks D. The reference
model is `spec/checker/src/repl.rs`; its rules are those of Raft as
summarised in the convergence notes below, applied to whole logs.

- **HC-18 Interface.** A replicated log offers the host the Store of E.3
  with two additions: `elect(term)`, which succeeds only with grants from
  a quorum of replicas, each of which then promises not to accept an
  append from a lower term, and returns the most up-to-date log among the
  granting quorum, ordered by (term of the last write, length); and an
  append at a term that a replica accepts only while it has promised no
  higher term. The barrier succeeds only when a quorum accepted every
  entry up to it, so a durable entry (HS-2) is a committed entry.
- **HC-19 Term fence in D.** A host that won `elect(t)` folds the log it
  received (HS-4 verification of the whole chain, then HC-8), then appends
  `host.Term` {leader, term} and makes it durable before admitting any
  input. `t` must exceed every term already in D. Every `host.Window`
  carries the term. A request emitted under an older term with no result
  in D gets one result, `failed` with `interrupted` (HC-3). The term in D
  is the fence token: an entry of D was appended by the holder of the
  most recent `host.Term` before it.
- **HC-20 Deposed admitter.** An admitter whose barrier is refused because
  a higher term was granted has had its role taken. By HS-3 it appends
  nothing further and acks nothing, so its unacked inputs are retransmitted
  to the new admitter and HC-16 removes those that did commit. Committed
  entries are never removed: the new admitter's log contains every entry
  a quorum had accepted, so D after a failover extends D before it.

Safety of HC-18 to HC-20 does not depend on clocks. A leader lease, which
would let the admitter serve reads without a quorum round, relies on
bounded clock skew and is not part of the contract.

Convergence notes. Calvin (Thomson et al., SIGMOD 2012) also fixes the
order of inputs before executing them: it "divides time into 10-millisecond
epochs", each sequencer compiles the requests of an epoch into a batch, and
schedulers interleave "(in a deterministic, round-robin manner) all
sequencers' batches for that epoch". Its epochs are wall clock intervals;
E.2a's windows are logical, and its merge key orders by priority, source
and sequence where Calvin interleaves by sequencer. Calvin also states that
"single-node sequencers ... represent potential single points of failure",
the concern HC-18 to HC-20 address. FoundationDB (Zhou et al., SIGMOD 2021)
keeps a singleton Sequencer that "assigns read and commit versions"; the
Coordinators "form a disk Paxos group" and select a ClusterController,
which recruits singletons that "are re-recruited if they fail or crash".
Its transaction processing "is divided into epochs, where each epoch
represents a generation of the transaction management system with its
unique Sequencer process", and a recovering Sequencer "locks the
coordinated states to prevent another Sequencer process from recovering at
the same time". RCE's terms play the role of those generations, recorded
in D as `host.Term`. Raft (Ongaro and Ousterhout, USENIX ATC 2014, extended
version) gives the properties HC-18 relies on: "Election Safety: at most
one leader can be elected in a given term"; "If a server receives a request
with a stale term number, it rejects the request"; and "Leader
Completeness: if a log entry is committed in a given term, then that entry
will be present in the logs of the leaders for all higher-numbered terms".
Raft notes that a heartbeat lease "would rely on timing for safety (it
assumes bounded clock skew)", which is why E.2b leaves leases out. The
reference model replicates whole logs instead of AppendEntries suffixes
and keeps `promised` in memory; a real host must persist the promise
before granting and ship suffixes with the log matching check.

### E.3### E.3 Storage: durable append (HS-1 to HS-5)

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
| `host.Window` | count, id, term, upto (tick) | when a window closes (HC-17) |
| `host.Input` | late (when true), seq, source, window | before each input of a window (HC-17) |
| `host.Term` | leader, term | when a host takes the admitter role (HC-19) |

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
- **HC-12 Placement.** Placement of a lane-wise array operation, a
  `for … par` body or a reduction is a function of the host entries of D
  (`host.Calibrated`, `host.Exact`) and of the cost table of C.6.4
  extended with per-fabric overhead and throughput (time = overhead +
  work / throughput), so it is the same on replay (T37). A program needs
  no annotation; `with device` (5.13) and device kernels stay hints that
  never change meaning. The reference host performs no automatic
  placement: it logs `host.FabricCaps`, never writes `host.Calibrated` or
  `host.Placed`, and runs every operation on the CPU. The placement
  function is specified by the conformance test
  `placement_is_a_function_of_host_entries`, which checks it on
  constructed host entries, not on a running host.
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

![Cost of strict IEEE determinism on GPU kernels and fabric dispatch model](https://raw.githubusercontent.com/openIE-dev/rce/main/docs/figures/host-survey-fig4-fabric.png)

*Figure (informative). Survey measurements behind E.6: cost of strict IEEE mode on GPU kernels and a calibrated dispatch model. Source: [docs/figures](https://github.com/openIE-dev/rce/tree/main/docs/figures).*

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

![RCEdb: log D, indexes as RCE folds, queries](https://raw.githubusercontent.com/openIE-dev/rce/main/docs/figures/rcedb.svg)

*Figure (informative). D is the one primary artefact; indexes are deterministic folds written in RCE (example 55); queries read them in a defined order. Source: [docs/figures](https://github.com/openIE-dev/rce/tree/main/docs/figures).*

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

#### E.8.1 The graph index (design refinement)

![RCEdb: the log, its graph projection, and queries](https://raw.githubusercontent.com/openIE-dev/rce/main/docs/figures/rcedb-graph.svg)

*Figure E.1. The base record is D, a hash-chained log and so a linear graph. The primary index is a typed property graph folded from a prefix of D by RCE code. Queries are typed RCE values evaluated against that graph. Source: [docs/figures](https://github.com/openIE-dev/rce/tree/main/docs/figures).*

The base record does not change: it is the hash-chained log of E.3, and every other structure is a projection of it (HS-7). Three consequences follow from rules that are already normative, and so they hold for any index shape, the graph index included:

- **As of is a fold of a prefix.** The state as of entry k is the fold of δ over the first k entries (HS-9, T37), so an "as of" query is the same query over a shorter prefix. There is no second time model to keep in step.
- **Results have one order.** A query result is an RCE value, so it has the canonical order of 7.1 (HS-8, and the 6.7 erratum for maps). No result order is left to the engine.
- **Queries terminate.** A query is an RCE function, so T2 and W-REC-1 apply: a query that the checker cannot show terminates is refused, not run with a timeout.

The rest of the refinement is proposed, not normative. It is listed in E.11 as G-1 to G-6 so that it can be adopted, changed or rejected on its own.

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
| HC-6, HC-7 | concurrency stays outside D in the reference host, numerics | existing T1, T9 and 6.3 tests |
| HC-7a | fuel per input | `tests/fuel_per_input.rs` |
| HC-8, HC-10, T37 | replay with adapters disconnected | `foreign_and_host_entries_replay` |
| HC-9 | priority admission | `admitter_order_is_d_order`, `simultaneous_order_does_not_depend_on_caller_order` |
| HC-15 to HC-17 | canonical windows, exactly once | `tests/host_admit.rs`: `two_hosts_different_arrival_orders_give_byte_identical_d`, `window_order_is_priority_then_source_then_seq`, `gap_is_held_and_admitted_once_filled`, `no_ack_without_a_durable_append`, `dedupe_state_survives_replay` |
| HC-18 to HC-20 | replicated D, term fence, failover | `admitter_failover_keeps_one_d_and_exactly_once` |
| HS-1 to HS-5, T38, T39 | durable append, fatal barrier, verify on open | `verify_on_open_cuts_bad_suffix`, `fatal_barrier_stops_admission` |
| HC-11 to HC-14 | fabrics, placement, exactness, no flags | `placement_is_a_function_of_host_entries` |
| HS-6 to HS-9 | RCEdb in RCE | `rcedb_example_runs` |

### E.11 Proposed rules (not normative)

The rules in this section are **proposed**. They come from a survey of other languages (LESSONS.md in the language-structure study) and are recorded here so that they can be reviewed against this appendix. None of them is part of the host contract yet: a conformant host need not meet them, and no test enforces them. Each one is adopted, changed or rejected by a later draft. Where a proposal restates a rule that already holds (for example 6.3 numerics or T1 determinism), it says so.

**Numerics.**

- **N-1 (proposed).** Float operations are IEEE 754 binary64 or binary32, round to nearest even. No fused multiply-add unless the program writes one. Every NaN result is the canonical NaN. Transcendental functions are a fixed reference algorithm or correctly rounded. Relation: tightens 6.3 and HC-7.
- **N-8 (proposed).** A float expression has exactly one evaluation: one rounding per operator, in source order. Contraction, reassociation and algebraic rewrites are not valid compilations, and a float rewrite is never admitted by the rewrite prover.
- **N-9 (proposed).** Every reduction is a left fold unless the program names a different fixed tree. A parallel schedule chosen by placement (HC-12) must reproduce the bits of the named tree. Thread count, fabric and batch size are never inputs to a numeric result.

**Ticks and causality.**

- **K-1 (proposed).** The step index is the only built-in time. Physical time enters only as a logged input (`host.WallTime`, HC-2).
- **K-2 (proposed).** Within a step each output field has exactly one defining expression, and evaluation order follows dependencies, not text order.
- **K-3 (proposed).** The dependency graph inside a step is acyclic; feedback flows only through the prior state S. The check is conservative: some valid programs are rejected.
- **K-4 (proposed).** Values sampled at different rates combine only through explicit sample and hold operators checked at compile time. There is no implicit buffering.
- **K-5 (proposed).** Each compiled step has a worst-case cost bound computed statically and reported by the toolchain, per fabric.
- **K-6 (proposed).** Every derived value in a step is computable by forward propagation from inputs and state. A step that is accepted only by guessing a fixpoint is rejected.
- **K-7 (proposed).** Parallel sub-steps share only read-only state and broadcast values; writes are local to the sub-step.
- **K-8 (proposed).** A merge of simultaneous writes uses a combiner from a checked library of commutative and associative operators, or one with a machine-checked proof. Float addition is not admitted.
- **K-9 (proposed).** Cancellation and suspension of a sub-step are language constructs with defined strong and weak semantics at tick boundaries, and are logged. The draft must say which clock a delayed read uses under suspension.

**Effects.**

- **E-x (proposed).** The effects emitted in one step are totally ordered by a deterministic rule, and the order is recorded in the log. No effect order is left unspecified. Relation: HC-1 already orders admission; this extends the order to outputs inside one step.

**Spec conformance.**

- **S-1 (proposed).** The reference interpreter is normative. Prose is derived from it, and a disagreement is a spec bug resolved in favor of a versioned reference. (LESSONS.md uses S-1 twice; this is the conformance rule. The other, "no behavior depends on caching, interning, allocation or scheduling", is already implied by T1 and HC-8.) The 6.7 map order was such a disagreement, settled for the text: the reference now iterates maps in canonical key order (erratum, C.15).
- **S-2 (proposed).** Every spec rule has machine-checkable conformance cases, and a runtime is conformant only if it passes all of them with bit-identical logs. E.10 follows this for the host rules.
- **S-3 (proposed).** The spec has no implementation-defined or unspecified clauses. Every case has one defined result or is a compile-time error.
- **S-4 (proposed).** Errors are values with a defined order of checking; which error a step yields is deterministic and logged.

**Hashes.**

- **C-3 (proposed).** The hash function, canonical serialization and byte order are fixed in the spec and versioned, so hashes are bit-identical across conformant runtimes. Relation: the D log already fixes SHA-256 over canonical JSON (E.3); the proposal adds a version tag to that choice.

**Verification.**

- **V-6 (proposed).** A release build replays a conformance and regression log on two independently built runtimes (for example the reference evaluator and an emitted target), and any divergence blocks the release.

**Tensor unification of primitives.**

![Proposed tensor unification: ranks 0 to 3, records over named axes, static shape check, fixed-order reductions, placement by shape and op, ragged values kept separate](https://raw.githubusercontent.com/openIE-dev/rce/main/docs/figures/tensor-unification.svg)

*Figure (informative, proposed). Tensor unification of primitives. Source: `figures/src/infographics/make_tensor.py`; [PNG](https://raw.githubusercontent.com/openIE-dev/rce/main/docs/figures/tensor-unification.png).*

- **TU-1 (proposed).** Every primitive value is an n-dimensional tensor `Tensor[elem, shape]`, where `elem` is one of the primitive element types of 5.1 and `shape` is a list of non-negative dimensions. A scalar is the rank 0 tensor `Tensor[elem, []]`. Relation: generalizes the primitive types of 5.1; a version 0.5 to 0.8 program keeps its meaning because each of its scalars is read as rank 0.
- **TU-2 (proposed).** The shape is part of the type and is checked statically. An operation whose operand shapes do not agree (for example a matrix product of `[m, k]` and `[j, n]` with k not equal to j) is a checker error, never a run-time trap. Dimensions are literals or names bound in the signature; a shape that cannot be determined statically is refused. Relation: a new W-rule in 9 when adopted.
- **TU-3 (proposed).** A record whose fields all have one element type and one shape may be read as a tensor over a named axis (`{x: f64, y: f64}` as `Tensor[f64, [axis: {x, y}]]`). Field names index that axis. A record with mixed field types stays a record.
- **TU-4 (proposed).** Every reduction over an axis follows N-9: a left fold over the axis in index order unless the program names a different fixed tree. The shape and the named tree, never the thread count or fabric, determine the bits.
- **TU-5 (proposed).** Text, maps, graphs and sum types remain separate canonical values. They are ragged and are never padded or densified into tensors; a tensor element is never one of them.
- **TU-6 (proposed).** Placement (HC-12) keys the fabric cost table by (pure operation, element type, shape). Choosing a fabric changes cost only; HC-7 and N-9 fix the result bits.

Convergence sources. Each source was read as noted; claims rest only on the passages cited.
- BQN specification, types (read in full): "An array is a rectangular collection of data. It is defined by a shape, which is a list of non-negative integer lengths, and a ravel" whose length is the product of the shape; non-arrays are atoms. This is the shape-plus-ravel model of TU-1. APL, BQN's ancestor, was not read for this proposal.
- Paszke et al., "Getting to the Point: Index Sets and Parallelism-Preserving Autodiff for Pointful Array Programming" (Dex), arXiv:2104.05372 (full text scanned, hits reviewed): arrays have types `n=>a` indexed by typed index sets (`IndexSet n`), so the index domain of an array is part of its type. This is the basis of TU-2 and of TU-3's named axes.
- Henriksen et al., "Futhark: Purely Functional GPU-Programming with Nested Parallelism and In-Place Array Updates", PLDI 2017 (full text scanned, hits reviewed): "All arrays must be regular, meaning that, all rows of an array must have the same shape"; where shape cannot be verified statically, Futhark inserts dynamic checks. TU-2 takes the regularity rule and refuses instead of checking dynamically; TU-5 keeps irregular data out of tensors.
- Julia manual, "Multi-dimensional Arrays" (read in full): "The base array type in Julia is the abstract type AbstractArray{T,N}. It is parameterized by the number of dimensions N and the element type T." Julia puts element type and rank in the type but checks sizes at run time; TU-2 goes further by putting the shape in the type.

**RCEdb graph index (E.8.1).** These refine HS-6 to HS-9. Convergence sources are listed after the rules; they are observations of convergent design, not proof that the rules are right.

- **G-1 (proposed).** The primary index of RCEdb is a typed property graph projected from the log: nodes and edges have declared RCE types (a label is a type, a property is a field), and the projection is ordinary RCE code, a fold of δ over a log prefix (HS-7).
- **G-2 (proposed).** Every node and edge carries the id of the event that created it, and the id of the event that retracted it if any. Provenance is a field of the value, so it is checked, hashed and replayed like any other field.
- **G-3 (proposed).** A query is a graph pattern written as a typed RCE value (node patterns, edge patterns, quantified paths, predicates as RCE functions), never as a string. Ill-typed patterns are refused by the checker; there is no query text to inject into (5.16). The pattern vocabulary is the read-only GQL fragment that MGQL mechanizes (node and edge atoms in five directions, label expressions with `&`, `|`, `!` and `%`, property maps, quantifiers `*`, `+`, `?`, `{i}`, `{i,j}`, pattern lists joined on shared variables), so that each RCE pattern has a stated GQL reading. A variable declared inside a quantifier has a list type outside it, and a variable under `?` is nullable.
- **G-4 (proposed).** A query states in its type whether it returns a set or a bag of rows, and the result is in canonical order (7.1). Bag is the default, as in GQL, where a match yields one row per homomorphism; a set result, `distinct` aggregates and `distinct` composite operators are opt-in. All composite operators in one composite query are the same operator with the same qualifier (GQL SR:3 of ISO section 14.2, as cited by MGQL).
- **G-5 (proposed).** The checker proves that every query terminates in one of two ways: each unbounded path quantifier is bounded by a declared length (`bound n`, 6.7) or by a restrictor that makes the match set finite (no repeated edge, no repeated node; under no repeated edge the repetition count of any quantifier is at most the edge count of the graph), or the query is recursive and is evaluated as a stratified fixpoint (negation and aggregation only over lower strata), which terminates because the graph is finite.
- **G-6 (proposed).** Tables and documents are special cases of the graph projection: a table is a node type with scalar fields and no edges, a document is a tree of nodes reached by containment edges. They are projections of the same log, not separate stores.

Convergence sources (read-ledger pass 8 of the language-structure survey, and the MGQL entry of the RCEdb GQL notes; the ISO text itself was not read, it is paywalled, and ISO section numbers below are as cited by GPML and MGQL):

- GQL, ISO/IEC 39075:2024, published 12 April 2024, is the ISO standard query language for property graphs (G-1). Its pattern-matching part (GPML, Deutsch et al., arXiv 2112.06217, section 5) states that "GPML queries must demonstrably terminate", and requires every unbounded quantifier to sit inside a restrictor (TRAIL, ACYCLIC, SIMPLE) or a selector (G-5).
- MGQL (Thimmaiah, Lin and Gligoric, PACMPL 10 OOPSLA2, article 327, 2026, doi 10.1145/3839459) is a small-step operational semantics and type system for a read-only GQL fragment, mechanized in Lean 4 (about 23,600 lines, no `sorry`), with a deterministic interpreter and a checked type checker. It is grounded in ISO/IEC 39075 section by section. It works under bag semantics (G-4), fixes the TRAIL path mode and DIFFERENT EDGES match mode because they make matching terminate, caps unbounded quantifiers at the graph's edge count (G-5), types graph elements by the graph they belong to and refines them by graph schemas (G-1), and gives quantified-path variables list types (G-3). It does not cover ORDER BY, selectors such as ANY SHORTEST, other path modes, or updates, and its validation is six LDBC SNB queries against golden results, not a comparison with shipping engines.
- Cypher (Francis et al., SIGMOD 2018) defines MATCH over tables that are bags of records, and departs from homomorphism matching by not repeating a relationship in one match, which keeps the match set finite (G-4, G-5). openCypher is the open specification of that language.
- Datalog and Soufflé (P-SOU, D-SOUR in pass 3): negation must be stratifiable, and evaluation is a fixpoint that ends when no new facts appear (G-5).
- Datomic stores "a set of immutable atomic facts called datoms", each a five-tuple that includes the transaction that asserted or retracted it, and an as-of view keeps only the datoms true at a transaction (G-2, E.8.1). XTDB keeps the full system-time history of every entity so that a past version stays queryable (E.8.1).

Where RCE differs from these sources: GQL and Cypher queries are strings and their result order is unspecified without ORDER BY; RCE patterns are typed values with one canonical order. GQL selectors such as ANY SHORTEST may pick one of several equal paths; under G-4 a selector that picks among ties must pick by canonical order.

**Read receipt.** Which fcntl command is the append barrier on macOS? (for example F_SETFL)

<!-- body-sha256: 78249361006a2a8dd651ef9a3b6f2d8338ed5cda64ba219613f7b50d8096fd7f -->
