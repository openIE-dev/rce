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
- **Delivery by declaration (HD-1 to HD-4).** The host delivers each
  value as the module declares its event, matching fields by name, and
  what a declaration cannot take is kept: recorded in D by `host.Kept`
  with its reason, not delivered. The rules are stated in E.4c: HD-1
  results, HD-2 HTTP, HD-3 received bytes, HD-4 host inputs. Sections
  before E.4c cite them as E.4c HD-n.

### E.2 Admission, time and authority (HC-1 to HC-9)

- **HC-1 One admitter per term.** At any log height a program instance
  has exactly one admitter, the holder of the current term (E.2b). Every
  input reaches δ only after the admitter appended it to D, in D's order.
  The admitter is a role, not a machine: when it fails, another host takes
  the role for a higher term and continues the same D (E.2b). A host
  without replication holds term 0 for its lifetime, which is the single
  admitter of earlier versions.
- **HC-2 Only the admitter reads clocks.** Wall time and monotonic time
  enter only as events (`Tick(n)`, `host.WallTime` when an adapter
  supplies it, and the result of a `WallClock` request, HC-32). Time
  inside the language is logical (6.3 N11, C.11.12).
- **HC-3 Typed output requests.** Each effect requested by λ (a write, a
  send, a draw, a foreign call) has exactly one result event: delivered,
  failed, timed out or unreachable. A result is delivered as the module
  declares its event, and a result the module declares no event for
  is recorded in D (E.4c, HD-1). A race inside the adapter (for
  example a file checked and then opened) is resolved by the adapter;
  the program sees one atomic outcome. The answer of a model runtime
  (F.5.6) is delivered only on an event the program declares
  `extern "model" event`, and the host fills no capability for it (HC-5);
  a program that declares none receives no model text.
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
  does not perform any effect without one, before any OS layer is involved. The
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
- **HC-5a The printed target is the installed target.** The line a host
  prints when it enters an OS target is generated from the rules it
  installed, so it names each restriction at the granularity the target
  enforces and never one the target did not install. The reference
  (`sandbox.rs`) lists every address an adapter may use with how it is
  used: connected to (`net`, `peer`), bound after the target is
  entered (`net`, and a `http` address whose bind is still being
  retried), or bound before (`http`, `chan`), whose listener is then
  only accepted on. It makes the rules and the line from that one list.
  On macOS the Seatbelt profile denies all network use, then allows
  binding and inbound connections on the port of each listening grant and
  outbound connections to the port of each connecting grant, at
  `localhost` for a loopback address and at any address otherwise,
  because Seatbelt names no other address; it allows the system name
  resolver only when a grant names a host by name. The line says that
  these rules cover TCP and UDP and that the net adapter checks the exact
  address. On Linux, with Landlock ABI 4 or later, TCP bind is allowed only
  on the ports of listening grants and connect only on the ports of
  connecting grants, at any address on those ports, and the line says "by
  port"; a listener bound before entering needs no rule; below ABI 4 the
  line says that Landlock does not restrict the network. seccomp does not
  permit new IP sockets unless some grant is connected to or bound after
  entering, so with `http` alone neither the host nor a PTY child can
  open a socket; otherwise no layer restricts UDP, and the line says so.
  Landlock also handles reading a file and listing a directory: they are
  allowed only under the log directory, `/dev`, the granted roots (`fs`,
  `blob`, `media`, `wasm`) and, under `names` and only then, the files
  `/etc/passwd` and `/etc/group` (HC-33), and the line names them.
  Landlock has no right for lstat or readlink, so it restricts neither, and
  the line says so; seccomp denies none of lstat, readlink or getdents, so
  the requests of HC-30 and HC-31 run under DIR. Under `pty`, whose
  programs read the whole system, and with a grant that names a host by
  name, whose resolver reads the system's configuration after entering,
  reads are not handled, and the line says which of the two is the reason.
  The channel adapters start their threads after the target is entered, so
  Landlock covers them as well. The Seatbelt rules follow Apple's profile
  language and are still to be confirmed on a macOS machine; the two tests
  that do so run only there.
- **HC-6 Tick scheduler.** Concurrency runs under the tick semantics of
  6.8. The host runs `fn main` (tasks, actors, supervisors; C.5, C.13)
  with its scheduler as a live task whose every transition is an entry
  of D. `host.MainStart` {links, max_ticks, seed} starts the task;
  `host.MainInput` {a, b, partition, tick} logs an input to it (a link
  change), and `host.MainInput` {event, tick} feeds it a program event
  (canonical event JSON): at that tick the scheduler logs `Input(..)`
  and delivers the event to every live actor whose type has an `on`
  handler for it, in actor-id order, or logs `Unhandled(..)` when there
  is none (an input event has no addressee, so its receivers are the
  program's handlers of it, as for the steps `on E` of a Form;
  `Host::main_event`, `conc::RunCfg.events`,
  `tests/host_sched.rs` `program_events_feed_the_live_task_and_replay`,
  `tests/conc06b_run.rs` `program_events_feed_the_running_task`; the
  emitted runtimes take the same inputs as `event TICK|Name|f=v,...`,
  a field's value given by its declared type as a decimal integer,
  `true`/`false` or text, and log `Input(..)` and `Unhandled(..)` with
  `actor-log`; a malformed event is an input error, exit status 2,
  not a program trap; `tests/emit_run.rs`
  `program_events_agree_on_every_target` on the Rust, Python,
  JavaScript and Scala runtimes); `host.Step` {cause, i, node, tick, what} logs one scheduler
  step, including the arm the environment chose at each `select nondet`;
  `host.MainEnd` {choices, outcome, ticks} ends it. Program inputs to δ
  may be admitted between any two of these entries. While the task is
  live its state (the start entry, the inputs logged so far, the number of
  steps logged and a hash chain over them) is part of the state D folds
  to, and the `state_hash` of every entry covers it, so the state after
  each entry is the fold of D up to that entry and a host rebuilt from any
  prefix continues the same run. Replay re-runs the scheduler on the
  logged inputs and stops at an entry unless the step it logs is the next
  step the scheduler takes and the end agrees in outcome and choices
  (T21). An input whose tick is not after the last logged step is not admitted:
  it is not logged, and an input is also not admitted if re-running on it
  would change a step already in D, so an input never rewrites the past.
  No snapshot is taken while a task is live. A host may execute tasks in
  parallel only when the result equals the tick semantics (T1, T21).
  Reference: `Host::main_begin`, `main_input`, `main_event`,
  `main_step`, `main_end`, and `run_main` for a run whose inputs are all
  known at the start. Derivation: law 1 of E.2c makes each
  nondeterministic choice of the run an entry of D, and the requirement
  that state be the fold of D at every entry, not only between runs,
  makes the task's progress part of that state. The task's inputs are
  link changes, program events and the environment's choices, each an
  entry of D under the same causality rule. The emitted runtimes do not
  take program events (G14).
- **HC-7 Numerics and limits.** The host reproduces 6.3 bit for bit and
  enforces fuel, SIZE_LIMIT and VALUE_LIMIT (6.7, N9) as the reference
  does, on every run.
- **HC-7a Fuel per input.** A host is event driven, so fuel is charged per
  admitted input: each input (one application of δ, including constructing
  its event) starts with the full fuel of 6.7, and no budget is shared
  across inputs. Every input has its own bound and a session never starves
  however long it runs. The reference trace (`rce trace`), the host and
  replay all reset fuel per input, so a kept step for fuel is a function of
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
  Exactly once covers only inputs that carry a source and sequence
  number. A typed input (HC-10b: a line of `rce host run` or `serve`) and
  an HTTP request (HC-24) carry none, so a client's retry of one is a new
  input, and a program that must apply a retried request once keeps its
  own key in its state. A duplicate is acked, and the window that finds it
  holds no outcome for it, so the client that retried learns that its
  input is in D but not how its step ended; it reads that from D.
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
implementation is `spec/checker/src/raft.rs` (Raft log replication behind a
`Transport` trait, with a loopback switch and a TCP transport, and
`RaftStore`, the E.3 Store of a host on the leader).
`spec/checker/src/repl.rs` is a smaller model of the same rules on whole
logs.

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
  carries the term. The new admitter's adapters are fresh, so it appends
  `host.Restarted` after `host.Term` (HC-22), and a request emitted under
  an older term with no result in D gets one result, `failed` with
  `interrupted` (HC-3). The term in D
  is the fence token: an entry of D was appended by the holder of the
  most recent `host.Term` before it.
- **HC-18a Transport.** Replication follows Raft's AppendEntries and
  RequestVote: an append carries (prev_idx, prev_term) and a follower
  does not accept it unless its log has an entry at prev_idx with that term (log
  matching), deletes a conflicting suffix and appends what is missing.
  The current term and the vote are written to stable storage before a
  candidate asks for votes, before a vote is granted, and before an
  append is acknowledged. A vote goes only to a candidate whose log is at
  least as up-to-date by (last term, last index). The leader advances the
  commit index only over entries of its own term held by a quorum. A
  follower that needs entries the leader has compacted receives the
  leader's snapshot (an HS-5 snapshot of the state at that index) and keeps
  any suffix of its own log that matches it. The transport may lose,
  delay, duplicate or reorder messages. Timeouts are counted in logical
  ticks, so a loopback run is a function of its seed.
  Snapshots travel in chunks of bounded size, each carrying its byte
  offset; a follower accepts a chunk only at the offset it expects,
  acknowledges the next offset it needs, and installs the snapshot only
  when the final chunk arrives, so loss, duplication and reordering of
  chunks cannot install a partial state. Membership changes use joint
  consensus: the leader appends a configuration entry C_old,new; a node
  uses the latest configuration in its log whether or not it is committed;
  while C_old,new is in force, elections and commitment need a majority of
  C_old and a majority of C_new; once C_old,new commits the leader appends
  C_new, and a leader that is not in C_new steps down once C_new commits.
  Derivation: safety rests on any two quorums intersecting (Quorum.lean,
  `quorum_intersect`). Switching directly from C_old to C_new admits a
  majority of each that share no node, hence two leaders in one term;
  requiring both majorities during the transition keeps every pair of
  deciding quorums intersecting. Between hosts the transport runs over TCP
  inside mutual TLS with pinned ed25519 keys: a frame is accepted
  only if the peer id it claims matches the key that authenticated the
  connection, so a node without a pinned key cannot vote or append.
- **HC-20 Deposed admitter.** An admitter whose barrier is not granted because
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
assumes bounded clock skew)", which is why E.2b leaves leases out. Raft's
Figure 2 lists currentTerm, votedFor and log[] as "Persistent state on all
servers: (Updated on stable storage before responding to RPCs)", which is
the order HC-18a requires. On membership, Raft states that "Agreement (for
elections and entry commitment) requires separate majorities from both the
old and new configurations" and that "a server always uses the latest
configuration in its log, regardless of whether the entry is committed";
HC-18a follows both. Raft's rule that servers disregard RequestVote while
they believe a current leader exists, which limits disruption by removed
servers, is a liveness measure and is not implemented.

### E.2c Derivation from laws (informative)

E.2a to E.2b and HC-6, HC-12 are derived from the following laws rather
than from the code that existed when they were written. Each law is stated,
then the design it forces.

1. **Deterministic state machines agree when their logs agree.** Raft:
   "Each log contains the same commands in the same order, so each state
   machine processes the same sequence of commands. Since the state
   machines are deterministic, each computes the same state and the same
   sequence of outputs." So every nondeterministic choice that can affect
   a result must be an entry of D: the order of inputs (HC-15), the
   scheduler's choices and the environment seed of a `fn main` run (HC-6),
   and placement decisions (HC-12). Replay re-derives each one and stops
   where a derivation differs from D, so D is checked, not trusted.
2. **Safety must not depend on timing; liveness must.** Raft: "safety must
   not depend on timing ... However, availability ... must inevitably
   depend on timing." The law behind it is Fischer, Lynch and Paterson
   (JACM 1985), Theorem 1: "No consensus protocol is totally correct in
   spite of one fault", in a model where processes "do not have access to
   synchronized clocks, so algorithms based on time-outs, for example,
   cannot be used". Their conclusion asks for "more refined models of
   distributed computing that better reflect realistic assumptions about
   processor and communication timings". Raft's election timeout is such
   an assumption, and it is the only place E.2b uses one.
   So terms, votes and fences decide safety (HC-18 to HC-20), timeouts
   count logical ticks and decide only when an election starts (HC-18a),
   window boundaries are logged ticks (HC-15), and leader leases, which
   would make safety depend on bounded clock skew, are excluded.
3. **Exactly once is at-least-once delivery plus idempotent dedupe.**
   Raft: "clients ... assign unique serial numbers to every command. Then,
   the state machine tracks the latest serial number processed for each
   client". So sources retransmit until acked and the admitter dedupes by
   (source, seq), with the dedupe state folded from D so that it survives
   failover (HC-16, HC-17). No transport is trusted to deliver once.
4. **End-to-end argument.** A guarantee that only the endpoints can check
   belongs at the endpoints. Durability and exactly-once are checked where
   D is committed: the admitter acks only after a quorum commit (HC-16,
   HC-18), and the transport may lose, duplicate or reorder (HC-18a).
5. **Under a partition, consistency or availability (CAP); otherwise,
   latency or consistency (PACELC).** RCE chooses consistency: a minority
   side cannot commit, so it cannot admit (HC-20). Without a partition each
   window costs one quorum round; windows batch inputs to amortise it, the
   same trade Calvin makes with its epochs.
6. **Floating-point addition is not associative.** A parallel reduction
   reproduces the sequential bits only if it evaluates one fixed tree. So
   placement may change cost and never bits (HC-12, HC-13, T18), and a
   fabric enters only after a probe equal to the reference.

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
  {event, prev_hash, seq, state_hash} with the event in its positional
  form (6.1), which the host rebuilds from the stored event and the
  module; a `host.*` entry is hashed as stored. D is cut after the longest
  prefix that verifies, and the host logs `host.Truncated` with the number
  of entries dropped. A torn tail after a crash, a corrupted entry and a
  misdirected write are detected, never folded into state (T38). Replay
  of the kept prefix then re-derives each `state_hash` (HC-8); a mismatch
  is a host fault (HS-3).
- **HS-5 Snapshots are caches.** A snapshot is (seq, state_hash, bytes),
  the bytes being the positional form of the state (6.1). It is used only
  when the hash of its bytes equals the `state_hash` of D at seq and the
  bytes decode, guided by the declared field types in declaration order,
  to a state that re-encodes to the same bytes; otherwise it is
  discarded. Loading one changes no meaning.

### E.4 Inputs are events (HC-10, HC-10a, HC-10b)

**HC-10.** Every value that enters a run is an event or a logged answer
in D. This includes key, pointer, resize and focus input; `Tick` and
`host.WallTime`; every file, network, PTY, audio and GPU readback result;
foreign results and traps; `select nondet` choices; Delivered, TimedOut
and Unreachable; capability minting; lifecycle (suspend, resume, process
kill); random bytes; and the host entries of E.5. An input that is not
in D does not exist for the program.

- **HC-10a Bytes are delivered exactly.** An adapter that receives bytes
  from outside delivers exactly the bytes it received, in order, as a
  `List[i64]` whose elements are in 0..255 (the byte type of
  `core.bytes`), and D records them as a list of integers (6.1). No `text`
  holds every byte sequence, and a read may end inside a UTF-8 character,
  so a host does not decode received bytes unless the module takes them
  as text (E.4c HD-3): the program decodes them (`text.utf8`), and a body that
  is not UTF-8 is a value the program can answer rather than a trap. One delivery holds at most SIZE_LIMIT (N9)
  bytes, so the delivered value can always be built. The data of
  `NetRecvDone` (HC-23) and the body of `HttpRequest` (HC-24) follow this
  rule. Declared as `text`, either is delivered by E.4c HD-3: as text when it
  is UTF-8, never split inside a character, and kept in D as
  bytes when it is not.
- **HC-10b Typed inputs are inputs.** An input read from the input stream
  (a line on stdin of `rce host serve`, the events of `rce host run`) is
  admitted whatever event of the module it names, an event the host
  delivers itself included: `StepKept`, `Restarted`, `Synced`, `WallTime`,
  `HttpRequest`, `HttpMalformed`, or a result `KDone` of a struct K the
  module declares, that is, every event of the module that some host
  delivers, whatever adapters this run is granted, so the mark is a
  function of the module and D. Before such
  an input the admitter appends `host.Typed` {seq}, seq being the typed
  input's own seq, so D records which of these inputs were typed and which
  the host delivered. A typed input is an ordinary input: it answers no
  request (HC-3), a typed `StepKept` starts no descent (HC-21), and the
  kept outcome of its step is delivered as for any other input. Replay reads the
  mark: the input at seq is folded as typed, and a `host.Typed` that the
  input does not follow (the host stopped between the two appends) records
  an input that was never admitted, which is not completed on open. Replay
  treats as a host fault a `StepKept` or `Restarted` that is neither the
  delivery after `host.StepKept` or `host.Restarted`, typed, nor announced
  by `host.Input` (E.2a). An input given by its name and arguments
  (`Host::admit_event`) is typed in the same way; the inputs of a window
  carry `host.Input` instead. Reference: `Host::admit`, `host::How`.

### E.4a Kept steps and restarts (HC-21, HC-22)

A kept step and a restart of the host are facts of the run. A program
that is not told of them waits for an effect the kept step would have
requested, or for a socket that belonged to a process that is gone, so
both enter D and reach the program as inputs.

- **HC-21 A kept step is an input.** When the step for an input is kept (T8), the
  admitter appends the input with the state unchanged, then `host.StepKept`
  {cause, seq}, where seq is the kept entry's seq and cause is `fail`
  (a declared error left the step), `fuel`, `size` (SIZE_LIMIT), `value`
  (VALUE_LIMIT), `depth` (the evaluation depth of 6.7) or `trap` (any
  other trap of 6.3 or 6.7). When the module declares `event StepKept`,
  the admitter then delivers `StepKept` with the same payload, as the
  module declares it (E.4c, HD-4), as an ordinary input with the full fuel of
  HC-7a, and appends `host.Bind` for it at start. Delivery is bounded. A step descends from a
  delivered `StepKept` when it is that delivery, or when it is the result
  of a request made by a step that descends from it (so the results of the
  `StepKept` step's requests, the results of the requests those steps make,
  and so on); a typed `StepKept` (HC-10b) is an ordinary input and descends
  from nothing. A kept step that descends from a delivered
  `StepKept` is recorded in D as `host.StepKept` and is not delivered again
  as a `StepKept`. Each input whose step is kept therefore leads to at most one
  delivered `StepKept`, and a `StepKept` step whose effects in turn lead
  to kept steps ends after its results are recorded. Replay derives the same
  descent from D, matching each result to the first request queued with
  its kind and req. The cause names no declaration, so D stays name-free (T7);
  the kept step's text, which can name a declared error, is not in D, and
  replay re-derives it. Replay requires each kept entry to be followed
  by the matching `host.StepKept` and, when `StepKept` is declared, by the
  entries its delivery appends (the delivered `StepKept`, or `host.Kept`),
  and treats a D in which one is missing, extra or different as a host
  fault. A D that ends before them, because the host
  stopped in between, is completed on open, and by a new admitter before
  its `host.Term` (HC-19), before anything else is appended; no snapshot
  (HS-5) is taken while D owes them. `rce host replay` and the validation
  after every run report the number of kept entries, and replay prints
  the cause and text of each. Reference: `Host::kept`,
  `interp::Cause`.
- **HC-22 A restart is an input, and ids are never reused.** Adapter
  resources (sockets, terminals, media tracks) belong to one host process.
  A host that opens a D holding entries appends, after the entries that
  open it (HS-4, E.5), `host.Restarted` {below}, below being that entry's
  own seq. Every adapter that issues resource ids then issues them from
  below upward, so an id issued before the restart never names a
  resource of the new process; when the module declares `event
  Restarted` the admitter delivers it with the same payload, as the
  module declares it (E.4c, HD-4); and
  only then does every request outstanding in D get its one result,
  `failed` with `interrupted` (HC-3). The requests the `Restarted` step
  makes are performed, not interrupted. A host that takes the admitter
  role does the same after its `host.Term` (HC-19). Replay treats as a
  host fault a `host.Restarted` whose below is not its own seq, or one not
  followed by the entries the delivery of `Restarted` appends when it is
  declared. A host loop started on an existing D continues `Tick(n)`
  from the last n in D, so tick numbers keep rising across restarts.
  Reference: `Host::restarted`, `Adapter::restart`.

### E.4b The net adapter and the HTTP event host (HC-23 to HC-26)

- **HC-23 The net adapter.** `net HOST:PORT`, given once per address,
  grants TCP as effects. Authority is the granted addresses, compared by
  their exact text (HC-5); a request that names an address not granted, or
  an id the adapter did not issue, fails. Each request has one result
  (HC-3):

  | Request | Data of the result | Fails with |
  |---|---|---|
  | `NetConnect(req, addr)` | the id of the connection | `timed_out` after NET_CONNECT_MS, or the error |
  | `NetListen(req, addr)` | the id of the listener | the bind error |
  | `NetAccept(req, id)` | the id of a new connection | `none` when no connection is waiting |
  | `NetSend(req, id, data)` | the number of bytes handed to the connection | `closed`, `send buffer full`, `byte outside 0..255` |
  | `NetRecv(req, id, max)` | the bytes received so far, at most max, possibly none (HC-10a); taken as text, up to 3 more to complete a character (E.4c HD-3) | `eof` when the peer closed and nothing is left, `closed`, the read error |
  | `NetClose(req, id)` | 0; err lists the bytes of a character carried for a read that did not come (E.4c HD-3) | `no socket N` |

  `data` is text, sent as its UTF-8, or a `List[i64]` of 0..255. A send
  hands the bytes to the connection's own writer and returns at once, so
  no request waits for a peer except `NetConnect`, which waits at most
  NET_CONNECT_MS (5,000 ms), and `NetRecv` and `NetAccept` never wait.
  The writer fails the connection when its peer takes no byte for
  NET_WRITE_MS (30,000 ms) while bytes wait; later requests on it fail
  with `closed`, and a send that would leave more than NET_QUEUE_MAX
  (1,000,000) bytes waiting fails with `send buffer full`. `NetClose`
  closes once the bytes handed over are written. `max` defaults to 65,536
  and is capped at SIZE_LIMIT. These values decide only which result a
  request gets, and the result is in D. Ids are never reused within D
  (HC-22). A program that serves HTTP on this adapter frames bytes itself
  and polls with `Tick`, one connection at a time
  (`spec/checker/tests/host/notes_net.rce`); the HTTP event host below
  serves the same program without polling.
- **HC-24 HTTP requests are inputs.** `rce host serve http HOST:PORT
  FILE D` (one address, under `serve` only) binds the address before
  the OS target is entered (HC-5a) and appends `host.HttpListen` {addr and
  the limits of HC-25}, or {addr, error} when the bind failed; a failed
  bind is retried every second, and the bind that succeeds appends
  `host.HttpListen` with the limits. Connections are served concurrently,
  at most CONNS_MAX (256) at once. The host frames each request (HC-25)
  and admits it as one input `HttpRequest(id, method, target, version,
  names, values, body)`: the method, the target and the version (for
  example `HTTP/1.1`) as sent, the field names as written and their values
  without surrounding spaces and tabs, in order, and the body as the bytes
  left after its framing is removed (HC-10a); trailer fields are checked
  and not delivered. A request that breaks the
  syntax, a limit or a deadline after its first byte is one input
  `HttpMalformed(id, status, why)`, with the status of HC-25 and the
  reason. A connection that closes before a request is complete, or stays
  idle for IDLE_MS before the first byte of one, adds nothing to D. The
  id is the seq of the entry that delivers the input (its own entry, or,
  when `host.Kept` records it first under E.4c HD-2, the entry after that
  one), so `StepKept(seq)` (HC-21) names the request. The requests of one
  connection are admitted one at a time, each after the response to the
  previous one, so responses keep the order of the requests (RFC 9112
  9.3.2); `Expect: 100-continue` from HTTP/1.1 on, with a body to come,
  gets the interim response `HTTP/1.1 100 Continue` once the head is
  accepted (HC-26). At start the host appends `host.Bind` for `HttpRequest` and
  `HttpMalformed`, when declared. Replay reads requests from D and never
  parses (HC-8). These are the host's values: to a module declared with
  exactly them they are delivered unchanged, and to any other declaration
  as E.4c HD-2 says (the `body` of `HttpRespond` may be `text` or `List[i64]`):

  ```
  event HttpRequest(id: u64, method: text, target: text, version: text, names: List[text], values: List[text], body: List[i64])
  event HttpMalformed(id: u64, status: u64, why: text)
  event HttpRespondDone(req: u64, status: text, data: u64?, err: text)

  struct HttpRespond
      req: u64
      status: u64
      names: List[text]
      values: List[text]
      body: text
  ```

  The host loop reads one queue (stdin lines, HTTP inputs, and SIGTERM and
  SIGINT, which one thread waits for) and admits in rounds of up to 64
  inputs, with one barrier per round. Each time it performs queued
  requests, on open (HC-22) too, it performs at most DRAIN_MAX (64) of
  them; the rest are performed in the next round, which starts at once,
  so a stop and other inputs are read between drains even while results
  keep requesting effects. An effect waits for a barrier only
  when the entry that requested it is not yet durable (HS-2), and a
  barrier that would only confirm the previous confirmation is not taken,
  so a host with nothing to admit appends nothing. Lines are read as
  bytes. A stdin line that is not events of the module (it is not UTF-8,
  does not parse, names no event of the module, or one of its events
  cannot be built) is held whole: none of its events is appended, and it
  is delivered as `Input(data)`, data being the bytes of the line with its
  line end, built into the declared type like any delivery (E.4c): as
  `List[i64]`, or as `text`
  when they are UTF-8 (E.4c HD-3). With no `Input` declared, or one the line
  cannot be built into, `host.Kept` records the line in D, as text when
  it is UTF-8 and as its bytes otherwise; `rce host run` holds its lines
  the same way. Only a storage fault (HS-3) stops the loop. With `http` the end of stdin does not end the loop;
  SIGTERM or SIGINT stops accepting connections, waits at most STOP_MS
  (5,000 ms) for the open requests, takes the final barrier and validates
  the run (HC-8). PTY children start with no signal blocked.
- **HC-25 Framing, limits and deadlines.** The syntax is RFC 9112 with
  the strict choices of `net.http`: lines end in CRLF; the request line
  is a method of `tchar` characters (RFC 9110 5.6.2), a visible-ASCII
  target and `HTTP/` with a digit, a dot and a digit, after at most 16
  empty lines; field names are of `tchar` characters too,
  and a field line does not continue an earlier one (obsolete line
  folding); a request has at most one Host field, and exactly one from
  HTTP/1.1 on (RFC 9112 3.2); a request has Transfer-Encoding or
  Content-Length, not both; Transfer-Encoding ends in `chunked` and names
  it once; the values listed in Content-Length agree. A request that
  breaks one of these (a bare CR or LF, a folded line, both framings, and
  so on) is delivered and recorded as `HttpMalformed` with status 400.
  The host's parser and `net.http.parse_message(b, true, false, false)`
  agree on every
  vector of `stdlib/tests/net/http_checks.rce` and on 10,000 seeded
  mutations of them. Only the host gives 505 for a version other than
  1.x, 501 for a transfer coding before `chunked`, and applies these
  limits and deadlines:

  | Name | Value | Past it |
  |---|---|---|
  | LINE_MAX | 8,192 bytes of request line | 414 |
  | HEAD_MAX | 65,536 bytes of head | 431 |
  | FIELDS_MAX | 100 field lines | 431 |
  | BODY_MAX | 1,000,000 bytes of body (SIZE_LIMIT); a chunked body's encoding at most BODY_MAX + HEAD_MAX | 413 |
  | HEAD_MS | 10,000 ms from the first byte to the end of the head | 408 |
  | BODY_MS | 30,000 ms from the end of the head to the end of the body | 408 |
  | IDLE_MS | 5,000 ms between requests | the connection closes, nothing is appended |
  | RESPOND_MS | 60,000 ms for the program's response | the connection closes; a later `HttpRespond` fails with `no request N` |
  | WRITE_MS | 30,000 ms to write one response | the connection closes |

  Any other error of syntax is 400. The values are engine constants
  (HC-14) and `host.HttpListen` records them, so D says under which limits
  its requests were admitted. A deadline decides only whether a request
  is admitted and when a connection closes, never what an admitted request
  holds. A connection closes once its `HttpMalformed` is answered, and
  once the host has answered a request itself (E.4c HD-2).
- **HC-26 Responses are a function of D.** The program answers with the
  effect `HttpRespond(req, status, names, values, body)`. Its result
  `HttpRespondDone` is `delivered` with, as data, the length in bytes of
  the response the host made and handed to the request's connection,
  which writes it after the result is logged; a write that then fails
  (the client closed, or WRITE_MS passed) closes the connection and is
  not reported, so `delivered` means handed to the connection, not
  received by the client. The host checks
  a response before it writes anything: status 200 to 599, field names
  of `tchar` characters, values of printable ASCII (space to `~`) or tab, as
  many values as names, no Content-Length or Transfer-Encoding (the host writes the
  framing), Connection only as `close`, and no body for 204 and 304. A
  response that fails a check fails the effect with the reason and leaves
  the request open, so the program can answer it again; a req that is not
  open fails with `no request N`. The host writes `HTTP/1.1`, the status
  and its reason phrase (RFC 9110 15; empty for a status it does not
  list), the program's fields in order, `Content-Length` (not for 204 and
  304), `Connection: close` when the connection closes after this
  response and the program did not write it, an empty line and the body
  (none for HEAD). The connection closes after the response when the
  request asked for it (from HTTP/1.1 on, `Connection: close`; for
  HTTP/1.0, a Connection field without `keep-alive`, or none), when the
  program wrote `Connection: close`, after an `HttpMalformed`, and after
  a response the host makes itself (E.4c HD-2). The method, the version and
  the fields that decide this are in D, in the host's whole value of the
  request: its entry, or the `host.Kept` that records it first when the
  module takes less of it (E.4c). So every byte of a response, and whether
  the connection closes after it, is a function of D: replay re-derives
  every `HttpRespond` request (`host::replay_outputs`);
  `host::http_inputs` reads, for each HTTP input of D, from that whole
  value and the module, whether it was HEAD, whether the connection
  closes, and the status of a response the host makes itself, as the live
  host decided them (a typed input, HC-10b, is not an HTTP input); and
  `http::response_bytes` makes from these the bytes the client received. The interim `HTTP/1.1 100 Continue` of HC-24 is
  transport, not output of the program: it is written when the head is
  accepted, before the body arrives, for a request that may never enter D
  (its client may close, or its body time out), and it is the one thing
  the host writes that is not a function of D.

### E.4c Delivery by declaration (HD-1 to HD-4)

A host starts on any module the checker admits. It compares each event it
delivers, and each request struct of a kind it performs, with the
module's declarations, and delivers each value as the module declares
it, matching fields by name.
Determinism is kept by recording in D what the host delivered and what it
did not, not by restricting what a module may declare.

- **Delivery.** The host's value of a delivery is a set of named fields:
  {req, status, data, err} for a result (HC-3), the payload of
  `host.StepKept`, `host.Restarted`, `host.Synced` and `host.WallTime`
  (E.5), {n} for the frame tick `Tick` (HC-2; none when `Tick` declares no
  field), {data} for a line held as `Input` (HC-24), and the fields of
  HC-24 for an HTTP input. A declared field takes
  the host's field of the same name when its value converts exactly to the
  declared type: an integer within the range of the declared integer
  type, a number to `f64`, a boolean, text to `text`, a list element by
  element, a value to `T?` as `some`, a type alias as the type it names,
  and received bytes to `text` only by HD-3. An optional field takes
  `none` when the host has no value for it. The host's fields the event
  does not declare are not delivered. A delivery cannot be built when a
  declared field that is not optional has no value, or when a value has
  no exact conversion.
- **Recording.** Before a delivery that is not the host's whole value,
  and in place of one that cannot be built or whose event the module does
  not declare, the admitter appends `host.Kept` {event, why}: event is
  the host's whole value as {payload, type}, and why says what was not
  delivered and why. When the host entry just before the delivery holds
  the whole value (`host.StepKept`, `host.Restarted`, `host.Synced`,
  `host.WallTime`), `host.Kept` is appended only for a delivery that
  cannot be built, and an event the module does not declare is not
  delivered, as before. Replay plans each delivery again from the module
  and the value that `host.Kept` or the host entry holds, requires the
  same reason and the same delivery after it, and treats a D that differs
  as a host fault; a D that ends between `host.Kept` and its delivery is
  completed on open (HC-21). What the host delivers is therefore a function
  of D (T37). Reference: `Host::build`, `Host::plan`.
- **Notice.** At start the host prints one line for each declaration
  that is not exactly the host's value, naming it and saying what the host
  will deliver and record (`Host::delivery_notes`). The run continues.

- **HD-1 Results.** Each request struct of a kind the host performs is
  bound, whatever its result event declares. The result `KDone`, K being
  the kind, is delivered as the module declares `KDone`. A
  result whose data has no exact conversion to the declared data type is
  delivered as `failed`, with the reason in err (HC-3), and `host.Kept`
  keeps the data. A result whose req is not delivered as an integer (for
  example `req: u64?`) is recorded by `host.Kept` before it: replay
  matches a result with no such record to its request by that integer,
  and one with the record by the req the record holds, so each request
  has its one result however `KDone` declares req (HC-3, HC-21). When the
  module declares no `KDone`, the effect is still
  performed and its result is recorded in D as `host.Kept`, not
  delivered; the result answers its request, so a restart does not
  interrupt it (HC-22).
- **HD-2 HTTP.** `http` serves any module. `HttpRequest`,
  `HttpMalformed` and the result `HttpRespondDone` are matched by field
  name like any delivery, the body by HD-3, and the request `HttpRespond`
  is read by field name: names, values and body are empty when the struct
  does not declare them, and a status that is not an integer from 200 to 599 fails
  the effect (HC-26). A request that cannot be delivered as `HttpRequest`,
  because the module does not declare it or this request cannot be built
  for the declaration, is recorded in D as `host.Kept` and delivered as
  `HttpMalformed` {id, status, why} when that can be built, why naming the
  reason. Its status is 415 Unsupported Media Type when the body is not
  UTF-8 and the module takes it as `text` (RFC 9110 15.5.16 allows the
  format problem to be found by inspecting the content directly), and 501
  Not Implemented otherwise: RFC 9110 15.6.2 gives 501 when the server
  does not support the functionality the request needs, which is the case
  of a module that takes no requests or none of this form, and unlike 500
  it reports a lasting property of the server rather than an unexpected
  condition, and unlike 503 it is not temporary. The program can answer
  an input delivered with its id, the seq of the entry that delivers it,
  as an integer, when the module declares `struct HttpRespond` with `req`
  and `status` of integer types. Any other input (one not delivered, one
  delivered without its id, or any input of a module whose `HttpRespond`
  cannot name a request and a status) the host answers itself once the
  batch is durable (HS-2), with the host's status of it: 501 for a
  request delivered as `HttpRequest`, the status above for one that is
  not, and the status of HC-25 for a malformed request, whatever
  `HttpMalformed` declares; with no fields and no body, and it closes the
  connection after it. The start notice says when a module's
  declarations leave every request to the host. Who answers, and with
  what status, is read from the host's whole value of the input and the
  module (`host::http_inputs`), so every request is in D and every byte
  the host writes is a function of D (HC-26).
- **HD-3 Received bytes.** Received bytes (the data of `NetRecvDone`, the
  body of `HttpRequest`) are delivered exactly as `List[i64]` (HC-10a).
  Declared as `text` (through an alias or `?`), they are delivered as
  text when they are UTF-8. A stream read never splits a character: when
  the module takes the data of `NetRecvDone` as text, a read that would
  end inside a character reads up to 3 bytes past max to complete it, a
  character whose bytes have not arrived is carried to the next read on
  that connection, and what was carried is delivered as it is at the end
  of the stream, with the bytes read before an error, and before the
  failure of the connection; a `NetClose` of a connection that still
  carries bytes lists them in its err. Bytes that are not UTF-8 are
  delivered as a `failed` result with err naming the offset of the first
  byte that is not, or as `HttpMalformed` 415 (HD-2), and `host.Kept`
  records the bytes in D, so nothing received is lost.
- **HD-4 Host inputs.** `StepKept` (HC-21), `Restarted` (HC-22), `Synced`,
  `WallTime`, `Presented` (HC-29) and the frame tick `Tick` (HC-2) are
  matched by field name like any delivery. Declared as
  `StepKept(seq: u64)`, a kept step is delivered with its seq, and the
  `host.StepKept` before it holds the cause. Declared with a field the host
  has no value for, `StepKept` is never delivered, and each `host.StepKept`
  is followed by `host.Kept` with the reason. Delivery stays bounded
  (HC-21) whatever the declaration. `Tick` takes the count in its field
  `n`, or no count when it declares no field; a `Tick` no count can be
  built into is not delivered, which the host loop says once at start,
  and a count that no longer converts (`Tick(n: u8)` past 255) is recorded
  once as `host.Kept`, after which no tick is delivered, since no later
  count converts either.

`rce check` does not make these comparisons, because what a host
delivers depends on the adapters it is granted.

### E.4d The program that wrote D (HC-27)

- **HC-27 A log is bound to the program that wrote it.** Beside the log D
  the host keeps `D.program`, the module hash of the program (F.5.3: the
  SHA-256 of its canonical unparse, so layout and comments are not part of
  it). `rce host run` and `rce host serve` write it when it is absent,
  which binds a new log and adopts a log written before this rule. A
  program whose module hash differs from `D.program` requires change on
  that log: the host names both hashes and opens nothing, so D is unchanged.
  `rce host replay` checks `D.program` when it is present. A log replays
  to the state its program derived only with that program, so a changed
  program starts a new log; for a change that `rce change` classifies as
  an erratum (10.11), the operator may write the new hash to `D.program`,
  and that act is the record of the decision. The module hash covers the
  program's own text, not the library modules it uses, which the checker's
  build pins (F.5.3).

### E.4e What the host reports to its terminal (HC-28)

- **HC-28 The host reports its status, and the report is not an effect.**
  `rce host run` and `rce host serve` write their status to stderr as
  records of OSC 7501 (an operating system command sequence), the Program
  Status Protocol that libghostty-vt parses and Herdr (herdr.dev) shows:
  `ESC ] 7501 ; state=S:app=rce:title=T:msg=M ESC \`, where T is the
  module name and M a short line of text, each in standard base64 with
  padding. S is `idle` once the host has opened D and waits for input,
  `working` while it admits an input from outside and steps the program,
  `idle` again once the entries are appended to D, `done` when the host
  stops cleanly (at the end of the input, and for `rce host serve` also on
  SIGTERM, SIGINT or an emitted `Exit`, HC-24), and `error` when it stops
  on a failure or on a program that requires change. The `idle` message
  gives the number of entries of D and the first 12 hex digits of the
  state hash. A round of the host loop that holds only a frame tick (HD-4),
  the host's own clock, writes no record. A record is written only when S
  changes. The host never reports `blocked`, since it asks no person
  anything. Records are written only when stderr is a terminal;
  `RCE_STATUS=on` writes them whatever stderr is, and `RCE_STATUS=off`
  writes none. A report is not in D, no program reads it, and replay and
  the state hash do not depend on it, so D is the same byte for byte with
  reports and without them.

### E.4f Frames and their confirmations (HC-29)

- **HC-29 Frames are derived from D, and a confirmation is an input.** A
  *view tree* is a value of `ui.view.View` (`stdlib/lib/ui/view.rce`): a
  flat arena of nodes with tags, keys, attributes, texts and child lists,
  node 0 its root. A *frame* is a view tree that a step emits as
  `Frame(view)`, an output struct `Frame` the module declares with a field
  `view` of type `View`, as it emits `Display(data)`; the frame's seq is
  the seq of the entry whose step emitted it. Its `view_digest` and
  `frame_digest` are those `ui.view` defines: BLAKE3 of the deterministic
  CBOR (RFC 8949, 4.2.1) of the tree, and BLAKE3 of the frame record
  `["rce.ui.frame/1", seq, view_digest]`. An emitted `Frame` whose `view`
  is not a tree (each node reached from node 0 once, with parent links
  that agree) is not a frame: the host says so on stderr, and replay
  derives the same. A *renderer* is code outside the program that shows
  frames to a person.

  *What is in D.* The inputs, as for every program (HC-10), and nothing
  for a frame: a frame is an output of δ for its entry, so the frames of a
  session, their digests and their order are a function of D, and
  `rce host replay frames FILE D` re-derives every one with every adapter
  disconnected (HC-8). *What is not in D.* The frames, their digests, the
  diff operations between them and their `ui.rules` findings, all derived;
  and what a renderer draws, when it draws it, and what a platform
  accessibility interface reports, none of which a program reads.

  For `ui.rules`, the step word of a frame is its field `ev: text` when
  `Frame` declares one, else `start` for the first frame of D, `tick` when
  its entry is a `Tick`, and the event type of its entry otherwise; its
  time is its field `ms: i64` when declared, else 0 for the first frame and
  16 ms (the frame period of HC-14) for each `Tick` entry since the
  previous frame. Both are a function of D.

  Under `frames FILE`, `rce host run` and `rce host serve` append one line
  per frame to FILE, after the entries of its batch are appended (HC-1),
  so a renderer is never shown a frame that D does not hold. A line is
  canonical JSON with keys sorted: `ev`, `frame_digest`, `input` (the
  entry's event type), `ms`, `ops` (the operations `view.diff` gives from
  the previous frame's view, in the order it gives them; `null` for the
  first frame of D), `seq`, `view` (the seven lists of the view tree) and
  `view_digest`. `rce host replay frames FILE D` prints the same lines,
  byte for byte, each followed by its `ui.rules` findings, then the report
  of every frame as one trace (`rules.findings`, `rules.report`), so the
  rules are checked on exactly the frames a session showed.

  A renderer's confirmation that it presented a frame is an input. Under
  `presented FILE`, `rce host serve` reads lines `SEQ FRAME_DIGEST` that a
  renderer appends to FILE. For one that names a frame derived from D (that
  seq and that frame digest), the admitter appends `host.Presented`
  {digest, seq}, then delivers `Presented(seq, digest)` as the module
  declares it (E.4c, HD-4). A confirmation that names no such frame is
  refused: the host says so on stderr and appends nothing. Replay treats a
  `host.Presented` that names no frame derived from the entries before it
  as a host fault. So a program counts a frame as shown, not only emitted,
  by its step on `Presented`, and that count is a function of D.

  A host takes and uses no snapshot (HS-5) of a module that declares
  `Frame`: its frames derive from every entry of D (the first frame's
  word, the ticks since the previous frame, the frames a confirmation can
  name), so it is always folded from the start of D.

  *Decision on open question Q4 of `spec/proposals/human-interface.md`
  (2026-11-01).* A confirmation enters D as a host-named input, and does
  not stay outside D as HC-28's status does. That a person was shown a
  frame is a fact from outside the program on which a program may act (an
  alarm counted as annunciated only once shown, ISA-18.2), and HC-10 gives
  such a fact no other way in; a status report changes nothing that a
  program reads. The confirmation names which frame was presented, not how
  it looked: pixels and the timing of presentation stay outside D.
  Reference: `spec/checker/src/frame.rs`, `Host::presented`.

### E.4g Directories, entries, the zone and account names (HC-30 to HC-33)

A listing such as `ls -l` (`programs/systems/ls.rce`) reads a directory's
names, each entry's own record and link text, the owner and group names
and the local time of each modification time. Each is a request with one
result event (HC-3), so each is in D (HC-10) and replay re-derives the
listing from D alone (HC-8).

- **HC-30 Directory names are bytes.** `FileList(req, path)` under `fs
  DIR` answers the names of the directory, without `.` and `..`, each as
  its exact bytes (HC-10a), in byte order. A module that takes them as
  `List[List[i64]]` receives two names that differ only in bytes that are
  not UTF-8 as two names. One that takes them as `List[text]` receives
  them by E.4c HD-3: as text when every name is UTF-8, and otherwise the
  result is delivered as `failed`, naming the element and the offset, and
  `host.Kept` keeps the bytes in D. A request's `path` is text or bytes,
  so a name that is not UTF-8 can be named again. A `Display(data)` whose
  data is a `List[i64]` with every element in 0..255 is written to stdout
  as those bytes, so such a name is printed as it is. Reference:
  `adapters::FileAdapter`, `host::BYTES_LIST_FIELDS`.
- **HC-31 lstat and readlink do not follow the entry.** `FileLstat(req,
  path)` answers the lstat(2) record {mode, nlink, uid, gid, size, mtime,
  mtime_ns, blocks} of the entry the path names (`mtime` in seconds since
  the Unix epoch, `mtime_ns` the nanoseconds past them, `blocks` in
  512-byte units), and `FileReadlink(req, path)` the text of a symbolic
  link as its exact bytes, unchanged. Neither follows a final symbolic
  link. The root check of HC-5 applies to the path that is read, not to
  where a link points: the directory that holds the entry, resolved with
  every link in it followed, must be under DIR, and a path whose last part
  is `.` or `..` names a directory itself, which must be under DIR. A link
  whose target leaves DIR is therefore an entry of DIR: its record and its
  text are delivered, and reading the text does not follow it. `FileRead`,
  `FileWrite`, `FileAppend` and `FileList` follow links and refuse a path
  that resolves outside DIR, as before. Reference: `FileAdapter::resolve_entry`.
- **HC-32 The clock and the zone's offset are answers.** `WallClock(req)`
  answers the wall clock in nanoseconds since the Unix epoch. `LocalOffset(req,
  at)` answers the offset east of UTC, in seconds, of the host's time zone
  at `at` seconds since the epoch, by the zone's rules (localtime(3): the
  zone of `TZ`, else `/etc/localtime`), which the host reads once at
  start, before it enters the OS target of HC-5. The offset is a request
  and not a field of `host.WallTime`, because a listing needs the offset
  at each modification time, and a zone's offset changes with summer
  time. Neither request needs a grant: like random bytes (HC-10), each
  answer is an input and gives no authority over any resource. When the
  module binds `LocalOffset`, the admitter appends `host.Zone` {file,
  sha256, tz} at start: the value of `TZ` (null when it is not set), the
  zone file the C library reads for it (empty for a POSIX rule such as
  `EST5EDT,M3.2.0,M11.1.0`, which names no file) and that file's SHA-256,
  so D says which rules gave its offsets. Replay reads every answer from
  D, and no clock or zone file. Reference: `adapters::WallClockAdapter`,
  `adapters::LocalOffsetAdapter`, `Host::zone`.
- **HC-33 Account names need the `names` grant.** `UserName(req, id)` and
  `GroupName(req, id)` answer the name of a user or group id as its exact
  bytes: the first field of the first line of `/etc/passwd` or
  `/etc/group` (the `files` source of the Name Service Switch) whose third
  field is the id, or `failed` when no line has it. They are performed
  only under `names`, a mode word of `rce host run` and `rce host serve`
  (PC-3; it takes no value, since the two files are fixed), because the
  databases are files outside every granted root and name the people of
  the machine, and HC-5 gives the host no ambient authority. Without
  `names` no adapter answers them, and the OS target of HC-5a does not
  allow reading the two files. The adapter reads the files and does not
  call getpwuid(3), whose other sources (NIS, LDAP, systemd) may answer
  from the network or from another process, which no file grant bounds.
  Reference: `adapters::NameAdapter`.

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
| `host.Placed` | class, fabric id, op id, work | when placement is decided, before the operation runs (HC-12) |
| `host.WallTime` | nanoseconds since the Unix epoch | when an adapter supplies wall time |
| `host.Energy` | source, basis, joules | when a counter is read (D.11) |
| `host.Bind` | host result kind, program event name | at start (E.8) |
| `host.Window` | count, id, term, upto (tick) | when a window closes (HC-17) |
| `host.Input` | late (when true), seq, source, window | before each input of a window (HC-17) |
| `host.Term` | leader, term | when a host takes the admitter role (HC-19) |
| `host.MainStart` | links, max_ticks, seed | when a `fn main` run starts (HC-6) |
| `host.MainInput` | a, b, partition, tick; or event, tick | a link change or a program event for the live task, after its last logged step (HC-6) |
| `host.Step` | cause, i, node, tick, what | one per scheduler step of the run (HC-6) |
| `host.MainEnd` | choices, outcome, ticks | when the run ends (HC-6) |
| `host.StepKept` | cause, seq | after a kept entry (HC-21) |
| `host.Restarted` | below (its own seq) | when a host opens a D that holds entries, or takes the admitter role (HC-22) |
| `host.Typed` | seq (of the typed input that follows) | before a typed input that names an event the host delivers (HC-10b) |
| `host.Kept` | event (the host's whole value, as payload and type), why | before a delivery that is not the whole value, and in place of one that cannot be built or whose event is not declared (E.4c) |
| `host.HttpListen` | addr and the limits of HC-25, or addr and error | at start under `http`, and when a retried bind succeeds (HC-24) |
| `host.Presented` | digest (the frame digest), seq | when a renderer confirms that it presented the frame of seq (HC-29) |
| `host.Zone` | file, sha256, tz | at start, when the module binds `LocalOffset` (HC-32) |

A program receives the content of a host entry only through a binding
(E.8): the admitter then appends, after the host entry, an ordinary event
the program declares, with the same payload, delivered as the program
declares it (E.4c). δ itself never reads a host entry.

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
  replay (T37): the host logs each decision as `host.Placed` and replay
  re-derives it and stops where the derivation differs from D. A fabric is a candidate for
  a class only after a `host.Exact` probe equal to the CPU reference
  (HC-13). Reference executor (`Host::place_sum`, `place_lanes`,
  `place_par`): the classes are `sum.f64`, `lanes.f64.<op>` for the
  lane-wise binary64 operations add, sub, mul, div, neg and sqrt, and
  `par.<class>` for a `for … par` body over `0..n` combined by a given
  operation. On `cpu` and `cpu.threads` lanes are split into contiguous
  chunks across threads, which cannot change a lane's result because
  lanes do not read each other, and every combination, sums included, is
  evaluated as the fixed tree of C.8 (`host::tree_split`: the left
  subtree takes the largest power of two below the length), whose shape
  is a function of the length alone, so the bits are the same for any thread count and any
  combine, associative or not. `Host::probe_fabric` runs the probe
  vector (signed zeros, subnormal results, infinities, NaN, and sums of
  every length up to 64 with cancellation) on the fabric and on the CPU
  reference and logs `host.Exact` with both digests. A placement onto a
  fabric without an executor is not performed and is not logged; replay
  re-derives placements from D and needs no executor. Derivation: law 6
  of E.2c fixes the tree; law 1 makes the choice of fabric an entry of D;
  HC-13 makes a fabric trusted only by evidence in D.
  Platform fabrics (informative interface). A GPU through wgpu, or an
  NPU, enters by implementing `FabricExec` {id, lanes(op, a, b), sum(xs)}
  and `Host::register_fabric`. The contract: each lane is the correctly
  rounded binary64 result of the named operation, with no contraction
  into fused multiply-add and no flushing of subnormals, and `sum`
  evaluates exactly the fixed tree of C.8. A device without binary64, or one
  that cannot meet the contract, fails its probe and is never placed. A
  run-time error from a placed fabric falls back to the CPU reference,
  which has the same bits. `for … par` bodies are host closures and run
  only on `cpu` or `cpu.threads`; placing them on a device needs the body
  compiled to a kernel, which belongs to the platform worker together with
  the wgpu implementation of this interface. The tick scheduler's
  `for … par` (`conc.rs`) and the four runtimes combine with the same
  `host::tree_split`, so a reduction gives the same bits whether the
  scheduler or the placement executor runs it (`host_place.rs`
  `forpar_reference_and_placement_share_the_tree`).
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
  does not apply at admission: the state is kept (5.1).
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
- **Queries terminate.** A query is an RCE function, so T2 and W-REC-1 apply: a query that the checker cannot show terminates requires change; it is not run with a timeout.

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

- **T40 Canonical windows.** The entries of a window are a function of
  its inputs: any permutation of a window's inputs gives the same entries,
  so D is independent of arrival order within every window of a sequence
  of windows, and sorting a window loses and adds no input. Proved on a
  model where an input has an injective key (the HC-15 key with the event
  bytes) and dedupe keeps one copy of each key.
- **T41 Prefix across terms.** A replica that promised a higher term
  leaves its log unchanged on an append from a lower one, and every
  committed prefix is a prefix of D after any number of term changes. This
  is T38 and T39 under leader change. Quorum intersection, Election Safety
  (one leader per term, from the durable single vote) and Leader
  Completeness are proved for every reachable state of a whole-log
  protocol model whose steps are elections under the vote rule, leader
  appends, fenced writes and commits, from the initial state and with no
  further hypotheses. The proof is strong induction on terms: a leader of
  a term above t shares a voter with the commit quorum, the vote rule makes
  its starting log at least as up to date as that voter's, so it either
  holds the committed prefix directly or was started from the log of an
  earlier term above t, which holds it by induction. Election Safety is
  derived in the same model, from the durable vote and quorum
  intersection, rather than assumed. Verdi (Woos et al., CPP 2016)
  reports a Coq proof of state machine safety for Raft in "530 lines of
  code and 50,000 lines of proof"; RCE's proof is smaller and covers the
  whole-log model only.

Lean: `RCE/Host.lean` (T37 to T39), `RCE/Quorum.lean` (T41: `quorum_intersect`, `election_safety`,
`elect_fresh`, `commit_seed`, `leader_completeness`, `committed_in_later_replicas`, proved for every reachable state of the protocol model with no further hypotheses; only Lean's standard axioms), `RCE/Admit.lean` (T40, T41:
`window_perm`, `log_permwise`, `window_complete`,
`fenced_append_unchanged`, `committed_prefix_preserved`). The link from
Lean to the host: `RCE/AdmitVectors.lean` extracts 40 vectors from
`window` (`lake env lean --run`), and `tests/lean_admit.rs` requires the
checked-in vectors to equal that output and `host.rs` to produce the same
window order for each vector's arrival order. Conformance: `spec/checker/tests/host_contract.rs`
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
| HC-6, HC-7 | tick scheduler logged in D, numerics | `tests/host_sched.rs` (`main_run_is_logged_and_replays_exactly`, `main_steps_interleave_with_inputs_and_state_is_the_fold_of_d`, `main_input_after_the_last_step_only`), existing T1, T9 and 6.3 tests |
| HC-7a | fuel per input | `tests/fuel_per_input.rs` |
| HC-8, HC-10, T37 | replay with adapters disconnected | `foreign_and_host_entries_replay` |
| HC-9 | priority admission | `admitter_order_is_d_order`, `simultaneous_order_does_not_depend_on_caller_order` |
| HC-15 to HC-17 | canonical windows, exactly once | `tests/host_admit.rs`: `two_hosts_different_arrival_orders_give_byte_identical_d`, `window_order_is_priority_then_source_then_seq`, `gap_is_held_and_admitted_once_filled`, `no_ack_without_a_durable_append`, `dedupe_state_survives_replay` |
| HC-18 to HC-20 | replicated D, term fence, failover | `admitter_failover_keeps_one_d_and_exactly_once`; `tests/host_raft.rs`: partitions, reordering and loss, crash and restart, durable vote, log matching, chunked snapshot install under reordering, joint consensus membership changes, TCP, mutual TLS with an unpinned intruder not accepted, `host_fails_over_through_raft` |
| HS-1 to HS-5, T38, T39 | durable append, fatal barrier, verify on open | `verify_on_open_cuts_bad_suffix`, `fatal_barrier_stops_admission` |
| HC-11 to HC-14 | fabrics, placement, exactness, no flags | `placement_is_a_function_of_host_entries`, `placement_executor_follows_d_and_replays`, `fixed_tree_bits_do_not_depend_on_the_fabric`; `tests/host_place.rs`: `lanes_and_trees_have_the_same_bits_for_every_thread_count`, `lane_and_par_placement_follow_d_and_replay`, `platform_fabric_is_used_only_after_an_exact_probe` |
| HS-6 to HS-9 | RCEdb in RCE | `rcedb_example_runs` |
| HC-5a | the printed target is the installed target | `tests/host_sandbox.rs`: `profile_restricts_network_to_granted_ports`, `description_is_generated_from_rules`, `linux_description_says_by_port_and_udp`, `http_only_has_no_new_ip_sockets`, `linux_reads_are_restricted_to_the_grants`, and on macOS only `seatbelt_blocks_an_ungranted_port`, `serve_under_seatbelt_answers_a_request` |
| HC-10a, HD-3 | bytes delivered exactly, or as text when UTF-8 | `tests/host_net.rs`: `recv_delivers_the_bytes_received`, `recv_into_a_text_field_is_text_when_utf8`, `recv_as_text_completes_a_character_past_max`, `recv_is_capped_at_the_size_limit`; `tests/host_http.rs`: `invalid_body_bytes_are_exact_and_get_400`, `http_body_is_delivered_as_text` |
| HC-10b | typed inputs are inputs | `tests/host_kept.rs`: `typed_step_kept_is_admitted_and_marked`, `replay_detects_a_forged_host_step_kept`; `tests/host_restart.rs`: `typed_restarted_and_typed_result_are_inputs`; `tests/host_http.rs`: `typed_http_request_is_an_input`; `tests/host_effects.rs`: `a_typed_result_is_marked_without_its_adapter` |
| HC-21, HD-4 | kept steps are inputs, matched by field name | `tests/host_kept.rs`: `kept_step_is_followed_by_host_step_kept_and_the_bound_event`, `each_cause_is_classified`, `a_kept_step_kept_input_is_not_redelivered`, `a_kept_step_whose_effect_is_kept_is_not_redelivered`, `step_kept_handler_has_full_fuel`, `replay_detects_a_log_without_host_step_kept`, `replay_detects_a_forged_host_step_kept`, `open_completes_a_trailing_kept_step`, `a_step_kept_of_another_shape_is_matched_by_field_name`; `tests/nameless.rs`: `kept_entries_are_name_free` |
| HC-22, HD-4 | restarts are inputs, ids never reused, matched by field name | `tests/host_restart.rs`: `restart_appends_restarted_then_interrupts`, `ids_are_not_reused_after_restart`, `tick_numbers_continue`, `take_role_appends_restarted`, `serve_on_an_existing_log_listens_again`, `a_restarted_of_another_shape_is_matched_by_field_name`, `a_tick_is_matched_by_field_name` |
| HC-23, HD-1 | the net adapter, results matched by field name | `tests/host_net.rs`: `tcp_loopback_random_and_blobs_replay`, `send_takes_text_or_bytes`; `tests/host_effects.rs`: `result_event_of_another_shape_is_matched_by_field_name`, `bound_effect_without_its_result_event_is_performed_and_recorded`, `a_result_whose_req_is_not_an_integer_answers_its_request_once` |
| HC-24 to HC-26, HD-2 | the HTTP event host | `tests/host_http.rs`: `concurrent_clients_are_served_while_one_stalls`, `idle_server_appends_nothing`, `keep_alive_and_pipelined_requests_answer_in_order`, `head_has_no_body`, `oversized_body_is_413_and_closes`, `kept_request_gets_the_programs_status`, `invalid_response_fails_and_the_id_stays_open`, `responses_are_a_function_of_d`, `http_1_0_and_keep_alive_are_framed_from_d`, `expect_continue_is_transport`, `parallel_load_replays_exactly`, `sigterm_drains_and_validates`, `restart_binds_again`, `sequential_requests_are_not_paced_by_frames`, `http_declarations_are_noted_at_start`, `http_body_is_delivered_as_text`, `http_without_request_event_is_answered_501`, `http_without_respond_is_answered_by_the_host`, `http_framing_and_host_answers_are_read_from_the_hosts_value`; `tests/host_kept.rs`: `serve_stops_while_effects_keep_coming`, `a_line_that_is_not_an_event_is_held`, `a_line_is_held_whole_and_as_bytes`; `tests/http_parse.rs`: `vectors_agree_with_net_http`, `mutations_agree_with_net_http`, `incremental_feed_equals_whole_buffer`, `limits_give_their_statuses` |
| HC-27 | a log is bound to the program that wrote it | `tests/host_program_bind.rs`: `a_log_runs_and_replays_only_with_the_program_that_wrote_it` |
| HC-28 | the host reports its status; the report is not an effect | `tests/host_status.rs`: `host_run_prints_status_records_in_order_on_stderr`, `host_serve_reports_idle_working_and_done`, `no_escape_byte_without_a_terminal`, `d_is_byte_identical_with_and_without_reports` |
| HC-29 | frames are derived from D; a renderer's confirmation is an input | `tests/host_frames.rs`: `frames_replay_byte_identically`, `ui_rules_report_is_the_same_on_live_and_replayed_frames`, `step_words_and_times_are_derived_from_d`, `presented_confirmation_round_trips`, `serve_reads_a_renderers_confirmations`, `ui_web_fixture_is_current`; the unit tests of `src/frame.rs` check BLAKE3 on its official vectors and both digests against `ui.view` |
| HC-30, HC-10a | directory names as exact bytes, written exactly by Display | `tests/host_fs.rs`: `directory_names_arrive_as_their_exact_bytes`, `ls_lists_through_the_host_as_coreutils_does` |
| HC-31 | lstat and readlink without following, checked on the directory that holds the entry | `tests/host_fs.rs`: `lstat_and_readlink_do_not_follow_and_replay`; `tests/host_adapters.rs`: `file_effects_stay_under_the_granted_root` |
| HC-32 | the clock and the zone's offset as logged answers | `tests/host_fs.rs`: `clock_and_offset_are_logged_answers`, `offset_follows_the_zone_at_each_instant` |
| HC-33 | account names under the `names` grant | `tests/host_fs.rs`: `account_names_come_from_the_databases_and_replay`, `names_grant_reads_the_databases_and_the_sandbox_says_so` |
| HC-8, HC-16, HC-21, HC-22 under faults | dropped, duplicated and reordered inputs, lost answers, kill -9, torn and partly written tails and power cuts, wall clock steps | `tests/host_faults.rs`: `host_contract_holds_under_seeded_faults`, `a_fault_seed_reruns_to_the_same_d`, `no_faults_means_no_mishandling` |

### E.11 Proposed rules (not normative)

The rules in this section are **proposed**. They come from a survey of other languages (LESSONS.md in the language-structure study) and are recorded here so that they can be reviewed against this appendix. None of them is part of the host contract yet: a conformant host need not meet them, and no test enforces them. Each one is adopted, changed or rejected by a later draft. Where a proposal restates a rule that already holds (for example 6.3 numerics or T1 determinism), it says so.

**Numerics.**

- **N-1 (adopted as 6.3 N3, F1 to F9).** Float operations are IEEE 754 binary64 or binary32, round to nearest even. No fused multiply-add unless the program writes one. No NaN reaches a value (F4, F5). Transcendental functions are correctly rounded RCE code in the `math` module or absent (F9).
- **N-8 (adopted as 6.3 N3 F2).** A float expression has exactly one evaluation: one rounding per operator, in source order. Contraction, reassociation and algebraic rewrites are not valid compilations, and a float rewrite is never admitted by the rewrite prover.
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
- **TU-2 (proposed).** The shape is part of the type and is checked statically. An operation whose operand shapes do not agree (for example a matrix product of `[m, k]` and `[j, n]` with k not equal to j) is a checker error, never a run-time trap. Dimensions are literals or names bound in the signature; a shape that cannot be determined statically requires change. Relation: a new W-rule in 9 when adopted.
- **TU-3 (proposed).** A record whose fields all have one element type and one shape may be read as a tensor over a named axis (`{x: f64, y: f64}` as `Tensor[f64, [axis: {x, y}]]`). Field names index that axis. A record with mixed field types stays a record.
- **TU-4 (proposed).** Every reduction over an axis follows N-9: a left fold over the axis in index order unless the program names a different fixed tree. The shape and the named tree, never the thread count or fabric, determine the bits.
- **TU-5 (proposed).** Text, maps, graphs and sum types remain separate canonical values. They are ragged and are never padded or densified into tensors; a tensor element is never one of them.
- **TU-6 (proposed).** Placement (HC-12) keys the fabric cost table by (pure operation, element type, shape). Choosing a fabric changes cost only; HC-7 and N-9 fix the result bits.

Convergence sources. Each source was read as noted; claims rest only on the passages cited.
- BQN specification, types (read in full): "An array is a rectangular collection of data. It is defined by a shape, which is a list of non-negative integer lengths, and a ravel" whose length is the product of the shape; non-arrays are atoms. This is the shape-plus-ravel model of TU-1. APL, BQN's ancestor, was not read for this proposal.
- Paszke et al., "Getting to the Point: Index Sets and Parallelism-Preserving Autodiff for Pointful Array Programming" (Dex), arXiv:2104.05372 (full text scanned, hits reviewed): arrays have types `n=>a` indexed by typed index sets (`IndexSet n`), so the index domain of an array is part of its type. This is the basis of TU-2 and of TU-3's named axes.
- Henriksen et al., "Futhark: Purely Functional GPU-Programming with Nested Parallelism and In-Place Array Updates", PLDI 2017 (full text scanned, hits reviewed): "All arrays must be regular, meaning that, all rows of an array must have the same shape"; where shape cannot be verified statically, Futhark inserts dynamic checks. TU-2 takes the regularity rule and requires change instead of checking dynamically; TU-5 keeps irregular data out of tensors.
- Julia manual, "Multi-dimensional Arrays" (read in full): "The base array type in Julia is the abstract type AbstractArray{T,N}. It is parameterized by the number of dimensions N and the element type T." Julia puts element type and rank in the type but checks sizes at run time; TU-2 goes further by putting the shape in the type.

**RCEdb graph index (E.8.1).** These refine HS-6 to HS-9. Convergence sources are listed after the rules; they are observations of convergent design, not proof that the rules are right.

- **G-1 (proposed).** The primary index of RCEdb is a typed property graph projected from the log: nodes and edges have declared RCE types (a label is a type, a property is a field), and the projection is ordinary RCE code, a fold of δ over a log prefix (HS-7).
- **G-2 (proposed).** Every node and edge carries the id of the event that created it, and the id of the event that retracted it if any. Provenance is a field of the value, so it is checked, hashed and replayed like any other field.
- **G-3 (proposed).** A query is a graph pattern written as a typed RCE value (node patterns, edge patterns, quantified paths, predicates as RCE functions), never as a string. Ill-typed patterns require change (the checker reports them); there is no query text to inject into (5.16). The pattern vocabulary is the read-only GQL fragment that MGQL mechanizes (node and edge atoms in five directions, label expressions with `&`, `|`, `!` and `%`, property maps, quantifiers `*`, `+`, `?`, `{i}`, `{i,j}`, pattern lists joined on shared variables), so that each RCE pattern has a stated GQL reading. A variable declared inside a quantifier has a list type outside it, and a variable under `?` is nullable.
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

<!-- body-sha256: a1cb0589278e11eed6df98021a03b6e4ba06f4823c93d4adea85c19b682df141 -->
