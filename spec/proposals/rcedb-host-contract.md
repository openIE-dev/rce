# Proposal: the RCEdb host contract stated against the database record

Status: proposal. Owner: David Charlot. Derived from E.3 (durable append),
E.8 (RCEdb), HS-1 to HS-9, the partial-order log proposal
(`partial-order-log.md`, T62) and the survey *RCEdb among database
systems and DeltaDB* of 23 October 2026
(`openie-research-experiments/rce/surveys/2026-10-23/`), which read the
published record of relational, log-centric, immutable, deterministic,
embedded and local-first systems and of Delta's DeltaDB.

In plain words:

1. The log-as-database design is not new; a decade of shared-log research
   and several production systems validate it. What RCEdb adds is a fold
   that is user-visible pure code, a digest at every sequence number, and
   a theorem about δ.
2. Every surveyed system that kept a log as its record ended up with the
   same three appendages that E.8 still leaves implicit or optional:
   checkpoints with a verifiable root, a split between a payload and its
   commitment so that a payload can be erased, and a written statement of
   what "durable" means per operating system and per replica count.
3. The partial-order proposal's safety rests on two conditions the field
   has already named, not on the hash DAG itself: independent events must
   commute (T62's obligation), and a read over replicated state must be
   monotone or wait for a cut.
4. Two claims are available to RCEdb that no surveyed system makes, and
   the evidence supports stating them: divergence between replicas is
   attributable, and an offline client needs no second implementation of
   its mutators.

The proposal states each appendage as a rule in the form of E.3 and E.8,
names what it changes in the normative text, and lists what stays open.
Nothing here changes δ or any valid program's meaning: under C.15 every
item is outside programs (the log and the host) or an extension that
admits only what was refused before.

## 1. Entries: a commitment to the payload, and a versioned header

- **HS-10 Payload commitment.** An entry of D hashes the event's
  positional form (6.1, T7) as it does today, and may instead hash a
  *commitment* to it: `{"payload": {"$commit": h}, "type": k}` where h is
  the SHA-256 of the canonical JSON of the positional payload. The payload
  itself is then held in a payload store keyed by h. A host that writes
  commitments states so in `host.StorageCaps`. The fold of an entry whose
  payload is absent meets the typed value `evicted` (section 3), so the
  chain and every digest are unchanged by eviction.
- **HS-11 Header.** The first line of a stored D is a header entry
  `host.Header` carrying the hash algorithm (`sha256`), the entry format
  (`rce-d/1`), and the program's shape hash (6.6). A header is itself an
  entry of the chain. A reader that does not know the algorithm or format
  refuses the log; it never guesses.

Why: XTDB's split of a document store from its transaction log is the
surveyed design that made erasure possible without rewriting history, and
every tamper-evident system in the record (Trillian, the Go checksum
database, certificate transparency) versions its hash algorithm in the
header rather than in the deployment. The commitment is the positional
payload's hash, so T7 holds for it: a renamed program commits to the same
bytes.

## 2. Cuts and checkpoints

- **HS-12 Checkpoints.** A `host.Checkpoint` entry carries
  {seq, state_hash, root, prev_checkpoint} where `root` is the root of a
  Merkle tree over the entry hashes of D up to seq and `prev_checkpoint`
  is the seq of the previous checkpoint (or none). A checkpoint is an
  entry: it is hashed into the chain and replayed like any other.
- **HS-9, restated.** The state as of seq k is the fold of D from the
  latest verified checkpoint at or below k. Without a checkpoint it is the
  fold from s₀, as today; the restatement changes no state, only the cost.
- **HS-4, refined.** Verify on open checks the chain of the suffix after
  the last checkpoint whose `root` recomputes, and the inclusion of that
  checkpoint in the tree below it. The whole-log check of today remains
  available (`rce host verify --full`) and is the reference; the suffix
  check must give the same answer whenever both run.
- **Cost labels.** HS-7 and HS-9 are O(|D|) without checkpoints and
  O(|suffix|) with them. The survey's estimate for a whole-log verify at
  the read rates of commodity storage is eleven to twelve minutes per
  terabyte; with checkpoints it is the suffix.

Why: every log-structured system in the record (Kafka's compaction, the
LSM family, Corfu's checkpoints, TigerBeetle's superblock) adds a
checkpoint whose root a reader can verify; without one, time travel is a
full fold and open is a full read. A Merkle tree over D also gives
O(log N) inclusion and consistency proofs, which is how a client can
check that two log heads are prefixes of one history (section 6).

## 3. Erasure

- **HS-13 Eviction is a tombstone.** A host may replace a committed
  payload (HS-10) by a tombstone. The fold then meets the value
  `evicted` of the event's declared type: a value of the union type
  `τ ! Evicted` for a payload field of type τ, which δ handles as any
  error union (6.2). The digest at every seq is unchanged, since the
  entry hashed the commitment. Rewriting, reordering or removing an entry
  is refused (HS-4 detects it).
- **Bounded history.** A program may declare on a state field
  `#history(n)` to say that an index over it keeps n versions; the
  declaration bounds index growth and nothing else. Its syntax is
  proposed in Appendix F and is not yet admitted.

Why: Datomic's `noHistory`, XTDB's evict and the data-protection record
(erasure requests against append-only stores) all converge on the same
answer: erase the payload, keep the commitment, type the hole.

## 4. Schema evolution

- **HS-14 Program version in the entry.** Every entry carries the shape
  hash (6.6) of the program that admitted it, by reference to the header
  (HS-11) and, when the program changes, by a `host.Program` entry naming
  the new shape hash. A change to δ is therefore a digest mismatch on
  replay under the old program, never a silently different state.
- **Upcasters.** An event declared in an older program version is folded
  by a pure function from the old payload type to the new one, written in
  RCE and declared next to the event (`upcast from <shape hash>`). An
  upcaster is applied at fold time and is itself part of the program, so
  it is hashed and checked. Syntax is proposed in Appendix F.

Why: event-sourced systems in the record (Axon, EventStoreDB, the
"upcasting" literature) keep events immutable and translate at read time;
schema registries (Kafka's) version the schema with the record. RCEdb has
the stronger tool, a hash of the program, and should use it.

## 5. Durability, stated

- **HS-1, refined.** "Durable" is the barrier of E.3 per operating
  system: `fdatasync` or `fsync` on Linux, `F_FULLFSYNC` on Apple
  platforms, `FlushFileBuffers` on Windows; on a replicated D (E.2b) it is
  the quorum of per-replica barriers. The row name is in
  `host.StorageCaps` (today) and the class in a new field
  `durability: per_os | replicated | power_tested`, where `power_tested`
  means the host ran the power-fault harness of section 8 on this
  storage.
- **HS-15 The Sync event.** `host.Synced` carries the acknowledged seq
  and the entry hash at that seq; a failed barrier is `host.StorageFault`
  (output only, as today), after which the admitter exits and a new
  admitter recovers from on-disk D only, truncating a torn tail (HS-4).
  The batching window and its latency floor are stated in
  `host.StorageCaps`. An output of λ that depends on an entry may be gated
  on that entry's durable mark; a program says so with the bound `Synced`
  event, as example 55 does.
- **Bookmarks and retries.** A client carries the admitted seq it last
  saw as a bookmark; a retry of an input is bounded and ends in an
  explicit failure event, never in a silent drop or a duplicate.
- **The trusted base, written.** The host's conformance report names its
  trusted computing base: the Lean kernel, SHA-256, the extraction or
  compiler, the operating system's sync semantics, the storage firmware
  and the replica count; and it carries `verified_host=false` until a
  verified host exists.

Why: the relational record's durability defaults (PostgreSQL's
`synchronous_commit`, MySQL's `innodb_flush_log_at_trx_commit`, SQLite's
`synchronous`) and the fsync failure studies (the 2018 "fsyncgate"
record, the 2020 study of fsync error handling) show that "durable" has
meant different things in different systems and that the difference was
only ever discovered by data loss. Stating the class with the figure is
the cheapest fix the record offers.

## 6. Queries and indexes

- **HS-8, refined.** Canonical order is total over the whole row,
  including ties, nulls and floats (IEEE 754 totalOrder as in 6.1, so
  there is no NaN and `-0.0` sorts before `0.0`). Aggregates fold in
  canonical order with no reassociation. A query's type records whether
  it is monotone; a non-monotone read over replicated state (E.2b) waits
  for a cut (the partial-order proposal) before it answers.
- **Recursion.** The path vocabulary is trails only (no repeated edge);
  selectors are out. Stratified fixpoints are evaluated semi-naively and
  are reconciled with HS-8's clause that recursive queries are not in
  D.5 by placing them in a named extension, D.5r, with its own test set.
- **Indexes may lag.** An index carries the seq of the prefix it reflects
  and the digest at that seq (HS-7). Its incremental form is DBSP's, with
  the set/bag distinction explicit in the index type. A sorted map is the
  reference structure; provenance (graph ids, and semiring annotations if
  they are ever adopted) lives in projections, never in D. A foreign
  store holding an index is a cache that is refolded, as HS-7 says.
- **The graph is an index.** E.8.1's property graph is a derived index
  with typed query values, not a second data model.

Why: the consistency-without-coordination record (CALM, the monotonicity
results of Laddad and colleagues) names the exact condition under which
a read over replicated state needs no cut; the DBSP paper gives the one
incremental form that is correct for both sets and bags; the graph
query standards converged on trails and on typed results.

## 7. The partial order

- **T62's obligation.** An op-based merge is expressed as `deps` only.
  Atomicity across chains is one entry or one cut, never a layered
  two-phase commit. T62 carries, as an explicit proof obligation per
  program, that dep-independent events commute; `rce check` reports the
  pairs of steps for which it cannot show commutation.
- **Offline chains.** An offline device appends to its own chain and
  refolds its pending chain after the admitted prefix once it reconnects;
  it never advances its admitted prefix while holding unadmitted
  entries. A browser is a replica with a pending chain, nothing more.
- **Attributable divergence.** A replica acknowledges with the pair
  (log-head hash, state hash). Two replicas with the same log head and
  different state hashes have a host fault (HS-3); two with different log
  heads name which one left the log. No surveyed system states this pair.

## 8. Assurance and energy

- **A deterministic host simulator.** The reference host runs under a
  simulator with a stubbed clock, network and append, which injects the
  anomalies the contract permits (reordering, delay, torn writes, a lost
  replica) and a power fault at any append, and checks that every
  replica's fold yields the same state hash. The simulator is the
  FoundationDB and TigerBeetle method applied to the reference code, not
  to a model of it.
- **Energy.** Every figure is labelled `est_j`, `reported_j` or
  `measured_j` (4.4 of the paper; `energy-calibration.md`), reported as a
  ratio to work and as the raw interval with the idle baseline
  subtracted. For a replicated deployment the unit is committed events
  per joule. No per-event advantage is claimed until a `measured_j` figure
  exists. The baselines the record offers are the 2010 audited figure of
  5.93 W per kilo-tpmC and the 2017 phone measurement of about 800 µJ per
  kilobyte; the first `measured_j` figure RCEdb publishes with an idle
  baseline will be, as far as the survey could find, the first of its kind
  for a database.

## 9. From Delta

The conversation that produced an event is itself a sequence of events in
D, so an answer's provenance is a fold like any other; emitted artefacts
anchor to Git blobs and commits through the emit-target registry
(`emit-targets.md`). Neither needs a new rule, only the two existing ones
(HC-10: inputs are events; HS-7: indexes are projections).

## 10. What stays open

The survey leaves eleven questions that the text should list as open
rather than decide by silence: multiple logs versus sharded chains and
identity across logs; batch-versus-event transaction semantics; halt
versus repair-from-replica on a verify-on-open mismatch; Viewstamped
Replication versus Raft for failover; in-process versus whole-VM
simulation; third-party auditability (split-view detection); a certified
optimiser and worst-case-optimal join evaluation for cyclic patterns; a
total order over bags with floats and nulls; a text or sequence rule for
collaborative editing; `F_FULLFSYNC` on Apple platforms for D; and the
energy of a durable append.

## 11. Changes to the normative text, if accepted

| Where | Change | C.15 class |
|---|---|---|
| E.3, HS-1 | durability per OS and per replica count; the `durability` field of `host.StorageCaps` | outside programs |
| E.3, HS-4, HS-5 | suffix verification from a checkpoint; checkpoint bytes as a snapshot root | outside programs |
| E.5 | entries `host.Header`, `host.Checkpoint`, `host.Program`; `host.Synced` carries the entry hash | outside programs |
| E.8, HS-7 to HS-9 | lag seq and digest on an index; restated HS-9; total canonical order; monotone reads; D.5r | outside programs; D.5r is an extension |
| 6.2, 5.1 | the `Evicted` error and `τ ! Evicted` on a committed payload | extension (admits only programs refused before) |
| Appendix F | `#history(n)`, `upcast from`, `monotone` in a query type | proposed syntax, not admitted |
| E.10 | the simulator and power-fault harness as conformance tests; `verified_host=false` and the trusted base in the report | outside programs |

Evidence that would close the proposal: the simulator finding no
divergence over a stated number of injected anomalies with replay equal
to live; a verify-on-open that gives the same answer with and without
checkpoints on every host test; and example 55 re-run under HS-10 with
the same state hashes as under the current text, since the commitment is
of the same positional payload.
