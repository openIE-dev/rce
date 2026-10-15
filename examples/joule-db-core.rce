spec "joule-db-core" v1.0.0

reason "Energy-aware database engine — B-tree storage, WAL, query engine, 7 query languages" by "david"
currency max_age=168h

# ── Objectives ──────────────────────────────────────────────────

objective O1 "ACID-compliant transactional storage"
  metric transaction_throughput target 10000
  reason "Foundation for all JouleDB operations — correctness is non-negotiable" by "david"

objective O2 "Support 7 query languages"
  metric query_languages target 7
  reason "SQL, Cypher, CQL, GraphQL, Datalog, SPARQL, Gremlin — unified engine" by "david"

objective O3 "Sub-millisecond point lookups"
  metric p99_lookup_ms target 1
  reason "B-tree must be fast for the cascade L0 LUT tier" by "david"

objective O4 "Crash recovery via WAL"
  metric recovery_time_seconds target 5
  reason "WAL must replay cleanly after any crash — no data loss" by "david"

objective O5 "Energy metering on every operation"
  metric operations_with_energy_receipt target 100
  reason "Every read/write/query produces a picojoule energy receipt — the core differentiator" by "david"

# ── Constraints ──────────────────────────────────────────────────

constraint C1 substrate "Pure Rust, no C dependencies, no unsafe in hot path"
  reason "Memory safety is a hard requirement — 6000+ tests verify correctness" by "david"

constraint C2 performance "Max 64MB memory for 1M row table"
  reason "Must run on edge devices (Cortex-M, RISC-V MCU)" by "david"

constraint C3 compatibility "pgwire protocol for PostgreSQL client compatibility"
  reason "psql, DBeaver, any PostgreSQL client must connect" by "david"

# ── Acceptance Criteria ──────────────────────────────────────────

acceptance A1 validates O1
  property "for all valid transactions, COMMIT persists and ROLLBACK discards"
  reason "ACID A — atomicity" by "david"

acceptance A2 validates O1
  property "for all concurrent transactions, no dirty reads or phantom reads"
  reason "ACID I — isolation" by "david"

acceptance A3 validates O2
  property "for all 7 query languages, parse and execute on sample data without error"
  reason "Language coverage must be complete" by "david"

acceptance A4 validates O3
  property "for all primary key lookups, latency < 1ms at 1M rows"
  reason "B-tree point lookup SLA" by "david"

acceptance A5 validates O4
  property "for all crash scenarios, WAL replay produces identical state"
  reason "Durability — crash recovery must be lossless" by "david"

acceptance A6 validates O5
  property "for all operations, energy_pj field is non-zero"
  reason "Every operation must have energy attribution" by "david"

# ── Deliverables ──────────────────────────────────────────────────

deliverable D1 fulfills O1
  schema "TransactionResult"
  reason "Commit/rollback outcome with affected row count" by "david"

deliverable D2 fulfills O2
  schema "QueryResult"
  reason "Unified query result across all 7 languages" by "david"

deliverable D3 fulfills O3
  schema "LookupResult"
  reason "Single-row point lookup result with energy receipt" by "david"

deliverable D4 fulfills O4
  schema "WalReplayResult"
  reason "Recovery outcome with replayed entry count" by "david"

deliverable D5 fulfills O5
  schema "EnergyReceipt"
  reason "Per-operation picojoule accounting" by "david"

# ── Roles ──────────────────────────────────────────────────────────

role R1 "btree-engine" execution_node
  reason "B-tree storage engine — page-level read/write with overflow chains" by "david"

role R2 "wal-manager" execution_node
  reason "Write-ahead log — append, checkpoint, replay" by "david"

role R3 "query-executor" execution_node
  reason "Parses and executes queries across all 7 languages" by "david"

role R4 "transaction-manager" execution_node
  reason "MVCC transaction coordination — begin, commit, rollback, isolation" by "david"

role R5 "energy-meter" execution_node
  reason "Tracks picojoule cost per operation from the calibrated energy table" by "david"

role R6 "pgwire-server" execution_node
  reason "PostgreSQL wire protocol handler for external client access" by "david"

role R7 "buffer-pool" execution_node
  reason "Page cache with LRU eviction — hot pages stay in memory" by "david"

role R8 "db-coordinator" coordinator
  reason "Coordinates cross-component operations (checkpoint, compaction, schema changes)" by "david"

# ── Dependencies ──────────────────────────────────────────────────

dependency DEP1 crate "jpb-core"
  reason "Shared constants and energy calibration tables" by "david"

# ── Escalation Conditions ──────────────────────────────────────────

escalation E1 from "btree-engine" to R8 timeout 5000ms
  reason "Page split or overflow chain creation — coordinator must update metadata" by "david"

escalation E2 from "wal-manager" to R8 timeout 10000ms
  reason "WAL checkpoint needed — coordinator decides when to flush" by "david"

escalation E3 from "query-executor" to R8 timeout 30000ms
  reason "Query plan too expensive — coordinator may reject or rewrite" by "david"

escalation E4 from "transaction-manager" to R8 timeout 5000ms
  reason "Deadlock detected — coordinator must choose victim" by "david"

# Data flows: real edges so a task takes more than one step.
flow F1 from DEP1 to R6
  reason "calibration tables configure the wire server" by "david"

flow F2 from R6 to R3
  reason "pgwire queries go to the executor" by "david"

flow F3 from R3 to R4
  reason "writes run inside a transaction" by "david"

flow F4 from R4 to R2
  reason "committed transactions are logged to the WAL" by "david"

flow F5 from R2 to D1
  reason "WAL acknowledgement is the transaction result" by "david"

flow F6 from R3 to D2
  reason "reads return a query result" by "david"

flow F7 from DEP1 to R5
  reason "calibration constants drive the energy meter" by "david"

flow F8 from R5 to D5
  reason "meter readings become energy receipts" by "david"

change_control approvers ["david"] review_period 48h
