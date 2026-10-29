# Interpreting rce-conformance tests

Format `rce-test/1`. A test is an RCE source file whose metadata block sits at
the end of the file, inside comment lines:

    #--- rce-test
    # id: numbers/i64-add-wrap-003
    # area: numbers
    # clauses: N2, N4
    # expect: run
    # outcome: trap:overflow
    #---

Values are JSON when they parse as JSON, otherwise comma lists, otherwise text.

| key | meaning |
|---|---|
| id, area | stable test id and area directory |
| clauses | clause ids from `clauses.json` the test exercises |
| expect | `accept`, `reject`, `run`, `trace`, `host`, `replicate` |
| diagnostic | for `reject`: the rule code the checker must report |
| events | event list applied before or instead of `fn main` |
| fuel | fuel bound for `fn main` runs |
| schedule_seeds | par schedules that must all give the same result (T16) |
| outcome | `done`, `deadlock`, `trap:<kind>`, `refused:<error>` |
| outputs | expected emitted lines |
| trace_head | for `trace`: the hash of the last entry of D (6.1) |
| trace_lines | for `trace`: the whole trace as the reference prints it, one item per line: the state after each event, each entry of D as `d applied HASH BODY` or `d kept HASH BODY` (BODY the canonical JSON that HASH is the SHA-256 of), the outputs, the reason an event was kept, `trace HEAD` and the example's `fuel N` |
| steps | for `host`: batches, sourced offers and window closes |
| tear, host_expect | host contract script and expectations |
| nodes, script, replicate_expect | replication scenario |
| fixtures | support modules (`*_FIXTURE.rce`), never run alone |
| flags | e.g. `no-emit` |
| provenance, note | where the test came from |

A trace passes only when every line equals the reference's, byte for byte;
the head alone commits to the events and state hashes, not to the outputs,
the reasons events were kept or the fuel.

Laws the runner checks on every applicable test, independent of metadata:
on every trace the hash chain of 6.1 (each HASH is the SHA-256 of its
BODY, `seq` counts from 0, each `prev_hash` is the hash before it, the head
is the last hash) and T8 (a kept event leaves the state hash as it was),
agreement across schedule seeds, HC-8 and HC-17
(replay = reopen = live state), committed-log prefix consistency in
replication. An adapter may answer `unsupported` for an op or a case; that is
reported separately and never counted as a pass.
