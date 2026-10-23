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
| trace_head, steps | expected trace digest and steps for `trace` |
| tear, host_expect | host contract script and expectations |
| nodes, script, replicate_expect | replication scenario |
| fixtures | support modules (`*_FIXTURE.rce`), never run alone |
| flags | e.g. `no-emit` |
| provenance, note | where the test came from |

Laws the runner checks on every applicable test, independent of metadata:
T8 on every trace, agreement across schedule seeds, HC-8 and HC-17
(replay = reopen = live state), committed-log prefix consistency in
replication. An adapter may answer `unsupported` for an op or a case; that is
reported separately and never counted as a pass.
