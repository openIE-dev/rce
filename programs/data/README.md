# Data: records and aggregation

Programs that summarise records.

| Program | What it shows |
|---|---|
| `totals.rce` | a list of `Row` structs, a total, and a group-by into a `Map` |

Limits in draft 0.5: there is no query construct and no persistent store;
data lives in values and states.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
