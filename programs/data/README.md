# Data: records and aggregation

Programs that summarise records.

| Program | What it shows |
|---|---|
| `totals.rce` | a list of `Row` structs, a total, and a group-by into a `Map` |

Limits in the current draft: there is no query construct; in these
programs data lives in values and states. A persistent store exists outside
the language: RCEdb (`rce db D CMD`, and `rcedb/rcedb.rce` at the top of
this repository) holds tables and edges as a fold of the log D.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
