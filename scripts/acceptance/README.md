# Production acceptance gate

The gate decides whether a claim is backed by an evidence record. It is a
program. It does not finish the RCE compiler, and a passing experimental
claim is not a production compiler.

A claim is `rce-prod-claim/1`. Evidence is `rce-prod-evidence/1`. The
obligations are `obligations.json`. The validator refuses a record whose
obligation flags differ from the flags in `accept.py`.

## Commands

```
python3 scripts/acceptance/accept.py validate CLAIM EVIDENCE REVISION
python3 scripts/acceptance/accept.py validate CLAIM EVIDENCE REVISION CORPUS_DIR
python3 scripts/acceptance/accept.py seal EVIDENCE OUT
python3 scripts/acceptance/accept.py assemble CASES_DIR OUT REVISION
scripts/acceptance/drive.sh OUT
python3 scripts/acceptance/test_accept.py
```

`REVISION` is 40 hexadecimal digits. `CORPUS_DIR` is required when the
evidence corpus is `spec-examples`; pass `spec/examples`. `drive.sh` writes
one evidence file for every name in `spec-examples.txt` and every
`spec/examples/*.rce` file. A missing file stays in the denominator as
unverified. `assemble` does the same for `CASES_DIR/manifest.json` and
`CASES_DIR/modules/<id>.json`.

Exit 0 is a claim the record supports. Exit 1 is a refusal, with one reason
per line. Exit 2 is a usage error.

## What the record has to show

The denominator is the module list. A syntactic program-function count does
not make those modules ready. Each claimed backend has its own cases. A case
is agreement only when outputs, state and refusal behaviour were compared
and agree. Exit success alone is not agreement.

`omega` in the semantic budget is fuel, cost, output and memo. An empty
joule field does not fail that obligation. A claim of measured joules needs
a meter reading (`kind` `meter`, a source, a numeric reading, the same
revision). A modelled or estimated figure does not count.

`emit` and `out` stay required for a surface that includes output, including
whole-module Form evaluation. Calling the target Form does not make them
optional.

A refusal can keep a run safe. It is not support. A semantic defect and a
process error are not support either. A proof statement or a test does not
discharge an obligation. A model proof is a `theorem` line in a file that is
not a test. A production claim needs every normative obligation of its
surface, including the whole-module path through Form. An experimental claim
passes only when `scope.text` says it is not a production claim, the
exclusions are allowed for that surface, and it does not assert production
or whole-module readiness.

Evidence is bound to `revision`, to `built_from` (the checker commit, for a
repository record) and to the bytes of `obligations.json`. A stale record
does not authorise a newer claim. The positive fixtures under the tests are
labelled `test-only`.
