# Proposal: evidence authority, when a lift's claim closes

Status: accepted and implemented 2026-11-02 (ledger row of that date). It
is specified in Appendix D.8 of the specification, L5 and L6, and in
sections 2 and 4.4 of [compile/CHARTER.md](../../compile/CHARTER.md) for
RCE Compile's gates. Owner: David Charlot.

## Terms

- **REA** (Reverse Engineer Anything, <https://github.com/morluto/rea>,
  MIT License, copyright 2026 morluto): a reverse-engineering server whose
  reconstruction ledgers decide when a claim about a reconstructed program
  is closed. Its documents `docs/reconstruction-obligation-ledgers.md`,
  `docs/reconstruction-coverage.md` and `docs/mcp-contracts.md` (read at
  `main` on 2026-11-02) are the source of this proposal. No code of REA is
  copied; the rule is restated in RCE's words and implemented anew.
- **Lift**, **lift table**, **oracle**, **observation** and **case**: as in
  [rce-form.md](rce-form.md) (sections 1, 16 and 20).
- **Claim**: a statement about one lift with an id, a revision, the
  authority level it requires and the names of the digests it is bound to.
  REA calls it an obligation.
- **Evidence record**: one verifier's answer to one claim: the claim's id
  and revision, the verifier, an authority, a kind, a result (`holds`,
  `differs` or `undecided`) and digests.
- **Digest**: the SHA-256 of a file's bytes, written as 64 lowercase hex
  digits.
- **Kind**: *observed* (a run of a pinned implementation), *derived*
  (computed from definitions) or *inferred* (proposed by a model or a
  heuristic). REA calls it confidence.

## In plain words

1. A lift says something about a program of another language: that its
   Form came from rows of a table, that it runs as the original ran, that
   it behaves as a hand-written program does. Each of these is a claim, and
   each needs evidence of its own strength.
2. A claim closes only when evidence of the strength it requires answers it
   by name and revision and is bound, by digest, to the exact source, lift
   table, lifter, lifted text and checker the report is about.
3. Evidence that a model or a heuristic produced never closes a claim. A
   verifier that passes on such evidence does not turn it into an
   observation; the claim is `undecided`.
4. A digest that does not match fails the claim closed. A record that
   differs fails it too, and no record that holds can hide it.

## 1. The principle

[rce-form.md](rce-form.md) accepts a lift by definition and exact
comparison, and Appendix D.8 already says that a lift is "evidence, not
proof" and "an input to a gate, never an authority". What it did not say
is how strong a piece of evidence must be for a given claim, or which
inputs the evidence must name. REA's ledgers answer both: closure needs a
verifier whose authority is comparable to the original observation, named
for the obligation, and bound to the exact implementation digest and
obligation revision. This proposal takes that rule into the lift
guarantees as L5 (the levels) and L6 (closure).

## 2. What was taken from REA

| REA | RCE | Where |
|---|---|---|
| an obligation closes only with a unique owner, every required case and a passing verifier whose authority is comparable to the original observation | a claim closes only with a record that holds at a level that closes the claim's required level, of that level's kind (L6) | D.8 L6; `spec/lift/authority.py`, `spec/checker/src/lift_authority.rs` |
| the verifier enumerates the obligation ID; its result is bound to the owner digest and the obligation revision | a record names the claim's id and revision and is bound to every digest the claim states | L5, L6 |
| authority `shipped-artifact` | `pinned`: the pinned implementation, its executable bound by the digest its oracle record names, run on the case now, byte equality | L5 table |
| authority `controlled-replay` | `replay`: an observation recorded from the pinned implementation, bound by digest, byte equality | L5 table |
| Evidence confidence `observed`, `derived`, `inferred` | kind `observed`, `derived`, `inferred` | L5 |
| a passing verifier summary cannot promote inferred Evidence into an observation: `unknown` | inferred evidence, or evidence of a kind below its stated authority, leaves the claim `undecided`, the word of `rce equiv` | L6 |
| contradictions, duplicates, residual unknowns and missing dependencies keep closure open | two records of one verifier, a stale revision, a dependency not closed: `open`; an applicable record that differs: `failed` | L6 |
| a failed applicable proof fails closure even when another proof passes | a record that differs fails the claim; no holding record hides it | L6 |
| integrity failures fail closed; a tampered record is an input error | a malformed record or a digest mismatch makes the claim `failed` and the report exit 4 | L6 |
| the producer documents the exact hashed inputs | every digest is the SHA-256 of one file's bytes, named in the report (`source`, `table`, `lifter`, `form`, `checker`, `recording`, `oracle`, `counterpart`) | L6, section 4 |

RCE adds one level REA does not have: `bisimilar`, the verdict of
`rce equiv` (`equivalent`, with neither exploration reaching its bound),
which is a proof of bisimilarity on the explored alphabet (T58) between
the lifted Form and a hand-written RCE program. It is derived, and it closes
only claims that require it. And RCE names REA's `derived` confidence as a
level of its own, `derived`: every construct of the lifted program comes
from a lift-table row that cites its clause of the defining reference
(K4 of rce-form.md).

## 3. What was left out

- REA's third level, `live-observation` (a running service watched in
  production), and its `external` contract: a lift is compared on pinned
  cases, never on a deployment, so no lift claim requires either. A record
  that names them is malformed and fails its claim closed.
- REA's manifest of owners, parser and schema types, case kinds
  (`positive`, `negative`, `malformed`, ...) and its MCP tools, CLI and
  Evidence bundles. The probes' own rules (`L-GO-*`, `L-PY-*`) and the case
  classes of rce-form.md section 20 already say which constructs and cases
  a lift covers.
- REA's ledger and closure digests over a whole result, and its
  `by_evidence_authority` summaries. A lift report is one JSON document; its
  bytes are its identity.
- Static application graphs, process capture, package proof and anything
  about binaries: RCE lifts source text.

## 4. The claims of the probes

The claims are data (`spec/lift/authority.json`, `claims`). A report states
each one with its bindings, its evidence and its verdict.

| Claim | Requires | Evidence the probe makes | Bound to |
|---|---|---|---|
| GO-ROWS | `derived` | `lift_go.py rows`: the rows of `spec/lift/go/table.json` the Form was built from, each with a clause of the Go specification and a derivation; `differs` if a row used has neither | source, table, lifter, form, checker (the lifter) |
| GO-AGREE | `replay` | none: the probe runs no Go, and `door.go` has no `main` for the Go oracle of `lang/go1.24` to run, so the claim stays `open` | source, table, lifter, form, checker, recording |
| GO-EQUIV | `bisimilar` | with `report FILE RCE HAND`: `rce equiv json` on the lifted Form and HAND; `holds` when equivalent and exact, `undecided` when a bound was reached, `differs` otherwise | source, table, lifter, form, checker (the `rce` binary), counterpart (HAND) |
| PY-RECORDED | `replay` | with `report FILE RCE RECORDED`: the lifted program's standard output under `rce run` against RECORDED, byte for byte | source, table (`runtime.rce`), lifter (`lift_py.py`), form, checker, recording |
| PY-PINNED | `pinned` | with `report FILE RCE`, when the interpreter of `lang/python3.13/oracle/cpython-3.13.16.toml` is present: that interpreter run now on FILE with the record's run line and environment, against the lifted program's standard output | source, table, lifter, form, checker, oracle (the executable's digest, which must equal the record's `sha256`) |

The Python lift has no table rows that cite clauses: its constructs are
carried by code, with `runtime.rce` as its data. It therefore states no
`derived` claim, and rce-form.md's K4 remains a plan for it.

The committed report `spec/lift/go/door.report.json` is the Go probe's
report on `door.go` without RCE: GO-ROWS `closed`, GO-AGREE and GO-EQUIV
`open`. With `rce` and `spec/examples/61_dim_minimal.rce`, GO-EQUIV closes at
`bisimilar` (3 and 3 states over 4 event instances, exact). The Python
lift's reports on its positive fixtures close PY-RECORDED with
`fixtures/pos/NAME.stdout` and PY-PINNED with CPython 3.13.16 present.

## 5. Why no W- rule

Section 7.1 accepts a row only for a rule the checker enforces at
admission (`spec/checker/tests/three_methods.rs`). No admission reads a
lift report: a lifted program reaches the checker as a proposal (L4) and is
admitted or requires change under the rules of section 7 alone. So L5 and
L6 are rules of the lift report, checked by `tests/lift_authority.rs`, and
no W- rule is added.

## 6. Tests

- `spec/lift/authority-vectors.json`: 30 vectors, one claim and its
  records each, with the verdict and reason; every level closes a claim it
  is meant to close and no other, inferred evidence never closes one
  (alone, as a stated `replay` or `pinned`, or beside a qualifying record,
  which still closes), digest mismatches and malformed records fail closed,
  a record that differs is not hidden, and stale revisions, duplicates and
  open dependencies keep a claim open. `spec/lift/test_authority.py` and
  `spec/checker/tests/lift_authority.rs` give every vector's verdict and
  reason.
- `tests/lift_authority.rs` also checks that D.8's table and
  `authority.json` agree, that `door.report.json` is bound to the digests of
  its five files and states every claim of the Go probe with the verdict
  the rule gives, that a copy with one digest changed fails, that a copy
  whose evidence is marked inferred is undecided, and that the record
  `rce equiv` gives for the lifted and the hand-written door closes GO-EQUIV
  at `bisimilar`.
- `spec/lift/go/test_lift_go.py` and `spec/lift/python/test_lift_py.py`
  run the probes' `report` commands: the committed Go report byte for
  byte, GO-EQUIV closed with `rce` and the hand-written door and failed with
  a counterpart that differs, PY-RECORDED and PY-PINNED closed on every
  positive fixture, and a recording of another fixture failing the report
  with exit 4.

## 7. RCE Compile

[compile/CHARTER.md](../../compile/CHARTER.md) section 4.4 requires each
gate to name its property, its linking assumption and its trusted
computing base. It now also names the authority of the gate's evidence and
the digests its verdict is bound to, with the same rule that inferred
evidence never makes a gate accept. The charter states this in its own
terms, since RCE Compile depends on none of RCE's checker, Form or
specification; E2's and E3's protocols record both for every gate and
every certified level, and HYPOTHESES.md records the change to the metrics
of H2 and H3.

## 8. What would show it wrong

A claim that closes on evidence of a lower authority, on inferred
evidence, or on a record bound to another digest; a report whose stated
verdict differs from the rule's on its own evidence; or a vector on which
`authority.py` and `lift_authority.rs` disagree.
