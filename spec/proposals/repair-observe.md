# Proposal: rung f of `rce repair`, observe mode and a registered promotion criterion

Status: accepted (ledger row of 2026-11-03). Implemented in
`spec/checker/src/repair.rs` (rung f) and `proposals.rs` (the proposals
file, the evaluator and the mode-row check), specified in 10.13 (RF-1 to
RF-7) of the specification, and tested in `tests/repair_observe.rs`. The
promotion criterion is registered in
[repair-observe/criterion.json](repair-observe/criterion.json), with its
SHA-256 in `criterion.json.sha256`. Owner: David Charlot.

## 1. Sources

**The principle.** RCE is deterministic first: at least 98% of what the
tools decide comes from exact, deterministic means and at most 2% from
stochastic ones, and every stochastic output passes a gate before it is
used. Hypothesis H6 of `compile/HYPOTHESES.md` states the same share for
the rungs of a compiler. The exact-proposer ladder of `rce repair` (10.13)
is the deterministic part: its third registered run closed 111 of 151 tasks
with no model call. This proposal adds the stochastic part and keeps it
inside the share.

**The source ideas.** Three ideas are taken from DeepSeek's
`dsh-superagent-prototype` (`github.com/deepseek-ai/dsh-superagent-prototype`,
MIT license), whose authorization design (`docs/002-authorization.md`,
`src/policy.ts`) already gave RCE its rule modes (7.1, RM-1 to RM-3, in
[admission-receipts.md](admission-receipts.md)). Only the ideas are taken;
no code or text of that repository is copied into this one.

1. **Accept modes `off`, `observe` and `auto`.** In `observe` a positive
   assessment is recorded as "would accept" and nothing is done; after
   review the mode is switched to `auto`.
2. **The model assesses, deterministic policy decides.** A model's output
   is an input to a rule, never the decision.
3. **As little human attention as possible.** What a run leaves for a
   person is counted, and a record that needs no action does not ask for
   one.

## 2. Terms

- **Exact rung**: a rung of 10.13, a to e, whose candidates are computed
  from the program, its modules, GOOD and the library by a fixed rule.
- **Proposal**: a candidate program for FILE written outside RCE, by a
  model (hosted or local) or by a person, with its **origin** label.
- **Rung f**: the rung that judges proposals, after the exact rungs.
- **Stochastic**: not computed by a fixed rule from the ladder's inputs, so
  equal inputs to the ladder do not give it. Every proposal is stochastic in
  this sense, whatever its origin.
- **Observed proposal**: a proposal the checker admits while rung f is in
  `observe` mode.
- **Close**: a task whose program the ladder closes.

## 3. Mapping

| Source | RCE | Where |
|---|---|---|
| a model assesses | an outside proposer writes candidate programs into a file; RCE reads the file and calls no model | RF-1, 10.1 item 2 |
| deterministic policy decides | the gate of every rung, `check_program` with the mutation gate (W-MUT-1), judges each proposal; a proposal that changes what judges it requires change | RF-2 |
| accept modes `off`, `observe`, `auto` | the modes of the altered rules (7.1), given to rung f by a modes table of 10.13 that `rce-spec-data` reads | RF-3 |
| `observe`: "would accept", nothing done | each proposal the checker admits is reported as `observed`, with its diff, digest and origin; the program stays open and nothing is written | RF-3, RF-4 |
| review, then switch to `auto` | promotion is a change of the mode row, allowed only when a criterion registered before any observation holds on a recorded evaluation | RF-5, RF-7 |
| as little human attention as possible | the attention line of AC-1 counts observed proposals as observed, not as findings that require change | RF-4 |

One point of the source does not carry over. There, a person reviews the
recorded acceptances and then switches the mode. Here the review is a
deterministic evaluation of a criterion registered in advance (RF-6), and
the mode row may change only when that evaluation, derived again from its
counts, shows the criterion met (RF-7). A person still makes the change,
but no reading of the records can make it valid.

## 4. Composition before addition (CONTRIBUTING rule 2)

What exists today, and why it does not suffice:

- A person can already edit FILE and run `rce check`. That admits the
  edit, but it records neither where the edit came from nor how often such
  edits would have closed what the exact rungs could not, so the share of
  the principle cannot be measured.
- `rce repair known GOOD` takes one outside text, but as a known-good
  version that steers rung e, not as a candidate; it has no mode and no
  record of origin.
- The rule modes of 7.1 belong to altered rules (RM-1). Rung f is not a
  rule and has no rewrite, so it gets its own modes table with the same
  three modes and the same default, read by the same machinery
  (`specdata.rs`, `specread.rs`, `admission::Mode`).

No construct of the language is added. The change is to `rce repair`, a
new subcommand `rce repair criterion`, the modes table of 10.13, and the
registration.

## 5. Decisions

1. **Proposals are data (RF-1).** The file is canonical JSON (6.1.1) of
   format `rce-proposals/1`, so one list of proposals has one file and one
   SHA-256, recorded in the report as `inputs.proposals`. Each proposal is
   recorded by its digest, SHA-256 under `rce-proposal/1` of its text, as
   the rule-set hash is formed. The file is a setting word
   (`proposals PROPOSALS`), as `known GOOD` is (PC-1).
2. **The same gate, and nothing that judges it may change (RF-2).** The
   exact rungs never edit an intent, example, check or law line, or a
   `check`, `requires` or `ensures` statement. A proposal is free text, so
   rung f checks the same thing explicitly: these obligations of the
   proposal have the canonical texts of FILE's. Without this check, a
   proposal that deleted a failing example would be admitted. The judged
   candidate is the proposal's canonical text, as every candidate of the
   ladder is a canonical text. The gate is `check_program`, which runs the
   mutation gate on every admitted text, so an admitted proposal never
   rests on a vacuous obligation (W-MUT-1).
3. **Rung f runs last, and only when the exact rungs close nothing.** The
   share counts closes, and a close that an exact rung can make is never
   given to rung f. When FILE does not parse, rung f does not run, as no
   rung does, because its obligations cannot be read.
4. **`observe` judges every proposal; `auto` stops at the first (RF-3).**
   In `observe` the record is the point: every proposal the checker would
   admit is reported, and the program stays open. In `auto` rung f is a
   rung like the others: the first admitted proposal closes the program,
   and the report labels it stochastic (`"stochastic": true`, `rung f
   outside proposal (stochastic)`).
5. **The mode is data, entered as `observe`.** As for the altered rules
   (RM-3), a rung without a row reads as `observe`, and no command word
   can set the mode. Tests reach `auto` and `off` only through the library
   (`repair::repair_with`).
6. **The report records origins, not calls (RF-4).** `model_calls` stays
   0, because RCE made no call. Each proposal's record holds its origin,
   its digest and its verdict. An admitted proposal's record also holds
   what W-MUT-1 measured and, when GOOD is given, how it compares with
   GOOD: `restored` (the same program hash), `behaviour-equal` or
   `contradicts` by `rce equiv`, `undecided`, or `not compared`. The JSON
   report is canonical JSON, and it records the SHA-256 of every input
   (`inputs`) and the bound, so an evaluation can tell which run each
   record is.
7. **The attention line.** `rce repair` now ends with the attention line
   of AC-1 on standard error. A program the ladder leaves open is one
   finding that requires change; each observed proposal is one observed.
   Standard output keeps its form.
8. **The criterion is registered before any observation (RF-5).** It is
   dated data beside this proposal, pinned by its SHA-256, in the form of
   the ladder's registrations in openIE-dev/openie-research-experiments. Its
   corpus is the 151 frozen tasks of ladder runs 2 and 3, fixed by path and
   SHA-256, because those runs measured what the exact rungs close on it.
   Its observation runs each task with the task's original as GOOD, so that
   condition 3 is decided on every observed proposal; the proposer never
   sees the original. Its three conditions follow from the principle: the
   share (at most 2% of the closes from rung f, counted per task), the gate
   (every observed proposal passes W-MUT-1), and agreement with the
   known-good version (no observed proposal contradicts GOOD). It also
   requires a proposal for every open task that rung f can judge, so that a
   proposer that proposes nothing cannot meet it.
9. **One function decides (RF-6, RF-7).** `proposals::reasons` derives
   the verdict from an evaluation's counts and the registration. The
   evaluator uses it to write `holds`; `rce-spec-check` uses it to check
   the recorded evaluation when the mode row is `auto`, and refuses an
   evaluation whose `holds` disagrees with its counts.

## 6. The registered criterion

[repair-observe/criterion.json](repair-observe/criterion.json), registered
2026-11-03. In words:

1. *Share.* Every task has exactly one record, made as registered, and
   every task that rungs a to e do not close and whose mutant parses had
   at least one proposal judged. With E the tasks closed by rungs a to e
   and F the tasks they leave open on which rung f observes an admitted
   proposal: E + F > 0 and 100 × F ≤ 2 × (E + F).
2. *W-MUT-1.* Every observed proposal passes W-MUT-1.
3. *GOOD.* No observed proposal contradicts the task's original.

On the third run's known arm, rungs a to e closed 116 of the 151 tasks, so
on this corpus the criterion allows rung f at most 2 more closes
(2 of 118 is 1.7%; 3 of 119 is 2.5%).

## 7. Not done, and why

- **No observation and no evaluation exist.** Rung f is `observe`, and
  `spec/proposals/repair-observe/evaluation.json` does not exist, so
  `rce-spec-check` accepts only `observe` or `off` for it.
- **The mode is not per origin.** An evaluation is evidence about the
  proposers whose origins it records; a promotion applies to every
  proposer. A later registration can narrow the row to an origin if the
  records show a need.
- **A program with no step is not compared with GOOD.** `rce equiv`
  compares the behaviour of steps (6.10), so such a proposal is reported
  as `not compared`, and condition 3 does not count it.
- **No proposer is part of RCE.** Writing proposals, with a model or by
  hand, happens outside the repository's tools, which never call a model.

## 8. Checks

`tests/repair_observe.rs` (RF-1 to RF-7); the conformance table of 10.13
lists each rule with its tests. `rce-spec-check` prints the mode of rung f,
the registration and whether an evaluation is recorded.
