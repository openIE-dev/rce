# Proposal: admission receipts, rule modes and the attention count

Status: accepted (ledger row of 2026-11-02). Implemented in
`spec/checker/src/admission.rs`, `receipt.rs` and `alter.rs`, specified in
7.1 (RM-1 to RM-3, AC-1) and 10.1, 10.2, 10.6, 10.7 and 10.15 (AR-1 to
AR-5) of the specification, and tested in `tests/admission_receipts.rs` and
`tests/rule_modes.rs`. Owner: David Charlot.

## 1. Source

Two ideas are taken from DeepSeek's `dsh-superagent-prototype`
(`github.com/deepseek-ai/dsh-superagent-prototype`, MIT license), an agent
system whose authorization design (`docs/002-authorization.md`,
`src/policy.ts`) gives every accepted unit of work a mandate and every
acceptance rule a mode. Only the ideas are taken; no code or text of that
repository is copied into this one.

1. **Mandate.** Every accepted unit of work carries an immutable record of
   why it was admitted: the sender, the rules matched and the route. The
   policy is hashed into a version. When the policy changes, open work is
   rechecked, and a mandate that no longer holds is revoked before the work
   resumes.
2. **Accept modes `off`, `observe` and `auto`.** In `observe`, the default
   there, a positive assessment is recorded as "would accept" and nothing
   is done; teams review the records and then switch to `auto`. Each
   assessment judges every enabled rule exactly once.

## 2. Terms

- **Admission**: the gate of 6.9, 6.11 and 6.12 that a program passes when
  every example, check and law holds and no obligation is vacuous.
- **Rule set**: the table of 7.1 (every static rule with its method) and
  the modes table of 7.1 (every altered rule with its mode).
- **Admission receipt**: the record of one admission (10.15).
- **Mode**: `off`, `observe` or `auto`, the mode of an altered rule (7.1).
- **Attention line**: the count of what a run leaves for a person (AC-1).

## 3. Mapping

| Source | RCE | Where |
|---|---|---|
| unit of work | a program: the modules given to `rce check` together | 4.1, 10.2 |
| mandate, immutable | admission receipt: canonical JSON, byte-stable, written once by `rce check write RECEIPT` | AR-1, AR-2 |
| sender | the program's name-free hash (10.8): what was admitted, independent of who typed it or how it is laid out | AR-1 `program` |
| rules matched | the whole rule set in force, each rule with its method and mode, and every obligation with its fuel or counts and result, what W-MUT-1 measured and the law attestations | AR-1 `rules`, `obligations`, `mutation`, `attestations` |
| route | the checker that judged: commit, specification draft and release label | AR-1 `checker` |
| policy hashed into a version | the rule-set hash, `rce-rules/1` over the canonical rule table | AR-1, `RuleSet::sha256` |
| recheck open work on a policy change | `rce check receipt RECEIPT FILE...` recomputes admission and names what changed: the program, the rule set (rules added, removed, changed, and whether the program is still admitted) or the checker | AR-4 |
| revoke a mandate that no longer holds | a receipt that does not hold is reported as a finding that requires change (status 1); nothing is revoked, because a receipt never granted anything | AR-3, AR-4 |
| accept modes `off`, `observe`, `auto` | the modes of the altered rules | RM-1, RM-2 |
| `observe`: "would accept", nothing done | `rce fmt` records each rewrite it would write as a canonical diff with its rule and writes nothing; `rce check` prints the report as `observed` and admits | RM-2 |
| review, then switch to `auto` | promotion: a change of the modes table, which changes the rule-set hash, so every earlier receipt reports it | RM-3 |
| each enabled rule judged exactly once | every rule of the rule set in force is judged once per run by the one admission code (`admission.rs`); a rule that is `off` is not judged | RM-2, 10.1 principle 3 |

Two points of the source do not carry over, and the reasons are stated.

- **A mandate is an authority there; a receipt is evidence here.** In the
  source, work runs because it holds a mandate. In RCE admission is a
  function of the program text and the rule set (10.1, principle 1), so
  computing it again gives the answer itself, and nothing needs to be
  trusted in its place. A receipt that could admit a program would be a second way to
  be admitted and could disagree with the first. So no tool reads a
  receipt to decide anything, and revalidation recomputes admission every
  time (AR-3).
- **`observe` is not the default for every rule.** Section 5 gives the
  default per rule.

## 4. Composition before addition (CONTRIBUTING rule 2)

What exists today, and why it does not suffice:

- `rce hash` prints the program hash, and an attestation (6.11) binds one
  law to the hashes of what it covers. Neither records which rules were in
  force, which checker judged, or what each obligation measured, so a
  change of the rule table is invisible to both.
- A recommended rule (7.1) is reported and never applied, which is what
  `observe` does for an altered rule; but a recommended rule has no
  rewrite, so there is no diff to record and nothing to promote. A mode is
  therefore defined only for altered rules, and a recommended rule keeps
  its method (RM-1).
- Changing an altered rule's method to recommended and back would act as a
  mode, but it would change the meaning of the method column (a rule with
  one rewrite would be listed as having none) and would lose the record of
  the rewrite while it is under review.

No construct of the language is added: the change is to the rule table,
which is data, and to the tools.

## 5. Decisions

1. **Modes are data, in the rule table.** The modes table of 7.1 is read
   by `rce-spec-data` into `spec/data/spec-data.json` (`w_modes`) and
   compiled into the checker with the method table. No program text,
   comment or command word sets a mode, so the rule "a comment line never
   changes what a program means" and the positional command line are kept,
   and two runs of one checker on one text always judge the same rules.
2. **Only altered rules have a mode (RM-1).** A rule that requires change
   cannot be switched off by any table row: `RuleSet::from_table` reads a
   table with a mode on such a row as invalid, `RuleSet::with_mode` leaves it unchanged, and
   `tests/rule_modes.rs` checks that the modes table names exactly the
   altered rules.
3. **Defaults, from the repository's own rules.** The two altered rules,
   W-UNUSED-1 and W-DROP-1, were written by `rce fmt` before modes existed;
   every program, example and library module of the repository is in the
   form their rewrites give, and their rewrites keep the meaning by T7 and
   by evaluation. They keep `auto`, so nothing `rce fmt` writes, and no
   verdict of `rce check`, changes. A rule that becomes altered later
   enters as `observe` (an altered rule without a row is read as
   `observe`), so its rewrite is reviewed on real programs before `rce fmt`
   writes it; it is promoted by changing its row to `auto`.
4. **`observe` does not block admission.** In the source, an observed
   acceptance does nothing. In RCE the corresponding statement is that an
   observed report neither blocks admission nor changes a file; it is
   printed (`observed` lines) and counted. Under `auto` the same report
   requires change until `rce fmt` writes it, as today. A promotion can
   therefore turn an admitted program into one that requires change, and
   that is exactly what a receipt written before the promotion reports
   (RM-3, AR-4).
5. **Observed rewrites are worked out after the `auto` ones.** `rce fmt`
   writes the `auto` rewrites, then works out the `observe` rewrites on that
   text in the same rounds as `alter.rs` uses, and records each as a
   canonical diff (`repair::diff`) from the text the earlier recorded
   rewrites leave. Applying the diffs in order gives the text that the same
   rules would write in `auto` after the others, and when the observed rules
   are the only ones judged, exactly the text `auto` writes
   (`tests/rule_modes.rs`, on every fixture of `spec/examples-0.9/altered/`
   and on a text with several reports of both rules).
6. **The receipt is canonical and order-free.** Canonical JSON (6.1.1) and
   one line end. Obligations are named by an obligation hash over their
   name-free text, not by line, and sorted, because the program hash is
   name-free, layout-free and order-free: a renaming, a comment or a moved
   definition gives the same receipt (`a_receipt_is_byte_stable`). The
   receipt holds no file names and reads no clock.
7. **Hash-bound, as an attestation.** `binding` is `PROGRAM;RULES;CHECKER`,
   the form of an attestation (`WHO;YYYY-MM-DD;COVER`, 6.11) for the three
   things the receipt is bound to. As with the break-even receipt of D.11,
   what can be derived is derived again when a receipt is read: a receipt
   whose rule-set hash is not the hash of the rules it lists, whose binding
   is not made of its own members, or whose bytes are not canonical is
   invalid (AR-5).
8. **Precedence of the revalidation outcomes.** The program first (a
   receipt of another program says nothing about this one), then the rule
   set, then the checker. The rule set is compiled into the checker, so
   every change of rules also changes the checker commit; naming the rules
   first says what changed. When only the checker changed, the outcome also
   says whether the record it gives is the receipt's, which separates a
   new build that judges alike from one that does not.
9. **The attention line goes to standard error.** The verdict lines of
   `rce check` and `rce fmt` on standard output are read by scripts, the
   conformance adapters and the book's listings, so they keep their form;
   the summary for a person is one more line, on standard error, at the
   end of every run. Only requires-change findings make the status 1; under
   `fmt check`, a rewrite not yet written and a layout to change are such
   findings, as they were.
10. **One admission code.** `rce check`, `rce test`, `rce index`,
    `rce-check` and the language server all reach their verdict through
    `admission::module`, so the modes apply to every tool alike and a
    receipt written by `rce check` describes the admission each of them
    computes.

## 6. Not done, and why

- `rce index` does not print receipts in its catalog and the language
  server does not publish observed reports; both use the same admission
  code, so either can be added without a second path.
- No rule is `observe` or `off` today, so the paths of those modes are
  reached only through rule sets built in the tests. The first rule that
  becomes altered after this change will be the first in `observe`.
- Nothing is revoked automatically, because a receipt grants nothing
  (section 3).

## 7. Checks

`tests/admission_receipts.rs` (AR-1 to AR-5, RM-3) and
`tests/rule_modes.rs` (RM-1, RM-2, AC-1); the conformance table of 10.15
lists each rule with its test.
