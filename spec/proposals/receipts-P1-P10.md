# Draft proposals: receipt fields from the Lux/MLI field mapping (P1 to P10)

Status: draft proposals, not normative. Source: Lux-Lang's
`field-mapping.md` (13 Oct 2026), section 4. Each proposal is decided by
derivation from three rules already in the spec:

- **Authority is never in a receipt.** Receipts are measurement records
  (C.15, D.11). The gate and the promotion path decide; a receipt field
  may be checked for consistency but never grants anything.
- **Least privilege and checkability.** A field is worth adding when a
  checker can verify it, or when without it a figure cannot be
  reproduced. A self-declared field that nothing can check is an
  annotation, kept outside the core schema.
- **No redundant state.** A field derivable from others is not stored,
  because a stored copy can disagree with its source.

Energy labels follow the paper (section 1): `est_j` is an analytical
estimate (a model provider's figure is an estimate too), `reported_j` is
an OS or on-chip counter, `measured_j` is a wall or shunt meter on a
named rail.

| # | Proposal | Decision | Reason |
|---|---|---|---|
| P1 | `zone` (Z1/Z2/Z3) | adopt as an extension (`ext.x_openie.zone`), not core | Describes how open-ended an answer is; useful to a reader, but neither checkable from the receipt nor an input to any engine decision. |
| P2 | `citations` required for Z1/Z2 | adopt as an extension with a receiver-side check | Checkable: a receiver with its own index can verify every hash and, for Z1, that the set is the full support. That makes it sound evidence for the receiver, but still not authority in the engine. |
| P3 | `effects`, with `llm` requiring a model-call receipt | adopt as an extension consistency rule | The rule "`llm` in effects implies an `LlmCallReceipt` (priced or explicitly unpriced)" is checkable and closes a hole where a model call hides in a G0 receipt. The effects list itself is self-declared, so it is a consistency check, not enforcement; the engine's effect discipline (5.15) stays the enforcement. |
| P4 | `reported_j_provenance` required with `reported_j` | adopt into the core schema | Soundness of the label: a counter reading without sampler, counter, interval, idle baseline and machine cannot be reproduced or compared, so a `reported_j` without it is not a reported figure. Refuse a receipt with `reported_j` and no provenance. |
| P5 | `boundary` required with `reported_j` or `measured_j` | adopt, merged into P4 (and the same object for `measured_j`) | Same reason: what the joules include is part of what was read. One provenance object, required for both labels. |
| P6 | `amortization` (per_call, batch_mean, per_solved_task) | adopt into the core schema, with a ledger rule | Summing a batch mean as a per-call figure is unsound arithmetic. Make the field required on `reported_j`/`measured_j` and refuse to add figures with different amortization in one total. |
| P7 | `priced: bool` | reject | Redundant: priced is exactly `est_j != null`, and the label is now an exact check (`PriceLabel`, commit cf075ff09: basis is `unpriced` or `unpriced: reason`, enforced both ways on gear and model-call receipts). A stored bool could disagree with the figure. |
| P8 | `state_hash` for receipt chains (receiver state = per-sender heads) | adopt as an informative profile | It is an instance of 6.1 with no change: state = {heads: {sender: envelope_id}}, event = the claim, `ts` outside every hash. It lets agent receipts sit in a D log and be checked by `rce-trace-diff.py`. |
| P9 | `n_star` break-even | already adopted (D.11), with two differences | n* is unpriced when any term is unpriced, as proposed. Differences: (a) "no break-even" is recorded as `never`, distinct from `unpriced`, since the two mean different things; (b) terms with different labels may be combined, and n* then carries the weakest label (unpriced < est < reported < measured), which is sound because the result is never claimed stronger than its weakest input. |
| P10 | `ext` object outside every hash | adopt into the core schema | Additive fields need a place that strict parsers accept and every check ignores. Receipts are not hashed today, so "outside the hash" holds trivially; the rule to state is that no check or decision reads `ext`. Namespaced keys (`x_openie`). |

Resolution of P4, P5 and P6 (14 Oct 2026): normative, by soundness,
now in RCE-SPEC.md D.11 ("Readings: provenance, boundary and
amortization"). The open question was whether to confirm the schema
change. It is decided by the same rules as the table, not by preference:
(1) a `reported` or `measured` label is a claim that something was read,
and a reading that cannot be reproduced does not support the claim, so
allowing the label without provenance would make the label unsound;
(2) adding figures of different amortization is arithmetic on different
units, so allowing it would make every total that contains it unsound;
(3) the change breaks nothing stored, because no receipt in the
repository holds a reading and the `est` wire form is unchanged. Two
details were settled in the same way: the boundary is a field of the one
provenance object (P5 merged into P4, as decided above), and an absent
amortization on an `est` term counts as its own kind, so a total of the
engine's own estimates still adds while any total that would mix a
declared kind with an undeclared one is refused.
