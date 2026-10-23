# Proposal ledger

One row per proposal, in the order decided. Reasons are from the fixed set
of CONTRIBUTING.md rule 5: feasibility, duplicate spelling, limited use,
breaks a theorem, breaks a principle. A proposal is *open* until it is
accepted into the normative text or declined; an accepted one names the
release and the example and theorem that admitted it.

| Date | Proposal | Decision | Reason | Evidence |
|---|---|---|---|---|
| 2026-10-22 | 6.10 behaviour, equivalence and dimension; T57 to T61; `rce dim` | open: 6.10 and `rce dim` in the text; T57, T58, T59 (explored alphabet) proved in `lake build` 2026-10-23; T60, T61 and the converses stated | | `spec/lean/RCE/Behaviour.lean`, `spec/examples/59-61`, `spec/checker/tests/dim.rs` |
| 2026-10-22 | Form IR format 2: content-addressed definitions (`form-ir-2.md`) | open | | 457 bricks, 269 lowered, 77 refused on size alone |
| 2026-10-22 | Effect handlers (F.2 non-goal 1) | declined, ground restated | breaks a theorem (T2 and T6: each extra resume clones captured state with a data-dependent count); loggability (T3) is answerable after evidence passing and defunctionalization and is not the ground | ML lineage survey, 2026-10-22 |
| 2026-10-22 | Second concurrency vocabularies beyond `spawn`/`chan`/`select` and `par` (actors, workgroups, supervisors as syntax) | open: move to library Forms over the two primitives | duplicate spelling | Appendix C gap register G1 to G24 |
| 2026-10-22 | Encapsulation criterion: U4 with five measurable coordinates, one PREREG per language (`encapsulation.md`) | open | | phase 4 gate of the roadmap |
| 2026-10-22 | D as a partial order: chains, deps, cuts; T62 (`partial-order-log.md`, `RCE/LogPO.lean`) | open; T62 model proved in `lake build` 2026-10-23 | | `RCE/LogPO.lean`: `fold_perm_eq`; link by T40 permutation tests |
| 2026-10-22 | Intent with holes, W-INT-1 non-vacuity, spectest completeness, derivation log D_I (`intent-holes.md`) | open | | nameless and ladder benches |
| 2026-10-22 | Two-regime est_j, `host.Calibrated`, RCE-F5 (`energy-calibration.md`) | open | | first run: KV260 rung 3, VCCINT, N ≥ 30 |
| 2026-10-22 | T7 finding: `state_hash` is over canonical JSON keyed by region and field names; the Lean model hashes De Bruijn keys. A: hash a positional embedding (every fixture changes; C.15 classes the D entry as outside a program's meaning, so an erratum). B: restate T7 for the implementation (6.6's 'names outside the hash' loses its ground). | open, recommendation A | | `tests/nameless.rs::state_hashes_are_not_yet_name_free` |
| 2026-10-23 | Unobservable intent: an `example` that reads a field no step ever emits or refuses on is reported by `rce check` (a diagnostic, not a refusal) | open | | dim-r1: 11 of 13 bounded state machines emit nothing |
| 2026-10-23 | Emit target registry: one Form-IR emitter with per-target tables, then every language with a substrate, in four tiers (`emit-targets.md`) | open; gated on the one-emitter change | | 5,717 generator lines with 445 per-target branches; 8,094 lines of four runtimes |
| 2026-10-22 | Sessions, LVars as normative syntax | open: proposal until T28 and T29 are proved on the checker, not a model | breaks rule 1 (theorem not linked to the checker) | 10-theorems.md status |

Running totals are computed from this table by `scripts/claim-counts.py`
once a `ledger` section is added to it: proposals, accepted, declined, open,
and the share per reason.
