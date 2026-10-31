# Closing the RCE loop deterministically

Scope: turn a stated intent into a checked, emitted Form with no model in the loop.
Sources audited: Mac worktree `_wt-rce-host-runtime/crates/rce` (actual branch `wt/rce-host-runtime`, not `rce/host`), stdlib `openIE-dev/rce-stdlib` (59 `.rce` files under `lib/`).

## 1. What exists

| Piece | Where | Role in a loop |
|---|---|---|
| Handler spec parser and compiler | `src/spec/parser.rs`, `src/spec/compiler.rs` | a spec is the closest thing to intent today |
| Form compiler: lower, egg e-graph, cost extraction, emit Rust/XState/SCXML/Python | `src/form/mod.rs:1-20`, `egraph.rs:104,305,361`, `extract.rs:89-96` | optimizes a given fit; does not search for one |
| Equivalence proposals behind a randomized check | `src/form/equiv.rs:251,316`, `propose.rs` (System One enumeration; optional LLM proposer, off by default, `src/bin/rce.rs:3231`) | grows the rewrite set |
| Gear stack G0 library, G1 rules, G2 enumerative synthesis, G3-G5 models | `src/foreman/gears.rs:1-30,75,173,268`, `synth.rs:1-22,149` | fills one escalating node from examples plus output schema, sizes 1-2 only |
| Recorded fits as oracles | spec D.4 `spec/RCE-SPEC.md:3272-3293`, `src/form/fit.rs` | replay table; miss escalates |
| `rce auto` (default `--foreman geared`) | `src/bin/rce.rs:13,381-470,2105` | runtime loop: escalation, gear, gate, promote |
| `rce scaffold`, `gen-tests`, `remediate` | `src/bin/rce.rs:377-380,1013,1035,2041`; `src/scaffold.rs` | stubs and proptest harnesses from a spec |
| Repair: G0 token edits, G1 indexed Java edits, TBar templates | `src/repair/g0.rs`, `g1.rs`, `tbar.rs`, `mod.rs:737` | test-gated patching of foreign code (Defects4J), not of Forms |
| Language checker: `check run test trace emit fmt lsp` | `spec/checker/src/cli.rs:606-730`, emitters `spec/checker/src/emit/` | the verifier for a candidate module |

No `rce new`, `rce gen`, `rce fit` command exists (rg found none). Model use sits only in G3-G5, `ask/oracle.rs` and the opt-in proposer; the gate decides.

## 2. Spec ahead of engine

- D.7 (`RCE-SPEC.md:3355-3361`): how an RCE module is used as a fit is undefined. The two inputs (handler spec, language module) do not meet, so stdlib bricks cannot be fit terms.
- Section 9 open questions (`:2263`): stdlib surface undecided.
- 10.8 (`:2687`): emitters skip concurrency and `extern` in observations; no compiled runtime.
- `synth.rs:21` says egg is not vendored, but `Cargo.toml:66` has egg 0.11 and `form/egraph.rs` uses it. G2 does not use the e-graph.

## 3. Missing to close the loop

1. Intent language. Nothing between prose and a full spec.
2. Brick catalog. stdlib has no manifest of signatures, contracts, effects, costs, or content hashes.
3. Composer. G2 enumerates at most size 2 over HandlerSpec combinators; no type-directed or contract-directed search over bricks, no holes.
4. Repair loop on checker diagnostics. Repair is test-gated mutation of Java/Rust/Python; checker diagnostics are not machine-addressable repair inputs.
5. Test oracle. Recorded tables and `example` lines exist, but no intent-derived oracle (examples plus properties) that the composer must satisfy.

## 4. Principles

P1 Determinism: same intent, catalog hash and engine version give byte-identical output. Every choice is ordered by a total key; ties break by content hash.
P2 Trust only the checker: generators propose, the checker and the oracle decide. A generator bug can cost completeness, never soundness.
P3 Totality of the search: every search has a declared finite bound; exhaustion yields a structured "unrealizable within bound" result, never a guess.
P4 Content addressing: intents, bricks, candidates, diagnostics and verdicts are hashed and logged in the same hash chain as Forms, so a synthesis run is replayable like a Form run.
P5 Smallest sufficient work: cheapest stage first (lookup, then composition, then repair), consistent with the gear stack.

### 4.1 Model components stay outside the trusted path
The engine already contains model-backed components: the gear stack's G3 (System One decision model), G4 (local model) and G5 (remote federated cascade) in `src/foreman/gears.rs`, and the LLM equivalence proposer in `src/form/propose.rs` (`RCE_LLM_PROPOSER_CMD`). All are off by default: G3 and G4 need an endpoint, G5 needs `--remote`, the proposer needs its command. In this design they stay that way and outside the trusted path: no stage of `rce index`, `rce fit` or `rce gen` calls them, a catalog or synthesis key never depends on their output, and anything they propose is only a candidate that must pass the same checker, oracle and gate as an enumerated one. Turning them off can cost completeness, never soundness (P2).

## 5. Design

### 5.1 Intent spec (`.intent`, subset of RCE syntax)
A Form with holes. Declares state type, events, outputs, and obligations:
- `ensures` predicates over (state, event, state', outputs) in the refinement fragment already decided by T35 (`holds_total`);
- `example` lines (reuse 10.3 syntax) as concrete traces;
- `invariant` over state;
- `uses` bounds: allowed bricks/effects; `budget`: max term size, max est_j.
Holes are `?name : τ {p}`: a typed, refined unknown. A program with zero holes is an ordinary module.

### 5.2 Brick catalog (`rce index`)
Built from stdlib by the checker, not by hand. One record per exported function/Form:
`{hash, path, name, type, refinement pre/post, effects, totality, est_j, laws[], vectors[]}`.
Keys: normalized type (alpha-renamed, argument-order canonical) for Hoogle-style lookup by type up to isomorphism; laws feed the e-graph. The catalog is itself hashed; its hash is part of every synthesis key. This also settles D.7: a brick is a module function lowered to a Form IR term, so a fit `term` may name a brick by hash.

### 5.3 Composer (`rce fit INTENT`)
Per hole, in this order, each stage bounded:
1. Lookup: catalog entries whose type unifies with the hole type and whose post implies the hole refinement (checked by the refinement checker).
2. Type-directed enumeration: bottom-up, size-ordered terms over catalog bricks plus Form IR combinators, pruned by types (only well-typed terms built) and by observational equivalence on the oracle inputs (one representative per behaviour class, as G2 does).
3. E-graph composition: insert all candidates for a hole into one e-graph with catalog laws; extract by (oracle pass, size, est_j, hash). Reuses `form/egraph.rs` and `extract.rs`.
4. Constraint step for constants and guards: finite-domain enumeration over literals from the intent and catalog; a solver is optional, and when present its answer is re-checked.
Holes are filled most-constrained first (fewest candidates), with backtracking on conflict; order fixed by (candidate count, hole name).

### 5.4 Check, oracle, repair
Candidate goes through `checker check`, then `test` on intent examples, then a seeded property run of `ensures`/`invariant` (the gear gate's property step, generalized), then emitter agreement via `rce trace` across targets.
Diagnostics become structured: `{code, span, hole?, expected, found}`. A fixed table maps diagnostic codes to repair actions (re-open the hole, narrow its type, add a guard brick, swap a brick of the same type). Repair re-enters 5.3 with the extra constraint, so repair is search with learned exclusions, not mutation. A loop bound and a seen-set of candidate hashes guarantee termination.

### 5.5 Oracle
Three layers, all deterministic: (a) intent examples, (b) properties sampled with a seed derived from the intent hash, (c) differential agreement between reference interpreter and emitted targets. Recorded handlers (D.4) remain the oracle for external effects; a miss is a failure, not a call out.

### 5.6 Commands
- `rce index LIB -> catalog.json`
- `rce new TEMPLATE NAME`: instantiate a Form template (an intent with holes) from the catalog
- `rce fit INTENT [--budget]`: compose, check, repair; writes module, proof log, and a synthesis log
- `rce gen target T MODULE`: existing emit path

Outputs: the module, a per-hole derivation (stage, candidates tried, winner hash), and the hash chain over the run.

## 6. Convergence notes
- Synquid (Polikarpova et al., PLDI 2016): refinement-typed holes with round-trip type checking; 5.1 holes and 5.3 step 1 converge on it, restricted to RCE's decidable predicate fragment.
- Type-directed synthesis (Osera and Zdancewic, PLDI 2015; Myth): example-guided, type-pruned enumeration; matches 5.3 step 2.
- egg (Willsey et al., POPL 2021): equality saturation plus cost extraction; already in the engine, extended from optimization to selection.
- Sketch / program synthesis by sketching (Solar-Lezama, PhD 2008; ASPLOS 2006): holes in a fixed skeleton filled by CEGIS; the intent is a sketch, 5.4 repair is the counterexample step.
- miniKanren (Byrd et al.): relational, complete interleaving search; informs the fair, bounded hole-filling order.
- Hoogle (Mitchell): search by type up to isomorphism; the catalog key in 5.2.
- Enumerative SyGuS / FlashFill and TBar: already reflected in `synth.rs` and `repair/`.

## 7. Increments
1. Catalog from stdlib, with D.7 defined as "brick = module function lowered to IR term". Done on joulesperbit `rce/host`: `rce index` and `rce search` (`spec/checker/src/brick.rs`, spec 10.10 and D.7, 23f751795); library dependencies (`# uses:`, `#!lib`, inferred `#!lib`; fcfdbd50b); lowering in `spec/checker/src/brick_lower.rs` widened in steps (bit operations 662c8e33b; statements, loops, lists, structs 92a38079a; text as UTF-8 bytes 13c3ebca1; generics and usize dbcc25eed); generated vectors and `rce-brick-verify` (1a5b269e9, e00d1cb8b).

   Current results on rce-stdlib 4ed1b7f: 457 bricks, all in admitted files; 269 lowered to Form IR terms; 252 of those agree with the reference interpreter on generated vectors, 0 disagree, 17 have no vector yet (generic structs such as `Deque[T]`, two loop-heavy pq functions).

   Not yet lowered (188): 77 term size or depth (calls are inlined because the IR has no named sub-terms), 50 mutable borrows, 38 error results (`T ! E`, `catch`, `try`), 7 `match`, 7 other calls, 5 slices and place assignments, 4 narrow integers or floats. The largest class needs named sub-terms in the Form IR (a definition table and a call node), which the evaluator and every emitter must carry; mutable borrows need a stated convention (the term returns the result and the updated arguments).
2. Holes in the checker (parse, type, report unfilled).
3. Lift G2 from HandlerSpec to catalog terms and wire it to the e-graph.
4. Structured diagnostics and the repair table.
5. `rce fit` end to end with logged derivations; then `rce new` templates.
