# Mechanized semantics and type soundness for RCE

## Abstract

This proposal sets out a complete Lean 4 mechanization of RCE: a formal
syntax and static semantics checked against the checker's rule table, a
fuel-indexed dynamic semantics for every construct, and the theorems that
the book states in chapter 10 (T1 to T5 and the replay law T3), with an
executable interpreter linked to the reference implementation by
differential testing. It is organised as parallel streams over a shared
core. The core (stream S1) is in `lean/RCE/Core/` and is proved with no
`sorry`.

In plain terms:

1. A program the checker accepts never gets into a state the language
   does not define: it ends with a value of its type, a named trap, or
   running out of fuel.
2. Running out of fuel is the only outcome that depends on the budget;
   with more fuel a finished result never changes, so two runs that both
   finish agree.
3. The same program, state and log always give the same result, and the
   state after a log is the fold of the step function over it.

## Rules the design follows

- Build what the laws require, not what the text happens to say. Where
  the chapters are ambiguous, the fix is derived from a stated law and
  goes into the book (edit `book/*.md`, then `scripts/regen-spec.sh`).
- The elaborated core is the object of proof. Surface syntax is mapped
  into it by a total elaboration whose rules are generated from the
  checker's rule table, the way `LVarRules.lean` is generated from
  `conc::LVAR_RULES`.
- Evaluation is a definitional interpreter indexed by fuel, returning
  `ok v | err t | oof | stuck`. Type soundness is "never `stuck`".

## Streams

The streams run in parallel; each depends only on the interfaces named.

### S1 Core expression language (started)

Ranged integers with checked arithmetic, booleans, text, products,
sums, `match`, `let`, declared errors and `catch`, top-level calls.

Proved (`RCE/Core/Sound.lean`, `RCE/Core/Fuel.lean`):

- `eval_sound`: well-typed program, expression and environment give a
  result `RTy r t` on every fuel (never `stuck`, values typed).
- `eval_total`: the result is a typed value, a defined trap, or `oof`.
- `eval_det`: evaluation is a function of its inputs.
- `eval_mono`, `eval_mono_le`, `eval_fuel_indep`: more fuel never changes
  a completed result.

Added: `try e else X` and `none` (option as `sum absent present`),
checked integer casts, and the call entry tick. A finding: `eval`'s fuel
bounds nesting depth, while the reference counts ticks across the whole
run, so tick counts are not comparable through `eval`. `RCE/Core/Count.lean`
adds `evalC`, which threads a tick counter (one per node, one on call
entry), with `evalC_sound`, `runC_total` and tick vectors; S6 compares
`ticksUsed` with the reference's `fuel-trace`.

Added in the S1 remainder batch:

- `RCE/Core/BigStep.lean`: `BS`, a relational big-step semantics with no
  fuel. `eval_bigstep` shows that a value or trap from `eval` at any fuel
  is derived by `BS`. `bigstep_eval` shows every `BS` derivation is reached
  by `eval` at some fuel. `eval_iff_bigstep` combines the two. Fuel bounds
  a fuel-free meaning; it never adds one.
- Bit operators `& | ^ << >>` on the two's-complement representation of
  the range (`iand`, `ior`, `ixor`, `wrap`). Shifts trap with `Trap.shift`
  when the amount is negative or at least the bit width (book 6.3 N2).
  `>>` is arithmetic, which is logical on unsigned values.
- Spec finding fixed in the core: `i64.MIN % -1` traps like
  `i64.MIN / -1` (book 6.3 N2; the reference checks the quotient first).
  The core had returned 0.

Suite rerun after this batch (rce-conformance 5f123f8): 122 pass, 1 fail
(the same reference-side `r5-generic-call`), 1867 unsupported.

Next in S1: closures (`fn(p) => e`) as first-order closure values with
captured environments; unary `~` and `-` with exact tick parity; `and`
and `or` with short-circuit evaluation.

### S2 Static semantics from the rule table

The surface types of book 5.1 and the rules of 5.3 to 5.6 as an
inductive judgement, emitted from the checker's rule table by a test in
the style of `checker/tests/lvar_rules.rs`, so a rule added to the checker
without a Lean counterpart fails the build. Elaboration to the S1 core:
structs to nested products, enums to nested sums, `match` to `case`
trees with an exhaustiveness proof. Ownership, borrowing and linearity
reuse `Ownership`, `Borrow`, `Linear`.

Done for the core (`RCE/Core/Shift.lean`, `Elab.lean`, `Rules.lean`):
`weaken` (de Bruijn shifting), exact enum encoding with `matchS` shifting
later arms, `matchS_ty`, `arms_exhaustive` (one arm per variant) and
`enum_val_inv` (every enum value is exactly one variant's injection, so
the match never reaches its fallback). The rule table is
`checker/src/core_rules.rs`; `checker/tests/core_rules.rs` generates
`Rules.lean` (one `#check` per modelled rule) and fails when book 5.3
names a rule the table lacks. Every book 5.3 rule now has a row.

[Var-Move] (`RCE/Core/Own.lean`): `HasTyO` threads a context of types
and liveness flags. Reading a live Copy variable keeps it live, and
reading a live Move variable (text and anything containing it, book 5.5)
marks it moved. Branches must agree on the output context, as
`Linear.lin` requires of consumption counts. Proved: `types_kept`
(ownership changes liveness, never types), `erase` (an ownership-typed
expression has the core typing, so `eval_sound` covers it), and
`moved_unread` (O1: a moved variable has no free occurrence in what
follows). Borrowing (`Borrow`) and loops (`Linear`) are not yet joined
to the core.

### S3 Floats per IEEE 754

f32 and f64 as binary interchange formats with round-to-nearest-even,
NaN canonicalised at every boundary (book N1/N3, F6/N6). Builds on
`F32.lean`. The operations are total functions on bit patterns, so S1's
soundness extends without a new trap.

Started (`RCE/Core/F64.lean`): binary64 on bit patterns by exact
arithmetic, each operation rounded once to nearest, ties to even;
`Ty.f64`/`Val.f64` in the core with `binop_sound` extended. Finding: the
reference does not canonicalise NaN, it traps on any non-finite result
(book 6.3 N3, "NaN cannot occur", interp.rs `cmp_canon`), so the core
follows the book with `Trap.nonFinite`; the sentence above about no new
trap is superseded by N3. `text` is the reference's `float_text`
(shortest round-trip digits, plain in [1e-5, 1e16), else `d.ddde±X`);
`totalCmp` is the float leg of `cmp_canon` (IEEE totalOrder, -0 < +0).
Kernel-checked vectors cover 0.1+0.2, 1/3, -0, 1e+20, the smallest
subnormal, division by zero and overflow. Open: proofs that `round` is
correctly rounded (the vectors are evidence, not proof), f32, float
literals and casts in `core_elab` so the 557 f64 conformance cases run.

### S4 Forms, the log and effects

The step function δ over the Form (book 6, 7), effects as logged
results (`Host`, `Replay`), atomicity T8, and the fold law T3 extended
from `Typing.lean`'s `run_wt` to the full core. Replay: the state is
`foldl δ s₀ D`, and replaying D with the logged effect results gives the
same state and log (T32, T37).

Started (`RCE/Core/Log.lean`, reusing `RCE.Replay`): `delta` runs the
core handler on `((state, event), answer)`; proved `delta_atomic` (T8),
`delta_wt` and `fold_wt` (preservation along the log), `replay_fold`
(T3: replay is the fold of δ over D), `live_is_fold` and `log_det` (the
live run under any effect oracle equals the fold of its own log, so
equal logs give equal states).

Executable event runs (`RCE/Core/Events.lean`, 2026-10-15): `regionStep`
runs the first step of a region whose event tag matches and whose guard
holds, with the event's payload bound above the state fields, through the
statement driver `execS`; `deltaS` runs every region on the pre-state and
the first step that does not apply (trap, declared error, exhaustion)
keeps the state for the whole event (`deltaS_kept_atomic`). The adapter answers event runs with it
(states, payload events, guards, `fail E(..)`, `try x.checked_op(y) else
E`); structs, kinds, lists, options and capabilities in a Form are still
outside the subset.

### S5 Concurrency

`par`, channels, `select`, LVars and sessions over the existing models
(`ParSem`, `ParFuel`, `LVarQD`, `LVarStore`, `LVarAlloc`, `MergeCert`,
`Quorum`, `SessionRec`, `Models06b`). The target theorem is determinism
of the log under every schedule, quasi-determinism where `freeze` is
used, and fuel accounting by `ParFuel.sumcheck_eq_seq`. Reused, not
restated: `LVarAlloc.stepAt_complete` and `stepAt_sound` (the executable
LVar step is exactly the relation), `Canon` (canonical Set and Map
values, which make value equality in the core's `eq` agree with set
semantics once S1 gains Set and Map).

Wired (`RCE/Core/Conc.lean`): a build-checked registry of the reused
lemmas (`ParFuel`, `LVarAlloc.stepAt_sound`/`stepAt_complete`/
`exec_sound`/`alloc_quasi_det`, `Canon`, `SessionRec` T28r and the
encoding), and two proved links: `branch_result_stable` discharges the
hypothesis `ParFuel` states in prose (a branch's result is fixed once it
completes), and `branches_budget_indep` shows the charging order of
`par` cannot change any branch result.

Counter frame (`RCE/Core/Frame.lean`): `evalC_frame` (a run that does not
exhaust is unchanged by extra depth, and extra ticks are returned
untouched) and `seqC_isolated` (branches run in sequence on one counter
give their isolated results and use the sum of their isolated uses).

Formal ParFuel link (`RCE/Core/ParLink.lean`): `par_core_link`. Take core
branches whose isolated tick counts are the uses `ParFuel` models as
numbers. If the reference's sum-checked account `ParFuel.sumTrap` finds
no trap on budget F, then the threaded run on F gives every branch its
isolated result and leaves F less the summed uses. The proof goes
through `sumcheck_eq_seq` and `seqC_isolated`, so the prose hypothesis
of `ParFuel` (a use is fixed by the branch) is now a theorem about the
core.

### S6 Executable link and differential testing

Built. The adapter `rce-conform-lean` (checker `src/bin/conform_lean.rs`)
speaks rce-adapter/1 for `run`. Entry (parse, check, admission) is the
reference front end's, as for the emitters; the run is Lean's. The
checker's `core_elab` elaborates `fn main` and one-parameter arrow
functions over integers, bools and text (`+ - * / % < > ==`, `if`,
`let`, calls, `emit`) into the core as JSON. `CoreMain.lean` reads it
and runs `RCE.Core.runMain` (`RCE/Core/Driver.lean`), which charges one
tick to enter `main`, one per statement and `evalC`'s ticks per
expression, so outcomes under a `fuel` limit are compared exactly.
Anything outside the subset is `unsupported`, never a pass. Compiled
Lean 4.12 binaries do not load on current macOS (dyld rejects the
`__DATA_CONST` segment), so the adapter runs `lake env lean --run
CoreMain.lean` in `RCE_LEAN_DIR`.

Suite run (rce-conformance round3-wip a158ea1, 2026-10-15): 120 pass,
1 fail, 1869 unsupported. The fail, `r5-generic-call`, is rejected at
entry by the reference front end too, so it is a reference or test
finding, not a core one. Largest unsupported groups: f64 (557, S3),
ops other than `run` (633), compound expressions and statements outside
the subset (373), method and path calls (204), event runs (53, S4).

Spec finding from the run: the reference's overflow trap names the value
and the type (`overflow: V is outside T`). The core's `Trap.overflow` now
carries the value and range, so the trap is determined by the program,
not only its kind.

Later rounds (2026-10-15): statements (`var`, assignment, `op=`,
`if`/`elif`/`else`, `while` with its bound, `break`, `continue`) in
`Driver.lean`; `for x in a..b par` with one `+=`/`*=` reduction target over
the fixed tree of C.8 (`Stmt.forpar`, `treeR`, split at the largest power
of two below the length, as host.rs `tree_split`); the `check` op. For
`check`, `RCE/Core/Check.lean` gives a decidable checker `inf` (type
synthesis with a hint) proved sound against `HasTy` (`inf_sound`) and
`checkProg_sound` for `WtProg`. The adapter answers `accept` only when
Lean shows the elaborated program well typed. The statement walk over
the proved expression checks is not itself a judgement yet. The core
has no rejection rules, so every `reject` case is `unsupported`. Suite
(rce-conformance with the harness honouring `unsupported` on `check`):
1208 pass, 17 fail, 766 unsupported. Sixteen of the fails are harvested
`57-float-vectors` cases that the reference front end rejects too: the
example on line 13 expects `-0.0` since d7aeecd65, and the harvested
copies are stale. The other is `r5-generic-call`, as before.

### S7 Refinement

A refinement from the Lean semantics to the reference interpreter,
staged: (a) differential agreement on the suite; (b) a simulation proof
for the ANF lowering (`Anf.lean`, T34); (c) per-target agreement T9 as forward
simulation for the emitters, reusing
`ParRefine` and `Refine`.

## Spec findings

- `catch` handles only declared errors; traps propagate (book 7,
  E-Catch). The core follows this.
- Book 7 rule **E1**: fuel exhaustion is uncatchable. Derived from fuel
  monotonicity; if `catch` could observe exhaustion, `eval_mono` would
  fail. Testable id E1 for the conformance suite.

## Reading ledger

| Work | Read | Sections | What it bears on here |
|---|---|---|---|
| Owens, Myreen, Kumar, Tan, Functional big-step semantics, ESOP 2016, doi 10.1007/978-3-662-49498-1_23 | 2026-10-15, in full (pp. 589-615) | 1-9 | Our `eval` is a clocked interpreter of this kind. Their type soundness for Core ML is stated for all clocks (timeout, exception or a value of the right type, never Rfail), which is the form of `eval_sound`. Their clock lemma (a non-timeout result is unchanged by adding clock, s3.4) is our `eval_mono_le`. Differences: they decrement the clock only on calls whose argument does not shrink, we decrement at every node; RCE's reference charges one tick per evaluated expression, per statement and per call (book 12, Fuel; 6.7), so a per-node charge is the one to match. The core does not yet charge a separate tick for the call itself; S6 compares outcomes, and exact tick parity is an S1 item. They note concurrency is outside the approach (s4), which is why S5 uses the existing small-step models, not `eval`. Their semantics for divergence (timeout for every clock) is not needed: RCE runs on a fixed budget. |
| Jung, Jourdan, Krebbers, Dreyer, RustBelt, POPL 2018, doi 10.1145/3158154 | 2026-10-15, in full (34 pp.) | 1-9 | Progress and preservation is a closed-world method (s1.1); RustBelt instead proves a fundamental theorem for each typing rule plus adequacy, so unsafe libraries can be added with a verification condition (s1.2, s7). RCE has no unsafe code, so the closed-world, syntactic proof in S1/S2 suffices; the semantic route is recorded for S2's ownership rules only if an extension mechanism (host adapters with typed contracts, book 16) needs to be proved safe by its contract. Their thread-relative ownership (Send/Sync as independence of the thread id, s4.1) is a candidate model for `par` footprints in S5; not adopted yet. |

## References not yet read

Outside work is cited only after it has been read in full. The works
below have not been read for this proposal, so the text makes no claim
about them. They are listed for David to obtain; a ledger entry
(read date, sections, what was taken) is added here before any note
returns.

| Work | DOI / source | Access |
|---|---|---|
| Watt, Mechanising and verifying the WebAssembly specification, CPP 2018 | 10.1145/3167082 | ACM; author copy may exist |
| Kumar et al., CakeML: a verified implementation of ML, POPL 2014 | 10.1145/2535838.2535841 | ACM |
| Leroy, Formal verification of a realistic compiler, CACM 2009 | 10.1145/1538788.1538814 | ACM |
| Milner, Tofte, Harper, MacQueen, The Definition of Standard ML (Revised), MIT Press 1997 | ISBN 978-0-262-63181-5 | book, paywalled |
| Rosu, Serbanuta, An overview of the K semantic framework, JLAP 2010 | 10.1016/j.jlap.2010.03.012 | Elsevier, paywalled |
