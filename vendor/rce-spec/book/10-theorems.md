## 8. Theorems to prove

This section states the properties the language guarantees, such as
determinism and termination, and records which are proved in Lean 4 (a
proof assistant that checks proofs by machine). A *theorem* here is a
statement about every well-formed program. Example: T8 (atomicity) says a
refused event leaves the state exactly as it was.

Stated for well-formed programs (section 7). Proofs are mechanized in Lean 4
(v4.12.0, no Mathlib) in `spec/lean/`, over a core calculus: expressions
(`RCE/Expr.lean`), statements, steps, regions, δ and the runtime fold
(`RCE/Form.lean`), typing of steps and states (`RCE/Typing.lean`), an
ownership core (`RCE/Ownership.lean`), the concurrency rules of 6.8
(`RCE/Conc.lean`), full borrow paths (`RCE/Borrow.lean`), the step bound
(`RCE/Bound.lean`), name-free hashing and ids (`RCE/NameFree.lean`) and the
Rust emitter (`RCE/Emit.lean`, printer `RCE/EmitPrint.lean`, differential
harness `emit-test/`), structural recursion (`RCE/Struct.lean`, draft
0.4), and the draft 0.5 additions: select fairness (`RCE/Select.lean`),
linear handles (`RCE/Linear.lean`), f32 (`RCE/F32.lean`), u64 arithmetic
(`RCE/U64.lean`) and iterators (`RCE/Iter.lean`), and foreign replay
(`RCE/Replay.lean`), `&mut` write-back (`RCE/WriteBack.lean`) and bound
composition (`RCE/Compose.lean`). Build with `lake build`; no `sorry`,
no added axioms (`#print axioms` lists only propext, Quot.sound,
Classical.choice). Status per theorem:

| | Status | Lean |
|---|---|---|
| T1 | proved for expressions, δ, and the 6.8 tick semantics (tasks, channels, select, actors): one tick from a configuration has exactly one result | `step_det`, `exec_unique`, `tstep_det`, `run_det`, `tick_det` |
| T2 | proved, including the explicit bound: a statement reaches its `exec` result in at most B(s) steps (B from loop bounds and expression sizes), a region in at most BR, δ in at most |s|·BR(P). The Lean expression core has no user function calls. Structural recursion (0.4) is proved separately on a model with total leaf and combine bodies: fuel equal to the value's size always suffices, the result equals the function's denotation, extra fuel changes nothing, and the number of calls equals the size | `exec_unique`, `eval_bound`, `stmt_bound`, `region_bound`, `delta_bound`, `struct_total`, `calls_eq_size`, `struct_fuel_irrelevant`; composition along the call graph (a callee's bound adds to the caller's, a chain is bounded by the sum): `comp_bounded`, `comp_result`, `chain_bounded` |
| T3 | proved | `fold_law`, `log_is_events`, `replay` |
| T4 | proved for expressions, statements, δ and ticks (some task moves, or every task is done or blocked: the run ends or is `Deadlock`) | `progress`, `stmt_progress`, `delta_bound`, `run_progress`, `progress` (Conc) |
| T5 | proved for expressions, statements, δ and the whole log; for ticks, the world invariant (typed channel contents, task-owned locations) is preserved | `preservation`, `exec_wt`, `delta_wt`, `run_wt`, `tick_inv` |
| T6 | proved for the ownership core; for full borrow paths (field paths, prefix overlap, partial moves, loans) in an instrumented semantics: well-typed bodies never fault, writes through a `&mut` path are exclusive, and write-while-borrowed, overlapping `&mut`, dangling borrows and read-after-partial-move are refused; tasks never share mutable data in any reachable configuration | `memory_safety`, `borrow_safety`, `borrow_no_fault`, `writeVia_exclusive`, `overlapping_mut_refused`, `dangling_refused`, `tasks_never_share` |
| T7 | proved for event/error names over δ and the log; for log hashes under any hash function H; for channel, actor and task ids and select choices of the concurrency model | `name_independence`, `log_hashes_name_independent`, `conc_name_independent`, `select_name_independent` |
| T8 | proved | `atomicity` |
| T9 | T9-F (Form IR targets: Rust, WASM, Python, XState, SCXML) tested: every target prints the reference trace byte for byte on the edge and seeded events (`tests/form_targets.rs`, `tests/form_joins.rs`). T9-L (program emitters of 10.5: `rust`, `python`, `js`, including `&mut` write-back on failure under `catch` and `try` and nested in strict positions via the ANF pass of T34, 6.2, `spec/checker/tests/write_back.rs`): proved for the Rust emitter on the core (step bodies, expressions, guarded step selection) as a verified translation into MiniRust, a deep embedding of the emitted Rust subset with Rust-style semantics; printed to Rust and checked against rustc by differential testing. Not covered: the other targets, the full surface language, and the claim that rustc implements MiniRust (trusted, tested) | `reval_compile`, `compile_correct`, `RExec.det`, `emit_agrees`, `region_agrees` |
| T10 | checked mechanically (LALR(1), 0 conflicts, from the EBNF of section 3, and of section 3 with C.2 for version 0.6) and sampled by `rce-fuzz`; the former `fuzz/gen.py` also found exactly one Earley parse per program | `rce-spec-check` |
| T11 | proved on the rotation model of 6.8: the arm taken is ready, and an arm ready at n consecutive executions is taken by one of them | `pick_ready`, `select_bounded_wait` |
| T12 | proved on a statement model with branches and bounded loops: if the W-LIN-1 count accepts a body, every execution path consumes the handle exactly once | `lin_sound`, `linear_exactly_once` |
| T13 | proved for any rounding that fixes representable values: widening f32 to f64 is exact, f32 multiply via f64 is a single rounding, and every f32 op gives the same bits on targets that agree on IEEE 754 binary64 ops and binary32 rounding. For + − / the single-rounding property is Figueroa's theorem, cited, not mechanized | `widen_exact`, `f32_mul_det`, `f32_op_det` |
| T14 | proved: checked u64 ops return the exact result exactly when it lies in [0, 2^64), and udiv/urem satisfy b·q + r = a, r < b | `uadd_spec`, `usub_spec`, `umul_spec`, `udiv_urem`, `udiv_range`, `urem_range` |
| T32 | proved for every oracle: replay of the log equals the live run, the log's events are the input events, an answer contract holds of every logged answer, and runs that logged the same answers agree | `replay_live`, `live_events`, `live_answers_satisfy`, `t32` |
| T33 | proved: copy-out on every outcome equals in-place update; copy-out only on success does not | `t33`, `copyOk_differs` |
| T34 | proved for a core fragment (literals, reads, write-back calls, a charging operator, emit) with full ANF: the let-bound program gives the same value, store, and trace of charges and emits; the implementation (`anf.rs`, which binds fewer nodes and covers the surface language) is tested differentially on every target, including a budget sweep across the trap point | `t34`, `run_frame` |
| T15 | proved for iterators whose step between yields is total (T2): every run ends, delivers at most LOOP_LIMIT values and traps only after exactly LOOP_LIMIT; `break` yields a prefix of the full run; a run with a smaller cap is a prefix of one with a larger cap (the evaluator's re-run strategy) | `drive_total`, `drive_break`, `gen_prefix` |

T16 to T31 belong to version 0.6 and are listed with their exact status
in Appendix C.13. Most are proofs about abstract models (T18, T19b, T21,
T22, T25 to T27, T29 to T31); T16, T17, T19a, T20 and T23 are proved on a
small-step semantics of a core with `par`; none is a proof about the Rust
reference evaluator.

**T1 Determinism.** For every program P, state s and event e there is exactly
one (s′, o) with δ_P(s, e) = (s′, o). For a command, the full configuration
sequence of 6.8 is a function of the input events.
∀ s e r₁ r₂. ⟨step P, s, e⟩ ⇓ r₁ ∧ ⟨step P, s, e⟩ ⇓ r₂ ⇒ r₁ = r₂.

**T2 Totality of δ.** For every s ∈ S and e ∈ E, evaluation of δ(s, e)
terminates, and the number of evaluation steps is at most
B(P) · (1 + size(s) + size(e)) for a bound B(P) computable from P
(from loop bounds, SIZE_LIMIT and the call graph, which is acyclic except
for structural self-calls; each of those contributes a factor bounded by
the size of the value it recurses on, itself bounded by N9).

**T3 Fold law.** For every log D = e₁…eₙ, the state the runtime reaches is
foldl(δ₁, s₀, D), and replaying D from s₀ reproduces every state hash and
every output: ∀ D. run(P, D) = foldl(δ, s₀, D) componentwise.

**T4 Progress.** If ∅; Γ ⊢ e : τ and the store σ agrees with Γ, then
⟨e, ρ, σ⟩ steps, or e is a value, or e is a `trap`, `fail`, `ret`, `brk` or
`cont` signal. (No stuck states other than defined outcomes.) For
configurations of 6.8: either some task can move, or the run ends with a
defined outcome (all tasks done, or `Deadlock`).

**T5 Preservation.** If Σ; Γ; Δ ⊢ e : τ ⊣ Γ′ and e → e′ (small-step
presentation of 6.2), then there is Δ′ with Σ; Γ; Δ′ ⊢ e′ : τ ⊣ Γ′; and every
value written to a state field of type τ is in ⟦τ⟧ (including its integer
range).

**T6 Memory safety.** A well-typed program never reads a value after it has
been dropped or moved, never observes a write through one path while another
path to the same location is live (shared XOR mutable, 5.6), never reads a
region after it is freed (5.7), and never shares mutable data between tasks
(5.8). Formally: in the instrumented semantics that tags each location with
an owner and a validity bit, no well-typed program reaches a configuration
where an access touches an invalid location or two tasks hold a mutable
path to one location. Foreign functions (5.11) are outside this
statement: T6 holds for the RCE side under the 5.11 contract on the callee.
(Approach: RustBelt-style semantic typing is not needed
because borrows are second class; a syntactic progress/preservation proof over
the instrumented semantics suffices.)

**T7 Name independence.** If P′ is P with identifiers consistently renamed,
then every log hash, every id and every select choice of P′ equals that of P.

**T8 Atomicity.** If δ(s, e) refuses e, then δ₁(s, e) = s.

**T9 Implementation agreement (per target).** For every target t,
⟦compile_t(P)⟧ = δ_P on every s, e, including every edge of 6.3. There are
two target sets, and a claim names its set. T9-F: the Form IR targets of
the compiler (`rce form --emit`): Rust (native), WASM (the Rust crate on
wasm32-wasip1), Python, XState and SCXML. T9-L: the program emitters of
10.5 (`rce emit rust | python | js`), which translate the checked module.
A target outside these sets has no T9 claim. This is the theorem the
differential tests sample; it is stated here so it is clear those tests
check a property of implementations, not of the language. A differential
test compares against the reference evaluator's trace, which the target
under test does not produce.

**T10 Grammar unambiguity.** The grammar of section 3 is LALR(1) (checked
mechanically, 3.12), hence unambiguous: every accepted text has exactly one
parse tree. The same holds for the version 0.6 grammar (section 3 with
Appendix C.2 applied), checked by the same machine.

**T11 Select fairness (bounded wait).** An arm of a `select` that is ready
at each of n consecutive executions of that `select` (n arms) is taken by
one of them.

**T12 Linear use.** In a well-formed program every value of a linear type
is consumed exactly once on every execution path.

**T13 f32 determinism.** Every f32 operation gives the same binary32 result
on every conforming target.

**T14 u64 arithmetic.** A u64 operation returns the exact mathematical
result when it lies in [0, 2^64) and traps otherwise.

**T15 Iterator termination.** A `for` loop over an iterator, and
`.collect()`, terminate after at most LOOP_LIMIT yields, and `break`
observes a prefix of the values the iterator would produce.

**T32 Foreign replay.** For every oracle o answering foreign calls and
`select nondet` choices (5.11, C.5), replaying the log of a live run
reproduces its state, whatever o is; no property of the foreign code is
assumed, and a contract every answer satisfies holds of every logged
answer.

**T33 `&mut` write-back.** A call that writes back its `&mut` arguments on
every outcome (6.2) gives the same store and outcome as updating them in
place.

**T34 Order-preserving A-normal form.** Flattening an expression into
`let`-bound steps in left-to-right order, with each temporary fresh,
gives the same value, final store, and ordered trace of budget charges
and emits as evaluating the expression.

**Numbering and assurance.** Theorem numbers are stable identifiers, not
positions: a number is never reused or changed, new theorems take the
next number (T32 on), and a split keeps its number with a letter (T19a,
T19b). Code, tests, Lean doc comments and papers cite these numbers, so
renumbering would break every citation for no gain in meaning. Each
row's status uses one of the levels proved (Lean, on the RCE semantics
or the named core), model (Lean, on an abstract model), checked (a
machine check that is exhaustive for its input, such as LALR(1)),
tested (sampled differential or property tests), in that order. A
counterexample is a failure, never a lower level, and a claim about
several parts has the lowest level among them. Every Lean name in the
tables is checked to exist (`spec/checker/tests/theorem_index.rs`).

**Read receipt.** What is the deep embedding of the emitted Rust subset called?

<!-- body-sha256: 182f9afb88e9a12e15e7e7ccc0a31cc1e895905a3e60aeb72b668b2fafea9921 -->
