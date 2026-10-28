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
0.4), mutual recursion (`RCE/Mutual.lean`), and the draft 0.5 additions: select fairness (`RCE/Select.lean`),
linear handles (`RCE/Linear.lean`), f32 (`RCE/F32.lean`), u64 arithmetic
(`RCE/U64.lean`) and iterators (`RCE/Iter.lean`), and foreign replay
(`RCE/Replay.lean`), `&mut` write-back (`RCE/WriteBack.lean`), bound
composition (`RCE/Compose.lean`) and the host contract of Appendix E
(`RCE/Host.lean`). Build with `lake build`; no `sorry`,
no added axioms (`#print axioms` lists only propext, Quot.sound,
Classical.choice). Status per theorem:

| | Status | Lean |
|---|---|---|
| T1 | proved for expressions, δ, and the 6.8 tick semantics (tasks, channels, select, actors): one tick from a configuration has exactly one result | `step_det`, `exec_unique`, `tstep_det`, `run_det`, `tick_det` |
| T2 | proved, including the explicit bound: a statement reaches its `exec` result in at most B(s) steps (B from loop bounds and expression sizes), a region in at most BR, δ in at most |s|·BR(P). The Lean expression core has no user function calls. Structural recursion (0.4) is proved separately on a model with total leaf and combine bodies: fuel equal to the value's size always suffices, the result equals the function's denotation, extra fuel changes nothing, and the number of calls equals the size. Mutual recursion (6.7) is proved on a model of a family of functions whose calls lower a measure: the checker's call condition lowers the summed measure (`call_dec`), with the rank of the function it lowers the combined measure (`lex_dec`), fuel above the measure always suffices and extra fuel changes nothing, and a chain of calls has at most the measure plus one calls; with several layers (6.7) a call that keeps every earlier layer and shrinks one, or keeps all and lowers the rank, lowers the tuple in a well-founded lexicographic order (`call_keep`, `layer_dec`), every evaluation then ends for some fuel, and more fuel never changes the value; a declared bound (6.7) is modelled by counts of active calls carried in the state, whose slack falls by one at every admitted call of a bounded function, so the same measure argument applies | `exec_unique`, `eval_bound`, `stmt_bound`, `region_bound`, `delta_bound`, `struct_total`, `calls_eq_size`, `struct_fuel_irrelevant`, `call_dec`, `lex_dec`, `mutual_total`, `mutual_fuel_irrelevant`, `mutual_depth`, `call_keep`, `layer_dec`, `wf_total`, `wf_fuel_irrelevant`, `bump_slack`, `bound_dec`; composition along the call graph (a callee's bound adds to the caller's, a chain is bounded by the sum): `comp_bounded`, `comp_result`, `chain_bounded` |
| T3 | proved | `fold_law`, `log_is_events`, `replay` |
| T4 | proved for expressions, statements, δ and ticks (some task moves, or every task is done or blocked: the run ends or is `Deadlock`) | `progress`, `stmt_progress`, `delta_bound`, `run_progress`, `progress` (Conc) |
| T5 | proved for expressions, statements, δ and the whole log; for ticks, the world invariant (typed channel contents, task-owned locations) is preserved | `preservation`, `exec_wt`, `delta_wt`, `run_wt`, `tick_inv` |
| T6 | proved for the ownership core; for full borrow paths (field paths, prefix overlap, partial moves, loans) in an instrumented semantics: well-typed bodies never fault, writes through a `&mut` path are exclusive, and write-while-borrowed, overlapping `&mut`, dangling borrows and read-after-partial-move are refused; tasks never share mutable data in any reachable configuration | `memory_safety`, `borrow_safety`, `borrow_no_fault`, `writeVia_exclusive`, `overlapping_mut_no_typing`, `dangling_no_typing`, `tasks_never_share` |
| T7 | proved for event/error names over δ and the log; for log hashes under any hash function H; for channel, actor and task ids and select choices of the concurrency model. The checker and the four emitted runtimes hash the positional forms of 6.1 since the erratum of C.15, so the implementation's log hashes are name-free as the model's are (`tests/nameless.rs`); `RCE/Positional.lean` models that embedding (records of declared types as field lists in declaration order, variants as `[k, fields]`, events by index, undeclared records by name) and proves it unchanged under any renaming that is injective and fixes undeclared names, hence the state hash and the whole entry chain for every hash function, and that its event index is the `idx` the De Bruijn model hashes | `name_independence`, `log_hashes_name_independent`, `conc_name_independent`, `select_name_independent`, `Positional.pos_rename`, `Positional.chain_name_independent`, `Positional.eventIndex_eq_idx` |
| T8 | proved | `atomicity` |
| T9 | T9-F (Form IR targets: Rust, WASM, Python, XState, SCXML) stated, not checked here: the Form IR compiler (`rce form --emit`) is in the engine crate, which is not in this repository (10.1, Appendix D), so no tool published here samples T9-F. T9-L (program emitters of 10.5: `rust`, `python`, `js`, including `&mut` write-back on failure under `catch` and `try` and nested in strict positions via the ANF pass of T34, 6.2, `spec/checker/tests/write_back.rs`): proved for the Rust emitter on the core (step bodies, expressions, guarded step selection) as a verified translation into MiniRust, a deep embedding of the emitted Rust subset with Rust-style semantics; printed to Rust and checked against rustc by differential testing. Not covered: the other targets, the full surface language, and the claim that rustc implements MiniRust (trusted, tested) | `reval_compile`, `compile_correct`, `RExec.det`, `emit_agrees`, `region_agrees` |
| T10 | checked mechanically (LALR(1), 0 conflicts, from the EBNF of section 3, and of section 3 with C.2) and sampled by `rce-fuzz`; the former `fuzz/gen.py` also found exactly one Earley parse per program | `rce-spec-check` |
| T11 | proved on the rotation model of 6.8: the arm taken is ready, and an arm ready at n consecutive executions is taken by one of them | `pick_ready`, `select_bounded_wait` |
| T12 | proved on a statement model with branches and bounded loops: if the W-LIN-1 count accepts a body, every execution path consumes the handle exactly once | `lin_sound`, `linear_exactly_once` |
| T13 | proved for any rounding that fixes representable values: widening f32 to f64 is exact, f32 multiply via f64 is a single rounding, and every f32 op gives the same bits on targets that agree on IEEE 754 binary64 ops and binary32 rounding. For + − / the single-rounding property is Figueroa's theorem, cited, not mechanized | `widen_exact`, `f32_mul_det`, `f32_op_det` |
| T14 | proved: checked u64 ops return the exact result exactly when it lies in [0, 2^64), and udiv/urem satisfy b·q + r = a, r < b | `uadd_spec`, `usub_spec`, `umul_spec`, `udiv_urem`, `udiv_range`, `urem_range` |
| T32 | proved for every oracle: replay of the log equals the live run, the log's events are the input events, an answer contract holds of every logged answer, and runs that logged the same answers agree | `replay_live`, `live_events`, `live_answers_satisfy`, `t32` |
| T33 | proved: a call that ends by `fail` leaves the caller's `&mut` state as at the call, and a returning call writes its final value back (the restore rule of F.4.4, as a model; W-TX-1 is a check, not part of the model) | `t33_failure_restores`, `t33_return_writes_back` |
| T36 | proved on models of 5.16: every metacharacter of rendered context text comes from a literal when the encoder emits none (and the HTML encoder emits none of `< > " '`); public outputs are equal for runs whose public inputs and secret-equality bits agree; a log of events without the capability leaves a guarded field unchanged. The checker rules are tested on accept/refuse cases and the encoders across the reference and the Rust, Python and JavaScript runtimes (`tests/indexed.rs`) | `ctx_meta_from_literals`, `html_enc_clean`, `noninterference`, `outputs_noninterfere`, `guarded_needs_cap` |
| T35 | proved on a model of the boundary check: the predicate fragment decides (a total Bool function), a value that passes the check satisfies its predicate, admission accepts only payloads that satisfy their field predicates, and every state reached by the checked δ from an initial state that satisfies the field predicates satisfies them, with a refused event leaving the state unchanged; the reference evaluator and the Rust, Python and JavaScript runtimes are tested to accept and trap alike (`tests/refined.rs`) | `holds_total`, `fit_sound`, `admit_sound`, `refine_preserved`, `refine_kept_unchanged` |
| T34 | proved for a core fragment (literals, reads, write-back calls, a charging operator, emit) with full ANF: the let-bound program gives the same value, store, and trace of charges and emits; the implementation (`anf.rs`, which binds fewer nodes and covers the surface language) is tested differentially on every target, including a budget sweep across the trap point | `t34`, `run_frame` |
| T37 | proved on a model of the host (Appendix E): δ extended with host entries is the identity on them, replay of the log with no oracle equals the live run for every adapter oracle, a log of host entries alone leaves the state unchanged, and placement computed from the logged host entries agrees for equal logs; the example logs are re-folded with host entries interleaved in `tests/host_contract.rs` | `host_entry_identity`, `t37_replay`, `t37_host_only`, `t37_placement` |
| T38 | proved for any hash function: the verified prefix is a prefix and a valid chain, a valid log is kept whole, and a valid log followed by an entry that does not link is cut back to it; the SHA-256 chain of every example log is tested with torn, flipped and swapped entries (`tests/host_contract.rs`) | `verify_prefix`, `verify_valid`, `verify_intact`, `verify_append`, `t38_torn` |
| T39 | proved on an admitter model: after a failed barrier nothing is appended, and a running admitter's log is exactly the entries it was given | `t39_halted`, `t39_fault_stops`, `t39_run_complete` |
| T40 | proved on a window model (Appendix E.2a): with an injective input key, the entries of a window are equal for any permutation of its inputs, D is equal for any per-window permutation of a sequence of windows, and sorting keeps every input; two hosts with different arrival orders give byte-identical D in `tests/host_admit.rs` | `window_perm`, `log_permwise`, `window_complete`, `ins_comm` |
| T41 | proved on a term model (Appendix E.2b): quorum intersection, Election Safety and Leader Completeness for every reachable state of the whole-log protocol model (elections with the vote rule, appends, fenced writes, commits), by strong induction on terms with no further hypotheses, a fenced append leaves a replica unchanged, and every committed prefix survives any number of term changes; partitions, reordering, crash and restart are tested in `tests/host_raft.rs` | `fenced_append_unchanged`, `committed_prefix_preserved`, `quorum_intersect`, `election_safety`, `elect_fresh`, `commit_seed`, `leader_completeness` |
| T42 | stated, unproved (Appendix F.9; F.12 lists what is implemented): Noninterference with labels: two runs whose inputs agree on every public-label field give equal public outputs, including the refusal outcome, for every accepted program (F.4.2) | none |
| T43 | stated, unproved (Appendix F.9; F.12 lists what is implemented): Robust declassification: a release is reached only on a path whose control context and deciding operands are trusted (F.4.2) | none |
| T44 | stated, unproved (Appendix F.9; F.12 lists what is implemented): Typed output: every structural metacharacter of a rendered `Text[C]` or `Template` of any context of 5.16 and F.5.2 comes from a literal segment (F.5.2, F.6.1) | none |
| T45 | stated, unproved (Appendix F.9; F.12 lists what is implemented): Constant time: two runs of a `ct fn` that differ only in secret-labelled inputs take the same sequence of branch, index and loop decisions (F.5.4) | none |
| T46 | stated, unproved (Appendix F.9; F.12 lists what is implemented): Contract enforcement: if a call returns, its `requires` held at entry and its `ensures` held at the return; otherwise the call's event is refused with the fixed record (F.6.2) | none |
| T47 | stated, unproved (Appendix F.9; F.12 lists what is implemented): Transactional calls: a call that returns by `fail` leaves the caller's `&mut` arguments and locals equal to their values at the call (F.4.4) | none |
| T48 | proved on a model of the measure rule: a step that strictly lowers a natural-number measure reaches the end of the run from every start (not a proof about the checker's acceptance of `decreases`): Termination by written measure: an accepted recursive group or loop with a `decreases` tuple ends (F.6.4) | `t48` (`RCE/Discharged.lean`) |
| T49 | stated, unproved (Appendix F.9; F.12 lists what is implemented): Ghost erasure: erasing ghost code does not change δ, λ or the log (F.6.3) | none |
| T50 | stated, unproved (Appendix F.9; F.12 lists what is implemented): Definition hash: two definitions with equal hash have equal meaning; a change that `rce api-diff` marks compatible does not change the result of any accepted caller for inputs satisfying its `requires` (F.4.3) | none |
| T51 | proved on a log model: a log with a revoke entry for a caretaker admits no use by it (not a proof about the host's implementation): Revocation: no capability use after a `host.revoke` entry for its caretaker in the log is admitted (F.5.1) | `t51` (`RCE/Discharged.lean`) |
| T52 | stated, unproved (Appendix F.9; F.12 lists what is implemented): Budgets: a program with a budget does at most the budget's units of work before `trap("budget")`, at the same operation on every target (F.5.5) | none |
| T53 | proved on a model of programs that ask an oracle: replay from the log of answers returns the live run's result (not a proof about the emitted code): World replay: T32 for every foreign call through a `World` and every logged input (F.4.1) | `t53` (`RCE/Discharged.lean`) |
| T54 | stated, unproved (Appendix F.9; F.12 lists what is implemented): Supply chain integrity: under SHA-256 collision resistance, an accepted build's modules are exactly the manifest's (F.5.3) | none |
| T55 | stated, unproved (Appendix F.9; F.12 lists what is implemented): Broker invariant: by induction on grant events, no session holds all three trifecta properties without an explicit grant event (F.7) | none |
| T56 | stated, unproved (Appendix F.9; F.12 lists what is implemented): Relaxed sequential borrows: the limited relaxation of 5.6 studied from group borrowing (a reference is invalidated only below steps that are not independent of it) preserves T6; the lemma is not proved and the relaxation is not adopted (5.6) | none |
| T15 | proved for iterators whose step between yields is total (T2): every run ends, delivers at most LOOP_LIMIT values and traps only after exactly LOOP_LIMIT; `break` yields a prefix of the full run; a run with a smaller cap is a prefix of one with a larger cap (the evaluator's re-run strategy) | `drive_total`, `drive_break`, `gen_prefix` |
| T57 | proved for the morphism equation (one-step unfolding of the behaviour of 6.10) on the Form of `RCE/Form.lean`; uniqueness of the morphism (Rutten 2000) is stated, unproved | `beh_cons` (`RCE/Behaviour.lean`) |
| T58 | proved on the Form of `RCE/Form.lean`, for one program and for two: a bisimulation forces equal behaviours and behavioural equality is a bisimulation | `bisim_beh`, `beh_bisim`, `behEq_iff_bisim`, `bisim2_beh`, `beh_bisim2` |
| T59 | proved over an explicit finite alphabet (the declared events over the boundary sets that `rce dim` explores), in both directions: agreement on every word of every length ≤ k puts the first difference beyond k, and a first difference beyond k (or none) gives agreement on every word of length ≤ k; the statement over all of E is stated, unproved | `words`, `AgreeAt`, `firstDiff`, `firstDiff_disagrees`, `firstDiff_le`, `firstDiff_none`, `trace_ball`, `ball_trace` |
| T60 | stated, unproved (a model proof by partition refinement is the next step): for finite S the quotient of reach(P) by the largest bisimulation is the minimal machine, with dim(P) states, unique up to isomorphism, and the canonical numbering of 6.10 makes its isomorphism class one string | none |
| T61 | tested for renaming only (`tests/dim.rs`, `shape_ctl_is_name_free`, against `rce fmt --nameless`); invariance under predicate-equivalent rewrites of guards is stated, unproved; the symbolic quotient of 6.10 is not implemented | none |
| T62 | proved on a model (`RCE/LogPO.lean`, spec/proposals/partial-order-log.md): entries with read and write footprints over an abstract store, independent when neither writes what the other reads or writes; independent entries commute, and any sequence of adjacent independent swaps of a layer folds to the same store. A model in the sense of this chapter: the link to the admitter is the HC-15 permutation tests of T40 | `commute`, `fold_swap`, `fold_perm_eq` |

T16 to T31 belong to the concurrency part (Appendix C) and are listed with their exact status
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
B_P(size(s) + size(e)) for a function B_P computable from P (from loop
bounds, SIZE_LIMIT and the call graph, which is acyclic except for
recursive cycles admitted by layered measures or declared bounds, 6.7). When the call graph
has no cycle, B_P(n) = B(P) · (1 + n). A recursive cycle contributes the
number of calls of a chain of calls, which is finite because its tuple of
measures falls in a well-founded order, and bounded by a function of the
sizes of the values it recurses on, themselves bounded by N9; that
function need not be linear (a function that calls itself twice on the
same component makes a number of calls exponential in the size).

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
parse tree. The same holds for the grammar with the concurrency productions (section 3 with
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

**T57 Finality.** beh_P of 6.10 is a coalgebra morphism:
beh_P(s)(e · w) = λ(s, e) · beh_P(δ₁(s, e))(w), and it is the unique such
map S → Beh.

**T58 Equivalence is bisimilarity.** beh_P(s) = beh_P(t) if and only if
there is a bisimulation R with s R t; likewise for two programs on
S_F × S_G.

**T59 Trace evidence is a ball.** If beh_F(s₀)(w) = beh_G(s₀)(w) for every
w with |w| ≤ k, then d(beh_F, beh_G) ≤ 2^-(k+1), and conversely.

**T60 Minimal machine.** For finite S, the quotient of reach(P) by the
largest bisimulation is a Mealy machine with the behaviour beh_P(s₀) and
dim(P) states; every Mealy machine with that behaviour has at least dim(P)
states, with equality only up to isomorphism; the canonical numbering of
6.10 makes the isomorphism class one string.

**T61 Control canonicity.** `shape_ctl` (6.10) is invariant under renaming
(T7) and under any rewrite of a guard to a predicate-equivalent guard; it
is not invariant under a change of a data update that changes a later
guard outcome.

**T62 Linearization invariance (proposal).** For a layer of D between
two cuts, any two orders that agree with every chain's `seq` order and
every `deps` edge fold δ₁ to the same state at the upper cut
(spec/proposals/partial-order-log.md).

**T32 Foreign replay.** For every oracle o answering foreign calls and
`select nondet` choices (5.11, C.5), replaying the log of a live run
reproduces its state, whatever o is; no property of the foreign code is
assumed, and a contract every answer satisfies holds of every logged
answer.

**T33 `&mut` restore on failure (F.4.4).** A call that ends by `fail` leaves
the caller's `&mut` state as it was at the call, and a call that returns writes
the callee's final value back. This is proved of a model that copies in, runs on
the copy and copies out only when the outcome is a value. The rule that wrote back
on every outcome, which this number used to state, is retired and is no longer
in the proved tree. That every effect since entry is undoable is W-TX-1, a check
of the checker; T42 to T56 state the remaining properties; T48, T51 and T53 are proved of models (`RCE/Discharged.lean`) and the other twelve are unproved.

**T36 Indexed values.** (5.16): untrusted text
forms no structure of a `Text[C]` value; outputs do not depend on
secrets beyond the equality bits; a guarded field changes only on events
that carry its capability.

**T35 Refinement invariance.** If the initial state satisfies the
predicates of its refined fields, every state in the fold of the log
(6.1) satisfies them, and an event whose payload or resulting state
breaks a predicate is refused with the state unchanged (5.1).

**T37 Host replay.** For every adapter oracle, replaying the log D with
every adapter disconnected gives the live run's state after every entry;
host entries leave the state unchanged; placement, a function of the
logged host entries, is the same on replay (Appendix E).

**T38 Verify on open.** The verified prefix of a hash-chained log is a
prefix of it and a valid chain, equals the log when the log is valid, and
cuts a valid log followed by an entry that does not link back to the
valid log (Appendix E, HS-4).

**T39 Fatal barrier.** After a failed storage barrier the admitter
appends nothing more, and a running admitter's log holds exactly the
entries it was given (Appendix E, HS-3).

**T34 Order-preserving A-normal form.** Flattening an expression into
`let`-bound steps in left-to-right order, with each temporary fresh,
gives the same value, final store, and ordered trace of budget charges
and emits as evaluating the expression.

T42 to T56 are stated in Appendix F.9 and are obligations of Appendix F. T48, T51
and T53 are proved, each of a small model of its rule (`spec/lean/RCE/Discharged.lean`,
in `lake build` and the audit); a model proof says the rule is coherent, not that the
checker or an emitter implements it. The other twelve are not proved: their statements
as Lean `Prop`s over abstract model records are in `spec/lean/obligations/Obligations.lean`;
that file has no theorem, is not part of `lake build` and is not read by
`scripts/lean-audit.sh`, so the audit counts do not include it.

**Numbering and assurance.** Theorem numbers are stable identifiers, not
positions: a number is never reused or changed, new theorems take the
next number (T32 on), and a split keeps its number with a letter (T19a,
T19b). Code, tests, Lean doc comments and papers cite these numbers, so
renumbering would break every citation for no gain in meaning. Each
row's status uses one of the levels proved (Lean, on the RCE semantics
or the named core), model (Lean, on an abstract model), checked (a
machine check that is exhaustive for its input, such as LALR(1)),
tested (sampled differential or property tests), in that order, with *stated, unproved* below them for a theorem whose
statement is written but which has no Lean statement, no test and no checker
rule yet (T42 to T56, Appendix F.9; T59 over all of E, T60 and T61, 6.10); such a row is an obligation and never a
claim. A
counterexample is a failure, never a lower level, and a claim about
several parts has the lowest level among them. Every Lean name in the
tables is checked to exist (`spec/checker/tests/theorem_index.rs`).

**Read receipt.** What is the deep embedding of the emitted Rust subset called?

<!-- body-sha256: 3c4b45822990cb3d6235a553c79ab4c18048d833709a51ee548de68f55ffd6ae -->
