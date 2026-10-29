# Findings

Each finding states the law, what the reference or spec does, and a derived
fix. Tests follow the law, so the reference fails them until fixed.
Machine-readable records are in `findings/*.json`; the tests the reference fails because of them are listed in `findings/known-failures-reference.txt`, which CI holds fixed in both directions.

1. **CLOSED. N2/N4 `as i64` trap text.** An out-of-range `x as i64` trap message shows
   a value saturated to i128 rather than the exact source value. Law: a trap
   diagnostic reports the value that failed. Fixed in the engine: `trunc_chk`
   formats the exact integer value of the float, in the evaluator and all four runtimes.
2. **CLOSED. F9/F6 `-0.0` text.** f64 `tostr(-0.0)` prints `0`; f32 prints `-0`. F9
   requires text round-trip, and `0` reads back as +0. Fixed in the engine:
   f64 prints `-0` in the evaluator and all runtimes; the Python runtime's
   `floor`/`ceil`/`round` also keep the sign of a zero result (F7). Tests
   whose outputs were read off the old reference now expect `-0`.
   The fix exposed that `gen_numbers.py` computed floor/ceil as +0 for a
   negative operand (IEEE 754 keeps the sign), so 11 generated expectations read
   `emit -0`, and the harvested `57-float-vectors` tests carry the corpus's `-0.0`
   for `Mul(0.0, -1.0)` (the engine repository (commit omitted)).
3. **T7 renames and D.** T7 says renaming keeps log hashes, but event and
   error names are part of D's input and D is injective, so renaming them
   must change the hash. Fix: restrict T7 to names that do not appear in D
   (functions, locals). Tests rename only those.
4. **CLOSED. W-TY-3 reported as W-TY-1.** Fixed: int/float mixing and implicit
   narrowing (a target range that does not contain the source range, such as
   i64 to usize or usize to i64) draw W-TY-3; implicit widening stays W-TY-1.
5. **CLOSED. W-STATE-3 reported as W-TY-2.** Fixed: the checker emits W-STATE-3.
6. **CLOSED. W-TRAIT-1 not enforced.** An impl whose method return type differs from
   the trait is accepted. Fixed: full signature comparison with `Self := T`.
7. **CLOSED. W-MOD-2 import cycle crashes.** The reference overflows its stack instead
   of reporting W-MOD-2. Fixed: the linker keeps an import stack and reports `a -> b -> a`.
8. **CLOSED. N9 text size limit.** `+` on text accepts results of 2^21 bytes. Fix:
   check SIZE_LIMIT in text concatenation as for other constructors. Fixed:
   SIZE_LIMIT is checked on every value result in the evaluator and all runtimes.
9. **CLOSED. R5 explicit generic call.** `id[i64](7)` does not resolve. Fix: accept
   explicit type arguments at call sites as the grammar allows. Fixed in
   checker, evaluator and emitters.
10. **Unreachable rules.** W-TY-2 on `τ??` and W-STATE-1 (state with no
    fields) cannot be triggered: the grammar refuses both as W-SYN-1. Fix:
    either drop the rules or let the grammar admit the forms and the rules
    reject them.
11. **T38/HC-8 torn D.** Replay of a torn log errors while reopen keeps the
    verified prefix. Law: replay and reopen are the same function of the
    durable prefix. Fix: replay truncates at the first unverifiable entry.
12. **Wasm playground depth.** Deep recursion (7000 frames, and a 10000
    depth trap test) crashes the playground instead of returning the result
    or a depth trap. Fix: bound depth with a trap before the wasm stack.
13. **tree-sitter-rce `with budget ... else`.** The grammar does not parse it.
    Fix: add the production.
14. **CLOSED. EXAMPLE-TRAP.** An `example` can state `kept: E` only for a declared
    error; no form states that the last event traps (`kept: trap` is
    W-NAME-1). So emitted programs cannot be tested on trap refusals. Fix: add
    `-> trapped(<kind>)` to example syntax. Fixed as `kept: trap("K")` (book
   3.2 grammar, 6.9 semantics derived from T2); the emit adapter now states
   trap refusals.
15. **Conservative, not a bug.** The lex Merge certificate accepts only the
    expression form and integer keys. Sound but incomplete.
16. **CLOSED. FUEL-1 not in the book.** Landed as E1 in chapter 7
    (the engine repository (commit omitted), derived from fuel monotonicity). Tests
    `evaluation-fuel/targeted/fuel-1-*` now cite E1; the pending entry is
    retired in `clauses-pending.json`.
17. **T38 on Windows.** Reopening a torn D failed on windows-x86_64 with
    `host.StorageFault: cut: Access is denied`: the store truncated through
    its append-mode handle, which on Windows lacks write-data access. Law:
    recovery (T38, HS-4) is the same on every target. Fixed in the engine
    (`FileStore::cut` uses its own write handle), found by
    `host-contract/targeted/t38-torn-tail-keeps-prefix` in the parity matrix.
18. **CLOSED. host_contract test drift.** `entry_hash` in
    `tests/host_contract.rs` hashed serde float text instead of the canonical
    N3 form, and the proposed-rules test still listed N-1 after adoption.
    Fixed in the engine test.
19. **CLOSED. `rce emit scala` refusal status.** A refused program gave status
    2 for Scala but 1 for the other targets; 10.1 rule 4 makes a refusal
    status 1. Fixed in `cli.rs`; found by `tooling/emit-scala-change`.
20. **Par reduction tree vs fold.** Not a bug: a `for ... par` reduction is
    evaluated on the normative split tree of C.4, so on non-associative float
    addition it can differ from `fold` (`proposed/tu-6-*`: 2 against 1.5).
    Both are fixed by the program, never by the schedule.
21. **Mutation survivors (round 4).** 112 is not equivalent: negating the
    stop check after an actor delivery (`conc.rs` `actors_phase`) serves one
    actor per tick, which reorders outputs when two actors have mail in the
    same tick. `proposed/act-1-every-actor-steps-each-tick` kills it, but it
    rests on the proposed rule in 22, so the kill counts only once 22 is
    decided. 133 is equivalent with respect to the spec: the sample loop
    runs only for a kind that is not an enumeration, and there C.11.5 makes
    the verdict depend on the certificate alone ("never by sampling"). A
    certified merge satisfies the laws on every sample (`MergeCert.cert_laws`),
    so it is accepted with or without the cap. An uncertified merge is refused
    as W-CRDT-1 either way. The mutant changes only the message text
    (a law counterexample against "no W-CRDT-1 certificate"), and the harness
    checks diagnostic codes, not text, which the book does not fix.
22. **6.8 actor scheduling is underspecified (proposed ACT-1).** "Each actor
    processes at most one message per tick" also allows serving one actor
    per tick, as mutant 112 does. The reference serves every actor with mail
    once per tick, in id order. Proposed fix (needs David's decision): "every
    actor whose mailbox is nonempty at the start of the actor phase processes
    exactly one message". Recorded as non-normative ACT-1 in
    `clauses-pending.json`.
