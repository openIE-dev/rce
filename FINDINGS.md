# Findings

Each finding states the law, what the reference or spec does, and a derived
fix. Tests follow the law, so the reference fails them until fixed.
Machine-readable records are in `findings/*.json`; the tests the reference fails because of them are listed in `findings/known-failures-reference.txt`, which CI holds fixed in both directions.

1. **N2/N4 `as i64` trap text.** An out-of-range `x as i64` trap message shows
   a value saturated to i128 rather than the exact source value. Law: a trap
   diagnostic reports the value that failed. Fix: format the exact operand.
2. **F9/F6 `-0.0` text.** f64 `tostr(-0.0)` prints `0`; f32 prints `-0`. F9
   requires text round-trip, and `0` reads back as +0. Fix: print `-0` for f64.
3. **T7 renames and D.** T7 says renaming keeps log hashes, but event and
   error names are part of D's input and D is injective, so renaming them
   must change the hash. Fix: restrict T7 to names that do not appear in D
   (functions, locals). Tests rename only those.
4. **W-TY-3 reported as W-TY-1.** Fix: emit the specific rule code.
5. **W-STATE-3 reported as W-TY-2.** Fix: emit the specific rule code.
6. **W-TRAIT-1 not enforced.** An impl whose method return type differs from
   the trait is accepted. Fix: compare signatures at impl check.
7. **W-MOD-2 import cycle crashes.** The reference overflows its stack instead
   of reporting W-MOD-2. Fix: track the import stack and report the cycle.
8. **N9 text size limit.** `+` on text accepts results of 2^21 bytes. Fix:
   check SIZE_LIMIT in text concatenation as for other constructors.
9. **R5 explicit generic call.** `id[i64](7)` does not resolve. Fix: accept
   explicit type arguments at call sites as the grammar allows.
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
14. **EXAMPLE-TRAP.** An `example` can state `refused: E` only for a declared
    error; no form states that the last event traps (`refused: trap` is
    W-NAME-1). So emitted programs cannot be tested on trap refusals. Fix: add
    `-> trapped(<kind>)` to example syntax.
15. **Conservative, not a bug.** The lex Merge certificate accepts only the
    expression form and integer keys. Sound but incomplete.
16. **FUEL-1 not in the book.** Fuel exhaustion is uncatchable (directive of
    2026-10-15), but no chapter says so; C.5 says `with budget ... else`
    "makes exhaustion a value" without excluding fuel. Derivation: fuel is
    the host's bound on a step (6.7, HC-7), not a resource the program
    holds; if an Else could observe it, the value of a program would depend
    on the host's limit and on how far a branch got before it ran out, so
    replay (T1) and schedule independence (T16) would no longer fix the
    outcome. Fix: add FUEL-1 to C.5 and 6.7 with the text in
    `clauses-pending.json`. The reference already behaves this way; six
    tests in `tests/evaluation-fuel/targeted/fuel-1-*` hold it, two of them
    controls showing budget exhaustion stays catchable.
