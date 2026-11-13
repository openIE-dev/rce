# Implementation contract: block breaks, exit depth, and the `fn table` measure

Status: an implementation contract for the Form bridge. It is not a change
to the Form, the gate, or the ledger. It does not amend
[rce-form.md](rce-form.md) or [break-carried-flag.md](break-carried-flag.md).
A model proposes this work. It does not authorize a rule change.

The bridge this contract talks about is `spec/checker/src/form/bridge.rs`
on `main` at `50294e05`. The witnesses are
`spec/checker/tests/break_fg8_witness.rs`. They call the public bridge
(`construct`, `schedule`, `closure`, `body_break_rewrite`) and the core
evaluator. They do not replace that bridge.

## 1. Baseline measured on this commit

Counted by the same program-function rule as
`examples_are_covered_or_require_change` in
`spec/checker/tests/form_bridge.rs`: a root is `fn`, `region`, `delta`, or
`initial state`. The first failure of those roots is the example's tag.
`construct(schedule(F))` byte equality with no body-break rewrite is the
syntactic floor. A whole module is `construct` of the lowered module, not
of one closure.

| Count | Measured | Coordinator |
|---|---|---|
| programs in `spec/examples` | 60 | 60 |
| syntactic floor | 5 | 5 |
| whole modules | 0 | 0 |
| `defect` among program functions | 0 | 0 |
| body-break examples (option B, off the floor) | 0 | 0 |
| `break` | 8 | 8 |
| `FG8` | 1 | 1 |
| `catch` | 10 | 10 |
| `global` | 1 | 1 |
| `loop` | 1 | 1 |
| lowering | 4 | 4 |
| no program root | 21 | 21 |
| byte mismatch, tag `round-trip` | 9, named below | 9 |

No mismatch with the coordinator's floor, whole-module count, or defect
count. Whole-module first tags, which that report did not list, are
`break` 2, `catch` 11, `global` 35, `out` 8, plus the 4 that do not lower.

The eight `break` examples and the one `FG8` example are named in the
witness test. `21_maps.rce` (`fn table`) is that `FG8` row. It is not one
of the nine byte mismatches. A future form with `body_break_rewrite` set
and `construct(schedule(F))` equal to F is the option B row, which is 0
today. It is not the syntactic floor, and it does not join the nine.

The nine files whose first program-function tag is `round-trip` are
`03_fib_loop.rce`, `10_comptime.rce`, `22_tuples.rce`, `25_labels.rce`,
`36_multiline_brackets.rce`, `43_simd_lanes.rce`, `45_byte_text.rce`,
`46_u64.rce`, and `49_loop_limit.rce`. The tag means the scheduled bytes
differ. `22_tuples.rce` is in this list because `fn divmod` differs; a
later root, `fn use_it`, is `FG8`. `56_utf8.rce` has roots whose bytes
differ and is not in the list: its first tag is `catch`. `fn fib` in
`03_fib_loop.rce` is one of the nine, not a `break`. The remaining
`break` rows are not the option B flag.

## 2. Two gaps, and what is not a gap

**Gap A is an implementation gap licensed by the existing rules.** A `break`
out of a block that is not a loop, a `break` whose depth leaves more than
one block or loop, and a nested `ret` inside a loop are exits
[rce-form.md](rce-form.md) §13 already maps: a non-loop block and a break
out of it become Branch outputs; `break d` for d > 0 carries an exit count
(ONE-15 in [one-emitter.md](one-emitter.md)); a nested `ret` is
restructured into the region's outputs. The bridge refuses them. That
refusal is the slice, not a contradiction in §13. Implementing the
representation below does not change FG8, C7, FG6, or FG7.

**Gap B is an implementation gap in the schedule's round trip, not a missing
measure rule.** `fn table` in `spec/examples/21_maps.rce` lowers to `mset`.
`construct` of that closure is applied, `body_break_rewrite` is set, and
the gate accepts the measures. `schedule` then writes a core loop.
`construct` of that loop fails FG8: the body region does not return i
raised by exactly one whenever it continues (`node 3`). The closed measure
of §6 already holds for the first Form. Acceptance of the scheduled core
does not need a wider measure. It needs the second construction to recover
the first Form.

**Not a gap, and not to be "fixed" by a new reading of FG8.** A continue
path that does not raise the index is not a measure (§6, constraint B5 in
the break note). Until 2026-11-13 a pure count had no applied form: FG7
gives every count class at least 1, and C7 removed a carried ω that neither
region uses (O8). The erratum of that date keeps the carried order, so a
pure count is applied. FG8 is not the rule that changed. A guard on one
value does not establish a partial operation on another value, or on a
later value of the same variable (§7, DF5). The tests in section 6 pin
these refusals, and the pure-count case now pins the kept order.

## 3. Representation of a break

The core evaluator (`spec/checker/src/one/eval.rs`) is the behaviour the
Form has to match. A `block` turns `Break(0)` into the continuation after
the block and `Break(d)` into `Break(d - 1)`. A `loop` does the same, and
`Break(0)` ends the loop. Statements after a `break` in the same statement
list do not run. Statements after the exited construct run only when the
depth that arrives there is 0.

### 3.1 Depth 0 out of a block that is not a loop

Source witness: a block whose then-arm is `break 0`, whose else-arm stores
a result, and whose continuation returns that result. The witness returns
0 when the break is taken and 1 when the arm falls through. A fuel tick
placed after the `break` and before the end of the block does not run on
the break path. A tick in the continuation runs on both paths.

Representation, from §13:

- The block is one Branch. The break arm and the fall-through arm are its
  two regions. Outputs are the variables either arm stores and that the
  continuation reads, ω last when either arm or the continuation needs it.
- The break arm does not contain the statements that follow the `break`.
  Those statements are on the fall-through arm only.
- The continuation after the block is not inside the Branch. It runs on
  both arms. It is not sunk into the break arm.
- An owned value used on both arms is one capture (the schedule already
  does this for a branch that does not break). Each arm may consume that
  capture, because one arm runs. The continuation sees the Branch output,
  not the capture, when either arm moved it.
- The flag of a loop is not this Branch. A depth-0 block break does not
  set the loop's continue flag.

Invariant: fuel, cost, and `emit` occur only on the arm that runs them,
in the source order, on the ω chain (FG7, §6 effect classes, §8
linearity). A tick the break skipped is not a node of the break arm.

### 3.2 Depth that names the loop, and depth that names a further construct

A `break 1` inside a block that is the body of a measure names the loop
(the block is depth 0, the loop is depth 1). The witness is that loop.
The core result for a stop on the first iteration is the index at entry
(0), and neither the tick after the break nor the tick after the block
runs. Four iterations with the stop clear return 4 and charge both ticks
on each iteration. Zero iterations charge nothing.

The bridge does not refuse this witness with the block sentence. It builds
a measure and the gate rejects it: FG8, the body does not return i raised
by exactly one whenever it continues. That is the same tracing gap as the
flat `break 0` before the increment (section 4.3). Depth 1 here names only
the loop, so the representation is the option B flag, not an exit count:
the break arm returns false and does not contain the increment, and the
continue arm returns true and `int.add` of the carried index and 1. The
gate already accepts that shape for `mset`. This statement order does not
present it. The fix is the trace of section 4.2, not a weaker FG8.

An exit count is the representation when the depth still names a construct
after the innermost loop. The following is the coordinator's semantic
target for that count. It is not, by itself, a decision that the Form's
text has been amended.

- The carried count is the number of blocks and loops still to leave.
- The source depth is the `d` written on `break d`. Residual depth is what
  remains after some of those constructs have been left. They are not the
  same number once a construct has been left.
- Two initialisations are distinct. They are not one rule, and they are
  not combined.
  - Origin first, source depth `d`. ONE-15 writes `set e = d; break 0`,
    and after each construct left that way, `if e > 0 { set e = e − 1;
    break 0 }`. The originating `break 0` leaves the innermost block or
    loop and does not decrement `e`. The first decrement is at the edge
    after that construct has been left. `e` then counts the blocks and
    loops still to leave.
  - Decrement at the first exit, initial value `d + 1`. `e` starts one
    higher, and the originating exit decrements `e` as it leaves the
    innermost construct. Later edges decrement once each, as in the other
    case.
  Taking the initial value from one case and the first-exit decrement
  from the other counts the innermost exit twice, or drops it. Neither
  ONE-15 nor §13 says that the second case is the first. The coordinator's
  target is the first case. The second case stays a different
  initialisation. An implementation uses one of them for the whole exit,
  never both.
- Decrement once per block or loop that is left. Do not decrement on `if`.
  The core evaluator does this (`Block` and `Loop` subtract one; `If`
  returns the inner control unchanged). ONE-15 says "construct" and does
  not say "not an if". §13 cites ONE-15 and does not restate the `if`
  exception. The evaluator and W2, which counts enclosing blocks and
  loops, agree with the coordinator. That agreement is not a new sentence
  in ONE-15.
- Residual 0 at a block or loop means that construct is the target: its
  continuation runs. Residual greater than 0 suppresses that continuation
  and propagates outward. For a count loop, the propagation happens before
  the next iteration's counter increment and before the "loop bound" halt.
  It does not erase a count or a tick that the source has already executed
  earlier in the same iteration. Bound 0 on the two-loop witness halts
  with "loop bound" because `count` stands before the inner break. Bound 1
  with `break 1` returns, and the counter is 1.
- Normal fall-through is not a break whose residual has become 0. ONE-15
  keeps them apart because the `break 0` statement is the exit and `e`
  only says whether another `break 0` follows. A representation that uses
  only an integer, with 0 meaning both "no exit" and "exit finished",
  conflates them. The block-inside-if witness returns 30 when `break 0`
  is taken and 31 when the arm falls through: the same continuations, and
  the statement after the break runs only on the fall-through.
- The option B bool is used only once the structural depth has resolved to
  the loop that is the target and no further block or loop remains. Until
  then the carried value is the count, not a bool. This is a composition
  of §13's exit count and the break note's flag. No sentence in those
  texts says "bool only after". It is the coordinator's composition, not a
  third rule.
- A `ret` stays a separate control from the count. The arms are not padded
  to a common width. The `defect` witness stands.

The Form shape below is the origin-first case only (`e` starts at source
depth `d`; the originating `break 0` does not subtract). It is not the
`d + 1` initialisation, and it does not include that decrement.

- The exit count is a plain integer. At the break, the arm stores the
  source depth `d` and leaves the innermost block or loop. It does not
  subtract from `d` there. After each further block or loop, a Branch on
  `int.lt` of 0 and the count either runs that construct's continuation
  and carries 0, or carries the count minus one and does not run the
  continuation.
- Decrement happens once per construct left, at that construct's edge, not
  inside the arm that issued the break. A block decrements. A loop
  decrements. An `if` does not.
- When the count arriving at a loop is greater than 0, that loop's test
  and body do not run again, and its bound check does not run. The count
  is tested in the same place the loop flag is tested: before the counter
  is raised and before `halt` of "loop bound" (§6, §13, O6'). A break must
  not become the trap "loop bound".
- The count is not ω. Nodes that compute it do not read or write ω (O7').
  C7 removes a carried value that neither region uses and both return
  unchanged (C7, B4). The count survives only because the break arm stores
  a different integer than the input, or a region uses it in the test
  above. A count both regions return unchanged is not an exit.
- A `break` that names the loop under construction, and no further
  construct, stays the body's existing bool flag (option B). The exit
  count is for a depth that still names another construct after that flag
  is applied. The two are not the same port. One bool cannot also be the
  trip count.

A break that names no enclosing block or loop is already ill-formed in the
core (W2). The construction has nothing to admit.

### 3.3 Nested return

A `ret` in one arm of an `if`, with another `ret` after the `if`, is
already a Branch whose arms are the two results. The witness round-trips
and agrees for both selectors. That shape stays as it is.

A `ret` inside a loop body is `defect: branch arms return 1, 2 values
for 1 outputs`. The construction builds a branch and the arms do not
have one width. That is a defect of the tool, not an applied form, and
not the clean `exit` sentence. §13 says a nested `ret` is restructured
into the region's outputs. The return is not the Iterate's continue
bool. Balancing the arms with a dummy output would hide the defect and
change the value. The witness core returns 7 when the return is taken,
the index 4 when it runs to the end, and 0 when the test fails at once.
A later change matches those three results. A path that both returns and
breaks stays `exit: a break path and a ret in one loop body`.

### 3.4 What the partner changes, and what the partner leaves

The change is in the construction's reading of `Block` and of `Break(d)`,
and in the schedule's writing of the Branch and the exit count so that
`construct(schedule(F))` is F for the forms that do not use the option B
flag. Where the option B flag is used, byte equality is the flag form
against its schedule, and the example stays off the syntactic floor, as
`form_bridge.rs` already requires.

Do not edit the gate's `measure_shape` to make a block break look like a
measure. Do not edit C7 to keep an unused carried bool. Do not edit FG7 to
drop ω from a count.

## 4. `fn table` and FG8

### 4.1 What the closure actually does

`fn table` builds `{"a": 1, "b": 2}` by two calls of
`mset:Map[text, int{lit}[1, 2]]`. `mset` has two loops, each
`loop { block { ... break 1 ... } }`. The search loop is a measure: the
test is `int.lt` of the index and the length, the break arm of the body
returns the bool false and does not step the index, and the continue arm
returns true and `int.add` of that same index and the literal 1. The add
sits under the domain Branch of `int.bad.add`, whose other arm halts. The
length test of `list.get` is the outer Branch, and its halt arm does not
continue. The copy loop that rebuilds the list has no body break: its flag
is the literal true, and every continuing arm raises the index.

`construct` of the closure is applied. The gate's `raised` walk accepts
this because the body's flag and the returned index are two outputs of one
Branch, the break arm's flag is a literal false with no inputs, and on the
continue arm the add's operands trace through captures to the carried
index and to a literal 1 (§7's two same-value edges: a capture, and a
read-return operand). `list.get` in that body is dominated by `int.lt` of
that same index and `list.len` of that same list (DF5).

So the closed shape is enough. The immutable index and the length are the
carried inputs. A capture of a carried input is that input. A loop-carried
update is a different source, and the next iteration's test sees the
update, not the previous value. Nothing in this closure needs a second
measure rule.

### 4.2 Where the round trip loses the shape

`schedule` writes each Iterate as one core `loop`: the test, stores of the
carried results, `if flag else break 0`, the body, stores, `if flag else
break 0` (`schedule_iter`). The body's `int.add` is inside the domain
Branch. The epilogue then stores the carried index from whatever that
region returned, and the flag is a separate `if`.

`construct` of that loop builds a measure the gate rejects at `node 3`
with "the body region does not return i raised by exactly one whenever it
continues". The first Form still has the add on the continue path. The
scheduled spelling, read back, does not present that add as the index
output on every path whose flag is true.

The fix is to keep, in the scheduled core and in the construction that
reads it, the same sources the first Form has:

- the continue flag and the returned index are outputs of one Branch;
- the break arm's flag is a fresh literal false, not the carried flag
  (B4: C7 would delete a bool both regions only pass through);
- `int.add` reads the carried index, or a capture of it, and the literal 1,
  or a capture of that literal;
- a domain Branch may wrap the add; its halt arm is not a continuing path;
- `j` is returned as the same carried input on every path, including the
  break path and a halt path's outputs where the region still has to name
  them;
- ω, if a region uses it, stays the first carried value and is not an
  operand of the add or the flag (O7', FG7).

When that holds, `construct(schedule(F))` is F. FG8 stays as written.
`body_break_rewrite` stays set, because the search loop's break was
restructured. The example moves from `FG8` to the option B row. It does
not enter the syntactic floor of 5. The witness that pins the refusal
today is replaced, in that change, by byte equality of the two forms and
by core eval, Form eval, and scheduled core eval of `table()` agreeing on
the list `(a, 1), (b, 2)`.

### 4.3 Same value, guards, and loop-carried state

§7 names two edges. A capture is the same value. A read-return operand is
the same value. A `let` of a plain integer is not one of those edges, and
this contract does not add it. The schedule must pass the carried index
into the add, not a distinct node that merely holds the same bits. Widening
"same value" to every equal integer would also make a guard on one list
cover a `get` of another list with the same elements. That is unsound
(DF5). The negative witness builds two lists that both hold 1, guards the
length of the first, and gets the second. The core rejects it with W4.

A guard on an index does not survive a store of that index. The stale
witness guards `0 < len`, adds one, and gets at the new index. W4 rejects
it. A missing guard is the same rule with no test at all.

An unchanged end is the carried input `j`, returned as `Src::In(j)`. The
copy loop of `mset` does this, and the gate accepts it. A body that stores
a new end is not that shape. The witness sets the end to the current index
and still terminates: the result is 0 when the end starts at 0, and 1 when
it starts at 4, because the next test sees the stored index. The bridge
refuses the loop with `loop` (none of the two-region shapes). It does not
apply it as a measure. Extending `find_measure` to this loop would still
have to fail FG8, whose test requires `j` unchanged. Accepting the loop
as a measure would let a later edit keep the test true forever.

## 5. Negative counterexamples

Each shortcut admits a program the rules do not admit. The witness module
is the smallest core that shows it. The test asserts the refusal that
`50294e05` actually gives.

1. **Skip the increment and keep the flag true.** A measure whose continue
   path sometimes does not add one (`if c { i = i + 1 }` with no break) is
   not the closed shape. The bridge refuses it with FG8 and does not build
   a count. Encoding the break the same way, with the flag left true, does
   not terminate when the break condition stays true. The search loop stops
   because the flag is false, not because the index moved.

2. **Turn the measure into a count to avoid the check.** A count may halt,
   so FG7 requires ω. Since 2026-11-13 C7 keeps a carried ω that neither
   region uses, so a pure count is applied, including one whose body breaks.
   The flag is a bool and is not a use of ω. Relabeling `mset` as a count
   still adds a "loop bound" halt the measure does not have. Empty input
   and a one-entry map must not trap with "loop bound". The witness checks
   the value, not a `bound` tag.

3. **Test the flag after the bound.** For a count that does use ω, the
   schedule tests the flag before raising the counter and before the bound
   halt (§6). Testing it later turns a break on the last legal iteration
   into "loop bound". The option B witnesses already in `form_bridge.rs`
   (`early_count`, bound 10 stopping at 2, and bound 2 trapping) stay the
   check. This contract adds no second bool after the halt.

4. **A pass-through flag.** Both regions returning the incoming bool, with
   neither using it, is deleted by C7. The early exit disappears. The
   break arm must store a literal false that is not the carried input.

5. **A flag that ticks.** A `fuel.tick` on the break arm, or on the nodes
   that build the flag, changes the fuel and the ω order (O7', B3). The
   block witness places a tick after the break and records that the core
   does not charge it. The Form arm must not contain it.

6. **Treat the block break as a loop break.** The continuation after a
   depth-0 block runs. The witness returns 1 from that continuation. A
   loop flag set by that break would skip the continuation and return the
   index instead.

7. **Forget to decrement, or treat a loop-naming break as a block break.**
   `break 1` from the inner block leaves the loop. The increment after
   the block does not run, and the core result is 0. Reading it as a
   depth-0 block break would run the increment. The bridge instead builds
   a measure the gate rejects (FG8). Either wrong reading is unsound; the
   flag of section 3.2 is the one that matches the core. A `break 1` from
   an inner loop, which names the outer loop as well, is refused with
   "a break that leaves more than one loop". It is not admitted as a
   measure. The exit count above is what a later change builds for it.

8. **Two consumers of one owned list.** Both arms of the block push the
   same list. The parent has one capture. Duplicating the source into each
   arm, while the continuation also reads the original, is two uses
   (FG6). The witness is refused today as a block break; the future form
   is applied only with one capture and two different list results, one
   per selector.

9. **Same bits, different value.** Two lists built by two `list.new` calls,
   each holding 1, are not the same value. A length test of one does not
   guard `list.get` of the other. W4 rejects the core. The Form must not
   be given a chance to accept it by comparing elements.

10. **A stale index.** The test and the `list.get` must see the same index
    source (§7, DF5). An add between them produces a new source. W4
    rejects the core.

## 6. Tests

`spec/checker/tests/break_fg8_witness.rs`, run with:

```sh
cargo test --test break_fg8_witness
```

from `spec/checker`. The corpus count is the slow one. No test is ignored.
A refusal is `assert_eq` on the tag and the reason text the bridge returns
today. The core evaluator's result is asserted beside it, and the contract
section above says what an applied form has to match. The tests do not
construct a second Form of their own.

| Witness | Today | Future, when the partner lands the fix |
|---|---|---|
| Corpus floor | 5 covered, 0 whole, 0 defect, 8 `break`, 1 `FG8` | Floor stays 5. `fn table` may leave `FG8` for the option B row. It does not become covered while `body_break_rewrite` is set. |
| `fn table` (`21_maps.rce`) | `FG8` today, not one of the nine byte mismatches. First `construct` applied, rewrite set; `construct(schedule(F))` is FG8 at node 3 | If the bytes then match and the rewrite flag stays set, the file moves to the option B row (0 today). It does not become covered, and it does not join the nine. Three-way eval of `table()` is the list `(a, 1), (b, 2)` |
| Block break, both selectors, ticks | tag `break`, the block sentence; core results 0 and 1; fuel as counted | Branch outputs; continuation runs; the tick after `break` is absent on that arm; fuel matches |
| `break 1` out of a block in a measure | FG8, index not raised on every continue path; core 0, 0, and 4; fuel 0, 8, and 0 | the option B flag, because this depth names the loop; increment and its tick do not run on the break; not "loop bound" |
| Flat measure with a body `break 0` | FG8, index not raised on every continue path; core results for 0, 1, and 4 iterations | Unchanged until a construction chooses to recognize this statement order. That choice is slice coverage, not a weaker gate. A recognized form still has to pass FG8 as `mset` does |
| Nested `ret` outside a loop | applied, round trip, both selectors | unchanged |
| Nested `ret` inside a loop | `defect`, arms return 1 and 2 values | region's outputs on the return path; results 7, 4, and 0; not a break flag |
| Owned list, asymmetric arms | tag `break` | one capture; the two list results; FG6 holds |
| Conditional increment | FG8, not a count | still refused |
| Mutated end | `loop`, not a measure; core results 0 and 1 | still not a measure |
| Pure count, stale and foreign guards | `bound`, or W4 | still refused |
| Two-loop `break 1` | tag `break`, "leaves more than one loop". Core: markers 2; counter 1; fuel 2. Bound 0 halts "loop bound". Inner `break 0` with bound 1 halts "loop bound" after fuel 3 | exit count, not a bool, while the outer loop remains. Next count does not run. The count already passed in this iteration still runs |
| Block inside `if` inside loop | Depth 0: tag `break`, the block sentence; core 30 if the break is taken and 31 if the arm falls through. Depth 1: FG8; core 8. Depth 2, and a `break 1` with no such construct: W2, not the evaluator's "break outside a block or loop" | `if` does not decrement. Depth 1 is the loop's flag only if nothing further remains; otherwise the count. W2 stays the refusal for a depth the function does not have |

## 7. The next change, for the partner who edits the bridge

One change, in the bridge only, after this contract:

1. Read a depth-0 `break` of a non-loop block as the Branch of section 3.1,
   for a block that stands alone and for the continue-block a `for` lowers
   around its body (`spec/checker/src/one/lower.rs`, `for_stmt`). Do not
   classify that block as "a block that is not a loop" and stop.
2. Read `break d` for d > 0 as the exit count of section 3.2, decremented
   once at each block and each loop, tested before a count bound. A break
   that names only the loop under construction stays the option B flag.
3. In `schedule_iter` and in the construction that reads that loop back,
   keep the measure body's flag and index as two outputs of one Branch,
   each tracing to the carried index and the literal 1 through captures,
   so `construct(schedule(F))` equals F for `fn table`. That file then
   sits on the option B row. It is not one of the nine byte mismatches,
   and it is not covered. `measure_shape` stays.
4. Leave FG8, FG6, and FG7 as they are. The C7 erratum of 2026-11-13 is
   already adopted: a pure count keeps its carried order and is not refused
   with `bound`. Do not restore that refusal. Leave catch, `out`, globals,
   and a loop that is not a measure or a count (`rt.f64.parts` in the
   `fn dot4` closure is "a break outside a recognised loop") for their own
   changes.

`spec/checker/tests/form_bridge.rs` is the other run's file. Extend
`break_fg8_witness.rs` rather than editing it. Raise the syntactic floor
only when a new example round-trips without `body_break_rewrite`.

## 8. Questions for David

1. Confirm that §13 and ONE-15 already license the Branch and the exit
   count, and that item 3 of the break note only marks the current slice.
   The note's "still requires change" would then stay true until this
   representation is built, and would not be a second rule.
2. Confirm that `fn table`, once the schedule round-trips, stays off the
   syntactic floor while the search loop's break is a flag. The floor
   stays 5.
3. Confirm that the gate's acceptance of `mset`'s first Form is the
   intended reading of "whenever it continues", and that the scheduled
   core must be brought up to that Form. The gate is not to be widened to
   accept the scheduled core as it is built today.
4. The flat `loop` with `break 0` before the increment is refused FG8 on
   the first construction. Recognizing that statement order is a choice of
   what the slice supports. Confirm that the choice is not a change to
   FG8, and that the witness stays FG8 until the construction changes.
5. A `ret` inside a loop is `defect` (the arms of the branch the bridge
   builds do not have one width), not `exit` and not the break flag.
   Confirm that a later change may restructure it as the region's outputs
   and must not encode it as the Iterate flag or hide the defect by
   padding an arm.
6. `rt.f64.parts` and the other breaks in the `fn dot4` closure are "a
   break outside a recognised loop". Confirm they are out of this contract.
   Forcing them into a measure would drop the "raised by exactly one"
   requirement.

## 9. Witnesses remeasured on 410bfd40

#157 (`d77075a2`) and #158 (`34c915fd`, merged as `fa2be77d`) landed before
this contract's witnesses. The witnesses in
`spec/checker/tests/break_fg8_witness.rs` were pinned to `50294e05` and to
the bridge at `924f88e4`. They now assert the bridge on `410bfd40`. The
section 1 table and the "today" column of section 6 stay the measurement
of `50294e05`. They are not the live rows.

What the witnesses check on `410bfd40`:

- Census `whole` is 2: `09_generics.rce` (entry-only) and `35_nested_if.rce`
  (a user function). The syntactic floor stays 5. `construct` alone, with
  no schedule, accepts 18 modules. One of those 18 is entry-only. The
  other 17 contain a user function, and 16 of them do not schedule back
  to the same bytes, so they are not `whole`. The 18 entry-only cores
  named in #158 were the count before library functions were kept in the
  core. This 18 is a different count.
- A break that is not inside a count or a measure is still refused:
  `break: a break outside a recognised loop is an exit this slice does not restructure`.
  That is the standalone block, the owned list, and `rt.f64.parts`.
- `break 1` inside a block in a measure constructs. The form returns the
  core's value (0, 0, and 4; fuel 0, 8, and 0). The schedule is FG8 at
  node 1, with the same "raised by exactly one" reason.
- A bare inner `loop { break 1 }` is not a recognised loop. The refusal is
  the `loop` shape sentence. The core results of that witness are unchanged
  (markers 2, counter 1, fuel 2 and 3, bound 0 halts with "loop bound").
- Depth 0 of the block-inside-if witness constructs. The form and the
  scheduled core return 30 when the break is taken and 31 when it is not.
  Depth 1 constructs and returns 8; its schedule is FG8 at node 1. Depth 2
  is still W2. The `if` does not decrement.
- `21_maps.rce` does not lower (`fn freq` mixes a literal with `u32`).
  `fn table`, lowered from its own line, is applied, `body_break_rewrite`
  is set, the value is `(a, 1), (b, 2)`, and `construct` of the schedule
  is FG8 at node 3.
- `fn main` in `02_hello_main.rce` is refused at `rt.escaped` with
  `exit: a loop body does not fall through`. The return is not the loop flag.
- The flat body `break 0`, the conditional increment, the mutated end, and
  a `return` inside a loop stay refused, as section 6 says. A pure count is
  applied: C7 keeps its carried order (erratum of 2026-11-13). The flag is
  still not a use of ω.
