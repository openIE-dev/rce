# Proposal: a carried break flag on the Form bridge

Status: open. The bridge follows option B. A `break` that names the loop and
sits in the body is that body's flag: the break path returns false, and the
continue path returns true with the carried values. Admission is three-way
eval, with the flag tested before the bound (O6') and the flag's nodes not
reading or writing ω (O7'). That is not syntactic identity.
`construct(schedule(F)) = F` as canonical bytes, where it holds, is the flag
form against its schedule. Option A is not implemented: `Stmt` in
`spec/lean/RCE/Form.lean` has no break constructor, so the Iterate has no
preimage there. The note does not change the Form encoding, the gate, the
runtime, or the Lean model, and it does not claim a production compiler.
The normative statement of Iterate, of ω, and of construct ∘ schedule stays
[rce-form.md](rce-form.md) §6, §13 and §14.
PR #145 (open, `cursor/form-bridge-widen-c155`) widens the bridge
implementation; this note does not edit `spec/checker/src/form/bridge.rs`
or `spec/checker/tests/form_bridge.rs`.

The mathground formal sketch of the bridge identity is not a file in this
repository. The obligations and the six breaks of syntactic identity below
are that sketch. The paths a reader can open are the ones the sketch
constrains: `spec/lean/RCE/Form.lean`, `spec/lean/RCE/Iter.lean`,
`spec/checker/src/form/canon.rs` (C7) and `spec/checker/src/form/gate.rs`
(FG7).

## 1. Terms

**Dom** is the set of canonical Forms for which the bridge slice claims
identity. A pure count loop is in Dom: the erratum of 2026-11-13 keeps
its carried order (§4.2). The earlier obligation O8 put that loop outside
Dom, because C7 then removed the order.

**Syntactic identity** on Dom is `construct(schedule(F)) = F` as canonical
bytes. Canonical bytes already apply C7 (capture order by digest, and
deletion of a carried value that neither region uses and both return
unchanged) and literal sharing (one plain node for one literal value, C4,
and a literal the region has already built staying that source). Identity
is not equality of a graph before those rules.

**Three-way eval** is a separate check on the same arguments: the core
evaluator, the Form evaluator, and the core evaluator on the scheduled
module agree, halt text included. On Dom it accompanies syntactic identity.
It does not replace syntactic identity.

**A two-region Iterate** is node kind 4, either `["count", …]` or
`["measure", i, j, …]`. Its test region and its body region each return
`[bool, carried values…]`, with ω first among the carried values when the
Iterate takes ω. The bool is the region's flag. The schedule writes
`if not k { break 0 }` after each region. There is no sixth node kind.

**The carried exit flag**, written `go` when a sentence needs a name, is
that existing bool used for a `break` that targets the loop. A break sets
the flag; the body returns false. `go` is not a new port and not a new
node.

**A pure count** is a count-bounded Iterate whose test and body do not use
ω except by carrying it. FG7 gives every count effect class at least 1, so
the node must consume and return ω. Until 2026-11-13, C7 then removed that
carried ω, and the slice had no applied form for that loop (`bound` in the
bridge). That was O8.

**Erratum, 2026-11-13.** C7 now keeps the carried order of a count-bounded
Iterate. A pure count has an applied form: the count carries order, and the
regions may only pass it through. A pass-through bool is still removed.
FG7 is unchanged. See rce-form.md §10.

**An exit sum** is a result with one constructor per way a loop ends. The
iterator model already has three: `.done`, `.stopped` and `.bound`
(`spec/lean/RCE/Iter.lean`). The Form `Stmt` does not.

Two letter sequences appear below, and they are not the same sequence.
**Step A** and **step B** are the code order after PR #145. **Option A**
and **option B** are the two ways this note allows the identity claim to
move. Item 3 is the design of the flag; step B is when code for that
design would start.

## 2. Sequence

The AI-Leapfrog attack order places this design at item 3.

Item 3 is a carried exit flag through the existing two-region Iterate. A
break sets the flag and the body returns false. No new node kind is added.
A break outside those shapes still requires change: a break out of a block
that is not the loop, a break whose depth does not name that loop, a break
in a loop that is not a recognised measure or count, and a nested `ret`.
The bridge already refuses those with the tags `break` and `exit`. Item 3
does not admit them.

Code for item 3 waits on the order after PR #145 merges:

1. Step A is a triage of `defect` and of FG6. It lands before any break
   change.
2. Step B is the carried exit flag of item 3. It lands after step A.

Later steps of that order (a catch shape, linking `out` to `emit`, ω for
globals) are not this note. Catch shapes are not edited here, and are not
part of item 3. C7 against FG7 on a pure count stays a decision of its own;
a break flag does not close it (O8, §4).

Corpus floors are not lowered. On `main`, `examples_are_covered_or_require_change`
requires 60 programs in `spec/examples` and 0 whole examples constructed,
and the test comment says to raise that floor rather than lower it. PR #145
proposes a second floor, 2 of 60 program functions (`fn escapes` in
`spec/examples/20_strings.rce`, `fn axpy` in `spec/examples/42_f32.rce`),
and keeps whole modules at 0 of 60. This note changes neither number. A
later implementation of step A or step B raises a floor only when a function
newly round-trips. It does not lower either floor to admit a partial result.

This slice is not the production compiler. The hand emitters still walk
the syntax tree. Nothing in this note, and nothing an implementation of
item 3 would print, is that claim. The claim stays refused while a break,
a catch shape, or the pure-count case (C7 against FG7) still requires
change.

## 3. Identity today

For F in Dom, `construct(schedule(F)) = F` in the sense of §1, and
three-way eval agrees. That is the direction [rce-form.md](rce-form.md)
§14 uses: construct ∘ schedule is the identity on canonical Forms. The
schedule writes no exit of its own beyond `if not k { break 0 }` after a
region. schedule ∘ construct is already not the identity on the core (§13).

The mathground sketch states that identity as obligations O1 to O10. This
note does not restate O1 to O7, O9 or O10; their wording is the sketch's.
The obligation this design kept visible was O8: a pure count was outside
Dom, because FG7 demands a carried ω and, until 2026-11-13, C7 removed it.
The erratum of that date keeps the carried order, so O8 does not hold
after it.

The obligations are a plan for a proof. They are not a proof, and this
note does not mark them proved.

## 4. What item 3 would change, and what it would not

The construction today, including the widening on PR #145, reads a
lowering loop as an Iterate when the loop is a measure (`find_measure`:
`int.lt` of the index and the end, or `int.le` with the index second) or a
count whose test or body uses ω. The body's flag is the literal `true`.
The only `break` that is read back as a flag is the loop's own exit test
(`peel_exit` on the test prefix). A `break` in the body is still an exit
the slice does not restructure. `exec` reports it as
`break: a break outside a recognised loop is an exit this slice does not restructure`.

Item 3, when it is implemented, would treat a `break` that names that loop
and sits in the body region as the body's flag. The break path returns
false. The path that continues returns true, and it still returns the
carried values. Statements after the break are not on the break path. The
test region is unchanged: it returns false when the loop condition fails,
and true when the body should run. The schedule already places
`if not k { break 0 }` before the counter is raised and before
`halt "loop bound"` (§6, §13). The flag test stays in that place.

### 4.1 Measure

A measure is applied only in one closed shape (§6). The test returns the
index and the end unchanged, and its flag is `int.lt` of those two carried
values. The body returns the end unchanged and the index raised by exactly
one whenever it continues. `find_measure` accepts that head and then
requires the tail to fall through.

A break is not a conditional increment. If the continue path sometimes
does not raise the index, the loop has left the measure shape, and
`find_measure` fails (constraint B5, §5). The continue path of a measure
that also breaks still raises the index by one. The break path returns
false and does not have to step the index, because that path does not
continue. Encoding the break as "do not increment" is not item 3, and it
is not a measure.

A measure does not take ω unless a region uses ω. A flag of type `bool`
does not by itself make the measure take ω.

### 4.2 Count, ω, FG7 and C7

A count may halt, so it always takes ω (§6). FG7 checks that a node of
class at least 1 consumes and returns ω. The flag is a plain `bool`. It is
not a use of ω. Carrying `go` is not what admits a pure count. The erratum
of 2026-11-13 keeps the carried order, and that is what admits it.

C7 keeps a carried value when either region uses it or either region
returns something other than that same input, and it keeps the order of a
count even when both regions only pass that order through. The break path
must return the literal false, not the incoming `go`. If both regions
return `go` unchanged and neither uses it, C7 removes the bool and the
early exit disappears. That removal is the rule for an unused carried
value other than a count's order. It is not O8. O8 recorded a carried
order that C7 used to remove (constraint B4, §5).

ω stays linear. Every path of a region that took ω returns it exactly once,
including the break path. The flag's own nodes do not read or write ω. If
they did, fuel and the order of effects would change (constraint B3, §5).

A count whose regions use ω for a reason other than the unused carrier,
and whose body break is the flag above, is the count item 3 can describe.
A pure count stayed `bound`, outside Dom, until C7 changed. The erratum of
2026-11-13 is that change. This note did not propose it.

## 5. A silent flag rewrite breaks syntactic identity

Rewriting a core `break` into a carried bool inside the Rust construction,
without a matching constructor in the Lean model, does not preserve
syntactic identity. The bytes of a canonical Form may still round-trip
inside the Rust slice. That is not a preimage under the model the identity
is stated in. The six constraints are the mathground sketch's. They are
why this note refuses to call that rewrite `construct(schedule(F)) = F`.

**B1.** `inductive Stmt` in `spec/lean/RCE/Form.lean` is `skip`, `assign`,
`seq`, `ifs`, `loop`, `fail` and `emit`. It has no `break`. A construction
that erases a break into a carried bool yields an Iterate term. No
constructor of this `Stmt` is the source of that term, so construct has no
preimage: the constructed term is a different term from the loop that was
offered. `brk` on `Stmt` in `spec/lean/RCE/Core/Driver.lean` is the run
driver for `fn main` (statements of a block, S6). It is not this `Stmt`,
and adding a driver constructor would not give Form's construct a preimage.

**B2.** `ExecR.loopBound` in `Form.lean` derives `.ref (.trap .loopBound)`
for `.loop 0` when the condition is true, before the body runs. An extra
test of `go` that is decided after that rule makes a loop that has already
broken take the bound trap. The flag is checked before the bound. The
schedule's order already does this for `k`. A second test that runs later
does not.

**B3.** If the break path, or the nodes that set the flag, read or write ω,
fuel and the order of operations change. ω orders effects and halts. A flag
that touches ω is a different computation, not a spelling of the same one.

**B4.** C7 drops `go` when it is only passed through, by the same rule that
drops an unused carried value other than the order of a count. The result
is a loop that cannot stop early. A pass-through bool is not an encoding
of break. The erratum of 2026-11-13 keeps a count's carried order and does
not keep a pass-through bool.

**B5.** A conditional increment leaves the measure shape. `find_measure`
requires `int.lt` (or the `int.le` spelling) of an unchanged index and end,
and a body that steps the index when it continues. A body that increments
only on some of the continuing paths is not that shape. The failure is a
measure refusal, not a silent rewrite into a count.

**B6.** `RCE.Iter.drive` returns `.done` when the iterator finishes,
`.stopped` when the consumer breaks, and `.bound` on the loop-bound trap.
`drive_break` needs those apart: a break delivers a prefix, and `.stopped`
is not `.done`. One bool has two values, so it cannot name three
outcomes. Under a flag alone, `.stopped` and `.done` become the same Iterate
result (carried values, flag false). `.bound` stays a Halt with the trap
text "loop bound" only when the flag is tested before the bound (B2). A
flag that is tested too late collapses `.stopped` into `.bound` as well.

## 6. Option A and option B

The bridge implements option B and does not state syntactic identity for a
restructured break. Option A is not implemented. A change that rewrites
breaks in Rust and still states syntactic identity has chosen neither, and
it is not this proposal.

**Option A. Give construct a preimage.** Add a break constructor, or an
exit sum with the three outcomes of `RCE.Iter.Out`, to `Stmt` in
`spec/lean/RCE/Form.lean` and to the Lean model of Iterate that construct
produces. The Form encoding stays five node kinds. The carried flag in the
two-region Iterate is the image of that constructor, not a bool invented
for a term the model cannot name. Syntactic identity can then be stated for
a Form whose break is that constructor: `construct(schedule(F)) = F` in
canonical bytes, with three-way eval as well. `.done`, `.stopped` and
`.bound` stay distinct, so B6 is answered by the sum rather than by a
collapse. B2 and B3 remain proof obligations on the new rules: the exit is
tested before `.loopBound`, and the flag does not touch ω. B4 remains: a
constructor that construct immediately turns into a pass-through carried
bool is still deleted by C7. B5 remains: the measure shape is unchanged.

Option A is a change to the Lean model and to the statement of identity.
It is not a change that can land as a comment in the Rust bridge alone.
Until those constructors exist, the identity claim does not cover a
restructured break.

**Option B. Weaken the claim.** Do not add a constructor. For a Form whose
schedule contains a break that item 3 restructured into a flag, stop
claiming syntactic identity. Claim semantic equivalence: three-way eval
agrees, and two further obligations hold.

- **O6'.** The flag is tested before the bound, so a break does not become
  `.loopBound` (B2).
- **O7'.** The operations that set or test the flag do not read or write ω,
  so fuel and the order of effects are the source's (B3).

O6' and O7' are additional obligations of option B. They are not a
renumbering of O1 to O10. Syntactic identity stays the claim on the part of
Dom that does not use this rewrite. A break-restructured Form is outside
that syntactic Dom. Option B states the collapse B6 names: `.stopped` and
`.done` are not distinguished by the Iterate's outputs. It does not call
that collapse `.bound`. A pass-through flag is still not a break (B4), and
a conditional increment is still not a measure (B5).

Option B is an honest weakening. It is not syntactic identity under another
name.

The schedule reads both arms of a branch from the same live set. One arm
runs, so both may use an owned capture. The parent keeps that capture only
when every arm that falls through still holds it. Leaving the first arm's
move in place made the second arm require change, including the branch a
body break becomes when the loop carries a list (`rt.dec_limbs`). The bytes
of that flag form, where they match, are the flag form against its
schedule. A plain branch that does not break stays inside syntactic
identity.

A function whose outputs are its halt node's outputs schedules as a halt.
Those outputs are the declared result and ω, not core values, so the core
does not return after the halt and does not drop what the halt still holds.
A loop region's epilogue stores the carried results. A tail that binds a
local the epilogue names stays in the region. Sinking that tail into an arm
would hide the local. Neither change is a new break shape, and neither
weakens FG8.

## 7. What a later implementation could unlock

Nothing in this section constructs today. The counts are the refusals PR
#145 records, not a new measurement, and they are not a floor.

PR #145 reports program-function refusals `break` 19, `catch` 9, `FG6` 3,
`defect` 2, lowering 4, and 21 with no program root. The `break` line is
the one item 3 is about. `catch`, `FG6` and `defect` are step A or a later
step, not this design.

`rt.dec_limbs` (`spec/checker/src/one/rt.rs`) writes the decimal text of a
list of base-2^16 digits. Its inner loop is the measure shape: `int.lt` of
the index and the length, otherwise `Break(1)` out of the loop block.
`find_measure` already looks for that head. Its outer loop is a count
(at most `5n + 1` iterations) whose body `break`s with `Break(1)` when the
limb list is empty. That body break is the exit `build_count` does not turn
into a flag. PR #145 names this function as where the `break` refusal
stops the closure.

`fn fib` in `spec/examples/03_fib_loop.rce` is a `for` over `0 .. n` with
no `break` in its own text. Checked addition in the lowered closure uses
the decimal runtime, so the closure includes `rt.dec_limbs`. PR #145 records
that this is why `fn fib` does not construct. Other program functions in
the 19 may be the same kind of closure, or a break in the function's own
body. This note does not split that 19 further.

After step A, an implementation of item 3 under option A or option B could
make `rt.dec_limbs` construct, and with it a closure such as `fn fib` whose
only `break` refusal was that function, provided no other tag still refuses
the closure and the outer count uses ω so O8 does not apply. The whole
example still would not construct: the observation driver writes `out` and
reads globals that are not part of ω, which is why the whole-module floor
is 0. Unlocking a function raises the program-function floor when the
round-trip holds. It does not lower the whole-module floor, and it does not
admit `fn fib` by editing the ratchet.

A break of depth greater than the loop block, a break out of a non-loop
block, and the exit count of ONE-15 for a break that leaves more than one
loop are outside item 3. `rt.dec_limbs` uses `Break(1)` because the loop
body is a block and depth 1 names the loop. That depth, naming the Iterate
under construction, is the break item 3 describes. A depth that names a
different construct is still refused.

## 8. Tests a later change would add

The bridge tests cover this list under option B. A byte comparison, where a
test has one, is the flag form against its schedule. It is not a comparison
of the break source with a flag-free term. Floors move only in the change
that makes a new syntactic case pass, and only upward. A break-restructured
example stays off the syntactic floor.

1. Leave `spec/checker/tests/form_bridge.rs` unedited until step B. Do not
   lower the 0-of-60 whole-example assertion on `main`, and do not lower
   the 2-of-60 program-function floor if PR #145 has merged it.
2. One core fixture, not a corpus edit: a count loop that uses ω in a
   region, with a `break` in the body that names the loop. Under option A,
   the Lean term contains the new constructor or the exit sum, and
   `construct(schedule(F))` equals F in canonical bytes. Under option B,
   the test does not assert those bytes against a flag-free source; it
   asserts three-way eval, O6' (a break before the last legal iteration
   does not halt with "loop bound") and O7' (fuel on the flag path equals
   fuel on the source path).
3. A pass-through bool that both regions return unchanged is removed by C7
   and is not accepted as a break (B4). The pure-count `bound` refusal
   stays (O8).
4. A body that increments its index only on some continuing paths is not a
   measure (B5). The refusal is not rewritten into a count.
5. A break out of a block that is not the loop, and a break that does not
   name the loop, still require change under the tag `break`.
6. No test of catch shapes changes. No test asserts that the hand emitters
   have been replaced, or that a production Form compiler exists.
7. A count loop that uses ω, carries a list, and breaks when the list is
   empty. The continue path pushes. Under option B the schedule consumes
   the list once, three-way eval agrees, and O6' and O7' hold. Byte
   equality of that flag form is not the claim when the canonical form
   differs; the corpus counts the difference as round-trip. A plain branch
   that moves one list on both arms, with no break, keeps syntactic identity.

## 9. What this note refuses

- Editing catch shapes, or reconciling `sum(tup(outputs…), halt)` with the
  core's `sum(ret, halt)`. That is a later step, and it changes a value the
  core evaluator returns.
- A pure-count rule change. O8 stands until a separate decision on C7 and
  FG7.
- A new node kind for break.
- Calling a silent flag rewrite syntactic identity.
- Lowering a corpus floor.
- Claiming a production compiler, in this note or in the implementation of
  item 3.
