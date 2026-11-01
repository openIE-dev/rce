# Proposal: laws, properties of every reachable state, decided exactly first

Status: open. The owner decided all eight questions of section 11 on
2026-11-01 ("proceed"); each decision is marked there and where it changes
the text. The first slice is implemented by the change on branch
`claude/laws` (ledger row of 2026-11-01): the `law` item of section 2
without `invariant` and `lean` lines, method D1, W-LAW-1, the law rows of
`rce change`, `rce law FILE`, and T63 proved in `spec/lean/RCE/Law.lean`.
It is specified in sections 3.2, 3.11 R10, 6.11, 7 and 10.11 of the
specification. D2a, D2b, D3 and W-LAW-2 to W-LAW-5 are proposals; section 7
lists what is built and what is not.

The proposal is derived from 6.9 (examples and checks as admission), 6.10
(behaviour and reach(P)), 10.11 (`rce dim`, `rce equiv`, `rce change`), T3
(fold law), T35 (the decidable predicate fragment), HC-27 (a log is bound
to its program), the attestation form of `stdlib/lib/ui/rules.rce`, and a
reading of Bend 2 (the language of Higher Order Company, read at
`bendlang/bend` commit `faeb02e` on 2026-11-01).

## Terms

- **s₀, δ, δ₁, λ, E**: as in 6.1 and 6.10. s₀ is the initial state; δ is the
  step function, with δ₁ its state part and λ its output part; E is the set
  of event instances.
- **reach(P)**: the states some event sequence leads to from s₀ (6.10).
  reach(P) is finite (6.3, N9) but may be very large.
- **Listable type**: a payload field type whose values the checker can list
  in full: `bool`, `()`, `u8`, `i8`, `T?` of a listable T, a `kind` without
  type parameters whose variants carry only listable fields, and a declared
  type `type N = T` (refined or not, 5.1) of a listable T.
- **Complete alphabet**: the event instances of P when every payload field
  of every event is listable and no event has more than the alphabet bound
  of section 5 (4,096 instances). A program without payloads has a complete
  alphabet. `rce dim`'s explored alphabet (10.11) is a sample; the complete
  alphabet is the whole of E.
- **Law**: a new item that states a property of every state in reach(P), or
  of every step taken from such a state. Since every state a fold of events
  reaches is in reach(P) (T3 and the definition), a law over every reachable
  state is the same as a law over every event sequence, of every length.
- **Verdict**: the result of deciding one law: `holds`, `fails`,
  `attested` or `undecided`, together with the **method** that gave it:
  `exhaustive` (D1), and as proposed `invariant` (D2), `lean` (D3) or
  `attestation` (D4).
- **Counterexample**: the shortest event sequence after which a law fails.
  Ties are broken by the order of the alphabet, compared event by event
  from the start, so the sequence is unique.
- **Inductive invariant** (proposed, D2): a bool expression I over the state
  that holds at s₀ and holds after every step from any state where it holds.
- **Slice** (proposed, D2a): the state fields a law and its invariant read,
  together with every field that a guard or update of those fields reads,
  closed under that rule. The slice is a static fact of the module.
- **Cover**: the hash an attestation is bound to (decision 5): a SHA-256
  over the law's claim, with every named definition replaced by its
  definition hash (F.4.3), and over the definition hashes of every `state`,
  `event`, `error` and `step` and of every definition the claim names.
- **Attestation**: a line `attest "WHO;YYYY-MM-DD;COVER"` saying that a
  named person decided, on that date, that the law holds. It is the form of
  `ui.rules` (`attest-ID`) with the cover added.

## In plain words

1. Examples state finitely many event sequences, and W-INT-2 already reports
   that they constrain nothing else. `rce equiv` and `rce change` compare
   outputs, to a bound. Before this proposal nothing in RCE could say "for
   every sequence of events, this never happens". A law says it.
2. The checker decides a law exactly whenever it can, in a fixed order:
   explore every reachable state; failing that (proposed), check an
   invariant a person wrote, then require a Lean proof; failing that,
   require a named person's attestation. Exactly one method gives the
   verdict, and the report names it. The first slice builds the first and
   the last.
3. A law that fails requires change. The report prints the counterexample
   as an `example` line that can be pasted into the program. A law that no
   method decides also requires change. It never passes in silence.

## 1. The problem, measured

The flag game of the Bend 2 demos, written as an RCE program (section 8),
has two versions: a board that wraps, with walls on its far edges, and the
same board without those walls. With the tools as they were before this
change:

```
$ rce dim flag_walls.rce   ->  states explored 72,  bound reached: no
$ rce dim flag_wrap.rce    ->  states explored 177, bound reached: no
$ rce equiv flag_walls.rce flag_wrap.rce  ->  equivalent (exact), status 0
$ rce change flag_walls.rce flag_wrap.rce ->  erratum, status 0
```

Neither program emits a value or keeps a state, so both have one behaviour
(6.10). Removing the walls lets the player win in 9 events, yet the tools
called the change an erratum. The property the game exists for is about
state, and no RCE construct stated it. Bend 2 keeps a project's laws in
`LAWS.bend` and their proofs in `PROOF.bend`; it states this property as a
law and pays for it with a 485-line `PROOF.bend` (392 lines of code). This proposal states
it in three lines and decides it by exploring 72 states. With the first
slice, `rce change flag_walls.rce flag_wrap.rce` reports a meaning change
with status 1 and the 9-event sequence.

## 2. Syntax (section 3.2)

As built (3.2; LALR(1), 0 conflicts; the tree-sitter port carries the
same rules):

```
Item       = ... | LawDecl .
IntentLine = ExampleDecl | CheckDecl | LawDecl | STRING NEWLINE .
LawDecl    = "law" NAME NEWLINE INDENT { STRING NEWLINE } LawClaim { LawNote } DEDENT .
LawClaim   = "always" Expr NEWLINE
           | "every" "step" [ "on" Pattern ] [ "if" Expr ] "->" LawOutcome NEWLINE .
LawOutcome = Expr | "kept" [ ":" NAME ] | "applied" .
LawNote    = STRING NEWLINE | "attest" STRING NEWLINE .
```

- A law has exactly one claim, an `always` line or an `every step` line,
  which the grammar enforces. A STRING line is prose and changes nothing.
  Laws stand at top level and inside `intent` blocks (decision 8).
- Proposed, not built: an `invariant Expr` line (D2) and a `lean STRING`
  line (D3). Each would be a `LawNote` once its method exists; until then
  such a line is not in the grammar (W-SYN-1), and the report says the
  method is proposed.
- `always p`: p is a bool expression over the state, with fields in scope by
  name, as in the `-> e` form of an example (6.9).
- `every step on PAT if g -> o`: for every s in reach(P) and every event
  instance e that matches PAT, with g true at (s, e), the step δ(s, e) has
  outcome o. `kept` and `kept: X` mean that the event is kept (with error
  X). `applied` means that it applies. An expression o is a bool over the
  state after the step (the state before it when the event is kept, T8), in
  which `old(f)` names field f before the step (decision 6). Example: a
  Pong-style law reads `every step on Key(code: 27, down: true) -> kept: Quit`.
- p, g, o and I may call the module's pure functions (5.4). They may not
  call foreign functions (5.11). A trap while evaluating them makes the law
  `fails` at that state, with the trap named.
- `law` is a word of Appendix F's reserved list whose construct now exists
  (2.4): used as a name it requires change (W-NAME-2), and inside a `trait`
  (the trait laws of F.6.6) it stays reserved, not implemented (W-RES-1).
  `always`, `every`, `attest`, `applied` and a bare `kept` are words only at
  their place in a law line (3.11 R10), so no identifier loses its use.
- The `law` items are part of the canonical text, so they change the module
  hash (F.5.3, HC-27) and the program hash (10.8).

## 3. Decision order

The checker decides each law by the first method below that applies. It
records which method decided it and never runs a later method once one has
decided. The first slice builds D1 and D4; D2 and D3 are proposals.

**D1. Exhaustive reachable-state exploration** (exact; needs no proof;
built).
- When it applies: always. With a complete alphabet it can give `holds`;
  with an incomplete one it explores the listable events and can give only
  `fails` or no verdict.
- How it works: the checker explores reach(P) breadth first from s₀,
  applying every instance of the alphabet in its order. It evaluates the
  law at each new state (for `always`) or at each step (for `every step`).
  Exploration stops when every law has a violation, or when no new state
  appears (the exploration is *closed*), or at the state bound (section 5).
- Verdicts: a violation gives `fails`, with the counterexample. A violation
  is exact even when the bound is reached or the alphabet is incomplete,
  because the sequence exists. A closed exploration with no violation over
  a complete alphabet gives `holds`, for every event sequence of every
  length (T63). Since δ is total and deterministic (T1, T2), a closed
  exploration is all of reach(P).
- When the bound is reached first, or the alphabet is incomplete, D1 gives
  no verdict. The report states how far it got and why (section 5).

**D2. Inductive invariant** (exact on a stated fragment; proposed). This
requires an `invariant I` line. The checker decides three obligations:
(a) I(s₀);
(b) for every state s with I(s) and every event instance e, I(δ₁(s, e));
(c) for every state s with I(s), the law holds at s (for `always`), or
for every step from s (for `every step`).
Together these imply the law, by induction on the length of the sequence
(T3; proposed T64, section 7). Obligation (b) needs no case for a kept
event, because a kept event leaves the state unchanged (T8), so I still
holds. The checker would decide (b) and (c) only in one of two fragments:

- **D2a, finite slice (decided by evaluation).** The fields of the slice
  have listable types, the alphabet is complete, and the number of slice
  valuations times the alphabet size is within the bound. The checker
  enumerates every valuation of the slice, reachable or not, that satisfies
  I, and evaluates (b) and (c) on it with the reference evaluator. The
  fields outside the slice cannot change the slice's guards or updates, by
  the slice's closure rule. Any expression of the language is allowed in
  this fragment, because the checker evaluates it and never reasons
  symbolically. D2a covers a law about a few small fields of a program whose
  full state (a `List`, a `Map`, a counter of `u64`) is too large for D1.
  It may follow the first slice (decision 3).
- **D2b, symbolic fixed-width arithmetic.** Not in the first slice and
  with no external satisfiability solver (decision 3). The fragment would
  be: integers of fixed width with `+ - * / %`, comparisons, bit operations
  and RCE's trap rules; `bool`; kinds without payloads; `if` and `match` on
  those kinds; calls of non-recursive functions within the fragment; and
  loops whose bounds are constants. Each obligation is then a
  quantifier-free formula over fixed-width bit-vectors, a decidable theory.
  Any later proposal for D2b has to say how an "unsatisfiable" answer is
  checked inside `rce`.

D2 verdicts (proposed):
- All obligations decided true: `holds` (method `invariant`).
- (a) fails: the invariant is wrong at the start. The law requires change,
  and the report names I.
- (b) fails at (s, e): the report prints s and e as a *counterexample to
  induction*. s may be unreachable, so this is not a counterexample to the
  law: the person strengthens I.
- (c) fails at an s that D1 reached: the law `fails`, with the sequence
  that reaches s.

**D3. A Lean proof** (checked by Lean; proposed, decision 4). This would
require a `lean "NAME"` line, where NAME is a theorem name.
- The obligation: an `rce emit lean` target, added with this step, writes
  an obligation file holding the program's δ, s₀, the law's predicate, and
  the theorem statement `∀ w, law (foldl δ₁ s₀ w)` (or its step form), over
  the Form model of `RCE/Form.lean`.
- The check: the person (or any tool) writes the proof. It is checked with
  `lake build` under the Lean pinned in `spec/lean/lean-toolchain` (4.12.0,
  no Mathlib, no `sorry`, and `#print axioms` listing only propext,
  Quot.sound and Classical.choice). A Lean upgrade is a separate decision
  (decision 4).
- The record: a proof record `rce-law-proof/1` holds the module hash, the
  SHA-256 of the generated statement, the theorem name, the Lean version and
  the hash of the proof file. `rce check` accepts D3 only when the record's
  hashes match the current program. Because `rce` generates the statement,
  the proof cannot prove a statement other than the law.
- What it trusts: the emission of δ into Lean is not proved. It is held to
  the reference by trace equality: the Lean-emitted δ is run on the states
  D1 explored and compared with `rce trace`, as every other target is.

**D4. Attestation** (recorded; never `holds`; built). When no method gives
a verdict, an `attest "WHO;YYYY-MM-DD;COVER"` line whose COVER equals the
law's current cover gives the verdict `attested`, with WHO and the date.
- An attestation never overrides a `fails`, as in `ui.rules`, where "an
  attestation never turns an exact fails or undecided into a meet".
- An attestation is bound to the cover (decision 5): when a definition the
  law covers changes, the cover changes, the attestation lapses, and the law
  is `undecided` until someone attests again. A definition that no step,
  no state, no event and no claim reaches changes without lapsing it. The
  report of an undecided law prints the current cover.

**Otherwise**: the verdict is `undecided`, and the program requires change
(W-LAW-1, decision 1). There is no other method and no silent pass.

## 4. What "requires change" means for a law

One rule, W-LAW-1 (decision 1; section 7 of the specification), with the
method *requires change*: a program is admitted only when every law is
`holds` or `attested`. Laws join examples and checks in the gate of 6.9.
W-LAW-1 reports, with the rule and the line, as 10.2 does:

- **a law that fails**: the method, the alphabet, the size of the search,
  what failed, and the counterexample written as an `example` line, in the
  `EventRef` syntax of 3.2, with payload values as constructor arguments,
  indented to paste into an `intent`. The line states what the program does
  now, so it holds and can be pasted as it is to record the defect: for
  `always p` its outcome is `not p` (`q` when p is `not q`); for an
  `every step` law it is `kept: X` when the last step is kept with X, and
  otherwise the fields of the state after the last step with their values.
  Changing its outcome to the law's turns it into the regression example
  that the corrected program must meet. For the flag game, as built:

  ```
  requires change flag_wrap.rce: line 4: W-LAW-1 law you_cant_win fails after 9 events (exhaustive, alphabet 4 complete, 86 states explored): the claim is false [requires change]
      example Up, Up, Up, Up, Right, Right, Right, Right, Right -> won
  ```

  `rce test` on `flag_wrap.rce` without the law and with this line pasted
  reports `ok (3 examples, 0 checks)`. With `-> not won` it reports
  `W-INT-1 example does not hold`. With `-> not won` on `flag_walls.rce` it
  reports `ok`. `tests/law.rs` checks all three.
- **a law that is undecided and not attested**: why the method gave no
  verdict (alphabet incomplete, naming the event, the field and its type;
  bound reached, with the count; a lapsed attestation), and the cover an
  attestation must carry.
- **an `attest` line that is not `WHO;YYYY-MM-DD;COVER`**.

Proposed, not built: W-LAW-3 (an invariant that does not hold, with D2) and
W-LAW-5 (vacuity, a recommended change, never a requirement): a law that
also holds for P's lookup-table control (6.9), the program that keeps every
event not listed in the examples, would be reported as recommended: "a
program that does nothing obeys this law". Safety laws say what never
happens; the examples have to say what does happen. In the first slice a
module with a law gets no W-INT-2 recommendation, since its examples are no
longer its only obligations on the steps.

## 5. Bounds and how they are reported

- **LAW_FUEL**: the state bound of D1 (and, when built, the valuation bound
  of D2a). It is a constant of the language, beside LOOP_LIMIT and
  VALUE_LIMIT (6.3, 6.7), with the value 1,000,000 (decision 2). Admission
  then depends only on the program text (10.1, item 1).
- **Alphabet bound**: 4,096 instances per event, as in `rce dim`.
- **`rce law FILE`** (built) reports every law with LAW_FUEL. Exploring
  further with `fuel N` (positional, as `rce dim`) is not built; a result
  found above LAW_FUEL could enter admission only through a proof record
  (D3) or an attestation (D4).
- **Report**: every law gets one line with the method, the counts, and
  whether the search closed. As built:

  ```
  law you_cant_win        holds      exhaustive  72 states, 288 steps, alphabet 4 (complete), closed
  law you_cant_win        fails      exhaustive  after 9 events; 86 states explored; the claim is false
  law below_four_billion  undecided  exhaustive  bound reached at 1,000,000 states (LAW_FUEL); attest with "WHO;YYYY-MM-DD;…"
  law never_shrinks       attested   attestation  by A. Person on 2026-11-01 (exhaustive: alphabet incomplete: Deposit.amount (u64))
  ```

- **Exit status of `rce law`**: 0 when every law holds or is attested; 1
  when any law fails or requires change; 3 when the only obstacle is the
  bound (as with `rce equiv`, 10.1 item 4). A JSON form `rce-law/1` is not
  built.

## 6. Laws, examples, checks and `rce change`

- **Examples and checks.** An example is one sequence and its last step. A
  check is a comptime proposition about values and functions. A law covers
  every reachable state. All three are obligations of admission (W-INT-1,
  W-LAW-1). None of them is ever edited by `rce repair` (10.13); a law
  becomes one more thing a candidate is judged by.
- **`rce change OLD NEW`** (built). A law is part of the program's meaning.
  - **CC-L1**: each law of OLD is decided on NEW, as written, over NEW's
    definitions. If NEW breaks a law of OLD (decided `fails` on NEW), the
    verdict is *meaning change* with status 1 and the counterexample,
    whatever `rce equiv` says. On the flag pair this turns the former
    *erratum* (status 0) into *meaning change* with the 9-event sequence.
  - **CC-L2**: a law added, removed or edited is listed as a row like an
    `api-diff` row. An edited or removed law is a meaning change. A law
    added to unchanged code is an erratum (decision 7): the states and
    outputs are unchanged.
- **Logs (HC-27).** A log D replays only with the program whose module hash
  is `D.program`. A law that holds for that program therefore holds at every
  entry of every log that program wrote (T3, since every state of D is in
  reach(P)). This is a statement about recorded history, not only about
  sequences not yet run. An erratum under CC-L2 lets the operator rebind
  `D.program`, as HC-27 already allows.

## 7. What is specification and what is tools

**Specification (normative), built in the first slice:**
- the syntax of section 2 in 3.2, and R10 in 3.11;
- section 6.11, "Laws": the meaning of `always` and `every step`, the
  complete alphabet, D1, LAW_FUEL, the verdicts, the counterexample, the
  cover and the attestation form;
- W-LAW-1 in section 7, with its method (requires change) in the table of
  7.1;
- **T63**, exploration decides a law: with a complete alphabet, a closed
  exploration from s₀ on whose every state a property holds proves it for
  every event sequence, for `always` and `every step` claims; one sequence
  refutes it; and when every explored state is reached, the law holds
  exactly when it holds on the explored states. Proved on the Form of
  `RCE/Form.lean` in `RCE/Law.lean` (in `lake build` and the axiom audit).

**Proposed, not built:** D2 (with W-LAW-3 and T64, invariant soundness:
D2(a), (b) and (c) imply the law), D3 (with `rce-law-proof/1`), W-LAW-5,
liveness.

**Tools (informative):**
- built: `rce law FILE`; the law rows of `rce change`; the W-LAW-1 report of
  `rce check` and `rce test`, with the counterexample line;
- proposed: `rce law fuel N`, `rce-law/1` JSON, `rce emit lean` for the D3
  obligations, language server diagnostics that offer to insert the
  counterexample as an example, and laws as gates in `rce repair` (as
  admission, they are already part of the gate it calls).

None of these calls a model (10.1, item 2).

## 8. First test case: the flag game

The program. The wrapping version is the same text with `north` and `west`
set to `false`. Both are in `spec/checker/tests/law/`; the walled one also
states two `every step` laws.

```
intent "a flag in a sealed room on a board that wraps: winning is impossible"
    example Left, Left, Up -> not won
    example Up, Up, Up, Up -> y == 1
    law you_cant_win
        "no sequence of moves from the start reaches the flag"
        always not won

const W: u8 = 12
const H: u8 = 8
const ROOM: u8 = 3
const FLAG_X: u8 = 1
const FLAG_Y: u8 = 1

state Game
    x: u8 = 8
    y: u8 = 5
    won: bool = false

event Up
event Down
event Left
event Right

fn wall(x: u8, y: u8) -> bool
    let east = x == ROOM and y <= ROOM
    let south = x <= ROOM and y == ROOM
    let north = x <= ROOM and y == H - 1
    let west = x == W - 1 and y <= ROOM
    return east or south or north or west

fn up(y: u8) -> u8
    if y == 0
        return H - 1
    return y - 1

fn down(y: u8) -> u8
    if y + 1 == H
        return 0
    return y + 1

fn left(x: u8) -> u8
    if x == 0
        return W - 1
    return x - 1

fn right(x: u8) -> u8
    if x + 1 == W
        return 0
    return x + 1

fn on_flag(x: u8, y: u8) -> bool
    return x == FLAG_X and y == FLAG_Y

step Game on Up if not wall(x, up(y))
    y = up(y)
    won = won or on_flag(x, y)
step Game on Down if not wall(x, down(y))
    y = down(y)
    won = won or on_flag(x, y)
step Game on Left if not wall(left(x), y)
    x = left(x)
    won = won or on_flag(x, y)
step Game on Right if not wall(right(x), y)
    x = right(x)
    won = won or on_flag(x, y)
```

The geometry and the start cell are Bend's `demos/app_win_is_bug_2d`. A move
into a wall selects no step, so the region is unchanged and the event
applies (6.4, frame).

Acceptance, each a test in `spec/checker/tests/law.rs`, all passing with
the first slice:

1. **The wrapping board fails.** `rce check flag_wrap.rce` requires change
   under W-LAW-1, method exhaustive, with exactly the line
   `example Up, Up, Up, Up, Right, Right, Right, Right, Right -> won`;
   pasted, it holds.
2. **The walls fix it, with no proof.** `rce check flag_walls.rce` is `ok`,
   and `rce law` reports `holds exhaustive 72 states, 288 steps … closed`.
   The program has no invariant line, no Lean proof and no attestation.
   Bend's equivalent is 485 lines of PROOF.bend; RCE's is 0 lines.
3. **The change is a meaning change.**
   `rce change flag_walls.rce flag_wrap.rce` reports *meaning change* with
   status 1 (CC-L1). Without the laws it reports *erratum* with status 0.
4. **Determinism and names.** Two runs give the same bytes (10.1). The
   name-free projection, `rce fmt nameless`, gives the same verdict and the
   same counterexample up to names (T7).
5. **The bound and attestation.** A `u32` counter is undecided at
   1,000,000 states (status 3). A law over a `u64` payload is undecided, is
   admitted once attested with its cover, stays attested when an uncovered
   function changes, and lapses when a function its step calls changes.

Not yet a test: vacuity (W-LAW-5 is proposed). Next cases, in order of
method: a bank balance law over a `u64` total with a `List` audit field,
decided by D2a over the slice {total}; insertion sort, `always sorted(out)`
for a list of unbounded length, where D1 and D2a do not apply, so D3 is
required.

## 9. Where Bend 2 is ahead, and where RCE differs

**Bend 2 is ahead:**
- It proves laws over unbounded data (`for xs: List<Nat>`, `for x: Nat`)
  inside the language. RCE has no such statement until D3 exists, and D3
  needs Lean outside `rce`.
- Its claims are its proofs' types, in one language: no separate model and
  no emission step.
- Its checker is fast on explicit terms, and it has a Lean-formalized kernel
  (`--verdict`) behind it.
- It has a parallel CPU and GPU runtime, a package hub, and sixteen demos
  with laws.

**RCE differs:**
- **The model is the program.** An RCE program's δ is its meaning (6.4).
  A law is decided on the same δ that `rce run` and the host execute. There
  is no second model that can drift, as Cedar's Lean model can from its
  Rust implementation (Cedar is Amazon's authorization policy language),
  which `cedar-drt` exists to catch by differential testing. Bend shares this property for its own semantics.
- **Emitted targets are checked by trace equality.** Bend's README says
  laws "are proven about Bend's semantics. The backends can still differ,
  for example in F32 NaN". RCE holds every emitted program to the reference
  by trace equality (T9-L, 10.5, the conformance suite), so a law decided on
  δ carries to an emitted program to the extent those traces agree.
- **HC-27 binds a log to its program.** A law that holds is therefore a
  property of every recorded log of that program (section 6). Bend has no
  log.
- **Exact before proof.** Bend requires a proof term even when the domain
  is finite. Its flag proof is an enumeration certificate (`chk_all`)
  written by hand around the checker's evaluation. RCE runs the enumeration
  itself (D1), and only the residue would reach Lean or a person.
- **The claim cannot be moved.** In Bend, nothing checks that `LAWS.bend`
  is unchanged; a loophole of this kind was found and closed in commit
  `228aad0d`, where an editable `on_flag` let a law be proved vacuously. In
  RCE, a law is canonical text: an edited law changes the program hash and
  is a meaning change under CC-L2.
- **No model in the loop.** Bend's proving agent and its proof search tool
  (SupGen) are proprietary. RCE's D1 and the attestation check are deterministic
  procedures; only an attestation comes from outside, and it is recorded
  with its cover.

## 10. What it does not claim

- Liveness ("eventually"). Laws are safety properties: some bad state or
  step never occurs. On a closed finite graph, liveness is decidable too
  (by finding cycles), but it is left for a later proposal.
- Commands and concurrency. A `fn main` with tasks has reach over schedules
  (6.8). Laws here are over the event steps of state machines.
- Decision of arbitrary properties. D1 is finite, and everything else goes
  to an attestation (and, when proposed methods exist, to D2 or Lean).
  Rice's theorem stands.

## 11. Questions for the owner, and the decisions

The owner answered "proceed" on 2026-11-01, accepting these answers:

1. Admission or report: should a failed or undecided law make a program
   require change, or start as a recommended change?
   **Decided: it requires change, under the new rule W-LAW-1, as examples
   do.**
2. Where the bound lives: a language constant LAW_FUEL, or a bound line in
   each law? **Decided: LAW_FUEL, a fixed language constant (1,000,000), as
   drafted.**
3. D2b: may the tool chain use an external SAT (propositional
   satisfiability) solver whose "unsatisfiable" answers are checked inside
   `rce`, or should the first slice ship D1 and D2a only? **Decided: no
   external SAT solver in the first slice. It ships
   D1 (exhaustive exploration) only; D2a (finite enumeration) may follow.**
4. D3: should `rce emit lean` be a new target, and should Lean stay pinned
   at 4.12.0? **Decided: `rce emit lean` becomes a target with the Lean
   step, later; any Lean upgrade is a separate decision.**
5. Attestation lapse: bound to the definitions the law covers, or to the
   whole module hash? **Decided: an attestation lapses when the definitions
   it covers change; it is bound to their hashes (the cover).**
6. The spelling of the before-state in `every step` outcomes.
   **Decided: `old(f)`.**
7. CC-L2: is a law added to an unchanged δ an erratum or a meaning change?
   **Decided: an erratum.**
8. Laws inside `intent` blocks only, at top level only, or both?
   **Decided: both.**
