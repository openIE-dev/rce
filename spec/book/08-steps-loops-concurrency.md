## 6 (continued). Dynamic semantics: steps, loops and concurrency

This section defines the *step function* δ (delta), which takes a state and
an event and returns the next state and an outcome, then errors, hashing,
loop bounds, concurrency and how examples are checked. A *trap* is a defined
failure after which the event does not apply and the state is kept. Example: a loop
with `bound 10` that starts an 11th iteration traps with "loop bound".

### 6.4 The step function δ

For event e = (Eₗ, v):

1. For each region Rᵢ in declaration order, take the steps `step Rᵢ on p if g`
   in source order whose pattern p matches e and whose guard g evaluates to
   true in (Rᵢ's pre-state fields, bindings of p). The first such step is
   **the** step of Rᵢ for e. If there is none, Rᵢ is unchanged (frame).
   A guard that traps (overflow, division by zero, …) does not fall through
   to the next step: the state is kept, with that trap (change S1; this is
   what makes δ a total function rather than leaving the case undefined).
2. Every selected step runs on the **pre-state** s (ASM semantics, Form law 2):
   no step sees another region's update for the same event.
3. The body of the step of Rᵢ is evaluated with fields of Rᵢ bound as `var`s
   to rᵢ, the pattern bindings as `let`s. Its result is
   - `ok(σ′)`: rᵢ′ = the field values in σ′ (each checked against its type
     range and VALUE_LIMIT), outputs oᵢ = the values emitted, in order;
   - `fail(X, w)` or `trap(m)`: the step is **kept**.
4. If any selected step is kept, the whole event does not apply:
   δ(s, e) = (s, [Kept(X or trap(m), e)]), where X is the reason of the
   first kept region in declaration order. Atomicity: an event either
   applies to every region or to none.
5. Otherwise δ(s, e) = ((r₁′, …, rₖ′), o₁ ++ … ++ oₖ).

The arrow form `step R on p -> e` is sugar for the block
`f = e` where f is R's only field.

**Order independence.** Since each step reads only its own region and the
event, the result of steps 2–5 does not depend on the order regions are
evaluated except through "first kept region", which is fixed by
declaration order. Outputs are concatenated in declaration order.

### 6.5 Errors, traps and kept steps

A `fail X(…)` or a trap inside a step keeps the state: the state is
unchanged, the log records the event with outcome `kept(X)` and λ outputs
`Kept`. Nothing is thrown past a step; no event is dropped silently. In a
pure function called from comptime, a trap or `fail` is a compile error. In `main`, a
trap or unhandled failure ends the run `stopped`, with the trap or failure as its output.

### 6.6 Determinism of identity: name-free hashing

Channel ids, actor ids and task ids are not counters. A channel id is the
SHA-256 of the canonical JSON of (creating construct's name-free key,
creating task id, per-task creation index), where the name-free key replaces
identifiers by their De Bruijn position (Form law 2). An actor id and a task
id are stable paths (C.11.11): an actor is keyed by its supervision position
`i.j` (supervisor i in declaration order, its child j) or `-` when no
supervisor declares its type, the creating task id, and its ordinal among
that task's creations at that position; a task by its parent's id and its
ordinal among the parent's spawns; `fn main` is the root. No key contains an
identifier, so renaming every identifier in a program leaves every id, every
log hash and every select decision unchanged (T7); no actor or task key
contains a source position, so an edit that keeps the supervision tree and
the order of creations keeps every actor and task id.

**The concept record (informative).** Names carry no meaning in the Form
(T7) and most of the meaning for a reader: obscuring identifiers costs a
model between a tenth and a quarter of its accuracy on what a program does
(arXiv 2510.03178, 2025), while removing formatting costs nothing
measurable (arXiv 2508.13666, 2025; both in section 11). So names have a
defined home outside the hash. The *concept record* of a module is the
table from each name-free key of 6.6 to: the name as written, the `intent`
strings of the item, and its doc comments. `rce emit form` prints it
under a `names` field after the hashed fields, keyed `#i` by declaration
index (the same index `shape_ctl` of 6.10 uses), with the `intent` strings;
doc comments are not yet collected. The Form JSON of 10.5 still carries
names in its `name` and `region` fields. It is never read by δ.
Two modules that differ only in it have the same outcomes, outputs, ids
and log hashes (6.1 hashes the positional forms, T7; the finding of
2026-10-22 that the hashes covered names is resolved, see C.15). A
projection (10.5) renders the concept record's names by
default; `rce fmt nameless` (10.6) is the projection that renders
positional names instead, which is the text the name-free keys of 6.6 are
computed from.

### 6.7 Loops and fuel

`for p in e` iterates over a finite value (list, set, map in canonical order
of its keys, as in 7.1, never insertion order (an erratum, C.15),
text by scalar value, or an integer range `a..b` / `a..=b`); its length is
fixed when the loop starts, and mutating the iterated collection inside the
loop requires change statically (it is borrowed). `while c bound n` and
`loop bound n` run at most n iterations, n a comptime-or-runtime u64
evaluated once; the default bound is LOOP_LIMIT = 1,000,000. A program may
set its own default with a module-level `const LOOP_LIMIT: u64 = n`, n a
positive literal (draft 0.5; any other declaration of that name requires change,
W-LOOP-1). Starting iteration n + 1 is `trap("loop bound")`. Comptime
evaluation uses the same bounds. `for x in f(…)` over an iterator function
(5.14) counts yields against LOOP_LIMIT in the same way; Lean:
`RCE/Iter.lean` (`drive_total`, `drive_break`).

**Structural recursion (draft 0.4).** A function may call itself when
every self-call passes, in one parameter position i fixed for the
function, a name bound inside a constructor pattern of a `match` whose
scrutinee is parameter i (or a name already known to be such a
component). The argument is then a strict component of the value
received, so along any chain of calls the value at position i strictly
shrinks and the depth of recursion is at most the number of constructor
nodes in that value, which N9 bounds. A name that is rebound by `let`,
`var`, assignment or a `for` pattern loses this status. The checkers of
drafts 0.4 to 0.7 let the position differ from call to call; that admits
a function that never returns (`f(x, y)` whose one self-call takes a
component of x and a fixed value for y, and whose other takes a fixed
value for x and a component of y, runs forever from `f(Leaf, Node(…))`),
so such a function requires change (W-REC-1; an erratum, C.15). Every function
this paragraph admits is admitted by the rule below with one layer.

**Recursion by layered measures.** Every cycle of the call graph (a
strongly connected component: one function that calls itself, or two or
more functions; trait methods excluded) is admitted when its calls can be
done in layers. A layer gives each function f of the cycle a set P f of
parameter positions and the *summed measure* μ_f(ā) = Σ_{i ∈ P f}
size(aᵢ), with size the number of constructor nodes. A call g(b̄)
written in f *keeps* the measure when the positions P g map one to one
into P f so that the argument at each j ∈ P g is the parameter at its
image i (unchanged, the name not rebound) or a component of it in the
sense of structural recursion above; it *shrinks* the measure when at
least one of them is a component. Layer 1 must be kept by every call of
the cycle and shrunk by at least one; the calls it shrinks are done.
Layer 2 must be kept by every call not yet done and shrunk by at least
one of them, and so on. The cycle is admitted when the calls left after
some layer k form no cycle among the functions. Then the tuple (μ₁, …,
μ_k, rank of the function) falls lexicographically at every call: a call
done at layer j keeps μ₁ … μ_{j−1} and shrinks μ_j, and a call left over
keeps every μ and lowers the rank. The order is well founded, so every
chain of calls is finite; with one layer a chain inside the cycle has at
most (μ + 1)·k calls, with μ the measure of the first call and k the
number of functions in the cycle, and N9 bounds μ. The reference checker
tries every choice of the sets of a layer, in a fixed order, for a cycle
with at most 16 parameters in total, takes the first choice that does
the most calls, and reports that a larger cycle that needs a layer requires change. Lean:
`RCE/Mutual.lean` (`call_dec`, `call_keep`, `layer_dec`, `wf_total`,
`wf_fuel_irrelevant`; one layer: `lex_dec`, `mutual_total`,
`mutual_fuel_irrelevant`, `mutual_depth`). Examples: `even(xs)` and
`odd(xs)` that each call the other on the tail of a matched `Cons`
(one layer); `h(m, n)` that calls `h(m', n)` and `h(m, n')` on
components m' of m and n' of n (layer {m} does the first call, layer {n}
the second).

**Declared bounds.** `fn f(…) -> τ bound n`, n a positive integer
literal, allows at most n active calls of f at once: a call of f, once
its arguments are evaluated and fitted to the parameter types, traps
with `trap("recursion bound")` when n calls of f are already active, and
otherwise counts as active until it returns or fails. A cycle that no
layers admit is admitted when every cycle of calls inside it passes
through a function declared with `bound` (the calls between functions
without one form no cycle). Then the slack Σ (n_g − active calls of g)
over the bounded functions falls at every call of a bounded function and
the rank falls at every other call, so every run of the cycle ends, by
its value or by the trap, within (slack + 1)·k nested calls. This covers
recursion on integers or indices, which has no constructor measure
(`gcd(a, b)` calling `gcd(b, a % b)`, a parser that recurses on a
position). The counts belong to the task (6.8) that makes the calls, so
`par` and `for … par` inside a function of a cycle with a bounded
function require change (W-REC-1): the branches could otherwise count calls
differently on one thread and on several. `bound` on an iterator
function (5.14), a foreign function or a trait signature is W-REC-1. The
check costs nothing (C.6.4 is unchanged). Lean: `RCE/Mutual.lean`
(`bump_slack`, `bound_dec`, with `lex_dec`, `mutual_total`,
`mutual_depth`); tests: `spec/checker/tests/bound.rs`.

Any other cycle in the call graph requires change (W-REC-1).

This is what makes δ total (T2): every construct terminates in a number of
steps bounded by a function of the program and the event.

### 6.8 Concurrency: tasks, channels, select, actors

Concurrency is defined by a small-step semantics over configurations
C = (T, K, A, L): a finite map T of tasks (id ↦ continuation), channels
K (id ↦ capacity, FIFO buffer, closed flag, held-send queue), actors A
(id ↦ region state, mailbox), and the log L.

Scheduling is a **tick**: tasks are ordered by id (6.6) and each runnable task
takes exactly one *communication step*: it runs pure code until its next
channel operation, spawn, `select`, emit or termination, and then performs
that operation. All operations of a tick are staged and delivered after every
task has moved (Form law 2: every update reads the pre-state). A tick is
logged as `Tick{n}` plus one entry per communication action, with payloads
(Form law 4).

```
  c has space                     ─→ v appended to c.buffer; task continues
  c full                          ─→ (c, v) held in c.held in arrival order;
                                     task blocked; Backpressure logged (no drop, law 3)
  <- c, buffer nonempty           ─→ first value; held send released in order
  <- c, empty and closed          ─→ none
  <- c, empty and open            ─→ task blocked
  spawn f(ā)                      ─→ new task id h(…); args moved in
  select: ready arms R ≠ ∅        ─→ execution k of this select statement in
                                     this task scans the arms cyclically from
                                     arm k mod n (n arms, source order) and
                                     takes the first ready one (draft 0.5)
  select: R = ∅, default present  ─→ default arm
  select: R = ∅, `after n`        ─→ blocked; after n ticks with R still ∅, after-arm
```

An actor is a region with a mailbox; `a <- M(v)` is an event `M(v)` admitted
to a's region at the next tick; the actor runs its handlers by 6.4 with its
own region as S. Each actor processes at most one message per tick, in
mailbox order.

**Fairness (draft 0.5).** The counter k is per task and per `select`
statement, starts at 0 and grows by one per execution. An arm that is ready
at each of n consecutive executions of its `select` is taken by one of them
(bounded wait). Draft 0.4 always took the minimal arm, so a busy first
channel could starve the others, the problem Go solves with a random choice;
the rotation gives the same guarantee without randomness. Lean:
`RCE/Select.lean` (`select_bounded_wait`).

Because the order of every choice is a function of ids, payloads and
per-select counters, and
ids are content hashes, the entire run of a program is a function of its
inputs: replaying the log reproduces it (T1, T3). The host may run tasks in
parallel provided the observable result equals the tick semantics.

Deadlock: a tick in which no task can move and no input is pending ends the
run with outcome `Deadlock` in λ (a defined outcome, not a hang).

### 6.9 Examples and checks

`example ē -> o` denotes the proposition: folding the events ē from s₀ gives
a final step whose outcome is o. The fold is the one of `rce run` and of
the host (HC-21): a step that keeps the state leaves it unchanged (T8) and
the next event is applied to that state, so an earlier event of ē may be
kept or applied and the proposition is about the last step and the state
after it. For `-> kept: X` the last step is kept with the error X, for
`-> kept: trap("K")` the last step is kept by the trap K, and for `-> e`
the last step is applied and then

- if the program has exactly one region with exactly one field f, the
  value of f equals ⟦e⟧;
- otherwise e must be a `bool` expression over the final state, with fields
  in scope by name (or `R.f` when ambiguous), and must be true.

Example: the counter `spec/examples/01_counter.rce` states
`example Dec, Inc -> 1`: `Dec` is kept with `Underflow`, the state stays
s₀, and `Inc` applies to it. Until the change of 2026-10-31 (ledger), an example held
only when every event before the last was applied, and `rce run` and
`rce trace` ended at the first kept step; the examples that held then hold
now, with the same traces, and an example whose earlier event is kept is
now a statement about the fold that goes on past it. An author who wants an
earlier step stated writes the prefix as an example of its own.

`check e` (in an intent or at comptime) is the proposition ⟦e⟧ = true,
evaluated at comptime. A program is admitted only if every example and check
holds (W-INT-1): the gate is part of the language, not of a test harness.
Derivation of the trap form: a step ends in a value, a named error or a trap
(T2), so a proposition about one final step needs all three outcomes, or a
program whose specified behavior is a trap has no example that states it and
no emitted program can be held to it.

**The lookup-table control (W-INT-2).** The examples of a module are
finitely many event lists folded from s₀, so they say nothing about any
other list. The *lookup-table control* of a module M is M with one more
field per region, the region's position in the trie of the example event
lists (each list restricted to the events the region has a step for and
that M applies when it folds the list): a step that M keeps, the control
keeps with the same error; each step that M applies, the control applies
only at a position that has a child for its event and moves the position
to that child (where M selects no step of the region, the control only
moves the position), and an event that M applies at a position with no
child for it is kept with an error of its own. The control equals M on every prefix of every example
list, so it meets every example of M, and it keeps the state on every other
list. A `check` is a proposition about comptime values and functions, and a
contract (F.6.2) is a proposition about a function, and the control keeps both unchanged, so neither tells M from its control. Hence, when a module has a step and example lines, `rce check`
admits it and reports the recommended change W-INT-2 (10.2): its
obligations do not constrain behaviour beyond the listed examples. A module
with a law (6.11) has an obligation beyond them, and W-INT-2 is not reported
for it. W-INT-2
never makes a program require change; it says what the examples do not
state.

### 6.10 Behaviour, equivalence and dimension

This section defines what "the same behaviour" means for two programs, and
the dimension of a program: the number of behaviours it can be in. Example:
a counter with a field that no step reads has the dimension of the counter
without it.

In plain words: (1) the behaviour of a program is the answer it gives to
every sequence of events; two programs are the same when their answers are
the same, and that is the only sense of "same" the Form needs; (2) two
different behaviours disagree at some first step, and a test that checks
the first k steps shows the programs are within 2^-(k+1) of each other,
not that they are equal; (3) the dimension of a program is the number of
different behaviours it can be in, and state that cannot change any future
answer is redundancy the checker can count.

**Behaviour.** Fix a program P with state space S, event alphabet E and
δ : S × E → S × O\* (6.1), δ₁ = π₁ ∘ δ, λ = π₂ ∘ δ. The *behaviour* of a
state s is the function beh_P(s) : E\* → (O\*)\* with beh_P(s)(ε) = ε and
beh_P(s)(e · w) = λ(s, e) · beh_P(δ₁(s, e))(w): the sequence of per-step
outputs along any event sequence. The behaviour of P is beh_P(s₀).
Derivation: (S, ⟨λ, δ₁⟩) is a coalgebra of the functor F(X) = (O\* × X)^E;
F has a final coalgebra Beh whose carrier is the set of such functions, and
beh_P is the unique coalgebra morphism S → Beh (Rutten 2000, Theorem 2.3).
Every Form, in every fit language, maps into the same Beh by the same
construction, so Beh is the one object every projection targets.

**Equivalence.** Two fits F and G are *equivalent*, F ≡ G, when
beh_F(s₀^F) = beh_G(s₀^G). A relation R ⊆ S_F × S_G is a *bisimulation*
when s R t implies, for every e ∈ E, λ_F(s, e) = λ_G(t, e) and
δ₁^F(s, e) R δ₁^G(t, e). T58 says ≡ is bisimilarity, so a proof of ≡
exhibits a bisimulation, and a proof of ≢ exhibits a distinguishing event
sequence.

**Distance.** For behaviours b₁, b₂ let n(b₁, b₂) be the length of the
shortest event sequence on which they differ (∞ when equal) and
d(b₁, b₂) = 2^-n. This is the prefix ultrametric (de Bakker and Zucker 1982):
complete and totally disconnected. Trace agreement on every sequence of
length ≤ k (the evidence of T9) certifies d ≤ 2^-(k+1), a ball around the
behaviour and not the behaviour; trace evidence therefore never enters a
hash or a key.

**Dimension.** reach(P) = {s : ∃ w ∈ E\*, s = foldl(δ₁, s₀, w)} is finite
(6.3 N9). dim(P) = |{beh_P(s) : s ∈ reach(P)}| is the number of distinct
behaviours P can be in; log₂ dim(P) is the bits of state the behaviour
needs, and log₂ |reach(P)| − log₂ dim(P) is the *redundancy* of the fit,
the bits of state that change no future answer. dim(P) is the number of
states of the minimal Mealy machine with behaviour beh_P(s₀) (the Nerode
index; Nerode 1958, Moore 1956, Hopcroft 1971), unique up to isomorphism
because δ is total and deterministic (T1, T2); an incompletely specified
machine has no unique minimum (Pfleeger 1973).

**Control dimension.** Let S = Σc × Σd (6.1). The *control projection*
P_c replaces every guard over Σd by its predicate symbol and every data
update by a symbolic update, giving a symbolic finite automaton over the
guard predicates of 6.4 (D'Antoni and Veanes 2014). dim_c(P) is the Nerode
index of P_c modulo predicate equivalence; dim_c(P) ≤ dim(P). The symbolic
quotient is not implemented; `rce dim control` (10.11) computes the
control quotient of the explored concrete machine instead.

**Canonical control form.** Let M_c be the minimal machine of P_c, its
states numbered by first visit in a breadth-first traversal from s₀ taking
events in declaration order (the name-free key of 6.6) and payloads in the
order of their columns in the explored alphabet of 10.11 (Reis, Moreira and Almeida 2005: the
canonical string of an initially connected DFA is unique and name-free by
construction). `shape_ctl(P)` is the SHA-256 over the canonical JSON (6.1.1)
of that string. `shape_ctl` equal means the same control behaviour up to
predicate equivalence; ≡ means the same behaviour, decidable for finite S
by partition refinement on reach(P) and in practice bounded by the
exploration fuel of `rce dim` (10.11). The proposal `form-ir-2.md` names
two further keys, a name-free term hash and an e-class key, that this text
does not define.

### 6.11 Laws

A *law* states a property of every state a program reaches, or of every
step taken from such a state, and the checker decides it before the
program is admitted. Example: the flag game of `spec/proposals/laws.md`
states `law you_cant_win` with the claim `always not won`; with walls on
the far edges of its board the checker explores its 72 reachable states and
the law holds; without them it fails, and the report prints the 9 events
that reach the flag as an `example` line.

In plain words: (1) an example states one sequence of events, and a law
states something about every sequence, of every length; (2) the checker
decides a law by visiting every state the program can reach, which is exact
when it finishes and when it finds a violation; (3) a law it cannot decide
within a fixed bound requires change, unless a named person attests it.

**Meaning.** Let s₀, δ, δ₁ and E be as in 6.1 and 6.10, and reach(P) the
states that some sequence of events leads to from s₀. A law has one claim
(3.2):

- `always p` holds when ⟦p⟧ = true at every s ∈ reach(P). p is a `bool`
  expression over the state, with fields in scope by name, as in the `-> e`
  of an example (6.9).
- `every step [on PAT] [if g] -> o` holds when, for every s ∈ reach(P) and
  every event instance e that matches PAT (every event, without `on`) and
  for which g is true at s, the step δ(s, e) has the outcome o. PAT is a
  step pattern (3.5) and binds its names for g and o; g is a `bool` over s.
  The outcome `kept` says the event is kept, `kept: X` that it is kept with
  the error X, and `applied` that it applies. An expression o is a `bool`
  over δ₁(s, e), the state after the step (s itself when the event is kept,
  T8), in which `old(f)` names the field f of s.

p, g and o may call the module's functions; they are checked as comptime
expressions (5.4), so they call no foreign function. A trap while
evaluating one of them is a violation at that state or step. By T3 every
state a fold of events reaches is in reach(P), so a law over reach(P) is a
law over every event sequence.

**The complete alphabet.** A payload field type is *listable* when the
checker can list its values in full: `bool`, `()`, `u8`, `i8`, `T?` of a
listable T, a `kind` without type parameters whose variants carry only
listable fields, and a type declared `type N = T` or `type N = T where p`
(5.1) of a listable T (a value outside p traps at construction and is no
event). The *alphabet* of a program is every instance of every event whose
fields are all listable, events in declaration order (the name-free key of
6.6) and the values of each field in their order: `false` before `true`,
integers ascending, `none` before `some`, variants in declaration order,
the fields of a payload varying last field fastest. An event with more than
4,096 instances, or with a field that is not listable, is left out, and
the alphabet is then *incomplete*.

**Method D1, exhaustive exploration.** The checker explores reach(P)
breadth first from s₀, applying every instance of the alphabet in its
order to each state, and keys a state by its canonical JSON (6.1.1). It
evaluates every `always` law at each new state, s₀ first, and every
`every step` law at each step. It stops when every law has a violation,
when no new state appears (the exploration is *closed*), or when a new
state would be the (LAW_FUEL + 1)th. **LAW_FUEL** is a constant of the
language, 1,000,000 states, so admission depends only on the program text
(10.1, item 1), as with LOOP_LIMIT (6.7).

**Verdicts.** Each law gets one verdict:

- `holds`: the exploration closed with no violation and the alphabet is
  complete. By T63 the law then holds after every event sequence of every
  length.
- `fails`: a violation was found. The sequence that reaches it exists, so
  the verdict is exact whether or not the bound was reached and whether or
  not the alphabet is complete.
- `undecided`: no violation, and the alphabet is incomplete or the bound was
  reached. The report names each event left out (`Deposit.amount (u64)`) and
  the bound.
- `attested`: undecided, and attested (below).

A program is admitted only when every law holds or is attested (W-INT-1
for examples and checks, W-LAW-1 for laws). An attestation never changes a
`fails`.

**The counterexample.** The sequence a failing law reports is the
shortest one after which the law fails, and among the shortest the first in
the order of the alphabet, compared event by event from the start: a
breadth-first exploration that takes events in a fixed order meets the
states in that order. It is printed as an `example` line (3.2) that states
what the program does now, so it holds and can be pasted into an `intent`
as it is: for `always p`, the outcome is `not p` (`q` when p is `not q`;
the field's value when the state has one field of type `bool`); for an
`every step` law, `kept: X` when the last step is kept with X, else the
fields of the state after the last step with their values
(`x == 1 and won == true`). Changing the outcome to the law's turns the
line into the example that the corrected program must meet. For the flag
game without walls:

```
requires change flag_wrap.rce: line 4: W-LAW-1 law you_cant_win fails after 9 events (exhaustive, alphabet 4 complete, 86 states explored): the claim is false [requires change]
    example Up, Up, Up, Up, Right, Right, Right, Right, Right -> won
```

**Attestation.** A line `attest "WHO;YYYY-MM-DD;COVER"` in a law says that
the person WHO decided on that date that the law holds; it is the form of
`ui.rules` (`attest-ID`) with the cover added. The *cover* of a law is a
SHA-256 over the law's claim, with every named definition replaced by its
definition hash (F.4.3), and over the sorted definition hashes of every
`state`, `event` and `error`, of every `step` (by its text, with the
definitions it names replaced by their hashes) and of every definition the
claim names. An attestation applies only while its COVER equals the
current cover: when a definition it covers changes, the attestation
*lapses* and the law is `undecided` again. A definition that no step, no
state, no event and no claim reaches can change without lapsing it. The
report of an undecided law prints the current cover. An `attest` line that
is not of this form requires change (W-LAW-1).

**Reports.** `rce check` reports each law that fails or is undecided and
not attested under W-LAW-1, with the counterexample line. `rce law FILE`
(10.11) prints one line per law with its verdict, the method and the counts,
for example
`law you_cant_win  holds      exhaustive  72 states, 288 steps, alphabet 4 (complete), closed`;
its status is 0 when every law holds or is attested, 3 when the only
obstacle is the bound, and 1 otherwise. `rce change OLD NEW` decides every
law of OLD on NEW (10.11).

Laws are part of the canonical text, so they are in the program hash
(10.8). The other methods of the proposal (an inductive invariant, a Lean
proof) are not part of the language; a law that states liveness
("eventually") or a property of `fn main` and its tasks is not expressible.

### 6.12 Mutation adequacy

An example, a check or a law carries information about a program only to
the extent that it can fail when the program changes. If no single edit of
the program makes an obligation fail, the obligation tells the program
apart from none of its edits, and within them it carries zero bits about
the program's behaviour (proposal `spec/proposals/mutation-adequacy.md`).
Example: in the fixture `spec/checker/tests/mutants/vacuous.rce`,
`example Reset -> n >= 0` holds whatever `Reset` sets `n` to, since `n` is
a `u8`, and `rce check` refuses it under W-MUT-1.

In plain words: (1) the checker makes every single edit of the program and
runs the obligations on each; (2) an obligation that fails on none of them
requires change; (3) an edit that no obligation catches is reported with
its diff, as a recommendation.

**Obligations, definitions and edits.** An *obligation* is an `example`, a
`check` or a `law` item of the program's own text; a *definition* is a
function, a method, a step, an actor, a constant or a state declaration of
it (10.13), not of an imported module. A *single edit* is one edit of rung d
of `rce repair` (10.13) at one expression of a definition. A *mutant* is
the program with one single edit; the *mutant set* is every mutant in the
order of rung d (definitions in source order, expressions in preorder,
edits in rung d's order), of which the first **MUTANT_BOUND** = 2,000 are
judged. MUTANT_BOUND is a constant of the language, like LAW_FUEL (6.11),
so admission depends only on the program text. Obligations and the
`check`, `requires` and `ensures` statements of a body are never edited.

**Judging a mutant.** A mutant is *in Form* when the checker gives it the
reports of the program and its constants and initial state evaluate; one
that is not is counted apart. An obligation *kills* an in-Form mutant when
it fails on it. An obligation runs on a mutant only when the mutant's
definition is in its *reach*: for a check, the functions and constants it
names and what they name; for an example or a law, every step, actor and
state declaration and what they name. A run has the fuel the obligation
took on the program, twice, plus 100,000 (at most the fuel of 6.7); a law
has four times the states it explored (at least 10,000, at most LAW_FUEL).
A run that ends by exhausting that budget below the fuel of 6.7 is
*undecided* and is neither a kill nor a hold. A mutant that no obligation
kills is *equivalent* when `rce equiv` (10.11) judges it equivalent to the
program with both explorations closed, and a *survivor* otherwise; a killed
mutant is never equivalent, since the obligation that fails on it has
observed a difference. The *score* of a program is its killed mutants
divided by its killed mutants and survivors.

**W-MUT-1 (vacuous obligation, requires change).** A *measured* obligation
that holds, decided, on every in-Form mutant of its reach. An obligation is
measured when it names a definition of the program or of a module it
imports; when its reach is within MUTANT_BOUND; when at least
one in-Form mutant edits an expression that its run on the program
evaluated, in a definition of its *core* (for a check its reach, for an
example the steps its events can select and what they name, without
constants and state declarations, and for an example not inside an `emit`,
since an example states the outcome and the state, never an output), or
for a law any mutant of a step or of
what the steps and the claim name; and when its reach's mutants times its
fuel on the program plus 10,000 are at most the fuel of 6.7 (for a law, its
reach's mutants times the steps of its exploration at most 1,000,000). An
obligation that is not measured is not refused: the edits cannot see what
it runs (`check 1 + 1 == 2` is a statement about the language, not about
the program), or deciding it would cost more than the bound.

**W-MUT-2 (surviving mutant, recommended).** A survivor. `rce mutants`
(10.14) prints each one with its edit, its diff on the canonical text and,
where `rce equiv` finds one, the shortest event sequence after which the
program and the mutant give different outputs.

### 6.13 Graph terms and tensor values

A step's numeric body may be a *graph term*: a FunctionGraph, the graph
form of the crate `rce-graph` (nodes, wires between ports, one output
node), applied to tensors. A graph term is part of δ like any other
expression: it reads its operands, gives one value, and that value is
written to the state or emitted by the ordinary statements of 6.4. Nothing
about S, E, D or λ changes. This section defines the terms, the one
library module that writes them, their meaning, the canonical form of
their values and the rules that guard them. Example: in
`compute/examples/dense.rce` the step body `y = tensor.relu(tensor.matvec(w, x) + b)`
is one graph term of five operation nodes, and `rce trace` gives the same
state hashes on every run; `compute/examples/block.rce` is a causal
attention block over embeddings, with two heads, RMSNorm and a SiLU
feed-forward layer, in eight terms.

**GT-1 (tensors).** A *tensor* is a nested fixed-length array of `f32`,
`f64`, `i8` or `i32` (5.1): `[[f64; 3]; 2]` is a tensor of *element type*
`f64` and *shape* (2, 3), the lengths from the outside in. Every length is
at least 1. A tensor is an ordinary value of 6.1, so no type is added. A
*scalar* of a term is a value of the element type, the result of an
operation that reduces a tensor to one element (`tensor.sum`). Integer
tensors hold token ids and the quantized values of block formats; their
arithmetic is RCE's integer arithmetic (6.3 N2).

**GT-2 (the term and its key).** `use std.tensor` makes the built-in
module `std.tensor` visible as `tensor` (4.1). A call `tensor.op(...)` is
the *root* of a graph term: the term is the maximal expression made of
`tensor` calls and lane-wise `+ - * /` between tensors under the root;
every other subexpression under it is an *operand*. The term is a
FunctionGraph whose parameters `p0`, `p1`, ... are the operands in the order
a reader meets them (left to right, depth first), together with its
*typing*: the element type and shape of each parameter and of the result,
and the level (GT-3) of each operation node. Its *key* is the BLAKE3
(a cryptographic hash with a 256-bit digest) content hash of its `.fg` file: a
`Graph` section holding the graph and a `TypeTable` section holding the
typing, each in CBOR (Concise Binary Object Representation, RFC 8949), hashed as the format of
`compute/crates/rce-format` hashes every section. A term holds no
declared name and no tensor bytes: its data enter only as operands (state
fields, payload fields, constants). So the key is name-free (T7) and
covers the whole body. In the core `rce-core/1` (10.12), the statement
`vN = graph K(a...)` applies the term with key K, which the module declares
in its `graphs` table; the key of the group that holds the statement (W5)
covers K, and the program hash (CC-1, 10.8) covers the key of every term of
the program.

**GT-3 (levels).** Every operation node carries one of three *levels*.
They are the levels of the RCE Compile study's experiment E3
(`compile/CHARTER.md` 4.3), and each states what a *realisation* of the
node (another backend, an emitted kernel, a fusion) must reproduce of the
reference's result (GT-6):

- `exact`: the reference's bits;
- `within N`: each element within N units in the last place (ULP) of its
  element type of the reference's element, N from 1 to 2^31 − 1;
- `lossy`: no equality is claimed; the operation lowers precision, and a
  realisation of it is a change of the program, recorded as one.

Each operation admits a strongest level (GT-4). The *level in force* is
`exact`, unless `tensor.within(N, t)`, N an integer literal, or
`tensor.lossy(t)` around the operation declares another for every
operation in t; an inner declaration takes the place of an outer one. A
node carries the level in force, which may be weaker than its operation's
level and never stronger. E3's fourth level, bitwise within an
accumulation class, is `exact` here: each operation fixes its order of
operations (GT-4), so its accumulation class has one member.

**GT-4 (operations).** `rce graph ops` prints this table. `e` is the
element type, the same for every tensor operand of one operation unless
the row names the types (N4: no promotion). Arithmetic is that of `e`: each
`+ - * /` and `sqrt` is one operation of `e` (binary32 for `f32`), rounded
once to nearest even, with no fused multiply-add. E(x), T(x) and R(x) are
`math.exp`, `math.tanh` and `math.erf` (6.3 F9) of the element as binary64,
which is exact, rounded once to `e`; for `f64` they are those functions,
and for `f32` the binary32 value nearest the binary64 one, which is the
correctly rounded binary32 function except where the binary64 value is a
binary32 midpoint. A sum runs in increasing index from +0.0 (from 0 for
integers). An axis, an epsilon, a shape or a permutation is a literal, so
the term's key covers it; an epsilon is rounded to `e`. Every operation of
this table but the three that lower precision is `exact`: its order of
operations is fixed and each of its roundings is correctly rounded, so a
realisation can reproduce the reference's bits.

| Operation | Operands and result | Order of operations | Level |
|---|---|---|---|
| `tensor.matmul(a, b)` | `[[e; k]; m]`, `[[e; n]; k]` to `[[e; n]; m]` | each element adds its k products in increasing k, starting from +0.0 | exact |
| `tensor.matvec(a, x)` | `[[e; k]; m]`, `[e; k]` to `[e; m]` | `matmul` with x as one column | exact |
| `tensor.transpose(a)` | `[[e; n]; m]` to `[[e; m]; n]` | no arithmetic | exact |
| `tensor.permute(t, [a1, …])` | a tensor to its axes in the order a1, … (a permutation of 0 to r − 1) | no arithmetic | exact |
| `tensor.reshape(t, [d1, …])` | a tensor to the same elements in row-major order with shape (d1, …) | no arithmetic; the element counts are equal | exact |
| `tensor.concat(a, b, AXIS)` | two tensors equal in every length but AXIS to a then b along AXIS | no arithmetic | exact |
| `tensor.add(a, b)`, `tensor.mul(a, b)` | two tensors whose shapes broadcast (aligned from the last length, each pair equal or one of them 1) to the broadcast shape | element by element | exact |
| `tensor.sum(t)` | a tensor to a scalar of `e` | every element added in row-major order; an integer partial sum outside `e` traps | exact |
| `tensor.relu(t)` | a tensor to one of its shape | each element x: x when x > 0, else +0.0 | exact |
| `tensor.exp(t)`, `tensor.tanh(t)` | a float tensor to one of its shape | E(x), T(x) | exact |
| `tensor.sigmoid(t)` | as `exp` | 1 / (1 + E(−x)) for x ≥ 0; E(x) / (1 + E(x)) for x < 0 | exact |
| `tensor.silu(t)` | as `exp` | x / (1 + E(−x)) for x ≥ 0; (x · E(x)) / (1 + E(x)) for x < 0 | exact |
| `tensor.gelu(t)` | as `exp` | (x · 0.5) · (1 + R(x · c)), c = 0.7071067811865476 | exact |
| `tensor.gelu_tanh(t)` | as `exp` | (0.5 · x) · (1 + T(c1 · (x + c2 · ((x · x) · x)))), c1 = 0.7978845608028654, c2 = 0.044715 | exact |
| `tensor.softmax(t, AXIS)` | a float tensor to one of its shape | per slice along AXIS: m its maximum, e_i = E(x_i − m), s = the sum of e_i, y_i = e_i / s | exact |
| `tensor.rmsnorm(x, w, EPS)` | x a float tensor, w `[e; d]`, d x's last length, to x's shape | per row of d elements: ss = the sum of x_i · x_i; inv = 1 / sqrt(ss / d + EPS); y_i = (x_i · inv) · w_i | exact |
| `tensor.layernorm(x, w, b, EPS)` | as `rmsnorm`, b `[e; d]` | per row: μ = (the sum of x_i) / d; v = (the sum of (x_i − μ) · (x_i − μ)) / d; inv = 1 / sqrt(v + EPS); y_i = ((x_i − μ) · inv) · w_i + b_i | exact |
| `tensor.attention(q, k, v)` | q `[[e; d]; sq]`, k `[[e; d]; sk]`, v `[[e; dv]; sk]` to `[[e; dv]; sq]`; or heads first, `[h, s, d]`, the heads of q a multiple of those of k and v (query head i reads key and value head i / (hq / hk)) | per query row: a_j = q · k_j (in increasing index); with s = 1/√d (binary64, rounded to `e`), m the maximum of a_j · s, p_j = E(a_j · s − m), Σ = the sum of p_j, p_j = p_j · (1 / Σ); y_t = the sum over j of p_j · v_jt | exact |
| `tensor.attention_causal(q, k, v)` | as `attention`, sq ≤ sk | as `attention`, query row i over the keys j ≤ i + sk − sq | exact |
| `tensor.embed(table, ids)` | table `[[e; d]; n]` of a float type, ids a tensor of `i32`, to ids' shape followed by d | the rows the ids name; an id outside 0 to n − 1 traps `index ID out of range` | exact |
| `tensor.matmul_q8_0(x, q, d)` | x `[[f32; k]; m]`, q `[[i8; k]; n]`, d `[[f32; k/32]; n]`, k a multiple of 32, to `[[f32; n]; m]` | y_ij = the sum over blocks b of d_jb · s_b, s_b the sum of (q_jt as f32) · x_it over the block's 32 positions in increasing t | exact |
| `tensor.matmul_q4_0(x, q, d)` | as `matmul_q8_0`, each q in −8 to 7 (else the trap `Q4_0 value V outside -8..7`) | as `matmul_q8_0`, each block summed in the order 0, 16, 1, 17, …, 15, 31 | exact |
| `tensor.round_bf16(t)` | a float tensor to one of its shape | each element rounded once to the nearest bfloat16 value (8 significant bits, the exponent range of binary32), ties to even | lossy |
| `tensor.round_q8_0(t)`, `tensor.round_q4_0(t)` | an `f32` tensor whose last length is a multiple of 32 to one of its shape | each block of 32 along the last axis replaced by the values its Q8_0 or Q4_0 quantization represents, quantized as GGUF's reference quantizers do (`rce_vm_backend_cpu::ml::nn::quantize_block`) | lossy |
| `a + b`, `a - b`, `a * b`, `a / b` | two tensors of one shape to one of that shape | element by element in row-major order; for integers each result is checked against `e` and `/` truncates | exact |

The orders of `rmsnorm`, `layernorm`, `attention`, `matmul_q8_0` and
`matmul_q4_0` are those of the portable kernels of `rce-op-nn-prims`,
`rce-op-attention` and `rce-op-qmm`, so those kernels differ from the
reference only where they take `exp` or `tanh` from the platform (and, for
`layernorm`, on a row of only −0.0, whose sum they start from −0.0). No
operation has a `within N` level of its own: `within N` is what a program
declares where it accepts a realisation that differs from the reference by
up to N units in the last place.

**GT-5 (order of evaluation).** The operands are evaluated left to right,
depth first, as the arguments of a call are (6.2), and then the term. A
trap or failure of an operand is the step's, as for any expression; the
term itself traps only as GT-6 states.

**GT-6 (the reference evaluation).** The reference interpreter evaluates
a term with RCE's CPU backend (`compute/crates/rce-vm-backend-cpu`,
`interpret_reference`): the nodes in one fixed topological order (Kahn's
algorithm, the ready nodes taken in ascending index), each by a
single-threaded kernel with the order of GT-4, one IEEE 754 rounding per
operation and no fused multiply-add. An `f32` operation is the binary32
operation, which for `+ - * /` equals computing in binary64 and rounding
once (6.3 N7), since 53 ≥ 2 × 24 + 2. After each node every float element
of its value is tested: NaN or an infinity stops the evaluation with the
trap `non-finite float` (6.3 N3), and so does a float division by zero. A
node whose element type is `i8` or `i32` (an `i8` tensor is held in `i32`
lanes) computes each element exactly and traps with RCE's integer traps
(6.3 N2: `overflow: V is outside T`, `division by zero`); an embedding
lookup traps on an id out of range. E, T and R are computed by
`crmath.rs` of the backend, which evaluates `math.rce`'s `exp`, `tanh` and
`erf` by the same method (Ziv's strategy over exact integers, the same
integers in base 2^32 limbs where `math.rce` uses 2^30) and reads ln 2 and
2/√π from `math.rce`. The decision, and its reason: running `math.rce` in
the reference interpreter for each element takes about 7 ms per `exp` and
30 ms per `erf` on the machine this was measured on, which no tensor of a
model can afford, while a correctly rounded value is unique, so a port
that is correctly rounded gives `math.rce`'s bits. That it is: the port
equals the MPFR-checked vectors of `stdlib/tests/math` (6,631 `exp`,
15,099 `tanh` and 14,761 `erf` lines) and an oracle in exact decimal
arithmetic on 1,951 arguments with edge cases
(`compute/crates/rce-vm-backend-cpu/tests/crmath.rs`), and every
elementwise operation equals its definition written in RCE over
`math.rce`, run by the reference interpreter
(`spec/checker/tests/graph/defs.rce`).
The result is therefore a function of the operands' bits, so `rce run`,
`rce test`, `rce trace`, the host and replay give the same bytes on every
run. This is tested (`spec/checker/tests/graph_term.rs`), not yet proved
in Lean. The core evaluator (`rce run-core`) evaluates `graph` statements
with the same backend and is compared with the reference trace like every
other core construct.

**GT-7 (charges).** An application of a term takes one tick and C_CALL
cost units, and then, for each scalar operation of its nodes, one tick and
two cost units (6.7, C.6.4): 2mnk operations for a matrix product and a
quantized one; for an attention of h heads, 2·h·sq·sk·(d + dv) +
3·h·sq·sk; for a softmax three, and for a normalisation four, per element;
none for a reshape; the element count of the result for the other
operations. A term that would take fuel or a budget past its limit traps as
any expression does.

**GT-8 (the canonical form of tensor values).** A tensor embeds in the
Form as nested lists: a tensor of shape (d₁, …, d_r), r ≥ 1, is the list of
its d₁ subtensors of shape (d₂, …, d_r), and a tensor of rank 0, a scalar,
is a float. Each element is written by rule 3 of 6.1.1. An `f32` element is
its binary32 value read exactly as binary64, so `0.1 as f32` is written
`0.10000000149011612`. -0.0 keeps its sign, and NaN and infinities never
occur (GT-6). The positional form (T7) of a tensor is the same list, since
a tensor holds no names. So the state hash, the entry hash of D and every
trace cover tensors with no rule beyond 6.1.1. Example: the state of
`dense.rce` after `Input(x: [1.0, 2.0, 3.0])` is hashed as
`[[[[0.5,-0.5,0.5],[-1.0,1.0,-1.0]],[0.5,-0.25],[1.5,0.0],1]]`. A tensor
whose list form exceeds VALUE_LIMIT cannot be held in the state (N9); an
encoding of large tensors by a hash of their bytes is not part of this
section.

**GT-9 (weights by digest).** A term names a tensor of data it does not
compute, a weight, by the content of its bytes: `tensor.blob_f32(DIGEST,
[d1, …])`, and `blob_f64`, `blob_i8`, `blob_i32`, with DIGEST a text literal
of 64 lowercase hexadecimal digits and the shape a literal. Its value is
the tensor whose bytes, its elements in row-major order each in
little-endian bytes (an `i8` in one byte, two's complement), have the
BLAKE3 hash DIGEST; their number must be the shape's element count times
the element's size. The digest is a literal of the term, so the term's
key, and with it the program hash, covers the weight, and a program names
its weights by content, never by path. The bytes come from a *store*: a
directory holding each blob as the file DIGEST
(`compute/crates/rce-fgm`), checked against its digest when read. The
toolchain reads the store `blobs` beside the program file it is given;
since a blob is the bytes its digest names, any store that holds it gives
the same value. A blob no store holds stops the evaluation (`graph term: no
store holds the blob DIGEST`), as a missing file does; it is not a trap of
the program. The host loads every blob a program names when it starts and
records the digests and lengths it loaded in D (`host.Blobs`, Appendix E)
before any input. `rce lift gguf` (10.17) writes such programs.

**W7 (graph terms in the core).** A core module read back from JSON
(`rce emit core FILE`, `rce run-core FILE`) requires change, with the rule
W7 and the term's key, when a term's key is not the content hash of its
graph and typing, when an operation node carries no level or a level
stronger than its operation admits, when the graph is not well formed or
its shapes do not type, or when a `graph` statement names a term the module
does not declare or gives operands of other types than the term's
parameters. W7 joins W1 to W6 of the one-emitter proposal (ONE-3).

The source rules are W-GRAPH-1 (shape), W-GRAPH-2 (operand) and
W-GRAPH-3 (level) of section 7. `rce emit core` prints graph terms;
`rust`, `python`, `js`, `scala` and `verilog` report status 2 for a program
with graph terms until their print tables have rows for them. `rce graph`
(10.16) writes a program's terms as `.fg` files, inspects them and
evaluates them.

**Read receipt.** Name-free hashing uses indices named after which logician? (two words)

<!-- body-sha256: 57c36fde0e8a007d5170183a68164c52b4fadbf25dc7e8ffa7fe8d82a91605db -->
