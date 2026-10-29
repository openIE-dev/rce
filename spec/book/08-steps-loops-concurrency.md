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
default; `rce fmt --nameless` (10.6) is the projection that renders
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
a final step whose outcome is o, where for `-> kept: X` the last step is
kept with the error X, for `-> kept: trap("K")` the last step is
kept by the trap K (earlier events must be applied in both cases), and for
`-> e` the event list is fully applied and then

- if the program has exactly one region with exactly one field f, the
  value of f equals ⟦e⟧;
- otherwise e must be a `bool` expression over the final state, with fields
  in scope by name (or `R.f` when ambiguous), and must be true.

`check e` (in an intent or at comptime) is the proposition ⟦e⟧ = true,
evaluated at comptime. A program is admitted only if every example and check
holds (W-INT-1): the gate is part of the language, not of a test harness.
Derivation of the trap form: a step ends in a value, a named error or a trap
(T2), so a proposition about one final step needs all three outcomes, or a
program whose specified behavior is a trap has no example that states it and
no emitted program can be held to it.

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
quotient is not implemented; `rce dim --control` (10.11) computes the
control quotient of the explored concrete machine instead.

**Canonical control form.** Let M_c be the minimal machine of P_c, its
states numbered by first visit in a breadth-first traversal from s₀ taking
events in declaration order (the name-free key of 6.6) and payloads in the
order of their boundary sets (Reis, Moreira and Almeida 2005: the
canonical string of an initially connected DFA is unique and name-free by
construction). `shape_ctl(P)` is the SHA-256 over the canonical JSON (6.1.1)
of that string. `shape_ctl` equal means the same control behaviour up to
predicate equivalence; ≡ means the same behaviour, decidable for finite S
by partition refinement on reach(P) and in practice bounded by the
exploration fuel of `rce dim` (10.11). The proposal `form-ir-2.md` names
two further keys, a name-free term hash and an e-class key, that this text
does not define.

**Read receipt.** Name-free hashing uses indices named after which logician? (two words)

<!-- body-sha256: 6e403282d3fe412ba0ac3a00a518a340e3b1337b0978399aeaba3b2571c162ad -->
