## 6 (continued). Dynamic semantics: steps, loops and concurrency

This section defines the *step function* δ (delta), which takes a state and
an event and returns the next state and an outcome, then errors, hashing,
loop bounds, concurrency and how examples are checked. A *trap* is a defined
failure that refuses the event and leaves state unchanged. Example: a loop
with `bound 10` that starts an 11th iteration traps with "loop bound".

### 6.4 The step function δ

For event e = (Eₗ, v):

1. For each region Rᵢ in declaration order, take the steps `step Rᵢ on p if g`
   in source order whose pattern p matches e and whose guard g evaluates to
   true in (Rᵢ's pre-state fields, bindings of p). The first such step is
   **the** step of Rᵢ for e. If there is none, Rᵢ is unchanged (frame).
   A guard that traps (overflow, division by zero, …) does not fall through
   to the next step: the event is refused with that trap (change S1; this is
   what makes δ a total function rather than leaving the case undefined).
2. Every selected step runs on the **pre-state** s (ASM semantics, Form law 2):
   no step sees another region's update for the same event.
3. The body of the step of Rᵢ is evaluated with fields of Rᵢ bound as `var`s
   to rᵢ, the pattern bindings as `let`s. Its result is
   - `ok(σ′)`: rᵢ′ = the field values in σ′ (each checked against its type
     range and VALUE_LIMIT), outputs oᵢ = the values emitted, in order;
   - `fail(X, w)` or `trap(m)`: the step is **refused**.
4. If any selected step is refused, the whole event is refused:
   δ(s, e) = (s, [Refused(X or trap(m), e)]), where X is the refusal of the
   first refused region in declaration order. Atomicity: an event either
   applies to every region or to none.
5. Otherwise δ(s, e) = ((r₁′, …, rₖ′), o₁ ++ … ++ oₖ).

The arrow form `step R on p -> e` is sugar for the block
`f = e` where f is R's only field.

**Order independence.** Since each step reads only its own region and the
event, the result of steps 2–5 does not depend on the order regions are
evaluated except through "first refused region", which is fixed by
declaration order. Outputs are concatenated in declaration order.

### 6.5 Errors, traps and refusals

A `fail X(…)` or a trap inside a step refuses the event: the state is
unchanged, the log records the event with outcome `refused(X)` and λ outputs
`Refused`. Nothing is thrown past a step; no event is dropped silently. In a
pure function called from comptime, refusal is a compile error. In `main`, a
trap or unhandled failure ends the command with the refusal as its output.

### 6.6 Determinism of identity: name-free hashing

Channel ids, actor ids and task ids are not counters. Each is the SHA-256 of
the canonical JSON of (creating construct's name-free key, creating task id,
per-task creation index), where the name-free key replaces identifiers by
their De Bruijn position (Form law 2). Renaming every identifier in a program
therefore leaves every id, every log hash and every select decision
unchanged (T7).

### 6.7 Loops and fuel

`for p in e` iterates over a finite value (list, set, map in canonical order,
text by scalar value, or an integer range `a..b` / `a..=b`); its length is
fixed when the loop starts, and mutating the iterated collection inside the
loop is refused statically (it is borrowed). `while c bound n` and
`loop bound n` run at most n iterations, n a comptime-or-runtime u64
evaluated once; the default bound is LOOP_LIMIT = 1,000,000. A program may
set its own default with a module-level `const LOOP_LIMIT: u64 = n`, n a
positive literal (draft 0.5; any other declaration of that name is refused,
W-LOOP-1). Starting iteration n + 1 is `trap("loop bound")`. Comptime
evaluation uses the same bounds. `for x in f(…)` over an iterator function
(5.14) counts yields against LOOP_LIMIT in the same way; Lean:
`RCE/Iter.lean` (`drive_total`, `drive_break`).

**Structural recursion (draft 0.4).** A function may call itself when
every self-call passes, in some parameter position i, a name bound inside a
constructor pattern of a `match` whose scrutinee is parameter i (or a name
already known to be such a component). The argument is then a strict
component of the value received, so along any chain of calls the value at
position i strictly shrinks and the depth of recursion is at most the
number of constructor nodes in that value, which N9 bounds. A name that is
rebound by `let`, `var`, assignment or a `for` pattern loses this status.
Any other cycle in the call graph, including every mutual recursion, is
refused (W-REC-1).

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
a final step whose outcome is o, where for `-> refused: X` the last event is
refused with X (earlier events must be applied), and for `-> e` the event list
is fully applied and then

- if the program has exactly one region with exactly one field f, the
  value of f equals ⟦e⟧;
- otherwise e must be a `bool` expression over the final state, with fields
  in scope by name (or `R.f` when ambiguous), and must be true.

`check e` (in an intent or at comptime) is the proposition ⟦e⟧ = true,
evaluated at comptime. A program is admitted only if every example and check
holds (W-INT-1): the gate is part of the language, not of a test harness.

**Read receipt.** Name-free hashing uses indices named after which logician? (two words)

<!-- body-sha256: 4d90e5d4d01a0c866448c178c9c6fc93ea320e987af09bb5f66a527cbc41184d -->
