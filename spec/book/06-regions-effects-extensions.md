## 5 (continued). Static semantics: regions, effects and extensions

This section continues the static rules for features that control where
values live and what code may do: *regions* (arenas freed as a block),
concurrency, traits, compile-time code, foreign functions, and the draft 0.5
linear types, device regions and iterators. Example:
`with arena r bound 4096` gives a block its own memory of at most 4096
canonical bytes, freed when the block ends.

### 5.7 Regions (arenas)

`with arena r` introduces region r for the block. A type `τ @r` is allocated
in r.

```
  Σ;Γ,r:region;Δ ⊢ body ⊣ Γ′    r ∉ free-regions(Γ′ restricted to dom(Γ))
  r ∉ free-regions(type of any value assigned to a place outside the block)
  ───────────────────────────────────────────────────────────── [Arena]
  Σ;Γ;Δ ⊢ with arena r body ⊣ Γ′ \ r
```

`with arena r bound n` gives r a capacity of n bytes (n : u64, evaluated
once at block entry). Capacity is counted in canonical bytes, the length
of the canonical JSON of each value allocated in r, so the limit is the
same on every implementation. An allocation that would take r past n is
`trap("size limit")`, and the event is refused as for any trap. Without
`bound` the capacity is unlimited except by N9. The reference evaluator
measures the values bound in the region after every statement of the
region's body. So nothing allocated in r escapes the block; copying out (`copy(x)`, or
pushing a Copy element) is how results leave. Each step body runs inside an
implicit region `step` freed when the step returns; state fields are not
region-typed (they live in the state's own storage).

### 5.8 Concurrency

- `chan[τ](n)` creates a channel with capacity n (a comptime or runtime
  usize; 0 means rendezvous) and type `chan τ`; τ must be a data type or a
  `chan`. Channels are Copy handles.
- `c <- v` moves v into the channel. `<- c` has type `τ?` (none when c is
  closed and empty).
- `spawn f(a₁…aₙ)` requires every argument to be a data value or channel
  (not a borrow, not region-typed): arguments are moved into the task
  (W-SPAWN-1). Tasks share nothing but channels and actor handles, so data
  races cannot be written.
- `select` arms: `recv p from c` (c : chan τ, p : τ?), `send v to c`,
  `after n` (n : u64 logical ticks), `default`. At most one `default`, at most
  one `after`.

### 5.9 Traits

A trait declares associated types and function signatures. `impl Tr for T`
must define every member with a signature equal to the trait's after
substituting `Self := T` (W-TRAIT-1). Coherence: at most one impl of a trait
for a type in the whole program, and an impl must be in the module of the
trait or of the type (orphan rule, W-TRAIT-2). Generic functions are
monomorphized (no dynamic dispatch, no vtables), so trait method calls are
resolved statically. Built-in traits: `Eq`, `Ord`, `Hash`, `Show`, `Copy`,
`Data`, derived structurally for every type whose components have them
(f64 is `Eq` and `Ord` with the total order of 6.3, since NaN never exists).

### 5.10 Comptime

An expression or block marked `comptime`, every `const` initializer, every
array length, every state field initializer, and every argument for a
`comptime` parameter is evaluated during compilation by the dynamic semantics
of section 6 (with the same edge semantics). A comptime evaluation that traps
or exceeds its fuel (6.7) is a compile error. A `comptime fn` may only be
called at comptime. Comptime results become constants in the Form IR.
Comptime evaluation cannot read files, clocks, environment or network
(W-CT-1), so compilation is deterministic.

### 5.11 Foreign functions

`extern "C" fn f(p₁: τ₁, …) -> σ` declares a function implemented outside
RCE with the platform C calling convention. It has no body. Rules (W-FFI-1):

- The ABI string is `"C"`. A foreign function is not `comptime`, not generic
  and has no `self`.
- Each parameter type is `bool`, an integer type, `f64`, `&text` (passed as
  a pointer to UTF-8 bytes and a byte length), `&[τ]` or `&mut [τ]` with τ one
  of those scalars (pointer and element count). Borrows are second class
  (5.6), so the foreign callee holds them for the duration of the call only;
  no copy is made.
- The result type is a scalar or `()`. No memory allocated by foreign code
  is ever owned by RCE, and no RCE allocation is freed by foreign code, so
  the two sides never need to agree on an allocator.
- A foreign function may be called only from `fn main` and task functions
  (W-EFF-1), never from a pure function, a step, a handler or comptime, and
  it cannot be spawned (W-SPAWN-1). δ therefore never calls foreign code.
- Each call is recorded in the log as `Foreign(f, arguments, result)`.
  Replay supplies the recorded result instead of calling f, so the run of a
  command remains a function of its inputs and its log (T1, T3), with
  foreign results counted as inputs.
- The memory behaviour of the foreign code is outside the language: T6
  holds for the RCE side on the assumption that the callee touches only the
  buffers it was given, writes only through `&mut` buffers, and keeps no
  pointer after it returns. This is the trust boundary; it is named by the
  `extern` keyword at the one place it occurs.

### 5.12 Linear types (draft 0.5)

`linear struct H` declares a type whose values must be used exactly once.
It is meant for handles to outside resources (a file, a socket, a device
buffer) that must be released once and only once. Rules (W-LIN-1):

- A linear value is moved, never copied: `copy(h)` is refused, and the type
  is not Copy whatever its fields are.
- Every linear local, parameter included, is consumed (moved into a call,
  returned, or sent into a foreign function) exactly once on every path:
  a linear local still live at the end of its block, at `return` or at
  `fail` is refused, and so is one consumed on some branches of an `if` or
  `match` but not others.
- A linear value is not moved inside a loop body unless it was created in
  that iteration.
- A linear type may not appear inside a collection, an array, an option,
  a channel, a state or actor field, an event or error payload, a struct
  field or an iterator's element type. So no data structure can hide a
  second copy or drop one.
- A foreign function (5.11) may take and return a linear struct whose one
  field is an integer or `bool`; the C side sees that scalar. This is how
  `open`/`close` pairs are typed.

Reading a field (`h.fd`) does not consume the value. Lean:
`RCE/Linear.lean` (`linear_exactly_once`).

### 5.13 Device regions (draft 0.5)

`with device d [bound n]` is a region (5.7) whose contents an implementation
may place in accelerator memory. In addition to the arena rules, a device
region holds fixed-size values only: no `List`, `Map`, `Set`, `text`,
slice, channel or iterator may be bound in it, and its body may not `emit`,
`spawn`, send, `select`, `yield` or call a foreign function (W-DEV-1). The
body is therefore pure arithmetic on scalars, tuples, structs of those and
arrays (with lane-wise operators), which is what accelerators run. Its
meaning is the same as the block run on the CPU, so the region is only a
placement hint; `bound` counts canonical bytes as for arenas.

### 5.14 Iterators (draft 0.5)

A function whose result type is `iter τ` is an iterator function. Its body
is a block that produces values with `yield e` (e : τ) and ends by falling
off the end or by `return` with no value. Rules (W-ITER-1):

- An iterator function is pure (no `chan` parameter, no effects), and
  `yield` appears only in one.
- A value of type `iter τ` may only be the operand of `for` or the receiver
  of `.collect()` (which returns `List[τ]`). It cannot be stored, passed or
  returned, so an iterator is never shared or resumed twice.

The meaning is the push model of Go 1.23 range-over-func: the loop body runs
at each `yield`, `break` in the body stops the iterator, and `return` in the
body returns from the enclosing function. The number of yields per run
counts against the loop limit (6.7), so `yield` number LOOP_LIMIT + 1 is
`trap("loop bound")`, and an iterator always terminates. Python generators
and Rust iterators are pull-based and can be resumed from anywhere; the push
form keeps delta total without any suspended state.

### 5.15 Standard library, effects and I/O

**Standard library (normative).** The standard library is closed: it is
exactly the builtin methods below plus the concurrency methods of 5.8 and
Appendix C (`close`, `join`, `offer`, `put`, `wait`, `freeze`, `pscan`,
`pscan_incl`) and the num.rs primitives of 6.3. A method not in the table
is W-NAME-1 at check time, so no program reaches an unknown method at run
time. Every method in the table is pure and deterministic: it reads its
receiver and arguments, and a method that "changes" a receiver (`push`,
`insert`, `sort`, ...) updates only that receiver variable; none performs
I/O. The table is checked
against the reference evaluator (`spec/checker/tests/stdlib_costs.rs`).

| Receiver | RCE type | Methods |
|---|---|---|
| `Int` | integers (iN, uN, usize) | `abs`, `checked_add`, `checked_div`, `checked_mul`, `checked_sub`, `max`, `min`, `saturating_add`, `saturating_mul`, `saturating_sub`, `tostr`, `wrapping_add`, `wrapping_mul`, `wrapping_sub` |
| `F` | f64 | `abs`, `ceil`, `floor`, `round`, `sqrt`, `tostr` |
| `Bool` | bool | `tostr` |
| `Text` | text | `chars`, `contains`, `endswith`, `isalpha`, `isdigit`, `isempty`, `isspace`, `join`, `len`, `lines`, `lower`, `reverse`, `split`, `startswith`, `strip`, `toint`, `tostr`, `upper`, `words` |
| `F32` | f32 | `abs`, `tostr` |
| `Gen` | iter τ | `collect` |
| `List` | List[τ], [τ; n] | `append`, `clear`, `contains`, `filter`, `fold`, `get`, `isempty`, `len`, `map`, `pop`, `push`, `reverse`, `sort`, `sum`, `with` |
| `Map` | Map[k, v] | `contains`, `get`, `insert`, `keys`, `len`, `remove`, `values` |
| `Opt` | τ? | `isnone`, `issome`, `or` |

**Effects and I/O (normative).** By derivation from least privilege, a
program has no ambient authority: there is no file, clock, network,
environment or randomness primitive. The only ways values enter are the
events of E (6.1) and, in `fn main` and task functions, channel receives
and foreign calls (5.11); the only ways values leave are `emit` (λ) and
sends. Each of these is an effect of the context table above, so the
checker, not run-time behaviour, decides which code may do I/O. Inputs
that are not events (a `select nondet` choice, a foreign result) are
logged, so replay is deterministic (T1, T21). A capability that is not in
this list cannot be added by a library, because the library is closed.

**Cost semantics.** Each standard library call costs `c_call`; in a
version 0.7 module it costs in addition the work of its complexity class
on the sizes of its receiver and argument (C.6.4, C.15); see C.6 for the
budget rules.

**Read receipt.** What kind of channel does a capacity of 0 give? (one word)

<!-- body-sha256: 0f0c4a2a2457da4a9dcedda9d940d7375ed0318aa79b79ec9ab38d8a37a6640a -->
