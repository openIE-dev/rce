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
- A foreign function is not a value: its name may appear only as the
  callee of a direct call (W-FFI-1). The context rule above is checked at
  the call, so a value would let a pure function receive a foreign
  function from `main` and call it.
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
| `F` | f64 | `abs`, `ceil`, `floor`, `max`, `min`, `round`, `sqrt`, `tostr` |
| `Bool` | bool | `tostr` |
| `Text` | text | `chars`, `contains`, `endswith`, `isalpha`, `isdigit`, `isempty`, `isspace`, `join`, `len`, `lines`, `lower`, `raw`, `reverse`, `sha256`, `split`, `startswith`, `strip`, `toint`, `tostr`, `upper`, `words` |
| `F32` | f32 | `abs`, `tostr` |
| `Gen` | iter τ | `collect` |
| `List` | List[τ], [τ; n] | `append`, `clear`, `contains`, `filter`, `fold`, `get`, `isempty`, `len`, `map`, `pop`, `push`, `reverse`, `sort`, `sum`, `utf8`, `with` |
| `Map` | Map[k, v] | `contains`, `get`, `insert`, `keys`, `len`, `merge`, `remove`, `values` |
| `Set` | Set[τ] | `insert`, `inter`, `tolist`, `union` |
| `Opt` | τ? | `isnone`, `issome`, `or` |

`raw` exists only in a version 0.8 module and only on `Text[C]` (5.16).

**Sets and map merge (normative).** A `Set[τ]` is a list in canonical
order without duplicates (7.1). `set_of(xs)` makes one from a
`List[τ]`. `s.union(t)` is the set of the elements of either,
`s.inter(t)` of the elements of both, `s.insert(x)` sets `s` to
`s.union(set_of([x]))` (a `var` receiver), and `s.tolist()` is the
elements as a list in that order. `m.merge(n)` on two `Map[k, v]` has
the keys of either; a key in both maps to `m[k] ⊔ n[k]` by the `Merge` of
`v` (a user `impl Merge`, or the built-in instance, C.11.5), and the
result is in canonical key order. `union`, `inter` and `merge` borrow
their argument. These are the join (and, for `inter`, the meet) of the
set and pointwise map lattices, so a `merge` written from them is
certified (C.11.5).

**Hashing (normative).** `t.sha256()` on a `text` t is the SHA-256
digest (FIPS 180-4) of the UTF-8 bytes of t, written as 64 lowercase
hexadecimal digits, as a `text`. It is pure, total and the same on every
target; it is the digest 6.1 and 6.8 apply to canonical JSON, so a
program given the canonical JSON of a value as text computes the same
hash as the Form log and the content-hash ids. It is in the linear class of C.6.4.

**Bytes to text (normative).** `b.utf8()` on a `List[i64]` b is the `text` whose UTF-8 encoding is exactly the bytes of b. It traps, and so refuses the event it runs in (T8), when an element is outside 0..255 or the bytes are not well-formed UTF-8 (RFC 3629: no overlong forms, no surrogates, nothing above U+10FFFF); it never substitutes characters, since that would change content silently. A leading byte-order mark is kept. It is pure, total on well-formed input, the same on every target, and in the linear class of C.6.4.

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

### 5.16 Indexed values (version 0.8)

In a module declaring `# rce-version: 0.8` four type names and four
contexts belong to the language: `Text[C]`, `Secret[τ]`, `Cap[A]`,
`Guarded[τ, A]`, and `Html`, `Url`, `Sql`, `Json`. A 0.8 module may not
declare an item with one of these names (W-NAME-2). Each is a type
*index*: the value is a value of the base type (text, τ, `()`, τ), and
the index is a static fact the checker carries. Indices are erased at
run time, so δ, the log and the emitters are unchanged except for the
encoders. Derivation (`spec/proposals/CORE-PROPERTIES.md`, P1): one
meaning per construct requires that a text which is HTML in one place
and SQL in another be two types; least privilege requires that
authority be a value, not an ambient fact; soundness requires that the
checker, not a convention, enforce both. The three indices are one
mechanism: a value carries an index, and it crosses an index only by a
named, total, costed construct.

**Context-typed text (W-CTX-1).** `Text[C]` for C one of the contexts
is text whose structure (markup, query syntax) is fixed by the program.
It is made only by `C.lit(s)`, where s is a string literal (program
text is trusted: its author wrote it), and by `C.encode(t)` for a `text`
t, which escapes every metacharacter of C. `a + b` joins two values of
one context; any other operator, or `+` with plain text or another
context, is W-CTX-1. `x.raw()` gives the text of a `Text[C]` as `text`
(going back only re-enters through `encode`). An event payload carries
plain text only (W-CTX-1): context text cannot come from outside.
`emit` may carry `Text[C]`; a host sink for C takes exactly that type.
The encoders (normative, every runtime): `Html` maps `&` `<` `>` `"` `'`
to `&amp;` `&lt;` `&gt;` `&quot;` `&#39;`; `Url` percent-encodes every
UTF-8 byte except `A-Z a-z 0-9 - . _ ~` with uppercase hex; `Sql` gives
a standard SQL string literal, the text between single quotes with each
`'` doubled; `Json` gives a JSON string literal with `"` `\` newline,
return and tab escaped and other code points below U+0020 as `\u00xx`.
Consequence (T36): every structural metacharacter of a `Text[C]` value
comes from a literal, so untrusted text cannot form markup or query
syntax. SQL injection and cross-site scripting are not expressible.
Example: `Sql.lit("WHERE name = ") + Sql.encode(b)` with b = `O'Brien`
is `WHERE name = 'O''Brien'`; `Sql.lit("WHERE name = ") + b` is W-CTX-1.

**Secrets (W-SEC-1).** `Secret[τ]` is opaque. A secret value comes only
from an event payload field of a `Secret` type: there is no constructor
and no literal of a secret type, so a credential cannot be written in
the program (hard-coded credentials are not expressible; comparing a
secret with a literal is W-TY-1). The only operations are binding,
storing (in a state field, a struct, an option), passing and `==`/`!=`
between two secrets of one type, whose boolean is the one release. Any
other operator or method on a secret is W-SEC-1. A value that can hold a
secret may not be emitted, sent on a channel, carried by an error
payload (a refusal is an output, 6.1) or passed to a foreign function
(W-SEC-1). So λ never carries a secret (T36, noninterference up to the
equality bits). The log D holds admitted events with their payloads, as
any store of the program's inputs does; it is not an output.

**Capabilities (W-CAP-1, W-CAP-2).** `Cap[A]`, for A a declared struct,
is the authority A as a value. It appears only as the type of an event
payload field or of a parameter (W-CAP-1): it cannot be stored in a
state, struct, collection, option or channel, so authority lives for one
event and is never ambient. It cannot be constructed by an expression:
the host that admits an event attests the capability (written `()` in an
event list or `example`), exactly as it supplies every other input, and
the attested event is logged. `Guarded[τ, A]` is the type of a state
field only (W-CAP-1); reading it gives a τ, it keeps τ's range checks
(6.3), and a step that writes it (assignment, `&mut` borrow, a mutating
method, or the arrow form of its only field) must match only events
whose payload has a field of type `Cap[A]` (W-CAP-2). So a missing
authorization check is not expressible: a log of events without the
capability never changes the guarded field (T36).

**Read receipt.** What kind of channel does a capacity of 0 give? (one word)

<!-- body-sha256: 350558d934b1128b283bfd3348c032658783e93b9d4242165fbfcd2dd77041ae -->
