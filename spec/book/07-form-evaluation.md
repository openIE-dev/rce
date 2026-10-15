## 6. Dynamic semantics: the mapping into the Form

This section gives the *dynamic semantics*: what a program computes, as a
mapping into the Form. It also lists the exact edge behaviour of every
primitive (overflow, division by zero, float rounding, text indexing), which
every implementation must reproduce. Example: `-7 / 2` is `-3`, and
`i64.MIN / -1` traps with "overflow: 9223372036854775808 is outside i64".

### 6.1 The Form of a program

Let the program declare regions R₁…Rₖ (`state` and `actor` declarations;
actors also carry a mailbox), events E₁…Eₘ and errors X₁…Xⱼ. Its Form
P = (S, E, δ, D, λ) is:

![Form P = (S, E, delta, D, lambda) with its log D](https://raw.githubusercontent.com/openIE-dev/rce/main/docs/figures/rce-overview.svg)

*Figure (informative). Where the Form sits between the checker, the log D and the emitters. Source: [docs/figures](https://github.com/openIE-dev/rce/tree/main/docs/figures).*

- **S** = ⟦R₁⟧ × … × ⟦Rₖ⟧, where ⟦R⟧ is the set of records {fᵢ : ⟦τᵢ⟧} of R's
  field types, with the initial state s₀ = (r₁⁰, …, rₖ⁰), rᵢ⁰ the record of
  field initializers (evaluated at comptime). Actor regions add a mailbox
  queue and a held-message queue (Form law 3).
- **E** = the disjoint union of the events' payload sets:
  E = ⊎ₗ {Eₗ} × ⟦payload(Eₗ)⟧, plus the built-in `Tick(n: u64)` and the
  task events of 6.8.
- **δ : S × E → S × O\*** (state and the outputs of the step) is defined in
  6.4. It is a total function (T2).
- **λ** maps each step to its output sequence O\*: values `emit`ted, and for a
  refused event the record `Refused(X, payload)`.
- **D** is the log: the sequence of admitted events e₁, e₂, …, each entry with
  its outcome (applied, or refused with X). The fold law: the state after the
  log is sₙ = foldl(δ₁, s₀, D) where δ₁ is the state component of δ.

**Canonical signature (normative).** This is the one signature; the engine
profile (D.6), the engine IR (`FormIr::step`) and the language interpreter
are instances of it, and any other text (FORM.md, papers) defers to it.
S = Σc × Σd (control regions, data: regions, clock, input count,
relations); δ : S × E → S × O\* total; λ = π₂ ∘ δ; δ₁ = π₁ ∘ δ. An entry of
D is the record {seq, event, prev_hash, state_hash, hash} where `seq` is
the 0-based index, `event` is {type, payload}, `prev_hash` is the `hash`
of the previous entry (64 zeros for the first), `state_hash` is the
SHA-256 of the canonical JSON of the state after the step, and `hash` is
the SHA-256 of the canonical JSON of {event, prev_hash, seq,
state_hash}. A log may also carry `outcome`, `out` and `state`; these are
derived and must equal replay (T1), so they are not hashed. A refused
event (T8) is recorded with its unchanged state and ends the example
log. Both producers are tested against this rule
(`spec/checker/tests/form_signature.rs`, `tests/form_signature.rs`).

Values ⟦τ⟧ are the Form's JSON-shaped values (form/types.rs): integers
(i64), floats (finite f64), text, bool, null, lists, records. The embedding
of RCE types into them is fixed:

| RCE type | Form value |
|---|---|
| bool | bool |
| iN, uN, usize | int, with the range of the RCE type checked at every write |
| f64 | float, finite, normalized (6.3 N1) |
| text | text (sequence of Unicode scalar values) |
| () | null |
| (a, b, …), [τ; n], List[τ] | list |
| Set[τ] | list, sorted by canonical order, no duplicates |
| Map[k, v] | list of [k, v] pairs sorted by canonical order of k |
| struct / state record | record (keys sorted, as canonical JSON) |
| kind variant V(f: …) | record {"$v": "V", f: …}; unit variant: {"$v": "V"} |
| τ? | null for none, the value for some (and `some(none)` is refused: τ may not itself be an option, W-TY-2) |
| chan τ, actor handle | record {"$chan": id} / {"$actor": id}, id the content hash of 6.6 |

Canonical order of values (used by Set, Map and select): null < bool < number <
text < list < record; numbers by value; text by code point; lists and records
lexicographically. This is a total order because NaN does not exist.

### 6.2 Evaluation

Expressions are evaluated by a big-step judgment
⟨e, ρ, σ⟩ ⇓ r, where ρ maps variables to values, σ is the store of `var`s
and fields, and the result r is one of

- `ok(v, σ′)`: a value and updated store,
- `trap(m)`: a defined failure of the Form. The trap texts are exactly
  those of 6.3 ("overflow: V is outside T", "division by zero",
  "shift amount out of range", "non-finite float", "index K out of range",
  "slice index out of range", "slice bounds reversed", "not a char
  boundary", "size limit", "value limit"), "loop bound" (6.7) and "budget"
  (C.5),
- `fail(X, v)`: a user error raised by `fail` or propagated by `try`,
- control signals `ret(v)`, `brk`, `cont` (statements only).

Evaluation order is fixed: left to right, operands before operators,
arguments before the call, receiver before arguments. `and`/`or` short-circuit.
There is no other source of nondeterminism in expressions.

Selected rules (the rest are the standard ones for a strict language):

```
  ⟨a,ρ,σ⟩ ⇓ ok(x,σ₁)   ⟨b,ρ,σ₁⟩ ⇓ ok(y,σ₂)   arith_ι(⊕,x,y) = v
  ──────────────────────────────────────────────────── [E-Arith]
  ⟨a ⊕ b,ρ,σ⟩ ⇓ ok(v,σ₂)

  ... arith_ι(⊕,x,y) = Err(m)
  ──────────────────────────────── [E-Arith-Trap]
  ⟨a ⊕ b,ρ,σ⟩ ⇓ trap(m)

  ⟨e,ρ,σ⟩ ⇓ ok(err(X,v),σ′)                ⟨e,ρ,σ⟩ ⇓ ok(val(v),σ′)
  ──────────────────────── [E-Try-F]       ─────────────────────── [E-Try-V]
  ⟨try e,ρ,σ⟩ ⇓ fail(X,v)                  ⟨try e,ρ,σ⟩ ⇓ ok(v,σ′)

  ⟨e,ρ,σ⟩ ⇓ ok(none or err(_),σ′)
  ─────────────────────────────────── [E-TryElse]
  ⟨try e else X,ρ,σ⟩ ⇓ fail(X, ⟦X⟧)

  ⟨e,ρ,σ⟩ ⇓ ok(err(X,v),σ′)   ⟨h,ρ[x↦X(v)],σ′⟩ ⇓ r
  ──────────────────────────────────────── [E-Catch]
  ⟨e catch |x| h,ρ,σ⟩ ⇓ r
```

`trap` and `fail` propagate outward through every construct (they are not
caught by `catch`: `catch` handles only declared errors and none; a trap
always refuses the event).

**`&mut` write-back (normative).** A call writes back each `&mut`
argument and a `&mut self` receiver on every outcome, including `fail`:
the callee's writes made before it failed stay in the caller's store,
exactly as an in-place update (`fail` above leaves the store at σ′, the
store when it was raised, not σ). The derivation: the Rust target updates
in place, and copy-in/copy-out agrees with in-place update only when it
also copies out on failure (T33, `WriteBack.t33`; `copyOk_differs` shows
that copying out only on success disagrees). Atomicity is per event, not
per call: a `trap`, or a `fail` that leaves the step, refuses the event
and the state is the pre-state (T8). The program emitters implement the
rule for a write-back call that is a whole statement or expression of a
`let`, `var`, assignment, `emit` or `return`, and for such a call under
`catch` or `try` (`f(&mut x) catch h`, `try f(&mut x)`): the call is the
first thing the statement evaluates, so it can be hoisted without
changing evaluation order or budget charges. A write-back call nested
deeper in a strict position (an operand of an operator, an argument, a
list or map element, a receiver) is first rewritten by one shared
A-normal-form pass that runs before all three emitters
(`spec/checker/src/anf.rs`): the operands evaluated before it are bound
to fresh `let` temporaries in left-to-right order, then the call, so
reads, budget charges and emits keep their order (T34). A write-back
call in a conditional position (right of `and`/`or`, an `if` branch, a
`catch` handler, a lambda or a comprehension) is not flattened, since
that would evaluate it eagerly; it is refused with a diagnostic, never
given a different meaning. Tested on every target
(`spec/checker/tests/write_back.rs`). Values of type `τ ! ε` are `val(v)` or
`err(X, v)`; they are first-class only until a `try` or `catch` consumes them.

### 6.3 Edge semantics of primitives (normative, from form/num.rs)

These are the Form's own rules; every implementation reproduces them
exactly.

- **N1 Numbers.** A Form number is an int (i64) or a float (finite f64). At
  every event boundary a payload is normalized: a finite float with no
  fractional part and |x| < 2^63 becomes the int (`3.0` is `3`, `-0.0` is
  `0`). RCE's static types make the int/float distinction visible to the
  programmer; normalization is applied only when values cross the Form
  boundary (event payloads, state, outputs) and is invisible inside a
  computation because each RCE value carries its static type.
- **N2 Integer arithmetic** is checked: overflow of the *static type's*
  range is `trap("overflow: V is outside T")` with V the exact result and
  T the static type (`i64` for an untyped literal), never a wrap. `/` truncates toward
  zero (`-7 / 2 = -3`); `%` has the sign of the dividend (`-3 % 2 = -1`).
  Division or remainder by zero is `trap("division by zero")`.
  Modular and clamped arithmetic are explicit methods, never the default:
  `a.wrapping_add(b)`, `wrapping_sub`, `wrapping_mul` give the result
  modulo 2^N mapped into the static type's range, and `saturating_add`,
  `saturating_sub`, `saturating_mul` clamp it to the range; neither traps.
  u64 and usize arithmetic uses the Form ops `uadd usub umul udiv urem`
  (draft 0.5), which compute the exact result and trap unless it lies in
  [0, 2^64); a value at or above 2^63 is carried as an unsigned Form int, so
  `18446744073709551615 - 1` is exact. Lean: `RCE/U64.lean`.
  `i64.MIN / -1` and `i64.MIN % -1` are
  `trap("overflow: 9223372036854775808 is outside i64")` (the quotient is
  out of range; Rust's `checked_rem`). Shifts: `x << n` and `x >> n` with
  n ≥ bit width or n < 0 are `trap("shift amount out of range")`; `>>` is arithmetic for signed and
  logical for unsigned types. Bitwise `& | ^ ~` act on the two's-complement
  representation of the static type.
- **N3 Floats.** The rules below are derived from IEEE 754-2019 and from
  two requirements of the Form: replay must give the same bits on every
  target (T1), and δ must be total (S1). `tests/float_rules.rs` checks
  each rule on the interpreter; the parity set carries the vectors of
  `examples/57_float_vectors.rce` to every CI target.
  - **F1 Correct rounding.** f64 is IEEE binary64. `+ - * /` and `sqrt` are
    the standard's basic operations, correctly rounded with
    roundTiesToEven, the default rounding-direction attribute; so are the
    conversion of an integer (`as f64`) and of a decimal literal. A
    correctly rounded operation is a function of its operands' bits alone,
    so every conforming target gives the same result: this is the only
    property determinism needs, and no other rounding mode exists in RCE.
  - **F2 One rounding per operator, in source order.** Float addition and
    multiplication are not associative and `a * b + c` with one rounding
    differs from two roundings (vector `MulSub`: `0.1 * 10.0 - 1.0` is `0`,
    fused it is `5.551115123125783e-17`). Contraction to fused
    multiply-add, reassociation, excess precision and every other
    value-changing optimization the standard allows a language to permit
    are therefore not valid compilations (HC-14). A fused multiply-add is
    itself correctly rounded, so it may exist only as an explicit named
    operation; RCE has none yet. Adding one needs a correctly rounded
    implementation on every runtime (Rust `f64::mul_add`, Java
    `Math.fma`, exact rational rounding in Python and JavaScript).
  - **F3 Subnormals are preserved.** Gradual underflow is part of the
    binary64 format; flush-to-zero and denormals-are-zero modes are not
    IEEE operations and would make `x - y == 0` hold for `x != y`. A host
    never enables them (HC-13); vectors `Mul(5e-324, 3.0)` and
    `Mul(2.2250738585072014e-308, 0.5)` detect either mode.
  - **F4 Non-finite results trap (from totality).** A Form value must
    have a reflexive equality and a total order (6.2, N5, N6) and must
    encode as a JSON number for hashing (N1). NaN is unequal to itself
    under IEEE comparison, and an infinity is not a JSON number and
    reintroduces NaN through `inf - inf` and `0 * inf`; the largest subset
    of binary64 closed under these requirements is the finite values. δ is
    total as a function into values and traps, so an operation whose IEEE
    result is not finite (the standard's overflow, divideByZero and
    invalid exceptions) is `trap("non-finite float")`. The alternatives
    fail a law: admitting NaN under the standard's totalOrder makes `==`
    disagree with IEEE comparison and needs a canonical NaN for hashing;
    saturating to the largest finite value is not an IEEE result and
    breaks `x * 2.0 / 2.0 == x` without a signal. With the trap every
    value is finite and every result that does not trap is the IEEE result.
    `x / 0.0` and `sqrt` of a negative number trap.
  - **F5 NaN payloads are never observable.** By F4 no NaN becomes a
    value, so its sign, payload and quiet bit cannot reach state, output
    or a hash; the standard leaves payload propagation to the
    implementation, which is why this matters. An implementation tests
    for a non-finite result before any NaN escapes, and the trap text is
    the same for every NaN. A fabric or target that can produce NaN (GPU,
    NPU, WASM SIMD) canonicalizes or traps before a value reaches state
    (HC-13; `wasm_float.rs` rewrites NaN to the canonical positive NaN).
  - **F6 Signed zero.** `-0.0` arises by the standard's sign rules, which
    are exact, so every target produces it in the same places (vector
    `Mul(0.0, -1.0)`). It compares equal to `0.0`, `tostr` gives `0` for
    both and `1.0 / -0.0` traps like `1.0 / 0.0`, but canonical JSON keeps
    the sign, so a state holding `-0.0` has a different `state_hash` from
    one holding `0.0`. (Open question for review: equal values with
    different hashes; canonicalizing `-0.0` to `0.0` in canonical JSON
    changes D for such states and must change every producer at once.)
  - **F7 Exception flags are not state.** Inexact and underflow raise
    nothing (F1 already fixes the result); overflow, divideByZero and
    invalid are the trap of F4. No program reads a flag.
  - **F8 Exact operations.** `floor`, `ceil` and `round` are the standard's
    roundToIntegral operations toward negative, toward positive and
    ties-away (`(2.5).round()` is `3`, `(-2.5).round()` is `-3`); `abs`
    clears the sign. `min` and `max` follow N5. None of these rounds.
  - **F9 Transcendentals are excluded.** IEEE 754-2019 recommends but does
    not require correctly rounded `exp`, `log`, `sin`, `pow` and the like,
    and platform math libraries differ in the last bit. RCE has none. One
    may be added only with a correctly rounded implementation in the
    CORE-MATH or CRlibm style (correct rounding proved, or checked
    exhaustively for binary32), carried by every runtime rather than taken
    from the platform library; until then device versions are output only
    or verified (HC-13).
  Canonical float text is the shortest round-trip digits; decimal notation
  for 1e-5 ≤ |x| < 1e16, else `d.ddde±N` with `+` on positive exponents
  (`1e+300`, `1.5e-7`).
  In canonical JSON (6.1) a float is written as Rust's `{:?}` writes it,
  which every producer reproduces: shortest round-trip digits, decimal with
  at least one fractional digit for 1e-4 ≤ |x| < 1e16 (`10.0`, `0.0001`),
  otherwise exponent form without `+` (`1e16`, `1e-5`, `5e-324`), and
  `-0.0` keeps its sign (`tests/traces.rs` pins these forms).
  **f32 (draft 0.5).** The Form ops `f32add f32sub f32mul f32div` round both
  operands to binary32, compute in binary64, and round the result to
  binary32 (round to nearest even). Binary64 has 53 ≥ 2·24 + 2 significand
  bits, so by Figueroa's theorem the result equals the correctly rounded
  binary32 result, and every target (Rust `f32`, Python `struct`, JavaScript
  `Math.fround`) gives the same bits. A result outside the binary32 range is
  `trap("non-finite float")`. `x as f32` (Form `tof32`) rounds once.
  Lean: `RCE/F32.lean`.
- **N4 Mixed arithmetic** does not exist in RCE (both operands of a binary
  operator have the same static type). `a as f64` converts an integer exactly
  when |a| ≤ 2^53 and otherwise rounds to nearest even; `x as iN` truncates
  toward zero and traps if the result is out of range or x is not finite.
- **N5 min / max** compare exactly on integers and by IEEE order on floats
  (total, since operands are finite); on a tie the receiver is returned.
- **N6 Equality** is structural value equality after normalization;
  `1 == 1.0` is a type error in RCE (different static types), so the Form's
  "1 equals 1.0" rule matters only at the boundary.
- **N7 Text** is a sequence of Unicode scalar values compared by code point,
  stored as UTF-8. From draft 0.5, `len`, indexing and slicing count UTF-8
  bytes (Form `blen`, `bget`, `bslice`): `s[i]` is a `u8`, and a slice whose
  ends are not character boundaries is `trap("not a char boundary")`.
  `for c in s` and `chars()` still go by character. Case mapping
  (`upper`, `lower`) and class tests (`isdigit`, `isalpha`, `isupper`,
  `islower`) are ASCII-only; other characters are unchanged / false.
  `strip`, `words` and `toint` treat as whitespace exactly the 25 code
  points of the White_Space property of Unicode 18.0.0: U+0009 to U+000D,
  U+0020, U+0085, U+00A0, U+1680, U+2000 to U+200A, U+2028, U+2029,
  U+202F, U+205F and U+3000 (`interp::WHITE_SPACE`). `isspace` is ASCII
  (space, \t, \n, \x0c, \r).
  **Unicode pin.** RCE pins Unicode 18.0.0
  (https://www.unicode.org/versions/Unicode18.0.0/) and reads no Unicode
  table at run time: scalar values and UTF-8 do not change between
  versions, case and class operations are ASCII, and the one property used,
  White_Space, is enumerated above rather than taken from the toolchain. A
  newer Rust, Python, JavaScript or JVM therefore cannot change a result;
  `tests/unicode_pin.rs` checks the set and scans the runtimes for calls
  that would read the toolchain's tables. Moving the pin is a spec change
  that must leave the enumerated set unchanged or bump the draft.
- **N8 Indexing** The index `i` of `a[i]` and the ends of `a[i:j]` have
  type `usize` (W-TY-1 otherwise), so no index is negative. `a[i]` with
  i ≥ n is `trap("index i out of range")`. A slice end above n is
  `trap("slice index out of range")`, and i > j is
  `trap("slice bounds reversed")`; an omitted end is 0 or n. Indexing and
  slicing are total or trapping by one rule: an out-of-range position is
  never silently moved.
- **N9 Limits.** A list or text built by any operation is at most
  SIZE_LIMIT = 1,000,000 elements (`trap("size limit")`). A value written to
  state is at most VALUE_LIMIT = 1,048,576 bytes of canonical JSON
  (`trap("value limit")`, the event is refused, the value is not recorded).
- **N10 Absence.** `none` and a present `()` differ, exactly as the Form's
  null-versus-absent rule: an option field holding none is present.
- **N11 Time** is the logical clock only: `Tick` events and `after n`
  in `select`. No construct reads a wall clock.

The built-in methods are exactly the table of 5.15 (the closed standard
library); each has the semantics of the Form primitive of the same name in
num.rs and ir.rs (`Prim`, `Map`, `Filter`, `Fold`, `SortBy`), at the RCE
static type. A Form primitive that is not in that table (for example
`pow`, `floordiv`, `pmod`, `zip`, `replace`, `unique`) is not an RCE
method and is W-NAME-1.
`sort` and `sort_by` are stable.

**Read receipt.** Which JavaScript function is named for rounding to binary32?

<!-- body-sha256: 8caa95f44e22a22df26601dd74e4845a9468250fb801422f7dd50199537ad996 -->
