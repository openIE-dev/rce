## 6. Dynamic semantics: the mapping into the Form

This section gives the *dynamic semantics*: what a program computes, as a
mapping into the Form. It also lists the exact edge behaviour of every
primitive (overflow, division by zero, float rounding, text indexing), which
every implementation must reproduce. Example: `-7 / 2` is `-3`, and
`i64.MIN / -1` traps with "integer overflow".

### 6.1 The Form of a program

Let the program declare regions R₁…Rₖ (`state` and `actor` declarations;
actors also carry a mailbox), events E₁…Eₘ and errors X₁…Xⱼ. Its Form
P = (S, E, δ, D, λ) is:

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
- `trap(m)`: a defined failure of the Form ("integer overflow", "division
  by zero", "non-finite result", "index out of range", "size limit",
  "loop bound", "value limit"),
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
always refuses the event). Values of type `τ ! ε` are `val(v)` or
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
  range is `trap("integer overflow")`, never a wrap. `/` truncates toward
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
  `i64.MIN / -1` and `i64.MIN % -1` are `trap("integer overflow")`
  (Rust's `checked_rem`). Shifts: `x << n` and `x >> n` with n ≥ bit width
  or n < 0 are `trap("integer overflow")`; `>>` is arithmetic for signed and
  logical for unsigned types. Bitwise `& | ^ ~` act on the two's-complement
  representation of the static type.
- **N3 Floats.** f64 operations are IEEE 754 binary64, round to nearest even.
  `%` is fmod (sign of the dividend). A non-finite result (±inf, NaN) is
  `trap("non-finite result")`: NaN and infinities never exist as values.
  `x / 0.0` is `trap("division by zero")`. Canonical float text is the
  shortest round-trip digits; decimal notation for 1e-5 ≤ |x| < 1e16, else
  `d.ddde±N` with `+` on positive exponents (`1e+300`, `1.5e-7`).
  **f32 (draft 0.5).** The Form ops `f32add f32sub f32mul f32div` round both
  operands to binary32, compute in binary64, and round the result to
  binary32 (round to nearest even). Binary64 has 53 ≥ 2·24 + 2 significand
  bits, so by Figueroa's theorem the result equals the correctly rounded
  binary32 result, and every target (Rust `f32`, Python `struct`, JavaScript
  `Math.fround`) gives the same bits. A result outside the binary32 range is
  `trap("non-finite result")`. `x as f32` (Form `tof32`) rounds once.
  Lean: `RCE/F32.lean`.
- **N4 Mixed arithmetic** does not exist in RCE (both operands of a binary
  operator have the same static type). `a as f64` converts an integer exactly
  when |a| ≤ 2^53 and otherwise rounds to nearest even; `x as iN` truncates
  toward zero and traps if the result is out of range or x is not finite.
- **N5 min / max** compare exactly on integers and by IEEE order on floats
  (total, since operands are finite).
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
  `strip` and `words` split on ASCII whitespace (space, \t, \n, \r, \x0b,
  \x0c).
- **N8 Indexing** `a[i]` with i < 0 counts from the end (Python style);
  an index outside [−n, n) is `trap("index out of range")`. Slices `a[i:j]`
  clamp both ends into [0, n] and never trap.
- **N9 Limits.** A list or text built by any operation is at most
  SIZE_LIMIT = 1,000,000 elements (`trap("size limit")`). A value written to
  state is at most VALUE_LIMIT = 1,048,576 bytes of canonical JSON
  (`trap("value limit")`, the event is refused, the value is not recorded).
- **N10 Absence.** `none` and a present `()` differ, exactly as the Form's
  null-versus-absent rule: an option field holding none is present.
- **N11 Time** is the logical clock only: `Tick` events and `after n`
  in `select`. No construct reads a wall clock.

The built-in methods on text and lists (`len reverse sort sum min max upper
lower strip chars words tostr toint abs all any unique flatten isdigit
isalpha isupper islower get contains count index_of split join startswith
endswith pow floordiv pmod zip replace append with map filter fold sort_by`)
have exactly the semantics of the Form primitive of the same name in num.rs
and ir.rs (`Prim`, `Map`, `Filter`, `Fold`, `SortBy`), at the RCE static type.
`sort` and `sort_by` are stable.

**Read receipt.** Which JavaScript function is named for rounding to binary32?

<!-- body-sha256: bf6c8917142ebcd54bfd4b3a40788ad0bc8b01456033e77eb5ae9142e45deb35 -->
