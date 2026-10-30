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
- **λ** maps each step to its output sequence O\*: values `emit`ted, and for an
  event that does not apply the record `Kept(X, payload)`.
- **D** is the log: the sequence of admitted events e₁, e₂, …, each entry with
  its outcome (applied, or kept with X). The fold law: the state after the
  log is sₙ = foldl(δ₁, s₀, D) where δ₁ is the state component of δ.

**Canonical signature (normative).** This is the one signature; the engine
profile (D.6), the engine IR (`FormIr::step`) and the language interpreter
are instances of it, and any other text (FORM.md, papers) defers to it.
S = Σc × Σd (control regions, data: regions, clock, input count,
relations); δ : S × E → S × O\* total; λ = π₂ ∘ δ; δ₁ = π₁ ∘ δ. An entry of
D is the record {seq, event, prev_hash, state_hash, hash} where `seq` is
the 0-based index, `event` is {type, payload}, `prev_hash` is the `hash`
of the previous entry (64 zeros for the first), `state_hash` is the
SHA-256 of the canonical JSON of the *positional form* of the state after
the step, and `hash` is the SHA-256 of the canonical JSON of {event,
prev_hash, seq, state_hash} with `event` in its positional form. The
positional form (T7) replaces every declared name by a position: the
state is the list of its regions in declaration order, each the list of
its field values in declaration order; a record of a declared struct,
event or error is the list of its field values in declaration order; a
variant record is `[k, [fields]]` with k the variant's index in its
kind; an event is {type, payload} with `type` the event's index among
the module's events and `payload` the positional record; every other
value embeds as the table below says. A record of a type the module
does not declare (a `host.*` record, a label) and the built-in `Tick`
keep their names, since no renaming of the program touches them. The
stored `event` and the readable `state` keep their names, so a log is
readable without the module; a log or trace that repeats the hashed
forms does so under `hashed` ({event, state}), which is derived. A log
may also carry `outcome`, `out` and `state`; these are derived and must
equal replay (T1), so they are not hashed. An event that does not apply (T8) is
recorded with its unchanged state and ends the example log. Both
producers are tested against this rule
(`spec/checker/tests/form_signature.rs`, `tests/form_signature.rs`), and
`tests/nameless.rs` checks that a renamed program has the same hashes;
`RCE/Positional.lean` proves that the positional form, and so the state
hash and the entry chain under any hash function, is unchanged by the
renamings of 10.6 (T7).

Values ⟦τ⟧ are the Form's JSON-shaped values (form/types.rs): integers
(i64), floats (finite f64), text, bool, null, lists, records. The embedding
of RCE types into them is fixed:

| RCE type | Form value |
|---|---|
| bool | bool |
| iN, uN, usize | int, with the range of the RCE type checked at every write |
| f64 | float, finite, type kept (6.3 N1) |
| text | text (sequence of Unicode scalar values) |
| () | null |
| (a, b, …), [τ; n], List[τ] | list |
| Set[τ] | list, sorted by canonical order, no duplicates |
| Map[k, v] | list of [k, v] pairs sorted by canonical order of k |
| struct / state record | record (keys sorted, as canonical JSON) |
| kind variant V(f: …) | record {"$v": "V", f: …}; unit variant: {"$v": "V"} |
| τ? | null for none, the value for some (and `some(none)` requires change: τ may not itself be an option, W-TY-2) |
| chan τ, actor handle | record {"$chan": id} / {"$actor": id}, id the content hash of 6.6 |

Canonical order of values (used by Set, Map and select): null < bool < int <
float < text < list < record; ints by value; floats by IEEE 754 totalOrder
(`-0.0 < 0.0`); text by code point; lists and records lexicographically. It
is a total order (NaN does not exist, N3 F4) and it agrees with state
identity: two values are equal in it exactly when their canonical JSON is
equal. Derivation: Set and Map keys are compared by this order, so an order
that called two observable values equal (`-0.0` and `0.0` under IEEE `==`,
or `1` and `1.0` by numeric value) would silently merge distinct keys; the
type tag and totalOrder keep every distinct state a distinct key.

#### 6.1.1 Canonical JSON bytes (normative)

Every hash of 6.1 (`state_hash`, `hash`), the value limit of N9 and every
trace written as JSON are taken over the same byte sequence, the canonical
JSON of a Form value. This subsection defines it byte for byte. It is
derived from one requirement: two implementations must produce the same
bytes for the same value, and the bytes must decode to that value and to no
other (6.3 N1), because replay rebuilds state from D.

The output is a UTF-8 byte sequence with no byte order mark, no whitespace
outside text and no trailing line end. By the kind of value:

1. **null, bool.** `null`, `true`, `false`.
2. **int.** The exact value in decimal: an optional `-`, then digits with no
   leading zero (`0` is `0`), no `+`, no exponent, no fraction. The range is
   -2^63 to 2^64 - 1 (6.3 N2). An integer is never written as a float and a
   float is never written as an integer.
3. **float.** Finite binary64 only (NaN and infinities are not Form values,
   6.3 N3). Let x be the value.
   - x = 0: `0.0`, or `-0.0` when the sign bit is set.
   - Otherwise take the shortest decimal digit string d1 d2 … dn that reads
     back as x (round to nearest, ties to even). When two digit strings of
     that length both read back as x, take the one closer in value to x; when
     they are equally close, take the one whose last digit is even. Let e be
     the decimal exponent of d1 (x = ±d1.d2…dn × 10^e).
   - If 10^-4 ≤ |x| < 10^16: decimal notation with no exponent. A sign `-`
     when x < 0, then the digits with the point placed after e + 1 digits;
     zeros are added to fill the integer part, and `0.` followed by
     -e - 1 zeros starts a value below 1. The result always has at least one
     digit after the point (`10.0`, not `10`).
   - Otherwise: a sign `-` when x < 0, d1, then `.d2…dn` only if n > 1, then
     `e`, then e as a decimal integer with `-` when negative and no `+` and
     no leading zero (`1e16`, `1.5e-7`, `5e-324`).
4. **text.** `"`, then each Unicode scalar value in order, then `"`. `"` is
   written `\"`, `\` is `\\`, U+000A `\n`, U+000D `\r`, U+0009 `\t`. Any
   other value below U+0020 is written `\u` and four lowercase hexadecimal
   digits, so U+0008 is `\u0008` and U+000C is `\u000c`. Every other scalar
   value, including U+007F, U+2028, U+2029 and `/`, is written as itself in
   UTF-8. Text is not normalised: `é` (U+00E9) and `é` (U+0065 U+0301) give
   different bytes.
5. **list.** `[`, the canonical JSON of each element in order separated by
   `,`, `]`.
6. **record.** `{`, then each field as the canonical JSON of its key (as
   text) `:` the canonical JSON of its value, separated by `,`, then `}`.
   Fields are in ascending order of key by Unicode code point (equivalently,
   by the bytes of the UTF-8 encoding). Keys are distinct.
7. **Set and Map.** As their list embeddings of 6.1, in the canonical order
   of 6.1, then rule 5.

**Relation to RFC 8785.** The JSON Canonicalization Scheme (RFC 8785,
sections 3.2.2 and 3.2.3) fixes whitespace, literals, the escaping of
control characters and the sorting of object members, and takes number
serialisation from ECMAScript. Rules 1, 5 and the absence of whitespace
agree with it. Rules 2, 3, 4 and 6 differ, each for a stated reason, so an
RFC 8785 library does not produce these bytes and must not be used to hash
a Form value:

| Case | RFC 8785 | This section | Reason |
|---|---|---|---|
| float with no fractional part | `10` | `10.0` | an int and a float must decode differently (N1) |
| negative zero | `0` | `-0.0` | the sign is observable state (N3 F6) |
| 10^-6 and 10^21 | `0.000001`, `1e+21` | `1e-6`, `1e21` | one fixed rule with no `+`; the thresholds are 10^-4 and 10^16 |
| integers above 2^53 | not exact in an ECMAScript number; RFC 8785 section 3.1 says to use text | decimal digits, exact to 2^64 - 1 | the language has u64 and i64 (N2) |
| U+0008, U+000C | `\b`, `\f` | `\u0008`, `\u000c` | one escape form for every control character not named in rule 4 |
| member order | UTF-16 code units | code points | the same unless two keys first differ at a supplementary-plane character against one in U+E000 to U+FFFF (the RFC's own sort data differs) |

**Golden vectors.** `spec/data/canonical-json-vectors.json` holds 77
vectors: a value, the exact canonical text, and the SHA-256 of its UTF-8
bytes. It covers every rule above, the boundaries 10^-4 and 10^16, the
extremes of i64, u64 and binary64, the sort data of RFC 8785 section 3.2.3,
five of the number rows of RFC 8785 Appendix B, and the
bytes behind one `state_hash` and one `hash`. Two programs check it:
`spec/checker/tests/canonical_json.rs` against the reference implementation,
and `scripts/canonical-json-vectors.py --check`, an encoder written from
this subsection alone. A third implementation is conformant when it
reproduces every canonical text and digest.

**Resolved items.** Two differences between the reference implementation
and this subsection were found by the vectors and are fixed.
- *Tie rule.* Rust's shortest-digit routine (`{:e}`) breaks an exact tie
  between two shortest candidates to the odd digit: for 1424953923781206.25
  (binary64 `0x43143ff3c1cb0959`) it gave `1424953923781206.3`, where the rule
  of rule 3 (as ECMAScript's `String(x)` and Python's `repr`) gives
  `1424953923781206.2`. The reference now takes the shortest digit count n
  from `{:e}` and then the n-digit decimal closest to x, ties to even
  (`interp::shortest_e`, used by `float_json`, `float_text` and
  `emit/rt.rs`). The vectors `float-1424953923781206.2` and the four
  `float-tie-*` vectors (round down, round up, and their negatives) pass.
- *Second writer.* `rce trace --json` now writes text through the same
  `jstr` as the hashed writer, so U+0008 and U+000C print as `\u0008` and
  `\u000c` and a trace shows the bytes that were hashed. The test
  `the_trace_writer_agrees_with_the_hashed_writer` covers every vector.
A decision between this text-based form and a length-prefixed binary form is
deferred until measured (the form defined here is the one all existing
hashes use).

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
always keeps the state).

- **E1 Fuel exhaustion is uncatchable.** Running out of fuel is not a
  declared error and not a trap a program can observe: no `catch`, `try`
  or `try e else X` handles it, it propagates outward through every
  construct, and it keeps the state exactly as a trap does (T8). The
  derivation: if a handler could observe exhaustion, a program could
  return a different value on a larger budget, and a completed result
  would depend on fuel. The fuel laws require that it does not: more fuel
  never changes a completed result (`RCE.Core.eval_mono`,
  `eval_fuel_indep` in `lean/RCE/Core/Fuel.lean`). A conforming run of
  `(e catch |x| h)` where `e` exhausts its fuel ends in exhaustion, never
  in `h`.

**`&mut` write-back (normative).** A call that returns writes back each `&mut`
argument and a `&mut self` receiver. A call that ends by `fail` leaves each of them as it was at
the call, provided every effect since entry is undoable, and requires change otherwise (F.4.4, W-TX-1); the
callee's writes made before the failure are undone. Atomicity of the event is separate: a `trap`, or a `fail`
that leaves the step, keeps the state: the state is the pre-state (T8). The earlier rule of this text
wrote back on every outcome, including `fail`; it is retired because F.4.4 makes a failed call a no-op, which is
what a caller can reason about. T33 now proves the restore rule (`WriteBack.t33_failure_restores`,
`t33_return_writes_back`); nothing about the retired rule remains in the proved tree. The program emitters implement the
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
that would evaluate it eagerly; it requires change, with a diagnostic, and is never
given a different meaning. Tested on every target
(`spec/checker/tests/write_back.rs`). Values of type `τ ! ε` are `val(v)` or
`err(X, v)`; they are first-class only until a `try` or `catch` consumes them.

### 6.3 Edge semantics of primitives (normative, from form/num.rs)

These are the Form's own rules; every implementation reproduces them
exactly.

- **N1 Numbers.** A Form number is an int (i64, or u64 per N2) or a float
  (finite f64). The type is part of the value and is kept across every
  boundary: a float with no fractional part stays a float (`10.0` is
  written `10.0` in canonical JSON, `10` is an int), and `-0.0` keeps its
  sign (N3 F6). Derivation: encoding must be injective and round-trip,
  `decode(encode(v)) == v`, because replay rebuilds state from D. Turning
  `10.0` into `10` at a boundary would decode as an int, changing the
  static type a later step sees and the bits of later float arithmetic,
  so no normalization is applied. RCE's static types make the int/float
  distinction visible to the programmer, and canonical JSON keeps it
  visible in D.
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
  - **F6 Signed zero is a distinct state.** `-0.0` arises by the
    standard's sign rules, which are exact, so every target produces it in
    the same places (vector `Mul(0.0, -1.0)`). It is observable: `tostr`
    prints `-0` (f64 and f32 alike, so text round-trips), and the sign
    decides later results through the same exact rules. The state hash
    must be injective on observable state, so canonical JSON keeps the
    sign and a state holding `-0.0` hashes differently from one holding
    `0.0`; the canonical order (6.1) is totalOrder, so they are distinct
    map keys. `-0.0 == 0.0` is still true: `==` is IEEE comparison.
  - **F7 Exception flags are not state.** Inexact and underflow raise
    nothing (F1 already fixes the result); overflow, divideByZero and
    invalid are the trap of F4. No program reads a flag.
  - **F8 Exact operations.** `floor`, `ceil` and `round` are the standard's
    roundToIntegral operations toward negative, toward positive and
    ties-away (`(2.5).round()` is `3`, `(-2.5).round()` is `-3`); `abs`
    clears the sign. `min` and `max` follow N5. None of these rounds.
  - **F9 Transcendentals are correctly rounded RCE code, or absent.**
    IEEE 754-2019 recommends but does not require correctly rounded `exp`,
    `log`, `sin`, `pow` and the like, and platform math libraries differ
    in the last bit, so no runtime takes one from its platform library and
    no Form op names one. A transcendental exists only as RCE source in
    the standard library module `math` (rce-stdlib `lib/math/math.rce`),
    built from the operations of F1 to F8 and integers, so every runtime
    carries the same algorithm and the result is the same bits on every
    target (vector parity: Rust, Python, JavaScript, Scala). Each function
    must be correctly rounded by construction: algebraic ones (`cbrt`,
    `hypot`) compare the exact value with the rounding midpoints in exact
    arithmetic (TwoSum, Veltkamp/Dekker TwoProduct, Shewchuk expansions;
    exact ties round to even); transcendental ones use Ziv's strategy over
    exact integers (the value is bracketed with an explicit error bound and
    the precision grows until both ends round alike; a transcendental
    value of a double is never a midpoint), `sin` and `cos` with
    Payne-Hanek reduction exact for every finite argument. Each is checked
    bit for bit against MPFR on the CORE-MATH worst cases. Present:
    `sqrt` (F1), `cbrt`, `hypot`, `exp`, `log`, `log2`, `log10`, `sin`,
    `cos`; any other (`pow`, `tan`, `atan2`, ...) is absent until it meets
    the same rule. A result that overflows traps (F4); `log` of x ≤ 0
    traps (also `log2`, `log10`). Device versions stay output only or
    verified (HC-13).
  Float *text* (the result of `tostr` and of `Show` on a float,
  `interp::float_text`) is the shortest round-trip digits; decimal notation
  for 1e-5 ≤ |x| < 1e16, else `d.ddde±N` with `+` on positive exponents
  (`1e+300`, `1.5e-7`). It is a program-visible value and is never hashed.
  Canonical JSON (6.1.1) is a different, separately specified byte form:
  it has one float form, `interp::float_json`, used for
  hashes and for every trace and log written as JSON, so a trace shows the
  bytes that were hashed: shortest round-trip digits, decimal with at
  least one fractional digit for 1e-4 ≤ |x| < 1e16 (`10.0`, `0.0001`),
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
- **N6 Equality** is structural value equality within one static type;
  `1 == 1.0` is a type error (the type is part of the value, N1). Float
  `==` is IEEE comparison, so `-0.0 == 0.0` although the two are distinct
  states (N3 F6).
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
  Display writes text in double quotes, with U+0000, tab, LF, CR, `"` and
  `\` as `\0`, `\t`, `\n`, `\r`, `\"` and `\\`, each code point of the
  *display escape set* as `\u{h}` (h its lower-case hexadecimal digits,
  without leading zeros), and every other code point as itself. The display
  escape set is the 911 ranges of code points listed in
  `spec/data/display-escape.json` (`interp::display_escaped`; the emitter
  writes the same ranges into each emitted runtime): the control
  characters, the code points that are not printable and the grapheme
  extenders, as Rust 1.98.1 escapes them in the Debug form of a string from
  its tables of Unicode 17.0.0, enumerated once. The file, not a Unicode
  version, defines the set.
  **Unicode pin.** RCE pins Unicode 18.0.0
  (https://www.unicode.org/versions/Unicode18.0.0/) and reads no Unicode
  table at run time: scalar values and UTF-8 do not change between
  versions, case and class operations are ASCII, and the two sets of code
  points a result depends on, White_Space and the display escape set, are
  enumerated above rather than taken from the toolchain. A newer Rust,
  Python, JavaScript or JVM therefore cannot change a result;
  `tests/unicode_pin.rs` checks both sets and scans the runtimes for calls
  that would read the toolchain's tables, `{:?}` of a text among them, and
  `tests/emit.rs` compares the display of both ends of every range, and of
  the code points beside them, on each emitted target with the reference.
  Moving the pin is a spec change that must leave the enumerated sets
  unchanged or bump the draft.
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
  (`trap("value limit")`, the state is kept, the value is not recorded).
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

<!-- body-sha256: 0dc0c2af67c908c2000eb760a3f14aa6dabf408b6729723fe42b4fcd39f788a8 -->
