## 5. Static semantics

This section gives the *static semantics*: the rules checked before a
program runs. A *type* is the set of values an expression may have;
*ownership* says which variable is responsible for a value, and a *borrow*
is a temporary reference to it. Example: after `let b = a` where `a` is a
`List`, using `a` again requires change because the list moved to `b`.

### 5.1 Types

```
τ ::= bool | i8 | i16 | i32 | i64 | u8 | u16 | u32 | u64 | usize | f64 | f32
    | text | ()                                  base types
    | (τ₁, …, τₙ)            n ≥ 2               tuple
    | [τ; n]                 n a comptime usize  array
    | List[τ] | Map[τ_k, τ_v] | Set[τ]            owned collections
    | [τ]                                         slice (only behind a borrow)
    | τ?                                          option (= kind Option[τ])
    | τ ! ε                                       error union, ε a set of errors
    | &τ | &mut τ                                 borrows (second class, 5.6)
    | τ @r                                        τ allocated in region r
    | chan τ                                      channel handle
    | iter τ                                      iterator (5.14)
    | fn(τ₁, …, τₙ) -> τ                          function value (no capture of borrows)
    | N[τ₁, …, τₙ]                                nominal: struct, kind, actor, state
    | α                                           type parameter
```

`usize` is a u64 used for lengths and indexes. `f32` (draft 0.5) is computed
in f64 and rounded once to f32 after each operation (6.3 N7), so every target
gives the same bits. Float literals are f64; an f32 value is written
`x as f32`, and f32 and f64 never mix without a cast. `Box[τ]` is `List`-like owned indirection of one
element, needed only for recursive kinds; a `&Box[τ]` (or `&mut Box[τ]`)
argument is accepted where `&τ` (`&mut τ`) is expected (deref coercion,
draft 0.4). Every type has a **kind** of
ownership (5.5): `Copy`, `Move`, or `Borrow`.

**Integer ranges.** An integer type denotes the mathematical integers in its
range: iN = [−2^(N−1), 2^(N−1)−1], uN = [0, 2^N−1]. Arithmetic is checked
against the *declared* type's range (6.3). Arithmetic on u64 and usize uses
the Form's unsigned operations (6.3 N8, draft 0.5), so the whole range
[0, 2^64) is available; drafts up to 0.4 trapped any u64 result at or above
2^63 (D7).

**Text is byte-indexed (draft 0.5).** `s.len_bytes()` is the number of UTF-8
bytes (`s.len_chars()` the number of characters; unqualified `len` on text requires change, W-LEN-1), `s[i]` is the byte at offset i as a `u8`, and `s[a:b]` is the text
between byte offsets a and b, which must both be character boundaries
(otherwise `trap("not a char boundary")`). Iterating `for c in s` still
yields one-character texts. Go, Rust and Zig all index text by byte; the
draft 0.4 code-point indexing made `s[i]` O(n).

**Lane-wise arithmetic (draft 0.5).** For arrays `a, b : [τ; n]` of one
numeric τ and one comptime length n, `a + b`, `a - b`, `a * b`, `a / b` (and
`a % b` for integers) apply the operator lane by lane and have type
`[τ; n]`. Each lane follows the scalar rules (overflow and division by zero
trap). Different lengths, or a non-numeric τ, require change (W-SIMD-1). An
implementation may use vector instructions; the result is defined by the
lanes, so it is the same on every target. A list literal is a `List` unless
it meets an array type of the same length, as an integer literal is i64
unless it meets another integer type.

**Type aliases.** `type A = τ` (and `type A[α₁, …, αₙ] = τ`) without
`where` gives τ a second name: A, with its parameters substituted, is the
type τ in every rule. It is not a nominal type, since N above ranges only
over structs, kinds, actors and states. So a parameter `c: A` with
`type A = chan τ′` is a `chan` parameter and makes its function a task
function (5.2), and a value of type A is accepted wherever τ is and the
reverse. Example: with `type Pipe = chan (u64, text)`, `fn feed(p: Pipe,
n: u64)` may send on p and be spawned (`spec/checker/tests/alias.rs`).

**Refined types (extension, all versions).** `type R = τ where p`
declares R as the values of τ that satisfy the predicate p, in which
`self` names the value. τ is an integer type, `text`, or a `List`, `Map`,
`Set` or array type. p is in a decidable fragment: comparisons (`<`, `<=`,
`>`, `>=`, `==`, `!=`) between `self` (integer τ), `self.len()` (text,
in UTF-8 bytes as in 5.15, or a collection) and integer literals; the
character classes `self.isdigit()`, `self.isalpha()`, `self.isspace()`,
`self.isempty()` of a text; and `and`, `or`, `not`. Anything else is
W-REF-1. `where` is contextual: no type is followed by a name, so the
word stays an ordinary identifier. A refined value is checked exactly
where a value is checked against its declared integer range (6.3, 6.4
step 3): at a typed `let` or `var`, a parameter, a field or payload
constructor, a field initializer and the write-back of a state field.
The check fits the value to τ first, then evaluates p; if p is false the
result is `trap("refinement: value is outside R")`. So an event whose
payload is outside its refined type does not apply and keeps the state
before any step runs, as an out-of-range integer payload does, and a step
whose new state breaks a field's predicate keeps the state (6.4, T8).
`R <: τ`: a refined value is used wherever τ is. Derivation: validation
that the type does not record can be skipped (soundness), a predicate
checked at one boundary has one outcome (determinism), and the
refinement reuses the one boundary the language already has, the range
check, rather than a second mechanism (one meaning per construct). The
fragment is decidable in time linear in p and in the value, so the check
has a fixed cost class (its work is bounded before it runs); full dependent types are
excluded by the same derivation, since an undecidable obligation would
make acceptance itself unbounded. Example: `type Pct = u8 where self <=
100`; `event SetVolume(pct: Pct)`; `SetVolume(pct: 101)` keeps the state with
`refinement: value is outside Pct`, and a step computing `level + 60` on
`level: Pct = 60` keeps the state. Lean: `RCE/Refine.lean`
(`holds_total`, `fit_sound`, `refine_preserved`,
`refine_kept_unchanged`, `admit_sound`; T35).

**Error sets.** In `τ ! ε`, ε is a finite set of declared `error` names,
written as one error name or as the name of a `kind` whose variants are all
errors (`type ParseErrs = …` is not needed; declare `kind ParseErrs` with
variants `ParseFailed(at: u32)` and `Empty`). Subtyping:
`τ ! ε₁ <: τ ! ε₂` iff ε₁ ⊆ ε₂, and `τ <: τ ! ε` (a plain value is a
success).

### 5.2 Contexts and modes

A typing context Γ maps each variable to (τ, m) with mode
m ∈ {let, var, field} and a liveness flag (live, moved). A step body runs in
context `step(R)`, a function body in `fn`, a comptime block in `comptime`,
and an actor handler in `actor(A)`. Effects are tied to the context:

| Context | may `emit` | may `fail` | may `spawn`/send/recv/`select` | may read state |
|---|---|---|---|---|
| `fn` (pure) | no | yes (if return type is `τ ! ε`) | no | no |
| task `fn` (has a `chan` parameter) | yes | yes | yes | no |
| `fn main` | yes | yes | yes | no |
| `step(R)` | yes | yes | send only (to actors and declared channels) | fields of R only |
| `actor(A)` handler | yes | yes | send only | fields of A only |
| `comptime` | no | yes (compile error) | no | no |
| call a foreign `extern` fn (5.11) | only from `fn main` and task functions | | | |

A function called from a step does not inherit the step's effects; only the
step body itself emits or sends. A function with a `chan` parameter is a
**task function**: it may `emit`, send, receive, `select` and `spawn`, and it
can be called (or spawned) only from `main` or another task function. Every
other `fn` is a pure function of its arguments; a pure function, a step or a
handler cannot call a task function (W-EFF-1). `spawn f(ā)` requires f to be a
task function (W-SPAWN-1). (Change S2: draft 0.1 made every `fn` but `main`
pure, which left nothing that `spawn` could start.)

**Tail values.** In a block-bodied function, if the last statement is an
expression statement, or a `match` or `if`/`else` whose every branch ends in
one (recursively), those expressions are the function's result and are typed
against its return type ([Tail]; change S5). Elsewhere an expression
statement's value is discarded, and an error union may not be discarded.

### 5.3 Typing expressions

Judgment: Σ; Γ; Δ ⊢ e : τ ⊣ Γ′, under declarations Σ, context Γ and live
borrows Δ, expression e has type τ and leaves context Γ′ (Γ′ differs from Γ
only by variables marked moved).

```
  (x : τ, m, live) ∈ Γ     τ Copy
  ─────────────────────────────────── [Var-Copy]
  Σ;Γ;Δ ⊢ x : τ ⊣ Γ

  (x : τ, m, live) ∈ Γ     τ Move     x ∉ dom(Δ)
  ─────────────────────────────────────────────── [Var-Move]
  Σ;Γ;Δ ⊢ x : τ ⊣ Γ[x ↦ moved]

  n literal    n ∈ range(ι)    ι an integer type expected by context (default i64)
  ───────────────────────────────────────────────────────────────── [Int]
  Σ;Γ;Δ ⊢ n : ι ⊣ Γ

  Σ;Γ;Δ ⊢ a : ι ⊣ Γ₁    Σ;Γ₁;Δ ⊢ b : ι ⊣ Γ₂    ⊕ ∈ {+,-,*,/,%}    ι numeric
  ───────────────────────────────────────────────────────────────── [Arith]
  Σ;Γ;Δ ⊢ a ⊕ b : ι ! {Overflow?, DivZero?} ⊣ Γ₂
```

Arithmetic failure is not part of the static error set the programmer
declares: it is a **trap** (6.5), the Form's defined `err` outcome, and keeps
the state. This matches the Form, where integer overflow is an error, never a
wrap. Both operands must have the same numeric type; there is no implicit
widening (use `as`).

```
  Σ;Γ;Δ ⊢ e : τ ! ε ⊣ Γ′    the enclosing function returns σ ! ε′    ε ⊆ ε′
  ───────────────────────────────────────────────────────────────── [Try]
  Σ;Γ;Δ ⊢ try e : τ ⊣ Γ′

  Σ;Γ;Δ ⊢ e : τ? ⊣ Γ₁  (or τ ! ε)    Σ;Γ₁;Δ ⊢ E : error ⊣ Γ₂    E ∈ ε′
  ───────────────────────────────────────────────────────────────── [Try-Else]
  Σ;Γ;Δ ⊢ try e else E : τ ⊣ Γ₂

  Σ;Γ;Δ ⊢ e : τ ! ε  (or τ?) ⊣ Γ₁    Σ;Γ₁,x:ε;Δ ⊢ h : τ ⊣ Γ₂
  ───────────────────────────────────────────────────────────────── [Catch]
  Σ;Γ;Δ ⊢ e catch |x| h : τ ⊣ Γ₂

  Σ;Γ;Δ ⊢ c : bool ⊣ Γ₁   Σ;Γ₁;Δ ⊢ a : τ ⊣ Γ₂   Σ;Γ₁;Δ ⊢ b : τ ⊣ Γ₃
  ───────────────────────────────────────────────────────────────── [If-Expr]
  Σ;Γ;Δ ⊢ if c then a else b : τ ⊣ Γ₂ ⊓ Γ₃
```

Γ₂ ⊓ Γ₃ marks a variable moved if it is moved in either branch (a moved-on-
one-path variable cannot be used after the join).

```
  f : fn[ᾱ: B̄](l₁ x₁: τ₁, …, lₙ xₙ: τₙ) -> σ ∈ Σ
  θ = unify(ᾱ, …)     θ(ᾱ) satisfies B̄     Σ;Γᵢ₋₁;Δ ⊢ aᵢ : θτᵢ ⊣ Γᵢ
  labels of the call equal l₁…lₙ (labels may be omitted only when lᵢ = xᵢ)
  ───────────────────────────────────────────────────────────────── [Call]
  Σ;Γ₀;Δ ⊢ f(a₁, …, aₙ) : θσ ⊣ Γₙ

  Σ;Γ;Δ ⊢ e : K[τ̄] ⊣ Γ′   arms pᵢ ⇒ bᵢ    Σ ⊢ pᵢ : K[τ̄] ▷ Γᵢ
  Σ;Γ′,Γᵢ;Δ ⊢ bᵢ ok     covers(K, p₁…pₙ)
  ───────────────────────────────────────────────────────────────── [Match]
  Σ;Γ;Δ ⊢ match e { pᵢ ⇒ bᵢ } ok
```

`covers` is the standard exhaustiveness relation over constructors
(Maranget 2007): every value of K[τ̄] matches some unguarded arm, or some arm
whose guard is the literal `true`. Guards do not count toward coverage.
Integer and text scrutinees need a `_` arm (their domains are not
enumerated) unless every value of a bounded integer type is covered by ranges.

Statements: Σ; Γ; Δ ⊢ s ⊣ Γ′. `let p = e` binds immutably; `var x = e`
binds mutably; `place op= e` needs `place` rooted at a `var`, a `&mut`
parameter, or (in a step) a field of the stepped region. `return e` needs `e`
to have the function's return type (or a subtype). `fail E(…)` needs `E` in
the function's declared error set (in a step: any declared `error`).

### 5.4 Types of declarations

- `state R` with fields fᵢ : τᵢ = eᵢ: every τᵢ is an **owned value type**
  (no borrows, no channels, no region types, no functions); every eᵢ is a
  comptime constant expression (W-STATE-2). R's type is the record
  {fᵢ : τᵢ}.
- `event E(fᵢ : τᵢ)` and `error E(fᵢ : τᵢ)`: payload types are **data types**:
  base types, tuples, arrays, `List`, `Map`, `Set`, options, structs and kinds
  built from data types. Events may additionally carry `chan τ` (a reply
  channel) when received by an actor.
- `step R on p if g -> e`: in context `step(R)`, the fields of R are bound as
  `var`, p is typed against the event sum, g : bool, and e must have the type
  of R's single field (only allowed when R has exactly one field; W-STEP-3).
  With a block body, the body is a statement list; the final state is the
  values of the field variables at the end of the block.
- `actor A` with fields and handlers: like a `state` whose steps are the
  handlers, plus an address type `A` (a `Copy` handle) that supports
  `a <- msg`.

### 5.5 Ownership

Every value has exactly one owner: a variable, a field, a collection slot, a
channel buffer, or the state. Types fall in three ownership kinds:

```
Copy   : bool, integers, f32, f64, (), chan τ, actor handles, fn values,
         and tuples/arrays/structs/kinds all of whose components are Copy
         and whose size is fixed.
Move   : text, List, Map, Set, Box, options/tuples/structs/kinds containing a Move component.
Borrow : &τ, &mut τ, [τ] (slices exist only borrowed).
```

Rules:

```
  O1  Use of a Move variable moves it ([Var-Move]); a moved variable cannot be
      read, borrowed, or moved again until reassigned.
  O2  Assignment `x = e` to a var drops the old value of x first and makes
      x live.
  O3  `copy(e)` duplicates a value of any data type (deep copy); it is the
      only way to duplicate a Move value. (Copy-on-write is an
      implementation choice, not visible.)
  O4  At the end of each block, live owned values introduced in it are
      dropped in reverse order of introduction. Drop has no user-visible
      effect (there are no destructors), so drop order is unobservable.
```

### 5.6 Borrowing

RCE borrows are **second class** (as in Hylo and Swift's `inout`): a borrow
can be created only as a function argument or as the operand of a `for`, a
`match`, a method receiver, or a `let` whose scope ends before any
conflicting use. A borrow cannot be stored in a field, a collection, a
channel message, an event, a state, a closure that escapes, or be returned:
with one exception: a function whose return type is a borrow (`-> &[T]`) must
return a borrow derived from a parameter, and the result is tied to that
parameter (R-RET). This removes lifetime annotations entirely.

Δ is a set of live borrows `(ℓ, k, path)` with k ∈ {shr, mut}. Paths are
`x`, `x.f`, `x[_]` (any element), ordered by prefix (`x` ⊑ `x.f`).

```
  path p rooted at a live variable    no (ℓ′, mut, q) ∈ Δ with p, q overlapping
  ───────────────────────────────────────────────────────────── [Borrow-Shr]
  Σ;Γ;Δ ⊢ &p : &τ  with Δ′ = Δ ∪ {(ℓ, shr, p)}

  p rooted at a var or &mut     no (ℓ′, _, q) ∈ Δ with p, q overlapping
  ───────────────────────────────────────────────────────────── [Borrow-Mut]
  Σ;Γ;Δ ⊢ &mut p : &mut τ  with Δ′ = Δ ∪ {(ℓ, mut, p)}

  writing, moving or dropping p requires no (ℓ′, _, q) ∈ Δ overlapping p   [No-Alias-Write]
  reading p requires no (ℓ′, mut, q) ∈ Δ overlapping p, q ≠ p's own borrow   [No-Mut-Read]
```

Two paths overlap iff one is a prefix of the other; `x[i]` and `x[j]` always
overlap (indexes are not compared statically). The borrows among the
arguments of one call are made together, so `f(&mut s.a, &mut s.b)` complies
with the Form, while `f(&mut s.a, &mut s)` and `f(&mut xs[i], &mut xs[j])`
require change under W-OWN-2. A borrow ℓ ends at the end of the call or
statement that created it (R-END), for a `let` borrow at the last use of its
name, or for R-RET at the last use of the returned value. Because borrows never
outlive the statement that made them except through a `let` borrow or R-RET,
Δ is empty at every statement boundary outside those ranges, which makes the
check a local, linear pass.

**`let` borrows.** A borrow made by `let a = &p` or `let a = &mut p` is live
from the `let` to the last statement of the same block that uses `a` or a
name bound from `a`. In that range a use of a path overlapping p that the
rules above forbid (writing, moving or `&mut`-borrowing it under either kind
of borrow, reading or borrowing it under a `&mut` borrow) requires change
under W-OWN-2: `let a = &mut n; n = 5; bump(a)` requires change, and
`let a = &mut n; bump(a); n = 5` complies with the Form. A write through `a`
is a write to the place p (`writeVia` of the Lean model of T6), so
`let a = &mut n; bump(a); emit n` emits 1. An index in p is evaluated once,
at the `let`.

**Shared XOR mutable (theorem T6 relies on this):** at every program point,
for every path p, either Δ holds no mut borrow overlapping p, or it holds
exactly one and no shared borrow overlapping p.

**Not relaxed in draft 0.9.** A limited relaxation, in which a reference is
invalidated only below steps that are not independent of it (the group
borrowing of the Valen language), is sound in the form studied, but its
preservation of T6 is not proved. The rules above are unchanged; the
obligation is T56 (Appendix F, F.9, stated and unproved), and the rule is
adopted only after that lemma is proved in Lean. Relaxing W-PAR-1 is not
sound and is excluded (F.2).

**Read receipt.** What is the name of the example kind used to illustrate error sets?

<!-- body-sha256: 857c1b43e187f03f4ad2efc72660d85d8627a71a67ebec9aa7f63636561fdb14 -->
