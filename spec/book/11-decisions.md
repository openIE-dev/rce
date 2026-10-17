## 9. Design decisions for review

This section lists the design choices the spec makes, each with a one-line
reason, and the changes made in each draft. Each decision can be reviewed
and reversed on its own. Example: decision 45 makes text indexed by UTF-8
byte, as in Go, Rust and Zig.

Gaps in SURFACE.md that this spec decides. Each is one line of rationale;
each can be reversed independently.

1. **Layout by indentation, spaces only, tabs refused** (Python), with
   brackets suspending layout. Keeps SURFACE.md's look; tabs removed to make
   layout unambiguous.
2. **`#` comments only**, matching the prototype; no block comments.
3. **Case-sensitive event names in examples.** The prototype wrote
   `example inc` for event `Inc`; the spec requires `Inc` (no hidden case
   mapping).
4. **Several `state` regions per program**, each step reading only its own
   region (pre-state, ASM), events atomic across regions. Matches Form law 2
   and the "rules read their own sink" limit.
5. **Totality = every event is handled somewhere; unmatched regions are
   unchanged.** Stricter than "every (region, event) pair", looser than "every
   event in every region"; catches typos without boilerplate.
6. **Errors refuse the event and leave state unchanged** (`example … ->
   refused: X`), and are logged and output, never thrown.
7. **Traps (overflow, division by zero, non-finite, index, size, loop bound)
   are Form errors, not part of declared error sets**; `catch` cannot hide
   them.
8. **Recursion only where it provably ends** (revised in 0.4 and later: a
   self-call passes a constructor field of a matched parameter, or a measure
   falls in layers, including mutual recursion, or the cycle carries a
   declared `bound`; 6.7 says exactly which) and **every loop bounded**
   (default 1,000,000, `bound n` to change), so δ is total by construction.
   (Originally structural recursion only, with mutual recursion refused.)
9. **Second-class borrows, no lifetime syntax.** Borrows only in parameters,
   receivers, loop/match operands and short `let`s, plus borrow returns tied to
   a parameter. Simpler than Rust, still no GC and no aliasing of mutable
   state.
10. **`struct` only, no `class`**; value semantics, explicit `copy()` for deep
    duplication of non-Copy values.
11. **Integer types checked against their own range**, lowered to the
    Form's i64, except u64, which uses the Form's unsigned operations and is
    exact over [0, 2^64) (6.3 N2, T14; draft 0.5). No implicit widening; no
    literal suffixes. (Originally u64 results at or above 2^63 trapped.)
12. **f64 is the Form's float.** f32 was added in draft 0.5 (decision 42): it
    is computed in f64 and rounded once. (Originally f64 only.)
13. **Bitwise operators bind tighter than comparisons** (unlike C/Python's `&`
    vs `==` in C), so `x & 1 == 1` means what it reads as.
14. **`if … then … else …` as the expression conditional**; no Python
    `a if c else b` (it collides with `try … else`).
15. **`try e`, `try e else E`, `e catch d`, `e catch |x| h`** as the error
    vocabulary (Zig plus the prototype's `else`).
16. **Option is `T?` with `some(x)` / `none`.**
17. **Generic arguments in square brackets** (`List[T]`, `f[T](x)`), Go-style,
    resolved by name resolution; impl parameters need `forall`.
18. **Argument labels** Swift-style (`fn move(from src: u8)`), optional when the
    label equals the parameter name.
19. **`comptime`** for consts, array lengths, state initializers, comptime
    parameters, and `comptime` blocks/expressions; comptime is pure and
    deterministic.
20. **Arenas are lexical** (`with arena r`), region-typed values cannot escape;
    each step has an implicit per-step arena.
21. **Concurrency is deterministic by construction:** ticks, staged delivery,
    content-hash ids, select takes the first ready arm from a rotating start (decision 48),
    bounded channels with held sends and Backpressure logs (no silent drop),
    `after n` in logical ticks, Deadlock is a defined outcome. (Originally
    select chose the least ready arm; it rotates its starting arm since draft
    0.5, decision 48 and T11. `par` and the rest of C.4 are in version 0.6.)
22. **Pure functions cannot emit, spawn or use channels**; only steps,
    handlers, `main` and task functions (those with a `chan` parameter) have
    effects (revised by S2).
23. **Actors are regions with mailboxes and handlers** (`on M(…)`), one message
    per tick.
24. **A program with `fn main` and no state is a command**: one event
    `Start(args)`, outputs are its emits.
25. **Examples and checks are part of admission** (the gate), and the outcome
    of a multi-field example is a boolean over the final state.
26. **Kind variants encode as `{"$v": name, …}` records** in Form values;
    maps/sets as sorted lists, so canonical JSON hashing stays well-defined.
27. **Identifiers are ASCII**; text values are full Unicode, case operations
    ASCII-only (as num.rs).
28. **No macros, no exceptions, no null, no inheritance, no operator
    overloading beyond traits `Eq`/`Ord`/`Show`**: kept out to keep one
    meaning per text.
29. **The grammar artifact is the EBNF of section 3, checked LALR(1)** by
    `rce-spec-check`: a deterministic LR table with mechanical conflict
    detection, with a tree-sitter port checked against it. (Originally a
    Lark file, `rce.lark`, now retired.)

30. **Tail values** in block-bodied functions (S5).
31. **A trapping guard refuses the event** (S1).
32. **Keywords reserved everywhere except after `.`** (S3).
33. **`std.text`, `std.collections`, `std.math` are built-in modules** (S4).
34. **Loop elements of non-Copy type are borrows**; storing one needs
    `copy(x)`; lists, constructors, channels and vars cannot hold borrows
    (W-OWN-3, made explicit by the checker). Map `get` returns `V ! Missing`.
35. **Structural recursion admitted** (R1). Bound: calls ≤ nodes of the
    value recursed on.
36. **C-ABI `extern` functions** (R2): scalars and borrowed buffers in,
    scalars out, no allocator crosses the boundary, callable only from
    `main` and task functions, results logged for replay. Unwinding across
    the boundary does not exist because RCE has no unwinding.
37. **Arena capacity** `with arena r bound n` in canonical bytes (R3).
38. **Explicit `wrapping_*` and `saturating_*` methods** (R4); `+ - *` stay
    checked in every build (there is no build profile that changes meaning).
39. **`&Box[τ]` coerces to `&τ`** (R5).
40. **Effects stay inferred from context** (5.2), with no effect annotations
    and no `async`/`await`; a task function is an ordinary function that
    has a channel parameter.
41. **Ownership stays affine, not linear**, except for types declared
    `linear` (decision 46): ordinary values drop with no observable effect
    (O4).
42. **f32 computed in f64, rounded once** (N3). Bit-identical on every
    target; the cost is one extra rounding instruction per operation.
43. **Lane-wise arithmetic on fixed arrays** instead of a separate vector
    type; the array length is the lane count.
44. **Device regions** are placement hints with CPU meaning; only
    fixed-size, effect-free code may run in them.
45. **Text indexed by UTF-8 byte**, with character-boundary checks on slices.
46. **Linear structs** for outside resources, consumed exactly once.
47. **Full-range u64** through dedicated unsigned Form ops.
48. **select rotates** its starting arm per execution instead of choosing at
    random (Go) or always the least (draft 0.4).
49. **LOOP_LIMIT is a per-program constant** with default 1,000,000.
50. **Push iterators** (Go 1.23) rather than pull generators (Python, Rust):
    no suspended state, so δ stays total and replay needs nothing new.
51. **Floats derived from IEEE 754-2019 and totality** (6.3 N3, F1 to F9):
    correctly rounded basic operations only, one rounding per operator,
    subnormals kept, non-finite results trap because NaN and infinities
    break equality, order and JSON encoding, and transcendentals only as
    correctly rounded RCE code in the `math` module (F9), so every target
    runs the same algorithm instead of its platform library.
52. **Unicode 18.0.0 pinned, no tables read at run time** (6.3 N7): the
    whitespace set is enumerated, so a toolchain upgrade cannot change a
    result.

53. **Draft 0.9 is a new version** (number 0.9, never reuse 0.6), because it
    refuses programs 0.8 accepts and changes `&mut` after a failed call
    (C.15 rule).
54. **A failed call restores its `&mut` arguments** when every effect since
    entry is undoable; a function that could fail after a non-undoable effect
    is refused (F.4.4).
55. **Contracts are always enforced** and a violation is one defined refusal
    with fixed fields; no switchable contracts (F.2, F.6.2).
56. **Information flow is checked by labels with a refusal channel and robust
    declassification**; labels are erased at run time (F.4.2).
57. **`World` is a borrowed capability**, not a threaded value (F.4.1).
58. **The definition hash covers everything that affects meaning and nothing
    else** (F.4.3).
59. **`decreases` is required only where 6.7 does not already admit the
    recursion, and never replaces the loop bound** (F.6.4).
60. **Unused values, dropped results and shadowing are refused** in 0.9 (F.3).
61. **The four constructs that would break a theorem are written as
    non-goals**: effect handlers, switchable contracts, mutable aliasing
    across `par`, runtime reflection (F.2).
62. **The relaxed sequential borrow rule is not adopted** until its T6 lemma
    (T56) is proved (5.6).
63. **New theorems T42 to T56 are numbered after T41 and are stated,
    unproved**; the survey's "T17" for noninterference became T42 because
    T17 is the `par` diamond lemma (F.9).

### 9.1 Changes made while mechanizing (draft 0.1 → 0.2)

- S1 (6.4) A guard that traps refuses the event. Draft 0.1 did not say; δ
  was not total on that case.
- S2 (5.2) Task functions. Draft 0.1 decision 22 made every function except
  `main` pure, so `spawn f(…)` could never be well-typed; a function with a
  `chan` parameter may now use effects and be spawned.
- S3 (2.4) Keywords are reserved (the Lark grammar now uses a basic lexer, so
  the machine grammar agrees with the text), except as member names after
  `.`. Example 25 now uses labels `source`/`target` since `from` is a keyword.
- S4 (4.1) Built-in standard modules, so `use std.text` resolves.
- S5 (5.3) Tail values in block-bodied functions; examples 05, 06, 23 and 29
  relied on this reading.
- Examples were made whole programs: declared errors (18, 19, 22) and the
  emitted records (04, 31), defined `report` (07) and `depth_shallow` (29),
  added `shop_pricing.rce` for 26, parenthesized `(try …) as u32` (07) and
  `(m.get(w) catch 0) + 1` (21, which previously meant `catch (0 + 1)`),
  `copy(i)` in 16, and 08/09 declare their own trait
  `Describe` (the built-in `Show` cannot be re-declared or given an orphan impl).

### 9.2 Changes in draft 0.3

- The admission gate of 6.9 is implemented: `rce-check` evaluates every
  `example` and `check`, every comptime block and every constant and state
  initializer, and refuses a program on W-INT-1 or W-CT-1. W-LEX-3 (an
  integer literal outside its type's range) is checked.
- `rce.lark` tree shapes (the language is unchanged): `T?` now yields an
  `opt_type` node, negative and `true`/`false`/`none` literal patterns, the
  three receiver forms (`self`, `&self`, `&mut self`), the two ends of a slice
  and `e()` versus `e` in example event lists are distinguishable in the
  tree. Draft 0.2 trees dropped these; found while porting to tree-sitter.

What the mechanized models simplify (stated so the coverage is clear):
within a tick, sends are staged but receives consume in task-id order;
`select` is modelled over channel ids only; actor state is a number; the
borrow model's static context has the shape of the run-time state; H is an
arbitrary function (no collision or SHA-256 model); concurrency is not
evaluable at comptime.

### 9.3 Changes in draft 0.4 (from `spec/LANGUAGE-REVIEW.md`)

- R1 (6.7, W-REC-1) Structural self-recursion is admitted; mutual recursion
  and every other cycle stay refused. Checker: `structural_violation`, since
  replaced by `lex_measure` (6.7: one fixed position, or layers).
  Lean: `RCE/Struct.lean`.
- R2 (5.11, W-FFI-1, `extern`) A C-ABI foreign function interface: scalar
  and borrowed-buffer parameters, scalar results, callable only from `main`
  and task functions, results logged and replayed. `extern` is a new keyword.
- R3 (5.7) `with arena r bound n`: an arena capacity in canonical bytes,
  exceeded by `trap("size limit")`. The checker types `n` as u64; the
  reference evaluator does not yet count arena bytes.
- R4 (6.3 N2) `wrapping_*` and `saturating_*` integer methods; the default
  stays checked.
- R5 (5.1) `&Box[τ]` coerces to `&τ`, which structural recursion over
  `Box` fields needs.

Draft 0.4 deferred the changes in 9.4 because each changes the Form or the
emitters.

### 9.4 Changes in draft 0.5

- f32 (5.1, 6.3 N3): new type, Form ops `f32add f32sub f32mul f32div` and
  prim `tof32` in `form/num.rs`, `form_core.py` and `form_core.js`.
  Lean: `RCE/F32.lean`.
- Lane-wise array arithmetic (5.1, W-SIMD-1). It lowers to Form `Map` over
  the zipped lanes, so the Form needed no change.
- Device regions (5.13, W-DEV-1, keyword `device`).
- Byte-indexed text (5.1, 6.3 N7): Form prims `blen`, `bget`, `bslice`.
- Full-range u64 (6.3 N2): Form ops `uadd usub umul udiv urem`.
  Lean: `RCE/U64.lean`.
- select rotation and bounded wait (6.8). Lean: `RCE/Select.lean`.
- Linear structs (5.12, W-LIN-1, keyword `linear`); `extern` may take and
  return linear structs with one scalar field. Lean: `RCE/Linear.lean`.
- `const LOOP_LIMIT: u64 = n` (6.7, W-LOOP-1).
- Iterators (5.14, W-ITER-1, keywords `iter` and `yield`). The reference
  evaluator runs the iterator with a yield cap, delivers values to the loop
  body, and re-runs with a doubled cap only when the body asks for more;
  `gen_prefix` shows this equals the push model. Lean: `RCE/Iter.lean`.
- R3 completed: the reference evaluator counts arena and device bytes
  (canonical JSON of the values bound in the region, after every statement)
  and traps "size limit".
- `lalr` is no longer reserved (2.4).
- The four Form runtimes (Rust crate and WASM through `num.rs`, Python,
  JavaScript for XState and SCXML) agree on the new ops; the
  `tests/form_num_v05.rs` cross-runtime test compares them on about 6,600
  cases.

Known open questions (not decided here): the standard library surface beyond
the num.rs primitives; foreign interfaces other than the C ABI (WASM
component model, GPU kernels); how an RCE module is used as a fit (the
handler spec and fit formats themselves are now Appendix D); whether emitters should lower device
regions and lane-wise operators to real accelerator or vector code (today
they run them on the CPU, which 5.13 allows).

### 9.5 Changes in draft 0.7

- Version 0.7 (2.7, C.15): version 0.6 with size-scaled standard library
  method costs (C.6.4). Versions 0.5 and 0.6 keep their meaning.
- Errata for every version, listed in C.15: closed standard library and
  no ambient I/O (5.15), `&mut` write-back on `fail` in the emitters (6.2,
  T33) with nested calls flattened by an order-preserving ANF pass (T34),
  exact u64 at or above 2^63 (6.3 N2), the fallible-return literal fix
  (5.1), saturating budget arithmetic (C.6), the hash-chained Form log
  (6.1).
- Refined types (5.1, W-REF-1, T35): `type R = τ where p` with p in a
  decidable fragment, checked where the declared integer range is
  checked. Derived from soundness and one meaning per construct
  (`spec/proposals/CORE-PROPERTIES.md`, P2). Closes the replacement-matrix
  rows "improper input validation" (defined: refused at entry) and
  "dependent specifications" (defined for the decidable fragment;
  undecidable obligations excluded by derivation). An extension for
  every version (C.15).
- `sha256` text method (5.15): SHA-256 of the UTF-8 bytes as 64 lowercase
  hex digits, linear cost class (C.6.4), in the reference evaluator and the
  Rust, Python and JavaScript runtimes. Derived from self-hosting: the
  hash-chained log (6.1) and content-hash ids (6.8) must be computable by
  an RCE program. An extension for every version, since it only admits
  programs refused before (C.15).

### 9.6 Changes in draft 0.8

- Version 0.8 (2.7, 5.16, C.16): 0.7 plus indexed values, context-typed
  text `Text[C]`, opaque `Secret[τ]`, capabilities `Cap[A]` and
  `Guarded[τ, A]` fields, with W-CTX-1, W-SEC-1, W-CAP-1, W-CAP-2 and T36.
  A new version because the names it takes (`Text`, `Html`, ...) could be
  a 0.7 program's own declarations, whose meaning would change. Derived
  from one meaning per construct, least privilege and soundness
  (`spec/proposals/CORE-PROPERTIES.md`, P1). Closes the replacement-matrix
  rows SQL injection and XSS (excluded), missing authorization
  (excluded), hard-coded credentials and information exposure (excluded).

### 9.7 Changes in draft 0.9

Version 0.9 (2.7, Appendix F) is specified and not yet implemented. Each item
below is applied in the tier order of its dependencies; the derivation is in
Appendix F next to the rule, and every theorem is stated and unproved (F.9).

- **Tier 0.** Stale text corrected (errata for every version): decisions 8, 11,
  12, 21 and 29, the Copy list of 5.5 (f32 added), W-VER-1 and 2.7 (0.8 and
  0.9 named), the overview status and its checker figures. Version 0.9 refuses
  unused values, dropped non-unit results and unused stores (F.3.2), and
  adds lexical hygiene: hidden and bidirectional characters, shadowing of any
  enclosing local, qualified variants, longest-token lexing (F.3.3).
- **Tier 1.** `World` capability for foreign calls (F.4.1); information-flow
  labels with the refusal channel and robust declassification (F.4.2, T42,
  T43); definition hashes and `rce api-diff` (F.4.3, T50); transactional
  calls (F.4.4, T47).
- **Tier 2.** File-system capabilities and caretaker revocation (F.5.1, T51);
  typed shell and prompt output and strict input (F.5.2, T44); the hash-pinned
  manifest (F.5.3, T54); `ct fn` (F.5.4, T45); declared budgets (F.5.5, T52).
- **Tier 3.** Templates `t"..."` (F.6.1); `requires` and `ensures` (F.6.2,
  T46); `ghost` (F.6.3, T49); `decreases` (F.6.4, T48); `if let` (F.6.5);
  trait laws, tested not proved (F.6.6).
- **Tier 4.** The broker contract for the lethal trifecta (F.7, T55).
- **Rules derived from the sources' later items** (F.8): closures, types,
  calls, operators, containers, indexing, classes, errors, reflection and
  derive, compile-time code, linear values, tooling, release gate.
- **Non-goals** (F.2): effect handlers, switchable contracts, mutable
  aliasing across `par`, runtime reflection, and the other constructs listed
  there, each with its reason.
- **Not adopted until proved:** relaxed sequential borrows (5.6, T56).
- **Version number.** The proposal list was drafted under the name "draft
  0.6"; 0.6, 0.7 and 0.8 are taken, so this is 0.9 (C.15: a version number
  is never reused).

### 9.8 Decision log for draft 0.9 (open questions, decided by derivation)

Each row decides one question left open by the survey and security study. The
principle named is one of the five in Appendix F: least privilege (a),
soundness (b), determinism (c), totality (d), one meaning per construct (e).
No row rests on taste: where no principle decides, the existing text stays.

| Q | Question | Decision | Derivation |
|---|---|---|---|
| 1 | Which effects are undoable for a transactional call | Writes to owned values and `emit` into the pending output are undoable; send, receive, `spawn`, foreign calls, `World` operations, host entries and consuming a linear handle are not; a path from a non-undoable effect to a `fail` is refused (W-TX-1) | (c): a restore must be total and need no handler; (b) T8 |
| 2 | Measure for mutual recursion | One lexicographic tuple for the whole group (F.6.4) | (d): one well-founded order for the group, as 6.7 already uses |
| 3 | Is tool metadata hashed | No. Comments, formatting, declaration order and attributes are not hashed; the version, signature, contracts, measure, ghost clauses, labels and referenced hashes are | (c) and T7: a hash is the meaning, nothing else |
| 4 | `?` in `par` arms | Refused | T16: `?` leaves the arm early, so an arm is not an expression with one result |
| 5 | Advisory format | Canonical JSON object signed with Ed25519 over its SHA-256 | (c): reuse the canonical JSON of 6.1; Ed25519 signatures are deterministic |
| 6 | Map iteration order | Ascending canonical key order of 7.1 | already 6.7 and the C.15 erratum; (c) |
| 7 | Half-open slices and inclusive ranges | Unchanged: slices as in 6.3 N8, ranges keep `..` and `..=` | (e): both already have one meaning; no principle favors removing one |
| 8 | Comparison-chain sugar | Not adopted; a chain is refused (W-OP-1) | (e): one meaning, and a chain would need a second reading of the operator |
| 9 | NaN payload and libm strategy | Neither exists: non-finite results trap, transcendentals are RCE code in `math` (F9) | already decision 51; (c) |
| 10 | Value-indexed types | Not adopted beyond comptime array lengths and the type indices of 5.16 | (d): a value in a type needs a decidable fragment, and none is specified |
| 11 | Linear types in the spec | They are (5.12, W-LIN-1, T12); a linear handle is not undoable | already in the spec; (b) |
| 12 | Inductive predicates | Not adopted; a predicate is a `ghost` pure total `bool` function | (d): inductive definitions need positivity and continuity checks no theorem needs |
| 13 | Are doc comments hashed | No doc comments exist (decision 2, `#` comments only), so nothing is hashed | decision 2; (e) |
| 14 | Maximal-munch token rule | Adopted as a requirement on the lexer, checked in F.11 | (c): the same text must tokenize the same everywhere |
| 15 | First-class type invariants | A refined type has `R.check(v) -> R?`; no witness clause; no implicit conversion; opaque outside its module | (b) T35: a value of R is only made by a checked path |
| 16 | Argument labels | Kept (decision 18) and checked: a wrong label is refused | (e): a label that is not checked has two meanings, the written and the real one |
| 17 | Constant conditions | An `if`, `elif`, `while` or guard on a comptime constant is refused (W-UNUSED-2) | (e): one branch is dead code; P8 |
| 18 | Display trait | `Show` of 5.9; a template field of a non-text type must implement it | (e): one display conversion, explicit; no new trait |
| 19 | Map equality independent of order | Yes, because the representation is canonical | (c): equality and iteration agree by construction |
| 20 | User-definable `derive` | Not offered; `derive` is the built-in closed set of 5.9 | T7: a user derive needs reflection |
| 21 | Literal default type when unconstrained | Unchanged: integer literals default to i64, float literals are f64 (5.3, 5.1) | (c): the default is a fixed rule; refusing adds no safety |
| 22 | Expression fields and nested quotes in `t"..."` | Not allowed: a field is an identifier or a field path, no quotes inside a field | (e) and S8: a field needing a nested lex has two readings |
| 23 | Inductive type syntax and its `decreases` rule | No new syntax: a recursive `kind` with `Box`, strictly positive because `fn` types cannot occur in data (W-TY-2) | (d): positivity follows from W-TY-2 |
| 24 | Is `refuse` the empty enum | No: `refuse` is not a value and has no type | (e): a bottom type brings subtyping, which RCE does not have |
| 25 | Fixed-size arrays as value parameter or ghost length | Array length stays a comptime constant `[τ; n]`; contracts may use `len` | (d): no value-indexed types (Q10) |
| 26 | Spelling of the `Copy` and `Discard` bounds | No new spelling; `Copy` is the built-in of 5.5 and generics stay affine (decision 41); a type parameter is not instantiated by a linear type | (b) T12 |
| 27 | Canonical form of propositions | A contract is hashed as the canonical parsed tree of its expression; no rewriting or solver normalization | (c): a hash must not depend on a solver |
| 28 | Source span in the refusal record | No; the record has the rule or clause, the definition hash and the clause index | (c) and T7: a span depends on whitespace |
| 29 | Layout-sensitive vector types | Not adopted; lane-wise fixed arrays only (decision 43) | no principle decides; the existing text stays |
| 30 | Any foreign boundary in 0.9 | Yes, the C ABI of 5.11 as a `World` operation, declared in the manifest | (a): authority is a value; (c) T32 |
| 31 | Big integer or ghost mathematical integer | `mathint`, ghost and contract code only; no runtime big integer | (d): an unbounded integer has size-dependent cost, which budgets cannot bound |
| 32 | Wording for reactive programs | A total step function driven by the host (6.1, Appendix E); no `while true` | (d) |
| 33 | Unsatisfiable contracts: static or first call | Static in the decidable fragment (W-CON-1); otherwise enforcement refuses every call | (b), (d): decide what is decidable, never succeed vacuously |
| 34 | Comment forms | Only `#` line comments | decision 2; (e) |
| 35 | Implication spelling in contracts | None; write `not a or b` | (e): `or` short-circuits, so the form is already guarded |
| 36 | Digit separators in literals | One `_` between digits, as 2.5 already says | (e): one canonical form, already specified |

### 9.9 Reconciliation of the published text with draft 0.9

The published text and the engine development tree were joined at draft 0.9.
No version number was reused or redefined: versions 0.7 and 0.8 keep the
meaning they had in the published text, and 0.9 is the version of Appendix F.
Where both trees had reworded the same stale sentence (decisions 8, 11, 12 and
29 above), the wording that states the original rule and then its revision was
kept, because it says what the rule was and what it is now. The canonical JSON
text of 6.1.1 and its vectors, and the claim counts of the published text, are
kept as published. The figures quoted in the overview, the syntax chapter and
the tooling chapter are the printed output of `rce-spec-check` on this tree,
checked by `scripts/claim-counts.py --check`.

**Read receipt.** The decisions list says it decides gaps in which file?

<!-- body-sha256: ff45ae94e1c1a3b84f3306b673307ff11857d3b8bb97053640d2e415c947e50c -->
