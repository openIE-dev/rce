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
8. **Only structural recursion** (revised in 0.4: a self-call must pass a
   constructor field of a matched parameter; mutual recursion refused. Since
   replaced by the measure of 6.7, which admits a self-call or a cycle whose
   calls shrink a measure, in layers where needed) and
   **every loop bounded** (default 1,000,000, `bound n` to change), so δ is
   total by construction. Recursion on integers (`f(n - 1)`) is still
   refused; write it as a bounded loop.
9. **Second-class borrows, no lifetime syntax.** Borrows only in parameters,
   receivers, loop/match operands and short `let`s, plus borrow returns tied to
   a parameter. Simpler than Rust, still no GC and no aliasing of mutable
   state.
10. **`struct` only, no `class`**; value semantics, explicit `copy()` for deep
    duplication of non-Copy values.
11. **Integer types checked against their own range**, all lowered to the
    Form's i64; u64 arithmetic results ≥ 2^63 trap. No implicit widening; no
    literal suffixes.
12. **f64 only** (no f32), because the Form has one float. Superseded in
    draft 0.5 by decision 42: f32 is a type, with the Form ops of 9.4.
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
    content-hash ids, select chooses the least ready arm in canonical order,
    bounded channels with held sends and Backpressure logs (no silent drop),
    `after n` in logical ticks, Deadlock is a defined outcome.
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
29. **Grammar artifact is a checked LALR(1) table** rather than tree-sitter: it is a
    deterministic LR table with mechanical conflict detection; a tree-sitter
    port is mechanical once the grammar is approved. Revised on 2026-10-12:
    this decision named Lark (`rce.lark`), which is retired. The normative
    grammar is the EBNF of section 3, and `rce-spec-check` builds the LALR(1)
    tables from it (3.12); the tree-sitter port is checked against it.

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
    break equality, order and JSON encoding, no transcendentals until a
    correctly rounded one runs on every target.
52. **Unicode 18.0.0 pinned, no tables read at run time** (6.3 N7): the
    whitespace set is enumerated, so a toolchain upgrade cannot change a
    result.

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

**Read receipt.** The decisions list says it decides gaps in which file?

<!-- body-sha256: b0d7cb2e7884f1c40bfc7c0f68e4458e4e5803de07bdfe849fb109f6f19f2fc7 -->
