## 7. Well-formedness: what is refused at entry

This section is the checklist of rules a program must pass to be accepted
(*well-formedness*). Each rule has a name such as W-TY-1, and a refused
program is reported with the rule it broke. Example: a function declared
`-> i64` whose body returns text is refused by W-TY-1.

A program is refused if any rule fails. Every diagnostic names its rule.

Lexical and syntax
- W-LEX-1 Not UTF-8, BOM, bare CR, or a tab character.
- W-LEX-2 Inconsistent dedent (2.6).
- W-LEX-3 A literal outside every possible type range, or an invalid `\u{}`.
- W-SYN-1 Not in the language of section 3.

Names and modules
- W-NAME-1 Unresolved name, or a name resolved in the wrong namespace.
- W-NAME-2 Duplicate declaration in one scope; shadowing a parameter, field
  or module-level name inside the same item.
- W-MOD-1 `use` of a private item; W-MOD-2 import cycle.
- W-SAFE-1 The identifiers `unsafe`, `Rc`, `Arc`, `RefCell`, `transmute`,
  `static`: refused with "memory safety is by construction" (there are no
  escape hatches in RCE other than the `extern` boundary of 5.11).
- W-FFI-1 An `extern` declaration whose ABI is not `"C"`, that is comptime
  or generic, or whose parameter or result types break 5.11.

Types
- W-TY-1 Any typing rule of section 5 fails.
- W-TY-2 Nested option `τ??`; borrow of a borrow `&&τ`; `[τ]` not behind a
  borrow; a borrow, region type or `fn` type inside a data type.
- W-TY-3 Integer/float mixing without `as`; implicit narrowing.
- W-PAT-1 A binding pattern whose name equals a variant in scope.
- W-MATCH-1 A non-exhaustive `match` (5.3 [Match]); W-MATCH-2 an unreachable
  arm (subsumed by earlier unguarded arms).
- W-PLACE Left side of an assignment or `op=` is not a mutable place.

Ownership and borrowing
- W-OWN-1 Use after move (O1). W-OWN-2 Violating [No-Alias-Write] or
  [No-Mut-Read]. W-OWN-3 Storing, returning (except R-RET) or capturing a
  borrow. W-REG-1 A region-typed value escaping its `with arena` block.
- W-SPAWN-1 Spawning with a borrow or region-typed argument.

The Form
- W-STATE-1 A `state` with no fields; W-STATE-2 a field initializer that is not
  a comptime constant; W-STATE-3 a field type that is not an owned data type.
- W-STEP-1 A `step` naming an undeclared region or matching an undeclared
  event. W-STEP-2 **Totality**: an event that no step and no actor handler can
  ever match. W-STEP-3 The arrow form on a region with more than one field.
  W-STEP-4 Within one region, two unguarded steps whose patterns overlap
  (the second is unreachable).
- W-EFF-1 `emit`, `spawn`, channel operations or `select` in a pure `fn`;
  reading another region's fields in a step; calling a foreign function
  anywhere but `fn main` or a task function.
- W-REC-1 A cycle in the call graph, including through trait methods after
  monomorphization, other than direct structural recursion (6.7).
- W-CT-1 Comptime code that calls a non-comptime-safe function or traps or
  exceeds fuel.
- W-INT-1 An `example` or `check` that does not hold.
- W-SIMD-1 A lane-wise operator on arrays of different lengths, of an
  unfixed length, or with non-numeric lanes (5.1).
- W-LIN-1 A linear value not consumed exactly once on every path, copied,
  moved inside a loop, or placed in a collection, field, payload, channel or
  iterator (5.12).
- W-DEV-1 A growable value (`List`, `Map`, `Set`, `text`, slice, channel,
  iterator) bound in a device region, or `emit`, `spawn`, send, `select`,
  `yield` or a foreign call inside one (5.13).
- W-ITER-1 An iterator function that is not pure or has an arrow body;
  `yield` outside an iterator function; an `iter τ` value used other than as
  a `for` operand or `.collect()` receiver (5.14).
- W-LOOP-1 A `LOOP_LIMIT` declaration that is not `const LOOP_LIMIT: u64 = n`
  with a positive literal n (6.7).
- W-TRAIT-1 An impl not matching its trait; W-TRAIT-2 incoherent or orphan
  impls.

**Read receipt.** Which rule refuses two unguarded steps in one region whose patterns overlap?

<!-- body-sha256: 1ce3b9d19e6e413684b1abf4d63dd945b6d947007ba000471e1e1b6b87fc8e6f -->
