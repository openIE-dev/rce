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
- W-RES-1 A construct that a rule of this document specifies and this
  implementation does not implement (a reserved word used as its construct,
  F.12): refused with the words *reserved, not implemented*.

Names and modules
- W-NAME-1 Unresolved name, or a name resolved in the wrong namespace.
- W-NAME-2 Duplicate declaration in one scope; shadowing a parameter, field
  or module-level name inside the same item.
- W-MOD-1 `use` of a private item; W-MOD-2 import cycle. The reference
  checker reports a private name reached through `use` as W-VIS-1, which
  is the same rule.
- W-MOD-3 An imported name that clashes with a declaration of the
  importing module or with a name imported from another module (imported
  names are not renamed).
- W-SAFE-1 The identifiers `unsafe`, `Rc`, `Arc`, `RefCell`, `transmute`,
  `static`: refused with "memory safety is by construction" (there are no
  escape hatches in RCE other than the `extern` boundary of 5.11).
- W-FFI-1 An `extern` declaration whose ABI is not `"C"`, that is comptime
  or generic, or whose parameter or result types break 5.11; or a foreign
  function's name used other than as the callee of a direct call.

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
  monomorphization, other than a cycle of plain functions (one function
  calling itself included) whose calls are done by layered summed
  measures, or each of whose cycles of calls passes through a function
  declared with `bound` (6.7); `par` or `for … par` in a function of a
  cycle with a bounded function; `bound` on an iterator, a foreign
  function or a trait signature.
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
- W-REF-1 A refined type (5.1) that is generic, refines a type other than
  an integer type, `text` or a collection, or whose predicate is outside
  the decidable fragment (a term other than `self`, `self.len()` or an
  integer literal, an operator other than a comparison, `and`, `or`,
  `not`, or a call other than a text character class).
- W-LOOP-1 A `LOOP_LIMIT` declaration that is not `const LOOP_LIMIT: u64 = n`
  with a positive literal n (6.7).
- W-TRAIT-1 An impl not matching its trait; W-TRAIT-2 incoherent or orphan
  impls; W-TRAIT-3 a type used with `==`, `!=`, an ordering operator or a
  generic bound without implementing the trait it needs.

Version 0.8 only (5.16)
- W-CTX-1 Context text made other than by `C.lit` of a string literal or
  `C.encode`; joined with another context or plain text; carried by an
  event payload; an unknown context.
- W-SEC-1 An operator or method on a secret other than `==`/`!=`; a
  secret in an emitted value, a channel, an error payload or a foreign
  call.
- W-CAP-1 A capability outside an event payload field or a parameter;
  `Guarded` outside a state field; an authority that is not a declared
  struct.
- W-CAP-2 A step that writes a `Guarded[τ, A]` field on an event without
  a `Cap[A]` field. (W-NAME-2 refuses a declaration named after a 5.16
  type or context.)

Version 0.9 only (Appendix F, specified and not yet implemented): the
refusals of F.3 to F.8, among them unused values, dropped results and
shadowing (F.3), `World` and information-flow labels (F.4), transactional
calls (F.4), capabilities, typed output, pinned builds, constant-time functions
and budgets (F.5), and templates, contracts, ghost code, written measures,
`if let` and trait laws (F.6). Their names and rules are listed where each is
defined and are not repeated here, so that this list names only rules the
reference checker implements.

Concurrency (Appendix C): W-PAR-1 to W-PAR-4, W-DL-1, W-ND-1, W-GRP-1 to
W-GRP-3, W-SES-1 to W-SES-5, W-SCOPE-1, W-ATOM-1, W-ATOM-2, W-CRDT-1,
W-LVAR-1, W-LVAR-2, W-NODE-1, W-SUP-1, W-SUP-2, W-INTR-1 and W-RT-1.

**Read receipt.** Which rule refuses two unguarded steps in one region whose patterns overlap?

<!-- body-sha256: 46e73d53d86f66272f6a543db803efeb9538bd8737ba7c81ceaa8b4b78767bd7 -->
