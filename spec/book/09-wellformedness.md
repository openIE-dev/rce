## 7. Well-formedness: what requires change at entry

This section is the checklist of rules a program must pass to be accepted
(*well-formedness*). Each rule has a name such as W-TY-1, and a
program that requires change is reported with the rule it broke. Example: a function declared
`-> i64` whose body returns text requires change: W-TY-1.

A program requires change if any rule fails. Every diagnostic names its rule.

Lexical and syntax
- W-LEX-1 Not UTF-8, BOM, bare CR, or a tab character.
- W-LEX-2 Inconsistent dedent (2.6).
- W-LEX-3 A literal outside every possible type range, or an invalid `\u{}`.
- W-SYN-1 Not in the language of section 3.
- W-RES-1 A construct that a rule of this document specifies and this
  implementation does not implement (a reserved word used as its construct,
  F.12): requires change, with the words *reserved, not implemented*.

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
  `static`: requires change, with "memory safety is by construction" (there are no
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
- W-INT-2 (recommended; never requires change) A module with a step and
  `example` lines, whose obligations its lookup-table control meets (6.9):
  `rce check` admits it and prints the recommendation (10.2).
- W-LAW-1 A law (6.11) that fails, or that is undecided and not attested
  (its alphabet is incomplete or the bound LAW_FUEL is reached), or an
  `attest` line that is not `WHO;YYYY-MM-DD;COVER`. The report of a failing
  law ends with its counterexample as an `example` line. The trait laws of
  F.6.6 fall under the same rule once they are implemented.
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

Indexed values (5.16)
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
  a `Cap[A]` field. (A declaration named after a 5.16 type or context
  requires change: W-NAME-2.)

Appendix F (specified and implemented in part, F.12): the
rules of F.3 to F.8 under which a text requires change, among them unused values, dropped results and
shadowing (F.3), `World` and information-flow labels (F.4), transactional
calls (F.4), capabilities, typed output, pinned builds, constant-time functions
and budgets (F.5), and templates, contracts, ghost code, written measures,
`if let` and trait laws (F.6). Their names and rules are listed where each is
defined and are not repeated here, so that this list names only rules the
reference checker implements.

Concurrency (Appendix C): W-PAR-1 to W-PAR-4, W-DL-1, W-ND-1, W-GRP-1 to
W-GRP-3, W-SES-1 to W-SES-5, W-SCOPE-1, W-ATOM-1, W-ATOM-2, W-CRDT-1,
W-LVAR-1, W-LVAR-2, W-NODE-1, W-SUP-1, W-SUP-2, W-INTR-1 and W-RT-1.

### 7.1 How a rule guides a program into Form

Each rule of this section, of Appendix C and of Appendix F guides a program
that does not meet it into Form by exactly one of three methods:

- **Altered.** The rule has one rewrite, determined by the text, that keeps
  the program's meaning. `rce fmt FILE` writes it and `rce fmt check FILE`
  lists it (10.6). Until it is written the text requires change, and
  `rce check` ends the report with `[altered by rce fmt]` (10.2). A rewrite
  that would add or change another report is not written; `rce fmt` then
  prints the report as requiring change.
- **Recommended.** The program is admitted, and `rce check` prints a
  `recommended` line that names the rule and the change (10.2).
- **Requires change.** The program is not admitted until its author changes
  it, because the change has more than one reading or states something only
  the author can state (a type, a declaration, a bound, an authority).
  `rce check` ends the report with `[requires change]`.

A rule is altered only when the reference `rce fmt` writes its rewrite. The
table gives every rule its method; the checker reads it from
`spec/data/spec-data.json` (`w_methods`), and `tests/three_methods.rs`
checks that every rule of the three lists has exactly one row and that the
altered rows are the rules `rce fmt` rewrites. Name independence (T7,
section 8) is the ground of a rewrite that renames.

| Rule | Method | Reason |
|---|---|---|
| W-LEX-1 | requires change | A tab, a bare CR or bytes that are not UTF-8 have more than one reading (an indentation width, a line end, an encoding). |
| W-LEX-2 | requires change | An inconsistent dedent has two block readings. |
| W-LEX-3 | requires change | The type or the character the literal was meant to name is the author's. |
| W-SYN-1 | requires change | A text outside the grammar of section 3 has no one reading. |
| W-RES-1 | requires change | The construct is specified and not implemented; the author writes it with implemented constructs. |
| W-NAME-1 | requires change | The author declares or imports the name, or names another. |
| W-NAME-2 | requires change | Which declaration takes another name, and which name, is the author's choice. |
| W-MOD-1 | requires change | A module's `pub` surface is its owner's to change. |
| W-VIS-1 | requires change | It is the checker's name for W-MOD-1. |
| W-MOD-2 | requires change | Which import breaks the cycle is the author's choice. |
| W-MOD-3 | requires change | Which name changes is the author's choice. |
| W-SAFE-1 | requires change | Each identifier has several forms in RCE (an owned value, a `state` field, the `extern` boundary). |
| W-FFI-1 | requires change | A foreign boundary is declared by its author. |
| W-TY-1 | requires change | A typing rule that fails has no one repair. |
| W-TY-2 | requires change | The type that replaces the nested or borrowed one is the author's choice. |
| W-TY-3 | requires change | The conversion and its rounding are the author's choice. |
| W-PAT-1 | requires change | Whether the name was meant as the variant or as a binding is the author's statement. |
| W-MATCH-1 | requires change | What the missing arms do is the author's statement. |
| W-MATCH-2 | requires change | An unreachable arm marks an arm the author meant to reach, or one to remove. |
| W-PLACE | requires change | Declaring the place with `var`, or writing another place, is the author's choice. |
| W-OWN-1 | requires change | A copy, a borrow or another order of uses is the author's choice. |
| W-OWN-2 | requires change | A copy in and out, or a snapshot, is the author's choice. |
| W-OWN-3 | requires change | Owning a copy, or passing a handle or an index, is the author's choice. |
| W-REG-1 | requires change | Owning a copy, or passing a handle or an index, is the author's choice. |
| W-SPAWN-1 | requires change | Owning a copy, or passing a handle or an index, is the author's choice. |
| W-STATE-1 | requires change | The region's fields are the author's to declare. |
| W-STATE-2 | requires change | A constant initializer, or a computation moved to an input, is the author's choice. |
| W-STATE-3 | requires change | The owned type of the field is the author's choice. |
| W-STEP-1 | requires change | The author declares the region or the event, or names another. |
| W-STEP-2 | requires change | A step for the event, or no such event, is the author's choice. |
| W-STEP-3 | requires change | The field the block form names is the author's choice. |
| W-STEP-4 | requires change | Which step was meant to run is the author's statement. |
| W-EFF-1 | requires change | Moving the effect to a task, a step or `fn main` is a choice of structure. |
| W-REC-1 | requires change | A measure or a `bound` is the author's statement. |
| W-CT-1 | requires change | The comptime expression is the author's to change. |
| W-INT-1 | requires change | An example or check that does not hold states another behaviour than the program has; which of the two is meant is the author's statement. |
| W-INT-2 | recommended | The module is admitted; the report recommends obligations beyond the listed examples (6.9). |
| W-LAW-1 | requires change | A law that fails or is undecided states a property the program does not show; whether the program, the law or an attestation changes is the author's statement. |
| W-SIMD-1 | requires change | The lengths and lane types are the author's. |
| W-LIN-1 | requires change | Where the linear value is consumed is the author's statement. |
| W-DEV-1 | requires change | Splitting the host and device parts is a choice of structure. |
| W-ITER-1 | requires change | Collecting the iterator, or another structure, is the author's choice. |
| W-REF-1 | requires change | The predicate is the author's statement. |
| W-LOOP-1 | requires change | The bound n is the author's statement. |
| W-TRAIT-1 | requires change | The missing or differing methods are the author's to write. |
| W-TRAIT-2 | requires change | Which impl remains, or a newtype, is the author's choice. |
| W-TRAIT-3 | requires change | Implementing the trait, or another use, is the author's choice. |
| W-CTX-1 | requires change | The context of a text is declared by its author. |
| W-SEC-1 | requires change | A digest, an equality bit or a release is the author's choice. |
| W-CAP-1 | requires change | Authority is placed by the author; a tool never adds it. |
| W-CAP-2 | requires change | Authority is placed by the author; a tool never adds it. |
| W-PAR-1 | requires change | Separating the branches' footprints is a choice of structure. |
| W-PAR-2 | requires change | Moving the effect or the exit out of the `par` is a choice of structure. |
| W-PAR-3 | requires change | Writing the body as a reduction or by slot is a choice of structure. |
| W-PAR-4 | requires change | Moving the effect or the exit out of the loop body is a choice of structure. |
| W-DL-1 | requires change | Moving the construct into `fn main` or a task function is a choice of structure. |
| W-ND-1 | requires change | Moving the construct into `fn main` or a task function is a choice of structure. |
| W-GRP-1 | requires change | Where the group runs is the author's choice. |
| W-GRP-2 | requires change | Where the barrier stands is the author's choice. |
| W-GRP-3 | requires change | Which lane writes which slot in which phase is the author's choice. |
| W-SES-1 | requires change | The protocol and its uses are the author's statement. |
| W-SES-2 | requires change | The protocol and its uses are the author's statement. |
| W-SES-3 | requires change | The protocol and its uses are the author's statement. |
| W-SES-4 | requires change | The protocol and its uses are the author's statement. |
| W-SES-5 | requires change | Which body passes which endpoint is the author's choice. |
| W-SCOPE-1 | requires change | The scope of a task is a choice of structure. |
| W-ATOM-1 | requires change | Moving the wait out of `atomic` is a choice of structure. |
| W-ATOM-2 | requires change | The placement of the actors is the author's choice. |
| W-CRDT-1 | requires change | The join's laws are the author's to meet. |
| W-LVAR-1 | requires change | The order of `put` and `freeze` is the author's statement. |
| W-LVAR-2 | requires change | The thresholds are the author's statement. |
| W-NODE-1 | requires change | Placement is declared by the author. |
| W-SUP-1 | requires change | Supervision is declared by the author. |
| W-SUP-2 | requires change | Supervision is declared by the author. |
| W-INTR-1 | requires change | The priority is the author's statement. |
| W-RT-1 | requires change | The deadline is the author's statement. |
| W-BUD-1 | requires change | The budget is the author's statement. |
| W-UNUSED-1 | altered | An unread binding `x` is named `_x`, a binding `_x` that is read is named `x`, and an unread `use` takes a name that begins with `_` and links the same module; names carry no meaning (T7). |
| W-UNUSED-2 | requires change | Which branch the constant condition was meant to choose is the author's statement. |
| W-DROP-1 | altered | `let _ = e` evaluates `e` and drops its value, as the expression statement did; an arm `P -> e` becomes a block arm holding `let _ = e`. |
| W-ASSIGN-1 | requires change | Which of the two stores was meant is the author's statement. |
| W-LEX-4 | requires change | The rule shows the author a character the text hides; what to write instead is the author's choice. |
| W-NAME-3 | requires change | The new name is the author's choice. |
| W-WORLD-1 | requires change | Where the `&World` handle comes from is a choice of structure. |
| W-IFC-1 | requires change | A label is lowered only by its author's `release`; a tool never lowers one. |
| W-IFC-2 | requires change | A label is lowered only by its author's `release`; a tool never lowers one. |
| W-IFC-3 | requires change | A label is lowered only by its author's `release`; a tool never lowers one. |
| W-IFC-4 | requires change | A label is lowered only by its author's `release`; a tool never lowers one. |
| W-IFC-5 | requires change | A release is the author's statement. |
| W-IFC-6 | requires change | The label is declared by its author. |
| W-IFC-7 | requires change | Endorsing a model answer is the author's statement. |
| W-TX-1 | requires change | The order of effects and failures is the author's statement. |
| W-CAP-3 | requires change | Authority is placed by the author; a tool never adds it. |
| W-CAP-4 | requires change | Authority is placed by the author; a tool never adds it. |
| W-CAP-5 | requires change | Authority is placed by the author; a tool never adds it. |
| W-PIN-1 | requires change | A pin or a dependency is changed by its owner. |
| W-CTFN-1 | requires change | Constant-time code is restructured by its author. |
| W-CTFN-2 | requires change | The emitter target is the author's choice. |
| W-CTX-2 | requires change | The context of a text is declared by its author. |
| W-CTX-3 | requires change | Where a model call stands is a choice of structure. |
| W-TPL-1 | requires change | How a template is rendered is the author's statement. |
| W-TPL-2 | requires change | The template's text is the author's. |
| W-CON-1 | requires change | A contract is the author's statement. |
| W-CON-2 | requires change | A contract is the author's statement. |
| W-GHOST-1 | requires change | Where a ghost value flows is the author's statement. |
| W-DEC-1 | requires change | A measure is the author's statement. |
| W-IFLET-1 | requires change | The comparison is the author's statement. |
| W-LEN-1 | requires change | `len_bytes` or `len_chars` is a choice of unit. |
| W-TRI-1 | requires change | A release path is the author's statement. |

**Read receipt.** Under which rule do two unguarded steps in one region whose patterns overlap require change?

**Read receipt.** Under which rule does a region with two unguarded steps whose patterns overlap require change?

<!-- body-sha256: bcd9aa8480eedf926c955c50278a1ea6fcbdaf5610859356a24afa973ca726e1 -->
