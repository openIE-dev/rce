## Appendix F. Draft 0.9 (normative for version 0.9 modules; implemented in part, see F.12)

Appendix F specifies version 0.9. It applies the proposals of the 2025+
language survey (`spec/LANGUAGE-REVIEW.md` records the earlier review) and of
the security study, after a full read of their sources. Each rule here is
derived from a stated principle or an existing theorem, and the derivation is
given next to the rule; none is taken from a source language for its own
sake. The decisions are recorded in 9.7 and 9.8.

**Status of this appendix.** It is a specification, implemented in part by the
reference checker. F.12 lists, rule by rule, what the reference checker enforces
and what it does not. A module that declares `# rce-version: 0.9` is accepted
(W-VER-1); a construct of this appendix that the checker does not implement is
*refused by version* (W-VER-2), never read as an ordinary name and never
silently ignored; a rule of this appendix that is not a construct and is not
implemented is listed in F.12 as not enforced. The Lean files, the emitters and
the tree-sitter port do not implement Appendix F (an emitter refuses a 0.9
module that needs a rule it does not implement, F.4.4). Every theorem it states
(T42 to T56, section 8) is an obligation marked *stated, unproved*; this
appendix and its implementation claim no proof.

**Version number.** Draft 0.9 is the next free number. Earlier working notes
called this list "draft 0.6"; 0.6, 0.7 and 0.8 are taken (C.15, C.16), and a
version number is never reused (C.15), so this is version 0.9. By the C.15
rule it is a version, not an erratum or extension: it refuses programs that
0.8 accepts (F.3, F.4) and gives `&mut` arguments a different value after a
failed call (F.4.4), so applying it silently to a 0.8 module would change that
module's meaning. A module of version 0.5 to 0.8 never sees Appendix F, and
modules of different versions may `use` each other (2.7).

**Principles used for derivation.** (a) Least privilege: authority is a value
that is passed, never ambient (5.15). (b) Soundness: a rule that the checker
enforces must be enough for the theorem that relies on it. (c) Determinism:
the same program and log give the same result on every target (T1, T3).
(d) Totality: every evaluation ends (T2, T4). (e) One meaning per construct,
and no meaning that depends on position, tool, platform or flag. A proposal
that breaks (b), (c) or (d) is excluded (F.2), whatever its benefit.

**Reserved words.** A version 0.9 module reserves these ten words in addition
to the 0.8 keywords (except directly after `.`):

```
requires  ensures   decreases ghost     ct
declassify endorse  release   label     law
```

and the type names `World`, `Template`, `Shell`, `Prompt` and `mathint`, which
a 0.9 module may not declare (W-NAME-2), as in 5.16. In a 0.5 to 0.8 module
all of these are ordinary names.

### F.1 How an item is applied

Each item has: the construct, its rules (named W-rules), its cost under C.6,
the theorem it relies on, and the assurance level of 8 (checked, tested or
stated). Tiers are dependency order: an item uses only constructs of its own
tier or an earlier one. Grammar changes are written here in the notation of
01-notation.md and are not yet merged into the checked grammar of section 3;
when they are, `rce-spec-check` must report 0 conflicts before this appendix
loses the word "specified, not yet implemented".

### F.2 Non-goals (excluded because each breaks a theorem)

1. **Effect handlers (resumable or multi-shot).** A handler that resumes a
   computation makes the meaning of a state change depend on where the handler
   is declared and how many times it resumes. That breaks T2 (the bound B_P is
   computed from the call graph, and a resumed computation re-enters it) and
   T3 (the log would need the continuation, not the events). The effect system
   stays the context table of 5.2, and failure stays a value (F.4.4).
2. **Switchable contracts.** A contract that is enforced in one build and
   ignored in another gives one program two meanings, which section 1 and
   decision 38 rule out (there is no build profile that changes meaning).
   Contracts (F.6.2) are always enforced.
3. **Mutable aliasing across `par`, and any relaxation of W-PAR-1.** W-PAR-1
   exists so that every interleaving gives the source-order result (T16).
   Two branches that may write what the other reads have more than one
   result. Branch footprints stay disjoint on writes.
4. **Runtime reflection.** A program that can inspect its own declarations,
   names or types at run time observes what T7 says it cannot: the program's
   names. Compile-time code is pure and cannot reflect either (F.8).
   Legitimate uses are met by closed `derive` of built-in traits (5.9) and
   by tool steps that emit ordinary source.
5. **Other excluded constructs:** size parameters read off values, pointers,
   `alloc`, `bitcast`, uninitialized memory, shared mutable pointers, `unsafe`,
   operator overloading of order operators, inheritance and runtime casts,
   `defer` and other implicit scope-end code, `await`, live variables, coroutine
   iterators, loop-else, comprehensions, glob imports, conditional
   compilation, build features, Python interop and any foreign call outside
   the `World` rule (F.4.1). Each either lets a value depend on something other
   than the program and its log (T1, T3, T7), lets two names alias mutable
   state (T6, T16), or adds a second spelling for a construct that already has
   one (principle e). The reasons are per item in F.8.

### F.3 Tier 0: repair and hygiene

**F.3.1 Stale text (P11).** Items of the specification that did not match the
specification itself or the implementations are corrected in this draft and
listed in 9.7: decisions 8, 11, 12, 21 and 29, the Copy list of 5.5 (it now
includes f32), W-VER-1 and 2.7 (they now name 0.8 and 0.9), and the status
paragraph of the overview. These are errata and apply to every version
(C.15 rule).

**F.3.2 Unused values (P8).** In a version 0.9 module:

- W-UNUSED-1: a `let`, `var`, parameter, pattern binding or `use` name that is
  never read is refused. The wildcard `_` and any name beginning with `_`
  are exempt as the one spelling for "deliberately unused"; a name beginning
  with `_` that is read is also refused (so the spelling means one thing).
- W-UNUSED-2: a `match` arm that cannot be reached (W-MATCH-2 already) and
  an `if`, `elif`, `while` or guard whose condition is a comptime constant
  (9.8, Q17) are refused.
- W-DROP-1: an expression statement whose value is not `()` is refused, except
  a call whose result type is declared `()`. (5.2 already refuses discarding
  an error union; this extends it to every non-unit value.) To drop on purpose,
  bind to `_`: `let _ = f(x)`.
- W-ASSIGN-1: an assignment whose value is overwritten before any read on
  every path is refused (an unused store).

Derivation: principle (e) and the S8 hygiene rule below; a value that is
computed and discarded has no observable effect (O4, no destructors), so it
is dead code or a mistake, and refusing it costs no expressiveness.
Examples and the fuzz generator must be written clean before the rules are
enabled (F.11).

**F.3.3 Lexical hygiene (S8).** In a version 0.9 module:

- W-LEX-4: a source text containing a bidirectional control character
  (U+202A to U+202E, U+2066 to U+2069), a zero-width or invisible format
  character (U+200B to U+200F, U+2060, U+FEFF after the first position) or
  a Unicode scalar value that is not assigned in the pinned Unicode version
  (6.3 N7) inside a string or comment is refused. A string that needs one of
  these writes it with `\u{...}`. Identifiers are ASCII (2.3), so look-alike
  names cannot occur. The reference checker also refuses, in a string or
  comment, the C0 and C1 controls other than the line feed, the other
  invisible or blank-looking format characters (U+00AD, U+034F, U+061C,
  U+115F, U+1160, U+17B4, U+17B5, U+180E, U+2061 to U+2064, U+3164, U+FFA0,
  U+FFF9 to U+FFFB), the tag block (U+E0000 to U+E007F) and Unicode
  noncharacters; for "unassigned" it refuses planes 4 to 13 and the
  noncharacters only, because no general-category table is vendored (F.12).
- W-NAME-3: shadowing. A declaration whose name is visible in any enclosing
  scope of the same function (a local, a parameter, a field of the state
  region, a module-level name or an imported name) is refused with both
  sites. W-NAME-2 refused shadowing only of a parameter, field or
  module-level name; this extends it to every enclosing local.
- Enum variants are always written qualified by their type
  (`Color.Red`), so a pattern name never means a variant by accident (this
  closes the case W-PAT-1 handles by refusal).
- The tokenizer takes the longest token at each position, and no token
  depends on context. This is a requirement on the lexer of 2.3 that the
  reference lexer must be checked against (F.11).

Derivation: T1 and principle (e): the meaning of a token or a name must not
change with a tool, a font or a position.

**F.3.4 Arithmetic and floats (restated, normative for 0.9).** These are
already the meaning of 6.3 and are restated so that Appendix F is complete:
`+ - *` on fixed-width integers trap on overflow; division by zero and an
out-of-range shift trap; wrapping, saturating and checked forms are named
methods (5.15); integers have the same width on every target; one float
operation is one IEEE 754 operation with one rounding (no contraction of
`a*b+c`, no extended precision); a float that would be NaN or infinite traps
(decision 51), so f64 `==` and the total order of 6.3 agree and `Eq` and
`Ord` hold for f64. New in 0.9: a bounded integer type is distinct from its
base (a refined type, 5.1) and an operation whose result leaves the range
refuses, never wraps; division and remainder name their rounding as methods
`div_floor`, `div_trunc`, `div_euclid` and `div_exact` (the last refuses on a
nonzero remainder), with the Euclidean law `a == (a / b) * b + a % b` and
`0 <= a % b < |b|` for `div_euclid`; conversions between int and float are
named methods with a stated rounding and refuse non-finite or
non-representable input; no hidden big integers (`mathint` exists only in
`ghost` code, F.6.3). Obligation: the two classes of `==` and `<` on `-0` and
`+0` must be shown to agree with the total order of 6.3 (check C-F1, F.11).

### F.4 Tier 1: base mechanisms

**F.4.1 `World` capability (P7).** `World` is an opaque type. A `World`
value is attested by the host for the event that starts a command or task,
exactly as a `Cap[A]` is (5.16), and it is subject to the rules of W-CAP-1: it
appears only as the type of a parameter or event payload field, never stored,
sent or captured. It is passed by borrow, `&World`, so a callee cannot keep
it. Rules:

- W-WORLD-1: an `extern` declaration (5.11) has `&World` as its first
  parameter; a call to a foreign function passes a `&World` that the caller
  itself holds; `fn main` and task functions may declare a `&World`
  parameter, and no other context may (a pure function, a step or a handler
  holds none, as 5.2 already forbids the call). A foreign function
  declared without it is W-WORLD-1.
- Time, randomness, files, network and the environment have no primitive
  (5.15). They are inputs: an event carrying the value (for time, an event the
  host supplies with the time as payload) or a foreign result that is logged
  (5.11). A call that is restored and retried (F.4.4) sees the same logged
  inputs.
- Randomness is a generator value passed explicitly with an explicit seed in
  an event, never a global generator. Sleeping and terminal output are
  foreign operations through `&World`.
- A program that must run indefinitely is not written with a loop; it is a
  total step function from (state, input) to (state, output) driven by the
  host, as 6.1 and Appendix E already define. `while true` stays refused by
  the loop bound of 6.7.

Derivation: least privilege (a): a foreign call is authority, so it must be
held by value; determinism (c): every input is an event or a logged result,
so T32 (replay) covers the whole boundary. `main(w: World) -> World` with the
world threaded back was not adopted: the log already sequences foreign
effects (T32), so threading adds nothing for determinism, and a returned
capability could be stored. Cost: `&World` is erased at run time (an index
like the types of 5.16), so δ, the log and the emitters are unchanged.
Theorem: T32 extended to every foreign call (stated, unproved, T53).

**F.4.2 Information-flow labels (S1).** A type may carry a label set:
`τ @ {A, B}`, where each name is declared `label A`. A label set is ordered by
inclusion (⊑); the empty set is public; the join of two sets is their union. A
value of label L may flow only to a place of label L' with L ⊑ L'. `Secret[τ]`
(5.16) is `τ @ {secret}` with a built-in label `secret`; W-SEC-1 becomes a
case of the rules below, and its equality release (`==` between two secrets)
stays the one release without an explicit `release`. Rules:

- W-IFC-1: a flow to a place of lower label (an assignment, a call argument,
  a return, a field, an `emit` field, a send) is refused. An `emit` field and a
  channel have the public label unless declared `@ {...}`.
- W-IFC-2: implicit flows. The label of the control context (the join of the
  labels of the conditions of every enclosing `if`, `match`, `while`, `for`
  and guard) joins into every value assigned and every `emit` or send made in
  it. A loop bound whose label is not public is refused (the number of
  iterations is observable).
- W-IFC-3: provenance. A label name records the input it came from (the event
  field or foreign result that introduced it); the diagnostic of a refused flow
  names the origin and the sink. Provenance is a property of the checker's
  report, not of the value, so it never changes a run.
- W-IFC-4: refusal channel. A trap or `fail` is itself observable (the
  refusal and its text are an output, 6.1). So a trap site whose deciding
  operands (a divisor, an index, an operand of an overflowing operation,
  a size) or whose control context have a non-public label is refused
  unless the operation is a checked form that returns `Option`, whose label
  is the join of its operands. The refusal text of a trap or `fail` carries
  the join of the labels of the operands that decide it; a `fail` payload
  with a non-public label is refused if the function can return it to a
  public context.
- W-IFC-5: robust declassification. `release e` (the only way to lower a
  label) is allowed only in a function that holds a `Cap[A]` (5.16) whose
  authority struct names the label, and only when the control context and the
  operands that decide whether to release have *trusted* integrity. Integrity
  is a second set, `{untrusted}`, carried the same way: event payload fields
  are `untrusted` unless declared otherwise, and `endorse e` (requires a
  `Cap[A]` that names it) is the only way to make a value trusted. A release
  decided by untrusted data is refused, so an attacker cannot steer what is
  released.
- `match`, function calls (the result takes the join of the arguments and the
  control context), the `?` operator, and the derived `Eq`, `Ord`, `Hash` and
  `Show` of 5.9 are label aware: each result carries the join of the labels it
  read. A generic function is checked per instantiation (monomorphized, 5.9).
- Ghost values (F.6.3) carry no label because they are never observable.

Derivation: soundness (b) for T36, which proves noninterference only for
secret equality bits; the general statement needs labels, implicit flows,
the refusal channel (W-IFC-4: otherwise a trap on a secret index leaks the
index) and robust declassification (W-IFC-5: otherwise the declassification
decision is itself an attacker-controlled channel). Cost: labels are erased
at run time. Theorems: T42 noninterference, T43 robust declassification
(stated, unproved).

**F.4.3 Definition hashes and `rce api-diff` (P10).** Every item (function,
type, kind, trait, event, error, constant, state region) has a *definition
hash*: SHA-256 of the canonical encoding (6.1) of its name-free form (6.6)
together with the version, the signature, every contract (F.6.2), `decreases`
measure, ghost clause, label annotation and the definition hashes of the
items it names. A mutually recursive group is hashed as one unit. Tool
metadata, formatting, comments and the order of declarations in a file are not
hashed (Q3). Module membership is an explicit manifest list
covered by the module hash (F.5.3); a module's name is an alias.

`rce api-diff OLD NEW` compares two module sets by exported definition and
prints one row per change from this closed table:

| Change | Class |
|---|---|
| an export is added | compatible |
| a body changes under an unchanged signature, contracts and measure | compatible |
| a `requires` is weakened, or an `ensures` is strengthened | compatible |
| a `requires` is strengthened, or an `ensures` is weakened | breaking |
| a parameter changes from borrow to consume or `&mut`, or the reverse | breaking |
| a record field or an enum variant is added | breaking (a pattern may now be incomplete) |
| an export is removed or renamed, or its kind changes | breaking |
| a parameter type, the result type or the shape of a type changes in any other way | breaking |
| a closure-taking parameter changes from borrowing to escaping | breaking |

Callers rely only on a callee's contracts; an opaque function promises only its
contract, and a *transparent* function (one whose `ensures` fully determines
its result) may be inlined by the checker for proof. Derivation: T7 extended
from event names to items (the hash must not depend on names), and
determinism (c): a build is a function of hashes. The tool is a pure function
of its two inputs. Theorem: T50 (stated, unproved).

**F.4.4 Transactional calls (P1).** In a version 0.9 module a call that
returns by `fail` leaves every `&mut` argument and every owned local of the
callee's caller as they were at the call. (In 0.5 to 0.8 `&mut` arguments are
written back on every outcome, 6.2 and T33; T33 stays true for those
versions.) Rules:

- The restore applies only if every effect since entry is *undoable*. The
  undoable effects are: writes to owned values (locals, `&mut` arguments, the
  state fields of the region of a step) and `emit` into the step's pending
  output (discarded with the event on a trap, 6.4). Channel send and receive,
  `spawn`, foreign calls, `World` operations and host entries are not undoable.
- W-TX-1: a function body in which a non-undoable effect can happen before a
  `fail` on some path (a path from a non-undoable effect to a `fail`, or to a
  call of a function that may `fail`) is refused. So a function that performs
  I/O does not also fail after it; it returns a `Result` value (F.8) the
  caller examines. (Q1.)
- A trap is unchanged: the whole event is refused and the state is as before
  the event (T8). A refusal inside `par` restores the whole `par` (C.4); a
  contract failure in `par` reports the lowest-numbered failing branch.
- A consumed linear value (5.12) is consumed whether or not the call fails;
  consuming a linear handle is not an undoable effect (Q11), so a
  function that consumes a linear value and then may `fail` is W-TX-1.
- Labels (F.4.2): after a restore, the only information in the outcome is
  that the call failed and the `fail` payload, which W-IFC-4 already labels.

Derivation: atomicity (T8) extended from events to calls; the static rule
makes the restore total with no handler stack, so there is nothing to nest
(user-defined handler nesting is excluded, F.2). Cost: the emitters already
copy `&mut` arguments out (T34); a call that may `fail` copies them in
instead, charged as the existing copy-out. Theorem: T47 (stated, unproved).

### F.5 Tier 2: items that use Tier 1

**F.5.1 Capability-only effects (S2).** 5.16 gives `Cap[A]` and `Guarded[τ, A]`;
version 0.9 adds the file-system and revocation rules on top of the same
mechanism. No new ambient authority exists: the process has no current
directory, no environment and no global default.

- W-CAP-3: a foreign function (5.11) that reads or writes a path takes a
  `Cap[DirRead]` or `Cap[DirWrite]` parameter after its `&World`. `DirRead`
  and `DirWrite` are declared authority structs with a `root: text` field. The
  host checks at the call that the path is under `root` after normalizing it
  (the normalization is part of the capability's contract, stated in the
  manifest, F.5.3); the capability's value is logged with the call (5.11), so
  replay does not re-check it.
- Attenuation only narrows: `c.narrow(sub)` returns a `Cap[A]` whose `root` is
  a path extension of c's. There is no operation that widens. A closure that
  captures a `Cap[A]`, `&World` or any value of a non-public label is refused
  (W-CAP-1 extended; F.8, closures): authority held by a closure must be
  visible in its capture list.
- W-CAP-4 (caretaker): a `Cap[A]` that is handed to another component is
  handed as a *caretaker*, a handle the granter keeps and can revoke with a
  logged event `host.revoke(id)`. A revoke is an entry of the log like any
  other, so whether a use is before or after a revoke is a function of the log
  order and replay is exact (T3).
- W-CAP-5: a use of a revoked caretaker is a refusal with the fixed record
  `{kind: "capability", reason: "revoked", id: ...}`; revoking twice is a
  no-op entry; a revoked caretaker is never valid again (there is no
  un-revoke).
- A bound on permission is not a bound on authority: a component with a
  `Cap[DirRead]` can pass what it read to any component it can call. The
  checker bounds which calls it can make (labels, F.4.2), not what a held
  capability may do with data it has already read. The spec states this limit
  so that no reader takes `narrow` for confinement.

Derivation: least privilege (a) and soundness (b): 5.16's `Guarded` rule is
correct only if capabilities cannot be forged, widened or stored; revocation
is added as a log entry because any other form (a mutable flag) would be a
second state that T1 does not cover. Theorem: T51 (stated, unproved).

**F.5.2 Typed output for shell, prompt and structured text (S3).** The
contexts of 5.16 (`Html`, `Url`, `Sql`, `Json`) gain two more, `Shell` and
`Prompt`, with the same mechanism (`Text[C]`, W-CTX-1, made only by
`C.lit`, `C.encode` and `C.t"..."` of F.6.1).

- `Shell`: a command is not a string. `Shell.cmd(program, args)` takes a string
  literal `program` and a `List[Text[ShellArg]]`; `ShellArg.encode(t)` makes
  one argument from any text, and the host runs the program with exactly that
  argument vector, with no shell parsing. There is no `Shell` text a program can
  assemble into a command line.
- `Prompt`: a prompt for a model is `Prompt.t"... {x} ..."` in which the
  literal segments are instructions and each field is encoded by
  `Prompt.encode`, which wraps the field as a quoted data block with a
  delimiter chosen so that the field text cannot contain it (the delimiter is
  derived from the SHA-256 of the field, so it is deterministic). The type
  records that the data inside the block is untrusted; W-IFC-5 then refuses
  to let a model's output decide a release.
- Parsing input is strict and has one meaning: duplicate keys, trailing bytes
  and rebinding are refused (Json input); a missing field has no default; a
  value of a refined type is made only by its validating constructor
  (`R.check`, F.6.2), so deserialization cannot create a value outside its
  predicate; migrations between versions of a persisted record are explicit
  versioned functions, never implicit defaults.
- Text operations name their unit: `len_bytes` and `len_chars` (code points);
  there is no unqualified `len` in a 0.9 module on text. Equality is exact
  code point equality; normalization is an explicit function at input
  boundaries. Identifiers that cannot be represented in a target are refused
  by the emitter, never transliterated.

Derivation: soundness (b) for T36, which covers four encoders; the same proof
shape (every structural metacharacter comes from a literal) extends to the
two new contexts. Theorem: T44 (stated, unproved).

**F.5.3 Hash-pinned supply chain (S5).** A program is built from a *manifest*,
a canonical JSON file (6.1) with one spelling per key; unknown keys, features,
build flags, globs and inheritance are refused. The manifest lists every
module (the explicit membership of F.4.3) with its SHA-256, and every
dependency by hash. Rules:

- W-PIN-1: a module whose hash differs from its manifest entry, a dependency
  that is not pinned by hash, an import that is a glob or relative path, and an
  import cycle (W-MOD-2) are refused. Imports are an unordered set; adding an
  export is always compatible. There is no resolver: the manifest is the whole
  answer.
- A published definition is immutable: a hash names one text forever. A
  correction is a new definition with a new hash. Withdrawal is a signed
  *advisory*, a canonical JSON object `{kind: "advisory", hash, reason,
  issued_by, seq}` signed with Ed25519 (RFC 8032, deterministic) over its
  SHA-256 (Q5). An advisory never changes what a hash means, only whether a
  tool warns on it.
- Compile-time file reads are listed in the manifest and hashed (they are
  pinned imports, not I/O, 5.10).
- A build records the SLSA level 1 provenance: source hash, compiler hash,
  manifest hash and artifact hash, as one log entry. An artifact is a cache
  keyed by the hash of source plus language version; no result may depend on
  whether the cache was used.
- The formatter is canonical and its version is pinned in the manifest; a
  definition's hash is taken over the parsed canonical form, so formatting
  never changes a hash.

Derivation: determinism (c): a build is a function of its pinned inputs.
Theorem: T54 (stated, unproved); it assumes SHA-256 is collision resistant,
which is an assumption, not a theorem of this document.

**F.5.4 Constant-time functions (S4).** `ct fn` declares a function whose
control flow and memory access do not depend on a secret.

- W-CTFN-1: in a `ct fn`, a value whose label contains `secret` (F.4.2) may not
  decide an `if`, `match` or `and`/`or` operand, an index, a loop bound, a
  shift amount, a divisor, or the argument of a call to a function that is not
  itself `ct`. The branchless `ct_choose(c, a, b)` and the bitwise and
  arithmetic operators on fixed-width integers (with the usual trap rules) are
  allowed. Operations that can trap on a secret are refused by W-IFC-4
  already.
- RCE refuses secret branching; it does not compile it away (a compiler that
  rewrites secret branches into selects changes the cost model silently).
- W-CTFN-2: an emitter whose target cannot give constant-time guarantees
  (Python, JavaScript) refuses to emit a `ct fn` with the fixed message
  "ct fn is not emitted for this target"; the reference evaluator runs it
  with its ordinary semantics, since time is not observable there. Whether a
  target does give the guarantee is stated per target in 10.5, not assumed.
- `ct` is compile-time information only; it does not change a value.

Derivation: S1 plus the observation that timing is a channel a label cannot
see; the rule closes it by making the program, not the compiler, secret
independent. Theorem: T45 (stated, unproved): for two runs that differ only in
secret-labelled inputs the sequence of branch and index decisions is equal.
The theorem is about the program's control flow, not about a processor.

**F.5.5 Declared budgets (S6).** A foreign function declared `model` in the
manifest (a call whose cost is chosen by the outside, such as a model
inference) may be called only inside a `with budget B` block (C.5).

- W-BUD-1: a call to a `model` foreign function outside `with budget` is
  refused. The budget is charged at one rule granularity per call (C.6): an
  emitter counts the same units as the evaluator, so `trap("budget")` falls on
  the same operation on every target (10.5).
- When a loop or recursion carries `decreases` (F.6.4) and the bound it
  implies is below the budget, W-BUD-1 does not require a run-time counter for
  it; the budget remains the limit, so the counter is an optimization and
  never a semantic change.
- A budget exhausted is a trap (the event is refused, T8), not a retry.

Derivation: totality (d): a call whose cost the program cannot bound must be
bounded by the program's own budget. Theorem: T52 (stated, unproved).
Implementation cost is the largest of the tier: every emitter must count at
one granularity (10.5).

### F.6 Tier 3: independent language features

**F.6.1 Template text `t"..."` (P6).** A template literal has fixed segments
and named fields: `t"Hello, {name}! You have {n} messages."`. Its type is
`Template[τ]` for the record τ of its fields (here `{name: text, n: i64}`),
and it is not text.

- Fields are `{name}` or `{a.b}`: an identifier or a field path, never an
  expression, and no nested quotes (Q22). A brace not followed by a valid field
  is refused (`{{` and `}}` write literal braces); an empty field is refused.
- W-TPL-1: a `Template` plus text, or a template used as text, is refused. A
  template is made into text only by a renderer: `C.t"..."` for a context C
  (F.5.2) renders each field through `C.encode`; `text.t"..."` renders each
  field through `tostr`, and a field type that is not text must implement
  `Show` (5.9; Q18). There is no implicit locale and no implicit display.
- Equality of two templates is structural: same fixed segments, same field
  names.

Derivation: one meaning per construct (e) and T36 (the typed renderer is the
only place text and structure meet). Cost: a render charges the linear class
of C.6.4 on the total size. Theorem: T44 (the typed render preserves the
literal-only structure).

**F.6.2 Contracts: `requires` and `ensures` (P2).** A function may carry
clauses after its signature:

```
fn withdraw(acct: &mut Account, amount: u64) -> u64
    requires amount > 0
    requires amount <= acct.balance
    ensures result == amount
    ...
```

- Each clause is a pure, total `bool` expression (a refined-type predicate
  fragment extended to any pure function call; `and`, `or`, `not`; no
  implication operator, `not a or b` is the spelling, Q35). `result` names the
  return value in `ensures` and is a reserved name there. `old` does not
  exist; to refer to an entry value, bind it with `ghost let` (F.6.3) before
  the first statement, which a clause may name.
- Contracts are *always enforced*: `requires` is checked on entry and
  `ensures` on every return. A violation is one defined refusal (a trap of
  the call's event) with the fixed record `{kind: "contract", clause:
  "requires" | "ensures", function: <definition hash>, index: <clause number
  in written order>}`. No source span (it would make output depend on
  whitespace, Q28). No unspecified behaviour exists, so C++26's "ignore" and
  "observe" semantics are not adopted (F.2, item 2).
- A contract with no satisfying value is an error: W-CON-1 refuses a function
  whose `requires` clauses are, in the decidable fragment of W-REF-1, false
  for every argument. Outside that fragment nothing is decided statically
  (Q33); the enforcement makes such a function refuse every call, never
  succeed vacuously.
- The checker may assume a callee's `ensures` after the call and must prove the
  callee's `requires` before it where the fragment decides it, and otherwise
  the call is checked at run time by the enforcement. Contracts are never
  inferred and are covered by the definition hash (F.4.3).
- A refined type `type R = τ where p` (5.1) has the total function
  `R.check(v: τ) -> R?` that returns `some` exactly when p holds (Q15).
  There is no witness clause and no implicit conversion from τ to R.
- A type with an invariant is opaque outside its module: construction only by
  the validating constructor; fields are not exported.
- A function value (a closure or `fn` value) has no contracts in 0.9, and a
  function with `requires` becomes a value only wrapped to return `Option` or
  `Result` (F.8).

Derivation: soundness (b), P2: a callee's meaning is its contract; an
enforced check is the cheapest rule that keeps every theorem (the check is an
ordinary guard, so δ stays total and T8 gives the rollback). Cost: one
comparison per clause, charged as C.6.4 operators. Theorem: T46 (stated,
unproved).

**F.6.3 `ghost` (P3).** `ghost let x = e`, `ghost fn` and `ghost` parameters
exist only for contracts and measures.

- W-GHOST-1: a ghost value may not flow into a non-ghost place, into an
  `emit`, a send, a branch condition of real code, a loop bound or an index.
  Ghost code is erased before evaluation, emission and hashing of the *run*
  (it is hashed in the definition, F.4.3). Ghost code is total (F.6.4) and
  pure.
- `mathint`, the mathematical integer, exists only in ghost code and in
  contracts (Q31). It is unbounded and has no runtime representation.
  Quantifiers, `forall` over a finite range or collection, are allowed in
  ghost expressions only; an unbounded quantifier is refused (it cannot be
  evaluated, and enforced contracts must be).
- Ghost values carry no label (F.4.2) and nothing is auto-initialized.

Derivation: noninterference (b): ghost data are not observable, so they may
not steer observable data. Theorem: T49 (stated, unproved): erasing ghost
code does not change δ or λ.

**F.6.4 `decreases` (P4).** A recursive group (6.7) and a loop may carry
`decreases m1, m2, ...`, a lexicographic tuple of measures.

- Each measure is a natural number (`u64` or `mathint` nonnegative), finite
  data under the structural order of sums and records, or a tuple of these.
  Anything else is refused (W-DEC-1).
- W-DEC-1: every call to a function of the same group, from any path, passes a
  tuple strictly smaller in the lexicographic order than the caller's entry
  tuple. A group's measure is one tuple for the whole group (Q2); a tail call
  does not waive the check.
- `decreases` is required for a recursive group that the structural rules
  or the layered summed measures of 6.7 do not already admit; where they admit
  it the clause is optional, and a written measure is checked and hashed.
  It never replaces the loop bound and the run-time `LOOP_LIMIT`: both hold
  (so T2's bound is unchanged and the proof of 6.7 stays valid).
- There is no unbounded `loop`: loops iterate a finite collection, a bounded
  range or a `while` with `decreases` and the loop bound of 6.7. Range bounds
  are evaluated left then right, both always. `first` is a search that returns
  `Option`.
- Strict positivity holds without a rule: a recursive `kind` mentions itself
  only as a direct field, because a `fn` type cannot occur inside a data type
  (W-TY-2) (Q23).

Derivation: totality (d), T2 and T4. Theorem: T48 (stated, unproved).

**F.6.5 `if let` (P5).** The statement is written in the block form of every
other statement of this language (blocks are indentation, 3.7):

```
if let P = e and c1 and c2
    A
else
    B
```

where `P` is a pattern of 5.3, `e` and each `c` are comparison expressions
(`CmpExpr`, so an `and` or `or` inside needs parentheses), and `A` and `B` are
blocks. It means the `match` with the arm `P if c1 and c2` (the clauses `c`
are evaluated left to right, only after `P` matched) running `A`, and a
wildcard arm running `B`; `e` is evaluated once. Bindings of `P` are in scope
in the `c` and in `A` only, and the borrow of the scrutinee ends with the
match, so `B` does not hold it (a `B` that changes the place `e` read is
allowed). The condition is a `bool` always (no truthiness); a failed pattern
or a false `c` goes to `B`; without `else` the failed case does nothing, and
`A` then has type `()`. There is no `elif` after an `if let`. A condition has no side effects
to roll back, because expressions in conditions are pure (W-EFF-1 context
rules, 5.2). W-IFLET-1: an operand `c` that is not a comparison is refused,
and a binding unused in `A` is W-UNUSED-1. An irrefutable pattern leaves the
wildcard arm unreachable and is refused by W-MATCH-2. The spelling of the
first draft, `if let P = e then A else B` with a chain `and let P2 = e2`, is
not the grammar of this language and is withdrawn: a chain with a second
`let` is specified for a later draft and is refused by version (W-VER-2).

Derivation: determinism (c); one construct style (e): the other statements
take blocks, not `then`. `if let` is sugar over `match`, so no theorem
changes (T10 holds because the grammar delta of F.13 is LALR(1), F.11). Not an
assurance claim beyond that.

**F.6.6 Trait laws (P9).** A `trait` may declare `law name(p: τ, ...): e`, a
pure total `bool` expression over the trait's methods. `impl` need not repeat
it. Conformance is explicit, declared where the type or the trait is defined
(the orphan rule of 5.9), all or nothing, with no duck typing and no partial
conformance; conflicting default methods are refused (W-TRAIT-1, W-TRAIT-2
extended). A generic function is bounded by traits only, with one instance
per trait and type, no variance and no subtyping.

- W-LAW-1: admission (6.9) evaluates every law of every impl on a fixed,
  deterministic sample: the values of the impl type that appear in `example`s
  and `check`s of the program, the boundary values of its fields (zero, one,
  the extremes of the range, empty, a singleton), and their pairs and triples
  up to 64 tuples in canonical order. A law that fails on any is refused.
- This is a test, not a proof. The law is not assumed by any theorem; an
  implementation may not use a law to rewrite code. Assurance level: tested.

Derivation: soundness (b): promising a law that is never checked is a hole;
promising only that it was sampled is true. Theorem: none (stated as tested).

### F.7 Tier 4: tooling contract (no language or emitter change)

**F.7.1 The editor and agent broker (S7).** A broker is a tool process that
mediates an assistant's use of a user's files, secrets, network and
processes. The *lethal trifecta* is the combination, in one session, of
(1) access to private data, (2) exposure to content an attacker can write, and
(3) a way to send data out. The broker contract:

- TRIFECTA-1: the broker tracks the three properties per session as grants. A
  session holds at most two without an explicit user grant event for the
  third. A grant is a logged event (T3), so what a session could do is a
  function of its log.
- TRIFECTA-2: content from outside the user's own workspace is `untrusted`
  (F.4.2). A tool call whose arguments carry `untrusted` data may not use a
  capability that is not already granted to untrusted data, and a broker
  never resolves ambiguity silently: it shows the call and waits for an
  event.
- TRIFECTA-3: outbound sends carry the label join of everything the session
  has read; the broker refuses a send whose label is not below the
  destination's, as W-IFC-1 does for programs.
- W-TRI-1 (static): a program that has `private` and `untrusted` inputs and an
  external sink with no `release` path that satisfies W-IFC-5 is refused; the
  rule is a consequence of W-IFC-1 and is stated so the broker and the
  checker agree.
- No program observes timing or hidden tool state: `rce` tools never expose a
  clock or an address to a program, proof automation reports its cost and
  keeps budgets, and an automation that uses a solver records a certificate or
  an audited axiom.

The broker invariant follows by induction on grants (T55, stated, unproved).
Residual risks are stated, not hidden: the user can grant all three, a tool
outside the broker is outside the contract, and a model's output is treated as
untrusted data, never as an instruction the checker can verify.

### F.8 Further rules (derived, grouped)

Each row states a rule of version 0.9 and its derivation; "already" means the
text of the cited section already gives that meaning and 0.9 only states it
where a reader would look.

| Topic | Rule | Derivation |
|---|---|---|
| Closures | Closures capture immutable values by copy or move, with an explicit capture list; no captured `var`; a closure that borrows must not escape (5.6); a capture of `&World`, `Cap` or a non-public label is refused (F.5.1) | T6, T16, least privilege |
| Types | No subtyping, no `any`, no implicit conversion, no cross-type equality, no implicit join of unlike `if` branches or array elements; generics are invariant | T6, T7 |
| Enums and `match` | Closed sums; exhaustive (W-MATCH-1); duplicate and unreachable arms refused (W-MATCH-2); variants qualified | T1 |
| Calls | One function per name in a scope (W-NAME-2), no overloading, no default parameters, arguments evaluated in written order; argument labels (decision 18) are checked: a wrong label is W-TY-1, so a label is documentation the checker verifies (Q16) | T1 |
| `par` | Arms are expressions; at least two; `return`, `break` and `?` written directly in an arm are refused (Q4); a refusal in any arm restores the whole `par` (C.4); a right side that writes what the place reads is a conflicting borrow; W-PAR-1 is unchanged | T16 |
| Assignment | Place operands first, left to right, then the right side | T1 |
| Imports | Acyclic; no glob; unordered set; explicit exports, nothing public by default | T1 |
| Comments | Only `#` line comments exist (Q34); they never nest and nothing inside a string starts one | T1 |
| Literals | One spelling per literal: a single `_` between digits is the only separator (2.5) (Q36); no `1.` or `.5`; one string form; `char` is a Unicode scalar value | T1 |
| Operators | Comparison operators are non-associative: a chain is refused (W-OP-1), no chain sugar (Q8); mixing bitwise, shift, arithmetic or `and`/`or` in one expression needs parentheses (W-OP-1); `and` and `or` short-circuit, apply to `bool` and return `bool` | T1, contracts need the left operand as a hypothesis for the right |
| Order operators | `<` is a total order on ordered types only; subset, prefix and merge are named methods | T1 |
| Containers | `Some`/`None` are explicit, never `false`; no tuple coercion; duplicate map-literal keys refused; map and set iteration in canonical `Ord` order, and equality is order independent because the representation is canonical (Q6, Q19); no weak references | T1, T6 |
| Indexing | `xs[i]` is refused unless the checker proves `i < len(xs)`; `get(xs, i)` returns `Option`; slices are as in 6.3 N8; ranges keep their two spellings, `..` and `..=` (Q7) | totality |
| Classes | There are none: no inheritance, no virtual dispatch, no runtime casts, no identity equality, no field defaults; constructors are ordinary pure functions | T6, T7, T16 |
| Errors | Failure is a `Result` value with a closed enum of error kinds; no `raise`, `try ... finally`, `with`; `refuse` is not a value and has no type (Q24) | T1, F.4.4 |
| Numeric names | Named conversions with stated rounding; non-finite refused; NaN does not exist (Q9) | T1 |
| Pointers and memory | None; recursive data are `kind` with `Box` (5.1); vectors have no layout promise (Q29) | T6, T16 |
| Reflection and derive | None at run time or compile time; `derive` is the closed built-in set of 5.9, user-defined `derive` is not offered (Q20); a type whose field lacks the trait cannot derive it (W-TRAIT-3) | T7 |
| Compile-time code | Pure, total, no declaration injection, no dependence on its call point, no platform conditionals, no `assume` (a `comptime assert` over symbolic values is refused); types are not values; value parameters only in the array-length fragment (Q10) | T1, T7 |
| Linear values | Capability and resource handles are consumed exactly once by named functions (5.12); a type parameter is instantiated only by non-linear types; no implicit `Discard` (Q26) | T12 |
| Numeric types | Bounded integers are distinct types; out-of-range refuses (F.3.4) | T14 |
| Function values | Total pure `fn` types only; a partial function is wrapped to return `Option` or `Result` | T2 |
| Iteration | Explicit finite folds and step functions; no coroutine iterators except `iter` (5.14) | T15 |
| Programs | A reactive program is a total step function driven by the host (Q32); an interpreter and a compiler must agree, and the conformance suite compares them | T1, T9 |
| Foreign | A foreign boundary is declared in the manifest with its full signature and library hash and is a `World` operation (Q30); raw pointers, C struct layouts and path-searched libraries are refused; Python interop is refused | T6, T32, T54 |
| Tooling | No flag changes meaning; build artifacts are caches; the audit report is a committed file and the checker compares against it; tests run in canonical order; plugins report only; emitter names are injective and never collide with a target's reserved words | T1 |
| Release gate | Lean proofs are built with `lake build`, no `sorry`, only the standard axioms (8); theorem statements are frozen in a module and compared; the Lean model is the reference semantics and every backend is tested against it | T9 |
| Diagnostics | A diagnostic names the true rule or refuses; version-unstable text is never a program value | T7 |
| Persistence | Persistent storage only through `World` with validating constructors | T32 |

### F.9 Theorem obligations (stated, unproved)

These rows are copied into section 8 with status *stated, unproved*. None has a
Lean statement, a checker test or a counterexample search yet.

| | Statement |
|---|---|
| T42 | Noninterference with labels: two runs whose inputs agree on every public-label field give equal public outputs, including the refusal outcome, for every accepted 0.9 program (F.4.2) |
| T43 | Robust declassification: a release is reached only on a path whose control context and deciding operands are trusted (F.4.2) |
| T44 | Typed output: every structural metacharacter of a rendered `Text[C]` or `Template` of any context of 5.16 and F.5.2 comes from a literal segment (F.5.2, F.6.1) |
| T45 | Constant time: two runs of a `ct fn` that differ only in secret-labelled inputs take the same sequence of branch, index and loop decisions (F.5.4) |
| T46 | Contract enforcement: if a call returns, its `requires` held at entry and its `ensures` held at the return; otherwise the call's event is refused with the fixed record (F.6.2) |
| T47 | Transactional calls: a call that returns by `fail` leaves the caller's `&mut` arguments and locals equal to their values at the call (F.4.4) |
| T48 | Termination by written measure: an accepted recursive group or loop with a `decreases` tuple ends (F.6.4) |
| T49 | Ghost erasure: erasing ghost code does not change δ, λ or the log (F.6.3) |
| T50 | Definition hash: two definitions with equal hash have equal meaning; a change that `rce api-diff` marks compatible does not change the result of any accepted caller for inputs satisfying its `requires` (F.4.3) |
| T51 | Revocation: no capability use after a `host.revoke` entry for its caretaker in the log is admitted (F.5.1) |
| T52 | Budgets: a program with a budget does at most the budget's units of work before `trap("budget")`, at the same operation on every target (F.5.5) |
| T53 | World replay: T32 for every foreign call through a `World` and every logged input (F.4.1) |
| T54 | Supply chain integrity: under SHA-256 collision resistance, an accepted build's modules are exactly the manifest's (F.5.3) |
| T55 | Broker invariant: by induction on grant events, no session holds all three trifecta properties without an explicit grant event (F.7) |
| T56 | Relaxed sequential borrows: the limited relaxation of 5.6 studied from group borrowing (a reference is invalidated only below steps that are not independent of it) preserves T6; the lemma is not proved and the relaxation is not adopted (5.6) |

**Not adopted until proved.** The relaxed sequential borrow rule studied from
Valen's group-borrowing design is sound in a limited form, and T56 is its
obligation. 5.6 is unchanged in 0.9. Relaxing W-PAR-1 is not sound and is
excluded (F.2, item 3).

### F.10 Cost, emitters and compatibility

- Labels, `World`, ghost code, contracts' proofs and indices are erased
  before evaluation; only enforced contract clauses, `R.check`, encoders
  (F.5.2), `ct_choose` and the restore of F.4.4 cost anything at run time,
  each charged as the C.6.4 operator or linear class named in its rule.
- An emitter that cannot implement a rule exactly refuses the program with a
  fixed message and never approximates (10.5); the targets for `ct fn` are
  stated in F.5.4.
- Compatibility: a 0.5 to 0.8 module is unchanged. A 0.9 module may `use` a
  module of any version and the reverse. Moving a module to 0.9 can make it
  refused by F.3.2, F.3.3, F.4.2, F.5.1 and F.4.4; it is never silently
  re-read.

### F.11 Checks before this appendix is implemented

1. Merge the grammar delta (keywords, `t"..."`, clauses, `if let`, `ct fn`,
   `ghost`, `release`) into section 3 and require 0 LALR(1) conflicts, with
   the tree-sitter port agreeing (T10).
2. Add 0.9 to the checker's known versions (done) and implement each W-rule of
   this appendix with accept and refuse examples (the implemented ones are in
   `spec/examples-0.9/`, one refused file per rule, checked by
   `tests/v09_examples.rs`; the rest are listed in F.12); the fuzz generator must produce 0.9 programs
   clean of F.3.2.
3. C-F1: show that `==` and `<` on f64 `-0` and `+0` agree with the total
   order of 6.3 (F.3.4); C-F2: check the reference lexer against the
   longest-token and context-free rules (F.3.3).
4. State T42 to T56 in Lean models and prove them in order of use; until a row
   of section 8 names a Lean declaration, its status stays *stated, unproved*.
5. Run the conformance suite across evaluator and emitters for the 0.9
   programs, including budget sweeps across trap points (F.5.5).

### F.12 Implementation status of the reference checker

Implemented and tested (`spec/checker/src/v09.rs`, `apidiff.rs`; examples in
`spec/examples-0.9/`; `rce v09-status` prints this list):

- W-VER-1 accepts `0.9`; W-VER-2 refuses the ten reserved words and the type
  names `Template`, `Shell`, `Prompt` and `mathint` (their constructs are not
  implemented), except directly after `.`.
- W-LEX-4 (F.3.3) as stated there, with the extra classes listed there.
- W-UNUSED-1 (F.3.2), including the `_` spelling and a read of a `_` name;
  W-UNUSED-2 for a literal boolean condition; W-DROP-1 (in the type checker);
  W-ASSIGN-1 for an overwrite later in the same block; W-NAME-3 (shadowing of
  any enclosing local, parameter, state field or module-level name).
- W-WORLD-1 (F.4.1): `&World` is held only by `fn main`, task functions and
  externs, is the first parameter of every extern, is passed whole and by name,
  and is never stored, returned, sent or captured. The checker erases it before
  evaluation, so the Form, the log and every emitter are unchanged
  (`rce form` shows the erased module).
- Definition hashes, `rce hash` and `rce api-diff OLD NEW` (F.4.3), without
  contracts, measures, ghost clauses or labels, which do not exist yet.
- W-TX-1 and the restore of `&mut` arguments on `fail` (F.4.4) in the reference
  evaluator. A `catch`ed call is not a fail point of its caller. The emitters
  and the Lean elaboration refuse a 0.9 module with a function that can fail
  after taking a `&mut` argument.

- Information-flow labels (F.4.2, S1): `T @ {A, B}` label sets on declared
  types (parameters, returns, fields, `let`, `var`), `label NAME [by AUTH]`
  declarations (W-IFC-6 for an undeclared label), label propagation with the
  control-flow label (implicit flows), W-IFC-1 (a labeled value reaches a public
  sink: `emit`, `send`, a store into a public place, a call result), W-IFC-2 (a
  labeled value decides a loop count), W-IFC-4 (an operation that can trap, and
  a `fail`, whose deciding operand or control context is labeled), W-IFC-5
  (`release` needs the held `Cap[AUTH]` and a trusted deciding context, and
  `endorse` needs `label untrusted by AUTH` and the held capability; event
  payloads are `untrusted` unless declared `@ {}`), with the source of each
  label named in the message and a per-instantiation check of calls. W-NAME-2
  refuses `label`, `release` and `endorse` used as names. The checker erases the
  labels, `release` and `endorse` before evaluation and emission, so a labeled
  program runs as the unlabeled one. `declassify` stays refused by version.
  The grammar delta is F.13; the Rust LALR(1) tables (0 conflicts), the
  tree-sitter grammar and its scanner, `parser.c` and the wasm are regenerated
  from it.

- S2 directory capabilities (F.5.1): W-CAP-3 (a foreign function with a `path`
  or `*_path` parameter takes a `Cap[DirRead]` or `Cap[DirWrite]` after its
  `&World`; `DirRead` and `DirWrite` are declared structs with `root: text`;
  `c.narrow(sub)` is typed `Cap[A]` and refused unless `sub` is, when a
  literal, a relative path with no `.` or `..` segment); there is no operation
  that widens. A closure that captures a `&World` or a capability, by declared
  type, is W-CAP-1. A capability is `()` at run time, so the checker erases
  the directory capabilities of a foreign call like the world, and `narrow`
  evaluates to `()`. The caretaker semantics of W-CAP-4 and W-CAP-5 are
  implemented as a pure function of the log (`caretaker.rs`: admit, refuse
  with the fixed record `{id, kind: "capability", reason: "revoked"}`, revoke
  twice is a no-op, no un-revoke), with unit tests. The call `host.revoke(id)`
  needs the host log, which the reference checker does not have, so it is
  refused by version (W-VER-2).
- S3 typed output (F.5.2): the contexts `ShellArg` and `Prompt` (with `lit` and
  `encode`; the `Prompt` delimiter is `data-` and the SHA-256 of the field,
  re-hashed until the field does not contain it) and `Shell`, made only by
  `Shell.cmd(program, args)` with a literal program and `List[Text[ShellArg]]`
  arguments, whose value is the canonical JSON `{"args": [...], "program":
  ...}` (an argument vector, never a command line). `len_bytes` and `len_chars`
  exist and `len` on text is W-LEN-1. Strict parsing, versioned migrations and
  the refined-type constructor `R.check` are not implemented. The templates
  `Prompt.t"..."` of F.6.1 are not implemented either (see below).
- S6 `model` foreign functions (F.5.5): `extern "model" fn` declares one, and
  W-BUD-1 refuses a call outside `with budget`. The budget counting is the
  one of C.5 and C.6, unchanged. No rule of Appendix F bounds allocation, so
  none is implemented; the arena capacity of 5.7 is the existing bound.
  Neither the emitters nor the Lean elaboration translate `narrow`, `Shell`,
  `ShellArg`, `Prompt`, `len_bytes` or `len_chars`: they refuse a module that
  uses one (W-VER-2).

- S4 `ct fn` (F.5.4): `ct` is a keyword in a 0.9 module (a stray use is
  W-NAME-2). W-CTFN-1 refuses, inside a `ct fn`, a `secret`-labeled value (F.4.2)
  that decides an `if` or `match`, an `and` or `or` operand, an index, a loop
  or `for` bound, a shift amount, a divisor, or an argument of a call to a
  function that is not a `ct fn` (a foreign function included); it is checked
  per call instantiation like the labels. `ct_choose(c, a, b)` evaluates both
  values and chooses one. W-CTFN-2: every emitter and the Lean elaboration
  refuse a module with a `ct fn` ("ct fn is not emitted for this target"):
  10.5 states no constant-time guarantee for any target, so none is assumed.
  The reference evaluator runs it with ordinary semantics; `ct` changes no value.

- S5 hash-pinned manifest (F.5.3, `rce manifest-check`, `manifest.rs`): the
  manifest is canonical JSON with the keys `dependencies` (list of `{name,
  sha256}`), `formatter`, `language` (the string "0.9"), `modules` (list of
  `{module, path, sha256}`) and optional `reads` (list of `{path, sha256}`,
  the compile-time file reads); one final line feed is ignored and any other
  non-canonical spelling is refused. The hash of a module is the SHA-256 of
  its canonical unparse, so formatting never changes it. W-PIN-1 refuses an
  unknown key, a module whose hash differs from its entry, a dependency not
  pinned by 64 lowercase hex digits, a path that is a glob, absolute or has a
  `.` or `..` segment, an import that is neither a listed module nor a pinned
  dependency, and an import cycle (W-MOD-2). A signed advisory is `{advisory:
  {kind: "advisory", hash, reason, issued_by, seq}, signature}` with the
  Ed25519 signature (RFC 8032, via `ring`) over the SHA-256 of the canonical
  JSON of the `advisory` object, by the key trusted for `issued_by` (given on
  the command line, `--key ISSUER=HEXPUBKEY`); `seq` of one issuer must
  increase. The lockfile's unsafe list is the pinned module and dependency
  hashes that carry a verified advisory; the tool prints them as `UNSAFE` and
  the manifest still checks (an advisory never changes what a hash means).
  SLSA provenance records, the build cache rule and the key distribution of
  advisories are not implemented. T54 stays unproved and assumes SHA-256 is
  collision resistant.

- P2 contracts and P3 `ghost` (F.6.2, F.6.3, `tier3.rs`): `requires E` and
  `ensures E` are statements (`ContractStmt`) written before the first
  statement of a fn body, after any `ghost let`; a clause elsewhere is
  refused (W-CON-2). Clauses are enforced always: the checker lowers the fn
  to guards, `requires` first in written order, then the kept ghost lets, then
  the body in which every `return` (and the tail value) first binds `result`
  and runs the `ensures` guards. A violated guard traps with the text
  `{"clause":"requires"|"ensures","function":"<definition hash>","index":N,
  "kind":"contract"}` (keys in that sorted order, no spaces), where N counts
  all the clauses of the fn in written order from 0 and the hash is the
  definition hash of the fn as written (F.4.3: for a top-level fn the
  `rce hash` value, for a method the SHA-256 of its canonical text under the
  same prefix). There is no source span. W-CON-1 refuses `requires` clauses
  that are false in the fragment of literal comparisons, `and` and the integer
  bounds of one name (`x > 5 and x < 3`); W-CON-2 refuses `result` in
  `requires`, a binding named `result` in a fn with `ensures`, and a channel
  receive in a clause. `ghost let` and `ghost fn`: W-GHOST-1 refuses a ghost
  name in code that runs; ghost code that no clause names (transitively) is
  erased, the rest is evaluated because an enforced clause needs it. The
  emitters refuse a module with a contract (W-VER-2). Not implemented:
  `ghost` parameters (W-VER-2), `mathint` and quantifiers, `R.check`, opaque
  types with invariants, assuming a callee's `ensures` and proving its
  `requires` statically (a call is checked by the enforcement), contracts on
  function values. T46 and T49 stay unproved.

- P4 `decreases` (F.6.4): `decreases m1, m2, ...` is a leading statement of a
  fn body (`ContractStmt`). W-DEC-1 decides this fragment: each measure is an
  expression linear in the unsigned integer parameters (`u8` to `u128`) of the
  fn, and every call to a fn of the same recursive group (the strongly
  connected component of the call graph among top-level fns, a fn that calls
  itself included) passes a tuple that is strictly smaller in the
  lexicographic order: the first position where the entry tuple and the
  callee's measure with the arguments substituted differ by a constant
  difference, and the positions before it are equal. A member of a group
  without a `decreases`, a different arity, a signed or non-linear measure
  and an argument outside the fragment are refused. A measure has no run-time
  meaning (the recursion bound of 6.7 and the loop limit still hold) and is in
  the definition hash. A `while` may carry a measure as the first statement of
  its body (the loop form of W-DEC-1): each part is linear in the unsigned
  integer parameters and in the `var` names with a written unsigned type, each
  bound once in the fn; the top-level assignments of the body (`x = e`,
  `x += e`, `x -= e`, `e` linear) give the measure at the end of the body in
  terms of the entry values, and that tuple must be strictly smaller in the
  lexicographic order (a constant difference at the first position where the
  two differ). A measure name assigned anywhere else in the body, borrowed
  mutably, or a `continue` in the body, is outside the fragment and is
  refused with W-DEC-1; a `decreases` on a `for` or `loop`, or anywhere else
  than the start of a fn body or of a `while` body, is refused by version
  (W-VER-2). The loop measure is erased before evaluation and emission, and
  `bound` and `LOOP_LIMIT` still hold. The checker does not prove that a
  measure stays a natural number (the function form does not either).
  Not implemented: structural measures over data, `mathint` measures, calls of
  methods or through function values (not followed). A call is judged on its
  own, not on the branch it is in. T48 stays unproved.

- P5 `if let` (F.6.5, block form): `IfLetStmt` of F.13, carried in the
  parser as `match` on `@iflet(e)` with the guard of the `and` comparisons and
  a wildcard arm (`pass` when there is no `else`); the formatter writes the
  `if let` back. W-IFLET-1 and the checks of `match` apply. A chain with a
  second `let` is refused by version.

- P6 templates (F.6.1): `TEMPLATE` is a token (F.13). Only the two rendered
  forms are implemented: `text.t"..."` and `C.t"..."` for `Html`, `Url`,
  `Sql`, `Json`, `ShellArg` and `Prompt`. The parser expands them at once to
  the concatenation they mean (`text`: literal segments and `f.tostr()` of
  each field; a context: `C.lit` of each segment and `C.encode` of each
  field, joined with `+`), so the fields are ordinary reads and the context's
  typing (W-CTX-1) decides what a field may be. Fields are an identifier or a
  field path; `{{` and `}}` are literal braces; W-TPL-2 refuses an empty
  field, an expression, a stray brace and an unclosed field; W-TPL-1 refuses
  a renderer that is neither `text` nor a context. A template that is not
  rendered at once is the `Template[τ]` value of F.6.1, which is refused by
  version, so W-TPL-1's text-plus-template cases cannot arise yet. The
  formatter writes the expansion, not the template (the meaning, and so the
  hash of the expansion, is the same). T44 stays unproved.

Not implemented (each is refused by version, or the closed list in
`crate::v09::UNIMPLEMENTED` says that it is not enforced):

- trait laws (P9, F.6.6): `law` is a reserved word that is refused by version;
  W-LAW-1 needs a sample of impl values and an evaluator of a law over them,
  and neither exists, so no law is ever run or assumed;
- `declassify` (F.4.2): always refused; `release` with a held capability is the
  one way a labeled value leaves its label;
- `ghost` parameters, `mathint` and quantifiers; `R.check` and opaque types
  with invariants; `Template[t]` as a value; structural measures over data
  and loop measures on `for` and `loop`; a chain `and let` in `if let`;
- the qualified spelling of variants (F.3.3) is not enforced; S7 (F.7) belongs
  to the IDE broker and not to this checker; the allocation bound of S6 has no
  statement in F.5.5 to implement; provenance records, the build cache and the
  advisory key distribution of S5.

**Read receipt.** How many words does a version 0.9 module reserve in addition to the 0.8 keywords? (a number)


### F.13 Grammar

These productions extend section 3 and Appendix C.2 for the constructs of
version 0.9 that the reference checker implements (the label sets, label
declarations, `release` and `endorse` of F.4.2, `ct fn` of F.5.4, and the
contract clauses, `ghost let` `ghost fn` and `decreases` of F.6.2 to F.6.4, `if let` of F.6.5, and the template renderers of F.6.1). A rule below with the name of
an earlier rule replaces it for a 0.9 module; the others are new. The machine
check of 3.12 builds the LALR(1) tables of section 3 with C.2 and then this
section applied (`lalr::book_grammar_09`), lexing with `label`, `release`,
`endorse`, `ct`, `requires`, `ensures`, `decreases` and `ghost` reserved and `by` contextual, and fails on any conflict. It runs that
machine over `spec/examples-0.9/ok`. `TEMPLATE` is the token `t` written directly before a STRING (no space,
and `t` not the end of a longer name); the checker accepts it only after a dot.
The tree-sitter port
(`spec/tree-sitter/grammar.js`) carries the same productions. The constructs
of F.5 to F.7 that are still refused by version (F.12) have no production yet.

```
Item         = UseDecl | IntentDecl | ExampleDecl | CheckDecl
             | [ "pub" ] ( ConstDecl | TypeDecl | StructDecl | KindDecl
                         | TraitDecl | FnDecl | ActorDecl | ProtocolDecl )
             | StateDecl | EventDecl | ErrorDecl | StepDecl | ImplDecl
             | ExternDecl | NodeDecl | SupervisorDecl | LabelDecl .
LabelDecl    = "label" NAME [ "by" NAME ] NEWLINE .
FnDecl       = [ "ct" | "ghost" ] FnSig ( Block | "=" Expr NEWLINE ) .
Stmt         = SimpleStmt | LetStmt | VarStmt | IfStmt | WhileStmt | LoopStmt
             | ForStmt | MatchStmt | ReturnStmt | BreakStmt | ContinueStmt
             | EmitStmt | FailStmt | SpawnStmt | SelectStmt | ArenaStmt
             | DeviceStmt | YieldStmt | ComptimeStmt | CheckDecl | PassStmt
             | ParStmt | BudgetStmt | DeadlineStmt | GroupStmt | BarrierStmt
             | ScopeStmt | AtomicStmt | ContractStmt | GhostLet | IfLetStmt .
IfLetStmt    = "if" "let" Pattern "=" CmpExpr { "and" CmpExpr } Block [ "else" Block ] .
Postfix      = Primary
             | Postfix "(" [ Args ] ")"
             | Postfix "[" Expr { "," Expr } "]"
             | Postfix "[" [ Expr ] ":" [ Expr ] "]"
             | Postfix "." ( NAME | INT | TEMPLATE ) .
ContractStmt = ( "requires" | "ensures" ) Expr NEWLINE
             | "decreases" Expr { "," Expr } NEWLINE .
GhostLet     = "ghost" LetStmt .
RegionType   = BaseType [ "@" ( NAME | LabelSet ) ] .
LabelSet     = "{" [ NAME { "," NAME } ] "}" .
UnaryExpr    = ( "-" | "~" | "&" [ "mut" ] | "<-" | "comptime" | "release" | "endorse" ) UnaryExpr
             | Postfix .
```

<!-- body-sha256: 56e6f134cbb5fa33512a5e189e42d3fca4a6d7b250d25b0ada08032b0d7a3b37 -->
