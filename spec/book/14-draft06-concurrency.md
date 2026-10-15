## Appendix C. Draft 0.6 and 0.6b concurrency (normative for version 0.6 and 0.7 modules)

This appendix defines the draft 0.6 language: structured parallelism
(`par`, `for ... par`), budgets, deadlines, nondeterministic select and
priorities (0.6), and the 0.6b extensions: workgroups, session types,
structured spawn, cross-actor transactions, supervisors, nodes,
interrupts, LVars and replicated state. It applies only to a module that
declares `# rce-version: 0.6` or `# rce-version: 0.7` (2.7); everything
said here of a 0.6 module holds for a 0.7 module, except where C.6.4 and
C.15 say otherwise. A 0.5 module never sees any of it:
its words are ordinary names there, and none of its rules apply. The
text is written from what the reference checker (`spec/checker`,
`check.rs`, `check06b.rs`, `conc.rs`, `link.rs`) and the Lean files
(`spec/lean/RCE/Par*.lean`, `Models06b.lean`, `Session*.lean`,
`TickClock.lean`, `VClock.lean`) actually enforce and prove; where the
two differ from an intended design, the checker is what is described.
Example: in a 0.6 module, `par` with two branches that write different
variables gives the same result as running the branches in source order.

### C.1 Version and keywords

The 23 words below are reserved in a version 0.6 module, in addition to
the 0.5 keywords of 2.4, except directly after `.` (so `c.offer()` and
`c.choose(L)` remain method calls):

```
par       branch    nondet    budget    deadline  priority
replicated interrupt session  lvar      group     lanes
barrier   scope     atomic    protocol  offer     choose
node      supervisor strategy child     restart
```

In a 0.5 module they are names (`fn f(node: i64)` is valid 0.5). The
reference checker keeps them in `parse::DRAFT06_KEYWORDS`, separate from
`parse::KEYWORDS`, and reserves them only when the module is version 0.6 or 0.7.

### C.2 Grammar

These productions extend section 3 for version 0.6 modules. They are the
productions of the tree-sitter port (`spec/tree-sitter/grammar.js`) and of
the Rust parser. A rule below with the name of a section-3 rule replaces
it for a 0.6 module; the others are new. The machine check of 3.12 also
builds the LALR(1) tables of section 3 with these rules applied
(`lalr::book_grammar_06`), lexing with the keywords of C.1 reserved, and
fails on any conflict; it runs that machine over the 0.6 corpora
(`spec/conc06`, `spec/conc06b`). So T10 (one parse per program) holds for
version 0.6 modules too.

```
Item         = UseDecl | IntentDecl | ExampleDecl | CheckDecl
             | [ "pub" ] ( ConstDecl | TypeDecl | StructDecl | KindDecl
                         | TraitDecl | FnDecl | ActorDecl | ProtocolDecl )
             | StateDecl | EventDecl | ErrorDecl | StepDecl | ImplDecl
             | ExternDecl | NodeDecl | SupervisorDecl .
StateDecl    = [ "replicated" ] "state" NAME NEWLINE INDENT Field { Field } DEDENT .
EventDecl    = [ "interrupt" ] "event" NAME [ Payload ] [ "priority" Expr ] NEWLINE .
StepDecl     = "step" NAME "on" Pattern [ "if" Expr ] [ "deadline" Expr ] StepBody .
ProtocolDecl = "protocol" NAME NEWLINE INDENT ProtoStep { ProtoStep } DEDENT .
ProtoStep    = "send" Type NEWLINE | "recv" Type NEWLINE
             | "offer" NEWLINE INDENT ProtoArm { ProtoArm } DEDENT
             | "choose" NEWLINE INDENT ProtoArm { ProtoArm } DEDENT
             | "loop" NEWLINE INDENT ProtoStep { ProtoStep } DEDENT
             | "break" NEWLINE .
ProtoArm     = NAME NEWLINE [ INDENT ProtoStep { ProtoStep } DEDENT ] .
NodeDecl     = "node" NAME "{" NAME { "," NAME } "}" NEWLINE .
SupervisorDecl = "supervisor" NAME "strategy" NAME NEWLINE INDENT SupChild { SupChild } DEDENT .
SupChild     = "child" NAME ":" TypePath [ "restart" Bound ] NEWLINE .
BaseType     = TypePath | "(" ")" | "(" Type "," [ Type { "," Type } [ "," ] ] ")"
             | "(" Type ")" | "[" Type "]" | "[" Type ";" Expr "]"
             | "chan" BaseType | "iter" BaseType
             | "fn" "(" [ TypeList ] ")" "->" BaseType
             | "session" TypePath | "lvar" BaseType .
Stmt         = SimpleStmt | LetStmt | VarStmt | IfStmt | WhileStmt | LoopStmt
             | ForStmt | MatchStmt | ReturnStmt | BreakStmt | ContinueStmt
             | EmitStmt | FailStmt | SpawnStmt | SelectStmt | ArenaStmt
             | DeviceStmt | YieldStmt | ComptimeStmt | CheckDecl | PassStmt
             | ParStmt | BudgetStmt | DeadlineStmt | GroupStmt | BarrierStmt
             | ScopeStmt | AtomicStmt .
ForStmt      = "for" Pattern "in" Expr [ "par" ] Block .
SpawnStmt    = "spawn" Expr [ "priority" Expr ] [ "->" NAME ] NEWLINE .
SelectStmt   = "select" [ "nondet" ] NEWLINE INDENT SelectArm { SelectArm } DEDENT .
ParStmt      = "par" NEWLINE INDENT Branch { Branch } DEDENT .
Branch       = "branch" NAME Block .
BudgetStmt   = "with" "budget" Expr Block .
DeadlineStmt = "with" "deadline" Expr Block [ "else" Block ] .
GroupStmt    = "with" "group" NAME "lanes" Expr Block .
BarrierStmt  = "barrier" NEWLINE .
ScopeStmt    = "scope" Block .
AtomicStmt   = "atomic" Block .
```

### C.3 Footprints

The footprint of a statement is the set of place paths (5.6) it reads and
the set it writes. Two branches are disjoint when no path written by one
is a prefix of, or has as prefix, a path read or written by the other.

### C.4 `par`

`par` runs its branches; each branch is a block. The result equals running
the branches in source order (C.4.2). Rules:

- W-PAR-1: the footprints of two branches overlap (write/write or
  write/read), including through a `&mut self` method call.
- W-PAR-2: a branch has an effect other than its own writes (`emit`,
  channel operations, spawn, foreign calls), or returns, breaks or
  continues out of the `par`.
- W-NAME-2 also refuses two branches with one name.

### C.5 Budgets, deadlines and nondeterminism

- `with budget n` runs the block with a cost budget n: every Form
  operation charges its cost (C.6.4) and the operation that takes the
  total past n is `trap("budget")`, at the same operation on every
  target. The cost table is the reference evaluator's (`interp` and
  `conc.rs`, `charge`), listed in C.6.4.
  W-TY-1 if n is not an integer.
- `with deadline n` runs the block with an expiry n logical ticks after
  entry; if the block has not completed by then, the `else` block runs
  instead (T22). W-DL-1: outside `fn main` or a task
  function.
- `select nondet` may take any ready arm; the choice is an input and is
  logged, so replay reproduces it (T21). W-ND-1: outside `fn main` or a
  task function. `select` without `nondet` keeps the rotation rule of 6.8.
- `spawn f(...) priority p` and `event E priority p`: admission prefers
  the highest pending priority (T25). W-TY-1 if p is not an integer.

### C.6 Cost semantics

C.6.1 A cost is a non-negative integer number of units. Cost is a
deterministic accounting of Form operations, the same on every target;
it is not a measurement of time or energy (est_j is separate).

C.6.2 Charging point. An operation charges its cost after its operands
are evaluated and before it acts. If the charge would take any
enclosing budget past its limit, the operation is `trap("budget")` and
does not act; otherwise every enclosing budget is charged. So nested
budgets see the same operation trap, and a budget n admits exactly the
operations whose running total stays at most n.

C.6.3 `par` and `for ... par` charge `c_fork` per branch (or chunk),
then each branch's own costs. The branches' costs are summed in source
order, and exhaustion is placed where the sequential run would hit it
(C.4.2), so the trap does not depend on scheduling.

C.6.4 The cost table (normative). Operations not listed (literals of
scalars, variable reads, assignment, control flow, `let`) cost 0. The
table is checked against the reference evaluator, and the emitted Rust,
Python and JavaScript runtimes charge the same costs
(`spec/checker/tests/stdlib_costs.rs`).

| Operation | Cost |
|---|---|
| `c_call`: call of a user function or a closure | 4 |
| standard library method (5.15) on receiver size n, first-argument size k | 0.7: 4 + work(m, n, k), below; 0.5 and 0.6: 4 |
| `c_chan`: channel send or receive | 8 |
| `c_spawn`: `spawn` | 32 |
| `c_fork`: one `par` branch or `for ... par` chunk | 16 |
| list literal `[a, b, ...]`, map literal, comprehension with n results | 4 + n |
| `[x; n]` | 4 + n |
| index `a[i]`, slice `a[i:j]`, member `x.f`, tuple member `t.0` | 1 |
| unary `-`, `~`, `not` | 1 |
| `and`, `or` | 1 |
| binary operator on integer, bool or text operands, comparisons | 1 |
| float (f64, f32) `+`, `-`, `*` | 2 |
| float (f64, f32) `/` | 8 |

**Method work (normative for version 0.7).** In a version 0.6 module a
standard library method costs `c_call` whatever its sizes, as draft 0.6
defined it; a budget there bounds the number of operations. In a 0.7
module it costs `c_call` plus work(m, n, k). A budget is a proxy for energy, and energy
scales with the operations performed, so a method's cost must track its
work. Sizes: text is its UTF-8 byte length, a list its length, a map its
number of entries, any other value 0. n is the receiver's size and k the
first argument's (0 when there is none). work(m, n, k) is fixed by the
method's complexity class in the reference evaluator:

| Class | Methods | work |
|---|---|---|
| linear, text receiver | `strip`, `upper`, `lower`, `reverse`, `tostr`, `chars`, `words`, `lines`, `split`, `isdigit`, `isalpha`, `isspace`, `contains`, `startswith`, `endswith`, `toint`, `join`, `sha256` | n + k |
| linear, list receiver | `reverse`, `contains`, `sum`, `map`, `filter`, `fold`, `utf8` | n + k |
| linear, map receiver (maps are ordered association lists) | `get`, `contains`, `insert`, `remove`, `keys`, `values` | n + k |
| n log n, list receiver | `sort` | n · ⌈log₂ n⌉, with ⌈log₂ n⌉ = 0 for n ≤ 1 |
| constant | every other method of 5.15 | 0 |

The calls a `map`, `filter` or `fold` makes to its function argument are
charged as calls in addition (`c_call` each, plus their bodies). The
arithmetic saturates at the largest i128, so it cannot overflow; no
reachable size comes near it, so the Python and JavaScript runtimes,
whose integers are unbounded, give the same numbers. work is monotone:
it never decreases when n or k grows. The reference evaluator
(`conc::method_work`) and the Rust, Python and JavaScript runtimes
compute the same function, and the budget boundary falls on the same
operation in all of them (`spec/checker/tests/method_costs.rs`, which
also checks monotonicity).

### C.7 Device kernels

Inside a device region (5.13), a version 0.6 module may use `par` and
`for ... par` with the same rules; the device rules of W-DEV-1 still
apply.

### C.8 `for ... par`

`for x in e par` runs the iterations as if in order. It is encoded in the
reference checker as the marker region `$forpar` (the AST has no new
variant). Rules:

- W-PAR-3: the body writes a place other than as a reduction (`+= *= &=
  |= ^=` on a variable never read in the body) or as slot `[i]` of a range
  loop over `i`.
- W-PAR-4: the body has an effect, or returns, breaks or continues.

A reduction is evaluated on a fixed tree, which equals the left fold for
an associative operation with an identity (T18).

### C.11 Draft 0.6b

- C.11.2 Workgroups. `with group g lanes n` runs n lanes in phases
  separated by `barrier`. W-GRP-1: a group not directly inside a device
  region. W-GRP-2: `barrier` other than as a statement at the top level of
  a group body. W-GRP-3: a lane reads another lane's slot in the same
  phase, or lanes share a scalar written in the phase.
- C.11.3 Sessions. `protocol P` declares a session type; a value of type
  `session P` is an endpoint. W-SES-1: an operation not allowed by the
  protocol state, an unknown label, `offer()` not matched over its labels,
  a match that does not handle every label, or branches leaving the
  endpoint in different states. W-SES-2: an endpoint not at the end of
  its protocol when it goes out of scope or is closed. W-SES-3: a label
  that appears twice, a protocol loop with no message before it repeats,
  or `break` outside a protocol loop. W-TY-1: a payload of the wrong type.
- C.11.5 Merge. `impl Merge for T` declares a join. The admission gate
  (`conc::crdt_gate`) checks the merge laws (commutative, associative,
  idempotent) by evaluation on generated values; a failure is W-CRDT-1.
- C.11.8 LVars. `lvar T` requires `T: Merge` (W-CRDT-1). W-LVAR-1:
  `freeze()` before the `par` or loop that puts to the LVar has joined.
  W-LVAR-2: `wait` with other than one threshold.
- C.11.9 Structured spawn. `spawn f(...) -> h` gives a join handle. In a
  version 0.6 module a task function may spawn only inside `scope`; `fn
  main` is an implicit scope (W-SCOPE-1). A join handle cannot leave its
  scope (W-SCOPE-1). A 0.5 module keeps the rule of 5.8: any task
  function may spawn.
- C.11.10 Transactions. `atomic` applies all its actor sends or none
  (T30). W-ATOM-1: `spawn`, a receive, `select` or another wait inside
  `atomic`. W-ATOM-2: actors of one `atomic` on different nodes.
- C.11.11 Placement and supervision. `node N {A, B}` places actors on a
  node. W-NODE-1: a placed name that is not an actor, or an actor placed
  twice. `supervisor S strategy k` with k one of `one_for_one`,
  `one_for_all`, `rest_for_one` (W-SUP-1); each `child` must be an actor
  (W-SUP-2), and W-NAME-2 refuses a duplicate child.
- C.11.12 Real time. `interrupt event E priority p` with p an integer
  literal in 0..=255 (W-INTR-1). `step ... deadline n` with n a positive
  integer literal in ticks, and `const TICK_NS: u64 = n` when declared
  (W-RT-1). TICK_NS only relates logical ticks to a nominal duration for
  reporting; no construct reads a wall clock (6.3 N11).
- C.11.4 Replicated state. `replicated state R` is a region whose log is
  shared by replicas; a replica that applied a prefix of the log holds the
  fold of that prefix (T31).

### C.12 Rules that are not draft 0.6

The reference checker also raises three rules that belong to the 0.5
language and are now listed in section 7: W-MOD-3 (an imported name that
clashes with a declaration or with another import; names are not
renamed), W-VIS-1 (use of a name private to its module, the form W-MOD-1
takes in `link.rs`) and W-TRAIT-3 (a type used with `==`, `<` or a
generic bound that does not implement the required trait).

### C.13 Theorems

Status is stated exactly. "Proved on a semantics" means a Lean proof over
a small-step semantics of a core of RCE. "Model" means a Lean proof about
an abstract model that is not connected to the RCE semantics or to the
reference evaluator. "Vectors" means the reference evaluator's results on
generated programs are checked against the model by the Lean kernel; that
is a correspondence on samples, not a proof about the evaluator.

| | Statement | Status | Lean |
|---|---|---|---|
| T16 | under W-PAR-1 every interleaving of `par` gives the source-order result | proved on a semantics (core: integer variables, assignment, sequence, `if`, binary `par`; no loops, calls, data or traps); refinement to a model of the evaluator with i64 traps; vectors | `par_det`, `refine`, `ParVectors` |
| T17 | steps of two well-formed branches commute | proved on a semantics | `diamond` |
| T18 | a fixed reduction tree equals the left fold | model | `tree_eq_foldl` |
| T19a | `par` neither deadlocks nor diverges | proved on a semantics | `progress`, `steps_bound` |
| T19b | acyclic waits-for leaves a task that waits on none | model | `t19b` |
| T20 | every complete run of `par` has the same length | proved on a semantics | `steps_count`, `work_par` |
| T21 | logged nondeterministic choices replay | model | `t21` |
| T22 | a deadline block has exactly one outcome, time bounded | model | `t22` |
| T23 | a workgroup gives the phase-then-lane result | proved on a semantics | `group_det` |
| T24 | happens-before iff vector clocks are ordered, for the tick rule | proved on a model of runs (both directions) | `vc_iff`, `TickClock.hb_iff` |
| T25 | priority admission takes a maximal pending priority | model | `t25` |
| T26 | a scope ends after every child | model | `t26` |
| T27 | an actor answers as its sequential specification | model; vectors | `t27`, `ModelVectors` |
| T28 | session duality, subject reduction and progress, with recursion | proved on a session calculus; encoding of surface protocols mechanized | `t28r_preserve`, `t28r_progress`, `enc_comp` |
| T29 | LVar puts commute; one-threshold reads are order independent | model; vectors | `lvar_interleave`, `t29_perm` |
| T30 | an atomic batch applies all or none | model; vectors | `t30` |
| T31 | replicas agree on equal prefixes | model (the statement is close to trivial) | `t31` |

None of T16 to T31 is a proof about the Rust reference evaluator. The
checker rules W-PAR-1 to 4 and W-GRP-1 to 3 are not linked to the Lean
footprints beyond the core of `ParSem.lean` (W-PAR-1 on variables).

### C.14 Emitters and running

`rce run` runs version 0.6 programs on the reference evaluator
(`conc.rs`). The program emitters (`emit/conc06.rs`) run `par` branches in
source order (C.4.2), which the meaning allows; a `par` whose branches
wait on LVars is run by restart, and budgets charge the same costs as the
evaluator, so `trap("budget")` falls on the same operation on every
target. Workgroups run lane after lane per phase. The 0.6b actor, session
and node constructs follow the limits of 10.5.

### C.15 Version 0.7

`# rce-version: 0.7` declares version 0.7. Its syntax, keywords, static
rules and evaluation are those of version 0.6, with one difference: a
standard library method costs `c_call + work(m, n, k)` (C.6.4) instead
of `c_call`. The rule that decides what needs a new version: a module's
meaning (its outputs, states and outcomes, including where a budget
traps) must be determined by its declared version, so a change that
gives a valid program of an existing version a different meaning that
the old text defined is a new version; a change that makes the
implementations meet text that already defined the meaning, or that
only touches artifacts outside a program's meaning, is an erratum and
applies to every version.

**Which version governs a cost (normative).** A program may join modules
of different versions (2.7). The cost of a standard library method call
is fixed by the declared version of the module that contains the call
expression, not by the version of the module that declared the function
being run, of its caller, or of the program's first module. A call
written in a 0.7 module, including one inside a closure written there,
costs `c_call + work(m, n, k)` wherever it is executed from, and a call
written in a 0.5 or 0.6 module costs `c_call` even when a 0.7 module
calls the function around it. The rule is lexical, so linking modules
changes no cost. Every other cost of C.6.4 is the same in 0.6 and 0.7.
Example: a 0.6 function that holds `with budget B` and calls an imported
0.7 function whose body sorts an 8-element list is charged 24 more than
the same call to an identical function declared in a 0.6 module.

| Change since draft 0.6 | Versions | Why |
|---|---|---|
| size-scaled method costs (C.6.4) | 0.7 only | draft 0.6 defined each method as `c_call`; where a 0.6 budget traps would change |
| saturating budget arithmetic (C.6) | all (erratum) | changes nothing reachable: no program reaches a sum near the largest i128 |
| closed standard library, no ambient I/O (5.15) | all (erratum) | states what the checker already enforced (W-NAME-1 on an unknown method); no valid program changes |
| `&mut` write-back on `fail` (6.2, T33) | all (erratum) | the reference evaluator already updated in place; the emitters now agree with it (and T34 flattens nested calls without changing order) |
| u64 values at or above 2^63 exact in order, min/max and f64 ops (6.3 N2) | all (erratum) | N2 has defined full-range u64 since draft 0.5; the runtimes now meet it |
| literals take the success type of a fallible return (5.1) | all (erratum) | a checker bug against `τ <: τ ! ε`; it admits programs the text already admitted |
| canonical D entry and hash-chained Form log (6.1) | all | the log format, not a program's meaning |
| `sha256` text method (5.15) | all (extension) | admits only programs that W-NAME-1 refused before, so no valid program of any version changes meaning |
| mutual recursion with a decreasing summed measure (6.7, W-REC-1) | all (extension) | admits only programs whose call graph has a cycle of two or more functions, which W-REC-1 refused before; the rule for a function that calls only itself is unchanged, so no valid program of any version changes meaning |
| maps iterate in canonical key order (6.7, 7.1) | all (erratum) | 6.7 has always said a map iterates in canonical order; the reference evaluator and the four emitter runtimes kept insertion order. They now keep every map sorted by the canonical order of its keys, so `for`, `keys()`, `values()` and display follow 6.7. A program whose output depended on insertion order was relying on behavior the text never gave it; tests/map_order.rs checks all targets |
| self-recursion needs one fixed shrinking position, or layers (6.7, W-REC-1) | all (erratum, and extension for layers) | 6.7 has said since draft 0.4 that the value at one position i shrinks along any chain of calls, and T2 that every run ends; the checker took a different position per call, which admitted functions that never return. Refusing them makes the checker meet the text. Every function the text admitted is admitted by one layer, and layers admit only programs refused before, so no program the text admitted changes meaning |
| a parameter typed by an alias of a channel type makes a task function (5.1, 5.2) | all (erratum) | the type grammar of 5.1 has no nominal alias, so `type A = chan τ` already named the channel type; the checker tested the written type, not the type it names, and refused such functions with W-SPAWN-1 or W-EFF-1. Only programs refused before are admitted |
| declared recursion bounds `fn … bound n` (6.7, W-REC-1) | all (extension) | `bound` after a signature was W-SYN-1, and a cycle it admits was W-REC-1, so only programs refused before are admitted; no valid program of any version changes meaning |
| est_j receipts | all | measurement records outside the language (C.5: est_j is separate from budgets) |
| `i64.MIN % -1` traps; `toint` traps outside i64 (6.3 N2) | all (erratum) | N2 already defined both; the reference and runtimes returned 0 and an out-of-range i64 |
| canonical float text in `tostr` and display (6.3 N3) | all (erratum) | N3 already defined it (`1e+300`, `1.5e-7`, `-0` is `0`); the implementations printed Rust's long decimal form |
| `x op= y` charges the cost of `op` (C.6.4) | all (erratum) | C.6.4 already charges every binary operator; compound assignment was charged 0, so a budget could be exceeded |
| 6.3 N2, N3, N8 and 6.2 state the trap texts, `usize` indices and trapping slices the implementations have always had; float `%` removed from N3 and C.6.4 | all (erratum) | the text contradicted the checker and every runtime; no accepted program changes, since the text described behaviour no program could reach (negative usize, float `%`) or contradicted what every implementation did |
| f64 `min`/`max` (6.3 N5, 5.15) evaluate | all (erratum) | the checker accepted them and N5 defined them, but the evaluators trapped; a program that trapped now computes what the text defined |
| f64 `pow` refused (W-NAME-1) | all (erratum) | not in the closed library of 5.15, which already made it W-NAME-1; the checker accepted it and every evaluator trapped on it, so no accepted program computed a value with it |
| refined types `type R = τ where p` (5.1, W-REF-1) | all (extension) | `where` after a type was W-SYN-1 and stays an ordinary identifier, so only programs refused before are admitted; no valid program of any version changes meaning |
| a foreign function is not a value (5.11, W-FFI-1) | all (erratum) | 5.11 already forbade a pure function calling foreign code; the value route was a checker hole |
| host contract (Appendix E: one admitter, durable append per OS, fatal barrier, verify on open, inputs as events, capabilities, fabric entries, automatic placement) | all | constrains hosts and the log, not programs: a host entry has a `host.` type that no program can declare and δ is the identity on it, so the fold of every log of every valid program is unchanged |
| host bindings (E.8: a host result delivered as an event the program declares) | all | the bound event is an ordinary declared event, admitted and typed as any other; a program that declares no binding target sees no new input |

So a 0.6 program keeps its meaning, a 0.5 program is unaffected (it has
no budgets), and a 0.7 program differs from the same program declared
0.6 only in where a `with budget` block traps. Checked by
`spec/checker/tests/method_costs.rs` (0.6 charges `c_call` alone, 0.7
charges the table, in the reference and every runtime) and
`spec/checker/tests/draft06_gate.rs`, and the per-call-site rule by
`spec/checker/tests/call_site_cost.rs` (0.5, 0.6 and 0.7 modules joined
by `use`, in the reference evaluator and the Rust, Python and JavaScript
emitters on the linked program); the tree-sitter port and the
LALR(1) check treat 0.7 as 0.6, since the grammar is the same.

### C.16 Version 0.8

`# rce-version: 0.8` declares version 0.8: the syntax, keywords, rules,
evaluation and costs of 0.7, plus 5.16. It reserves no keyword, so the
grammar, the LALR(1) check and the tree-sitter port treat it as 0.7. It
is a version, not an extension, by the C.15 rule: the eight names of
5.16 are ordinary names in 0.7, and a 0.7 program that declares one
(`struct Text`, `struct Html`) would change meaning if 5.16 applied to
it. In a 0.8 module `C.lit(s)` costs `c_call`, `C.encode(t)` costs
`c_call` plus the size of t (the linear class of C.6.4) and `.raw()`
costs `c_call` (the text is not copied), in the reference evaluator and
every runtime; the indices are erased, so nothing else in C.6 changes. A 0.5, 0.6 or 0.7 module never
sees 5.16 (`spec/checker/tests/indexed.rs`, gate test), and modules of
different versions may `use` each other (2.7).

**Read receipt.** How many words does a version 0.6 module reserve in addition to the 0.5 keywords? (a number)

<!-- body-sha256: d6a1a6894f575dd98ceb07c7ce0115eb8ffda51623cc70a9e520e36522db35f3 -->
