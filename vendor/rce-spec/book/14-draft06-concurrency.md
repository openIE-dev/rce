## Appendix C. Draft 0.6 and 0.6b concurrency (normative for version 0.6 modules)

This appendix defines the draft 0.6 language: structured parallelism
(`par`, `for ... par`), budgets, deadlines, nondeterministic select and
priorities (0.6), and the 0.6b extensions: workgroups, session types,
structured spawn, cross-actor transactions, supervisors, nodes,
interrupts, LVars and replicated state. It applies only to a module that
declares `# rce-version: 0.6` (2.7). A 0.5 module never sees any of it:
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
`parse::KEYWORDS`, and reserves them only when the module is version 0.6.

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
| standard library method (5.15) on receiver size n, first-argument size k | 4 + work(m, n, k), below |
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
| float (f64, f32) `/`, `%` | 8 |

**Method work (normative).** A budget is a proxy for energy, and energy
scales with the operations performed, so a method's cost must track its
work. Sizes: text is its UTF-8 byte length, a list its length, a map its
number of entries, any other value 0. n is the receiver's size and k the
first argument's (0 when there is none). work(m, n, k) is fixed by the
method's complexity class in the reference evaluator:

| Class | Methods | work |
|---|---|---|
| linear, text receiver | `strip`, `upper`, `lower`, `reverse`, `tostr`, `chars`, `words`, `lines`, `split`, `isdigit`, `isalpha`, `isspace`, `contains`, `startswith`, `endswith`, `toint`, `join` | n + k |
| linear, list receiver | `reverse`, `contains`, `sum`, `map`, `filter`, `fold` | n + k |
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

**Read receipt.** How many words does a version 0.6 module reserve in addition to the 0.5 keywords? (a number)

<!-- body-sha256: 2e7604ee602db61fd17615ca0bde1454756accc45249b689328b5d5c15d1281d -->
