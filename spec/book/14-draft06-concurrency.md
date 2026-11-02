## Appendix C. Concurrency (draft 0.6 and 0.6b; normative)

This appendix defines the concurrency part of the language: structured parallelism
(`par`, `for ... par`), budgets, deadlines, nondeterministic select and
priorities (drafts 0.6), and the 0.6b extensions: workgroups, session types,
structured spawn, cross-actor transactions, supervisors, nodes,
interrupts, LVars and replicated state. It applies to every module; no
declaration selects it (2.7). The
text is written from what the reference checker (`spec/checker`,
`check.rs`, `check06b.rs`, `conc.rs`, `link.rs`) and the Lean files
(`spec/lean/RCE/Par*.lean`, `Models06b.lean`, `Session*.lean`,
`TickClock.lean`, `VClock.lean`) actually enforce and prove; where the
two differ from an intended design, the checker is what is described.
Example: `par` with two branches that write different
variables gives the same result as running the branches in source order.

### C.1 Keywords

The 23 words below are on the reserved list of 2.4, in addition to
the keywords of 2.4, except directly after `.` (so `c.offer()` and
`c.choose(L)` remain method calls):

```
par       branch    nondet    budget    deadline  priority
replicated interrupt session  lvar      group     lanes
barrier   scope     atomic    protocol  offer     choose
node      supervisor strategy child     restart
```

They are not names in any module: `fn f(node: i64)` requires change. The
reference checker keeps them in `parse::DRAFT06_KEYWORDS`, separate from
`parse::KEYWORDS` only as a source layout; the reserved list is their union
with the words of Appendix F, one list for every module.

### C.2 Grammar

These productions extend section 3. They are the
productions of the tree-sitter port (`tree-sitter/grammar.js`) and of
the Rust parser. A rule below with the name of a section-3 rule replaces
it; the others are new. The grammar is the productions of section 3 with
these applied, and the same for every module. The machine check of 3.12 builds the
LALR(1) tables of section 3 with these rules applied
(`lalr::book_grammar_06`), lexing with the keywords of C.1 reserved, and
fails on any conflict; it runs that machine over the concurrency corpora
(`spec/conc06`, `spec/conc06b`). So T10 (one parse per program) holds for
the whole language.

```
Item         = UseDecl | IntentDecl | ExampleDecl | CheckDecl | LawDecl
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
the branches in source order (T16). Rules:

- W-PAR-1: the footprints of two branches overlap (write/write or
  write/read), including through a `&mut self` method call.
- W-PAR-2: a branch has an effect other than its own writes (`emit`,
  channel operations, spawn, foreign calls), or returns, breaks or
  continues out of the `par`.
- Two branches with one name also require change (W-NAME-2).

Derivation of the rules (confluence). Determinism of a parallel
composition is a confluence property: every way of interleaving the
branches' steps must reach one result, which is then the result of any
fixed order, source order among them. By the Church-Rosser method it is
enough to show the diamond: one step of branch a and one step of branch
b, taken from the same configuration in either order, reach the same
configuration. A step changes only its write set and depends only on
its free variables, so the two steps commute exactly when neither writes
what the other reads or writes. That is W-PAR-1, derived, not chosen: it
is the weakest footprint condition under which the diamond holds for
every store. Termination of each branch (T19) and the diamond give
confluence (Newman's lemma), so every complete interleaving gives the
source-order store (T16). An effect that leaves the store (`emit`,
channel operations, spawn, a foreign call) is observed in the order it
happens, so it would make the observation depend on the interleaving
even when the store does not; W-PAR-2 removes these. Traps follow the
same law: a trap is observed at the first branch in source order that
traps, because that is where the sequential run traps; the reference
and the emitters place both arithmetic traps and resource traps (fuel,
budget) at that branch (book/12-tooling.md 10.5). Mechanized in
`spec/lean`: the diamond on a core with `par` (`RCE.ParSem.diamond`),
determinism of complete runs (`RCE.ParSem.par_det`), refinement to a
model of the reference evaluator including overflow traps
(`RCE.ParSem.refine`), and the fuel account, whose sum-checked trap
point equals the sequential one and whose evaluated work is at most k
times the fuel left at the fork (`RCE.ParFuel.sumcheck_eq_seq`,
`RCE.ParFuel.overrun_bound`). LVars weaken the diamond to
quasi-determinism (C.11.8, T29q). Open: the core has no loops, calls or
data structures, so the link from the core to the full evaluator rests
on the generated vectors (`ParVectors.lean`) and the differential tests,
not on a proof.

### C.5 Budgets, deadlines and nondeterminism

- `with budget n` runs the block with a cost budget n: every Form
  operation charges its cost (C.6.4) and the operation that takes the
  total past n is `trap("budget")`, at the same operation on every
  target. The cost table is the reference evaluator's (`interp` and
  `conc.rs`, `charge`), listed in C.6.4.
  W-TY-1 if n is not an integer.
  `with budget n Block else [as e] Else` makes exhaustion a value.
  Derivation. Totality (T2) asks every step to end in a value; a trap
  that keeps the state is an outcome of the step, not a value the
  program can act on. To act on it the program needs a point where the
  trap is caught, and the state at that point must be one that some
  sequential execution reaches at a well-defined place, or Else observes
  a state no specification of Block describes. The only such states are
  the entry of Block and its completion; the trap rules out completion,
  so a caught trap restores the entry state. Block is therefore atomic:
  all of its writes or none (the same law as T30 for `atomic`).
  Normative rules:
  - A `trap("budget")` raised while Block runs, by this budget or any
    enclosing one, is caught by the innermost enclosing budget that has
    an `else`. Every write Block made (locals, struct fields, state
    fields, `self`) is undone, then Else runs.
  - `as e` binds `e: Exhausted`, the builtin
    `struct Exhausted { limit: u64, used: u64 }`, where `limit` is n and
    `used` is the units this frame had spent before the operation that
    would have passed a limit.
  - Charges stay. Cost counts operations performed (C.6.1), so the
    units spent in Block remain on every enclosing budget. When the trap
    came from an outer budget, Else charges that budget, its next charge
    traps again, and the trap reaches the next budget out with an
    `else`, or keeps the state.
  - W-BUD-1: Block of a budget with an `else` holds an effect that a
    trap cannot undo (`emit`, channel operations, spawn, select, yield,
    lvar `put` or `freeze`), or returns, breaks or continues out of it.
  - A budget without `else` behaves as before: the trap keeps the
    state.
  - The charge rule: a charge is all or nothing, leaves no frame past
    its limit, and is additive. Every runtime's `charge` implements the
    same rule; in this repository the link between those copies is the
    differential tests, not a proof.
  The trap falls on the same operation on every target, so whether Else
  runs, and the value of `e`, are functions of the program and its
  inputs, and replay re-derives them. The emitters save every local the
  block writes at entry (struct fields and state fields are locals of
  the emitted step) and restore them on the trap. Checked by
  `spec/checker/tests/conc06_run.rs` (`budget_else_catches`,
  `budget_else_with_undoable_effects_requires_change`), `emit_run.rs`
  (`budget_else_*`, reference against the Rust, Python and JavaScript
  runtimes) and `emit06b.rs` (`budget_else_is_atomic_on_every_target`,
  the limit swept across the block's cost on all four targets).
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
(C.4, T16), so the trap does not depend on scheduling.

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
| float (f64, f32) `/` | 8 |

**Method work (normative).** A standard library method costs
`c_call` plus work(m, n, k); draft 0.6 charged `c_call` alone, which is not
the language of this text. A budget counts work, so a method's cost must
track its work; the count is not a measurement of energy (C.6.1). Sizes: text is its UTF-8 byte length, a list its length, a set its number of
elements, a map its number of entries, any other value 0. n is the receiver's size and k the
first argument's (0 when there is none). work(m, n, k) is fixed by the
method's complexity class in the reference evaluator:

| Class | Methods | work |
|---|---|---|
| linear, text receiver | `strip`, `upper`, `lower`, `reverse`, `tostr`, `chars`, `words`, `lines`, `split`, `isdigit`, `isalpha`, `isspace`, `contains`, `startswith`, `endswith`, `toint`, `join`, `sha256` | n + k |
| linear, list receiver | `reverse`, `contains`, `sum`, `map`, `filter`, `fold`, `utf8` | n + k |
| linear, map receiver (maps are ordered association lists) | `get`, `contains`, `insert`, `remove`, `keys`, `values`, `merge` | n + k |
| linear, set receiver (a set is held as its elements in canonical order) | `union`, `inter`, `insert`, `contains`, `tolist` | n + k |
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

Inside a device region (5.13), a module may use `par` and
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
an associative operation with an identity (T18). The tree is normative,
since a float operation is not associative and the bits depend on it:
for m >= 2 contributions, the left subtree takes the first h, where h is
the largest power of two below m, and the right subtree the rest; one
contribution is itself; none is the identity. Derivation: placement must
give the same bits on every fabric (HC-12), and the reductions that
vector units and GPUs execute are complete binary trees over
power-of-two blocks. With this split every left subtree is complete, so
the tree decomposes into such blocks and a fabric can evaluate it
natively. `pscan` uses the same split for its up-sweep and down-sweep.
One definition, `host::tree_split`, serves the reference evaluator and
the placement executor (`place_par`, `place_sum`), and every runtime
computes the same h. Checked by `spec/checker/tests/host_place.rs`
(`forpar_reference_and_placement_share_the_tree`: a non-associative
float sum gives the same bits from `run_main` and from the placement
executor for every length up to 40) and by the emitter parity tests.
Before 2026-10-15 the reference and runtimes split at ceil(m/2) while
the placement executor split at the power of two; the two disagreed in
the last bits of float reductions from five contributions on.
The Lean model defines the same split, `RCE.Par.split m = 2 ^ log2 (m -
1)` (checked against `host::tree_split` on sample lengths), and proves
`tree_eq_foldl` for the tree built with it, so T18 is proved for the tree
the code runs. `RCE.ParFuel` reduces branch uses in that tree:
`tree_use_eq_sum` (the tree total is the source-order sum) and
`tree_work_eq` (the threaded work reduced in the tree is the work of
`overrun_bound`, at most k·F).

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
  The two endpoints of `chan[session P]` have the types P and dual(P),
  where dual swaps send with receive and choose with offer (duality).
  W-SES-4 (linearity): an endpoint is a capability that exactly one
  name holds. A protocol operation consumes the endpoint at one state
  and yields it at the next. A bare use of the name (an argument, a
  spawn argument, a payload sent on another endpoint, a rebinding) moves
  the endpoint, and `close` consumes it; after either the name holds
  nothing, and a further use requires change (no use after move, send or
  close). Paths that join must agree on whether an endpoint was
  consumed, so a branch that moves it while another keeps it in use
  requires change, as does a loop body that moves an endpoint the loop does not
  create. With W-SES-2 this makes endpoints linear (affine plus
  must-close): on every path to the end of its scope each endpoint is at
  the end of its protocol, closed, or moved to a holder that takes the
  obligation over. `let m = x` moves the endpoint to `m`, which owes the
  rest of x's protocol; a name bound inside a block goes out of scope at
  the end of that block, so its endpoint must be finished or passed on
  there. The checker's protocol automaton (`check06b`, `ProtoCursor`) is
  linked to T28 by vectors (C.13). Linearity is a typing judgment
  (`SessionLin.Ty`): the checker's W-SES-1, W-SES-2 and W-SES-4 for
  bodies of send, receive, close, move and branch are proved sound
  against it, and a typed body never uses a gone endpoint, never closes
  an unfinished one and ends with every endpoint gone or done, on every
  path (T28l).
  W-SES-5 (session graph a tree): a `spawn` passes at most one session
  endpoint, and a body that holds both ends of one channel acts on
  neither until it has passed one on. The session graph has processes
  as nodes and channels as edges. A spawn with one endpoint adds one
  edge to a new node, and sending an endpoint to the holder of the
  other end of the carrying channel moves an edge along a path, so the
  graph stays a tree. On a tree, if every channel's two ends are
  compatible, either every process has finished or some channel has
  both ends waiting on it and they can step (T28d). Two channels
  between the same two processes break this: each can wait on a
  different channel.
  W-SES-5 by priorities: a program that requires change under the tree rule is accepted when
  its session graph has a priority order. Each exchange on a channel is
  an action (channel, step), shared by both ends, and a spawn is a
  one-step action between spawner and spawned. Each process's actions in
  program order must strictly increase in priority, and each channel's
  ends must make the same number of exchanges; the checker finds such
  priorities exactly when the order the processes impose on actions is
  acyclic. Then every reachable state is finished or can step, on any
  session graph, cycles included (T28p): the waiting action of least
  priority has its partner waiting on that same action. A cycle of
  waits, as in two channels between two processes used in opposite
  orders, is a cycle in the order and requires change. The checker reads
  priorities from programs whose endpoints are created, sent on,
  received on and passed to spawns at the top level of bodies, over
  protocols of messages; outside that fragment the tree rule decides.
  Multiparty global types (T28g): a global type says which role sends
  to which and in what order, with binary choices; its projection on
  each role uses plain merge (roles not told of a choice must have the
  same local type in both branches). Every configuration reachable from
  the projections of a global type is finished or steps, and a step
  lands on the projections of a global type again. A global type of
  messages only gives priorities by event order, so its projections pass
  the priority rule and compose with the other channels of a program.
  The checker's projection equals the Lean projection on 200 generated
  global types (C.13).
  Full merge and asynchrony (T28a): a role not told of a choice may
  receive differently in the two branches; its projection merges them
  into one branching that accepts the labels of both (sends must still
  agree). Sends never block: each ordered pair of roles has one FIFO
  queue, so two messages from one sender to one receiver arrive in
  sending order, and the global type records which labels are in
  flight. Each role's local type need only accept at least what the
  projection of the current global type accepts (association). Every
  asynchronous step of an associated configuration is a step of the
  global type and stays associated; every receiver accepts the label at
  the head of its queue; and every configuration reachable from the
  projections with empty queues is final (all roles done, all queues
  empty) or steps. The checker's full-merge projection equals the Lean
  projection on 300 generated global types, and its exploration of the
  asynchronous runs from each fully projecting one finds no stuck
  configuration and none whose receiver does not accept the label heading
  its queue (C.13).
  Convergence notes. Honda, Vasconcelos and Kubo (ESOP 1998) keep
  channels in a linear realm and pass them by delegation; their type
  soundness theorem gives subject reduction and the absence of run-time
  communication errors, and does not claim deadlock freedom. Gay and
  Hole (Acta Informatica 2005) give each end to exactly one process by
  environment addition, end a derivation only at a completed
  environment (as W-SES-2 does), prove their typechecking algorithm
  sound for the declarative rules (the shape of `chkBody_sound`), and
  for polarity inference forbid using one end of a channel while the
  same thread holds the other (as W-SES-5 does); they show a typable
  process over two channels between two processes that deadlocks.
  Wadler (Propositions as Sessions, ICFP 2012) composes processes by a
  cut on exactly one channel, which gives deadlock freedom; a cut on two
  channels admits loops that may deadlock, and he notes that a primitive
  that creates a channel and returns both ends in a pair may suffer from
  deadlock. RCE's `chan[session P]` is such a primitive; W-SES-5 keeps
  its uses to the one-channel shape at spawns. Kobayashi (CONCUR 2006)
  annotates each channel use with an obligation and a capability level
  and proves that a well-typed process is deadlock free (and with time
  tags, lock free); Padovani (CSL-LICS 2014) gives the linear pi-calculus
  one priority per channel type, requires each process to use its
  channels in increasing priority, and proves deadlock freedom, encoding
  multiparty sessions so that a global type's events give the
  priorities (his App. C). T28p is this rule on RCE's networks, with one
  priority per exchange. Honda, Yoshida and Carbone (POPL 2008) introduce
  global types, projection and the progress of a single well-formed
  session; T28g is the synchronous, plain-merge form of their result,
  and T28a keeps their order-preserving queues (their Sec. 2) with the
  full merge and association of Hou, Yoshida and Kuhn (arXiv:2402.16741,
  Def. 3.1 and 3.7).
  The reading ledger is spec/proposals/session-reading-ledger.md.
  Open: an endpoint and its dual sent separately to the same third
  process close a loop that W-SES-5 does not see; T28d assumes the tree
  and the preservation of the tree under delegation is argued above,
  not mechanized (the priority rule closes this loop when the program is
  in the fragment it reads); T28p and T28g model rendezvous, and T28a
  models buffered delivery for global types without recursion or
  payloads; the checker reads priorities only for straight-line
  bodies over message protocols, and priorities for global types with
  choice or recursion are not derived; `SessionPrio.decideP` is proved
  sound, not complete (the checker's acyclicity test is the complete
  decision, linked by vectors); there is no surface syntax for global
  types; T28l covers message steps and local branching, not
  choose/offer, loops or recursion (those are covered by the automaton
  vectors of T28).
- C.11.5 Merge. `impl Merge for T` declares a join. Its laws
  (associative, commutative, idempotent) are an admission obligation
  (W-CRDT-1, `conc::crdt_gate`), discharged by a certificate and never by
  sampling. `bottom()` is the initial state; the states an LVar can reach
  are the up-set of it, which the join preserves, so the laws on that
  up-set are the laws of the type. For a kind of nullary variants the
  gate evaluates every triple, which decides them
  (`MergeLaws.laws_of_enum`). Otherwise the gate reads the body of
  `merge` as a term over proven combinators (`conc::merge_certs`): per
  field, `max` or `min` on an integer, `or` or `and` on a bool,
  `union` or `inter` on a `Set[τ]`, `merge` on a `Map[k, v]` (pointwise,
  certified when the `Merge` of `v` is: an integer, a bool, a set, a map
  of such, or a type with a certified impl), or `.merge` of a field whose
  type has a certified impl (componentwise product); on a record of one integer field, a selection that keeps
  `o` when `o.k > self.k` and `self` otherwise (max, or min when the
  smaller side is kept); and lexicographic on an integer key: `self` when
  `self.k > o.k`, `o` when `o.k > self.k`, and otherwise the key with the
  other fields merged by the rules above. Each term is a `Cert` of
  `MergeCert.lean`, and `Cert.semi` turns every certificate into a proof
  of the three laws, so they compose by construction. The lexicographic
  rule needs its key to be a chain (a total order); without that it is
  not a join (`MergeCert.lex_needs_chain`), which is why the key must be
  an integer. Any other impl requires change, also a lawful one written
  differently (for example bitwise `|`), with a message naming the
  combinators; a search on a finite sample runs first and, when it finds
  a counterexample, the report names it. The built-in
  instances (integers under `max`, `bool` under `or`, sets and lists
  under union, maps pointwise, tuples and records componentwise, `none` as
  bottom) have the laws by `MergeLaws.lean`; that sorted lists without
  duplicates represent sets and maps canonically is `Canon.set_canon` and
  `Canon.map_canon`, and that the canonical order they are sorted by is a
  strict total order is `ValOrder.lt_total`. A componentwise instance
  exists only when every component has one: a map needs a `Merge` on its
  value type, an option on its payload, and a struct without its own
  `impl Merge` on every field, recursively through nested structs and the
  struct's type arguments (a struct reached again through itself adds no
  condition). Otherwise `lvar_new`, `lvar T` and `Map.merge` require change
  (W-CRDT-1) with the path to the first field that has none. Sets and
  lists merge by union, which needs only equality, so their element type
  needs no `Merge`.
- C.11.6 Prefix scan. `xs.pscan(f, e)` and `xs.pscan_incl(f, e)` take a
  list of T, `f: (T, T) -> T` and `e: T` (W-TY-1 otherwise) and return a
  list of the same length. Element i of `pscan` combines xs[0..i] and
  element i of `pscan_incl` combines xs[0..=i]; `pscan` gives `e` at
  index 0, the empty prefix, and the empty list scans to the empty list.
  Evaluation is on a fixed tree (the up-sweep and down-sweep of
  Blelloch) of C.8: the range [lo, hi) splits at lo + p, with p the
  largest power of two below hi - lo (`host::tree_split`). The
  up-sweep stores the combination of each node's range, left operand
  before right. The down-sweep gives each node an offset: none at the
  root; the parent's offset to the left child; to the right child the
  left child's sum, combined after the parent's offset when there is
  one. Element i of `pscan` is the offset of leaf i, or `e` when it has
  none; element i of `pscan_incl` is xs[i] combined after that offset.
  The tree depends on n alone, so every run and every target applies f
  to the same operands in the same bracketing and order, and the result
  is bit-identical for an f that is itself deterministic on each target,
  floats included. Law: if f is associative, every bracketing of a
  prefix has the same value, so element i equals the left fold of its
  prefix. No identity law is needed: `e` is never combined, it only
  stands for the empty prefix. If f is not associative (float addition)
  the result is still fixed, but can differ from the left fold, as for a
  reduction on the fixed tree (C.8, T18). Work: fewer than 2n applications
  of f for `pscan` and fewer than 3n for `pscan_incl`; depth O(log n).
  The reference (`conc.rs`) and the runtimes of every emitter (rt.rs,
  rt.py, rt.js, rt.scala) use the same split.
- C.11.8 LVars. `lvar T` requires `T: Merge` (W-CRDT-1). An LVar holds a
  value, the optional top of `lvar_new(bottom, top)`, and a frozen bit.
  `put(d)` sets the value `v` to `v ⊔ d` with the program's `Merge`; it
  is `trap("lvar conflict")` when the result is the top, and
  `trap("lvar put after freeze")` when the LVar is frozen and the result
  differs from `v` (a put that changes nothing is allowed). `freeze()`
  sets the bit and returns the exact value. `wait(Q)` returns the `t` in
  `Q` with `t ⊔ v = v`, in the same `Merge` order as `put`, and blocks
  until there is one. In a task of `fn main`, outside the branches of a
  `par`, a `wait` on an LVar that has passed from one task to another (a
  `spawn` argument, a message, a task's result, a closure that captures
  it) and is at no threshold is a communication step (6.8): the task is
  blocked until a put of another task or actor brings the LVar to a
  threshold, and when no task can move any more the blocked waits trap
  with `trap("lvar wait cannot be satisfied")`. A wait on an LVar that
  has not passed between tasks traps at once, since no other put can
  come. Once an LVar has passed between tasks, the tasks of each tick
  move one after another in task order, so the puts, waits and freezes
  of a tick happen in one order (10.3). The threshold tests charge no fuel and no budget,
  so the cost of a `wait` does not depend on how often a schedule
  rechecks it. A `Q` with more than one element must be pairwise
  incompatible: distinct elements join to the top. Every `wait` checks
  this before it reads the value and is otherwise
  `trap("lvar thresholds are not pairwise incompatible")`; the check does
  not read the value, so its outcome does not depend on the schedule. A
  trap is a defined result, so δ stays total. W-LVAR-1: inside the
  branches of one `par`, or the body of a `for ... par`, a `freeze`
  while some branch may `put`, where a call to a function or method that
  puts to (or freezes) an LVar it did not create counts as a put (or a
  freeze). An LVar a branch creates with `lvar_new` and does not rebind is
  its own and does not count. The rule does not match the name of a put
  with the name of a freeze, so an
  alias, a parameter or a helper cannot hide a freeze. W-LVAR-2: an empty
  threshold list, or a list with two elements that are, or on some input
  can be, one below the other in the program's `Merge` (`conc::lvar_gate`).
  This is decided at admission when every element takes finitely many
  values: a constant, or `if c then a else b` over such, with every
  combination checked, since two thresholds where one is below the other
  are compatible for any top. A parameter whose declared type is finite
  (a kind of nullary variants, or `bool`) and which the body never
  rebinds ranges over every value of its type, so it is decided too;
  since every value of such a type is at or above its bottom, a list
  with such a parameter and another element requires change whenever the
  bottom is below that element. Where an element is not finite in this
  sense (a parameter of an unbounded type, a computed value),
  incompatibility depends on the input, and the check every `wait` makes
  against the top, with its trap, is the defined behaviour. Model: T29q,
  T29s, T29a; the rule table
  `conc::LVAR_RULES` ties each rule and trap to its Lean declaration
  (see T29s).
- C.11.8 convergence notes. The frozen bit, and the rule that only a put
  that would change a frozen LVar is an error, follow Kuper, Turon,
  Krishnaswami and Newton (POPL 2014, Sec. 3.3 and 4.2). Threshold sets
  and their `incomp` premise follow Kuper and Newton (FHPC 2013, Sec.
  3.3). T29q has the shape of the POPL 2014 Theorem 1 for this fragment.
  Their proof goes through an Independence lemma and quasi-confluence;
  the one here uses an invariant (the value joined with the puts still to
  come is the join of all puts), which suffices because there is no
  allocation: T29s takes a store of many LVars as the product lattice
  over locations and proves the invariant per location, and T29a adds
  `lvar_new` as a step that allocates a location named by a
  deterministic counter (thread i's k-th allocation is `(i, k)`), so
  locations do not depend on the schedule and the POPL 2014 permutation
  of locations is not needed. Their library marks code that freezes as
  `QuasiDet` in its type. In RCE, the freezes that could race with a put
  instead require change (W-LVAR-1), so an admitted program is deterministic,
  and the frozen bit turns a race the rule misses into an error instead
  of a different answer. FHPC 2013 suggests validating user lattices
  with QuickCheck; the sample search here is that kind of test, and is
  used only to find counterexamples.
- C.11.9 Structured spawn. `spawn f(...) -> h` gives a join handle. In a
  module a task function may spawn only inside `scope`; `fn
  main` is an implicit scope (W-SCOPE-1). A join handle cannot leave its
  scope (W-SCOPE-1).
- C.11.10 Transactions. `atomic` applies all its actor sends or none
  (T30). W-ATOM-1: `spawn`, a receive, `select` or another wait inside
  `atomic`. W-ATOM-2: actors of one `atomic` on different nodes.
- C.11.11 Placement and supervision. `node N {A, B}` places actors on a
  node. W-NODE-1: a placed name that is not an actor, or an actor placed
  twice. `supervisor S strategy k` with k one of `one_for_one`,
  `one_for_all`, `rest_for_one` (W-SUP-1); each `child` must be an actor
  (W-SUP-2), and a duplicate child requires change (W-NAME-2).
  Run-time meaning (normative). A supervisor declaration is one static
  supervisor; it supervises every actor whose type is one of its
  children, so the set an actor belongs to is fixed by the program, not
  by spawn order. When the step of a handler of such an actor is kept for
  its message (a trap or a `fail`), the run log records `Kept(reason)`, then
  `Restart(child, instance, reason, tick t, k)`, where `instance` is the
  actor's id (6.6) and k counts this child's restarts in the window
  below. The strategy then resets the region of the failed actor
  (`one_for_one`), of every actor of the supervisor's child types
  (`one_for_all`), or of those whose child is declared at or after the
  failed one (`rest_for_one`), to the actor's initializers. Intensity is
  measured in logical ticks, never wall time: with `const
  RESTART_WINDOW: u64 = W` only restarts at ticks t' with t - t' < W
  count, otherwise every restart of the run counts; more than the
  child's `restart bound` stops the supervisor with `Stopped(supervisor
  S)` and the run ends `Stopped`. Mailbox policy: a restarted actor keeps
  its mailbox, since Form law 3 drops no message; only the message that
  failed is consumed, and it is logged as kept. A handler that fails
  inside `atomic` (C.11.10) leaves every handler of the batch
  uncommitted (T30), and the failing actor is then supervised as above.
  Restart as input. Derivation. The state of a run is the fold of its
  log (T3). A restart changes actor state without a message, so it must
  be an entry of the log that the fold applies, or the state is not a
  function of the log. The decision is therefore an input of the run:
  at a crash the run records `RestartIn{tick, sup, child, id, reset,
  stop}` (`RunResult.restarts`), and a replay given these inputs
  (`RunCfg.restarts`) applies each logged decision at its crash without
  consulting the supervisor's policy. A log thus replays under the
  decisions it records even after the strategy, bound or
  `RESTART_WINDOW` in the source has changed (the resolver rule of
  Appendix D, applied to supervision). A replayed decision for another
  actor or supervisor than the one that crashed is `replay diverged`.
  Checked by `spec/checker/tests/conc06b_run.rs`
  (`supervisor_restarts_then_stops`,
  `restart_is_an_input_replay_consumes`: replay reproduces the log,
  logged decisions govern under a changed bound, an edited decision is
  applied as logged, a mismatched one diverges) and `emit_run.rs`
  `actor_kept_outcomes_match` (the Rust, Python and JavaScript runtimes take
  the same decisions as the reference, with and without a window) and
  `restart_inputs_replay_on_every_target` (each runtime prints its
  decisions with `restart-log`, replays a logged list given as
  `restart TICK|SUP|CHILD|ID|STOP|RESET,...` under a changed bound
  exactly as the reference does, and diverges on a mismatched decision).
  Stable ids and upgrades. Derivation. A logged decision names actors
  by id, and replay applies it to the actor with that id, so a log
  replays on an upgraded program exactly when the upgrade maps each
  logged id to the same instance. An id built from source text (a line,
  a count of unrelated creations) changes under edits that do not touch
  supervision, and T7 forbids identifiers in ids. What both an upgrade
  that keeps supervision and a renaming preserve is the position of an
  instance: its place in the supervisor tree and its order among its
  creating task's creations at that place. Ids are therefore those paths
  (6.6): `i.j`, creating task, ordinal for a supervised actor; `-`, task,
  ordinal for an unsupervised one; parent and ordinal for a task. Since
  ids are positional, a log is meaningful only against the tree it was
  recorded under, so the tree is the log's schema: version 1 of the
  restart log is a header `restartlog v1 tree=S/c,...` (every position
  by its declared supervisor and child names, in declaration order;
  `RunResult.tree`, `conc::sup_tree`) followed by the `restartin`
  lines. Migration rule (normative): a version 1 log replays on a
  program in which every recorded position still holds the same
  declared supervisor and child, that is the recorded supervisors are
  the program's first ones in order and each keeps its recorded children
  as its first ones in order; new children may follow a supervisor's
  recorded ones and new supervisors may follow the recorded ones, since
  neither moves a recorded position. Any other change (a position
  removed or renamed, a supervisor or child inserted before a recorded
  one, a reorder) is detected before the first step: the run stops
  with `trap: upgrade removed supervisor position S/c` or `trap: upgrade
  reordered supervisor position S/c` for the first recorded position
  that fails, and nothing is replayed (`conc::tree_migrates`,
  `RunCfg.restart_tree`). A log without a header is version 0: its
  decisions replay with each decision's position checked when it is
  taken, so a decision whose position is gone stops the run with the
  removed-position trap at that crash, and a decision for another actor is `replay
  diverged`; neither is ever applied to a different instance. A log
  whose tree does not migrate is converted by recording it again under
  the new program; no rule rewrites positional ids across a moved
  position, because the instance a moved position denotes is not
  determined by the log. Checked by `conc06b_run.rs`
  (`restart_log_schema_migration_rule`,
  `actor_and_task_ids_survive_structure_preserving_edits`) and
  `emit_run.rs` `restart_replay_survives_upgrades_and_traps_on_tree_changes`
  (a log recorded on one program replays on an upgrade that shifts every
  line and adds a function, an actor type and a supervisor; removal and
  reorder traps; the version 0 per-decision check; on the reference and
  on the Rust, Python, JavaScript and Scala runtimes, which print the
  header with `restart-log` and take it as `restart-tree S/c,...`).
  The run log of `fn
  main` is not D: actors run by `rce run`, not by the host of Appendix E
  (HC-6).
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
language and are now listed in section 7: W-MOD-3 (an import's name that
clashes with a declaration or with another import's name), W-VIS-1 (use of a name private to its module, the form W-MOD-1
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
| T28 | session duality, subject reduction and progress, with recursion | proved on a session calculus; encoding of surface protocols mechanized; the checker's dual and protocol automaton equal the model's on 60 generated protocols along a joint run (enabled actions at each point, and the end); linearity is T28l | `t28r_preserve`, `t28r_progress`, `enc_comp`, `SessionVectors.lean`, `checker/tests/lean_session.rs` |
| T28l | linear endpoints: the checker's W-SES-1, W-SES-2 and W-SES-4 are sound for a linear typing judgment, and a typed body runs every path without using a gone endpoint or closing an unfinished one, and ends with every endpoint gone or done | proved on a calculus of bodies (send, receive, close, move, branch over message protocols); the checker agrees with the model's `chkBody` (accept, or the rule under which the body requires change) on 120 generated bodies, which found that the checker did not track an endpoint moved by `let m = x` (fixed) | `SessionLin.chkBody_sound`, `ty_safe`, `chkBody_safe`, `SessionLinVectors.lean`, `checker/tests/lean_session_lin.rs` |
| T28d | deadlock freedom: on a tree-shaped session graph with compatible ends, every process has finished or some channel has both ends waiting and they step | proved on a model of networks (processes blocked on one endpoint or finished); the tree is enforced by W-SES-5 at spawns and argued for delegation | `SessionDead.deadlock_free` |
| T28p | deadlock freedom by priorities: if each process's actions strictly increase in priority and both ends of a channel share each action, every reachable state is finished or steps, on any session graph (cycles included); the checker's W-SES-5 accepts a program that requires change under the tree rule when the order on its actions is acyclic | proved on a model of networks (rendezvous, spawns as one-step channels); priorities computed by relaxation are proved to satisfy the rule, and a cycle in the order refutes every priority assignment; the checker agrees with the model (tree rule or priorities, else W-SES-5) on 150 generated programs, 9 of them cyclic and accepted only by priorities | `SessionPrio.deadlock_free`, `decideP_sound`, `cycle_chk_false`, `SessionPrioVectors.lean`, `checker/tests/lean_session_prio.rs` |
| T28g | multiparty global types: the projections of a well-formed global type (plain merge) step only to projections of a global type, and every reachable configuration is finished or steps; a global type of messages only satisfies the priority rule with priority = event index | proved on a synchronous calculus of global and local types with messages and binary choice; the checker's projection equals the model's on 200 generated global types, and its priority decision accepts every message-only one | `SessionGlobal.sr`, `progress`, `deadlock_free`, `global_prio`, `SessionGlobalVectors.lean`, `checker/tests/lean_session_global.rs` |
| T28a | multiparty global types with full merge and asynchronous, order-preserving delivery: every step of a configuration associated with a global type is a step of that global type and stays associated; a receiver accepts the label heading its queue; every configuration reachable from the projections with empty queues is final or steps | proved on an asynchronous calculus of global and local types with binary choice and one FIFO queue per ordered pair of roles; the checker's full-merge projection equals the model's on 300 generated global types, and its exploration of their asynchronous runs finds no stuck configuration and none whose receiver does not accept the label heading its queue | `SessionAsync.sr`, `safe`, `progress`, `deadlock_free`, `SessionAsyncVectors.lean`, `checker/tests/lean_session_async.rs` |
| T29 | LVar puts commute; one-threshold reads are order independent | model; vectors | `lvar_interleave`, `t29_perm` |
| T29q | LVars over any join-semilattice with an absorbing top, threshold reads in the same order, and a frozen bit: two runs that complete every thread without error end in the same configuration (value, frozen bit, every result); without `freeze`, a program with a completing run has no erroring run; with `freeze`, whether a run errs can depend on the schedule | proved on a calculus (put, get with incompatible sets, freeze, on one LVar shared by any number of threads); runs that block forever are not covered; the built-in Merge instances are semilattices | `LVarQD.quasi_det`, `det_no_err`, `quasi_not_det`, `MergeLaws` |
| T29s | A store of many LVars shared by any number of threads (the product lattice over locations, a frozen bit per location): two runs that complete every thread without error end in the same store, frozen bits and results; without `freeze`, no run errs if one completes. W-CRDT-1 certificates are lawful by construction. Each row of `conc::LVAR_RULES` (rule, trap, Lean declaration) and the certificate of each certified test program is generated into `LVarRules.lean`, which `lake build` elaborates, and the same test checks each trap string in the evaluator and every emitter runtime | locations allocated before the threads start (T29a adds allocation); the generated table checks that the named declarations exist and the certificates elaborate, not that the evaluator refines the calculus | `LVarStore.store_quasi_det`, `store_det_no_err`, `MergeCert.cert_laws`, `LVarRules`, `checker/tests/lvar_rules.rs` |
| T29a | The store of T29s with dynamic allocation: `new l` allocates `l` (only once; operations on `l` wait for it), locations from a deterministic counter (`counter_owner`). Two runs that complete every thread without error reach the same store, frozen bits, allocation map and results; without `freeze`, no run errs if one completes. An executable semantics `stepAt` / `exec` is proved sound and complete against the relation: every step `stepAt` takes is a `Step` (`stepAt_sound`, `exec_sound`), and from a configuration with no value at top every `Step` is `stepAt c i` for some thread index i (`stepAt_complete`), so the schedules of `exec` are exactly the runs of the relation. The sorted representation of a `Set[τ]` and the key-sorted one of a `Map[k, v]` are unique: equal membership (equal lookup) implies equal lists (`Canon.set_canon`, `Canon.map_canon`), so the Merge laws stated on membership and lookup (`MergeCert.unionS`, `mapS`) hold of the runtime values. 60 runs extracted from `stepAt` over the lattice 0..6 under max are replayed sequentially on the evaluator and the Rust, Python and JavaScript emitters, which must give the same `wait` and `freeze` results and the same error; every schedule of every vector (each interleaving of the threads' operations in program order in which each wait is enabled, 221 in all) is run on the evaluator's deterministic sequential execution, explored depth first with the evaluator deciding enabledness: without `freeze` every schedule gives Lean's results, or every one errs when Lean's run does; with `freeze` all complete schedules agree and Lean's outcome is among them. All 60 are also run as a threaded `par` on OS threads, one branch per Lean thread, with each `freeze` sequenced after the join, where W-LVAR-1 accepts it; the frozen value is then the join of all puts to the cell, and Lean's mid-run frozen value is at most that | the vectors cover `put`, single-threshold `get`, `freeze` and allocation on integer cells, with at most 9 operations each; the explorer enumerates every interleaving with no partial-order reduction, and allocation is placed before the other operations; `ValOrder` proves the order of `cmp_canon` on Form values (rank null < bool < int < float < text < list < record, IEEE 754 totalOrder on floats, lexicographic text, lists and record entries), and its `cmp` is written from that definition, not extracted from the code | `LVarAlloc.alloc_quasi_det`, `alloc_det_no_err`, `stepAt_sound`, `stepAt_complete`, `exec_sound`, `Canon.set_canon`, `Canon.map_canon`, `ValOrder.lt_total`, `ValOrder.set_canon_val`, `ValOrder.map_canon_val`, `LVarVectors`, `checker/tests/lean_lvar.rs` |
| T30 | an atomic batch applies all or none | model; vectors | `t30` |
| T31 | replicas agree on equal prefixes | model (the statement is close to trivial) | `t31` |

None of T16 to T31 is a proof about the Rust reference evaluator. The
checker rules W-PAR-1 to 4 and W-GRP-1 to 3 are not linked to the Lean
footprints beyond the core of `ParSem.lean` (W-PAR-1 on variables).

### C.14 Emitters and running

`rce run` runs programs on the reference evaluator
(`conc.rs`). The program emitters (`emit/conc06.rs`) run the branches of a
`par` with more than one branch at the same time where the target allows
it (Rust and Python threads, JavaScript workers) and in source order in
Scala; in every case the store, the outputs and the trap point are those
of source order (C.4, T16), the branches' results and costs being merged
in source order after they finish; a `par` whose branches
wait on LVars is run by restart (a repeated attempt puts nothing new,
since puts are joins and W-CRDT-1 makes every join idempotent), and budgets charge the same costs as the
evaluator, so `trap("budget")` falls on the same operation on every
target. Workgroups run lane after lane per phase. The 0.6b actor, session
and node constructs follow the limits of 10.5.

### C.15 Releases

`# rce-version: V` is a release label (2.7): tolerated, ignored, not hashed. **A release label names exactly
one spec commit**: the text of this document at that commit is the language of that release, there is one
language in it, and no label selects among alternatives inside a text. The rule that decides
what needs a new release: the meaning of a module (its outputs, states and outcomes, including where a
budget traps) is determined by the text of its release, so a change that gives a valid program of an existing
release a different meaning that the old text defined is a new release with a new label, never a silent change
to the old text; a change that makes the implementations meet text that already defined the meaning, or that
only touches artifacts outside a program's meaning, is an erratum and goes into the text that the next release
names. A release is never edited after it is named, and a label is never reused for a different commit.

**Costs.** A standard library method call costs `c_call + work(m, n, k)` (C.6.4) wherever it is written and
wherever it is executed from; linking modules changes no cost, since there is only one rule. Example: a function
that holds `with budget B` and calls an imported function whose body sorts an 8-element list is charged 24 more
than a call to a function with no body, in the reference evaluator and every runtime.

| Change since the first text of the item | Kind | Why |
|---|---|---|
| size-scaled method costs (C.6.4) | meaning change | the earlier text defined each method as `c_call`; where a budget traps changed, so it was a new release (now the one text) |
| saturating budget arithmetic (C.6) | erratum | changes nothing reachable: no program reaches a sum near the largest i128 |
| closed standard library, no ambient I/O (5.15) | erratum | states what the checker already enforced (W-NAME-1 on an unknown method); no valid program changes |
| `&mut` write-back on `fail` (6.2, the earlier T33 rule, retired) | erratum | the reference evaluator already updated in place; the emitters now agree with it (and T34 flattens nested calls without changing order) |
| u64 values at or above 2^63 exact in order, min/max and f64 ops (6.3 N2) | erratum | N2 has defined full-range u64 since draft 0.5; the runtimes now meet it |
| literals take the success type of a fallible return (5.1) | erratum | a checker bug against `τ <: τ ! ε`; it admits programs the text already admitted |
| canonical D entry and hash-chained Form log (6.1) | outside programs | the log format, not a program's meaning |
| `sha256` text method (5.15) | extension | admits only programs that required change under W-NAME-1 before, so no valid program changes meaning |
| mutual recursion with a decreasing summed measure (6.7, W-REC-1) | extension | admits only programs whose call graph has a cycle of two or more functions, which required change under W-REC-1 before; the rule for a function that calls only itself is unchanged, so no valid program changes meaning |
| maps iterate in canonical key order (6.7, 7.1) | erratum | 6.7 has always said a map iterates in canonical order; the reference evaluator and the four emitter runtimes kept insertion order. They now keep every map sorted by the canonical order of its keys, so `for`, `keys()`, `values()` and display follow 6.7. A program whose output depended on insertion order was relying on behavior the text never gave it; tests/map_order.rs checks all targets |
| self-recursion needs one fixed shrinking position, or layers (6.7, W-REC-1) | erratum, and extension for layers | 6.7 has said since draft 0.4 that the value at one position i shrinks along any chain of calls, and T2 that every run ends; the checker took a different position per call, which admitted functions that never return. Requiring change in them makes the checker meet the text. Every function the text admitted is admitted by one layer, and layers admit only programs that required change before, so no program the text admitted changes meaning |
| a parameter typed by an alias of a channel type makes a task function (5.1, 5.2) | erratum | the type grammar of 5.1 has no nominal alias, so `type A = chan τ` already named the channel type; the checker tested the written type, not the type it names, and reported that such functions required change (W-SPAWN-1 or W-EFF-1). Only programs that required change before are admitted |
| declared recursion bounds `fn … bound n` (6.7, W-REC-1) | extension | `bound` after a signature was W-SYN-1, and a cycle it admits was W-REC-1, so only programs that required change before are admitted; no valid program changes meaning |
| est_j receipts | outside programs | measurement records outside the language (C.5: est_j is separate from budgets) |
| `i64.MIN % -1` traps; `toint` traps outside i64 (6.3 N2) | erratum | N2 already defined both; the reference and runtimes returned 0 and an out-of-range i64 |
| canonical float text in `tostr` and display (6.3 N3) | erratum | N3 already defined it (`1e+300`, `1.5e-7`, `-0` is `0`); the implementations printed Rust's long decimal form |
| `x op= y` charges the cost of `op` (C.6.4) | erratum | C.6.4 already charges every binary operator; compound assignment was charged 0, so a budget could be exceeded |
| 6.3 N2, N3, N8 and 6.2 state the trap texts, `usize` indices and trapping slices the implementations have always had; float `%` removed from N3 and C.6.4 | erratum | the text contradicted the checker and every runtime; no accepted program changes, since the text described behaviour no program could reach (negative usize, float `%`) or contradicted what every implementation did |
| f64 `min`/`max` (6.3 N5, 5.15) evaluate | erratum | the checker accepted them and N5 defined them, but the evaluators trapped; a program that trapped now computes what the text defined |
| f64 `pow` requires change (W-NAME-1) | erratum | not in the closed library of 5.15, which already made it W-NAME-1; the checker accepted it and every evaluator trapped on it, so no accepted program computed a value with it |
| refined types `type R = τ where p` (5.1, W-REF-1) | extension | `where` after a type was W-SYN-1 and stays an ordinary identifier, so only programs that required change before are admitted; no valid program changes meaning |
| a foreign function is not a value (5.11, W-FFI-1) | erratum | 5.11 already forbade a pure function calling foreign code; the value route was a checker hole |
| host contract (Appendix E: one admitter, durable append per OS, fatal barrier, verify on open, inputs as events, capabilities, fabric entries, automatic placement) | outside programs | constrains hosts and the log, not programs: a host entry has a `host.` type that no program can declare and δ is the identity on it, so the fold of every log of every valid program is unchanged |
| host bindings (E.8: a host result delivered as an event the program declares) | outside programs | the bound event is an ordinary declared event, admitted and typed as any other; a program that declares no binding target sees no new input |
| `state_hash` and the entry `hash` over the positional forms (6.1, T7) | erratum, outside programs | the finding of 2026-10-22: the Lean model of T7 hashes declaration indices, the checker and the four runtimes hashed the canonical JSON keyed by region, field and type names, so a renamed program had different log hashes. The hashes now cover the positional forms and a renamed program has the same hashes, as T7 states. No program's outputs, states or outcomes change; every trace fixture, the conformance pins and the trace format (`rce-trace/2`) change, which is the log format and not a program's meaning |
| a kept step does not end a run, a trace or an example (6.9, 10.3, 10.9) | erratum for runs and traces, extension for examples | T3 folds every event and HC-21 applies the next input after a kept step; `rce run`, `rce trace`, `rce run-core` and the emitted runtimes ended at the first kept step, so they now meet T3. An example whose earlier event is kept required change under W-INT-1 before and now holds when its last step and final state are as it states; every example that held before holds with the same trace, so no valid program changes meaning |

The first row is the one change since draft 0.6 that moved where a `with budget` block traps; it is
what made the earlier text a different release. Every other row is an erratum, an extension or a
change outside a program's meaning. Checked by `spec/checker/tests/method_costs.rs` (every method charges the table in
the reference and every runtime), `spec/checker/tests/call_site_cost.rs` (the same cost in the reference
evaluator and the Rust, Python and JavaScript emitters on a linked program) and
`spec/checker/tests/reserved_words.rs` (the one reserved list).

### C.16 Indexed values

The eight names of 5.16 are on the reserved list (2.4): a program may not declare `struct Text` or
`struct Html`, which earlier drafts allowed, and that requirement is part of the release that first carried
the list (C.15). `C.lit(s)` costs `c_call`, `C.encode(t)` costs `c_call` plus the size of t (the linear class
of C.6.4) and `.raw()` costs `c_call` (the text is not copied), in the reference evaluator and every runtime;
the indices are erased, so nothing else in C.6 changes (`spec/checker/tests/indexed.rs`).

### C.17 Appendix F

Appendix F is part of the language. Under it, programs that earlier drafts accepted require change (unused values, shadowing,
dropped results, flows to lower labels), and it changes the value of a `&mut` argument after a failed call, so it
is a new release by the C.15 rule. It is specified and implemented in part: a construct of it that an
implementation does not have requires change (W-RES-1, *reserved, not implemented*, F.12), and no cost of C.6 is
defined for it beyond the costs F.10 names.

**Read receipt.** How many concurrency words does C.1 add to the keywords of 2.4? (a number)

<!-- body-sha256: cbf255393d0ebb75e6883a0dcba027fa9680e9a25235d3cffa785e9b1bfb27d9 -->
