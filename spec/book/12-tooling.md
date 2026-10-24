## 10. Tooling: the `rce` command and the language server

This section defines the tools that read RCE programs: the `rce` command
and the language server it contains. It adds no rules to the language; it
says what each tool prints and returns for a given input. Example:
`rce run counter.rce "Inc, Dec, Dec"` prints the state after each event
and stops at the refused third event with exit status 1.

### 10.1 Principles

1. **Determinism.** Every subcommand of 10.2 to 10.10 is a function of its
   arguments and the bytes of the files it reads. Two runs on the same input
   give the same output bytes and the same exit status. None of them reads
   the clock, the network, environment variables or random numbers. The same
   binary also carries the host commands (`rce host ...`) and `rce db`,
   defined by Appendix E; the host loop reads the clock (the frame period
   of `Tick`) and the network by design, and logs what it reads as inputs
   (HC-2, HC-14), so this item does not describe them. The handler-spec
   staleness rule W-HS-CUR-2 takes a time as an input and is part of the
   engine profile (D.9, D.10), not of these subcommands.
2. **No model in the loop.** No subcommand and no language server request
   calls a language model, a remote service or any learned component. A
   diagnostic, a run, a test verdict, an emitted file and a formatted file
   come only from the rules of sections 2 to 7.
3. **One checker.** `rce check`, `rce test`, `rce run`, `rce emit` and the
   language server all use the same parser, type checker and admission gate
   (6.9) as the reference checker `rce-check`. A file accepted by one is
   accepted by all.
4. **Exit status.** 0 means success. 1 means a program was refused, an
   example or check failed, a run reached a refused event, or `fmt --check`
   found a file to change. 2 means a usage error: unknown subcommand, wrong
   number of arguments, unreadable file, or an unknown emit target or
   construct an emitter does not translate (10.5).

`rce` is one binary, built from the reference checker crate in
`spec/checker` (crate `rce-spec-check`, `cargo build --release --bin rce`).
The language commands of this section are implemented in
`spec/checker/src/cli.rs` (entry point `cli::main`), with their tests in
`spec/checker/tests/`. This `rce` carries the language commands only; the
Form command line (`rce form ...`), which reads the Form spec format rather
than RCE programs, is not part of this build, and `rce form` exits with
status 2.

The tables of 5.15 (which methods each receiver has) and C.6.4 (the work
class of each method) are not typed into the checker a second time: it
reads them from `spec/data/spec-data.json`, which `rce-spec-data`
generates from this book. The two tables are cut from it into
`spec/data/spec-tables.json`, the file compiled into `rce`, the checker
and its wasm build; `rce-spec-data --check` fails when either file
differs from what the book gives.

### 10.2 `rce check FILE|DIR...`

All FILEs form one program: `use` resolves among them (4.1). A DIR is a
package: it stands for its `.rce` files, recursively, in sorted order,
and reads nothing outside it (the multi-module example is the package
`spec/examples/26_use_module/`, so `rce check spec/examples/26_use_module`
checks it alone). Each file is
parsed, type checked (section 5) and admitted (6.9: every `const`,
`comptime` block, `check` and `example` is evaluated). For each file, in
the order given, one line is printed:

```
ok FILE
REFUSED FILE: line N: RULE message[; line M: RULE message ...]
```

Every refusal names its rule from section 7. Exit status 1 if any file is
refused, else 0.

### 10.3 `rce run FILE [EVENTS]`

The file must pass `rce check` alone; otherwise one `REFUSED` line is
printed and the status is 1.

**With EVENTS.** EVENTS is written exactly as the event list of an
`example` line (3.2, `EventRef { "," EventRef }`), for example
`"Inc, Inc, Dec"` or `"Deposit(amount: 50), Withdraw(amount: 20)"`.
Let s₀ be the initial state (6.1) and e₁ … eₙ the events. For k = 1 … n the command computes

```
(sₖ, oₖ) = δ(sₖ₋₁, eₖ)
```

where δ is the step function of 6.4 and oₖ is the sequence of values the
step emitted (λ). It prints line 0 as `0 (initial) STATE`, then for each k
the line `k EVENT STATE` followed by one line `  emit VALUE` per element of
oₖ, in order. STATE lists every region as `Region { field = value, ... }`,
regions separated by `; `, in declaration order. If δ refuses eₖ, the line
is `k EVENT refused: ERROR`, no later event is applied, and the status is
1. Otherwise the status is 0. An event name that is not declared is a usage
of a wrong event list: one `REFUSED` line, status 1.

For the counter `spec/examples/01_counter.rce`:

```
$ rce run 01_counter.rce "Inc, Inc, Dec, Dec, Dec"
0 (initial) Counter { n = 0 }
1 Inc Counter { n = 1 }
2 Inc Counter { n = 2 }
3 Dec Counter { n = 1 }
4 Dec Counter { n = 0 }
5 Dec refused: Underflow
```

**Without EVENTS.** The module must declare `fn main()` with no
parameters. The command evaluates its body by the rules of section 6 and
prints `emit VALUE` for each value emitted, in order. If the body traps or
fails, the last line is `refused: REASON` and the status is 1. Calls to
foreign functions (5.11) are not executed by the reference evaluator and
trap. If there is no `fn main()`, the command says so and the status is 1.

The evaluator is the reference interpreter of the checker, with the fuel bound
of 6.7 and the VALUE_LIMIT of 6.3. It is a definition of meaning, not a fast
runtime.

### 10.4 `rce test FILE|DIR...`

The test cases of a module are its `example` lines and its `check` items,
including those inside `intent` blocks. `rce test` admits the files as
`rce check` does, which evaluates every case (6.9, W-INT-1). It prints, per
file,

```
ok FILE (E examples, C checks)
FAIL FILE: line N: RULE message[; ...]
```

with E and C the numbers of cases in that file, then a total line
`P passed, F failed`. Exit status 1 if F > 0. A test verdict therefore
equals the admission verdict: a module whose examples do not hold is not a
program (section 7).

### 10.5 `rce emit TARGET FILE`

TARGET names an output. The file must pass `rce check` on its own.

| TARGET | Output |
|---|---|
| `form` | the Form P = (S, E, δ, D, λ) as one line of JSON (6.1), then a line end |
| `rust` | one Rust 2024 source file with a `main` |
| `python` | one Python 3 source file |
| `js` | one JavaScript module (ES2020 or later, BigInt) for Node |
| `scala` | one Scala 3 source file with a `@main` for the JVM (Java 19 or later) |

The Form JSON lists objects in canonical order (6.1), so it is a
deterministic function of the program text.

The four program targets are the ones this checker emits. The language
places no bound on targets: every language is both an output of the Form
(a target, printed from the Form IR through a per-language table) and an
input (a lift, the reverse map, from the language's grammar and a
semantics table), and every target passes the same gate, the reference
trace of every example program byte for byte (T9). The registry of targets
and the one-emitter mechanism are the proposal `spec/proposals/emit-targets.md`
(informative); a target that the table cannot pin to 6.3 is observation-only,
never a projection.

**The program emitters.** `rust`, `python`, `js` and `scala` translate the checked
module, not the Form. Each output is self-contained. Its first line is a
comment naming the program hash (10.8: the SHA-256 over the definition
hashes of F.4.3, name-free and layout-free) and the checker version and
commit that printed it, so an emitted file is tied to its source. It starts with a
runtime that defines the values of 6.3 and their operations, following the
reference evaluator (`spec/checker/src/interp.rs`) value by value. That
includes the integer range checks, the trap messages, canonical map key order
and the display of floats and text. After the runtime comes one target
function per RCE function, method, step and constant. An expression becomes
calls to runtime helpers with its operands in evaluation order; `if`,
`while`, `loop`, `for` and `match` become the target's own control flow,
with one counter per loop against its bound. Values are never changed in
place: an assignment to `a.f[i]` rebuilds the value, as the reference
evaluator's `assign` does.

The Rust output follows the shape of the verified emitter of T9
(`spec/lean/RCE/Emit.lean`): one value type, `Result` with `?` for traps
and refusals, and i128 arithmetic checked against each integer type's
range. The theorem of T9 is about the Lean model of that emitter. The
emitter in `rce` is tested against the reference evaluator; it is not
proved.

The Scala output is the functional-language target. Its values are the
same dynamic values as the JavaScript output's (integers as `BigInt` with
their type, floats as JVM `Double`, whose arithmetic is IEEE 754 binary64
with round to nearest even and no contraction since Java 17, and whose
`toString` is the shortest round trip since Java 19). Loops are `while`
loops, and `break` and `continue` are control exceptions caught at the
loop, so no RCE loop grows the stack; the observations run on one thread
with a 2 GiB stack.

**What an emitted program does.** Run it, and it prints the module's
*observations*: one line per constant, per `check`, per function without
parameters (other than `main`), per event of each example (the state after
it, its outputs, or the refusal), and finally the outputs and result of
`fn main`. These are the lines `emit::observe` computes with the reference
evaluator, so that a target and the reference can be compared line for
line.

**How the emitters are tested.** `spec/checker/tests/emit.rs` emits all 62
programs of `spec/examples/` (the 60 files and the 2 of `26_use_module`) in
all three targets. To exercise function
bodies, it first adds *probes*: for each function and method, up to three
calls on small sample arguments built from the parameter types, plus one
example that sends every event. It then runs each output (`rustc`,
`python3`, `node`) and requires its output to equal the reference
evaluator's, line for line. On 2026-10-16, with rustc 1.98.1, Python 3.13.5
and Node 20.19.2, all 58 agree on all three targets, with 474 observation
lines per target. (The dated runs below were made when the corpus was 51
programs, and 55 for Scala, with 299 and 361 observation lines.) Four programs (14, 15, 34, 47) contain only
concurrency and give no observation. A separate program
(`tests/emit_edge.rce`) covers range traps, loop bounds, `&mut` write-back,
closures and float display. The Scala target has its own test in the same
file: on 2026-10-14, on the Apple M5 Max below, with Scala 3.10.0 run by
scala-cli 1.18.0 on OpenJDK 27, all 55 programs of the current corpus
(with the package files of example 26) agree with the reference, 0
refused, 361 observation lines, and the edge program agrees on all four
targets. Its oracle is `RCE_SCALA` (default `scala-cli`). That agreement is for this examples corpus only. On the larger `rce-conformance` suite the Scala emitter passes 52 of 1,845 results (its README gives 1,189 failures and 604 unsupported), so the Scala target does not yet agree with the reference on that suite.
When `rustc`, `python3` or `node` is missing,
its test prints `SKIPPED` and the reason; a skipped target is never
counted as a pass. The oracles used were Python 3.13.5 and Node 20.19.2.
The test reads the oracle paths from `RCE_RUSTC`, `RCE_PYTHON` and
`RCE_NODE` (default `rustc`, `python3`, `node` on the path) and prints each
oracle's version. On 2026-10-12 the same tests were also run on an Apple
M5 Max (macOS 27.0.1, arm64) with rustc 1.98.1, Python 3.14.5 and Node
26.11.1: 51 of 51 agree on each of the three targets, 0 refused, 0
skipped, 299 observation lines per target, and the edge program agrees on
all three.

**Fuel.** Each emitted program counts fuel (6.7) at the reference
evaluator's charge points: one tick per evaluated expression, per
executed statement, per call of a user function or method, and per
element of a list comprehension and of `map`, `filter` and `fold`. A
temporary introduced by the emitter's own lowering is not charged. The
limit is 50,000,000; it is reset before each observation, as the reference
starts a new evaluator for each, and running out traps with the message
"comptime fuel exhausted". The argument `--fuel=N` sets the limit to N and
`--fuel-trace` prints `fuel K` after each observation, K being the ticks
used. These two arguments exist for the test below; the default program
takes no argument.

`spec/checker/tests/fuel.rs` compares the ticks used, line for line, on
the 50 programs of the emitter tests (with probes) against
`emit::observe_fuel` of the reference evaluator, in all three targets. It
then runs ten programs at every limit from 0 up to 300 and at 400 evenly
spaced limits above, up to one more than the program uses, and requires
each target to trap at the same observation with the same output as the
reference run at the same limit. Within one observation, the emitted
order of ticks can differ from the reference where the emitter evaluates
arguments before entering a call; the trap observation and the total are
what the test fixes. The Scala target is in the same test (the emitted
file is packaged once into a jar and run with `java -jar`); on 2026-10-14
it agreed on all 54 programs of that corpus and at all 1,366 swept limits
(`RCE_FUEL_LANGS=scala`).


**Running `fn main`.** Run with the argument `--run`, an emitted program
instead evaluates `fn main` with tasks, channels, `select`, actors and
`par` on a runtime of its own: every task is a thread (a worker in
JavaScript), and one scheduler admits their moves tick by tick in the
order of `conc::run_main`, the reference of `rce run` without EVENTS. It
prints `emit VALUE` per output and the line `outcome Done`,
`outcome Deadlock` or `outcome Refused: REASON`. A module without
`fn main()` prints the reference's message ("nothing to run") on standard
error and exits with status 1, as `rce run` does. Fuel follows the reference's
pools: each task has its own, starting full (its first ticks are the call
of its function and one per argument); actor handlers share the
scheduler's pool; a `par` branch or `for ... par` chunk starts from its
parent's remaining fuel, and the branches are charged back in source
order, so the trap falls where the reference's sequential run puts it.

`spec/checker/tests/emit_run.rs` compares the three runtimes with
`conc::run_main` on a corpus of task programs. Among them are the four
concurrency examples 14, 15, 34 and 47, each run with a `fn main` from
`spec/checker/tests/drivers/` appended to its unchanged text.
`spec/checker/tests/fuel_run.rs` runs seven of these programs at every
fuel limit where the reference's result changes and at the limit before
it (258 runs over the three targets), and requires the same outputs and
outcome.

**Limits of the emitters.**

- In the observations, channels, `spawn`, `select`, actors and `extern`
  functions trap with the same message as in the reference evaluator
  ("not available at comptime"); they run only under `--run`. `extern`
  functions trap there too.
- A `par` whose branches wait on an lvar runs its branches at the same
  time in the reference (`ParMode::Threads`). Derivation. The governing
  law is schedule independence: a `par` must be observationally equal to
  one fixed sequential order, source order (C.4). Fuel is additive over
  sequential composition, so the sequential run traps at the first
  branch i where the uses of branches 1..i sum past the fuel left at the
  fork. The reference therefore gives each branch a copy of that fuel,
  sums the branches' uses in source order after they finish, and traps
  with "comptime fuel exhausted" at the first branch whose running sum
  passes it, before that branch's own result is taken. The trap point is
  the sequential one. A split of the fuel before the fork would make the
  set of programs that trap depend on the split rule and would agree
  with no sequential order, with no gain in soundness, so it is not
  used. The cost of running branches ahead is bounded: at most k times
  the fuel left at the fork is evaluated for a `par` of k branches, of
  which the excess over the sequential run is discarded. A wait charges
  no fuel while blocked, so a branch's use does not depend on the
  schedule. The emitters run such a `par` by restart and restore the
  fuel and the budgets of an abandoned attempt (`par_snap`,
  `par_restore`), so each branch is charged once, for its completing
  attempt. `spec/checker/tests/fuel_run.rs` checks this at every change
  point on two programs (`lvarpar`, and `lvartrap`, where a late
  division by zero must lose to the fuel trap when fuel runs out first).
- Deep recursion ends with the target's own stack error.
- A call that writes back its receiver or a `&mut` argument (`xs.push(x)`,
  `f(&mut a)`) is translated only as a whole statement or as the right side
  of `let`, `var`, an assignment, `return`, `emit` or `fail`. Elsewhere,
  `rce emit` refuses the program with status 2 and names the construct. No
  example program is refused.
- When a function whose result is a union fails, its `&mut` arguments are
  not written back. The reference evaluator writes them back.
- The Python output has been checked with Python 3.13.5 and 3.14.5, and
  the JavaScript output with Node 20.19.2 and 26.11.1. Other versions have
  not been run.
- The Scala output has been checked with Scala 3.10.0 on OpenJDK 27 only.
  It has no `--run` runtime yet: with `--run` it prints that `--run` is not
  available on this target and exits with status 2. A `par` runs its
  branches in source order on one thread (by restart when they wait on
  lvars), as the JavaScript output does when a local cannot go to a worker.


For an unknown target, or a program that one of these emitters does not
translate, `rce emit` prints nothing on standard output, prints the reason
on standard error, and exits with status 2. It never prints a partial file.

### 10.6 `rce fmt [--check] FILE...`

`rce fmt` rewrites each file into canonical layout. Let tok(t) be the token
sequence of text t produced by the lexer and layout pass of 2.6, with line
numbers dropped. The formatter is a function fmt on texts with:

1. **Meaning preserved.** tok(fmt(t)) = tok(t). The formatter checks this
   equality on every file before writing it; if it would fail, the file is
   left unchanged, a `REFUSED` line is printed and the status is 1.
2. **Canonical layout.** In fmt(t): a block that is k levels deep starts its
   lines with exactly 4k spaces; no line has trailing spaces; there are no
   blank lines before the first line, no two blank lines in a row, and the
   file ends with exactly one line end. A comment-only line takes the
   indentation of the next code line. A line that continues inside an open
   bracket keeps its position relative to the line that opened it.
3. **Idempotence.** fmt(fmt(t)) = fmt(t). This is tested on every program
   in `spec/examples/` and on the formatter's tests; it is not proved.

A text the lexer refuses (2.6, W-LEX-1, W-LEX-2) is refused by `fmt` with
the same reason. `fmt` does not type check: a file that is refused by
`rce check` can still be formatted. With `--check`, no file is written;
each file that would change is printed as `would reformat FILE` and the
status is 1. Without it, each changed file is written and printed as
`formatted FILE`.

**`rce fmt --nameless FILE`** prints the name-free projection of one file
to standard output and writes nothing. Every identifier the module declares
is replaced by a positional name in order of first declaration, by kind:
regions `r0, r1, …`, events `e0, …`, errors `x0, …`, functions `f0, …`,
types `t0, …`, kind variants `k0, …`, fields, parameters and locals
`v0, …`. Keywords, built-in types, standard library names, string contents
and `intent` text are unchanged; the wildcard `_` and a `_` prefix are
kept, as are `main`, the built-in traits and their methods, the host
events and the named constants of 6.7. A standard library method after
`.` keeps its name even when the module declares a field of that name
(the method table of 5.15 is closed and is not a declaration of the
program); a function the module declares is renamed at its call sites. The projection is deterministic and
idempotent, and it has the same outcomes, outputs, ids and log hashes as
the source: `rce trace --json` of the two is equal step for step except
in the readable `event` field, since 6.1 hashes the positional forms
(T7). A name the
projection does not recognise as declared is left as written, so the
output is accepted whenever the input is (`tests/nameless.rs`, every
example). It is the text of the concept record's complement (6.6).

### 10.7 The language server: `rce lsp`

`rce lsp` speaks the Language Server Protocol 3.17 over standard input and
output (JSON-RPC 2.0 messages framed by `Content-Length` headers). It keeps
the full text of each open document; it does not read other files.

| Message | Behaviour |
|---|---|
| `initialize` | replies with capabilities: full-text sync (`textDocumentSync` = 1), formatting, hover, definition, document symbols, and completion triggered by `.` |
| `textDocument/didOpen`, `didChange` | stores the text and publishes diagnostics for that document |
| `textDocument/didClose` | forgets the text and publishes an empty diagnostic list |
| `textDocument/formatting` | returns one edit replacing the whole text with fmt(t) (10.6), or no edit if t is canonical or refused |
| `textDocument/hover` | the type or declaration of the name at the position, or `null` |
| `textDocument/definition` | the location of that name's binding or item, or `null` |
| `textDocument/completion` | the names that can be written at the position |
| `textDocument/documentSymbol` | the document's items, each with its fields, variants or methods |
| `shutdown` | replies `null` |
| `exit` | ends the server: status 0 after `shutdown`, 1 without it |
| any other request | error -32601, method not found |

**Diagnostics.** For a document with text t, the published diagnostics are
exactly the refusals `rce check` gives for t, one per refusal, with severity
Error and source `rce`. A refusal `line N: ...` covers line N − 1 in the
protocol's 0-based numbering, from column 0 to the start of the next line.
`use` resolves among the module paths of all open documents.

**Names.** Hover, definition, completion and symbols all read one index of
the document. The index is built from the parsed module and from the
bindings the type checker declares while it checks the module: each local
binding, with its line and its inferred type. A document that does not
parse has an empty index, so these requests answer `null` or an empty list.

- *Hover* on a local binding shows `name: T`, with the type T the checker
  inferred. On an item it shows the item's declaration: the line of a
  function, constant, event or alias, or the whole block of a kind, struct,
  state or trait. On a variant, field or method it shows that line.
- *Definition* gives the line of the binding or item, and the column where
  the name appears on it.
- *Completion* after `.` offers user methods, fields and the builtin
  methods of the reference evaluator. Elsewhere it offers the local
  bindings of the enclosing item, the module's items, variants and fields,
  and the keywords. Each list is filtered by the part of the name already
  typed.
- *Document symbols* list the items in order, with fields of structs and
  states, variants of kinds and methods of impls as children.

Limits: the AST has no columns, so a name is found by scanning its line,
and columns are counted in Unicode scalar values rather than the UTF-16
units of the protocol; the two counts differ only on a line with a character outside
the Basic Multilingual Plane. A local is visible from its line to the end
of its enclosing item: block scope is not tracked, so a binding of a closed
block is still offered. Names are resolved within one document, not across
`use`.

Because diagnostics come from the same checker, an editor shows a program
as clean exactly when `rce check` accepts it (10.1, principle 3).

### 10.8 What is not yet implemented

- The program emitters do not run concurrency or actors in their
  observations, only under `--run`, and never run `extern` functions
  (10.5).
- The language server resolves names within one document, without block
  scope, and counts columns in Unicode scalar values (10.7). It has no
  rename, references or semantic tokens.
- `rce run` uses the reference interpreter; there is no compiled runtime.
- Idempotence of `fmt` is tested, not proved.
- Version 0.9 (Appendix F) is implemented in part (F.12). `rce hash`,
  `rce api-diff OLD NEW` (F.4.3) and `rce v09-status` exist: `rce hash FILE|DIR...`
  prints the definition hash, kind and name of every named item (`pub` marks an
  export) and, last, the program hash (`program`: the SHA-256 of the canonical
  JSON list, sorted, of the definition hashes of the name-free projection
  (10.6) of the program, with every unnamed item (steps, impls, examples,
  checks) hashed by its masked text, under the domain `rce-program/1`;
  name-free, layout-free and order-free, so a program and its renaming have
  one hash and a changed step body changes it), `rce api-diff OLD NEW` prints one row per change of the exports
  (`compatible` or `breaking`, then kind and name, then the change; status 1 when
  any row is breaking) and `rce v09-status` prints the implemented and the
  unimplemented rules. `rce manifest-check` (F.5.3) checks a manifest's
  module and dependency hashes. The broker contract (F.7), the provenance
  record and the build cache of F.5.3, and the emitter refusals of F.5.4 do
  not exist yet. Results are
  reproducible from the inputs alone (10.1).

### 10.9 `rce trace --json FILE [EVENTS]`: reference traces (informative)

The trace format lets an independent projector (another compiler or
runtime of the same Form) show that it agrees with the reference
evaluator step by step. It adds no meaning: every field is a field of the
log D of 6.1, so a trace is the log, and conformance to 6.1 implies
conformance to the trace.

`rce trace --json FILE` prints one trace per `example` of the module, in
source order (the same logs `rce emit form` puts in `logs`); with
`--streams PATH` it prints one trace per event list of PATH (one list per
line, in the syntax of an `example` line; a line starting with `#` is
skipped), named `stream@k`, in one run of the checker, with `streams`
carrying their number; with EVENTS
(the syntax of an `example` line, as in 10.3) it prints one trace of that
event list. A module that is refused, or a trace of EVENTS that yields no
step, is status 1. The output is one JSON object (format `rce-trace/2`;
`rce-trace/1` differed in hashing the named forms, before the T7
resolution of C.15), compact, with sorted keys:

| Field | Meaning |
|---|---|
| `format` | `"rce-trace/2"` |
| `file` | the file name, without directories |
| `events` | the EVENTS argument, or null |
| `genesis` | the first `prev_hash`: 64 zeros (6.1) |
| `traces` | list of {`source`, `steps`}; `source` is `"example@L"` (the example's line) or `"events"` |
| `steps[i].seq` | required: 0-based index in the trace |
| `steps[i].event` | required: {`type`, `payload`}, payload values as in 6.1 |
| `steps[i].state_hash` | required: SHA-256 (lower-case hex) of the canonical JSON of the positional form of the state after the step (6.1) |
| `steps[i].prev_hash`, `steps[i].hash` | the 6.1 chain: `hash` is the SHA-256 of the canonical JSON of {event, prev_hash, seq, state_hash} with the positional form of the event |
| `steps[i].outcome` | `"applied"` or {`refused`: error name}; derived, not hashed |
| `steps[i].hashed` | {`event`, `state`}: the positional forms the two hashes are over (6.1), so a reader recomputes both without the module; derived |

A refused event is the last step of its trace, with the unchanged state's
hash (T8). A projector conforms on a module when, for every trace, its
steps have the same `seq`, `event` and `state_hash` as the reference, in
order and with none missing or extra; when it also emits `prev_hash`,
`hash` or `outcome` they must be equal too. Since `state_hash` covers
the whole state, equal hashes mean equal states without exchanging them.

`spec/traces/` holds the reference traces of the example corpus: one
file per top-level example (the package `26_use_module/` has no state
machine), plus two event-list traces, `01_counter.events.json` (the
counter on a longer list) and `16_multi_region.events.json` (two regions
stepped by one event, which has no `example` lines). Examples without a
state machine have an empty `traces` list. `spec/traces/rce-trace-diff.py REFERENCE OTHER` (Python 3,
standard library only) compares two trace files or two directories of
them and prints `exact match` or the first divergence (trace, `seq`,
field, both values); exit 0 on full match. `spec/checker/tests/traces.rs`
checks that the fixtures are current, that every step is on the 6.1
chain, and that the runner finds a planted divergence.

### 10.10 `rce index [--vectors N] FILE|DIR...` and `rce search CATALOG QUERY`: the brick catalog (informative)

`rce index` checks the files as one program (10.2) and prints one JSON
object (format `rce-catalog/1`, compact, sorted keys) with one entry per
top-level function of each module (the `pub` ones when the module marks
any `pub`, else all; methods are not bricks), sorted by module and name.
Library dependencies are resolved first: a `# uses: a/b` line adds
`a/b.rce` from the nearest ancestor directory that has it; a `#!lib NAME`
line checks the module with the items of module NAME in front (its checks
dropped, its own items winning), as the rce binary does; and a module
refused only for unknown names that exactly one single-segment module of
the program declares is checked again with that module as its `#!lib`.
`resolved` records each resolution and whether it was a directive or
inferred.
An entry holds `module`, `name`, `file`, `line`, `generics`, `params`
(name and type), `ret`, `key` (the type with generic parameters renamed
`t0`, `t1`, ... in order of first occurrence), `hash` (SHA-256 of the
module path and the canonical text of the function alone, so layout and
position do not change it), `checked` (the verdict of its file) and
`lowering` (`{"term": T}`, the D.7 lowering, or `{"refused": reason}`).
The catalog's `hash` is the SHA-256 of the canonical JSON of its entries;
`refused_files` maps each refused file to its diagnostics. With
`--vectors N`, `vectors` maps a lowered brick's hash to up to N
`{"args", "out"}` or `{"args", "trap"}` records of the reference
evaluator on inputs generated from the parameter types by a fixed rule
(an input that needs more than 200000 interpreter steps gives none, as the
store-passing terms do more work than the interpreter); they are outside the catalog
hash. `rce-brick-verify CATALOG` (`src/form/brick.rs`) evaluates each
lowered term on its vectors and reports, per brick, agreement or the
first disagreement: a result must be equal, a trap must be an error.

`rce search` reads a catalog and a query `A -> R` or `(A, B) -> R`, where
one-letter lowercase names are type variables, and prints `RANK
module.name : key` per match, best first. A brick matches when its
arguments and result unify with the query's, in some order of the
arguments. RANK is 0 for a type equal up to renaming of variables, 1 for
an instance, plus 2 when the arguments match only in another order. Ties
break by module and name. Both commands are deterministic: equal inputs
give equal output bytes.

### 10.11 `rce dim [--fuel N] [--control] [--json] FILE`: behaviours and redundancy (informative)

`rce dim` checks the file (10.2) and measures the quantities of 6.10 on its
Form. It explores reach(P) breadth-first from s₀: at each state it applies
every declared event with payloads drawn from each field's refinement type
(T35 fragment: the bounds and the values next to them) or, for a field
without one, from the boundary set of 6.3 (the type's minimum, −1, 0, 1,
maximum; the empty text, list or map and one of one element). Exploration
stops at `--fuel` states (default 100,000) and reports whether the bound
was reached. On the explored graph it runs partition refinement (Hopcroft)
with the per-step output of λ as the initial partition, giving the number
of distinct behaviours among explored states, which is dim(P) exactly when
the bound was not reached and a lower bound otherwise. It prints, per
program and per region:

```
states explored     12480    bound reached: no
behaviours          1206     dim(P) (exact)
redundancy          3.37     bits: log2(states) - log2(behaviours)
  region Counter    0.00
  region Audit      3.37
control states      17       dim_c(P)          [--control]
shape_ctl           sha256:…                   [--control]
```

A region's redundancy is computed from the projection of the explored
graph onto that region's fields. `--control` computes the control quotient
of 6.10 on the same explored graph: each step is labelled by its control
outcome only (the refusal's error, by declaration index; else the
constructors of what was emitted, by declaration index or type class) and
never by a value, a name or a state change, so the quotient is a coarser
output function on the same deterministic machine and dim_c ≤ dim holds by
construction. Predicate equivalence is decided by concrete
evaluation on reach(P): exact on the explored set, a lower bound when the
bound is reached. The canonical control string numbers the quotient's
states by first visit in breadth-first order from s₀ with events in
canonical order, one line per (state, event) with the successor's number
and the label; `shape_ctl` is the SHA-256 of its canonical JSON text. It
is invariant under renaming (T61, tested by `tests/dim.rs` against
`rce fmt --nameless`).
The report also says whether some explored step has a λ output (6.1): an
emitted value, or the record of a refusal (`has_output` in the JSON, with
`emits` and `refuses` reported separately, since a refusal is the rejection
of an input and an emitted value is the program's answer), and how many
distinct output sequences the explored steps carry (`distinct_outputs`; a
silent step is excluded). A program with no such step has one behaviour by
definition, so its redundancy is its whole state: the state is in D and
readable, and an `example` may check it, but no future output depends on
it. A program whose every output is the same has one behaviour as well.
The dimension benchmark reports these strata separately.
`--json` prints the same as one JSON object (format `rce-dim/1`, compact,
sorted keys; `has_output`, `emits`, `refuses` and `distinct_outputs` since 2026-10-24, replacing `observable` and `distinct_labels`). The command is deterministic: equal inputs give equal output
bytes, and the exploration order is the canonical order of 6.1 on events
and payloads. It is informative: no admission depends on it.

`rce equiv [--fuel N] [--json] A B` asks whether two programs have the same
behaviour (6.10): it explores both from their initial states over the same
event instances, compares every step by its name-free output (emitted
values in their positional form, a refusal by its error's declaration
index, T7), refines the two graphs together by those outputs and reads
whether the two initial states fall in one class (T58: equivalence is
bisimilarity). Status 0 and `equivalent` when they do; otherwise status 1
with the shortest distinguishing event sequence and the two outputs at its
end, or the event instances one program explores and the other does not.
The verdict is over the explored alphabet and, when a bound is reached, to
that bound; T59 bounds what agreement to a depth proves. A program and its
name-free projection (10.6) are equivalent by T7. JSON format
`rce-equiv/1`.

`rce change [--fuel N] OLD NEW` classifies a change by the rule of C.15 and
proposal CC-2: it prints both program hashes (equal hashes mean a renaming
or a reformatting, the same program), the `rce api-diff` rows (F.4.3) and
the behaviour check of `rce equiv`, and decides: *erratum* when every
export is compatible and every behaviour is kept (status 0; "to the
explored bound" when a bound was reached); *meaning change* when a
behaviour changed (status 1: a new release under C.15); *interface change*
when behaviour is kept but an export is broken (status 1: callers must
change); *compatible* or *breaking* from the rows alone when neither
program is a Form. The decision is the tool's, not a version number's.
`scripts/t7-equiv.sh` runs `rce equiv` between every corpus program and
its name-free projection (T7) and is a continuous-integration step.

**Read receipt.** Which JSON-RPC error code does the language server return for a request it does not know? (a number)

<!-- body-sha256: 3b0ee43df69ba8b03ffd972d23448003474fe2279a214fba3da33ede8f247970 -->
