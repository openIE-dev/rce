## 10. Tooling: the `rce` command and the language server

This section defines the tools that read RCE programs: the `rce` command
and the language server it contains. It adds no rules to the language; it
says what each tool prints and returns for a given input. Example:
`rce run counter.rce "Inc, Dec, Dec"` prints the state after each event
and stops at the third event, whose step keeps the state, with exit status 1.

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
4. **Exit status.** 0 means success. 1 means a program requires change, an
   example or check failed, a run reached a kept step, or `fmt --check`
   found a file to change. 2 means a usage error: unknown subcommand, wrong
   number of arguments, unreadable file, or an unknown emit target or
   construct an emitter does not translate (10.5). 3 means a question the
   command could not decide within its bound: `rce equiv` and `rce change`
   when exploration reached `--fuel` before either equivalence or a
   distinguishing event sequence was established (10.11). 4 means a run
   the host cannot perform: `fn main` needs task threads and the host
   cannot start one (10.3).

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

All FILEs form one program: `use` resolves among them and then in the built-in standard library (4.1). A DIR is a
package: it stands for its `.rce` files, recursively, in sorted order,
and reads nothing outside it (the multi-module example is the package
`spec/examples/26_use_module/`, so `rce check spec/examples/26_use_module`
checks it alone). Each file is
parsed, type checked (section 5) and admitted (6.9: every `const`,
`comptime` block, `check` and `example` is evaluated). For each file, in
the order given, one line is printed:

```
ok FILE
requires change FILE: line N: RULE message[; line M: RULE message ...]
```

Every `requires change` report names its rule from section 7. Exit status 1 if any file
requires change, else 0.

### 10.3 `rce run FILE [EVENTS]`

The file must pass `rce check` alone; otherwise one `requires change` line is
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
regions separated by `; `, in declaration order. If the step for eₖ keeps the state, the line
is `k EVENT kept: ERROR`, no later event is applied, and the status is
1. Otherwise the status is 0. An event name that is not declared is a usage
of a wrong event list: one `requires change` line, status 1.

For the counter `spec/examples/01_counter.rce`:

```
$ rce run 01_counter.rce "Inc, Inc, Dec, Dec, Dec"
0 (initial) Counter { n = 0 }
1 Inc Counter { n = 1 }
2 Inc Counter { n = 2 }
3 Dec Counter { n = 1 }
4 Dec Counter { n = 0 }
5 Dec kept: Underflow
```

**Without EVENTS.** The module must declare `fn main()` with no
parameters. The command evaluates its body by the rules of section 6 and
prints `emit VALUE` for each value emitted, in order. If the body traps or
fails, the last line is `kept: REASON` and the status is 1. Calls to
foreign functions (5.11) are not executed by the reference evaluator and
trap. If there is no `fn main()`, the command says so and the status is 1.

The evaluator is the reference interpreter of the checker, with the fuel bound
of 6.7 and the VALUE_LIMIT of 6.3. It is a definition of meaning, not a fast
runtime. The body runs on the tick runtime of 6.8 and Appendix C, whose log
of the run is the trace of `fn main` in `rce trace` (10.9).

The reference tick runtime runs each task, `fn main` included, on its own
thread of the host. A host that cannot start a thread, such as WASI
preview 1 (`rce` built for `wasm32-wasip1`), cannot perform the run: the
command prints nothing on standard output and one line on standard error,
`FILE: fn main needs task threads, which this host does not provide
(HOST: REASON)`, HOST being the host's architecture and system and REASON
the host's error, and the status is 4. Whatever output the run had
produced before the failing start is not printed. A scheduler that runs
tasks as state machines, without threads, is the next slice of the
proposal `one-emitter.md` (section 13, 10.12); with it this run needs no
thread.

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

TARGET names an output. The file must pass `rce check` on its own; for
`core` it may instead be a core module, a JSON object whose `format` is
`rce-core/1`, which is read only if it is well formed (rules W1 to W6 of
the proposal `spec/proposals/one-emitter.md`, ONE-3).

| TARGET | Output |
|---|---|
| `form` | the Form P = (S, E, δ, D, λ) as one line of JSON (6.1), then a line end |
| `core` | the core module of FILE, format `rce-core/1` (the module lowered once, or the core module read), as one line of canonical JSON (6.1.1), then a line end (informative: the core is the proposal `one-emitter.md`, ONE-1) |
| `rust` | one Rust 2024 source file with a `main` |
| `python` | one Python 3 source file |
| `js` | one JavaScript module (ES2020 or later, BigInt) for Node |
| `scala` | one Scala 3 source file with a `@main` for the JVM (Java 19 or later) |

The Form JSON lists objects in canonical order (6.1), so it is a
deterministic function of the program text.

The four program targets are the ones this checker emits. The language
places no bound on targets, and it is designed so that every language is
both an output of the Form (a target, printed through a per-language
table) and an input (a lift, the reverse map, from the language's grammar
and a semantics table), each passing the same gate, the reference trace of
every example program byte for byte (T9). That design
(`spec/proposals/rce-form.md`, informative) is not built; of the four
program targets, `rust`, `python` and `js` pass the gate today. The registry of targets
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
and kept steps, and i128 arithmetic checked against each integer type's
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
it, its outputs, or the kept record), and finally the outputs and result of
`fn main`. These are the lines `emit::observe` computes with the reference
evaluator, so that a target and the reference can be compared line for
line.

**Embedding an emitted program.** An emitted file is also the step
function of the module, for a host to call (Appendix E: the host contract
is met around δ). Imported rather than run, it prints nothing: the Python
file prints its observations only under `__name__ == "__main__"`, and the
JavaScript module only when `node` runs that file itself; the JavaScript
module exports `initialState`, `delta`, `showState`, `show`, `construct`,
`fuelReset`, `chk`, `Kept`, the value classes `Int`, `F`, `Rec`, `Lst`,
`Mp` and `Opt`, `EVENT_FIELDS`, `formValue` and `fvJson`, and loads where
there is no `process` (a browser), where steps run and tasks and `par` do
not. The Rust file, used as a module (`#[path = "FILE"] mod m;`), makes
`initial_state`, `delta`, `show_state` and `event_fields` public. A host
calls the initial state once and `delta(state, event)` once per input,
after resetting the fuel (HC-7a); `delta` returns the next state and the
outputs, or reports the step as kept with the reason's display text.
`examples/interop/` holds such hosts (an asyncio and a FastAPI service in
Python, a tokio service in Rust, a `node:http` service and a React
component in JavaScript) and `examples/interop/check.py`, which requires
each host's transcript to equal `rce run FILE EVENTS` byte for byte.

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
not printed, 361 observation lines, and the edge program agrees on all four
targets. Its oracle is `RCE_SCALA` (default `scala-cli`). That agreement is for this examples corpus only. On the larger conformance suite (`conformance/`), a run on 2026-10-28 with Scala 3.10.0 on OpenJDK 21, which compared each trace by its head only, passes 1,459 tests with 0 failures, and 520 are not applicable to an emitter; the Scala emitter has not yet been run with the whole-trace comparison of the emit baseline (`conformance/README.md`).
The targets a test run covers are named as data,
one word per line, in `spec/checker/tests/emit_targets.txt` (`rust`,
`python`, `js`, `scala`), never by an environment variable or a flag. Every
test that runs emitted programs runs them on each covered target it uses
and fails when a covered target's toolchain does not start; a target the
file does not name is reported as not covered by the run, never counted as
a pass, and a test of that target alone is reported as ignored. The file
names `rust`, `python` and `js`, the toolchains of the continuous
integration runs; Scala is covered by adding its word. The
oracles used were Python 3.13.5 and Node 20.19.2.
The test reads the oracle paths from `RCE_RUSTC`, `RCE_PYTHON` and
`RCE_NODE` (default `rustc`, `python3`, `node` on the path) and prints each
oracle's version. On 2026-10-12 the same tests were also run on an Apple
M5 Max (macOS 27.0.1, arm64) with rustc 1.98.1, Python 3.14.5 and Node
26.11.1: 51 of 51 agree on each of the three targets, 0 not printed, 0
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

**The log D.** With the argument `--trace`, an emitted program also prints
the log D (6.1) of each example. Before the line of each event that δ
applied or kept comes its entry, `d applied HASH BODY` or `d kept HASH
BODY`, where BODY is the canonical JSON of `{event, prev_hash, seq,
state_hash}` over the positional forms and HASH is its SHA-256; after the
example's last event comes `trace HEAD`, the hash of the last entry (64
zeros when there is none). `emit::observe_trace` computes the same lines
with the reference evaluator, the fuel lines of `--fuel-trace` included.
The head commits to the events and the state hashes only. The state shown
after each event, the outputs, the reason an event was kept and the fuel
are not hashed, so a trace is compared whole: every line of an example's
block, the entries of D included. The emit adapters of `conformance/`
answer a trace with that block, and the harness compares it line for line
with the block `rce-conform` prints for the same program and events.

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
(a run that covered Scala alone).


**Running `fn main`.** Run with the argument `--run`, an emitted program
instead evaluates `fn main` with tasks, channels, `select`, actors and
`par` on a runtime of its own: every task is a thread (a worker in
JavaScript), and one scheduler admits their moves tick by tick in the
order of `conc::run_main`, the reference of `rce run` without EVENTS. It
prints `emit VALUE` per output and the line `outcome Done`,
`outcome Deadlock` or `outcome Stopped: REASON`. A module without
`fn main()` prints the reference's message ("nothing to run") on standard
error and exits with status 1, as `rce run` does. Fuel follows the reference's
pools: each task has its own, starting full (its first ticks are the call
of its function and one per argument); actor handlers share the
scheduler's pool; a `par` branch or `for ... par` chunk starts from its
parent's remaining fuel, and the branches are charged back in source
order, so the trap falls where the reference's sequential run puts it.
With `--trace` as well, the program prints the trace of `fn main` (10.9)
before the outcome line: per entry of the run's log, in order, the line
`d applied HASH BODY` (`d kept HASH BODY` for a `Kept` entry), BODY the
canonical JSON of `{event, prev_hash, seq, state_hash}` of that step and
HASH its SHA-256, then `trace HEAD`.

`spec/checker/tests/emit_run.rs` compares the three runtimes with
`conc::run_main` on a corpus of task programs. Among them are the four
concurrency examples 14, 15, 34 and 47, each run with a `fn main` from
`spec/checker/tests/drivers/` appended to its unchanged text. Its test
`main_traces_agree_on_every_target` compares the traces of `fn main` of
the three runtimes with the reference's, line for line, on 66 runs: that
corpus, the supervised actors of the same file, example 37, two session
programs of `spec/conc06b/accept/`, the partition cases and program
events, which together write an entry of every kind of 10.9.
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
  `rce emit` does not yet print the program: it exits with status 2 and names
  the construct. Every example program is printed.
- When a function whose result is a union fails, its `&mut` arguments are
  not written back. The reference evaluator writes them back.
- The Python output has been checked with Python 3.13.5 and 3.14.5, and
  the JavaScript output with Node 20.19.2 and 26.11.1. Other versions have
  not been run.
- The Scala output has been checked with Scala 3.10.0 only, on OpenJDK 27
  (the examples corpus) and OpenJDK 21 (the conformance suite). With
  `--run` it runs `fn main` through its task runtime, the same tick
  scheduler as the other targets, and prints that there is nothing to run
  when no `fn main()` is declared. A `par` runs its branches in source
  order on one thread (by restart when they wait on lvars), as the
  JavaScript output does when a local cannot go to a worker.


For an unknown target, or a program that one of these emitters does not
translate, `rce emit` prints nothing on standard output, prints the reason
on standard error, and exits with status 2. It never prints a partial file.

### 10.6 `rce fmt [--check] FILE...`

`rce fmt` rewrites each file into canonical layout. Let tok(t) be the token
sequence of text t produced by the lexer and layout pass of 2.6, with line
numbers dropped. The formatter is a function fmt on texts with:

1. **Meaning preserved.** tok(fmt(t)) = tok(t). The formatter checks this
   equality on every file before writing it; if it would fail, the file is
   left unchanged, a `requires change` line is printed and the status is 1.
2. **Canonical layout.** In fmt(t): a block that is k levels deep starts its
   lines with exactly 4k spaces; no line has trailing spaces; there are no
   blank lines before the first line, no two blank lines in a row, and the
   file ends with exactly one line end. A comment-only line takes the
   indentation of the next code line. A line that continues inside an open
   bracket keeps its position relative to the line that opened it.
3. **Idempotence.** fmt(fmt(t)) = fmt(t). This is tested on every program
   in `spec/examples/` and on the formatter's tests; it is not proved.

A text that requires change under the lexer (2.6, W-LEX-1, W-LEX-2) requires change
under `fmt` with the same reason. `fmt` does not type check: a file that requires
change under `rce check` can still be formatted. With `--check`, no file is written;
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
events and the named constants of 6.7. A use is renamed by its position
as well as its name. After `.`, a call is renamed only when it names a
method of the module's own `impl` blocks, since a call after `.` reaches a
function of the module in no other way (5.15): a standard library method,
or a function of another module, keeps its name even when the module
declares a function, field or local of that name. After `.`, a name that is
not called is renamed only as a field of the module's own types. A label
`NAME:` in parentheses is renamed only when the callee is the module's own
function, type, variant, event or error. A field name that a type of a
standard library module the program uses (directly or through that
module's own `use` lines) also declares keeps its text wherever it names a
field, since a receiver's type is not known to the projection; it is still
renamed where it names a local. Function calls by name are renamed at their
call sites. The projection is deterministic and
idempotent, and it has the same outcomes, outputs, ids and log hashes as
the source: `rce trace --json` of the two is equal step for step except
in the readable `event` field, since 6.1 hashes the positional forms
(T7). A name the
projection does not recognise as declared is left as written, so the
output is accepted whenever the input is for every example
(`tests/nameless.rs`), and for every one of the 1,876 files of the
repository that `rce check` admits on their own. The expressions inside a
template's `{…}` (F.6.1) are projected like code, and the authority structs
`DirRead` and `DirWrite` keep their names and their field `root`, which the
checker reads by name (F.5.1). It is the text of the concept record's
complement (6.6).

### 10.7 The language server: `rce lsp`

`rce lsp` speaks the Language Server Protocol 3.17 over standard input and
output (JSON-RPC 2.0 messages framed by `Content-Length` headers). It keeps
the full text of each open document; it does not read other files.

| Message | Behaviour |
|---|---|
| `initialize` | replies with capabilities: full-text sync (`textDocumentSync` = 1), formatting, hover, definition, document symbols, and completion triggered by `.` |
| `textDocument/didOpen`, `didChange` | stores the text and publishes diagnostics for that document |
| `textDocument/didClose` | forgets the text and publishes an empty diagnostic list |
| `textDocument/formatting` | returns one edit replacing the whole text with fmt(t) (10.6), or no edit if t is canonical or requires change |
| `textDocument/hover` | the type or declaration of the name at the position, or `null` |
| `textDocument/definition` | the location of that name's binding or item, or `null` |
| `textDocument/completion` | the names that can be written at the position |
| `textDocument/documentSymbol` | the document's items, each with its fields, variants or methods |
| `shutdown` | replies `null` |
| `exit` | ends the server: status 0 after `shutdown`, 1 without it |
| any other request | error -32601, method not found |

**Diagnostics.** For a document with text t, the published diagnostics are
exactly the reports `rce check` gives for t that the text requires change, one per
report, with severity Error and source `rce`. A report `line N: ...` covers line N − 1 in the
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
  unimplemented rules. `rce manifest DIR` writes the manifest of a directory (F.5.3: every
  `.rce` file as a module pinned by the hash of its canonical text, canonical
  JSON; a directory whose modules import the built-in library gets the one
  dependency `stdlib`, pinned by the hash of the library's manifest) and
  `rce manifest-check` verifies one. The standard library is pinned this way
  (`stdlib/lib/MANIFEST.json`), and the checker carries that manifest with
  the library. `rce lib` prints the library's hash after the modules, then
  each module whose text differs from its pin (status 1); while such a
  difference exists, `rce manifest` writes no `stdlib` pin and
  `rce manifest-check` accepts none. The broker contract (F.7), the provenance
  record and the build cache of F.5.3, and the emitter messages of F.5.4 for a
  construct not yet printed do not exist yet. Results are
  reproducible from the inputs alone (10.1).

### 10.9 `rce trace --json FILE [EVENTS]`: reference traces (informative)

The trace format lets an independent projector (another compiler or
runtime of the same Form) show that it agrees with the reference
evaluator step by step. It adds no meaning: every field is a field of the
log D of 6.1, so a trace is the log, and conformance to 6.1 implies
conformance to the trace.

`rce trace --json FILE` prints one trace per `example` of the module, in
source order (the same logs `rce emit form` puts in `logs`), then, when
the module declares `fn main()`, the trace of its run (below); with
`--streams PATH` it prints one trace per event list of PATH (one list per
line, in the syntax of an `example` line; a line starting with `#` is
skipped), named `stream@k`, in one run of the checker, with `streams`
carrying their number; with EVENTS
(the syntax of an `example` line, as in 10.3) it prints one trace of that
event list. A module that requires change, or a trace of EVENTS that yields no
step, is status 1. The output is one JSON object (format `rce-trace/2`;
`rce-trace/1` differed in hashing the named forms, before the T7
resolution of C.15), compact, with sorted keys:

| Field | Meaning |
|---|---|
| `format` | `"rce-trace/2"` |
| `file` | the file name, without directories |
| `events` | the EVENTS argument, or null |
| `genesis` | the first `prev_hash`: 64 zeros (6.1) |
| `traces` | list of {`source`, `steps`}; `source` is `"example@L"` (the example's line), `"events"` or `"main"`; a `"main"` trace also has `outcome` (below) |
| `steps[i].seq` | required: 0-based index in the trace |
| `steps[i].event` | required: {`type`, `payload`}, payload values as in 6.1 |
| `steps[i].state_hash` | required: SHA-256 (lower-case hex) of the canonical JSON of the positional form of the state after the step (6.1) |
| `steps[i].prev_hash`, `steps[i].hash` | the 6.1 chain: `hash` is the SHA-256 of the canonical JSON of {event, prev_hash, seq, state_hash} with the positional form of the event |
| `steps[i].outcome` | `"applied"` or {`kept`: error name}; derived, not hashed |
| `steps[i].hashed` | {`event`, `state`}: the positional forms the two hashes are over (6.1), so a reader recomputes both without the module; derived |

A kept step is the last step of its trace, with the unchanged state's
hash (T8). A projector conforms on a module when, for every trace, its
steps have the same `seq`, `event` and `state_hash` as the reference, in
order and with none missing or extra; when it also emits `prev_hash`,
`hash` or `outcome` they must be equal too. Since `state_hash` covers
the whole state, equal hashes mean equal states without exchanging them.

**The trace of `fn main`.** Its run is the one `rce run FILE` performs
without EVENTS (10.3): `fn main()` on the tick runtime of 6.8 and
Appendix C, with no link, event or restart input and the default seed of
`select nondet`. The scheduler writes a log of the run (C.11.4), one entry
per action, and a step of the trace is one entry, in the order written.
Within a tick the order is the scheduler's: link inputs, then program
events, then the actors (transactions, then one message per actor in id
order), then each task's pending operation in task order (higher priority
first, then id). Every choice in that order is a function of ids,
payloads and per-`select` counters (6.8), so the trace is a function of
the program, as the run is (T1), however the tasks are placed on threads.
The text of an entry is `K` or `K(A)`, A showing values as `emit` lines
do (10.3):

| Entry | Written when |
|---|---|
| `Spawn(task ID, priority P)` | a spawn is admitted, before the new task first moves |
| `Done(task ID)` | a task's function returns, traps, fails or is cancelled |
| `Emit(V)` | a task emits V, or a committed handler emitted V |
| `Send(V)` | a send of V is admitted: to a channel, to an actor, by a `select` arm, inside `atomic`, or by a committed handler |
| `Backpressure` | the channel of a send is full: the sender is held (6.8) |
| `Recv(V)`, `Recv(none)` | a receive or a `select` arm takes V, or finds the channel closed and empty |
| `Close` | a channel is closed |
| `Atomic` | the sends of an `atomic` block are admitted together (C.11.10) |
| `Choice(task ID, select line L, execution K, arm A)` | a `select nondet` takes arm A (C.5) |
| `Cancelled(task ID)` | a deadline of the task has passed at its communication step (C.6.2) |
| `Deliver(V)` | an actor's handler takes the message V |
| `Kept(R)` | an actor's handler kept its state for reason R, or another handler of the same transaction was not committed (T30) |
| `Restart(CHILD, ID, R, tick T, K)`, `Stopped(supervisor S)` | a supervisor restarts a child or stops (C.11.11) |
| `Input(V)`, `Unhandled(V)` | a program event arrives, and no actor handles it (HC-6) |
| `Partition(A, B)`, `Heal(A, B)`, `Held(A -> B)` | a link input, and a message held at a partitioned link (C.11.4) |

The last two rows need inputs of the run, so they occur only in a trace
of a run given those inputs (`spec/checker/tests/emit_run.rs` gives them).
A step's `event` is {`type`: K, `payload`: A}, with payload null for an
entry `K`; its state is `[tick, actors]`: the tick at which the entry was
written (the first tick is 1) and, for every live actor in id order,
`[ID, [v₁, …]]`, its field values in declaration order in the positional
form of 6.1. `hashed` holds the two, `state_hash` is the SHA-256 of the
canonical JSON of the state, and `prev_hash` and `hash` chain from the
genesis as an example's entries do. `outcome` is {`kept`: R} for a step
`Kept(R)` and `"applied"` for every other step; unlike an example's, a
kept step is not the last: the trace ends where the run ends. The
trace's `outcome` is `"done"`, `"deadlock"` or {`stopped`: REASON}, as
`rce run` ends. An entry's event is display text, which names declared
types (`Ping(n: 7, reply: <chan>)`), so renaming a declaration changes the
hashes of a main trace, where an example's (T7) stay the same. With
`--trace`, an emitted program run with `--run` prints the trace of
`fn main` as lines of D (10.5). The core evaluator (10.12) does not run
`fn main` on the tick runtime, so it prints no trace of `fn main`.

On a host that cannot start a task thread (10.3, WASI preview 1) the run
does not happen, and the trace of `fn main` is absent: it is
`{"source": "main", "outcome": {"absent": REASON}, "steps": []}`, REASON
being the line of 10.3 after `FILE: `. The other traces are printed as on
any host, standard error has one line `FILE: no trace of fn main:
REASON`, and the status is 4. A projector conforms on such a module only
on a host that runs `fn main`; the cross-architecture parity job
(`.github/workflows/xarch-parity.yml`) lists the five examples with
`fn main()` as not compared on `wasm32-wasip1`, for this reason.

`spec/traces/` holds the reference traces of the example corpus: one
file per top-level example (the package `26_use_module/` has no state
machine), plus two event-list traces, `01_counter.events.json` (the
counter on a longer list) and `16_multi_region.events.json` (two regions
stepped by one event, which has no `example` lines). Five examples
declare `fn main()`: 02, 13 and 37 run to `done`, and 40 and 48 stop at
their first foreign call, which the reference evaluator does not execute
(10.3). Examples without a state machine or `fn main()` have an empty
`traces` list. `spec/traces/rce-trace-diff.py REFERENCE OTHER` (Python 3,
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
A `use` item names a module given with the files or one of the built-in
library (4.1), and a file is listed as requiring change in the catalog exactly when
`rce check` reports that it requires change.
An entry holds `module`, `name`, `file`, `line`, `generics`, `params`
(name and type), `ret`, `key` (the type with generic parameters renamed
`t0`, `t1`, ... in order of first occurrence), `hash` (SHA-256 of the
module path and the canonical text of the function alone, so layout and
position do not change it), `checked` (the verdict of its file) and
`lowering` (`{"term": T}`, the D.7 lowering, or `{"kept": reason}`).
The catalog's `hash` is the SHA-256 of the canonical JSON of its entries;
`pending_files` maps each file that requires change to its diagnostics. With
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
outcome only (the kept step's error, by declaration index; else the
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
emitted value, or the record of a kept step (`has_output` in the JSON, with
`emits` and `keeps` reported separately, since a kept step and an emitted
value are two kinds of the program's answer: a kept step leaves the state as
it was and records why, an emitted value is an output), and how many
distinct output sequences the explored steps carry (`distinct_outputs`; a
silent step is excluded). A program with no such step has one behaviour by
definition, so its redundancy is its whole state: the state is in D and
readable, and an `example` may check it, but no future output depends on
it. A program whose every output is the same has one behaviour as well.
The dimension benchmark reports these strata separately.
`--json` prints the same as one JSON object (format `rce-dim/1`, compact,
sorted keys; `has_output`, `emits`, `keeps` and `distinct_outputs` since 2026-10-24, replacing `observable` and `distinct_labels`). The command is deterministic: equal inputs give equal output
bytes, and the exploration order is the canonical order of 6.1 on events
and payloads. It is informative: no admission depends on it.

`rce equiv [--fuel N] [--json] A B` asks whether two programs have the same
behaviour (6.10): it explores both from their initial states over the same
event instances, compares every step by its name-free output (emitted
values in their positional form, a kept step by its error's declaration
index, T7), refines the two graphs together by those outputs and reads
whether the two initial states fall in one class (T58: equivalence is
bisimilarity). Status 0 and `equivalent` when they do; status 1 and
`different` with the shortest distinguishing event sequence and the two
outputs at its end, or the event instances one program explores and the
other does not. When an exploration reached its bound and the refinement
separates the initial states but no distinguishing sequence exists within
the explored graph, the separation may come from the unexplored frontier,
so the verdict is `undecided` with status 3, never `different`.
The verdict is over the explored alphabet and, when a bound is reached, to
that bound; T59 bounds what agreement to a depth proves. A program and its
name-free projection (10.6) are equivalent by T7. JSON format
`rce-equiv/1`.

`rce change [--fuel N] OLD NEW` classifies a change to a program by the
classes of C.15 and proposal CC-2, comparing the two programs under one
text: it prints both program hashes (equal hashes mean a renaming
or a reformatting, the same program), the `rce api-diff` rows (F.4.3) and
the behaviour check of `rce equiv`, and decides: *erratum* when every
export is compatible and every behaviour is kept (status 0; "to the
explored bound" when a bound was reached); *meaning change* when a
behaviour changed (status 1: the program's meaning changed); *interface change*
when behaviour is kept but an export is broken (status 1: callers must
change); *undecided* when `rce equiv` is undecided (status 3: raise
`--fuel`); *compatible* or *breaking* from the rows alone when neither
program is a Form. The decision is the tool's, not a version number's.
`scripts/t7-equiv.sh` runs `rce equiv` between every corpus program and
its name-free projection (T7) and is a continuous-integration step.

### 10.12 `rce run-core FILE`: the core evaluator (informative)

`rce run-core FILE` lowers FILE to its core module, as `rce emit core`
does (FILE is an RCE module or an `rce-core/1` core module, read only if
it is well formed), runs that module with the core evaluator of the
proposal `one-emitter.md` (ONE-1), and prints what the run writes. The
core evaluator is an implementation under test, compared with the
reference evaluator like any target: the expected bytes of every
comparison are the reference trace (10.9), never this output. Exit status
0 when the run ends; 1 when the checker reports that FILE requires change, when a core module
is not well formed, or when the run halts; 2 when FILE uses a construct
the lowering does not cover, or on an internal error.

The core evaluator observes `fn main` as one call of its body, as the
`main` block of the proposal's trace (ONE-5) does, not as a run on the
tick runtime: the lowering covers no channel, `spawn`, `select`, actor or
supervisor (status 2), and the core has no scheduler, so `rce run-core`
prints no trace of `fn main` (10.9). The next slice of the proposal is a
scheduler in the core that runs tasks as state machines, without threads
of the host; the reference's run of `fn main` needs a thread per task, so
on a host without threads (WASI preview 1) it is absent (10.3, 10.9), and
that slice is what makes the run possible there.

**Read receipt.** Which JSON-RPC error code does the language server return for a request it does not know? (a number)

<!-- body-sha256: 6068f55aa66d7e498cad66bd807e7478ef6eb18e14223b3d0e6b9ecf82f673a6 -->
