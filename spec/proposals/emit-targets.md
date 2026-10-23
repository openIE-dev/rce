# Proposal: the emit target registry

Status: proposal. Owner: David Charlot. Derived from 10.5 (`rce emit`),
T9 (every target prints the reference trace), 6.3 (the numeric contract
every target must keep), the atlas of 2026-10-22 and the encapsulation
criterion (`encapsulation.md`).

In plain words:

1. A target is a language with a compiler or runtime that something needs
   to run on. RCE emits to every such language; the list below is the
   registry, and a language is added by adding a row, never by writing
   a new emitter.
2. Today each target is a hand-written emitter and a hand-written runtime
   (Rust, Python, JavaScript, Scala: 5,717 lines of generator with 445
   per-target branches and 8,094 lines of four runtimes that each rebuild
   the hash chain). That does not scale past four. So the registry is
   gated on one change first: a single emitter that walks the Form IR and
   reads a per-target table.
3. Every target passes the same gate: the reference trace of every example
   program, byte for byte (T9), plus the lens law where a lift exists.

## 0. The deterministic interface: three artefacts per language

A language enters the registry as data, not code. For a language L:

| Artefact | From | Gives | Deterministic because |
|---|---|---|---|
| `lang/L/grammar` | the language's tree-sitter grammar, vendored and pinned by commit (about 150 exist) | text → concrete syntax tree | GLR with fixed conflict resolution; the same bytes give the same tree |
| `lang/L/semantics.toml` | the language's own specification | node kind → Form construct; the edge rules of 6.3 as the language fixes them (integer width and overflow, division sign, float mode, string unit, map order, whether coroutines are stackless) | a table is looked up, never inferred |
| `lang/L/print.toml` | the same grammar, read backwards | Form IR node → syntax-tree template → text | printing is a function of the IR |

Lift λ_L = grammar + semantics table. Emit π_L = print table + grammar.
The gate on the pair is the lens law, hash(λ_L(π_L(F))) = hash(F), checked
on the corpus, plus T9. What tree-sitter does not give is also stated:
name resolution and types (the Language Server Protocol gives those where
a server exists; otherwise the semantics table carries scope rules), macro
and comptime expansion (Rust, Zig, C: the tree is not the program until
these run, so λ_L for them is on the expanded tree), and semantics of
anything the language leaves undefined (the table refuses it, as W-RES-1
refuses a reserved word).

Probe (2026-10-23, `spec/lift/go`): a Go struct with one method per event,
lifted by `tree-sitter-go` and a 12-line semantics table with no model,
checks, runs, has 3 behaviours and 3 control states, and has the same
`shape_ctl` as the hand-written `61_dim_minimal.rce`. The probe found one
defect in the control label (items indexed across kinds) that the
hand-written corpus alone would not have shown.

## 1. The mechanism before the list

**One emitter.** `rce emit TARGET` lowers the module once (the ANF pass of
T34 extended to the whole language, the IR of `form-ir-2.md`) and prints
it through a table `emit/targets/<name>.toml` with, per target:

| Table field | What it fixes | Example (Zig) |
|---|---|---|
| `int.checked` | how `+ - *` trap on overflow (native, helper, or wrap-and-check) | native (`@addWithOverflow`) |
| `int.div`, `int.rem` | truncating division and dividend-sign remainder | `@divTrunc`, `@rem` |
| `float.strict` | one rounding per op, no contraction, no x87 | `@setFloatMode(.strict)` |
| `text.unit` | UTF-8 bytes (6.3 N7) | native |
| `map.order` | canonical key order on iteration | sort on write |
| `option` | `T?` | `?T` |
| `error` | `T ! E` | error union |
| `sha256`, `canon_json` | the chain of 6.1 | runtime kernel |
| `spawn` | stackless lowering only (atlas: stackful cannot be logged per step) | state machine |
| `par` | structured step (never a product automaton) | threads with ordered merge |
| `cost.regime` | A (count-linear) or B (time-linear), `energy-calibration.md` | A on MCU, B on desktop |

**One runtime kernel per target**, ~300 lines, generated from the same
table where the language allows and hand-written where it does not: the
canonical JSON writer, SHA-256, the fold, the scheduler of 6.8. The 77
canonical-JSON vectors (`spec/data/canonical-json-vectors.json`) are the
kernel's conformance test on every target.

**The gate per target.** (a) `tests/emit.rs`: every example program's
observations equal the reference interpreter's; (b) the 77 vectors; (c)
the T9 trace on the 529-event edge program; (d) where a lift λ_L exists,
hash(λ_L(π_L(F))) = hash(F) on the corpus; (e) a `measured_j` row on at
least one host for the encapsulation criterion. A target is *registered*
at (a)–(c), *encapsulated* at (d)–(e).

## 2. The registry

Tier = order of work. Within a tier, by lift cost (cheapest AST first).
"Leaks" names the coordinate the target cannot keep natively and the table
must repair (6.3 and the atlas, 2026-10-22).

### Tier 0: registered today

| Target | Runs on | Leaks repaired by the table | Status |
|---|---|---|---|
| Rust | native, WASM (wasm32-wasip1) | none for the contract; `Pin` exposed by async lowering | registered; MiniRust proof on the core (T9-L) |
| Python | CPython, PyPy | ints unbounded (range checks), floats via `float`, no overflow trap | registered |
| JavaScript | V8, Node, browsers | i64 needs BigInt, no integer type, map order insertion | registered |
| Scala | JVM | int wrap (checked helpers), no `?T` (Option) | registered |
| XState, SCXML | statechart runtimes | control projection only (Form IR targets) | registered (T9-F) |

### Tier 1: the systems and platform languages (every device class)

| Target | Why | Leaks | Lift cost |
|---|---|---|---|
| C | every platform, embedded, FFI floor | UB on overflow and shifts; `int` width; no bounds | low AST, UB semantics hard |
| Zig | native checked arithmetic, cross-compilation to every triple, WASM; `zig cc` for the C target | build-mode semantics (pin ReleaseSafe) | comptime must be evaluated first |
| Go | servers, CLIs, cheapest lift | int wrap; nil/zero conflation; map order random | lowest |
| TypeScript | the typed web | same as JS plus a type layer to emit | low after type erasure |
| Java | JVM, Android | int wrap; no unsigned | low |
| Kotlin | Android, JVM, native | as Java; coroutines are stackless (good) | low |
| Swift | iOS, macOS | int traps natively (good); value types (good); actors differ from 6.8 | medium |
| C# | .NET, Unity | `checked` context available; decimal separate | low |
| Fortran | HPC, numerics; `pure`/`intent` give the aliasing freedom 6.3 wants | int wrap; FP flags per compiler | medium (flang AST) |

### Tier 2: the mainstream and the dynamic lineage

| Target | Why | Leaks |
|---|---|---|
| C++ | engines, HPC, legacy | as C; coroutines stackless (good) |
| Ruby | web, scripting | as Python |
| PHP | web | as Python; int 64 wraps to float |
| Lua | embedded scripting, games | 5.3+ ints wrap; one number type before 5.3 |
| Julia | scientific | ints wrap; floats good |
| Dart | Flutter | web ints 53-bit |
| Perl | scripting | numbers untyped |
| R | statistics | numbers are doubles |
| Elixir / Erlang | telecom, actors | ints unbounded; actors differ from 6.8 (library Form) |
| OCaml | compilers, finance | 63-bit ints |
| Haskell | correctness-first shops | laziness (strict lowering), ints unbounded or wrap |
| F# | .NET functional | as C# |
| Elm | web, the architecture closest to the Form | no i64, no mutation (fine) |
| Gleam | BEAM typed | as Elixir |
| Nim, D, Odin, Hare, Crystal, V | systems, each with users | as C with per-language int rules |
| Clojure, Scheme (R7RS), Racket, Common Lisp | the homoiconic lineage; Racket for the web-server continuation lesson | numbers unbounded; order via sorted maps |
| Smalltalk (Pharo) | image systems | numbers unbounded |
| Mojo | Modular/Qualcomm accelerators | ownership by convention, as Rust |
| Objective-C | Apple legacy | as C |
| Pascal / Object Pascal (Free Pascal, Delphi) | legacy, education | range-checked ints available |
| Ada / SPARK | certified systems; contracts always on (the one language that already does F.6) | none for ints (ranged types native) |
| COBOL | the installed base of finance | decimal arithmetic; no map |
| Visual Basic (.NET) | installed base | as C# |
| Groovy | JVM scripting | as Java |
| PowerShell, Bash / POSIX sh | automation; `Shell` typed output already exists (5.16) | numbers; the Form runs as a step per invocation |
| Tcl | tooling | everything is text (6.3 is the schema) |
| Assembly (x86-64, AArch64, RISC-V) | the floor; via the C or Zig target, not hand emission | none: it is the machine |

### Tier 3: the non-von-Neumann lineages (the closest charts to the Form)

| Target | Why | Note |
|---|---|---|
| SQL (SQLite, PostgreSQL, DuckDB) | the log is the database; `rce db emit` already prints these | a Form as tables, δ as triggers or an incremental view (DBSP) |
| Datalog (Soufflé), Dedalus-style | the nearest chart to the engine profile | monotone rules direct; non-monotone via stratification |
| Cypher / Gremlin | graph stores (`rce db emit cypher` exists) | edges relation |
| Verilog / SystemVerilog, VHDL | hardware; joules per step become real | one step per clock; bounded state only; the Esterel size ceiling applies |
| Bluespec, Chisel, SpinalHDL, Amaranth, Clash | guarded atomic actions = `step … on … if` | the natural hardware chart |
| Lustre / SCADE, Esterel | certified synchronous systems | ticks are instants; no allocation (matches 6.7) |
| Simulink / Stateflow | control engineering | statechart projection |
| TLA+ / PlusCal, Alloy, Event-B, P | model checkers; the spec of the spec | δ as the next-state relation; examples as traces |
| Lean 4, Dafny, Verus, F\*, Coq/Rocq, Agda, Isabelle | proof targets: emit the Form as a definition and its examples as theorems | the T9 gate becomes a `decide` |
| Prolog, miniKanren | logic engines | δ as clauses; the log as facts |
| APL, J, K/q | array engines, finance | lane-wise arrays only; one glyph per op |
| Futhark, Halide, Triton, CUDA, OpenCL, GLSL/WGSL/HLSL/MSL, SPIR-V | accelerators and shaders | device regions (5.13) only: fixed-size, effect-free kernels |
| Forth, PostScript, Factor | stack machines, printers | the step as a word |
| Excel / Google Sheets formulas, LabVIEW | the most used dataflow languages | a region as a sheet; a step as a recalculation |
| WebAssembly (direct, component model) | everything that runs WASM; canonical ABI lifts/lowers exactly as 6.3 needs | already via Rust; direct emission removes rustc from the path |
| LLVM IR, MLIR dialects | compiler infrastructure | the "atlas of dialects", one dialect per Form layer |
| BPF / eBPF | kernel probes | bounded loops only (matches 6.7 exactly) |
| Solidity / Move / Vyper | ledgers with gas (C.6 units map to gas) | checked arithmetic native in 0.8+ |
| Regex, JSON Schema, Protocol Buffers, Cap'n Proto | not programs: schemas; emitted from `E` and the contracts of T35 | the payload contract as a schema |

### Tier 4: registered for completeness, no platform of their own

Languages whose every substrate is already covered by a tier above, or
with no maintained compiler: ALGOL 60/68, Modula-2/3, Oberon, Simula,
BCPL, PL/I, Dylan, Self, Io, Miranda, Clean, Idris, ATS, Koka, Effekt,
Roc, Unison, Verse, Carbon, Hylo, Vale, Austral, Cyclone, Jai, Nickel,
CUE, Dhall, Pkl, Scratch, Logo, BASIC. Each gets a table row when someone
needs it; none blocks the encapsulation criterion.

## 3. What is refused

- A target whose semantics the table cannot pin to 6.3 (an interpreter with
  unspecified float formatting and no way to call a fixed one) is
  registered as *observation-only*: it may print the trace for T9 but is
  not a projection π_L.
- Hand-written emitters after the one-emitter change. A new target is a
  table and a kernel, with the gate of section 1.
- Any target-specific semantics. The Form means one thing; a target that
  cannot mean that thing is not a target.

## 4. Order of work

1. One emitter over the Form IR with the per-target table; the four
   existing targets re-expressed as tables; `tests/emit.rs` unchanged and
   green. This is roadmap phase 1 and the only prerequisite.
2. Tier 1 in lift order: Go, C, Zig, TypeScript, Java, Kotlin, Swift, C#,
   Fortran. Each lands with (a)–(c); the lift and the joule row follow.
3. Tier 2 by demand; Tier 3 where a platform needs it (SQL, Verilog and
   Lean first, because they are the log, the joule and the proof).
4. The registry file `emit/targets/REGISTRY.md` generated from the tables,
   with the gate status per target, counted by `scripts/claim-counts.py`.
