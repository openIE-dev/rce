# Proposal: the emit target registry

Status: proposal. Owner: David Charlot. Derived from 10.5 (`rce emit`),
T9 (every target prints the reference trace), 6.3 (the numeric contract
every target must keep), the atlas of 2026-10-22 and the encapsulation
criterion (`encapsulation.md`).

In plain words:

1. A target is a language with a compiler or runtime that something needs
   to run on. RCE is designed to emit to every such language; the list below is
   the registry, and a language is to be added by adding a row, never by
   writing a new emitter.
2. Today each target is a hand-written emitter and a hand-written runtime
   (Rust, Python, JavaScript, Scala: 5,717 lines of generator with 445
   per-target branches and 8,094 lines of four runtimes that each rebuild
   the hash chain). That does not scale past four. So the registry is
   gated on one change first: a single emitter that walks the Form IR and
   reads a per-target table.
3. Every target is to pass the same gate: the reference trace of every example
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
anything the language leaves undefined (the table has no row for it, as
no name may be a reserved word under W-RES-1).

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
| Scala | JVM | int wrap (checked helpers), no `?T` (Option) | registered; emitted today, 1,459 conformance tests pass, 0 fail |
| XState, SCXML | statechart runtimes | control projection only (Form IR targets) | registered (T9-F) |
| Verilog (IEEE 1364-2005) | Icarus Verilog, Verilator; synthesis by Yosys | δ only, for the subset EV-1 of [emit-verilog.md](emit-verilog.md): fixed-width state in registers, one step per clock, loops unrolled to constant bounds, integers computed exactly and checked | first slice: `rce emit verilog`; simulated traces equal the reference on the programs of `tests/emit_verilog.rs` (CI: `.github/workflows/verilog.yml`); no board run |

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
| Verilog / SystemVerilog, VHDL | hardware; joules per step become real | one step per clock; bounded state only; the Esterel size ceiling applies (Verilog's first slice is in Tier 0: [emit-verilog.md](emit-verilog.md), with a ceiling of 20,000 unrolled statements) |
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

### Tier 5: the new entrants of 2024 to 2026 (registered 24 October 2026)

Every language first made public between January 2024 and October 2026
that the survey of 24 October 2026 catalogued
(`openie-research-experiments/rce/surveys/2026-10-24/`, working notes 5
and 6) is a row of the registry: an output of the Form by its print table
and an input by its lift, like every other row. A row here means
*registered as a target and a source*; the grammar is pinned and the
tables are written when the one-emitter change of section 4 lands, in the
order of section 4, and a language whose semantics the table cannot pin
to 6.3 is observation-only (section 3). A language that is a schema, a
prompt format or a tool rather than a language is marked so in the survey
and gets a schema row (Tier 3) instead. Mechanisms, status and the
self-reported claims of each entrant are in the survey notes; the
registry records only the name, the author, the first public date and the
canonical URL.

Languages for or by machine authors (72; 48 from the agentlanguages.dev
catalogue read at its source repository, 24 beyond it):

| Target | Author or organisation | First public | URL |
|---|---|---|---|
| AILANG | Mark Edmondson / Sunholo | 2025-09 | github.com/sunholo-data/ailang |
| Aver | jasisz | 2026-02 | github.com/jasisz/aver |
| Axis | Vladimir Melnic | 2026-05 | github.com/vmelnic/axis |
| B-IR / TBIR / Loom | Jason Hall (Chainguard) | 2026-01 | articles.imjasonh.com/llm-programming-language.md |
| BHC / hx | Raffael Schneider | 2026-04 | raskell.io |
| Boruna | escapeboy | 2026-04 | github.com/escapeboy/boruna |
| Canlang | Vince Liem | 2026-10 | sodaos.dev/canlang |
| Codex (damiant3) | Damian Tedrow "with a multi-agent AI team" | 2026-03 | github.com/damiant3/NewRepository |
| Codong | Brett (brettinhere) | 2026-03 | codong.org |
| Cohesive | Leo Gorodinski / Cohesive Systems | 2026-10 | github.com/cohesivesystems/cohesive |
| Faber Romanus | Ian Zepp | 2026-07 | faberlang.dev |
| Fabro | Bryan Helmkamp (Qlty) | 2026-03 | fabro.sh |
| Gooo | asketeddy (kimjooyoon) | 2026-08 | github.com/kimjooyoon/meta-ontology-go |
| Hale | Riley Rook | 2026-05 | hale-lang.org |
| ilo | Daniel Morris | 2026-02 | ilo-lang.ai |
| Intent | lhaig | 2026-02 | github.com/lhaig/intent |
| Koru | anonymous (korulang) | 2025-12 | korulang.org |
| Laze | kerv | 2026-04 | github.com/kerv/laze |
| LLMLang | Paul Williams | 2026-05 | github.com/paulprogrammer/llmlang |
| Lom | wytyKen "with LLM co-authorship throughout" | 2026-08 | github.com/lom-lang/lom |
| Lume | Marcelo Augusto Vilas Boas | 2026-05 | github.com/mavboas/lume |
| Lumen | alliecatowo | 2026-02 | github.com/alliecatowo/lumen |
| Modula-9 (M9) | Alex T. Vermeulen | 2026-08 | tutorial.modula9.net |
| Magpie | "Magpie Language Developers" | 2026-04 | magpie-lang.com |
| Marsha | David Ellis (Alan Technologies) | 2023-07 | github.com/alantech/marsha |
| Mog | Voltropy | 2026-03 | moglang.org |
| MoonBit | Hongbo Zhang / IDEA Shenzhen | 2023-01 | moonbitlang.com |
| Muro | Ricardo Corral-Corral | 2026-09 | muro-lang.dev |
| NanoLang | Jordan Hubbard | 2025-09 | jordanhubbard.github.io/nanolang |
| NERD | Guru Sattanathan | 2026-01 | nerd-lang.org |
| Pact | Viktor Kikot | 2026-04 | github.com/KikotVit/pact-lang |
| Pel | Behnam Mohammadi (CMU) | 2025-04/05 | arxiv.org/abs/2505.13453 |
| Plasm | Ryan Roberts | 2026-04 | plasm.tools |
| Plumbing | William Waites / Leith Document Company | 2026-03 | johncarlosbaez.wordpress.com/2026/03/11/… |
| Prove | Magnus Knutas | 2026-02 | prove.botwork.se |
| Quasar | Stephen Mell et al. (Penn) | 2025-06 | arxiv.org/abs/2506.12202 |
| reqlan | Tony Cerqui | 2026-06 | reqlan.com |
| SEMAPRAX | Wavect GmbH | 2026-08 | wavect.io/semaprax |
| Sever | Avital Tamir | 2026-02 per catalogue | github.com/AvitalTamir/sever |
| Spec | M. Abdullah Onus | 2026-04 | github.com/mronus/spec |
| Tacit | weetster | 2026-04 | github.com/weetster/tacit |
| Thermite | dollspace-gay and maxinelevesque | 2026-06 | github.com/dollspace-gay/Thermite |
| Valea | Hans Voetsch | 2026-03 | github.com/hvoetsch/valea |
| Vera | Alasdair Allan (also catalogue maintainer) | 2026-02 | veralang.dev |
| Vow | Paulo Matos | 2026-02 | vow-lang.com |
| With | Eric Hartford | 2026-02 | withlang.org |
| X07 | author unknown | 2026-04 | x07lang.org |
| Zero | Chris Tate & Matt Van Horn / Vercel Labs | 2026-05 | zerolang.ai |
| Goose | Wouter van Oortmerssen (design) + "Claude Fable" (implementation) | repo created 2026-08-30; HN threads 4974 | github.com/aardappel/goose |
| Rue | Steve Klabnik | repo 2025-06-05; public coverage Dec 202 | github.com/rue-language/rue |
| Claude's C Compiler (CCC) | Anthropic | repo 2026-02-04 | github.com/anthropics/claudes-c-compiler |
| Cutlet | s3thi | repo 2026-02-16 | github.com/s3thi/cutlet |
| Maxon | maxon-lang org (no named author) | repo 2025-11-13; releases 0.3.0–0.5.0 an | github.com/maxon-lang/maxon |
| Heroes | heroes-lang (single anonymous learner) | repo 2026-08-03; v0.3.0/0.4.0 | github.com/heroes-lang/heroes |
| Align | sanohiro | repo 2026-06-17 | github.com/sanohiro/align |
| Kōdo | rfunix | repo 2026-03-09 | github.com/rfunix/kodo |
| LMLang (LM) | rb3m | repo 2026-10-19 | github.com/rb3m/LMLang |
| Sigil | inerte | repo 2026-02-24; Show HN ~Apr 2026 | github.com/inerte/sigil |
| Vibe (vibec) | jvierine | repo 2026-09-10 | github.com/jvierine/vibe |
| AiLang (AiLangCore) | Todd Henderson / Magic Voodoo LLC | repo 2026-02-11 | github.com/AiLangCore/AiLang |
| Sui (粋) / Isu | Takato Honda | repo 2025-12-04; PyPI sui-lang 0.5.0 | github.com/TakatoHonda/sui-lang |
| lotml | protonspy | repo 2026-10-05 | github.com/protonspy/lotml |
| Lacon | FaulknerMassimo | repo 2026-10-06 | github.com/FaulknerMassimo/lacon |
| Nyxel | nyxel-lang | repo 2026-10-05 | github.com/nyxel-lang/nyxel |
| Notlob | Adam T. Burke (QUT) | repo 2026-05-19; arXiv 2608.24644 | github.com/adamburkegh/notlob |
| VIBEE | Dmitrii Vasilev | repo 2026-01-12; archived 2026-04-04 | github.com/gHashTag/vibee-lang |
| DriftScript | Seamus Brady | arXiv 2604.00043 | github.com/seamus-brady/DriftNARS |
| LACUNA | Zhao, Xu, Bračevac, Pham, Wu, Odersky (EPFL) | arXiv 2605.28617 | arxiv.org/abs/2605.28617 |
| Mirror | Austin Z. Henley | repo 2024-10-13 | github.com/AZHenley/Mirror |
| llm-lang | wilbowma (William Bowman) | repo 2024-05-09 | github.com/wilbowma/llm-lang |
| Microsoft POML | Microsoft | repo 2025-08-13; announced ~4 Sep 2025;  | github.com/microsoft/poml |
| Jac / byLLM (Jaseci) | Jaseci Labs | MTP paper arXiv 2405.08965 | github.com/jaseci-labs/jac |

General-purpose and systems languages first public in the window (39):

| Target | Author or organisation | First public | URL |
|---|---|---|---|
| Bend | HigherOrderCO | 2024-05 | github.com/HigherOrderCO/Bend |
| Goose | van Oortmerssen | 2026-10 | github.com/aardappel/goose |
| Valen | Ovadia | 2026 | github.com/valen-lang/valen |
| Micron | Keller | 2024-02 | github.com/rochus-keller/Micron |
| Luon | Keller | 2024-10 | github.com/rochus-keller/Luon |
| Par | Štrba | 2025 | github.com/par-team/par-lang |
| Vine | tjjfvi | 2025-02 | github.com/VineLang/vine |
| Fram | University of Wrocław | 2024-09 | github.com/fram-lang/dbl |
| Zeta | Murphy | 2025-11 | github.com/murphsicles/zeta |
| Cangjie | Huawei | 2024-06 | gitcode.com/Cangjie/cangjie_compiler |
| Pkl | Apple | 2024-02 | github.com/apple/pkl |
| Scrapscript | Troesh, Bernstein, Gregory | 2024-01 | github.com/tekknolagi/scrapscript |
| Amber | amber-lang | 2024-05 | github.com/amber-lang/amber |
| Kind2 | HigherOrderCO | 2024 | github.com/HigherOrderCO/Kind |
| Cursed | Huntley | 2025-09 | github.com/ghuntley/cursed |
| Rue | Klabnik | 2025-12 | github.com/rue-language/rue |
| Otterlang | jonathanmagambo | 2025-12 | github.com/Ottrlang/otterlang |
| Soppo | halcyonnouveau | 2025-12 | github.com/halcyonnouveau/soppo |
| Coi | io-eric | 2026-01 | github.com/coi/coi |
| Sky | anzellai | 2026-04 | github.com/anzellai/sky |
| Lisette | ivov | 2026-04 | github.com/ivov/lisette |
| Jolt | Sotnikov | 2026-07 | github.com/jolt-lang/jolt |
| Fun | omdxp | 2026 | github.com/omdxp/fun |
| Wyzer | Wyzer-Lang | 2026-08 | github.com/Wyzer-Lang/wyzer |
| Gossamer | Pozmanter | 2026-06 | github.com/gossamer-lang/gossamer |
| Han | xodn348 | 2026-03 | github.com/xodn348/han |
| Rux | Muzyka | 2026 | github.com/rux-lang/Rux |
| Kairo | Ze7111 | 2024+ | github.com/kairolang/kairo |
| Dawn | dawnop | 2024+ | github.com/dawnop/dawn-lang |
| Perk | Alex23087 | 2024+ | github.com/Alex23087/Perk |
| Plush | Chevalier-Boisvert | 2024+ | github.com/maximecb/plush |
| ClojureWasm | clojurewasm | 2024+ | github.com/clojurewasm/ClojureWasm |
| Hexa | dancinlab | 2024+ | github.com/dancinlab/hexa-lang |
| Gauntlet | gauntlet-lang | 2024+ | github.com/gauntlet-lang/gauntlet |
| Vix | vixlang | 2024+ | github.com/vixlang/Vix-lang |
| Cyrus | tahadostifam | 2024+ | github.com/cyrus-lang/Cyrus |
| Revo | if-not-nil | 2024+ | github.com/if-not-nil/revo |
| Aria | arialang | 2024+ | github.com/arialang/aria |
| Rad | amterp | 2024+ | github.com/amterp/rad |

Pre-2024 languages with a milestone in the window (Gleam 1.0, Koka 3,
Unison 1.0, Rhombus 1.0, Mojo 1.0, MoonBit, jank, Zig, Roc, Carbon, Hylo,
Inko, Austral, Flix, Verus, C3, Jakt, Onyx, Toit, Grain, Wing, Ballerina,
Jule, V, Lean 4, Vale, Effekt, Steel, Wa, Borgo, Nature) keep their rows in
Tiers 2 to 4; Hare, Hy, Pipefish, Odin, Rux and Nature are added to Tier 2.

What the entrants converge on, from the survey's tallies: validation of
input at the boundary (54 of 72 machine-author entrants) and
emission to another language (57 of 72; 17 of 39 general entrants) are
near-universal; a canonical form (25 of 72), examples inside the program
(21 of 72; 2 of 39), a written specification (16 of 72; 14 of 39),
deterministic evaluation as a stated gate (13 of 72; 3 of 39) and memory
safety without a collector (8 of 72; 7 of 39) are minorities; an
append-only log with replay (5 of 72; 0 of 39) and hash-identified
definitions (2 of 72; 3 of 39) are rare; an intermediate form as the
source of truth is absent from the new general-purpose languages. The
registry takes every entrant as a target and a source on the same terms,
and the rare mechanisms are the ones the Form adds on emission: a program
emitted into any of these languages carries the log, the hashes and the
determinism of the Form it was printed from.

## 3. What is not a target

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
