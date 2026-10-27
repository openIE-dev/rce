# Proposal: `rce-form/1`, the one form that lift and emit share

Status: proposal. Owner: David Charlot. Derived from 6.1 (the Form of a program) and 6.1.1 (the canonical JavaScript Object Notation bytes), 6.2 to 6.9 (evaluation, the edge rules of primitives, the step function δ, kept events, name-free hashing, loops and fuel, concurrency, examples and checks), 4.2 (the program's Form), 5.1, 5.2, 5.11 and 5.14, section 7 (well-formedness), 10.1 and 10.5, C.5, C.6 and C.15, D.8, Appendix E (HC-7a, HC-8), the theorems T2, T7, T8, T9 and T37, CONTRIBUTING rules 1 to 3, and three proposals: `one-emitter.md` (the core `rce-core/1`, the print table and its gate, ONE-1 to ONE-27), `compliance.md` (the outcome words, and how the Form holds what is offered to it) and `emit-targets.md`. Nothing in this proposal has been built. No normative chapter changes with it.

Sources, all read only:

- `openIE-dev/rce` at `origin/main` ace40713. No file under `spec/checker` differs between d91a9fc8 and ace40713, so a citation of the checker holds at both.
- The branch `claude/one-emitter` of this repository at 123f5eee ("the branch"), which holds the core of `one-emitter.md`, its well-formedness check, its lowering and its evaluator under `spec/checker/src/one/`, with the differential tests `tests/one_lower.rs` and `tests/one_conformance.rs`. A path written `one/…` is a file of that directory on the branch.
- flowG's public kernel vocabulary (`openIE-dev/flow-g`): 22 primitives in 6 categories and 11 wire kinds, with their names and kernel numbers.

In plain words:

1. The Form of RCE (the Reality Code Engine), P = (S, E, δ, D, λ) of 6.1 (states, events, a total step function, a hash-chained log, outputs), stays the meaning of every program, unchanged. This proposal fixes one encoding of it, `rce-form/1`. It is the only thing that is hashed: a frame of tables for the state, the events, the steps, the errors and the declared types, and content-addressed definitions whose bodies are graphs over five node kinds.
2. The five node kinds keep flowG's kernel numbers: Bind 0, Apply 1, Branch 3, Iterate 4 and Halt 13 (flowG's ErrorPropagate). Wires carry no kind. Each of flowG's wire kinds is derived from a value's type and the port it enters, so no wire can be labelled wrongly.
3. One owned value, the order value, threads every effect and every halt. It holds the fuel, the cost counters, the outputs gathered so far and the cells that hold evaluated constants, for one input on a host or one observation in a trace. It never enters the state, the log or the outputs, so nothing outlives an input except the state.
4. Every other notation is a view of the encoding. The core `rce-core/1` of `one-emitter.md` is the schedule view that printers read; RCE text is the view RCE's own print table prints; the flowG view restores all 22 kernel kinds and all 11 wire kinds and has an inverse, so an edit drawn on a graph is a candidate program that the gate applies or keeps.
5. A language is data: a pinned grammar, a lift table, a print table and a runtime library written in RCE. Nothing is approximated. What a table cannot carry exactly is kept, with its reason.
6. Acceptance is by definition and exact comparison. A lifted program's observations equal, byte for byte, the observations of the language's own implementation run under its pins; an emitted program prints the reference trace byte for byte.
7. Where it stands: no line of the form layer exists. The core, its lowering and its evaluator exist on the branch, where the lowering reproduces the reference interpreter's trace on 56 of the 60 single-file examples of `spec/examples` (the other 4 are kept, each naming the construct it stops at) and on at least 1,435 conformance tests.

## 1. Terms

**Conventions.**

- A bare section number such as 6.1, C.6 or E.2 names a section of the specification (the chapters of `spec/book/`, generated into `spec/RCE-SPEC.md`). Rules and theorems of the specification keep their names: T2, W-TY-2, HC-7a, N9, F1 to F9, and E1, the rule of 6.2 that fuel exhaustion cannot be caught.
- §n names section n of this proposal. A reference to a later section says "below". The sections are: §1 Terms; §2 The decision; §3 Types; §4 The catalogue; §5 The order value; §6 Nodes; §7 Domain facts; §8 Port modes, linearity and wire kinds; §9 The frame; §10 Encoding and identity; §11 The gate; §12 flowG's kinds and the view; §13 The schedule and the construction; §14 Closure; §15 What is closed and what is open; §16 Lift; §17 The Lift IR and stage B; §18 The lift report; §19 Emit; §20 Acceptance; §21 The toolchain; §22 What admission needs; §23 Changes if accepted; §24 What stays open.
- ONE-n names a rule of `one-emitter.md`, and W1 to W6 its well-formedness rules of the core (ONE-3).
- Label families this proposal introduces, each numbered in the order a reader meets it and defined in the section named, below: domain facts DF1 to DF8 (§7), canonical rules C1 to C11 (§10), gate rules FG1 to FG15 (§11), the transforms of stage B, B1 to B7 (§17), and corpus rules K1 to K7 (§20).

**Abbreviations.** Each is expanded here, before its first use.

| Abbreviation | Expansion |
|---|---|
| ANF | A-normal form: every operand of an operation is a variable or a literal |
| CI | continuous integration: the workflows of `.github/workflows/` |
| CPython | the reference implementation of Python |
| DOT | the graph language of Graphviz |
| HOPL | the online History of Programming Languages, hopl.info |
| IDE | integrated development environment |
| IEEE 754 | the floating-point standard of the Institute of Electrical and Electronics Engineers; binary64 and binary32 are its 64-bit and 32-bit formats, and NaN its "not a number" |
| IR | intermediate representation |
| JSON | JavaScript Object Notation; canonical JSON is the byte form of 6.1.1 |
| OCI | Open Container Initiative, whose image format holds a pinned toolchain |
| PEP | Python Enhancement Proposal |
| SCXML | State Chart XML, the statechart notation of the World Wide Web Consortium, written in the Extensible Markup Language |
| SHA-256 | the Secure Hash Algorithm with a 256-bit digest (Federal Information Processing Standard 180-4) |
| TOML | Tom's Obvious, Minimal Language |
| UB | undefined behaviour: behaviour for which a language's defining reference places no requirement |
| UTF-8, UTF-16 | the Unicode encodings in 8-bit and 16-bit units |

**The meaning.**

- **Form**: a program's meaning P = (S, E, δ, D, λ) of 6.1. S is the set of states, a product of region records, with an initial state; E the events, the payload types of the declared events plus the built-in `Tick(n)` and the task events of 6.8; δ the total step function S × E → S × O\*, where O\* is a sequence of outputs (T2); D the hash-chained log of the events given to δ, each with its outcome; λ the outputs, the second part of δ.
- **Applied** and **kept**: the two outcomes of δ for an event, and of the gate (next item) for a program or an edit. An event is applied when the selected steps compute the next state. It is kept when the state stays as it was and λ records the reason. The specification and the checker spell the kept outcome `refused` today (6.4, 6.5; `Refused(X, payload)` in λ), and `compliance.md` would respell it `kept`. This proposal writes "kept" in prose and the specification's present spelling in bytes until that proposal is decided.
- **Gate**: the checker's decision on a program, an edit or an event, applied or kept, with the reason. A program the gate applies **complies with the Form**.
- **Reference interpreter** ("the reference"): `spec/checker/src/interp.rs`, which defines δ for RCE text. **Reference trace**: the bytes the branch's `one::trace::reference` prints for an RCE module (`one/trace.rs`): one block per observation (each constant, each check, each zero-argument function and each example, then `fn main`), each block followed by `fuel N`, and after the events of an example the line `trace H`, the head of that example's log chain. An observation starts a new evaluator, so the initial state and all the events of one example share one fuel budget and one set of evaluated constants.
- **Fuel**: RCE's budget of evaluation steps, 50,000,000 (6.7). A host gives each input full fuel (HC-7a); the reference trace gives each observation full fuel. Exhausting it keeps the event, and no `catch` handles it (E1). **Cost units**: the deterministic work counts of C.6, with the cost table of C.6.4.
- **The N9 limits** (6.3): a list or text holds at most SIZE_LIMIT = 1,000,000 elements, and a value written to state holds at most VALUE_LIMIT = 1,048,576 bytes of canonical JSON.

**The core.**

- **rce-core/1** ("the core"): the typed core of `one-emitter.md` section 3 (ONE-2): 9 types, 3 kinds of atom, 13 statements and 3 declarations in ANF, and on the branch 101 primitives, the 100 of ONE-2 and `f64.sign` (`one/ir.rs`). **The lowering**: RCE text to the core (ONE-4, `one/lower.rs`). **The core evaluator**: `one/eval.rs`. **The runtime in the core**: fuel, cost, decimal text, float text and SHA-256, written once as core functions (`one/rt.rs`, `rtf.rs`, `rtt.rs`). **The ratchet**: `COVERED_AT_LEAST = 1435` in `tests/one_conformance.rs`, the least number of conformance tests whose core trace must equal the reference trace.
- **Integer-range inclusion**: a value of type `int[a, b]` may stand where `int[c, d]` is expected when c ≤ a and b ≤ d, the core's rule (`one/wf.rs`).

**The one form.**

- **rce-form/1**: the encoding this proposal defines. **Frame**: its tables for S, E, δ, errors and declared types (§9, below). **Frame type**: a type that a field of S, an event payload, an error or an output of λ may have, an RCE value type of 5.1 with the positional form of 6.1 (§3, below).
- **Definition**: a content-addressed body, a signature and one region. **Group**: one definition, or a set of mutually recursive definitions; the unit of content addressing. **Region**: an acyclic graph with numbered inputs, a list of nodes and an ordered list of outputs; Branch and Iterate nodes own regions. **Node**: one of the five kinds (§6, below). **Port**: a numbered input or output of a node. **Source**: an output of a node or an input of a region; an **edge** joins a source to an input port.
- **Plain** and **owned**: plain values (booleans, integers, floats, byte strings) may be used any number of times. Owned values (arrays, lists, tuples, sums, declared types that hold them, and the order value) are used at most once, and an owned value that nothing uses is dropped where its region ends.
- **Order value** (ω, omega): the owned value of type `order` that every node with an effect, and every node that may halt, consumes and returns (§5, below).
- **Catalogue**: the closed, versioned list of primitive operations of the encoding (§4, below).
- **Runtime library** of a language L: RCE definitions under `lang/<L>/runtime/` that state L's value model and built-in operations exactly.
- **Capability**: a typed, content-addressed host operation (a file, a clock, a foreign function), used only where 5.11 and Appendix E allow it.
- **Key**: the lower-case hexadecimal SHA-256 of a value's canonical JSON bytes. **Module key**: the key of a module. **Canonical**: in the unique representation that the canonical rules fix, so that equal programs have equal bytes. **Merkle digest**: a digest computed bottom up from the digests of a node's parts.
- **Concept record**: the names, source spans, origin, printing hints and labels kept beside a module and never hashed into it (§10, below).
- **Schedule**: the deterministic map from a Form to core statements. **Construction**: the map from core statements back to a canonical Form. **View**: a total, deterministic function that presents a Form in another notation.

**flowG.**

- **flowG**: the parent project of RCE, a first pass at one shared form for all code. **The kernel**: flowG's closed vocabulary, 22 primitives in 6 categories and 11 wire kinds. **The flowG view**: the view of a Form in the kernel's vocabulary, format `rce-graph-view/1`; **unview**: its inverse, from a view to a candidate Form and its concept record (§12, below).

**Lift and emit.**

- **Lift**: the map from the source bytes of a program in a language L to a Form, or to a kept outcome with its reasons. **Emit**: the map from a Form to text in L, or to a kept outcome. **Table**: a data file that drives lift or emit for one language; a **row** is one of its entries. A **lift table** drives lift; a **print table** (ONE-6) drives emit.
- **Defining reference** of a language: a standard, a report, or a pinned implementation at a stated release, which fixes the observable behaviours of every program on every input. **Oracle**: a pinned implementation of a language, named by the digest of its toolchain image, whose runs give the observations that a lift or an emit is compared with. For the language `rce`, the defining reference and the oracle are both the reference interpreter.
- **Clause**: the passage of a language's defining reference that defines a construct: a section of its standard or report, or the source lines of its pinned implementation at a stated release. **Derivation**: a row's stated argument that its template computes what its clause defines, or, for a kept row, why its construct is not carried exactly. Every row of a lift or print table cites its clause and states its derivation (§16 and §19, below).
- **tree-sitter**: a parser generator whose parsers build a concrete syntax tree and mark syntax errors with ERROR and MISSING nodes; a grammar's `node-types.json` lists its node kinds.
- **Lift IR** (`rce-lift/1`): the core's JSON form with a few added statements that stage B removes (§17, below). **Route T** and **route I**: the two targets of a lift row, RCE text or the Lift IR (§16, below).
- **Heap threading**: carrying a language's mutable, aliased objects as a heap value passed into and out of every definition that touches it. **Step splitting**: turning a computation that waits, or loops with no derivable bound, into a region whose steps run between its suspension points. Both are transforms of stage B (§17, below).
- **Floor**, **carrier** and **`lib` row**: as ONE-10, ONE-14 and ONE-17 define them. An **`idiom` row** is a print row that prints a pattern of several nodes as one native construct.
- **Observation**: the canonical record of one run (§20, below). **Case**: one program with one input. **Corpus**: a manifest of cases pinned by SHA-256, each with an expected observation recorded from an oracle.
- **Fixture**: a small program with its inputs that exercises one row. A lift row's positive fixture carries the observation recorded from the language's oracle, which the lifted Form must give byte for byte; a print row's fixtures are its row vectors (stage G1 of ONE-19), whose printed program must print the reference trace byte for byte on the target's oracle; a negative fixture carries the id of the kept row that keeps it. A row is correct when its derivation is stated and its fixtures match exactly.

## 2. The decision: one meaning, one encoding, many views

1. **One meaning.** The Form of 6.1 is kept byte for byte: the canonical signature, the D entry `{seq, event, prev_hash, state_hash, hash}`, the positional form, canonical JSON with its 77 vectors (`spec/data/canonical-json-vectors.json`), and the edge rules N1 to N11 and F1 to F9 of 6.3. It changes only through three open questions of §24 (below): nested options in the positional form, the enforcement of VALUE_LIMIT, and the shape of lifted commands.
2. **One encoding, `rce-form/1`.** A module is the frame plus content-addressed definitions whose bodies are regions over five node kinds. Only this encoding is hashed.
3. **Views, derived and never hashed.** `rce-core/1` is the schedule view, read by printers and the core evaluator. RCE text is the view that RCE's print table prints. Every language L is a view through its print table. The flowG view names the kernel's 22 kinds and 11 wire kinds and has an inverse.
4. **Lift and emit are tables over the encoding** (§16 to §19, below).
5. **One reference for a Form's meaning.** This proposal states the meaning of nodes, of the order value and of the frame. A small Form evaluator (`spec/checker/src/form/eval.rs`, new) implements it and reuses the core evaluator's code for every primitive, so each primitive has one implementation, checked by its catalogue vectors; only the evaluation of nodes, regions and the frame is new. On RCE's corpus three computations must agree byte for byte, observation by observation: the reference interpreter on the RCE text, the Form evaluator on the lifted Form, and the core evaluator on the schedule. A disagreement is a defect in one of them. Two alternatives were considered and set aside. Raising a Form to RCE's syntax tree and running the reference is not total for Forms lifted from other languages (heap values, step-split regions, dynamic-value sums), and a raised tree skips RCE's parse-time rules unless it is printed and parsed again. Taking the core evaluator on the schedule as the definition would let the schedule judge itself.

**Why the graph, and not the statement core, is the hashed encoding.**

- **Convergence.** In a graph, independent pure operations have no order, so two programs that differ only in the order of independent pure statements have one key; the statement core gives them two. That all syntax converges to one form is a statement about identity, so identity must forget what surfaces disagree on and meaning does not depend on.
- **Sharing.** Equal pure computations whose inputs and outputs are all plain are one node (C4, §10, below).
- **Sound facts.** Domain facts are stated on values that cannot change (§7, below), so they cannot go stale. On the branch, W4's check invalidates a fact inside a loop only for the places the loop sets directly, not for a global that a call in the loop sets (`one/wf.rs`, `sets`); the encoding has no globals, so that case cannot be written.
- **Lineage and targets.** The graph is flowG's form, and dataflow is what spatial targets (graphics processors, hardware) consume.
- **Nothing is lost.** The core is derived from the graph by a linear schedule and read back by a linear construction (§13, below), so the one-emitter work (the lowering, the core evaluator, the runtime in the core, the table design) is kept, not replaced.

## 3. Types

**Body types (closed).** These are the types of values inside definitions.

| Constructor | JSON | Plain or owned | Source |
|---|---|---|---|
| bool | `["bool"]` | plain | the core (ONE-2) |
| integers in a range | `["int","lo","hi"]`, −2^63 ≤ lo ≤ hi ≤ 2^64 − 1, in decimal text | plain | the core |
| binary64 | `["f64"]`, finite (F4) | plain | the core |
| binary32, held as binary64 | `["f32"]` | plain | the core |
| byte strings | `["bytes"]` | plain | the core |
| fixed-length arrays | `["arr",τ]` | owned | the core |
| growable lists | `["list",τ]` | owned | the core |
| products | `["tup",τ…]` | owned | the core |
| tagged unions, the tag in `int[0, n−1]` | `["sum",τ…]` | owned | the core |
| a declared type, possibly recursive | `["decl",i]`, an index into the frame's `types` table | as its declaration | new: the core has no recursive type, and the lowering keeps a program that needs one (`39_structural_recursion`, `tests/one_lower.rs`) |
| the order value | `["order"]` | owned | new |

The type `halt` is `sum(bytes, tup(bytes, bytes))`, the core's Halt type (ONE-2): a trap text from the closed list of 6.2, or a declared error as its name and its display.

**Frame types.** Fields of S, event payloads, error payloads and the values that `emit` appends to λ have frame types, the RCE value types of 5.1. The frame keeps them because a host decodes event payloads by declared type and `state_hash` hashes the positional form of 6.1, which the reference computes from the declared type. This proposal spells them:

```
T = ["bool"] | ["int","lo","hi"] | ["f64"] | ["f32"] | ["text"] | ["unit"]
  | ["tup",T,T…] | ["arr",T,"n"] | ["list",T] | ["set",T] | ["map",T,T]
  | ["opt",T] | ["res",T,[x…]] | ["decl",i] | ["chan",T] | ["actor",r]
```

An integer type is spelled by its range, so `u64` and `usize`, which have one range, have one spelling. `x` is an index into the `errors` table (the errors of ε, in ascending order); `i` an index into `types`; `r` the index of an actor region. An alias is the type it names (5.1), so it never appears. Each frame type has the core representation the lowering already gives it (text as bytes, records and tuples as `tup`, options, kinds and results as `sum`, sets and maps as lists in canonical order) and the positional form of 6.1.

**Declared types.** `types` holds records, kinds and refined types:

```
TypeDecl = ["record",[T…]] | ["kind",[[T…]…]] | ["refined",T,REF]
```

A kind lists the field types of each variant in declaration order. A refined type names, by a reference to its group (written REF in the schema of §10, below), the class-0 definition of signature `[[τ],[["bool"]],0]` that evaluates its predicate of 5.1, so a predicate is a definition like any other. Field and variant names appear in the encoding only where a program can observe them (the last rule below); every other name is in the concept record.

**Rules.**

- Values are immutable finite trees, so a recursive declared type is inductive: equality, order and hashing are total structural recursions, and no value contains a cycle.
- The type `order` never appears in a field of S, a payload, an output of λ or a literal.
- Maps and sets are a runtime library over sorted pair lists, as the lowering generates them today with a linear search. Options, kinds and results are sums. The indexed types of 5.16 are erased at run time and stay facts of RCE's surface checker. Channels and iterators are frame constructs, or buffers held in ω (§5, below). Function values are removed by the lift (B3, §17, below).
- Tensors wait for the decision on the tensor proposals TU-1 to TU-6 of E.11. Until then every tensor operation is kept with the reason `tensor`.
- **No colliding positional forms.** The positional form writes `some(x)` as x and `none` as null, so `some(v)` and `none` share one state hash whenever v's own positional form is null. W-TY-2 already keeps a module that writes `τ??`. The collision remains where the inner type's positional form can be null without being written as an option: `()?`, and an alias or a type parameter that stands for an option. The gate keeps a module whose frame types hold an option of such a type (FG14, §11, below); a lift writes a declared kind instead, whose positional form `[k, fields]` does not collide.
- **Observable names are data.** A name that a language lets a program observe is part of its meaning: Python's `__name__` and the `repr` of classes and functions, exception class names in printed tracebacks, field names that a serializer writes. Such names are byte-string literals in the Form (in the runtime library's class and record tables), so renaming them changes the key. Only names no program can observe go to the concept record, and T7 extends to whole programs as "renaming a name that no program can observe never changes a key".
- **A language's value model is a runtime library** over these types: Python's unbounded `int` is a list of digits with definitions for its operations, JavaScript strings are lists of UTF-16 code units, and C's `int`, under a profile that fixes it at 32 bits, is `int[-2^31, 2^31-1]` with tests for UB.

## 4. The catalogue (closed)

**Contents.** The catalogue holds the primitives of the core except four, and adds the entries that act on the order value.

- `io.byte`, `mem.get` and `mem.set` act on standard output and on one memory array that the core evaluator holds outside every value. They stay in `rce-core/1`, where printed carriers use them (ONE-14), and no Form can name them, so no Form can keep state outside S or write outside λ.
- `drop` is never written in a Form, because an owned value that nothing uses is dropped where its region ends (§8, below); the schedule writes the core's `drop` for it.
- The entries on ω are `emit[T]`, `fuel.tick`, `cost.charge` with the budget entries of C.5 and C.6, `memo.get` and `memo.put` (§5, below).
- A name outside the catalogue fails decoding (FG1, §11, below).

**Read primitives return their operand.** The core reads without consuming the first operand of `list.len`, `arr.len`, `copy` and `sum.tag`, and every operand of `list.get`, `arr.get` and `tup.get` (`one/wf.rs`, `reads`). In the Form each of these returns its owned operand as an extra trailing output, the same value, so every use of an owned value is a consumption and "used at most once" is a count. The schedule drops the extra output, so printed programs are unchanged.

**Each entry states** its name; its type parameters; the mode of each input (consume, read-return, update-return; §8, below); its outputs; its domain test (the `bad.*` tests, and the index, length, tag and divisor tests of W4); its cost class (C.6.4); its exactness class (exact integer; IEEE 754 correctly rounded under F1 to F9; exact bytes; library-defined); its operand symmetry (the positions that may be exchanged, which C3 uses, §10, below); and its edge vectors, generated by the reference at the edges of its types, in `lang/vectors/catalogue/<entry>.json` (format `rce-row-vectors/1`, the vectors of stage G1 of ONE-19). An entry joins a catalogue version only with its vectors, and every row that uses it inherits them.

**File.** `spec/data/form-catalogue.json` (format `rce-catalogue/1`), generated from the book by `rce-spec-data` as `spec-data.json` is today. Every module names the key of the catalogue it was written against. A change of meaning to any entry is a release with a new catalogue key (C.15).

**What the catalogue is not.** It is not the place for library operations. A language's operations, and flowG's operation vocabulary, become runtime-library definitions written down to the catalogue (as F9 has it for mathematics: transcendentals are correctly rounded RCE code, or absent), capabilities, or kept constructs. Each of flowG's operations goes to exactly one of four places: a catalogue entry at a type, a runtime definition, a capability, or kept with its reason. That table is generated from flowG's operation list, and generation fails when an operation has no place. A capability named by flowG at run time, through a registry, is resolved once at lift time to the key of one `rce-cap/1` signature and recorded in the lift report, never resolved at run time.

## 5. The order value and its two runners

The order value ω stands for the working context of one run of δ: the fuel left, the cost spent and the budgets in force, the outputs gathered so far, and the memo cells. δ takes ω with the event and returns it with the next state. ω is never part of S, E, D or λ, and whoever runs δ supplies it. There are two runners, and they differ only in when they start a fresh ω.

- **On a host** (HC-7a), each input starts with a fresh ω: full fuel, no cost spent and no budget in force, an empty λ and empty memo cells. When the input is applied, λ is the outputs it gathered. When it is kept, λ is `[Refused(X, payload)]` in the specification's present spelling, and the rest of ω is dropped. So a kept input leaves nothing behind, and an input's outcome, a fuel trap included, is a function of that input and the state before it, as HC-7a states.
- **In an observation** of the reference trace (10.5), the reference starts one evaluator per observation, so an example's initial state and all its events share one fuel budget and one set of evaluated constants, and the example's `fuel N` line is their total. The runner threads one ω through the observation: after each event it takes λ out and renews nothing else. The branch's observation driver does the same ("each observation from full fuel and with no constant evaluated", `one/lower.rs`), and so does the Form evaluator, which is what keeps the three computations of §2 equal byte for byte, `fuel N` lines included.

The host rule differs from the reference interpreter's host in one respect. The reference keeps its evaluated constants for a whole session while it resets fuel before each input, so a constant that an earlier input evaluated costs a later input no fuel, while a host restored from a stored state starts with none evaluated and charges the same input more. Fresh memo cells per input charge a constant again in every input that reads it. Which rule hosts follow is open question 6 of §24 (below).

**Entries that act on ω.** Each consumes ω and returns it.

- `emit[T]` appends a value of frame type T to λ.
- `fuel.tick` takes n units of fuel, or halts with the reference's fuel trap when fewer than n are left (the runtime's `rt.tick`, `one/rt.rs`, whose trap text is the reference's "comptime fuel exhausted").
- `cost.charge` and the budget entries keep the counters of C.5 and C.6 (the runtime's `rt.charge` and its budget counters).
- `memo.get` and `memo.put` read and fill a memo cell of ω, named by content: `["memo", REF]` holds the value of the zero-input definition REF, evaluated at its first use while that ω lives (one input on a host, one observation in a trace) and reused afterwards, as the reference evaluates RCE constants; `["buffer", REF]` holds the buffers of the iterator definition REF (5.14). An owned cell is taken and put back before the function returns or calls, the rule the core applies to owned globals (W3).

**Why this removes hidden state.** The lowering keeps all of this in mutable globals of the core: the output list, a flag and a value per constant, iterator buffers, fuel and its limit, and the cost counters, and it resets them at the start of every observation. In `rce-form/1` they are all inside ω, and the encoding has no globals. So no definition can carry a value from one input to the next except through S (FG10, §11, below), `state_hash` and replay see every bit of state that crosses inputs, and on a host a kept input leaves nothing behind. Because a memo cell is named by the definition it belongs to, a group's meaning never depends on the module that holds it, as a core global's index does.

**The fuel limit** is the reference's limit, per input on a host and per observation in a trace. The fuel sweep of the emit gate (stage G5 of ONE-19) runs the same Form at other limits through `rce trace-at` (§21, below), which records the limit in its output; the limit is never a field of a module.

## 6. Nodes (closed: five kinds)

A node is `[k, attrs, ins, regions]`. The kind numbers are flowG's kernel numbers, so the flowG view and the lineage are kept.

| k | Kind | attrs | ins | regions | outputs | Meaning |
|---|---|---|---|---|---|---|
| 0 | Bind | `["lit",τ,LIT]`, τ plain | none | none | one value of τ | a literal; parameters are region inputs, not nodes |
| 1 | Apply | `["prim",NAME,[ι…]]`, a catalogue entry at type arguments ι (for `memo.get` and `memo.put`, `["prim",NAME,[ι…],CELL]`); `["call",REF]`, a definition; `["catch",REF]`, a definition whose halt becomes a value; `["cap",KEY]`, a capability | per the signature, ω last when the entry or callee has an effect or may halt | none | per the signature, ω last; `catch` returns `sum(tup(outputs…), halt)` and ω | apply an operation |
| 3 | Branch | `[n,[τ…]]`: n ≥ 2 arms and the declared output types | the selector (`bool`, or `int[0, 1]`, when n = 2; otherwise `int[0, n−1]`), then the captured values | n regions, each taking the captures as inputs, with outputs that lie within the declared types | one value of each declared type | evaluate only the selected arm: arm k for selector value k, arm 0 for `false` |
| 4 | Iterate | `["count",[τ…]]` or `["measure",[i,j],[τ…]]`, with the declared carried types | count: the bound n (a source of type `int[0, 2^64−1]`, evaluated once before the loop), then the initial carried values; measure: the initial carried values. When the Iterate takes ω, ω is the first carried value | two regions, test and body, each taking the carried values and returning `[bool, carried values…]`; neither captures anything | the final carried values, ω first when taken | repeat under a bound (6.7) |
| 13 | Halt | `[[τ…]]`, the output types the enclosing region needs | a value of type `halt`, then ω | none | none is ever produced | stop with a trap text or a declared error |

**Semantics.**

- **Evaluation order.** A region's nodes are evaluated once each, in index order. The canonical numbering (C5, §10, below) is a topological order, so the order is a function of the graph.
- **Calls.** `call` evaluates the callee's body on the arguments. A halt in the callee halts the caller, except under `catch`, which returns the halt as the second variant of its result. On that path `catch` returns the ω that the callee's Halt consumed: fuel and cost spent and outputs gathered before the halt stay, as in the reference, whose `catch` restores none of them. The fuel trap passes through every `catch`, because fuel exhaustion cannot be caught (E1). The core's `catch` catches every halt, the fuel trap included (`one/eval.rs`), so the schedule writes after each `catch` a Branch that halts again when the caught halt is the fuel trap, and the construction reads that pair back as one Form `catch` (§13, below). RCE's own `catch` handles only declared errors (6.2), so the lift of RCE writes after each Form `catch` a Branch on the halt's tag that halts again on a trap text, except the "budget" trap that the `else` of `with budget n … else …` handles (C.5).
- **Capabilities.** `cap` is allowed only in the context of `main` and of task bodies (5.11). Its results enter D as `Foreign(f, arguments, result)` and are replayed, never run again (HC-8).
- **Iterate with a count bound.** With bound n and carried values c (ω first): set i = 0; then repeat: run the test region on c, giving (k, c); if k is `false`, output c; else raise i by one, and if i > n, halt with the trap text "loop bound"; run the body region on c, giving (k, c); if k is `false`, output c. This is the order of the reference's loop, which the lowering reproduces in `run_loop` (`one/lower.rs`): the condition is tested, the counter raised, the bound tested, and only then does the body run. So iteration n + 1 traps when it starts, before any of its work, as 6.7 states ("Starting iteration n + 1 is `trap("loop bound")`"). A loop with no condition has a test region that returns `true`; a `break` in the body returns `false`. Because the bound is a source, a bound computed at run time (`while c bound n`, n "comptime-or-runtime") and the default LOOP_LIMIT are the same construct. A count-bounded Iterate may halt, so it always takes ω.
- **Iterate with a measure.** `["measure",[i,j],[τ…]]` is applied only when the gate proves termination by one closed shape: i and j are positions among the carried values; the test region returns carried values i and j unchanged, and its first output is `int.lt` of i and j, or `bool.and` with that `int.lt` as an operand; the body region returns j unchanged and i raised by exactly one whenever it continues. This is the shape of `for` over ranges and lists in the lowering. Such a loop cannot run past its measure, so the Iterate has no run-time check and no trap text. A `for` over a range keeps the reference's check before the loop, as a Branch and a Halt with the trap text "loop bound" when the range holds more items than the reference allows. Every other loop carries a count bound, or is step-split (B5, §17, below).
- **Halt.** A halt stops the definition. At the top of a step it keeps the event: the whole state stays and λ is `[Refused(X, payload)]` (6.4, 6.5, T8).
- **Effect classes.** A function's signature carries one of: 0 total, 1 may halt, 2 emits (step context), 3 host (the context of `main` and of task bodies, where capabilities and foreign calls are allowed). 5.2 gives `fn main` and task functions the same effects, and 5.11 allows foreign calls from both, so one class holds both, and the classes form a chain 0 < 1 < 2 < 3. A function of class 1 or above takes ω as its last input and returns it as its last output; a step body is class 2 by the frame, and `main` and task bodies are class 3. Every node that may halt (a Halt, a count-bounded Iterate, a call or capability that may halt, `fuel.tick`, `cost.charge`) consumes and returns ω. So every halt, fuel exhaustion included, lies on the ω chain, and effects and halts are totally ordered by linearity alone (§8, below).
- **Fuel.** Fuel is charged only by `fuel.tick` nodes, so it is a function of the Form. The lift of RCE writes them where the reference charges fuel, as the lowering does, so traces run observation by observation keep their `fuel N` lines byte for byte. Other languages' tables write one tick per loop iteration and per call. Whether fuel should later be charged per node kind is open question 7 of §24 (below).

**Why five kinds suffice.** Data values (Bind), operations and calls (Apply), choice (Branch), bounded repetition (Iterate) and halting (Halt) are the constructs of a total, first-order, structured program. Of the other 17 kernel kinds, 16 name a part of the frame or are compositions of these five, and CodeBlock appears only in the view of a kept lift (§12, below). Each of the five is needed: without Branch only straight-line code exists; without Iterate every loop becomes a bounded recursive group, which many printed languages run on a limited stack, while the core has a loop and every complete table can print one (ONE-10); without Halt, partial operations cannot be total. Recursion needs no node kind: it is a group whose functions call each other through `["rec", i]`, with a bound. The five shapes are the core's statement families `let`, `prim`, `call`; `if`; `loop` and `break`; and `halt` (ONE-2).

## 7. Domain facts

Facts are stated on values. A value keeps its identity along the two edges that pass it on unchanged: a capture into a Branch arm (the arm's input is the captured source), and the returned operand of a read-return port. Two sources joined by such edges are **the same value**. A node is **guarded** by a test when it lies inside the arm of a Branch on that test that the test's in-domain outcome selects.

A partial entry may occur only where one fact from the closed list below establishes its domain on the same values as its operands. There is one fact for each domain case of W4:

- **DF1 Types.** The operand types imply the domain (the core's integer types carry ranges), a non-negative index or length included.
- **DF2 Overflow tests.** `int.add`, `int.sub`, `int.mul`, `int.neg`, `int.abs` and `int.cast` are guarded by their `int.bad.*` test, `int.shl` and `int.shr` by `int.bad.shift`, and the `f64` and `f32` operations by theirs.
- **DF3 Division.** `int.quo` and `int.rem` are guarded by a test of the divisor against 0 and by `int.bad.quo`.
- **DF4 Tags.** `sum.get k` lies inside arm k of a Branch on `sum.tag` of the same value.
- **DF5 Indices.** `bytes.get`, `list.get`, `arr.get`, `list.set` and `arr.set` have an index that is not negative (by DF1, or guarded by a test) and are guarded by a test that the index lies below the length that `bytes.len`, `list.len` or `arr.len` returned for the same value.
- **DF6 Slices.** `bytes.slice` has a start that is not negative, a start not after the end, and an end not after the length of the same value, each by DF1 or by a guarding test.
- **DF7 Non-empty lists.** `list.pop` is guarded by a test that the length of the same value is above 0.
- **DF8 New arrays.** `arr.new` has a length that is not negative, by DF1 or by a guarding test.

This is W4, restated on values that cannot change.

## 8. Sources, port modes, linearity and the eleven wire kinds

**Sources.** An input of a node is a source: `[i, p]`, output p of node i of the same region, or `[-1, j]`, input j of the region. There is no wire-kind field.

**Port modes.** Each input port has a mode fixed by the node kind and the catalogue: **consume**; **read-return** (the operand comes back unchanged as an output); **update-return** (the operand comes back changed); **selector**; **capture** (into a Branch arm); **carried** (an initial carried value of an Iterate, ω among them when it is taken); **bound** (an Iterate's count); and **order** (ω, on every other node). An Iterate's regions capture nothing: a value used inside a loop is carried, and returned unchanged when the loop does not change it.

**Linearity.** ω is used exactly once: every output and region input of type `order` is the source of exactly one input, or is an output of its region. Every other output and region input of an owned type is used at most once, a region output counting as a use; one that nothing uses is dropped where its region ends (the core's `drop`, which the schedule writes and the construction reads back as no use). Plain values may be used any number of times. A drop needs no node, so the encoding holds none, and a region that ends in a Halt drops what it holds, as W3 lets a halting path do. Because ω is used exactly once, the effects and halts of a region form one chain.

**The eleven wire kinds, as derived labels.** flowG's kernel has 11 wire kinds. Each is a label derived from a source's type and the mode of the port it enters, never stored.

| flowG wire kind | Class | Holds for an edge when | RCE counterpart |
|---|---|---|---|
| Move | edge | the type is owned and the port consumes it | the ownership kind Move, W-OWN-1 (5.5) |
| Borrow | edge | the type is owned and the port is read-return | `&τ` (5.6) |
| MutBorrow | edge | the type is owned and the port is update-return | `&mut τ`, written back by state passing (5.6) |
| Copy | edge | the type is plain, or the source is an output of `copy` | the kind Copy, `copy(e)` |
| Error | edge | the type is `halt` or a `catch` result, and the edge leads to a Branch on its tag or to a Halt | `τ ! ε`, `try`, `catch` (5.3) |
| Lazy | edge | the edge is a capture into a Branch arm, evaluated only if that arm is selected | the conditional positions of 6.2 |
| Feedback | edge and frame | a carried output of an Iterate's body region, which becomes the value of the next run; in the frame, a step's output field to the next state | loop state; the fold law of 6.1 |
| Channel | frame only | an output request that the host delivers as an event to another region's mailbox | `chan τ` (6.8) |
| Stream | frame only | a resumable producer: a frame value with a resume definition, or a task region | `iter τ` (5.14) |
| Shared | never an edge | sharing an immutable value cannot be observed, so the lift writes Copy or Borrow; shared mutable state becomes a heap value or a region; otherwise the construct is kept | tasks share nothing (W-SPAWN-1) |
| Weak | never an edge | a weak reference becomes an index into a heap or a state map; observable collection is kept | none |

**Precedence.** Where several rows hold for one edge (an owned capture is also consumed; a carried owned value is also moved), the first of this order is its label: Error, Feedback, Lazy, MutBorrow, Borrow, Copy, Move. An edge that carries ω takes the label these rules derive, as any owned value's edge does. The view also draws each region's ω chain as one Sequence node, because Sequence is a kernel node kind, not a wire kind (§12, below).

So the statement that an invalid wire cannot be drawn holds of the encoding: there is no kind to mislabel, and the gate's linearity rule (FG6, §11, below) keeps any module that uses an owned value twice.

## 9. The frame: S, E, δ, D and λ

The frame is a set of closed tables, the same in every program and every language.

| Table | Content | Form part | flowG kind it absorbs |
|---|---|---|---|
| `types` | the declared types of §3, minimised and ordered by C8 (§10, below) | the types of S, E and bodies | TypeDefine (20) |
| `regions` | per region: `"state"` or `"actor"`, then its fields `[T, VALUE]`, each with its initial value evaluated at comptime (5.10); actor regions add a mailbox and a held-message queue | S and its initial state | FeedbackDelay (21) |
| `events` | per declared event, its payload field types; an event's index is its position; `Tick(n: u64)` and the task events of 6.8 are built in and keep their names, as 6.1's positional form keeps them | E | Listen (18) |
| `errors` | per declared error, its field types | the X of a kept event | ErrorPropagate's error types |
| `steps` | `[region, event, guard REF or null, body REF, deadline or null]` in declaration order | δ | Listen (18); Abstract (8) for the bodies |
| `main` | the body REF of a command's `Start` event, in host context, or null | δ of a command (4.2) | Abstract (8) |
| `defs` | the groups, keyed by group key | the bodies | Abstract (8); Import (9), resolved to keys |
| `caps` | the keys of the capability signatures the module uses | the host boundary (Appendix E) | Acquire (10), Release (11) |

**δ.** For an event of type t with payload v: for each region r in declaration order, the first step of r whose event is t and whose guard holds on r's fields and v is r's step; every selected step runs on the pre-state, on r's fields, v and ω; its outputs are r's next fields. A region with no matching step keeps its fields. If any selected step, or a guard, halts outside a `catch`, the event is kept: the state stays, and λ is `[Refused(X, payload)]`, X from the first region in declaration order that halted. Otherwise the next state is the list of next fields, and λ is the values the bodies emitted, regions in declaration order, each in ω order. These are the semantics of 6.4, read off tables.

**Signatures fixed by the frame.** A guard of a step of region r on event e has inputs (r's fields, e's payload fields) and output `[bool]`, of class 0 or 1 (a guard that may halt takes ω). A body has inputs (r's fields, e's payload fields, ω) and outputs (r's fields, ω), of class 2, or of class 3 in a task region. The lift compiles patterns on payloads into guards and into the destructuring at the start of the body.

**Commands.** A command is a program whose root declares `fn main()` and no state; it denotes the Form with one region `Main` of unit type, one event `Start(args: List[text])`, and δ given by running `main` (4.2). The `main` body is in host context (class 3), so it may use capabilities and call foreign functions, which 5.11 allows only from `fn main` and task functions and never from a step. A lifted batch program keeps that shape with two departures. `Main` holds the source's module state as fields, and `Start` carries the environment and standard input beside the arguments, `Start(args: List[List[u8]], env: List[(List[u8], List[u8])], stdin: List[u8])`, as byte lists because arguments and environment values need not be UTF-8. A lifted command is therefore not a command of 4.2: it is a Form with state whose `Start` body runs in host context. Whether the specification admits this shape is open question 3 of §24 (below).

**D and λ.** Unchanged. A D entry is `{seq, event, prev_hash, state_hash, hash}` over positional canonical JSON, with a genesis of 64 zeros (6.1). The D chain of RCE text keeps its format, `rce-trace/2` (10.9). The D chain of a Form, headed by its module key, has a format of its own, `rce-trace/3`, because its header changes; the reference trace has `rce-ref-trace/1` (§21, below). Nothing of the encoding, the concept record or any table is written into D.

**Concurrency lives only in the frame.** Tasks, channels, `select` and actors are regions, mailbox fields, the built-in task events and the tick scheduler of 6.8: a `spawn` is an output request, a join is a step on the completion event, and `select` is a set of guarded steps with its rotating start held in S. No body node spawns, joins, waits or yields, so every body stays sequential, total and replayable.

## 10. Encoding and identity

**Schema.** All bytes are canonical JSON (6.1.1).

```
Module   = {"format":"rce-form/1", "catalogue":KEY, "types":[TypeDecl…], "regions":[Region…],
            "events":[[T…]…], "errors":[[T…]…], "steps":[Step…], "main":REF|null,
            "caps":[KEY…], "defs":{KEY:Group…}}
TypeDecl = ["record",[T…]] | ["kind",[[T…]…]] | ["refined",T,REF]             (§3)
Region   = ["state"|"actor", [[T, VALUE]…]]
Step     = [r, EV, REF|null, REF, "N"|null]   // region, event, guard, body, deadline in ticks
EV       = i | NAME                            // an index into events, or a built-in event's name
Group    = {"fns":[Fn…]}                       // one function, or a recursive group
Fn       = {"sig":[[τ…], [τ…], EFF], "bound":null | "N" | ["layers",[[[i…]…]…]], "body":Body}
Body     = {"nodes":[Node…], "out":[Src…]}     // a region; its inputs are the signature's inputs
Node     = [k, attrs, [Src…], [Body…]]         // the regions of a Branch or an Iterate
Src      = [i, p] | [-1, j]                    // output p of node i | input j of the region
REF      = [KEY, i] | ["rec", i]               // function i of a group; "rec" only inside its group
CELL     = ["memo", REF] | ["buffer", REF]     // a memo cell of ω, named by its definition
T        = a frame type (§3)
τ        = ["bool"] | ["int","lo","hi"] | ["f64"] | ["f32"] | ["bytes"] | ["arr",τ] | ["list",τ]
         | ["tup",τ…] | ["sum",τ…] | ["decl",i] | ["order"]
LIT      = true | false | "decimal" | "16 hex digits of the binary64 bits" | "hex of the bytes"
VALUE    = the positional canonical JSON value of 6.1
EFF      = 0 total | 1 may halt | 2 emits | 3 host (main and task bodies)
attrs    = k 0:  ["lit",τ,LIT]
           k 1:  ["prim",NAME,[ι…]] | ["prim",NAME,[ι…],CELL] | ["call",REF] | ["catch",REF] | ["cap",KEY]
           k 3:  [n, [τ…]]                     // arms; output types
           k 4:  ["count",[τ…]] | ["measure",[i,j],[τ…]]   // carried types, ["order"] first when taken
           k 13: [[τ…]]
```

A type argument ι is a body type, or a frame type for `emit`. A function's `bound` is null when it is not recursive; a number N when it is declared `bound N` (at most N active calls, then `trap("recursion bound")`, 6.7); or the layered-measure certificate of 6.7 (the parameter positions of each layer), which the gate checks again on the graph. The literal encodings, the reference form and the type spellings of bodies are those of the core's JSON form (`one-emitter.md` section 3.4), so a reader of one is most of a reader of the other.

**Canonical rules.**

- **C1 Strict decoding.** Every field is present and no unknown field is; integers are canonical decimal text and floats 16 lower-case hexadecimal digits; a module with an unknown kind, entry, type or catalogue key is kept. Nothing defaults.
- **C2 Merkle digest.** md(input j) = SHA-256 of `["in", j]`; md(output p of node i) = SHA-256 of `["out", md(node i), p]`; md(node) = SHA-256 of `[k, attrs, [md of each source], [md of each region]]`; md(region) = SHA-256 of `["region", [md of each output]]`; all over canonical JSON, computed bottom up, regions first.
- **C3 Symmetric operands.** For an entry whose catalogue record declares two operand positions exchangeable (`int.add`, `f64.mul`, `bool.and`), the two sources are ordered by ascending Merkle digest.
- **C4 Sharing.** Two nodes of one region with the same Merkle digest whose inputs and outputs are all plain are one node. A node with an owned input or output is never merged with another: two equal allocations are two values under linearity, and owned twins share a digest, so two nodes that consume twins (two `tup.open` or `sum.get` of equal tuples) also share one, and merging them would leave a twin unused. Such twins are interchangeable, so C5 still numbers them uniquely.
- **C5 Numbering.** Nodes are numbered in the first-completion order of an iterative depth-first walk from the region's outputs, in output order, visiting each node's sources in port order. The numbering is topological and is the evaluation order.
- **C6 Reachability.** Every node is reachable from its region's outputs. Every node with an effect or that may halt is, because the ω it returns must reach an output. Canonicalisation removes an unreachable node; only a pure, total node can be one, and an owned value it held is dropped, so the removal changes no meaning.
- **C7 Captures and carried values.** A Branch captures exactly the outer sources its arms use, ordered by Merkle digest. An Iterate's carried values keep the lift's order, since loop variables are positional, with ω first when taken; a carried value that neither region uses and both return unchanged is removed.
- **C8 Types.** Declared types that are structurally equal, recursive ones included, are merged (bisimulation on the type graph); the table is ordered by the digest of each declaration's graph, and indices are renumbered.
- **C9 Recursive groups.** Members are ordered by an iterated digest: in round 0 each member is digested with its references inside the group written `["rec","?"]`; in round r + 1 each such reference is replaced by the round-r digest of its target; after as many rounds as there are members, members are sorted by digest, and members with equal final digests are one function (they are bisimilar). So a group's key does not depend on the order in which a module first used its members.
- **C10 No unobservable names.** No field holds a name that no program can observe, a comment, a span or layout. Observable names are data (§3).
- **C11 Guarded primitives inline.** A call of a function that is not recursive and whose body is exactly a partial primitive's domain test, the Branch around the primitive and a Halt with the runtime's trap text is replaced by that body. So a lift row that calls a guarded runtime definition and a lowering that writes the guard inline give one key. No other call is inlined, so the rule is closed.

**Keys.**

- A group key is the SHA-256 of the canonical JSON of its Group (W5). A module key is the SHA-256 of the canonical JSON of the Module, whose `defs` object canonical JSON orders by key.
- Changing one function changes its group key, the keys of the groups that reference it directly or transitively, and the module key, and nothing else, so lift, gate and emit work incrementally by key.
- One hash function, SHA-256, serves groups, modules, D, vectors, reports and evidence records, as D, the Lean proofs and the 77 canonical JSON vectors already require.
- Because no unobservable name enters the bytes, two programs that differ only in such names, in comments or in layout have one key: equal structure, one key. Difference and merge over Forms are not designed here.

**The concept record, `rce-concept/1`.** Beside each module, never hashed into it:

```
{"format":"rce-concept/1", "module":KEY,
 "origin":{"lang":L, "sources":[[path, sha256]…], "lift_table":KEY, "runtime":[KEY…],
           "grammar":"repository@commit", "config":KEY|null},
 "names":{…}, "spans":{…}, "hints":{…}, "view":{…}, "desugar":{…}, "labels":{…}, "trivia":{…}, "energy":{…}}
```

`origin` gives the source language, the source files with their SHA-256, the keys of the lift table and of the runtime libraries, the grammar's pin, and `config`, the key of the build configuration the source was lifted under (format `rce-build-config/1`, §18, below), or null. `names` gives the source name of every region, field, event, error, function and local whose name no program can observe. `spans` gives the byte range of each node's source. `hints` holds the facts that do not change meaning but help print idiomatic text, among them the source construct of each node. `view` gives the kernel kind of each node where the lift knows it. `desugar` names the lift row that produced each node, so emission can print the source idiom back. `labels` gives each definition's evidence label (§16, below). `trivia` holds comments and layout for reprinting. `energy` holds declared annotations with their provenance, never meaning.

**Worked example (illustrative; keys symbolic).** The counter `state C: n: u64 = 0`, `event Inc`, `event Dec`, `step C on Inc: n = n + 1`, `step C on Dec if n > 0: n = n − 1`, with T = `["int","0","18446744073709551615"]`, which is both the body type and the frame type of `u64`:

```
"regions": [["state", [[T, 0]]]],
"events":  [[], []],
"steps":   [[0, 0, null, [K_inc,0], null],
            [0, 1, [K_pos,0], [K_dec,0], null]]

K_inc: {"fns":[{"sig":[[T,["order"]], [T,["order"]], 2], "bound":null,
  "body":{"nodes":[
     [0, ["lit",["int","1","1"],"1"], [], []],                           // 0: the literal 1
     [1, ["prim","int.bad.add",[T]], [[-1,0],[0,0]], []],                // 1: would n + 1 leave T?
     [3, [2,[T,["order"]]], [[1,0], [-1,1],[-1,0],[0,0]],                // 2: Branch on node 1, capturing ω, n and 1
        [{"nodes":[[1, ["prim","int.add",[T]], [[-1,1],[-1,2]], []]],    //   arm 0: n + 1
          "out":[[0,0],[-1,0]]},
         {"nodes":[[1, ["call",[K_ovf,0]], [[-1,1],[-1,2]], []],         //   arm 1: the exact trap text,
                   [13, [[T,["order"]]], [[0,0],[-1,0]], []]],           //   then halt
          "out":[[1,0],[1,1]]}]]],
   "out":[[2,0],[2,1]]}}]}

K_pos: {"fns":[{"sig":[[T], [["bool"]], 0], "bound":null,
  "body":{"nodes":[[0, ["lit",["int","0","0"],"0"], [], []],
                   [1, ["prim","int.lt",[T]], [[0,0],[-1,0]], []]],
          "out":[[1,0]]}}]}
```

K_dec is K_inc with `int.bad.sub` and `int.sub`. The literal 1 has type `int[1, 1]` and enters ports of type T by integer-range inclusion (FG5, §11, below). The capture order shown is the logical one; C7 orders captures by digest. Arm 0's `int.add` reads the captures of n and of the literal, the same values that node 1's `int.bad.add` tested, so DF2 holds. `int.lt` reads plain operands, so it returns none. K_ovf is the runtime definition that writes the overflow text from the exact operands, as the reference writes it. The `fuel.tick` nodes that the lift of RCE writes at the reference's charging points are left out of this illustration.

## 11. The gate on `rce-form/1`

The checker keeps a module at the first rule that fails and names the rule, the group key, the function index and the node path. A module that passes every rule complies with the Form.

| Rule | Check |
|---|---|
| FG1 Decode | C1: every field, kind, entry, type, literal and the catalogue key decode; nothing defaults |
| FG2 Keys | every group key equals the SHA-256 of its bytes; every REF and CELL resolves; the graph of keys is acyclic, with `rec` only inside its group |
| FG3 Canonical | C2 to C11 hold: decoding and encoding again gives the same bytes |
| FG4 Ports | each node has the arity its kind and catalogue entry give; every source exists and precedes its consumer in the numbering |
| FG5 Types | the type of every source, derived from its producer (an entry's or callee's signature, or the declared types of a Branch, an Iterate or a Halt) and never stored on the edge, equals the type its port requires, or both are integer types and the source's range lies inside the port's (integer-range inclusion); by the same rule, each arm's outputs lie within its Branch's declared types and each run's carried values within its Iterate's; every function is monomorphic, generics being monomorphised by the lift (ONE-4) |
| FG6 Linearity | every output and region input of type `order` has exactly one consumer or is a region output; every other owned output and owned region input has at most one consumer, a region output counting as one, and is dropped where its region ends when it has none; owned memo cells are taken and put back before a return or a call |
| FG7 Effects | each function's class is at least the greatest class among its nodes and callees, in the chain 0 < 1 < 2 < 3; every node with an effect or that may halt consumes and returns ω; `cap` only in class 3; `emit` only in classes 2 and 3; the context table of 5.2 |
| FG8 Totality | every partial entry has one of the facts DF1 to DF8 on the same values as its operands; every Iterate has a count bound or a measure of the closed shape of §6; every recursive function has a declared bound or a layered-measure certificate that holds on the graph (6.7), and every cycle of calls passes through one; every Branch has one region per selector value |
| FG9 Acyclic regions | every region is a directed acyclic graph; repetition only by Iterate; cycles only in the frame (state to state) |
| FG10 State | each region field has a comptime initial value; a step body's signature is its region's fields; a body reads only its own region; the module has no state outside S: no globals exist, and ω never appears in S, E, λ or a literal |
| FG11 Events | every step names a declared or built-in event; guard and body signatures match its payload; `main` names region `Main` and its `Start` event |
| FG12 Numbers | every arithmetic entry is typed at a width; floats follow F1 to F9; no entry depends on the host; `io.byte`, `mem.get` and `mem.set` do not occur |
| FG13 Boundary | every capability has an `rce-cap/1` signature whose types are core types; answers of a model arrive only as events (F.5.6) |
| FG14 Frame types | every frame type is an RCE value type with a positional form, and no two of its values share one: no option of a type whose positional form can be null (`()`, an option, or an alias or parameter that stands for either), which W-TY-2 does not catch unless the nesting is written |
| FG15 Admission | the module's admission vectors (`rce-vectors/1`, §18, below) hold on the Form evaluator, as RCE's examples and checks do today (6.9) |

The rules of section 7 that concern RCE's surface (lexical rules, names, hygiene) stay on RCE text. The rules about meaning (W-OWN, W-REC, W-ITER-1, W-INT-1 and the rules of Appendix F that survive erasure) are restated by FG1 to FG15. W-STEP-2, which keeps a module with an event that no step and no actor handler can ever match, also stays on RCE text, as an authoring rule: in a Form such an event is applied with the identity update of the frame rule (6.4), and a lifted program may declare events it ignores, as a statechart does. `compliance.md` reads that rule the same way.

## 12. flowG's 22 kinds and 11 wire kinds: the view and its inverse

flowG's kernel vocabulary is the canonical vocabulary onto which every executable form maps, and a conversion between it and any other form should be generated, checked and invertible, so that saving after loading is the identity. This proposal carries all 22 kinds and all 11 wire kinds, and draws one line: the encoding holds the five kinds that carry meaning; the flowG view restores every one of the 22 names and the 11 wire kinds from the encoding, the concept record and the lift report; and the view has an inverse, so the 22-kind graph is an editing vocabulary, not only a picture.

**The 22 kinds,** numbered and grouped in the kernel's six categories: computation (0 to 9), resource (10 to 12), error and contract (13 and 14), concurrency (15 to 18), data and state (19 and 20), and temporal (21).

| # | Kind (category) | Class | In `rce-form/1` | How the view recognises it |
|---|---|---|---|---|
| 0 | Bind (computation) | encoded | node Bind (literals); parameters are region inputs; a mutable binding becomes rebinding (edges, Branch outputs, Iterate carried values), or a field of S when it is module state | each Bind node, each region input and each field's initial value |
| 1 | Apply (computation) | encoded | node Apply: a catalogue entry, a call, a `catch`, a capability | each Apply that no more specific row below claims |
| 2 | Mutate (computation) | composition | an Apply of an update-return entry (`arr.set`, `list.set`, `list.push`, `list.pop`), or a tuple rebuilt by `tup.open` and `tup.mk` | an Apply whose entry has an update-return port |
| 3 | Branch (computation) | encoded | node Branch | a Branch not drawn as Match |
| 4 | Iterate (computation) | encoded | node Iterate, with a count bound or a proven measure; a loop with no derivable bound is step-split | each Iterate node |
| 5 | Match (computation) | composition | a Branch with one arm per tag on `sum.tag` of one source, or a chain of Branches on comparisons that the lift compiles, in order, from literal and range patterns, exhaustive by 5.3's `covers` relation | a Branch on `sum.tag` of a source with one arm per tag, or a chain the concept record's `view` marks as one match |
| 6 | Sequence (computation) | composition | the ω chain of a region | each region's ω chain, drawn as one Sequence over its nodes with effects, in order |
| 7 | Compose (computation) | composition | chained Applies, each output the only input of the next | a chain the concept record's hints mark as composed; otherwise the Applies themselves |
| 8 | Abstract (computation) | frame | a function in `defs`, and the guards, step bodies and `main` that name functions | each function as a subgraph; purity from its effect class |
| 9 | Import (computation) | frame | resolved by the lift to group keys; an unresolved import is kept | a node per referenced key whose origin, in the concept record, is another module or package |
| 10 | Acquire (resource) | composition | an Apply of a capability that returns a linear handle, in host context; in a step, an output request whose result arrives as an event | an Apply of `cap` whose signature returns a linear handle |
| 11 | Release (resource) | composition | a drop where a region ends (an owned value that nothing uses), or an Apply of a capability that consumes the handle | each drop; each Apply that consumes a linear handle |
| 12 | Observe (resource) | composition | `emit`, when the source's observable output contains the value | an Apply of `emit` |
| 13 | ErrorPropagate (error and contract) | encoded as Halt; also a composition | node Halt (a trap that keeps the event); a sum result with a Branch (a recoverable error); `catch` (a handled error) | each Halt; each `catch` not drawn as Transact |
| 14 | CodeBlock (error and contract) | never in an applied Form | none: text in another language is lifted by that language's table, and a foreign function is a capability | drawn only in the view of a kept lift, at the span of each kept construct, from the lift report |
| 15 | Spawn (concurrency) | frame | a task region, a `spawn` output request, the task events of 6.8; threads that share memory are kept | each task region with its spawn requests |
| 16 | JoinAwait (concurrency) | frame | a step on a completion event | each step whose event is a task completion |
| 17 | YieldSuspend (concurrency) | frame | a resume point of a step-split region, or a frame value with a pure resume definition | each resume point and each resume definition |
| 18 | Listen (concurrency) | frame | a step on a declared event | each event with its steps |
| 19 | Transact (data and state) | composition | `catch` around a function whose arguments are restored on failure (W-TX-1, F.4.4) | a `catch` whose failure arm returns the call's arguments unchanged |
| 20 | TypeDefine (data and state) | frame | the `types`, `events` and `errors` tables; a type with a predicate is a refined type | one node per declaration |
| 21 | FeedbackDelay (temporal) | frame | a field of S, updated by the steps of its tick source; a delay of more than one tick is a chain of fields | one node per region field, closed by a Feedback wire |

Count: 5 kinds are encoded nodes (0, 1, 3, 4, 13); 8 are compositions the view recognises (2, 5, 6, 7, 10, 11, 12, 19); 8 are frame constructs (8, 9, 15, 16, 17, 18, 20, 21); and 1 (14) never appears in an applied Form and is drawn only from the lift report.

**The 11 wire kinds** are carried as the table of §8 states: 7 are labels derived on edges (Move, Borrow, MutBorrow, Copy, Error, Lazy, Feedback), 2 exist only in the frame (Channel, Stream), and 2 are never edges, because the lift normalises them (Shared, Weak). The view draws each edge with its derived label, by the precedence of §8, and each region's ω chain as one Sequence node.

**The kernel's per-kind attributes.** Each attribute of a kernel kind, and each of its values, goes to exactly one of five places: **meaning** (a catalogue entry, a runtime definition or a part of the encoding, so in the key); **frame** (a frame construct, also meaning); **derived** (computed from the Form, so it adds nothing); **hint** (the concept record, never meaning); or **kept** (the values named, kept with their reason). None becomes a boolean or a mode setting. The placement table is generated with the recognition table from the kernel's one source, so an attribute or value without a place fails generation. The split between meaning and hint is checked, not assumed: the emit-agreement law (§20, below) must hold with and without every hint.

**The view.** The view is a total, deterministic function `view: (rce-form/1 | none) × rce-concept/1 × rce-lift-report/1 → rce-graph-view/1`. It takes the Form when the lift applied one, and none for a kept lift, whose applied groups and kept constructs the lift report lists. For each node it takes the first row of the table above whose pattern matches, the more specific compositions tried before the encoded kinds (Match before Branch; Transact before ErrorPropagate; Mutate, Acquire, Release and Observe before Apply). Where the concept record's `view` names a kind whose pattern also matches, that kind is drawn. Attributes come from the pattern, and from the concept record's hints where the pattern does not fix them. An Apply's operation name comes from the concept record when the lift recorded the source's operation, and otherwise from the generated operation table of §4. The recognition table is generated from one source for every view of the kernel, so no hand-kept copy of the 22 and the 11 remains.

**The inverse.** `unview: rce-graph-view/1 → rce-form/1 × rce-concept/1` maps each view node of a kind to that kind's pattern, each frame kind to its frame entry, each wire to a source, and each operation name back through the operation table; it returns the candidate Form and a concept record that holds the names, hints and view kinds the drawing carries. It gives no Form, and the edit is kept, for a wire whose drawn kind differs from the kind derived for it, for an unknown kind, attribute or operation name, and for a CodeBlock, which no applied Form holds. The candidate Form is made canonical and goes to the gate. The view-inverse law (§20, below) requires that unview(view(F, c, r)) give F byte for byte, with a concept record that agrees with c on everything the view draws, for every Form F the corpora produce, and that view(unview(G)) = G for every view G the view itself produced. An editor that reads and writes `rce-graph-view/1` offers only the wire kinds the rules of §8 derive for a port and a type, from a legality table generated from those rules, so a wrong wire is never offered and, if written, is kept by unview. RCE's IDE has a graph panel today that edits RCE text, each graph operation checked by `rce check` (`ide/crates/rce-ide/src/graph_edit.rs`); no editor of `rce-graph-view/1` exists yet.

**Why the encoding holds five kinds and not 22.** With all 22 in the bytes, Mutate and an Apply of `list.set`, Match and a Branch on a tag, and Transact and a `catch` would be two spellings of one meaning, so two programs with one meaning could have two keys, and stored wire kinds would state twice what port modes already state. A 22-kind encoding needs canonical rules that pick one spelling per meaning, which reduces it to these five in effect. The 22 and the 11 stay the closed vocabulary of every view and of editing, and the generated, checked conversion with its inverse is the table above. This is open question 2 of §24 (below).

## 13. `rce-core/1` as the schedule, and the construction

Each construct of the core maps onto a construct of the Form, with one exception, exits: a `ret` or a `break` nested inside other statements, and a `break` out of a `block` that is not a loop, have no node of their own. The construction restructures them into Branch outputs and Iterate carried values (an exit flag carried out of the enclosing Branches and Iterates). The schedule writes no such exit. So construct ∘ schedule is the identity on canonical Forms, the direction the round-trip argument of §14 (below) uses; schedule ∘ construct is not the identity on the core.

| `rce-core/1` (ONE-2) | `rce-form/1` |
|---|---|
| atoms `vN`, literals | sources; Bind literals |
| `let`, `prim`, `call`, `catch` | sources; Apply |
| `var`, and `set` of a local | removed: rebinding, Branch outputs, Iterate carried values |
| `if` | a Branch with 2 arms |
| `block`, `loop`, `break d` | an Iterate, from the bounded loop shapes the lowering writes (count: `run_loop`; measure: `for` over ranges and lists); a break of depth d > 0 carries an exit count (ONE-15); a `block` that is not a loop, and a `break` out of it, become Branch outputs |
| `ret` | the region's outputs; a nested `ret` is restructured |
| `halt` | Halt |
| `catch`, then the Branch that halts again on the fuel trap | `catch` (§6) |
| `drop` | none: an owned value that nothing uses (§8) |
| none in `rce-core/1`; a host statement in `rce-core/2` | an Apply of `cap`; until `rce-core/2` exists, the schedule keeps a Form with a `cap` node |
| `out`, and the global output list | `emit` on ω (the lowering writes each `emit` through a call of a per-type runtime definition, so the construction maps it one to one) |
| calls of `rt.tick`, `rt.charge` and the budget counters | `fuel.tick`, `cost.charge` and the budget entries on ω |
| globals of constants and of iterator buffers | memo cells of ω, named by their definition |
| `io.byte`, `mem.get`, `mem.set` | none: not in the catalogue; printed carriers only |
| W1, W2, W3, W4, W5, W6 | FG5, the structure of regions, FG6, FG8, C9 and the keys, and nothing (there are no globals) |

**The schedule.** For each group in key order, and each function in group order:

- the parameters are the region inputs other than ω, which the core does not have; locals are numbered in textual order, the core's rule (`one-emitter.md` section 3.4);
- nodes are written in index order: Bind as an atom; an Apply of an entry as `prim`, where a read-return output names the operand's local; `call` as `call`; `catch` as `catch` followed by a Branch that halts again when the caught halt is the fuel trap; `cap` as a host statement of `rce-core/2`, made in the phase whose lift first needs a capability, until when the schedule keeps the Form (exit 2); `emit[T]` as a call of the per-type runtime definition that shows the value and appends it to the output global; `fuel.tick` and `cost.charge` as calls of `rt.tick` and `rt.charge` over the fuel and cost globals; `memo.get` and `memo.put` as reads and writes of one global per cell. The observation driver resets all of these at the start of every observation, as the lowering does;
- a Branch with 2 arms becomes one `var` of its declared type per output other than ω, then `if sel { arm 1 } else { arm 0 }` with `set` of the outputs; for an `int[0, 1]` selector the test is `int.eq(sel, 1)`, since the core's `if` takes only a `bool`; with n arms, the same over a chain of `if` on `int.eq(sel, k)` for k = 0 … n − 2, the last arm unconditional;
- an Iterate with a count bound becomes the shape of `run_loop`: one `var` per carried value except ω, of the type the Iterate declares for it, and a counter; then `loop { test region; set the carried values from its outputs; if not k { break 0 }; raise the counter; if bound < counter { halt "loop bound" }; body region; set the carried values from its outputs; if not k { break 0 } }`; an Iterate with a measure becomes the same loop with no counter and no bound test;
- a Halt becomes `halt`;
- at the end of each region, a `drop` of every owned value that nothing uses, in index order; a path that ends in `halt` needs none (W3);
- the frame becomes one function per region, the δ function, the initial state, the log chain and the observation driver, which the lowering already writes; the driver prints the observations of the module's vectors, labelled from its concept record.

The schedule is a function of the Form, so the same Form always gives the same core bytes, and its result is checked by W1 to W6; a failure is a defect of the tool (exit 6), as the branch's `core_of` treats a lowering that is not well formed.

**The construction.** Core statements become regions by symbolic execution. An environment maps each local to a source; `let`, `prim`, `call` and `catch` add nodes; a `catch` followed by the Branch that halts again on the fuel trap becomes one Form `catch`; `var` and `set` of locals rebind. An `if` becomes a Branch whose captures are the outer sources the arms use and whose outputs are the variables either arm sets, plus ω when an arm has an effect. A core `loop` becomes an Iterate when it has one of the bounded shapes the lowering and stage B write. A read of an owned value takes the read-return output as the value's new source. The core's `drop` adds no node. Calls of `rt.tick` and `rt.charge` become `fuel.tick` and `cost.charge`; reads and writes of the globals of constants and iterator buffers become `memo.get` and `memo.put` on the cell of their definition. Then the canonical rules make the result canonical.

The construction keeps a module when a loop has none of the bounded shapes, or when a global is not part of the working context that ω holds. The loops the lowering writes for RCE's own loops have these shapes, but some loops of the runtime in the core do not: `rt.dec_limbs` divides a list of base-2^16 digits by ten until the list is empty (`one/rt.rs`), and the float runtime scales a value while a test on it holds and prints digits until they read back (`one/rtf.rs`). Each such loop is rewritten with a count bound computed before it from its operand's size or from the exponent range of binary64 (for `rt.dec_limbs`, five decimal digits per base-2^16 digit, plus one), a bound the loop cannot reach, so no trace changes; until then the construction keeps the modules that use them, each with the reason `loop-shape`. The same construction turns the lowering into the lift of RCE: RCE text, then the checked module, then the core (existing), then the Form (new).

**The Form and the frame of an RCE program** are identical by construction, because such a program's state is the product of its region records (6.1): the frame is the canonical signature of 6.1 written as tables. The canonical signature also covers the engine profile, whose S = Σc × Σd adds a clock, an input count and relations; the frame has no table for those, and the engine's readers move to `rce-form/1` only by open question 13 of §24 (below).

## 14. Why the form is closed under lift and emit

The claims below are arguments, not proofs. Each is checked by the laws of §20 (below) on corpora, and §22 (below) lists the statements that a Lean proof would carry.

**Claim L (closure under lift).** Every language whose defining reference gives each program a computable operational semantics over finitely representable configurations has, for every program, an exact image in `rce-form/1`.

1. Write the semantics as a step relation on configurations (a program counter, an environment, a heap, stacks, pending input). A configuration is a finite tree of integers, byte strings, lists, tuples and sums, which the body types represent, unbounded integers included as lists of digits.
2. One small step is a bounded computation on a configuration, so it is a total function: a successor configuration with zero or more outputs, or a stuck configuration, which becomes a Halt. A total, bounded step is what a step body is.
3. Take S = the configurations, E = the program's inputs plus `Tick`, δ = one step per `Tick` (and the step that consumes an input per input event), and λ = the step's outputs. This Form is the program's semantics by construction; D logs the inputs, and replay derives every state again.
4. Where the defining reference leaves a choice open (the interleaving of threads, an unspecified order), the choice becomes an event, so the Form's behaviours under all event logs are exactly the behaviours the reference allows, provided the choice events stand for exactly the choices it leaves open. Where the source's own mechanism is ported with its parameter (a hash seed, a scheduler seed), the event carries that parameter, and each value gives the one behaviour the source gives for it. Where the reference leaves behaviour undefined, it allows every behaviour, so a trap that keeps the event is one of the behaviours it allows.
5. So an exact image exists for every run whose configurations stay within the N9 limits. A run that passes a limit is kept with the limit named, never answered otherwise. A table does not use this small-step image when it can do better: it writes large exact steps (whole bounded functions, loops with derived bounds) and splits steps only where the source has a suspension point or a loop with no derivable bound.

Claim L does not say that tables exist for a language (that is engineering, measured per language and direction), that the image is cheap, or that it holds past the N9 limits.

**Claim E (closure under emit).** Every program in `rce-form/1` prints in every language whose print table meets the floor and has a row for every capability the program uses; a language that only states data prints the parts of a Form it can state.

1. The schedule maps every Form without a `cap` node to `rce-core/1`, totally and deterministically. A Form with one waits for the host statement of `rce-core/2`, so until then the claim covers Forms without capabilities.
2. Every construct of the core outside the floor has a definition in the core, down to the floor (output, control, locals, functions, integers below a word of at least 256, aggregates; ONE-10), through carriers and control definitions (ONE-14, ONE-15). These definitions are `one-emitter.md`'s; none is built, so this step rests on that proposal's worked tables until the printer implements the definitions and their vectors pass.
3. A table is complete for a program when every construct the program reaches prints, a least fixpoint the printer computes (ONE-9). A complete table therefore prints every Form whose capabilities it has rows for. A capability is a host operation: a target that cannot perform it cannot print it, so a missing `cap` row keeps the emit (exit 2), and that is not a failure of the claim.
4. A language that only states data (a schema language, a configuration format) has no output or control of its own; its emit is total on the types and values of S and E and keeps the rest (ONE-10).

**Claim R (the round trip).** For a language L whose print table is injective on the constructs a Form reaches, and whose lift table includes the inverse rows generated from that print table, lift_L(emit_L(F)) has the key of F.

1. emit_L = print_L ∘ schedule; and lift_L, restricted to text that print_L wrote, is construct ∘ unprint_L, where unprint_L is the print table read backwards.
2. unprint_L ∘ print_L is the identity on the core when the rows are injective and the grammar parses them unambiguously; both are checked per table, by vectors and by grammar agreement.
3. construct ∘ schedule is the identity on canonical Forms: the schedule writes each node once, in canonical order, the construction derives the same sources again, and the canonical rules restore the same canonical form.
4. So the keys are equal. This is the lens law of `emit-targets.md` section 0 and stage G6 of ONE-19, derived from the structure of lift and emit.

Injectivity is a real premise. A language's own table prints its runtime definitions natively through `lib` rows (Python's addition, say, as `{0} + {1}`); if the same table also prints the core's `int.add` at a word type as `{0} + {1}`, the two rows are not injective, and Python's lift reads `a + b` back as Python's addition. The generated inverse rows therefore include **carrier rows**: rows that recognise printed runtime-library code and carriers by the keys named in the provenance line, and lift them back to the entries they print. A table whose rows still collide fails its table check, unless the operand types tell the rows apart. A Form whose emitted text does not lift back to the same key through L fails the Form-lens law for that case, and the case reports both keys.

**Claim C (no pairwise translators).** A translation from L to M is emit_M ∘ lift_L. It inherits the laws of both rows, and no code from L to M exists to drift. N lift tables and M print tables replace N × M translators.

## 15. What is closed and what is open

| Layer | Status | Changes by |
|---|---|---|
| The five node kinds and their attribute schemas, the port modes, the body types, the rules on frame types, the canonical rules, the gate rules, the frame tables, ω, D | closed | a release under C.15 with a new `format` string |
| The catalogue (names, signatures, modes, domain tests, cost and exactness classes, vectors) | closed per version | a catalogue release; a new entry is a change of meaning |
| Functions, declared types, regions, events, steps: every program | open by content address | a new key; no format change |
| Runtime libraries (`lang/<L>/runtime/`) | open by content address | new keys, gated like any program |
| Capabilities, events from a model, fabrics | open at the boundary, typed, versioned, logged | the host contract (Appendix E) |
| Lift and print tables | open, data, keyed by SHA-256 | a new table key; never a change to the form |
| Unobservable names, spans, layout, origin, hints, labels, declared energy | the concept record | free; never hashed |

## 16. Lift: the stages and the per-language files

Every language goes through the same eight stages. The first five read the language's data; the last three are written once for every language. No stage calls a model, reads a clock, the environment or the network, or iterates a hash map into its output; the determinism and model-free laws of §20 (below) check it on every run.

| Stage | Input → output | Rule | Kept when |
|---|---|---|---|
| 1 Read | files → the bytes of each file, in sorted path order | the encoding that `lang.toml` names (UTF-8 for most), and, where the language lets a file name its own encoding, that declaration: for Python, the coding line of PEP 263, decoded with the codec it names from a pinned codec table. A directory is the sorted list of its files that carry the language's extensions, as `rce` reads a package as its `.rce` files | bytes that do not decode under the encoding that applies (exit 3); a PEP 263 codec outside the pinned table (exit 1, reason `encoding`, since the oracle accepts the file) |
| 2 Parse | bytes → tree | the pinned tree-sitter grammar of `grammar.lock`, or the language's own pinned front end where it is code RCE already has (RCE's parser) | any ERROR or MISSING node, with each node's byte range, as RCE's own tree-sitter path reports them (exit 3) |
| 3 Expand | tree → expanded tree, only where the language's program is the expanded tree (the C preprocessor, Rust's `macro_rules!`) | the pinned reference expander named in `lang.toml` by image digest, with the form of its output, recorded in the report; where macros are hygienic, the output carries names with hygiene resolved, so two names the expansion keeps apart stay apart | macros that run arbitrary code (Rust's procedural macros, Lisp's reader macros); an expansion that cannot run deterministically in its image; an expander whose printed output loses hygiene; Zig's `comptime`, which is evaluation and which no pinned tool writes out as a tree |
| 4 Resolve | expanded tree → tree with names, scopes, imports and static types resolved | the scope rows of the lift table; static types by RCE's checker through route T (below) where the language's typing rules coincide with RCE's after desugaring, otherwise by rows that state the language's typing rules or by a typed tree printed by the language's own toolchain (open question 15 of §24, below) | an unresolved name or import |
| 5 Stage A | resolved tree → route T text or route I Lift IR | one row per node kind of the lift table, a pure template; every node kind of `node-types.json` has a row, checked when the table is checked | a node kind whose row is a kept row |
| 6 Stage B | route T: text → core, by print, parse, check and the existing lowering; route I: Lift IR → core, by the transforms B1 to B7 (§17, below) | the checker's parse, link, check and admission, and the lowering; or stage B | a check rule of RCE (its W-code is reported); a construct the lowering does not cover; a construct a transform cannot carry exactly |
| 7 Construct | core → canonical `rce-form/1` | the construction and the canonical rules (§13, §10) | a core loop whose shape is none of the bounded shapes; a core global that is not part of the working context ω holds |
| 8 Gate | `rce-form/1` → applied or kept | FG1 to FG15 | the first failing rule, with its location |

**Route T and route I.** A lift row writes one of two things.

- **Route T: RCE text.** For statically typed languages whose typing rules coincide with RCE's after desugaring, a row writes RCE text whose operations call `lang.<L>`, the language's runtime library, declared as RCE traits with one impl per type. For Go's `+`, `lang.go` declares a trait with wrapping impls for each integer width, `f64.add` for floats and concatenation for strings; RCE's checker monomorphises trait calls and resolves them statically, and an impl may live in its trait's module (5.9), so for these languages RCE's own checker does the source's type resolution. Typing rules that are not RCE's (Go's untyped constants, C's integer promotions, Java's overload resolution) are not reproduced by RCE's checker: route T needs rows that state them, or the toolchain's own typed tree. The text is printed and parsed again, never handed over as a tree, because RCE's parse-time hygiene rules run only when text is parsed; this needs `unparse` to be an inverse of `parse` on every text a row writes, which the row's vectors check. The text is internal: it is neither the identity nor the meaning, which are the graph that stage 7 builds.
- **Route I: the Lift IR.** For semantics that RCE's surface cannot state exactly (dynamic values, exceptions with class hierarchies, closures that bind late, aliasing, generators and `async`), a row writes the Lift IR, which stage B reduces to core statements.

Both routes end in core statements and the same construction, so Forms from either route are in one encoding and can share keys.

**No macros, directives or flags.** The Form has none. A source's macros are expanded at stage 3, and its directives are lifted as part of its meaning: a pragma that selects a semantics (Python's `from __future__ import division`, C's `#pragma STDC FP_CONTRACT`) chooses which runtime definitions the rows call, as the source's own reference says. A source that depends on a build configuration (C's `-D` and `-I` options or its `compile_commands.json`, Rust's `#[cfg(...)]` and the features of the source's crates, Python's version markers) is lifted under one configuration given as data: a file of format `rce-build-config/1`, named as a positional argument of the lift and hashed, its key recorded in the concept record's origin and in the lift report. Another configuration is another lift with another file, never a switch of the table or of the tool, as ONE-12 says of printing.

**No opaque node enters an applied program.** A construct that a table cannot carry exactly is never stood in for by an opaque node, because an applied program with one would keep events the source answers while claiming to comply with the Form. The frontier is drawn at function granularity: every function of the source lifts to a group or is kept, with its reasons and the dependency cone of each kept construct (the statements whose values depend on it); a group whose callees are all applied is applied on its own, with its key, and recorded in the report, so an applied function can be called, emitted and reused; a module is applied only when every group reachable from its steps and `main` is applied; and a foreign function the source calls is part of the source's meaning and lifts to a capability.

**Labels.** A lifted program complies with the Form once the gate applies it; whether it equals its source is a separate fact, recorded as a label in the concept record. The lift guarantees of D.8 become: every lifted definition is labelled `unproven` until an acceptance run executes it in a case that holds, when it becomes `checked`, with the evidence record's id and level (the Form evaluator reports which definition keys each run entered); every construct not carried exactly is kept with its span; code emitted from a lifted Form steps as the Form does (the emit-agreement law); and agreement with the source is evidence on recorded cases, never proof. Labels never enter a key.

**The per-language files.** A language is the directory `lang/<L>/`. Its id names its defining reference and release and, for a language whose definition leaves values to the implementation, the profile of values its table fixes: `python3.13`, `es2020`, `go1.23`, `rust2024`, `scala3`, `c11-lp64-x86_64` (the 2011 C standard on x86-64, under the data model in which `long` and pointers are 64 bits wide), `scxml1.0`. Every file names its format.

| File | Format | Holds | Used by |
|---|---|---|---|
| `lang.toml` | `rce-lang/1` | the name, `defined_by`, extensions, file names, shebangs, manifest rules, encoding, grammar or front end, expander image, runtime module list, class | lift, emit, gate |
| `grammar.lock` | `rce-grammar-lock/1` | the pinned grammar | stage 2 |
| `lift.toml` | `rce-lift-table/1` | the rows and the scope rows | stages 4 and 5 |
| `runtime/*.rce` | RCE modules | the runtime library `lang.<L>`, checked by `rce check` like any program | stage 6, and the print table's `lib` rows |
| `vectors/` | `rce-row-vectors/1` | vectors for every runtime definition, recorded from the pinned oracle at edge values | gate |
| `observe.toml` | `rce-observe/1` | how a native run maps to an observation (§20, below) | gate |
| `print.toml` | `rce-print-table/1` | the print table of ONE-6 to ONE-8, with the `idiom` and `cap` sections of §19 (below) | emit |
| `oracle/<O>.toml` | `rce-oracle/1` | one record per pinned implementation (§20, below) | gate |
| `corpus/manifest.toml`, `corpus/files/`, `corpus/expected/` | `rce-corpus/1`, `rce-obs/1` | the pinned corpus and its recorded expectations (§20, below) | gate |
| `fixtures/pos/`, `fixtures/neg/` | source files with their inputs, and the observations recorded from the oracle (`rce-obs/1`) or the ids of the rows that keep them | one positive fixture per mapped row, one negative fixture per kept row | `rce table-check`, gate |
| `failing.txt` | `rce-failing/1`, named on its first line | the two-way list of ONE-22: CI fails when a listed case holds or an unlisted case does not | gate |
| `STATUS.md` | generated from `rce-registry/1`, headed by the registry's key | level, counts, kept reasons, the rows of `lift.toml` and `print.toml` that lack a clause or a derivation and the rows whose fixtures do not match, the failing list | readers |

**`lang.toml`.**

```toml
format     = "rce-lang/1"
id         = "python3.13"
name       = "Python 3.13"
defined_by = "The Python Language Reference, release 3.13"
class      = "general-purpose"
extensions = [".py"]
shebangs   = ["python3"]
encoding   = "utf-8"                    # with PEP 263 coding lines, through the pinned codec table
codecs     = "sha256:…"                 # the pinned codec table
grammar    = "grammar.lock"
expander   = []                         # Python has no expansion stage
runtime    = ["runtime/int.rce", "runtime/str.rce", "runtime/list.rce", "runtime/dict.rce",
              "runtime/exc.rce", "runtime/value.rce", "runtime/builtins.rce"]
```

**`grammar.lock`.**

```toml
format     = "rce-grammar-lock/1"
lang       = "go1.23"
repository = "https://github.com/tree-sitter/tree-sitter-go"
commit     = "…"                        # 40 hexadecimal digits
generator  = "tree-sitter 0.27.1"       # the tool that generated the parser, as lang/tools.toml pins it
parser     = "sha256:…"                 # src/parser.c as generated
node_types = "sha256:…"                 # src/node-types.json
```

For the language `rce`, the lock names this repository, the commit, and the digests of RCE's own tree-sitter grammar (`tree-sitter/`); RCE's Rust parser is the first front end, and the three-parser agreement of `rce-spec-check` keeps the two in step.

**`lift.toml`.** A row maps one node kind, optionally narrowed by its fields, its children's kinds, its operator text or a resolved type, to exactly one of: a route T template (`rce`), a route I template (`ir`), the kind of trivia under which the concept record keeps the node (`trivia`, `comment` or `layout`), or a kept reason (`kept`). Rows hold templates and evidence and nothing else; no field is a boolean and none selects a behaviour, the rule the print table already follows (ONE-8). A row is a definition derived from its language's defining reference, and its evidence says so: every row cites the clause it implements (`clause`) and states how its template follows from that clause (`derivation`), and a mapped or kept row names its fixtures (`fixtures`). A row is correct when its derivation is stated and every fixture matches its recorded observation byte for byte; nothing else decides it. Rows of one kind are tried in table order, and the first whose `when` holds applies.

```toml
format = "rce-lift-table/1"
lang   = "python3.13"

[[scope]]
kind  = "function_definition"           # a node kind that opens a scope
opens = "function"
binds = ["name", "parameters"]

[[row]]
id         = "py.binop.floordiv"
kind       = "binary_operator"          # must exist in the pinned node-types.json
when       = { operator = "//" }
ir         = "call py.floordiv({left}, {right})"   # the runtime definition decides by the operands' values
clause     = "https://docs.python.org/3.13/reference/expressions.html#binary-arithmetic-operations"
derivation = "numeric operands are converted to a common type, the result is their mathematical quotient with floor applied, and a zero divisor raises ZeroDivisionError; other operands go through __floordiv__ and __rfloordiv__, which the same clause names; py.floordiv states exactly these cases"
fixtures   = ["pos/floordiv.py"]        # its observation recorded from the pinned CPython

[[row]]
id         = "py.comment"
kind       = "comment"
trivia     = "comment"
clause     = "https://docs.python.org/3.13/reference/lexical_analysis.html#comments"
derivation = "a comment runs from # outside a string literal to the end of the line and the syntax ignores it, so it carries no meaning and the concept record keeps it"

[[row]]
id         = "py.call.exec"
kind       = "call"
when       = { function = "exec" }
kept       = "dynamic-code"
clause     = "https://docs.python.org/3.13/library/functions.html#exec"
derivation = "exec runs as Python code a value computed at run time; the lift reads code only before it runs, so the construct is not carried exactly"
fixtures   = ["neg/exec.py"]
```

```toml
# lang/go1.23/lift.toml (route T)
[[row]]
id         = "go.binop.add"
kind       = "binary_expression"
when       = { operator = "+" }
rce        = "({left}).go_add({right})"   # lang.go's trait: wrapping for integers
clause     = "go1.23.0:doc/go_spec.html#Arithmetic_operators"
derivation = "integer + wraps modulo 2^n in the operands' type (Integer overflow); float + is IEEE 754 addition; string + concatenates; lang.go's trait has one impl for each"
fixtures   = ["pos/add_wrap.go"]
```

The lift of RCE itself has no rows: its stage 5 and stage 6 are the checker and the lowering, and its runtime is the runtime in the core.

**`rce table-check LANG`** exits 2 when the table cannot be read or names an unknown node kind, catalogue entry or runtime definition (every `kind` must exist in the pinned node types, and every entry and definition a row names must be known). It exits 1 when one of these checks fails: every node kind of the pinned node types has at least one row (mapped, trivia or kept), so the table is total over the grammar; every row of `lift.toml`, and every entry of `print.toml` that §19 (below) names, cites its clause and states its derivation; every mapped row has a positive fixture that covers the edge values of its operand types; every kept row has a negative fixture that the lift keeps with that row's id. "The table maps a of b node kinds" is then a computed sentence, never a typed one.

**Semantics are rows, not settings.** `emit-targets.md` section 1 lists a language's edge rules as fields of a semantics table: integer width and overflow, division sign, float mode, string unit, map order, coroutine model. Here none of these is a field: each is fixed by which catalogue entry or runtime definition a row calls (Go's `+` wraps through `int.wadd`; Python's `//` calls `py.floordiv`; JavaScript's `+` on numbers calls `f64.add`). Values that a language leaves to its implementation (the width of C's `int`, the sign of `char`, CPython's `sys.maxsize`) are fixed by the profile the table's id names and appear as types and constants in its rows and runtime library, checked by vectors. Implementations that agree on every value the table fixes share the table and are all its oracles; one that differs on any of them is another table (ONE-11, ONE-12). A limit an implementation meets at run time, such as CPython's recursion limit, is not a constant of a lift table: reaching it is the oracle's own resource limit (§20, below). A field nobody reads cannot exist.

**The runtime library is the language's semantics, written once.** `lang/python3.13/runtime/` holds, for example, `int.rce` (unbounded integers as lists of digits, with Python's floor division and modulo), `str.rce` (strings of code points), `list.rce` and `dict.rce` (heap objects; dictionaries keep insertion order), `exc.rce` (exception classes and their method resolution order), `value.rce` (the sum of dynamic values) and `builtins.rce` (`len`, `range`, `print` with Python's `repr` rules). Each definition is content-addressed, gated, and checked by vectors against the pinned oracle before any program that uses it is counted. RCE's own library already has exact pieces a runtime library calls: wide integers (`stdlib/lib/core/exact.rce`) and correctly rounded transcendentals. `ui.ecma` (`stdlib/lib/ui/ecma.rce`), an evaluator of a subset of ECMAScript 5.1 written in RCE, with integer numbers and no loop statement, seeds the JavaScript runtime library and is a definitional reference for programs inside that subset whose numbers stay integers within ±2^53. A language's standard library is lifted from its own source where it is written in that language, and written in RCE only where it is native.

## 17. The Lift IR and stage B

**The Lift IR, `rce-lift/1`,** is the core's JSON form (`one-emitter.md` section 3.4) with these statements added, each removed by a transform of stage B.

| Added construct | Meaning | Removed by |
|---|---|---|
| `["try", body, [[test REF, binder, handler]…], finally]`, `["raise", atom]` | exceptions | B2 |
| `["fnval", REF, atoms…]`, `["callv", atom, atoms…]` | closure values and calls through them | B3 |
| `["loop*", body]`; a module-level `var` | a loop with no bound; the heap and other mutable module state | B4, or B5 |
| `["suspend", kind, atoms…]`, kind `await`, `yield`, `input` or `turn` | a point where the computation waits | B5 |
| `["ret", atom]` in any position, `["continue", d]`, a loop's `else` | exits that are not structured | B6 |
| the type `["dyn", L]` | L's dynamic value, a declared sum of its runtime library | B7 |

**Stage B, in order.**

- **B1 Resolve.** Every call names a catalogue entry, a runtime definition (by key) or a capability; anything else is kept.
- **B2 Exceptions.** Every function that can raise returns `sum(tup(results…), exc)`; every call site is followed by a Branch on the tag, which continues or propagates; `try` becomes a Branch that tests the exception with the handler's test definition (for Python, the runtime's `isinstance` walks the class table, so `except Exception` catches subclasses). `finally` is copied onto every exit of the `try` body: the normal path, each exception path, and each `return`, `break` and `continue` that leaves the body, which B6 then turns into structured blocks. The source's unwinding order is the order of the Branches. An exception that reaches the top of `main` is a value: the runtime library writes the source's error report to `Stderr` and requests `Exit` with the source's status, as the source's runtime does.
- **B3 Closures.** A higher-order function is specialised for each function argument known at its call site, so `map(f, xs)` becomes a function of its own whose key depends only on `map` and `f`. Where specialisation does not terminate (function arguments that vary along a recursion), closures are defunctionalised: a sum of capture records over the program's closures and one dispatch function. Captured variables that the source rebinds after capture (Python's late binding) become heap cells, so `fs = [lambda: i for i in range(3)]; print([f() for f in fs])` prints `[2, 2, 2]`, as CPython does.
- **B4 Bounds and heap threading.** First a count bound is derived for each `loop*` by a closed list of rules: a counted range (its trip count), iteration over a list (its length), a carried integer that falls by at least one per run and is bounded below (its initial value minus the lower bound, evaluated once before the loop). A loop with no derivable bound goes to B5, or carries the count bound the language's table declares, and an exceeded bound keeps the event. Then the heap and other module state are threaded. The heap is a value of a declared type, a list of object records with a variant for a free slot; a reference is an index into it; allocation is `list.push` or the reuse of an index from the runtime library's free list; reading a record moves it out of the list and puts it back after the access (with a catalogue entry that returns the element it replaces, added by a catalogue release with heap threading); object identity (`is`, `===`, pointer equality) is index equality. The heap is threaded as an input and output of every function that touches it, transitively, as RCE passes `&mut` arguments as state, and because it is owned and consumed exactly once, every update prints as an update in place. The runtime library frees an object where the source's own deterministic rule says it is unreachable (an explicit `free` in C, reference counts in CPython); a program whose output depends on addresses or on the timing of collection is kept.
- **B5 Step splitting.** A computation that waits (for input, a timer, a task, a consumer) or loops with no derivable bound becomes a region: its suspension points are numbered; the region's fields are a program counter `pc : int[0, m]`, the live values at each point as a sum, and the heap when B4 applies; each kind of suspension declares its resumption event (`input` waits for `Input(data)`, `await` for the awaited task's completion event, `turn` for `Tick`, `yield` for the consumer's resume); and there is one step per resumption, which restores the live values, runs to the next suspension point, stores the new `pc` and live values, and emits the output request the suspension needs. Given the same inputs and choice events, the region emits the same outputs in the same order as the source, and D logs every resumption. Each `Tick` is a new event with fresh fuel, so step splitting is also how a long-running program stays within the fuel of one input.
- **B6 Structure.** Early returns, `continue` and a loop's `else` become structured blocks with exit flags and exit counts (the control definitions of ONE-15).
- **B7 Specialisation of dynamic values.** Where every producer of a `dyn` value gives one runtime variant (only small integers, only strings), the value is retyped to that variant's type and the dispatching calls are replaced by the variant's operations. The list of rules is closed and deterministic, and it changes cost, never meaning.

## 18. The lift report, the build configuration and the admission vectors

**Outcomes and exit statuses.** Each group is applied or kept, and so is the module. `rce-lift` exits 0 when the module is applied; 1 when it is kept (by the lifter, the construction or the gate; the report names every reason); 2 for a usage error (an unknown language, a malformed table); 3 when the input is not a program of the language (bytes that do not decode, an ERROR or MISSING node, a front end that does not accept it); 5 when a pinned tool or image is missing; and 6 for an internal failure of the tool.

**The lift report, `rce-lift-report/1`,** printed on standard output; its SHA-256 identifies the lift in evidence records:

```
{"format":"rce-lift-report/1", "lang":L, "lift_table":KEY, "runtime":[KEY…], "grammar":"repository@commit",
 "lifter":{"version":V, "commit":C}, "tools":[{"name":N, "image":"sha256:…"}…],
 "sources":[[path, sha256]…], "config":KEY|null, "outcome":"applied"|"kept",
 "module":KEY|null, "concept":KEY|null, "vectors":KEY|null,
 "groups":[{"source":[path,start,end], "key":KEY|null, "outcome":"applied"|"kept",
            "reasons":[{"stage":STAGE, "rule":RULE, "row":ROW_ID|null, "construct":NODE_KIND,
                        "span":[path,start,end], "cone":[[path,start,end]…], "text":TEXT}…]}…],
 "properties":[{"span":[path,start,end], "property":PROPERTY}…],
 "counts":{"by_construct":{NODE_KIND:{"mapped":n, "trivia":n, "kept":n}…}, "by_rule":{RULE:n…}}}
```

`config` is the key of the `rce-build-config/1` file the source was lifted under, or null. `STAGE` is a stage of §16 or a transform B1 to B7. `RULE` is a gate rule (FG1 to FG15), an RCE check code, or a named construct reason: `encoding`, `syntax`, `row-kept`, `macro`, `resolve`, `foreign`, `capability`, `dynamic-code`, `concurrency`, `shared-memory`, `text`, `number`, `semantics` (the source's meaning is a set of results and RCE's is not shown to be one of them, such as a function defined only within an error bound), `choice` (the construct's result depends on a choice the defining reference leaves open, such as the order of a set or the interleaving of threads, and the table carries neither the oracle's rule for that choice, computed from the case's inputs, nor a choice event for it; §20, below), `tensor`, `limit` or `loop-shape`. `properties` lists, per construct, facts a reader may want (a trap for UB, a choice event, a declared bound); they inform readers and never decide a result. Every count is per construct and per rule. Nothing is approximated, so there is no counter of approximations. Nothing in the report depends on a clock, a host or the order of a file system: two lifts of the same inputs with the same lifter give the same bytes.

**The build configuration, `rce-build-config/1`,** is canonical JSON: `{"format":"rce-build-config/1", "lang":L, "defines":[[name, value]…], "include":[path…], "compile_commands":KEY|null, "features":[name…], "markers":[[name, value]…]}`, every list in sorted order. Its key is recorded in the concept record's origin and in the report, and it is read only by stage 3.

**Admission vectors, `rce-vectors/1`.** For every `deterministic` corpus case of a lifted program (§20, below), the harness records the oracle's observation as a case of the module's admission vectors:

```
{"format":"rce-vectors/1", "module":KEY,
 "observations":[OBS…],                    // for RCE sources: constants, checks, zero-argument functions,
                                           // examples and fn main, in the reference trace's order
 "cases":[{"inputs":[EVENT…], "expect":{"outputs":[V…], "outcome":O, "state":HASH|null},
           "source":"oracle:" ORACLE_KEY, "case":CASE_KEY}…]}
```

FG15 checks these cases on the Form evaluator at every admission, so a lift's evidence travels with the program, and an edit that breaks a recorded case is kept unless it replaces the case on purpose, as a logged change of intent. The vectors sit outside the module key, so one computation recorded with different cases keeps one key; their own key, and the bundle key (the SHA-256 of `[module key, vectors key]`), name a module with its evidence. For RCE sources, RCE's own `example` and `check` lines become this file, which keeps the admission rule of 6.9. The reference trace of a Form is defined over the Form, its concept record (which supplies the constant, function and example names the trace prints) and its vectors.

## 19. Emit: the print table, faithful and idiomatic output

Emission is the design of `one-emitter.md` (ONE-1 to ONE-27), fed from the Form through the schedule (§13). This proposal changes three things in it: the printer's input is a Form, read through its schedule; the expected bytes of every gate come from the reference, through `rce trace`, never from the core evaluator (§21, below); and every emitted file names its evidence as keys. The core evaluator becomes one more implementation compared with the reference, the target `core`.

**The print table, `rce-print-table/1`,** is `lang/<L>/print.toml` with the sections and fields of ONE-8 (`lang`, `layout`, `lex`, `type`, `stmt`, `prim`, `lib`, `host`, `helper`), no boolean field and no field that selects a behaviour, plus three sections:

- **`idiom`**: patterns of several nodes printed as one native construct. Examples: a Branch on the tag of a `catch` result as `try`/`except`; a Branch on `sum.tag` as `match`; a heap-threaded object pattern as native objects and fields; a step-split region as an `async` function or a generator.
- **`cap`**: one row per capability key that the target can perform through its own library.
- **`evidence`**: the clause and the derivation of every row and field of the table except `name`, `defined_by` and `catalogue` of the `lang` section, which name the defining reference itself. ONE-7 keeps a row to text and holes, so the evidence sits beside the row, under the row's section and key: `clause`, the passage of the target's defining reference the row relies on, and `derivation`, why the row's text computes the construct it prints. A print row's fixtures are its row vectors (stage G1 of ONE-19).

```toml
# lang/python3.13/print.toml
[prim]
"int.add[int]" = "({0} + {1})"

[evidence.prim."int.add[int]"]
clause     = "https://docs.python.org/3.13/reference/expressions.html#binary-arithmetic-operations"
derivation = "+ on two int operands gives their sum, and Python's integers have unlimited range (the standard type hierarchy, numbers.Integral), so the text gives the exact result of int.add at every integer type"
```

Its oracle records are the files of `oracle/` (§20, below), shared with the lift; ONE-20's `[[oracle]]` entries move there.

**Faithful and idiomatic output.**

- **Faithful output** prints the core through native rows and, where a row is absent, through definitions down to the floor; runtime definitions of other languages print from their RCE bodies. It exists for every complete table.
- **Idiomatic output** also uses `lib` rows (one runtime definition printed natively, keyed by its key, such as `"py.add@KEY" = "{0} + {1}"`) and `idiom` rows, each gated by its own vectors before any program uses it (stage G1 of ONE-19). A native integer row is used only where the operand types' ranges exclude overflow. The choice is a function of the Form and the table; no option, environment variable or directive selects a style.
- **Idiomatic printing never changes the trace.** The emit-agreement law (§20, below) holds with and without the `idiom` rows and the concept record's hints.
- **The inverse property.** A language's own table has a `lib` row for every definition of its own runtime library. Then a Python program lifted from Python and emitted back prints Python's own operators, objects and exceptions, not runtime definitions, which is what the source-lens law checks. A table that lacks some of these rows still prints correct, faithful text.
- **Hints choose, never decide.** Where two `idiom` rows apply, the concept record's hint (the source construct) chooses; with no hint, the first row in table order is used.

**Carriers and the floor** apply as ONE-10, ONE-14 and ONE-15 state them, with four additions: a declared type prints through the carriers of the tuples and sums it unfolds to, one level per value, so recursive types need no new row; ω is erased by the schedule and never printed; a carrier for IEEE 754 values with infinities and NaN prints through software binary64 where a target's floats are not binary64; and a carrier's evidence is library evidence, tested once against the reference on the vectors of stage G2 of ONE-19 and inherited by every table that prints it.

**Two entries in every emitted program.** `trace` takes no argument and prints the trace of the module's vectors, the gate's expected bytes (ONE-5). `host` reads events as canonical JSON lines on standard input and writes λ outputs and D entries as canonical JSON lines on standard output (framing `rce-host-stdio/1`); for a command, a thin driver maps `Start(args, env, stdin)` and `Input(data)` onto the process's arguments, environment and standard input, and the output requests `Stdout`, `Stderr` and `Exit` onto its streams and exit status, so an emitted Python program is an ordinary Python command. A printed program selects its entry by a positional word: `PROG trace`, `PROG trace-at FUEL`, `PROG host`.

**When an emit is kept.** An emitter that cannot print a construct exactly keeps the program and writes nothing (F.4.4), with exit status 2, as 10.1 gives it to an unknown emit target or a construct an emitter does not translate: an unknown language; an incomplete table, with the missing floor rows named; a capability with no `cap` row; a language that only states data asked to print control; a construct the schedule cannot yet write. An emitter never substitutes zero, a comment, a stub, a default case or a placeholder for a construct.

**Byte stability.** Printed bytes are a function of the module key, the concept record's key, the table key, the runtime keys, the catalogue key and the printer's commit, and of nothing else. Tables are read into canonical JSON before use (their key is the SHA-256 of that JSON, ONE-6), every collection is ordered, helpers are printed once in a fixed order, and display data is pinned (ONE-18).

## 20. Acceptance by definition and exact comparison

**The principle.** A lift or an emit is accepted by definition and exact comparison, against the language's defining reference and against its own implementation. The defining reference enters through every row, which cites the clause it implements and states its derivation (K4, below), and through the case classes below where the reference leaves behaviour undefined. The implementation enters as the oracle: pinned, it runs the source; RCE's reference runs the Form; and a case holds when the two observations are equal byte for byte. For the language `rce`, the defining reference and its implementation are both the reference interpreter. Every admitted case is compared. No case is a deliberately wrong program or row, and no case is set aside from the comparison. Where the defining reference leaves behaviour undefined, acceptance follows from that definition (the case classes below); where it leaves a choice open, the case is compared under the oracle's pins, and a construct whose choice the table does not carry is kept with the reason `choice` (§18); and where a construct cannot be compared exactly, it is kept with its reason and counted as kept, never as holding.

**Observations, `rce-obs/1`.**

```
Obs = {"format":"rce-obs/1", "profile":P, "case":CASE_KEY,
       "status":int,               // process exit status (command profile); 0 otherwise
       "stdout_hex":HEX,           // exact bytes, hexadecimal, nothing trimmed
       "stderr_class":TEXT,        // from observe.toml; "" when nothing was written
       "outputs":[V…],             // λ values in order, canonical JSON
       "outcome":"done" | "refused:" X | "trap:" K,
       "d_head":HEX | null,        // event profile, when both sides keep D
       "entered":[DEF_KEY…]}       // Form side only; never compared
```

Two observations are equal when every compared field of the profile is equal byte for byte. Nothing is trimmed or normalised after the fact, with one declared exception: the `query` profile compares rows as a sorted bag unless the query fixes their order, because the dialect's definition leaves that order open. Whitespace, trailing newlines and zero bytes count.

| Profile | Inputs of a case | Oracle side | Form side | Compared |
|---|---|---|---|---|
| `command` | arguments, standard-input bytes, an environment record, a read-only fixture directory | the oracle record's `run` command in its image, under its pins | `Start(args, env, stdin)`, then `Input(data)` events for an interactive program, then `Tick` events until the Form emits `Exit` or a limit of 10^7 ticks is reached | `status`, `stdout_hex`, `stderr_class` |
| `function` | an entry name and an argument vector in canonical JSON | a driver from `observe.toml` that calls the entry and prints the canonical JSON of the result, or the error class | `rce call FORM ENTRY ARGS` | `outputs`, `outcome` |
| `event` | an event sequence | the oracle's own event interface through its driver (an SCXML processor, an actor runtime) | `rce run` on the sequence for its outputs and outcomes, and `rce chain` for its D chain | `outputs` and `outcome` per step; `d_head` when both sides are RCE |
| `query` | a database fixture (schema and rows) and a query | a fresh container of the pinned service, loaded with the fixture | the lifted query applied to the fixture relation in S | `outputs`, `outcome` |
| `data` | document bytes | the reference parser, printing the canonical JSON of the parsed value | the lifted value as canonical JSON | `outputs` |
| `hdl` | per-cycle input vectors | a pinned simulator with a pinned test bench | one δ step per clock edge | per-cycle `outputs` |
| `kernel` | input buffers | a device or software adapter behind the fabric exactness probe (HC-12, HC-13) | the lifted definition | output buffer bytes in `outputs` |

**`observe.toml`, `rce-observe/1`,** maps a native run to these fields: for example, an uncaught Python exception gives exit status 1 and the class `exc:<Name>` from the traceback's last line; a process ended by a signal gives `signal:<n>`; the oracle's own resource limits (`RecursionError`, `MemoryError`, a stack overflow) have named classes. Only the class of standard error is compared by default, because messages are not part of most languages' definitions; a row of `observe.toml` may declare that the text is compared too.

**Case classes**, decided once, when a case enters a corpus, from the oracle's behaviour and never from anything the lift says:

- **`oracle-nondet`**: the oracle runs the case twice under its pins and the two observations differ. The oracle is then not a function of the case's inputs, so the case is not admitted.
- **`nondet:<source>`**: the case's observation changes under one of the perturbations the oracle record lists (a clock offset, another order of the environment, another order of directory entries, another thread count). The case reads something its inputs do not name, so its observation is not a function of its inputs either, and it is not admitted, as with `oracle-nondet`; the admission report names the source. The program may enter the corpus again as a new case once that source is one of its inputs: a seed the source's own mechanism uses (a hash seed in the case's `env`, a scheduler seed) or a value the case's inputs carry, which the oracle receives under its pins and the Form receives in its `Start` payload or as a choice event, so that no perturbation changes the observation.
- **`ub`**: for a language with UB, the case runs under the sanitizers the record names, and it reaches UB. The defining reference allows every behaviour there, so by definition the case holds when the Form keeps the event with the row's UB reason at the point the sanitizer reports, with equal observations before it.
- **`resource`**: the oracle's observation is one of the resource-limit classes of `observe.toml`. The point where an implementation exhausts its own resources is not fixed by the defining reference, so by definition the case holds when the Form keeps the event with a resource reason, or gives the oracle's observation.
- **`deterministic`**: every other case. It is compared byte for byte. Where the defining reference leaves a choice open (the order of a set, the interleaving of threads) and the oracle's pins fix it, the Form must give the oracle's observation; a lift whose table carries neither the oracle's rule for that choice nor a choice event for it keeps the construct with the reason `choice` (§18), and the case is kept.

Only the last three classes are admitted.

**Results**, exactly one per case and stage:

- **holds**: the comparison of the case's class holds.
- **differs**: the comparison does not hold; the record gives both observations and the first differing byte. A lift that says an input is not in the language while the oracle accepts it, and a target that does not accept text RCE emitted, also differ.
- **kept**: the lifter, the emitter or the gate kept the case with a named reason. It never counts as holding.
- **kept for a resource**: the lift applied a module, and in a case of class `deterministic` or `ub` the Form kept an event with a resource reason (loop bound, recursion bound, budget, size limit, value limit, evaluation depth, fuel, tick limit) where the oracle answered. The observations differ, so the case does not satisfy the lift-agreement law (below); it is counted in its own column, never counts as holding, and blocks level 4 (below), so a lift defect that makes a loop run on can never pass as a level.
- **internal**: a panic, an internal failure of an RCE tool (exit 6), or an RCE tool that did not end within the case's limit.
- **not-run**: a stage could not run because a pinned tool, image or input is missing. It fails the run.
- **oracle-fail**: the oracle did not reproduce, for an admitted case, the observation recorded at admission. It fails the run, because the instrument, not the lift, is broken. A source that raises an exception on its oracle is not an oracle failure: its observation (exit status and error class) is compared like any other.

A halt keeps its whole event, so the output a batch command writes before a halt is not in that event's λ: for a batch command, a `ub` case is compared on the kept event's reason and point, and on the output before the halt only where the program is step-split.

**The laws.** Notation: K is a pinned corpus; a case is (p, x), program p in language L with input x; O a pinned oracle of L; R the reference (the Form evaluator for a Form, the reference interpreter for RCE text); obs_O(p, x) and obs_R(M, x) observations; key(M) a module key.

| Law | Statement |
|---|---|
| lift agreement | for every case (p, x) of K whose lift applies a module M, obs_R(M, x) equals obs_O(p, x) under the comparison of the case's class; such a case holds, and a case kept for a resource does not satisfy the law |
| emit agreement | for every module M of a Form corpus with emit_L(M) = q, q's `trace` entry, built and run on every oracle of L, prints exactly the bytes of `rce trace` on M, fuel lines included, and q's `host` entry obeys the host laws (replay equals reopen equals live; a torn tail recovers); with and without `idiom` rows and the concept record's hints. These are stages G1 to G4 of ONE-19, with expected bytes from the reference |
| Form lens | if emit_L(M) = q, then lift_L(q) applies a module M′ with key(M′) = key(M); a kept lift of text that emit_L printed does not satisfy it. For the language `rce`, this is construct ∘ schedule together with parse ∘ unparse |
| source lens | if lift_L(p) applies M, then emit_L(M) prints a program q with obs_O(q, x) = obs_O(p, x) for every input of the case |
| byte determinism | every artefact (module, concept record, vectors, report, emitted text, evidence record) is identical across two runs in separate processes, the second perturbed (another working directory, reversed directory order, another order of the environment, 1 against 8 worker threads, another clock, locale and time zone where no pin fixes them); nightly runs add a second architecture (x86-64 and AArch64) |
| fixed point | emit_L(lift_L(emit_L(M))) = emit_L(M), byte for byte |
| totality | on every corpus input and fixture, lift and emit end with a status of §21 (below); every negative fixture is kept with its expected row id; every kept outcome names its stage, rule, construct and span, and writes no Form and no printed file |
| work-count agreement | emitted programs print the reference's fuel and cost lines at every fuel limit of the sweep (stage G5 of ONE-19), and the Form evaluator, the core evaluator and every emitted program agree on the fuel of every event |
| one form | for each task of the pairs corpus (`lang/pairs/<task>/pair.toml`, format `rce-pairs/1`), the programs of a group declared `same` lift to one key, and for each declared difference `rce equiv` finds a distinguishing event that satisfies the difference's declared predicate; until fuel is placed by one rule for every language, keys are compared with their `fuel.tick` nodes removed and the canonical rules applied again |
| gate | no evidence record cites a module the gate did not apply, and replaying the evidence log through the registry program rebuilds `lang/registry.json` byte for byte |
| model-free path | lift, emit, the harness and every oracle run in an empty network namespace, and the dependency closure of `rce`, `rce-lift` and `rce-gate` contains no client of a model; every evidence record states that its run was free of network access |
| replay | replaying D through δ with every adapter disconnected reproduces every state hash (HC-8, T37) on every corpus run that keeps D |
| view inverse | for every Form F the corpora produce, with its concept record c and lift report r, unview(view(F, c, r)) gives F byte for byte and a concept record that agrees with c on everything the view draws; view(unview(G)) = G for every view G the view produced; an edit that holds a CodeBlock, or a wire whose drawn kind differs from the kind derived for it, is kept |

**Levels.** Each level requires every lower one, and a run gives the highest level whose conditions all hold. Lift and emit have one ladder each; the lens is the level the two share. A level is a number, 0 to 6.

| Level | Lift | Emit |
|---|---|---|
| 0 named | a registry row with a catalogue id (HOPL, GitHub Linguist) | the same |
| 1 built | a pinned front end and a lift table that passes `rce table-check`; the totality law on every fixture | a pinned print table complete to the fixpoint of ONE-9; the totality law on every fixture |
| 2 target-accepted | the front end accepts exactly the inputs the oracle's front end accepts on the corpus, in both directions; where the oracle shares the lift's front end, this shows nothing and is reported as such | every emitted corpus program is accepted by the oracle's own build or parse command |
| 3 rows | the table has at least one mapped row; every mapped row cites its clause and states its derivation, and its positive fixture lifts and holds under the lift-agreement law; every kept row's negative fixture is kept with its row id | every entry of the print table that §19 names cites its clause and states its derivation; stages G1 (row vectors) and G2 (library vectors) of ONE-19 hold on every oracle |
| 4 programs | the lift-agreement and byte-determinism laws on the pinned corpus: every admitted case holds or is kept by a stage with a named reason, at least one `deterministic` case holds, and no case differs, is kept for a resource, is internal, is not run or is an oracle failure; the corpus meets K1 to K7 | stages G3 (examples), G4 (conformance) and G5 (fuel sweep): the emit-agreement, work-count and byte-determinism laws on every oracle: every case holds or is kept by the emitter with a named reason, at least one case holds, and no case differs, is internal, is not run or is an oracle failure; the failing list empty |
| 5 lens | level 4 in both directions, plus the Form-lens, source-lens and fixed-point laws on every case they cover of L's corpus and of the Form corpus, with at least one such case for each law | (shared with the lift column) |
| 6 independent | level 5 against a second oracle that shares no implementation lineage with the first (for Python, CPython and PyPy), or with L's upstream test suite making up the share K6 requires (below) | (shared) |

So a table with no mapped row stops at level 2, and a corpus with no case that holds stops at level 3. A level is never stated alone: it is published only with the counts of the run that reached it beside it (the corpus key and size; the number of cases that hold, that are kept by a stage and that are kept for a resource; the coverage; the oracle names), printed from the registry.

The level function is deterministic and written in RCE: `lang/levels.rce` holds the registry as its state and takes each evidence record's key as an event; its δ checks the record's place in the evidence log and computes the entries it affects again. Running it with the reference over `lang/evidence/log.jsonl` produces `lang/registry.json`, so the registry is itself a program that complies with the Form.

**The oracle record, `rce-oracle/1`.** It extends the record of ONE-20 with the fields that the comparison needs.

```toml
format   = "rce-oracle/1"
name     = "cpython-3.13.1"
kind     = "maintained"            # maintained | emulated | reconstructed | live-service | suite | reference
lang     = "python3.13"
image    = "sha256:…"              # OCI image digest; the whole toolchain is inside it
target   = "x86_64-unknown-linux-gnu"
lineage  = "cpython"               # two oracles are independent only when their lineages differ
version  = ["python3", "-VV"]
parse    = ["python3", "-I", "{parse_driver}", "{src}"]
build    = []
run      = ["python3", "-I", "{src}"]
device   = "octets"
limit    = 10                      # seconds per case
runs     = 2                       # every case twice; the two observations must be equal
suite    = { name = "CPython Lib/test (pinned subset)", commit = "…", result = "…" }

[pins]
env      = { LC_ALL = "C.UTF-8", TZ = "UTC", PYTHONHASHSEED = "0", PYTHONDONTWRITEBYTECODE = "1" }
clock    = "2026-01-01T00:00:00Z"  # the wall clock starts at this instant in every run
network  = "none"
fs       = "fixture-read-only"

[perturb]                          # run once per entry when a case enters the corpus
clock     = "+3600s"
env_order = "reversed"
dir_order = "reversed"
threads   = [1, 8]

[sanitize]                         # languages with UB
commands  = []
```

A case's own `env` adds to `pins.env`, and the Form's `Start` payload receives the same environment record, so a seed the source reads from its environment (`PYTHONHASHSEED`) is an input of the case on both sides. The arguments in `version`, `parse` and `run` belong to the oracle's own command line, outside RCE's toolchain; RCE's commands take none. The rules:

1. **No option that changes meaning.** No oracle command carries an option that changes what the program text means: no contraction, overflow, fast-mathematics or optimisation option (ONE-20).
2. **The image is the toolchain.** The image digest and the output of `version` go into every evidence record, so "the oracle" never means whatever happens to be installed.
3. **The oracle is a function of the case.** A case whose two runs differ is `oracle-nondet`, and a case whose observation changes under a perturbation is `nondet:<source>`; neither is admitted.
4. **Sanitizers classify at admission**, as the case classes state.
5. **The reference oracle.** The kind `reference` is the `rce` binary built from a pinned commit, with its version and commit recorded as the emitted provenance line records them.
6. **Live services** run in a fresh container per case, loaded by the record's `load` command and discarded after the case.
7. **Suites** are corpora with their own expected outputs. When a suite's expectation and a maintained oracle disagree, both are recorded and the case is not admitted, with the reason "suite-oracle disagreement", the conformance suite's rule that tests encode laws, not one implementation (`conformance/README.md`).

Oracle results are cached under the SHA-256 of (case bytes, image digest, command); a cache hit is reused and recorded as such.

**The corpus manifest, `rce-corpus/1`.**

```toml
format     = "rce-corpus/1"
lang       = "python3.13"
oracles    = ["oracle/cpython-3.13.1.toml", "oracle/pypy-7.3.17.toml"]

[[case]]
file       = "files/2f/2f9c….py"    # stored under its SHA-256
sha256     = "2f9c…"
tier       = "T2"                   # T1 basics | T2 algorithms | T3 boundaries | U upstream suite | G generated | P pairs
profile    = "command"
inputs     = [{ argv = [], stdin_hex = "", env = {} }]
expected   = "expected/2f9c….obs.json"   # recorded from the oracle; never typed
class      = "deterministic"        # deterministic | ub | resource: the admitted classes
provenance = { source = "…", item = "sieve", commit = "…", licence = "…" }
```

**The corpus rules.**

- **K1 Recorded expectations.** An expected observation comes only from oracle runs. Its file names the oracle record's key and the hashes of both runs, and `rce-gate corpus-verify` fails on an expected file without them. The only typed expectations are the row ids of negative fixtures.
- **K2 Provenance and licence.** Every file names its source, its commit or address and its licence; code of unknown origin never enters a corpus.
- **K3 Cases are never edited.** An edit is a new file under a new hash. A removal records a reason from a closed list (licence, an oracle found later not to be a function of the case, duplicate).
- **K4 Derived rows.** Every row of a lift or print table is a definition derived from its language's defining reference: it cites the clause it implements, states its derivation and names its fixtures, and the lift or print of each positive fixture records the row among the rows it used. A row is correct when its derivation is stated and every fixture matches its recorded observation byte for byte. A row that lacks a clause or a derivation is listed in `STATUS.md`, and level 3 allows none among mapped rows. A case that differs is a defect, fixed in the row, the runtime library or the tool, and the case stays (K3).
- **K5 Negative fixtures.** Every kept row, and every class of construct kept per program (reflection on computed names, string `eval`, concurrency over shared memory, an unsequenced modification), has a negative fixture; a crash never counts as a correct kept outcome.
- **K6 Breadth.** For level 6, L's upstream test or conformance suite, where one exists, makes up at least 30% of the cases, and every tier is present.
- **K7 Classes.** Only `deterministic` cases count in the columns that hold and that differ under the byte-for-byte comparison; each other class has its own column.

For the language `rce`, the corpus is RCE's own: the 60 single-file examples of `spec/examples`, the conformance tests and the standard library, with the reference as the oracle and no expectation to record beyond its traces. A Form corpus for every emitted language comes from RCE's generator of random programs that the gate applies, which needs only the reference.

**The evidence record, `rce-accept/1`,** one per run, canonical JSON named by its SHA-256:

```
Record = {"format":"rce-accept/1", "lang":L, "direction":"lift"|"emit"|"lens",
  "inputs": {"rce":{"version":V,"commit":C}, "front":KEY, "rows":{ROW_ID:ROW_KEY…},
             "catalogue":KEY, "corpus":MANIFEST_KEY, "oracles":[ORACLE_KEY…], "images":[DIGEST…]},
  "environment": {"arch":A, "os":OS, "network":"none", "perturbations":[P…]},
  "cases": [{"case":CASE_KEY, "class":CLASS, "stage":N, "module":KEY|null, "text":SHA|null,
             "obs":{"oracle":OBS_KEY, "form":OBS_KEY}, "rows":[ROW_ID…], "diff":D|null,
             "work":{"fuel":n, "cost":n, "nodes":n, "source_bytes":n}|null}…],
  "counts": {"holds":n, "differs":n, "kept_stage":n, "kept_resource":n, "internal":n,
             "not_run":n, "oracle_fail":n, "ub":n, "resource":n},
  "coverage": {"programs_applied":r, "node_kinds_mapped":r, "rows_derived":r},
  "kept_reasons": [[REASON, n]…],
  "level": 0 | 1 | 2 | 3 | 4 | 5 | 6}
```

In `coverage`, `programs_applied` is the share of corpus programs whose lift applies a module, `node_kinds_mapped` the share of the pinned node kinds that have a mapped row, and `rows_derived` the share of the table's rows that cite a clause and state a derivation (K4). `work` holds, for each case of a lift that holds, the Form's fuel and its cost units under C.6.4, both exact, and its size in nodes beside the source's size in bytes. No level depends on it.

**The evidence log and the registry.** `lang/evidence/log.jsonl` (format `rce-evidence-log/1`) is a chain in D's entry format (6.1): each line is `{seq, event:{record:KEY}, prev_hash, state_hash, hash}`, the state being the registry. The registry is the fold of the log, and a changed or removed record breaks the chain, which loading detects because verification computes it again. `lang/registry.json` (format `rce-registry/1`) has one entry per language: its id, name and class; its catalogue ids (ONE-27); for lift, emit and lens, the level, counts, coverage, oracle kinds and record id; and the cost columns. `lang/REGISTRY.md` and each `lang/<L>/STATUS.md` are generated from it and headed by its key.

## 21. The toolchain

**Principles.**

- **Positional arguments only.** Every command is one or two command words followed by positional arguments. No option, flag or switch exists, so nothing but the command word and the bytes it names selects a meaning, a format or an outcome. Commands that take options today get positional spellings in a separate change.
- **A command is a function of its arguments and the bytes it reads.** It reads no clock, environment variable, network or random source, and calls no model (10.1, principles 1 and 2); the host commands log what they read, as 10.1 states.
- **One checker.** `rce check`, `run`, `trace`, `emit` and the language server use one parser, type checker and gate (10.1, principle 3); `rce-lift` links the same library, and `rce-gate` runs `rce` for every check, so it holds no checker of its own.
- **No build feature selects behaviour.** `rce-lift` and `rce-gate` are separate programs, so `rce` needs no tree-sitter and no optional feature.

**Exit statuses, for every command.** 10.1's three statuses keep their meanings, and distinct statuses follow them.

| Status | Name | Meaning |
|---|---|---|
| 0 | OK | success: applied, printed, recorded |
| 1 | KEPT | as 10.1 states it, in this proposal's words: a program was kept, an example or check failed, a run reached a kept event, or a check command found something to change; also a lift or an edit that was kept |
| 2 | USAGE | as 10.1: an unknown command, a wrong number of arguments, an unreadable file, an unknown language or table, a malformed table, or an unknown emit target or construct an emitter does not translate |
| 3 | NOT-IN-LANGUAGE | the input is not a program of the declared language: bytes that do not decode, an ERROR or MISSING node, a front end that does not accept it |
| 4 | DIFFERS | an acceptance run found a case whose observations differ |
| 5 | NOT-RUN | no result could be given: a pinned tool, image, oracle or input is missing, or the oracle failed |
| 6 | INTERNAL | a defect of an RCE tool: a panic or a broken invariant |

An acceptance run that meets several of these exits with the first of 6, 4 and 5 that applies, and otherwise 0.

**Programs.**

| Program | Where | Build |
|---|---|---|
| `rce` | the crate `rce-spec-check`, `spec/checker` | Rust, no tree-sitter and no optional feature: parse, check, run, trace, the Form layer (`src/form/`), the core (`src/one/`), the printer (`src/print/`), the views |
| `rce-lift` | the crate `rce-lift`, `lift/` | depends on `rce-spec-check` as a library and on the grammar crates of the languages it lifts, each pinned by its `grammar.lock`; needs a C compiler. RCE's own tree-sitter port moves here as the second front end of the language `rce`, with the program fuzzer and the tree-sitter half of the reject-set cross-check |
| `rce-gate` | `conformance/harness` (Python, extending `run.py` and `rcetest.py`), launched by `conformance/rce-gate` | the only program that starts oracle containers and empty network namespaces; runs `rce` and `rce-lift` as programs; reads `lang/`, writes evidence records |

**Commands.**

| Command | Program | Reads | Writes | Exit |
|---|---|---|---|---|
| `rce check FILE…` | `rce` | RCE text or `rce-form/1`, told apart by the `"format"` field | diagnostics on standard error, one line each; for a Form, for example, `FG6 KEY 0 node 12: owned value consumed twice` | 0, 1, 2 |
| `rce check FORM VECTORS` | `rce` | a Form and its `rce-vectors/1` | the same | 0, 1, 2 |
| `rce key FORM` | `rce` | `rce-form/1` | the module key and a newline | 0; 1 when not canonical; 2 |
| `rce run FILE [EVENTS]` | `rce` | RCE text or `rce-form/1`; events in RCE's event syntax | outputs and kept events | 0, 1, 2 |
| `rce trace FILE` | `rce` | RCE text | the reference trace, `rce-ref-trace/1` | 0, 1, 2 |
| `rce trace FORM CONCEPT VECTORS` | `rce` | a Form, its concept record and vectors | the reference trace of the Form, `rce-ref-trace/1` headed by its module key: the bytes every emitted program must print | 0, 1, 2 |
| `rce trace-at FORM CONCEPT VECTORS FUEL` | `rce` | the same, and a fuel limit | the trace at that limit, with the limit recorded | 0, 1, 2 |
| `rce chain FILE [EVENTS]` | `rce` | RCE text; events in RCE's event syntax | the D chains of the module's examples, or of EVENTS, in `rce-trace/2` (today `rce trace --json`, 10.9) | 0, 1, 2 |
| `rce chain FORM EVENTS` | `rce` | a Form and an event list in canonical JSON | its D chain, headed by its module key, in `rce-trace/3` | 0, 1, 2 |
| `rce call FORM ENTRY ARGS` | `rce` | a Form, an entry name, canonical JSON arguments | an `rce-obs/1` observation | 0, 1, 2 |
| `rce emit LANG FORM` | `rce` | a Form, `lang/LANG/print.toml` and its runtime; LANG may also be `core`, `rce` or `dot` | the program text on standard output, `rce-emit-receipt/1` on standard error | 0, 1, 2, 6 |
| `rce view FORM CONCEPT REPORT` | `rce` | a Form, its concept record, a lift report | `rce-graph-view/1` | 0, 1, 2 |
| `rce unview VIEW OUTDIR` | `rce` | `rce-graph-view/1` | `OUTDIR/form.json` and `OUTDIR/concept.json`, a candidate for `rce check` | 0, 1, 2 |
| `rce run-core CORE` | `rce` | `rce-core/1` | the core evaluator's trace: an implementation under test, never the expected bytes | 0, 1, 2 |
| `rce table-check LANG` | `rce` | `lang/LANG/` and its pinned `node-types.json` | the report of totality, fixtures, clauses and derivations | 0; 1 when a check fails (a node kind with no row; a row with no fixture, clause or derivation); 2 when the table cannot be read or names an unknown node kind, entry or definition |
| `rce vectors` | `rce` | the catalogue and the tables | `lang/vectors/`, generated by the reference | 0, 2, 6 |
| `rce lang LANG` | `rce` | `lang/LANG/`, the registry | table keys, the grammar's commit, runtime keys, the level per direction | 0, 2 |
| `rce-lift LANG SRC OUTDIR` | `rce-lift` | `lang/LANG/`; SRC a file, or a directory read as its sorted files | `OUTDIR/form.json` (only when applied), `concept.json`, `report.json` (also on standard output) and `groups/KEY.json` for each applied group; for `rce` sources also `vectors.json` from the module's own examples and checks | 0, 1, 2, 3, 5, 6 |
| `rce-lift LANG SRC OUTDIR CONFIG` | `rce-lift` | the same, and an `rce-build-config/1` file | the same | 0, 1, 2, 3, 5, 6 |
| `rce-gate vectors LANG SRC INPUTS OUTDIR` | `rce-gate` | a source program, its inputs, the language's oracle | `OUTDIR/vectors.json` (`rce-vectors/1`), recorded from the oracle | 0, 2, 5 |
| `rce-gate corpus-admit LANG FILES…` | `rce-gate` | files and their inputs | manifest entries; expectations recorded from the oracle, with perturbation and sanitizer runs | 0, 2, 5 |
| `rce-gate corpus-verify LANG` | `rce-gate` | the manifest | the report of K1 to K7 | 0, 1, 2 |
| `rce-gate oracle ORACLE CASE` | `rce-gate` | an oracle record and a case | an observation | 0, 2, 5 |
| `rce-gate accept LANG DIRECTION REPORT` | `rce-gate` | `lang/LANG/`, the corpus, the oracle images from a local store; DIRECTION is `lift`, `emit` or `lens` | REPORT (`rce-accept/1`) and a line of the evidence log | 0, 2, 4, 5, 6 |
| `rce-gate conform ADAPTER TESTS OUT` | `rce-gate` | an adapter record, a test list, the conformance tests | the conformance report, as `run.py` writes it today | 0, 1, 2 |
| `rce-gate registry` | `rce-gate` | `lang/evidence/log.jsonl` | `lang/registry.json`, `lang/REGISTRY.md`, `lang/<L>/STATUS.md` | 0, 2 |
| `rce-gate registry-check` | `rce-gate` | the same | nothing; exits 1 when they are stale | 0, 1, 2 |
| `rce-gate changed` | `rce-gate` | the repository | runs every gate whose inputs changed, by content key (ONE-22) | 0, 4, 5, 6 |

`rce trace FILE` here names the reference trace. No command of `rce` prints it today: today's `rce trace --json` prints D chains and becomes `rce chain`, and the branch's `rce emit trace` evaluates the lowered core (`one/mod.rs`, `trace_of`) and becomes `rce run-core`. So the expected bytes of every gate come from the reference, and the core evaluator is measured like any other implementation; its conformance traces that equal the reference's become a measured property of the target `core`.

**Pinned tools.** Every evidence record and every emitted provenance line names the versions of the tools that produced it. Some pins exist today: the checker's `rust-version = "1.98.1"` (`spec/checker/Cargo.toml`) with a committed `Cargo.lock`; Lean `v4.12.0` (`spec/lean/lean-toolchain`); the tree-sitter command-line tool `0.27.1` (`.github/workflows/ci.yml`); Node 20 and Python 3.12 in CI (`.github/workflows/book.yml`, `conformance.yml`). The toolchain keeps them in one file, `lang/tools.toml`, which CI installs from and `rce version` names by its key:

```toml
format      = "rce-tools/1"
rust        = { version = "1.98.1" }
tree-sitter = { version = "0.27.1" }
node        = { version = "20.…", image = "sha256:…" }   # an exact release, and the image when a container provides it
python      = { version = "3.12.…", image = "sha256:…" }
lean        = { version = "v4.12.0" }
```

**File formats.** Every file names its format, and a reader stops with status 2 on a format it does not know (C1).

| Format | Content | Defined in | Hashed |
|---|---|---|---|
| `rce-form/1` | a module (§10) | this proposal | yes: the module key |
| `rce-catalogue/1` | the catalogue, `spec/data/form-catalogue.json` (§4) | this proposal, generated from the book | yes: the catalogue key |
| `rce-concept/1` | names, spans, origin, hints, view kinds, desugar map, labels, trivia, energy annotations (§10) | this proposal | its own key, never part of the module key |
| `rce-vectors/1` | admission vectors and observations (§18) | this proposal | its own key; the bundle key |
| `rce-lift/1` | the Lift IR, written for inspection and tests (§17) | this proposal, informative | no |
| `rce-lift-report/1` | a lift's outcomes, reasons, cones and counts (§18) | this proposal | its SHA-256 identifies the lift |
| `rce-build-config/1` | a project's build configuration as data (§18) | this proposal | yes |
| `rce-emit-receipt/1` | an emit's input keys, the SHA-256 of its text, its status and the table's level | this proposal | its own key |
| `rce-graph-view/1` | the flowG view (§12) | this proposal | no (derived) |
| `rce-cap/1` | a capability signature: name, version, core types, effects, contexts | Appendix E, extended | yes |
| `rce-core/1` | the core, the schedule view; `rce-core/2` when the host statement is added | `one-emitter.md` | yes: group keys |
| `rce-ref-trace/1` | the reference trace, headed by the module key when the input is a Form | this proposal | its SHA-256 names it in evidence records |
| `rce-trace/2` | the D chains of RCE text, unchanged | 10.9 | chained by SHA-256, as D |
| `rce-trace/3` | the D chain of a Form, headed by its module key | this proposal | chained by SHA-256, as D |
| `rce-host-stdio/1` | the framing of an emitted program's `host` entry (§19) | this proposal | no |
| `rce-tools/1` | the pinned tools | this proposal | yes |
| `rce-lang/1`, `rce-grammar-lock/1`, `rce-lift-table/1`, `rce-print-table/1`, `rce-observe/1` | per-language data (TOML, read as canonical JSON for keys) (§16, §19, §20) | this proposal | yes: each file's key |
| `rce-oracle/1`, `rce-corpus/1`, `rce-obs/1` | oracle records, corpus manifests, observations (§20) | this proposal | yes |
| `rce-accept/1`, `rce-registry/1`, `rce-evidence-log/1` | evidence records, the registry, the evidence log (§20) | this proposal | records by key; the registry is the fold of the log; the log is chained by SHA-256 |
| `rce-row-vectors/1` | row and catalogue vectors (`lang/vectors/`, `lang/<L>/vectors/`) | this proposal | yes |
| `rce-failing/1`, `rce-pairs/1` | `failing.txt` lists; `lang/pairs/<task>/pair.toml` | this proposal | yes |

A change to a closed layer (§15) is a release under C.15 with a new format string, and so is a change to a format's header (`rce-trace/3` beside `rce-trace/2`). Tables, runtime libraries, grammars, corpora and oracle records are named by SHA-256: a new table version never changes the key of a Form, and a new runtime definition changes the keys of the programs that call it, the intended Merkle behaviour.

**Where each piece lives.**

```
spec/book/ (a new chapter, once admitted)      rce-form/1, the catalogue, canonical and gate rules, table schemas, laws
spec/data/form-catalogue.json                  the catalogue, generated from the book
spec/checker/src/form/                         ir.rs, json.rs, canon.rs (C1 to C11, keys), gate.rs (FG1 to FG15),
                                               eval.rs (the Form evaluator), schedule.rs, construct.rs, view.rs, unview.rs
spec/checker/src/one/                          the core, W1 to W6, the core evaluator, the lowering, the runtime in the core
spec/checker/src/print/                        the table printer
lift/                                          rce-lift: reading, grammars, resolution, rows, report, stage B, front ends
lang/<L>/                                      per-language data (§16)
lang/vectors/, lang/pairs/<task>/, lang/catalogue/, lang/tools.toml
lang/levels.rce, lang/evidence/log.jsonl, lang/registry.json, lang/REGISTRY.md
conformance/                                   the harness, rce-gate, adapters, reports
```

## 22. What admission needs (CONTRIBUTING rules 1 and 2)

`rce-form/1` adds no construct to RCE's grammar: it is an encoding of the Form of 6.1, whose frame is 6.1's canonical signature written as tables. Its rules become normative in a chapter of the book only with the three things rule 1 asks for.

**Examples.** Accepted modules: the Form of every example the lowering covers, written by `rce-lift rce`, with its key and its three equal traces. Kept modules: at least one per gate rule FG1 to FG15, each kept by its own rule, and one per reason the construction keeps a module (a loop shape, a global).

**Lean statements**, linked to the checker by vectors:

1. Linearity of ω orders effects and halts: in a region that passes FG6 and FG7, the nodes with an effect or that may halt form one chain.
2. Totality: in a module that passes FG8, no partial entry is applied outside its domain, and every Iterate and every recursive group ends within its bound.
3. Canonical form: decoding and encoding give the same bytes, and the canonical rules give one representative per graph up to the sharing of plain nodes.
4. The round trip: construct ∘ schedule is the identity on canonical Forms.
5. The frame: for the Form of an RCE program, δ read off the frame equals δ of 6.4.

`spec/lean/RCE/Positional.lean` (T7, the positional form is name-free) and `spec/lean/RCE/Emit.lean` (the bounded loop shape) are the existing proofs these statements build on.

**Checks.** `rce check` on Forms (FG1 to FG15); `rce key`; for the 56 examples the lowering covers and the conformance tests the ratchet covers, the reference interpreter's trace, the Form evaluator's trace and the core evaluator's trace of the schedule are byte-identical, fuel lines included, and the ratchet does not fall; construct(schedule(F)) has the key of F for every one of those Forms; every Form, key and trace is byte-identical in two processes and on x86-64 and AArch64; the 77 canonical JSON vectors pass, and every catalogue entry passes its vectors.

**Composition before addition (rule 2).** The encoding is not a second spelling of anything the Form already says: it is the one spelling, and every other notation, the core and RCE text included, is a derived view. The five node kinds are the core's statement families, and the derived wire kinds are labels, not fields.

## 23. Changes to the specification and the ledger, if accepted

- **A new chapter** of the book defines `rce-form/1` (§3 to §13 of this proposal), the catalogue, the canonical and gate rules, the concept record, the table schemas of §16 to §19 and the acceptance formats of §20; `rce-spec-data` generates `spec/data/form-catalogue.json` from it.
- **6.1**: a note that `rce-form/1` is the encoding of the Form and that the module key names a program.
- **10.1**: the exit statuses 3 to 6 beside 0, 1 and 2.
- **10.5 and 10.9**: `rce trace` names the reference trace, `rce chain` the D chains, and `rce run-core` the core evaluator's run; the expected bytes of every emit gate come from `rce trace`.
- **D.8**: the lift guarantees become the labels of §16.
- **`spec/proposals/LEDGER.md`**: one row, "The one form, `rce-form/1` (`rce-form.md`)", decision open.

## 24. What stays open

1. **What is hashed.** This proposal hashes the graph, with `rce-core/1` as its schedule view. The other choice is to hash the statement core, with a graph view.
2. **The encoding's node set.** Five encoded kinds, with all 22 kernel kinds and all 11 wire kinds restored by the view and its inverse; the other choice is all 22 kinds and stored wire kinds in the hashed bytes (§12).
3. **The shape of lifted commands.** Whether the specification admits a lifted batch program's shape (§9), with the environment and standard input in the `Start` payload, which D logs, or lifted commands keep 4.2's exact shape and read their environment and standard input through capabilities.
4. **Nested options in the positional form.** FG14 keeps a module whose frame types hold an option of a type whose positional form can be null; the other choice is a change to 6.1's positional form, which changes the hashes of states that hold such options.
5. **VALUE_LIMIT.** N9 states it and the reference does not enforce it (no use of VALUE_LIMIT in `spec/checker/src`). Enforcing it in the reference, as an erratum with conformance tests, or changing the text; until then the Form follows the reference.
6. **Memo cells on a host.** Fresh memo cells per input (§5), filed as an erratum to the reference's host under C.15 with a test that a restored host charges each input as a continuous one does, or a cache that lives for a session, as the reference keeps it.
7. **Where fuel is charged.** At the reference's points, as explicit `fuel.tick` nodes, so traces stay equal to the ratchet and the existing emitters stay valid oracles; or per node kind of the Form, with the effect on every trace measured first.
8. **A host statement in the core.** Capabilities need a statement of the schedule view, `rce-core/2`; until it exists, emit keeps a Form with a `cap` node.
9. **The reference for a Form's meaning.** The Form evaluator of §2, a third implementation kept in agreement by the ratchet, or one of the two alternatives set aside there.
10. **The spelling of the kept outcome.** `compliance.md` respells `refused` as `kept` everywhere. The new formats of this proposal (`rce-form/1`, `rce-obs/1`, `rce-lift-report/1`, `rce-accept/1`) should take the spelling the project chooses before any of them is frozen, so none is renamed twice.
11. **Maps.** A balanced tree in the runtime library in place of the linear search over sorted pair lists, measured by fuel counts before and after; the change of fuel counts is a meaning change under C.15.
12. **Tensors.** Every tensor operation stays kept with the reason `tensor` until the tensor proposals of E.11 are decided.
13. **Today's Form JSON and Form IR.** `rce emit form` and `rce unform`, and the Form IR terms of D.7 and `form-ir-2.md`, retired in favour of `rce-form/1`, with the engine profile's readers moved once the frame has tables for its clock, input count and relations.
14. **First-class functions.** Removed at lift (B3) through the first phases, with a measurement of how often defunctionalisation makes keys depend on a program's whole set of closures.
15. **Type resolution for statically typed languages.** Route T where typing coincides with RCE's after desugaring; rows for small, closed rule sets; or a typed tree printed by the language's own toolchain in a pinned image, run only in the gate.
16. **Floats beyond the finite binary64 of F4.** An IEEE 754 carrier library for the infinities and NaN of JavaScript, Python and C now, or a core `f64` with full IEEE 754 binary64 and one canonical NaN once the carrier's cost is measured.
17. **Programs and the harness.** Three programs (`rce`, `rce-lift`, `rce-gate`), with the harness in Python, extending what exists; or one program with optional features, which this proposal sets aside because a feature would select behaviour.
