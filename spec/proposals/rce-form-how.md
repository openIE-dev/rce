# Proposal: the how in `rce-form/1`, one meaning and many checked realizations

Status: proposal. Owner: David Charlot. Extends `rce-form.md` (the one form that lift and emit share), and is derived from 6.1 (the Form of a program), 6.2, 6.3 (the edge rules N1 to N11 and F1 to F9), 6.7 (loops and fuel), 6.8 (concurrency), 5.1, 5.6, 5.7 (arenas), C.6 and C.8 (cost and the reduction tree), D.11 (receipts), Appendix E (HC-12 to HC-14), section 7 (W-LIN-1, W-PAR-1 to W-PAR-3, W-SIMD-1, W-SPAWN-1), CONTRIBUTING rules 1 to 3, and the proposals `rce-form.md`, `one-emitter.md`, `compliance.md`, `one-library-every-fabric.md` and `energy-calibration.md`. Nothing in this proposal has been built. No normative chapter changes with it.

Sources, all read only:

- `openIE-dev/rce` at `origin/main` 2364aa97, and its branches `claude/one-emitter` (123f5eee), `claude/p0-core`, `claude/p0-catalogue`, `claude/p0-emit` and `claude/host-robustness-http-events`, cited by path and commit.
- Goose, a memory-safe systems language by Wouter van Oortmerssen: https://github.com/aardappel/goose at commit cce94b919d27458856ae2685974049eac78debcf, cited "Goose spec" for `docs/goose_spec.md` and "Goose README" for `README.md` at that commit. Goose's compiler was read, not built or run: every Goose behaviour stated here is what its specification says.
- The defining references of the other languages named, cited by URL where they are used: the Rust Reference (https://doc.rust-lang.org/reference/) and the Cargo profiles (https://doc.rust-lang.org/cargo/reference/profiles.html); the Zig Language Reference 0.15 (https://ziglang.org/documentation/0.15.1/) and Zig's overview (https://ziglang.org/learn/overview/); ISO/IEC 9899:2011, through its final draft N1570 (https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf), and the GCC manual (https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html); the Go specification (https://go.dev/ref/spec); the Python 3.13 documentation (https://docs.python.org/3.13/); ECMAScript (https://tc39.es/ecma262/); the WebGPU Shading Language (https://www.w3.org/TR/WGSL/); the RISC-V vector extension (https://github.com/riscvarchive/riscv-v-spec/blob/master/v-spec.adoc).

In plain words:

1. Languages that look alike on the page differ in *how* a program is carried out: where each value lives and when its memory is given back, whether arithmetic and indexing are checked, how data is laid out, how work is spread over cores and vector units, and what all of it costs. Goose keeps every value inline on compiler-managed stacks and has its compiler work out what each reference is rooted in. Rust has the programmer write lifetimes and uses a heap. Zig passes an allocator by hand and chooses safety by build mode. C leaves all of it to the programmer. Each language builds one answer into its syntax, and every program written in it inherits that answer and its limits.
2. The how is real: it decides speed, memory and energy. It is not syntax. Two facts follow. Safety belongs to the how: Goose's promise that a reference never dangles holds because its memory never moves, and the same text on an ordinary heap would not be safe. And in each of these languages a build switch can change what a program means: Rust wraps or panics on overflow by profile, Zig's ReleaseFast turns a checked error into undefined behaviour, Goose aborts in debug and wraps in release.
3. RCE (the Reality Code Engine) separates the two halves. Its Form, P = (S, E, δ, D, λ) of 6.1 (states, events, a total deterministic step, a hash-chained log, outputs), fixes once every choice that can change what a program prints or records, and no build mode exists that changes any of them. Every language lifts into that one meaning, and its how is written down beside the module as data, not thrown away.
4. The how is then chosen per program, per value and per target, and every choice is checked. Goose's stacks, Zig's arenas, Rust's borrows and in-place updates, C's laid-out records and a collector's heap are values of one record, chosen site by site. A choice is applied only when a decidable rule shows it cannot change the meaning, and the emitted program is compared with the Form byte for byte. A target's unchecked mode is used only where a stated rule shows that no check it removes could ever fail.
5. Cost is counted exactly, and energy is claimed only from a meter. Each realization comes with exact counts of the work it does; prices turn the counts into a modelled figure labelled as such; a joule figure needs a meter reading.
6. So RCE is not one more language with one more answer to the how. Goose's model is one value of one field of RCE's realization record, Rust's another, Zig's a third, and a module may use several in different places, each checked where it is used. What no single language offers, RCE offers by construction: the same meaning on every target, and the how chosen for each target and shown not to change it.
7. Where it stands: nothing of this proposal is built. It adds eight meaning items to `rce-form/1` (§4, below), which change the module key and so must land before that key is first fixed, and a realization layer beside the module (§5 to §14, below), which never enters the module key.

## 1. Terms

**Conventions.**

- A bare section number such as 6.1, C.8 or E.2 names a section of the specification (the chapters of `spec/book/`, generated into `spec/RCE-SPEC.md`). Its rules and theorems keep their names (W-LIN-1, HC-12, N9, F1 to F9).
- "rce-form §n" names section n of `rce-form.md`, and its labels keep their names there: domain facts DF1 to DF8, canonical rules C1 to C11, gate rules FG1 to FG15, the evidence levels 0 to 6 of its §20 and the corpus rules K1 to K7.
- §n names section n of this proposal. A reference to a later section says "below". The sections are: §1 Terms; §2 The point; §3 The dividing rule; §4 Meaning additions; §5 The realization record; §6 Derived facts; §7 Target profile, resource vector, bounds and prices; §8 Obligations; §9 How a record is made; §10 The gate rules; §11 Lift; §12 Emit; §13 Acceptance; §14 Hardware and energy; §15 Examples; §16 What the built work shows; §17 Changes if accepted; §18 Phasing; §19 What stays open.
- The one label family this proposal introduces is the gate rules HW1 to HW30, numbered in the order a reader meets them in §10 (below).

**Abbreviations.** Each is expanded here, before its first use.

| Abbreviation | Expansion |
|---|---|
| ABI | application binary interface |
| BRAM | block random-access memory of an FPGA |
| CPU | central processing unit |
| FFI | foreign function interface |
| FMA | fused multiply-add |
| FPGA | field-programmable gate array |
| GC | garbage collector |
| GPU | graphics processing unit |
| HDL | hardware description language |
| IB | illegal behaviour, Zig's term for an operation its reference forbids |
| II | initiation interval of a pipelined loop |
| ISA | instruction set architecture |
| JSON | JavaScript Object Notation; canonical JSON is the byte form of 6.1.1 |
| libm | a platform's C mathematics library |
| MCU | microcontroller unit |
| RAPL | Intel's Running Average Power Limit counters |
| SHA-256 | the Secure Hash Algorithm with a 256-bit digest |
| SIMD | single instruction, multiple data |
| TOML | Tom's Obvious, Minimal Language |
| UB | undefined behaviour: behaviour for which a language's defining reference places no requirement |
| UTF-8 | the Unicode encoding in 8-bit units |
| WGSL | the WebGPU Shading Language |

**Terms from `rce-form.md`, used as it defines them.** Form, applied and kept, gate, complies with the Form, reference interpreter and reference trace, fuel and cost units, the N9 limits, `rce-core/1` and the core evaluator, the Form evaluator, `rce-form/1`, frame, group, region, node, port, source and edge, plain and owned values, the order value ω, the catalogue, runtime library, capability, key and module key, Merkle digest, concept record, schedule and construction, lift and emit, table and row, lift table and print table, defining reference, oracle, clause and derivation, observation, case, corpus, fixture, carrier and floor, heap threading, language id (the name of one lift table's reading of a language, such as `rust2024-debug`). A **domain test** is the test node and Branch that rce-form §7 places before a partial catalogue entry; a failing test leads to a Halt with the exact trap text. **flowG** is the parent of RCE (rce-form §1).

**The how.**

- **The how**: everything about running a program beyond its meaning: memory and lifetime discipline, data layout, how checks are carried out, allocation and growth, schedule, the mapping of concurrent work onto workers, the hardware, and the cost and energy.
- **Realization**: one complete choice of the how for one module on one target.
- **Realization record** (`rce-real/1`): the data that states a realization, kept beside the module and never part of the module key (§5, below). A record the gate applies is **applied under target T**; a module emitted under it is **realized on T**. A kept record never keeps the module: "complies with the Form" stays a property of the module alone.
- **Facet**: one field of a realization record (memory, reads, updates, copies, types, checks, growth, schedule, order, calls, fabric, and the frame's workers).
- **Source record**: the record a lift writes, stating how the source's own language carried the program out. **Target record**: the record an emit uses.
- **Derived record**: the record a fixed recipe computes from the module and the target profile alone; the default for every target. **Faithful record**: the derived record on the `portable` profile (heap places, copies, every test printed, sequential, serial).
- **Derived facts** (`rce-real-facts/1`): storage chains, live intervals, roots, lane, tree and fork facts, layout tables and bounds, computed by the gate and bound to a record by key (§6, below).
- **Storage chain**: the path of one piece of storage through a function's graph. **View**: a read carried out as a reference into a storage chain instead of a copy. **Root**: the storage chain a view or an output is drawn from.
- **Target profile** (`rce-target/1`): the data that fixes a target: instruction set, data model, float controls, mathematics library, build command and flags, stores, lanes, stack bytes, fabric class. **Resource vector** (`rce-res/1`): the exact counts of work one case does on the Form evaluator under one record. **Static bounds** (`rce-bounds/1`): upper bounds on parts of that vector, derived by closed rules. **Price record** (`rce-price/1`): per-target prices of counted work, labelled `est` (§7, below).
- **Obligation**: a named property of a scope from the closed list `rce-obligations/1` (no residual test, no allocation, bounded memory and others). A **claim** is an obligation asserted for one scope, by a source construct or a committed file, and the gate decides it (§8, below).
- **Probe**: a count read from a run of a separately instrumented build of an emitted program, outside the compared observation. **Static build fact**: a value read from the pinned build's own report (frame bytes, remaining checks, escape decisions). Both are admitted only when shown deterministic; otherwise they are **probe-nondet** and not used.
- **Choice classes**, named with words so that no label collides with the levels 0 to 6 or the specification's rule names:
  - a **meaning choice** changes the observation of some program (wrap or trap on overflow);
  - a **resource choice** changes only time, memory, code size or energy, for every program;
  - a **fault choice** changes only what an erroneous program does (an unchecked build that turns a defined trap into UB);
  - a **measure** is physical and not a function of the program text and inputs (joules, seconds, cache misses).
- **Evidence kinds**: a **trace comparison** (observations equal byte for byte); a **rule** (a decidable check on the module, the record and the facts); an **artefact fact** (a deterministic property of the built program: a compile-time assertion, a static build fact, Goose's `checkfree` refusal); a **receipt** (a measurement written as data with basis and provenance, D.11, which never decides an outcome).

## 2. The point: why the Form carries the how explicitly

The public discussion that compares Goose with Rust is the Zig, Rust and C discussion seen from one more side. Goose is "a memory-safe systems language with no heap allocator, garbage collector, or lifetime annotations" (Goose README, lines 9 to 11): a program's memory is the native call stack, a statically computed number of data stacks and static data; lifetimes on one stack are strictly nested; at most one growable value is live per stack and it is the topmost, so growth is a pointer bump that "never moves memory"; the compiler infers what each reference is rooted in; and it emits C (Goose spec, lines 60 to 111). Rust asks for lifetime annotations where references cross functions and uses a heap allocator (https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html). Zig passes an allocator explicitly and selects run-time safety per build mode and per scope, with the slogan "Performance and Safety: Choose Two" (https://ziglang.org/learn/overview/). C leaves storage, aliasing and checking to the programmer, and makes signed overflow and out-of-bounds access undefined (N1570, 6.5 and J.2).

What each language builds into its syntax is one how, and its programs inherit what that how cannot express. Goose cannot hold a growable list inside a growable list, and recursion may not keep growable data across the recursive call (Goose spec, lines 441 to 445 and 2882 to 2907). Rust needs annotations or index workarounds where references would not survive growth. Zig and C buy speed by switching checks off for the whole build or scope, not where a check is shown never to fail.

Two facts follow, and the design rests on them.

1. **Safety belongs to the how, not to the text.** Goose's "never dangles" holds because data stacks never move and only checked shrinks free memory; the same text on a heap that moves or frees storage would not be safe. Rust's borrow rules hold because the heap is accessed only through them. So a language's safety is a property of the pair (its text, its how), and a translation that keeps the text and changes the how can lose it.
2. **A build switch can change what a program means.** Rust's `overflow-checks` makes `+` panic in the `dev` profile and wrap in `release` (https://doc.rust-lang.org/reference/expressions/operator-expr.html#overflow); Zig's ReleaseFast turns checked IB into UB (https://ziglang.org/documentation/0.15.1/#Illegal-Behavior); Goose's signed overflow aborts in debug and wraps in release (Goose spec, section 7); GCC contracts `a*b+c` across statements unless `-ffp-contract=off` is given (GCC manual, `-ffp-contract`).

RCE's answer has four parts.

- **One meaning, fixed once.** The Form fixes overflow by declared range (a trap), one float rounding per operation (F1 to F9), one evaluation order, one reduction tree (C.8), one map order and one set of resource limits. No build mode exists that changes any of them, so safety is never traded for speed.
- **Every language lifts into that meaning, and its how is kept as data.** A Rust, Zig, C, Go, Goose, Python or JavaScript program becomes a module of the one Form; its how is written beside the module as a source record, so emitting the program back to its own language can keep it (§11, below).
- **The how is chosen per site and per target, and every choice is checked.** A choice is applied only when a decidable rule shows it cannot change the meaning (§10, below), and the emitted program is then compared with the Form byte for byte (§13, below). A target's unchecked operation is printed only at a site where a domain fact or certificate shows its check cannot fail.
- **Cost is counted, energy is measured.** The resource vector counts work exactly; prices make a modelled figure labelled `est`; a joule figure is a receipt (§7 and §14, below).

## 3. The dividing rule and the classification

### 3.1 The rule

1. **A how choice that can change an observation is meaning.** It is in the Form and its module key, or in the language id. Two sources that differ in it have two keys. Where a source leaves it to a build mode, each mode is its own language id.
2. **A how choice that cannot change an observation is a realization.** It is data of the realization record and the target profile, never of the module, and the record has its own key.
3. **A how entry is never enforced at run time; a construct that enforces a limit at run time is meaning.** An allocator that can fail observably, a recursion limit a program can catch, an arena bound that traps: each is in the Form. A record entry is decided before any run, so it cannot change an observation.
4. **A fault choice is never a realization.** A target operation that would be UB or unchecked IB on some input is printed only where a domain fact, a certificate or a derived fact shows that input cannot reach it. On the source side a fault is lifted as rce-form §20 lifts UB: a test and a trap, with the case class `ub`.
5. **A measure is never a gate.** Energy, seconds and cache behaviour are receipts, beside the evidence and never inside it.
6. **Every realization claim has evidence**: a rule, an artefact fact, or nothing claimed.

### 3.2 The classification

| Choice | Class | Where it lives | Example |
|---|---|---|---|
| Integer overflow: trap, wrap, saturate | meaning | catalogue entries (the trapping `int.add` with its test, the wrapping and saturating entries) | Rust `dev` against `release`; Zig ReleaseFast; Goose debug against release |
| Division, remainder rounding, shift range | meaning | catalogue entries; Euclidean forms as runtime definitions | Goose's `%` is Euclidean (Goose spec, lines 2016 to 2023) |
| Float contraction, reassociation, excess precision, subnormal flushing | meaning | F1 to F9; a relaxed source construct is kept with reason `semantics` | Zig `@setFloatMode(.Optimized)`; Go fusing across statements |
| Transcendental functions | meaning | the language id names the libm or engine by digest | CPython on glibc; V8's own `Math` |
| Evaluation and effect order | meaning | C5 numbering and the ω chain | |
| Reduction order | meaning | the C.8 tree | RISC-V V `vfredusum`, an unordered sum |
| Map and set iteration order, hash seed | meaning for the order; resource for the hash algorithm once the order is canonical | canonical order, or the source's insertion order as runtime library code; an unpinned seed is not admitted | Go map order; JavaScript `Map` insertion order |
| Thread interleaving, weak memory orders | meaning | the tick semantics of 6.8; shared memory kept with reason `shared-memory` | Rust and C atomics |
| Resource limits a program can meet or read | meaning | the `limits` frame table (§4, below), fuel excepted | Python's recursion limit |
| Arena byte bound | meaning, because it traps | the ω entry `bound.check[n]` (§4, below) | `with arena r bound n` (5.7) |
| Linear handles consumed exactly once | meaning | the `linear` flag of a declared type (§4, below) | W-LIN-1 |
| Allocation failure a program observes | meaning where deterministic; the `resource` case class otherwise | runtime library code that counts bytes by the profile's layout | Zig `FixedBufferAllocator` returning `error.OutOfMemory` |
| Slot reuse read through a stale reference | meaning | the pool's free-list order as runtime library code under heap threading | Goose `reusable` pools |
| Layout read by `sizeof`, an offset, a bit cast, or at a boundary | meaning | lift rows compute it from the source's normative layout; boundary layouts in `rce-cap/2` (§4, below) | C `sizeof`; Zig `@bitCast` of a `packed struct` |
| Destructor with effects | meaning | the lifted drop function called on the ω chain at the source's drop point | Rust `Drop` that prints |
| Host value representation, width, rate | meaning | `rce-cap/2` fields | an 8-bit tick count past 255 |
| Unchecked or unsafe build mode | fault | never a realization | Zig ReleaseFast, C signed overflow |
| Where each owned value lives and when it is freed | resource | record: `memory` facet | Rust heap, Goose stacks, Zig arenas, Python reference counts, Go collector |
| Borrowing instead of copying; update in place; copy on write | resource | record: `reads`, `updates`, `copies` facets over derived roots | Rust `&T` and `&mut T`, Goose rooted references, C `const T *` |
| Data layout inside the program | resource | record: `types` facet | Rust `repr(Rust)`, Goose packing, array of records or record of arrays |
| How a domain test is carried out; whether a certificate removes it | resource with a certificate; fault without | record: `checks` facet | Goose `checkfree fn` |
| Growth policy and capacity | resource | record: `growth` facet | Go `append`, Goose `T[>..]` |
| Schedule: unroll, tile, vectorise, fuse; order of pure nodes | resource under stated conditions | record: `schedule`, `order` facets | Halide and TVM schedules |
| Native call stack, explicit stack, inlining | resource under the depth limit | record: `calls` facet | ONE-15 explicit stacks |
| Worker count, partition of `par`, task-to-thread mapping | resource | record: frame workers | Goose threads |
| Fabric and vector width | resource, for entries on the fabric's exact list | record: `fabric`; target profile | Goose `simd fn`; HC-12 placement |
| Optimisation level and inlining by the target compiler | resource | target profile flags from a closed classified list | HC-14 |
| Seconds, joules, cache misses | measure | receipts | RAPL |

Two pairs of rows show where the line falls.

- **A bounded buffer.** In Zig a full `FixedBufferAllocator` returns an error the program reads, so its byte count is meaning. In Goose a data stack that runs out hits a guard page and the process aborts; Goose's specification does not fix where, so for a Goose source it is the oracle's `resource` class. On the emit side, a bounded store that RCE prints is a resource choice only because the gate proves its capacity is never reached; otherwise the value goes to the heap.
- **Layout.** Inside a program, layout is a resource choice. The moment a program reads `sizeof` or reinterprets bytes, the lift computes the value read from the source's normative layout as runtime library code, or keeps the construct with the new kept reason `representation` where the layout is not normative or not pinned.

## 4. Meaning additions to `rce-form/1`

Each item is meaning, so it enters the module key, the catalogue key or the language id, and each must land before those keys are first fixed.

1. **The `limits` frame table.** `Limits = {"depth":[N, HALT] | null, "loop":"N", "size":"N", "value":"N"}`, every field present, with `HALT = ["trap", TEXT] | ["error", x]` (a trap text, or an index into the module's errors table).
   - `depth` is the greatest number of active calls; the Form evaluator's call rule halts with HALT when a call would pass N. No node is added per call. For `rce` it is the reference interpreter's 20,000 with the trap text "evaluation depth" (`EVAL_DEPTH`, `spec/checker/src/interp.rs:251`), which 6.7 is asked to state. For `python3.13` it is CPython's 1,000 with the declared error `RecursionError`, which a program catches with the existing `catch` call (https://docs.python.org/3.13/library/sys.html#sys.getrecursionlimit). For a language whose limit is in bytes of a growable stack (Go), it is `null`: the Form has no depth limit, every evaluator keeps its own stack, and the oracle's own exhaustion stays its `resource` class.
   - `loop`, `size` and `value` are LOOP_LIMIT and the N9 limits, the specification's values unless the language id's profile states others with clause and derivation.
   - **The fuel limit is not in the table.** rce-form §5 says it is never a field of a module, and the fuel sweep varies it. It becomes an argument of the reference, the Form evaluator and the core evaluator, carried in ω's initial value, which removes the process-wide `FUEL_LIMIT` static (`spec/checker/src/interp.rs:244`).
   - A limit a program reads (`sys.getrecursionlimit()`, CPython's 4,300-digit conversion limit, https://docs.python.org/3.13/library/stdtypes.html#int-max-str-digits) is read as a literal the lift row writes, equal to the table value or to the profile constant.
2. **Arena bounds as one ω entry.** The catalogue gains `bound.check[n]`: it consumes ω, takes the values a statement left live in the arena by read-return, and halts with "size limit" when their canonical JSON length passes n. The lift of RCE writes it after each statement of a `with arena r bound n` block, where 5.7 measures. Where the arena's values physically live is the record's business.
3. **Linear types.** A declared type may carry the flag `linear`: `["record",[T…],"linear"]`. FG6 then requires exactly one consumer for every source of that type, as for ω; a path that ends without consuming one is kept. This restates W-LIN-1 on the graph.
4. **Host value representation, `rce-cap/2`.** Each capability field and host-event payload field states `["bytes"]` or `["text","utf-8"]`; for an integer, its range and its behaviour past the range (`"trap"`, `"wrap"`, `"refuse"`), with no default; for a periodic event, its period as a rational number of seconds. A layout that crosses the boundary (a C struct passed to a foreign function) is stated here, as meaning.
5. **Language id fields.** A language id's profile names its data model; its ownership discipline where one text admits two readings; the mathematics library its transcendentals follow, by digest, or the engine that carries its own; the float fusion rule of its pinned compiler and architecture; its build mode where the mode changes meaning; and its profile constants for readable limits. Each is in the lift table's key.
6. **Catalogue fields.** Each catalogue entry gains `cost` (a C.6.4 class: a constant, `linear n + k` over a named operand length, or `nlogn`), `builds` (`none`, `fixed`, `grow`) and `arith` (`exact`, `trap`, `wrap`, `sat`, `trunc`, `none`). They count work for every language and decide obligations; they change no observation; they are in the catalogue key, so they are added before that key is first fixed.
7. **`cost.try_charge` settled.** The entry exists only because the reference evaluates an assignment's place twice, and no clause defines it (`spec/data/form-catalogue.json` on `claude/p0-catalogue`). The reference is changed to evaluate the place once, with a new catalogue key, before the catalogue key is first fixed; until then, a gate rule confines the entry to modules lifted from `rce`.
8. **Drop with effects.** A source destructor with an effect (a Rust `Drop` that prints, a C++ destructor) is lifted as a call of the lifted drop function on the ω chain at the source's drop point (for Rust, reverse declaration order at scope end, https://doc.rust-lang.org/reference/destructors.html). Its timing is then part of the meaning, and FG6's drop of the storage is a separate, effect-free fact.

Nothing else in the meaning changes. The five node kinds, the port modes, the derived wire kinds, DF1 to DF8, the frame and D stay as rce-form §6 to §9 define them. Body values stay immutable finite trees and body types hold no reference, so aliasing, lifetimes and storage cannot be expressed in the meaning at all: they are derived facts of the realization (§6, below).

## 5. The realization record, `rce-real/1`

**Where it lives and how it is keyed.**

- One record per (module, target). The target is `["source", L]` for the record a lift of language L writes, or `["target", PROFILE_KEY]` for an emit.
- It sits beside the module, as the concept record and the admission vectors do, under its own key, the SHA-256 of its canonical JSON. **It never enters the module key.** Convergence, identity by content and "complies with the Form" are statements about module keys and are unchanged.
- It is content-addressed per group: each group's entry has its own key. An edit that changes one group re-derives only that group's entry and the frame entry. Stack indices are relative to the caller's base, as Goose's are, so no entry depends on its callers.
- The **realized bundle key** is the SHA-256 of `["real", module key, record key, facts key, profile key]`. An emit's evidence record names it, and a printed program's provenance line names the module key and the record key.
- It is never written into D, an observation or a printed program's semantics.
- It replaces the concept record's `hints` for every flowG annotation that is a how choice (SimdHint, MemorySpace other than memory-mapped input and output, LoopKind and unrolling, ExecutionContext, InlineHint). Those were hints nothing checked; they become source-record entries that the gate checks. `hints` keeps only printing choices.

**Schema.** All bytes are canonical JSON; every field is present and no unknown field is; numbers are canonical decimal text; lists are sorted by path; nothing defaults outside the derived record (§9, below).

```
Real      = {"format":"rce-real/1", "module":KEY, "target":TARGET, "source":KEY|null,
             "facts":KEY, "types":[[τ, LAYOUT]…], "frame":FrameReal,
             "groups":{KEY:GroupReal…}, "relation":[Rel…], "claims":[Claim…]}
TARGET    = ["source", L] | ["target", PROFILE_KEY]
GroupReal = {"group":KEY, "fns":[FnReal…]}                // one per function, in C9 order
FnReal    = {"sig":[[PLACE…],[PLACE…]],                    // a place per input and per output
             "memory":[[OUT, PLACE]…],                      // owned node outputs and owned region inputs
             "reads":[[PATH, "copy"|"view"]…],              // read entries with an owned result
             "updates":[[PATH, "inplace"|"fresh"]…],        // update-return ports
             "copies":[[PATH, "deep"|"cow"]…],              // the `copy` entry
             "checks":[[PATH, CHECK]…],                     // domain test nodes
             "growth":[[OUT, GROWTH]…],                     // list and bytes values
             "schedule":[[PATH, SCHED]…],                   // Iterates and tree calls
             "order":[[RPATH, [i…]]…],                      // a printed order of a region's nodes
             "calls":[[PATH, "native"|"explicit"|"inline"]…],
             "fabric":FABRIC,
             "rows":[[SITE, ROW_ID]…]}                      // source records only: the lift row behind each entry
PATH      = [i] | [i, r, PATH…]                             // node i of the body; or region r of node i, then a path in it
RPATH     = a PATH that ends at a region
OUT       = [PATH, p] | [-1, j]                             // output p of a node; input j of the body
PLACE     = ["reg"] | ["value"] | ["stack", k] | ["arena", a] | ["heap"] | ["rc"] | ["traced"]
          | ["state"] | ["static"] | ["device", CLASS] | ["view", ROOT]
ROOT      = ["in", j] | [PATH, p]
LAYOUT    = ["int", "BITS", "s"|"u"] | ["f64"] | ["f32"]
          | ["tup", [perm…], "aligned"|"packed"]
          | ["sum", "TAGBITS", "fixed"|"var"]
          | ["arr", "inline"|"indirect"] | ["list", "inline"|"indirect"] | ["bytes", "inline"|"indirect"]
          | ["soa", [i…]] | ["decl", "indirect"] | ["carrier", KEY]
CHECK     = ["test", MECH] | ["proved", CERT]
MECH      = "compare" | "flag" | "widen" | "lib"
CERT      = ["measure", PATH] | ["interval"]
GROWTH    = ["fixed", "N"] | ["geometric", "NUM", "DEN", "INITIAL"] | ["bump", k]
SCHED     = ["seq"] | ["unroll", "K"] | ["tile", ["N"…]] | ["vector", "W"] | ["fuse", PATH]
          | ["tree", "W"] | ["pipeline", "II"]
FABRIC    = ["cpu"] | ["simd", ISA] | ["gpu", CLASS] | ["fpga", PART] | ["mcu", PART] | ["host-placed"]
FrameReal = {"stacks":"N", "stores":[[STORE_ID, "N"]…], "tasks":[[r, WORKER]…],
             "par":WORKER, "queues":[[r, "N"]…], "crossings":[[CLOCK, CLOCK, SYNC]…]}
WORKER    = ["serial"] | ["threads", "N"] | ["fabric", FABRIC]
Claim     = [SCOPE, OBLIGATION, [PARAM…], ORIGIN]
SCOPE     = ["module"] | ["def", REF] | ["region", r]
ORIGIN    = ["source", SPAN_KEY] | ["file", PATH_TEXT]
Rel       = [FACET, GROUP_KEY, PATH|null, "honoured"|"mapped"|"changed", ROW_ID|REASON]
REASON    = "target-lacks" | "lower-price" | "obligation" | "committed" | "carried"
```

What each facet means:

- **`sig`, `memory`**: where each owned value lives and when its storage is released. `reg` is a plain value in registers; `value` an owned fixed-size value inline (registers or the native frame); `stack k` is Goose's model, data stack k relative to the caller's base, bump-allocated and released when the region that drops the value ends; `arena a` the frame's store a, released at its scope's end; `heap` the target's allocator, released at the drop FG6 derives; `rc` reference counting; `traced` a tracing collector; `state` the fields of S; `static` literals and memo cells; `device` an accelerator's memory, for fixed-size values only. `view` carries a value as a reference into its root's storage: a signature output `["view", ["in", j]]` is how Rust's `&T` and `&mut T` parameters and Goose's rooted references appear, as derived roots, never as annotations.
- **`reads`, `updates`, `copies`**: a read entry (`list.get`, `arr.get`, `tup.get`, `sum.get`) returns an owned element as a copy in the meaning; `view` carries it out by reference. An update-return port puts its result in the operand's own storage (`inplace`) or in fresh storage (`fresh`, counted as a copy). The `copy` entry is a deep copy or copy on write, which 6.3's O3 calls an implementation choice, not visible.
- **`types`**: the layout of each body type, keyed by its canonical spelling. Layout is per module, because types are structural.
- **`checks`**: each domain test is printed by a mechanism (compare operands first, an overflow flag, a wider computation, a library call) or omitted with a certificate from the closed list of certificates in §10 (below). The test node stays in the Form either way; a certificate changes only what is printed.
- **`growth`**: capacity and growth of each list or bytes value.
- **`schedule`, `order`**: the transform of each Iterate or tree call, and a printed order of a region's pure nodes.
- **`calls`**: a call on the native stack, on an explicit stack in a store (ONE-15), or inlined.
- **`fabric`**: where each function runs. `host-placed` leaves the choice to the host's run-time placement (HC-12), logged in D.
- **`frame`**: the number of data stacks, each bounded store's capacity in layout bytes, the worker of each task region and of `par` work, each mailbox's capacity, and each clock-domain crossing with its synchroniser kind (HDL targets).
- **`relation`**: in a target record, how each entry relates to the source record's entry at the same site.
- **`claims`**: obligations asserted for scopes, with their origin (a source construct such as Goose's `checkfree fn`, or a committed file).

There is no field for overflow behaviour, float mode, evaluation order, reduction order, iteration order, a build switch, a safety level or an environment variable. That absence is the first half of the guarantee that a record cannot change meaning; the meaning-closure rule of §10 (below) is the second.

## 6. Derived facts, `rce-real-facts/1`

The record holds choices. The facts the choices rest on are computed by the gate from the module, the record and the target profile, stored under their own key and bound to the record by `facts`. A fact cannot be written by hand: it is what the gate computes, or the record is kept. Each is a deterministic pass in canonical order, linear in the size of a function except where stated.

- **Storage chains and versions.** A chain starts at a node output that creates an owned value (a literal of an owned type, `list.new`, `tup.mk`, a `fresh` update, a `deep` copy, a call output placed in a store) or at an owned region input, and continues through every edge that passes the same storage on: a read-return output, an `inplace` update-return output (raising the version by one), a capture into a Branch arm, an Iterate's carried value, and a call output whose signature root returns an input. **Linearity gives the aliasing fact for free**: FG6 consumes an owned value at most once, so at any point of the evaluation order a storage has exactly one live version and no two live owned sources share one. Goose gives up aliasing information to keep aliasing legal; Rust asks the programmer to state it; the Form has it because its values are owned and immutable.
- **Live intervals.** The range of canonical indices from a source's producer to its last use in its region, extended to the whole of a Branch or Iterate that uses it inside a nested region. This is Goose's liveness by mention made exact: every use is an edge.
- **Roots.** For each function output: `["fresh"]`, `["in", j, "same"]` or `["in", j, "updated"]`, computed callees first and, in a recursive group, as a fixed point in rounds that ends because a root only moves from `same` to `updated` to `fresh`. A view's root is the chain it reads. This is 5.6's return rule restated on the graph: a printer can print a borrowed parameter where Rust would need a lifetime, and the borrowed output needs no annotation.
- **Lane, tree and fork facts.** A lane fact holds for a measure-shaped Iterate (rce-form §6) whose carried values are its index pair, values returned unchanged, and owned arrays or lists updated only by `arr.set` or `list.set` at index i with values computed from plain values and reads at index i (W-PAR-3 and W-SIMD-1 restated). A tree fact holds for a call of C.8's fixed reduction tree. A fork fact holds for the calls the lift of RCE writes for a `par` block's branches: disjoint chains (W-PAR-1) and no emit (W-PAR-2).
- **Layout table.** Size, alignment and field offsets in bytes of every laid-out type under the target's data model and the store library's stated per-allocation header and alignment.
- **Bounds.** By the closed rules of `rce-bounds/1` (§7, below): per function and per bounded store, the **allocated-total bound** (the sum, in layout bytes, of every value written into the store during one activation of its scope, because a bump stack, an arena or a fixed buffer does not reclaim a dead value before its scope ends), the peak-live bound, the length bound of each growable value, and the work, check and depth bounds.

## 7. Target profile, resource vector, static bounds and prices

**Target profile, `rce-target/1`.** A file `lang/<L>/target/<name>.toml`, read into canonical JSON; its key is part of the print table's key, so a flag change re-keys every emitted artefact.

```toml
format     = "rce-target/1"
id         = "c11-lp64-x86_64-gcc14"
print      = "c11-lp64-x86_64"          # the language id of the print table
isa        = "x86-64-v3"
data_model = "LP64"
build      = ["gcc", "-std=c11", "-O2", "-ffp-contract=off", "-fno-strict-aliasing", "{src}", "-o", "{out}"]
float      = { contract = "off", ftz = "off", excess = "off" }
libm       = "none"                      # transcendentals are RCE code (F9)
stack      = "8388608"                   # native stack bytes of the main thread
lanes      = { f64 = "4", i64 = "4", i32 = "8" }
exact      = ["int.*", "f64.add", "f64.sub", "f64.mul", "f64.div", "f64.sqrt"]
stores     = ["reg", "value", "stack", "arena", "heap", "state", "static", "view"]
fabric     = "cpu"
threads    = ["1", "2", "4"]             # worker counts the worker-independence law runs
clock      = null                        # an HDL target: its clocks and domains
facts      = ["frame-bytes:gcc-su"]      # static build facts this profile admits, with their parsers
```

- A profile is admitted only with vectors: the catalogue's edge vectors run on it, built with its `build` command, and every entry named in `exact` gives the reference's bytes (HC-13's exactness probe, by digest).
- Build flags come from a closed list per target family, `spec/data/meaning-flags.json`, that classifies each flag as resource or meaning-pinning and gives the meaning-safe value of the second kind (`-ffp-contract=off`, `-fexcess-precision=standard`, no `-ffast-math`; Zig `@setFloatMode(.Strict)` at every scope; Go `CGO_ENABLED=0` and `GOTOOLCHAIN=local`). No RCE command takes a flag; a profile is chosen by naming its file as a positional argument.
- `portable` is the profile with `stores = ["reg", "value", "heap", "state", "static"]`, `lanes = {}`, `exact = []` and `stack = null`, which every print table carries out. The faithful record targets it.

**Resource vector, `rce-res/1`.** For one observation, the Form evaluator with the record records, as integers: `fuel`; `charged` (cost units, meaning, because `with budget` reads them); `work` (the sum of each evaluated Apply's catalogue `cost` class at its operand sizes, plus the C.6.4 constant of each node kind); `nodes` by kind; `checks` (domain tests evaluated whose record entry is `test`); `builds` (evaluated entries of `builds` class `fixed` or `grow`); `copies` and `bytes_copied` (`copy` entries, `fresh` updates, `copy` reads of owned elements, relocations out of a store); `allocs` per place kind; `views`; `inplace`; `calls`; `depth_peak`; `cells_peak` (the greatest live cell count at a node boundary, liveness fixed by C5 and FG6, a read-return output counted once with its operand); `live_peak` per store, in layout bytes; `lanes` per schedule kind; `carriers` per carrier key; `library` (bytes of each floor library the module carries). It lives in the observation record as `res`, on the Form side only, and replaces the evidence record's `work` object for every case of every language. It is never compared with the oracle side and never enters a key or D. The record is read only to classify each evaluated node, so the vector is a function of module, record and case.

**Static bounds, `rce-bounds/1`.** Per definition and per store, upper bounds on `cells_peak`, `depth_peak`, `work`, `checks`, the allocated-total and peak-live bytes of each bounded store, and the length of each growable value, each a canonical decimal or `unbounded`, by a closed list of structural rules written into the gate:

- types give a maximum size: SIZE_LIMIT for an unrefined list, k for a refined length bound `len(x) <= k` of 5.1 in its decidable fragment;
- a region's bound is the maximum, over its node boundaries, of the statically live sources plus a callee's bound at a call;
- a Branch takes the maximum of its arms;
- an Iterate with a count bound n contributes n + 1 tests and n bodies, and one with a measure the span of the measure's types;
- a recursive group contributes its declared bound times its members' longest path, and `unbounded` for a layered-measure certificate.

No solver, search or heuristic is on the path; two gate runs give identical bounds. Bounds are loose where SIZE_LIMIT decides them, and a loose bound is still a true bound. They live in the gate report, recomputed on every run.

**Price record, `rce-price/1`.**

```
{"format":"rce-price/1", "target":PROFILE_KEY, "unit":"aJ"|"cycle", "basis":"est", "source":TEXT,
 "rows":[[ITEM, ["NUM","DEN"]]…]}         // a price as a fraction; no item is priced at zero
ITEM = ENTRY | NODE_KIND | ["copy-byte"] | ["alloc", PLACE] | ["test", MECH] | ["call", "native"|"explicit"]
     | ["carrier", KEY] | ["library-byte"]
```

The modelled cost of a case is its resource vector times the prices, labelled `est`, never added to a measured figure. Prices are fractions in attojoules or cycles, so a cheap operation is never rounded down to zero. A price record never enters a key, and it is never on the default emit path: emitted bytes depend on prices only through a priced record committed as a file (§9, below).

## 8. Obligations

`spec/data/obligations.json`, format `rce-obligations/1`, is a closed, versioned list generated from the book like the catalogue and named by its key. Each obligation has parameters, the scope kinds it takes, a **Form part** decided from the module and the facts, and a **realization part** decided from the record, the print rows used, the target profile and admitted static build facts. Version 1:

| Obligation | Form part | Realization part |
|---|---|---|
| `checks.none` | none (a realization obligation) | every domain test in scope is `proved` by a certificate, or the profile admits a static build fact listing no remaining target-inserted check |
| `arith.exact` | no entry of `arith` class `wrap`, `sat` or `trunc` in scope | no print row used in scope relies on target UB |
| `halts.none` | no Halt, no count-bounded Iterate and no call of effect class 1 or above in scope, `fuel.tick` excepted | none |
| `alloc.none` | no entry of `builds` class `grow`; every `fixed` construction has a bounded size | every place in scope is `reg`, `value`, `view`, `stack` or `static` |
| `alloc.bounded n` | the static peak-live bound of the scope is at most n bytes under the profile's layout table | the bounded stores' capacities cover it |
| `stack.bounded n` | the static depth bound is at most n | the admitted `frame-bytes` fact, summed along the deepest path, is at most the profile's `stack` |
| `work.bounded n` | the static work bound of one activation is at most n | for `fpga`, the table's cycle image of n is within the profile's cycle budget per step |
| `copy.none` | no `copy` entry in scope | no `copy` read, `fresh` update or relocation in scope |
| `heap.none` | none | no place in scope is `heap`, `rc` or `traced` |
| `fabric c` | the fabric class conditions of §10 (below) | the profile names class c, and for floats its exactness probe digest |

A race-freedom obligation is deliberately absent: it holds for every Form by W-SPAWN-1, so stating it would be an empty claim. Constant time, latency, energy, cache behaviour and throughput are absent because none is a function of the Form. **A kept claim keeps the claim, never the record or the module**: an emit made under a record holding a kept claim is kept with the reason `claim:<obligation>`, and the same module emitted without that claim is unaffected.

## 9. How a record is made, the commands and the outcomes

1. **The derived record.** `derive(module, profile)` is a total deterministic function of the module and the target profile only:
   1. *Layouts*: each integer type the narrowest native width covering its range; tuples in declaration order, `aligned`; sums `fixed` with the narrowest tag; fixed-size elements `inline`; recursive declared types `indirect`.
   2. *Places*, callees first, groups in key order, canonical order within a function: a fixed-size owned value whose interval ends in its call takes `value`; a value that the stack rules admit takes the lowest-numbered stack not in use at that point, as registers are assigned (Goose's recipe, Goose spec, lines 86 to 111); a value in a `with arena` block takes that scope's store; everything else `heap`. A bounded store whose allocated-total bound is `unbounded`, or above what the profile allows, is replaced by `heap`. A place the profile lacks is skipped.
   3. *Reads and updates*: `view` wherever the view rules admit it, `copy` otherwise; `inplace` everywhere; `deep` for every `copy`.
   4. *Checks*: `proved` wherever a certificate of the closed list holds by its closed rules, otherwise `test` with the first mechanism the print table lists for the entry's type.
   5. *Growth*: `fixed` at the length bound where one exists and is below the profile's limit, otherwise `geometric 2 1 4`.
   6. *Schedule*: `seq` on `portable`; elsewhere `vector` at the profile's lane width wherever the lane rule admits it.
   7. *Calls*: `native` where the call rule admits it, `explicit` otherwise, no `inline`; *workers*: `serial`; *fabric*: `cpu`, or `simd` where `vector` was chosen.

   The recipe always succeeds, because `heap`, `copy`, `test`, `seq`, `explicit` and `serial` are admissible everywhere. So every module has an applied record for every admitted target with no choice written by anyone. A record stores only the entries that differ from the derived record, so one set of decisions has one record key.
2. **Honouring the source.** `rce realize-source MODULE PROFILE SOURCE_RECORD` starts from the derived record and takes, site by site, the source record's entry wherever the target's print table can print it and the gate admits it; every other site keeps the derived entry, and `relation` states each outcome.
3. **Carrying over.** `rce realize-carry MODULE PROFILE OLD_RECORD` keeps each entry whose path still names a node of the same kind with an unchanged Merkle digest (C2) and that the new target admits; every other site falls back to the derived entry, with relation `carried` or its reason. An edit therefore re-derives only what it touched.
4. **A priced record.** `rce price MODULE PROFILE PRICE` prints a record that takes, at each site, the admissible candidate with the lowest modelled cost on the site's static work bound under the named price record; on equal cost it keeps the derived entry, then the first candidate in table order. Its input is fully defined (no case is needed); it reads no clock, measurement, model or environment variable; and what it establishes is **per-site minimality under the stated prices and static bounds**, nothing more: interactions between sites are not optimised, and no corpus-wide lowest cost is claimed. A priced record reaches an emit only by being committed as a file.
5. **A committed record.** A record written by a person or a tool is committed as a file and passed as a positional argument. It goes through the same gate, and its entries' relation reads `committed`. HC-14 ("a knob is admissible only when it cannot change meaning") admits it, because the schema cannot state a meaning change.

**Commands**, positional only, each with a fixed arity, with the toolchain's exit statuses (0 applied, 1 kept, 2 usage, 6 internal):

| Command | Writes |
|---|---|
| `rce realize MODULE PROFILE` | the derived record |
| `rce realize-source MODULE PROFILE SOURCE_RECORD` | the derived record with the source's choices honoured where admitted |
| `rce realize-carry MODULE PROFILE OLD_RECORD` | the old record's choices carried over |
| `rce price MODULE PROFILE PRICE` | the priced record |
| `rce real-check MODULE RECORD` | applied or kept, naming the first rule that fails, and each claim's outcome |
| `rce facts MODULE RECORD` | `rce-real-facts/1` |
| `rce bounds MODULE RECORD` | `rce-bounds/1` |
| `rce res MODULE RECORD VECTORS` | one `rce-res/1` per case |
| `rce emit MODULE RECORD` | the realized program for the record's profile |

**Outcomes.**

- **Meaning compliance** is unchanged: a module complies with the Form when FG1 to FG15 and the meaning rules of §10.1 (below) apply it. No record is needed for it.
- **Realized compliance** is a second, separate outcome: a record is applied under target T, or kept with the first failing rule. An emit needs an applied record; `rce emit` exits 1 on a kept one and names the rule.
- **Claims** are decided one by one; a kept claim keeps only itself.

## 10. The gate rules, HW1 to HW30

The checker keeps at the first rule that fails and names the rule, the group key, the function index and the path. Every rule is decidable on the module, the record, the facts, the target profile, the print table and admitted static build facts; none runs the program, samples inputs or reads a measurement. HW1 to HW5 decide the module's meaning and join FG1 to FG15 (§17, below); HW6 to HW30 decide a record or a claim.

### 10.1 Meaning rules

| Rule | Check |
|---|---|
| HW1 Limits | the `limits` table is present, every field decodes, and fuel is not in it; its values equal the profile constants of the lift table's language id, each with a clause and derivation; every literal a lift row writes for a readable limit equals the table or profile value; the Form evaluator halts exactly at these limits with HALT or 6.2's trap texts |
| HW2 Arena bounds | every `with arena r bound n` block of an RCE source has `bound.check[n]` after each of its statements; `bound.check` takes exactly the values left live in the arena by read-return |
| HW3 Linear types | every source of a type flagged `linear` has exactly one consumer or is a region output; no path drops one |
| HW4 Host values | every `rce-cap/2` field and host-event payload field states representation, range, past-range behaviour and, for a periodic event, its period; payload types agree with them |
| HW5 Quirk entries | an entry marked `quirk` in the catalogue (today only `cost.try_charge`) occurs only in a module whose concept record names the language `rce`; the rule is withdrawn when the reference change of §4 item 7 lands |

### 10.2 Decoding, binding and separation

| Rule | Check |
|---|---|
| HW6 Decode | every field, facet value, place, layout, mechanism, certificate and target decodes; no unknown field; every entry differs from the derived record's entry at its site |
| HW7 Keys | `module` names a module the gate applied; every group key in `groups` is a group of that module; each entry's key is the SHA-256 of its bytes; `source`, the profile key and the print table key resolve; the profile is admitted |
| HW8 Meaning closure | the record names no catalogue entry and changes no node, edge, literal, type, limit or frame table of the module; the module bytes are identical with and without the record; no record entry or claim outcome is written into D, the module or λ |
| HW9 Facts | the gate recomputes `rce-real-facts/1` from the module, the record and the profile, and its key equals `facts` |
| HW10 Sites | every path names a node of the kind its facet needs under C5 numbering (`checks` a domain test, `schedule` an Iterate or tree call, `reads` a read entry with an owned result, `updates` an update-return port, `copies` a `copy`, `calls` a call, `memory` an owned output or owned region input, `growth` a list or bytes value); at most one entry per facet per site; every site without an entry takes the derived record's entry, which the gate checks like any other |

### 10.3 Places and stores

| Rule | Check |
|---|---|
| HW11 Value places | `value` only for a type whose layout table fixes its size, at most the profile's per-value limit, with a live interval ending within its call unless returned by value |
| HW12 Stack places | in each data stack, within a function, live intervals are nested or disjoint; at most one growable value is live per stack and it is the most recently placed; a growable value grows only while no later-placed value on its stack is live; a callee's values lie above the caller's live values; a growable value inside a growable value never takes a `stack` place; the gate recomputes the stack assignment by the recipe of §9 item 1.2 and requires the recorded indices to equal it |
| HW13 Stacks and recursion | no function of a recursive group holds a `stack` value live across a call into its own group, Goose's own restriction (Goose spec, lines 2882 to 2907) |
| HW14 Arena places | every value placed in store a has a live interval inside the evaluation of the region a's scope names; a value that leaves that region is relocated, and the relocation is counted as a copy |
| HW15 Capacity | every bounded store (`stack`, `arena`, a `fixed` growth, a `bump` growth, `device`) has a capacity in the frame at least its allocated-total bound in layout bytes, including the store library's stated header and alignment per allocation; a store whose bound is `unbounded` takes `heap`; so exhausting a store RCE prints can never become an observation the meaning lacks |
| HW16 Other places | `heap`, `rc` and `traced` release storage only at or after the drop FG6 derives; a linear handle, and every value whose release the source made observable, takes neither `rc` nor `traced`, because its release is on the ω chain |
| HW17 Views | a view is created only by a `reads` entry or a signature place `["view", ["in", j]]`; every use of it precedes the next consumption or `inplace` update of its root's storage; it is never stored (it never enters a consume port that builds an owned value); it leaves a region only when its root is an input of that region whose root fact is `same`; a view carried across Iterate runs has a root both regions carry unchanged |
| HW18 Updates and copies | `inplace` is admissible at every update-return port, because FG6 makes the operand consumed by that port; `fresh` everywhere, counted; `cow` only where the print table has a copy-on-write carrier with its library vectors |

### 10.4 Layout, checks and target preconditions

| Rule | Check |
|---|---|
| HW19 Layout | each layout covers every field of its type exactly once; an integer width covers the type's range and its signedness the range's sign; a tag width covers the variant count; a `var` sum is never changed to another variant in place; a recursive declared type is `indirect`; offsets respect alignment and do not overlap; a print row that narrows a value (a width, a tag field, a packed payload) is used only where the Form's type lies inside the narrowed representation; a printed program for a target that states static assertions carries one per layout row it relies on |
| HW20 Certificates | `proved` is admissible only when its certificate holds: `["measure", PATH]` when the test compares an index with a length, the index is position i of the measure-shaped Iterate at PATH, the bound is `list.len`, `arr.len` or `bytes.len` of a value the Iterate carries unchanged, and the tested value is that value; `["interval"]` when interval arithmetic over the operand types' ranges and the tests that dominate the node (closed rules for `int.add`, `int.sub`, `int.mul`, `int.neg` and comparisons) shows the in-domain outcome; no other omission exists, and a `test` entry's mechanism has a print row for its type |
| HW21 Target preconditions | every print row the realized program uses that carries `requires` (a target operation with UB or unchecked IB outside a domain: C signed `+`, a Zig `+` in ReleaseFast, an unchecked index, C `restrict`) is used only at a node whose operands carry the matching domain fact (DF1 to DF8), a `proved` certificate for that domain, or, for `restrict` and `noalias`, a storage-chain fact that the two pointers' chains are disjoint and neither is a view of the other's root |
| HW22 Growth | a `fixed` capacity is at least the static length bound of its value; a `bump` growth names the stack its value is placed on, and that value is the stack's single growable value |

### 10.5 Schedule, calls, workers and fabric

| Rule | Check |
|---|---|
| HW23 Schedule | `unroll` and `tile` apply to an Iterate with a count bound or a measure and keep the ω chain's order; `vector`, `fuse` and `pipeline` need a lane fact; `tree W` needs a tree fact, splits at C.8's own points, and W is a power of two at most the profile's lanes; a vector or pipelined body whose lanes may halt is carried out by **block replay**: lanes run in blocks of W from a saved ω and, when any lane of a block halts, the block's results are discarded and the block runs again in index order from the saved ω, so the halt, its index and its fuel are the sequential ones; a reduction by any other means is admitted only for an entry exact under reassociation (wrapping integer entries, Boolean and bitwise entries) or for a checked integer entry whose interval certificate bounds every partial sum of every grouping (n times the largest operand magnitude lies in range); construct(schedule) gives the module key back; on an `hdl` module no schedule entry spans steps |
| HW24 Order | an `order` entry is a topological order of its region in which every node that consumes ω keeps its place in the ω chain |
| HW25 Calls | `native` is admissible for a call only when `limits.depth` is a number and the profile's `stack` is at least `limits.depth` times the largest frame reachable through the call, frame bytes taken from an admitted `frame-bytes` static build fact of the pinned build; otherwise the call is `explicit`, its frames in a heap or stack store; `inline` only for a function outside every recursive group |
| HW26 Workers | `tasks` maps only task regions and `par` only `par` work; every print row used for a worker states, with clause and derivation, that it keeps 6.8's tick semantics (tasks stepped in id order, deliveries after all moves, a task run only in its tick); reductions use the C.8 tree; no worker field and no row names an environment variable or the host's processor count |
| HW27 Fabric | a function placed off `cpu` uses only entries on the profile's `exact` list or entries a carrier implements exactly; class conditions hold: `simd`, lane-wise and tree entries and exact entries only, no `grow` entry, halts only through block replay; `gpu`, as `simd` plus no recursion, effect class at most 1 and `alloc.none`; `fpga`, every state field of bounded size and `work.bounded` for each step body; `mcu`, `alloc.bounded` and `stack.bounded` for the profile's memory and stack; a placement off `cpu` and `simd` needs a `FabricExec` on a real path for that class, so until one exists such a record is kept with the reason `fabric` |
| HW28 Build configuration | the profile's `build` command has clause and derivation, and every flag `meaning-flags.json` lists for the target family is set to its meaning-safe value |

### 10.6 Claims, relation and hardware timing

| Rule | Check |
|---|---|
| HW29 Claims and relation | every claim names an obligation of the list at its version, a resolvable scope closed over calls, and an origin; its Form part is decided on the module and facts, its realization part on the record, the rows used and admitted static build facts; a source construct the gate cannot decide gives a kept claim with its reason. In a target record, every site with a source entry has a `relation`: `honoured` (the same value), `mapped` (the print row that carries out the source's value by another mechanism) or `changed` (a reason from the closed list); every source-record entry names in `rows` the lift row that wrote it, which cites its clause |
| HW30 HDL timing | on an `fpga` profile under the `hdl` observation profile, `pipeline` and `unroll` stay inside one step; every crossing between two clock domains of the profile's `clock` record is named in `frame.crossings` with its synchroniser kind, and an unnamed crossing keeps the record |

**Obligation fixtures.** Every rule and every obligation has, at its version, positive fixtures (correct modules and records the rule applies) and negative fixtures (correct modules paired with a record or claim that asks for a choice the module's shape does not admit, such as a stack place for growable data kept across a recursive call), each with its expected outcome and kept reason, generated at the edges of the rule's pattern (a bound equal to n and to n + 1; exactly one `grow` entry). A negative fixture is a correct program that does not meet a requirement, as K5's kept fixtures are (rce-form §20); it is not a planted wrong answer. The gate's decision on every fixture equals the recorded outcome byte for byte.

## 11. Lift: capturing each source's how

For each language, meaning choices go into the Form through the lift table and the language id; realization choices go into the source record, each entry naming the row that wrote it; source claims go into the record's `claims` with origin `source`. A source record states only what the source text and its pinned build configuration fix. A fact the source's compiler decides on its own (Go's escape analysis, Rust's inlining) is recorded only from an admitted static build fact; without one, the facet takes the language's documented default. A lift table that writes no source rows for a facet is complete without them: the facet then takes that default, and only the realization lens (§13, below) reads the rows, so the cost of capturing a language's how is paid when that language goes for level 5, not before.

**Rust.**

- *Meaning.* The Cargo profile selects the language id: `rust2024-debug` (overflow panics, lifted to the trapping entries) or `rust2024-release` (overflow wraps, the wrapping entries). `/` and `%` of the minimum by −1 panic in both. The panic strategy is part of the id where `catch_unwind` can observe it. A `Drop` impl with an effect is lifted as a call on the ω chain at Rust's drop point. Layout is meaning only where observed (`size_of`, `transmute`, `repr(C)` across FFI). `unsafe` blocks are kept unless a row maps them to a named form.
- *Source record.* Moves, `&T` and `&mut T` become signature places `["view", ["in", j]]` with roots `same` and `updated` and the `inplace` update; `Box<T>` and `Vec<T>` give `heap`; `Vec`'s documented growth gives a `geometric` entry with the initial capacity a row datum; `Rc` and `Arc` give `rc` on the heap-threaded value; arena crates give `arena`; `repr(C)` and `repr(packed)` give `types` rows; iterator chains give `fuse`; `#[inline]` stays a printing hint.
- *Claims.* `#![no_std]` with no `alloc` crate gives `alloc.none` on the module; `#![forbid(unsafe_code)]` gives nothing, because the Form has no unsafe construct.

**Zig.**

- *Meaning.* The build mode is in the language id. In Debug and ReleaseSafe checked IB panics, lifted to the trapping entries (`zig0.15-safe`). In ReleaseFast and ReleaseSmall it is UB (https://ziglang.org/documentation/0.15.1/#Build-Mode); a trap refines UB, so `zig0.15-fast` gives the same module on every defined case and differs only in which cases admission classifies `ub`. `@setRuntimeSafety(false)` scopes are lifted under the fast rows. `+%` and `+|` map to the wrapping and saturating entries, `@addWithOverflow` to the checked entry with its bit. `@setFloatMode(.Optimized)` is kept with reason `semantics`. A `FixedBufferAllocator`'s exhaustion point depends on the pinned target's sizes, so it lifts as runtime library code that counts bytes by the profile's layout and returns `error.OutOfMemory` where Zig does; until those rows exist it is kept with reason `representation`. A general allocator's failure is the `resource` class. `@bitCast` of a `packed struct` lifts as bit arithmetic over Zig's normative packed layout. `comptime` results are literals from the pinned front end.
- *Source record.* Each allocator argument gives the place: `FixedBufferAllocator` a bounded store with its capacity and `fixed` growth; `ArenaAllocator` an `arena`; the general and page allocators `heap`. `extern struct` and `packed struct` give `types` rows; `@Vector(n, T)` gives `fabric simd` and `vector n`.
- *Claims.* A function with no allocator parameter gives `alloc.none`, exact on the source side because Zig makes allocation visible.

**C.**

- *Meaning.* The data model and libm are in the language id (`c11-lp64-x86_64`). Signed overflow, out-of-bounds access, use after free and violated `restrict` are UB (N1570, J.2), lifted as tests and traps with class `ub`; unsigned arithmetic wraps. `-fwrapv` selects another language id. The oracle pins `-std=c11 -ffp-contract=off`, because GCC's default for GNU dialects contracts across statements; a source built with contraction on is kept with reason `semantics`. `sizeof`, `offsetof` and casts through `char *` are constants of the data model. Pointers are (allocation index, offset) pairs into per-allocation byte arrays; shared-memory threads are kept with reason `shared-memory`.
- *Source record.* `malloc` and `free` give `heap`, each source `free` site recorded in `rows`; FG6's drop may come earlier, which a later emit states as `changed`, reason `lower-price`. `restrict` gives no entry: aliasing facts come from storage chains, never from a promise. Local arrays give `value`; `alloca` and variable-length arrays give `stack` where the stack rules hold. `struct` gives its layout in declaration order.
- *Claims.* C has no syntax for them; a committed claims file can still be decided, which tells a C user exactly whether a routine allocates.

**Go.**

- *Meaning.* Integers wrap. "An implementation may combine multiple floating-point operations into a single fused operation, possibly across statements", and an explicit conversion prevents it (https://go.dev/ref/spec#Arithmetic_operators), so the language id names the pinned architecture and compiler; a case whose observation depends on fusion is found at admission by a second architecture's run and kept with reason `choice` until a row carries Go's fusion rule exactly. Map iteration order is unspecified (https://go.dev/ref/spec#For_range), so a case that observes it is `nondet` and not admitted. Goroutines and channels that share nothing lift to task regions under the tick semantics; scheduler-dependent output is kept with reason `choice`; shared memory with locks is kept. `limits.depth` is `null`, so `boundaries/deep_recursion` of `lang/go1.24/corpus` (one million frames, admitted `deterministic`) holds on an evaluator that keeps its own stack, and an oracle stack exhaustion is the `resource` class. Finalizers and weak pointers are kept.
- *Source record.* Values Go's pinned compiler reports as escaping (`go build -gcflags=-m`, admitted as a static build fact) take `traced`, others `value`; without the fact, `traced`. `append` gives `growth` with Go's documented rule as a row datum. `GOGC` and `GOMAXPROCS` are oracle pins and appear in the source record's frame as the source's workers, never as meaning.

**Goose.** Goose's compiler has not been built or run for this proposal, and its emitted C needs a pinned C toolchain in its oracle record, so Goose starts at level 0 (a registry row). This states what its lift table would carry.

- *Meaning.* `goose-release` is the defining table, because Goose's specification calls debug mode a bug-finding tool rather than a promise; `goose-debug` is another id. Signed overflow wraps in release; unsigned arithmetic wraps; `%` is Euclidean (Goose spec, section 7). A push past `T[..k]`'s bound aborts, so k is a Form test and trap. The 48-bit size limit is normative (Goose spec, lines 4071 to 4101), so it is a profile constant. `mul_add` may fuse by target, so it is kept with reason `choice` unless the oracle pins one ISA. A relative reference overflows by Goose's normative packed layout, so it lifts as runtime library code over that layout, or is kept with reason `representation`. A `reusable` pool, whose freed slot "remains a valid, live value of its type", lifts under heap threading with Goose's free-list order as runtime library code, so a stale reference reads its slot's next owner exactly as Goose defines. Threads lift to task regions, and flat values through typed queues become channel messages. A data-stack overflow (a guard-page abort) is the `resource` class.
- *Source record.* Every non-fixed value gets `stack k` from Goose's recipe applied to the Goose program, and the lift compares it with the recipe of §9 applied to the module; where the two differ, the lift is kept with reason `representation`, because one of the two readings is wrong. A reference passed down a call becomes a signature view with its root. `T[>..]` and `T[>..<]` give `bump`; `T[..k]` gives `fixed k`; fixed and variable enum modes give `["sum", …, "fixed"|"var"]`; `simd fn` gives `fabric simd`.
- *Claims.* `checkfree fn`, which makes any surviving bounds check a compile error (Goose spec, lines 3398 to 3444), gives `checks.none` on that function, which RCE then decides with its own certificates; Goose's whole-program "no general heap" gives `heap.none` on the module.

Goose's model is then one memory-facet value, `stack`, beside Rust's `heap` with views, Zig's `arena`, Python's `rc` and Go's `traced`.

**Python.**

- *Meaning.* Integers are unbounded, so the lift uses wide-integer runtime definitions. `boundaries/recursion_caught` of `lang/python3.13/corpus` reads `sys.getrecursionlimit()` (1000), which the lift writes as a literal, and catches `RecursionError` from `limits.depth = [1000, ["error", x]]`. The 4,300-digit conversion limit is a profile constant of the runtime library. The language id names CPython's version, its run flags `-P -s -S` (because `-I` implies `-E`, which ignores `PYTHONHASHSEED`) and the libm digest the oracle links. Unpinned constructs are kept out by the oracle's `[[unpinned]]` patterns (`lang/README.md`).
- *Source record.* Every object takes `rc`; the cycle collector, finalizers and `id()` are not admitted. Lists give `geometric` growth with CPython's over-allocation rule as a row datum.

**JavaScript.**

- *Meaning.* `es2020` on a pinned V8. Every Number is binary64, so integer operations are float operations, which the lift writes as f64 entries; `BigInt` uses the wide-integer definitions; `|0` and the bitwise operators lift to their ToInt32 runtime definitions. ECMAScript leaves `Math.sin` and the other transcendentals implementation-approximated (https://tc39.es/ecma262/#sec-function-properties-of-the-math-object), so the language id names the engine by digest, as a libm is named. Strings are sequences of UTF-16 code units, lifted as such by runtime definitions. `Map` and `Set` iterate in insertion order, which the specification fixes, so they lift as runtime library code that keeps insertion order. `Math.random`, `Date.now` and timers are unpinned and kept out. Promises and the job queue lift to task regions under the tick semantics where their order is the specification's; `SharedArrayBuffer` and `Atomics` are kept with reason `shared-memory`.
- *Source record.* Every object takes `traced`; typed arrays give `types` rows and `fixed` growth at their length. Array growth is engine-defined, so it takes the documented default of no growth entry.

## 12. Emit: honouring, mapping or changing the how

Each print table gains a `real` section: rows that map facet values to target text, each with clause and derivation like every row (rce-form §19), and with a `requires` field naming any UB or unchecked-IB precondition of the printed text, which HW21 checks. Emit reads the module through the schedule and prints each record entry through those rows; the relation states, per site, what the emit did to the source's how.

| Facet | Rust | Zig | C | Go | Goose | Python, JavaScript |
|---|---|---|---|---|---|---|
| memory | `value` locals; `heap` as `Box` and `Vec`; `stack` and `arena` through a bump library written once in RCE | every place native, as an explicit allocator passed down (`FixedBufferAllocator`, `ArenaAllocator`, a page allocator); `stack` as a fixed-buffer bump | `malloc` and `free` at the FG6 drop; bump-stack and arena carriers of a few dozen lines | `traced` only, plus `value` where Go's escape analysis keeps it local; every other place `changed`, reason `target-lacks` | `value` and `stack` native; `heap` as a `reusable` pool; a growable inside a growable mapped to a pool of rows with offsets; growable data across recursion to a pool argument; otherwise kept with reason `obligation` | `traced` (JavaScript) or `rc` (Python) only |
| reads, views | `&` and `&mut` parameters from root facts, so the printed program passes the borrow checker by construction, checked at level 2 (target accepted) | slices and pointer parameters | `const T *`; a pointer pair printed `restrict` only where HW21's chain fact holds | slices | rooted references | copies; references are the runtime's own |
| checks | `checked_add` and a trap call (`flag`) or widened arithmetic (`widen`); `proved` prints no test, and an index after a passed test uses `get` with an unreachable `None` arm, never `get_unchecked` unless printed `unsafe` is admitted (§19, below) | `@addWithOverflow` (`flag`); wrapping printed as `+%`, never plain `+`; ReleaseFast admissible because HW21 requires a fact for every unchecked operation | `__builtin_add_overflow` (`flag`) or comparison first; native signed `+` only behind a fact | `math/bits.Add64` carry or comparison | explicit comparisons (release wraps); a function whose every test is `proved` prints as `checkfree fn` | the Form's tests as printed comparisons |
| types | `repr(C)` with `const` size assertions | `extern struct` or `packed struct` with `comptime` assertions | structs with `_Static_assert` on every size and offset | structs, no layout claim | Goose's packed layout | none |
| growth | `Vec::with_capacity` and doubling | allocator-specific | carrier | `make` with capacity, `append` | `T[>..]`, `T[..k]` | the runtime's own |
| schedule | loops; `vector` through `core::arch` under the profile's features | `@Vector` | loops, intrinsics under the profile | loops | loops, `simd fn` | loops |
| calls | native or an explicit stack (HW25) | native or explicit | native or explicit, `frame-bytes` from `-fstack-usage` | explicit stack only where `limits.depth` is a number; otherwise Go's growable stack | native, explicit across recursion | explicit stack where `limits.depth` exceeds the runtime's own limit |
| workers | `std::thread::scope` with C.8 chunks | `std.Thread` with C.8 chunks | threads through a floor library, or `serial` on bare metal | goroutines for `par` chunks with an ordered merge; tasks on the runtime library's tick scheduler | a thread function per task region, flat messages | `serial` |
| float rows | strict by construction | `.Strict` at every scope | `-ffp-contract=off` from the profile | every product through an explicit `float64(…)` conversion, which the Go specification says prevents fusion | Goose never fuses `*` then `+` | strict by the language |

Three effects of this table:

1. **Emitting back to the source language honours its how.** A Goose program lifted and emitted to Goose through `rce realize-source` keeps every `stack` index and growth entry (`honoured`); a Rust program keeps its `Box`, `Vec` and borrowed parameters.
2. **Emitting across languages states every change.** A Goose program emitted to Go turns every `stack` place into `traced`, each `changed` with reason `target-lacks`; a Go program emitted to Goose meets the stack rules at every site, maps the rest to pools, or is kept naming the first site that cannot be printed.
3. **Unchecked target modes become usable, safely.** Because HW21 requires a fact for every target operation with UB, a Zig program may be built in ReleaseFast and a C program may use native signed arithmetic exactly where RCE has shown no overflow, and nowhere else. The speed a build switch buys elsewhere for a whole program is bought here per site, with a proof.

## 13. Acceptance of the how, exactly

Acceptance stays by definition and exact comparison (rce-form §20): every admitted case is compared, no case is a planted wrong answer, no case is set aside, and every comparison is byte equality. The how adds laws, columns and fixtures; it adds no level.

**The laws**, named rather than numbered so that they collide with no existing label:

| Law | Statement |
|---|---|
| realization agreement | for every module M of a corpus, every record ρ applied for M on profile T, and every admitted case: emit(M, ρ), built by T's build command and run on every oracle of T, prints exactly the bytes of `rce trace` on M, fuel lines included, and its `host` entry obeys the host laws |
| realization erasure | the same holds for the faithful record, so a realization can always be removed without changing a byte; when the record under test equals the faithful one, one run covers both |
| worker independence | for every module with tasks or `par` work, the realized program run with `serial` workers and with `threads n` for each n of the profile's `threads` list prints the same bytes |
| resource agreement | the Form evaluator on M under ρ and the core evaluator on the schedule of M give the same `rce-res/1` field by field; for `fuel` and `charged` the reference interpreter agrees too |
| bound law | for every admitted case and every definition it enters, each observed activation's `cells_peak`, `depth_peak`, `work`, `checks` and per-store allocated and peak-live bytes are at most the static bounds; a breach is a defect of the bound rules, recorded with both numbers, and fails the run, so the bound rules are checked on every case with no held-out set |
| probe agreement | for every target with an admitted probe and every emitted case, the probe's reading equals the prediction from the resource vector and the table's resource images (a row whose image is a bound states `at-most`, and then the reading is at most the prediction); a difference means a print row does not do what its table says; a `probe-nondet` probe is not used, and its column reads `probe-nondet` |
| work determinism | `rce res` gives the same vector for the same (module, record, case) in two runs under the byte-determinism perturbations |
| realization lens | for a language whose printed text shows its realization (Rust, Zig, C, Goose): if emit_L(M, ρ) = q, then lift_L(q) applies M and writes a source record that `rce realize-source` turns back into ρ on every facet L's lift table reads; a target that hides its realization (Go, Python, JavaScript) reports the law as not applicable, never as holding |
| claim | every published realization statement (a copy count, a residual test count, a memory bound, "no allocation") equals the record, fact or obligation outcome it cites; every energy or time statement cites a receipt |

**Probes and static build facts.** The oracle record `rce-oracle/1` gains `[[probe]]` (a counter compiled only into the printed runtime's own entries of a separately instrumented build, written to a stream the profile does not compare, so no counter enters a compared program) and `[[fact]]` (a command of the pinned toolchain and a closed parser in the gate: `-fstack-usage` frame bytes, Go's escape and bounds-check reports, Goose's `checkfree` refusal, a synthesis report's achieved II). Each is admitted per target exactly as a case is: run twice under the pins and once per perturbation; one whose readings differ is `probe-nondet` and is not used. Printers therefore never check their own counts: the work vector is computed on the Form evaluator, the probes read an independent instrumented build, and the static facts come from the target's own tools. Compile-time assertions (C `_Static_assert`, Zig `comptime`, Rust `const`) let the target compiler check every layout row, and Goose's `checkfree` refusal is an independent second check of a `checks.none` claim.

**Where the laws enter the levels.**

- **Emit level 4** for a language and target additionally requires realization agreement and realization erasure for the derived record of every module of the Form corpus and for the source-honouring record of every module that has a source record, worker independence for every module with tasks or `par`, and the bound law. **Level 5** adds the realization lens where it applies.
- A **board** column records an HDL target's run on a real board as its own evidence; simulation agreement never implies board agreement.
- A **second toolchain** column records the same emitted program built by a second pinned toolchain, and a cross-architecture run of realized programs beside the existing `xarch-parity` workflow.
- A **how column** per language and direction lists the obligations its table can discharge, each backed by fixtures, and the probes and facts its oracle records admit, printed with the counts beside it.

**Cost of acceptance.** Compiled targets multiply builds. Erasure costs nothing extra where the derived record equals the faithful one (every `portable` emit); worker independence runs only for modules with tasks or `par`; probes need one instrumented build per probed target. Compiled targets run their realization gate on a schedule on a named machine, and the registry prints the date and machine with each count.

**What acceptance never does.** It never compares seconds, joules, cache behaviour or memory the operating system reports; it never sets a threshold on a ratio; it never averages runs; it never uses a probe that varies; it never accepts a realization because it is faster. A statement of lower cost must cite a comparison of resource vectors, and a statement of lower energy must cite paired receipts on one pinned host with both target profiles named. That last rule answers speed figures that compare two ISAs: Goose's own `matmul` figure of about 2x against C comes from `simd fn` selecting AVX2 while the C baseline was built for baseline x86-64 (the benchmark notes of the Goose repository at the commit cited).

## 14. Hardware specialisation and energy

**SIMD.** Lane-wise arithmetic on `[τ; n]` already has per-lane meaning (5.1, W-SIMD-1); `vector W` and `simd ISA` are resource choices for lane-wise and tree entries on the profile's exact list. Reductions are vectorised on the C.8 tree, whose shape matches a vector unit's complete binary tree at any power-of-two width, or for entries exact under reassociation (HW23). An unordered reduction (RISC-V V `vfredusum`, which "may evaluate any binary reduction tree") is never printed for floats. Halting lanes use block replay, so a float trap in lane 3 of 4 is reported exactly as the sequential loop reports it. The ISA is a profile field, so a comparison of realizations always names both ISAs.

**GPU.** A kernel is placed on a GPU only for entries on its probe's exact list, with the GPU class conditions of HW27. WGSL may reassociate, flush subnormals and split `fma`, and defines no `f64` (https://www.w3.org/TR/WGSL/#floating-point-evaluation), so WebGPU can carry integer kernels and, under a binary32 class, f32 kernels whose probe passes; binary64 runs there only through the software carrier, at its counted cost. A WebGPU view of an RCE program is therefore not a bit-exact float target, and a page that shows one is labelled simulated and non-normative.

**FPGA and HDL.** The hardware reading of XF-12 (`one-library-every-fabric.md`) becomes the realization of an `fpga` target: a step is a synchronous tick, fixed-size state takes `state` places as registers, bounded stores become BRAM with capacities proven by HW15, `unroll` and `pipeline` stay within a step under the `hdl` profile (HW30), and clock-domain crossings are named with their synchronisers. Under the `event` profile per-event outputs are compared and the II is a resource; under `hdl`, per-cycle outputs are the observation. A `FabricExec` on a real path is a prerequisite of every placement off `cpu` and `simd`; until one exists such records are kept with reason `fabric`. A design that passes its simulator can still fail on the board at a clock-domain crossing, so the board column is separate evidence.

**Energy.**

- Energy is a measure. It never enters the module, the record, a key, D or a printed program.
- The deterministic proxy is the resource vector; the modelled figure is the resource vector times the price record, labelled `est`; a joule figure is a receipt with basis and provenance (D.11), under the two-regime protocol of `energy-calibration.md` (count-linear on in-order cores, MCUs and FPGA lanes; time-linear on out-of-order CPUs and GPUs), which decides which prices a target can have. RAPL readings on a busy host are dominated by noise, so they are used only under that protocol.
- "Lowest cost and energy" has a deterministic rule here: `rce price` chooses, per site, the admissible candidate with the lowest modelled cost on its static work bound under stated prices. That is a statement about counted work and stated prices. "Lower energy than L" still needs paired receipts on one host. Until a meter reading exists, every RCE energy figure is "modelled".
- An energy budget declared in a source program goes to the concept record's `energy` field with a schema (`rce-energy-decl/1`: the declaration, its unit normalised, its span and source, the label `declared`), never to a record or a key.

## 15. Examples

**Example 1: one function, three realizations.** The RCE function

```
fn sum(xs: List[int[0, 1000]]) -> int[0, 1000000000] {
    let mut s = 0
    for i in 0..len(xs) { s = s + xs[i] }
    s
}
```

lifts to one group whose body holds a measure-shaped Iterate over `0..list.len(xs)`, a domain test before `list.get` (DF5) and a domain test before the trapping `int.add`. Its module key is the same for every target. Under the derived record for `c11-lp64-x86_64-gcc14`:

- `xs` takes `["view", ["in", 0]]`, root `same`: the function reads the list and returns nothing that holds it, so it is printed as `const int64_t *xs, size_t n` with no copy;
- the index test takes `["proved", ["measure", [3]]]` by HW20: the index is position i of the Iterate and the bound is `list.len` of the list the Iterate carries unchanged, so no index test is printed;
- the addition test takes `["test", "flag"]` and prints `__builtin_add_overflow`: an interval certificate fails, because the type of `xs` bounds each element but not the list's length, so the partial sum's range is not bounded by the types;
- the schedule is `vector 4` only if HW23's reduction rule admits it; it does not here, because the checked addition is not exact under reassociation and no interval certificate bounds every partial sum, so the schedule stays `seq`.

Under `portable` the same module prints every test and copies nothing more, because a read of a plain element is not an owned copy. With a refined length, `xs: List[int[0, 1000]]` with `len(xs) <= 1000` (5.1), the module changes, because a refined type is meaning. HW23's partial-sum condition then holds (1,000 times 1,000 lies inside `int[0, 1000000000]`), so the derived record takes `vector 4` on the C.8 tree, each lane's addition still printed with its test. Each record changed only what is printed, and each emitted program prints the same reference trace as its module.

**Example 2: a Goose stack place and the recursion rule.** A Goose function that builds a growable list `ys` with `T[>..]` and returns its length gives a source record entry `[[[2], 0], ["stack", 0]]` for `ys` and `[[[2], 0], ["bump", 0]]` for its growth. The gate recomputes the stack assignment (HW12), finds `ys` the only growable value on stack 0, finds the allocated-total bound of stack 0 from the length bound of `ys` (HW15), and applies the record. If the function were in a recursive group and `ys` live across the recursive call, HW13 would keep the record with the rule named; the derived record would then take `heap` for `ys`, emitting to Rust as `Vec` and to Goose as a pool argument, each `changed` with its reason, and the module, its key and its trace would be untouched.

**Example 3: Zig ReleaseFast, site by site.** A Zig source built in ReleaseSafe lifts under `zig0.15-safe`, with each `+` a trapping entry and its test. Emitting the module to Zig under a profile whose build is `-O ReleaseFast` uses the print row for plain `+`, whose `requires` names "no signed overflow". HW21 admits that row at each `+` whose test is `proved` by an interval certificate, and requires the `@addWithOverflow` row at every other `+`. The printed program is therefore as fast as ReleaseFast where the proof exists and as safe as ReleaseSafe everywhere, and realization agreement checks it byte for byte on every admitted case.

## 16. What the built work shows

The form layer is not built, but the parts that exist on the branches named under Sources contradict assumptions the earlier text made. Each row is a finding from public code and the change this proposal makes.

| Finding | Evidence | Change |
|---|---|---|
| Resource limits are RCE's, not the source's | `boundaries/deep_recursion` of `lang/go1.24/corpus` recurses 1,000,000 frames and is admitted `deterministic`; the reference traps at `EVAL_DEPTH` 20,000 (`interp.rs:251`), a figure 6.7 does not state; the core evaluator grows its native stack in segments | the `limits` table, `depth` per language id and `null` for Go (§4 item 1, HW1); 6.7 states the evaluation depth |
| A limit can be part of an observation | `boundaries/recursion_caught` prints `sys.getrecursionlimit()` and catches `RecursionError`; `boundaries/int_str_limit` meets the 4,300-digit limit; both admitted `deterministic` (`lang/python3.13/corpus/manifest.toml`) | readable limits are profile constants, and `limits.depth` carries a declared error that `catch` catches (§4, §11) |
| Oracle records pin more kinds of thing than rce-form §18 names | `[toolchain]` holds executable and shared-object digests, `[[unpinned]]` patterns exclude programs (`lang/README.md` and the Python and Go oracle records) | `[toolchain]`, `absent`, `extension_dirs` and `[[unpinned]]` become named fields of `rce-oracle/1`, beside `[[probe]]` and `[[fact]]` (§13); oracle pins that are how choices (`GOGC`, `GOMAXPROCS`) appear in a source record's frame (§11) |
| The catalogue carries no cost | the 106 entries of `spec/data/form-catalogue.json` on `claude/p0-catalogue` have no cost class; lifted Go and Python get fuel and no C.6 cost | `cost`, `builds` and `arith` per entry (§4 item 6); `rce-res/1` counts work for every language (§7) |
| A quirk of the reference became a catalogue entry | `cost.try_charge` exists because the reference evaluates an assignment's place twice, and no clause defines it | the reference changes before the catalogue key is fixed (§4 item 7); HW5 confines the entry until then |
| Run-wide state lives outside ω | `pub static FUEL_LIMIT` (`interp.rs:244`); the core's conformance test runs as its own binary because of it | the fuel limit is an evaluator argument carried in ω's initial value (§4 item 1) |
| Every program carries a large runtime | the runtime in the core is several thousand lines of core functions with bounded loops (`claude/p0-core`, 4c699c22) | floor libraries and carriers are content-addressed, with their bytes in `rce-res/1`'s `library` and prices per byte (§7) |
| C9 is built | the core orders recursive-group members canonically (`claude/p0-core`, 16051d63) | per-group record entries are ordered by C9 (§5) |
| Compiled targets could not run whole on a small machine | Python and JavaScript emit 1,459 conformance programs byte-identical (`claude/p0-emit`, f5ccd037); a whole Rust run is too long for a 4-processor machine and runs in a separate workflow (`conformance/README.md` there) | realization runs of compiled targets are scheduled on a named machine, with date and machine printed (§13) |
| Matching answers miss runtime defects | the existing emitters' runtimes have no depth limit, so a deeply recursive printed program crashes where the reference traps, although every matching conformance answer passes | HW25: a printed program meets `limits.depth` or uses an explicit stack |
| The host boundary failed on representation | PR 31 of this repository: an 8-bit tick count undeliverable after 255 ticks at 60 per second; non-UTF-8 input read as end of input (`claude/host-robustness-http-events`, b8585076); HTTP framing not reproducible from D (76a7e06f) | `rce-cap/2` representation, range, past-range behaviour and period fields (§4 item 4, HW4) |
| Placement has no caller | `place_sum`, `place_lanes`, `place_par` and `probe_fabric` have no callers outside `host.rs`, and the only `FabricExec` implementations are test doubles | placement off `cpu` and `simd` needs a real `FabricExec`; until then such records are kept with reason `fabric` (HW27) |
| Build configuration changes meaning | GCC contracts across statements without `-ffp-contract=off`; the Go oracle fixes `CGO_ENABLED=0` and `GOTOOLCHAIN=local` | build command and flags are target-profile data in the print table's key, from the closed list `meaning-flags.json` (§7, HW28) |
| Transcendentals follow each oracle's library | the CPython oracle pins `libm.so.6` by digest; Go's mathematics is Go code; V8 carries its own | the language id and the target profile name the library or engine by digest (§4 item 5, §7, §11) |
| `rce-form.md` has no cost and energy section | rce-form §20's evidence record holds a `work` object and nothing on prices, bounds or receipts | §7 and §14 of this proposal |

## 17. Changes to `rce-form.md` and the specification, if accepted

- **`rce-form.md` §1**: the terms of §1 above; "complies with the Form" stays a property of the module alone. **§2**: the dividing rule of §3 and a paragraph stating the point of §2, citing this proposal. **§3**: the `linear` flag. **§4**: the entry fields `cost`, `builds` and `arith`; the entry `bound.check[n]`; the `quirk` mark on `cost.try_charge` and its removal. **§5**: the fuel limit as an evaluator argument carried in ω's initial value; the `limits` table seeds ω. **§7**: the certificates `measure` and `interval` as the only ways a test may go unprinted, the test node staying in the Form. **§8**: exactly-once consumption for `linear` types in FG6, and the storage-chain aliasing consequence stated as a property. **§9 and §10**: `Module` gains `"limits":Limits`; `TypeDecl` gains the `linear` form; capability keys move to `rce-cap/2`; the concept record's `hints` loses every how hint and gains `energy`. **§11**: HW1 to HW5 added as FG16 Limits, FG17 Arena bounds, FG18 Linear types, FG19 Host values and FG20 Quirk entries; FG13 cites `rce-cap/2`. **§12**: the placement table sends SimdHint, MemorySpace, LoopKind, ExecutionContext and InlineHint to source-record facets. **§13**: construct ∘ schedule = identity also covers scheduled records. **§16**: the language-id fields and the source record. **§18**: the oracle record fields of §13 above. **§19**: the `real` section of each print table, its `requires` field, and a target profile inside the print table's key. **§20**: `rce-res/1` replaces `work`; the laws, columns and level additions of §13. **§21**: the nine commands of §9. **§24**: the questions of §19 below.
- **6.7**: states the evaluation-depth limit and its trap text. **5.7**: cited by `bound.check`.
- **A new chapter** of the book, beside the one `rce-form.md` asks for, defines `rce-real/1`, `rce-real-facts/1`, `rce-target/1`, `rce-res/1`, `rce-bounds/1`, `rce-price/1`, `rce-obligations/1`, `rce-cap/2` and `rce-energy-decl/1`, and HW6 to HW30.
- **`spec/proposals/LEDGER.md`**: one row, "The how in `rce-form/1` (`rce-form-how.md`)", decision open.

## 18. Phasing

The phasing keeps the meaning additions on the form's critical path, because they change the module key, and lands the record with the printer.

| Phase | Work | Exit check a machine computes | Size (estimate, lines) |
|---|---|---|---|
| Lock the form | §4's meaning additions; FG16 to FG20 with negative fixtures; `rce-res/1` and `rce-bounds/1` on the Form evaluator for RCE sources; resource agreement between the Form and core evaluators | the 56 covered examples and at least 1,435 conformance tests give byte-identical traces on the three evaluators with the new fields; resource agreement and the bound law hold on every one; one kept fixture per FG16 to FG20 | 2,500 to 4,000 Rust |
| Ingest flowG | flowG's how hints map to source-record entries; the schemas, decoders and canonical forms of `rce-real/1`, `rce-real-facts/1` and `rce-target/1` | every imported how hint has one facet or a kept reason; schema vectors pass | 1,500 to 2,500 Rust |
| Go, Python, JavaScript lifts | `limits.depth` per language; readable limits; source records with `traced` and `rc`; `rce-res/1` for every matching case | `deep_recursion` and `recursion_caught` hold; every matching case has a resource vector; the bound law holds on every case | 800 to 1,500 Rust, rows in each lift table |
| Printer and tables | the derived record, `rce realize`, `rce real-check`, facts, HW6 to HW30 with fixtures; `real` sections and target profiles for `rust2024`, `python3.13`, `es2020`; facets in this order: `checks` and `types`, then `memory` with `value`, `heap` and views, then `calls` (HW25), then workers | emit level 4 under the derived record: realization agreement, erasure, worker independence, bound law | 4,000 to 6,500 Rust, 300 to 600 lines of table data per target |
| Round trips | `rce realize-source`, `rce realize-carry`, the realization lens; probes and static build facts; `rce price` | the realization lens on Rust where it applies; probe agreement for every admitted probe; `probe-nondet` reported, never used | 1,500 to 2,500 Rust and Python |
| Breadth | C, Zig and, once a pinned build and oracle exist, Goose: `stack`, `arena` and bounded stores with HW12 to HW15 after the stack model proof; HW21 for ReleaseFast and native signed C; `checkfree` as a second check; obligations and claims | C emit level 4 with every UB case classified and every `requires` row discharged; Zig level 4 in ReleaseFast with HW21; Goose at level 1 or above | per language, plus the bump-stack and arena carriers once in RCE (500 to 1,000 lines of RCE) |
| Machines and hardware | `fabric` off `cpu` after a real `FabricExec`; integer WGSL; HDL with HW30 and the board column; energy receipts under the calibration protocol | integer WGSL emit level 4 behind the exactness probe; an HDL board run recorded beside simulation; the first measured energy receipts, every other figure "modelled" | per fabric |

The stack recipe is about 800 lines and needs one theorem, "every record the gate applies preserves the Form's meaning", proved on a model of the Form evaluator before `stack` places are applied outside fixtures. The two certificates together are about 400 to 800 lines.

## 19. What stays open

**Risks.**

1. **The schema may be too narrow.** A generational check, a moving collector or a new memory space needs a schema version. That is deliberate: a closed schema is what lets HW8 say no record can change meaning.
2. **The stack recipe may be wrong where Goose's is subtle.** Goose's own audit found a stale-reference hole in its checker (`docs/implementation.md`, section 11, at the commit cited). The Form removes root derivation from the problem (no references in bodies, explicit consumption), but the recipe still needs its model proof.
3. **Bounds are loose.** Through SIZE_LIMIT, allocated-total bounds are large, so many bounded stores fall back to `heap` and many `alloc.bounded` claims are kept until refined length types are common. A loose bound is still a true bound, and the fallback is always correct.
4. **Host exhaustion is outside D.** A `heap` store or the host can run out of memory; the host then stops without a D entry, as at a power failure, and replay is unaffected. An emitted program can therefore fail on a small machine where the Form applies the event.
5. **Leaks of layout into meaning are expensive to lift.** Zig's fixed-buffer allocator, Goose's relative offsets and C's `sizeof` need runtime library code over a normative layout; until it exists they are kept with reason `representation`.
6. **Acceptance cost grows with compiled targets**, bounded by the measures of §13.
7. **Targets that hide the how.** For Go, Python and JavaScript most choices are invisible in printed text; the lens does not apply, and only resource vectors and admitted facts speak about them.
8. **Static build facts depend on toolchain report formats.** Each parser is pinned with the toolchain, and an unparsable report makes the fact unavailable, so the call rule and the claims fall back to `explicit` and kept.
9. **The record could become a back door for flags.** HW8, the closed flag list and the absence of any meaning field guard against it; a committed or priced record is checked exactly as a derived one.
10. **The Goose evidence is all from reading.** No Goose behaviour here was observed, and no RCE statement cites Goose's speed; the Goose rows wait for a pinned build and an oracle record.

**Questions for the RCE project.**

1. **Home of the layer**: this separate proposal, cited from `rce-form.md` (recommended, with §4's meaning additions moved into `rce-form.md` because they change the module key), or one section of `rce-form.md`.
2. **Go's depth**: `null` (recommended), or a frame count derived from a stated frame shape.
3. **Committed and priced records**: admissible as committed files (recommended), or only the derived and source-honouring records.
4. **Printed `unsafe` in emitted Rust**: whether a `get_unchecked` row behind HW21 is admissible, or emitted Rust stays free of `unsafe` and `checks.none` stays kept on Rust.
5. **Host exhaustion**: a D entry `host.Exhausted` that replay reproduces, or outside D as now.
6. **Lane-wise and tree entries in the catalogue**: kept as derived facts (recommended, no re-key), or `lanes.*` and `tree.*` entries added before the catalogue key is fixed.
7. **A third certificate**: whether a dominating test on the same storage version joins `measure` and `interval`.
8. **Declared energy budgets**: whether a source's declared budget may become a `with budget` in cost units, which would be meaning.
9. **Where compiled targets' realization runs happen, and how often.**
10. **A deterministic collector in a language's runtime library**, for long-running lifted Go and Java programs, run at points that are a function of the Form, which would be meaning chosen per table.
