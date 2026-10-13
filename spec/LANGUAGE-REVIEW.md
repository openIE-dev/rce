# What RCE learned from six languages

Review for RCE-SPEC draft 0.4, 2026-10-12.

**Abstract.** This review reads the current definitions of Rust, Go, Python,
Mojo, Zig and Haskell, each of which solved hard problems that RCE builds
on. For each language it lists the ideas RCE learned from it and the
trade-offs RCE chose differently for its own goals, gives the mechanism
behind each, and attaches a measured or published cost where one exists. It
then records, feature by feature, what RCE does and why, using the interfaces where cost is paid: the
ABI, the allocator, the scheduler, the foreign-function boundary, the
serialization boundary and the CPU/accelerator boundary.

In plain words:

1. Most expensive software failures come from a few known causes: memory
   that is used after it is freed, values that are "nothing" when code
   expects "something", work that pauses unpredictably, and costly
   translation every time data crosses from one part of a system to another.
2. Each of the six languages found a strong answer to some of these causes,
   and each made trade-offs that fit its own goals, users and history.
3. RCE is a small language with narrower goals (deterministic, replayable
   programs). It learns from each language's answers, chooses differently
   where its goals differ, and states the cost of each choice in bytes,
   time or defects.

Scope and method. Sources are the language references, release notes,
RFCs, PEPs and proposals of 2024 to 2026, plus published incident reports and
measurements. A number is quoted only with its source. Where a source's
method limits its precision, that is said next to the number. "RCE today"
means draft 0.3; "0.4" means the changes this review applies.

---

## Part 1. Shared costs, defined

These quantities recur below.

- **Memory-safety defect share.** About 70% of the CVEs Microsoft assigns each
  year are memory-safety issues (MSRC, "A proactive approach to more secure
  code", 2019). Around 70% of Chromium's high-severity security bugs are
  memory-safety bugs (Chromium Security, "Memory safety"). In Android,
  memory-safety vulnerabilities fell below 20% of all vulnerabilities in
  2025 as new code moved to memory-safe languages; Google estimates 0.2
  memory-safety vulnerabilities per million lines of Rust against about
  1,000 per million lines of its historical C and C++ (Google Security
  Blog, "Rust in Android: move fast and fix things", Nov 2025). The Rust
  figure rests on one near-miss in about 5 million lines, so its ratio is an
  order-of-magnitude estimate, not a precise one (LWN discussion, 2025).
- **Null.** Hoare called the null reference his "billion-dollar mistake"
  (QCon London, 2009). The mechanism: one value inhabits every reference
  type, so every dereference is a possible failure the type system cannot
  see.
- **Collector pause and scan cost.** A tracing collector must visit live
  pointers. Discord's Read States service, written in Go, showed latency
  spikes about every two minutes because the runtime forced a collection
  that scanned a large live cache, although little garbage existed; the
  Rust rewrite freed entries when evicted and the spikes disappeared
  ("Why Discord is switching from Go to Rust", 2020).
- **Energy by language.** Pereira et al. measured the benchmarks-game
  programs and report energy normalized to C = 1.00: Rust 1.03, Go 3.23,
  Haskell 3.10, Python 75.88 ("Ranking programming languages by energy
  efficiency", Science of Computer Programming 205, 2021). These are
  small kernels, so they measure the runtime model (interpretation, boxing,
  collection) more than whole applications.
- **Function colouring.** When asynchronous functions can be called only from
  asynchronous functions, every API is written twice or the asynchronous
  kind spreads through the call graph (Nystrom, "What color is your
  function?", 2015).

---

## Part 2. Per language

Each item: **mechanism**, then **cost or evidence**, then **RCE**. "What RCE
learned" lists ideas RCE takes from the language. "Where RCE chose
differently" lists trade-offs that suit the language's own goals but not
RCE's; they are not judgements of the language.

### 2.1 Rust (2024 edition, Rust 1.85 to current)

What RCE learned

- **Ownership with move semantics and shared-XOR-mutable borrowing.**
  Mechanism: each value has one owner; a mutable borrow excludes every other
  access for its duration; the compiler checks this statically, so no
  collector is needed. Evidence: the Android density figures above.
  RCE: has it (5.5, 5.6), with second-class borrows instead of lifetimes.
- **Sum types and exhaustive `match`.** Mechanism: a missing case is a
  compile error. RCE: has it (W-MATCH-1).
- **`Result` and `?` for errors.** Errors are values and propagation is one
  visible token. RCE: has it as `τ ! ε` and `try` (from Zig, with sets).
- **Monomorphized traits.** Zero dispatch cost at run time. RCE: has it (5.9).
- **Explicit trust boundary in the 2024 edition.** `extern` blocks must be
  written `unsafe extern`, and `unsafe_op_in_unsafe_fn` warns by default, so
  every unchecked operation is marked where it occurs (Rust Edition Guide,
  Rust 2024). RCE 0.4: adopts the principle; `extern` is the one marked
  boundary (5.11).
- **Defined unwinding across FFI.** RFC 2945 added the `"C-unwind"` ABI
  because a panic unwinding through a `"C"` frame was undefined behaviour.
  RCE: there is no unwinding (errors are values, traps refuse the event), so
  this failure class cannot arise at the boundary.

Where RCE chose differently

- **Lifetime annotations, and the programs a borrow checker cannot yet accept.** Mechanism:
  first-class references need lifetime parameters to be stored and returned;
  the current non-lexical checker still rejects some correct programs, the
  standard case being a function that conditionally returns a borrow and
  otherwise mutates the same value ("problem case #3" of RFC 2094, accepted
  only by the Polonius analysis, still in progress). Cost: programmer time
  and workarounds (clones, indices, `Rc<RefCell<…>>`), which add copies and
  run-time borrow counting. RCE: has no first-class references (5.6);
  borrows exist only for a call, a loop, a match or a short `let`, plus
  returns tied to a parameter. No lifetime syntax exists.
- **Build-profile-dependent integer meaning.** Mechanism: overflow panics in
  debug builds and wraps in release builds unless `overflow-checks` is set
  (RFC 560). The same text has two meanings. RCE: overflow traps in every
  build (6.3 N2); wrapping is a separate, named method (0.4).
- **Async colouring and runtime split.** `async fn` returns a future that
  only an executor can drive; libraries pick one executor; a future that
  holds a non-`Send` value across `.await` cannot move between threads.
  RCE: no `async`/`await`; tasks are plain functions run by the
  deterministic tick scheduler (6.8).
- **Safe leaks and reference-count cycles.** `mem::forget` is safe and
  `Rc` cycles are never freed. RCE: no reference counting (W-SAFE-1), no
  destructors, so neither exists.
- **Compile-time cost of unrestricted monomorphization and procedural
  macros.** RCE keeps monomorphization but has no macros (decision 28).

### 2.2 Go (1.22 to 1.26)

What RCE learned

- **A short, complete grammar in EBNF**, readable in one sitting. RCE: the
  spec follows its format.
- **Goroutines without colouring.** Mechanism: any function can run as a
  goroutine; blocking calls park the goroutine, not the thread; stacks start
  small and grow. RCE: task functions are ordinary functions; no colour.
- **Per-iteration loop variables (Go 1.22).** Mechanism: before 1.22 one
  variable was shared by all iterations, so a pointer or closure taken in the
  loop saw the last value. Evidence: Let's Encrypt's 2020 CAA bug took the
  address of the shared variable and rechecked one domain many times,
  leading to the revocation of 3,048,289 certificates (Mozilla bug 1619047;
  Go blog, "Fixing For Loops in Go 1.22"). RCE: loop bindings are fresh each
  iteration and cannot be captured by reference (borrows cannot escape).
- **Range over function iterators (1.23), generic type aliases (1.24), Swiss
  table maps (1.24), container-aware GOMAXPROCS (1.25), Green Tea collector
  default (1.26).** Green Tea cuts collector CPU time by about 10% typically
  and up to 40% on some workloads (Go blog, "The Green Tea Garbage
  Collector", 2025). RCE: no collector to tune; iterators are not adopted
  (see Part 4: finite `for` keeps T2 simple).

Where RCE chose differently

- **nil, and the nil interface.** Mechanism: an interface value is a (type,
  pointer) pair; an interface holding a nil pointer is not equal to nil, so
  `return p` of a nil `*MyErr` as `error` makes `err != nil` true (Go FAQ,
  "Why is my nil error value not equal to nil?"). Nil maps panic on write;
  nil pointers panic on dereference. RCE: no null and no interface values;
  absence is `τ?` and must be matched.
- **Data races that break memory safety.** Mechanism: interface values and
  slices are multiword; a racy write can be observed torn, giving a type
  confusion (Cox, "Off to the Races", 2010). Go is memory safe only for
  race-free programs. RCE: tasks share nothing but channels and actor
  handles (5.8, T6 `tasks_never_share`).
- **Random `select`.** The spec chooses uniformly among ready cases, so runs
  are not reproducible. RCE: deterministic choice (6.8); fairness is open.
- **Tracing collection.** Cost: pause and scan work proportional to live
  pointers (Discord, above) and heap headroom. RCE: no collector.
- **Explicit error checks at every call.** Go keeps `if err != nil` and in
  2025 decided not to add new syntax for it ("[ On | No ] syntactic support
  for error handling", Go blog, June 2025). RCE: `try` and `catch` from the
  start.

### 2.3 Python (3.13, 3.14)

What RCE learned

- **Layout by indentation and readable surface.** RCE: same layout
  algorithm as Python's tokenizer (2.6).
- **Free-threaded build (PEP 703, supported in 3.14 by PEP 779).** It
  shows the price of retrofitting parallelism onto a shared-object runtime:
  single-thread overhead about 1% (macOS aarch64) to 8% (x86-64 Linux) on
  pyperformance, and about 15 to 20% more memory (Python docs, "Python
  support for free threading"; PEP 779). RCE: adopts the lesson, not the
  mechanism; parallelism is designed in through share-nothing tasks, so no
  global lock or per-object locking exists to remove.
- **Type parameter syntax (PEP 695) and deferred annotations (PEP 649/749)**
  show demand for static types in Python code. RCE: types are static and
  checked; there is nothing optional about them.

Where RCE chose differently

- **Dynamic typing at run time.** Mechanism: every value is a heap object
  with a type pointer and reference count; a small int is 28 bytes
  (`sys.getsizeof(1)` on 64-bit CPython) against 8 for an `i64`, and a list
  of ints holds pointers to such objects. Cost: the 75.88× energy figure
  above; type errors found at run time. RCE: static types; ints are 8 bytes
  or fewer.
- **The GIL legacy and C-API coupling.** Extensions depend on reference
  counts and the GIL; free threading needed new per-object locking and
  extension rebuilds. RCE: the FFI passes scalars and borrowed buffers only
  (5.11), so foreign code never manipulates RCE object headers.
- **Mutable default arguments and late-binding closures**, which can give
  surprising results.
  RCE: defaults are comptime constants; closures cannot capture borrows.

### 2.4 Mojo (1.0, open-sourced August 2026)

What RCE learned

- **Argument conventions and origins.** Mechanism: parameters are `read`,
  `mut`, `owned` or `out`; the compiler tracks an "origin" for each
  reference (Mojo manual, "Lifetimes, origins, and references"). RCE: the
  same three conventions as `&τ`, `&mut τ` and by-value move; origins are
  unnecessary because borrows are second class.
- **SIMD as a core type.** `SIMD[dtype, width]` maps to vector registers;
  scalars are width 1; widths above twice the register width perform poorly
  (Mojo std docs, `SIMD`). Cost basis: one AVX-512 instruction operates on
  16 f32 lanes. RCE: not yet; proposed in Part 4 (needs a Form change).
- **Compile-time parameters** in `[...]`. RCE: has `comptime` parameters.
- **One language for CPU and accelerator**, with types remapped at the
  device boundary. RCE: not yet; proposed in Part 4.

Where RCE chose differently

- **Untracked and wildcard origins.** `alloc()` returns memory with an
  untracked origin; the programmer frees it, which gives systems code full
  control. RCE: not available (W-SAFE-1); only the `extern` boundary is
  trusted.
- **ASAP destruction timing.** Values are destroyed after their last use,
  not at scope end, so destructor side effects move when code is edited.
  RCE: drop has no observable effect (O4), so timing is not observable.
- **Meaning defined by the implementation.** Mojo is defined by its compiler,
  which lets it evolve quickly. RCE: the meaning is fixed by this document
  and its theorems, independent of any implementation.

### 2.5 Zig (0.14, 0.15)

What RCE learned

- **`comptime`** as ordinary code run at compile time. RCE: has it (5.10),
  with fuel and no I/O.
- **Error unions and error sets.** RCE: has them (`τ ! ε`, set subtyping).
- **Allocators as explicit parameters; no hidden allocation.** Cost basis:
  allocation policy is visible at the call site, so arenas and fixed
  buffers are a choice of argument. RCE: arenas are explicit and lexical;
  0.4 adds a capacity, `with arena r bound n` (5.7).
- **Wrapping `+%` and saturating `+|` operators** separate from checked `+`.
  RCE 0.4: adopted as named methods (6.3 N2), so the default stays checked.
- **Asynchrony moved out of the language.** Zig 0.15 removed the `async` and
  `await` keywords; concurrency is provided by an `Io` interface passed by
  the caller, like an allocator (0.15.1 release notes; ziglang/zig PR
  25592). The stated aim is code that runs unchanged on blocking and
  evented implementations. RCE: same position; concurrency is a property of
  task functions and the scheduler, not of a function colour.
- **Concrete stream interfaces with the buffer in the interface**
  ("Writergate", 0.15): hot paths are concrete, only buffer refills use
  dynamic dispatch. RCE: relevant to the standard library, not the language.

Where RCE chose differently

- **No temporal memory safety.** Mechanism: use after free, double free and
  dangling pointers are undefined behaviour; ReleaseFast also removes the
  spatial and overflow checks (Zig language reference, "Undefined
  Behavior", "Build Mode"). Cost: the defect class behind the 70% figures.
  RCE: ownership and second-class borrows (T6).
- **Build-mode-dependent meaning.** As with Rust's overflow, a program's
  result depends on the build mode. RCE: one meaning; implementations differ
  only in cost (1, Conformance).
- **`anytype` duck typing.** Flexible, but errors appear at instantiation,
  deep in a call chain. RCE: trait bounds are checked at the definition.
- **Evolving before 1.0.** Zig improves its interfaces between releases
  (0.15 replaced the reader and writer interfaces). RCE: the spec is the
  stable artifact.

### 2.6 Haskell (GHC 9.x)

What RCE learned

- **Purity by default, effects in types.** RCE: pure functions by default;
  effects are allowed only in steps, handlers, `main` and task functions
  (5.2), inferred from context rather than annotated.
- **Algebraic data types, type classes, exhaustiveness.** RCE: kinds,
  traits, W-MATCH-1.
- **Linear types (GHC 9.0, `LinearTypes`)** let a function promise to use an
  argument exactly once (Bernardy et al., "Linear Haskell", POPL 2018).
  Mechanism value: safe in-place update and resource protocols. RCE: stays
  affine (decision 41) because drop is unobservable; linear handles are a
  Part 4 proposal for foreign resources.
- **Structural recursion over algebraic data.** RCE 0.4: admitted, with a
  proved bound (6.7, `RCE/Struct.lean`).

Where RCE chose differently

- **Lazy evaluation by default.** Mechanism: unevaluated thunks accumulate;
  `foldl (+) 0` over a large list builds a chain of thunks before adding.
  Space leaks are a recognised, hard-to-diagnose defect class (Mitchell,
  "Leaking Space", CACM 56(11), 2013). Cost: memory proportional to the
  pending work, and evaluation order that is hard to predict. RCE: strict,
  left-to-right evaluation (6.2).
- **Partial functions in the prelude** (`head`, `fromJust`, incomplete
  patterns as warnings). RCE: refused by the checker or trapped with a
  defined outcome.
- **`String` as a linked list of `Char`.** Several machine words and a pointer
  chase per character. RCE: text is a value type with defined operations.
- **Stop-the-world collection.** GHC added a nonmoving collector for the
  old generation to bound pauses (Gamari and Dietz, "Alligator Collector",
  ISMM 2020). RCE: no collector.

---

## Part 3. Decision table

| Feature | Where RCE learned it | RCE choice | Why, at which interface |
|---|---|---|---|
| Memory safety | Rust (ownership) | Ownership plus second-class borrows (has) | Allocator: no collector, no reference counts. Defect class: the 70% figures |
| Reference ergonomics | Hylo/Swift (second-class), Mojo conventions | Second-class borrows, no lifetimes (has) | Accepts the RFC 2094 problem case #3 pattern without run-time checks |
| Absence | Rust/Haskell `Option` | `τ?`, no null (has) | No nil values, so no nil-interface comparisons |
| Errors | Zig error sets, Rust `?` | `τ ! ε`, `try`, `catch` (has) | Serialization boundary: refusals are logged values |
| Integer overflow | Zig's operator split | Checked always, `wrapping_*`/`saturating_*` named (0.4) | One meaning in every build |
| Evaluation order | Rust, Go, Zig (strict) | Strict, left to right (has) | Memory use follows the text; no pending thunks |
| Recursion | Haskell/ML structural | Structural self-recursion with a size bound (0.4) | Stack: depth ≤ value nodes, bounded by N9 |
| Concurrency | Go (no colour), Zig 0.15 (Io outside the language) | Share-nothing tasks, deterministic ticks (has) | Scheduler: no colour, no data races, replayable |
| Effects | Haskell (typed) | Inferred from context (has); no annotations (decision 40) | Steps stay pure functions of state and event |
| Generic dispatch | Rust/Zig monomorphization | Monomorphized traits (has) | ABI: no vtables |
| Compile-time code | Zig `comptime` | `comptime` with fuel, no I/O (has) | Build: deterministic |
| Allocation control | Zig explicit allocators | Lexical arenas with capacity (0.4) | Allocator: per-block budget, freed in one step |
| FFI | Rust 2024 `unsafe extern`, RFC 2945 | C ABI, scalars and borrowed buffers, logged results (0.4) | FFI and allocator: nothing allocated crosses; no unwinding |
| SIMD | Mojo `SIMD[T, w]`, Zig `@Vector` | Proposed (Part 4) | CPU: vector registers |
| Accelerators | Mojo | Proposed (Part 4) | CPU/accelerator boundary: explicit device regions |
| Narrow floats | Mojo, Zig, Rust | Proposed f32 (Part 4) | Memory bandwidth: 4 bytes vs 8 |
| Text | Rust/Go UTF-8 bytes with char iteration | Scalar-indexed (has), conflict (Part 4) | Memory: O(1) byte slicing vs O(n) scalar indexing |
| Grammar | Go EBNF | LALR(1), tree-sitter port (has) | Tooling: one parse per text (T10) |

---

## Part 4. Comparison with RCE-SPEC draft 0.3

### 4.1 Already in RCE

Ownership and shared-XOR-mutable borrowing; second-class borrows; no null;
sum types with exhaustive matching; error unions with `try`/`catch`;
checked arithmetic with defined traps; strict evaluation; pure functions by
default with context-inferred effects; no `async`/`await`; share-nothing
tasks over bounded channels; deterministic `select` and content-hash ids;
monomorphized traits; `comptime`; lexical arenas; no garbage collector, no
reference counting, no destructors; no undefined behaviour; layout by
indentation; a conflict-free LALR(1) grammar.

### 4.2 Missing, added in 0.4

| Change | Grammar | Checker | Examples | Lean |
|---|---|---|---|---|
| R1 structural recursion (6.7, W-REC-1) | none | `structural_violation`; mutual recursion still refused | `39_structural_recursion`; refused: `rec_not_structural`, `rec_same_arg`, existing `recursion` | new `RCE/Struct.lean`: `struct_total`, `calls_eq_size`, `struct_fuel_irrelevant` |
| R2 `extern "C" fn` (5.11, W-FFI-1) | `extern_decl`, keyword `extern` | type rules, call only from main/tasks, no spawn | `40_extern`; refused: `extern_in_pure`, `extern_owned_param`, `extern_spawn`, `extern_abi`, `extern_text_result` | not modelled; T1/T3 rely on logging foreign results as inputs; T6 now states the trust boundary |
| R3 arena capacity (5.7) | `arena_stmt` takes `bound?` | bound typed u64; evaluator does not count bytes yet | `41_arena_bound_wrapping` | none needed: one more trap, T8 unchanged |
| R4 `wrapping_*`, `saturating_*` (6.3 N2) | none | typed and evaluated | `41_arena_bound_wrapping` checks that `(250 as u8).wrapping_add(10)` is 4 and `saturating_add(10)` is 255 | not modelled |
| R5 `&Box[τ]` to `&τ` (5.1) | none | coercion | used by 39 | none |

Machine results after the change: `rce.lark` LALR(1), 0 conflicts, 519
rules; grammar 42/42 accepted and 12/12 refused; checker 42/42 accepted,
12/12 syntax rejects and 37/37 ill-typed or inadmissible refused;
tree-sitter against Lark on 91 examples plus 2,000 generated programs
(13,983 inputs) with 0 disagreements; `lake build` succeeds with no
`sorry` and the new theorems depend only on `propext` and `Quot.sound`.
The grammar fuzzer (`spec/fuzz/`) was not changed in this pass, so the
random programs do not yet exercise R1 to R3.

### 4.3 Conflicts and gaps that needed a decision (applied in draft 0.5)

Each of these changes the Form's value domain or the emitters, so each
needed a maintainer decision. All nine were approved and are applied in draft
0.5 (RCE-SPEC 9.4); the note after each item says how.

1. **f32 (and f16/bf16).** RCE has f64 only because the Form has one float.
   Cost: twice the bytes per element, so half the elements per cache line and
   per vector register, for workloads that do not need 53 bits. Proposal: an
   `f32` type whose operations are computed in f64 and rounded to f32. For
   `+ − × ÷ √` this double rounding is exact because 53 ≥ 2·24 + 2
   (Figueroa, "When is double rounding innocuous?", SIGNUM 1995), so every
   implementation gets the same bits and the Form can store f32 values as
   f64 without loss.
   *Applied:* 5.1, 6.3 N3, Form ops `f32add…f32div`, `tof32`. f16/bf16 not added.
2. **Lane-wise SIMD vectors.** Proposal: arithmetic on `[τ; n]` for numeric τ
   is lane-wise; any lane that traps refuses the event. The meaning is the
   scalar meaning per lane, so T1 and T5 extend lane by lane; the emitter
   chooses vector instructions.
   *Applied:* 5.1, W-SIMD-1, on `[τ; n]` arrays of equal length.
3. **Device regions.** Proposal: a region kind for accelerator memory, where
   values move between host and device only by explicit `copy`, making each
   transfer visible in the text.
   *Applied:* 5.13, `with device d [bound n]`, W-DEV-1 (fixed-size values,
   no effects).
4. **Text indexing.** N7 counts Unicode scalar values, so `s[i]` is O(n) on
   UTF-8 storage or needs 4 bytes per character. Rust and Go index bytes and
   iterate characters. Options: keep it, or make `s[i]` a byte index with
   scalar iteration.
   *Applied:* byte index with character iteration (6.3 N7).
5. **u64 range.** Arithmetic results in [2^63, 2^64) trap because the Form's
   int is i64. Rust, Go and Zig support the full range. Option: widen the
   Form's int to i128 or add an unsigned representation.
   *Applied:* unsigned Form ops `uadd…urem` (6.3 N2).
6. **`select` fairness.** Least-hash-first can starve a channel. Go picks at
   random, which breaks replay. Option: rotate the starting arm by tick
   number, which stays deterministic.
   *Applied:* rotation per `select` execution, bounded wait proved (6.8, T11).
7. **Linear handle types** for foreign resources (files, sockets, device
   buffers) if foreign functions are later allowed to return handles.
   *Applied:* `linear struct` (5.12, W-LIN-1, T12).
8. **The default loop bound of 1,000,000** traps long but correct loops;
   whether the default should be per-program comptime configuration is open.
   *Applied:* `const LOOP_LIMIT: u64 = n` (6.7, W-LOOP-1).
9. **Finite `for` only.** Go 1.23 iterators and Python generators are not
   adopted, because a pull iterator's length is not known when the loop
   starts and T2 would need a separate bound for it.
   *Applied:* push iterators (Go 1.23 form), yields counted against
   LOOP_LIMIT so T2 needs no separate bound (5.14, T15).

### 4.4 A spec defect found during this review

`rce.lark` lists `"lalr"` among member names and the tree-sitter scanner
reserves `lalr` as a keyword, but section 2.4 does not list it. Both
grammars agree with each other, so the comparison passes; the fix is to
remove `lalr` from both, which changes which identifiers are legal.
*Applied in draft 0.5:* `lalr` is removed from both grammars.

---

## References

- Microsoft Security Response Center, "A proactive approach to more secure code", 2019. msrc.microsoft.com/blog/2019/07/a-proactive-approach-to-more-secure-code/
- Chromium Security, "Memory safety". chromium.org/Home/chromium-security/memory-safety/
- J. Vander Stoep, "Rust in Android: move fast and fix things", Google Security Blog, Nov 2025; LWN discussion of the estimate, lwn.net/Articles/1046397/
- C. A. R. Hoare, "Null References: The Billion Dollar Mistake", QCon London, 2009.
- Discord, "Why Discord is switching from Go to Rust", 2020. discord.com/blog/why-discord-is-switching-from-go-to-rust
- R. Pereira et al., "Ranking programming languages by energy efficiency", Science of Computer Programming 205, 2021.
- B. Nystrom, "What color is your function?", 2015.
- The Rust Edition Guide, "Rust 2024". doc.rust-lang.org/edition-guide/rust-2024/
- Rust RFC 560 (integer overflow), RFC 2094 (non-lexical lifetimes), RFC 2945 ("C-unwind" ABI).
- The Go Programming Language Specification; Go FAQ, "Why is my nil error value not equal to nil?"; Go blog, "Fixing For Loops in Go 1.22"; Mozilla bug 1619047 (Let's Encrypt CAA rechecking); Go blog, "The Green Tea Garbage Collector", 2025; Go blog, "[ On | No ] syntactic support for error handling", June 2025; Go 1.23 to 1.26 release notes.
- R. Cox, "Off to the Races", research.swtch.com/gorace, 2010.
- PEP 703, PEP 779, PEP 695, PEP 649, PEP 749; Python documentation, "Python support for free threading".
- Modular, "Mojo is now open source", 2026; Mojo manual, "Lifetimes, origins, and references"; Mojo standard library, `SIMD`.
- Zig 0.15.1 release notes; ziglang/zig PR 24329 ("Writergate") and PR 25592 (`std.Io`); Zig language reference, "Undefined Behavior" and "Build Mode".
- J.-P. Bernardy, M. Boespflug, R. Newton, S. Peyton Jones, A. Spiwack, "Linear Haskell", POPL 2018.
- N. Mitchell, "Leaking Space", Communications of the ACM 56(11), 2013.
- B. Gamari, L. Dietz, "Alligator Collector: A Latency-Optimized Garbage Collector for Functional Programming Languages", ISMM 2020.
- S. A. Figueroa, "When is double rounding innocuous?", ACM SIGNUM Newsletter 30(3), 1995.
- D. Racordon et al., "Implementation Strategies for Mutable Value Semantics", JOT 2022.
