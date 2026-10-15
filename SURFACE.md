# The RCE native surface

**Status: informative.** The language is defined only by the
specification in `spec/` (`spec/RCE-SPEC.md`); where this document and
the spec differ, the spec is right. This file records the design intent
and lineage of the surface; the gaps it left are decided in spec section
9, and every RCE program in it is checked by the reference checker
(`spec/checker/tests/surface_status.rs`).

The surface is the text a person writes when authoring natively in RCE. It
is not a language beside the Form; it is a notation for the Form
P = (S, E, δ, D, λ) (FORM.md), so every construct below names the Form
element it denotes. flowG is the language project built on RCE; this
document is the shared design input, and flowG owns its own grammar.

Design goal: the best of Rust, Zig, Go, Swift and Python in one surface,
with no garbage collector, memory safety by construction, native code at
the speed of hand-written C or Rust, and the shortest path from human
intent to a complete program.

## 1. What each source language contributes

| From | Feature | Surface construct | Form element |
|---|---|---|---|
| Rust | ownership, borrowing, no GC [1][2] | every value has one owner; `&` shared and `&mut` exclusive borrows; drop at scope end | S is an owned value tree; δ consumes S and returns S', so aliasing of mutable state cannot be written |
| Rust | traits, algebraic data types, `match` exhaustiveness [1] | `kind`, `trait`, total `match` | event and state sums; δ is total over E |
| Zig | `comptime` [3] | `comptime` blocks and parameters | evaluated during lowering; results become constants in the Form IR (FORM.md part 2) |
| Zig | explicit allocators [3] | `with arena { ... }`; region-typed collections | regions of the Form; one region is freed per step, so allocation cost is visible and bounded |
| Zig | error unions [3] | `T ! E`, `try`, `catch` | an error is an output in λ and an event in D, never an unlogged throw |
| Go | goroutines, channels, `select` [4][5] | `spawn`, `chan`, `select` | each spawn is a region; each send is an event in D; `select` resolves by the name-free content-hash order (FORM.md law 2), so concurrency replays identically |
| Swift | value types, copy on write [6] | `struct` is a value; `class` does not exist | S is a value; sharing is a borrow; copy on write is a lowering choice |
| Swift | protocols with associated types, argument labels [6] | `trait` with associated types; labeled arguments | interface contracts in the fit file |
| Swift | actors [7] | `actor` | a Form region with its own mailbox (law 3, no silent drop) |
| Python | readability [8] | indentation blocks, few sigils, `for x in xs`, comprehensions | none; pure notation |
| (RCE) | intent capture | `intent` blocks: the statement of what is wanted, in prose plus typed examples | contracts and oracles of the Form IR; examples become checks at the gate |

## 2. A program

```
intent "a counter that never goes negative"
    example Inc, Inc, Dec      -> 1
    example Dec                -> refused: Underflow

state Counter
    n: u64 = 0

event Inc
event Dec

error Underflow

step Counter on Inc -> n + 1
step Counter on Dec -> try n.checked_sub(1) else Underflow
```

The program is the Form: one state, two events, a pure step, and an error
that is logged, not thrown. The `example` lines are checked at the gate
before the program is admitted.

## 3. Memory and speed

There is no garbage collector and no reference counting in the generated
code. The step is a pure function from an owned state and an event to an
owned state; the emitter (form/emit_rust.rs) produces Rust that the Rust
borrow checker re-checks, and rustc/LLVM produce native code. Per-step
regions (Zig's arena pattern) make allocation a bump and a reset.
Concurrency compiles to Go-style lightweight tasks on the host, while the
logged order of events keeps replay deterministic.

## 4. Status of the prototype

`src/surface/` (prototype) parses the program above and emits Form IR,
and from it the Rust crate. Benchmarks against hand-written C and Rust on
Benchmarks Game kernels [9] are specified in `bench/surface/PREREG.md`.

## References

[1] S. Klabnik, C. Nichols. The Rust Programming Language. doc.rust-lang.org/book.
[2] R. Jung, J.-H. Jourdan, R. Krebbers, D. Dreyer. RustBelt: securing the foundations of the Rust programming language. POPL 2018. doi:10.1145/3158154.
[3] Zig Language Reference, ziglang.org/documentation/master (comptime, allocators, error unions).
[4] The Go Programming Language Specification, go.dev/ref/spec.
[5] C. A. R. Hoare. Communicating Sequential Processes. CACM 21(8), 1978. doi:10.1145/359576.359585.
[6] The Swift Programming Language, docs.swift.org/swift-book.
[7] SE-0306 Actors, github.com/apple/swift-evolution/blob/main/proposals/0306-actors.md.
[8] G. van Rossum, B. Warsaw, N. Coghlan. PEP 8, peps.python.org/pep-0008.
[9] The Computer Language Benchmarks Game, benchmarksgame-team.pages.debian.net.
