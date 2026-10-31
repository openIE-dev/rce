# The RCE native surface

**Status: informative.** The language is defined only by the
specification in `spec/` (`spec/RCE-SPEC.md`); where this document and
the spec differ, the spec is right. This file records the design intent
and lineage of the surface; the gaps it left are decided in spec section
9, and every RCE program in it is checked by the reference checker
(`spec/checker/tests/surface_status.rs`).

The surface is the text a person writes when authoring natively in RCE. It
is not a language beside the Form; it is a notation for the Form
P = (S, E, δ, D, λ) of a program, defined in spec 6.1
(`spec/book/07-form-evaluation.md`), so every construct below names the
Form element it denotes. flowG is the parent of RCE: it was the first pass at
one shared form for all code, on the idea that the syntax of every
language converges on one computer language, and RCE is the newer pass at
that form. flowG reads a program as a graph of nodes joined by typed
wires; its kernel, the closed vocabulary of 22 node primitives in 6
categories and 11 wire kinds (`openIE-dev/flow-g`), is proposed as the
vocabulary of the graph view of the Form (`spec/proposals/rce-form.md`
§12, a proposal; the view is not built).

Design goal (a goal, not a measured result): the best of Rust, Zig, Go,
Swift and Python in one surface, with no garbage collector, memory safety
by construction, native code at the speed of hand-written C or Rust, and
the shortest path from human intent to a complete program.

## 1. What each source language contributes

| From | Feature | Surface construct | Form element |
|---|---|---|---|
| Rust | ownership, borrowing, no GC [1][2] | every value has one owner; `&` shared and `&mut` exclusive borrows; drop at scope end (spec 5.5, 5.6) | S is an owned value tree; δ consumes S and returns S', so aliasing of mutable state cannot be written |
| Rust | traits, algebraic data types, `match` exhaustiveness [1] | `kind`, `trait`, total `match` (spec 5.9) | event and state sums; δ is total over E |
| Zig | `comptime` [3] | `comptime` blocks and parameters (spec 5.10) | evaluated during compilation; results become constants of the Form |
| Zig | explicit allocators [3] | `with arena r [bound n]`; region-typed values `T @r` (spec 5.7) | regions; each step body runs in an implicit region freed when the step returns, and a bound makes the capacity of a region explicit |
| Zig | error unions [3] | `T ! E`, `try`, `catch` | an error is an output in λ and an event in D, never an unlogged throw |
| Go | goroutines, channels, `select` [4][5] | `spawn`, `chan`, `select` (spec 5.8, 6.8) | each spawn is a task with a name-free id; each tick is logged in D with one entry per communication action; `select` takes the first ready arm scanning cyclically from a per-statement counter (spec 6.8), so concurrency replays identically |
| Swift | value types, copy on write [6] | `struct` is a value; `class` does not exist (a text that uses it requires change) | S is a value; sharing is a borrow; copy on write is a lowering choice |
| Swift | protocols with associated types, argument labels [6] | `trait` with associated types; labeled arguments | interface contracts, checked statically (W-TRAIT-1, W-TRAIT-2) |
| Swift | actors [7] | `actor` | a region with its own mailbox; no message is dropped silently (spec 6.1, 6.8) |
| Python | readability [8] | indentation blocks, few sigils, `for x in xs`, comprehensions | none; pure notation |
| (RCE) | intent capture | `intent` blocks: the statement of what is wanted, in prose plus typed examples | examples become checks at the gate (W-INT-1) |

## 2. A program

```
intent "a counter that never goes negative"
    example Inc, Inc, Dec      -> 1
    example Dec                -> kept: Underflow

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

The language rules give no garbage collector and no reference counting:
the step is a pure function from an owned state and an event to an owned
state (spec 6.4). `rce emit rust` (spec 10.5; the emitters are in
`spec/checker/src/emit/`) prints a Rust program that rustc, with its own
borrow checker, compiles to native code; the task statements of spec 6.8
are emitted onto the tick runtime `spec/checker/src/emit/rt.rs`, where the
logged order of events keeps replay deterministic.

## 4. What is in this repository

The surface above is the language the reference checker implements: the
grammar is `spec/book/03-syntax.md`, and `spec/checker/` parses, checks,
runs and emits it (`rce check`, `rce run`, `rce emit`). There is no
separate surface prototype, and no benchmark of emitted code against
hand-written C or Rust (for example on the Benchmarks Game kernels [9]) is
in this repository; such a comparison is a plan, not a result.

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
