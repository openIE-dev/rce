# RCE

RCE is a small programming language defined by a written specification.
Every program is given one meaning, a *Form*: a state machine made of
states, events, a step function that maps a state and an event to a new
state and its outputs, and the log of events. This repository holds the
specification (draft 0.7), its grammar in EBNF, and `rce`, the reference
checker and interpreter that decides which programs are accepted and what
they do.

In plain words:

1. **A program gives the same answer on every computer.** Arithmetic, text
   and floating point each have one defined result, so nothing depends on
   the machine or the compiler.
2. **Every program finishes.** Loops have limits and recursion must work on
   smaller and smaller parts of its input, so a program cannot hang.
3. **Mistakes are caught before the program runs.** Using memory after
   giving it away, forgetting a case, or writing a test that does not hold
   makes the checker refuse the program, and it names the rule that was
   broken.

## Install

`rce` is built from source. You need [Rust](https://rustup.rs); the
version used is pinned in `spec/checker/rust-toolchain.toml` (1.98.1) and
`rustup` installs it on first build.

```sh
git clone https://github.com/openIE-dev/rce
cd rce/spec/checker
cargo build --release
install -m 755 target/release/rce ~/.local/bin/
```

The default build is pure Rust and needs no C compiler.

## Quickstart

Save this as `counter.rce`:

```
intent "a counter that never goes below zero"
    example Inc, Inc, Dec -> 1
    example Dec -> refused: Underflow

state Counter
    n: u64 = 0

event Inc
event Dec

error Underflow

step Counter on Inc -> n + 1
step Counter on Dec -> try n.checked_sub(1) else Underflow
```

Check it, run its tests, and apply some events:

```text
$ rce test counter.rce
ok counter.rce (2 examples, 0 checks)
1 passed, 0 failed

$ rce run counter.rce "Inc, Inc, Dec, Dec, Dec"
0 (initial) Counter { n = 0 }
1 Inc Counter { n = 1 }
2 Inc Counter { n = 2 }
3 Dec Counter { n = 1 }
4 Dec Counter { n = 0 }
5 Dec refused: Underflow
```

The two `example` lines are part of the program: if either one did not
hold, `rce check` would refuse the file. To learn the language step by
step, read [The RCE Book](https://github.com/openIE-dev/rce-book).

## Commands

| Command | What it does |
|---|---|
| `rce check FILE...` | parse, type check and admit; one verdict per file |
| `rce test FILE...` | the same, and count each file's examples and checks |
| `rce run FILE [EVENTS]` | apply events, or run `fn main()` |
| `rce trace --json FILE [EVENTS]` | reference trace of each example, or of EVENTS, for checking another implementation step by step |
| `rce emit TARGET FILE` | `form` (the Form as JSON), `rust`, `python` or `js` |
| `rce fmt [--check] FILE...` | rewrite files in the canonical layout |
| `rce lsp` | language server on standard input and output |
| `rce --version` | the checker version, the specification draft and the language versions it accepts |

Section 10 of the specification defines each command exactly.

## What is in this repository

| Path | Contents |
|---|---|
| `spec/book/` | the specification, one chapter per file; start at `00-overview.md` |
| `spec/RCE-SPEC.md` | the same specification as one file, generated from `spec/book/` |
| `spec/book/03-syntax.md` | the normative grammar, in EBNF, read directly by the tools |
| `spec/examples/` | programs that must be accepted, and in `reject/`, `ill_typed/` and `inadmissible/`, programs that must be refused |
| `spec/checker/` | the reference parser, type checker, interpreter, emitters, formatter and language server (Rust) |
| `spec/checker-wasm/` | the same checker compiled to WebAssembly for the browser |
| `spec/tree-sitter/` | a tree-sitter port of the grammar, used as an independent second parser |
| `spec/lean/` | the Lean 4 model of the core language and its proofs (section 8) |
| `spec/fuzz/` | results of the grammar-level program fuzzer |
| `spec/LANGUAGE-REVIEW.md` | what RCE learned from Rust, Go, Python, Mojo, Zig and Haskell; drafts 0.4 and 0.5 follow it |
| `spec/conc06/`, `spec/conc06b/` | test programs for the version 0.6 extensions of Appendix C |
| `spec/checker-wasm/pkg/` | the built WebAssembly package (wasm-pack) |
| `spec/wasm-test/` | Node test of the WebAssembly package against the examples |

## Status

The specification is **draft 0.7, published for review**. It can still
change. The reference interpreter defines what programs mean and is not
built for speed. There is no production compiler yet. Section 10.8 lists
what the tools do not do yet.

## Language versions 0.5, 0.6 and 0.7

Each module selects its language version with a first-line comment
(section 2.7). A module without it is version 0.5, the language of
sections 2 to 10. A module that starts with

```
# rce-version: 0.6
```

also gets Appendix C (`spec/book/14-draft06-concurrency.md`): `par`,
`for ... par`, `with budget`, workgroups, LVars, sessions, supervisors and
the cost table of C.6.4. Version 0.7 (`# rce-version: 0.7`, C.15) is 0.6
with one difference: a standard library method costs its size-scaled work
in addition to the call, so only where a budget traps can change. The
version of the module that contains a call expression decides that cost,
so modules of different versions can be joined with `use`. Appendix D
(`spec/book/15-handler-spec-profile.md`) defines handler specs and fits.
A version 0.5 module never sees Appendix C, so every valid 0.5 program is
accepted unchanged. The cargo feature `conc06` is a development aid only:
it reads undeclared modules as 0.6 for the draft's test corpora and
enables the 0.6 test suites.

```sh
cargo test --release --features conc06
```

## How the specification is checked

```sh
cd spec/checker
cargo run --release --bin rce-spec-check
```

This reads the grammar from `spec/book/03-syntax.md`, builds LALR(1)
tables and fails on any conflict, parses every example with two parsers
that must agree, runs the checker over all accepted and refused examples,
and checks that `RCE-SPEC.md` matches the chapters. With
`--features tree-sitter` (needs a C compiler and the
[tree-sitter CLI](https://tree-sitter.github.io/)) it also regenerates the
tree-sitter parser and uses it as a third parser. `cargo test --release`
runs the checker's own tests, including the emitter tests, which need
`rustc`, `python3` and `node` (a missing one is reported as skipped).

## Related repositories

- [rce-book](https://github.com/openIE-dev/rce-book): a guide to the
  language from first principles, with every example checked.
- [rce-examples](https://github.com/openIE-dev/rce-examples): longer
  programs grouped by area.
- [tree-sitter-rce](https://github.com/openIE-dev/tree-sitter-rce): the
  grammar packaged for editors.
- [rce-ide](https://github.com/openIE-dev/rce-ide): an editor for RCE built
  on this checker.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Everyone taking part follows the
[code of conduct](CODE_OF_CONDUCT.md).

## License

Copyright (c) 2026 Open Interface Engineering, Inc.

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. Unless you explicitly state otherwise, any contribution
you intentionally submit for inclusion in this work shall be dual licensed
as above, without any additional terms or conditions.
