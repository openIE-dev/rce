# Contributing to RCE

Thank you for helping. Questions, bug reports, corrections to the
specification and pull requests are all welcome.

## Ways to help

- **Ask a question or report a bug.** Open an issue. For a bug in the
  checker, include the smallest program that shows it, the command you ran,
  what `rce` printed, and what you expected.
- **Point out an unclear passage in the specification.** Quote the section
  number and say what you found confusing. Clarity fixes are as valuable as
  code.
- **Propose a language change.** Open an issue first and describe the
  problem before the solution. Section 9 of the specification
  (`spec/book/11-decisions.md`) shows how earlier decisions were written up.

## Working on the specification

The chapters in `spec/book/` are the source. `spec/RCE-SPEC.md`,
`spec/book/INDEX.md` and the hash line at the end of each chapter are
generated. After editing a chapter:

```sh
cd spec/checker
cargo run --release --bin rce-book -- build
cargo run --release --bin rce-spec-check
```

The grammar is the EBNF in `spec/book/03-syntax.md`. Changing it changes
the language for every tool, so a grammar change also needs:

- an accepted example in `spec/examples/` or a refused one in
  `spec/examples/reject/`;
- the matching change to `spec/tree-sitter/grammar.js`, regenerated with
  `tree-sitter generate --js-runtime native` in `spec/tree-sitter/`;
- a passing `cargo run --release --features tree-sitter --bin rce-spec-check`.

A new rule that refuses programs needs a program in
`spec/examples/ill_typed/` or `spec/examples/inadmissible/` that the rule
refuses.

## Working on the checker

```sh
cd spec/checker
cargo build --release
cargo test --release
cargo clippy --release --all-targets
```

Keep the default build pure Rust. Code that needs a C compiler goes behind
the `tree-sitter` feature.

## Pull requests

- Keep each pull request to one change, and explain why it is needed.
- Run the commands above before you open it.
- Write commit messages and documentation in plain language.

## License

By contributing, you agree that your contribution is licensed under
MIT OR Apache-2.0, as described in [COPYRIGHT](COPYRIGHT).
