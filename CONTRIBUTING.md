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
  `spec/examples/fit/syntax/`;
- the matching change to `tree-sitter/grammar.js`, regenerated with
  `tree-sitter generate --js-runtime native` in `tree-sitter/`;
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

## How the language grows, and how it shrinks

RCE is defined by one text, and the rules below exist so that the text
stays small. They are derived from what kept other languages small (Go's
proposal process, which declined 40.9% of 1,095 proposals in one study (arXiv 2510.06984, 2025); Lua's
"mechanisms, not policies") and from what did not (a standards committee
with hundreds of papers a year and nothing ever removed). Each rule names
the artefact that enforces it.

1. **Admission is a mechanical check, not a vote.** A construct enters the
   normative grammar only with: an accepted and a refused example; a Lean
   statement of the theorem it relies on, linked to the checker by a
   generated rule table or by vectors; and 0 LALR(1) conflicts. A construct
   with any of these missing is a proposal in `spec/proposals/` or a
   reserved word refused with W-RES-1, never a production.
2. **Composition before addition.** Before a proposal adds a construct, it
   shows the same program written with existing constructs, or shows that
   none exists. A second spelling of something the Form already says is
   declined on that ground alone (principle (e) of Appendix F). The
   incompatibility cost of composing by library instead of by syntax is
   paid outside the gate and recorded in the proposal.
3. **The language changes as the project learns.** RCE is in development,
   so any construct can be added, changed, generalised, replaced or removed
   when a better form is found. The change and every program, example and
   library module of the repository that it touches move together in one
   commit, so nothing is left on an old form, and the ledger row says what
   changed and why.
4. **One decision locus per interface.** Every interface in Appendix E and
   every emitter target has one owner in `SURFACE.md`'s table of owners,
   and a change to the interface is reviewed by that owner, because an
   organisational boundary that runs through an interface is a predicted
   defect site (Nagappan, Murphy and Basili 2008).
5. **The gate's metrics are published.** `spec/proposals/LEDGER.md`
   records every proposal, its decision and its reason in a fixed set
   (feasibility, duplicates an existing spelling, limited use, breaks a
   theorem, breaks a principle), so the decline rate and its reasons are
   numbers anyone can read, as Go's are.
6. **The market does not prune.** Adoption follows installed code and
   libraries, not language quality (Meyerovich and Rabkin 2013), so a
   construct is never kept because people use it; it is kept because it
   meets rule 1.
7. **Model authorship changes the rate, not the rule.** Proposals and code
   written by models go through the same gate with the same evidence. A
   higher generation rate is a reason to make the gate faster, never
   looser.

## Pull requests

- Keep each pull request to one change, and explain why it is needed.
- Run the commands above before you open it.
- Write commit messages and documentation in plain language.

## License

By contributing, you agree that your contribution is licensed under
MIT OR Apache-2.0, as described in [COPYRIGHT](COPYRIGHT).
