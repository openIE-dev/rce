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

RCE is defined by one text. The accepted language only grows: the Form
gains a form when a program needs one, as an extension in the sense of C.15
that changes no accepted program, so no change needs a new release. The
rules below keep the core of that text small, the set of primitive
constructs everything else is defined by. They are derived from what kept
other languages small (Go's proposal process, which declined 40.9% of 1,095
proposals in one study (arXiv 2510.06984, 2025); Lua's "mechanisms, not
policies") and from what did not (a standards committee with hundreds of
papers a year, each adding to the core). Each rule names the artefact that
enforces it.

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
3. **The core contracts; the accepted language does not.** A change that
   would refuse a program the text accepts, or give it a different meaning
   (C.15: its outputs, states and outcomes, including where a budget
   traps), is declined, so the language needs no new release. What
   contracts is the set of primitive constructs.
   - **A second spelling becomes a derived form.** A construct is a second
     spelling (rule 2 read backwards) when, in every context where the text
     accepts it, a composition of constructs that stay primitive is
     accepted and gives the same value, store, outputs and sequence of C.6
     charges. Of two constructs each of which is a composition of the
     other, the one admitted later is the second spelling. The text then
     defines it by that composition, as 6.4 defines the arrow step by a
     block and F.6.5 defines `if let` by `match`, and records the change as
     a row of the C.15 table.
   - **The evidence is a proved theorem.** The redefinition needs a Lean
     theorem, proved (section 8) and linked to the checker by vectors, that
     the construct and its composition have the same meaning in every
     program, because unlike an admission it re-derives programs that
     already have a meaning. Until then the construct stays primitive.
     `rce equiv` and `rce change` are regression evidence to their explored
     bound, never this ground: they compare explored behaviour, which can
     agree where charges differ.
   - **Nothing a program depends on changes.** A derived form keeps its
     syntax, its refusals, its words and every definition hash of the
     programs that use it (F.4.3), so no program is rewritten and no
     manifest pin or log changes (F.5.3). Only its own semantic rule,
     theorem obligation and implementation cases leave the text.
   - **Only what no implementation has may leave.** The specification of a
     construct that no implementation has, which W-RES-1 refuses so that no
     accepted program uses it, returns to `spec/proposals/`, and its word
     stays reserved (CC-5).
   - **Use decides nothing.** A construct that no example uses still belongs
     to the shape, and rule 1 already requires an example of it.
   - **Missing evidence is an obligation, never a reason to refuse an
     accepted program.** A theorem is reported at the level its row records
     (section 8; C.13 for T16 to T31), a claim that rests on it carries that
     level, and a row that is stated and unproved carries no claim. An
     admitted construct whose theorem has no Lean statement yet stays in
     the text while the statement is owed. A change that would break an
     admitted construct's evidence is declined under rule 1.
   - **A counterexample refutes a theorem, never a program.** An
     implementation that departs from the text is corrected, which is an
     erratum (C.15) even when it had accepted a program the text refuses,
     as with W-REC-1. Otherwise the model or the statement is corrected or
     narrowed to what holds, its row records the failure (section 8), and
     every claim that rests on it is restated. Where the text gives the
     counterexample no single meaning, it is completed with the meaning the
     reference evaluator gives (10.3).
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
   libraries, not language quality (Meyerovich and Rabkin 2013), so the
   number of programs that use a construct decides nothing: a construct is
   never kept primitive because people use it, and never made derived or
   withdrawn because they do not. Rule 2, read backwards by rule 3, decides
   whether it is primitive, and a gap in its evidence is an obligation
   under rule 3, never a ground to withdraw it.
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
