# Working on RCE as an agent

This file is for any automated agent (Claude, Copilot, Cursor and others)
that works in this repository. It adds to [CONTRIBUTING.md](CONTRIBUTING.md);
read that first. Where the two differ, CONTRIBUTING.md wins.

## How work is handed out

- Every piece of work is a GitHub issue labelled `agent-task`. The issue
  states the goal, the files it touches, and the checks that decide it is
  done.
- Take one issue at a time. Say so in a comment on the issue before you
  start, so two agents do not take the same one.
- Work on a branch named `claude/<short-name>` (or your tool's own prefix),
  made from the current `main`. Open one pull request per issue and link the
  issue in it.
- A pull request is done when every CI check passes and a reviewer has
  approved it. CI is the gate; a description of a passing run is not.

## Rules every change follows

- **Exact, not approximate.** RCE is logical: a program is accepted by its
  definition and by exact comparison, never by statistics, planted wrong
  answers or held-out cases. Expected bytes come from the reference, never
  from the program under test.
- **The Form evolves.** The Form is not frozen. It follows the physics of
  computation, the changing ways algorithms are implemented, and what
  people create under converging machine constraints. A change to it moves
  the specification, the tools and every program together, in one change.
  Content addressing keeps every earlier Form's keys and logs replayable,
  so evolving the Form never changes what an applied program meant.
- **Words for RCE.** A program text that is not in Form is never admitted:
  it "requires change", and the report names the rule and the line. A step
  of an admitted program whose event does not apply is "kept": the state
  is unchanged and D records why, so "kept" describes a state, never code.
  A program in Form "complies with the Form". Do not describe RCE with the
  words token, guess, fit, learn or train, or as AI inference.
- **No flags, macros or directives of RCE's own.** Commands take positional
  arguments only. A comment line never changes what a program means.
- **Nothing private.** This repository is public. Do not copy code, text,
  paths or names from private repositories into it.
- **State only what the code does.** A document, ledger row or pull request
  says what is built and checked today; a plan is called a plan or a
  proposal. Energy figures are "modelled" unless a meter reading backs them.
- **Writing.** Define a term before using it; number labels in the order a
  reader meets them; use plain words. Name the work ("the RCE project",
  "this proposal"), not a narrated author, and do not write a vague "we".
- **Generated files are regenerated, never edited by hand:**
  `spec/RCE-SPEC.md`, `spec/book/INDEX.md`, `spec/data/*.json`,
  `stdlib/lib/MANIFEST.json`. Use `scripts/regen-spec.sh` and the generators
  the files name.
- **History.** Merge `main` into your branch; never rebase, amend or
  force-push a branch someone else may have fetched.
- **Commits** say what changed and why, in plain language, and name no
  model.

## Checks to run before opening a pull request

```sh
cd spec/checker
cargo build --release
bash ../../scripts/ci-tests.sh fast
cargo clippy --release --all-targets
cargo run --release --bin rce-spec-check
cargo run --release --bin rce-book -- build --check
```

Run the slow test binaries your change touches with
`scripts/ci-tests.sh bin NAME`. The full Rust emitter conformance suite runs
on CI, not locally.

## Where things are

- `spec/book/`: the specification's source chapters.
- `spec/proposals/`: proposals and `LEDGER.md`, which records each one and
  its decision. `rce-form.md` is the one form that lifting and emitting
  share; `rce-form-how.md` (when merged) covers how a program is realised.
- `spec/checker/`: the reference checker `rce`.
- `conformance/`: the language-independent conformance suite.
- `lang/`: pinned oracles and corpora for lifting other languages.
- `stdlib/`: the standard library.
