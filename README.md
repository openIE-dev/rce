# rce-conformance

An implementation-independent conformance suite for RCE. Every test is a plain
`.rce` file with a metadata block. The harness talks to any implementation
through a line-oriented JSON adapter protocol and reports, per normative
clause of the RCE book, which tests exercise it and whether they pass.

The guiding rule: tests encode what the laws of computing require (IEEE 754,
two's complement, injectivity of the log digest, replay equals reopen), not
whatever the spec text or the reference happens to say. Where the two differ,
the test follows the law and the difference is filed in `FINDINGS.md` with a
derived fix. No test encodes a known bug.

## Layout

- `clauses.json`: every clause extracted from `crates/rce/spec/book/*.md`
  (rule ids such as `W-PAR-1` and `HC-15`, theorem rows, MUST/SHALL sentences
  as `CHnn-M-<hash>`), each with a stable id and a profile (`language`,
  `handler-spec`, `tooling`, `proposed`, `editorial`).
- `tests/<area>/`: the tests, one area per directory.
- `harness/`: `rcetest.py` (file format) and `run.py` (runner).
- `adapters/`: the reference interpreter, the Rust, Python, JS and Scala
  emitters, and the wasm playground.
- `tools/`: clause extraction, harvesting, generators, tagging, coverage,
  emission parity, tree-sitter cross-check, mutation testing.
- `findings/` and `FINDINGS.md`: every place an implementation or the spec
  disagrees with the law.
- `reports/`: coverage, per-adapter results, parity, mutation score.

See `INTERPRETING.md` for the file format and `ADAPTERS.md` for the protocol.

## Running

    export RCE_CONFORM=/path/to/rce-conform   # cargo build -p rce-spec-check --bin rce-conform
    python3 harness/run.py --adapter adapters/reference.json --jobs 8
    python3 tools/coverage.py
    python3 tools/parity.py

## Status

Generated from `reports/` on the reference at joulesperbit `rce/host` (2026-10-15):

- 1952 tests. Language profile: 207 of 207 normative clauses covered (100%) (E1, fuel exhaustion is uncatchable, included).
  Not yet covered: the handler-spec profile (chapter 15, 14 clauses, implemented
  in `crates/rce` rather than the reference checker), tooling (7), proposed (6),
  editorial (1).
- Reference: 1932 pass, 20 fail; every failure is a finding, listed in
  `findings/known-failures-reference.txt`.
- Emitters (Python, JS, Rust): 1321 pass, 611 unsupported (ops an emitted
  program cannot answer: check, trace, host, replicate, trap refusals), 14 fail,
  all shared with the reference. Scala and the wasm playground: see
  `reports/results-*.json`.
- Mutation: 200 of 712 schema sites in the checker and evaluator sampled
  (seed 1); 189 killed (94.5%). Survivors are in `reports/mutation.json`.

## Convergence notes

The suite converges on ideas already established elsewhere. These notes cite
only material read while building it.

- Test262 (`INTERPRETING.md` in tc39/test262): metadata lives in the test file,
  negative tests state the phase and the error type they expect, flags change
  how a file is run, and files named with `_FIXTURE` are support files rather
  than tests. Here `expect: reject` plus a diagnostic code plays the role of
  the negative phase and type, and `fixtures:` names support modules.
- The WebAssembly spec testsuite (README of WebAssembly/testsuite and
  `test/core/README`): assertions about traps, invalid modules and
  exhaustion are first-class alongside assertions about results. Here a run
  states `outcome: trap:<kind>`, `refused:<error>`, `done` or `deadlock`, and
  fuel exhaustion is a trap kind.
- ACATS (the ACATS overview page and the User's Guide, section 3.1): tests are
  organised by test objective tied to rules of the reference manual. Here
  every test lists the clause ids it exercises and the coverage report lists
  every clause with no test.

## License

Apache-2.0. See `LICENSE`.
